"""
CNNBasic: a compact, from-scratch CNN baseline exposing the same interface
as MultiModelFramework (same constructor kwarg shape, same forward
signature), used as a lightweight point of comparison.
"""
import numpy as np
import torch
import torch.nn as nn
from typing import Any, Dict, Tuple

from pidu.model.networks.activations import ParameterActivations
from pidu.model.networks.common import select_dict_from_subset


class CNNBasic(nn.Module):
  """A basic per-system CNN with the same interface as MultiModelFramework."""

  def __init__(
    self,
    system_info: Dict[str, Dict[str, Any]],
    model_info: Dict[str, Dict[str, Any]],
    *args,
    **kwargs,
  ):
    super().__init__()
    self.dt = kwargs.get('dtype', torch.float32)
    self.dv = kwargs.get('device', torch.device('cpu'))
    self.verbose = kwargs.get('verbose', 0)

    self.systems_info = {k: v for k, v in system_info['systems'].items() if v is not None}
    self.model_info = model_info
    self.active_systems = list(self.systems_info.keys())
    self.active_models = model_info['active_models']
    self.n_layers = system_info['dataset_info']['n_layers']
    self.n_sys_params = system_info['dataset_info']['n_system_parameters']
    self.dataset_info = system_info['dataset_info']
    self.p_gen_params = system_info.get('p_gen', {})
    self.n_unfolds = 1  # Single unfold for basic model
    self.use_physics = False

    self.mask_sys = model_info.get('mask_x_sys', {})
    self.mask_osc = model_info.get('mask_x_osc', {})
    self.cnn_size = float(model_info.get('cnn_size', 1.0))
    if self.cnn_size <= 0:
      raise ValueError(f"cnn_size must be positive, got {self.cnn_size}")

    def scaled_width(base_width: int) -> int:
      return max(1, int(round(base_width * self.cnn_size)))

    conv_channels = (scaled_width(32), scaled_width(64), scaled_width(128))
    fc_widths = (scaled_width(512), scaled_width(256))
    cnn_out_size = conv_channels[-1] * 8

    self.param_activations = self._create_activations()

    # CNN feature extractor - separate network per system
    self.cnn_per_system = nn.ModuleDict()
    self.fc_per_system = nn.ModuleDict()

    for systemname in self.active_systems:
      self.cnn_per_system[systemname] = nn.Sequential(
        nn.Conv1d(2, conv_channels[0], kernel_size=7, stride=2, padding=3, dtype=self.dt, device=self.dv),  # Input: 2 channels (y1, y2)
        nn.ReLU(),
        nn.BatchNorm1d(conv_channels[0], dtype=self.dt, device=self.dv),
        nn.MaxPool1d(kernel_size=2),  # ~2048 -> 512

        nn.Conv1d(conv_channels[0], conv_channels[1], kernel_size=5, stride=2, padding=2, dtype=self.dt, device=self.dv),
        nn.ReLU(),
        nn.BatchNorm1d(conv_channels[1], dtype=self.dt, device=self.dv),
        nn.MaxPool1d(kernel_size=2),  # ~512 -> 128

        nn.Conv1d(conv_channels[1], conv_channels[2], kernel_size=3, stride=1, padding=1, dtype=self.dt, device=self.dv),
        nn.ReLU(),
        nn.BatchNorm1d(conv_channels[2], dtype=self.dt, device=self.dv),
        nn.AdaptiveAvgPool1d(8),  # Reduce to fixed size
      )

      self.fc_per_system[systemname] = nn.Sequential(
        nn.Linear(cnn_out_size, fc_widths[0], dtype=self.dt, device=self.dv),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(fc_widths[0], fc_widths[1], dtype=self.dt, device=self.dv),
        nn.ReLU(),
        nn.Dropout(0.3),
      )

    # -- Output heads --
    self.sys_output_heads = nn.ModuleDict()
    for systemname in self.active_systems:
      mask_shape = self.mask_sys.get(systemname)
      n_sys_out = mask_shape.shape[-1] if mask_shape is not None else self.n_sys_params
      self.sys_output_heads[systemname] = nn.Linear(fc_widths[1], self.n_layers * n_sys_out, dtype=self.dt, device=self.dv)

    self.osc_output_heads = nn.ModuleDict()
    for systemname in self.active_systems:
      mask_shape = self.mask_osc.get(systemname)
      if mask_shape is not None:
        osc_out_size = int(np.prod(mask_shape.shape))
        self.osc_output_heads[systemname] = nn.Linear(fc_widths[1], osc_out_size, dtype=self.dt, device=self.dv)

    self.architecture_dims = {
      'conv_channels': conv_channels,
      'fc_widths': fc_widths,
      'cnn_out_size': cnn_out_size,
    }
    self._print(f"CNNBasic size={self.cnn_size:.3f} -> conv={conv_channels}, fc={fc_widths}")
    self._print(f"CNNBasic initialized with {self._count_parameters()}k parameters")

  def _create_activations(self) -> ParameterActivations:
    """Build the shared system/oscillator activation functions for this CNN."""
    systems = self.systems_info
    wn = systems[self.active_systems[0]].get('wn') if self.active_systems else None
    region_width = systems[self.active_systems[0]].get('region_width', [256]) if self.active_systems else [256]
    if not isinstance(region_width, list):
      region_width = [region_width]

    return ParameterActivations(
      n_layers=self.n_layers,
      wn=wn,
      region_width=region_width,
      p_gen_params=self.p_gen_params,
      dt=self.dt,
      dv=self.dv,
      build_system=True,
      build_oscillator=True,
      factor=1,
    )

  def _print(self, string: str, indent: int = 0, level: int = 1):
    if self.verbose >= level:
      print(f"{' ' * indent}{string}")

  def _count_parameters(self) -> int:
    """Count total trainable parameters in thousands."""
    total = sum(p.numel() for p in self.parameters() if p.requires_grad)
    return total // 1000

  def _select_dict_from_subset(self, dict_: dict, model_name: str) -> Dict[str, Any]:
    return select_dict_from_subset(dict_, model_name)

  def _activate_to_p(
    self,
    z_sys: torch.Tensor | None = None,
    z_osc: torch.Tensor | None = None,
    systemname: str | None = None,
  ) -> Tuple[torch.Tensor | None, torch.Tensor | None]:
    """Apply activation functions to network outputs to get physical parameters."""
    p_sys, p_osc = None, None
    pa = self.param_activations

    if z_sys is not None:
      p_sys = pa.to_system_params(z_sys)

    if z_osc is not None and pa.act_w0:
      n_regions_per_scale = (
        self.systems_info[systemname]['n_regions_per_scale']
        if systemname is not None
        else self.system_info['n_regions_per_scale']
      )
      p_osc = torch.zeros_like(z_osc, dtype=self.dt, device=self.dv)

      for idx_layer in range(len(pa.act_w0)):
        for idx_scale in range(len(pa.act_w0[idx_layer])):
          n_regions = n_regions_per_scale[idx_scale] if idx_scale < len(n_regions_per_scale) else z_osc.shape[-2]
          p_osc[:, idx_layer, idx_scale, :n_regions, 0] = pa.sigmoid(
            pa.act_os(z_osc[:, idx_layer, idx_scale, :n_regions, 0])
          )  # has_obj
          p_osc[:, idx_layer, idx_scale, :n_regions, 1] = pa.act_w0[idx_layer][idx_scale](
            z_osc[:, idx_layer, idx_scale, :n_regions, 1]
          )  # w0
          p_osc[:, idx_layer, idx_scale, :n_regions, 2] = pa.act_wp[idx_layer][idx_scale](
            z_osc[:, idx_layer, idx_scale, :n_regions, 2]
          )  # wp
          p_osc[:, idx_layer, idx_scale, :n_regions, 3] = pa.act_g[idx_layer][idx_scale](
            z_osc[:, idx_layer, idx_scale, :n_regions, 3]
          )  # g

    return p_sys, p_osc

  def summary(self):
    """Print model summary."""
    num_trainable_params = self._count_parameters()
    print(f"Model Summary for CNNBasic:")
    print(f" - Size factor: {self.cnn_size}")
    print(f" - Conv channels: {self.architecture_dims['conv_channels']}")
    print(f" - FC widths: {self.architecture_dims['fc_widths']}")
    print(f" - Total trainable parameters: {num_trainable_params}k")

  def get_info(self) -> Dict[str, Any]:
    """Return model information."""
    return {
      'name': 'CNN',
      'type': 'CNNBasic',
      'cnn_size': self.cnn_size,
      'architecture_dims': self.architecture_dims,
    }

  def forward(
    self,
    wn: Dict[str, torch.Tensor],
    y: Dict[str, torch.Tensor],
    x_sys: Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
    x_osc: Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
  ) -> Dict[str, Any]:
    """
    Forward pass with the same interface as MultiModelFramework.

    Args:
      wn: Dictionary of wavenumbers per system
      y: Dictionary of input signals per model
      x_sys: Tuple of (z_sys_init, p_sys_true)
      x_osc: Tuple of (z_osc_init, p_osc_true)

    Returns:
      Dictionary with keys: 'y', 'z_sys', 'z_osc', 'p_sys', 'p_osc'
    """
    dt, dv = self.dt, self.dv
    z_sys_init, p_sys_true = x_sys
    z_osc_init, p_osc_true = x_osc

    z_hat_sys, z_hat_osc = {}, {}
    p_hat_sys, p_hat_osc = {}, {}
    y_hat = {}
    to_pred = {}

    # Initialize storage for predictions (single unfold for basic model)
    for modelname in self.active_models:
      systemname = [k for k in self.active_systems if modelname.startswith(k)][0]
      to_pred[modelname] = {}

      mask_sys = self.mask_sys.get(systemname)
      if mask_sys is not None:
        z_hat_sys[modelname] = torch.where(
          mask_sys.bool(),
          z_sys_init[systemname],
          torch.zeros_like(z_sys_init[systemname], dtype=dt, device=dv),
        )
        p_hat_sys[systemname] = torch.zeros_like(p_sys_true[systemname], dtype=dt, device=dv)
        to_pred[modelname]['sys'] = True
      else:
        to_pred[modelname]['sys'] = False
        z_hat_sys[modelname] = [None] * self.n_unfolds
        p_hat_sys[systemname] = [None] * self.n_unfolds

      mask_osc = self.mask_osc.get(systemname)
      if mask_osc is not None:
        z_hat_osc[modelname] = torch.where(
          mask_osc.bool(),
          z_osc_init[systemname],
          torch.zeros_like(z_osc_init[systemname], dtype=dt, device=dv),
        )
        p_hat_osc[systemname] = torch.zeros_like(p_osc_true[systemname], dtype=dt, device=dv)
        to_pred[modelname]['osc'] = True
      else:
        to_pred[modelname]['osc'] = False
        z_hat_osc[modelname] = [None] * self.n_unfolds
        p_hat_osc[modelname] = [None] * self.n_unfolds

      y_hat[modelname] = torch.zeros_like(y[modelname], dtype=dt, device=dv)

    # Process each model
    for modelname in self.active_models:
      systemname = [k for k in self.active_systems if modelname.startswith(k)][0]
      y_input = y[modelname]  # Shape: [batch_size, 2, seq_len] or [batch_size, seq_len]

      if y_input.ndim == 2:
        y_input = y_input.unsqueeze(1)  # [batch, 1, seq_len]
        if y_input.shape[1] == 1:
          y_input = y_input.repeat(1, 2, 1)  # Replicate to 2 channels

      features = self.cnn_per_system[systemname](y_input)  # [batch, 128, 8]
      features_flat = features.flatten(1)  # [batch, 1024]
      fc_out = self.fc_per_system[systemname](features_flat)  # [batch, 256]

      if systemname in self.sys_output_heads and to_pred[modelname]['sys']:
        z_sys_pred = self.sys_output_heads[systemname](fc_out)
        z_sys_pred = z_sys_pred.reshape(z_sys_pred.shape[0], self.n_layers, -1)
        z_hat_sys[modelname] = z_sys_pred

        p_sys_pred, _ = self._activate_to_p(z_sys=z_sys_pred, z_osc=None, systemname=systemname)

        mask_sys = self.mask_sys.get(systemname)
        if mask_sys is not None and p_sys_pred is not None:
          p_hat_sys[systemname] = torch.where(mask_sys.bool(), p_sys_pred, p_sys_true[systemname])
        else:
          p_hat_sys[systemname] = p_sys_pred if p_sys_pred is not None else z_sys_pred

      if systemname in self.osc_output_heads and to_pred[modelname]['osc']:
        z_osc_pred = self.osc_output_heads[systemname](fc_out)

        osc_shape = z_osc_init[systemname].shape if systemname in z_osc_init else None
        if osc_shape is not None:
          z_osc_pred = z_osc_pred.reshape(z_osc_pred.shape[0], *osc_shape[1:])
        z_hat_osc[modelname] = z_osc_pred

        _, p_osc_pred = self._activate_to_p(z_sys=None, z_osc=z_osc_pred, systemname=systemname)

        mask_osc = self.mask_osc.get(systemname)
        if mask_osc is not None and p_osc_pred is not None:
          p_hat_osc[systemname] = torch.where(mask_osc.bool(), p_osc_pred, p_osc_true[systemname])
        else:
          p_hat_osc[systemname] = p_osc_pred if p_osc_pred is not None else z_osc_pred

      y_hat[modelname] = y_input[:, 0, :]  # Return first channel

    # Simulate the unfolds
    y_hat = {k: v[None, ...] for k, v in y_hat.items()}
    p_hat_sys = {k: v[None, ...] for k, v in p_hat_sys.items()}
    p_hat_osc = {k: v[None, ...] for k, v in p_hat_osc.items() if not v == [None]}

    return {
      'y': y_hat,
      'z_sys': z_hat_sys,
      'z_osc': z_hat_osc,
      'p_sys': p_hat_sys,
      'p_osc': p_hat_osc,
    }
