"""
Detection heads: turn a Backbone feature map into raw system/oscillator
parameter updates. Two flavors, matching CNNModel/SSDModel in
unfold_model.py:

  HeadCnnModel  - single-scale, system-parameters only.
  HeadSSDModel  - multi-scale (SSD-style), one HeadSSDScaleSmall per scale,
                  each producing both system and oscillator outputs via a
                  shared HeadSSDScaleSmallSysNetwork and a per-region
                  HeadSSDScaleSmallCell.

Class names HeadCnnModel/HeadSSDModel are relied on elsewhere via
`__class__.__name__` (see scripts/generate_architecture_tables.py) and must
not change.
"""
from typing import Any, Dict

import torch
import torch.nn as nn
from torch.nn.functional import pad

from pidu.model.networks.blocks import act_str_to_fn, ScalableTanhOutput


class HeadCnnModel(nn.Module):
  """Single-scale detection head: predicts system parameters only."""

  def __init__(self, shape_info: Dict[str, int], sys_info: Dict[str, int], *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.n_features_in = shape_info['in_features']
    self.n_sys_pars = shape_info['n_sys_pars']
    self.n_layers = n_layers = shape_info['n_layers']
    self.dt = sys_info['dtype']
    self.dv = sys_info['device']

    dropout = 0.1
    strides = [4, 2, 2]
    kernels = [7, 7, 5]

    n_neurons_fc1 = shape_info['x_len'] // 2 ** (sum(strides) // 2)
    n_neurons_fc1 += self.n_sys_pars * n_layers  # Append z_sys

    self.cnn = nn.ModuleDict({
      'cnn1': nn.Conv1d(
        in_channels=self.n_features_in, out_channels=self.n_features_in * 2,
        kernel_size=kernels[0], stride=strides[0], padding=3, bias=False,
        padding_mode='replicate', dtype=self.dt, device=self.dv,
      ),
      'cnn2': nn.Conv1d(
        in_channels=self.n_features_in * 2, out_channels=self.n_features_in * 4,
        kernel_size=kernels[1], stride=strides[1], padding=3, bias=False,
        padding_mode='replicate', dtype=self.dt, device=self.dv,
      ),
      'cnn3': nn.Conv1d(
        in_channels=self.n_features_in * 4, out_channels=self.n_features_in * 4,
        kernel_size=kernels[2], stride=strides[2], padding=2, bias=False,
        padding_mode='replicate', dtype=self.dt, device=self.dv,
      ),
    })

    self.fc = nn.ModuleDict({
      'fc1': nn.Linear(n_neurons_fc1, self.n_sys_pars * n_layers, dtype=self.dt, device=self.dv),
      'fc2': nn.Linear(self.n_sys_pars * n_layers, self.n_sys_pars * n_layers, dtype=self.dt, device=self.dv),
      'fc3': nn.Linear(self.n_sys_pars * n_layers, self.n_sys_pars * n_layers, dtype=self.dt, device=self.dv),
    })

    self.norm = nn.ModuleDict({
      'instance1': nn.InstanceNorm1d(self.n_features_in * 2, dtype=self.dt, device=self.dv),
      'instance2': nn.InstanceNorm1d(self.n_features_in * 4, dtype=self.dt, device=self.dv),
    })

    self.dropout = nn.Dropout(dropout)

    self.act = nn.ModuleDict({
      'act1': act_str_to_fn('tanh'),
      'act2': act_str_to_fn('tanh'),
      'act3': act_str_to_fn('tanh'),
      'fc_act1': ScalableTanhOutput(self.n_sys_pars * n_layers, self.dt, self.dv),
      'fc_act2': ScalableTanhOutput(self.n_sys_pars * n_layers, self.dt, self.dv),
    })

    self.flatten = nn.Conv1d(in_channels=self.n_features_in * 4, out_channels=1, kernel_size=1, stride=1, padding=0, dtype=self.dt, device=self.dv)

    self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

  def _init_weights(self, net: nn.Linear) -> nn.Linear:
    """Diagonal init for the weight matrix, with small noise added."""
    net.weight.data = torch.eye(net.weight.shape[0], net.weight.shape[1], dtype=self.dt, device=self.dv) + \
      torch.normal(0, 0.05, size=net.weight.shape, dtype=self.dt, device=self.dv)

    return net

  def forward(self, x_in: torch.Tensor, z_sys: torch.Tensor | None, z_osc: torch.Tensor | None, **kwargs) -> Dict[str, torch.Tensor]:
    """x_in: [batch_size, n_features_in, signal_length] -> {'sys': [batch_size, n_layers, n_sys_pars]}"""
    x_l1 = self.act['act1'](self.norm['instance1'](self.cnn['cnn1'](x_in)))  # [n_samples, n_features_in*2, signal_length//4]
    x_l2 = self.act['act2'](self.norm['instance2'](self.cnn['cnn2'](x_l1)))  # [n_samples, n_features_in*4, signal_length//8]
    x_l3 = self.act['act3'](self.cnn['cnn3'](x_l2))                          # [n_samples, n_features_in*4, signal_length//16]

    x_flat = self.flatten(x_l3).flatten(1)
    x_fc_in = torch.concat((x_flat, z_sys.flatten(1)), axis=1)
    x_fc1 = self.dropout(self.act['fc_act1'](self.fc['fc1'](x_fc_in)))
    x_fc2 = self.dropout(self.act['fc_act2'](self.fc['fc2'](x_fc1)))
    x_out = self.fc['fc3'](x_fc2)

    x_out = x_out[:, None, :]  # [n_samples, n_layers, n_sys_pars] (n_layers == 1)

    return {'sys': x_out}


class HeadSSDModel(nn.Module):
  """Multi-scale (SSD-style) detection head: one HeadSSDScaleSmall per anchor scale."""

  def __init__(self, shape_info: Dict[str, int], sys_info: Dict[str, int], *args, **kwargs):
    super().__init__(*args, **kwargs)
    len_x = shape_info['x_len']

    self.models = nn.ModuleDict()
    self.n_params_scales = {}
    self.strides = []
    self.kernel_sizes = []
    self.n_regions = []

    for i_scale, region_width in enumerate(shape_info['region_width']):
      factor = (shape_info['x_range'][1] - shape_info['x_range'][0]) / shape_info['x_range'][2]
      stride = int(region_width / factor)
      kernel_size = int(shape_info['anchor_width'][i_scale] / factor)

      shape_info['stride'] = stride
      shape_info['kernel_size'] = kernel_size
      shape_info['n_regions'] = len_x // stride

      self.models[f"scale_{i_scale}"] = HeadSSDScaleSmall(i_scale, shape_info, sys_info)
      self.n_params_scales[f'scale_{i_scale}'] = sum(p.numel() for p in self.models[f"scale_{i_scale}"].parameters() if p.requires_grad)
      self.strides.append(stride)
      self.kernel_sizes.append(kernel_size)
      self.n_regions.append(shape_info['n_regions'])

    self.n_params = sum(self.n_params_scales.values())

  def forward(self, x_in: torch.Tensor, z_sys: torch.Tensor, z_osc: torch.Tensor) -> Dict[str, torch.Tensor]:
    x_sys_out = torch.zeros_like(z_sys)  # [n_samples, n_layers, n_sys_pars]
    x_osc_out = torch.zeros_like(z_osc)  # [n_samples, n_layers, n_scales, n_regions, 4]

    for i_scale, model_scale in enumerate(self.models):
      model = self.models[model_scale]
      x_oscs, x_sys = model(x_in, z_sys, z_osc[:, :, i_scale, :model.n_regions])

      x_sys_out += x_sys
      x_osc_out[:, :, i_scale, :model.n_regions] = x_oscs

    return {'osc': x_osc_out, 'sys': x_sys_out}


class HeadSSDScaleSmall(nn.Module):
  """One anchor scale of HeadSSDModel: a shared system network plus a per-region cell network."""

  def __init__(self, i_scale: int, shape_info: Dict[str, int], sys_info: Dict[str, int], *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.i_scale = i_scale
    self.n_features_in = shape_info['in_features']
    self.x_len = shape_info['x_len']
    self.n_sys_pars = shape_info['n_sys_pars']
    self.n_layers = n_layers = shape_info['n_layers']
    self.n_regions = shape_info['n_regions']
    self.dt = sys_info['dtype']
    self.dv = sys_info['device']
    self.kernel_size = shape_info['kernel_size']
    self.stride = shape_info['stride']
    self.tanh = nn.Tanh()

    if shape_info['stride'] >= 64:
      self.norm_factor = shape_info['stride'] // 64
      self.norm_layer = nn.Conv1d(
        in_channels=self.n_features_in, out_channels=self.n_features_in,
        kernel_size=self.norm_factor * 2, stride=self.norm_factor, padding=self.norm_factor // 2,
        dtype=self.dt, device=self.dv,
      )
    elif shape_info['stride'] < 32:
      self.norm_factor = 32 // shape_info['stride']
      self.norm_layer = nn.Upsample(scale_factor=self.norm_factor, mode='linear', align_corners=True)

    self.downsample = nn.Upsample(size=64, mode='linear', align_corners=True)
    self.cell_network = HeadSSDScaleSmallCell(((self.n_features_in, 64, self.n_layers), self.n_layers * 4), 'stanh', dt=self.dt, dv=self.dv)
    self.sys_network = HeadSSDScaleSmallSysNetwork(((self.n_features_in, self.x_len, self.n_layers), self.n_layers * self.n_sys_pars), 'stanh', dt=self.dt, dv=self.dv)

  def forward(self, x_in: torch.Tensor, z_sys: torch.Tensor, z_osc_scale: torch.Tensor, **kwargs):
    x_in_scale_padded = pad(
      x_in,
      ((self.kernel_size - self.stride) // 2, (self.kernel_size - self.stride) // 2),
      mode='constant',
    )

    output_osc = torch.zeros_like(z_osc_scale, dtype=self.dt, device=self.dv)
    output_sys = self.sys_network(x_in, z_sys, data_info={'n_layers': self.n_layers})

    for i_region in range(self.n_regions):
      left_boundary = i_region * self.stride
      right_boundary = left_boundary + self.kernel_size
      x_scale_region = x_in_scale_padded[..., left_boundary:right_boundary]
      z_region = torch.concat([
        z_sys.flatten(1),
        z_osc_scale[..., i_region, :].flatten(1),
        self.tanh(torch.full((z_sys.size(0), 1), i_region, dtype=self.dt, device=self.dv)),
      ], axis=1)  # [n_samples, n_sys_pars + 4 + 1]

      x_scale_region_ds = self.downsample(x_scale_region)  # Downsample to fixed size: 64
      out_osc = self.cell_network(x_scale_region_ds, z_region, data_info={'n_layers': self.n_layers})

      output_osc[..., i_region, :] = out_osc

    return output_osc, output_sys


class HeadSSDScaleSmallCell(nn.Module):
  """Per-region oscillator head: small CNN over the region window + FC combining with z_region."""

  def __init__(self, input_sizes: tuple[tuple[int, int, int], int], act_fun_str: str, dt: torch.dtype, dv: torch.device):
    super().__init__()
    # input_sizes: ((n_channels_in, x_len, n_layers), output_size)
    self.dt = dt
    self.dv = dv

    self.cnn_layers = nn.ModuleList([
      nn.Conv1d(in_channels=input_sizes[0][0], out_channels=16, kernel_size=3, stride=2, padding=1, padding_mode='replicate', dtype=self.dt, device=self.dv),
      nn.Conv1d(in_channels=16, out_channels=32, kernel_size=7, stride=2, padding=3, padding_mode='replicate', dtype=self.dt, device=self.dv),
      nn.Conv1d(in_channels=32, out_channels=4, kernel_size=7, stride=2, padding=3, padding_mode='replicate', dtype=self.dt, device=self.dv),
    ])

    self.norm_layers = nn.ModuleList([
      nn.InstanceNorm1d(16, dtype=self.dt, device=self.dv),
      nn.InstanceNorm1d(32, dtype=self.dt, device=self.dv),
      nn.InstanceNorm1d(4, dtype=self.dt, device=self.dv),
    ])

    self.fc_layers = nn.ModuleList([
      nn.Linear(in_features=(input_sizes[0][1] // 8) * 4 + 15 * input_sizes[0][2] + 1, out_features=64, dtype=self.dt, device=self.dv),
      nn.Linear(in_features=64, out_features=32, dtype=self.dt, device=self.dv),
      nn.Linear(in_features=32, out_features=input_sizes[1], dtype=self.dt, device=self.dv),
    ])

    self.act_layers = nn.ModuleList([
      act_str_to_fn(act_fun_str, device=self.dv, dtype=self.dt) for _ in range(5)  # cnn1, cnn2, cnn3, fc1, fc2
    ])

    self.flatten = nn.Flatten()

  def forward(self, x_in: torch.Tensor, z_in: torch.Tensor, data_info: dict[str, Any] | None = None) -> torch.Tensor:
    n_samples = x_in.shape[0]
    n_layers = data_info.get('n_layers', 1) if data_info is not None else 1

    x_cnn1 = self.act_layers[0](self.norm_layers[0](self.cnn_layers[0](x_in)))    # [n_samples, 16, n_regions]
    x_cnn2 = self.act_layers[1](self.norm_layers[1](self.cnn_layers[1](x_cnn1)))  # [n_samples, 32, n_regions]
    x_cnn3 = self.act_layers[2](self.norm_layers[2](self.cnn_layers[2](x_cnn2)))  # [n_samples, 4, n_regions]

    out = self.flatten(x_cnn3)
    x_f = torch.concat([out, z_in.flatten(1)], axis=1)
    x_fc1 = self.act_layers[3](self.fc_layers[0](x_f))
    x_fc2 = self.act_layers[4](self.fc_layers[1](x_fc1))
    x_out = self.fc_layers[2](x_fc2)  # [n_samples, output_size]

    return x_out.reshape(n_samples, n_layers, -1)

  def get_weights(self) -> Dict[str, torch.Tensor]:
    return {name: param for name, param in self.named_parameters() if param.requires_grad}

  def get_n_params(self) -> int:
    return sum(p.numel() for p in self.parameters() if p.requires_grad)


class HeadSSDScaleSmallSysNetwork(nn.Module):
  """Shared system-parameter head for one anchor scale, run once over the full signal."""

  def __init__(self, input_sizes: tuple[tuple[int, int], int], act_fun_str: str, dt: torch.dtype, dv: torch.device):
    super().__init__()
    self.dt = dt
    self.dv = dv

    self.cnn_layers = nn.ModuleList([
      nn.Conv1d(in_channels=input_sizes[0][0], out_channels=16, kernel_size=7, stride=4, padding=3, padding_mode='replicate', dtype=self.dt, device=self.dv),
      nn.Conv1d(in_channels=16, out_channels=16, kernel_size=7, stride=4, padding=3, padding_mode='replicate', dtype=self.dt, device=self.dv),
      nn.Conv1d(in_channels=16, out_channels=32, kernel_size=7, stride=4, padding=3, padding_mode='replicate', dtype=self.dt, device=self.dv),
      nn.Conv1d(in_channels=32, out_channels=4, kernel_size=7, stride=4, padding=3, padding_mode='replicate', dtype=self.dt, device=self.dv),
    ])

    self.norm_layers = nn.ModuleList([
      nn.InstanceNorm1d(16, dtype=self.dt, device=self.dv),
      nn.InstanceNorm1d(16, dtype=self.dt, device=self.dv),
      nn.InstanceNorm1d(32, dtype=self.dt, device=self.dv),
    ])

    self.fc_layers = nn.ModuleList([
      nn.Linear(in_features=input_sizes[0][1] // 256 * 4 + input_sizes[1], out_features=64, dtype=self.dt, device=self.dv),
      nn.Linear(in_features=64, out_features=32, dtype=self.dt, device=self.dv),
      nn.Linear(in_features=32, out_features=input_sizes[1], dtype=self.dt, device=self.dv),
    ])

    self.act_layers = nn.ModuleList([
      act_str_to_fn(act_fun_str, device=self.dv, dtype=self.dt) for _ in range(6)  # cnn1-4, fc1, fc2
    ])

    self.flatten = nn.Flatten()

  def forward(self, x_in: torch.Tensor, z_in: torch.Tensor, data_info: dict[str, Any] | None = None) -> torch.Tensor:
    n_samples = x_in.shape[0]
    n_layers = data_info.get('n_layers', 1) if data_info is not None else 1

    x_cnn1 = self.act_layers[0](self.norm_layers[0](self.cnn_layers[0](x_in)))    # [n_samples, 16, n_regions]
    x_cnn2 = self.act_layers[1](self.norm_layers[1](self.cnn_layers[1](x_cnn1)))  # [n_samples, 16, n_regions]
    x_cnn3 = self.act_layers[2](self.norm_layers[2](self.cnn_layers[2](x_cnn2)))  # [n_samples, 32, n_regions]
    x_cnn4 = self.act_layers[3](self.cnn_layers[3](x_cnn3))                       # [n_samples, 4, n_regions]

    out = self.flatten(x_cnn4)
    x_f = torch.concat([out, z_in.flatten(1)], axis=1)
    x_fc1 = self.act_layers[4](self.fc_layers[0](x_f))
    x_fc2 = self.act_layers[5](self.fc_layers[1](x_fc1))
    x_out = self.fc_layers[2](x_fc2)  # [n_samples, output_size]

    return x_out.reshape(n_samples, n_layers, -1)

  def get_weights(self) -> Dict[str, torch.Tensor]:
    return {name: param for name, param in self.named_parameters() if param.requires_grad}

  def get_n_params(self) -> int:
    return sum(p.numel() for p in self.parameters() if p.requires_grad)
