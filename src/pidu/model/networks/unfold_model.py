"""
Per-unfold, per-system detection models: a shared BaseModel (backbone +
detection head, activation of raw outputs into physical parameters) and its
two concrete flavors, CNNModel and SSDModel.

Class names CNNModel/SSDModel are relied on elsewhere via
`__class__.__name__` (see scripts/generate_architecture_tables.py) and must
not change. Same goes for HeadCnnModel/HeadSSDModel in heads.py.
"""
import torch
import torch.nn as nn
from typing import Any, Dict

from pidu.model.networks.activations import ParameterActivations
from pidu.model.networks.blocks import Backbone
from pidu.model.networks.heads import HeadCnnModel, HeadSSDModel


class BaseModel(nn.Module):
  """Wraps a backbone + detection head for one system, at one unfold step."""

  def __init__(
    self,
    systemname: str,
    unfold: int,
    system_info: Dict[str, int | torch.Tensor],
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device('cpu'),
    use_backbone: bool = True,
    *args,
    **kwargs,
  ):
    super().__init__(*args, **kwargs)
    self.dt = dtype
    self.dv = device
    self.name = systemname
    self.modeltype = system_info['model']  # 'CNN' | 'SSD'
    self.unfold = unfold
    self.system_info = system_info
    self.conf_threshold = system_info.get('conf_threshold', 0.75)
    self.p_init = system_info.get('p_init', None)

    self.pred_sys = system_info.get('pred_sys', False)
    self.pred_osc = system_info.get('pred_osc', False)
    self.n_layers = system_info.get('n_layers', 1)
    self.use_backbone = use_backbone

    self.backbone: nn.Module | None = None
    self.detection_head: nn.Module
    self.param_activations: ParameterActivations | None = None
    self.use_diff_y = False

  def __str__(self):
    n_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
    return f"{self.modeltype}({self.name}-{self.unfold}-{n_trainable_params})"

  # -- Init --
  def init_act_fns(self, p_gen_params: Dict[str, Any], dt: torch.dtype = None, dv: torch.device = None, **kwargs):
    """Build the activation functions used to turn raw outputs into physical parameters."""
    self.param_activations = ParameterActivations(
      n_layers=p_gen_params.get('n_layers'),
      wn=self.system_info.get('wn'),
      region_width=self.system_info.get('region_width'),
      p_gen_params=p_gen_params,
      dt=dt,
      dv=dv,
      build_system=self.pred_sys,
      build_oscillator=self.pred_osc,
      factor=4,
    )

  def _create_info_backbone(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
    return {
      'use_diff_y': True,
      'in_channels': 3,
      'out_channels': 8,
      'act_fun_str': model_info['backbone_act_func'],
      'norm': model_info.get('backbone_norm', 'none'),  # batchnorm|layernorm|instancenorm|groupnorm|none
      'dropout_rate': model_info['dropout_rate'],
      'dtype': self.dt,
      'device': self.dv,
      'use_residual': model_info.get('use_residual', False),
    }

  def _create_info_detection_head(self, model_info: Dict[str, Any], system_info: Dict[str, Any]) -> Dict[str, Any]:
    shape_info = {
      'in_features': 8,
      'n_sys_pars': model_info['n_system_parameters'],
      'n_layers': system_info['n_layers'],
      'x_len': model_info['x_len'],
      'mask_osc': model_info['mask_x_osc'],
      'region_width': system_info.get('region_width'),
      'anchor_width': system_info.get('anchors'),
      'x_range': system_info['wn_ran'],
    }
    sys_info = {
      'dtype': self.dt,
      'device': self.dv,
    }
    return shape_info, sys_info

  def _adjust_head_info_for_no_backbone(self, detection_head_info: Dict[str, Any]) -> Dict[str, Any]:
    """When there's no backbone, the head reads the raw 3-channel input directly."""
    if not self.use_backbone:
      shape_info, sys_info = detection_head_info
      shape_info['in_features'] = 3
    return detection_head_info

  def _create_model(self, backbone_info, detection_head_info, *args, **kwargs):
    raise NotImplementedError

  # -- Activation of raw outputs into physical parameters --
  def activate_to_p(self, z_sys: torch.Tensor | None = None, z_osc: torch.Tensor | None = None):
    p_sys, p_osc = None, None
    pa = self.param_activations

    if z_sys is not None and self.pred_sys:
      p_sys = pa.to_system_params(z_sys)

    if z_osc is not None and self.pred_osc:
      n_regions_per_scale = self.system_info['n_regions_per_scale']
      n_layers = z_osc.shape[1]
      p_update = torch.zeros_like(z_osc, dtype=self.dt, device=self.dv)

      for idx_layer in range(n_layers):
        for idx_scale, n_regions in enumerate(n_regions_per_scale):
          p_update[..., idx_scale, :n_regions, 0] = pa.sigmoid(pa.act_os(z_osc[..., idx_scale, :n_regions, 0]))  # has_obj
          p_update[..., idx_scale, :n_regions, 1] = pa.act_w0[idx_layer][idx_scale](z_osc[..., idx_scale, :n_regions, 1])  # w0
          p_update[..., idx_scale, :n_regions, 2] = pa.act_wp[idx_layer][idx_scale](z_osc[..., idx_scale, :n_regions, 2])  # wp
          p_update[..., idx_scale, :n_regions, 3] = pa.act_g[idx_layer][idx_scale](z_osc[..., idx_scale, :n_regions, 3])   # g

      p_osc = p_update

    return p_sys, p_osc

  # -- Forward --
  def forward_backbone(self, z):
    return self.backbone(z)

  def forward_detection_head(self, feature_map, z_hat_sys, z_hat_osc):
    return self.detection_head(feature_map, z_sys=z_hat_sys, z_osc=z_hat_osc)

  def forward(
    self,
    y: torch.Tensor,
    z_hat_sys: torch.Tensor | None = None,
    z_hat_osc: torch.Tensor | None = None,
  ) -> Dict[str, torch.Tensor | None]:
    """One unfold step: extract features, predict an update to sys/osc parameters."""
    to_return = {'sys': None, 'osc': None}

    if self.use_diff_y:
      y_diff = (y[:, 0] - y[:, 1])[:, None, :]  # difference in y for contrast enhancement
      y = torch.concat((y, y_diff), dim=1)

    z = y
    feature_map = self.forward_backbone(z) if self.backbone else z
    z_update = self.forward_detection_head(feature_map, z_hat_sys=z_hat_sys, z_hat_osc=z_hat_osc)

    if 'sys' in z_update:
      to_return['sys'] = z_hat_sys + z_update['sys']

    if 'osc' in z_update:
      to_return['osc'] = z_hat_osc + z_update['osc']

    return to_return

  def assert_device(self, device: torch.device):
    """Ensure all model parameters are on the specified device (cuda vs cuda:0 are treated as equal)."""
    expected_device = torch.device(device)
    for param in self.parameters():
      param_device = param.device
      if param_device.type != expected_device.type:
        raise ValueError(f"Model parameter on device {param_device}, expected {expected_device}\nModel: {self}, Param: {param}")
      if param_device.type == 'cuda':
        param_idx = param_device.index or 0
        expected_idx = expected_device.index or 0
        if param_idx != expected_idx:
          raise ValueError(f"Model parameter on device {param_device}, expected {expected_device}\nModel: {self}, Param: {param}")


class CNNModel(BaseModel):
  """BaseModel using the CNN detection head (single-scale, no anchors)."""

  def __init__(
    self,
    systemname: str,
    unfold: int,
    system_info: Dict[str, int | torch.Tensor],
    model_info: Dict[str, int | torch.Tensor],
    *args,
    **kwargs,
  ):
    super().__init__(systemname, unfold, system_info, *args, **kwargs)

    backbone_info = self._create_info_backbone(model_info)
    detection_head_info = self._adjust_head_info_for_no_backbone(
      self._create_info_detection_head(model_info, system_info)
    )
    self._create_model(backbone_info, detection_head_info)

  def _create_model(self, backbone_info: Dict[str, Any], detection_head_info: Dict[str, Any], *args, **kwargs):
    self.use_diff_y = backbone_info.pop('use_diff_y', False)
    shape_info, sys_info = detection_head_info

    if self.use_backbone:
      self.backbone = Backbone(**backbone_info)
    self.detection_head = HeadCnnModel(shape_info, sys_info, **kwargs)


class SSDModel(BaseModel):
  """BaseModel using the SSD detection head (multi-scale, anchor-based)."""

  def __init__(
    self,
    systemname: str,
    unfold: int,
    system_info: Dict[str, int | torch.Tensor],
    model_info: Dict[str, int | torch.Tensor],
    msd,
    *args,
    **kwargs,
  ):
    super().__init__(systemname, unfold, system_info, *args, **kwargs)
    self.msd = msd

    backbone_info = self._create_info_backbone(model_info)
    detection_head_info = self._adjust_head_info_for_no_backbone(
      self._create_info_detection_head(model_info, system_info)
    )
    self._create_model(backbone_info, detection_head_info)

  def _create_model(self, backbone_info: Dict[str, Any], detection_head_info: Dict[str, Any], *args, **kwargs):
    self.use_diff_y = backbone_info.pop('use_diff_y', False)
    shape_info, sys_info = detection_head_info

    osc_shape = [v for k, v in shape_info['mask_osc'].items() if self.name.startswith(k)][0].shape
    shape_info['osc_shape'] = osc_shape

    if self.use_backbone:
      self.backbone = Backbone(**backbone_info)
    self.detection_head = HeadSSDModel(shape_info, sys_info, **kwargs)
