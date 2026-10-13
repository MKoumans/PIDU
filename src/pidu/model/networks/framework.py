"""
MultiModelFramework: the orchestrator that combines per-system, per-unfold
detection models (CNNModel/SSDModel) into an iterative fit-refine loop —
combine per-model predictions into per-system parameters, activate them into
physical units, mask against known/true values, simulate the physical system
response, and feed the result back in for the next unfold.
"""
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from pidu.data.physics import NoTmm
from pidu.data.utils import pt2np
from pidu.model.networks.common import select_dict_from_subset
from pidu.model.networks.unfold_model import BaseModel, CNNModel, SSDModel
from pidu.model.utils import create_mask_wn


class MultiModelFramework(nn.Module):
  def __init__(
    self,
    n_unfolds: int,
    system_info: Dict[str, Dict[str, Any]],  # Information about the system parameters
    model_info: Dict[str, Dict[str, Any]],
    *args,
    **kwargs,
  ):
    super().__init__()
    self.n_unfolds = n_unfolds
    self.dt = kwargs.get('dtype', torch.float32)
    self.dv = kwargs.get('device', torch.device('cpu'))
    self.verbose = kwargs.get('verbose', 0)

    self.p_gen = system_info.get('p_gen', None)
    self.systems_info = {k: v for k, v in system_info['systems'].items() if v is not None}
    self.model_info = model_info
    self.init_model = False
    self.use_physics = model_info.get('physics', True)

    self.active_systems = list(self.systems_info.keys())
    self.active_models = model_info['active_models']

    self._init_systems(self.use_physics, system_info.get('dataset_info', None))
    self._init_models(model_info)
    self._init_masks(
      model_info.get('mask_x_sys', {}),
      model_info.get('mask_x_osc', {}),
      model_info.get('create_wn_mask', False),
    )

  def __str__(self):
    return f"MMF[{len(self.active_models)}]"

  def __call__(self, wn: torch.Tensor, y: torch.Tensor, x_sys: torch.Tensor = None, x_osc: torch.Tensor = None):
    return self.forward(wn, y, x_sys, x_osc)

  def _print(self, string: str, indent: int = 0, level: int = 1):
    if self.verbose >= level:
      print(f"{' ' * indent}{string}")

  def _nvtx_push(self, name: str):
    if getattr(self, 'profile_nvtx', False) and torch.cuda.is_available() and self.dv.type == 'cuda':
      torch.cuda.nvtx.range_push(name)

  def _nvtx_pop(self):
    if getattr(self, 'profile_nvtx', False) and torch.cuda.is_available() and self.dv.type == 'cuda':
      torch.cuda.nvtx.range_pop()

  def _select_dict_from_subset(self, dict_: dict, model_name: str) -> Dict[str, Any]:
    return select_dict_from_subset(dict_, model_name)

  # -- Init --
  def _init_systems(self, use_physics: bool, dataset_info: Dict[str, Any]):
    self.dataset_info = dataset_info
    self.system_function = dataset_info['system_function']

    if not use_physics:
      for k, v in self.systems_info.items():
        self.system_function[k] = NoTmm(self.dt, self.dv, export_mode=v['return_mode'])

    n_outs = len(dataset_info['data_y_shape'])
    self.n_sys_params = nsp = dataset_info['n_system_parameters']
    self.n_osc_params = nop = 4 * 4  # 4 parameters per cell, 4 cells
    self.n_layers = nl = dataset_info['n_layers']

    self.sys_W = nn.Linear(n_outs * nsp * nl, nsp * nl, bias=False, dtype=self.dt, device=self.dv)
    self.osc_W = nn.Linear(n_outs * nop * nl, nop * nl, bias=False, dtype=self.dt, device=self.dv)

  def _init_models(self, model_info: Dict[str, Dict[str, Any]]):
    n_samples, n_layers = self.dataset_info['n_samples'], self.dataset_info['n_layers']
    modelnames = self.active_models

    msd = {
      'n_scales': self.dataset_info.get('n_scales', None),
      'n_regions_per_scale': self.dataset_info.get('n_regions_per_scale', None),
      'n_regions_per_scale_max': self.dataset_info.get('n_regions_per_scale_max', None),
      'n_attribs_label': self.dataset_info.get('n_attribs_label', None),
    }
    msd['label_shapes'] = (n_samples, n_layers, msd['n_scales'], msd['n_regions_per_scale_max'], msd['n_attribs_label'])

    sys_info = self.systems_info
    dat_info = self.dataset_info

    models: nn.ModuleDict = nn.ModuleDict()
    for modelname in modelnames:
      systeminfo = self._select_dict_from_subset(sys_info, modelname)
      model_type = systeminfo.get('model', None)  # 'CNN' | 'SSD'

      model_info.update({
        'n_system_parameters': dat_info['n_system_parameters'],
        'x_len': self._select_dict_from_subset(dat_info['x_len'], modelname),
      })
      systeminfo.update({'n_layers': self.n_layers})

      models_unfold = [
        self._create_model(model_type, modelname, i_unfold, systeminfo, model_info, msd=msd)
        for i_unfold in range(self.n_unfolds)
      ]
      models[modelname] = nn.ModuleList(models_unfold)

    self.system_models: nn.ModuleDict = models
    self.init_model = True
    self.conf_threshold = model_info.get('conf_threshold', 0.75)

    for modelname in self.active_models:
      for model in self.system_models[modelname]:
        model.assert_device(self.dv)

  def _init_masks(self, mask_sys: Dict[str, torch.Tensor], mask_osc: Dict[str, torch.Tensor], create_wn_mask: bool = False):
    self.mask_sys = mask_sys
    self.mask_osc = mask_osc
    self.wn_mask = {}

    for system_name in self.systems_info.keys():
      mask = create_mask_wn(self.systems_info[system_name]['wn'], system_name)
      self.wn_mask[system_name] = mask[None, None, :] if create_wn_mask else torch.ones_like(mask)

  def _init_forward(
    self,
    x_sys: Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
    x_osc: Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
    y: Dict[str, torch.Tensor],
  ):
    dt, dv = self.dt, self.dv
    z_sys_init, p_sys_true = x_sys
    z_osc_init, p_osc_true = x_osc

    z_hat_sys, z_hat_osc = {}, {}
    p_hat_sys, p_hat_osc, y_hat = {}, {}, {}

    for modelname in self.active_models:
      systemname = [k for k in self.active_systems if modelname.startswith(k)][0]
      mask_sys = self.mask_sys.get(systemname, None)
      mask_osc = self.mask_osc.get(systemname, None)

      if mask_sys is not None:
        z_hat_sys[modelname] = torch.where(
          mask_sys.bool(),
          z_sys_init[systemname],
          torch.zeros_like(z_sys_init[systemname], dtype=dt, device=dv),
        )
        zeros_shape = [self.n_unfolds] + list(p_sys_true[systemname].shape)
        p_hat_sys[systemname] = torch.zeros(zeros_shape, dtype=dt, device=dv)
      else:
        p_hat_sys[systemname] = [None] * self.n_unfolds

      if mask_osc is not None:
        z_hat_osc[modelname] = torch.where(
          mask_osc.bool(),
          z_osc_init[systemname],
          torch.zeros_like(z_osc_init[systemname], dtype=dt, device=dv),
        )
        zeros_shape = [self.n_unfolds] + list(p_osc_true[systemname].shape)
        p_hat_osc[systemname] = torch.zeros(zeros_shape, dtype=dt, device=dv)
      else:
        p_hat_osc[systemname] = [None] * self.n_unfolds

      zeros_shape = [self.n_unfolds] + list(y[modelname].shape)
      y_hat[modelname] = torch.zeros(zeros_shape, dtype=dt, device=dv)

    return (z_hat_sys, z_hat_osc), (p_hat_sys, p_hat_osc, y_hat), (p_sys_true, p_osc_true)

  def _create_model(
    self,
    model_type: str,
    systemname: str,
    i_unfold: int,
    systeminfo: Dict,
    model_info: Dict,
    **kwargs,
  ) -> BaseModel:
    common_arguments = {
      'systemname': systemname,
      'unfold': i_unfold,
      'system_info': systeminfo,
      'model_info': model_info,
      'dtype': self.dt,
      'device': self.dv,
      'use_backbone': False,
    }

    if model_type == 'CNN':
      model = CNNModel(**common_arguments)
    elif model_type == 'SSD':
      model = SSDModel(**common_arguments, msd=kwargs.get('msd'))
    else:
      raise ValueError(f"Unknown model type: {model_type}")

    model.init_act_fns(p_gen_params=self.p_gen, dt=self.dt, dv=self.dv)
    return model

  # -- Combine / activate / mask --
  def _combine(
    self,
    sys: Dict[str, torch.Tensor],
    osc: Dict[str, torch.Tensor],
    mask: Dict[str, Any],
  ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Combine per-model predictions into one prediction per system (mean if only one model, learned linear combination otherwise)."""
    sys_out = [v for k, v in sys.items() if k in self.active_models and v is not None]
    osc_out = [v for k, v in osc.items() if k in self.active_models and v is not None]

    if len(sys_out) > 1:
      sys_out = torch.stack(sys_out, dim=1)  # [n_samples, n_systems, n_layers, n_sys_pars]
      sys_out_mean = self.sys_W(sys_out.flatten(1, 3)).reshape(-1, self.n_layers, self.n_sys_params)
    elif len(sys_out) == 1:
      sys_out_mean = sys_out[0]
    else:
      sys_out_mean = None

    if len(osc_out) > 1:
      osc_out_shape = osc_out[0].shape
      osc_out = torch.stack(osc_out, dim=1)
      osc_out_mean = self.osc_W(osc_out.flatten(1, 5)).reshape(osc_out_shape)
    elif len(osc_out) == 1:
      osc_out_mean = osc_out[0]
    else:
      osc_out_mean = None

    to_return_sys = {name: sys_out_mean for name in self.active_systems}
    to_return_osc = {name: osc_out_mean for name in self.active_systems}
    return to_return_sys, to_return_osc

  def _activate(self, i_unfold: int, z_sys: torch.Tensor, z_osc: torch.Tensor) -> Tuple[Dict, Dict]:
    """Apply activation functions to the combined latent variables, per system."""
    p_sys, p_osc = {}, {}
    for systemname in self.active_systems:
      # Use the activations of the first model that belongs to this system cluster.
      model: BaseModel = [v for k, v in self.system_models.items() if k.startswith(systemname)][0][i_unfold]
      p_sys[systemname], p_osc[systemname] = model.activate_to_p(z_sys[systemname], z_osc[systemname])
    return p_sys, p_osc

  def _mask(
    self,
    true: Tuple[torch.Tensor, torch.Tensor],
    hat_sys: Dict[str, torch.Tensor],
    hat_osc: Dict[str, torch.Tensor],
  ):
    """Combine predicted and true parameters according to the sys/osc masks, and gate oscillators by objectness score."""
    sys, osc, osc_opt, out = {}, {}, {}, {}
    sys_true, osc_true = true

    for systemname in self.active_systems:
      sys[systemname] = torch.where(self.mask_sys[systemname].bool(), hat_sys[systemname], sys_true[systemname])

    for systemname in self.active_systems:
      if hat_osc and systemname in osc_true:
        n_samples, n_layers, n_scales, n_regions_max, n_attribs = hat_osc[systemname].shape
        osc[systemname] = torch.where(self.mask_osc[systemname].bool(), hat_osc[systemname], osc_true[systemname])

        # Only keep oscillator parameters whose objectness score clears the confidence threshold.
        mask_to_update = osc[systemname][..., 0] > self.conf_threshold
        mask_to_update_complete = mask_to_update[..., None]

        p_osc_shaped = osc[systemname][..., -3:]  # w0, wp, g
        p_osc_optics = torch.where(mask_to_update_complete, p_osc_shaped, torch.zeros_like(p_osc_shaped))
        osc_opt[systemname] = p_osc_optics.reshape(n_samples, n_layers, -1)
      else:
        osc[systemname] = None
        osc_opt[systemname] = None

    for systemname in self.active_systems:
      if systemname in sys and osc[systemname] is not None:
        out[systemname] = torch.concat((sys[systemname], osc_opt[systemname]), dim=-1)
      else:
        out[systemname] = sys[systemname]

    return out, sys, osc

  def _forward_system(self, p, wn: Dict[str, torch.Tensor], *args, **kwargs):
    """Simulate the physical system response for each system's current parameters."""
    y = {}
    for system_name in self.active_systems:
      wn_system = wn[system_name]
      if wn_system.ndim > 1:
        wn_system = wn_system[0]

      y_ = self.system_function[system_name](p[system_name], wn_system, *args, **kwargs)
      if isinstance(y_, tuple):
        for i in range(len(y_)):
          y[f"{system_name}_{i}"] = y_[i][:, None, :]
      else:
        y[system_name] = y_[:, None, :]

    assert all(k in self.active_models for k in y.keys()), f"Not all models have a y: {y.keys()} vs {self.active_models}."
    return y

  # -- Forward --
  def forward(
    self,
    wn: Dict[str, torch.Tensor],
    y: Dict[str, torch.Tensor],
    x_sys: Dict[str, torch.Tensor],
    x_osc: Dict[str, torch.Tensor],
  ) -> Dict[str, Any]:
    self._nvtx_push('mmf_forward')
    self._nvtx_push('init_forward')
    preds, store, p_true = self._init_forward(x_sys, x_osc, y)
    (z_hat_sys, z_hat_osc) = preds
    (p_hat_sys, p_hat_osc, y_hat) = store
    self._nvtx_pop()

    # -- Initial step --
    z_sys_hat, z_osc_hat = self._combine(z_hat_sys, z_hat_osc, self.mask_sys)
    activated = self._activate(0, z_sys_hat, z_osc_hat)
    p, p_sys, p_osc = self._mask(p_true, *activated)

    self._nvtx_push('initial_system_sim')
    with torch.no_grad():
      y_hat_unfold = self._forward_system(p, wn)
    self._nvtx_pop()

    self._print(f"{self} Forward function:")
    for i_unfold in range(self.n_unfolds):
      self._nvtx_push(f'unfold_{i_unfold}')
      self._print(f"Unfold {i_unfold}:", 2)

      for modelname, model in self.system_models.items():
        self._nvtx_push(f'model_{modelname}_u{i_unfold}')
        y_model = y[modelname][:, None, :]
        y_hat_model = y_hat_unfold[modelname]

        z_sys_hat_model = self._select_dict_from_subset(z_sys_hat, modelname)
        z_osc_hat_model = self._select_dict_from_subset(z_osc_hat, modelname)
        wn_mask = self._select_dict_from_subset(self.wn_mask, modelname)

        y_model = torch.concat((y_hat_model, y_model), dim=1)
        y_model = torch.where(wn_mask.bool(), y_model, torch.zeros_like(y_model))

        output_model = model[i_unfold](y_model, z_sys_hat_model, z_osc_hat_model)

        y_hat[modelname][i_unfold] = y_model[:, 0]  # Remove 'channel' dimension.
        z_sys_hat[modelname] = output_model['sys']
        z_osc_hat[modelname] = output_model['osc']

        self._print(f"Output y: {y_model[:, 0].shape}", 4)
        self._nvtx_pop()

      self._nvtx_push('combine_activate_mask')
      z_sys_hat, z_osc_hat = self._combine(z_sys_hat, z_osc_hat, self.mask_sys)
      activated = self._activate(i_unfold, z_sys_hat, z_osc_hat)
      p, p_sys, p_osc = self._mask(p_true, *activated)
      self._nvtx_pop()

      self._nvtx_push('system_sim')
      with torch.no_grad():
        y_hat_unfold = self._forward_system(p, wn)
        for key, y_ in y_hat_unfold.items():
          y_hat[key][i_unfold] = y_[:, 0]  # Remove 'channel' dimension.
      self._nvtx_pop()

      for systemname in self.active_systems:
        p_hat_sys[systemname][i_unfold] = p_sys[systemname]
        p_hat_osc[systemname][i_unfold] = p_osc[systemname]
      self._nvtx_pop()

    self._nvtx_pop()
    return {
      'y': y_hat,
      'z_sys': z_sys_hat,
      'z_osc': z_osc_hat,
      'p_sys': p_hat_sys,
      'p_osc': p_hat_osc,
    }

  # -- Introspection --
  def summary(self):
    info = self.get_info()
    print(f"MultiModelFramework Summary:")
    print(f" - Number of unfolds: {info['n_unfolds']}")
    print(f" - Active systems: {info['active_systems']}")
    print(f" - Active models: {info['active_models']}")

    print(f"\n - ModelFramework:")
    for modelname, model_info in info['model_framework'].items():
      print(f"  - Model: {modelname}")
      for unfold_data in model_info['unfolds']:
        print(f"   - Unfold {unfold_data['unfold']}: {unfold_data['model']}")
        print(f"    - Backbone: {unfold_data['backbone_parameters']}")
        if 'detectionhead_sys_parameters' in unfold_data and 'detectionhead_osc_parameters' in unfold_data:
          print(f"    - Detectionhead (sys): {unfold_data['detectionhead_sys_parameters']}")
          print(f"    - Detectionhead (osc): {unfold_data['detectionhead_osc_parameters']}")
        else:
          print(f"    - Detectionhead: {unfold_data['detectionhead_parameters']}")

        if 'ssd_scales' in unfold_data:
          for ssd_scale in unfold_data['ssd_scales']:
            print(f"   - SSD Scale {ssd_scale['scale']}: {ssd_scale['cnn_parameters']}, {ssd_scale['fc_parameters']}")

    print(f"Total number of trainable parameters: {info['total_number_of_model_parameters_k']}k")

  def train(self, mode: bool = True):
    super().train(mode)
    for systemname, models_unfold in self.system_models.items():
      for model in models_unfold:
        model.train(mode)

  def eval(self):
    super().eval()
    for systemname, models_unfold in self.system_models.items():
      for model in models_unfold:
        model.eval()

  def get_info(self) -> Dict[str, Any]:
    model_framework_info: Dict[str, Any] = {}
    total_number_of_model_params = 0

    for modelname, models in self.system_models.items():
      unfold_info = []

      for i_unfold, model in enumerate(models):
        n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_params_backbone = model.backbone.n_params if getattr(model, 'backbone', None) else 0
        n_params_head = model.detection_head.n_params if getattr(model, 'detection_head', None) else 0
        n_params_head_sys = None
        n_params_head_osc = None

        # Split detection head parameters when a head (or scale-head) has explicit
        # system and oscillator subnetworks.
        detection_head = getattr(model, 'detection_head', None)
        if detection_head is not None:
          if hasattr(detection_head, 'sys_network') and hasattr(detection_head, 'cell_network'):
            n_params_head_sys = sum(p.numel() for p in detection_head.sys_network.parameters() if p.requires_grad)
            n_params_head_osc = sum(p.numel() for p in detection_head.cell_network.parameters() if p.requires_grad)
          elif hasattr(detection_head, 'models'):
            total_sys_params = 0
            total_osc_params = 0
            has_split_scale = False
            for scale_model in detection_head.models.values():
              if hasattr(scale_model, 'sys_network') and hasattr(scale_model, 'cell_network'):
                has_split_scale = True
                total_sys_params += sum(p.numel() for p in scale_model.sys_network.parameters() if p.requires_grad)
                total_osc_params += sum(p.numel() for p in scale_model.cell_network.parameters() if p.requires_grad)

            if has_split_scale:
              n_params_head_sys = total_sys_params
              n_params_head_osc = total_osc_params

        unfold_data = {
          'unfold': i_unfold,
          'model': str(model),
          'trainable_parameters': n_trainable_params,
          'backbone_parameters': n_params_backbone,
          'detectionhead_parameters': n_params_head,
        }

        if n_params_head_sys is not None and n_params_head_osc is not None:
          unfold_data['detectionhead_sys_parameters'] = n_params_head_sys
          unfold_data['detectionhead_osc_parameters'] = n_params_head_osc

        if 'SSD' in modelname and hasattr(model, 'models'):
          unfold_data['ssd_scales'] = [
            {
              'scale': i_ssd_scale,
              'cnn_parameters': ssd_scale.n_params_cnn,
              'fc_parameters': ssd_scale.n_params_fc,
            }
            for i_ssd_scale, ssd_scale in enumerate(model.models)
          ]

        unfold_info.append(unfold_data)
        total_number_of_model_params += n_trainable_params

      model_framework_info[modelname] = {'unfolds': unfold_info}

    return {
      'n_unfolds': self.n_unfolds,
      'active_systems': self.active_systems,
      'active_models': self.active_models,
      'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad),
      'model_framework': model_framework_info,
      'total_number_of_model_parameters': total_number_of_model_params,
      'total_number_of_model_parameters_k': total_number_of_model_params / 1000,
    }

  def get_unfold_parameters(self, i_unfold: int) -> list:
    system_models = nn.ModuleDict()
    for systemname, models_system in self.system_models.items():
      system_models[systemname] = models_system[i_unfold]

    params_ = [p for module in system_models.values() for p in module.parameters()]
    return [p for p in params_ if p.requires_grad]

  def get_gradient(self, i_unfold: int | None = None) -> Tuple[Any, list]:
    system_models = nn.ModuleDict()
    idx_new_system = [0]
    params_ = []

    i_unfolds = list(range(self.n_unfolds)) if i_unfold is None else [i_unfold]

    for systemname, models_system in self.system_models.items():
      for i_unfold in i_unfolds:
        system_models[f"{systemname}_u{i_unfold}"] = models_system[i_unfold]

    for name, module in system_models.items():
      params_ += list(module.parameters())
      idx_new_system.append(sum(p.numel() for p in params_))

    param_grads = [p.grad.flatten() for p in params_ if (p.requires_grad and p.grad is not None)]
    if len(param_grads) > 0:
      param_grads = pt2np(torch.cat(param_grads))

    return param_grads, idx_new_system
