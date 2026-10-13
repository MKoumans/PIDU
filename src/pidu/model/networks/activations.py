"""
Shared construction of the per-layer/per-scale activation functions that map
raw network outputs (z) to physically-meaningful parameters (p): layer
thickness/permittivity/Drude terms plus before/after boundary oscillators for
the system branch, and per-region resonance frequency/strength/damping for
the oscillator branch.

This logic used to be duplicated almost verbatim between BaseModel and
CNNBasic (unfold_model.py / cnn_basic.py); both now build one
ParameterActivations instance instead.
"""
import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional

from pidu.model.networks.param_activations import (
  Activation, ActivationD, ActivationdG, ActivationdWp, ActivationEinf,
  ActivationOScore, ActivationW0, ActivationW0A, ActivationW0B,
)


class ParameterActivations(nn.Module):
  """Builds and applies the system/oscillator activation functions for one model."""

  def __init__(
    self,
    n_layers: int,
    wn: Optional[torch.Tensor],
    region_width: Optional[List[int]],
    p_gen_params: Optional[Dict[str, Any]],
    dt: torch.dtype,
    dv: torch.device,
    build_system: bool = True,
    build_oscillator: bool = True,
    factor: float = 4,
  ):
    super().__init__()
    self.dt = dt
    self.dv = dv

    self.sigmoid: nn.Module | None = None
    self.act_os: nn.Module | None = None
    self.act_e_inf: nn.ModuleList | None = None
    self.act_d: nn.ModuleList | None = None
    self.act_wp_d: nn.ModuleList | None = None
    self.act_g_d: nn.ModuleList | None = None
    self.act_w0b: nn.ModuleList | None = None
    self.act_wpb: nn.ModuleList | None = None
    self.act_gb: nn.ModuleList | None = None
    self.act_w0a: nn.ModuleList | None = None
    self.act_wpa: nn.ModuleList | None = None
    self.act_ga: nn.ModuleList | None = None
    self.act_w0: nn.ModuleList | None = None
    self.act_wp: nn.ModuleList | None = None
    self.act_g: nn.ModuleList | None = None

    if build_system:
      self._build_system(n_layers, wn, p_gen_params, factor)
    if build_oscillator:
      self._build_oscillator(n_layers, wn, region_width, p_gen_params, factor)

  # -- Construction --
  def _build_system(
    self,
    n_layers: int,
    wn: Optional[torch.Tensor],
    p_gen_params: Optional[Dict[str, Any]],
    factor: float,
  ):
    """Per-layer activations for thickness, eps_inf, Drude terms, and before/after boundary oscillators."""
    dt, dv = self.dt, self.dv
    self.sigmoid = nn.Sigmoid()
    self.act_os = ActivationOScore()

    has_boundary_freq = wn is not None
    has_boundary_osc = bool(p_gen_params) and 'oscs_start' in p_gen_params and 'oscs_end' in p_gen_params

    act_e_inf, act_d, act_wp_d, act_g_d = [], [], [], []
    act_w0b, act_wpb, act_gb = [], [], []
    act_w0a, act_wpa, act_ga = [], [], []

    for i_layer in range(n_layers):
      act_e_inf.append(ActivationEinf())  # eps_inf
      act_d.append(ActivationD())         # thickness
      act_wp_d.append(ActivationdG())     # drude - plasma frequency
      act_g_d.append(ActivationdWp())     # drude - damping factor

      if has_boundary_freq:
        act_w0b.append(ActivationW0B(wn[0], dt, dv))
        act_w0a.append(ActivationW0A(wn[-1], dt, dv))

      if has_boundary_osc:
        act_wpb.append(Activation(
          p_gen_params['oscs_start'][1][i_layer][0],
          p_gen_params['oscs_start'][1][i_layer][1],
          factor=factor,
        ))
        act_gb.append(Activation(
          p_gen_params['oscs_start'][2][i_layer][0],
          p_gen_params['oscs_start'][2][i_layer][1],
          factor=factor,
        ))
        act_wpa.append(Activation(
          p_gen_params['oscs_end'][1][i_layer][0],
          p_gen_params['oscs_end'][1][i_layer][1],
          factor=factor,
        ))
        act_ga.append(Activation(
          p_gen_params['oscs_end'][2][i_layer][0],
          p_gen_params['oscs_end'][2][i_layer][1],
          factor=factor,
        ))

    self.act_e_inf = nn.ModuleList(act_e_inf)
    self.act_d = nn.ModuleList(act_d)
    self.act_wp_d = nn.ModuleList(act_wp_d)
    self.act_g_d = nn.ModuleList(act_g_d)
    self.act_w0b = nn.ModuleList(act_w0b) if act_w0b else None
    self.act_w0a = nn.ModuleList(act_w0a) if act_w0a else None
    self.act_wpb = nn.ModuleList(act_wpb) if act_wpb else None
    self.act_gb = nn.ModuleList(act_gb) if act_gb else None
    self.act_wpa = nn.ModuleList(act_wpa) if act_wpa else None
    self.act_ga = nn.ModuleList(act_ga) if act_ga else None

  def _build_oscillator(
    self,
    n_layers: int,
    wn: Optional[torch.Tensor],
    region_width: Optional[List[int]],
    p_gen_params: Optional[Dict[str, Any]],
    factor: float,
  ):
    """Per-layer, per-scale activations for oscillator resonance frequency, strength, and damping."""
    dt, dv = self.dt, self.dv
    has_ranges = bool(p_gen_params) and 'wp' in p_gen_params and 'g' in p_gen_params

    act_w0, act_wp, act_g = [], [], []
    if has_ranges and wn is not None:
      n_scales = len(region_width)
      for i_layer in range(n_layers):
        ranges_wp = p_gen_params['wp'][i_layer]
        ranges_g = p_gen_params['g'][i_layer]
        wp_min, wp_max = min(r[0] for r in ranges_wp), max(r[1] for r in ranges_wp)
        g_min, g_max = min(r[0] for r in ranges_g), max(r[1] for r in ranges_g)

        act_w0_layer: List[Any] = [None] * n_scales
        act_wp_layer: List[Any] = [None] * n_scales
        act_g_layer: List[Any] = [None] * n_scales

        for i_scale in range(n_scales):
          act_w0_layer[i_scale] = ActivationW0(wn, region_width, i_scale, dt, dv)  # resonance frequency
          act_wp_layer[i_scale] = Activation(wp_min, wp_max, factor=factor)       # strength
          act_g_layer[i_scale] = Activation(g_min, g_max, factor=2)               # damping

        act_w0.append(nn.ModuleList(act_w0_layer))
        act_wp.append(nn.ModuleList(act_wp_layer))
        act_g.append(nn.ModuleList(act_g_layer))

    self.act_w0 = nn.ModuleList(act_w0)
    self.act_wp = nn.ModuleList(act_wp)
    self.act_g = nn.ModuleList(act_g)

  # -- Application --
  def to_system_params(self, z_sys: torch.Tensor) -> torch.Tensor:
    """Map raw system-branch outputs to physical parameters, one layer at a time."""
    n_layers = z_sys.shape[1]
    p_sys = torch.zeros_like(z_sys, dtype=self.dt, device=self.dv)

    for i_layer in range(n_layers):
      p_sys[:, i_layer, :] = torch.stack([
        z_sys[:, i_layer, 0],                                                                # angle of incidence
        self.act_d[i_layer](z_sys[:, i_layer, 1]),                                           # thickness
        self.act_e_inf[i_layer](z_sys[:, i_layer, 2]),                                       # eps_inf
        self.act_wp_d[i_layer](z_sys[:, i_layer, 3]),                                        # drude - plasma frequency
        self.act_g_d[i_layer](z_sys[:, i_layer, 4]),                                         # drude - damping factor
        self.act_w0b[i_layer](z_sys[:, i_layer, 5]) if self.act_w0b else z_sys[:, i_layer, 5],   # w0 before
        self.act_wpb[i_layer](z_sys[:, i_layer, 6]) if self.act_wpb else z_sys[:, i_layer, 6],   # wp before
        self.act_gb[i_layer](z_sys[:, i_layer, 7]) if self.act_gb else z_sys[:, i_layer, 7],     # g before
        self.act_w0a[i_layer](z_sys[:, i_layer, 8]) if self.act_w0a else z_sys[:, i_layer, 8],   # w0 after
        self.act_wpa[i_layer](z_sys[:, i_layer, 9]) if self.act_wpa else z_sys[:, i_layer, 9],   # wp after
        self.act_ga[i_layer](z_sys[:, i_layer, 10]) if self.act_ga else z_sys[:, i_layer, 10],   # g after
      ], dim=-1)

    return p_sys
