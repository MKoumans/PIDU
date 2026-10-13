"""
Unit tests for model/networks/activations.py::ParameterActivations.

Uses small hand-built wn/region_width/p_gen_params instead of a real
simulated dataset - this class has no dependency on the data pipeline.
"""
import torch

from pidu.model.networks.activations import ParameterActivations

DT = torch.float32
DV = torch.device('cpu')
N_LAYERS = 2
WN = torch.linspace(1.0, 256.0, 256, dtype=DT)
REGION_WIDTH = [64, 128]  # 2 scales


def make_p_gen_params():
  wp_ranges_by_layer = [(1.0, 2.0), (1.0, 2.0)]
  g_ranges_by_layer = [(0.1, 0.5), (0.1, 0.5)]
  return {
    'n_layers': N_LAYERS,
    'oscs_start': [None, wp_ranges_by_layer, g_ranges_by_layer],
    'oscs_end': [None, wp_ranges_by_layer, g_ranges_by_layer],
    'wp': [[(1.0, 2.0), (2.0, 3.0)] for _ in range(N_LAYERS)],
    'g': [[(0.1, 0.3)] for _ in range(N_LAYERS)],
  }


def make_activations(**overrides):
  kwargs = dict(
    n_layers=N_LAYERS,
    wn=WN,
    region_width=REGION_WIDTH,
    p_gen_params=make_p_gen_params(),
    dt=DT,
    dv=DV,
  )
  kwargs.update(overrides)
  return ParameterActivations(**kwargs)


def test_build_system_only_leaves_oscillator_unset():
  pa = make_activations(build_system=True, build_oscillator=False)

  assert len(pa.act_d) == N_LAYERS
  assert len(pa.act_e_inf) == N_LAYERS
  assert len(pa.act_wp_d) == N_LAYERS
  assert len(pa.act_g_d) == N_LAYERS
  assert len(pa.act_w0b) == N_LAYERS
  assert len(pa.act_w0a) == N_LAYERS
  assert pa.act_w0 is None
  assert pa.act_wp is None
  assert pa.act_g is None


def test_build_oscillator_only_leaves_system_unset():
  pa = make_activations(build_system=False, build_oscillator=True)

  assert pa.sigmoid is None
  assert pa.act_os is None
  assert pa.act_d is None
  assert len(pa.act_w0) == N_LAYERS
  for per_layer in pa.act_w0:
    assert len(per_layer) == len(REGION_WIDTH)


def test_missing_wn_disables_boundary_activations_but_not_per_layer_ones():
  pa = make_activations(wn=None, build_system=True, build_oscillator=True)

  # Per-layer activations that don't depend on wn still build.
  assert len(pa.act_d) == N_LAYERS
  assert len(pa.act_e_inf) == N_LAYERS

  # Boundary-frequency activations need wn and must not be built.
  assert pa.act_w0b is None
  assert pa.act_w0a is None

  # Oscillator activations also need wn.
  assert len(pa.act_w0) == 0
  assert len(pa.act_wp) == 0
  assert len(pa.act_g) == 0


def test_missing_p_gen_keys_disables_boundary_oscillator_activations():
  p_gen_params = make_p_gen_params()
  del p_gen_params['oscs_start']
  del p_gen_params['oscs_end']

  pa = make_activations(p_gen_params=p_gen_params, build_system=True, build_oscillator=False)

  assert pa.act_wpb is None
  assert pa.act_gb is None
  assert pa.act_wpa is None
  assert pa.act_ga is None
  # Unaffected fields still build.
  assert len(pa.act_d) == N_LAYERS


def test_to_system_params_shape_and_angle_passthrough():
  pa = make_activations(build_system=True, build_oscillator=False)
  batch = 4
  z_sys = torch.randn(batch, N_LAYERS, 11, dtype=DT)

  p_sys = pa.to_system_params(z_sys)

  assert p_sys.shape == z_sys.shape
  # Index 0 (angle of incidence) is passed through unmodified by design.
  assert torch.equal(p_sys[:, :, 0], z_sys[:, :, 0])


def test_to_system_params_falls_back_to_raw_value_when_boundary_activations_missing():
  p_gen_params = make_p_gen_params()
  del p_gen_params['oscs_start']
  del p_gen_params['oscs_end']

  pa = make_activations(wn=None, p_gen_params=p_gen_params, build_system=True, build_oscillator=False)
  batch = 3
  z_sys = torch.randn(batch, N_LAYERS, 11, dtype=DT)

  p_sys = pa.to_system_params(z_sys)

  # w0/wp/g before+after (indices 5-10) have no activation built -> raw passthrough.
  assert torch.equal(p_sys[:, :, 5:], z_sys[:, :, 5:])
