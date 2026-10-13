"""
Shared fixtures for the model-layer tests (tests/test_model_*.py).

Two tiers:
  - unit tier: no dataset needed, CPU/float32, instant.
  - integration tier (marked `slow`): builds one tiny simulated dataset per
    test session via `data.Simulation.create_dataset` and reuses it, since
    BaseModel/CNNModel/SSDModel are only ever constructed through
    MultiModelFramework, which needs the real backbone/detection-head shape
    contracts from model/networks/blocks.py and model/networks/heads.py.
"""
import pytest
import torch
from torch.utils.data import DataLoader

from pidu.data.simulation import create_dataset, init_usecase
from pidu.model.utils import create_masks_osc, create_masks_sys


def pytest_configure(config):
  config.addinivalue_line("markers", "slow: needs a real simulated dataset (data.Simulation.create_dataset)")


@pytest.fixture(autouse=True)
def _isolated_case_dir(tmp_path, monkeypatch):
  """Send everything written under the case directory (e.g. benchmark_lma's loss_lma_* exports) to a
  temporary folder instead of the real outputs/case<N>/."""
  monkeypatch.setenv('PIDU_CASE_DIR', str(tmp_path / 'case'))


USECASE = 2  # system1 -> SSD, system2 -> CNN; exercises both model types.
TINY_N_SAMPLES = 2**3


@pytest.fixture(scope="session")
def tiny_dataset_bundle():
  """One small (8-sample) usecase-2 dataset, generated once per test session."""
  p_gen_params, sys_pars = init_usecase(usecase=USECASE, device='cpu')
  dataset = create_dataset(
    usecase=USECASE,
    setname=f"pytest_tiny_{TINY_N_SAMPLES}",
    n_samples=TINY_N_SAMPLES,
    system_params=sys_pars,
    p_gen_params=p_gen_params,
    verbose=False,
    export=False,
    structured_noise=[False] * len(sys_pars),
    device='cpu',
  )
  dataset.y_return_mode = 'noised'  # as in training: models take the noised spectrum only
  device, dtype = dataset.get_device_dtype()
  dataset_info = dataset.get_info_for_model(False)
  return {
    'dataset': dataset,
    'dataset_info': dataset_info,
    'p_gen_params': p_gen_params,
    'sys_pars': sys_pars,
    'device': device,
    'dtype': dtype,
  }


def build_model_info(bundle, active_models, cnn_size=1.0, physics=True, unfolds=2):
  dataset_info = bundle['dataset_info']
  dtype, device = bundle['dtype'], bundle['device']

  return {
    'physics': physics,
    'backbone_act_func': 'tanh',
    'backbone_norm': 'instancenorm',
    'dropout_rate': 0.2,
    'cnn_size': cnn_size,
    'active_models': active_models,
    'mask_x_sys': create_masks_sys(
      usecase=USECASE,
      n_layers=dataset_info['n_layers'],
      n_sys_pars=dataset_info['n_system_parameters'],
      dt=dtype,
      dv=device,
    ),
    'mask_x_osc': create_masks_osc(
      n_layers=dataset_info['n_layers'],
      n_scales=dataset_info['n_scales'],
      n_regions_max=dataset_info['n_regions_per_scale_max'],
      n_attribs_label=dataset_info['n_attribs_label'],
      std_val={'system1': 1, 'system2': 1},
      dt=dtype,
      dv=device,
    ),
    'create_wn_mask': True,
    'use_residual': True,
    'conf_threshold': 0.75,
  }


def build_system_info(bundle):
  return {
    'p_gen': bundle['p_gen_params'],
    'dataset_info': bundle['dataset_info'],
    'systems': bundle['sys_pars'],
  }


@pytest.fixture
def model_info_builder(tiny_dataset_bundle):
  """Callable: model_info_builder(active_models, **overrides) -> model_info dict."""
  def _build(active_models, **overrides):
    return build_model_info(tiny_dataset_bundle, active_models, **overrides)
  return _build


@pytest.fixture
def system_info(tiny_dataset_bundle):
  return build_system_info(tiny_dataset_bundle)


@pytest.fixture
def device(tiny_dataset_bundle):
  return tiny_dataset_bundle['device']


@pytest.fixture
def dtype(tiny_dataset_bundle):
  return tiny_dataset_bundle['dtype']


@pytest.fixture
def unfolds_batch(tiny_dataset_bundle):
  """One batch's worth of (wn, x_sys, x_osc, y) straight from the tiny dataset."""
  loader = DataLoader(tiny_dataset_bundle['dataset'], batch_size=TINY_N_SAMPLES, shuffle=False)
  wn, x_sys, x_osc, y, idx = next(iter(loader))
  return wn, x_sys, x_osc, y
