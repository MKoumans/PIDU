"""
Integration tests for model/networks/unfold_model.py (BaseModel/CNNModel/SSDModel).

BaseModel/CNNModel/SSDModel are never constructed directly outside
MultiModelFramework._create_model, so these tests go through the framework
with a real (tiny) simulated dataset to get valid backbone/detection-head
shapes from model/networks/blocks.py and model/networks/heads.py.
"""
import pytest
import torch

from pidu.model.networks.framework import MultiModelFramework
from pidu.model.networks.unfold_model import CNNModel, SSDModel

pytestmark = pytest.mark.slow

# usecase 2: system1 -> SSD, system2 -> CNN (see conftest.USECASE).
ACTIVE_MODELS = ['system1', 'system2_0', 'system2_1']


@pytest.fixture
def framework(system_info, model_info_builder, device, dtype):
  model_info = model_info_builder(ACTIVE_MODELS, unfolds=1)
  return MultiModelFramework(
    n_unfolds=1,
    system_info=system_info,
    model_info=model_info,
    dtype=dtype,
    device=device,
    verbose=0,
  )


def test_system1_uses_ssd_model(framework):
  model = framework.system_models['system1'][0]
  assert isinstance(model, SSDModel)
  assert model.__class__.__name__ == 'SSDModel'
  assert model.detection_head is not None


def test_system2_uses_cnn_model(framework):
  model = framework.system_models['system2_0'][0]
  assert isinstance(model, CNNModel)
  assert model.__class__.__name__ == 'CNNModel'
  assert model.detection_head is not None


def test_str_reports_model_type_name_and_unfold(framework):
  model = framework.system_models['system1'][0]
  text = str(model)
  assert model.modeltype in text
  assert model.name in text
  assert str(model.unfold) in text


def test_assert_device_passes_for_matching_device(framework, device):
  model = framework.system_models['system1'][0]
  model.assert_device(device)  # must not raise


def test_activate_to_p_produces_correctly_shaped_system_params(framework):
  model = framework.system_models['system1'][0]
  n_layers = model.n_layers
  n_sys_pars = 11
  batch = 2
  z_sys = torch.randn(batch, n_layers, n_sys_pars, dtype=model.dt, device=model.dv)

  p_sys, p_osc = model.activate_to_p(z_sys=z_sys, z_osc=None)

  if model.pred_sys:
    assert p_sys.shape == z_sys.shape
  else:
    assert p_sys is None
  assert p_osc is None
