"""Integration tests for model/networks/cnn_basic.py::CNNBasic."""
import pytest
import torch

from pidu.model.networks.cnn_basic import CNNBasic

pytestmark = pytest.mark.slow

ACTIVE_MODELS = ['system1', 'system2_0', 'system2_1']


@pytest.fixture
def cnn(system_info, model_info_builder, device, dtype):
  model_info = model_info_builder(ACTIVE_MODELS, physics=False)
  return CNNBasic(system_info=system_info, model_info=model_info, dtype=dtype, device=device, verbose=0)


def test_forward_returns_all_expected_keys_and_shapes(cnn, unfolds_batch):
  wn, x_sys, x_osc, y = unfolds_batch

  out = cnn(wn=wn, y=y, x_sys=x_sys, x_osc=x_osc)

  assert set(out.keys()) == {'y', 'z_sys', 'z_osc', 'p_sys', 'p_osc'}
  for modelname in ACTIVE_MODELS:
    assert modelname in out['y']
    assert out['y'][modelname].shape[0] == 1  # single unfold dimension


def test_active_systems_matches_dataset_systems(cnn, system_info):
  assert set(cnn.active_systems) == set(system_info['systems'].keys())


def test_cnn_size_scales_conv_and_fc_widths(system_info, model_info_builder, device, dtype):
  base_model_info = model_info_builder(ACTIVE_MODELS, physics=False, cnn_size=1.0)
  small_model_info = model_info_builder(ACTIVE_MODELS, physics=False, cnn_size=0.5)

  base = CNNBasic(system_info=system_info, model_info=base_model_info, dtype=dtype, device=device, verbose=0)
  small = CNNBasic(system_info=system_info, model_info=small_model_info, dtype=dtype, device=device, verbose=0)

  assert small.architecture_dims['conv_channels'] == tuple(max(1, round(c * 0.5)) for c in base.architecture_dims['conv_channels'])
  assert small.architecture_dims['fc_widths'] == tuple(max(1, round(w * 0.5)) for w in base.architecture_dims['fc_widths'])


def test_non_positive_cnn_size_raises(system_info, model_info_builder, device, dtype):
  model_info = model_info_builder(ACTIVE_MODELS, physics=False, cnn_size=0.0)
  with pytest.raises(ValueError):
    CNNBasic(system_info=system_info, model_info=model_info, dtype=dtype, device=device, verbose=0)
