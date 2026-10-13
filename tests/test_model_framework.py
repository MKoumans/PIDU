"""Integration tests for model/networks/framework.py::MultiModelFramework."""
import pytest

from pidu.model.networks.framework import MultiModelFramework

pytestmark = pytest.mark.slow

ACTIVE_MODELS = ['system1', 'system2_0', 'system2_1']
N_UNFOLDS = 2


def _build_framework(system_info, model_info_builder, device, dtype, physics):
  model_info = model_info_builder(ACTIVE_MODELS, physics=physics, unfolds=N_UNFOLDS)
  return MultiModelFramework(
    n_unfolds=N_UNFOLDS,
    system_info=system_info,
    model_info=model_info,
    dtype=dtype,
    device=device,
    verbose=0,
  )


@pytest.fixture(params=[True, False], ids=['physics_on_PIDU', 'physics_off_DU'])
def framework(request, system_info, model_info_builder, device, dtype):
  return _build_framework(system_info, model_info_builder, device, dtype, physics=request.param)


def test_forward_returns_all_expected_keys(framework, unfolds_batch):
  wn, x_sys, x_osc, y = unfolds_batch

  out = framework(wn, y, x_sys, x_osc)

  assert set(out.keys()) == {'y', 'z_sys', 'z_osc', 'p_sys', 'p_osc'}


def test_forward_output_has_one_row_per_unfold(framework, unfolds_batch):
  wn, x_sys, x_osc, y = unfolds_batch

  out = framework(wn, y, x_sys, x_osc)

  for modelname in ACTIVE_MODELS:
    assert out['y'][modelname].shape[0] == framework.n_unfolds


def test_get_info_reports_consistent_totals(framework):
  info = framework.get_info()

  assert info['n_unfolds'] == framework.n_unfolds
  assert set(info['active_systems']) == set(framework.active_systems)
  assert set(info['active_models']) == set(framework.active_models)
  assert info['total_number_of_model_parameters'] == sum(
    unfold_data['trainable_parameters']
    for model_info in info['model_framework'].values()
    for unfold_data in model_info['unfolds']
  )


def test_get_unfold_parameters_returns_only_trainable_params(framework):
  params = framework.get_unfold_parameters(0)

  assert len(params) > 0
  assert all(p.requires_grad for p in params)


def test_get_gradient_without_backward_is_empty(framework):
  grads, idx_new_system = framework.get_gradient()

  assert len(grads) == 0
  assert idx_new_system[0] == 0
  assert len(idx_new_system) == len(framework.system_models) * framework.n_unfolds + 1


def test_get_gradient_after_backward_is_nonempty(framework, unfolds_batch):
  wn, x_sys, x_osc, y = unfolds_batch
  out = framework(wn, y, x_sys, x_osc)

  loss = sum(v.pow(2).sum() for v in out['z_sys'].values() if v is not None)
  loss.backward()

  grads, idx_new_system = framework.get_gradient()

  assert len(grads) > 0
