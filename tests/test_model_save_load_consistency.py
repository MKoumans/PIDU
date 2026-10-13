"""
Integration tests: does a MultiModelFramework survive a save/load round trip
with identical parameters and identical forward-pass outputs?

compare_state_dicts / compare_forward_outputs are kept as reusable helpers
(the only genuinely useful part of the old tests/test_save_load_consistency.py)
but are now wired into real assert-based pytest tests instead of a
hand-run CLI diagnostic script, and use tmp_path instead of writing
checkpoints into the working directory.
"""
from typing import Callable, Dict, Tuple

import pytest
import torch
import torch.nn as nn

from pidu.model.networks import io as io_module
from pidu.model.networks.framework import MultiModelFramework
from pidu.model.networks.io import load_pt_model, save_pt_model

pytestmark = pytest.mark.slow

ACTIVE_MODELS = ['system1', 'system2_0', 'system2_1']


def compare_state_dicts(
  state_dict_1: Dict,
  state_dict_2: Dict,
  tolerance: float = 1e-5,
) -> Tuple[bool, float]:
  """Compare two state dicts: same keys, same shapes, values within tolerance."""
  keys_1, keys_2 = set(state_dict_1.keys()), set(state_dict_2.keys())
  if keys_1 != keys_2:
    return False, float('inf')

  all_match = True
  max_diff = 0.0
  for key in keys_1:
    param_1, param_2 = state_dict_1[key], state_dict_2[key]
    if param_1.shape != param_2.shape:
      all_match = False
      continue
    diff = torch.abs(param_1 - param_2).max().item()
    max_diff = max(max_diff, diff)
    if diff > tolerance:
      all_match = False

  return all_match, max_diff


def compare_forward_outputs(
  model_1: nn.Module,
  model_2: nn.Module,
  forward_fn: Callable,
  tolerance: float = 1e-5,
) -> Tuple[bool, float]:
  """Compare one forward pass of two models under the same input, per nested output key."""
  model_1.eval()
  model_2.eval()

  all_match = True
  max_output_diff = 0.0

  with torch.no_grad():
    output_1 = forward_fn(model_1)
    output_2 = forward_fn(model_2)

    def compare(out_1, out_2):
      """Compare tensors; recurse into dicts and lists (entries may be None); returns (match, max diff)."""
      if isinstance(out_1, dict):
        results = [compare(out_1[k], out_2[k]) for k in out_1.keys()]
      elif isinstance(out_1, (list, tuple)):
        results = [compare(a, b) for a, b in zip(out_1, out_2)]
      elif out_1 is None or out_2 is None:
        return out_1 is out_2, 0.0
      elif out_1.shape != out_2.shape:
        return False, 0.0
      else:
        diff = torch.abs(out_1 - out_2).max().item()
        return diff <= tolerance, diff
      return all(m for m, _ in results), max((d for _, d in results), default=0.0)

    all_match, max_output_diff = compare(output_1, output_2)

  return all_match, max_output_diff


@pytest.fixture
def framework(system_info, model_info_builder, device, dtype):
  model_info = model_info_builder(ACTIVE_MODELS, physics=False, unfolds=1)
  return MultiModelFramework(
    n_unfolds=1,
    system_info=system_info,
    model_info=model_info,
    dtype=dtype,
    device=device,
    verbose=0,
  )


def test_state_dict_survives_save_and_load(tmp_path, monkeypatch, framework):
  monkeypatch.setattr(io_module, "MODEL_DIR", str(tmp_path))

  save_pt_model(framework, "consistency_state_dict")
  loaded = load_pt_model("consistency_state_dict")

  all_match, max_diff = compare_state_dicts(framework.state_dict(), loaded.state_dict(), tolerance=0)
  assert all_match, f"state dicts differ after save/load, max diff={max_diff}"


def test_forward_output_survives_save_and_load(tmp_path, monkeypatch, framework, unfolds_batch):
  monkeypatch.setattr(io_module, "MODEL_DIR", str(tmp_path))
  wn, x_sys, x_osc, y = unfolds_batch

  save_pt_model(framework, "consistency_forward_output")
  loaded = load_pt_model("consistency_forward_output")

  run = lambda m: m(wn, y, x_sys, x_osc)
  all_match, max_diff = compare_forward_outputs(framework, loaded, run, tolerance=1e-5)
  assert all_match, f"forward outputs differ after save/load, max diff={max_diff}"


def test_compare_state_dicts_detects_a_real_difference():
  a = {'w': torch.zeros(3)}
  b = {'w': torch.ones(3)}
  all_match, max_diff = compare_state_dicts(a, b, tolerance=1e-5)
  assert not all_match
  assert max_diff == pytest.approx(1.0)


def test_compare_state_dicts_detects_key_mismatch():
  a = {'w': torch.zeros(3)}
  b = {'v': torch.zeros(3)}
  all_match, max_diff = compare_state_dicts(a, b)
  assert not all_match
  assert max_diff == float('inf')
