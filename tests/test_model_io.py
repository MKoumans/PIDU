"""
Unit tests for model/networks/io.py::save_pt_model / load_pt_model.

Uses a plain nn.Linear instead of a full MultiModelFramework/CNNBasic -
save/load is agnostic to what nn.Module it's given, so there's no reason to
pay simulated-dataset construction cost here.
"""
import torch
import torch.nn as nn

from pidu.model.networks import io as io_module
from pidu.model.networks.io import load_pt_model, save_pt_model


def test_save_then_load_round_trips_state_dict(tmp_path, monkeypatch):
  monkeypatch.setattr(io_module, "MODEL_DIR", str(tmp_path))
  model = nn.Linear(4, 3)

  save_pt_model(model, "roundtrip_test")
  loaded = load_pt_model("roundtrip_test")

  assert isinstance(loaded, nn.Linear)
  for key, value in model.state_dict().items():
    assert torch.equal(value, loaded.state_dict()[key])


def test_save_creates_nested_directory(tmp_path, monkeypatch):
  monkeypatch.setattr(io_module, "MODEL_DIR", str(tmp_path))
  model = nn.Linear(2, 2)

  save_pt_model(model, "trailA/nested_test")

  assert (tmp_path / "full_model" / "trailA" / "nested_test.pt").exists()


def test_load_returns_none_for_none_name():
  assert load_pt_model(None) is None
