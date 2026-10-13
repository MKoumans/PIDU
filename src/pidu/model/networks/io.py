"""Save/load a full MultiModelFramework (or CNNBasic) checkpoint.

Checkpoints are whole pickled modules (`torch.save(model)`). Checkpoints written
before the `src/pidu` restructure reference the old top-level modules
(`model.networks.*`, `data.Physics`, ...); `load_checkpoint` maps those names
onto the `pidu` package so both old and new files load.
"""
import os
import pickle
import types

import torch

from pidu.paths import MODELS_DIR

MODEL_DIR = str(MODELS_DIR)

# Old module path -> new module path (prefix match, longest first).
LEGACY_MODULES = {
  'data.Physics': 'pidu.data.physics',
  'data.Simulation': 'pidu.data.simulation',
  'data.DataImporter': 'pidu.data.importer',
  'data.models': 'pidu.data.optical_parameters',
  'data.tmm_core': 'pidu.data.tmm_core',
  'data.utils': 'pidu.data.utils',
  'model.Trainers': 'pidu.model.trainer',
  'model.Optimisation': 'pidu.model.optimisation',
  'model.Initializer': 'pidu.model.initializer',
  'model.utils': 'pidu.model.utils',
  'model.networks': 'pidu.model.networks',
  'config': 'pidu.config',
}


def _rename(module: str) -> str:
  for old in sorted(LEGACY_MODULES, key=len, reverse=True):
    if module == old or module.startswith(old + '.'):
      return LEGACY_MODULES[old] + module[len(old):]
  return module


class _LegacyUnpickler(pickle.Unpickler):
  def find_class(self, module, name):
    return super().find_class(_rename(module), name)


legacy_pickle = types.ModuleType('pidu_legacy_pickle')
legacy_pickle.Unpickler = _LegacyUnpickler
legacy_pickle.load = lambda f, **kw: _LegacyUnpickler(f, **kw).load()
legacy_pickle.__dict__.update({k: getattr(pickle, k) for k in ('dump', 'dumps', 'loads', 'HIGHEST_PROTOCOL')})


def _retarget_devices(obj, device: torch.device, seen: set):
  """Replace every stored `torch.device` (e.g. `dv`, a physics layer's `device`) reachable from `obj`."""
  if id(obj) in seen or isinstance(obj, (torch.Tensor, str, bytes, int, float, type(None))):
    return
  seen.add(id(obj))
  if isinstance(obj, dict):
    container = obj
  elif isinstance(obj, (list, tuple)):
    for item in obj:
      _retarget_devices(item, device, seen)
    return
  elif hasattr(obj, '__dict__') and not isinstance(obj, type):
    container = vars(obj)
  else:
    return
  for key, value in list(container.items()):
    if isinstance(value, torch.device):
      container[key] = device
    else:
      _retarget_devices(value, device, seen)


def load_checkpoint(path, map_location=None) -> torch.nn.Module:
  """Load a full-model checkpoint, including ones saved with the old module layout.

  With `map_location`, the device attributes stored on the model (used to create
  tensors in the forward pass) are retargeted too, so a CUDA checkpoint runs on CPU.
  """
  model = torch.load(path, map_location=map_location, weights_only=False, pickle_module=legacy_pickle)
  if map_location is not None:
    _retarget_devices(model, torch.device(map_location), set())
  return model


def save_pt_model(model: torch.nn.Module, name: str):
  os.makedirs(os.path.join(MODEL_DIR, 'full_model', name.split('/')[0]), exist_ok=True)
  torch.save(model, os.path.join(MODEL_DIR, 'full_model', f"{name}.pt"))


def load_pt_model(name: str | None = None) -> torch.nn.Module | None:
  if name is None:
    return None
  return load_checkpoint(os.path.join(MODEL_DIR, 'full_model', f"{name}.pt"))
