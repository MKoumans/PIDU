"""Filesystem locations of datasets, checkpoints and outputs.

All locations hang off `ROOT`: the `PIDU_ROOT` environment variable if set,
otherwise the repository checkout that contains this package (editable
install), otherwise the current working directory.
"""
import os
from pathlib import Path


def _find_root() -> Path:
  env = os.environ.get('PIDU_ROOT')
  if env:
    return Path(env).resolve()
  checkout = Path(__file__).resolve().parents[2]  # <root>/src/pidu/paths.py
  if (checkout / 'pyproject.toml').is_file():
    return checkout
  return Path.cwd()


ROOT = _find_root()
DATA_DIR = ROOT / 'data'
DATASETS_DIR = DATA_DIR / 'datasets'            # simulated datasets (<usecase>/<name>.npz)
LABELS_DIR = DATA_DIR / 'labels'                # label mapping workbooks
EXPERIMENTAL_DIR = DATA_DIR / 'experimental_data'
MODELS_DIR = ROOT / 'models'                    # trained checkpoints (usecase<N>/<TYPE>-tTT-II.pt)
OUTPUTS_DIR = ROOT / 'outputs'


def case_dir(usecase: int) -> Path:
  """Root of everything written for one use case: `PIDU_CASE_DIR` if set (paper scripts' --output),
  otherwise outputs/case<N>. A relative `PIDU_CASE_DIR` is taken against the working directory."""
  env = os.environ.get('PIDU_CASE_DIR')
  return Path(env).resolve() if env else OUTPUTS_DIR / f'case{usecase}'


def case_outputs_dir(usecase: int) -> Path:
  """Per-run test exports of training (`Trainer.test`) and LMA: <case_dir>/outputs."""
  return case_dir(usecase) / 'outputs'
