"""Shared helpers for the analysis package: paths, run-directory lookup and
DataFrame utilities."""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from pidu.paths import ROOT as PROJECT_DIR, case_dir, case_outputs_dir


def outputs_dir(usecase: int) -> Path:
    """Root output directory for a use case (outputs/case<usecase>, or PIDU_CASE_DIR)."""
    return case_dir(usecase)


def runs_dir(usecase: int) -> Path:
    """Per-run test exports of training and LMA (outputs/case<usecase>/outputs)."""
    return case_outputs_dir(usecase)


def figures_dir(usecase: int) -> Path:
    """Figure export directory for a use case; created if missing."""
    path = outputs_dir(usecase) / 'figures'
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_rank(path: Path) -> int:
    """Rank a run directory by its run id, e.g. 'loss_CNN_<run_id>_<seed>'."""
    match = re.search(r'_(\d+)_\d+$', path.name)
    return int(match.group(1)) if match else -1


def latest_run_dir(root: Path, prefix: str, pattern: str | None = None) -> Path | None:
    """Return the run directory under `root` matching `prefix` with the highest run id.

    Args:
        root: Directory containing the run folders.
        prefix: Glob prefix of the run folders (e.g. 'loss_CNN_').
        pattern: Optional regex the full folder name must match.
    """
    candidates = [p for p in root.glob(f'{prefix}*') if p.is_dir()]
    if pattern is not None:
        candidates = [p for p in candidates if re.match(pattern, p.name)]
    if not candidates:
        return None
    return max(candidates, key=lambda p: (run_rank(p), p.name))


def released_run_dir(usecase: int, model_type: str) -> Path:
    """Run folder of the released checkpoint (the one on the Hugging Face Hub, pidu.hub.RELEASED_SEEDS),
    e.g. loss_PIDU_0_001 for models/usecase2/PIDU-t01-00.pt."""
    from pidu.hub import RELEASED_SEEDS
    return runs_dir(usecase) / f'loss_{model_type}_{RELEASED_SEEDS[usecase][model_type]}_001'


def drop_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop pandas' auto-generated 'Unnamed: *' index columns."""
    unnamed = [c for c in df.columns if str(c).lower().startswith('unnamed')]
    return df.drop(columns=unnamed, errors='ignore')


def read_excel_clean(path: Path, sheet_name: str = 'system1') -> pd.DataFrame:
    """Read one sheet of an Excel workbook and drop unnamed index columns."""
    return drop_unnamed_columns(pd.read_excel(path, sheet_name=sheet_name))


def as_dataframe(value) -> pd.DataFrame:
    """Coerce a DataFrame, dict, ndarray or list to a DataFrame."""
    if isinstance(value, pd.DataFrame):
        return value
    if isinstance(value, dict):
        return pd.DataFrame([value])
    if isinstance(value, (np.ndarray, list)):
        return pd.DataFrame(value)
    raise ValueError(f'Cannot convert {type(value)} to DataFrame')
