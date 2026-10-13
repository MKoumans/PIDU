"""Parameter tracking across methods (use case 2) and unfold-loss comparison."""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .datasets import extract_gt_params, load_dataset
from .timings import LEARNED_METHODS
from .utils import PROJECT_DIR, figures_dir, read_excel_clean, released_run_dir, runs_dir

OSC_PARAM = re.compile(r'^(w0|wp|g)_(\d+)$')
OSC_MATCH_TOL = 0.5  # a predicted oscillator is paired with a GT one only if |dw0| <= tol * w0_GT


def _oscillators(df: pd.DataFrame) -> np.ndarray:
    """Oscillator triplets [n_samples, n_oscillators, (w0, wp, g)] from the w0_<i>/wp_<i>/g_<i> columns.

    The numbering differs per source (GT: w0_1.. by frequency; networks: w0_00.. by detection rank;
    LMA: w0_00.. per detection cell), so oscillators are paired by frequency, not by column name.
    """
    ids = [m.group(2) for c in df.columns if (m := OSC_PARAM.match(str(c))) and m.group(1) == 'w0']
    ids = [i for i in ids if f'wp_{i}' in df.columns and f'g_{i}' in df.columns]
    cols = [[f'{a}_{i}' for a in ('w0', 'wp', 'g')] for i in ids]
    return np.stack([df[c].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float) for c in cols], axis=1)


def match_oscillator(gt_df: pd.DataFrame, method_df: pd.DataFrame, gt_index: int) -> np.ndarray:
    """Per sample, the (w0, wp, g) of the predicted oscillator nearest in w0 to GT oscillator `gt_index`
    (GT column w0_<gt_index>); NaN where no prediction lies within OSC_MATCH_TOL."""
    target = gt_df[f'w0_{gt_index}'].to_numpy(dtype=float)
    pred = _oscillators(method_df)[: len(target)]
    w0 = np.where(np.isfinite(pred[..., 0]) & (pred[..., 0] != 0), pred[..., 0], np.nan)
    dist = np.abs(w0 - target[: len(w0), None])
    out = np.full((len(w0), 3), np.nan)
    for s in range(len(w0)):
        if np.isfinite(dist[s]).any():
            k = np.nanargmin(dist[s])
            if dist[s, k] <= OSC_MATCH_TOL * abs(target[s]):
                out[s] = pred[s, k]
    return out


def _param_values(gt_df: pd.DataFrame, df: pd.DataFrame, param: str, n: int) -> np.ndarray:
    """Values of one parameter: system parameters by column, oscillators (w0_1, ...) by frequency."""
    m = OSC_PARAM.match(param)
    if m is None:
        return pd.to_numeric(df[param], errors='coerce').to_numpy(dtype=float)[:n]
    if df is gt_df:
        return pd.to_numeric(gt_df[param], errors='coerce').to_numpy(dtype=float)[:n]
    return match_oscillator(gt_df, df, int(m.group(2)))[:n, ('w0', 'wp', 'g').index(m.group(1))]


def _load_released_last_unfold(usecase: int, model_type: str, split_name: str, sheet_name: str) -> pd.DataFrame:
    """Load the last_unfold workbook of the released run (the checkpoint on the Hugging Face Hub)."""
    run_dir = released_run_dir(usecase, model_type)
    file_path = run_dir / split_name / 'last_unfold.xlsx'
    if not file_path.exists():
        raise FileNotFoundError(f'Missing last_unfold file of the released {model_type} run: {file_path}')

    print(f'Using released {model_type} run: {run_dir.name} ({split_name})')
    return read_excel_clean(file_path, sheet_name=sheet_name)


def _lma_exp_file(run_dir: Path) -> Path:
    """last_unfold.xlsx of an LMA run on the experimental set. LMA names its split after the dataset
    (test_case2_experimental); test_exp is the name used by the training runs."""
    for split in ('test_exp', 'test_case2_experimental'):
        if (run_dir / split / 'last_unfold.xlsx').exists():
            return run_dir / split / 'last_unfold.xlsx'
    raise FileNotFoundError(f'No experimental LMA results in {run_dir} (expected test_exp/ or test_case2_experimental/); '
                            f'run research/lma/run_all_schemes.py --dataset usecase2/case2_experimental')


def compute_visible_ylim(
    sub_df: pd.DataFrame,
    value_cols: list[str],
    anchor_col: str = 'GT',
    q_low: float = 0.02,
    q_high: float = 0.98,
    padding_ratio: float = 0.08,
) -> tuple[float, float] | None:
    """Robust y-limits: quantile range of all series, always covering the anchor column."""
    values = [
        vals[np.isfinite(vals)]
        for col in value_cols if col in sub_df.columns
        for vals in [pd.to_numeric(sub_df[col], errors='coerce').to_numpy(dtype=float)]
    ]
    values = [v for v in values if v.size]
    if not values:
        return None

    stacked = np.concatenate(values)
    low = float(np.nanquantile(stacked, q_low))
    high = float(np.nanquantile(stacked, q_high))

    if anchor_col in sub_df.columns:
        anchor = pd.to_numeric(sub_df[anchor_col], errors='coerce').to_numpy(dtype=float)
        anchor = anchor[np.isfinite(anchor)]
        if anchor.size:
            low = min(low, float(np.min(anchor)))
            high = max(high, float(np.max(anchor)))

    span = high - low
    if span <= 0:
        pad = max(abs(low), 1.0) * 0.05
    else:
        pad = span * padding_ratio
    return (low - pad, high + pad)


def _plot_parameter_tracking(tracking_df: pd.DataFrame, params: list[str], plot_cols: list[str], plot_path: Path):
    fig, axes = plt.subplots(3, 3, figsize=(20, 12), sharex=True)
    axes = axes.flatten()

    for idx, param in enumerate(params):
        ax = axes[idx]
        sub = tracking_df[tracking_df['parameter'] == param]
        for col in plot_cols:
            ax.plot(sub['sample_idx'].to_numpy(), sub[col].to_numpy(), label=col, linewidth=1.2, alpha=0.9)

        ylim = compute_visible_ylim(sub, plot_cols)
        if ylim is not None:
            ax.set_ylim(*ylim)

        ax.set_title(param)
        ax.grid(alpha=0.25)
        if idx % 3 == 0:
            ax.set_ylabel('Estimated value')
        if idx >= 6:
            ax.set_xlabel('Sample index')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle('Usecase 2 Parameter Tracking: GT vs Methods', fontsize=15, y=0.995)
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.965), ncol=7, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f'Saved parameter tracking plot to {plot_path}')


def get_usecase2_parameter_tracking(device: str, sheet_name: str = 'system1') -> dict[str, pd.DataFrame]:
    """GT vs. per-method oscillator parameter comparison for use case 2.

    Saves a 3x3 tracking plot to outputs/case2/figures/ and returns one
    DataFrame per parameter with columns: sample_idx, parameter, GT, <methods>.
    """
    target_params = [
        'w0_before', 'wp_before', 'g_before',
        'w0_1', 'wp_1', 'g_1',
        'w0_2', 'wp_2', 'g_2',
    ]
    root = runs_dir(2)

    gt_ds = load_dataset(filename='usecase2/case2_experimental', device=device, y_return_mode='both')
    gt_df = extract_gt_params(gt_ds, system_name=sheet_name)

    method_frames = {
        f"LMA({name.split('_')[-1]})": read_excel_clean(_lma_exp_file(root / name), sheet_name=sheet_name)
        for name in ['loss_lma_baseline', 'loss_lma_random', 'loss_lma_noisy']
    }
    method_frames |= {
        model: _load_released_last_unfold(2, model, 'test_exp', sheet_name)
        for model in LEARNED_METHODS
    }

    n_samples = min(len(gt_df), *(len(df) for df in method_frames.values()))

    per_param_frames = []
    for param in target_params:
        frame = {
            'sample_idx': np.arange(n_samples),
            'parameter': [param] * n_samples,
            'GT': _param_values(gt_df, gt_df, param, n_samples),
        }
        for method_name, method_df in method_frames.items():
            frame[method_name] = _param_values(gt_df, method_df, param, n_samples)
        per_param_frames.append(pd.DataFrame(frame))

    tracking_df = pd.concat(per_param_frames, ignore_index=True)

    plot_cols = ['GT'] + list(method_frames)
    _plot_parameter_tracking(tracking_df, target_params, plot_cols, figures_dir(2) / 'usecase2_parameter_tracking.png')

    return {param: df for param, df in tracking_df.groupby('parameter')}


def get_unfolds_loss_comparison() -> dict[str, pd.DataFrame]:
    """Summarise the last-5-epoch losses per unfold count from the pre-saved
    train/val unfold analysis workbooks in outputs/."""
    root = PROJECT_DIR / 'outputs'
    workbooks = {
        'train': pd.read_excel(root / 'analysis_tables_train_unfolds.xlsx', sheet_name=None),
        'val': pd.read_excel(root / 'analysis_tables_val_unfolds.xlsx', sheet_name=None),
    }

    out = {}
    for split, sheets in workbooks.items():
        rows = []
        for sheet_name, df in sheets.items():
            unfold = sheet_name.split('-')[-1]
            last5 = df.tail(5)
            rows.append([unfold, last5['mean_loss'].mean(), last5['min_loss'].mean(), last5['min_loss'].var()])
        out[split] = pd.DataFrame(rows, columns=['unfold', 'mean_last5', 'min_last5', 'var_last5'])

    return out
