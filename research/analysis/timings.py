"""Forward-pass timing benchmarks, MAPE collection and the LaTeX timing table."""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from pidu.data.simulation import SimulationDatasetBatch

from pidu.model.networks.io import load_checkpoint
from pidu.paths import DATASETS_DIR, MODELS_DIR

from .metrics import network_result, trf_result
from .utils import drop_unnamed_columns, released_run_dir, runs_dir

LMA_METHODS = ['loss_lma_baseline', 'loss_lma_noisy', 'loss_lma_random']
LEARNED_METHODS = {
    'CNN': ('loss_CNN_', r'^loss_CNN_\d+_\d+$'),
    'DU': ('loss_DU_', r'^loss_DU_\d+_\d+$'),
    'PIDU': ('loss_PIDU_', r'^loss_PIDU_\d+_\d+$'),
}
# Split the MAPE is computed on, the same for every method: the experimental set for use case 2,
# the simulated test set for use case 1. The networks export it as test_exp/test_test, LMA as
# test_<dataset name>; the first existing folder matching a pattern is used.
MAPE_SPLITS = {1: ['test_test', 'test_case1_test_*'], 2: ['test_exp', 'test_case2_experimental']}
# Ground truth of that split, needed for the oscillator precision of the networks.
EVAL_DATASETS = {1: 'usecase1/case1_test_n2_10_nwsn', 2: 'usecase2/case2_experimental'}
OSC_TOL = 0.2  # relative w0 tolerance of a correctly detected oscillator (as in Fig. 2)


def _timing_stats(times_ms: np.ndarray) -> dict[str, float]:
    """Statistics over the timed samples; NaN entries (failed LMA fits) are left out."""
    times_ms = np.asarray(times_ms, dtype=float)
    times_ms = times_ms[np.isfinite(times_ms)]
    if times_ms.size == 0:
        return dict.fromkeys(('mean_ms', 'std_ms', 'median_ms', 'min_ms', 'max_ms'), np.nan)
    return {
        'mean_ms': float(np.mean(times_ms)),
        'std_ms': float(np.std(times_ms)),
        'median_ms': float(np.median(times_ms)),
        'min_ms': float(np.min(times_ms)),
        'max_ms': float(np.max(times_ms)),
    }


def _mape_from_percentage_file(file_path: Path) -> float | None:
    """MAPE as the mean over non-zero per-column mean absolute percentage errors.

    NaN entries (samples without a value) count as 0 % error.
    """
    try:
        df = pd.read_excel(file_path)
    except Exception as exc:
        print(f'  Warning: could not read percentage residuals from {file_path}: {exc}')
        return None

    numeric = drop_unnamed_columns(df).apply(pd.to_numeric, errors='coerce').fillna(0.0)
    col_mape = numeric.abs().mean(axis=0)
    nonzero = col_mape[col_mape != 0.0]
    return float(nonzero.mean()) if not nonzero.empty else None


def _eval_split(run_dir: Path | None, usecase: int) -> Path | None:
    """Folder of the evaluation split (MAPE_SPLITS) of a run, or None."""
    if run_dir is None or not run_dir.exists():
        return None
    for pattern in MAPE_SPLITS[usecase]:
        for file_path in sorted(run_dir.glob(f'{pattern}/res_dis_percentage.xlsx')):
            return file_path.parent
    print(f'  Warning: no {" or ".join(MAPE_SPLITS[usecase])} residuals in {run_dir}; MAPE left empty')
    return None


def _precision(split: Path, usecase: int, learned: bool) -> float | None:
    """Oscillator precision TP / n_predicted (metrics.py); None without the ground-truth dataset."""
    dataset = DATASETS_DIR / f'{EVAL_DATASETS[usecase]}.npz'
    if learned and not dataset.exists():
        print(f'  Warning: {dataset} not found; precision left empty')
        return None
    r = network_result(split, dataset, OSC_TOL) if learned else trf_result(split, OSC_TOL)
    return r.tp / r.n_pred if r.n_pred else None


def collect_mape_by_method(usecase: int) -> dict[str, dict[str, float | None]]:
    """MAPE and oscillator precision per method on the same split: the LMA folders and the released
    run per learned model."""
    root = runs_dir(usecase)
    runs = {name: (root / name, False) for name in LMA_METHODS}
    runs.update({name: (released_run_dir(usecase, name), True) for name in LEARNED_METHODS})
    result = {}
    for name, (run_dir, learned) in runs.items():
        split = _eval_split(run_dir, usecase)
        if split is None:
            result[name] = {'mape': None, 'precision': None}
            continue
        print(f'  MAPE {run_dir.name}: {split.name}')
        result[name] = {'mape': _mape_from_percentage_file(split / 'res_dis_percentage.xlsx'),
                        'precision': _precision(split, usecase, learned)}
    return result


def _lma_timing_rows(usecase: int, method_mape: dict) -> tuple[list[dict], list[pd.DataFrame]]:
    """Read pre-saved LMA per-sample timing CSVs into summary rows and raw frames."""
    rows, raw_dfs = [], []

    for name in LMA_METHODS:
        csv_path = runs_dir(usecase) / name / 'timing_per_sample.csv'
        if not csv_path.exists():
            continue
        lma_df = pd.read_csv(csv_path)
        if 'total_time_ms' not in lma_df.columns:
            continue

        lma_df['source'] = name
        raw_dfs.append(lma_df)
        rows.append({
            'method': name,
            'batch_size': None,
            **method_mape.get(name, {}),
            'n_params': None,
            'model_size_mb': None,
            'n_samples': int(lma_df['total_time_ms'].notna().sum()),  # timed (not failed) fits
            **_timing_stats(lma_df['total_time_ms'].to_numpy()),
        })

    return rows, raw_dfs


def _time_model_forward(
    model,
    ds: SimulationDatasetBatch,
    batch_size: int,
    n_warmup: int,
    n_repeats: int,
    is_cuda: bool,
) -> np.ndarray:
    """Per-sample forward-pass times (ms) for one model at one batch size."""
    y_mode = ds.y_return_mode
    if len(ds) < batch_size:  # small sets (experimental): cycle the samples to fill the batch
        ds = Subset(ds, [i % len(ds) for i in range(batch_size)])
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    wn, x_sys, x_osc, y, _idx = next(iter(loader))
    if y_mode == 'both':
        y = {k: v[..., 1] for k, v in y.items()}  # select noised y values

    times_ms = []
    with torch.no_grad():
        for _ in range(n_warmup):
            model(wn=wn, y=y, x_sys=x_sys, x_osc=x_osc)
        if is_cuda:
            torch.cuda.synchronize()

        for _ in range(n_repeats):
            if is_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(wn=wn, y=y, x_sys=x_sys, x_osc=x_osc)
            if is_cuda:
                torch.cuda.synchronize()
            times_ms.append((time.perf_counter() - t0) * 1000 / batch_size)

    return np.array(times_ms)


def _learned_timing_rows(
    usecase: int,
    device: str,
    ds: SimulationDatasetBatch,
    method_mape: dict,
    n_warmup: int,
    n_repeats: int,
    batch_sizes: list[int],
) -> list[dict]:
    """Benchmark all saved model checkpoints at each batch size."""
    models_dir = MODELS_DIR / f'usecase{usecase}'
    is_cuda = str(device).startswith('cuda')
    rows = []

    for model_type in LEARNED_METHODS:
        for pt_path in sorted(models_dir.glob(f'{model_type}-t01-*.pt')):  # trail of the released runs
            try:
                model = load_checkpoint(pt_path, map_location=device)
            except Exception as exc:
                print(f'  Warning: could not load {pt_path.name}: {exc}')
                continue

            model.eval()
            n_params = sum(p.numel() for p in model.parameters())
            model_size_mb = round(
                sum(p.element_size() * p.numel() for p in model.parameters()) / (1024 ** 2), 3
            )

            for bs in batch_sizes:
                times_ms = _time_model_forward(model, ds, bs, n_warmup, n_repeats, is_cuda)
                rows.append({
                    'method': pt_path.stem,  # e.g. 'CNN-t10-00'
                    'batch_size': bs,
                    **method_mape.get(model_type, {}),
                    'n_params': n_params,
                    'model_size_mb': model_size_mb,
                    'n_samples': n_repeats,
                    **_timing_stats(times_ms),
                })

            print(f'Finished timing for {pt_path.stem} (params: {n_params}, size: {model_size_mb} MB)')

    return rows


def get_timings(
    usecase: int,
    device: str,
    ds: SimulationDatasetBatch,
    n_warmup: int = 3,
    n_repeats: int = 20,
    batch_sizes: list[int] | None = None,
) -> dict[str, pd.DataFrame]:
    """Benchmark forward-pass timing for all methods.

    LMA methods are read from pre-saved timing_per_sample.csv files; learned
    models (CNN/DU/PIDU) are loaded from models/usecase<N>/ and timed at each batch size.

    Returns:
        'timing_summary': one row per (method, batch_size).
        'lma_per_sample': raw LMA per-sample timings (when available).
    """
    if batch_sizes is None:
        batch_sizes = [1, 4, 16, 64, 256]

    method_mape = collect_mape_by_method(usecase)
    lma_rows, lma_dfs = _lma_timing_rows(usecase, method_mape)
    learned_rows = _learned_timing_rows(usecase, device, ds, method_mape, n_warmup, n_repeats, batch_sizes)

    out = {'timing_summary': pd.DataFrame(lma_rows + learned_rows)}
    if lma_dfs:
        out['lma_per_sample'] = pd.concat(lma_dfs, ignore_index=True)
    return out


def _format_mape(value) -> str:
    if value is None or pd.isna(value):
        return ''
    return f'{float(value):.2f}'.rstrip('0').rstrip('.')


def _format_time_std(mean_value, std_value) -> str:
    if mean_value is None or std_value is None or pd.isna(mean_value) or pd.isna(std_value):
        return ''
    return f'${float(mean_value):.2f} \\pm {float(std_value):.2f}$'


def _embolden(text: str) -> str:
    if text.startswith('$') and text.endswith('$'):
        return f'$\\mathbf{{{text[1:-1]}}}$'
    return f'\\textbf{{{text}}}'


def _lma_table_row(df: pd.DataFrame, key: str) -> tuple[str, str, str, str, float | None]:
    label = f"LMA ({key.split('_')[-1]})"
    subset = df[df['method'].str.contains(key, case=False, na=False)]
    if subset.empty:
        return (label, '', '', '', None)
    row = subset.iloc[0]
    mean_ms = None if pd.isna(row['mean_ms']) else float(row['mean_ms'])
    return (label, _format_mape(row.get('mape')), _format_mape(row.get('precision')),
            _format_time_std(row['mean_ms'], row['std_ms']), mean_ms)


def _model_table_row(df: pd.DataFrame, model: str, batch_size: int) -> tuple[str, str, str, str, float | None]:
    """Aggregate all checkpoints of a model family at one batch size."""
    label = f'{model} ($b={batch_size}$)'
    mask = df['method'].str.startswith(model, na=False)
    mask &= pd.to_numeric(df['batch_size'], errors='coerce') == batch_size
    subset = df[mask]
    if subset.empty:
        return (label, '', '', '', None)
    mean_ms = subset['mean_ms'].mean()
    std_ms = subset['std_ms'].mean()
    mape, precision = (subset[c].dropna().mean() if c in subset.columns else np.nan for c in ('mape', 'precision'))
    return (
        label,
        _format_mape(mape),
        _format_mape(precision),
        _format_time_std(mean_ms, std_ms),
        None if pd.isna(mean_ms) else float(mean_ms),
    )


def timings_to_latex_table(
    timing_summary: pd.DataFrame,
    caption: str = 'Performance comparison.',
    label: str = 'tab:timings_case1',
) -> str:
    """Render the timing summary as a LaTeX table (method, MAPE, oscillator precision, time ± std).

    The fastest learned model at batch size 1 is highlighted in bold.
    """
    df = timing_summary
    rows = [
        _lma_table_row(df, 'lma_baseline'),
        _lma_table_row(df, 'lma_noisy'),
        _lma_table_row(df, 'lma_random'),
        _model_table_row(df, 'CNN', 1),
        _model_table_row(df, 'DU', 1),
        _model_table_row(df, 'PIDU', 1),
        None,  # midrule
        _model_table_row(df, 'CNN', 256),
        _model_table_row(df, 'DU', 256),
        _model_table_row(df, 'PIDU', 256),
    ]

    b1_labels = {'CNN ($b=1$)', 'DU ($b=1$)', 'PIDU ($b=1$)'}
    b1_rows = [r for r in rows if r is not None and r[0] in b1_labels and r[4] is not None]
    best_b1 = min(b1_rows, key=lambda r: r[4])[0] if b1_rows else None

    lines = [
        r'\begin{table}[t]',
        r'\centering',
        r'\small',
        f'\\caption{{\\textbf{{{caption}}}}}',
        f'\\label{{{label}}}',
        r'\begin{tabular}{rlll}',
        r'\toprule',
        r'\textbf{Method} & \textbf{MAPE} & \textbf{Precision} & \textbf{Time $\pm$ Std (ms)} \\',
        r'\midrule',
    ]

    for row in rows:
        if row is None:
            lines.append(r'\cmidrule{1-4}')
            continue
        name, mape, precision, time_std, _mean = row
        if name == best_b1 and time_std:
            name, time_std = _embolden(name), _embolden(time_std.strip())
        lines.append(f'{name} & {mape} & {precision} & {time_std} \\\\')

    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines) + '\n'
