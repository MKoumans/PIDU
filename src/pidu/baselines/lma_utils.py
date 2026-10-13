"""Shared utilities for the LMA optimisation pipeline."""
from __future__ import annotations

import datetime
import json
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment

log = logging.getLogger(__name__)

# ── type / dtype helpers ─────────────────────────────────────────────────────

_TORCH_TO_NP = {torch.float16: np.float16, torch.float32: np.float32, torch.float64: np.float64}


def to_np_dtype(dtype: Any) -> np.dtype:
    if isinstance(dtype, torch.dtype):
        return np.dtype(_TORCH_TO_NP.get(dtype, np.float64))
    if isinstance(dtype, (np.dtype, type)):
        return np.dtype(dtype)
    return np.dtype(np.float64)


def as_tensor(x: Any) -> torch.Tensor:
    return x if isinstance(x, torch.Tensor) else torch.as_tensor(x)


# ── parameter names ──────────────────────────────────────────────────────────

_SYS_PARAM_NAMES = [
    "th0", "d", "eps_inf", "drude_p", "drude_g",
    "w0_before", "wp_before", "g_before",
    "w0_after", "wp_after", "g_after",
]


def param_names(n: int) -> List[str]:
    """Generate parameter column names for *n* total parameters."""
    names = list(_SYS_PARAM_NAMES)
    for i in range((max(0, n - len(names))) // 3):
        names += [f"w0_{i:02d}", f"wp_{i:02d}", f"g_{i:02d}"]
    while len(names) < n:
        names.append(f"param_{len(names)}")
    return names[:n]


# ── oscillator post-processing (Hungarian matching) ──────────────────────────

def postprocess_oscillators(
    predicted: np.ndarray,
    truth: np.ndarray,
    *,
    enabled: bool = True,
    fixed_prefix: int = 11,
    suppress_non_osc: bool = False,
) -> Dict[str, Any]:
    """Reorder predicted oscillator triplets to best match ground-truth ordering.

    Uses the Hungarian algorithm on a relative-cost matrix.
    Returns a dict with 'params' (reordered), 'changed', 'n_triplets',
    'assignment', 'mae_before', 'mae_after'.
    """
    pred = np.asarray(predicted, dtype=np.float64).ravel()
    gt = np.asarray(truth, dtype=np.float64).ravel()
    null = dict(enabled=enabled, changed=False, n_triplets=0,
                assignment=np.array([], int), mae_before=0.0, mae_after=0.0,
                params=pred.copy())

    if not enabled or pred.shape != gt.shape:
        return null
    tail = pred.size - fixed_prefix
    if tail <= 0 or tail % 3 != 0:
        return null

    n = tail // 3
    p_trips = pred[fixed_prefix:].reshape(n, 3)
    t_trips = gt[fixed_prefix:].reshape(n, 3)

    cost = np.mean(
        np.abs(p_trips[:, None] - t_trips[None]) / (np.abs(t_trips[None]) + 1e-12),
        axis=2,
    )
    rows, cols = linear_sum_assignment(cost)

    reordered = np.zeros_like(p_trips)
    for r, c in zip(rows, cols):
        reordered[c] = p_trips[r]

    if suppress_non_osc:
        for i in range(n):
            if np.allclose(t_trips[i], 0, atol=1e-6):
                reordered[i] = 0.0

    assignment = np.full(n, -1, int)
    assignment[rows] = cols

    return dict(
        enabled=True,
        changed=bool(np.any(assignment != np.arange(n))),
        n_triplets=n,
        assignment=assignment,
        mae_before=float(np.mean(np.abs(p_trips - t_trips))),
        mae_after=float(np.mean(np.abs(reordered - t_trips))),
        params=np.concatenate([pred[:fixed_prefix], reordered.ravel()]),
    )


# ── I/O helpers ──────────────────────────────────────────────────────────────

def export_outputs(
    output: np.ndarray,
    res_dis: np.ndarray,
    res_pct: np.ndarray,
    sample_indices: List[int],
    output_dir: str,
    split: str,
    system: str = "system1",
    gt: Optional[np.ndarray] = None,
) -> None:
    """Write parameter predictions and residuals to Excel files."""
    split_dir = os.path.join(output_dir, split)
    os.makedirs(split_dir, exist_ok=True)
    cols = param_names(output.shape[1])
    index = [f"sample_{i:03d}" for i in sample_indices]

    files = [
        ("output.xlsx", output),
        ("last_unfold.xlsx", output),
        ("res_dis.xlsx", res_dis),
        ("res_dis_percentage.xlsx", res_pct),
    ]
    if gt is not None:
        files.append(("gt.xlsx", gt))
    for fname, data in files:
        with pd.ExcelWriter(os.path.join(split_dir, fname)) as w:
            pd.DataFrame(data, columns=cols, index=index).to_excel(w, sheet_name=system)
    log.info("Outputs exported to %s", split_dir)


def save_timing(sample_timings: List[Dict[str, Any]], save_dir: str) -> None:
    """Write per-sample timing CSV and summary JSON."""
    if not sample_timings:
        return
    os.makedirs(save_dir, exist_ok=True)
    rows = [{"sample_idx": s["sample_idx"], "total_time_ms": s["total_time"] * 1e3}
            for s in sample_timings]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(save_dir, "timing_per_sample.csv"), index=False)
    agg = {k: float(df["total_time_ms"].agg(k)) for k in ("mean", "std", "min", "max")}
    with open(os.path.join(save_dir, "timing_summary.json"), "w") as f:
        json.dump(agg, f, indent=2)
    log.info("Timing saved to %s", save_dir)


def save_run_summary(results: Dict[str, Any], successes: List[bool], save_dir: str) -> None:
    """Write a human-readable run_summary.txt."""
    os.makedirs(save_dir, exist_ok=True)

    n = results["num_samples"]
    n_ok = int(sum(successes))
    times_arr = np.array([t["total_time"] for t in results.get("sample_timings", [])])
    n_crashed = int(np.sum(np.isnan(times_arr)))
    n_diverged = n - n_ok - n_crashed

    pct = lambda v: f"{100.0 * v / max(n, 1):.1f} %"

    lines = [
        "=" * 60, "LMA Run Summary",
        f"  Generated : {datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
        "=" * 60, "",
        "[Configuration]",
        f"  Scheme          : {results.get('initialization_scheme', '?')}",
        f"  Backend         : {results.get('backend', '?')}",
        f"  Max iterations  : {results.get('max_iterations', '?')}",
        f"  Noise factor    : {results.get('noise_factor', '?')}",
        "",
        "[Samples]",
        f"  Total           : {n}",
        f"  Converged       : {n_ok}  ({pct(n_ok)})",
        f"  Failed (no conv): {n_diverged}  ({pct(n_diverged)})",
        f"  Crashed         : {n_crashed}  ({pct(n_crashed)})",
        "",
        "[Timing  (ms/sample)]",
        f"  Mean            : {results['time_per_sample_ms']:.2f}",
        f"  Std             : {results['wall_time_std'] * 1e3:.2f}",
        f"  Throughput      : {results['throughput']:.1f} samples/s",
        "",
        "[Cost]",
        f"  Mean            : {results['cost_mean']:.4e}",
        f"  Mean iterations : {results['iterations_mean']:.1f}",
        "",
        "[Relative error (fraction)]",
    ]
    rel_mean = results.get("rel_error_mean")
    if rel_mean is not None:
        names = param_names(len(rel_mean))
        rel_std = results.get("rel_error_std", [float("nan")] * len(rel_mean))
        for nm, m, s in zip(names, rel_mean, rel_std):
            lines.append(f"  {nm:<18} mean={float(m):.4f}  std={float(s):.4f}")
        lines.append(f"  {'OVERALL':<18} mean={float(np.mean(rel_mean)):.4f}")
    lines += ["", "=" * 60]

    with open(os.path.join(save_dir, "run_summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    log.info("Run summary: %s", save_dir)
