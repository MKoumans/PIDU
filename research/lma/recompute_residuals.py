"""
Recompute res_dis and res_dis_percentage from last_unfold.xlsx + gt.xlsx.

For each sample the oscillator triplets are re-matched to the ground-truth
triplets with the Hungarian algorithm (minimising relative cost), which
corrects any ordering artefacts from the original LMA post-processing.

Samples are flagged as **badly converged** when either:
  • any predicted value is negative for a parameter whose GT value is > 0
  • any per-parameter percentage error exceeds `--pct-threshold` (default 100 %)

Summary statistics are reported for *well-converged* samples only.
All samples (including bad ones) are written to the output files so the
full row structure is preserved; bad-sample rows contain NaN.

Usage examples
--------------
  python recompute_residuals.py \\
      --pred  outputs/case2/loss_lma_noisy/test_exp/last_unfold.xlsx \\
      --gt    outputs/case2/loss_lma_noisy/test_exp/gt.xlsx

  python recompute_residuals.py \\
      --pred  path/to/last_unfold.xlsx \\
      --gt    path/to/gt.xlsx \\
      --fixed-prefix 11 \\
      --pct-threshold 1.0 \\
      --out-dir path/to/output/
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


from pidu.baselines.lma_utils import param_names as _param_names, postprocess_oscillators


# ─────────────────────────────────────────────────────────────────────────────
# Bad-convergence detection
# ─────────────────────────────────────────────────────────────────────────────

def _bad_convergence_reasons(
    pred:          np.ndarray,
    gt:            np.ndarray,
    fixed_prefix:  int,
    col_names:     List[str],
    pct_threshold: float,
) -> List[str]:
    """Return a list of reason strings; empty list means well-converged."""
    reasons: List[str] = []

    for i in range(len(pred)):
        gt_val   = gt[i]
        pred_val = pred[i]

        # skip parameters that are inactive (zero or sentinel -1) in GT
        if gt_val <= 0:
            continue

        name = col_names[i] if i < len(col_names) else f"col_{i}"

        # negative predicted value for a physically positive parameter
        if pred_val < 0:
            reasons.append(
                f"  {name}: predicted={pred_val:.4f} is negative (gt={gt_val:.4f})"
            )

        # percentage error > threshold
        pct = abs(pred_val - gt_val) / (abs(gt_val) + 1e-10)
        if pct > pct_threshold:
            reasons.append(
                f"  {name}: {pct * 100:.1f}% error "
                f"(predicted={pred_val:.4f}, gt={gt_val:.4f})"
            )

    return reasons


# ─────────────────────────────────────────────────────────────────────────────
# Main processing
# ─────────────────────────────────────────────────────────────────────────────

def recompute(
    pred_path:     str,
    gt_path:       str,
    fixed_prefix:  int   = 11,
    pct_threshold: float = 1.0,
    out_dir:       Optional[str] = None,
) -> None:

    # ── load files ────────────────────────────────────────────────────────────
    pred_df = pd.read_excel(pred_path, sheet_name=0, index_col=0)
    gt_df   = pd.read_excel(gt_path,   sheet_name=0, index_col=0)

    if pred_df.shape != gt_df.shape:
        raise ValueError(
            f"Shape mismatch: last_unfold {pred_df.shape} vs gt {gt_df.shape}. "
            "Both files must have the same rows and columns."
        )

    col_names = list(pred_df.columns)
    n_cols    = len(col_names)
    n_samples = len(pred_df)
    n_osc     = (n_cols - fixed_prefix) // 3

    print(f"\n{'='*60}")
    print(f"  last_unfold : {pred_path}")
    print(f"  gt          : {gt_path}")
    print(f"  samples     : {n_samples}")
    print(f"  params      : {n_cols}  (prefix={fixed_prefix}, osc_triplets={n_osc})")
    print(f"  pct threshold : {pct_threshold * 100:.0f}%")
    print(f"{'='*60}\n")

    pred_arr = pred_df.to_numpy(dtype=float)
    gt_arr   = gt_df.to_numpy(dtype=float)

    out_res:     List[np.ndarray] = []
    out_res_pct: List[np.ndarray] = []
    out_pred:    List[np.ndarray] = []

    bad_indices: List[str]       = []
    bad_details: Dict[str, List[str]] = {}

    n_crashed      = 0
    n_bad          = 0
    n_good         = 0
    good_abs_errs:  List[np.ndarray] = []
    good_pct_errs:  List[np.ndarray] = []

    for i, sample_name in enumerate(pred_df.index):
        pred_row = pred_arr[i]
        gt_row   = gt_arr[i]

        # crashed sample: all NaN
        if np.all(np.isnan(pred_row)):
            n_crashed += 1
            nan_row = np.full(n_cols, np.nan)
            out_pred.append(nan_row)
            out_res.append(nan_row)
            out_res_pct.append(nan_row)
            bad_indices.append(sample_name)
            bad_details[sample_name] = ["  [CRASHED — all NaN]"]
            continue

        # re-match oscillators
        pp = postprocess_oscillators(pred_row, gt_row, fixed_prefix=fixed_prefix)
        matched_pred = pp["params"]

        # compute errors
        abs_err = np.abs(matched_pred - gt_row)
        pct_err = abs_err / (np.abs(gt_row) + 1e-10)

        # bad-convergence check
        reasons = _bad_convergence_reasons(
            matched_pred, gt_row, fixed_prefix, col_names, pct_threshold
        )

        if reasons:
            n_bad += 1
            bad_indices.append(sample_name)
            bad_details[sample_name] = reasons
            nan_row = np.full(n_cols, np.nan)
            out_pred.append(matched_pred)   # keep prediction for inspection
            out_res.append(nan_row)         # NaN so it's excluded from statistics
            out_res_pct.append(nan_row)
        else:
            n_good += 1
            good_abs_errs.append(abs_err)
            good_pct_errs.append(pct_err)
            out_pred.append(matched_pred)
            out_res.append(abs_err)
            out_res_pct.append(pct_err)

    # ── print badly-converged details ─────────────────────────────────────────
    total_bad = n_crashed + n_bad
    print(f"{'─'*60}")
    print(f"  Converged samples   : {n_good} / {n_samples}")
    print(f"  Badly converged     : {n_bad}  (negative value or >{pct_threshold*100:.0f}% error)")
    print(f"  Crashed (all-NaN)   : {n_crashed}")
    print(f"{'─'*60}\n")

    if bad_indices:
        print(f"Badly converged / crashed samples ({len(bad_indices)}):\n")
        for sname in bad_indices:
            print(f"  [{sname}]")
            for r in bad_details[sname]:
                print(r)
        print()

    # ── print summary statistics for good samples ─────────────────────────────
    if good_abs_errs:
        ae = np.array(good_abs_errs)   # (n_good, n_cols)
        pe = np.array(good_pct_errs)

        # per-parameter means (only for active columns: GT mean > 0)
        gt_mean_per_col = np.nanmean(gt_arr, axis=0)
        active_cols = np.where(gt_mean_per_col > 0)[0]

        print(f"Error statistics over {n_good} well-converged samples")
        print(f"(only active parameters, i.e. mean GT > 0):\n")
        print(f"  {'Parameter':<14}  {'MAE':>12}  {'MAPE':>10}")
        print(f"  {'─'*14}  {'─'*12}  {'─'*10}")
        for ci in active_cols:
            name  = col_names[ci] if ci < len(col_names) else f"col_{ci}"
            mae_  = float(np.nanmean(ae[:, ci]))
            mape_ = float(np.nanmean(pe[:, ci])) * 100
            print(f"  {name:<14}  {mae_:>12.4f}  {mape_:>9.2f}%")

        overall_mae  = float(np.nanmean(ae[:, active_cols]))
        overall_mape = float(np.nanmean(pe[:, active_cols])) * 100
        print(f"\n  {'OVERALL':<14}  {overall_mae:>12.4f}  {overall_mape:>9.2f}%")
    else:
        print("No well-converged samples — no error statistics available.")

    # ── save output files ─────────────────────────────────────────────────────
    if out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(pred_path))
    os.makedirs(out_dir, exist_ok=True)

    index    = list(pred_df.index)
    res_df   = pd.DataFrame(np.array(out_res),     columns=col_names, index=index)
    pct_df   = pd.DataFrame(np.array(out_res_pct), columns=col_names, index=index)
    pred_out = pd.DataFrame(np.array(out_pred),    columns=col_names, index=index)

    for fname, df in [
        ("res_dis_recomputed.xlsx",            res_df),
        ("res_dis_percentage_recomputed.xlsx", pct_df),
        ("last_unfold_recomputed.xlsx",        pred_out),
    ]:
        fpath = os.path.join(out_dir, fname)
        with pd.ExcelWriter(fpath) as w:
            df.to_excel(w, sheet_name="system1")
        print(f"  Saved: {fpath}")

    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Recompute LMA residuals with Hungarian oscillator matching "
                    "and bad-convergence filtering.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--pred", required=True,
        help="Path to last_unfold.xlsx (LMA predictions).",
    )
    p.add_argument(
        "--gt", required=True,
        help="Path to gt.xlsx (ground-truth parameters, same shape as last_unfold).",
    )
    p.add_argument(
        "--fixed-prefix", type=int, default=11, dest="fixed_prefix",
        help="Number of system (non-oscillator) parameters at the start of each row.",
    )
    p.add_argument(
        "--pct-threshold", type=float, default=1.0, dest="pct_threshold",
        help="Relative-error threshold above which a sample is flagged as badly "
             "converged (1.0 = 100%%).",
    )
    p.add_argument(
        "--out-dir", default=None, dest="out_dir",
        help="Directory to write output files (default: same directory as --pred).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    recompute(
        pred_path     = args.pred,
        gt_path       = args.gt,
        fixed_prefix  = args.fixed_prefix,
        pct_threshold = args.pct_threshold,
        out_dir       = args.out_dir,
    )
