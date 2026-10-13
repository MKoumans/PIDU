"""Common MAPE and oscillator precision/recall for TRF and the networks.

Definition (identical for all methods):
  * System parameters: relative error |pred - GT| / |GT| over the parameters whose GT is non-zero
    (Case 2: system1 and system2).
  * Oscillators (w0, wp, g triplets): a predicted oscillator is paired with a GT oscillator
      - TRF: by slot (its output is already Hungarian-matched to the GT slots),
      - networks: by rank on the region grid (k-th confident prediction <-> k-th GT oscillator,
        the pairing used in the trainer export).
    A pair is a true positive (TP) if |w0_pred - w0_GT| <= tol * w0_GT.
    MAPE includes only the TP pairs; unmatched/missed oscillators are reported via recall.
  * Per-spectrum MAPE = mean over the evaluated columns of that spectrum (%); reported as
    mean/std over spectra.  precision = TP / n_pred, recall = TP / n_GT (pooled over spectra).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

N_SYS = 11
OSC_CAP = 5  # oscillator slots exported for the networks (trainer osc_max)


@dataclass
class Result:
    mape_per_spectrum: np.ndarray  # %
    tp: int
    n_pred: int
    n_gt: int


def load_gt(dataset: Path, system: str = 'system1'):
    """GT system parameters [N, 11] and, for system1, the list of GT oscillators per spectrum ([k, 3])."""
    d = np.load(dataset, allow_pickle=True)
    sys_gt = d[f'data_x_sys_tr_{system}'][:, 0, :]
    if system != 'system1':
        return sys_gt, None
    osc = d['data_x_osc_tr_system1'][:, 0]  # N, scales, regions, (conf, w0, wp, g)
    osc = osc.reshape(osc.shape[0], -1, 4)  # scale-major, region order
    return sys_gt, [o[o[:, 0] > 0.75][:, 1:] for o in osc]


def _sys_rel(pred_or_res_pct, gt, is_pct):
    """Per-spectrum list of relative errors over GT-non-zero system parameters."""
    mask = np.abs(gt) > 0
    rel = pred_or_res_pct if is_pct else np.abs(pred_or_res_pct - gt) / np.where(mask, np.abs(gt), 1)
    return [rel[i][mask[i]] for i in range(len(gt))]


def _pairs(pred, gt, tol):
    """Relative errors (w0, wp, g) of TP pairs for one spectrum; pred, gt: [k, 3] (pairs aligned by row)."""
    k = min(len(pred), len(gt))
    if k == 0:
        return np.empty(0), 0
    p, g = pred[:k], gt[:k]
    ok = np.abs(p[:, 0] - g[:, 0]) <= tol * np.abs(g[:, 0])
    rel = (np.abs(p - g) / np.abs(g))[ok].ravel()
    return rel, int(ok.sum())


def _assemble(sys_rel, osc_rel, tp, n_pred, n_gt):
    per = np.array([100 * np.mean(np.concatenate([s, o])) if len(s) + len(o) else np.nan
                    for s, o in zip(sys_rel, osc_rel)])
    return Result(per, int(sum(tp)), int(sum(n_pred)), int(sum(n_gt)))


def network_result(run_split: Path, dataset: Path, tol: float, systems=('system1',)) -> Result:
    """Network run: <run>/<split>/{last_unfold,res_dis_percentage}.xlsx against the dataset GT."""
    sys_gt, osc_gt = load_gt(dataset, 'system1')
    n = len(sys_gt)
    pct = {s: pd.read_excel(run_split / 'res_dis_percentage.xlsx', sheet_name=s, index_col=0) for s in systems}
    sys_rel = [np.concatenate(x) for x in zip(*[
        _sys_rel(pct[s].iloc[:, :N_SYS].to_numpy(), load_gt(dataset, s)[0], True) for s in systems])]
    lu = pd.read_excel(run_split / 'last_unfold.xlsx', sheet_name='system1', index_col=0)
    assert len(lu) == n, 'prediction/GT length mismatch'
    osc_rel, tp, n_pred, n_gt = [], [], [], []
    for i in range(n):
        pr = lu.iloc[i, N_SYS:].to_numpy(float).reshape(-1, 3)
        pr = pr[~np.isnan(pr).any(axis=1)]
        g = osc_gt[i][:OSC_CAP]
        rel, k = _pairs(pr, g, tol)
        osc_rel.append(rel); tp.append(k); n_pred.append(len(pr)); n_gt.append(len(g))
    return _assemble(sys_rel, osc_rel, tp, n_pred, n_gt)


def trf_result(split_dir: Path, tol: float) -> Result:
    """TRF run: gt.xlsx / output.xlsx (columns: 11 system params, then oscillator slots w0, wp, g)."""
    gt = pd.read_excel(split_dir / 'gt.xlsx', index_col=0).to_numpy(float)
    out = pd.read_excel(split_dir / 'output.xlsx', index_col=0).to_numpy(float)
    failed = np.isnan(out).any(axis=1)  # crashed fits: no prediction at all
    sys_rel = [np.empty(0) if failed[i] else r for i, r in enumerate(_sys_rel(out[:, :N_SYS], gt[:, :N_SYS], False))]
    osc_rel, tp, n_pred, n_gt = [], [], [], []
    for i in range(len(gt)):
        g = gt[i, N_SYS:].reshape(-1, 3)
        gp = g[:, 0] > 0
        p = np.zeros_like(g) if failed[i] else out[i, N_SYS:].reshape(-1, 3)
        present = (np.abs(p) > 0).any(axis=1)
        both = gp & present
        rel, k = _pairs(p[both], g[both], tol)
        osc_rel.append(rel); tp.append(k); n_pred.append(int(present.sum())); n_gt.append(int(gp.sum()))
    return _assemble(sys_rel, osc_rel, tp, n_pred, n_gt)
