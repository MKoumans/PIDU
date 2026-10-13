"""
Levenberg-Marquardt spectral fitting benchmark.

Usage:
    python lma.py --help
    python lma.py --dataset usecase1/case1_test_n2_10 --scheme random --backend numpy
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import warnings
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, Union

warnings.filterwarnings("ignore", category=RuntimeWarning, module=r".*Physics.*")               # 
warnings.filterwarnings("ignore", category=RuntimeWarning, module=r".*utils.*")                 # 
warnings.filterwarnings("ignore", message="overflow encountered", category=RuntimeWarning)      # 
warnings.filterwarnings("ignore", message="invalid value encountered", category=RuntimeWarning) # 

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import least_squares
from tqdm.auto import tqdm

from pidu.data.physics import TmMethod, TmMethodPyTorch
from pidu.data.simulation import SimulationDatasetBatch, load_sim_ds, p_gen_usecase1, p_gen_usecase2
from pidu.data.utils import np2pt, pt2np
from pidu.model.utils import create_masks_p, create_mask_wn
from pidu.config import usecase
from pidu.paths import case_outputs_dir

from pidu.baselines.lma_utils import (
    as_tensor, to_np_dtype, param_names, postprocess_oscillators,
    export_outputs, save_timing, save_run_summary,
)

# re-export for backward compat (tests, recompute_residuals)
_param_names = param_names
_postprocess_oscillators = postprocess_oscillators
_export_outputs = export_outputs
_save_timing = save_timing
_save_run_summary = save_run_summary

log = logging.getLogger(__name__)

Tensor = Union[torch.Tensor, np.ndarray]
TmmBackend = Union[Dict[str, Union[TmMethodPyTorch, TmMethod]], TmMethodPyTorch, TmMethod]


# ─────────────────────────────────────────────────────────────────────────────
# TmMethodWrapper
# ─────────────────────────────────────────────────────────────────────────────

class TmMethodWrapper:
    """Thin adapter: wraps numpy/pytorch TMM callable and returns a flat array."""

    def __init__(self, f: TmmBackend, backend: str = "torch") -> None:
        self.backend = backend.lower()
        self.tmm = f if isinstance(f, dict) else {"system1": f}
        self.system_names = sorted(self.tmm.keys())
        self.overrides: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    def set_overrides(self, overrides: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> None:
        """Set system-specific fixed parameters, e.g. a different sample thickness per system.

        The free parameters are shared between systems (same material); each entry maps a
        system name to (column indices, values) that replace the shared vector for that system.
        """
        self.overrides = overrides

    def _system_params(self, name: str, params: Tensor) -> Tensor:
        if name not in self.overrides:
            return params
        idx, vals = self.overrides[name]
        if self.backend == "torch":
            p = params.clone()
            p[..., torch.as_tensor(idx, device=p.device)] = torch.as_tensor(vals, dtype=p.dtype, device=p.device)
            return p
        p = np.array(params, copy=True)
        p[..., idx] = vals
        return p

    def __call__(self, wn: Dict[str, Any], params: Tensor) -> Tensor:
        pieces = list(self.channels(wn, params).values())
        if self.backend == "torch":
            return torch.cat([as_tensor(o).reshape(-1) for o in pieces])
        return np.concatenate([np.asarray(pt2np(o)).reshape(-1) for o in pieces])

    def channels(self, wn: Dict[str, Any], params: Tensor) -> Dict[str, Any]:
        """Simulated response per channel: '<system>', or '<system>_<i>' for a system with several outputs."""
        params = self._prep_params(params)
        wn_ = self._prep_wn(wn)
        out: Dict[str, Any] = {}
        for name in self.system_names:
            pieces = self._flatten(self.tmm[name](self._system_params(name, params), wn_[name]))
            if len(pieces) == 1:
                out[name] = pieces[0]
            else:
                out.update({f"{name}_{i}": piece for i, piece in enumerate(pieces)})
        return out

    def _prep_params(self, p: Tensor) -> Tensor:
        if self.backend == "torch":
            t = p if isinstance(p, torch.Tensor) else torch.as_tensor(p)
            if t.ndim == 1: t = t.unsqueeze(0).unsqueeze(0)
            elif t.ndim == 2: t = t.unsqueeze(1)
            return t
        arr = np.asarray(p)
        if arr.ndim == 1: arr = arr[np.newaxis, np.newaxis]
        elif arr.ndim == 2: arr = arr[:, np.newaxis]
        return arr

    def _prep_wn(self, wn: Dict[str, Any]) -> Dict[str, Any]:
        if self.backend == "torch":
            return wn
        return {k: np.asarray(pt2np(v)) for k, v in wn.items()}

    @staticmethod
    def _flatten(result: Any) -> List[Any]:
        if isinstance(result, (list, tuple)):
            out: List[Any] = []
            for r in result:
                out.extend(TmMethodWrapper._flatten(r))
            return out
        return [result]


# ─────────────────────────────────────────────────────────────────────────────
# LevenbergMarquardt
# ─────────────────────────────────────────────────────────────────────────────

class LevenbergMarquardt:
    """scipy least_squares wrapper for spectral parameter fitting."""

    def __init__(
        self,
        wn: Dict[str, Any],
        model: TmMethodWrapper,
        *,
        method: str = "lm",
        max_nfev: int = 100,
        tolerance: float = 1e-8,
        dtype: Union[torch.dtype, np.dtype, type] = torch.float64,
        backend: str = "torch",
        device: Union[str, torch.device] = "cpu",
        all_osc: bool = True,
        diff_step: Optional[float] = None,
        bound_free_relative_50: bool = False,
        debug_capture_y: bool = False,
        debug_capture_limit: Optional[int] = None,
        debug_export_dir: str = "lma_results/debug_y",
        record_iterates: bool = False,
    ) -> None:
        self.backend = backend.lower()
        self.is_numpy = self.backend == "numpy"
        self.device = device
        self.model = model
        self.method = method
        self.max_nfev = max_nfev
        self.tolerance = tolerance
        self.all_osc = all_osc
        self.diff_step = diff_step
        self.bound_free_relative_50 = bound_free_relative_50
        self.debug_capture_y = debug_capture_y
        self.debug_capture_limit = debug_capture_limit
        self.debug_export_dir = debug_export_dir
        self.record_iterates = record_iterates  # keep the full parameter vector of every iteration

        self.np_dtype = to_np_dtype(dtype)
        self.torch_dtype = dtype if isinstance(dtype, torch.dtype) else torch.float64
        self.wn = self._prep_wn(wn)
        self._wn_mask_np = self._build_wn_masks(self.wn)
        self._reset_state()

    # ── helpers ──────────────────────────────────────────────────────────────

    def _prep_wn(self, wn: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in wn.items():
            arr = np.asarray(pt2np(v), dtype=self.np_dtype)
            out[k] = arr if self.is_numpy else torch.as_tensor(arr, dtype=self.torch_dtype, device=self.device)
        return out

    def _to_np(self, x: Any) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            return pt2np(x).astype(self.np_dtype)
        return np.asarray(x, dtype=self.np_dtype)

    def _build_wn_masks(self, wn: Dict[str, Any]) -> Dict[str, np.ndarray]:
        masks: Dict[str, np.ndarray] = {}
        for key, value in wn.items():
            wn_t = torch.as_tensor(np.asarray(pt2np(value)).reshape(-1))
            masks[key] = np.asarray(pt2np(create_mask_wn(wn_t, key)), dtype=bool).reshape(-1)
        return masks

    def _diff_step_val(self) -> Optional[float]:
        if self.diff_step is not None:
            return self.diff_step
        return 1e-3 if self.np_dtype == np.float32 else None

    # ── debug capture ────────────────────────────────────────────────────────

    def _reset_state(self) -> None:
        self._y_cap: List[np.ndarray] = []
        self._r_norm: List[float] = []
        self._e_idx: List[int] = []
        self._counter = 0
        self._mask_bool: Optional[np.ndarray] = None
        self._free_idx: Optional[np.ndarray] = None
        self._p_true_flat: Optional[np.ndarray] = None
        self._full_work: Optional[np.ndarray] = None
        self._y_mask_flat: Optional[np.ndarray] = None
        self._param_shape: Optional[tuple] = None
        self._init_shape: Optional[tuple] = None

    def _capture(self, y_pred: np.ndarray, res: np.ndarray) -> None:
        if not self.debug_capture_y:
            return
        if self.debug_capture_limit is not None and len(self._y_cap) >= self.debug_capture_limit:
            self._counter += 1
            return
        self._y_cap.append(y_pred.astype(np.float64).copy())
        self._r_norm.append(float(np.linalg.norm(res)))
        self._e_idx.append(self._counter)
        self._counter += 1

    def _save_debug(self) -> Dict[str, Any]:
        if not self.debug_capture_y or not self._y_cap:
            return {}

        y = np.array(self._y_cap).reshape(len(self._y_cap), -1)
        mu, sd = np.mean(y, 0), np.std(y, 0)
        ei, rn = np.array(self._e_idx), np.array(self._r_norm)
        all_equal = bool(np.allclose(y, y[0:1], rtol=0, atol=0))

        os.makedirs(self.debug_export_dir, exist_ok=True)
        npz = os.path.join(self.debug_export_dir, "debug_y.npz")
        fig_path = os.path.join(self.debug_export_dir, "debug_y.png")
        np.savez(npz, y_stack=y, y_mean=mu, y_std=sd, eval_indices=ei, residual_norms=rn)

        x = np.arange(y.shape[1])
        sel = np.linspace(0, len(y) - 1, min(10, len(y)), dtype=int)
        fig, axes = plt.subplots(3, 1, figsize=(12, 9), tight_layout=True)
        for i in sel:
            axes[0].plot(x, y[i], alpha=0.4, lw=0.8)
        axes[0].plot(x, mu, "k", lw=1.5, label="mean")
        axes[0].set(title="y per evaluation", ylabel="y"); axes[0].legend(fontsize=8)
        axes[1].plot(x, mu, lw=1.5, label="mean")
        axes[1].fill_between(x, mu - sd, mu + sd, alpha=0.3, label="±σ")
        axes[1].set(title="Pointwise variance", ylabel="y"); axes[1].legend(fontsize=8)
        axes[2].plot(ei, rn, "o-", ms=3, lw=1)
        axes[2].set(title="Residual norm", xlabel="Evaluation index", ylabel="‖r‖₂")
        fig.savefig(fig_path, dpi=150); plt.close(fig)

        if all_equal:
            log.warning("Debug: all captured y identical — diff_step likely too small for %s.", self.np_dtype)
        return dict(n_evals=int(y.shape[0]), signal_len=int(y.shape[1]),
                    mean_std=float(np.mean(sd)), max_std=float(np.max(sd)),
                    all_equal=all_equal, npz_path=npz, fig_path=fig_path)

    # ── residual / fit ───────────────────────────────────────────────────────

    def _residual(self, free: np.ndarray, y_target: np.ndarray) -> np.ndarray:
        self._full_work[self._free_idx] = free
        params = self._full_work.reshape(self._param_shape)
        if self.is_numpy:
            y_pred = np.asarray(self.model(self.wn, params)).reshape(-1)
        else:
            p_t = np2pt(params, dtype=self.torch_dtype, device=self.device).reshape(1, -1)
            y_pred = pt2np(self.model(self.wn, p_t)).reshape(-1)

        y_eval = y_pred if self._y_mask_flat is None else y_pred[self._y_mask_flat]
        res = y_eval - y_target
        self._capture(y_eval, res)
        return res

    def fit(
        self,
        y_data: Union[Dict[str, Tensor], Tensor],
        initial_params: Tensor,
        p_true: Tensor,
        mask: Tensor,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """Run LMA on one sample. Returns (optimized_params, history_dict)."""
        self._reset_state()
        self._init_shape = tuple(initial_params.shape) if hasattr(initial_params, "shape") else (-1,)

        p_true_np = self._to_np(p_true)
        init_np = self._to_np(initial_params)

        # ── prepare target signal ────────────────────────────────────────────
        if isinstance(y_data, dict):
            preferred = [k for k in self.model.system_names if k in y_data]
            extra = sorted(k for k in y_data if k not in set(preferred))
            y_chunks, y_masks = [], []
            for key in preferred + extra:
                y_k = self._to_np(y_data[key]).reshape(-1)
                m_k = self._wn_mask_np.get(key)
                if m_k is None or m_k.size != y_k.size:
                    if m_k is not None and m_k.size != y_k.size:
                        log.warning("wn_mask size mismatch for %s — using unmasked.", key)
                    m_k = np.ones(y_k.size, dtype=bool)
                y_chunks.append(y_k[m_k])
                y_masks.append(m_k)
            y_np = np.concatenate(y_chunks)
            self._y_mask_flat = np.concatenate(y_masks).astype(bool, copy=False)
        else:
            y_np = self._to_np(y_data).reshape(-1)
            self._y_mask_flat = np.ones(y_np.size, dtype=bool)

        # ── build active mask (0 = free, 1 = fixed) ─────────────────────────
        active = self._to_np(mask).astype(bool).ravel()
        if not self.all_osc and active.size > 11:
            osc = p_true_np.ravel()[11:]
            if osc.size % 3 == 0:
                eps = np.finfo(self.np_dtype).eps * 10.0
                osc_present = np.any(np.abs(osc.reshape(-1, 3)) > eps, axis=1)
                active[11:] |= ~np.repeat(osc_present, 3)
            else:
                active[11:] |= ~(np.abs(osc) > np.finfo(self.np_dtype).eps * 10.0)

        self._mask_bool = active
        self._free_idx = np.flatnonzero(~active)
        self._p_true_flat = p_true_np.ravel().astype(self.np_dtype)
        self._full_work = self._p_true_flat.copy()
        self._param_shape = p_true_np.shape

        free0 = init_np.ravel()[self._free_idx].astype(self.np_dtype, copy=False)
        if free0.size == 0:
            raise ValueError("No free parameters — check your mask.")

        # ── bounds (wp, g >= 0; optional relative bounds) ────────────────────
        lb = np.full(free0.size, -np.inf, dtype=self.np_dtype)
        ub = np.full(free0.size, np.inf, dtype=self.np_dtype)
        for j, fi in enumerate(self._free_idx):
            if fi >= 11 and (fi - 11) % 3 != 0:
                lb[j] = 0.0

        if self.bound_free_relative_50:
            lb_floor = lb.copy()
            rel_lb = np.minimum(free0 * 0.5, free0 * 1.5)
            rel_ub = np.maximum(free0 * 0.5, free0 * 1.5)
            lb = np.maximum(lb_floor, rel_lb)
            ub = np.minimum(ub, rel_ub)

            degenerate = ub <= lb
            if np.any(degenerate):
                eps = np.maximum(np.abs(free0[degenerate]), 1.0) * 1e-3
                lb[degenerate] = np.maximum(lb_floor[degenerate], lb[degenerate] - eps)
                ub[degenerate] = ub[degenerate] + eps
            free0 = np.clip(free0, lb, ub)

        # ── solve ────────────────────────────────────────────────────────────
        iterates: List[np.ndarray] = []

        def record(free: np.ndarray) -> None:
            full = self._p_true_flat.copy()
            full[self._free_idx] = free
            iterates.append(full)

        callback = None
        if self.record_iterates:
            record(free0)
            callback = record  # called with x after every iteration (trf/dogbox only)

        t0 = time.perf_counter()
        result = least_squares(
            self._residual, free0, args=(y_np,),
            method=self.method,
            ftol=self.tolerance, xtol=self.tolerance, gtol=self.tolerance,
            max_nfev=self.max_nfev, diff_step=self._diff_step_val(),
            bounds=(lb, ub), x_scale="jac", verbose=0, jac="3-point",
            callback=callback,
        )
        elapsed = time.perf_counter() - t0
        log.debug("LMA: %s | nfev=%d | cost=%.4e | %.2fs", result.message, result.nfev, result.cost, elapsed)

        # ── reconstruct full params ──────────────────────────────────────────
        full_opt = self._p_true_flat.copy()
        full_opt[~self._mask_bool] = result.x
        opt_arr = full_opt.reshape(self._param_shape)

        if self.is_numpy:
            optimized = opt_arr.reshape(self._init_shape)
        else:
            optimized = np2pt(opt_arr, dtype=self.torch_dtype, device=self.device).view(*self._init_shape)

        history = dict(
            success=result.success, status=result.status, message=result.message,
            nfev=result.nfev, njev=result.njev, cost=result.cost,
            optimality=result.optimality, fitting_time=elapsed,
            x=full_opt, debug_y=self._save_debug(),
        )
        if self.record_iterates:
            history["iterates"] = np.stack(iterates)  # [n_iterations + 1, n_params], row 0 = initial

        return optimized, history


# ─────────────────────────────────────────────────────────────────────────────
# Dataset helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_tmm(dataset: SimulationDatasetBatch, backend: str) -> TmmBackend:
    sys_dicts = getattr(dataset, "system_dicts", None)
    if not isinstance(sys_dicts, dict) or not sys_dicts:
        return dataset.tmm
    out: Dict[str, Union[TmMethodPyTorch, TmMethod]] = {}
    for name, cfg in sys_dicts.items():
        rmode = cfg.get("return_mode", "dict") if isinstance(cfg, dict) else "dict"
        if backend == "numpy":
            ndt = np.float32 if getattr(dataset, "dtype", None) == torch.float32 else np.float64
            out[name] = TmMethod(dtype=ndt, dtc=np.complex64 if ndt == np.float32 else np.complex128,
                                 return_mode=rmode, return_mode_osc="forward")
        else:
            out[name] = TmMethodPyTorch(device=getattr(dataset, "device", "cpu"),
                                        dtype=getattr(dataset, "dtype", torch.float32),
                                        return_mode=rmode, return_mode_osc="forward")
    return out


def _dataset_param_ranges(
    dataset: SimulationDatasetBatch, system: str = "system1",
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Column-wise min/max of sys + osc training parameters."""
    if not hasattr(dataset, "data_x_sys_tr") or system not in dataset.data_x_sys_tr:
        return None
    try:
        sys_np = pt2np(dataset.data_x_sys_tr[system])
        cols = sys_np.reshape(sys_np.shape[0], -1)
        if hasattr(dataset, "data_x_osc_tr") and system in dataset.data_x_osc_tr:
            osc_raw = dataset.data_x_osc_tr[system]
            if osc_raw is not None:
                osc_np = pt2np(osc_raw)
                if osc_np.size > 1:
                    cols = np.concatenate([cols, osc_np[..., -3:].reshape(osc_np.shape[0], -1)], axis=1)
        return np.min(cols, 0), np.max(cols, 0)
    except Exception:
        log.exception("Could not compute dataset parameter ranges")
        return None


N_SYS_PARAMS = 11  # th0, d, eps_inf, drude_p, drude_g, w0/wp/g before, w0/wp/g after


def _training_param_ranges(p_gen: Dict[str, Any], p_true_row: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-column sampling bounds of the simulated training data (p_gen_usecase*).

    System columns use the generator bounds directly. Each oscillator triplet uses the
    bounds of the generator oscillator whose w0 range contains the true w0, i.e. only
    the identity of the peak is taken from the reference (as for the fixed oscillator
    set), not its parameter values. Absent oscillators keep their (fixed) reference value.
    """
    def rng(entry: Any) -> Tuple[float, float]:
        entry = entry[0] if isinstance(entry, dict) else entry  # 'd' may be per-system
        lo_hi = entry[0]
        return float(lo_hi[0]), float(lo_hi[1])

    truth = np.asarray(p_true_row, dtype=np.float64).ravel()
    lo, hi = truth.copy(), truth.copy()

    sys_ranges = [rng(p_gen["th0"]), rng(p_gen["d"]), rng(p_gen["e_inf"]),
                  rng(p_gen["drude_p"]), rng(p_gen["drude_g"])]
    sys_ranges += [tuple(map(float, a[0])) for a in p_gen["oscs_start"]]
    sys_ranges += [tuple(map(float, a[0])) for a in p_gen["oscs_end"]]
    for i, (l, h) in enumerate(sys_ranges[:min(N_SYS_PARAMS, truth.size)]):
        lo[i], hi[i] = l, h

    w0_r, wp_r, g_r = p_gen["w0"][0], p_gen["wp"][0], p_gen["g"][0]
    for s in range(N_SYS_PARAMS, truth.size - 2, 3):
        w0 = truth[s]
        j = next((j for j, (l, h) in enumerate(w0_r) if l <= w0 <= h), None)
        if j is None:
            continue
        lo[s:s + 3] = w0_r[j][0], wp_r[j][0], g_r[j][0]
        hi[s:s + 3] = w0_r[j][1], wp_r[j][1], g_r[j][1]
    return lo, hi


def _random_init_training(
    p_true: torch.Tensor,
    mask: torch.Tensor,
    p_gen: Dict[str, Any],
    dtype: torch.dtype,
    device: Union[str, torch.device],
) -> torch.Tensor:
    """Sample a uniform random initialisation within the training bounds."""
    initial = p_true.clone()
    lo_np, hi_np = _training_param_ranges(p_gen, pt2np(p_true[0]))
    lo = torch.as_tensor(lo_np, dtype=dtype, device=device)
    hi = torch.as_tensor(hi_np, dtype=dtype, device=device)
    rand = lo + (hi - lo) * torch.rand(p_true.shape[0], p_true.shape[1], dtype=dtype, device=device)
    free = mask[0] == 0
    initial[:, free] = rand[:, free]
    return initial


def _random_init(
    p_true: torch.Tensor,
    mask: torch.Tensor,
    ds_ranges: Optional[Tuple[np.ndarray, np.ndarray]],
    relative_span: float,
    dtype: torch.dtype,
    device: Union[str, torch.device],
) -> torch.Tensor:
    """Sample uniform random initialisation from dataset-wide per-parameter ranges."""
    initial = p_true.clone()
    n_p = p_true.shape[1]
    truth = p_true.reshape(-1)
    span = torch.maximum(truth.abs() * relative_span, torch.full_like(truth, 1e-3))

    if ds_ranges is not None and len(ds_ranges[0]) >= n_p:
        lo = torch.as_tensor(ds_ranges[0][:n_p], dtype=dtype, device=device)
        hi = torch.as_tensor(ds_ranges[1][:n_p], dtype=dtype, device=device)
        wide = (hi - lo) <= 1e-12
        lo = torch.where(wide, truth - span, lo)
        hi = torch.where(wide, truth + span, hi)
    else:
        lo, hi = truth - span, truth + span

    rand = lo + (hi - lo) * torch.rand(p_true.shape[0], n_p, dtype=dtype, device=device)
    free = mask[0] == 0
    initial[:, free] = rand[:, free]
    return initial


# ─────────────────────────────────────────────────────────────────────────────
# benchmark_lma
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_lma(
    dataset: SimulationDatasetBatch,
    num_samples: Optional[int] = None,
    max_iterations: int = 100,
    initialization_scheme: str = "baseline",
    noise_factor: float = 0.2,
    all_osc: bool = True,
    backend: str = "torch",
    post_process_oscillators: bool = True,
    random_init_relative_span: float = 0.25,
    diff_step: Optional[float] = None,
    debug_capture_y: bool = False,
    debug_capture_limit: Optional[int] = None,
    suppress_non_osc: bool = False,
    bound_free_relative_50: bool = False,
    random_seed: Optional[int] = None,
    random_range: str = "training",
) -> Dict[str, Any]:
    """Run LMA on *num_samples* samples and return aggregated statistics.

    random_range: 'training' samples the random initialisation within the simulation
    bounds of the training data; 'dataset' (legacy) uses the min/max of the reference
    values of the evaluated dataset, which is an oracle prior for small test sets.
    """
    if not isinstance(dataset, SimulationDatasetBatch):
        raise TypeError("dataset must be a SimulationDatasetBatch")

    if dataset.y_return_mode != "noised":
        # LMA fits against a single noised spectrum; 'both' stacks clean+noised along
        # the last dim, which doubles the flattened signal length and breaks the wn mask.
        log.info("Overriding dataset.y_return_mode=%r -> 'noised' for LMA fitting.",
                 dataset.y_return_mode)
        dataset.y_return_mode = "noised"

    scheme = initialization_scheme.lower()
    if scheme not in {"baseline", "noisy", "random"}:
        raise ValueError(f"initialization_scheme must be baseline/noisy/random, got {scheme!r}")

    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    n = len(dataset) if (num_samples is None or num_samples == -1) else min(num_samples, len(dataset))
    log.info("LMA benchmark | n=%d | scheme=%s | backend=%s | max_nfev=%d", n, scheme, backend, max_iterations)

    if random_range not in {"training", "dataset"}:
        raise ValueError(f"random_range must be training/dataset, got {random_range!r}")
    p_gen = {1: p_gen_usecase1, 2: p_gen_usecase2}.get(int(usecase))
    if scheme == "random" and random_range == "training" and p_gen is None:
        raise ValueError(f"No training bounds defined for usecase {usecase}")

    backend_tmm = _build_tmm(dataset, backend)
    ds_ranges = _dataset_param_ranges(dataset, "system1")
    # Only a run over the whole dataset owns loss_lma_<scheme>/ (the input of the paper tables); a partial
    # run (tests, timing probes, debugging) writes next to it, so it can never overwrite those results.
    run_name = f"loss_lma_{scheme}" if n == len(dataset) else f"loss_lma_{scheme}_n{n}"
    export_dir = str(case_outputs_dir(usecase) / run_name)

    # ── accumulators ─────────────────────────────────────────────────────────
    times, costs, iters, successes = [], [], [], []
    statuses, messages = [], []
    abs_errs, rel_errs, abs_errs_raw, rel_errs_raw = [], [], [], []
    output_rows, res_rows, res_pct_rows, gt_rows = [], [], [], []
    out_indices, pp_rows, sample_timings = [], [], []
    last_sample: Optional[Dict] = None

    model = TmMethodWrapper(f=backend_tmm, backend=backend)
    use_rel_bounds = bound_free_relative_50 and scheme != "baseline"

    for idx in tqdm(range(n), desc=f"LMA samples ({scheme})", leave=False):
        wn, x_sys, x_osc, y, sample_idx = dataset[idx]
        (x_sys_init, x_sys_tr), (x_osc_init, x_osc_tr) = x_sys, x_osc

        p_true = {k: torch.cat([x_sys_tr[k], x_osc_tr[k][..., -3:].flatten(1)], dim=1)
                  for k in ["system1"]}
        mask = create_masks_p(usecase, n_layers=1, n_sys_pars=x_sys_init,
                              n_osc_pars=x_osc_init, dv=dataset.device)

        # ── system-specific fixed parameters (e.g. thickness per sample/system) ──
        overrides: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        ref_sys = pt2np(x_sys_tr["system1"]).ravel()
        fixed_sys = pt2np(mask["system1"]).ravel()[:ref_sys.size].astype(bool)
        for name in model.system_names:
            if name == "system1" or name not in x_sys_tr:
                continue
            sys_k = pt2np(x_sys_tr[name]).ravel()
            idx = np.flatnonzero(fixed_sys & ~np.isclose(sys_k, ref_sys))
            if idx.size:
                overrides[name] = (idx, sys_k[idx])
        model.set_overrides(overrides)

        # ── initialisation ───────────────────────────────────────────────────
        if scheme == "noisy":
            initial = {k: torch.where(
                mask[k] == 0,
                p_true[k] * (1.0 + (torch.rand_like(p_true[k]) - 0.5) * noise_factor),
                p_true[k].clone(),
            ) for k in p_true}
        elif scheme == "random" and random_range == "training":
            initial = {"system1": _random_init_training(
                p_true["system1"], mask["system1"], p_gen, dataset.dtype, dataset.device,
            )}
        elif scheme == "random":
            initial = {"system1": _random_init(
                p_true["system1"], mask["system1"], ds_ranges,
                random_init_relative_span, dataset.dtype, dataset.device,
            )}
        else:
            ones = {k: torch.ones_like(v) for k, v in p_true.items()}
            initial = {k: p_true[k] * mask[k] + ones[k] * (1 - mask[k]).abs() for k in p_true}

        # ── convert to backend ───────────────────────────────────────────────
        if backend == "numpy":
            y_fit = y
            init_fit = {k: pt2np(v) for k, v in initial.items()}
            p_tr_fit = {k: pt2np(v) for k, v in p_true.items()}
            mask_fit = {k: pt2np(v).astype(bool) for k, v in mask.items()}
        else:
            y_fit, init_fit, p_tr_fit, mask_fit = y, initial, p_true, mask

        # ── fit ──────────────────────────────────────────────────────────────
        lma = LevenbergMarquardt(
            wn=wn, model=model, method="trf", max_nfev=max_iterations,
            dtype=dataset.dtype, backend=backend, device=dataset.device, all_osc=all_osc,
            diff_step=diff_step, debug_capture_y=debug_capture_y,
            bound_free_relative_50=use_rel_bounds,
            debug_capture_limit=debug_capture_limit,
            debug_export_dir=os.path.join(
                export_dir, f"test_{dataset.settype}",
                f"sample_{int(sample_idx):03d}", "debug_y",
            ),
        )

        t0 = time.perf_counter()
        try:
            final, history = lma.fit(
                y_fit, 
                init_fit["system1"],
                p_tr_fit["system1"], 
                mask_fit["system1"]
            )
        except Exception as exc:
            log.warning("Sample %d/%d failed: %s", idx + 1, n, exc)
            _n_p = pt2np(p_tr_fit["system1"]).ravel().shape[0]
            _nan = np.full(_n_p, np.nan)
            times.append(np.nan); costs.append(np.nan); iters.append(0); successes.append(False)
            statuses.append(None); messages.append(str(exc))
            sample_timings.append({"sample_idx": sample_idx, "total_time": np.nan})
            abs_errs_raw.append(_nan); rel_errs_raw.append(_nan)
            abs_errs.append(_nan); rel_errs.append(_nan)
            output_rows.append(_nan); res_rows.append(_nan); res_pct_rows.append(_nan)
            gt_rows.append(pt2np(p_tr_fit["system1"]).ravel())
            out_indices.append(int(sample_idx))
            pp_rows.append({"sample_idx": int(sample_idx), "changed": False,
                            "n_triplets": 0, "assignment": [],
                            "mae_before": np.nan, "mae_after": np.nan})
            continue
        elapsed = time.perf_counter() - t0

        times.append(elapsed); costs.append(history["cost"])
        iters.append(history["nfev"]); successes.append(history["success"])
        statuses.append(history["status"]); messages.append(history["message"])
        sample_timings.append({"sample_idx": sample_idx, "total_time": elapsed})

        # ── errors & post-processing ─────────────────────────────────────────
        p_true_flat = pt2np(p_true["system1"]).ravel()
        p_opt_raw = pt2np(final).ravel()
        pp = postprocess_oscillators(p_opt_raw, p_true_flat,
                                     enabled=post_process_oscillators,
                                     suppress_non_osc=suppress_non_osc)
        p_opt = pp["params"]

        abs_raw = np.abs(p_opt_raw - p_true_flat)
        rel_raw = abs_raw / (np.abs(p_true_flat) + 1e-10)
        abs_e = np.abs(p_opt - p_true_flat)
        rel_e = abs_e / (np.abs(p_true_flat) + 1e-10)

        abs_errs_raw.append(abs_raw); rel_errs_raw.append(rel_raw)
        abs_errs.append(abs_e); rel_errs.append(rel_e)
        output_rows.append(p_opt); res_rows.append(abs_e); res_pct_rows.append(rel_e)
        gt_rows.append(p_true_flat)
        out_indices.append(int(sample_idx))
        pp_rows.append({"sample_idx": int(sample_idx), "changed": pp["changed"],
                        "n_triplets": pp["n_triplets"], "assignment": pp["assignment"].tolist(),
                        "mae_before": pp["mae_before"], "mae_after": pp["mae_after"]})

        log.debug("Sample %d/%d | cost=%.4e | nfev=%d | %.2fs", idx+1, n, history["cost"], history["nfev"], elapsed)

        # capture last sample for plotting
        if idx == n - 1:
            p_fin = np2pt(p_opt.reshape(1, -1), dtype=dataset.dtype, device=dataset.device) if backend != "numpy" else p_opt.reshape(1, -1)
            y_opt_pred = pt2np(model(wn, p_fin)).ravel()
            if lma._y_mask_flat is not None:
                y_opt_pred = y_opt_pred[lma._y_mask_flat]

            if isinstance(y_fit, dict):
                y_keys = [k for k in model.system_names if k in y_fit] + sorted(k for k in y_fit if k not in set(model.system_names))
                y_true_np = np.concatenate([pt2np(y_fit[k]).ravel() for k in y_keys])
                if lma._y_mask_flat is not None:
                    y_true_np = y_true_np[lma._y_mask_flat]
            else:
                y_true_np = (y_fit if isinstance(y_fit, np.ndarray) else pt2np(y_fit)).ravel()
                if lma._y_mask_flat is not None:
                    y_true_np = y_true_np[lma._y_mask_flat]

            last_sample = dict(
                idx=sample_idx, p_true=p_true_flat,
                p_initial=pt2np(initial["system1"]).ravel(),
                p_optimized=p_opt, p_optimized_raw=p_opt_raw,
                y_true=y_true_np, y_optimized=y_opt_pred,
                cost=history["cost"], iterations=history["nfev"],
                success=history["success"], status=history["status"],
                message=history["message"], postprocess=pp,
            )

    # ── aggregate ────────────────────────────────────────────────────────────
    t = np.array(times)
    ae, re = np.array(abs_errs), np.array(rel_errs)
    ae_raw, re_raw = np.array(abs_errs_raw), np.array(rel_errs_raw)

    results: Dict[str, Any] = dict(
        num_samples=n, max_iterations=max_iterations,
        initialization_scheme=scheme,
        random_range=random_range, random_seed=random_seed,
        system_overrides={k: {"idx": v[0].tolist(), "values": v[1].tolist()}
                          for k, v in model.overrides.items()},
        bound_free_relative_50=bool(use_rel_bounds),
        noise_factor=noise_factor, backend=backend,
        wall_time_mean=float(np.nanmean(t)), wall_time_std=float(np.nanstd(t)),
        wall_time_median=float(np.nanmedian(t)),
        wall_time_min=float(np.nanmin(t)), wall_time_max=float(np.nanmax(t)),
        time_per_sample_ms=float(np.nanmean(t) * 1e3),
        throughput=float(1.0 / np.nanmean(t)),
        cost_mean=float(np.nanmean(costs)), cost_std=float(np.nanstd(costs)),
        iterations_mean=float(np.nanmean(iters)),
        success_rate=float(np.mean(successes)),
        abs_error_mean=np.nanmean(ae, 0), abs_error_std=np.nanstd(ae, 0),
        rel_error_mean=np.nanmean(re, 0), rel_error_std=np.nanstd(re, 0),
        abs_error_mean_raw=np.nanmean(ae_raw, 0) if ae_raw.size else None,
        rel_error_mean_raw=np.nanmean(re_raw, 0) if re_raw.size else None,
        postprocess_changed_frac=float(np.mean([r["changed"] for r in pp_rows])),
        postprocess_details=pp_rows, sample_timings=sample_timings,
        nfev_mean=float(np.nanmean(iters)), nfev_median=float(np.nanmedian(iters)),
        nfev=list(iters), statuses=list(statuses), messages=list(messages),
        stop_reason_counts=dict(Counter(messages)),
    )
    if last_sample:
        results["last_sample"] = last_sample

    # ── log summary ──────────────────────────────────────────────────────────
    log.info(
        "Results | time=%.1f±%.1f ms | cost=%.4e | iter=%.1f | success=%.0f%%",
        results["time_per_sample_ms"], results["wall_time_std"] * 1e3,
        results["cost_mean"], results["iterations_mean"], results["success_rate"] * 100,
    )

    # ── persist ──────────────────────────────────────────────────────────────
    os.makedirs(export_dir, exist_ok=True)
    if last_sample:
        saveable = {k: v for k, v in last_sample.items() if not isinstance(v, dict)}
        np.savez(os.path.join(export_dir, "last_sample.npz"), **saveable)
        try:
            fig, axes = plt.subplots(2, 1, figsize=(12, 7), tight_layout=True)
            x = np.arange(len(last_sample["y_true"]))
            res = last_sample["y_true"] - last_sample["y_optimized"]
            axes[0].plot(x, last_sample["y_true"], "k-", lw=1.5, label="True")
            axes[0].plot(x, last_sample["y_optimized"], "r--", lw=1.5, alpha=0.8, label="Optimized")
            axes[0].set(ylabel="Transmission",
                        title=f"Sample {last_sample['idx']} | cost={last_sample['cost']:.4e}")
            axes[0].legend()
            axes[1].plot(x, res, "g-", lw=1)
            axes[1].axhline(0, c="k", ls="--", alpha=0.5)
            axes[1].set(xlabel="Index", ylabel="Residual",
                        title=f"RMSE={np.sqrt(np.mean(res**2)):.4e}")
            fig.savefig(os.path.join(export_dir, "last_sample_fit.png"), dpi=150)
            plt.close(fig)
        except Exception:
            log.warning("Could not save last-sample plot.")

    if output_rows:
        _export_outputs(
            np.array(output_rows), np.array(res_rows), np.array(res_pct_rows),
            out_indices, export_dir, f"test_{dataset.settype}",
            gt=np.array(gt_rows) if gt_rows else None,
        )
    _save_timing(sample_timings, export_dir)
    _save_run_summary(results, successes, export_dir)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Levenberg-Marquardt spectral fitting benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", default=f"usecase{usecase}/case{usecase}_test_n2_10")
    p.add_argument("--scheme", default="noisy", choices=["baseline", "noisy", "random"])
    p.add_argument("--samples", default=100, type=int)
    p.add_argument("--max-iter", default=100, type=int, dest="max_iter")
    p.add_argument("--noise", default=0.20, type=float)
    p.add_argument("--backend", default="numpy", choices=["numpy", "torch"])
    p.add_argument("--all-osc", action="store_true", dest="all_osc")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], dest="log_level")
    p.add_argument("--suppress-fp", action="store_true", dest="suppress_fp")
    p.add_argument("--bound-free-50", action="store_true", dest="bound_free_50",
                   help="Constrain free parameters to 50%%..150%% of initial value.")
    p.add_argument("--seed", default=0, type=int, help="Random seed for the noisy/random initialisation.")
    p.add_argument("--random-range", default="training", choices=["training", "dataset"], dest="random_range",
                   help="'training': simulation bounds; 'dataset': min/max of the evaluated dataset (legacy).")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")

    log.info("Device: %s, usecase: %s", torch.device("cuda" if torch.cuda.is_available() else "cpu"), usecase)
    dataset = load_sim_ds(args.dataset, y_return_mode="noised")
    results = benchmark_lma(
        dataset=dataset, num_samples=args.samples, max_iterations=args.max_iter,
        initialization_scheme=args.scheme, noise_factor=args.noise,
        backend=args.backend, all_osc=args.all_osc,
        debug_capture_y=args.debug, suppress_non_osc=args.suppress_fp,
        bound_free_relative_50=args.bound_free_50,
        random_seed=args.seed, random_range=args.random_range,
    )
    log.info("Done | mean time/sample: %.2f ms", results["time_per_sample_ms"])
