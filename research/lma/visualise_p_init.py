"""
Visualise the initial-parameter deviation for the *noisy* and *random*
LMA initialisation schemes.

For each sample, the per-parameter deviation is computed as a fraction of
the actual sampling range used by each scheme:

  noisy  : range_i = |p_true_i| * noise_factor          (proportional noise)
  random : range_i = hi_i - lo_i  (dataset-wide or fallback span)

The normalised deviation is then  |p_init_i - p_true_i| / range_i.
Averaging over free parameters gives one scalar per sample.

NOTE: oscillator parameters for absent oscillators have p_true ≈ 0.
The mask from create_masks_p marks ALL oscillator slots as free regardless of
whether they are active, so a relative-to-p_true metric would explode there.
Normalising by the sampling range avoids that pathology.

Usage:
    python research/lma/visualise_p_init.py
    python research/lma/visualise_p_init.py --samples 200 --seed 42
    python research/lma/visualise_p_init.py --noise 0.10 --span 0.50

The defaults match the TRF runs of paper/lma.sh (full test set, random scheme
within the training bounds). Each run writes <output>_data.csv; the CSVs of
use case 1 and 2 are plotted together by scripts/figure_p_init.py (Fig. S5):

    PIDU_USECASE=1 python research/lma/visualise_p_init.py --seed 0 --output outputs/case1/figures/p_init_deviation.png
    PIDU_USECASE=2 python research/lma/visualise_p_init.py --seed 0 --dataset usecase2/case2_experimental --output outputs/case2/figures/p_init_deviation.png
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


from pidu.config import usecase
from pidu.data.simulation import SimulationDatasetBatch, load_sim_ds, p_gen_usecase1, p_gen_usecase2
from pidu.data.utils import pt2np
from pidu.model.utils import create_masks_p
from pidu.baselines.lma import _dataset_param_ranges, _random_init, _random_init_training

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_random_ranges(
    p_true: torch.Tensor,
    mask: torch.Tensor,
    ds_ranges: Optional[Tuple[np.ndarray, np.ndarray]],
    relative_span: float,
    dtype: torch.dtype,
    device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Replicate the lo/hi computation from _random_init so we can use (hi - lo)
    as the normalisation denominator in the deviation metric.
    """
    n_p = p_true.shape[1]
    if ds_ranges is not None and len(ds_ranges[0]) >= n_p:
        lo = torch.as_tensor(ds_ranges[0][:n_p], dtype=dtype, device=device)
        hi = torch.as_tensor(ds_ranges[1][:n_p], dtype=dtype, device=device)
        truth = p_true.reshape(-1)
        span  = torch.maximum(truth.abs() * relative_span, torch.full_like(truth, 1e-3))
        wide  = (hi - lo) <= 1e-12
        lo    = torch.where(wide, truth - span, lo)
        hi    = torch.where(wide, truth + span, hi)
    else:
        truth = p_true.reshape(-1)
        span  = torch.maximum(truth.abs() * relative_span, torch.full_like(truth, 1e-3))
        lo, hi = truth - span, truth + span
    return pt2np(lo).reshape(-1), pt2np(hi).reshape(-1)


def _collect_p_init_deviations(
    dataset: SimulationDatasetBatch,
    scheme: str,
    num_samples: int,
    noise_factor: float,
    random_init_relative_span: float,
    ds_ranges: Optional[Tuple[np.ndarray, np.ndarray]],
    active_threshold: float = 1e-3,
    random_range: str = "training",
) -> np.ndarray:
    """
    Return an array of shape (num_samples,) containing the mean relative
    deviation from GT, averaged over free parameters that are active
    (|p_true| > active_threshold).

    Metric per active free parameter:
        rel_dev_i = |p_init_i - p_true_i| / |p_true_i|

    Parameters with p_true ≈ 0 (absent oscillators) are excluded to avoid
    division-by-zero artefacts — they are truly uninformative for comparing
    how far the initialisation is from the correct value.
    """
    mean_devs: List[float] = []

    for idx in range(num_samples):
        wn, x_sys, x_osc, _y, _sample_idx = dataset[idx]
        (x_sys_init, x_sys_tr), (x_osc_init, x_osc_tr) = x_sys, x_osc

        p_true = torch.cat(
            [x_sys_tr["system1"], x_osc_tr["system1"][..., -3:].flatten(1)],
            dim=1,
        )  # (1, n_params)

        mask = create_masks_p(
            usecase,
            n_layers=1,
            n_sys_pars=x_sys_init,
            n_osc_pars=x_osc_init,
            dv=dataset.device,
        )["system1"]  # (1, n_params)  — 0 = free, 1 = fixed

        p_true_np = pt2np(p_true).reshape(-1)
        free_mask = pt2np(mask == 0).reshape(-1).astype(bool)
        # Only consider free parameters where the GT value is non-trivially non-zero
        active_mask = free_mask & (np.abs(p_true_np) > active_threshold)

        if scheme == "noisy":
            p_init = torch.where(
                mask == 0,
                p_true * (1.0 + (torch.rand_like(p_true) - 0.5) * noise_factor),
                p_true.clone(),
            )
        elif scheme == "random" and random_range == "training":
            p_gen = {1: p_gen_usecase1, 2: p_gen_usecase2}[int(usecase)]
            p_init = _random_init_training(
                p_true, mask, p_gen, dataset.dtype, dataset.device,
            )
        elif scheme == "random":
            p_init = _random_init(
                p_true, mask, ds_ranges,
                random_init_relative_span,
                dataset.dtype, dataset.device,
            )
        else:
            raise ValueError(f"Unknown scheme: {scheme!r}")

        p_init_np = pt2np(p_init).reshape(-1)
        rel_dev = np.abs(p_init_np - p_true_np) / np.abs(p_true_np)

        if active_mask.sum() > 0:
            mean_devs.append(float(np.mean(rel_dev[active_mask])))
        else:
            mean_devs.append(np.nan)

    return np.array(mean_devs)


# ─────────────────────────────────────────────────────────────────────────────
# plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_p_init_distributions(
    dataset: SimulationDatasetBatch,
    num_samples: int = 100,
    noise_factor: float = 0.20,
    random_init_relative_span: float = 0.25,
    n_bins: int = 40,
    output_path: str = "research/lma/p_init_deviation.png",
    random_seed: Optional[int] = None,
    random_range: str = "training",
) -> None:
    """
    Generate and save a two-subfigure comparison of the mean relative
    initial-parameter deviation for the noisy and random schemes.
    """
    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        log.info("Random seed: %d", random_seed)

    n = min(num_samples, len(dataset))
    log.info("Collecting deviations for %d samples…", n)

    ds_ranges = _dataset_param_ranges(dataset, "system1")
    if ds_ranges is not None:
        log.debug("Dataset ranges computed for %d parameters.", len(ds_ranges[0]))

    # Collect deviations — re-seed before each scheme so comparisons are fair
    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
    dev_noisy = _collect_p_init_deviations(
        dataset, "noisy", n, noise_factor, random_init_relative_span, ds_ranges,
        random_range=random_range,
    )

    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
    dev_random = _collect_p_init_deviations(
        dataset, "random", n, noise_factor, random_init_relative_span, ds_ranges,
        random_range=random_range,
    )

    # ── figure ────────────────────────────────────────────────────────────────
    # Metric: |p_init - p_true| / |p_true|, averaged over active free params.
    # "Active" = free AND |p_true| > threshold (excludes absent oscillators).
    # A value of 10% means the initial guess is 10% away from the GT value.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False, tight_layout=True)

    noisy_label  = f"Noisy  (±{noise_factor * 50:.0f}% of true value)"
    random_label = (
        "Random (training bounds)" if random_range == "training"
        else f"Random (dataset-wide bounds, span fallback={random_init_relative_span:.2f})"
    )
    scheme_data = [
        ("noisy",  dev_noisy,  "#2196F3", noisy_label),
        ("random", dev_random, "#FF5722", random_label),
    ]

    for ax, (scheme, devs, colour, label) in zip(axes, scheme_data):
        devs_pct = devs[~np.isnan(devs)] * 100.0  # relative deviation in %

        # clip display at 99th percentile to avoid long-tail distortion
        p99   = np.percentile(devs_pct, 99)
        edges = np.linspace(0.0, max(p99, 1.0), n_bins + 1)

        counts, _ = np.histogram(devs_pct, bins=edges)
        widths = np.diff(edges)
        ax.bar(
            edges[:-1], counts, width=widths, align="edge",
            color=colour, alpha=0.75, edgecolor="white", linewidth=0.5,
        )

        mu   = float(np.mean(devs_pct))
        med  = float(np.median(devs_pct))
        sd   = float(np.std(devs_pct))
        p95  = float(np.percentile(devs_pct, 95))

        ax.axvline(mu,  color="black",  lw=1.5, linestyle="--", label=f"Mean   {mu:.1f}%")
        ax.axvline(med, color="grey",   lw=1.2, linestyle=":",  label=f"Median {med:.1f}%")
        ax.axvline(p95, color="crimson",lw=1.2, linestyle="-.", label=f"P95    {p95:.1f}%")

        n_valid = int(np.sum(~np.isnan(devs)))
        ax.set(
            xlabel="Mean relative deviation from GT  [%]  =  |p_init − p_true| / |p_true|",
            ylabel="Number of samples",
            title=f"{label}\nn={n_valid}   mean={mu:.1f}%   std={sd:.1f}%",
        )
        ax.legend(fontsize=9)

    seed_tag = f"seed={random_seed}" if random_seed is not None else "no seed"
    fig.suptitle(
      f"Distribution of initial-parameter relative deviation from GT  —  {seed_tag}\n"
        f"(averaged over active free parameters only,  |p_true| > threshold)",
        fontsize=11, y=1.02,
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Figure saved → %s", output_path)

    # ── export CSV ────────────────────────────────────────────────────────────
    import pandas as pd
    csv_path = os.path.splitext(os.path.abspath(output_path))[0] + "_data.csv"
    df = pd.DataFrame({
        "sample_idx":           np.arange(n),
        "noisy_rel_dev_pct":    dev_noisy  * 100.0,
        "random_rel_dev_pct":   dev_random * 100.0,
    })
    df.to_csv(csv_path, index=False, float_format="%.6f")
    log.info("Data  saved → %s", csv_path)
    
    # ── console summary ───────────────────────────────────────────────────────
    for name, devs in [("Noisy ", dev_noisy), ("Random", dev_random)]:
        d = devs[~np.isnan(devs)] * 100.0
        log.info(
            "%s | mean=%.1f%%  std=%.1f%%  median=%.1f%%  p95=%.1f%%  (|p_init−p_true|/|p_true|)",
            name, np.mean(d), np.std(d), np.median(d), np.percentile(d, 95),
        )
    if ds_ranges is None:
        log.warning(
            "ds_ranges=None: random scheme used p_true-centred fallback ranges. "
            "Dataset may not have data_x_sys_tr."
        )
    else:
        log.info("ds_ranges: %d parameters from dataset training data.", len(ds_ranges[0]))


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualise p_init deviation for noisy vs random LMA schemes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dataset",
        default={1: "usecase1/case1_test_n2_10_nwsn", 2: "usecase2/case2_test_n2_10_wsn"}.get(usecase),
        help="Dataset name passed to load_sim_ds (default: LMA_DATASET of paper/common.sh)",
    )
    p.add_argument("--samples",  default=-1,   type=int,   help="Number of samples to analyse (-1: all)")
    p.add_argument("--noise",    default=0.20, type=float, help="Noise factor for the noisy scheme")
    p.add_argument("--span",     default=0.25, type=float, dest="span",
                   help="Relative span for the random scheme (_random_init)")
    p.add_argument("--random-range", default="training", choices=["training", "dataset"], dest="random_range",
                   help="'training': simulation bounds (as paper/lma.sh); 'dataset': legacy dataset min/max")
    p.add_argument("--bins",     default=40,   type=int,   help="Number of histogram bins")
    p.add_argument("--seed",     default=None, type=int,   help="Random seed (for reproducibility)")
    p.add_argument(
        "--output",
        default="research/lma/p_init_deviation.png",
        help="Output figure path",
    )
    p.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"], dest="log_level",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s | usecase: %s", device, usecase)
    log.info("Loading dataset: %s", args.dataset)
    dataset = load_sim_ds(args.dataset, y_return_mode="noised")

    plot_p_init_distributions(
        dataset                  = dataset,
        num_samples              = len(dataset) if args.samples == -1 else args.samples,
        noise_factor             = args.noise,
        random_init_relative_span= args.span,
        n_bins                   = args.bins,
        output_path              = args.output,
        random_seed              = args.seed,
        random_range             = args.random_range,
    )
