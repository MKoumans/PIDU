"""
Run LMA benchmark for all initialization schemes on the full dataset.

Usage:
    python research/lma/run_all_schemes.py
    python research/lma/run_all_schemes.py --dataset usecase1/case1_test_n2_10 --backend numpy
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import List

import torch


from pidu.config import usecase
from pidu.data.simulation import load_sim_ds
from pidu.baselines.lma import benchmark_lma

log = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LMA for baseline/noisy/random on full dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset",
        default=f"usecase{usecase}/case{usecase}_test_n2_10_wsn",
        help="Dataset name passed to load_sim_ds",
    )
    parser.add_argument("--samples",
        type=int,
        default=-1,
        help="Number of samples (-1 means full dataset)",
    )
    parser.add_argument("--max-iter", type=int, default=200, dest="max_iter")
    parser.add_argument("--noise", type=float, default=0.20)
    parser.add_argument("--backend", choices=["numpy", "torch"], default="numpy")
    parser.add_argument("--all-osc", action="store_true", dest="all_osc")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for the noisy/random initialisation")
    parser.add_argument("--random-range", choices=["training", "dataset"], default="training", dest="random_range",
        help="'training': simulation bounds; 'dataset': min/max of the evaluated dataset (legacy)",
    )
    parser.add_argument("--schemes", nargs="+", default=["baseline", "noisy", "random"],
        choices=["baseline", "noisy", "random"], help="Initialisation schemes to run",
    )
    parser.add_argument("--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        dest="log_level",
    )
    return parser.parse_args()


def main() -> None:
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

    schemes: List[str] = args.schemes
    all_results = {}

    for scheme in schemes:
        log.info("\n===== Running scheme=%s | samples=%s =====", scheme, args.samples)
        results = benchmark_lma(
            dataset=dataset,
            num_samples=args.samples,
            max_iterations=args.max_iter,
            initialization_scheme=scheme,
            noise_factor=args.noise,
            all_osc=args.all_osc,
            backend=args.backend,
            suppress_non_osc=True, # Suppress non-oscillatory samples for all schemes
            bound_free_relative_50=True, # Use relative free bounds for all schemes except baseline
            random_seed=args.seed,
            random_range=args.random_range,
        )
        all_results[scheme] = results
        log.info(
            "Finished %s | time/sample=%.2f ms | success=%.1f%% | cost=%.4e",
            scheme,
            results["time_per_sample_ms"],
            100.0 * results["success_rate"],
            results["cost_mean"],
        )

    log.info("\n===== Summary =====")
    for scheme in schemes:
        r = all_results[scheme]
        log.info(
            "%s | samples=%d | time/sample=%.2f ms | success=%.1f%% | cost=%.4e",
            scheme,
            r["num_samples"],
            r["time_per_sample_ms"],
            100.0 * r["success_rate"],
            r["cost_mean"],
        )


if __name__ == "__main__":
    main()
