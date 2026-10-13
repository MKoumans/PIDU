import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


EPOCH_PATTERN = re.compile(r"^Epoch\s+(\d+)/(\d+):")
VAL_PATTERN = re.compile(r"^\s*-\s*val:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
EXP_PATTERN = re.compile(r"^\s*-\s*exp:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")


def classify_run_group(filename: str) -> Optional[str]:
    """Classify run logs into one of: wpg, wp, baseline.

    Expected filename endings:
      - ...-wpg.out
      - ...-wp.out
      - ...-.out
    """
    if filename.endswith("-wpg.out"):
        return "wpg"
    if filename.endswith("-wp.out"):
        return "wp"
    if filename.endswith("-base.out"):
        return "baseline"
    return None


def read_epoch_losses(log_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read epoch, validation loss and experimental loss from a single run log."""
    epoch_to_val: Dict[int, float] = {}
    epoch_to_exp: Dict[int, float] = {}

    current_epoch: Optional[int] = None

    with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")

            epoch_match = EPOCH_PATTERN.match(line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                continue

            if current_epoch is None:
                continue

            val_match = VAL_PATTERN.match(line)
            if val_match:
                epoch_to_val[current_epoch] = float(val_match.group(1))
                continue

            exp_match = EXP_PATTERN.match(line)
            if exp_match:
                epoch_to_exp[current_epoch] = float(exp_match.group(1))
                continue

    if not epoch_to_val and not epoch_to_exp:
        return np.array([]), np.array([]), np.array([])

    epochs = sorted(set(epoch_to_val.keys()) | set(epoch_to_exp.keys()))
    val_losses = np.array([epoch_to_val.get(epoch, np.nan) for epoch in epochs], dtype=float)
    exp_losses = np.array([epoch_to_exp.get(epoch, np.nan) for epoch in epochs], dtype=float)

    return np.array(epochs, dtype=int), val_losses, exp_losses


def aggregate_mean_std(curves: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate multiple (epoch, loss) curves to mean/std by epoch.

    Returns:
        epochs, mean, std, counts
    """
    if not curves:
        return np.array([]), np.array([]), np.array([]), np.array([])

    all_epochs = sorted({int(epoch) for epochs, _ in curves for epoch in epochs})
    matrix = np.full((len(curves), len(all_epochs)), np.nan, dtype=float)

    epoch_to_idx = {epoch: idx for idx, epoch in enumerate(all_epochs)}
    for run_idx, (epochs, losses) in enumerate(curves):
        for epoch, loss in zip(epochs, losses):
            matrix[run_idx, epoch_to_idx[int(epoch)]] = loss

    with np.errstate(invalid="ignore"):
        mean = np.nanmean(matrix, axis=0)
        std = np.nanstd(matrix, axis=0)
    counts = np.sum(~np.isnan(matrix), axis=0)

    return np.array(all_epochs, dtype=int), mean, std, counts


def find_log_files(input_dir: Path, pattern: str, recursive: bool) -> List[Path]:
    if recursive:
        return sorted(input_dir.rglob(pattern))
    return sorted(input_dir.glob(pattern))


def plot_grouped_curves(
    grouped_runs: Dict[str, Dict[str, List[Tuple[np.ndarray, np.ndarray]]]],
    output_path: Path,
    log_scale: bool = False,
    show: bool = False,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

    group_order = ["wpg", "wp", "baseline"]
    group_labels = {
        "wpg": "wpg",
        "wp": "wp",
        "baseline": "baseline (-)",
    }

    metric_to_axis = {
        "val": axes[0],
        "exp": axes[1],
    }
    metric_titles = {
        "val": "Validation loss",
        "exp": "Experimental loss",
    }

    for metric, axis in metric_to_axis.items():
        for group in group_order:
            curves = grouped_runs[group][metric]
            if not curves:
                continue

            epochs, mean, std, _ = aggregate_mean_std(curves)
            if epochs.size == 0:
                continue

            run_count = len(curves)
            label = f"{group_labels[group]} (n={run_count})"

            axis.plot(epochs, mean, linewidth=2.0, label=label)
            axis.fill_between(epochs, mean - std, mean + std, alpha=0.2)

        axis.set_title(metric_titles[metric])
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss")
        if log_scale:
            axis.set_yscale("log")
        axis.grid(True, alpha=0.3)
        axis.legend()

    fig.suptitle("Training curves from Snellius run logs (mean ± std)")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure to: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    output_path = Path(args.output)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    files = find_log_files(input_dir, args.glob, args.recursive)
    if not files:
        raise FileNotFoundError(f"No files matched pattern '{args.glob}' in {input_dir}")

    grouped_runs = {
        "wpg": {"val": [], "exp": []},
        "wp": {"val": [], "exp": []},
        "baseline": {"val": [], "exp": []},
    }

    skipped = 0
    for log_file in files:
        group = classify_run_group(log_file.name)
        if group is None:
            skipped += 1
            continue

        epochs, val_losses, exp_losses = read_epoch_losses(log_file)
        if epochs.size == 0:
            print(f"Warning: no epoch losses parsed in {log_file}")
            continue

        grouped_runs[group]["val"].append((epochs, val_losses))
        grouped_runs[group]["exp"].append((epochs, exp_losses))

    print("Parsed runs:")
    for group in ["wpg", "wp", "baseline"]:
        n_runs = len(grouped_runs[group]["val"])
        print(f"  - {group}: {n_runs} runs")
        if args.expected_runs and n_runs != args.expected_runs:
            print(f"    Warning: expected {args.expected_runs} runs, found {n_runs}")

    if all(len(grouped_runs[group]["val"]) == 0 for group in grouped_runs):
        raise RuntimeError("No valid run logs were parsed for wpg/wp/baseline groups.")

    if skipped > 0:
        print(f"Skipped {skipped} files that do not match *-wpg.out, *-wp.out, or *-.out")

    plot_grouped_curves(
        grouped_runs=grouped_runs,
        output_path=output_path,
        log_scale=args.log_scale,
        show=args.show,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot mean±std training curves from Snellius .out run logs."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="snellius-out",
        help="Directory containing run log files",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="runs-case2-*.out",
        help="Glob pattern to select run logs",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search recursively in subfolders",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="research/legacy/figures/snellius_training_curves_case2.png",
        help="Output image path",
    )
    parser.add_argument(
        "--expected-runs",
        type=int,
        default=8,
        help="Expected number of runs per group (for warnings only)",
    )
    parser.add_argument(
        "--log-scale",
        action="store_true",
        help="Use logarithmic y-axis",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plot interactively",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
