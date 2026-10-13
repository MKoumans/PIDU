"""
Class-based residual analysis across multiple runs and methods.

For each base-pattern (method), for each parameter, a long list of residual
errors is collected over all runs.  That data is exported to Excel and
visualised as distribution plots.

Usage examples
--------------
python research/legacy/analyse_residuals.py --usecase 2 --trail 9 --export
python research/legacy/analyse_residuals.py --usecase 2 --trail 9 --export --show
python research/legacy/analyse_residuals.py --usecase 2 --trail 9 --export \
    --base-patterns loss_CNN loss_DU loss_PIDU loss_lma_baseline \
    --params w0 wp g
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Plotting defaults
# ---------------------------------------------------------------------------
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 6)
_PALETTE = "Set2"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PARAM_GROUPS: Dict[str, str] = {
    "eps_inf": "eps_inf",
    "w0":      "w0",
    "wp":      "wp",
    "g":       "g",
}


def _group_of(col: str) -> Optional[str]:
    """Return the canonical group name for a column, or None if unrecognised."""
    c = col.lower()
    if "eps_inf" in c:
        return "eps_inf"
    if c.startswith("w0") or "_w0" in c:
        return "w0"
    if c.startswith("wp") or "_wp" in c:
        return "wp"
    if c == "g" or c.startswith("g_"):
        return "g"
    return None


def _short_name(pattern: str) -> str:
    """loss_CNN_0_009 → CNN,  loss_lma_baseline → lma_baseline."""
    s = pattern.removeprefix("loss_")
    return s


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------


class ResidualAnalyser:
    """
    Load, aggregate, export, and visualise residual distributions.

    Parameters
    ----------
    base_folder:
        Root outputs directory, e.g.  outputs/case2
    trail:
        Trail index used to build indexed folder names.
    num_runs:
        Maximum number of indexed runs to search per pattern.
    dataset_type:
        Sub-folder inside each run, e.g. "test_test" or "test_exp".
        Falls back to "test_test" automatically when the requested split is
        absent.
    params:
        Optional list of parameter *group* names (``w0``, ``wp``, ``g``,
        ``eps_inf``) to include.  ``None`` means include all found groups.
    distribution_mode:
        ``'grouped'``  – pool all parameters in a group (w0_1, w0_2 → w0).
        ``'split'``    – treat every individual column as its own group.
    """

    def __init__(
        self,
        base_folder: Path,
        trail: int = 0,
        num_runs: int = 8,
        dataset_type: str = "test_test",
        params: Optional[Sequence[str]] = None,
        distribution_mode: str = "grouped",
    ) -> None:
        self.base_folder = Path(base_folder)
        self.trail = trail
        self.num_runs = num_runs
        self.dataset_type = dataset_type
        self.params_filter: Optional[List[str]] = list(params) if params else None
        self.distribution_mode = distribution_mode

        # After load_all() these are populated:
        # residuals[pattern][param_group] = List[float]
        self.residuals: Dict[str, Dict[str, List[float]]] = {}
        # Ordered list of patterns that were actually loaded
        self.patterns: List[str] = []
        # Set of all parameter groups found across all patterns
        self.param_groups: List[str] = []

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_file(self, run_folder: Path) -> Optional[pd.DataFrame]:
        """Load res_dis.xlsx from a run folder, with fallback to test_test."""
        candidates = [self.dataset_type]
        if self.dataset_type != "test_test":
            candidates.append("test_test")

        for split in candidates:
            fp = run_folder / split / "res_dis.xlsx"
            if not fp.exists():
                continue
            if split != self.dataset_type:
                print(
                    f"  [fallback] '{split}' used instead of '{self.dataset_type}'"
                    f" for {run_folder.name}"
                )
            try:
                return pd.read_excel(fp)
            except Exception as exc:
                print(f"  [error] could not read {fp}: {exc}")
                return None

        return None

    def _load_pattern(self, pattern: str) -> Dict[str, List[float]]:
        """
        Load all runs for *pattern* and return a dict
        ``{param_group: [residual, ...]}``.
        """
        raw: Dict[str, List[float]] = {}
        found_indexed = False

        for run_id in range(self.num_runs):
            folder = self.base_folder / f"{pattern}_{run_id}_{self.trail:03d}"
            if not folder.exists():
                continue
            found_indexed = True
            df = self._load_file(folder)
            if df is not None:
                self._accumulate(df, raw)

        if not found_indexed:
            # Try bare pattern folder (LMA-style)
            folder = self.base_folder / pattern
            if folder.exists():
                df = self._load_file(folder)
                if df is not None:
                    self._accumulate(df, raw)
                    found_indexed = True  # suppress warning below

        if not found_indexed:
            print(f"  [warning] no folders found for pattern '{pattern}'")

        return raw

    def _accumulate(self, df: pd.DataFrame, raw: Dict[str, List[float]]) -> None:
        """
        Append values from *df* columns into *raw*, respecting the
        distribution_mode and params_filter.
        """
        for col in df.columns:
            group = self._resolve_group(col)
            if group is None:
                continue
            if self.params_filter and group not in self.params_filter:
                continue
            values = df[col].dropna().tolist()
            raw.setdefault(group, []).extend(values)

    def _resolve_group(self, col: str) -> Optional[str]:
        """Map a column name to its group key (respects distribution_mode).

        System parameters (columns ending with ``_before`` or ``_after``) are
        always kept as their own individual group, irrespective of
        *distribution_mode*.  All other sub-parameters (e.g. w0_1, w0_2) are
        pooled in ``'grouped'`` mode or kept separate in ``'split'`` mode.
        """
        family = _group_of(col)
        if family is None:
            return None

        c = col.lower()
        # System parameters always get their own group
        if c.endswith("_before") or c.endswith("_after"):
            return col

        if self.distribution_mode == "split":
            return col
        # grouped mode: pool oscillator indices under the family name
        return family

    def load_all(self, patterns: Sequence[str]) -> "ResidualAnalyser":
        """
        Load residuals for every pattern.  Patterns that produce no data are
        silently skipped.

        Returns self for chaining.
        """
        print(f"\nBase folder : {self.base_folder}")
        print(f"Dataset type: {self.dataset_type}")
        print(f"Trail       : {self.trail}")
        print(f"Num runs    : {self.num_runs}")
        print("=" * 60)

        all_groups: set[str] = set()
        for pattern in patterns:
            print(f"\nLoading '{pattern}' …")
            raw = self._load_pattern(pattern)
            if not raw:
                print(f"  [skip] no residuals found for '{pattern}'")
                continue
            self.residuals[pattern] = raw
            self.patterns.append(pattern)
            all_groups.update(raw.keys())
            sizes = {g: len(v) for g, v in raw.items()}
            print(f"  loaded groups: {sizes}")

        # Sort groups in a canonical order.
        # _before/_after columns sort immediately after their parent family.
        families = ["eps_inf", "w0", "wp", "g"]

        def _sort_key(g: str):
            lo = g.lower()
            # Determine which family this group belongs to
            for i, fam in enumerate(families):
                if lo == fam or lo.startswith(fam + "_"):
                    # _before < everything < _after, then alphabetical
                    if lo.endswith("_before"):
                        sub = 0
                    elif lo.endswith("_after"):
                        sub = 2
                    else:
                        sub = 1
                    return (i, sub, g)
            return (len(families), 1, g)

        self.param_groups = sorted(all_groups, key=_sort_key)

        print(f"\nPatterns loaded : {self.patterns}")
        print(f"Parameter groups: {self.param_groups}")
        return self

    # ------------------------------------------------------------------
    # Build DataFrame helper
    # ------------------------------------------------------------------

    def to_dataframe(self, group: str) -> pd.DataFrame:
        """
        Return a DataFrame where each column is a pattern/method and each
        row is a residual value (NaN-padded to equal length).
        """
        series: Dict[str, pd.Series] = {}
        for pattern in self.patterns:
            values = self.residuals.get(pattern, {}).get(group, [])
            if values:
                series[_short_name(pattern)] = pd.Series(values, dtype=float)
        if not series:
            return pd.DataFrame()
        return pd.concat(series, axis=1)

    # ------------------------------------------------------------------
    # Excel export
    # ------------------------------------------------------------------

    def export_excel(self, save_path: Path) -> None:
        """
        Export one sheet per parameter group to *save_path*.

        Each sheet has one column per method containing the raw residual list
        (NaN-padded so all columns fit the same number of rows).
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"\nExporting residuals to: {save_path}")
        written = 0
        with pd.ExcelWriter(save_path, engine="openpyxl") as writer:
            for group in self.param_groups:
                df = self.to_dataframe(group)
                if df.empty:
                    print(f"  [skip] no data for group '{group}'")
                    continue
                sheet = group.replace("/", "_")[:31]
                df.to_excel(writer, sheet_name=sheet, index=False)
                written += 1
                print(f"  sheet '{sheet}': {df.shape[0]} rows × {df.shape[1]} cols")

        print(f"Wrote {written} sheets.")

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_distributions(
        self,
        save_dir: Optional[Path] = None,
        show: bool = False,
        dataset_tag: str = "",
        yscale: str = "linear",
    ) -> None:
        """
        For each parameter group create a violin + box plot comparing all methods.

        Parameters
        ----------
        save_dir:
            Directory to save PNG files.  Skipped when *None*.
        show:
            Display interactively with plt.show().
        dataset_tag:
            String appended to the saved filename (e.g. "test_test").
        yscale:
            Y-axis scale: ``'linear'`` or ``'log'``.
        """
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        palette = sns.color_palette(_PALETTE, len(self.patterns))

        for group in self.param_groups:
            df = self.to_dataframe(group)
            if df.empty:
                continue

            # Melt to long form for seaborn
            long = df.melt(var_name="Method", value_name="Residual").dropna()

            fig, ax = plt.subplots(figsize=(max(8, 2 * len(self.patterns)), 6))
            sns.violinplot(
                data=long,
                x="Method",
                y="Residual",
                palette=palette,
                inner=None,
                linewidth=1.2,
                ax=ax,
            )
            sns.boxplot(
                data=long,
                x="Method",
                y="Residual",
                width=0.15,
                color="white",
                linewidth=1.2,
                flierprops={"marker": ".", "markersize": 3, "alpha": 0.4},
                ax=ax,
            )
            ax.set_title(
                f"Residual distribution – {group.upper()}"
                + (f"  [{dataset_tag}]" if dataset_tag else ""),
                fontsize=14,
                fontweight="bold",
            )
            ax.set_xlabel("Method", fontsize=12)
            ax.set_ylabel("Residual error", fontsize=12)
            ax.set_yscale(yscale)
            ax.grid(True, axis="y", alpha=0.4)
            plt.tight_layout()

            if save_dir:
                tag = f"_{dataset_tag}" if dataset_tag else ""
                fname = save_dir / f"dist_{group}{tag}.png"
                fig.savefig(fname, dpi=150, bbox_inches="tight")
                print(f"  saved: {fname.name}")

            if show:
                plt.show()
            else:
                plt.close(fig)

    def plot_summary(
        self,
        save_dir: Optional[Path] = None,
        show: bool = False,
        dataset_tag: str = "",
        yscale: str = "linear",
    ) -> None:
        """
        Bar chart of mean ± std for each parameter group and method.

        Parameters
        ----------
        save_dir:
            Directory to save PNG files.  Skipped when *None*.
        show:
            Display interactively with plt.show().
        dataset_tag:
            String appended to the saved filename.
        yscale:
            Y-axis scale: ``'linear'`` or ``'log'``.
        """
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        n_groups = len(self.param_groups)
        if n_groups == 0:
            print("[warning] no parameter groups to plot")
            return

        n_methods = len(self.patterns)
        width = 0.8 / max(n_methods, 1)
        x = np.arange(n_groups)
        palette = sns.color_palette(_PALETTE, n_methods)

        fig, axes = plt.subplots(1, 2, figsize=(max(10, 3 * n_groups), 6))

        mean_ax, std_ax = axes

        for m_idx, pattern in enumerate(self.patterns):
            name = _short_name(pattern)
            means, stds, medians = [], [], []
            for group in self.param_groups:
                vals = np.array(self.residuals.get(pattern, {}).get(group, []))
                if vals.size:
                    means.append(vals.mean())
                    stds.append(vals.std())
                    medians.append(np.median(vals))
                else:
                    means.append(np.nan)
                    stds.append(np.nan)
                    medians.append(np.nan)

            offset = (m_idx - n_methods / 2 + 0.5) * width
            color = palette[m_idx]

            mean_ax.bar(
                x + offset,
                means,
                width,
                label=name,
                color=color,
                alpha=0.85,
                yerr=stds,
                capsize=4,
                error_kw={"elinewidth": 1.2},
            )
            std_ax.bar(
                x + offset,
                stds,
                width,
                label=name,
                color=color,
                alpha=0.85,
            )

        for ax, title, ylabel in (
            (mean_ax, "Mean residual per parameter group", "Mean ± Std"),
            (std_ax,  "Std of residuals per parameter group", "Std"),
        ):
            ax.set_xticks(x)
            ax.set_xticklabels([g.upper() for g in self.param_groups], rotation=30, ha="right")
            ax.set_title(
                title + (f"  [{dataset_tag}]" if dataset_tag else ""),
                fontsize=12,
                fontweight="bold",
            )
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_yscale(yscale)
            ax.legend(fontsize=9)
            ax.grid(True, axis="y", alpha=0.4)

        plt.tight_layout()

        if save_dir:
            tag = f"_{dataset_tag}" if dataset_tag else ""
            fname = save_dir / f"summary{tag}.png"
            fig.savefig(fname, dpi=150, bbox_inches="tight")
            print(f"  saved: {fname.name}")

        if show:
            plt.show()
        else:
            plt.close(fig)

    def plot_all(
        self,
        save_dir: Optional[Path] = None,
        show: bool = False,
        dataset_tag: str = "",
        yscale: str = "linear",
    ) -> None:
        """Convenience: run both plot_distributions and plot_summary."""
        self.plot_distributions(save_dir=save_dir, show=show, dataset_tag=dataset_tag, yscale=yscale)
        self.plot_summary(save_dir=save_dir, show=show, dataset_tag=dataset_tag, yscale=yscale)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_DEFAULT_PATTERNS = [
    "loss_lma_baseline",
    "loss_lma_noisy",
    "loss_lma_random",
    "loss_CNN",
    "loss_DU",
    "loss_PIDU",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Analyse residual distributions across methods/runs.\n"
            "Loads res_dis.xlsx files, pools residuals per parameter, "
            "then exports Excel and/or plots."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--base-patterns",
        nargs="+",
        default=_DEFAULT_PATTERNS,
        metavar="PATTERN",
        help=(
            "Space-separated list of output folder prefixes.  "
            "Indexed runs (<prefix>_<id>_<trail:03d>) and bare folders "
            "(<prefix>) are both supported.  "
            f"Default: {_DEFAULT_PATTERNS}"
        ),
    )
    p.add_argument(
        "--usecase", type=int, default=1,
        help="Case number – determines outputs/case<N> path."
    )
    p.add_argument(
        "--trail", type=int, default=0,
        help="Trail index used in indexed folder names."
    )
    p.add_argument(
        "--num-runs", type=int, default=8,
        help="Maximum number of indexed runs to load per pattern."
    )
    p.add_argument(
        "--ds-type", type=str, default="test_test",
        metavar="SPLIT",
        help=(
            "Sub-folder to read inside each run "
            "(e.g. test_test, test_val, test_exp).  "
            "Falls back to test_test when the requested split is absent."
        ),
    )
    p.add_argument(
        "--params",
        nargs="+",
        default=None,
        metavar="GROUP",
        help=(
            "Parameter groups to include (eps_inf w0 wp g).  "
            "Default: all groups found in the data."
        ),
    )
    p.add_argument(
        "--distribution-mode",
        choices=["grouped", "split"],
        default="grouped",
        help=(
            "grouped: pool all w0_* columns into one 'w0' group.  "
            "split: keep each column separate."
        ),
    )
    p.add_argument(
        "--yscale",
        choices=["linear", "log"],
        default="linear",
        help="Y-axis scale for all plots (linear or log).  Default: linear.",
    )
    p.add_argument(
        "--export",
        action="store_true",
        help="Save Excel file and PNG plots to outputs/case<N>/analyse/residual_comparison/.",
    )
    p.add_argument(
        "--show", action="store_true",
        help="Show plots interactively (in addition to, or instead of, saving)."
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from pidu.paths import ROOT as project_root
    base_folder = project_root / "outputs" / f"case{args.usecase}"
    output_folder = (
        project_root / "outputs" / f"case{args.usecase}" / "analyse" / "residual_comparison"
    )

    analyser = ResidualAnalyser(
        base_folder=base_folder,
        trail=args.trail,
        num_runs=args.num_runs,
        dataset_type=args.ds_type,
        params=args.params,
        distribution_mode=args.distribution_mode,
    )
    analyser.load_all(args.base_patterns)

    if not analyser.patterns:
        print("\n[error] No data loaded – check --base-patterns and --usecase.")
        return

    dataset_tag = args.ds_type

    if args.export:
        excel_path = output_folder / f"residual_distribution_case{args.usecase}.xlsx"
        analyser.export_excel(excel_path)
        analyser.plot_all(save_dir=output_folder, show=args.show, dataset_tag=dataset_tag, yscale=args.yscale)
    elif args.show:
        analyser.plot_all(save_dir=None, show=True, dataset_tag=dataset_tag, yscale=args.yscale)
    else:
        print(
            "\n[info] Neither --export nor --show specified – data loaded but nothing output.\n"
            "       Add --export to save Excel + PNG files, or --show to view interactively."
        )


if __name__ == "__main__":
    main()
