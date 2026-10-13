"""Fig. S5: initial-parameter deviation |p_init - p_true| / |p_true| of the noisy and
random TRF initialisation schemes, (a) use case 1 and (b) use case 2.

Reads the per-sample CSVs of research/lma/visualise_p_init.py (one per use case):

    python scripts/figure_p_init.py [--case1 CSV] [--case2 CSV] [--output-dir DIR]

Writes fig_s5_p_init.{png,pdf} and the binned counts per panel and scheme
(bin centre, count, cumulative sum, cumulative percent) as fig_s5_p_init_hist.csv.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = {c: ROOT / 'outputs' / f'case{c}' / 'figures' / 'p_init_deviation_data.csv' for c in (1, 2)}
DEFAULT_OUT = ROOT / 'outputs' / 'fig-s5'

# column, label, colour, bin width (fraction); noisy drawn first so random sits on top as in the SI
SCHEMES = [
    ('noisy_rel_dev_pct', 'Noisy', '#9DC3E6', 0.005),
    ('random_rel_dev_pct', 'Random', '#1F4E79', 0.01),
]
PANELS = {1: dict(title='Case1', tag='(a)', xlim=(0.0, 0.8)),
          2: dict(title='Case2', tag='(b)', xlim=(0.0, 0.4))}


def load_deviations(path: Path) -> dict:
    """Per-scheme deviations as a fraction, NaN rows (no active free parameter) dropped."""
    df = pd.read_csv(path)
    return {col: df[col].dropna().to_numpy() / 100.0 for col, *_ in SCHEMES}


def histogram(x: np.ndarray, width: float) -> pd.DataFrame:
    edges = np.arange(np.floor(x.min() / width) * width, x.max() + width, width)
    counts, edges = np.histogram(x, bins=edges)
    cum = np.cumsum(counts)
    return pd.DataFrame({'bin_center': edges[:-1] + width / 2, 'count': counts,
                         'cumulative_sum': cum, 'cumulative_percent': 100.0 * cum / cum[-1]})


def plot_p_init(data: dict, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), tight_layout=True)
    tables = []
    for ax, (case, devs) in zip(axes, data.items()):
        panel = PANELS[case]
        for col, label, colour, width in SCHEMES:
            x = devs[col]
            h = histogram(x, width)
            ax.bar(h.bin_center, h['count'], width=width, color=colour, edgecolor='black',
                   linewidth=0.3, label=label)
            # normal distribution curve scaled to counts, as Origin's histogram overlay
            xs = np.linspace(*panel['xlim'], 400)
            ax.plot(xs, norm.pdf(xs, x.mean(), x.std()) * x.size * width, color=colour, lw=1.0)
            tables.append(h.assign(case=case, scheme=label.lower()))
        ax.set(xlim=panel['xlim'], ylim=(0, None), title=panel['title'],
               xlabel=r'$|p_{\mathrm{init}} - p_{\mathrm{true}}|\, /\, |p_{\mathrm{true}}|$', ylabel='Count')
        ax.text(0.97, 0.95, panel['tag'], transform=ax.transAxes, ha='right', va='top')
        ax.legend(loc='upper right', bbox_to_anchor=(0.9, 1.0), frameon=False, fontsize=8)

    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(output_dir / f'fig_s5_p_init.{ext}', dpi=300)
    plt.close(fig)
    cols = ['case', 'scheme', 'bin_center', 'count', 'cumulative_sum', 'cumulative_percent']
    pd.concat(tables)[cols].to_csv(output_dir / 'fig_s5_p_init_hist.csv', index=False, float_format='%.6g')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--case1', type=Path, default=DEFAULT_CSV[1])
    p.add_argument('--case2', type=Path, default=DEFAULT_CSV[2])
    p.add_argument('--output-dir', type=Path, default=DEFAULT_OUT)
    args = p.parse_args()
    data = {1: load_deviations(args.case1), 2: load_deviations(args.case2)}
    for case, devs in data.items():
        for col, label, *_ in SCHEMES:
            x = devs[col]
            print(f'case {case} {label:6s} n={x.size:4d} mean={x.mean():.3f} median={np.median(x):.3f}')
    plot_p_init(data, args.output_dir)
    print(f'Saved -> {args.output_dir}')


if __name__ == '__main__':
    main()
