"""Summary figure: MAPE versus inference time per spectrum for all methods.

Data loading (load_summary) is separate from plotting (plot_summary); rerun after
the final results by replacing scripts/figure_summary_data.csv:

    python scripts/figure_summary.py [--data CSV] [--output-dir DIR]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / 'figure_summary_data.csv'
DEFAULT_OUT = HERE.parent / 'outputs' / 'fig-summary'

CASE_TITLES = {1: '(a) Case 1: numerical', 2: '(b) Case 2: experimental'}
XLIM, YLIM = (3e-3, 1e5), (2.0, 100.0)
CLIP_Y = 70.0  # MAPE above YLIM[1] is drawn as an up-triangle here, labelled with its value

# Okabe-Ito based (colourblind-safe). TODO: swap for the manuscript palette of fig-case1/fig-case2.
# family: iterative fitting = circles, amortised = squares. 'fill' encodes prior knowledge for TRF.
STYLE = {
    'TRF (baseline)': dict(color='#5B5B5B', marker='o', fill=0.0, size=4.5, dx=-6, dy=0, ha='right'),
    'TRF (random)':   dict(color='#5B5B5B', marker='o', fill=0.5, size=4.5, dx=-6, dy=0, ha='right'),
    'TRF (noisy)':    dict(color='#5B5B5B', marker='o', fill=1.0, size=4.5, dx=-6, dy=0, ha='right'),
    'CNN':            dict(color='#0072B2', marker='s', fill=1.0, size=4.5, dx=6, dy=-1, ha='left'),
    'DU':             dict(color='#009E73', marker='D', fill=1.0, size=4.0, dx=6, dy=1, ha='left'),
    'PIDU (ours)':    dict(color='#D55E00', marker='*', fill=1.0, size=9.5, dx=7, dy=-2, ha='left'),
}

# label offsets (dx, dy in points, ha) per case where the default in STYLE would collide
LABEL = {
    1: {'TRF (baseline)': (-8, 0, 'right'), 'TRF (random)': (-6, 10, 'right'), 'TRF (noisy)': (10, -12, 'left'),
        'DU': (8, 14, 'left'), 'CNN': (0, 12, 'center'), 'PIDU (ours)': (0, -14, 'center')},
    2: {'TRF (random)': (-8, 12, 'right'), 'TRF (noisy)': (14, -22, 'right'), 'DU': (-6, 12, 'right'),
        'CNN': (-6, 12, 'right'), 'PIDU (ours)': (-8, -14, 'right')},
}


def load_summary(path: Path = DEFAULT_DATA) -> pd.DataFrame:
    """Read the summary CSV and convert MAPE to percent (columns: mape_pct, mape_pct_std)."""
    df = pd.read_csv(path, comment='#')
    scale = df['mape_unit'].map({'fraction': 100.0, 'percent': 1.0})
    if scale.isna().any():
        raise ValueError("mape_unit must be 'fraction' or 'percent'")
    df['mape_pct'] = df['mape_mean'] * scale
    df['mape_pct_std'] = df['mape_std'] * scale
    return df


def _errbar(mean, std):
    """Asymmetric (lower, upper) error for a log axis; the lower arm never reaches zero."""
    if pd.isna(std):
        return None
    return np.array([[min(std, 0.9 * mean)], [std]])


def plot_summary(df: pd.DataFrame, out_dir: Path) -> None:
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.linewidth': 0.6,
    })
    fig, axes = plt.subplots(1, 2, figsize=(16 / 2.54, 6.5 / 2.54), sharey=True)

    for ax, case in zip(axes, (1, 2)):
        ax.set_xscale('log'), ax.set_yscale('log')
        ax.set_xlim(*XLIM), ax.set_ylim(*YLIM)
        ax.grid(True, which='major', color='0.88', lw=0.5), ax.set_axisbelow(True)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        ax.set_xlabel('Inference time per spectrum (ms)')
        ax.text(0.5, 1.04, CASE_TITLES[case], transform=ax.transAxes, fontweight='bold', va='bottom', ha='center')

        for _, r in df[df['case'] == case].iterrows():
            st = STYLE[r['method']]
            face = (*plt.matplotlib.colors.to_rgb(st['color']), st['fill'])  # opacity = prior knowledge needed
            ours = r['method'].startswith('PIDU')
            ph = str(r.get('source', 'runs')).startswith('placeholder')
            name = r['method'] + (' \u2020' if ph else '')
            x, y = r['time_b1_ms'], r['mape_pct']
            clipped = y > YLIM[1]
            if clipped:
                val = f'{y:.0f}' if y < 1e4 else f'${y / 10 ** int(np.log10(y)):.1f}\\times10^{{{int(np.log10(y))}}}$'
                label = f"{name}\n{val} %"
                ax.plot(x, CLIP_Y, '^', ms=5, mfc='white', mec=st['color'], mew=0.9, zorder=4)
                ax.annotate(label, (x, CLIP_Y), xytext=(-6, 0), textcoords='offset points', ha='right',
                            va='center', fontsize=7.5, linespacing=1.1, zorder=6)
                continue
            faint = r['mape_pct_std'] > y
            ax.errorbar(x, y, xerr=_errbar(x, r['time_b1_std_ms']), yerr=_errbar(y, r['mape_pct_std']),
                        fmt='none', ecolor=(*plt.matplotlib.colors.to_rgb(st['color']), 0.35 if faint else 1.0), elinewidth=0.7, capsize=1.5, zorder=3)
            if ph:  # placeholder (not from the runs): dashed outline
                ax.scatter(x, y, marker=st['marker'], s=(1.35 * st['size']) ** 2, facecolors=[face], edgecolors=st['color'],
                           linewidths=0.9, linestyles='--', zorder=4)
            else:
                ax.plot(x, y, st['marker'], ms=st['size'], mfc=face, mec=st['color'], mew=0.9,
                        ls='none', zorder=5 if ours else 4)
            if pd.notna(r['time_batch_ms']):
                x2 = r['time_batch_ms']
                ax.plot([x, x2], [y, y], '-', color=st['color'], lw=0.5, alpha=0.6, zorder=2)
                ax.errorbar(x2, y, xerr=_errbar(x2, r['time_batch_std_ms']), yerr=_errbar(y, r['mape_pct_std']),
                            fmt='none', ecolor=st['color'], elinewidth=0.7, capsize=1.5, zorder=3)
                ax.plot(x2, y, st['marker'], ms=st['size'], mfc='white', mec=st['color'], mew=0.9, zorder=4)
            dx, dy, ha, *leader = LABEL[case].get(r['method'], (st['dx'], st['dy'], st['ha']))
            ax.annotate(name, (x, y), xytext=(dx, dy), textcoords='offset points',
                        ha=ha, va='center', fontsize=7.5,
                        arrowprops=dict(arrowstyle='-', color='0.5', lw=0.5, shrinkA=1, shrinkB=6) if leader else None,
                        fontweight='bold' if ours else 'normal', color='black', zorder=6)


    axes[0].set_ylabel('MAPE (%)')
    footnote = df['source'].astype(str).str.startswith('placeholder').any()
    if footnote:
        fig.text(0.5, 0.005, '\u2020 placeholder (Table 1 mean, no spread; not from the rerun) - see ASSUMPTIONS.md',
                 ha='center', va='bottom', fontsize=6, color='0.4')
    fig.tight_layout(w_pad=1.2, rect=(0, 0.04 if footnote else 0, 1, 1))
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / 'figure-summary.pdf')
    fig.savefig(out_dir / 'figure-summary.png', dpi=300)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--data', type=Path, default=DEFAULT_DATA)
    ap.add_argument('--output-dir', type=Path, default=DEFAULT_OUT)
    a = ap.parse_args()
    plot_summary(load_summary(a.data), a.output_dir)
    print(f'Wrote {a.output_dir}/figure-summary.pdf and .png')
