"""Copy the timings of a paper/timings.sh run into paper/timings/, the timing set of Fig. 2 and Table 1.

    python scripts/export_timings.py [--run outputs/timings_final] [--out paper/timings]

Writes per use case N:
  trf_case<N>_<scheme>.csv  time per spectrum of each TRF fit (sample_idx, total_time_ms)
  nn_case<N>.csv            forward-pass time per spectrum of the released CNN/DU/PIDU checkpoints per batch size
"""

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCHEMES = ('baseline', 'noisy', 'random')
NN_COLUMNS = ['method', 'batch_size', 'n_params', 'n_samples', 'mean_ms', 'std_ms', 'median_ms', 'min_ms', 'max_ms']


def export(run: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    for case in (1, 2):
        for s in SCHEMES:
            t = pd.read_csv(run / f'case{case}' / 'outputs' / f'loss_lma_{s}' / 'timing_per_sample.csv')
            t[['sample_idx', 'total_time_ms']].to_csv(out / f'trf_case{case}_{s}.csv', index=False)
        xlsx = run / f'case{case}' / 'figures' / f'usecase{case}_data_figures.xlsx'
        nn = pd.read_excel(xlsx, sheet_name='time_timing_summary')
        nn = nn[nn['batch_size'].notna()][NN_COLUMNS]
        nn = nn.astype({'batch_size': int, 'n_params': int, 'n_samples': int})
        nn.to_csv(out / f'nn_case{case}.csv', index=False)
        print(f'case {case}: {len(SCHEMES)} TRF schemes, {nn["method"].nunique()} networks -> {out}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--run', type=Path, default=ROOT / 'outputs' / 'timings_final')
    ap.add_argument('--out', type=Path, default=ROOT / 'paper' / 'timings')
    a = ap.parse_args()
    export(a.run, a.out)


if __name__ == '__main__':
    main()
