"""Every number of the abstract / results text that follows from Fig. 2, computed from the summary CSV.

Input is scripts/figure_summary_data.csv (written by scripts/collect_figure_summary.py from the run exports);
nothing is typed in by hand except the value the manuscript currently states, listed in CLAIMS below, which
is only used for the OK / DIFFERS column. Writes outputs/fig-summary/paper_numbers.{md,csv}.

    python scripts/paper_numbers.py [--data CSV] [--output-dir DIR]

Definitions (all from the 'mape_mean' and 'time_*' columns):
  reduction(A, B)  = 100 * (1 - MAPE_A / MAPE_B)                  "A lowers the error of B by x %"
  speedup(A, B)    = time_B / time_A, orders = log10(speedup)    time per spectrum, TRF at b = 1;
                     'b=1'  : PIDU at b = 1,   'batch': PIDU at the large batch (b = batch_size column).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TRF = ('TRF (baseline)', 'TRF (random)', 'TRF (noisy)')


def _row(df, case, method):
    return df[(df['case'] == case) & (df['method'] == method)].iloc[0]


def reduction(df, case, a, b):
    return 100 * (1 - _row(df, case, a)['mape_mean'] / _row(df, case, b)['mape_mean'])


def speedup(df, case, a, b, batch=False):
    """Time of b (b = 1) divided by time of a (b = 1, or the large batch)."""
    ta = _row(df, case, a)['time_batch_ms' if batch else 'time_b1_ms']
    return _row(df, case, b)['time_b1_ms'] / ta


def build_claims(df: pd.DataFrame) -> list[dict]:
    """(where, statement, paper value, computed value, unit, tolerance for OK)."""
    P = 'PIDU (ours)'
    c = []

    def add(where, text, paper, value, unit, tol):
        c.append(dict(where=where, claim=text, paper=paper, computed=value, unit=unit, tol=tol))

    # Accuracy
    add('Abstract, Sec. 5.1', 'Case 1: PIDU vs TRF (noisy): MAPE difference (0 = "matches")', 0,
        _row(df, 1, P)['mape_mean'] - _row(df, 1, 'TRF (noisy)')['mape_mean'], '% points', 1.0)
    add('Abstract, Sec. 5.1', 'Case 1: PIDU lowers the error of TRF (random)', 85, reduction(df, 1, P, 'TRF (random)'), '%', 0.5)
    add('Abstract, Sec. 5.2', 'Case 2: PIDU lowers the error of TRF (noisy)', 27, reduction(df, 2, P, 'TRF (noisy)'), '%', 0.5)
    add('Abstract, Sec. 5.2', 'Case 2: PIDU lowers the error of TRF (random)', 48, reduction(df, 2, P, 'TRF (random)'), '%', 0.5)
    add('Sec. 5.2', 'Case 2: PIDU lowers the error of TRF (baseline)', 97, reduction(df, 2, P, 'TRF (baseline)'), '%', 0.5)
    add('Abstract, Sec. 5.2', 'Case 2: PIDU lowers the error of DU (inductive bias)', 15, reduction(df, 2, P, 'DU'), '%', 0.5)
    # Speed
    add('Sec. 5.2', 'Case 2: PIDU inference time (b=1) lower than TRF (noisy)', 99,
        100 * (1 - 1 / speedup(df, 2, P, 'TRF (noisy)')), '%', 0.5)
    add('Sec. 5.2', 'Case 2: PIDU speed-up over TRF (noisy), batched', 1.5e4, speedup(df, 2, P, 'TRF (noisy)', batch=True), 'x', 0.05)
    for case in (1, 2):
        s1 = [np.log10(speedup(df, case, P, t)) for t in TRF]
        sb = [np.log10(speedup(df, case, P, t, batch=True)) for t in TRF]
        add('Abstract', f'Case {case}: smallest PIDU speed-up over any TRF variant, b=1', 1, min(s1), 'orders', 0.5)
        add('Abstract', f'Case {case}: largest PIDU speed-up over any TRF variant, batched', 4, max(sb), 'orders', 0.5)
    # Absolute timings quoted in Sec. 5.1
    add('Sec. 5.1', 'Case 1: CNN time per spectrum, large batch', 0.01, _row(df, 1, 'CNN')['time_batch_ms'], 'ms', 0.005)
    add('Sec. 5.1', 'Case 1: PIDU time per spectrum, large batch', 0.29, _row(df, 1, P)['time_batch_ms'], 'ms', 0.05)
    t = [_row(df, 1, m)['time_b1_ms'] / 1000 for m in TRF]
    add('Sec. 5.1', 'Case 1: fastest TRF variant, time per spectrum', 3, min(t), 's', 0.5)
    add('Sec. 5.1', 'Case 1: slowest TRF variant, time per spectrum', 16, max(t), 's', 0.5)
    add('Sec. 5.1', 'Case 1: slowest neural model (b=1), time per spectrum', 'tens of ms',
        max(_row(df, 1, m)['time_b1_ms'] for m in ('CNN', 'DU', P)), 'ms', None)
    return c


def build_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = build_claims(df)
    for r in rows:
        num = isinstance(r['paper'], (int, float)) and r['tol'] is not None
        r['status'] = ('OK' if abs(r['computed'] - r['paper']) <= (r['tol'] * (abs(r['paper']) if r['unit'] == 'x' else 1))
                       else 'DIFFERS') if num else 'check'
    return pd.DataFrame(rows).drop(columns='tol')


def _fmt(v):
    if isinstance(v, str):
        return v
    return f'{v:.3g}' if abs(v) < 1 or abs(v) >= 1e4 else f'{v:.1f}' if abs(v) < 100 else f'{v:.0f}'


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--data', type=Path, default=HERE / 'figure_summary_data.csv')
    ap.add_argument('--output-dir', type=Path, default=HERE.parent / 'outputs' / 'fig-summary')
    a = ap.parse_args()

    df = pd.read_csv(a.data, comment='#')
    if (df['source'] != 'runs').any():
        print('WARNING: rows not computed from the runs:', df.loc[df['source'] != 'runs', ['case', 'method']].values.tolist())

    tab = build_table(df)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    tab.to_csv(a.output_dir / 'paper_numbers.csv', index=False)

    inputs = df[['case', 'method', 'n_seeds', 'mape_mean', 'mape_std', 'time_b1_ms', 'time_batch_ms', 'batch_size']]
    lines = ['# Numbers of the paper recomputed from Fig. 2 data', '', f'Source: `{a.data.name}` (batch size {int(df["batch_size"].iloc[0])})', '',
             '## Fig. 2 values', '', '| case | method | seeds | MAPE % (mean ± std) | time b=1 (ms) | time batch (ms) |', '|---|---|---|---|---|---|']
    for _, r in inputs.iterrows():
        lines.append(f'| {r["case"]} | {r["method"]} | {int(r["n_seeds"])} | {_fmt(r["mape_mean"])} ± {_fmt(r["mape_std"])} | '
                     f'{_fmt(r["time_b1_ms"])} | {"" if pd.isna(r["time_batch_ms"]) else _fmt(r["time_batch_ms"])} |')
    lines += ['', '## Claims', '', '| where | claim | paper | computed | unit | |', '|---|---|---|---|---|---|']
    for _, r in tab.iterrows():
        lines.append(f'| {r["where"]} | {r["claim"]} | {_fmt(r["paper"])} | {_fmt(r["computed"])} | {r["unit"]} | {r["status"]} |')
    (a.output_dir / 'paper_numbers.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
