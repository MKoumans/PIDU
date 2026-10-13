"""Collect the data for scripts/figure_summary.py from the run outputs.

MAPE and oscillator precision/recall use ONE definition for all methods (see research/analysis/metrics.py):
system parameters with non-zero GT (Case 2: both systems for the networks) plus the (w0, wp, g) of
oscillators matched to a GT oscillator within --tol on w0; missed oscillators are excluded from MAPE and
appear in recall.  TRF: Hungarian-matched output vs GT slots; networks: rank pairing on the region grid.

* Networks: MAPE = mean over seeds of the per-seed mean over spectra, std over seeds (as in the Fig. 2 caption).
  Seeds = those present for ALL of CNN, DU and PIDU in the use case (override: --seeds), so the three
  networks are always compared on the same seeds. Precision/recall: mean/std over seeds.
* TRF: Case 1 from outputs/case1/outputs/loss_lma_<scheme>; Case 2 from outputs/case2/outputs/loss_lma_<scheme> (run on the experimental set).
* Anything not available from the runs is taken from the placeholder CSV (column 'source').
* Times: from the timing set paper/timings/ (default; written by paper/timings.sh, all methods timed on one
  machine): TRF mean/std over spectra of trf_case<N>_<scheme>.csv, networks mean/std over repetitions of the
  released checkpoint (seed 0) in nn_case<N>.csv. --timings runs uses the run folders instead (TRF:
  timing_per_sample.csv; networks: outputs/fig-summary/timing_case<N>.csv of scripts/benchmark_timings.py,
  mean/std over the seeds' checkpoints).

    python scripts/collect_figure_summary.py [--batch-size 256] [--seeds 0 1 2 3] [--tol 0.2] [--out scripts/figure_summary_data.csv]
    python scripts/collect_figure_summary.py --timings-only      # replace only the time columns of --out
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from research.analysis import metrics as fm  # noqa: E402
FIG = ROOT / 'outputs' / 'fig-summary'
TRF = {'TRF (baseline)': 'baseline', 'TRF (random)': 'random', 'TRF (noisy)': 'noisy'}
NETS = {'CNN': 'CNN', 'DU': 'DU', 'PIDU (ours)': 'PIDU'}
NET_SPLIT = {1: 'test_test', 2: 'test_exp'}

NET_DATASET = {1: ROOT / 'data/datasets/usecase1/case1_test_n2_10_nwsn.npz',
               2: ROOT / 'data/datasets/usecase2/case2_experimental.npz'}
NET_SYSTEMS = {1: ('system1',), 2: ('system1', 'system2')}
TRF_RUNS = {1: ROOT / 'outputs/case1/outputs', 2: ROOT / 'outputs/case2/outputs'}
TIMINGS = ROOT / 'paper' / 'timings'
TIME_COLS = ['time_b1_ms', 'time_b1_std_ms', 'time_batch_ms', 'time_batch_std_ms']


def _summ(per_spectrum, prec, rec, per_seed_mape=None):
    """per_spectrum: pooled for TRF (std over spectra); per_seed_mape: networks (std over seeds)."""
    m = per_seed_mape if per_seed_mape is not None else per_spectrum
    return dict(mape_mean=np.nanmean(m), mape_std=np.nanstd(m),
                precision=np.mean(prec), recall=np.mean(rec),
                precision_std=np.std(prec) if len(prec) > 1 else np.nan,
                recall_std=np.std(rec) if len(rec) > 1 else np.nan)


def _seed(run_dir: Path) -> int:
    return int(run_dir.name.split('_')[-2])


def common_seeds(root: Path, split: str) -> list[int]:
    """Seeds whose run has the evaluation export (<split>) for every network type."""
    sets = [{_seed(d) for d in root.glob(f'loss_{p}_*_001') if (d / split / 'last_unfold.xlsx').exists()} for p in NETS.values()]
    return sorted(set.intersection(*sets))


def timing_set_times(timings: Path, case: int, label: str, batch_size: int) -> dict:
    """Time columns of one method from the timing set (paper/timings/)."""
    if label in TRF:
        t = pd.read_csv(timings / f'trf_case{case}_{TRF[label]}.csv')['total_time_ms']
        return dict(time_b1_ms=t.mean(), time_b1_std_ms=t.std(ddof=0), time_batch_ms=np.nan, time_batch_std_ms=np.nan)
    nn = pd.read_csv(timings / f'nn_case{case}.csv')
    nn = nn[nn['method'] == f'{NETS[label]}-t01-00']
    row = {}
    for bs, key in ((1, 'time_b1'), (batch_size, 'time_batch')):
        r = nn[nn['batch_size'] == bs]
        if len(r) != 1:
            raise SystemExit(f'nn_case{case}.csv lacks {NETS[label]}-t01-00 at batch size {bs}')
        row[f'{key}_ms'], row[f'{key}_std_ms'] = r['mean_ms'].iloc[0], r['std_ms'].iloc[0]
    return row


def network_rows(case: int, batch_size: int, tol: float, seeds=None, run_timings=True) -> dict:
    root = ROOT / 'outputs' / f'case{case}' / 'outputs'
    seeds = list(seeds) if seeds else common_seeds(root, NET_SPLIT[case])
    print(f'Case {case}: seeds {seeds}')
    out = {}
    for label, prefix in NETS.items():
        runs = [root / f'loss_{prefix}_{s}_001' for s in seeds]
        res = [fm.network_result(d / NET_SPLIT[case], NET_DATASET[case], tol, NET_SYSTEMS[case]) for d in runs]
        row = _summ(None, [r.tp / r.n_pred for r in res], [r.tp / r.n_gt for r in res],
                    per_seed_mape=[np.nanmean(r.mape_per_spectrum) for r in res])
        row.update(n_seeds=len(runs), mape_unit='percent')
        out[label] = row
        if not run_timings:
            continue
        timing = pd.read_csv(FIG / f'timing_case{case}.csv')
        t = timing[timing['method'].isin([f'{prefix}-t01-{s:02d}' for s in seeds])]  # t01 = reported configuration
        if t['method'].nunique() != len(seeds):
            raise SystemExit(f'timing_case{case}.csv lacks checkpoints {prefix}-t01-<seed> for seeds {seeds}')
        for bs, key in ((1, 'time_b1'), (batch_size, 'time_batch')):
            m = t[t['batch_size'] == bs]['mean_ms']
            row[f'{key}_ms'], row[f'{key}_std_ms'] = m.mean(), m.std(ddof=0)
    return out


def trf_row(case: int, scheme: str, tol: float):
    """TRF run on the matching evaluation set, or None (-> placeholder)."""
    run = TRF_RUNS[case] / f'loss_lma_{scheme}'
    split = next(iter(sorted(run.glob('test_*/gt.xlsx'))), None)
    t_csv = run / 'timing_per_sample.csv'
    if split is None or not t_csv.exists():
        return None
    n_expected = len(np.load(NET_DATASET[case], allow_pickle=True)['data_x_sys_tr_system1'])
    r = fm.trf_result(split.parent, tol)
    if len(r.mape_per_spectrum) < n_expected:  # incomplete run
        return None
    t = pd.read_csv(t_csv)['total_time_ms']
    return dict(**_summ(r.mape_per_spectrum, [r.tp / max(r.n_pred, 1)], [r.tp / r.n_gt]), n_seeds=1,
                mape_unit='percent', time_b1_ms=t.mean(), time_b1_std_ms=t.std(ddof=0),
                time_batch_ms=np.nan, time_batch_std_ms=np.nan)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--seeds', type=int, nargs='+', help='network seeds (default: seeds shared by CNN, DU and PIDU)')
    ap.add_argument('--tol', type=float, default=0.2, help='relative w0 tolerance for a true-positive oscillator')
    ap.add_argument('--placeholder', type=Path, default=ROOT / 'scripts' / 'figure_summary_placeholder.csv')
    ap.add_argument('--out', type=Path, default=ROOT / 'scripts' / 'figure_summary_data.csv')
    ap.add_argument('--timings', default=str(TIMINGS),
                    help="timing set directory (default paper/timings), or 'runs' for the run folders")
    ap.add_argument('--timings-only', action='store_true',
                    help='only replace the time columns of the existing --out from the timing set (no runs needed)')
    a = ap.parse_args()
    timings = None if a.timings == 'runs' else Path(a.timings)

    if a.timings_only:
        if timings is None:
            raise SystemExit('--timings-only needs a timing set directory')
        lines = a.out.read_text().splitlines(keepends=True)
        df = pd.read_csv(a.out, comment='#')
        for i, r in df.iterrows():
            for k, v in timing_set_times(timings, r['case'], r['method'], int(r['batch_size'])).items():
                df.at[i, k] = v
        a.out.write_text(''.join(l for l in lines if l.startswith('#')) + df.to_csv(index=False))
        print(df[['case', 'method', *TIME_COLS]].to_string())
        return

    ph = pd.read_csv(a.placeholder, comment='#')
    rows = []
    for case in (1, 2):
        nets = network_rows(case, a.batch_size, a.tol, a.seeds, run_timings=timings is None)
        for label in [*TRF, *NETS]:
            ph_row = ph[(ph['case'] == case) & (ph['method'] == label)].iloc[0]
            new = nets[label] if label in NETS else trf_row(case, TRF[label], a.tol)
            if new is None:  # fall back to Table 1 placeholder (fractions -> %)
                new = dict(mape_mean=100 * ph_row['mape_mean'], mape_std=np.nan, precision=np.nan, recall=np.nan, precision_std=np.nan, recall_std=np.nan, n_seeds=ph_row['n_seeds'],
                           mape_unit='percent', time_b1_ms=ph_row['time_b1_ms'], time_b1_std_ms=np.nan,
                           time_batch_ms=np.nan, time_batch_std_ms=np.nan)
                source = 'placeholder (Table 1)'
            else:
                source = 'runs'
                if timings is not None:
                    new.update(timing_set_times(timings, case, label, a.batch_size))
            rows.append(dict(case=case, method=label, **new, batch_size=a.batch_size, tol=a.tol, source=source))
    df = pd.DataFrame(rows)
    header = ('# Generated by scripts/collect_figure_summary.py. mape_*: percent; time_*: ms per spectrum; '
              'time_batch_*: batch_size spectra per call. source=placeholder rows are NOT from the rerun.\n')
    a.out.write_text(header + df.to_csv(index=False))
    print(df.to_string())


if __name__ == '__main__':
    main()
