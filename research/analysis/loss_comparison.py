"""Loss trajectories of CNN, DU and PIDU: median over seeds with a min-max band.

Reads `<case dir>/loss/loss_<TYPE>_<seed>_001/loss_system1.xlsx` (sheets train/val/test), averages
the logged steps within each epoch and writes `<case dir>/figures/usecase<N>_loss_comparison.{png,pdf,csv}`.

    python -m research.analysis.loss_comparison --usecase 1 [--output DIR] [--trail 1] [--seeds 0 1 2 3]

Also writes `usecase<N>_training_dynamics.xlsx` (sheets train, val, exp, test; columns epoch from 0, then
`<model> mean (<split>)` and `<model> std (<split>)` per model) and `usecase<N>_loss_dynamics_origin.xlsx` for re-plotting (e.g. in OriginLab): one sheet per split,
column A = epoch, then per model the loss of every seed plus mean, std, median, min, max over those seeds.
Default seeds: those present for all model types (the seeds used for Fig. 2); --seeds overrides.
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pidu.paths import case_dir

COLORS = {'CNN': '#0072B2', 'DU': '#009E73', 'PIDU': '#D55E00'}
SPLITS = ('train', 'val', 'test', 'exp')  # 'exp' (experimental set) only exists for use case 2
TITLES = {'train': 'train', 'val': 'validation', 'test': 'test (simulated)', 'exp': 'experimental'}


def load_run(run_dir: Path) -> dict[str, np.ndarray]:
  """Per-epoch mean loss of each split of one run (steps are spread evenly over the epochs)."""
  n_epochs = len(pd.read_excel(run_dir / 'loss_matrix.xlsx', sheet_name='train'))
  sheets = pd.read_excel(run_dir / 'loss_system1.xlsx', sheet_name=None)
  out = {}
  for split in SPLITS:
    if split not in sheets:
      continue
    df = sheets[split]
    y = df.iloc[:, 0].to_numpy(dtype=float)
    out[split] = np.array([c.mean() for c in np.array_split(y, n_epochs)])
  return out


def _runs(loss_dir: Path, model: str, trail: int) -> dict[int, Path]:
  found = {int(re.search(rf'{model}_(\d+)_', p.name).group(1)): p for p in loss_dir.glob(f'loss_{model}_*_{trail:03d}')}
  return {k: v for k, v in sorted(found.items()) if (v / 'loss_system1.xlsx').exists() and (v / 'loss_matrix.xlsx').exists()}


def collect(loss_dir: Path, trail: int, seeds=None) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, list[int]]]:
  """({model: {split: array [seeds, epochs]}}, {model: seed ids}); default seeds = shared by all model types."""
  data, used = {}, {}
  all_runs = {m: _runs(loss_dir, m, trail) for m in COLORS}
  shared = set.intersection(*[set(r) for r in all_runs.values() if r]) if any(all_runs.values()) else set()
  for model in COLORS:
    runs = [v for k, v in all_runs[model].items() if k in (set(seeds) if seeds else shared)]
    if not runs:
      continue
    used[model] = [k for k in all_runs[model] if k in (set(seeds) if seeds else shared)]
    per_run = [load_run(r) for r in runs]
    n = min(len(r['train']) for r in per_run)
    splits = [s for s in SPLITS if all(s in r for r in per_run)]
    data[model] = {s: np.stack([r[s][:n] for r in per_run]) for s in splits}
  return data, used


def plot(data, title: str):
  splits = [s for s in SPLITS if all(s in d for d in data.values())]
  fig, axes = plt.subplots(1, len(splits), figsize=(3.7 * len(splits), 3.4), sharey=True, constrained_layout=True)
  for ax, split in zip(axes, splits):
    for model, d in data.items():
      y = d[split]
      x = np.arange(1, y.shape[1] + 1)
      ax.fill_between(x, y.min(0), y.max(0), color=COLORS[model], alpha=0.2, lw=0)
      ax.plot(x, np.median(y, 0), color=COLORS[model], lw=1.5, label=f'{model} (n={y.shape[0]})')
    ax.set(yscale='log', xlabel='Epoch', title=TITLES[split])
    ax.grid(alpha=0.3, which='both', lw=0.4)
  axes[0].set_ylabel('Loss')
  axes[0].legend(frameon=False)
  fig.suptitle(title, fontsize=10)
  return fig


def write_origin(data, used, path: Path):
  """Wide tables, one sheet per split: epoch | <model> seed <i> ... | <model> mean, std, median, min, max."""
  with pd.ExcelWriter(path) as xl:
    for split in SPLITS:
      cols = {}
      for model, d in data.items():
        if split not in d:
          continue
        y = d[split]
        cols.setdefault('epoch', np.arange(1, y.shape[1] + 1))
        for seed, row in zip(used[model], y):
          cols[f'{model} seed {seed}'] = row
        for stat, f in (('mean', np.mean), ('std', np.std), ('median', np.median), ('min', np.min), ('max', np.max)):
          cols[f'{model} {stat}'] = f(y, axis=0)
      if cols:
        pd.DataFrame(cols).to_excel(xl, sheet_name=split, index=False)


def write_training_dynamics(data, path: Path):
  """Compact Origin book: per split epoch (from 0) and mean/std over seeds of each model."""
  with pd.ExcelWriter(path) as xl:
    for split in ('train', 'val', 'exp', 'test'):
      cols = {}
      for model, d in data.items():
        if split not in d:
          continue
        y = d[split]
        cols.setdefault('epoch', np.arange(y.shape[1]))
        cols[f'{model} mean ({split})'] = y.mean(0)
        cols[f'{model} std ({split})'] = y.std(0)
      if cols:
        pd.DataFrame(cols).to_excel(xl, sheet_name=split, index=False)


def main():
  p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  p.add_argument('--usecase', type=int, default=1)
  p.add_argument('--trail', type=int, default=1)
  p.add_argument('--seeds', type=int, nargs='+', help='seeds to use (default: shared by CNN, DU and PIDU)')
  p.add_argument('--output', type=Path, default=None, help='Case directory (default: outputs/case<N>)')
  a = p.parse_args()

  base = a.output or case_dir(a.usecase)
  data, used = collect(base / 'loss', a.trail, a.seeds)
  if not data:
    raise SystemExit(f'No loss_<CNN|DU|PIDU>_<seed>_{a.trail:03d} runs in {base / "loss"}')
  out = base / 'figures'
  out.mkdir(parents=True, exist_ok=True)
  stem = out / f'usecase{a.usecase}_loss_comparison'

  fig = plot(data, f'Use case {a.usecase}: loss per epoch, median and min-max over seeds')
  for ext in ('png', 'pdf'):
    fig.savefig(f'{stem}.{ext}', dpi=200)
  rows = [dict(model=m, split=s, epoch=e + 1, median=np.median(d[s][:, e]), min=d[s][:, e].min(), max=d[s][:, e].max(), n_seeds=d[s].shape[0])
          for m, d in data.items() for s in d for e in range(d[s].shape[1])]
  pd.DataFrame(rows).to_csv(f'{stem}.csv', index=False)
  write_training_dynamics(data, out / f'usecase{a.usecase}_training_dynamics.xlsx')
  write_origin(data, used, out / f'usecase{a.usecase}_loss_dynamics_origin.xlsx')
  print(f'Wrote {stem}.png/.pdf/.csv and usecase{a.usecase}_loss_dynamics_origin.xlsx; seeds', used)


if __name__ == '__main__':
  main()
