"""Build the assets of the project page (docs/index.html): the fit-animation pool and the explorer data.

Run once per use case; pidu.config reads PIDU_USECASE at import, so each use case is its own process:

    python docs/tools/build_site_assets.py --usecase 1
    python docs/tools/build_site_assets.py --usecase 2

Writes to docs/static/:
  fits/case<N>/<k>.gif        animated fit (TRF vs PIDU, step by step) of sample k, from scripts/fit_evolution.py
  fits/manifest.js            window.PIDU_FITS = {case<N>: [{src, index, title}, ...], ...}
  explorer/case<N>/<k>.json   measured spectrum, PIDU spectrum and parameters after every unfold, reference parameters
  explorer/manifest.js        window.PIDU_EXPLORER = {case<N>: {label, dataset, indices, channels}, ...}

The animations default to the experimental SrTiO3 spectra for use case 2 (local copy only; the experimental
dataset is not on the Hugging Face Hub). The explorer only uses simulated test spectra unless
--explorer-experimental is passed, because its JSON exposes the measured spectra as numbers.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path


def _preparse_usecase() -> int:
  parser = argparse.ArgumentParser(add_help=False)
  parser.add_argument('--usecase', type=int, required=True)
  return parser.parse_known_args()[0].usecase


os.environ['PIDU_USECASE'] = str(_preparse_usecase())

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))

import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch
from PIL import Image

import fit_evolution as fe
from pidu.config import usecase

STATIC = ROOT / 'docs' / 'static'
FIT_DATASET = {1: 'usecase1/case1_test_n2_10_nwsn', 2: 'usecase2/case2_experimental'}
FIT_METHODS = {1: ['TRF(random)', 'PIDU'], 2: ['TRF(noisy)', 'PIDU']}
FIT_TITLE = {1: 'Simulated spectrum #{i}', 2: 'SrTiO$_3$ measurement #{i}'}
EXPLORER_LABEL = {1: 'Use case 1 · simulated', 2: 'Use case 2 · simulated'}
CHANNEL_LABELS = {'system1': 'Reflectance', 'system2_0': 'Transmission (Re t)', 'system2_1': 'Transmission (Im t)'}
MAX_POINTS = 512  # spectra are resampled to at most this many points for the explorer


def loop_ms(gif: Path) -> int:
  """Length of one loop of an animated GIF in ms (the page shows each fit for two loops)."""
  im = Image.open(gif)
  total = 0
  for k in range(im.n_frames):
    im.seek(k)
    total += im.info.get('duration', 250)
  return total


def write_manifest(path: Path, var: str, key: str, value):
  """Update one key of a `window.<var> = {...};` manifest, keeping the other use cases."""
  data = {}
  if path.exists():
    m = re.search(r'=\s*(\{.*\});?\s*$', path.read_text(encoding='utf-8'), flags=re.S)
    if m:
      data = json.loads(m.group(1))
  data[key] = value
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(f'window.{var} = {json.dumps(dict(sorted(data.items())), indent=1)};\n', encoding='utf-8')


def build_fits(args, model):
  ds_name = args.fit_dataset or FIT_DATASET[usecase]
  if not fe.dataset_file(ds_name, download_missing=False).exists():
    ds_name = fe.resolve_dataset(None)
  dataset = fe.load_dataset(ds_name, args.device)
  rng = np.random.default_rng(args.seed)
  indices = sorted(rng.choice(len(dataset), size=min(args.n_fits, len(dataset)), replace=False).tolist())
  methods = FIT_METHODS[usecase]
  params = fe.DEFAULT_PARAMS[usecase]
  title = FIT_TITLE[usecase] if 'experimental' in ds_name else FIT_TITLE[1]

  out = STATIC / 'fits' / f'case{usecase}'
  out.mkdir(parents=True, exist_ok=True)
  entries = []
  for i in indices:
    sample = fe.get_sample(dataset, i)
    trajs = fe.run_methods(dataset, [i], methods, pidu_model=model, max_iter=args.max_iter, verbose=True)[0]
    stem = out / f'{i:03d}'
    gif = fe.make_fit_gif(sample, trajs, params, stem, frame_ms=args.frame_ms, hold_ms=args.hold_ms,
                          dpi=args.dpi, data=False, title=title.format(i=i))['gif']
    entries.append({'src': f'static/fits/case{usecase}/{i:03d}.gif', 'index': i, 'ms': loop_ms(gif),
                    'title': title.format(i=i).replace('$_3$', '₃')})
    # Written after every fit, so a long run shows up on the page while it is still going.
    write_manifest(STATIC / 'fits' / 'manifest.js', 'PIDU_FITS', f'case{usecase}',
                   {'dataset': ds_name, 'methods': methods, 'items': entries})


def _resample(wn: np.ndarray, *ys: np.ndarray):
  if wn.size <= MAX_POINTS:
    return (wn, *ys)
  grid = np.linspace(wn.min(), wn.max(), MAX_POINTS)
  return (grid, *[np.stack([np.interp(grid, wn, r) for r in np.atleast_2d(y)]).squeeze() for y in ys])


def _round(a, digits=4):
  return np.round(np.asarray(a, dtype=float), digits).tolist()


def build_explorer(args, model):
  ds_name = args.explorer_dataset or (FIT_DATASET[2] if (usecase == 2 and args.explorer_experimental)
                                      else fe.SIMULATED_TEST[usecase])
  dataset = fe.load_dataset(ds_name, args.device)
  rng = np.random.default_rng(args.seed + 1)
  indices = sorted(rng.choice(len(dataset), size=min(args.n_explorer, len(dataset)), replace=False).tolist())
  out = STATIC / 'explorer' / f'case{usecase}'
  out.mkdir(parents=True, exist_ok=True)

  channels = None
  for start in range(0, len(indices), args.batch):
    batch = indices[start:start + args.batch]
    trajs = fe.run_pidu(model, dataset, batch)
    for i, traj in zip(batch, trajs):
      sample = fe.get_sample(dataset, i)
      # Parameters shown: system parameters that PIDU estimates, and the oscillators present in the reference.
      keep = [k for k, n in enumerate(sample.names)
              if not re.search(r'_slot\d+$', n) and not np.allclose(traj.p[:, k], sample.p_true[k])]
      spectra = {}
      for ch in sample.y:
        wn, y, y_hat = _resample(sample.wn[ch], sample.y[ch], traj.y[ch])
        spectra[ch] = {'wn': _round(wn, 2), 'y': _round(y), 'y_hat': _round(y_hat)}
      channels = channels or [{'id': ch, 'label': CHANNEL_LABELS.get(ch, ch)} for ch in sample.y]
      p_true = sample.p_true[keep]
      p_hat = traj.p[:, keep]
      ape = np.abs(p_hat[-1] - p_true) / np.maximum(np.abs(p_true), 1e-12) * 100
      record = {
        'index': i,
        'n_unfolds': int(traj.n_steps - 1),
        'names': [sample.names[k] for k in keep],
        'p_true': _round(p_true, 4),
        'p_hat': _round(p_hat, 4),  # [step, param]; step 0 is PIDU's initial estimate
        'mape': round(float(ape.mean()), 3),
        'spectra': spectra,
      }
      (out / f'{i:04d}.json').write_text(json.dumps(record, separators=(',', ':')), encoding='utf-8')
    print(f'[explorer] case{usecase}: {min(start + args.batch, len(indices))}/{len(indices)}')

  label = 'Use case 2 · SrTiO₃ measured' if 'experimental' in ds_name else EXPLORER_LABEL[usecase]
  write_manifest(STATIC / 'explorer' / 'manifest.js', 'PIDU_EXPLORER', f'case{usecase}',
                 {'label': label, 'dataset': ds_name, 'indices': indices, 'channels': channels})


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('--usecase', type=int, required=True)
  parser.add_argument('--only', choices=['fits', 'explorer'], default=None, help='build only one of the two')
  parser.add_argument('--n-fits', type=int, default=20, dest='n_fits', help='animations in the pool')
  parser.add_argument('--n-explorer', type=int, default=64, dest='n_explorer', help='spectra in the explorer')
  parser.add_argument('--fit-dataset', default=None, dest='fit_dataset')
  parser.add_argument('--explorer-dataset', default=None, dest='explorer_dataset')
  parser.add_argument('--explorer-experimental', action='store_true', dest='explorer_experimental',
                      help='use case 2 explorer on the measured SrTiO3 spectra (publishes them as JSON)')
  parser.add_argument('--checkpoint', type=Path, default=None)
  parser.add_argument('--max-iter', type=int, default=200, dest='max_iter')
  parser.add_argument('--frame-ms', type=int, default=250, dest='frame_ms')
  parser.add_argument('--hold-ms', type=int, default=2500, dest='hold_ms')
  parser.add_argument('--dpi', type=int, default=100)
  parser.add_argument('--batch', type=int, default=32)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
  args = parser.parse_args()
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)

  model = fe.load_pidu(args.checkpoint, args.device)
  if args.only in (None, 'explorer'):
    build_explorer(args, model)
  if args.only in (None, 'fits'):
    build_fits(args, model)


if __name__ == '__main__':
  sys.exit(main())
