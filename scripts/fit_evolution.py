"""Animate how TRF and PIDU fit one spectrum, step by step.

GIF layout: one column per method (--methods) and four rows: the spectrum (measured vs. the simulated
spectrum of the current estimate) and three tracked parameters (default: the soft mode
w0_before, wp_before, g_before), each plotted against the update step with the reference
value as a dashed line. A TRF step is one iteration of scipy's least_squares (method='trf');
a PIDU step is one unfold (step 0 is the initial estimate). PIDU stops after its
unfolds and freezes while TRF continues. Every frame takes --frame-ms; runs longer than
--max-frames steps are shown at logarithmically spaced steps.

    python scripts/fit_evolution.py                              # experimental sample 0: TRF(noisy) vs PIDU
    python scripts/fit_evolution.py --index 40 --methods "TRF(baseline)|TRF(random)|TRF(noisy)|PIDU"
    python scripts/fit_evolution.py --usecase 1 --dataset usecase1/case1_test_n2_10_nwsn --params w0_1 wp_1 g_1

Methods (--methods, separated by spaces or '|'; default "TRF(noisy)|PIDU"), the TRF schemes as in
pidu.baselines.lma:
  TRF(noisy)     start at the reference +-10 %, free parameters bounded to +-50 % of the start
  TRF(random)    start uniform within the training bounds, bounded to +-50 % of the start
  TRF(baseline)  free parameters start at 1, unbounded
  TRF(pidu)      start at PIDU's initial estimate, bounded to +-50 % of the start
  PIDU           starts from its own (learned) initial estimate

Inputs are fetched from the Hugging Face Hub when missing: the released PIDU checkpoint
(pidu.hub.RELEASED_SEEDS) and simulated datasets. The experimental dataset is not on the Hub;
without a local copy the default falls back to the simulated test set of the released model.

Writes to <output> (default outputs/case<N>/figures/fit_evolution/), stem
fit_evolution_<dataset>_<index>_<methods>:
  <stem>.gif           the animation
  <stem>_params.csv    method, step, parameter, value, true (all steps)
  <stem>_frames.csv    frame, duration_ms and the step shown per method
  <stem>.npz           wavenumbers, measured and simulated spectra per channel and step,
                       full parameter vectors per step, reference parameters and names
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence


def _preparse_usecase() -> int:
  """`--usecase` must be known before pidu is imported (pidu.config reads PIDU_USECASE at import)."""
  parser = argparse.ArgumentParser(add_help=False)
  parser.add_argument('--usecase', type=int, default=int(os.environ.get('PIDU_USECASE', 2)))
  return parser.parse_known_args()[0].usecase


os.environ['PIDU_USECASE'] = str(_preparse_usecase())
warnings.filterwarnings('ignore', category=RuntimeWarning)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset

from pidu.baselines.lma import LevenbergMarquardt, TmMethodWrapper, _build_tmm, _random_init_training
from pidu.config import usecase
from pidu.data.simulation import SimulationDatasetBatch, load_sim_ds, p_gen_usecase1, p_gen_usecase2
from pidu.data.utils import pt2np
from pidu.hub import dataset_file, released_checkpoint
from pidu.model.networks.io import load_checkpoint
from pidu.model.utils import create_masks_p
from pidu.paths import case_dir

TRF_SCHEMES = ('noisy', 'random', 'baseline', 'pidu')
DEFAULT_METHODS = ['TRF(noisy)', 'PIDU']
# Fixed colour per method (categorical slots 1, 2, 3, 7, 5), whichever subset is shown.
COLORS = {'TRF(noisy)': '#2a78d6', 'PIDU': '#eb6834', 'TRF(random)': '#1baf7a',
          'TRF(baseline)': '#4a3aa7', 'TRF(pidu)': '#e87ba4'}
INK, INK_MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
CHANNEL_LABELS = {'system1': 'reflectance', 'system2_0': 'Re t', 'system2_1': 'Im t'}
SIMULATED_TEST = {1: 'usecase1/case1_test_n2_10_nwsn', 2: 'usecase2/case2_test_n2_10_wsn'}  # test sets of the released models
DEFAULT_DATASET = {1: SIMULATED_TEST[1], 2: 'usecase2/case2_experimental'}
DEFAULT_PARAMS = {1: ['w0_1', 'wp_1', 'g_1'], 2: ['w0_before', 'wp_before', 'g_before']}
NOISE_FACTOR = 0.2  # 'noisy' init: truth * (1 +- NOISE_FACTOR / 2), as in pidu.baselines.lma


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Sample:
  """One spectrum: measured channels on their wavenumber grids and the reference parameters."""
  index: int
  wn: Dict[str, np.ndarray]   # channel -> wavenumbers (cm^-1)
  y: Dict[str, np.ndarray]    # channel -> measured spectrum
  p_true: np.ndarray          # [n_params]: system parameters, then (w0, wp, g) per oscillator slot
  names: List[str]            # parameter name per entry of p_true


@dataclass
class Trajectory:
  """Estimate of one method at every update step; row 0 is the initial estimate."""
  method: str
  p: np.ndarray               # [n_steps, n_params]
  y: Dict[str, np.ndarray]    # channel -> [n_steps, n_wn]

  @property
  def n_steps(self) -> int:
    return self.p.shape[0]


def _system_of(channel: str, wn: Dict) -> str:
  return channel if channel in wn else channel.rsplit('_', 1)[0]


def resolve_dataset(name: Optional[str] = None) -> str:
  """The dataset to use: `name`, or the default of the use case. The experimental default is not on the
  Hugging Face Hub; without a local copy the simulated test set is used instead. A simulated dataset
  that is not available locally is downloaded."""
  if name is None:
    name = DEFAULT_DATASET[usecase]
    if not dataset_file(name, download_missing=False).exists() and name != SIMULATED_TEST[usecase]:
      print(f'[fit_evolution] {name} is not available locally (the experimental data is not on the Hugging Face '
            f'Hub); using the simulated test set {SIMULATED_TEST[usecase]} instead. Pass --dataset to choose another.')
      name = SIMULATED_TEST[usecase]
  if not dataset_file(name).exists():
    raise SystemExit(f'Dataset {name} not found in {dataset_file(name, download_missing=False).parent}')
  return name


def load_dataset(name: str, device: str) -> SimulationDatasetBatch:
  return load_sim_ds(name, y_return_mode='noised', device=device)


def get_sample(dataset: SimulationDatasetBatch, index: int) -> Sample:
  wn, x_sys, x_osc, y, _ = dataset[index]
  sys_true, osc_true = x_sys[1]['system1'], x_osc[1]['system1']
  p_true = pt2np(torch.cat([sys_true, osc_true[..., -3:].flatten(1)], dim=1)).ravel().astype(np.float64)

  names = list(dataset.col_names_sys['system1'])
  n_sys = len(names)
  osc = p_true[n_sys:].reshape(-1, 3)
  # Oscillators present in the reference are named w0_1, wp_1, g_1, ... in slot order; the others w0_slot<i>.
  present = np.flatnonzero(np.any(osc != 0, axis=1))
  for s in range(osc.shape[0]):
    tag = f'_{np.searchsorted(present, s) + 1}' if s in present else f'_slot{s}'
    names += [a + tag for a in ('w0', 'wp', 'g')]

  return Sample(
    index=index,
    wn={ch: pt2np(wn[_system_of(ch, wn)]).ravel() for ch in y},
    y={ch: pt2np(v).ravel() for ch, v in y.items()},
    p_true=p_true,
    names=names,
  )


def param_index(sample: Sample, name: str) -> int:
  """Index of a parameter: a system parameter ('w0_before'), the k-th reference oscillator
  ('w0_1') or an oscillator slot ('w0_slot2')."""
  if name in sample.names:
    return sample.names.index(name)
  valid = [n for n in sample.names if not re.search(r'_slot\d+$', n)]
  raise ValueError(f'Unknown parameter {name!r} for sample {sample.index}; choose from {valid} or <w0|wp|g>_slot<i>')


# ─────────────────────────────────────────────────────────────────────────────
# Methods
# ─────────────────────────────────────────────────────────────────────────────

def default_checkpoint() -> Path:
  """The released PIDU checkpoint (pidu.hub.RELEASED_SEEDS), downloaded from the Hugging Face Hub if missing."""
  return released_checkpoint(usecase, 'PIDU')


def load_pidu(checkpoint: Optional[Path], device: str):
  model = load_checkpoint(checkpoint or default_checkpoint(), map_location=device)
  model.eval()
  return model


def run_pidu(model, dataset: SimulationDatasetBatch, indices: Sequence[int]) -> List[Trajectory]:
  """Run PIDU on a batch of samples and record the estimate and its spectrum after every unfold.

  The framework calls `_mask` (parameters) and `_forward_system` (spectrum) once for the initial
  estimate and once per unfold; both are wrapped on the instance to record their outputs.
  """
  wn, x_sys, x_osc, y, _ = next(iter(DataLoader(Subset(dataset, list(indices)), batch_size=len(indices))))
  params, spectra = [], []
  mask, forward_system = model._mask, model._forward_system

  def record_mask(*args, **kwargs):
    out = mask(*args, **kwargs)
    _, sys, osc = out
    # Ungated oscillator values: the estimate of a slot also when its confidence is below threshold.
    params.append(torch.cat([sys['system1'][:, 0], osc['system1'][:, 0, ..., -3:].flatten(1)], dim=1))
    return out

  def record_forward(*args, **kwargs):
    out = forward_system(*args, **kwargs)
    spectra.append({ch: v[:, 0] for ch, v in out.items()})
    return out

  model._mask, model._forward_system = record_mask, record_forward
  try:
    with torch.no_grad():
      model(wn=wn, y=y, x_sys=x_sys, x_osc=x_osc)
  finally:
    del model._mask, model._forward_system

  p = pt2np(torch.stack(params, dim=1)).astype(np.float64)  # [batch, n_steps, n_params]
  ys = {ch: pt2np(torch.stack([s[ch] for s in spectra], dim=1)).astype(np.float64) for ch in spectra[0]}
  return [Trajectory('PIDU', p[b], {ch: v[b] for ch, v in ys.items()}) for b in range(len(indices))]


def trf_problem(dataset: SimulationDatasetBatch, index: int):
  """Forward model, reference parameters, free-parameter mask and inputs of one sample, set up as
  in pidu.baselines.lma.benchmark_lma (numpy backend)."""
  wn, x_sys, x_osc, y, _ = dataset[index]
  (x_sys_init, x_sys_tr), (x_osc_init, x_osc_tr) = x_sys, x_osc
  p_true = torch.cat([x_sys_tr['system1'], x_osc_tr['system1'][..., -3:].flatten(1)], dim=1)
  mask = create_masks_p(usecase, n_layers=1, n_sys_pars=x_sys_init, n_osc_pars=x_osc_init,
                        dv=dataset.device)['system1']

  model = TmMethodWrapper(f=_build_tmm(dataset, 'numpy'), backend='numpy')
  ref_sys = pt2np(x_sys_tr['system1']).ravel()
  fixed_sys = pt2np(mask).ravel()[:ref_sys.size].astype(bool)
  overrides = {}
  for name in model.system_names:  # system-specific fixed parameters, e.g. the thickness per system
    if name != 'system1' and name in x_sys_tr:
      sys_k = pt2np(x_sys_tr[name]).ravel()
      idx = np.flatnonzero(fixed_sys & ~np.isclose(sys_k, ref_sys))
      if idx.size:
        overrides[name] = (idx, sys_k[idx])
  model.set_overrides(overrides)
  return model, wn, y, p_true, mask


def trf_init(scheme: str, p_true: torch.Tensor, mask: torch.Tensor, dataset: SimulationDatasetBatch,
             pidu_init: Optional[np.ndarray] = None) -> torch.Tensor:
  """Initial TRF estimate; fixed parameters keep their reference value."""
  free = mask == 0
  if scheme == 'random':
    p_gen = {1: p_gen_usecase1, 2: p_gen_usecase2}[usecase]
    return _random_init_training(p_true, mask, p_gen, dataset.dtype, dataset.device)
  if scheme == 'noisy':
    return torch.where(free, p_true * (1.0 + (torch.rand_like(p_true) - 0.5) * NOISE_FACTOR), p_true)
  if scheme == 'baseline':
    return torch.where(free, torch.ones_like(p_true), p_true)
  if scheme == 'pidu':
    if pidu_init is None:
      raise ValueError("init 'pidu' needs PIDU's initial estimate")
    return torch.where(free, torch.as_tensor(pidu_init, dtype=p_true.dtype, device=p_true.device)[None], p_true)
  raise ValueError(f'Unknown init scheme {scheme!r}')


def run_trf(
  dataset: SimulationDatasetBatch,
  index: int,
  init: str = 'random',
  p_init: Optional[np.ndarray] = None,
  max_iter: int = 200,
  relative_bounds: Optional[bool] = None,
) -> Trajectory:
  """Fit one sample with TRF and record the estimate and its spectrum after every iteration.

  p_init (full parameter vector) overrides the `init` scheme. relative_bounds bounds the free
  parameters to +-50 % of the start (default: on unless init='baseline', as in the paper).
  Only the oscillator slots present in the reference are fitted (all_osc=False).
  """
  model, wn, y, p_true, mask = trf_problem(dataset, index)
  if p_init is not None:
    initial = torch.as_tensor(np.asarray(p_init).reshape(1, -1), dtype=p_true.dtype, device=p_true.device)
  else:
    initial = trf_init(init, p_true, mask, dataset)
  if relative_bounds is None:
    relative_bounds = init != 'baseline'

  lma = LevenbergMarquardt(wn=wn, model=model, method='trf', max_nfev=max_iter, dtype=dataset.dtype,
                           backend='numpy', all_osc=False, bound_free_relative_50=relative_bounds,
                           record_iterates=True)
  _, history = lma.fit(y, pt2np(initial), pt2np(p_true), pt2np(mask).astype(bool))
  p = history['iterates'].astype(np.float64)

  per_step = [model.channels(lma.wn, row[None]) for row in p]
  ys = {ch: np.stack([np.asarray(s[ch], dtype=np.float64).ravel() for s in per_step]) for ch in per_step[0]}
  return Trajectory(f'TRF({init})', p, ys)


def parse_methods(specs: Sequence[str]) -> List[str]:
  """Canonical method labels from e.g. ['TRF(baseline)|trf(noisy)', 'PIDU']; a bare 'TRF' is TRF(noisy)."""
  labels = []
  for spec in specs:
    for part in re.split(r'[|,]', spec):
      part = part.strip()
      if not part:
        continue
      if part.upper() == 'PIDU':
        label = 'PIDU'
      elif m := re.fullmatch(r'TRF(?:\((\w+)\))?', part, flags=re.IGNORECASE):
        scheme = (m.group(1) or 'noisy').lower()
        if scheme not in TRF_SCHEMES:
          raise ValueError(f'Unknown TRF scheme {scheme!r} in {part!r}; choose from {TRF_SCHEMES}')
        label = f'TRF({scheme})'
      else:
        raise ValueError(f'Unknown method {part!r}; use PIDU or TRF(<{"|".join(TRF_SCHEMES)}>)')
      if label not in labels:
        labels.append(label)
  return labels


def trf_scheme(method: str) -> str:
  return re.fullmatch(r'TRF\((\w+)\)', method).group(1)


def fit_evolution(
  method: str,
  dataset: SimulationDatasetBatch,
  index: int,
  p_init: Optional[np.ndarray] = None,
  pidu_model=None,
  max_iter: int = 200,
) -> Trajectory:
  """Estimate per update step of one method ('PIDU' | 'TRF(<scheme>)') on one sample.

  p_init (full parameter vector) overrides the start of a TRF method; PIDU starts from its own
  (learned) initial estimate. TRF(pidu) needs pidu_model.
  """
  return run_methods(dataset, [index], [method], pidu_model=pidu_model, max_iter=max_iter, p_init=p_init)[0][method]


def run_methods(
  dataset: SimulationDatasetBatch,
  indices: Sequence[int],
  methods: Sequence[str],
  pidu_model=None,
  max_iter: int = 200,
  p_init: Optional[np.ndarray] = None,
  verbose: bool = False,
) -> List[Dict[str, Trajectory]]:
  """Trajectories per sample, keyed by method label in the order of `methods`."""
  methods = parse_methods(methods)
  needs_pidu = any(m in ('PIDU', 'TRF(pidu)') for m in methods)
  if needs_pidu and pidu_model is None:
    raise ValueError(f'{methods} needs a PIDU model')
  pidu = run_pidu(pidu_model, dataset, indices) if needs_pidu else [None] * len(indices)

  out = []
  for index, pidu_traj in zip(indices, pidu):
    trajs = {}
    for m in methods:
      if m == 'PIDU':
        trajs[m] = pidu_traj
        continue
      scheme, start = trf_scheme(m), p_init
      if start is None and scheme == 'pidu':
        _, _, _, p_true, mask = trf_problem(dataset, index)
        start = pt2np(trf_init('pidu', p_true, mask, dataset, pidu_traj.p[0])).ravel()
      trajs[m] = run_trf(dataset, index, init=scheme, p_init=start, max_iter=max_iter)
    if verbose:
      print(f'sample {index}: ' + ' | '.join(f'{m} {t.n_steps - 1} steps' for m, t in trajs.items()))
    out.append({m: trajs[m] for m in methods})
  return out


def method_slug(method: str) -> str:
  """'TRF(noisy)' -> 'TRF_noisy': for file names, npz keys and csv columns."""
  return re.sub(r'\W+', '_', method).strip('_')


# ─────────────────────────────────────────────────────────────────────────────
# Frames
# ─────────────────────────────────────────────────────────────────────────────

def frame_steps(n_steps: int, max_frames: int = 20) -> np.ndarray:
  """Steps shown in the animation: all steps if they fit in max_frames, otherwise max_frames
  logarithmically spaced steps (dense at the start), always including the first and last."""
  if n_steps <= max_frames:
    return np.arange(n_steps)
  for num in range(max_frames, 10 * n_steps):
    steps = np.unique(np.round(np.geomspace(1, n_steps, num)).astype(int) - 1)
    if steps.size >= max_frames:
      return steps
  return np.arange(n_steps)


def frame_durations(n_frames: int, frame_ms: int, hold_ms: int) -> List[int]:
  return [frame_ms] * (n_frames - 1) + [max(frame_ms, hold_ms)]


def _style(ax):
  ax.grid(True, color=GRID, lw=0.6)
  ax.set_axisbelow(True)
  for side in ('top', 'right'):
    ax.spines[side].set_visible(False)
  for side in ('left', 'bottom'):
    ax.spines[side].set_color(INK_MUTED)
  ax.tick_params(colors=INK_MUTED, labelsize=8)


def _limits(anchor: np.ndarray, values: Sequence[np.ndarray], pad: float = 0.06, q: float = 0.02):
  """Axis limits covering `anchor` (measured spectrum / reference value) fully and, per method, the
  2-98 % quantile range of its values: a few steps of one diverging method do not squash the other
  panels, and a method with few steps (PIDU) stays fully visible."""
  bounds = [b for x in [anchor] for b in np.asarray(x, dtype=float).ravel()]
  for x in values:
    v = np.asarray(x, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if v.size:
      bounds += list(np.quantile(v, [q, 1 - q]))
  bounds = [b for b in bounds if np.isfinite(b)]
  lo, hi = (min(bounds), max(bounds)) if bounds else (0.0, 1.0)
  span = hi - lo if hi > lo else max(abs(lo), 1.0) * 0.1
  return lo - pad * span, hi + pad * span


class SpectrumPanel:
  """Measured spectrum and the simulated spectrum of the current estimate."""

  def __init__(self, ax, sample: Sample, traj: Trajectory, channel: str, ylim):
    self.ax, self.traj, self.channel = ax, traj, channel
    wn = sample.wn[channel]
    _style(ax)
    ax.plot(wn, sample.y[channel], color=INK, lw=1.0, label='measured')
    (self.line,) = ax.plot(wn, traj.y[channel][0], color=COLORS[traj.method], lw=1.6, label=f'{traj.method} estimate')
    ax.set_xlim(wn.min(), wn.max())
    ax.set_ylim(*ylim)

  def update(self, step: int):
    s = min(step, self.traj.n_steps - 1)
    self.line.set_ydata(self.traj.y[self.channel][s])
    return s


class ParamPanel:
  """One parameter against the update step, revealed up to the current step."""

  def __init__(self, ax, traj: Trajectory, idx: int, true: float, ylim):
    self.ax, self.traj, self.idx = ax, traj, idx
    _style(ax)
    ax.axhline(true, color=INK_MUTED, lw=1.0, ls='--', label='reference')
    (self.line,) = ax.plot([], [], color=COLORS[traj.method], lw=1.6)
    (self.dot,) = ax.plot([], [], 'o', color=COLORS[traj.method], ms=5, mec='white', mew=1.0)
    ax.set_ylim(*ylim)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

  def update(self, step: int):
    s = min(step, self.traj.n_steps - 1)
    v = self.traj.p[: s + 1, self.idx]
    self.line.set_data(np.arange(s + 1), v)
    self.dot.set_data([s], [v[-1]])
    self.ax.set_xlim(0, max(s, 1))
    return s


def render_gif(fig, update: Callable[[int], None], steps: Sequence[int], durations: Sequence[int], path: Path, dpi: int):
  """Render one frame per step and save the frames as a looping GIF with per-frame durations."""
  fig.set_dpi(dpi)
  frames = []
  for step in steps:
    update(int(step))
    fig.canvas.draw()
    rgb = Image.frombuffer('RGBA', fig.canvas.get_width_height(physical=True), fig.canvas.buffer_rgba()).convert('RGB')
    frames.append(rgb.quantize(colors=128, method=Image.Quantize.MEDIANCUT))
  path.parent.mkdir(parents=True, exist_ok=True)
  frames[0].save(path, save_all=True, append_images=frames[1:], duration=list(durations), loop=0, optimize=True)


def _step_title(traj: Trajectory, step: int) -> str:
  last = traj.n_steps - 1
  s = min(step, last)
  return f'{traj.method}  ·  step {s}/{last}' + ('  (done)' if step >= last else '')


# ─────────────────────────────────────────────────────────────────────────────
# Single-sample animation
# ─────────────────────────────────────────────────────────────────────────────

def make_fit_gif(
  sample: Sample,
  trajectories: Dict[str, Trajectory],
  params: Sequence[str],
  path: Path,
  channel: str = 'system1',
  frame_ms: int = 250,
  hold_ms: int = 2000,
  max_frames: int = 20,
  gif: bool = True,
  data: bool = True,
  dpi: int = 100,
  title: str = '',
) -> Dict[str, Path]:
  """GIF of the spectrum and the tracked parameters per step (columns: methods), and its data."""
  methods = list(trajectories)
  idxs = [param_index(sample, name) for name in params]
  n_steps = max(t.n_steps for t in trajectories.values())
  steps = frame_steps(n_steps, max_frames)
  durations = frame_durations(len(steps), frame_ms, hold_ms)
  written: Dict[str, Path] = {}

  if gif:
    fig, axes = plt.subplots(1 + len(idxs), len(methods), figsize=(4.6 * len(methods), 2.0 + 1.7 * len(idxs)),
                             squeeze=False, sharey='row', gridspec_kw={'height_ratios': [1.6] + [1] * len(idxs)})
    ylim_y = _limits(sample.y[channel], [t.y[channel] for t in trajectories.values()])
    panels = []
    for c, m in enumerate(methods):
      traj = trajectories[m]
      panels.append(SpectrumPanel(axes[0, c], sample, traj, channel, ylim_y))
      for r, (name, i) in enumerate(zip(params, idxs), start=1):
        ylim = _limits(sample.p_true[i:i + 1], [t.p[:, i] for t in trajectories.values()])
        panels.append(ParamPanel(axes[r, c], traj, i, sample.p_true[i], ylim))
        if c == 0:
          axes[r, c].set_ylabel(name, color=INK, fontsize=9)
        if r == len(idxs):
          axes[r, c].set_xlabel('update step', color=INK, fontsize=9)
      axes[0, c].set_xlabel('wavenumber (cm$^{-1}$)', color=INK, fontsize=9)
      axes[0, c].legend(loc='best', fontsize=7, frameon=False)
    axes[0, 0].set_ylabel(CHANNEL_LABELS.get(channel, channel), color=INK, fontsize=9)
    axes[1, 0].legend(loc='best', fontsize=7, frameon=False)
    fig.suptitle(title, color=INK, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    def update(step: int):
      for p in panels:
        p.update(step)
      for c, m in enumerate(methods):
        axes[0, c].set_title(_step_title(trajectories[m], step), color=INK, fontsize=10, loc='left')

    written['gif'] = path.with_suffix('.gif')
    render_gif(fig, update, steps, durations, written['gif'], dpi)
    plt.close(fig)

  if data:
    written |= export_data(path, [sample], [trajectories], steps, durations, params=params)
  return written


def export_data(path: Path, samples: Sequence[Sample], trajectories: Sequence[Dict[str, Trajectory]],
                steps: Sequence[int], durations: Sequence[int], params: Optional[Sequence[str]] = None) -> Dict[str, Path]:
  """Write the data behind a GIF: frames (csv), tracked parameters (csv) and all arrays (npz)."""
  path.parent.mkdir(parents=True, exist_ok=True)
  written = {}

  frames = pd.DataFrame({'frame': np.arange(len(steps)), 'step': steps, 'duration_ms': durations})
  for m in trajectories[0]:
    n = max(t[m].n_steps for t in trajectories)
    frames[f'step_{method_slug(m)}'] = np.minimum(steps, n - 1)
  written['frames'] = path.with_name(path.stem + '_frames.csv')
  frames.to_csv(written['frames'], index=False)

  if params:
    rows = []
    for sample, trajs in zip(samples, trajectories):
      for name in params:
        i = param_index(sample, name)
        for m, t in trajs.items():
          rows += [{'sample_idx': sample.index, 'method': m, 'step': s, 'parameter': name,
                    'value': t.p[s, i], 'true': sample.p_true[i]} for s in range(t.n_steps)]
    written['params'] = path.with_name(path.stem + '_params.csv')
    pd.DataFrame(rows).to_csv(written['params'], index=False)

  arrays = {'frame_steps': np.asarray(steps), 'frame_durations_ms': np.asarray(durations)}
  for sample, trajs in zip(samples, trajectories):
    pre = f's{sample.index}_'
    arrays[pre + 'p_true'] = sample.p_true
    arrays[pre + 'param_names'] = np.asarray(sample.names)
    for ch in sample.y:
      arrays[pre + f'wn_{ch}'] = sample.wn[ch]
      arrays[pre + f'y_measured_{ch}'] = sample.y[ch]
    for m, t in trajs.items():
      arrays[pre + f'{method_slug(m)}_p'] = t.p  # [n_steps, n_params]
      for ch, v in t.y.items():
        arrays[pre + f'{method_slug(m)}_y_{ch}'] = v  # [n_steps, n_wn]
  written['npz'] = path.with_suffix('.npz')
  np.savez_compressed(written['npz'], **arrays)
  return written


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def add_common_args(parser: argparse.ArgumentParser):
  parser.add_argument('--usecase', type=int, default=usecase, help='use case (sets PIDU_USECASE)')
  parser.add_argument('--dataset', default=None, help=f'dataset name (default: {DEFAULT_DATASET}; without the experimental data: {SIMULATED_TEST[2]}). Simulated datasets are downloaded if missing')
  parser.add_argument('--checkpoint', type=Path, default=None, help='PIDU checkpoint (default: the released one, pidu.hub.RELEASED_SEEDS; downloaded if missing)')
  parser.add_argument('--methods', nargs='+', default=DEFAULT_METHODS,
                      help='columns, separated by spaces or "|": PIDU, TRF(noisy), TRF(random), TRF(baseline), TRF(pidu)')
  parser.add_argument('--max-iter', type=int, default=200, dest='max_iter', help='TRF max_nfev (as in the paper)')
  parser.add_argument('--channel', default='system1', help='spectrum channel to plot (system1 | system2_0 | system2_1)')
  parser.add_argument('--frame-ms', type=int, default=250, dest='frame_ms', help='duration of one frame')
  parser.add_argument('--hold-ms', type=int, default=2000, dest='hold_ms', help='duration of the last frame')
  parser.add_argument('--max-frames', type=int, default=20, dest='max_frames', help='log-spaced steps beyond this many')
  parser.add_argument('--no-gif', action='store_false', dest='gif')
  parser.add_argument('--no-data', action='store_false', dest='data')
  parser.add_argument('--dpi', type=int, default=100)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
  parser.add_argument('--output', type=Path, default=None, help='output directory (default: <case dir>/figures/fit_evolution)')


def output_dir(args) -> Path:
  return args.output or case_dir(usecase) / 'figures' / 'fit_evolution'


def load_for_methods(args, methods: Sequence[str]):
  """Dataset name, dataset and (only if a method needs it) the PIDU model."""
  ds_name = resolve_dataset(args.dataset)
  dataset = load_dataset(ds_name, args.device)
  model = load_pidu(args.checkpoint, args.device) if {'PIDU', 'TRF(pidu)'} & set(methods) else None
  return ds_name, dataset, model


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  add_common_args(parser)
  parser.add_argument('--index', type=int, default=0, help='sample index in the dataset')
  parser.add_argument('--params', nargs='+', default=None,
                      help=f'tracked parameters (default: {DEFAULT_PARAMS}); system names, w0_1 (k-th reference oscillator) or w0_slot<i>')
  args = parser.parse_args()
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)
  methods = parse_methods(args.methods)

  ds_name, dataset, model = load_for_methods(args, methods)
  sample = get_sample(dataset, args.index)
  params = args.params or DEFAULT_PARAMS[usecase]
  for name in params:
    param_index(sample, name)  # fail before fitting

  trajectories = run_methods(dataset, [args.index], methods, pidu_model=model, max_iter=args.max_iter, verbose=True)[0]

  stem = f"fit_evolution_{ds_name.split('/')[-1]}_{args.index:03d}_{'_'.join(map(method_slug, methods))}"
  written = make_fit_gif(
    sample, trajectories, params, output_dir(args) / stem, channel=args.channel,
    frame_ms=args.frame_ms, hold_ms=args.hold_ms, max_frames=args.max_frames, gif=args.gif, data=args.data,
    dpi=args.dpi, title=f'{ds_name}  ·  sample {args.index}',
  )
  for kind, p in written.items():
    print(f'Saved {kind:6s} {p}')


if __name__ == '__main__':
  sys.exit(main())
