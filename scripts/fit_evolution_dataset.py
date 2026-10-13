"""Animate how TRF and PIDU fit randomly drawn spectra of a dataset, step by step.

The spectrum panel of scripts/fit_evolution.py without the parameter tracking, for several
samples at once: one row per sample, one column per method (--methods, default
"TRF(noisy)|PIDU"; see scripts/fit_evolution.py). All panels advance together; PIDU freezes
after its unfolds, and each TRF fit freezes at its last iteration.

    python scripts/fit_evolution_dataset.py                          # 4 random experimental spectra (use case 2)
    python scripts/fit_evolution_dataset.py --n-samples 6 --seed 3 --methods "TRF(random)|TRF(noisy)|PIDU"
    python scripts/fit_evolution_dataset.py --indices 0 25 50 74 --channel system2_0
    python scripts/fit_evolution_dataset.py --usecase 1 --dataset usecase1/case1_test_n2_10_nwsn

Writes to <output> (default outputs/case<N>/figures/fit_evolution/), stem
fit_evolution_<dataset>_samples_<i>-<j>-..._<methods>:
  <stem>.gif           the animation
  <stem>_frames.csv    frame, duration_ms and the step shown per method
  <stem>.npz           per sample: wavenumbers, measured and simulated spectra per channel and step,
                       full parameter vectors per step, reference parameters and names
"""
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

import fit_evolution as fe


def make_dataset_gif(samples, trajectories, path, channel='system1', frame_ms=250, hold_ms=2000,
                     max_frames=20, gif=True, data=True, dpi=100, title=''):
  """GIF of the spectrum per step for several samples (rows) and methods (columns), and its data."""
  methods = list(trajectories[0])
  n_steps = max(t.n_steps for trajs in trajectories for t in trajs.values())
  steps = fe.frame_steps(n_steps, max_frames)
  durations = fe.frame_durations(len(steps), frame_ms, hold_ms)
  written = {}

  if gif:
    fig, axes = plt.subplots(len(samples), len(methods), figsize=(4.6 * len(methods), 0.8 + 2.3 * len(samples)),
                             squeeze=False, sharey='row')
    panels = []
    for r, (sample, trajs) in enumerate(zip(samples, trajectories)):
      ylim = fe._limits(sample.y[channel], [t.y[channel] for t in trajs.values()])
      for c, m in enumerate(methods):
        panels.append((r, c, trajs[m], fe.SpectrumPanel(axes[r, c], sample, trajs[m], channel, ylim)))
      axes[r, 0].set_ylabel(f'sample {sample.index}\n{fe.CHANNEL_LABELS.get(channel, channel)}', color=fe.INK, fontsize=9)
    for c in range(len(methods)):
      axes[-1, c].set_xlabel('wavenumber (cm$^{-1}$)', color=fe.INK, fontsize=9)
      axes[0, c].legend(loc='best', fontsize=7, frameon=False)
    fig.suptitle(title, color=fe.INK, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    def update(step: int):
      for r, c, traj, panel in panels:
        panel.update(step)
        axes[r, c].set_title(fe._step_title(traj, step), color=fe.INK, fontsize=9, loc='left')

    written['gif'] = path.with_suffix('.gif')
    fe.render_gif(fig, update, steps, durations, written['gif'], dpi)
    plt.close(fig)

  if data:
    written |= fe.export_data(path, samples, trajectories, steps, durations)
  return written


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  fe.add_common_args(parser)
  parser.add_argument('--n-samples', type=int, default=4, dest='n_samples', help='number of random samples')
  parser.add_argument('--indices', type=int, nargs='+', default=None, help='sample indices (instead of random ones)')
  args = parser.parse_args()
  torch.manual_seed(args.seed)
  rng = np.random.default_rng(args.seed)
  methods = fe.parse_methods(args.methods)

  ds_name, dataset, model = fe.load_for_methods(args, methods)
  indices = args.indices or sorted(rng.choice(len(dataset), size=min(args.n_samples, len(dataset)), replace=False).tolist())
  samples = [fe.get_sample(dataset, i) for i in indices]
  trajectories = fe.run_methods(dataset, indices, methods, pidu_model=model, max_iter=args.max_iter, verbose=True)

  stem = (f"fit_evolution_{ds_name.split('/')[-1]}_samples_{'-'.join(map(str, indices))}"
          f"_{'_'.join(map(fe.method_slug, methods))}")
  written = make_dataset_gif(
    samples, trajectories, fe.output_dir(args) / stem, channel=args.channel, frame_ms=args.frame_ms,
    hold_ms=args.hold_ms, max_frames=args.max_frames, gif=args.gif, data=args.data, dpi=args.dpi,
    title=ds_name,
  )
  for kind, p in written.items():
    print(f'Saved {kind:6s} {p}')


if __name__ == '__main__':
  sys.exit(main())
