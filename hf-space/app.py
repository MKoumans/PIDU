"""PIDU live demo (Hugging Face Space): pick a simulated test spectrum and run the released PIDU model on it.

pidu.config reads PIDU_USECASE at import, so every use case runs in its own worker process; the Gradio
app in the main process only plots. Checkpoints and test sets are downloaded from the Hugging Face Hub
on first use (MKoumans/pidu-optical-parameter-extraction, MKoumans/pidu-data) into PIDU_ROOT.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault('PIDU_ROOT', str(Path(__file__).resolve().parent / 'pidu-root'))

DATASETS = {1: 'usecase1/case1_test_n2_10_nwsn', 2: 'usecase2/case2_test_n2_10_wsn'}
CHANNEL_LABELS = {'system1': 'Reflectance', 'system2_0': 'Transmission (Re t)', 'system2_1': 'Transmission (Im t)'}

# ─────────────────────────────────────────────────────────────────────────────
# Worker process (one per use case)
# ─────────────────────────────────────────────────────────────────────────────

_state = {}


def _init_worker(usecase: int):
  os.environ['PIDU_USECASE'] = str(usecase)
  import torch
  from pidu.data.simulation import load_sim_ds
  from pidu.hub import dataset_file, released_checkpoint
  from pidu.model.networks.io import load_checkpoint

  torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
  dataset_file(DATASETS[usecase])
  model = load_checkpoint(released_checkpoint(usecase, 'PIDU'), map_location='cpu')
  model.eval()
  _state.update(usecase=usecase, model=model, dataset=load_sim_ds(DATASETS[usecase], y_return_mode='noised', device='cpu'))


def _size() -> int:
  return len(_state['dataset'])


def _infer(index: int) -> dict:
  """PIDU's estimate and simulated spectrum after every unfold for one test spectrum (as in
  scripts/fit_evolution.py: `_mask` and `_forward_system` are wrapped to record each stage)."""
  import numpy as np
  import torch
  from torch.utils.data import DataLoader, Subset
  from pidu.data.utils import pt2np

  model, dataset = _state['model'], _state['dataset']
  wn, x_sys, x_osc, y, _ = next(iter(DataLoader(Subset(dataset, [index]), batch_size=1)))
  params, spectra = [], []
  mask, forward_system = model._mask, model._forward_system

  def record_mask(*args, **kwargs):
    out = mask(*args, **kwargs)
    _, sys, osc = out
    params.append(torch.cat([sys['system1'][:, 0], osc['system1'][:, 0, ..., -3:].flatten(1)], dim=1))
    return out

  def record_forward(*args, **kwargs):
    out = forward_system(*args, **kwargs)
    spectra.append({ch: v[:, 0] for ch, v in out.items()})
    return out

  model._mask, model._forward_system = record_mask, record_forward
  t0 = time.perf_counter()
  try:
    with torch.no_grad():
      model(wn=wn, y=y, x_sys=x_sys, x_osc=x_osc)
  finally:
    del model._mask, model._forward_system
  ms = (time.perf_counter() - t0) * 1000

  p_hat = pt2np(torch.cat(params)).astype(float)  # [step, param]
  wn1, xs1, xo1, y1, _ = dataset[index]  # unbatched, as in scripts/fit_evolution.get_sample
  p_true = pt2np(torch.cat([xs1[1]['system1'], xo1[1]['system1'][..., -3:].flatten(1)], dim=1)).ravel().astype(float)
  names = list(dataset.col_names_sys['system1'])
  osc = p_true[len(names):].reshape(-1, 3)
  present = np.flatnonzero(np.any(osc != 0, axis=1))
  for s in range(osc.shape[0]):
    tag = f'_{np.searchsorted(present, s) + 1}' if s in present else f'_slot{s}'
    names += [a + tag for a in ('w0', 'wp', 'g')]
  keep = [k for k, n in enumerate(names) if not re.search(r'_slot\d+$', n) and not np.allclose(p_hat[:, k], p_true[k])]

  def wn_of(ch):
    return pt2np(wn1[ch if ch in wn1 else ch.rsplit('_', 1)[0]]).ravel()

  return {
    'ms': ms,
    'names': [names[k] for k in keep],
    'p_true': p_true[keep],
    'p_hat': p_hat[:, keep],
    'spectra': {ch: (wn_of(ch), pt2np(y1[ch]).ravel(), np.stack([pt2np(s[ch]).ravel() for s in spectra])) for ch in y1},
  }


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────

GREEK = {'w0': 'ω₀', 'wp': 'ωₚ', 'g': 'γ'}


def label(name: str) -> str:
  if name == 'eps_inf':
    return 'ε∞'
  if m := re.fullmatch(r'(w0|wp|g)_before', name):
    return f'{GREEK[m[1]]} soft mode'
  if m := re.fullmatch(r'(w0|wp|g)_(\d+)', name):
    return f'{GREEK[m[1]]} osc. {m[2]}'
  return name


def main():
  import gradio as gr
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  import numpy as np
  import pandas as pd

  ctx = mp.get_context('spawn')
  workers = {uc: ProcessPoolExecutor(1, mp_context=ctx, initializer=_init_worker, initargs=(uc,)) for uc in DATASETS}
  sizes = {uc: workers[uc].submit(_size) for uc in DATASETS}  # starts loading both use cases right away
  cache = {}

  def run(uc: int, index: int) -> dict:
    key = (uc, index)
    if key not in cache:
      cache[key] = workers[uc].submit(_infer, index).result()
      if len(cache) > 256:
        cache.pop(next(iter(cache)))
    return cache[key]

  def figure(res: dict, step: int):
    chans = list(res['spectra'])
    fig, axes = plt.subplots(len(chans), 1, figsize=(10, 2.6 * len(chans)), squeeze=False)
    for ax, ch in zip(axes[:, 0], chans):
      wn, y, y_hat = res['spectra'][ch]
      ax.plot(wn, y, color='#222', lw=1.0, label='measured')
      ax.plot(wn, y_hat[step], color='#eb6834', lw=1.8, label=f'PIDU · step {step}')
      ax.set_ylabel(CHANNEL_LABELS.get(ch, ch), fontsize=9)
      ax.set_xlim(wn.min(), wn.max())
      ax.grid(alpha=.3)
      for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8, loc='best')
    axes[-1, 0].set_xlabel('wavenumber (cm$^{-1}$)')
    fig.tight_layout()
    plt.close(fig)
    return fig

  def table(res: dict, step: int):
    p, t = res['p_hat'][step], res['p_true']
    err = np.abs(p - t) / np.maximum(np.abs(t), 1e-12) * 100
    return pd.DataFrame({'parameter': [label(n) for n in res['names']], 'reference': t.round(3),
                         'PIDU': p.round(3), 'error (%)': err.round(1)})

  def update(uc_label: str, index: int, step: int):
    uc = 1 if uc_label.startswith('Use case 1') else 2
    res = run(uc, int(index))
    step = min(int(step), len(res['p_hat']) - 1)
    final = res['p_hat'][-1]
    mape = float(np.mean(np.abs(final - res['p_true']) / np.maximum(np.abs(res['p_true']), 1e-12)) * 100)
    info = f'**Final MAPE {mape:.1f}%** · PIDU inference {res["ms"]:.0f} ms on a shared CPU (3 unfolds)'
    return figure(res, step), table(res, step), info

  def pick_case(uc_label: str):
    uc = 1 if uc_label.startswith('Use case 1') else 2
    return gr.update(maximum=sizes[uc].result() - 1, value=0)

  with gr.Blocks(title='PIDU live demo') as demo:
    gr.Markdown(
      '# Physics-infused Deep Unfolding: live demo\n'
      'Pick a simulated test spectrum; the released PIDU checkpoint fits it in three unfolding stages. '
      'Step 0 is PIDU\'s own initial estimate. '
      '[Project page](https://mkoumans.github.io/PIDU/) · [Code](https://github.com/MKoumans/PIDU) · '
      '[Models](https://huggingface.co/MKoumans/pidu-optical-parameter-extraction) · '
      '[Data](https://huggingface.co/datasets/MKoumans/pidu-data)')
    with gr.Row():
      case = gr.Radio(['Use case 1 · single layer', 'Use case 2 · SrTiO₃-like, multi-channel'],
                      value='Use case 2 · SrTiO₃-like, multi-channel', label='Model')
      index = gr.Slider(0, 1023, value=0, step=1, label='Test spectrum index')
      step = gr.Slider(0, 3, value=3, step=1, label='Unfolding step')
    info = gr.Markdown()
    plot = gr.Plot(label='Spectrum')
    params = gr.Dataframe(label='Parameters', interactive=False)

    inputs = [case, index, step]
    outputs = [plot, params, info]
    case.change(pick_case, case, index).then(update, inputs, outputs)
    index.release(update, inputs, outputs)
    step.change(update, inputs, outputs)
    demo.load(update, inputs, outputs)

  demo.queue(default_concurrency_limit=2).launch()


if __name__ == '__main__':
  main()
