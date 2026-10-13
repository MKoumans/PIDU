"""Quickstart: extract Drude-Lorentz parameters from simulated thin-film spectra with PIDU.

Use case 1: a single layer measured in transmission, 0-4 Lorentz oscillators.

    python examples/quickstart.py             # released model if downloaded, else a short training run
    pidu-hub download --what models --usecase 1   # fetch the released checkpoints first (optional)

Writes outputs/examples/quickstart.png (measured vs. PIDU-reconstructed spectrum).
"""
import argparse
import os

os.environ.setdefault('PIDU_USECASE', '1')  # read by pidu at import time

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from pidu import train
from pidu.data.simulation import create_dataset, init_usecase
from pidu.model.networks.io import load_checkpoint
from pidu.model.trainer import Trainer
from pidu.paths import MODELS_DIR, OUTPUTS_DIR

USECASE = 1
RELEASED = MODELS_DIR / 'usecase1' / 'PIDU-t01-00.pt'


def simulate(name, n_log2, device):
  """Simulate 2**n_log2 spectra with random layer/oscillator parameters (TMM forward model)."""
  p_gen, sys_pars = init_usecase(usecase=USECASE, device=device)
  return create_dataset(usecase=USECASE, setname=f'quickstart_{name}', n_samples=2**n_log2,
                        system_params=sys_pars, p_gen_params=p_gen, verbose=False, export=False,
                        device=device)


def train_model(device, epochs):
  """Train a PIDU model from scratch with the settings of configs/case1.yaml."""
  trainset, valset = simulate('train', 10, device), simulate('val', 6, device)
  trainset.y_return_mode = valset.y_return_mode = 'noised'
  model = train.get_model(model_type='PIDU', ds_info=trainset.get_info_for_model(), device=device,
                          dtype=torch.float32, usecase=USECASE, active_models=['system1'])
  cfg = train.load_config('configs/case1.yaml')
  run_folder = (OUTPUTS_DIR / 'examples' / 'runs').as_posix()
  trainer = Trainer(model, training_system=train.build_training_system(cfg, run_folder), loss_matrix=True)
  trainer.load_dataset(to_import={'train': trainset, 'val': valset})
  trainer.train(epochs, save_epoch=[], save_idx=[], plot_intermediate=False,
                history_save_intermediate=False, outputname_history='quickstart')
  return model


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('--epochs', type=int, default=10, help='training epochs when no released model is present')
  parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
  args = parser.parse_args()
  device = torch.device(args.device)
  torch.manual_seed(0)

  if RELEASED.exists():
    print(f'Loading released model {RELEASED.name}')
    model = load_checkpoint(RELEASED, map_location=device)
  else:
    print(f'No released model at {RELEASED}; training PIDU for {args.epochs} epochs')
    model = train_model(device, args.epochs)
  model.eval()

  # Predict: the model refines its estimate over `n_unfolds` steps, each re-simulating the spectrum.
  testset = simulate('test', 4, device)
  testset.y_return_mode = 'noised'
  wn, x_sys, x_osc, y, _ = next(iter(DataLoader(testset, batch_size=16)))
  with torch.no_grad():
    out = model(wn=wn, y=y, x_sys=x_sys, x_osc=x_osc)

  s = 0                                        # sample to report
  p_sys_hat = out['p_sys']['system1'][-1, s, 0]  # last unfold: [n_system_parameters]
  p_sys_true = x_sys[1]['system1'][s, 0]
  print('\nSystem parameters (true -> predicted):')
  for name, t, p in zip(testset.col_names_sys['system1'], p_sys_true.tolist(), p_sys_hat.tolist()):
    print(f'  {name:>10s} {t:12.4g} -> {p:12.4g}')

  osc_true = x_osc[1]['system1'][s, 0].reshape(-1, 4)          # [cells, (confidence, w0, wp, g)]
  osc_hat = out['p_osc']['system1'][-1, s, 0].reshape(-1, 4)
  fmt = lambda v: '(' + ', '.join(f'{x:7.1f}' for x in v.tolist()) + ')'
  print('\nOscillators per detection cell (confidence, w0, wp, g in cm^-1; true -> predicted):')
  for c in range(osc_true.shape[0]):
    if osc_true[c, 0] > 0.5 or osc_hat[c, 0] > 0.75:  # 0.75: the model's confidence threshold
      print(f'  cell {c}: {fmt(osc_true[c])} -> {fmt(osc_hat[c])}')

  path = OUTPUTS_DIR / 'examples' / 'quickstart.png'
  path.parent.mkdir(parents=True, exist_ok=True)
  fig, ax = plt.subplots(figsize=(7, 3.5))
  ax.plot(wn['system1'][s].cpu(), y['system1'][s].cpu(), 'k', lw=1.2, label='measured (simulated, noised)')
  ax.plot(wn['system1'][s].cpu(), out['y']['system1'][-1, s].cpu(), 'r--', lw=1.2, label='PIDU reconstruction')
  ax.set(xlabel='wavenumber (cm$^{-1}$)', ylabel='transmittance')
  ax.legend()
  fig.tight_layout()
  fig.savefig(path, dpi=150)
  print(f'\nSaved {path}')


if __name__ == '__main__':
  main()
