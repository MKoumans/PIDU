import sys
import os
from turtle import title
import torch
from typing import Callable
import matplotlib.pyplot as plt
import pandas as pd


from pidu.model.networks.param_activations import ActivationEinf, ActivationW0, ActivationdWp
from pidu.model.networks.experimental import ActivationG
from pidu.data.utils import pt2np

wn = torch.linspace(0, 1000, 1000, dtype=torch.float32)

act_e_inf = ActivationEinf()
act_w0 = ActivationW0(wn, [125], 0)
act_wp = ActivationdWp()
act_g = ActivationG()

x = torch.linspace(-5, 5, 1000, dtype=torch.float32)

export_dat = True

def info_act_w0(f:Callable, x:torch.Tensor, export_dat:bool=False) -> dict:
  abs_min = torch.tensor(-100.0, dtype=torch.float32)
  abs_max = torch.tensor(100.0, dtype=torch.float32)
  
  # y = f(x) # 
  x = x[:,None,None]  # Add batch and system dimensions
  y = f(x)       # 
  
  print(f"{f}({x.min():.2f}, {x.max():.2f}) = ({y.min():.2f}, {y.max():.2f})")
  
  fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
  for region in range(y.shape[-1]):
    ax.plot(pt2np(x[:,0,0]), pt2np(y[:,0,region]))
    ax.set(
      title=f,
      xlabel='x',
      ylabel='y',
      xlim=(-5, 5),
      xscale='linear',
      yscale='linear',
    )
  
  os.makedirs('research/legacy/figures', exist_ok=True)
  fig.savefig(f'research/legacy/figures/{f}.png')
  plt.close(fig)
  
  if export_dat:
    os.makedirs('research/legacy/activation_data', exist_ok=True)
    export_path = f'research/legacy/activation_data/{f}.dat'
    print(f"Exporting data to {export_path}...")
    df = pd.DataFrame({
      'x': pt2np(x[:,0,0]),
      **{f'y_region_{i}': pt2np(y[:,0,i]) for i in range(y.shape[-1])}
    })
    df.to_csv(export_path, index=False, sep='\t')
    print(f"Data exported to {export_path}")

def info_act(f:Callable, x:torch.Tensor, export_dat:bool=False, yscale:str='log') -> dict:
  """

  """
  abs_min = torch.tensor(-100.0, dtype=torch.float32)
  abs_max = torch.tensor(100.0, dtype=torch.float32)
  y = f(x) # 
  
  print(f"{f}({x.min():.2f}, {x.max():.2f}) = ({y.min():.2f}, {y.max():.2f})")
  print(f"Absolute min: {pt2np(f(abs_min)):.4f}, Absolute max: {pt2np(f(abs_max)):.2f}")
  
  fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
  ax.plot(pt2np(x), pt2np(y))
  ax.set(
    title=f,
    xlabel='x',
    ylabel='y',
    xlim=(-5, 5),
    xscale='linear',
    yscale=yscale,
  )
  
  os.makedirs('research/legacy/figures', exist_ok=True)
  fig.savefig(f'research/legacy/figures/{f}.png')
  plt.close(fig)
  
  if export_dat:
    os.makedirs('research/legacy/activation_data', exist_ok=True)
    export_path = f'research/legacy/activation_data/{f}.dat'
    print(f"Exporting data to {export_path}...")
    df = pd.DataFrame({
      'x': pt2np(x),
      'y': pt2np(y)
    })
    df.to_csv(export_path, index=False, sep='\t')
    print(f"Data exported to {export_path}")

# info_act(act_e_inf, x, export_dat)  # 
info_act(act_wp, x, export_dat, yscale='linear')  #
info_act(act_g, x, export_dat, yscale='linear')  #
# info_act_w0(act_w0, x, export_dat)  # 