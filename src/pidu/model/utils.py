from typing import Any, Dict, List, Tuple
import matplotlib.pyplot as plt
from os import makedirs, listdir, path
import numpy as np
from numpy.typing import NDArray
import glob
import sys
from PIL import Image
from matplotlib.gridspec import GridSpec


import torch

from pidu.data.utils import pt2np
from pidu.paths import MODELS_DIR

def create_masks_sys(usecase:int, n_layers:int, n_sys_pars:int, **kwargs):
  dt = kwargs.get('dt', torch.float32)
  dv = kwargs.get('dv', 'cpu')
  masks_sys = {}
  
  if usecase == 1:
    mask_sys = torch.zeros((n_layers, n_sys_pars), dtype=dt, device=dv)
    # mask_sys[:, -6:] = torch.tensor(1, dtype=dt, device=dv)   # Only last 6 parameters are trainable (A, w, d, w0, g, wp)
    mask_sys[:, 2] = torch.tensor(1, dtype=dt, device=dv)     # Make 'eps_inf' trainable
  elif usecase == 2:
    mask_sys = torch.zeros((n_layers, n_sys_pars), dtype=dt, device=dv)
    # mask_sys = torch.ones((n_layers, n_sys_pars), dtype=dt, device=dv)
    # mask_sys[:, :2] = torch.tensor(0, dtype=dt, device=dv)   # Suppress first 2 parameters (th0, d)
    # # mask_sys[:, 3:5] = torch.tensor(0, dtype=dt, device=dv)  # Suppress 4th and 5th parameters (drude_wp, drude_g)
    # mask_sys[:, -3:] = torch.tensor(0, dtype=dt, device=dv)  # Suppress last 3 parameters (w0, g, wp) after
    mask_sys[:, -6:-3] = torch.tensor(1, dtype=dt, device=dv) # Make parameters A, w, d trainable
  elif usecase == 3:
    mask_sys = torch.zeros((n_layers, n_sys_pars), dtype=dt, device=dv)
    mask_sys[:, 2] = torch.tensor(1, dtype=dt, device=dv)   # Make 'eps_inf' trainable
    mask_sys[:, -6:] = torch.tensor(1, dtype=dt, device=dv) # Make last 6 parameters trainable (A, w, d, w0, g, wp)

  masks_sys['system1'] = mask_sys  # Mask for the first system
  masks_sys['system2'] = mask_sys  # Mask for the second system 
  
  return masks_sys

def create_masks_osc(n_layers, n_scales, n_regions_max, n_attribs_label, std_val:Dict[str, int], **kwargs):
  dt = kwargs.get('dt', torch.float32)
  dv = kwargs.get('dv', 'cpu')
  
  masks_osc = {}
  
  systems_scales = list(n_regions_max.keys())
  systems_regions = list(n_regions_max.keys())

  assert systems_scales == systems_regions, "Systems scales and regions must match"
  
  for system_name in systems_scales:
    value = value= std_val.get(system_name, 1)
    system_n_reg = n_regions_max[system_name]
    system_scale = n_scales[system_name]
    
    mask_osc = torch.full((n_layers, system_scale, system_n_reg, n_attribs_label), value, dtype=dt, device=dv)
    # mask_osc = mask_osc.reshape(1, n_layers, -1)  

    masks_osc[system_name] = mask_osc

  # mask_osc[:,0] = 0 # Suppress first scale 
  # ---

  # -- Create mask [2-layer] --
  # n_layers = 2
  # mask_sys = np.zeros((n_layers, args.n_system_parameters))
  # mask_osc = np.ones((n_layers, 4))
  # mask_sys[:, 3:] = 1 
  
  # model_system['mask_sys'] = mask_sys
  # model_system['mask_osc'] = mask_osc
  # ---

  return masks_osc

def create_masks_p(usecase:int, n_layers:int, n_sys_pars:dict, n_osc_pars:dict, **kwargs) -> Dict[str, torch.Tensor]:
  dt = kwargs.get('dt', torch.int16)
  dv = kwargs.get('dv', 'cpu')
  masks = {}
  
  # 0 = free, 1 = fixed
  for system_name in ['system1']:
      mask_sys = torch.ones_like(n_sys_pars[system_name], dtype=dt, device=dv)
      if usecase == 1:
          mask_sys[:, 2] = torch.tensor(0, dtype=dt, device=dv)  # Only 'eps_inf' trainable
          # mask_sys[:, -6:] = torch.tensor(0, dtype=dt, device=dv) 
      elif usecase == 2:
          mask_sys[:, -6:-3] = torch.tensor(0, dtype=dt, device=dv) # Only parameters A, w, d trainable
      
      mask_osc = torch.zeros_like(n_osc_pars[system_name][..., -3:], dtype=dt, device=dv).flatten(1) 
      
      masks[system_name] = torch.cat([mask_sys, mask_osc], dim=1)
  
  masks['system2'] = masks['system1'].clone()  # Use the same mask for system2
  
  return masks

def create_mask_wn(wn:torch.Tensor, system_name:str) -> torch.Tensor:
  # -- wn mask ---
  if system_name.startswith('system1'):
    ranges = (30, torch.inf)
  elif system_name.startswith('system2'):
    ranges = (2.5, 65)
  
  mask_fd_r = torch.logical_and(ranges[0] <= wn, wn <= ranges[1])
  
  return mask_fd_r

def plot_intermediate_y(
  y:torch.Tensor|Dict[str, torch.Tensor], 
  idx:List[int], 
  info:Dict[str, str],
  plot_kwgs:Dict[str, Any],
):
  ds_name = info['ds']        # Dataset name
  epoch = info['epoch']       # Epoch
  system_name = info.get('sys', 'system')   # System name
  wn_mask = plot_kwgs.get('wn_mask', None)  # Wavenumber mask
  plt_res = plot_kwgs.get('plt_res', False) # DPI

  save_path = plot_kwgs.get('plot_dir', f'outputs/intermediate/{ds_name}/y') # Save path
  makedirs(save_path, exist_ok=True) # Create directory for each dataset
  
  if isinstance(y, torch.Tensor):
    y = {'system': y}

  for i_sample in range(len(idx)):
    y_keys = list(y.keys())
    n_unfolds = y[y_keys[0]].shape[0] - 2  # minus x, y_true
    n_systems = len(y_keys) # 
    dataset_idx = idx[i_sample]
    colors = ['tab:green', 'tab:blue', 'tab:red']
    
    fig_width = max(10, 3 * n_systems)
    fig_height = max(3, 2 * n_unfolds)
    plt.ioff() # Turn off interactive mode to prevent blocking
    fig, axes = plt.subplots(
      n_unfolds, 
      n_systems, 
      figsize=(fig_width, fig_height), 
      dpi=300, 
      sharex='col',
      squeeze=False
    )
    
    plt.subplots_adjust(
      wspace=0.2,
      hspace=0
    )
    
    # for row in range(n_unfolds):
    #   axes[row, 2].sharey(axes[row, 1]) # system2 shares y-axis
    for i_system, system_name in enumerate(y_keys):
      y_np = pt2np(y[system_name])
      if wn_mask is not None:
        wn_mask_subsystem = next((v for k, v in wn_mask.items() if system_name.startswith(k)), None)
        if wn_mask_subsystem is not None:
          y_np[1:] = np.where(pt2np(wn_mask_subsystem), y_np[1:], np.zeros_like(y_np)[1:])

      x_ = y_np[0, i_sample, :]     # 
      y_true = y_np[1, i_sample, :] # 
    
      for i_unfold in range(n_unfolds):
        ax = axes[i_unfold][i_system]
        
        x_sys_pd = info.get('sys_pd', None) # 
        x_sys_tr = info.get('sys_tr', None) # 
      
        ax.set(
          # xlabel='Wavenumbers (cm^-1)',
          # ylabel='Signal',
          # ylim=(y_true.min()*1.1, y_true.max()*1.1),
          xlim=(x_.min(), x_.max()),
        )

        ax.grid(which='major', color='black', linestyle='-', linewidth=0.5)

        if i_unfold < n_unfolds - 2: 
          ax.label_outer()
        else: 
          ax.set_xlabel('Wavenumbers (cm^-1)')
          ax.tick_params(axis='x', labelrotation=45, labelsize=8)

        ax.plot(x_, y_true, color=colors[0], label='True')
        ax.plot(x_, y_np[2+i_unfold, i_sample, :], color=colors[1], label='Predicted')
        if plt_res: 
          abs_res = np.abs(y_np[2+i_unfold, i_sample, :] - y_true)
          ax.plot(x_, abs_res, linestyle='dashed', color=colors[2], label=f'Residuals{abs_res.sum():.1e}')
        
        # ax.legend(loc='upper right', fontsize=4)
  
    # Add parameter / metadata text in right column (formatted and wrapped)
    # try to retrieve parameter predictions / truths if available in info
    p_sys = info.get('sys', None) #
    p_osc = info.get('osc', None) # 
    
    meta_lines = []
    meta_lines.append(f"Datasetname: {ds_name}\nEpoch:{epoch}\nSample:{dataset_idx}]")
    if p_sys is not None:
      try:
        p_sys_true = pt2np(p_sys)[ds_name][system_name][0,i_sample,0]   # 
        p_sys_pred = pt2np(p_sys)[ds_name][system_name][1:,i_sample,0]  # 
        meta_lines.append(f" - True: {p_sys_true.shape}")
        meta_lines.append(f" - Pred: {p_sys_pred.shape}")
      except Exception:
        meta_lines.append("System parameters: (unavailable)")
    if p_osc is not None:
      try:
        p_osc_true = pt2np(p_osc)[ds_name][system_name][0,i_sample,:,:]   # 
        p_osc_pred = pt2np(p_osc)[ds_name][system_name][1:,i_sample,:,:]  # 
        meta_lines.append(f" - True: {p_osc_true.shape}")
        meta_lines.append(f" - Pred: {p_osc_pred.shape}")
      except Exception:
        meta_lines.append("Oscillator parameters: (unavailable)")

    # If detailed numeric formatting available (keep brief)
    # meta_lines.append(f" - System parameters:")
    # if p_sys_true is not None and p_sys_pred is not None:
    #   try:
    #     pd_vals = np.round(p_sys_pred, 2)
    #     tr_vals = np.round(p_sys_true, 2)
    #     meta_lines.append(f'p_true:\n-{tr_vals[dataset_idx, 0, :5]}\n-{tr_vals[dataset_idx, 0, 5:8]}\n-{tr_vals[dataset_idx, 0, 8:11]}')
    #     for i in range(n_unfolds):
    #       meta_lines.append(f'{i}:\n-{pd_vals[i,dataset_idx, 0, :5]}\n-{pd_vals[i, dataset_idx, 0, 5:8]}\n-{pd_vals[i, dataset_idx, 0, 8:11]}')
    #   except Exception:
    #     pass

    # meta_lines.append(f" - Oscillator parameters:")
    # if p_osc_true is not None and p_osc_pred is not None:
    #   try:
    #     od_vals = np.round(p_osc_pred, 2)
    #     tr_vals = np.round(p_osc_true, 2)
    #     meta_lines.append(f'p_true:\n-{tr_vals[dataset_idx, 0, :5]}\n-{tr_vals[dataset_idx, 0, 5:8]}\n-{tr_vals[dataset_idx, 0, 8:11]}')
    #     for i in range(n_unfolds):
    #       meta_lines.append(f'{i}:\n-{od_vals[i,dataset_idx, 0, :5]}\n-{od_vals[i, dataset_idx, 0, 5:8]}\n-{od_vals[i, dataset_idx, 0, 8:11]}')
    #   except Exception:
    #     pass

    # Draw text
    text_block = "\n".join(meta_lines)
    # ax_text.text(
    #   0, 1,  
    #   text_block, 
    #   transform=ax_text.transAxes,
    #   va='top', 
    #   ha='left', 
    #   fontsize=8, 
    #   # family='monospace'
    #   bbox=dict(facecolor='white', alpha=0.5, edgecolor='black')
    # )
  
    if len(y_keys) > 1:
      system_name = path.commonprefix(y_keys).rstrip('_')
    
    plt.savefig(f'{save_path}/{system_name}_{epoch}_{dataset_idx}.png')
    plt.close(fig)
    plt.ion() # Turn on interactive mode again  

def plot_intermediate_x(
  sys:torch.Tensor|Dict[str, torch.Tensor], 
  osc:torch.Tensor|Dict[str, torch.Tensor]|None, 
  params_to_plot:List[int],
  idx:List[int], 
  info:Dict[str,str]={},
  plot_kwgs:Dict[str, Any]={},
):
  ds_name = info['ds']        # 
  epoch = info['epoch']       #
  plt_res = plot_kwgs.get('plt_res', False) #

  save_path = f'outputs/intermediate/{ds_name}/p' # 
  makedirs(save_path, exist_ok=True)  # Create directory for each dataset

  if isinstance(sys, torch.Tensor): sys = {'system': sys}
  if isinstance(osc, torch.Tensor): osc = {'system': osc}

  for i_sample in range(len(idx)):
    dataset_idx = idx[i_sample] # Get the index of the sample in the dataset, given list of indices
    n_params_to_plot = len(params_to_plot)
    sys_keys = list(sys.keys())
    n_systems = len(sys_keys) #
    colors = ['tab:green', 'tab:blue', 'tab:red']

    fig_width = max(10, 2 * n_systems)
    fig_height = max(3, 2 * n_params_to_plot)
    fig, axes = plt.subplots(
      n_params_to_plot, 
      n_systems, 
      figsize=(fig_width, fig_height), 
      dpi=300, 
      sharex='col',
    )
    
    if n_params_to_plot == 1 and n_systems == 1:
      axes = np.array([[axes]])
    elif n_params_to_plot == 1 or n_systems == 1:
      axes = axes.reshape(n_params_to_plot, n_systems)
    
    plt.subplots_adjust(
      wspace=0.2,
      hspace=0
    )
     
    x_ = np.zeros(())

    for i_row, param_to_plot in enumerate(params_to_plot):    
      for i_param, system_name in enumerate(sys.keys()):
        ax = axes[i_row, i_param]

        sys_true = pt2np(sys[system_name])[0,i_sample,0,param_to_plot]  # 
        sys_pred = pt2np(sys[system_name])[1:,i_sample,0,param_to_plot] # 
        x_steps = np.arange(1, len(sys_pred)+1) #

        ax.grid(which='major', color='black', linestyle='-', linewidth=0.5)

        if i_row != (n_params_to_plot - 1): 
          ax.label_outer()
        else: 
          ax.set_xlabel('Unfold')
          ax.tick_params(axis='x', labelrotation=45, labelsize=8)

        ax.hlines(sys_true, x_steps.min(), x_steps.max(), color=colors[0], linestyle='--', label='True')
        ax.plot(x_steps, sys_pred, color=colors[1], linestyle='-', marker='o', label='Predicted')
        if plt_res: 
          abs_res = np.abs(sys_pred - sys_true)
          ax.stem(x_steps, abs_res, label=f'Residuals{abs_res.sum():.1e}')

        ax.legend(loc='upper right', fontsize=8)
  
    if n_systems > 1:
      system_name = path.commonprefix(sys_keys).rstrip('_')

    plt.savefig(f'{save_path}/{system_name}_{epoch}_{dataset_idx}.png')
    plt.close(fig)

def create_gif_from_fitting(
  sample_idx:int, 
  system_name:str, 
  ds_name:str='train', 
  output_path:str='outputs/intermediate'
):
  # string template: system_name_epoch_sample_idx.png
  folder_path = f"{output_path}/{ds_name}/"
  
  # Find all PNG files matching the pattern
  pattern = path.join(folder_path, f"{system_name}_*_*.png")
  files = glob.glob(pattern)
  
  # Filter files that match the sample_idx
  matching_files = [f for f in files if f.endswith(f"_{sample_idx}.png")]
  
  # Extract epochs and sort by epoch
  def extract_epoch(filename):
    parts = path.basename(filename).split('_')
    if len(parts) >= 3:
      try:
        return int(parts[1])  # Assuming epoch is the second part
      except ValueError:
        return 0
    return 0
  
  matching_files.sort(key=extract_epoch)
  
  if not matching_files:
    print(f"No matching files found for sample {sample_idx} in {folder_path}")
    return
  
  # Create GIF
  images = [Image.open(f) for f in matching_files]
  gif_path = path.join(folder_path, f"{system_name}_{sample_idx}.gif")
  images[0].save(gif_path, save_all=True, append_images=images[1:], duration=500, loop=0)
  
  print(f"GIF created: {gif_path}")

def print_memory_usage(verbose:int=0):
  if verbose < 1:
    return
  
  list_strs = ""
  
  for device_id in range(torch.cuda.device_count()):
    device = torch.device(f'cuda:{device_id}')
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
    cached = torch.cuda.memory_reserved(device) / (1024 ** 3)
    list_strs += f"[GPU {device_id}] Allocated: {allocated:.2f} GB, Cached: {cached:.2f} GB\n"
  
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
  return list_strs

def print_available_gpus():
  if not torch.cuda.is_available():
    print("No GPU available.")
    return
  
  n_gpus = torch.cuda.device_count()
  print(f"Number of available GPUs: {n_gpus}")
  for i in range(n_gpus):
    gpu_name = torch.cuda.get_device_name(i)
    total_mem = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
    print(f"{i}: {gpu_name} [{torch.cuda._get_device(i)}], Total Memory: {total_mem:.2f} GB")

def plot_unfold_fits(ds_sys, i_sample:int, export_path:str) -> dict[str, NDArray]:
  was_interactive = plt.isinteractive()
  if was_interactive:
    plt.ioff()
  n_unfolds = list(ds_sys.values())[0].shape[0]-2
  n_sys_counts = len(ds_sys)
  sys_count = 0
  fig, ax = plt.subplots(
    n_unfolds, 
    n_sys_counts, 
    figsize=(10,5), 
    dpi=300,
    sharex=False,
    sharey=False,
  )
  
  if n_sys_counts == 1 and n_unfolds == 1:
    ax = [ax]
  
  if n_sys_counts == 1 and n_unfolds > 1:
    ax = ax.reshape(-1, 1)  # Convert to 2D: (n_unfolds, 1)
  
  data = {}
  
  for sys_name, sys_y in ds_sys.items():
    y_min = min(sys_y[2:,i_sample,:].min(), sys_y[1,i_sample,:].min())
    y_max = max(sys_y[2:,i_sample,:].max(), sys_y[1,i_sample,:].max())
    
    data_system = np.zeros((1 + 1 + n_unfolds, sys_y.shape[-1])) # x, y_true, y_pred1, y_pred2, ...
    data_system[0,:] = wn = sys_y[0,i_sample,:]      # x
    data_system[1,:] = y_true = sys_y[1,i_sample,:]      # y
    
    for i_unfold in range(0, n_unfolds):
      if n_unfolds == 1:
        ax_unfold = ax
      else:
        ax_unfold = ax[i_unfold]
      
      data_system[i_unfold+2,:] = y_pred = sys_y[i_unfold+2,i_sample,:]
      
      ax_unfold[sys_count].plot(wn, y_true, color='tab:blue', label='True')
      ax_unfold[sys_count].plot(wn, y_pred, color='tab:orange', linestyle='--', label=f'Unfold {i_unfold}')
      # ax_unfold[sys_count].legend()
      ax_unfold[sys_count].set(
        ylim=(y_min*1.1, y_max*1.1),
        xlim=(wn.min(), wn.max()),
      )
      ax_unfold[sys_count].grid(True, which='both', linestyle='--', linewidth=0.5)
      if i_unfold < n_unfolds - 1:
        ax_unfold[sys_count].set_xticks([])
    sys_count += 1
    
    data[f"{sys_name}_{i_sample:03d}"] = data_system
    
    if sys_count == sys_y.shape[1]: break

  plt.subplots_adjust(hspace=0, left=0.07, right=0.97, top=0.97, bottom=0.07)
  fig.tight_layout(pad=0.0) # 
  fig.savefig(export_path) # 
  plt.close(fig) # 
  if was_interactive:
    plt.ion()
  
  return data

def save_model_by_trail(
  model: torch.nn.Module,
  usecase: int,
  model_type: str,
  trail: int,
  model_id: int | None = None,
) -> str:
  """
  Save model to:
    models/usecase{usecase}/{model_type}-t{trail:02d}-{id:02d}.pt
  """
  
  models_dir = path.join(MODELS_DIR, f"usecase{usecase}")
  makedirs(models_dir, exist_ok=True)

  if model_id is None:
    model_id = _next_model_id_for_trail(models_dir, model_type, trail)

  filename = f"{model_type}-t{trail:02d}-{model_id:02d}.pt"
  save_path = path.join(models_dir, filename)
  torch.save(model, save_path)
  return save_path

def _next_model_id_for_trail(
  models_dir: str, 
  model_type: str, 
  trail: int
) -> int:
  """Return next available two-digit model id for a given model type and trail."""
  from re import compile, match, escape
  
  if not path.isdir(models_dir):
    return 0

  pattern = compile(rf"^{escape(model_type)}-t{trail:02d}-(?P<id>\d{{2}})\.pt$")
  used_ids = []
  for filename in listdir(models_dir):
    match = pattern.match(filename)
    if match:
      used_ids.append(int(match.group('id')))

  return (max(used_ids) + 1) if used_ids else 0

if __name__ == "__main__":
  print_available_gpus()