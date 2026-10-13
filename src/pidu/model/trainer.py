from collections import defaultdict
import os
import sys
import numpy as np
import pickle
import time
from typing import List, Tuple, Dict, DefaultDict
from torch.profiler import profile, ProfilerActivity, record_function
import seaborn as sns
from numpy.typing import NDArray
from itertools import chain
import pandas as pd

import matplotlib.pyplot as plt
import torch
import torch.fft as fft
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, SequentialLR, \
  CosineAnnealingWarmRestarts


from pidu.paths import ROOT
PROJECT_DIR = str(ROOT)


# from Data.DataImporter import DataImporter
from pidu.config import usecase, active_models
from pidu.data.utils import pt2np, np2pt, get_matches, savefig_debug
from pidu.data.simulation import load_sim_ds, p_gen_params, SimulationDatasetBatch, sys_pars
# from Analyse.Analyser import main as analyser_main

from pidu.model.optimisation import PCGrad, BaseOptimiser, LossFunction, NashMTL
from pidu.model.initializer import NetworkInitializer
from pidu.model.networks.framework import MultiModelFramework
from pidu.model.networks.io import load_pt_model, save_pt_model
from pidu.model.utils import create_masks_osc, create_masks_sys, create_mask_wn, \
  plot_intermediate_y, plot_intermediate_x, print_memory_usage, plot_unfold_fits


torch.set_printoptions(precision=2, sci_mode=False) # 


# MODELS
class History():
  def __init__(self):
    self.loss:Dict[str,NDArray] # Datasets, systems, batches
    self.loss_per_unfold:Dict[str,NDArray] # Datasets, systems, batches, unfolds
    
    self.gradients = defaultdict()
    self.gradient_norms = defaultdict()
    self.gradient_lines = defaultdict()
    
    self.batch_sizes = {}
    
    self.initialised = False

  def init_history(
    self, 
    run_folder:str, 
    epochs:int, 
    loaders:Dict[str, DataLoader], 
    n_systems:int, 
    n_unfolds:int,
    save_intermediate:bool=False,
    save_intermediate_name:str='loss',
    save_trail_id:int|None=None,
    info:Dict|None=None,
  ):
    # ds: [epochs, n_systems, n_batches, -]
    self.dataset_names = list(loaders.keys()) 
    self.system_names = list(loaders[self.dataset_names[0]].dataset.system_dicts.keys()) # 
    self.n_epochs = epochs      # 
    self.n_systems = n_systems  # 
    self.n_unfolds = n_unfolds  # 
    self.ds_n_batches = {key: len(value) for key, value in loaders.items()} # Number of batches per dataset
    self.batch_sizes.update({key: min(value.batch_size, value.dataset.n) for key, value in loaders.items()}) # Batch sizes per dataset
    self.loss = {key: np.zeros((epochs, n_systems, self.ds_n_batches[key], 2)) for key in loaders.keys()}
    self.loss_matrix = {key: np.zeros((epochs, 1, (1+self.n_systems)*4)) for key in loaders.keys()}
    self.save_intermediate = save_intermediate
    self.run_folder = run_folder
    
    self.loss_matrix_names = ['']

    self.initialised = True
    
    self.save_intermediate_name = save_intermediate_name
    if self.save_intermediate:
        os.makedirs(run_folder, exist_ok=True)
        if save_trail_id is None:
          list_files = [_i for _i in os.listdir(run_folder) if _i.startswith(f'{self.save_intermediate_name}')]
          # find max run number
          if len(list_files) == 0:
              self.current_loss_run = 0
          else:
              self.current_loss_run = max([int(_i.split('_')[-1].split('.')[0]) for _i in list_files]) + 1
        else:
          # If save_trail_id is provided, use it as the current loss run number (useful for resuming or specific runs)
          self.current_loss_run = save_trail_id
        
        print(f"Saving intermediate results to {run_folder}/{self.save_intermediate_name}_{self.current_loss_run:03d}.txt")
        with open(f'{run_folder}/{self.save_intermediate_name}_{self.current_loss_run:03d}.txt', 'w') as f:
            f.write(self.header_run_info(info))
            f.write("dataset,system,batch,epoch,loss,time\n")
        self.save_intermediate_folder_name = f"{self.save_intermediate_name}_{self.current_loss_run:03d}"
    
        return self.save_intermediate_folder_name
    
  def init_p_preds(self, loaders:Dict[str, DataLoader], **kwargs):
    self.wn_mask = kwargs.get('wn_mask', None)  # Wavenumber mask if available
    
    self.testset_names = loader_names = list(loaders.keys())
    self.system_names_tests = list(loaders[loader_names[0]].dataset.system_dicts.keys()) # 
    self.system_parameter_names = loaders[loader_names[0]].dataset.col_names_sys
    self.batch_sizes.update({key: min(value.batch_size, value.dataset.n) for key, value in loaders.items()}) # Batch sizes per dataset (if dataset is smaller than batch size, use dataset size)
    
    shapes_of_system_loaders = {loader: loaders[loader].dataset.get_system_shapes() for loader in loader_names} # 
    self.p_sys_pred_shape = {ds_name: shapes_of_system_loaders[ds_name]['sys'] for ds_name in loader_names} #  
    self.p_osc_pred_shape = {ds_name: shapes_of_system_loaders[ds_name]['osc'] for ds_name in loader_names} # 
    self.y_pred_shape = {ds_name: shapes_of_system_loaders[ds_name]['y'] for ds_name in loader_names}       # 
    
    self.y_pred = {}      # 
    self.p_sys_pred = {}  # 
    self.p_osc_pred = {}  # 

    for loader_name in loader_names:
      ds_name = loader_name
      self.y_pred[loader_name] = {sys_name: np.zeros([self.n_unfolds+2] + self.y_pred_shape.get(ds_name).get(sys_name)) for sys_name in list(loaders[loader_names[0]].dataset.data_y.keys())}
      self.p_sys_pred[loader_name] = {sys_name: np.zeros([self.n_unfolds+2] + list(self.p_sys_pred_shape.get(ds_name).get(sys_name))) for sys_name in self.system_names_tests}
      self.p_osc_pred[loader_name] = {sys_name: np.zeros([self.n_unfolds+2] + list(self.p_osc_pred_shape.get(ds_name).get(sys_name))) for sys_name in self.system_names_tests if self.p_osc_pred_shape.get(ds_name).get(sys_name) is not None} 
      # p: p_true, p_init, p_unfold_0, p_unfold_1, ...
  
  # -- Update functions --
  def batch_update(self, loss:float, loss_matrix:NDArray|dict|None, info:Dict[str, float|int]):
    system = info['system']   # 
    dataset = info['dataset'] # 
    batch = info['batch']     # 
    epoch = info['epoch']     # 
    time = info['time']       #

    self.loss[dataset][epoch, system, batch, 0] = loss
    self.loss[dataset][epoch, system, batch, 1] = epoch
    if len(loss_matrix) != 0:
      self.loss_matrix[dataset][epoch, 0] = list(loss_matrix.values())[0][-1].flatten() # From the first system, take the last unfold loss
    
    if self.save_intermediate:
      with open(f'{self.run_folder}/{self.save_intermediate_name}_{self.current_loss_run:03d}.txt', 'a') as f:
        f.write(f"{dataset},{system},{batch},{epoch},{loss},{time}\n")

  def test_update(self, return_dict:Dict, info:Dict[str, float|int]):
    dataset = info['dataset'] # 
    system = info['system']   # 
    batch = info['batch']     # 

    for sys_name, sys_p in return_dict['p_sys'].items():
      self.p_sys_pred[dataset][sys_name][0, self.batch_sizes[dataset]*batch:self.batch_sizes[dataset]*(batch+1)] = pt2np(return_dict['p_sys_true'][sys_name])
      self.p_sys_pred[dataset][sys_name][1, self.batch_sizes[dataset]*batch:self.batch_sizes[dataset]*(batch+1)] = pt2np(return_dict['p_sys_init'][sys_name])
      self.p_sys_pred[dataset][sys_name][2:, self.batch_sizes[dataset]*batch:self.batch_sizes[dataset]*(batch+1)] = pt2np(sys_p)

    for sys_name, osc_p in return_dict['p_osc'].items():
      if osc_p is None or osc_p[-1] is None: continue
      self.p_osc_pred[dataset][sys_name][0, self.batch_sizes[dataset]*batch:self.batch_sizes[dataset]*(batch+1)] = pt2np(return_dict['p_osc_true'][sys_name])
      self.p_osc_pred[dataset][sys_name][1, self.batch_sizes[dataset]*batch:self.batch_sizes[dataset]*(batch+1)] = pt2np(return_dict['p_osc_init'][sys_name])
      self.p_osc_pred[dataset][sys_name][2:, self.batch_sizes[dataset]*batch:self.batch_sizes[dataset]*(batch+1)] = pt2np(osc_p)
    
    for sys_name, sys_y in return_dict['y'].items():
      self.y_pred[dataset][sys_name][:,self.batch_sizes[dataset]*batch:self.batch_sizes[dataset]*(batch+1)] = pt2np(sys_y)

  def gradient_update(
    self, 
    epoch:int, 
    i_batch:int, 
    optim_name:str, 
    gradients:torch.Tensor, 
    norm:torch.Tensor,
    lines:List[int],
  ):
    if optim_name not in self.gradients.keys():
      n_items = gradients.shape[0]            # Number of items in the gradient
      n_batches = self.ds_n_batches['train']  # Number of batches in the training set
      
      self.gradients[optim_name] = np.zeros(
        (self.n_epochs, n_batches, n_items), 
        dtype=np.float32
      )
      self.gradient_norms[optim_name] = np.zeros(
        (self.n_epochs, n_batches), 
        dtype=np.float32
      )
      self.gradient_lines[optim_name] = lines

    self.gradients[optim_name][epoch, i_batch, :] = gradients #   
    self.gradient_norms[optim_name][epoch, i_batch] = norm #
    
    assert self.gradient_lines[optim_name] == lines, "Lines should be equal" 

  def info(
    self,
    print_info:bool=True,
    plot_info:bool=True,
    **kwargs
  ) -> None:
    plt_loss = kwargs.get('plt_loss', False) # 
    plt_grads = kwargs.get('plt_grads', False) # 
    output_dir = f"{'/'.join(self.run_folder.split('/')[:-1])}/loss/{self.save_intermediate_folder_name}/"
    
    if not plot_info:
      return
    
    os.makedirs(output_dir, exist_ok=True)
    
    if plt_loss:
      save_fig = kwargs.get('save_fig', False)  # 
      show_fig = kwargs.get('show_fig', True)   # 
      yscale = kwargs.get('yscale', 'linear') # linear|log
      
      loss_shapes = self.loss[list(self.loss.keys())[0]].shape
      n_epochs = loss_shapes[0]  # 
      n_systems = loss_shapes[1] # 
      cmap = plt.get_cmap('tab10')
      colors = {key: cmap(count) for count, key in enumerate(self.loss.keys())}

      fig, ax = plt.subplots(nrows=n_systems, ncols=1, figsize=(12, 6*n_systems), dpi=300)
      
      if isinstance(ax, plt.Axes):
        ax = [ax]

      for i_system in range(n_systems):
        data_system = []
        for ds_name, ds_value in self.loss.items():
          loss_system = ds_value[:, i_system,:,0].flatten() # include [.sum(-1)] if n_unfolds present
          x = np.linspace(0, n_epochs, len(loss_system))
          ax[i_system].plot(x, loss_system, label=ds_name, color=colors[ds_name])
          data_system.append(loss_system)
      
        ax[i_system].set(
          xlabel='Epoch',
          ylabel='Loss',
          title='Losses',
          yscale=yscale,
          ylim=(1e1, 1e6)
        )
        ax[i_system].legend()
        ax[i_system].grid(True, which='both', linewidth=0.5)
        
        try:
          with pd.ExcelWriter(f'{output_dir}loss_{self.system_names[i_system]}.xlsx') as writer:
            for i, ds_name in enumerate(self.loss.keys()):
              df = pd.DataFrame(data_system[i], columns=[f'Loss_{ds_name}'])
              df.to_excel(writer, sheet_name=ds_name, index=False)
        except:
          print(" - Could not save loss to xlsx")
        
        if save_fig:
          fig.savefig(f'{output_dir}/loss.png')

        if show_fig:
          plt.show()

      plt.close(fig)
    
    if plt_loss and (self.loss_matrix is not None and list(self.loss_matrix.values())[0].sum() != 0):
      os.makedirs(output_dir, exist_ok=True)
      save_fig = kwargs.get('save_fig', False)  # 
      show_fig = kwargs.get('show_fig', True)   # 
      yscale = kwargs.get('yscale', 'linear') # linear|log
      
      keys_ = list(self.loss_matrix.keys())
      
      loss_shapes = self.loss_matrix[keys_[0]].shape
      n_epochs = loss_shapes[0]  # 
      n_systems = loss_shapes[1] # 
      cmap = plt.get_cmap('tab10')
      colors = [cmap(count) for count in range(loss_shapes[-1])] # color for each different loss component

      fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6*n_systems), dpi=300)
      
      if isinstance(ax, plt.Axes):
        ax = [ax]

      data_system = []
      for ds_name, ds_value in self.loss_matrix.items():
        loss = ds_value[:, 0]
        x = np.linspace(0, n_epochs, len(loss))
        for i_ds_component in range(loss_shapes[-1]):
          ax[0].plot(x, loss[:, i_ds_component], color=colors[i_ds_component])
        data_system.append(loss)
    
      ax[0].set(
        xlabel='Epoch',
        ylabel='Loss matrix',
        title='Losses',
        yscale=yscale,
        ylim=(1e1, 1e6)
      )
      # ax[0].legend()
      ax[0].grid(True, which='both', linewidth=0.5)
      
      try:
        with pd.ExcelWriter(f'{output_dir}/loss_matrix.xlsx') as writer:
          for i, ds_name in enumerate(self.loss.keys()):
            df = pd.DataFrame(data_system[i])
            df.to_excel(writer, sheet_name=ds_name, index=False)
      except:
        print(" - Could not save loss to xlsx")
        
      if save_fig:
        fig.savefig(f'{output_dir}/loss_matrix.png')

      if show_fig:
        plt.show()

      plt.close(fig)
    
    if plt_grads:
      optim_names = self.gradients.keys()
      
      for optim_name in optim_names:
        from matplotlib.colors import LogNorm, Normalize
        
        grad = self.gradients[optim_name]
        grad_mean = grad.mean(axis=1).T # Average over batches
        grad_std = grad.std(axis=1).T   # Standard deviation over batches
        x = np.arange(self.n_epochs)
        grad_mean_abs = np.abs(grad_mean)+1e-10
        
        fig, ax = plt.subplots(figsize=(10,6), dpi=300)
        
        # Heatmap
        sns.heatmap(
          grad_mean_abs, 
          ax=ax, 
          norm=LogNorm(vmin=grad_mean_abs.min(), vmax=grad_mean_abs.max()),
          cmap='viridis', 
          cbar=True,
          cbar_kws={'label': 'Log |Gradient|'}
        )
        
        # vlines
        for line in self.gradient_lines[optim_name]:
          ax.axhline(y=line, color='tab:red', linestyle='--', alpha=0.5, linewidth=2)

        ax.set(
          title=f"Gradient Heatmap - {optim_name}", 
          xlabel='Epoch', 
          ylabel='Parameter',
          # yscale='log',
        ) 
        
        # Second axis for norm
        norm = self.gradient_norms[optim_name].mean(axis=1)
        norm_std = self.gradient_norms[optim_name].std(axis=1)
        
        ax2 = ax.twiny()
        ax2.set(
          title=f"Gradient Norm - {optim_name}", 
          xlabel='Epoch', 
          ylabel='Norm',
          ylim=((norm.min()-norm_std.max())*1.05, (norm.max()+norm_std.max())*1.05),
        )
        
        ax2.spines['right'].set_color('tab:blue')
        ax2.spines['right'].set_linewidth(2)
        ax2.spines['left'].set_visible(False)  # Hide left spine since we're using right

        ax2.plot(x, norm, color='tab:blue')
        ax2.fill_between(x, norm - norm_std, norm + norm_std, color='tab:blue', alpha=0.5)

        xticks = np.arange(0, self.n_epochs, 10)
        ax.set_xticks(xticks)   # 
        ax2.set_xticks(xticks)  # 

        if save_fig:
          fig.savefig(f'{output_dir}/gradients_{optim_name}.png')
        if show_fig:
          plt.show()
        
        plt.close(fig)

  def header_run_info(self, info:Dict[str, Dict[str, str|int]]) -> str:
    header = f"Run information:\n"
    for key, value in info.items():
      header += f"-{key}:\n"
      for sub_key, sub_value in value.items():
        header += f" -{key}_{sub_key}: {sub_value}\n"
      
    header += "\n"
    return header

  def export_test_result(
    self, 
    output_dir:str|None=None,
    output_name:str|None=None,
  ):
    if output_dir is None: 
        output_dir = os.path.join(PROJECT_DIR, 'outputs', 'tests', time.strftime("result_%Y%m%d-%H%M"))
    else:
        output_dir = f"{'/'.join(self.run_folder.split('/')[:-1])}/outputs"
    if output_name is not None:
        output_dir = f"{output_dir}/{output_name}"
    
    max_to_plot = {ds_name: min(16, self.y_pred[ds_name][list(self.y_pred[ds_name].keys())[0]].shape[1]) for ds_name in self.y_pred.keys()}
    
    os.makedirs(output_dir, exist_ok=True)
    print(f" - export path: {output_dir}")
    print(f" - Max samples to save and plot: {max_to_plot}.")
    
    n_sys_pars, n_osc_pars = 0, 0
    n_samples = 0
    data_to_export = {}
    
    # -- Save parameters [system] --
    results = {}
    col_names_sys = self.system_parameter_names
    for ds_name, ds_results in self.p_sys_pred.items():
      results[ds_name] = {}
      for sys_name, sys_p in ds_results.items():
        n_layers = sys_p.shape[-2]  # 
        n_samples = sys_p.shape[1]  # 
        results[ds_name][sys_name] = {}
        for i_sample in range(max_to_plot[ds_name]):
          results[ds_name][sys_name][f"{i_sample:03d}"] = sys_p[:, i_sample].reshape(sys_p.shape[0], -1) # true + init + unfolds
        
        data_to_export[f"{ds_name}_{sys_name}"] = {}
        data_to_export[f"{ds_name}_{sys_name}"]['true'] = sys_p[0, :].reshape(sys_p[0].shape[0], -1) # 1 x n_samples x n_params
        data_to_export[f"{ds_name}_{sys_name}"]['pred'] = sys_p[-1, :].reshape(sys_p[0].shape[0], -1) # 1 x n_samples x n_params
        data_to_export[f"{ds_name}_{sys_name}"]['res'] = np.abs(data_to_export[f"{ds_name}_{sys_name}"]['true'] - data_to_export[f"{ds_name}_{sys_name}"]['pred']) # 1 x n_samples x n_params
        
        # dictionary of tables to excel sheets
        # with pd.ExcelWriter(os.path.join(output_dir, f"{ds_name}_{sys_name}_sys.xlsx")) as writer:
        #   for sample_number, sample_data in results[ds_name][sys_name].items():
        #     df = pd.DataFrame(sample_data, columns=col_names[sys_name])
        #     df.to_excel(writer, sheet_name=sample_number, index=True)

      if n_layers > 1:
        col_names_sys = {k: [f"{i_layer}_{item}" for i_layer in range(2) for item in self.system_parameter_names['system1']] for k,v in self.system_parameter_names.items()}
      
      # with pd.ExcelWriter(os.path.join(full_path, f"{ds_name}_{sys_name}_sys.xlsx")) as writer:
      #   for sys_name, df in results.items():
      #     df.to_excel(writer, sheet_name=sys_name, index=False)
    
    def filter_mask_unfold(sample:NDArray, i_unfold:int, osc_max:int=5, i_sample:int|None=None) -> NDArray:
      if i_sample is None:
        mask_unfold = sample[i_unfold, :, :, ..., 0] > 0.75 # obj_score
      else:
        mask_unfold = sample[i_unfold, i_sample, :, ..., 0] > 0.75 # obj_score
      
      if mask_unfold.any():
        if i_sample is None:
          # Perform masking for all samples
          res = np.where(mask_unfold[...,None], sample[i_unfold], -1)[..., 1:]
          res_ = np.zeros((res.shape[0], 3*osc_max))
          for i in range(res.shape[0]):
            # inefficient but works
            rows = res[i].reshape(-1, res[i].shape[-1])
            res_[i] = rows[np.argsort(rows[:, 0])[::-1]][:osc_max].reshape(-1)
        else:
          # raise NotImplementedError("Filtering multiple samples at once is not implemented.")
          res = np.where(mask_unfold[...,None], sample[i_unfold, i_sample], -1)[..., 1:]
          rows = res.reshape(-1, res.shape[-1])
          res_ = rows[np.argsort(rows[:, 0])[::-1]][:osc_max].reshape(-1)  # sort oscillators by w0, keep (w0, wp, g) together
        return res_
      else:
        if i_sample is None:
          return np.full((sample.shape[1], 3*osc_max), -1)
        return np.full((3*osc_max,), -1)
    
    osc_max = 5
    osc_col_names = [f"{param}_{i:02d}" for i in range(osc_max) for param in ['w0', 'wp', 'g']]
    
    # -- Save parameters [oscillator] --
    for ds_name, ds_results in self.p_osc_pred.items():
      for sys_name, osc_p in ds_results.items():
        if osc_p.size == 1: continue
        osc_p_act = np.zeros((max_to_plot[ds_name], 2 + self.n_unfolds, len(osc_col_names))) * np.nan
        for i_sample in range(max_to_plot[ds_name]):
          # sample = sys_p[:, i_sample, 0, :] # [n_unfolds+2, n_regions, n_params]
          for i_unfold in range(2 + self.n_unfolds):
            filtered = filter_mask_unfold(osc_p, i_unfold, osc_max=osc_max, i_sample=i_sample)  # +2 to skip true and init
            if filtered.shape[0] == 0: continue
            osc_p_act[i_sample, i_unfold, :len(filtered)] = filtered  # Average over regions
          
          results[ds_name][sys_name][f"{i_sample:03d}"] = np.concatenate([results[ds_name][sys_name][f"{i_sample:03d}"], osc_p_act[i_sample]], axis=1)
        
        data_to_export[f"{ds_name}_{sys_name}"]['true'] = filter_mask_unfold(osc_p, 0, osc_max=5) # 1 x n_regions x n_params
        data_to_export[f"{ds_name}_{sys_name}"]['pred'] = filter_mask_unfold(osc_p, -1, osc_max=5) # 1 x n_regions x n_params
        data_to_export[f"{ds_name}_{sys_name}"]['res'] = np.abs(filter_mask_unfold(osc_p, 0, osc_max=5) - filter_mask_unfold(osc_p, -1, osc_max=5)) # 1 x n_regions x n_params
        # results[f"{sys_name}"] = pd.DataFrame(sys_p_act, columns=col_names)

        # with pd.ExcelWriter(os.path.join(full_path, f"{ds_name}_{sys_name}_osc.xlsx")) as writer:
        #   for sys_name, df in results.items():
        #     df.to_excel(writer, sheet_name=sys_name, index=False)

    # Information_total
    col_names1 = col_names_sys
    # Only add oscillator columns for systems that have oscillator data
    col_names2 = {}
    for key in col_names1.keys():
      if key in self.p_osc_pred.get(list(self.p_osc_pred.keys())[0], {}):
        col_names2[key] = osc_col_names
      else:
        col_names2[key] = []
    col_names_sys = {key: col_names1[key] + col_names2[key] for key in col_names1.keys()}
    
    for ds_name, ds_results in results.items():
      os.makedirs(os.path.join(output_dir, ds_name), exist_ok=True)
      for sys_name, sys_results in ds_results.items():
        try:
          with pd.ExcelWriter(os.path.join(output_dir, ds_name, f"{sys_name}.xlsx")) as writer:
            for sample_name, sample_data in sys_results.items():
              try:
                df = pd.DataFrame(sample_data, columns=col_names_sys[sys_name])
                df.to_excel(writer, sheet_name=sample_name, index=True)
              except ValueError as e:
                print(f" - Warning: Failed to create DataFrame for {ds_name}/{sys_name}/{sample_name}: {e}")
                print(f"   Data shape: {sample_data.shape}, Expected columns: {len(col_names_sys[sys_name])}")
                # Create a DataFrame without column names as fallback
                df = pd.DataFrame(sample_data)
                df.to_excel(writer, sheet_name=sample_name, index=True)
        except Exception as e:
          print(f" - Error exporting {ds_name}/{sys_name}: {e}")
    
    # -- Export residual distributions --
    print(f" - Exporting residual distributions...")
    for ds_name in self.p_sys_pred.keys():
      last_unfold_data = {}
      residual_data = {}
      residual_data_percentage = {}
      
      for sys_name in self.p_sys_pred[ds_name].keys():
        # Get full system parameters for all samples: [n_unfolds+2, n_samples, n_layers, n_params]
        sys_p = self.p_sys_pred[ds_name][sys_name]
        n_samples = sys_p.shape[1]
        
        # Extract true (index 0) and last unfold (index -1) for all samples
        # Shape after slicing: [n_samples, n_layers, n_params]
        true_vals = sys_p[0, ...].reshape(sys_p.shape[1], -1)     # 
        last_unfold = sys_p[-1, ...].reshape(sys_p.shape[1], -1)  # 
        
        # Calculate residuals for system parameters: [n_samples, n_params_sys]
        sys_last_unfold = last_unfold
        sys_residuals = np.abs(true_vals - last_unfold)
        sys_residuals_percentage = np.abs(true_vals - last_unfold) / (np.abs(true_vals) + 1e-8)
        
        # Add oscillator residuals if they exist
        if sys_name in self.p_osc_pred.get(ds_name, {}):
          osc_p = self.p_osc_pred[ds_name][sys_name]
          if osc_p.size > 1:
            # Process oscillator parameters similarly
            osc_last_unfold_list = []
            osc_residuals_list = []
            osc_residuals_percentage_list = []
            
            for i_sample in range(n_samples):
              sample = osc_p[:, i_sample, 0, :]  # [n_unfolds+2, n_regions, n_params]
              
              # Get last unfold oscillator data (filtered by confidence)
              def filter_mask_unfold_local(sample_data, i_unfold, osc_max=5):
                mask_unfold = sample_data[i_unfold, ..., 0] > 0.75
                if mask_unfold.any():
                  return sample_data[i_unfold, mask_unfold, 1:][:osc_max]
                else:
                  return np.zeros((0, sample_data.shape[-1]-1))
              
              true_osc = filter_mask_unfold_local(sample, 0, osc_max=5)
              last_osc = filter_mask_unfold_local(sample, -1, osc_max=5)
              
              # Calculate residuals and flatten
              osc_last = np.full(len(osc_col_names), np.nan)
              osc_residual = np.full(len(osc_col_names), np.nan)
              osc_residual_percentage = np.full(len(osc_col_names), np.nan)
              osc_last[:last_osc.size] = last_osc.flatten()
              if true_osc.shape[0] > 0 and last_osc.shape[0] > 0:
                # Match regions and calculate residuals
                min_regions = min(true_osc.shape[0], last_osc.shape[0])
                residual_vals = np.abs(true_osc[:min_regions] - last_osc[:min_regions]).flatten()
                residual_vals_percentage = (np.abs(true_osc[:min_regions] - last_osc[:min_regions]) / (np.abs(true_osc[:min_regions]) + 1e-8)).flatten()
                
                osc_residual[:len(residual_vals)] = residual_vals
                osc_residual_percentage[:len(residual_vals_percentage)] = residual_vals_percentage
              
              osc_last_unfold_list.append(osc_last)
              osc_residuals_list.append(osc_residual)
              osc_residuals_percentage_list.append(osc_residual_percentage)
            
            osc_last_unfolds = np.array(osc_last_unfold_list) # [n_samples, n_osc_params]
            osc_residuals = np.array(osc_residuals_list) # [n_samples, n_osc_params]
            osc_residuals_percentage = np.array(osc_residuals_percentage_list) # [n_samples, n_osc_params]
            
            # Concatenate system and oscillator residuals
            last_unfold_data[sys_name] = np.concatenate([sys_last_unfold, osc_last_unfolds], axis=1)
            residual_data[sys_name] = np.concatenate([sys_residuals, osc_residuals], axis=1)
            residual_data_percentage[sys_name] = np.concatenate([sys_residuals_percentage, osc_residuals_percentage], axis=1)
          
          else:
            last_unfold_data[sys_name] = sys_last_unfold
            residual_data[sys_name] = sys_residuals
            residual_data_percentage[sys_name] = sys_residuals_percentage
        
        else:
          residual_data[sys_name] = sys_residuals
          residual_data_percentage[sys_name] = sys_residuals_percentage
      
      # Export last unfold data to Excel with one sheet per system
      try:
        last_unfold_path = os.path.join(output_dir, ds_name, "last_unfold.xlsx")
        with pd.ExcelWriter(last_unfold_path) as writer:
          for sys_name, last_unfold in last_unfold_data.items():
            try:
              df_last_unfold = pd.DataFrame(
                last_unfold, 
                columns=col_names_sys[sys_name],
                index=[f"sample_{i:03d}" for i in range(last_unfold.shape[0])]
              )
              df_last_unfold.to_excel(writer, sheet_name=sys_name, index=True)
            except ValueError as e:
              print(f" - Warning: Failed to create last unfold DataFrame for {ds_name}/{sys_name}: {e}")
              print(f"   Last unfold shape: {last_unfold.shape}, Expected columns: {len(col_names_sys[sys_name])}")
              # Fallback without column names
              df_last_unfold = pd.DataFrame(last_unfold, index=[f"sample_{i:03d}" for i in range(last_unfold.shape[0])])
              df_last_unfold.to_excel(writer, sheet_name=sys_name, index=True)
        print(f" - Saved last unfold data: {last_unfold_path}")
      except Exception as e:
        print(f" - Error exporting last unfold data for {ds_name}: {e}")
      
      # Export to Excel with one sheet per system
      try:
        residual_path = os.path.join(output_dir, ds_name, "res_dis.xlsx")
        with pd.ExcelWriter(residual_path) as writer:
          for sys_name, residuals in residual_data.items():
            try:
              df_residuals = pd.DataFrame(
                residuals, 
                columns=col_names_sys[sys_name],
                index=[f"sample_{i:03d}" for i in range(len(residuals))]
              )
              df_residuals.to_excel(writer, sheet_name=sys_name, index=True)
            except ValueError as e:
              print(f" - Warning: Failed to create residual DataFrame for {ds_name}/{sys_name}: {e}")
              print(f"   Residual shape: {residuals.shape}, Expected columns: {len(col_names_sys[sys_name])}")
              # Fallback without column names
              df_residuals = pd.DataFrame(residuals, index=[f"sample_{i:03d}" for i in range(len(residuals))])
              df_residuals.to_excel(writer, sheet_name=sys_name, index=True)
        print(f" - Saved residual distribution: {residual_path}")
      except Exception as e:
        print(f" - Error exporting residual distribution for {ds_name}: {e}")
      
      # Export percentage residuals
      try:
        residual_path = os.path.join(output_dir, ds_name, "res_dis_percentage.xlsx")
        with pd.ExcelWriter(residual_path) as writer:
          for sys_name, residuals in residual_data_percentage.items():
            try:
              df_residuals = pd.DataFrame(
                residuals, 
                columns=col_names_sys[sys_name],
                index=[f"sample_{i:03d}" for i in range(len(residuals))]
              )
              df_residuals.to_excel(writer, sheet_name=sys_name, index=True)
            except ValueError as e:
              print(f" - Warning: Failed to create residual DataFrame for {ds_name}/{sys_name}: {e}")
              print(f"   Residual shape: {residuals.shape}, Expected columns: {len(col_names_sys[sys_name])}")
              # Fallback without column names
              df_residuals = pd.DataFrame(residuals, index=[f"sample_{i:03d}" for i in range(len(residuals))])
              df_residuals.to_excel(writer, sheet_name=sys_name, index=True)
        print(f" - Saved residual distribution: {residual_path}")
      except Exception as e:
        print(f" - Error exporting residual distribution for {ds_name}: {e}")

    # -- Plot unfolded fits and save data for plotting --
    print(f" - Plotting unfolded fits and saving data...")
    data = {}    
    for ds_name, ds_results in self.y_pred.items():
      output_path_ds = output_dir
      if 'exp' in ds_name.lower():
        max_to_plot[ds_name] = list(ds_results.values())[0].shape[1]
        output_path_ds = os.path.join(output_dir, 'png_unfolds_exp')
        os.makedirs(output_path_ds, exist_ok=True)
      for i_sample in range(max_to_plot[ds_name]):
        _path = os.path.join(output_path_ds, f"{ds_name}_{sys_name}_{i_sample:02d}.png")
        data.update(plot_unfold_fits(ds_results, i_sample, _path))
    
    # save data export to excel
    with pd.ExcelWriter(os.path.join(output_dir, f"{output_name}_ypreds.xlsx")) as writer:
      for data_name, table_data in data.items():
        try:
          df = pd.DataFrame(table_data)
          df.to_excel(writer, sheet_name=data_name, index=False)
        except ValueError as e:
          print(f" - Warning: Failed to create DataFrame for {data_name}: {e}")


class Trainer(nn.Module): 
  def __init__(
    self, 
    model:nn.Module|MultiModelFramework|str, 
    training_system:Dict[str, str|list|int|float]|None=None, 
    **kwargs
  ) -> None:
    super().__init__()
    self.model = self._create_model(model, training_system.get('init_mode', None))
    
    # Get the base model (unwrapped if DataParallel)
    base_model = self.base_model
    self.active_models = base_model.active_models    # 
    self.active_systems = base_model.active_systems  # 
    self.dtype, self.device = dt, dv = base_model.dt, base_model.dv

    self.verbose = training_system.get('verbose', 1)
    self.plot_inter_interval = kwargs.get('plot_inter_interval', 10)
    self.plot_intermediate = kwargs.get('plot_intermediate', False)

    self.loaders:Dict[str, DataLoader] = {}
    self.datasets:Dict[str, DataLoader] = {}
    self.datasets_return_mode:Dict[str, str] = {}

    # Intermediate results
    self.history = History()
    self.track_gradient = training_system.get('track_gradient', False)
    self.clip_grad_norm = training_system.get('clip_grad_norm', 1.0)
    self.run_folder = training_system.get('run_folder', 'outputs/')
    self.wgn_factor = training_system.get('wgn_factor', 0.0)
    self.to_shrink_and_perturb = training_system.get('shrink_and_perturb', None)
    self.shrink_factor = training_system.get('shrink_factor', 0.9)
    self.perturb_factor = training_system.get('perturb_factor', 0.1)

    self._init_training_system(training_system) #
    self._init_optimiser(training_system)       #
    self._init_scheduler(training_system)       #
    self._init_mixed_precision(training_system) # 
    
    n_regions_per_scale_per_system = {systemname: systeminfo['n_regions_per_scale'] for systemname, systeminfo in base_model.systems_info.items() if 'n_regions_per_scale' in systeminfo}
    self.loss_fn = LossFunction(self.active_systems, dt, dv, n_regions_per_scale_per_system=n_regions_per_scale_per_system, loss_matrix=kwargs.get('loss_matrix', False), loss_alpha=training_system.get('loss_alpha', 0.1))
    self.epoch = 0

  @property
  def base_model(self) -> MultiModelFramework:
    """Get the unwrapped model (handles DataParallel)"""
    return self.model.module if isinstance(self.model, nn.DataParallel) else self.model

  def _print(self, string:str, indent:int=0, level:int=0):
    if self.verbose >= level:
      print(f"{' '*indent}{string}")

  def _init_training_system(self, training_system:Dict[str, str|list|int|float]={}):
    self.batch_size = training_system.get('batch_size', 128)
    self.shuffle = training_system.get('shuffle', True)
    self.milestones = training_system.get('scheduler_milestones', [])

  def _init_mixed_precision(self, training_system:Dict[str, str|list|int|float]|None=None):
    training_system = training_system or {}
    self.use_mixed_precision = bool(training_system.get('mixed_precision', False))

    amp_dtype_name = str(training_system.get('amp_dtype', 'bfloat16')).lower()
    amp_dtype_map = {
      'float16': torch.float16,
      'fp16': torch.float16,
      'bfloat16': torch.bfloat16,
      'bf16': torch.bfloat16,
    }
    self.amp_dtype = amp_dtype_map.get(amp_dtype_name, torch.float16)

    self.amp_device_type = self.device.type if isinstance(self.device, torch.device) else str(self.device).split(':')[0]
    supported_amp_device = self.amp_device_type in {'cuda', 'cpu'}
    self.amp_enabled = self.use_mixed_precision and supported_amp_device

    if self.amp_enabled and self.amp_device_type == 'cpu' and self.amp_dtype == torch.float16:
      self.amp_dtype = torch.bfloat16
      self._print("Mixed precision on CPU uses bfloat16; switching amp_dtype from float16 to bfloat16.", level=1)

    use_scaler = self.amp_enabled and self.amp_device_type == 'cuda' and self.amp_dtype == torch.float16
    self.scaler = torch.amp.GradScaler(device='cuda', enabled=use_scaler)
    self._print(f"Mixed precision enabled: {self.amp_enabled} (device={self.amp_device_type}, dtype={self.amp_dtype}, scaler={self.scaler.is_enabled()})", level=1)
  
  # -- MODEL LOADING --
  def _create_model(self, model:nn.Module|MultiModelFramework|str, init_mode:str|None) -> MultiModelFramework:
    if isinstance(model, str):
      return load_pt_model(model)
    
    model:MultiModelFramework = model
    
    if init_mode is None: return model

    initializer = NetworkInitializer(model)
    model = initializer.initialize(
        mode=init_mode,
        activation_bb='tanh', # TODO: Implement pass of activation func
        activation_dh='tanh', # TODO: Implement pass of activation func
        scale=1.0,
        distribution='normal'
    )
    
    return model

  # -- OPTIMISER --
  def _init_optimiser(self, training_system:Dict[str,str|list|int|float], **kwargs):
    base_model = self.base_model
    n_unfolds = base_model.n_unfolds
    optimisers:Dict[str, BaseOptimiser] = {}
    # Get n_layers from dataset_info if available, otherwise from model attribute
    if hasattr(base_model, 'dataset_info') and 'n_layers' in base_model.dataset_info:
      n_layers = base_model.dataset_info['n_layers']
    elif hasattr(base_model, 'n_layers'):
      n_layers = base_model.n_layers
    else:
      # Fallback: try to get from systems_info (for backward compatibility)
      n_layers = base_model.systems_info[list(base_model.systems_info.keys())[0]].get('n_layers', 1)
    self.system_info = base_model.systems_info

    # Add support for different optimisers
    self.optimisation_mode = training_system.get('optimisation_mode', 'complete')    # 
    optimiser_type = training_system.get('optimiser', 'adam').lower()           # 
    self.lr_base = training_system.get('lr', 1e-3)                              # 
    optimiser_algo = training_system.get('optimiser_algo', 'base').lower()  # base|pcgrad

    params = self.model.parameters()
    
    # if self.optimisation_mode == 'complete':
    #   optimiser_names = ['main']
    #   trainable_params = [self.model.parameters()]  # Use self.model (wrapped) for parameters
    # elif self.optimisation_mode == 'unfold':
    #   optimiser_names = [f"unfold_{i}" for i in range(n_unfolds)]
    #   trainable_params = [base_model.get_unfold_parameters(i) for i in range(n_unfolds)]
    # elif self.optimisation_mode == 'scale':
    #   optimiser_names = [f"main"]
    #   trainable_params = [self.model.parameters()]
    # elif self.optimisation_mode == 'optics-type':
    #   optimiser_names = ['main']
    #   trainable_params = [self.model.parameters()]
    # elif self.optimisation_mode == 'system':
    #   optimiser_names = ['main']
    #   trainable_params = [self.model.parameters()]
    # elif self.optimisation_mode == 'none':
    #   optimiser_names = ['main']
    #   trainable_params = [self.model.parameters()]

    # for i_name, name in enumerate(optimiser_names):
    #   params = trainable_params[i_name]

    if optimiser_type == 'adam':
        base_optim = torch.optim.Adam(
            params, 
            lr=self.lr_base,
            betas=training_system.get('betas', (0.9, 0.999)),
            eps=training_system.get('eps', 1e-4),
            weight_decay=training_system.get('weight_decay', 0),
        )
    elif optimiser_type == 'adamw':
        base_optim = torch.optim.AdamW(
            params,
            lr=self.lr_base,
            betas=training_system.get('betas', (0.9, 0.999)),
            eps=training_system.get('eps', 1e-4),
            weight_decay=training_system.get('weight_decay', 1e-2),
        )
    elif optimiser_type == 'radam':
        base_optim = torch.optim.RAdam(
            params,
            lr=self.lr_base,
            betas=training_system.get('betas', (0.9, 0.999)),
            eps=training_system.get('eps', 1e-4),
            weight_decay=training_system.get('weight_decay', 1e-4),
            decoupled_weight_decay=True,
        )
    elif optimiser_type == 'sgd':
        base_optim = torch.optim.SGD(
            params,
            lr=self.lr_base,
            momentum=training_system.get('momentum', 0.9),
            weight_decay=training_system.get('weight_decay', 1e-4),
            nesterov=training_system.get('nesterov', True)
        )
    else:
        raise ValueError(f"optimiser {optimiser_type} not supported")
    
    if optimiser_algo == 'pcgrad':
      optimisers['pcgrad'] = PCGrad(base_optim, base_model.dt, base_model.dv, 'sum', self.optimisation_mode)
    elif optimiser_algo == 'base':
      optimisers['base'] = BaseOptimiser(base_optim, base_model.dt, base_model.dv, 'sum', self.optimisation_mode)
    elif optimiser_algo == 'nash':
      dict_opt_to_task = {
        'system': n_layers * 2 * n_unfolds * 4, # TODO: Implement dynamically
        'unfold': n_unfolds,
        'complete': 1,
        'scale': 1,
        'optics-type': 1,
        'none': 1,
        'scale-optics-type': 12,
      }
      optimisers['nash'] = NashMTL(base_optim, n_tasks=dict_opt_to_task[self.optimisation_mode], device=base_model.dv, max_norm=training_system.get('clip_grad_norm', 1.0), update_weights_every=10, optim_niter=5)
    else:
      optimisers['base'] = BaseOptimiser(base_optim, base_model.dt, base_model.dv, 'sum', self.optimisation_mode)
    
    self.weight_optimiser = optimisers

  def _init_scheduler(self, training_system:Dict[str,str|list|int|float]):
    schedulers = {}
    scheduler_type = training_system.get('scheduler_type', 'sequential')
    
    for key, value in self.weight_optimiser.items():
      if scheduler_type == 'StepLR':
        schedulers[key] = StepLR(
            value.optimiser, 
          step_size=training_system.get('scheduler_step_size1', 10), 
          gamma=training_system.get('scheduler_gamma1', 0.1)
      )
      if scheduler_type == 'sequential':
        schedulers[key] = SequentialLR(
            value.optimiser, 
            [
                StepLR(
                    value.optimiser, 
                    step_size=self.training_system['scheduler_step_size1'], 
                    gamma=self.training_system['scheduler_gamma1']
                ),
                StepLR(
                    value.optimiser, 
                    step_size=self.training_system['scheduler_step_size2'], 
                    gamma=self.training_system['scheduler_gamma2']
                )
            ], 
            milestones=self.training_system['scheduler_milestones']
          )
      if scheduler_type == 'plateau':
        schedulers[key] = ReduceLROnPlateau(
          value.optimiser, 
          patience=5, 
          factor=0.5, 
          min_lr=1e-4
        )
      if scheduler_type == 'CosineAnnealingWR':
        schedulers[key] = CosineAnnealingWarmRestarts(
          value.optimiser,
          T_0=training_system.get('T_0', 20),
          T_mult=training_system.get('T_mult', 1),
          eta_min=training_system.get('eta_min', 1e-3),
          
        )

    self.scheduler = schedulers
  
  # -- Tracker --
  def _init_history(self, epochs:int, info_header:Dict[str, str], **kwargs):
    save_intermediate = kwargs.get('history_save_intermediate', False)
    self.save_intermediate_name = self.history.init_history(
      run_folder=self.run_folder,
      epochs=epochs,
      loaders=self.loaders,
      n_systems=len(self.active_systems),
      n_unfolds=self.base_model.n_unfolds,
      save_intermediate=save_intermediate,
      save_intermediate_name=kwargs.get('outputname_history', 'loss'),
      save_trail_id=kwargs.get('save_trail_id', None),
      info=info_header
    )
  
  def _init_loss(self, epochs:int):
    self.loss_fn.n_epochs = epochs
  
  def _init_save_intermediate(self, n_samples_to_save:int=1):
    if n_samples_to_save == 0:
      self.save_intermediate_y = None
      return
    
    # -- Save intermediate --
    self.save_intermediate_y:Dict[str, Dict[str, torch.Tensor]] = defaultdict(lambda: defaultdict(dict))
    self.save_intermediate_p:Dict[str, Dict[str, torch.Tensor]] = defaultdict(lambda: defaultdict(dict))

    for ds_name, item in self.datasets.items():
      # (to_return_wn), (to_return_x_sys_init, to_return_x_sys_tr), (to_return_x_osc_init, to_return_x_osc_tr), to_return_y, idx
      sample = item[0]
      _, _, _, y, _ = sample
      for system_key, system_y in y.items():
        # [wn, y_true] + [y_unfolds]
        self.save_intermediate_y[ds_name][system_key] = np.zeros(
          (self.base_model.n_unfolds+1+1, n_samples_to_save, len(system_y)),
        )
        # self.save_intermediate_p['sys'][ds_name][system_key] = np.zeros(
        #   (self.model.n_unfolds+1, n_samples_to_save, self.datasets[ds_name].n_layers, 11), # TODO: Code this properly
        # )
        # self.save_intermediate_p['osc'][ds_name][system_key] = np.zeros(
        #   (self.model.n_unfolds+1, n_samples_to_save, self.datasets[ds_name].n_layers, 2, 32, 4), # TODO: Code this properly
        # )
    sizes_  = {k: {sk: v.shape for sk, v in self.save_intermediate_y[k].items()} for k in self.save_intermediate_y.keys()}
    print(f" - Initialized saving of intermediate results for {n_samples_to_save} samples per dataset.")
    print(f" - Will be in of size: {sizes_}")

  # -- DATASETS --
  def load_dataset(
    self, 
    to_import:Dict[str, str|SimulationDatasetBatch], 
    **kwargs
  ):
    
    batch_size = kwargs.get('batch_size', self.batch_size)
    shuffle = kwargs.get('shuffle', self.shuffle)
    shuffle_test = kwargs.get('shuffle_test', False)
    
    for key, value in to_import.items():
      if isinstance(value, SimulationDatasetBatch):
        ds = value
        ds.purge_ds(self.base_model.active_systems)
        loader = DataLoader(
          value, 
          batch_size=batch_size, 
          shuffle=shuffle_test if key in ['exp', 'test'] else shuffle,
          # num_workers=4,    # 
          # pin_memory=True,  # 
        )
        self.datasets[key] = value
        self.datasets_return_mode[key] = ds.y_return_mode
      # elif isinstance(value, DataImporter):
      #   loader = DataLoader(value, batch_size=batch_size, shuffle=False)
      #   self.datasets[key] = value
      elif isinstance(value, DataLoader):
        loader = value
      else:
        raise ValueError(f"Value of type {type(value)} is not supported.")

      self.loaders[key] = loader # DataLoader for each dataset
    
  # -- MODELS --
  def save(self, save_model:str|None=None, save_history:str|None=None, **kwargs):
    """
    Save the model and/or training history.
    Parameters:
    - save_model (str | None): Path to save the model. If None, the model is not saved.
    - save_history (str | None): Path to save the training history. If None, the history is not saved.
    - **kwargs: Additional keyword arguments to pass to the saving functions.
    Keyword Arguments:
    - save_mode (str): Mode to save the history. Options are 'direct' or 'lmza'. Default is 'direct'.
    The method saves the model and/or training history based on the provided paths and options.
    """
    if save_model is not None:
      save_pt_model(self.model, save_model)

    if save_history is not None and self.history is not None:
      save_path = kwargs.get('history_path', f"{PROJECT_DIR}/outputs/history_saves")
      
      os.makedirs(save_path, exist_ok=True)
      save_name = f"{save_history}.pkl"
      save_path = os.path.join(save_path, save_name)
      
      with open(save_path, 'wb') as f:
        pickle.dump(self.history, f)

  # -- TRAINING --
  def _shrink_and_perturb(self, model: nn.Module, shrink_factor: float = 0.9, perturb_std: float = 0.01):
    """
    Implements shrink and perturb regularization to help escape local minima.
    Shrinks weights by a factor and adds random perturbation.
    """
    # Get the actual model (unwrapped if DataParallel)
    actual_model = model.module if isinstance(model, nn.DataParallel) else model
    
    with torch.no_grad():
      for param in actual_model.parameters():
        if param.requires_grad:
          # Shrink weights
          param.mul_(shrink_factor)
          # Add perturbation
          param.add_(torch.randn_like(param) * perturb_std)
    
    for name, optim in self.weight_optimiser.items():
      optim.optimiser.param_groups[0]['lr'] = min(self.lr_base*0.75, optim.optimiser.param_groups[0]['lr']*10)  # Adjust learning rate as well
    
    print(f"Applied shrink and perturb: factor={shrink_factor}, std={perturb_std}")
            
  def train(self, epochs:int, **kwargs) -> History:
    """Main training loop with advanced techniques"""
    
    # Prepare datasets
    # train_loader = self.loaders.get('train')  # 
    # val_loader = self.loaders.get('val')      # 
    # inter_loader = self.loaders.get('exp')    # 
    loaders = self.loaders
    
    # -- init save_intermediate --
    save_epoch = kwargs.get('save_epoch', [-1])  # Save every n epochs
    save_idx = kwargs.get('save_idx', [])    # Save specific epochs
    save_model = kwargs.get('save_model', None)  # Save model every n epochs
    save_inter = {str(epoch): save_idx for epoch in save_epoch}
    self.plot_intermediate = kwargs.get('plot_intermediate', False)
    self.plot_dir = kwargs.get('plot_dir', f"{PROJECT_DIR}/outputs/intermediate")
    self._init_save_intermediate(len(save_idx))

    # Initialize LR scheduler
    _ = self._lr_update(0) # Initial learning rate
    
    # -- All the information to be saved as metadata for the history save --
    info_header = {}
    info_header['data'] = {}
    info_header['model'] = self.model.get_info()
    info_header['training'] = {
      'optimisation_mode': self.optimisation_mode,
      'batch_size': self.batch_size,
    }
    
    # Initialize tracker
    self._init_history(epochs, info_header, **kwargs)
    self._init_loss(epochs)
    
    # Initialize loss
    self.loss_fn.init_epochs(epochs, n_unfolds=self.model.n_unfolds)

    # Loop over epochs
    for epoch in range(epochs):
      self.epoch = epoch
      print(f"Epoch {self.epoch+1}/{epochs}:")
      timings = {}
      
      if str(epoch) in save_inter.keys():
        idx_to_save = save_inter[str(epoch)] # 
      else:
        idx_to_save = [-1] # 

      for loader_name, loader in loaders.items():
        if loader is None: continue
        
        losses_train, details = self.loader_update(
          loader, 
          mode=loader_name, 
          idx_to_save=idx_to_save,
        )
        timings[loader_name] = details['time_elapsed']
        self._print(f" - {loader_name}: {losses_train.mean():.4f} (t={timings[loader_name]:.1f}s)", indent=2)
        self._print(f" - Memory Usage: {print_memory_usage(verbose=0)}", indent=2, level=2)

      # Update learning rate with standard scheduler
      _ = self._lr_update(epoch) # lr
      
      # Apply shrink and perturb if configured
      if self.to_shrink_and_perturb and (epoch + 1) % self.to_shrink_and_perturb == 0:
        # Check if last epoch
        if epoch == epochs - 1:
          print("Last epoch reached, not applying shrink and perturb.")
        else:
          self._shrink_and_perturb(self.model, self.shrink_factor, self.perturb_factor)
    
    if save_model is not None:
      self.save(save_model=save_model)
    
    if self.loss_fn.weight_history is not None:
      df = pd.DataFrame(self.loss_fn.weight_history)
      df.to_excel(os.path.join(self.run_folder, 'loss_weights_history.xlsx'), index=False)
    
    return self.history

  def init_evaluation(self, epochs:int, **kwargs):
    """Set up history and loss as `train(epochs)` leaves them, without training, so `test`
    can export the predictions of a loaded model. Takes the history kwargs of `train`."""
    self.plot_intermediate = False
    self._init_save_intermediate(0)
    info_header = {
      'data': {},
      'model': self.model.get_info(),
      'training': {'optimisation_mode': self.optimisation_mode, 'batch_size': self.batch_size},
    }
    self._init_history(epochs, info_header, **kwargs)
    self._init_loss(epochs)
    self.loss_fn.init_epochs(epochs, n_unfolds=self.model.n_unfolds)
    self.epoch = epochs - 1

  def loader_update(
    self, 
    loader, 
    mode:str='train', 
    update_history:bool=True, 
    **kwargs):
    idx_to_save = kwargs.get('idx_to_save', [-1])     # Save intermediate results
    return_y = kwargs.get('return_y', False)          # Whether to return y values
    return_p = kwargs.get('return_p', False)          # Whether to return p values
    losses = np.zeros(len(loader), dtype=np.float32)  # 
    test_mode = 'test' in mode
    
    time_start = time.time()

    if mode == 'train':
      self.model.train()  # 
      optimise = True     # 
    else:
      self.model.eval() # 
      optimise = False  # 
    
    # if 'test' in mode:
    #   shapes_of_system_loaders = loader.dataset.get_system_shapes()
    #   self.p_sys_pred = {sys_name: np.zeros(shapes_of_system_loaders[sys_name]['sys']) for sys_name in shapes_of_system_loaders.keys()}
    #   self.p_osc_pred = {sys_name: np.zeros(shapes_of_system_loaders[sys_name]['osc']) for sys_name in shapes_of_system_loaders.keys()}
    #   test_mode = True
      
    with torch.set_grad_enabled(optimise):
      for i_batch, response_loader in enumerate(loader):
        batchtime_start = time.time()
        loss, return_dict_batch = self.batch_update(
          response_loader,
          optimise,
          return_y=return_y,
          return_p=return_p,
          idx_to_save=idx_to_save,
          ds_name=mode,
          i_batch=i_batch,
        )
        batchtime_end = time.time() 
        
        if update_history:
          self.history.batch_update(loss, return_dict_batch['loss_matrix'], {
            'dataset': mode,
            'system': 0,
            'epoch': self.epoch,
            'batch': i_batch,
            'time': batchtime_end - batchtime_start # Expressed in seconds
          })
        
        if test_mode:
          self.history.test_update(return_dict_batch, {
            'dataset': mode,
            'system': 0,
            'batch': i_batch
          })
        
        losses[i_batch] = loss
        self._print(f"Batch {i_batch+1}/{len(loader)}: Loss = {loss:.4f}", indent=2, level=2)

    time_elapsed = time.time() - time_start 
    
    if (not self.plot_intermediate or (idx_to_save == [-1] or idx_to_save == [])) or not self.model.use_physics or not mode in ['val', 'exp']:
      return losses, {'time_elapsed': time_elapsed, 'y': return_dict_batch.get('y', None)}

    # TODO: Implement
    info = {'ds':mode, 'epoch': self.epoch,
      # 'sys': pt2np(self.save_intermediate_p['sys']),
      # 'osc': pt2np(self.save_intermediate_p['osc']),
    }
    plot_intermediate_y(
      self.save_intermediate_y[mode], 
      idx_to_save, 
      info,
      {
        'plt_res':True,
        'plot_dir': self.plot_dir,
        'wn_mask': self.base_model.wn_mask if hasattr(self.base_model, 'wn_mask') else None,
      }
    )
    # plot_intermediate_x(
    #   sys=pt2np(self.save_intermediate_p['sys'][mode]),
    #   osc=pt2np(self.save_intermediate_p['osc'][mode]),
    #   params_to_plot=[5,6,7], # w0_before, wp_before, g_before
    #   idx=idx_to_save,
    #   info=info,
    #   plot_kwgs={'plt_res':True},
    # )
    
    return_dict = {}
    return_dict['time_elapsed'] = time_elapsed
    return_dict['y'] = return_dict.get('y', None)

    return losses, return_dict
    
  def batch_update(
    self,
    response_loader:Tuple[torch.Tensor],
    optimise:bool=True,
    **kwargs
  ) -> None:
    """Process a single batch with mixed precision support"""
    epoch = kwargs.get('epoch', self.epoch)       # Current epoch
    ds_name = kwargs.get('ds_name', '')           # Dataset name
    i_batch = kwargs.get('i_batch', 0)            # Current batch index
    return_y = kwargs.get('return_y', False)      # Whether to return y values
    return_p = kwargs.get('return_p', False)      # Whether to return p values
    idx_to_save = kwargs.get('idx_to_save', [-1]) # Indices to save
    i_unfold = kwargs.get('i_unfold', 0)          # Current unfold index, def:0

    model = self.model  # Use wrapped model for forward pass
    base_model = self.base_model  # Use unwrapped for attribute access
    loss_fn = self.loss_fn
    
    def wgn(x:Tuple[torch.Tensor]|torch.Tensor, wgn_factor:float):
      if isinstance(x, tuple):
        return tuple(wgn(xi, wgn_factor) for xi in x)
      return x + wgn_factor * torch.randn_like(x)  # Add noise to the input

    # 1. Unpack data
    wn, x_sys, x_osc, y, idx = response_loader
    wn = {k:v[0] for k, v in wn.items()}  # Select only the first x-values for each system
    x_sys_true = x_sys[1]
    x_osc_true = x_osc[1]

    if self.datasets_return_mode[ds_name] == 'both':
      y = {k: v[..., 1] for k, v in y.items()}  # Select noised y values
    
    # 2. Forward pass
    if self.wgn_factor > 0:
      y_noised = {}
      for key, value in y.items():
        y_noised[key] = wgn(value, self.wgn_factor)
    else:
      y_noised = y
    # target_out, (p_outs, y_outs, outs), history = model(wn, x_sys, x_osc, y_td, y_fd)

    with torch.amp.autocast(
      device_type=self.amp_device_type,
      dtype=self.amp_dtype,
      enabled=self.amp_enabled,
    ):
      response_model = model(wn, y=y_noised, x_sys=x_sys, x_osc=x_osc)

    # 3. Calculate loss outside autocast for better numerical stability
    loss, loss_matrix, _ = loss_fn.forward(
      self.epoch,
      response_model['p_sys'],
      response_model['p_osc'],
      x_sys_true,
      x_osc_true,
      reduce=self.optimisation_mode, # complete|scale|unfold|optics-type|system|none
    )

    loss_item = loss.item() if (isinstance(loss, torch.Tensor) and loss.ndim == 0) else (loss.sum().item() if isinstance(loss, torch.Tensor) else sum([l.item() for l in loss]))
    # 4. Backward pass
    if optimise:
      count = 0
      skip_optimisers = set()
      
      for name, optim in self.weight_optimiser.items():
        optim_params = optim.optimiser.param_groups[0]['params']

        optim.backward(
          loss,
          count=count,
          shared_parameters=model.parameters(),
          scaler=self.scaler if self.scaler.is_enabled() else None,
        )  # Backpropagate the loss

        if self.scaler.is_enabled():
          self.scaler.unscale_(optim.optimiser)

        has_non_finite_grad = any(
          p.grad is not None and not torch.isfinite(p.grad).all()
          for p in optim_params
        )

        if has_non_finite_grad:
          skip_optimisers.add(name)
          norm_of_optim = torch.tensor(float('nan'), device=self.device)
          self._print(f"Optimise {name}: non-finite gradients detected; skipping step.", indent=3, level=1)
        else:
          norm_of_optim = torch.nn.utils.clip_grad_norm_(optim_params, max_norm=self.clip_grad_norm)
          if not torch.isfinite(norm_of_optim):
            skip_optimisers.add(name)
            self._print(f"Optimise {name}: non-finite gradient norm detected; skipping step.", indent=3, level=1)
        
        if self.track_gradient:
          grads, lines = base_model.get_gradient()
          self.history.gradient_update(self.epoch, i_batch, name, grads, norm_of_optim, lines)
        
        if torch.isfinite(norm_of_optim):
          self._print(f"Optimise {name}: Norm = {norm_of_optim:.2f} (before clipping)", indent=3, level=2)

      for name, optim in self.weight_optimiser.items():
        if name in skip_optimisers:
          optim.zero_grad(set_to_none=True)
          continue
        if self.scaler.is_enabled():
          self.scaler.step(optim.optimiser)
        else:
          optim.step()
        optim.zero_grad(set_to_none=True)
        count+=1

      if self.scaler.is_enabled():
        self.scaler.update()

    loss_value = loss_item
    del loss
    
    response_model_np = {k: pt2np(v) for k, v in response_model.items()}

    # 5. Save intermediate results if required
    mask_to_save = torch.isin(idx, torch.tensor(idx_to_save, device=idx.device))
    if mask_to_save.any():    
      wn_np = {k: pt2np(v) for k, v in wn.items()}
      y_np = {k: pt2np(v) for k, v in y.items()}
      x_sys_true_np = {k: pt2np(v) for k, v in x_sys_true.items()}
      x_osc_true_np = {k: pt2np(v) for k, v in x_osc_true.items()} if x_osc_true is not None else None
      # wn_mask = self.model.wn_mask if hasattr(self.model, 'wn_mask') else None
      self._save_intermediate(
        ds_name, 
        idx, 
        mask_to_save, 
        idx_to_save, 
        response_model_np, 
        wn_np, 
        y_np, 
        x_sys_true_np, 
        x_osc_true_np,
      )
    
    return_dict = defaultdict(dict)
    if return_y: 
      x = wn
      y_true = y_noised
      y_pred = response_model_np['y']
      return_dict['y'] = {sys_name: torch.cat(
        [
          base_model._select_dict_from_subset(x, sys_name)[None,None,:].repeat(1,y_true[sys_name].shape[0],1), 
          torch.as_tensor(y_true[sys_name]).unsqueeze(0),
          torch.as_tensor(y_pred[sys_name])
        ], dim=0) for sys_name in y_true.keys()
      }
    if return_p: 
      return_dict['p_sys'] = response_model_np['p_sys']
      return_dict['p_osc'] = response_model_np['p_osc']
      return_dict['p_sys_init'] = {k: pt2np(v) for k, v in x_sys[0].items()} # x_sys = (init, true)
      return_dict['p_osc_init'] = {k: pt2np(v) for k, v in x_osc[0].items()} # x_osc = (init, true)
      return_dict['p_sys_true'] = {k: pt2np(v) for k, v in x_sys_true.items()}
      return_dict['p_osc_true'] = {k: pt2np(v) for k, v in x_osc_true.items()} if x_osc_true is not None else None    
    if loss_matrix is not None:
      return_dict['loss_matrix'] = pt2np(loss_matrix)
    
    return loss_value, return_dict

  # Testing
  def test(self, loaders:List[str|DataLoader], **kwargs):
    to_export = kwargs.get('to_export', False)    # Whether to export results
    to_plot = kwargs.get('to_plot', False)        # Whether to plot results
    idx_to_save_mode = kwargs.get('idx_to_save', None) # Indices to save
    
    loaders_ = {}
    idx_to_save = {}
    
    for loader in loaders:
      if isinstance(loader, str) and loader in self.loaders:
        loaders_[f"test_{loader}"] = self.loaders[loader]
        idx_to_save[f"test_{loader}"] = idx_to_save_mode if idx_to_save_mode is not None else range(len(self.datasets[loader]))
        self.datasets_return_mode[f"test_{loader}"] = self.datasets_return_mode[loader]
      # else:
      #   raise ValueError(f"Loader {loader} not recognized.")
      elif isinstance(loader, DataLoader):
        loaders_[f"test_{loader}"] = loader
        idx_to_save[f"test_{loader}"] = idx_to_save_mode if idx_to_save_mode is not None else range(len(loader.dataset))
        
    self.history.init_p_preds(loaders_, **{
      'wn_mask': self.base_model.wn_mask if hasattr(self.base_model, 'wn_mask') else None
    })

    print("Testing:")
    for loader_name, loader in loaders_.items():
      if loader is None: continue
      
      losses_train, details = self.loader_update(
        loader, 
        mode=loader_name, 
        update_history=False,
        return_p=True,
        return_y=True,
        # idx_to_save=idx_to_save[loader_name],
      )
      self._print(f" - {loader_name}: {losses_train.mean():.4f} (t={details['time_elapsed']:.1f}s)", indent=1)
    
    if to_export:
      self.history.export_test_result(self.run_folder, self.save_intermediate_name)
      # idx_to_save = range(len(self.datasets[loader_name[5:]]))
      # plot_intermediate_y(details['y'], idx_to_save, {'ds':loader_name, 'epoch':self.epoch}, {'plt_res':True})
  
  def _lr_update(self, epoch:int) -> float:
    last_lr = new_lr = None
    for name, scheduler in self.scheduler.items():
      if epoch == 0:
        return self.weight_optimiser[name].optimiser.param_groups[0]['lr']

      last_lr = self.weight_optimiser[name].optimiser.param_groups[0]['lr']
      scheduler.step()
      new_lr = self.weight_optimiser[name].optimiser.param_groups[0]['lr']

      if epoch in self.milestones and isinstance(scheduler, SequentialLR):
        print(f"\nMilestone reached, lr: {last_lr:.3e}, {new_lr:.3e}\n")
        for param_group in scheduler.optimiser.param_groups:
          param_group['lr'] = last_lr
          new_lr = last_lr
    
    return (last_lr, new_lr)
  
  def _save_intermediate(
    self, 
    ds_name, 
    batch_idx, 
    mask_to_save,
    idx_to_save,
    response_model,
    wn,
    y_true,
    x_sys_true,
    x_osc_true,
  ): 
    _select = self.base_model._select_dict_from_subset
    
    matched = torch.isin(torch.tensor(idx_to_save),batch_idx[mask_to_save])
    for system_name, system_y in response_model['y'].items():
      self.save_intermediate_y[ds_name][system_name][0, matched] = _select(wn, system_name)
      self.save_intermediate_y[ds_name][system_name][1, matched] = pt2np(y_true[system_name][mask_to_save])
      self.save_intermediate_y[ds_name][system_name][2:, matched] = pt2np(system_y[:,mask_to_save,:])
      
      # self.save_intermediate_p['sys'][ds_name][system_name][0 ,matched] = _select(x_sys_true, system_name)[mask_to_save]
      # self.save_intermediate_p['sys'][ds_name][system_name][1:,matched] = pt2np(_select(response_model['p_sys'], system_name)[:,mask_to_save])

      # if _select(x_osc_true, system_name) is not None:
        # n_regions = _select(x_osc_true, system_name).shape[-2]
        # self.save_intermediate_p['osc'][ds_name][system_name][0, matched] = _select(x_osc_true, system_name)[mask_to_save]
        # self.save_intermediate_p['osc'][ds_name][system_name][1:, matched] = pt2np(_select(response_model['p_osc'], system_name)[:,mask_to_save])

