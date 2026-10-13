from collections import defaultdict
from json import load
from math import exp
import os
import sys
import numpy as np
import subprocess
import pickle
import time
from typing import List, Tuple, Dict, DefaultDict, Any
import seaborn as sns
from numpy.typing import NDArray
from itertools import chain
import pandas as pd
from scipy.optimize import least_squares

import matplotlib.pyplot as plt
import test
import torch
import torch.fft as fft
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, SequentialLR

from pidu.paths import ROOT
PROJECT_DIR = str(ROOT)

# from Data.DataImporter import DataImporter
from pidu.config import usecase, active_models
from pidu.data.simulation import load_sim_ds, init_usecase, create_dataset
from pidu.data.simulation import p_gen_params, sys_pars

from pidu.model.networks.framework import MultiModelFramework
from pidu.model.networks.io import load_pt_model
from pidu.model.utils import create_masks_osc, create_masks_sys, print_memory_usage

from pidu.baselines.lma import benchmark_lma

def get_dataset(
  usecase: int,
  trainset: str|None=None,
  valset: str|None=None,
  testset: str|None=None,
  expset: str|None|bool=None,
):
  n_train, n_val, n_test = 12, 8, 10
  p_gen_params, sys_pars = init_usecase(usecase=usecase) # Initialize usecase
  structured_noise = [False, False] # TODO: adjust per usecase
  
  if usecase == 2:
    structured_noise = [False, True]
  
  if isinstance(trainset, str):
    trainset = load_sim_ds(trainset, y_return_mode='noised')
  else:
    trainset = create_dataset(
      usecase=usecase,
      setname=f"case{usecase}_train_n2_{n_train}",
      n_samples=2**n_train,
      system_params=sys_pars,
      p_gen_params=p_gen_params,
      verbose=True,
      export=True,
      structured_noise=structured_noise,
    )
  
  if isinstance(valset, str):
    valset = load_sim_ds(valset, y_return_mode='noised')
  else:
    valset = create_dataset(
      usecase=usecase,
      setname=f"case{usecase}_val_n2_{n_val}",
      n_samples=2**n_val,
      system_params=sys_pars,
      p_gen_params=p_gen_params,
      verbose=True,
      export=True,
      structured_noise=structured_noise,
    )
  
  if isinstance(testset, str):
    testset = load_sim_ds(testset)
  else:
    testset = create_dataset(
      usecase=usecase,
      setname=f"case{usecase}_test_n2_{n_test}",
      n_samples=2**n_test,
      system_params=sys_pars,
      p_gen_params=p_gen_params,
      verbose=True,
      export=True,
      structured_noise=structured_noise,
    )
  
  if isinstance(expset, str):
    expset = load_sim_ds(expset)
  elif expset is False:
    expset = None
  else:
    expset = create_dataset(
      usecase=usecase,
      setname=f"case{usecase}_experimental",
      n_samples=2**n_test,
      system_params=sys_pars,
      p_gen_params=p_gen_params,
      verbose=True,
      export=True,
    )
  
  device, dtype = trainset.get_device_dtype()
  dataset_info = trainset.get_info_for_model(True)

  to_return = {k: v for k, v in {'train': trainset, 'val': valset}.items() if v is not None}

  return to_return, dataset_info, device, dtype

def get_model(model_name:str|None=None, ds_info=None, device=None, dtype=None, **kwargs):
    model_info = {
      'physics': kwargs.get('physics', True),
      'backbone_act_func': 'tanh',
      'backbone_norm': 'instancenorm', # batchnorm|layernorm|instancenorm|groupnorm|none
      'dropout_rate': 0.2,
      'active_models': kwargs.get('active_models', ['system1']), # 
      'mask_x_sys': create_masks_sys(
        n_layers=dataset_info['n_layers'], 
        n_sys_pars=dataset_info['n_system_parameters'],
        dt=dtype,
        dv=device
      ),
      'mask_x_osc': create_masks_osc(
        n_layers=dataset_info['n_layers'], 
        n_scales=dataset_info['n_scales'], 
        n_regions_max=dataset_info['n_regions_per_scale_max'],
        n_attribs_label=dataset_info['n_attribs_label'],
        std_val={'system1': 1, 'system2': 1},  # Standard value for the mask
        dt=dtype,
        dv=device
      ),
      'create_wn_mask': False,
      'use_residual': True,
      'conf_threshold': 0.75,
    }
  
    system_info = {
      'p_gen': p_gen_params,
      'dataset_info': ds_info,
      'systems': sys_pars
    }

    if model_name is not None:
      framework = load_pt_model(model_name)
      return framework

    framework = MultiModelFramework(
      n_unfolds=3,
      system_info=system_info,
      model_info=model_info,
      dtype=dtype,
      device=device,
      verbose=0,
    )
    framework.summary()

    return framework

def benchmark_inference(
  model: MultiModelFramework,
  dataset: torch.utils.data.Dataset,
  batch_size: int = 1,
  num_warmup: int = 10,
  num_iterations: int = 100,
  device: torch.device = None,
  verbose: bool = True,
):
  """
  Benchmark model inference performance.
  
  Args:
    model: The model to benchmark
    dataset: Dataset to use for inference
    batch_size: Batch size for inference
    num_warmup: Number of warmup iterations (not counted in timing)
    num_iterations: Number of iterations to time
    device: Device to run on
    verbose: Whether to print results
  
  Returns:
    Dict containing timing statistics and FLOPS estimates
  """
  from torch.utils.data import DataLoader
  from torch.profiler import profile, ProfilerActivity
  import torch.utils.benchmark as benchmark
  
  model.eval()
  loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
  
  # Get a single batch for profiling
  wn, x_sys, x_osc, y, idx = next(iter(loader))
  
  if device is None:
    device = next(model.parameters()).device
  
  results = {
    'batch_size': batch_size,
    'num_iterations': num_iterations,
    'device': str(device),
  }
  
  # === 1. Warmup ===
  if verbose:
    print(f"\n{'='*60}")
    print(f"Warming up with {num_warmup} iterations...")
  
  with torch.no_grad():
    for i in range(num_warmup):
      _ = model(wn=wn, y=y, x_sys=x_sys, x_osc=x_osc)
  
  if device.type == 'cuda':
    torch.cuda.synchronize()
  
  # === 2. Wall-clock time measurement ===
  if verbose:
    print(f"Running {num_iterations} timed iterations...")
  
  times = []
  with torch.no_grad():
    for i in range(num_iterations):
      if device.type == 'cuda':
        torch.cuda.synchronize()
      
      start_time = time.perf_counter()
      output = model(wn=wn, y=y, x_sys=x_sys, x_osc=x_osc)
      
      if device.type == 'cuda':
        torch.cuda.synchronize()
      
      end_time = time.perf_counter()
      times.append(end_time - start_time)
  
  times = np.array(times)
  results['wall_time_mean'] = np.mean(times)
  results['wall_time_std'] = np.std(times)
  results['wall_time_median'] = np.median(times)
  results['wall_time_min'] = np.min(times)
  results['wall_time_max'] = np.max(times)
  results['throughput_samples_per_sec'] = batch_size / results['wall_time_mean']
  results['time_per_sample_ms'] = (results['wall_time_mean'] / batch_size) * 1000
  
  # === 3. FLOPS estimation using profiler ===
  if verbose:
    print(f"Profiling FLOPS...")
  
  try:
    with torch.no_grad():
      with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if device.type == 'cuda' else [ProfilerActivity.CPU],
                   record_shapes=True,
                   with_flops=True) as prof:
        output = model(wn=wn, y=y, x_sys=x_sys, x_osc=x_osc)
    
    # Extract FLOPS from profiler
    total_flops = 0
    for event in prof.key_averages():
      if event.flops is not None and event.flops > 0:
        total_flops += event.flops
    
    results['flops_total'] = total_flops
    results['flops_per_sample'] = total_flops / batch_size
    results['gflops_total'] = total_flops / 1e9
    results['gflops_per_sample'] = results['flops_per_sample'] / 1e9
    
    # FLOPS/s
    results['flops_per_second'] = total_flops / results['wall_time_mean']
    results['gflops_per_second'] = results['flops_per_second'] / 1e9
    
  except Exception as e:
    if verbose:
      print(f"Warning: FLOPS profiling failed: {e}")
    results['flops_total'] = None
    results['gflops_total'] = None
  
  # === 4. Memory usage ===
  if device.type == 'cuda':
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
      output = model(wn=wn, y=y, x_sys=x_sys, x_osc=x_osc)
    
    results['memory_allocated_mb'] = torch.cuda.memory_allocated() / (1024**2)
    results['memory_reserved_mb'] = torch.cuda.memory_reserved() / (1024**2)
    results['memory_peak_mb'] = torch.cuda.max_memory_allocated() / (1024**2)
  
  # === 5. Model parameters ===
  total_params = sum(p.numel() for p in model.parameters())
  trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
  
  results['total_parameters'] = total_params
  results['trainable_parameters'] = trainable_params
  results['model_size_mb'] = (total_params * 4) / (1024**2)  # Assuming float32
  
  # Add function evaluations (n_unfolds for neural network)
  results['n_function_evaluations'] = getattr(model, 'n_unfolds', 1)
  
  # === Print results ===
  if verbose:
    print(f"\n{'='*60}")
    print(f"BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"\nModel Information:")
    print(f"  Total parameters:      {results['total_parameters']:,}")
    print(f"  Trainable parameters:  {results['trainable_parameters']:,}")
    print(f"  Model size:            {results['model_size_mb']:.2f} MB")
    
    print(f"\nTiming Statistics (batch_size={batch_size}):")
    print(f"  Mean time:             {results['wall_time_mean']*1000:.3f} ms")
    print(f"  Std time:              {results['wall_time_std']*1000:.3f} ms")
    print(f"  Median time:           {results['wall_time_median']*1000:.3f} ms")
    print(f"  Min time:              {results['wall_time_min']*1000:.3f} ms")
    print(f"  Max time:              {results['wall_time_max']*1000:.3f} ms")
    print(f"  Time per sample:       {results['time_per_sample_ms']:.3f} ms")
    print(f"  Throughput:            {results['throughput_samples_per_sec']:.2f} samples/sec")
    
    if results.get('flops_total'):
      print(f"\nFLOPS Statistics:")
      print(f"  Total FLOPS:           {results['flops_total']:,}")
      print(f"  GFLOPS (total):        {results['gflops_total']:.3f}")
      print(f"  GFLOPS per sample:     {results['gflops_per_sample']:.3f}")
      print(f"  GFLOPS/s:              {results['gflops_per_second']:.3f}")
    
    if device.type == 'cuda':
      print(f"\nMemory Usage:")
      print(f"  Allocated:             {results['memory_allocated_mb']:.2f} MB")
      print(f"  Reserved:              {results['memory_reserved_mb']:.2f} MB")
      print(f"  Peak:                  {results['memory_peak_mb']:.2f} MB")
    
    print(f"{'='*60}\n")
  
  return results

def perform_inference(
  model: MultiModelFramework,
  dataset: Dict[str, any],
):
  return 

def compare_methods(
  results_dict: dict[str, dict], 
  export_overleaf_table:bool=False,
  save_path: str = None
) -> pd.DataFrame:
  """
  Compare multiple methods/models based on benchmark results.
  
  Args:
    results_dict: Dictionary mapping method names to their benchmark results
    save_path: Optional path to save comparison plot
  """
  methods = list(results_dict.keys())
  
  # Create comparison table
  df_data = []
  for method_name, results in results_dict.items():
    row = {
      'Method': method_name,
      'Time/Sample (ms)': results.get('time_per_sample_ms', np.nan),
      'Throughput (samples/s)': results.get('throughput_samples_per_sec', np.nan),
      'N_func_evals': results.get('n_function_evaluations', np.nan),
      'GFLOPS': results.get('gflops_total', np.nan),
      'GFLOPS/sample': results.get('gflops_per_sample', np.nan),
      'Parameters (M)': results.get('total_parameters', 0) / 1e6,
      'Model Size (MB)': results.get('model_size_mb', np.nan),
    }
    
    if 'memory_peak_mb' in results:
      row['Peak Memory (MB)'] = results['memory_peak_mb']
    
    df_data.append(row)
  
  df = pd.DataFrame(df_data)
  
  print("\n" + "="*80)
  print("METHOD COMPARISON")
  print("="*80)
  print(df.to_string(index=False))
  print("="*80 + "\n")
  
  if export_overleaf_table:
    export_to_overleaf_table(usecase, {
      'trf1_mean': results_dict.get('LMA_baseline', {}).get('wall_time_mean', np.nan),
      'trf1_median': results_dict.get('LMA_baseline', {}).get('wall_time_median', np.nan),
      'trf1_std': results_dict.get('LMA_baseline', {}).get('wall_time_std', np.nan),
      'trf1_gflops_per_sample': results_dict.get('LMA_baseline', {}).get('gflops_per_sample', np.nan),
      'trf1_n_func_evals': results_dict.get('LMA_baseline', {}).get('n_function_evaluations', np.nan),
      'trf2_mean': results_dict.get('LMA_noisy_init', {}).get('wall_time_mean', np.nan),
      'trf2_median': results_dict.get('LMA_noisy_init', {}).get('wall_time_median', np.nan),
      'trf2_std': results_dict.get('LMA_noisy_init', {}).get('wall_time_std', np.nan),
      'trf2_gflops_per_sample': results_dict.get('LMA_noisy_init', {}).get('gflops_per_sample', np.nan),
      'trf2_n_func_evals': results_dict.get('LMA_noisy_init', {}).get('n_function_evaluations', np.nan),
      'du_mean': results_dict.get('DU_batch_1', {}).get('wall_time_mean', np.nan),
      'du_median': results_dict.get('DU_batch_1', {}).get('wall_time_median', np.nan),
      'du_std': results_dict.get('DU_batch_1', {}).get('wall_time_std', np.nan),
      'du_gflops_per_sample': results_dict.get('DU_batch_1', {}).get('gflops_per_sample', np.nan),
      'du_n_func_evals': results_dict.get('DU_batch_1', {}).get('n_function_evaluations', np.nan),
      'pidu_mean': results_dict.get('PIDU_batch_1', {}).get('wall_time_mean', np.nan),
      'pidu_median': results_dict.get('PIDU_batch_1', {}).get('wall_time_median', np.nan),
      'pidu_std': results_dict.get('PIDU_batch_1', {}).get('wall_time_std', np.nan),
      'pidu_gflops_per_sample': results_dict.get('PIDU_batch_1', {}).get('gflops_per_sample', np.nan),
      'pidu_n_func_evals': results_dict.get('PIDU_batch_1', {}).get('n_function_evaluations', np.nan),
    })
  
  # Create comparison plots
  if len(methods) > 1:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Performance Comparison', fontsize=16)
    
    # Plot 1: Time per sample
    ax = axes[0, 0]
    times = [results_dict[m].get('time_per_sample_ms', 0) for m in methods]
    ax.bar(methods, times, color='steelblue')
    ax.set_ylabel('Time per Sample (ms)')
    ax.set_title('Inference Time')
    ax.tick_params(axis='x', rotation=45)
    
    # Plot 2: Throughput
    ax = axes[0, 1]
    throughput = [results_dict[m].get('throughput_samples_per_sec', 0) for m in methods]
    ax.bar(methods, throughput, color='forestgreen')
    ax.set_ylabel('Samples per Second')
    ax.set_title('Throughput')
    ax.tick_params(axis='x', rotation=45)
    
    # Plot 3: GFLOPS
    ax = axes[1, 0]
    gflops = [results_dict[m].get('gflops_total', 0) for m in methods]
    if any(g > 0 for g in gflops):
      ax.bar(methods, gflops, color='coral')
      ax.set_ylabel('GFLOPS')
      ax.set_title('Computational Cost')
      ax.tick_params(axis='x', rotation=45)
    else:
      ax.text(0.5, 0.5, 'FLOPS data not available', 
              ha='center', va='center', transform=ax.transAxes)
      ax.set_title('Computational Cost')
    
    # Plot 4: Model Size
    ax = axes[1, 1]
    params = [results_dict[m].get('total_parameters', 0) / 1e6 for m in methods]
    ax.bar(methods, params, color='mediumpurple')
    ax.set_ylabel('Parameters (Millions)')
    ax.set_title('Model Size')
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
      plt.savefig(save_path, dpi=300, bbox_inches='tight')
      print(f"Comparison plot saved to: {save_path}")
    
    plt.show()
  
  return df

def export_to_overleaf_table(case: int|str, statistics: dict[str, Any]) -> None:
  overleaf_table_template = r"""
    \begin{table}[h]
      \centering
      \caption{Placeholder Caption}
      \label{tab:placeholder_label}
      \begin{tabular}{lccccc}
          \toprule
          & Mean & Median & Std & N\_Evals & FLOPs \\
          \midrule
          TRF1 & {value1} & {value2} & {value3} & {value4} & {value5} \\
          TRF2 & {value6} & {value7} & {value8} & {value9} & {value10} \\
          DU   & {value11} & {value12} & {value13} & {value14} & {value15} \\
          PIDU & {value16} & {value17} & {value18} & {value19} & {value20} \\
          \bottomrule
      \end{tabular}
  \end{table}
  """
  
  overleaf_table_template = overleaf_table_template.replace("{value1}", f"{statistics['trf1_mean']:.3f}")
  overleaf_table_template = overleaf_table_template.replace("{value2}", f"{statistics['trf1_median']:.3f}")
  overleaf_table_template = overleaf_table_template.replace("{value3}", f"{statistics['trf1_std']:.3f}")
  overleaf_table_template = overleaf_table_template.replace("{value4}", f"{statistics['trf1_n_func_evals']:.1f}")
  overleaf_table_template = overleaf_table_template.replace("{value5}", f"{statistics['trf1_gflops_per_sample']:.3f}")
  overleaf_table_template = overleaf_table_template.replace("{value6}", f"{statistics['trf2_mean']:.3f}")
  overleaf_table_template = overleaf_table_template.replace("{value7}", f"{statistics['trf2_median']:.3f}")
  overleaf_table_template = overleaf_table_template.replace("{value8}", f"{statistics['trf2_std']:.3f}")
  overleaf_table_template = overleaf_table_template.replace("{value9}", f"{statistics['trf2_n_func_evals']:.1f}")
  overleaf_table_template = overleaf_table_template.replace("{value10}", f"{statistics['trf2_gflops_per_sample']:.3f}")
  overleaf_table_template = overleaf_table_template.replace("{value11}", f"{statistics['du_mean']:.3f}")
  overleaf_table_template = overleaf_table_template.replace("{value12}", f"{statistics['du_median']:.3f}")
  overleaf_table_template = overleaf_table_template.replace("{value13}", f"{statistics['du_std']:.3f}")
  overleaf_table_template = overleaf_table_template.replace("{value14}", f"{statistics['du_n_func_evals']:.1f}")
  overleaf_table_template = overleaf_table_template.replace("{value15}", f"{statistics['du_gflops_per_sample']:.3f}")
  overleaf_table_template = overleaf_table_template.replace("{value16}", f"{statistics['pidu_mean']:.3f}")
  overleaf_table_template = overleaf_table_template.replace("{value17}", f"{statistics['pidu_median']:.3f}")
  overleaf_table_template = overleaf_table_template.replace("{value18}", f"{statistics['pidu_std']:.3f}")
  overleaf_table_template = overleaf_table_template.replace("{value19}", f"{statistics['pidu_n_func_evals']:.1f}")
  overleaf_table_template = overleaf_table_template.replace("{value20}", f"{statistics['pidu_gflops_per_sample']:.3f}")
  
  os.makedirs("outputs/timings", exist_ok=True)
  with open(f"outputs/timings/overleaf_table_{case}.txt", "w") as f:
      f.write(overleaf_table_template)
  
  print(f"Overleaf table template exported to outputs/timings/overleaf_table_{case}.txt")

if __name__ == '__main__':  
  # -- Load data
  print(print_memory_usage(verbose=1))
  datasets, dataset_info, device, dtype = get_dataset(
    usecase=usecase,
    trainset=f'usecase{usecase}/case{usecase}_train_n2_{12}',
    valset=f'usecase{usecase}/case{usecase}_val_n2_{8}',
    testset=f'usecase{usecase}/case{usecase}_test_n2_{10}',
    # expset=f'usecase{uc}/case{uc}_experimental', # 
    expset=False,
    )

  # -- Load model
  print(print_memory_usage(verbose=1))
  model = get_model(
    ds_info=dataset_info, 
    device=device, 
    dtype=dtype, 
    physics=True, 
    active_models=active_models
  )
  
  # Setup output directory
  output_dir = os.path.join(PROJECT_DIR, 'outputs', 'timings')
  os.makedirs(output_dir, exist_ok=True)
  
  # ========================================================================
  # BENCHMARK PIDU
  # ========================================================================
  print("\n" + "="*80)
  print("BENCHMARKING PIDU")
  print("="*80)
  
  nn_results = {}
  for batch_size in [1]:
    print(f"\n--- Batch Size: {batch_size} ---")
    result = benchmark_inference(
      model=model,
      dataset=datasets['val'],
      batch_size=batch_size,
      num_warmup=10,
      num_iterations=100,
      device=device,
      verbose=True
    )
    nn_results[f'PIDU_batch_{batch_size}'] = result
    
  # ========================================================================
  # BENCHMARK DU
  # ========================================================================
  print("\n" + "="*80)
  print("BENCHMARKING DU")
  print("="*80)
  
  print(print_memory_usage(verbose=1))
  model = get_model(
    ds_info=dataset_info, 
    device=device, 
    dtype=dtype, 
    physics=False, 
    active_models=active_models,
  )
  
  for batch_size in [1]:
    print(f"\n--- Batch Size: {batch_size} ---")
    result = benchmark_inference(
      model=model,
      dataset=datasets['val'],
      batch_size=batch_size,
      num_warmup=10,
      num_iterations=100,
      device=device,
      verbose=True
    )
    nn_results[f'DU_batch_{batch_size}'] = result
  
  # ========================================================================
  # BENCHMARK LEVENBERG-MARQUARDT ALGORITHM
  # ========================================================================
  print("\n" + "="*80)
  print("BENCHMARKING LEVENBERG-MARQUARDT ALGORITHM")
  print("="*80)
  
  lma_results = {}
  
  # LMA with baseline initialization (zeros)
  print("\n--- LMA with Baseline Initialization ---")
  lma_baseline = benchmark_lma(
      dataset=datasets['val'],
      num_samples=5,
      max_iterations=400,
      initialization_scheme='noisy',
      noise_factor=0.20,
      all_osc=False,
      backend='numpy',
  )
  lma_results['LMA_baseline'] = lma_baseline

  print("\n--- LMA with Baseline (2) Initialization ---")
  lma_baseline2 = benchmark_lma(
      dataset=datasets['val'],
      num_samples=5,
      max_iterations=100,
      initialization_scheme='baseline',
      noise_factor=0.25,
      all_osc=True,
      backend='numpy',
  )
  lma_results['LMA_baseline_2'] = lma_baseline2

  # LMA with noisy initialization (starting near true values)
  print("\n--- LMA with Noisy Initialization ---")
  lma_noisy = benchmark_lma(
      dataset=datasets['val'],
      num_samples=5,
      max_iterations=100,
      initialization_scheme='noisy',
      noise_factor=0.25,
      all_osc=False,
      backend='numpy',
  )
  lma_results['LMA_noisy_init'] = lma_noisy

  # LMA with noisy initialization (2) (starting near true values)
  print("\n--- LMA with Noisy Initialization ---")
  lma_noisy2 = benchmark_lma(
      dataset=datasets['val'],
      num_samples=5,
      max_iterations=100,
      initialization_scheme='noisy',
      noise_factor=0.25,
      all_osc=True,
      backend='numpy',
  )
  lma_results['LMA_noisy_init_2'] = lma_noisy2
  
  # ========================================================================
  # COMPARE ALL METHODS
  # ========================================================================
  print("\n" + "="*80)
  print("COMPARING ALL METHODS")
  print("="*80)
  
  # Combine all results for comparison
  all_results = {**nn_results, **lma_results}
  
  # Create comparison DataFrame and plots
  comparison_df = compare_methods(
    all_results,
    export_overleaf_table=True,
    save_path=os.path.join(output_dir, 'method_comparison.png')
  )
  
  # ========================================================================
  # SAVE RESULTS
  # ========================================================================
  
  # Save as pickle
  with open(os.path.join(output_dir, 'benchmark_results.pkl'), 'wb') as f:
    pickle.dump({'nn': nn_results, 'lma': lma_results}, f)
  
  # Save NN results as CSV
  df_nn = pd.DataFrame([
    {
      'method': key,
      'batch_size': res.get('batch_size', 1),
      'time_per_sample_ms': res['time_per_sample_ms'],
      'throughput': res['throughput_samples_per_sec'],
      'gflops_total': res.get('gflops_total', np.nan),
      'gflops_per_sample': res.get('gflops_per_sample', np.nan),
      'total_params': res['total_parameters'],
    }
    for key, res in nn_results.items()
  ])
  df_nn.to_csv(os.path.join(output_dir, 'nn_benchmark_results.csv'), index=False)
  
  # Save LMA results as CSV
  df_lma = pd.DataFrame([
    {
      'method': key,
      'num_samples': res['num_samples'],
      'time_per_sample_ms': res['time_per_sample_ms'],
      'throughput': res['throughput_samples_per_sec'],
      'mean_iterations': res['iterations_mean'],
      'success_rate': res['success_rate'],
      'mean_cost': res['cost_mean'],
      'mean_abs_error': np.mean(res['abs_error_mean']),
      'mean_rel_error': np.mean(res['rel_error_mean']),
    }
    for key, res in lma_results.items()
  ])
  df_lma.to_csv(os.path.join(output_dir, 'lma_benchmark_results.csv'), index=False)
  
  # Save comparison table
  comparison_df.to_excel(os.path.join(output_dir, f'method_comparison_case{usecase}.xlsx'), index=False)
  
  print(f"\nAll results saved to: {output_dir}")
  print(f"  - benchmark_results.pkl")
  print(f"  - nn_benchmark_results.csv")
  print(f"  - lma_benchmark_results.csv")
  print(f"  - method_comparison.csv")
  print(f"  - method_comparison.png")
  
  # ========================================================================
  # SUMMARY
  # ========================================================================
  print("\n" + "="*80)
  print("SUMMARY")
  print("="*80)
  print(f"\nPIDU (batch_size=1):")
  print(f"  Time per sample: {nn_results['PIDU_batch_1']['time_per_sample_ms']:.3f} ms")
  print(f"  Throughput:      {nn_results['PIDU_batch_1']['throughput_samples_per_sec']:.2f} samples/sec")

  print(f"\nDU (batch_size=1):")
  print(f"  Time per sample: {nn_results['DU_batch_1']['time_per_sample_ms']:.3f} ms")
  print(f"  Throughput:      {nn_results['DU_batch_1']['throughput_samples_per_sec']:.2f} samples/sec")

  print(f"\nLMA (baseline init):")
  print(f"  Time per sample: {lma_baseline['time_per_sample_ms']:.3f} ms")
  print(f"  Throughput:      {lma_baseline['throughput_samples_per_sec']:.2f} samples/sec")
  print(f"  Success rate:    {lma_baseline['success_rate']*100:.1f}%")
  
  print(f"\nLMA (2) (baseline init):")
  print(f"  Time per sample: {lma_baseline2['time_per_sample_ms']:.3f} ms")
  print(f"  Throughput:      {lma_baseline2['throughput_samples_per_sec']:.2f} samples/sec")
  print(f"  Success rate:    {lma_baseline2['success_rate']*100:.1f}%")
  
  print(f"\nLMA (noisy init):")
  print(f"  Time per sample: {lma_noisy['time_per_sample_ms']:.3f} ms")
  print(f"  Throughput:      {lma_noisy['throughput_samples_per_sec']:.2f} samples/sec")
  print(f"  Success rate:    {lma_noisy['success_rate']*100:.1f}%")
  
  print(f"\nLMA (2) (noisy init):")
  print(f"  Time per sample: {lma_noisy2['time_per_sample_ms']:.3f} ms")
  print(f"  Throughput:      {lma_noisy2['throughput_samples_per_sec']:.2f} samples/sec")
  print(f"  Success rate:    {lma_noisy2['success_rate']*100:.1f}%")
  
  speedup = lma_baseline['time_per_sample_ms'] / nn_results['PIDU_batch_1']['time_per_sample_ms']
  print(f"\nSpeedup (NN vs LMA baseline): {speedup:.2f}x")
  print("="*80)