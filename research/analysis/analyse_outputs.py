"""
Analyze residual performance across multiple runs.
Compares DU vs PIDU performance for different parameter groups.
"""

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from pidu.model.networks.io import load_checkpoint
from pidu.paths import ROOT, case_dir
from typing import Dict, List, Tuple, Optional
import seaborn as sns
import argparse

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def load_residual_data(run_folder: Path, dataset_type: str = "test_test", filename: str = "res_dis.xlsx") -> Optional[pd.DataFrame]:
    """
    Load residual distribution data from a specific run folder.
    
    Args:
        run_folder: Path to the run folder (e.g., loss_DU_0_000)
        dataset_type: Preferred split folder (e.g. "test_test", "test_val", "test_exp")
        filename: Name of the Excel file to load (e.g. "res_dis.xlsx" or "res_dis_percentage.xlsx")
    
    Returns:
        DataFrame with residual data or None if file doesn't exist
    """
    candidate_splits = [dataset_type]
    if dataset_type != "test_test":
        candidate_splits.append("test_test")

    for split_name in candidate_splits:
        file_path = run_folder / split_name / filename
        if not file_path.exists():
            continue

        if split_name != dataset_type:
            print(f"Info: Using fallback split '{split_name}' for {run_folder.name} (requested '{dataset_type}')")

        try:
            return pd.read_excel(file_path)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None

    print(f"Warning: No residual file found for {run_folder} with splits {candidate_splits} (file: {filename})")
    return None


def load_all_runs(base_folder: Path, trail: int, run_prefix: str, num_runs: int = 8, 
                  dataset_type: str = "test_test", filename: str = "res_dis.xlsx") -> Dict[int, pd.DataFrame]:
    """
    Load residual data from all runs for a given prefix.
    
    Args:
        base_folder: Base outputs folder
        run_prefix: Prefix for run folders (e.g., "loss_DU", "loss_PIDU", "loss_lma_baseline")
        num_runs: Number of runs to load
        dataset_type: Either "test_test" or "test_val"
        filename: Name of the Excel file to load (e.g. "res_dis.xlsx" or "res_dis_percentage.xlsx")
    
    Returns:
        Dictionary mapping run_id to DataFrame
    """
    runs_data = {}
    
    found_indexed_run = False

    for run_id in range(num_runs):
        folder_name = f"{run_prefix}_{run_id}_{trail:03d}"
        run_folder = base_folder / folder_name
        
        if not run_folder.exists():
            continue
        
        found_indexed_run = True
        
        df = load_residual_data(run_folder, dataset_type, filename=filename)
        if df is not None:
            runs_data[run_id] = df

    if found_indexed_run:
        print(f"Loaded {len(runs_data)} runs for {run_prefix}")
        return runs_data

    # Fallback for single-run folders without indexed suffixes
    single_run_folder = base_folder / run_prefix
    if single_run_folder.exists():
        df = load_residual_data(single_run_folder, dataset_type, filename=filename)
        if df is not None:
            runs_data[0] = df
            print(f"Loaded single run for {run_prefix} from {single_run_folder.name}")
        else:
            print(f"Warning: Could not load residual data from single run folder {single_run_folder}")
    else:
        print(f"Warning: No indexed runs or single-run folder found for {run_prefix}")
    
    print(f"Loaded {len(runs_data)} runs for {run_prefix}")
    return runs_data


def output_folder_to_model_path(
    output_folder_name: str,
    usecase: int,
) -> Tuple[Path, str, int, int]:
    """
    Resolve the model .pt file that corresponds to a given output folder.

    Naming conventions
    ------------------
    Output folder : ``loss_{model_type}_{run_id}_{trail:03d}``
                    e.g. ``loss_CNN_0_001``  → model_type=CNN, run_id=0, trail=1
    Model file    : ``{model_type}-t{trail:02d}{run_id:03d}.pt``
                    e.g. ``CNN-t01000.pt``   ← trail=1 (01), run=0 (000)

    Returns
    -------
    (model_path, model_type, run_id, trail)
    """
    m = re.match(r'^loss_([A-Za-z]+)_(\d+)_(\d+)$', output_folder_name)
    if m is None:
        raise ValueError(
            f"Cannot parse output folder name '{output_folder_name}'. "
            "Expected format: loss_<ModelType>_<run_id>_<trail:03d>"
        )
    model_type = m.group(1)
    run_id     = int(m.group(2))
    trail      = int(m.group(3))

    models_dir  = ROOT / 'models'
    model_file  = f'{model_type}-t{trail:02d}{run_id:03d}.pt'
    model_path  = models_dir / model_file

    return model_path, model_type, run_id, trail


def validate_model_output(
    output_folder_name: str,
    usecase: int,
    device: str = 'cpu',
    batch_size: int = 32,
) -> Dict[str, 'pd.DataFrame']:
    """
    Re-run inference for the model that corresponds to *output_folder_name*,
    compute fresh percentage residuals vs ground truth, and compare their
    distribution against the saved ``res_dis_percentage.xlsx``.

    Dataset is always ``data/datasets/usecase{n}/case{n}_test_n2_10``.
    Model is always loaded from ``models/``.
    """
    import torch
    from torch.utils.data import DataLoader

    PROJECT_DIR = ROOT

    from pidu.data.simulation import load_sim_ds
    from pidu.data.utils import pt2np

    # ---- Resolve paths -------------------------------------------------------
    model_path, model_type, run_id, trail = output_folder_to_model_path(
        output_folder_name, usecase
    )
    if not model_path.exists():
        raise FileNotFoundError(f'Model file not found: {model_path}')

    dataset_split = 'test_test'
    sheet_name    = 'system1'

    outputs_dir   = case_dir(usecase)
    output_folder = outputs_dir / output_folder_name
    saved_path    = output_folder / dataset_split / 'res_dis_percentage.xlsx'
    if not saved_path.exists():
        raise FileNotFoundError(f'Saved residuals not found: {saved_path}')

    print(f'Model   : {model_path.name}')
    print(f'Outputs : {output_folder.name} / {dataset_split}')

    # ---- Load saved percentage residuals ------------------------------------
    saved_df = pd.read_excel(saved_path, sheet_name=sheet_name, index_col=0)
    saved_df = saved_df.drop(
        columns=[c for c in saved_df.columns if str(c).lower().startswith('unnamed')],
        errors='ignore',
    )
    saved_num = saved_df.apply(pd.to_numeric, errors='coerce')

    # ---- Load model ----------------------------------------------------------
    model = load_checkpoint(model_path, map_location=device)
    model.eval()
    print(f'Loaded model with {sum(p.numel() for p in model.parameters()):,} parameters')

    # ---- Load dataset --------------------------------------------------------
    try:
        ds_name = f'case{usecase}_test_n2_10_wsn'
        ds = load_sim_ds(f'usecase{usecase}/{ds_name}', y_return_mode='both', device=device)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    except Exception as e:
        ds_name = f'case{usecase}_test_n2_10'
        ds = load_sim_ds(f'usecase{usecase}/{ds_name}', y_return_mode='both', device=device)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    
    print(f'Dataset : {ds_name}  ({len(ds)} samples)')

    # ---- Run inference -------------------------------------------------------
    all_p_sys:   List[np.ndarray] = []
    all_true_sys: List[np.ndarray] = []

    with torch.no_grad():
        for wn, x_sys, x_osc, y, _idx in loader:
            # wn is batched [batch, n_wn]; model expects [n_wn] — take first sample (same for all)
            wn = {k: v[0] for k, v in wn.items()}
            # x_sys = (init, true); use noised y channel when available
            y_in = {k: v[..., 1] if v.ndim == 3 else v for k, v in y.items()}
            out  = model(wn=wn, y=y_in, x_sys=x_sys, x_osc=x_osc)

            p_sys_out = out.get('p_sys', {})
            if sheet_name in p_sys_out:
                p = p_sys_out[sheet_name]
                if isinstance(p, torch.Tensor):
                    last = p[-1]  # last unfold: [batch, n_layers, n_params]
                    all_p_sys.append(pt2np(last).reshape(last.shape[0], -1))

            # Ground truth for this system
            x_true = x_sys[1].get(sheet_name) if isinstance(x_sys[1], dict) else x_sys[1]
            if x_true is not None and isinstance(x_true, torch.Tensor):
                all_true_sys.append(pt2np(x_true).reshape(x_true.shape[0], -1))

    if not all_p_sys:
        raise RuntimeError(
            f'Model produced no p_sys output for system "{sheet_name}". '
            'Check that the model is compatible with this system name.'
        )

    n_samples = len(ds)
    pred_arr = np.concatenate(all_p_sys,    axis=0)[:n_samples]
    true_arr = np.concatenate(all_true_sys, axis=0)[:n_samples]

    # ---- Compute fresh percentage residuals ----------------------------------
    # Align column count to whatever the saved file has
    n_cols = min(pred_arr.shape[1], len(saved_num.columns))
    col_names = saved_num.columns[:n_cols]

    fresh_pct = np.abs(true_arr[:, :n_cols] - pred_arr[:, :n_cols]) / (np.abs(true_arr[:, :n_cols]) + 1e-8)
    fresh_df  = pd.DataFrame(fresh_pct, columns=col_names)

    saved_aligned = saved_num.iloc[:n_samples, :n_cols].reset_index(drop=True)

    # ---- Print distribution comparison ---------------------------------------
    print(f'\nResidual distribution (fraction)  [{output_folder_name} | {dataset_split} | {sheet_name}]')
    print(f'  Samples : {n_samples}   Columns : {n_cols}')
    print(f'  {"Parameter":<24}  {"saved mean":>10}  {"fresh mean":>10}  {"saved std":>10}  {"fresh std":>10}')
    print(f'  {"-"*68}')
    for col in col_names:
        s = saved_aligned[col].dropna()
        f = fresh_df[col].dropna()
        print(f'  {col:<24}  {s.mean():>10.3f}  {f.mean():>10.3f}  {s.std():>10.3f}  {f.std():>10.3f}')

    overall_saved = saved_aligned.to_numpy(dtype=float, na_value=np.nan).ravel()
    overall_fresh = fresh_df.to_numpy(dtype=float, na_value=np.nan).ravel()
    print(f'  {"-"*68}')
    print(f'  {"OVERALL":<24}  {np.nanmean(overall_saved):>10.3f}  {np.nanmean(overall_fresh):>10.3f}  {np.nanstd(overall_saved):>10.3f}  {np.nanstd(overall_fresh):>10.3f}')

    return {
        'saved_pct' : saved_aligned,
        'fresh_pct' : fresh_df,
    }


def group_parameters(columns: List[str]) -> Dict[str, List[str]]:
    """
    Group parameter columns by type.
    
    Args:
        columns: List of column names
    
    Returns:
        Dictionary with groups: 'eps_inf', 'w0', 'wp', 'g'
    """
    groups = {
        'eps_inf': [],
        'w0': [],
        'wp': [],
        'g': []
    }
    
    for col in columns:
        col_lower = col.lower()
        if 'eps_inf' in col_lower or col_lower == 'eps_inf':
            groups['eps_inf'].append(col)
        elif 'w0' in col_lower or col_lower.startswith('w0'):
            groups['w0'].append(col)
        elif 'wp' in col_lower or col_lower.startswith('wp'):
            groups['wp'].append(col)
        elif col_lower == 'g' or col_lower.startswith('g_'):
            groups['g'].append(col)
    
    # Remove empty groups
    groups = {k: sorted(v) for k, v in groups.items() if v}
    
    return groups


def get_parameter_groups(columns: List[str], distribution_mode: str = "grouped") -> Dict[str, List[str]]:
    """
    Build parameter groups for residual analysis.

    Args:
        columns: List of dataframe columns
        distribution_mode: 'grouped' (w0/wp/g buckets) or 'split' (one sheet/group per parameter)

    Returns:
        Dictionary mapping group name to list of parameter columns
    """
    grouped = group_parameters(columns)

    if distribution_mode == "split":
        return {
            param_name: [param_name]
            for params in grouped.values()
            for param_name in params
        }

    return grouped


def compute_run_statistics(runs_data: Dict[int, pd.DataFrame], 
                          param_groups: Dict[str, List[str]]) -> Dict[str, pd.DataFrame]:
    """
    Compute statistics (mean, std, var) for each parameter group across runs.
    
    Args:
        runs_data: Dictionary mapping run_id to DataFrame
        param_groups: Dictionary of parameter groups
    
    Returns:
        Dictionary mapping group name to statistics DataFrame
    """
    statistics = {}
    
    for group_name, param_cols in param_groups.items():
        stats_list = []
        
        for param in param_cols:
            param_stats = []
            
            for run_id, df in runs_data.items():
                if param in df.columns:
                    # Get the residual values for this parameter
                    values = df[param].dropna()
                    if len(values) > 0:
                        param_stats.append({
                            'run_id': run_id,
                            'parameter': param,
                            'mean': values.mean(),
                            'std': values.std(),
                            'var': values.var(),
                            'median': values.median(),
                            'min': values.min(),
                            'max': values.max(),
                            'count': len(values)
                        })
            
            if param_stats:
                stats_list.extend(param_stats)
        
        if stats_list:
            statistics[group_name] = pd.DataFrame(stats_list)
    
    return statistics


def aggregate_across_runs(statistics: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Aggregate statistics across all runs for each parameter.
    
    Args:
        statistics: Dictionary mapping group name to statistics DataFrame
    
    Returns:
        Dictionary mapping group name to aggregated statistics DataFrame
    """
    aggregated = {}
    
    for group_name, stats_df in statistics.items():
        # Group by parameter and compute statistics of statistics
        agg_df = stats_df.groupby('parameter').agg({
            'mean': ['mean', 'std', 'min', 'max'],
            'std': ['mean', 'std', 'min', 'max'],
            'var': ['mean', 'std', 'min', 'max'],
            'median': ['mean', 'std', 'min', 'max'],
            'count': 'sum'
        }).reset_index()
        
        # Flatten column names
        agg_df.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                          for col in agg_df.columns.values]
        
        aggregated[group_name] = agg_df
    
    return aggregated


def compare_runs(stats1: Dict[str, pd.DataFrame], 
                stats2: Dict[str, pd.DataFrame],
                name1: str = "DU", 
                name2: str = "PIDU") -> Dict[str, pd.DataFrame]:
    """
    Compare statistics between two run types (e.g., DU vs PIDU).
    
    Args:
        stats1: Aggregated statistics for first run type
        stats2: Aggregated statistics for second run type
        name1: Name of first run type
        name2: Name of second run type
    
    Returns:
        Dictionary mapping group name to comparison DataFrame
    """
    comparisons = {}
    
    for group_name in stats1.keys():
        if group_name not in stats2:
            print(f"Warning: Group {group_name} not in {name2}")
            continue
        
        df1 = stats1[group_name].copy()
        df2 = stats2[group_name].copy()
        
        # Rename columns to indicate which run type
        df1 = df1.rename(columns={col: f"{name1}_{col}" if col != 'parameter' else col 
                                  for col in df1.columns})
        df2 = df2.rename(columns={col: f"{name2}_{col}" if col != 'parameter' else col 
                                  for col in df2.columns})
        
        # Merge on parameter
        comparison = pd.merge(df1, df2, on='parameter', how='outer')
        
        # Compute differences and ratios
        if f'{name1}_mean_mean' in comparison.columns and f'{name2}_mean_mean' in comparison.columns:
            comparison['mean_diff'] = comparison[f'{name1}_mean_mean'] - comparison[f'{name2}_mean_mean']
            comparison['mean_ratio'] = comparison[f'{name1}_mean_mean'] / comparison[f'{name2}_mean_mean']
            
            comparison['var_diff'] = comparison[f'{name1}_var_mean'] - comparison[f'{name2}_var_mean']
            comparison['var_ratio'] = comparison[f'{name1}_var_mean'] / comparison[f'{name2}_var_mean']
        
        comparisons[group_name] = comparison
    
    return comparisons


def plot_comparison(
  comparison: pd.DataFrame, 
  group_name: str, 
  name1: str = "DU", 
  name2: str = "PIDU",
  save_path: Optional[Path] = None,
  show_plots: bool = False,
  ) -> None:
    """
    Plot comparison between two run types for a parameter group.
    
    Args:
        comparison: Comparison DataFrame
        group_name: Name of parameter group
        name1: Name of first run type
        name2: Name of second run type
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    parameters = comparison['parameter'].values
    x_pos = np.arange(len(parameters))
    
    # Plot 1: Mean comparison
    ax = axes[0, 0]
    mean1 = comparison[f'{name1}_mean_mean'].values
    mean2 = comparison[f'{name2}_mean_mean'].values
    std1 = comparison[f'{name1}_mean_std'].values
    std2 = comparison[f'{name2}_mean_std'].values
    
    width = 0.35
    ax.bar(x_pos - width/2, mean1, width, label=name1, alpha=0.8, yerr=std1, capsize=5)
    ax.bar(x_pos + width/2, mean2, width, label=name2, alpha=0.8, yerr=std2, capsize=5)
    ax.set_ylabel('Mean Residual', fontsize=12)
    ax.set_title(f'{group_name.upper()} - Mean Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(parameters, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Variance comparison
    ax = axes[0, 1]
    var1 = comparison[f'{name1}_var_mean'].values
    var2 = comparison[f'{name2}_var_mean'].values
    var_std1 = comparison[f'{name1}_var_std'].values
    var_std2 = comparison[f'{name2}_var_std'].values
    
    ax.bar(x_pos - width/2, var1, width, label=name1, alpha=0.8, yerr=var_std1, capsize=5)
    ax.bar(x_pos + width/2, var2, width, label=name2, alpha=0.8, yerr=var_std2, capsize=5)
    ax.set_ylabel('Variance', fontsize=12)
    ax.set_title(f'{group_name.upper()} - Variance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(parameters, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Mean difference
    ax = axes[1, 0]
    if 'mean_diff' in comparison.columns:
        mean_diff = comparison['mean_diff'].values
        colors = ['green' if x < 0 else 'red' for x in mean_diff]
        ax.bar(x_pos, mean_diff, color=colors, alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.set_ylabel(f'Mean Difference ({name1} - {name2})', fontsize=12)
        ax.set_title(f'{group_name.upper()} - Mean Difference', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(parameters, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
    
    # Plot 4: Variance difference
    ax = axes[1, 1]
    if 'var_diff' in comparison.columns:
        var_diff = comparison['var_diff'].values
        colors = ['green' if x < 0 else 'red' for x in var_diff]
        ax.bar(x_pos, var_diff, color=colors, alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.set_ylabel(f'Variance Difference ({name1} - {name2})', fontsize=12)
        ax.set_title(f'{group_name.upper()} - Variance Difference', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(parameters, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    if show_plots:
        plt.show()
    
    # Clean up to free memory
    plt.close()
    plt.cla()
    plt.clf()


def plot_multi_loss_comparison(
  all_agg: Dict[str, pd.DataFrame], 
  group_name: str,
  loss_types: List[str], 
  save_path: Optional[Path] = None,
  y_max: Optional[float] = None,
  show_plots: bool = False,
):
    """
    Plot simultaneous comparison of multiple loss types for a parameter group.
    
    Args:
        all_agg: Dictionary mapping loss type (e.g., 'loss_DU') to aggregated statistics DataFrame
        group_name: Name of parameter group
        loss_types: List of loss type prefixes (e.g., ['loss_CNN', 'loss_DU', 'loss_PIDU'])
        save_path: Optional path to save the plot
    """
    # Extract shortened names for display
    loss_names = [lt.replace('loss_', '') for lt in loss_types]
    colors = plt.cm.Set2(np.linspace(0, 1, len(loss_types)))
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    # Build the union of parameters across all loss types that have data for this group
    seen = set()
    parameters = []
    for lt in loss_types:
        if lt in all_agg and group_name in all_agg[lt]:
            for p in all_agg[lt][group_name]['parameter'].values:
                if p not in seen:
                    seen.add(p)
                    parameters.append(p)
    parameters = sorted(parameters)
    if not parameters:
        print(f"Warning: No data for group {group_name} in any loss type")
        return
    x_pos = np.arange(len(parameters))
    
    # Plot 1: Mean comparison across all loss types
    ax = axes[0]
    width = 0.8 / len(loss_types)
    
    for idx, (loss_type, loss_name) in enumerate(zip(loss_types, loss_names)):
        if loss_type in all_agg and group_name in all_agg[loss_type]:
            df = all_agg[loss_type][group_name].set_index('parameter').reindex(parameters)
            means = df['mean_mean'].values
            stds = np.nan_to_num(df['mean_std'].values, nan=0.0)
            positions = x_pos + (idx - len(loss_types)/2 + 0.5) * width
            ax.bar(positions, means, width, label=loss_name, alpha=0.85, 
                   yerr=stds, capsize=4, color=colors[idx])
    
    if y_max is not None:
        axes[0].set_ylim(top=y_max)
    ax.set_ylabel('Mean Residual', fontsize=12)
    ax.set_title(f'{group_name.upper()} - Mean Residual Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos) 
    ax.set_xticklabels(parameters, rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Variance comparison across all loss types
    ax = axes[1]
    
    for idx, (loss_type, loss_name) in enumerate(zip(loss_types, loss_names)):
        if loss_type in all_agg and group_name in all_agg[loss_type]:
            df = all_agg[loss_type][group_name].set_index('parameter').reindex(parameters)
            vars = df['var_mean'].values
            var_stds = np.nan_to_num(df['var_std'].values, nan=0.0)
            positions = x_pos + (idx - len(loss_types)/2 + 0.5) * width
            ax.bar(positions, vars, width, label=loss_name, alpha=0.85,
                   yerr=var_stds, capsize=4, color=colors[idx])
    
    if y_max is not None:
        axes[1].set_ylim(top=y_max)
    ax.set_ylabel('Variance', fontsize=12)
    ax.set_title(f'{group_name.upper()} - Variance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(parameters, rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'Multi-Loss Type Comparison: {" vs ".join(loss_names)}', 
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    if show_plots:
      plt.show()
    
    # Clean up to free memory
    plt.close()
    plt.cla()
    plt.clf()


def export_residuals_to_excel(
  all_runs: Dict[str, Dict[int, pd.DataFrame]], 
  param_groups: Dict[str, List[str]],
  save_path: Path
):
    """
    Export residual values to Excel with one sheet per parameter.
    Each column represents a different method.
    
    Args:
        all_runs: Dictionary mapping loss_type to dict of run_id to DataFrame
        param_groups: Dictionary mapping group name to list of parameters
        save_path: Path to save the Excel file
    """
    
    print(f"\nExporting residuals to Excel: {save_path}")
    
    # Create a dictionary to hold data for each parameter
    param_data = {}
    
    for param_main, param_values in param_groups.items():
        columns_for_sheet: Dict[str, pd.Series] = {}

        for loss_type, runs in all_runs.items():
            merged_values: List[float] = []
            for _, df in runs.items():
                for param_name in param_values:
                    if param_name in df.columns:
                        merged_values.extend(df[param_name].dropna().tolist())

            if merged_values:
                columns_for_sheet[loss_type] = pd.Series(merged_values, name=loss_type)

        if columns_for_sheet:
            # Different methods can have different lengths; Series alignment prevents length mismatch errors.
            param_data[param_main] = pd.concat(columns_for_sheet.values(), axis=1)
    
    # Write to Excel
    with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
        for param, df in param_data.items():
            # Sanitize sheet name (Excel has 31 char limit and doesn't allow certain chars)
            sheet_name = param.replace('/', '_').replace('\\', '_').replace(':', '_')[:31]
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    print(f"Exported {len(param_data)} parameter sheets to {save_path}")


def print_percentage_average_table(
    all_runs_pct: Dict[str, Dict[int, pd.DataFrame]],
    param_groups: Dict[str, List[str]],
):
    """
    For each method print the linear average of mean percentage errors across parameter
    groups, computed only from non-zero values.  One row per method, one column per
    group plus an 'Overall' column that averages the group means.

    param_groups is used as a seed; columns present in any method but not yet mapped
    are also discovered and appended to the appropriate group so that no method is
    silently excluded.
    """
    # Build a unified set of groups from ALL methods' columns so that every method's
    # columns are covered, not just those of the first method.
    all_columns: List[str] = []
    for runs in all_runs_pct.values():
        for df in runs.values():
            for col in df.columns:
                if col not in all_columns:
                    all_columns.append(col)
    unified_groups = get_parameter_groups(all_columns, distribution_mode="grouped")

    group_names = list(unified_groups.keys())
    rows = []

    for loss_type, runs in all_runs_pct.items():
        row: Dict[str, object] = {'Method': loss_type.replace('loss_', '')}
        group_means = []

        for group_name in group_names:
            param_cols = unified_groups[group_name]
            values: List[float] = []
            for df in runs.values():
                for col in param_cols:
                    if col in df.columns:
                        vals = df[col].dropna().values
                        values.extend(vals[vals != 0.0].tolist())

            if values:
                gm = float(np.mean(np.abs(values)))
            else:
                gm = float('nan')

            row[group_name] = gm
            if not np.isnan(gm):
                group_means.append(gm)

        row['Overall'] = float(np.mean(group_means)) if group_means else float('nan')
        rows.append(row)

    if not rows:
        print("No percentage data available for summary table.")
        return

    summary_df = pd.DataFrame(rows).set_index('Method')
    col_order = group_names + ['Overall']
    summary_df = summary_df[[c for c in col_order if c in summary_df.columns]]

    print("\n" + "="*80)
    print("PERCENTAGE ERROR SUMMARY  (mean |pct error|, non-zero only)")
    print("="*80)
    print(summary_df.to_string(float_format=lambda x: f"{x:.4f}"))
    print("="*80)


def print_comparison_summary(comparisons: Dict[str, pd.DataFrame], 
                            name1: str = "DU", name2: str = "PIDU"):
    """
    Print a summary of the comparison between two run types.
    
    Args:
        comparisons: Dictionary mapping group name to comparison DataFrame
        name1: Name of first run type
        name2: Name of second run type
    """
    print("\n" + "="*80)
    print(f"COMPARISON SUMMARY: {name1} vs {name2}")
    print("="*80)
    
    for group_name, comp_df in comparisons.items():
        print(f"\n{'='*80}")
        print(f"Parameter Group: {group_name.upper()}")
        print(f"{'='*80}")
        
        for _, row in comp_df.iterrows():
            param = row['parameter']
            print(f"\n  Parameter: {param}")
            print(f"  {'-'*76}")
            
            # Mean statistics
            if f'{name1}_mean_mean' in row and f'{name2}_mean_mean' in row:
                mean1 = row[f'{name1}_mean_mean']
                mean2 = row[f'{name2}_mean_mean']
                mean_diff = row.get('mean_diff', mean1 - mean2)
                mean_pct = ((mean1 - mean2) / mean2 * 100) if mean2 != 0 else np.inf
                
                print(f"  Mean Residual:")
                print(f"    {name1}: {mean1:.6e} ± {row[f'{name1}_mean_std']:.6e}")
                print(f"    {name2}: {mean2:.6e} ± {row[f'{name2}_mean_std']:.6e}")
                print(f"    Difference: {mean_diff:.6e} ({mean_pct:+.2f}%)")
            
            # Variance statistics
            if f'{name1}_var_mean' in row and f'{name2}_var_mean' in row:
                var1 = row[f'{name1}_var_mean']
                var2 = row[f'{name2}_var_mean']
                var_diff = row.get('var_diff', var1 - var2)
                var_pct = ((var1 - var2) / var2 * 100) if var2 != 0 else np.inf
                
                print(f"  Variance:")
                print(f"    {name1}: {var1:.6e} ± {row[f'{name1}_var_std']:.6e}")
                print(f"    {name2}: {var2:.6e} ± {row[f'{name2}_var_std']:.6e}")
                print(f"    Difference: {var_diff:.6e} ({var_pct:+.2f}%)")
        
        print(f"\n{'='*80}")


def main(
    loss_types: List[str] = None,
    usecase: int = 1,
    trail: int = 0,
    num_runs: int = 8,
    dataset_type: str = "test_test",
    distribution_mode: str = "grouped",
    save_plots: bool = True,
    show_plots: bool = False
):
    """
    Main analysis function for comparing multiple loss types.
    
    Args:
        loss_types: List of loss type prefixes to compare (e.g., ['loss_DU', 'loss_PIDU', 'loss_CNN']).
                   Defaults to ['loss_DU', 'loss_PIDU']
        usecase: Integer specifying the case number to analyze
        trail: Trail number for the runs
        num_runs: Number of runs to load for each loss type
        dataset_type: Either "test_test" or "test_val"
        distribution_mode: 'grouped' to combine (w0/wp/g) or 'split' for per-parameter distributions
        save_plots: Whether to save plots to disk
        show_plots: Whether to show plots interactively
    """
    if loss_types is None:
        loss_types = ['loss_DU', 'loss_PIDU']
    
    # Default to case outputs folder
    base_folder = case_dir(usecase)
    
    print(f"Analyzing residuals from: {base_folder}")
    print(f"Loss types to compare: {loss_types}")
    print(f"Dataset type: {dataset_type}")
    print(f"Distribution mode: {distribution_mode}")
    print("="*80)
    
    # Load all runs for each loss type (absolute and percentage residuals)
    all_runs = {}
    all_runs_pct = {}
    all_stats = {}
    all_agg = {}
    all_stats_pct = {}
    all_agg_pct = {}
    
    for loss_type in loss_types:
        print(f"\nLoading {loss_type} runs...")
        runs = load_all_runs(base_folder, trail, loss_type, num_runs=num_runs, dataset_type=dataset_type)
        if runs:
            all_runs[loss_type] = runs
            # Compute statistics for this loss type
            first_df = next(iter(runs.values()))
            param_groups = get_parameter_groups(first_df.columns.tolist(), distribution_mode=distribution_mode)
            all_stats[loss_type] = compute_run_statistics(runs, param_groups)
            all_agg[loss_type] = aggregate_across_runs(all_stats[loss_type])

        runs_pct = load_all_runs(base_folder, trail, loss_type, num_runs=num_runs,
                                  dataset_type=dataset_type, filename="res_dis_percentage.xlsx")
        if runs_pct:
            all_runs_pct[loss_type] = runs_pct
            first_df_pct = next(iter(runs_pct.values()))
            param_groups_pct = get_parameter_groups(first_df_pct.columns.tolist(), distribution_mode=distribution_mode)
            all_stats_pct[loss_type] = compute_run_statistics(runs_pct, param_groups_pct)
            all_agg_pct[loss_type] = aggregate_across_runs(all_stats_pct[loss_type])
    
    if len(all_runs) < 2:
        print("Error: Could not load data from at least 2 loss types")
        return
    
    # Get parameter groups from first run
    first_loss_type = list(all_runs.keys())[0]
    first_df = next(iter(all_runs[first_loss_type].values()))
    param_groups = get_parameter_groups(first_df.columns.tolist(), distribution_mode=distribution_mode)
    
    print("\nParameter groups found:")
    for group_name, params in param_groups.items():
        print(f"  {group_name}: {params}")
    
    # Create simultaneous comparison plots for all loss types
    if save_plots:
        output_folder = case_dir(usecase) / "analyse" / "residual_comparison"
        output_folder.mkdir(parents=True, exist_ok=True)
        print(f"\nSaving plots to: {output_folder}")
    
    loss_type_list = list(all_runs.keys())
    comparison_name = "_vs_".join([lt.replace('loss_', '') for lt in loss_type_list])
    
    # Export absolute residuals to Excel
    if save_plots:
        excel_path = output_folder / f"residual_distribution_case{usecase}.xlsx"
        export_residuals_to_excel(all_runs, param_groups, excel_path)

    # Export percentage residuals to Excel
    if save_plots and all_runs_pct:
        first_pct_df = next(iter(next(iter(all_runs_pct.values())).values()))
        param_groups_pct = get_parameter_groups(first_pct_df.columns.tolist(), distribution_mode=distribution_mode)
        excel_path_pct = output_folder / f"residual_distribution_percentage_case{usecase}.xlsx"
        export_residuals_to_excel(all_runs_pct, param_groups_pct, excel_path_pct)
    
    for group_name in param_groups.keys():
        print(f"\nPlotting comparison for {group_name}...")
        if save_plots:
            save_path = output_folder / f"comparison_{comparison_name}_{group_name}_{dataset_type}.png"
        else:
            save_path = None
        plot_multi_loss_comparison(all_agg, group_name, loss_type_list, save_path=save_path)

    if all_agg_pct:
        pct_loss_type_list = list(all_agg_pct.keys())
        first_pct_df = next(iter(next(iter(all_runs_pct.values())).values()))
        param_groups_pct = get_parameter_groups(first_pct_df.columns.tolist(), distribution_mode=distribution_mode)
        for group_name in param_groups_pct.keys():
            print(f"\nPlotting percentage comparison for {group_name}...")
            if save_plots:
                save_path = output_folder / f"comparison_{comparison_name}_{group_name}_{dataset_type}_percentage.png"
            else:
                save_path = None
            plot_multi_loss_comparison(all_agg_pct, group_name, pct_loss_type_list, save_path=save_path, y_max=2.0)

        print_percentage_average_table(all_runs_pct, param_groups_pct)

    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)
    
    return {
        'all_runs': all_runs,
        'all_runs_pct': all_runs_pct,
        'all_stats': all_stats,
        'all_stats_pct': all_stats_pct,
        'all_agg': all_agg,
        'all_agg_pct': all_agg_pct,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze residual performance across multiple runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # 1) Default quick run (common baseline set, grouped distribution, test_test)\n"
            "  python -m research.analysis.analyse_outputs\n\n"
            "  # 2) Your cnn-size case (single pattern, trail 6, case 2, save plots)\n"
            "  python -m research.analysis.analyse_outputs --base-patterns loss_CNN_cnn_size_-0p1 --trail 6 --usecase 2 --export-plots\n\n"
            "  # 3) Compare multiple cnn-size variants in one run\n"
            "  python -m research.analysis.analyse_outputs --base-patterns loss_CNN_cnn_size_-0p1 loss_CNN_cnn_size_-0p5 --trail 6 --usecase 2 --export-plots\n\n"
            "  # 4) Standard DU vs PIDU comparison\n"
            "  python -m research.analysis.analyse_outputs --base-patterns loss_DU loss_PIDU --trail 0 --usecase 1 --export-plots\n\n"
            "  # 5) Grouped parameter distributions on validation split\n"
            "  python -m research.analysis.analyse_outputs --base-patterns loss_DU loss_PIDU loss_CNN --ds-type test_val --distribution-mode grouped --export-plots\n\n"
            "  # 6) Split mode (one sheet/group per parameter)\n"
            "  python -m research.analysis.analyse_outputs --base-patterns loss_DU loss_PIDU --distribution-mode split --export-plots\n\n"
            "  # 7) Interactive plotting (no PNG export)\n"
            "  python -m research.analysis.analyse_outputs --base-patterns loss_DU loss_PIDU --show-plots\n\n"
            "  # 8) Validate all CNN/DU/PIDU indexed runs against saved last_unfold.xlsx\n"
            "  python -m research.analysis.analyse_outputs --validate --usecase 1\n\n"
            "  # 9) Validate with GPU inference\n"
            "  python -m research.analysis.analyse_outputs --validate --usecase 1 --device cuda:0\n"
        )
    )
    parser.add_argument('--base-patterns', nargs='+', default=['loss_lma_baseline', 'loss_lma_noisy', 'loss_lma_random', 'loss_CNN', 'loss_DU', 'loss_PIDU'], help="Loss folder prefixes. Supports indexed folders (<prefix>_<run_id>_<trail>) and single-run folders (<prefix>).")
    parser.add_argument('--trail', type=int, default=0, help="Trail number for file naming")
    parser.add_argument('--usecase', type=int, default=1, help="Use case number to determine folder structure")
    parser.add_argument('--num-runs', type=int, default=8, help="Number of runs to load and analyze")
    parser.add_argument('--show-plots', action='store_true', help="Whether to show plots interactively instead of saving to disk")
    parser.add_argument('--export-plots', action='store_true', help="Whether to export plots to disk instead of showing them interactively")
    parser.add_argument('--ds-type', type=str, default='test_test', help="Whether to analyze test_val dataset instead of test_test")
    parser.add_argument('--distribution-mode', choices=['grouped', 'split'], default='grouped',
                        help="Residual distribution mode: grouped (w0/wp/g) or split (w0_before, w0_1, ...)")
    parser.add_argument('--validate', action='store_true',
                        help="Re-run model inference (test set) and compare against saved last_unfold.xlsx "
                             "for all indexed CNN/DU/PIDU runs discovered under the case outputs folder.")
    parser.add_argument('--device', type=str, default='cpu',
                        help="PyTorch device for model inference during validation (default: cpu).")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.validate:
        # Auto-discover all indexed run folders matching loss_<Type>_<run>_<trail>
        _PROJECT_DIR = ROOT
        _outputs_dir = case_dir(args.usecase)
        _validate_patterns = ['loss_CNN', 'loss_DU', 'loss_PIDU']
        folders_to_validate: List[str] = []
        for _pattern in _validate_patterns:
            for _candidate in sorted(_outputs_dir.glob(f'{_pattern}_*')):
                if re.match(r'^loss_[A-Za-z]+_\d+_\d+$', _candidate.name):
                    try:
                        _model_path, _, _, _ = output_folder_to_model_path(_candidate.name, args.usecase)
                        if _model_path.exists():
                            folders_to_validate.append(_candidate.name)
                    except ValueError:
                        pass

        if not folders_to_validate:
            print(f"No indexed run folders found under {_outputs_dir} for patterns: {_validate_patterns}")
        else:
            print(f"Validating {len(folders_to_validate)} output folder(s) ...")
            print("=" * 80)
            for _folder in folders_to_validate:
                try:
                    validate_model_output(
                        output_folder_name=_folder,
                        usecase=args.usecase,
                        device=args.device,
                    )
                except Exception as _exc:
                    print(f"  Warning: validation failed for {_folder}: {_exc}")
                print("-" * 80)
    else:
        print("Analyzing test_test dataset...")
        results_test = main(
            loss_types=[item for item in args.base_patterns if isinstance(item, str) and item.strip()],
            usecase=args.usecase,
            trail=args.trail,
            num_runs=args.num_runs,
            dataset_type=args.ds_type,
            distribution_mode=args.distribution_mode,
            save_plots=args.export_plots,
            show_plots=args.show_plots
        )