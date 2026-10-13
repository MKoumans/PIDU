import os
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from pidu.config import usecase


def load_loss_curve(run_file_path: str) -> pd.DataFrame:
    """Load loss curve data from a run file.
    
    Args:
        run_file_path: Path to the run file (e.g., loss_DU_0_000.txt)
        
    Returns:
        DataFrame with columns: dataset, system, batch, epoch, loss, time
    """
    if not Path(run_file_path).exists():
        raise FileNotFoundError(f"Run file not found: {run_file_path}")
    
    # Read the CSV file, skipping the first line which contains metadata
    df = pd.read_csv(run_file_path, skiprows=13)
    return df


def _decode_hyperparam_value(raw_value: str) -> str:
    """Decode hyperparameter value token from filename for display.

    Current saver encodes minus signs as underscores for scientific notation
    (e.g., `1e_3` means `1e-3`). We only decode that specific numeric pattern.
    """
    if 'e_' in raw_value:
        return raw_value.replace('e_', 'e-')
    return raw_value


def extract_hyperparameter_info(filename: str) -> Dict[str, object]:
    """Extract model, hyperparameters, and run identifiers from run filename.

    Supported patterns include:
    - loss_PIDU_5_002.txt
    - loss_PIDU_mixed_precision_-on__lr_-1e_3__2_003.txt

    Parsing rules:
    - model is token right after `loss_`
    - hyperparameter pairs are split on `__`, and each pair uses `_-` as separator
    - numeric suffix tokens are read from the end and interpreted as run indices
      (`run_number` is always the last numeric token)
    """
    stem = Path(filename).stem
    info: Dict[str, object] = {
        'model': 'unknown',
        'hyperparameters': {},
        'numeric_suffix': [],
        'variant': None,
        'run_number': None,
        'config_key': '',
        'config_label': '',
    }

    if not stem.startswith('loss_'):
        return info

    parts = stem.split('_', 2)
    if len(parts) < 2:
        return info

    model = parts[1]
    remainder = parts[2] if len(parts) > 2 else ''
    info['model'] = model

    if not remainder:
        info['config_key'] = model
        info['config_label'] = model
        return info

    remainder_segments = remainder.split('__')

    numeric_suffix = []
    while remainder_segments:
        tail = remainder_segments[-1]
        tail_tokens = tail.split('_')
        if all(token.isdigit() for token in tail_tokens):
            numeric_suffix = tail_tokens + numeric_suffix
            remainder_segments.pop()
        else:
            break

    # Legacy names without a `__` separator (loss_PIDU_wgn_nostruct_0_021): trailing digit tokens of the
    # last segment are the run indices, the rest is the legacy config.
    if not numeric_suffix and remainder_segments and '_-' not in remainder_segments[-1]:
        tokens = remainder_segments[-1].split('_')
        n_digits = 0
        while n_digits < len(tokens) and tokens[-1 - n_digits].isdigit():
            n_digits += 1
        if n_digits:
            numeric_suffix = tokens[len(tokens) - n_digits:]
            if len(tokens) > n_digits:
                remainder_segments[-1] = '_'.join(tokens[:len(tokens) - n_digits])
            else:
                remainder_segments.pop()

    if numeric_suffix:
        info['numeric_suffix'] = numeric_suffix
        info['run_number'] = numeric_suffix[-1]
        if len(numeric_suffix) >= 2:
            info['variant'] = numeric_suffix[-2]

    hyperparameters: Dict[str, str] = {}
    for segment in remainder_segments:
        if '_-' in segment:
            name, raw_value = segment.split('_-', 1)
            if name:
                hyperparameters[name] = _decode_hyperparam_value(raw_value)

    if not hyperparameters and remainder_segments:
        legacy = '_'.join(remainder_segments)
        if legacy:
            hyperparameters['config'] = legacy

    info['hyperparameters'] = hyperparameters

    if hyperparameters:
        sorted_items = sorted(hyperparameters.items(), key=lambda item: item[0])
        config_key = '__'.join(f"{name}={value}" for name, value in sorted_items)
    else:
        config_key = 'default'

    info['config_key'] = f"{model}__{config_key}"
    info['config_label'] = f"{model} | {config_key}"

    return info


def aggregate_loss_by_epoch(
  df: pd.DataFrame, 
  dataset: str = 'train', 
  aggregation: str = 'mean'
) -> pd.DataFrame:
    """Aggregate loss values by epoch.
    
    Args:
        df: DataFrame with loss curve data
        dataset: Filter by dataset type ('train', 'val', 'test')
        aggregation: Aggregation method ('mean', 'min', 'max', 'median')
        
    Returns:
        DataFrame with columns: epoch, loss
    """
    filtered = df[df['dataset'] == dataset].copy()
    
    if aggregation == 'mean':
        agg_func = 'mean'
    elif aggregation == 'min':
        agg_func = 'min'
    elif aggregation == 'max':
        agg_func = 'max'
    elif aggregation == 'median':
        agg_func = 'median'
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")
    
    result = filtered.groupby('epoch')['loss'].agg(agg_func).reset_index()
    return result


def compare_loss_curves_by_hyperparameter(
    case_dir: str,
    trail: Optional[int] = None,
    hyperparameter_name: Optional[str] = None,
    dataset: str = 'train',
    aggregation: str = 'mean',
    use_log_scale: bool = True,
    show_variance: bool = True,
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (12, 8)
) -> Dict[str, Dict[str, np.ndarray]]:
    """Compare loss curves for different values of a specific hyperparameter.
    
    This function loads run files that test different values of the specified
    hyperparameter and plots their loss curves for comparison.
    
    Args:
        case_dir: Path to the case directory (e.g., 'outputs/case1')
        trail: Optional filter for the trailing run number in filename (last numeric token)
        hyperparameter_name: Optional hyperparameter name to filter on
        dataset: Which dataset to analyze ('train', 'val', 'test')
        aggregation: How to aggregate across systems/batches ('mean', 'min', 'max', 'median')
        use_log_scale: Whether to use logarithmic scale for y-axis
        show_variance: Whether to show ±std shading across runs
        save_path: Optional path to save the figure
        show: Whether to display the plot
        figsize: Figure size (width, height)
        
    Returns:
        Dictionary mapping hyperparameter values to their aggregated loss curves
    """
    runs_dir = Path(case_dir) / 'runs'
    
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")
    
    # Find all run files
    run_files = sorted(runs_dir.glob('*.txt'))
    run_files = [f for f in run_files if f.name.startswith('loss_')]
    
    if not run_files:
        raise FileNotFoundError(f"No run files found in {runs_dir}")
    
    # Load and process runs, grouped by full hyperparameter configuration
    grouped_curves: Dict[str, list] = {}
    
    for run_file in run_files:
        # Extract hyperparameter info from filename
        info = extract_hyperparameter_info(run_file.name)
        hyperparams = info.get('hyperparameters', {})
        run_number = info.get('run_number')
        
        # Optional hyperparameter filter
        if hyperparameter_name and hyperparameter_name not in hyperparams:
            continue
        
        # Optional run filter on trailing run number
        if trail is not None and run_number != f"{trail:03d}":
            continue
        
        # Load and aggregate loss curve
        print(
            f" - Processing: {run_file.name} | model={info.get('model')} "
            f"| run={run_number}"
        )
        try:
            df = load_loss_curve(str(run_file))
            agg_loss = aggregate_loss_by_epoch(df, dataset=dataset, aggregation=aggregation)
            agg_loss = agg_loss.to_numpy()

            label = str(info.get('config_label'))

            if label not in grouped_curves:
                grouped_curves[label] = []
            grouped_curves[label].append(agg_loss)
            
        except Exception as e:
            print(f"Warning: Could not process {run_file.name}: {e}")
            continue
    
    if not grouped_curves:
        print(f"No loss curves found for hyperparameter filter: {hyperparameter_name}")
        print("Available parsed configurations (first 5):")
        for run_file in run_files[:5]:
            info = extract_hyperparameter_info(run_file.name)
            print(f"  {run_file.name}: {info.get('config_label')}")
        return {}
    
    # Mean and std over run files for each configuration
    loss_curves: Dict[str, Dict[str, np.ndarray]] = {}
    for label, curves in grouped_curves.items():
        curves = np.array(curves)
        mean = np.mean(curves, axis=0)
        std = np.std(curves, axis=0)
        loss_curves[label] = {
            'epoch': mean[:, 0],
            'loss': mean[:, 1],
            'std': std[:, 1],
            'num_runs': np.array([curves.shape[0]]),
        }
    
    # Create comparison plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use a color palette with enough colors
    n_curves = len(loss_curves)
    colors = sns.color_palette("husl", n_curves)
    
    # Plot each loss curve
    for idx, (label, loss_data) in enumerate(sorted(loss_curves.items())):
        epochs = loss_data['epoch']
        loss_mean = loss_data['loss']
        loss_std = loss_data['std']
        
        # Get final epoch statistics
        final_mean = loss_mean[-1]
        final_std = loss_std[-1]
        
        # Create enhanced label with final statistics
        n_runs = int(loss_data['num_runs'][0])
        enhanced_label = f"{label} | runs={n_runs} | Final: {final_mean:.2e} ± {final_std:.2e}"
        
        # Plot mean line
        ax.plot(epochs, loss_mean, 
                label=enhanced_label, color=colors[idx], linewidth=2, alpha=0.8)
        
        # Plot standard deviation as shaded region
        if show_variance:
            lower = np.maximum(loss_mean - loss_std, 1e-20) if use_log_scale else (loss_mean - loss_std)
            upper = loss_mean + loss_std
            ax.fill_between(epochs, lower, upper, color=colors[idx], alpha=0.2)
    
    # Configure plot
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel(f'{aggregation.capitalize()} Loss', fontsize=12)
    
    if use_log_scale:
        ax.set_yscale('log')
        ax.set_ylabel(f'{aggregation.capitalize()} Loss (log scale)', fontsize=12)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=10)
    
    # Create title
    if hyperparameter_name:
        title = f'Loss Curves: {hyperparameter_name.replace("_", " ").title()}\n{dataset.capitalize()} Set (mean over runs)'
    else:
        title = f'Loss Curves: All Configurations\n{dataset.capitalize()} Set (mean over runs)'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return loss_curves


def compare_final_loss_values(
    case_dir: str,
    trail: Optional[int] = None,
    hyperparameter_name: Optional[str] = None,
    dataset: str = 'train',
    aggregation: str = 'mean',
    final_epochs: int = 10,
    save_path: Optional[str] = None,
    show: bool = True,
    export_xlsx: bool = False,
    figsize: Tuple[int, int] = (10, 6)
) -> pd.DataFrame:
    """Compare final loss values for different values of a specific hyperparameter.
    
    Creates a bar plot showing the final loss values (averaged over last N epochs)
    for different values of the specified hyperparameter.
    
    Args:
        case_dir: Path to the case directory
        hyperparameter_name: Name of the hyperparameter to compare
        dataset: Which dataset to analyze
        aggregation: How to aggregate across systems/batches
        final_epochs: Number of final epochs to average
        save_path: Optional path to save the figure
        show: Whether to display the plot
        figsize: Figure size
        
    Returns:
        DataFrame with final loss statistics for each hyperparameter value
    """
    runs_dir = Path(case_dir) / 'runs'
    run_files = sorted(runs_dir.glob('*.txt'))
    
    grouped_results: Dict[str, Dict[str, object]] = {}
    
    for run_file in run_files:
        info = extract_hyperparameter_info(run_file.name)
        hyperparams = info.get('hyperparameters', {})
        run_number = info.get('run_number')
        
        if not str(run_file.name).startswith('loss_'):
            continue

        # Optional hyperparameter filter
        if hyperparameter_name and hyperparameter_name not in hyperparams:
            continue

        # Optional run filter on trailing run number
        if trail is not None and run_number != f"{trail:03d}":
            continue
        
        try:
            df = load_loss_curve(str(run_file))
            agg_loss = aggregate_loss_by_epoch(df, dataset=dataset, aggregation=aggregation)
            
            # Get final loss (average of last N epochs)
            final_loss = agg_loss['loss'].tail(final_epochs).mean()
            min_loss = agg_loss['loss'].min()
            
            label = str(info.get('config_label'))

            if label not in grouped_results:
                grouped_results[label] = {
                    'value': label,
                    'model': info.get('model', 'unknown'),
                    'final_loss_values': [],
                    'min_loss_values': [],
                }

            grouped_results[label]['final_loss_values'].append(final_loss)
            grouped_results[label]['min_loss_values'].append(min_loss)
            
        except Exception as e:
            print(f"Warning: Could not process {run_file.name}: {e}")
            continue
    
    if not grouped_results:
        print(f"No valid results found for hyperparameter filter: {hyperparameter_name}")
        return pd.DataFrame()
    
    # Create DataFrame (mean/std over runs)
    results = []
    for label, values in grouped_results.items():
        final_loss_values = np.array(values['final_loss_values'])
        min_loss_values = np.array(values['min_loss_values'])
        results.append({
            'value': label,
            'model': values['model'],
            'final_loss': float(final_loss_values.mean()),
            'final_loss_std': float(final_loss_values.std()),
            'min_loss': float(min_loss_values.mean()),
            'num_runs': int(final_loss_values.shape[0]),
        })

    results_df = pd.DataFrame(results)
    if export_xlsx: 
        xlsx_path = save_path.replace('.png', '.xlsx') if save_path else f'final_loss_comparison_{hyperparameter_name}.xlsx'
        Path(xlsx_path).parent.mkdir(parents=True, exist_ok=True)
        results_df.to_excel(xlsx_path, index=False)
        print(f"Final loss statistics exported to: {xlsx_path}")
        
    results_df = results_df.sort_values('final_loss')
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=figsize)
    
    x_pos = np.arange(len(results_df))
    bars = ax.bar(x_pos, results_df['final_loss'], alpha=0.7)
    
    # Color bars by model
    models = results_df['model'].unique()
    colors = sns.color_palette("Set2", len(models))
    color_map = dict(zip(models, colors))
    
    for bar, model in zip(bars, results_df['model']):
        bar.set_color(color_map[model])
    
    ax.set_xlabel('Hyperparameter Value', fontsize=12)
    ax.set_ylabel(f'Final Loss (avg last {final_epochs} epochs)', fontsize=12)
    if hyperparameter_name:
        title = f'Final Loss: {hyperparameter_name.replace("_", " ").title()}\n{dataset.capitalize()} Set (mean over runs)'
    else:
        title = f'Final Loss: All Configurations\n{dataset.capitalize()} Set (mean over runs)'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(results_df['value'], rotation=45, ha='right', fontsize=10)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Add legend for models if there are multiple
    if len(models) > 1:
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color_map[m], label=m) for m in models]
        ax.legend(handles=legend_elements, title='Model', loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return results_df


def main():
    """Example usage of the hyperparameter analysis functions."""
    
    # Specify which hyperparameter to analyze
    hyperparameter_to_analyze = 'initialiser'  # Set to None to compare all configurations
    usecase = 2
    trail = 6  # Set int to filter a specific trailing run number
    
    # Example 1: Compare loss curves for different values of the hyperparameter
    print(f"Comparing loss curves for hyperparameter: {hyperparameter_to_analyze}...")
    
    loss_curves = compare_loss_curves_by_hyperparameter(
        case_dir=f'outputs/case{usecase}',
        trail=trail,
        hyperparameter_name=hyperparameter_to_analyze,
        dataset='test',
        aggregation='mean',
        use_log_scale=True,
        show_variance=True,
        save_path=f'outputs/case{usecase}/analyse/comparison/loss_curves_{hyperparameter_to_analyze}.png',
        show=False
    )
    print(f"Processed {len(loss_curves)} configurations\n")
    
    # Example 2: Compare final loss values
    print(f"Comparing final loss values for hyperparameter: {hyperparameter_to_analyze}...")
    final_losses = compare_final_loss_values(
        case_dir=f'outputs/case{usecase}',
        trail=trail,
        hyperparameter_name=hyperparameter_to_analyze,
        dataset='test',
        aggregation='mean',
        final_epochs=5,
        save_path=f'outputs/case{usecase}/analyse/comparison/final_loss_{hyperparameter_to_analyze}.png',
        show=False,
        export_xlsx=True
    )
    
    if not final_losses.empty:
        print(f"\nFinal Loss Rankings for {hyperparameter_to_analyze}:")
        print(final_losses[['value', 'final_loss', 'final_loss_std', 'min_loss', 'model', 'num_runs']].to_string(index=False))


if __name__ == '__main__':
    main()
