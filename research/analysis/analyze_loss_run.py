import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from pidu.paths import ROOT, case_dir
from os import path, makedirs
from typing import Optional
import argparse

def find_header_row(filepath):
    """Find the line index of the CSV header (the line starting with 'dataset,')."""
    with open(filepath) as f:
        for i, line in enumerate(f):
            if line.startswith('dataset,'):
                return i
    raise ValueError(f"No 'dataset,...' header line found in {filepath}")

def load_loss_data(filepath, skiprows:int|None=None):
    """Load loss data from CSV file."""
    if skiprows is None:
        skiprows = find_header_row(filepath)
    df = pd.read_csv(filepath, skiprows=skiprows, header=0)
    return df

def plot_loss_by_epoch(df, save_path=None):
    """Plot training, validation, and test loss across epochs."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Group by dataset type and epoch
    train_loss = df[df['dataset'] == 'train'].groupby('epoch')['loss'].mean()
    val_loss = df[df['dataset'] == 'val'].groupby('epoch')['loss'].mean()
    test_loss = df[df['dataset'] == 'test'].groupby('epoch')['loss'].mean()
    
    # Plot
    ax.plot(train_loss.index, train_loss.values, 'o-', label='Train Loss', linewidth=2)
    ax.plot(val_loss.index, val_loss.values, 's-', label='Validation Loss', linewidth=2)
    ax.plot(test_loss.index, test_loss.values, '^-', label='Test Loss', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Loss vs Epoch', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.show()
    return fig, ax

def plot_training_batches(df, epoch=None, save_path=None):
    """Plot training loss for each batch within epochs."""
    train_df = df[df['dataset'] == 'train'].copy()
    
    if epoch is not None:
        train_df = train_df[train_df['epoch'] == epoch]
        title = f'Training Loss per Batch - Epoch {epoch}'
    else:
        title = 'Training Loss per Batch - All Epochs'
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    if epoch is None:
        # Color by epoch
        epochs = train_df['epoch'].unique()
        for ep in epochs:
            epoch_data = train_df[train_df['epoch'] == ep]
            ax.plot(range(len(epoch_data)), epoch_data['loss'].values, 
                   'o-', alpha=0.7, label=f'Epoch {ep}', markersize=3)
    else:
        ax.plot(train_df['batch'].values, train_df['loss'].values, 'o-', linewidth=2)
    
    ax.set_xlabel('Batch Index', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    if epoch is None:
        ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.show()
    return fig, ax

def plot_loss_log_scale(df, save_path=None):
    """Plot loss in log scale to better visualize convergence."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Group by dataset type and epoch
    train_loss = df[df['dataset'] == 'train'].groupby('epoch')['loss'].mean()
    val_loss = df[df['dataset'] == 'val'].groupby('epoch')['loss'].mean()
    test_loss = df[df['dataset'] == 'test'].groupby('epoch')['loss'].mean()
    
    # Plot with log scale
    ax.semilogy(train_loss.index, train_loss.values, 'o-', label='Train Loss', linewidth=2)
    ax.semilogy(val_loss.index, val_loss.values, 's-', label='Validation Loss', linewidth=2)
    ax.semilogy(test_loss.index, test_loss.values, '^-', label='Test Loss', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (log scale)', fontsize=12)
    ax.set_title('Loss vs Epoch (Log Scale)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_ylim(1e1, 1e6)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.show()
    return fig, ax

def print_statistics(df):
    """Print summary statistics for the loss data."""
    print("\n" + "="*60)
    print("LOSS STATISTICS SUMMARY")
    print("="*60)
    
    for dataset_type in ['train', 'val', 'test']:
        subset = df[df['dataset'] == dataset_type]
        if len(subset) > 0:
            print(f"\n{dataset_type.upper()} Dataset:")
            print(f"  Number of entries: {len(subset)}")
            print(f"  Initial loss: {subset[subset['epoch'] == subset['epoch'].min()]['loss'].mean():.4f}")
            print(f"  Final loss: {subset[subset['epoch'] == subset['epoch'].max()]['loss'].mean():.4f}")
            print(f"  Minimum loss: {subset['loss'].min():.4f} (Epoch {subset.loc[subset['loss'].idxmin(), 'epoch']})")
            print(f"  Maximum loss: {subset['loss'].max():.4f} (Epoch {subset.loc[subset['loss'].idxmax(), 'epoch']})")
            print(f"  Mean loss: {subset['loss'].mean():.4f}")
            print(f"  Std loss: {subset['loss'].std():.4f}")
    
    # Loss improvement
    train_df = df[df['dataset'] == 'train']
    if len(train_df) > 0:
        initial_loss = train_df[train_df['epoch'] == train_df['epoch'].min()]['loss'].mean()
        final_loss = train_df[train_df['epoch'] == train_df['epoch'].max()]['loss'].mean()
        improvement = (initial_loss - final_loss) / initial_loss * 100
        print(f"\n{'='*60}")
        print(f"Training improvement: {improvement:.2f}% reduction in loss")
        print(f"{'='*60}\n")

def analyze_convergence(df):
    """Analyze convergence by looking at loss change rate."""
    train_df = df[df['dataset'] == 'train'].groupby('epoch')['loss'].mean()
    
    if len(train_df) > 1:
        loss_changes = train_df.diff().abs()
        
        print("\nCONVERGENCE ANALYSIS:")
        print(f"  Average loss change per epoch: {loss_changes.mean():.4f}")
        print(f"  Last 3 epochs average change: {loss_changes.tail(3).mean():.4f}")
        
        # Check if converging
        recent_change = loss_changes.tail(3).mean()
        if recent_change < train_df.mean() * 0.01:  # Less than 1% of mean loss
            print("  Status: Model appears to be converging ✓")
        else:
            print("  Status: Model still improving significantly")

def analyze_timing(df):
    """Analyze timing statistics from the training run."""
    print("\n" + "="*60)
    print("TIMING ANALYSIS")
    print("="*60)
    
    # Calculate time differences for training batches
    train_df = df[df['dataset'] == 'train'].copy()
    train_df['time_diff'] = train_df['time'].diff()
    
    # Remove negative diffs (epoch boundaries) and outliers
    train_df_clean = train_df[train_df['time_diff'] > 0]
    
    if len(train_df_clean) > 0:
        print(f"\nTraining Batch Timing:")
        print(f"  Mean time per batch: {train_df_clean['time_diff'].mean():.3f} seconds")
        print(f"  Median time per batch: {train_df_clean['time_diff'].median():.3f} seconds")
        print(f"  Min time per batch: {train_df_clean['time_diff'].min():.3f} seconds")
        print(f"  Max time per batch: {train_df_clean['time_diff'].max():.3f} seconds")
        print(f"  Std time per batch: {train_df_clean['time_diff'].std():.3f} seconds")
    
    # Epoch timing
    epoch_times = []
    for epoch in df['epoch'].unique():
        epoch_data = df[df['epoch'] == epoch]
        if len(epoch_data) > 1:
            epoch_duration = epoch_data['time'].max() - epoch_data['time'].min()
            epoch_times.append(epoch_duration)
    
    if epoch_times:
        print(f"\nEpoch Timing:")
        print(f"  Mean time per epoch: {np.mean(epoch_times):.2f} seconds ({np.mean(epoch_times)/60:.2f} minutes)")
        print(f"  Median time per epoch: {np.median(epoch_times):.2f} seconds")
        print(f"  Total training time: {sum(epoch_times):.2f} seconds ({sum(epoch_times)/60:.2f} minutes)")
    
    # Dataset-specific timing
    for dataset_type in ['train', 'val', 'test']:
        subset = df[df['dataset'] == dataset_type].copy()
        if len(subset) > 1:
            subset['time_diff'] = subset['time'].diff()
            subset_clean = subset[subset['time_diff'] > 0]
            if len(subset_clean) > 0:
                print(f"\n{dataset_type.upper()} Dataset:")
                print(f"  Mean time per batch: {subset_clean['time_diff'].mean():.3f} seconds")
                print(f"  Total batches: {len(subset)}")

def plot_timing_analysis(df, save_path=None):
    """Create comprehensive timing visualization."""
    fig = plt.figure(figsize=(16, 10))
    gs = plt.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Time per batch over training
    ax1 = fig.add_subplot(gs[0, :])
    train_df = df[df['dataset'] == 'train'].copy()
    train_df['time_diff'] = train_df['time'].diff()
    train_df_clean = train_df[train_df['time_diff'] > 0]
    
    ax1.plot(range(len(train_df_clean)), train_df_clean['time_diff'].values, 
             'o-', alpha=0.6, markersize=3, linewidth=1)
    ax1.axhline(train_df_clean['time_diff'].mean(), color='r', linestyle='--', 
                linewidth=2, label=f'Mean: {train_df_clean["time_diff"].mean():.3f}s')
    ax1.set_xlabel('Batch Index', fontsize=11)
    ax1.set_ylabel('Time per Batch (seconds)', fontsize=11)
    ax1.set_title('Training Time per Batch', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. Distribution of batch times
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.hist(train_df_clean['time_diff'].values, bins=30, edgecolor='black', alpha=0.7)
    ax2.axvline(train_df_clean['time_diff'].mean(), color='r', linestyle='--', 
                linewidth=2, label=f'Mean: {train_df_clean["time_diff"].mean():.3f}s')
    ax2.axvline(train_df_clean['time_diff'].median(), color='g', linestyle='--', 
                linewidth=2, label=f'Median: {train_df_clean["time_diff"].median():.3f}s')
    ax2.set_xlabel('Time per Batch (seconds)', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Distribution of Batch Times', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Time per epoch
    ax3 = fig.add_subplot(gs[1, 1])
    epoch_times = []
    epochs = []
    for epoch in sorted(df['epoch'].unique()):
        epoch_data = df[df['epoch'] == epoch]
        if len(epoch_data) > 1:
            epoch_duration = epoch_data['time'].max() - epoch_data['time'].min()
            epoch_times.append(epoch_duration)
            epochs.append(epoch)
    
    ax3.bar(epochs, epoch_times, alpha=0.7, edgecolor='black')
    ax3.axhline(np.mean(epoch_times), color='r', linestyle='--', 
                linewidth=2, label=f'Mean: {np.mean(epoch_times):.2f}s')
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('Time (seconds)', fontsize=11)
    ax3.set_title('Time per Epoch', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Cumulative time
    ax4 = fig.add_subplot(gs[2, 0])
    cumulative_time = np.cumsum(epoch_times)
    ax4.plot(epochs, cumulative_time / 60, 'o-', linewidth=2, markersize=6)
    ax4.set_xlabel('Epoch', fontsize=11)
    ax4.set_ylabel('Cumulative Time (minutes)', fontsize=11)
    ax4.set_title('Cumulative Training Time', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 5. Timing by dataset type
    ax5 = fig.add_subplot(gs[2, 1])
    timing_by_type = {}
    for dataset_type in ['train', 'val', 'test']:
        subset = df[df['dataset'] == dataset_type].copy()
        if len(subset) > 1:
            subset['time_diff'] = subset['time'].diff()
            subset_clean = subset[subset['time_diff'] > 0]
            if len(subset_clean) > 0:
                timing_by_type[dataset_type] = subset_clean['time_diff'].mean()
    
    if timing_by_type:
        colors = {'train': '#1f77b4', 'val': '#ff7f0e', 'test': '#2ca02c'}
        dataset_types = list(timing_by_type.keys())
        times = list(timing_by_type.values())
        bars = ax5.bar(dataset_types, times, alpha=0.7, edgecolor='black',
                      color=[colors.get(dt, '#gray') for dt in dataset_types])
        ax5.set_ylabel('Mean Time per Batch (seconds)', fontsize=11)
        ax5.set_title('Average Batch Time by Dataset Type', fontsize=13, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, time in zip(bars, times):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{time:.3f}s', ha='center', va='bottom', fontsize=10)
    
    plt.suptitle('Training Timing Analysis', fontsize=16, fontweight='bold', y=0.995)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved timing plot to {save_path}")
    
    plt.show()
    return fig

def create_timing_infographic(df, save_path=None):
    """Create an infographic-style summary of timing statistics."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # Calculate key metrics
    train_df = df[df['dataset'] == 'train'].copy()
    train_df['time_diff'] = train_df['time'].diff()
    train_df_clean = train_df[train_df['time_diff'] > 0]
    
    epoch_times = []
    for epoch in df['epoch'].unique():
        epoch_data = df[df['epoch'] == epoch]
        if len(epoch_data) > 1:
            epoch_duration = epoch_data['time'].max() - epoch_data['time'].min()
            epoch_times.append(epoch_duration)
    
    total_time = sum(epoch_times)
    total_batches = len(train_df)
    avg_batch_time = train_df_clean['time_diff'].mean()
    avg_epoch_time = np.mean(epoch_times)
    num_epochs = len(epoch_times)
    
    # Create infographic layout
    title_y = 0.95
    fig.text(0.5, title_y, 'Training Performance Summary', 
             ha='center', fontsize=24, fontweight='bold')
    
    # Box style
    box_props = dict(boxstyle='round,pad=0.8', facecolor='lightblue', alpha=0.3, edgecolor='black', linewidth=2)
    box_props_highlight = dict(boxstyle='round,pad=0.8', facecolor='lightgreen', alpha=0.4, edgecolor='black', linewidth=2)
    
    # Total time (top center, highlighted)
    y_pos = 0.75
    fig.text(0.5, y_pos, f'{total_time/60:.1f} minutes', 
             ha='center', fontsize=32, fontweight='bold', bbox=box_props_highlight)
    fig.text(0.5, y_pos - 0.08, 'Total Training Time', 
             ha='center', fontsize=14)
    
    # Row 1: Key metrics
    y_pos = 0.55
    metrics = [
        (0.2, f'{num_epochs}', 'Total Epochs'),
        (0.5, f'{total_batches}', 'Total Batches'),
        (0.8, f'{avg_batch_time:.3f}s', 'Avg Batch Time')
    ]
    
    for x, value, label in metrics:
        fig.text(x, y_pos, value, ha='center', fontsize=24, fontweight='bold', bbox=box_props)
        fig.text(x, y_pos - 0.08, label, ha='center', fontsize=12)
    
    # Row 2: More detailed metrics
    y_pos = 0.32
    metrics2 = [
        (0.2, f'{avg_epoch_time:.1f}s', 'Avg Epoch Time'),
        (0.5, f'{total_time/total_batches:.3f}s', 'Time per Batch\n(overall)'),
        (0.8, f'{60/avg_batch_time:.1f}', 'Batches per\nMinute')
    ]
    
    for x, value, label in metrics2:
        fig.text(x, y_pos, value, ha='center', fontsize=20, fontweight='bold', bbox=box_props)
        fig.text(x, y_pos - 0.08, label, ha='center', fontsize=11)
    
    # Bottom: Performance indicator
    y_pos = 0.08
    throughput = total_batches / (total_time / 3600)  # batches per hour
    fig.text(0.5, y_pos, f'Throughput: {throughput:.1f} batches/hour', 
             ha='center', fontsize=16, style='italic')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved timing infographic to {save_path}")
    
    plt.show()
    return fig

def load_multiple_runs(base_pattern: str, trail: int = 0, num_runs: int = 8, usecase: int = 1):
    """Load loss data from multiple runs and combine them."""
    all_dfs = []
    runs_folder = case_dir(usecase) / "runs" # old
    # runs_folder = case_dir(usecase) / "loss"  # new
    
    for run_idx in range(num_runs):
        filename = base_pattern + f"_{run_idx}_{trail:03d}.txt"
        filepath = runs_folder / filename
        
        if filepath.exists():
            print(f"Loading run {run_idx}: {filename}")
            df = load_loss_data(filepath)
            df['run_id'] = run_idx  # Add run identifier
            all_dfs.append(df)
        else:
            print(f"Warning: File not found: {filepath}")
    
    if not all_dfs:
        raise FileNotFoundError(f"No run files found matching pattern: {base_pattern}")
    
    # Combine all runs
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nLoaded {len(all_dfs)} runs with {len(combined_df)} total records")
    
    return combined_df, all_dfs

def compute_average_statistics(df):
    """Compute average loss across runs for each epoch and dataset type."""
    # Group by dataset, epoch, and batch, then average across runs
    avg_by_epoch = df.groupby(['dataset', 'epoch'])['loss'].agg(['mean', 'std', 'min', 'max']).reset_index()
    avg_by_epoch.columns = ['dataset', 'epoch', 'loss_mean', 'loss_std', 'loss_min', 'loss_max']
    
    return avg_by_epoch

def plot_loss_by_epoch_with_runs(df, save_path=None):
    """Plot training, validation, and test loss across epochs with individual runs and averages."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    dataset_types = ['train', 'val', 'test']
    colors = {'train': 'blue', 'val': 'orange', 'test': 'green'}
    
    for idx, dataset_type in enumerate(dataset_types):
        ax = axes[idx]
        subset = df[df['dataset'] == dataset_type]
        
        # Plot individual runs with transparency
        for run_id in sorted(subset['run_id'].unique()):
            run_data = subset[subset['run_id'] == run_id].groupby('epoch')['loss'].mean()
            ax.plot(run_data.index, run_data.values, 'o-', alpha=0.3, 
                   linewidth=1, markersize=3, color=colors[dataset_type])
        
        # Plot average across runs
        avg_data = subset.groupby('epoch')['loss'].mean()
        std_data = subset.groupby('epoch')['loss'].std()
        
        ax.plot(avg_data.index, avg_data.values, 'o-', linewidth=3, markersize=6,
               color=colors[dataset_type], label='Average', zorder=10)
        
        # Add standard deviation band
        ax.fill_between(avg_data.index, 
                        avg_data.values - std_data.values,
                        avg_data.values + std_data.values,
                        alpha=0.2, color=colors[dataset_type])
        
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Loss', fontsize=11)
        ax.set_title(f'{dataset_type.upper()} Loss', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Loss vs Epoch (Individual Runs + Average)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.show()
    return fig, axes

def plot_loss_by_epoch_average_only(df, save_path=None):
    """Plot only the average training, validation, and test loss across epochs."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = {'train': 'blue', 'val': 'orange', 'test': 'green'}
    markers = {'train': 'o', 'val': 's', 'test': '^'}
    
    for dataset_type in ['train', 'val', 'test']:
        subset = df[df['dataset'] == dataset_type]
        avg_data = subset.groupby('epoch')['loss'].mean()
        std_data = subset.groupby('epoch')['loss'].std()
        
        ax.plot(avg_data.index, avg_data.values, marker=markers[dataset_type], 
               linewidth=2, markersize=8, color=colors[dataset_type],
               label=f'{dataset_type.capitalize()} (avg ± std)')
        
        # Add standard deviation band
        ax.fill_between(avg_data.index, 
                        avg_data.values - std_data.values,
                        avg_data.values + std_data.values,
                        alpha=0.2, color=colors[dataset_type])
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Average Loss vs Epoch (across all runs)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.show()
    return fig, ax

def plot_loss_log_scale_with_runs(df, save_path=None):
    """Plot loss in log scale with individual runs and averages."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = {'train': 'blue', 'val': 'orange', 'test': 'green'}
    markers = {'train': 'o', 'val': 's', 'test': '^'}
    
    for dataset_type in ['train', 'val', 'test']:
        subset = df[df['dataset'] == dataset_type]
        
        # Plot individual runs with transparency
        for run_id in sorted(subset['run_id'].unique()):
            run_data = subset[subset['run_id'] == run_id].groupby('epoch')['loss'].mean()
            ax.semilogy(run_data.index, run_data.values, alpha=0.2, 
                       linewidth=1, color=colors[dataset_type])
        
        # Plot average
        avg_data = subset.groupby('epoch')['loss'].mean()
        ax.semilogy(avg_data.index, avg_data.values, marker=markers[dataset_type],
                   linewidth=3, markersize=6, color=colors[dataset_type],
                   label=f'{dataset_type.capitalize()} (average)', zorder=10)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (log scale)', fontsize=12)
    ax.set_title('Loss vs Epoch (Log Scale) - All Runs', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_ylim(1e1, 1e6)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.show()
    return fig, ax

def plot_model_runs_log_scale(
    df: pd.DataFrame,
    dataset_type: str = 'test',
    model_name: Optional[str] = None,
    save_path: Optional[Path] = None
):
    """Plot all runs for a specific model on log scale with clear run-number identification."""
    if 'run_id' not in df.columns:
        raise ValueError("Dataframe must contain 'run_id' column for multi-run plotting")

    subset = df[df['dataset'] == dataset_type].copy()
    if len(subset) == 0:
        raise ValueError(f"No data found for dataset type: {dataset_type}")

    run_ids = sorted(subset['run_id'].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(run_ids)))

    fig, ax = plt.subplots(figsize=(12, 7))

    for idx, run_id in enumerate(run_ids):
        run_data = (
            subset[subset['run_id'] == run_id]
            .groupby('epoch')['loss']
            .mean()
            .sort_index()
        )

        if len(run_data) == 0:
            continue

        ax.semilogy(
            run_data.index,
            run_data.values,
            marker='o',
            linewidth=2,
            markersize=4,
            color=colors[idx],
            label=f'Run {run_id}'
        )

        # Number annotation at the last point for quick visual identification
        x_last = run_data.index[-1]
        y_last = run_data.values[-1]
        ax.annotate(
            f'{run_id}',
            xy=(x_last, y_last),
            xytext=(6, 0),
            textcoords='offset points',
            color=colors[idx],
            fontsize=9,
            fontweight='bold',
            va='center'
        )

    # Keep the log axis in a readable range regardless of outlier runs
    ax.set_ylim(1e1, 1e6)

    title_model = model_name if model_name else 'Model'
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel(f'{dataset_type.capitalize()} Loss (log scale)', fontsize=12)
    ax.set_title(f'{title_model}: {dataset_type.upper()} Loss per Run (Log Scale)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, ncol=2, title='Run Number')
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved per-run log-scale plot to {save_path}")

    plt.show()
    return fig, ax

def print_statistics_multi_run(df):
    """Print summary statistics for multiple runs."""
    print("\n" + "="*60)
    print("LOSS STATISTICS SUMMARY (ACROSS ALL RUNS)")
    print("="*60)
    
    num_runs = df['run_id'].nunique()
    print(f"\nNumber of runs analyzed: {num_runs}")
    
    for dataset_type in ['train', 'val', 'test']:
        subset = df[df['dataset'] == dataset_type]
        if len(subset) > 0:
            print(f"\n{dataset_type.upper()} Dataset:")
            
            # Statistics per epoch averaged across runs
            initial_losses = subset[subset['epoch'] == subset['epoch'].min()].groupby('run_id')['loss'].mean()
            final_losses = subset[subset['epoch'] == subset['epoch'].max()].groupby('run_id')['loss'].mean()
            
            print(f"  Initial loss (avg across runs): {initial_losses.mean():.4f} ± {initial_losses.std():.4f}")
            print(f"  Final loss (avg across runs): {final_losses.mean():.4f} ± {final_losses.std():.4f}")
            
            # Overall statistics
            print(f"  Minimum loss observed: {subset['loss'].min():.4f}")
            print(f"  Maximum loss observed: {subset['loss'].max():.4f}")
            print(f"  Mean loss (all data): {subset['loss'].mean():.4f}")
            print(f"  Std loss (all data): {subset['loss'].std():.4f}")
            
            # Improvement per run
            improvements = ((initial_losses - final_losses) / initial_losses * 100)
            print(f"  Improvement per run: {improvements.mean():.2f}% ± {improvements.std():.2f}%")
    
    # Overall training improvement
    train_df = df[df['dataset'] == 'train']
    if len(train_df) > 0:
        initial_losses = train_df[train_df['epoch'] == train_df['epoch'].min()].groupby('run_id')['loss'].mean()
        final_losses = train_df[train_df['epoch'] == train_df['epoch'].max()].groupby('run_id')['loss'].mean()
        avg_improvement = ((initial_losses - final_losses) / initial_losses * 100).mean()
        
        print(f"\n{'='*60}")
        print(f"Average training improvement: {avg_improvement:.2f}% reduction in loss")
        print(f"{'='*60}\n")

def compute_comparison_statistics(df1, df2, name1="Run 1", name2="Run 2"):
    """Compute comparison statistics between two sets of runs."""
    stats = {}
    
    for dataset_type in ['train', 'val', 'test']:
        subset1 = df1[df1['dataset'] == dataset_type]
        subset2 = df2[df2['dataset'] == dataset_type]
        
        if len(subset1) > 0 and len(subset2) > 0:
            # Compute per-epoch averages across runs
            avg1 = subset1.groupby('epoch')['loss'].mean()
            avg2 = subset2.groupby('epoch')['loss'].mean()
            
            # Compute final epoch statistics
            final_epoch = min(subset1['epoch'].max(), subset2['epoch'].max())
            final1 = subset1[subset1['epoch'] == final_epoch]['loss']
            final2 = subset2[subset2['epoch'] == final_epoch]['loss']
            
            stats[dataset_type] = {
                'mean1': final1.mean(),
                'mean2': final2.mean(),
                'var1': final1.var(),
                'var2': final2.var(),
                'std1': final1.std(),
                'std2': final2.std(),
                'min1': final1.min(),
                'min2': final2.min(),
                'max1': final1.max(),
                'max2': final2.max(),
                'overall_mean1': subset1['loss'].mean(),
                'overall_mean2': subset2['loss'].mean(),
                'overall_var1': subset1['loss'].var(),
                'overall_var2': subset2['loss'].var(),
            }
    
    return stats

def print_comparison_statistics(stats, name1="Run 1", name2="Run 2"):
    """Print comparison statistics in a formatted table."""
    print("\n" + "="*80)
    print(f"COMPARISON: {name1} vs {name2}")
    print("="*80)
    
    for dataset_type in ['train', 'val', 'test']:
        if dataset_type in stats:
            s = stats[dataset_type]
            print(f"\n{dataset_type.upper()} Dataset (Final Epoch):")
            print("-" * 80)
            print(f"{'Metric':<25} {name1:>20} {name2:>20} {'Difference':>12}")
            print("-" * 80)
            
            # Mean comparison
            mean_diff = s['mean2'] - s['mean1']
            mean_pct = (mean_diff / s['mean1'] * 100) if s['mean1'] != 0 else 0
            print(f"{'Mean Loss':<25} {s['mean1']:>20.6f} {s['mean2']:>20.6f} {mean_pct:>11.2f}%")
            
            # Variance comparison
            var_diff = s['var2'] - s['var1']
            var_pct = (var_diff / s['var1'] * 100) if s['var1'] != 0 else 0
            print(f"{'Variance':<25} {s['var1']:>20.6f} {s['var2']:>20.6f} {var_pct:>11.2f}%")
            
            # Std comparison
            std_diff = s['std2'] - s['std1']
            std_pct = (std_diff / s['std1'] * 100) if s['std1'] != 0 else 0
            print(f"{'Std Deviation':<25} {s['std1']:>20.6f} {s['std2']:>20.6f} {std_pct:>11.2f}%")
            
            # Min/Max comparison
            print(f"{'Min Loss':<25} {s['min1']:>20.6f} {s['min2']:>20.6f} {(s['min2']-s['min1']):>12.6f}")
            print(f"{'Max Loss':<25} {s['max1']:>20.6f} {s['max2']:>20.6f} {(s['max2']-s['max1']):>12.6f}")
            
            print(f"\n{dataset_type.upper()} Dataset (Overall):")
            print("-" * 80)
            overall_mean_diff = s['overall_mean2'] - s['overall_mean1']
            overall_mean_pct = (overall_mean_diff / s['overall_mean1'] * 100) if s['overall_mean1'] != 0 else 0
            print(f"{'Overall Mean Loss':<25} {s['overall_mean1']:>20.6f} {s['overall_mean2']:>20.6f} {overall_mean_pct:>11.2f}%")
            
            overall_var_diff = s['overall_var2'] - s['overall_var1']
            overall_var_pct = (overall_var_diff / s['overall_var1'] * 100) if s['overall_var1'] != 0 else 0
            print(f"{'Overall Variance':<25} {s['overall_var1']:>20.6f} {s['overall_var2']:>20.6f} {overall_var_pct:>11.2f}%")
    
    print("\n" + "="*80)

def plot_comparison(df1, df2, name1="Run 1", name2="Run 2", save_path=None):
    """Create comparison plots between two sets of runs."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    dataset_types = ['train', 'val', 'test']
    colors1 = {'train': 'blue', 'val': 'orange', 'test': 'green'}
    colors2 = {'train': 'darkblue', 'val': 'darkorange', 'test': 'darkgreen'}
    
    # Row 1: Normal scale
    for idx, dataset_type in enumerate(dataset_types):
        ax = axes[0, idx]
        
        # Plot first run set
        subset1 = df1[df1['dataset'] == dataset_type]
        avg1 = subset1.groupby('epoch')['loss'].mean()
        std1 = subset1.groupby('epoch')['loss'].std()
        
        ax.plot(avg1.index, avg1.values, 'o-', linewidth=2, markersize=6,
               color=colors1[dataset_type], label=name1, zorder=10)
        ax.fill_between(avg1.index, avg1.values - std1.values, avg1.values + std1.values,
                        alpha=0.2, color=colors1[dataset_type])
        
        # Plot second run set
        subset2 = df2[df2['dataset'] == dataset_type]
        avg2 = subset2.groupby('epoch')['loss'].mean()
        std2 = subset2.groupby('epoch')['loss'].std()
        
        ax.plot(avg2.index, avg2.values, 's-', linewidth=2, markersize=6,
               color=colors2[dataset_type], label=name2, zorder=10)
        ax.fill_between(avg2.index, avg2.values - std2.values, avg2.values + std2.values,
                        alpha=0.2, color=colors2[dataset_type])
        
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Loss', fontsize=11)
        ax.set_title(f'{dataset_type.upper()} Loss Comparison', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # Row 2: Log scale
    for idx, dataset_type in enumerate(dataset_types):
        ax = axes[1, idx]
        
        # Plot first run set
        subset1 = df1[df1['dataset'] == dataset_type]
        avg1 = subset1.groupby('epoch')['loss'].mean()
        std1 = subset1.groupby('epoch')['loss'].std()
        
        ax.semilogy(avg1.index, avg1.values, 'o-', linewidth=2, markersize=6,
                   color=colors1[dataset_type], label=name1, zorder=10)
        
        # Plot second run set
        subset2 = df2[df2['dataset'] == dataset_type]
        avg2 = subset2.groupby('epoch')['loss'].mean()
        std2 = subset2.groupby('epoch')['loss'].std()
        
        ax.semilogy(avg2.index, avg2.values, 's-', linewidth=2, markersize=6,
                   color=colors2[dataset_type], label=name2, zorder=10)
        
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Loss (log scale)', fontsize=11)
        ax.set_title(f'{dataset_type.upper()} Loss (Log)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, which='both')
        ax.set_ylim(1e1, 1e6)
    
    plt.suptitle(f'Comparison: {name1} vs {name2}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot to {save_path}")
    
    plt.show()
    return fig, axes

# Fixed color per method so identity is stable across figures
METHOD_COLORS = {
    'CNN': '#35A2A2',   # teal
    'DU': '#E3B33F',    # gold
    'PIDU': '#4C7FBE',  # blue
}
_FALLBACK_COLORS = ['#8C6BB1', '#C05B4D', '#5B8C5A', '#7F7F7F']

def _method_color(method_name: str, fallback_idx: int) -> str:
    # Longest key first so e.g. 'PIDU' matches before its substring 'DU'
    for key in sorted(METHOD_COLORS, key=len, reverse=True):
        if key.lower() in method_name.lower():
            return METHOD_COLORS[key]
    return _FALLBACK_COLORS[fallback_idx % len(_FALLBACK_COLORS)]

def _epoch_mean_std(df: pd.DataFrame, dstype: str):
    """Mean and std of loss per epoch across runs for one dataset type."""
    subset = df[df['dataset'] == dstype]
    if len(subset) == 0:
        return None, None, None
    grouped = subset.groupby('epoch')['loss']
    mean = grouped.mean().sort_index()
    std = grouped.std().sort_index().fillna(0.0)
    return mean.index.values, mean.values, std.values

def plot_loss_log_comparison(
  dfs: dict[str, pd.DataFrame],
  method_names: list[str],
  test_dataset: str = 'test',
  save_path: Optional[Path] = None
):
    """
    Two-panel log-scale loss comparison across methods.

    Left panel (a): train (filled markers, solid) and val (open markers) loss.
    Right panel (b): test loss (experimental dataset if available) with a
    ±1 std band across runs. Panels share the log y-axis.

    Args:
        dfs: Dictionary mapping method name to combined dataframe
        method_names: List of method names to display
        test_dataset: Dataset type shown in the right panel ('test' or 'exp')
        save_path: Optional path to save the plot
    """
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(11, 4.5), sharey=True,
        gridspec_kw={'wspace': 0.04}
    )

    for idx, method_name in enumerate(method_names):
        if method_name not in dfs:
            print(f"Warning: {method_name} not found in dataframes")
            continue

        df = dfs[method_name]
        color = _method_color(method_name, idx)

        # Left panel: train (filled) and val (open) curves
        for dstype, filled in [('train', True), ('val', False)]:
            epochs, mean, std = _epoch_mean_std(df, dstype)
            if epochs is None:
                print(f"Warning: no {dstype} data for {method_name}")
                continue
            marker_kw = dict(marker='o', markersize=4, markevery=max(1, len(epochs) // 20))
            if not filled:
                marker_kw.update(markerfacecolor='white', markeredgecolor=color)
            ax_a.semilogy(
                epochs, mean, linewidth=1.8, color=color,
                label=f'{method_name} ({dstype})', zorder=10, **marker_kw
            )
            lower = np.clip(mean - std, a_min=np.finfo(float).tiny, a_max=None)
            ax_a.fill_between(epochs, lower, mean + std, alpha=0.15,
                              color=color, linewidth=0)

        # Right panel: test / experimental loss with std band
        epochs, mean, std = _epoch_mean_std(df, test_dataset)
        if epochs is None:
            print(f"Warning: no {test_dataset} data for {method_name}")
            continue
        ax_b.semilogy(
            epochs, mean, linewidth=1.8, color=color,
            marker='s', markersize=4, markevery=max(1, len(epochs) // 20),
            label=f'{method_name} (test)', zorder=10
        )
        lower = np.clip(mean - std, a_min=np.finfo(float).tiny, a_max=None)
        ax_b.fill_between(epochs, lower, mean + std, alpha=0.15,
                          color=color, linewidth=0)

    # Std bands clipped to tiny positive values drag the log autoscale
    # down to ~1e-280; keep the axis in a readable range instead.
    ax_a.set_ylim(1e1, 1e6)

    for ax, panel in [(ax_a, '(a)'), (ax_b, '(b)')]:
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, which='both')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.text(0.97, 0.97, panel, transform=ax.transAxes,
                ha='right', va='top', fontsize=12)
        ax.legend(fontsize=9, loc='upper center', ncol=1, frameon=True,
                  framealpha=0.9, edgecolor='0.8')
        ax.tick_params(direction='in', which='both')
        ax.margins(x=0.02)

    ax_a.set_ylabel('Normalized Loss', fontsize=12)
    ax_b.tick_params(labelleft=False)

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved loss comparison plot to {save_path}")

    plt.show()
    return fig, (ax_a, ax_b)

def create_table(dfs: pd.DataFrame, method_name: str):
    """
    Extract mean and std loss per epoch, averaged over all runs.

    Args:
        dfs: Combined dataframe with columns: epoch, dataset, run_id, loss
        method_name: Name of the method (for column naming)

    Returns:
        Tuple of (train_dict, val_dict, test_dict, exp_dict) where each dict has:
        - f'{method_name} mean ({dataset})': mean loss per epoch (averaged over runs)
        - f'{method_name} std ({dataset})': std loss per epoch (computed over runs)
    """
    train_dict = {}
    val_dict = {}
    test_dict = {}
    exp_dict = {}

    # Process each dataset type
    for ds_type, ds_dict in [('train', train_dict), ('val', val_dict), ('test', test_dict), ('exp', exp_dict)]:
        subset = dfs[dfs['dataset'] == ds_type]
        if len(subset) == 0:
            continue

        # Group by epoch and compute mean/std across runs.
        # Keep as epoch-indexed Series so tables with different epoch
        # counts align on epoch instead of breaking on length.
        epoch_groups = subset.groupby('epoch')['loss']

        # Store with explicit dataset suffix in headers
        ds_dict[f'{method_name} mean ({ds_type})'] = epoch_groups.mean()
        ds_dict[f'{method_name} std ({ds_type})'] = epoch_groups.std()

    return train_dict, val_dict, test_dict, exp_dict

def main(
  base_patterns: list = ['loss'], 
  trail: int = 0, 
  usecase: int = 1, 
  num_runs: int = 8, 
  compare: bool = False, 
  export_plots: bool = True
):
    """Main analysis function for multiple runs.

    Each entry in *base_patterns* may be either a plain string or a
    ``(pattern, trail)`` tuple.  When a tuple is given its trail value
    takes precedence over the global *trail* argument, which makes it
    easy to compare runs whose filenames embed different experiment ids::

        main(
            base_patterns=[
                ("loss_PIDU_nowgn_nostruct", 20),
                ("loss_PIDU_wgn_nostruct",   21),
                ("loss_PIDU_wgn_struct",     22),
            ],
            compare=True,
        )
    """
    # Load all runs
    dfs = []
    names = []
    
    train, val, test, exp = {}, {}, {}, {}
    
    base_path = case_dir(usecase)
    
    for entry in base_patterns:
        # Support (pattern, trail) tuples for per-pattern trail overrides
        if isinstance(entry, tuple):
            base_pattern, effective_trail = entry
        else:
            base_pattern, effective_trail = entry, trail
        print("="*60)
        print(f"MULTI-RUN LOSS ANALYSIS FOR PATTERN: {base_pattern}")
        print("="*60)

        try:
            if num_runs > 1:
                combined_df, individual_dfs = load_multiple_runs(base_pattern, effective_trail, num_runs, usecase=usecase)
            else:
                combined_df = load_loss_data(base_path / "runs" / (base_pattern))
                individual_dfs = [combined_df]
                combined_df['run_id'] = 0  # Single run
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return
        
        print(f"\nEpochs: {combined_df['epoch'].min()} to {combined_df['epoch'].max()}")
        print(f"Dataset types: {combined_df['dataset'].unique()}")
        print(f"Runs: {sorted(combined_df['run_id'].unique())}")
        
        # Create output directory
        analyse_folder = base_path / f"analyse-{effective_trail:03d}" / base_pattern
        if export_plots:
            makedirs(analyse_folder, exist_ok=True)
            print(f"\nOutput directory: {analyse_folder}")
        
        # Print statistics
        print_statistics_multi_run(combined_df)
        
        # Analyze convergence on average data
        print("\nCONVERGENCE ANALYSIS (Average across runs):")
        avg_train = combined_df[combined_df['dataset'] == 'train'].groupby('epoch')['loss'].mean()
        if len(avg_train) > 1:
            loss_changes = avg_train.diff().abs()
            print(f"  Average loss change per epoch: {loss_changes.mean():.4f}")
            print(f"  Last 3 epochs average change: {loss_changes.tail(3).mean():.4f}")
            
            recent_change = loss_changes.tail(3).mean()
            if recent_change < avg_train.mean() * 0.01:
                print("  Status: Model appears to be converging ✓")
            else:
                print("  Status: Model still improving significantly")
        
        # Analyze timing on combined data
        # analyze_timing(combined_df)
        
        if export_plots:
            output_dir = analyse_folder
            
            # Generate plots
            print("\nGenerating plots...")
            # plot_loss_by_epoch_with_runs(combined_df, save_path=output_dir / "loss_vs_epoch_all_runs.png")
            # plot_loss_by_epoch_average_only(combined_df, save_path=output_dir / "loss_vs_epoch_average.png")
            # plot_loss_log_scale_with_runs(combined_df, save_path=output_dir / "loss_vs_epoch_log_all_runs.png")
            
            # Use original functions on average data
            avg_df_for_plots = combined_df.groupby(['dataset', 'epoch', 'batch']).agg({
                'loss': 'mean',
                'time': 'mean',
                'system': 'first'
            }).reset_index()
            
            # plot_loss_by_epoch(avg_df_for_plots, save_path=output_dir / "loss_vs_epoch_simple.png")
            # plot_training_batches(avg_df_for_plots, save_path=output_dir / "training_batches_all.png")

            # Plot all runs for this model with clear run-number identification
            model_name = base_pattern.split('_')[1] if '_' in base_pattern else base_pattern
            runs_dataset = 'exp' if (combined_df['dataset'] == 'exp').any() else 'test'
            plot_model_runs_log_scale(
                combined_df,
                dataset_type=runs_dataset,
                model_name=model_name,
                save_path=output_dir / f"{model_name}_{runs_dataset}_runs_logscale.png"
            )
            
            # Plot last epoch in detail
            # last_epoch = combined_df['epoch'].max()
            # plot_training_batches(avg_df_for_plots, epoch=last_epoch,
            #                     save_path=output_dir / f"training_batches_epoch_{last_epoch}.png")
            
            # Generate timing plots
            print("\nGenerating timing analysis...")
            # plot_timing_analysis(avg_df_for_plots, save_path=output_dir / "timing_analysis.png")
            # create_timing_infographic(avg_df_for_plots, save_path=output_dir / "timing_infographic.png")
        
        print("\nAnalysis complete!")
        dfs.append(combined_df)
        names.append(base_pattern.replace('loss_', '').replace('_0_000', ''))
        
        # Store information to export to excel later
        train_dict, val_dict, test_dict, exp_dict = create_table(combined_df, base_pattern.split('_')[1])
        train.update(train_dict)
        val.update(val_dict)
        test.update(test_dict)
        exp.update(exp_dict)
    
    def _to_export_df(table_dict: dict) -> pd.DataFrame:
        # Keep model headers grouped as: <Model> mean, <Model> std
        model_cols = sorted(table_dict.keys(), key=lambda name: (name.rsplit(' ', 1)[0], name.rsplit(' ', 1)[1]))
        # Columns are epoch-indexed Series; pandas aligns them on epoch
        df = pd.DataFrame({k: table_dict[k] for k in model_cols})
        df.index.name = 'epoch'
        return df.reset_index()

    # Export comparison tables as a single Excel workbook with dataset sheets
    tables = [('train', train), ('val', val), ('test', test), ('exp', exp)]
    tables = [(sheet, table) for sheet, table in tables if len(table) > 0]
    if tables:
        comparison_folder = base_path / f"analyse-{trail:03d}" / "comparison"
        makedirs(comparison_folder, exist_ok=True)
        comparison_xlsx = comparison_folder / "comparison.xlsx"
        with pd.ExcelWriter(comparison_xlsx) as writer:
            for sheet_name, table in tables:
                _to_export_df(table).to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"Exported comparison workbook: {comparison_xlsx}")
    
    if compare and len(dfs) > 1:
        print("\n" + "="*60)
        print("COMPARATIVE ANALYSIS BETWEEN LOSS FUNCTIONS")
        print("="*60)
        
        # Create comparison dataframe dictionary
        dfs_dict = {name: df for name, df in zip(names, dfs)}
        
        # Create output directory for comparisons
        if export_plots:
            compare_folder = base_path / f"analyse-{trail:03d}" / "comparison"
            makedirs(compare_folder, exist_ok=True)
            print(f"\nOutput directory: {compare_folder}")
            
            # Plot two-panel loss comparison on log scale (train/val | test)
            print("\nGenerating loss comparison plot...")
            plot_loss_log_comparison(
                dfs_dict,
                names,
                test_dataset='exp' if len(exp) > 0 else 'test',
                save_path=compare_folder / "loss_log_comparison.png"
            )
            
            # Also generate full comparison plots if only 2 methods
            if len(dfs) == 2:
                print("\nGenerating full comparison plots...")
                plot_comparison(
                    dfs[0], 
                    dfs[1], 
                    names[0], 
                    names[1], 
                    save_path=compare_folder / f"comparison_{names[0]}_vs_{names[1]}.png"
                )
        
        print("\nComparative analysis complete!")

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze loss data from multiple runs")
    parser.add_argument(
        '--base-patterns', nargs='+', default=['loss_CNN,loss_DU,loss_PIDU'],
        help=(
            "Base patterns for run files. Each pattern may embed a per-pattern "
            "trail with a colon suffix, e.g. "
            "'loss_PIDU_nowgn_nostruct:20 loss_PIDU_wgn_nostruct:21 loss_PIDU_wgn_struct:22'. "
            "Patterns without a colon use the global --trail value."
        )
    )
    parser.add_argument('--trail', type=int, default=0, help="Trail number for file naming (global default, overridden per pattern with pattern:trail)")
    parser.add_argument('--usecase', type=int, default=1, help="Use case number to determine folder structure")
    parser.add_argument('--num-runs', type=int, default=8, help="Number of runs to load and analyze")
    parser.add_argument('--export-plots', action='store_true', help="Whether to export generated plots to files")
    parser.add_argument('--compare', action='store_true', help="Whether to perform comparative analysis between methods")
      
    return parser.parse_args()

def _parse_pattern(s: str):
    """Convert 'pattern:trail' string to (pattern, int) tuple, or return str."""
    if ':' in s:
        pat, t = s.rsplit(':', 1)
        return (pat, int(t))
    return s

if __name__ == "__main__":
    print(f"main")
    
    args = parse_args()

    raw_patterns = args.base_patterns[0].split(',') if len(args.base_patterns) == 1 else args.base_patterns
    parsed_patterns = [_parse_pattern(p) for p in raw_patterns]
    
    main(
        parsed_patterns,
        args.trail, 
        usecase=args.usecase, 
        num_runs=args.num_runs, 
        compare=args.compare, 
        export_plots=args.export_plots
    )