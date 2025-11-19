"""Generate comparison plots from experimental results."""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse


def load_results(results_dir, task, dataset_size):
    """Load all results for a given task and dataset size."""
    gen_dir = Path(results_dir) / "generalization"

    results = {}
    for json_file in gen_dir.glob(f"{task}_*.json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                # Check if this file matches our dataset size (within 10%)
                train_size = data['dataset']['train_size']
                inferred_size = int(train_size / 0.7)
                if abs(inferred_size - dataset_size) < dataset_size * 0.1:
                    model = data['metadata']['model_type']
                    results[model] = data
        except Exception as e:
            print(f"Error loading {json_file}: {e}")

    return results


def plot_task_comparison(task, dataset_size, results_dir="results", output_dir="results/figures", use_steps=False):
    """Create comparison plot for one task and dataset size.

    Creates a grid with:
    - Columns: Different models (GIN, GCN, GAT, GPS, etc.)
    - Rows: Loss (top) and Accuracy (bottom)
    - Lines: Train, Val, Test

    Args:
        use_steps: If True, plot by training steps instead of epochs
    """
    results = load_results(results_dir, task, dataset_size)

    if not results:
        print(f"No results found for {task} on {dataset_size} graphs")
        return

    models = sorted(results.keys())
    n_models = len(models)

    # Create figure with 2 rows (loss, accuracy) and n_models columns
    fig, axes = plt.subplots(2, n_models, figsize=(5*n_models, 8))

    # Handle single model case
    if n_models == 1:
        axes = axes.reshape(2, 1)

    for col, model in enumerate(models):
        data = results[model]

        if use_steps and 'step_metrics' in data['training'] and data['training']['step_metrics']:
            # Use step-level metrics
            step_metrics = data['training']['step_metrics']
            steps = [m['global_step'] for m in step_metrics]
            train_loss = [m['loss'] for m in step_metrics]
            train_acc = [m['batch_acc'] for m in step_metrics]
            x_label = 'Step'
            x_data = steps

            # For val, we need to interpolate or just use epoch boundaries
            epoch_metrics = data['training']['epoch_metrics']
            val_steps = []
            val_loss = []
            val_acc = []
            steps_per_epoch = len(step_metrics) // len(epoch_metrics) if epoch_metrics else 1
            for i, em in enumerate(epoch_metrics):
                val_steps.append((i + 1) * steps_per_epoch)
                val_loss.append(em['val_loss'])
                val_acc.append(em['val_acc'])
        else:
            # Use epoch-level metrics
            epoch_metrics = data['training']['epoch_metrics']
            epochs = [m['epoch'] for m in epoch_metrics]
            train_loss = [m['train_loss'] for m in epoch_metrics]
            train_acc = [m['train_acc'] for m in epoch_metrics]
            val_loss = [m['val_loss'] for m in epoch_metrics]
            val_acc = [m['val_acc'] for m in epoch_metrics]
            x_label = 'Epoch'
            x_data = epochs
            val_steps = epochs

        # Get final test metrics
        final = data['final_results']
        test_acc = final['test']['accuracy']
        test_loss = final['test']['loss']

        # Plot loss (top row)
        ax_loss = axes[0, col]
        ax_loss.plot(x_data, train_loss, label='Train', linewidth=2, alpha=0.8)
        ax_loss.plot(val_steps, val_loss, label='Val', linewidth=2, alpha=0.8, marker='o' if use_steps else '')
        ax_loss.axhline(y=test_loss, color='red', linestyle='--', label=f'Test={test_loss:.3f}', linewidth=2, alpha=0.8)
        ax_loss.set_xlabel(x_label)
        ax_loss.set_ylabel('Loss')
        ax_loss.set_title(f'{model}\n(Params: {data["metadata"]["total_params"]:,})')
        ax_loss.legend()
        ax_loss.grid(True, alpha=0.3)

        # Plot accuracy (bottom row)
        ax_acc = axes[1, col]
        ax_acc.plot(x_data, train_acc, label='Train', linewidth=2, alpha=0.8)
        ax_acc.plot(val_steps, val_acc, label='Val', linewidth=2, alpha=0.8, marker='o' if use_steps else '')
        ax_acc.axhline(y=test_acc, color='red', linestyle='--', label=f'Test={test_acc:.3f}', linewidth=2, alpha=0.8)
        ax_acc.set_xlabel(x_label)
        ax_acc.set_ylabel('Accuracy')
        ax_acc.legend()
        ax_acc.grid(True, alpha=0.3)
        ax_acc.set_ylim([0, 1.05])

        # Add overfitting indicator
        analysis = data['analysis']
        if analysis['overfitting_detected']:
            ax_acc.text(0.5, 0.95, '⚠️ Overfitting', transform=ax_acc.transAxes,
                       ha='center', va='top', bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
        elif analysis['moderate_overfitting']:
            ax_acc.text(0.5, 0.95, '⚠️ Moderate', transform=ax_acc.transAxes,
                       ha='center', va='top', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
        else:
            ax_acc.text(0.5, 0.95, '✓ Good Gen', transform=ax_acc.transAxes,
                       ha='center', va='top', bbox=dict(boxstyle='round', facecolor='green', alpha=0.3))

    # Overall title
    task_name = task.replace('_', ' ').title()
    fig.suptitle(f'{task_name} - {dataset_size} Graphs', fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout()

    # Save figure
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    suffix = "_steps" if use_steps else ""
    filename = output_path / f"{task}_{dataset_size}graphs_comparison{suffix}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {filename}")

    plt.close()


def plot_all_results(results_dir="results", output_dir="results/figures", use_steps=False):
    """Generate all comparison plots."""

    # Find all unique combinations
    gen_dir = Path(results_dir) / "generalization"
    if not gen_dir.exists():
        print(f"No generalization directory found at {gen_dir}")
        return

    combinations = set()
    for json_file in gen_dir.glob("*.json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                task = data['metadata']['task']
                # Infer dataset size from train size
                train_size = data['dataset']['train_size']
                # Map train size to dataset size (train is 70%)
                dataset_size = int(train_size / 0.7)
                combinations.add((task, dataset_size))
        except Exception as e:
            print(f"Error reading {json_file}: {e}")

    print(f"Found {len(combinations)} unique combinations:")
    for task, size in sorted(combinations):
        print(f"  - {task}: {size} graphs")

    # Generate plots for each combination
    for task, dataset_size in sorted(combinations):
        print(f"\nGenerating plot for {task} on {dataset_size} graphs...")
        plot_task_comparison(task, dataset_size, results_dir, output_dir, use_steps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Results directory')
    parser.add_argument('--output_dir', type=str, default='results/figures',
                        help='Output directory for plots')
    parser.add_argument('--task', type=str, default=None,
                        help='Specific task to plot (optional)')
    parser.add_argument('--dataset_size', type=int, default=None,
                        help='Specific dataset size to plot (optional)')
    parser.add_argument('--use_steps', action='store_true',
                        help='Plot by training steps instead of epochs')
    args = parser.parse_args()

    if args.task and args.dataset_size:
        # Plot specific combination
        plot_task_comparison(args.task, args.dataset_size, args.results_dir, args.output_dir, args.use_steps)
    else:
        # Plot all results
        plot_all_results(args.results_dir, args.output_dir, args.use_steps)


if __name__ == "__main__":
    main()
