"""Generate training curve plots showing loss and accuracy over STEPS with rolling mean."""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.style.use('default')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11

def rolling_mean(data, window=10):
    """Compute rolling mean with given window size."""
    if len(data) < window:
        return data
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / window


def detect_convergence_step(results, task, threshold=0.02, window=50):
    """Detect the step where most models have converged.

    Convergence is defined as when the rolling mean of loss changes
    by less than threshold over the last window steps.
    """
    convergence_steps = []

    for key, data in results.items():
        if not key.startswith(task):
            continue

        step_metrics = data['training'].get('step_metrics', [])
        if not step_metrics or len(step_metrics) < window * 2:
            continue

        losses = np.array([m['loss'] for m in step_metrics])
        rm_losses = rolling_mean(losses, window)

        # Find where loss stabilizes (derivative is small)
        for i in range(len(rm_losses) - window):
            recent = rm_losses[i:i+window]
            if len(recent) > 0:
                change = np.abs(recent[-1] - recent[0]) / (recent[0] + 1e-6)
                if change < threshold:
                    # Add some buffer (20% more steps)
                    conv_step = int((i + window) * 1.2)
                    convergence_steps.append(min(conv_step, len(step_metrics)))
                    break
        else:
            # Didn't converge, use all steps
            convergence_steps.append(len(step_metrics))

    if convergence_steps:
        # Use the max convergence step (so all models are shown to convergence)
        return max(convergence_steps)
    return None

def load_latest_results():
    """Load the most recent results for each model/task combination."""
    results_dir = Path('results/generalization')
    results = {}

    models = ['GIN', 'GCN', 'GAT', 'GPS', 'AutoGraph', 'graph-token']
    tasks = ['cycle', 'shortest_path']

    for model in models:
        for task in tasks:
            pattern = f"{task}_{model}_*.json"
            files = sorted(results_dir.glob(pattern), reverse=True)
            if files:
                with open(files[0]) as f:
                    data = json.load(f)
                    key = f"{task}_{model}"
                    results[key] = data
                    print(f"Loaded: {files[0].name}")

    return results

def plot_step_curves(results, output_dir, window=20):
    """Create training curves over steps with rolling mean."""

    tasks_config = {
        'cycle': 'Cycle Detection',
        'shortest_path': 'Shortest Path Prediction'
    }

    # Define colors for each model
    colors = {
        'GIN': '#1f77b4',      # blue
        'GCN': '#17becf',      # cyan
        'GAT': '#9467bd',      # purple
        'GPS': '#ff7f0e',      # orange
        'AutoGraph': '#2ca02c', # green
        'graph-token': '#d62728' # red
    }

    models = ['GIN', 'GCN', 'GAT', 'GPS', 'AutoGraph', 'graph-token']

    for task, task_name in tasks_config.items():
        # Create 2x2 subplot: Loss and Accuracy for Train and Val
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle(f'{task_name} - Training Curves over Steps (5000 graphs, 100 epochs)',
                    fontsize=16, fontweight='bold')

        for model in models:
            key = f"{task}_{model}"
            if key not in results:
                continue

            data = results[key]
            step_metrics = data['training'].get('step_metrics', [])

            if not step_metrics:
                print(f"No step metrics for {key}")
                continue

            # Extract step-level data
            steps = [m['global_step'] for m in step_metrics]
            train_loss = [m['loss'] for m in step_metrics]
            train_acc = [m['batch_acc'] * 100 for m in step_metrics]

            color = colors.get(model, '#333333')

            # Training Loss - raw data (light) + rolling mean (bold)
            axes[0, 0].plot(steps, train_loss, '-', color=color, alpha=0.15, linewidth=0.5)
            if len(train_loss) >= window:
                rm_loss = rolling_mean(np.array(train_loss), window)
                rm_steps = steps[window-1:]
                axes[0, 0].plot(rm_steps, rm_loss, '-', color=color,
                               label=model, linewidth=2)
            else:
                axes[0, 0].plot(steps, train_loss, '-', color=color,
                               label=model, linewidth=2)

            # Training Accuracy - raw data (light) + rolling mean (bold)
            axes[1, 0].plot(steps, train_acc, '-', color=color, alpha=0.15, linewidth=0.5)
            if len(train_acc) >= window:
                rm_acc = rolling_mean(np.array(train_acc), window)
                rm_steps = steps[window-1:]
                axes[1, 0].plot(rm_steps, rm_acc, '-', color=color,
                               label=model, linewidth=2)
            else:
                axes[1, 0].plot(steps, train_acc, '-', color=color,
                               label=model, linewidth=2)

        # Also plot epoch-level validation metrics
        for model in models:
            key = f"{task}_{model}"
            if key not in results:
                continue

            data = results[key]
            epoch_metrics = data['training'].get('epoch_metrics', [])

            if not epoch_metrics:
                continue

            # Convert epochs to approximate steps (estimate based on step count)
            step_metrics = data['training'].get('step_metrics', [])
            if step_metrics:
                steps_per_epoch = len(step_metrics) // len(epoch_metrics) if epoch_metrics else 1
            else:
                steps_per_epoch = 100  # fallback estimate

            epochs = [m['epoch'] * steps_per_epoch for m in epoch_metrics]
            val_loss = [m['val_loss'] for m in epoch_metrics]
            val_acc = [m['val_acc'] * 100 for m in epoch_metrics]

            color = colors.get(model, '#333333')

            # Validation Loss
            axes[0, 1].plot(epochs, val_loss, '-', color=color,
                          label=model, linewidth=2, alpha=0.9)

            # Validation Accuracy
            axes[1, 1].plot(epochs, val_acc, '-', color=color,
                          label=model, linewidth=2, alpha=0.9)

        # Configure subplots
        axes[0, 0].set_title('Training Loss (per step with rolling mean)', fontsize=13, fontweight='bold')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend(loc='upper right')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim(bottom=0)

        axes[0, 1].set_title('Validation Loss (per epoch)', fontsize=13, fontweight='bold')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend(loc='upper right')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim(bottom=0)

        axes[1, 0].set_title('Training Accuracy (per step with rolling mean)', fontsize=13, fontweight='bold')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Accuracy (%)')
        axes[1, 0].legend(loc='lower right')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim([0, 105])

        axes[1, 1].set_title('Validation Accuracy (per epoch)', fontsize=13, fontweight='bold')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Accuracy (%)')
        axes[1, 1].legend(loc='lower right')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim([0, 105])

        plt.tight_layout()

        filename = f'step_curves_{task}.png'
        filepath = output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.close()


def plot_combined_step_comparison(results, output_dir, window=20):
    """Create combined loss and accuracy comparison over steps."""

    colors = {
        'GIN': '#1f77b4', 'GCN': '#17becf', 'GAT': '#9467bd',
        'GPS': '#ff7f0e', 'AutoGraph': '#2ca02c', 'graph-token': '#d62728'
    }

    models = ['GIN', 'GCN', 'GAT', 'GPS', 'AutoGraph', 'graph-token']

    for task in ['cycle', 'shortest_path']:
        task_name = 'Cycle Detection' if task == 'cycle' else 'Shortest Path'

        # Find the minimum total steps (slowest model in terms of steps/epoch)
        # This gives us a fair comparison at the same number of epochs
        min_steps = float('inf')
        for model in models:
            key = f"{task}_{model}"
            if key in results:
                step_metrics = results[key]['training'].get('step_metrics', [])
                if step_metrics:
                    total_steps = len(step_metrics)
                    min_steps = min(min_steps, total_steps)

        max_step = min_steps if min_steps != float('inf') else 10000
        print(f"  {task}: Using x-axis limit at {max_step} steps (slowest model's total)")

        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        fig.suptitle(f'{task_name} - Loss & Accuracy vs Steps (rolling mean, window={window})',
                    fontsize=16, fontweight='bold')

        for model in models:
            key = f"{task}_{model}"
            if key not in results:
                continue

            data = results[key]
            step_metrics = data['training'].get('step_metrics', [])
            epoch_metrics = data['training'].get('epoch_metrics', [])

            if not step_metrics or len(step_metrics) < window:
                continue

            color = colors.get(model, '#333333')

            # Training metrics (dashed, transparent)
            steps = [m['global_step'] for m in step_metrics]
            train_loss = [m['loss'] for m in step_metrics]
            train_acc = [m['batch_acc'] * 100 for m in step_metrics]

            # Rolling mean for training
            rm_loss = rolling_mean(np.array(train_loss), window)
            rm_acc = rolling_mean(np.array(train_acc), window)
            rm_steps = steps[window-1:]

            # Training Loss (dashed, see-through)
            axes[0].plot(rm_steps, rm_loss, '--', color=color, alpha=0.4, linewidth=2)

            # Training Accuracy (dashed, see-through)
            axes[1].plot(rm_steps, rm_acc, '--', color=color, alpha=0.4, linewidth=2)

            # Validation metrics (solid lines)
            if epoch_metrics:
                # Convert epochs to approximate steps
                steps_per_epoch = len(step_metrics) // len(epoch_metrics) if epoch_metrics else 1
                val_steps = [m['epoch'] * steps_per_epoch for m in epoch_metrics]
                val_loss = [m['val_loss'] for m in epoch_metrics]
                val_acc = [m['val_acc'] * 100 for m in epoch_metrics]

                # Validation Loss (solid)
                axes[0].plot(val_steps, val_loss, '-', color=color, label=model, linewidth=2, alpha=0.9)

                # Validation Accuracy (solid)
                axes[1].plot(val_steps, val_acc, '-', color=color, label=model, linewidth=2, alpha=0.9)

        axes[0].set_xlabel('Step', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Loss vs Steps (dashed=train, solid=val)', fontsize=14)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(bottom=0)
        axes[0].set_xlim(0, max_step)

        axes[1].set_xlabel('Step', fontsize=12)
        axes[1].set_ylabel('Accuracy (%)', fontsize=12)
        axes[1].set_title('Accuracy vs Steps (dashed=train, solid=val)', fontsize=14)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 105])
        axes[1].set_xlim(0, max_step)

        plt.tight_layout()

        filepath = output_dir / f'step_comparison_{task}.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.close()


def main():
    output_dir = Path('results/figures')
    output_dir.mkdir(exist_ok=True)

    print("Loading results...")
    results = load_latest_results()

    print(f"\nCreating step-based training curves with rolling mean...")
    plot_step_curves(results, output_dir, window=20)

    print("\nCreating combined step comparison plots...")
    plot_combined_step_comparison(results, output_dir, window=20)

    print("\nAll step-based visualizations created successfully!")
    print(f"Output directory: {output_dir.absolute()}")

if __name__ == '__main__':
    main()
