"""Generate training curve plots showing loss and accuracy over epochs."""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.style.use('default')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11

def load_latest_results():
    """Load the most recent results for each model/task/dataset combination."""
    results_dir = Path('results/generalization')
    results = {}

    models = ['GIN', 'GPS', 'AutoGraph', 'graph-token']
    tasks = ['cycle', 'shortest_path']
    datasets = ['500', '5000']

    for model in models:
        for task in tasks:
            for dataset in datasets:
                pattern = f"{task}_{model}_*.json"
                files = sorted(results_dir.glob(pattern), reverse=True)
                if files:
                    with open(files[0]) as f:
                        data = json.load(f)
                        key = f"{task}_{dataset}_{model}"
                        results[key] = data
                        print(f"Loaded: {files[0].name}")
    return results

def plot_training_curves_by_task(results, output_dir):
    """Create training curves organized by task."""

    tasks_config = {
        'cycle': {'name': 'Cycle Detection', 'datasets': ['500', '5000']},
        'shortest_path': {'name': 'Shortest Path Prediction', 'datasets': ['500', '5000']}
    }

    models = ['GIN', 'GPS', 'AutoGraph', 'graph-token']
    colors = {'GIN': '#1f77b4', 'GPS': '#ff7f0e', 'AutoGraph': '#2ca02c', 'graph-token': '#d62728'}

    for task, config in tasks_config.items():
        for dataset in config['datasets']:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'{config["name"]} - {dataset} Graphs Training Curves',
                        fontsize=16, fontweight='bold', y=0.995)

            # Plot each model
            for model in models:
                key = f"{task}_{dataset}_{model}"
                if key not in results:
                    continue

                data = results[key]
                epoch_metrics = data['training']['epoch_metrics']

                if not epoch_metrics:
                    continue

                epochs = [m['epoch'] for m in epoch_metrics]
                train_loss = [m['train_loss'] for m in epoch_metrics]
                val_loss = [m['val_loss'] for m in epoch_metrics]
                train_acc = [m['train_acc'] * 100 for m in epoch_metrics]
                val_acc = [m['val_acc'] * 100 for m in epoch_metrics]

                color = colors[model]

                # Training Loss
                axes[0, 0].plot(epochs, train_loss, '-', color=color,
                              label=model, linewidth=2, alpha=0.8)

                # Validation Loss
                axes[0, 1].plot(epochs, val_loss, '-', color=color,
                              label=model, linewidth=2, alpha=0.8)

                # Training Accuracy
                axes[1, 0].plot(epochs, train_acc, '-', color=color,
                              label=model, linewidth=2, alpha=0.8)

                # Validation Accuracy
                axes[1, 1].plot(epochs, val_acc, '-', color=color,
                              label=model, linewidth=2, alpha=0.8)

            # Configure subplots
            axes[0, 0].set_title('Training Loss', fontsize=13, fontweight='bold')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            axes[0, 1].set_title('Validation Loss', fontsize=13, fontweight='bold')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

            axes[1, 0].set_title('Training Accuracy', fontsize=13, fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Accuracy (%)')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_ylim([0, 105])

            axes[1, 1].set_title('Validation Accuracy', fontsize=13, fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Accuracy (%)')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_ylim([0, 105])

            plt.tight_layout()

            filename = f'training_curves_{task}_{dataset}graphs.png'
            filepath = output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved: {filepath}")
            plt.close()

def plot_combined_train_val_curves(results, output_dir):
    """Create combined train/val curves showing overfitting."""

    tasks_config = {
        'cycle': 'Cycle Detection',
        'shortest_path': 'Shortest Path Prediction'
    }

    models = ['GIN', 'GPS', 'AutoGraph', 'graph-token']
    colors = {'GIN': '#1f77b4', 'GPS': '#ff7f0e', 'AutoGraph': '#2ca02c', 'graph-token': '#d62728'}

    for task, task_name in tasks_config.items():
        fig, axes = plt.subplots(1, 4, figsize=(22, 5))
        fig.suptitle(f'{task_name} - Train vs Validation Accuracy (5000 graphs)',
                    fontsize=16, fontweight='bold')

        for idx, model in enumerate(models):
            key = f"{task}_5000_{model}"
            if key not in results:
                continue

            data = results[key]
            epoch_metrics = data['training']['epoch_metrics']

            if not epoch_metrics:
                continue

            epochs = [m['epoch'] for m in epoch_metrics]
            train_acc = [m['train_acc'] * 100 for m in epoch_metrics]
            val_acc = [m['val_acc'] * 100 for m in epoch_metrics]

            color = colors[model]

            axes[idx].plot(epochs, train_acc, '-', color=color,
                         label='Train', linewidth=2.5, alpha=0.9)
            axes[idx].plot(epochs, val_acc, '--', color=color,
                         label='Validation', linewidth=2.5, alpha=0.9)

            # Add shaded region between train and val
            axes[idx].fill_between(epochs, train_acc, val_acc,
                                  color=color, alpha=0.2)

            # Add final gap annotation
            final_gap = data['analysis']['train_test_gap'] * 100
            axes[idx].text(0.95, 0.05, f'Final Gap: {final_gap:.1f}%',
                         transform=axes[idx].transAxes,
                         ha='right', va='bottom',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                         fontsize=11, fontweight='bold')

            axes[idx].set_title(f'{model}', fontsize=14, fontweight='bold')
            axes[idx].set_xlabel('Epoch', fontsize=12)
            axes[idx].set_ylabel('Accuracy (%)', fontsize=12)
            axes[idx].legend(fontsize=11)
            axes[idx].grid(True, alpha=0.3)
            axes[idx].set_ylim([0, 105])

        plt.tight_layout()

        filename = f'overfitting_analysis_{task}.png'
        filepath = output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.close()

def main():
    output_dir = Path('results/figures')
    output_dir.mkdir(exist_ok=True)

    print("Loading results...")
    results = load_latest_results()

    print("\nCreating training curves by task...")
    plot_training_curves_by_task(results, output_dir)

    print("\nCreating overfitting analysis plots...")
    plot_combined_train_val_curves(results, output_dir)

    print("\nAll training curve visualizations created successfully!")
    print(f"Output directory: {output_dir.absolute()}")

if __name__ == '__main__':
    main()
