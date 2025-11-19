"""Generate visualizations and tables for model comparison results."""

import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11

def load_latest_results():
    """Load the most recent results for each model/task/dataset combination."""
    results_dir = Path('results/generalization')

    results = {}

    # Define what we're looking for
    models = ['GIN', 'GPS', 'AutoGraph']
    tasks = ['cycle', 'shortest_path']
    datasets = ['500', '5000']

    for model in models:
        for task in tasks:
            for dataset in datasets:
                # Find most recent file
                pattern = f"{task}_{model}_*.json"
                files = sorted(results_dir.glob(pattern), reverse=True)

                if files:
                    with open(files[0]) as f:
                        data = json.load(f)
                        key = f"{task}_{dataset}_{model}"
                        results[key] = data
                        print(f"Loaded: {files[0].name}")

    return results

def create_comparison_table(results):
    """Create comparison table of final accuracies."""

    rows = []
    for key, data in results.items():
        parts = key.split('_')
        if len(parts) >= 3:
            if parts[0] == 'shortest':
                task = 'shortest_path'
                dataset = parts[2]
                model = parts[3]
            else:
                task = parts[0]
                dataset = parts[1]
                model = parts[2]

            final = data['results']['final_evaluation']

            rows.append({
                'Task': task.replace('_', ' ').title(),
                'Dataset Size': dataset,
                'Model': model,
                'Parameters': f"{data['config']['model_parameters']:,}",
                'Train Acc': f"{final['train']['accuracy']:.2f}%",
                'Valid Acc': f"{final['valid']['accuracy']:.2f}%",
                'Test Acc': f"{final['test']['accuracy']:.2f}%",
                'Train-Test Gap': f"{data['analysis']['accuracy_gaps']['train_test']:.2f}%",
            })

    df = pd.DataFrame(rows)
    df = df.sort_values(['Task', 'Dataset Size', 'Model'])

    return df

def plot_accuracy_comparison(results, output_dir):
    """Create bar plots comparing test accuracies."""

    # Prepare data
    data_by_task = {'cycle': [], 'shortest_path': []}

    for key, data in results.items():
        parts = key.split('_')
        if len(parts) >= 3:
            if parts[0] == 'shortest':
                task = 'shortest_path'
                dataset = parts[2]
                model = parts[3]
            else:
                task = parts[0]
                dataset = parts[1]
                model = parts[2]

            if task in data_by_task:
                final = data['results']['final_evaluation']
                data_by_task[task].append({
                    'model': model,
                    'dataset': f"{dataset} graphs",
                    'test_acc': final['test']['accuracy']
                })

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (task, task_data) in enumerate(data_by_task.items()):
        if not task_data:
            continue

        df = pd.DataFrame(task_data)

        # Create grouped bar chart
        datasets = sorted(df['dataset'].unique())
        models = ['GIN', 'GPS', 'AutoGraph']

        x = np.arange(len(datasets))
        width = 0.25

        for i, model in enumerate(models):
            model_data = df[df['model'] == model]
            accs = [model_data[model_data['dataset'] == ds]['test_acc'].values[0]
                   if len(model_data[model_data['dataset'] == ds]) > 0 else 0
                   for ds in datasets]

            offset = (i - 1) * width
            bars = axes[idx].bar(x + offset, accs, width, label=model)

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                             f'{height:.1f}%',
                             ha='center', va='bottom', fontsize=9)

        axes[idx].set_xlabel('Dataset Size', fontsize=12)
        axes[idx].set_ylabel('Test Accuracy (%)', fontsize=12)
        axes[idx].set_title(f'{task.replace("_", " ").title()} Detection', fontsize=14, fontweight='bold')
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(datasets)
        axes[idx].legend()
        axes[idx].set_ylim([0, 105])
        axes[idx].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'accuracy_comparison.png'}")
    plt.close()

def plot_generalization_gaps(results, output_dir):
    """Plot train-test accuracy gaps."""

    data = []
    for key, result in results.items():
        parts = key.split('_')
        if len(parts) >= 3:
            if parts[0] == 'shortest':
                task = 'Shortest Path'
                dataset = parts[2]
                model = parts[3]
            else:
                task = 'Cycle Detection'
                dataset = parts[1]
                model = parts[2]

            gap = result['analysis']['accuracy_gaps']['train_test']
            data.append({
                'Task': task,
                'Dataset': f"{dataset} graphs",
                'Model': model,
                'Gap': gap
            })

    df = pd.DataFrame(data)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, task in enumerate(['Cycle Detection', 'Shortest Path']):
        task_df = df[df['Task'] == task]

        datasets = sorted(task_df['Dataset'].unique())
        models = ['GIN', 'GPS', 'AutoGraph']

        x = np.arange(len(datasets))
        width = 0.25

        for i, model in enumerate(models):
            model_data = task_df[task_df['Model'] == model]
            gaps = [model_data[model_data['Dataset'] == ds]['Gap'].values[0]
                   if len(model_data[model_data['Dataset'] == ds]) > 0 else 0
                   for ds in datasets]

            offset = (i - 1) * width
            bars = axes[idx].bar(x + offset, gaps, width, label=model)

            # Add value labels
            for bar in bars:
                height = bar.get_height()
                axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                             f'{height:.1f}%',
                             ha='center', va='bottom' if height > 0 else 'top',
                             fontsize=9)

        axes[idx].set_xlabel('Dataset Size', fontsize=12)
        axes[idx].set_ylabel('Train-Test Gap (%)', fontsize=12)
        axes[idx].set_title(f'{task}', fontsize=14, fontweight='bold')
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(datasets)
        axes[idx].legend()
        axes[idx].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[idx].grid(axis='y', alpha=0.3)

        # Add shading for memorization zone
        axes[idx].axhspan(10, axes[idx].get_ylim()[1], alpha=0.1, color='red', label='High Memorization')

    plt.tight_layout()
    plt.savefig(output_dir / 'generalization_gaps.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'generalization_gaps.png'}")
    plt.close()

def plot_model_parameters(results, output_dir):
    """Plot model parameter counts."""

    models_params = {}
    for key, result in results.items():
        parts = key.split('_')
        if len(parts) >= 3:
            if parts[0] == 'shortest':
                model = parts[3]
            else:
                model = parts[2]

            if model not in models_params:
                models_params[model] = result['config']['model_parameters']

    fig, ax = plt.subplots(figsize=(8, 5))

    models = list(models_params.keys())
    params = [models_params[m]/1000 for m in models]  # In thousands

    bars = ax.bar(models, params, color=['#1f77b4', '#ff7f0e', '#2ca02c'])

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}K',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('Parameters (thousands)', fontsize=12)
    ax.set_title('Model Size Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'model_parameters.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'model_parameters.png'}")
    plt.close()

def main():
    output_dir = Path('results/figures')
    output_dir.mkdir(exist_ok=True)

    print("Loading results...")
    results = load_latest_results()

    print("\nCreating comparison table...")
    df = create_comparison_table(results)

    # Save table as CSV
    df.to_csv(output_dir / 'results_comparison.csv', index=False)
    print(f"Saved: {output_dir / 'results_comparison.csv'}")

    # Print table
    print("\n" + "="*100)
    print("RESULTS COMPARISON TABLE")
    print("="*100)
    print(df.to_string(index=False))
    print("="*100)

    print("\nCreating visualizations...")
    plot_accuracy_comparison(results, output_dir)
    plot_generalization_gaps(results, output_dir)
    plot_model_parameters(results, output_dir)

    print("\nAll visualizations created successfully!")
    print(f"Output directory: {output_dir.absolute()}")

if __name__ == '__main__':
    main()
