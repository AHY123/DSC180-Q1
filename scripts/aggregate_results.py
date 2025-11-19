"""Aggregate and compare results from all experiments."""

import json
import pandas as pd
from pathlib import Path
import argparse
from datetime import datetime


def load_all_results(results_dir="results"):
    """Load all JSON results from multiple directories."""
    results_path = Path(results_dir)

    all_results = {
        'training_logs': [],
        'generalization': [],
        'metadata': {
            'aggregation_time': datetime.now().isoformat(),
            'total_experiments': 0
        }
    }

    # Load training logs
    training_logs_dir = results_path / "training_logs"
    if training_logs_dir.exists():
        for json_file in training_logs_dir.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    data['_source_file'] = str(json_file.name)
                    all_results['training_logs'].append(data)
            except Exception as e:
                print(f"Error loading {json_file}: {e}")

    # Load generalization results
    gen_dir = results_path / "generalization"
    if gen_dir.exists():
        for json_file in gen_dir.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    data['_source_file'] = str(json_file.name)
                    all_results['generalization'].append(data)
            except Exception as e:
                print(f"Error loading {json_file}: {e}")

    all_results['metadata']['total_experiments'] = (
        len(all_results['training_logs']) + len(all_results['generalization'])
    )

    return all_results


def create_comparison_table(results_dict):
    """Create comparison tables from results."""

    # Training logs table
    training_rows = []
    for r in results_dict['training_logs']:
        # Extract final metrics
        final_metrics = r.get('final_metrics', {})

        # Get last epoch metrics
        history = r.get('training_history', [])
        last_epoch = history[-1] if history else {}

        row = {
            'run_id': r.get('run_id', 'unknown'),
            'model': r.get('model_name', 'unknown'),
            'experiment': r.get('experiment_name', 'unknown'),
            'timestamp': r.get('timestamp', 'unknown'),
            'params': r.get('model_params', 0),
            'device': r.get('device', 'unknown'),
            'total_time_s': r.get('total_time_seconds', 0),
            'epochs': len(history),
            '_source': r.get('_source_file', '')
        }

        # Add final metrics
        row.update({f'final_{k}': v for k, v in final_metrics.items()})

        # Add last epoch metrics
        row.update({f'last_epoch_{k}': v for k, v in last_epoch.items()
                   if k not in ['epoch', 'epoch_time']})

        training_rows.append(row)

    # Generalization results table
    gen_rows = []
    for r in results_dict['generalization']:
        metadata = r.get('metadata', {})
        config = r.get('config', {})
        final_results = r.get('final_results', {})
        analysis = r.get('analysis', {})
        training = r.get('training', {})

        row = {
            'task': metadata.get('task', 'unknown'),
            'model': metadata.get('model_type', 'unknown'),
            'timestamp': metadata.get('timestamp', 'unknown'),
            'params': metadata.get('total_params', 0),
            'device': metadata.get('device', 'unknown'),
            'hidden_dim': config.get('hidden_dim', 0),
            'num_layers': config.get('num_layers', 0),
            'batch_size': config.get('batch_size', 0),
            'lr': config.get('learning_rate', 0),
            'epochs_run': training.get('epochs_run', 0),
            'best_epoch': training.get('best_epoch', 0),
            'total_time_s': training.get('total_time', 0),
            'train_acc': final_results.get('train', {}).get('accuracy', 0),
            'train_loss': final_results.get('train', {}).get('loss', 0),
            'val_acc': final_results.get('valid', {}).get('accuracy', 0),
            'val_loss': final_results.get('valid', {}).get('loss', 0),
            'test_acc': final_results.get('test', {}).get('accuracy', 0),
            'test_loss': final_results.get('test', {}).get('loss', 0),
            'train_val_gap': analysis.get('train_val_gap', 0),
            'train_test_gap': analysis.get('train_test_gap', 0),
            'overfitting': analysis.get('overfitting_detected', False),
            '_source': r.get('_source_file', '')
        }

        gen_rows.append(row)

    training_df = pd.DataFrame(training_rows) if training_rows else pd.DataFrame()
    gen_df = pd.DataFrame(gen_rows) if gen_rows else pd.DataFrame()

    return training_df, gen_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Root results directory')
    parser.add_argument('--output_csv', type=str, default='results/aggregated_results.csv',
                        help='Output CSV file')
    parser.add_argument('--output_json', type=str, default='results/aggregated_results.json',
                        help='Output JSON file')
    args = parser.parse_args()

    print(f"Loading results from {args.results_dir}...")
    results = load_all_results(args.results_dir)

    print(f"Found {results['metadata']['total_experiments']} total experiments:")
    print(f"  - Training logs: {len(results['training_logs'])}")
    print(f"  - Generalization: {len(results['generalization'])}")

    # Create comparison tables
    training_df, gen_df = create_comparison_table(results)

    # Save aggregated results
    output_json_path = Path(args.output_json)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved aggregated JSON to {output_json_path}")

    # Save CSV tables
    if not training_df.empty:
        csv_path = Path(args.output_csv).parent / "training_logs_comparison.csv"
        training_df.to_csv(csv_path, index=False)
        print(f"Saved training logs CSV to {csv_path}")

        print(f"\nTraining Logs Summary ({len(training_df)} runs):")
        if 'model' in training_df.columns:
            print(training_df.groupby('model').agg({
                'params': 'first',
                'epochs': 'mean',
                'total_time_s': 'mean'
            }).round(2))

    if not gen_df.empty:
        csv_path = Path(args.output_csv).parent / "generalization_comparison.csv"
        gen_df.to_csv(csv_path, index=False)
        print(f"\nSaved generalization CSV to {csv_path}")

        print(f"\nGeneralization Summary ({len(gen_df)} runs):")
        if 'task' in gen_df.columns and 'model' in gen_df.columns:
            summary = gen_df.groupby(['task', 'model']).agg({
                'train_acc': 'mean',
                'val_acc': 'mean',
                'test_acc': 'mean',
                'train_val_gap': 'mean',
                'overfitting': 'sum'
            }).round(4)
            print(summary)

    # Print quick stats
    print(f"\n{'='*70}")
    print("Quick Statistics")
    print(f"{'='*70}")

    if not gen_df.empty:
        print("\nGeneralization Results:")
        for _, row in gen_df.iterrows():
            print(f"  {row['task']:15s} | {row['model']:5s} | "
                  f"Train: {row['train_acc']*100:5.1f}% | "
                  f"Val: {row['val_acc']*100:5.1f}% | "
                  f"Test: {row['test_acc']*100:5.1f}% | "
                  f"Gap: {row['train_val_gap']*100:+5.1f}%")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
