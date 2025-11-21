"""Evaluate models on train/val/test splits to check for memorization.

This script trains models on the train split and evaluates on train/val/test
to detect overfitting and memorization.
"""

import sys
import os
sys.path.insert(0, os.getcwd())

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import argparse
import time
import json
from datetime import datetime
from pathlib import Path
import wandb

from src.datasets.synthetic.universal_graphs import UniversalSyntheticDataset
from src.tasks.cycle_detection import CycleDetectionTask
from src.tasks.shortest_path import ShortestPathTask


def load_split_dataset(base_dir, split, task_config, max_graphs=None):
    """Load a specific split (train/valid/test)."""
    split_dir = os.path.join(base_dir, split)

    dataset_config = {
        'name': f'er_{split}',
        'graph_sources': [split_dir],
        'cache_path': f'data/processed/eval_{split}_{task_config["name"]}.pt',
        'max_graphs': max_graphs
    }

    universal_dataset = UniversalSyntheticDataset(dataset_config)
    universal_graphs = universal_dataset.load_data()

    # Apply task
    if task_config['name'] == 'cycle_detection':
        task = CycleDetectionTask(task_config)
    elif task_config['name'] == 'shortest_path':
        task = ShortestPathTask(task_config)
    else:
        raise ValueError(f"Unknown task: {task_config['name']}")

    labeled_graphs = task.prepare_data(universal_dataset)

    return labeled_graphs, task


def train_epoch_gin(model, loader, optimizer, criterion, device, step_metrics=None, epoch=0):
    """Train GIN for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, batch in enumerate(loader):
        batch = batch.to(device)
        optimizer.zero_grad()

        out = model(batch)
        loss = criterion(out, batch.y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        pred = out.argmax(dim=1)
        correct += (pred == batch.y).sum().item()
        total += batch.num_graphs

        # Log step-level metrics
        if step_metrics is not None:
            step_metrics.append({
                'epoch': epoch,
                'step': batch_idx,
                'global_step': len(step_metrics),
                'loss': loss.item(),
                'batch_acc': (pred == batch.y).float().mean().item(),
                'batch_size': batch.num_graphs
            })

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate_gin(model, loader, criterion, device):
    """Evaluate GIN model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        loss = criterion(out, batch.y)

        total_loss += loss.item() * batch.num_graphs
        pred = out.argmax(dim=1)
        correct += (pred == batch.y).sum().item()
        total += batch.num_graphs

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default='data/synthetic_er_test',
                        help='Base directory with train/valid/test splits')
    parser.add_argument('--task', type=str, choices=['cycle', 'shortest_path'],
                        default='cycle', help='Task to evaluate')
    parser.add_argument('--model_type', type=str, default='GIN',
                        choices=['GIN', 'GCN', 'GAT'],
                        help='Model type')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=4,
                        help='Number of layers')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--k_pairs', type=int, default=1,
                        help='Node pairs per graph (shortest path only)')
    parser.add_argument('--max_distance', type=int, default=10,
                        help='Max distance (shortest path only)')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable wandb logging')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Base directory: {args.base_dir}")
    print(f"Task: {args.task}")
    print(f"Model: {args.model_type}")

    # Initialize wandb
    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(
            project="DSC180-GNN-vs-Transformers",
            name=f"{args.task}_{args.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "model_type": args.model_type,
                "task": args.task,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "epochs": args.epochs,
                "dataset": os.path.basename(args.base_dir),
                "k_pairs": args.k_pairs if args.task == 'shortest_path' else None,
                "max_distance": args.max_distance if args.task == 'shortest_path' else None,
            },
            tags=[args.task, args.model_type, "generalization", "train-val-test-splits"]
        )
        print(f"Wandb run: {wandb.run.name}")
    else:
        print("Wandb logging disabled")

    # Task configuration
    if args.task == 'cycle':
        task_config = {'name': 'cycle_detection'}
        num_classes = 2
        num_features = 1  # Just degree
    else:  # shortest_path
        task_config = {
            'name': 'shortest_path',
            'k_pairs': args.k_pairs,
            'max_distance': args.max_distance,
            'connected_only': True,
            'output_type': 'classification'
        }
        num_classes = args.max_distance + 1
        num_features = 3  # [degree, is_source, is_target]

    # Load datasets
    print("\n=== Loading Train Split ===")
    train_data, task = load_split_dataset(args.base_dir, 'train', task_config)
    print(f"Train: {len(train_data)} examples")

    print("\n=== Loading Valid Split ===")
    valid_data, _ = load_split_dataset(args.base_dir, 'valid', task_config)
    print(f"Valid: {len(valid_data)} examples")

    print("\n=== Loading Test Split ===")
    test_data, _ = load_split_dataset(args.base_dir, 'test', task_config)
    print(f"Test: {len(test_data)} examples")

    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    # Create model
    print(f"\n=== Creating {args.model_type} Model ===")

    # Import model classes
    from scripts.train_mpnn_cycle import TinyGIN, TinyGCN, TinyGAT

    if args.model_type == 'GIN':
        model = TinyGIN(
            num_features=num_features,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_classes=num_classes
        ).to(device)
    elif args.model_type == 'GCN':
        model = TinyGCN(
            num_features=num_features,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_classes=num_classes
        ).to(device)
    elif args.model_type == 'GAT':
        model = TinyGAT(
            num_features=num_features,
            hidden_dim=args.hidden_dim,
            num_layers=max(3, args.num_layers),
            num_classes=num_classes
        ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    print(f"\n=== Training for {args.epochs} epochs ===")
    print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>10} | {'Val Acc':>9} | {'Time':>6}")
    print("-" * 70)

    best_val_acc = 0.0
    best_epoch = 0

    # Track per-epoch metrics and per-step metrics
    epoch_metrics = []
    step_metrics = []
    training_start_time = time.time()

    for epoch in range(args.epochs):
        epoch_start = time.time()

        # Train
        train_loss, train_acc = train_epoch_gin(model, train_loader, optimizer, criterion, device,
                                                  step_metrics=step_metrics, epoch=epoch+1)

        # Validate
        val_loss, val_acc = evaluate_gin(model, valid_loader, criterion, device)

        epoch_time = time.time() - epoch_start

        # Store metrics
        epoch_metrics.append({
            'epoch': epoch + 1,
            'train_loss': float(train_loss),
            'train_acc': float(train_acc),
            'val_loss': float(val_loss),
            'val_acc': float(val_acc),
            'epoch_time': float(epoch_time)
        })

        # Log to wandb
        if use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train/loss': train_loss,
                'train/accuracy': train_acc,
                'val/loss': val_loss,
                'val/accuracy': val_acc,
                'epoch_time': epoch_time,
            })

        if (epoch + 1) % 10 == 0 or epoch < 5:
            print(f"{epoch+1:5d} | {train_loss:10.4f} | {train_acc:9.4f} | {val_loss:10.4f} | {val_acc:9.4f} | {epoch_time:6.2f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1

        # Early stopping disabled - train for full epochs
        # if val_acc >= 0.95 and train_acc >= 0.95:
        #     print(f"\n✓ Both train and val achieved 95% accuracy")
        #     print(f"Stopping early at epoch {epoch+1}")
        #     break

    total_training_time = time.time() - training_start_time

    # Final evaluation on all splits
    print("\n" + "=" * 70)
    print("=== Final Evaluation ===")
    print("=" * 70)

    train_loss, train_acc = evaluate_gin(model, train_loader, criterion, device)
    val_loss, val_acc = evaluate_gin(model, valid_loader, criterion, device)
    test_loss, test_acc = evaluate_gin(model, test_loader, criterion, device)

    print(f"\nTrain: Loss={train_loss:.4f}, Accuracy={train_acc*100:.2f}%")
    print(f"Valid: Loss={val_loss:.4f}, Accuracy={val_acc*100:.2f}%")
    print(f"Test:  Loss={test_loss:.4f}, Accuracy={test_acc*100:.2f}%")

    # Memorization analysis
    print("\n" + "=" * 70)
    print("=== Memorization Analysis ===")
    print("=" * 70)

    train_val_gap = train_acc - val_acc
    train_test_gap = train_acc - test_acc
    val_test_gap = val_acc - test_acc

    print(f"\nAccuracy Gaps:")
    print(f"  Train - Valid: {train_val_gap*100:+.2f}%")
    print(f"  Train - Test:  {train_test_gap*100:+.2f}%")
    print(f"  Valid - Test:  {val_test_gap*100:+.2f}%")

    print(f"\nInterpretation:")
    if train_val_gap > 0.10:
        print("  ⚠️  STRONG OVERFITTING: Train >> Valid (gap > 10%)")
        print("      Model is memorizing training data")
    elif train_val_gap > 0.05:
        print("  ⚠️  MODERATE OVERFITTING: Train > Valid (gap 5-10%)")
        print("      Some memorization, but not severe")
    else:
        print("  ✓  GOOD GENERALIZATION: Train ≈ Valid (gap < 5%)")
        print("      Model is learning generalizable patterns")

    if abs(val_test_gap) > 0.05:
        print(f"\n  ⚠️  Val-Test gap is {abs(val_test_gap)*100:.1f}% - splits may not be i.i.d.")
    else:
        print(f"\n  ✓  Val and Test are consistent (gap < 5%)")

    # Save results
    results_dir = Path("results/generalization")
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{args.task}_{args.model_type}_{timestamp}"

    # Comprehensive JSON results
    results_json = {
        'metadata': {
            'timestamp': timestamp,
            'task': args.task,
            'model_type': args.model_type,
            'base_dir': args.base_dir,
            'total_params': total_params,
            'device': str(device)
        },
        'config': {
            'hidden_dim': args.hidden_dim,
            'num_layers': args.num_layers,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'max_epochs': args.epochs,
            'k_pairs': args.k_pairs if args.task == 'shortest_path' else None,
            'max_distance': args.max_distance if args.task == 'shortest_path' else None
        },
        'dataset': {
            'train_size': len(train_data),
            'valid_size': len(valid_data),
            'test_size': len(test_data),
            'num_classes': num_classes,
            'num_features': num_features
        },
        'training': {
            'total_time': float(total_training_time),
            'epochs_run': len(epoch_metrics),
            'best_epoch': best_epoch,
            'best_val_acc': float(best_val_acc),
            'epoch_metrics': epoch_metrics,
            'step_metrics': step_metrics  # Per-batch/step metrics
        },
        'final_results': {
            'train': {
                'loss': float(train_loss),
                'accuracy': float(train_acc)
            },
            'valid': {
                'loss': float(val_loss),
                'accuracy': float(val_acc)
            },
            'test': {
                'loss': float(test_loss),
                'accuracy': float(test_acc)
            }
        },
        'analysis': {
            'train_val_gap': float(train_val_gap),
            'train_test_gap': float(train_test_gap),
            'val_test_gap': float(val_test_gap),
            'overfitting_detected': train_val_gap > 0.10,
            'moderate_overfitting': 0.05 < train_val_gap <= 0.10,
            'good_generalization': train_val_gap <= 0.05
        }
    }

    # Save JSON
    json_file = results_dir / f"{base_name}.json"
    with open(json_file, 'w') as f:
        json.dump(results_json, f, indent=2)

    # Save summary text
    results_file = results_dir / f"{base_name}_summary.txt"
    with open(results_file, 'w') as f:
        f.write(f"Task: {args.task}\n")
        f.write(f"Model: {args.model_type}\n")
        f.write(f"Parameters: {total_params:,}\n")
        f.write(f"Base dir: {args.base_dir}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"\nResults:\n")
        f.write(f"  Train: {train_acc*100:.2f}% (loss={train_loss:.4f})\n")
        f.write(f"  Valid: {val_acc*100:.2f}% (loss={val_loss:.4f})\n")
        f.write(f"  Test:  {test_acc*100:.2f}% (loss={test_loss:.4f})\n")
        f.write(f"\nGaps:\n")
        f.write(f"  Train - Valid: {train_val_gap*100:+.2f}%\n")
        f.write(f"  Train - Test:  {train_test_gap*100:+.2f}%\n")
        f.write(f"  Valid - Test:  {val_test_gap*100:+.2f}%\n")
        f.write(f"\nTraining:\n")
        f.write(f"  Best val accuracy: {best_val_acc*100:.2f}% at epoch {best_epoch}\n")
        f.write(f"  Total epochs: {len(epoch_metrics)}\n")
        f.write(f"  Total time: {total_training_time:.2f}s\n")

    print(f"\nResults saved to:")
    print(f"  JSON: {json_file}")
    print(f"  Summary: {results_file}")

    # Log final results to wandb
    if use_wandb:
        wandb.log({
            'final/train_loss': train_loss,
            'final/train_accuracy': train_acc,
            'final/val_loss': val_loss,
            'final/val_accuracy': val_acc,
            'final/test_loss': test_loss,
            'final/test_accuracy': test_acc,
            'final/train_val_gap': train_val_gap,
            'final/train_test_gap': train_test_gap,
            'final/val_test_gap': val_test_gap,
            'training/total_time': total_training_time,
            'training/best_epoch': best_epoch,
            'training/best_val_acc': best_val_acc,
            'model/total_params': total_params,
        })

        # Log the final results JSON as an artifact
        artifact = wandb.Artifact(
            name=f"results-{args.task}-{args.model_type}",
            type="results",
            description=f"Results for {args.model_type} on {args.task}"
        )
        artifact.add_file(str(json_file))
        wandb.log_artifact(artifact)

        wandb.finish()
        print("\nWandb run completed")


if __name__ == "__main__":
    main()
