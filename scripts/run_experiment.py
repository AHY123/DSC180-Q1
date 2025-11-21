#!/usr/bin/env python3
"""Unified experiment runner using model registry and unified trainer.

This script demonstrates the new unified infrastructure:
- Model Registry: Create any model type with consistent interface
- Unified Trainer: Train any model with single training loop
- Wandb Integration: Optional experiment tracking

Usage:
    # GNN models
    uv run python scripts/run_experiment.py --model GIN --task cycle --data data/synthetic_er_5000

    # GPS model
    uv run python scripts/run_experiment.py --model GPS --task shortest_path --data data/synthetic_er_5000 --gps_config configs/GPS/...

    # Transformer models
    uv run python scripts/run_experiment.py --model AutoGraph --task cycle --data data/synthetic_er_5000
"""

import argparse
import sys
from pathlib import Path
import torch
from datetime import datetime
import wandb

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.model_registry import get_registry
from src.core.trainer import UnifiedTrainer
from src.datasets.synthetic_dataset import load_synthetic_graph_dataset
from src.adapters.autograph_adapter import AutoGraphTokenizer
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.utils.data import DataLoader, TensorDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Run unified experiment')

    # Model selection
    parser.add_argument('--model', type=str, required=True,
                       choices=['GIN', 'GCN', 'GAT', 'GPS', 'AutoGraph', 'graph-token'],
                       help='Model type')

    # Task and data
    parser.add_argument('--task', type=str, required=True,
                       choices=['cycle', 'shortest_path'],
                       help='Task type')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to dataset directory')

    # Model hyperparameters
    parser.add_argument('--hidden_dim', type=int, default=64,
                       help='Hidden dimension size')
    parser.add_argument('--num_layers', type=int, default=3,
                       help='Number of layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')

    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--early_stopping', action='store_true',
                       help='Enable early stopping')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')

    # GPS-specific
    parser.add_argument('--gps_config', type=str, default=None,
                       help='Path to GPS config file (required for GPS model)')

    # Transformer-specific
    parser.add_argument('--d_model', type=int, default=32,
                       help='Transformer model dimension')
    parser.add_argument('--nhead', type=int, default=4,
                       help='Number of attention heads')
    parser.add_argument('--max_len', type=int, default=2048,
                       help='Max sequence length')

    # Shortest path task
    parser.add_argument('--k_pairs', type=int, default=10,
                       help='Number of node pairs for shortest path')
    parser.add_argument('--max_distance', type=int, default=5,
                       help='Max distance for shortest path buckets')

    # Experiment tracking
    parser.add_argument('--no_wandb', action='store_true',
                       help='Disable wandb logging')
    parser.add_argument('--output_dir', type=str, default='results/generalization',
                       help='Directory for results')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                       help='Directory for model checkpoints')

    return parser.parse_args()


def load_data(args):
    """Load and prepare data based on model type."""
    print(f"\n=== Loading {args.task} data from {args.data} ===")

    base_dir = Path(args.data)
    train_graphs = load_synthetic_graph_dataset(base_dir / 'train', args.task)
    val_graphs = load_synthetic_graph_dataset(base_dir / 'val', args.task)
    test_graphs = load_synthetic_graph_dataset(base_dir / 'test', args.task)

    print(f"Loaded {len(train_graphs)} train, {len(val_graphs)} val, {len(test_graphs)} test graphs")

    # Get task info
    num_classes = train_graphs[0].y.item() + 1 if args.task == 'cycle' else (args.max_distance + 2)
    num_features = train_graphs[0].num_node_features

    # Create data loaders based on model type
    is_transformer = args.model in ['AutoGraph', 'graph-token']

    if is_transformer:
        # Tokenize for transformers
        print(f"\nTokenizing graphs for {args.model}...")
        tokenizer = AutoGraphTokenizer(
            serialization='dfs' if args.model == 'AutoGraph' else 'index',
            max_seq_len=args.max_len
        )

        train_tokens = tokenizer.tokenize_batch(train_graphs)
        val_tokens = tokenizer.tokenize_batch(val_graphs)
        test_tokens = tokenizer.tokenize_batch(test_graphs)

        # Extract labels
        train_labels = torch.tensor([g.y.item() for g in train_graphs])
        val_labels = torch.tensor([g.y.item() for g in val_graphs])
        test_labels = torch.tensor([g.y.item() for g in test_graphs])

        # Create dataloaders
        train_dataset = TensorDataset(train_tokens['input_ids'],
                                     train_tokens['attention_mask'],
                                     train_labels)
        val_dataset = TensorDataset(val_tokens['input_ids'],
                                   val_tokens['attention_mask'],
                                   val_labels)
        test_dataset = TensorDataset(test_tokens['input_ids'],
                                    test_tokens['attention_mask'],
                                    test_labels)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        vocab_size = tokenizer.get_vocab_size()

    else:
        # PyG data loaders for GNNs and GPS
        train_loader = PyGDataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
        val_loader = PyGDataLoader(val_graphs, batch_size=args.batch_size, shuffle=False)
        test_loader = PyGDataLoader(test_graphs, batch_size=args.batch_size, shuffle=False)
        vocab_size = None

    return train_loader, val_loader, test_loader, num_classes, num_features, vocab_size


def create_model(args, num_features, num_classes, vocab_size):
    """Create model using model registry."""
    print(f"\n=== Creating {args.model} model ===")

    registry = get_registry()

    # Build model config
    if args.model in ['GIN', 'GCN', 'GAT']:
        config = {
            'num_features': num_features,
            'num_classes': num_classes,
            'hidden_dim': args.hidden_dim,
            'num_layers': args.num_layers,
            'dropout': args.dropout
        }
        if args.model == 'GAT':
            config['num_heads'] = 4

    elif args.model == 'GPS':
        if args.gps_config is None:
            raise ValueError("GPS model requires --gps_config argument")
        config = {
            'num_features': num_features,
            'num_classes': num_classes,
            'config_file': args.gps_config
        }

    elif args.model in ['AutoGraph', 'graph-token']:
        config = {
            'vocab_size': vocab_size,
            'num_classes': num_classes,
            'd_model': args.d_model,
            'nhead': args.nhead,
            'num_layers': args.num_layers,
            'max_len': args.max_len,
            'dropout': args.dropout
        }

    # Create model
    model = registry.create_model(args.model, config)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {total_params:,} parameters")

    return model


def main():
    args = parse_args()

    print("=" * 80)
    print(f"Unified Experiment: {args.model} on {args.task}")
    print("=" * 80)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Load data
    train_loader, val_loader, test_loader, num_classes, num_features, vocab_size = load_data(args)

    # Create model
    model = create_model(args, num_features, num_classes, vocab_size)

    # Initialize wandb
    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(
            project="DSC180-GNN-vs-Transformers",
            name=f"{args.task}_{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "model_type": args.model,
                "task": args.task,
                "hidden_dim": args.hidden_dim if args.model not in ['AutoGraph', 'graph-token'] else args.d_model,
                "num_layers": args.num_layers,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "epochs": args.epochs,
                "dropout": args.dropout,
                "dataset": Path(args.data).name,
                "early_stopping": args.early_stopping,
                "k_pairs": args.k_pairs if args.task == 'shortest_path' else None,
                "max_distance": args.max_distance if args.task == 'shortest_path' else None,
            },
            tags=[args.task, args.model, "generalization", "unified-infrastructure"]
        )

    # Create trainer
    trainer = UnifiedTrainer(
        model=model,
        device=device,
        model_type=args.model,
        task_type=args.task,
        num_classes=num_classes,
        use_wandb=use_wandb
    )

    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else None
    results = trainer.train(
        train_loader=train_loader,
        valid_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        num_epochs=args.epochs,
        early_stopping=args.early_stopping,
        early_stopping_patience=args.patience,
        checkpoint_dir=checkpoint_dir
    )

    # Save results
    output_dir = Path(args.output_dir)
    config_dict = {
        'task': args.task,
        'model': args.model,
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'epochs': args.epochs,
        'dataset': Path(args.data).name,
    }

    json_path = trainer.save_results(results, output_dir, config_dict)

    # Finish wandb
    if use_wandb:
        wandb.finish()

    print("\n" + "=" * 80)
    print("Experiment complete!")
    print("=" * 80)
    print(f"Results saved to: {json_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
