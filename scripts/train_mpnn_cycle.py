"""Train MPNN (GIN/GCN/GAT) on cycle detection task."""

import sys
import os
sys.path.insert(0, os.getcwd())

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GCNConv, GATConv, global_mean_pool, global_add_pool
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import degree
import argparse
import time
import json
from datetime import datetime
from pathlib import Path

# Import project modules
from src.datasets.synthetic.universal_graphs import UniversalSyntheticDataset
from src.tasks.cycle_detection import CycleDetectionTask
from src.utils.training_logger import TrainingLogger


class TinyGIN(nn.Module):
    """Tiny GIN for graph classification (Tier 1: ~30K params)."""

    def __init__(self, num_features=1, hidden_dim=32, num_layers=4, num_classes=2, dropout=0.1):
        super().__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Input projection
        self.node_encoder = nn.Linear(num_features, hidden_dim)

        # GIN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Classifier
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Encode node features
        x = self.node_encoder(x)

        # GIN layers
        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)

        # Global pooling
        x = global_add_pool(x, batch)

        # Classification
        x = self.dropout(x)
        x = self.classifier(x)

        return x


class TinyGCN(nn.Module):
    """Tiny GCN for graph classification (Tier 1: ~30K params)."""

    def __init__(self, num_features=1, hidden_dim=32, num_layers=4, num_classes=2, dropout=0.1):
        super().__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # GCN layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(num_features, hidden_dim))

        for i in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        # Classifier
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # GCN layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:  # No ReLU on last layer
                x = F.relu(x)
                x = self.dropout(x)

        # Global pooling
        x = global_mean_pool(x, batch)

        # Classification
        x = self.dropout(x)
        x = self.classifier(x)

        return x


class TinyGAT(nn.Module):
    """Tiny GAT for graph classification (Tier 1: ~30K params)."""

    def __init__(self, num_features=1, hidden_dim=32, num_layers=3, num_classes=2,
                 heads=4, dropout=0.1):
        super().__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # GAT layers
        self.convs = nn.ModuleList()

        # First layer
        self.convs.append(GATConv(num_features, hidden_dim // heads, heads=heads, dropout=dropout))

        # Hidden layers
        for i in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout))

        # Last layer (single head)
        self.convs.append(GATConv(hidden_dim, hidden_dim, heads=1, concat=False, dropout=dropout))

        # Classifier
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # GAT layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:  # No ReLU on last layer
                x = F.elu(x)
                x = self.dropout(x)

        # Global pooling
        x = global_mean_pool(x, batch)

        # Classification
        x = self.dropout(x)
        x = self.classifier(x)

        return x


def add_node_features(data):
    """Add degree as node feature."""
    if data.x is None:
        deg = degree(data.edge_index[0], num_nodes=data.num_nodes, dtype=torch.float)
        data.x = deg.unsqueeze(1)
    return data


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in loader:
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

    return total_loss / total, correct / total


@torch.no_grad()
def eval_model(model, loader, criterion, device):
    """Evaluate model."""
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
    parser.add_argument('--data_dir', type=str, default='data/synthetic_er_500',
                        help='Directory containing graph data')
    parser.add_argument('--num_graphs', type=int, default=500,
                        help='Number of graphs to use')
    parser.add_argument('--model_type', type=str, default='GIN', choices=['GIN', 'GCN', 'GAT'],
                        help='MPNN model type')
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
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Data directory: {args.data_dir}")
    print(f"Model type: {args.model_type}")

    # Load dataset
    print("\n=== Loading Dataset ===")
    dataset_name = 'er_test' if 'test' in args.data_dir else 'er'
    cache_name = f'mpnn_{args.num_graphs}graphs.pt'
    dataset_config = {
        'name': dataset_name,
        'graph_sources': [args.data_dir],
        'cache_path': f'data/processed/{cache_name}',
        'max_graphs': args.num_graphs
    }
    universal_dataset = UniversalSyntheticDataset(dataset_config)
    universal_graphs = universal_dataset.load_data()
    print(f"Loaded {len(universal_graphs)} graphs")

    # Apply cycle detection task
    print("\n=== Applying Cycle Detection Task ===")
    task = CycleDetectionTask({'name': 'cycle_detection'})
    labeled_graphs = task.prepare_data(universal_dataset)
    print(f"Labeled {len(labeled_graphs)} graphs")

    # Add node features
    print("\n=== Adding Node Features ===")
    for data in labeled_graphs:
        add_node_features(data)

    # Create dataloader
    loader = DataLoader(labeled_graphs, batch_size=args.batch_size, shuffle=True)

    # Create model
    print(f"\n=== Creating {args.model_type} Model ===")
    if args.model_type == 'GIN':
        model = TinyGIN(
            num_features=1,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_classes=2
        ).to(device)
    elif args.model_type == 'GCN':
        model = TinyGCN(
            num_features=1,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_classes=2
        ).to(device)
    elif args.model_type == 'GAT':
        model = TinyGAT(
            num_features=1,
            hidden_dim=args.hidden_dim,
            num_layers=max(3, args.num_layers),  # GAT needs at least 3
            num_classes=2
        ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Initialize logger
    logger = TrainingLogger(
        model_name=args.model_type,
        experiment_name=f"cycle_detection_{args.num_graphs}graphs",
        results_dir="results/training_logs"
    )

    config = {
        'model_type': args.model_type,
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'num_graphs': args.num_graphs
    }

    dataset_info = {
        'num_graphs': len(labeled_graphs),
        'graph_type': f'ER(n=variable, p=0.3)',
        'task': 'cycle_detection',
        'node_features': 'degree'
    }

    logger.start(config, str(device), total_params, dataset_info)

    # Training loop
    print(f"\n=== Training for {args.epochs} epochs ===")
    best_acc = 0.0
    start_time = time.time()

    for epoch in range(args.epochs):
        epoch_start = time.time()
        train_loss, train_acc = train_epoch(model, loader, optimizer, criterion, device)
        epoch_time = time.time() - epoch_start

        logger.log_epoch(epoch + 1, {
            'loss': train_loss,
            'accuracy': train_acc
        }, epoch_time)

        if train_acc > best_acc:
            best_acc = train_acc

        # Early stopping
        if train_acc >= 0.98:
            print(f"\n✓ Model has memorized the training set (acc={train_acc*100:.2f}%)")
            print(f"Stopping early at epoch {epoch+1}")
            break

    # Save model
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = f"checkpoints/{args.model_type.lower()}_cycle_{args.num_graphs}graphs.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'model_type': args.model_type,
            'hidden_dim': args.hidden_dim,
            'num_layers': args.num_layers,
            'num_classes': 2
        },
        'accuracy': best_acc
    }, checkpoint_path)

    # Log final results
    logger.finish({
        'best_accuracy': best_acc,
        'epochs_to_convergence': epoch + 1,
        'final_loss': train_loss
    }, checkpoint_path)

    print(f"\nBest training accuracy: {best_acc*100:.2f}%")
    print(f"Model saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
