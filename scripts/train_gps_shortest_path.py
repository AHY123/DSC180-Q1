"""Train GPS model on shortest path prediction.

GPS works directly on graph structure with node features indicating source/target pairs.
Task: Predict shortest path distance between marked nodes (11-class classification: 0-10).
"""

import sys
import os
sys.path.insert(0, 'external/GraphGPS')
sys.path.insert(0, 'src')
sys.path.insert(0, '.')

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import time
import argparse
from pathlib import Path

# Import graphgps to register GPS models
import graphgps  # noqa

# GPS imports
from torch_geometric.graphgym.config import cfg, set_cfg
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.optim import create_optimizer

# Our imports
from src.datasets.synthetic.universal_graphs import UniversalSyntheticDataset
from src.tasks.shortest_path import ShortestPathTask
from src.utils.training_logger import TrainingLogger


def prepare_gps_data(pyg_data_list, cfg):
    """Prepare PyG graphs for GPS.

    Note: ShortestPathTask already adds node features [degree, is_source, is_target].
    We just need to add Laplacian PE if enabled.
    """
    from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix
    import scipy.sparse as sp
    import numpy as np

    max_freqs = cfg.posenc_LapPE.eigen.max_freqs if cfg.posenc_LapPE.enable else 0

    for data in pyg_data_list:
        # Node features already set by task: [degree, is_source, is_target]
        # data.x shape: [num_nodes, 3]

        # Compute Laplacian PE if enabled
        if cfg.posenc_LapPE.enable:
            num_nodes = data.num_nodes

            # Get Laplacian
            edge_index, edge_weight = get_laplacian(
                data.edge_index, normalization=None, num_nodes=num_nodes
            )
            L = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes)

            # Compute eigenvalues and eigenvectors
            try:
                eig_vals, eig_vecs = sp.linalg.eigsh(
                    L.toarray(), k=min(max_freqs, num_nodes-1), which='SM'
                )
            except:
                eig_vals, eig_vecs = np.linalg.eigh(L.toarray())
                eig_vals = eig_vals[:max_freqs]
                eig_vecs = eig_vecs[:, :max_freqs]

            # Pad to max_freqs if needed
            if eig_vals.shape[0] < max_freqs:
                eig_vals = np.pad(eig_vals, (0, max_freqs - eig_vals.shape[0]),
                                constant_values=np.nan)
                eig_vecs = np.pad(eig_vecs, ((0, 0), (0, max_freqs - eig_vecs.shape[1])),
                                constant_values=np.nan)

            # Convert to tensors
            data.EigVecs = torch.from_numpy(eig_vecs).float()
            data.EigVals = torch.from_numpy(eig_vals).float().unsqueeze(0).repeat(num_nodes, 1).unsqueeze(2)

        # Ensure y is scalar tensor
        if isinstance(data.y, torch.Tensor):
            if data.y.dim() == 1 and len(data.y) == 1:
                data.y = data.y[0]

    return pyg_data_list


def train_epoch(model, loader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # GPS forward pass
        out = model(batch)
        if isinstance(out, tuple):
            out = out[0]

        # Multi-class classification
        loss = F.cross_entropy(out, batch.y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        pred = out.argmax(dim=1)
        correct += (pred == batch.y).sum().item()
        total += batch.num_graphs

    return total_loss / total, correct / total


@torch.no_grad()
def eval_model(model, loader, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    mae_sum = 0

    for batch in loader:
        batch = batch.to(device)

        out = model(batch)
        if isinstance(out, tuple):
            out = out[0]

        loss = F.cross_entropy(out, batch.y)

        total_loss += loss.item() * batch.num_graphs
        pred = out.argmax(dim=1)
        correct += (pred == batch.y).sum().item()

        # MAE
        mae_sum += (pred - batch.y).abs().sum().item()
        total += batch.num_graphs

    return total_loss / total, correct / total, mae_sum / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/gps_shortest_path_tiny.yaml',
                        help='GPS config file')
    parser.add_argument('--data_dir', type=str, default='data/synthetic_er_500',
                        help='Directory containing graph data')
    parser.add_argument('--num_graphs', type=int, default=500,
                        help='Number of graphs to use')
    parser.add_argument('--k_pairs', type=int, default=1,
                        help='Number of node pairs per graph')
    parser.add_argument('--max_distance', type=int, default=10,
                        help='Maximum distance to predict')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Data directory: {args.data_dir}")

    # Load GPS config
    set_cfg(cfg)
    cfg.merge_from_file(args.config)

    # Override config settings
    cfg.train.batch_size = args.batch_size
    cfg.optim.max_epoch = args.epochs
    cfg.optim.base_lr = args.lr
    cfg.dataset.task_type = 'classification'
    cfg.share.num_classes = args.max_distance + 1  # 0 to max_distance

    print("\n=== Loading Dataset ===")
    dataset_name = 'er_test' if 'test' in args.data_dir else 'er'
    cache_name = f'gps_sp_{args.num_graphs}graphs_k{args.k_pairs}.pt'
    dataset_config = {
        'name': dataset_name,
        'graph_sources': [args.data_dir],
        'cache_path': f'data/processed/{cache_name}',
        'max_graphs': args.num_graphs
    }
    universal_dataset = UniversalSyntheticDataset(dataset_config)
    universal_graphs = universal_dataset.load_data()
    print(f"Loaded {len(universal_graphs)} graphs")

    # Apply shortest path task
    print("\n=== Applying Shortest Path Task ===")
    task = ShortestPathTask({
        'name': 'shortest_path',
        'k_pairs': args.k_pairs,
        'max_distance': args.max_distance,
        'connected_only': True,
        'output_type': 'classification'
    })
    labeled_graphs = task.prepare_data(universal_dataset)
    print(f"Created {len(labeled_graphs)} node pairs")

    # Prepare for GPS
    print("\n=== Preparing GPS Format ===")
    gps_graphs = prepare_gps_data(labeled_graphs, cfg)
    print(f"Added Laplacian PE to {len(gps_graphs)} graphs")

    # Create dataloader
    loader = DataLoader(gps_graphs, batch_size=cfg.train.batch_size, shuffle=True)

    # Create model
    print("\n=== Creating GPS Model ===")
    # Update config with actual dimensions
    cfg.dataset.node_encoder_num_types = 0
    cfg.dataset.node_encoder_bn = False
    cfg.dataset.edge_encoder_num_types = 0
    cfg.dataset.edge_encoder_bn = False

    # Node features: [degree, is_source, is_target] = 3 dimensions
    cfg.share.dim_in = 3

    model = create_model(to_device=False, dim_in=3, dim_out=args.max_distance + 1)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Optimizer
    optimizer = create_optimizer(model.parameters(), cfg.optim)

    # Logger
    logger = TrainingLogger(
        model_name="GPS",
        experiment_name=f"shortest_path_{args.num_graphs}graphs_k{args.k_pairs}",
        results_dir="results/training_logs"
    )

    config = {
        'data_dir': args.data_dir,
        'num_graphs': args.num_graphs,
        'k_pairs': args.k_pairs,
        'max_distance': args.max_distance,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'num_classes': args.max_distance + 1
    }

    dataset_info = {
        'num_pairs': len(labeled_graphs),
        'graph_type': 'ER(n=variable, p=0.3)',
        'task': 'shortest_path',
        'node_features': 'degree + source/target indicators'
    }

    logger.start(config, str(device), total_params, dataset_info)

    # Training loop
    print(f"\n=== Training for {args.epochs} epochs ===")
    best_acc = 0.0
    best_mae = float('inf')

    for epoch in range(args.epochs):
        epoch_start = time.time()
        train_loss, train_acc = train_epoch(model, loader, optimizer, device)
        epoch_time = time.time() - epoch_start

        # Eval
        eval_loss, eval_acc, eval_mae = eval_model(model, loader, device)

        logger.log_epoch(epoch + 1, {
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'eval_loss': eval_loss,
            'eval_accuracy': eval_acc,
            'eval_mae': eval_mae
        }, epoch_time)

        if train_acc > best_acc:
            best_acc = train_acc
        if eval_mae < best_mae:
            best_mae = eval_mae

        # Early stopping
        if train_acc >= 0.95:
            print(f"\n✓ Model achieved 95% training accuracy")
            print(f"Stopping early at epoch {epoch+1}")
            break

    # Save model
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = f"checkpoints/gps_shortest_path_{args.num_graphs}graphs_k{args.k_pairs}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'accuracy': best_acc,
        'mae': best_mae
    }, checkpoint_path)

    # Log final results
    logger.finish({
        'best_accuracy': best_acc,
        'best_mae': best_mae,
        'epochs_to_convergence': epoch + 1,
        'final_train_loss': train_loss,
        'final_eval_mae': eval_mae
    }, checkpoint_path)

    print(f"\nBest training accuracy: {best_acc*100:.2f}%")
    print(f"Best eval MAE: {best_mae:.4f}")
    print(f"Model saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
