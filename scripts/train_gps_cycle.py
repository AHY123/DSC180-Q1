"""Train GPS model on cycle detection.

Key difference from tokenization approaches:
- GPS works directly on graph structure (no tokenization)
- Task is implicit in the prediction target (binary classification)
- Uses node features (degree) + positional encodings
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
from pathlib import Path

# Import graphgps to register GPS models
import graphgps  # noqa

# GPS imports (uses PyG's GraphGym)
from torch_geometric.graphgym.config import cfg, set_cfg, load_cfg
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.optim import create_optimizer, create_scheduler

# Our imports
from src.datasets.synthetic.universal_graphs import UniversalSyntheticDataset
from src.tasks.cycle_detection import CycleDetectionTask
from src.utils.training_logger import TrainingLogger


def prepare_gps_data(pyg_data_list, cfg):
    """Prepare PyG graphs for GPS.

    GPS expects:
    - x: node features [num_nodes, feat_dim]
    - edge_index: edge connectivity
    - y: graph-level label
    - EigVals, EigVecs: Precomputed Laplacian eigenvalues and eigenvectors

    We use node degree as a simple feature.
    """
    from torch_geometric.utils import degree, get_laplacian, to_scipy_sparse_matrix
    import scipy.sparse as sp
    import numpy as np

    max_freqs = cfg.posenc_LapPE.eigen.max_freqs if cfg.posenc_LapPE.enable else 0

    for data in pyg_data_list:
        # Add node degree as feature (GPS needs some node features)
        deg = degree(data.edge_index[0], num_nodes=data.num_nodes, dtype=torch.float)
        data.x = deg.unsqueeze(1)  # [num_nodes, 1]

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
                # For very small graphs, fall back to full eigendecomposition
                eig_vals, eig_vecs = np.linalg.eigh(L.toarray())
                eig_vals = eig_vals[:max_freqs]
                eig_vecs = eig_vecs[:, :max_freqs]

            # Pad to max_freqs if needed
            if eig_vals.shape[0] < max_freqs:
                eig_vals = np.pad(eig_vals, (0, max_freqs - eig_vals.shape[0]),
                                constant_values=np.nan)
                eig_vecs = np.pad(eig_vecs, ((0, 0), (0, max_freqs - eig_vecs.shape[1])),
                                constant_values=np.nan)

            # Convert to tensors in GPS format
            # EigVecs: [num_nodes, max_freqs]
            # EigVals: [num_nodes, max_freqs, 1]
            data.EigVecs = torch.from_numpy(eig_vecs).float()
            data.EigVals = torch.from_numpy(eig_vals).float().unsqueeze(0).repeat(num_nodes, 1).unsqueeze(2)

        # Ensure y is correct shape for classification
        # data.y should be a scalar tensor (0 or 1 for binary classification)
        if isinstance(data.y, torch.Tensor):
            if data.y.dim() == 0:
                # Already scalar tensor - keep as is
                pass
            elif data.y.dim() == 1 and len(data.y) == 1:
                # Extract scalar from 1-element tensor but keep as tensor
                data.y = data.y[0]
            else:
                # Should not happen, but handle it
                data.y = data.y[0] if data.y.numel() > 0 else torch.tensor(0, dtype=torch.long)
        else:
            # Convert to tensor
            data.y = torch.tensor(int(data.y), dtype=torch.long)

    return pyg_data_list


def train_epoch(model, loader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for i, batch in enumerate(loader):
        batch = batch.to(device)
        optimizer.zero_grad()

        # GPS forward pass (may return tuple)
        out = model(batch)
        if isinstance(out, tuple):
            out = out[0]  # Extract predictions from tuple

        # GPS uses binary classification with BCE loss (single output with sigmoid)
        loss = F.binary_cross_entropy_with_logits(out.squeeze(), batch.y.float())

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        # Binary classification: use sigmoid and threshold at 0.5
        pred = (torch.sigmoid(out.squeeze()) > 0.5).long()
        correct += (pred == batch.y).sum().item()
        total += batch.num_graphs

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        if isinstance(out, tuple):
            out = out[0]  # Extract predictions from tuple
        loss = F.binary_cross_entropy_with_logits(out.squeeze(), batch.y.float())

        total_loss += loss.item() * batch.num_graphs
        # Binary classification: use sigmoid and threshold at 0.5
        pred = (torch.sigmoid(out.squeeze()) > 0.5).long()
        correct += (pred == batch.y).sum().item()
        total += batch.num_graphs

    return total_loss / total, correct / total


def main():
    # Configuration
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/synthetic_er_test/train')
    parser.add_argument('--num_graphs', type=int, default=50)
    parser.add_argument('--config', type=str, default='configs/gps_cycle_tiny.yaml')
    args = parser.parse_args()

    # Load GPS config (must call set_cfg first to register custom configs)
    set_cfg(cfg)
    cfg.merge_from_file(args.config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Data directory: {args.data_dir}")

    # Load dataset
    print("\n=== Loading Dataset ===")
    dataset_config = {
        'name': 'er_test' if 'test' in args.data_dir else 'er',
        'graph_sources': [args.data_dir],
        'cache_path': f'data/processed/gps_{args.num_graphs}graphs.pt'
    }
    universal_dataset = UniversalSyntheticDataset(dataset_config)
    universal_graphs = universal_dataset.load_data()
    print(f"Loaded {len(universal_graphs)} graphs")

    # Apply cycle detection task
    print("\n=== Applying Cycle Detection Task ===")
    task = CycleDetectionTask({'name': 'cycle_detection'})
    labeled_graphs = task.prepare_data(universal_dataset)
    print(f"Labeled {len(labeled_graphs)} graphs")

    # Prepare for GPS (add node features and LapPE)
    print("\n=== Preparing GPS Format ===")
    gps_graphs = prepare_gps_data(labeled_graphs, cfg)
    print(f"Added node features (degree) and LapPE to {len(gps_graphs)} graphs")

    # Create dataloader
    loader = DataLoader(gps_graphs, batch_size=cfg.train.batch_size, shuffle=True)

    # Create model
    print("\n=== Creating GPS Model ===")
    # Update config with actual dimensions
    cfg.share.dim_in = 1  # Node degree feature
    cfg.share.dim_out = 2  # Binary classification

    model = create_model().to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Create optimizer and scheduler
    optimizer = create_optimizer(model.parameters(), cfg.optim)
    scheduler = create_scheduler(optimizer, cfg.optim)

    # Initialize logger
    logger = TrainingLogger(
        experiment_name=f"cycle_detection_{args.num_graphs}graphs",
        model_name="GPS",
        results_dir="results/training_logs"
    )

    logger.start(
        config={
            "layers": cfg.gt.layers,
            "dim_hidden": cfg.gt.dim_hidden,
            "n_heads": cfg.gt.n_heads,
            "batch_size": cfg.train.batch_size,
            "learning_rate": cfg.optim.base_lr,
            "num_epochs": cfg.optim.max_epoch,
            "early_stopping": True,
            "posenc": "LapPE (4 freqs, 4 dim)",
            "architecture": cfg.gt.layer_type
        },
        device=str(device),
        model_params=total_params,
        dataset_info={
            "num_graphs": len(gps_graphs),
            "graph_type": "ER(n=10-20, p=0.3)",
            "task": "cycle_detection",
            "node_features": "degree",
            "pos_encoding": "Laplacian PE"
        }
    )

    # Training loop
    print(f"\n=== Training for {cfg.optim.max_epoch} epochs ===")
    best_acc = 0.0

    for epoch in range(cfg.optim.max_epoch):
        epoch_start = time.time()

        train_loss, train_acc = train_epoch(model, loader, optimizer, device)
        scheduler.step()

        epoch_time = time.time() - epoch_start

        # Log epoch
        logger.log_epoch(
            epoch=epoch + 1,
            metrics={"loss": train_loss, "accuracy": train_acc},
            epoch_time=epoch_time
        )

        if train_acc > best_acc:
            best_acc = train_acc

        # Early stopping
        if train_acc >= 0.98:
            print(f"\n✓ Model has memorized the training set (acc={train_acc*100:.2f}%)")
            print(f"Stopping early at epoch {epoch+1}")
            break

    # Save model
    save_dir = Path("checkpoints")
    save_dir.mkdir(exist_ok=True)
    save_path = save_dir / f"gps_cycle_{args.num_graphs}graphs.pt"

    torch.save({
        'model_state_dict': model.state_dict(),
        'config': cfg,
        'num_params': total_params
    }, save_path)

    # Finish logging
    logger.finish(
        final_metrics={
            "best_accuracy": best_acc,
            "epochs_to_convergence": epoch + 1,
            "final_loss": train_loss
        },
        save_model_path=str(save_path)
    )

    print(f"\nBest training accuracy: {best_acc*100:.2f}%")
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
