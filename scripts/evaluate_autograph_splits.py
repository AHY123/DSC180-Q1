"""Evaluate AutoGraph model on train/val/test splits for generalization testing."""

import sys
import os
sys.path.insert(0, 'external/AutoGraph')
sys.path.insert(0, os.getcwd())

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import argparse
import time
import json
from datetime import datetime
from pathlib import Path

# Import AutoGraph tokenizer
from autograph.datamodules.data.tokenizer import Graph2TrailTokenizer

# Our imports
from src.datasets.synthetic.universal_graphs import UniversalSyntheticDataset
from src.tasks.cycle_detection import CycleDetectionTask
from src.tasks.shortest_path import ShortestPathTask


class TinyTransformer(nn.Module):
    """Tiny transformer for graph tasks."""

    def __init__(self, vocab_size, d_model=32, nhead=4, num_layers=4, num_classes=2):
        super().__init__()
        self.d_model = d_model

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(1024, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape

        # Token embeddings
        token_emb = self.token_embedding(input_ids)

        # Positional embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embedding(positions)

        # Combine embeddings
        x = token_emb + pos_emb

        # Create attention mask for transformer
        if attention_mask is not None:
            # Convert to transformer format (True = don't attend)
            mask = (attention_mask == 0)
        else:
            mask = None

        # Transformer
        x = self.transformer(x, src_key_padding_mask=mask)

        # Pool: use mean of non-padding tokens
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand(x.size())
            sum_embeddings = torch.sum(x * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled = sum_embeddings / sum_mask
        else:
            pooled = x.mean(dim=1)

        # Classification
        logits = self.classifier(pooled)
        return logits


class TokenizedGraphDataset(Dataset):
    """Dataset of tokenized graphs."""

    def __init__(self, pyg_graphs, labels):
        self.token_ids = pyg_graphs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.token_ids[idx], self.labels[idx]


def collate_fn(batch, pad_token_id=5):
    """Collate function for batching variable-length sequences."""
    token_ids_list, labels = zip(*batch)

    # Pad sequences
    max_len = max(len(ids) for ids in token_ids_list)
    padded_ids = []
    attention_masks = []

    for ids in token_ids_list:
        padding_length = max_len - len(ids)
        padded = torch.cat([ids, torch.full((padding_length,), pad_token_id, dtype=torch.long)])
        mask = torch.cat([torch.ones(len(ids)), torch.zeros(padding_length)])
        padded_ids.append(padded)
        attention_masks.append(mask)

    return (torch.stack(padded_ids),
            torch.stack(attention_masks),
            torch.tensor(labels, dtype=torch.long))


def train_epoch(model, dataloader, optimizer, criterion, device, step_metrics=None, epoch=0):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (input_ids, attention_mask, labels) in enumerate(dataloader):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # Log step-level metrics
        if step_metrics is not None:
            step_metrics.append({
                'epoch': epoch,
                'step': batch_idx,
                'global_step': len(step_metrics),
                'loss': loss.item(),
                'batch_acc': (preds == labels).float().mean().item(),
                'batch_size': labels.size(0)
            })

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for input_ids, attention_mask, labels in dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy


def load_split_dataset(base_dir, split, task_config, tokenizer):
    """Load and tokenize a specific split."""
    from torch_geometric.utils import to_networkx
    
    split_dir = os.path.join(base_dir, split)

    dataset_config = {
        'name': f'er_{split}',
        'graph_sources': [split_dir],
        'cache_path': f'data/processed/autograph_eval_{split}_{task_config["name"]}.pt',
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

    # Tokenize graphs
    tokenized_graphs = []
    labels = []
    for data in labeled_graphs:
        # AutoGraph tokenizer expects PyG Data objects
        token_ids = tokenizer(data)
        tokenized_graphs.append(token_ids)
        # Convert label to int if needed
        label = data.y.item() if isinstance(data.y, torch.Tensor) else int(data.y)
        labels.append(label)

    return tokenized_graphs, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default='data/synthetic_er_test',
                        help='Base directory with train/valid/test splits')
    parser.add_argument('--task', type=str, choices=['cycle', 'shortest_path'],
                        default='cycle', help='Task to evaluate')
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
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Base directory: {args.base_dir}")
    print(f"Task: {args.task}")
    print(f"Model: AutoGraph")

    # Task configuration
    if args.task == 'cycle':
        task_config = {'name': 'cycle_detection'}
        num_classes = 2
    else:  # shortest_path
        task_config = {
            'name': 'shortest_path',
            'k_pairs': args.k_pairs,
            'max_distance': args.max_distance,
            'connected_only': True,
            'output_type': 'classification'
        }
        num_classes = args.max_distance + 1

    # Load datasets first to find max nodes
    print("\n=== Loading Train Split ===")
    # Load without tokenizer first to get max nodes
    from src.datasets.synthetic.universal_graphs import UniversalSyntheticDataset
    from src.tasks.cycle_detection import CycleDetectionTask
    from src.tasks.shortest_path import ShortestPathTask

    train_split_dir = os.path.join(args.base_dir, 'train')
    train_dataset_config = {
        'name': 'er_train',
        'graph_sources': [train_split_dir],
        'cache_path': f'data/processed/autograph_eval_train_{task_config["name"]}.pt',
    }
    train_universal = UniversalSyntheticDataset(train_dataset_config)
    train_graphs = train_universal.load_data()

    if task_config['name'] == 'cycle_detection':
        task_obj = CycleDetectionTask(task_config)
    else:
        task_obj = ShortestPathTask(task_config)

    train_labeled = task_obj.prepare_data(train_universal)

    # Find max nodes across all splits (check train only for simplicity)
    max_nodes = max(data.num_nodes for data in train_labeled)
    print(f"Max nodes in dataset: {max_nodes}")

    # Initialize tokenizer
    print("\n=== Initializing AutoGraph Tokenizer ===")
    tokenizer = Graph2TrailTokenizer(
        undirected=True,
        append_eos=True,
        max_length=-1,
    )
    print(f"Setting max_num_nodes to {max_nodes}")
    tokenizer.set_num_nodes(max_nodes)
    vocab_size = len(tokenizer)
    print(f"Vocab size: {vocab_size}")

    # Now tokenize train
    print(f"\nTokenizing train split...")
    train_tokens, train_labels = load_split_dataset(args.base_dir, 'train', task_config, tokenizer)
    print(f"Train: {len(train_labels)} examples")

    print("\n=== Loading Valid Split ===")
    valid_tokens, valid_labels = load_split_dataset(args.base_dir, 'valid', task_config, tokenizer)
    print(f"Valid: {len(valid_labels)} examples")

    print("\n=== Loading Test Split ===")
    test_tokens, test_labels = load_split_dataset(args.base_dir, 'test', task_config, tokenizer)
    print(f"Test: {len(test_labels)} examples")

    # Create datasets
    train_dataset = TokenizedGraphDataset(train_tokens, train_labels)
    valid_dataset = TokenizedGraphDataset(valid_tokens, valid_labels)
    test_dataset = TokenizedGraphDataset(test_tokens, test_labels)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_fn)

    # Create model
    print(f"\n=== Creating AutoGraph Model ===")
    model = TinyTransformer(
        vocab_size=vocab_size,
        d_model=32,
        nhead=4,
        num_layers=4,
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

    epoch_metrics = []
    step_metrics = []
    training_start_time = time.time()

    for epoch in range(args.epochs):
        epoch_start = time.time()

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device,
                                            step_metrics, epoch+1)
        val_loss, val_acc = evaluate(model, valid_loader, criterion, device)

        epoch_time = time.time() - epoch_start

        epoch_metrics.append({
            'epoch': epoch + 1,
            'train_loss': float(train_loss),
            'train_acc': float(train_acc),
            'val_loss': float(val_loss),
            'val_acc': float(val_acc),
            'epoch_time': float(epoch_time)
        })

        if (epoch + 1) % 10 == 0 or epoch < 5:
            print(f"{epoch+1:5d} | {train_loss:10.4f} | {train_acc:9.4f} | {val_loss:10.4f} | {val_acc:9.4f} | {epoch_time:6.2f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1

        if val_acc >= 0.95 and train_acc >= 0.95:
            print(f"\n✓ Both train and val achieved 95% accuracy")
            print(f"Stopping early at epoch {epoch+1}")
            break

    total_training_time = time.time() - training_start_time

    # Final evaluation
    print("\n" + "=" * 70)
    print("=== Final Evaluation ===")
    print("=" * 70)

    train_loss, train_acc = evaluate(model, train_loader, criterion, device)
    val_loss, val_acc = evaluate(model, valid_loader, criterion, device)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    print(f"\nTrain: Loss={train_loss:.4f}, Accuracy={train_acc*100:.2f}%")
    print(f"Valid: Loss={val_loss:.4f}, Accuracy={val_acc*100:.2f}%")
    print(f"Test:  Loss={test_loss:.4f}, Accuracy={test_acc*100:.2f}%")

    # Analysis
    train_val_gap = train_acc - val_acc
    train_test_gap = train_acc - test_acc
    val_test_gap = val_acc - test_acc

    print("\n" + "=" * 70)
    print("=== Memorization Analysis ===")
    print("=" * 70)
    print(f"\nAccuracy Gaps:")
    print(f"  Train - Valid: {train_val_gap*100:+.2f}%")
    print(f"  Train - Test:  {train_test_gap*100:+.2f}%")
    print(f"  Valid - Test:  {val_test_gap*100:+.2f}%")

    # Save results
    results_dir = Path("results/generalization")
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{args.task}_AutoGraph_{timestamp}"

    results_json = {
        'metadata': {
            'timestamp': timestamp,
            'task': args.task,
            'model_type': 'AutoGraph',
            'base_dir': args.base_dir,
            'total_params': total_params,
            'device': str(device)
        },
        'config': {
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'max_epochs': args.epochs,
            'k_pairs': args.k_pairs if args.task == 'shortest_path' else None,
            'max_distance': args.max_distance if args.task == 'shortest_path' else None
        },
        'dataset': {
            'train_size': len(train_labels),
            'valid_size': len(valid_labels),
            'test_size': len(test_labels),
            'num_classes': num_classes,
            'vocab_size': vocab_size
        },
        'training': {
            'total_time': float(total_training_time),
            'epochs_run': len(epoch_metrics),
            'best_epoch': best_epoch,
            'best_val_acc': float(best_val_acc),
            'epoch_metrics': epoch_metrics,
            'step_metrics': step_metrics
        },
        'final_results': {
            'train': {'loss': float(train_loss), 'accuracy': float(train_acc)},
            'valid': {'loss': float(val_loss), 'accuracy': float(val_acc)},
            'test': {'loss': float(test_loss), 'accuracy': float(test_acc)}
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

    json_file = results_dir / f"{base_name}.json"
    with open(json_file, 'w') as f:
        json.dump(results_json, f, indent=2)

    print(f"\nResults saved to: {json_file}")


if __name__ == "__main__":
    main()
