"""Evaluate graph-token model on train/val/test splits for generalization testing."""

import sys
import os
sys.path.insert(0, 'external/graph-token')
sys.path.insert(0, os.getcwd())

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import networkx as nx
import argparse
import time
import json
from datetime import datetime
from pathlib import Path

# Import graph-token's task
from graph_task import CycleCheck, ShortestPath


class TinyTransformer(nn.Module):
    """Tiny transformer for graph tasks."""

    def __init__(self, vocab_size, d_model=32, nhead=4, num_layers=4, num_classes=2):
        super().__init__()
        self.d_model = d_model

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(512, d_model)

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

        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.pos_embedding(positions)
        x = token_emb + pos_emb

        if attention_mask is not None:
            src_key_padding_mask = ~attention_mask.bool()
        else:
            src_key_padding_mask = None

        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            x = x.mean(dim=1)

        logits = self.classifier(x)
        return logits


class GraphTokenDataset(Dataset):
    """Dataset that tokenizes graphs using graph-token."""

    def __init__(self, graphml_dir: str, vocab: dict, task):
        self.graphml_dir = Path(graphml_dir)
        self.vocab = vocab
        self.task = task

        # Load all graphs
        self.graphs = []
        graphml_files = sorted(self.graphml_dir.glob("*.graphml"))

        for i, filepath in enumerate(graphml_files):
            graph = nx.read_graphml(filepath)
            if graph.is_directed():
                graph = graph.to_undirected()
            graph = nx.convert_node_labels_to_integers(graph)
            self.graphs.append((str(i), graph))

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        name, graph = self.graphs[idx]

        # Tokenize using graph-token format
        # tokenize_graph returns Dict[graph_id, List[str]] where each string is a tokenized sample
        tokenized_data = self.task.tokenize_graph(graph, name)
        # Get the first sample for this graph (or the only one for cycle check)
        tokens_str = list(tokenized_data.values())[0][0]  # First graph_id, first sample
        tokens = tokens_str.split()  # Split string into tokens
        token_ids = torch.tensor([self.vocab.get(t, self.vocab['<unk>']) for t in tokens], dtype=torch.long)

        # Extract label from the tokenized output (after '<p>' token)
        # Find the token after '<p>'
        if '<p>' in tokens:
            p_idx = tokens.index('<p>')
            label_token = tokens[p_idx + 1] if p_idx + 1 < len(tokens) else None

            if self.task.name == 'cycle_check':
                # "yes" -> 1, "no" -> 0
                label = 1 if label_token == 'yes' else 0
            elif self.task.name == 'shortest_path':
                # "lenD" -> D, "INF" -> max_distance (or some large value)
                if label_token == 'INF':
                    label = 10  # Use 10 as INF label
                elif label_token and label_token.startswith('len'):
                    label = int(label_token[3:])  # Extract D from "lenD"
                else:
                    label = 0
        else:
            label = 0

        return token_ids, label


def collate_fn(batch, pad_token_id):
    """Collate function for batching."""
    token_ids_list, labels = zip(*batch)

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

        if step_metrics is not None:
            step_metrics.append({
                'epoch': epoch,
                'step': batch_idx,
                'global_step': len(step_metrics),
                'loss': loss.item(),
                'batch_acc': (preds == labels).float().mean().item(),
                'batch_size': labels.size(0)
            })

    return total_loss / len(dataloader), correct / total


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

    return total_loss / len(dataloader), correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default='data/synthetic_er_test',
                        help='Base directory with train/valid/test splits')
    parser.add_argument('--task', type=str, choices=['cycle', 'shortest_path'],
                        default='cycle', help='Task to evaluate')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--k_pairs', type=int, default=1)
    parser.add_argument('--max_distance', type=int, default=10)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Base directory: {args.base_dir}")
    print(f"Task: {args.task}")
    print(f"Model: graph-token")

    # Task configuration
    if args.task == 'cycle':
        task = CycleCheck()
        num_classes = 2
    else:  # shortest_path
        task = ShortestPath()
        num_classes = args.max_distance + 1  # We'll filter distances during data loading

    # Build vocabulary dynamically from data
    print("\n=== Building Vocabulary from Data ===")
    vocab = {'<pad>': 0, '<unk>': 1}

    # Sample graphs from train to build vocabulary
    train_graphml_dir = Path(os.path.join(args.base_dir, 'train'))
    sample_graphs = []
    for i, filepath in enumerate(sorted(train_graphml_dir.glob("*.graphml"))[:50]):  # Sample 50 graphs
        graph = nx.read_graphml(filepath)
        if graph.is_directed():
            graph = graph.to_undirected()
        graph = nx.convert_node_labels_to_integers(graph)
        sample_graphs.append((str(i), graph))

    # Tokenize samples and collect all unique tokens
    for name, graph in sample_graphs:
        tokenized_data = task.tokenize_graph(graph, name)
        for samples_list in tokenized_data.values():
            for sample_str in samples_list:
                tokens = sample_str.split()
                for token in tokens:
                    if token not in vocab:
                        vocab[token] = len(vocab)

    vocab_size = len(vocab)
    print(f"Vocab size: {vocab_size} (built from {len(sample_graphs)} sample graphs)")

    # Load datasets
    print("\n=== Loading Splits ===")
    train_dataset = GraphTokenDataset(os.path.join(args.base_dir, 'train'), vocab, task)
    valid_dataset = GraphTokenDataset(os.path.join(args.base_dir, 'valid'), vocab, task)
    test_dataset = GraphTokenDataset(os.path.join(args.base_dir, 'test'), vocab, task)

    print(f"Train: {len(train_dataset)} examples")
    print(f"Valid: {len(valid_dataset)} examples")
    print(f"Test: {len(test_dataset)} examples")

    # Create dataloaders
    pad_token_id = vocab['<pad>']
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=lambda b: collate_fn(b, pad_token_id))
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                              collate_fn=lambda b: collate_fn(b, pad_token_id))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             collate_fn=lambda b: collate_fn(b, pad_token_id))

    # Create model
    print(f"\n=== Creating graph-token Model ===")
    model = TinyTransformer(vocab_size=vocab_size, d_model=32, nhead=4,
                           num_layers=4, num_classes=num_classes).to(device)

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

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion,
                                            device, step_metrics, epoch+1)
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

    # Save results
    results_dir = Path("results/generalization")
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{args.task}_graph-token_{timestamp}"

    results_json = {
        'metadata': {
            'timestamp': timestamp,
            'task': args.task,
            'model_type': 'graph-token',
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
            'train_size': len(train_dataset),
            'valid_size': len(valid_dataset),
            'test_size': len(test_dataset),
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
