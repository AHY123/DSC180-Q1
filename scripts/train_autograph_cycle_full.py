"""Train AutoGraph tokenization model on cycle detection with tiny transformer."""

import sys
import os
sys.path.insert(0, 'external/AutoGraph')
# Add project root to path so 'src.' imports work
sys.path.insert(0, os.getcwd())

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import networkx as nx
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import argparse

# Import AutoGraph tokenizer
from autograph.datamodules.data.tokenizer import Graph2TrailTokenizer

# Import our universal dataset and tasks
from src.datasets.synthetic.universal_graphs import UniversalSyntheticDataset
from src.tasks.cycle_detection import CycleDetectionTask


class TinyTransformer(nn.Module):
    """Tiny transformer for graph tasks (Tier 1: ~100K params)."""

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

        # Embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.pos_embedding(positions)
        x = token_emb + pos_emb

        # Transformer
        if attention_mask is not None:
            attention_mask = attention_mask.bool()
            src_key_padding_mask = ~attention_mask
        else:
            src_key_padding_mask = None

        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        # Global pooling
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            x = x.mean(dim=1)

        # Classification
        logits = self.classifier(x)
        return logits


class AutoGraphDataset(Dataset):
    """Dataset that tokenizes graphs using AutoGraph's Graph2TrailTokenizer."""

    def __init__(self, pyg_data_list: List, tokenizer: Graph2TrailTokenizer):
        self.pyg_data_list = pyg_data_list
        self.tokenizer = tokenizer
        print(f"Created AutoGraph dataset with {len(pyg_data_list)} graphs")

    def __len__(self):
        return len(self.pyg_data_list)

    def __getitem__(self, idx):
        data = self.pyg_data_list[idx]

        # Tokenize using AutoGraph
        token_ids = self.tokenizer.tokenize(data)

        # Extract label (added by CycleDetectionTask)
        label = data.y.item() if data.y.dim() == 1 else data.y[0].item()

        return token_ids, label


def collate_fn(batch, pad_token_id=5):  # AutoGraph's pad token is 5
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


def train_epoch(model, dataloader, optimizer, criterion, device):
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


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/synthetic_er/train',
                        help='Directory containing graph data')
    parser.add_argument('--num_graphs', type=int, default=500,
                        help='Number of graphs to use')
    args = parser.parse_args()

    # Configuration
    DATA_DIR = args.data_dir
    BATCH_SIZE = 8
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.001
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {DEVICE}")
    print(f"Data directory: {DATA_DIR}")

    # Load universal synthetic dataset
    print("\n=== Loading Universal Synthetic Dataset ===")
    dataset_name = 'er_test' if 'test' in DATA_DIR else 'er'
    cache_name = f'er_{args.num_graphs}graphs_autograph.pt'
    dataset_config = {
        'name': dataset_name,
        'graph_sources': [DATA_DIR],
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
    print(f"Labeled {len(labeled_graphs)} graphs with cycle detection labels")

    # Initialize AutoGraph tokenizer
    print("\n=== Initializing AutoGraph Tokenizer ===")
    # Find max number of nodes
    max_nodes = max(data.num_nodes for data in labeled_graphs)
    print(f"Max nodes in dataset: {max_nodes}")

    tokenizer = Graph2TrailTokenizer(
        max_num_nodes=max_nodes,
        undirected=True,
        append_eos=True,
        max_length=-1,  # No truncation
    )
    tokenizer.set_num_nodes(max_nodes)

    vocab_size = len(tokenizer)
    print(f"AutoGraph vocabulary size: {vocab_size}")

    # Tokenize a sample graph to check
    print("\n=== Sample Tokenization ===")
    sample_tokens = tokenizer.tokenize(labeled_graphs[0])
    print(f"Sample graph tokenized to {len(sample_tokens)} tokens")
    print(f"Sample tokens (first 20): {sample_tokens[:20]}")

    # Create dataset
    print("\n=== Creating Dataset ===")
    dataset = AutoGraphDataset(labeled_graphs, tokenizer)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, pad_token_id=tokenizer.pad)
    )

    # Create model
    print("\n=== Creating Model ===")
    model = TinyTransformer(
        vocab_size=vocab_size,
        d_model=32,
        nhead=4,
        num_layers=4,
        num_classes=2
    ).to(DEVICE)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    print(f"\n=== Training for {NUM_EPOCHS} epochs ===")
    best_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_epoch(model, dataloader, optimizer, criterion, DEVICE)

        if (epoch + 1) % 10 == 0 or epoch < 5:
            print(f"Epoch {epoch+1:3d}/{NUM_EPOCHS} | "
                  f"Loss: {train_loss:.4f} | "
                  f"Acc: {train_acc*100:.2f}%")

        if train_acc > best_acc:
            best_acc = train_acc

        # Early stopping if we've memorized the dataset
        if train_acc >= 0.98:
            print(f"\n✓ Model has memorized the training set (acc={train_acc*100:.2f}%)")
            print(f"Stopping early at epoch {epoch+1}")
            break

    print(f"\nBest training accuracy: {best_acc*100:.2f}%")

    # Save model
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'tokenizer_config': {
            'max_num_nodes': max_nodes,
            'vocab_size': vocab_size
        },
        'config': {
            'd_model': 32,
            'nhead': 4,
            'num_layers': 4
        }
    }, "checkpoints/autograph_cycle_tiny_full.pt")
    print("Model saved to checkpoints/autograph_cycle_tiny_full.pt")


if __name__ == "__main__":
    main()
