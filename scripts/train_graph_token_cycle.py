"""Train graph-token model on cycle detection with tiny transformer."""

import sys
import os
sys.path.insert(0, 'external/graph-token')
sys.path.insert(0, 'src')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import networkx as nx
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

# Import graph-token's cycle check task
from graph_task import CycleCheck


class TinyTransformer(nn.Module):
    """Tiny transformer for graph tasks (Tier 1: ~100K params)."""

    def __init__(self, vocab_size, d_model=32, nhead=4, num_layers=4, num_classes=2):
        super().__init__()
        self.d_model = d_model

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(512, d_model)  # Max sequence length

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
        # input_ids: [batch_size, seq_len]
        batch_size, seq_len = input_ids.shape

        # Embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.pos_embedding(positions)
        x = token_emb + pos_emb

        # Transformer
        if attention_mask is not None:
            # Convert mask to transformer format (True = attend, False = ignore)
            attention_mask = attention_mask.bool()
            # Invert for transformer (expects padding positions to be True)
            src_key_padding_mask = ~attention_mask
        else:
            src_key_padding_mask = None

        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        # Global pooling (mean over sequence)
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            x = x.mean(dim=1)

        # Classification
        logits = self.classifier(x)
        return logits


class GraphTokenDataset(Dataset):
    """Dataset that tokenizes graphs using graph-token's CycleCheck."""

    def __init__(self, graphml_dir: str, vocab: Dict[str, int]):
        self.graphml_dir = Path(graphml_dir)
        self.vocab = vocab
        self.task = CycleCheck()

        # Load all graphs
        self.graphs = []
        graphml_files = sorted(self.graphml_dir.glob("*.graphml"))
        print(f"Loading {len(graphml_files)} graphs from {graphml_dir}...")

        for i, filepath in enumerate(graphml_files):
            graph = nx.read_graphml(filepath)
            # Convert to undirected if needed
            if graph.is_directed():
                graph = graph.to_undirected()
            # Ensure nodes are labeled 0, 1, 2, ...
            graph = nx.convert_node_labels_to_integers(graph)
            self.graphs.append((str(i), graph))

        print(f"Loaded {len(self.graphs)} graphs")

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph_id, graph = self.graphs[idx]

        # Tokenize using graph-token's CycleCheck
        tokenized = self.task.tokenize_graph(graph, graph_id)
        text = tokenized[graph_id][0]  # Returns {graph_id: [text_string]}

        # Convert text to token IDs
        tokens = text.split()
        token_ids = [self.vocab.get(token, self.vocab['<unk>']) for token in tokens]

        # Extract label from the tokenized string
        # Format: "... <p> yes <eos>" or "... <p> no <eos>"
        p_idx = tokens.index('<p>')
        label_token = tokens[p_idx + 1]
        label = 1 if label_token == 'yes' else 0

        return torch.tensor(token_ids, dtype=torch.long), label


def build_vocab(graphml_dir: str) -> Dict[str, int]:
    """Build vocabulary from all tokenized graphs."""
    task = CycleCheck()
    vocab_set = set()

    graphml_files = sorted(Path(graphml_dir).glob("*.graphml"))
    print(f"Building vocabulary from {len(graphml_files)} graphs...")

    for i, filepath in enumerate(graphml_files):
        graph = nx.read_graphml(filepath)
        if graph.is_directed():
            graph = graph.to_undirected()
        graph = nx.convert_node_labels_to_integers(graph)

        tokenized = task.tokenize_graph(graph, str(i))
        text = tokenized[str(i)][0]
        tokens = text.split()
        vocab_set.update(tokens)

    # Create vocab dict with special tokens
    vocab = {'<pad>': 0, '<unk>': 1}
    for i, token in enumerate(sorted(vocab_set), start=2):
        vocab[token] = i

    print(f"Vocabulary size: {len(vocab)}")
    print(f"Sample tokens: {list(vocab.keys())[:20]}")

    return vocab


def collate_fn(batch, pad_token_id=0):
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
    # Configuration
    DATA_DIR = "data/synthetic_er_test/train"
    BATCH_SIZE = 8
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.001
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {DEVICE}")
    print(f"Data directory: {DATA_DIR}")

    # Build vocabulary
    vocab = build_vocab(DATA_DIR)

    # Create dataset
    dataset = GraphTokenDataset(DATA_DIR, vocab)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, pad_token_id=vocab['<pad>'])
    )

    # Create model
    model = TinyTransformer(
        vocab_size=len(vocab),
        d_model=32,
        nhead=4,
        num_layers=4,
        num_classes=2
    ).to(DEVICE)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    print(f"\nTraining for {NUM_EPOCHS} epochs...")
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
        'vocab': vocab,
        'config': {
            'd_model': 32,
            'nhead': 4,
            'num_layers': 4
        }
    }, "checkpoints/graph_token_cycle_tiny.pt")
    print("Model saved to checkpoints/graph_token_cycle_tiny.pt")


if __name__ == "__main__":
    main()
