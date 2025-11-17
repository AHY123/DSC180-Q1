"""Sequence model adapter with tokenization support."""

import torch
import torch.nn as nn
from typing import Any, List, Dict

from ..core.base_model import BaseModel
from ..core.registry import register_model


@register_model("sequence")
class SequenceAdapter(BaseModel):
    """Sequence model adapter with graph tokenization.
    
    Converts PyG graphs to token sequences and processes them
    with transformer-based sequence models.
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.vocab_size = config.get('vocab_size', 1000)
        self.hidden_dim = config.get('hidden_dim', 256)
        self.num_layers = config.get('num_layers', 4)
        self.max_seq_len = config.get('max_seq_len', 512)
        
        # Special tokens
        self.vocab = self._create_vocab()
        
        # Initialize sequence model (dummy transformer)
        self.model = self._create_sequence_model()
        
    def _create_vocab(self) -> Dict[str, int]:
        """Create vocabulary with special tokens."""
        return {
            'PAD': 0,
            'CLS': 1,
            'SEP': 2,
            'MASK': 3,
            # Node and edge tokens start from 4
        }
    
    def _create_sequence_model(self):
        """Create transformer-based sequence model."""
        return nn.Sequential(
            nn.Embedding(self.vocab_size, self.hidden_dim),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.hidden_dim,
                    nhead=8,
                    dim_feedforward=self.hidden_dim * 4,
                    batch_first=True
                ),
                num_layers=self.num_layers
            ),
            nn.Linear(self.hidden_dim, 2)  # Binary classification
        )
    
    def _preprocess_batch(self, batch):
        """Convert PyG batch to token sequences.
        
        Args:
            batch: PyG Batch object or list of Data objects
            
        Returns:
            Padded tensor of token sequences [batch_size, seq_len]
        """
        if isinstance(batch, list):
            sequences = [self._tokenize_graph(data) for data in batch]
        else:
            # Handle batch object - convert to individual graphs first
            sequences = []
            # Dummy: assume batch contains single graph for now
            sequences.append(self._tokenize_graph(batch))
        
        # Pad sequences to same length
        return self._pad_sequences(sequences)
    
    def _tokenize_graph(self, data) -> torch.Tensor:
        """Convert single graph to token sequence with task embedding.
        
        Args:
            data: PyG Data object
            
        Returns:
            Token sequence tensor
        """
        tokens = [self.vocab['CLS']]
        
        # Add node tokens (map node indices to vocab range)
        if hasattr(data, 'x') and data.x is not None:
            node_tokens = (data.x.squeeze() % (self.vocab_size - 10) + 4).long()
            tokens.extend(node_tokens.tolist())
        
        tokens.append(self.vocab['SEP'])
        
        # Add edge tokens
        if hasattr(data, 'edge_index'):
            edge_tokens = self._edges_to_tokens(data.edge_index)
            tokens.extend(edge_tokens)
        
        tokens.append(self.vocab['SEP'])
        
        # Task-specific tokens
        if hasattr(data, 'src') and hasattr(data, 'dst'):
            # Shortest path: add src/dst tokens
            src_token = (data.src.item() % (self.vocab_size - 10) + 4)
            dst_token = (data.dst.item() % (self.vocab_size - 10) + 4)
            tokens.extend([src_token, dst_token, self.vocab['SEP']])
        
        # Truncate if too long
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
            
        return torch.tensor(tokens, dtype=torch.long)
    
    def _edges_to_tokens(self, edge_index) -> List[int]:
        """Convert edge_index to sequence of edge tokens.
        
        Args:
            edge_index: Edge index tensor [2, num_edges]
            
        Returns:
            List of edge tokens
        """
        tokens = []
        max_edges = 100  # Limit number of edges to include
        
        num_edges = min(edge_index.size(1), max_edges)
        for i in range(num_edges):
            src = edge_index[0, i].item()
            dst = edge_index[1, i].item()
            # Map edge endpoints to vocab range
            src_token = src % (self.vocab_size - 10) + 4
            dst_token = dst % (self.vocab_size - 10) + 4
            tokens.extend([src_token, dst_token])
            
        return tokens
    
    def _pad_sequences(self, sequences: List[torch.Tensor]) -> torch.Tensor:
        """Pad sequences to same length.
        
        Args:
            sequences: List of token sequence tensors
            
        Returns:
            Padded tensor [batch_size, max_len]
        """
        max_len = min(max(len(seq) for seq in sequences), self.max_seq_len)
        batch_size = len(sequences)
        
        padded = torch.full((batch_size, max_len), self.vocab['PAD'], dtype=torch.long)
        
        for i, seq in enumerate(sequences):
            seq_len = min(len(seq), max_len)
            padded[i, :seq_len] = seq[:seq_len]
            
        return padded
    
    def forward(self, batch):
        """Forward pass through sequence model.
        
        Args:
            batch: Input batch
            
        Returns:
            Model predictions
        """
        # Convert to token sequences
        token_sequences = self._preprocess_batch(batch)
        
        # Forward through transformer
        embeddings = self.model[0](token_sequences)  # Embedding layer
        transformer_out = self.model[1](embeddings)  # Transformer
        
        # Global pooling (use CLS token or mean pooling)
        pooled = transformer_out.mean(dim=1)  # Mean pooling
        
        # Classification head
        logits = self.model[2](pooled)
        
        return logits
    
    def loss(self, predictions, targets):
        """Compute loss for predictions and targets."""
        if targets.dim() == 0:
            targets = targets.unsqueeze(0)
            
        return torch.nn.functional.cross_entropy(predictions, targets)
    
    def get_embeddings(self, batch):
        """Extract graph embeddings from the model."""
        token_sequences = self._preprocess_batch(batch)
        embeddings = self.model[0](token_sequences)
        transformer_out = self.model[1](embeddings)
        
        # Return pooled embeddings
        return transformer_out.mean(dim=1)
    
    @property
    def supported_tasks(self) -> List[str]:
        """List of tasks this model supports."""
        return ['graph_classification', 'cycle_detection', 'shortest_path']