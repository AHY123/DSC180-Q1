"""AutoGraph model adapter with AutoGraph tokenization."""

import torch
import torch.nn as nn
from typing import Any, List

from ..core.base_model import BaseModel
from ..core.registry import register_model


@register_model("autograph")
class AutoGraphAdapter(BaseModel):
    """AutoGraph model adapter with AutoGraph tokenization.
    
    Implements the AutoGraph tokenization strategy for converting
    graphs to sequences, then processes with transformer models.
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.vocab_size = config.get('vocab_size', 1000)
        self.hidden_dim = config.get('hidden_dim', 256)
        self.num_layers = config.get('num_layers', 4)
        self.max_seq_len = config.get('max_seq_len', 512)
        
        # AutoGraph specific parameters
        self.use_bfs_order = config.get('use_bfs_order', True)
        self.include_node_features = config.get('include_node_features', True)
        
        # Initialize model (dummy implementation)
        self.model = self._create_autograph_model()
        
    def _create_autograph_model(self):
        """Create AutoGraph-style transformer model."""
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
        """Convert PyG batch to AutoGraph token sequences.
        
        Args:
            batch: PyG Batch object or list of Data objects
            
        Returns:
            Padded tensor of AutoGraph token sequences
        """
        if isinstance(batch, list):
            sequences = [self._tokenize_autograph(data) for data in batch]
        else:
            # Handle batch object
            sequences = [self._tokenize_autograph(batch)]
        
        return self._pad_sequences(sequences)
    
    def _tokenize_autograph(self, data) -> torch.Tensor:
        """Convert graph to AutoGraph token sequence.
        
        This is a simplified version of AutoGraph tokenization.
        The full implementation would follow the AutoGraph paper's
        canonical tokenization strategy.
        
        Args:
            data: PyG Data object
            
        Returns:
            AutoGraph token sequence
        """
        tokens = []
        
        # AutoGraph canonical ordering (simplified)
        if self.use_bfs_order:
            node_order = self._get_bfs_order(data)
        else:
            node_order = list(range(data.num_nodes))
        
        # Node sequence with adjacency information
        for node_id in node_order:
            # Add node token
            node_token = (node_id % (self.vocab_size - 10) + 4)
            tokens.append(node_token)
            
            # Add adjacency information (simplified)
            if hasattr(data, 'edge_index'):
                neighbors = self._get_neighbors(data.edge_index, node_id)
                for neighbor in neighbors[:5]:  # Limit neighbors
                    neighbor_token = (neighbor % (self.vocab_size - 10) + 4)
                    tokens.append(neighbor_token)
            
            # Separator between nodes
            tokens.append(2)  # SEP token
        
        # Task-specific additions
        if hasattr(data, 'src') and hasattr(data, 'dst'):
            # For shortest path task
            src_token = (data.src.item() % (self.vocab_size - 10) + 4)
            dst_token = (data.dst.item() % (self.vocab_size - 10) + 4)
            tokens.extend([src_token, dst_token])
        
        # Truncate if too long
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
            
        return torch.tensor(tokens, dtype=torch.long)
    
    def _get_bfs_order(self, data) -> List[int]:
        """Get BFS ordering of nodes (simplified).
        
        Args:
            data: PyG Data object
            
        Returns:
            List of node indices in BFS order
        """
        # Simplified BFS - in practice would use proper graph traversal
        return list(range(data.num_nodes))
    
    def _get_neighbors(self, edge_index, node_id: int) -> List[int]:
        """Get neighbors of a given node.
        
        Args:
            edge_index: Edge index tensor
            node_id: Target node ID
            
        Returns:
            List of neighbor node IDs
        """
        # Find edges where node_id is the source
        mask = edge_index[0] == node_id
        neighbors = edge_index[1, mask].tolist()
        return neighbors
    
    def _pad_sequences(self, sequences: List[torch.Tensor]) -> torch.Tensor:
        """Pad sequences to same length."""
        max_len = min(max(len(seq) for seq in sequences), self.max_seq_len)
        batch_size = len(sequences)
        
        padded = torch.full((batch_size, max_len), 0, dtype=torch.long)  # PAD = 0
        
        for i, seq in enumerate(sequences):
            seq_len = min(len(seq), max_len)
            padded[i, :seq_len] = seq[:seq_len]
            
        return padded
    
    def forward(self, batch):
        """Forward pass through AutoGraph model."""
        token_sequences = self._preprocess_batch(batch)
        
        # Forward through transformer
        embeddings = self.model[0](token_sequences)
        transformer_out = self.model[1](embeddings)
        
        # Global pooling
        pooled = transformer_out.mean(dim=1)
        
        # Classification
        logits = self.model[2](pooled)
        
        return logits
    
    def loss(self, predictions, targets):
        """Compute loss for predictions and targets."""
        if targets.dim() == 0:
            targets = targets.unsqueeze(0)
            
        return torch.nn.functional.cross_entropy(predictions, targets)
    
    def get_embeddings(self, batch):
        """Extract graph embeddings."""
        token_sequences = self._preprocess_batch(batch)
        embeddings = self.model[0](token_sequences)
        transformer_out = self.model[1](embeddings)
        
        return transformer_out.mean(dim=1)
    
    @property
    def supported_tasks(self) -> List[str]:
        """List of tasks this model supports."""
        return ['graph_classification', 'cycle_detection', 'shortest_path']