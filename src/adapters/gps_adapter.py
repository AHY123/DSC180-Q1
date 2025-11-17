"""GPS model adapter with preprocessing support."""

import sys
import os
import torch
from typing import Any, List

# Add external GPS path for imports (when submodule is added)
GPS_PATH = os.path.join(os.path.dirname(__file__), '../../external/GraphGPS')
if os.path.exists(GPS_PATH):
    sys.path.append(GPS_PATH)

from ..core.base_model import BaseModel
from ..core.registry import register_model


@register_model("gps")
class GPSAdapter(BaseModel):
    """GPS model adapter with Laplacian positional encoding.
    
    Wraps the external GPS implementation and handles preprocessing
    like adding positional encodings to graph data.
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.hidden_dim = config.get('hidden_dim', 256)
        self.num_layers = config.get('num_layers', 6)
        self.pe_dim = config.get('pe_dim', 16)
        
        # Initialize PE transform (dummy implementation)
        self.pe_transform = self._create_pe_transform()
        
        # Initialize GPS model (dummy - will be replaced with actual GPS)
        self.model = self._create_gps_model()
        
    def _create_pe_transform(self):
        """Create Laplacian positional encoding transform."""
        try:
            from torch_geometric.transforms import LaplacianEigenvectorPE
            return LaplacianEigenvectorPE(k=self.pe_dim, attr_name='pe')
        except ImportError:
            print("Warning: LaplacianEigenvectorPE not available, using dummy PE")
            return self._dummy_pe_transform
    
    def _dummy_pe_transform(self, data):
        """Dummy positional encoding for testing."""
        if not hasattr(data, 'pe'):
            # Add random PE features as placeholder
            data.pe = torch.randn(data.num_nodes, self.pe_dim)
        return data
    
    def _create_gps_model(self):
        """Create GPS model instance."""
        # Dummy implementation - replace with actual GPS model
        return torch.nn.Sequential(
            torch.nn.Linear(1 + self.pe_dim, self.hidden_dim),  # x + pe
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, 2)  # Binary classification
        )
    
    def _preprocess_batch(self, batch):
        """Add GPS-specific features like LaplacianPE.
        
        Args:
            batch: PyG Batch object or list of Data objects
            
        Returns:
            Processed batch with PE features
        """
        if isinstance(batch, list):
            # Process individual Data objects
            processed = []
            for data in batch:
                if not hasattr(data, 'pe'):
                    data = self.pe_transform(data)
                processed.append(data)
            return processed
        else:
            # Process batch object
            if not hasattr(batch, 'pe'):
                # Apply PE to each graph in batch
                batch = self.pe_transform(batch)
            return batch
    
    def forward(self, batch):
        """Forward pass through GPS model.
        
        Args:
            batch: Input batch
            
        Returns:
            Model predictions
        """
        batch = self._preprocess_batch(batch)
        
        # Dummy forward pass - replace with actual GPS forward
        if hasattr(batch, 'x') and hasattr(batch, 'pe'):
            # Concatenate node features with PE
            x = torch.cat([batch.x, batch.pe], dim=-1)
            # Simple linear layers for demo
            return self.model(x).mean(dim=0, keepdim=True)  # Global pooling
        else:
            # Fallback for individual data objects
            return torch.randn(1, 2)  # Dummy output
    
    def loss(self, predictions, targets):
        """Compute loss for predictions and targets."""
        if predictions.dim() == 1:
            predictions = predictions.unsqueeze(0)
        if targets.dim() == 0:
            targets = targets.unsqueeze(0)
            
        return torch.nn.functional.cross_entropy(predictions, targets)
    
    def get_embeddings(self, batch):
        """Extract graph embeddings from the model."""
        batch = self._preprocess_batch(batch)
        
        # Dummy embeddings - replace with actual GPS embeddings
        if hasattr(batch, 'x'):
            return torch.mean(batch.x, dim=0, keepdim=True)
        else:
            return torch.randn(1, self.hidden_dim)
    
    @property
    def supported_tasks(self) -> List[str]:
        """List of tasks this model supports."""
        return ['graph_classification', 'cycle_detection', 'shortest_path']