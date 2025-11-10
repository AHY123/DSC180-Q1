"""Base model interface for all graph learning models."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import torch


class BaseModel(ABC):
    """Abstract base class for all graph learning models."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the model with configuration.
        
        Args:
            config: Model configuration dictionary
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @abstractmethod
    def forward(self, batch) -> Any:
        """Forward pass through the model.
        
        Args:
            batch: Input batch in standardized format
            
        Returns:
            Model output (logits, embeddings, etc.)
        """
        pass
    
    @abstractmethod
    def loss(self, predictions, targets) -> torch.Tensor:
        """Compute loss for the given predictions and targets.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Computed loss tensor
        """
        pass
    
    @abstractmethod
    def get_embeddings(self, batch) -> torch.Tensor:
        """Extract node/graph embeddings from the model.
        
        Args:
            batch: Input batch
            
        Returns:
            Embedding tensor
        """
        pass
    
    @property
    @abstractmethod
    def supported_tasks(self) -> List[str]:
        """List of tasks this model supports.
        
        Returns:
            List of task names (e.g., ['graph_classification', 'node_classification'])
        """
        pass
    
    def to(self, device):
        """Move model to specified device."""
        self.device = device
        return self
    
    def train_mode(self):
        """Set model to training mode."""
        pass
    
    def eval_mode(self):
        """Set model to evaluation mode."""
        pass