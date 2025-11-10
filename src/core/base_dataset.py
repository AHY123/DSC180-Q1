"""Base dataset interface for all graph datasets."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import torch


class BaseDataset(ABC):
    """Abstract base class for all graph datasets."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the dataset with configuration.
        
        Args:
            config: Dataset configuration dictionary
        """
        self.config = config
        self.name = config.get('name', 'unnamed')
        self.data_dir = config.get('data_dir', 'data')
        
    @abstractmethod
    def load_data(self) -> Any:
        """Load the raw dataset.
        
        Returns:
            Loaded dataset object
        """
        pass
    
    @abstractmethod
    def get_splits(self) -> Tuple[Any, Any, Any]:
        """Get train/validation/test splits.
        
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        pass
    
    @property
    @abstractmethod
    def num_features(self) -> int:
        """Number of input features per node.
        
        Returns:
            Feature dimension
        """
        pass
    
    @property
    @abstractmethod
    def num_classes(self) -> int:
        """Number of target classes.
        
        Returns:
            Number of classes
        """
        pass
    
    @property
    @abstractmethod
    def num_graphs(self) -> int:
        """Total number of graphs in dataset.
        
        Returns:
            Number of graphs
        """
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics.
        
        Returns:
            Dictionary with dataset statistics
        """
        return {
            'name': self.name,
            'num_features': self.num_features,
            'num_classes': self.num_classes,
            'num_graphs': self.num_graphs
        }