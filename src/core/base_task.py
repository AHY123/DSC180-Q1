"""Base task interface for all graph learning tasks."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
import torch


class BaseTask(ABC):
    """Abstract base class for all graph learning tasks."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the task with configuration.
        
        Args:
            config: Task configuration dictionary
        """
        self.config = config
        self.name = config.get('name', 'unnamed')
        self.task_type = config.get('type', 'classification')
        
    @abstractmethod
    def prepare_data(self, dataset) -> Any:
        """Prepare dataset for this specific task.
        
        Args:
            dataset: Base dataset object
            
        Returns:
            Task-specific dataset
        """
        pass
    
    @abstractmethod
    def evaluate(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Evaluate predictions against targets.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Dictionary of metric name -> value
        """
        pass
    
    @property
    @abstractmethod
    def metric_names(self) -> List[str]:
        """List of metrics computed by this task.
        
        Returns:
            List of metric names (e.g., ['accuracy', 'f1_score'])
        """
        pass
    
    @property
    @abstractmethod
    def primary_metric(self) -> str:
        """Primary metric for model selection.
        
        Returns:
            Name of primary metric
        """
        pass
    
    def is_better(self, new_score: float, best_score: float) -> bool:
        """Check if new score is better than current best.
        
        Args:
            new_score: New metric score
            best_score: Current best score
            
        Returns:
            True if new_score is better
        """
        # Default: higher is better (accuracy, f1, etc.)
        # Override for metrics where lower is better (loss, error)
        return new_score > best_score