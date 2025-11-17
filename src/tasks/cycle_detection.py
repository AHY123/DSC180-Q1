"""Cycle detection task implementation."""

import torch
import networkx as nx
from torch_geometric.utils import to_networkx
from typing import Dict, List

from ..core.base_task import BaseTask
from ..core.registry import register_task


@register_task("cycle_detection")
class CycleDetectionTask(BaseTask):
    """Cycle detection task - binary classification.
    
    Determines whether a graph contains a cycle. Uses NetworkX
    forest detection (forests are acyclic).
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.name = "cycle_detection"
        self.task_type = "classification"
        self._processed_cache = {}
    
    def prepare_data(self, dataset) -> List:
        """Add cycle detection labels to universal graphs.
        
        Args:
            dataset: UniversalSyntheticDataset instance
            
        Returns:
            List of PyG Data objects with cycle labels
        """
        cache_key = f"{dataset.cache_path}_cycle"
        if cache_key in self._processed_cache:
            return self._processed_cache[cache_key]
            
        print("Preparing cycle detection task data...")
        universal_data = dataset.load_data()
        processed_data = []
        
        cycle_count = 0
        for i, data in enumerate(universal_data):
            # Convert back to networkx for cycle detection
            nx_graph = to_networkx(data, to_undirected=True)
            
            # Generate task label - check if graph has cycle
            has_cycle = not nx.is_forest(nx_graph)
            data.y = torch.tensor([1 if has_cycle else 0], dtype=torch.long)
            
            if has_cycle:
                cycle_count += 1
            
            processed_data.append(data)
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(universal_data)} graphs for cycle detection")
        
        print(f"Cycle detection: {cycle_count}/{len(processed_data)} graphs have cycles "
              f"({100 * cycle_count / len(processed_data):.1f}%)")
        
        self._processed_cache[cache_key] = processed_data
        return processed_data
    
    def evaluate(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Evaluate cycle detection predictions.
        
        Args:
            predictions: Model predictions (logits or probabilities)
            targets: Ground truth labels (0 or 1)
            
        Returns:
            Dictionary of metrics
        """
        # Handle different prediction formats
        if predictions.dim() == 2 and predictions.size(1) == 2:
            # Classification logits [batch_size, 2]
            pred_labels = predictions.argmax(dim=1)
        elif predictions.dim() == 2 and predictions.size(1) == 1:
            # Single output [batch_size, 1]
            pred_labels = (predictions.squeeze() > 0.5).long()
        else:
            # Single output [batch_size]
            pred_labels = (predictions > 0.5).long()
        
        # Ensure targets are the right shape
        if targets.dim() == 2:
            targets = targets.squeeze()
        
        # Calculate metrics
        correct = (pred_labels == targets).float()
        accuracy = correct.mean().item()
        
        # Calculate precision, recall, F1 for binary classification
        true_positives = ((pred_labels == 1) & (targets == 1)).float().sum()
        false_positives = ((pred_labels == 1) & (targets == 0)).float().sum()
        false_negatives = ((pred_labels == 0) & (targets == 1)).float().sum()
        
        precision = true_positives / (true_positives + false_positives + 1e-8)
        recall = true_positives / (true_positives + false_negatives + 1e-8)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        return {
            "accuracy": accuracy,
            "precision": precision.item(),
            "recall": recall.item(),
            "f1_score": f1_score.item()
        }
    
    @property
    def metric_names(self) -> List[str]:
        """List of metrics computed by this task."""
        return ["accuracy", "precision", "recall", "f1_score"]
    
    @property
    def primary_metric(self) -> str:
        """Primary metric for model selection."""
        return "accuracy"
    
    @property
    def num_classes(self) -> int:
        """Number of classes for cycle detection."""
        return 2