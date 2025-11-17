"""Shortest path distance task implementation."""

import torch
import networkx as nx
import random
from torch_geometric.utils import to_networkx
from typing import Dict, List, Tuple

from ..core.base_task import BaseTask
from ..core.registry import register_task


@register_task("shortest_path")
class ShortestPathTask(BaseTask):
    """Shortest path distance task - regression.
    
    Predicts the shortest path distance between two randomly
    selected nodes in the graph.
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.name = "shortest_path"
        self.task_type = "regression"
        self.max_distance = config.get('max_distance', 10)
        self.max_attempts = config.get('max_attempts', 50)
        self._processed_cache = {}
        
        # Set random seed for reproducible node pair selection
        self.random_seed = config.get('random_seed', 42)
        random.seed(self.random_seed)
    
    def prepare_data(self, dataset) -> List:
        """Add shortest path task data to universal graphs.
        
        Args:
            dataset: UniversalSyntheticDataset instance
            
        Returns:
            List of PyG Data objects with src, dst, and distance labels
        """
        cache_key = f"{dataset.cache_path}_shortest_path"
        if cache_key in self._processed_cache:
            return self._processed_cache[cache_key]
            
        print("Preparing shortest path task data...")
        universal_data = dataset.load_data()
        processed_data = []
        
        skipped_count = 0
        distance_stats = []
        
        for i, data in enumerate(universal_data):
            # Convert back to networkx for shortest path computation
            nx_graph = to_networkx(data, to_undirected=True)
            
            # Sample source and destination nodes
            src, dst, distance = self._sample_node_pair_with_distance(nx_graph)
            
            if distance is not None and distance <= self.max_distance:
                # Add task-specific attributes
                data.src = torch.tensor([src], dtype=torch.long)
                data.dst = torch.tensor([dst], dtype=torch.long)
                data.y = torch.tensor([distance], dtype=torch.float)
                
                processed_data.append(data)
                distance_stats.append(distance)
            else:
                skipped_count += 1
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(universal_data)} graphs for shortest path")
        
        print(f"Shortest path: {len(processed_data)} valid graphs, {skipped_count} skipped")
        if distance_stats:
            print(f"Distance stats: mean={torch.tensor(distance_stats).float().mean():.2f}, "
                  f"min={min(distance_stats)}, max={max(distance_stats)}")
        
        self._processed_cache[cache_key] = processed_data
        return processed_data
    
    def _sample_node_pair_with_distance(self, nx_graph: nx.Graph) -> Tuple[int, int, float]:
        """Sample a pair of nodes and compute shortest path distance.
        
        Args:
            nx_graph: NetworkX graph
            
        Returns:
            Tuple of (src_node, dst_node, distance) or (None, None, None) if failed
        """
        nodes = list(nx_graph.nodes())
        if len(nodes) < 2:
            return None, None, None
        
        for _ in range(self.max_attempts):
            src, dst = random.sample(nodes, 2)
            
            try:
                distance = nx.shortest_path_length(nx_graph, src, dst)
                if distance <= self.max_distance:
                    return src, dst, float(distance)
            except nx.NetworkXNoPath:
                continue  # Try again with different nodes
        
        # Failed to find valid pair within max_attempts
        return None, None, None
    
    def evaluate(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Evaluate shortest path distance predictions.
        
        Args:
            predictions: Model predictions (distances)
            targets: Ground truth distances
            
        Returns:
            Dictionary of regression metrics
        """
        # Ensure predictions and targets are the right shape
        if predictions.dim() == 2:
            predictions = predictions.squeeze()
        if targets.dim() == 2:
            targets = targets.squeeze()
        
        # Calculate regression metrics
        mse = torch.mean((predictions - targets) ** 2)
        mae = torch.mean(torch.abs(predictions - targets))
        rmse = torch.sqrt(mse)
        
        # Calculate R²
        ss_res = torch.sum((targets - predictions) ** 2)
        ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-8)
        
        # Accuracy within tolerance
        tolerance = self.config.get('distance_tolerance', 0.5)
        within_tolerance = (torch.abs(predictions - targets) <= tolerance).float().mean()
        
        return {
            "mse": mse.item(),
            "mae": mae.item(), 
            "rmse": rmse.item(),
            "r2": r2.item(),
            "accuracy_within_tolerance": within_tolerance.item()
        }
    
    @property
    def metric_names(self) -> List[str]:
        """List of metrics computed by this task."""
        return ["mse", "mae", "rmse", "r2", "accuracy_within_tolerance"]
    
    @property
    def primary_metric(self) -> str:
        """Primary metric for model selection."""
        return "mae"  # Lower is better for MAE
    
    def is_better(self, new_score: float, best_score: float) -> bool:
        """Check if new score is better (lower MAE is better)."""
        return new_score < best_score
    
    @property
    def num_classes(self) -> int:
        """Not applicable for regression task."""
        return 1  # Single continuous output