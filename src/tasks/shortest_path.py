"""Shortest path distance prediction task.

This task takes graphs and creates node-pair shortest path prediction problems.
For each graph, k random connected pairs are selected, and the model must predict
the shortest path distance between them.
"""

import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, degree
from typing import List, Dict, Any, Tuple
import random
import numpy as np

from src.core.base_task import BaseTask
from src.core.registry import register_task


@register_task("shortest_path")
class ShortestPathTask(BaseTask):
    """Shortest path distance prediction task.

    Given a graph and two nodes (source, target), predict the shortest path distance.

    Configuration:
        - k_pairs: Number of node pairs per graph (default: 1)
        - max_distance: Maximum distance to consider, distances >= this are capped (default: 10)
        - connected_only: Only select connected pairs (default: True)
        - output_type: 'classification' or 'regression' (default: 'classification')
        - random_seed: Seed for reproducible pair selection (default: 42)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = "shortest_path"
        self.k_pairs = config.get('k_pairs', 1)
        self.max_distance = config.get('max_distance', 10)
        self.connected_only = config.get('connected_only', True)
        self.output_type = config.get('output_type', 'classification')
        self.random_seed = config.get('random_seed', 42)
        self.task_type = self.output_type

        random.seed(self.random_seed)

    def prepare_data(self, dataset) -> List[Data]:
        """Prepare shortest path data.

        For each graph, selects k random node pairs and creates training examples.
        Each example includes:
        - Original graph structure
        - Source and target node indices
        - Shortest path distance as label
        - Node features indicating source/target (for graph-native models)

        Args:
            dataset: UniversalSyntheticDataset with graphs

        Returns:
            List of Data objects with shortest path labels
        """
        graphs = dataset.load_data()
        labeled_data = []

        print(f"Preparing shortest path task data (k={self.k_pairs} pairs per graph)...")

        distance_counts = {i: 0 for i in range(self.max_distance + 1)}
        skipped = 0

        for idx, pyg_data in enumerate(graphs):
            # Convert to networkx for shortest path computation
            G = to_networkx(pyg_data, to_undirected=True)
            num_nodes = pyg_data.num_nodes

            # Get valid nodes (from largest connected component if needed)
            if self.connected_only:
                if nx.is_connected(G):
                    valid_nodes = list(G.nodes())
                else:
                    # Use largest component
                    components = list(nx.connected_components(G))
                    largest_component = max(components, key=len)
                    valid_nodes = list(largest_component)

                if len(valid_nodes) < 2:
                    skipped += 1
                    continue
            else:
                valid_nodes = list(G.nodes())

            # Sample k pairs
            pairs_created = 0
            attempts = 0
            max_attempts = min(100, len(valid_nodes) * (len(valid_nodes) - 1))

            while pairs_created < self.k_pairs and attempts < max_attempts:
                attempts += 1

                # Select random pair
                source, target = random.sample(valid_nodes, 2)

                # Check if connected
                if not nx.has_path(G, source, target):
                    if not self.connected_only:
                        distance = self.max_distance  # Use max for disconnected
                    else:
                        continue
                else:
                    distance = nx.shortest_path_length(G, source, target)

                # Cap distance
                distance = min(distance, self.max_distance)
                distance_counts[distance] += 1

                # Create new data object with source/target information
                new_data = pyg_data.clone()

                # Store source/target indices
                new_data.source_idx = source
                new_data.target_idx = target

                # Add source/target indicators as node features
                # Format: [original_features, is_source, is_target]
                source_indicators = torch.zeros(num_nodes, 1)
                target_indicators = torch.zeros(num_nodes, 1)
                source_indicators[source] = 1.0
                target_indicators[target] = 1.0

                # If no existing features, create degree features
                if new_data.x is None:
                    deg = degree(new_data.edge_index[0], num_nodes=num_nodes, dtype=torch.float)
                    new_data.x = deg.unsqueeze(1)

                # Concatenate source/target indicators
                new_data.x = torch.cat([
                    new_data.x,
                    source_indicators,
                    target_indicators
                ], dim=1)

                # Set label
                if self.output_type == 'classification':
                    new_data.y = torch.tensor(distance, dtype=torch.long)
                else:  # regression
                    new_data.y = torch.tensor(distance, dtype=torch.float)

                labeled_data.append(new_data)
                pairs_created += 1

            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{len(graphs)} graphs")

        print(f"\nShortest path task statistics:")
        print(f"  Total pairs: {len(labeled_data)}")
        print(f"  Skipped graphs (too small): {skipped}")
        print(f"  Distance distribution:")
        for dist in sorted(distance_counts.keys()):
            count = distance_counts[dist]
            if count > 0:
                pct = 100 * count / len(labeled_data) if len(labeled_data) > 0 else 0
                print(f"    Distance {dist}: {count} pairs ({pct:.1f}%)")

        return labeled_data

    def evaluate(self, predictions: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """Evaluate predictions.

        Args:
            predictions: Model predictions
            labels: Ground truth labels

        Returns:
            Dictionary of metrics
        """
        if self.output_type == 'classification':
            # Classification metrics
            if predictions.dim() > 1:
                predicted_classes = predictions.argmax(dim=1)
            else:
                predicted_classes = predictions

            accuracy = (predicted_classes == labels).float().mean().item()

            # Mean absolute error
            mae = (predicted_classes - labels).abs().float().mean().item()

            # Off-by-one accuracy (within 1 of correct distance)
            off_by_one = ((predicted_classes - labels).abs() <= 1).float().mean().item()

            return {
                'accuracy': accuracy,
                'mae': mae,
                'off_by_one_accuracy': off_by_one
            }
        else:
            # Regression metrics
            mae = (predictions - labels).abs().mean().item()
            mse = ((predictions - labels) ** 2).mean().item()
            rmse = np.sqrt(mse)

            return {
                'mae': mae,
                'mse': mse,
                'rmse': rmse
            }

    @property
    def num_classes(self) -> int:
        """Number of classes for classification."""
        if self.output_type == 'classification':
            return self.max_distance + 1  # 0 to max_distance
        else:
            return 1  # Single continuous output

    @property
    def metric_names(self) -> List[str]:
        """List of metrics computed by this task."""
        if self.output_type == 'classification':
            return ["accuracy", "mae", "off_by_one_accuracy"]
        else:
            return ["mae", "mse", "rmse"]

    @property
    def primary_metric(self) -> str:
        """Primary metric for model selection."""
        if self.output_type == 'classification':
            return "accuracy"
        else:
            return "mae"

    def is_better(self, new_score: float, best_score: float) -> bool:
        """Check if new score is better."""
        if self.output_type == 'classification':
            return new_score > best_score  # Higher accuracy is better
        else:
            return new_score < best_score  # Lower MAE is better
