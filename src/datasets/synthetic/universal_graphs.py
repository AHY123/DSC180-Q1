"""Universal synthetic graph dataset implementation."""

import os
import glob
from typing import List, Any
import torch
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

from src.core.base_dataset import BaseDataset
from src.core.registry import register_dataset


@register_dataset("universal_synthetic")
class UniversalSyntheticDataset(BaseDataset):
    """Universal synthetic graph dataset - neutral PyG format.
    
    Loads networkx graphs from GraphML files and converts them to
    PyTorch Geometric format with basic node features. This serves
    as the universal base for all model-specific preprocessing.
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.graph_sources = config.get('graph_sources', [])
        self.cache_path = config.get('cache_path', 'data/processed/universal_synthetic.pt')
        self.max_graphs = config.get('max_graphs', None)
        self._data_cache = None
    
    def load_data(self) -> List[Data]:
        """Load universal synthetic dataset.
        
        Returns:
            List of PyG Data objects with basic node features
        """
        if self._data_cache is not None:
            return self._data_cache
            
        if os.path.exists(self.cache_path):
            print(f"Loading cached universal dataset from {self.cache_path}")
            self._data_cache = torch.load(self.cache_path, weights_only=False)
            return self._data_cache
        
        print("Generating universal synthetic dataset...")
        data_list = []
        
        for source_path in self.graph_sources:
            nx_graphs = self._load_graphml_files(source_path)
            for i, nx_graph in enumerate(nx_graphs):
                if self.max_graphs and len(data_list) >= self.max_graphs:
                    break
                    
                pyg_data = self._convert_to_pyg(nx_graph)
                data_list.append(pyg_data)
                
                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1} graphs from {source_path}")
        
        print(f"Generated {len(data_list)} graphs total")
        
        # Cache the results
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        torch.save(data_list, self.cache_path)
        
        self._data_cache = data_list
        return data_list
    
    def _load_graphml_files(self, source_path: str) -> List[nx.Graph]:
        """Load networkx graphs from GraphML files.
        
        Args:
            source_path: Path to directory containing .graphml files
            
        Returns:
            List of networkx graphs
        """
        if not os.path.exists(source_path):
            print(f"Warning: Source path {source_path} does not exist")
            return []
            
        graphml_files = glob.glob(os.path.join(source_path, "*.graphml"))
        if not graphml_files:
            print(f"Warning: No .graphml files found in {source_path}")
            return []
            
        print(f"Found {len(graphml_files)} .graphml files in {source_path}")
        
        graphs = []
        for file_path in graphml_files:
            try:
                nx_graph = nx.read_graphml(file_path)
                # Convert to undirected if needed
                if nx_graph.is_directed():
                    nx_graph = nx_graph.to_undirected()
                graphs.append(nx_graph)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
                
        return graphs
    
    def _convert_to_pyg(self, nx_graph: nx.Graph) -> Data:
        """Convert networkx graph to PyG Data with basic features.
        
        Args:
            nx_graph: NetworkX graph
            
        Returns:
            PyG Data object with node indices as features
        """
        # Convert to PyG format
        data = from_networkx(nx_graph)
        
        # Add simple node features (node indices)
        num_nodes = nx_graph.number_of_nodes()
        data.x = torch.arange(num_nodes, dtype=torch.float).unsqueeze(1)
        
        # Add basic graph statistics as attributes
        data.num_nodes = num_nodes
        data.num_edges = nx_graph.number_of_edges()
        
        return data
    
    def get_splits(self):
        """Get train/validation/test splits.
        
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        data_list = self.load_data()
        
        # Default split ratios
        train_ratio = self.config.get('train_ratio', 0.8)
        val_ratio = self.config.get('val_ratio', 0.1)
        
        n_total = len(data_list)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        # Shuffle data if requested
        if self.config.get('shuffle', True):
            indices = torch.randperm(n_total)
            data_list = [data_list[i] for i in indices]
        
        train_data = data_list[:n_train]
        val_data = data_list[n_train:n_train + n_val]
        test_data = data_list[n_train + n_val:]
        
        return train_data, val_data, test_data
    
    @property
    def num_features(self) -> int:
        """Number of input features per node."""
        return 1  # Simple node indices
    
    @property
    def num_classes(self) -> int:
        """Number of target classes (task-dependent)."""
        return -1  # Will be set by specific tasks
    
    @property
    def num_graphs(self) -> int:
        """Total number of graphs in dataset."""
        return len(self.load_data())