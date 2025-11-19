"""Synthetic data generation utilities."""

import os
import argparse
import networkx as nx
import torch
from typing import List, Tuple

from src.core.registry import dataset_registry, task_registry


def generate_test_graphs(output_dir: str, num_graphs: int = 100, graph_types: List[str] = None):
    """Generate test graphs for development and testing.
    
    Args:
        output_dir: Directory to save GraphML files
        num_graphs: Number of graphs to generate per type
        graph_types: Types of graphs to generate
    """
    if graph_types is None:
        graph_types = ['erdos_renyi', 'barabasi_albert', 'cycle', 'tree']
    
    os.makedirs(output_dir, exist_ok=True)
    
    for graph_type in graph_types:
        type_dir = os.path.join(output_dir, graph_type)
        os.makedirs(type_dir, exist_ok=True)
        
        print(f"Generating {num_graphs} {graph_type} graphs...")
        
        for i in range(num_graphs):
            graph = _generate_single_graph(graph_type, i)
            if graph is not None:
                filename = os.path.join(type_dir, f"{graph_type}_{i:04d}.graphml")
                nx.write_graphml(graph, filename)
        
        print(f"Saved {graph_type} graphs to {type_dir}")


def _generate_single_graph(graph_type: str, seed: int) -> nx.Graph:
    """Generate a single graph of specified type.
    
    Args:
        graph_type: Type of graph to generate
        seed: Random seed for reproducibility
        
    Returns:
        NetworkX graph or None if generation failed
    """
    # Use seed for reproducible generation
    import random
    random.seed(seed)
    
    try:
        if graph_type == 'erdos_renyi':
            n = random.randint(10, 50)
            p = random.uniform(0.1, 0.5)
            return nx.erdos_renyi_graph(n, p, seed=seed)
            
        elif graph_type == 'barabasi_albert':
            n = random.randint(10, 50)
            m = random.randint(1, min(5, n-1))
            return nx.barabasi_albert_graph(n, m, seed=seed)
            
        elif graph_type == 'cycle':
            n = random.randint(10, 30)
            graph = nx.cycle_graph(n)
            # Add some random edges to make it more interesting
            for _ in range(random.randint(0, n//4)):
                u, v = random.sample(list(graph.nodes()), 2)
                graph.add_edge(u, v)
            return graph
            
        elif graph_type == 'tree':
            n = random.randint(10, 50)
            return nx.random_tree(n, seed=seed)
            
        else:
            print(f"Unknown graph type: {graph_type}")
            return None
            
    except Exception as e:
        print(f"Error generating {graph_type} graph: {e}")
        return None


def test_dataset_pipeline(dataset_config: dict, task_config: dict):
    """Test the dataset and task pipeline.
    
    Args:
        dataset_config: Dataset configuration
        task_config: Task configuration
    """
    print("Testing dataset and task pipeline...")
    
    # Create dataset and task
    dataset = dataset_registry.create(dataset_config['type'], dataset_config)
    task = task_registry.create(task_config['type'], task_config)
    
    print(f"Dataset: {dataset.name}")
    print(f"Task: {task.name}")
    
    # Test data loading
    print("Loading universal data...")
    universal_data = dataset.load_data()
    print(f"Loaded {len(universal_data)} graphs")
    
    if len(universal_data) > 0:
        sample_data = universal_data[0]
        print(f"Sample graph: {sample_data.num_nodes} nodes, {sample_data.num_edges} edges")
    
    # Test task preparation
    print("Preparing task data...")
    task_data = task.prepare_data(dataset)
    print(f"Prepared {len(task_data)} task samples")
    
    if len(task_data) > 0:
        sample_task = task_data[0]
        print(f"Sample task data: {sample_task}")
        if hasattr(sample_task, 'y'):
            print(f"Task label: {sample_task.y}")
    
    # Test data splits
    print("Testing data splits...")
    train_data, val_data, test_data = dataset.get_splits()
    print(f"Splits - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    return dataset, task, task_data


def main():
    """Main function for data generation utilities."""
    parser = argparse.ArgumentParser(description='Synthetic data generation utilities')
    parser.add_argument('--mode', choices=['generate', 'test'], required=True,
                       help='Mode: generate test graphs or test pipeline')
    parser.add_argument('--output-dir', default='data/raw/test_graphs',
                       help='Output directory for generated graphs')
    parser.add_argument('--num-graphs', type=int, default=100,
                       help='Number of graphs to generate per type')
    parser.add_argument('--dataset-config', 
                       help='Path to dataset configuration file')
    parser.add_argument('--task-config',
                       help='Path to task configuration file')
    
    args = parser.parse_args()
    
    if args.mode == 'generate':
        generate_test_graphs(args.output_dir, args.num_graphs)
        print(f"Generated test graphs in {args.output_dir}")
        
    elif args.mode == 'test':
        # Test with default configs if not provided
        dataset_config = {
            'type': 'universal_synthetic',
            'name': 'test_synthetic',
            'graph_sources': [args.output_dir + '/cycle', args.output_dir + '/tree'],
            'cache_path': 'data/processed/test_universal.pt',
            'max_graphs': 50
        }
        
        task_config = {
            'type': 'cycle_detection',
            'name': 'test_cycle_detection'
        }
        
        # Load configs from files if provided
        if args.dataset_config:
            import yaml
            with open(args.dataset_config, 'r') as f:
                dataset_config.update(yaml.safe_load(f))
        
        if args.task_config:
            import yaml
            with open(args.task_config, 'r') as f:
                task_config.update(yaml.safe_load(f))
        
        test_dataset_pipeline(dataset_config, task_config)


if __name__ == '__main__':
    main()