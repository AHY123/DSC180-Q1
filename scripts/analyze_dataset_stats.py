"""Analyze graph dataset statistics."""

import networkx as nx
from pathlib import Path
import numpy as np
from collections import defaultdict
import json

def analyze_dataset(data_dir):
    """Compute comprehensive statistics for a graph dataset."""
    graphml_files = sorted(Path(data_dir).glob("*.graphml"))

    stats = {
        'num_graphs': len(graphml_files),
        'nodes': [],
        'edges': [],
        'density': [],
        'avg_degree': [],
        'has_cycle': [],
        'num_components': [],
        'diameter': [],
        'clustering_coeff': []
    }

    print(f"Analyzing {len(graphml_files)} graphs from {data_dir}...")

    for i, filepath in enumerate(graphml_files):
        G = nx.read_graphml(filepath)
        if G.is_directed():
            G = G.to_undirected()

        n = G.number_of_nodes()
        m = G.number_of_edges()

        stats['nodes'].append(n)
        stats['edges'].append(m)

        # Density
        if n > 1:
            density = 2 * m / (n * (n - 1))
        else:
            density = 0
        stats['density'].append(density)

        # Average degree
        if n > 0:
            avg_deg = 2 * m / n
        else:
            avg_deg = 0
        stats['avg_degree'].append(avg_deg)

        # Cycle detection
        try:
            has_cycle = not nx.is_forest(G)
            stats['has_cycle'].append(has_cycle)
        except:
            stats['has_cycle'].append(None)

        # Number of connected components
        stats['num_components'].append(nx.number_connected_components(G))

        # Diameter (for connected graphs)
        if nx.is_connected(G) and n > 1:
            try:
                diam = nx.diameter(G)
                stats['diameter'].append(diam)
            except:
                stats['diameter'].append(None)
        else:
            stats['diameter'].append(None)

        # Clustering coefficient
        try:
            cc = nx.average_clustering(G)
            stats['clustering_coeff'].append(cc)
        except:
            stats['clustering_coeff'].append(None)

        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(graphml_files)} graphs...")

    return stats

def compute_summary(stats, stat_name):
    """Compute summary statistics for a numeric stat."""
    values = [v for v in stats[stat_name] if v is not None]
    if not values:
        return None

    return {
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'median': float(np.median(values)),
        'total': float(np.sum(values)) if stat_name in ['nodes', 'edges'] else None
    }

def print_stats(dataset_name, stats):
    """Print formatted statistics."""
    print(f"\n{'='*70}")
    print(f"DATASET: {dataset_name}")
    print(f"{'='*70}")
    print(f"Total Graphs: {stats['num_graphs']}")

    # Cycle statistics
    cycle_count = sum(1 for c in stats['has_cycle'] if c)
    print(f"Graphs with Cycles: {cycle_count}/{stats['num_graphs']} ({100*cycle_count/stats['num_graphs']:.1f}%)")

    print(f"\n{'-'*70}")
    print(f"NODE STATISTICS")
    print(f"{'-'*70}")
    node_stats = compute_summary(stats, 'nodes')
    print(f"  Total Nodes:    {int(node_stats['total']):,}")
    print(f"  Min Nodes:      {int(node_stats['min'])}")
    print(f"  Max Nodes:      {int(node_stats['max'])}")
    print(f"  Mean Nodes:     {node_stats['mean']:.2f} ± {node_stats['std']:.2f}")
    print(f"  Median Nodes:   {node_stats['median']:.1f}")

    print(f"\n{'-'*70}")
    print(f"EDGE STATISTICS")
    print(f"{'-'*70}")
    edge_stats = compute_summary(stats, 'edges')
    print(f"  Total Edges:    {int(edge_stats['total']):,}")
    print(f"  Min Edges:      {int(edge_stats['min'])}")
    print(f"  Max Edges:      {int(edge_stats['max'])}")
    print(f"  Mean Edges:     {edge_stats['mean']:.2f} ± {edge_stats['std']:.2f}")
    print(f"  Median Edges:   {edge_stats['median']:.1f}")

    print(f"\n{'-'*70}")
    print(f"GRAPH PROPERTIES")
    print(f"{'-'*70}")
    density_stats = compute_summary(stats, 'density')
    print(f"  Density:        {density_stats['mean']:.3f} ± {density_stats['std']:.3f}")

    degree_stats = compute_summary(stats, 'avg_degree')
    print(f"  Avg Degree:     {degree_stats['mean']:.2f} ± {degree_stats['std']:.2f}")

    comp_stats = compute_summary(stats, 'num_components')
    print(f"  Components:     {comp_stats['mean']:.2f} ± {comp_stats['std']:.2f}")

    diam_stats = compute_summary(stats, 'diameter')
    if diam_stats:
        print(f"  Diameter:       {diam_stats['mean']:.2f} ± {diam_stats['std']:.2f} (connected graphs only)")

    clust_stats = compute_summary(stats, 'clustering_coeff')
    if clust_stats:
        print(f"  Clustering:     {clust_stats['mean']:.3f} ± {clust_stats['std']:.3f}")

    # Distribution of nodes
    print(f"\n{'-'*70}")
    print(f"NODE DISTRIBUTION")
    print(f"{'-'*70}")
    node_dist = defaultdict(int)
    for n in stats['nodes']:
        node_dist[n] += 1

    for n in sorted(node_dist.keys()):
        count = node_dist[n]
        pct = 100 * count / stats['num_graphs']
        print(f"  {n:2d} nodes: {count:5d} graphs ({pct:5.1f}%)")

def main():
    datasets = [
        ('50 Graphs (ER p=0.3)', 'data/synthetic_er/train', 50),
        ('500 Graphs (ER p=0.3)', 'data/synthetic_er/train', 500),
        ('5000 Graphs (ER p=0.3)', 'data/large_5000', 5000)
    ]

    all_results = {}

    for name, path, max_graphs in datasets:
        if not Path(path).exists():
            print(f"Skipping {name} - path {path} not found")
            continue

        stats = analyze_dataset(path)

        # Limit the count based on max_graphs
        if stats['num_graphs'] > max_graphs:
            for key in stats:
                if isinstance(stats[key], list):
                    stats[key] = stats[key][:max_graphs]
            stats['num_graphs'] = max_graphs

        print_stats(name, stats)
        all_results[name] = stats

    # Save to JSON
    print(f"\n{'='*70}")
    print("Saving results to results/dataset_statistics.json")

    # Convert for JSON serialization
    json_results = {}
    for dataset_name, stats in all_results.items():
        json_results[dataset_name] = {
            'num_graphs': stats['num_graphs'],
            'cycle_count': sum(1 for c in stats['has_cycle'] if c),
            'cycle_percentage': 100 * sum(1 for c in stats['has_cycle'] if c) / stats['num_graphs'],
            'nodes': compute_summary(stats, 'nodes'),
            'edges': compute_summary(stats, 'edges'),
            'density': compute_summary(stats, 'density'),
            'avg_degree': compute_summary(stats, 'avg_degree'),
            'num_components': compute_summary(stats, 'num_components'),
            'diameter': compute_summary(stats, 'diameter'),
            'clustering_coeff': compute_summary(stats, 'clustering_coeff')
        }

    Path('results').mkdir(exist_ok=True)
    with open('results/dataset_statistics.json', 'w') as f:
        json.dump(json_results, f, indent=2)

    print("✓ Results saved")

if __name__ == "__main__":
    main()
