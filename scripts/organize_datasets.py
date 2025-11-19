"""Organize datasets: create subsets and per-directory statistics."""

import networkx as nx
from pathlib import Path
import numpy as np
from collections import defaultdict
import json
import shutil

def compute_graph_stats(G):
    """Compute statistics for a single graph."""
    n = G.number_of_nodes()
    m = G.number_of_edges()

    stats = {
        'nodes': n,
        'edges': m,
        'density': 2 * m / (n * (n - 1)) if n > 1 else 0,
        'avg_degree': 2 * m / n if n > 0 else 0,
        'has_cycle': not nx.is_forest(G),
        'num_components': nx.number_connected_components(G),
    }

    if nx.is_connected(G) and n > 1:
        try:
            stats['diameter'] = nx.diameter(G)
        except:
            stats['diameter'] = None
    else:
        stats['diameter'] = None

    try:
        stats['clustering_coeff'] = nx.average_clustering(G)
    except:
        stats['clustering_coeff'] = None

    return stats

def analyze_directory(data_dir):
    """Compute comprehensive statistics for a directory of graphs."""
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

    for filepath in graphml_files:
        G = nx.read_graphml(filepath)
        if G.is_directed():
            G = G.to_undirected()

        graph_stats = compute_graph_stats(G)
        for key in ['nodes', 'edges', 'density', 'avg_degree', 'has_cycle',
                    'num_components', 'diameter', 'clustering_coeff']:
            stats[key].append(graph_stats[key])

    return stats

def compute_summary(stats, stat_name):
    """Compute summary statistics."""
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

def create_summary_stats(stats):
    """Create summary statistics dictionary."""
    cycle_count = sum(1 for c in stats['has_cycle'] if c)

    return {
        'num_graphs': stats['num_graphs'],
        'cycle_count': cycle_count,
        'cycle_percentage': 100 * cycle_count / stats['num_graphs'] if stats['num_graphs'] > 0 else 0,
        'nodes': compute_summary(stats, 'nodes'),
        'edges': compute_summary(stats, 'edges'),
        'density': compute_summary(stats, 'density'),
        'avg_degree': compute_summary(stats, 'avg_degree'),
        'num_components': compute_summary(stats, 'num_components'),
        'diameter': compute_summary(stats, 'diameter'),
        'clustering_coeff': compute_summary(stats, 'clustering_coeff')
    }

def write_stats_files(data_dir, stats, dataset_name):
    """Write statistics files to dataset directory."""
    data_path = Path(data_dir)

    # Create JSON
    summary = create_summary_stats(stats)
    json_path = data_path / "dataset_stats.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Create README
    readme_path = data_path / "README.md"
    cycle_count = summary['cycle_count']
    cycle_pct = summary['cycle_percentage']

    readme_content = f"""# {dataset_name}

## Overview
- **Total Graphs:** {stats['num_graphs']}
- **Graphs with Cycles:** {cycle_count}/{stats['num_graphs']} ({cycle_pct:.1f}%)

## Node Statistics
```
Total Nodes:    {int(summary['nodes']['total']):,}
Min Nodes:      {int(summary['nodes']['min'])}
Max Nodes:      {int(summary['nodes']['max'])}
Mean Nodes:     {summary['nodes']['mean']:.2f} ± {summary['nodes']['std']:.2f}
Median Nodes:   {summary['nodes']['median']:.1f}
```

## Edge Statistics
```
Total Edges:    {int(summary['edges']['total']):,}
Min Edges:      {int(summary['edges']['min'])}
Max Edges:      {int(summary['edges']['max'])}
Mean Edges:     {summary['edges']['mean']:.2f} ± {summary['edges']['std']:.2f}
Median Edges:   {summary['edges']['median']:.1f}
```

## Graph Properties
```
Density:        {summary['density']['mean']:.3f} ± {summary['density']['std']:.3f}
Avg Degree:     {summary['avg_degree']['mean']:.2f} ± {summary['avg_degree']['std']:.2f}
Components:     {summary['num_components']['mean']:.2f} ± {summary['num_components']['std']:.2f}
"""

    if summary['diameter']:
        readme_content += f"Diameter:       {summary['diameter']['mean']:.2f} ± {summary['diameter']['std']:.2f} (connected graphs only)\n"

    if summary['clustering_coeff']:
        readme_content += f"Clustering:     {summary['clustering_coeff']['mean']:.3f} ± {summary['clustering_coeff']['std']:.3f}\n"

    readme_content += f"""
```

## Node Distribution
"""

    node_dist = defaultdict(int)
    for n in stats['nodes']:
        node_dist[n] += 1

    for n in sorted(node_dist.keys()):
        count = node_dist[n]
        pct = 100 * count / stats['num_graphs']
        readme_content += f"- {n:2d} nodes: {count:5d} graphs ({pct:5.1f}%)\n"

    with open(readme_path, 'w') as f:
        f.write(readme_content)

    print(f"  ✓ Created {json_path}")
    print(f"  ✓ Created {readme_path}")

def create_subset(source_dir, target_dir, num_graphs):
    """Create a subset of graphs."""
    source_path = Path(source_dir)
    target_path = Path(target_dir)

    # Create target directory
    target_path.mkdir(parents=True, exist_ok=True)

    # Copy first num_graphs
    graphml_files = sorted(source_path.glob("*.graphml"))[:num_graphs]

    print(f"Creating subset: {target_dir} with {num_graphs} graphs...")
    for i, src_file in enumerate(graphml_files):
        dst_file = target_path / f"graph_{i:05d}.graphml"
        shutil.copy2(src_file, dst_file)

        if (i + 1) % 500 == 0:
            print(f"  Copied {i + 1}/{num_graphs} graphs...")

    print(f"  ✓ Created {num_graphs} graphs in {target_dir}")

def main():
    # Define datasets
    datasets = [
        {
            'name': 'ER Small (50 graphs, n=5-19, p=0.3)',
            'source': 'data/synthetic_er/train',
            'target': 'data/synthetic_er_50',
            'num_graphs': 50
        },
        {
            'name': 'ER Medium (500 graphs, n=5-19, p=0.3)',
            'source': 'data/synthetic_er/train',
            'target': 'data/synthetic_er_500',
            'num_graphs': 500
        },
        {
            'name': 'ER Large-1K (1000 graphs, n=10-20, p=0.3)',
            'source': 'data/large_5000',
            'target': 'data/synthetic_er_1000',
            'num_graphs': 1000
        },
        {
            'name': 'ER Large-2K (2000 graphs, n=10-20, p=0.3)',
            'source': 'data/large_5000',
            'target': 'data/synthetic_er_2000',
            'num_graphs': 2000
        },
        {
            'name': 'ER Large-5K (5000 graphs, n=10-20, p=0.3)',
            'source': 'data/large_5000',
            'target': 'data/synthetic_er_5000',
            'num_graphs': 5000
        }
    ]

    print("="*70)
    print("ORGANIZING DATASETS")
    print("="*70)

    for dataset in datasets:
        print(f"\n{dataset['name']}")
        print("-"*70)

        # Create subset if needed
        if dataset['target'] != dataset['source']:
            if not Path(dataset['target']).exists():
                create_subset(dataset['source'], dataset['target'], dataset['num_graphs'])
            else:
                existing_count = len(list(Path(dataset['target']).glob("*.graphml")))
                if existing_count != dataset['num_graphs']:
                    print(f"  WARNING: {dataset['target']} has {existing_count} graphs, expected {dataset['num_graphs']}")
                    print(f"  Recreating subset...")
                    shutil.rmtree(dataset['target'])
                    create_subset(dataset['source'], dataset['target'], dataset['num_graphs'])
                else:
                    print(f"  ✓ Subset already exists with {existing_count} graphs")

        # Analyze and write stats
        print(f"Analyzing {dataset['target']}...")
        stats = analyze_directory(dataset['target'])
        write_stats_files(dataset['target'], stats, dataset['name'])

    print("\n" + "="*70)
    print("DATASET ORGANIZATION COMPLETE")
    print("="*70)
    print("\nCreated directories:")
    for dataset in datasets:
        print(f"  - {dataset['target']}")
        print(f"    - README.md (human-readable stats)")
        print(f"    - dataset_stats.json (machine-readable)")
        print(f"    - {dataset['num_graphs']} .graphml files")

if __name__ == "__main__":
    main()
