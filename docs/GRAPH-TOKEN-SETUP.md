# Setup Guide

This repository has been configured to use a conda environment.

## Environment Setup

The conda environment `graph-token` has been created and all dependencies have been installed.

### Activating the Environment

To activate the conda environment, run:

```bash
conda activate graph-token
```

### Verifying Installation

You can verify that all dependencies are installed correctly:

```bash
conda activate graph-token
python -c "import networkx; import numpy; import absl; import tensorflow; print('All imports successful!')"
```

## Generating Graphs

To generate graphs for all algorithms (Erdős–Rényi, Barabási–Albert, SBM, etc.):

```bash
conda activate graph-token
./graph_generator.sh
```

This will create graphs in the `graphs/<algorithm>/<split>/` directory structure.

Alternatively, you can run the Python script directly:

```bash
conda activate graph-token
python graph_generator.py \
    --algorithm="er" \
    --number_of_graphs=500 \
    --split=train \
    --output_path="graphs"
```

## Generating Task Files

After generating graphs, you can create task-specific tokenized samples:

```bash
conda activate graph-token
./task_generator.sh
```

This will create tokenized task files in the `tasks/<task>/<algorithm>/<split>/` directory structure.

Alternatively, you can run the Python script directly:

```bash
conda activate graph-token
python graph_task_generator.py \
    --task="node_degree" \
    --algorithm="er" \
    --task_dir="tasks" \
    --graphs_dir="graphs" \
    --split=train \
    --random_seed=1234
```

## Available Tasks

The following tasks are available:
- `edge_existence` - Determines whether an edge exists between two nodes
- `node_degree` - Predicts the degree of a given node
- `node_count` - Counts the total number of nodes
- `edge_count` - Counts the total number of edges
- `connected_nodes` - Lists all nodes connected to a specific node
- `cycle_check` - Checks whether the graph contains a cycle
- `disconnected_nodes` - Identifies isolated nodes
- `reachability` - Determines whether a path exists between two nodes
- `shortest_path` - Computes the shortest path length between two nodes
- `maximum_flow` - Calculates the maximum flow between two nodes
- `triangle_counting` - Counts the total number of triangles
- `node_classification` - Predicts the community/class label of each node

## Available Graph Algorithms

- `er` - Erdős–Rényi random graphs
- `ba` - Barabási–Albert scale-free graphs
- `sbm` - Stochastic Block Model
- `sfn` - Scale-Free Network (Holme–Kim / Power-Law)
- `complete` - Complete graphs
- `star` - Star graphs
- `path` - Path (chain) graphs

## Notes

- The shell scripts (`graph_generator.sh` and `task_generator.sh`) automatically activate the conda environment
- Make sure you have conda initialized in your shell (usually done automatically, but if not, run: `eval "$(conda shell.bash hook)"`)
- Generated graphs are saved in `.graphml` format
- Task files are saved in `.json` format with tokenized sequences

