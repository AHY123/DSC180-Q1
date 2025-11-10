# Graph Learning Research Platform

A modular framework for comparing graph-native and sequence-based transformers for graph representation learning.

## Research Question

Is the "fancy" graph-native design (GPS) truly necessary, or can "simple" sequence-based approaches (AutoGraph) achieve comparable expressiveness?

## Models Compared

- **GPS**: Graph-native Transformer with local + global attention
- **AutoGraph**: Sequence-based graph tokenization  
- **Baseline**: Classical MPNN implementations

## Tasks

- **Synthetic**: Cycle detection, shortest-path distance
- **Real-world**: Graph classification on ZINC 12k

## Quick Start

```bash
# Setup all environments
make setup-all

# Run experiment
make train CONFIG=experiments/runs/gps_zinc.yaml

# Different models
make train-gps CONFIG=experiments/runs/gps_zinc.yaml
make train-autograph CONFIG=experiments/runs/autograph_cycles.yaml
```

## Structure

- `src/` - Core framework and adapters
- `external/` - External model repositories (submodules)
- `experiments/` - Configuration files
- `docs/` - Detailed documentation
- `legacy/` - Previous weekly work

## Documentation

- [Architecture Overview](docs/PROJECT_ARCHITECTURE.md)
- [Setup Instructions](docs/GPS-SETUP.md)
- [Coding Guide](docs/CODING_GUIDE.md)