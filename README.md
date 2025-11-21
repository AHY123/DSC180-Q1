# Graph Learning Research Platform

A modular framework for comparing graph-native and sequence-based transformers for graph representation learning.

## Research Question

Is the "fancy" graph-native design (GPS) truly necessary, or can "simple" sequence-based approaches (AutoGraph) achieve comparable expressiveness?

## Models Compared

- **GPS**: Graph-native Transformer with local + global attention
- **AutoGraph**: Sequence-based graph tokenization with DFS traversal
- **graph-token**: Index-based tokenization approach
- **Baseline MPNNs**: GIN, GCN, GAT (classical message-passing networks)

## Tasks

- **Synthetic**: Cycle detection, shortest-path distance
- **Real-world**: Graph classification on ZINC 12k

---

## Quick Start

### 1. Setup Environment

```bash
# Install base dependencies (uv environment)
uv sync

# Setup all model-specific environments
make setup-all

# Or setup individually:
make setup-base        # Core project (GIN/GCN/GAT)
make setup-gps         # GPS model
make setup-autograph   # AutoGraph & graph-token
```

### 2. Run Your First Experiment

Using the new unified infrastructure:

```bash
# Run GIN on cycle detection (5000 graphs, 100 epochs)
uv run python scripts/run_experiment.py \
  --model GIN \
  --task cycle \
  --data data/synthetic_er_5000 \
  --epochs 100 \
  --batch_size 32

# Results saved to: results/generalization/cycle_GIN_<timestamp>.json
```

---

## Running Experiments

### Option 1: Unified Infrastructure (Recommended)

**Single script for all models** - Uses model registry and unified trainer:

```bash
# General pattern
uv run python scripts/run_experiment.py \
  --model {MODEL} \
  --task {TASK} \
  --data {DATA_DIR} \
  --epochs {EPOCHS} \
  [additional options]
```

**Available Models**: `GIN`, `GCN`, `GAT`, `GPS`, `AutoGraph`, `graph-token`
**Available Tasks**: `cycle`, `shortest_path`

#### Examples by Model Type:

**GNN Models (GIN, GCN, GAT):**
```bash
# Cycle detection with GIN
uv run python scripts/run_experiment.py \
  --model GIN \
  --task cycle \
  --data data/synthetic_er_5000 \
  --epochs 100 \
  --batch_size 32 \
  --hidden_dim 64 \
  --num_layers 3

# Shortest path with GCN
uv run python scripts/run_experiment.py \
  --model GCN \
  --task shortest_path \
  --data data/synthetic_er_5000 \
  --epochs 100 \
  --batch_size 32 \
  --k_pairs 10 \
  --max_distance 5
```

**GPS Model:**
```bash
# Requires GPS config file
uv run python scripts/run_experiment.py \
  --model GPS \
  --task cycle \
  --data data/synthetic_er_5000 \
  --gps_config configs/GPS/cycle.yaml \
  --epochs 100 \
  --batch_size 32
```

**Transformer Models (AutoGraph, graph-token):**
```bash
# AutoGraph with DFS serialization
uv run python scripts/run_experiment.py \
  --model AutoGraph \
  --task cycle \
  --data data/synthetic_er_5000 \
  --epochs 100 \
  --batch_size 8 \
  --d_model 32 \
  --num_layers 4

# graph-token with index-based tokenization
uv run python scripts/run_experiment.py \
  --model graph-token \
  --task shortest_path \
  --data data/synthetic_er_5000 \
  --epochs 100 \
  --batch_size 8 \
  --k_pairs 10 \
  --max_distance 5
```

#### Common Options:

```bash
--hidden_dim 64          # Hidden layer dimension (GNN models)
--d_model 32             # Model dimension (Transformers)
--num_layers 3           # Number of layers
--dropout 0.1            # Dropout rate
--lr 0.001               # Learning rate
--early_stopping         # Enable early stopping
--patience 10            # Early stopping patience
--no_wandb               # Disable wandb logging
--output_dir PATH        # Results directory
--checkpoint_dir PATH    # Model checkpoint directory
```

### Option 2: Legacy Scripts (Model-Specific)

**Original evaluation scripts** - Still fully functional:

**GIN/GCN/GAT (Base environment):**
```bash
uv run python scripts/evaluate_with_splits.py \
  --base_dir data/synthetic_er_5000 \
  --task cycle \
  --model_type GIN \
  --batch_size 32 \
  --epochs 100
```

**GPS (GPS environment):**
```bash
# Activate GPS conda environment first
eval "$(conda shell.bash hook)"
mamba run -n gps python scripts/evaluate_gps_splits.py \
  --base_dir data/synthetic_er_5000 \
  --task cycle \
  --batch_size 32 \
  --epochs 100
```

**AutoGraph (AutoGraph environment):**
```bash
mamba run -n autograph python scripts/evaluate_autograph_splits.py \
  --base_dir data/synthetic_er_5000 \
  --task cycle \
  --batch_size 8 \
  --epochs 100
```

**graph-token (AutoGraph environment):**
```bash
mamba run -n autograph python scripts/evaluate_graphtoken_splits.py \
  --base_dir data/synthetic_er_5000 \
  --task shortest_path \
  --batch_size 8 \
  --epochs 100 \
  --k_pairs 10 \
  --max_distance 5
```

---

## Experiment Tracking with Wandb

All scripts support **Weights & Biases** for experiment tracking.

### Using Wandb:

```bash
# Default: wandb enabled
uv run python scripts/run_experiment.py --model GIN --task cycle --data data/synthetic_er_5000

# Disable wandb
uv run python scripts/run_experiment.py --model GIN --task cycle --data data/synthetic_er_5000 --no_wandb

# Offline mode (no login required)
WANDB_MODE=offline uv run python scripts/run_experiment.py --model GIN --task cycle --data data/synthetic_er_5000
```

### What Wandb Tracks:

**Hyperparameters:**
- Model type, task, dataset
- Architecture (hidden_dim, num_layers, etc.)
- Training config (batch_size, lr, epochs)

**Per-Epoch Metrics:**
- train/loss, train/accuracy
- val/loss, val/accuracy
- epoch_time

**Final Results:**
- Test accuracy and loss
- Overfitting metrics (train-val gap, train-test gap)
- Best epoch, total training time
- Model parameter count

**Artifacts:**
- Complete results JSON file

**System Info (automatic):**
- Git commit and branch
- Environment details
- Hardware info

---

## Data Preparation

### Synthetic Datasets

Generate synthetic graph datasets:

```bash
# Generate Erdos-Renyi graphs with cycle labels
python scripts/generate_synthetic_data.py \
  --output_dir data/synthetic_er_5000 \
  --num_graphs 5000 \
  --graph_type erdos_renyi \
  --min_nodes 10 \
  --max_nodes 50 \
  --edge_prob 0.3 \
  --task cycle

# Generate with shortest path labels
python scripts/generate_synthetic_data.py \
  --output_dir data/synthetic_er_5000 \
  --num_graphs 5000 \
  --graph_type erdos_renyi \
  --min_nodes 10 \
  --max_nodes 50 \
  --edge_prob 0.3 \
  --task shortest_path \
  --k_pairs 10 \
  --max_distance 5
```

### Dataset Splits

All datasets are automatically split into:
- **Train**: 70% (3500 graphs)
- **Val**: 15% (750 graphs)
- **Test**: 15% (750 graphs)

Splits are created in subdirectories: `train/`, `val/`, `test/`

---

## Results and Outputs

### Output Structure

```
results/
├── generalization/              # Experiment results (JSON)
│   ├── cycle_GIN_20251120_114616.json
│   ├── shortest_path_GCN_20251120_120304.json
│   └── ...
└── figures/                     # Visualizations
    ├── step_comparison_cycle.png
    ├── step_curves_shortest_path.png
    └── ...
```

### JSON Result Format

```json
{
  "metadata": {
    "timestamp": "20251120_114616",
    "task": "cycle",
    "model_type": "GIN",
    "total_params": 50432
  },
  "config": { ... },
  "training": {
    "total_time": 145.23,
    "epochs_run": 100,
    "best_epoch": 87,
    "best_val_acc": 0.9867
  },
  "final_results": {
    "train": {"loss": 0.0234, "accuracy": 0.9914},
    "valid": {"loss": 0.0456, "accuracy": 0.9867},
    "test": {"loss": 0.0523, "accuracy": 0.9800}
  },
  "analysis": {
    "train_val_gap": 0.0047,
    "train_test_gap": 0.0114,
    "overfitting_detected": false,
    "good_generalization": true
  },
  "epoch_metrics": [ ... ],
  "step_metrics": [ ... ]
}
```

### Analyzing Results

View summary for a specific result:
```bash
cat results/generalization/cycle_GIN_20251120_114616_summary.txt
```

List all recent results:
```bash
ls -lht results/generalization/*.json | head -10
```

Extract test accuracies from all results:
```bash
python -c "
import json
import glob
for f in sorted(glob.glob('results/generalization/*.json')):
    d = json.load(open(f))
    print(f\"{d['metadata']['model_type']:12s} | {d['metadata']['task']:14s} | Test: {d['final_results']['test']['accuracy']*100:5.2f}%\")
"
```

---

## Visualization

### Generate Training Curves

```bash
# Create step-based training curves
python scripts/create_training_curves.py \
  --results_dir results/generalization \
  --output_dir results/figures

# Create epoch-based curves
python scripts/create_results_visualizations.py \
  --results_dir results/generalization \
  --output_dir results/figures
```

### Plot Types Generated:

1. **Step-based curves**: Training progress with rolling mean
2. **Epoch-based curves**: Per-epoch accuracy/loss
3. **Comparison plots**: Side-by-side model comparison
4. **Convergence analysis**: Time to convergence

---

## Project Structure

```
DSC180-Q1/
├── src/                         # Source code
│   ├── core/                    # Core infrastructure (NEW)
│   │   ├── model_registry.py   # Centralized model factory
│   │   └── trainer.py           # Unified trainer
│   ├── adapters/                # Model adapters
│   ├── datasets/                # Dataset implementations
│   ├── tasks/                   # Task definitions
│   └── utils/                   # Utilities
├── scripts/                     # Evaluation scripts
│   ├── run_experiment.py             # Unified runner (NEW)
│   ├── evaluate_with_splits.py       # GIN/GCN/GAT
│   ├── evaluate_gps_splits.py        # GPS
│   ├── evaluate_autograph_splits.py  # AutoGraph
│   ├── evaluate_graphtoken_splits.py # graph-token
│   └── generate_synthetic_data.py    # Data generation
├── data/                        # Datasets
│   ├── synthetic_er_5000/       # Example dataset
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── DATASETS_INDEX.md        # Dataset catalog
├── results/                     # Experiment results
│   ├── generalization/          # JSON results
│   └── figures/                 # Plots
├── docs/                        # Documentation
│   ├── INDEX.md                 # Documentation index
│   ├── WORKFLOW_IMPLEMENTATION_PROGRESS.md
│   └── ...
├── external/                    # Git submodules
│   ├── GraphGPS/
│   ├── AutoGraph/
│   └── graph-token/
├── environments/                # Conda environment configs
├── pyproject.toml              # Python dependencies (uv)
└── README.md                   # This file
```

---

## Common Workflows

### Running a Complete Experiment Suite

Run all models on both tasks:

```bash
# GNN models (parallel execution)
for model in GIN GCN GAT; do
  for task in cycle shortest_path; do
    uv run python scripts/run_experiment.py \
      --model $model \
      --task $task \
      --data data/synthetic_er_5000 \
      --epochs 100 \
      --batch_size 32 &
  done
done

# Wait for GNN models
wait

# GPS (requires specific environment)
eval "$(conda shell.bash hook)"
mamba run -n gps python scripts/evaluate_gps_splits.py \
  --base_dir data/synthetic_er_5000 \
  --task cycle \
  --epochs 100

# Transformers (smaller batch size)
for model in AutoGraph graph-token; do
  for task in cycle shortest_path; do
    uv run python scripts/run_experiment.py \
      --model $model \
      --task $task \
      --data data/synthetic_er_5000 \
      --epochs 100 \
      --batch_size 8 &
  done
done

wait
```

### Quick Testing (Short Runs)

Test your setup with 2 epochs:

```bash
# Quick test with GIN
uv run python scripts/run_experiment.py \
  --model GIN \
  --task cycle \
  --data data/synthetic_er_5000 \
  --epochs 2 \
  --batch_size 32 \
  --no_wandb
```

### Generating Visualizations

After running experiments:

```bash
# Generate all plots
python scripts/create_training_curves.py
python scripts/create_results_visualizations.py

# View results
ls results/figures/
```

---

## Troubleshooting

### Environment Issues

```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Verify PyTorch Geometric installation
python -c "import torch_geometric; print(torch_geometric.__version__)"

# List conda environments
conda env list
```

### Common Errors

**"Module not found"**: Ensure you're in the correct environment
```bash
# For GNN models
uv run python ...

# For GPS
mamba run -n gps python ...

# For Transformers
mamba run -n autograph python ...
```

**"Dataset not found"**: Generate synthetic data first
```bash
python scripts/generate_synthetic_data.py --output_dir data/synthetic_er_5000 --num_graphs 5000
```

**"Out of memory"**: Reduce batch size
```bash
--batch_size 16  # or even smaller for transformers
```

---

## Documentation

- **[Documentation Index](docs/INDEX.md)** - Complete documentation catalog
- **[Architecture Overview](docs/PROJECT_ARCHITECTURE.md)** - System design
- **[Workflow Progress](docs/WORKFLOW_IMPLEMENTATION_PROGRESS.md)** - Implementation status
- **[Coding Guide](docs/CODING_GUIDE.md)** - Development standards
- **[GPS Setup](docs/GPS-SETUP.md)** - GPS model installation
- **[Dataset Catalog](data/DATASETS_INDEX.md)** - Available datasets

---

## Citation

```bibtex
@misc{dsc180-graph-learning,
  title={Comparing Graph-Native and Sequence-Based Transformers for Graph Learning},
  author={DSC180A Capstone Team},
  year={2025}
}
```

---

## Development Status

**Phase 1**: ✅ Wandb Integration (Complete)
**Phase 2**: ✅ Model Registry & Unified Trainer (Complete)
**Current**: Ready for final experiments

See [WORKFLOW_IMPLEMENTATION_PROGRESS.md](docs/WORKFLOW_IMPLEMENTATION_PROGRESS.md) for detailed status.
