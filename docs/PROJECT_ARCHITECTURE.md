# Project Architecture Plan

## Overview

This document outlines the architecture for a modular graph learning research platform that supports multiple models (GPS, AutoGraph, sequence-based), datasets, and tasks with standardized training workflows.

## 1. External Repository Management Strategy

### Problem
Research models (GPS, AutoGraph, etc.) are often distributed as standalone GitHub repositories with their own:
- Directory structures
- Dependencies and environments  
- Training scripts and configurations
- Dataset loaders
- Evaluation code

### Recommended Approach: Git Submodules + Adapter Pattern

#### Option 1: Git Submodules (Recommended)
```bash
# Add external repos as submodules
git submodule add https://github.com/rampasek/GraphGPS.git external/GraphGPS
git submodule add https://github.com/deepmind/graph_nets.git external/GraphNets
git submodule add https://github.com/user/AutoGraph.git external/AutoGraph

# Update submodules
git submodule update --init --recursive
git submodule update --remote
```

**Advantages:**
- Preserves original repo structure and git history
- Easy to pull updates from upstream
- Clear attribution and licensing
- Can pin to specific commits for reproducibility

**Project Structure with Submodules:**
```
graph-learning-platform/
├── src/                     # Our code
│   ├── adapters/           # Adapter classes for external models
│   │   ├── gps_adapter.py
│   │   ├── autograph_adapter.py
│   │   └── sequence_adapter.py
│   └── core/               # Our framework
├── external/               # External repositories (submodules)
│   ├── GraphGPS/          # GPS implementation
│   ├── AutoGraph/         # AutoGraph implementation
│   └── SequenceModels/    # Sequence-based models
├── experiments/           # Our experiment configs
└── environments/          # Environment definitions
```

#### Option 2: Package Installation (Alternative)
For models available as pip packages:
```bash
# In environment configs
pip install torch-geometric-gps
pip install autograph-transforms
```

#### Option 3: Vendored Code (Last Resort)
Copy code directly into our repo only if:
- Repo is unmaintained
- Significant modifications needed
- No clear licensing

### Integration Strategy

#### Adapter Pattern Implementation
```python
# src/adapters/gps_adapter.py
import sys
sys.path.append('external/GraphGPS')

from external.GraphGPS.graphgps.model import GPS
from src.core.base_model import BaseModel

class GPSAdapter(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        # Convert our config to GPS format
        gps_config = self._convert_config(config)
        self.model = GPS(gps_config)
    
    def forward(self, batch):
        # Convert our batch format to GPS format
        gps_batch = self._convert_batch(batch)
        return self.model(gps_batch)
    
    def _convert_config(self, config):
        # Transform our unified config to GPS-specific format
        return {
            'model_type': 'GPS',
            'dim_hidden': config.hidden_dim,
            'num_layers': config.num_layers,
            # ... other mappings
        }

#### Environment-Specific Runners
```python
# src/core/model_runner.py
class ModelRunner:
    def __init__(self, model_type: str):
        self.model_type = model_type
        self.env_config = self._load_env_config(model_type)
    
    def run_in_environment(self, script_path: str, args: Dict):
        env_name = self.env_config['environment']
        cmd = f"conda run -n {env_name} python {script_path}"
        
        # Set environment variables for submodule paths
        env_vars = {
            'PYTHONPATH': f"external/{self.model_type}:$PYTHONPATH",
            **self.env_config.get('env_vars', {})
        }
        
        subprocess.run(cmd, env=env_vars, shell=True)
```

#### Configuration Translation Layer
```python
# src/core/config_translator.py
class ConfigTranslator:
    """Translates between our unified config and model-specific configs"""
    
    def to_gps_config(self, unified_config):
        return {
            'model': {
                'type': 'GPS',
                'dim_hidden': unified_config.model.hidden_dim,
                'num_layers': unified_config.model.num_layers,
                'attn_type': unified_config.model.get('attention_type', 'multihead')
            },
            'dataset': {
                'name': unified_config.dataset.name,
                'dir': unified_config.dataset.data_dir
            }
        }
    
    def to_autograph_config(self, unified_config):
        # Different format for AutoGraph
        return {
            'model_config': {
                'hidden_size': unified_config.model.hidden_dim,
                'num_attention_heads': unified_config.model.num_heads
            }
        }
```

### Repository Management Commands
```bash
# Makefile targets for submodule management
setup-submodules:
	git submodule update --init --recursive
	
update-submodules:
	git submodule update --remote
	
pin-submodule:
	cd external/GraphGPS && git checkout $(COMMIT_HASH)
	git add external/GraphGPS
	git commit -m "Pin GraphGPS to $(COMMIT_HASH)"

# Individual model setup
setup-gps: setup-submodules
	conda run -n gps pip install -e external/GraphGPS/
	
setup-autograph: setup-submodules  
	conda run -n autograph pip install -e external/AutoGraph/
```

### Handling Model-Specific Dependencies
```yaml
# environments/gps/environment.yml
name: gps
dependencies:
  - pytorch=2.0
  - pytorch-geometric=2.3
  - pip:
    - wandb
    - yacs  # GPS-specific dependency
    - performer-pytorch  # GPS-specific
    
# environments/autograph/environment.yml  
name: autograph
dependencies:
  - tensorflow=2.12
  - networkx
  - pip:
    - graph-nets  # AutoGraph-specific
    - sonnet  # AutoGraph-specific

## 2. Environment Management Strategy

### Conda vs UV Analysis
- **Recommendation**: Use **Conda** for model environments, **UV** for core project management
- **Rationale**:
  - Different models have conflicting CUDA/PyTorch requirements
  - GPS requires specific PyTorch Geometric versions
  - AutoGraph may need TensorFlow
  - Conda handles binary dependencies (CUDA) better than UV

### Environment Structure
```
environments/
├── base/                    # Core project environment (UV managed)
│   ├── pyproject.toml
│   └── requirements.txt
├── gps/                     # GPS model environment
│   ├── environment.yml
│   └── requirements.txt
├── autograph/               # AutoGraph environment  
│   ├── environment.yml
│   └── requirements.txt
└── sequence/                # Sequence-based models
    ├── environment.yml
    └── requirements.txt
```

### Environment Management Commands
```bash
# Setup all environments
make setup-all

# Setup specific environment
make setup-gps
make setup-autograph

# Activate environment for training
make train-gps CONFIG=experiments/gps_zinc.yaml
```

## 2. Project Structure

```
DSC180-Q1/                  # Root project directory
├── src/
│   ├── core/               # Core framework
│   │   ├── __init__.py
│   │   ├── base_model.py   # Abstract base classes
│   │   ├── base_dataset.py
│   │   ├── base_task.py
│   │   ├── trainer.py      # Unified training logic
│   │   ├── evaluator.py    # Evaluation framework
│   │   ├── logger.py       # Logging utilities
│   │   ├── model_runner.py # Environment-specific execution
│   │   ├── config_translator.py # Config format translation
│   │   └── registry.py     # Model/dataset/task registration
│   ├── adapters/           # External model adapters
│   │   ├── __init__.py
│   │   ├── gps_adapter.py
│   │   ├── autograph_adapter.py
│   │   └── sequence_adapter.py
│   ├── datasets/           # Dataset implementations
│   │   ├── __init__.py
│   │   ├── synthetic/
│   │   │   ├── __init__.py
│   │   │   ├── cycle_check.py
│   │   │   └── shortest_path.py
│   │   ├── real_world/
│   │   │   ├── __init__.py
│   │   │   ├── zinc.py
│   │   │   ├── imdb.py
│   │   │   └── cora.py
│   │   └── loaders/
│   │       ├── __init__.py
│   │       └── graph_loader.py
│   ├── tasks/              # Task definitions
│   │   ├── __init__.py
│   │   ├── graph_classification.py
│   │   ├── node_classification.py
│   │   ├── cycle_detection.py
│   │   └── shortest_path.py
│   └── utils/
│       ├── __init__.py
│       ├── config.py       # Configuration management
│       ├── metrics.py      # Evaluation metrics
│       └── visualization.py
├── external/               # External repositories (submodules)
│   ├── GraphGPS/          # GPS implementation
│   ├── AutoGraph/         # AutoGraph implementation
│   └── SequenceModels/    # Sequence-based models
├── environments/           # Environment definitions
│   ├── base/
│   │   ├── pyproject.toml
│   │   └── requirements.txt
│   ├── gps/
│   │   ├── environment.yml
│   │   └── requirements.txt
│   ├── autograph/
│   │   ├── environment.yml
│   │   └── requirements.txt
│   └── sequence/
│       ├── environment.yml
│       └── requirements.txt
├── experiments/            # Experiment configurations
│   ├── configs/
│   │   ├── models/
│   │   │   ├── gps_base.yaml
│   │   │   ├── autograph_base.yaml
│   │   │   └── sequence_base.yaml
│   │   ├── datasets/
│   │   │   ├── zinc.yaml
│   │   │   ├── synthetic_cycles.yaml
│   │   │   └── synthetic_paths.yaml
│   │   └── tasks/
│   │       ├── graph_classification.yaml
│   │       ├── cycle_detection.yaml
│   │       └── shortest_path.yaml
│   └── runs/               # Complete experiment configs
│       ├── gps_zinc.yaml
│       ├── autograph_cycles.yaml
│       └── sequence_paths.yaml
├── scripts/                # Utility scripts
│   ├── setup_environments.sh
│   ├── run_experiment.py
│   ├── generate_synthetic_data.py
│   └── evaluate_results.py
├── notebooks/              # Research notebooks
│   ├── exploratory/
│   ├── visualization/
│   └── analysis/
├── data/                   # Data storage
│   ├── raw/
│   ├── processed/
│   └── synthetic/
├── results/                # Experiment outputs
│   ├── checkpoints/
│   ├── logs/
│   └── figures/
├── legacy/                 # Previous weekly work
│   ├── week1/
│   ├── week2/
│   ├── week3/
│   ├── week4/
│   └── week6/
├── Makefile               # Development commands
├── main.py                # Entry point
├── pyproject.toml         # Core project config
├── requirements.txt       # Legacy requirements
├── uv.lock               # UV lock file
├── CLAUDE.md             # Claude Code guidance
├── PROJECT_ARCHITECTURE.md # This document
├── CODING_GUIDE.md       # Coding standards
├── Research_Goals.md     # Research proposal
├── GPS-SETUP.md          # GPS setup instructions
├── GRAPH-TOKEN-SETUP.md  # Graph tokenization setup
└── README.md             # Project documentation
```

## 3. Core Abstractions

### Base Model Interface
```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class BaseModel(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    @abstractmethod
    def forward(self, batch) -> Any:
        pass
    
    @abstractmethod
    def loss(self, predictions, targets) -> torch.Tensor:
        pass
    
    @abstractmethod
    def get_embeddings(self, batch) -> torch.Tensor:
        pass
    
    @property
    @abstractmethod
    def supported_tasks(self) -> List[str]:
        pass
```

### Base Dataset Interface
```python
class BaseDataset(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    @abstractmethod
    def load_data(self) -> Any:
        pass
    
    @abstractmethod
    def get_splits(self) -> Tuple[Any, Any, Any]:
        pass
    
    @property
    @abstractmethod
    def num_features(self) -> int:
        pass
    
    @property
    @abstractmethod
    def num_classes(self) -> int:
        pass
```

### Task Interface
```python
class BaseTask(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    @abstractmethod
    def prepare_data(self, dataset) -> Any:
        pass
    
    @abstractmethod
    def evaluate(self, predictions, targets) -> Dict[str, float]:
        pass
    
    @property
    @abstractmethod
    def metric_names(self) -> List[str]:
        pass
```

## 4. Configuration Management

### Hierarchical YAML Configuration
```yaml
# experiments/runs/gps_zinc.yaml
experiment:
  name: "gps_zinc_comparison"
  description: "GPS model on ZINC dataset for graph classification"
  
model:
  type: "gps"
  config_file: "configs/models/gps_base.yaml"
  overrides:
    hidden_dim: 256
    num_layers: 6
    
dataset:
  type: "zinc"
  config_file: "configs/datasets/zinc.yaml"
  overrides:
    subset_size: 12000
    
task:
  type: "graph_classification"
  config_file: "configs/tasks/graph_classification.yaml"
  
training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
  optimizer: "adam"
  scheduler: "cosine"
  
logging:
  wandb:
    project: "graph-learning-comparison"
    entity: "your-username"
    tags: ["gps", "zinc", "graph-classification"]
  local:
    log_dir: "results/logs"
    save_checkpoints: true
    checkpoint_freq: 10
```

## 5. Training Pipeline

### Unified Trainer
```python
class UniversalTrainer:
    def __init__(self, model, dataset, task, config):
        self.model = model
        self.dataset = dataset
        self.task = task
        self.config = config
        self.logger = self._setup_logging()
    
    def train(self):
        for epoch in range(self.config.epochs):
            train_loss = self._train_epoch()
            val_metrics = self._validate_epoch()
            
            self.logger.log({
                "epoch": epoch,
                "train_loss": train_loss,
                **val_metrics
            })
            
            if self._should_checkpoint(epoch):
                self._save_checkpoint(epoch)
    
    def _setup_logging(self):
        if self.config.logging.wandb.enabled:
            return WandbLogger(self.config.logging.wandb)
        return LocalLogger(self.config.logging.local)
```

### Environment-Aware Execution
```python
def run_experiment(config_path: str):
    config = load_config(config_path)
    
    # Determine required environment
    model_type = config.model.type
    env_name = get_environment_for_model(model_type)
    
    if not is_environment_active(env_name):
        raise EnvironmentError(f"Please activate environment: {env_name}")
    
    # Load components
    model = create_model(config.model)
    dataset = create_dataset(config.dataset)
    task = create_task(config.task)
    
    # Train
    trainer = UniversalTrainer(model, dataset, task, config.training)
    trainer.train()
```

## 6. Modular Extension System

### Model Registration
```python
# src/core/registry.py
MODEL_REGISTRY = {}

def register_model(name: str):
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator

# In model implementations
@register_model("gps")
class GPSModel(BaseModel):
    pass

@register_model("autograph")
class AutoGraphModel(BaseModel):
    pass
```

### Plugin System for New Models
```python
# Adding a new model
# 1. Create src/models/new_model/
# 2. Implement adapter inheriting from BaseModel
# 3. Add environment config
# 4. Register model
# 5. Add experiment configs
```

## 7. Logging and Experiment Tracking

### Wandb Integration
```python
class WandbLogger:
    def __init__(self, config):
        wandb.init(
            project=config.project,
            entity=config.entity,
            tags=config.tags,
            config=config.experiment_config
        )
    
    def log_metrics(self, metrics: Dict[str, float], step: int):
        wandb.log(metrics, step=step)
    
    def log_artifact(self, path: str, name: str):
        artifact = wandb.Artifact(name, type="model")
        artifact.add_file(path)
        wandb.log_artifact(artifact)
```

### Comprehensive Logging
- Training metrics (loss, accuracy, etc.)
- Model checkpoints
- Configuration tracking
- Dataset metadata
- Computational resources (GPU usage, time)
- Hyperparameter sweeps

## 8. Development Workflow

### Makefile Commands
```makefile
# Environment setup
setup-all: setup-base setup-gps setup-autograph setup-sequence

setup-base:
	uv sync

setup-gps:
	conda env create -f environments/gps/environment.yml

# Training
train:
	python scripts/run_experiment.py --config $(CONFIG)

train-gps:
	conda activate gps && python scripts/run_experiment.py --config $(CONFIG)

# Data generation
generate-synthetic:
	python scripts/generate_synthetic_data.py --config $(CONFIG)

# Testing
test:
	pytest tests/

# Analysis
analyze:
	python scripts/evaluate_results.py --experiment $(EXP)

# Cleanup
clean:
	rm -rf results/checkpoints/* results/logs/*
```

## 9. Additional Considerations

### Performance Optimization
- Lazy loading of large datasets
- Efficient graph batching strategies
- GPU memory management
- Distributed training support (future)

### Reproducibility
- Seed management across all random number generators
- Environment version locking
- Configuration versioning
- Data version tracking

### Error Handling
- Graceful environment detection
- Model compatibility checking
- Data validation
- Checkpoint recovery

### Documentation
- Auto-generated API docs
- Experiment result summaries
- Model comparison reports
- Configuration examples

## 10. Implementation Phases

### Phase 1: Core Framework
1. Setup base environment management
2. Implement core abstractions
3. Create configuration system
4. Build basic trainer

### Phase 2: Model Integration
1. GPS model adapter
2. AutoGraph model adapter
3. Sequence model implementation
4. Model registry system

### Phase 3: Dataset & Task System
1. Dataset loaders for ZINC, synthetic tasks
2. Task implementations
3. Evaluation framework
4. Data generation utilities

### Phase 4: Advanced Features
1. Wandb integration
2. Hyperparameter sweeps
3. Model comparison tools
4. Result visualization

This architecture provides a scalable, modular foundation that supports your research goals while maintaining flexibility for future extensions.