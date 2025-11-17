"""Experiment runner script for the graph learning platform."""

import os
import sys
import argparse
import yaml
from typing import Dict, Any

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.registry import model_registry, dataset_registry, task_registry


def load_config(config_path: str) -> Dict[str, Any]:
    """Load experiment configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load and merge base configurations
    if 'model' in config and 'config_file' in config['model']:
        model_config_path = os.path.join('experiments', config['model']['config_file'])
        if os.path.exists(model_config_path):
            with open(model_config_path, 'r') as f:
                base_model_config = yaml.safe_load(f)
                base_model_config.update(config['model'].get('overrides', {}))
                config['model'].update(base_model_config)
    
    if 'dataset' in config and 'config_file' in config['dataset']:
        dataset_config_path = os.path.join('experiments', config['dataset']['config_file'])
        if os.path.exists(dataset_config_path):
            with open(dataset_config_path, 'r') as f:
                base_dataset_config = yaml.safe_load(f)
                base_dataset_config.update(config['dataset'].get('overrides', {}))
                config['dataset'].update(base_dataset_config)
    
    if 'task' in config and 'config_file' in config['task']:
        task_config_path = os.path.join('experiments', config['task']['config_file'])
        if os.path.exists(task_config_path):
            with open(task_config_path, 'r') as f:
                base_task_config = yaml.safe_load(f)
                config['task'].update(base_task_config)
    
    return config


def check_environment(required_env: str):
    """Check if the required environment is active.
    
    Args:
        required_env: Required conda environment name
    """
    # This is a simplified check - in practice would verify conda env
    print(f"Note: This experiment expects '{required_env}' environment to be active")
    print("Ensure you have run the appropriate setup commands")


def create_components(config: Dict[str, Any]):
    """Create model, dataset, and task components from configuration.
    
    Args:
        config: Experiment configuration
        
    Returns:
        Tuple of (model, dataset, task)
    """
    # Create dataset
    print(f"Creating dataset: {config['dataset']['type']}")
    dataset = dataset_registry.create(config['dataset']['type'], config['dataset'])
    
    # Create task
    print(f"Creating task: {config['task']['type']}")
    task = task_registry.create(config['task']['type'], config['task'])
    
    # Create model
    print(f"Creating model: {config['model']['type']}")
    model = model_registry.create(config['model']['type'], config['model'])
    
    return model, dataset, task


def run_experiment(config_path: str):
    """Run a complete experiment.
    
    Args:
        config_path: Path to experiment configuration
    """
    print(f"Loading configuration from {config_path}")
    config = load_config(config_path)
    
    experiment_name = config['experiment']['name']
    print(f"Starting experiment: {experiment_name}")
    
    # Check environment requirements
    if 'environment' in config:
        check_environment(config['environment'])
    
    # Create components
    model, dataset, task = create_components(config)
    
    # Prepare data
    print("Preparing task data...")
    task_data = task.prepare_data(dataset)
    print(f"Prepared {len(task_data)} samples")
    
    # Get data splits
    print("Creating data splits...")
    train_data, val_data, test_data = dataset.get_splits()
    print(f"Data splits - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # For now, just test the pipeline - full training would be implemented here
    print("Testing model forward pass...")
    if len(task_data) > 0:
        sample_data = task_data[0]
        try:
            # Test model preprocessing and forward pass
            output = model.forward([sample_data])  # Pass as list for batch processing
            print(f"Model output shape: {output.shape if hasattr(output, 'shape') else type(output)}")
            
            # Test loss computation
            if hasattr(sample_data, 'y'):
                loss = model.loss(output, sample_data.y)
                print(f"Sample loss: {loss.item() if hasattr(loss, 'item') else loss}")
            
            # Test evaluation
            if hasattr(sample_data, 'y'):
                metrics = task.evaluate(output, sample_data.y.unsqueeze(0))
                print(f"Sample metrics: {metrics}")
                
        except Exception as e:
            print(f"Error in model forward pass: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"Experiment {experiment_name} pipeline test completed")
    
    # TODO: Implement full training loop
    print("Note: Full training loop not yet implemented")
    
    return model, dataset, task


def main():
    """Main function for experiment runner."""
    parser = argparse.ArgumentParser(description='Run graph learning experiments')
    parser.add_argument('--config', required=True,
                       help='Path to experiment configuration file')
    parser.add_argument('--dry-run', action='store_true',
                       help='Test configuration and pipeline without training')
    
    args = parser.parse_args()
    
    try:
        run_experiment(args.config)
    except Exception as e:
        print(f"Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())