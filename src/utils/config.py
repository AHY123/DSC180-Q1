"""Configuration management utilities."""

import os
import yaml
from typing import Dict, Any, Union


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two configuration dictionaries.
    
    Args:
        base_config: Base configuration
        override_config: Override configuration
        
    Returns:
        Merged configuration dictionary
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def save_config(config: Dict[str, Any], config_path: str):
    """Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Output file path
    """
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def validate_config(config: Dict[str, Any], required_keys: Dict[str, type]) -> bool:
    """Validate configuration has required keys with correct types.
    
    Args:
        config: Configuration to validate
        required_keys: Dictionary of key -> expected type
        
    Returns:
        True if valid, raises ValueError if invalid
    """
    for key, expected_type in required_keys.items():
        if key not in config:
            raise ValueError(f"Required configuration key missing: {key}")
        
        if not isinstance(config[key], expected_type):
            raise ValueError(f"Configuration key '{key}' should be {expected_type.__name__}, "
                           f"got {type(config[key]).__name__}")
    
    return True