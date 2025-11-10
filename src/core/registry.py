"""Registry system for models, datasets, and tasks."""

from typing import Dict, Type, Any, Optional


class Registry:
    """Base registry class for registering components."""
    
    def __init__(self):
        self._registry: Dict[str, Type] = {}
    
    def register(self, name: str, cls: Type):
        """Register a class with the given name.
        
        Args:
            name: Registration name
            cls: Class to register
        """
        if name in self._registry:
            raise ValueError(f"Component '{name}' already registered")
        self._registry[name] = cls
    
    def get(self, name: str) -> Optional[Type]:
        """Get registered class by name.
        
        Args:
            name: Registration name
            
        Returns:
            Registered class or None
        """
        return self._registry.get(name)
    
    def create(self, name: str, config: Dict[str, Any]):
        """Create instance of registered class.
        
        Args:
            name: Registration name
            config: Configuration for instance creation
            
        Returns:
            Created instance
        """
        cls = self.get(name)
        if cls is None:
            raise ValueError(f"Component '{name}' not found in registry")
        return cls(config)
    
    def list_registered(self):
        """List all registered component names.
        
        Returns:
            List of registered names
        """
        return list(self._registry.keys())


class ModelRegistry(Registry):
    """Registry for graph learning models."""
    pass


class DatasetRegistry(Registry):
    """Registry for graph datasets."""
    pass


class TaskRegistry(Registry):
    """Registry for graph learning tasks."""
    pass


# Global registries
model_registry = ModelRegistry()
dataset_registry = DatasetRegistry()
task_registry = TaskRegistry()


def register_model(name: str):
    """Decorator to register a model.
    
    Args:
        name: Model name for registration
    """
    def decorator(cls):
        model_registry.register(name, cls)
        return cls
    return decorator


def register_dataset(name: str):
    """Decorator to register a dataset.
    
    Args:
        name: Dataset name for registration
    """
    def decorator(cls):
        dataset_registry.register(name, cls)
        return cls
    return decorator


def register_task(name: str):
    """Decorator to register a task.
    
    Args:
        name: Task name for registration
    """
    def decorator(cls):
        task_registry.register(name, cls)
        return cls
    return decorator