"""Dataset implementations and loaders."""

# Import synthetic datasets
from .synthetic.universal_graphs import UniversalSyntheticDataset

# Real-world datasets will be imported here as they are implemented
# from .real_world import ZincDataset, IMDBDataset, CoraDataset

__all__ = [
    'UniversalSyntheticDataset'
]