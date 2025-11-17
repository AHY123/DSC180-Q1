"""Adapters for external model implementations."""

# Import implemented adapters
from .gps_adapter import GPSAdapter
from .sequence_adapter import SequenceAdapter  
from .autograph_adapter import AutoGraphAdapter

__all__ = [
    'GPSAdapter',
    'SequenceAdapter',
    'AutoGraphAdapter'
]