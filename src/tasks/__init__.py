"""Task implementations."""

# Import implemented tasks
from .cycle_detection import CycleDetectionTask
from .shortest_path import ShortestPathTask

# Additional tasks will be imported here as they are implemented
# from .graph_classification import GraphClassificationTask
# from .node_classification import NodeClassificationTask

__all__ = [
    'CycleDetectionTask',
    'ShortestPathTask'
]