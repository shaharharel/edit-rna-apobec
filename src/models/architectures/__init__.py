"""Neural network architectures for multi-task learning."""

from .multi_head import SharedBackbone, TaskHead, MultiTaskHead
from .hierarchical_rna import (
    RNAUNet,
    HierarchicalRNATransformer,
    HierarchicalGraphPooling,
    SetTransformerRNA,
    MultiscaleGraphTransformer,
    create_hierarchical_model,
)

__all__ = [
    # Multi-task
    'SharedBackbone',
    'TaskHead',
    'MultiTaskHead',
    # Hierarchical RNA
    'RNAUNet',
    'HierarchicalRNATransformer',
    'HierarchicalGraphPooling',
    'SetTransformerRNA',
    'MultiscaleGraphTransformer',
    'create_hierarchical_model',
]
