"""
Machine learning models for RNA editing property prediction.
"""

# Predictors (each with their own PyTorch Lightning modules)
from .predictors import (
    PropertyPredictor,
    PropertyPredictorMLP,
    EditEffectPredictor,
    EditEffectMLP,
)

# Multi-task architectures
from .architectures import (
    SharedBackbone,
    TaskHead,
    MultiTaskHead,
)

# Training utilities
from .dataset import EditDataset, create_dataloaders, create_datasets_from_embeddings
from .trainer import Trainer

__all__ = [
    # Predictors
    'PropertyPredictor',
    'PropertyPredictorMLP',
    'EditEffectPredictor',
    'EditEffectMLP',

    # Multi-task architectures
    'SharedBackbone',
    'TaskHead',
    'MultiTaskHead',

    # Training
    'EditDataset',
    'create_dataloaders',
    'create_datasets_from_embeddings',
    'Trainer',
]
