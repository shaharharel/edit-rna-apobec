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

# --- EditRNA-A3A modular components ---

# Edit embedding (core innovation)
from .apobec_edit_embedding import APOBECEditEmbedding

# Encoders
from .encoders import (
    RNAFMEncoderWrapper,
    StructureGNNEncoder,
    ContactMapViT,
    MockRNAEncoder,
)

# Fusion
from .fusion import GatedModalityFusion, CrossAttentionFusion

# Prediction heads
from .prediction_heads import (
    BinaryEditHead,
    EditingRateHead,
    EnzymeSpecificityHead,
    StructureTypeHead,
    TissueSpecificityHead,
    NTissuesHead,
    FunctionalImpactHead,
    ConservationHead,
    CancerSurvivalHead,
    TissueRateHead,
    HEK293RateHead,
    EditEffectHead,
    APOBECMultiTaskHead,
    APOBECMultiTaskLoss,
    # Label encoding maps
    ENZYME_CLASSES,
    STRUCTURE_CLASSES,
    TISSUE_SPEC_CLASSES,
    FUNCTION_CLASSES,
    # Task name lists
    PRIMARY_TASKS,
    SECONDARY_TASKS,
    TERTIARY_TASKS,
    AUXILIARY_TASKS,
    ALL_TASKS,
)

# Multi-task tabular MLP (best-performing tabular model from Exp 2b-4)
from .multitask_tabular import (
    MultiTaskTabularMLP,
    UncertaintyWeightedLoss,
    DEFAULT_FEATURES,
    FEATURE_GROUPS,
    TASK_CONFIGS,
)

# Main model
from .editrna_a3a import (
    EditRNAConfig,
    EditRNA_A3A,
    create_editrna_rnafm,
    create_editrna_mock,
    create_editrna_utrlm,
)

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

    # EditRNA-A3A: Edit embedding
    'APOBECEditEmbedding',

    # EditRNA-A3A: Encoders
    'RNAFMEncoderWrapper',
    'StructureGNNEncoder',
    'ContactMapViT',
    'MockRNAEncoder',

    # EditRNA-A3A: Fusion
    'GatedModalityFusion',
    'CrossAttentionFusion',

    # EditRNA-A3A: Prediction heads (11 tasks)
    'BinaryEditHead',
    'EditingRateHead',
    'EnzymeSpecificityHead',
    'StructureTypeHead',
    'TissueSpecificityHead',
    'NTissuesHead',
    'FunctionalImpactHead',
    'ConservationHead',
    'CancerSurvivalHead',
    'TissueRateHead',
    'HEK293RateHead',
    'EditEffectHead',
    'APOBECMultiTaskHead',
    'APOBECMultiTaskLoss',
    # Label encoding maps
    'ENZYME_CLASSES',
    'STRUCTURE_CLASSES',
    'TISSUE_SPEC_CLASSES',
    'FUNCTION_CLASSES',
    # Task name lists
    'PRIMARY_TASKS',
    'SECONDARY_TASKS',
    'TERTIARY_TASKS',
    'AUXILIARY_TASKS',
    'ALL_TASKS',

    # Multi-task tabular MLP
    'MultiTaskTabularMLP',
    'UncertaintyWeightedLoss',
    'DEFAULT_FEATURES',
    'FEATURE_GROUPS',
    'TASK_CONFIGS',

    # EditRNA-A3A: Main model
    'EditRNAConfig',
    'EditRNA_A3A',
    'create_editrna_rnafm',
    'create_editrna_mock',
    'create_editrna_utrlm',
]
