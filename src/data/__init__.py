"""
RNA data module for APOBEC editing.

Provides tools for:
- Loading and processing APOBEC C-to-U editing site data
- Extracting RNA sequence pairs with delta-expression labels
- RNA-specific sequence utilities
- Graph construction and caching for GNN models
"""

from .sequence_utils import (
    validate_rna_sequence,
    compute_edit_distance,
    extract_edit,
    compute_kozak_score,
    find_uaugs,
    reverse_complement,
    dna_to_rna,
    rna_to_dna,
)
from .dataset import RNAPairDataset
from .graph_cache import GraphCache
from .editing_sites import EditingSiteDataset, EditingSite, load_gtex_tissue_rates
from .rna_structure import (
    RNAStructure,
    predict_structure,
    batch_predict_structures,
    compute_structural_features,
    extract_sequence_context,
)
from .apobec_dataset import (
    APOBECDataset,
    APOBECDatasetBuilder,
    APOBECDataConfig,
    APOBECSiteSample,
    apobec_collate_fn,
    create_apobec_dataloaders,
    split_dataset,
)

__all__ = [
    # APOBEC ML dataset
    'APOBECDataset',
    'APOBECDatasetBuilder',
    'APOBECDataConfig',
    'APOBECSiteSample',
    'apobec_collate_fn',
    'create_apobec_dataloaders',
    'split_dataset',
    # APOBEC-specific
    'EditingSiteDataset',
    'EditingSite',
    'load_gtex_tissue_rates',
    # RNA structure
    'RNAStructure',
    'predict_structure',
    'batch_predict_structures',
    'compute_structural_features',
    'extract_sequence_context',
    # General RNA datasets
    'RNAPairDataset',
    'GraphCache',
    # Sequence utilities
    'validate_rna_sequence',
    'compute_edit_distance',
    'extract_edit',
    'compute_kozak_score',
    'find_uaugs',
    'reverse_complement',
    'dna_to_rna',
    'rna_to_dna',
]
