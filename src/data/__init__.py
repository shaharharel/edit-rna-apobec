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

__all__ = [
    # APOBEC-specific
    'EditingSiteDataset',
    'EditingSite',
    'load_gtex_tissue_rates',
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
