"""
RNA embedding module for RNA editing prediction.

Sequence Embedders (sequence -> vector):
- NucleotideEmbedder: Simple baseline (one-hot + k-mer + stats)
- RNAFMEmbedder: RNA-FM pretrained embeddings (640-dim)
- RNABERTEmbedder: RNABERT pretrained embeddings (768-dim)
- UTRLMEmbedder: UTR-LM for 5' UTR sequences (128-dim)

Edit Embedders (edit -> vector):
- RNAEditEmbedder: Simple difference (emb_B - emb_A)
- PositionAwareEditEmbedder: One-hot features + RNA-FM per-token at edit position
- StructuredRNAEditEmbedder: Rich edit embeddings with mutation type, context, structure

Structure predictors:
- RNAplfoldPredictor: ViennaRNA-based local folding
- EternaFoldPredictor: EternaFold structure prediction

Factory functions:
- create_rnafm_structured_embedder(): StructuredRNAEditEmbedder with RNA-FM
- create_utrlm_structured_embedder(): StructuredRNAEditEmbedder with UTR-LM

APOBEC-specific:
- RNAFMEncoder: Lower-level nn.Module wrapper for end-to-end RNA-FM training
"""

from .base import RNAEmbedder
from .nucleotide import NucleotideEmbedder
from .edit_embedder import RNAEditEmbedder, TrainableRNAEditEmbedder
from .position_aware_edit_embedder import (
    PositionAwareEditEmbedder,
    create_edit_features,
    get_position_embeddings,
    EDIT_TYPES
)

# APOBEC-specific RNA-FM wrapper (nn.Module for end-to-end training)
from .rna_fm_encoder import RNAFMEncoder

# Conditionally import pretrained embedders
try:
    from .rnafm import RNAFMEmbedder
except ImportError:
    RNAFMEmbedder = None

try:
    from .rnabert import RNABERTEmbedder
except ImportError:
    RNABERTEmbedder = None

# Structured edit embedder
try:
    from .structured_edit_embedder import (
        StructuredRNAEditEmbedder,
        create_rnafm_structured_embedder,
        create_utrlm_structured_embedder,
        MUTATION_TYPES,
        NUC_TO_IDX
    )
except ImportError:
    StructuredRNAEditEmbedder = None
    create_rnafm_structured_embedder = None
    create_utrlm_structured_embedder = None
    MUTATION_TYPES = None
    NUC_TO_IDX = None

# Structure predictors
try:
    from .rnaplfold import RNAplfoldPredictor, RNAfoldPredictor
except ImportError:
    RNAplfoldPredictor = None
    RNAfoldPredictor = None

try:
    from .eternafold import EternaFoldPredictor, CombinedStructurePredictor
except ImportError:
    EternaFoldPredictor = None
    CombinedStructurePredictor = None

# UTR-LM embedder
try:
    from .utrlm import UTRLMEmbedder, load_utrlm
except ImportError:
    UTRLMEmbedder = None
    load_utrlm = None

# CodonBERT
try:
    from .codonbert import CodonBERTEmbedder, CodonDeltaEmbedder, load_codonbert
except ImportError:
    CodonBERTEmbedder = None
    CodonDeltaEmbedder = None
    load_codonbert = None

# Structure graph utilities (for GNN models)
try:
    from .structure_graph import (
        parse_dot_bracket,
        compute_structure_features,
        build_rna_graph,
        batch_build_graphs,
        StructureGraphBuilder
    )
except ImportError:
    parse_dot_bracket = None
    compute_structure_features = None
    build_rna_graph = None
    batch_build_graphs = None
    StructureGraphBuilder = None

__all__ = [
    # Base
    'RNAEmbedder',
    'NucleotideEmbedder',
    # Edit embedders
    'RNAEditEmbedder',
    'TrainableRNAEditEmbedder',
    'PositionAwareEditEmbedder',
    'StructuredRNAEditEmbedder',
    # Position-aware helpers
    'create_edit_features',
    'get_position_embeddings',
    'EDIT_TYPES',
    # Factory functions
    'create_rnafm_structured_embedder',
    'create_utrlm_structured_embedder',
    # Constants
    'MUTATION_TYPES',
    'NUC_TO_IDX',
    # Pretrained
    'RNAFMEmbedder',
    'RNABERTEmbedder',
    # Structure predictors
    'RNAplfoldPredictor',
    'RNAfoldPredictor',
    'EternaFoldPredictor',
    'CombinedStructurePredictor',
    # UTR-LM
    'UTRLMEmbedder',
    'load_utrlm',
    # CodonBERT
    'CodonBERTEmbedder',
    'CodonDeltaEmbedder',
    'load_codonbert',
    # Structure graph
    'parse_dot_bracket',
    'compute_structure_features',
    'build_rna_graph',
    'batch_build_graphs',
    'StructureGraphBuilder',
    # APOBEC-specific
    'RNAFMEncoder',
]
