"""
Utility modules for RNA editing prediction.

Provides logging, metrics, splitting strategies, embedding caching, and normalization.
"""

from .logging import setup_logger
from .embedding_cache import (
    EmbeddingCache,
    get_or_compute_embeddings_for_pairs,
    get_or_compute_embeddings_for_molecules,
    compute_all_embeddings_once,
    compute_all_embeddings_with_fragments,
    map_embeddings_to_pairs,
    map_embeddings_to_molecules,
    map_fragment_embeddings_to_pairs
)
from .splits import (
    RNASplitter,
    RandomRNASplitter,
    SequenceSimilaritySplitter,
    MotifSplitter,
    EditTypeSplitter,
    GCStratifiedSplitter,
    LengthStratifiedSplitter,
    PositionSplitter,
    NeighborhoodContextSplitter,
    ExperimentalContextSplitter,
    EffectMagnitudeSplitter,
    NucleotideChangeSplitter,
    GeneralizationBenchmark,
    get_rna_splitter
)
from .metrics import (
    RegressionMetrics,
    MultiTaskMetrics,
    RankingMetrics,
    ChemistryMetrics,
    print_metrics_summary
)
from .apobec_eval import (
    binary_classification_metrics,
    evaluate_multitask,
    analyze_edit_embeddings,
    permutation_importance,
    analyze_misclassifications,
    compare_models,
)

__all__ = [
    'setup_logger',
    'EmbeddingCache',
    'get_or_compute_embeddings_for_pairs',
    'get_or_compute_embeddings_for_molecules',
    'compute_all_embeddings_once',
    'compute_all_embeddings_with_fragments',
    'map_embeddings_to_pairs',
    'map_embeddings_to_molecules',
    'map_fragment_embeddings_to_pairs',
    # RNA splitters
    'RNASplitter',
    'RandomRNASplitter',
    'SequenceSimilaritySplitter',
    'MotifSplitter',
    'EditTypeSplitter',
    'GCStratifiedSplitter',
    'LengthStratifiedSplitter',
    'PositionSplitter',
    'NeighborhoodContextSplitter',
    'ExperimentalContextSplitter',
    'EffectMagnitudeSplitter',
    'NucleotideChangeSplitter',
    'GeneralizationBenchmark',
    'get_rna_splitter',
    # Metrics
    'RegressionMetrics',
    'MultiTaskMetrics',
    'RankingMetrics',
    'ChemistryMetrics',
    'print_metrics_summary',
    # APOBEC evaluation
    'binary_classification_metrics',
    'evaluate_multitask',
    'analyze_edit_embeddings',
    'permutation_importance',
    'analyze_misclassifications',
    'compare_models',
]
