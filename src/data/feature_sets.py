"""Clean feature set definitions for the feature leakage audit.

Defines which features are safe to use for each prediction task,
preventing information leakage in the tabular baseline experiments.

Feature Leakage Matrix:
======================

Feature Category           | Binary | Rate | Enzyme | Structure | Tissue | Conservation
--------------------------|--------|------|--------|-----------|--------|-------------
Genomic (cat_*, chrom_num) |   OK   |  OK  |   OK   |    OK     |   OK   |     OK
Tissue rates (max_rate...) | LEAKS  | LEAKS|   OK   |    OK     | LEAKS  |     OK
Structure type indicators  |   OK   | OK   |   OK   |  LEAKS    |   OK   |     OK
Conservation features      |   OK   | OK   |   OK   |    OK     |   OK   |   LEAKS
Cancer/survival features   |   OK   | OK   |   OK   |    OK     |   OK   |     OK
HEK293 rate                | LEAKS  | LEAKS|   OK   |    OK     |   OK   |     OK
N tissues edited           | LEAKS  | LEAKS|   OK   |    OK     | LEAKS  |     OK
Loop length                |   OK   | OK   |   OK   |  PARTIAL  |   OK   |     OK
Structure concordant       |   OK   | OK   |   OK   |  PARTIAL  |   OK   |     OK
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set


# ---------------------------------------------------------------------------
# Individual feature name constants
# ---------------------------------------------------------------------------

# Genomic / positional features -- always safe
GENOMIC_FEATURES: List[str] = [
    "cat_cds",
    "cat_noncoding",
    "cat_other",
    "chrom_num",
]

# Tissue-level editing rate summary statistics.
# These are computed from GTEx mismatch data and LEAK for:
#   - Binary task: negatives have tissue rates too, but the rate distributions
#     are fundamentally different (negatives = sequencing noise, positives =
#     real editing). Using rates to separate pos/neg is circular.
#   - Rate regression: predicting max_rate from mean_rate/median_rate/etc. is
#     trivially auto-correlated -- you're predicting a summary stat from other
#     summary stats of the same underlying signal.
#   - Tissue specificity: tissue_specificity and n_tissues_with_rate directly
#     encode the pattern we're trying to predict.
TISSUE_RATE_FEATURES: List[str] = [
    "max_rate",
    "mean_rate",
    "median_rate",
    "std_rate",
    "n_tissues_with_rate",
    "rate_q25",
    "rate_q75",
    "rate_iqr",
    "tissue_specificity",
    "mean_coverage",
    "max_coverage",
    "hek293_rate",
    "n_tissues_edited",
]

# Structure type one-hot indicators.
# LEAK for structure type classification (they ARE the label, one-hot encoded).
# PARTIAL leak for structure-adjacent features (loop_length, structure_concordant)
# because they correlate with structure type but are not identical to it.
STRUCTURE_TYPE_FEATURES: List[str] = [
    "is_in_loop",
    "is_dsrna",
    "is_ssrna_bulge",
    "is_open_ssrna",
]

# Structure-adjacent features -- not direct one-hot encodings of the label,
# but correlated with structure type. Safe for most tasks, partial leak for
# structure type prediction.
STRUCTURE_ADJACENT_FEATURES: List[str] = [
    "loop_length",
    "structure_concordant",
]

# Conservation and cross-species editing features.
# LEAK for conservation prediction tasks.
# For binary classification: these are only annotated for positive sites
# (negatives get NaN), so any non-NaN value trivially identifies a positive.
CONSERVATION_FEATURES: List[str] = [
    "mammalian_conservation",
    "primate_editing",
    "nonprimate_editing",
    "conservation_level",
]

# Cancer and clinical features.
# Same problem as conservation for binary task: only annotated for positives.
CANCER_FEATURES: List[str] = [
    "has_survival_assoc",
    "n_cancer_types",
]

# All features in the binary classification feature matrix
# (from build_feature_matrix in exp0_tabular_baseline.py)
ALL_BINARY_FEATURES: List[str] = (
    GENOMIC_FEATURES
    + TISSUE_RATE_FEATURES
    + STRUCTURE_TYPE_FEATURES
    + STRUCTURE_ADJACENT_FEATURES
    + CONSERVATION_FEATURES
    + CANCER_FEATURES
)

# All features in the positive-only feature matrix
# (from build_positive_feature_matrix in exp0_tabular_baseline.py)
# Note: this matrix uses raw rate columns instead of summary stats
POSITIVE_ONLY_RATE_FEATURES: List[str] = [
    "log2_max_rate",
    "max_gtex_rate",
    "mean_gtex_rate",
    "sd_gtex_rate",
    "n_tissues_edited",
]

ALL_POSITIVE_FEATURES: List[str] = (
    ["cat_cds", "cat_noncoding", "chrom_num"]  # no cat_other in positive matrix
    + POSITIVE_ONLY_RATE_FEATURES
    + STRUCTURE_TYPE_FEATURES
    + STRUCTURE_ADJACENT_FEATURES
    + CONSERVATION_FEATURES
    + CANCER_FEATURES
    + ["hek293_rate"]
)


# ---------------------------------------------------------------------------
# Feature set definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FeatureSet:
    """A named set of features with documentation of why they are included/excluded."""

    name: str
    description: str
    features: tuple  # frozen requires tuple, not list
    excluded: tuple = ()
    leakage_notes: str = ""

    @property
    def feature_list(self) -> List[str]:
        return list(self.features)

    @property
    def excluded_list(self) -> List[str]:
        return list(self.excluded)

    def __contains__(self, feature_name: str) -> bool:
        return feature_name in self.features

    def filter_columns(self, columns: List[str]) -> List[str]:
        """Return only columns that are in this feature set."""
        return [c for c in columns if c in self.features]


# -- SEQUENCE_ONLY --
# Minimal genomic category features. Used alongside RNA-FM / UTR-LM embeddings
# where the embedder provides the sequence representation.
SEQUENCE_ONLY = FeatureSet(
    name="sequence_only",
    description=(
        "Only genomic category features (CDS/noncoding/other) and chromosome. "
        "Safe for ALL tasks. Intended for use with RNA foundation model embeddings "
        "that capture sequence-level information."
    ),
    features=tuple(GENOMIC_FEATURES),
    excluded=tuple(
        TISSUE_RATE_FEATURES
        + STRUCTURE_TYPE_FEATURES
        + STRUCTURE_ADJACENT_FEATURES
        + CONSERVATION_FEATURES
        + CANCER_FEATURES
    ),
    leakage_notes="No leakage risk. These features are available for all sites.",
)

# -- STRUCTURE_SAFE --
# Genomic features + structure-adjacent features (loop_length, concordance)
# but NOT the structure type one-hot indicators.
STRUCTURE_SAFE = FeatureSet(
    name="structure_safe",
    description=(
        "Genomic features plus structure-adjacent features (loop_length, "
        "structure_concordant). Excludes structure type one-hot indicators "
        "to avoid leakage when predicting structure type. Safe for binary, "
        "rate, enzyme, tissue, and conservation tasks."
    ),
    features=tuple(GENOMIC_FEATURES + STRUCTURE_ADJACENT_FEATURES),
    excluded=tuple(
        TISSUE_RATE_FEATURES
        + STRUCTURE_TYPE_FEATURES
        + CONSERVATION_FEATURES
        + CANCER_FEATURES
    ),
    leakage_notes=(
        "loop_length and structure_concordant are correlated with structure type "
        "but are not the label itself. Partial leak for structure prediction."
    ),
)

# -- RATE_EXCLUDED --
# All features EXCEPT rate-related ones. For tasks where rate features leak.
RATE_EXCLUDED = FeatureSet(
    name="rate_excluded",
    description=(
        "All features except rate-related ones. Safe for binary classification "
        "when combined with careful handling of positive-only features, and for "
        "rate regression (where rate features are the target). Also safe for "
        "enzyme, structure, and conservation tasks."
    ),
    features=tuple(
        GENOMIC_FEATURES
        + STRUCTURE_TYPE_FEATURES
        + STRUCTURE_ADJACENT_FEATURES
        + CONSERVATION_FEATURES
        + CANCER_FEATURES
    ),
    excluded=tuple(TISSUE_RATE_FEATURES),
    leakage_notes=(
        "Removes all rate summary statistics (max_rate, mean_rate, median_rate, "
        "std_rate, n_tissues_with_rate, rate_q25, rate_q75, rate_iqr, "
        "tissue_specificity, mean_coverage, max_coverage, hek293_rate, "
        "n_tissues_edited). These leak for binary and rate prediction tasks."
    ),
)

# -- BINARY_SAFE --
# Features safe for binary editing site classification (positive vs negative).
# Must exclude:
#   1. Rate features: rate distributions are fundamentally different between
#      real editing sites and sequencing noise sites.
#   2. Conservation/cancer features: only annotated for positive sites, so
#      any non-NaN value trivially identifies a positive.
BINARY_SAFE = FeatureSet(
    name="binary_safe",
    description=(
        "Features safe for binary editing site classification. Excludes rate "
        "features (circular: rates differ by definition between edited and "
        "non-edited sites) and conservation/cancer features (only annotated "
        "for positive sites, trivially separating pos from neg). Retains "
        "genomic category, structure type indicators, loop length, and "
        "structure concordance."
    ),
    features=tuple(
        GENOMIC_FEATURES
        + STRUCTURE_TYPE_FEATURES
        + STRUCTURE_ADJACENT_FEATURES
    ),
    excluded=tuple(
        TISSUE_RATE_FEATURES
        + CONSERVATION_FEATURES
        + CANCER_FEATURES
    ),
    leakage_notes=(
        "Rate features leak because the rate distributions are fundamentally "
        "different between real editing sites (positives) and sequencing noise "
        "sites (negatives) -- using rates to classify is circular.\n"
        "Conservation/cancer features leak because they are only annotated for "
        "positive sites (negatives get NaN). Any imputation strategy still "
        "leaves the NaN-vs-real distinction as a trivial discriminator."
    ),
)

# -- FULL --
# All features, for reference / comparison. Acknowledges leakage.
FULL = FeatureSet(
    name="full",
    description=(
        "All features included. FOR REFERENCE ONLY -- this set has known "
        "leakage for binary classification (rate features, conservation/cancer "
        "features), rate regression (rate summary stats), structure prediction "
        "(structure one-hot indicators), and tissue specificity (rate features). "
        "Use only to quantify leakage impact by comparing against safe sets."
    ),
    features=tuple(ALL_BINARY_FEATURES),
    excluded=(),
    leakage_notes=(
        "KNOWN LEAKAGE in multiple tasks. Use only for leakage quantification."
    ),
)


# ---------------------------------------------------------------------------
# Registry and lookup
# ---------------------------------------------------------------------------

_FEATURE_SETS: Dict[str, FeatureSet] = {
    "sequence_only": SEQUENCE_ONLY,
    "structure_safe": STRUCTURE_SAFE,
    "rate_excluded": RATE_EXCLUDED,
    "binary_safe": BINARY_SAFE,
    "full": FULL,
}


def get_feature_set(name: str) -> FeatureSet:
    """Look up a feature set by name.

    Args:
        name: One of 'sequence_only', 'structure_safe', 'rate_excluded',
              'binary_safe', 'full'.

    Returns:
        The corresponding FeatureSet.

    Raises:
        KeyError: If name is not recognized.
    """
    if name not in _FEATURE_SETS:
        available = ", ".join(sorted(_FEATURE_SETS.keys()))
        raise KeyError(
            f"Unknown feature set '{name}'. Available: {available}"
        )
    return _FEATURE_SETS[name]


def list_feature_sets() -> List[str]:
    """Return names of all registered feature sets."""
    return sorted(_FEATURE_SETS.keys())


def get_safe_features(task_name: str) -> FeatureSet:
    """Return the recommended safe feature set for a given prediction task.

    Args:
        task_name: One of 'binary', 'rate', 'enzyme', 'structure',
                   'tissue', 'conservation'.

    Returns:
        The recommended FeatureSet that avoids leakage for this task.

    Raises:
        KeyError: If task_name is not recognized.
    """
    task_map: Dict[str, str] = {
        # Binary: exclude rates (circular) + conservation/cancer (positive-only annotations)
        "binary": "binary_safe",
        # Rate regression: exclude rate summary stats (auto-correlation)
        "rate": "rate_excluded",
        # Enzyme classification: genomic + structure features are safe;
        # rates are OK (not predicting rates). Use rate_excluded to be conservative,
        # or full if you accept that rate features don't leak for enzyme prediction.
        "enzyme": "rate_excluded",
        # Structure type: exclude structure one-hot indicators (they ARE the label)
        "structure": "structure_safe",
        # Tissue specificity: exclude rate features (they encode tissue patterns)
        "tissue": "binary_safe",
        # Conservation: exclude conservation features (they ARE the label)
        "conservation": "rate_excluded",
    }
    if task_name not in task_map:
        available = ", ".join(sorted(task_map.keys()))
        raise KeyError(
            f"Unknown task '{task_name}'. Available: {available}"
        )
    return _FEATURE_SETS[task_map[task_name]]


# ---------------------------------------------------------------------------
# Leakage report
# ---------------------------------------------------------------------------

FEATURE_LEAKAGE_REPORT: str = """
================================================================================
FEATURE LEAKAGE ANALYSIS -- APOBEC TABULAR BASELINE
================================================================================

This report documents all identified sources of information leakage in the
feature set used by exp0_tabular_baseline.py.

1. BINARY CLASSIFICATION (positive vs negative editing sites)
   ---------------------------------------------------------
   LEAKED FEATURES:
   - Tissue rate features (max_rate, mean_rate, median_rate, std_rate,
     n_tissues_with_rate, rate_q25, rate_q75, rate_iqr, tissue_specificity,
     mean_coverage, max_coverage):
       Positive sites are real C-to-U editing events; negative sites are
       unedited cytidines. By definition, positives have measurable editing
       rates while negatives have only sequencing noise. Using rate features
       to classify is circular -- you're using the label (edited vs not) as
       a feature.
   - hek293_rate:
       Same as above. HEK293 editing rate is a direct measurement of editing.
   - n_tissues_edited:
       Number of tissues showing editing. Positives have n >= 1; negatives
       should have n = 0 (or noise). Trivial discriminator.
   - Conservation features (mammalian_conservation, primate_editing,
     nonprimate_editing, conservation_level):
       Only annotated for positive sites. For negatives, these are set to NaN.
       After imputation, the NaN-vs-real-value distinction trivially separates
       positives from negatives.
   - Cancer features (has_survival_assoc, n_cancer_types):
       Same issue as conservation -- only annotated for positive sites.

   SAFE FEATURES for binary task:
   - cat_cds, cat_noncoding, cat_other, chrom_num (genomic position)
   - is_in_loop, is_dsrna, is_ssrna_bulge, is_open_ssrna (structure type)
   - loop_length, structure_concordant (structure-adjacent)
   NOTE: Structure features are NaN for negatives too, but this reflects a
   genuine biological question (we don't know the structure context of
   non-edited sites). This is a data limitation, not circular reasoning.

   IMPACT: The exp0 baseline reports AUROC ~0.99 on binary classification.
   With BINARY_SAFE features, expect AUROC to drop substantially, revealing
   the true difficulty of the prediction task and the headroom for neural
   models.


2. RATE REGRESSION (predicting log2_max_rate on positive sites)
   -------------------------------------------------------------
   LEAKED FEATURES:
   - max_gtex_rate, mean_gtex_rate, sd_gtex_rate:
       These are summary statistics of the same underlying GTEx rate data.
       Predicting max_rate from mean_rate is trivially auto-correlated.
   - n_tissues_edited, tissue_specificity, hek293_rate:
       Strongly correlated with the target variable by construction.

   SAFE FEATURES for rate task:
   - All non-rate features: genomic, structure, conservation, cancer.

   IMPACT: Rate regression without rate features will rely on structure and
   conservation features. This is the scientifically meaningful question:
   can structural/conservation context predict editing efficiency?


3. STRUCTURE TYPE CLASSIFICATION (positive sites only)
   ----------------------------------------------------
   LEAKED FEATURES:
   - is_in_loop, is_dsrna, is_ssrna_bulge, is_open_ssrna:
       These ARE the one-hot encoding of the structure type label.
       Using them as features is using the label to predict the label.

   PARTIALLY LEAKED:
   - loop_length: Strongly associated with "In Loop" class. Sites in loops
     have defined loop lengths; other structure types may have NaN.
   - structure_concordant: Correlated with structure type but not identical.

   SAFE FEATURES for structure task:
   - Genomic features, rate features, conservation features, cancer features.
   - loop_length and structure_concordant with caution (partial leak).


4. TISSUE SPECIFICITY CLASSIFICATION (positive sites only)
   --------------------------------------------------------
   LEAKED FEATURES:
   - n_tissues_with_rate, n_tissues_edited:
       Tissue specificity classes (Blood Specific, Ubiquitous, etc.) are
       defined by the tissue editing pattern. These count features directly
       encode that pattern.
   - tissue_specificity:
       Literally the ratio max_rate/mean_rate, encoding specificity.
   - All rate summary statistics (max_rate, mean_rate, etc.):
       The tissue specificity classes are derived from the per-tissue rate
       profile, so aggregate rate statistics leak the class.

   SAFE FEATURES for tissue task:
   - Genomic features, structure features, conservation features, cancer
     features. The question becomes: can structure/conservation predict
     tissue specificity?


5. ENZYME CLASSIFICATION (APOBEC3A Only / APOBEC3G Only / Both / Neither)
   -----------------------------------------------------------------------
   No direct leakage identified. Rate features, structure features, and
   conservation features are all legitimate predictors of enzyme preference.
   The RATE_EXCLUDED set is recommended as a conservative choice.


6. CONSERVATION PREDICTION
   ------------------------
   LEAKED FEATURES:
   - mammalian_conservation, primate_editing, nonprimate_editing,
     conservation_level:
       These ARE the conservation annotations. Using them to predict
       conservation is circular.

   SAFE FEATURES for conservation task:
   - Genomic features, rate features, structure features, cancer features.


RECOMMENDATIONS
===============
1. Always use task-appropriate feature sets via get_safe_features(task_name).
2. Report results with BOTH full and safe feature sets to quantify leakage.
3. The gap between full and safe performance reveals how much the baseline
   was inflated by leakage -- this is the true headroom for neural models.
4. For the binary task especially, the safe baseline is the honest bar that
   sequence-based models (RNA-FM, UTR-LM + edit embeddings) must beat.
================================================================================
"""
