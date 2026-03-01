#!/usr/bin/env python
"""Comprehensive cross-dataset and cross-species analysis for APOBEC editing.

Provides deep analysis of dataset relationships, generalization patterns,
and biological differences across all editing site datasets:

1. **Levanon (advisor) deep characterization**: 636-site core dataset profiling
2. **Per-dataset statistics**: Motif distribution, gene overlap, rate distributions
3. **Cross-dataset generalization matrix**: Train on A, test on B (NxN)
4. **Combined training experiments**: Impact of adding datasets
5. **Dataset-specific bias analysis**: TC-motif shift, coverage confounds
6. **Leave-one-dataset-out evaluation**
7. **Cross-species conservation analysis**

Usage:
    # Full analysis
    python scripts/apobec3a/cross_dataset_analysis.py

    # Only dataset characterization (no model training)
    python scripts/apobec3a/cross_dataset_analysis.py --characterize-only

    # Only cross-dataset generalization matrix
    python scripts/apobec3a/cross_dataset_analysis.py --generalization-only
"""

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logger = logging.getLogger(__name__)

# Paths
DATA_DIR = PROJECT_ROOT / "data" / "processed"
EMB_DIR = DATA_DIR / "embeddings"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "apobec" / "outputs" / "cross_dataset_analysis"

ALL_DATASETS = ["advisor_c2t", "asaoka_2019", "sharma_2015", "alqassim_2021"]
DATASET_LABELS = {
    "advisor_c2t": "Levanon (636 sites)",
    "asaoka_2019": "Asaoka 2019 (5,208 sites)",
    "sharma_2015": "Sharma 2015 (333 sites)",
    "alqassim_2021": "Alqassim 2021 (209 sites)",
    "baysal_2016": "Baysal 2016 (4,373 sites)",
}


# ---------------------------------------------------------------------------
# 1. Dataset Characterization
# ---------------------------------------------------------------------------

def characterize_datasets(combined_df: pd.DataFrame) -> Dict:
    """Produce comprehensive per-dataset characterization."""
    results = {}

    for ds_name in combined_df["dataset_source"].unique():
        subset = combined_df[combined_df["dataset_source"] == ds_name].copy()
        stats = {
            "n_sites": len(subset),
            "n_unique_genes": subset["gene"].dropna().nunique(),
            "n_chromosomes": subset["chr"].dropna().nunique(),
            "chromosomes": sorted(subset["chr"].dropna().unique().tolist()),
        }

        # Editing rate distribution
        rates = subset["editing_rate"].dropna()
        if len(rates) > 0:
            stats["rate_mean"] = float(rates.mean())
            stats["rate_median"] = float(rates.median())
            stats["rate_std"] = float(rates.std())
            stats["rate_min"] = float(rates.min())
            stats["rate_max"] = float(rates.max())
            stats["rate_q25"] = float(rates.quantile(0.25))
            stats["rate_q75"] = float(rates.quantile(0.75))
        else:
            stats["rate_mean"] = None

        # Feature/region distribution
        if "feature" in subset.columns:
            feat_counts = subset["feature"].value_counts().to_dict()
            stats["feature_distribution"] = feat_counts

        # Strand distribution
        if "strand" in subset.columns:
            strand_counts = subset["strand"].value_counts().to_dict()
            stats["strand_distribution"] = strand_counts

        results[ds_name] = stats

    return results


def characterize_levanon_deep(
    labels_df: pd.DataFrame,
    t1_df: Optional[pd.DataFrame] = None,
    t3_df: Optional[pd.DataFrame] = None,
    t5_df: Optional[pd.DataFrame] = None,
) -> Dict:
    """Deep characterization of the Levanon (advisor) dataset.

    Analyzes the 636-site core dataset in detail:
    - Multi-task label distributions
    - Tissue editing profiles
    - Structure types
    - Conservation patterns
    - Enzyme specificity
    - Cancer associations
    """
    stats = {"n_sites": len(labels_df)}

    # Genomic category distribution
    if "genomic_category" in labels_df.columns:
        stats["genomic_category"] = labels_df["genomic_category"].value_counts().to_dict()

    # Exonic function distribution
    if "exonic_function" in labels_df.columns:
        stats["exonic_function"] = labels_df["exonic_function"].value_counts().to_dict()

    # APOBEC class distribution
    if "apobec_class" in labels_df.columns:
        stats["apobec_class"] = labels_df["apobec_class"].value_counts().to_dict()

    # Structure type distribution
    if "structure_type" in labels_df.columns:
        stats["structure_type"] = labels_df["structure_type"].value_counts().to_dict()

    # Tissue class distribution
    if "tissue_class" in labels_df.columns:
        stats["tissue_class"] = labels_df["tissue_class"].value_counts().to_dict()

    # Conservation
    for col in ["any_mammalian_conservation", "any_primate_editing", "any_nonprimate_editing"]:
        if col in labels_df.columns:
            stats[col] = labels_df[col].value_counts().to_dict()

    # Editing rate statistics
    if "max_gtex_rate" in labels_df.columns:
        rates = labels_df["max_gtex_rate"].dropna()
        stats["gtex_rate"] = {
            "mean": float(rates.mean()),
            "median": float(rates.median()),
            "std": float(rates.std()),
            "min": float(rates.min()),
            "max": float(rates.max()),
        }

    # N tissues edited
    if "n_tissues_edited" in labels_df.columns:
        n_tiss = labels_df["n_tissues_edited"].dropna()
        stats["n_tissues_edited"] = {
            "mean": float(n_tiss.mean()),
            "median": float(n_tiss.median()),
            "min": int(n_tiss.min()),
            "max": int(n_tiss.max()),
        }

    # Cancer survival associations
    if "has_survival_association" in labels_df.columns:
        stats["has_survival_association"] = labels_df["has_survival_association"].value_counts().to_dict()

    if "n_cancer_types" in labels_df.columns:
        stats["n_cancer_types_mean"] = float(labels_df["n_cancer_types"].mean())
        stats["n_cancer_types_max"] = int(labels_df["n_cancer_types"].max())

    # Per-chromosome distribution
    if "chr" in labels_df.columns:
        stats["chromosome_distribution"] = labels_df["chr"].value_counts().to_dict()

    # Gene distribution (top 20)
    gene_col = None
    for c in ["gene_name", "gene_refseq", "gene"]:
        if c in labels_df.columns:
            gene_col = c
            break
    if gene_col:
        gene_counts = labels_df[gene_col].value_counts()
        stats["top_genes"] = gene_counts.head(20).to_dict()
        stats["n_unique_genes"] = int(gene_counts.count())
        stats["sites_per_gene_mean"] = float(gene_counts.mean())
        stats["sites_per_gene_median"] = float(gene_counts.median())

    return stats


# ---------------------------------------------------------------------------
# 2. Cross-Dataset Overlap Analysis
# ---------------------------------------------------------------------------

def analyze_cross_dataset_overlap(combined_df: pd.DataFrame) -> Dict:
    """Analyze coordinate and gene overlap between datasets."""
    results = {"coordinate_overlap": {}, "gene_overlap": {}}

    datasets = combined_df["dataset_source"].unique()

    # Coordinate-level overlap (chr:start)
    coord_sets = {}
    gene_sets = {}
    for ds in datasets:
        subset = combined_df[combined_df["dataset_source"] == ds]
        coord_sets[ds] = set(
            subset.apply(lambda r: f"{r['chr']}:{r['start']}", axis=1)
        )
        gene_sets[ds] = set(subset["gene"].dropna().unique())

    # Pairwise overlap
    for ds1 in datasets:
        for ds2 in datasets:
            if ds1 >= ds2:
                continue
            coord_overlap = coord_sets[ds1] & coord_sets[ds2]
            gene_overlap = gene_sets[ds1] & gene_sets[ds2]
            key = f"{ds1} & {ds2}"
            results["coordinate_overlap"][key] = {
                "n_shared_sites": len(coord_overlap),
                "pct_of_ds1": len(coord_overlap) / max(len(coord_sets[ds1]), 1) * 100,
                "pct_of_ds2": len(coord_overlap) / max(len(coord_sets[ds2]), 1) * 100,
            }
            results["gene_overlap"][key] = {
                "n_shared_genes": len(gene_overlap),
                "pct_of_ds1": len(gene_overlap) / max(len(gene_sets[ds1]), 1) * 100,
                "pct_of_ds2": len(gene_overlap) / max(len(gene_sets[ds2]), 1) * 100,
                "shared_genes_sample": sorted(list(gene_overlap))[:20],
            }

    return results


# ---------------------------------------------------------------------------
# 3. Motif and Sequence Context Analysis
# ---------------------------------------------------------------------------

def analyze_motif_distribution(
    combined_df: pd.DataFrame,
    sequences: Optional[Dict[str, str]] = None,
) -> Dict:
    """Analyze flanking dinucleotide motif distribution per dataset.

    This is critical for understanding the Sharma TC-motif shift confound.
    """
    if sequences is None:
        logger.warning("No sequences provided, skipping motif analysis")
        return {}

    results = {}

    for ds_name in combined_df["dataset_source"].unique():
        subset = combined_df[combined_df["dataset_source"] == ds_name]
        motif_counts = Counter()

        for _, row in subset.iterrows():
            sid = str(row["site_id"])
            if sid not in sequences:
                continue
            seq = sequences[sid]
            center = len(seq) // 2
            if center > 0 and center < len(seq):
                preceding = seq[center - 1].upper()
                motif = preceding + "C"
                motif_counts[motif] += 1

        total = sum(motif_counts.values())
        results[ds_name] = {
            "counts": dict(motif_counts),
            "fractions": {k: v / max(total, 1) for k, v in motif_counts.items()},
            "n_with_sequence": total,
            "tc_fraction": motif_counts.get("UC", 0) / max(total, 1),
        }

    return results


# ---------------------------------------------------------------------------
# 4. Cross-Dataset Generalization Matrix
# ---------------------------------------------------------------------------

def compute_generalization_matrix(
    splits_df: pd.DataFrame,
    pooled_orig: Dict,
    pooled_edited: Dict,
    neg_ids: Set[str],
    neg_ratio: int = 5,
    seed: int = 42,
) -> Dict:
    """Compute NxN cross-dataset generalization matrix.

    For each pair (train_dataset, test_dataset):
    - Train a simple model on train_dataset positives + shared negatives
    - Evaluate on test_dataset positives + held-out negatives

    Uses LogisticRegression on RNA-FM diff features for speed.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, average_precision_score

    rng = np.random.RandomState(seed)

    # Get dataset-specific site IDs
    available_ids = set(pooled_orig.keys()) & set(pooled_edited.keys())
    pos_by_dataset = {}

    for ds in splits_df["dataset_source"].unique():
        pos = splits_df[
            (splits_df["dataset_source"] == ds) &
            (splits_df["label"] == 1)
        ]
        ids = sorted(set(pos["site_id"]) & available_ids)
        if ids:
            pos_by_dataset[ds] = ids

    # Get negatives
    neg_available = sorted(neg_ids & available_ids)

    # Feature extraction helper
    def get_diff_features(site_ids):
        orig = np.stack([pooled_orig[sid].numpy() for sid in site_ids])
        edit = np.stack([pooled_edited[sid].numpy() for sid in site_ids])
        return edit - orig

    results = {"auroc": {}, "auprc": {}, "n_train": {}, "n_test": {}}
    datasets = sorted(pos_by_dataset.keys())

    for train_ds in datasets:
        for test_ds in datasets:
            train_pos = pos_by_dataset[train_ds]
            test_pos = pos_by_dataset[test_ds]

            # For same-dataset evaluation, use a 70/30 split
            if train_ds == test_ds:
                n_train = int(0.7 * len(train_pos))
                shuffled = list(train_pos)
                rng.shuffle(shuffled)
                train_pos_split = shuffled[:n_train]
                test_pos_split = shuffled[n_train:]
            else:
                train_pos_split = train_pos
                test_pos_split = test_pos

            if len(train_pos_split) < 5 or len(test_pos_split) < 5:
                continue

            # Sample negatives
            n_train_neg = min(len(train_pos_split) * neg_ratio, len(neg_available) // 2)
            n_test_neg = min(len(test_pos_split) * neg_ratio, len(neg_available) // 2)

            train_neg = list(rng.choice(neg_available, size=n_train_neg, replace=False))
            remaining_neg = [n for n in neg_available if n not in set(train_neg)]
            test_neg = list(rng.choice(remaining_neg,
                                       size=min(n_test_neg, len(remaining_neg)),
                                       replace=False))

            # Build features
            X_train = get_diff_features(train_pos_split + train_neg)
            y_train = np.array([1] * len(train_pos_split) + [0] * len(train_neg))

            X_test = get_diff_features(test_pos_split + test_neg)
            y_test = np.array([1] * len(test_pos_split) + [0] * len(test_neg))

            # Train and evaluate
            clf = LogisticRegression(max_iter=1000, C=1.0, random_state=seed)
            clf.fit(X_train, y_train)
            y_score = clf.predict_proba(X_test)[:, 1]

            key = f"{train_ds} -> {test_ds}"
            results["auroc"][key] = float(roc_auc_score(y_test, y_score))
            results["auprc"][key] = float(average_precision_score(y_test, y_score))
            results["n_train"][key] = len(y_train)
            results["n_test"][key] = len(y_test)

    return results


# ---------------------------------------------------------------------------
# 5. Combined Training Experiments
# ---------------------------------------------------------------------------

def run_combined_training_experiments(
    splits_df: pd.DataFrame,
    pooled_orig: Dict,
    pooled_edited: Dict,
    neg_ids: Set[str],
    seed: int = 42,
) -> Dict:
    """Evaluate impact of combining datasets for training.

    Experiments:
    1. Levanon only
    2. Levanon + each dataset individually
    3. All datasets combined
    4. Leave-one-dataset-out
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, average_precision_score

    rng = np.random.RandomState(seed)
    available_ids = set(pooled_orig.keys()) & set(pooled_edited.keys())

    # Get per-dataset positive IDs
    pos_by_dataset = {}
    for ds in splits_df["dataset_source"].unique():
        pos = splits_df[
            (splits_df["dataset_source"] == ds) & (splits_df["label"] == 1)
        ]
        ids = sorted(set(pos["site_id"]) & available_ids)
        if ids:
            pos_by_dataset[ds] = ids

    neg_available = sorted(neg_ids & available_ids)

    def get_diff_features(site_ids):
        orig = np.stack([pooled_orig[sid].numpy() for sid in site_ids])
        edit = np.stack([pooled_edited[sid].numpy() for sid in site_ids])
        return edit - orig

    def train_and_eval(train_pos_ids, test_pos_ids, train_neg, test_neg):
        X_train = get_diff_features(train_pos_ids + train_neg)
        y_train = np.array([1] * len(train_pos_ids) + [0] * len(train_neg))
        X_test = get_diff_features(test_pos_ids + test_neg)
        y_test = np.array([1] * len(test_pos_ids) + [0] * len(test_neg))

        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=seed)
        clf.fit(X_train, y_train)
        y_score = clf.predict_proba(X_test)[:, 1]

        return {
            "auroc": float(roc_auc_score(y_test, y_score)),
            "auprc": float(average_precision_score(y_test, y_score)),
            "n_train_pos": len(train_pos_ids),
            "n_test_pos": len(test_pos_ids),
        }

    # Split negatives
    half = len(neg_available) // 2
    train_neg = neg_available[:half]
    test_neg = neg_available[half:]

    results = {}

    # Test on each dataset individually
    for test_ds, test_ids in pos_by_dataset.items():
        # Use 30% of test dataset for testing
        n_test = max(int(0.3 * len(test_ids)), 5)
        shuffled = list(test_ids)
        rng.shuffle(shuffled)
        test_subset = shuffled[:n_test]
        remaining = shuffled[n_test:]

        # Experiment 1: Levanon only
        if "advisor_c2t" in pos_by_dataset:
            levanon_train = pos_by_dataset["advisor_c2t"]
            # Remove any overlap
            levanon_train_clean = [x for x in levanon_train if x not in set(test_subset)]
            if len(levanon_train_clean) >= 5:
                results[f"levanon_only -> {test_ds}"] = train_and_eval(
                    levanon_train_clean, test_subset, train_neg[:len(levanon_train_clean)*5], test_neg[:n_test*5]
                )

        # Experiment 2: Levanon + this dataset
        if "advisor_c2t" in pos_by_dataset and test_ds != "advisor_c2t":
            combined_train = pos_by_dataset["advisor_c2t"] + remaining
            combined_train = [x for x in combined_train if x not in set(test_subset)]
            if len(combined_train) >= 5:
                results[f"levanon+{test_ds} -> {test_ds}"] = train_and_eval(
                    combined_train, test_subset, train_neg[:len(combined_train)*5], test_neg[:n_test*5]
                )

        # Experiment 3: All datasets combined
        all_train = []
        for ds, ids in pos_by_dataset.items():
            all_train.extend(ids)
        all_train = [x for x in all_train if x not in set(test_subset)]
        all_train = list(set(all_train))
        if len(all_train) >= 5:
            results[f"all_combined -> {test_ds}"] = train_and_eval(
                all_train, test_subset, train_neg[:len(all_train)*3], test_neg[:n_test*5]
            )

        # Experiment 4: Leave-one-dataset-out
        lodo_train = []
        for ds, ids in pos_by_dataset.items():
            if ds != test_ds:
                lodo_train.extend(ids)
        lodo_train = [x for x in lodo_train if x not in set(test_subset)]
        lodo_train = list(set(lodo_train))
        if len(lodo_train) >= 5:
            results[f"leave_{test_ds}_out -> {test_ds}"] = train_and_eval(
                lodo_train, test_subset, train_neg[:len(lodo_train)*3], test_neg[:n_test*5]
            )

    return results


# ---------------------------------------------------------------------------
# 6. Cross-Species Conservation Analysis
# ---------------------------------------------------------------------------

def analyze_cross_species_conservation(labels_df: pd.DataFrame) -> Dict:
    """Analyze conservation patterns of editing sites across species."""
    results = {}

    # Conservation columns
    conserv_cols = [c for c in labels_df.columns if "conservation" in c.lower() or "primate" in c.lower() or "nonprimate" in c.lower()]
    for col in conserv_cols:
        results[col] = labels_df[col].value_counts().to_dict()

    # Cross-tabulate conservation with other features
    if "any_mammalian_conservation" in labels_df.columns and "structure_type" in labels_df.columns:
        ct = pd.crosstab(
            labels_df["any_mammalian_conservation"],
            labels_df["structure_type"],
        )
        results["conservation_x_structure"] = ct.to_dict()

    if "any_mammalian_conservation" in labels_df.columns and "apobec_class" in labels_df.columns:
        ct = pd.crosstab(
            labels_df["any_mammalian_conservation"],
            labels_df["apobec_class"],
        )
        results["conservation_x_enzyme"] = ct.to_dict()

    # Conservation level distribution
    if "conservation_level" in labels_df.columns:
        results["conservation_level_distribution"] = labels_df["conservation_level"].value_counts().to_dict()

    # Rate by conservation status
    if "any_mammalian_conservation" in labels_df.columns and "max_gtex_rate" in labels_df.columns:
        for conserved in [True, False]:
            mask = labels_df["any_mammalian_conservation"] == conserved
            rates = labels_df.loc[mask, "max_gtex_rate"].dropna()
            if len(rates) > 0:
                key = f"rate_conserved={conserved}"
                results[key] = {
                    "mean": float(rates.mean()),
                    "median": float(rates.median()),
                    "n": len(rates),
                }

    return results


# ---------------------------------------------------------------------------
# 7. Dataset Bias Analysis
# ---------------------------------------------------------------------------

def analyze_dataset_biases(
    combined_df: pd.DataFrame,
    sequences: Optional[Dict[str, str]] = None,
) -> Dict:
    """Identify potential confounds and biases in each dataset."""
    results = {}

    for ds_name in combined_df["dataset_source"].unique():
        subset = combined_df[combined_df["dataset_source"] == ds_name]
        bias = {}

        # Gene concentration (are sites clustered in few genes?)
        if "gene" in subset.columns:
            gene_counts = subset["gene"].value_counts()
            bias["top_gene"] = gene_counts.index[0] if len(gene_counts) > 0 else None
            bias["top_gene_count"] = int(gene_counts.iloc[0]) if len(gene_counts) > 0 else 0
            bias["top_gene_fraction"] = float(gene_counts.iloc[0] / len(subset)) if len(gene_counts) > 0 else 0
            bias["n_genes"] = int(gene_counts.count())
            bias["gini_concentration"] = float(_gini(gene_counts.values))

        # Chromosome concentration
        if "chr" in subset.columns:
            chrom_counts = subset["chr"].value_counts()
            bias["top_chromosome"] = chrom_counts.index[0] if len(chrom_counts) > 0 else None
            bias["top_chromosome_fraction"] = float(chrom_counts.iloc[0] / len(subset)) if len(chrom_counts) > 0 else 0

        # Strand bias
        if "strand" in subset.columns:
            strand_counts = subset["strand"].value_counts()
            plus = strand_counts.get("+", 0)
            minus = strand_counts.get("-", 0)
            bias["strand_plus_fraction"] = float(plus / max(plus + minus, 1))

        # Rate distribution shape (is it bimodal? right-skewed?)
        if "editing_rate" in subset.columns:
            rates = subset["editing_rate"].dropna()
            if len(rates) > 10:
                bias["rate_skewness"] = float(rates.skew())
                bias["rate_kurtosis"] = float(rates.kurtosis())

        # Sequence context (from motif analysis)
        if sequences is not None:
            tc_count = 0
            total = 0
            for _, row in subset.iterrows():
                sid = str(row["site_id"])
                if sid in sequences:
                    seq = sequences[sid]
                    center = len(seq) // 2
                    if center > 0:
                        preceding = seq[center - 1].upper()
                        if preceding + "C" == "UC":
                            tc_count += 1
                        total += 1
            if total > 0:
                bias["tc_motif_fraction"] = float(tc_count / total)
                bias["non_tc_fraction"] = float(1 - tc_count / total)

        results[ds_name] = bias

    return results


def _gini(values: np.ndarray) -> float:
    """Compute Gini concentration coefficient."""
    values = np.sort(values).astype(float)
    n = len(values)
    if n == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * values) / (n * np.sum(values))) - (n + 1) / n)


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_dataset_characterization(char_results: Dict, levanon_deep: Dict):
    """Print formatted dataset characterization."""
    print("\n" + "=" * 80)
    print("DATASET CHARACTERIZATION")
    print("=" * 80)

    for ds_name, stats in char_results.items():
        label = DATASET_LABELS.get(ds_name, ds_name)
        print(f"\n--- {label} ---")
        print(f"  Sites: {stats['n_sites']}")
        print(f"  Unique genes: {stats['n_unique_genes']}")
        print(f"  Chromosomes: {stats['n_chromosomes']}")
        if stats.get("rate_mean") is not None:
            print(f"  Rate: mean={stats['rate_mean']:.4f}, "
                  f"median={stats['rate_median']:.4f}, "
                  f"std={stats['rate_std']:.4f}")
        if "feature_distribution" in stats:
            print(f"  Features: {stats['feature_distribution']}")

    # Deep Levanon characterization
    print("\n" + "=" * 80)
    print("LEVANON (ADVISOR) DEEP CHARACTERIZATION")
    print("=" * 80)
    for key, val in levanon_deep.items():
        if isinstance(val, dict) and len(val) <= 10:
            print(f"  {key}: {val}")
        elif isinstance(val, (int, float, str)):
            print(f"  {key}: {val}")


def print_overlap_analysis(overlap: Dict):
    """Print cross-dataset overlap analysis."""
    print("\n" + "=" * 80)
    print("CROSS-DATASET OVERLAP")
    print("=" * 80)

    print("\n  Coordinate Overlap:")
    for key, val in overlap["coordinate_overlap"].items():
        print(f"    {key}: {val['n_shared_sites']} sites "
              f"({val['pct_of_ds1']:.1f}% / {val['pct_of_ds2']:.1f}%)")

    print("\n  Gene Overlap:")
    for key, val in overlap["gene_overlap"].items():
        print(f"    {key}: {val['n_shared_genes']} genes "
              f"({val['pct_of_ds1']:.1f}% / {val['pct_of_ds2']:.1f}%)")


def print_generalization_matrix(gen_results: Dict):
    """Print cross-dataset generalization matrix."""
    print("\n" + "=" * 80)
    print("CROSS-DATASET GENERALIZATION MATRIX (AUROC)")
    print("=" * 80)

    # Extract unique datasets
    datasets = set()
    for key in gen_results["auroc"]:
        train_ds, test_ds = key.split(" -> ")
        datasets.add(train_ds)
        datasets.add(test_ds)
    datasets = sorted(datasets)

    # Print header
    train_test_label = "Train \\ Test"
    header = f"{train_test_label:<20}"
    for ds in datasets:
        short = ds.replace("_20", "").replace("advisor_c2t", "levanon")[:12]
        header += f" {short:>12}"
    print(header)
    print("-" * (20 + 13 * len(datasets)))

    for train_ds in datasets:
        short_train = train_ds.replace("_20", "").replace("advisor_c2t", "levanon")[:20]
        row = f"{short_train:<20}"
        for test_ds in datasets:
            key = f"{train_ds} -> {test_ds}"
            auroc = gen_results["auroc"].get(key, float("nan"))
            if np.isnan(auroc):
                row += f" {'---':>12}"
            else:
                row += f" {auroc:>12.4f}"
        print(row)


def print_combined_training(combined_results: Dict):
    """Print combined training experiment results."""
    print("\n" + "=" * 80)
    print("COMBINED TRAINING EXPERIMENTS")
    print("=" * 80)
    print(f"{'Experiment':<50} {'AUROC':>8} {'AUPRC':>8} {'N_train':>8} {'N_test':>8}")
    print("-" * 82)

    for exp_name, metrics in sorted(combined_results.items()):
        print(f"{exp_name:<50} "
              f"{metrics['auroc']:>8.4f} {metrics['auprc']:>8.4f} "
              f"{metrics['n_train_pos']:>8} {metrics['n_test_pos']:>8}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive cross-dataset analysis for APOBEC editing"
    )
    parser.add_argument("--characterize-only", action="store_true",
                        help="Only run dataset characterization (no model training)")
    parser.add_argument("--generalization-only", action="store_true",
                        help="Only run cross-dataset generalization matrix")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--neg-ratio", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading datasets...")
    combined_df = pd.read_csv(DATA_DIR / "all_datasets_combined.csv")
    labels_df = pd.read_csv(DATA_DIR / "editing_sites_labels.csv")
    logger.info("  Combined: %d sites, Labels: %d sites", len(combined_df), len(labels_df))

    # Load supplementary tables for deep Levanon analysis
    t1_path = DATA_DIR / "advisor" / "t1_gtex_editing_&_conservation.csv"
    t1_df = pd.read_csv(t1_path) if t1_path.exists() else None

    t3_path = DATA_DIR / "advisor" / "supp_t3_structures.csv"
    t3_df = pd.read_csv(t3_path) if t3_path.exists() else None

    t5_path = DATA_DIR / "advisor" / "t5_tcga_survival.csv"
    t5_df = pd.read_csv(t5_path) if t5_path.exists() else None

    # Load sequences if available
    sequences = None
    seq_path = DATA_DIR / "site_sequences.json"
    if seq_path.exists():
        with open(seq_path) as f:
            sequences = json.load(f)
        logger.info("  Loaded %d sequences", len(sequences))

    all_results = {}

    # ---- 1. Dataset Characterization ----
    logger.info("Characterizing datasets...")
    char_results = characterize_datasets(combined_df)
    all_results["dataset_characterization"] = char_results

    levanon_deep = characterize_levanon_deep(labels_df, t1_df, t3_df, t5_df)
    all_results["levanon_deep"] = levanon_deep

    print_dataset_characterization(char_results, levanon_deep)

    # ---- 2. Cross-Dataset Overlap ----
    logger.info("Analyzing cross-dataset overlap...")
    overlap = analyze_cross_dataset_overlap(combined_df)
    all_results["overlap"] = overlap
    print_overlap_analysis(overlap)

    # ---- 3. Motif Distribution ----
    logger.info("Analyzing motif distributions...")
    motif_results = analyze_motif_distribution(combined_df, sequences)
    all_results["motif_distribution"] = motif_results

    if motif_results:
        print("\n" + "=" * 80)
        print("FLANKING MOTIF DISTRIBUTION")
        print("=" * 80)
        for ds, m in motif_results.items():
            label = DATASET_LABELS.get(ds, ds)
            tc_frac = m.get("tc_fraction", 0)
            print(f"  {label}: TC={tc_frac:.1%}, counts={m['counts']}")

    # ---- 4. Dataset Bias Analysis ----
    logger.info("Analyzing dataset biases...")
    bias_results = analyze_dataset_biases(combined_df, sequences)
    all_results["dataset_biases"] = bias_results

    print("\n" + "=" * 80)
    print("DATASET BIAS ANALYSIS")
    print("=" * 80)
    for ds, bias in bias_results.items():
        label = DATASET_LABELS.get(ds, ds)
        print(f"\n  {label}:")
        for k, v in bias.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.4f}")
            else:
                print(f"    {k}: {v}")

    # ---- 5. Cross-Species Conservation ----
    logger.info("Analyzing cross-species conservation...")
    conserv_results = analyze_cross_species_conservation(labels_df)
    all_results["cross_species_conservation"] = conserv_results

    if args.characterize_only:
        logger.info("Characterization-only mode. Saving results.")
        _save_results(all_results, output_dir)
        return

    # ---- Load embeddings for model-based analyses ----
    logger.info("Loading embeddings for model-based analyses...")
    pooled_orig = torch.load(EMB_DIR / "rnafm_pooled.pt", weights_only=False)
    pooled_edited = torch.load(EMB_DIR / "rnafm_pooled_edited.pt", weights_only=False)
    logger.info("  Loaded %d pooled embeddings", len(pooled_orig))

    # Load splits
    splits_path = DATA_DIR / "splits_expanded.csv"
    if not splits_path.exists():
        logger.error("splits_expanded.csv not found. Run expand_dataset.py first.")
        _save_results(all_results, output_dir)
        return

    splits_df = pd.read_csv(splits_path)
    neg_ids = set(
        splits_df[splits_df["label"] == 0]["site_id"]
    )

    # ---- 6. Cross-Dataset Generalization Matrix ----
    logger.info("Computing cross-dataset generalization matrix...")
    gen_results = compute_generalization_matrix(
        splits_df, pooled_orig, pooled_edited, neg_ids,
        neg_ratio=args.neg_ratio, seed=args.seed,
    )
    all_results["generalization_matrix"] = gen_results
    print_generalization_matrix(gen_results)

    # ---- 7. Combined Training Experiments ----
    if not args.generalization_only:
        logger.info("Running combined training experiments...")
        combined_results = run_combined_training_experiments(
            splits_df, pooled_orig, pooled_edited, neg_ids, seed=args.seed,
        )
        all_results["combined_training"] = combined_results
        print_combined_training(combined_results)

    # Save all results
    _save_results(all_results, output_dir)
    logger.info("\nAll results saved to %s", output_dir)


def _save_results(results: Dict, output_dir: Path):
    """Save results to JSON."""
    def _convert(obj):
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, set):
            return sorted(list(obj))
        return str(obj)

    with open(output_dir / "cross_dataset_analysis.json", "w") as f:
        json.dump(results, f, indent=2, default=_convert)


if __name__ == "__main__":
    main()
