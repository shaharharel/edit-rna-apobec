#!/usr/bin/env python
"""Tabular baseline for APOBEC editing site prediction (Experiment 0).

Trains Random Forest and XGBoost classifiers on hand-engineered features
to establish a non-deep-learning baseline. This baseline serves as:
1. Lower bound for the neural edit-effect model to beat
2. Feature importance analysis to identify most informative signals
3. Quick iteration on feature engineering

Features used (available for both positives and negatives):
- Genomic category (CDS, Non-Coding, Other)
- Tissue-level mismatch/editing rates (max, mean, std, n_tissues)
- Chromosome (encoded)
- For positive sites: structure features, conservation, APOBEC class

Usage:
    python experiments/apobec/tabular_baseline.py
    python experiments/apobec/tabular_baseline.py --output_dir experiments/apobec/outputs/baseline_v1
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logger = logging.getLogger(__name__)

# Tissue columns present in negative_controls_ct.csv
TISSUE_COLUMNS = [
    "Adipose_Subcutaneous", "Adipose_Visceral_Omentum", "Adrenal_Gland",
    "Artery_Aorta", "Artery_Coronary", "Artery_Tibial", "Bladder",
    "Brain_Amygdala", "Brain_Anterior_cingulate_cortex_BA24",
    "Brain_Caudate_basal_ganglia", "Brain_Cerebellar_Hemisphere",
    "Brain_Cerebellum", "Brain_Cortex", "Brain_Frontal_Cortex_BA9",
    "Brain_Hippocampus", "Brain_Hypothalamus",
    "Brain_Nucleus_accumbens_basal_ganglia", "Brain_Putamen_basal_ganglia",
    "Brain_Spinal_cord_cervical_c-1", "Brain_Substantia_nigra",
    "Breast_Mammary_Tissue", "Cells_Cultured_fibroblasts",
    "Cells_EBV-transformed_lymphocytes", "Cervix_Ectocervix",
    "Cervix_Endocervix", "Colon_Sigmoid", "Colon_Transverse",
    "Esophagus_Gastroesophageal_Junction", "Esophagus_Mucosa",
    "Esophagus_Muscularis", "Fallopian_Tube", "Heart_Atrial_Appendage",
    "Heart_Left_Ventricle", "Kidney_Cortex", "Kidney_Medulla", "Liver",
    "Lung", "Minor_Salivary_Gland", "Muscle_Skeletal", "Nerve_Tibial",
    "Ovary", "Pancreas", "Pituitary", "Prostate",
    "Skin_Not_Sun_Exposed_Suprapubic", "Skin_Sun_Exposed_Lower_leg",
    "Small_Intestine_Terminal_Ileum", "Spleen", "Stomach", "Testis",
    "Thyroid", "Uterus", "Vagina", "Whole_Blood",
]


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def parse_tissue_rate(val) -> Tuple[float, float, float]:
    """Parse a tissue rate value, handling 'mismatch;coverage;rate' format.

    Returns (rate, coverage, mismatch_count). Rate in percentage.
    """
    if pd.isna(val) or str(val).strip() == "":
        return np.nan, np.nan, np.nan

    s = str(val).strip()
    if ";" in s:
        parts = s.split(";")
        if len(parts) == 3:
            try:
                mismatch = float(parts[0])
                coverage = float(parts[1])
                rate = float(parts[2])
                return rate, coverage, mismatch
            except ValueError:
                return np.nan, np.nan, np.nan
    try:
        return float(s), np.nan, np.nan
    except ValueError:
        return np.nan, np.nan, np.nan


def extract_tissue_features(row: pd.Series, tissue_cols: List[str]) -> Dict[str, float]:
    """Extract aggregated tissue-level features from a row."""
    rates = []
    coverages = []
    mismatches = []

    for tc in tissue_cols:
        if tc in row.index:
            rate, cov, mm = parse_tissue_rate(row[tc])
            if not np.isnan(rate):
                rates.append(rate)
            if not np.isnan(cov):
                coverages.append(cov)
            if not np.isnan(mm):
                mismatches.append(mm)

    features = {}

    # Rate statistics
    if rates:
        features["max_rate"] = max(rates)
        features["mean_rate"] = np.mean(rates)
        features["median_rate"] = np.median(rates)
        features["std_rate"] = np.std(rates) if len(rates) > 1 else 0.0
        features["n_tissues_with_rate"] = len(rates)
        features["rate_q25"] = np.percentile(rates, 25)
        features["rate_q75"] = np.percentile(rates, 75)
        features["rate_iqr"] = features["rate_q75"] - features["rate_q25"]
        # Tissue specificity: high max but low mean = tissue-specific
        features["tissue_specificity"] = features["max_rate"] / (features["mean_rate"] + 1e-6)
    else:
        for k in ["max_rate", "mean_rate", "median_rate", "std_rate",
                   "n_tissues_with_rate", "rate_q25", "rate_q75", "rate_iqr",
                   "tissue_specificity"]:
            features[k] = 0.0

    # Coverage statistics
    if coverages:
        features["mean_coverage"] = np.mean(coverages)
        features["max_coverage"] = max(coverages)
    else:
        features["mean_coverage"] = 0.0
        features["max_coverage"] = 0.0

    return features


def extract_positive_features(row: pd.Series) -> Dict[str, float]:
    """Extract features specific to positive sites from labels CSV."""
    features = {}

    # Structure type (one-hot)
    structure_type = str(row.get("structure_type", "")).lower()
    features["is_stem_loop"] = 1.0 if "stem" in structure_type and "loop" in structure_type else 0.0
    features["is_dsrna"] = 1.0 if "dsrna" in structure_type else 0.0
    features["is_open_ssrna"] = 1.0 if "open" in structure_type or "ssrna" in structure_type else 0.0

    # Loop length (can be NaN)
    features["loop_length"] = float(row.get("loop_length", np.nan))

    # Structure concordance
    concordance = str(row.get("structure_concordance", "")).lower()
    features["structure_concordant"] = 1.0 if concordance == "concordant" else 0.0

    # Conservation
    features["mammalian_conservation"] = 1.0 if row.get("any_mammalian_conservation", False) else 0.0
    features["primate_editing"] = 1.0 if row.get("any_primate_editing", False) else 0.0
    features["nonprimate_editing"] = 1.0 if row.get("any_nonprimate_editing", False) else 0.0
    features["conservation_level"] = float(row.get("conservation_level", 0))

    # Cancer survival
    features["has_survival_assoc"] = 1.0 if row.get("has_survival_association", False) else 0.0
    features["n_cancer_types"] = float(row.get("n_cancer_types", 0))

    # HEK293 rate
    features["hek293_rate"] = float(row.get("hek293_rate", np.nan))

    # Tissue editing breadth
    features["n_tissues_edited"] = float(row.get("n_tissues_edited", 0))

    return features


def build_feature_matrix(
    labels_df: pd.DataFrame,
    negatives_df: pd.DataFrame,
    splits_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Build the feature matrix for all samples.

    Returns:
        (features_df, labels, splits, site_ids)
    """
    # Find matching tissue columns in the data
    neg_tissue_cols = [c for c in TISSUE_COLUMNS if c in negatives_df.columns]

    all_features = []
    all_labels = []
    all_splits = []
    all_site_ids = []

    site_to_split = dict(zip(splits_df["site_id"], splits_df["split"]))

    # Process positive sites
    # Load GTEx tissue data from T1
    t1_path = PROJECT_ROOT / "data" / "processed" / "advisor" / "t1_gtex_editing_&_conservation.csv"
    t1_df = None
    if t1_path.exists():
        t1_df = pd.read_csv(t1_path)
        # Build lookup by coordinates
        t1_key = t1_df.apply(lambda r: f"{r.get('Chr', r.get('chr', ''))}:{r.get('Start', r.get('start', ''))}", axis=1)
        t1_df["_key"] = t1_key

    for _, row in labels_df.iterrows():
        site_id = str(row["site_id"])
        features = {}

        # Genomic category
        cat = str(row.get("genomic_category", "")).lower()
        features["cat_cds"] = 1.0 if "cds" in cat else 0.0
        features["cat_noncoding"] = 1.0 if "non" in cat and "coding" in cat else 0.0
        features["cat_other"] = 1.0 if features["cat_cds"] == 0 and features["cat_noncoding"] == 0 else 0.0

        # Chromosome
        chrom = str(row.get("chr", ""))
        features["chrom_num"] = _chrom_to_num(chrom)

        # Tissue rate features
        # Try to get from T1 data
        coord_key = f"{row['chr']}:{row['start']}"
        tissue_features = {}
        if t1_df is not None and coord_key in t1_df["_key"].values:
            t1_row = t1_df[t1_df["_key"] == coord_key].iloc[0]
            t1_tissue_cols = [c for c in TISSUE_COLUMNS if c in t1_df.columns]
            tissue_features = extract_tissue_features(t1_row, t1_tissue_cols)
        else:
            # Use labels CSV values
            tissue_features["max_rate"] = float(row.get("max_gtex_rate", 0))
            tissue_features["mean_rate"] = float(row.get("mean_gtex_rate", 0))
            tissue_features["std_rate"] = float(row.get("sd_gtex_rate", 0))
            tissue_features["n_tissues_with_rate"] = float(row.get("n_tissues_edited", 0))
            tissue_features["median_rate"] = tissue_features["mean_rate"]
            tissue_features["rate_q25"] = 0.0
            tissue_features["rate_q75"] = tissue_features["max_rate"]
            tissue_features["rate_iqr"] = tissue_features["rate_q75"] - tissue_features["rate_q25"]
            tissue_features["tissue_specificity"] = (
                tissue_features["max_rate"] / (tissue_features["mean_rate"] + 1e-6)
            )
            tissue_features["mean_coverage"] = 0.0
            tissue_features["max_coverage"] = 0.0

        features.update(tissue_features)

        # Positive-specific features
        pos_features = extract_positive_features(row)
        features.update(pos_features)

        all_features.append(features)
        all_labels.append(1)
        all_splits.append(site_to_split.get(site_id, "train"))
        all_site_ids.append(site_id)

    # Process negative sites
    for _, row in negatives_df.iterrows():
        site_id = str(row["site_id"])
        features = {}

        # Genomic category
        cat = str(row.get("genomic_category", "")).lower()
        features["cat_cds"] = 1.0 if "cds" in cat else 0.0
        features["cat_noncoding"] = 1.0 if "non" in cat and "coding" in cat else 0.0
        features["cat_other"] = 1.0 if features["cat_cds"] == 0 and features["cat_noncoding"] == 0 else 0.0

        # Chromosome
        chrom = str(row.get("chr", ""))
        features["chrom_num"] = _chrom_to_num(chrom)

        # Tissue rate features from columns
        tissue_features = extract_tissue_features(row, neg_tissue_cols)
        features.update(tissue_features)

        # Positive-specific features default to 0/NaN for negatives
        features["is_stem_loop"] = np.nan  # unknown
        features["is_dsrna"] = np.nan
        features["is_open_ssrna"] = np.nan
        features["loop_length"] = np.nan
        features["structure_concordant"] = np.nan
        features["mammalian_conservation"] = np.nan
        features["primate_editing"] = np.nan
        features["nonprimate_editing"] = np.nan
        features["conservation_level"] = np.nan
        features["has_survival_assoc"] = np.nan
        features["n_cancer_types"] = np.nan
        features["hek293_rate"] = np.nan
        features["n_tissues_edited"] = np.nan

        all_features.append(features)
        all_labels.append(0)
        all_splits.append(site_to_split.get(site_id, "train"))
        all_site_ids.append(site_id)

    features_df = pd.DataFrame(all_features)
    labels_arr = np.array(all_labels)
    splits_arr = np.array(all_splits)
    site_ids_arr = np.array(all_site_ids)

    return features_df, labels_arr, splits_arr, site_ids_arr


def _chrom_to_num(chrom: str) -> float:
    """Convert chromosome string to numeric value."""
    chrom = chrom.lower().replace("chr", "")
    if chrom == "x":
        return 23
    elif chrom == "y":
        return 24
    elif chrom == "m" or chrom == "mt":
        return 25
    try:
        return float(chrom)
    except ValueError:
        return 0


# ---------------------------------------------------------------------------
# Model training and evaluation
# ---------------------------------------------------------------------------

def train_and_evaluate(
    features_df: pd.DataFrame,
    labels: np.ndarray,
    splits: np.ndarray,
    output_dir: Path,
    feature_set: str = "all",
) -> Dict:
    """Train RF and XGBoost and evaluate on val/test splits.

    Args:
        features_df: Feature matrix (samples x features)
        labels: Binary labels
        splits: Split assignments ('train', 'val', 'test')
        output_dir: Directory for saving results
        feature_set: Which features to use ('all', 'common_only')

    Returns:
        Dict with results for each model
    """
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        classification_report,
        f1_score,
        precision_recall_curve,
        roc_auc_score,
    )
    from sklearn.impute import SimpleImputer

    # Select feature columns based on feature set
    if feature_set == "common_only":
        # Features available for both positives and negatives
        common_cols = [c for c in features_df.columns if c not in [
            "is_stem_loop", "is_dsrna", "is_open_ssrna", "loop_length",
            "structure_concordant", "mammalian_conservation", "primate_editing",
            "nonprimate_editing", "conservation_level", "has_survival_assoc",
            "n_cancer_types", "hek293_rate", "n_tissues_edited",
        ]]
        X = features_df[common_cols].copy()
        feature_names = common_cols
    else:
        X = features_df.copy()
        feature_names = list(features_df.columns)

    # Split
    train_mask = splits == "train"
    val_mask = splits == "val"
    test_mask = splits == "test"

    X_train, y_train = X[train_mask].values, labels[train_mask]
    X_val, y_val = X[val_mask].values, labels[val_mask]
    X_test, y_test = X[test_mask].values, labels[test_mask]

    logger.info("Feature set: %s (%d features)", feature_set, len(feature_names))
    logger.info("  Train: %d (pos=%d, neg=%d)", len(y_train), y_train.sum(), len(y_train) - y_train.sum())
    logger.info("  Val:   %d (pos=%d, neg=%d)", len(y_val), y_val.sum(), len(y_val) - y_val.sum())
    logger.info("  Test:  %d (pos=%d, neg=%d)", len(y_test), y_test.sum(), len(y_test) - y_test.sum())

    # Impute NaN values
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_val = imputer.transform(X_val)
    X_test = imputer.transform(X_test)

    # Class weight for imbalance
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    class_weight = {0: 1.0, 1: n_neg / n_pos}

    results = {}

    # --- Random Forest ---
    logger.info("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    rf_val_probs = rf.predict_proba(X_val)[:, 1]
    rf_test_probs = rf.predict_proba(X_test)[:, 1]

    rf_val_metrics = _compute_metrics(y_val, rf_val_probs)
    rf_test_metrics = _compute_metrics(y_test, rf_test_probs)

    # Feature importance
    rf_importance = pd.DataFrame({
        "feature": feature_names,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)

    results["random_forest"] = {
        "val": rf_val_metrics,
        "test": rf_test_metrics,
        "feature_importance": rf_importance.to_dict(orient="records"),
    }

    logger.info("  RF Val:  AUROC=%.4f  AUPRC=%.4f  F1=%.4f",
                rf_val_metrics["auroc"], rf_val_metrics["auprc"], rf_val_metrics["f1"])
    logger.info("  RF Test: AUROC=%.4f  AUPRC=%.4f  F1=%.4f",
                rf_test_metrics["auroc"], rf_test_metrics["auprc"], rf_test_metrics["f1"])

    # --- Gradient Boosting ---
    logger.info("Training Gradient Boosting...")
    gb = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
    )
    # Apply sample weights for class balance
    sample_weights = np.where(y_train == 1, class_weight[1], class_weight[0])
    gb.fit(X_train, y_train, sample_weight=sample_weights)

    gb_val_probs = gb.predict_proba(X_val)[:, 1]
    gb_test_probs = gb.predict_proba(X_test)[:, 1]

    gb_val_metrics = _compute_metrics(y_val, gb_val_probs)
    gb_test_metrics = _compute_metrics(y_test, gb_test_probs)

    # Feature importance
    gb_importance = pd.DataFrame({
        "feature": feature_names,
        "importance": gb.feature_importances_,
    }).sort_values("importance", ascending=False)

    results["gradient_boosting"] = {
        "val": gb_val_metrics,
        "test": gb_test_metrics,
        "feature_importance": gb_importance.to_dict(orient="records"),
    }

    logger.info("  GB Val:  AUROC=%.4f  AUPRC=%.4f  F1=%.4f",
                gb_val_metrics["auroc"], gb_val_metrics["auprc"], gb_val_metrics["f1"])
    logger.info("  GB Test: AUROC=%.4f  AUPRC=%.4f  F1=%.4f",
                gb_test_metrics["auroc"], gb_test_metrics["auprc"], gb_test_metrics["f1"])

    # Try XGBoost if available
    try:
        import xgboost as xgb

        logger.info("Training XGBoost...")
        scale_pos_weight = n_neg / n_pos
        xgb_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
        )
        xgb_model.fit(X_train, y_train)

        xgb_val_probs = xgb_model.predict_proba(X_val)[:, 1]
        xgb_test_probs = xgb_model.predict_proba(X_test)[:, 1]

        xgb_val_metrics = _compute_metrics(y_val, xgb_val_probs)
        xgb_test_metrics = _compute_metrics(y_test, xgb_test_probs)

        xgb_importance = pd.DataFrame({
            "feature": feature_names,
            "importance": xgb_model.feature_importances_,
        }).sort_values("importance", ascending=False)

        results["xgboost"] = {
            "val": xgb_val_metrics,
            "test": xgb_test_metrics,
            "feature_importance": xgb_importance.to_dict(orient="records"),
        }

        logger.info("  XGB Val:  AUROC=%.4f  AUPRC=%.4f  F1=%.4f",
                    xgb_val_metrics["auroc"], xgb_val_metrics["auprc"], xgb_val_metrics["f1"])
        logger.info("  XGB Test: AUROC=%.4f  AUPRC=%.4f  F1=%.4f",
                    xgb_test_metrics["auroc"], xgb_test_metrics["auprc"], xgb_test_metrics["f1"])
    except ImportError:
        logger.info("XGBoost not available, skipping")

    return results


def _compute_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    """Compute classification metrics."""
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        precision_recall_curve,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    metrics = {}

    if len(np.unique(y_true)) < 2:
        return {k: float("nan") for k in [
            "auroc", "auprc", "f1", "precision", "recall", "accuracy",
            "optimal_threshold", "n_positive", "n_negative",
        ]}

    metrics["auroc"] = roc_auc_score(y_true, y_score)
    metrics["auprc"] = average_precision_score(y_true, y_score)

    # Optimal threshold
    precision_arr, recall_arr, thresholds = precision_recall_curve(y_true, y_score)
    f1_arr = 2 * precision_arr[:-1] * recall_arr[:-1] / (
        precision_arr[:-1] + recall_arr[:-1] + 1e-8
    )
    best_idx = np.argmax(f1_arr)
    threshold = thresholds[best_idx] if len(thresholds) > 0 else 0.5

    y_pred = (y_score >= threshold).astype(int)
    metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)
    metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["optimal_threshold"] = float(threshold)
    metrics["n_positive"] = int(y_true.sum())
    metrics["n_negative"] = int(len(y_true) - y_true.sum())

    return metrics


def print_results_table(results: Dict, feature_set: str):
    """Print results in a formatted table."""
    print(f"\n{'='*80}")
    print(f"TABULAR BASELINE RESULTS (feature_set={feature_set})")
    print(f"{'='*80}")
    print(f"{'Model':<25} {'Split':<8} {'AUROC':>8} {'AUPRC':>8} {'F1':>8} {'Prec':>8} {'Recall':>8} {'Acc':>8}")
    print("-" * 80)

    for model_name, model_results in results.items():
        for split in ["val", "test"]:
            m = model_results[split]
            print(f"{model_name:<25} {split:<8} {m['auroc']:>8.4f} {m['auprc']:>8.4f} {m['f1']:>8.4f} "
                  f"{m['precision']:>8.4f} {m['recall']:>8.4f} {m['accuracy']:>8.4f}")
    print("=" * 80)

    # Feature importance (from best model)
    best_model = max(results.keys(), key=lambda k: results[k]["val"]["auroc"])
    print(f"\nTop 15 features ({best_model}):")
    print("-" * 50)
    importance = results[best_model]["feature_importance"]
    for i, feat in enumerate(importance[:15]):
        print(f"  {i+1:2d}. {feat['feature']:<35s} {feat['importance']:.4f}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Tabular baseline for APOBEC editing prediction")
    parser.add_argument("--output_dir", type=str,
                        default=str(PROJECT_ROOT / "experiments" / "apobec" / "outputs" / "tabular_baseline"))
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading data...")
    labels_df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "editing_sites_labels.csv")
    negatives_df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "advisor" / "negative_controls_ct.csv")
    splits_df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "splits.csv")

    logger.info("  Positives: %d", len(labels_df))
    logger.info("  Negatives: %d", len(negatives_df))

    # Build feature matrix
    logger.info("Building feature matrix...")
    features_df, labels, splits, site_ids = build_feature_matrix(labels_df, negatives_df, splits_df)
    logger.info("  Feature matrix: %s", features_df.shape)
    logger.info("  Features: %s", list(features_df.columns))

    # Save feature matrix
    features_df["label"] = labels
    features_df["split"] = splits
    features_df["site_id"] = site_ids
    features_df.to_csv(output_dir / "feature_matrix.csv", index=False)

    # --- Experiment 1: Common features only (fair comparison) ---
    logger.info("\n=== Experiment: Common features only ===")
    results_common = train_and_evaluate(
        features_df.drop(columns=["label", "split", "site_id"]),
        labels, splits, output_dir,
        feature_set="common_only",
    )
    print_results_table(results_common, "common_only")

    # --- Experiment 2: All features (positives have extra features, NaN-imputed for negatives) ---
    logger.info("\n=== Experiment: All features ===")
    results_all = train_and_evaluate(
        features_df.drop(columns=["label", "split", "site_id"]),
        labels, splits, output_dir,
        feature_set="all",
    )
    print_results_table(results_all, "all")

    # Save all results
    all_results = {
        "common_only": {
            k: {split: {mk: float(mv) for mk, mv in metrics.items()}
                for split, metrics in v.items() if split in ("val", "test")}
            for k, v in results_common.items()
        },
        "all_features": {
            k: {split: {mk: float(mv) for mk, mv in metrics.items()}
                for split, metrics in v.items() if split in ("val", "test")}
            for k, v in results_all.items()
        },
    }

    with open(output_dir / "baseline_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=lambda x: float(x))

    # Save feature importance as CSV
    for feature_set_name, results in [("common_only", results_common), ("all", results_all)]:
        for model_name, model_results in results.items():
            imp_df = pd.DataFrame(model_results["feature_importance"])
            imp_df.to_csv(
                output_dir / f"feature_importance_{feature_set_name}_{model_name}.csv",
                index=False,
            )

    logger.info("All results saved to %s", output_dir)


if __name__ == "__main__":
    main()
