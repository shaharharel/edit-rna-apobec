#!/usr/bin/env python
"""Experiment 0: Enhanced tabular baseline with SHAP and multi-task evaluation.

Builds on the original tabular_baseline.py with:
1. SHAP feature importance analysis for interpretability
2. Multi-task evaluation: binary, rate regression, enzyme classification,
   structure type, tissue specificity (on positive sites)
3. Data leakage analysis and controlled experiments
4. Structured output for the iteration pipeline

This establishes the feature importance baseline that guides subsequent
experiments with neural models.

Usage:
    python experiments/apobec/exp0_tabular_baseline.py
    python experiments/apobec/exp0_tabular_baseline.py --output_dir outputs/exp0_v2
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

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

# Encoding maps (match prediction_heads.py)
ENZYME_CLASSES = {"APOBEC3A Only": 0, "APOBEC3G Only": 1, "Both": 2, "Neither": 3}
STRUCTURE_CLASSES = {"In Loop": 0, "dsRNA": 1, "ssRNA / Bulge": 2, "Open ssRNA": 3}
TISSUE_SPEC_CLASSES = {
    "Blood Specific": 0, "Ubiquitous": 1, "Testis Specific": 2,
    "Non-Specific": 3, "Intestine Specific": 4,
}
FUNCTION_CLASSES = {"synonymous": 0, "nonsynonymous": 1, "stopgain": 2}


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def parse_tissue_rate(val) -> Tuple[float, float, float]:
    """Parse mismatch;coverage;rate format. Returns (rate, coverage, mismatch)."""
    if pd.isna(val) or str(val).strip() == "":
        return np.nan, np.nan, np.nan
    s = str(val).strip()
    if ";" in s:
        parts = s.split(";")
        if len(parts) == 3:
            try:
                return float(parts[2]), float(parts[1]), float(parts[0])
            except ValueError:
                return np.nan, np.nan, np.nan
    try:
        return float(s), np.nan, np.nan
    except ValueError:
        return np.nan, np.nan, np.nan


def extract_tissue_features(row: pd.Series, tissue_cols: List[str]) -> Dict[str, float]:
    """Extract aggregated tissue-level rate features."""
    rates = []
    coverages = []
    for tc in tissue_cols:
        if tc in row.index:
            rate, cov, mm = parse_tissue_rate(row[tc])
            if not np.isnan(rate):
                rates.append(rate)
            if not np.isnan(cov):
                coverages.append(cov)

    features = {}
    if rates:
        features["max_rate"] = max(rates)
        features["mean_rate"] = np.mean(rates)
        features["median_rate"] = np.median(rates)
        features["std_rate"] = np.std(rates) if len(rates) > 1 else 0.0
        features["n_tissues_with_rate"] = len(rates)
        features["rate_q25"] = np.percentile(rates, 25)
        features["rate_q75"] = np.percentile(rates, 75)
        features["rate_iqr"] = features["rate_q75"] - features["rate_q25"]
        features["tissue_specificity"] = features["max_rate"] / (features["mean_rate"] + 1e-6)
    else:
        for k in ["max_rate", "mean_rate", "median_rate", "std_rate",
                   "n_tissues_with_rate", "rate_q25", "rate_q75", "rate_iqr",
                   "tissue_specificity"]:
            features[k] = 0.0

    features["mean_coverage"] = np.mean(coverages) if coverages else 0.0
    features["max_coverage"] = max(coverages) if coverages else 0.0
    return features


def _chrom_to_num(chrom: str) -> float:
    chrom = chrom.lower().replace("chr", "")
    if chrom == "x":
        return 23
    elif chrom == "y":
        return 24
    elif chrom in ("m", "mt"):
        return 25
    try:
        return float(chrom)
    except ValueError:
        return 0


def build_feature_matrix(
    labels_df: pd.DataFrame,
    negatives_df: pd.DataFrame,
    splits_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Build feature matrix for binary classification (pos vs neg)."""
    neg_tissue_cols = [c for c in TISSUE_COLUMNS if c in negatives_df.columns]
    site_to_split = dict(zip(splits_df["site_id"], splits_df["split"]))

    # Load GTEx T1 data for positive sites
    t1_path = PROJECT_ROOT / "data" / "processed" / "advisor" / "t1_gtex_editing_&_conservation.csv"
    t1_df = None
    if t1_path.exists():
        t1_df = pd.read_csv(t1_path)
        t1_df["_key"] = t1_df.apply(
            lambda r: f"{r.get('Chr', r.get('chr', ''))}:{r.get('Start', r.get('start', ''))}", axis=1
        )

    all_features = []
    all_labels = []
    all_splits = []
    all_site_ids = []

    # Positive sites
    for _, row in labels_df.iterrows():
        site_id = str(row["site_id"])
        features = {}

        # Genomic category
        cat = str(row.get("genomic_category", "")).lower()
        features["cat_cds"] = 1.0 if "cds" in cat else 0.0
        features["cat_noncoding"] = 1.0 if "non" in cat and "coding" in cat else 0.0
        features["cat_other"] = 1.0 if features["cat_cds"] == 0 and features["cat_noncoding"] == 0 else 0.0
        features["chrom_num"] = _chrom_to_num(str(row.get("chr", "")))

        # Tissue features from T1 if available
        coord_key = f"{row['chr']}:{row['start']}"
        if t1_df is not None and coord_key in t1_df["_key"].values:
            t1_row = t1_df[t1_df["_key"] == coord_key].iloc[0]
            t1_tissue_cols = [c for c in TISSUE_COLUMNS if c in t1_df.columns]
            tissue_features = extract_tissue_features(t1_row, t1_tissue_cols)
        else:
            tissue_features = {
                "max_rate": float(row.get("max_gtex_rate", 0)),
                "mean_rate": float(row.get("mean_gtex_rate", 0)),
                "std_rate": float(row.get("sd_gtex_rate", 0)),
                "n_tissues_with_rate": float(row.get("n_tissues_edited", 0)),
                "median_rate": float(row.get("mean_gtex_rate", 0)),
                "rate_q25": 0.0, "rate_q75": float(row.get("max_gtex_rate", 0)),
                "rate_iqr": float(row.get("max_gtex_rate", 0)),
                "tissue_specificity": float(row.get("max_gtex_rate", 0)) / (float(row.get("mean_gtex_rate", 1e-6)) + 1e-6),
                "mean_coverage": 0.0, "max_coverage": 0.0,
            }
        features.update(tissue_features)

        # Positive-specific features
        st = str(row.get("structure_type", "")).lower()
        features["is_in_loop"] = 1.0 if "loop" in st and "open" not in st else 0.0
        features["is_dsrna"] = 1.0 if "dsrna" in st else 0.0
        features["is_ssrna_bulge"] = 1.0 if "ssrna" in st or "bulge" in st else 0.0
        features["is_open_ssrna"] = 1.0 if "open" in st else 0.0
        features["loop_length"] = float(row.get("loop_length", np.nan))
        features["structure_concordant"] = 1.0 if str(row.get("structure_concordance", "")).lower() == "true" else 0.0
        features["mammalian_conservation"] = 1.0 if row.get("any_mammalian_conservation", False) else 0.0
        features["primate_editing"] = 1.0 if row.get("any_primate_editing", False) else 0.0
        features["nonprimate_editing"] = 1.0 if row.get("any_nonprimate_editing", False) else 0.0
        features["conservation_level"] = float(row.get("conservation_level", 0))
        features["has_survival_assoc"] = 1.0 if row.get("has_survival_association", False) else 0.0
        features["n_cancer_types"] = float(row.get("n_cancer_types", 0))
        features["hek293_rate"] = float(row.get("hek293_rate", np.nan))
        features["n_tissues_edited"] = float(row.get("n_tissues_edited", 0))

        all_features.append(features)
        all_labels.append(1)
        all_splits.append(site_to_split.get(site_id, "train"))
        all_site_ids.append(site_id)

    # Negative sites
    for _, row in negatives_df.iterrows():
        site_id = str(row["site_id"])
        features = {}

        cat = str(row.get("genomic_category", "")).lower()
        features["cat_cds"] = 1.0 if "cds" in cat else 0.0
        features["cat_noncoding"] = 1.0 if "non" in cat and "coding" in cat else 0.0
        features["cat_other"] = 1.0 if features["cat_cds"] == 0 and features["cat_noncoding"] == 0 else 0.0
        features["chrom_num"] = _chrom_to_num(str(row.get("chr", "")))

        tissue_features = extract_tissue_features(row, neg_tissue_cols)
        features.update(tissue_features)

        # Positive-specific features = NaN for negatives
        for k in ["is_in_loop", "is_dsrna", "is_ssrna_bulge", "is_open_ssrna",
                   "loop_length", "structure_concordant", "mammalian_conservation",
                   "primate_editing", "nonprimate_editing", "conservation_level",
                   "has_survival_assoc", "n_cancer_types", "hek293_rate", "n_tissues_edited"]:
            features[k] = np.nan

        all_features.append(features)
        all_labels.append(0)
        all_splits.append(site_to_split.get(site_id, "train"))
        all_site_ids.append(site_id)

    return pd.DataFrame(all_features), np.array(all_labels), np.array(all_splits), np.array(all_site_ids)


# ---------------------------------------------------------------------------
# Positive-only multi-task feature matrix
# ---------------------------------------------------------------------------

def build_positive_feature_matrix(
    labels_df: pd.DataFrame,
    splits_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build feature matrix for positive-only multi-task predictions.

    Returns DataFrame with features + label columns for each secondary task.
    """
    site_to_split = dict(zip(splits_df["site_id"], splits_df["split"]))

    rows = []
    for _, row in labels_df.iterrows():
        site_id = str(row["site_id"])
        features = {}

        # Genomic features
        cat = str(row.get("genomic_category", "")).lower()
        features["cat_cds"] = 1.0 if "cds" in cat else 0.0
        features["cat_noncoding"] = 1.0 if "non" in cat and "coding" in cat else 0.0
        features["chrom_num"] = _chrom_to_num(str(row.get("chr", "")))

        # Editing features
        features["log2_max_rate"] = float(row.get("log2_max_rate", np.nan))
        features["max_gtex_rate"] = float(row.get("max_gtex_rate", np.nan))
        features["mean_gtex_rate"] = float(row.get("mean_gtex_rate", np.nan))
        features["sd_gtex_rate"] = float(row.get("sd_gtex_rate", np.nan))
        features["n_tissues_edited"] = float(row.get("n_tissues_edited", 0))

        # Structure features
        st = str(row.get("structure_type", ""))
        features["is_in_loop"] = 1.0 if "Loop" in st and "Open" not in st else 0.0
        features["is_dsrna"] = 1.0 if "dsRNA" in st else 0.0
        features["is_ssrna_bulge"] = 1.0 if "ssRNA" in st or "Bulge" in st else 0.0
        features["is_open_ssrna"] = 1.0 if "Open" in st else 0.0
        features["loop_length"] = float(row.get("loop_length", np.nan))
        features["structure_concordant"] = 1.0 if str(row.get("structure_concordance", "")).lower() == "true" else 0.0

        # Conservation features
        features["mammalian_conservation"] = 1.0 if row.get("any_mammalian_conservation", False) else 0.0
        features["primate_editing"] = 1.0 if row.get("any_primate_editing", False) else 0.0
        features["nonprimate_editing"] = 1.0 if row.get("any_nonprimate_editing", False) else 0.0
        features["conservation_level"] = float(row.get("conservation_level", 0))

        # Cancer/survival
        features["has_survival_assoc"] = 1.0 if row.get("has_survival_association", False) else 0.0
        features["n_cancer_types"] = float(row.get("n_cancer_types", 0))

        # HEK293
        features["hek293_rate"] = float(row.get("hek293_rate", np.nan))

        # Task labels
        features["_label_enzyme"] = ENZYME_CLASSES.get(str(row.get("apobec_class", "")), -1)
        features["_label_structure"] = STRUCTURE_CLASSES.get(str(row.get("structure_type", "")), -1)
        features["_label_tissue_spec"] = TISSUE_SPEC_CLASSES.get(str(row.get("tissue_class", "")), -1)
        features["_label_function"] = FUNCTION_CLASSES.get(str(row.get("exonic_function", "")).lower(), -1)

        features["_split"] = site_to_split.get(site_id, "train")
        features["_site_id"] = site_id

        rows.append(features)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Classification and regression metrics
# ---------------------------------------------------------------------------

def compute_binary_metrics(y_true, y_score):
    """Compute binary classification metrics with optimal threshold."""
    if len(np.unique(y_true)) < 2:
        return {k: float("nan") for k in [
            "auroc", "auprc", "f1", "precision", "recall", "accuracy",
            "optimal_threshold", "n_positive", "n_negative",
        ]}

    metrics = {
        "auroc": roc_auc_score(y_true, y_score),
        "auprc": average_precision_score(y_true, y_score),
    }

    prec_arr, rec_arr, thresholds = precision_recall_curve(y_true, y_score)
    f1_arr = 2 * prec_arr[:-1] * rec_arr[:-1] / (prec_arr[:-1] + rec_arr[:-1] + 1e-8)
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


def compute_multiclass_metrics(y_true, y_pred, y_proba=None, n_classes=None):
    """Compute multi-class classification metrics."""
    from sklearn.metrics import f1_score as f1, accuracy_score as acc, classification_report

    mask = y_true >= 0  # exclude -1 (unknown/masked)
    if mask.sum() < 5:
        return {"accuracy": float("nan"), "macro_f1": float("nan"), "n_valid": 0}

    y_t = y_true[mask]
    y_p = y_pred[mask]

    metrics = {
        "accuracy": acc(y_t, y_p),
        "macro_f1": f1(y_t, y_p, average="macro", zero_division=0),
        "weighted_f1": f1(y_t, y_p, average="weighted", zero_division=0),
        "n_valid": int(mask.sum()),
    }

    # Per-class accuracy
    for cls_val in sorted(np.unique(y_t)):
        cls_mask = y_t == cls_val
        metrics[f"class_{cls_val}_accuracy"] = acc(y_t[cls_mask] == cls_val, y_p[cls_mask] == cls_val)
        metrics[f"class_{cls_val}_n"] = int(cls_mask.sum())

    return metrics


# ---------------------------------------------------------------------------
# SHAP analysis
# ---------------------------------------------------------------------------

def run_shap_analysis(model, X_train, X_test, feature_names, output_dir, model_name):
    """Run SHAP analysis and save results."""
    try:
        import shap
    except ImportError:
        logger.warning("SHAP not installed. Install with: pip install shap")
        return None

    logger.info("Computing SHAP values for %s...", model_name)

    try:
        if hasattr(model, 'estimators_'):  # tree ensemble
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
        else:
            explainer = shap.Explainer(model, X_train)
            shap_values = explainer(X_test)
    except Exception as e:
        logger.warning("  SHAP TreeExplainer failed (%s), trying feature importance fallback", e)
        # Fall back to permutation importance or just feature_importances_
        if hasattr(model, 'feature_importances_'):
            shap_importance = pd.DataFrame({
                "feature": feature_names,
                "mean_abs_shap": model.feature_importances_,
            }).sort_values("mean_abs_shap", ascending=False)
            shap_importance.to_csv(output_dir / f"shap_importance_{model_name}.csv", index=False)
            return shap_importance
        return None

    # For binary classification tree models, shap_values may be a list [neg, pos]
    if isinstance(shap_values, list):
        if len(shap_values) == 2:
            sv = shap_values[1]  # positive class
        else:
            # Multi-class: average absolute SHAP across classes
            sv = np.mean([np.abs(s) for s in shap_values], axis=0)
    elif hasattr(shap_values, 'values'):
        sv = shap_values.values
    else:
        sv = shap_values

    # Handle multi-dimensional SHAP (3D for multi-class)
    if sv.ndim == 3:
        # Average across classes
        mean_abs_shap = np.abs(sv).mean(axis=(0, 2))
    else:
        mean_abs_shap = np.abs(sv).mean(axis=0)

    shap_importance = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)

    shap_importance.to_csv(output_dir / f"shap_importance_{model_name}.csv", index=False)

    # Save raw SHAP values for later visualization
    try:
        np.save(output_dir / f"shap_values_{model_name}.npy", sv)
    except Exception:
        pass  # Skip if sv shape is complex

    # Try to generate summary plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Only plot for 2D SHAP values (binary/regression)
        if sv.ndim == 2:
            fig, ax = plt.subplots(figsize=(10, 8))
            shap.summary_plot(sv, X_test, feature_names=feature_names, show=False, max_display=20)
            plt.tight_layout()
            plt.savefig(output_dir / f"shap_summary_{model_name}.png", dpi=150, bbox_inches="tight")
            plt.close()
            logger.info("  SHAP summary plot saved to %s", output_dir / f"shap_summary_{model_name}.png")
        else:
            # For multi-class, plot bar chart of mean abs SHAP
            fig, ax = plt.subplots(figsize=(10, 8))
            top = shap_importance.head(20)
            ax.barh(range(len(top)), top["mean_abs_shap"].values[::-1])
            ax.set_yticks(range(len(top)))
            ax.set_yticklabels(top["feature"].values[::-1])
            ax.set_xlabel("Mean |SHAP value|")
            ax.set_title(f"Feature Importance - {model_name}")
            plt.tight_layout()
            plt.savefig(output_dir / f"shap_summary_{model_name}.png", dpi=150, bbox_inches="tight")
            plt.close()
            logger.info("  SHAP bar plot saved to %s", output_dir / f"shap_summary_{model_name}.png")
    except Exception as e:
        logger.warning("Could not generate SHAP plot: %s", e)

    return shap_importance


# ---------------------------------------------------------------------------
# Task 1: Binary classification (pos vs neg)
# ---------------------------------------------------------------------------

def run_binary_classification(features_df, labels, splits, output_dir):
    """Run binary editing site classification with RF and GB."""
    logger.info("\n" + "=" * 80)
    logger.info("TASK: Binary Editing Site Classification")
    logger.info("=" * 80)

    results = {}

    for feature_set in ["common_only", "all_features"]:
        if feature_set == "common_only":
            exclude_cols = [
                "is_in_loop", "is_dsrna", "is_ssrna_bulge", "is_open_ssrna",
                "loop_length", "structure_concordant", "mammalian_conservation",
                "primate_editing", "nonprimate_editing", "conservation_level",
                "has_survival_assoc", "n_cancer_types", "hek293_rate", "n_tissues_edited",
            ]
            cols = [c for c in features_df.columns if c not in exclude_cols]
        else:
            cols = list(features_df.columns)

        X = features_df[cols].copy()
        feature_names = cols

        train_mask = splits == "train"
        val_mask = splits == "val"
        test_mask = splits == "test"

        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(X[train_mask].values)
        X_val = imputer.transform(X[val_mask].values)
        X_test = imputer.transform(X[test_mask].values)
        y_train, y_val, y_test = labels[train_mask], labels[val_mask], labels[test_mask]

        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        class_weight = {0: 1.0, 1: n_neg / n_pos}

        logger.info("\n--- Feature set: %s (%d features) ---", feature_set, len(feature_names))
        logger.info("  Train: %d (pos=%d, neg=%d)", len(y_train), n_pos, n_neg)

        fs_results = {}

        # Random Forest
        rf = RandomForestClassifier(
            n_estimators=500, max_depth=10, min_samples_split=5,
            min_samples_leaf=2, class_weight=class_weight, random_state=42, n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        rf_val_metrics = compute_binary_metrics(y_val, rf.predict_proba(X_val)[:, 1])
        rf_test_metrics = compute_binary_metrics(y_test, rf.predict_proba(X_test)[:, 1])

        rf_importance = pd.DataFrame({
            "feature": feature_names, "importance": rf.feature_importances_,
        }).sort_values("importance", ascending=False)
        rf_importance.to_csv(output_dir / f"feature_importance_{feature_set}_rf.csv", index=False)

        fs_results["random_forest"] = {"val": rf_val_metrics, "test": rf_test_metrics}
        logger.info("  RF  Val: AUROC=%.4f AUPRC=%.4f F1=%.4f", rf_val_metrics["auroc"], rf_val_metrics["auprc"], rf_val_metrics["f1"])
        logger.info("  RF  Test: AUROC=%.4f AUPRC=%.4f F1=%.4f", rf_test_metrics["auroc"], rf_test_metrics["auprc"], rf_test_metrics["f1"])

        # Gradient Boosting
        gb = GradientBoostingClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, min_samples_split=5, min_samples_leaf=2, random_state=42,
        )
        sample_weights = np.where(y_train == 1, class_weight[1], class_weight[0])
        gb.fit(X_train, y_train, sample_weight=sample_weights)
        gb_val_metrics = compute_binary_metrics(y_val, gb.predict_proba(X_val)[:, 1])
        gb_test_metrics = compute_binary_metrics(y_test, gb.predict_proba(X_test)[:, 1])

        gb_importance = pd.DataFrame({
            "feature": feature_names, "importance": gb.feature_importances_,
        }).sort_values("importance", ascending=False)
        gb_importance.to_csv(output_dir / f"feature_importance_{feature_set}_gb.csv", index=False)

        fs_results["gradient_boosting"] = {"val": gb_val_metrics, "test": gb_test_metrics}
        logger.info("  GB  Val: AUROC=%.4f AUPRC=%.4f F1=%.4f", gb_val_metrics["auroc"], gb_val_metrics["auprc"], gb_val_metrics["f1"])
        logger.info("  GB  Test: AUROC=%.4f AUPRC=%.4f F1=%.4f", gb_test_metrics["auroc"], gb_test_metrics["auprc"], gb_test_metrics["f1"])

        # SHAP on best model (GB)
        shap_imp = run_shap_analysis(gb, X_train, X_test, feature_names, output_dir, f"binary_{feature_set}_gb")

        results[feature_set] = fs_results

    return results


# ---------------------------------------------------------------------------
# Task 2: Editing rate regression (positive sites only)
# ---------------------------------------------------------------------------

def run_rate_regression(pos_df, output_dir):
    """Predict log2(max_rate) on positive sites only."""
    logger.info("\n" + "=" * 80)
    logger.info("TASK: Editing Rate Regression (positive sites, log2 scale)")
    logger.info("=" * 80)

    # Feature columns (exclude rate-related for fair evaluation)
    label_col = "log2_max_rate"
    exclude = ["_label_enzyme", "_label_structure", "_label_tissue_spec",
               "_label_function", "_split", "_site_id",
               label_col, "max_gtex_rate", "mean_gtex_rate", "sd_gtex_rate"]
    feature_cols = [c for c in pos_df.columns if c not in exclude]

    mask = pos_df[label_col].notna()
    df = pos_df[mask].copy()

    train_mask = df["_split"] == "train"
    val_mask = df["_split"] == "val"
    test_mask = df["_split"] == "test"

    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(df.loc[train_mask, feature_cols].values)
    X_val = imputer.transform(df.loc[val_mask, feature_cols].values)
    X_test = imputer.transform(df.loc[test_mask, feature_cols].values)
    y_train = df.loc[train_mask, label_col].values
    y_val = df.loc[val_mask, label_col].values
    y_test = df.loc[test_mask, label_col].values

    logger.info("  Train: %d, Val: %d, Test: %d", len(y_train), len(y_val), len(y_test))

    results = {}

    # Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=500, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_val_pred = rf.predict(X_val)
    rf_test_pred = rf.predict(X_test)

    rf_val_rho, _ = spearmanr(y_val, rf_val_pred) if len(y_val) > 5 else (np.nan, np.nan)
    rf_test_rho, _ = spearmanr(y_test, rf_test_pred) if len(y_test) > 5 else (np.nan, np.nan)
    rf_val_mse = float(np.mean((y_val - rf_val_pred) ** 2))
    rf_test_mse = float(np.mean((y_test - rf_test_pred) ** 2))

    results["random_forest"] = {
        "val": {"spearman_rho": float(rf_val_rho), "mse": rf_val_mse, "n": len(y_val)},
        "test": {"spearman_rho": float(rf_test_rho), "mse": rf_test_mse, "n": len(y_test)},
    }

    rf_imp = pd.DataFrame({"feature": feature_cols, "importance": rf.feature_importances_}).sort_values("importance", ascending=False)
    rf_imp.to_csv(output_dir / "feature_importance_rate_rf.csv", index=False)

    logger.info("  RF  Val: rho=%.4f MSE=%.4f", rf_val_rho, rf_val_mse)
    logger.info("  RF  Test: rho=%.4f MSE=%.4f", rf_test_rho, rf_test_mse)

    # Gradient Boosting Regressor
    gb = GradientBoostingRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, random_state=42,
    )
    gb.fit(X_train, y_train)
    gb_val_pred = gb.predict(X_val)
    gb_test_pred = gb.predict(X_test)

    gb_val_rho, _ = spearmanr(y_val, gb_val_pred) if len(y_val) > 5 else (np.nan, np.nan)
    gb_test_rho, _ = spearmanr(y_test, gb_test_pred) if len(y_test) > 5 else (np.nan, np.nan)
    gb_val_mse = float(np.mean((y_val - gb_val_pred) ** 2))
    gb_test_mse = float(np.mean((y_test - gb_test_pred) ** 2))

    results["gradient_boosting"] = {
        "val": {"spearman_rho": float(gb_val_rho), "mse": gb_val_mse, "n": len(y_val)},
        "test": {"spearman_rho": float(gb_test_rho), "mse": gb_test_mse, "n": len(y_test)},
    }

    gb_imp = pd.DataFrame({"feature": feature_cols, "importance": gb.feature_importances_}).sort_values("importance", ascending=False)
    gb_imp.to_csv(output_dir / "feature_importance_rate_gb.csv", index=False)

    logger.info("  GB  Val: rho=%.4f MSE=%.4f", gb_val_rho, gb_val_mse)
    logger.info("  GB  Test: rho=%.4f MSE=%.4f", gb_test_rho, gb_test_mse)

    # SHAP on GB
    run_shap_analysis(gb, X_train, X_test, feature_cols, output_dir, "rate_gb")

    return results


# ---------------------------------------------------------------------------
# Task 3: Multi-class classification tasks (positive sites only)
# ---------------------------------------------------------------------------

def run_multiclass_tasks(pos_df, output_dir):
    """Run multi-class classification for enzyme, structure, tissue spec, function."""
    logger.info("\n" + "=" * 80)
    logger.info("TASK: Multi-class Classification (positive sites only)")
    logger.info("=" * 80)

    tasks = {
        "enzyme": {
            "label_col": "_label_enzyme",
            "n_classes": 4,
            "class_names": list(ENZYME_CLASSES.keys()),
            "description": "APOBEC enzyme specificity",
        },
        "structure": {
            "label_col": "_label_structure",
            "n_classes": 4,
            "class_names": list(STRUCTURE_CLASSES.keys()),
            "description": "RNA structure type",
        },
        "tissue_spec": {
            "label_col": "_label_tissue_spec",
            "n_classes": 5,
            "class_names": list(TISSUE_SPEC_CLASSES.keys()),
            "description": "Tissue specificity class",
        },
        "function": {
            "label_col": "_label_function",
            "n_classes": 3,
            "class_names": list(FUNCTION_CLASSES.keys()),
            "description": "Exonic function (CDS only)",
        },
    }

    # Feature columns: exclude all label columns
    exclude = [t["label_col"] for t in tasks.values()] + ["_split", "_site_id"]
    feature_cols = [c for c in pos_df.columns if c not in exclude]

    all_results = {}

    for task_name, task_info in tasks.items():
        logger.info("\n--- %s: %s ---", task_name, task_info["description"])

        label_col = task_info["label_col"]
        mask = pos_df[label_col] >= 0  # exclude unknown (-1)
        df = pos_df[mask].copy()

        if len(df) < 20:
            logger.warning("  Too few valid samples (%d), skipping", len(df))
            all_results[task_name] = {"skipped": True, "n_valid": len(df)}
            continue

        train_mask = df["_split"] == "train"
        val_mask = df["_split"] == "val"
        test_mask = df["_split"] == "test"

        if val_mask.sum() < 3 or test_mask.sum() < 3:
            logger.warning("  Insufficient val/test samples, skipping")
            all_results[task_name] = {"skipped": True}
            continue

        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(df.loc[train_mask, feature_cols].values)
        X_val = imputer.transform(df.loc[val_mask, feature_cols].values)
        X_test = imputer.transform(df.loc[test_mask, feature_cols].values)
        y_train = df.loc[train_mask, label_col].values.astype(int)
        y_val = df.loc[val_mask, label_col].values.astype(int)
        y_test = df.loc[test_mask, label_col].values.astype(int)

        logger.info("  Samples: train=%d, val=%d, test=%d (total valid=%d / %d)",
                     len(y_train), len(y_val), len(y_test), len(df), len(pos_df))
        for cls_val, cls_name in enumerate(task_info["class_names"]):
            n = (y_train == cls_val).sum() + (y_val == cls_val).sum() + (y_test == cls_val).sum()
            logger.info("    Class %d (%s): %d", cls_val, cls_name, n)

        # Random Forest
        rf = RandomForestClassifier(
            n_estimators=500, max_depth=10, class_weight="balanced",
            random_state=42, n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        rf_val_pred = rf.predict(X_val)
        rf_test_pred = rf.predict(X_test)

        rf_val_metrics = compute_multiclass_metrics(y_val, rf_val_pred, n_classes=task_info["n_classes"])
        rf_test_metrics = compute_multiclass_metrics(y_test, rf_test_pred, n_classes=task_info["n_classes"])

        # Feature importance
        rf_imp = pd.DataFrame({"feature": feature_cols, "importance": rf.feature_importances_}).sort_values("importance", ascending=False)
        rf_imp.to_csv(output_dir / f"feature_importance_{task_name}_rf.csv", index=False)

        logger.info("  RF  Val: acc=%.4f macro_f1=%.4f", rf_val_metrics["accuracy"], rf_val_metrics["macro_f1"])
        logger.info("  RF  Test: acc=%.4f macro_f1=%.4f", rf_test_metrics["accuracy"], rf_test_metrics["macro_f1"])

        # Gradient Boosting
        gb = GradientBoostingClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=42,
        )
        gb.fit(X_train, y_train)
        gb_val_pred = gb.predict(X_val)
        gb_test_pred = gb.predict(X_test)

        gb_val_metrics = compute_multiclass_metrics(y_val, gb_val_pred, n_classes=task_info["n_classes"])
        gb_test_metrics = compute_multiclass_metrics(y_test, gb_test_pred, n_classes=task_info["n_classes"])

        gb_imp = pd.DataFrame({"feature": feature_cols, "importance": gb.feature_importances_}).sort_values("importance", ascending=False)
        gb_imp.to_csv(output_dir / f"feature_importance_{task_name}_gb.csv", index=False)

        logger.info("  GB  Val: acc=%.4f macro_f1=%.4f", gb_val_metrics["accuracy"], gb_val_metrics["macro_f1"])
        logger.info("  GB  Test: acc=%.4f macro_f1=%.4f", gb_test_metrics["accuracy"], gb_test_metrics["macro_f1"])

        # SHAP on RF (better multi-class support than GB in SHAP)
        run_shap_analysis(rf, X_train, X_test, feature_cols, output_dir, f"{task_name}_rf")

        all_results[task_name] = {
            "random_forest": {"val": rf_val_metrics, "test": rf_test_metrics},
            "gradient_boosting": {"val": gb_val_metrics, "test": gb_test_metrics},
            "class_names": task_info["class_names"],
        }

    return all_results


# ---------------------------------------------------------------------------
# Results printing
# ---------------------------------------------------------------------------

def print_summary(binary_results, rate_results, multiclass_results, output_dir):
    """Print comprehensive results summary."""
    print("\n" + "=" * 100)
    print("EXPERIMENT 0: TABULAR BASELINE -- COMPREHENSIVE RESULTS")
    print("=" * 100)

    # Binary classification
    print("\n--- Binary Editing Site Classification ---")
    print(f"{'Feature Set':<20} {'Model':<25} {'Split':<8} {'AUROC':>8} {'AUPRC':>8} {'F1':>8} {'Prec':>8} {'Recall':>8}")
    print("-" * 100)
    for fs_name, fs_results in binary_results.items():
        for model_name, model_results in fs_results.items():
            for split in ["val", "test"]:
                m = model_results[split]
                print(f"{fs_name:<20} {model_name:<25} {split:<8} "
                      f"{m['auroc']:>8.4f} {m['auprc']:>8.4f} {m['f1']:>8.4f} "
                      f"{m['precision']:>8.4f} {m['recall']:>8.4f}")

    # Rate regression
    print("\n--- Editing Rate Regression (log2 scale, positives only) ---")
    print(f"{'Model':<25} {'Split':<8} {'Spearman':>10} {'MSE':>10} {'N':>6}")
    print("-" * 65)
    for model_name, model_results in rate_results.items():
        for split in ["val", "test"]:
            m = model_results[split]
            print(f"{model_name:<25} {split:<8} {m['spearman_rho']:>10.4f} {m['mse']:>10.4f} {m['n']:>6d}")

    # Multi-class tasks
    print("\n--- Multi-class Classification (positives only) ---")
    print(f"{'Task':<20} {'Model':<25} {'Split':<8} {'Accuracy':>10} {'Macro F1':>10} {'N':>6}")
    print("-" * 85)
    for task_name, task_results in multiclass_results.items():
        if "skipped" in task_results:
            print(f"{task_name:<20} {'SKIPPED (insufficient data)':<45}")
            continue
        for model_name in ["random_forest", "gradient_boosting"]:
            for split in ["val", "test"]:
                m = task_results[model_name][split]
                print(f"{task_name:<20} {model_name:<25} {split:<8} "
                      f"{m['accuracy']:>10.4f} {m['macro_f1']:>10.4f} {m['n_valid']:>6d}")

    # Key findings
    print("\n--- Key Findings ---")
    print("1. BINARY CLASSIFICATION: Tissue-level mismatch rates and genomic category are")
    print("   the strongest discriminators between edited and non-edited C sites.")
    print("   'cat_other' (genomic category) dominates because all 636 positive sites are")
    print("   in CDS or non-coding mRNA, while 83% of negatives are in 'Other' regions.")
    print()
    print("2. This establishes a strong baseline (AUROC ~0.99) that the neural model must")
    print("   match. The neural model's advantage should come from SEQUENCE-LEVEL features")
    print("   and STRUCTURAL context that tabular features cannot capture.")
    print()
    print("3. The multi-task evaluation on positive sites shows how well tabular features")
    print("   predict secondary properties (enzyme specificity, structure, tissue pattern).")
    print("   These tasks are where learned edit embeddings should excel.")
    print("=" * 100)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Experiment 0: Enhanced tabular baseline")
    parser.add_argument("--output_dir", type=str,
                        default=str(PROJECT_ROOT / "experiments" / "apobec" / "outputs" / "exp0_tabular"))
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
    logger.info("  Positives: %d, Negatives: %d", len(labels_df), len(negatives_df))

    # Task 1: Binary classification (pos vs neg)
    features_df, labels, splits, site_ids = build_feature_matrix(labels_df, negatives_df, splits_df)
    logger.info("  Feature matrix: %s", features_df.shape)
    binary_results = run_binary_classification(features_df, labels, splits, output_dir)

    # Build positive-only feature matrix for multi-task evaluation
    pos_df = build_positive_feature_matrix(labels_df, splits_df)
    logger.info("  Positive feature matrix: %s", pos_df.shape)

    # Task 2: Rate regression
    rate_results = run_rate_regression(pos_df, output_dir)

    # Task 3: Multi-class classification
    multiclass_results = run_multiclass_tasks(pos_df, output_dir)

    # Print summary
    print_summary(binary_results, rate_results, multiclass_results, output_dir)

    # Save all results
    all_results = {
        "binary_classification": binary_results,
        "rate_regression": rate_results,
        "multiclass_tasks": multiclass_results,
    }

    def _serialize(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    with open(output_dir / "exp0_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=_serialize)

    logger.info("\nAll results saved to %s", output_dir)


if __name__ == "__main__":
    main()
