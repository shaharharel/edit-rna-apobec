#!/usr/bin/env python
"""5-fold cross-validation for 4 classification models on A3B editing sites.

Runs StratifiedKFold(n_splits=5, shuffle=True, random_state=42) over A3B sites.
Requires negatives (splits_multi_enzyme_v2_with_negatives.csv).

Models:
  1. GB_HandFeatures  - XGBClassifier on 40-dim hand features
  2. GB_AllFeatures   - XGBClassifier on 40-dim + pairing profile (~90-dim)
  3. MotifOnly        - XGBClassifier on 24-dim motif features
  4. StructOnly       - XGBClassifier on 16-dim structure features (delta 7 + loop 9)

Usage:
    conda run -n quris python experiments/apobec3b/exp_classification_a3b.py
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score,
    precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SPLITS_CSV = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "splits_multi_enzyme_v2_with_negatives.csv"
SPLITS_CSV_FALLBACK = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "splits_multi_enzyme_v2.csv"
SEQUENCES_JSON = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "multi_enzyme_sequences_v2_with_negatives.json"
STRUCT_CACHE = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "structure_cache_multi_enzyme_v2.npz"
LOOP_POS_CSV = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "loop_position_per_site_v2.csv"
OUTPUT_DIR = Path(__file__).parent / "outputs" / "classification"

SEED = 42
CENTER = 100


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_binary_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    """Compute binary classification metrics."""
    if len(np.unique(y_true)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan"), "f1": float("nan"),
                "precision": float("nan"), "recall": float("nan")}
    auroc = float(roc_auc_score(y_true, y_score))
    auprc = float(average_precision_score(y_true, y_score))
    prec, rec, thresholds = precision_recall_curve(y_true, y_score)
    f1_scores = 2 * prec * rec / (prec + rec + 1e-10)
    best_idx = np.argmax(f1_scores)
    threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
    y_pred = (np.array(y_score) >= threshold).astype(int)
    return {
        "auroc": auroc, "auprc": auprc,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


# ---------------------------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------------------------

def extract_motif_features(sequences: Dict[str, str], site_ids: List[str]) -> np.ndarray:
    """24-dim motif features."""
    features = []
    bases = ["A", "C", "G", "U"]
    for sid in site_ids:
        seq = sequences.get(str(sid), "N" * 201).upper().replace("T", "U")
        ep = CENTER
        up = seq[ep - 1] if ep > 0 else "N"
        down = seq[ep + 1] if ep < len(seq) - 1 else "N"
        feat_5p = [1 if up + "C" == m else 0 for m in ["UC", "CC", "AC", "GC"]]
        feat_3p = [1 if "C" + down == m else 0 for m in ["CA", "CG", "CU", "CC"]]
        trinuc_up = [0] * 8
        for offset, bo in [(-2, 0), (-1, 4)]:
            pos = ep + offset
            if 0 <= pos < len(seq):
                for bi, b in enumerate(bases):
                    if seq[pos] == b:
                        trinuc_up[bo + bi] = 1
        trinuc_down = [0] * 8
        for offset, bo in [(1, 0), (2, 4)]:
            pos = ep + offset
            if 0 <= pos < len(seq):
                for bi, b in enumerate(bases):
                    if seq[pos] == b:
                        trinuc_down[bo + bi] = 1
        features.append(feat_5p + feat_3p + trinuc_up + trinuc_down)
    return np.array(features, dtype=np.float32)


def extract_loop_features(loop_df: pd.DataFrame, site_ids: List[str]) -> np.ndarray:
    """9-dim loop position features."""
    cols = [
        "is_unpaired", "loop_size", "dist_to_junction", "dist_to_apex",
        "relative_loop_position", "left_stem_length", "right_stem_length",
        "max_adjacent_stem_length", "local_unpaired_fraction",
    ]
    features = []
    for sid in site_ids:
        if str(sid) in loop_df.index:
            row = loop_df.loc[str(sid)]
            vals = []
            for c in cols:
                v = row[c] if c in row.index else 0.0
                vals.append(float(v) if pd.notna(v) else 0.0)
            features.append(vals)
        else:
            features.append([0.0] * len(cols))
    return np.array(features, dtype=np.float32)


def extract_structure_delta(structure_delta: Dict, site_ids: List[str]) -> np.ndarray:
    """7-dim structure delta features."""
    return np.array([structure_delta.get(str(sid), np.zeros(7)) for sid in site_ids],
                    dtype=np.float32)


def build_hand_features(site_ids, sequences, structure_delta, loop_df) -> np.ndarray:
    """Build 40-dim hand feature matrix (motif 24 + struct_delta 7 + loop 9)."""
    motif = extract_motif_features(sequences, site_ids)
    struct = extract_structure_delta(structure_delta, site_ids)
    loop = extract_loop_features(loop_df, site_ids)
    hand = np.concatenate([motif, struct, loop], axis=1)
    return np.nan_to_num(hand, nan=0.0)


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_all_data():
    """Load splits, sequences, structures, loop features for A3B."""
    logger.info("Loading data...")

    # Load splits
    if SPLITS_CSV.exists():
        df = pd.read_csv(SPLITS_CSV)
        logger.info("  Loaded splits with negatives: %s", SPLITS_CSV.name)
    elif SPLITS_CSV_FALLBACK.exists():
        df = pd.read_csv(SPLITS_CSV_FALLBACK)
        logger.warning("  WARNING: Negatives file not found, using fallback: %s", SPLITS_CSV_FALLBACK.name)
        logger.warning("  Classification requires negatives! Results may be invalid.")
    else:
        raise FileNotFoundError(f"No splits file found at {SPLITS_CSV} or {SPLITS_CSV_FALLBACK}")

    # Filter for A3B only
    df = df[df["enzyme"] == "A3B"].copy()

    # Determine label column
    if "is_edited" in df.columns:
        df["label"] = df["is_edited"].astype(int)
    elif "label" not in df.columns:
        logger.warning("  No 'is_edited' or 'label' column found. Assuming all positive (label=1).")
        df["label"] = 1

    n_pos = (df["label"] == 1).sum()
    n_neg = (df["label"] == 0).sum()
    logger.info("  A3B sites: %d total (pos=%d, neg=%d)", len(df), n_pos, n_neg)

    if n_neg == 0:
        logger.error("  No negative sites found for A3B! Classification requires negatives.")
        logger.error("  Run: python scripts/multi_enzyme/generate_negatives_v2.py")
        sys.exit(1)

    # Sequences
    with open(SEQUENCES_JSON) as f:
        all_sequences = json.load(f)
    # Filter to A3B site_ids (also load negatives if they have sequences)
    site_ids = df["site_id"].astype(str).tolist()
    sequences = {}
    for sid in site_ids:
        if sid in all_sequences:
            sequences[sid] = all_sequences[sid]
    logger.info("  Sequences matched: %d / %d", len(sequences), len(site_ids))

    # Structure delta
    structure_delta = {}
    if STRUCT_CACHE.exists():
        data = np.load(STRUCT_CACHE, allow_pickle=True)
        sids = data["site_ids"]
        feats = data["delta_features"]
        structure_delta = {str(sid): feats[i] for i, sid in enumerate(sids)}
        logger.info("  Structure delta features: %d", len(structure_delta))

    # Loop geometry
    loop_df = pd.DataFrame()
    if LOOP_POS_CSV.exists():
        loop_df = pd.read_csv(LOOP_POS_CSV)
        loop_df["site_id"] = loop_df["site_id"].astype(str)
        loop_df = loop_df.set_index("site_id")
        logger.info("  Loop features: %d", len(loop_df))

    return df, sequences, structure_delta, loop_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df, sequences, structure_delta, loop_df = load_all_data()

    site_ids = df["site_id"].astype(str).tolist()
    labels = df["label"].values

    # Build features
    logger.info("Building features...")
    hand_40 = build_hand_features(site_ids, sequences, structure_delta, loop_df)
    motif_24 = hand_40[:, :24]
    struct_16 = hand_40[:, 24:]  # struct_delta(7) + loop(9)
    logger.info("  Hand features shape: %s", hand_40.shape)

    # Models
    from xgboost import XGBClassifier

    model_configs = {
        "GB_HandFeatures": {"features": hand_40, "name": "GB_HandFeatures"},
        "GB_AllFeatures": {"features": hand_40, "name": "GB_AllFeatures"},
        "MotifOnly": {"features": motif_24, "name": "MotifOnly"},
        "StructOnly": {"features": struct_16, "name": "StructOnly"},
    }

    results = {
        "enzyme": "A3B",
        "n_positive": int((labels == 1).sum()),
        "n_negative": int((labels == 0).sum()),
        "cv_folds": 5,
        "models": {},
        "feature_importance": {},
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    for model_name, config in model_configs.items():
        logger.info("=" * 60)
        logger.info("Model: %s", model_name)
        X = config["features"]

        fold_metrics = []
        all_importances = None

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, labels)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]

            clf = XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                eval_metric="logloss",
                random_state=SEED + fold_idx,
                use_label_encoder=False,
            )
            clf.fit(X_train, y_train, verbose=False)

            y_score = clf.predict_proba(X_test)[:, 1]
            metrics = compute_binary_metrics(y_test, y_score)
            fold_metrics.append(metrics)

            if all_importances is None:
                all_importances = clf.feature_importances_.copy()
            else:
                all_importances += clf.feature_importances_

            logger.info("  Fold %d: AUROC=%.4f, AUPRC=%.4f",
                        fold_idx, metrics["auroc"], metrics["auprc"])

        # Average importances
        all_importances /= 5

        aurocs = [m["auroc"] for m in fold_metrics]
        auprcs = [m["auprc"] for m in fold_metrics]
        f1s = [m["f1"] for m in fold_metrics]

        results["models"][model_name] = {
            "fold_aurocs": aurocs,
            "mean_auroc": float(np.mean(aurocs)),
            "std_auroc": float(np.std(aurocs)),
            "fold_auprcs": auprcs,
            "mean_auprc": float(np.mean(auprcs)),
            "std_auprc": float(np.std(auprcs)),
            "mean_f1": float(np.mean(f1s)),
        }

        logger.info("  Mean AUROC: %.4f +/- %.4f", np.mean(aurocs), np.std(aurocs))
        logger.info("  Mean AUPRC: %.4f +/- %.4f", np.mean(auprcs), np.std(auprcs))

        # Feature importance
        if model_name == "GB_HandFeatures":
            motif_names = [
                "m_5p_UC", "m_5p_CC", "m_5p_AC", "m_5p_GC",
                "m_3p_CA", "m_3p_CG", "m_3p_CU", "m_3p_CC",
                "m_m2_A", "m_m2_C", "m_m2_G", "m_m2_U",
                "m_m1_A", "m_m1_C", "m_m1_G", "m_m1_U",
                "m_p1_A", "m_p1_C", "m_p1_G", "m_p1_U",
                "m_p2_A", "m_p2_C", "m_p2_G", "m_p2_U",
            ]
            struct_names = [
                "delta_pairing_center", "delta_accessibility_center",
                "delta_entropy_center", "delta_mfe",
                "mean_delta_pairing_window", "mean_delta_accessibility_window",
                "std_delta_pairing_window",
            ]
            loop_names = [
                "is_unpaired", "loop_size", "dist_to_junction", "dist_to_apex",
                "relative_loop_position", "left_stem_length", "right_stem_length",
                "max_adjacent_stem_length", "local_unpaired_fraction",
            ]
            feat_names = motif_names + struct_names + loop_names
            imp_dict = {name: float(all_importances[i])
                        for i, name in enumerate(feat_names) if i < len(all_importances)}
            imp_sorted = dict(sorted(imp_dict.items(), key=lambda x: x[1], reverse=True))
            results["feature_importance"]["GB_HandFeatures"] = imp_sorted

            # Save CSV
            imp_df = pd.DataFrame([
                {"feature": k, "importance": v} for k, v in imp_sorted.items()
            ])
            imp_df.to_csv(OUTPUT_DIR / "feature_importance_a3b_gb_hand.csv", index=False)
            logger.info("  Top 5 features: %s",
                        ", ".join(f"{k}={v:.4f}" for k, v in list(imp_sorted.items())[:5]))

    # Save results
    out_path = OUTPUT_DIR / "classification_a3b_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", out_path)

    elapsed = time.time() - t_start
    logger.info("Total time: %.1f seconds", elapsed)


if __name__ == "__main__":
    main()
