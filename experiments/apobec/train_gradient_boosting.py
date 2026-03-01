#!/usr/bin/env python
"""Gradient Boosting baseline for binary classification of RNA editing sites.

Trains XGBoost (or sklearn GradientBoostingClassifier fallback) binary classifiers
using handcrafted features from multiple sources:
  1. Motif features: 3-mer around edit site, TC motif indicator, trinucleotide one-hot
  2. Structure delta features (7 dims): from ViennaRNA structure cache
  3. Loop position features (9 dims): from loop_position_per_site.csv
  4. Pooled RNA-FM embedding delta (640 dims): after - before

Two model variants are compared:
  - GB_HandFeatures: structure + motif + loop features only (no embeddings)
  - GB_AllFeatures: all features including embedding delta

Usage:
    python experiments/apobec/train_gradient_boosting.py
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SPLITS_CSV = PROJECT_ROOT / "data" / "processed" / "splits_expanded.csv"
SEQUENCES_JSON = PROJECT_ROOT / "data" / "processed" / "site_sequences.json"
STRUCTURE_CACHE = PROJECT_ROOT / "data" / "processed" / "embeddings" / "vienna_structure_cache.npz"
LOOP_POSITION_CSV = PROJECT_ROOT / "experiments" / "apobec" / "outputs" / "loop_position" / "loop_position_per_site.csv"
POOLED_ORIG_PT = PROJECT_ROOT / "data" / "processed" / "embeddings" / "rnafm_pooled.pt"
POOLED_EDITED_PT = PROJECT_ROOT / "data" / "processed" / "embeddings" / "rnafm_pooled_edited.pt"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "apobec" / "outputs" / "baselines" / "gradient_boosting"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_binary_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    """Compute binary classification metrics with optimal F1 threshold."""
    if len(np.unique(y_true)) < 2:
        return {k: float("nan") for k in [
            "auroc", "auprc", "f1", "precision", "recall", "accuracy",
        ]}

    metrics = {
        "auroc": float(roc_auc_score(y_true, y_score)),
        "auprc": float(average_precision_score(y_true, y_score)),
    }

    prec_arr, rec_arr, thresholds = precision_recall_curve(y_true, y_score)
    f1_arr = 2 * prec_arr[:-1] * rec_arr[:-1] / (prec_arr[:-1] + rec_arr[:-1] + 1e-8)
    best_idx = np.argmax(f1_arr)
    threshold = thresholds[best_idx] if len(thresholds) > 0 else 0.5
    y_pred = (y_score >= threshold).astype(int)

    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    return metrics


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def load_splits() -> pd.DataFrame:
    """Load splits_expanded.csv with site_id, label, split columns."""
    df = pd.read_csv(SPLITS_CSV)
    logger.info("Loaded splits: %d sites, splits=%s, labels=%s",
                len(df), df["split"].value_counts().to_dict(),
                df["label"].value_counts().to_dict())
    return df


def extract_motif_features(splits_df: pd.DataFrame) -> pd.DataFrame:
    """Extract motif features from sequence data.

    Uses the 201nt sequence with the edit at position 100 (0-indexed).
    Extracts the 3-mer at positions 99/100/101 (-1/0/+1 around edit).
    Creates binary features for TC motif and trinucleotide identity.
    """
    logger.info("Loading sequences from %s ...", SEQUENCES_JSON)
    with open(SEQUENCES_JSON) as f:
        sequences = json.load(f)
    logger.info("  Loaded %d sequences", len(sequences))

    # Top trinucleotide motifs to one-hot encode
    top_motifs = ["UCG", "UCA", "UCC", "UCU", "ACA", "GCA", "CCA",
                  "ACG", "GCG", "CCG", "ACU", "GCU", "CCU",
                  "ACA", "ACC", "GCC"]
    # Deduplicate while preserving order
    seen = set()
    unique_motifs = []
    for m in top_motifs:
        if m not in seen:
            seen.add(m)
            unique_motifs.append(m)
    top_motifs = unique_motifs

    rows = []
    for _, row in splits_df.iterrows():
        site_id = str(row["site_id"])
        features = {"site_id": site_id}

        if site_id in sequences:
            seq = sequences[site_id]
            # Extract trinucleotide: positions 99, 100, 101 (0-indexed)
            if len(seq) >= 102:
                trinuc = seq[99:102]
                # TC motif: position 99 = U and position 100 = C
                features["has_TC_motif"] = 1.0 if (seq[99] == "U" and seq[100] == "C") else 0.0
                # Upstream nucleotide identity
                features["upstream_A"] = 1.0 if seq[99] == "A" else 0.0
                features["upstream_C"] = 1.0 if seq[99] == "C" else 0.0
                features["upstream_G"] = 1.0 if seq[99] == "G" else 0.0
                features["upstream_U"] = 1.0 if seq[99] == "U" else 0.0
                # Downstream nucleotide identity
                features["downstream_A"] = 1.0 if seq[101] == "A" else 0.0
                features["downstream_C"] = 1.0 if seq[101] == "C" else 0.0
                features["downstream_G"] = 1.0 if seq[101] == "G" else 0.0
                features["downstream_U"] = 1.0 if seq[101] == "U" else 0.0
                # Trinucleotide one-hot
                for motif in top_motifs:
                    features[f"trinuc_{motif}"] = 1.0 if trinuc == motif else 0.0
            else:
                features["has_TC_motif"] = 0.0
                for k in ["upstream_A", "upstream_C", "upstream_G", "upstream_U",
                           "downstream_A", "downstream_C", "downstream_G", "downstream_U"]:
                    features[k] = 0.0
                for motif in top_motifs:
                    features[f"trinuc_{motif}"] = 0.0
        else:
            features["has_TC_motif"] = 0.0
            for k in ["upstream_A", "upstream_C", "upstream_G", "upstream_U",
                       "downstream_A", "downstream_C", "downstream_G", "downstream_U"]:
                features[k] = 0.0
            for motif in top_motifs:
                features[f"trinuc_{motif}"] = 0.0

        rows.append(features)

    motif_df = pd.DataFrame(rows)
    motif_cols = [c for c in motif_df.columns if c != "site_id"]
    logger.info("  Motif features: %d columns: %s", len(motif_cols), motif_cols)
    return motif_df


def extract_structure_delta_features(splits_df: pd.DataFrame) -> pd.DataFrame:
    """Extract 7-dim structure delta features from vienna_structure_cache.npz.

    Features: delta_pairing_at_pos, delta_accessibility_at_pos, delta_entropy_at_pos,
    delta_mfe, delta_local_pairing, delta_local_accessibility, local_pairing_std
    """
    logger.info("Loading structure deltas from %s ...", STRUCTURE_CACHE)
    data = np.load(STRUCTURE_CACHE, allow_pickle=True)
    site_ids = data["site_ids"]
    delta_features = data["delta_features"]
    logger.info("  Loaded %d structure delta features, shape=%s",
                len(site_ids), delta_features.shape)

    # Build lookup dict
    struct_dict = {str(sid): delta_features[i] for i, sid in enumerate(site_ids)}

    feature_names = [
        "delta_pairing_at_pos", "delta_accessibility_at_pos", "delta_entropy_at_pos",
        "delta_mfe", "delta_local_pairing", "delta_local_accessibility", "local_pairing_std",
    ]

    rows = []
    for _, row in splits_df.iterrows():
        site_id = str(row["site_id"])
        features = {"site_id": site_id}
        if site_id in struct_dict:
            vals = struct_dict[site_id]
            for j, name in enumerate(feature_names):
                features[name] = float(vals[j])
        else:
            for name in feature_names:
                features[name] = 0.0
        rows.append(features)

    struct_df = pd.DataFrame(rows)
    logger.info("  Structure delta features: %d columns", len(feature_names))
    return struct_df


def extract_loop_position_features(splits_df: pd.DataFrame) -> pd.DataFrame:
    """Extract loop position features from loop_position_per_site.csv.

    Features: is_unpaired, loop_size, dist_to_junction, dist_to_apex,
    relative_loop_position, left_stem_length, right_stem_length,
    max_adjacent_stem_length, local_unpaired_fraction
    """
    logger.info("Loading loop position features from %s ...", LOOP_POSITION_CSV)
    loop_df = pd.read_csv(LOOP_POSITION_CSV)
    logger.info("  Loaded %d rows", len(loop_df))

    loop_feature_cols = [
        "is_unpaired", "loop_size", "dist_to_junction", "dist_to_apex",
        "relative_loop_position", "left_stem_length", "right_stem_length",
        "max_adjacent_stem_length", "local_unpaired_fraction",
    ]

    # Merge by site_id
    loop_subset = loop_df[["site_id"] + [c for c in loop_feature_cols if c in loop_df.columns]].copy()
    # Convert boolean to float
    if "is_unpaired" in loop_subset.columns:
        loop_subset["is_unpaired"] = loop_subset["is_unpaired"].astype(float)

    merged = splits_df[["site_id"]].merge(loop_subset, on="site_id", how="left")

    # Fill missing values with 0
    for col in loop_feature_cols:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0.0)
        else:
            merged[col] = 0.0

    logger.info("  Loop position features: %d columns, %d matched",
                len(loop_feature_cols), loop_subset["site_id"].isin(splits_df["site_id"]).sum())
    return merged


def extract_embedding_delta_features(splits_df: pd.DataFrame) -> pd.DataFrame:
    """Extract RNA-FM pooled embedding delta (after - before) as 640-dim features."""
    import torch

    logger.info("Loading pooled embeddings...")
    logger.info("  Original: %s", POOLED_ORIG_PT)
    pooled_orig = torch.load(POOLED_ORIG_PT, map_location="cpu", weights_only=False)
    logger.info("  Edited: %s", POOLED_EDITED_PT)
    pooled_edited = torch.load(POOLED_EDITED_PT, map_location="cpu", weights_only=False)
    logger.info("  Loaded %d original, %d edited pooled embeddings",
                len(pooled_orig), len(pooled_edited))

    d_model = 640
    rows = []
    n_matched = 0
    for _, row in splits_df.iterrows():
        site_id = str(row["site_id"])
        features = {"site_id": site_id}

        if site_id in pooled_orig and site_id in pooled_edited:
            delta = (pooled_edited[site_id] - pooled_orig[site_id]).numpy()
            for j in range(d_model):
                features[f"emb_delta_{j}"] = float(delta[j])
            n_matched += 1
        else:
            for j in range(d_model):
                features[f"emb_delta_{j}"] = 0.0

        rows.append(features)

    emb_df = pd.DataFrame(rows)
    logger.info("  Embedding delta features: %d dims, %d/%d matched",
                d_model, n_matched, len(splits_df))
    return emb_df


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def get_classifier(n_features: int):
    """Get the best available gradient boosting classifier.

    Tries XGBoost first, falls back to sklearn GradientBoostingClassifier.
    """
    try:
        import xgboost as xgb
        logger.info("Using XGBoost %s", xgb.__version__)
        return xgb.XGBClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            min_child_weight=10,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=1,
            eval_metric="logloss",
            verbosity=0,
        ), "xgboost"
    except ImportError:
        pass

    try:
        import lightgbm as lgb
        logger.info("Using LightGBM %s", lgb.__version__)
        return lgb.LGBMClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            min_child_leaf=10,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        ), "lightgbm"
    except ImportError:
        pass

    from sklearn.ensemble import GradientBoostingClassifier
    logger.info("Using sklearn GradientBoostingClassifier (fallback)")
    return GradientBoostingClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=10,
        random_state=42,
    ), "sklearn"


def train_and_evaluate(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    feature_names: List[str],
    variant_name: str,
) -> Dict:
    """Train a gradient boosting classifier and evaluate on val/test sets."""
    logger.info("\n--- Training %s (%d features) ---", variant_name, len(feature_names))
    logger.info("  Train: %d (pos=%d, neg=%d)",
                len(y_train), int(y_train.sum()), int(len(y_train) - y_train.sum()))
    logger.info("  Val: %d (pos=%d, neg=%d)",
                len(y_val), int(y_val.sum()), int(len(y_val) - y_val.sum()))
    logger.info("  Test: %d (pos=%d, neg=%d)",
                len(y_test), int(y_test.sum()), int(len(y_test) - y_test.sum()))

    # Handle class imbalance with scale_pos_weight
    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    scale_pos_weight = n_neg / max(n_pos, 1)

    clf, backend = get_classifier(len(feature_names))

    # Set scale_pos_weight for XGBoost
    if backend == "xgboost":
        clf.set_params(scale_pos_weight=scale_pos_weight)
    elif backend == "lightgbm":
        clf.set_params(scale_pos_weight=scale_pos_weight)
    # For sklearn, use sample_weight instead

    t_start = time.time()

    if backend == "sklearn":
        sample_weights = np.where(y_train == 1, scale_pos_weight, 1.0)
        clf.fit(X_train, y_train, sample_weight=sample_weights)
    elif backend == "xgboost":
        clf.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
    else:
        clf.fit(X_train, y_train)

    elapsed = time.time() - t_start
    logger.info("  Training took %.1f seconds (%s backend)", elapsed, backend)

    # Evaluate
    val_proba = clf.predict_proba(X_val)[:, 1]
    test_proba = clf.predict_proba(X_test)[:, 1]

    val_metrics = compute_binary_metrics(y_val, val_proba)
    test_metrics = compute_binary_metrics(y_test, test_proba)

    logger.info("  Val:  AUROC=%.4f  AUPRC=%.4f  F1=%.4f  Prec=%.4f  Recall=%.4f",
                val_metrics["auroc"], val_metrics["auprc"], val_metrics["f1"],
                val_metrics["precision"], val_metrics["recall"])
    logger.info("  Test: AUROC=%.4f  AUPRC=%.4f  F1=%.4f  Prec=%.4f  Recall=%.4f",
                test_metrics["auroc"], test_metrics["auprc"], test_metrics["f1"],
                test_metrics["precision"], test_metrics["recall"])

    # Feature importances
    importances = clf.feature_importances_
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    return {
        "model": clf,
        "backend": backend,
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "train_time_seconds": elapsed,
        "importance_df": importance_df,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info("=" * 80)
    logger.info("GRADIENT BOOSTING BASELINE FOR BINARY RNA EDITING SITE CLASSIFICATION")
    logger.info("=" * 80)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load splits
    # ------------------------------------------------------------------
    splits_df = load_splits()

    # ------------------------------------------------------------------
    # 2. Extract all feature sets
    # ------------------------------------------------------------------
    logger.info("\n--- Extracting features ---")

    motif_df = extract_motif_features(splits_df)
    struct_df = extract_structure_delta_features(splits_df)
    loop_df = extract_loop_position_features(splits_df)
    emb_df = extract_embedding_delta_features(splits_df)

    # ------------------------------------------------------------------
    # 3. Merge all features
    # ------------------------------------------------------------------
    logger.info("\n--- Merging features ---")

    # Motif feature columns (exclude site_id)
    motif_cols = [c for c in motif_df.columns if c != "site_id"]
    struct_cols = [c for c in struct_df.columns if c != "site_id"]
    loop_cols = [c for c in loop_df.columns if c != "site_id"]
    emb_cols = [c for c in emb_df.columns if c != "site_id"]

    hand_feature_cols = motif_cols + struct_cols + loop_cols
    all_feature_cols = hand_feature_cols + emb_cols

    logger.info("  Motif features: %d", len(motif_cols))
    logger.info("  Structure delta features: %d", len(struct_cols))
    logger.info("  Loop position features: %d", len(loop_cols))
    logger.info("  Embedding delta features: %d", len(emb_cols))
    logger.info("  Total hand features: %d", len(hand_feature_cols))
    logger.info("  Total all features: %d", len(all_feature_cols))

    # Build full feature matrix
    # All DataFrames are aligned by index (same order as splits_df)
    X_motif = motif_df[motif_cols].values
    X_struct = struct_df[struct_cols].values
    X_loop = loop_df[loop_cols].values
    X_emb = emb_df[emb_cols].values

    X_hand = np.concatenate([X_motif, X_struct, X_loop], axis=1)
    X_all = np.concatenate([X_motif, X_struct, X_loop, X_emb], axis=1)

    y = splits_df["label"].values
    splits = splits_df["split"].values

    # Handle any NaN/inf
    X_hand = np.nan_to_num(X_hand, nan=0.0, posinf=0.0, neginf=0.0)
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)

    # ------------------------------------------------------------------
    # 4. Split data
    # ------------------------------------------------------------------
    train_mask = splits == "train"
    val_mask = splits == "val"
    test_mask = splits == "test"

    logger.info("\n--- Data splits ---")
    logger.info("  Train: %d (pos=%d, neg=%d)", train_mask.sum(),
                y[train_mask].sum(), train_mask.sum() - y[train_mask].sum())
    logger.info("  Val:   %d (pos=%d, neg=%d)", val_mask.sum(),
                y[val_mask].sum(), val_mask.sum() - y[val_mask].sum())
    logger.info("  Test:  %d (pos=%d, neg=%d)", test_mask.sum(),
                y[test_mask].sum(), test_mask.sum() - y[test_mask].sum())

    # ------------------------------------------------------------------
    # 5. Train models
    # ------------------------------------------------------------------
    results = {}

    # Variant 1: Hand features only
    r1 = train_and_evaluate(
        X_hand[train_mask], y[train_mask],
        X_hand[val_mask], y[val_mask],
        X_hand[test_mask], y[test_mask],
        hand_feature_cols,
        "GB_HandFeatures",
    )
    results["GB_HandFeatures"] = r1

    # Variant 2: All features (hand + embedding delta)
    r2 = train_and_evaluate(
        X_all[train_mask], y[train_mask],
        X_all[val_mask], y[val_mask],
        X_all[test_mask], y[test_mask],
        all_feature_cols,
        "GB_AllFeatures",
    )
    results["GB_AllFeatures"] = r2

    # ------------------------------------------------------------------
    # 6. Determine best variant
    # ------------------------------------------------------------------
    best_variant = max(results.keys(), key=lambda k: results[k]["test_metrics"]["auroc"])
    best_test_metrics = results[best_variant]["test_metrics"]

    # ------------------------------------------------------------------
    # 7. Print feature importances for hand features model
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCES (GB_HandFeatures - Top 30)")
    print("=" * 80)
    imp_df = results["GB_HandFeatures"]["importance_df"]
    print(f"{'Rank':<6} {'Feature':<40} {'Importance':>12}")
    print("-" * 60)
    for rank, (_, row) in enumerate(imp_df.head(30).iterrows(), 1):
        print(f"{rank:<6} {row['feature']:<40} {row['importance']:>12.6f}")

    # Save importance CSV
    imp_df.to_csv(OUTPUT_DIR / "feature_importance_hand.csv", index=False)
    results["GB_AllFeatures"]["importance_df"].to_csv(
        OUTPUT_DIR / "feature_importance_all.csv", index=False
    )

    # ------------------------------------------------------------------
    # 8. Print summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("GRADIENT BOOSTING BASELINE RESULTS")
    print("=" * 80)
    print(f"{'Variant':<22} {'N_feat':>7} {'Val AUROC':>10} {'Val AUPRC':>10} "
          f"{'Test AUROC':>10} {'Test AUPRC':>10} {'Test F1':>8} {'Test Prec':>10} {'Test Recall':>11}")
    print("-" * 100)
    for name, r in results.items():
        vm = r["val_metrics"]
        tm = r["test_metrics"]
        print(f"{name:<22} {r['n_features']:>7} "
              f"{vm['auroc']:>10.4f} {vm['auprc']:>10.4f} "
              f"{tm['auroc']:>10.4f} {tm['auprc']:>10.4f} "
              f"{tm['f1']:>8.4f} {tm['precision']:>10.4f} {tm['recall']:>11.4f}")
    print("=" * 80)
    print(f"Best variant: {best_variant}")

    # ------------------------------------------------------------------
    # 9. Save results JSON
    # ------------------------------------------------------------------
    output_json = {
        "model": "gradient_boosting",
        "variants": {},
        "best_variant": best_variant,
        "test_metrics": best_test_metrics,
    }

    for name, r in results.items():
        output_json["variants"][name] = {
            "n_features": r["n_features"],
            "feature_names": r["feature_names"],
            "backend": r["backend"],
            "train_time_seconds": r["train_time_seconds"],
            "val_metrics": r["val_metrics"],
            "test_metrics": r["test_metrics"],
        }

    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(output_json, f, indent=2)

    logger.info("\nResults saved to %s", results_path)

    # ------------------------------------------------------------------
    # 10. Print full results
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("FULL RESULTS JSON")
    print("=" * 80)
    print(json.dumps(output_json, indent=2))
    print("=" * 80)

    return output_json


if __name__ == "__main__":
    main()
