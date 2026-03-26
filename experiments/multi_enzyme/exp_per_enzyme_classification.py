#!/usr/bin/env python
"""Per-enzyme GB_HandFeatures classification (5-fold CV).

For each enzyme with sufficient data (n_pos >= 100 and negatives available),
trains a gradient boosting classifier using the same 40-dim hand features as
the A3A classifier:
  - Motif (24-dim): trinucleotide context one-hot
  - Structure delta (7-dim): ViennaRNA delta features
  - Loop geometry (9-dim): loop position, size, stem lengths

Metrics: AUROC, AUPRC, F1, Precision, Recall (5-fold KFold CV)

Input:  data/processed/multi_enzyme/splits_multi_enzyme_v2.csv
        data/processed/multi_enzyme/multi_enzyme_sequences_v2.json
        data/processed/multi_enzyme/loop_position_per_site_v2.csv
        data/processed/multi_enzyme/structure_cache_multi_enzyme_v2.npz
Output: experiments/multi_enzyme/outputs/classification/

Usage:
    conda run -n quris python experiments/multi_enzyme/exp_per_enzyme_classification.py
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, precision_recall_curve,
)
from sklearn.model_selection import KFold
from xgboost import XGBClassifier

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
SPLITS_CSV = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "splits_multi_enzyme_v2.csv"
SEQ_JSON = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "multi_enzyme_sequences_v2.json"
LOOP_POS_CSV = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "loop_position_per_site_v2.csv"
STRUCT_CACHE = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "structure_cache_multi_enzyme_v2.npz"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "multi_enzyme" / "outputs" / "classification"

SEED = 42
MIN_POSITIVE_SITES = 100  # Minimum positives to run classification
MIN_NEGATIVE_SITES = 50   # Minimum negatives


# ---------------------------------------------------------------------------
# Feature extraction (matches A3A classifier exactly)
# ---------------------------------------------------------------------------

def extract_motif_features(sequences: Dict[str, str], site_ids: List[str]) -> np.ndarray:
    """24-dim motif features (same as A3A classifier)."""
    features = []
    bases = ["A", "C", "G", "U"]
    for sid in site_ids:
        seq = sequences.get(str(sid), "N" * 201).upper().replace("T", "U")
        ep = 100
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
            features.append(loop_df.loc[str(sid), cols].values.astype(np.float32))
        else:
            features.append(np.zeros(len(cols), dtype=np.float32))
    return np.array(features, dtype=np.float32)


def extract_structure_delta(structure_delta: Dict, site_ids: List[str]) -> np.ndarray:
    """7-dim structure delta features."""
    return np.array(
        [structure_delta.get(str(sid), np.zeros(7)) for sid in site_ids],
        dtype=np.float32,
    )


def build_hand_features(site_ids, sequences, structure_delta, loop_df) -> np.ndarray:
    """Build 40-dim hand feature matrix (motif 24 + struct_delta 7 + loop 9)."""
    motif = extract_motif_features(sequences, site_ids)
    struct = extract_structure_delta(structure_delta, site_ids)
    loop = extract_loop_features(loop_df, site_ids)
    hand = np.concatenate([motif, struct, loop], axis=1)
    return np.nan_to_num(hand, nan=0.0)


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
        "auroc": auroc,
        "auprc": auprc,
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


FEATURE_NAMES = (
    # Motif (24-dim)
    ["motif_UC", "motif_CC", "motif_AC", "motif_GC",
     "motif_CA", "motif_CG", "motif_CU", "motif_CC_3p",
     "trinuc_m2_A", "trinuc_m2_C", "trinuc_m2_G", "trinuc_m2_U",
     "trinuc_m1_A", "trinuc_m1_C", "trinuc_m1_G", "trinuc_m1_U",
     "trinuc_p1_A", "trinuc_p1_C", "trinuc_p1_G", "trinuc_p1_U",
     "trinuc_p2_A", "trinuc_p2_C", "trinuc_p2_G", "trinuc_p2_U"] +
    # Structure delta (7-dim)
    ["delta_pairing_center", "delta_accessibility_center", "delta_entropy_center",
     "delta_mfe", "mean_delta_pairing_window", "std_delta_pairing_window",
     "mean_delta_accessibility_window"] +
    # Loop geometry (9-dim)
    ["is_unpaired", "loop_size", "dist_to_junction", "dist_to_apex",
     "relative_loop_position", "left_stem_length", "right_stem_length",
     "max_adjacent_stem_length", "local_unpaired_fraction"]
)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    if not SPLITS_CSV.exists():
        logger.error("Splits CSV not found: %s", SPLITS_CSV)
        sys.exit(1)
    if not SEQ_JSON.exists():
        logger.error("Sequences JSON not found: %s", SEQ_JSON)
        sys.exit(1)

    df = pd.read_csv(SPLITS_CSV)
    logger.info("Loaded %d sites", len(df))

    with open(SEQ_JSON) as f:
        sequences = json.load(f)
    logger.info("Loaded %d sequences", len(sequences))

    # Load loop features — HARD FAIL if missing to prevent silent zero-vector bug.
    # Without loop_position_per_site_v2.csv, GB_HandFeatures trains on zero loop features,
    # producing wrong importances and ~0.015 lower AUROC with no error raised.
    if not LOOP_POS_CSV.exists():
        logger.error(
            "FATAL: loop_position_per_site_v2.csv not found: %s\n"
            "Run loop_position step (exp_loop_position_analysis.py or equivalent) FIRST.\n"
            "Without it, all 9 loop geometry features (is_unpaired, relative_loop_position, "
            "etc.) would be silent zero-vectors — GB_HandFeatures results would be wrong.",
            LOOP_POS_CSV,
        )
        sys.exit(1)
    loop_df = pd.read_csv(LOOP_POS_CSV)
    loop_df["site_id"] = loop_df["site_id"].astype(str)
    loop_df = loop_df.set_index("site_id")
    logger.info("Loaded %d loop features", len(loop_df))

    # Load structure cache — HARD FAIL if missing.
    # Without this, all 7 structure delta features (delta_pairing, delta_mfe, etc.)
    # would be zero-vectors, reducing classifier from 40-dim to 33-dim silently.
    if not STRUCT_CACHE.exists():
        logger.error(
            "FATAL: Structure cache not found: %s\n"
            "Run structure cache generation first! Without it, all 7 structure delta "
            "features would be zero-vectors — classification results would be incomplete.",
            STRUCT_CACHE,
        )
        sys.exit(1)
    struct_data = np.load(str(STRUCT_CACHE), allow_pickle=True)
    struct_sids = [str(s) for s in struct_data["site_ids"]]
    delta_features = struct_data["delta_features"]
    structure_delta = dict(zip(struct_sids, delta_features))
    logger.info("Loaded %d structure delta entries", len(structure_delta))

    if "enzyme" not in df.columns:
        logger.error("No 'enzyme' column in splits CSV")
        sys.exit(1)

    label_col = "is_edited" if "is_edited" in df.columns else "label"
    enzymes = sorted(df["enzyme"].unique())
    logger.info("Enzymes: %s", enzymes)

    all_results = {}

    for enzyme in enzymes:
        logger.info("=" * 60)
        logger.info("Classification for: %s", enzyme)
        logger.info("=" * 60)

        enz_df = df[df["enzyme"] == enzyme].copy()
        n_pos = (enz_df[label_col] == 1).sum()
        n_neg = (enz_df[label_col] == 0).sum()

        logger.info("  Positive: %d, Negative: %d", n_pos, n_neg)

        if n_pos < MIN_POSITIVE_SITES:
            logger.warning("  Skipping %s: only %d positives (need >= %d)",
                           enzyme, n_pos, MIN_POSITIVE_SITES)
            all_results[enzyme] = {
                "enzyme": enzyme,
                "skipped": True,
                "reason": f"Insufficient positives ({n_pos} < {MIN_POSITIVE_SITES})",
                "n_positive": int(n_pos),
                "n_negative": int(n_neg),
            }
            continue

        if n_neg < MIN_NEGATIVE_SITES:
            logger.warning("  Skipping %s: only %d negatives (need >= %d)",
                           enzyme, n_neg, MIN_NEGATIVE_SITES)
            all_results[enzyme] = {
                "enzyme": enzyme,
                "skipped": True,
                "reason": f"Insufficient negatives ({n_neg} < {MIN_NEGATIVE_SITES})",
                "n_positive": int(n_pos),
                "n_negative": int(n_neg),
            }
            continue

        # Build features
        site_ids = enz_df["site_id"].tolist()
        X = build_hand_features(site_ids, sequences, structure_delta, loop_df)
        y = enz_df[label_col].values.astype(int)

        logger.info("  Feature matrix: %s", X.shape)

        # Sanity: check for all-zero loop features
        loop_start = 24 + 7  # after motif and struct delta
        loop_block = X[:, loop_start:loop_start + 9]
        if np.all(loop_block == 0) and n_pos > 50:
            logger.warning(
                "  SANITY FAIL: All 9 loop features are zero for %s! "
                "Loop position CSV may be missing or not covering these sites.",
                enzyme
            )

        # 5-fold CV
        kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
        fold_results = []
        fold_importances = []

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # XGBoost with same hyperparameters as A3A classifier
            scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
            model = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                random_state=SEED + fold_idx,
                eval_metric="logloss",
            )
            model.fit(X_train, y_train, verbose=False)

            y_score = model.predict_proba(X_test)[:, 1]
            metrics = compute_binary_metrics(y_test, y_score)

            # Sanity checks
            if metrics["auroc"] < 0.5:
                logger.warning("  SANITY FAIL: %s fold %d AUROC=%.3f < 0.5. "
                               "Labels may be inverted!", enzyme, fold_idx, metrics["auroc"])
            if metrics["auroc"] > 0.99:
                logger.warning("  SANITY FAIL: %s fold %d AUROC=%.3f > 0.99. "
                               "Possible data leakage!", enzyme, fold_idx, metrics["auroc"])

            fold_results.append(metrics)
            fold_importances.append(model.feature_importances_)
            logger.info("  Fold %d: AUROC=%.4f, AUPRC=%.4f, F1=%.4f",
                        fold_idx, metrics["auroc"], metrics["auprc"], metrics["f1"])

        # Aggregate across folds
        agg = {}
        for metric_name in ["auroc", "auprc", "f1", "precision", "recall"]:
            vals = [fr[metric_name] for fr in fold_results if not np.isnan(fr[metric_name])]
            if vals:
                agg[metric_name] = float(np.mean(vals))
                agg[f"{metric_name}_std"] = float(np.std(vals))
            else:
                agg[metric_name] = float("nan")
                agg[f"{metric_name}_std"] = float("nan")

        # Feature importance (averaged across all 5 folds)
        mean_importance = np.mean(fold_importances, axis=0)
        feat_imp = sorted(
            zip(FEATURE_NAMES, mean_importance.tolist()),
            key=lambda x: -x[1],
        )

        all_results[enzyme] = {
            "enzyme": enzyme,
            "skipped": False,
            "n_positive": int(n_pos),
            "n_negative": int(n_neg),
            "n_folds": 5,
            "metrics": agg,
            "fold_results": fold_results,
            "top_features": [
                {"feature": name, "importance": imp}
                for name, imp in feat_imp[:15]
            ],
        }

        # Print fold summary
        print(f"\n--- {enzyme} Classification (5-fold CV) ---")
        print(f"  n_pos={n_pos}, n_neg={n_neg}")
        print(f"  AUROC: {agg['auroc']:.4f} +/- {agg['auroc_std']:.4f}")
        print(f"  AUPRC: {agg['auprc']:.4f} +/- {agg['auprc_std']:.4f}")
        print(f"  F1:    {agg['f1']:.4f} +/- {agg['f1_std']:.4f}")
        print(f"  Top 5 features:")
        for name, imp in feat_imp[:5]:
            print(f"    {name}: {imp:.4f}")

        # Significance criterion: AUROC > 0.75 = meaningful
        if agg["auroc"] > 0.75:
            print(f"  => SIGNIFICANT: Model learns meaningful signal (AUROC > 0.75)")
        else:
            print(f"  => NOT SIGNIFICANT: AUROC <= 0.75, model does not clearly distinguish sites")

    # ---------------------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------------------
    def serialize(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    with open(OUTPUT_DIR / "per_enzyme_classification_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=serialize)
    logger.info("Results saved to %s", OUTPUT_DIR / "per_enzyme_classification_results.json")

    # Print final summary
    print("\n" + "=" * 80)
    print("CLASSIFICATION SUMMARY (GB_HandFeatures, 5-fold CV)")
    print("=" * 80)
    print(f"{'Enzyme':>6s} {'n_pos':>6s} {'n_neg':>6s} {'AUROC':>10s} {'AUPRC':>10s} "
          f"{'F1':>10s} {'Top Feature':>25s} {'Signal':>10s}")
    print("-" * 95)
    for enzyme in sorted(all_results.keys()):
        r = all_results[enzyme]
        if r.get("skipped"):
            print(f"{enzyme:>6s} {r['n_positive']:>6d} {r['n_negative']:>6d} "
                  f"{'SKIPPED':>10s} {'':>10s} {'':>10s} {r['reason']:>25s} {'N/A':>10s}")
            continue
        m = r["metrics"]
        top_feat = r["top_features"][0]["feature"] if r["top_features"] else "N/A"
        signal = "YES" if m["auroc"] > 0.75 else "NO"
        print(f"{enzyme:>6s} {r['n_positive']:>6d} {r['n_negative']:>6d} "
              f"{m['auroc']:.4f}+/-{m['auroc_std']:.3f} "
              f"{m['auprc']:.4f}+/-{m['auprc_std']:.3f} "
              f"{m['f1']:.4f}+/-{m['f1_std']:.3f} "
              f"{top_feat:>25s} {signal:>10s}")
    print("=" * 80)


if __name__ == "__main__":
    main()
