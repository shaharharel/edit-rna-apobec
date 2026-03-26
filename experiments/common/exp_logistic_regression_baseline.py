#!/usr/bin/env python
"""Logistic regression baseline on 40-dim hand features.

Tests whether XGBoost's nonlinearity contributes signal beyond a linear model.
Runs for A3A (primary dataset) and all new Levanon categories.

Usage:
    conda run -n quris python experiments/common/exp_logistic_regression_baseline.py
"""

import gc
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.apobec_feature_extraction import (
    extract_motif_features, extract_loop_features,
    extract_structure_delta_features,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_ME_DIR = PROJECT_ROOT / "data/processed/multi_enzyme"
_V3 = _ME_DIR / "splits_multi_enzyme_v3_with_negatives.csv"
_V2 = _ME_DIR / "splits_multi_enzyme_v2_with_negatives.csv"
SPLITS_CSV = _V3 if _V3.exists() else _V2

_V3_SEQ = _ME_DIR / "multi_enzyme_sequences_v3_with_negatives.json"
_V2_SEQ = _ME_DIR / "multi_enzyme_sequences_v2_with_negatives.json"
SEQS_JSON = _V3_SEQ if _V3_SEQ.exists() else _V2_SEQ

_V3_SC = _ME_DIR / "structure_cache_multi_enzyme_v3.npz"
_V2_SC = _ME_DIR / "structure_cache_multi_enzyme_v2.npz"
STRUCT_CACHE = _V3_SC if _V3_SC.exists() else _V2_SC

_V3_LP = _ME_DIR / "loop_position_per_site_v3.csv"
_V2_LP = _ME_DIR / "loop_position_per_site_v2.csv"
LOOP_CSV = _V3_LP if _V3_LP.exists() else _V2_LP

# Also test on A3A-specific splits
A3A_SPLITS = PROJECT_ROOT / "data/processed/splits_expanded_a3a.csv"
A3A_SEQS = PROJECT_ROOT / "data/processed/site_sequences.json"
A3A_STRUCT = PROJECT_ROOT / "data/processed/embeddings/vienna_structure_cache.npz"
A3A_LOOP = PROJECT_ROOT / "experiments/apobec3a/outputs" / "loop_position" / "loop_position_per_site.csv"

OUTPUT_DIR = PROJECT_ROOT / "experiments/common/outputs/logistic_regression"

SEED = 42
LOOP_COLS = [
    "is_unpaired", "loop_size", "dist_to_junction", "dist_to_apex",
    "relative_loop_position", "left_stem_length", "right_stem_length",
    "max_adjacent_stem_length", "local_unpaired_fraction",
]


def load_features(splits_csv, seqs_json, struct_cache_path, loop_csv_path, enzyme_filter=None):
    """Load features for a dataset."""
    df = pd.read_csv(splits_csv)
    if enzyme_filter:
        df = df[df["enzyme"] == enzyme_filter].copy()

    if "is_edited" in df.columns and "label" not in df.columns:
        df["label"] = df["is_edited"]

    n_pos = (df["label"] == 1).sum()
    n_neg = (df["label"] == 0).sum()
    if n_neg == 0:
        return None, None, None

    site_ids = df["site_id"].values
    labels = df["label"].values.astype(int)
    needed = set(str(s) for s in site_ids)

    # Sequences
    with open(seqs_json) as f:
        all_seqs = json.load(f)
    seqs = {k: v for k, v in all_seqs.items() if k in needed}
    del all_seqs

    # Structure delta
    struct_delta = {}
    if struct_cache_path.exists():
        data = np.load(str(struct_cache_path), allow_pickle=True)
        sids = [str(s) for s in data["site_ids"]]
        for i, sid in enumerate(sids):
            if sid in needed:
                struct_delta[sid] = data["delta_features"][i]
        del data; gc.collect()

    # Loop features
    loop_df = pd.DataFrame()
    if loop_csv_path.exists():
        loop_df = pd.read_csv(loop_csv_path)
        loop_df["site_id"] = loop_df["site_id"].astype(str)
        loop_df = loop_df.set_index("site_id")

    # Build 40-dim features
    motif = extract_motif_features(seqs, list(site_ids))
    struct = extract_structure_delta_features(struct_delta, list(site_ids))
    loop = extract_loop_features(loop_df, list(site_ids))
    hand_40 = np.concatenate([motif, struct, loop], axis=1)
    hand_40 = np.nan_to_num(hand_40, nan=0.0)

    return hand_40, labels, n_pos


def run_logreg(X, y, enzyme_name):
    """Run 5-fold CV with logistic regression and return results."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_aurocs = []
    fold_auprcs = []

    for fold_i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_te_scaled = scaler.transform(X_te)

        clf = LogisticRegression(max_iter=1000, random_state=SEED, C=1.0)
        clf.fit(X_tr_scaled, y_tr)
        y_score = clf.predict_proba(X_te_scaled)[:, 1]

        auroc = roc_auc_score(y_te, y_score)
        auprc = average_precision_score(y_te, y_score)
        fold_aurocs.append(auroc)
        fold_auprcs.append(auprc)

    mean_auroc = np.mean(fold_aurocs)
    std_auroc = np.std(fold_aurocs)
    mean_auprc = np.mean(fold_auprcs)
    std_auprc = np.std(fold_auprcs)

    logger.info("  %s: LR AUROC=%.3f±%.3f, AUPRC=%.3f±%.3f",
                enzyme_name, mean_auroc, std_auroc, mean_auprc, std_auprc)

    return {
        "enzyme": enzyme_name,
        "model": "LogisticRegression",
        "n_features": int(X.shape[1]),
        "mean_auroc": float(mean_auroc),
        "std_auroc": float(std_auroc),
        "mean_auprc": float(mean_auprc),
        "std_auprc": float(std_auprc),
        "fold_aurocs": [float(x) for x in fold_aurocs],
        "fold_auprcs": [float(x) for x in fold_auprcs],
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    # A3A (from splits_expanded_a3a.csv — the canonical A3A dataset)
    logger.info("=== A3A (canonical dataset) ===")
    # Find the loop CSV — try multiple locations
    a3a_loop = A3A_LOOP
    if not a3a_loop.exists():
        # Try alternative path
        alt = PROJECT_ROOT / "data/processed/loop_position_per_site.csv"
        if alt.exists():
            a3a_loop = alt
        else:
            # Use multi-enzyme v3 loop (covers A3A sites)
            a3a_loop = LOOP_CSV

    X, y, n_pos = load_features(A3A_SPLITS, A3A_SEQS, A3A_STRUCT, a3a_loop)
    if X is not None:
        logger.info("  A3A: %d samples (%d pos), %d features", len(y), n_pos, X.shape[1])
        results.append(run_logreg(X, y, "A3A"))

    # Multi-enzyme categories
    for enzyme in ["A3B", "A3G", "A3A_A3G", "Neither", "Unknown"]:
        logger.info("\n=== %s ===", enzyme)
        X, y, n_pos = load_features(SPLITS_CSV, SEQS_JSON, STRUCT_CACHE, LOOP_CSV, enzyme)
        if X is not None:
            logger.info("  %s: %d samples (%d pos), %d features", enzyme, len(y), n_pos, X.shape[1])
            results.append(run_logreg(X, y, enzyme))
        else:
            logger.warning("  %s: skipped (no negatives)", enzyme)

    # Save
    with open(OUTPUT_DIR / "logistic_regression_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print comparison table
    print("\n=== Logistic Regression vs GB Baseline ===")
    print(f"{'Enzyme':<12} {'LR AUROC':<14} {'GB AUROC (ref)':<14}")
    gb_ref = {"A3A": 0.923, "A3B": 0.831, "A3G": 0.929, "A3A_A3G": 0.941, "Neither": 0.840}
    for r in results:
        gb = gb_ref.get(r["enzyme"], "—")
        print(f"  {r['enzyme']:<10} {r['mean_auroc']:.3f}±{r['std_auroc']:.3f}   {gb}")

    logger.info("\nSaved to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
