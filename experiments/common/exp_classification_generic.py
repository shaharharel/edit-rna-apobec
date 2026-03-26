#!/usr/bin/env python
"""Generic classification experiment for any APOBEC enzyme category.

Runs 5-fold StratifiedKFold CV with GB_HandFeatures, MotifOnly, StructOnly.
Supports small datasets (bootstrap CI) and larger datasets.

Usage:
    conda run -n quris python experiments/common/exp_classification_generic.py --enzyme A3G
    conda run -n quris python experiments/common/exp_classification_generic.py --enzyme A3A_A3G
    conda run -n quris python experiments/common/exp_classification_generic.py --enzyme Neither
    conda run -n quris python experiments/common/exp_classification_generic.py --enzyme Unknown
"""

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.apobec_feature_extraction import (
    extract_motif_features, extract_loop_features,
    extract_structure_delta_features, build_hand_features,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Paths - use v3 if available, fall back to v2
_ME_DIR = PROJECT_ROOT / "data/processed/multi_enzyme"
_V3_NEG = _ME_DIR / "splits_multi_enzyme_v3_with_negatives.csv"
_V2_NEG = _ME_DIR / "splits_multi_enzyme_v2_with_negatives.csv"
SPLITS_CSV = _V3_NEG if _V3_NEG.exists() else _V2_NEG

_V3_SEQ = _ME_DIR / "multi_enzyme_sequences_v3_with_negatives.json"
_V2_SEQ = _ME_DIR / "multi_enzyme_sequences_v2_with_negatives.json"
SEQUENCES_JSON = _V3_SEQ if _V3_SEQ.exists() else _V2_SEQ

_V3_SC = _ME_DIR / "structure_cache_multi_enzyme_v3.npz"
_V2_SC = _ME_DIR / "structure_cache_multi_enzyme_v2.npz"
STRUCT_CACHE = _V3_SC if _V3_SC.exists() else _V2_SC

_V3_LP = _ME_DIR / "loop_position_per_site_v3.csv"
_V2_LP = _ME_DIR / "loop_position_per_site_v2.csv"
LOOP_POS_CSV = _V3_LP if _V3_LP.exists() else _V2_LP

SEED = 42


def compute_binary_metrics(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return {k: float("nan") for k in ["auroc", "auprc", "f1", "precision", "recall"]}
    auroc = float(roc_auc_score(y_true, y_score))
    auprc = float(average_precision_score(y_true, y_score))
    prec_arr, rec_arr, thresholds = precision_recall_curve(y_true, y_score)
    f1_arr = 2 * prec_arr[:-1] * rec_arr[:-1] / (prec_arr[:-1] + rec_arr[:-1] + 1e-8)
    best_idx = int(np.argmax(f1_arr))
    threshold = float(thresholds[best_idx]) if len(thresholds) > 0 else 0.5
    y_pred = (np.array(y_score) >= threshold).astype(int)
    return {
        "auroc": auroc, "auprc": auprc,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def bootstrap_auroc(y_true, y_score, n_bootstrap=1000, seed=42):
    rng = np.random.RandomState(seed)
    aurocs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aurocs.append(roc_auc_score(y_true[idx], y_score[idx]))
    if not aurocs:
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(aurocs)), float(np.percentile(aurocs, 2.5)), float(np.percentile(aurocs, 97.5))


def load_data(enzyme):
    """Load data for a specific enzyme category."""
    logger.info("Loading data for enzyme=%s...", enzyme)

    if not SPLITS_CSV.exists():
        raise FileNotFoundError(f"Splits file not found: {SPLITS_CSV}")
    df = pd.read_csv(SPLITS_CSV)

    # Filter to target enzyme
    df = df[df["enzyme"] == enzyme].copy()
    if "is_edited" in df.columns and "label" not in df.columns:
        df["label"] = df["is_edited"]
    elif "label" not in df.columns:
        df["label"] = 1

    n_pos = int((df["label"] == 1).sum())
    n_neg = int((df["label"] == 0).sum())
    logger.info("  %s: %d sites (pos=%d, neg=%d)", enzyme, len(df), n_pos, n_neg)

    if n_neg == 0:
        logger.error("No negatives for %s. Exiting.", enzyme)
        return None

    needed_sids = set(df["site_id"].astype(str))

    # Sequences
    with open(SEQUENCES_JSON) as f:
        _seq = json.load(f)
    sequences = {k: v for k, v in _seq.items() if str(k) in needed_sids}
    del _seq
    logger.info("  %d sequences", len(sequences))

    # Structure delta
    structure_delta = {}
    if STRUCT_CACHE.exists():
        data = np.load(str(STRUCT_CACHE), allow_pickle=True)
        struct_sids = [str(s) for s in data["site_ids"]]
        for i, sid in enumerate(struct_sids):
            if sid in needed_sids:
                structure_delta[sid] = data["delta_features"][i]
        del data; gc.collect()
    logger.info("  %d structure delta features", len(structure_delta))

    # Loop features
    if not LOOP_POS_CSV.exists():
        raise FileNotFoundError(
            f"Loop position CSV not found: {LOOP_POS_CSV}. "
            "Run generate_loop_positions.py first!"
        )
    loop_df = pd.read_csv(LOOP_POS_CSV)
    loop_df["site_id"] = loop_df["site_id"].astype(str)
    loop_df = loop_df.set_index("site_id")
    loop_df = loop_df[loop_df.index.isin(needed_sids)]
    logger.info("  %d loop features", len(loop_df))

    return {
        "df": df, "sequences": sequences,
        "structure_delta": structure_delta, "loop_df": loop_df,
    }


def run_classification(enzyme, output_dir):
    t_start = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_data(enzyme)
    if data is None:
        return

    df = data["df"]
    sequences = data["sequences"]
    structure_delta = data["structure_delta"]
    loop_df = data["loop_df"]

    site_ids = df["site_id"].values
    labels = df["label"].values.astype(int)
    n_pos = int((labels == 1).sum())

    # Build features
    LOOP_COLS = [
        "is_unpaired", "loop_size", "dist_to_junction", "dist_to_apex",
        "relative_loop_position", "left_stem_length", "right_stem_length",
        "max_adjacent_stem_length", "local_unpaired_fraction",
    ]

    motif_24 = extract_motif_features(sequences, list(site_ids))
    loop_9 = extract_loop_features(loop_df, list(site_ids))
    struct_7 = extract_structure_delta_features(structure_delta, list(site_ids))
    hand_40 = np.concatenate([motif_24, struct_7, loop_9], axis=1)
    hand_40 = np.nan_to_num(hand_40, nan=0.0)

    # Adjust model complexity for small datasets
    small_dataset = n_pos < 200
    xgb_params = {
        "n_estimators": 200 if small_dataset else 500,
        "max_depth": 4 if small_dataset else 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": SEED,
        "use_label_encoder": False,
        "eval_metric": "logloss",
    }

    models = {
        "GB_HandFeatures": hand_40,
        "MotifOnly": motif_24,
        "StructOnly": np.nan_to_num(np.concatenate([struct_7, loop_9], axis=1), nan=0.0),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results = {}
    all_predictions = {name: {"y_true": [], "y_score": []} for name in models}
    fold_importances = {name: [] for name in models}

    for name, X in models.items():
        logger.info("\n--- %s (dim=%d) ---", name, X.shape[1])
        fold_metrics = []

        for fold_i, (train_idx, test_idx) in enumerate(skf.split(X, labels)):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = labels[train_idx], labels[test_idx]

            clf = XGBClassifier(**xgb_params)
            clf.fit(X_tr, y_tr)
            y_score = clf.predict_proba(X_te)[:, 1]

            metrics = compute_binary_metrics(y_te, y_score)
            fold_metrics.append(metrics)
            all_predictions[name]["y_true"].extend(y_te.tolist())
            all_predictions[name]["y_score"].extend(y_score.tolist())
            fold_importances[name].append(clf.feature_importances_)

            logger.info("  Fold %d: AUROC=%.3f AUPRC=%.3f", fold_i + 1,
                         metrics["auroc"], metrics["auprc"])

        # Average metrics
        avg = {k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
        std = {k: np.std([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
        logger.info("  Mean: AUROC=%.3f±%.3f AUPRC=%.3f±%.3f",
                     avg["auroc"], std["auroc"], avg["auprc"], std["auprc"])

        # Bootstrap CI
        all_yt = np.array(all_predictions[name]["y_true"])
        all_ys = np.array(all_predictions[name]["y_score"])
        bs_mean, bs_lo, bs_hi = bootstrap_auroc(all_yt, all_ys)

        results[name] = {
            "mean_metrics": avg,
            "std_metrics": std,
            "fold_metrics": fold_metrics,
            "bootstrap_auroc": {"mean": bs_mean, "ci_lo": bs_lo, "ci_hi": bs_hi},
        }

    # Feature importance (averaged across folds)
    feature_names_map = {
        "GB_HandFeatures": (
            [f"motif_{i}" for i in range(24)] +
            [f"struct_delta_{i}" for i in range(7)] +
            LOOP_COLS
        ),
        "MotifOnly": [f"motif_{i}" for i in range(24)],
        "StructOnly": [f"struct_delta_{i}" for i in range(7)] + LOOP_COLS,
    }

    importance_data = {}
    for name in models:
        if fold_importances[name]:
            avg_imp = np.mean(fold_importances[name], axis=0)
            feat_names = feature_names_map.get(name, [f"f{i}" for i in range(len(avg_imp))])
            importance_data[name] = dict(zip(feat_names, avg_imp.tolist()))

    # Save results
    out = {
        "enzyme": enzyme,
        "n_positives": n_pos,
        "n_negatives": int((labels == 0).sum()),
        "n_folds": 5,
        "models": results,
        "feature_importance": importance_data,
        "xgb_params": xgb_params,
    }

    with open(output_dir / "classification_results.json", "w") as f:
        json.dump(out, f, indent=2, default=str)

    # Feature importance CSV
    if "GB_HandFeatures" in importance_data:
        imp_df = pd.DataFrame([
            {"feature": k, "importance": v}
            for k, v in sorted(importance_data["GB_HandFeatures"].items(),
                               key=lambda x: -x[1])
        ])
        imp_df.to_csv(output_dir / "feature_importance.csv", index=False)

    elapsed = time.time() - t_start
    logger.info("\n=== %s Classification Complete (%.0fs) ===", enzyme, elapsed)
    for name, r in results.items():
        m = r["mean_metrics"]
        bs = r["bootstrap_auroc"]
        logger.info("  %s: AUROC=%.3f [%.3f–%.3f] AUPRC=%.3f",
                     name, m["auroc"], bs["ci_lo"], bs["ci_hi"], m["auprc"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--enzyme", required=True,
                        help="Enzyme category (A3G, A3A_A3G, Neither, Unknown)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: experiments/<enzyme>/outputs/classification)")
    args = parser.parse_args()

    # Map enzyme to experiment directory
    enzyme_dir_map = {
        "A3G": "apobec3g",
        "A3A_A3G": "apobec_both",
        "Neither": "apobec_neither",
        "Unknown": "apobec_unknown",
        "A3A": "apobec3a",
        "A3B": "apobec3b",
    }
    exp_dir = enzyme_dir_map.get(args.enzyme, f"apobec_{args.enzyme.lower()}")
    output_dir = args.output_dir or (PROJECT_ROOT / "experiments" / exp_dir / "outputs" / "classification")

    run_classification(args.enzyme, output_dir)


if __name__ == "__main__":
    main()
