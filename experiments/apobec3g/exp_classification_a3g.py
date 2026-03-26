#!/usr/bin/env python
"""Classification experiment for APOBEC3G editing site prediction.

Small dataset (n=119 positives from Dang 2019 NK_Hyp) requires special handling:
- Bootstrap CI for AUROC (1000 iterations)
- Reduced model complexity (max_depth=4, n_estimators=200)
- StratifiedKFold to maintain class balance

Models: GB_HandFeatures, GB_AllFeatures, MotifOnly, StructOnly

Usage:
    conda run -n quris python experiments/apobec3g/exp_classification_a3g.py
"""

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
SPLITS_CSV = PROJECT_ROOT / "data/processed/multi_enzyme/splits_multi_enzyme_v2_with_negatives.csv"
SPLITS_CSV_FALLBACK = PROJECT_ROOT / "data/processed/multi_enzyme/splits_multi_enzyme_v2.csv"
SEQUENCES_JSON = PROJECT_ROOT / "data/processed/multi_enzyme/multi_enzyme_sequences_v2_with_negatives.json"
STRUCT_CACHE = PROJECT_ROOT / "data/processed/multi_enzyme/structure_cache_multi_enzyme_v2.npz"
LOOP_POS_CSV = PROJECT_ROOT / "data/processed/multi_enzyme/loop_position_per_site_v2.csv"
OUTPUT_DIR = Path(__file__).parent / "outputs" / "classification"

SEED = 42


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_binary_metrics(y_true, y_score):
    """Compute binary classification metrics."""
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
    """Bootstrap 95% CI for AUROC."""
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


# ---------------------------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------------------------

def extract_motif_features(sequences, site_ids):
    """24-dim motif features."""
    features = []
    for sid in site_ids:
        seq = sequences.get(str(sid), "N" * 201).upper().replace("T", "U")
        ep = 100
        up = seq[ep - 1] if ep > 0 else "N"
        down = seq[ep + 1] if ep < len(seq) - 1 else "N"
        feat_5p = [1 if up + "C" == m else 0 for m in ["UC", "CC", "AC", "GC"]]
        feat_3p = [1 if "C" + down == m else 0 for m in ["CA", "CG", "CU", "CC"]]
        trinuc_up = [0] * 8
        bases = ["A", "C", "G", "U"]
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


def extract_loop_features(loop_df, site_ids):
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


def extract_structure_delta(structure_delta, site_ids):
    """7-dim structure delta features."""
    return np.array([structure_delta.get(str(sid), np.zeros(7)) for sid in site_ids],
                    dtype=np.float32)


def build_hand_features(site_ids, sequences, structure_delta, loop_df):
    """Build 40-dim hand feature matrix (motif 24 + struct_delta 7 + loop 9)."""
    motif = extract_motif_features(sequences, site_ids)
    struct = extract_structure_delta(structure_delta, site_ids)
    loop = extract_loop_features(loop_df, site_ids)
    hand = np.concatenate([motif, struct, loop], axis=1)
    return np.nan_to_num(hand, nan=0.0)


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_data():
    """Load A3G sites with negatives, sequences, structure, loop features."""
    logger.info("Loading data...")

    # Load splits
    if SPLITS_CSV.exists():
        df = pd.read_csv(SPLITS_CSV)
        logger.info("  Loaded from %s", SPLITS_CSV.name)
    elif SPLITS_CSV_FALLBACK.exists():
        df = pd.read_csv(SPLITS_CSV_FALLBACK)
        logger.info("  Loaded from fallback %s", SPLITS_CSV_FALLBACK.name)
    else:
        raise FileNotFoundError(f"Neither {SPLITS_CSV} nor {SPLITS_CSV_FALLBACK} found")

    # Filter to A3G
    df = df[df["enzyme"] == "A3G"].copy()

    # Check for is_edited / label column
    if "is_edited" in df.columns and "label" not in df.columns:
        df["label"] = df["is_edited"]
    elif "label" not in df.columns:
        # Positives only (no negatives file yet)
        logger.warning("No label column found. All sites assumed positive.")
        df["label"] = 1

    n_pos = int((df["label"] == 1).sum())
    n_neg = int((df["label"] == 0).sum())
    logger.info("  A3G sites: %d (pos=%d, neg=%d)", len(df), n_pos, n_neg)

    if n_neg == 0:
        logger.error("No negative sites found for A3G. Need splits_multi_enzyme_v2_with_negatives.csv")
        return None

    needed_sids = set(df["site_id"].astype(str).tolist())

    # Sequences
    with open(SEQUENCES_JSON) as f:
        _seq = json.load(f)
    sequences = {k: v for k, v in _seq.items() if str(k) in needed_sids}
    del _seq
    logger.info("  %d sequences loaded", len(sequences))

    # Structure cache
    structure_delta = {}
    if STRUCT_CACHE.exists():
        data = np.load(str(STRUCT_CACHE), allow_pickle=True)
        struct_sids = [str(s) for s in data["site_ids"]]
        for i, sid in enumerate(struct_sids):
            if sid in needed_sids:
                structure_delta[sid] = data["delta_features"][i]
        del data
        gc.collect()
    logger.info("  %d structure delta features loaded", len(structure_delta))

    # Loop features
    loop_df = pd.DataFrame()
    if LOOP_POS_CSV.exists():
        loop_df = pd.read_csv(LOOP_POS_CSV)
        loop_df["site_id"] = loop_df["site_id"].astype(str)
        loop_df = loop_df.set_index("site_id")
        loop_df = loop_df[loop_df.index.isin(needed_sids)]
        logger.info("  %d loop features loaded", len(loop_df))

    return {
        "df": df,
        "sequences": sequences,
        "structure_delta": structure_delta,
        "loop_df": loop_df,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_data()
    if data is None:
        logger.error("Data loading failed. Exiting.")
        return

    df = data["df"]
    sequences = data["sequences"]
    structure_delta = data["structure_delta"]
    loop_df = data["loop_df"]

    site_ids = df["site_id"].values
    labels = df["label"].values.astype(int)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())

    # Build features
    logger.info("Building features...")
    hand_40 = build_hand_features(site_ids, sequences, structure_delta, loop_df)
    motif_24 = extract_motif_features(sequences, site_ids)
    struct_delta_7 = extract_structure_delta(structure_delta, site_ids)
    loop_9 = extract_loop_features(loop_df, site_ids)

    logger.info("  Hand features shape: %s", hand_40.shape)

    # Model configurations
    models = {
        "GB_HandFeatures": {"features": hand_40, "desc": "40-dim (motif+struct+loop)"},
        "GB_AllFeatures": {"features": hand_40, "desc": "40-dim (same as hand for A3G)"},
        "MotifOnly": {"features": motif_24, "desc": "24-dim motif"},
        "StructOnly": {"features": np.concatenate([struct_delta_7, loop_9], axis=1),
                       "desc": "16-dim (struct_delta+loop)"},
    }

    # 5-fold StratifiedKFold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results = {}

    # Collect all predictions for bootstrap
    all_predictions = {name: {"y_true": [], "y_score": []} for name in models}

    for name, cfg in models.items():
        logger.info("\n--- %s (%s) ---", name, cfg["desc"])
        X = cfg["features"]
        fold_results = []

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, labels)):
            fold_seed = SEED + fold_idx

            X_train, y_train = X[train_idx], labels[train_idx]
            X_test, y_test = X[test_idx], labels[test_idx]

            try:
                from xgboost import XGBClassifier
                gb = XGBClassifier(
                    n_estimators=200, max_depth=4, learning_rate=0.1,
                    subsample=0.8, min_child_weight=5,
                    scale_pos_weight=n_neg / max(n_pos, 1),
                    random_state=fold_seed, n_jobs=1,
                    verbosity=0, eval_metric="logloss",
                )
                gb.fit(X_train, y_train)
                y_score = gb.predict_proba(X_test)[:, 1]
            except ImportError:
                logger.error("xgboost not available")
                return

            metrics = compute_binary_metrics(y_test, y_score)
            fold_results.append(metrics)
            all_predictions[name]["y_true"].extend(y_test.tolist())
            all_predictions[name]["y_score"].extend(y_score.tolist())

            logger.info("  Fold %d: AUROC=%.4f AUPRC=%.4f F1=%.4f",
                        fold_idx + 1, metrics["auroc"], metrics["auprc"], metrics["f1"])

            # Save feature importance for the last fold of GB_HandFeatures
            if name == "GB_HandFeatures" and fold_idx == 4:
                feat_names = (
                    [f"motif_{i}" for i in range(24)] +
                    [f"struct_delta_{i}" for i in range(7)] +
                    ["is_unpaired", "loop_size", "dist_to_junction", "dist_to_apex",
                     "relative_loop_position", "left_stem_length", "right_stem_length",
                     "max_adjacent_stem_length", "local_unpaired_fraction"]
                )
                importances = gb.feature_importances_
                feat_imp = sorted(zip(feat_names, importances),
                                  key=lambda x: x[1], reverse=True)
                imp_df = pd.DataFrame(feat_imp, columns=["feature", "importance"])
                imp_df.to_csv(OUTPUT_DIR / "feature_importance_a3g.csv", index=False)
                logger.info("  Feature importance saved")

            del gb

        # Bootstrap AUROC
        y_true_all = np.array(all_predictions[name]["y_true"])
        y_score_all = np.array(all_predictions[name]["y_score"])
        boot_mean, boot_lo, boot_hi = bootstrap_auroc(y_true_all, y_score_all)

        results[name] = {
            "fold_aurocs": [r["auroc"] for r in fold_results],
            "fold_auprcs": [r["auprc"] for r in fold_results],
            "fold_f1s": [r["f1"] for r in fold_results],
            "mean_auroc": float(np.mean([r["auroc"] for r in fold_results])),
            "std_auroc": float(np.std([r["auroc"] for r in fold_results])),
            "mean_auprc": float(np.mean([r["auprc"] for r in fold_results])),
            "std_auprc": float(np.std([r["auprc"] for r in fold_results])),
            "mean_f1": float(np.mean([r["f1"] for r in fold_results])),
            "bootstrap_auroc_mean": boot_mean,
            "bootstrap_auroc_ci95_lo": boot_lo,
            "bootstrap_auroc_ci95_hi": boot_hi,
        }

        logger.info("  Mean AUROC=%.4f +/- %.4f  Bootstrap 95%% CI=[%.4f, %.4f]",
                    results[name]["mean_auroc"], results[name]["std_auroc"],
                    boot_lo, boot_hi)

    # Save results
    output = {
        "enzyme": "A3G",
        "n_positive": n_pos,
        "n_negative": n_neg,
        "small_dataset_warning": f"n={n_pos} limits statistical reliability",
        "cv_folds": 5,
        "models": results,
    }

    # Add feature importance if available
    imp_path = OUTPUT_DIR / "feature_importance_a3g.csv"
    if imp_path.exists():
        imp_df = pd.read_csv(imp_path)
        output["feature_importance"] = {
            row["feature"]: float(row["importance"])
            for _, row in imp_df.iterrows()
        }

    out_path = OUTPUT_DIR / "classification_a3g_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    elapsed = time.time() - t_start
    logger.info("\nResults saved to %s (%.1f sec)", out_path, elapsed)


if __name__ == "__main__":
    main()
