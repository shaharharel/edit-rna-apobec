#!/usr/bin/env python
"""Unified multi-enzyme APOBEC editing prediction experiment.

Two tasks evaluated simultaneously:
  Task 1 — Binary: Is this cytidine edited? (pos vs neg, per-enzyme)
  Task 2 — Enzyme: Which enzyme edits it? (6-class among positives)

Models compared:
  1. GB_HandFeatures (40-dim): Interpretable baseline
  2. GB_MotifOnly (24-dim): Motif contribution
  3. GB_StructOnly (16-dim): Structure contribution
  4. GB_LoopOnly (9-dim): Loop geometry alone
  5. LogisticRegression (40-dim): Linear baseline
  6. GB_MotifAblated (16-dim): Hand features WITHOUT motif → residual signal
  7. RandomForest (40-dim): Alternative ensemble

Evaluation:
  - 5-fold StratifiedKFold CV for binary (per-enzyme)
  - 5-fold CV for 6-class enzyme classification (positives only)
  - Feature importance per enzyme (interpretability)
  - Cross-enzyme transfer (train on one, predict another)
  - Confusion matrix for enzyme prediction

Usage:
    conda run -n quris python experiments/multi_enzyme/exp_unified_multi_enzyme.py
"""

import gc
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    classification_report,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.apobec_feature_extraction import (
    extract_motif_features, extract_loop_features,
    extract_structure_delta_features,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_ME_DIR = PROJECT_ROOT / "data/processed/multi_enzyme"
SPLITS_CSV = _ME_DIR / "splits_multi_enzyme_v3_with_negatives.csv"
SEQS_JSON = _ME_DIR / "multi_enzyme_sequences_v3_with_negatives.json"
STRUCT_CACHE = _ME_DIR / "structure_cache_multi_enzyme_v3.npz"
LOOP_CSV = _ME_DIR / "loop_position_per_site_v3.csv"
OUTPUT_DIR = Path(__file__).parent / "outputs" / "unified_multi_enzyme"

SEED = 42
N_FOLDS = 5

# 40-dim feature names
FEATURE_NAMES = (
    ['5p_UC', '5p_CC', '5p_AC', '5p_GC',
     '3p_CA', '3p_CG', '3p_CU', '3p_CC',
     'm2_A', 'm2_C', 'm2_G', 'm2_U',
     'm1_A', 'm1_C', 'm1_G', 'm1_U',
     'p1_A', 'p1_C', 'p1_G', 'p1_U',
     'p2_A', 'p2_C', 'p2_G', 'p2_U'] +
    ['delta_pairing_center', 'delta_accessibility_center', 'delta_entropy_center',
     'delta_mfe', 'mean_delta_pairing_window', 'mean_delta_accessibility_window',
     'std_delta_pairing_window'] +
    ['is_unpaired', 'loop_size', 'dist_to_junction', 'dist_to_apex',
     'relative_loop_position', 'left_stem_length', 'right_stem_length',
     'max_adjacent_stem_length', 'local_unpaired_fraction']
)

MOTIF_IDX = list(range(0, 24))
STRUCT_DELTA_IDX = list(range(24, 31))
LOOP_IDX = list(range(31, 40))


def load_all_data():
    """Load all data for unified experiment."""
    logger.info("Loading data...")
    df = pd.read_csv(SPLITS_CSV)
    logger.info("  %d sites, enzymes: %s", len(df), df["enzyme"].value_counts().to_dict())

    with open(SEQS_JSON) as f:
        seqs = json.load(f)

    structure_delta = {}
    if STRUCT_CACHE.exists():
        data = np.load(str(STRUCT_CACHE), allow_pickle=True)
        sids = [str(s) for s in data["site_ids"]]
        for i, sid in enumerate(sids):
            structure_delta[sid] = data["delta_features"][i]
        del data; gc.collect()
    logger.info("  %d structure delta features", len(structure_delta))

    if not LOOP_CSV.exists():
        raise FileNotFoundError(f"Loop CSV not found: {LOOP_CSV}")
    loop_df = pd.read_csv(LOOP_CSV)
    loop_df["site_id"] = loop_df["site_id"].astype(str)
    loop_df = loop_df.set_index("site_id")
    logger.info("  %d loop features", len(loop_df))

    # Build 40-dim features for ALL sites
    site_ids = df["site_id"].values
    motif = extract_motif_features(seqs, list(site_ids))
    struct = extract_structure_delta_features(structure_delta, list(site_ids))
    loop = extract_loop_features(loop_df, list(site_ids))
    hand_40 = np.concatenate([motif, struct, loop], axis=1)
    hand_40 = np.nan_to_num(hand_40, nan=0.0)

    logger.info("  Feature matrix: %s", hand_40.shape)
    return df, hand_40


def bootstrap_auroc(y_true, y_score, n_boot=1000):
    rng = np.random.RandomState(SEED)
    aurocs = []
    for _ in range(n_boot):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aurocs.append(roc_auc_score(y_true[idx], y_score[idx]))
    if not aurocs:
        return 0, 0, 0
    return float(np.mean(aurocs)), float(np.percentile(aurocs, 2.5)), float(np.percentile(aurocs, 97.5))


def run_binary_per_enzyme(df, X_all, enzyme):
    """Run binary classification for one enzyme category."""
    mask = df["enzyme"] == enzyme
    sub_df = df[mask].copy()
    if "is_edited" in sub_df.columns:
        sub_df["label"] = sub_df["is_edited"]
    n_pos = (sub_df["label"] == 1).sum()
    n_neg = (sub_df["label"] == 0).sum()
    if n_neg == 0 or n_pos == 0:
        return None

    X = X_all[mask.values]
    y = sub_df["label"].values.astype(int)
    small = n_pos < 200

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    # Define models
    xgb_params = {
        "n_estimators": 200 if small else 500,
        "max_depth": 4 if small else 6,
        "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8,
        "random_state": SEED, "eval_metric": "logloss",
    }

    models = {
        "GB_HandFeatures": ("xgb", list(range(40)), xgb_params),
        "GB_MotifOnly": ("xgb", MOTIF_IDX, xgb_params),
        "GB_StructOnly": ("xgb", STRUCT_DELTA_IDX + LOOP_IDX, xgb_params),
        "GB_LoopOnly": ("xgb", LOOP_IDX, xgb_params),
        "GB_NoMotif": ("xgb", STRUCT_DELTA_IDX + LOOP_IDX, xgb_params),
        "CatBoost": ("catboost", list(range(40)), {
            "iterations": 500 if not small else 200,
            "depth": 6 if not small else 4,
            "learning_rate": 0.1, "random_seed": SEED,
            "verbose": 0, "auto_class_weights": "Balanced",
        }),
        "LogisticReg": ("lr", list(range(40)), {}),
        "RandomForest": ("rf", list(range(40)), {"n_estimators": 500, "max_depth": 10, "random_state": SEED}),
    }

    results = {}
    for mname, (mtype, feat_idx, params) in models.items():
        X_sub = X[:, feat_idx]
        fold_aurocs, fold_auprcs, fold_f1s = [], [], []
        all_yt, all_ys = [], []
        fold_importances = []

        for train_idx, test_idx in skf.split(X_sub, y):
            X_tr, X_te = X_sub[train_idx], X_sub[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            if mtype == "xgb":
                clf = XGBClassifier(**params)
                clf.fit(X_tr, y_tr)
                y_score = clf.predict_proba(X_te)[:, 1]
                fold_importances.append(clf.feature_importances_)
            elif mtype == "lr":
                scaler = StandardScaler()
                clf = LogisticRegression(max_iter=1000, C=1.0, random_state=SEED)
                clf.fit(scaler.fit_transform(X_tr), y_tr)
                y_score = clf.predict_proba(scaler.transform(X_te))[:, 1]
            elif mtype == "catboost":
                clf = CatBoostClassifier(**params)
                clf.fit(X_tr, y_tr)
                y_score = clf.predict_proba(X_te)[:, 1]
                fold_importances.append(clf.get_feature_importance())
            elif mtype == "rf":
                clf = RandomForestClassifier(**params)
                clf.fit(X_tr, y_tr)
                y_score = clf.predict_proba(X_te)[:, 1]
                fold_importances.append(clf.feature_importances_)

            fold_aurocs.append(roc_auc_score(y_te, y_score))
            fold_auprcs.append(average_precision_score(y_te, y_score))
            y_pred = (y_score >= 0.5).astype(int)
            fold_f1s.append(f1_score(y_te, y_pred, zero_division=0))
            all_yt.extend(y_te.tolist())
            all_ys.extend(y_score.tolist())

        bs_mean, bs_lo, bs_hi = bootstrap_auroc(np.array(all_yt), np.array(all_ys))

        # Average feature importance
        avg_imp = None
        if fold_importances:
            avg_imp = np.mean(fold_importances, axis=0)
            feat_names = [FEATURE_NAMES[i] for i in feat_idx]
            avg_imp = dict(zip(feat_names, avg_imp.tolist()))

        results[mname] = {
            "auroc": float(np.mean(fold_aurocs)),
            "auroc_std": float(np.std(fold_aurocs)),
            "auprc": float(np.mean(fold_auprcs)),
            "f1": float(np.mean(fold_f1s)),
            "bootstrap_ci": [bs_lo, bs_hi],
            "feature_importance": avg_imp,
        }

    return {"enzyme": enzyme, "n_pos": int(n_pos), "n_neg": int(n_neg), "models": results}


def run_enzyme_classification(df, X_all):
    """6-class enzyme classification among positives only."""
    pos = df[df["is_edited"] == 1].copy()
    X = X_all[df["is_edited"].values == 1]
    le = LabelEncoder()
    y = le.fit_transform(pos["enzyme"].values)
    classes = le.classes_.tolist()
    n_classes = len(classes)

    logger.info("Enzyme classification: %d positives, %d classes: %s",
                len(pos), n_classes, dict(zip(classes, np.bincount(y).tolist())))

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    models_spec = {
        "GB_HandFeatures": ("xgb", list(range(40))),
        "GB_MotifOnly": ("xgb", MOTIF_IDX),
        "GB_StructOnly": ("xgb", STRUCT_DELTA_IDX + LOOP_IDX),
        "LogisticReg": ("lr", list(range(40))),
    }

    results = {}
    for mname, (mtype, feat_idx) in models_spec.items():
        X_sub = X[:, feat_idx]
        all_yt, all_yp = [], []
        fold_accs = []

        for train_idx, test_idx in skf.split(X_sub, y):
            X_tr, X_te = X_sub[train_idx], X_sub[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            if mtype == "xgb":
                clf = XGBClassifier(
                    n_estimators=500, max_depth=6, learning_rate=0.1,
                    subsample=0.8, colsample_bytree=0.8,
                    random_state=SEED, eval_metric="mlogloss",
                    num_class=n_classes,
                )
                clf.fit(X_tr, y_tr)
                y_pred = clf.predict(X_te)
            elif mtype == "lr":
                scaler = StandardScaler()
                clf = LogisticRegression(max_iter=1000, C=1.0, random_state=SEED)
                clf.fit(scaler.fit_transform(X_tr), y_tr)
                y_pred = clf.predict(scaler.transform(X_te))

            fold_accs.append(float(np.mean(y_pred == y_te)))
            all_yt.extend(y_te.tolist())
            all_yp.extend(y_pred.tolist())

        cm = confusion_matrix(all_yt, all_yp)
        report = classification_report(all_yt, all_yp, target_names=classes, output_dict=True)

        results[mname] = {
            "accuracy": float(np.mean(fold_accs)),
            "accuracy_std": float(np.std(fold_accs)),
            "confusion_matrix": cm.tolist(),
            "per_class": {
                cls: {
                    "precision": report[cls]["precision"],
                    "recall": report[cls]["recall"],
                    "f1": report[cls]["f1-score"],
                    "support": report[cls]["support"],
                }
                for cls in classes
            },
            "macro_f1": report["macro avg"]["f1-score"],
            "weighted_f1": report["weighted avg"]["f1-score"],
        }
        logger.info("  %s: accuracy=%.3f, macro_f1=%.3f, weighted_f1=%.3f",
                     mname, np.mean(fold_accs), report["macro avg"]["f1-score"],
                     report["weighted avg"]["f1-score"])

    return {"classes": classes, "class_counts": np.bincount(y).tolist(), "models": results}


def run_cross_enzyme_transfer(df, X_all):
    """Train on one enzyme, test on another (transfer matrix)."""
    enzymes = ["A3A", "A3B", "A3G", "A3A_A3G", "Neither"]
    transfer_matrix = {}

    for train_enz in enzymes:
        train_mask = df["enzyme"] == train_enz
        train_df = df[train_mask]
        if "is_edited" not in train_df.columns or train_df["is_edited"].nunique() < 2:
            continue
        X_train = X_all[train_mask.values]
        y_train = train_df["is_edited"].values.astype(int)

        clf = XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=SEED, eval_metric="logloss",
        )
        clf.fit(X_train, y_train)

        for test_enz in enzymes:
            test_mask = df["enzyme"] == test_enz
            test_df = df[test_mask]
            if test_df["is_edited"].nunique() < 2:
                continue
            X_test = X_all[test_mask.values]
            y_test = test_df["is_edited"].values.astype(int)
            y_score = clf.predict_proba(X_test)[:, 1]

            auroc = roc_auc_score(y_test, y_score)
            transfer_matrix[f"{train_enz}_to_{test_enz}"] = float(auroc)

    return transfer_matrix


def main():
    t_start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df, X_all = load_all_data()

    # =========================================================================
    # Part 1: Binary classification per enzyme
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("PART 1: Binary Classification (edited vs not) per enzyme")
    logger.info("=" * 60)

    binary_results = {}
    for enzyme in ["A3A", "A3B", "A3G", "A3A_A3G", "Neither", "Unknown"]:
        logger.info("\n--- %s ---", enzyme)
        r = run_binary_per_enzyme(df, X_all, enzyme)
        if r:
            binary_results[enzyme] = r
            for mname, mr in r["models"].items():
                ci = mr["bootstrap_ci"]
                logger.info("  %s: AUROC=%.3f [%.3f–%.3f]", mname, mr["auroc"], ci[0], ci[1])

    # =========================================================================
    # Part 2: 6-class enzyme classification
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("PART 2: Enzyme Classification (6-class, positives only)")
    logger.info("=" * 60)

    enzyme_results = run_enzyme_classification(df, X_all)

    # =========================================================================
    # Part 3: Cross-enzyme transfer
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("PART 3: Cross-Enzyme Transfer Matrix")
    logger.info("=" * 60)

    transfer = run_cross_enzyme_transfer(df, X_all)
    enzymes = ["A3A", "A3B", "A3G", "A3A_A3G", "Neither"]
    logger.info("\nTransfer matrix (train → test AUROC):")
    header = f"{'Train↓ Test→':>15}" + "".join(f"{e:>10}" for e in enzymes)
    logger.info(header)
    for train_enz in enzymes:
        row = f"{train_enz:>15}"
        for test_enz in enzymes:
            key = f"{train_enz}_to_{test_enz}"
            val = transfer.get(key, 0)
            row += f"{val:10.3f}"
        logger.info(row)

    # =========================================================================
    # Save everything
    # =========================================================================
    all_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_total_sites": len(df),
        "n_folds": N_FOLDS,
        "feature_names": FEATURE_NAMES,
        "binary_per_enzyme": binary_results,
        "enzyme_classification": enzyme_results,
        "cross_enzyme_transfer": transfer,
        "elapsed_seconds": time.time() - t_start,
    }

    with open(OUTPUT_DIR / "unified_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Feature importance comparison CSV
    fi_rows = []
    for enzyme, er in binary_results.items():
        gb_fi = er["models"].get("GB_HandFeatures", {}).get("feature_importance", {})
        if gb_fi:
            for feat, imp in sorted(gb_fi.items(), key=lambda x: -x[1]):
                fi_rows.append({"enzyme": enzyme, "feature": feat, "importance": imp})
    if fi_rows:
        pd.DataFrame(fi_rows).to_csv(OUTPUT_DIR / "feature_importance_all_enzymes.csv", index=False)

    # Summary table
    print("\n" + "=" * 80)
    print("UNIFIED MULTI-ENZYME RESULTS SUMMARY")
    print("=" * 80)
    print(f"\n{'Enzyme':>10} {'n_pos':>6} {'n_neg':>6} {'GB_Hand':>10} {'MotifOnly':>10} "
          f"{'StructOnly':>10} {'LoopOnly':>10} {'NoMotif':>10} {'CatBoost':>10} {'LR':>10} {'RF':>10}")
    for enz, r in binary_results.items():
        row = f"{enz:>10} {r['n_pos']:>6} {r['n_neg']:>6}"
        for m in ["GB_HandFeatures", "GB_MotifOnly", "GB_StructOnly", "GB_LoopOnly",
                   "GB_NoMotif", "CatBoost", "LogisticReg", "RandomForest"]:
            auroc = r["models"].get(m, {}).get("auroc", 0)
            row += f"{auroc:10.3f}"
        print(row)

    print(f"\nEnzyme classification (6-class): "
          f"GB accuracy={enzyme_results['models']['GB_HandFeatures']['accuracy']:.3f}, "
          f"macro_f1={enzyme_results['models']['GB_HandFeatures']['macro_f1']:.3f}")

    elapsed = time.time() - t_start
    logger.info("\nTotal time: %.0fs", elapsed)
    logger.info("Results saved to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
