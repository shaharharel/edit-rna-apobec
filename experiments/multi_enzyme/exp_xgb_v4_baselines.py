#!/usr/bin/env python3
"""XGB baselines on v3 + all v4 datasets. Runs on CPU.

Usage:
    conda run -n quris python experiments/multi_enzyme/exp_xgb_v4_baselines.py
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.apobec_feature_extraction import build_hand_features, LOOP_FEATURE_COLS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "multi_enzyme" / "outputs" / "architecture_screen"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "multi_enzyme"
STRUCT_CACHE = DATA_DIR / "structure_cache_multi_enzyme_v3.npz"
LOOP_CSV = DATA_DIR / "loop_position_per_site_v3.csv"
SEQS_JSON = DATA_DIR / "multi_enzyme_sequences_v3_with_negatives.json"

SEED = 42
ENZYMES = ["A3A", "A3B", "A3G", "A3A_A3G", "Neither"]
ENZYME_TO_IDX = {e: i for i, e in enumerate(ENZYMES + ["Unknown"])}

DATASETS = {
    "v3": DATA_DIR / "splits_multi_enzyme_v3_with_negatives.csv",
    "v4_random": DATA_DIR / "splits_v4_random_negatives.csv",
    "v4_hard": DATA_DIR / "splits_v4_hard_negatives.csv",
    "v4_large": DATA_DIR / "splits_v4_large_negatives.csv",
}


def load_features(splits_path):
    """Load splits and build 40d hand features."""
    splits = pd.read_csv(splits_path)
    logger.info(f"  {len(splits)} samples ({splits['is_edited'].sum()} pos)")

    with open(SEQS_JSON) as f:
        seqs = json.load(f)

    loop_df = pd.read_csv(LOOP_CSV).drop_duplicates(subset=["site_id"])
    loop_df["site_id"] = loop_df["site_id"].astype(str)
    loop_df = loop_df.set_index("site_id")

    sc = np.load(str(STRUCT_CACHE), allow_pickle=True)
    struct_map = {str(sid): sc["delta_features"][i] for i, sid in enumerate(sc["site_ids"])}

    site_ids = splits["site_id"].astype(str).tolist()
    hand = build_hand_features(site_ids, seqs, struct_map, loop_df)

    labels_binary = splits["is_edited"].values
    labels_enzyme = splits["enzyme"].values

    return hand, labels_binary, labels_enzyme, site_ids


def run_xgb(X, y_binary, y_enzyme, dataset_name, n_folds=2):
    """Run XGB with per-enzyme evaluation."""
    logger.info(f"\n{'='*60}")
    logger.info(f"XGB on {dataset_name} ({len(X)} samples, {n_folds}-fold CV)")
    logger.info(f"{'='*60}")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    fold_results = []
    for fold_i, (tr_idx, val_idx) in enumerate(skf.split(X, y_binary)):
        t0 = time.time()
        xgb = XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=SEED, eval_metric="logloss",
            use_label_encoder=False, verbosity=0,
        )
        xgb.fit(X[tr_idx], y_binary[tr_idx])
        proba = xgb.predict_proba(X[val_idx])[:, 1]

        overall = roc_auc_score(y_binary[val_idx], proba)

        # Per-enzyme AUROC
        per_enzyme = {}
        for enz in ENZYMES:
            enz_pos = (y_binary[val_idx] == 1) & (y_enzyme[val_idx] == enz)
            enz_neg = (y_binary[val_idx] == 0) & (y_enzyme[val_idx] == enz)
            mask = enz_pos | enz_neg
            if mask.sum() > 10 and enz_pos.sum() > 5:
                try:
                    auc = roc_auc_score(y_binary[val_idx][mask], proba[mask])
                    per_enzyme[enz] = auc
                except ValueError:
                    pass

        fold_results.append({"overall": overall, "per_enzyme": per_enzyme})
        enz_str = ", ".join(f"{e}={per_enzyme.get(e, 0):.3f}" for e in ENZYMES if e in per_enzyme)
        logger.info(f"  Fold {fold_i}: overall={overall:.4f}, {enz_str} ({time.time()-t0:.0f}s)")

    # Aggregate
    mean_overall = np.mean([r["overall"] for r in fold_results])
    mean_per_enzyme = {}
    for enz in ENZYMES:
        vals = [r["per_enzyme"].get(enz, 0) for r in fold_results if enz in r["per_enzyme"]]
        if vals:
            mean_per_enzyme[enz] = {"mean": np.mean(vals), "std": np.std(vals)}

    logger.info(f"\n  MEAN: overall={mean_overall:.4f}")
    for enz, v in mean_per_enzyme.items():
        logger.info(f"    {enz}: {v['mean']:.4f} ± {v['std']:.4f}")

    return {
        "dataset": dataset_name,
        "model": "XGB_40d",
        "n_samples": len(X),
        "n_folds": n_folds,
        "overall_auroc": {"mean": mean_overall, "std": np.std([r["overall"] for r in fold_results])},
        "per_enzyme_auroc": mean_per_enzyme,
        "fold_results": fold_results,
    }


def main():
    t_start = time.time()
    all_results = {}

    for ds_name, ds_path in DATASETS.items():
        if not ds_path.exists():
            logger.warning(f"SKIP {ds_name}: {ds_path} not found")
            continue

        logger.info(f"\nLoading {ds_name}...")
        X, y_binary, y_enzyme, _ = load_features(ds_path)
        result = run_xgb(X, y_binary, y_enzyme, ds_name)
        all_results[ds_name] = result

    # Save
    out_path = OUTPUT_DIR / "xgb_v4_baselines.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nSaved to {out_path}")

    # Summary table
    logger.info(f"\n{'='*80}")
    logger.info("SUMMARY: XGB 40d Baselines")
    logger.info(f"{'='*80}")
    logger.info(f"{'Dataset':<15} | {'Overall':>8} | {'A3A':>8} | {'A3B':>8} | {'A3G':>8} | {'Neither':>8}")
    logger.info("-" * 65)
    for ds_name, r in all_results.items():
        ov = r["overall_auroc"]["mean"]
        vals = [r["per_enzyme_auroc"].get(e, {}).get("mean", 0) for e in ["A3A", "A3B", "A3G", "Neither"]]
        logger.info(f"{ds_name:<15} | {ov:>8.4f} | {vals[0]:>8.4f} | {vals[1]:>8.4f} | {vals[2]:>8.4f} | {vals[3]:>8.4f}")

    logger.info(f"\nTotal time: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
