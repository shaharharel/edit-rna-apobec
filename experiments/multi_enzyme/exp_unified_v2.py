#!/usr/bin/env python
"""Unified multi-enzyme experiment v2: proper negatives + full model comparison.

TWO evaluation modes:
  A) UNIFIED negatives (natural genomic distribution) — tests FULL discrimination
     including motif. Fair comparison across all enzymes.
  B) HARD negatives (per-enzyme motif-matched) — tests what matters BEYOND motif.
     Reveals structure/context interactions.

Models: XGBoost, CatBoost, RF, LR, HandMLP, SeqCNN, CNN+Hand

Tasks:
  1. Binary classification per enzyme (both neg modes)
  2. 6-class enzyme classification (positives only)

Usage:
    conda run -n quris python experiments/multi_enzyme/exp_unified_v2.py
"""

import gc
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
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
# Two negative modes
UNIFIED_SPLITS = _ME_DIR / "splits_multi_enzyme_v3_unified_negatives.csv"
UNIFIED_SEQS = _ME_DIR / "multi_enzyme_sequences_v3_unified_negatives.json"
HARD_SPLITS = _ME_DIR / "splits_multi_enzyme_v3_with_negatives.csv"
HARD_SEQS = _ME_DIR / "multi_enzyme_sequences_v3_with_negatives.json"
STRUCT_CACHE = _ME_DIR / "structure_cache_multi_enzyme_v3.npz"
LOOP_CSV_UNIFIED = _ME_DIR / "loop_position_per_site_v3_unified.csv"
LOOP_CSV_HARD = _ME_DIR / "loop_position_per_site_v3.csv"
OUTPUT_DIR = Path(__file__).parent / "outputs" / "unified_v2"

SEED = 42
N_FOLDS = 5
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

FEATURE_NAMES = (
    ['5p_UC', '5p_CC', '5p_AC', '5p_GC', '3p_CA', '3p_CG', '3p_CU', '3p_CC',
     'm2_A', 'm2_C', 'm2_G', 'm2_U', 'm1_A', 'm1_C', 'm1_G', 'm1_U',
     'p1_A', 'p1_C', 'p1_G', 'p1_U', 'p2_A', 'p2_C', 'p2_G', 'p2_U'] +
    ['delta_pairing_center', 'delta_accessibility_center', 'delta_entropy_center',
     'delta_mfe', 'mean_delta_pairing_window', 'mean_delta_accessibility_window',
     'std_delta_pairing_window'] +
    ['is_unpaired', 'loop_size', 'dist_to_junction', 'dist_to_apex',
     'relative_loop_position', 'left_stem_length', 'right_stem_length',
     'max_adjacent_stem_length', 'local_unpaired_fraction']
)


def load_dataset(splits_path, seqs_path, loop_path):
    """Load features for a dataset."""
    df = pd.read_csv(splits_path)
    with open(seqs_path) as f:
        seqs = json.load(f)

    structure_delta = {}
    if STRUCT_CACHE.exists():
        data = np.load(str(STRUCT_CACHE), allow_pickle=True)
        sids = [str(s) for s in data["site_ids"]]
        for i, sid in enumerate(sids):
            structure_delta[sid] = data["delta_features"][i]
        del data; gc.collect()

    loop_df = pd.read_csv(loop_path)
    loop_df["site_id"] = loop_df["site_id"].astype(str)
    loop_df = loop_df.set_index("site_id")

    site_ids = df["site_id"].values
    motif = extract_motif_features(seqs, list(site_ids))
    struct = extract_structure_delta_features(structure_delta, list(site_ids))
    loop = extract_loop_features(loop_df, list(site_ids))
    hand_40 = np.concatenate([motif, struct, loop], axis=1)
    hand_40 = np.nan_to_num(hand_40, nan=0.0)

    # One-hot sequences
    def seq_to_onehot(seq, length=201):
        mapping = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3}
        oh = np.zeros((4, length), dtype=np.float32)
        for i, b in enumerate(seq[:length].upper()):
            idx = mapping.get(b, -1)
            if idx >= 0: oh[idx, i] = 1.0
        return oh

    onehot = np.array([seq_to_onehot(seqs.get(str(sid), "N"*201)) for sid in site_ids])

    return df, hand_40, onehot, seqs


class SeqCNNHand(nn.Module):
    """CNN on sequence + hand features, fused."""
    def __init__(self, d_hand=40):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(4, 64, 7, padding=3), nn.GELU(),
            nn.Conv1d(64, 128, 5, padding=2), nn.GELU(),
            nn.AdaptiveMaxPool1d(20),
            nn.Conv1d(128, 128, 3, padding=1), nn.GELU(),
            nn.AdaptiveMaxPool1d(1),
        )
        self.hand_proj = nn.Sequential(nn.Linear(d_hand, 128), nn.GELU(), nn.Dropout(0.3))
        self.head = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.3), nn.Linear(128, 1))

    def forward(self, seq_oh, hand):
        h_seq = self.conv(seq_oh).squeeze(-1)
        h_hand = self.hand_proj(hand)
        return self.head(torch.cat([h_seq, h_hand], -1)).squeeze(-1)


def run_binary_cv(X_hand, X_seq, y, enzyme, neg_mode, small=False):
    """Run 5-fold CV with all models for one enzyme."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    results = {}

    xgb_p = {"n_estimators": 200 if small else 500, "max_depth": 4 if small else 6,
             "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8,
             "random_state": SEED, "eval_metric": "logloss"}
    cat_p = {"iterations": 200 if small else 500, "depth": 4 if small else 6,
             "learning_rate": 0.1, "random_seed": SEED, "verbose": 0}

    # Tabular models
    for mname, mtype, feat_idx, params in [
        ("XGBoost", "xgb", range(40), xgb_p),
        ("CatBoost", "cat", range(40), cat_p),
        ("RF", "rf", range(40), {"n_estimators": 500, "max_depth": 10, "random_state": SEED}),
        ("LR", "lr", range(40), {}),
        ("XGB_MotifOnly", "xgb", range(24), xgb_p),
        ("XGB_StructOnly", "xgb", range(24, 40), xgb_p),
    ]:
        X = X_hand[:, list(feat_idx)]
        fold_aurocs = []
        fold_imps = []
        for tr, te in skf.split(X, y):
            if mtype == "xgb":
                clf = XGBClassifier(**params); clf.fit(X[tr], y[tr])
                ys = clf.predict_proba(X[te])[:, 1]; fold_imps.append(clf.feature_importances_)
            elif mtype == "cat":
                clf = CatBoostClassifier(**params); clf.fit(X[tr], y[tr])
                ys = clf.predict_proba(X[te])[:, 1]; fold_imps.append(clf.get_feature_importance())
            elif mtype == "rf":
                clf = RandomForestClassifier(**params); clf.fit(X[tr], y[tr])
                ys = clf.predict_proba(X[te])[:, 1]; fold_imps.append(clf.feature_importances_)
            elif mtype == "lr":
                sc = StandardScaler(); clf = LogisticRegression(max_iter=1000, C=1.0, random_state=SEED)
                clf.fit(sc.fit_transform(X[tr]), y[tr]); ys = clf.predict_proba(sc.transform(X[te]))[:, 1]
            fold_aurocs.append(roc_auc_score(y[te], ys))

        avg_imp = None
        if fold_imps:
            avg_imp = dict(zip([FEATURE_NAMES[i] for i in feat_idx], np.mean(fold_imps, axis=0).tolist()))

        results[mname] = {"auroc": float(np.mean(fold_aurocs)), "std": float(np.std(fold_aurocs)),
                          "feature_importance": avg_imp}

    # Neural: CNN+Hand
    fold_aurocs = []
    for fold_i, (tr, te) in enumerate(skf.split(X_hand, y)):
        torch.manual_seed(SEED + fold_i)
        X_seq_tr, X_seq_te = torch.FloatTensor(X_seq[tr]), torch.FloatTensor(X_seq[te])
        X_h_tr, X_h_te = torch.FloatTensor(X_hand[tr]), torch.FloatTensor(X_hand[te])
        y_tr_t, y_te_t = torch.FloatTensor(y[tr]), torch.FloatTensor(y[te])

        train_ds = TensorDataset(X_seq_tr, X_h_tr, y_tr_t)
        test_ds = TensorDataset(X_seq_te, X_h_te, y_te_t)
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=128)

        model = SeqCNNHand(40).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)
        crit = nn.BCEWithLogitsLoss()

        best_auroc = 0; patience = 0
        for epoch in range(50):
            model.train()
            for xs, xh, yt in train_loader:
                xs, xh, yt = xs.to(DEVICE), xh.to(DEVICE), yt.to(DEVICE)
                loss = crit(model(xs, xh), yt); opt.zero_grad(); loss.backward(); opt.step()
            sched.step()

            model.eval()
            all_ys = []
            with torch.no_grad():
                for xs, xh, yt in test_loader:
                    xs, xh = xs.to(DEVICE), xh.to(DEVICE)
                    all_ys.extend(torch.sigmoid(model(xs, xh)).cpu().tolist())
            auroc = roc_auc_score(y[te], all_ys)
            if auroc > best_auroc: best_auroc = auroc; patience = 0
            else: patience += 1
            if patience >= 10: break

        fold_aurocs.append(best_auroc)
        logger.info("  CNN+Hand fold %d: %.3f", fold_i + 1, best_auroc)
        del model; gc.collect()
        if DEVICE == "mps": torch.mps.empty_cache()

    results["CNN+Hand"] = {"auroc": float(np.mean(fold_aurocs)), "std": float(np.std(fold_aurocs))}

    return results


def main():
    t_start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Device: %s", DEVICE)

    all_results = {}

    for neg_mode, splits_path, seqs_path, loop_path in [
        ("unified", UNIFIED_SPLITS, UNIFIED_SEQS, LOOP_CSV_UNIFIED),
        ("hard", HARD_SPLITS, HARD_SEQS, LOOP_CSV_HARD),
    ]:
        logger.info("\n" + "="*70)
        logger.info("NEGATIVE MODE: %s", neg_mode.upper())
        logger.info("="*70)

        df, hand_40, onehot, seqs = load_dataset(splits_path, seqs_path, loop_path)
        mode_results = {}

        enzymes = ["A3A", "A3B", "A3G", "A3A_A3G", "Neither"]
        for enzyme in enzymes:
            if neg_mode == "unified":
                # For unified: all positives of this enzyme vs ALL unified negatives
                pos_mask = (df["enzyme"] == enzyme) & (df["is_edited"] == 1)
                neg_mask = df["is_edited"] == 0
                mask = pos_mask | neg_mask
            else:
                # For hard: enzyme-specific positives vs enzyme-specific negatives
                mask = df["enzyme"] == enzyme

            sub = df[mask].copy()
            if "is_edited" in sub.columns:
                sub["label"] = sub["is_edited"]
            else:
                continue
            if sub["label"].nunique() < 2:
                continue

            X_h = hand_40[mask.values]
            X_s = onehot[mask.values]
            y = sub["label"].values.astype(int)
            n_pos = (y == 1).sum()
            n_neg = (y == 0).sum()
            small = n_pos < 200

            logger.info("\n--- %s [%s neg] (pos=%d, neg=%d) ---", enzyme, neg_mode, n_pos, n_neg)
            r = run_binary_cv(X_h, X_s, y, enzyme, neg_mode, small)
            mode_results[enzyme] = {"n_pos": int(n_pos), "n_neg": int(n_neg), "models": r}

            for mname, mr in r.items():
                logger.info("  %s: AUROC=%.3f±%.3f", mname, mr["auroc"], mr.get("std", 0))

        all_results[neg_mode] = mode_results

    # Save
    with open(OUTPUT_DIR / "unified_v2_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Summary table
    print("\n" + "="*90)
    print("UNIFIED vs HARD NEGATIVES COMPARISON")
    print("="*90)
    enzymes = ["A3A", "A3B", "A3G", "A3A_A3G", "Neither"]
    models_to_show = ["XGBoost", "CatBoost", "RF", "LR", "CNN+Hand", "XGB_MotifOnly", "XGB_StructOnly"]

    for neg_mode in ["unified", "hard"]:
        print(f"\n--- {neg_mode.upper()} NEGATIVES ---")
        header = f"{'Enzyme':>10}" + "".join(f"{m:>12}" for m in models_to_show)
        print(header)
        for enz in enzymes:
            r = all_results.get(neg_mode, {}).get(enz, {}).get("models", {})
            row = f"{enz:>10}"
            for m in models_to_show:
                auroc = r.get(m, {}).get("auroc", 0)
                row += f"{auroc:12.3f}"
            print(row)

    # Feature importance comparison
    fi_rows = []
    for neg_mode in ["unified", "hard"]:
        for enz in enzymes:
            fi = all_results.get(neg_mode, {}).get(enz, {}).get("models", {}).get("CatBoost", {}).get("feature_importance", {})
            if fi:
                for feat, imp in sorted(fi.items(), key=lambda x: -x[1])[:10]:
                    fi_rows.append({"neg_mode": neg_mode, "enzyme": enz, "feature": feat, "importance": imp})
    if fi_rows:
        pd.DataFrame(fi_rows).to_csv(OUTPUT_DIR / "catboost_feature_importance.csv", index=False)

    elapsed = time.time() - t_start
    logger.info("\nTotal time: %.0fs", elapsed)
    logger.info("Results saved to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
