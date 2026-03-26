#!/usr/bin/env python
"""ClinVar analysis for APOBEC3G editing sites.

Trains a GB classifier on A3G editing sites (n=119 positive, CC-context focus)
and scores ClinVar C>U variants to assess pathogenic enrichment.

IMPORTANT differences from A3A ClinVar analysis:
1. Small training set (n=119) -- include confidence caveat
2. A3G targets CC-context (not TC) -- filter ClinVar to CC-context variants
3. Simpler model to reduce overfitting risk

Pipeline:
  Phase 1: Train GB model with 5-fold CV on A3G sites
  Phase 2: Load pre-computed ClinVar features from A3A experiment
  Phase 3: Score ClinVar variants and compute enrichment

Usage:
    conda run -n quris python experiments/apobec3g/exp_clinvar_a3g.py
"""

import gc
import json
import logging
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu
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

# ClinVar (reuse A3A pre-computed features)
CLINVAR_CSV = PROJECT_ROOT / "data/processed/clinvar_c2u_variants.csv"
A3A_SCORES_CSV = PROJECT_ROOT / "experiments/apobec3a/outputs/clinvar_prediction/clinvar_all_scores.csv"

OUTPUT_DIR = Path(__file__).parent / "outputs" / "clinvar"

SEED = 42
CENTER = 100


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


# ---------------------------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------------------------

def extract_motif_from_seq(seq):
    """Extract 24-dim motif features from a 201nt sequence."""
    seq = seq.upper().replace("T", "U")
    ep = CENTER
    bases = ["A", "C", "G", "U"]
    up = seq[ep - 1] if ep > 0 else "N"
    down = seq[ep + 1] if ep < len(seq) - 1 else "N"
    feat_5p = [1.0 if up + "C" == m else 0.0 for m in ["UC", "CC", "AC", "GC"]]
    feat_3p = [1.0 if "C" + down == m else 0.0 for m in ["CA", "CG", "CU", "CC"]]
    trinuc_up = [0.0] * 8
    for offset, bo in [(-2, 0), (-1, 4)]:
        pos = ep + offset
        if 0 <= pos < len(seq):
            for bi, b in enumerate(bases):
                if seq[pos] == b:
                    trinuc_up[bo + bi] = 1.0
    trinuc_down = [0.0] * 8
    for offset, bo in [(1, 0), (2, 4)]:
        pos = ep + offset
        if 0 <= pos < len(seq):
            for bi, b in enumerate(bases):
                if seq[pos] == b:
                    trinuc_down[bo + bi] = 1.0
    return np.array(feat_5p + feat_3p + trinuc_up + trinuc_down, dtype=np.float32)


def extract_motif_features_batch(sequences, site_ids):
    """Extract motif features for a list of site IDs."""
    features = []
    for sid in site_ids:
        seq = sequences.get(str(sid), "N" * 201)
        features.append(extract_motif_from_seq(seq))
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
    """Build 40-dim hand features (motif 24 + struct_delta 7 + loop 9)."""
    motif = extract_motif_features_batch(sequences, site_ids)
    struct = extract_structure_delta(structure_delta, site_ids)
    loop = extract_loop_features(loop_df, site_ids)
    hand = np.concatenate([motif, struct, loop], axis=1)
    return np.nan_to_num(hand, nan=0.0)


def is_cc_context(seq):
    """Check if sequence has CC context (C at position -1 before center)."""
    if len(seq) < CENTER + 1:
        return False
    seq = seq.upper().replace("T", "U")
    return seq[CENTER - 1] == "C"


def simplify_significance(sig):
    """Simplify ClinVar clinical significance."""
    sig_lower = str(sig).lower()
    if "pathogenic" in sig_lower and "likely" not in sig_lower and "conflicting" not in sig_lower:
        if "benign" not in sig_lower:
            return "Pathogenic"
    if "likely_pathogenic" in sig_lower:
        return "Likely_pathogenic"
    if "benign" in sig_lower and "likely" not in sig_lower and "pathogenic" not in sig_lower:
        if "conflicting" not in sig_lower:
            return "Benign"
    if "likely_benign" in sig_lower:
        return "Likely_benign"
    if "uncertain" in sig_lower:
        return "VUS"
    if "conflicting" in sig_lower:
        return "Conflicting"
    return "Other"


def compute_enrichment(scores, significance, threshold, label=""):
    """Compute pathogenic enrichment at a given score threshold."""
    above = scores >= threshold
    below = ~above

    cats = ["Pathogenic", "Likely_pathogenic"]
    is_path = significance.isin(cats)
    is_benign = significance.isin(["Benign", "Likely_benign"])

    # 2x2 table: [path_above, path_below; benign_above, benign_below]
    a = int((is_path & above).sum())
    b = int((is_path & below).sum())
    c = int((is_benign & above).sum())
    d = int((is_benign & below).sum())

    if min(a, b, c, d) == 0:
        # Add pseudocount
        a += 1; b += 1; c += 1; d += 1

    odds_ratio, p_value = fisher_exact([[a, b], [c, d]], alternative="two-sided")

    return {
        "threshold": float(threshold),
        "n_above": int(above.sum()),
        "n_below": int(below.sum()),
        "path_above": a, "path_below": b,
        "benign_above": c, "benign_below": d,
        "odds_ratio": float(odds_ratio),
        "p_value": float(p_value),
        "label": label,
    }


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_training_data():
    """Load A3G sites with negatives for training."""
    logger.info("Loading training data...")

    if SPLITS_CSV.exists():
        df = pd.read_csv(SPLITS_CSV)
    elif SPLITS_CSV_FALLBACK.exists():
        df = pd.read_csv(SPLITS_CSV_FALLBACK)
    else:
        raise FileNotFoundError("No splits file found")

    df = df[df["enzyme"] == "A3G"].copy()

    if "is_edited" in df.columns and "label" not in df.columns:
        df["label"] = df["is_edited"]
    elif "label" not in df.columns:
        df["label"] = 1

    n_pos = int((df["label"] == 1).sum())
    n_neg = int((df["label"] == 0).sum())
    logger.info("  A3G: %d sites (pos=%d, neg=%d)", len(df), n_pos, n_neg)

    return df


def load_structure_data(needed_sids):
    """Load structure cache and loop features."""
    structure_delta = {}
    if STRUCT_CACHE.exists():
        data = np.load(str(STRUCT_CACHE), allow_pickle=True)
        struct_sids = [str(s) for s in data["site_ids"]]
        for i, sid in enumerate(struct_sids):
            if sid in needed_sids:
                structure_delta[sid] = data["delta_features"][i]
        del data
        gc.collect()

    loop_df = pd.DataFrame()
    if LOOP_POS_CSV.exists():
        loop_df = pd.read_csv(LOOP_POS_CSV)
        loop_df["site_id"] = loop_df["site_id"].astype(str)
        loop_df = loop_df.set_index("site_id")
        loop_df = loop_df[loop_df.index.isin(needed_sids)]

    return structure_delta, loop_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Phase 1: Train model ---
    logger.info("=" * 70)
    logger.info("PHASE 1: TRAIN A3G MODEL")
    logger.info("=" * 70)

    train_df = load_training_data()
    n_pos = int((train_df["label"] == 1).sum())
    n_neg = int((train_df["label"] == 0).sum())

    if n_neg == 0:
        logger.error("No negatives for A3G. Need splits_multi_enzyme_v2_with_negatives.csv")
        # Save minimal results
        results = {
            "error": "No negative sites available for training",
            "confidence_caveat": "Small training set (n=119) limits model reliability.",
        }
        with open(OUTPUT_DIR / "a3g_clinvar_results.json", "w") as f:
            json.dump(results, f, indent=2)
        return

    needed_sids = set(train_df["site_id"].astype(str).tolist())

    # Load sequences
    with open(SEQUENCES_JSON) as f:
        sequences = json.load(f)
    logger.info("  %d sequences loaded", len(sequences))

    # Load structure
    structure_delta, loop_df = load_structure_data(needed_sids)
    logger.info("  %d structure delta, %d loop features", len(structure_delta), len(loop_df))

    # Build features
    site_ids = train_df["site_id"].values
    labels = train_df["label"].values.astype(int)
    hand_40 = build_hand_features(site_ids, sequences, structure_delta, loop_df)
    logger.info("  Hand features shape: %s", hand_40.shape)

    # 5-fold CV
    logger.info("\n5-fold cross-validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(hand_40, labels)):
        fold_seed = SEED + fold_idx

        try:
            from xgboost import XGBClassifier
            gb = XGBClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                subsample=0.8, min_child_weight=5,
                scale_pos_weight=n_neg / max(n_pos, 1),
                random_state=fold_seed, n_jobs=1,
                verbosity=0, eval_metric="logloss",
            )
            gb.fit(hand_40[train_idx], labels[train_idx])
            y_score = gb.predict_proba(hand_40[test_idx])[:, 1]
            metrics = compute_binary_metrics(labels[test_idx], y_score)
            fold_results.append(metrics)
            logger.info("  Fold %d: AUROC=%.4f", fold_idx + 1, metrics["auroc"])
            del gb
        except ImportError:
            logger.error("xgboost not available")
            return

    cv_auroc_mean = float(np.mean([r["auroc"] for r in fold_results]))
    cv_auroc_std = float(np.std([r["auroc"] for r in fold_results]))
    logger.info("  CV AUROC: %.4f +/- %.4f", cv_auroc_mean, cv_auroc_std)

    # Train final model on all data
    logger.info("\nTraining final model on all %d sites...", len(labels))
    from xgboost import XGBClassifier
    gb_final = XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        subsample=0.8, min_child_weight=5,
        scale_pos_weight=n_neg / max(n_pos, 1),
        random_state=SEED, n_jobs=1,
        verbosity=0, eval_metric="logloss",
    )
    gb_final.fit(hand_40, labels)

    # --- Phase 2: Load ClinVar data ---
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 2: SCORE CLINVAR VARIANTS")
    logger.info("=" * 70)

    # Check if pre-computed A3A scores exist (has features we can reuse)
    if not CLINVAR_CSV.exists():
        logger.error("ClinVar data not found: %s", CLINVAR_CSV)
        results = {
            "training_n_positive": n_pos,
            "training_n_negative": n_neg,
            "cv_auroc_mean": cv_auroc_mean,
            "cv_auroc_std": cv_auroc_std,
            "error": "ClinVar data file not found",
            "confidence_caveat": "Small training set (n=119) limits model reliability.",
        }
        with open(OUTPUT_DIR / "a3g_clinvar_results.json", "w") as f:
            json.dump(results, f, indent=2)
        return

    clinvar_df = pd.read_csv(CLINVAR_CSV, low_memory=False)
    logger.info("  Loaded %d ClinVar C>U variants", len(clinvar_df))

    # Add simplified significance
    clinvar_df["significance_simple"] = clinvar_df["clinical_significance"].apply(simplify_significance)

    # Check for sequence column
    if "sequence" not in clinvar_df.columns:
        logger.error("ClinVar data missing 'sequence' column")
        return

    # Filter to valid 201nt sequences
    valid_mask = clinvar_df["sequence"].str.len() == 201
    clinvar_valid = clinvar_df[valid_mask].copy()
    logger.info("  Valid 201nt sequences: %d", len(clinvar_valid))

    # Identify CC-context variants
    clinvar_valid["is_cc_context"] = clinvar_valid["sequence"].apply(is_cc_context)
    n_cc = int(clinvar_valid["is_cc_context"].sum())
    logger.info("  CC-context variants: %d (%.1f%%)", n_cc, 100 * n_cc / len(clinvar_valid))

    # --- Phase 3: Score and analyze ---
    # We need to compute features for ClinVar variants
    # For efficiency, use pre-computed A3A features if available
    if A3A_SCORES_CSV.exists():
        logger.info("Loading pre-computed A3A ClinVar scores for reference...")
        a3a_scores = pd.read_csv(A3A_SCORES_CSV, low_memory=False)
        logger.info("  A3A scores loaded: %d variants", len(a3a_scores))

    # Compute A3G-specific features for ClinVar
    # Use motif-only features (fast, no ViennaRNA needed) for initial scoring
    logger.info("\nComputing A3G motif-based scores for ClinVar variants...")

    # Train motif-only model
    motif_train = extract_motif_features_batch(sequences, site_ids)
    gb_motif = XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        scale_pos_weight=n_neg / max(n_pos, 1),
        random_state=SEED, n_jobs=1, verbosity=0,
    )
    gb_motif.fit(motif_train, labels)

    # Score ClinVar variants with motif-only model
    batch_size = 50000
    n_variants = len(clinvar_valid)
    all_scores = np.zeros(n_variants, dtype=np.float32)

    for start in range(0, n_variants, batch_size):
        end = min(start + batch_size, n_variants)
        batch_seqs = clinvar_valid.iloc[start:end]["sequence"].values
        batch_motif = np.array([extract_motif_from_seq(s) for s in batch_seqs], dtype=np.float32)
        all_scores[start:end] = gb_motif.predict_proba(batch_motif)[:, 1]
        logger.info("  Scored %d/%d variants...", end, n_variants)

    clinvar_valid["a3g_score"] = all_scores

    # --- Enrichment analysis ---
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 3: ENRICHMENT ANALYSIS")
    logger.info("=" * 70)

    sig = clinvar_valid["significance_simple"]
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

    # Enrichment on ALL C>U variants
    enrichment_all = {}
    for thresh in thresholds:
        result = compute_enrichment(all_scores, sig, thresh, label=f"all_c2u_t{thresh}")
        enrichment_all[f"threshold_{thresh}"] = result
        logger.info("  ALL C>U t=%.1f: OR=%.3f, p=%.2e, path_above=%d, benign_above=%d",
                    thresh, result["odds_ratio"], result["p_value"],
                    result["path_above"], result["benign_above"])

    # Enrichment on CC-context only
    cc_mask = clinvar_valid["is_cc_context"].values
    cc_scores = all_scores[cc_mask]
    cc_sig = sig[cc_mask]

    enrichment_cc = {}
    if len(cc_scores) >= 100:
        for thresh in thresholds:
            result = compute_enrichment(cc_scores, cc_sig, thresh, label=f"cc_only_t{thresh}")
            enrichment_cc[f"threshold_{thresh}"] = result
            logger.info("  CC-only t=%.1f: OR=%.3f, p=%.2e, path_above=%d, benign_above=%d",
                        thresh, result["odds_ratio"], result["p_value"],
                        result["path_above"], result["benign_above"])
    else:
        enrichment_cc = {"note": f"insufficient CC-context variants (n={len(cc_scores)})"}

    # Save scored variants
    scored_path = OUTPUT_DIR / "a3g_clinvar_scored.csv"
    clinvar_valid[["site_id", "significance_simple", "is_cc_context", "a3g_score"]].to_csv(
        scored_path, index=False)
    logger.info("  Scored variants saved to %s", scored_path.name)

    # --- Save results ---
    results = {
        "training_n_positive": n_pos,
        "training_n_negative": n_neg,
        "cv_auroc_mean": cv_auroc_mean,
        "cv_auroc_std": cv_auroc_std,
        "cv_fold_aurocs": [r["auroc"] for r in fold_results],
        "confidence_caveat": (
            "Small training set (n=119) limits model reliability. "
            "Interpret ClinVar enrichment with caution."
        ),
        "model_type": "GB_MotifOnly (24-dim, no ViennaRNA for ClinVar)",
        "n_clinvar_total": len(clinvar_valid),
        "n_cc_context_variants": n_cc,
        "enrichment_all_c2u": enrichment_all,
        "enrichment_cc_context_only": enrichment_cc,
        "significance_breakdown": {
            str(k): int(v) for k, v in clinvar_valid["significance_simple"].value_counts().items()
        },
    }

    out_path = OUTPUT_DIR / "a3g_clinvar_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    elapsed = time.time() - t_start
    logger.info("\nResults saved to %s (%.1f sec)", out_path, elapsed)


if __name__ == "__main__":
    main()
