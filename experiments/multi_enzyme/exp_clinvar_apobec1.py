#!/usr/bin/env python
"""ClinVar C>U variant scoring with APOBEC1 ("Neither")-trained GB model.

Trains XGBClassifier on "Neither" positives + negatives, scores ClinVar C>U variants,
and computes pathogenic enrichment at multiple thresholds.

Requires:
  - splits_multi_enzyme_v3_with_negatives.csv ("Neither" positives + negatives)
  - clinvar_features_cache.npz (pre-computed ClinVar features)
  - Pre-computed structure and loop data

Usage:
    conda run -n quris python experiments/multi_enzyme/exp_clinvar_apobec1.py
"""

import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
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
_ME_DIR = PROJECT_ROOT / "data" / "processed" / "multi_enzyme"
SPLITS_CSV = _ME_DIR / "splits_multi_enzyme_v3_with_negatives.csv"
SEQUENCES_JSON = _ME_DIR / "multi_enzyme_sequences_v3_with_negatives.json"
STRUCT_CACHE = _ME_DIR / "structure_cache_multi_enzyme_v3.npz"
LOOP_POS_CSV = _ME_DIR / "loop_position_per_site_v3.csv"

CLINVAR_CSV = PROJECT_ROOT / "data" / "processed" / "clinvar_c2u_variants.csv"
CLINVAR_FEATURES_CACHE = PROJECT_ROOT / "data" / "processed" / "clinvar_features_cache.npz"

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "multi_enzyme" / "outputs" / "apobec1_integration"

SEED = 42
CENTER = 100
ENZYME_LABEL = "Neither"  # maps to APOBEC1 hypothesis


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


# ---------------------------------------------------------------------------
# Training data
# ---------------------------------------------------------------------------

def load_training_data():
    """Load Neither/APOBEC1 positives + negatives for training."""
    if not SPLITS_CSV.exists():
        logger.error("Splits file not found: %s", SPLITS_CSV)
        sys.exit(1)

    df = pd.read_csv(SPLITS_CSV)
    df = df[df["enzyme"] == ENZYME_LABEL].copy()

    if "is_edited" in df.columns:
        df["label"] = df["is_edited"].astype(int)
    elif "label" in df.columns:
        pass
    else:
        logger.error("No label column found")
        sys.exit(1)

    n_pos = (df["label"] == 1).sum()
    n_neg = (df["label"] == 0).sum()
    logger.info("%s training data: %d sites (pos=%d, neg=%d)", ENZYME_LABEL, len(df), n_pos, n_neg)

    if n_neg == 0:
        logger.error("No negatives found")
        sys.exit(1)

    return df


def build_training_features(df, sequences, structure_delta, loop_df):
    """Build 46-dim features: motif(24) + struct_delta(7) + loop(9) + baseline_struct(6)."""
    loop_cols = [
        "is_unpaired", "loop_size", "dist_to_junction", "dist_to_apex",
        "relative_loop_position", "left_stem_length", "right_stem_length",
        "max_adjacent_stem_length", "local_unpaired_fraction",
    ]

    features = []
    for sid in df["site_id"].astype(str):
        seq = sequences.get(sid, "N" * 201)
        motif = extract_motif_from_seq(seq)

        sd = structure_delta.get(sid, np.zeros(7, dtype=np.float32))
        if not isinstance(sd, np.ndarray):
            sd = np.array(sd, dtype=np.float32)

        if sid in loop_df.index:
            row = loop_df.loc[sid]
            loop = []
            for c in loop_cols:
                v = row[c] if c in row.index else 0.0
                loop.append(float(v) if pd.notna(v) else 0.0)
            loop = np.array(loop, dtype=np.float32)
        else:
            loop = np.zeros(len(loop_cols), dtype=np.float32)

        baseline = np.zeros(6, dtype=np.float32)
        feat = np.concatenate([motif, sd, loop, baseline])
        features.append(feat)

    result = np.array(features, dtype=np.float32)
    return np.nan_to_num(result, nan=0.0)


# ---------------------------------------------------------------------------
# ClinVar
# ---------------------------------------------------------------------------

def load_clinvar():
    """Load ClinVar C>U variants."""
    if not CLINVAR_CSV.exists():
        logger.error("ClinVar data not found: %s", CLINVAR_CSV)
        return None

    df = pd.read_csv(CLINVAR_CSV, low_memory=False)
    logger.info("Loaded %d ClinVar C>U variants", len(df))

    def simplify_significance(sig):
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

    df["significance_simple"] = df["clinical_significance"].apply(simplify_significance)
    return df


def compute_enrichment(scores, labels, threshold):
    """Compute pathogenic enrichment at a given threshold."""
    path_mask = labels.isin(["Pathogenic", "Likely_pathogenic"])
    benign_mask = labels.isin(["Benign", "Likely_benign"])

    relevant = path_mask | benign_mask
    if relevant.sum() < 10:
        return None

    scores_rel = scores[relevant]
    path_rel = path_mask[relevant]

    above = scores_rel >= threshold
    below = scores_rel < threshold

    a = (above & path_rel).sum()
    b = (above & ~path_rel).sum()
    c = (below & path_rel).sum()
    d = (below & ~path_rel).sum()

    if min(a, b, c, d) == 0:
        table = np.array([[a + 0.5, b + 0.5], [c + 0.5, d + 0.5]])
    else:
        table = np.array([[a, b], [c, d]])

    odds_ratio, p_value = fisher_exact([[a, b], [c, d]], alternative="two-sided")
    return {
        "threshold": float(threshold),
        "path_above": int(a),
        "benign_above": int(b),
        "path_below": int(c),
        "benign_below": int(d),
        "odds_ratio": float(odds_ratio),
        "p_value": float(p_value),
        "n_above_threshold": int(above.sum()),
        "frac_above_threshold": float(above.sum() / len(scores_rel)),
    }


def bayesian_recalibrate(p_model, pi_model=0.5, pi_real=0.019):
    """Recalibrate model probability from training prior to real prior."""
    if p_model <= 0 or p_model >= 1:
        return p_model
    lr = p_model / (1 - p_model) * ((1 - pi_model) / pi_model)
    p_real = lr * pi_real / (lr * pi_real + (1 - pi_real))
    return float(p_real)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Phase 1: Train APOBEC1 model ---
    logger.info("=" * 60)
    logger.info("Phase 1: Training %s (APOBEC1) classifier", ENZYME_LABEL)

    train_df = load_training_data()

    with open(SEQUENCES_JSON) as f:
        sequences = json.load(f)

    structure_delta = {}
    if STRUCT_CACHE.exists():
        data = np.load(STRUCT_CACHE, allow_pickle=True)
        sids = data["site_ids"]
        feats = data["delta_features"]
        structure_delta = {str(sid): feats[i] for i, sid in enumerate(sids)}
        del data
        gc.collect()

    loop_df = pd.DataFrame()
    if LOOP_POS_CSV.exists():
        loop_df = pd.read_csv(LOOP_POS_CSV)
        loop_df["site_id"] = loop_df["site_id"].astype(str)
        loop_df = loop_df.set_index("site_id")

    X_train = build_training_features(train_df, sequences, structure_delta, loop_df)
    y_train = train_df["label"].values
    logger.info("Training features shape: %s", X_train.shape)

    # 5-fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_aurocs = []
    cv_auprcs = []

    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        clf = XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=SEED + fold_idx, eval_metric="logloss",
            use_label_encoder=False,
        )
        clf.fit(X_train[tr_idx], y_train[tr_idx], verbose=False)
        y_score = clf.predict_proba(X_train[val_idx])[:, 1]
        auroc = roc_auc_score(y_train[val_idx], y_score)
        auprc = average_precision_score(y_train[val_idx], y_score)
        cv_aurocs.append(float(auroc))
        cv_auprcs.append(float(auprc))
        logger.info("  Fold %d: AUROC=%.4f, AUPRC=%.4f", fold_idx, auroc, auprc)

    logger.info("  CV AUROC: %.4f +/- %.4f", np.mean(cv_aurocs), np.std(cv_aurocs))

    # Train final model on all data
    final_clf = XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=SEED, eval_metric="logloss", use_label_encoder=False,
    )
    final_clf.fit(X_train, y_train, verbose=False)
    logger.info("Final model trained on %d sites", len(X_train))

    # Feature importance from final model
    importance = final_clf.feature_importances_
    feat_names = (
        [f"motif_{i}" for i in range(24)]
        + [f"struct_delta_{i}" for i in range(7)]
        + ["is_unpaired", "loop_size", "dist_to_junction", "dist_to_apex",
           "relative_loop_position", "left_stem_length", "right_stem_length",
           "max_adjacent_stem_length", "local_unpaired_fraction"]
        + [f"baseline_struct_{i}" for i in range(6)]
    )
    feat_imp = sorted(zip(feat_names, importance), key=lambda x: -x[1])
    logger.info("Top 10 features:")
    for name, imp in feat_imp[:10]:
        logger.info("  %s: %.4f", name, imp)

    # --- Phase 2: Score ClinVar ---
    logger.info("=" * 60)
    logger.info("Phase 2: Scoring ClinVar variants")

    clinvar_df = load_clinvar()
    if clinvar_df is None:
        results = {
            "enzyme": "APOBEC1_Neither",
            "cv_aurocs": cv_aurocs,
            "cv_auprcs": cv_auprcs,
            "mean_cv_auroc": float(np.mean(cv_aurocs)),
            "clinvar_scored": False,
            "error": "ClinVar data not found",
        }
        with open(OUTPUT_DIR / "apobec1_clinvar_results.json", "w") as f:
            json.dump(results, f, indent=2)
        return

    clinvar_features_loaded = False
    clinvar_hand_46 = None
    clinvar_site_ids = None

    if CLINVAR_FEATURES_CACHE.exists():
        logger.info("Loading pre-computed ClinVar features from %s...", CLINVAR_FEATURES_CACHE.name)
        cache_data = np.load(CLINVAR_FEATURES_CACHE, allow_pickle=True)
        clinvar_site_ids = list(cache_data["site_ids"])
        clinvar_hand_46 = cache_data["hand_46"]
        clinvar_features_loaded = True
        logger.info("  Loaded features for %d ClinVar variants (%d dims)",
                    len(clinvar_site_ids), clinvar_hand_46.shape[1])

    if not clinvar_features_loaded:
        logger.error("No pre-computed ClinVar features found at %s", CLINVAR_FEATURES_CACHE)
        logger.error("Run A3A ClinVar pipeline first to build the feature cache.")
        sys.exit(1)

    # Score
    logger.info("Scoring %d ClinVar variants with APOBEC1 model...", len(clinvar_hand_46))
    if clinvar_hand_46.shape[1] != X_train.shape[1]:
        logger.warning("Feature dimension mismatch: ClinVar=%d, training=%d",
                       clinvar_hand_46.shape[1], X_train.shape[1])
        n_feat = X_train.shape[1]
        if clinvar_hand_46.shape[1] > n_feat:
            clinvar_hand_46 = clinvar_hand_46[:, :n_feat]
        else:
            pad = np.zeros((clinvar_hand_46.shape[0], n_feat - clinvar_hand_46.shape[1]),
                           dtype=np.float32)
            clinvar_hand_46 = np.hstack([clinvar_hand_46, pad])

    apobec1_scores = final_clf.predict_proba(clinvar_hand_46)[:, 1]

    if len(clinvar_site_ids) == len(clinvar_df):
        clinvar_df["apobec1_score"] = apobec1_scores
    else:
        logger.warning("ClinVar size mismatch: features=%d, df=%d",
                       len(clinvar_site_ids), len(clinvar_df))
        clinvar_df["apobec1_score"] = np.nan
        for i in range(min(len(clinvar_site_ids), len(clinvar_df))):
            clinvar_df.iloc[i, clinvar_df.columns.get_loc("apobec1_score")] = float(apobec1_scores[i])

    scored_mask = clinvar_df["apobec1_score"].notna()
    logger.info("Scored %d / %d ClinVar variants", scored_mask.sum(), len(clinvar_df))

    # --- Phase 3: Enrichment ---
    logger.info("=" * 60)
    logger.info("Phase 3: Pathogenic enrichment analysis")

    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    enrichment_results = []
    scored_df = clinvar_df[scored_mask].copy()

    for thresh in thresholds:
        enr = compute_enrichment(
            scored_df["apobec1_score"].values,
            scored_df["significance_simple"],
            thresh,
        )
        if enr:
            enrichment_results.append(enr)
            logger.info("  Threshold=%.1f: OR=%.3f, p=%.2e, n_above=%d (%.1f%%)",
                        thresh, enr["odds_ratio"], enr["p_value"],
                        enr["n_above_threshold"], enr["frac_above_threshold"] * 100)

    # Bayesian calibration
    calibrated_results = []
    for thresh in thresholds:
        cal_thresh = bayesian_recalibrate(thresh, pi_model=0.5, pi_real=0.019)
        enr = compute_enrichment(
            scored_df["apobec1_score"].values,
            scored_df["significance_simple"],
            cal_thresh,
        )
        if enr:
            enr["original_threshold"] = float(thresh)
            enr["calibrated_threshold"] = float(cal_thresh)
            calibrated_results.append(enr)
            logger.info("  Calibrated t=%.4f (from %.1f): OR=%.3f, p=%.2e",
                        cal_thresh, thresh, enr["odds_ratio"], enr["p_value"])

    # Save results
    results = {
        "enzyme": "APOBEC1_Neither",
        "n_train_pos": int((y_train == 1).sum()),
        "n_train_neg": int((y_train == 0).sum()),
        "cv_aurocs": cv_aurocs,
        "cv_auprcs": cv_auprcs,
        "mean_cv_auroc": float(np.mean(cv_aurocs)),
        "std_cv_auroc": float(np.std(cv_aurocs)),
        "mean_cv_auprc": float(np.mean(cv_auprcs)),
        "clinvar_scored": True,
        "n_clinvar_total": int(len(clinvar_df)),
        "n_clinvar_scored": int(scored_mask.sum()),
        "enrichment_raw": enrichment_results,
        "enrichment_calibrated": calibrated_results,
        "significance_distribution": {
            k: int(v) for k, v in
            scored_df["significance_simple"].value_counts().items()
        },
        "feature_importance": {name: float(imp) for name, imp in feat_imp},
    }

    out_path = OUTPUT_DIR / "apobec1_clinvar_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", out_path)

    # Save scored variants
    scored_csv = OUTPUT_DIR / "clinvar_apobec1_scores.csv"
    scored_df[["apobec1_score", "significance_simple"]].to_csv(scored_csv, index=False)
    logger.info("Scores saved to %s", scored_csv)

    elapsed = time.time() - t_start
    logger.info("Total time: %.1f seconds", elapsed)


if __name__ == "__main__":
    main()
