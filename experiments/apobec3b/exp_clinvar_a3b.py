#!/usr/bin/env python
"""ClinVar C>U variant scoring with A3B-trained GB model.

Trains XGBClassifier on A3B positives + negatives, scores ClinVar C>U variants,
and computes pathogenic enrichment at multiple thresholds.

Requires:
  - splits_multi_enzyme_v2_with_negatives.csv (A3B positives + negatives)
  - clinvar_c2u_variants.csv (pre-processed ClinVar variants)
  - Pre-computed feature caches OR ViennaRNA for on-the-fly computation

Usage:
    conda run -n quris python experiments/apobec3b/exp_clinvar_a3b.py
"""

import gc
import json
import logging
import os
import sys
import time
from multiprocessing import Pool
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
SPLITS_CSV = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "splits_multi_enzyme_v2_with_negatives.csv"
SPLITS_CSV_FALLBACK = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "splits_multi_enzyme_v2.csv"
SEQUENCES_JSON = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "multi_enzyme_sequences_v2_with_negatives.json"
STRUCT_CACHE = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "structure_cache_multi_enzyme_v2.npz"
LOOP_POS_CSV = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "loop_position_per_site_v2.csv"

# ClinVar data (shared with A3A experiment)
CLINVAR_CSV = PROJECT_ROOT / "data" / "processed" / "clinvar_c2u_variants.csv"
# Use A3A's pre-computed ClinVar scores for sequences + features if available
CLINVAR_A3A_SCORES = PROJECT_ROOT / "experiments" / "apobec3a" / "outputs" / "clinvar_prediction" / "clinvar_all_scores.csv"
CLINVAR_FEATURES_CACHE = PROJECT_ROOT / "data" / "processed" / "clinvar_features_cache.npz"

OUTPUT_DIR = Path(__file__).parent / "outputs" / "clinvar"

SEED = 42
CENTER = 100
N_WORKERS = 8
CHUNK_SIZE = 50_000


# ---------------------------------------------------------------------------
# Feature Extraction (same as classification experiment)
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


def compute_vienna_features(seq):
    """Compute ViennaRNA structure features for original and C->U edited sequences.

    Returns: struct_delta (7-dim), loop_feats (9-dim), baseline_struct (6-dim)
    """
    import RNA

    seq = seq.upper().replace("T", "U")
    n = len(seq)
    center = n // 2

    md = RNA.md()
    md.temperature = 37.0
    fc = RNA.fold_compound(seq, md)
    mfe_structure, mfe = fc.mfe()
    fc.pf()
    bpp_raw = np.array(fc.bpp())
    bpp = bpp_raw[1:n+1, 1:n+1]
    pairing_prob = np.clip(np.sum(bpp, axis=0) + np.sum(bpp, axis=1), 0, 1)
    accessibility = 1.0 - pairing_prob

    # Entropy at center
    probs = bpp[center, :]
    probs = probs[probs > 1e-10]
    unpaired = max(0, 1.0 - np.sum(probs))
    if unpaired > 1e-10:
        probs = np.append(probs, unpaired)
    entropy_center = float(-np.sum(probs * np.log2(probs + 1e-10))) if len(probs) > 0 else 0.0

    # Edited
    seq_list = list(seq)
    if center < len(seq_list) and seq_list[center].upper() == "C":
        seq_list[center] = "U"
    edited_seq = "".join(seq_list)

    fc_ed = RNA.fold_compound(edited_seq, md)
    mfe_structure_ed, mfe_ed = fc_ed.mfe()
    fc_ed.pf()
    bpp_raw_ed = np.array(fc_ed.bpp())
    bpp_ed = bpp_raw_ed[1:n+1, 1:n+1]
    pairing_prob_ed = np.clip(np.sum(bpp_ed, axis=0) + np.sum(bpp_ed, axis=1), 0, 1)
    accessibility_ed = 1.0 - pairing_prob_ed

    probs_ed = bpp_ed[center, :]
    probs_ed = probs_ed[probs_ed > 1e-10]
    unpaired_ed = max(0, 1.0 - np.sum(probs_ed))
    if unpaired_ed > 1e-10:
        probs_ed = np.append(probs_ed, unpaired_ed)
    entropy_center_ed = float(-np.sum(probs_ed * np.log2(probs_ed + 1e-10))) if len(probs_ed) > 0 else 0.0

    window = 10
    start = max(0, center - window)
    end = min(n, center + window + 1)
    dp = pairing_prob_ed - pairing_prob
    da = accessibility_ed - accessibility

    struct_delta = np.zeros(7, dtype=np.float32)
    struct_delta[0] = dp[center]
    struct_delta[1] = da[center]
    struct_delta[2] = entropy_center_ed - entropy_center
    struct_delta[3] = mfe_ed - mfe
    struct_delta[4] = np.mean(dp[start:end])
    struct_delta[5] = np.mean(da[start:end])
    struct_delta[6] = np.std(dp[start:end])

    # Loop geometry from MFE structure
    loop_feats = _extract_loop_geometry(mfe_structure, center)

    # Baseline structure
    w_start = max(0, center - window)
    w_end = min(n, center + window + 1)
    baseline_struct = np.array([
        pairing_prob[center], accessibility[center], entropy_center, mfe,
        np.mean(pairing_prob[w_start:w_end]),
        np.mean(accessibility[w_start:w_end]),
    ], dtype=np.float32)

    return struct_delta, loop_feats, baseline_struct


def _extract_loop_geometry(dot_bracket, pos):
    """Extract 9-dim loop geometry features."""
    n = len(dot_bracket)
    feats = np.zeros(9, dtype=np.float32)
    if not dot_bracket or pos >= n or pos < 0:
        return feats

    is_unpaired = dot_bracket[pos] == "."
    feats[0] = float(is_unpaired)

    w = 10
    w_start = max(0, pos - w)
    w_end = min(n, pos + w + 1)
    local_region = dot_bracket[w_start:w_end]
    feats[8] = sum(1 for c in local_region if c == ".") / len(local_region)

    if is_unpaired:
        left = pos - 1
        while left >= 0 and dot_bracket[left] == ".":
            left -= 1
        right = pos + 1
        while right < n and dot_bracket[right] == ".":
            right += 1

        loop_start = (left + 1) if left >= 0 else 0
        loop_end = (right - 1) if right < n else n - 1
        loop_size = loop_end - loop_start + 1

        dist_left = pos - loop_start
        dist_right = loop_end - pos
        relative_pos = dist_left / (loop_size - 1) if loop_size > 1 else 0.5
        apex = (loop_start + loop_end) / 2.0

        left_stem = _stem_length(dot_bracket, left, "left")
        right_stem = _stem_length(dot_bracket, right, "right")

        feats[1] = loop_size
        feats[2] = min(dist_left, dist_right)
        feats[3] = abs(pos - apex)
        feats[4] = relative_pos
        feats[5] = left_stem
        feats[6] = right_stem
        feats[7] = max(left_stem, right_stem)
    else:
        dist_left = 0
        i = pos - 1
        while i >= 0 and dot_bracket[i] in "()":
            dist_left += 1
            i -= 1
        dist_right = 0
        j = pos + 1
        while j < n and dot_bracket[j] in "()":
            dist_right += 1
            j += 1
        feats[2] = min(dist_left, dist_right)
        stem_left = _stem_length(dot_bracket, pos, "left")
        stem_right = _stem_length(dot_bracket, pos, "right")
        feats[5] = stem_left
        feats[6] = stem_right
        feats[7] = max(stem_left, stem_right)

    return feats


def _stem_length(dot_bracket, boundary_pos, direction):
    n = len(dot_bracket)
    if boundary_pos < 0 or boundary_pos >= n:
        return 0
    if dot_bracket[boundary_pos] not in "()":
        return 0
    count = 0
    if direction == "left":
        i = boundary_pos
        while i >= 0 and dot_bracket[i] in "()":
            count += 1
            i -= 1
    else:
        i = boundary_pos
        while i < n and dot_bracket[i] in "()":
            count += 1
            i += 1
    return count


def compute_clinvar_features_worker(args):
    """Worker: compute 46-dim features for a ClinVar variant."""
    site_id, sequence = args
    try:
        seq = sequence.upper().replace("T", "U")
        motif = extract_motif_from_seq(seq)
        struct_delta, loop_feats, baseline_struct = compute_vienna_features(seq)
        hand_46 = np.concatenate([motif, struct_delta, loop_feats, baseline_struct])
        return site_id, hand_46
    except Exception:
        return site_id, None


# ---------------------------------------------------------------------------
# Training data loading
# ---------------------------------------------------------------------------

def load_training_data():
    """Load A3B positives + negatives for training."""
    if not SPLITS_CSV.exists():
        logger.error("Negatives file not found: %s", SPLITS_CSV)
        logger.error("Run: python scripts/multi_enzyme/generate_negatives_v2.py")
        sys.exit(1)

    df = pd.read_csv(SPLITS_CSV)
    df = df[df["enzyme"] == "A3B"].copy()

    if "is_edited" in df.columns:
        df["label"] = df["is_edited"].astype(int)
    elif "label" in df.columns:
        pass
    else:
        logger.error("No label column found. Need splits_multi_enzyme_v2_with_negatives.csv")
        sys.exit(1)

    n_pos = (df["label"] == 1).sum()
    n_neg = (df["label"] == 0).sum()
    logger.info("A3B training data: %d sites (pos=%d, neg=%d)", len(df), n_pos, n_neg)

    if n_neg == 0:
        logger.error("No negatives found. Classification requires negatives!")
        logger.error("Run: python scripts/multi_enzyme/generate_negatives_v2.py")
        sys.exit(1)

    return df


def build_training_features(df, sequences, structure_delta, loop_df):
    """Build 46-dim features for training sites.

    46 = motif(24) + struct_delta(7) + loop(9) + baseline_struct(6)
    """
    loop_cols = [
        "is_unpaired", "loop_size", "dist_to_junction", "dist_to_apex",
        "relative_loop_position", "left_stem_length", "right_stem_length",
        "max_adjacent_stem_length", "local_unpaired_fraction",
    ]

    features = []
    for sid in df["site_id"].astype(str):
        # Motif (24-dim)
        seq = sequences.get(sid, "N" * 201)
        motif = extract_motif_from_seq(seq)

        # Structure delta (7-dim)
        sd = structure_delta.get(sid, np.zeros(7, dtype=np.float32))
        if not isinstance(sd, np.ndarray):
            sd = np.array(sd, dtype=np.float32)

        # Loop (9-dim)
        if sid in loop_df.index:
            row = loop_df.loc[sid]
            loop = []
            for c in loop_cols:
                v = row[c] if c in row.index else 0.0
                loop.append(float(v) if pd.notna(v) else 0.0)
            loop = np.array(loop, dtype=np.float32)
        else:
            loop = np.zeros(len(loop_cols), dtype=np.float32)

        # Baseline struct (6-dim) — use zeros if not available from cache
        baseline = np.zeros(6, dtype=np.float32)

        feat = np.concatenate([motif, sd, loop, baseline])
        features.append(feat)

    result = np.array(features, dtype=np.float32)
    return np.nan_to_num(result, nan=0.0)


# ---------------------------------------------------------------------------
# ClinVar analysis
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
    """Compute pathogenic enrichment at a given threshold.

    labels: "Pathogenic"/"Likely_pathogenic" vs "Benign"/"Likely_benign"
    """
    path_mask = labels.isin(["Pathogenic", "Likely_pathogenic"])
    benign_mask = labels.isin(["Benign", "Likely_benign"])

    relevant = path_mask | benign_mask
    if relevant.sum() < 10:
        return None

    scores_rel = scores[relevant]
    path_rel = path_mask[relevant]

    above = scores_rel >= threshold
    below = scores_rel < threshold

    # 2x2: [path_above, benign_above], [path_below, benign_below]
    a = (above & path_rel).sum()
    b = (above & ~path_rel).sum()
    c = (below & path_rel).sum()
    d = (below & ~path_rel).sum()

    if min(a, b, c, d) == 0:
        # Add 0.5 continuity correction
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

    # --- Phase 1: Train A3B model ---
    logger.info("=" * 60)
    logger.info("Phase 1: Training A3B classifier")

    train_df = load_training_data()

    # Load pre-computed features
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
    from xgboost import XGBClassifier

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_aurocs = []
    cv_auprcs = []

    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        clf = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
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
        n_estimators=200, max_depth=6, learning_rate=0.1,
        random_state=SEED, eval_metric="logloss", use_label_encoder=False,
    )
    final_clf.fit(X_train, y_train, verbose=False)
    logger.info("Final model trained on %d sites", len(X_train))

    # --- Phase 2: Score ClinVar ---
    logger.info("=" * 60)
    logger.info("Phase 2: Scoring ClinVar variants")

    clinvar_df = load_clinvar()
    if clinvar_df is None:
        logger.error("ClinVar data not available. Saving partial results.")
        results = {
            "enzyme": "A3B",
            "cv_aurocs": cv_aurocs,
            "cv_auprcs": cv_auprcs,
            "mean_cv_auroc": float(np.mean(cv_aurocs)),
            "clinvar_scored": False,
            "error": "ClinVar data not found",
        }
        with open(OUTPUT_DIR / "a3b_clinvar_results.json", "w") as f:
            json.dump(results, f, indent=2)
        return

    # Try to load pre-computed ClinVar features from A3A experiment
    clinvar_features_loaded = False
    clinvar_hand_46 = None
    clinvar_site_ids = None

    # Check for consolidated ClinVar features cache
    if CLINVAR_FEATURES_CACHE.exists():
        logger.info("Loading pre-computed ClinVar features from %s...", CLINVAR_FEATURES_CACHE.name)
        cache_data = np.load(CLINVAR_FEATURES_CACHE, allow_pickle=True)
        clinvar_site_ids = list(cache_data["site_ids"])
        clinvar_hand_46 = cache_data["hand_46"]
        clinvar_features_loaded = True
        logger.info("  Loaded features for %d ClinVar variants (%d dims)",
                    len(clinvar_site_ids), clinvar_hand_46.shape[1])

    if not clinvar_features_loaded:
        # Must compute features from scratch (slow!)
        logger.warning("No pre-computed ClinVar features found.")
        logger.warning("Computing features from scratch (this may take hours)...")

        # Build sequences for ClinVar sites
        if "sequence" not in clinvar_df.columns:
            logger.error("ClinVar DataFrame has no 'sequence' column. Cannot compute features.")
            logger.error("Ensure clinvar_c2u_variants.csv has pre-extracted sequences.")
            sys.exit(1)

        site_ids = clinvar_df["site_id"].astype(str).tolist() if "site_id" in clinvar_df.columns else [str(i) for i in range(len(clinvar_df))]
        seqs = clinvar_df["sequence"].tolist()

        args = list(zip(site_ids, seqs))
        results_list = []

        n_chunks = (len(args) + CHUNK_SIZE - 1) // CHUNK_SIZE
        for chunk_idx in range(n_chunks):
            chunk_start = chunk_idx * CHUNK_SIZE
            chunk_end = min(len(args), (chunk_idx + 1) * CHUNK_SIZE)
            chunk_args = args[chunk_start:chunk_end]

            logger.info("Processing chunk %d/%d (%d sites)...",
                        chunk_idx + 1, n_chunks, len(chunk_args))

            with Pool(processes=N_WORKERS) as pool:
                chunk_results = pool.map(compute_clinvar_features_worker, chunk_args)
            results_list.extend(chunk_results)

        clinvar_site_ids = []
        clinvar_feats = []
        for sid, feat in results_list:
            if feat is not None:
                clinvar_site_ids.append(sid)
                clinvar_feats.append(feat)
        clinvar_hand_46 = np.array(clinvar_feats, dtype=np.float32)

    # Score with A3B model
    logger.info("Scoring %d ClinVar variants with A3B model...", len(clinvar_hand_46))
    # The A3A cache has 46-dim features, but our model uses 46-dim too (motif+delta+loop+baseline)
    # If cache features match our feature space, use directly
    if clinvar_hand_46.shape[1] != X_train.shape[1]:
        logger.warning("Feature dimension mismatch: ClinVar=%d, training=%d",
                       clinvar_hand_46.shape[1], X_train.shape[1])
        logger.warning("Truncating/padding to match training dimensions")
        n_feat = X_train.shape[1]
        if clinvar_hand_46.shape[1] > n_feat:
            clinvar_hand_46 = clinvar_hand_46[:, :n_feat]
        else:
            pad = np.zeros((clinvar_hand_46.shape[0], n_feat - clinvar_hand_46.shape[1]),
                           dtype=np.float32)
            clinvar_hand_46 = np.hstack([clinvar_hand_46, pad])

    a3b_scores = final_clf.predict_proba(clinvar_hand_46)[:, 1]

    # Match scores to ClinVar dataframe
    if clinvar_features_loaded:
        # Map site_id -> score
        score_map = {sid: float(s) for sid, s in zip(clinvar_site_ids, a3b_scores)}
        clinvar_df["a3b_score"] = clinvar_df.index.map(
            lambda i: score_map.get(str(clinvar_site_ids[i]) if i < len(clinvar_site_ids) else "", np.nan)
        )
        # Better approach: match by position in array
        if len(clinvar_site_ids) == len(clinvar_df):
            clinvar_df["a3b_score"] = a3b_scores
        else:
            logger.warning("ClinVar size mismatch: features=%d, df=%d",
                           len(clinvar_site_ids), len(clinvar_df))
            clinvar_df["a3b_score"] = np.nan
            for i, sid in enumerate(clinvar_site_ids):
                mask = clinvar_df.index == i
                if mask.any():
                    clinvar_df.loc[mask, "a3b_score"] = float(a3b_scores[i])
    else:
        clinvar_df["a3b_score"] = np.nan
        sid_to_idx = {str(sid): i for i, sid in enumerate(clinvar_site_ids)}
        for idx, row in clinvar_df.iterrows():
            sid = str(row.get("site_id", idx))
            if sid in sid_to_idx:
                clinvar_df.at[idx, "a3b_score"] = float(a3b_scores[sid_to_idx[sid]])

    scored_mask = clinvar_df["a3b_score"].notna()
    logger.info("Scored %d / %d ClinVar variants", scored_mask.sum(), len(clinvar_df))

    # --- Phase 3: Enrichment analysis ---
    logger.info("=" * 60)
    logger.info("Phase 3: Pathogenic enrichment analysis")

    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    enrichment_results = []

    scored_df = clinvar_df[scored_mask].copy()
    for thresh in thresholds:
        enr = compute_enrichment(
            scored_df["a3b_score"].values,
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
            scored_df["a3b_score"].values,
            scored_df["significance_simple"],
            cal_thresh,
        )
        if enr:
            enr["original_threshold"] = float(thresh)
            enr["calibrated_threshold"] = float(cal_thresh)
            calibrated_results.append(enr)

    # Save results
    results = {
        "enzyme": "A3B",
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
    }

    out_path = OUTPUT_DIR / "a3b_clinvar_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", out_path)

    # Save scored ClinVar variants
    scored_csv = OUTPUT_DIR / "clinvar_a3b_scores.csv"
    scored_df[["a3b_score", "significance_simple"]].to_csv(scored_csv, index=False)
    logger.info("Scores saved to %s", scored_csv)

    elapsed = time.time() - t_start
    logger.info("Total time: %.1f seconds", elapsed)


if __name__ == "__main__":
    main()
