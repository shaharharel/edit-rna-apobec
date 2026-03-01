#!/usr/bin/env python
"""ClinVar C>U variant scoring: GB_HandFeatures vs RNAsee head-to-head.

Scores ALL ~1.68M ClinVar C>U variants using parallelized ViennaRNA structure
computation. Trains GB_Full (46-dim hand features) and RNAsee_RF (50-bit binary
encoding) on known A3A editing sites, then applies to all ClinVar variants.

Pipeline:
  Phase 1: Train models on known A3A editing sites (5-fold CV + final model)
  Phase 2: Compute ViennaRNA features for all ClinVar variants (multiprocessing)
  Phase 3: Score all variants with both models
  Phase 4: Analysis and visualization (head-to-head comparison)

Usage:
    conda run -n quris python experiments/apobec/exp_clinvar_prediction.py
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ks_2samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, precision_recall_curve,
)
from sklearn.model_selection import KFold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SPLITS_A3A_CSV = PROJECT_ROOT / "data" / "processed" / "splits_expanded_a3a.csv"
ALLC_NEG_CSV = PROJECT_ROOT / "data" / "processed" / "negatives_per_dataset_all_c.csv"
SEQ_JSON = PROJECT_ROOT / "data" / "processed" / "site_sequences.json"
STRUCT_CACHE = PROJECT_ROOT / "data" / "processed" / "embeddings" / "vienna_structure_cache.npz"
LOOP_POS_CSV = (
    PROJECT_ROOT / "experiments" / "apobec" / "outputs"
    / "loop_position" / "loop_position_per_site.csv"
)
CLINVAR_CSV = PROJECT_ROOT / "data" / "processed" / "clinvar_c2u_variants.csv"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "apobec" / "outputs" / "clinvar_prediction"

SEED = 42
N_WORKERS = 10  # Leave 4 cores free on 14-core machine
CHUNK_SIZE = 50_000  # Process ClinVar in chunks for progress + resume
CENTER = 100  # Edit site in 201nt window (0-indexed)


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
# Feature extraction: Motif (24-dim)
# ---------------------------------------------------------------------------

def extract_motif_from_seq(seq):
    """Extract 24-dim motif features from a 201nt sequence. Center at pos 100."""
    seq = seq.upper().replace("T", "U")
    ep = CENTER
    bases = ["A", "C", "G", "U"]

    up = seq[ep - 1] if ep > 0 else "N"
    down = seq[ep + 1] if ep < len(seq) - 1 else "N"

    # 5' dinucleotide (4-dim)
    feat_5p = [1.0 if up + "C" == m else 0.0 for m in ["UC", "CC", "AC", "GC"]]
    # 3' dinucleotide (4-dim)
    feat_3p = [1.0 if "C" + down == m else 0.0 for m in ["CA", "CG", "CU", "CC"]]

    # Trinucleotide upstream context (8-dim: m2 + m1)
    trinuc_up = [0.0] * 8
    for offset, bo in [(-2, 0), (-1, 4)]:
        pos = ep + offset
        if 0 <= pos < len(seq):
            for bi, b in enumerate(bases):
                if seq[pos] == b:
                    trinuc_up[bo + bi] = 1.0

    # Trinucleotide downstream context (8-dim: p1 + p2)
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


# ---------------------------------------------------------------------------
# Feature extraction: RNAsee 50-bit encoding
# ---------------------------------------------------------------------------

def encode_rnasee_from_seq(seq):
    """Encode a 201nt sequence using RNAsee's exact 50-bit binary encoding.

    15nt upstream + 10nt downstream of center C (excluded).
    Each nt: 2 bits (is_purine, pairs_GC). A=(1,0), G=(1,1), C=(0,1), U=(0,0).
    """
    BWD, FWD = 15, 10
    features = np.zeros((BWD + FWD) * 2, dtype=np.float32)
    if len(seq) < CENTER + FWD + 1:
        return features

    feat_idx = 0
    for offset in range(-BWD, 0):
        pos = CENTER + offset
        if pos < 0:
            feat_idx += 2
            continue
        nuc = seq[pos].upper()
        features[feat_idx] = 1.0 if nuc in ("A", "G") else 0.0
        features[feat_idx + 1] = 1.0 if nuc in ("G", "C") else 0.0
        feat_idx += 2

    for offset in range(1, FWD + 1):
        pos = CENTER + offset
        if pos >= len(seq):
            feat_idx += 2
            continue
        nuc = seq[pos].upper()
        features[feat_idx] = 1.0 if nuc in ("A", "G") else 0.0
        features[feat_idx + 1] = 1.0 if nuc in ("G", "C") else 0.0
        feat_idx += 2

    return features


def encode_rnasee_features_batch(sequences, site_ids):
    """Encode RNAsee features for a list of site IDs."""
    features = []
    for sid in site_ids:
        seq = sequences.get(str(sid), "N" * 201)
        features.append(encode_rnasee_from_seq(seq))
    return np.array(features, dtype=np.float32)


# ---------------------------------------------------------------------------
# Feature extraction: ViennaRNA structure + loop geometry
# ---------------------------------------------------------------------------

def _entropy_at_pos(bpp, pos):
    """Compute structure entropy at a single position from BPP matrix."""
    probs = bpp[pos, :]
    probs = probs[probs > 1e-10]
    if len(probs) == 0:
        return 0.0
    unpaired = max(0, 1.0 - np.sum(probs))
    if unpaired > 1e-10:
        probs = np.append(probs, unpaired)
    return float(-np.sum(probs * np.log2(probs + 1e-10)))


def compute_vienna_features(seq):
    """Compute ViennaRNA structure features for original and C->U edited sequences.

    Optimized: only computes entropy at center position (not all 201 positions).

    Returns:
        struct_delta (7-dim): delta pairing/accessibility/entropy at center + window
        loop_feats (9-dim): loop geometry features
        baseline_struct (6-dim): baseline structure features of original sequence
    """
    import RNA

    seq = seq.upper().replace("T", "U")
    n = len(seq)
    center = n // 2

    # -- Original sequence --
    md = RNA.md()
    md.temperature = 37.0
    fc = RNA.fold_compound(seq, md)
    mfe_structure, mfe = fc.mfe()
    _, pf_energy = fc.pf()
    bpp_raw = np.array(fc.bpp())
    bpp = bpp_raw[1:n+1, 1:n+1]
    pairing_prob = np.clip(np.sum(bpp, axis=0) + np.sum(bpp, axis=1), 0, 1)
    accessibility = 1.0 - pairing_prob
    entropy_center = _entropy_at_pos(bpp, center)

    # -- Edited sequence (C->U at center) --
    seq_list = list(seq)
    if center < len(seq_list) and seq_list[center].upper() == "C":
        seq_list[center] = "U"
    edited_seq = "".join(seq_list)

    fc_ed = RNA.fold_compound(edited_seq, md)
    _, mfe_ed = fc_ed.mfe()
    _, _ = fc_ed.pf()
    bpp_raw_ed = np.array(fc_ed.bpp())
    bpp_ed = bpp_raw_ed[1:n+1, 1:n+1]
    pairing_prob_ed = np.clip(np.sum(bpp_ed, axis=0) + np.sum(bpp_ed, axis=1), 0, 1)
    accessibility_ed = 1.0 - pairing_prob_ed
    entropy_center_ed = _entropy_at_pos(bpp_ed, center)

    # -- 7-dim structure delta features --
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

    # -- 9-dim loop geometry features from MFE structure --
    loop_feats = _extract_loop_geometry(mfe_structure, center)

    # -- 6-dim baseline structure features (original sequence) --
    w_start = max(0, center - window)
    w_end = min(n, center + window + 1)
    baseline_struct = np.array([
        pairing_prob[center],                         # baseline_pairing_center
        accessibility[center],                        # baseline_accessibility_center
        entropy_center,                               # baseline_entropy_center
        mfe,                                          # baseline_mfe
        np.mean(pairing_prob[w_start:w_end]),          # baseline_pairing_local_mean
        np.mean(accessibility[w_start:w_end]),         # baseline_accessibility_local_mean
    ], dtype=np.float32)

    return struct_delta, loop_feats, baseline_struct


def _extract_loop_geometry(dot_bracket, pos):
    """Extract 9-dim loop geometry features from dot-bracket structure.

    Features: is_unpaired, loop_size, dist_to_junction, dist_to_apex,
              relative_loop_position, left_stem_length, right_stem_length,
              max_adjacent_stem_length, local_unpaired_fraction
    """
    n = len(dot_bracket)
    feats = np.zeros(9, dtype=np.float32)

    if not dot_bracket or pos >= n or pos < 0:
        return feats

    is_unpaired = dot_bracket[pos] == "."
    feats[0] = float(is_unpaired)

    # Local unpaired fraction (within +/- 10 nt)
    w = 10
    w_start = max(0, pos - w)
    w_end = min(n, pos + w + 1)
    local_region = dot_bracket[w_start:w_end]
    feats[8] = sum(1 for c in local_region if c == ".") / len(local_region)

    if is_unpaired:
        # Find loop boundaries
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
        dist_to_nearest = min(dist_left, dist_right)
        relative_pos = dist_left / (loop_size - 1) if loop_size > 1 else 0.5
        apex = (loop_start + loop_end) / 2.0
        dist_to_apex = abs(pos - apex)

        # Stem lengths
        left_stem = _stem_length(dot_bracket, left, "left")
        right_stem = _stem_length(dot_bracket, right, "right")

        feats[1] = loop_size
        feats[2] = dist_to_nearest  # dist_to_junction
        feats[3] = dist_to_apex
        feats[4] = relative_pos
        feats[5] = left_stem
        feats[6] = right_stem
        feats[7] = max(left_stem, right_stem)
    else:
        # In a stem
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
        feats[2] = min(dist_left, dist_right)  # dist_to_junction

        stem_left = _stem_length(dot_bracket, pos, "left")
        stem_right = _stem_length(dot_bracket, pos, "right")
        feats[5] = stem_left
        feats[6] = stem_right
        feats[7] = max(stem_left, stem_right)

    return feats


def _stem_length(dot_bracket, boundary_pos, direction):
    """Count consecutive paired characters from boundary_pos."""
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


# ---------------------------------------------------------------------------
# Worker function for multiprocessing ClinVar feature computation
# ---------------------------------------------------------------------------

def compute_clinvar_features(args):
    """Compute all features for one ClinVar variant.

    Returns (site_id, hand_46, rnasee_50) or (site_id, None, None) on failure.
    """
    site_id, sequence = args
    try:
        seq = sequence.upper().replace("T", "U")

        # 1. Motif features (24-dim) -- instant
        motif = extract_motif_from_seq(seq)

        # 2. RNAsee 50-bit encoding -- instant
        rnasee = encode_rnasee_from_seq(seq)

        # 3. ViennaRNA structure features (slow)
        struct_delta, loop_feats, baseline_struct = compute_vienna_features(seq)

        # 4. Concatenate hand features: motif(24) + struct_delta(7) + loop(9) + baseline(6) = 46
        hand_46 = np.concatenate([motif, struct_delta, loop_feats, baseline_struct])

        return site_id, hand_46, rnasee
    except Exception:
        return site_id, None, None


# ---------------------------------------------------------------------------
# Feature extraction for known training sites (from pre-computed caches)
# ---------------------------------------------------------------------------

def load_known_features():
    """Load pre-computed features for known editing sites from caches.

    Returns:
        sequences: dict of site_id -> 201nt sequence
        structure_delta: dict of site_id -> 7-dim delta features
        loop_df: DataFrame with loop geometry features indexed by site_id
    """
    logger.info("Loading pre-computed features for known sites...")

    # Sequences
    sequences = {}
    if SEQ_JSON.exists():
        with open(SEQ_JSON) as f:
            sequences = json.load(f)
    logger.info("  %d sequences loaded", len(sequences))

    # Structure delta features from cache
    structure_delta = {}
    struct_data = {}
    if STRUCT_CACHE.exists():
        data = np.load(STRUCT_CACHE, allow_pickle=True)
        sids = data["site_ids"]
        feats = data["delta_features"]
        structure_delta = {str(sid): feats[i] for i, sid in enumerate(sids)}

        # Also extract baseline structure features for known sites
        pp = data["pairing_probs"]
        acc = data["accessibilities"]
        ent = data["entropies"] if "entropies" in data else None
        mfes = data["mfes"]

        for i, sid in enumerate(sids):
            sid_str = str(sid)
            center = CENTER
            w = 10
            w_start = max(0, center - w)
            w_end = min(201, center + w + 1)
            baseline = np.array([
                pp[i][center],
                acc[i][center],
                ent[i][center] if ent is not None else 0.0,
                mfes[i],
                np.mean(pp[i][w_start:w_end]),
                np.mean(acc[i][w_start:w_end]),
            ], dtype=np.float32)
            struct_data[sid_str] = baseline

        del data
        gc.collect()
    logger.info("  %d structure delta features loaded", len(structure_delta))

    # Loop geometry features
    loop_df = pd.DataFrame()
    if LOOP_POS_CSV.exists():
        loop_df = pd.read_csv(LOOP_POS_CSV)
        loop_df["site_id"] = loop_df["site_id"].astype(str)
        loop_df = loop_df.set_index("site_id")
        logger.info("  %d loop features loaded", len(loop_df))

    return sequences, structure_delta, loop_df, struct_data


def build_hand_46_for_known(site_ids, sequences, structure_delta, loop_df, baseline_struct):
    """Build 46-dim hand features for known sites from pre-computed caches.

    46 = motif(24) + struct_delta(7) + loop(9) + baseline_struct(6)
    """
    loop_cols = [
        "is_unpaired", "loop_size", "dist_to_junction", "dist_to_apex",
        "relative_loop_position", "left_stem_length", "right_stem_length",
        "max_adjacent_stem_length", "local_unpaired_fraction",
    ]

    features = []
    for sid in site_ids:
        sid_str = str(sid)

        # Motif (24-dim)
        seq = sequences.get(sid_str, "N" * 201)
        motif = extract_motif_from_seq(seq)

        # Structure delta (7-dim)
        sd = structure_delta.get(sid_str, np.zeros(7, dtype=np.float32))
        if not isinstance(sd, np.ndarray):
            sd = np.array(sd, dtype=np.float32)

        # Loop geometry (9-dim)
        if sid_str in loop_df.index:
            loop = loop_df.loc[sid_str, loop_cols].values.astype(np.float32)
        else:
            loop = np.zeros(len(loop_cols), dtype=np.float32)

        # Baseline structure (6-dim)
        bl = baseline_struct.get(sid_str, np.zeros(6, dtype=np.float32))

        feat = np.concatenate([motif, sd, loop, bl])
        features.append(feat)

    result = np.array(features, dtype=np.float32)
    return np.nan_to_num(result, nan=0.0)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_training_data():
    """Load A3A positives + Asaoka all-C negatives for training.

    Returns DataFrame with site_id, label, dataset_source columns.
    """
    logger.info("Loading training data...")

    # A3A positives from splits_expanded_a3a.csv
    a3a_df = pd.read_csv(SPLITS_A3A_CSV)
    pos_df = a3a_df[a3a_df["is_edited"] == 1][["site_id", "is_edited", "dataset_source"]].copy()
    pos_df = pos_df.rename(columns={"is_edited": "label"})
    logger.info("  A3A positives: %d", len(pos_df))

    # Asaoka negatives (all-cytidine, 1:3 ratio)
    neg_df = pd.read_csv(ALLC_NEG_CSV)
    asaoka_neg = neg_df[neg_df["dataset_source"] == "asaoka_2019"][
        ["site_id", "is_edited", "dataset_source"]
    ].copy()
    asaoka_neg = asaoka_neg.rename(columns={"is_edited": "label"})

    # Subsample to 1:3 ratio
    target_neg = len(pos_df) * 3
    if len(asaoka_neg) > target_neg:
        asaoka_neg = asaoka_neg.sample(n=target_neg, random_state=SEED)
    logger.info("  Asaoka all-C negatives: %d", len(asaoka_neg))

    combined = pd.concat([pos_df, asaoka_neg], ignore_index=True)
    logger.info("  Total training sites: %d (pos=%d, neg=%d)",
                len(combined), (combined["label"] == 1).sum(),
                (combined["label"] == 0).sum())
    return combined


def load_clinvar_data():
    """Load ClinVar C>U variants with simplified significance."""
    if not CLINVAR_CSV.exists():
        logger.error("ClinVar data not found: %s", CLINVAR_CSV)
        logger.error("Run: python scripts/apobec/download_clinvar.py")
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

    logger.info("Clinical significance breakdown:")
    for sig, count in df["significance_simple"].value_counts().items():
        logger.info("  %s: %d", sig, count)

    return df


# ---------------------------------------------------------------------------
# Fill missing ViennaRNA features for training sites
# ---------------------------------------------------------------------------

def _compute_training_site_features(args):
    """Worker: compute ViennaRNA features for a single training site.

    Returns (site_id, struct_delta_7, loop_feats_9, baseline_struct_6) or None.
    """
    site_id, sequence = args
    try:
        struct_delta, loop_feats, baseline_struct = compute_vienna_features(sequence)
        return site_id, struct_delta, loop_feats, baseline_struct
    except Exception:
        return site_id, None, None, None


def _fill_missing_vienna_features(site_ids, sequences, structure_delta, loop_df, baseline_struct):
    """Compute ViennaRNA features for training sites not in pre-computed caches.

    Updates structure_delta and baseline_struct dicts in-place.
    Returns updated loop_df (may be a new DataFrame with additional rows).
    This prevents feature-availability leakage where negatives get zero vectors
    for structure features while positives get real values.
    """
    loop_cols = [
        "is_unpaired", "loop_size", "dist_to_junction", "dist_to_apex",
        "relative_loop_position", "left_stem_length", "right_stem_length",
        "max_adjacent_stem_length", "local_unpaired_fraction",
    ]

    # Find sites missing from ANY cache
    missing_sids = []
    for sid in site_ids:
        sid_str = str(sid)
        seq = sequences.get(sid_str)
        if not seq or len(seq) < 10:
            continue
        has_delta = sid_str in structure_delta
        has_loop = sid_str in loop_df.index if len(loop_df) > 0 else False
        has_baseline = sid_str in baseline_struct
        if not (has_delta and has_loop and has_baseline):
            missing_sids.append(sid_str)

    if not missing_sids:
        logger.info("  All training sites have ViennaRNA features in cache")
        return loop_df

    logger.info("  %d training sites missing ViennaRNA features from caches",
                len(missing_sids))

    # Check for cached training features from a previous run
    cache_path = OUTPUT_DIR / "training_neg_vienna_cache.npz"
    cached_delta = {}
    cached_loop = {}
    cached_baseline = {}
    if cache_path.exists():
        cached = np.load(str(cache_path), allow_pickle=True)
        cached_sids = [str(s) for s in cached["site_ids"]]
        for i, sid in enumerate(cached_sids):
            cached_delta[sid] = cached["struct_deltas"][i]
            cached_loop[sid] = cached["loop_feats"][i]
            cached_baseline[sid] = cached["baseline_structs"][i]
        logger.info("  Loaded %d cached training features from previous run", len(cached_sids))

    # Determine which sites still need computation
    still_missing = [sid for sid in missing_sids if sid not in cached_delta]

    # Apply cached features
    n_from_cache = 0
    new_loop_rows = []
    for sid in missing_sids:
        if sid in cached_delta:
            structure_delta[sid] = cached_delta[sid]
            baseline_struct[sid] = cached_baseline[sid]
            row = {col: float(cached_loop[sid][i]) for i, col in enumerate(loop_cols)}
            row["site_id"] = sid
            new_loop_rows.append(row)
            n_from_cache += 1

    if n_from_cache > 0:
        logger.info("  Applied %d features from cache", n_from_cache)

    # Compute remaining missing sites
    compute_results = []
    if still_missing:
        logger.info("  Computing ViennaRNA features for %d remaining sites...", len(still_missing))
        args = [(sid, sequences[sid]) for sid in still_missing]

        t0 = time.time()
        n_workers = min(N_WORKERS, len(args))

        if n_workers > 1 and len(args) > 10:
            with Pool(processes=n_workers) as pool:
                compute_results = pool.map(_compute_training_site_features, args)
        else:
            compute_results = [_compute_training_site_features(a) for a in args]

        n_success = 0
        for sid_str, sd, lf, bl in compute_results:
            if sd is None:
                continue
            n_success += 1
            structure_delta[sid_str] = sd
            baseline_struct[sid_str] = bl
            row = {col: float(lf[i]) for i, col in enumerate(loop_cols)}
            row["site_id"] = sid_str
            new_loop_rows.append(row)

        elapsed = time.time() - t0
        logger.info("  Computed %d/%d sites in %.1f sec (%.1f sites/sec)",
                    n_success, len(still_missing), elapsed,
                    n_success / max(elapsed, 0.01))

        # Save/update cache for future runs
        all_cache_sids = list(cached_delta.keys()) + [
            r[0] for r in compute_results if r[1] is not None
        ]
        all_cache_deltas = [cached_delta.get(s, None) for s in cached_delta] + [
            r[1] for r in compute_results if r[1] is not None
        ]
        all_cache_loops = [cached_loop.get(s, None) for s in cached_delta] + [
            r[2] for r in compute_results if r[1] is not None
        ]
        all_cache_baselines = [cached_baseline.get(s, None) for s in cached_delta] + [
            r[3] for r in compute_results if r[1] is not None
        ]
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            site_ids=np.array(all_cache_sids),
            struct_deltas=np.array(all_cache_deltas, dtype=np.float32),
            loop_feats=np.array(all_cache_loops, dtype=np.float32),
            baseline_structs=np.array(all_cache_baselines, dtype=np.float32),
        )
        logger.info("  Saved training neg cache (%d sites) to %s",
                    len(all_cache_sids), cache_path.name)

    # Merge new loop rows into loop_df
    if new_loop_rows:
        new_loop_df = pd.DataFrame(new_loop_rows).set_index("site_id")
        if len(loop_df) > 0:
            loop_df = pd.concat([loop_df, new_loop_df])
        else:
            loop_df = new_loop_df

    return loop_df


# ---------------------------------------------------------------------------
# Phase 1: Train models on known sites
# ---------------------------------------------------------------------------

def run_phase1(train_df, sequences, structure_delta, loop_df, baseline_struct):
    """Train GB_Full and RNAsee_RF with 5-fold CV, then train final models on all data.

    Returns (gb_model, rf_model, cv_results).
    """
    logger.info("=" * 70)
    logger.info("PHASE 1: TRAIN MODELS ON KNOWN EDITING SITES")
    logger.info("=" * 70)

    site_ids = train_df["site_id"].values
    labels = train_df["label"].values.astype(np.float32)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    logger.info("Training data: %d sites (pos=%d, neg=%d)", len(labels), n_pos, n_neg)

    # Extract features for ALL training sites
    logger.info("Extracting features for training sites...")

    # Extract sequences for negative sites not in sequence cache
    neg_sids_missing = [str(sid) for sid in site_ids if str(sid) not in sequences]
    if neg_sids_missing:
        logger.info("  %d sites missing sequences (negatives without cached sequences)",
                    len(neg_sids_missing))
        # For negative sites without sequences, try to extract from genome
        neg_with_coords = train_df[train_df["site_id"].isin(neg_sids_missing)]
        _extract_neg_sequences(neg_with_coords, sequences)

    # Compute ViennaRNA features for training sites missing from caches.
    # Without this, negatives get zero-vector structure/loop/baseline features,
    # causing the model to learn "has features = positive" instead of actual patterns.
    loop_df = _fill_missing_vienna_features(
        site_ids, sequences, structure_delta, loop_df, baseline_struct
    )

    # Build 46-dim hand features
    hand_46 = build_hand_46_for_known(site_ids, sequences, structure_delta, loop_df, baseline_struct)
    logger.info("  Hand features shape: %s", hand_46.shape)

    # Build 50-bit RNAsee features
    rnasee_50 = encode_rnasee_features_batch(sequences, [str(s) for s in site_ids])
    logger.info("  RNAsee features shape: %s", rnasee_50.shape)

    # ---- 5-fold CV ----
    logger.info("\n5-fold cross-validation...")
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

    gb_fold_results = []
    rf_fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(site_ids)):
        fold_seed = SEED + fold_idx
        np.random.seed(fold_seed)

        # Inner split: 80% train, 20% val
        n_remain = len(train_idx)
        inner_perm = np.random.RandomState(fold_seed).permutation(n_remain)
        n_val = int(n_remain * 0.2)
        val_inner = train_idx[inner_perm[:n_val]]
        train_inner = train_idx[inner_perm[n_val:]]

        X_train_h, y_train = hand_46[train_inner], labels[train_inner]
        X_val_h, y_val = hand_46[val_inner], labels[val_inner]
        X_test_h, y_test = hand_46[test_idx], labels[test_idx]

        # GB_Full (46-dim)
        try:
            from xgboost import XGBClassifier
            gb = XGBClassifier(
                n_estimators=500, max_depth=5, learning_rate=0.05,
                subsample=0.8, min_child_weight=10,
                scale_pos_weight=n_neg / max(n_pos, 1),
                random_state=fold_seed, n_jobs=1,
                early_stopping_rounds=30, verbosity=0,
                eval_metric="logloss",
            )
            gb.fit(X_train_h, y_train, eval_set=[(X_val_h, y_val)], verbose=False)
            y_score_gb = gb.predict_proba(X_test_h)[:, 1]
            gb_metrics = compute_binary_metrics(y_test, y_score_gb)
            gb_fold_results.append(gb_metrics)
            del gb
        except ImportError:
            logger.error("xgboost not available!")
            return None, None, {}

        # RNAsee_RF (50-bit)
        X_train_r = rnasee_50[train_inner]
        X_test_r = rnasee_50[test_idx]
        rf = RandomForestClassifier(random_state=fold_seed, n_jobs=-1)
        rf.fit(X_train_r, y_train)
        y_score_rf = rf.predict_proba(X_test_r)[:, 1]
        rf_metrics = compute_binary_metrics(y_test, y_score_rf)
        rf_fold_results.append(rf_metrics)
        del rf

        logger.info("  Fold %d: GB_Full AUROC=%.4f  RNAsee_RF AUROC=%.4f",
                    fold_idx + 1, gb_metrics["auroc"], rf_metrics["auroc"])

    # CV summary
    cv_results = {
        "GB_Full": {
            "fold_results": gb_fold_results,
            "mean_auroc": float(np.mean([r["auroc"] for r in gb_fold_results])),
            "std_auroc": float(np.std([r["auroc"] for r in gb_fold_results])),
            "mean_auprc": float(np.mean([r["auprc"] for r in gb_fold_results])),
        },
        "RNAsee_RF": {
            "fold_results": rf_fold_results,
            "mean_auroc": float(np.mean([r["auroc"] for r in rf_fold_results])),
            "std_auroc": float(np.std([r["auroc"] for r in rf_fold_results])),
            "mean_auprc": float(np.mean([r["auprc"] for r in rf_fold_results])),
        },
    }

    logger.info("\nCV Summary:")
    logger.info("  GB_Full:    AUROC=%.4f +/- %.4f  AUPRC=%.4f",
                cv_results["GB_Full"]["mean_auroc"],
                cv_results["GB_Full"]["std_auroc"],
                cv_results["GB_Full"]["mean_auprc"])
    logger.info("  RNAsee_RF:  AUROC=%.4f +/- %.4f  AUPRC=%.4f",
                cv_results["RNAsee_RF"]["mean_auroc"],
                cv_results["RNAsee_RF"]["std_auroc"],
                cv_results["RNAsee_RF"]["mean_auprc"])

    # ---- Train final models on ALL data ----
    logger.info("\nTraining final models on all %d sites...", len(labels))

    from xgboost import XGBClassifier
    gb_final = XGBClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        subsample=0.8, min_child_weight=10,
        scale_pos_weight=n_neg / max(n_pos, 1),
        random_state=SEED, n_jobs=1,
        verbosity=0, eval_metric="logloss",
    )
    gb_final.fit(hand_46, labels)
    logger.info("  GB_Full trained (46 features)")

    rf_final = RandomForestClassifier(random_state=SEED, n_jobs=-1)
    rf_final.fit(rnasee_50, labels)
    logger.info("  RNAsee_RF trained (50 features)")

    return gb_final, rf_final, cv_results


def _extract_neg_sequences(neg_df, sequences):
    """Extract sequences for negative sites from genome (in-place update)."""
    genome_path = PROJECT_ROOT / "data" / "raw" / "genomes" / "hg19.fa"
    if not genome_path.exists():
        # Try hg38
        genome_path = PROJECT_ROOT / "data" / "raw" / "genomes" / "hg38.fa"
    if not genome_path.exists():
        logger.warning("Genome not found, cannot extract negative sequences")
        return

    try:
        from pyfaidx import Fasta
        genome = Fasta(str(genome_path))
    except ImportError:
        logger.warning("pyfaidx not available, cannot extract negative sequences")
        return

    extracted = 0
    for _, row in neg_df.iterrows():
        sid = str(row["site_id"])
        if sid in sequences:
            continue
        chrom = str(row.get("chr", ""))
        pos = int(row.get("start", 0))
        strand = str(row.get("strand", "+"))

        if chrom not in genome:
            continue
        chrom_len = len(genome[chrom])
        g_start = pos - 100
        g_end = pos + 101
        if g_start < 0 or g_end > chrom_len:
            continue

        dna_seq = str(genome[chrom][g_start:g_end]).upper()
        if len(dna_seq) != 201:
            continue

        if strand == "-":
            comp = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
            dna_seq = "".join(comp.get(b, "N") for b in reversed(dna_seq))

        sequences[sid] = dna_seq.replace("T", "U")
        extracted += 1

    if extracted > 0:
        logger.info("  Extracted %d negative sequences from genome", extracted)


# ---------------------------------------------------------------------------
# Phase 2: Compute features for all ClinVar variants
# ---------------------------------------------------------------------------

def run_phase2(clinvar_df, output_dir):
    """Compute ViennaRNA features for all ClinVar variants using multiprocessing.

    Processes in chunks of CHUNK_SIZE with progress logging and resume support.
    Returns (hand_46_array, rnasee_50_array, valid_indices).
    """
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 2: COMPUTE FEATURES FOR ALL CLINVAR VARIANTS")
    logger.info("=" * 70)

    # Filter to valid 201nt sequences
    valid_mask = clinvar_df["sequence"].str.len() == 201
    valid_df = clinvar_df[valid_mask].copy()
    n_valid = len(valid_df)
    n_skipped = len(clinvar_df) - n_valid
    if n_skipped > 0:
        logger.info("Skipping %d variants without valid 201nt sequences", n_skipped)
    logger.info("Processing %d valid ClinVar variants with %d workers",
                n_valid, N_WORKERS)

    # Check for existing intermediate results (resume support)
    features_dir = output_dir / "intermediate"
    features_dir.mkdir(parents=True, exist_ok=True)

    all_site_ids = valid_df["site_id"].values
    all_sequences = valid_df["sequence"].values

    # Determine which chunks are already done
    n_chunks = (n_valid + CHUNK_SIZE - 1) // CHUNK_SIZE
    completed_chunks = set()
    for chunk_idx in range(n_chunks):
        chunk_file = features_dir / f"chunk_{chunk_idx:04d}.npz"
        if chunk_file.exists():
            completed_chunks.add(chunk_idx)
    if completed_chunks:
        logger.info("Found %d/%d completed chunks (resume mode)", len(completed_chunks), n_chunks)

    # Process remaining chunks
    t0 = time.time()
    n_success_total = 0
    n_fail_total = 0

    for chunk_idx in range(n_chunks):
        chunk_file = features_dir / f"chunk_{chunk_idx:04d}.npz"

        if chunk_idx in completed_chunks:
            # Load existing chunk to count successes
            data = np.load(chunk_file, allow_pickle=True)
            n_success_total += int(data.get("n_success", len(data["site_ids"])))
            continue

        start = chunk_idx * CHUNK_SIZE
        end = min(start + CHUNK_SIZE, n_valid)
        chunk_sids = all_site_ids[start:end]
        chunk_seqs = all_sequences[start:end]

        logger.info("Processing chunk %d/%d (sites %d-%d)...",
                    chunk_idx + 1, n_chunks, start, end - 1)
        t_chunk = time.time()

        # Prepare args for multiprocessing
        args = [(str(sid), str(seq)) for sid, seq in zip(chunk_sids, chunk_seqs)]

        # Process with multiprocessing
        chunk_hand = np.zeros((len(args), 46), dtype=np.float32)
        chunk_rnasee = np.zeros((len(args), 50), dtype=np.float32)
        chunk_valid = np.zeros(len(args), dtype=bool)

        with Pool(processes=N_WORKERS) as pool:
            results = pool.map(compute_clinvar_features, args)

        n_success = 0
        n_fail = 0
        for i, (sid, hand, rnasee) in enumerate(results):
            if hand is not None:
                chunk_hand[i] = hand
                chunk_rnasee[i] = rnasee
                chunk_valid[i] = True
                n_success += 1
            else:
                n_fail += 1

        n_success_total += n_success
        n_fail_total += n_fail

        # Save chunk
        np.savez_compressed(
            chunk_file,
            site_ids=chunk_sids,
            hand_46=chunk_hand,
            rnasee_50=chunk_rnasee,
            valid=chunk_valid,
            n_success=n_success,
        )

        elapsed_chunk = time.time() - t_chunk
        elapsed_total = time.time() - t0
        rate = n_success / max(elapsed_chunk, 1)
        remaining = n_valid - end
        eta = remaining / max(n_success_total / max(elapsed_total, 1), 1)
        logger.info("  Chunk %d: %d/%d success (%.1f sites/sec, ETA: %.0f min)",
                    chunk_idx + 1, n_success, len(args), rate, eta / 60)

        del results
        gc.collect()

    elapsed_total = time.time() - t0
    logger.info("\nPhase 2 complete: %d/%d success, %d failed (%.1f min)",
                n_success_total, n_valid, n_fail_total, elapsed_total / 60)

    # Load all chunks and concatenate
    logger.info("Loading and concatenating all chunks...")
    all_hand = []
    all_rnasee = []
    all_valid = []
    all_chunk_sids = []

    for chunk_idx in range(n_chunks):
        chunk_file = features_dir / f"chunk_{chunk_idx:04d}.npz"
        data = np.load(chunk_file, allow_pickle=True)
        all_hand.append(data["hand_46"])
        all_rnasee.append(data["rnasee_50"])
        all_valid.append(data["valid"])
        all_chunk_sids.append(data["site_ids"])

    hand_46_all = np.concatenate(all_hand, axis=0)
    rnasee_50_all = np.concatenate(all_rnasee, axis=0)
    valid_all = np.concatenate(all_valid, axis=0)
    sids_all = np.concatenate(all_chunk_sids, axis=0)

    logger.info("Total features: %d sites, hand=%s, rnasee=%s, valid=%d",
                len(sids_all), hand_46_all.shape, rnasee_50_all.shape,
                valid_all.sum())

    return sids_all, hand_46_all, rnasee_50_all, valid_all


# ---------------------------------------------------------------------------
# Phase 3: Score all variants
# ---------------------------------------------------------------------------

def run_phase3(clinvar_df, sids_all, hand_46_all, rnasee_50_all, valid_all,
               gb_model, rf_model, output_dir):
    """Score all ClinVar variants with trained models."""
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 3: SCORE ALL CLINVAR VARIANTS")
    logger.info("=" * 70)

    # Build lookup from ClinVar DataFrame
    clinvar_lookup = {}
    for _, row in clinvar_df.iterrows():
        clinvar_lookup[str(row["site_id"])] = row

    # Score valid variants
    n_valid = int(valid_all.sum())
    logger.info("Scoring %d valid variants...", n_valid)

    valid_idx = np.where(valid_all)[0]
    hand_valid = hand_46_all[valid_idx]
    rnasee_valid = rnasee_50_all[valid_idx]
    sids_valid = sids_all[valid_idx]

    # Replace any NaN in features
    hand_valid = np.nan_to_num(hand_valid, nan=0.0)
    rnasee_valid = np.nan_to_num(rnasee_valid, nan=0.0)

    # Score with GB_Full
    logger.info("Scoring with GB_Full (46-dim)...")
    p_gb = gb_model.predict_proba(hand_valid)[:, 1]

    # Score with RNAsee_RF
    logger.info("Scoring with RNAsee_RF (50-bit)...")
    p_rf = rf_model.predict_proba(rnasee_valid)[:, 1]

    # Build results DataFrame
    logger.info("Building results DataFrame...")
    rows = []
    for i, sid in enumerate(sids_valid):
        sid_str = str(sid)
        if sid_str not in clinvar_lookup:
            continue
        row = clinvar_lookup[sid_str]
        rows.append({
            "site_id": sid_str,
            "chr": row["chr"],
            "start": row["start"],
            "gene": row.get("gene", ""),
            "clinical_significance": row.get("clinical_significance", ""),
            "significance_simple": row.get("significance_simple", "Other"),
            "condition": str(row.get("condition", "")),
            "is_known_editing_site": bool(row.get("is_known_editing_site", False)),
            "editing_dataset": str(row.get("editing_dataset", "")),
            "p_edited_gb": float(p_gb[i]),
            "p_edited_rnasee": float(p_rf[i]),
        })

    scores_df = pd.DataFrame(rows)

    # Save
    scores_path = output_dir / "clinvar_all_scores.csv"
    scores_df.to_csv(scores_path, index=False)
    logger.info("Saved %d scored variants to %s (%.1f MB)",
                len(scores_df), scores_path.name,
                scores_path.stat().st_size / 1e6)

    return scores_df


# ---------------------------------------------------------------------------
# Phase 4: Analysis and visualization
# ---------------------------------------------------------------------------

def run_phase4(scores_df, cv_results, output_dir):
    """Analysis and visualization of ClinVar prediction results."""
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 4: ANALYSIS AND VISUALIZATION")
    logger.info("=" * 70)

    results = {
        "cv_results": cv_results,
        "n_clinvar_scored": int(len(scores_df)),
    }

    # ---- Head-to-head comparison ----
    logger.info("\n--- Head-to-Head: GB_Full vs RNAsee_RF ---")

    for threshold in [0.5, 0.8]:
        gb_candidates = scores_df[scores_df["p_edited_gb"] >= threshold]
        rf_candidates = scores_df[scores_df["p_edited_rnasee"] >= threshold]
        both = scores_df[
            (scores_df["p_edited_gb"] >= threshold) &
            (scores_df["p_edited_rnasee"] >= threshold)
        ]
        gb_only = scores_df[
            (scores_df["p_edited_gb"] >= threshold) &
            (scores_df["p_edited_rnasee"] < threshold)
        ]
        rf_only = scores_df[
            (scores_df["p_edited_gb"] < threshold) &
            (scores_df["p_edited_rnasee"] >= threshold)
        ]

        logger.info("  Threshold >= %.1f:", threshold)
        logger.info("    GB_Full candidates: %d", len(gb_candidates))
        logger.info("    RNAsee_RF candidates: %d", len(rf_candidates))
        logger.info("    Both: %d", len(both))
        logger.info("    GB_Full only: %d", len(gb_only))
        logger.info("    RNAsee_RF only: %d", len(rf_only))

        results[f"candidates_t{threshold}"] = {
            "gb_full": int(len(gb_candidates)),
            "rnasee_rf": int(len(rf_candidates)),
            "both": int(len(both)),
            "gb_only": int(len(gb_only)),
            "rf_only": int(len(rf_only)),
        }

    # ---- Known site recovery ----
    known = scores_df[scores_df["is_known_editing_site"] == True]
    logger.info("\n--- Known Site Recovery (%d sites) ---", len(known))
    if len(known) > 0:
        gb_recall_50 = (known["p_edited_gb"] >= 0.5).mean()
        rf_recall_50 = (known["p_edited_rnasee"] >= 0.5).mean()
        gb_recall_80 = (known["p_edited_gb"] >= 0.8).mean()
        rf_recall_80 = (known["p_edited_rnasee"] >= 0.8).mean()
        logger.info("  GB_Full recall at P>=0.5: %.1f%% (%d/%d)",
                    100 * gb_recall_50, int(gb_recall_50 * len(known)), len(known))
        logger.info("  RNAsee_RF recall at P>=0.5: %.1f%% (%d/%d)",
                    100 * rf_recall_50, int(rf_recall_50 * len(known)), len(known))
        logger.info("  GB_Full recall at P>=0.8: %.1f%% (%d/%d)",
                    100 * gb_recall_80, int(gb_recall_80 * len(known)), len(known))
        logger.info("  RNAsee_RF recall at P>=0.8: %.1f%% (%d/%d)",
                    100 * rf_recall_80, int(rf_recall_80 * len(known)), len(known))

        results["known_site_recovery"] = {
            "n_known": int(len(known)),
            "gb_recall_50": float(gb_recall_50),
            "rf_recall_50": float(rf_recall_50),
            "gb_recall_80": float(gb_recall_80),
            "rf_recall_80": float(rf_recall_80),
            "gb_mean_score": float(known["p_edited_gb"].mean()),
            "rf_mean_score": float(known["p_edited_rnasee"].mean()),
        }

    # ---- Pathogenicity enrichment ----
    logger.info("\n--- Pathogenicity Enrichment ---")
    categories = ["Pathogenic", "Likely_pathogenic", "VUS", "Conflicting",
                   "Likely_benign", "Benign"]
    cat_scores = {}
    for cat in categories:
        mask = scores_df["significance_simple"] == cat
        if mask.sum() > 0:
            cat_scores[cat] = {
                "gb": scores_df[mask]["p_edited_gb"].values,
                "rf": scores_df[mask]["p_edited_rnasee"].values,
            }
            logger.info("  %s (n=%d): GB mean=%.4f  RNAsee mean=%.4f",
                        cat, mask.sum(),
                        np.mean(cat_scores[cat]["gb"]),
                        np.mean(cat_scores[cat]["rf"]))

    # Statistical tests
    if "Pathogenic" in cat_scores and "Benign" in cat_scores:
        for model_name, key in [("GB_Full", "gb"), ("RNAsee_RF", "rf")]:
            stat, p = mannwhitneyu(cat_scores["Pathogenic"][key],
                                   cat_scores["Benign"][key],
                                   alternative="two-sided")
            logger.info("  %s Pathogenic vs Benign: Mann-Whitney p=%.2e", model_name, p)
            results[f"{model_name}_path_vs_benign_p"] = float(p)

    # ---- Gene-level aggregation ----
    _analyze_gene_level(scores_df, output_dir, results)

    # ---- Pathogenic high-score candidates ----
    _analyze_pathogenic_candidates(scores_df, output_dir, results)

    # ---- Plots ----
    _create_plots(scores_df, cat_scores, known, output_dir)

    # Save results JSON
    with open(output_dir / "clinvar_prediction_results.json", "w") as f:
        json.dump(results, f, indent=2, default=_serialize)

    return results


def _analyze_gene_level(scores_df, output_dir, results):
    """Gene-level aggregation of editing scores."""
    logger.info("\n--- Gene-Level Analysis ---")

    gene_stats = scores_df.groupby("gene").agg(
        n_variants=("p_edited_gb", "count"),
        mean_p_gb=("p_edited_gb", "mean"),
        max_p_gb=("p_edited_gb", "max"),
        mean_p_rf=("p_edited_rnasee", "mean"),
        n_pathogenic=("significance_simple",
                      lambda x: x.isin(["Pathogenic", "Likely_pathogenic"]).sum()),
        n_high_gb=("p_edited_gb", lambda x: (x >= 0.5).sum()),
        n_high_rf=("p_edited_rnasee", lambda x: (x >= 0.5).sum()),
    ).reset_index()

    gene_stats = gene_stats[gene_stats["n_variants"] >= 3].sort_values(
        "mean_p_gb", ascending=False)

    # Genes with pathogenic variants AND high editing scores
    interesting = gene_stats[
        (gene_stats["n_pathogenic"] > 0) & (gene_stats["n_high_gb"] > 0)
    ]
    logger.info("Genes with pathogenic + high-scoring variants: %d", len(interesting))
    if len(interesting) > 0:
        for _, row in interesting.head(10).iterrows():
            logger.info("  %s: n=%d, %d pathogenic, %d high-score, mean_P_gb=%.3f",
                        row["gene"], row["n_variants"], row["n_pathogenic"],
                        row["n_high_gb"], row["mean_p_gb"])

    gene_stats.to_csv(output_dir / "gene_level_scores.csv", index=False)
    results["n_genes_analyzed"] = int(len(gene_stats))
    results["n_genes_pathogenic_high"] = int(len(interesting))


def _analyze_pathogenic_candidates(scores_df, output_dir, results):
    """Identify pathogenic variants with high editing scores."""
    logger.info("\n--- Pathogenic Editing Candidates ---")

    pathogenic = scores_df[
        scores_df["significance_simple"].isin(["Pathogenic", "Likely_pathogenic"])
    ]

    for threshold in [0.5, 0.8]:
        gb_high = pathogenic[pathogenic["p_edited_gb"] >= threshold]
        rf_high = pathogenic[pathogenic["p_edited_rnasee"] >= threshold]
        logger.info("  Pathogenic with P>=%s: GB=%d, RNAsee=%d",
                    threshold, len(gb_high), len(rf_high))

        if threshold == 0.5:
            results["pathogenic_candidates_gb_50"] = int(len(gb_high))
            results["pathogenic_candidates_rf_50"] = int(len(rf_high))

    # Save top pathogenic candidates
    high_path = pathogenic[
        (pathogenic["p_edited_gb"] >= 0.5) | (pathogenic["p_edited_rnasee"] >= 0.5)
    ].sort_values("p_edited_gb", ascending=False)

    if len(high_path) > 0:
        high_path.to_csv(output_dir / "pathogenic_candidates.csv", index=False)
        logger.info("  Saved %d pathogenic candidates to pathogenic_candidates.csv",
                    len(high_path))

        logger.info("\n  Top 15 pathogenic editing candidates (by GB_Full):")
        for _, row in high_path.head(15).iterrows():
            logger.info("    %s %s:%s | GB=%.3f RF=%.3f | %s | %s",
                        row["gene"], row["chr"], row["start"],
                        row["p_edited_gb"], row["p_edited_rnasee"],
                        row["clinical_significance"],
                        str(row["condition"])[:60])


def _create_plots(scores_df, cat_scores, known, output_dir):
    """Create all visualization plots."""
    colors = {"Pathogenic": "#d32f2f", "Likely_pathogenic": "#ff7043",
              "VUS": "#ffb74d", "Conflicting": "#90a4ae",
              "Likely_benign": "#81c784", "Benign": "#43a047"}
    categories = ["Pathogenic", "Likely_pathogenic", "VUS", "Conflicting",
                   "Likely_benign", "Benign"]

    # ---- Plot 1: Score distributions by significance ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plot_cats = [c for c in categories if c in cat_scores]

    for ax_idx, (model_key, model_name) in enumerate([("gb", "GB_Full"), ("rf", "RNAsee_RF")]):
        plot_data = [cat_scores[c][model_key] for c in plot_cats]
        bp = axes[ax_idx].boxplot(plot_data, labels=plot_cats, patch_artist=True, showfliers=False)
        for patch, cat in zip(bp["boxes"], plot_cats):
            patch.set_facecolor(colors.get(cat, "#bdbdbd"))
            patch.set_alpha(0.7)
        axes[ax_idx].set_ylabel("P(edited)")
        axes[ax_idx].set_title(f"{model_name}: Editing Probability by Clinical Significance")
        axes[ax_idx].tick_params(axis="x", rotation=45)
        for i, cat in enumerate(plot_cats):
            n = len(cat_scores[cat][model_key])
            axes[ax_idx].text(i + 1, axes[ax_idx].get_ylim()[0] - 0.02,
                              f"n={n}", ha="center", fontsize=7, color="gray")

    plt.tight_layout()
    plt.savefig(output_dir / "score_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ---- Plot 2: Model comparison (GB vs RNAsee scatter) ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter of GB vs RNAsee scores
    for cat in plot_cats:
        mask = scores_df["significance_simple"] == cat
        if mask.sum() > 0:
            # Subsample for plotting if too many
            cat_df = scores_df[mask]
            if len(cat_df) > 5000:
                cat_df = cat_df.sample(5000, random_state=SEED)
            axes[0].scatter(cat_df["p_edited_gb"], cat_df["p_edited_rnasee"],
                            alpha=0.15, s=3, color=colors.get(cat, "#bdbdbd"),
                            label=f"{cat} (n={mask.sum()})")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1)
    axes[0].set_xlabel("GB_Full P(edited)")
    axes[0].set_ylabel("RNAsee_RF P(edited)")
    axes[0].set_title("Model Score Comparison")
    axes[0].legend(fontsize=6, markerscale=5)
    axes[0].grid(True, alpha=0.2)

    # Venn-style bar chart at P>=0.5
    gb_set = set(scores_df[scores_df["p_edited_gb"] >= 0.5]["site_id"])
    rf_set = set(scores_df[scores_df["p_edited_rnasee"] >= 0.5]["site_id"])
    both_n = len(gb_set & rf_set)
    gb_only_n = len(gb_set - rf_set)
    rf_only_n = len(rf_set - gb_set)
    bars = [gb_only_n, both_n, rf_only_n]
    bar_labels = ["GB only", "Both", "RNAsee only"]
    bar_colors = ["#1976D2", "#7B1FA2", "#F57C00"]
    axes[1].bar(bar_labels, bars, color=bar_colors, alpha=0.8, edgecolor="white")
    axes[1].set_ylabel("Number of Candidates")
    axes[1].set_title("Candidate Overlap at P(edited) >= 0.5")
    for i, (v, lbl) in enumerate(zip(bars, bar_labels)):
        axes[1].text(i, v + max(bars) * 0.02, str(v), ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ---- Plot 3: Pathogenicity enrichment ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax_idx, (model_key, col_name, model_name) in enumerate([
        ("gb", "p_edited_gb", "GB_Full"),
        ("rf", "p_edited_rnasee", "RNAsee_RF"),
    ]):
        for cat in plot_cats:
            sorted_vals = np.sort(cat_scores[cat][model_key])
            cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
            axes[ax_idx].plot(sorted_vals, cdf,
                              label=f"{cat} (n={len(sorted_vals)})",
                              color=colors.get(cat, "#bdbdbd"), linewidth=1.5)
        axes[ax_idx].axvline(0.5, color="red", linestyle="--", alpha=0.5, label="P=0.5")
        axes[ax_idx].set_xlabel(f"{model_name} P(edited)")
        axes[ax_idx].set_ylabel("Cumulative Fraction")
        axes[ax_idx].set_title(f"{model_name}: Score CDF by Significance")
        axes[ax_idx].legend(fontsize=7, loc="upper left")
        axes[ax_idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "pathogenicity_enrichment.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ---- Plot 4: Known site analysis ----
    if len(known) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # P(edited) distributions for known sites
        axes[0].hist(known["p_edited_gb"], bins=30, alpha=0.6, color="#1976D2",
                     label="GB_Full", edgecolor="white")
        axes[0].hist(known["p_edited_rnasee"], bins=30, alpha=0.6, color="#F57C00",
                     label="RNAsee_RF", edgecolor="white")
        axes[0].axvline(0.5, color="red", linestyle="--", alpha=0.7, label="P=0.5")
        axes[0].set_xlabel("P(edited)")
        axes[0].set_ylabel("Count")
        axes[0].set_title(f"Known Editing Sites (n={len(known)})")
        axes[0].legend()

        # Clinical significance breakdown of known sites
        sig_counts = known["significance_simple"].value_counts()
        sig_order = ["Pathogenic", "Likely_pathogenic", "VUS", "Conflicting",
                     "Likely_benign", "Benign", "Other"]
        sig_vals = [sig_counts.get(s, 0) for s in sig_order]
        sig_colors = [colors.get(s, "#bdbdbd") for s in sig_order]
        axes[1].bar(range(len(sig_order)), sig_vals, color=sig_colors)
        axes[1].set_xticks(range(len(sig_order)))
        axes[1].set_xticklabels(sig_order, rotation=45, ha="right", fontsize=8)
        axes[1].set_ylabel("Count")
        axes[1].set_title(f"Known Editing Sites in ClinVar: Clinical Significance")

        plt.tight_layout()
        plt.savefig(output_dir / "known_editing_sites_clinvar.png", dpi=150, bbox_inches="tight")
        plt.close()

    logger.info("All plots saved to %s", output_dir)


def _serialize(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return str(obj)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.random.seed(SEED)
    t_total = time.time()

    # -----------------------------------------------------------------------
    # Load data and features
    # -----------------------------------------------------------------------
    sequences, structure_delta, loop_df, baseline_struct = load_known_features()
    train_df = load_training_data()

    # -----------------------------------------------------------------------
    # Phase 1: Train models
    # -----------------------------------------------------------------------
    gb_model, rf_model, cv_results = run_phase1(
        train_df, sequences, structure_delta, loop_df, baseline_struct
    )
    if gb_model is None:
        logger.error("Model training failed. Exiting.")
        return

    # -----------------------------------------------------------------------
    # Phase 2: Compute features for ClinVar
    # -----------------------------------------------------------------------
    clinvar_df = load_clinvar_data()
    if clinvar_df is None:
        return

    sids_all, hand_46_all, rnasee_50_all, valid_all = run_phase2(clinvar_df, OUTPUT_DIR)

    # -----------------------------------------------------------------------
    # Phase 3: Score all variants
    # -----------------------------------------------------------------------
    scores_df = run_phase3(
        clinvar_df, sids_all, hand_46_all, rnasee_50_all, valid_all,
        gb_model, rf_model, OUTPUT_DIR
    )

    # -----------------------------------------------------------------------
    # Phase 4: Analysis and visualization
    # -----------------------------------------------------------------------
    results = run_phase4(scores_df, cv_results, OUTPUT_DIR)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    elapsed = time.time() - t_total
    logger.info("\n" + "=" * 70)
    logger.info("COMPLETE (%.1f min)", elapsed / 60)
    logger.info("=" * 70)
    logger.info("Output directory: %s", OUTPUT_DIR)
    logger.info("Key files:")
    logger.info("  clinvar_all_scores.csv - All ~1.68M variants scored by GB_Full + RNAsee_RF")
    logger.info("  clinvar_prediction_results.json - Summary statistics + comparison")
    logger.info("  score_distributions.png - P(edited) by clinical significance")
    logger.info("  model_comparison.png - GB vs RNAsee candidate overlap")
    logger.info("  pathogenicity_enrichment.png - Editing vs pathogenicity")
    logger.info("  known_editing_sites_clinvar.png - Known site analysis")
    logger.info("  gene_level_scores.csv - Gene-level aggregation")
    logger.info("  pathogenic_candidates.csv - Pathogenic variants with high editing scores")


if __name__ == "__main__":
    main()
