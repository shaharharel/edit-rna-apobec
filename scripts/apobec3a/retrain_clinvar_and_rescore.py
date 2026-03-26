#!/usr/bin/env python
"""Retrain ClinVar models with fixed training data and re-score using existing chunks.

Fixes the bug where Asaoka all-C negatives had zero feature vectors because their
site IDs were not in site_sequences.json. The fix: extract sequences from the genome
for these negatives before training.

Usage:
    conda run -n quris python scripts/apobec3a/retrain_clinvar_and_rescore.py
"""

import gc
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths (same as exp_clinvar_prediction.py)
# ---------------------------------------------------------------------------
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "apobec3a" / "outputs" / "clinvar_prediction"
INTERMEDIATE_DIR = OUTPUT_DIR / "intermediate"
SPLITS_A3A_CSV = PROJECT_ROOT / "data" / "processed" / "splits_expanded_a3a.csv"
ALLC_NEG_CSV = PROJECT_ROOT / "data" / "processed" / "negatives_per_dataset_all_c.csv"
SEQ_JSON = PROJECT_ROOT / "data" / "processed" / "site_sequences.json"
STRUCT_CACHE = PROJECT_ROOT / "data" / "processed" / "embeddings" / "vienna_structure_cache.npz"
LOOP_POS_CSV = PROJECT_ROOT / "experiments" / "apobec3a" / "outputs" / "loop_position" / "loop_position_per_site.csv"
GENOME_HG38 = PROJECT_ROOT / "data" / "raw" / "genomes" / "hg38.fa"
GENOME_HG19 = PROJECT_ROOT / "data" / "raw" / "genomes" / "hg19.fa"

SEED = 42
CENTER = 100


# ---------------------------------------------------------------------------
# Feature extraction (copied from exp_clinvar_prediction.py)
# ---------------------------------------------------------------------------
def extract_neg_sequences_from_genome(neg_df, sequences):
    """Extract 201nt sequences for negative sites from genome."""
    genome_path = GENOME_HG38 if GENOME_HG38.exists() else GENOME_HG19
    if not genome_path.exists():
        logger.warning("Genome not found, negatives will have zero features")
        return

    try:
        from pyfaidx import Fasta
        genome = Fasta(str(genome_path))
    except ImportError:
        logger.warning("pyfaidx not available")
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

    logger.info("  Extracted %d negative sequences from genome", extracted)


def extract_motif_features(sequences, site_ids):
    """24-dim motif features (same as exp_clinvar_prediction.py)."""
    features = []
    for sid in site_ids:
        seq = sequences.get(str(sid), "N" * 201).upper().replace("T", "U")
        ep = CENTER
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


def encode_rnasee_features(sequences, site_ids):
    """50-bit RNAsee features (15nt up + 10nt down, 2 bits each = 50 bits)."""
    features = []
    for sid in site_ids:
        seq = sequences.get(str(sid), "N" * 201).upper().replace("T", "U")
        ep = CENTER
        upstream = seq[max(0, ep - 15):ep]
        downstream = seq[ep + 1:ep + 11]
        upstream = upstream.rjust(15, "N")
        downstream = downstream.ljust(10, "N")
        feat = []
        for nt in upstream + downstream:
            feat.extend([1 if nt == "A" else 0, 1 if nt == "C" else 0])
        features.append(feat[:50])
    return np.array(features, dtype=np.float32)


def build_hand_46(site_ids, sequences, structure_delta, loop_df, baseline_struct):
    """Build 46-dim hand features: motif(24)+struct_delta(7)+loop(9)+baseline(6)."""
    loop_cols = [
        "is_unpaired", "loop_size", "dist_to_junction", "dist_to_apex",
        "relative_loop_position", "left_stem_length", "right_stem_length",
        "max_adjacent_stem_length", "local_unpaired_fraction",
    ]
    features = []
    for sid in site_ids:
        sid_str = str(sid)
        seq = sequences.get(sid_str, "N" * 201)
        motif = extract_motif_features({sid_str: seq}, [sid_str])[0]
        struct = structure_delta.get(sid_str, np.zeros(7, dtype=np.float32))
        baseline = baseline_struct.get(sid_str, np.zeros(6, dtype=np.float32))
        if sid_str in loop_df.index:
            loop = loop_df.loc[sid_str, loop_cols].values.astype(np.float32)
        else:
            loop = np.zeros(len(loop_cols), dtype=np.float32)
        hand = np.concatenate([motif, struct, loop, baseline])
        hand = np.nan_to_num(hand, nan=0.0)
        features.append(hand)
    return np.array(features, dtype=np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    logger.info("=" * 70)
    logger.info("RETRAIN ClinVar models with fixed training data")
    logger.info("=" * 70)

    # Load pre-computed caches
    logger.info("Loading pre-computed caches...")
    sequences = {}
    if SEQ_JSON.exists():
        with open(SEQ_JSON) as f:
            sequences = json.load(f)
    logger.info("  %d sequences from cache", len(sequences))

    structure_delta = {}
    baseline_struct = {}
    if STRUCT_CACHE.exists():
        data = np.load(STRUCT_CACHE, allow_pickle=True)
        sids = data["site_ids"]
        feats = data["delta_features"]
        structure_delta = {str(s): feats[i].astype(np.float32) for i, s in enumerate(sids)}
        pp = data["pairing_probs"]
        acc = data["accessibilities"]
        ent = data["entropies"] if "entropies" in data else None
        mfes = data["mfes"]
        for i, s in enumerate(sids):
            sid_str = str(s)
            ep = CENTER
            w = 10
            baseline = np.array([
                pp[i][ep], acc[i][ep],
                ent[i][ep] if ent is not None else 0.0,
                float(mfes[i]),
                float(pp[i][max(0, ep-w):ep+w+1].mean()),
                float(acc[i][max(0, ep-w):ep+w+1].mean()),
            ], dtype=np.float32)
            baseline_struct[sid_str] = baseline
        del data; gc.collect()
    logger.info("  %d structure delta features", len(structure_delta))

    loop_df = pd.DataFrame()
    if LOOP_POS_CSV.exists():
        loop_df = pd.read_csv(LOOP_POS_CSV)
        loop_df["site_id"] = loop_df["site_id"].astype(str)
        loop_df = loop_df.set_index("site_id")
    logger.info("  %d loop features", len(loop_df))

    # Load training data: A3A positives + hybrid negatives matching TC rate.
    # Problem with tier2/tier3-only negatives: they are 99.9% TC while positives
    # are 86.1% TC → model inverts TC signal (predicts non-TC ClinVar as positive).
    # Problem with Asaoka all-C only: 24.4% TC → trivially separable, AUROC≈1.
    # Solution: combine tier2/tier3 (all TC) + Asaoka all-C non-TC negatives to
    # match positive TC rate (86.1%). This neutralizes TC as a discriminating
    # feature, forcing the model to learn structure. The model will correctly
    # assign higher probability to TC+good-structure ClinVar variants.
    logger.info("\nLoading training data...")
    a3a_df = pd.read_csv(SPLITS_A3A_CSV)
    pos_df = a3a_df[a3a_df["is_edited"] == 1].copy()
    pos_df["label"] = 1
    logger.info("  A3A positives: %d", len(pos_df))

    # tier2/tier3 TC negatives
    tier_neg_df = a3a_df[a3a_df["is_edited"] == 0][["site_id", "chr", "start", "end", "strand", "dataset_source"]].copy()
    tier_neg_df["label"] = 0

    # Target: same TC fraction in negatives as positives (86.1%)
    # All tier2/tier3 are TC (99.9%). Add non-TC Asaoka all-C negatives to dilute.
    pos_tc_frac = 0.861  # from training data
    n_tier = len(tier_neg_df)  # 2966 TC negatives
    # n_tier / (n_tier + n_nontc) = pos_tc_frac → n_nontc = n_tier/pos_tc_frac - n_tier
    n_nontc_target = int(n_tier / pos_tc_frac - n_tier)
    logger.info("  TC-context negatives (tier2/tier3): %d", n_tier)
    logger.info("  Target non-TC negatives to add: %d (to match %.1f%% TC in negatives)", n_nontc_target, pos_tc_frac * 100)

    allc_df = pd.read_csv(ALLC_NEG_CSV)
    asaoka_allc = allc_df[allc_df["dataset_source"] == "asaoka_2019"].copy()

    # Extract sequences for Asaoka all-C negatives
    asaoka_missing = [str(sid) for sid in asaoka_allc["site_id"] if str(sid) not in sequences]
    if asaoka_missing:
        logger.info("  Extracting %d asaoka all-C sequences from genome...", len(asaoka_missing))
        extract_neg_sequences_from_genome(asaoka_allc, sequences)

    # Filter to non-TC asaoka sites (to dilute TC fraction in negatives)
    asaoka_allc["site_id_str"] = asaoka_allc["site_id"].astype(str)
    nontc_mask = []
    for sid in asaoka_allc["site_id_str"]:
        seq = sequences.get(sid, "N" * 201)
        center = 100
        is_tc = len(seq) > center and seq[center-1] in "TU" and seq[center] == "C"
        nontc_mask.append(not is_tc)
    asaoka_nontc = asaoka_allc[nontc_mask].copy()
    logger.info("  Asaoka all-C non-TC sites: %d", len(asaoka_nontc))

    if len(asaoka_nontc) > n_nontc_target:
        asaoka_nontc = asaoka_nontc.sample(n=n_nontc_target, random_state=SEED)
    asaoka_nontc["label"] = 0
    logger.info("  Using %d non-TC Asaoka negatives", len(asaoka_nontc))

    neg_df = pd.concat([tier_neg_df, asaoka_nontc], ignore_index=True)
    logger.info("  Total negatives: %d", len(neg_df))

    # Build feature matrices
    logger.info("\nBuilding feature matrices...")
    all_site_ids = list(pos_df["site_id"].astype(str)) + list(neg_df["site_id"].astype(str))
    all_labels = np.array(
        list(pos_df["label"].values) + list(neg_df["label"].values), dtype=int
    )

    hand_46 = build_hand_46(all_site_ids, sequences, structure_delta, loop_df, baseline_struct)
    rnasee_50 = encode_rnasee_features(sequences, all_site_ids)
    logger.info("  hand_46 shape: %s", hand_46.shape)
    logger.info("  rnasee_50 shape: %s", rnasee_50.shape)

    # Check motif distribution
    tc_pos = float((hand_46[all_labels == 1, 0] > 0.5).mean())
    tc_neg = float((hand_46[all_labels == 0, 0] > 0.5).mean())
    logger.info("  5p_UC (TC motif): pos=%.1f%%, neg=%.1f%%", tc_pos * 100, tc_neg * 100)

    # 5-fold CV
    logger.info("\nRunning 5-fold CV...")
    from sklearn.model_selection import KFold
    from sklearn.metrics import roc_auc_score
    from xgboost import XGBClassifier
    from sklearn.ensemble import RandomForestClassifier

    n_pos = int((all_labels == 1).sum())
    n_neg = int((all_labels == 0).sum())
    logger.info("  n_pos=%d, n_neg=%d, scale_pos_weight=%.2f", n_pos, n_neg, n_neg / max(n_pos, 1))

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    gb_aucs, rf_aucs = [], []
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(all_site_ids)):
        np.random.seed(SEED + fold_idx)
        n_remain = len(train_idx)
        inner_perm = np.random.RandomState(SEED + fold_idx).permutation(n_remain)
        n_val = int(n_remain * 0.2)
        val_inner = train_idx[inner_perm[:n_val]]
        train_inner = train_idx[inner_perm[n_val:]]

        X_tr_h = hand_46[train_inner]; y_tr = all_labels[train_inner]
        X_val_h = hand_46[val_inner]; y_val = all_labels[val_inner]
        X_te_h = hand_46[test_idx]; y_te = all_labels[test_idx]

        X_tr_r = rnasee_50[train_inner]
        X_te_r = rnasee_50[test_idx]

        gb = XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.05,
                           subsample=0.8, min_child_weight=10,
                           scale_pos_weight=n_neg / max(n_pos, 1),
                           random_state=SEED + fold_idx, n_jobs=4,
                           early_stopping_rounds=30, verbosity=0, eval_metric="logloss")
        gb.fit(X_tr_h, y_tr, eval_set=[(X_val_h, y_val)], verbose=False)
        gb_auc = roc_auc_score(y_te, gb.predict_proba(X_te_h)[:, 1])
        gb_aucs.append(gb_auc)

        rf = RandomForestClassifier(n_estimators=500, max_depth=None, random_state=SEED + fold_idx,
                                    n_jobs=4, class_weight="balanced")
        rf.fit(X_tr_r, y_tr)
        rf_auc = roc_auc_score(y_te, rf.predict_proba(X_te_r)[:, 1])
        rf_aucs.append(rf_auc)

        logger.info("  Fold %d: GB AUROC=%.4f  RF AUROC=%.4f", fold_idx + 1, gb_auc, rf_auc)

    logger.info("  GB mean AUROC: %.4f ± %.4f", np.mean(gb_aucs), np.std(gb_aucs))
    logger.info("  RF mean AUROC: %.4f ± %.4f", np.mean(rf_aucs), np.std(rf_aucs))

    # Train final models on all data
    logger.info("\nTraining final models on all data...")
    gb_final = XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.05,
                              subsample=0.8, min_child_weight=10,
                              scale_pos_weight=n_neg / max(n_pos, 1),
                              random_state=SEED, n_jobs=4, verbosity=0, eval_metric="logloss")
    gb_final.fit(hand_46, all_labels)

    rf_final = RandomForestClassifier(n_estimators=500, max_depth=None,
                                       random_state=SEED, n_jobs=4, class_weight="balanced")
    rf_final.fit(rnasee_50, all_labels)
    logger.info("  Models trained.")

    # Re-score ClinVar using existing intermediate chunks
    logger.info("\nRe-scoring ClinVar from existing intermediate chunks...")
    chunk_files = sorted(INTERMEDIATE_DIR.glob("chunk_*.npz"))
    logger.info("  Found %d chunks", len(chunk_files))

    # Load original clinvar_all_scores to get metadata
    orig_scores = pd.read_csv(OUTPUT_DIR / "clinvar_all_scores.csv")
    orig_lookup = {str(r["site_id"]): r for _, r in orig_scores.iterrows()}
    logger.info("  Original scores: %d variants", len(orig_scores))

    all_rows = []
    for chunk_path in chunk_files:
        chunk = np.load(str(chunk_path), allow_pickle=True)
        sids = chunk["site_ids"]
        hand_c = chunk["hand_46"].astype(np.float32)
        rnasee_c = chunk["rnasee_50"].astype(np.float32)
        valid_c = chunk["valid"]
        hand_c = np.nan_to_num(hand_c, nan=0.0)
        rnasee_c = np.nan_to_num(rnasee_c, nan=0.0)
        valid_idx = np.where(valid_c)[0]
        if len(valid_idx) == 0:
            continue
        sids_v = sids[valid_idx]
        p_gb = gb_final.predict_proba(hand_c[valid_idx])[:, 1]
        p_rf = rf_final.predict_proba(rnasee_c[valid_idx])[:, 1]
        for i, sid in enumerate(sids_v):
            sid_str = str(sid)
            if sid_str not in orig_lookup:
                continue
            row = orig_lookup[sid_str]
            all_rows.append({
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

    new_scores = pd.DataFrame(all_rows)
    logger.info("  Re-scored %d variants", len(new_scores))

    # Save
    out_path = OUTPUT_DIR / "clinvar_all_scores.csv"
    new_scores.to_csv(out_path, index=False)
    logger.info("  Saved to %s (%.1f MB)", out_path.name, out_path.stat().st_size / 1e6)

    # Quick enrichment check
    definitive = new_scores[new_scores["significance_simple"].isin(["Pathogenic", "Likely_pathogenic", "Benign", "Likely_benign"])]
    pathogenic = definitive[definitive["significance_simple"].isin(["Pathogenic", "Likely_pathogenic"])]
    benign = definitive[definitive["significance_simple"].isin(["Benign", "Likely_benign"])]
    for thresh in [0.5, 0.8]:
        gb_pos = new_scores[new_scores["p_edited_gb"] >= thresh]
        n_gb = len(gb_pos)
        path_in_gb = definitive[definitive["site_id"].isin(gb_pos["site_id"]) & definitive["significance_simple"].isin(["Pathogenic", "Likely_pathogenic"])]
        from scipy.stats import chi2_contingency
        n_path = len(pathogenic); n_ben = len(benign)
        n_path_gb = len(path_in_gb)
        bg_path_frac = n_path / max(n_path + n_ben, 1)
        if n_gb > 0:
            gb_path_frac = n_path_gb / max(definitive["site_id"].isin(gb_pos["site_id"]).sum(), 1)
            logger.info("  GB P>=%.1f: n=%d, path%%=%.1f%%, bg%%=%.1f%%",
                        thresh, n_gb, gb_path_frac * 100, bg_path_frac * 100)
        else:
            logger.info("  GB P>=%.1f: n=0", thresh)

    logger.info("\nDone! Run exp_clinvar_calibrated.py and generate_html_report.py next.")


if __name__ == "__main__":
    main()
