#!/usr/bin/env python3
"""Multi-class enzyme-tissue-disease triangle.

Trains ONE unified multi-class XGBoost that outputs per-enzyme editing
probabilities (A3A, A3B, A3G, A3A_A3G, Neither, Unknown, Negative), then
scores TCGA cancer mutations and matched controls.

Key test: which enzyme's P(enzyme) best predicts mutations in which cancer?
- P(A3A) should predict APOBEC-mutated cancers (BLCA, CESC, LUSC, BRCA)
- P(Neither/APOBEC1) should predict GI cancers (ESCA, STAD, LIHC)
- P(A3G) may predict broadly due to structure dominance

Usage:
    conda run -n quris python experiments/multi_enzyme/exp_multiclass_enzyme_tcga.py
"""

import gc
import gzip
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyfaidx import Fasta
from scipy import stats
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.apobec_feature_extraction import (
    extract_motif_from_seq,
    LOOP_FEATURE_COLS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# =============================================================================
# Paths
# =============================================================================
DATA_DIR = PROJECT_ROOT / "data"
HG19_FA = DATA_DIR / "raw/genomes/hg19.fa"
REFGENE_HG19 = DATA_DIR / "raw/genomes/refGene_hg19.txt"
MULTI_SPLITS = DATA_DIR / "processed/multi_enzyme/splits_multi_enzyme_v3_with_negatives.csv"
MULTI_SEQS = DATA_DIR / "processed/multi_enzyme/multi_enzyme_sequences_v3_with_negatives.json"
LOOP_CSV = DATA_DIR / "processed/multi_enzyme/loop_position_per_site_v3.csv"
STRUCT_CACHE = DATA_DIR / "processed/multi_enzyme/structure_cache_multi_enzyme_v3.npz"

TCGA_DIR = DATA_DIR / "raw/tcga"
VIENNA_CACHE_DIR = PROJECT_ROOT / "experiments/multi_enzyme/outputs/tcga_gnomad/vienna_cache"
OUTPUT_DIR = PROJECT_ROOT / "experiments/multi_enzyme/outputs/tcga_gnomad/enzyme_probability"

SEED = 42
N_CONTROLS = 5

# Class labels for the multi-class model
ENZYME_CLASSES = ["A3A", "A3B", "A3G", "A3A_A3G", "Neither", "Unknown", "Negative"]
CLASS_TO_IDX = {c: i for i, c in enumerate(ENZYME_CLASSES)}

# Cancers with ViennaRNA caches
CANCERS = ["blca", "brca", "cesc", "lusc", "hnsc", "skcm", "esca", "stad", "lihc"]


# =============================================================================
# Feature extraction from cached fold results (same as exp_enzyme_tissue_disease)
# =============================================================================

def derive_features_from_fold(fold_result):
    """Derive 16-dim structure features (7 delta + 9 loop) from cached fold data."""
    if fold_result is None:
        return None

    center = 100
    pair_wt = fold_result["bpp_wt_center"]
    pair_ed = fold_result["bpp_ed_center"]
    delta_pairing = pair_ed - pair_wt
    delta_accessibility = -delta_pairing
    delta_mfe = fold_result["mfe_ed"] - fold_result["mfe_wt"]

    bpp_wt_w = fold_result["bpp_wt_window"]
    bpp_ed_w = fold_result["bpp_ed_window"]

    def _entropy_from_p(p_val):
        if p_val <= 0 or p_val >= 1:
            return 0.0
        return -(p_val * np.log2(p_val + 1e-10) + (1 - p_val) * np.log2(1 - p_val + 1e-10))

    delta_entropy = _entropy_from_p(pair_ed) - _entropy_from_p(pair_wt)
    delta_window = np.array(bpp_ed_w) - np.array(bpp_wt_w)
    mean_delta_pairing = float(np.mean(delta_window)) if len(delta_window) > 0 else 0.0
    std_delta_pairing = float(np.std(delta_window)) if len(delta_window) > 0 else 0.0

    struct_delta = np.array([
        delta_pairing, delta_accessibility, delta_entropy,
        delta_mfe, mean_delta_pairing, std_delta_pairing, -mean_delta_pairing
    ], dtype=np.float32)

    # Loop geometry (9-dim) from WT structure
    struct_wt = fold_result["struct_wt"]
    is_unpaired = 1.0 if struct_wt[center] == "." else 0.0

    loop_size = 0.0
    dist_to_junction = 0.0
    dist_to_apex = 0.0
    relative_loop_position = 0.5
    left_stem = 0.0
    right_stem = 0.0
    max_adj_stem = 0.0
    local_unpaired = 0.0

    if is_unpaired:
        left = center
        while left > 0 and struct_wt[left] == ".":
            left -= 1
        right = center
        while right < len(struct_wt) - 1 and struct_wt[right] == ".":
            right += 1

        loop_size = float(right - left - 1)
        if loop_size > 0:
            pos_in_loop = center - left - 1
            relative_loop_position = pos_in_loop / max(loop_size - 1, 1)
            dist_to_apex = abs(pos_in_loop - (loop_size - 1) / 2)

        ls = 0
        i = left
        while i >= 0 and struct_wt[i] in "()":
            ls += 1
            i -= 1
        left_stem = float(ls)

        rs = 0
        i = right
        while i < len(struct_wt) and struct_wt[i] in "()":
            rs += 1
            i += 1
        right_stem = float(rs)
        max_adj_stem = max(left_stem, right_stem)

    local_region = struct_wt[max(0, center - 10):min(len(struct_wt), center + 11)]
    local_unpaired = sum(1 for ch in local_region if ch == ".") / max(len(local_region), 1)

    loop_features = np.array([
        is_unpaired, loop_size, dist_to_junction, dist_to_apex,
        relative_loop_position, left_stem, right_stem,
        max_adj_stem, local_unpaired
    ], dtype=np.float32)

    return {"struct_delta": struct_delta, "loop_features": loop_features}


def build_40dim_features(sequences, fold_results):
    """Build 40-dim feature matrix from sequences + fold results."""
    n = len(sequences)
    features = np.zeros((n, 40), dtype=np.float32)
    for i, (seq, fold_raw) in enumerate(zip(sequences, fold_results)):
        if seq is not None and len(seq) == 201:
            features[i, :24] = extract_motif_from_seq(seq)
        derived = derive_features_from_fold(fold_raw)
        if derived is not None:
            features[i, 24:31] = derived["struct_delta"]
            features[i, 31:40] = derived["loop_features"]
    return np.nan_to_num(features, nan=0.0)


def build_40dim_from_training(splits, sequences_dict, loop_df, struct_data):
    """Build 40-dim features for training data from splits + caches."""
    site_ids = splits["site_id"].astype(str).tolist()

    # Motif (24-dim)
    motif_feats = np.array(
        [extract_motif_from_seq(sequences_dict.get(sid, "N" * 201)) for sid in site_ids],
        dtype=np.float32,
    )

    # Structure delta (7-dim)
    struct_feats = np.zeros((len(site_ids), 7), dtype=np.float32)
    if struct_data is not None:
        struct_sids = [str(s) for s in struct_data["site_ids"]]
        sid_to_idx = {s: i for i, s in enumerate(struct_sids)}
        for i, sid in enumerate(site_ids):
            if sid in sid_to_idx:
                struct_feats[i] = struct_data["delta_features"][sid_to_idx[sid]]

    # Loop geometry (9-dim)
    loop_feats = np.zeros((len(site_ids), len(LOOP_FEATURE_COLS)), dtype=np.float32)
    for i, sid in enumerate(site_ids):
        if sid in loop_df.index:
            for j, col in enumerate(LOOP_FEATURE_COLS):
                if col in loop_df.columns:
                    val = loop_df.loc[sid, col]
                    if not isinstance(val, (int, float)):
                        val = val.iloc[0] if hasattr(val, "iloc") else 0.0
                    loop_feats[i, j] = float(val) if pd.notna(val) else 0.0

    X = np.concatenate([motif_feats, struct_feats, loop_feats], axis=1)
    return np.nan_to_num(X, nan=0.0)


# =============================================================================
# TCGA mutation parsing and sequence extraction
# (same as exp_enzyme_tissue_disease.py -- must use same SEED for cache match)
# =============================================================================

def parse_exons(refgene_path):
    """Parse refGene for coding exonic regions."""
    exons_by_gene = defaultdict(list)
    seen = set()
    with open(refgene_path) as f:
        for line in f:
            fields = line.strip().split("\t")
            if len(fields) < 13:
                continue
            chrom = fields[2]
            strand = fields[3]
            cds_s, cds_e = int(fields[6]), int(fields[7])
            if cds_s == cds_e:
                continue
            gene = fields[12]
            for es_str, ee_str in zip(fields[9].rstrip(",").split(","),
                                       fields[10].rstrip(",").split(",")):
                if not es_str or not ee_str:
                    continue
                es, ee = max(int(es_str), cds_s), min(int(ee_str), cds_e)
                if es >= ee:
                    continue
                key = (chrom, es, ee)
                if key not in seen:
                    seen.add(key)
                    exons_by_gene[gene].append((chrom, es, ee, strand, gene))
    return exons_by_gene


def parse_ct_mutations(maf_path):
    """Parse C>T and G>A SNPs from MAF."""
    rows = []
    for chunk in pd.read_csv(maf_path, sep="\t", comment="#", low_memory=False, chunksize=100000):
        if "Variant_Type" in chunk.columns:
            chunk = chunk[chunk["Variant_Type"] == "SNP"]
        ct = (chunk["Reference_Allele"] == "C") & (chunk["Tumor_Seq_Allele2"] == "T")
        ga = (chunk["Reference_Allele"] == "G") & (chunk["Tumor_Seq_Allele2"] == "A")
        sub = chunk[ct | ga].copy()
        if len(sub) > 0:
            chrom = sub["Chromosome"].astype(str)
            if not chrom.str.startswith("chr").any():
                chrom = "chr" + chrom
            sub["chrom"] = chrom
            sub["pos"] = sub["Start_Position"].astype(int) - 1
            sub["strand_inf"] = np.where(sub["Reference_Allele"] == "C", "+", "-")
            sub["gene"] = sub.get("Hugo_Symbol", "unknown")
            sub["sample"] = sub.get("Tumor_Sample_Barcode", "unknown")
            rows.append(sub[["chrom", "pos", "strand_inf", "gene", "sample"]].copy())
    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    df_dedup = df.drop_duplicates(subset=["chrom", "pos"]).copy()
    rec = df.groupby(["chrom", "pos"]).size().reset_index(name="recurrence")
    df_dedup = df_dedup.merge(rec, on=["chrom", "pos"], how="left")
    return df_dedup


def get_matched_controls(mutations_df, genome, exons_by_gene, n_controls=N_CONTROLS):
    """For each mutation, pick n_controls random C positions in same gene's exons."""
    rng = np.random.RandomState(SEED)
    controls = []
    for _, row in mutations_df.iterrows():
        gene = row["gene"]
        gene_exons = exons_by_gene.get(gene, [])
        if not gene_exons:
            continue
        c_positions = []
        for chrom, start, end, strand, _ in gene_exons:
            if chrom != row["chrom"]:
                continue
            try:
                exon_seq = str(genome[chrom][start:end]).upper()
                for i, base in enumerate(exon_seq):
                    pos = start + i
                    if pos == row["pos"]:
                        continue
                    if base == "C":
                        c_positions.append((chrom, pos, "+"))
                    elif base == "G":
                        c_positions.append((chrom, pos, "-"))
            except (KeyError, ValueError):
                continue
        if len(c_positions) >= n_controls:
            chosen = rng.choice(len(c_positions), n_controls, replace=False)
            for idx in chosen:
                ch, p, s = c_positions[idx]
                controls.append({"chrom": ch, "pos": p, "strand_inf": s, "gene": gene})
        elif c_positions:
            for ch, p, s in c_positions[:n_controls]:
                controls.append({"chrom": ch, "pos": p, "strand_inf": s, "gene": gene})
    return pd.DataFrame(controls) if controls else pd.DataFrame()


def extract_sequences(positions_df, genome):
    """Extract 201-nt sequences for each position."""
    seqs = []
    for _, row in positions_df.iterrows():
        chrom = row["chrom"]
        pos = int(row["pos"])
        strand = row["strand_inf"]
        try:
            chrom_len = len(genome[chrom])
            start = pos - 100
            end = pos + 101
            if start < 0 or end > chrom_len:
                seqs.append(None)
                continue
            seq = str(genome[chrom][start:end]).upper()
            if strand == "-":
                comp = str.maketrans("ACGT", "TGCA")
                seq = seq.translate(comp)[::-1]
            seqs.append(seq)
        except (KeyError, ValueError):
            seqs.append(None)
    return seqs


# =============================================================================
# Enrichment computation
# =============================================================================

def compute_enrichment(mut_scores, ctrl_scores,
                       thresholds=(0.1, 0.2, 0.3, 0.4, 0.5)):
    """Compute enrichment ORs at multiple thresholds."""
    results = {}
    for t in thresholds:
        mut_above = int((mut_scores >= t).sum())
        mut_below = int((mut_scores < t).sum())
        ctrl_above = int((ctrl_scores >= t).sum())
        ctrl_below = int((ctrl_scores < t).sum())
        if mut_below > 0 and ctrl_above > 0 and ctrl_below > 0 and mut_above > 0:
            table = [[mut_above, mut_below], [ctrl_above, ctrl_below]]
            odds_ratio, p_value = stats.fisher_exact(table)
        else:
            odds_ratio, p_value = float("nan"), 1.0
        results[str(t)] = {
            "OR": float(odds_ratio),
            "p": float(p_value),
            "n_mut_above": mut_above,
            "frac_mut": float(mut_above / max(len(mut_scores), 1)),
            "frac_ctrl": float(ctrl_above / max(len(ctrl_scores), 1)),
        }
    return results


def compute_percentile_enrichment(mut_scores, ctrl_scores,
                                   percentiles=(50, 75, 90, 95)):
    """Compute enrichment at percentile thresholds of the CONTROL distribution."""
    results = {}
    all_scores = np.concatenate([mut_scores, ctrl_scores])
    for pct in percentiles:
        t = np.percentile(ctrl_scores, pct)
        mut_above = int((mut_scores >= t).sum())
        mut_below = int((mut_scores < t).sum())
        ctrl_above = int((ctrl_scores >= t).sum())
        ctrl_below = int((ctrl_scores < t).sum())
        if mut_below > 0 and ctrl_above > 0 and ctrl_below > 0 and mut_above > 0:
            table = [[mut_above, mut_below], [ctrl_above, ctrl_below]]
            odds_ratio, p_value = stats.fisher_exact(table)
        else:
            odds_ratio, p_value = float("nan"), 1.0
        results[f"p{pct}"] = {
            "threshold": float(t),
            "OR": float(odds_ratio),
            "p": float(p_value),
            "frac_mut": float(mut_above / max(len(mut_scores), 1)),
            "frac_ctrl": float(ctrl_above / max(len(ctrl_scores), 1)),
        }
    return results


# =============================================================================
# Main
# =============================================================================

def main():
    t_start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: Train multi-class model ----
    logger.info("=" * 70)
    logger.info("STEP 1: Train multi-class XGBoost on multi-enzyme v3 data")
    logger.info("=" * 70)

    splits = pd.read_csv(MULTI_SPLITS)
    with open(MULTI_SEQS) as f:
        sequences_dict = json.load(f)

    loop_df = pd.read_csv(LOOP_CSV)
    loop_df["site_id"] = loop_df["site_id"].astype(str)
    loop_df = loop_df.set_index("site_id")

    struct_data = None
    if STRUCT_CACHE.exists():
        struct_data = dict(np.load(str(STRUCT_CACHE), allow_pickle=True))
        struct_data["site_ids"] = [str(s) for s in struct_data["site_ids"]]

    # Build 40-dim features
    logger.info("Building 40-dim training features...")
    X_train = build_40dim_from_training(splits, sequences_dict, loop_df, struct_data)

    # Build multi-class labels
    # Positives: enzyme label; Negatives: "Negative"
    mc_labels = []
    for _, row in splits.iterrows():
        if row["is_edited"] == 1:
            mc_labels.append(row["enzyme"])
        else:
            mc_labels.append("Negative")
    mc_labels = np.array(mc_labels)

    y_train = np.array([CLASS_TO_IDX[lbl] for lbl in mc_labels])

    logger.info(f"Training data: {len(X_train)} samples, {X_train.shape[1]} features")
    for cls_name, cls_idx in CLASS_TO_IDX.items():
        n = int((y_train == cls_idx).sum())
        logger.info(f"  {cls_name}: {n}")

    # Cross-validate to assess quality
    logger.info("\n5-fold cross-validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_logloss = []
    cv_edited_auroc = []

    for fold_i, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        xgb_cv = XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            objective="multi:softprob", num_class=len(ENZYME_CLASSES),
            random_state=SEED, eval_metric="mlogloss",
            use_label_encoder=False, verbosity=0,
        )
        xgb_cv.fit(X_train[tr_idx], y_train[tr_idx])
        proba = xgb_cv.predict_proba(X_train[val_idx])
        ll = log_loss(y_train[val_idx], proba, labels=list(range(len(ENZYME_CLASSES))))
        cv_logloss.append(ll)

        # Binary edited vs negative AUROC
        neg_idx_cls = CLASS_TO_IDX["Negative"]
        p_edited = 1 - proba[:, neg_idx_cls]
        is_edited_val = (y_train[val_idx] != neg_idx_cls).astype(int)
        from sklearn.metrics import roc_auc_score
        auroc = roc_auc_score(is_edited_val, p_edited)
        cv_edited_auroc.append(auroc)
        logger.info(f"  Fold {fold_i}: logloss={ll:.4f}, P(edited) AUROC={auroc:.4f}")

    logger.info(f"  Mean logloss: {np.mean(cv_logloss):.4f} +/- {np.std(cv_logloss):.4f}")
    logger.info(f"  Mean P(edited) AUROC: {np.mean(cv_edited_auroc):.4f} +/- {np.std(cv_edited_auroc):.4f}")

    # Train final model on all data
    logger.info("\nTraining final model on all data...")
    model = XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        objective="multi:softprob", num_class=len(ENZYME_CLASSES),
        random_state=SEED, eval_metric="mlogloss",
        use_label_encoder=False, verbosity=0,
    )
    model.fit(X_train, y_train)

    # Training sanity check
    train_proba = model.predict_proba(X_train)
    train_pred = np.argmax(train_proba, axis=1)
    train_acc = np.mean(train_pred == y_train)
    logger.info(f"Training accuracy: {train_acc:.4f}")

    # Feature importance
    importances = model.feature_importances_
    feat_names = (
        [f"motif_{i}" for i in range(24)]
        + [f"struct_{i}" for i in range(7)]
        + [f"loop_{i}" for i in range(9)]
    )
    top_feats = sorted(zip(feat_names, importances), key=lambda x: -x[1])[:10]
    logger.info("Top 10 features:")
    for fn, imp in top_feats:
        logger.info(f"  {fn}: {imp:.4f}")

    del struct_data, sequences_dict
    gc.collect()

    # ---- Step 2: Load genome and parse exons ----
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: Parse exons and load genome")
    logger.info("=" * 70)
    exons_by_gene = parse_exons(REFGENE_HG19)
    genome = Fasta(str(HG19_FA))
    logger.info(f"  {len(exons_by_gene):,} genes with exonic regions")

    # ---- Step 3: Process each cancer type ----
    all_results = {}
    summary_rows = []

    for cancer in CANCERS:
        logger.info("\n" + "=" * 70)
        logger.info(f"Processing {cancer.upper()}")
        logger.info("=" * 70)

        maf_path = TCGA_DIR / f"{cancer}_tcga_pan_can_atlas_2018_mutations.txt"
        vienna_cache_file = VIENNA_CACHE_DIR / f"{cancer}_vienna_raw.json.gz"

        if not maf_path.exists():
            logger.warning(f"No MAF for {cancer}, skipping")
            continue
        if not vienna_cache_file.exists():
            logger.warning(f"No ViennaRNA cache for {cancer}, skipping")
            continue

        # Parse mutations
        logger.info("  Parsing MAF...")
        mut_df = parse_ct_mutations(maf_path)
        if len(mut_df) == 0:
            logger.warning(f"No C>T mutations for {cancer}")
            continue
        logger.info(f"  {len(mut_df):,} unique C>T mutation positions")

        # Extract sequences
        logger.info("  Extracting mutation sequences...")
        mut_seqs = extract_sequences(mut_df, genome)

        # Get controls (same SEED as cache creator)
        logger.info("  Generating matched controls...")
        ctrl_df = get_matched_controls(mut_df, genome, exons_by_gene, N_CONTROLS)
        logger.info(f"  {len(ctrl_df):,} control positions")
        ctrl_seqs = extract_sequences(ctrl_df, genome)

        # Filter valid
        valid_mut = [(i, s) for i, s in enumerate(mut_seqs)
                     if s is not None and len(s) == 201 and s[100] == "C"]
        valid_ctrl = [(i, s) for i, s in enumerate(ctrl_seqs)
                      if s is not None and len(s) == 201 and s[100] == "C"]

        mut_valid_seqs = [s for _, s in valid_mut] if valid_mut else []
        ctrl_valid_seqs = [s for _, s in valid_ctrl] if valid_ctrl else []

        all_seqs = list(mut_valid_seqs) + list(ctrl_valid_seqs)
        n_mut_seqs = len(mut_valid_seqs)
        n_ctrl_seqs = len(ctrl_valid_seqs)
        logger.info(f"  Valid: {n_mut_seqs:,} mutations, {n_ctrl_seqs:,} controls")

        # Load cached ViennaRNA folds
        logger.info("  Loading cached ViennaRNA structures...")
        with gzip.open(str(vienna_cache_file), "rt") as fz:
            cache_data = json.load(fz)

        fold_results = cache_data["fold_results"]
        cached_n_mut = cache_data["n_mut"]
        cached_n_ctrl = cache_data["n_ctrl"]

        # The cache was created with same MAF parsing + same SEED controls
        # Use cache n_mut/n_ctrl to split fold_results
        logger.info(f"  Cache: {cached_n_mut:,} mut + {cached_n_ctrl:,} ctrl = {len(fold_results):,} folds")

        # Build 40-dim features using fold results for structure
        # We need sequences for motif features, and fold results for struct+loop features
        # The cache fold_results are ordered: [mutations..., controls...]
        # Rebuild sequences list to match cache ordering
        n_mut = cached_n_mut
        n_ctrl = cached_n_ctrl

        # If our fresh seq extraction matches the cache count, use our sequences
        # Otherwise, we still need sequences for motif; fold_results give structure
        if n_mut_seqs == n_mut and n_ctrl_seqs == n_ctrl:
            logger.info("  Sequence counts match cache -- using fresh sequences for motif")
        else:
            logger.warning(f"  Sequence count mismatch: fresh={n_mut_seqs}+{n_ctrl_seqs}, "
                          f"cache={n_mut}+{n_ctrl}. Using cache counts.")
            # Trim or pad sequences to match cache
            if n_mut_seqs > n_mut:
                mut_valid_seqs = mut_valid_seqs[:n_mut]
            elif n_mut_seqs < n_mut:
                mut_valid_seqs.extend([None] * (n_mut - n_mut_seqs))
            if n_ctrl_seqs > n_ctrl:
                ctrl_valid_seqs = ctrl_valid_seqs[:n_ctrl]
            elif n_ctrl_seqs < n_ctrl:
                ctrl_valid_seqs.extend([None] * (n_ctrl - n_ctrl_seqs))
            all_seqs = list(mut_valid_seqs) + list(ctrl_valid_seqs)

        logger.info("  Building 40-dim features...")
        features_40d = build_40dim_features(all_seqs, fold_results)

        # Score with multi-class model
        proba = model.predict_proba(features_40d)
        # proba columns: A3A, A3B, A3G, A3A_A3G, Neither, Unknown, Negative

        neg_class_idx = CLASS_TO_IDX["Negative"]
        p_edited = 1.0 - proba[:, neg_class_idx]

        mut_p = proba[:n_mut]
        ctrl_p = proba[n_mut:]

        # Context analysis
        tc_mut = sum(1 for s in mut_valid_seqs[:n_mut] if s and len(s) > 99 and s[99] in "TU")
        tc_ctrl = sum(1 for s in ctrl_valid_seqs[:n_ctrl] if s and len(s) > 99 and s[99] in "TU")
        cc_mut = sum(1 for s in mut_valid_seqs[:n_mut] if s and len(s) > 99 and s[99] == "C")
        cc_ctrl = sum(1 for s in ctrl_valid_seqs[:n_ctrl] if s and len(s) > 99 and s[99] == "C")

        cancer_results = {
            "cancer": cancer,
            "n_mutations": n_mut,
            "n_controls": n_ctrl,
            "tc_frac_mutations": tc_mut / max(n_mut, 1),
            "tc_frac_controls": tc_ctrl / max(n_ctrl, 1),
            "cc_frac_mutations": cc_mut / max(n_mut, 1),
            "cc_frac_controls": cc_ctrl / max(n_ctrl, 1),
            "enzymes": {},
        }

        # Per-enzyme probability analysis
        for enz_name in ENZYME_CLASSES[:-1]:  # skip Negative
            enz_idx = CLASS_TO_IDX[enz_name]
            mut_scores = mut_p[:, enz_idx]
            ctrl_scores = ctrl_p[:, enz_idx]

            mw_stat, mw_p_val = stats.mannwhitneyu(
                mut_scores, ctrl_scores, alternative="greater"
            ) if n_mut > 0 and n_ctrl > 0 else (0, 1)

            enrichment = compute_enrichment(mut_scores, ctrl_scores)
            pct_enrichment = compute_percentile_enrichment(mut_scores, ctrl_scores)

            cancer_results["enzymes"][enz_name] = {
                "mean_mut": float(np.mean(mut_scores)),
                "mean_ctrl": float(np.mean(ctrl_scores)),
                "delta": float(np.mean(mut_scores) - np.mean(ctrl_scores)),
                "mann_whitney_p": float(mw_p_val),
                "enrichment": enrichment,
                "percentile_enrichment": pct_enrichment,
            }

        # P(edited) = 1 - P(Negative)
        mut_p_edited = p_edited[:n_mut]
        ctrl_p_edited = p_edited[n_mut:]
        mw_stat, mw_p_val = stats.mannwhitneyu(
            mut_p_edited, ctrl_p_edited, alternative="greater"
        )
        enrichment_edited = compute_enrichment(mut_p_edited, ctrl_p_edited)
        pct_enrichment_edited = compute_percentile_enrichment(mut_p_edited, ctrl_p_edited)

        cancer_results["enzymes"]["P_edited"] = {
            "mean_mut": float(np.mean(mut_p_edited)),
            "mean_ctrl": float(np.mean(ctrl_p_edited)),
            "delta": float(np.mean(mut_p_edited) - np.mean(ctrl_p_edited)),
            "mann_whitney_p": float(mw_p_val),
            "enrichment": enrichment_edited,
            "percentile_enrichment": pct_enrichment_edited,
        }

        all_results[cancer] = cancer_results

        # Print summary
        logger.info(f"\n  --- {cancer.upper()} Summary ---")
        logger.info(f"  TC context: mut={tc_mut/max(n_mut,1):.3f}, ctrl={tc_ctrl/max(n_ctrl,1):.3f}")
        logger.info(f"  {'Enzyme':<12} {'mean_mut':>10} {'mean_ctrl':>10} {'delta':>10} {'MWU_p':>12} {'OR@p75':>10} {'OR@p90':>10}")
        for enz_name in list(ENZYME_CLASSES[:-1]) + ["P_edited"]:
            r = cancer_results["enzymes"][enz_name]
            or_p75 = r["percentile_enrichment"].get("p75", {}).get("OR", float("nan"))
            or_p90 = r["percentile_enrichment"].get("p90", {}).get("OR", float("nan"))
            logger.info(f"  {enz_name:<12} {r['mean_mut']:>10.4f} {r['mean_ctrl']:>10.4f} "
                       f"{r['delta']:>10.4f} {r['mann_whitney_p']:>12.2e} "
                       f"{or_p75:>10.3f} {or_p90:>10.3f}")

            if enz_name in ["A3A", "A3G", "Neither", "P_edited"]:
                summary_rows.append({
                    "cancer": cancer.upper(),
                    "enzyme": enz_name,
                    "mean_mut": r["mean_mut"],
                    "mean_ctrl": r["mean_ctrl"],
                    "delta": r["delta"],
                    "mwu_p": r["mann_whitney_p"],
                    "OR_p75": or_p75,
                    "OR_p90": or_p90,
                })

        del fold_results, features_40d, proba, cache_data
        gc.collect()

    # ---- Step 4: Generate figures ----
    logger.info("\n" + "=" * 70)
    logger.info("STEP 4: Generate figures")
    logger.info("=" * 70)

    cancers_done = [c for c in CANCERS if c in all_results]
    key_enzymes = ["A3A", "A3G", "Neither", "P_edited"]
    colors = {"A3A": "#dc2626", "A3G": "#2563eb", "Neither": "#16a34a",
              "A3A_A3G": "#9333ea", "A3B": "#f59e0b", "Unknown": "#6b7280",
              "P_edited": "#000000"}

    if len(cancers_done) >= 2:
        # Figure 1: 2x2 panel
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Panel A: Mean P(enzyme) delta (mut - ctrl) for key enzymes
        ax = axes[0, 0]
        x = np.arange(len(cancers_done))
        width = 0.8 / len(key_enzymes)
        for ei, enz in enumerate(key_enzymes):
            deltas = []
            for c in cancers_done:
                d = all_results[c]["enzymes"].get(enz, {}).get("delta", 0)
                deltas.append(d)
            ax.bar(x + ei * width - 0.4 + width / 2, deltas, width,
                   label=enz, color=colors.get(enz, "#000"), alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([c.upper() for c in cancers_done], rotation=45, ha="right")
        ax.set_ylabel("Mean P(enzyme) delta\n(mutations - controls)")
        ax.set_title("A) Per-Enzyme Score Difference")
        ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
        ax.legend(fontsize=8, loc="upper left")

        # Panel B: OR at 75th percentile
        ax = axes[0, 1]
        for ei, enz in enumerate(key_enzymes):
            ors = []
            for c in cancers_done:
                e = all_results[c]["enzymes"].get(enz, {}).get("percentile_enrichment", {})
                ors.append(e.get("p75", {}).get("OR", float("nan")))
            ax.bar(x + ei * width - 0.4 + width / 2, ors, width,
                   label=enz, color=colors.get(enz, "#000"), alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([c.upper() for c in cancers_done], rotation=45, ha="right")
        ax.set_ylabel("Odds Ratio")
        ax.set_title("B) Enrichment OR @ 75th Percentile of Controls")
        ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
        ax.legend(fontsize=8)

        # Panel C: Heatmap of OR@p90 -- enzyme x cancer
        ax = axes[1, 0]
        all_enzymes_plot = ["A3A", "A3B", "A3G", "A3A_A3G", "Neither", "Unknown", "P_edited"]
        or_matrix = np.full((len(all_enzymes_plot), len(cancers_done)), np.nan)
        for ei, enz in enumerate(all_enzymes_plot):
            for ci, c in enumerate(cancers_done):
                e = all_results[c]["enzymes"].get(enz, {}).get("percentile_enrichment", {})
                or_matrix[ei, ci] = e.get("p90", {}).get("OR", float("nan"))

        im = ax.imshow(or_matrix, cmap="RdBu_r", vmin=0.8, vmax=1.3, aspect="auto")
        ax.set_xticks(range(len(cancers_done)))
        ax.set_xticklabels([c.upper() for c in cancers_done], rotation=45, ha="right")
        ax.set_yticks(range(len(all_enzymes_plot)))
        ax.set_yticklabels(all_enzymes_plot)
        for ei in range(len(all_enzymes_plot)):
            for ci in range(len(cancers_done)):
                val = or_matrix[ei, ci]
                if not np.isnan(val):
                    ax.text(ci, ei, f"{val:.2f}", ha="center", va="center", fontsize=7,
                           color="white" if abs(val - 1.05) > 0.15 else "black")
        plt.colorbar(im, ax=ax, label="Odds Ratio @ P90")
        ax.set_title("C) Enzyme x Cancer Heatmap (OR @ P90)")

        # Panel D: OR across percentile thresholds for BLCA
        ax = axes[1, 1]
        ref_cancer = "blca" if "blca" in all_results else cancers_done[0]
        pct_labels = ["p50", "p75", "p90", "p95"]
        for enz in key_enzymes:
            ors = []
            for p in pct_labels:
                e = all_results[ref_cancer]["enzymes"].get(enz, {}).get("percentile_enrichment", {})
                ors.append(e.get(p, {}).get("OR", float("nan")))
            ax.plot([50, 75, 90, 95], ors, marker="o", label=enz,
                    color=colors.get(enz, "#000"), linewidth=2)
        ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel("Percentile threshold (of controls)")
        ax.set_ylabel("Odds Ratio")
        ax.set_title(f"D) Enrichment Profile ({ref_cancer.upper()})")
        ax.legend(fontsize=8)

        plt.tight_layout()
        fig_path = OUTPUT_DIR / "multiclass_enzyme_tcga.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Figure saved: {fig_path}")

        # Figure 2: Per-enzyme probability distributions (violin/box)
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        for ai, enz in enumerate(["A3A", "A3B", "A3G", "Neither", "P_edited", "A3A_A3G"]):
            ax = axes[ai // 3, ai % 3]
            cancer_labels = []
            data_mut = []
            data_ctrl = []
            for c in cancers_done:
                enz_idx = CLASS_TO_IDX.get(enz, -1)
                if enz == "P_edited":
                    mut_s = 1.0 - all_results[c]["enzymes"]["P_edited"]["mean_mut"]  # placeholder
                    # Get raw data from stored means
                else:
                    pass
                r = all_results[c]["enzymes"].get(enz, {})
                data_mut.append(r.get("mean_mut", 0))
                data_ctrl.append(r.get("mean_ctrl", 0))
                cancer_labels.append(c.upper())

            x = np.arange(len(cancer_labels))
            ax.bar(x - 0.2, data_mut, 0.35, label="Mutations", color="#dc2626", alpha=0.7)
            ax.bar(x + 0.2, data_ctrl, 0.35, label="Controls", color="#3b82f6", alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(cancer_labels, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel(f"Mean P({enz})")
            ax.set_title(f"P({enz})")
            ax.legend(fontsize=7)

        plt.tight_layout()
        fig_path = OUTPUT_DIR / "multiclass_mean_probs.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Figure saved: {fig_path}")

    # ---- Save results ----
    with open(OUTPUT_DIR / "multiclass_enzyme_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"Results saved: {OUTPUT_DIR / 'multiclass_enzyme_results.json'}")

    # ---- Final summary table ----
    elapsed = time.time() - t_start
    logger.info(f"\n{'='*70}")
    logger.info(f"COMPLETE ({elapsed/60:.1f} min)")
    logger.info(f"{'='*70}")

    logger.info("\n===== SUMMARY: Multi-class Enzyme x Cancer OR (percentile-based) =====\n")

    # Table 1: OR @ 75th percentile
    logger.info("--- OR @ 75th percentile of controls ---")
    header = f"{'Cancer':<8}" + "".join(f"{'P('+e+')':>14}" for e in key_enzymes)
    logger.info(header)
    logger.info("-" * len(header))
    for c in cancers_done:
        row = f"{c.upper():<8}"
        for enz in key_enzymes:
            r = all_results[c]["enzymes"].get(enz, {}).get("percentile_enrichment", {})
            orval = r.get("p75", {}).get("OR", float("nan"))
            pval = r.get("p75", {}).get("p", 1.0)
            sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else ""))
            row += f"{orval:>10.3f}{sig:>4}"
            row = row.rstrip() + " "
        logger.info(row)

    logger.info("")

    # Table 2: OR @ 90th percentile
    logger.info("--- OR @ 90th percentile of controls ---")
    logger.info(header)
    logger.info("-" * len(header))
    for c in cancers_done:
        row = f"{c.upper():<8}"
        for enz in key_enzymes:
            r = all_results[c]["enzymes"].get(enz, {}).get("percentile_enrichment", {})
            orval = r.get("p90", {}).get("OR", float("nan"))
            pval = r.get("p90", {}).get("p", 1.0)
            sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else ""))
            row += f"{orval:>10.3f}{sig:>4}"
            row = row.rstrip() + " "
        logger.info(row)

    logger.info("")

    # Table 3: Mean delta (mut - ctrl)
    logger.info("--- Mean P(enzyme) delta: mutations - controls ---")
    header2 = f"{'Cancer':<8}" + "".join(f"{'P('+e+')':>14}" for e in key_enzymes)
    logger.info(header2)
    logger.info("-" * len(header2))
    for c in cancers_done:
        row = f"{c.upper():<8}"
        for enz in key_enzymes:
            r = all_results[c]["enzymes"].get(enz, {})
            delta = r.get("delta", 0)
            pval = r.get("mann_whitney_p", 1.0)
            sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else ""))
            row += f"{delta:>+10.5f}{sig:>4}"
            row = row.rstrip() + " "
        logger.info(row)

    # Key hypothesis tests
    logger.info("\n===== KEY HYPOTHESIS TESTS =====")
    for c in cancers_done:
        apobec_cancers = ["blca", "brca", "cesc", "lusc", "hnsc", "skcm"]
        gi_cancers = ["esca", "stad", "lihc"]
        if c in apobec_cancers:
            label = "APOBEC-driven"
        elif c in gi_cancers:
            label = "GI cancer"
        else:
            label = "other"
        a3a_or = all_results[c]["enzymes"].get("A3A", {}).get("percentile_enrichment", {}).get("p75", {}).get("OR", float("nan"))
        neither_or = all_results[c]["enzymes"].get("Neither", {}).get("percentile_enrichment", {}).get("p75", {}).get("OR", float("nan"))
        logger.info(f"  {c.upper()} ({label}): P(A3A) OR@p75={a3a_or:.3f}, P(Neither) OR@p75={neither_or:.3f}")


if __name__ == "__main__":
    main()
