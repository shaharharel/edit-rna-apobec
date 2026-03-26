#!/usr/bin/env python3
"""COADREAD (colorectal) TCGA enrichment: binary GB + multi-class enzyme model.

Processes coadread_tcga_pan_can_atlas_2018 MAF through the same pipeline as
tcga_full_model_enrichment.py and exp_multiclass_enzyme_tcga.py:

1. Train full 40-dim GB binary model on multi-enzyme v3 data
2. Train 7-class XGBoost on multi-enzyme v3 data
3. Parse COADREAD MAF for C>T and G>A mutations
4. Generate 5 matched controls per mutation from same-exon C positions
5. Extract 201-nt sequences from hg19 genome
6. Fold ALL sequences with ViennaRNA (14 workers, parallel)
7. Cache raw ViennaRNA structures to vienna_cache/coadread_vienna_raw.json.gz
8. Build 40-dim features, score with binary model, compute enrichment
9. Save raw scores to raw_scores/coadread_scores.csv
10. Run TC-stratified analysis
11. Score with 7-class enzyme model, compute per-enzyme enrichment

COADREAD is critical: colorectal cancer is APOBEC1 territory.
We expect P(Neither/APOBEC1) to show strong enrichment.

Usage:
    conda run -n quris python scripts/multi_enzyme/tcga_coadread_enrichment.py
"""

import gc
import gzip
import json
import logging
import multiprocessing as mp
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
from sklearn.metrics import roc_auc_score
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

MAF_PATH = DATA_DIR / "raw/tcga/coadread_tcga_pan_can_atlas_2018_mutations.txt"
OUTPUT_DIR = PROJECT_ROOT / "experiments/multi_enzyme/outputs/tcga_gnomad"
VIENNA_CACHE_DIR = OUTPUT_DIR / "vienna_cache"
SCORES_DIR = OUTPUT_DIR / "raw_scores"
ENZYME_PROB_DIR = OUTPUT_DIR / "enzyme_probability"

N_CONTROLS = 5
N_WORKERS = 14
SEED = 42
CANCER = "coadread"

# Multi-class enzyme labels
ENZYME_CLASSES = ["A3A", "A3B", "A3G", "A3A_A3G", "Neither", "Unknown", "Negative"]
CLASS_TO_IDX = {c: i for i, c in enumerate(ENZYME_CLASSES)}


# =============================================================================
# ViennaRNA folding (parallelizable)
# =============================================================================

def fold_single_sequence(seq_201nt):
    """Fold a single 201-nt sequence with ViennaRNA. Returns raw fold data or None."""
    import RNA

    if seq_201nt is None or len(seq_201nt) != 201:
        return None

    center = 100
    seq = seq_201nt.upper().replace("U", "T")

    if seq[center] != "C":
        return None

    try:
        fc_wt = RNA.fold_compound(seq)
        struct_wt, mfe_wt = fc_wt.mfe()
        fc_wt.exp_params_rescale(mfe_wt)
        fc_wt.pf()
        bpp_wt = fc_wt.bpp()

        seq_ed = seq[:center] + "T" + seq[center + 1:]
        fc_ed = RNA.fold_compound(seq_ed)
        struct_ed, mfe_ed = fc_ed.mfe()
        fc_ed.exp_params_rescale(mfe_ed)
        fc_ed.pf()
        bpp_ed = fc_ed.bpp()

        c = center + 1  # 1-indexed
        w = 10
        bpp_wt_window = []
        bpp_ed_window = []
        for i in range(max(1, c - w), min(len(bpp_wt), c + w + 1)):
            bpp_wt_window.append(float(sum(bpp_wt[i])) if i < len(bpp_wt) else 0.5)
            bpp_ed_window.append(float(sum(bpp_ed[i])) if i < len(bpp_ed) else 0.5)

        return {
            "struct_wt": struct_wt,
            "struct_ed": struct_ed,
            "mfe_wt": float(mfe_wt),
            "mfe_ed": float(mfe_ed),
            "bpp_wt_center": float(sum(bpp_wt[c])) if c < len(bpp_wt) else 0.5,
            "bpp_ed_center": float(sum(bpp_ed[c])) if c < len(bpp_ed) else 0.5,
            "bpp_wt_window": bpp_wt_window,
            "bpp_ed_window": bpp_ed_window,
        }
    except Exception:
        return None


def fold_batch(sequences):
    """Fold a batch of sequences."""
    return [fold_single_sequence(s) for s in sequences]


def parallel_fold(sequences, n_workers=N_WORKERS, chunk_size=500):
    """Fold sequences in parallel."""
    n = len(sequences)
    logger.info(f"Folding {n:,} sequences with {n_workers} workers (chunk={chunk_size})...")

    chunks = [sequences[i:i + chunk_size] for i in range(0, n, chunk_size)]
    results = []
    t0 = time.time()
    with mp.Pool(n_workers) as pool:
        for i, batch_result in enumerate(pool.imap(fold_batch, chunks)):
            results.extend(batch_result)
            if (i + 1) % 20 == 0:
                done = min((i + 1) * chunk_size, n)
                elapsed = time.time() - t0
                rate = done / elapsed
                eta = (n - done) / rate if rate > 0 else 0
                logger.info(f"  Folded {done:,}/{n:,} ({100*done/n:.0f}%) "
                            f"[{rate:.0f}/sec, ETA {eta/60:.0f} min]")

    elapsed = time.time() - t0
    n_ok = sum(1 for r in results if r is not None)
    logger.info(f"  Done: {n_ok:,}/{n:,} successful ({elapsed/60:.1f} min)")
    return results


# =============================================================================
# Feature derivation from fold results
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

    # Loop geometry (9-dim)
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


# =============================================================================
# Training data feature builder
# =============================================================================

def build_training_features(splits, sequences_dict, loop_df, struct_data):
    """Build 40-dim features for training data from splits + caches."""
    site_ids = splits["site_id"].astype(str).tolist()

    motif_feats = np.array(
        [extract_motif_from_seq(sequences_dict.get(sid, "N" * 201)) for sid in site_ids],
        dtype=np.float32,
    )

    struct_feats = np.zeros((len(site_ids), 7), dtype=np.float32)
    if struct_data is not None:
        struct_sids = [str(s) for s in struct_data["site_ids"]]
        sid_to_idx = {s: i for i, s in enumerate(struct_sids)}
        for i, sid in enumerate(site_ids):
            if sid in sid_to_idx:
                struct_feats[i] = struct_data["delta_features"][sid_to_idx[sid]]

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
# Train models
# =============================================================================

def train_binary_model(X_train, y_binary):
    """Train binary GB_HandFeatures (40-dim)."""
    model = XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=SEED, eval_metric="logloss", use_label_encoder=False,
    )
    model.fit(X_train, y_binary)
    train_auc = roc_auc_score(y_binary, model.predict_proba(X_train)[:, 1])
    logger.info(f"  Binary model training AUC: {train_auc:.4f}")
    return model


def train_multiclass_model(X_train, y_multi):
    """Train 7-class XGBoost."""
    model = XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        objective="multi:softprob", num_class=len(ENZYME_CLASSES),
        random_state=SEED, eval_metric="mlogloss",
        use_label_encoder=False, verbosity=0,
    )
    model.fit(X_train, y_multi)
    train_pred = np.argmax(model.predict_proba(X_train), axis=1)
    train_acc = np.mean(train_pred == y_multi)
    logger.info(f"  Multi-class model training accuracy: {train_acc:.4f}")
    return model


# =============================================================================
# TCGA parsing
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
    logger.info(f"  {len(df):,} C>T/G>A mutations -> {len(df_dedup):,} unique positions")
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
                controls.append({"chrom": ch, "pos": p, "strand_inf": s,
                                 "gene": gene, "is_control": True})
        elif c_positions:
            for ch, p, s in c_positions[:n_controls]:
                controls.append({"chrom": ch, "pos": p, "strand_inf": s,
                                 "gene": gene, "is_control": True})
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
# Enrichment
# =============================================================================

def compute_enrichment(mut_scores, ctrl_scores,
                       thresholds=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)):
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
            "n_mut_total": len(mut_scores),
            "n_ctrl_above": ctrl_above,
            "n_ctrl_total": len(ctrl_scores),
            "frac_mut": float(mut_above / max(len(mut_scores), 1)),
            "frac_ctrl": float(ctrl_above / max(len(ctrl_scores), 1)),
        }
    return results


def compute_percentile_enrichment(mut_scores, ctrl_scores,
                                   percentiles=(50, 75, 90, 95)):
    """Compute enrichment at percentile thresholds of the CONTROL distribution."""
    results = {}
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


def tc_stratified_analysis(all_seqs, scores, n_mut, n_ctrl):
    """Run TC-stratified enrichment analysis."""
    # Split into TC and non-TC
    mut_tc_scores = []
    mut_nontc_scores = []
    ctrl_tc_scores = []
    ctrl_nontc_scores = []

    for i in range(n_mut):
        seq = all_seqs[i]
        is_tc = seq is not None and len(seq) > 99 and seq[99] in "TU"
        if is_tc:
            mut_tc_scores.append(scores[i])
        else:
            mut_nontc_scores.append(scores[i])

    for i in range(n_ctrl):
        seq = all_seqs[n_mut + i]
        is_tc = seq is not None and len(seq) > 99 and seq[99] in "TU"
        if is_tc:
            ctrl_tc_scores.append(scores[i + n_mut])
        else:
            ctrl_nontc_scores.append(scores[i + n_mut])

    results = {}
    for label, ms, cs in [("TC_only", mut_tc_scores, ctrl_tc_scores),
                           ("nonTC_only", mut_nontc_scores, ctrl_nontc_scores)]:
        ms = np.array(ms)
        cs = np.array(cs)
        if len(ms) > 0 and len(cs) > 0:
            mw_stat, mw_p = stats.mannwhitneyu(ms, cs, alternative="two-sided")
            enrichment = compute_enrichment(ms, cs)
            results[label] = {
                "n_mut": len(ms),
                "n_ctrl": len(cs),
                "mean_mut": float(np.mean(ms)),
                "mean_ctrl": float(np.mean(cs)),
                "delta": float(np.mean(ms) - np.mean(cs)),
                "mann_whitney_p": float(mw_p),
                "enrichment": enrichment,
            }
        else:
            results[label] = {"n_mut": len(ms), "n_ctrl": len(cs), "note": "insufficient data"}

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    t_start = time.time()

    # Create output dirs
    for d in [OUTPUT_DIR, VIENNA_CACHE_DIR, SCORES_DIR, ENZYME_PROB_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Step 1: Load training data and train models
    # =========================================================================
    logger.info("=" * 70)
    logger.info("STEP 1: Load training data and train models")
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

    logger.info("Building 40-dim training features...")
    X_train = build_training_features(splits, sequences_dict, loop_df, struct_data)

    # Binary labels
    y_binary = splits["is_edited"].values
    logger.info(f"Training data: {len(X_train)} samples, {X_train.shape[1]} features")
    logger.info(f"  Positives: {y_binary.sum():,}, Negatives: {(1 - y_binary).sum():,}")

    # Multi-class labels
    mc_labels = []
    for _, row in splits.iterrows():
        if row["is_edited"] == 1:
            mc_labels.append(row["enzyme"])
        else:
            mc_labels.append("Negative")
    y_multi = np.array([CLASS_TO_IDX[lbl] for lbl in mc_labels])

    for cls_name, cls_idx in CLASS_TO_IDX.items():
        n = int((y_multi == cls_idx).sum())
        logger.info(f"  {cls_name}: {n}")

    logger.info("\nTraining binary GB model (40-dim)...")
    binary_model = train_binary_model(X_train, y_binary)

    logger.info("\nTraining 7-class XGBoost...")
    multi_model = train_multiclass_model(X_train, y_multi)

    del struct_data, sequences_dict
    gc.collect()

    # =========================================================================
    # Step 2: Parse exons and load genome
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: Parse exons and load hg19 genome")
    logger.info("=" * 70)

    exons_by_gene = parse_exons(REFGENE_HG19)
    genome = Fasta(str(HG19_FA))
    logger.info(f"  {len(exons_by_gene):,} genes with exonic regions")

    # =========================================================================
    # Step 3: Parse COADREAD MAF
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 3: Parse COADREAD MAF")
    logger.info("=" * 70)

    mut_df = parse_ct_mutations(MAF_PATH)
    if len(mut_df) == 0:
        logger.error("No C>T mutations found!")
        return
    logger.info(f"  {len(mut_df):,} unique C>T mutation positions")

    # Recurrence distribution
    rec_counts = mut_df["recurrence"].describe()
    logger.info(f"  Recurrence stats: mean={rec_counts['mean']:.1f}, max={rec_counts['max']:.0f}")

    # =========================================================================
    # Step 4: Generate matched controls
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 4: Generate matched controls")
    logger.info("=" * 70)

    ctrl_df = get_matched_controls(mut_df, genome, exons_by_gene, N_CONTROLS)
    logger.info(f"  {len(ctrl_df):,} control positions")

    # =========================================================================
    # Step 5: Extract 201-nt sequences
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 5: Extract 201-nt sequences")
    logger.info("=" * 70)

    logger.info("  Extracting mutation sequences...")
    mut_seqs = extract_sequences(mut_df, genome)
    logger.info("  Extracting control sequences...")
    ctrl_seqs = extract_sequences(ctrl_df, genome)

    # Filter valid
    valid_mut = [(i, s) for i, s in enumerate(mut_seqs)
                 if s is not None and len(s) == 201 and s[100] == "C"]
    valid_ctrl = [(i, s) for i, s in enumerate(ctrl_seqs)
                  if s is not None and len(s) == 201 and s[100] == "C"]

    mut_valid_seqs = [s for _, s in valid_mut]
    ctrl_valid_seqs = [s for _, s in valid_ctrl]

    all_seqs = list(mut_valid_seqs) + list(ctrl_valid_seqs)
    n_mut = len(mut_valid_seqs)
    n_ctrl = len(ctrl_valid_seqs)
    logger.info(f"  Valid: {n_mut:,} mutations, {n_ctrl:,} controls")

    # =========================================================================
    # Step 6: ViennaRNA folding with cache
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 6: ViennaRNA folding")
    logger.info("=" * 70)

    vienna_cache_file = VIENNA_CACHE_DIR / f"{CANCER}_vienna_raw.json.gz"

    if vienna_cache_file.exists():
        logger.info(f"  Loading cached ViennaRNA structures from {vienna_cache_file}")
        with gzip.open(str(vienna_cache_file), "rt") as fz:
            cache_data = json.load(fz)
        fold_results = cache_data["fold_results"]
        cached_n_mut = cache_data["n_mut"]
        cached_n_ctrl = cache_data["n_ctrl"]
        logger.info(f"  Loaded {len(fold_results):,} cached folds "
                    f"({cached_n_mut:,} mut + {cached_n_ctrl:,} ctrl)")

        # Use cached counts
        if cached_n_mut != n_mut or cached_n_ctrl != n_ctrl:
            logger.warning(f"  Count mismatch: fresh={n_mut}+{n_ctrl}, cache={cached_n_mut}+{cached_n_ctrl}")
            n_mut = cached_n_mut
            n_ctrl = cached_n_ctrl
    else:
        logger.info(f"  Folding {len(all_seqs):,} sequences with {N_WORKERS} workers...")
        fold_results = parallel_fold(all_seqs, N_WORKERS)

        logger.info(f"  Caching ViennaRNA structures to {vienna_cache_file}")
        cache_data = {
            "cancer": CANCER,
            "n_mut": n_mut,
            "n_ctrl": n_ctrl,
            "n_sequences": len(all_seqs),
            "fold_results": fold_results,
        }
        with gzip.open(str(vienna_cache_file), "wt") as fz:
            json.dump(cache_data, fz)
        logger.info(f"  Cache saved ({vienna_cache_file.stat().st_size / 1e6:.0f} MB)")

    # =========================================================================
    # Step 7: Build features and score
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 7: Build 40-dim features and score")
    logger.info("=" * 70)

    logger.info("  Building 40-dim features from ViennaRNA structures...")
    features_40d = build_40dim_features(all_seqs, fold_results)

    # Binary model scores
    binary_scores = binary_model.predict_proba(features_40d)[:, 1]
    mut_binary_scores = binary_scores[:n_mut]
    ctrl_binary_scores = binary_scores[n_mut:]

    # Multi-class model scores
    multi_proba = multi_model.predict_proba(features_40d)
    mut_multi = multi_proba[:n_mut]
    ctrl_multi = multi_proba[n_mut:]

    # =========================================================================
    # Step 8: Save raw scores
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 8: Save raw scores")
    logger.info("=" * 70)

    score_records = []
    for i in range(n_mut):
        rec = {
            "type": "mutation",
            "binary_score": float(mut_binary_scores[i]),
            "tc_context": 1 if all_seqs[i][99] in "TU" else 0,
        }
        for enz_name in ENZYME_CLASSES:
            rec[f"P_{enz_name}"] = float(mut_multi[i, CLASS_TO_IDX[enz_name]])
        score_records.append(rec)

    for i in range(n_ctrl):
        rec = {
            "type": "control",
            "binary_score": float(ctrl_binary_scores[i]),
            "tc_context": 1 if all_seqs[n_mut + i][99] in "TU" else 0,
        }
        for enz_name in ENZYME_CLASSES:
            rec[f"P_{enz_name}"] = float(ctrl_multi[i, CLASS_TO_IDX[enz_name]])
        score_records.append(rec)

    scores_df = pd.DataFrame(score_records)
    # Also save binary-only format for compatibility
    binary_records = []
    for i in range(n_mut):
        binary_records.append({"type": "mutation", "score": float(mut_binary_scores[i]),
                               "tc_context": 1 if all_seqs[i][99] in "TU" else 0})
    for i in range(n_ctrl):
        binary_records.append({"type": "control", "score": float(ctrl_binary_scores[i]),
                               "tc_context": 1 if all_seqs[n_mut + i][99] in "TU" else 0})
    pd.DataFrame(binary_records).to_csv(SCORES_DIR / f"{CANCER}_scores.csv", index=False)
    logger.info(f"  Saved binary scores: {SCORES_DIR / f'{CANCER}_scores.csv'}")

    # Full scores with enzyme probabilities
    scores_df.to_csv(SCORES_DIR / f"{CANCER}_full_scores.csv", index=False)
    logger.info(f"  Saved full scores: {SCORES_DIR / f'{CANCER}_full_scores.csv'}")

    # =========================================================================
    # Step 9: Binary model enrichment + TC-stratified analysis
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 9: Binary model enrichment analysis")
    logger.info("=" * 70)

    # Overall enrichment
    enrichment = compute_enrichment(mut_binary_scores, ctrl_binary_scores)

    # Mann-Whitney
    mw_stat, mw_p = stats.mannwhitneyu(mut_binary_scores, ctrl_binary_scores, alternative="two-sided")

    # TC context
    tc_mut = sum(1 for s in mut_valid_seqs[:n_mut] if s[99] in "TU")
    tc_ctrl = sum(1 for s in ctrl_valid_seqs[:n_ctrl] if s[99] in "TU")

    binary_results = {
        "cancer_type": CANCER,
        "model": "GB_HandFeatures_40dim",
        "n_mutations": n_mut,
        "n_controls": n_ctrl,
        "mean_score_mutations": float(np.mean(mut_binary_scores)),
        "mean_score_controls": float(np.mean(ctrl_binary_scores)),
        "delta": float(np.mean(mut_binary_scores) - np.mean(ctrl_binary_scores)),
        "mann_whitney_p": float(mw_p),
        "enrichment": enrichment,
        "tc_frac_mutations": tc_mut / max(n_mut, 1),
        "tc_frac_controls": tc_ctrl / max(n_ctrl, 1),
        "tc_enrichment_OR": (tc_mut / max(n_mut - tc_mut, 1)) / (tc_ctrl / max(n_ctrl - tc_ctrl, 1))
            if tc_ctrl > 0 and n_ctrl > tc_ctrl else 0,
    }

    logger.info(f"\n  BINARY MODEL RESULTS:")
    logger.info(f"    n_mutations: {n_mut:,}, n_controls: {n_ctrl:,}")
    logger.info(f"    Mean score: mut={binary_results['mean_score_mutations']:.4f}, "
                f"ctrl={binary_results['mean_score_controls']:.4f}, "
                f"delta={binary_results['delta']:.4f}")
    logger.info(f"    Mann-Whitney p={mw_p:.2e}")
    logger.info(f"    TC context: mut={binary_results['tc_frac_mutations']:.3f}, "
                f"ctrl={binary_results['tc_frac_controls']:.3f}, "
                f"TC OR={binary_results['tc_enrichment_OR']:.3f}")
    logger.info(f"\n    Enrichment by threshold:")
    for t in ["0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "0.95"]:
        e = enrichment[t]
        logger.info(f"      OR@{t}: {e['OR']:.3f} (p={e['p']:.2e}, "
                    f"mut={e['frac_mut']:.3f}, ctrl={e['frac_ctrl']:.3f})")

    # TC-stratified analysis
    logger.info(f"\n  TC-STRATIFIED ANALYSIS:")
    tc_results = tc_stratified_analysis(all_seqs, binary_scores, n_mut, n_ctrl)
    for label, res in tc_results.items():
        if "note" in res:
            logger.info(f"    {label}: {res['note']}")
        else:
            logger.info(f"    {label}: n_mut={res['n_mut']:,}, n_ctrl={res['n_ctrl']:,}, "
                        f"delta={res['delta']:.4f}, MWU p={res['mann_whitney_p']:.2e}")
            for t in ["0.3", "0.5", "0.7"]:
                e = res["enrichment"][t]
                logger.info(f"      OR@{t}: {e['OR']:.3f} (p={e['p']:.2e})")

    binary_results["tc_stratified"] = tc_results

    # =========================================================================
    # Step 10: Multi-class enzyme model analysis
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 10: Multi-class enzyme model analysis")
    logger.info("=" * 70)

    neg_class_idx = CLASS_TO_IDX["Negative"]
    p_edited = 1.0 - multi_proba[:, neg_class_idx]

    cc_mut = sum(1 for s in mut_valid_seqs[:n_mut] if s and len(s) > 99 and s[99] == "C")
    cc_ctrl = sum(1 for s in ctrl_valid_seqs[:n_ctrl] if s and len(s) > 99 and s[99] == "C")

    enzyme_results = {
        "cancer": CANCER,
        "n_mutations": n_mut,
        "n_controls": n_ctrl,
        "tc_frac_mutations": tc_mut / max(n_mut, 1),
        "tc_frac_controls": tc_ctrl / max(n_ctrl, 1),
        "cc_frac_mutations": cc_mut / max(n_mut, 1),
        "cc_frac_controls": cc_ctrl / max(n_ctrl, 1),
        "enzymes": {},
    }

    logger.info(f"\n  {'Enzyme':<12} {'mean_mut':>10} {'mean_ctrl':>10} {'delta':>10} "
                f"{'MWU_p':>12} {'OR@p75':>10} {'OR@p90':>10}")
    logger.info("  " + "-" * 76)

    for enz_name in list(ENZYME_CLASSES[:-1]) + ["P_edited"]:
        if enz_name == "P_edited":
            mut_scores = p_edited[:n_mut]
            ctrl_scores = p_edited[n_mut:]
        else:
            enz_idx = CLASS_TO_IDX[enz_name]
            mut_scores = mut_multi[:, enz_idx]
            ctrl_scores = ctrl_multi[:, enz_idx]

        mw_stat, mw_p_val = stats.mannwhitneyu(
            mut_scores, ctrl_scores, alternative="greater"
        ) if n_mut > 0 and n_ctrl > 0 else (0, 1)

        enrichment_enz = compute_enrichment(
            mut_scores, ctrl_scores,
            thresholds=(0.1, 0.2, 0.3, 0.4, 0.5)
        )
        pct_enrichment = compute_percentile_enrichment(mut_scores, ctrl_scores)

        enzyme_results["enzymes"][enz_name] = {
            "mean_mut": float(np.mean(mut_scores)),
            "mean_ctrl": float(np.mean(ctrl_scores)),
            "delta": float(np.mean(mut_scores) - np.mean(ctrl_scores)),
            "mann_whitney_p": float(mw_p_val),
            "enrichment": enrichment_enz,
            "percentile_enrichment": pct_enrichment,
        }

        or_p75 = pct_enrichment.get("p75", {}).get("OR", float("nan"))
        or_p90 = pct_enrichment.get("p90", {}).get("OR", float("nan"))
        logger.info(f"  {enz_name:<12} {float(np.mean(mut_scores)):>10.4f} "
                    f"{float(np.mean(ctrl_scores)):>10.4f} "
                    f"{float(np.mean(mut_scores) - np.mean(ctrl_scores)):>10.4f} "
                    f"{float(mw_p_val):>12.2e} {or_p75:>10.3f} {or_p90:>10.3f}")

    # =========================================================================
    # Save all results
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("Saving results")
    logger.info("=" * 70)

    all_output = {
        "binary_model": binary_results,
        "multiclass_model": enzyme_results,
    }

    result_path = OUTPUT_DIR / f"{CANCER}_enrichment_results.json"
    with open(result_path, "w") as f:
        json.dump(all_output, f, indent=2, default=str)
    logger.info(f"  Results: {result_path}")

    # =========================================================================
    # Generate figure
    # =========================================================================
    logger.info("\n  Generating figure...")

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f"COADREAD (Colorectal) TCGA Enrichment Analysis\n"
                 f"n_mut={n_mut:,}, n_ctrl={n_ctrl:,}", fontsize=14)

    # Panel A: Binary model score distributions
    ax = axes[0, 0]
    ax.hist(mut_binary_scores, bins=50, alpha=0.6, label="Mutations", color="#dc2626", density=True)
    ax.hist(ctrl_binary_scores, bins=50, alpha=0.6, label="Controls", color="#3b82f6", density=True)
    ax.set_xlabel("Binary editability score")
    ax.set_ylabel("Density")
    ax.set_title("A) Score distributions (binary model)")
    ax.legend()
    ax.axvline(np.mean(mut_binary_scores), color="#dc2626", linestyle="--", alpha=0.7)
    ax.axvline(np.mean(ctrl_binary_scores), color="#3b82f6", linestyle="--", alpha=0.7)

    # Panel B: Binary enrichment by threshold
    ax = axes[0, 1]
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    ors = [enrichment[str(t)]["OR"] for t in thresholds]
    ax.plot(thresholds, ors, marker="o", color="#dc2626", linewidth=2)
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Editability threshold")
    ax.set_ylabel("Odds Ratio")
    ax.set_title("B) Binary model enrichment")
    for t, orv in zip(thresholds, ors):
        if not np.isnan(orv):
            ax.annotate(f"{orv:.2f}", (t, orv), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=7)

    # Panel C: Per-enzyme mean probability (mut vs ctrl)
    ax = axes[0, 2]
    key_enzymes = ["A3A", "A3B", "A3G", "Neither", "P_edited"]
    x = np.arange(len(key_enzymes))
    mut_means = [enzyme_results["enzymes"][e]["mean_mut"] for e in key_enzymes]
    ctrl_means = [enzyme_results["enzymes"][e]["mean_ctrl"] for e in key_enzymes]
    w = 0.35
    ax.bar(x - w/2, mut_means, w, label="Mutations", color="#dc2626", alpha=0.8)
    ax.bar(x + w/2, ctrl_means, w, label="Controls", color="#3b82f6", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(key_enzymes, rotation=30, ha="right")
    ax.set_ylabel("Mean P(enzyme)")
    ax.set_title("C) Multi-class enzyme probabilities")
    ax.legend(fontsize=8)

    # Panel D: Per-enzyme OR at percentile 75 and 90
    ax = axes[1, 0]
    all_enz = ["A3A", "A3B", "A3G", "A3A_A3G", "Neither", "Unknown", "P_edited"]
    or_p75 = [enzyme_results["enzymes"][e]["percentile_enrichment"]["p75"]["OR"] for e in all_enz]
    or_p90 = [enzyme_results["enzymes"][e]["percentile_enrichment"]["p90"]["OR"] for e in all_enz]
    x2 = np.arange(len(all_enz))
    ax.bar(x2 - 0.2, or_p75, 0.35, label="OR @ P75", color="#f59e0b", alpha=0.8)
    ax.bar(x2 + 0.2, or_p90, 0.35, label="OR @ P90", color="#dc2626", alpha=0.8)
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xticks(x2)
    ax.set_xticklabels(all_enz, rotation=30, ha="right")
    ax.set_ylabel("Odds Ratio")
    ax.set_title("D) Per-enzyme enrichment (percentile-based)")
    ax.legend(fontsize=8)

    # Panel E: TC-stratified binary enrichment
    ax = axes[1, 1]
    tc_labels = []
    tc_ors_05 = []
    tc_colors_plot = []
    for label, color in [("TC_only", "#dc2626"), ("nonTC_only", "#3b82f6")]:
        res = tc_results.get(label, {})
        if "enrichment" in res:
            tc_labels.append(label)
            tc_ors_05.append(res["enrichment"]["0.5"]["OR"])
            tc_colors_plot.append(color)
    if tc_labels:
        bars = ax.bar(range(len(tc_labels)), tc_ors_05, color=tc_colors_plot, alpha=0.8)
        ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
        ax.set_xticks(range(len(tc_labels)))
        ax.set_xticklabels(tc_labels)
        ax.set_ylabel("Odds Ratio @ 0.5")
        ax.set_title("E) TC-stratified enrichment (binary)")
        for bar, orv in zip(bars, tc_ors_05):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{orv:.3f}", ha="center", fontsize=9)

    # Panel F: Comparison with other GI cancers (read from saved results)
    ax = axes[1, 2]
    gi_comparison = {"COADREAD": binary_results}
    # Try to load other cancer results for comparison
    for other_cancer in ["esca", "stad", "lihc", "blca", "skcm"]:
        other_file = OUTPUT_DIR / f"tcga_full_model_results.json"
        if other_file.exists():
            try:
                with open(other_file) as f:
                    other_data = json.load(f)
                if other_cancer in other_data:
                    gi_comparison[other_cancer.upper()] = other_data[other_cancer]
            except Exception:
                pass

    cancer_labels = list(gi_comparison.keys())
    deltas = []
    for c_name in cancer_labels:
        d = gi_comparison[c_name].get("delta", 0)
        deltas.append(d)

    colors_bar = []
    for c_name in cancer_labels:
        if c_name in ["COADREAD", "ESCA", "STAD", "LIHC"]:
            colors_bar.append("#16a34a")  # GI green
        elif c_name == "SKCM":
            colors_bar.append("#94a3b8")  # control gray
        else:
            colors_bar.append("#dc2626")  # APOBEC red

    if len(cancer_labels) > 1:
        bars = ax.bar(range(len(cancer_labels)), deltas, color=colors_bar, alpha=0.8)
        ax.set_xticks(range(len(cancer_labels)))
        ax.set_xticklabels(cancer_labels, rotation=30, ha="right")
        ax.set_ylabel("Mean score delta\n(mutations - controls)")
        ax.set_title("F) COADREAD vs other cancers")
        ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
    else:
        ax.text(0.5, 0.5, "Run other cancers first\nfor comparison",
                transform=ax.transAxes, ha="center", va="center")
        ax.set_title("F) Comparison (no other data)")

    plt.tight_layout()
    fig_path = OUTPUT_DIR / f"{CANCER}_enrichment.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Figure: {fig_path}")

    # =========================================================================
    # Final summary
    # =========================================================================
    elapsed = time.time() - t_start
    logger.info(f"\n{'=' * 70}")
    logger.info(f"COADREAD ANALYSIS COMPLETE ({elapsed/60:.1f} min)")
    logger.info(f"{'=' * 70}")

    logger.info(f"\n===== COADREAD KEY RESULTS =====\n")
    logger.info(f"Binary model (40-dim GB):")
    logger.info(f"  Mean score: mut={binary_results['mean_score_mutations']:.4f}, "
                f"ctrl={binary_results['mean_score_controls']:.4f}")
    logger.info(f"  Delta: {binary_results['delta']:.4f}")
    logger.info(f"  Mann-Whitney p: {binary_results['mann_whitney_p']:.2e}")
    logger.info(f"  TC context: mut={binary_results['tc_frac_mutations']:.3f}, "
                f"ctrl={binary_results['tc_frac_controls']:.3f}")

    logger.info(f"\nEnrichment (binary model):")
    for t in ["0.3", "0.5", "0.7", "0.9"]:
        e = enrichment[t]
        logger.info(f"  OR@{t}: {e['OR']:.3f} (p={e['p']:.2e})")

    logger.info(f"\nMulti-class enzyme probabilities (mean delta, mut-ctrl):")
    for enz in ["A3A", "A3G", "Neither", "P_edited"]:
        r = enzyme_results["enzymes"][enz]
        logger.info(f"  P({enz}): delta={r['delta']:+.5f}, "
                    f"MWU p={r['mann_whitney_p']:.2e}, "
                    f"OR@p75={r['percentile_enrichment']['p75']['OR']:.3f}, "
                    f"OR@p90={r['percentile_enrichment']['p90']['OR']:.3f}")

    logger.info(f"\n===== HYPOTHESIS: APOBEC1 territory =====")
    neither_r = enzyme_results["enzymes"]["Neither"]
    a3a_r = enzyme_results["enzymes"]["A3A"]
    logger.info(f"  P(Neither/APOBEC1): delta={neither_r['delta']:+.5f}, "
                f"OR@p75={neither_r['percentile_enrichment']['p75']['OR']:.3f}, "
                f"OR@p90={neither_r['percentile_enrichment']['p90']['OR']:.3f}")
    logger.info(f"  P(A3A): delta={a3a_r['delta']:+.5f}, "
                f"OR@p75={a3a_r['percentile_enrichment']['p75']['OR']:.3f}, "
                f"OR@p90={a3a_r['percentile_enrichment']['p90']['OR']:.3f}")

    if neither_r["percentile_enrichment"]["p75"]["OR"] > a3a_r["percentile_enrichment"]["p75"]["OR"]:
        logger.info("  --> P(Neither) shows STRONGER enrichment than P(A3A) -- consistent with APOBEC1 hypothesis!")
    else:
        logger.info("  --> P(A3A) shows stronger enrichment than P(Neither)")

    logger.info(f"\nOutput files:")
    logger.info(f"  {result_path}")
    logger.info(f"  {SCORES_DIR / f'{CANCER}_scores.csv'}")
    logger.info(f"  {SCORES_DIR / f'{CANCER}_full_scores.csv'}")
    logger.info(f"  {vienna_cache_file}")
    logger.info(f"  {fig_path}")


if __name__ == "__main__":
    main()
