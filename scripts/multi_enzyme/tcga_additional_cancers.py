#!/usr/bin/env python3
"""TCGA somatic mutation enrichment for 5 ADDITIONAL cancer types.

Same pipeline as tcga_full_model_enrichment.py but for:
  hnsc, esca, stad, lihc, coad

Downloads MAFs from cBioPortal, parses C>T/G>A SNPs, generates matched controls,
folds with ViennaRNA (14 workers), caches raw structures, scores with full 40-dim
GB model, and computes enrichment ORs.

Usage:
    conda run -n quris python scripts/multi_enzyme/tcga_additional_cancers.py
"""

import gc
import gzip
import json
import logging
import multiprocessing as mp
import os
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyfaidx import Fasta
from scipy import stats
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.apobec_feature_extraction import (
    extract_motif_from_seq,
    extract_motif_features,
    extract_loop_features,
    extract_structure_delta_features,
    LOOP_FEATURE_COLS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = PROJECT_ROOT / "data"
HG19_FA = DATA_DIR / "raw/genomes/hg19.fa"
REFGENE_HG19 = DATA_DIR / "raw/genomes/refGene_hg19.txt"
MULTI_SPLITS = DATA_DIR / "processed/multi_enzyme/splits_multi_enzyme_v3_with_negatives.csv"
MULTI_SEQS = DATA_DIR / "processed/multi_enzyme/multi_enzyme_sequences_v3_with_negatives.json"
LOOP_CSV = DATA_DIR / "processed/multi_enzyme/loop_position_per_site_v3.csv"
STRUCT_CACHE = DATA_DIR / "processed/multi_enzyme/structure_cache_multi_enzyme_v3.npz"

OUTPUT_DIR = PROJECT_ROOT / "experiments/multi_enzyme/outputs/tcga_gnomad"
TCGA_CACHE = DATA_DIR / "raw/tcga"

N_CONTROLS = 5
N_WORKERS = 14
SEED = 42

CBIO_BASE = "https://media.githubusercontent.com/media/cBioPortal/datahub/master/public"
CANCER_STUDIES = {
    "hnsc": "hnsc_tcga_pan_can_atlas_2018",
    "esca": "esca_tcga_pan_can_atlas_2018",
    "stad": "stad_tcga_pan_can_atlas_2018",
    "lihc": "lihc_tcga_pan_can_atlas_2018",
    "coad": "coad_tcga_pan_can_atlas_2018",
}


# ============================================================================
# ViennaRNA folding (parallelizable)
# ============================================================================

def fold_single_sequence(seq_201nt):
    """Fold a single 201-nt sequence with ViennaRNA.

    Returns RAW fold data (structure strings, MFE, pairing probabilities at center +/-10)
    that can be cached and reused to derive ANY downstream features.
    Returns None on failure.
    """
    import RNA

    if seq_201nt is None or len(seq_201nt) != 201:
        return None

    center = 100
    seq = seq_201nt.upper().replace("U", "T")

    if seq[center] != "C":
        return None

    try:
        # WT fold
        fc_wt = RNA.fold_compound(seq)
        struct_wt, mfe_wt = fc_wt.mfe()
        fc_wt.exp_params_rescale(mfe_wt)
        fc_wt.pf()
        bpp_wt = fc_wt.bpp()

        # Edited fold (C->T at center)
        seq_ed = seq[:center] + "T" + seq[center + 1:]
        fc_ed = RNA.fold_compound(seq_ed)
        struct_ed, mfe_ed = fc_ed.mfe()
        fc_ed.exp_params_rescale(mfe_ed)
        fc_ed.pf()
        bpp_ed = fc_ed.bpp()

        # Extract pairing probabilities at center +/-10
        c = center + 1  # 1-indexed for bpp
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
    """Fold a batch of sequences. Returns list of results (same order, None for failures)."""
    return [fold_single_sequence(s) for s in sequences]


def derive_features_from_fold(fold_result):
    """Derive 16-dim structure features from cached raw fold data.

    Returns dict with struct_delta (7-dim) and loop_features (9-dim), or None.
    """
    if fold_result is None:
        return None

    center = 100

    # Structure delta (7-dim)
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

    # Loop geometry (9-dim) from WT structure string
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


def parallel_fold(sequences, n_workers=N_WORKERS, chunk_size=500):
    """Fold sequences in parallel using multiprocessing."""
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


# ============================================================================
# Build 40-dim features from fold results + sequences
# ============================================================================

def build_40dim_features(sequences, fold_results):
    """Build 40-dim feature matrix from sequences and their raw fold results."""
    n = len(sequences)
    features = np.zeros((n, 40), dtype=np.float32)

    for i, (seq, fold_raw) in enumerate(zip(sequences, fold_results)):
        # Motif features (24-dim)
        if seq is not None and len(seq) == 201:
            features[i, :24] = extract_motif_from_seq(seq)

        # Structure features (16-dim)
        derived = derive_features_from_fold(fold_raw)
        if derived is not None:
            features[i, 24:31] = derived["struct_delta"]   # 7-dim
            features[i, 31:40] = derived["loop_features"]  # 9-dim

    return np.nan_to_num(features, nan=0.0)


# ============================================================================
# Train full GB model on multi-enzyme v3 data
# ============================================================================

def train_full_model():
    """Train GB_HandFeatures (40-dim) on multi-enzyme v3 data."""
    logger.info("Training full 40-dim GB model...")

    splits = pd.read_csv(MULTI_SPLITS)
    with open(MULTI_SEQS) as f:
        sequences = json.load(f)

    site_ids = splits["site_id"].astype(str).tolist()
    labels = splits["is_edited"].values

    # Motif features
    motif_feats = np.array([extract_motif_from_seq(sequences.get(sid, "N" * 201))
                            for sid in site_ids], dtype=np.float32)

    # Loop features from pre-computed CSV
    loop_df = pd.read_csv(LOOP_CSV)
    loop_df["site_id"] = loop_df["site_id"].astype(str)
    loop_df = loop_df.set_index("site_id")
    loop_cols = LOOP_FEATURE_COLS
    loop_feats = np.zeros((len(site_ids), len(loop_cols)), dtype=np.float32)
    for i, sid in enumerate(site_ids):
        if sid in loop_df.index:
            for j, col in enumerate(loop_cols):
                if col in loop_df.columns:
                    val = loop_df.loc[sid, col]
                    if not isinstance(val, (int, float)):
                        val = val.iloc[0] if hasattr(val, 'iloc') else 0.0
                    loop_feats[i, j] = float(val) if pd.notna(val) else 0.0

    # Structure delta from cache
    struct_feats = np.zeros((len(site_ids), 7), dtype=np.float32)
    if STRUCT_CACHE.exists():
        data = np.load(str(STRUCT_CACHE), allow_pickle=True)
        struct_sids = [str(s) for s in data["site_ids"]]
        sid_to_idx = {s: i for i, s in enumerate(struct_sids)}
        for i, sid in enumerate(site_ids):
            if sid in sid_to_idx:
                struct_feats[i] = data["delta_features"][sid_to_idx[sid]]
        del data
        gc.collect()

    # Combine: 24 motif + 7 struct + 9 loop = 40
    X = np.concatenate([motif_feats, struct_feats, loop_feats], axis=1)
    X = np.nan_to_num(X, nan=0.0)

    logger.info(f"Training on {len(labels):,} sites ({labels.sum():,} pos), {X.shape[1]} features")

    model = XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=SEED, eval_metric="logloss", use_label_encoder=False,
    )
    model.fit(X, labels)

    from sklearn.metrics import roc_auc_score
    train_auc = roc_auc_score(labels, model.predict_proba(X)[:, 1])
    logger.info(f"Full model training AUC: {train_auc:.4f}")

    return model


# ============================================================================
# Download and parse TCGA mutations
# ============================================================================

def download_maf(cancer_type):
    """Download MAF file, cache locally."""
    TCGA_CACHE.mkdir(parents=True, exist_ok=True)
    study = CANCER_STUDIES[cancer_type]
    local = TCGA_CACHE / f"{study}_mutations.txt"
    if local.exists():
        logger.info(f"Cached: {local} ({local.stat().st_size / 1e6:.0f} MB)")
        return local

    url = f"{CBIO_BASE}/{study}/data_mutations.txt"
    logger.info(f"Downloading {cancer_type} MAF from {url}...")
    urllib.request.urlretrieve(url, local)
    logger.info(f"Downloaded: {local.stat().st_size / 1e6:.0f} MB")
    return local


def parse_ct_mutations(maf_path):
    """Parse C>T and G>A SNPs from MAF. Returns DataFrame."""
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
            sub["pos"] = sub["Start_Position"].astype(int) - 1  # 0-based
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
    """For each mutation, pick n_controls random C positions in the same gene's exons."""
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


# ============================================================================
# Enrichment analysis
# ============================================================================

def compute_enrichment(mut_scores, ctrl_scores,
                       thresholds=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)):
    """Compute enrichment ORs at multiple thresholds."""
    results = {}
    for t in thresholds:
        mut_above = int((mut_scores >= t).sum())
        mut_below = int((mut_scores < t).sum())
        ctrl_above = int((ctrl_scores >= t).sum())
        ctrl_below = int((ctrl_scores < t).sum())

        if mut_below > 0 and ctrl_above > 0 and ctrl_below > 0:
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


# ============================================================================
# Main
# ============================================================================

def main():
    t_start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Parse exonic regions for control matching
    logger.info("=" * 60)
    logger.info("Step 1: Parse exonic regions from refGene_hg19.txt")
    exons = []
    exons_by_gene = defaultdict(list)
    with open(REFGENE_HG19) as f:
        seen = set()
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
                    exon_tuple = (chrom, es, ee, strand, gene)
                    exons.append(exon_tuple)
                    exons_by_gene[gene].append(exon_tuple)
    logger.info(f"  {len(exons):,} exonic regions, {len(exons_by_gene):,} genes")

    # Step 2: Train full model
    logger.info("=" * 60)
    logger.info("Step 2: Train full 40-dim GB model")
    model = train_full_model()

    # Step 3: Load genome
    logger.info("=" * 60)
    logger.info("Step 3: Load hg19 genome")
    genome = Fasta(str(HG19_FA))

    # Step 4: Process each cancer type
    all_results = {}
    cancer_list = ["hnsc", "esca", "stad", "lihc", "coad"]

    for cancer in cancer_list:
        logger.info("=" * 60)
        logger.info(f"Processing {cancer.upper()}")

        # Download MAF
        maf_path = download_maf(cancer)
        if maf_path is None:
            continue

        # Parse mutations
        mut_df = parse_ct_mutations(maf_path)
        if len(mut_df) == 0:
            logger.warning(f"No mutations for {cancer}")
            continue

        # Get matched controls
        logger.info(f"  Generating {N_CONTROLS} controls per mutation...")
        ctrl_df = get_matched_controls(mut_df, genome, exons_by_gene, N_CONTROLS)
        logger.info(f"  {len(ctrl_df):,} control positions")

        # Extract sequences
        logger.info("  Extracting mutation sequences...")
        mut_seqs = extract_sequences(mut_df, genome)
        logger.info("  Extracting control sequences...")
        ctrl_seqs = extract_sequences(ctrl_df, genome)

        # Filter valid sequences (201-nt with C at center)
        valid_mut = [(i, s) for i, s in enumerate(mut_seqs)
                     if s is not None and len(s) == 201 and s[100] == "C"]
        valid_ctrl = [(i, s) for i, s in enumerate(ctrl_seqs)
                      if s is not None and len(s) == 201 and s[100] == "C"]

        logger.info(f"  Valid: {len(valid_mut):,} mutations, {len(valid_ctrl):,} controls")

        mut_indices, mut_valid_seqs = zip(*valid_mut) if valid_mut else ([], [])
        ctrl_indices, ctrl_valid_seqs = zip(*valid_ctrl) if valid_ctrl else ([], [])

        all_seqs = list(mut_valid_seqs) + list(ctrl_valid_seqs)
        n_mut = len(mut_valid_seqs)
        n_ctrl = len(ctrl_valid_seqs)

        # ---- ViennaRNA folding with RAW STRUCTURE CACHE ----
        cache_dir = OUTPUT_DIR / "vienna_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        vienna_cache_file = cache_dir / f"{cancer}_vienna_raw.json.gz"

        if vienna_cache_file.exists():
            logger.info(f"  Loading cached ViennaRNA structures from {vienna_cache_file}")
            with gzip.open(str(vienna_cache_file), "rt") as fz:
                cache_data = json.load(fz)
            fold_results = cache_data["fold_results"]
            n_mut = cache_data["n_mut"]
            n_ctrl = cache_data["n_ctrl"]
            all_seqs = list(mut_valid_seqs) + list(ctrl_valid_seqs)
            logger.info(f"  Loaded {len(fold_results):,} cached folds")
        else:
            # Fold all sequences with ViennaRNA (parallel, 14 workers)
            logger.info(f"  Folding {len(all_seqs):,} sequences with {N_WORKERS} workers...")
            fold_results = parallel_fold(all_seqs, N_WORKERS)

            # Cache RAW ViennaRNA structures
            logger.info(f"  Caching {len(fold_results):,} ViennaRNA structures to {vienna_cache_file}")
            cache_data = {
                "cancer": cancer,
                "n_mut": n_mut,
                "n_ctrl": n_ctrl,
                "n_sequences": len(all_seqs),
                "fold_results": fold_results,
            }
            with gzip.open(str(vienna_cache_file), "wt") as fz:
                json.dump(cache_data, fz)
            logger.info(f"  Cache saved ({vienna_cache_file.stat().st_size / 1e6:.0f} MB)")

        # Build 40-dim features from raw folds
        logger.info("  Building 40-dim features from ViennaRNA structures...")
        features_40d = build_40dim_features(all_seqs, fold_results)

        # Score with full model
        scores = model.predict_proba(features_40d)[:, 1]
        mut_scores = scores[:n_mut]
        ctrl_scores = scores[n_mut:]

        # Save raw scores to CSV
        scores_dir = OUTPUT_DIR / "raw_scores"
        scores_dir.mkdir(parents=True, exist_ok=True)
        score_records = []
        for i in range(n_mut):
            score_records.append({"type": "mutation", "score": float(mut_scores[i]),
                                  "tc_context": 1 if all_seqs[i][99] in "TU" else 0})
        for i in range(n_ctrl):
            score_records.append({"type": "control", "score": float(ctrl_scores[i]),
                                  "tc_context": 1 if all_seqs[n_mut + i][99] in "TU" else 0})
        pd.DataFrame(score_records).to_csv(scores_dir / f"{cancer}_scores.csv", index=False)
        logger.info(f"  Saved {len(score_records):,} scores to {scores_dir / f'{cancer}_scores.csv'}")

        # Compute enrichment
        enrichment = compute_enrichment(mut_scores, ctrl_scores)

        # TC context analysis
        tc_mut = sum(1 for s in mut_valid_seqs if s[99] in "TU")
        tc_ctrl = sum(1 for s in ctrl_valid_seqs if s[99] in "TU")

        # Mann-Whitney
        if n_mut > 0 and n_ctrl > 0:
            mw_stat, mw_p = stats.mannwhitneyu(mut_scores, ctrl_scores, alternative="two-sided")
        else:
            mw_stat, mw_p = 0, 1

        result = {
            "cancer_type": cancer,
            "model": "GB_HandFeatures_40dim",
            "n_mutations": n_mut,
            "n_controls": n_ctrl,
            "mean_score_mutations": float(np.mean(mut_scores)) if n_mut > 0 else 0,
            "mean_score_controls": float(np.mean(ctrl_scores)) if n_ctrl > 0 else 0,
            "delta": float(np.mean(mut_scores) - np.mean(ctrl_scores)) if n_mut > 0 else 0,
            "mann_whitney_p": float(mw_p),
            "enrichment": enrichment,
            "tc_frac_mutations": tc_mut / max(n_mut, 1),
            "tc_frac_controls": tc_ctrl / max(n_ctrl, 1),
            "tc_enrichment_OR": (tc_mut / max(n_mut - tc_mut, 1)) / (tc_ctrl / max(n_ctrl - tc_ctrl, 1))
                if tc_ctrl > 0 and n_ctrl > tc_ctrl else 0,
        }

        all_results[cancer] = result

        logger.info(f"  {cancer.upper()} RESULTS:")
        logger.info(f"    Mean score: mut={result['mean_score_mutations']:.4f}, "
                     f"ctrl={result['mean_score_controls']:.4f}, "
                     f"delta={result['delta']:.4f}")
        logger.info(f"    Mann-Whitney p={mw_p:.2e}")
        for t in ["0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "0.95"]:
            e = enrichment[t]
            logger.info(f"    OR@{t}: {e['OR']:.3f} (p={e['p']:.2e})")

        # Free memory
        del features_40d, scores, fold_results
        gc.collect()

    # Save results
    results_file = OUTPUT_DIR / "tcga_additional_cancers_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    # Generate figure
    logger.info("Generating figures...")
    cancers = [c for c in cancer_list if c in all_results]
    if not cancers:
        logger.warning("No cancer results to plot.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Mean scores
    ax = axes[0]
    x = np.arange(len(cancers))
    mut_means = [all_results[c]["mean_score_mutations"] for c in cancers]
    ctrl_means = [all_results[c]["mean_score_controls"] for c in cancers]
    w = 0.35
    ax.bar(x - w / 2, mut_means, w, label="C>T mutations", color="#dc2626", alpha=0.8)
    ax.bar(x + w / 2, ctrl_means, w, label="Matched controls", color="#94a3b8", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([c.upper() for c in cancers])
    ax.set_ylabel("Mean editability score")
    ax.set_title("Full 40-dim GB Model (Additional Cancers)")
    ax.legend()

    # Panel 2: OR at multiple thresholds
    ax = axes[1]
    for cancer in cancers:
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        ors = [all_results[cancer]["enrichment"].get(str(t), {}).get("OR", float("nan"))
               for t in thresholds]
        ax.plot(thresholds, ors, marker="o", label=cancer.upper(), linewidth=2)
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Editability threshold")
    ax.set_ylabel("Odds Ratio (mutation enrichment)")
    ax.set_title("Enrichment by threshold")
    ax.legend(fontsize=8)

    # Panel 3: OR at t=0.5
    ax = axes[2]
    ors_05 = [all_results[c]["enrichment"]["0.5"]["OR"] for c in cancers]
    colors = ["#dc2626", "#f59e0b", "#10b981", "#3b82f6", "#8b5cf6"]
    bars = ax.bar(x, ors_05, color=colors[:len(cancers)], alpha=0.8)
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([c.upper() for c in cancers])
    ax.set_ylabel("Odds Ratio")
    ax.set_title("OR at threshold 0.5")
    for bar, orval in zip(bars, ors_05):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{orval:.2f}", ha="center", fontsize=9)

    plt.tight_layout()
    fig_path = OUTPUT_DIR / "tcga_additional_cancers_enrichment.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()

    elapsed = time.time() - t_start
    logger.info(f"\n{'='*60}")
    logger.info(f"COMPLETE ({elapsed/60:.0f} min)")
    logger.info(f"Results: {results_file}")
    logger.info(f"Figure: {fig_path}")


if __name__ == "__main__":
    main()
