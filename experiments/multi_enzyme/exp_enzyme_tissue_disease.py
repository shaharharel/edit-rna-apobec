#!/usr/bin/env python3
"""B2: Enzyme-Tissue-Disease Triangle.

Train enzyme-specific models (A3A, A3G, Neither/APOBEC1) and score TCGA cancer
mutations with ALL three models. Tests the tissue-disease hypothesis:
- A3A (blood/immune) -> should predict APOBEC-mutated cancers (BLCA, CESC, LUSC, BRCA)
- A3G (blood/lymphoid) -> similar pattern but CC-context specific
- Neither/APOBEC1 (intestine) -> should predict colorectal mutations (COAD) better

Uses pre-computed ViennaRNA caches from tcga_full_model_enrichment.py.

Usage:
    conda run -n quris python experiments/multi_enzyme/exp_enzyme_tissue_disease.py
"""

import gc
import gzip
import json
import logging
import os
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
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.apobec_feature_extraction import (
    extract_motif_from_seq,
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

TCGA_CACHE = DATA_DIR / "raw/tcga"
VIENNA_CACHE_DIR = PROJECT_ROOT / "experiments/multi_enzyme/outputs/tcga_gnomad/vienna_cache"
OUTPUT_DIR = PROJECT_ROOT / "experiments/multi_enzyme/outputs/tcga_gnomad/enzyme_tissue"

SEED = 42
N_CONTROLS = 5

# Cancer types with existing caches
CANCERS_WITH_CACHE = ["blca", "brca", "cesc", "lusc"]

# Additional cancers to download (COAD for intestine hypothesis)
CBIO_BASE = "https://media.githubusercontent.com/media/cBioPortal/datahub/master/public"
ADDITIONAL_CANCERS = {
    "coad": "coad_tcga_pan_can_atlas_2018",
}


# ============================================================================
# Feature building from cached fold results
# ============================================================================

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


# ============================================================================
# Train enzyme-specific models
# ============================================================================

def train_enzyme_model(enzyme_name, splits_df, sequences, loop_df, struct_data):
    """Train a GB model on sites for a specific enzyme category."""
    mask = splits_df["enzyme"] == enzyme_name
    enzyme_df = splits_df[mask].copy()

    if len(enzyme_df) == 0:
        logger.warning(f"No data for enzyme {enzyme_name}")
        return None

    site_ids = enzyme_df["site_id"].astype(str).tolist()
    labels = enzyme_df["is_edited"].values

    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    logger.info(f"  {enzyme_name}: {n_pos} positives, {n_neg} negatives")

    # Motif (24-dim)
    motif_feats = np.array([extract_motif_from_seq(sequences.get(sid, "N" * 201))
                            for sid in site_ids], dtype=np.float32)

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
                        val = val.iloc[0] if hasattr(val, 'iloc') else 0.0
                    loop_feats[i, j] = float(val) if pd.notna(val) else 0.0

    X = np.concatenate([motif_feats, struct_feats, loop_feats], axis=1)
    X = np.nan_to_num(X, nan=0.0)

    model = XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=SEED, eval_metric="logloss", use_label_encoder=False,
    )
    model.fit(X, labels)

    from sklearn.metrics import roc_auc_score
    train_auc = roc_auc_score(labels, model.predict_proba(X)[:, 1])
    logger.info(f"  {enzyme_name} training AUC: {train_auc:.4f}")

    return model


# ============================================================================
# TCGA mutation parsing and sequence extraction
# ============================================================================

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


def download_maf(cancer_type):
    """Download MAF file for additional cancers."""
    TCGA_CACHE.mkdir(parents=True, exist_ok=True)
    study = ADDITIONAL_CANCERS.get(cancer_type)
    if study is None:
        return None
    local = TCGA_CACHE / f"{study}_mutations.txt"
    if local.exists():
        logger.info(f"Cached: {local} ({local.stat().st_size / 1e6:.0f} MB)")
        return local

    import urllib.request
    url = f"{CBIO_BASE}/{study}/data_mutations.txt"
    logger.info(f"Downloading {cancer_type} MAF from {url}...")
    try:
        urllib.request.urlretrieve(url, local)
        logger.info(f"Downloaded: {local.stat().st_size / 1e6:.0f} MB")
        return local
    except Exception as e:
        logger.error(f"Failed to download {cancer_type}: {e}")
        return None


# ============================================================================
# Enrichment computation
# ============================================================================

def compute_enrichment(mut_scores, ctrl_scores,
                       thresholds=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8)):
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

    # ---- Step 1: Load training data ----
    logger.info("=" * 60)
    logger.info("Step 1: Load training data and train enzyme-specific models")

    splits = pd.read_csv(MULTI_SPLITS)
    with open(MULTI_SEQS) as f:
        sequences = json.load(f)

    loop_df = pd.read_csv(LOOP_CSV)
    loop_df["site_id"] = loop_df["site_id"].astype(str)
    loop_df = loop_df.set_index("site_id")

    struct_data = None
    if STRUCT_CACHE.exists():
        struct_data = dict(np.load(str(STRUCT_CACHE), allow_pickle=True))
        struct_data["site_ids"] = [str(s) for s in struct_data["site_ids"]]

    # Train enzyme-specific models
    enzyme_models = {}
    for enzyme in ["A3A", "A3G", "Neither"]:
        logger.info(f"\nTraining {enzyme} model...")
        model = train_enzyme_model(enzyme, splits, sequences, loop_df, struct_data)
        if model is not None:
            enzyme_models[enzyme] = model

    # Also train unified (all enzymes) model for comparison
    logger.info("\nTraining unified (all enzymes) model...")
    unified_model = train_enzyme_model_all(splits, sequences, loop_df, struct_data)
    if unified_model is not None:
        enzyme_models["Unified"] = unified_model

    del struct_data
    gc.collect()

    logger.info(f"\nTrained {len(enzyme_models)} models: {list(enzyme_models.keys())}")

    # ---- Step 2: Parse exons and load genome ----
    logger.info("=" * 60)
    logger.info("Step 2: Parse exons and load genome")
    exons_by_gene = parse_exons(REFGENE_HG19)
    genome = Fasta(str(HG19_FA))
    logger.info(f"  {len(exons_by_gene):,} genes with exonic regions")

    # ---- Step 3: Process each cancer type ----
    all_cancers = CANCERS_WITH_CACHE + list(ADDITIONAL_CANCERS.keys())
    all_results = {}

    for cancer in all_cancers:
        logger.info("=" * 60)
        logger.info(f"Processing {cancer.upper()}")

        # Get MAF file
        maf_path = TCGA_CACHE / f"{cancer}_tcga_pan_can_atlas_2018_mutations.txt"
        if not maf_path.exists():
            maf_path = download_maf(cancer)
        if maf_path is None or not maf_path.exists():
            logger.warning(f"No MAF for {cancer}, skipping")
            continue

        # Parse mutations
        mut_df = parse_ct_mutations(maf_path)
        if len(mut_df) == 0:
            logger.warning(f"No C>T mutations for {cancer}")
            continue
        logger.info(f"  {len(mut_df):,} unique C>T mutation positions")

        # Extract sequences
        logger.info("  Extracting mutation sequences...")
        mut_seqs = extract_sequences(mut_df, genome)

        # Get controls
        logger.info("  Generating matched controls...")
        ctrl_df = get_matched_controls(mut_df, genome, exons_by_gene, N_CONTROLS)
        logger.info(f"  {len(ctrl_df):,} control positions")
        ctrl_seqs = extract_sequences(ctrl_df, genome)

        # Filter valid sequences
        valid_mut = [(i, s) for i, s in enumerate(mut_seqs)
                     if s is not None and len(s) == 201 and s[100] == "C"]
        valid_ctrl = [(i, s) for i, s in enumerate(ctrl_seqs)
                      if s is not None and len(s) == 201 and s[100] == "C"]

        mut_indices, mut_valid_seqs = zip(*valid_mut) if valid_mut else ([], [])
        ctrl_indices, ctrl_valid_seqs = zip(*valid_ctrl) if valid_ctrl else ([], [])

        all_seqs = list(mut_valid_seqs) + list(ctrl_valid_seqs)
        n_mut = len(mut_valid_seqs)
        n_ctrl = len(ctrl_valid_seqs)
        logger.info(f"  Valid: {n_mut:,} mutations, {n_ctrl:,} controls")

        # Load cached ViennaRNA folds if available
        vienna_cache_file = VIENNA_CACHE_DIR / f"{cancer}_vienna_raw.json.gz"
        fold_results = None

        if vienna_cache_file.exists():
            logger.info(f"  Loading cached ViennaRNA structures...")
            with gzip.open(str(vienna_cache_file), "rt") as fz:
                cache_data = json.load(fz)
            fold_results = cache_data["fold_results"]
            cached_n_mut = cache_data["n_mut"]
            cached_n_ctrl = cache_data["n_ctrl"]

            # Validate cache matches current data
            if len(fold_results) == cached_n_mut + cached_n_ctrl:
                n_mut = cached_n_mut
                n_ctrl = cached_n_ctrl
                logger.info(f"  Using cached folds: {len(fold_results):,} entries "
                            f"({n_mut:,} mut + {n_ctrl:,} ctrl)")
            else:
                logger.warning(f"  Cache size mismatch, will fold from scratch")
                fold_results = None

        if fold_results is None:
            # Need to fold with ViennaRNA
            import multiprocessing as mp
            n_workers = max(1, mp.cpu_count() - 2)
            logger.info(f"  Folding {len(all_seqs):,} sequences with {n_workers} workers...")

            def fold_single(seq):
                import RNA
                if seq is None or len(seq) != 201 or seq[100] != "C":
                    return None
                try:
                    center = 100
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

                    c = center + 1
                    w = 10
                    bpp_wt_w = []
                    bpp_ed_w = []
                    for idx in range(max(1, c - w), min(len(bpp_wt), c + w + 1)):
                        bpp_wt_w.append(float(sum(bpp_wt[idx])) if idx < len(bpp_wt) else 0.5)
                        bpp_ed_w.append(float(sum(bpp_ed[idx])) if idx < len(bpp_ed) else 0.5)

                    return {
                        "struct_wt": struct_wt,
                        "struct_ed": struct_ed,
                        "mfe_wt": float(mfe_wt),
                        "mfe_ed": float(mfe_ed),
                        "bpp_wt_center": float(sum(bpp_wt[c])) if c < len(bpp_wt) else 0.5,
                        "bpp_ed_center": float(sum(bpp_ed[c])) if c < len(bpp_ed) else 0.5,
                        "bpp_wt_window": bpp_wt_w,
                        "bpp_ed_window": bpp_ed_w,
                    }
                except Exception:
                    return None

            def fold_batch(seqs):
                return [fold_single(s) for s in seqs]

            chunk_size = 500
            chunks = [all_seqs[i:i + chunk_size] for i in range(0, len(all_seqs), chunk_size)]

            fold_results = []
            t0 = time.time()
            with mp.Pool(n_workers) as pool:
                for ci, batch in enumerate(pool.imap(fold_batch, chunks)):
                    fold_results.extend(batch)
                    if (ci + 1) % 20 == 0:
                        done = min((ci + 1) * chunk_size, len(all_seqs))
                        elapsed = time.time() - t0
                        rate = done / elapsed if elapsed > 0 else 0
                        logger.info(f"    Folded {done:,}/{len(all_seqs):,} "
                                    f"({rate:.0f}/sec)")

            # Cache for future use
            logger.info(f"  Caching fold results...")
            cache_data = {
                "cancer": cancer,
                "n_mut": n_mut,
                "n_ctrl": n_ctrl,
                "n_sequences": len(all_seqs),
                "fold_results": fold_results,
            }
            VIENNA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with gzip.open(str(vienna_cache_file), "wt") as fz:
                json.dump(cache_data, fz)

        # Build 40-dim features
        logger.info("  Building 40-dim features...")
        features_40d = build_40dim_features(all_seqs, fold_results)

        # Score with each enzyme model
        cancer_results = {
            "cancer": cancer,
            "n_mutations": n_mut,
            "n_controls": n_ctrl,
        }

        for enzyme_name, model in enzyme_models.items():
            scores = model.predict_proba(features_40d)[:, 1]
            mut_scores = scores[:n_mut]
            ctrl_scores = scores[n_mut:]

            # TC context analysis
            tc_mut = sum(1 for s in mut_valid_seqs if len(s) > 99 and s[99] in "TU")
            tc_ctrl = sum(1 for s in ctrl_valid_seqs if len(s) > 99 and s[99] in "TU")

            # CC context (A3G-specific)
            cc_mut = sum(1 for s in mut_valid_seqs if len(s) > 99 and s[99] == "C")
            cc_ctrl = sum(1 for s in ctrl_valid_seqs if len(s) > 99 and s[99] == "C")

            enrichment = compute_enrichment(mut_scores, ctrl_scores)

            if n_mut > 0 and n_ctrl > 0:
                mw_stat, mw_p = stats.mannwhitneyu(mut_scores, ctrl_scores, alternative="greater")
            else:
                mw_stat, mw_p = 0, 1

            cancer_results[enzyme_name] = {
                "mean_score_mutations": float(np.mean(mut_scores)) if n_mut > 0 else 0,
                "mean_score_controls": float(np.mean(ctrl_scores)) if n_ctrl > 0 else 0,
                "delta": float(np.mean(mut_scores) - np.mean(ctrl_scores)) if n_mut > 0 else 0,
                "mann_whitney_p": float(mw_p),
                "enrichment": enrichment,
                "tc_frac_mutations": tc_mut / max(n_mut, 1),
                "tc_frac_controls": tc_ctrl / max(n_ctrl, 1),
                "cc_frac_mutations": cc_mut / max(n_mut, 1),
                "cc_frac_controls": cc_ctrl / max(n_ctrl, 1),
            }

            logger.info(f"  {enzyme_name}: mut_mean={np.mean(mut_scores):.4f}, "
                         f"ctrl_mean={np.mean(ctrl_scores):.4f}, "
                         f"delta={np.mean(mut_scores) - np.mean(ctrl_scores):.4f}, "
                         f"MWU_p={mw_p:.2e}")
            for t in ["0.5", "0.7"]:
                e = enrichment.get(t, {})
                logger.info(f"    OR@{t}: {e.get('OR', 'N/A'):.3f} (p={e.get('p', 1):.2e})")

        all_results[cancer] = cancer_results

        del fold_results, features_40d
        gc.collect()

    # ---- Step 4: Generate comparison figures ----
    logger.info("=" * 60)
    logger.info("Step 4: Generate comparison figures")

    cancers_done = [c for c in all_cancers if c in all_results]
    enzymes = [e for e in ["A3A", "A3G", "Neither", "Unified"] if e in enzyme_models]

    if len(cancers_done) > 0 and len(enzymes) > 0:
        # Figure 1: Mean score delta (mut - ctrl) per enzyme per cancer
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Panel A: Mean score delta
        ax = axes[0]
        x = np.arange(len(cancers_done))
        width = 0.8 / len(enzymes)
        colors = {"A3A": "#dc2626", "A3G": "#2563eb", "Neither": "#16a34a", "Unified": "#94a3b8"}
        for ei, enz in enumerate(enzymes):
            deltas = []
            for c in cancers_done:
                d = all_results[c].get(enz, {}).get("delta", 0)
                deltas.append(d)
            ax.bar(x + ei * width - 0.4 + width / 2, deltas, width,
                   label=enz, color=colors.get(enz, "#000"), alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([c.upper() for c in cancers_done], rotation=45)
        ax.set_ylabel("Mean score delta (mut - ctrl)")
        ax.set_title("Enzyme-Specific Score Difference")
        ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
        ax.legend(fontsize=8)

        # Panel B: OR at threshold 0.5
        ax = axes[1]
        for ei, enz in enumerate(enzymes):
            ors = []
            for c in cancers_done:
                e = all_results[c].get(enz, {}).get("enrichment", {}).get("0.5", {})
                ors.append(e.get("OR", float("nan")))
            ax.bar(x + ei * width - 0.4 + width / 2, ors, width,
                   label=enz, color=colors.get(enz, "#000"), alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([c.upper() for c in cancers_done], rotation=45)
        ax.set_ylabel("Odds Ratio")
        ax.set_title("Mutation Enrichment (OR @ threshold 0.5)")
        ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
        ax.legend(fontsize=8)

        # Panel C: OR across thresholds for BLCA (APOBEC-dominant)
        ax = axes[2]
        ref_cancer = "blca" if "blca" in all_results else cancers_done[0]
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        for enz in enzymes:
            ors = []
            for t in thresholds:
                e = all_results[ref_cancer].get(enz, {}).get("enrichment", {}).get(str(t), {})
                ors.append(e.get("OR", float("nan")))
            ax.plot(thresholds, ors, marker="o", label=enz,
                    color=colors.get(enz, "#000"), linewidth=2)
        ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel("Editability threshold")
        ax.set_ylabel("Odds Ratio")
        ax.set_title(f"Enrichment Profile ({ref_cancer.upper()})")
        ax.legend(fontsize=8)

        plt.tight_layout()
        fig_path = OUTPUT_DIR / "enzyme_tissue_disease.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Figure saved: {fig_path}")

        # Figure 2: Heatmap of enzyme x cancer OR
        fig, ax = plt.subplots(figsize=(10, 6))
        or_matrix = np.zeros((len(enzymes), len(cancers_done)))
        for ei, enz in enumerate(enzymes):
            for ci, c in enumerate(cancers_done):
                e = all_results[c].get(enz, {}).get("enrichment", {}).get("0.5", {})
                or_matrix[ei, ci] = e.get("OR", float("nan"))

        im = ax.imshow(or_matrix, cmap="RdBu_r", vmin=0.5, vmax=1.5, aspect="auto")
        ax.set_xticks(range(len(cancers_done)))
        ax.set_xticklabels([c.upper() for c in cancers_done], rotation=45)
        ax.set_yticks(range(len(enzymes)))
        ax.set_yticklabels(enzymes)
        for ei in range(len(enzymes)):
            for ci in range(len(cancers_done)):
                val = or_matrix[ei, ci]
                if not np.isnan(val):
                    ax.text(ci, ei, f"{val:.2f}", ha="center", va="center", fontsize=9)
        plt.colorbar(im, label="Odds Ratio @ 0.5")
        ax.set_title("Enzyme x Cancer Enrichment (OR @ threshold 0.5)")
        plt.tight_layout()
        fig_path = OUTPUT_DIR / "enzyme_cancer_heatmap.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Heatmap saved: {fig_path}")

    # ---- Save results ----
    with open(OUTPUT_DIR / "enzyme_tissue_disease_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    elapsed = time.time() - t_start
    logger.info(f"\n{'='*60}")
    logger.info(f"COMPLETE ({elapsed/60:.0f} min)")
    logger.info(f"Results: {OUTPUT_DIR / 'enzyme_tissue_disease_results.json'}")

    # Summary table
    logger.info("\n--- ENZYME x CANCER OR SUMMARY (threshold=0.5) ---")
    header = f"{'Cancer':<8}" + "".join(f"{e:>12}" for e in enzymes)
    logger.info(header)
    for c in cancers_done:
        row = f"{c.upper():<8}"
        for enz in enzymes:
            e = all_results[c].get(enz, {}).get("enrichment", {}).get("0.5", {})
            orval = e.get("OR", float("nan"))
            pval = e.get("p", 1.0)
            sig = "*" if pval < 0.05 else " "
            row += f"{orval:>10.3f}{sig} "
            row = row.rstrip() + " "
        logger.info(row)


def train_enzyme_model_all(splits_df, sequences, loop_df, struct_data):
    """Train a unified model on ALL enzyme data."""
    site_ids = splits_df["site_id"].astype(str).tolist()
    labels = splits_df["is_edited"].values

    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    logger.info(f"  Unified: {n_pos} positives, {n_neg} negatives")

    motif_feats = np.array([extract_motif_from_seq(sequences.get(sid, "N" * 201))
                            for sid in site_ids], dtype=np.float32)

    struct_feats = np.zeros((len(site_ids), 7), dtype=np.float32)
    if struct_data is not None:
        struct_sids = struct_data["site_ids"]
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
                        val = val.iloc[0] if hasattr(val, 'iloc') else 0.0
                    loop_feats[i, j] = float(val) if pd.notna(val) else 0.0

    X = np.concatenate([motif_feats, struct_feats, loop_feats], axis=1)
    X = np.nan_to_num(X, nan=0.0)

    model = XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=SEED, eval_metric="logloss", use_label_encoder=False,
    )
    model.fit(X, labels)

    from sklearn.metrics import roc_auc_score
    train_auc = roc_auc_score(labels, model.predict_proba(X)[:, 1])
    logger.info(f"  Unified training AUC: {train_auc:.4f}")

    return model


if __name__ == "__main__":
    main()
