#!/usr/bin/env python3
"""A4: Full Exome Editability Map with ViennaRNA (FULL MODEL).

Scores ALL ~8.5M exonic C positions with ViennaRNA + full 40-dim GB model.
Processes one chromosome at a time to manage memory. Saves ViennaRNA structure
caches per chromosome.

This is a ~24h compute job. Start with chr22 (smallest) as test.

Usage:
    # Test on chr22 only:
    conda run -n quris python experiments/multi_enzyme/exp_exome_editability_full.py --chr22-only

    # Full run (all chromosomes):
    conda run -n quris python experiments/multi_enzyme/exp_exome_editability_full.py

    # Resume from specific chromosome:
    conda run -n quris python experiments/multi_enzyme/exp_exome_editability_full.py --start-chr chr5
"""

import argparse
import gc
import gzip
import json
import logging
import multiprocessing as mp
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

OUTPUT_DIR = PROJECT_ROOT / "experiments/multi_enzyme/outputs/exome_map"
VIENNA_CACHE_DIR = OUTPUT_DIR / "vienna_cache"

N_WORKERS = max(1, mp.cpu_count() - 2)
SEED = 42
CHUNK_SIZE = 500  # sequences per ViennaRNA worker chunk

ALL_CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]


# ============================================================================
# ViennaRNA folding
# ============================================================================

def fold_single_sequence(seq_201nt):
    """Fold a single 201-nt sequence. Returns raw fold data or None."""
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
    """Fold a batch of sequences."""
    return [fold_single_sequence(s) for s in sequences]


def parallel_fold(sequences, n_workers=N_WORKERS, chunk_size=CHUNK_SIZE):
    """Fold sequences in parallel."""
    n = len(sequences)
    logger.info(f"  Folding {n:,} sequences with {n_workers} workers (chunk={chunk_size})...")

    chunks = [sequences[i:i + chunk_size] for i in range(0, n, chunk_size)]

    results = []
    t0 = time.time()
    with mp.Pool(n_workers) as pool:
        for i, batch_result in enumerate(pool.imap(fold_batch, chunks)):
            results.extend(batch_result)
            if (i + 1) % 50 == 0:
                done = min((i + 1) * chunk_size, n)
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (n - done) / rate if rate > 0 else 0
                logger.info(f"    Folded {done:,}/{n:,} ({100*done/n:.0f}%) "
                            f"[{rate:.0f}/sec, ETA {eta/60:.0f} min]")

    elapsed = time.time() - t0
    n_ok = sum(1 for r in results if r is not None)
    logger.info(f"    Done: {n_ok:,}/{n:,} successful ({elapsed/60:.1f} min)")
    return results


# ============================================================================
# Feature building from fold results
# ============================================================================

def derive_features_from_fold(fold_result):
    """Derive 16-dim structure features from cached fold data."""
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

    struct_wt = fold_result["struct_wt"]
    is_unpaired = 1.0 if struct_wt[center] == "." else 0.0

    loop_size = 0.0
    dist_to_junction = 0.0
    dist_to_apex = 0.0
    relative_loop_position = 0.5
    left_stem = 0.0
    right_stem = 0.0
    max_adj_stem = 0.0

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


def build_features_and_score(sequences, fold_results, model):
    """Build 40-dim features and score with model. Returns scores array."""
    n = len(sequences)
    features = np.zeros((n, 40), dtype=np.float32)

    for i, (seq, fold_raw) in enumerate(zip(sequences, fold_results)):
        if seq is not None and len(seq) == 201:
            features[i, :24] = extract_motif_from_seq(seq)
        derived = derive_features_from_fold(fold_raw)
        if derived is not None:
            features[i, 24:31] = derived["struct_delta"]
            features[i, 31:40] = derived["loop_features"]

    features = np.nan_to_num(features, nan=0.0)

    # Also compute motif-only score for comparison
    motif_features = features[:, :24].copy()
    # Pad to 40 dims for motif-only scoring (zeros for structure)
    motif_padded = np.zeros((n, 40), dtype=np.float32)
    motif_padded[:, :24] = motif_features

    scores_full = model.predict_proba(features)[:, 1]
    scores_motif = model.predict_proba(motif_padded)[:, 1]

    return scores_full, scores_motif


# ============================================================================
# Train model
# ============================================================================

def train_full_model():
    """Train GB_HandFeatures (40-dim) on multi-enzyme v3 data."""
    logger.info("Training full 40-dim GB model...")

    splits = pd.read_csv(MULTI_SPLITS)
    with open(MULTI_SEQS) as f:
        sequences = json.load(f)

    site_ids = splits["site_id"].astype(str).tolist()
    labels = splits["is_edited"].values

    motif_feats = np.array([extract_motif_from_seq(sequences.get(sid, "N" * 201))
                            for sid in site_ids], dtype=np.float32)

    loop_df = pd.read_csv(LOOP_CSV)
    loop_df["site_id"] = loop_df["site_id"].astype(str)
    loop_df = loop_df.set_index("site_id")
    loop_feats = np.zeros((len(site_ids), len(LOOP_FEATURE_COLS)), dtype=np.float32)
    for i, sid in enumerate(site_ids):
        if sid in loop_df.index:
            for j, col in enumerate(LOOP_FEATURE_COLS):
                if col in loop_df.columns:
                    val = loop_df.loc[sid, col]
                    if not isinstance(val, (int, float)):
                        val = val.iloc[0] if hasattr(val, 'iloc') else 0.0
                    loop_feats[i, j] = float(val) if pd.notna(val) else 0.0

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
# Parse exonic regions
# ============================================================================

def parse_refgene_exons():
    """Parse refGene_hg19.txt for coding exonic regions, grouped by chromosome."""
    cols = [
        "bin", "name", "chrom", "strand", "txStart", "txEnd",
        "cdsStart", "cdsEnd", "exonCount", "exonStarts", "exonEnds",
        "score", "name2", "cdsStartStat", "cdsEndStat", "exonFrames",
    ]
    df = pd.read_csv(REFGENE_HG19, sep="\t", header=None, names=cols, low_memory=False)

    valid_chroms = set(ALL_CHROMS)
    df = df[df["chrom"].isin(valid_chroms)]
    df = df[df["cdsStart"] != df["cdsEnd"]]

    # Keep longest transcript per gene
    df["tx_len"] = df["txEnd"] - df["txStart"]
    df = df.sort_values("tx_len", ascending=False).drop_duplicates(subset="name2", keep="first")

    exons_by_chrom = defaultdict(list)
    for _, row in df.iterrows():
        chrom = row["chrom"]
        strand = row["strand"]
        gene = row["name2"]
        cds_start = row["cdsStart"]
        cds_end = row["cdsEnd"]

        starts = [int(x) for x in row["exonStarts"].rstrip(",").split(",")]
        ends = [int(x) for x in row["exonEnds"].rstrip(",").split(",")]

        for es, ee in zip(starts, ends):
            cs = max(es, cds_start)
            ce = min(ee, cds_end)
            if cs < ce:
                exons_by_chrom[chrom].append((cs, ce, strand, gene))

    return exons_by_chrom


# ============================================================================
# Process one chromosome
# ============================================================================

def process_chromosome(chrom, exons, genome, model):
    """Process all exonic C positions in one chromosome.

    Returns DataFrame with scored positions.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing {chrom}: {len(exons)} exonic intervals")

    chrom_len = len(genome[chrom])
    chrom_seq = str(genome[chrom])

    # Step 1: Find all C positions in coding exons
    positions = []  # (pos, strand, gene, trinuc)
    for exon_start, exon_end, strand, gene in exons:
        for pos in range(exon_start, exon_end):
            base = chrom_seq[pos].upper()

            if strand == "+" and base == "C":
                if pos >= 100 and pos + 101 <= chrom_len:
                    trinuc = chrom_seq[max(0, pos - 1):pos + 2].upper()
                    positions.append((pos, "+", gene, trinuc))

            elif strand == "-" and base == "G":
                if pos >= 100 and pos + 101 <= chrom_len:
                    comp = str.maketrans("ACGT", "TGCA")
                    trinuc = chrom_seq[max(0, pos - 1):pos + 2].upper().translate(comp)[::-1]
                    positions.append((pos, "-", gene, trinuc))

    logger.info(f"  Found {len(positions):,} C positions")

    if len(positions) == 0:
        return pd.DataFrame()

    # Step 2: Check for cached ViennaRNA structures
    cache_file = VIENNA_CACHE_DIR / f"{chrom}_vienna.json.gz"
    fold_results = None

    if cache_file.exists():
        logger.info(f"  Loading cached ViennaRNA structures from {cache_file}")
        try:
            with gzip.open(str(cache_file), "rt") as fz:
                cache_data = json.load(fz)
            if cache_data.get("n_positions") == len(positions):
                fold_results = cache_data["fold_results"]
                logger.info(f"  Cache valid: {len(fold_results):,} entries")
            else:
                logger.warning(f"  Cache size mismatch ({cache_data.get('n_positions')} vs {len(positions)}), "
                                f"recomputing")
        except Exception as e:
            logger.warning(f"  Cache load failed: {e}, recomputing")

    # Step 3: Extract sequences
    logger.info("  Extracting 201-nt sequences...")
    sequences = []
    comp_table = str.maketrans("ACGT", "TGCA")
    for pos, strand, gene, trinuc in positions:
        window = chrom_seq[pos - 100:pos + 101].upper()
        if strand == "-":
            window = window.translate(comp_table)[::-1]
        sequences.append(window)

    # Step 4: Fold with ViennaRNA if not cached
    if fold_results is None:
        logger.info(f"  Folding {len(sequences):,} sequences with ViennaRNA ({N_WORKERS} workers)...")
        fold_results = parallel_fold(sequences, N_WORKERS, CHUNK_SIZE)

        # Cache
        logger.info(f"  Caching ViennaRNA structures to {cache_file}...")
        VIENNA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_data = {
            "chrom": chrom,
            "n_positions": len(positions),
            "fold_results": fold_results,
        }
        with gzip.open(str(cache_file), "wt") as fz:
            json.dump(cache_data, fz)
        cache_size_mb = cache_file.stat().st_size / 1024 / 1024
        logger.info(f"  Cache saved: {cache_size_mb:.0f} MB")

    # Step 5: Build features and score
    logger.info("  Building features and scoring...")
    scores_full, scores_motif = build_features_and_score(sequences, fold_results, model)

    # Build result DataFrame
    rows = []
    for i, (pos, strand, gene, trinuc) in enumerate(positions):
        rows.append({
            "chr": chrom,
            "pos": pos,
            "strand": strand,
            "score_full": float(scores_full[i]),
            "score_motif": float(scores_motif[i]),
            "trinuc": trinuc,
            "gene": gene,
        })

    df = pd.DataFrame(rows)

    logger.info(f"  {chrom} complete: {len(df):,} positions scored")
    logger.info(f"    Full model: mean={scores_full.mean():.4f}, "
                 f"median={np.median(scores_full):.4f}, max={scores_full.max():.4f}")
    logger.info(f"    Motif only: mean={scores_motif.mean():.4f}")

    # Free memory
    del fold_results, sequences, scores_full, scores_motif
    gc.collect()

    return df


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chr22-only", action="store_true",
                        help="Only process chr22 (test mode)")
    parser.add_argument("--start-chr", type=str, default=None,
                        help="Resume from this chromosome (e.g., chr5)")
    args = parser.parse_args()

    t_start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VIENNA_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Train model
    logger.info("=" * 60)
    logger.info("Step 1: Train full 40-dim GB model")
    model = train_full_model()

    # Step 2: Parse exonic regions
    logger.info("=" * 60)
    logger.info("Step 2: Parse refGene_hg19 for coding exons")
    exons_by_chrom = parse_refgene_exons()
    total_exons = sum(len(v) for v in exons_by_chrom.values())
    logger.info(f"  {total_exons:,} exonic intervals across {len(exons_by_chrom)} chromosomes")

    # Step 3: Load genome
    logger.info("=" * 60)
    logger.info("Step 3: Load hg19 genome")
    genome = Fasta(str(HG19_FA))

    # Step 4: Process chromosomes
    logger.info("=" * 60)
    logger.info("Step 4: Process chromosomes")

    if args.chr22_only:
        chroms_to_process = ["chr22"]
    else:
        # Order: smallest first for faster initial feedback
        chrom_sizes = {c: sum(e[1] - e[0] for e in exons_by_chrom.get(c, []))
                       for c in ALL_CHROMS}
        chroms_to_process = sorted(ALL_CHROMS, key=lambda c: chrom_sizes.get(c, 0))

    if args.start_chr:
        try:
            start_idx = chroms_to_process.index(args.start_chr)
            chroms_to_process = chroms_to_process[start_idx:]
            logger.info(f"  Resuming from {args.start_chr}")
        except ValueError:
            logger.warning(f"  {args.start_chr} not found in chromosome list")

    all_dfs = []
    total_positions = 0

    for chrom in chroms_to_process:
        if chrom not in exons_by_chrom:
            logger.info(f"  Skipping {chrom}: no exonic regions")
            continue

        chrom_df = process_chromosome(chrom, exons_by_chrom[chrom], genome, model)
        if len(chrom_df) > 0:
            # Save per-chromosome results incrementally
            chrom_csv = OUTPUT_DIR / f"exome_editability_{chrom}.csv.gz"
            chrom_df.to_csv(chrom_csv, index=False, compression="gzip")
            logger.info(f"  Saved: {chrom_csv} ({chrom_csv.stat().st_size / 1024 / 1024:.1f} MB)")

            all_dfs.append(chrom_df)
            total_positions += len(chrom_df)

        elapsed = time.time() - t_start
        logger.info(f"  Running total: {total_positions:,} positions ({elapsed/3600:.1f} hours elapsed)")

        gc.collect()

    # Step 5: Merge all chromosomes
    if all_dfs:
        logger.info("=" * 60)
        logger.info("Step 5: Merge all chromosomes")
        full_df = pd.concat(all_dfs, ignore_index=True)

        out_csv = OUTPUT_DIR / "exome_editability_full_model.csv.gz"
        full_df.to_csv(out_csv, index=False, compression="gzip")
        logger.info(f"Saved: {out_csv} ({out_csv.stat().st_size / 1024 / 1024:.1f} MB)")

        # Summary statistics
        logger.info(f"\n{'='*60}")
        logger.info("SUMMARY")
        logger.info(f"  Total positions: {len(full_df):,}")
        logger.info(f"  Genes: {full_df['gene'].nunique():,}")
        logger.info(f"  Full model: mean={full_df['score_full'].mean():.4f}, "
                     f"median={full_df['score_full'].median():.4f}")
        logger.info(f"  Motif only: mean={full_df['score_motif'].mean():.4f}, "
                     f"median={full_df['score_motif'].median():.4f}")

        # Score distribution
        for t in [0.3, 0.5, 0.7, 0.8, 0.9]:
            n_above = (full_df["score_full"] >= t).sum()
            logger.info(f"  score_full >= {t}: {n_above:,} ({100*n_above/len(full_df):.1f}%)")

        # Trinucleotide breakdown
        logger.info("\n  Trinucleotide mean scores:")
        trinuc_stats = full_df.groupby("trinuc").agg(
            n=("score_full", "size"),
            mean_full=("score_full", "mean"),
            mean_motif=("score_motif", "mean"),
        ).sort_values("mean_full", ascending=False)
        for trinuc, row in trinuc_stats.head(10).iterrows():
            logger.info(f"    {trinuc}: n={int(row['n']):,}, full={row['mean_full']:.4f}, "
                         f"motif={row['mean_motif']:.4f}")

        # Generate summary figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # Panel A: Score distribution
        ax = axes[0, 0]
        ax.hist(full_df["score_full"], bins=100, alpha=0.7, color="steelblue",
                label="Full model (40-dim)", density=True)
        ax.hist(full_df["score_motif"], bins=100, alpha=0.5, color="firebrick",
                label="Motif only (24-dim)", density=True)
        ax.set_xlabel("Editability score")
        ax.set_ylabel("Density")
        ax.set_title(f"Exome Editability Distribution (n={len(full_df):,})")
        ax.legend()

        # Panel B: Per-chromosome position counts
        ax = axes[0, 1]
        chrom_counts = full_df["chr"].value_counts()
        chrom_order = [c for c in ALL_CHROMS if c in chrom_counts.index]
        counts = [chrom_counts[c] for c in chrom_order]
        ax.barh(range(len(chrom_order)), counts, color="steelblue", alpha=0.8)
        ax.set_yticks(range(len(chrom_order)))
        ax.set_yticklabels(chrom_order, fontsize=7)
        ax.set_xlabel("Number of exonic C positions")
        ax.set_title("Positions per Chromosome")
        ax.invert_yaxis()

        # Panel C: Full vs Motif score scatter (subsample)
        ax = axes[1, 0]
        subsample = full_df.sample(min(50000, len(full_df)), random_state=42)
        ax.scatter(subsample["score_motif"], subsample["score_full"],
                   alpha=0.05, s=1, color="steelblue")
        ax.plot([0, 1], [0, 1], "r--", alpha=0.5)
        rho, _ = stats.spearmanr(subsample["score_motif"], subsample["score_full"])
        ax.set_xlabel("Motif-only score")
        ax.set_ylabel("Full model score")
        ax.set_title(f"Full vs Motif Score (rho={rho:.3f})")

        # Panel D: Gene-level mean score distribution
        ax = axes[1, 1]
        gene_means = full_df.groupby("gene")["score_full"].mean()
        ax.hist(gene_means, bins=100, color="steelblue", alpha=0.7, edgecolor="none")
        ax.set_xlabel("Mean editability per gene")
        ax.set_ylabel("Number of genes")
        ax.set_title(f"Gene-Level Distribution (n={len(gene_means):,} genes)")
        ax.axvline(gene_means.mean(), color="red", linestyle="--",
                   label=f"Mean={gene_means.mean():.3f}")
        ax.legend()

        plt.tight_layout()
        fig_path = OUTPUT_DIR / "exome_editability_full_model.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Figure: {fig_path}")

    elapsed = time.time() - t_start
    logger.info(f"\n{'='*60}")
    logger.info(f"COMPLETE ({elapsed/3600:.1f} hours)")
    logger.info(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
