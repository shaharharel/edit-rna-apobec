#!/usr/bin/env python3
"""Fixed TCGA scoring: Neural model with REAL hand features.

Previous neural TCGA scoring zeroed out the 40-dim hand features (motif+structure+loop),
which the model was trained with. This script fixes that by:
1. Loading ViennaRNA cache (struct features: delta 7-dim + loop 9-dim)
2. Re-extracting sequences from genome (motif features: 24-dim)
3. Combining RNA-FM(640) + delta(640) + hand(40) = 1320-dim for neural scoring

Usage:
    conda run -n quris python experiments/multi_enzyme/exp_neural_tcga_fixed.py
"""

import gc
import gzip
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from pyfaidx import Fasta

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.apobec_feature_extraction import extract_motif_from_seq

# ============================================================================
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "multi_enzyme" / "outputs" / "neural_tcga_fixed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(OUTPUT_DIR / "run.log", mode="w")],
)
logger = logging.getLogger(__name__)

# ============================================================================
DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else torch.device("cpu")
)
SEED = 42
BATCH_SIZE = 512
ENZYMES = ["A3A", "A3B", "A3G", "A3A_A3G", "Neither"]
N_PER_ENZYME = len(ENZYMES)
D_RNAFM = 640; D_EDIT_DELTA = 640; D_HAND = 40; D_INPUT = 1320; D_SHARED = 128; N_ENZYMES = 6

DATA_DIR = PROJECT_ROOT / "data"
EMB_DIR = DATA_DIR / "processed" / "multi_enzyme" / "embeddings"
VIENNA_CACHE_DIR = PROJECT_ROOT / "experiments" / "multi_enzyme" / "outputs" / "tcga_gnomad" / "vienna_cache"
RAW_SCORES_DIR = PROJECT_ROOT / "experiments" / "multi_enzyme" / "outputs" / "tcga_gnomad" / "raw_scores"
TCGA_DIR = DATA_DIR / "raw" / "tcga"
HG19_FA = DATA_DIR / "raw" / "genomes" / "hg19.fa"
REFGENE_HG19 = DATA_DIR / "raw" / "genomes" / "refGene_hg19.txt"
NEURAL_MODEL_PATH = (
    PROJECT_ROOT / "experiments" / "multi_enzyme" / "outputs"
    / "phase3_neural_true_validation" / "phase3_neural_full.pt"
)

CANCERS = ["blca", "brca", "cesc", "lusc", "hnsc", "skcm", "esca", "stad", "lihc"]


# ============================================================================
# Model (same as exp_phase3_neural_true_validation.py)
# ============================================================================
class Phase3Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_encoder = nn.Sequential(
            nn.Linear(D_INPUT, 256), nn.GELU(), nn.Dropout(0.3), nn.LayerNorm(256),
            nn.Linear(256, D_SHARED), nn.GELU(), nn.Dropout(0.2),
        )
        self.binary_head = nn.Linear(D_SHARED, 1)
        self.enzyme_adapters = nn.ModuleDict({
            enz: nn.Sequential(nn.Linear(D_SHARED, 32), nn.GELU(), nn.Linear(32, 1))
            for enz in ENZYMES
        })
        self.enzyme_classifier = nn.Sequential(
            nn.Linear(D_SHARED, 64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, N_ENZYMES),
        )
    def forward(self, x):
        shared = self.shared_encoder(x)
        binary_logit = self.binary_head(shared).squeeze(-1)
        per_enzyme_logits = [self.enzyme_adapters[enz](shared).squeeze(-1) for enz in ENZYMES]
        return binary_logit, per_enzyme_logits, None, shared


# ============================================================================
# Feature helpers (same as exp_phase3_replication.py)
# ============================================================================
def derive_features_from_fold(fold_result):
    """Derive 16-dim structure features (7 delta + 9 loop) from cached fold data."""
    if fold_result is None:
        return None
    center = 100
    pair_wt = fold_result["bpp_wt_center"]
    pair_ed = fold_result["bpp_ed_center"]
    delta_pairing = pair_ed - pair_wt
    delta_mfe = fold_result["mfe_ed"] - fold_result["mfe_wt"]
    bpp_wt_w = fold_result["bpp_wt_window"]
    bpp_ed_w = fold_result["bpp_ed_window"]

    def _entropy(p):
        if p <= 0 or p >= 1: return 0.0
        return -(p * np.log2(p + 1e-10) + (1 - p) * np.log2(1 - p + 1e-10))

    delta_entropy = _entropy(pair_ed) - _entropy(pair_wt)
    dw = np.array(bpp_ed_w) - np.array(bpp_wt_w)
    mean_dw = float(np.mean(dw)) if len(dw) > 0 else 0.0
    std_dw = float(np.std(dw)) if len(dw) > 0 else 0.0

    struct_delta = np.array([delta_pairing, -delta_pairing, delta_entropy,
                             delta_mfe, mean_dw, std_dw, -mean_dw], dtype=np.float32)

    s = fold_result["struct_wt"]
    is_up = 1.0 if s[center] == "." else 0.0
    ls = dj = da = 0.0; rlp = 0.5; lst = rst = mas = lu = 0.0

    if is_up:
        left = center
        while left > 0 and s[left] == ".": left -= 1
        right = center
        while right < len(s) - 1 and s[right] == ".": right += 1
        ls = float(right - left - 1)
        if ls > 0:
            p = center - left - 1
            rlp = p / max(ls - 1, 1)
            da = abs(p - (ls - 1) / 2)
        i = left; c = 0
        while i >= 0 and s[i] in "()": c += 1; i -= 1
        lst = float(c)
        i = right; c = 0
        while i < len(s) and s[i] in "()": c += 1; i += 1
        rst = float(c)
        mas = max(lst, rst)

    reg = s[max(0, center-10):min(len(s), center+11)]
    lu = sum(1 for ch in reg if ch == ".") / max(len(reg), 1)

    loop_features = np.array([is_up, ls, dj, da, rlp, lst, rst, mas, lu], dtype=np.float32)
    return {"struct_delta": struct_delta, "loop_features": loop_features}


def parse_exons(refgene_path):
    """Parse exonic regions from refGene."""
    from collections import defaultdict
    exons = defaultdict(list)
    with open(refgene_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 11: continue
            chrom, strand, gene = parts[2], parts[3], parts[12] if len(parts) > 12 else parts[1]
            starts = [int(x) for x in parts[9].split(",") if x]
            ends = [int(x) for x in parts[10].split(",") if x]
            for s, e in zip(starts, ends):
                exons[gene].append((chrom, s, e, strand, gene))
    return exons


def parse_ct_mutations(maf_path):
    """Parse C>T mutations from TCGA MAF."""
    rows = []
    for chunk in pd.read_csv(maf_path, sep="\t", comment="#", low_memory=False, chunksize=50000):
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
            rows.append(sub[["chrom", "pos", "strand_inf", "gene"]].copy())
    if not rows: return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    return df.drop_duplicates(subset=["chrom", "pos"]).copy()


def get_matched_controls(mutations_df, genome, exons_by_gene, n_controls=5):
    """For each mutation, pick n_controls random C positions in same gene's exons."""
    rng = np.random.RandomState(SEED)
    controls = []
    for _, row in mutations_df.iterrows():
        gene = row["gene"]
        gene_exons = exons_by_gene.get(gene, [])
        if not gene_exons: continue
        c_positions = []
        for chrom, start, end, strand, _ in gene_exons:
            if chrom != row["chrom"]: continue
            try:
                exon_seq = str(genome[chrom][start:end]).upper()
                for i, base in enumerate(exon_seq):
                    pos = start + i
                    if pos == row["pos"]: continue
                    if base == "C": c_positions.append((chrom, pos, "+"))
                    elif base == "G": c_positions.append((chrom, pos, "-"))
            except (KeyError, ValueError): continue
        if len(c_positions) >= n_controls:
            chosen = rng.choice(len(c_positions), n_controls, replace=False)
            for idx in chosen:
                controls.append({"chrom": c_positions[idx][0], "pos": c_positions[idx][1],
                                 "strand_inf": c_positions[idx][2], "gene": gene})
        elif c_positions:
            for ch, p, s in c_positions[:n_controls]:
                controls.append({"chrom": ch, "pos": p, "strand_inf": s, "gene": gene})
    return pd.DataFrame(controls) if controls else pd.DataFrame()


def extract_sequences(positions_df, genome):
    """Extract 201-nt sequences for each position."""
    seqs = []
    for _, row in positions_df.iterrows():
        chrom, pos, strand = row["chrom"], int(row["pos"]), row["strand_inf"]
        try:
            clen = len(genome[chrom])
            start, end = pos - 100, pos + 101
            if start < 0 or end > clen:
                seqs.append(None); continue
            seq = str(genome[chrom][start:end]).upper()
            if strand == "-":
                seq = seq.translate(str.maketrans("ACGT", "TGCA"))[::-1]
            seqs.append(seq)
        except (KeyError, ValueError):
            seqs.append(None)
    return seqs


# ============================================================================
# Enrichment computation
# ============================================================================
def enrichment_stats(scores, mut_mask, ctrl_mask, tc_context=None):
    """Compute enrichment stats."""
    mut_s = scores[mut_mask]; ctrl_s = scores[ctrl_mask]
    result = {
        "mean_mut": float(np.mean(mut_s)), "mean_ctrl": float(np.mean(ctrl_s)),
        "delta": float(np.mean(mut_s) - np.mean(ctrl_s)),
    }
    try:
        _, p = stats.mannwhitneyu(mut_s, ctrl_s, alternative="greater")
        result["mw_p"] = float(p)
    except: result["mw_p"] = 1.0

    if tc_context is not None:
        tc = tc_context == 1
        tc_mut = mut_s[tc[mut_mask]]; tc_ctrl = ctrl_s[tc[ctrl_mask]]
        ntc_mut = mut_s[~tc[mut_mask]]; ntc_ctrl = ctrl_s[~tc[ctrl_mask]]
        for prefix, ms, cs in [("TC", tc_mut, tc_ctrl), ("nonTC", ntc_mut, ntc_ctrl)]:
            result[f"{prefix}_delta"] = float(np.mean(ms) - np.mean(cs)) if len(ms) > 0 and len(cs) > 0 else 0.0
            try:
                _, tp = stats.mannwhitneyu(ms, cs, alternative="greater")
                result[f"{prefix}_p"] = float(tp)
            except: result[f"{prefix}_p"] = 1.0

    all_s = np.concatenate([mut_s, ctrl_s])
    thresholds = {}
    for pct in [50, 75, 90, 95]:
        thresh = np.percentile(all_s, pct)
        ma, mb = int((mut_s >= thresh).sum()), int((mut_s < thresh).sum())
        ca, cb = int((ctrl_s >= thresh).sum()), int((ctrl_s < thresh).sum())
        if all(x > 0 for x in [ma, mb, ca, cb]):
            OR, pv = stats.fisher_exact([[ma, mb], [ca, cb]])
        else: OR, pv = float("nan"), 1.0
        thresholds[f"p{pct}"] = {"threshold": float(thresh), "OR": float(OR), "p": float(pv),
                                  "mut_above": ma, "ctrl_above": ca}
    result["thresholds"] = thresholds
    return result


# ============================================================================
# Main
# ============================================================================
def main():
    t_start = time.time()

    logger.info(f"Device: {DEVICE}")
    logger.info("Loading neural model...")
    model = Phase3Model()
    model.load_state_dict(torch.load(NEURAL_MODEL_PATH, weights_only=True, map_location="cpu"))
    model.to(DEVICE); model.eval()

    logger.info("Loading hg19 genome...")
    genome = Fasta(str(HG19_FA))
    exons = parse_exons(str(REFGENE_HG19))
    logger.info(f"  {len(exons):,} genes")

    all_results = {}

    for cancer in CANCERS:
        tc = time.time()
        logger.info(f"\n{'='*60}\n  {cancer.upper()}\n{'='*60}")

        # Check all inputs exist
        emb_path = EMB_DIR / f"rnafm_tcga_{cancer}.pt"
        vienna_path = VIENNA_CACHE_DIR / f"{cancer}_vienna_raw.json.gz"
        maf_path = TCGA_DIR / f"{cancer}_tcga_pan_can_atlas_2018_mutations.txt"

        for p, name in [(emb_path, "RNA-FM"), (vienna_path, "ViennaRNA"), (maf_path, "MAF")]:
            if not p.exists():
                logger.warning(f"  Missing {name}: {p}")
                continue

        if not all(p.exists() for p in [emb_path, vienna_path, maf_path]):
            continue

        # Load RNA-FM embeddings
        d = torch.load(emb_path, weights_only=False)
        n_mut, n_ctrl = d["n_mut"], d["n_ctrl"]
        pooled_orig = d["pooled_orig"].numpy()
        pooled_edited = d["pooled_edited"].numpy()
        edit_delta = pooled_edited - pooled_orig
        logger.info(f"  RNA-FM: {n_mut} mut, {n_ctrl} ctrl")

        # Load ViennaRNA cache
        logger.info(f"  Loading ViennaRNA cache...")
        with gzip.open(str(vienna_path), "rt") as fz:
            cache = json.load(fz)
        fold_results = cache["fold_results"]
        logger.info(f"  ViennaRNA: {len(fold_results)} folds")

        # Extract sequences (for motif features)
        logger.info(f"  Extracting sequences from genome...")
        mut_df = parse_ct_mutations(str(maf_path))
        logger.info(f"  {len(mut_df)} C>T mutations")

        mut_seqs = extract_sequences(mut_df, genome)
        ctrl_df = get_matched_controls(mut_df, genome, exons, n_controls=5)
        ctrl_seqs = extract_sequences(ctrl_df, genome)

        # Filter valid
        mut_valid = [s for s in mut_seqs if s is not None and len(s) == 201 and s[100] == "C"]
        ctrl_valid = [s for s in ctrl_seqs if s is not None and len(s) == 201 and s[100] == "C"]
        logger.info(f"  Valid sequences: {len(mut_valid)} mut, {len(ctrl_valid)} ctrl")

        # Align to ViennaRNA cache size
        if len(mut_valid) > cache["n_mut"]:
            mut_valid = mut_valid[:cache["n_mut"]]
        if len(ctrl_valid) > cache["n_ctrl"]:
            ctrl_valid = ctrl_valid[:cache["n_ctrl"]]

        all_seqs = mut_valid + ctrl_valid
        n_total = min(len(all_seqs), len(fold_results), n_mut + n_ctrl)

        # Build 40-dim hand features
        logger.info(f"  Building 40-dim hand features (n={n_total})...")
        hand_features = np.zeros((n_total, 40), dtype=np.float32)
        for i in range(n_total):
            seq = all_seqs[i] if i < len(all_seqs) else None
            fr = fold_results[i] if i < len(fold_results) else None
            if seq is not None and len(seq) == 201:
                hand_features[i, :24] = extract_motif_from_seq(seq)
            derived = derive_features_from_fold(fr)
            if derived is not None:
                hand_features[i, 24:31] = derived["struct_delta"]
                hand_features[i, 31:40] = derived["loop_features"]
        hand_features = np.nan_to_num(hand_features, nan=0.0)

        logger.info(f"  Hand features: motif nonzero={float((hand_features[:, :24] != 0).mean()):.3f}, "
                     f"struct nonzero={float((hand_features[:, 24:31] != 0).mean()):.3f}, "
                     f"loop nonzero={float((hand_features[:, 31:40] != 0).mean()):.3f}")

        # Build 1320-dim features: FIXED (with real hand features)
        X_fixed = np.concatenate([
            pooled_orig[:n_total], edit_delta[:n_total], hand_features
        ], axis=1).astype(np.float32)

        # Also build ZEROS version for comparison
        X_zeros = np.concatenate([
            pooled_orig[:n_total], edit_delta[:n_total],
            np.zeros((n_total, 40), dtype=np.float32)
        ], axis=1).astype(np.float32)

        # Score both versions
        def score_neural(X):
            all_b, all_e = [], [[] for _ in range(N_PER_ENZYME)]
            with torch.no_grad():
                for start in range(0, len(X), BATCH_SIZE):
                    end = min(start + BATCH_SIZE, len(X))
                    x = torch.from_numpy(X[start:end]).to(DEVICE)
                    bl, pel, _, _ = model(x)
                    all_b.append(torch.sigmoid(bl).cpu().numpy())
                    for h in range(N_PER_ENZYME):
                        all_e[h].append(torch.sigmoid(pel[h]).cpu().numpy())
            return np.concatenate(all_b), [np.concatenate(p) for p in all_e]

        logger.info(f"  Scoring (fixed)...")
        binary_fixed, enzyme_fixed = score_neural(X_fixed)
        logger.info(f"  Scoring (zeros)...")
        binary_zeros, enzyme_zeros = score_neural(X_zeros)

        # Load tc_context
        scores_path = RAW_SCORES_DIR / f"{cancer}_scores.csv"
        if scores_path.exists():
            raw_df = pd.read_csv(scores_path)
            types = raw_df["type"].values[:n_total]
            tc_ctx = raw_df["tc_context"].values[:n_total]
        else:
            types = np.array(["mutation"] * min(n_mut, n_total) + ["control"] * max(0, n_total - n_mut))
            tc_ctx = None

        mut_mask = types == "mutation"
        ctrl_mask = types == "control"

        # Compute enrichment for both
        r_fixed = {
            "cancer": cancer, "n_mutations": int(mut_mask.sum()), "n_controls": int(ctrl_mask.sum()),
            "binary_head": enrichment_stats(binary_fixed, mut_mask, ctrl_mask, tc_ctx),
        }
        r_zeros = {
            "cancer": cancer, "n_mutations": int(mut_mask.sum()), "n_controls": int(ctrl_mask.sum()),
            "binary_head": enrichment_stats(binary_zeros, mut_mask, ctrl_mask, tc_ctx),
        }
        for h, enz in enumerate(ENZYMES):
            r_fixed[f"adapter_{enz}"] = enrichment_stats(enzyme_fixed[h], mut_mask, ctrl_mask, tc_ctx)
            r_zeros[f"adapter_{enz}"] = enrichment_stats(enzyme_zeros[h], mut_mask, ctrl_mask, tc_ctx)

        all_results[cancer] = {"fixed": r_fixed, "zeros": r_zeros}

        # Log comparison
        bf = r_fixed["binary_head"]; bz = r_zeros["binary_head"]
        logger.info(f"\n  {'Head':<12} {'Fixed OR@p90':>14} {'Zeros OR@p90':>14} {'Diff':>8}")
        logger.info(f"  {'-'*52}")
        logger.info(f"  {'Binary':<12} {bf['thresholds']['p90']['OR']:>14.3f} {bz['thresholds']['p90']['OR']:>14.3f} "
                     f"{bf['thresholds']['p90']['OR'] - bz['thresholds']['p90']['OR']:>+8.3f}")
        for enz in ENZYMES:
            ef = r_fixed[f"adapter_{enz}"]["thresholds"]["p90"]["OR"]
            ez = r_zeros[f"adapter_{enz}"]["thresholds"]["p90"]["OR"]
            logger.info(f"  {enz:<12} {ef:>14.3f} {ez:>14.3f} {ef - ez:>+8.3f}")

        logger.info(f"  Elapsed: {time.time() - tc:.0f}s")

        del d, pooled_orig, pooled_edited, edit_delta, cache, fold_results
        gc.collect()

    # Save
    with open(OUTPUT_DIR / "tcga_fixed_vs_zeros.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary
    logger.info(f"\n{'='*80}")
    logger.info("SUMMARY: Fixed (with hand features) vs Zeros (original bug)")
    logger.info(f"{'='*80}")
    logger.info(f"{'Cancer':>8} | {'Fixed Bin':>10} {'p':>10} | {'Zeros Bin':>10} {'p':>10} | {'Fixed Neither':>14} | {'Zeros Neither':>14}")
    logger.info("-" * 90)
    for cancer in CANCERS:
        if cancer not in all_results: continue
        rf = all_results[cancer]["fixed"]; rz = all_results[cancer]["zeros"]
        fb = rf["binary_head"]["thresholds"]["p90"]
        zb = rz["binary_head"]["thresholds"]["p90"]
        fn = rf["adapter_Neither"]["thresholds"]["p90"]
        zn = rz["adapter_Neither"]["thresholds"]["p90"]
        logger.info(f"{cancer:>8} | {fb['OR']:>10.3f} {fb['p']:>10.2e} | {zb['OR']:>10.3f} {zb['p']:>10.2e} "
                     f"| {fn['OR']:>14.3f} | {zn['OR']:>14.3f}")

    logger.info(f"\nTotal time: {(time.time() - t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
