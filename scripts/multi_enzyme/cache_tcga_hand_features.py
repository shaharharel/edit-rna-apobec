#!/usr/bin/env python3
"""Cache 40-dim hand features for all TCGA cancers.

Uses existing ViennaRNA cache (structure features) + genome (motif features).
Saves features as .npy files for fast reuse by neural model scoring.

The control positions are regenerated deterministically (seed=42) using the same
precompute_gene_c_positions approach as compute_rnafm_tcga_clinvar.py.

Usage:
    conda run -n quris python scripts/multi_enzyme/cache_tcga_hand_features.py
"""

import gzip
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from pyfaidx import Fasta

sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from data.apobec_feature_extraction import extract_motif_from_seq

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = PROJECT_ROOT / "data"
HG19_FA = DATA_DIR / "raw" / "genomes" / "hg19.fa"
REFGENE_HG19 = DATA_DIR / "raw" / "genomes" / "refGene_hg19.txt"
TCGA_DIR = DATA_DIR / "raw" / "tcga"
VIENNA_CACHE_DIR = PROJECT_ROOT / "experiments" / "multi_enzyme" / "outputs" / "tcga_gnomad" / "vienna_cache"
OUTPUT_DIR = DATA_DIR / "processed" / "multi_enzyme" / "tcga_hand_features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_CONTROLS = 5
CENTER = 100

CANCER_STUDIES = {
    "blca": "blca_tcga_pan_can_atlas_2018",
    "brca": "brca_tcga_pan_can_atlas_2018",
    "cesc": "cesc_tcga_pan_can_atlas_2018",
    "lusc": "lusc_tcga_pan_can_atlas_2018",
    "hnsc": "hnsc_tcga_pan_can_atlas_2018",
    "skcm": "skcm_tcga_pan_can_atlas_2018",
    "esca": "esca_tcga_pan_can_atlas_2018",
    "stad": "stad_tcga_pan_can_atlas_2018",
    "lihc": "lihc_tcga_pan_can_atlas_2018",
}


def derive_features_from_fold(fold_result):
    """7-dim struct delta + 9-dim loop = 16-dim from ViennaRNA fold."""
    if fold_result is None:
        return None
    center = 100
    pw, pe = fold_result["bpp_wt_center"], fold_result["bpp_ed_center"]
    dp = pe - pw
    dm = fold_result["mfe_ed"] - fold_result["mfe_wt"]
    dw = np.array(fold_result["bpp_ed_window"]) - np.array(fold_result["bpp_wt_window"])

    def ent(p):
        if p <= 0 or p >= 1:
            return 0.0
        return -(p * np.log2(p + 1e-10) + (1 - p) * np.log2(1 - p + 1e-10))

    de = ent(pe) - ent(pw)
    md = float(np.mean(dw)) if len(dw) > 0 else 0.0
    sd = float(np.std(dw)) if len(dw) > 0 else 0.0
    struct_delta = np.array([dp, -dp, de, dm, md, sd, -md], dtype=np.float32)

    s = fold_result["struct_wt"]
    up = 1.0 if s[center] == "." else 0.0
    ls = dj = da = 0.0
    rlp = 0.5
    lst = rst = mas = lu = 0.0
    if up:
        l = center
        while l > 0 and s[l] == ".":
            l -= 1
        r = center
        while r < len(s) - 1 and s[r] == ".":
            r += 1
        ls = float(r - l - 1)
        if ls > 0:
            p = center - l - 1
            rlp = p / max(ls - 1, 1)
            da = abs(p - (ls - 1) / 2)
        i = l
        c = 0
        while i >= 0 and s[i] in "()":
            c += 1
            i -= 1
        lst = float(c)
        i = r
        c = 0
        while i < len(s) and s[i] in "()":
            c += 1
            i += 1
        rst = float(c)
        mas = max(lst, rst)
    reg = s[max(0, center - 10):min(len(s), center + 11)]
    lu = sum(1 for ch in reg if ch == ".") / max(len(reg), 1)
    loop_features = np.array([up, ls, dj, da, rlp, lst, rst, mas, lu], dtype=np.float32)
    return struct_delta, loop_features


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
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows).drop_duplicates(subset=["chrom", "pos"])


def precompute_gene_c_positions(exons_by_gene, genome):
    """Pre-compute all C/G positions in exonic regions for fast control generation."""
    logger.info("Pre-computing exonic C/G positions...")
    t0 = time.time()
    gene_chrom_positions = {}
    for gene, gene_exons in exons_by_gene.items():
        by_chrom = defaultdict(list)
        for chrom, start, end, strand in gene_exons:
            by_chrom[chrom].append((start, end))
        for chrom, intervals in by_chrom.items():
            c_positions = []
            for start, end in intervals:
                try:
                    exon_seq = str(genome[chrom][start:end]).upper()
                    for i, base in enumerate(exon_seq):
                        pos = start + i
                        if base == "C":
                            c_positions.append((chrom, pos, "+"))
                        elif base == "G":
                            c_positions.append((chrom, pos, "-"))
                except (KeyError, ValueError):
                    continue
            if c_positions:
                gene_chrom_positions[(gene, chrom)] = c_positions
    logger.info(f"  {len(gene_chrom_positions)} (gene,chrom) pairs in {time.time()-t0:.0f}s")
    return gene_chrom_positions


def get_matched_controls(mutations_df, gene_chrom_positions, n_controls=5):
    """Deterministic control generation (seed=42), same as compute_rnafm_tcga_clinvar.py."""
    rng = np.random.RandomState(SEED)
    controls = []
    for _, row in mutations_df.iterrows():
        gene, chrom, mut_pos = row["gene"], row["chrom"], row["pos"]
        c_positions = gene_chrom_positions.get((gene, chrom), [])
        if not c_positions:
            continue
        filtered = [(ch, p, s) for ch, p, s in c_positions if p != mut_pos]
        if len(filtered) >= n_controls:
            chosen = rng.choice(len(filtered), n_controls, replace=False)
            for idx in chosen:
                controls.append({"chrom": filtered[idx][0], "pos": filtered[idx][1],
                                 "strand_inf": filtered[idx][2], "gene": gene})
        elif filtered:
            for ch, p, s in filtered[:n_controls]:
                controls.append({"chrom": ch, "pos": p, "strand_inf": s, "gene": gene})
    return pd.DataFrame(controls) if controls else pd.DataFrame()


def extract_sequences(positions_df, genome):
    """Extract 201-nt sequences."""
    seqs = []
    for _, row in positions_df.iterrows():
        chrom, pos, strand = row["chrom"], int(row["pos"]), row["strand_inf"]
        try:
            start, end = pos - 100, pos + 101
            if start < 0 or end > len(genome[chrom]):
                seqs.append(None)
                continue
            seq = str(genome[chrom][start:end]).upper()
            if strand == "-":
                seq = seq.translate(str.maketrans("ACGT", "TGCA"))[::-1]
            seqs.append(seq)
        except (KeyError, ValueError):
            seqs.append(None)
    return seqs


def parse_exons(refgene_path):
    """Parse exons from refGene."""
    exons = defaultdict(list)
    with open(refgene_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 11:
                continue
            chrom = parts[2]
            strand = parts[3]
            gene = parts[12] if len(parts) > 12 else parts[1]
            starts = [int(x) for x in parts[9].split(",") if x]
            ends = [int(x) for x in parts[10].split(",") if x]
            for s, e in zip(starts, ends):
                exons[gene].append((chrom, s, e, strand))
    return exons


def main():
    t_start = time.time()

    logger.info("Loading genome...")
    genome = Fasta(str(HG19_FA))

    logger.info("Parsing exons...")
    exons = parse_exons(str(REFGENE_HG19))

    logger.info("Pre-computing gene C/G positions...")
    gene_c_pos = precompute_gene_c_positions(exons, genome)

    for cancer, study in CANCER_STUDIES.items():
        output_path = OUTPUT_DIR / f"{cancer}_hand40.npy"
        if output_path.exists():
            logger.info(f"SKIP {cancer}: {output_path} exists")
            continue

        tc = time.time()
        logger.info(f"\n{'='*50}")
        logger.info(f"  {cancer.upper()}")

        maf_path = TCGA_DIR / f"{study}_mutations.txt"
        vienna_path = VIENNA_CACHE_DIR / f"{cancer}_vienna_raw.json.gz"

        if not maf_path.exists():
            logger.warning(f"  No MAF: {maf_path}")
            continue
        if not vienna_path.exists():
            logger.warning(f"  No ViennaRNA: {vienna_path}")
            continue

        # Parse mutations
        mut_df = parse_ct_mutations(str(maf_path))
        logger.info(f"  {len(mut_df)} mutations")

        # Get controls (deterministic, same as RNA-FM computation)
        ctrl_df = get_matched_controls(mut_df, gene_c_pos, N_CONTROLS)
        logger.info(f"  {len(ctrl_df)} controls")

        # Extract sequences
        mut_seqs = extract_sequences(mut_df, genome)
        ctrl_seqs = extract_sequences(ctrl_df, genome)

        # Filter valid
        mut_valid = [s for s in mut_seqs if s is not None and len(s) == 201 and s[CENTER] == "C"]
        ctrl_valid = [s for s in ctrl_seqs if s is not None and len(s) == 201 and s[CENTER] == "C"]
        logger.info(f"  Valid: {len(mut_valid)} mut, {len(ctrl_valid)} ctrl")

        all_seqs = mut_valid + ctrl_valid

        # Load ViennaRNA cache
        logger.info(f"  Loading ViennaRNA cache...")
        with gzip.open(str(vienna_path), "rt") as f:
            cache = json.load(f)
        fold_results = cache["fold_results"]
        logger.info(f"  {len(fold_results)} folds (cache: {cache['n_mut']} mut + {cache['n_ctrl']} ctrl)")

        # Align
        n = min(len(all_seqs), len(fold_results))
        if len(all_seqs) != len(fold_results):
            logger.warning(f"  SIZE MISMATCH: seqs={len(all_seqs)}, folds={len(fold_results)}")
            # Pad sequences if shorter
            while len(all_seqs) < len(fold_results):
                all_seqs.append(None)

        # Build 40-dim features
        logger.info(f"  Building 40-dim features (n={n})...")
        features = np.zeros((n, 40), dtype=np.float32)
        for i in range(n):
            seq = all_seqs[i] if i < len(all_seqs) else None
            if seq is not None and len(seq) == 201:
                features[i, :24] = extract_motif_from_seq(seq)
            fr = fold_results[i]
            res = derive_features_from_fold(fr)
            if res is not None:
                features[i, 24:31] = res[0]
                features[i, 31:40] = res[1]
        features = np.nan_to_num(features, nan=0.0)

        # Save
        np.save(str(output_path), features)
        logger.info(f"  Saved {features.shape} to {output_path}")
        logger.info(f"  Nonzero: motif={float((features[:,:24]!=0).mean()):.3f}, "
                     f"struct={float((features[:,24:31]!=0).mean()):.3f}, "
                     f"loop={float((features[:,31:40]!=0).mean()):.3f}")
        logger.info(f"  Elapsed: {time.time()-tc:.0f}s")

        del fold_results, cache

    logger.info(f"\nTotal: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
