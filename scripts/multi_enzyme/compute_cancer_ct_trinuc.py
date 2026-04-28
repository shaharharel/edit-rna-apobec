#!/usr/bin/env python
"""Compute cancer C>T trinucleotide context distribution from TCGA + PCAWG.

For each C>T or G>A SNV in the cancer MAFs, look up the trinucleotide
context (5'N + C + 3'N) using hg19 fasta. G>A SNVs are strand-corrected
by reverse-complement so all are reported on the C>T strand.

Output: data/processed/multi_enzyme/cancer_ct_trinuc_distribution.csv
        with columns: trinuc, count, fraction.

Usage:
    conda run -n quris python scripts/multi_enzyme/compute_cancer_ct_trinuc.py
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from pyfaidx import Fasta

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TCGA_DIR = PROJECT_ROOT / "data/raw/tcga"
PCAWG_DIR = PROJECT_ROOT / "data/raw/pcawg/by_cancer"
HG19_FA = PROJECT_ROOT / "data/raw/genomes/hg19.fa"

OUTPUT_CSV = PROJECT_ROOT / "data/processed/multi_enzyme/cancer_ct_trinuc_distribution.csv"

COMP = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}


def revcomp(s: str) -> str:
    return "".join(COMP.get(b, "N") for b in reversed(s.upper()))


def normalize_chrom(chrom: str) -> str:
    """Make chromosome name compatible with hg19.fa keys (chr-prefixed)."""
    chrom = str(chrom)
    if not chrom.startswith("chr"):
        chrom = "chr" + chrom
    return chrom


def read_maf(path: Path) -> pd.DataFrame:
    """Read MAF/mutations file robustly."""
    # All TCGA and PCAWG-coding files are tab-delimited with a one-line header
    df = pd.read_csv(path, sep="\t", low_memory=False, comment="#")
    return df


def extract_ct_snvs(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Return SNVs whose substitution is C>T or G>A (treated as C>T on opposite strand)."""
    needed_cols = ["Chromosome", "Start_Position", "Reference_Allele",
                   "Tumor_Seq_Allele2", "Variant_Type"]
    for c in needed_cols:
        if c not in df.columns:
            logger.warning("[%s] missing column %s, skipping", source_name, c)
            return pd.DataFrame()
    sub = df[df["Variant_Type"] == "SNP"].copy()
    ref = sub["Reference_Allele"].astype(str).str.upper()
    alt = sub["Tumor_Seq_Allele2"].astype(str).str.upper()
    is_ct = (ref == "C") & (alt == "T")
    is_ga = (ref == "G") & (alt == "A")
    keep = sub[is_ct | is_ga].copy()
    keep["_is_ct_strand"] = (ref[is_ct | is_ga] == "C").values
    keep = keep[["Chromosome", "Start_Position", "_is_ct_strand"]]
    keep["source"] = source_name
    return keep


def lookup_trinuc(genome: Fasta, chrom: str, pos: int, ref_is_c: bool) -> str:
    """Return strand-normalized trinuc with center C (5'N + C + 3'N).

    pos is 1-based. If ref_is_c is False (i.e. ref is G), reverse-complement
    the trinucleotide so that the center base is C.
    """
    chrom = normalize_chrom(chrom)
    if chrom not in genome:
        return ""
    chrom_len = len(genome[chrom])
    p0 = int(pos) - 1
    if p0 < 1 or p0 + 2 > chrom_len:
        return ""
    triplet = str(genome[chrom][p0 - 1:p0 + 2]).upper()
    if len(triplet) != 3 or "N" in triplet:
        return ""
    if ref_is_c:
        if triplet[1] != "C":
            return ""
        return triplet
    else:
        # ref is G; revcomp triplet so center becomes C
        if triplet[1] != "G":
            return ""
        return revcomp(triplet)


def main():
    if not HG19_FA.exists():
        raise SystemExit(f"hg19 fasta not found: {HG19_FA}")
    logger.info("Loading hg19 fasta...")
    genome = Fasta(str(HG19_FA))

    # Collect all C>T SNVs
    all_ct = []

    for path in sorted(TCGA_DIR.glob("*_mutations.txt")):
        logger.info("Reading TCGA %s", path.name)
        try:
            df = read_maf(path)
        except Exception as e:
            logger.warning("Failed to read %s: %s", path.name, e)
            continue
        ct = extract_ct_snvs(df, f"tcga:{path.stem}")
        if len(ct):
            logger.info("  %s: %d C>T/G>A SNVs", path.name, len(ct))
            all_ct.append(ct)

    for path in sorted(PCAWG_DIR.glob("*_pcawg_mutations.txt")):
        logger.info("Reading PCAWG %s", path.name)
        try:
            df = read_maf(path)
        except Exception as e:
            logger.warning("Failed to read %s: %s", path.name, e)
            continue
        ct = extract_ct_snvs(df, f"pcawg:{path.stem}")
        if len(ct):
            logger.info("  %s: %d C>T/G>A SNVs", path.name, len(ct))
            all_ct.append(ct)

    if not all_ct:
        raise SystemExit("No C>T SNVs found in any input file.")
    big = pd.concat(all_ct, ignore_index=True)
    logger.info("Total C>T/G>A SNVs across all cancers: %d", len(big))

    # Look up trinucleotide context
    counter = {}
    bad = 0
    for chrom, pos, ref_is_c in zip(big["Chromosome"], big["Start_Position"], big["_is_ct_strand"]):
        try:
            pos_int = int(pos)
        except Exception:
            bad += 1
            continue
        trinuc = lookup_trinuc(genome, chrom, pos_int, bool(ref_is_c))
        if not trinuc or len(trinuc) != 3:
            bad += 1
            continue
        counter[trinuc] = counter.get(trinuc, 0) + 1

    total = sum(counter.values())
    logger.info("Trinuc lookups: %d ok, %d skipped (bad chrom / bound / N)", total, bad)

    # Build full 16-bin histogram (all NCN with N in {A,C,G,T})
    bases = ["A", "C", "G", "T"]
    rows = []
    for b5 in bases:
        for b3 in bases:
            triplet = f"{b5}C{b3}"
            count = counter.get(triplet, 0)
            rows.append({
                "trinuc": triplet,
                "count": count,
                "fraction": count / total if total else 0.0,
            })
    out = pd.DataFrame(rows).sort_values("fraction", ascending=False).reset_index(drop=True)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)
    logger.info("Saved: %s", OUTPUT_CSV)

    # Print top bins
    print("\nTop trinucleotide contexts (C>T cancer mutations):")
    for _, r in out.iterrows():
        print(f"  {r['trinuc']}: {r['count']:>10,d}  ({100 * r['fraction']:5.2f}%)")
    print(f"  TOTAL: {total:,d}")


if __name__ == "__main__":
    main()
