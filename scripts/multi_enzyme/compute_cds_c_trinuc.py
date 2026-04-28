#!/usr/bin/env python
"""Compute CDS-C trinucleotide distribution from the panel parquet.

Reads experiments/multi_enzyme/outputs/pcawg_tcw_panel/panel_scores_cds.parquet
and builds the strand-collapsed 16-bin trinucleotide histogram.

The panel uses hg19 coordinates that are 0-indexed (BED-style); confirmed by
spot-checking 2,000 sites against the genome where 0-indexed lookup matches
2000/2000 vs 613/2000 for 1-indexed.

Output: data/processed/multi_enzyme/cds_c_trinuc_distribution.csv

Usage:
    conda run -n quris python scripts/multi_enzyme/compute_cds_c_trinuc.py
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from pyfaidx import Fasta

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PANEL_PARQUET = PROJECT_ROOT / "experiments/multi_enzyme/outputs/pcawg_tcw_panel/panel_scores_cds.parquet"
HG19_FA = PROJECT_ROOT / "data/raw/genomes/hg19.fa"
OUTPUT_CSV = PROJECT_ROOT / "data/processed/multi_enzyme/cds_c_trinuc_distribution.csv"

COMP = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}


def revcomp(s: str) -> str:
    return "".join(COMP.get(b, "N") for b in reversed(s.upper()))


def main():
    if not PANEL_PARQUET.exists():
        raise SystemExit(f"Missing panel parquet: {PANEL_PARQUET}")
    if not HG19_FA.exists():
        raise SystemExit(f"Missing hg19 fasta: {HG19_FA}")

    logger.info("Loading panel parquet ...")
    df = pd.read_parquet(PANEL_PARQUET, columns=["chrom", "pos", "strand", "valid"])
    if "valid" in df.columns:
        df = df[df["valid"] == True].drop(columns="valid")
    logger.info("Panel CDS-C positions: %d", len(df))

    logger.info("Loading hg19 fasta ...")
    genome = Fasta(str(HG19_FA))

    bases = ("A", "C", "G", "T")
    counts = {f"{a}C{b}": 0 for a in bases for b in bases}

    bad = 0
    n = 0
    # The panel is 0-indexed (BED-like). Triplet = genome[chrom][pos-1:pos+2] in 0-based slicing.
    for chrom, pos, strand in zip(df["chrom"].values, df["pos"].values, df["strand"].values):
        chrom = str(chrom)
        if chrom not in genome:
            bad += 1
            continue
        chrom_len = len(genome[chrom])
        p0 = int(pos)  # 0-indexed center
        if p0 < 1 or p0 + 2 > chrom_len:
            bad += 1
            continue
        triplet = str(genome[chrom][p0 - 1: p0 + 2]).upper()
        if len(triplet) != 3 or "N" in triplet:
            bad += 1
            continue
        if strand == "+":
            if triplet[1] != "C":
                bad += 1
                continue
            tri = triplet
        else:
            if triplet[1] != "G":
                bad += 1
                continue
            tri = revcomp(triplet)
        if tri[1] != "C":
            bad += 1
            continue
        counts[tri] = counts.get(tri, 0) + 1
        n += 1
        if n % 500000 == 0:
            logger.info("  processed %d / %d", n, len(df))

    total = sum(counts.values())
    logger.info("CDS-C trinuc lookups: %d ok, %d skipped", total, bad)

    rows = []
    for tri, cnt in counts.items():
        rows.append({"trinuc": tri, "count": cnt, "fraction": cnt / total if total else 0.0})
    out = pd.DataFrame(rows).sort_values("fraction", ascending=False).reset_index(drop=True)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)
    logger.info("Saved: %s", OUTPUT_CSV)

    print("\nCDS-C trinucleotide distribution (top bins):")
    for _, r in out.iterrows():
        print(f"  {r['trinuc']}: {r['count']:>10,d}  ({100*r['fraction']:5.2f}%)")
    print(f"  TOTAL: {total:,d}")


if __name__ == "__main__":
    main()
