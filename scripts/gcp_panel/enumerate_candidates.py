#!/usr/bin/env python3
"""Enumerate all C positions on both strands within CDS/promoter/3'UTR intervals.

Pipeline:
  1. Parse refGene hg19: select the longest transcript per gene (by total CDS length).
  2. Build three interval sets per transcript:
       - CDS: union of exonic intervals intersected with [cdsStart, cdsEnd].
       - Promoter (2 kb): [txStart - 2000, txStart] on '+' strand,
                           [txEnd, txEnd + 2000] on '-' strand.
       - 3' UTR: on '+' [cdsEnd, txEnd]; on '-' [txStart, cdsStart].
  3. Merge overlapping intervals across (strand, region_type) separately,
     then scan hg19 for every C on '+' strand and every G on '-' strand
     (i.e., every C on the transcript coding strand).
  4. Candidate rows: (chrom, pos, strand, region_type, gene).
     Overlaps across region types are resolved by priority: CDS > 3UTR > Promoter.
  5. Filter to primary contigs (chr1-22, chrX, chrY, chrM) and enforce a
     ±100 nt buffer so every candidate has a full 201-nt flanking window.

Output: data/processed/gcp_panel/candidates.parquet
        data/processed/gcp_panel/candidates_summary.json

Runtime: ~5-10 minutes locally.
"""
from __future__ import annotations

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
HG19_FA = PROJECT_ROOT / "data" / "raw" / "genomes" / "hg19.fa"
REFGENE_HG19 = PROJECT_ROOT / "data" / "raw" / "genomes" / "refGene_hg19.txt"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "gcp_panel"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

PROMOTER_LEN = 2000
FLANK = 100  # need ±100 nt for a 201-nt window
PRIMARY_CHROMS = {f"chr{i}" for i in range(1, 23)} | {"chrX", "chrY", "chrM"}

# Region priority (lower number = higher priority)
REGION_PRIORITY = {"CDS": 0, "3UTR": 1, "promoter": 2}


def parse_refgene(path: Path):
    """Parse refGene.txt, return per-transcript dict.

    Each record: dict(chrom, strand, tx_start, tx_end, cds_start, cds_end,
                      exon_starts, exon_ends, gene, cds_len).
    """
    records = []
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 13:
                continue
            chrom = parts[2]
            if chrom not in PRIMARY_CHROMS:
                continue
            strand = parts[3]
            tx_s, tx_e = int(parts[4]), int(parts[5])
            cds_s, cds_e = int(parts[6]), int(parts[7])
            exon_s = [int(x) for x in parts[9].rstrip(",").split(",") if x]
            exon_e = [int(x) for x in parts[10].rstrip(",").split(",") if x]
            gene = parts[12] if len(parts) > 12 else parts[1]

            # CDS length = sum over exons of intersection with [cds_s, cds_e]
            cds_len = 0
            for s, e in zip(exon_s, exon_e):
                a, b = max(s, cds_s), min(e, cds_e)
                if b > a:
                    cds_len += b - a

            records.append({
                "chrom": chrom,
                "strand": strand,
                "tx_start": tx_s,
                "tx_end": tx_e,
                "cds_start": cds_s,
                "cds_end": cds_e,
                "exon_starts": exon_s,
                "exon_ends": exon_e,
                "gene": gene,
                "cds_len": cds_len,
            })
    return records


def select_longest_per_gene(records):
    """Pick one transcript per gene: longest CDS; ties broken by txEnd - txStart."""
    by_gene = defaultdict(list)
    for r in records:
        by_gene[r["gene"]].append(r)
    chosen = []
    for gene, recs in by_gene.items():
        # Keep records with a real CDS (coding) separately from non-coding
        coding = [r for r in recs if r["cds_len"] > 0]
        pool = coding if coding else recs
        pool.sort(
            key=lambda r: (r["cds_len"], r["tx_end"] - r["tx_start"]),
            reverse=True,
        )
        chosen.append(pool[0])
    return chosen


def build_intervals(transcripts, chrom_sizes):
    """For each (chrom, strand, region_type) build raw intervals.

    Returns dict: (chrom, strand, region_type) -> list of (start, end, gene).
    Coordinates are 0-based half-open, clipped to chromosome bounds.
    """
    intervals = defaultdict(list)

    for t in transcripts:
        chrom = t["chrom"]
        strand = t["strand"]
        gene = t["gene"]
        chrom_len = chrom_sizes.get(chrom)
        if chrom_len is None:
            continue

        tx_s, tx_e = t["tx_start"], t["tx_end"]
        cds_s, cds_e = t["cds_start"], t["cds_end"]

        # --- CDS: exonic ∩ [cds_s, cds_e] ---
        if cds_e > cds_s:
            for es, ee in zip(t["exon_starts"], t["exon_ends"]):
                a = max(es, cds_s)
                b = min(ee, cds_e)
                a = max(a, 0)
                b = min(b, chrom_len)
                if b > a:
                    intervals[(chrom, strand, "CDS")].append((a, b, gene))

        # --- Promoter: 2 kb upstream of TSS on the transcript's strand ---
        if strand == "+":
            p_s = max(0, tx_s - PROMOTER_LEN)
            p_e = min(tx_s, chrom_len)
        else:
            p_s = max(0, tx_e)
            p_e = min(tx_e + PROMOTER_LEN, chrom_len)
        if p_e > p_s:
            intervals[(chrom, strand, "promoter")].append((p_s, p_e, gene))

        # --- 3' UTR: after stop codon to txEnd on '+', or txStart to cds_start on '-' ---
        if cds_e > cds_s:  # only for coding transcripts
            if strand == "+":
                u_s = max(cds_e, 0)
                u_e = min(tx_e, chrom_len)
            else:
                u_s = max(tx_s, 0)
                u_e = min(cds_s, chrom_len)
            # Intersect with exons
            if u_e > u_s:
                for es, ee in zip(t["exon_starts"], t["exon_ends"]):
                    a = max(es, u_s)
                    b = min(ee, u_e)
                    a = max(a, 0)
                    b = min(b, chrom_len)
                    if b > a:
                        intervals[(chrom, strand, "3UTR")].append((a, b, gene))

    return intervals


def merge_intervals(iv_list):
    """Merge overlapping (start, end, gene). Keep earliest gene on overlap."""
    if not iv_list:
        return []
    iv_list = sorted(iv_list, key=lambda x: (x[0], x[1]))
    merged = [list(iv_list[0])]
    for s, e, g in iv_list[1:]:
        if s <= merged[-1][1]:
            if e > merged[-1][1]:
                merged[-1][1] = e
        else:
            merged.append([s, e, g])
    return [tuple(x) for x in merged]


def enumerate_c_positions(intervals, genome, chrom_sizes):
    """For each (chrom, strand, region_type) merged interval, scan genome for
    C on + strand or G on - strand. Apply ±FLANK bounds.
    Returns dict: (chrom, strand) -> list of (pos, region_type, gene).
    """
    per_chrom_strand = defaultdict(list)

    # Merge per key first
    merged_by_key = {}
    for key, ivs in intervals.items():
        merged_by_key[key] = merge_intervals(ivs)

    for key, merged in merged_by_key.items():
        chrom, strand, region_type = key
        chrom_len = chrom_sizes.get(chrom, 0)
        chrom_seq = None  # lazy load
        for s, e, gene in merged:
            # enforce flank
            s2 = max(s, FLANK)
            e2 = min(e, chrom_len - FLANK)
            if e2 <= s2:
                continue
            if chrom_seq is None:
                chrom_seq = str(genome[chrom][:]).upper()
            sub = chrom_seq[s2:e2]
            base = "C" if strand == "+" else "G"
            # Find all positions of base
            idx = 0
            while True:
                j = sub.find(base, idx)
                if j < 0:
                    break
                per_chrom_strand[(chrom, strand)].append((s2 + j, region_type, gene))
                idx = j + 1

    return per_chrom_strand


def resolve_priority(per_chrom_strand):
    """Within each (chrom, strand), group by pos and keep highest-priority region type."""
    rows = []
    for (chrom, strand), entries in per_chrom_strand.items():
        # group by pos
        best = {}
        for pos, rt, gene in entries:
            pri = REGION_PRIORITY[rt]
            if pos not in best or pri < best[pos][0]:
                best[pos] = (pri, rt, gene)
        for pos, (_, rt, gene) in best.items():
            rows.append((chrom, pos, strand, rt, gene))
    return rows


def main():
    t0 = time.time()

    logger.info("Loading hg19 genome index ...")
    genome = Fasta(str(HG19_FA))
    chrom_sizes = {c: len(genome[c]) for c in genome.keys() if c in PRIMARY_CHROMS}
    logger.info("  %d primary chromosomes", len(chrom_sizes))

    logger.info("Parsing refGene ...")
    records = parse_refgene(REFGENE_HG19)
    logger.info("  %d transcripts on primary contigs", len(records))

    logger.info("Selecting longest transcript per gene ...")
    transcripts = select_longest_per_gene(records)
    logger.info("  %d genes", len(transcripts))
    coding = sum(1 for t in transcripts if t["cds_len"] > 0)
    logger.info("  coding: %d, non-coding: %d", coding, len(transcripts) - coding)

    logger.info("Building intervals (CDS / promoter / 3UTR) ...")
    intervals = build_intervals(transcripts, chrom_sizes)

    # Report raw interval counts and total bp
    for rt in ("CDS", "promoter", "3UTR"):
        bp = 0
        n = 0
        for key, ivs in intervals.items():
            if key[2] == rt:
                n += len(ivs)
                for s, e, _ in ivs:
                    bp += e - s
        logger.info("  %s: %d raw intervals, %.1f Mb", rt, n, bp / 1e6)

    logger.info("Enumerating C positions (this reads full chromosomes) ...")
    per_chrom_strand = enumerate_c_positions(intervals, genome, chrom_sizes)
    n_raw = sum(len(v) for v in per_chrom_strand.values())
    logger.info("  %d raw (pos, region) tuples", n_raw)

    logger.info("Resolving overlaps by priority (CDS > 3UTR > promoter) ...")
    rows = resolve_priority(per_chrom_strand)
    logger.info("  %d unique (chrom, pos, strand) candidates", len(rows))

    # Build dataframe and summarize
    df = pd.DataFrame(rows, columns=["chrom", "pos", "strand", "region_type", "gene"])
    df = df.sort_values(["chrom", "pos", "strand"]).reset_index(drop=True)

    # Summary stats
    breakdown = df.groupby(["region_type", "strand"]).size().unstack(fill_value=0)
    logger.info("\nBreakdown by (region_type, strand):\n%s", breakdown.to_string())
    by_region = df["region_type"].value_counts().to_dict()
    by_strand = df["strand"].value_counts().to_dict()

    out_path = OUT_DIR / "candidates.parquet"
    df.to_parquet(out_path, index=False)
    logger.info("Wrote %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)

    summary = {
        "n_candidates": int(len(df)),
        "n_genes": int(df["gene"].nunique()),
        "by_region_type": {k: int(v) for k, v in by_region.items()},
        "by_strand": {k: int(v) for k, v in by_strand.items()},
        "elapsed_sec": round(time.time() - t0, 1),
    }
    with open(OUT_DIR / "candidates_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary: %s", json.dumps(summary, indent=2))
    logger.info("Total: %.1f min", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
