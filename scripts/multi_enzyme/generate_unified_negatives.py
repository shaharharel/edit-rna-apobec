#!/usr/bin/env python
"""Generate UNIFIED negatives for the multi-enzyme task.

Problem with current negatives: each enzyme has its OWN motif-matched negatives,
creating per-enzyme motif overlap that makes the unified task meaningless.
A3B negatives have 99% motif overlap with positives → GB can't learn motif signal.

Solution: ONE shared negative pool with NATURAL genomic distribution (~25% each
dinucleotide), used for ALL enzymes. This way:
- The binary task (edited vs not) can use ALL signals (motif, structure, context)
- The enzyme task (which enzyme) is defined over positives only
- Each enzyme's classifier sees negatives that DON'T match its motif → motif becomes informative

Strategy:
- Sample random cytidines from genes containing ANY positive site
- Natural dinucleotide distribution (no motif matching)
- 1:1 ratio vs total positives
- Exclude all known editing sites across all enzymes

Output:
- splits_multi_enzyme_v3_unified_negatives.csv
- multi_enzyme_sequences_v3_unified_negatives.json

Usage:
    conda run -n quris python scripts/multi_enzyme/generate_unified_negatives.py
"""

import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.apobec_negatives import extract_from_genome

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
SPLITS_V3_POS = DATA_DIR / "processed/multi_enzyme/splits_multi_enzyme_v3.csv"
SEQS_V3 = DATA_DIR / "processed/multi_enzyme/multi_enzyme_sequences_v3_with_negatives.json"
GENOME_HG38 = DATA_DIR / "raw/genomes/hg38.fa"
GENOME_HG19 = DATA_DIR / "raw/genomes/hg19.fa"

OUTPUT_SPLITS = DATA_DIR / "processed/multi_enzyme/splits_multi_enzyme_v3_unified_negatives.csv"
OUTPUT_SEQS = DATA_DIR / "processed/multi_enzyme/multi_enzyme_sequences_v3_unified_negatives.json"

SEED = 42
FLANK = 100
CENTER = 100


def main():
    from pyfaidx import Fasta

    # Load positives only
    pos_df = pd.read_csv(SPLITS_V3_POS)
    pos_df["is_edited"] = 1
    n_pos = len(pos_df)
    logger.info("Positives: %d sites across %d enzymes", n_pos, pos_df["enzyme"].nunique())
    logger.info("Enzyme breakdown: %s", pos_df["enzyme"].value_counts().to_dict())

    # Load existing sequences
    with open(SEQS_V3) as f:
        all_seqs = json.load(f)

    # Collect ALL known editing sites to exclude
    known_sites = set()
    for _, row in pos_df.iterrows():
        if pd.notna(row.get("chr")) and pd.notna(row.get("start")):
            known_sites.add((str(row["chr"]), int(row["start"])))
    # Also exclude from A3A-specific dataset
    a3a_path = DATA_DIR / "processed/splits_expanded_a3a.csv"
    if a3a_path.exists():
        a3a_df = pd.read_csv(a3a_path)
        for _, row in a3a_df.iterrows():
            if pd.notna(row.get("chr")) and pd.notna(row.get("start")):
                known_sites.add((str(row["chr"]), int(row["start"])))
    logger.info("Known sites to exclude: %d", len(known_sites))

    # Target: same number of negatives as positives
    n_target = n_pos
    logger.info("Target negatives: %d (1:1 ratio)", n_target)

    # Open genomes
    hg38 = Fasta(str(GENOME_HG38))
    hg19 = Fasta(str(GENOME_HG19))

    # Collect candidate regions: ±5000 around ALL positive sites
    # Use the correct genome for each site
    rng = random.Random(SEED)
    candidates = []
    search_window = 5000

    # Determine which genome each site uses
    genome_map = {
        "kockler_2026": ("hg19", hg19),
        "dang_2019": ("hg38", hg38),
        "zhang_2024": ("hg38", hg38),
        "levanon_advisor": ("hg38", hg38),
    }

    logger.info("Scanning for candidate negatives near all positive sites...")
    for _, row in pos_df.iterrows():
        chrom = str(row["chr"])
        pos = int(row["start"])
        strand = str(row["strand"])
        ds = str(row["dataset_source"])

        genome_name, genome = genome_map.get(ds, ("hg38", hg38))

        if chrom not in genome:
            continue
        chrom_len = len(genome[chrom])
        start = max(0, pos - search_window)
        end = min(chrom_len, pos + search_window)

        # Sample random positions (not all — that would be too many)
        n_sample = max(5, n_target // n_pos * 3)  # oversample 3x
        for _ in range(n_sample):
            rand_pos = rng.randint(start, end)
            if (chrom, rand_pos) in known_sites:
                continue
            # Check if position has enough flanking
            if rand_pos - 1 < FLANK or rand_pos + FLANK >= chrom_len:
                continue
            candidates.append({
                "chrom": chrom, "pos": rand_pos, "strand": strand,
                "genome_name": genome_name, "ds": ds,
            })

    logger.info("Candidate positions: %d", len(candidates))
    rng.shuffle(candidates)

    # Extract sequences and validate center=C
    negatives = []
    neg_seqs = {}
    used_positions = set(known_sites)

    for cand in candidates:
        if len(negatives) >= n_target:
            break

        chrom = cand["chrom"]
        pos = cand["pos"]
        strand = cand["strand"]
        genome_name = cand["genome_name"]
        genome = hg19 if genome_name == "hg19" else hg38

        if (chrom, pos) in used_positions:
            continue

        seq = extract_from_genome(genome, chrom, pos, strand)
        if seq is None or len(seq) != 201 or seq[CENTER] != "C":
            continue

        sid = f"uneg_{chrom}:{pos}:{strand}"
        used_positions.add((chrom, pos))
        neg_seqs[sid] = seq

        negatives.append({
            "site_id": sid,
            "chr": chrom,
            "start": pos,
            "strand": strand,
            "enzyme": "unified_neg",
            "dataset_source": "unified_negative",
            "coordinate_system": genome_name,
            "editing_rate": 0.0,
            "is_edited": 0,
            "source_type": "unified_negative",
        })

    logger.info("Generated %d unified negatives", len(negatives))

    # Check motif distribution of negatives (should be ~natural genomic)
    tc = cc = ac = gc = 0
    for sid, seq in neg_seqs.items():
        up = seq[99].upper()
        if up in ("U", "T"): tc += 1
        elif up == "C": cc += 1
        elif up == "A": ac += 1
        elif up == "G": gc += 1
    n = len(neg_seqs)
    logger.info("Unified negative motif: TC=%.1f%%, CC=%.1f%%, AC=%.1f%%, GC=%.1f%%",
                tc/n*100, cc/n*100, ac/n*100, gc/n*100)

    # Combine positives + unified negatives
    neg_df = pd.DataFrame(negatives)

    # For the unified task, each positive keeps its enzyme label
    # Each negative gets enzyme="unified_neg" (not assigned to any enzyme)
    combined = pd.concat([pos_df, neg_df], ignore_index=True)
    combined = combined.sort_values(["enzyme", "chr", "start"]).reset_index(drop=True)

    # Merge sequences
    unified_seqs = {}
    for sid in pos_df["site_id"]:
        if str(sid) in all_seqs:
            unified_seqs[str(sid)] = all_seqs[str(sid)]
    unified_seqs.update(neg_seqs)

    # Save
    combined.to_csv(OUTPUT_SPLITS, index=False)
    logger.info("Saved: %s (%d rows: %d pos, %d neg)",
                OUTPUT_SPLITS.name, len(combined),
                (combined["is_edited"] == 1).sum(),
                (combined["is_edited"] == 0).sum())

    with open(OUTPUT_SEQS, "w") as f:
        json.dump(unified_seqs, f)
    logger.info("Saved: %s (%d sequences)", OUTPUT_SEQS.name, len(unified_seqs))

    # Summary
    print("\n=== Unified Negative Summary ===")
    print(f"Positives: {n_pos}")
    print(f"Negatives: {len(negatives)}")
    print(f"Negative motif: TC={tc/n*100:.1f}%, CC={cc/n*100:.1f}%, AC={ac/n*100:.1f}%, GC={gc/n*100:.1f}%")
    print("\nPositive enzyme breakdown:")
    for enz, cnt in pos_df["enzyme"].value_counts().items():
        print(f"  {enz}: {cnt}")
    print(f"\nTotal dataset: {len(combined)} sites")


if __name__ == "__main__":
    main()
