"""
Generate negative control sites (unedited cytidines) for multi-enzyme dataset.

For each positive editing site, selects a matched negative C from the same gene/region.
Ensures negatives are not in any known editing site list.

Strategy:
- For sites with hg38 genome coordinates, find nearby C bases in the same gene region
- Match the number of negatives to positives per enzyme
- Verify the negative C is not in any known editing site

Output: Appended to splits_multi_enzyme.csv with is_edited=0

Usage:
    conda run -n quris python scripts/multi_enzyme/generate_negatives.py
"""
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

SPLITS_CSV = DATA_DIR / "processed/multi_enzyme/splits_multi_enzyme.csv"
SEQ_JSON = DATA_DIR / "processed/multi_enzyme/multi_enzyme_sequences.json"
GENOME_FA = DATA_DIR / "raw/genomes/hg38.fa"

FLANK_SIZE = 100  # ±100nt around negative site
SEARCH_WINDOW = 5000  # Search for negatives within ±5kb of each positive


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    df = pd.read_csv(SPLITS_CSV)
    positives = df[df["is_edited"] == 1].copy()
    logger.info("Positive sites: %d", len(positives))

    # Load all known editing site positions
    known_sites = set()
    for _, row in df.iterrows():
        known_sites.add((row["chr"], int(row["start"])))

    # Also load existing A3A sites to avoid overlap
    a3a_csv = DATA_DIR / "processed/splits_expanded_a3a.csv"
    if a3a_csv.exists():
        a3a_df = pd.read_csv(a3a_csv)
        for _, row in a3a_df.iterrows():
            if "chr" in row and "start" in row:
                known_sites.add((str(row["chr"]), int(row["start"])))
    logger.info("Known editing sites (to exclude): %d", len(known_sites))

    from pyfaidx import Fasta
    genome = Fasta(str(GENOME_FA))

    random.seed(42)
    negatives = []
    skipped = 0

    # Only generate negatives for sites with valid hg38 coordinates
    # (Kockler sites don't match hg38, so skip those)
    genome_sites = positives[
        positives["dataset_source"].isin(["zhang_2024", "levanon_t3", "dang_2019"])
    ].copy()

    logger.info("Sites with valid hg38 coordinates: %d", len(genome_sites))

    for _, row in genome_sites.iterrows():
        chrom = row["chr"]
        pos = int(row["start"])
        strand = row["strand"]
        enzyme = row["enzyme"]

        if chrom not in genome:
            skipped += 1
            continue

        chrom_len = len(genome[chrom])

        # Search for unedited C bases within ±SEARCH_WINDOW
        search_start = max(0, pos - SEARCH_WINDOW)
        search_end = min(chrom_len, pos + SEARCH_WINDOW)

        # Get sequence in search window
        region_seq = str(genome[chrom][search_start:search_end]).upper()

        # Find all C positions (for + strand) or G positions (for - strand)
        target_base = "C" if strand == "+" else "G"
        candidates = []
        for i, base in enumerate(region_seq):
            if base == target_base:
                genome_pos = search_start + i + 1  # Convert to 1-based
                if (chrom, genome_pos) not in known_sites and genome_pos != pos:
                    # Check that the candidate has at least ±FLANK_SIZE context
                    if genome_pos - 1 >= FLANK_SIZE and genome_pos - 1 + FLANK_SIZE < chrom_len:
                        candidates.append(genome_pos)

        if not candidates:
            skipped += 1
            continue

        # Pick one random negative
        neg_pos = random.choice(candidates)
        neg_site_id = f"{chrom}:{neg_pos}:{strand}"

        # Extract sequence
        pos_0 = neg_pos - 1
        seq_start = pos_0 - FLANK_SIZE
        seq_end = pos_0 + FLANK_SIZE + 1
        neg_seq = str(genome[chrom][seq_start:seq_end]).upper()

        if strand == "-":
            comp = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
            neg_seq = "".join(comp.get(b, "N") for b in reversed(neg_seq))

        rna_seq = neg_seq.replace("T", "U")

        negatives.append({
            "site_id": neg_site_id,
            "chr": chrom,
            "start": neg_pos,
            "end": neg_pos,
            "strand": strand,
            "enzyme": enzyme,
            "dataset_source": row["dataset_source"] + "_neg",
            "is_edited": 0,
            "editing_rate": 0.0,
            "coordinate_system": "hg38",
            "source_type": "negative_control",
            "flanking_seq": None,
            "sequence": rna_seq,
        })

        # Add to known sites to prevent reuse
        known_sites.add((chrom, neg_pos))

    logger.info("Generated %d negative controls, %d skipped", len(negatives), skipped)

    neg_df = pd.DataFrame(negatives)
    seq_col = neg_df.pop("sequence")

    # Append to existing splits
    combined = pd.concat([df, neg_df.drop(columns=["sequence"], errors="ignore")], ignore_index=True)
    combined.to_csv(SPLITS_CSV, index=False)
    logger.info("Updated %s: %d total sites (%d positive, %d negative)",
                SPLITS_CSV, len(combined),
                (combined["is_edited"] == 1).sum(),
                (combined["is_edited"] == 0).sum())

    # Update sequences JSON
    with open(SEQ_JSON) as f:
        sequences = json.load(f)

    for neg_row, rna_seq in zip(negatives, seq_col):
        sequences[neg_row["site_id"]] = rna_seq

    with open(SEQ_JSON, "w") as f:
        json.dump(sequences, f, indent=2)
    logger.info("Updated %s: %d total sequences", SEQ_JSON, len(sequences))

    # Summary
    for enzyme in sorted(neg_df["enzyme"].unique()):
        subset = neg_df[neg_df["enzyme"] == enzyme]
        print(f"  {enzyme}: {len(subset)} negative controls generated")


if __name__ == "__main__":
    main()
