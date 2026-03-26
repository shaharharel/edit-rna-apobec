#!/usr/bin/env python
"""Re-extract full 201nt genomic sequences for ALL multi-enzyme sites.

Replaces N-padded Kockler (41nt→201nt) and Dang (31nt→201nt) sequences
with proper genome-extracted 201nt sequences.

Genome assignments (verified by center-base check):
  - Kockler 2026: hg19 (labeled hg38 in CSV, but actually hg19)
  - Dang 2019: hg38 (labeled hg19 in CSV, but actually hg38)
  - Zhang 2024: hg38 (correct, already 201nt)
  - Levanon: hg38 (correct, already 201nt)
  - All negatives: hg38 (correct, already 201nt)

After this, ALL sites will have full 201nt from the correct genome.
ViennaRNA structure features, loop positions, and classification results
must be recomputed after running this script.

Usage:
    conda run -n quris python scripts/multi_enzyme/reextract_full_201nt_sequences.py
"""

import json
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.apobec_negatives import extract_from_genome

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
SPLITS_V3 = DATA_DIR / "processed/multi_enzyme/splits_multi_enzyme_v3_with_negatives.csv"
SEQS_V3 = DATA_DIR / "processed/multi_enzyme/multi_enzyme_sequences_v3_with_negatives.json"
HG38_FA = DATA_DIR / "raw/genomes/hg38.fa"
HG19_FA = DATA_DIR / "raw/genomes/hg19.fa"

OUTPUT_SEQS = DATA_DIR / "processed/multi_enzyme/multi_enzyme_sequences_v3_with_negatives.json"
OUTPUT_SPLITS = DATA_DIR / "processed/multi_enzyme/splits_multi_enzyme_v3_with_negatives.csv"

# Correct genome assignments (verified empirically)
GENOME_MAP = {
    "kockler_2026": "hg19",
    "kockler_2026_neg": "hg38",  # negatives were generated from hg38
    "dang_2019": "hg38",
    "dang_2019_neg": "hg38",    # negatives generated from hg38
    "zhang_2024": "hg38",
    "zhang_2024_neg": "hg38",
    "levanon_advisor": "hg38",
    "levanon_advisor_neg": "hg38",
    "tier2_negative": "hg38",
    "tier3_negative": "hg38",
}


def main():
    from pyfaidx import Fasta

    splits = pd.read_csv(SPLITS_V3)
    with open(SEQS_V3) as f:
        seqs = json.load(f)

    logger.info("Loaded %d sites, %d sequences", len(splits), len(seqs))

    hg38 = Fasta(str(HG38_FA))
    hg19 = Fasta(str(HG19_FA))
    genomes = {"hg38": hg38, "hg19": hg19}

    # Count current N-padded
    n_padded_before = sum(1 for s in seqs.values() if "N" in s.upper())
    logger.info("N-padded sequences before: %d", n_padded_before)

    replaced = 0
    failed = 0
    already_ok = 0

    for idx, row in splits.iterrows():
        sid = str(row["site_id"])
        ds = str(row["dataset_source"])
        chrom = str(row["chr"])
        pos = int(row["start"])
        strand = str(row["strand"])

        # Check if sequence needs replacement
        current_seq = seqs.get(sid, "")
        has_n = "N" in current_seq.upper() if current_seq else True
        is_short = len(current_seq) < 201 if current_seq else True

        if not has_n and not is_short:
            # Already a full 201nt sequence without N
            already_ok += 1
            continue

        # Determine correct genome
        genome_build = GENOME_MAP.get(ds, "hg38")
        genome = genomes[genome_build]

        # Extract 201nt
        new_seq = extract_from_genome(genome, chrom, pos, strand)

        if new_seq is not None and len(new_seq) == 201 and new_seq[100] == "C":
            seqs[sid] = new_seq
            replaced += 1
        else:
            # Try the other genome as fallback
            alt_build = "hg19" if genome_build == "hg38" else "hg38"
            alt_seq = extract_from_genome(genomes[alt_build], chrom, pos, strand)
            if alt_seq is not None and len(alt_seq) == 201 and alt_seq[100] == "C":
                seqs[sid] = alt_seq
                replaced += 1
                logger.debug("  %s: used %s instead of %s", sid, alt_build, genome_build)
            else:
                failed += 1
                if failed <= 10:
                    logger.warning("  Failed: %s (%s:%d:%s) ds=%s, build=%s",
                                   sid, chrom, pos, strand, ds, genome_build)

    logger.info("Results: replaced=%d, already_ok=%d, failed=%d", replaced, already_ok, failed)

    # Validate
    n_padded_after = sum(1 for s in seqs.values() if "N" in s.upper())
    n_correct_center = sum(1 for sid in splits["site_id"].astype(str)
                           if sid in seqs and len(seqs[sid]) == 201 and seqs[sid][100] == "C")
    logger.info("N-padded after: %d (was %d)", n_padded_after, n_padded_before)
    logger.info("Center=C: %d/%d", n_correct_center, len(splits))

    # Also fix the coordinate_system column in splits
    for ds, build in GENOME_MAP.items():
        mask = splits["dataset_source"] == ds
        if mask.any():
            splits.loc[mask, "coordinate_system"] = build

    # Save
    with open(OUTPUT_SEQS, "w") as f:
        json.dump(seqs, f)
    logger.info("Saved sequences: %s", OUTPUT_SEQS)

    splits.to_csv(OUTPUT_SPLITS, index=False)
    logger.info("Saved splits: %s", OUTPUT_SPLITS)

    # Summary per dataset
    print("\n=== Per-Dataset Summary ===")
    for ds in sorted(splits["dataset_source"].unique()):
        sub = splits[splits["dataset_source"] == ds]
        sids = sub["site_id"].astype(str)
        n_full = sum(1 for sid in sids if sid in seqs and "N" not in seqs[sid].upper() and len(seqs[sid]) == 201)
        n_c = sum(1 for sid in sids if sid in seqs and len(seqs[sid]) == 201 and seqs[sid][100] == "C")
        print(f"  {ds}: {len(sub)} sites, {n_full} full 201nt, {n_c} center=C, genome={GENOME_MAP.get(ds, '?')}")

    print(f"\nTotal: {len(splits)} sites, {n_padded_after} still N-padded, {n_correct_center} center=C")
    if n_padded_after > 0:
        print("WARNING: Some sites still have N-padding. These may have invalid genomic coordinates.")


if __name__ == "__main__":
    main()
