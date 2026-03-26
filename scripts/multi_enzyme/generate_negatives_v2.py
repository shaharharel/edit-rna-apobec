#!/usr/bin/env python
"""Generate negative control sites for all 3 enzymes (A3A, A3B, A3G).

Strategy:
  A3A: Reuse 2,966 negatives from splits_expanded_a3a.csv (pre-curated TC-context negatives).
  A3B: Generate genome-based negatives matching TC%~32% near A3B positive sites (hg38).
  A3G: Generate genome-based negatives matching CC%~91% near A3G positive sites (hg19).

Output:
  - data/processed/multi_enzyme/splits_multi_enzyme_v2_with_negatives.csv
  - data/processed/multi_enzyme/multi_enzyme_sequences_v2_with_negatives.json

Usage:
    conda run -n quris python scripts/multi_enzyme/generate_negatives_v2.py
"""

import json
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.apobec_negatives import (
    generate_negatives_from_genome,
    generate_negatives_from_splits,
)

DATA_DIR = PROJECT_ROOT / "data"
SPLITS_V2 = DATA_DIR / "processed/multi_enzyme/splits_multi_enzyme_v2.csv"
SEQS_V2 = DATA_DIR / "processed/multi_enzyme/multi_enzyme_sequences_v2.json"
SPLITS_A3A = DATA_DIR / "processed/splits_expanded_a3a.csv"
SEQS_A3A = DATA_DIR / "processed/site_sequences.json"
GENOME_HG38 = DATA_DIR / "raw/genomes/hg38.fa"
GENOME_HG19 = DATA_DIR / "raw/genomes/hg19.fa"

OUTPUT_SPLITS = DATA_DIR / "processed/multi_enzyme/splits_multi_enzyme_v2_with_negatives.csv"
OUTPUT_SEQS = DATA_DIR / "processed/multi_enzyme/multi_enzyme_sequences_v2_with_negatives.json"

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load existing positives
    positives_df = pd.read_csv(SPLITS_V2)
    positives_df["is_edited"] = 1
    logger.info("Loaded %d positive sites from %s", len(positives_df), SPLITS_V2.name)

    with open(SEQS_V2) as f:
        all_sequences = json.load(f)
    logger.info("Loaded %d sequences from %s", len(all_sequences), SEQS_V2.name)

    # Collect known sites to exclude from negative generation
    known_sites = set()
    for _, row in positives_df.iterrows():
        if pd.notna(row.get("chr")) and pd.notna(row.get("start")):
            known_sites.add((str(row["chr"]), int(row["start"])))

    # Also exclude A3A known sites
    a3a_df = pd.read_csv(SPLITS_A3A)
    for _, row in a3a_df.iterrows():
        if pd.notna(row.get("chr")) and pd.notna(row.get("start")):
            known_sites.add((str(row["chr"]), int(row["start"])))
    logger.info("Total known sites to exclude: %d", len(known_sites))

    all_negatives = []

    # ---------------------------------------------------------------
    # A3A: Load 2,966 pre-curated negatives from splits_expanded_a3a
    # ---------------------------------------------------------------
    logger.info("--- A3A: loading negatives from splits_expanded_a3a.csv ---")
    a3a_neg = generate_negatives_from_splits(SPLITS_A3A, is_edited_col="is_edited")

    with open(SEQS_A3A) as f:
        a3a_seqs = json.load(f)

    a3a_neg_records = []
    for _, row in a3a_neg.iterrows():
        sid = str(row["site_id"])
        if sid not in a3a_seqs:
            continue
        all_sequences[sid] = a3a_seqs[sid]
        a3a_neg_records.append({
            "site_id": sid,
            "chr": row.get("chr", ""),
            "start": row.get("start", 0),
            "strand": row.get("strand", "+"),
            "enzyme": "A3A",
            "dataset_source": row.get("dataset_source", "a3a_neg"),
            "coordinate_system": "hg38",
            "editing_rate": 0.0,
            "flanking_seq": None,
            "seq_center": None,
            "is_edited": 0,
        })

    a3a_neg_df = pd.DataFrame(a3a_neg_records)
    all_negatives.append(a3a_neg_df)
    logger.info("A3A negatives: %d sites loaded", len(a3a_neg_df))

    # ---------------------------------------------------------------
    # A3B: Generate genome-based negatives matching TC%~32%
    # ---------------------------------------------------------------
    logger.info("--- A3B: generating genome negatives (TC%%~32%%) ---")
    a3b_pos = positives_df[positives_df["enzyme"] == "A3B"].copy()
    n_a3b = len(a3b_pos)
    logger.info("A3B positives: %d", n_a3b)

    # Only use sites with valid hg38 coords (skip kockler_2026 if coords invalid)
    a3b_genome_pos = a3b_pos[a3b_pos["coordinate_system"] == "hg38"].copy()

    a3b_neg_seqs = {}
    a3b_neg_df = generate_negatives_from_genome(
        positives_df=a3b_genome_pos,
        genome_fa=GENOME_HG38,
        target_tc_fraction=0.32,
        target_cc_fraction=0.25,
        n_negatives=n_a3b,
        output_seqs=a3b_neg_seqs,
        known_sites=known_sites,
        search_window=5000,
        seed=42,
    )

    # Add enzyme column and merge sequences
    if len(a3b_neg_df) > 0:
        a3b_neg_df["enzyme"] = "A3B"
        all_sequences.update(a3b_neg_seqs)
        # Add missing columns to match schema
        for col in ["flanking_seq", "seq_center"]:
            if col not in a3b_neg_df.columns:
                a3b_neg_df[col] = None
        all_negatives.append(a3b_neg_df)
    logger.info("A3B negatives: %d generated", len(a3b_neg_df))

    # ---------------------------------------------------------------
    # A3G: Generate genome-based negatives matching CC%~91%
    # ---------------------------------------------------------------
    logger.info("--- A3G: generating genome negatives (CC%%~91%%) ---")
    a3g_pos = positives_df[positives_df["enzyme"] == "A3G"].copy()
    n_a3g = len(a3g_pos)
    logger.info("A3G positives: %d", n_a3g)

    # A3G uses hg19 coordinates
    a3g_neg_seqs = {}
    a3g_neg_df = generate_negatives_from_genome(
        positives_df=a3g_pos,
        genome_fa=GENOME_HG19,
        target_tc_fraction=0.0,
        target_cc_fraction=0.91,
        n_negatives=n_a3g,
        output_seqs=a3g_neg_seqs,
        known_sites=known_sites,
        search_window=5000,
        seed=123,
    )

    if len(a3g_neg_df) > 0:
        a3g_neg_df["enzyme"] = "A3G"
        a3g_neg_df["coordinate_system"] = "hg19"
        all_sequences.update(a3g_neg_seqs)
        for col in ["flanking_seq", "seq_center"]:
            if col not in a3g_neg_df.columns:
                a3g_neg_df[col] = None
        all_negatives.append(a3g_neg_df)
    logger.info("A3G negatives: %d generated", len(a3g_neg_df))

    # ---------------------------------------------------------------
    # Combine everything
    # ---------------------------------------------------------------
    neg_combined = pd.concat(all_negatives, ignore_index=True)
    logger.info("Total negatives: %d (A3A=%d, A3B=%d, A3G=%d)",
                len(neg_combined),
                len(a3a_neg_df),
                len(a3b_neg_df),
                len(a3g_neg_df))

    # Merge with positives
    combined = pd.concat([positives_df, neg_combined], ignore_index=True)

    # Ensure consistent columns
    output_cols = [
        "site_id", "chr", "start", "strand", "enzyme", "dataset_source",
        "coordinate_system", "editing_rate", "is_edited",
    ]
    extra_cols = [c for c in combined.columns if c not in output_cols]
    final_cols = output_cols + extra_cols
    final_cols = [c for c in final_cols if c in combined.columns]

    combined = combined[final_cols]
    combined.to_csv(OUTPUT_SPLITS, index=False)
    logger.info("Saved %s: %d total sites (%d pos, %d neg)",
                OUTPUT_SPLITS.name, len(combined),
                (combined["is_edited"] == 1).sum(),
                (combined["is_edited"] == 0).sum())

    # Save combined sequences
    with open(OUTPUT_SEQS, "w") as f:
        json.dump(all_sequences, f, indent=2)
    logger.info("Saved %s: %d sequences", OUTPUT_SEQS.name, len(all_sequences))

    # Summary
    logger.info("--- Summary ---")
    for enzyme in sorted(combined["enzyme"].unique()):
        sub = combined[combined["enzyme"] == enzyme]
        n_pos = (sub["is_edited"] == 1).sum()
        n_neg = (sub["is_edited"] == 0).sum()
        logger.info("  %s: %d positives, %d negatives", enzyme, n_pos, n_neg)


if __name__ == "__main__":
    main()
