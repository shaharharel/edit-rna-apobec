#!/usr/bin/env python
"""Generate negative controls for expanded v3 dataset.

Reuses existing v2 negatives (A3A=2966, A3B=4177, A3G=119) and generates
new negatives for:
  - A3G (expanded): +60 negatives for new Levanon A3G sites → total 179
  - A3A_A3G: 178 negatives, match actual motif (TC~33%, CC~65%)
  - Neither: 206 negatives, match actual motif (TC~24%, CC~35%)
  - Unknown: 72 negatives, match actual motif (TC~43%, CC~31%)

Output:
  - data/processed/multi_enzyme/splits_multi_enzyme_v3_with_negatives.csv
  - data/processed/multi_enzyme/multi_enzyme_sequences_v3_with_negatives.json

Usage:
    conda run -n quris python scripts/multi_enzyme/generate_negatives_v3.py
"""

import json
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.apobec_negatives import generate_negatives_from_genome

DATA_DIR = PROJECT_ROOT / "data"
SPLITS_V3 = DATA_DIR / "processed/multi_enzyme/splits_multi_enzyme_v3.csv"
SEQS_V3 = DATA_DIR / "processed/multi_enzyme/multi_enzyme_sequences_v3.json"
V2_NEG_SPLITS = DATA_DIR / "processed/multi_enzyme/splits_multi_enzyme_v2_with_negatives.csv"
V2_NEG_SEQS = DATA_DIR / "processed/multi_enzyme/multi_enzyme_sequences_v2_with_negatives.json"
GENOME_HG38 = DATA_DIR / "raw/genomes/hg38.fa"

OUTPUT_SPLITS = DATA_DIR / "processed/multi_enzyme/splits_multi_enzyme_v3_with_negatives.csv"
OUTPUT_SEQS = DATA_DIR / "processed/multi_enzyme/multi_enzyme_sequences_v3_with_negatives.json"

logger = logging.getLogger(__name__)

# Motif fractions from Phase 1 parsing output
ENZYME_MOTIF_CONFIG = {
    "A3G": {"tc": 0.02, "cc": 0.91, "label": "A3G expanded"},
    "A3A_A3G": {"tc": 0.33, "cc": 0.65, "label": "Both (A3A+A3G)"},
    "Neither": {"tc": 0.24, "cc": 0.35, "label": "Neither"},
    "Unknown": {"tc": 0.43, "cc": 0.31, "label": "Unknown"},
}


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load v3 positives
    v3_pos = pd.read_csv(SPLITS_V3)
    logger.info("v3 positives: %d sites", len(v3_pos))

    with open(SEQS_V3) as f:
        v3_seqs = json.load(f)
    logger.info("v3 sequences: %d", len(v3_seqs))

    # Load v2 negatives (reuse A3A, A3B, A3G_dang negatives)
    v2_all = pd.read_csv(V2_NEG_SPLITS)
    v2_neg = v2_all[v2_all["is_edited"] == 0].copy()
    logger.info("v2 negatives to reuse: %d (A3A=%d, A3B=%d, A3G=%d)",
                len(v2_neg),
                (v2_neg["enzyme"] == "A3A").sum(),
                (v2_neg["enzyme"] == "A3B").sum(),
                (v2_neg["enzyme"] == "A3G").sum())

    with open(V2_NEG_SEQS) as f:
        v2_seqs = json.load(f)

    # Merge v2 neg sequences into v3
    for sid in v2_neg["site_id"]:
        sid = str(sid)
        if sid in v2_seqs and sid not in v3_seqs:
            v3_seqs[sid] = v2_seqs[sid]

    # Collect all known sites to exclude
    known_sites = set()
    for df in [v3_pos, v2_neg]:
        for _, row in df.iterrows():
            if pd.notna(row.get("chr")) and pd.notna(row.get("start")):
                known_sites.add((str(row["chr"]), int(row["start"])))
    logger.info("Known sites to exclude: %d", len(known_sites))

    # Generate negatives for each new category
    new_negatives = []

    for enzyme, config in ENZYME_MOTIF_CONFIG.items():
        pos_subset = v3_pos[v3_pos["enzyme"] == enzyme].copy()

        # For A3G, only generate negatives for the NEW Levanon sites
        # (Dang 119 already have negatives in v2)
        if enzyme == "A3G":
            pos_subset = pos_subset[pos_subset["dataset_source"] == "levanon_advisor"].copy()

        n_pos = len(pos_subset)
        if n_pos == 0:
            logger.info("Skipping %s: no positives", enzyme)
            continue

        # Filter to sites with valid genomic coordinates
        valid_pos = pos_subset[
            pos_subset["chr"].notna() &
            pos_subset["start"].notna() &
            (pos_subset["strand"].isin(["+", "-"]))
        ].copy()

        if len(valid_pos) == 0:
            logger.warning("%s: no sites with valid coordinates, skipping", enzyme)
            continue

        logger.info("--- %s: generating %d negatives (TC=%.0f%%, CC=%.0f%%) ---",
                     config["label"], n_pos, config["tc"] * 100, config["cc"] * 100)

        neg_seqs = {}
        neg_df = generate_negatives_from_genome(
            positives_df=valid_pos,
            genome_fa=GENOME_HG38,
            target_tc_fraction=config["tc"],
            target_cc_fraction=config["cc"],
            n_negatives=n_pos,
            output_seqs=neg_seqs,
            known_sites=known_sites,
            search_window=5000,
            seed=42 + hash(enzyme) % 1000,
        )

        if len(neg_df) > 0:
            neg_df["enzyme"] = enzyme
            v3_seqs.update(neg_seqs)
            # Track new positions as known
            for _, row in neg_df.iterrows():
                known_sites.add((str(row["chr"]), int(row["start"])))
            new_negatives.append(neg_df)
            logger.info("%s: generated %d negatives", enzyme, len(neg_df))
        else:
            logger.warning("%s: failed to generate any negatives", enzyme)

    # Combine everything
    all_neg = pd.concat([v2_neg] + new_negatives, ignore_index=True)
    logger.info("Total negatives: %d", len(all_neg))

    # Merge positives + negatives
    combined = pd.concat([v3_pos, all_neg], ignore_index=True)

    # Ensure consistent columns
    output_cols = [
        "site_id", "chr", "start", "strand", "enzyme", "dataset_source",
        "coordinate_system", "editing_rate", "is_edited",
        "flanking_seq", "seq_center", "end", "source_type",
    ]
    final_cols = [c for c in output_cols if c in combined.columns]
    extra = [c for c in combined.columns if c not in output_cols]
    combined = combined[final_cols + extra]

    combined.to_csv(OUTPUT_SPLITS, index=False)
    logger.info("Saved: %s (%d rows)", OUTPUT_SPLITS.name, len(combined))

    with open(OUTPUT_SEQS, "w") as f:
        json.dump(v3_seqs, f)
    logger.info("Saved: %s (%d sequences)", OUTPUT_SEQS.name, len(v3_seqs))

    # Summary
    print("\n=== v3 Dataset with Negatives ===")
    for enzyme in sorted(combined["enzyme"].unique()):
        sub = combined[combined["enzyme"] == enzyme]
        n_pos = (sub["is_edited"] == 1).sum()
        n_neg = (sub["is_edited"] == 0).sum()
        print(f"  {enzyme}: {n_pos} pos, {n_neg} neg (ratio {n_neg/max(n_pos,1):.2f})")
    total_pos = (combined["is_edited"] == 1).sum()
    total_neg = (combined["is_edited"] == 0).sum()
    print(f"  TOTAL: {total_pos} pos, {total_neg} neg = {len(combined)} sites")


if __name__ == "__main__":
    main()
