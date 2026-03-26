"""Build expanded multi-enzyme dataset v3 with Levanon categories.

Merges existing v2 dataset (A3A=2749, A3B=4180, A3G=119 + negatives)
with newly parsed Levanon categories:
  - A3G: +60 (Levanon A3G-only) → total 179
  - A3A_A3G: +178 (Levanon Both)
  - Neither: +206 (Levanon Neither)
  - Unknown: +72 (Levanon NaN enzyme assignment)

Site IDs: new sites use existing C2U_NNNN IDs from Levanon T1.
Sequences: already in data/processed/site_sequences.json for all 636.

Output:
- data/processed/multi_enzyme/splits_multi_enzyme_v3.csv (positives only)
- data/processed/multi_enzyme/multi_enzyme_sequences_v3.json

Usage:
    conda run -n quris python scripts/multi_enzyme/build_multi_enzyme_dataset_v3.py
"""

import json
import logging
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

# Input files
V2_SPLITS = DATA_DIR / "processed/multi_enzyme/splits_multi_enzyme_v2_with_negatives.csv"
V2_SEQS = DATA_DIR / "processed/multi_enzyme/multi_enzyme_sequences_v2_with_negatives.json"
LEVANON_CSV = DATA_DIR / "processed/multi_enzyme/levanon_all_categories.csv"
ALL_SEQS = DATA_DIR / "processed/site_sequences.json"

# Output files
OUTPUT_CSV = DATA_DIR / "processed/multi_enzyme/splits_multi_enzyme_v3.csv"
OUTPUT_SEQ = DATA_DIR / "processed/multi_enzyme/multi_enzyme_sequences_v3.json"


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Check inputs
    for path in [V2_SPLITS, V2_SEQS, LEVANON_CSV, ALL_SEQS]:
        if not path.exists():
            logger.error("Missing: %s", path)
            sys.exit(1)

    # Load v2 dataset (positives only for base)
    v2 = pd.read_csv(V2_SPLITS)
    logger.info("v2 dataset: %d rows", len(v2))
    logger.info("v2 enzyme breakdown:\n%s", v2.groupby(["enzyme", "is_edited"]).size().to_string())

    # Load v2 sequences
    with open(V2_SEQS) as f:
        v2_seqs = json.load(f)
    logger.info("v2 sequences: %d", len(v2_seqs))

    # Load all Levanon/Advisor sequences
    with open(ALL_SEQS) as f:
        all_seqs = json.load(f)
    logger.info("All sequences (site_sequences.json): %d", len(all_seqs))

    # Load Levanon categories
    lev = pd.read_csv(LEVANON_CSV)
    logger.info("Levanon categories: %d rows", len(lev))
    logger.info("Category breakdown:\n%s", lev["enzyme_category"].value_counts().to_string())

    # Get existing v2 site_ids to avoid duplicates
    v2_site_ids = set(v2["site_id"].astype(str))

    # The 120 A3A-only Levanon sites are already in v2 as advisor_c2t with C2U_ IDs
    # The A3G-only sites need to be added to existing A3G (Dang 119)
    # A3A_A3G, Neither, and Unknown are entirely new

    # Identify which Levanon sites to add
    # Categories to add: A3G (expand), A3A_A3G (new), Neither (new), Unknown (new)
    new_categories = ["A3G", "A3A_A3G", "Neither", "Unknown"]
    lev_new = lev[lev["enzyme_category"].isin(new_categories)].copy()

    # Check for duplicates with v2 (by chr+start)
    if "chr" in lev_new.columns and "start" in lev_new.columns:
        v2_positions = set(
            v2["chr"].astype(str) + ":" + v2["start"].astype(str)
        )
        lev_new["pos_key"] = lev_new["chr"].astype(str) + ":" + lev_new["start"].astype(str)
        dupes = lev_new["pos_key"].isin(v2_positions)
        if dupes.any():
            logger.warning("Found %d Levanon sites already in v2 (by position), skipping",
                           dupes.sum())
            lev_new = lev_new[~dupes].copy()
        lev_new = lev_new.drop(columns=["pos_key"])

    logger.info("New sites to add from Levanon: %d", len(lev_new))
    logger.info("  A3G: %d", (lev_new["enzyme_category"] == "A3G").sum())
    logger.info("  A3A_A3G: %d", (lev_new["enzyme_category"] == "A3A_A3G").sum())
    logger.info("  Neither: %d", (lev_new["enzyme_category"] == "Neither").sum())
    logger.info("  Unknown: %d", (lev_new["enzyme_category"] == "Unknown").sum())

    # Build new rows matching v2 column schema
    new_rows = []
    for _, row in lev_new.iterrows():
        new_rows.append({
            "site_id": row["site_id"],
            "chr": row["chr"],
            "start": row["start"],
            "strand": row["strand"],
            "enzyme": row["enzyme_category"],
            "dataset_source": "levanon_advisor",
            "coordinate_system": "hg38",
            "editing_rate": row.get("editing_rate", 0.0),
            "is_edited": 1,
            "flanking_seq": None,
            "seq_center": 100.0,
            "end": row.get("end", None),
            "source_type": "expression_correlated",
        })

    new_df = pd.DataFrame(new_rows)

    # Merge with v2 positives only (we'll regenerate negatives in Phase 3)
    v2_positives = v2[v2["is_edited"] == 1].copy()
    v3 = pd.concat([v2_positives, new_df], ignore_index=True)

    # Sort by enzyme, chr, start
    v3 = v3.sort_values(["enzyme", "chr", "start"]).reset_index(drop=True)

    logger.info("\nv3 dataset (positives only): %d rows", len(v3))
    logger.info("v3 enzyme breakdown:\n%s", v3.groupby("enzyme").size().to_string())

    # Build v3 sequences
    v3_seqs = dict(v2_seqs)
    added_seqs = 0
    missing_seqs = 0
    for _, row in new_df.iterrows():
        sid = row["site_id"]
        if sid in all_seqs:
            v3_seqs[sid] = all_seqs[sid]
            added_seqs += 1
        else:
            logger.warning("No sequence for %s", sid)
            missing_seqs += 1

    logger.info("Sequences: %d total (%d new, %d missing)", len(v3_seqs), added_seqs, missing_seqs)

    # Validate center base
    wrong_center = 0
    for sid in v3["site_id"]:
        seq = v3_seqs.get(str(sid), "")
        if len(seq) == 201 and seq[100] != "C":
            wrong_center += 1
    logger.info("Center base validation: %d wrong (should be 0)", wrong_center)

    # Save outputs
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    v3.to_csv(OUTPUT_CSV, index=False)
    logger.info("Saved: %s", OUTPUT_CSV)

    with open(OUTPUT_SEQ, "w") as f:
        json.dump(v3_seqs, f)
    logger.info("Saved: %s", OUTPUT_SEQ)

    # Print summary
    print("\n=== v3 Multi-Enzyme Dataset Summary ===")
    for enzyme in sorted(v3["enzyme"].unique()):
        subset = v3[v3["enzyme"] == enzyme]
        sources = subset["dataset_source"].value_counts().to_dict()
        source_str = ", ".join(f"{s}={n}" for s, n in sources.items())
        print(f"  {enzyme}: {len(subset)} positives ({source_str})")
    print(f"  TOTAL: {len(v3)} positives")
    print(f"  Sequences: {len(v3_seqs)}")


if __name__ == "__main__":
    main()
