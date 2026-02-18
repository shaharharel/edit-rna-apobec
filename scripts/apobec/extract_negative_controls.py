"""Extract negative control sites from the Supp TX non-AG mismatch data.

The Supp TX sheet contains 5,230 non-AG mismatch sites, including 2,587
C-to-T (CT) sites. These CT sites that were dropped by the editing filter
serve as negative controls -- genomic positions with C-to-T mismatches that
are NOT genuine APOBEC editing events (sequencing errors, SNPs, etc).

This script creates:
1. negative_controls_ct.csv: All CT mismatch sites not in the positive set
2. positive_negative_combined.csv: Merged table with labels for ML training

Usage:
    python scripts/apobec/extract_negative_controls.py
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "advisor"


def extract_negative_controls():
    """Extract CT mismatch sites as negative controls."""
    # Load the non-AG mismatch sites
    mm_path = PROCESSED_DIR / "supp_tx_all_non_ag_mm_sites.csv"
    if not mm_path.exists():
        logger.error("Supp TX CSV not found. Run parse_advisor_excel.py first.")
        return

    mm_df = pd.read_csv(mm_path)
    logger.info("Loaded %d non-AG mismatch sites", len(mm_df))

    # Filter to CT mismatches only
    ct_sites = mm_df[mm_df["Mismatch"] == "CT"].copy()
    logger.info("CT mismatch sites: %d", len(ct_sites))

    # Load the positive editing sites to exclude them
    unified_path = PROCESSED_DIR / "unified_editing_sites.csv"
    if not unified_path.exists():
        logger.error("Unified site table not found.")
        return

    pos_df = pd.read_csv(unified_path)
    logger.info("Positive editing sites: %d", len(pos_df))

    # Create coordinate keys for matching
    pos_keys = set(
        zip(pos_df["chr"], pos_df["start"], pos_df["end"])
    )

    ct_keys = list(
        zip(ct_sites["Chr"], ct_sites["Start"], ct_sites["End"])
    )

    # Mark which CT sites overlap with positive set
    ct_sites["is_positive"] = [k in pos_keys for k in ct_keys]
    n_overlap = ct_sites["is_positive"].sum()
    logger.info("CT sites overlapping with positive set: %d", n_overlap)

    # Negative controls = CT sites NOT in the positive set
    negatives = ct_sites[~ct_sites["is_positive"]].copy()
    negatives = negatives.drop(columns=["is_positive"])

    # Add site IDs
    negatives.insert(0, "site_id", [f"NEG_{i:04d}" for i in range(len(negatives))])

    # Standardize column names to match unified table
    negatives = negatives.rename(columns={
        "Chr": "chr", "Start": "start", "End": "end",
        "Strand": "strand", "Mismatch": "mismatch",
        "Was_Sites_Dropped_By_the_Filter": "was_filtered",
        "Genomic_Category": "genomic_category",
    })

    neg_path = PROCESSED_DIR / "negative_controls_ct.csv"
    negatives.to_csv(neg_path, index=False)
    logger.info("Negative controls: %d sites -> %s", len(negatives), neg_path)

    # Build combined positive/negative table for ML
    # Core columns from positives
    pos_core = pos_df[["site_id", "chr", "start", "end", "genomic_category"]].copy()
    pos_core["label"] = 1
    pos_core["source"] = "positive_editing"

    # Core columns from negatives
    neg_core = negatives[["site_id", "chr", "start", "end", "genomic_category"]].copy()
    neg_core["label"] = 0
    neg_core["source"] = "negative_ct_mismatch"

    combined = pd.concat([pos_core, neg_core], ignore_index=True)
    combined_path = PROCESSED_DIR / "positive_negative_combined.csv"
    combined.to_csv(combined_path, index=False)
    logger.info(
        "Combined dataset: %d sites (%d pos, %d neg) -> %s",
        len(combined), len(pos_core), len(neg_core), combined_path
    )

    # Summary statistics
    logger.info("\n=== Summary ===")
    logger.info("Positive (true editing): %d", len(pos_core))
    logger.info("Negative (CT mismatch): %d", len(neg_core))
    logger.info("  - Dropped by filter: %d", (negatives["was_filtered"] == True).sum())
    logger.info("  - Not dropped: %d", (negatives["was_filtered"] == False).sum())
    logger.info("Genomic categories (negatives):")
    for cat, count in negatives["genomic_category"].value_counts().items():
        logger.info("  %s: %d", cat, count)

    return combined


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    extract_negative_controls()


if __name__ == "__main__":
    main()
