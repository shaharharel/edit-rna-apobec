"""Parse all 636 Levanon/Advisor sites into enzyme categories.

Reads C2TFinalSites.DB.xlsx T1 sheet and produces:
1. levanon_all_categories.csv — 636 rows with enzyme_category, tissue rates, metadata
2. levanon_tissue_rates.csv — 636 × 54 tissue editing rate matrix (rates as percentages)

The 636 sites break into 5 categories:
  APOBEC3A Only: 120 (already in A3A pipeline as C2U_NNNN)
  APOBEC3G Only: 60
  Both (A3A+A3G): 178
  Neither: 206
  Unknown (NaN): 72

Existing site_ids (C2U_0000..C2U_0635) from parse_advisor_excel.py are preserved.
Sequences already exist in data/processed/site_sequences.json for all 636 sites.

Usage:
    conda run -n quris python scripts/multi_enzyme/parse_levanon_all_categories.py
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

EXCEL_PATH = PROJECT_ROOT / "data" / "raw" / "C2TFinalSites.DB.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "multi_enzyme"

# Tissue columns in T1 (54 GTEx tissues — all columns after conservation)
# These columns contain data in format "edited_reads;total_reads;rate(%)"
TISSUE_START_COL = "Adipose Subcutaneous"

# Map Excel enzyme labels to our categories
ENZYME_MAP = {
    "APOBEC3A Only": "A3A",
    "APOBEC3G Only": "A3G",
    "Both": "A3A_A3G",
    "Neither": "Neither",
}


def parse_tissue_rate(val) -> float:
    """Parse tissue rate from 'edited;total;rate(%)' format.

    Returns rate as percentage (0-100), or NaN if unparseable/no coverage.
    """
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if not s or s in ("NoCoverage", "InsufficientCoverage", "NoCoverage "):
        return np.nan
    parts = s.split(";")
    if len(parts) >= 3:
        try:
            return float(parts[2])
        except (ValueError, IndexError):
            return np.nan
    return np.nan


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not EXCEL_PATH.exists():
        logger.error("Input file not found: %s", EXCEL_PATH)
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Read T1 sheet
    logger.info("Reading T1 sheet from %s", EXCEL_PATH.name)
    t1 = pd.read_excel(EXCEL_PATH, sheet_name="T1-GTEx Editing & Conservation", header=1)
    logger.info("T1 shape: %d rows × %d cols", *t1.shape)

    # Assign site_ids matching existing convention
    t1["site_id"] = [f"C2U_{i:04d}" for i in range(len(t1))]

    # Map enzyme categories
    enzyme_col = "Affecting Over Expressed APOBEC"
    t1["enzyme_category"] = t1[enzyme_col].map(ENZYME_MAP).fillna("Unknown")

    # Log category counts
    logger.info("Enzyme categories:")
    for cat, count in t1["enzyme_category"].value_counts().items():
        logger.info("  %s: %d", cat, count)

    # Identify tissue columns (all columns from TISSUE_START_COL onward)
    col_list = list(t1.columns)
    tissue_start_idx = col_list.index(TISSUE_START_COL)
    tissue_cols = col_list[tissue_start_idx:]
    # Filter to only actual tissue columns (exclude any non-tissue cols that snuck in)
    tissue_cols = [c for c in tissue_cols if c not in [
        "site_id", "enzyme_category", enzyme_col
    ]]
    logger.info("Found %d tissue columns", len(tissue_cols))

    # =========================================================================
    # Output 1: levanon_all_categories.csv — metadata + mean editing rate
    # =========================================================================
    meta_cols = [
        "site_id", "Chr", "Start", "End", "enzyme_category",
        "Genomic Category", "Gene (RefSeq)", "mRNA location (RefSeq)",
        "Exonic Function ", "Edited In # Tissues", "Tissue Classification",
        "Max GTEx Editing Rate", "Mean GTEx Editing Rate", "GTEx Editing Rate SD",
    ]
    # Only keep columns that exist
    meta_cols = [c for c in meta_cols if c in t1.columns]
    meta_df = t1[meta_cols].copy()

    # Clean column names
    meta_df.columns = [
        c.strip().lower().replace(" ", "_").replace("(", "").replace(")", "")
        .replace("#", "num").replace("%", "pct").replace("≥", "gte")
        for c in meta_df.columns
    ]

    # Load sequences for motif analysis
    import json
    seq_path = PROJECT_ROOT / "data" / "processed" / "site_sequences.json"
    if seq_path.exists():
        with open(seq_path) as f:
            seqs = json.load(f)
        logger.info("Loaded %d sequences from %s", len(seqs), seq_path.name)
    else:
        logger.warning("No sequences file found at %s", seq_path)
        seqs = {}

    # Get strand from existing splits_expanded.csv (already correctly determined
    # during the original data pipeline — strand inference from genome is unreliable
    # because the Start coordinate doesn't always point to the C/G base directly)
    splits_path = PROJECT_ROOT / "data" / "processed" / "splits_expanded.csv"
    if splits_path.exists():
        splits_df = pd.read_csv(splits_path)
        advisor_strands = splits_df[splits_df["dataset_source"] == "advisor_c2t"][
            ["site_id", "strand"]
        ].set_index("site_id")
        meta_df["strand"] = meta_df["site_id"].map(advisor_strands["strand"])
        n_plus = (meta_df["strand"] == "+").sum()
        n_minus = (meta_df["strand"] == "-").sum()
        n_missing = meta_df["strand"].isna().sum()
        logger.info("Strand from splits_expanded: + = %d, - = %d, missing = %d",
                     n_plus, n_minus, n_missing)
    else:
        logger.warning("splits_expanded.csv not found, strand will be missing")
        meta_df["strand"] = np.nan

    # Add dataset_source column for downstream compatibility
    meta_df["dataset_source"] = "levanon_advisor"
    meta_df["coordinate_system"] = "hg38"
    meta_df["is_edited"] = 1

    # Add editing_rate (mean GTEx rate as fraction 0-1, dividing percentage by 100)
    if "mean_gtex_editing_rate" in meta_df.columns:
        meta_df["editing_rate"] = meta_df["mean_gtex_editing_rate"] / 100.0
    else:
        meta_df["editing_rate"] = np.nan

    out_meta = OUTPUT_DIR / "levanon_all_categories.csv"
    meta_df.to_csv(out_meta, index=False)
    logger.info("Wrote %s (%d rows)", out_meta.name, len(meta_df))

    # =========================================================================
    # Output 2: levanon_tissue_rates.csv — 636 × 54 tissue editing rate matrix
    # =========================================================================
    tissue_rates = pd.DataFrame({"site_id": t1["site_id"]})
    tissue_rates["enzyme_category"] = t1["enzyme_category"].values

    for col in tissue_cols:
        clean_name = col.strip().lower().replace(" ", "_")
        tissue_rates[clean_name] = t1[col].apply(parse_tissue_rate)

    out_tissue = OUTPUT_DIR / "levanon_tissue_rates.csv"
    tissue_rates.to_csv(out_tissue, index=False)
    logger.info("Wrote %s (%d rows × %d tissue cols)",
                out_tissue.name, len(tissue_rates), len(tissue_cols))

    # =========================================================================
    # Summary statistics
    # =========================================================================
    logger.info("\n=== Summary ===")
    for cat in ["A3A", "A3G", "A3A_A3G", "Neither", "Unknown"]:
        subset = meta_df[meta_df["enzyme_category"] == cat]
        if len(subset) == 0:
            continue
        # Compute motif from sequences
        tc_count = 0
        cc_count = 0
        for sid in subset["site_id"]:
            seq = seqs.get(sid, "N" * 201)
            if len(seq) >= 102:
                up = seq[99].upper()
                if up in ("U", "T"):
                    tc_count += 1
                elif up == "C":
                    cc_count += 1
        tc_frac = tc_count / len(subset) * 100 if len(subset) > 0 else 0
        cc_frac = cc_count / len(subset) * 100 if len(subset) > 0 else 0

        rate = subset["editing_rate"].dropna()
        logger.info(
            "%s: n=%d, TC=%.1f%%, CC=%.1f%%, mean_rate=%.3f, strand(+/-)=%d/%d",
            cat, len(subset), tc_frac, cc_frac,
            rate.mean() if len(rate) > 0 else 0,
            len(subset[subset["strand"] == "+"]),
            len(subset[subset["strand"] == "-"]),
        )


if __name__ == "__main__":
    main()
