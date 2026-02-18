"""Build unified multi-dataset table combining all APOBEC editing datasets.

Combines:
1. Levanon dataset (636 sites from advisor's C2TFinalSites.DB.xlsx)
2. Alqassim et al. 2021 (209 sites from monocyte differentiation)
3. Asaoka et al. 2019 (5,208 editing sites from 293T cells)
4. Sharma et al. 2015 (333 C-to-U sites from hypoxia/macrophage)

Output: data/processed/all_datasets_combined.csv

Usage:
    python scripts/apobec/build_unified_dataset.py
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Input paths
ADVISOR_CSV = PROJECT_ROOT / "data" / "processed" / "advisor" / "unified_editing_sites.csv"
ALQASSIM_CSV = PROJECT_ROOT / "data" / "processed" / "published" / "alqassim_2021_editing_sites.csv"
ASAOKA_CSV = PROJECT_ROOT / "data" / "processed" / "published" / "asaoka_2019_editing_sites.csv"
SHARMA_CSV = PROJECT_ROOT / "data" / "processed" / "published" / "sharma_2015_editing_sites.csv"

# Output
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "all_datasets_combined.csv"


def load_advisor_dataset(path: Path) -> pd.DataFrame:
    """Load and standardize the advisor's dataset."""
    df = pd.read_csv(path)
    logger.info("Loaded advisor dataset: %d sites", len(df))

    result = pd.DataFrame()
    result["site_id"] = df["site_id"]
    result["chr"] = df["chr"]
    result["start"] = df["start"]
    result["end"] = df["end"]
    result["strand"] = "+"  # Default; will be updated from supp TX
    result["gene"] = df["gene_refseq"]
    result["edit_type"] = "C-to-U"
    result["is_edited"] = 1
    result["dataset_source"] = "advisor_c2t"

    # Editing rates (max across GTEx tissues)
    if "max_gtex_rate" in df.columns:
        result["editing_rate"] = df["max_gtex_rate"]
    elif "max_gtex_editing_rate" in df.columns:
        result["editing_rate"] = df["max_gtex_editing_rate"]
    else:
        result["editing_rate"] = np.nan

    # Feature type
    if "exonic_function" in df.columns:
        result["feature"] = df["exonic_function"]
    else:
        result["feature"] = "unknown"

    # Update strand from supp TX if available
    supptx_path = PROJECT_ROOT / "data" / "processed" / "advisor" / "supp_tx_all_non_ag_mm_sites.csv"
    if supptx_path.exists():
        mm = pd.read_csv(supptx_path)
        ct = mm[mm["Mismatch"] == "CT"]
        strand_map = dict(zip(zip(ct["Chr"], ct["Start"]), ct["Strand"]))
        result["strand"] = result.apply(
            lambda r: strand_map.get((r["chr"], r["start"]), "+"), axis=1
        )
        logger.info("  Updated strand info from supp TX (%d mapped)", len(strand_map))

    return result


def load_alqassim_dataset(path: Path) -> pd.DataFrame:
    """Load and standardize the Alqassim 2021 dataset."""
    df = pd.read_csv(path)
    logger.info("Loaded Alqassim 2021 dataset: %d sites", len(df))

    result = pd.DataFrame()
    result["site_id"] = df["site_id"]
    result["chr"] = df["chr"]
    result["start"] = df["start"]  # Already 0-based from parsing
    result["end"] = df["end"]
    result["strand"] = df["strand"]
    result["gene"] = df["gene"]
    result["edit_type"] = "C-to-U"  # All are APOBEC C-to-U (CtT or GtA on opposite strand)
    result["is_edited"] = 1
    result["dataset_source"] = "alqassim_2021"
    result["editing_rate"] = df["rate_stemcell_avg"]  # Use stem cell rate as primary
    result["feature"] = df["feature"]

    return result


def load_asaoka_dataset(path: Path) -> pd.DataFrame:
    """Load and standardize the Asaoka 2019 dataset."""
    df = pd.read_csv(path)
    logger.info("Loaded Asaoka 2019 dataset: %d sites", len(df))

    result = pd.DataFrame()
    result["site_id"] = df["site_id"]
    result["chr"] = df["chr"]
    result["start"] = df["start"]
    result["end"] = df["end"]
    result["strand"] = df["strand"]
    result["gene"] = df["gene"]
    result["edit_type"] = "C-to-U"
    result["is_edited"] = 1
    result["dataset_source"] = "asaoka_2019"
    result["editing_rate"] = df["editing_rate"]
    result["feature"] = df["feature"]

    return result


def load_sharma_dataset(path: Path) -> pd.DataFrame:
    """Load and standardize the Sharma 2015 dataset.

    Deduplicates sites present in both conditions (hypoxia + macrophage),
    keeping the entry with higher editing rate.
    """
    df = pd.read_csv(path)
    logger.info("Loaded Sharma 2015 dataset: %d entries", len(df))

    # Deduplicate by coordinate, keeping highest editing rate
    df = df.sort_values("editing_rate", ascending=False)
    df = df.drop_duplicates(subset=["chr", "start"], keep="first")
    logger.info("  After dedup: %d unique sites", len(df))

    result = pd.DataFrame()
    result["site_id"] = df["site_id"]
    result["chr"] = df["chr"]
    result["start"] = df["start"]
    result["end"] = df["end"]
    result["strand"] = df["strand"]
    result["gene"] = df["gene"]
    result["edit_type"] = "C-to-U"
    result["is_edited"] = 1
    result["dataset_source"] = "sharma_2015"
    result["editing_rate"] = df["editing_rate"]
    result["feature"] = df["feature"]

    return result


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    all_datasets = []

    # 1. Advisor dataset
    if ADVISOR_CSV.exists():
        advisor = load_advisor_dataset(ADVISOR_CSV)
        all_datasets.append(advisor)
        logger.info("  Advisor: %d sites", len(advisor))
    else:
        logger.error("Advisor dataset not found: %s", ADVISOR_CSV)

    # 2. Alqassim 2021
    if ALQASSIM_CSV.exists():
        alqassim = load_alqassim_dataset(ALQASSIM_CSV)
        all_datasets.append(alqassim)
        logger.info("  Alqassim: %d sites", len(alqassim))
    else:
        logger.warning("Alqassim dataset not found: %s. Run parse_alqassim_2021.py first.", ALQASSIM_CSV)

    # 3. Asaoka 2019
    if ASAOKA_CSV.exists():
        asaoka = load_asaoka_dataset(ASAOKA_CSV)
        all_datasets.append(asaoka)
        logger.info("  Asaoka: %d sites", len(asaoka))
    else:
        logger.warning("Asaoka dataset not found: %s. Run parse_asaoka_2019.py first.", ASAOKA_CSV)

    # 4. Sharma 2015
    if SHARMA_CSV.exists():
        sharma = load_sharma_dataset(SHARMA_CSV)
        all_datasets.append(sharma)
        logger.info("  Sharma: %d sites", len(sharma))
    else:
        logger.warning("Sharma dataset not found: %s. Run parse_sharma_2015.py first.", SHARMA_CSV)

    # Combine
    combined = pd.concat(all_datasets, ignore_index=True)

    # Check for coordinate overlaps between real datasets
    real = combined[combined["start"] >= 0].copy()
    real["coord_key"] = real["chr"] + ":" + real["start"].astype(str)
    dup_coords = real[real.duplicated("coord_key", keep=False)]
    if len(dup_coords) > 0:
        n_dup = dup_coords["coord_key"].nunique()
        logger.info("Found %d overlapping coordinates across datasets:", n_dup)
        # Show overlap matrix between datasets
        for coord, group in dup_coords.groupby("coord_key"):
            datasets = sorted(group["dataset_source"].unique())
            if len(datasets) >= 2:
                pass  # Count below
        overlap_pairs = {}
        for _, group in dup_coords.groupby("coord_key"):
            ds_list = sorted(group["dataset_source"].unique())
            key = " & ".join(ds_list)
            overlap_pairs[key] = overlap_pairs.get(key, 0) + 1
        for pair, count in sorted(overlap_pairs.items(), key=lambda x: -x[1]):
            logger.info("  %s: %d sites", pair, count)

        # Mark overlapping sites
        combined["in_multiple_datasets"] = False
        dup_set = set(dup_coords["coord_key"])
        mask = combined.apply(
            lambda r: f"{r['chr']}:{r['start']}" in dup_set if r["start"] >= 0 else False,
            axis=1
        )
        combined.loc[mask, "in_multiple_datasets"] = True
    else:
        combined["in_multiple_datasets"] = False

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_PATH, index=False)
    logger.info("Saved unified dataset to %s", OUTPUT_PATH)

    # Summary
    logger.info("\n=== Multi-Dataset Summary ===")
    logger.info("Total entries: %d", len(combined))
    logger.info("By dataset:")
    for ds, group in combined.groupby("dataset_source"):
        n_real = (group["start"] >= 0).sum()
        logger.info("  %s: %d entries (%d with coordinates)", ds, len(group), n_real)
    logger.info("Overlapping sites: %d", combined["in_multiple_datasets"].sum())
    logger.info("Unique genes: %d", combined[combined["gene"].notna() & (combined["gene"] != "")]["gene"].nunique())


if __name__ == "__main__":
    main()
