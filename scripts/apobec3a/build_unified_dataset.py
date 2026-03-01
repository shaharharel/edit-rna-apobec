"""Build unified multi-dataset table combining all APOBEC editing datasets.

All coordinates are standardized to GRCh38 (hg38).

Combines:
1. Levanon dataset (636 sites from advisor's C2TFinalSites.DB.xlsx) — native hg38
2. Alqassim et al. 2021 (209 sites from monocyte differentiation) — native hg38
3. Asaoka et al. 2019 (5,208 editing sites from 293T cells) — native hg38
4. Sharma et al. 2015 (333 C-to-U sites from hypoxia/macrophage) — native hg19, LiftOver to hg38
5. Baysal et al. 2016 (4,373 C-to-U sites) — native hg38 (stored as hg38_pos in parsed CSV)

Output: data/processed/all_datasets_combined.csv

Usage:
    python scripts/apobec3a/build_unified_dataset.py
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from pyliftover import LiftOver

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Input paths
ADVISOR_CSV = PROJECT_ROOT / "data" / "processed" / "advisor" / "unified_editing_sites.csv"
ALQASSIM_CSV = PROJECT_ROOT / "data" / "processed" / "published" / "alqassim_2021_editing_sites.csv"
ASAOKA_CSV = PROJECT_ROOT / "data" / "processed" / "published" / "asaoka_2019_editing_sites.csv"
SHARMA_CSV = PROJECT_ROOT / "data" / "processed" / "published" / "sharma_2015_editing_sites.csv"
BAYSAL_CSV = PROJECT_ROOT / "data" / "processed" / "published" / "baysal_2016_editing_sites.csv"

# Output
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "all_datasets_combined.csv"


def load_advisor_dataset(path: Path) -> pd.DataFrame:
    """Load the advisor's (Levanon) dataset — native hg38, no LiftOver needed.

    Coordinates in C2TFinalSites.DB.xlsx are in GRCh38 (hg38), confirmed by
    cross-referencing gene positions (e.g., GLUD2 on chrX: 121,048,265 matches
    hg38 gene boundaries, not hg19).
    """
    df = pd.read_csv(path)
    logger.info("Loaded advisor dataset: %d sites (native hg38)", len(df))

    # Get strand from supp TX (uses same hg38 coords)
    strand_map = {}
    supptx_path = PROJECT_ROOT / "data" / "processed" / "advisor" / "supp_tx_all_non_ag_mm_sites.csv"
    if supptx_path.exists():
        mm = pd.read_csv(supptx_path)
        ct = mm[mm["Mismatch"] == "CT"]
        strand_map = dict(zip(zip(ct["Chr"], ct["Start"]), ct["Strand"]))
        logger.info("  Loaded strand info from supp TX (%d CT sites)", len(strand_map))

    result = pd.DataFrame()
    result["site_id"] = df["site_id"].values
    result["chr"] = df["chr"].values
    result["start"] = df["start"].values  # Already 0-based hg38
    result["end"] = df["end"].values
    result["strand"] = [strand_map.get((row["chr"], row["start"]), "+")
                        for _, row in df.iterrows()]
    result["gene"] = df["gene_refseq"].values
    result["edit_type"] = "C-to-U"
    result["is_edited"] = 1
    result["dataset_source"] = "advisor_c2t"

    # Editing rates (max across GTEx tissues)
    # NOTE: These rates are on a 0-100 percentage scale from the GTEx data.
    # Other datasets (Alqassim, Sharma, Baysal) use 0-1 fraction scale.
    # The downstream normalize_editing_rates.py script adds an
    # editing_rate_normalized column that puts everything on 0-1 scale.
    if "max_gtex_rate" in df.columns:
        result["editing_rate"] = df["max_gtex_rate"]
    elif "max_gtex_editing_rate" in df.columns:
        result["editing_rate"] = df["max_gtex_editing_rate"]
    else:
        result["editing_rate"] = np.nan

    # Feature type
    if "exonic_function" in df.columns:
        result["feature"] = df["exonic_function"].values
    else:
        result["feature"] = "unknown"

    return result


def load_alqassim_dataset(path: Path) -> pd.DataFrame:
    """Load the Alqassim 2021 dataset — native hg38, no LiftOver needed."""
    df = pd.read_csv(path)
    logger.info("Loaded Alqassim 2021 dataset: %d sites (native hg38)", len(df))

    result = pd.DataFrame()
    result["site_id"] = df["site_id"].values
    result["chr"] = df["chr"].values
    result["start"] = df["start"].values  # Already 0-based hg38
    result["end"] = df["end"].values
    result["strand"] = df["strand"].values
    result["gene"] = df["gene"].values
    result["edit_type"] = "C-to-U"
    result["is_edited"] = 1
    result["dataset_source"] = "alqassim_2021"
    result["editing_rate"] = df["rate_stemcell_avg"].values
    result["feature"] = df["feature"].values

    return result


def load_asaoka_dataset(path: Path) -> pd.DataFrame:
    """Load the Asaoka 2019 dataset — native hg38, no LiftOver needed."""
    df = pd.read_csv(path)
    logger.info("Loaded Asaoka 2019 dataset: %d sites (native hg38)", len(df))

    result = pd.DataFrame()
    result["site_id"] = df["site_id"].values
    result["chr"] = df["chr"].values
    result["start"] = df["start"].values  # Already 0-based hg38
    result["end"] = df["end"].values
    result["strand"] = df["strand"].values
    result["gene"] = df["gene"].values
    result["edit_type"] = "C-to-U"
    result["is_edited"] = 1
    result["dataset_source"] = "asaoka_2019"
    result["editing_rate"] = df["editing_rate"].values
    result["feature"] = df["feature"].values

    return result


def load_sharma_dataset(path: Path) -> pd.DataFrame:
    """Load the Sharma 2015 dataset — native hg19, LiftOver to hg38.

    Deduplicates sites present in both conditions (hypoxia + macrophage),
    keeping the entry with higher editing rate.
    """
    df = pd.read_csv(path)
    logger.info("Loaded Sharma 2015 dataset: %d entries (native hg19)", len(df))

    # Deduplicate by coordinate, keeping highest editing rate
    df = df.sort_values("editing_rate", ascending=False)
    df = df.drop_duplicates(subset=["chr", "start"], keep="first")
    logger.info("  After dedup: %d unique sites", len(df))

    # LiftOver from hg19 to hg38
    logger.info("  Performing LiftOver from hg19 to GRCh38...")
    converter = LiftOver("hg19", "hg38")
    hg38_starts = []
    liftover_failed = 0

    for _, row in df.iterrows():
        chrom = row["chr"]
        pos_hg19 = int(row["start"])
        result_lo = converter.convert_coordinate(chrom, pos_hg19)
        if result_lo and len(result_lo) > 0:
            _, hg38_pos, _, _ = result_lo[0]
            hg38_starts.append(int(hg38_pos))
        else:
            hg38_starts.append(None)
            liftover_failed += 1

    logger.info("  LiftOver: %d succeeded, %d failed",
                len(df) - liftover_failed, liftover_failed)

    result = pd.DataFrame()
    result["site_id"] = df["site_id"].values
    result["chr"] = df["chr"].values
    result["start"] = hg38_starts
    result["end"] = [s + 1 if s is not None else None for s in hg38_starts]
    result["strand"] = df["strand"].values
    result["gene"] = df["gene"].values
    result["edit_type"] = "C-to-U"
    result["is_edited"] = 1
    result["dataset_source"] = "sharma_2015"
    result["editing_rate"] = df["editing_rate"].values
    result["feature"] = df["feature"].values

    # Drop failed LiftOver sites
    valid_mask = result["start"].notna()
    n_dropped = (~valid_mask).sum()
    if n_dropped > 0:
        logger.warning("  Dropped %d sites with failed LiftOver", n_dropped)
    result = result[valid_mask].copy()
    result["start"] = result["start"].astype(int)
    result["end"] = result["end"].astype(int)

    return result


def load_baysal_dataset(path: Path) -> pd.DataFrame:
    """Load the Baysal 2016 dataset — use original hg38 coordinates.

    The parsed CSV has hg19 coordinates (via LiftOver in parse_baysal_2016.py)
    and the original hg38 coordinates stored in hg38_chr/hg38_pos columns.
    We use the hg38 originals directly.
    """
    df = pd.read_csv(path)
    logger.info("Loaded Baysal 2016 dataset: %d sites (restoring native hg38)", len(df))

    result = pd.DataFrame()
    result["site_id"] = df["site_id"].values
    result["chr"] = df["hg38_chr"].values  # Use original hg38 chromosome
    result["start"] = df["hg38_pos"].astype(int).values  # Use original hg38 position
    result["end"] = result["start"] + 1
    result["strand"] = df["strand"].values
    result["gene"] = df["gene"].values
    result["edit_type"] = "C-to-U"
    result["is_edited"] = 1
    result["dataset_source"] = "baysal_2016"
    result["editing_rate"] = df["editing_rate"].values
    result["feature"] = df["feature"].values

    return result


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    all_datasets = []

    # 1. Advisor dataset (native hg38)
    if ADVISOR_CSV.exists():
        advisor = load_advisor_dataset(ADVISOR_CSV)
        all_datasets.append(advisor)
        logger.info("  Advisor: %d sites", len(advisor))
    else:
        logger.error("Advisor dataset not found: %s", ADVISOR_CSV)

    # 2. Alqassim 2021 (native hg38)
    if ALQASSIM_CSV.exists():
        alqassim = load_alqassim_dataset(ALQASSIM_CSV)
        all_datasets.append(alqassim)
        logger.info("  Alqassim: %d sites", len(alqassim))
    else:
        logger.warning("Alqassim dataset not found: %s. Run parse_alqassim_2021.py first.", ALQASSIM_CSV)

    # 3. Asaoka 2019 (native hg38)
    if ASAOKA_CSV.exists():
        asaoka = load_asaoka_dataset(ASAOKA_CSV)
        all_datasets.append(asaoka)
        logger.info("  Asaoka: %d sites", len(asaoka))
    else:
        logger.warning("Asaoka dataset not found: %s. Run parse_asaoka_2019.py first.", ASAOKA_CSV)

    # 4. Sharma 2015 (hg19 → LiftOver to hg38)
    if SHARMA_CSV.exists():
        sharma = load_sharma_dataset(SHARMA_CSV)
        all_datasets.append(sharma)
        logger.info("  Sharma: %d sites", len(sharma))
    else:
        logger.warning("Sharma dataset not found: %s. Run parse_sharma_2015.py first.", SHARMA_CSV)

    # 5. Baysal 2016 (use original hg38 coords)
    if BAYSAL_CSV.exists():
        baysal = load_baysal_dataset(BAYSAL_CSV)
        all_datasets.append(baysal)
        logger.info("  Baysal: %d sites", len(baysal))
    else:
        logger.warning("Baysal dataset not found: %s. Run parse_baysal_2016.py first.", BAYSAL_CSV)

    # Combine
    combined = pd.concat(all_datasets, ignore_index=True)

    # Check for coordinate overlaps between datasets
    real = combined[combined["start"] >= 0].copy()
    real["coord_key"] = real["chr"] + ":" + real["start"].astype(str)
    dup_coords = real[real.duplicated("coord_key", keep=False)]
    if len(dup_coords) > 0:
        n_dup = dup_coords["coord_key"].nunique()
        logger.info("Found %d overlapping coordinates across datasets:", n_dup)
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
    logger.info("Saved unified dataset to %s (genome build: GRCh38/hg38)", OUTPUT_PATH)

    # Summary
    logger.info("\n=== Multi-Dataset Summary (GRCh38/hg38) ===")
    logger.info("Total entries: %d", len(combined))
    logger.info("By dataset:")
    for ds, group in combined.groupby("dataset_source"):
        n_real = (group["start"] >= 0).sum()
        logger.info("  %s: %d entries (%d with coordinates)", ds, len(group), n_real)
    logger.info("Overlapping sites: %d", combined["in_multiple_datasets"].sum())
    logger.info("Unique genes: %d", combined[combined["gene"].notna() & (combined["gene"] != "")]["gene"].nunique())


if __name__ == "__main__":
    main()
