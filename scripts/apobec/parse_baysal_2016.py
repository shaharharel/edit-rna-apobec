"""Parse Sharma, Patnaik, Kemer & Baysal supplementary data.

Paper: "Transient overexpression of exogenous APOBEC3A causes C-to-U
       RNA editing of thousands of genes" (RNA Biology 2017)
DOI: 10.1080/15476286.2016.1184387

Identified 4,200+ APOBEC3A-mediated C-to-U RNA editing sites by
transient overexpression of APOBEC3A in HEK293T cells.
These sites are a subset of Asaoka 2019 (which found 4,933 A3A sites).

IMPORTANT: Coordinates in the supplementary data are in GRCh38 (hg38).
We use pyliftover to convert to hg19 coordinates for consistency with
the rest of the project.

Output:
  - data/processed/published/baysal_2016_editing_sites.csv

Usage:
    python scripts/apobec/parse_baysal_2016.py --input <path_to_supplementary_file>
    python scripts/apobec/parse_baysal_2016.py --download
"""

import argparse
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "published" / "baysal_2016"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "published"

# Expected supplementary file name
DEFAULT_RAW_PATH = RAW_DIR / "baysal_2016_supp_data.xlsx"


def download_supplementary(output_dir: Path) -> Path:
    """Attempt to download supplementary data from the journal.

    The supplementary materials for DOI 10.1080/15476286.2016.1184387
    are hosted on Taylor & Francis Online.

    Returns:
        Path to downloaded file, or raises RuntimeError on failure.
    """
    import urllib.request

    url = (
        "https://www.tandfonline.com/action/downloadSupplement?"
        "doi=10.1080/15476286.2016.1184387&file=krnb-13-07-1184387-s002.xlsx"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "baysal_2016_supp_data.xlsx"

    if out_path.exists():
        logger.info("Supplementary file already exists: %s", out_path)
        return out_path

    logger.info("Downloading supplementary data from: %s", url)
    try:
        urllib.request.urlretrieve(url, out_path)
        logger.info("Downloaded to: %s", out_path)
        return out_path
    except Exception as e:
        raise RuntimeError(
            f"Failed to download supplementary data: {e}\n"
            f"Please download manually from the journal website and provide "
            f"the path via --input flag.\n"
            f"DOI: 10.1080/15476286.2016.1184387"
        )


def liftover_hg38_to_hg19(chrom: str, pos: int, converter) -> tuple:
    """Convert a single GRCh38 coordinate to hg19 using pyliftover.

    Args:
        chrom: Chromosome (e.g., "chr1").
        pos: 0-based position in GRCh38.
        converter: pyliftover.LiftOver instance.

    Returns:
        (hg19_chrom, hg19_pos) or (None, None) if liftover fails.
    """
    result = converter.convert_coordinate(chrom, pos)
    if result and len(result) > 0:
        hg19_chrom, hg19_pos, hg19_strand, _ = result[0]
        return hg19_chrom, hg19_pos
    return None, None


def parse_editing_sites(path: Path, converter) -> pd.DataFrame:
    """Parse Baysal 2016 supplementary table of C-to-U editing sites.

    The supplementary data contains columns for chromosome, position,
    strand, gene name, editing rates, and functional annotations.
    Coordinates are in GRCh38 and are lifted over to hg19.

    Args:
        path: Path to the supplementary Excel file.
        converter: pyliftover.LiftOver instance for hg38->hg19.
    """
    # Try different sheet names (supplementary files vary)
    df = None
    for sheet_name in ["Data", 0, "Sheet1", "Table S1", "Supplementary Table", "C-to-U sites"]:
        try:
            df = pd.read_excel(path, sheet_name=sheet_name)
            if len(df) > 0:
                logger.info("Loaded %d rows from sheet '%s'", len(df), sheet_name)
                break
        except Exception:
            continue

    if df is None or len(df) == 0:
        logger.error("Could not read any data from %s", path)
        return pd.DataFrame()

    # Log available columns for debugging
    logger.info("Columns found: %s", list(df.columns))

    # Standardize column names (handle various naming conventions)
    col_map = {}
    for col in df.columns:
        col_lower = str(col).lower().strip()
        if "chrom" in col_lower and "chr" not in col_map:
            col_map["chr"] = col
        elif col_lower in ("chr", "chromosome"):
            col_map["chr"] = col
        elif col_lower in ("position", "pos", "start", "genomic_position"):
            col_map["pos"] = col
        elif col_lower in ("strand",):
            col_map["strand"] = col
        elif col_lower in ("gene", "gene_name", "genename", "symbol", "gene symbol"):
            col_map["gene"] = col
        elif col_lower in ("region", "feature", "generegion", "gene_region", "annotation"):
            col_map["feature"] = col
        elif "edit" in col_lower and "rate" in col_lower:
            col_map["editing_rate"] = col
        elif "averagevariationlevel" in col_lower:
            col_map["editing_rate"] = col
        elif col_lower in ("ref", "reference", "ref_base", "reference base"):
            col_map["ref"] = col
        elif col_lower in ("alt", "alternative", "alt_base", "altered base"):
            col_map["alt"] = col
        elif col_lower in ("variation",) and "alt" not in col_map:
            col_map["alt"] = col

    logger.info("Column mapping: %s", col_map)

    # Validate required columns
    if "chr" not in col_map or "pos" not in col_map:
        logger.error(
            "Required columns (chromosome, position) not found. "
            "Available columns: %s", list(df.columns)
        )
        return pd.DataFrame()

    # Extract chromosome with chr prefix
    chr_col = df[col_map["chr"]].astype(str)
    if not chr_col.iloc[0].startswith("chr"):
        chr_col = "chr" + chr_col

    # Original positions (1-based in supplementary data -> 0-based)
    positions_1based = df[col_map["pos"]].astype(int)
    positions_0based_hg38 = positions_1based - 1

    # LiftOver from GRCh38 to hg19
    logger.info("Performing LiftOver from GRCh38 to hg19 for %d sites...", len(df))
    hg19_chroms = []
    hg19_positions = []
    liftover_failed = 0

    for i, (chrom, pos) in enumerate(zip(chr_col, positions_0based_hg38)):
        hg19_chr, hg19_pos = liftover_hg38_to_hg19(chrom, pos, converter)
        hg19_chroms.append(hg19_chr)
        hg19_positions.append(hg19_pos)
        if hg19_chr is None:
            liftover_failed += 1

        if (i + 1) % 1000 == 0:
            logger.info("  LiftOver progress: %d/%d (failed: %d)",
                        i + 1, len(df), liftover_failed)

    logger.info("LiftOver complete: %d succeeded, %d failed (%.1f%%)",
                len(df) - liftover_failed, liftover_failed,
                100 * liftover_failed / max(1, len(df)))

    # Build result dataframe
    result = pd.DataFrame()
    result["hg19_chr"] = hg19_chroms
    result["hg19_pos"] = hg19_positions

    # Copy original data columns
    result["hg38_chr"] = chr_col.values
    result["hg38_pos"] = positions_0based_hg38.values

    # Strand
    if "strand" in col_map:
        result["strand"] = df[col_map["strand"]].values
    else:
        # Infer strand from ref base if available
        if "ref" in col_map:
            ref_bases = df[col_map["ref"]].astype(str).str.upper()
            result["strand"] = ref_bases.map({"C": "+", "G": "-"}).fillna("+")
        else:
            result["strand"] = "+"

    # Gene
    if "gene" in col_map:
        result["gene"] = df[col_map["gene"]].values
    else:
        result["gene"] = ""

    # Feature / region
    if "feature" in col_map:
        result["feature"] = df[col_map["feature"]].values
    else:
        result["feature"] = "unknown"

    # Editing rate
    if "editing_rate" in col_map:
        result["editing_rate"] = pd.to_numeric(
            df[col_map["editing_rate"]], errors="coerce"
        ).values
    else:
        result["editing_rate"] = np.nan

    # Ref/alt bases
    if "ref" in col_map:
        result["ref"] = df[col_map["ref"]].values
    if "alt" in col_map:
        result["alt"] = df[col_map["alt"]].values

    # Filter out failed liftover
    valid_mask = result["hg19_chr"].notna()
    n_dropped = (~valid_mask).sum()
    result = result[valid_mask].copy()
    logger.info("Dropped %d sites with failed LiftOver", n_dropped)

    # Build standardized output
    output = pd.DataFrame()
    output["chr"] = result["hg19_chr"]
    output["start"] = result["hg19_pos"].astype(int)
    output["end"] = output["start"] + 1
    output["strand"] = result["strand"]

    # Site ID
    output["site_id"] = output.apply(
        lambda r: f"baysal_{r['chr']}_{r['start']}", axis=1
    )

    output["gene"] = result["gene"]
    output["feature"] = result["feature"]
    output["edit_type"] = "C-to-U"
    output["is_edited"] = 1
    output["dataset_source"] = "baysal_2016"
    output["editing_rate"] = result["editing_rate"]

    # Preserve original hg38 coordinates for reference
    output["hg38_chr"] = result["hg38_chr"]
    output["hg38_pos"] = result["hg38_pos"]

    # Deduplicate by hg19 coordinate
    n_before = len(output)
    output = output.drop_duplicates(subset=["chr", "start"], keep="first")
    n_dedup = n_before - len(output)
    if n_dedup > 0:
        logger.info("Removed %d duplicate sites after LiftOver (same hg19 coordinate)",
                    n_dedup)

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Parse Baysal et al. 2016 C-to-U editing sites"
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Path to supplementary data file (Excel). "
             "If not provided, uses default path in data/raw/published/baysal_2016/"
    )
    parser.add_argument(
        "--download", action="store_true",
        help="Attempt to download supplementary data from journal website"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Determine input file
    if args.input:
        input_path = Path(args.input)
    elif args.download:
        input_path = download_supplementary(RAW_DIR)
    else:
        input_path = DEFAULT_RAW_PATH

    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        logger.error(
            "Please provide the path to the Baysal 2016 supplementary data "
            "via --input or use --download to attempt automatic download."
        )
        sys.exit(1)

    # Initialize LiftOver converter (hg38 -> hg19)
    try:
        from pyliftover import LiftOver
    except ImportError:
        logger.error(
            "pyliftover is required for coordinate conversion. "
            "Install with: pip install pyliftover"
        )
        sys.exit(1)

    logger.info("Initializing LiftOver (hg38 -> hg19)...")
    converter = LiftOver("hg38", "hg19")

    # Parse editing sites
    sites = parse_editing_sites(input_path, converter)

    if len(sites) == 0:
        logger.error("No editing sites parsed!")
        sys.exit(1)

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "baysal_2016_editing_sites.csv"
    sites.to_csv(out_path, index=False)
    logger.info("Saved %d C-to-U editing sites to %s", len(sites), out_path)

    # Summary statistics
    logger.info("\n=== Baysal 2016 Summary ===")
    logger.info("Total C-to-U sites (after LiftOver + dedup): %d", len(sites))
    logger.info("Strand distribution:")
    for strand, count in sites["strand"].value_counts().items():
        logger.info("  %s: %d", strand, count)
    logger.info("Region distribution:")
    for region, count in sites["feature"].value_counts().head(10).items():
        logger.info("  %s: %d", region, count)
    logger.info("Unique genes: %d", sites["gene"].nunique())

    # Editing rate statistics (if available)
    rate_valid = sites["editing_rate"].dropna()
    if len(rate_valid) > 0:
        logger.info("Editing rate statistics:")
        logger.info("  Mean: %.3f", rate_valid.mean())
        logger.info("  Median: %.3f", rate_valid.median())
        logger.info("  Range: [%.3f, %.3f]", rate_valid.min(), rate_valid.max())

    # Chromosome distribution
    logger.info("Chromosome distribution (top 10):")
    for chrom, count in sites["chr"].value_counts().head(10).items():
        logger.info("  %s: %d", chrom, count)

    # Check overlap with existing datasets
    existing_path = PROJECT_ROOT / "data" / "processed" / "all_datasets_combined.csv"
    if existing_path.exists():
        existing = pd.read_csv(existing_path)
        existing_coords = set(zip(existing["chr"], existing["start"]))
        overlap = sites[
            sites.apply(
                lambda r: (r["chr"], r["start"]) in existing_coords, axis=1
            )
        ]
        logger.info("Overlap with existing datasets: %d sites", len(overlap))
        if len(overlap) > 0:
            logger.info("  Overlapping datasets:")
            overlap_existing = existing[
                existing.apply(
                    lambda r: (r["chr"], r["start"]) in set(
                        zip(overlap["chr"], overlap["start"])
                    ), axis=1
                )
            ]
            for ds, count in overlap_existing["dataset_source"].value_counts().items():
                logger.info("    %s: %d", ds, count)


if __name__ == "__main__":
    main()
