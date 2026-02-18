"""Parse Asaoka et al. 2019 (originally cited as Pecori 2019) supplementary data.

Paper: "APOBEC3A-mediated RNA editing in breast cancer is associated with
        heightened immune activity and improved survival" (Int J Mol Sci 2019)

Table S1 contains:
  - 5,208 C-to-U editing sites (exonic, UTR, intronic, ncRNA)
  - 198 negative control sites (non-edited Cs in same transcripts)

Positions are 1-based genomic coordinates.
Reference base: C (plus strand) or G (minus strand → C-to-U on mRNA).

Output:
  - data/processed/published/asaoka_2019_editing_sites.csv
  - data/processed/published/asaoka_2019_negatives.csv

Usage:
    python scripts/apobec/parse_asaoka_2019.py
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "published" / "asaoka_2019_table_s1.xls"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "published"


def parse_editing_sites(path: Path) -> pd.DataFrame:
    """Parse the 'Editing sites' sheet."""
    df = pd.read_excel(path, sheet_name="Editing sites")
    logger.info("Loaded %d editing sites from %s", len(df), path.name)

    result = pd.DataFrame()

    # Chromosome - add "chr" prefix
    result["chr"] = "chr" + df["Chromosome"].astype(str)

    # Position: 1-based in file → convert to 0-based
    result["start"] = df["Position"].astype(int) - 1
    result["end"] = df["Position"].astype(int)

    # Strand inferred from reference base: C=plus, G=minus
    result["strand"] = df["Reference base"].map({"C": "+", "G": "-"})

    # Site ID
    result["site_id"] = result.apply(
        lambda r: f"asaoka_{r['chr']}_{r['start']}", axis=1
    )

    result["ref"] = df["Reference base"]
    result["gene"] = df["Gene"]
    result["feature"] = df["Region"]
    result["edit_type"] = "C-to-U"
    result["is_edited"] = 1
    result["dataset_source"] = "asaoka_2019"

    # Parse amino acid change info
    result["codon_change"] = df["Codon change"]
    result["aa_change"] = df["Amino acid change"]
    result["comment"] = df["Comment"]  # Cell line info (293T-3A, etc.)

    # No editing rates in this dataset
    result["editing_rate"] = float("nan")

    logger.info("  Strand distribution: %s",
                dict(result["strand"].value_counts()))
    logger.info("  Region distribution: %s",
                dict(result["feature"].value_counts()))

    return result


def parse_negative_controls(path: Path) -> pd.DataFrame:
    """Parse the 'Negative control sites' sheet."""
    df = pd.read_excel(path, sheet_name="Negative control sites")
    logger.info("Loaded %d negative control sites", len(df))

    result = pd.DataFrame()
    result["chr"] = "chr" + df["Chromosome"].astype(str)
    result["start"] = df["Position"].astype(int) - 1
    result["end"] = df["Position"].astype(int)
    result["strand"] = "+"  # Unknown; will need genome lookup to determine
    result["site_id"] = result.apply(
        lambda r: f"asaoka_neg_{r['chr']}_{r['start']}", axis=1
    )
    result["ref"] = ""
    result["gene"] = ""
    result["feature"] = ""
    result["edit_type"] = "C-to-U"
    result["is_edited"] = 0
    result["dataset_source"] = "asaoka_2019"
    result["codon_change"] = ""
    result["aa_change"] = ""
    result["comment"] = "negative_control"
    result["editing_rate"] = float("nan")

    return result


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not RAW_PATH.exists():
        logger.error("Raw file not found: %s", RAW_PATH)
        return

    # Parse editing sites
    sites = parse_editing_sites(RAW_PATH)

    # Parse negative controls
    negatives = parse_negative_controls(RAW_PATH)

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sites_path = OUTPUT_DIR / "asaoka_2019_editing_sites.csv"
    sites.to_csv(sites_path, index=False)
    logger.info("Saved %d editing sites to %s", len(sites), sites_path)

    neg_path = OUTPUT_DIR / "asaoka_2019_negatives.csv"
    negatives.to_csv(neg_path, index=False)
    logger.info("Saved %d negative controls to %s", len(negatives), neg_path)

    # Summary
    logger.info("\n=== Asaoka 2019 Summary ===")
    logger.info("Editing sites: %d", len(sites))
    logger.info("  Exonic: %d", (sites["feature"] == "exonic").sum())
    logger.info("  UTR3: %d", (sites["feature"] == "UTR3").sum())
    logger.info("  UTR5: %d", (sites["feature"] == "UTR5").sum())
    logger.info("  Intronic: %d", (sites["feature"] == "intronic").sum())
    logger.info("Negative controls: %d", len(negatives))

    # Check overlap with advisor dataset
    combined_path = PROJECT_ROOT / "data" / "processed" / "all_datasets_combined.csv"
    if combined_path.exists():
        existing = pd.read_csv(combined_path)
        existing_coords = set(
            zip(existing["chr"], existing["start"])
        )
        overlap = sites[
            sites.apply(lambda r: (r["chr"], r["start"]) in existing_coords, axis=1)
        ]
        logger.info("Overlap with existing datasets: %d sites", len(overlap))


if __name__ == "__main__":
    main()
