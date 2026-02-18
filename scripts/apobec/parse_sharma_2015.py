"""Parse Sharma et al. 2015 supplementary data.

Paper: "APOBEC3A cytidine deaminase induces RNA editing in monocytes and
        macrophages" (Nature Communications 2015)

First identification of APOBEC3A RNA editing in monocytes.

Supplementary data contains:
  - Hypoxia treatment: 3,166 editing sites (211 C-to-U, rest A-to-G/ADAR)
  - Macrophage polarization: 141 editing sites (122 C-to-U)

We keep only C-to-U (CU) sites, which are APOBEC3A-mediated.

Positions are 1-based genomic coordinates.
Strand is explicitly provided as TranscribedChromStrand.

Output:
  - data/processed/published/sharma_2015_editing_sites.csv

Usage:
    python scripts/apobec/parse_sharma_2015.py
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "published" / "sharma_2015_supp_data.xls"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "published"


def parse_cu_sites(df: pd.DataFrame, condition: str) -> pd.DataFrame:
    """Extract C-to-U editing sites from a Sharma 2015 sheet.

    Args:
        df: Raw dataframe from one sheet.
        condition: "hypoxia" or "macrophage".
    """
    # Filter to C-to-U only
    cu = df[df["RNABaseAlteration"] == "CU"].copy()
    logger.info("  %s: %d C-to-U sites (from %d total)", condition, len(cu), len(df))

    if len(cu) == 0:
        return pd.DataFrame()

    result = pd.DataFrame()

    # Chromosome
    result["chr"] = "chr" + cu["Chromosome"].astype(str)

    # Position: 1-based → 0-based
    result["start"] = cu["Position"].astype(int) - 1
    result["end"] = cu["Position"].astype(int)

    result["strand"] = cu["TranscribedChromStrand"].values

    # Site ID
    result["site_id"] = result.apply(
        lambda r: f"sharma_{condition}_{r['chr']}_{r['start']}", axis=1
    )

    # Determine ref base from strand (C on plus strand, G on minus strand)
    result["ref"] = cu["RefGenomeBase"].values
    result["alt"] = cu["AlteredGenomeBase"].values

    result["gene"] = cu["GeneName"].values
    result["feature"] = cu["GeneRegion"].values
    result["edit_type"] = "C-to-U"
    result["is_edited"] = 1
    result["dataset_source"] = "sharma_2015"
    result["condition"] = condition

    # Editing rates
    result["editing_rate_test"] = cu["MeanTestBaseAlterationFraction"].values
    result["editing_rate_control"] = cu["MeanControlBaseAlterationFraction"].values
    result["editing_rate"] = cu["MeanTestBaseAlterationFraction"].values

    # Fold change and significance
    result["log2fc"] = cu["Log2FoldChangeBaseAlterationIBB"].values
    result["pvalue"] = cu["RawPValueDiffBaseAlteration"].values
    result["padj"] = cu["CorrectedPValueDiffBaseAlteration"].values

    # Functional annotation
    result["codon_change"] = cu["BaseAlterationEffect"].values
    result["aa_change"] = cu["AAChange"].values

    # Sequence context (already provided)
    result["motif_minus3to0"] = cu["Minus3To0Motif"].values

    # Structural context (RNAfold from original paper)
    result["rnafold_loop"] = cu["RNAFoldLoop"].values
    result["rnafold_loop_size"] = cu["RNAFoldLoopSize"].values

    return result


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not RAW_PATH.exists():
        logger.error("Raw file not found: %s", RAW_PATH)
        return

    all_cu = []

    # Hypoxia treatment
    logger.info("Parsing Hypoxia treatment sheet...")
    hyp = pd.read_excel(RAW_PATH, sheet_name="Hypoxia treatment")
    cu_hyp = parse_cu_sites(hyp, "hypoxia")
    if len(cu_hyp) > 0:
        all_cu.append(cu_hyp)

    # Macrophage polarization
    logger.info("Parsing Macrophage polarization sheet...")
    mac = pd.read_excel(RAW_PATH, sheet_name="Macrophage polarization")
    cu_mac = parse_cu_sites(mac, "macrophage")
    if len(cu_mac) > 0:
        all_cu.append(cu_mac)

    if not all_cu:
        logger.error("No C-to-U sites found!")
        return

    combined = pd.concat(all_cu, ignore_index=True)

    # Deduplicate by coordinate (same site may appear in both conditions)
    coord_counts = combined.groupby(["chr", "start"]).size()
    n_dup = (coord_counts > 1).sum()
    if n_dup > 0:
        logger.info("Found %d sites in both conditions — keeping both with distinct IDs", n_dup)

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "sharma_2015_editing_sites.csv"
    combined.to_csv(out_path, index=False)
    logger.info("Saved %d C-to-U editing sites to %s", len(combined), out_path)

    # Summary
    logger.info("\n=== Sharma 2015 Summary ===")
    logger.info("Total C-to-U sites: %d", len(combined))
    logger.info("  Hypoxia: %d", (combined["condition"] == "hypoxia").sum())
    logger.info("  Macrophage: %d", (combined["condition"] == "macrophage").sum())
    logger.info("Region distribution:")
    for region, count in combined["feature"].value_counts().items():
        logger.info("  %s: %d", region, count)

    # Unique genomic positions
    unique_coords = combined.drop_duplicates(subset=["chr", "start"])
    logger.info("Unique genomic positions: %d", len(unique_coords))

    # Check overlap with existing datasets
    existing_path = PROJECT_ROOT / "data" / "processed" / "all_datasets_combined.csv"
    if existing_path.exists():
        existing = pd.read_csv(existing_path)
        existing_coords = set(zip(existing["chr"], existing["start"]))
        overlap = unique_coords[
            unique_coords.apply(
                lambda r: (r["chr"], r["start"]) in existing_coords, axis=1
            )
        ]
        logger.info("Overlap with existing datasets: %d sites", len(overlap))


if __name__ == "__main__":
    main()
