"""Parse Alqassim et al. 2021 supplementary data into clean CSVs.

Paper: "APOBEC3A deaminase activity drives C-to-U editing in human monocytes
        during differentiation to macrophages" (NAR 2021)

Supp Data 1: 209 APOBEC-dependent C-to-U editing sites identified in human
             monocytes (macrophage differentiation). Contains genomic
             coordinates, editing rates, gene annotations.

Supp Data 2: Differential gene expression (DESeq2) between APOBEC3A
             knockdown and scrambled control. Not used for editing sites.

Output:
  - data/processed/published/alqassim_2021_editing_sites.csv
  - data/processed/published/alqassim_2021_deg.csv (gene expression)

Usage:
    python scripts/apobec/parse_alqassim_2021.py
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUPP1_PATH = PROJECT_ROOT / "data" / "raw" / "published" / "alqassim_2021" / "alqassim_2021_supp_data1.xlsx"
SUPP2_PATH = PROJECT_ROOT / "data" / "raw" / "published" / "alqassim_2021" / "alqassim_2021_supp_data2.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "published"


def parse_supp_data1(path: Path) -> pd.DataFrame:
    """Parse Supplementary Data 1: APOBEC editing sites.

    The file contains 209 C-to-U editing sites identified in monocytes.
    Positions are 1-based (standard genomic convention).

    For sites on the minus strand, the reference base is G and the alt is A
    (the C-to-U edit is on the mRNA strand).
    """
    df = pd.read_excel(path)
    logger.info("Loaded %d rows from %s", len(df), path.name)

    # Create standardized output
    result = pd.DataFrame()

    # Site ID based on coordinates
    result["site_id"] = df.apply(
        lambda r: f"alq_{r['chr']}_{r['pos']}_{r['EditEv']}", axis=1
    )

    result["chr"] = df["chr"]
    # Convert 1-based to 0-based (BED format) for consistency with advisor data
    result["start"] = df["pos"] - 1
    result["end"] = df["pos"]
    result["strand"] = df["strand"].replace("*", "+")
    result["ref"] = df["ref"]
    result["alt"] = df["alt"]
    result["edit_type"] = df["EditEv"]
    result["gene"] = df["Symbol"]
    result["feature"] = df["Feature"]

    # Editing rates
    result["rate_monocyte_avg"] = df["Mo.ave"]
    result["rate_stemcell_avg"] = df["SC.ave"]
    result["rate_knockdown_avg"] = df["KD.ave"]
    result["fold_change_sc_vs_m0"] = df["FC_SCvsM0"]

    # Statistical significance
    result["glm_pvalue"] = df["glmPV"]
    result["padj"] = df["padj"]
    result["glm_OR"] = df["glmOR"]

    # Codon/AA change
    result["codon"] = df["Codon"]
    result["aa_change"] = df["AAChange"]

    # Flanking sequences
    result["upstream_seq"] = df["up_plusmRNASeq"]
    result["downstream_seq"] = df["down_plusmRNASeq"]

    # dbSNP
    result["dbsnp"] = df["dbSNP144"]

    # Total reads
    result["total_reads"] = df["TotReads"]

    # Dataset source
    result["dataset"] = "alqassim_2021"

    logger.info("Parsed %d editing sites", len(result))
    logger.info("  C-to-T: %d, G-to-A: %d",
                (result["edit_type"] == "CtT").sum(),
                (result["edit_type"] == "GtA").sum())
    logger.info("  Genomic features: %s", result["feature"].value_counts().to_dict())
    logger.info("  Mean editing rate (SC): %.3f", result["rate_stemcell_avg"].mean())

    return result


def parse_supp_data2(path: Path) -> pd.DataFrame:
    """Parse Supplementary Data 2: Differential gene expression.

    Contains DESeq2 results for genes differentially expressed between
    APOBEC3A knockdown (KD) and scrambled control (SC).
    """
    # Read Up and Down sheets
    up = pd.read_excel(path, sheet_name="Up")
    down = pd.read_excel(path, sheet_name="Down")

    up["direction"] = "up"
    down["direction"] = "down"

    df = pd.concat([up, down], ignore_index=True)

    result = pd.DataFrame()
    result["gene_ensembl"] = df["Unnamed: 0"]
    result["gene"] = df["Gene"]
    result["baseMean"] = df["baseMean"]
    result["log2FoldChange"] = df["log2FoldChange"]
    result["pvalue"] = df["pvalue"]
    result["padj"] = df["padj"]
    result["direction"] = df["direction"]

    logger.info("Parsed %d DEGs (%d up, %d down)",
                len(result),
                (result["direction"] == "up").sum(),
                (result["direction"] == "down").sum())

    return result


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Parse Supp Data 1: Editing sites
    if SUPP1_PATH.exists():
        sites = parse_supp_data1(SUPP1_PATH)
        out_path = OUTPUT_DIR / "alqassim_2021_editing_sites.csv"
        sites.to_csv(out_path, index=False)
        logger.info("Saved %d editing sites to %s", len(sites), out_path)
    else:
        logger.error("Supp Data 1 not found: %s", SUPP1_PATH)

    # Parse Supp Data 2: DEGs
    if SUPP2_PATH.exists():
        degs = parse_supp_data2(SUPP2_PATH)
        out_path = OUTPUT_DIR / "alqassim_2021_deg.csv"
        degs.to_csv(out_path, index=False)
        logger.info("Saved %d DEGs to %s", len(degs), out_path)
    else:
        logger.error("Supp Data 2 not found: %s", SUPP2_PATH)


if __name__ == "__main__":
    main()
