"""Parse the advisor's C2TFinalSites.DB.xlsx into clean CSVs.

Reads each sheet from the multi-sheet Excel workbook and writes
normalized CSVs to data/processed/advisor/.

Usage:
    python scripts/apobec/parse_advisor_excel.py [--input PATH] [--output-dir PATH]
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "C2TFinalSites.DB.xlsx"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "advisor"

# Sheets that use a merged header row (row 0 is group label, row 1 is column name)
MERGED_HEADER_SHEETS = {
    "T1-GTEx Editing & Conservation",
    "T2-Non GTEx Editing Sum.",
    "T3-APOBECs Correlations",
    "T4-Leukocytes Editing Rates",
    "T5 -TCGA Survival",
    "Supp TX All Non AG MM Sites ",
}

# Sheets with simple single-row headers
SIMPLE_HEADER_SHEETS = {
    "Supp. T1 PhlyP difference P-val",
    "Supp. T1 CADD Stats",
    "Supp T2 Non Human Summary stats",
    "Supp T3 Structures",
    "Supp TX Bases Editors Diff. ",
    "Supp TX Viral Editing",
    "Supp TX Non-Human Sample List",
}


def clean_column_name(name: str) -> str:
    """Normalize column name: strip whitespace, replace spaces with underscores."""
    s = str(name).strip()
    # Remove trailing whitespace and normalize internal spaces
    s = "_".join(s.split())
    return s


def parse_merged_header_sheet(xl_path: Path, sheet_name: str) -> pd.DataFrame:
    """Parse a sheet with merged header rows (row 0 = group, row 1 = column names)."""
    df = pd.read_excel(xl_path, sheet_name=sheet_name, header=1)
    df.columns = [clean_column_name(c) for c in df.columns]
    return df


def parse_simple_header_sheet(xl_path: Path, sheet_name: str) -> pd.DataFrame:
    """Parse a sheet with a single header row."""
    df = pd.read_excel(xl_path, sheet_name=sheet_name, header=0)
    df.columns = [clean_column_name(c) for c in df.columns]
    return df


def make_safe_filename(sheet_name: str) -> str:
    """Convert sheet name to a safe filename."""
    s = sheet_name.strip()
    s = s.replace(" ", "_").replace(".", "").replace("-", "_")
    s = s.replace("≤", "leq").replace("≥", "geq")
    # Remove consecutive underscores
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_").lower()


def parse_all_sheets(xl_path: Path, output_dir: Path) -> dict[str, pd.DataFrame]:
    """Parse all sheets and save as CSVs. Returns dict of sheet_name -> DataFrame."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    xl = pd.ExcelFile(xl_path)
    for sheet_name in xl.sheet_names:
        logger.info("Parsing sheet: %s", sheet_name)
        if sheet_name in MERGED_HEADER_SHEETS:
            df = parse_merged_header_sheet(xl_path, sheet_name)
        elif sheet_name in SIMPLE_HEADER_SHEETS:
            df = parse_simple_header_sheet(xl_path, sheet_name)
        else:
            logger.warning("Unknown sheet '%s', using simple header parse", sheet_name)
            df = parse_simple_header_sheet(xl_path, sheet_name)

        fname = make_safe_filename(sheet_name) + ".csv"
        out_path = output_dir / fname
        df.to_csv(out_path, index=False)
        logger.info("  -> %s (%d rows, %d cols)", out_path.name, len(df), len(df.columns))
        results[sheet_name] = df

    return results


def build_unified_site_table(sheets: dict[str, pd.DataFrame], output_dir: Path) -> pd.DataFrame:
    """Build a unified editing site table from the main T1 sheet.

    Extracts core columns: genomic coordinates, gene, editing info,
    tissue classification, conservation, structure, and editing rates.
    """
    t1 = sheets["T1-GTEx Editing & Conservation"]

    # Core locus columns
    core_cols = ["Chr", "Start", "End", "Genomic_Category", "Gene_(RefSeq)",
                 "mRNA_location_(RefSeq)", "Exonic_Function"]
    # Editing summary columns
    editing_cols = ["Edited_In_#_Tissues", "Tissue_Classification",
                    "Affecting_Over_Expressed_APOBEC",
                    "Max_GTEx_Editing_Rate", "Mean_GTEx_Editing_Rate",
                    "GTEx_Editing_Rate_SD"]
    # Conservation columns
    conservation_cols = ["Any_Non-Primate_Editing", "Any_Non-Primate_Editing_≥_1%",
                         "Any_Primate_Editing", "Any_Primate_Editing_≥_1%",
                         "Any_Mammalian_Editing", "Any_Mammlian_Editing_≥_1%"]

    # Collect available columns (names may vary slightly)
    available = set(t1.columns)
    select_cols = []
    for c in core_cols + editing_cols + conservation_cols:
        if c in available:
            select_cols.append(c)
        else:
            # Try fuzzy match
            matches = [ac for ac in available if c.lower().replace("_", " ") in ac.lower().replace("_", " ")]
            if matches:
                select_cols.append(matches[0])
            else:
                logger.warning("Column '%s' not found in T1", c)

    unified = t1[select_cols].copy()

    # Merge in structure info from Supp T3 if available
    if "Supp T3 Structures" in sheets:
        structures = sheets["Supp T3 Structures"]
        merge_cols = ["Chr", "Start", "End"]
        struct_cols = [c for c in structures.columns if c not in merge_cols]
        # Rename structure columns to avoid conflicts
        rename_map = {c: f"structure_{c}" for c in struct_cols
                      if c not in ("Tissue_Specificity", "Affecting_Over_Expressed_APOBEC")}
        structures_renamed = structures.rename(columns=rename_map)
        # Drop duplicate info columns from structures before merge
        drop_cols = [c for c in structures_renamed.columns
                     if c in unified.columns and c not in merge_cols]
        structures_renamed = structures_renamed.drop(columns=drop_cols, errors="ignore")

        unified = unified.merge(structures_renamed, on=merge_cols, how="left")

    # Create a unique site ID
    unified.insert(0, "site_id", [f"C2U_{i:04d}" for i in range(len(unified))])

    # Rename columns to clean snake_case
    unified.columns = [c.lower().replace(" ", "_").replace("(", "").replace(")", "")
                        .replace("#", "num").replace("%", "pct").replace("≥", "gte")
                        .replace("≤", "lte") for c in unified.columns]

    out_path = output_dir / "unified_editing_sites.csv"
    unified.to_csv(out_path, index=False)
    logger.info("Unified site table: %s (%d sites)", out_path, len(unified))
    return unified


def main():
    parser = argparse.ArgumentParser(description="Parse advisor Excel to CSVs")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="Path to C2TFinalSites.DB.xlsx")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                        help="Output directory for CSVs")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not args.input.exists():
        logger.error("Input file not found: %s", args.input)
        return

    sheets = parse_all_sheets(args.input, args.output_dir)
    build_unified_site_table(sheets, args.output_dir)
    logger.info("Done. Output in %s", args.output_dir)


if __name__ == "__main__":
    main()
