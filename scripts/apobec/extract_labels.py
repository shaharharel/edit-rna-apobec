"""Extract ML labels, hard negatives, and train/val/test splits.

Reads C2TFinalSites.DB.xlsx and produces:
  1. editing_sites_labels.csv  -- 636 positive editing sites with all labels
  2. hard_negatives.csv        -- 276 C-to-T sites in mRNA dropped by filter
  3. splits.csv                -- stratified 70/15/15 train/val/test split

Usage:
    python scripts/apobec/extract_labels.py [--input PATH] [--output-dir PATH]
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "C2TFinalSites.DB.xlsx"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


def read_t1(xl_path: Path) -> pd.DataFrame:
    """Read T1 (GTEx Editing & Conservation) with header on row 1."""
    return pd.read_excel(xl_path, sheet_name="T1-GTEx Editing & Conservation", header=1)


def read_t2(xl_path: Path) -> pd.DataFrame:
    """Read T2 (Non-GTEx Editing Summary) with header on row 2."""
    return pd.read_excel(xl_path, sheet_name="T2-Non GTEx Editing Sum.", header=2)


def read_t5(xl_path: Path) -> pd.DataFrame:
    """Read T5 (TCGA Survival) with header on row 1."""
    return pd.read_excel(xl_path, sheet_name="T5 -TCGA Survival", header=1)


def read_structures(xl_path: Path) -> pd.DataFrame:
    """Read Supp T3 Structures (simple header)."""
    return pd.read_excel(xl_path, sheet_name="Supp T3 Structures")


def extract_hek293_rates(t2: pd.DataFrame) -> pd.Series:
    """Extract HEK293 APOBEC3A/G overexpression editing rates.

    Returns a Series aligned to t2 index with float rates or NaN.
    """
    col = "HEK293 APOBEC3A/G Over Expression (PRJNA261741)"
    rates = pd.to_numeric(t2[col], errors="coerce")
    # Values that are "No", "Coverage < 100", "No Coverage" become NaN via coerce
    # Zero means the site was tested but not edited; keep as 0
    return rates


def build_labels(xl_path: Path) -> pd.DataFrame:
    """Build the complete label DataFrame for all 636 editing sites."""
    logger.info("Reading sheets from %s", xl_path)
    t1 = read_t1(xl_path)
    t2 = read_t2(xl_path)
    t5 = read_t5(xl_path)
    structs = read_structures(xl_path)

    n_sites = len(t1)
    logger.info("Found %d editing sites in T1", n_sites)

    # --- Genomic coordinates and gene info ---
    labels = pd.DataFrame()
    labels["chr"] = t1["Chr"]
    labels["start"] = t1["Start"]
    labels["end"] = t1["End"]
    labels["gene_name"] = t1["Gene (RefSeq)"]
    labels["genomic_category"] = t1["Genomic Category"]
    labels["mrna_location"] = t1["mRNA location (RefSeq)"]

    # --- Exonic function ---
    labels["exonic_function"] = t1["Exonic Function "].str.strip()
    # Ensure NaN stays NaN (non-coding sites have no exonic function)

    # --- APOBEC specificity ---
    apobec_raw = t1["Affecting Over Expressed APOBEC"]
    labels["apobec_class"] = apobec_raw.fillna("Unknown")

    # --- Editing rates ---
    labels["max_gtex_rate"] = t1["Max GTEx Editing Rate"]
    labels["mean_gtex_rate"] = t1["Mean GTEx Editing Rate"]
    labels["sd_gtex_rate"] = t1["GTEx Editing Rate SD"]

    # Log-transformed rate for regression
    labels["log2_max_rate"] = np.log2(labels["max_gtex_rate"] + 0.01)

    # --- Number of tissues edited ---
    labels["n_tissues_edited"] = t1["Edited In # Tissues"].astype(int)

    # --- Tissue specificity classification ---
    labels["tissue_class"] = t1["Tissue Classification"]

    # --- Edited tissues list ---
    labels["edited_tissues"] = t1["Edited Tissues (Z score ≥ 2)"]

    # --- Conservation ---
    labels["any_mammalian_conservation"] = t1["Any Mammalian Editing"].astype(bool)
    labels["any_primate_editing"] = (t1["Any Primate Editing"] == "Yes")
    labels["any_nonprimate_editing"] = (t1["Any Non-Primate Editing"] == "Yes")

    # Ordinal conservation level: 0=none, 1=primate only, 2=mammalian (incl non-primate)
    labels["conservation_level"] = 0
    labels.loc[labels["any_primate_editing"] & ~labels["any_nonprimate_editing"],
               "conservation_level"] = 1
    labels.loc[labels["any_nonprimate_editing"], "conservation_level"] = 2

    # --- RNA secondary structure (from Supp T3) ---
    # Merge on coordinates
    struct_merge = structs[["Chr", "Start", "Structure Type", "Loop Length",
                            "Min Distance to Strcutre (for open ssRNA)",
                            "Structure Type mRNA", "Structure TypePre  mRNA"]].copy()
    struct_merge = struct_merge.rename(columns={
        "Chr": "chr",
        "Start": "start",
        "Structure Type": "structure_type",
        "Loop Length": "loop_length",
        "Min Distance to Strcutre (for open ssRNA)": "min_dist_to_structure",
        "Structure Type mRNA": "structure_type_mRNA",
        "Structure TypePre  mRNA": "structure_type_premRNA",
    })
    labels = labels.merge(struct_merge, on=["chr", "start"], how="left")

    # Structure concordance: does mRNA structure match pre-mRNA structure?
    # Use object dtype to allow NaN alongside True/False
    concordance = labels["structure_type_mRNA"] == labels["structure_type_premRNA"]
    missing_struct = labels["structure_type_mRNA"].isna() | labels["structure_type_premRNA"].isna()
    concordance = concordance.astype(object)
    concordance[missing_struct] = np.nan
    labels["structure_concordance"] = concordance

    # --- TCGA survival (from T5) ---
    surv_col = "# Cancers with Editing Significantly Associated with Survival "
    cancer_col = "Cancers with Editing Significantly Associated with Survival"
    # T5 has 636 rows but only 252 have data; rest are NaN
    t5_merge = t5[["Chr", "Start", surv_col, cancer_col]].copy()
    t5_merge = t5_merge.dropna(subset=["Chr"])  # Keep only rows with data
    t5_merge = t5_merge.rename(columns={
        "Chr": "chr",
        "Start": "start_float",
        surv_col: "n_cancer_types",
        cancer_col: "cancer_types_survival",
    })
    t5_merge["start"] = t5_merge["start_float"].astype(int)
    t5_merge = t5_merge.drop(columns=["start_float"])

    labels = labels.merge(t5_merge, on=["chr", "start"], how="left")
    labels["n_cancer_types"] = labels["n_cancer_types"].fillna(0).astype(int)
    labels["has_survival_association"] = labels["n_cancer_types"] > 0

    # --- HEK293 editing rate (from T2) ---
    hek_rates = extract_hek293_rates(t2)
    # T2 is aligned 1:1 with T1 (same 636 sites, same order), so use index
    labels["hek293_rate"] = hek_rates.values

    # --- Create a unique site ID ---
    labels.insert(0, "site_id", [f"C2U_{i:04d}" for i in range(n_sites)])

    return labels


def read_nonag_sites(xl_path: Path) -> pd.DataFrame:
    """Read Supp TX All Non AG MM Sites with header on row 1."""
    return pd.read_excel(
        xl_path, sheet_name="Supp TX All Non AG MM Sites ", header=1
    )


def _parse_max_mismatch_rate(row: pd.Series, tissue_cols: list[str]) -> tuple[float, int]:
    """Parse max mismatch rate and count of tissues with signal from a row."""
    max_rate = 0.0
    n_tissues = 0
    for tissue in tissue_cols:
        val = row[tissue]
        if pd.notna(val):
            parts = str(val).split(";")
            if len(parts) == 3:
                try:
                    rate = float(parts[2])
                    if rate > max_rate:
                        max_rate = rate
                    if rate > 0:
                        n_tissues += 1
                except (ValueError, IndexError):
                    pass
    return max_rate, n_tissues


def build_hard_negatives(xl_path: Path) -> pd.DataFrame:
    """Extract 276 hard negative C-to-T sites in mRNA dropped by filter.

    These are C nucleotides that showed some C-to-T mismatch signal in GTEx
    but failed the statistical filter (FDR-corrected binomial test). They are
    in CDS or Non-Coding mRNA regions, making them the hardest negatives for
    binary editing site prediction.
    """
    logger.info("Extracting hard negatives from Supp TX sheet")
    nonag = read_nonag_sites(xl_path)

    # Filter: CT mismatch, dropped by filter, in mRNA regions
    mask = (
        (nonag["Mismatch"] == "CT")
        & (nonag["Was Sites Dropped By the Filter"] == True)  # noqa: E712
        & (nonag["Genomic Category"].isin(["CDS", "Non Coding mRNA"]))
    )
    hard_neg = nonag[mask].copy()
    logger.info("Found %d hard negative sites", len(hard_neg))

    # Parse max mismatch rate across tissues
    # Tissue columns: everything after the 7 metadata columns
    tissue_cols = list(nonag.columns[7:])
    parsed = hard_neg.apply(
        lambda row: _parse_max_mismatch_rate(row, tissue_cols), axis=1
    )
    hard_neg["max_mismatch_rate"] = [r[0] for r in parsed]
    hard_neg["n_tissues_with_signal"] = [r[1] for r in parsed]

    # Build clean output DataFrame
    result = pd.DataFrame()
    result["site_id"] = [f"NEG_{i:04d}" for i in range(len(hard_neg))]
    result["chr"] = hard_neg["Chr"].values
    result["start"] = hard_neg["Start"].values
    result["end"] = hard_neg["End"].values
    result["strand"] = hard_neg["Strand"].values
    result["genomic_category"] = hard_neg["Genomic Category"].values
    result["max_mismatch_rate"] = hard_neg["max_mismatch_rate"].values
    result["n_tissues_with_signal"] = hard_neg["n_tissues_with_signal"].values
    result["is_edited"] = False

    return result


def build_splits(
    labels: pd.DataFrame, seed: int = 42
) -> pd.DataFrame:
    """Create stratified train/val/test splits (70/15/15).

    Stratifies by the combination of apobec_class and tissue_class to
    ensure each split has representative samples of all biological groups.
    Falls back to random assignment for very small strata.
    """
    from sklearn.model_selection import StratifiedShuffleSplit

    logger.info("Creating stratified train/val/test splits (70/15/15)")

    # Create stratification key from apobec_class + tissue_class
    strat_key = labels["apobec_class"] + "__" + labels["tissue_class"]

    # Some apobec_class x tissue_class combos are very rare.
    # StratifiedShuffleSplit requires >= 2 members per class.
    # Iteratively merge small strata until all are large enough.
    def _make_safe_strata(strat: pd.Series, min_count: int = 2) -> pd.Series:
        """Merge small strata into larger ones until all have >= min_count."""
        safe = strat.copy()
        for _ in range(10):  # iterate until stable
            counts = safe.value_counts()
            rare = counts[counts < min_count].index
            if len(rare) == 0:
                break
            for val in rare:
                mask = safe == val
                # Try collapsing to apobec_class part (before "__")
                if "__" in str(val):
                    safe[mask] = val.split("__")[0]
                else:
                    # Already at single-level; merge with largest group
                    largest = counts[counts >= min_count].idxmax()
                    safe[mask] = largest
        return safe

    strat_key_safe = _make_safe_strata(strat_key, min_count=4)

    rng = np.random.RandomState(seed)

    # First split: 70% train vs 30% temp
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=rng)
    train_idx, temp_idx = next(sss1.split(labels, strat_key_safe))

    # Second split: 50/50 of the 30% -> 15% val, 15% test
    temp_strat = strat_key_safe.iloc[temp_idx]
    temp_strat_safe = _make_safe_strata(temp_strat, min_count=2)

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=rng)
    val_local_idx, test_local_idx = next(
        sss2.split(labels.iloc[temp_idx], temp_strat_safe)
    )
    val_idx = temp_idx[val_local_idx]
    test_idx = temp_idx[test_local_idx]

    # Build splits DataFrame
    splits = pd.DataFrame({
        "site_id": labels["site_id"],
        "split": "train",
    })
    splits.loc[val_idx, "split"] = "val"
    splits.loc[test_idx, "split"] = "test"

    logger.info(
        "Split sizes: train=%d, val=%d, test=%d",
        (splits["split"] == "train").sum(),
        (splits["split"] == "val").sum(),
        (splits["split"] == "test").sum(),
    )

    return splits


def main():
    parser = argparse.ArgumentParser(
        description="Extract ML labels, hard negatives, and splits"
    )
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT,
        help="Path to C2TFinalSites.DB.xlsx",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Output directory for CSVs",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for splits",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not args.input.exists():
        logger.error("Input file not found: %s", args.input)
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Positive site labels ---
    labels = build_labels(args.input)
    labels_path = args.output_dir / "editing_sites_labels.csv"
    labels.to_csv(labels_path, index=False)
    logger.info("Wrote %d positive sites to %s", len(labels), labels_path)

    # --- 2. Hard negatives ---
    hard_neg = build_hard_negatives(args.input)
    neg_path = args.output_dir / "hard_negatives.csv"
    hard_neg.to_csv(neg_path, index=False)
    logger.info("Wrote %d hard negatives to %s", len(hard_neg), neg_path)

    # --- 3. Stratified splits ---
    splits = build_splits(labels, seed=args.seed)
    splits_path = args.output_dir / "splits.csv"
    splits.to_csv(splits_path, index=False)
    logger.info("Wrote splits to %s", splits_path)

    # --- Summary ---
    print(f"\n{'='*60}")
    print("EXTRACTION SUMMARY")
    print(f"{'='*60}")

    print(f"\n--- Positive Sites ({len(labels)}) ---")
    print(f"Genomic category: {labels['genomic_category'].value_counts().to_dict()}")
    print(f"APOBEC class: {labels['apobec_class'].value_counts().to_dict()}")
    print(f"Tissue class: {labels['tissue_class'].value_counts().to_dict()}")
    print(f"Max GTEx rate: mean={labels['max_gtex_rate'].mean():.2f}%, "
          f"median={labels['max_gtex_rate'].median():.2f}%")
    print(f"HEK293 rate available: {labels['hek293_rate'].notna().sum()}")

    print(f"\n--- Hard Negatives ({len(hard_neg)}) ---")
    print(f"Genomic category: {hard_neg['genomic_category'].value_counts().to_dict()}")
    print(f"Max mismatch rate: mean={hard_neg['max_mismatch_rate'].mean():.2f}%, "
          f"median={hard_neg['max_mismatch_rate'].median():.2f}%")

    print(f"\n--- Train/Val/Test Splits ---")
    split_counts = splits["split"].value_counts()
    for s in ["train", "val", "test"]:
        n = split_counts.get(s, 0)
        print(f"  {s}: {n} ({n/len(splits)*100:.1f}%)")

    # Verify stratification quality
    merged = labels.merge(splits, on="site_id")
    print(f"\nAPOBEC class distribution per split:")
    ct = pd.crosstab(merged["split"], merged["apobec_class"], normalize="index")
    print(ct.round(3).to_string())
    print(f"\nTissue class distribution per split:")
    ct2 = pd.crosstab(merged["split"], merged["tissue_class"], normalize="index")
    print(ct2.round(3).to_string())

    print(f"\nOutput files:")
    print(f"  {labels_path}")
    print(f"  {neg_path}")
    print(f"  {splits_path}")


if __name__ == "__main__":
    main()
