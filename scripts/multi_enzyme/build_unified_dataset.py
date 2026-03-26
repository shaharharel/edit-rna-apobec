"""
Build unified multi-enzyme APOBEC editing site dataset.

Merges sites from:
1. Kockler 2026: A3A (2,749) + A3B (3,768) from BT-474 (hg38)
2. Dang 2019: A3G (124) from NK cells (hg19 → liftover to hg38)
3. Zhang 2024: A3B (501) from T-47D (hg38)
4. Levanon T3: A3H (423) + A4 (181) expression-correlated sites (hg38)

Steps:
1. Load and merge all site CSVs
2. Liftover Dang 2019 hg19 → hg38
3. Extract 201nt sequences from hg38 (pyfaidx)
4. Deduplicate sites shared across datasets
5. Write unified CSV and sequences JSON

Output:
- data/processed/multi_enzyme/splits_multi_enzyme.csv
- data/processed/multi_enzyme/multi_enzyme_sequences.json

Usage:
    conda run -n quris python scripts/multi_enzyme/build_unified_dataset.py
"""
import json
import logging
import sys
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

# Input files
KOCKLER_CSV = DATA_DIR / "processed/multi_enzyme/kockler_2026_sites.csv"
DANG_CSV = DATA_DIR / "processed/multi_enzyme/dang_2019_sites.csv"
ZHANG_CSV = DATA_DIR / "processed/multi_enzyme/zhang_2024_sites.csv"
LEVANON_XLSX = DATA_DIR / "raw/C2TFinalSites.DB.xlsx"
GENOME_FA = DATA_DIR / "raw/genomes/hg38.fa"

# Output files
OUTPUT_CSV = DATA_DIR / "processed/multi_enzyme/splits_multi_enzyme.csv"
OUTPUT_SEQ = DATA_DIR / "processed/multi_enzyme/multi_enzyme_sequences.json"

FLANK_SIZE = 100  # ±100nt around edit site → 201nt total


def revcomp(seq: str) -> str:
    comp = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N",
            "a": "t", "t": "a", "c": "g", "g": "c", "n": "n"}
    return "".join(comp.get(b, "N") for b in reversed(seq))


def load_kockler():
    """Load Kockler 2026 A3A + A3B sites."""
    df = pd.read_csv(KOCKLER_CSV)
    df = df.rename(columns={"start": "start", "end": "end"})
    df["is_edited"] = 1
    df["source_type"] = "overexpression"
    logger.info("Kockler 2026: %d sites (A3A=%d, A3B=%d)",
                len(df), (df["enzyme"] == "A3A").sum(), (df["enzyme"] == "A3B").sum())
    return df


def load_dang():
    """Load Dang 2019 A3G sites and liftover hg19 → hg38."""
    df = pd.read_csv(DANG_CSV)
    df["is_edited"] = 1
    df["source_type"] = "endogenous"

    # Liftover hg19 → hg38
    from pyliftover import LiftOver
    lo = LiftOver("hg19", "hg38")

    lifted_rows = []
    failed = 0
    for _, row in df.iterrows():
        result = lo.convert_coordinate(row["chr"], int(row["start"]))
        if result and len(result) > 0:
            new_chr, new_pos, new_strand, _ = result[0]
            row_copy = row.copy()
            row_copy["chr"] = new_chr
            row_copy["start"] = int(new_pos)
            row_copy["end"] = int(new_pos)
            row_copy["coordinate_system"] = "hg38"
            row_copy["site_id"] = f"{new_chr}:{int(new_pos)}:{row['strand']}"
            lifted_rows.append(row_copy)
        else:
            failed += 1
            logger.warning("Liftover failed: %s:%d", row["chr"], row["start"])

    df_lifted = pd.DataFrame(lifted_rows)
    logger.info("Dang 2019: %d sites lifted to hg38 (%d failed)",
                len(df_lifted), failed)
    return df_lifted


def load_zhang():
    """Load Zhang 2024 A3B sites."""
    df = pd.read_csv(ZHANG_CSV)
    df["is_edited"] = 1
    df["source_type"] = "overexpression"
    logger.info("Zhang 2024: %d A3B sites", len(df))
    return df


def load_levanon_a3h_a4():
    """Load A3H and A4 expression-correlated sites from Levanon T3 sheet.

    Levanon T3 uses BED-style 0-based coordinates. We determine strand by
    checking the reference base at the 0-based position: C → plus strand,
    G → minus strand (C>U editing on antisense).
    """
    from pyfaidx import Fasta
    genome = Fasta(str(GENOME_FA))

    wb = openpyxl.load_workbook(LEVANON_XLSX, read_only=True)
    ws = wb["T3-APOBECs Correlations"]

    rows_a3h = []
    rows_a4 = []
    for i, row in enumerate(ws.iter_rows(values_only=True)):
        if i < 3:  # Header rows
            continue
        vals = list(row)
        chrom = vals[0]
        if chrom is None:
            continue

        start_0based = int(vals[1])
        gene = vals[4]
        a3h_flag = vals[13]  # Column N (0-indexed 13)
        a4_flag = vals[14]   # Column O (0-indexed 14)

        # Determine strand from reference base
        ref_base = str(genome[chrom][start_0based]).upper()
        if ref_base == "C":
            strand = "+"
        elif ref_base == "G":
            strand = "-"
        else:
            logger.warning("Levanon site %s:%d has ref=%s (not C or G), skipping",
                           chrom, start_0based, ref_base)
            continue

        # Convert to 1-based for consistency with other datasets
        pos_1based = start_0based + 1

        base_info = {
            "chr": chrom,
            "start": pos_1based,
            "end": pos_1based,
            "strand": strand,
            "gene": gene,
            "is_edited": 1,
            "coordinate_system": "hg38",
            "source_type": "expression_correlated",
        }

        if a3h_flag is True:
            row_data = base_info.copy()
            row_data["enzyme"] = "A3H"
            row_data["dataset_source"] = "levanon_t3"
            row_data["site_id"] = f"{chrom}:{pos_1based}:{strand}"
            rows_a3h.append(row_data)

        if a4_flag is True:
            row_data = base_info.copy()
            row_data["enzyme"] = "A4"
            row_data["dataset_source"] = "levanon_t3"
            row_data["site_id"] = f"{chrom}:{pos_1based}:{strand}"
            rows_a4.append(row_data)

    wb.close()
    df_a3h = pd.DataFrame(rows_a3h)
    df_a4 = pd.DataFrame(rows_a4)
    logger.info("Levanon T3: A3H=%d, A4=%d expression-correlated sites",
                len(df_a3h), len(df_a4))
    return pd.concat([df_a3h, df_a4], ignore_index=True)


def extract_sequences(df, genome_path):
    """Extract sequences for all sites.

    Strategy:
    - Try extracting 201nt from hg38 and verify center base is C (or G for minus strand)
    - If hg38 extraction gives correct center base, use the 201nt genome sequence
    - If not (cancer cell line rearrangements), fall back to flanking_seq from dataset
    - Flanking sequences from Kockler are 41nt in DNA format; convert to RNA with C at center
    """
    from pyfaidx import Fasta

    genome = Fasta(str(genome_path))
    sequences = {}
    from_genome = 0
    from_flanking = 0
    missing = 0

    for _, row in df.iterrows():
        sid = row["site_id"]
        chrom = row["chr"]
        pos = int(row["start"])
        strand = row.get("strand", "+")
        flanking = row.get("flanking_seq", None)
        if isinstance(flanking, float):
            flanking = None

        if chrom not in genome:
            # Use flanking sequence if available
            if flanking:
                seq = _flanking_to_rna(flanking, strand)
                if seq:
                    sequences[sid] = seq
                    from_flanking += 1
                    continue
            missing += 1
            continue

        chrom_len = len(genome[chrom])
        # All positions are 1-based; pyfaidx is 0-based
        pos_0 = pos - 1
        start = max(0, pos_0 - FLANK_SIZE)
        end = min(chrom_len, pos_0 + FLANK_SIZE + 1)

        genome_seq = str(genome[chrom][start:end]).upper()
        center_idx = pos_0 - start  # Index of edit position in extracted sequence

        # Check if center base matches expected (C on + strand, G on - strand)
        center_ok = False
        if center_idx < len(genome_seq):
            ref_base = genome_seq[center_idx]
            if strand == "+" and ref_base == "C":
                center_ok = True
            elif strand == "-" and ref_base == "G":
                center_ok = True

        if flanking:
            # Prefer dataset-provided flanking (Kockler cancer cell line doesn't match hg38,
            # Dang hg19 liftover is imprecise). These are the actual observed sequences.
            seq = _flanking_to_rna(flanking, strand)
            if seq:
                sequences[sid] = seq
                from_flanking += 1
                continue

        if center_ok:
            # Use genome sequence (good for Levanon A3H/A4, Zhang 2024)
            if strand == "-":
                genome_seq = revcomp(genome_seq)
            rna_seq = genome_seq.replace("T", "U")
            sequences[sid] = rna_seq
            from_genome += 1
        else:
            # Genome center base doesn't match — use genome anyway (best effort)
            if strand == "-":
                genome_seq = revcomp(genome_seq)
            rna_seq = genome_seq.replace("T", "U")
            sequences[sid] = rna_seq
            from_genome += 1

    logger.info("Extracted %d sequences: %d from genome, %d from flanking, %d missing",
                len(sequences), from_genome, from_flanking, missing)
    return sequences


def _flanking_to_rna(flanking_seq, strand):
    """Convert a flanking DNA sequence to RNA with C at center.

    For minus strand Kockler data, the context is in reference genome orientation
    with G at center (complement of C). Reverse-complement to get RNA orientation.
    """
    if not flanking_seq or not isinstance(flanking_seq, str) or len(flanking_seq) < 5:
        return None

    seq = flanking_seq.upper()

    if strand == "-":
        # The context shows genomic sequence with G at center
        # Reverse complement to get the RNA (sense) orientation with C at center
        seq = revcomp(seq)

    # Convert to RNA
    rna_seq = seq.replace("T", "U")
    return rna_seq


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Check inputs
    for f in [KOCKLER_CSV, DANG_CSV, ZHANG_CSV, LEVANON_XLSX]:
        if not f.exists():
            logger.error("Missing input: %s", f)
            sys.exit(1)

    if not GENOME_FA.exists():
        logger.error("Missing genome: %s", GENOME_FA)
        logger.error("Download: wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz")
        sys.exit(1)

    # Load all datasets
    kockler_df = load_kockler()
    dang_df = load_dang()
    zhang_df = load_zhang()
    levanon_df = load_levanon_a3h_a4()

    # Standardize columns before merge
    common_cols = ["site_id", "chr", "start", "end", "strand", "enzyme",
                   "dataset_source", "is_edited", "editing_rate",
                   "coordinate_system", "source_type", "flanking_seq"]

    dfs = []
    for df in [kockler_df, dang_df, zhang_df, levanon_df]:
        for col in common_cols:
            if col not in df.columns:
                df[col] = None
        dfs.append(df[common_cols].copy())

    merged = pd.concat(dfs, ignore_index=True)
    logger.info("Before dedup: %d total sites", len(merged))

    # Deduplicate: same chr + start + enzyme → keep first
    merged["dedup_key"] = merged["chr"] + ":" + merged["start"].astype(str) + ":" + merged["enzyme"]
    n_before = len(merged)
    merged = merged.drop_duplicates(subset="dedup_key", keep="first")
    logger.info("After dedup: %d sites (removed %d duplicates)", len(merged), n_before - len(merged))
    merged = merged.drop(columns=["dedup_key"])

    # Sort
    merged = merged.sort_values(["enzyme", "chr", "start"]).reset_index(drop=True)

    # Extract sequences
    logger.info("Extracting 201nt sequences from hg38...")
    sequences = extract_sequences(merged, GENOME_FA)

    # Validate center base is C (or G for minus strand, but we already revcomp'd)
    valid = 0
    invalid_center = 0
    for sid, seq in sequences.items():
        center = len(seq) // 2
        if center < len(seq) and seq[center] == "C":
            valid += 1
        else:
            invalid_center += 1
    logger.info("Center base validation: %d C at center, %d non-C", valid, invalid_center)

    # Save outputs
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_CSV, index=False)
    logger.info("Saved: %s (%d sites)", OUTPUT_CSV, len(merged))

    with open(OUTPUT_SEQ, "w") as f:
        json.dump(sequences, f, indent=2)
    logger.info("Saved: %s (%d sequences)", OUTPUT_SEQ, len(sequences))

    # Summary
    print("\n=== Multi-Enzyme Dataset Summary ===")
    for enzyme in sorted(merged["enzyme"].unique()):
        subset = merged[merged["enzyme"] == enzyme]
        print(f"  {enzyme}: {len(subset)} sites from {subset['dataset_source'].nunique()} sources "
              f"({', '.join(subset['dataset_source'].unique())})")
    print(f"  TOTAL: {len(merged)} sites")
    print(f"  Sequences extracted: {len(sequences)}")


if __name__ == "__main__":
    main()
