"""
Parse Dang 2019 (Genome Biology, 10.1186/s13059-019-1651-1) A3G RNA editing sites.

Source: Natural killer (NK) cells under hypoxia showing APOBEC3G-mediated C>U editing.
Data: Additional file 4 (MOESM4) contains 119 C>U sites in NK cells under hypoxia.

The paper reports these as APOBEC3G-attributed based on:
- CC dinucleotide preference (A3G edits 3' C of CC motif)
- Hypoxia-induced upregulation of A3G
- 5 sites in normoxia (NK_Norm sheet) as baseline

Coordinates: hg19 (the paper uses hg19; MOESM7 has an hg38 column for some sites).
Context: upstream/downstream 15nt flanking sequences in columns.

Output: data/processed/multi_enzyme/dang_2019_sites.csv
"""
import sys
from pathlib import Path

import openpyxl
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "data/raw/published/dang_2019/13059_2019_1651_MOESM4_ESM.xlsx"
# MOESM4 has all 119 NK_Hyp sites; MOESM5 has 40 FDR-filtered ones
# We use MOESM4 so downstream analyses can apply their own thresholds
OUTPUT_FILE = PROJECT_ROOT / "data/processed/multi_enzyme/dang_2019_sites.csv"


def parse_sheet(filepath, sheet_name):
    """Parse a sheet from the Dang 2019 supplementary file."""
    # data_only=True reads cached formula values (AveHypoxic is an Excel formula)
    wb = openpyxl.load_workbook(filepath, data_only=True)
    ws = wb[sheet_name]

    headers = None
    rows = []
    for i, row in enumerate(ws.iter_rows(values_only=True)):
        vals = list(row)
        if i == 0:
            headers = vals
            continue

        chrom = vals[headers.index("chr")]
        pos = int(vals[headers.index("pos")])
        ref = vals[headers.index("ref")]
        alt = vals[headers.index("alt")]

        # Determine strand from ref/alt
        if ref == "C" and alt == "T":
            strand = "+"
        elif ref == "G" and alt == "A":
            strand = "-"
        else:
            print(f"  WARNING: Unexpected edit type {ref}>{alt} at {chrom}:{pos}, skipping")
            continue

        # Get flanking sequences
        up_seq = vals[headers.index("up_plusmRNASeq")] if "up_plusmRNASeq" in headers else None
        down_seq = vals[headers.index("down_plusmRNASeq")] if "down_plusmRNASeq" in headers else None

        # Build 31nt context: 15nt upstream + edited base + 15nt downstream
        if up_seq and down_seq:
            edited_base = ref  # in DNA notation
            flanking_seq = f"{up_seq}{edited_base}{down_seq}"
        else:
            flanking_seq = None

        # Get gene name
        gene = vals[headers.index("Gene")] if "Gene" in headers else None

        # Get editing rate: use AveHypoxic (paper-computed average, 0-1 scale)
        if "AveHypoxic" in headers:
            avg_rate = vals[headers.index("AveHypoxic")]
            avg_rate = float(avg_rate) if isinstance(avg_rate, (int, float)) else None
        else:
            # Fallback: average hypoxic EdLv columns (contain "-NK-H" in name)
            edlv_hyp_cols = [j for j, h in enumerate(headers) if h and "EdLv" in str(h) and "-NK-H" in str(h)]
            rates = [float(vals[j]) for j in edlv_hyp_cols if isinstance(vals[j], (int, float)) and vals[j] > 0]
            avg_rate = sum(rates) / len(rates) if rates else None

        # Get FDR
        fdr = vals[headers.index("glmFDR")] if "glmFDR" in headers else None

        rows.append({
            "site_id": f"{chrom}:{pos}:{strand}",
            "chr": chrom,
            "start": pos,
            "end": pos,
            "strand": strand,
            "enzyme": "A3G",
            "dataset_source": "dang_2019",
            "flanking_seq": flanking_seq,
            "editing_rate": avg_rate,
            "gene": gene,
            "fdr": fdr,
            "condition": sheet_name,
            "cell_type": "NK_cells",
            "coordinate_system": "hg19",
        })

    wb.close()
    return rows


def main():
    if not INPUT_FILE.exists():
        print(f"ERROR: Input file not found: {INPUT_FILE}")
        sys.exit(1)

    print("Loading Dang 2019 A3G NK cell editing sites...")

    # Parse hypoxia sites (main dataset - 119 sites)
    hyp_rows = parse_sheet(INPUT_FILE, "NK_Hyp")
    print(f"  NK_Hyp: {len(hyp_rows)} C>U sites")

    # Parse normoxia sites (baseline - 5 sites)
    norm_rows = parse_sheet(INPUT_FILE, "NK_Norm")
    print(f"  NK_Norm: {len(norm_rows)} C>U sites")

    # Combine (keep both conditions, distinguish by 'condition' column)
    all_rows = hyp_rows + norm_rows
    df = pd.DataFrame(all_rows)
    df = df.sort_values(["chr", "start"]).reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nOutput: {OUTPUT_FILE}")
    print(f"  Total sites: {len(df)}")
    print(f"  NK_Hyp: {(df['condition'] == 'NK_Hyp').sum()}")
    print(f"  NK_Norm: {(df['condition'] == 'NK_Norm').sum()}")
    print(f"  Sites with editing rate: {df['editing_rate'].notna().sum()}")
    print(f"  Sites with flanking seq: {df['flanking_seq'].notna().sum()}")
    print(f"  Coordinate system: hg19 (liftover to hg38 needed)")


if __name__ == "__main__":
    main()
