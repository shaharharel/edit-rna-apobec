"""
Parse Kockler 2026 (Zenodo DOI: 10.5281/zenodo.18079216) RNA editing sites.

Source: BT-474 breast cancer cell line hyperexpressing APOBEC3A or APOBEC3B.
Data: ANZ4 (motif-annotated) format from Supp_RNA_Human_ANZ4_ZAK4_DAG1.xlsx.

The file contains sheets: Human_A3A_rtg1, Human_A3B_rtg1, Human_EV_rtg1.
We extract C>U editing sites (C>T or G>A on DNA) for A3A and A3B,
subtracting the empty vector (EV) background to get enzyme-specific sites.

Coordinates: hg38 (BT-474 aligned to hg38 per P-MACD pipeline).
Context: CONTEXT(+/-20) column has 41nt flanking sequence in DNA format.

Output: data/processed/multi_enzyme/kockler_2026_sites.csv
"""
import sys
from pathlib import Path

import openpyxl
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "data/raw/published/kockler_2026/anz4_MAFs/anz4/Supp_RNA_Human_ANZ4_ZAK4_DAG1.xlsx"
OUTPUT_FILE = PROJECT_ROOT / "data/processed/multi_enzyme/kockler_2026_sites.csv"


def load_c2u_sites(filepath, sheet_name):
    """Load C>U editing sites from an ANZ4 sheet.

    Returns dict of (chr, pos) -> {strand, context, rFrequency, ...}
    C>T on + strand and G>A on - strand both represent C>U RNA editing.
    """
    wb = openpyxl.load_workbook(filepath, read_only=True)
    ws = wb[sheet_name]

    headers = None
    sites = {}
    for i, row in enumerate(ws.iter_rows(values_only=True)):
        vals = list(row)
        if i == 0:
            headers = vals
            continue

        ref = vals[headers.index("Reference_Allele")]
        alt = vals[headers.index("Tumor_Seq_Allele2")]
        chrom = vals[headers.index("Chromosome")]
        pos_raw = vals[headers.index("Start_position")]
        if pos_raw is None or ref is None or alt is None:
            continue
        pos = int(pos_raw)

        # Only keep C>U edits (C>T or reverse-complement G>A)
        if ref == "C" and alt == "T":
            strand = "+"
        elif ref == "G" and alt == "A":
            strand = "-"
        else:
            continue

        context_col = headers.index("CONTEXT(+/-20)")
        context = vals[context_col] if context_col < len(vals) else None

        freq_col = headers.index("rFrequency") if "rFrequency" in headers else None
        frequency = vals[freq_col] if freq_col is not None and freq_col < len(vals) else None

        key = (chrom, pos)
        if key not in sites:
            sites[key] = {
                "chr": chrom,
                "pos": pos,
                "strand": strand,
                "context_41nt": context,
                "editing_rate": frequency / 100.0 if frequency and isinstance(frequency, (int, float)) else None,
            }
    wb.close()
    return sites


def main():
    if not INPUT_FILE.exists():
        print(f"ERROR: Input file not found: {INPUT_FILE}")
        print("Run: unzip data/raw/published/kockler_2026/anz4_MAFs.zip -d data/raw/published/kockler_2026/")
        sys.exit(1)

    print("Loading Kockler 2026 human RNA editing sites...")
    a3a_sites = load_c2u_sites(INPUT_FILE, "Human_A3A_rtg1")
    a3b_sites = load_c2u_sites(INPUT_FILE, "Human_A3B_rtg1")
    ev_sites = load_c2u_sites(INPUT_FILE, "Human_EV_rtg1")

    print(f"  A3A total C>U sites: {len(a3a_sites)}")
    print(f"  A3B total C>U sites: {len(a3b_sites)}")
    print(f"  EV (background) C>U sites: {len(ev_sites)}")

    # Subtract EV background to get enzyme-specific sites
    a3a_specific = {k: v for k, v in a3a_sites.items() if k not in ev_sites}
    a3b_specific = {k: v for k, v in a3b_sites.items() if k not in ev_sites}

    print(f"  A3A-specific (not in EV): {len(a3a_specific)}")
    print(f"  A3B-specific (not in EV): {len(a3b_specific)}")

    # Build output dataframe
    rows = []
    for key, info in a3a_specific.items():
        rows.append({
            "site_id": f"{info['chr']}:{info['pos']}:{info['strand']}",
            "chr": info["chr"],
            "start": info["pos"],
            "end": info["pos"],
            "strand": info["strand"],
            "enzyme": "A3A",
            "dataset_source": "kockler_2026",
            "flanking_seq": info["context_41nt"],
            "editing_rate": info["editing_rate"],
            "cell_line": "BT-474",
            "coordinate_system": "hg38",
        })

    for key, info in a3b_specific.items():
        rows.append({
            "site_id": f"{info['chr']}:{info['pos']}:{info['strand']}",
            "chr": info["chr"],
            "start": info["pos"],
            "end": info["pos"],
            "strand": info["strand"],
            "enzyme": "A3B",
            "dataset_source": "kockler_2026",
            "flanking_seq": info["context_41nt"],
            "editing_rate": info["editing_rate"],
            "cell_line": "BT-474",
            "coordinate_system": "hg38",
        })

    df = pd.DataFrame(rows)
    # Sort by chromosome and position
    df = df.sort_values(["chr", "start"]).reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nOutput: {OUTPUT_FILE}")
    print(f"  Total sites: {len(df)}")
    print(f"  A3A: {(df['enzyme'] == 'A3A').sum()}")
    print(f"  A3B: {(df['enzyme'] == 'A3B').sum()}")
    print(f"  Sites with editing rate: {df['editing_rate'].notna().sum()}")
    print(f"  Sites with flanking seq: {df['flanking_seq'].notna().sum()}")


if __name__ == "__main__":
    main()
