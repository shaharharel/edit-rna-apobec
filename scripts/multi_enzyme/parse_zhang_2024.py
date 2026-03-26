"""
Parse Zhang 2024 (Oncogene, 10.1038/s41388-024-03171-5) APOBEC3B DVR sites.

Source: T-47D breast cancer cell line with doxycycline-induced APOBEC3B expression.
Data: GEO GSE245700 - GSE245700_Processed_data_DVR_Calling.xlsx

Sheets:
- T47D_A3B_PolyA_Enrichment24H: 6,364 DVR calls at 24h
- T47D_A3B_PolyA_Enrichment72H: 5,876 DVR calls at 72h
- T47D_A3B_RibosomalDepletion: 31,550 DVR calls (ribo-depleted)
- T47D_A3B(m)_RibosomalDepletion: 27,360 DVR calls (catalytic mutant, negative control)

We extract significant C>T DVR sites (FDR < 0.05, alt_allele_fraction_diff > 0)
from the A3B wild-type conditions, excluding sites also found in the catalytic mutant.

Coordinates: hg38 (GRCh38, standard for GEO submissions since ~2019).

Output: data/processed/multi_enzyme/zhang_2024_sites.csv
"""
import sys
from pathlib import Path

import openpyxl
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "data/raw/published/zhang_2024/GSE245700_Processed_data_DVR_Calling.xlsx"
OUTPUT_FILE = PROJECT_ROOT / "data/processed/multi_enzyme/zhang_2024_sites.csv"

FDR_THRESHOLD = 0.05
MIN_ALT_DIFF = 0.0  # Any positive differential editing


def load_significant_ct_sites(filepath, sheet_name, fdr_thresh=FDR_THRESHOLD):
    """Load significant C>T DVR sites from a sheet.

    Returns set of (chr, pos) for site identification, plus detailed info.
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

        ref = vals[headers.index("Ref_allele")]
        alt = vals[headers.index("Alt_allele")]

        # Only C>T (C>U editing on + strand)
        if ref != "C" or alt != "T":
            continue

        chrom_num = vals[headers.index("Chrom")]
        pos = int(vals[headers.index("Site")])
        # Zhang reports all variants as C>T on the reference strand.
        # The RNA-seqStrand column indicates sequencing strand, not editing strand.
        # Since Ref_allele is always C, the edit position has C on the + reference strand.
        strand = "+"

        fdr = vals[headers.index("FDR")]
        alt_diff = vals[headers.index("Alt_allele_fraction_diff")]

        # Filter: significant and positive differential editing
        if not isinstance(fdr, (int, float)) or fdr >= fdr_thresh:
            continue
        if not isinstance(alt_diff, (int, float)) or alt_diff <= MIN_ALT_DIFF:
            continue

        chrom = f"chr{chrom_num}"
        key = (chrom, pos)

        if key not in sites or (isinstance(alt_diff, (int, float)) and
                                (sites[key]["alt_diff"] is None or alt_diff > sites[key]["alt_diff"])):
            sites[key] = {
                "chr": chrom,
                "pos": pos,
                "strand": strand if strand else "+",
                "fdr": fdr,
                "alt_diff": alt_diff,
                "sheet": sheet_name,
            }

    wb.close()
    return sites


def main():
    if not INPUT_FILE.exists():
        print(f"ERROR: Input file not found: {INPUT_FILE}")
        print("Download from GEO: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE245700")
        sys.exit(1)

    print("Loading Zhang 2024 APOBEC3B DVR sites...")

    # Load significant C>T sites from each condition
    polyA_24h = load_significant_ct_sites(INPUT_FILE, "T47D_A3B_PolyA_Enrichment24H")
    polyA_72h = load_significant_ct_sites(INPUT_FILE, "T47D_A3B_PolyA_Enrichment72H")
    ribo_dep = load_significant_ct_sites(INPUT_FILE, "T47D_A3B_RibosomalDepletion")

    # Load catalytic mutant sites (negative control)
    mutant = load_significant_ct_sites(INPUT_FILE, "T47D_A3B(m)_RibosomalDepletion")

    print(f"  PolyA 24h significant C>T: {len(polyA_24h)}")
    print(f"  PolyA 72h significant C>T: {len(polyA_72h)}")
    print(f"  Ribo-depleted significant C>T: {len(ribo_dep)}")
    print(f"  Catalytic mutant C>T (background): {len(mutant)}")

    # Union of all A3B WT conditions, subtract mutant background
    all_wt_sites = {}
    for sites_dict, condition in [(polyA_24h, "PolyA_24h"), (polyA_72h, "PolyA_72h"), (ribo_dep, "RiboDepletion")]:
        for key, info in sites_dict.items():
            if key not in all_wt_sites:
                all_wt_sites[key] = {**info, "conditions": [condition]}
            else:
                all_wt_sites[key]["conditions"].append(condition)

    # Remove mutant background
    a3b_specific = {k: v for k, v in all_wt_sites.items() if k not in mutant}

    print(f"\n  Union of WT conditions: {len(all_wt_sites)}")
    print(f"  After removing mutant background: {len(a3b_specific)}")

    # Build output
    rows = []
    for key, info in a3b_specific.items():
        rows.append({
            "site_id": f"{info['chr']}:{info['pos']}:{info['strand']}",
            "chr": info["chr"],
            "start": info["pos"],
            "end": info["pos"],
            "strand": info["strand"],
            "enzyme": "A3B",
            "dataset_source": "zhang_2024",
            "flanking_seq": None,  # Not available in GEO data
            "editing_rate": info["alt_diff"],  # Differential editing rate
            "fdr": info["fdr"],
            "conditions": ";".join(info["conditions"]),
            "cell_line": "T-47D",
            "coordinate_system": "hg38",
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(["chr", "start"]).reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    # Count how many appear in multiple conditions
    multi_cond = df[df["conditions"].str.contains(";")].shape[0]

    print(f"\nOutput: {OUTPUT_FILE}")
    print(f"  Total A3B-specific DVR sites: {len(df)}")
    print(f"  Sites in multiple conditions: {multi_cond}")
    print(f"  Mean differential editing rate: {df['editing_rate'].mean():.4f}")
    print(f"  Coordinate system: hg38")


if __name__ == "__main__":
    main()
