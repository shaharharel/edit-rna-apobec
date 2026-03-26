"""Filter splits_expanded.csv to APOBEC3A-only sites.

Creates splits_expanded_a3a.csv by:
1. Filtering Advisor (Levanon) sites to only 120 "APOBEC3A Only" annotated sites
   (out of 636 total — removes APOBEC3G, Both, Unknown enzyme sites)
2. Removing Baysal 2016 entirely (subset of Asaoka 2019, same A3A overexpression
   in HEK293T — deduplicated so those sites appear under asaoka_2019)
3. Keeping all other datasets unchanged (asaoka_2019, alqassim_2021, sharma_2015,
   tier2/tier3 negatives)

Usage:
    python scripts/apobec3a/filter_a3a_splits.py

Input:  data/processed/splits_expanded.csv
Output: data/processed/splits_expanded_a3a.csv
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def main():
    # Load full expanded splits
    splits_path = PROCESSED_DIR / "splits_expanded.csv"
    if not splits_path.exists():
        print(f"ERROR: {splits_path} not found. Run expand_dataset.py first.")
        sys.exit(1)

    df = pd.read_csv(splits_path)
    n_before = len(df)
    print(f"Loaded {n_before} rows from splits_expanded.csv")

    # Load advisor metadata to identify A3A-only sites
    advisor_path = PROCESSED_DIR / "advisor" / "unified_editing_sites.csv"
    if not advisor_path.exists():
        print(f"ERROR: {advisor_path} not found. Run parse_advisor_excel.py first.")
        sys.exit(1)

    advisor = pd.read_csv(advisor_path)
    a3a_only_ids = set(
        advisor[advisor["affecting_over_expressed_apobec"] == "APOBEC3A Only"]["site_id"]
    )
    print(f"Found {len(a3a_only_ids)} APOBEC3A-only sites in advisor data")

    # Filter: keep advisor_c2t only if A3A-only
    is_advisor = df["dataset_source"] == "advisor_c2t"
    is_a3a = df["site_id"].isin(a3a_only_ids)
    advisor_keep = is_advisor & is_a3a
    non_advisor = ~is_advisor

    df_filtered = df[advisor_keep | non_advisor].copy()
    n_advisor_removed = is_advisor.sum() - advisor_keep.sum()
    print(f"Removed {n_advisor_removed} non-A3A advisor sites")

    # Relabel baysal_2016 as asaoka_2019 (both use A3A overexpression in HEK293T).
    # Per CLAUDE.md: "Baysal sites already appear under asaoka_2019 in splits_expanded_a3a.csv"
    is_baysal = df_filtered["dataset_source"] == "baysal_2016"
    n_baysal = is_baysal.sum()
    df_filtered.loc[is_baysal, "dataset_source"] = "asaoka_2019"
    print(f"Relabeled {n_baysal} baysal_2016 sites as asaoka_2019 (same A3A experiment)")

    # Summary
    print(f"\nResult: {len(df_filtered)} rows (was {n_before})")
    print("\nDataset breakdown:")
    print(df_filtered["dataset_source"].value_counts().to_string())

    # Save
    out_path = PROCESSED_DIR / "splits_expanded_a3a.csv"
    df_filtered.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
