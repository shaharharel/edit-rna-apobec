#!/usr/bin/env python3
"""Merge retrained APOBEC1 v4 scores into the existing v4 panel parquets.

Inputs:
  Existing v4 parquets (full v4 columns: score_binary, score_A3X, score_apobec1_v3):
    panel_scores_v4_cancer.parquet
    panel_scores_v4_cds.parquet
  Retrained-apobec1 parquets (same row order, but score_apobec1_v3 = new v4 head):
    _retrain_raw/scored_v4cds_apo1cancer.parquet
    _retrain_raw/scored_v4cds_apo1cds.parquet

Outputs:
  panel_scores_v4_cancer_apobec1retrained.parquet — all original columns +
                                          score_apobec1_v4_cancer (replacing v3)
  panel_scores_v4_cds_apobec1retrained.parquet    — all original columns +
                                          score_apobec1_v4_cds (replacing v3)

The non-apobec1 score columns are kept from the original v4 parquets so the
binary/A3X heads remain consistent with their training (phase3_v4_cancer or
phase3_v4_cds). Only the apobec1 column is replaced.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
V4_DIR = ROOT / "experiments" / "multi_enzyme" / "outputs" / "pcawg_tcw_panel" / "v4_outputs"

CASES = [
    ("cancer", V4_DIR / "panel_scores_v4_cancer.parquet",
     V4_DIR / "_retrain_raw" / "scored_v4cds_apo1cancer.parquet",
     V4_DIR / "panel_scores_v4_cancer_apobec1retrained.parquet"),
    ("cds", V4_DIR / "panel_scores_v4_cds.parquet",
     V4_DIR / "_retrain_raw" / "scored_v4cds_apo1cds.parquet",
     V4_DIR / "panel_scores_v4_cds_apobec1retrained.parquet"),
]

for variant, base_path, new_apo_path, out_path in CASES:
    print(f"\n=== variant: {variant} ===")
    base = pd.read_parquet(base_path)
    new = pd.read_parquet(new_apo_path)
    print(f"  base    {base.shape}  cols={list(base.columns)}")
    print(f"  new     {new.shape}   cols={list(new.columns)}")

    # Sanity: same row count + same chrom/pos/strand alignment
    assert len(base) == len(new), f"row count mismatch {len(base)} vs {len(new)}"
    assert (base["chrom"].values == new["chrom"].values).all(), "chrom mismatch"
    assert (base["pos"].values == new["pos"].values).all(), "pos mismatch"
    assert (base["strand"].values == new["strand"].values).all(), "strand mismatch"

    out = base.copy()
    new_col = f"score_apobec1_v4_{variant}"
    # Replace v3 column with v4 retrained scores; drop old v3 column
    out[new_col] = new["score_apobec1_v3"].values.astype(np.float32)
    out = out.drop(columns=["score_apobec1_v3"])

    print(f"  -> writing {out_path.name}  shape={out.shape}  cols={list(out.columns)}")
    out.to_parquet(out_path, index=False)

    # Quick sanity stats on new column
    valid = out["valid"].values
    new_vals = out[new_col].values
    print(f"  new {new_col}: mean={np.nanmean(new_vals[valid]):.4f}  std={np.nanstd(new_vals[valid]):.4f}")

print("\nMerge complete.")
