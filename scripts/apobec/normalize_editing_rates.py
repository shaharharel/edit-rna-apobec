"""Normalize editing rates to a common 0-1 scale across all datasets.

Problem: editing rates are on different scales across datasets:
  - advisor_c2t (Levanon/GTEx): 0-100 percentage scale (max_gtex_editing_rate)
  - alqassim_2021: 0-1 fraction scale
  - sharma_2015: 0-1 fraction scale
  - asaoka_2019: no rates (NaN)
  - tier2_negative / tier3_negative: 0.0

Fix: add an `editing_rate_normalized` column where all rates are on 0-1 scale.
  - advisor_c2t rates are divided by 100
  - all other datasets are already on 0-1 scale (or zero/NaN)

Usage:
    python scripts/apobec/normalize_editing_rates.py
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Files to normalize
FILES = [
    PROCESSED_DIR / "splits_expanded.csv",
    PROCESSED_DIR / "splits_expanded_a3a.csv",
]

# Datasets whose editing_rate is on 0-100 percentage scale
PERCENTAGE_SCALE_DATASETS = {"advisor_c2t"}


def normalize_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Add editing_rate_normalized column with all rates on 0-1 scale."""
    df = df.copy()

    # Start with the original editing_rate
    df["editing_rate_normalized"] = df["editing_rate"].copy()

    # Divide percentage-scale datasets by 100
    for ds in PERCENTAGE_SCALE_DATASETS:
        mask = df["dataset_source"] == ds
        n_affected = mask.sum()
        if n_affected > 0:
            original_rates = df.loc[mask, "editing_rate"]
            df.loc[mask, "editing_rate_normalized"] = original_rates / 100.0
            logger.info(
                "  %s: divided %d rates by 100 (%.4f-%.4f -> %.6f-%.6f)",
                ds, n_affected,
                original_rates.min(), original_rates.max(),
                df.loc[mask, "editing_rate_normalized"].min(),
                df.loc[mask, "editing_rate_normalized"].max(),
            )

    return df


def report_distributions(df: pd.DataFrame, filename: str):
    """Report editing_rate_normalized distributions by dataset."""
    logger.info("\n=== Normalized rate distributions for %s ===", filename)
    for ds, g in sorted(df.groupby("dataset_source")):
        rates = g["editing_rate_normalized"].dropna()
        if len(rates) > 0 and rates.max() > 0:
            logger.info(
                "  %s: n=%d, min=%.6f, max=%.6f, mean=%.6f, median=%.6f",
                ds, len(rates), rates.min(), rates.max(),
                rates.mean(), rates.median(),
            )
        elif len(rates) > 0:
            logger.info("  %s: n=%d, all zero", ds, len(rates))
        else:
            logger.info("  %s: n=%d, no rates (all NaN)", ds, len(g))

    # Overall stats for sites with non-zero, non-NaN rates
    positive_rates = df[
        (df["editing_rate_normalized"] > 0)
        & df["editing_rate_normalized"].notna()
    ]["editing_rate_normalized"]
    if len(positive_rates) > 0:
        logger.info(
            "  OVERALL (non-zero): n=%d, min=%.6f, max=%.6f, mean=%.6f, median=%.6f",
            len(positive_rates), positive_rates.min(), positive_rates.max(),
            positive_rates.mean(), positive_rates.median(),
        )


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    for filepath in FILES:
        if not filepath.exists():
            logger.warning("File not found, skipping: %s", filepath)
            continue

        logger.info("Processing %s ...", filepath.name)
        df = pd.read_csv(filepath)

        # Check if column already exists
        if "editing_rate_normalized" in df.columns:
            logger.info("  Column 'editing_rate_normalized' already exists, overwriting.")

        df = normalize_rates(df)
        report_distributions(df, filepath.name)

        # Save
        df.to_csv(filepath, index=False)
        logger.info("Saved %s with editing_rate_normalized column (%d rows)\n",
                     filepath.name, len(df))


if __name__ == "__main__":
    main()
