"""
Loop position analysis for multi-enzyme editing sites.

Computes per-site structural features: is_unpaired, loop_size, relative_loop_position,
dist_to_apex, left/right_stem_length, max_adjacent_stem_length, dist_to_junction,
local_unpaired_fraction.

Uses ViennaRNA for structure prediction on sequences from multi_enzyme_sequences.json.

Output: data/processed/multi_enzyme/loop_position_per_site.csv

Usage:
    conda run -n quris python scripts/multi_enzyme/loop_position_analysis.py
"""
import json
import logging
import sys
from pathlib import Path

import pandas as pd

# Import loop analysis functions from the A3A version
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "apobec3a"))
from loop_position_analysis import (
    fold_sequence,
    analyze_site_structure,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SPLITS_CSV = PROJECT_ROOT / "data/processed/multi_enzyme/splits_multi_enzyme.csv"
SEQ_JSON = PROJECT_ROOT / "data/processed/multi_enzyme/multi_enzyme_sequences.json"
OUTPUT_CSV = PROJECT_ROOT / "data/processed/multi_enzyme/loop_position_per_site.csv"


def main():
    if not SPLITS_CSV.exists() or not SEQ_JSON.exists():
        logger.error("Missing inputs: %s or %s", SPLITS_CSV, SEQ_JSON)
        sys.exit(1)

    df = pd.read_csv(SPLITS_CSV)
    with open(SEQ_JSON) as f:
        sequences = json.load(f)

    logger.info("Sites: %d, Sequences: %d", len(df), len(sequences))

    rows = []
    computed = 0
    skipped = 0

    for _, row in df.iterrows():
        sid = row["site_id"]
        if sid not in sequences:
            skipped += 1
            continue

        seq = sequences[sid]
        edit_pos = len(seq) // 2

        # Fold sequence
        dot_bracket, mfe = fold_sequence(seq)

        # Analyze structure at edit position
        features = analyze_site_structure(dot_bracket, edit_pos)
        features["mfe"] = mfe
        features["site_id"] = sid
        features["enzyme"] = row["enzyme"]
        features["dataset_source"] = row["dataset_source"]
        features["seq_length"] = len(seq)

        rows.append(features)
        computed += 1

        if computed % 500 == 0:
            logger.info("  %d/%d computed", computed, len(df))

    result_df = pd.DataFrame(rows)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUTPUT_CSV, index=False)

    logger.info("Saved: %s (%d sites, %d skipped)", OUTPUT_CSV, len(result_df), skipped)

    # Summary
    for enzyme in sorted(result_df["enzyme"].unique()):
        subset = result_df[result_df["enzyme"] == enzyme]
        unpaired = subset["is_unpaired"].sum()
        total = len(subset)
        print(f"  {enzyme}: {total} sites, {unpaired}/{total} unpaired ({unpaired/total*100:.1f}%)")


if __name__ == "__main__":
    main()
