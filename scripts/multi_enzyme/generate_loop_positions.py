#!/usr/bin/env python
"""Generate loop position features for multi-enzyme editing sites.

Adapted from scripts/apobec3a/loop_position_analysis.py.
Reads multi_enzyme_sequences_v2.json, folds each sequence with ViennaRNA,
and computes loop geometry features for the edit site at position 100.

Output: data/processed/multi_enzyme/loop_position_per_site_v2.csv

Usage:
    conda run -n quris python scripts/multi_enzyme/generate_loop_positions.py
"""
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import RNA

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Use latest version available: v3 > v2, prefer with_negatives
_ME_DIR = PROJECT_ROOT / "data/processed/multi_enzyme"
SEQ_JSON_CANDIDATES = [
    _ME_DIR / "multi_enzyme_sequences_v3_with_negatives.json",
    _ME_DIR / "multi_enzyme_sequences_v2_with_negatives.json",
    _ME_DIR / "multi_enzyme_sequences_v3.json",
    _ME_DIR / "multi_enzyme_sequences_v2.json",
]
SEQ_JSON = next((p for p in SEQ_JSON_CANDIDATES if p.exists()), SEQ_JSON_CANDIDATES[-1])

SPLITS_CSV_CANDIDATES = [
    _ME_DIR / "splits_multi_enzyme_v3_with_negatives.csv",
    _ME_DIR / "splits_multi_enzyme_v2_with_negatives.csv",
    _ME_DIR / "splits_multi_enzyme_v3.csv",
    _ME_DIR / "splits_multi_enzyme_v2.csv",
]
SPLITS_CSV = next((p for p in SPLITS_CSV_CANDIDATES if p.exists()), SPLITS_CSV_CANDIDATES[-1])

# Output matches the latest version
_version = "v3" if "v3" in SEQ_JSON.name else "v2"
OUTPUT_FILE = _ME_DIR / f"loop_position_per_site_{_version}.csv"

EDIT_POS = 100   # 0-indexed center of 201-nt window


# ---------------------------------------------------------------------------
# Structure analysis functions (same logic as apobec3a/loop_position_analysis.py)
# ---------------------------------------------------------------------------

def fold_sequence(seq: str):
    """Fold RNA, return (dot_bracket, mfe)."""
    seq = seq.upper().replace("T", "U")
    md = RNA.md()
    md.temperature = 37.0
    fc = RNA.fold_compound(seq, md)
    return fc.mfe()


def find_loop_boundaries(dot_bracket: str, pos: int):
    if pos < 0 or pos >= len(dot_bracket) or dot_bracket[pos] in "()":
        return -1, -1
    n = len(dot_bracket)
    left = pos - 1
    while left >= 0 and dot_bracket[left] == ".":
        left -= 1
    right = pos + 1
    while right < n and dot_bracket[right] == ".":
        right += 1
    return left, right


def compute_stem_length(dot_bracket: str, boundary_pos: int, direction: str) -> int:
    n = len(dot_bracket)
    if boundary_pos < 0 or boundary_pos >= n or dot_bracket[boundary_pos] not in "()":
        return 0
    count = 0
    if direction == "left":
        i = boundary_pos
        while i >= 0 and dot_bracket[i] in "()":
            count += 1
            i -= 1
    else:
        i = boundary_pos
        while i < n and dot_bracket[i] in "()":
            count += 1
            i += 1
    return count


def classify_loop_type(dot_bracket: str, left: int, right: int) -> str:
    n = len(dot_bracket)
    if left < 0 or right >= n:
        return "external"
    lc, rc = dot_bracket[left], dot_bracket[right]
    if lc == "(" and rc == ")":
        return "hairpin"
    if lc == ")" and rc == "(":
        return "multiloop"
    if lc == "(" and rc == "(":
        return "bulge_left"
    if lc == ")" and rc == ")":
        return "bulge_right"
    return "other"


def analyze_site_structure(dot_bracket: str, mfe: float, pos: int = EDIT_POS) -> dict:
    n = len(dot_bracket)
    result = {
        "is_unpaired": None,
        "loop_type": None,
        "loop_size": None,
        "dist_to_left_boundary": None,
        "dist_to_right_boundary": None,
        "dist_to_nearest_stem": None,
        "relative_loop_position": None,
        "dist_to_apex": None,
        "left_stem_length": None,
        "right_stem_length": None,
        "max_adjacent_stem_length": None,
        "dist_to_junction": None,
        "local_unpaired_fraction": None,
        "mfe": float(mfe),
    }
    if not dot_bracket or pos >= n or pos < 0:
        return result

    is_unpaired = dot_bracket[pos] == "."
    result["is_unpaired"] = bool(is_unpaired)

    window = 10
    w_start = max(0, pos - window)
    w_end = min(n, pos + window + 1)
    local_region = dot_bracket[w_start:w_end]
    result["local_unpaired_fraction"] = sum(1 for c in local_region if c == ".") / max(len(local_region), 1)

    if is_unpaired:
        left, right = find_loop_boundaries(dot_bracket, pos)
        loop_start = (left + 1) if left >= 0 else 0
        loop_end   = (right - 1) if right < n else n - 1
        loop_size  = loop_end - loop_start + 1

        dist_left  = pos - loop_start
        dist_right = loop_end - pos
        rel_pos    = dist_left / (loop_size - 1) if loop_size > 1 else 0.5
        apex       = (loop_start + loop_end) / 2.0
        dist_apex  = abs(pos - apex)

        result["loop_size"] = loop_size
        result["dist_to_left_boundary"] = dist_left
        result["dist_to_right_boundary"] = dist_right
        result["dist_to_nearest_stem"] = min(dist_left, dist_right)
        result["relative_loop_position"] = rel_pos
        result["dist_to_apex"] = dist_apex
        result["loop_type"] = classify_loop_type(dot_bracket, left, right)
        result["left_stem_length"] = compute_stem_length(dot_bracket, left, "left")
        result["right_stem_length"] = compute_stem_length(dot_bracket, right, "right")
        result["max_adjacent_stem_length"] = max(result["left_stem_length"], result["right_stem_length"])
        result["dist_to_junction"] = result["dist_to_nearest_stem"]
    else:
        dist_left_stem = 0
        i = pos - 1
        while i >= 0 and dot_bracket[i] in "()":
            dist_left_stem += 1
            i -= 1
        dist_right_stem = 0
        j = pos + 1
        while j < n and dot_bracket[j] in "()":
            dist_right_stem += 1
            j += 1
        result["dist_to_junction"] = min(dist_left_stem, dist_right_stem)
        stem_l = compute_stem_length(dot_bracket, pos, "left")
        stem_r = compute_stem_length(dot_bracket, pos, "right")
        result["left_stem_length"] = stem_l
        result["right_stem_length"] = stem_r
        result["max_adjacent_stem_length"] = max(stem_l, stem_r)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not SEQ_JSON.exists():
        logger.error("Sequences JSON not found: %s", SEQ_JSON)
        sys.exit(1)

    with open(SEQ_JSON) as f:
        sequences = json.load(f)
    logger.info("Loaded %d sequences", len(sequences))

    # Load splits for metadata
    df_splits = pd.read_csv(SPLITS_CSV) if SPLITS_CSV.exists() else pd.DataFrame()
    splits_meta = {}
    if len(df_splits) > 0:
        for _, row in df_splits.iterrows():
            splits_meta[str(row["site_id"])] = row

    rows = []
    t0 = time.time()
    n_done = 0
    n_skip = 0

    for sid, seq in sequences.items():
        if len(seq) != 201:
            n_skip += 1
            continue

        seq_rna = seq.upper().replace("T", "U")
        try:
            dot_bracket, mfe = fold_sequence(seq_rna)
        except Exception as exc:
            logger.warning("Fold failed for %s: %s", sid, exc)
            n_skip += 1
            continue

        feat = analyze_site_structure(dot_bracket, mfe, EDIT_POS)
        feat["site_id"] = sid

        # Add metadata from splits
        meta = splits_meta.get(sid, {})
        feat["enzyme"] = meta.get("enzyme", None) if hasattr(meta, "get") else None
        feat["dataset_source"] = meta.get("dataset_source", None) if hasattr(meta, "get") else None
        feat["label"] = int(meta.get("is_edited", 1)) if hasattr(meta, "get") else 1

        rows.append(feat)
        n_done += 1

        if n_done % 500 == 0:
            elapsed = time.time() - t0
            rate = n_done / elapsed
            remaining = (len(sequences) - n_done) / max(rate, 1e-6)
            logger.info("  %d/%d done (%.0f/min, ~%.0f sec remaining)",
                        n_done, len(sequences), rate * 60, remaining)

    logger.info("Done: %d computed, %d skipped in %.0f sec",
                n_done, n_skip, time.time() - t0)

    out_df = pd.DataFrame(rows)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUTPUT_FILE, index=False)
    logger.info("Saved loop positions → %s (%d rows)", OUTPUT_FILE, len(out_df))

    # Summary
    in_loop = out_df["is_unpaired"].sum()
    total = len(out_df)
    logger.info("In loop fraction: %.3f (%d/%d)", in_loop / max(total, 1), in_loop, total)
    if "enzyme" in out_df.columns:
        for enz, grp in out_df.groupby("enzyme"):
            enz_loop = grp["is_unpaired"].sum()
            logger.info("  %s: %.3f in loop (%d/%d)", enz, enz_loop / max(len(grp), 1), enz_loop, len(grp))


if __name__ == "__main__":
    main()
