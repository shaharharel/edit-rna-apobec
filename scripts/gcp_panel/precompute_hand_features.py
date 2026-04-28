#!/usr/bin/env python3
"""Precompute hand features (motif + loop + zeroed-struct_delta = 40-d) for every
cache-aligned CDS candidate.

Runs locally on the Mac. Reads:
    data/processed/gcp_panel/candidates_cache_aligned.parquet  (8.45 M positions)
    data/raw/genomes/hg19.fa
    experiments/multi_enzyme/outputs/exome_map/vienna_cache/chr{N}_vienna.json.gz

Writes:
    data/processed/gcp_panel/hand40_cache_aligned.npy   (8.45M × 40 fp32)
    data/processed/gcp_panel/valid_hand_cache_aligned.npy (8.45M bool)
    data/processed/gcp_panel/hand40_cache_aligned.meta.json

Layout of each 40-d hand vector:
    [0:24]    motif (fresh from hg19 201-nt window)
    [24:31]   struct_delta — HELD AT 0 (MFE-only regime)
    [31:40]   loop geometry (canonical from cached struct_wt)

Runtime: ~10 min (I/O-bound on cache read + gzip decompress; motif is trivial).
"""
from __future__ import annotations

import gzip
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pyfaidx import Fasta

sys.stdout.reconfigure(line_buffering=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.apobec_feature_extraction import (  # noqa: E402
    extract_motif_from_seq, _extract_loop_geometry, LOOP_FEATURE_COLS,
)

HG19_FA = PROJECT_ROOT / "data" / "raw" / "genomes" / "hg19.fa"
CAND_PATH = PROJECT_ROOT / "data" / "processed" / "gcp_panel" / "candidates_cache_aligned.parquet"
CACHE_DIR = PROJECT_ROOT / "experiments" / "multi_enzyme" / "outputs" / "exome_map" / "vienna_cache"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "gcp_panel"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
COMP = str.maketrans("ACGTN", "TGCAN")
CENTER = 100
STRUCT_DELTA_START = 24
STRUCT_DELTA_END = 31


def extract_seq(genome, chrom, pos, strand, flank=100):
    try:
        clen = len(genome[chrom])
        s, e = pos - flank, pos + flank + 1
        if s < 0 or e > clen:
            return None
        seq = str(genome[chrom][s:e]).upper()
        if strand == "-":
            seq = seq.translate(COMP)[::-1]
        if len(seq) != 201 or seq[CENTER] != "C":
            return None
        return seq
    except Exception:
        return None


def main():
    t0 = time.time()
    logger.info("Loading hg19 genome ...")
    genome = Fasta(str(HG19_FA))

    logger.info("Loading candidates_cache_aligned.parquet ...")
    cand = pd.read_parquet(CAND_PATH)
    n_all = len(cand)
    logger.info("  total candidates: %d", n_all)

    # Allocate
    hand40 = np.zeros((n_all, 40), dtype=np.float32)
    valid = np.zeros(n_all, dtype=bool)

    for chrom in ALL_CHROMS:
        cache_path = CACHE_DIR / f"{chrom}_vienna.json.gz"
        if not cache_path.exists():
            logger.warning("Missing cache for %s — skipping", chrom)
            continue
        logger.info("Loading cache %s ...", cache_path)
        with gzip.open(cache_path, "rt") as f:
            cache = json.load(f)
        fr = cache["fold_results"]

        sub_idx = cand.index[cand["chrom"] == chrom].to_numpy()
        sub = cand.iloc[sub_idx]
        n_sub = len(sub)
        if n_sub != len(fr):
            logger.warning("  %s: cand=%d vs cache=%d — mismatch, using min", chrom, n_sub, len(fr))
            n_sub = min(n_sub, len(fr))
            sub_idx = sub_idx[:n_sub]

        t_chrom = time.time()
        for i, global_row in enumerate(sub_idx):
            r = cand.iloc[global_row]
            chrom_name = r["chrom"]; pos = int(r["pos"]); strand = r["strand"]
            seq = extract_seq(genome, chrom_name, pos, strand)
            if seq is None:
                continue
            motif = extract_motif_from_seq(seq)
            loop = _extract_loop_geometry(fr[i]["struct_wt"], CENTER)
            hand40[global_row, 0:24] = motif
            # struct_delta stays zero
            hand40[global_row, 31:40] = loop
            valid[global_row] = True
        chrom_time = time.time() - t_chrom
        n_valid = int(valid[sub_idx].sum())
        logger.info("  %s: %d/%d valid, %.1f s (%.1f/s)", chrom, n_valid, n_sub, chrom_time, n_sub / max(chrom_time, 1e-9))

    hand40 = np.nan_to_num(hand40, nan=0.0)
    out_hand = OUT_DIR / "hand40_cache_aligned.npy"
    out_valid = OUT_DIR / "valid_hand_cache_aligned.npy"
    np.save(out_hand, hand40)
    np.save(out_valid, valid)
    logger.info("Wrote %s (%.1f MB)", out_hand, out_hand.stat().st_size / 1e6)
    logger.info("Wrote %s (%.1f MB)", out_valid, out_valid.stat().st_size / 1e6)
    meta = {
        "n_candidates": int(n_all),
        "n_valid": int(valid.sum()),
        "regime": "mfe_only",
        "struct_delta_zeroed_slice": [STRUCT_DELTA_START, STRUCT_DELTA_END],
        "feature_order": (
            "[0:24]=motif, [24:31]=struct_delta (zeroed), [31:40]=loop"
        ),
        "loop_cols": LOOP_FEATURE_COLS,
        "runtime_min": (time.time() - t0) / 60.0,
    }
    with open(OUT_DIR / "hand40_cache_aligned.meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Total runtime: %.1f min. valid=%d/%d (%.2f%%)",
                meta["runtime_min"], meta["n_valid"], n_all,
                100.0 * meta["n_valid"] / n_all)


if __name__ == "__main__":
    main()
