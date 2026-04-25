#!/usr/bin/env python3
"""Score per-chromosome RNA-FM output as it completes, writing per-chrom score files
and DELETING the source npz afterward to free disk space on ai-gpu2.

The RNA-FM pipeline writes 24 × ~7 GB npz files but ai-gpu2 only has 81 GB free.
This script watches for completed npzs, scores them against phase3_mfe_only +
apobec1_head_mfe_only, writes a small per-chrom scored parquet, then unlinks the npz.

Runs on ai-gpu2. Needs:
    ~/data/panel/candidates_cache_aligned.parquet
    ~/data/panel/hand40_cache_aligned.npy
    ~/data/panel/valid_hand_cache_aligned.npy
    ~/data/panel/candidates_all.parquet       (RNA-FM key set)
    ~/data/panel/rnafm/rnafm_chr{N}.npz       (produced by compute_rnafm.py — watched)
    ~/data/models/phase3_mfe_only.pt
    ~/data/models/apobec1_head_mfe_only.pt

Output:
    ~/data/panel/scored_chroms/chr{N}_scores.parquet
        (chrom, pos, strand, gene, score_binary, score_A3A, ..., score_apobec1, valid)
    Final merged: ~/data/panel/panel_scores_exome_mfe_only.parquet

Usage:
    tmux new -d -s score_watcher 'python3 score_per_chrom_and_cleanup.py 2>&1 | tee ~/logs/score_watcher.log'
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ALL_CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
D_INPUT = 1320
D_SHARED = 128
ENZYMES = ["A3A", "A3B", "A3G", "A3A_A3G", "Neither"]
N_ENZYMES_CLS = 6
STRUCT_DELTA_START_ABS = 1304
STRUCT_DELTA_END_ABS = 1311


class Phase3Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_encoder = nn.Sequential(
            nn.Linear(D_INPUT, 256), nn.GELU(), nn.Dropout(0.3), nn.LayerNorm(256),
            nn.Linear(256, D_SHARED), nn.GELU(), nn.Dropout(0.2),
        )
        self.binary_head = nn.Linear(D_SHARED, 1)
        self.enzyme_adapters = nn.ModuleDict({
            enz: nn.Sequential(nn.Linear(D_SHARED, 32), nn.GELU(), nn.Linear(32, 1))
            for enz in ENZYMES
        })
        self.enzyme_classifier = nn.Sequential(
            nn.Linear(D_SHARED, 64), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(64, N_ENZYMES_CLS),
        )


class APOBEC1Head(nn.Module):
    def __init__(self, d_shared=128):
        super().__init__()
        self.species_proj = nn.Sequential(nn.Linear(1, 16), nn.GELU(), nn.Linear(16, d_shared))
        self.head = nn.Sequential(nn.Linear(d_shared, 32), nn.GELU(), nn.Linear(32, 1))


def score_chrom(chrom: str, npz_path: Path, cand: pd.DataFrame, rnafm_cand: pd.DataFrame,
                hand40: np.ndarray, valid_hand: np.ndarray,
                phase3, apo1, device, batch: int, out_dir: Path) -> dict:
    logger = logging.getLogger(__name__)
    logger.info("[%s] Loading %s ...", chrom, npz_path)
    z = np.load(npz_path, allow_pickle=False)
    z_pos = z["pos"].astype(np.int64)
    z_strand = np.array([s.decode() if isinstance(s, bytes) else s for s in z["strand"]], dtype="U1")
    z_orig = z["orig"]
    z_delta = z["delta"]
    z_valid = z["valid"].astype(bool)
    logger.info("[%s]   RNA-FM positions in chrom: %d (valid=%d)", chrom, len(z_pos), int(z_valid.sum()))

    # Build (pos, strand) -> z-idx map
    z_key = {(int(p), s): i for i, (p, s) in enumerate(zip(z_pos, z_strand))}

    # Select cand rows on this chrom
    cand_mask = (cand["chrom"] == chrom).to_numpy()
    cand_idx = np.where(cand_mask)[0]
    n_chrom = len(cand_idx)
    logger.info("[%s]   cache-aligned candidates: %d", chrom, n_chrom)

    rnafm_orig = np.zeros((n_chrom, 640), dtype=np.float16)
    rnafm_delta = np.zeros((n_chrom, 640), dtype=np.float16)
    rnafm_valid = np.zeros(n_chrom, dtype=bool)
    for j, ci in enumerate(cand_idx):
        key = (int(cand["pos"].iloc[ci]), cand["strand"].iloc[ci])
        zi = z_key.get(key, None)
        if zi is None or not z_valid[zi]:
            continue
        rnafm_orig[j] = z_orig[zi]
        rnafm_delta[j] = z_delta[zi]
        rnafm_valid[j] = True

    logger.info("[%s]   joined RNA-FM: %d/%d", chrom, int(rnafm_valid.sum()), n_chrom)

    # Chrom-specific hand features slice
    hand_chrom = hand40[cand_idx]
    valid_chrom = valid_hand[cand_idx] & rnafm_valid

    s_binary = np.zeros(n_chrom, dtype=np.float32)
    s_enz = {enz: np.zeros(n_chrom, dtype=np.float32) for enz in ENZYMES}
    s_apo1 = np.zeros(n_chrom, dtype=np.float32)

    phase3.eval(); apo1.eval()
    t0 = time.time()
    for bi in range((n_chrom + batch - 1) // batch):
        b0 = bi * batch
        b1 = min(b0 + batch, n_chrom)
        x_orig = rnafm_orig[b0:b1].astype(np.float32)
        x_delta = rnafm_delta[b0:b1].astype(np.float32)
        x_hand = hand_chrom[b0:b1]
        X = np.concatenate([x_orig, x_delta, x_hand], axis=1).astype(np.float32)
        X[:, STRUCT_DELTA_START_ABS:STRUCT_DELTA_END_ABS] = 0.0  # safety re-zero
        xb = torch.from_numpy(X).to(device)
        with torch.no_grad():
            shared = phase3.shared_encoder(xb)
            s_binary[b0:b1] = torch.sigmoid(phase3.binary_head(shared).squeeze(-1)).cpu().numpy()
            for enz in ENZYMES:
                l = phase3.enzyme_adapters[enz](shared).squeeze(-1)
                s_enz[enz][b0:b1] = torch.sigmoid(l).cpu().numpy()
            sp = torch.zeros((b1 - b0, 1), dtype=torch.float32, device=device)
            bias = apo1.species_proj(sp)
            s_apo1[b0:b1] = torch.sigmoid(apo1.head(shared + bias).squeeze(-1)).cpu().numpy()
    elapsed = time.time() - t0
    logger.info("[%s]   scored %d rows in %.1f s (%.0f/s)", chrom, n_chrom, elapsed, n_chrom / max(elapsed, 1e-9))

    out_df = pd.DataFrame({
        "chrom": cand["chrom"].iloc[cand_idx].values,
        "pos": cand["pos"].iloc[cand_idx].values,
        "strand": cand["strand"].iloc[cand_idx].values,
        "gene": cand["gene"].iloc[cand_idx].values,
        "score_binary": s_binary,
        "score_A3A": s_enz["A3A"],
        "score_A3B": s_enz["A3B"],
        "score_A3G": s_enz["A3G"],
        "score_A3A_A3G": s_enz["A3A_A3G"],
        "score_Neither": s_enz["Neither"],
        "score_apobec1": s_apo1,
        "valid": valid_chrom,
    })
    score_cols = [c for c in out_df.columns if c.startswith("score_")]
    out_df.loc[~valid_chrom, score_cols] = np.nan
    out_path = out_dir / f"{chrom}_scores.parquet"
    out_df.to_parquet(out_path, index=False)
    logger.info("[%s]   wrote %s (%.1f MB)", chrom, out_path, out_path.stat().st_size / 1e6)
    # KEEP RNA-FM npz on disk as reusable embedding cache. Move to a kept/ subdir
    # so we know it has been scored and don't re-score it.
    kept_dir = npz_path.parent / "kept"
    kept_dir.mkdir(exist_ok=True)
    new_npz = kept_dir / npz_path.name
    npz_size_gb = npz_path.stat().st_size / 1e9
    npz_path.rename(new_npz)
    logger.info("[%s]   moved %s -> %s (%.2f GB preserved)", chrom, npz_path.name, new_npz, npz_size_gb)
    return {"chrom": chrom, "n": n_chrom, "valid": int(valid_chrom.sum()),
            "scored_path": str(out_path), "kept_npz": str(new_npz), "preserved_gb": npz_size_gb}


def is_chrom_complete(rnafm_cand: pd.DataFrame, chrom: str, progress_log: Path) -> bool:
    """Rely on the progress log to know when a chrom has completed (rate=0 for that chrom,
    or next-chrom rows appear). Simpler: check the compute_rnafm script's resume semantics —
    the per-chrom npz has `done` mask; chrom is complete when done.sum() == len(done)."""
    # Just check if the npz exists and all `done` bits are True
    npz_path = Path.home() / "data" / "panel" / "rnafm" / f"rnafm_{chrom}.npz"
    if not npz_path.exists():
        return False
    try:
        z = np.load(npz_path, mmap_mode="r")
        done_mask = z["done"]
        return bool(done_mask.all())
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rnafm-dir", type=Path, default=Path.home() / "data/panel/rnafm")
    ap.add_argument("--candidates", type=Path, default=Path.home() / "data/panel/candidates_cache_aligned.parquet")
    ap.add_argument("--hand40", type=Path, default=Path.home() / "data/panel/hand40_cache_aligned.npy")
    ap.add_argument("--hand-valid", type=Path, default=Path.home() / "data/panel/valid_hand_cache_aligned.npy")
    ap.add_argument("--rnafm-cand", type=Path, default=Path.home() / "data/panel/candidates_all.parquet")
    ap.add_argument("--phase3", type=Path, default=Path.home() / "data/models/phase3_mfe_only.pt")
    ap.add_argument("--apobec1", type=Path, default=Path.home() / "data/models/apobec1_head_mfe_only.pt")
    ap.add_argument("--out-dir", type=Path, default=Path.home() / "data/panel/scored_chroms")
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--poll-interval", type=int, default=300, help="Seconds between polls")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(Path.home() / "logs/score_watcher.log", mode="a")])
    logger = logging.getLogger(__name__)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info("Device: %s", device)

    # Load static artifacts
    logger.info("Loading cache-aligned candidates ...")
    cand = pd.read_parquet(args.candidates)
    logger.info("  %d positions", len(cand))
    logger.info("Loading hand40 ...")
    hand40 = np.load(args.hand40)
    valid_hand = np.load(args.hand_valid)
    assert hand40.shape[0] == len(cand)
    assert float(np.abs(hand40[:, 24:31]).max()) == 0.0, "struct_delta not zeroed!"
    logger.info("Loading rnafm-cand keys ...")
    rnafm_cand = pd.read_parquet(args.rnafm_cand)

    # Load models
    logger.info("Loading models ...")
    phase3 = Phase3Model()
    phase3.load_state_dict(torch.load(args.phase3, weights_only=False, map_location="cpu"))
    phase3.to(device)
    apo1 = APOBEC1Head(D_SHARED)
    apo1.load_state_dict(torch.load(args.apobec1, weights_only=False, map_location="cpu"))
    apo1.to(device)

    done = set(p.stem.replace("_scores", "") for p in args.out_dir.glob("*_scores.parquet"))
    logger.info("Already scored chroms: %s", sorted(done))

    # Poll loop
    while True:
        remaining = [c for c in ALL_CHROMS if c not in done]
        if not remaining:
            logger.info("All 24 chromosomes scored. Merging final parquet ...")
            merge_final(args.out_dir)
            logger.info("DONE.")
            return 0
        for chrom in remaining:
            npz_path = args.rnafm_dir / f"rnafm_{chrom}.npz"
            if not npz_path.exists():
                continue
            if not is_chrom_complete(rnafm_cand, chrom, Path.home() / "logs/rnafm_progress.log"):
                continue
            try:
                score_chrom(chrom, npz_path, cand, rnafm_cand, hand40, valid_hand,
                            phase3, apo1, device, args.batch, args.out_dir)
                done.add(chrom)
            except Exception as ex:
                logger.error("Chrom %s scoring failed: %s", chrom, ex, exc_info=True)
                # Don't add to done — will retry on next poll
                continue
        # Sleep and poll again
        remaining2 = [c for c in ALL_CHROMS if c not in done]
        if remaining2:
            logger.info("Waiting %d s for more chroms to complete (%d done, %d remaining) ...",
                        args.poll_interval, len(done), len(remaining2))
            time.sleep(args.poll_interval)


def merge_final(scored_dir: Path):
    logger = logging.getLogger(__name__)
    dfs = []
    for chrom in ALL_CHROMS:
        p = scored_dir / f"{chrom}_scores.parquet"
        if p.exists():
            dfs.append(pd.read_parquet(p))
    if not dfs:
        logger.warning("No scored chroms found in %s", scored_dir)
        return
    merged = pd.concat(dfs, ignore_index=True)
    out = Path.home() / "data/panel/panel_scores_exome_mfe_only.parquet"
    merged.to_parquet(out, index=False)
    logger.info("Merged %d chroms into %s (%.1f MB, %d rows)",
                len(dfs), out, out.stat().st_size / 1e6, len(merged))


if __name__ == "__main__":
    main()
