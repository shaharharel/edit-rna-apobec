#!/usr/bin/env python3
"""Score all cache-aligned CDS candidates with phase3_mfe_only + apobec1_head_mfe_only.

Runs on ai-gpu2 (V100). Inputs (all local to ai-gpu2):
    ~/data/panel/candidates_cache_aligned.parquet    (8.45 M rows, cache-aligned order)
    ~/data/panel/hand40_cache_aligned.npy            (8.45 M × 40 fp32, precomputed on Mac)
    ~/data/panel/valid_hand_cache_aligned.npy        (8.45 M bool)
    ~/data/panel/candidates_all.parquet              (28.6 M rows — RNA-FM key set)
    ~/data/panel/rnafm/rnafm_chr{N}.npz              (per-chrom: pos, strand, orig[N,640] fp16, delta[N,640] fp16, valid)
    ~/data/models/phase3_mfe_only.pt                 (model)
    ~/data/models/apobec1_head_mfe_only.pt           (model)

Output (on ai-gpu2, transferred back to Mac):
    ~/data/panel/panel_scores_exome_mfe_only.parquet
        chrom, pos, strand, gene, trinuc,
        score_binary, score_A3A, score_A3B, score_A3G, score_A3A_A3G, score_Neither,
        score_apobec1, valid

Scoring: build per-position 1320-d input as
    [rnafm_orig(640)] + [rnafm_delta(640)] + [hand40 with struct_delta zeroed(40)]
then run phase3_mfe_only (binary + 5 enzyme adapters) and apobec1_head_mfe_only (species=0).

Usage on ai-gpu2:
    tmux new -d -s score_panel 'python3 score_panel_mfe_only.py --batch 4096 2>&1 | tee ~/logs/score_panel.log'
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

D_INPUT = 1320
D_SHARED = 128
ENZYMES = ["A3A", "A3B", "A3G", "A3A_A3G", "Neither"]
N_ENZYMES_CLS = 6

STRUCT_DELTA_START_ABS = 640 + 640 + 24  # 1304
STRUCT_DELTA_END_ABS = STRUCT_DELTA_START_ABS + 7  # 1311

ALL_CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]


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
        self.species_proj = nn.Sequential(
            nn.Linear(1, 16), nn.GELU(), nn.Linear(16, d_shared),
        )
        self.head = nn.Sequential(
            nn.Linear(d_shared, 32), nn.GELU(), nn.Linear(32, 1),
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=Path, default=Path.home() / "data/panel/candidates_cache_aligned.parquet")
    ap.add_argument("--hand40", type=Path, default=Path.home() / "data/panel/hand40_cache_aligned.npy")
    ap.add_argument("--hand-valid", type=Path, default=Path.home() / "data/panel/valid_hand_cache_aligned.npy")
    ap.add_argument("--rnafm-dir", type=Path, default=Path.home() / "data/panel/rnafm")
    ap.add_argument("--rnafm-cand", type=Path, default=Path.home() / "data/panel/candidates_all.parquet",
                    help="Parquet used by the RNA-FM run (for key alignment)")
    ap.add_argument("--phase3", type=Path, default=Path.home() / "data/models/phase3_mfe_only.pt")
    ap.add_argument("--apobec1", type=Path, default=Path.home() / "data/models/apobec1_head_mfe_only.pt")
    ap.add_argument("--out", type=Path, default=Path.home() / "data/panel/panel_scores_exome_mfe_only.parquet")
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=0, help="Smoke test: limit to first N positions")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info("Device: %s", device)
    if device.type == "cuda":
        logger.info("  GPU: %s", torch.cuda.get_device_name(0))

    logger.info("Loading candidates %s", args.candidates)
    cand = pd.read_parquet(args.candidates)
    n_all = len(cand)
    logger.info("  n=%d", n_all)

    logger.info("Loading hand40 %s", args.hand40)
    hand40 = np.load(args.hand40)
    valid_hand = np.load(args.hand_valid)
    assert hand40.shape == (n_all, 40), f"hand40 shape {hand40.shape} vs n_all={n_all}"
    logger.info("  valid_hand: %d/%d", int(valid_hand.sum()), n_all)

    # Sanity: verify struct_delta slots are zero
    sd = hand40[:, 24:31]
    assert float(np.abs(sd).max()) == 0.0, "struct_delta slots are not zeroed in hand40!"

    # === Load RNA-FM ===
    # The RNA-FM run used `candidates_all.parquet` (28.6 M rows). We need to match
    # each cache-aligned candidate to its row in the RNA-FM per-chrom npz.
    logger.info("Loading RNA-FM candidates (for key alignment)")
    rnafm_cand = pd.read_parquet(args.rnafm_cand)
    # Build (chrom, pos, strand) -> rnafm-row index
    rnafm_cand = rnafm_cand.reset_index(drop=True)
    key_to_rfidx = {}
    for i, r in enumerate(rnafm_cand.itertuples(index=False)):
        key_to_rfidx[(r.chrom, int(r.pos), r.strand)] = i

    # Allocate output RNA-FM matrix aligned to `cand`
    rnafm_orig = np.zeros((n_all, 640), dtype=np.float16)
    rnafm_delta = np.zeros((n_all, 640), dtype=np.float16)
    valid_rnafm = np.zeros(n_all, dtype=bool)
    n_missing_rnafm = 0

    # For each chromosome, load the per-chrom RNA-FM npz and scatter into the aligned arrays
    # The per-chrom npz keys are (pos, strand) within that chromosome, scattered over rnafm_cand's rows
    # So we need to look up each of our cand rows (c, p, s) -> rnafm_cand row -> npz(chrom)[within-chrom index]
    # Easier: build per-chrom (pos, strand) -> rfidx mapping from rnafm_cand, then per-chrom load the
    # npz's `pos` / `strand` / `orig` / `delta` / `valid`, joined on (pos, strand).

    for chrom in ALL_CHROMS:
        npz_path = args.rnafm_dir / f"rnafm_{chrom}.npz"
        if not npz_path.exists():
            logger.warning("  missing %s — skipping", npz_path)
            continue
        z = np.load(npz_path, allow_pickle=False)
        z_pos = z["pos"].astype(np.int64)
        z_strand = np.array([s.decode() if isinstance(s, bytes) else s for s in z["strand"]], dtype="U1")
        z_orig = z["orig"]  # (n_chrom, 640) fp16
        z_delta = z["delta"]
        z_valid = z["valid"].astype(bool)

        # Build mapping (pos, strand) -> z-index
        z_key_to_idx = {(int(p), s): i for i, (p, s) in enumerate(zip(z_pos, z_strand))}

        # Iterate cand rows for this chrom
        cand_mask = (cand["chrom"] == chrom).to_numpy()
        cand_idxs = np.where(cand_mask)[0]
        for ci in cand_idxs:
            key = (int(cand["pos"].iloc[ci]), cand["strand"].iloc[ci])
            zi = z_key_to_idx.get(key, None)
            if zi is None:
                n_missing_rnafm += 1
                continue
            if not z_valid[zi]:
                continue
            rnafm_orig[ci] = z_orig[zi]
            rnafm_delta[ci] = z_delta[zi]
            valid_rnafm[ci] = True
        logger.info("  %s: valid_rnafm so far: %d", chrom, int(valid_rnafm.sum()))

    logger.info("RNA-FM aligned: %d/%d valid (%d missing, %d with invalid flag)",
                int(valid_rnafm.sum()), n_all, n_missing_rnafm, n_all - int(valid_rnafm.sum()) - n_missing_rnafm)

    valid_all = valid_hand & valid_rnafm
    logger.info("Combined valid (hand AND rnafm): %d/%d", int(valid_all.sum()), n_all)

    # === Load models ===
    logger.info("Loading %s", args.phase3)
    phase3 = Phase3Model()
    phase3.load_state_dict(torch.load(args.phase3, weights_only=False, map_location="cpu"))
    phase3.to(device).eval()

    logger.info("Loading %s", args.apobec1)
    apo1 = APOBEC1Head(D_SHARED)
    apo1.load_state_dict(torch.load(args.apobec1, weights_only=False, map_location="cpu"))
    apo1.to(device).eval()

    # === Score in batches ===
    if args.limit > 0:
        idx_pool = np.arange(min(args.limit, n_all))
    else:
        idx_pool = np.arange(n_all)

    s_binary = np.zeros(n_all, dtype=np.float32)
    s_enz = {enz: np.zeros(n_all, dtype=np.float32) for enz in ENZYMES}
    s_apobec1 = np.zeros(n_all, dtype=np.float32)

    t0 = time.time()
    n_batches = (len(idx_pool) + args.batch - 1) // args.batch
    logger.info("Scoring %d rows in %d batches of %d", len(idx_pool), n_batches, args.batch)
    for bi in range(n_batches):
        b0 = bi * args.batch
        b1 = min(b0 + args.batch, len(idx_pool))
        rows = idx_pool[b0:b1]
        # Build X for these rows
        x_orig = rnafm_orig[rows].astype(np.float32)
        x_delta = rnafm_delta[rows].astype(np.float32)
        x_hand = hand40[rows]
        X = np.concatenate([x_orig, x_delta, x_hand], axis=1).astype(np.float32)
        # Safety: re-zero struct_delta slots (redundant)
        X[:, STRUCT_DELTA_START_ABS:STRUCT_DELTA_END_ABS] = 0.0
        xb = torch.from_numpy(X).to(device)
        with torch.no_grad():
            shared = phase3.shared_encoder(xb)
            b_logit = phase3.binary_head(shared).squeeze(-1)
            s_binary[rows] = torch.sigmoid(b_logit).cpu().numpy()
            for enz in ENZYMES:
                l = phase3.enzyme_adapters[enz](shared).squeeze(-1)
                s_enz[enz][rows] = torch.sigmoid(l).cpu().numpy()
            sp_vec = torch.zeros((len(rows), 1), dtype=torch.float32, device=device)
            bias = apo1.species_proj(sp_vec)
            l1 = apo1.head(shared + bias).squeeze(-1)
            s_apobec1[rows] = torch.sigmoid(l1).cpu().numpy()
        if bi % 50 == 0 or bi + 1 == n_batches:
            elapsed = time.time() - t0
            rate = (b1) / max(elapsed, 1e-9)
            eta = (len(idx_pool) - b1) / max(rate, 1e-9) / 60.0
            logger.info("  batch %d/%d  rows=%d/%d  rate=%.0f/s  ETA=%.1f min",
                        bi + 1, n_batches, b1, len(idx_pool), rate, eta)

    # === Write parquet ===
    logger.info("Writing %s", args.out)
    out = pd.DataFrame({
        "chrom": cand["chrom"].values,
        "pos": cand["pos"].values,
        "strand": cand["strand"].values,
        "gene": cand["gene"].values,
        "score_binary": s_binary,
        "score_A3A": s_enz["A3A"],
        "score_A3B": s_enz["A3B"],
        "score_A3G": s_enz["A3G"],
        "score_A3A_A3G": s_enz["A3A_A3G"],
        "score_Neither": s_enz["Neither"],
        "score_apobec1": s_apobec1,
        "valid": valid_all,
    })
    # Apply validity: mask invalid rows to NaN in score columns
    score_cols = [c for c in out.columns if c.startswith("score_")]
    out.loc[~valid_all, score_cols] = np.nan
    out.to_parquet(args.out, index=False)

    meta = {
        "n_candidates": n_all,
        "n_valid": int(valid_all.sum()),
        "n_valid_hand": int(valid_hand.sum()),
        "n_valid_rnafm": int(valid_rnafm.sum()),
        "runtime_min": (time.time() - t0) / 60.0,
        "model_phase3": str(args.phase3),
        "model_apobec1": str(args.apobec1),
        "regime": "mfe_only",
    }
    with open(args.out.with_suffix(".meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Done. Wrote %s (%.1f MB). Runtime: %.1f min",
                args.out, args.out.stat().st_size / 1e6, meta["runtime_min"])


if __name__ == "__main__":
    main()
