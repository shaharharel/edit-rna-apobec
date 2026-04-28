#!/usr/bin/env python3
"""Score all candidates with Phase3 + APOBEC1 heads.

Runs on ai-gpu2 after both RNA-FM (local) and ViennaRNA (synced in from ai-chem)
arrays are present. Builds the 1320-d input and writes per-head sigmoid scores.

Outputs:
  ~/data/panel/panel_scores.parquet   (chrom, pos, strand, region_type, gene,
                                        score_binary, score_A3A, score_A3B,
                                        score_A3G, score_A3A_A3G, score_Neither,
                                        score_apobec1, valid)

Usage:
  python3 score_panel.py \\
      --candidates ~/data/panel/candidates.parquet \\
      --orig ~/data/panel/rnafm_orig.npy \\
      --delta ~/data/panel/rnafm_delta.npy \\
      --hand40 ~/data/panel/hand40.npy \\
      --valid ~/data/panel/valid_mask.npy \\
      --phase3 ~/data/models/phase3_neural_full.pt \\
      --apobec1 ~/data/models/apobec1_head.pt \\
      --out ~/data/panel/panel_scores.parquet \\
      --batch 4096
"""
from __future__ import annotations

import argparse
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
EMB_DIM = 640


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
    def __init__(self, d_shared: int = 128):
        super().__init__()
        self.species_proj = nn.Sequential(
            nn.Linear(1, 16), nn.GELU(), nn.Linear(16, d_shared),
        )
        self.head = nn.Sequential(
            nn.Linear(d_shared, 32), nn.GELU(), nn.Linear(32, 1),
        )

    def forward(self, shared, species):
        bias = self.species_proj(species)
        return self.head(shared + bias).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True, type=Path)
    ap.add_argument("--orig", required=True, type=Path)
    ap.add_argument("--delta", required=True, type=Path)
    ap.add_argument("--hand40", required=True, type=Path)
    ap.add_argument("--valid", required=True, type=Path)
    ap.add_argument("--phase3", required=True, type=Path)
    ap.add_argument("--apobec1", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--batch", type=int, default=4096)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(__name__)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info("Device: %s", device)

    logger.info("Loading candidates ...")
    cand = pd.read_parquet(args.candidates)
    n = len(cand)
    logger.info("  %d rows", n)

    logger.info("Memory-mapping feature arrays ...")
    orig = np.load(args.orig, mmap_mode="r")
    delta = np.load(args.delta, mmap_mode="r")
    hand40 = np.load(args.hand40, mmap_mode="r")
    valid = np.load(args.valid)
    for name, arr in [("orig", orig), ("delta", delta), ("hand40", hand40)]:
        logger.info("  %s: %s %s", name, arr.shape, arr.dtype)
    assert orig.shape[0] == n == delta.shape[0] == hand40.shape[0] == valid.shape[0], \
        f"shape mismatch: {orig.shape} {delta.shape} {hand40.shape} {valid.shape} vs {n}"

    logger.info("Loading Phase3Model ...")
    ckpt = torch.load(args.phase3, weights_only=False, map_location="cpu")
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model = Phase3Model()
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        logger.warning("  missing: %s", missing)
        logger.warning("  unexpected: %s", unexpected)
    model = model.eval().to(device)

    logger.info("Loading APOBEC1Head ...")
    a1 = APOBEC1Head()
    a1.load_state_dict(torch.load(args.apobec1, weights_only=False, map_location="cpu"))
    a1 = a1.eval().to(device)

    HEADS = ["binary"] + ENZYMES + ["apobec1"]
    scores = {h: np.zeros(n, dtype=np.float32) for h in HEADS}

    t0 = time.time()
    B = args.batch
    with torch.no_grad():
        for i in range(0, n, B):
            j = min(i + B, n)
            # Build input on-the-fly (fp32)
            xb = np.empty((j - i, D_INPUT), dtype=np.float32)
            xb[:, :EMB_DIM] = orig[i:j].astype(np.float32)
            xb[:, EMB_DIM:2 * EMB_DIM] = delta[i:j].astype(np.float32)
            xb[:, 2 * EMB_DIM:] = hand40[i:j].astype(np.float32)
            # Zero out invalid rows so they don't propagate NaNs
            mv = valid[i:j]
            if not mv.all():
                xb[~mv] = 0.0
            xt = torch.from_numpy(xb).to(device)
            shared = model.shared_encoder(xt)
            scores["binary"][i:j] = torch.sigmoid(
                model.binary_head(shared).squeeze(-1)
            ).cpu().numpy()
            for enz in ENZYMES:
                scores[enz][i:j] = torch.sigmoid(
                    model.enzyme_adapters[enz](shared).squeeze(-1)
                ).cpu().numpy()
            sp = torch.zeros((j - i, 1), dtype=torch.float32, device=device)
            scores["apobec1"][i:j] = torch.sigmoid(a1(shared, sp)).cpu().numpy()
            if (i // B) % 50 == 0:
                logger.info(
                    "  %d/%d (%.2f%%) elapsed %.1f min",
                    j, n, 100 * j / n, (time.time() - t0) / 60,
                )

    logger.info("Assembling output dataframe ...")
    out = cand.copy()
    for h in HEADS:
        out[f"score_{h}"] = scores[h]
    out["valid"] = valid
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    logger.info("Wrote %s (%.1f MB)", args.out, args.out.stat().st_size / 1e6)
    logger.info("Total: %.1f min", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
