#!/usr/bin/env python3
"""Score the full 1.69M ClinVar C>T variant set with v4_cds Phase3 + APOBEC1 heads.

Inputs (all already cached):
  data/processed/multi_enzyme/embeddings/rnafm_clinvar.pt   (pooled_orig + pooled_edited, 1.69M x 640)
  data/processed/clinvar_features_cache.npz                  (hand_46, 1.69M x 46; first 40 = v4-compat)
  experiments/multi_enzyme/outputs/v4_cds_unbiased/phase3_v4_cds.pt
  experiments/multi_enzyme/outputs/apobec1_head_v4_cds/apobec1_head_v4_cds.pt

Output:
  experiments/apobec3a/outputs/clinvar_v4_scored/clinvar_scores_v4.parquet
    columns: site_id, chrom, pos, strand,
             score_binary, score_A3A, score_A3B, score_A3G, score_A3A_A3G, score_apobec1_v4_cds
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path("/Users/shaharharel/Documents/github/edit-rna-apobec")
RNAFM_CACHE = ROOT / "data/processed/multi_enzyme/embeddings/rnafm_clinvar.pt"
HAND_CACHE = ROOT / "data/processed/clinvar_features_cache.npz"
PHASE3_PT = ROOT / "experiments/multi_enzyme/outputs/v4_cds_unbiased/phase3_v4_cds.pt"
APOBEC1_PT = ROOT / "experiments/multi_enzyme/outputs/apobec1_head_v4_cds/apobec1_head_v4_cds.pt"
OUT_DIR = ROOT / "experiments/apobec3a/outputs/clinvar_v4_scored"
OUT_PARQUET = OUT_DIR / "clinvar_scores_v4.parquet"

D_INPUT, D_SHARED = 1320, 128
ENZYMES = ["A3A", "A3B", "A3G", "A3A_A3G"]
N_ENZYME_CLS = 5
STRUCT_DELTA_START = 640 + 640 + 24  # 1304
STRUCT_DELTA_END = STRUCT_DELTA_START + 7  # 1311
BATCH = 16384

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    stream=sys.stdout)
log = logging.getLogger(__name__)


class Phase3Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_encoder = nn.Sequential(
            nn.Linear(D_INPUT, 256), nn.GELU(), nn.Dropout(0.3), nn.LayerNorm(256),
            nn.Linear(256, D_SHARED), nn.GELU(), nn.Dropout(0.2),
        )
        self.binary_head = nn.Linear(D_SHARED, 1)
        self.enzyme_adapters = nn.ModuleDict({
            e: nn.Sequential(nn.Linear(D_SHARED, 32), nn.GELU(), nn.Linear(32, 1))
            for e in ENZYMES
        })
        self.enzyme_classifier = nn.Sequential(
            nn.Linear(D_SHARED, 64), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(64, N_ENZYME_CLS),
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


def parse_site_id(sid: str) -> tuple[str, int, str]:
    # 'clinvar_chr1_69240_+'  -> (chr1, 69240, +)
    parts = sid.split("_")
    if len(parts) >= 4:
        chrom = parts[1]
        pos = int(parts[2])
        strand = parts[3]
        return chrom, pos, strand
    return ("", -1, "")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
    log.info("Device: %s", device)

    # Load RNA-FM cache
    log.info("Loading RNA-FM cache: %s", RNAFM_CACHE)
    rnafm = torch.load(RNAFM_CACHE, weights_only=False, map_location="cpu")
    rnafm_site_ids = rnafm["site_ids"]
    pooled_orig = rnafm["pooled_orig"].numpy()
    pooled_edited = rnafm["pooled_edited"].numpy()
    n_rnafm = len(rnafm_site_ids)
    log.info("  rnafm: %d sites, pooled_orig %s, pooled_edited %s",
             n_rnafm, pooled_orig.shape, pooled_edited.shape)

    # Load hand features cache
    log.info("Loading hand features cache: %s", HAND_CACHE)
    hand = np.load(HAND_CACHE, allow_pickle=True)
    hand_site_ids = hand["site_ids"]
    hand_46 = hand["hand_46"]
    n_hand = len(hand_site_ids)
    log.info("  hand: %d sites, hand_46 %s", n_hand, hand_46.shape)

    assert n_rnafm == n_hand, f"rnafm n={n_rnafm} != hand n={n_hand}"

    # Verify alignment by site_id
    if list(rnafm_site_ids[:5]) == list(hand_site_ids[:5]):
        log.info("  site_ids aligned (first 5 match)")
    else:
        log.warning("  site_ids first 5 do NOT match — building alignment map")
        sid_to_hand_idx = {sid: i for i, sid in enumerate(hand_site_ids)}
        order = np.array([sid_to_hand_idx.get(sid, -1) for sid in rnafm_site_ids])
        if (order == -1).any():
            n_missing = (order == -1).sum()
            log.warning("  %d / %d rnafm sites missing in hand", n_missing, n_rnafm)
        valid = order >= 0
        rnafm_site_ids = [rnafm_site_ids[i] for i in np.where(valid)[0]]
        pooled_orig = pooled_orig[valid]
        pooled_edited = pooled_edited[valid]
        hand_46 = hand_46[order[valid]]
        log.info("  realigned to %d sites", len(rnafm_site_ids))

    n = len(rnafm_site_ids)

    # Load v4 models
    log.info("Loading v4_cds Phase3: %s", PHASE3_PT)
    phase3 = Phase3Model().to(device).eval()
    phase3.load_state_dict(torch.load(PHASE3_PT, weights_only=False, map_location=device))

    log.info("Loading v4_cds APOBEC1: %s", APOBEC1_PT)
    apobec1 = APOBEC1Head(D_SHARED).to(device).eval()
    apobec1.load_state_dict(torch.load(APOBEC1_PT, weights_only=False, map_location=device))

    # Score in batches
    s_binary = np.zeros(n, dtype=np.float32)
    s_enz = {e: np.zeros(n, dtype=np.float32) for e in ENZYMES}
    s_apobec1 = np.zeros(n, dtype=np.float32)

    log.info("Scoring %d ClinVar variants in batches of %d", n, BATCH)
    n_batches = (n + BATCH - 1) // BATCH
    t_score = time.time()
    for bi in range(n_batches):
        b0 = bi * BATCH
        b1 = min(b0 + BATCH, n)
        x_orig = pooled_orig[b0:b1].astype(np.float32)
        x_edited = pooled_edited[b0:b1].astype(np.float32)
        x_delta = x_edited - x_orig
        x_hand = hand_46[b0:b1, :40].astype(np.float32)

        # Assemble 1320-d input + zero struct_delta slots
        X = np.concatenate([x_orig, x_delta, x_hand], axis=1)
        X[:, STRUCT_DELTA_START:STRUCT_DELTA_END] = 0.0
        xb = torch.from_numpy(X).to(device)

        with torch.no_grad():
            shared = phase3.shared_encoder(xb)
            s_binary[b0:b1] = torch.sigmoid(phase3.binary_head(shared).squeeze(-1)).cpu().numpy()
            for e in ENZYMES:
                logit = phase3.enzyme_adapters[e](shared).squeeze(-1)
                s_enz[e][b0:b1] = torch.sigmoid(logit).cpu().numpy()
            sp_vec = torch.zeros((b1 - b0, 1), dtype=torch.float32, device=device)
            bias = apobec1.species_proj(sp_vec)
            l1 = apobec1.head(shared + bias).squeeze(-1)
            s_apobec1[b0:b1] = torch.sigmoid(l1).cpu().numpy()

        if bi % 10 == 0 or bi + 1 == n_batches:
            el = time.time() - t_score
            rate = b1 / max(el, 1e-9)
            eta = (n - b1) / max(rate, 1e-9) / 60.0
            log.info("  batch %d/%d  rows=%d/%d  rate=%.0f/s  ETA=%.1f min",
                     bi + 1, n_batches, b1, n, rate, eta)

    log.info("Inference done in %.1f s", time.time() - t_score)

    # Parse site_id -> chrom, pos, strand
    log.info("Parsing site_ids ...")
    chroms, poses, strands = [], [], []
    for sid in rnafm_site_ids:
        c, p, s = parse_site_id(sid)
        chroms.append(c); poses.append(p); strands.append(s)

    out = pd.DataFrame({
        "site_id": rnafm_site_ids,
        "chrom": chroms,
        "pos": poses,
        "strand": strands,
        "score_binary": s_binary,
        "score_A3A": s_enz["A3A"],
        "score_A3B": s_enz["A3B"],
        "score_A3G": s_enz["A3G"],
        "score_A3A_A3G": s_enz["A3A_A3G"],
        "score_apobec1_v4_cds": s_apobec1,
    })
    out.to_parquet(OUT_PARQUET, index=False)
    log.info("Wrote %s (%.1f MB, %d rows)",
             OUT_PARQUET, OUT_PARQUET.stat().st_size / 1e6, len(out))
    log.info("Total wall time: %.1f min", (time.time() - t0) / 60.0)


if __name__ == "__main__":
    main()
