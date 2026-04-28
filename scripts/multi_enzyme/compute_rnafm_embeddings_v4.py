#!/usr/bin/env python
"""Generate RNA-FM embeddings for v4 sites (cancer_matched + cds_unbiased).

Loads existing v3 RNA-FM embeddings (from multi_enzyme/embeddings/rnafm_pooled_v3.pt)
and only computes new entries for v4 negatives. Saves a single .npz file per
version with {site_ids: array, pooled: float32 (N,640), pooled_edited: float32}.

Output:
    data/processed/embeddings/rnafm_v4_cancer_matched.npz
    data/processed/embeddings/rnafm_v4_cds_unbiased.npz

Usage:
    conda run -n quris python scripts/multi_enzyme/compute_rnafm_embeddings_v4.py
    # optional: --version cancer_matched   to do only one
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ME_DIR = PROJECT_ROOT / "data/processed/multi_enzyme"
EMB_DIR = PROJECT_ROOT / "data/processed/embeddings"

# v3 cached embeddings (per-site dicts)
V3_POOLED = ME_DIR / "embeddings/rnafm_pooled_v3.pt"
V3_POOLED_ED = ME_DIR / "embeddings/rnafm_pooled_edited_v3.pt"
V3_IDS = ME_DIR / "embeddings/rnafm_site_ids_v3.json"

CENTER = 100
BATCH_SIZE = 8
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


def load_v3_embeddings() -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    pooled: Dict[str, torch.Tensor] = {}
    pooled_ed: Dict[str, torch.Tensor] = {}
    if V3_POOLED.exists():
        obj = torch.load(V3_POOLED, map_location="cpu", weights_only=True)
        if isinstance(obj, dict):
            pooled = {str(k): v for k, v in obj.items()}
        else:
            ids = json.load(open(V3_IDS)) if V3_IDS.exists() else []
            pooled = {str(s): obj[i] for i, s in enumerate(ids)}
    if V3_POOLED_ED.exists():
        obj = torch.load(V3_POOLED_ED, map_location="cpu", weights_only=True)
        if isinstance(obj, dict):
            pooled_ed = {str(k): v for k, v in obj.items()}
        else:
            ids = json.load(open(V3_IDS)) if V3_IDS.exists() else []
            pooled_ed = {str(s): obj[i] for i, s in enumerate(ids)}
    return pooled, pooled_ed


def embed_batch(model, batch_converter, seqs_batch):
    data = [(f"seq_{i}", seq) for i, seq in enumerate(seqs_batch)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(DEVICE)
    with torch.no_grad():
        results = model(tokens, repr_layers=[12])
    emb = results["representations"][12]
    pooled = emb[:, 1:-1, :].mean(dim=1)
    return pooled.cpu()


def compute_for_version(version: str, model, batch_converter,
                         pooled_cache: Dict[str, torch.Tensor],
                         pooled_ed_cache: Dict[str, torch.Tensor]) -> None:
    seqs_path = ME_DIR / f"multi_enzyme_sequences_v4_{version}.json"
    out_path = EMB_DIR / f"rnafm_v4_{version}.npz"
    if not seqs_path.exists():
        logger.warning("Skipping v4_%s: %s missing", version, seqs_path)
        return

    with open(seqs_path) as f:
        seqs = json.load(f)
    site_ids = list(seqs.keys())
    needed = [(sid, seqs[sid]) for sid in site_ids
              if sid not in pooled_cache or sid not in pooled_ed_cache]
    needed = [(sid, s) for sid, s in needed if len(s) == 201]
    logger.info("v4_%s: %d total sites, need to compute %d new RNA-FM embeddings",
                version, len(site_ids), len(needed))

    if needed:
        t0 = time.time()
        for i in range(0, len(needed), BATCH_SIZE):
            batch = needed[i:i + BATCH_SIZE]
            sids = [sid for sid, _ in batch]
            origs = [s.upper().replace("T", "U") for _, s in batch]
            eds = []
            for s in origs:
                lst = list(s)
                if lst[CENTER] == "C":
                    lst[CENTER] = "U"
                eds.append("".join(lst))
            p_o = embed_batch(model, batch_converter, origs)
            p_e = embed_batch(model, batch_converter, eds)
            for j, sid in enumerate(sids):
                pooled_cache[sid] = p_o[j]
                pooled_ed_cache[sid] = p_e[j]
            done = i + len(batch)
            if (i // BATCH_SIZE + 1) % 25 == 0 or done == len(needed):
                elapsed = time.time() - t0
                rate = done / elapsed
                rem = (len(needed) - done) / max(rate, 1e-6)
                logger.info("  %d/%d (%.1f/sec, ~%.1fm left)",
                            done, len(needed), rate, rem / 60)

    # Build output for the v4 set
    out_ids = []
    p_arr = []
    p_ed_arr = []
    missing = 0
    for sid in site_ids:
        if sid in pooled_cache and sid in pooled_ed_cache:
            out_ids.append(sid)
            p_arr.append(pooled_cache[sid].numpy())
            p_ed_arr.append(pooled_ed_cache[sid].numpy())
        else:
            missing += 1
    if missing:
        logger.warning("  %d sites missing embeddings (probably <201 nt)", missing)

    p_arr = np.stack(p_arr).astype(np.float32)
    p_ed_arr = np.stack(p_ed_arr).astype(np.float32)
    sids_arr = np.array(out_ids, dtype=object)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, site_ids=sids_arr, pooled=p_arr, pooled_edited=p_ed_arr)
    logger.info("Saved %s (%d entries, pooled=%s, pooled_edited=%s)",
                out_path, len(out_ids), p_arr.shape, p_ed_arr.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=["cancer_matched", "cds_unbiased", "both"], default="both")
    args = ap.parse_args()

    logger.info("Loading v3 RNA-FM caches ...")
    pooled, pooled_ed = load_v3_embeddings()
    logger.info("v3 cache: %d pooled, %d pooled_edited", len(pooled), len(pooled_ed))

    logger.info("Loading RNA-FM model on %s ...", DEVICE)
    import fm
    model, alphabet = fm.pretrained.rna_fm_t12()
    model = model.eval().to(DEVICE)
    batch_converter = alphabet.get_batch_converter()
    logger.info("RNA-FM ready.")

    versions = ["cancer_matched", "cds_unbiased"] if args.version == "both" else [args.version]
    for v in versions:
        compute_for_version(v, model, batch_converter, pooled, pooled_ed)
    logger.info("Done.")


if __name__ == "__main__":
    main()
