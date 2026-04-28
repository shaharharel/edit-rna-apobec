#!/usr/bin/env python3
"""Compute structure cache, RNA-FM embeddings, hand features, and loop-position CSV
for APOBEC1 training sites (mouse mm9).

Inputs (produced by build_apobec1_dataset.py):
  - data/processed/apobec1/apobec1_v1_with_negatives.csv
  - data/processed/apobec1/apobec1_v1_sequences.json

Outputs:
  - data/processed/apobec1/structure_cache_apobec1_v1.npz
  - data/processed/apobec1/rnafm_apobec1_v1.pt              (dict: site_id -> original 640-d)
  - data/processed/apobec1/rnafm_apobec1_v1_edited.pt       (dict: site_id -> edited 640-d)
  - data/processed/apobec1/apobec1_hand40.npy               (N × 40)
  - data/processed/apobec1/loop_position_apobec1_v1.csv

Usage:
    conda run -n quris python scripts/multi_enzyme/compute_apobec1_features.py [--stage structure|rnafm|hand|loop|all]
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.apobec_feature_extraction import (  # noqa: E402
    _extract_loop_geometry,
    build_hand_features,
    LOOP_FEATURE_COLS,
)

PROC_DIR = PROJECT_ROOT / "data" / "processed" / "apobec1"
SPLITS_CSV = PROC_DIR / "apobec1_v1_with_negatives.csv"
SEQS_JSON = PROC_DIR / "apobec1_v1_sequences.json"

STRUCT_CACHE = PROC_DIR / "structure_cache_apobec1_v1.npz"
RNAFM_ORIG = PROC_DIR / "rnafm_apobec1_v1.pt"
RNAFM_EDITED = PROC_DIR / "rnafm_apobec1_v1_edited.pt"
HAND40_NPY = PROC_DIR / "apobec1_hand40.npy"
LOOP_CSV = PROC_DIR / "loop_position_apobec1_v1.csv"

CENTER = 100
SEQ_LEN = 201

logger = logging.getLogger(__name__)


def _select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = _select_device()


# ---------------------------------------------------------------------------
# Structure cache
# ---------------------------------------------------------------------------

def _compute_structure_features_full(seq: str, temperature: float = 37.0) -> dict:
    """Replica of scripts/multi_enzyme/generate_structure_cache.compute_structure_features."""
    import RNA

    s = seq.upper().replace("T", "U")
    n = len(s)
    md = RNA.md()
    md.temperature = temperature
    fc = RNA.fold_compound(s, md)
    mfe_structure, mfe = fc.mfe()
    _, pf_energy = fc.pf()
    bpp_raw = np.array(fc.bpp())
    bpp = bpp_raw[1: n + 1, 1: n + 1]
    pairing_prob = np.clip(np.sum(bpp, axis=0) + np.sum(bpp, axis=1), 0, 1)
    accessibility = 1.0 - pairing_prob
    entropy = np.zeros(n)
    for i in range(n):
        probs = bpp[i, :]
        probs = probs[probs > 1e-10]
        if len(probs) > 0:
            unpaired = max(0, 1.0 - np.sum(probs))
            if unpaired > 1e-10:
                probs = np.append(probs, unpaired)
            entropy[i] = -np.sum(probs * np.log2(probs + 1e-10))
    return {
        "pairing_prob": pairing_prob.astype(np.float32),
        "accessibility": accessibility.astype(np.float32),
        "entropy": entropy.astype(np.float32),
        "mfe": float(mfe),
        "dot_bracket": mfe_structure,
    }


def _compute_site_structure(seq: str) -> dict | None:
    if not seq or len(seq) < 10:
        return None
    center = len(seq) // 2
    feat = _compute_structure_features_full(seq)
    ed_list = list(seq)
    if center < len(ed_list) and ed_list[center].upper() == "C":
        ed_list[center] = "U"
    feat_ed = _compute_structure_features_full("".join(ed_list))

    window = 10
    start = max(0, center - window)
    end = min(len(seq), center + window + 1)
    dp = feat_ed["pairing_prob"] - feat["pairing_prob"]
    da = feat_ed["accessibility"] - feat["accessibility"]
    de = feat_ed["entropy"] - feat["entropy"]
    delta = np.zeros(7, dtype=np.float32)
    delta[0] = dp[center]
    delta[1] = da[center]
    delta[2] = de[center]
    delta[3] = feat_ed["mfe"] - feat["mfe"]
    delta[4] = np.mean(dp[start:end])
    delta[5] = np.mean(da[start:end])
    delta[6] = np.std(dp[start:end])
    return {
        "delta_features": delta,
        "mfe": feat["mfe"],
        "mfe_edited": feat_ed["mfe"],
        "dot_bracket": feat["dot_bracket"],
    }


def stage_structure(sequences: dict) -> None:
    logger.info("Computing ViennaRNA structure cache for %d sites", len(sequences))
    results = {}
    t0 = time.time()
    errors = 0
    for i, (sid, seq) in enumerate(sequences.items(), 1):
        try:
            res = _compute_site_structure(seq)
            if res is None:
                errors += 1
                continue
            results[sid] = res
        except Exception as exc:
            errors += 1
            if errors <= 5:
                logger.warning("Structure error %s: %s", sid, exc)
        if i % 200 == 0:
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed else 0
            logger.info("  %d/%d (%.1f sites/sec)", i, len(sequences), rate)

    sids = np.array(list(results.keys()), dtype=object)
    delta_arr = np.array([results[s]["delta_features"] for s in sids], dtype=np.float32)
    mfes = np.array([results[s]["mfe"] for s in sids], dtype=np.float32)
    mfes_ed = np.array([results[s]["mfe_edited"] for s in sids], dtype=np.float32)
    np.savez_compressed(
        STRUCT_CACHE,
        site_ids=sids,
        delta_features=delta_arr,
        mfes=mfes,
        mfes_edited=mfes_ed,
    )
    logger.info("Wrote %s (%d sites, %d errors)", STRUCT_CACHE, len(sids), errors)


# ---------------------------------------------------------------------------
# Loop positions
# ---------------------------------------------------------------------------

def stage_loop(sequences: dict) -> None:
    import RNA
    logger.info("Computing loop geometry for %d sites", len(sequences))
    rows = []
    t0 = time.time()
    for i, (sid, seq) in enumerate(sequences.items(), 1):
        try:
            s = seq.upper().replace("T", "U")
            md = RNA.md(); md.temperature = 37.0
            fc = RNA.fold_compound(s, md)
            db, _mfe = fc.mfe()
            geom = _extract_loop_geometry(db, CENTER)
            row = {"site_id": sid}
            for j, col in enumerate(LOOP_FEATURE_COLS):
                row[col] = float(geom[j])
            rows.append(row)
        except Exception as exc:
            logger.warning("Loop geom error %s: %s", sid, exc)
        if i % 500 == 0:
            logger.info("  loop %d/%d (%.1f sec)", i, len(sequences), time.time() - t0)
    df = pd.DataFrame(rows)
    df.to_csv(LOOP_CSV, index=False)
    logger.info("Wrote %s (%d rows)", LOOP_CSV, len(df))


# ---------------------------------------------------------------------------
# Hand features
# ---------------------------------------------------------------------------

def stage_hand(sequences: dict) -> None:
    if not STRUCT_CACHE.exists():
        logger.error("Structure cache missing: run --stage structure first")
        sys.exit(1)
    if not LOOP_CSV.exists():
        logger.error("Loop CSV missing: run --stage loop first")
        sys.exit(1)

    sc = np.load(STRUCT_CACHE, allow_pickle=True)
    struct_map = {sid: sc["delta_features"][i] for i, sid in enumerate(sc["site_ids"])}
    loop_df = pd.read_csv(LOOP_CSV).drop_duplicates(subset=["site_id"]).set_index("site_id")

    splits = pd.read_csv(SPLITS_CSV)
    site_ids = splits["site_id"].tolist()

    hand = build_hand_features(site_ids, sequences, struct_map, loop_df)
    np.save(HAND40_NPY, hand)
    logger.info("Wrote %s (shape=%s)", HAND40_NPY, hand.shape)


# ---------------------------------------------------------------------------
# RNA-FM
# ---------------------------------------------------------------------------

def stage_rnafm(sequences: dict) -> None:
    import fm  # noqa

    logger.info("Loading RNA-FM...")
    model, alphabet = fm.pretrained.rna_fm_t12()
    model = model.eval().to(DEVICE)
    batch_converter = alphabet.get_batch_converter()
    logger.info("RNA-FM loaded on %s", DEVICE)

    # Align with splits CSV ordering
    splits = pd.read_csv(SPLITS_CSV)
    site_ids = splits["site_id"].tolist()
    ordered_seqs = [sequences[s] for s in site_ids]
    n = len(ordered_seqs)

    pooled_orig = torch.zeros(n, 640)
    pooled_edited = torch.zeros(n, 640)
    BATCH = 16

    t0 = time.time()
    for i in range(0, n, BATCH):
        end = min(i + BATCH, n)
        batch_seqs = ordered_seqs[i:end]
        # Original
        data = [(f"seq_{k}", s) for k, s in enumerate(batch_seqs)]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(DEVICE)
        with torch.no_grad():
            out = model(tokens, repr_layers=[12])
        emb = out["representations"][12]
        pooled = emb[:, 1:-1, :].mean(dim=1).cpu()
        pooled_orig[i:end] = pooled
        if DEVICE == "mps":
            torch.mps.synchronize()

        # Edited (C→U at center)
        ed_seqs = []
        for s in batch_seqs:
            s_list = list(s)
            if s_list[CENTER] == "C":
                s_list[CENTER] = "U"
            ed_seqs.append("".join(s_list))
        data_ed = [(f"seq_{k}", s) for k, s in enumerate(ed_seqs)]
        _, _, tokens_ed = batch_converter(data_ed)
        tokens_ed = tokens_ed.to(DEVICE)
        with torch.no_grad():
            out_ed = model(tokens_ed, repr_layers=[12])
        emb_ed = out_ed["representations"][12]
        pooled_ed = emb_ed[:, 1:-1, :].mean(dim=1).cpu()
        pooled_edited[i:end] = pooled_ed
        if DEVICE == "mps":
            torch.mps.synchronize()
            torch.mps.empty_cache()

        if i // BATCH % 10 == 0:
            logger.info("  RNA-FM %d/%d (%.1fs)", end, n, time.time() - t0)

    # Save as site_id -> tensor dict (matching v3 format)
    orig_dict = {sid: pooled_orig[i] for i, sid in enumerate(site_ids)}
    edited_dict = {sid: pooled_edited[i] for i, sid in enumerate(site_ids)}
    torch.save(orig_dict, RNAFM_ORIG)
    torch.save(edited_dict, RNAFM_EDITED)
    logger.info("Wrote %s, %s (n=%d)", RNAFM_ORIG, RNAFM_EDITED, n)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["structure", "rnafm", "hand", "loop", "all"], default="all")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if not SPLITS_CSV.exists() or not SEQS_JSON.exists():
        logger.error("Run build_apobec1_dataset.py first (missing %s or %s)",
                     SPLITS_CSV, SEQS_JSON)
        sys.exit(1)

    with open(SEQS_JSON) as f:
        sequences = json.load(f)
    logger.info("Loaded %d sequences", len(sequences))

    if args.stage in ("structure", "all"):
        if STRUCT_CACHE.exists():
            logger.info("Structure cache already exists; skipping")
        else:
            stage_structure(sequences)

    if args.stage in ("loop", "all"):
        if LOOP_CSV.exists():
            logger.info("Loop CSV exists; skipping")
        else:
            stage_loop(sequences)

    if args.stage in ("rnafm", "all"):
        if RNAFM_ORIG.exists() and RNAFM_EDITED.exists():
            logger.info("RNA-FM embeddings exist; skipping")
        else:
            stage_rnafm(sequences)
        gc.collect()

    if args.stage in ("hand", "all"):
        if HAND40_NPY.exists():
            logger.info("Hand features exist; skipping")
        else:
            stage_hand(sequences)


if __name__ == "__main__":
    main()
