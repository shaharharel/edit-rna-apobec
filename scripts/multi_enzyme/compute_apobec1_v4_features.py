#!/usr/bin/env python3
"""Compute structure cache, RNA-FM, hand40, loop CSV for APOBEC1 v4 datasets.

Reuses positive site features from the v3 caches (positives are unchanged).
Only computes features for new v4 negatives.

Inputs:
  - data/raw/apobec1/v4/apobec1_<ver>_with_negatives.csv
  - data/raw/apobec1/v4/apobec1_<ver>_sequences.json
  - data/processed/apobec1/structure_cache_apobec1_v1.npz   (positives)
  - data/processed/apobec1/rnafm_apobec1_v1.pt               (positives)
  - data/processed/apobec1/rnafm_apobec1_v1_edited.pt        (positives)
  - data/processed/apobec1/loop_position_apobec1_v1.csv      (positives)

Outputs (per <ver> = v4_cancer or v4_cds):
  - data/processed/apobec1/v4/structure_cache_apobec1_<ver>.npz
  - data/processed/apobec1/v4/rnafm_apobec1_<ver>.pt
  - data/processed/apobec1/v4/rnafm_apobec1_<ver>_edited.pt
  - data/processed/apobec1/v4/apobec1_hand40_<ver>.npy
  - data/processed/apobec1/v4/loop_position_apobec1_<ver>.csv
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

DATA_DIR = PROJECT_ROOT / "data"
APOBEC1_PROC = DATA_DIR / "processed" / "apobec1"
V4_RAW = DATA_DIR / "raw" / "apobec1" / "v4"
V4_PROC = APOBEC1_PROC / "v4"
V4_PROC.mkdir(parents=True, exist_ok=True)

# v3 positive caches (reused)
V3_STRUCT = APOBEC1_PROC / "structure_cache_apobec1_v1.npz"
V3_RNAFM_O = APOBEC1_PROC / "rnafm_apobec1_v1.pt"
V3_RNAFM_E = APOBEC1_PROC / "rnafm_apobec1_v1_edited.pt"
V3_LOOP = APOBEC1_PROC / "loop_position_apobec1_v1.csv"

CENTER = 100

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = _select_device()
logger.info("Device: %s", DEVICE)


# ---------------------------------------------------------------------------
# Structure helpers (replicate compute_apobec1_features.py)
# ---------------------------------------------------------------------------

def _compute_structure_features_full(seq: str, temperature: float = 37.0) -> dict:
    import RNA
    s = seq.upper().replace("T", "U")
    n = len(s)
    md = RNA.md()
    md.temperature = temperature
    fc = RNA.fold_compound(s, md)
    mfe_structure, mfe = fc.mfe()
    _, _ = fc.pf()
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


def stage_structure(version: str, splits: pd.DataFrame, sequences: dict) -> Path:
    """Compute structure cache, reusing positives from v3."""
    out_path = V4_PROC / f"structure_cache_apobec1_{version}.npz"
    if out_path.exists():
        logger.info("Structure cache exists at %s; skipping", out_path)
        return out_path

    # Load existing v3 positive cache
    v3 = np.load(V3_STRUCT, allow_pickle=True)
    v3_map = {sid: i for i, sid in enumerate(v3["site_ids"])}

    site_ids = splits["site_id"].tolist()
    delta_arr = np.zeros((len(site_ids), 7), dtype=np.float32)
    mfes = np.zeros(len(site_ids), dtype=np.float32)
    mfes_ed = np.zeros(len(site_ids), dtype=np.float32)

    n_reuse = 0
    n_new = 0
    t0 = time.time()
    for i, sid in enumerate(site_ids):
        if sid in v3_map:
            j = v3_map[sid]
            delta_arr[i] = v3["delta_features"][j]
            mfes[i] = v3["mfes"][j]
            mfes_ed[i] = v3["mfes_edited"][j]
            n_reuse += 1
        else:
            seq = sequences.get(sid, "")
            res = _compute_site_structure(seq)
            if res is None:
                continue
            delta_arr[i] = res["delta_features"]
            mfes[i] = res["mfe"]
            mfes_ed[i] = res["mfe_edited"]
            n_new += 1
            if n_new % 50 == 0:
                logger.info("  [%s] structure new=%d (total i=%d) %.1fs",
                            version, n_new, i, time.time() - t0)

    np.savez_compressed(
        out_path,
        site_ids=np.array(site_ids, dtype=object),
        delta_features=delta_arr,
        mfes=mfes,
        mfes_edited=mfes_ed,
    )
    logger.info("[%s] structure cache: reused=%d new=%d -> %s",
                version, n_reuse, n_new, out_path)
    return out_path


def stage_loop(version: str, splits: pd.DataFrame, sequences: dict) -> Path:
    import RNA
    out_path = V4_PROC / f"loop_position_apobec1_{version}.csv"
    if out_path.exists():
        logger.info("Loop CSV exists at %s; skipping", out_path)
        return out_path

    v3_loop = pd.read_csv(V3_LOOP).drop_duplicates(subset=["site_id"])
    v3_lmap = v3_loop.set_index("site_id").to_dict(orient="index")

    rows = []
    n_reuse = 0
    n_new = 0
    t0 = time.time()
    for i, sid in enumerate(splits["site_id"].tolist()):
        if sid in v3_lmap:
            row = {"site_id": sid}
            for col in LOOP_FEATURE_COLS:
                row[col] = float(v3_lmap[sid].get(col, 0.0))
            rows.append(row)
            n_reuse += 1
            continue
        seq = sequences.get(sid, "")
        if not seq:
            continue
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
            n_new += 1
            if n_new % 50 == 0:
                logger.info("  [%s] loop new=%d %.1fs",
                            version, n_new, time.time() - t0)
        except Exception as exc:
            logger.warning("  [%s] loop error %s: %s", version, sid, exc)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    logger.info("[%s] loop CSV: reused=%d new=%d -> %s",
                version, n_reuse, n_new, out_path)
    return out_path


def stage_rnafm(version: str, splits: pd.DataFrame, sequences: dict) -> tuple[Path, Path]:
    """Compute RNA-FM, reusing positives from v3."""
    out_o = V4_PROC / f"rnafm_apobec1_{version}.pt"
    out_e = V4_PROC / f"rnafm_apobec1_{version}_edited.pt"
    if out_o.exists() and out_e.exists():
        logger.info("RNA-FM caches exist; skipping")
        return out_o, out_e

    v3_o = torch.load(V3_RNAFM_O, weights_only=False, map_location="cpu")
    v3_e = torch.load(V3_RNAFM_E, weights_only=False, map_location="cpu")

    # Find new site_ids that need computation
    site_ids = splits["site_id"].tolist()
    new_ids = [sid for sid in site_ids if sid not in v3_o or sid not in v3_e]
    logger.info("[%s] RNA-FM: %d new sites to compute (reuse %d positives)",
                version, len(new_ids), len(site_ids) - len(new_ids))

    out_orig = {}
    out_edit = {}
    # Reuse positives
    for sid in site_ids:
        if sid in v3_o and sid in v3_e:
            out_orig[sid] = v3_o[sid]
            out_edit[sid] = v3_e[sid]

    if new_ids:
        import fm
        logger.info("Loading RNA-FM ...")
        model, alphabet = fm.pretrained.rna_fm_t12()
        model = model.eval().to(DEVICE)
        batch_converter = alphabet.get_batch_converter()
        logger.info("RNA-FM loaded on %s", DEVICE)

        BATCH = 16
        n = len(new_ids)
        t0 = time.time()
        for i in range(0, n, BATCH):
            end = min(i + BATCH, n)
            batch_ids = new_ids[i:end]
            batch_seqs = [sequences[sid] for sid in batch_ids]

            # Original
            data = [(f"seq_{k}", s) for k, s in enumerate(batch_seqs)]
            _, _, tokens = batch_converter(data)
            tokens = tokens.to(DEVICE)
            with torch.no_grad():
                out = model(tokens, repr_layers=[12])
            emb = out["representations"][12]
            pooled = emb[:, 1:-1, :].mean(dim=1).cpu()

            # Edited
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

            for k, sid in enumerate(batch_ids):
                out_orig[sid] = pooled[k]
                out_edit[sid] = pooled_ed[k]

            if DEVICE == "mps":
                torch.mps.synchronize()
                torch.mps.empty_cache()
            if (i // BATCH) % 5 == 0:
                logger.info("  [%s] RNA-FM %d/%d (%.1fs)", version, end, n, time.time() - t0)

    torch.save(out_orig, out_o)
    torch.save(out_edit, out_e)
    logger.info("[%s] RNA-FM saved -> %s, %s", version, out_o, out_e)
    return out_o, out_e


def stage_hand(version: str, splits: pd.DataFrame, sequences: dict,
               struct_path: Path, loop_path: Path) -> Path:
    out_path = V4_PROC / f"apobec1_hand40_{version}.npy"
    if out_path.exists():
        logger.info("Hand40 exists at %s; skipping", out_path)
        return out_path

    sc = np.load(struct_path, allow_pickle=True)
    struct_map = {sid: sc["delta_features"][i] for i, sid in enumerate(sc["site_ids"])}
    loop_df = pd.read_csv(loop_path).drop_duplicates(subset=["site_id"]).set_index("site_id")
    site_ids = splits["site_id"].tolist()
    hand = build_hand_features(site_ids, sequences, struct_map, loop_df)
    np.save(out_path, hand)
    logger.info("[%s] hand40 -> %s shape=%s", version, out_path, hand.shape)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=["v4_cancer", "v4_cds", "all"], default="all")
    ap.add_argument("--stage", choices=["structure", "loop", "rnafm", "hand", "all"],
                    default="all")
    args = ap.parse_args()

    versions = ["v4_cancer", "v4_cds"] if args.version == "all" else [args.version]

    for ver in versions:
        splits_path = V4_RAW / f"apobec1_{ver}_with_negatives.csv"
        seqs_path = V4_RAW / f"apobec1_{ver}_sequences.json"
        if not splits_path.exists() or not seqs_path.exists():
            logger.error("Missing %s or %s — run build_apobec1_v4_datasets.py first",
                         splits_path, seqs_path)
            sys.exit(1)
        splits = pd.read_csv(splits_path)
        with open(seqs_path) as f:
            sequences = json.load(f)
        logger.info("=" * 70)
        logger.info("Processing APOBEC1 %s (n_sites=%d)", ver, len(splits))
        logger.info("=" * 70)

        struct_path = V4_PROC / f"structure_cache_apobec1_{ver}.npz"
        loop_path = V4_PROC / f"loop_position_apobec1_{ver}.csv"

        if args.stage in ("structure", "all"):
            struct_path = stage_structure(ver, splits, sequences)
        if args.stage in ("loop", "all"):
            loop_path = stage_loop(ver, splits, sequences)
        if args.stage in ("rnafm", "all"):
            stage_rnafm(ver, splits, sequences)
            gc.collect()
        if args.stage in ("hand", "all"):
            if not struct_path.exists():
                logger.error("struct cache missing for %s", ver); sys.exit(1)
            if not loop_path.exists():
                logger.error("loop csv missing for %s", ver); sys.exit(1)
            stage_hand(ver, splits, sequences, struct_path, loop_path)


if __name__ == "__main__":
    main()
