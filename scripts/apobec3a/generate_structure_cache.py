#!/usr/bin/env python
"""Generate ViennaRNA 2D structure feature cache for all editing sites.

Computes per-position structure features (pairing probability, accessibility,
entropy) and global features (MFE, ensemble energy) for both original and
C->U edited sequences. Also computes delta features for the edit effect.

Must be run in the `vienna` conda environment (has RNA Python module).

Output: data/processed/embeddings/vienna_structure_cache.npz

Usage:
    /opt/miniconda3/envs/vienna/bin/python scripts/apobec3a/generate_structure_cache.py
    /opt/miniconda3/envs/vienna/bin/python scripts/apobec3a/generate_structure_cache.py --incremental
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import RNA

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEQUENCES_JSON = PROJECT_ROOT / "data" / "processed" / "site_sequences.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "embeddings"


def compute_structure_features(sequence: str, temperature: float = 37.0) -> dict:
    """Compute full structure features for a single RNA sequence.

    Returns dict with:
        pairing_prob: (n,) per-position base-pairing probability
        accessibility: (n,) per-position accessibility (1 - pairing_prob)
        entropy: (n,) per-position structure entropy
        mfe: float, minimum free energy
        ensemble_energy: float, partition function energy
    """
    seq = sequence.upper().replace("T", "U")
    n = len(seq)

    md = RNA.md()
    md.temperature = temperature

    fc = RNA.fold_compound(seq, md)

    # MFE structure
    mfe_structure, mfe = fc.mfe()

    # Partition function
    _, pf_energy = fc.pf()

    # Base pair probability matrix
    bpp_raw = np.array(fc.bpp())
    bpp = bpp_raw[1:n+1, 1:n+1]

    # Per-position pairing probability
    pairing_prob = np.clip(np.sum(bpp, axis=0) + np.sum(bpp, axis=1), 0, 1)
    accessibility = 1.0 - pairing_prob

    # Per-position entropy
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
        "ensemble_energy": float(pf_energy),
        "dot_bracket": mfe_structure,
    }


def _compute_site(sid, seq, seq_len=201):
    """Compute all structure features for a single site. Returns dict or None."""
    if not seq or len(seq) < 10:
        return None

    actual_len = len(seq)
    center = actual_len // 2
    L = min(actual_len, seq_len)

    # Original sequence
    feat = compute_structure_features(seq)
    pp = np.zeros(seq_len, dtype=np.float32)
    acc = np.zeros(seq_len, dtype=np.float32)
    ent = np.zeros(seq_len, dtype=np.float32)
    pp[:L] = feat["pairing_prob"][:L]
    acc[:L] = feat["accessibility"][:L]
    ent[:L] = feat["entropy"][:L]

    # Edited sequence (C->U at center)
    seq_list = list(seq)
    if center < len(seq_list) and seq_list[center].upper() == "C":
        seq_list[center] = "U"
    edited_seq = "".join(seq_list)

    feat_ed = compute_structure_features(edited_seq)
    pp_ed = np.zeros(seq_len, dtype=np.float32)
    acc_ed = np.zeros(seq_len, dtype=np.float32)
    ent_ed = np.zeros(seq_len, dtype=np.float32)
    pp_ed[:L] = feat_ed["pairing_prob"][:L]
    acc_ed[:L] = feat_ed["accessibility"][:L]
    ent_ed[:L] = feat_ed["entropy"][:L]

    # 7-dim delta features
    window = 10
    start = max(0, center - window)
    end = min(actual_len, center + window + 1)
    dp = feat_ed["pairing_prob"] - feat["pairing_prob"]
    da = feat_ed["accessibility"] - feat["accessibility"]
    de = feat_ed["entropy"] - feat["entropy"]

    delta = np.zeros(7, dtype=np.float32)
    delta[0] = dp[center] if center < len(dp) else 0.0
    delta[1] = da[center] if center < len(da) else 0.0
    delta[2] = de[center] if center < len(de) else 0.0
    delta[3] = feat_ed["mfe"] - feat["mfe"]
    delta[4] = np.mean(dp[start:end])
    delta[5] = np.mean(da[start:end])
    delta[6] = np.std(dp[start:end])

    return {
        "pairing_probs": pp, "accessibilities": acc, "entropies": ent,
        "mfe": feat["mfe"],
        "pairing_probs_edited": pp_ed, "accessibilities_edited": acc_ed,
        "entropies_edited": ent_ed, "mfe_edited": feat_ed["mfe"],
        "delta_features": delta,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate ViennaRNA structure cache")
    parser.add_argument("--incremental", action="store_true",
                        help="Only compute missing sites, merge with existing cache")
    parser.add_argument("--needed-csv", type=str, default=None,
                        help="CSV file with site_id column; only compute sites from this file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Load sequences
    if not SEQUENCES_JSON.exists():
        logger.error("Sequences JSON not found: %s", SEQUENCES_JSON)
        return

    with open(SEQUENCES_JSON) as f:
        sequences = json.load(f)

    logger.info("Loaded %d sequences from JSON", len(sequences))
    logger.info("ViennaRNA version: %s", RNA.__version__)

    output_path = OUTPUT_DIR / "vienna_structure_cache.npz"
    seq_len = 201

    # ---- Incremental mode ----
    if args.incremental:
        if not output_path.exists():
            logger.warning("No existing cache found, running full computation instead")
            args.incremental = False

    existing_sids = set()
    existing_data = None
    if args.incremental:
        existing_data = np.load(str(output_path), allow_pickle=True)
        existing_sids = set(str(s) for s in existing_data["site_ids"])
        logger.info("Existing cache has %d sites", len(existing_sids))

    # Determine which sites to compute
    all_seq_sids = list(sequences.keys())

    # Optionally restrict to sites from a CSV (e.g., splits_expanded.csv)
    needed_sids = None
    if args.needed_csv:
        import pandas as pd
        needed_df = pd.read_csv(args.needed_csv)
        needed_sids = set(needed_df["site_id"].astype(str))
        logger.info("Restricting to %d sites from %s", len(needed_sids), args.needed_csv)
        all_seq_sids = [s for s in all_seq_sids if s in needed_sids]

    if args.incremental:
        missing_sids = [s for s in all_seq_sids if s not in existing_sids]
        logger.info("Missing sites to compute: %d", len(missing_sids))
        if not missing_sids:
            logger.info("All sites already in cache, nothing to do")
            return
        compute_sids = missing_sids
    else:
        compute_sids = all_seq_sids

    # Compute features for needed sites
    n_compute = len(compute_sids)
    pairing_probs = np.zeros((n_compute, seq_len), dtype=np.float32)
    accessibilities = np.zeros((n_compute, seq_len), dtype=np.float32)
    entropies = np.zeros((n_compute, seq_len), dtype=np.float32)
    mfes = np.zeros(n_compute, dtype=np.float32)
    pairing_probs_edited = np.zeros((n_compute, seq_len), dtype=np.float32)
    accessibilities_edited = np.zeros((n_compute, seq_len), dtype=np.float32)
    entropies_edited = np.zeros((n_compute, seq_len), dtype=np.float32)
    mfes_edited = np.zeros(n_compute, dtype=np.float32)
    delta_features = np.zeros((n_compute, 7), dtype=np.float32)

    t0 = time.time()
    n_success = 0

    for i, sid in enumerate(compute_sids):
        seq = sequences[sid]
        try:
            result = _compute_site(sid, seq, seq_len)
            if result is not None:
                pairing_probs[i] = result["pairing_probs"]
                accessibilities[i] = result["accessibilities"]
                entropies[i] = result["entropies"]
                mfes[i] = result["mfe"]
                pairing_probs_edited[i] = result["pairing_probs_edited"]
                accessibilities_edited[i] = result["accessibilities_edited"]
                entropies_edited[i] = result["entropies_edited"]
                mfes_edited[i] = result["mfe_edited"]
                delta_features[i] = result["delta_features"]
                n_success += 1
        except Exception as e:
            logger.warning("Failed for site %s: %s", sid, e)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            logger.info("Processed %d/%d (%.1f sites/sec)", i + 1, n_compute, rate)

    elapsed = time.time() - t0
    logger.info("Computed %d/%d sites in %.1f seconds", n_success, n_compute, elapsed)

    # ---- Merge with existing cache if incremental ----
    if args.incremental and existing_data is not None:
        logger.info("Merging with existing cache...")
        old_sids = list(existing_data["site_ids"])
        new_sids = old_sids + compute_sids
        n_total = len(new_sids)

        def _merge(old_key, new_arr):
            old = existing_data[old_key]
            return np.concatenate([old, new_arr], axis=0)

        merged = {
            "site_ids": np.array(new_sids),
            "pairing_probs": _merge("pairing_probs", pairing_probs),
            "accessibilities": _merge("accessibilities", accessibilities),
            "entropies": _merge("entropies", entropies),
            "mfes": _merge("mfes", mfes),
            "pairing_probs_edited": _merge("pairing_probs_edited", pairing_probs_edited),
            "accessibilities_edited": _merge("accessibilities_edited", accessibilities_edited),
            "entropies_edited": _merge("entropies_edited", entropies_edited),
            "mfes_edited": _merge("mfes_edited", mfes_edited),
            "delta_features": _merge("delta_features", delta_features),
        }
        logger.info("Merged cache: %d total sites (%d existing + %d new)",
                     n_total, len(old_sids), len(compute_sids))

        # Save merged
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, **merged)
    else:
        # Save fresh
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            site_ids=np.array(compute_sids),
            pairing_probs=pairing_probs,
            accessibilities=accessibilities,
            entropies=entropies,
            mfes=mfes,
            pairing_probs_edited=pairing_probs_edited,
            accessibilities_edited=accessibilities_edited,
            entropies_edited=entropies_edited,
            mfes_edited=mfes_edited,
            delta_features=delta_features,
        )

    logger.info("Saved to %s (%.1f MB)", output_path, output_path.stat().st_size / 1e6)

    # Summary
    logger.info("\n=== Summary ===")
    logger.info("Sites computed this run: %d/%d", n_success, n_compute)
    if mfes[mfes != 0].size > 0:
        logger.info("Mean MFE (original): %.2f kcal/mol", np.mean(mfes[mfes != 0]))
        logger.info("Mean MFE (edited): %.2f kcal/mol", np.mean(mfes_edited[mfes_edited != 0]))
        logger.info("Mean delta MFE: %.4f kcal/mol", np.mean(delta_features[:, 3]))
        logger.info("Mean delta pairing at edit: %.4f", np.mean(delta_features[:, 0]))


if __name__ == "__main__":
    main()
