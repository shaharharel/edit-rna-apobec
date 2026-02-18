#!/usr/bin/env python
"""Generate ViennaRNA 2D structure feature cache for all editing sites.

Computes per-position structure features (pairing probability, accessibility,
entropy) and global features (MFE, ensemble energy) for both original and
C->U edited sequences. Also computes delta features for the edit effect.

Must be run in the `vienna` conda environment (has RNA Python module).

Output: data/processed/embeddings/vienna_structure_cache.npz

Usage:
    /opt/miniconda3/envs/vienna/bin/python scripts/apobec/generate_structure_cache.py
"""

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
    pairing_prob = np.clip(np.sum(bpp, axis=1), 0, 1)
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


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Load sequences
    if not SEQUENCES_JSON.exists():
        logger.error("Sequences JSON not found: %s", SEQUENCES_JSON)
        return

    with open(SEQUENCES_JSON) as f:
        sequences = json.load(f)

    site_ids = list(sequences.keys())
    logger.info("Loaded %d sequences", len(site_ids))
    logger.info("ViennaRNA version: %s", RNA.__version__)

    # Allocate arrays
    n_sites = len(site_ids)
    seq_len = 201  # Expected length

    # Per-position features for original sequences
    pairing_probs = np.zeros((n_sites, seq_len), dtype=np.float32)
    accessibilities = np.zeros((n_sites, seq_len), dtype=np.float32)
    entropies = np.zeros((n_sites, seq_len), dtype=np.float32)
    mfes = np.zeros(n_sites, dtype=np.float32)

    # Per-position features for edited sequences (C->U at center)
    pairing_probs_edited = np.zeros((n_sites, seq_len), dtype=np.float32)
    accessibilities_edited = np.zeros((n_sites, seq_len), dtype=np.float32)
    entropies_edited = np.zeros((n_sites, seq_len), dtype=np.float32)
    mfes_edited = np.zeros(n_sites, dtype=np.float32)

    # Delta features (7-dim per site)
    delta_features = np.zeros((n_sites, 7), dtype=np.float32)

    t0 = time.time()
    n_success = 0

    for i, sid in enumerate(site_ids):
        seq = sequences[sid]
        if not seq or len(seq) < 10:
            continue

        actual_len = len(seq)
        center = actual_len // 2

        try:
            # Original sequence
            feat = compute_structure_features(seq)
            L = min(actual_len, seq_len)
            pairing_probs[i, :L] = feat["pairing_prob"][:L]
            accessibilities[i, :L] = feat["accessibility"][:L]
            entropies[i, :L] = feat["entropy"][:L]
            mfes[i] = feat["mfe"]

            # Edited sequence (C->U at center)
            seq_list = list(seq)
            if center < len(seq_list) and seq_list[center].upper() == "C":
                seq_list[center] = "U"
            edited_seq = "".join(seq_list)

            feat_ed = compute_structure_features(edited_seq)
            pairing_probs_edited[i, :L] = feat_ed["pairing_prob"][:L]
            accessibilities_edited[i, :L] = feat_ed["accessibility"][:L]
            entropies_edited[i, :L] = feat_ed["entropy"][:L]
            mfes_edited[i] = feat_ed["mfe"]

            # Compute 7-dim delta features
            window = 10
            start = max(0, center - window)
            end = min(actual_len, center + window + 1)

            dp = feat_ed["pairing_prob"] - feat["pairing_prob"]
            da = feat_ed["accessibility"] - feat["accessibility"]
            de = feat_ed["entropy"] - feat["entropy"]

            delta_features[i, 0] = dp[center] if center < len(dp) else 0.0
            delta_features[i, 1] = da[center] if center < len(da) else 0.0
            delta_features[i, 2] = de[center] if center < len(de) else 0.0
            delta_features[i, 3] = feat_ed["mfe"] - feat["mfe"]
            delta_features[i, 4] = np.mean(dp[start:end])
            delta_features[i, 5] = np.mean(da[start:end])
            delta_features[i, 6] = np.std(dp[start:end])

            n_success += 1

        except Exception as e:
            logger.warning("Failed for site %s: %s", sid, e)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            logger.info("Processed %d/%d (%.1f sites/sec)", i + 1, n_sites, rate)

    elapsed = time.time() - t0
    logger.info("Processed %d/%d sites in %.1f seconds", n_success, n_sites, elapsed)

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "vienna_structure_cache.npz"

    np.savez_compressed(
        output_path,
        site_ids=np.array(site_ids),
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
    logger.info("Sites: %d/%d processed", n_success, n_sites)
    logger.info("Mean MFE (original): %.2f kcal/mol", np.mean(mfes[mfes != 0]))
    logger.info("Mean MFE (edited): %.2f kcal/mol", np.mean(mfes_edited[mfes_edited != 0]))
    logger.info("Mean delta MFE: %.4f kcal/mol", np.mean(delta_features[:, 3]))
    logger.info("Mean delta pairing at edit: %.4f", np.mean(delta_features[:, 0]))


if __name__ == "__main__":
    main()
