"""Consolidate A3A ClinVar intermediate chunk files into a single features cache.

The A3A ClinVar experiment (exp_clinvar_prediction.py) saves 34 intermediate .npz
chunk files, each containing pre-computed features for ~50k ClinVar variants.
This script merges them into a single cache that A3B, A3G, and other enzyme
ClinVar experiments can reuse (since features are enzyme-agnostic).

Each chunk contains:
  - site_ids: (N,) object array of variant IDs like 'clinvar_chr1_69240_+'
  - hand_46: (N, 46) float32 array of hand-crafted features (motif + loop + struct delta)
  - rnasee_50: (N, 50) float32 array of RNAsee-style features
  - valid: (N,) bool array indicating successfully computed variants
  - n_success: scalar int64 count of valid variants

Output cache contains only valid variants:
  - site_ids, hand_46, rnasee_50 (concatenated across all chunks, filtered to valid only)
"""
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

CHUNK_DIR = Path("experiments/apobec3a/outputs/clinvar_prediction/intermediate")
CACHE_PATH = Path("data/processed/clinvar_features_cache.npz")


def main():
    chunks = sorted(CHUNK_DIR.glob("chunk_*.npz"))
    logger.info(f"Found {len(chunks)} chunks in {CHUNK_DIR}")

    all_site_ids = []
    all_hand_46 = []
    all_rnasee_50 = []
    total_valid = 0
    total_variants = 0

    for i, chunk_path in enumerate(chunks):
        data = np.load(chunk_path, allow_pickle=True)
        valid = data["valid"]
        n = int(valid.sum())
        total_valid += n
        total_variants += len(valid)

        # Keep only valid variants
        all_site_ids.append(data["site_ids"][valid])
        all_hand_46.append(data["hand_46"][valid])
        all_rnasee_50.append(data["rnasee_50"][valid])

        if (i + 1) % 10 == 0 or i == len(chunks) - 1:
            logger.info(f"  Processed {i+1}/{len(chunks)} chunks ({total_valid} valid so far)")

    site_ids = np.concatenate(all_site_ids)
    hand_46 = np.concatenate(all_hand_46)
    rnasee_50 = np.concatenate(all_rnasee_50)

    logger.info(f"Total variants: {total_variants}, valid: {total_valid} ({100*total_valid/total_variants:.1f}%)")
    logger.info(f"site_ids: {site_ids.shape}, hand_46: {hand_46.shape}, rnasee_50: {rnasee_50.shape}")

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE_PATH,
        site_ids=site_ids,
        hand_46=hand_46,
        rnasee_50=rnasee_50,
    )
    size_mb = CACHE_PATH.stat().st_size / 1024 / 1024
    logger.info(f"Cache saved to {CACHE_PATH} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
