"""
Generate ViennaRNA structure feature cache for multi-enzyme editing sites.

Computes per-position structure features (pairing probability, accessibility,
entropy) and delta features (original vs C->U edited) for all sites.

Uses the same approach as scripts/apobec3a/generate_structure_cache.py.

Must be run in the `quris` conda environment (needs RNA module from ViennaRNA).

Output: data/processed/multi_enzyme/structure_cache_multi_enzyme_v2.npz

Usage:
    conda run -n quris python scripts/multi_enzyme/generate_structure_cache.py
"""
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Use latest version available: v3 > v2, prefer with_negatives
_ME_DIR = PROJECT_ROOT / "data/processed/multi_enzyme"
_SEQ_CANDIDATES = [
    _ME_DIR / "multi_enzyme_sequences_v3_with_negatives.json",
    _ME_DIR / "multi_enzyme_sequences_v2_with_negatives.json",
    _ME_DIR / "multi_enzyme_sequences_v3.json",
    _ME_DIR / "multi_enzyme_sequences_v2.json",
]
SEQUENCES_JSON = next((p for p in _SEQ_CANDIDATES if p.exists()), _SEQ_CANDIDATES[-1])

OUTPUT_DIR = _ME_DIR
_version = "v3" if "v3" in SEQUENCES_JSON.name else "v2"
OUTPUT_FILE = OUTPUT_DIR / f"structure_cache_multi_enzyme_{_version}.npz"

SEQ_LEN = 201  # Standard padded length


def compute_structure_features(sequence: str, temperature: float = 37.0) -> dict:
    """Compute ViennaRNA structure features for a single RNA sequence."""
    import RNA

    seq = sequence.upper().replace("T", "U")
    n = len(seq)

    md = RNA.md()
    md.temperature = temperature

    fc = RNA.fold_compound(seq, md)
    mfe_structure, mfe = fc.mfe()
    _, pf_energy = fc.pf()

    bpp_raw = np.array(fc.bpp())
    bpp = bpp_raw[1:n+1, 1:n+1]

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
        "ensemble_energy": float(pf_energy),
        "dot_bracket": mfe_structure,
    }


def compute_site(sid, seq):
    """Compute all structure features for a single site."""
    if not seq or len(seq) < 10:
        return None

    actual_len = len(seq)
    center = actual_len // 2
    L = min(actual_len, SEQ_LEN)

    # Original sequence
    feat = compute_structure_features(seq)
    pp = np.zeros(SEQ_LEN, dtype=np.float32)
    acc = np.zeros(SEQ_LEN, dtype=np.float32)
    ent = np.zeros(SEQ_LEN, dtype=np.float32)
    pp[:L] = feat["pairing_prob"][:L]
    acc[:L] = feat["accessibility"][:L]
    ent[:L] = feat["entropy"][:L]

    # Edited sequence (C->U at center)
    seq_list = list(seq)
    if center < len(seq_list) and seq_list[center].upper() == "C":
        seq_list[center] = "U"
    edited_seq = "".join(seq_list)

    feat_ed = compute_structure_features(edited_seq)
    pp_ed = np.zeros(SEQ_LEN, dtype=np.float32)
    acc_ed = np.zeros(SEQ_LEN, dtype=np.float32)
    ent_ed = np.zeros(SEQ_LEN, dtype=np.float32)
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
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not SEQUENCES_JSON.exists():
        logger.error("Sequences not found: %s", SEQUENCES_JSON)
        sys.exit(1)

    with open(SEQUENCES_JSON) as f:
        sequences = json.load(f)

    logger.info("Loaded %d sequences", len(sequences))

    # Load existing cache if present (flat array format)
    existing_results = {}
    existing_sids_set = set()
    if OUTPUT_FILE.exists():
        data = np.load(OUTPUT_FILE, allow_pickle=True)
        if "site_ids" in data:
            existing_sids = list(data["site_ids"])
            existing_sids_set = set(existing_sids)
            # Reconstruct result dicts from arrays
            delta_features = data["delta_features"]
            mfes = data["mfes"]
            mfes_edited = data["mfes_edited"]
            for i, sid in enumerate(existing_sids):
                existing_results[sid] = {
                    "delta_features": delta_features[i],
                    "mfe": float(mfes[i]),
                    "mfe_edited": float(mfes_edited[i]),
                }
            logger.info("Loaded %d existing entries from cache", len(existing_results))
        elif "site_data" in data:
            existing_results = dict(data["site_data"].item())
            existing_sids_set = set(existing_results.keys())
            logger.info("Loaded %d existing entries from legacy cache", len(existing_results))

    # Compute missing sites
    to_compute = {sid: seq for sid, seq in sequences.items() if sid not in existing_sids_set}
    logger.info("Need to compute %d new sites", len(to_compute))

    if not to_compute:
        logger.info("All sites already cached")
        return

    t0 = time.time()
    results = dict(existing_results)
    computed = 0
    errors = 0

    for sid, seq in to_compute.items():
        try:
            result = compute_site(sid, seq)
            if result:
                results[sid] = result
                computed += 1
            else:
                errors += 1
        except Exception as e:
            errors += 1
            if errors <= 5:
                logger.warning("Error for %s: %s", sid, e)

        if computed % 100 == 0 and computed > 0:
            elapsed = time.time() - t0
            rate = computed / elapsed
            remaining = (len(to_compute) - computed) / rate if rate > 0 else 0
            logger.info("  %d/%d done (%.1f sites/sec, ~%.0f sec remaining)",
                        computed, len(to_compute), rate, remaining)

    elapsed = time.time() - t0
    logger.info("Computed %d sites in %.1f sec (%.1f sites/sec), %d errors",
                computed, elapsed, computed / elapsed if elapsed > 0 else 0, errors)

    # Save in array format expected by exp_per_enzyme_structure.py
    # site_ids, delta_features, mfes, mfes_edited as flat arrays
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sids_arr = np.array(list(results.keys()), dtype=object)
    delta_arr = np.array([results[s]["delta_features"] for s in sids_arr], dtype=np.float32)
    mfes_arr = np.array([results[s]["mfe"] for s in sids_arr], dtype=np.float32)
    mfes_ed_arr = np.array([results[s]["mfe_edited"] for s in sids_arr], dtype=np.float32)
    np.savez_compressed(
        OUTPUT_FILE,
        site_ids=sids_arr,
        delta_features=delta_arr,
        mfes=mfes_arr,
        mfes_edited=mfes_ed_arr,
    )
    logger.info("Saved cache: %s (%d sites)", OUTPUT_FILE, len(results))


if __name__ == "__main__":
    main()
