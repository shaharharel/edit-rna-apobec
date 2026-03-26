#!/usr/bin/env python
"""Generate RNA-FM embeddings for all multi-enzyme v3 sites.

Computes 640-dim pooled embeddings for original and C→U edited sequences.
Appends to existing embedding cache.

Usage:
    conda run -n quris python scripts/multi_enzyme/generate_rnafm_embeddings_v3.py
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SEQS_JSON = PROJECT_ROOT / "data/processed/multi_enzyme/multi_enzyme_sequences_v3_with_negatives.json"
EXISTING_POOLED = PROJECT_ROOT / "data/processed/embeddings/rnafm_pooled.pt"
EXISTING_POOLED_ED = PROJECT_ROOT / "data/processed/embeddings/rnafm_pooled_edited.pt"
EXISTING_IDS = PROJECT_ROOT / "data/processed/embeddings/rnafm_site_ids.json"

OUTPUT_DIR = PROJECT_ROOT / "data/processed/multi_enzyme/embeddings"
OUTPUT_POOLED = OUTPUT_DIR / "rnafm_pooled_v3.pt"
OUTPUT_POOLED_ED = OUTPUT_DIR / "rnafm_pooled_edited_v3.pt"
OUTPUT_IDS = OUTPUT_DIR / "rnafm_site_ids_v3.json"

BATCH_SIZE = 8
CENTER = 100
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


def load_rnafm():
    """Load RNA-FM model."""
    import fm
    model, alphabet = fm.pretrained.rna_fm_t12()
    model = model.eval().to(DEVICE)
    batch_converter = alphabet.get_batch_converter()
    return model, batch_converter


def embed_batch(model, batch_converter, seqs_batch):
    """Embed a batch of sequences, return pooled embeddings."""
    data = [(f"seq_{i}", seq) for i, seq in enumerate(seqs_batch)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(DEVICE)

    with torch.no_grad():
        results = model(tokens, repr_layers=[12])
    embeddings = results["representations"][12]  # (B, L+2, 640)
    # Strip BOS/EOS, mean pool
    pooled = embeddings[:, 1:-1, :].mean(dim=1)  # (B, 640)
    return pooled.cpu()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load sequences
    with open(SEQS_JSON) as f:
        all_seqs = json.load(f)
    logger.info("Total v3 sequences: %d", len(all_seqs))

    # Load existing embeddings
    existing_ids = []
    existing_pooled = None
    existing_pooled_ed = None

    if EXISTING_IDS.exists():
        existing_ids = json.load(open(EXISTING_IDS))
        existing_pooled = torch.load(EXISTING_POOLED, map_location="cpu", weights_only=True)
        if EXISTING_POOLED_ED.exists():
            existing_pooled_ed = torch.load(EXISTING_POOLED_ED, map_location="cpu", weights_only=True)
        logger.info("Existing embeddings: %d sites", len(existing_ids))

    existing_set = set(str(s) for s in existing_ids)

    # Find sites needing embeddings
    needed = [(sid, seq) for sid, seq in all_seqs.items()
              if str(sid) not in existing_set and len(seq) == 201]
    logger.info("Need to compute: %d new embeddings", len(needed))

    if not needed:
        logger.info("All embeddings already computed!")
        return

    # Load RNA-FM
    logger.info("Loading RNA-FM model...")
    model, batch_converter = load_rnafm()
    logger.info("RNA-FM loaded on %s", DEVICE)

    # Compute embeddings in batches
    new_ids = []
    new_pooled = []
    new_pooled_ed = []

    t0 = time.time()
    for i in range(0, len(needed), BATCH_SIZE):
        batch = needed[i:i + BATCH_SIZE]
        sids = [sid for sid, _ in batch]
        orig_seqs = [seq.upper().replace("T", "U") for _, seq in batch]

        # Edited sequences (C→U at center)
        ed_seqs = []
        for seq in orig_seqs:
            seq_list = list(seq)
            if seq_list[CENTER] == "C":
                seq_list[CENTER] = "U"
            ed_seqs.append("".join(seq_list))

        # Embed original
        pooled_orig = embed_batch(model, batch_converter, orig_seqs)
        # Embed edited
        pooled_edit = embed_batch(model, batch_converter, ed_seqs)

        new_ids.extend(sids)
        new_pooled.append(pooled_orig)
        new_pooled_ed.append(pooled_edit)

        if (i // BATCH_SIZE + 1) % 50 == 0:
            elapsed = time.time() - t0
            done = i + BATCH_SIZE
            rate = done / elapsed
            remaining = (len(needed) - done) / rate
            logger.info("  %d/%d (%.1f/sec, ~%.0fm remaining)",
                         done, len(needed), rate, remaining / 60)

    # Concatenate
    new_pooled_t = torch.cat(new_pooled, dim=0)
    new_pooled_ed_t = torch.cat(new_pooled_ed, dim=0)
    logger.info("New embeddings: %s", new_pooled_t.shape)

    # Merge with existing — existing is dict format {site_id: tensor(640)}
    all_pooled_dict = {}
    all_pooled_ed_dict = {}

    if existing_pooled is not None:
        if isinstance(existing_pooled, dict):
            all_pooled_dict.update(existing_pooled)
        else:
            # Tensor format: use existing_ids to map
            for i, sid in enumerate(existing_ids):
                all_pooled_dict[str(sid)] = existing_pooled[i]

    if existing_pooled_ed is not None:
        if isinstance(existing_pooled_ed, dict):
            all_pooled_ed_dict.update(existing_pooled_ed)
        else:
            for i, sid in enumerate(existing_ids):
                all_pooled_ed_dict[str(sid)] = existing_pooled_ed[i]

    # Add new embeddings
    for i, sid in enumerate(new_ids):
        all_pooled_dict[str(sid)] = new_pooled_t[i]
        all_pooled_ed_dict[str(sid)] = new_pooled_ed_t[i]

    logger.info("Total embeddings: %d (orig), %d (edited)",
                len(all_pooled_dict), len(all_pooled_ed_dict))

    # Save as dict format (matching existing convention)
    torch.save(all_pooled_dict, OUTPUT_POOLED)
    torch.save(all_pooled_ed_dict, OUTPUT_POOLED_ED)

    # Also save ordered list for indexed access
    all_ids = list(all_pooled_dict.keys())
    with open(OUTPUT_IDS, "w") as f:
        json.dump(all_ids, f)

    elapsed = time.time() - t0
    logger.info("Done in %.0fs (%.1f seqs/sec). Saved to %s", elapsed, len(needed) / elapsed, OUTPUT_DIR)


if __name__ == "__main__":
    main()
