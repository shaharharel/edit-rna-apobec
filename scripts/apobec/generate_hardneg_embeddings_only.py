#!/usr/bin/env python
"""Generate pooled-only RNA-FM embeddings for hard negative sites.

Reads from the already-saved hardneg_site_sequences.json and generates
only pooled (mean-pool over tokens) embeddings. Token-level embeddings
are NOT stored (they would be 27GB+ for 53K sites).

Processes in batches with periodic checkpoint saves to prevent OOM.

Expected RAM: ~2GB peak (RNA-FM model ~400MB + batch tensors)
Expected output: ~137MB per file (53,505 x 640 x 4 bytes)

Usage:
    python scripts/apobec/generate_hardneg_embeddings_only.py
    python scripts/apobec/generate_hardneg_embeddings_only.py --batch-size 32 --save-every 5000
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

EMB_DIR = PROJECT_ROOT / "data" / "processed" / "embeddings"
SEQ_PATH = PROJECT_ROOT / "data" / "processed" / "hardneg_site_sequences.json"
WINDOW_SIZE = 100

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def encode_pooled_batch(model, batch_converter, site_ids, sequences, batch_size=32):
    """Encode sequences and return only pooled embeddings (no token storage)."""
    pooled_dict = {}
    n_total = len(site_ids)

    for i in range(0, n_total, batch_size):
        batch_ids = site_ids[i:i + batch_size]
        batch_seqs = [sequences[sid] for sid in batch_ids]

        data = [(sid, seq) for sid, seq in zip(batch_ids, batch_seqs)]
        _, _, tokens = batch_converter(data)

        with torch.no_grad():
            results = model(tokens, repr_layers=[12])

        # Extract ONLY pooled embeddings — token-level emb is discarded
        emb = results['representations'][12]  # (B, L, 640)
        pooled = emb.mean(dim=1).cpu()  # (B, 640)

        for j, sid in enumerate(batch_ids):
            pooled_dict[sid] = pooled[j]

        # Explicitly free batch memory
        del tokens, results, emb, pooled
        if i % (batch_size * 10) == 0 and i > 0:
            gc.collect()

    return pooled_dict


def main():
    parser = argparse.ArgumentParser(description="Generate pooled-only RNA-FM embeddings for hard negatives")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for RNA-FM encoding (default: 32)")
    parser.add_argument("--save-every", type=int, default=5000,
                        help="Save checkpoint every N sequences (default: 5000)")
    args = parser.parse_args()

    # Load sequences
    logger.info("Loading sequences from %s", SEQ_PATH)
    with open(SEQ_PATH) as f:
        sequences = json.load(f)
    logger.info("Loaded %d sequences", len(sequences))

    site_ids = list(sequences.keys())
    n_total = len(site_ids)

    out_orig = EMB_DIR / "hardneg_rnafm_pooled.pt"
    out_edited = EMB_DIR / "hardneg_rnafm_pooled_edited.pt"

    # Check if both already done
    if out_orig.exists() and out_edited.exists():
        orig = torch.load(out_orig, weights_only=False)
        edited = torch.load(out_edited, weights_only=False)
        if len(orig) == n_total and len(edited) == n_total:
            logger.info("Both embedding files already exist with %d entries. Nothing to do.", n_total)
            return
        del orig, edited

    # Load RNA-FM model
    logger.info("Loading RNA-FM model...")
    import fm
    model, alphabet = fm.pretrained.rna_fm_t12()
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    logger.info("RNA-FM loaded.")

    # ==========================================
    # Phase 1: Original sequences
    # ==========================================
    if out_orig.exists():
        orig = torch.load(out_orig, weights_only=False)
        if len(orig) == n_total:
            logger.info("Original embeddings already complete (%d). Skipping.", len(orig))
            pooled_orig = orig
        else:
            logger.info("Original embeddings incomplete (%d/%d). Re-generating.", len(orig), n_total)
            del orig
            pooled_orig = None
    else:
        pooled_orig = None

    if pooled_orig is None:
        logger.info("Generating ORIGINAL pooled embeddings for %d sequences...", n_total)
        t0 = time.time()
        pooled_orig = {}
        checkpoint_path = EMB_DIR / "hardneg_pooled_orig_checkpoint.pt"

        # Resume from checkpoint
        start_idx = 0
        if checkpoint_path.exists():
            ckpt = torch.load(checkpoint_path, weights_only=False)
            pooled_orig = ckpt["data"]
            start_idx = ckpt["next_idx"]
            logger.info("Resuming original from index %d (%d done)", start_idx, len(pooled_orig))
            del ckpt
            gc.collect()

        for i in range(start_idx, n_total, args.batch_size):
            batch_end = min(i + args.batch_size, n_total)
            batch_ids = site_ids[i:batch_end]
            batch_seqs = [sequences[sid] for sid in batch_ids]

            data = [(sid, seq) for sid, seq in zip(batch_ids, batch_seqs)]
            _, _, tokens = batch_converter(data)

            with torch.no_grad():
                results = model(tokens, repr_layers=[12])
            emb = results['representations'][12]
            pooled = emb.mean(dim=1).cpu()

            for j, sid in enumerate(batch_ids):
                pooled_orig[sid] = pooled[j]

            del tokens, results, emb, pooled

            n_done = len(pooled_orig)
            if n_done % 500 < args.batch_size or batch_end == n_total:
                elapsed = time.time() - t0
                rate = max(1, n_done - start_idx) / max(elapsed, 0.1)
                remaining = (n_total - n_done) / max(rate, 0.1)
                logger.info("  [orig] %d/%d (%.1f%%) | %.1f seq/s | ETA: %.0fs",
                            n_done, n_total, 100 * n_done / n_total, rate, remaining)

            if n_done % args.save_every < args.batch_size and n_done < n_total:
                torch.save({"data": pooled_orig, "next_idx": batch_end}, checkpoint_path)
                gc.collect()

        # Save final
        torch.save(pooled_orig, out_orig)
        logger.info("Saved %d original pooled embeddings (%.1fMB)",
                    len(pooled_orig), out_orig.stat().st_size / 1e6)
        if checkpoint_path.exists():
            checkpoint_path.unlink()

    # Free original pooled from memory
    del pooled_orig
    gc.collect()

    # ==========================================
    # Phase 2: Edited sequences (C→U at center)
    # ==========================================
    if out_edited.exists():
        edited = torch.load(out_edited, weights_only=False)
        if len(edited) == n_total:
            logger.info("Edited embeddings already complete (%d). Done!", len(edited))
            return
        else:
            logger.info("Edited embeddings incomplete (%d/%d). Re-generating.", len(edited), n_total)
            del edited

    # Build edited sequences
    edited_sequences = {}
    for sid, seq in sequences.items():
        seq_list = list(seq)
        if WINDOW_SIZE < len(seq_list) and seq_list[WINDOW_SIZE].upper() == "C":
            seq_list[WINDOW_SIZE] = "U"
        edited_sequences[sid] = "".join(seq_list)

    logger.info("Generating EDITED pooled embeddings for %d sequences...", n_total)
    t0 = time.time()
    pooled_edited = {}
    checkpoint_path = EMB_DIR / "hardneg_pooled_edited_checkpoint.pt"

    # Resume from checkpoint
    start_idx = 0
    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, weights_only=False)
        pooled_edited = ckpt["data"]
        start_idx = ckpt["next_idx"]
        logger.info("Resuming edited from index %d (%d done)", start_idx, len(pooled_edited))
        del ckpt
        gc.collect()

    for i in range(start_idx, n_total, args.batch_size):
        batch_end = min(i + args.batch_size, n_total)
        batch_ids = site_ids[i:batch_end]
        batch_seqs = [edited_sequences[sid] for sid in batch_ids]

        data = [(sid, seq) for sid, seq in zip(batch_ids, batch_seqs)]
        _, _, tokens = batch_converter(data)

        with torch.no_grad():
            results = model(tokens, repr_layers=[12])
        emb = results['representations'][12]
        pooled = emb.mean(dim=1).cpu()

        for j, sid in enumerate(batch_ids):
            pooled_edited[sid] = pooled[j]

        del tokens, results, emb, pooled

        n_done = len(pooled_edited)
        if n_done % 500 < args.batch_size or batch_end == n_total:
            elapsed = time.time() - t0
            rate = max(1, n_done - start_idx) / max(elapsed, 0.1)
            remaining = (n_total - n_done) / max(rate, 0.1)
            logger.info("  [edited] %d/%d (%.1f%%) | %.1f seq/s | ETA: %.0fs",
                        n_done, n_total, 100 * n_done / n_total, rate, remaining)

        if n_done % args.save_every < args.batch_size and n_done < n_total:
            torch.save({"data": pooled_edited, "next_idx": batch_end}, checkpoint_path)
            gc.collect()

    # Save final
    torch.save(pooled_edited, out_edited)
    logger.info("Saved %d edited pooled embeddings (%.1fMB)",
                len(pooled_edited), out_edited.stat().st_size / 1e6)
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    logger.info("DONE - all hardneg pooled RNA-FM embeddings generated")


if __name__ == "__main__":
    main()
