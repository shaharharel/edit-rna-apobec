#!/usr/bin/env python
"""Compute RNA-FM embeddings for all datasets and cache to disk.

Priority 1: v4 negatives (~370K new sequences)
Priority 2: Cross-species chimp sequences (~3.6K)
Priority 3: TCGA mutation + control sequences (~3.3M) — future
Priority 4: ClinVar (1.69M) — future

Usage:
    conda run -n quris python scripts/multi_enzyme/compute_rnafm_embeddings.py [--priority 1|2|3|4] [--batch-size 16] [--checkpoint-every 5000]

    # Run Priority 1 (v4 negatives):
    nohup /opt/miniconda3/envs/quris/bin/python scripts/multi_enzyme/compute_rnafm_embeddings.py --priority 1 \
        > experiments/multi_enzyme/outputs/deep_architectures/logs/rnafm_embedding.log 2>&1 &
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Paths
EMB_DIR = PROJECT_ROOT / "data/processed/multi_enzyme/embeddings"
V3_POOLED = EMB_DIR / "rnafm_pooled_v3.pt"
V3_POOLED_ED = EMB_DIR / "rnafm_pooled_edited_v3.pt"
V4_SEQS = PROJECT_ROOT / "data/processed/multi_enzyme/multi_enzyme_sequences_v4_large.json"
CROSS_SPECIES_DIR = PROJECT_ROOT / "experiments/multi_enzyme/outputs/cross_species"

# Output paths
V4_NEG_POOLED = EMB_DIR / "rnafm_pooled_v4_negatives.pt"
V4_NEG_POOLED_ED = EMB_DIR / "rnafm_pooled_edited_v4_negatives.pt"
CHIMP_POOLED = EMB_DIR / "rnafm_pooled_chimp.pt"
CHIMP_POOLED_ED = EMB_DIR / "rnafm_pooled_edited_chimp.pt"

# Checkpoint paths (for crash recovery)
V4_NEG_CHECKPOINT = EMB_DIR / "rnafm_v4_neg_checkpoint.pt"

CENTER = 100
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


def load_rnafm():
    """Load RNA-FM model on device."""
    import fm
    model, alphabet = fm.pretrained.rna_fm_t12()
    model = model.eval().to(DEVICE)
    batch_converter = alphabet.get_batch_converter()
    logger.info("RNA-FM loaded on %s", DEVICE)
    return model, batch_converter


def embed_batch(model, batch_converter, seqs_batch):
    """Embed a batch of sequences, return pooled (640-dim) embeddings."""
    data = [(f"seq_{i}", seq) for i, seq in enumerate(seqs_batch)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(DEVICE)

    with torch.no_grad():
        results = model(tokens, repr_layers=[12])
    embeddings = results["representations"][12]  # (B, L+2, 640)
    # Strip BOS/EOS, mean pool
    pooled = embeddings[:, 1:-1, :].mean(dim=1)  # (B, 640)
    return pooled.cpu()


def make_edited(seq, center=CENTER):
    """Replace C at center with U."""
    seq_list = list(seq)
    if seq_list[center] == "C":
        seq_list[center] = "U"
    return "".join(seq_list)


def compute_embeddings_batched(model, batch_converter, site_ids, sequences,
                                batch_size=16, checkpoint_every=5000,
                                checkpoint_path=None, compute_edited=True):
    """Compute embeddings in batches with periodic checkpointing.

    Returns:
        dict of {site_id: tensor(640)} for original
        dict of {site_id: tensor(640)} for edited (or None)
    """
    pooled_dict = {}
    pooled_ed_dict = {} if compute_edited else None

    # Try to load checkpoint
    start_idx = 0
    if checkpoint_path and checkpoint_path.exists():
        logger.info("Loading checkpoint from %s", checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        pooled_dict = ckpt["pooled"]
        if compute_edited:
            pooled_ed_dict = ckpt.get("pooled_edited", {})
        start_idx = ckpt["next_idx"]
        logger.info("Resuming from index %d (%d already cached)", start_idx, len(pooled_dict))

    t0 = time.time()
    total = len(site_ids)

    for i in range(start_idx, total, batch_size):
        batch_sids = site_ids[i:i + batch_size]
        batch_seqs = [sequences[j].upper().replace("T", "U") for j in range(i, min(i + batch_size, total))]

        # Embed original
        pooled_orig = embed_batch(model, batch_converter, batch_seqs)

        # Embed edited
        pooled_edit = None
        if compute_edited:
            ed_seqs = [make_edited(seq) for seq in batch_seqs]
            pooled_edit = embed_batch(model, batch_converter, ed_seqs)

        for k, sid in enumerate(batch_sids):
            pooled_dict[sid] = pooled_orig[k]
            if compute_edited:
                pooled_ed_dict[sid] = pooled_edit[k]

        # Progress logging
        done = i + len(batch_sids)
        if done % (batch_size * 50) < batch_size or done == total:
            elapsed = time.time() - t0
            computed = done - start_idx
            rate = computed / elapsed if elapsed > 0 else 0
            remaining = (total - done) / rate if rate > 0 else 0
            logger.info("  %d/%d (%.1f seq/sec, ~%.0fm remaining, %d total cached)",
                        done, total, rate, remaining / 60, len(pooled_dict))

        # Checkpoint
        if checkpoint_path and (done - start_idx) >= checkpoint_every and done % checkpoint_every < batch_size:
            logger.info("Saving checkpoint at %d...", done)
            ckpt_data = {"pooled": pooled_dict, "next_idx": done}
            if compute_edited:
                ckpt_data["pooled_edited"] = pooled_ed_dict
            torch.save(ckpt_data, checkpoint_path)
            logger.info("Checkpoint saved (%d embeddings)", len(pooled_dict))

    elapsed = time.time() - t0
    computed = total - start_idx
    logger.info("Computed %d embeddings in %.0fs (%.1f seq/sec)", computed, elapsed,
                computed / elapsed if elapsed > 0 else 0)

    return pooled_dict, pooled_ed_dict


def priority1_v4_negatives(batch_size=16, checkpoint_every=5000):
    """Compute RNA-FM embeddings for v4 negative sequences."""
    EMB_DIR.mkdir(parents=True, exist_ok=True)

    # Load v4 sequences
    logger.info("Loading v4 sequences from %s", V4_SEQS)
    with open(V4_SEQS) as f:
        v4_seqs = json.load(f)
    logger.info("Total v4 sequences: %d", len(v4_seqs))

    # Load existing v3 cache to skip already-embedded sites
    existing_keys = set()
    if V3_POOLED.exists():
        v3 = torch.load(V3_POOLED, map_location="cpu", weights_only=False)
        existing_keys = set(v3.keys())
        logger.info("Existing v3 embeddings: %d", len(existing_keys))
        del v3  # free memory

    # Also check if partial v4 output already exists
    if V4_NEG_POOLED.exists():
        v4_existing = torch.load(V4_NEG_POOLED, map_location="cpu", weights_only=False)
        existing_keys.update(v4_existing.keys())
        logger.info("Existing v4 embeddings: %d", len(v4_existing))
        del v4_existing

    # Filter to only new sequences
    needed = [(sid, seq) for sid, seq in v4_seqs.items()
              if sid not in existing_keys and len(seq) == 201]
    logger.info("Need to compute: %d new embeddings (skipping %d existing)",
                len(needed), len(v4_seqs) - len(needed))

    if not needed:
        logger.info("All v4 embeddings already computed!")
        return

    site_ids = [sid for sid, _ in needed]
    sequences = [seq for _, seq in needed]

    # Load model
    model, batch_converter = load_rnafm()

    # Compute
    pooled_dict, pooled_ed_dict = compute_embeddings_batched(
        model, batch_converter, site_ids, sequences,
        batch_size=batch_size,
        checkpoint_every=checkpoint_every,
        checkpoint_path=V4_NEG_CHECKPOINT,
        compute_edited=True,
    )

    # Save
    logger.info("Saving v4 negatives: %d embeddings", len(pooled_dict))
    torch.save(pooled_dict, V4_NEG_POOLED)
    torch.save(pooled_ed_dict, V4_NEG_POOLED_ED)
    logger.info("Saved to %s and %s", V4_NEG_POOLED, V4_NEG_POOLED_ED)

    # Clean up checkpoint
    if V4_NEG_CHECKPOINT.exists():
        V4_NEG_CHECKPOINT.unlink()
        logger.info("Removed checkpoint file")


def priority2_chimp(batch_size=16):
    """Compute RNA-FM embeddings for cross-species chimp sequences."""
    EMB_DIR.mkdir(parents=True, exist_ok=True)

    # Load chimp data
    csv_path = CROSS_SPECIES_DIR / "editing_sites_cross_species.csv"
    if not csv_path.exists():
        logger.error("Cross-species CSV not found: %s", csv_path)
        return

    import pandas as pd
    df = pd.read_csv(csv_path)
    logger.info("Cross-species CSV: %d rows, columns: %s", len(df), list(df.columns))

    # Check for chimp sequence columns
    chimp_cols = [c for c in df.columns if "chimp" in c.lower() or "pan" in c.lower() or "ortholog" in c.lower()]
    seq_cols = [c for c in df.columns if "seq" in c.lower()]
    logger.info("Chimp-related columns: %s", chimp_cols)
    logger.info("Sequence columns: %s", seq_cols)

    # We need to find or extract chimp sequences
    # Check if there's a separate chimp sequences file
    chimp_seq_files = list(CROSS_SPECIES_DIR.glob("*chimp*")) + list(CROSS_SPECIES_DIR.glob("*sequences*"))
    logger.info("Potential chimp sequence files: %s", [f.name for f in chimp_seq_files])

    if not seq_cols and not chimp_cols:
        logger.warning("No chimp sequence data found in cross-species CSV. "
                       "Chimp sequences may need to be extracted from genome first.")
        return

    # If we find the sequences, embed them
    # Check for a chimp sequence column in the dataframe
    chimp_seq_col = None
    for col in df.columns:
        if "chimp" in col.lower() and "seq" in col.lower():
            chimp_seq_col = col
            break
    if chimp_seq_col is None:
        for col in seq_cols:
            if col != "sequence":  # skip human sequence
                chimp_seq_col = col
                break

    if chimp_seq_col is None:
        logger.warning("Could not identify chimp sequence column. Available columns: %s", list(df.columns))
        logger.info("Skipping chimp embeddings — sequences need to be extracted from chimp genome first.")
        return

    # Extract sequences
    chimp_data = df[df[chimp_seq_col].notna()].copy()
    logger.info("Chimp sequences found: %d", len(chimp_data))

    if len(chimp_data) == 0:
        logger.info("No chimp sequences to embed.")
        return

    # Create site IDs
    if "site_id" in chimp_data.columns:
        site_ids = chimp_data["site_id"].tolist()
    else:
        site_ids = [f"chimp_{i}" for i in range(len(chimp_data))]

    sequences = chimp_data[chimp_seq_col].tolist()

    # Filter valid 201-nt sequences
    valid = [(sid, seq) for sid, seq in zip(site_ids, sequences)
             if isinstance(seq, str) and len(seq) == 201]
    if not valid:
        logger.warning("No valid 201-nt chimp sequences found. Sequence lengths: %s",
                       [len(s) for s in sequences[:10] if isinstance(s, str)])
        return

    site_ids = [s for s, _ in valid]
    sequences = [seq for _, seq in valid]
    logger.info("Valid chimp sequences to embed: %d", len(site_ids))

    model, batch_converter = load_rnafm()

    pooled_dict, pooled_ed_dict = compute_embeddings_batched(
        model, batch_converter, site_ids, sequences,
        batch_size=batch_size,
        checkpoint_path=None,
        compute_edited=True,
    )

    torch.save(pooled_dict, CHIMP_POOLED)
    torch.save(pooled_ed_dict, CHIMP_POOLED_ED)
    logger.info("Saved chimp embeddings to %s", CHIMP_POOLED)


def main():
    parser = argparse.ArgumentParser(description="Compute RNA-FM embeddings for all datasets")
    parser.add_argument("--priority", type=int, default=1, choices=[1, 2, 3, 4],
                        help="Which priority to compute (1=v4 negatives, 2=chimp, 3=TCGA, 4=ClinVar)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for RNA-FM inference")
    parser.add_argument("--checkpoint-every", type=int, default=5000,
                        help="Save checkpoint every N sequences")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("RNA-FM Embedding Computation — Priority %d", args.priority)
    logger.info("Device: %s", DEVICE)
    logger.info("Batch size: %d, checkpoint every: %d", args.batch_size, args.checkpoint_every)
    logger.info("=" * 60)

    if args.priority == 1:
        priority1_v4_negatives(batch_size=args.batch_size, checkpoint_every=args.checkpoint_every)
    elif args.priority == 2:
        priority2_chimp(batch_size=args.batch_size)
    elif args.priority == 3:
        logger.info("Priority 3 (TCGA) not yet implemented — requires genome re-extraction")
    elif args.priority == 4:
        logger.info("Priority 4 (ClinVar) not yet implemented — requires genome extraction")

    logger.info("Done.")


if __name__ == "__main__":
    main()
