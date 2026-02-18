#!/usr/bin/env python
"""Pre-compute and cache RNA-FM embeddings for all editing sites.

Generates embedding caches for fast training iteration. Saves:
- Per-token embeddings (B, L, 640) for each sequence window
- Pooled embeddings (B, 640) for each site
- Metadata mapping site_id -> embedding index

Usage:
    # Generate RNA-FM embeddings
    python scripts/apobec/generate_embeddings.py \
        --sequences_json data/sequences/site_sequences.json \
        --encoder rnafm \
        --output_dir data/processed/embeddings

    # Generate for both original and edited sequences (for edit effect)
    python scripts/apobec/generate_embeddings.py \
        --sequences_json data/sequences/site_sequences.json \
        --encoder rnafm \
        --include_edited \
        --output_dir data/processed/embeddings
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logger = logging.getLogger(__name__)


def generate_embeddings(
    sequences: Dict[str, str],
    encoder_name: str = "rnafm",
    batch_size: int = 32,
    device: Optional[str] = None,
    include_edited: bool = False,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Generate embeddings for a set of sequences.

    Args:
        sequences: Dict of site_id -> sequence string.
        encoder_name: "rnafm" or "utrlm".
        batch_size: Encoding batch size.
        device: Torch device string. Auto-detects if None.
        include_edited: Also encode C->U edited versions.

    Returns:
        Dict with:
        - "pooled": {site_id: (d_model,) tensor}
        - "tokens": {site_id: (L, d_model) tensor}
        - optionally "pooled_edited" and "tokens_edited"
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Load encoder
    if encoder_name == "rnafm":
        from embedding.rna_fm_encoder import RNAFMEncoder
        encoder = RNAFMEncoder(finetune_last_n=0).to(device)
        encoder.eval()
    elif encoder_name == "utrlm":
        from embedding.utrlm import load_utrlm
        encoder = load_utrlm(trainable=False).to(device)
        encoder.eval()
    else:
        raise ValueError(f"Unknown encoder: {encoder_name}")

    site_ids = list(sequences.keys())
    site_seqs = [sequences[sid] for sid in site_ids]

    logger.info("Generating %s embeddings for %d sequences...", encoder_name, len(site_seqs))

    # Encode original sequences
    pooled_dict, tokens_dict = _batch_encode(
        encoder, site_ids, site_seqs, batch_size, device, encoder_name
    )

    result = {"pooled": pooled_dict, "tokens": tokens_dict}

    # Encode edited sequences (C->U at center)
    if include_edited:
        edited_seqs = []
        for seq in site_seqs:
            center = len(seq) // 2
            seq_list = list(seq)
            if center < len(seq_list) and seq_list[center].upper() == "C":
                seq_list[center] = "U"
            edited_seqs.append("".join(seq_list))

        pooled_ed, tokens_ed = _batch_encode(
            encoder, site_ids, edited_seqs, batch_size, device, encoder_name
        )
        result["pooled_edited"] = pooled_ed
        result["tokens_edited"] = tokens_ed

    return result


@torch.no_grad()
def _batch_encode(
    encoder,
    site_ids: List[str],
    sequences: List[str],
    batch_size: int,
    device: torch.device,
    encoder_name: str,
) -> tuple:
    """Encode sequences in batches."""
    pooled_dict = {}
    tokens_dict = {}

    for i in range(0, len(sequences), batch_size):
        batch_ids = site_ids[i:i + batch_size]
        batch_seqs = sequences[i:i + batch_size]

        if encoder_name == "rnafm":
            out = encoder(sequences=batch_seqs)
            batch_pooled = out["pooled"].cpu()
            batch_tokens = out["embeddings"].cpu()
        else:
            batch_tokens_all = encoder(batch_seqs, return_all_tokens=True).cpu()
            batch_pooled = batch_tokens_all.mean(dim=1)
            batch_tokens = batch_tokens_all

        for j, sid in enumerate(batch_ids):
            pooled_dict[sid] = batch_pooled[j]
            tokens_dict[sid] = batch_tokens[j]

        if (i // batch_size + 1) % 10 == 0:
            logger.info("  Processed %d / %d sequences", min(i + batch_size, len(sequences)), len(sequences))

    return pooled_dict, tokens_dict


def save_embeddings(
    embeddings: Dict[str, Dict[str, torch.Tensor]],
    output_dir: Path,
    encoder_name: str,
):
    """Save embeddings to disk as .pt files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for key in embeddings:
        data = embeddings[key]
        # Save as a dict of tensors
        save_path = output_dir / f"{encoder_name}_{key}.pt"
        torch.save(data, save_path)
        logger.info("Saved %s (%d items) to %s", key, len(data), save_path)

    # Save site ID list for reference
    site_ids = list(embeddings["pooled"].keys())
    meta_path = output_dir / f"{encoder_name}_site_ids.json"
    with open(meta_path, "w") as f:
        json.dump(site_ids, f)
    logger.info("Saved %d site IDs to %s", len(site_ids), meta_path)


def main():
    parser = argparse.ArgumentParser(description="Generate RNA embeddings for APOBEC sites")
    parser.add_argument("--sequences_json", type=str, required=True,
                        help="Path to {site_id: sequence} JSON file")
    parser.add_argument("--encoder", type=str, default="rnafm", choices=["rnafm", "utrlm"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--include_edited", action="store_true",
                        help="Also encode C->U edited sequences")
    parser.add_argument("--output_dir", type=str,
                        default=str(PROJECT_ROOT / "data" / "processed" / "embeddings"))
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Load sequences
    with open(args.sequences_json) as f:
        sequences = json.load(f)
    logger.info("Loaded %d sequences from %s", len(sequences), args.sequences_json)

    # Generate embeddings
    t0 = time.time()
    embeddings = generate_embeddings(
        sequences,
        encoder_name=args.encoder,
        batch_size=args.batch_size,
        device=args.device,
        include_edited=args.include_edited,
    )
    elapsed = time.time() - t0
    logger.info("Embedding generation took %.1f seconds", elapsed)

    # Save
    output_dir = Path(args.output_dir)
    save_embeddings(embeddings, output_dir, args.encoder)

    logger.info("Done. Embeddings saved to %s", output_dir)


if __name__ == "__main__":
    main()
