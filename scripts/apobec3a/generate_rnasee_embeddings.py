"""Generate RNA-FM pooled embeddings for RNAsee comparison negative sites.

These negative sites (from negatives_per_dataset_all_c.csv and
negatives_per_dataset.csv) have genomic coordinates but no sequences or
embeddings cached yet. This script:

1. Loads existing site_sequences.json and rnafm_site_ids.json
2. Loads all negative site coordinates from both CSV files
3. Extracts 201nt sequences from hg38 for sites not yet in sequence cache
4. Generates RNA-FM pooled embeddings (original + edited C->U at center)
5. Merges into existing embedding cache files and updates site_sequences.json

Usage:
    conda run -n quris python scripts/apobec3a/generate_rnasee_embeddings.py
    conda run -n quris python scripts/apobec3a/generate_rnasee_embeddings.py --batch-size 32
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logger = logging.getLogger(__name__)

# Paths
GENOME_PATH = PROJECT_ROOT / "data" / "raw" / "genomes" / "hg38.fa"
EMBEDDING_DIR = PROJECT_ROOT / "data" / "processed" / "embeddings"
SEQUENCES_JSON = PROJECT_ROOT / "data" / "processed" / "site_sequences.json"
ALLC_NEG_CSV = PROJECT_ROOT / "data" / "processed" / "negatives_per_dataset_all_c.csv"
PER_DATASET_NEG_CSV = PROJECT_ROOT / "data" / "processed" / "negatives_per_dataset.csv"

FLANK_SIZE = 100  # 201nt window


def revcomp_dna(seq: str) -> str:
    comp = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
    return "".join(comp.get(b, "N") for b in reversed(seq.upper()))


def extract_sequences(sites_df: pd.DataFrame, existing_seqs: dict) -> dict:
    """Extract 201nt RNA sequences from hg38 for sites not in cache."""
    missing = sites_df[~sites_df["site_id"].isin(existing_seqs)]
    if len(missing) == 0:
        logger.info("All %d sites already have sequences.", len(sites_df))
        return {}

    logger.info("Extracting sequences for %d sites from hg38...", len(missing))

    from pyfaidx import Fasta
    genome = Fasta(str(GENOME_PATH))

    new_seqs = {}
    failed = 0

    for _, row in missing.iterrows():
        sid = row["site_id"]
        chrom = str(row["chr"])
        pos = int(row["start"])
        strand = str(row.get("strand", "+"))
        if strand not in ("+", "-"):
            strand = "+"

        if chrom not in genome:
            failed += 1
            continue

        chrom_len = len(genome[chrom])
        g_start = pos - FLANK_SIZE
        g_end = pos + FLANK_SIZE + 1

        if g_start < 0 or g_end > chrom_len:
            failed += 1
            continue

        dna_seq = str(genome[chrom][g_start:g_end]).upper()
        if len(dna_seq) != 2 * FLANK_SIZE + 1:
            failed += 1
            continue

        if strand == "-":
            dna_seq = revcomp_dna(dna_seq)

        rna_seq = dna_seq.replace("T", "U")
        new_seqs[sid] = rna_seq

    logger.info("Extracted %d new sequences (%d failed)", len(new_seqs), failed)
    return new_seqs


@torch.no_grad()
def batch_encode(encoder, site_ids, sequences, batch_size, device):
    """Encode sequences in batches, returning pooled embeddings only."""
    pooled_dict = {}
    total = len(sequences)

    for i in range(0, total, batch_size):
        batch_ids = site_ids[i:i + batch_size]
        batch_seqs = sequences[i:i + batch_size]

        out = encoder(sequences=batch_seqs)
        batch_pooled = out["pooled"].cpu()

        for j, sid in enumerate(batch_ids):
            pooled_dict[sid] = batch_pooled[j]

        processed = min(i + batch_size, total)
        if processed % 500 == 0 or processed == total:
            logger.info("  Encoded %d/%d sequences (%.1f%%)",
                        processed, total, 100 * processed / total)

    return pooled_dict


def generate_pooled_embeddings(sequences: dict, batch_size: int = 16,
                                device: str = None):
    """Generate RNA-FM pooled embeddings for original and edited sequences."""
    if not sequences:
        return {}, {}

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    device_obj = torch.device(device)

    from embedding.rna_fm_encoder import RNAFMEncoder
    encoder = RNAFMEncoder(finetune_last_n=0).to(device_obj)
    encoder.eval()

    site_ids = list(sequences.keys())
    site_seqs = [sequences[sid] for sid in site_ids]

    logger.info("Generating RNA-FM pooled embeddings for %d sequences on %s...",
                len(site_seqs), device)

    # Original sequences
    logger.info("--- Encoding original sequences ---")
    pooled_orig = batch_encode(encoder, site_ids, site_seqs, batch_size, device_obj)

    # Edited sequences (C->U at center)
    edited_seqs = []
    for seq in site_seqs:
        center = len(seq) // 2
        seq_list = list(seq)
        if center < len(seq_list) and seq_list[center].upper() == "C":
            seq_list[center] = "U"
        edited_seqs.append("".join(seq_list))

    logger.info("--- Encoding edited sequences ---")
    pooled_edited = batch_encode(encoder, site_ids, edited_seqs, batch_size, device_obj)

    return pooled_orig, pooled_edited


def merge_into_cache(embedding_dir: Path, pooled_orig: dict, pooled_edited: dict):
    """Merge new embeddings into existing cache files."""
    n_new = len(pooled_orig)

    for fname, new_data in [("rnafm_pooled.pt", pooled_orig),
                             ("rnafm_pooled_edited.pt", pooled_edited)]:
        fpath = embedding_dir / fname
        if fpath.exists():
            existing = torch.load(fpath, weights_only=False)
            existing.update(new_data)
            torch.save(existing, fpath)
            logger.info("Updated %s: %d total entries (+%d new)",
                        fname, len(existing), n_new)
        else:
            torch.save(new_data, fpath)
            logger.info("Created %s: %d entries", fname, n_new)

    # Update site IDs list
    ids_path = embedding_dir / "rnafm_site_ids.json"
    if ids_path.exists():
        with open(ids_path) as f:
            existing_ids = set(json.load(f))
    else:
        existing_ids = set()

    new_ids = set(pooled_orig.keys())
    all_ids = sorted(existing_ids | new_ids)
    with open(ids_path, "w") as f:
        json.dump(all_ids, f)
    logger.info("Updated site IDs: %d total (+%d new)", len(all_ids), len(new_ids))


def main():
    parser = argparse.ArgumentParser(
        description="Generate RNA-FM pooled embeddings for RNAsee comparison negatives")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for RNA-FM encoding")
    parser.add_argument("--device", type=str, default=None,
                        help="Device for RNA-FM (auto-detect if not set)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    # ---------------------------------------------------------------
    # 1. Load existing data
    # ---------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("GENERATE RNA-FM EMBEDDINGS FOR RNASEE COMPARISON NEGATIVES")
    logger.info("=" * 70)

    # Load existing sequences
    existing_seqs = {}
    if SEQUENCES_JSON.exists():
        with open(SEQUENCES_JSON) as f:
            existing_seqs = json.load(f)
    logger.info("Existing sequences in cache: %d", len(existing_seqs))

    # Load existing embedding IDs
    existing_emb_ids = set()
    emb_ids_path = EMBEDDING_DIR / "rnafm_site_ids.json"
    if emb_ids_path.exists():
        with open(emb_ids_path) as f:
            existing_emb_ids = set(json.load(f))
    logger.info("Existing embedding IDs: %d", len(existing_emb_ids))

    # ---------------------------------------------------------------
    # 2. Load negative site coordinates from both CSV files
    # ---------------------------------------------------------------
    all_neg_frames = []
    for csv_path in [ALLC_NEG_CSV, PER_DATASET_NEG_CSV]:
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            logger.info("Loaded %d rows from %s", len(df), csv_path.name)
            all_neg_frames.append(df)
        else:
            logger.warning("Not found: %s", csv_path)

    if not all_neg_frames:
        logger.error("No negative CSV files found!")
        return

    all_neg = pd.concat(all_neg_frames, ignore_index=True)
    all_neg = all_neg.drop_duplicates(subset=["site_id"], keep="first")
    logger.info("Total unique negative sites: %d", len(all_neg))

    # ---------------------------------------------------------------
    # 3. Extract sequences for sites missing from cache
    # ---------------------------------------------------------------
    new_seqs = extract_sequences(all_neg, existing_seqs)

    # Merge and save updated sequences
    if new_seqs:
        all_seqs = {**existing_seqs, **new_seqs}
        with open(SEQUENCES_JSON, "w") as f:
            json.dump(all_seqs, f)
        logger.info("Saved %d total sequences to %s (+%d new)",
                    len(all_seqs), SEQUENCES_JSON.name, len(new_seqs))
    else:
        all_seqs = existing_seqs

    # ---------------------------------------------------------------
    # 4. Identify sites with sequences but no embeddings
    # ---------------------------------------------------------------
    sites_needing_embeddings = {}
    for sid in all_neg["site_id"]:
        if sid in all_seqs and sid not in existing_emb_ids:
            sites_needing_embeddings[sid] = all_seqs[sid]

    logger.info("Sites needing new embeddings: %d", len(sites_needing_embeddings))

    if not sites_needing_embeddings:
        logger.info("All sites already have embeddings. Nothing to do.")
        return

    # ---------------------------------------------------------------
    # 5. Generate RNA-FM pooled embeddings
    # ---------------------------------------------------------------
    pooled_orig, pooled_edited = generate_pooled_embeddings(
        sites_needing_embeddings,
        batch_size=args.batch_size,
        device=args.device,
    )

    # ---------------------------------------------------------------
    # 6. Merge into existing cache
    # ---------------------------------------------------------------
    merge_into_cache(EMBEDDING_DIR, pooled_orig, pooled_edited)

    # ---------------------------------------------------------------
    # 7. Summary
    # ---------------------------------------------------------------
    # Reload to verify
    with open(emb_ids_path) as f:
        final_ids = json.load(f)
    with open(SEQUENCES_JSON) as f:
        final_seqs = json.load(f)

    logger.info("=" * 70)
    logger.info("DONE")
    logger.info("  New embeddings generated: %d", len(pooled_orig))
    logger.info("  Total sequences in cache: %d", len(final_seqs))
    logger.info("  Total embedding IDs in cache: %d", len(final_ids))

    # Check file sizes
    for fname in ["rnafm_pooled.pt", "rnafm_pooled_edited.pt"]:
        fpath = EMBEDDING_DIR / fname
        if fpath.exists():
            size_mb = fpath.stat().st_size / (1024 * 1024)
            logger.info("  %s: %.1f MB", fname, size_mb)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
