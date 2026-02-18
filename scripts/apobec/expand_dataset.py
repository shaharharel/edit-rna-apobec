"""Expand training dataset with new sites from published datasets and tiered negatives.

Pipeline:
1. Load all datasets (Levanon, Alqassim, Asaoka, Sharma) + tiered negatives
2. Deduplicate by genomic coordinate
3. Check which sites already have cached RNA-FM embeddings
4. Extract 201nt sequences from hg19 for NEW sites only
5. Generate RNA-FM embeddings for new sites
6. Merge into existing embedding cache
7. Create expanded splits CSV with gene-based splitting

Usage:
    # Full expansion (all datasets + Tier 2 negatives)
    python scripts/apobec/expand_dataset.py

    # Only add Tier 2 negatives (faster, no new positives)
    python scripts/apobec/expand_dataset.py --tier2-only --max-tier2-neg 2000

    # Skip embedding generation (just prepare sequences)
    python scripts/apobec/expand_dataset.py --sequences-only
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
GENOME_PATH = PROJECT_ROOT / "data" / "raw" / "genomes" / "hg19.fa"
EMBEDDING_DIR = PROJECT_ROOT / "data" / "processed" / "embeddings"
SEQUENCES_JSON = PROJECT_ROOT / "data" / "processed" / "site_sequences.json"

# Dataset paths
COMBINED_CSV = PROJECT_ROOT / "data" / "processed" / "all_datasets_combined.csv"
TIER2_CSV = PROJECT_ROOT / "data" / "processed" / "negatives_tier2.csv"
TIER3_CSV = PROJECT_ROOT / "data" / "processed" / "negatives_tier3.csv"
EXISTING_NEG_CSV = PROJECT_ROOT / "data" / "processed" / "advisor" / "negative_controls_filtered.csv"
LABELS_CSV = PROJECT_ROOT / "data" / "processed" / "editing_sites_labels.csv"

# Output
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
RNAFOLD = Path("/opt/miniconda3/envs/vienna/bin/RNAfold")

FLANK_SIZE = 100  # 201nt window


def load_genome():
    """Load hg19 genome."""
    from pyfaidx import Fasta
    logger.info("Loading hg19 genome...")
    genome = Fasta(str(GENOME_PATH))
    return genome


def revcomp_dna(seq: str) -> str:
    comp = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
    return "".join(comp.get(b, "N") for b in reversed(seq.upper()))


def extract_sequence(genome, chrom: str, pos: int, strand: str,
                     flank: int = FLANK_SIZE) -> str:
    """Extract 201nt RNA sequence centered on position."""
    if chrom not in genome:
        return ""
    chrom_len = len(genome[chrom])
    g_start = pos - flank
    g_end = pos + flank + 1

    if g_start < 0 or g_end > chrom_len:
        return ""

    dna_seq = str(genome[chrom][g_start:g_end]).upper()
    if len(dna_seq) != 2 * flank + 1:
        return ""

    if strand == "-":
        dna_seq = revcomp_dna(dna_seq)

    rna_seq = dna_seq.replace("T", "U")
    return rna_seq


def load_all_positive_sites() -> pd.DataFrame:
    """Load all positive editing sites from unified dataset."""
    if not COMBINED_CSV.exists():
        logger.error("Combined dataset not found. Run build_unified_dataset.py first.")
        return pd.DataFrame()

    df = pd.read_csv(COMBINED_CSV)
    df = df[df["is_edited"] == 1].copy()

    # Deduplicate by coordinate (keep first occurrence, prefer advisor)
    dataset_priority = {"advisor_c2t": 0, "alqassim_2021": 1,
                        "asaoka_2019": 2, "sharma_2015": 3}
    df["priority"] = df["dataset_source"].map(dataset_priority).fillna(99)
    df = df.sort_values("priority")
    df = df.drop_duplicates(subset=["chr", "start"], keep="first")
    df = df.drop(columns=["priority"])

    logger.info("Loaded %d unique positive sites from %d datasets",
                len(df), df["dataset_source"].nunique())
    for ds, count in df["dataset_source"].value_counts().items():
        logger.info("  %s: %d", ds, count)

    return df


def load_tier_negatives(tier2_only: bool = False,
                        max_tier2: int = 2000,
                        max_tier3: int = 1000,
                        seed: int = 42) -> pd.DataFrame:
    """Load and subsample tiered negatives."""
    rng = np.random.RandomState(seed)
    all_neg = []

    if TIER2_CSV.exists():
        tier2 = pd.read_csv(TIER2_CSV)
        logger.info("Loaded %d Tier 2 negatives", len(tier2))

        # Stratified subsampling by gene
        if len(tier2) > max_tier2:
            genes = tier2["gene"].unique()
            per_gene = max(1, max_tier2 // len(genes))
            sampled = tier2.groupby("gene").apply(
                lambda g: g.sample(n=min(len(g), per_gene), random_state=rng),
                include_groups=False,
            ).reset_index(drop=True)
            # If under target, add more random samples
            if len(sampled) < max_tier2:
                remaining = tier2[~tier2.index.isin(sampled.index)]
                extra = remaining.sample(
                    n=min(len(remaining), max_tier2 - len(sampled)),
                    random_state=rng,
                )
                sampled = pd.concat([sampled, extra], ignore_index=True)
            tier2 = sampled
        logger.info("  Subsampled to %d Tier 2 negatives", len(tier2))
        all_neg.append(tier2)

    if not tier2_only and TIER3_CSV.exists():
        tier3 = pd.read_csv(TIER3_CSV)
        logger.info("Loaded %d Tier 3 negatives", len(tier3))
        if len(tier3) > max_tier3:
            tier3 = tier3.sample(n=max_tier3, random_state=rng)
            logger.info("  Subsampled to %d Tier 3 negatives", len(tier3))
        all_neg.append(tier3)

    if not all_neg:
        logger.warning("No tiered negatives found. Run generate_tiered_negatives.py first.")
        return pd.DataFrame()

    neg = pd.concat(all_neg, ignore_index=True)

    # Standardize columns
    neg["is_edited"] = 0
    neg["dataset_source"] = neg["tier"].apply(lambda t: f"tier{t}_negative")
    if "editing_rate" not in neg.columns:
        neg["editing_rate"] = 0.0

    return neg


def build_expanded_site_table(
    positives: pd.DataFrame,
    tier_negatives: pd.DataFrame,
    existing_negatives: pd.DataFrame,
) -> pd.DataFrame:
    """Build the full expanded site table."""
    all_sites = []

    # Positives
    if len(positives) > 0:
        pos = positives[["site_id", "chr", "start", "end", "strand", "gene",
                         "is_edited", "dataset_source", "editing_rate", "feature"]].copy()
        pos["label"] = 1
        all_sites.append(pos)

    # Existing negatives (from advisor)
    if len(existing_negatives) > 0:
        existing_negatives = existing_negatives.copy()
        existing_negatives["label"] = 0
        existing_negatives["is_edited"] = 0
        existing_negatives["dataset_source"] = "advisor_negative"
        if "editing_rate" not in existing_negatives.columns:
            existing_negatives["editing_rate"] = 0.0
        if "feature" not in existing_negatives.columns:
            existing_negatives["feature"] = ""
        cols = ["site_id", "chr", "start", "end", "strand", "gene",
                "is_edited", "dataset_source", "editing_rate", "feature", "label"]
        existing_negatives = existing_negatives[[c for c in cols if c in existing_negatives.columns]]
        all_sites.append(existing_negatives)

    # Tiered negatives
    if len(tier_negatives) > 0:
        tn = tier_negatives.copy()
        tn["label"] = 0
        cols = ["site_id", "chr", "start", "end", "strand", "gene",
                "is_edited", "dataset_source", "editing_rate", "feature", "label"]
        tn = tn[[c for c in cols if c in tn.columns]]
        all_sites.append(tn)

    if not all_sites:
        return pd.DataFrame()

    combined = pd.concat(all_sites, ignore_index=True)

    # Final dedup by coordinate
    combined = combined.drop_duplicates(subset=["chr", "start"], keep="first")

    logger.info("Expanded site table: %d sites", len(combined))
    logger.info("  Positive: %d", (combined["label"] == 1).sum())
    logger.info("  Negative: %d", (combined["label"] == 0).sum())
    logger.info("  By source:")
    for ds, count in combined["dataset_source"].value_counts().items():
        logger.info("    %s: %d", ds, count)

    return combined


def extract_new_sequences(
    sites: pd.DataFrame,
    existing_sequences: dict,
    genome,
) -> dict:
    """Extract sequences for sites not already in cache."""
    new_sequences = {}
    already_cached = 0
    failed = 0

    for _, row in sites.iterrows():
        sid = row["site_id"]
        if sid in existing_sequences:
            already_cached += 1
            continue

        strand = row.get("strand", "+")
        if pd.isna(strand) or strand not in ("+", "-"):
            strand = "+"

        seq = extract_sequence(genome, row["chr"], row["start"], strand)
        if seq and len(seq) == 2 * FLANK_SIZE + 1:
            new_sequences[sid] = seq
        else:
            failed += 1

    logger.info("Sequence extraction: %d cached, %d new, %d failed",
                already_cached, len(new_sequences), failed)
    return new_sequences


def generate_rnafm_embeddings(
    sequences: dict,
    batch_size: int = 16,
    device: str = None,
) -> dict:
    """Generate RNA-FM embeddings for new sequences."""
    if not sequences:
        return {"pooled": {}, "tokens": {}, "pooled_edited": {}, "tokens_edited": {}}

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

    logger.info("Generating RNA-FM embeddings for %d new sequences on %s...",
                len(site_seqs), device)

    # Encode original
    pooled_dict, tokens_dict = _batch_encode(
        encoder, site_ids, site_seqs, batch_size, device_obj
    )

    # Encode edited (C->U at center)
    edited_seqs = []
    for seq in site_seqs:
        center = len(seq) // 2
        seq_list = list(seq)
        if center < len(seq_list) and seq_list[center].upper() == "C":
            seq_list[center] = "U"
        edited_seqs.append("".join(seq_list))

    pooled_ed, tokens_ed = _batch_encode(
        encoder, site_ids, edited_seqs, batch_size, device_obj
    )

    return {
        "pooled": pooled_dict,
        "tokens": tokens_dict,
        "pooled_edited": pooled_ed,
        "tokens_edited": tokens_ed,
    }


@torch.no_grad()
def _batch_encode(encoder, site_ids, sequences, batch_size, device):
    """Encode sequences in batches."""
    pooled_dict = {}
    tokens_dict = {}
    total = len(sequences)

    for i in range(0, total, batch_size):
        batch_ids = site_ids[i:i + batch_size]
        batch_seqs = sequences[i:i + batch_size]

        out = encoder(sequences=batch_seqs)
        batch_pooled = out["pooled"].cpu()
        batch_tokens = out["embeddings"].cpu()

        for j, sid in enumerate(batch_ids):
            pooled_dict[sid] = batch_pooled[j]
            tokens_dict[sid] = batch_tokens[j]

        processed = min(i + batch_size, total)
        if processed % 100 == 0 or processed == total:
            logger.info("  Encoded %d/%d sequences", processed, total)

    return pooled_dict, tokens_dict


def merge_embedding_caches(existing_dir: Path, new_embeddings: dict):
    """Merge new embeddings into existing cache files."""
    for key, new_data in new_embeddings.items():
        if not new_data:
            continue

        # Map key names to file names
        file_map = {
            "pooled": "rnafm_pooled.pt",
            "tokens": "rnafm_tokens.pt",
            "pooled_edited": "rnafm_pooled_edited.pt",
            "tokens_edited": "rnafm_tokens_edited.pt",
        }

        fname = file_map.get(key)
        if not fname:
            continue

        fpath = existing_dir / fname
        if fpath.exists():
            existing = torch.load(fpath, weights_only=False)
            existing.update(new_data)
            torch.save(existing, fpath)
            logger.info("Updated %s: %d total entries (+%d new)",
                        fname, len(existing), len(new_data))
        else:
            torch.save(new_data, fpath)
            logger.info("Created %s: %d entries", fname, len(new_data))

    # Update site IDs list
    ids_path = existing_dir / "rnafm_site_ids.json"
    if ids_path.exists():
        with open(ids_path) as f:
            existing_ids = set(json.load(f))
    else:
        existing_ids = set()

    new_ids = set(new_embeddings.get("pooled", {}).keys())
    all_ids = sorted(existing_ids | new_ids)
    with open(ids_path, "w") as f:
        json.dump(all_ids, f)
    logger.info("Updated site IDs: %d total (+%d new)", len(all_ids), len(new_ids))


def create_expanded_splits(
    sites: pd.DataFrame,
    cached_site_ids: set,
    seed: int = 42,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> pd.DataFrame:
    """Create gene-based train/val/test splits for expanded dataset.

    Only includes sites that have cached embeddings.
    """
    rng = np.random.RandomState(seed)

    # Filter to sites with embeddings
    sites = sites[sites["site_id"].isin(cached_site_ids)].copy()
    logger.info("Sites with embeddings: %d", len(sites))

    # Gene-based splitting
    genes = sites["gene"].dropna().unique()
    rng.shuffle(genes)

    n_val = int(len(genes) * val_frac)
    n_test = int(len(genes) * test_frac)

    test_genes = set(genes[:n_test])
    val_genes = set(genes[n_test:n_test + n_val])
    train_genes = set(genes[n_test + n_val:])

    def assign_split(row):
        gene = row.get("gene", "")
        if pd.isna(gene) or gene == "":
            return "train"  # No gene → train
        if gene in test_genes:
            return "test"
        if gene in val_genes:
            return "val"
        return "train"

    sites["split"] = sites.apply(assign_split, axis=1)

    logger.info("Expanded splits:")
    for split in ["train", "val", "test"]:
        subset = sites[sites["split"] == split]
        n_pos = (subset["label"] == 1).sum()
        n_neg = (subset["label"] == 0).sum()
        logger.info("  %s: %d total (%d pos, %d neg)", split, len(subset), n_pos, n_neg)

    return sites


def main():
    parser = argparse.ArgumentParser(description="Expand training dataset")
    parser.add_argument("--tier2-only", action="store_true",
                        help="Only add Tier 2 negatives (no Tier 3)")
    parser.add_argument("--max-tier2-neg", type=int, default=2000,
                        help="Max Tier 2 negatives to sample")
    parser.add_argument("--max-tier3-neg", type=int, default=1000,
                        help="Max Tier 3 negatives to sample")
    parser.add_argument("--sequences-only", action="store_true",
                        help="Only extract sequences, skip embedding generation")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for RNA-FM encoding")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # 1. Load all positive sites
    positives = load_all_positive_sites()

    # 2. Load tiered negatives
    tier_negs = load_tier_negatives(
        tier2_only=args.tier2_only,
        max_tier2=args.max_tier2_neg,
        max_tier3=args.max_tier3_neg,
        seed=args.seed,
    )

    # 3. Load existing negatives
    existing_neg = pd.DataFrame()
    if EXISTING_NEG_CSV.exists():
        existing_neg = pd.read_csv(EXISTING_NEG_CSV)
        logger.info("Loaded %d existing negatives", len(existing_neg))

    # 4. Build expanded site table
    expanded = build_expanded_site_table(positives, tier_negs, existing_neg)
    if len(expanded) == 0:
        logger.error("No sites to process!")
        return

    # 5. Load existing sequences
    existing_seqs = {}
    if SEQUENCES_JSON.exists():
        with open(SEQUENCES_JSON) as f:
            existing_seqs = json.load(f)
        logger.info("Loaded %d existing sequences", len(existing_seqs))

    # 6. Extract sequences for new sites
    genome = load_genome()
    new_seqs = extract_new_sequences(expanded, existing_seqs, genome)

    # Merge and save sequences
    all_seqs = {**existing_seqs, **new_seqs}
    with open(SEQUENCES_JSON, "w") as f:
        json.dump(all_seqs, f)
    logger.info("Total sequences: %d (saved to %s)", len(all_seqs), SEQUENCES_JSON)

    if args.sequences_only:
        logger.info("Sequences-only mode. Skipping embedding generation.")
        return

    # 7. Generate RNA-FM embeddings for sites that have sequences but no embeddings
    # Check which sites already have embeddings
    existing_emb_ids = set()
    emb_ids_path = EMBEDDING_DIR / "rnafm_site_ids.json"
    if emb_ids_path.exists():
        with open(emb_ids_path) as f:
            existing_emb_ids = set(json.load(f))

    # Find sites with sequences but no embeddings
    sites_needing_embeddings = {}
    for sid in expanded["site_id"]:
        if sid in all_seqs and sid not in existing_emb_ids:
            sites_needing_embeddings[sid] = all_seqs[sid]

    # Also include newly extracted sequences
    for sid, seq in new_seqs.items():
        if sid not in existing_emb_ids:
            sites_needing_embeddings[sid] = seq

    logger.info("Sites needing new embeddings: %d (of %d with sequences)",
                len(sites_needing_embeddings),
                sum(1 for sid in expanded["site_id"] if sid in all_seqs))

    if sites_needing_embeddings:
        new_embeddings = generate_rnafm_embeddings(
            sites_needing_embeddings,
            batch_size=args.batch_size,
            device=args.device,
        )

        # 8. Merge into existing cache
        merge_embedding_caches(EMBEDDING_DIR, new_embeddings)
    else:
        logger.info("All sites already have embeddings.")

    # 9. Create expanded splits
    # Reload site IDs from updated cache
    with open(EMBEDDING_DIR / "rnafm_site_ids.json") as f:
        cached_ids = set(json.load(f))

    splits = create_expanded_splits(expanded, cached_ids, seed=args.seed)

    # Save expanded splits
    splits_path = OUTPUT_DIR / "splits_expanded.csv"
    splits.to_csv(splits_path, index=False)
    logger.info("Saved expanded splits to %s", splits_path)

    # Summary
    logger.info("\n=== Expansion Summary ===")
    logger.info("Total sites with embeddings: %d", len(splits))
    logger.info("Positive: %d", (splits["label"] == 1).sum())
    logger.info("Negative: %d", (splits["label"] == 0).sum())
    logger.info("Datasets: %s", dict(splits["dataset_source"].value_counts()))


if __name__ == "__main__":
    main()
