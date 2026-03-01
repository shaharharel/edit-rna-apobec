#!/usr/bin/env python
"""Generate per-dataset structure-matched hard negatives with embeddings.

Full pipeline:
1. For each dataset, find TC-motif cytosines in that dataset's genes
2. Extract 201-nt windows from hg38
3. Compute ViennaRNA delta_MFE (original vs C→U edited)
4. Filter to |delta_MFE| <= 0.1 (structure-matched to positives)
5. Compute full ViennaRNA 7-feature deltas
6. Generate RNA-FM embeddings (pooled + token-level, original + edited)
7. Save everything for downstream experiments

Output files:
  - data/processed/hardneg_per_dataset.csv
  - data/processed/hardneg_site_sequences.json
  - data/processed/embeddings/hardneg_rnafm_pooled.pt
  - data/processed/embeddings/hardneg_rnafm_pooled_edited.pt
  - data/processed/embeddings/hardneg_rnafm_tokens.pt
  - data/processed/embeddings/hardneg_rnafm_tokens_edited.pt
  - data/processed/embeddings/hardneg_vienna_structure.npz

Usage:
    python scripts/apobec3a/generate_hardneg_pipeline.py
    python scripts/apobec3a/generate_hardneg_pipeline.py --neg-ratio 5 --candidates-ratio 50
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logger = logging.getLogger(__name__)

DEFAULT_GENOME_PATH = PROJECT_ROOT / "data" / "raw" / "genomes" / "hg38.fa"
REFGENE_PATH = PROJECT_ROOT / "data" / "raw" / "genomes" / "refGene.txt"
COMBINED_PATH = PROJECT_ROOT / "data" / "processed" / "all_datasets_combined.csv"
SPLITS_CSV = PROJECT_ROOT / "data" / "processed" / "splits_expanded.csv"
EMB_DIR = PROJECT_ROOT / "data" / "processed" / "embeddings"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

WINDOW_SIZE = 100  # ±100 nt = 201 nt total


def load_refgene(path):
    cols = [
        "bin", "name", "chrom", "strand", "txStart", "txEnd",
        "cdsStart", "cdsEnd", "exonCount", "exonStarts", "exonEnds",
        "score", "name2", "cdsStartStat", "cdsEndStat", "exonFrames",
    ]
    df = pd.read_csv(path, sep="\t", header=None, names=cols)
    std_chroms = {f"chr{i}" for i in range(1, 23)} | {"chrX", "chrY"}
    return df[df["chrom"].isin(std_chroms)]


def get_exon_intervals(row):
    starts = [int(s) for s in row["exonStarts"].rstrip(",").split(",") if s]
    ends = [int(e) for e in row["exonEnds"].rstrip(",").split(",") if e]
    return list(zip(starts, ends))


def get_representative_transcripts(refgene, gene_set):
    gene_tx = refgene[refgene["name2"].isin(gene_set)]
    nm_tx = gene_tx[gene_tx["name"].str.startswith("NM_")]
    genes_with_nm = set(nm_tx["name2"])
    genes_without_nm = gene_set - genes_with_nm
    if genes_without_nm:
        other_tx = gene_tx[gene_tx["name2"].isin(genes_without_nm)]
        transcripts = pd.concat([nm_tx, other_tx])
    else:
        transcripts = nm_tx
    transcripts = transcripts.copy()
    transcripts["cds_length"] = transcripts["cdsEnd"] - transcripts["cdsStart"]
    transcripts = transcripts.sort_values("cds_length", ascending=False)
    return transcripts.drop_duplicates(subset="name2", keep="first")


def extract_window_from_genome(genome, chrom, position, strand, window_size=WINDOW_SIZE):
    """Extract a ±window_size window around the position from hg38."""
    if chrom not in genome:
        return None

    chrom_seq = genome[chrom]
    chrom_len = len(chrom_seq)

    start = max(0, position - window_size)
    end = min(chrom_len, position + window_size + 1)

    seq = str(chrom_seq[start:end]).upper()

    if strand == "-":
        comp = {"A": "U", "T": "A", "C": "G", "G": "C", "N": "N"}
        seq = "".join(comp.get(b, "N") for b in reversed(seq))
    else:
        # Convert T to U for RNA
        seq = seq.replace("T", "U")

    # Pad if at chromosome boundary
    left_pad = window_size - (position - start)
    right_pad = window_size - (end - position - 1)
    if left_pad > 0:
        seq = "N" * left_pad + seq
    if right_pad > 0:
        seq = seq + "N" * right_pad

    return seq


def compute_vienna_delta_mfe(seq_orig, edit_pos):
    """Compute delta MFE for a C→U edit at edit_pos using ViennaRNA."""
    import RNA

    seq_edit = list(seq_orig)
    if edit_pos < len(seq_edit):
        seq_edit[edit_pos] = "U"
    seq_edit = "".join(seq_edit)

    try:
        fc_orig = RNA.fold_compound(seq_orig)
        _, mfe_orig = fc_orig.mfe()

        fc_edit = RNA.fold_compound(seq_edit)
        _, mfe_edit = fc_edit.mfe()

        return mfe_edit - mfe_orig
    except Exception:
        return None


def _worker_delta_mfe(args):
    """Worker function for parallel ViennaRNA delta_MFE computation."""
    import RNA
    seq, edit_pos, threshold = args
    seq_edit = list(seq)
    if edit_pos < len(seq_edit):
        seq_edit[edit_pos] = "U"
    seq_edit = "".join(seq_edit)
    try:
        fc_orig = RNA.fold_compound(seq)
        _, mfe_orig = fc_orig.mfe()
        fc_edit = RNA.fold_compound(seq_edit)
        _, mfe_edit = fc_edit.mfe()
        delta = mfe_edit - mfe_orig
        if abs(delta) <= threshold:
            return delta
    except Exception:
        pass
    return None


def compute_vienna_full_features(seq_orig, edit_pos, window=10):
    """Compute all 7 ViennaRNA delta features."""
    import RNA

    seq_edit = list(seq_orig)
    if edit_pos < len(seq_edit):
        seq_edit[edit_pos] = "U"
    seq_edit = "".join(seq_edit)

    try:
        # Original structure
        fc_orig = RNA.fold_compound(seq_orig)
        ss_orig, mfe_orig = fc_orig.mfe()
        fc_orig.exp_params_rescale(mfe_orig)
        fc_orig.pf()
        bpp_orig = fc_orig.bpp()

        # Edited structure
        fc_edit = RNA.fold_compound(seq_edit)
        ss_edit, mfe_edit = fc_edit.mfe()
        fc_edit.exp_params_rescale(mfe_edit)
        fc_edit.pf()
        bpp_edit = fc_edit.bpp()

        n = len(seq_orig)

        # Pairing probability at edit site
        def pairing_prob(bpp, pos):
            p = 0.0
            idx = pos + 1  # 1-indexed in ViennaRNA
            if idx < len(bpp):
                for j in range(1, len(bpp)):
                    if j != idx:
                        ii, jj = min(idx, j), max(idx, j)
                        if ii < len(bpp) and jj < len(bpp[ii]):
                            p += bpp[ii][jj]
            return min(p, 1.0)

        pair_orig = pairing_prob(bpp_orig, edit_pos)
        pair_edit = pairing_prob(bpp_edit, edit_pos)

        # Accessibility = 1 - pairing
        acc_orig = 1.0 - pair_orig
        acc_edit = 1.0 - pair_edit

        # Entropy at edit site
        def positional_entropy(bpp, pos):
            idx = pos + 1
            probs = []
            if idx < len(bpp):
                for j in range(1, len(bpp)):
                    if j != idx:
                        ii, jj = min(idx, j), max(idx, j)
                        if ii < len(bpp) and jj < len(bpp[ii]):
                            p = bpp[ii][jj]
                            if p > 1e-10:
                                probs.append(p)
            unpaired = max(0, 1.0 - sum(probs))
            if unpaired > 1e-10:
                probs.append(unpaired)
            return -sum(p * np.log2(p) for p in probs if p > 1e-10)

        ent_orig = positional_entropy(bpp_orig, edit_pos)
        ent_edit = positional_entropy(bpp_edit, edit_pos)

        # Delta features at edit site
        delta_pairing = pair_edit - pair_orig
        delta_accessibility = acc_edit - acc_orig
        delta_entropy = ent_edit - ent_orig
        delta_mfe = mfe_edit - mfe_orig

        # Regional features (±window nt around edit site)
        start = max(0, edit_pos - window)
        end = min(n, edit_pos + window + 1)

        regional_delta_pair = []
        regional_delta_acc = []
        for pos in range(start, end):
            p_orig = pairing_prob(bpp_orig, pos)
            p_edit = pairing_prob(bpp_edit, pos)
            regional_delta_pair.append(p_edit - p_orig)
            regional_delta_acc.append((1 - p_edit) - (1 - p_orig))

        mean_delta_pair = np.mean(regional_delta_pair) if regional_delta_pair else 0.0
        mean_delta_acc = np.mean(regional_delta_acc) if regional_delta_acc else 0.0
        std_delta_pair = np.std(regional_delta_pair) if regional_delta_pair else 0.0

        return np.array([
            delta_pairing,
            delta_accessibility,
            delta_entropy,
            delta_mfe,
            mean_delta_pair,
            mean_delta_acc,
            std_delta_pair,
        ], dtype=np.float32)

    except Exception as e:
        logger.warning("ViennaRNA error: %s", e)
        return None


def generate_rnafm_embeddings(sequences_dict, batch_size=16, pooled_only=False):
    """Generate RNA-FM embeddings for given sequences.

    Args:
        pooled_only: If True, only store mean-pooled embeddings (640-dim per site).
            Token-level embeddings are discarded to prevent OOM on large datasets.
    """
    import torch
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    import fm

    logger.info("Loading RNA-FM model...")
    model, alphabet = fm.pretrained.rna_fm_t12()
    model.eval()
    batch_converter = alphabet.get_batch_converter()

    site_ids = list(sequences_dict.keys())
    site_seqs = [sequences_dict[sid] for sid in site_ids]

    pooled_dict = {}
    tokens_dict = {}

    mode_str = "pooled-only" if pooled_only else "pooled + tokens"
    logger.info("Encoding %d sequences with RNA-FM (%s)...", len(site_seqs), mode_str)

    for i in range(0, len(site_seqs), batch_size):
        batch_ids = site_ids[i:i + batch_size]
        batch_seqs = site_seqs[i:i + batch_size]

        data = [(sid, seq) for sid, seq in zip(batch_ids, batch_seqs)]
        _, _, tokens = batch_converter(data)

        with torch.no_grad():
            results = model(tokens, repr_layers=[12])

        emb = results['representations'][12]  # (B, L, 640)
        pooled = emb.mean(dim=1)  # (B, 640)

        for j, sid in enumerate(batch_ids):
            pooled_dict[sid] = pooled[j].cpu()
            if not pooled_only:
                tokens_dict[sid] = emb[j].cpu()

        del tokens, results, emb, pooled

        if (i // batch_size + 1) % 20 == 0:
            logger.info("  Processed %d / %d sequences",
                        min(i + batch_size, len(site_seqs)), len(site_seqs))

    return pooled_dict, tokens_dict


def main():
    parser = argparse.ArgumentParser(description="Generate per-dataset structure-matched hard negatives")
    parser.add_argument("--neg-ratio", type=int, default=5,
                        help="Target negatives-per-positive ratio")
    parser.add_argument("--candidates-ratio", type=int, default=50,
                        help="Candidate ratio before structure filtering (default: 50)")
    parser.add_argument("--delta-mfe-threshold", type=float, default=0.1,
                        help="Max |delta_MFE| for structure matching (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=6,
                        help="Number of parallel workers for ViennaRNA (default: 6)")
    parser.add_argument("--pooled-only", action="store_true",
                        help="Generate only pooled embeddings (skip 27GB+ token-level storage)")
    parser.add_argument("--skip-embeddings", action="store_true",
                        help="Skip RNA-FM embedding generation")
    parser.add_argument("--skip-vienna-features", action="store_true",
                        help="Skip full ViennaRNA 7-feature computation (saves hours, only needed for tabular models)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    rng = np.random.RandomState(args.seed)

    # Validate inputs
    if not DEFAULT_GENOME_PATH.exists():
        logger.error("Genome not found: %s", DEFAULT_GENOME_PATH)
        return
    if not REFGENE_PATH.exists():
        logger.error("RefGene not found: %s", REFGENE_PATH)
        return

    t_total = time.time()

    # ===================================================================
    # Step 1: Load reference data
    # ===================================================================
    from pyfaidx import Fasta
    logger.info("Loading hg38 genome...")
    genome = Fasta(str(DEFAULT_GENOME_PATH))

    logger.info("Loading RefSeq annotations...")
    refgene = load_refgene(REFGENE_PATH)

    # Load all known positive sites
    combined = pd.read_csv(COMBINED_PATH)
    splits_df = pd.read_csv(SPLITS_CSV)
    positive_coords = set(zip(combined["chr"], combined["start"]))

    # Also exclude existing negatives
    existing_neg = splits_df[splits_df["label"] == 0]
    existing_neg_coords = set()
    for _, row in existing_neg.iterrows():
        if pd.notna(row.get("chr")) and pd.notna(row.get("start")):
            existing_neg_coords.add((str(row["chr"]), int(row["start"])))

    exclude_coords = positive_coords | existing_neg_coords
    logger.info("Excluding %d known positions (%d positive, %d existing negative)",
                len(exclude_coords), len(positive_coords), len(existing_neg_coords))

    # Get representative transcripts
    all_genes = set(combined["gene"].dropna().str.strip())
    all_genes.discard("")
    representative = get_representative_transcripts(refgene, all_genes)
    logger.info("Representative transcripts: %d", len(representative))

    # ===================================================================
    # Step 2: Generate TC-motif candidates per dataset
    # ===================================================================
    dataset_labels = {
        "advisor_c2t": "Levanon",
        "asaoka_2019": "Asaoka",
        "alqassim_2021": "Alqassim",
        "sharma_2015": "Sharma",
        "baysal_2016": "Baysal",
    }

    all_candidates = []

    for ds_name, ds_group in combined.groupby("dataset_source"):
        if ds_name not in dataset_labels:
            continue

        ds_genes = set(ds_group["gene"].dropna().str.strip())
        ds_genes.discard("")
        n_pos = len(ds_group)
        target_candidates = n_pos * args.candidates_ratio

        logger.info("\n=== %s: %d positives, %d genes, target %d candidates ===",
                    dataset_labels[ds_name], n_pos, len(ds_genes), target_candidates)

        candidates = []
        ds_transcripts = representative[representative["name2"].isin(ds_genes)]

        for _, tx in ds_transcripts.iterrows():
            gene = tx["name2"]
            chrom = tx["chrom"]
            strand = tx["strand"]
            exons = get_exon_intervals(tx)

            # Extract exonic sequence and coordinate map
            seq_parts, coord_map = [], []
            if chrom not in genome:
                continue
            chrom_len = len(genome[chrom])
            for start, end in sorted(exons):
                s, e = max(0, start), min(chrom_len, end)
                if s >= e:
                    continue
                seg = str(genome[chrom][s:e]).upper()
                seq_parts.append(seg)
                coord_map.extend(range(s, e))

            exonic_seq = "".join(seq_parts)
            if strand == "-":
                comp = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
                exonic_seq = "".join(comp.get(b, "N") for b in reversed(exonic_seq))
                coord_map = list(reversed(coord_map))

            # Find TC-motif positions
            for i in range(1, len(exonic_seq)):
                if exonic_seq[i] == "C" and exonic_seq[i - 1] == "T":
                    genomic_pos = coord_map[i]
                    if (chrom, genomic_pos) not in exclude_coords:
                        candidates.append({
                            "chr": chrom,
                            "start": genomic_pos,
                            "strand": strand,
                            "gene": gene,
                            "dataset_source": ds_name,
                        })

        # Deduplicate
        seen = set()
        unique_candidates = []
        for c in candidates:
            key = (c["chr"], c["start"])
            if key not in seen:
                seen.add(key)
                unique_candidates.append(c)

        # Subsample to target
        if len(unique_candidates) > target_candidates:
            rng.shuffle(unique_candidates)
            unique_candidates = unique_candidates[:target_candidates]

        all_candidates.extend(unique_candidates)
        logger.info("  Found %d unique TC-motif candidates", len(unique_candidates))

    logger.info("\nTotal candidates across all datasets: %d", len(all_candidates))

    # ===================================================================
    # Step 3: Extract windows and compute ViennaRNA delta_MFE (parallel)
    # ===================================================================
    logger.info("\nExtracting windows...")

    # Extract all windows first (fast, genome I/O)
    window_data = []
    for cand in all_candidates:
        seq = extract_window_from_genome(
            genome, cand["chr"], cand["start"], cand["strand"]
        )
        if seq is not None and len(seq) == 2 * WINDOW_SIZE + 1:
            window_data.append((cand, seq))

    logger.info("Extracted %d valid windows from %d candidates", len(window_data), len(all_candidates))

    # Parallel ViennaRNA delta_MFE computation
    logger.info("Computing ViennaRNA delta_MFE with %d workers...", args.workers)
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing

    worker_args = [(seq, WINDOW_SIZE, args.delta_mfe_threshold) for _, seq in window_data]

    matched_candidates = []
    t_vienna = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Submit in chunks for better progress reporting
        chunk_size = 1000
        n_total = len(worker_args)
        n_processed = 0

        for chunk_start in range(0, n_total, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_total)
            chunk_args = worker_args[chunk_start:chunk_end]
            chunk_data = window_data[chunk_start:chunk_end]

            results = list(executor.map(_worker_delta_mfe, chunk_args))

            for i, delta_mfe in enumerate(results):
                if delta_mfe is not None:
                    cand, seq = chunk_data[i]
                    cand["sequence"] = seq
                    cand["delta_mfe"] = delta_mfe
                    matched_candidates.append(cand)

            n_processed += len(chunk_args)
            if n_processed % 10000 < chunk_size:
                elapsed = time.time() - t_vienna
                rate = n_processed / max(elapsed, 0.1)
                remaining = (n_total - n_processed) / max(rate, 1)
                logger.info("  %d/%d processed (%.0f/s), %d matched, ETA: %.0fs",
                            n_processed, n_total, rate,
                            len(matched_candidates), remaining)

    elapsed_vienna = time.time() - t_vienna
    logger.info("ViennaRNA filtering: %d -> %d matched (%.1fs, %.0f/s)",
                len(window_data), len(matched_candidates), elapsed_vienna,
                len(window_data) / max(elapsed_vienna, 0.1))

    # ===================================================================
    # Step 4: Sample per dataset
    # ===================================================================
    logger.info("\nSampling per dataset (target ratio 1:%d)...", args.neg_ratio)

    matched_df = pd.DataFrame(matched_candidates)
    sampled_rows = []

    for ds_name in dataset_labels:
        ds_matched = matched_df[matched_df["dataset_source"] == ds_name]
        n_pos = len(combined[combined["dataset_source"] == ds_name])
        target_n = n_pos * args.neg_ratio

        if len(ds_matched) >= target_n:
            sampled = ds_matched.sample(n=target_n, random_state=rng)
        else:
            sampled = ds_matched
            logger.warning("  %s: only %d matched (target %d)",
                          dataset_labels[ds_name], len(sampled), target_n)

        sampled_rows.append(sampled)
        logger.info("  %s: %d matched -> %d sampled (%.1fx)",
                    dataset_labels[ds_name], len(ds_matched), len(sampled),
                    len(sampled) / max(1, n_pos))

    final_df = pd.concat(sampled_rows, ignore_index=True)
    final_df["site_id"] = final_df.apply(
        lambda r: f"hardneg_{r['dataset_source']}_{r['chr']}_{r['start']}", axis=1
    )
    final_df["label"] = 0
    final_df["is_edited"] = 0

    # Assign train/val/test splits (80/10/10)
    n = len(final_df)
    perm = rng.permutation(n)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    splits = np.array(["test"] * n, dtype="U5")
    splits[perm[:n_train]] = "train"
    splits[perm[n_train:n_train + n_val]] = "val"
    final_df["split"] = splits

    logger.info("\nFinal: %d structure-matched per-dataset negatives", len(final_df))
    logger.info("  Train: %d, Val: %d, Test: %d",
                (splits == "train").sum(), (splits == "val").sum(), (splits == "test").sum())
    logger.info("  Mean |delta_MFE|: %.4f", abs(final_df["delta_mfe"]).mean())

    # ===================================================================
    # Step 5: Compute full ViennaRNA features
    # ===================================================================
    if not args.skip_vienna_features:
        logger.info("\nComputing full ViennaRNA features for %d sites...", len(final_df))
        t_feat = time.time()

        all_features = []
        all_feature_ids = []

        for idx, row in final_df.iterrows():
            feats = compute_vienna_full_features(row["sequence"], WINDOW_SIZE)
            if feats is not None:
                all_features.append(feats)
                all_feature_ids.append(row["site_id"])

            if (len(all_feature_ids)) % 200 == 0:
                logger.info("  %d / %d features computed", len(all_feature_ids), len(final_df))

        logger.info("Full features: %d/%d computed (%.1fs)",
                    len(all_features), len(final_df), time.time() - t_feat)

        # Save structure features
        np.savez(
            EMB_DIR / "hardneg_vienna_structure.npz",
            site_ids=np.array(all_feature_ids),
            delta_features=np.array(all_features),
        )
    else:
        logger.info("\nSkipping full ViennaRNA feature computation (--skip-vienna-features)")

    # ===================================================================
    # Step 6: Save sequences
    # ===================================================================
    sequences_dict = {}
    for _, row in final_df.iterrows():
        sequences_dict[row["site_id"]] = row["sequence"]

    seq_path = OUTPUT_DIR / "hardneg_site_sequences.json"
    with open(seq_path, "w") as f:
        json.dump(sequences_dict, f)
    logger.info("Saved %d sequences to %s", len(sequences_dict), seq_path)

    # Save CSV (without sequence column to keep it clean)
    csv_cols = ["site_id", "chr", "start", "strand", "gene", "dataset_source",
                "label", "is_edited", "split", "delta_mfe"]
    final_df[csv_cols].to_csv(OUTPUT_DIR / "hardneg_per_dataset.csv", index=False)
    logger.info("Saved CSV to %s", OUTPUT_DIR / "hardneg_per_dataset.csv")

    # ===================================================================
    # Step 7: Generate RNA-FM embeddings
    # ===================================================================
    if not args.skip_embeddings:
        logger.info("\nGenerating RNA-FM embeddings...")

        # Original sequences
        orig_sequences = {sid: seq for sid, seq in sequences_dict.items()}

        # Edited sequences (C→U at center)
        edited_sequences = {}
        for sid, seq in sequences_dict.items():
            edit_pos = WINDOW_SIZE
            seq_list = list(seq)
            if edit_pos < len(seq_list) and seq_list[edit_pos].upper() == "C":
                seq_list[edit_pos] = "U"
            edited_sequences[sid] = "".join(seq_list)

        # Generate embeddings
        import torch

        pooled_orig, tokens_orig = generate_rnafm_embeddings(orig_sequences, pooled_only=args.pooled_only)
        logger.info("Original embeddings: %d pooled, %d tokens", len(pooled_orig), len(tokens_orig))

        pooled_edited, tokens_edited = generate_rnafm_embeddings(edited_sequences, pooled_only=args.pooled_only)
        logger.info("Edited embeddings: %d pooled, %d tokens", len(pooled_edited), len(tokens_edited))

        # Save
        torch.save(pooled_orig, EMB_DIR / "hardneg_rnafm_pooled.pt")
        torch.save(pooled_edited, EMB_DIR / "hardneg_rnafm_pooled_edited.pt")
        if not args.pooled_only:
            torch.save(tokens_orig, EMB_DIR / "hardneg_rnafm_tokens.pt")
            torch.save(tokens_edited, EMB_DIR / "hardneg_rnafm_tokens_edited.pt")
        logger.info("Saved RNA-FM embeddings to %s", EMB_DIR)
    else:
        logger.info("Skipping RNA-FM embedding generation (--skip-embeddings)")

    # ===================================================================
    # Summary
    # ===================================================================
    elapsed_total = time.time() - t_total
    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE in %.1fs", elapsed_total)
    logger.info("=" * 70)

    logger.info("\nPer-dataset structure-matched negatives:")
    for ds_name in dataset_labels:
        ds_neg = final_df[final_df["dataset_source"] == ds_name]
        n_pos = len(combined[combined["dataset_source"] == ds_name])
        logger.info("  %s: %d negatives (%.1fx positives), mean |delta_MFE|=%.4f",
                    dataset_labels[ds_name], len(ds_neg), len(ds_neg) / max(1, n_pos),
                    abs(ds_neg["delta_mfe"]).mean())

    logger.info("\nOutput files:")
    logger.info("  CSV: %s", OUTPUT_DIR / "hardneg_per_dataset.csv")
    logger.info("  Sequences: %s", OUTPUT_DIR / "hardneg_site_sequences.json")
    logger.info("  Structure: %s", EMB_DIR / "hardneg_vienna_structure.npz")
    if not args.skip_embeddings:
        logger.info("  Embeddings: %s/hardneg_rnafm_*.pt", EMB_DIR)


if __name__ == "__main__":
    main()
