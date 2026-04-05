#!/usr/bin/env python3
"""Compute RNA-FM embeddings for TCGA mutations/controls and ClinVar variants.

Generates 640-dim pooled RNA-FM embeddings (original + edited) for:
  1. TCGA C>T/G>A somatic mutations + matched controls (~5.9M rows across 10 cancers)
  2. ClinVar C>T variants (~1.69M)

These embeddings are needed for Phase 3 neural model experiments in the multi-enzyme
advisor report.

For TCGA, sequences are re-extracted from hg19 genome using the same pipeline as
tcga_full_model_enrichment.py (same seed=42 for controls). Each cancer is saved as
a separate .pt file with incremental checkpointing.

For ClinVar, sequences are extracted from hg38 genome.

Usage:
    # Start with smallest cancer (LIHC) as test:
    /opt/miniconda3/envs/quris/bin/python scripts/multi_enzyme/compute_rnafm_tcga_clinvar.py --mode tcga --cancer lihc

    # All TCGA cancers:
    /opt/miniconda3/envs/quris/bin/python scripts/multi_enzyme/compute_rnafm_tcga_clinvar.py --mode tcga

    # ClinVar:
    /opt/miniconda3/envs/quris/bin/python scripts/multi_enzyme/compute_rnafm_tcga_clinvar.py --mode clinvar

    # Both:
    /opt/miniconda3/envs/quris/bin/python scripts/multi_enzyme/compute_rnafm_tcga_clinvar.py --mode all
"""

import argparse
import gc
import json
import logging
import multiprocessing as mp
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pyfaidx import Fasta

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Ensure stdout is unbuffered for nohup
sys.stdout.reconfigure(line_buffering=True)

# ============================================================================
# Paths
# ============================================================================
DATA_DIR = PROJECT_ROOT / "data"
HG19_FA = DATA_DIR / "raw/genomes/hg19.fa"
HG38_FA = DATA_DIR / "raw/genomes/hg38.fa"
REFGENE_HG19 = DATA_DIR / "raw/genomes/refGene_hg19.txt"
TCGA_MAF_DIR = DATA_DIR / "raw/tcga"
CLINVAR_CSV = (
    PROJECT_ROOT
    / "experiments/apobec3a/outputs/clinvar_prediction/clinvar_all_scores.csv"
)

EMB_DIR = DATA_DIR / "processed/multi_enzyme/embeddings"

CENTER = 100
SEED = 42
N_CONTROLS = 5
def _select_device():
    """Select compute device. MPS can hang on long runs, so allow override."""
    import os
    override = os.environ.get("RNAFM_DEVICE", "").strip()
    if override:
        return override
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

DEVICE = _select_device()

# All 10 cancers with their MAF file stems
CANCER_STUDIES = {
    "blca": "blca_tcga_pan_can_atlas_2018",
    "brca": "brca_tcga_pan_can_atlas_2018",
    "cesc": "cesc_tcga_pan_can_atlas_2018",
    "lusc": "lusc_tcga_pan_can_atlas_2018",
    "skcm": "skcm_tcga_pan_can_atlas_2018",
    "hnsc": "hnsc_tcga_pan_can_atlas_2018",
    "esca": "esca_tcga_pan_can_atlas_2018",
    "stad": "stad_tcga_pan_can_atlas_2018",
    "lihc": "lihc_tcga_pan_can_atlas_2018",
    "coadread": "coadread_tcga_pan_can_atlas_2018",
}

# Process order: smallest first
CANCER_ORDER = [
    "lihc", "esca", "hnsc", "lusc", "brca", "cesc", "blca", "stad",
    "coadread", "skcm",
]


# ============================================================================
# RNA-FM model loading
# ============================================================================

def load_rnafm():
    """Load RNA-FM model on device."""
    import fm

    model, alphabet = fm.pretrained.rna_fm_t12()
    model = model.eval().to(DEVICE)
    batch_converter = alphabet.get_batch_converter()
    logger.info("RNA-FM loaded on %s (device)", DEVICE)
    return model, alphabet, batch_converter


def embed_batch(model, batch_converter, seqs_batch):
    """Embed a batch of sequences, return pooled (640-dim) embeddings on CPU."""
    data = [(f"seq_{i}", seq) for i, seq in enumerate(seqs_batch)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(DEVICE)

    with torch.no_grad():
        results = model(tokens, repr_layers=[12])
    embeddings = results["representations"][12]  # (B, L+2, 640)
    # Strip BOS/EOS, mean pool
    pooled = embeddings[:, 1:-1, :].mean(dim=1)  # (B, 640)
    result = pooled.cpu()

    # Synchronize MPS to prevent async operation buildup
    if DEVICE == "mps":
        torch.mps.synchronize()
        # Periodically empty MPS cache to prevent memory buildup
        torch.mps.empty_cache()

    return result


def make_edited(seq, center=CENTER):
    """Replace C at center with U."""
    seq_list = list(seq)
    if seq_list[center] == "C":
        seq_list[center] = "U"
    return "".join(seq_list)


# ============================================================================
# Batched embedding with checkpointing
# ============================================================================

def compute_embeddings_batched(
    model,
    batch_converter,
    sequences,
    batch_size=16,
    checkpoint_every=50000,
    checkpoint_path=None,
):
    """Compute RNA-FM embeddings (original + edited) with incremental checkpointing.

    On MPS, periodically resets the model to prevent GPU hangs (known MPS issue
    with long-running inference). The model is moved off-device and back every
    MPS_RESET_EVERY sequences.

    Args:
        model: RNA-FM model
        batch_converter: RNA-FM batch converter
        sequences: list of 201-nt RNA strings (already T->U converted, C at center)
        batch_size: GPU batch size
        checkpoint_every: save checkpoint every N sequences
        checkpoint_path: path for checkpoint file (None = no checkpointing)

    Returns:
        pooled_orig: tensor (N, 640)
        pooled_edited: tensor (N, 640)
    """
    MPS_RESET_EVERY = 999999999  # Disabled — testing showed no hang without resets, 2x faster

    n = len(sequences)
    pooled_orig = torch.zeros(n, 640)
    pooled_edited = torch.zeros(n, 640)

    start_idx = 0

    # Resume from checkpoint
    if checkpoint_path and checkpoint_path.exists():
        logger.info("Loading checkpoint from %s", checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        start_idx = ckpt["next_idx"]
        pooled_orig[:start_idx] = ckpt["pooled_orig"][:start_idx]
        pooled_edited[:start_idx] = ckpt["pooled_edited"][:start_idx]
        logger.info("Resuming from index %d / %d", start_idx, n)
        del ckpt
        gc.collect()

    if start_idx >= n:
        logger.info("All %d embeddings already computed from checkpoint", n)
        return pooled_orig, pooled_edited

    t0 = time.time()
    seqs_since_reset = 0

    for i in range(start_idx, n, batch_size):
        end = min(i + batch_size, n)
        batch_seqs = sequences[i:end]
        this_batch = end - i

        # MPS hang prevention: periodically cycle model off/on device
        if DEVICE == "mps" and seqs_since_reset >= MPS_RESET_EVERY:
            logger.info("  MPS reset at %d (cycling model to prevent hang)...", i)
            model = model.cpu()
            torch.mps.synchronize()
            torch.mps.empty_cache()
            gc.collect()
            time.sleep(0.5)
            model = model.to(DEVICE)
            seqs_since_reset = 0

        # Original embeddings
        orig = embed_batch(model, batch_converter, batch_seqs)
        pooled_orig[i:end] = orig

        # Edited embeddings (C->U at center)
        ed_seqs = [make_edited(seq) for seq in batch_seqs]
        edited = embed_batch(model, batch_converter, ed_seqs)
        pooled_edited[i:end] = edited

        seqs_since_reset += this_batch

        # Progress logging every ~800 sequences
        done = end
        if (done - start_idx) % (batch_size * 50) < batch_size or done == n:
            elapsed = time.time() - t0
            computed = done - start_idx
            rate = computed / elapsed if elapsed > 0 else 0
            remaining = (n - done) / rate if rate > 0 else 0
            logger.info(
                "  %d/%d (%.1f seq/sec, ~%.0fm remaining)",
                done, n, rate, remaining / 60,
            )

        # Checkpoint
        if (
            checkpoint_path
            and done > start_idx
            and done % checkpoint_every < batch_size
            and done < n
        ):
            logger.info("Saving checkpoint at %d...", done)
            torch.save(
                {
                    "next_idx": done,
                    "pooled_orig": pooled_orig[:done],
                    "pooled_edited": pooled_edited[:done],
                },
                checkpoint_path,
            )
            logger.info("Checkpoint saved")

    elapsed = time.time() - t0
    computed = n - start_idx
    logger.info(
        "Computed %d embeddings in %.0fs (%.1f seq/sec)",
        computed, elapsed, computed / elapsed if elapsed > 0 else 0,
    )

    return pooled_orig, pooled_edited


# ============================================================================
# TCGA: parse mutations and generate controls (mirrors tcga_full_model_enrichment.py)
# ============================================================================

def parse_exonic_regions():
    """Parse exonic regions from refGene (hg19) for control matching."""
    exons_by_gene = defaultdict(list)
    seen = set()
    with open(REFGENE_HG19) as f:
        for line in f:
            fields = line.strip().split("\t")
            if len(fields) < 13:
                continue
            chrom = fields[2]
            strand = fields[3]
            cds_s, cds_e = int(fields[6]), int(fields[7])
            if cds_s == cds_e:
                continue
            gene = fields[12]
            for es_str, ee_str in zip(
                fields[9].rstrip(",").split(","),
                fields[10].rstrip(",").split(","),
            ):
                if not es_str or not ee_str:
                    continue
                es = max(int(es_str), cds_s)
                ee = min(int(ee_str), cds_e)
                if es >= ee:
                    continue
                key = (chrom, es, ee)
                if key not in seen:
                    seen.add(key)
                    exons_by_gene[gene].append((chrom, es, ee, strand, gene))
    logger.info("Parsed %d genes with exonic regions", len(exons_by_gene))
    return exons_by_gene


def parse_ct_mutations(maf_path):
    """Parse C>T and G>A SNPs from MAF. Returns DataFrame with (chrom, pos, strand_inf, gene)."""
    rows = []
    for chunk in pd.read_csv(
        maf_path, sep="\t", comment="#", low_memory=False, chunksize=100000
    ):
        if "Variant_Type" in chunk.columns:
            chunk = chunk[chunk["Variant_Type"] == "SNP"]

        ct = (chunk["Reference_Allele"] == "C") & (chunk["Tumor_Seq_Allele2"] == "T")
        ga = (chunk["Reference_Allele"] == "G") & (chunk["Tumor_Seq_Allele2"] == "A")
        sub = chunk[ct | ga].copy()

        if len(sub) > 0:
            chrom = sub["Chromosome"].astype(str)
            if not chrom.str.startswith("chr").any():
                chrom = "chr" + chrom
            sub["chrom"] = chrom
            sub["pos"] = sub["Start_Position"].astype(int) - 1  # 0-based
            sub["strand_inf"] = np.where(sub["Reference_Allele"] == "C", "+", "-")
            sub["gene"] = sub.get("Hugo_Symbol", "unknown")
            rows.append(sub[["chrom", "pos", "strand_inf", "gene"]].copy())

    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    df_dedup = df.drop_duplicates(subset=["chrom", "pos"]).copy()
    logger.info(
        "  %d C>T/G>A mutations -> %d unique positions", len(df), len(df_dedup)
    )
    return df_dedup


def precompute_gene_c_positions(exons_by_gene, genome):
    """Pre-load all C/G positions in exonic regions for each (gene, chrom) pair.

    This is much faster than scanning the genome per-mutation because we only
    read each exon once instead of once per mutation in the same gene.

    Returns:
        dict of (gene, chrom) -> list of (chrom, pos, strand) tuples
    """
    logger.info("Pre-computing C/G positions in exonic regions...")
    t0 = time.time()
    gene_chrom_positions = {}
    n_exons = 0

    for gene, exon_list in exons_by_gene.items():
        # Group exons by chromosome
        by_chrom = defaultdict(list)
        for chrom, start, end, strand, _ in exon_list:
            by_chrom[chrom].append((start, end))

        for chrom, intervals in by_chrom.items():
            c_positions = []
            for start, end in intervals:
                try:
                    exon_seq = str(genome[chrom][start:end]).upper()
                    for i, base in enumerate(exon_seq):
                        pos = start + i
                        if base == "C":
                            c_positions.append((chrom, pos, "+"))
                        elif base == "G":
                            c_positions.append((chrom, pos, "-"))
                    n_exons += 1
                except (KeyError, ValueError):
                    continue
            if c_positions:
                gene_chrom_positions[(gene, chrom)] = c_positions

    elapsed = time.time() - t0
    logger.info(
        "  Pre-computed C/G positions for %d (gene, chrom) pairs from %d exons in %.0fs",
        len(gene_chrom_positions), n_exons, elapsed,
    )
    return gene_chrom_positions


def get_matched_controls(mutations_df, gene_chrom_positions, n_controls=N_CONTROLS):
    """For each mutation, pick n_controls random C positions in the same gene's exons.

    Uses pre-computed gene C/G positions (from precompute_gene_c_positions) for speed.
    """
    rng = np.random.RandomState(SEED)
    controls = []

    for _, row in mutations_df.iterrows():
        gene = row["gene"]
        chrom = row["chrom"]
        mut_pos = row["pos"]

        c_positions = gene_chrom_positions.get((gene, chrom), [])
        if not c_positions:
            continue

        # Filter out the mutation position itself
        c_positions_filtered = [
            (ch, p, s) for ch, p, s in c_positions if p != mut_pos
        ]

        if len(c_positions_filtered) >= n_controls:
            chosen = rng.choice(len(c_positions_filtered), n_controls, replace=False)
            for idx in chosen:
                ch, p, s = c_positions_filtered[idx]
                controls.append(
                    {"chrom": ch, "pos": p, "strand_inf": s, "gene": gene}
                )
        elif c_positions_filtered:
            for ch, p, s in c_positions_filtered[:n_controls]:
                controls.append(
                    {"chrom": ch, "pos": p, "strand_inf": s, "gene": gene}
                )

    return pd.DataFrame(controls) if controls else pd.DataFrame()


def extract_sequences(positions_df, genome):
    """Extract 201-nt sequences for each position. Returns list (some may be None)."""
    seqs = []
    for _, row in positions_df.iterrows():
        chrom = row["chrom"]
        pos = int(row["pos"])
        strand = row["strand_inf"]

        try:
            chrom_len = len(genome[chrom])
            start = pos - 100
            end = pos + 101
            if start < 0 or end > chrom_len:
                seqs.append(None)
                continue

            seq = str(genome[chrom][start:end]).upper()
            if strand == "-":
                comp = str.maketrans("ACGT", "TGCA")
                seq = seq.translate(comp)[::-1]
            seqs.append(seq)
        except (KeyError, ValueError):
            seqs.append(None)

    return seqs


def extract_sequences_from_coords(chroms, positions, strands, genome):
    """Vectorized sequence extraction from genome. Returns list of 201-nt sequences or None."""
    seqs = []
    for chrom, pos, strand in zip(chroms, positions, strands):
        try:
            chrom_len = len(genome[chrom])
            start = int(pos) - 100
            end = int(pos) + 101
            if start < 0 or end > chrom_len:
                seqs.append(None)
                continue

            seq = str(genome[chrom][start:end]).upper()
            if strand == "-":
                comp = str.maketrans("ACGT", "TGCA")
                seq = seq.translate(comp)[::-1]
            seqs.append(seq)
        except (KeyError, ValueError):
            seqs.append(None)
    return seqs


# ============================================================================
# TCGA pipeline
# ============================================================================

def process_tcga_cancer(cancer, model, batch_converter, genome, gene_chrom_positions, batch_size, chunk=None, total_chunks=None):
    """Process one TCGA cancer type: extract sequences, embed, save."""
    output_path = EMB_DIR / f"rnafm_tcga_{cancer}.pt"
    checkpoint_path = EMB_DIR / f"rnafm_tcga_{cancer}_checkpoint.pt"

    # Check if already complete
    if output_path.exists():
        data = torch.load(output_path, map_location="cpu", weights_only=False)
        logger.info(
            "SKIP %s: already complete (%d mutation + %d control embeddings)",
            cancer.upper(),
            data["n_mut"],
            data["n_ctrl"],
        )
        del data
        return

    study = CANCER_STUDIES[cancer]
    maf_path = TCGA_MAF_DIR / f"{study}_mutations.txt"
    if not maf_path.exists():
        logger.warning("MAF not found for %s: %s", cancer, maf_path)
        return

    logger.info("=" * 60)
    logger.info("Processing %s", cancer.upper())

    # Parse mutations
    mut_df = parse_ct_mutations(maf_path)
    if len(mut_df) == 0:
        logger.warning("No mutations for %s", cancer)
        return

    # Generate matched controls (same seed=42 as original)
    logger.info("  Generating %d controls per mutation...", N_CONTROLS)
    ctrl_df = get_matched_controls(mut_df, gene_chrom_positions, N_CONTROLS)
    logger.info("  %d control positions", len(ctrl_df))

    # Extract sequences
    logger.info("  Extracting mutation sequences...")
    mut_seqs = extract_sequences(mut_df, genome)
    logger.info("  Extracting control sequences...")
    ctrl_seqs = extract_sequences(ctrl_df, genome)

    # Filter valid: 201-nt with C at center
    valid_mut = [
        s for s in mut_seqs if s is not None and len(s) == 201 and s[CENTER] == "C"
    ]
    valid_ctrl = [
        s for s in ctrl_seqs if s is not None and len(s) == 201 and s[CENTER] == "C"
    ]

    logger.info("  Valid: %d mutations, %d controls", len(valid_mut), len(valid_ctrl))

    # Combine all sequences (mutations first, then controls)
    all_seqs = valid_mut + valid_ctrl
    n_mut = len(valid_mut)
    n_ctrl = len(valid_ctrl)

    if len(all_seqs) == 0:
        logger.warning("  No valid sequences for %s", cancer)
        return

    # Convert T->U for RNA-FM
    all_seqs_rna = [s.upper().replace("T", "U") for s in all_seqs]

    # Chunk splitting: if chunk args passed via process_tcga_cancer kwargs
    # chunk and total_chunks come from function args
    if chunk is not None and total_chunks is not None:
        n_total = len(all_seqs_rna)
        chunk_size = (n_total + total_chunks - 1) // total_chunks
        start = chunk * chunk_size
        end = min(start + chunk_size, n_total)
        logger.info("  CHUNK %d/%d: processing indices %d-%d of %d", chunk, total_chunks, start, end, n_total)
        all_seqs_rna = all_seqs_rna[start:end]
        # Adjust n_mut/n_ctrl for this chunk
        n_mut_chunk = max(0, min(n_mut, end) - max(0, start))
        n_ctrl_chunk = len(all_seqs_rna) - n_mut_chunk
        n_mut, n_ctrl = n_mut_chunk, n_ctrl_chunk
        output_path = EMB_DIR / f"rnafm_tcga_{cancer}_chunk{chunk}.pt"
        checkpoint_path = EMB_DIR / f"rnafm_tcga_{cancer}_chunk{chunk}_checkpoint.pt"
        if output_path.exists():
            logger.info("  SKIP %s chunk %d: already complete", cancer, chunk)
            return

    # Compute embeddings
    logger.info("  Computing RNA-FM embeddings for %d sequences...", len(all_seqs_rna))
    pooled_orig, pooled_edited = compute_embeddings_batched(
        model,
        batch_converter,
        all_seqs_rna,
        batch_size=batch_size,
        checkpoint_every=50000,
        checkpoint_path=checkpoint_path,
    )

    # Save final result
    logger.info("  Saving to %s", output_path)
    torch.save(
        {
            "cancer": cancer,
            "n_mut": n_mut,
            "n_ctrl": n_ctrl,
            "n_total": len(all_seqs_rna),
            "chunk": chunk,
            "total_chunks": total_chunks,
            "pooled_orig": pooled_orig,
            "pooled_edited": pooled_edited,
        },
        output_path,
    )
    logger.info(
        "  Saved: %d embeddings (%d mut + %d ctrl), file size: %.1f MB",
        len(all_seqs_rna),
        n_mut,
        n_ctrl,
        output_path.stat().st_size / 1e6,
    )

    # Clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        logger.info("  Removed checkpoint file")

    # Free memory
    del pooled_orig, pooled_edited, all_seqs_rna, all_seqs
    gc.collect()


def run_tcga(cancers=None, batch_size=16, chunk=None, total_chunks=None):
    """Run RNA-FM embedding for TCGA cancers."""
    if cancers is None:
        cancers = CANCER_ORDER

    EMB_DIR.mkdir(parents=True, exist_ok=True)

    # Parse exonic regions (needed for control matching)
    logger.info("Parsing exonic regions from refGene (hg19)...")
    exons_by_gene = parse_exonic_regions()

    # Load genome
    logger.info("Loading hg19 genome...")
    genome = Fasta(str(HG19_FA))

    # Pre-compute C/G positions in all gene exons (done once, reused for all cancers)
    gene_chrom_positions = precompute_gene_c_positions(exons_by_gene, genome)

    # Load RNA-FM
    model, alphabet, batch_converter = load_rnafm()

    for cancer in cancers:
        if cancer not in CANCER_STUDIES:
            logger.warning("Unknown cancer type: %s", cancer)
            continue
        process_tcga_cancer(
            cancer, model, batch_converter, genome, gene_chrom_positions, batch_size,
            chunk=chunk, total_chunks=total_chunks,
        )

    logger.info("=" * 60)
    logger.info("TCGA embedding complete for: %s", ", ".join(cancers))


# ============================================================================
# ClinVar pipeline
# ============================================================================

def run_clinvar(batch_size=16, chunk=None, total_chunks=None):
    """Run RNA-FM embedding for ClinVar variants."""
    EMB_DIR.mkdir(parents=True, exist_ok=True)

    output_path = EMB_DIR / "rnafm_clinvar.pt"
    checkpoint_path = EMB_DIR / "rnafm_clinvar_checkpoint.pt"

    if output_path.exists():
        data = torch.load(output_path, map_location="cpu", weights_only=False)
        logger.info(
            "SKIP ClinVar: already complete (%d embeddings)", data["n_total"]
        )
        del data
        return

    # Load ClinVar scores CSV
    logger.info("Loading ClinVar scores CSV...")
    df = pd.read_csv(CLINVAR_CSV)
    logger.info("ClinVar: %d variants, columns: %s", len(df), list(df.columns))

    # Parse coordinates from site_id: "clinvar_chr1_69240_+"
    logger.info("Parsing coordinates from site_id...")
    parts = df["site_id"].str.split("_", expand=True)
    # site_id format: clinvar_chr1_69240_+
    df["chrom_parsed"] = parts[1]
    df["pos_parsed"] = parts[2].astype(int)
    df["strand_parsed"] = parts[3]

    # If chr/start columns exist, prefer them, otherwise use parsed
    if "chr" in df.columns and "start" in df.columns:
        chroms = df["chr"].tolist()
        positions = df["start"].tolist()
    else:
        chroms = df["chrom_parsed"].tolist()
        positions = df["pos_parsed"].tolist()

    # ClinVar uses hg38; infer strand from site_id
    strands = df["strand_parsed"].tolist()

    # Load hg38 genome
    logger.info("Loading hg38 genome...")
    genome = Fasta(str(HG38_FA))

    # Extract sequences
    logger.info("Extracting %d sequences from hg38...", len(df))
    t0 = time.time()
    all_seqs = extract_sequences_from_coords(chroms, positions, strands, genome)
    elapsed = time.time() - t0
    logger.info("Sequence extraction took %.0fs", elapsed)

    # Filter valid sequences
    valid_indices = []
    valid_seqs = []
    for i, s in enumerate(all_seqs):
        if s is not None and len(s) == 201 and s[CENTER] == "C":
            valid_indices.append(i)
            valid_seqs.append(s.upper().replace("T", "U"))

    logger.info(
        "Valid ClinVar sequences: %d / %d (%.1f%%)",
        len(valid_seqs),
        len(all_seqs),
        100.0 * len(valid_seqs) / max(len(all_seqs), 1),
    )

    if not valid_seqs:
        logger.error("No valid ClinVar sequences. Check genome / coordinate parsing.")
        return

    # Chunk splitting: if --chunk and --total-chunks are set, only process a slice
    if chunk is not None and total_chunks is not None:
        n_total = len(valid_seqs)
        chunk_size = (n_total + total_chunks - 1) // total_chunks
        start = chunk * chunk_size
        end = min(start + chunk_size, n_total)
        logger.info("CHUNK %d/%d: processing indices %d-%d of %d", chunk, total_chunks, start, end, n_total)
        valid_seqs = valid_seqs[start:end]
        valid_indices = valid_indices[start:end]
        output_path = EMB_DIR / f"rnafm_clinvar_chunk{chunk}.pt"
        checkpoint_path = EMB_DIR / f"rnafm_clinvar_chunk{chunk}_checkpoint.pt"
        if output_path.exists():
            logger.info("SKIP ClinVar chunk %d: already complete", chunk)
            return

    # Load RNA-FM
    model, alphabet, batch_converter = load_rnafm()

    # Compute embeddings
    logger.info("Computing RNA-FM embeddings for %d ClinVar sequences...", len(valid_seqs))
    pooled_orig, pooled_edited = compute_embeddings_batched(
        model,
        batch_converter,
        valid_seqs,
        batch_size=batch_size,
        checkpoint_every=50000,
        checkpoint_path=checkpoint_path,
    )

    # Save: store site_ids for valid entries so we can map back
    valid_site_ids = [df.iloc[i]["site_id"] for i in valid_indices]
    logger.info("Saving to %s", output_path)
    torch.save(
        {
            "site_ids": valid_site_ids,
            "valid_indices": valid_indices,
            "n_total": len(valid_seqs),
            "n_original": len(df) if chunk is None else len(valid_seqs),
            "chunk": chunk,
            "total_chunks": total_chunks,
            "pooled_orig": pooled_orig,
            "pooled_edited": pooled_edited,
        },
        output_path,
    )
    logger.info(
        "Saved: %d embeddings, file size: %.1f MB",
        len(valid_seqs),
        output_path.stat().st_size / 1e6,
    )

    # Clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        logger.info("Removed checkpoint file")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compute RNA-FM embeddings for TCGA and ClinVar"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="tcga",
        choices=["tcga", "clinvar", "all"],
        help="What to compute: tcga, clinvar, or all",
    )
    parser.add_argument(
        "--cancer",
        type=str,
        default=None,
        help="Specific cancer type(s), comma-separated (e.g., lihc,esca). Default: all 10",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="RNA-FM batch size (default 16)",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=None,
        help="Chunk index (0-based) for parallel splitting. Use with --total-chunks.",
    )
    parser.add_argument(
        "--total-chunks",
        type=int,
        default=None,
        help="Total number of chunks to split into. Each chunk processes a disjoint slice.",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("RNA-FM Embedding: TCGA + ClinVar")
    logger.info("Mode: %s", args.mode)
    logger.info("Device: %s", DEVICE)
    logger.info("Batch size: %d", args.batch_size)
    logger.info("=" * 60)

    if args.mode in ("tcga", "all"):
        cancers = None
        if args.cancer:
            cancers = [c.strip() for c in args.cancer.split(",")]
            logger.info("TCGA cancers: %s", cancers)
        run_tcga(cancers=cancers, batch_size=args.batch_size,
                 chunk=args.chunk, total_chunks=args.total_chunks)

    if args.mode in ("clinvar", "all"):
        run_clinvar(batch_size=args.batch_size,
                    chunk=args.chunk, total_chunks=args.total_chunks)

    logger.info("=" * 60)
    logger.info("ALL DONE")


if __name__ == "__main__":
    main()
