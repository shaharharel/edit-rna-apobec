#!/usr/bin/env python
"""Download and process ClinVar data for C-to-U editing analysis.

Downloads ClinVar VCF for GRCh38/hg38, filters to C>T (sense) and G>A
(antisense) SNVs that represent potential C-to-U editing events, and
cross-references with known APOBEC3A editing sites.

Output:
    data/processed/clinvar_c2u_variants.csv

Usage:
    python scripts/apobec/download_clinvar.py
"""

import gzip
import json
import logging
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logger = logging.getLogger(__name__)

CLINVAR_URL = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz"
CLINVAR_VCF_GZ = PROJECT_ROOT / "data" / "raw" / "clinvar" / "clinvar_grch38.vcf.gz"
COMBINED_PATH = PROJECT_ROOT / "data" / "processed" / "all_datasets_combined.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "clinvar_c2u_variants.csv"
GENOME_PATH = PROJECT_ROOT / "data" / "raw" / "genomes" / "hg38.fa"

WINDOW_SIZE = 100  # ±100 nt for sequence extraction


def download_clinvar(url, output_path):
    """Download ClinVar VCF if not already present."""
    if output_path.exists():
        logger.info("ClinVar VCF already exists: %s (%.1f MB)",
                     output_path, output_path.stat().st_size / 1e6)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading ClinVar VCF from %s ...", url)
    logger.info("This may take a few minutes (~50 MB)")

    t0 = time.time()

    def _report(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100.0 / total_size)
            if block_num % 500 == 0:
                logger.info("  %.1f%% (%.1f / %.1f MB)",
                            pct, downloaded / 1e6, total_size / 1e6)

    urllib.request.urlretrieve(url, str(output_path), reporthook=_report)
    elapsed = time.time() - t0
    logger.info("Downloaded in %.1fs (%.1f MB)",
                elapsed, output_path.stat().st_size / 1e6)


def parse_info_field(info_str):
    """Parse VCF INFO field into a dictionary."""
    result = {}
    for entry in info_str.split(";"):
        if "=" in entry:
            key, val = entry.split("=", 1)
            result[key] = val
        else:
            result[entry] = True
    return result


def parse_clinvar_vcf(vcf_path):
    """Parse ClinVar VCF and extract C>T and G>A SNVs.

    C>T on + strand = C-to-U editing (sense)
    G>A on + strand = C-to-U editing on antisense strand
    """
    logger.info("Parsing ClinVar VCF: %s", vcf_path)
    t0 = time.time()

    variants = []
    n_total = 0
    n_snv = 0

    open_fn = gzip.open if str(vcf_path).endswith(".gz") else open

    with open_fn(vcf_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue

            n_total += 1
            fields = line.strip().split("\t")
            if len(fields) < 8:
                continue

            chrom, pos, var_id, ref, alt, qual, filt, info_str = fields[:8]

            # Only SNVs
            if len(ref) != 1 or len(alt) != 1:
                continue

            n_snv += 1

            # C>T (sense strand C-to-U) or G>A (antisense C-to-U)
            is_c2t = ref == "C" and alt == "T"
            is_g2a = ref == "G" and alt == "A"

            if not (is_c2t or is_g2a):
                continue

            # Parse INFO
            info = parse_info_field(info_str)

            # Add chr prefix if needed
            if not chrom.startswith("chr"):
                chrom = f"chr{chrom}"

            # Determine editing strand
            if is_c2t:
                editing_strand = "+"
            else:
                editing_strand = "-"

            variant = {
                "chr": chrom,
                "start": int(pos) - 1,  # VCF is 1-based, convert to 0-based
                "ref": ref,
                "alt": alt,
                "editing_strand": editing_strand,
                "clinvar_id": var_id,
                "clinical_significance": info.get("CLNSIG", "unknown"),
                "review_status": info.get("CLNREVSTAT", "unknown"),
                "condition": info.get("CLNDN", "unknown"),
                "gene": info.get("GENEINFO", "unknown").split(":")[0],
                "molecular_consequence": info.get("MC", "unknown"),
                "origin": info.get("ORIGIN", "unknown"),
                "dbsnp_id": info.get("RS", ""),
            }
            variants.append(variant)

            if n_total % 500000 == 0:
                logger.info("  Processed %d lines, %d SNVs, %d C>T/G>A",
                            n_total, n_snv, len(variants))

    elapsed = time.time() - t0
    logger.info("Parsed %d total variants, %d SNVs, %d C>T/G>A variants (%.1fs)",
                n_total, n_snv, len(variants), elapsed)

    return pd.DataFrame(variants)


def cross_reference_editing_sites(clinvar_df, combined_path):
    """Cross-reference ClinVar variants with known APOBEC3A editing sites."""
    if not combined_path.exists():
        logger.warning("Combined dataset not found: %s", combined_path)
        clinvar_df["is_known_editing_site"] = False
        clinvar_df["editing_dataset"] = ""
        return clinvar_df

    combined = pd.read_csv(combined_path)
    known_coords = {}
    for _, row in combined.iterrows():
        key = (str(row["chr"]), int(row["start"]))
        known_coords[key] = row.get("dataset_source", "unknown")

    is_known = []
    ds_source = []
    for _, row in clinvar_df.iterrows():
        key = (row["chr"], row["start"])
        if key in known_coords:
            is_known.append(True)
            ds_source.append(known_coords[key])
        else:
            is_known.append(False)
            ds_source.append("")

    clinvar_df["is_known_editing_site"] = is_known
    clinvar_df["editing_dataset"] = ds_source

    n_overlap = sum(is_known)
    logger.info("Cross-reference: %d ClinVar C>T variants overlap known editing sites",
                n_overlap)

    return clinvar_df


def extract_sequences(clinvar_df, genome_path, window_size=WINDOW_SIZE):
    """Extract genomic sequences around each ClinVar variant."""
    if not genome_path.exists():
        logger.warning("Genome not found: %s. Skipping sequence extraction.", genome_path)
        clinvar_df["sequence"] = ""
        return clinvar_df

    from pyfaidx import Fasta
    logger.info("Loading genome for sequence extraction...")
    genome = Fasta(str(genome_path))

    sequences = []
    for _, row in clinvar_df.iterrows():
        chrom = row["chr"]
        pos = row["start"]
        strand = row["editing_strand"]

        if chrom not in genome:
            sequences.append("")
            continue

        chrom_len = len(genome[chrom])
        start = max(0, pos - window_size)
        end = min(chrom_len, pos + window_size + 1)

        seq = str(genome[chrom][start:end]).upper()

        if strand == "-":
            comp = {"A": "U", "T": "A", "C": "G", "G": "C", "N": "N"}
            seq = "".join(comp.get(b, "N") for b in reversed(seq))
        else:
            seq = seq.replace("T", "U")

        # Pad if at chromosome boundary
        left_pad = window_size - (pos - start)
        right_pad = window_size - (end - pos - 1)
        if left_pad > 0:
            seq = "N" * left_pad + seq
        if right_pad > 0:
            seq = seq + "N" * right_pad

        sequences.append(seq)

    clinvar_df["sequence"] = sequences
    n_with_seq = sum(1 for s in sequences if len(s) == 2 * window_size + 1)
    logger.info("Extracted %d sequences (201 nt) for %d variants",
                n_with_seq, len(clinvar_df))

    return clinvar_df


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    t0 = time.time()

    # Step 1: Download ClinVar VCF
    download_clinvar(CLINVAR_URL, CLINVAR_VCF_GZ)

    # Step 2: Parse VCF, filter to C>T / G>A
    clinvar_df = parse_clinvar_vcf(CLINVAR_VCF_GZ)

    if len(clinvar_df) == 0:
        logger.error("No C>T/G>A variants found in ClinVar!")
        return

    # Step 3: Cross-reference with known editing sites
    clinvar_df = cross_reference_editing_sites(clinvar_df, COMBINED_PATH)

    # Step 4: Extract sequences from hg19
    clinvar_df = extract_sequences(clinvar_df, GENOME_PATH)

    # Step 5: Create site_ids
    clinvar_df["site_id"] = clinvar_df.apply(
        lambda r: f"clinvar_{r['chr']}_{r['start']}_{r['editing_strand']}", axis=1
    )

    # Step 6: Add summary statistics
    logger.info("\n" + "=" * 70)
    logger.info("CLINVAR C>T/G>A SUMMARY")
    logger.info("=" * 70)
    logger.info("Total C>T/G>A variants: %d", len(clinvar_df))

    # Pathogenicity breakdown
    sig_counts = clinvar_df["clinical_significance"].value_counts()
    logger.info("\nClinical significance:")
    for sig, count in sig_counts.head(10).items():
        logger.info("  %s: %d", sig, count)

    # Strand breakdown
    strand_counts = clinvar_df["editing_strand"].value_counts()
    logger.info("\nEditing strand: + (C>T): %d, - (G>A): %d",
                strand_counts.get("+", 0), strand_counts.get("-", 0))

    # Known editing sites
    n_known = clinvar_df["is_known_editing_site"].sum()
    logger.info("\nOverlap with known editing sites: %d (%.2f%%)",
                n_known, 100 * n_known / len(clinvar_df))

    if n_known > 0:
        known = clinvar_df[clinvar_df["is_known_editing_site"]]
        logger.info("Known editing sites in ClinVar:")
        for _, row in known.head(20).iterrows():
            logger.info("  %s %s:%d %s (%s) - %s",
                        row["gene"], row["chr"], row["start"],
                        row["clinical_significance"],
                        row["editing_dataset"],
                        row["condition"])

    # Chromosome breakdown
    chrom_counts = clinvar_df["chr"].value_counts()
    logger.info("\nTop chromosomes:")
    for chrom, count in chrom_counts.head(5).items():
        logger.info("  %s: %d", chrom, count)

    # Step 7: Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    clinvar_df.to_csv(OUTPUT_PATH, index=False)
    logger.info("\nSaved %d variants to %s", len(clinvar_df), OUTPUT_PATH)

    elapsed = time.time() - t0
    logger.info("Total pipeline time: %.1fs", elapsed)


if __name__ == "__main__":
    main()
