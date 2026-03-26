"""Negative site generation utilities for APOBEC editing prediction.

Provides functions for generating negative control sites (unedited cytidines)
from existing split files or from genome sequences with motif matching.

All sequences are 201-nt with the edit site (C) at center position 100 (0-indexed).
"""

import logging
import random
from pathlib import Path
from typing import Dict, Optional, Set

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FLANK_SIZE = 100  # +/-100nt around edit site -> 201nt total
CENTER = 100      # Edit site position (0-indexed)


def generate_negatives_from_splits(
    source_splits_csv: Path,
    is_edited_col: str = "is_edited",
    label_col: str = "label",
) -> pd.DataFrame:
    """Return rows where is_edited==0 (or label==0) from a splits CSV.

    Tries is_edited_col first, then falls back to label_col.

    Args:
        source_splits_csv: Path to splits CSV file.
        is_edited_col: Column name for editing status (default 'is_edited').
        label_col: Fallback column name for label (default 'label').

    Returns:
        DataFrame containing only negative (unedited) rows.
    """
    df = pd.read_csv(source_splits_csv)

    if is_edited_col in df.columns:
        neg_df = df[df[is_edited_col] == 0].copy()
        logger.info("Found %d negatives via '%s' column in %s",
                     len(neg_df), is_edited_col, source_splits_csv.name)
    elif label_col in df.columns:
        neg_df = df[df[label_col] == 0].copy()
        logger.info("Found %d negatives via '%s' column in %s",
                     len(neg_df), label_col, source_splits_csv.name)
    else:
        raise ValueError(
            f"Neither '{is_edited_col}' nor '{label_col}' found in {source_splits_csv}"
        )

    return neg_df


def extract_from_genome(genome, chrom: str, pos: int, strand: str) -> Optional[str]:
    """Extract 201-nt RNA sequence centered on pos (1-based) from genome.

    Performs reverse complement for minus-strand sites and converts DNA to RNA.
    Verifies the center base is C (the edit target).

    Args:
        genome: pyfaidx.Fasta object.
        chrom: Chromosome name (e.g., 'chr1').
        pos: 1-based genomic position.
        strand: '+' or '-'.

    Returns:
        201-nt RNA sequence with C at center, or None if out of bounds
        or center base is not C.
    """
    if chrom not in genome:
        return None

    chrom_len = len(genome[chrom])
    pos_0 = pos - 1  # Convert to 0-based

    seq_start = pos_0 - FLANK_SIZE
    seq_end = pos_0 + FLANK_SIZE + 1

    if seq_start < 0 or seq_end > chrom_len:
        return None

    dna_seq = str(genome[chrom][seq_start:seq_end]).upper()

    if strand == "-":
        comp = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
        dna_seq = "".join(comp.get(b, "N") for b in reversed(dna_seq))

    rna_seq = dna_seq.replace("T", "U")

    # Verify center is C
    if rna_seq[CENTER] != "C":
        return None

    return rna_seq


def generate_negatives_from_genome(
    positives_df: pd.DataFrame,
    genome_fa: Path,
    target_tc_fraction: float,
    target_cc_fraction: float,
    n_negatives: int,
    output_seqs: Dict[str, str],
    known_sites: Optional[Set] = None,
    search_window: int = 5000,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate motif-matched negatives from genome near positive sites.

    For each positive site, searches within +/- search_window for unedited C bases.
    Selects negatives to approximately match the desired TC% and CC% fractions.

    Args:
        positives_df: DataFrame of positive sites with chr, start, strand, enzyme,
                      dataset_source columns.
        genome_fa: Path to hg38.fa genome file.
        target_tc_fraction: Desired fraction of negatives with TC dinucleotide context.
        target_cc_fraction: Desired fraction of negatives with CC dinucleotide context.
        n_negatives: Total number of negatives to generate.
        output_seqs: Dict to append {site_id: sequence} into (modified in place).
        known_sites: Set of (chrom, pos) tuples to exclude. If None, empty set used.
        search_window: Search radius around each positive site (default 5000).
        seed: Random seed for reproducibility.

    Returns:
        DataFrame of generated negative sites with columns matching positives_df.
    """
    from pyfaidx import Fasta

    genome = Fasta(str(genome_fa))
    rng = random.Random(seed)

    if known_sites is None:
        known_sites = set()
    # Make a mutable copy to track used positions
    used_sites = set(known_sites)

    # Collect all candidate negatives from regions near positives
    all_candidates = []

    valid_positives = positives_df[
        positives_df["chr"].notna() & positives_df["start"].notna()
    ].copy()
    logger.info("Searching for negative candidates near %d positive sites...",
                len(valid_positives))

    for _, row in valid_positives.iterrows():
        chrom = str(row["chr"])
        pos = int(row["start"])
        strand = str(row["strand"])
        enzyme = str(row["enzyme"])
        ds = str(row["dataset_source"])

        if chrom not in genome:
            continue

        chrom_len = len(genome[chrom])
        search_start = max(0, pos - search_window)
        search_end = min(chrom_len, pos + search_window)

        region_seq = str(genome[chrom][search_start:search_end]).upper()

        # Find C positions (+ strand) or G positions (- strand)
        target_base = "C" if strand == "+" else "G"
        for i, base in enumerate(region_seq):
            if base == target_base:
                genome_pos = search_start + i + 1  # 1-based
                if (chrom, genome_pos) in used_sites or genome_pos == pos:
                    continue
                # Ensure enough flanking context
                pos_0 = genome_pos - 1
                if pos_0 < FLANK_SIZE or pos_0 + FLANK_SIZE >= chrom_len:
                    continue

                # Get the upstream base to determine dinucleotide context
                if strand == "+":
                    up_base = region_seq[i - 1].upper() if i > 0 else "N"
                    dinuc = up_base + "C"
                else:
                    down_base = region_seq[i + 1].upper() if i + 1 < len(region_seq) else "N"
                    comp = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
                    dinuc = comp.get(down_base, "N") + "C"

                is_tc = dinuc == "TC"
                is_cc = dinuc == "CC"

                all_candidates.append({
                    "chrom": chrom,
                    "genome_pos": genome_pos,
                    "strand": strand,
                    "enzyme": enzyme,
                    "dataset_source": ds,
                    "is_tc": is_tc,
                    "is_cc": is_cc,
                })

    logger.info("Found %d candidate negative positions", len(all_candidates))

    if not all_candidates:
        logger.warning("No candidates found, returning empty DataFrame")
        return pd.DataFrame()

    # Split candidates by motif context
    tc_cands = [c for c in all_candidates if c["is_tc"]]
    cc_cands = [c for c in all_candidates if c["is_cc"]]
    other_cands = [c for c in all_candidates if not c["is_tc"] and not c["is_cc"]]

    # Determine how many of each motif type to select
    n_tc = int(round(n_negatives * target_tc_fraction))
    n_cc = int(round(n_negatives * target_cc_fraction))
    n_other = n_negatives - n_tc - n_cc

    # Clip to available candidates
    n_tc = min(n_tc, len(tc_cands))
    n_cc = min(n_cc, len(cc_cands))
    n_other = min(n_other, len(other_cands))

    # If we don't have enough of a specific type, fill from others
    remaining = n_negatives - n_tc - n_cc - n_other
    if remaining > 0:
        # Try to fill from whichever pool has extras
        for pool, current_n in [(tc_cands, n_tc), (cc_cands, n_cc), (other_cands, n_other)]:
            extra_available = len(pool) - current_n
            if extra_available > 0:
                add = min(remaining, extra_available)
                if pool is tc_cands:
                    n_tc += add
                elif pool is cc_cands:
                    n_cc += add
                else:
                    n_other += add
                remaining -= add
                if remaining == 0:
                    break

    rng.shuffle(tc_cands)
    rng.shuffle(cc_cands)
    rng.shuffle(other_cands)

    selected = tc_cands[:n_tc] + cc_cands[:n_cc] + other_cands[:n_other]
    rng.shuffle(selected)

    logger.info("Selected %d negatives (TC=%d, CC=%d, other=%d)",
                len(selected), n_tc, n_cc, n_other)

    # Extract sequences and build output
    negatives = []
    for cand in selected:
        chrom = cand["chrom"]
        genome_pos = cand["genome_pos"]
        strand = cand["strand"]

        if (chrom, genome_pos) in used_sites:
            continue

        rna_seq = extract_from_genome(genome, chrom, genome_pos, strand)
        if rna_seq is None:
            continue

        site_id = f"{chrom}:{genome_pos}:{strand}"
        used_sites.add((chrom, genome_pos))
        output_seqs[site_id] = rna_seq

        negatives.append({
            "site_id": site_id,
            "chr": chrom,
            "start": genome_pos,
            "end": genome_pos,
            "strand": strand,
            "enzyme": cand["enzyme"],
            "dataset_source": cand["dataset_source"] + "_neg",
            "is_edited": 0,
            "editing_rate": 0.0,
            "coordinate_system": "hg38",
            "source_type": "negative_control",
        })

    neg_df = pd.DataFrame(negatives)
    logger.info("Generated %d negative controls", len(neg_df))

    # Report actual motif fractions
    if len(neg_df) > 0 and len(output_seqs) > 0:
        tc_count = sum(
            1 for _, row in neg_df.iterrows()
            if output_seqs.get(row["site_id"], "N" * 201)[CENTER - 1] == "U"
        )
        cc_count = sum(
            1 for _, row in neg_df.iterrows()
            if output_seqs.get(row["site_id"], "N" * 201)[CENTER - 1] == "C"
        )
        logger.info("Actual motif fractions: TC=%.1f%%, CC=%.1f%%",
                     100 * tc_count / len(neg_df), 100 * cc_count / len(neg_df))

    return neg_df
