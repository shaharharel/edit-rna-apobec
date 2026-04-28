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


# ---------------------------------------------------------------------------
# v4: cancer-matched trinucleotide negative generation
# ---------------------------------------------------------------------------

_BASES = ("A", "C", "G", "T")
_ALL_TRINUCS = tuple(f"{a}C{b}" for a in _BASES for b in _BASES)
_COMP = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}


def _revcomp(s: str) -> str:
    return "".join(_COMP.get(b, "N") for b in reversed(s.upper()))


def _trinuc_distance(a: str, b: str) -> int:
    """Hamming-like distance between two trinuc contexts. Center is always C."""
    if len(a) != 3 or len(b) != 3:
        return 999
    return int(a[0] != b[0]) + int(a[2] != b[2])


def _fallback_order(target: str) -> list:
    """Return ordered list of trinucs from closest to farthest from target.

    Distance: number of mismatched flanking bases (0, 1 or 2). Within the
    same distance, ordered by insertion order in _ALL_TRINUCS.
    """
    return sorted(
        [t for t in _ALL_TRINUCS if t != target],
        key=lambda t: (_trinuc_distance(target, t), _ALL_TRINUCS.index(t)),
    )


def generate_negatives_cancer_matched(
    positives_df: "pd.DataFrame",
    genomes: Dict[str, "Path"],
    cancer_trinuc_fractions: Dict[str, float],
    n_negatives: int,
    output_seqs: Dict[str, str],
    known_sites: Optional[Set] = None,
    search_window: int = 5000,
    seed: int = 20260427,
) -> "pd.DataFrame":
    """Generate negatives whose trinucleotide distribution matches a target.

    Designed for v4: removes the v3 anti-TCW bias by sampling negatives so the
    distribution of (5'N, C, 3'N) contexts matches cancer C>T mutations.

    Algorithm:
      1) For each positive site, search +/- search_window bp on the
         appropriate genome (selected via the row's coordinate_system) for
         unedited C bases. Skip the positive site itself and any known sites.
      2) Bin candidates by 16 trinucleotide contexts.
      3) Compute per-bin quotas from cancer_trinuc_fractions * n_negatives.
      4) Sample without replacement from each bin. If a bin runs out, fill
         from the next-most-similar bin (Hamming distance over the two
         flanking bases).

    Args:
        positives_df: DataFrame of positive sites with chr, start, strand,
            enzyme, dataset_source. Must also have a coordinate_system column;
            rows with hg19 are looked up against genomes['hg19'], rows with
            anything else against genomes['hg38'].
        genomes: Mapping {'hg19': Path, 'hg38': Path} to FASTA files.
        cancer_trinuc_fractions: Dict {trinuc: fraction} (e.g. from cancer
            distribution CSV). Should sum to ~1.0. Missing keys default to 0.
        n_negatives: Total number of negatives to generate.
        output_seqs: Dict to append {site_id: sequence} into (modified in-place).
        known_sites: Set of (chrom, pos) tuples to exclude. If None, empty.
        search_window: Search radius around each positive site (default 5000).
        seed: Random seed (default 20260427).

    Returns:
        DataFrame of generated negative sites.
    """
    from pyfaidx import Fasta

    if "hg19" not in genomes or "hg38" not in genomes:
        raise ValueError("genomes dict must contain both 'hg19' and 'hg38' keys")

    g_hg19 = Fasta(str(genomes["hg19"]))
    g_hg38 = Fasta(str(genomes["hg38"]))
    rng = random.Random(seed)

    if known_sites is None:
        known_sites = set()
    used_sites = set(known_sites)

    # 1) Collect candidates, binned by trinuc.
    candidates_by_trinuc: Dict[str, list] = {t: [] for t in _ALL_TRINUCS}

    if "coordinate_system" not in positives_df.columns:
        # Best-effort fallback: assume hg38
        positives_df = positives_df.copy()
        positives_df["coordinate_system"] = "hg38"

    valid_positives = positives_df[
        positives_df["chr"].notna() & positives_df["start"].notna()
    ].copy()
    logger.info("v4 cancer-matched: scanning near %d positives (window=%d)",
                len(valid_positives), search_window)

    n_processed = 0
    for _, row in valid_positives.iterrows():
        chrom = str(row["chr"])
        pos = int(row["start"])
        strand = str(row["strand"])
        enzyme = str(row["enzyme"])
        ds = str(row["dataset_source"])
        coord = str(row.get("coordinate_system", "hg38")).lower()
        genome = g_hg19 if coord == "hg19" else g_hg38

        if chrom not in genome:
            continue

        chrom_len = len(genome[chrom])
        # Need 1 extra base on each side to extract the trinucleotide.
        search_start = max(1, pos - search_window)
        search_end = min(chrom_len, pos + search_window)
        if search_end - search_start < 3:
            continue

        # Pull the slice plus one base on each side for trinuc extraction
        slice_start = max(0, search_start - 1)
        slice_end = min(chrom_len, search_end + 1)
        region_seq = str(genome[chrom][slice_start:slice_end]).upper()

        target_base = "C" if strand == "+" else "G"
        # Iterate over bases in [search_start, search_end), skipping first & last
        # of region_seq (those are flanking).
        for i in range(1, len(region_seq) - 1):
            base = region_seq[i]
            if base != target_base:
                continue
            genome_pos = slice_start + i + 1  # 1-based
            if (chrom, genome_pos) in used_sites or genome_pos == pos:
                continue
            # Need 100 bp context on each side for the 201-nt sequence
            pos_0 = genome_pos - 1
            if pos_0 < FLANK_SIZE or pos_0 + FLANK_SIZE >= chrom_len:
                continue

            # Trinucleotide on the editing strand
            if strand == "+":
                triplet = region_seq[i - 1:i + 2]  # plus-strand flanks + C
            else:
                # Negative strand: revcomp the genomic G-centered triplet
                triplet = _revcomp(region_seq[i - 1:i + 2])
            if len(triplet) != 3 or "N" in triplet or triplet[1] != "C":
                continue

            candidates_by_trinuc[triplet].append({
                "chrom": chrom,
                "genome_pos": genome_pos,
                "strand": strand,
                "enzyme": enzyme,
                "dataset_source": ds,
                "coordinate_system": coord,
                "trinuc": triplet,
            })
        n_processed += 1
        if n_processed % 1000 == 0:
            logger.info("  scanned %d / %d positives, total candidates=%d",
                        n_processed, len(valid_positives),
                        sum(len(v) for v in candidates_by_trinuc.values()))

    total_cands = sum(len(v) for v in candidates_by_trinuc.values())
    logger.info("v4: %d candidates across 16 trinuc bins", total_cands)
    for t in _ALL_TRINUCS:
        logger.info("  %s: %d candidates", t, len(candidates_by_trinuc[t]))

    # 2) Compute target counts per bin (largest-remainder rounding).
    fractions = {t: max(0.0, float(cancer_trinuc_fractions.get(t, 0.0))) for t in _ALL_TRINUCS}
    fr_sum = sum(fractions.values())
    if fr_sum <= 0:
        raise ValueError("cancer_trinuc_fractions sum to 0; cannot build quotas")
    fractions = {t: v / fr_sum for t, v in fractions.items()}

    raw = {t: fractions[t] * n_negatives for t in _ALL_TRINUCS}
    floors = {t: int(np.floor(raw[t])) for t in _ALL_TRINUCS}
    remainder = n_negatives - sum(floors.values())
    # Distribute remainder by largest fractional part
    fr_parts = sorted(_ALL_TRINUCS, key=lambda t: (raw[t] - floors[t]), reverse=True)
    targets = dict(floors)
    for t in fr_parts[:max(0, remainder)]:
        targets[t] += 1

    logger.info("v4 quotas (target -> available):")
    for t in _ALL_TRINUCS:
        logger.info("  %s: target=%d available=%d", t, targets[t], len(candidates_by_trinuc[t]))

    # 3) Sample from each bin; spill overflow to next-most-similar bin.
    # Shuffle each bin once for unbiased selection.
    for t in _ALL_TRINUCS:
        rng.shuffle(candidates_by_trinuc[t])

    selected = []
    leftover_target = 0  # target bin units that were not satisfied (will be redistributed)
    bin_overflow = {t: 0 for t in _ALL_TRINUCS}  # how short each bin is

    # First pass: take min(target, available) from each bin.
    take_idx = {t: 0 for t in _ALL_TRINUCS}
    for t in _ALL_TRINUCS:
        avail = len(candidates_by_trinuc[t])
        want = targets[t]
        take = min(want, avail)
        if take > 0:
            selected.extend(candidates_by_trinuc[t][:take])
            take_idx[t] = take
        if want > avail:
            bin_overflow[t] = want - avail
            leftover_target += want - avail

    # Second pass: redistribute deficits to most-similar bins.
    if leftover_target > 0:
        logger.info("v4: %d slots to redistribute from depleted bins", leftover_target)
        for t in _ALL_TRINUCS:
            need = bin_overflow[t]
            if need <= 0:
                continue
            for fb in _fallback_order(t):
                if need <= 0:
                    break
                avail = len(candidates_by_trinuc[fb]) - take_idx[fb]
                if avail <= 0:
                    continue
                grab = min(need, avail)
                selected.extend(candidates_by_trinuc[fb][take_idx[fb]:take_idx[fb] + grab])
                take_idx[fb] += grab
                need -= grab
            if need > 0:
                logger.warning("  could not fully fill %s: still short %d", t, need)

    rng.shuffle(selected)
    logger.info("v4: %d candidates selected (target=%d)", len(selected), n_negatives)

    # 4) Extract sequences and build output rows.
    negatives = []
    for cand in selected:
        chrom = cand["chrom"]
        genome_pos = cand["genome_pos"]
        strand = cand["strand"]
        coord = cand["coordinate_system"]
        if (chrom, genome_pos) in used_sites:
            continue
        genome = g_hg19 if coord == "hg19" else g_hg38
        rna_seq = extract_from_genome(genome, chrom, genome_pos, strand)
        if rna_seq is None:
            continue
        site_id = f"{chrom}:{genome_pos}:{strand}"
        if site_id in output_seqs:
            # Avoid duplicate site_id collisions across coord systems
            site_id = f"{chrom}:{genome_pos}:{strand}:{coord}"
            if site_id in output_seqs:
                continue
        used_sites.add((chrom, genome_pos))
        output_seqs[site_id] = rna_seq

        negatives.append({
            "site_id": site_id,
            "chr": chrom,
            "start": genome_pos,
            "end": genome_pos,
            "strand": strand,
            "enzyme": cand["enzyme"],
            "dataset_source": cand["dataset_source"] + "_neg_v4",
            "is_edited": 0,
            "editing_rate": 0.0,
            "coordinate_system": coord,
            "source_type": "negative_control_cancer_matched",
            "trinuc": cand["trinuc"],
        })

    neg_df = pd.DataFrame(negatives)
    logger.info("v4: produced %d negative rows (after sequence extraction)", len(neg_df))
    return neg_df
