#!/usr/bin/env python
"""Generate v4 negative datasets for deep learning training.

Creates three negative datasets with the SAME 7,564 positives from v3:
  1. v4-random:  75,640 random exonic C positions (1:10 ratio, no motif matching)
  2. v4-hard:    37,820 TC-context + unpaired loop negatives (1:5 ratio)
  3. v4-large:   378,200 mixed negatives (1:50 ratio)

All sequences are 201-nt with C at position 100 (0-indexed).
All negatives use hg38 coordinates.

Output files:
  data/processed/multi_enzyme/splits_v4_random_negatives.csv
  data/processed/multi_enzyme/multi_enzyme_sequences_v4_random.json
  data/processed/multi_enzyme/splits_v4_hard_negatives.csv
  data/processed/multi_enzyme/multi_enzyme_sequences_v4_hard.json
  data/processed/multi_enzyme/splits_v4_large_negatives.csv
  data/processed/multi_enzyme/multi_enzyme_sequences_v4_large.json

Usage:
    conda run -n quris python scripts/multi_enzyme/generate_v4_negatives.py [--step 1|2|3|all]
"""

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.apobec_negatives import extract_from_genome, FLANK_SIZE, CENTER

logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
V3_SPLITS = DATA_DIR / "processed/multi_enzyme/splits_multi_enzyme_v3_with_negatives.csv"
V3_SEQS = DATA_DIR / "processed/multi_enzyme/multi_enzyme_sequences_v3_with_negatives.json"
GENOME_HG38 = DATA_DIR / "raw/genomes/hg38.fa"
REFGENE_HG38 = DATA_DIR / "raw/genomes/refGene.txt"

OUT_DIR = DATA_DIR / "processed/multi_enzyme"

# Target counts
N_POSITIVES = 7564
N_RANDOM = 75640       # 1:10
N_HARD = 37820         # 1:5
N_LARGE = 378200       # 1:50

# Exclusion radius around known editing sites (bp)
EXCLUSION_RADIUS = 100

SEED = 42


def parse_refgene_exons(refgene_path: Path) -> dict:
    """Parse refGene.txt and return dict of chrom -> list of (start, end, strand) exon intervals.

    Uses coding exonic regions (cdsStart to cdsEnd intersected with exons).
    Falls back to full exon spans for non-coding transcripts.
    Coordinates are 0-based half-open.
    """
    logger.info("Parsing refGene exons from %s", refgene_path)
    exons_by_chrom = defaultdict(list)

    with open(refgene_path) as f:
        for line in f:
            fields = line.strip().split("\t")
            if len(fields) < 11:
                continue
            chrom = fields[2]
            strand = fields[3]
            cds_start = int(fields[6])  # 0-based
            cds_end = int(fields[7])    # 0-based exclusive

            # Skip non-standard chromosomes
            if "_" in chrom or chrom.startswith("chrUn"):
                continue

            exon_starts = [int(x) for x in fields[9].rstrip(",").split(",") if x]
            exon_ends = [int(x) for x in fields[10].rstrip(",").split(",") if x]

            for es, ee in zip(exon_starts, exon_ends):
                # Intersect with CDS if available
                if cds_start < cds_end:
                    clip_start = max(es, cds_start)
                    clip_end = min(ee, cds_end)
                    if clip_start >= clip_end:
                        continue
                    exons_by_chrom[chrom].append((clip_start, clip_end, strand))
                else:
                    # Non-coding — include full exon
                    exons_by_chrom[chrom].append((es, ee, strand))

    # Merge overlapping intervals per chromosome
    merged = {}
    for chrom, intervals in exons_by_chrom.items():
        # Keep strand info — just deduplicate
        # For efficiency, merge intervals ignoring strand (we'll scan both C and G)
        intervals.sort()
        merged_intervals = []
        for start, end, strand in intervals:
            if merged_intervals and start <= merged_intervals[-1][1]:
                merged_intervals[-1] = (merged_intervals[-1][0],
                                        max(end, merged_intervals[-1][1]))
            else:
                merged_intervals.append((start, end))
        merged[chrom] = merged_intervals

    total = sum(len(v) for v in merged.values())
    total_bp = sum(e - s for v in merged.values() for s, e in v)
    logger.info("Parsed %d merged exonic intervals across %d chroms (%.1f Mbp)",
                total, len(merged), total_bp / 1e6)
    return merged


def build_exclusion_set(v3_splits: pd.DataFrame, radius: int = EXCLUSION_RADIUS) -> dict:
    """Build per-chrom set of excluded positions (within radius of known sites).

    Returns dict of chrom -> set of 0-based positions to exclude.
    Handles both hg19 and hg38 coordinates (conservatively excludes both).
    """
    exclusion = defaultdict(set)
    for _, row in v3_splits.iterrows():
        chrom = str(row["chr"])
        pos = int(row["start"])  # 1-based
        pos_0 = pos - 1
        for p in range(pos_0 - radius, pos_0 + radius + 1):
            exclusion[chrom].add(p)
    logger.info("Built exclusion set: %d positions across %d chroms",
                sum(len(v) for v in exclusion.values()), len(exclusion))
    return exclusion


def find_all_exonic_c_positions(genome, exons_by_chrom, exclusion, tc_only=False):
    """Scan all exonic regions and collect C positions on + strand, G on - strand.

    Uses numpy structured arrays for memory efficiency (~26M candidates).

    Args:
        genome: pyfaidx.Fasta object.
        exons_by_chrom: dict of chrom -> list of (start, end) intervals.
        exclusion: dict of chrom -> set of 0-based positions to exclude.
        tc_only: If True, only collect TC-context positions.

    Returns:
        List of (chrom, pos_1based, strand, dinuc_context) tuples.
    """
    logger.info("Scanning exonic regions for C/G positions%s...",
                " (TC-only)" if tc_only else "")
    candidates = []
    comp = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}

    chroms_to_scan = sorted(exons_by_chrom.keys(),
                            key=lambda c: (len(c), c))

    for ci, chrom in enumerate(chroms_to_scan):
        if chrom not in genome:
            continue
        chrom_len = len(genome[chrom])
        chrom_excl = exclusion.get(chrom, set())
        intervals = exons_by_chrom[chrom]

        for start, end in intervals:
            # Ensure enough flanking for 201-nt extraction
            scan_start = max(start, FLANK_SIZE)
            scan_end = min(end, chrom_len - FLANK_SIZE - 1)
            if scan_start >= scan_end:
                continue

            seq = str(genome[chrom][scan_start:scan_end]).upper()

            for i, base in enumerate(seq):
                genome_pos_0 = scan_start + i

                if genome_pos_0 in chrom_excl:
                    continue

                genome_pos_1 = genome_pos_0 + 1

                if base == "C":
                    up_base = seq[i - 1].upper() if i > 0 else str(genome[chrom][genome_pos_0 - 1]).upper()
                    dinuc = up_base + "C"
                    if tc_only and dinuc != "TC":
                        continue
                    candidates.append((chrom, genome_pos_1, "+", dinuc))
                elif base == "G":
                    down_base = seq[i + 1].upper() if i + 1 < len(seq) else str(genome[chrom][genome_pos_0 + 1]).upper()
                    dinuc = comp.get(down_base, "N") + "C"
                    if tc_only and dinuc != "TC":
                        continue
                    candidates.append((chrom, genome_pos_1, "-", dinuc))

        if (ci + 1) % 5 == 0 or ci == len(chroms_to_scan) - 1:
            logger.info("  Scanned %d/%d chroms, %d candidates so far",
                        ci + 1, len(chroms_to_scan), len(candidates))

    logger.info("Total candidate C positions: %d", len(candidates))
    return candidates


def extract_sequence_batch(args):
    """Extract sequences for a batch of candidates. Used by multiprocessing."""
    genome_path, batch = args
    from pyfaidx import Fasta
    genome = Fasta(str(genome_path))
    results = []
    for chrom, pos, strand, dinuc in batch:
        seq = extract_from_genome(genome, chrom, pos, strand)
        if seq is not None:
            results.append((chrom, pos, strand, dinuc, seq))
    return results


def fold_and_check_unpaired(seq_201nt: str) -> bool:
    """Fold a 201-nt sequence with ViennaRNA and check if center is unpaired."""
    try:
        import RNA
        # Convert U back to T for ViennaRNA (it handles both, but be safe)
        dna_like = seq_201nt.replace("U", "T")
        structure, mfe = RNA.fold(dna_like)
        return structure[CENTER] == "."
    except Exception:
        return False


def fold_batch(seqs_with_ids):
    """Fold a batch of sequences and return those with unpaired center."""
    import RNA
    results = []
    for idx, seq in seqs_with_ids:
        dna_like = seq.replace("U", "T")
        structure, mfe = RNA.fold(dna_like)
        if structure[CENTER] == ".":
            results.append(idx)
    return results


def make_neg_dataframe(selected_candidates, dataset_source, output_seqs_dict, genome):
    """Create DataFrame and extract sequences for selected negative candidates."""
    rows = []
    for chrom, pos, strand, dinuc in selected_candidates:
        site_id = f"v4_{dataset_source}_{chrom}:{pos}:{strand}"
        seq = extract_from_genome(genome, chrom, pos, strand)
        if seq is None:
            continue
        output_seqs_dict[site_id] = seq
        rows.append({
            "site_id": site_id,
            "chr": chrom,
            "start": pos,
            "strand": strand,
            "enzyme": "all",  # These are general negatives
            "dataset_source": dataset_source,
            "coordinate_system": "hg38",
            "editing_rate": 0.0,
            "is_edited": 0,
            "flanking_seq": "",
            "seq_center": "",
            "end": pos,
            "source_type": "negative_control",
        })
    return pd.DataFrame(rows)


def save_dataset(positives_df, positives_seqs, neg_df, neg_seqs, splits_path, seqs_path):
    """Combine positives + negatives and save."""
    combined = pd.concat([positives_df, neg_df], ignore_index=True)
    combined.to_csv(splits_path, index=False)
    logger.info("Saved splits: %s (%d rows: %d pos + %d neg)",
                splits_path.name, len(combined),
                len(positives_df), len(neg_df))

    all_seqs = {}
    all_seqs.update(positives_seqs)
    all_seqs.update(neg_seqs)
    with open(seqs_path, "w") as f:
        json.dump(all_seqs, f)
    logger.info("Saved sequences: %s (%d seqs)", seqs_path.name, len(all_seqs))


def step1_random(positives_df, positives_seqs, all_candidates, genome):
    """Generate v4-random: 75,640 random exonic C positions (no motif matching)."""
    logger.info("=" * 60)
    logger.info("STEP 1: v4-random (1:10, random exonic C positions)")
    logger.info("=" * 60)

    np_rng = np.random.RandomState(SEED)
    indices = np_rng.choice(len(all_candidates), size=min(N_RANDOM, len(all_candidates)), replace=False)
    selected = [all_candidates[i] for i in indices]
    if len(selected) < N_RANDOM:
        logger.warning("Only %d candidates available (need %d)", len(selected), N_RANDOM)

    logger.info("Selected %d random negatives", len(selected))

    # Count motif distribution
    tc_count = sum(1 for _, _, _, d in selected if d == "TC")
    cc_count = sum(1 for _, _, _, d in selected if d == "CC")
    logger.info("Motif distribution: TC=%.1f%%, CC=%.1f%%, other=%.1f%%",
                100 * tc_count / len(selected),
                100 * cc_count / len(selected),
                100 * (len(selected) - tc_count - cc_count) / len(selected))

    neg_seqs = {}
    neg_df = make_neg_dataframe(selected, "v4_random", neg_seqs, genome)
    logger.info("Extracted %d sequences", len(neg_df))

    save_dataset(
        positives_df, positives_seqs, neg_df, neg_seqs,
        OUT_DIR / "splits_v4_random_negatives.csv",
        OUT_DIR / "multi_enzyme_sequences_v4_random.json",
    )
    return selected, neg_df, neg_seqs


def step2_hard(positives_df, positives_seqs, tc_candidates, genome, n_workers=14):
    """Generate v4-hard: 37,820 TC-context + unpaired loop negatives."""
    logger.info("=" * 60)
    logger.info("STEP 2: v4-hard (1:5, TC + unpaired loop negatives)")
    logger.info("=" * 60)

    logger.info("TC-context candidates: %d", len(tc_candidates))

    # Sample a pool to fold. Expect ~40-60% unpaired, so take ~2.5x what we need.
    np_rng = np.random.RandomState(SEED + 1)
    pool_size = min(len(tc_candidates), N_HARD * 3)
    indices = np_rng.choice(len(tc_candidates), size=pool_size, replace=False)
    tc_pool = [tc_candidates[i] for i in indices]
    logger.info("TC pool for folding: %d candidates", len(tc_pool))

    # Extract sequences first
    logger.info("Extracting sequences for TC pool...")
    tc_seqs = {}
    valid_tc = []
    for chrom, pos, strand, dinuc in tc_pool:
        seq = extract_from_genome(genome, chrom, pos, strand)
        if seq is not None:
            idx = len(valid_tc)
            valid_tc.append((chrom, pos, strand, dinuc))
            tc_seqs[idx] = seq

    logger.info("Valid TC sequences: %d", len(valid_tc))

    # Fold with ViennaRNA to find unpaired positions
    logger.info("Folding %d sequences with ViennaRNA (%d workers)...", len(tc_seqs), n_workers)

    # Split into batches for multiprocessing
    items = list(tc_seqs.items())
    batch_size = max(1, len(items) // n_workers)
    batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

    unpaired_indices = []
    with Pool(n_workers) as pool:
        results = pool.map(fold_batch, batches)
        for batch_result in results:
            unpaired_indices.extend(batch_result)

    logger.info("Unpaired TC positions: %d / %d (%.1f%%)",
                len(unpaired_indices), len(valid_tc),
                100 * len(unpaired_indices) / len(valid_tc) if valid_tc else 0)

    # Select hard negatives
    rng2 = random.Random(SEED + 2)
    rng2.shuffle(unpaired_indices)
    selected_indices = unpaired_indices[:N_HARD]

    if len(selected_indices) < N_HARD:
        logger.warning("Only %d unpaired TC candidates (need %d). Using all available.",
                        len(selected_indices), N_HARD)

    selected = [valid_tc[i] for i in selected_indices]
    logger.info("Selected %d hard negatives", len(selected))

    neg_seqs = {}
    neg_df = make_neg_dataframe(selected, "v4_hard", neg_seqs, genome)
    logger.info("Extracted %d sequences", len(neg_df))

    save_dataset(
        positives_df, positives_seqs, neg_df, neg_seqs,
        OUT_DIR / "splits_v4_hard_negatives.csv",
        OUT_DIR / "multi_enzyme_sequences_v4_hard.json",
    )
    return selected, neg_df, neg_seqs


def step3_large(positives_df, positives_seqs, all_candidates,
                random_neg_df, random_neg_seqs,
                hard_neg_df, hard_neg_seqs,
                v3_neg_df, v3_neg_seqs,
                genome):
    """Generate v4-large: 378,200 mixed negatives (1:50 ratio)."""
    logger.info("=" * 60)
    logger.info("STEP 3: v4-large (1:50, mixed negatives)")
    logger.info("=" * 60)

    # Component 1: v3 motif-matched negatives (7,778)
    n_v3 = len(v3_neg_df)
    logger.info("Component 1 (v3 motif-matched): %d", n_v3)

    # Component 2: v4-random negatives (75,640)
    n_random = len(random_neg_df)
    logger.info("Component 2 (v4-random): %d", n_random)

    # Component 3: v4-hard negatives (all available)
    n_hard = len(hard_neg_df)
    logger.info("Component 3 (v4-hard): %d", n_hard)

    # Component 4: fill remaining with additional random exonic C positions
    current_total = n_v3 + n_random + n_hard
    n_fill = N_LARGE - current_total
    logger.info("Need %d additional random negatives to reach %d", n_fill, N_LARGE)

    # Collect all already-used positions
    used_positions = set()
    for _, row in v3_neg_df.iterrows():
        used_positions.add((str(row["chr"]), int(row["start"])))
    for _, row in random_neg_df.iterrows():
        used_positions.add((str(row["chr"]), int(row["start"])))
    for _, row in hard_neg_df.iterrows():
        used_positions.add((str(row["chr"]), int(row["start"])))
    logger.info("Already used %d negative positions", len(used_positions))

    # Filter candidates to unused positions
    remaining_candidates = [
        (c, p, s, d) for c, p, s, d in all_candidates
        if (c, p) not in used_positions
    ]
    logger.info("Remaining candidate pool: %d", len(remaining_candidates))

    np_rng = np.random.RandomState(SEED + 3)
    if n_fill > 0 and len(remaining_candidates) > 0:
        indices = np_rng.choice(len(remaining_candidates),
                                size=min(n_fill, len(remaining_candidates)),
                                replace=False)
        fill_selected = [remaining_candidates[i] for i in indices]
    else:
        fill_selected = []

    if len(fill_selected) < n_fill:
        logger.warning("Only %d fill candidates available (need %d). Using all.",
                        len(fill_selected), n_fill)

    fill_seqs = {}
    fill_df = make_neg_dataframe(fill_selected, "v4_fill", fill_seqs, genome)
    logger.info("Fill negatives: %d", len(fill_df))

    # Combine all components
    combined_neg_df = pd.concat([v3_neg_df, random_neg_df, hard_neg_df, fill_df],
                                ignore_index=True)
    combined_neg_seqs = {}
    combined_neg_seqs.update(v3_neg_seqs)
    combined_neg_seqs.update(random_neg_seqs)
    combined_neg_seqs.update(hard_neg_seqs)
    combined_neg_seqs.update(fill_seqs)

    logger.info("Total v4-large negatives: %d (v3=%d + random=%d + hard=%d + fill=%d)",
                len(combined_neg_df), n_v3, n_random, n_hard, len(fill_df))

    save_dataset(
        positives_df, positives_seqs, combined_neg_df, combined_neg_seqs,
        OUT_DIR / "splits_v4_large_negatives.csv",
        OUT_DIR / "multi_enzyme_sequences_v4_large.json",
    )


def main():
    parser = argparse.ArgumentParser(description="Generate v4 negative datasets")
    parser.add_argument("--step", choices=["1", "2", "3", "all"], default="all",
                        help="Which step to run (1=random, 2=hard, 3=large, all=all)")
    parser.add_argument("--workers", type=int, default=14,
                        help="Number of workers for ViennaRNA folding")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
        force=True,
    )
    # Ensure unbuffered output
    sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
    sys.stderr.reconfigure(line_buffering=True) if hasattr(sys.stderr, 'reconfigure') else None

    # Load v3 data
    logger.info("Loading v3 dataset...")
    v3_df = pd.read_csv(V3_SPLITS)
    with open(V3_SEQS) as f:
        v3_seqs = json.load(f)

    positives_df = v3_df[v3_df["is_edited"] == 1].copy()
    v3_neg_df = v3_df[v3_df["is_edited"] == 0].copy()
    logger.info("v3: %d positives, %d negatives", len(positives_df), len(v3_neg_df))

    # Split sequences
    pos_site_ids = set(positives_df["site_id"])
    neg_site_ids = set(v3_neg_df["site_id"])
    positives_seqs = {k: v for k, v in v3_seqs.items() if k in pos_site_ids}
    v3_neg_seqs = {k: v for k, v in v3_seqs.items() if k in neg_site_ids}
    logger.info("Positive sequences: %d, v3 negative sequences: %d",
                len(positives_seqs), len(v3_neg_seqs))

    # Load genome
    logger.info("Loading hg38 genome...")
    from pyfaidx import Fasta
    genome = Fasta(str(GENOME_HG38))

    # Build exclusion set from ALL known sites (positives + negatives)
    exclusion = build_exclusion_set(v3_df)

    # Parse exonic regions
    exons = parse_refgene_exons(REFGENE_HG38)

    # Run requested steps
    random_neg_df = random_neg_seqs = None
    hard_neg_df = hard_neg_seqs = None
    random_selected = hard_selected = None
    all_candidates = None  # Lazy-loaded

    if args.step in ("1", "all", "3"):
        # Steps 1 and 3 need all candidates
        all_candidates = find_all_exonic_c_positions(genome, exons, exclusion)

    if args.step in ("1", "all"):
        random_selected, random_neg_df, random_neg_seqs = step1_random(
            positives_df, positives_seqs, all_candidates, genome
        )

    if args.step in ("2", "all"):
        # Step 2 only needs TC candidates — scan separately if needed
        if all_candidates is not None:
            tc_candidates = [(c, p, s, d) for c, p, s, d in all_candidates if d == "TC"]
        else:
            tc_candidates = find_all_exonic_c_positions(genome, exons, exclusion, tc_only=True)
        hard_selected, hard_neg_df, hard_neg_seqs = step2_hard(
            positives_df, positives_seqs, tc_candidates, genome,
            n_workers=args.workers
        )

    if args.step in ("3", "all"):
        # For step 3, we need results from steps 1 and 2
        if random_neg_df is None:
            # Load from disk
            random_splits = OUT_DIR / "splits_v4_random_negatives.csv"
            random_seqs_path = OUT_DIR / "multi_enzyme_sequences_v4_random.json"
            if random_splits.exists():
                tmp = pd.read_csv(random_splits)
                random_neg_df = tmp[tmp["is_edited"] == 0].copy()
                with open(random_seqs_path) as f:
                    all_random_seqs = json.load(f)
                random_neg_seqs = {k: v for k, v in all_random_seqs.items()
                                   if k in set(random_neg_df["site_id"])}
                logger.info("Loaded v4-random from disk: %d negatives", len(random_neg_df))
            else:
                raise RuntimeError("v4-random not found. Run step 1 first.")

        if hard_neg_df is None:
            hard_splits = OUT_DIR / "splits_v4_hard_negatives.csv"
            hard_seqs_path = OUT_DIR / "multi_enzyme_sequences_v4_hard.json"
            if hard_splits.exists():
                tmp = pd.read_csv(hard_splits)
                hard_neg_df = tmp[tmp["is_edited"] == 0].copy()
                with open(hard_seqs_path) as f:
                    all_hard_seqs = json.load(f)
                hard_neg_seqs = {k: v for k, v in all_hard_seqs.items()
                                 if k in set(hard_neg_df["site_id"])}
                logger.info("Loaded v4-hard from disk: %d negatives", len(hard_neg_df))
            else:
                raise RuntimeError("v4-hard not found. Run step 2 first.")

        step3_large(
            positives_df, positives_seqs, all_candidates,
            random_neg_df, random_neg_seqs,
            hard_neg_df, hard_neg_seqs,
            v3_neg_df, v3_neg_seqs,
            genome,
        )

    logger.info("Done!")


if __name__ == "__main__":
    main()
