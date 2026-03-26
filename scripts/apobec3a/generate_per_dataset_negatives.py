"""Generate per-dataset negative control sets for cross-dataset generalization.

For each dataset in all_datasets_combined.csv, generates matched negative
controls by finding TC-motif C sites in the same genes that are NOT known
editing sites across any dataset.

This is needed for the cross-dataset generalization study (Iteration 4),
where we train on one dataset and evaluate on another. Having dataset-specific
negatives ensures balanced training sets for each dataset.

Strategy:
  1. Load all positive sites from all_datasets_combined.csv
  2. For each dataset's positive sites:
     a) Identify genes containing positive sites
     b) Find TC-motif C positions in those genes (from hg19 + refGene.txt)
     c) Exclude all known positive sites (from ALL datasets, not just current)
     d) Sample negatives at configurable ratio per dataset
  3. Output unified CSV with dataset_source column

Output:
  - data/processed/negatives_per_dataset.csv

Usage:
    python scripts/apobec3a/generate_per_dataset_negatives.py
    python scripts/apobec3a/generate_per_dataset_negatives.py --neg-ratio 10
    python scripts/apobec3a/generate_per_dataset_negatives.py --genome /path/to/hg19.fa
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GENOME_PATH = PROJECT_ROOT / "data" / "raw" / "genomes" / "hg38.fa"
REFGENE_PATH = PROJECT_ROOT / "data" / "raw" / "genomes" / "refGene.txt"
COMBINED_PATH = PROJECT_ROOT / "data" / "processed" / "all_datasets_combined.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


def load_refgene(path: Path) -> pd.DataFrame:
    """Load UCSC RefSeq gene annotations."""
    cols = [
        "bin", "name", "chrom", "strand", "txStart", "txEnd",
        "cdsStart", "cdsEnd", "exonCount", "exonStarts", "exonEnds",
        "score", "name2", "cdsStartStat", "cdsEndStat", "exonFrames",
    ]
    df = pd.read_csv(path, sep="\t", header=None, names=cols)
    # Filter to standard chromosomes
    std_chroms = {f"chr{i}" for i in range(1, 23)} | {"chrX", "chrY"}
    df = df[df["chrom"].isin(std_chroms)]
    return df


def get_exon_intervals(row: pd.Series) -> list:
    """Parse exon start/end strings into list of (start, end) tuples."""
    starts = [int(s) for s in row["exonStarts"].rstrip(",").split(",") if s]
    ends = [int(e) for e in row["exonEnds"].rstrip(",").split(",") if e]
    return list(zip(starts, ends))


def extract_transcript_exonic_sequence(genome, chrom: str, exons: list,
                                       strand: str) -> tuple:
    """Extract concatenated exonic sequence and build coordinate map.

    Returns:
        (exonic_sequence, coord_map) where coord_map[i] = genomic position
        of the i-th position in the exonic sequence.
    """
    if chrom not in genome:
        return "", []

    chrom_len = len(genome[chrom])
    seq_parts = []
    coord_map = []

    for start, end in sorted(exons):
        s = max(0, start)
        e = min(chrom_len, end)
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

    return exonic_seq, coord_map


def find_tc_motif_positions(exonic_seq: str) -> list:
    """Find C positions preceded by T (TC dinucleotide context).

    APOBEC3A preferentially edits Cs in TC context.
    Returns list of 0-based positions in the exonic sequence.
    """
    positions = []
    for i in range(1, len(exonic_seq)):
        if exonic_seq[i] == "C" and exonic_seq[i - 1] == "T":
            positions.append(i)
    return positions


def find_c_positions(exonic_seq: str) -> list:
    """Find ALL cytidine positions in the exonic sequence.

    Used for RNAsee-style all-cytidine negatives (not restricted to TC motif).
    Returns list of 0-based positions in the exonic sequence.
    """
    return [i for i, b in enumerate(exonic_seq) if b == "C"]


def get_representative_transcripts(refgene: pd.DataFrame,
                                   gene_set: set) -> pd.DataFrame:
    """Select one representative transcript per gene (longest CDS).

    Prefers NM_ transcripts (mRNA) over others.
    """
    gene_tx = refgene[refgene["name2"].isin(gene_set)]

    # Prefer NM_ transcripts
    nm_tx = gene_tx[gene_tx["name"].str.startswith("NM_")]
    genes_with_nm = set(nm_tx["name2"])
    genes_without_nm = gene_set - genes_with_nm

    if genes_without_nm:
        other_tx = gene_tx[gene_tx["name2"].isin(genes_without_nm)]
        transcripts = pd.concat([nm_tx, other_tx])
    else:
        transcripts = nm_tx

    # Select longest CDS per gene
    transcripts = transcripts.copy()
    transcripts["cds_length"] = transcripts["cdsEnd"] - transcripts["cdsStart"]
    transcripts = transcripts.sort_values("cds_length", ascending=False)
    representative = transcripts.drop_duplicates(subset="name2", keep="first")

    return representative


def generate_negatives_for_dataset(
    dataset_name: str,
    dataset_genes: set,
    positive_coords_all: set,
    representative_tx: pd.DataFrame,
    genome,
    neg_ratio: int,
    n_positives: int,
    rng: np.random.RandomState,
    motif_mode: str = "tc_motif",
) -> pd.DataFrame:
    """Generate negative controls for a single dataset.

    Args:
        dataset_name: Name of the dataset (e.g., "sharma_2015").
        dataset_genes: Set of gene names with positive sites in this dataset.
        positive_coords_all: Set of (chr, start) for ALL known positive sites.
        representative_tx: DataFrame of representative transcripts for dataset genes.
        genome: pyfaidx Fasta object for hg19.
        neg_ratio: Target negatives-per-positive ratio.
        n_positives: Number of positive sites in this dataset.
        rng: Random state for reproducibility.
        motif_mode: "tc_motif" for TC-context only, "all_c" for all cytidines.

    Returns:
        DataFrame of negative control sites.
    """
    candidate_rows = []

    for _, tx in representative_tx.iterrows():
        gene = tx["name2"]
        if gene not in dataset_genes:
            continue

        chrom = tx["chrom"]
        strand = tx["strand"]
        exons = get_exon_intervals(tx)

        exonic_seq, coord_map = extract_transcript_exonic_sequence(
            genome, chrom, exons, strand
        )

        if not exonic_seq:
            continue

        if motif_mode == "all_c":
            c_positions = find_c_positions(exonic_seq)
        else:
            c_positions = find_tc_motif_positions(exonic_seq)

        for pos_in_seq in c_positions:
            genomic_pos = coord_map[pos_in_seq]
            # Exclude all known positive sites across all datasets
            if (chrom, genomic_pos) not in positive_coords_all:
                candidate_rows.append({
                    "chr": chrom,
                    "start": genomic_pos,
                    "end": genomic_pos + 1,
                    "strand": strand,
                    "gene": gene,
                    "transcript": tx["name"],
                    "pos_in_transcript": pos_in_seq,
                })

    if not candidate_rows:
        logger.warning("  No candidate negatives found for %s", dataset_name)
        return pd.DataFrame()

    candidates_df = pd.DataFrame(candidate_rows)

    # Deduplicate by coordinate
    candidates_df = candidates_df.drop_duplicates(subset=["chr", "start"], keep="first")

    # Sample to target ratio
    target_n = min(n_positives * neg_ratio, len(candidates_df))
    if target_n < len(candidates_df):
        candidates_df = candidates_df.sample(n=target_n, random_state=rng)
    else:
        logger.warning(
            "  %s: Only %d candidates available for %d target negatives "
            "(ratio %.1f instead of %d)",
            dataset_name, len(candidates_df), n_positives * neg_ratio,
            len(candidates_df) / max(1, n_positives), neg_ratio
        )

    # Build output
    result = pd.DataFrame()
    result["site_id"] = candidates_df.apply(
        lambda r: f"neg_{dataset_name}_{r['chr']}_{r['start']}", axis=1
    )
    result["chr"] = candidates_df["chr"].values
    result["start"] = candidates_df["start"].values
    result["end"] = candidates_df["end"].values
    result["strand"] = candidates_df["strand"].values
    result["gene"] = candidates_df["gene"].values
    result["is_edited"] = 0
    result["dataset_source"] = dataset_name
    result["tier"] = "all_cytidine" if motif_mode == "all_c" else "tc_motif"
    result["feature"] = "exonic"

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Generate per-dataset negative control sets"
    )
    parser.add_argument(
        "--neg-ratio", type=int, default=5,
        help="Target negatives-per-positive ratio (default: 5)"
    )
    parser.add_argument(
        "--genome", type=str, default=None,
        help="Path to hg38.fa (default: data/raw/genomes/hg38.fa)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--all-cytidines", action="store_true",
        help="Use ALL cytidine positions as negatives (not just TC-motif). "
             "Produces easier negatives matching RNAsee's methodology."
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    rng = np.random.RandomState(args.seed)

    # Validate inputs
    genome_path = Path(args.genome) if args.genome else DEFAULT_GENOME_PATH
    if not genome_path.exists():
        logger.error("Genome file not found: %s", genome_path)
        return

    if not REFGENE_PATH.exists():
        logger.error("RefSeq annotations not found: %s", REFGENE_PATH)
        return

    if not COMBINED_PATH.exists():
        logger.error(
            "Combined dataset not found: %s\n"
            "Run build_unified_dataset.py first.", COMBINED_PATH
        )
        return

    # Load genome
    from pyfaidx import Fasta
    logger.info("Loading hg19 genome...")
    genome = Fasta(str(genome_path))
    logger.info("Loaded %d chromosomes", len(genome.keys()))

    # Load RefSeq annotations
    logger.info("Loading RefSeq annotations...")
    refgene = load_refgene(REFGENE_PATH)
    logger.info("Loaded %d transcripts for %d genes",
                len(refgene), refgene["name2"].nunique())

    # Load all positive sites
    combined = pd.read_csv(COMBINED_PATH)
    logger.info("Loaded %d positive sites from %d datasets",
                len(combined), combined["dataset_source"].nunique())

    # Build global positive coordinate set (exclude from ALL datasets)
    positive_coords_all = set(zip(combined["chr"], combined["start"]))
    logger.info("Global positive coordinate set: %d unique positions",
                len(positive_coords_all))

    # Get all genes across all datasets
    all_genes = set(combined["gene"].dropna().str.strip())
    all_genes.discard("")
    logger.info("Total unique genes across all datasets: %d", len(all_genes))

    # Get representative transcripts for all genes at once
    logger.info("Selecting representative transcripts...")
    representative = get_representative_transcripts(refgene, all_genes)
    logger.info("Selected %d representative transcripts", len(representative))

    # Determine motif mode
    motif_mode = "all_c" if args.all_cytidines else "tc_motif"
    logger.info("Motif mode: %s", motif_mode)

    # Generate negatives per dataset
    all_negatives = []

    for dataset_name, group in combined.groupby("dataset_source"):
        n_pos = len(group)
        dataset_genes = set(group["gene"].dropna().str.strip())
        dataset_genes.discard("")

        logger.info("\n=== Processing %s (%d positives, %d genes) ===",
                    dataset_name, n_pos, len(dataset_genes))

        negatives = generate_negatives_for_dataset(
            dataset_name=dataset_name,
            dataset_genes=dataset_genes,
            positive_coords_all=positive_coords_all,
            representative_tx=representative,
            genome=genome,
            neg_ratio=args.neg_ratio,
            n_positives=n_pos,
            rng=rng,
            motif_mode=motif_mode,
        )

        if len(negatives) > 0:
            all_negatives.append(negatives)
            logger.info("  Generated %d negatives (ratio 1:%.1f)",
                        len(negatives), len(negatives) / max(1, n_pos))

    if not all_negatives:
        logger.error("No negatives generated for any dataset!")
        return

    # Combine all per-dataset negatives
    result = pd.concat(all_negatives, ignore_index=True)

    # Check for duplicates across datasets (same negative site in multiple datasets)
    coord_counts = result.groupby(["chr", "start"]).size()
    n_shared = (coord_counts > 1).sum()
    if n_shared > 0:
        logger.info("\n%d negative sites appear in multiple dataset sets "
                    "(expected for overlapping genes)", n_shared)

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_name = "negatives_per_dataset_all_c.csv" if args.all_cytidines else "negatives_per_dataset.csv"
    out_path = OUTPUT_DIR / out_name
    result.to_csv(out_path, index=False)
    logger.info("\nSaved %d per-dataset negatives to %s", len(result), out_path)

    # Summary
    logger.info("\n=== Per-Dataset Negatives Summary ===")
    logger.info("Negative ratio target: 1:%d", args.neg_ratio)
    logger.info("%-20s  %8s  %8s  %8s", "Dataset", "Positives", "Negatives", "Ratio")
    logger.info("-" * 52)
    for dataset_name, group in result.groupby("dataset_source"):
        n_neg = len(group)
        n_pos = len(combined[combined["dataset_source"] == dataset_name])
        ratio = n_neg / max(1, n_pos)
        logger.info("%-20s  %8d  %8d  %8.1f", dataset_name, n_pos, n_neg, ratio)
    logger.info("-" * 52)
    logger.info("%-20s  %8d  %8d  %8.1f",
                "TOTAL", len(combined), len(result),
                len(result) / max(1, len(combined)))


if __name__ == "__main__":
    main()
