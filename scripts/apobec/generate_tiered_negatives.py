"""Generate 3-tier negative control sets from hg19 transcript sequences.

Tier 1: All exonic C positions in transcripts harboring positive editing sites.
         These are Cs that COULD be edited but are NOT. (~1:200 ratio)

Tier 2: TC-motif C positions only (APOBEC3A preferred context).
         Subset of Tier 1 filtered to TC dinucleotide context. (~1:50 ratio)

Tier 3: TC-motif Cs in predicted stem-loop structures (RNAfold).
         Subset of Tier 2 further filtered by structural context. (~1:5-20 ratio)

Requirements:
  - hg19.fa indexed with pyfaidx
  - RefSeq gene annotations (refGene.txt from UCSC)
  - ViennaRNA RNAfold for Tier 3

Output:
  - data/processed/negatives_tier1.csv
  - data/processed/negatives_tier2.csv
  - data/processed/negatives_tier3.csv

Usage:
    python scripts/apobec/generate_tiered_negatives.py
    python scripts/apobec/generate_tiered_negatives.py --tier3-batch-size 100
"""

import argparse
import logging
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GENOME_PATH = PROJECT_ROOT / "data" / "raw" / "genomes" / "hg19.fa"
REFGENE_PATH = PROJECT_ROOT / "data" / "raw" / "genomes" / "refGene.txt"
LABELS_PATH = PROJECT_ROOT / "data" / "processed" / "editing_sites_labels.csv"
UNIFIED_PATH = PROJECT_ROOT / "data" / "processed" / "advisor" / "unified_editing_sites.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
RNAFOLD = Path("/opt/miniconda3/envs/vienna/bin/RNAfold")

FLANK_SIZE = 100  # nt on each side


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
        # Reverse complement for minus strand
        comp = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
        exonic_seq = "".join(comp.get(b, "N") for b in reversed(exonic_seq))
        coord_map = list(reversed(coord_map))

    return exonic_seq, coord_map


def find_c_positions(exonic_seq: str) -> list:
    """Find all C positions in the exonic sequence (0-indexed)."""
    return [i for i, b in enumerate(exonic_seq) if b == "C"]


def find_tc_motif_positions(exonic_seq: str) -> list:
    """Find C positions preceded by T (TC dinucleotide context).

    APOBEC3A preferentially edits Cs in TC context.
    """
    positions = []
    for i in range(1, len(exonic_seq)):
        if exonic_seq[i] == "C" and exonic_seq[i - 1] == "T":
            positions.append(i)
    return positions


def predict_structures_for_windows(genome, sites_df: pd.DataFrame,
                                   flank: int = FLANK_SIZE,
                                   batch_size: int = 50) -> list:
    """Predict RNA structures for windows around each site.

    Returns list of (dot_bracket, mfe, is_in_loop) tuples.
    """
    if not RNAFOLD.exists():
        logger.error("RNAfold not found at %s", RNAFOLD)
        return [("", np.nan, False)] * len(sites_df)

    results = []
    total = len(sites_df)

    for batch_start in range(0, total, batch_size):
        batch = sites_df.iloc[batch_start:batch_start + batch_size]
        sequences = []
        valid_indices = []

        for i, (_, row) in enumerate(batch.iterrows()):
            chrom = row["chr"]
            pos = row["genomic_pos"]

            if chrom not in genome:
                sequences.append(None)
                continue

            chrom_len = len(genome[chrom])
            g_start = max(0, pos - flank)
            g_end = min(chrom_len, pos + flank + 1)

            if g_end - g_start < 2 * flank + 1:
                sequences.append(None)
                continue

            dna_seq = str(genome[chrom][g_start:g_end]).upper()
            strand = row.get("strand", "+")

            if strand == "-":
                comp = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
                dna_seq = "".join(comp.get(b, "N") for b in reversed(dna_seq))

            rna_seq = dna_seq.replace("T", "U")
            sequences.append(rna_seq)
            valid_indices.append(i)

        # Run RNAfold on valid sequences
        valid_seqs = [sequences[i] for i in valid_indices if sequences[i]]
        valid_idx_filtered = [i for i in valid_indices if sequences[i]]

        batch_results = [("", np.nan, False)] * len(batch)

        if valid_seqs:
            try:
                input_text = "\n".join(valid_seqs)
                result = subprocess.run(
                    [str(RNAFOLD), "--noPS", "-T37"],
                    input=input_text,
                    capture_output=True,
                    text=True,
                    timeout=600,
                )

                if result.returncode == 0:
                    lines = result.stdout.strip().split("\n")
                    struct_idx = 0
                    line_idx = 0
                    while line_idx < len(lines) - 1 and struct_idx < len(valid_idx_filtered):
                        struct_line = lines[line_idx + 1].strip()
                        dot_bracket = ""
                        mfe = np.nan

                        for j in range(len(struct_line) - 1, -1, -1):
                            if struct_line[j] == "(" and j > 0:
                                remainder = struct_line[j:]
                                if ")" in remainder:
                                    dot_bracket = struct_line[:j].strip()
                                    try:
                                        mfe = float(remainder.strip("() "))
                                    except ValueError:
                                        pass
                                    break

                        # Check if center position is in a loop
                        center = flank
                        is_in_loop = False
                        if dot_bracket and center < len(dot_bracket):
                            is_in_loop = dot_bracket[center] == "."

                        orig_idx = valid_idx_filtered[struct_idx]
                        batch_results[orig_idx] = (dot_bracket, mfe, is_in_loop)
                        struct_idx += 1
                        line_idx += 2
            except Exception as e:
                logger.warning("RNAfold batch error: %s", e)

        results.extend(batch_results)

        processed = min(batch_start + len(batch), total)
        if processed % 500 == 0 or processed >= total:
            logger.info("Structure prediction: %d/%d", processed, total)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate 3-tier negative control sets")
    parser.add_argument("--tier3-batch-size", type=int, default=100,
                        help="Batch size for RNAfold in Tier 3")
    parser.add_argument("--max-tier1-per-gene", type=int, default=500,
                        help="Max Tier 1 negatives per gene (to avoid huge imbalance)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    rng = np.random.RandomState(args.seed)

    # Load genome
    from pyfaidx import Fasta
    logger.info("Loading hg19 genome...")
    genome = Fasta(str(GENOME_PATH))
    logger.info("Loaded %d chromosomes", len(genome.keys()))

    # Load RefSeq annotations
    logger.info("Loading RefSeq annotations...")
    refgene = load_refgene(REFGENE_PATH)
    logger.info("Loaded %d transcripts for %d genes",
                len(refgene), refgene["name2"].nunique())

    # Load positive editing sites
    labels = pd.read_csv(LABELS_PATH)
    logger.info("Loaded %d positive editing sites", len(labels))

    # Get genes harboring positive sites
    unified = pd.read_csv(UNIFIED_PATH)
    positive_genes = set(unified["gene_refseq"].dropna().str.strip())
    logger.info("Positive sites in %d unique genes", len(positive_genes))

    # Build set of positive site coordinates for exclusion
    positive_coords = set(zip(labels["chr"], labels["start"]))
    logger.info("Positive coordinate set: %d sites", len(positive_coords))

    # Filter RefSeq to transcripts of positive-site genes
    # Use NM_ transcripts (mRNA) preferentially
    gene_transcripts = refgene[refgene["name2"].isin(positive_genes)]
    nm_transcripts = gene_transcripts[gene_transcripts["name"].str.startswith("NM_")]
    logger.info("Found %d NM_ transcripts for %d positive-site genes",
                len(nm_transcripts), nm_transcripts["name2"].nunique())

    # For genes without NM_ transcripts, use any transcript
    genes_with_nm = set(nm_transcripts["name2"])
    genes_without_nm = positive_genes - genes_with_nm
    if genes_without_nm:
        other_tx = gene_transcripts[gene_transcripts["name2"].isin(genes_without_nm)]
        logger.info("  + %d transcripts for %d genes without NM_",
                    len(other_tx), other_tx["name2"].nunique())
        transcripts = pd.concat([nm_transcripts, other_tx])
    else:
        transcripts = nm_transcripts

    # Select one representative transcript per gene (longest CDS)
    transcripts = transcripts.copy()
    transcripts["cds_length"] = transcripts["cdsEnd"] - transcripts["cdsStart"]
    transcripts = transcripts.sort_values("cds_length", ascending=False)
    representative = transcripts.drop_duplicates(subset="name2", keep="first")
    logger.info("Selected %d representative transcripts", len(representative))

    # -----------------------------------------------------------------------
    # Tier 1: All exonic C positions in positive-site gene transcripts
    # -----------------------------------------------------------------------
    logger.info("\n=== Generating Tier 1 negatives (all exonic Cs) ===")
    tier1_rows = []
    genes_processed = 0

    for _, tx in representative.iterrows():
        gene = tx["name2"]
        chrom = tx["chrom"]
        strand = tx["strand"]
        exons = get_exon_intervals(tx)

        exonic_seq, coord_map = extract_transcript_exonic_sequence(
            genome, chrom, exons, strand
        )

        if not exonic_seq:
            continue

        c_positions = find_c_positions(exonic_seq)

        # Exclude positions that match positive editing sites
        neg_positions = []
        for pos_in_seq in c_positions:
            genomic_pos = coord_map[pos_in_seq]
            if (chrom, genomic_pos) not in positive_coords:
                neg_positions.append((pos_in_seq, genomic_pos))

        # Subsample if too many
        if len(neg_positions) > args.max_tier1_per_gene:
            indices = rng.choice(len(neg_positions), args.max_tier1_per_gene, replace=False)
            neg_positions = [neg_positions[i] for i in sorted(indices)]

        for pos_in_seq, genomic_pos in neg_positions:
            tier1_rows.append({
                "site_id": f"T1_{chrom}_{genomic_pos}",
                "chr": chrom,
                "start": genomic_pos,
                "end": genomic_pos + 1,
                "strand": strand,
                "gene": gene,
                "transcript": tx["name"],
                "pos_in_transcript": pos_in_seq,
                "label": 0,
                "tier": 1,
            })

        genes_processed += 1
        if genes_processed % 100 == 0:
            logger.info("  Processed %d/%d genes (%d Tier 1 negatives so far)",
                        genes_processed, len(representative), len(tier1_rows))

    tier1_df = pd.DataFrame(tier1_rows)
    logger.info("Tier 1: %d negative C positions from %d genes",
                len(tier1_df), genes_processed)

    # -----------------------------------------------------------------------
    # Tier 2: TC-motif C positions only
    # -----------------------------------------------------------------------
    logger.info("\n=== Generating Tier 2 negatives (TC-motif Cs) ===")
    tier2_rows = []
    genes_processed = 0

    for _, tx in representative.iterrows():
        gene = tx["name2"]
        chrom = tx["chrom"]
        strand = tx["strand"]
        exons = get_exon_intervals(tx)

        exonic_seq, coord_map = extract_transcript_exonic_sequence(
            genome, chrom, exons, strand
        )

        if not exonic_seq:
            continue

        tc_positions = find_tc_motif_positions(exonic_seq)

        for pos_in_seq in tc_positions:
            genomic_pos = coord_map[pos_in_seq]
            if (chrom, genomic_pos) not in positive_coords:
                tier2_rows.append({
                    "site_id": f"T2_{chrom}_{genomic_pos}",
                    "chr": chrom,
                    "start": genomic_pos,
                    "end": genomic_pos + 1,
                    "strand": strand,
                    "gene": gene,
                    "transcript": tx["name"],
                    "pos_in_transcript": pos_in_seq,
                    "label": 0,
                    "tier": 2,
                })

        genes_processed += 1

    tier2_df = pd.DataFrame(tier2_rows)
    logger.info("Tier 2: %d TC-motif C positions from %d genes",
                len(tier2_df), genes_processed)

    # -----------------------------------------------------------------------
    # Tier 3: TC-motif Cs in predicted stem-loops (RNAfold)
    # -----------------------------------------------------------------------
    logger.info("\n=== Generating Tier 3 negatives (TC in stem-loops) ===")

    if len(tier2_df) > 0 and RNAFOLD.exists():
        # Add genomic_pos column for structure prediction
        tier2_for_struct = tier2_df.copy()
        tier2_for_struct["genomic_pos"] = tier2_for_struct["start"]

        logger.info("Running RNAfold on %d Tier 2 sites...", len(tier2_for_struct))
        structures = predict_structures_for_windows(
            genome, tier2_for_struct, batch_size=args.tier3_batch_size
        )

        # Filter to sites in stem-loop context (unpaired = in loop)
        tier3_rows = []
        for i, (dot_bracket, mfe, is_in_loop) in enumerate(structures):
            if dot_bracket and not is_in_loop:
                # Position is paired (in stem) -- check if it's in a stem-loop
                # A stem-loop has paired positions flanking an unpaired loop
                center = FLANK_SIZE
                if center < len(dot_bracket):
                    # Check local context: is there a loop nearby?
                    window = 10
                    start = max(0, center - window)
                    end = min(len(dot_bracket), center + window + 1)
                    local = dot_bracket[start:end]
                    has_paired = any(c in "()" for c in local)
                    has_unpaired = any(c == "." for c in local)
                    if has_paired and has_unpaired:
                        # In or near a stem-loop
                        row = tier2_df.iloc[i].to_dict()
                        row["site_id"] = f"T3_{row['chr']}_{row['start']}"
                        row["tier"] = 3
                        row["dot_bracket_center"] = dot_bracket[center] if center < len(dot_bracket) else ""
                        row["mfe"] = mfe
                        tier3_rows.append(row)

        tier3_df = pd.DataFrame(tier3_rows)
        logger.info("Tier 3: %d stem-loop TC-motif C positions", len(tier3_df))
    else:
        tier3_df = pd.DataFrame()
        if not RNAFOLD.exists():
            logger.warning("RNAfold not found; skipping Tier 3")
        else:
            logger.warning("No Tier 2 sites; skipping Tier 3")

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tier1_path = OUTPUT_DIR / "negatives_tier1.csv"
    tier1_df.to_csv(tier1_path, index=False)
    logger.info("Saved Tier 1 to %s", tier1_path)

    tier2_path = OUTPUT_DIR / "negatives_tier2.csv"
    tier2_df.to_csv(tier2_path, index=False)
    logger.info("Saved Tier 2 to %s", tier2_path)

    tier3_path = OUTPUT_DIR / "negatives_tier3.csv"
    tier3_df.to_csv(tier3_path, index=False)
    logger.info("Saved Tier 3 to %s", tier3_path)

    # Summary
    logger.info("\n=== Negative Set Summary ===")
    logger.info("Positive editing sites: %d", len(labels))
    logger.info("Tier 1 (all exonic Cs): %d (ratio 1:%.0f)",
                len(tier1_df), len(tier1_df) / max(1, len(labels)))
    logger.info("Tier 2 (TC-motif Cs): %d (ratio 1:%.0f)",
                len(tier2_df), len(tier2_df) / max(1, len(labels)))
    if len(tier3_df) > 0:
        logger.info("Tier 3 (TC in stem-loops): %d (ratio 1:%.1f)",
                    len(tier3_df), len(tier3_df) / max(1, len(labels)))
    else:
        logger.info("Tier 3 (TC in stem-loops): 0")


if __name__ == "__main__":
    main()
