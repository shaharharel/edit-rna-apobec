"""Extract RNA sequences from hg19 and compute secondary structures.

Processes both positive editing sites (editing_sites_labels.csv) and
negative control sites (positive_negative_combined.csv).

Supports two modes:
- Local hg19 FASTA via pyfaidx (fast, preferred)
- UCSC REST API fallback (slower, network-dependent)

Output: data/processed/sequences_and_structures.csv

Usage:
    python scripts/apobec/extract_sequences_and_structures.py
    python scripts/apobec/extract_sequences_and_structures.py --genome data/raw/genomes/hg19.fa
    python scripts/apobec/extract_sequences_and_structures.py --negatives  # also process negatives
"""

import argparse
import json
import logging
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LABELS_PATH = PROJECT_ROOT / "data" / "processed" / "editing_sites_labels.csv"
COMBINED_PATH = PROJECT_ROOT / "data" / "processed" / "advisor" / "positive_negative_combined.csv"
NEG_PATH = PROJECT_ROOT / "data" / "processed" / "advisor" / "negative_controls_ct.csv"
SUPPTX_PATH = PROJECT_ROOT / "data" / "processed" / "advisor" / "supp_tx_all_non_ag_mm_sites.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "sequences_and_structures.csv"
DEFAULT_GENOME = PROJECT_ROOT / "data" / "raw" / "genomes" / "hg19.fa"
RNAFOLD = Path("/opt/miniconda3/envs/vienna/bin/RNAfold")

FLANK_SIZE = 100  # nt on each side -> 201nt total window
UCSC_API_BASE = "https://api.genome.ucsc.edu/getData/sequence"


def revcomp_dna(seq: str) -> str:
    """Reverse complement a DNA sequence."""
    comp = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
    return "".join(comp.get(b, "N") for b in reversed(seq.upper()))


def dna_to_rna(seq: str) -> str:
    """Convert DNA to RNA (T -> U)."""
    return seq.upper().replace("T", "U")


# ---------------------------------------------------------------------------
# Sequence fetching: local genome or UCSC API
# ---------------------------------------------------------------------------

class GenomeFetcher:
    """Fetch sequences from hg19 -- local FASTA or UCSC REST API."""

    def __init__(self, genome_path=None):
        self.genome = None
        self.use_api = True

        if genome_path and Path(genome_path).exists():
            try:
                from pyfaidx import Fasta
                self.genome = Fasta(str(genome_path))
                self.use_api = False
                logger.info("Using local genome: %s (%d chromosomes)",
                            genome_path, len(self.genome.keys()))
            except Exception as e:
                logger.warning("Failed to open genome FASTA: %s. Falling back to API.", e)
        else:
            logger.info("No local genome. Using UCSC REST API (slower).")

    def fetch(self, chrom: str, start: int, end: int) -> str:
        """Fetch DNA sequence (0-based half-open coordinates)."""
        if self.use_api:
            return self._fetch_api(chrom, start, end)
        else:
            return self._fetch_local(chrom, start, end)

    def _fetch_local(self, chrom: str, start: int, end: int) -> str:
        """Fetch from local FASTA via pyfaidx (0-based)."""
        try:
            if chrom not in self.genome:
                return ""
            chrom_len = len(self.genome[chrom])
            s = max(0, start)
            e = min(chrom_len, end)
            if s >= e:
                return ""
            seq = str(self.genome[chrom][s:e]).upper()
            return seq
        except Exception as e:
            logger.warning("Local fetch failed for %s:%d-%d: %s", chrom, start, end, e)
            return ""

    def _fetch_api(self, chrom: str, start: int, end: int,
                   retries: int = 3, delay: float = 0.5) -> str:
        """Fetch from UCSC REST API (0-based)."""
        url = f"{UCSC_API_BASE}?genome=hg19;chrom={chrom};start={start};end={end}"
        for attempt in range(retries):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                resp = urllib.request.urlopen(req, timeout=20)
                data = json.loads(resp.read().decode())
                seq = data.get("dna", "").upper()
                if seq:
                    return seq
            except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
                logger.warning("UCSC API attempt %d failed for %s:%d-%d: %s",
                               attempt + 1, chrom, start, end, e)
                if attempt < retries - 1:
                    time.sleep(delay * (attempt + 1))
        return ""


# ---------------------------------------------------------------------------
# Sequence extraction for sites
# ---------------------------------------------------------------------------

def extract_sequences(sites_df: pd.DataFrame, fetcher: GenomeFetcher,
                      strand_map: dict, flank: int = FLANK_SIZE) -> pd.DataFrame:
    """Extract 201nt RNA sequences for all sites.

    For each site:
    - Fetches ±flank nt genomic DNA from hg19
    - Applies strand-aware conversion (revcomp for - strand)
    - Converts DNA to RNA (T -> U)
    """
    results = []
    total = len(sites_df)

    for i, (_, row) in enumerate(sites_df.iterrows()):
        site_id = row["site_id"]
        chrom = row["chr"]
        pos = row["start"]  # 0-based

        g_start = pos - flank
        g_end = pos + flank + 1

        dna_seq = fetcher.fetch(chrom, g_start, g_end)

        if not dna_seq or len(dna_seq) != 2 * flank + 1:
            results.append({
                "site_id": site_id,
                "chr": chrom,
                "start": pos,
                "strand": strand_map.get((chrom, pos), "?"),
                "label": row.get("label", 1),
                "sequence_201nt": "",
                "genomic_dna_201nt": dna_seq or "",
                "edit_position": flank,
            })
            if fetcher.use_api:
                time.sleep(0.2)
            continue

        strand = strand_map.get((chrom, pos), "+")

        if strand == "-":
            mRNA_dna = revcomp_dna(dna_seq)
            rna_seq = dna_to_rna(mRNA_dna)
        else:
            rna_seq = dna_to_rna(dna_seq)

        results.append({
            "site_id": site_id,
            "chr": chrom,
            "start": pos,
            "strand": strand,
            "label": row.get("label", 1),
            "sequence_201nt": rna_seq,
            "genomic_dna_201nt": dna_seq,
            "edit_position": flank,
        })

        if (i + 1) % 100 == 0:
            logger.info("Fetched %d/%d sequences", i + 1, total)

        if fetcher.use_api:
            time.sleep(0.3)

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Structure prediction
# ---------------------------------------------------------------------------

def predict_structures_batch(sequences: list, batch_size: int = 50) -> list:
    """Predict RNA secondary structures using RNAfold in batches."""
    if not RNAFOLD.exists():
        logger.error("RNAfold not found at %s", RNAFOLD)
        return [{"dot_bracket": "", "mfe": np.nan}] * len(sequences)

    all_results = []

    for batch_start in range(0, len(sequences), batch_size):
        batch = sequences[batch_start:batch_start + batch_size]
        valid_indices = [i for i, s in enumerate(batch) if s and str(s) != "nan"]
        valid_seqs = [batch[i] for i in valid_indices]

        if not valid_seqs:
            all_results.extend([{"dot_bracket": "", "mfe": np.nan}] * len(batch))
            continue

        batch_results = [{"dot_bracket": "", "mfe": np.nan}] * len(batch)

        try:
            input_text = "\n".join(valid_seqs)
            result = subprocess.run(
                [str(RNAFOLD), "--noPS", "-T37"],
                input=input_text,
                capture_output=True,
                text=True,
                timeout=600,
            )

            if result.returncode != 0:
                logger.error("RNAfold failed: %s", result.stderr[:200])
                all_results.extend(batch_results)
                continue

            lines = result.stdout.strip().split("\n")
            struct_idx = 0
            line_idx = 0
            while line_idx < len(lines) - 1 and struct_idx < len(valid_indices):
                struct_line = lines[line_idx + 1].strip()

                dot_bracket = ""
                mfe = np.nan
                for j in range(len(struct_line) - 1, -1, -1):
                    if struct_line[j] == "(" and j > 0:
                        remainder = struct_line[j:]
                        if ")" in remainder and any(
                            c.isdigit() or c == "-" for c in remainder
                        ):
                            dot_bracket = struct_line[:j].strip()
                            try:
                                mfe = float(remainder.strip("() "))
                            except ValueError:
                                pass
                            break

                orig_idx = valid_indices[struct_idx]
                batch_results[orig_idx] = {"dot_bracket": dot_bracket, "mfe": mfe}
                struct_idx += 1
                line_idx += 2

        except subprocess.TimeoutExpired:
            logger.error("RNAfold timed out on batch starting at %d", batch_start)
        except Exception as e:
            logger.error("Structure prediction error: %s", e)

        all_results.extend(batch_results)

        if (batch_start + len(batch)) % 200 == 0 or batch_start + len(batch) >= len(sequences):
            logger.info("Predicted structures for %d/%d",
                        batch_start + len(batch), len(sequences))

    return all_results


def classify_structure_at_position(dot_bracket: str, position: int) -> str:
    """Classify structural context at the edit position."""
    if not dot_bracket or position >= len(dot_bracket) or position < 0:
        return "unknown"
    return "paired" if dot_bracket[position] in "()" else "unpaired"


def compute_local_features(dot_bracket: str, position: int, window: int = 5) -> dict:
    """Compute local structural features around the edit position."""
    if not dot_bracket:
        return {
            "structure_at_edit": "unknown",
            "local_paired_fraction": np.nan,
        }

    n = len(dot_bracket)
    start = max(0, position - window)
    end = min(n, position + window + 1)
    local = dot_bracket[start:end]
    paired = sum(1 for c in local if c in "()")
    total = len(local)

    return {
        "structure_at_edit": classify_structure_at_position(dot_bracket, position),
        "local_paired_fraction": paired / total if total > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract RNA sequences and structures for editing sites")
    parser.add_argument("--genome", type=Path, default=DEFAULT_GENOME,
                        help="Path to hg19 FASTA (default: data/raw/genomes/hg19.fa)")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Batch size for RNAfold")
    parser.add_argument("--negatives", action="store_true",
                        help="Also process negative control sites")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH,
                        help="Output CSV path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Initialize genome fetcher
    fetcher = GenomeFetcher(args.genome)

    # Load strand info from Supp TX
    strand_map = {}
    if SUPPTX_PATH.exists():
        mm = pd.read_csv(SUPPTX_PATH)
        ct = mm[mm["Mismatch"] == "CT"]
        for _, r in ct.iterrows():
            strand_map[(r["Chr"], r["Start"])] = r["Strand"]
        logger.info("Loaded strand info for %d CT sites", len(strand_map))

    # Load positive sites
    if not LABELS_PATH.exists():
        logger.error("Labels file not found: %s", LABELS_PATH)
        return
    pos_sites = pd.read_csv(LABELS_PATH)
    pos_sites["label"] = 1
    logger.info("Loaded %d positive editing sites", len(pos_sites))

    # Optionally load negative sites
    if args.negatives and NEG_PATH.exists():
        neg_sites = pd.read_csv(NEG_PATH)
        neg_sites["label"] = 0
        # Standardize column names
        if "chr" not in neg_sites.columns and "Chr" in neg_sites.columns:
            neg_sites = neg_sites.rename(columns={"Chr": "chr", "Start": "start", "End": "end"})
        logger.info("Loaded %d negative control sites", len(neg_sites))

        # Combine
        sites = pd.concat([
            pos_sites[["site_id", "chr", "start", "label"]],
            neg_sites[["site_id", "chr", "start", "label"]],
        ], ignore_index=True)
    else:
        sites = pos_sites[["site_id", "chr", "start", "label"]].copy()
        if args.negatives:
            logger.warning("Negative controls file not found: %s", NEG_PATH)

    logger.info("Total sites to process: %d", len(sites))

    # Step 1: Extract sequences
    logger.info("Extracting 201nt genomic windows from hg19...")
    seq_df = extract_sequences(sites, fetcher, strand_map)
    n_success = (seq_df["sequence_201nt"] != "").sum()
    logger.info("Successfully extracted %d/%d sequences", n_success, len(seq_df))

    # Step 2: Predict RNA structures
    logger.info("Predicting RNA secondary structures with RNAfold...")
    sequences = seq_df["sequence_201nt"].tolist()
    structures = predict_structures_batch(sequences, batch_size=args.batch_size)

    # Step 3: Add structure columns
    seq_df["dot_bracket"] = [s["dot_bracket"] for s in structures]
    seq_df["mfe"] = [s["mfe"] for s in structures]

    # Step 4: Compute structural features
    edit_pos = FLANK_SIZE
    features = [compute_local_features(s["dot_bracket"], edit_pos) for s in structures]
    seq_df["structure_at_edit"] = [f["structure_at_edit"] for f in features]
    seq_df["local_paired_fraction"] = [f["local_paired_fraction"] for f in features]

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    seq_df.to_csv(args.output, index=False)
    logger.info("Saved to %s", args.output)

    # Summary
    logger.info("\n=== Summary ===")
    logger.info("Total sites: %d", len(seq_df))
    logger.info("Sequences extracted: %d", n_success)
    logger.info("Structures predicted: %d",
                (seq_df["dot_bracket"].astype(str) != "").sum())
    if "label" in seq_df.columns:
        logger.info("By label: %s", seq_df.groupby("label").size().to_dict())
    logger.info("Structure at edit: %s",
                seq_df["structure_at_edit"].value_counts().to_dict())
    logger.info("Mean MFE: %.2f kcal/mol", seq_df["mfe"].mean())
    logger.info("Strand: %s", seq_df["strand"].value_counts().to_dict())


if __name__ == "__main__":
    main()
