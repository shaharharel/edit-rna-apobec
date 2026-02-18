"""RNA secondary structure prediction using ViennaRNA.

Provides an interface to RNAfold for predicting RNA secondary structures
and extracting structural features relevant to APOBEC editing sites.

Requires ViennaRNA to be installed. Default path: /opt/miniconda3/envs/vienna/bin/
"""

import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_RNAFOLD = Path("/opt/miniconda3/envs/vienna/bin/RNAfold")


@dataclass
class RNAStructure:
    """RNA secondary structure prediction result."""
    sequence: str
    dot_bracket: str
    mfe: float  # minimum free energy in kcal/mol
    is_paired: list = field(default_factory=list)
    pair_partner: list = field(default_factory=list)

    def __post_init__(self):
        if not self.is_paired:
            self.is_paired, self.pair_partner = parse_dot_bracket(self.dot_bracket)

    @property
    def n_paired(self) -> int:
        return sum(self.is_paired)

    @property
    def paired_fraction(self) -> float:
        if not self.is_paired:
            return 0.0
        return self.n_paired / len(self.is_paired)

    def classify_position(self, position: int) -> str:
        """Classify a position as stem, hairpin_loop, internal_loop, etc."""
        if position < 0 or position >= len(self.dot_bracket):
            return "out_of_range"

        db = self.dot_bracket
        n = len(db)

        if db[position] in "()":
            return "stem"

        # Find nearest paired positions on each side
        left_paired = -1
        right_paired = -1
        for i in range(position - 1, -1, -1):
            if db[i] in "()":
                left_paired = i
                break
        for i in range(position + 1, n):
            if db[i] in "()":
                right_paired = i
                break

        if left_paired == -1 or right_paired == -1:
            return "external_loop"
        if db[left_paired] == "(" and db[right_paired] == ")":
            return "hairpin_loop"
        elif db[left_paired] == ")" and db[right_paired] == "(":
            return "internal_loop"
        else:
            return "bulge_or_multiloop"


def parse_dot_bracket(db: str) -> tuple:
    """Parse dot-bracket notation into paired/unpaired annotations.

    Returns:
        (is_paired, pair_partner): Lists of bools and partner indices (-1 if unpaired).
    """
    n = len(db)
    is_paired = [False] * n
    pair_partner = [-1] * n
    stack = []

    for i, c in enumerate(db):
        if c == "(":
            stack.append(i)
        elif c == ")":
            if stack:
                j = stack.pop()
                is_paired[i] = True
                is_paired[j] = True
                pair_partner[i] = j
                pair_partner[j] = i

    return is_paired, pair_partner


def predict_structure(
    sequence: str,
    rnafold_path: Path = DEFAULT_RNAFOLD,
    temperature: float = 37.0,
) -> Optional[RNAStructure]:
    """Predict RNA secondary structure using RNAfold.

    Args:
        sequence: RNA sequence (ACGU)
        rnafold_path: Path to RNAfold binary
        temperature: Folding temperature in Celsius

    Returns:
        RNAStructure object, or None on error.
    """
    if not rnafold_path.exists():
        logger.error("RNAfold not found at %s", rnafold_path)
        return None

    try:
        result = subprocess.run(
            [str(rnafold_path), "--noPS", f"-T{temperature}"],
            input=sequence,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            logger.error("RNAfold failed: %s", result.stderr)
            return None

        lines = result.stdout.strip().split("\n")
        if len(lines) < 2:
            logger.error("Unexpected RNAfold output: %s", result.stdout)
            return None

        seq_out = lines[0].strip()
        struct_line = lines[1].strip()

        # Parse "(((...))) (-1.23)" format
        # Find the energy value in parentheses at the end
        for i in range(len(struct_line) - 1, -1, -1):
            if struct_line[i] == "(" and i > 0:
                remainder = struct_line[i:]
                if ")" in remainder and any(c.isdigit() or c == "-" for c in remainder):
                    dot_bracket = struct_line[:i].strip()
                    mfe_str = remainder.strip("() ")
                    mfe = float(mfe_str)
                    return RNAStructure(
                        sequence=seq_out, dot_bracket=dot_bracket, mfe=mfe
                    )

        return None
    except subprocess.TimeoutExpired:
        logger.error("RNAfold timed out for sequence of length %d", len(sequence))
        return None
    except Exception as e:
        logger.error("Structure prediction failed: %s", e)
        return None


def batch_predict_structures(
    sequences: list,
    rnafold_path: Path = DEFAULT_RNAFOLD,
    temperature: float = 37.0,
) -> list:
    """Predict structures for multiple sequences in a single RNAfold call."""
    if not sequences:
        return []

    if not rnafold_path.exists():
        logger.error("RNAfold not found at %s", rnafold_path)
        return [None] * len(sequences)

    try:
        input_text = "\n".join(sequences)
        result = subprocess.run(
            [str(rnafold_path), "--noPS", f"-T{temperature}"],
            input=input_text,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            logger.error("RNAfold batch failed: %s", result.stderr)
            return [None] * len(sequences)

        lines = result.stdout.strip().split("\n")
        structures = []
        i = 0
        while i < len(lines) - 1:
            seq_line = lines[i].strip()
            struct_line = lines[i + 1].strip()

            parsed = False
            for j in range(len(struct_line) - 1, -1, -1):
                if struct_line[j] == "(" and j > 0:
                    remainder = struct_line[j:]
                    if ")" in remainder and any(c.isdigit() or c == "-" for c in remainder):
                        dot_bracket = struct_line[:j].strip()
                        mfe_str = remainder.strip("() ")
                        try:
                            mfe = float(mfe_str)
                            structures.append(RNAStructure(
                                sequence=seq_line, dot_bracket=dot_bracket, mfe=mfe
                            ))
                            parsed = True
                        except ValueError:
                            structures.append(None)
                            parsed = True
                        break
            if not parsed:
                structures.append(None)
            i += 2

        while len(structures) < len(sequences):
            structures.append(None)

        return structures
    except subprocess.TimeoutExpired:
        logger.error("RNAfold batch timed out")
        return [None] * len(sequences)
    except Exception as e:
        logger.error("Batch structure prediction failed: %s", e)
        return [None] * len(sequences)


def compute_structural_features(structure: RNAStructure, edit_position: int) -> dict:
    """Compute structural features around an editing site.

    Args:
        structure: RNA secondary structure
        edit_position: 0-based position of the C-to-U edit in the sequence

    Returns:
        Dictionary of structural features for ML input.
    """
    n = len(structure.sequence)
    features = {
        "mfe": structure.mfe,
        "mfe_per_nt": structure.mfe / n if n > 0 else 0.0,
        "paired_fraction": structure.paired_fraction,
        "edit_is_paired": structure.is_paired[edit_position] if edit_position < n else False,
        "edit_structural_context": structure.classify_position(edit_position),
    }

    # Local structure around edit (5nt window)
    window = 5
    start = max(0, edit_position - window)
    end = min(n, edit_position + window + 1)
    local_paired = sum(structure.is_paired[start:end])
    local_total = end - start
    features["local_paired_fraction"] = local_paired / local_total if local_total > 0 else 0.0

    # Distance to nearest stem
    if edit_position < n and not structure.is_paired[edit_position]:
        left_dist = 0
        for i in range(edit_position - 1, -1, -1):
            left_dist += 1
            if structure.is_paired[i]:
                break
        right_dist = 0
        for i in range(edit_position + 1, n):
            right_dist += 1
            if structure.is_paired[i]:
                break
        features["distance_to_nearest_stem"] = min(left_dist, right_dist)
    else:
        features["distance_to_nearest_stem"] = 0

    return features


def extract_sequence_context(
    chrom: str,
    position: int,
    flank_size: int = 50,
    genome_fasta: Optional[Path] = None,
) -> Optional[str]:
    """Extract flanking RNA sequence around a genomic position from hg19.

    Args:
        chrom: Chromosome (e.g., 'chr1')
        position: 0-based genomic position
        flank_size: Number of bases on each side
        genome_fasta: Path to indexed hg19 FASTA file (.fa with .fai index)

    Returns:
        Uppercase RNA sequence (T->U) of length 2*flank_size+1, or None on error.
    """
    if genome_fasta is None:
        logger.error("No genome FASTA path provided.")
        return None

    try:
        from pyfaidx import Fasta
    except ImportError:
        logger.error("pyfaidx not installed. Install with: pip install pyfaidx")
        return None

    try:
        genome = Fasta(str(genome_fasta))
        start = max(0, position - flank_size)
        end = position + flank_size + 1
        seq = str(genome[chrom][start:end]).upper()
        rna_seq = seq.replace("T", "U")
        return rna_seq
    except Exception as e:
        logger.error("Failed to extract sequence at %s:%d: %s", chrom, position, e)
        return None
