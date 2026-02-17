"""
RNA sequence utilities.

Provides functions for:
- Sequence validation and conversion
- Edit detection and characterization
- Regulatory motif analysis (Kozak, uAUG, uORF)
- Secondary structure prediction (optional)
"""

import re
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from collections import Counter


# =============================================================================
# Sequence Validation and Conversion
# =============================================================================

def validate_rna_sequence(seq: str, allow_n: bool = False) -> bool:
    """
    Validate that sequence contains only valid RNA nucleotides.

    Args:
        seq: Input sequence
        allow_n: If True, allow N for ambiguous nucleotides

    Returns:
        True if valid, False otherwise
    """
    if allow_n:
        valid_chars = set('ACGUNT')
    else:
        valid_chars = set('ACGUT')

    return set(seq.upper()) <= valid_chars


def dna_to_rna(seq: str) -> str:
    """Convert DNA sequence to RNA (T -> U)."""
    return seq.upper().replace('T', 'U')


def rna_to_dna(seq: str) -> str:
    """Convert RNA sequence to DNA (U -> T)."""
    return seq.upper().replace('U', 'T')


def reverse_complement(seq: str, is_rna: bool = True) -> str:
    """
    Compute reverse complement of sequence.

    Args:
        seq: Input sequence
        is_rna: If True, treat as RNA (U); if False, treat as DNA (T)

    Returns:
        Reverse complement sequence
    """
    if is_rna:
        complement = {'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
    else:
        complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}

    seq = seq.upper()
    return ''.join(complement.get(n, 'N') for n in reversed(seq))


def gc_content(seq: str) -> float:
    """Calculate GC content of sequence."""
    seq = seq.upper()
    gc = seq.count('G') + seq.count('C')
    return gc / len(seq) if seq else 0.0


# =============================================================================
# Edit Detection and Characterization
# =============================================================================

def compute_edit_distance(seq_a: str, seq_b: str) -> int:
    """
    Compute Levenshtein edit distance between two sequences.

    Args:
        seq_a: First sequence
        seq_b: Second sequence

    Returns:
        Edit distance (number of insertions, deletions, substitutions)
    """
    m, n = len(seq_a), len(seq_b)

    # Use dynamic programming
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq_a[i-1] == seq_b[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],      # Deletion
                    dp[i][j-1],      # Insertion
                    dp[i-1][j-1]     # Substitution
                )

    return dp[m][n]


def compute_hamming_distance(seq_a: str, seq_b: str) -> int:
    """
    Compute Hamming distance (for equal-length sequences).

    Returns -1 if sequences have different lengths.
    """
    if len(seq_a) != len(seq_b):
        return -1

    return sum(a != b for a, b in zip(seq_a.upper(), seq_b.upper()))


def extract_edit(
    seq_a: str,
    seq_b: str,
    context_size: int = 10
) -> Dict:
    """
    Extract detailed edit information between two sequences.

    Handles:
    - Single nucleotide variants (SNVs)
    - Insertions
    - Deletions
    - Complex edits (multiple changes)

    Args:
        seq_a: Reference sequence
        seq_b: Variant sequence
        context_size: Number of nucleotides to include around edit

    Returns:
        Dict with edit information:
        {
            'edit_type': 'SNV' | 'insertion' | 'deletion' | 'complex',
            'position': int (0-indexed position in seq_a),
            'edit_from': str (nucleotides removed from seq_a),
            'edit_to': str (nucleotides added in seq_b),
            'local_context_a': str (context around edit in seq_a),
            'local_context_b': str (context around edit in seq_b),
            'edit_size': int (number of nucleotides changed),
            'hamming_distance': int (if equal length)
        }
    """
    seq_a = seq_a.upper()
    seq_b = seq_b.upper()

    result = {
        'edit_type': 'unknown',
        'position': -1,
        'edit_from': '',
        'edit_to': '',
        'local_context_a': '',
        'local_context_b': '',
        'edit_size': 0,
        'hamming_distance': -1
    }

    # Equal length: look for SNVs
    if len(seq_a) == len(seq_b):
        result['hamming_distance'] = compute_hamming_distance(seq_a, seq_b)

        # Find differing positions
        diff_positions = [i for i in range(len(seq_a)) if seq_a[i] != seq_b[i]]

        if len(diff_positions) == 0:
            result['edit_type'] = 'none'
            return result

        elif len(diff_positions) == 1:
            # Single SNV
            pos = diff_positions[0]
            result['edit_type'] = 'SNV'
            result['position'] = pos
            result['edit_from'] = seq_a[pos]
            result['edit_to'] = seq_b[pos]
            result['edit_size'] = 1

        else:
            # Multiple changes
            result['edit_type'] = 'complex'
            result['position'] = diff_positions[0]
            result['edit_from'] = ''.join(seq_a[i] for i in diff_positions)
            result['edit_to'] = ''.join(seq_b[i] for i in diff_positions)
            result['edit_size'] = len(diff_positions)

    # Different lengths: insertion or deletion
    else:
        len_diff = len(seq_b) - len(seq_a)

        # Find first differing position using simple alignment
        first_diff = 0
        for i in range(min(len(seq_a), len(seq_b))):
            if seq_a[i] != seq_b[i]:
                first_diff = i
                break
        else:
            first_diff = min(len(seq_a), len(seq_b))

        result['position'] = first_diff

        if len_diff > 0:
            # Insertion in seq_b
            result['edit_type'] = 'insertion'
            result['edit_from'] = ''
            result['edit_to'] = seq_b[first_diff:first_diff + len_diff]
            result['edit_size'] = len_diff

        else:
            # Deletion from seq_a
            result['edit_type'] = 'deletion'
            result['edit_from'] = seq_a[first_diff:first_diff - len_diff]
            result['edit_to'] = ''
            result['edit_size'] = -len_diff

    # Extract local context
    pos = result['position']
    start_a = max(0, pos - context_size)
    end_a = min(len(seq_a), pos + context_size + 1)
    start_b = max(0, pos - context_size)
    end_b = min(len(seq_b), pos + context_size + 1)

    result['local_context_a'] = seq_a[start_a:end_a]
    result['local_context_b'] = seq_b[start_b:end_b]

    return result


def find_all_differences(seq_a: str, seq_b: str) -> List[Dict]:
    """
    Find all differing positions between two equal-length sequences.

    Returns list of dicts, each describing one difference.
    """
    if len(seq_a) != len(seq_b):
        raise ValueError("Sequences must have equal length")

    differences = []
    for i, (a, b) in enumerate(zip(seq_a.upper(), seq_b.upper())):
        if a != b:
            differences.append({
                'position': i,
                'from': a,
                'to': b
            })

    return differences


# =============================================================================
# Regulatory Motif Analysis
# =============================================================================

# Kozak consensus: GCC(A/G)CCAUGG
# Positions: -6 to +4 relative to AUG
# Most important: -3 (should be A or G), +4 (should be G)
KOZAK_WEIGHT_MATRIX = {
    -6: {'G': 0.5, 'A': 0.3, 'C': 0.1, 'U': 0.1},
    -5: {'C': 0.5, 'G': 0.2, 'A': 0.2, 'U': 0.1},
    -4: {'C': 0.5, 'A': 0.2, 'G': 0.2, 'U': 0.1},
    -3: {'A': 0.5, 'G': 0.4, 'C': 0.05, 'U': 0.05},  # Critical position
    -2: {'C': 0.4, 'A': 0.2, 'G': 0.2, 'U': 0.2},
    -1: {'C': 0.4, 'A': 0.2, 'G': 0.2, 'U': 0.2},
    # 0, 1, 2 = AUG (fixed)
    +4: {'G': 0.5, 'A': 0.2, 'C': 0.2, 'U': 0.1},  # Critical position
}


def compute_kozak_score(seq: str, aug_position: int) -> float:
    """
    Compute Kozak consensus strength at a given AUG position.

    The Kozak sequence is the context around the start codon (AUG)
    that affects translation initiation efficiency.

    Strong Kozak: (A/G)CCAUGG (A or G at -3, G at +4)
    Weak Kozak: poor context

    Args:
        seq: RNA sequence (should contain the AUG)
        aug_position: 0-indexed position of the A in AUG

    Returns:
        Kozak score between 0 and 1 (1 = perfect consensus)
    """
    seq = seq.upper()

    # Check that AUG is at the specified position
    if aug_position + 2 >= len(seq):
        return 0.0

    if seq[aug_position:aug_position + 3] != 'AUG':
        return 0.0

    score = 0.0
    max_score = 0.0

    for offset, weights in KOZAK_WEIGHT_MATRIX.items():
        pos = aug_position + offset
        if 0 <= pos < len(seq):
            nuc = seq[pos]
            score += weights.get(nuc, 0)
        max_score += max(weights.values())

    return score / max_score if max_score > 0 else 0.0


def find_uaugs(seq: str) -> List[int]:
    """
    Find all upstream AUG (uAUG) positions in a 5' UTR.

    uAUGs can reduce translation of the main ORF by:
    - Initiating translation at the wrong start
    - Creating upstream ORFs (uORFs)

    Args:
        seq: 5' UTR sequence

    Returns:
        List of 0-indexed positions where AUG occurs
    """
    seq = seq.upper()
    positions = []

    start = 0
    while True:
        pos = seq.find('AUG', start)
        if pos == -1:
            break
        positions.append(pos)
        start = pos + 1

    return positions


def find_uorfs(seq: str, min_length: int = 9) -> List[Dict]:
    """
    Find upstream open reading frames (uORFs) in a 5' UTR.

    A uORF starts with AUG and ends with a stop codon (UAA, UAG, UGA).
    uORFs can significantly reduce translation of the main ORF.

    Args:
        seq: 5' UTR sequence
        min_length: Minimum uORF length in nucleotides (default: 9 = 3 codons)

    Returns:
        List of dicts with uORF information:
        {
            'start': int (AUG position),
            'stop': int (stop codon position),
            'length': int (nt),
            'sequence': str,
            'kozak_score': float
        }
    """
    seq = seq.upper()
    stop_codons = ['UAA', 'UAG', 'UGA']
    uorfs = []

    for aug_pos in find_uaugs(seq):
        # Search for in-frame stop codon
        for i in range(aug_pos + 3, len(seq) - 2, 3):
            codon = seq[i:i+3]
            if codon in stop_codons:
                length = i + 3 - aug_pos
                if length >= min_length:
                    uorfs.append({
                        'start': aug_pos,
                        'stop': i,
                        'length': length,
                        'sequence': seq[aug_pos:i+3],
                        'kozak_score': compute_kozak_score(seq, aug_pos)
                    })
                break

    return uorfs


def find_are_motifs(seq: str) -> List[Dict]:
    """
    Find AU-rich elements (AREs) in a 3' UTR.

    AREs are regulatory elements that typically destabilize mRNA.
    Common patterns: AUUUA, UUAUUUA(U/A)(U/A)

    Args:
        seq: 3' UTR sequence

    Returns:
        List of dicts with ARE information
    """
    seq = seq.upper()
    ares = []

    # Class I AREs: scattered AUUUA motifs
    pattern_class1 = re.compile(r'AUUUA')
    for match in pattern_class1.finditer(seq):
        ares.append({
            'type': 'class_I',
            'position': match.start(),
            'sequence': match.group(),
            'length': 5
        })

    # Class II AREs: overlapping AUUUA
    pattern_class2 = re.compile(r'AUUUAUUUA')
    for match in pattern_class2.finditer(seq):
        ares.append({
            'type': 'class_II',
            'position': match.start(),
            'sequence': match.group(),
            'length': 9
        })

    # Class III AREs: U-rich regions
    pattern_class3 = re.compile(r'U{5,}')
    for match in pattern_class3.finditer(seq):
        ares.append({
            'type': 'class_III',
            'position': match.start(),
            'sequence': match.group(),
            'length': len(match.group())
        })

    return ares


def compute_cap_accessibility(seq: str, window_size: int = 30) -> float:
    """
    Estimate 5' cap accessibility based on predicted structure.

    A more accessible (less structured) 5' end typically leads
    to more efficient translation initiation.

    This is a simple heuristic based on local GC content and
    self-complementarity. For accurate predictions, use ViennaRNA.

    Args:
        seq: 5' UTR sequence
        window_size: Size of 5' window to analyze

    Returns:
        Accessibility score between 0 and 1 (1 = fully accessible)
    """
    seq = seq.upper()
    window = seq[:min(window_size, len(seq))]

    if len(window) == 0:
        return 1.0

    # GC content reduces accessibility (more stable structures)
    gc = gc_content(window)

    # Self-complementarity (potential for hairpins)
    rc = reverse_complement(window)
    self_comp = 0
    for i in range(len(window) - 4):
        if window[i:i+4] in rc:
            self_comp += 1
    self_comp_score = self_comp / max(len(window) - 3, 1)

    # Combine factors (higher GC and self-comp = lower accessibility)
    accessibility = 1.0 - (0.5 * gc + 0.5 * self_comp_score)

    return max(0.0, min(1.0, accessibility))


# =============================================================================
# Secondary Structure (Optional - requires ViennaRNA)
# =============================================================================

def predict_secondary_structure(seq: str) -> Optional[Dict]:
    """
    Predict RNA secondary structure using ViennaRNA (if available).

    Args:
        seq: RNA sequence

    Returns:
        Dict with structure information, or None if ViennaRNA not available:
        {
            'structure': str (dot-bracket notation),
            'mfe': float (minimum free energy in kcal/mol),
            'paired_fraction': float,
            'ensemble_diversity': float
        }
    """
    try:
        import RNA
    except ImportError:
        return None

    seq = seq.upper().replace('T', 'U')

    # Compute MFE structure
    structure, mfe = RNA.fold(seq)

    # Compute partition function for ensemble properties
    fc = RNA.fold_compound(seq)
    fc.pf()

    # Ensemble diversity
    ensemble_diversity = fc.mean_bp_distance()

    # Paired fraction
    paired = structure.count('(') + structure.count(')')
    paired_fraction = paired / len(structure) if structure else 0

    return {
        'structure': structure,
        'mfe': mfe,
        'paired_fraction': paired_fraction,
        'ensemble_diversity': ensemble_diversity
    }


def compute_local_structure(
    seq: str,
    position: int,
    window_size: int = 50
) -> Optional[Dict]:
    """
    Compute local secondary structure around a specific position.

    Useful for analyzing structure context of edits.

    Args:
        seq: Full RNA sequence
        position: Position of interest
        window_size: Size of window around position

    Returns:
        Dict with local structure information
    """
    start = max(0, position - window_size // 2)
    end = min(len(seq), position + window_size // 2)

    local_seq = seq[start:end]
    local_pos = position - start

    result = predict_secondary_structure(local_seq)

    if result is not None:
        result['local_position'] = local_pos
        result['is_paired'] = result['structure'][local_pos] != '.' if local_pos < len(result['structure']) else False

    return result


# =============================================================================
# Codon Analysis (for coding regions)
# =============================================================================

CODON_TABLE = {
    'UUU': 'F', 'UUC': 'F', 'UUA': 'L', 'UUG': 'L',
    'UCU': 'S', 'UCC': 'S', 'UCA': 'S', 'UCG': 'S',
    'UAU': 'Y', 'UAC': 'Y', 'UAA': '*', 'UAG': '*',
    'UGU': 'C', 'UGC': 'C', 'UGA': '*', 'UGG': 'W',
    'CUU': 'L', 'CUC': 'L', 'CUA': 'L', 'CUG': 'L',
    'CCU': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'CAU': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'CGU': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'AUU': 'I', 'AUC': 'I', 'AUA': 'I', 'AUG': 'M',
    'ACU': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'AAU': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
    'AGU': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GUU': 'V', 'GUC': 'V', 'GUA': 'V', 'GUG': 'V',
    'GCU': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'GAU': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'GGU': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
}


def translate(seq: str, frame: int = 0) -> str:
    """
    Translate RNA sequence to protein.

    Args:
        seq: RNA sequence
        frame: Reading frame (0, 1, or 2)

    Returns:
        Amino acid sequence (stops at first stop codon)
    """
    seq = seq.upper()
    protein = []

    for i in range(frame, len(seq) - 2, 3):
        codon = seq[i:i+3]
        aa = CODON_TABLE.get(codon, 'X')
        if aa == '*':
            break
        protein.append(aa)

    return ''.join(protein)


def is_synonymous_edit(
    seq_a: str,
    seq_b: str,
    frame: int = 0
) -> bool:
    """
    Check if an edit is synonymous (doesn't change protein sequence).

    Args:
        seq_a: Reference sequence
        seq_b: Variant sequence
        frame: Reading frame

    Returns:
        True if edit is synonymous
    """
    return translate(seq_a, frame) == translate(seq_b, frame)
