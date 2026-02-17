"""
RNA-specific data splitting strategies for sequence-based prediction.

This module implements splitting strategies designed for RNA sequences,
analogous to the molecular splitters but using sequence-based similarity
and regulatory motif features.

Strategies:
- Random split: Baseline random splitting
- Sequence similarity split: Split by k-mer similarity (analogous to scaffold split)
- Motif-based split: Split by regulatory motif presence (Kozak, uAUG, uORF)
- Edit type split: Split by type of edit (SNV, insertion, deletion)
- GC content stratified: Split with stratification by GC content
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional, Set
from collections import defaultdict, Counter
from abc import ABC, abstractmethod
import warnings


class RNASplitter(ABC):
    """Base class for RNA sequence data splitting strategies."""

    def __init__(
        self,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
        random_state: Optional[int] = 42
    ):
        """
        Initialize splitter.

        Args:
            train_size: Fraction for training set
            val_size: Fraction for validation set
            test_size: Fraction for test set
            random_state: Random seed for reproducibility
        """
        assert abs(train_size + val_size + test_size - 1.0) < 1e-6, \
            f"Sizes must sum to 1.0, got {train_size + val_size + test_size}"

        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.random_state = random_state

    @abstractmethod
    def split(
        self,
        df: pd.DataFrame,
        seq_col: str = 'seq_a'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split dataframe into train/val/test sets.

        Args:
            df: DataFrame with RNA sequence data
            seq_col: Column name containing sequences

        Returns:
            train, val, test DataFrames
        """
        pass

    def _split_indices_to_dataframes(
        self,
        df: pd.DataFrame,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        test_idx: np.ndarray
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Convert indices to train/val/test DataFrames.

        NOTE: We intentionally do NOT reset_index here because downstream code
        uses df.index.values to index into embedding arrays. Resetting would
        cause all splits to have indices 0,1,2,... leading to train/test overlap.
        """
        return (
            df.iloc[train_idx],
            df.iloc[val_idx],
            df.iloc[test_idx]
        )


class RandomRNASplitter(RNASplitter):
    """Random split - baseline splitting strategy for RNA data."""

    def split(
        self,
        df: pd.DataFrame,
        seq_col: str = 'seq_a'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Randomly split data."""
        n = len(df)
        indices = np.arange(n)

        np.random.seed(self.random_state)
        np.random.shuffle(indices)

        train_end = int(n * self.train_size)
        val_end = train_end + int(n * self.val_size)

        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]

        return self._split_indices_to_dataframes(df, train_idx, val_idx, test_idx)


class SequenceSimilaritySplitter(RNASplitter):
    """
    Split by sequence similarity using k-mer overlap.

    Analogous to scaffold split for molecules - ensures that similar
    sequences are in the same split to test generalization to novel
    sequence contexts.

    Uses k-mer frequency vectors and clustering to group similar sequences.
    """

    def __init__(
        self,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
        random_state: Optional[int] = 42,
        kmer_size: int = 4,
        similarity_threshold: float = 0.7,
        max_cluster_size: int = 1000
    ):
        """
        Initialize sequence similarity splitter.

        Args:
            kmer_size: K-mer size for computing sequence similarity
            similarity_threshold: Minimum k-mer overlap for clustering (0-1)
            max_cluster_size: Maximum cluster size before breaking up
        """
        super().__init__(train_size, val_size, test_size, random_state)
        self.kmer_size = kmer_size
        self.similarity_threshold = similarity_threshold
        self.max_cluster_size = max_cluster_size

    def _get_kmer_set(self, seq: str) -> Set[str]:
        """Extract k-mers from sequence."""
        seq = seq.upper().replace('T', 'U')
        kmers = set()
        for i in range(len(seq) - self.kmer_size + 1):
            kmer = seq[i:i + self.kmer_size]
            if all(c in 'ACGU' for c in kmer):
                kmers.add(kmer)
        return kmers

    def _jaccard_similarity(self, set1: Set[str], set2: Set[str]) -> float:
        """Compute Jaccard similarity between two k-mer sets."""
        if not set1 or not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0

    def _cluster_sequences(
        self,
        sequences: List[str],
        kmer_sets: List[Set[str]]
    ) -> List[List[int]]:
        """
        Cluster sequences by k-mer similarity using greedy clustering.

        Returns list of clusters, where each cluster is a list of indices.
        """
        n = len(sequences)
        clustered = [False] * n
        clusters = []

        # Sort by sequence length (longer sequences first as cluster centers)
        sorted_indices = sorted(range(n), key=lambda i: len(sequences[i]), reverse=True)

        for center_idx in sorted_indices:
            if clustered[center_idx]:
                continue

            # Start new cluster
            cluster = [center_idx]
            clustered[center_idx] = True
            center_kmers = kmer_sets[center_idx]

            # Find similar sequences
            for other_idx in range(n):
                if clustered[other_idx]:
                    continue

                similarity = self._jaccard_similarity(center_kmers, kmer_sets[other_idx])

                if similarity >= self.similarity_threshold:
                    cluster.append(other_idx)
                    clustered[other_idx] = True

                    if len(cluster) >= self.max_cluster_size:
                        break

            clusters.append(cluster)

        return clusters

    def split(
        self,
        df: pd.DataFrame,
        seq_col: str = 'seq_a'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split by sequence similarity clustering.

        Strategy:
        1. Compute k-mer sets for each sequence
        2. Cluster by k-mer overlap
        3. Assign clusters to splits
        """
        print(f"Computing {self.kmer_size}-mer sets for sequence similarity clustering...")

        # Get unique sequences and their indices
        if 'seq_b' in df.columns:
            # For pairs data, consider both sequences
            all_seqs = list(set(df[seq_col].tolist() + df['seq_b'].tolist()))
        else:
            all_seqs = df[seq_col].unique().tolist()

        # Compute k-mer sets
        kmer_sets = [self._get_kmer_set(seq) for seq in all_seqs]

        print(f"  Processing {len(all_seqs)} unique sequences...")

        # Cluster sequences
        clusters = self._cluster_sequences(all_seqs, kmer_sets)

        print(f"  Found {len(clusters)} sequence clusters")
        if clusters:
            print(f"  Largest cluster: {len(clusters[0])} sequences")
            print(f"  Smallest cluster: {len(clusters[-1])} sequences")

        # Map clusters back to DataFrame rows
        seq_to_cluster = {}
        for cluster_id, cluster in enumerate(clusters):
            for seq_idx in cluster:
                seq_to_cluster[all_seqs[seq_idx]] = cluster_id

        # Assign rows to clusters (use seq_a as primary)
        row_to_cluster = []
        for _, row in df.iterrows():
            seq = row[seq_col]
            cluster_id = seq_to_cluster.get(seq, -1)
            row_to_cluster.append(cluster_id)

        # Group rows by cluster
        cluster_to_rows = defaultdict(list)
        for row_idx, cluster_id in enumerate(row_to_cluster):
            cluster_to_rows[cluster_id].append(row_idx)

        # Sort clusters by size
        cluster_sizes = [(cid, len(rows)) for cid, rows in cluster_to_rows.items()]
        cluster_sizes.sort(key=lambda x: x[1], reverse=True)

        # Allocate clusters to splits
        n = len(df)
        target_train = int(n * self.train_size)
        target_val = int(n * self.val_size)

        train_idx, val_idx, test_idx = [], [], []
        train_count, val_count = 0, 0

        np.random.seed(self.random_state)
        np.random.shuffle(cluster_sizes)

        for cluster_id, size in cluster_sizes:
            rows = cluster_to_rows[cluster_id]

            if train_count < target_train:
                train_idx.extend(rows)
                train_count += size
            elif val_count < target_val:
                val_idx.extend(rows)
                val_count += size
            else:
                test_idx.extend(rows)

        print(f"  Split sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

        return self._split_indices_to_dataframes(
            df,
            np.array(train_idx),
            np.array(val_idx),
            np.array(test_idx)
        )


class MotifSplitter(RNASplitter):
    """
    Split by regulatory motif presence.

    Groups sequences by presence/absence of regulatory motifs:
    - Kozak consensus
    - Upstream AUG (uAUG)
    - Upstream open reading frames (uORFs)
    - AU-rich elements (AREs)

    Ensures that sequences with similar regulatory context are in same split.
    """

    # Kozak consensus pattern (simplified)
    KOZAK_PATTERN = 'ACCAUGG'  # Strong Kozak

    # AU-rich element patterns
    ARE_PATTERNS = ['AUUUA', 'UUAUUUAUU']

    def __init__(
        self,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
        random_state: Optional[int] = 42,
        motif_types: List[str] = None
    ):
        """
        Initialize motif splitter.

        Args:
            motif_types: List of motif types to consider.
                        Options: ['kozak', 'uaug', 'uorf', 'are', 'gc_rich']
                        Default: all
        """
        super().__init__(train_size, val_size, test_size, random_state)
        self.motif_types = motif_types or ['kozak', 'uaug', 'uorf', 'are']

    def _has_kozak(self, seq: str) -> bool:
        """Check for Kozak consensus."""
        seq = seq.upper().replace('T', 'U')
        # Look for strong Kozak or partial matches
        if 'ACCAUGG' in seq or 'GCCAUGG' in seq:
            return True
        # Weaker check: A/G at -3 and G at +4
        for i in range(len(seq) - 6):
            if seq[i:i+3] == 'AUG':
                if i >= 3 and seq[i-3] in 'AG':
                    return True
        return False

    def _has_uaug(self, seq: str, main_aug_pos: int = None) -> bool:
        """Check for upstream AUG."""
        seq = seq.upper().replace('T', 'U')
        # If main AUG position not specified, assume it's the first one
        if main_aug_pos is None:
            main_aug_pos = seq.find('AUG')
            if main_aug_pos == -1:
                return False

        # Look for AUG before main position
        for i in range(main_aug_pos):
            if seq[i:i+3] == 'AUG':
                return True
        return False

    def _has_uorf(self, seq: str) -> bool:
        """Check for upstream ORF (AUG followed by in-frame stop)."""
        seq = seq.upper().replace('T', 'U')
        stop_codons = {'UAA', 'UAG', 'UGA'}

        for i in range(len(seq) - 5):
            if seq[i:i+3] == 'AUG':
                # Check for in-frame stop codon
                for j in range(i + 3, len(seq) - 2, 3):
                    if seq[j:j+3] in stop_codons:
                        return True
        return False

    def _has_are(self, seq: str) -> bool:
        """Check for AU-rich elements."""
        seq = seq.upper().replace('T', 'U')
        for pattern in self.ARE_PATTERNS:
            if pattern in seq:
                return True
        return False

    def _is_gc_rich(self, seq: str, threshold: float = 0.6) -> bool:
        """Check if sequence is GC-rich."""
        seq = seq.upper().replace('T', 'U')
        gc_count = seq.count('G') + seq.count('C')
        return gc_count / len(seq) >= threshold if len(seq) > 0 else False

    def _get_motif_signature(self, seq: str) -> tuple:
        """Get motif signature for sequence."""
        features = []

        if 'kozak' in self.motif_types:
            features.append(self._has_kozak(seq))
        if 'uaug' in self.motif_types:
            features.append(self._has_uaug(seq))
        if 'uorf' in self.motif_types:
            features.append(self._has_uorf(seq))
        if 'are' in self.motif_types:
            features.append(self._has_are(seq))
        if 'gc_rich' in self.motif_types:
            features.append(self._is_gc_rich(seq))

        return tuple(features)

    def split(
        self,
        df: pd.DataFrame,
        seq_col: str = 'seq_a'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split by motif signature.

        Strategy:
        1. Compute motif signature for each sequence
        2. Group by signature
        3. Assign signature groups to splits
        """
        print(f"Computing motif signatures ({', '.join(self.motif_types)})...")

        # Compute signatures
        signatures = df[seq_col].apply(self._get_motif_signature)

        # Group by signature
        sig_to_indices = defaultdict(list)
        for idx, sig in enumerate(signatures):
            sig_to_indices[sig].append(idx)

        # Sort by group size
        sig_sizes = [(sig, len(indices)) for sig, indices in sig_to_indices.items()]
        sig_sizes.sort(key=lambda x: x[1], reverse=True)

        print(f"  Found {len(sig_sizes)} unique motif signatures")
        print(f"  Largest group: {sig_sizes[0][1]} sequences")

        # Show distribution
        for sig, size in sig_sizes[:5]:
            features = []
            for i, motif_type in enumerate(self.motif_types):
                if sig[i]:
                    features.append(f"+{motif_type}")
                else:
                    features.append(f"-{motif_type}")
            print(f"    {' '.join(features)}: {size} sequences")

        # Allocate groups to splits
        n = len(df)
        target_train = int(n * self.train_size)
        target_val = int(n * self.val_size)

        train_idx, val_idx, test_idx = [], [], []
        train_count, val_count = 0, 0

        np.random.seed(self.random_state)
        np.random.shuffle(sig_sizes)

        for sig, size in sig_sizes:
            indices = sig_to_indices[sig]

            if train_count < target_train:
                train_idx.extend(indices)
                train_count += size
            elif val_count < target_val:
                val_idx.extend(indices)
                val_count += size
            else:
                test_idx.extend(indices)

        print(f"  Split sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

        return self._split_indices_to_dataframes(
            df,
            np.array(train_idx),
            np.array(val_idx),
            np.array(test_idx)
        )


class EditTypeSplitter(RNASplitter):
    """
    Split by type of edit between sequence pairs.

    For paired data (seq_a → seq_b), groups by:
    - Edit type (SNV, insertion, deletion, multiple)
    - Edit position (5' region, middle, 3' region)
    - Nucleotide change type (transition vs transversion)

    Tests model generalization to different edit types.
    """

    def __init__(
        self,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
        random_state: Optional[int] = 42,
        position_bins: int = 3
    ):
        """
        Initialize edit type splitter.

        Args:
            position_bins: Number of position bins (default: 3 for 5'/mid/3')
        """
        super().__init__(train_size, val_size, test_size, random_state)
        self.position_bins = position_bins

    def _classify_edit(self, seq_a: str, seq_b: str) -> tuple:
        """
        Classify the edit between two sequences.

        Returns:
            (edit_type, position_bin, nuc_change_type)
        """
        seq_a = seq_a.upper().replace('T', 'U')
        seq_b = seq_b.upper().replace('T', 'U')

        len_a = len(seq_a)
        len_b = len(seq_b)

        # Determine edit type based on length
        if len_a == len_b:
            # Count differences
            diffs = [(i, seq_a[i], seq_b[i]) for i in range(len_a) if seq_a[i] != seq_b[i]]

            if len(diffs) == 0:
                return ('identical', 0, 'none')
            elif len(diffs) == 1:
                edit_type = 'snv'
                pos = diffs[0][0]
                old_nuc, new_nuc = diffs[0][1], diffs[0][2]
            else:
                edit_type = 'multiple_snv'
                pos = diffs[0][0]  # Use first diff position
                old_nuc, new_nuc = diffs[0][1], diffs[0][2]

        elif len_b > len_a:
            edit_type = 'insertion'
            pos = 0  # Simplified - would need alignment
            old_nuc, new_nuc = 'N', 'N'

        else:
            edit_type = 'deletion'
            pos = 0
            old_nuc, new_nuc = 'N', 'N'

        # Determine position bin
        ref_len = max(len_a, len_b)
        if ref_len > 0:
            rel_pos = pos / ref_len
            position_bin = min(int(rel_pos * self.position_bins), self.position_bins - 1)
        else:
            position_bin = 0

        # Classify nucleotide change (for SNVs)
        purines = {'A', 'G'}
        pyrimidines = {'C', 'U'}

        if edit_type in ['snv', 'multiple_snv']:
            if (old_nuc in purines and new_nuc in purines) or \
               (old_nuc in pyrimidines and new_nuc in pyrimidines):
                nuc_change = 'transition'
            else:
                nuc_change = 'transversion'
        else:
            nuc_change = 'indel'

        return (edit_type, position_bin, nuc_change)

    def split(
        self,
        df: pd.DataFrame,
        seq_col: str = 'seq_a'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split by edit type classification.

        Requires DataFrame with 'seq_a' and 'seq_b' columns.
        """
        if 'seq_b' not in df.columns:
            raise ValueError("EditTypeSplitter requires 'seq_b' column for pair data")

        print(f"Classifying edits for {len(df)} pairs...")

        # Classify all edits
        edit_classes = []
        for _, row in df.iterrows():
            edit_class = self._classify_edit(row[seq_col], row['seq_b'])
            edit_classes.append(edit_class)

        # Group by edit class
        class_to_indices = defaultdict(list)
        for idx, edit_class in enumerate(edit_classes):
            class_to_indices[edit_class].append(idx)

        # Show distribution
        print(f"  Found {len(class_to_indices)} unique edit classes")
        class_sizes = [(cls, len(indices)) for cls, indices in class_to_indices.items()]
        class_sizes.sort(key=lambda x: x[1], reverse=True)

        for cls, size in class_sizes[:10]:
            edit_type, pos_bin, nuc_change = cls
            pos_label = ['5\'', 'mid', '3\''][pos_bin] if pos_bin < 3 else str(pos_bin)
            print(f"    {edit_type}/{pos_label}/{nuc_change}: {size}")

        # Allocate classes to splits
        n = len(df)
        target_train = int(n * self.train_size)
        target_val = int(n * self.val_size)

        train_idx, val_idx, test_idx = [], [], []
        train_count, val_count = 0, 0

        np.random.seed(self.random_state)
        np.random.shuffle(class_sizes)

        for cls, size in class_sizes:
            indices = class_to_indices[cls]

            if train_count < target_train:
                train_idx.extend(indices)
                train_count += size
            elif val_count < target_val:
                val_idx.extend(indices)
                val_count += size
            else:
                test_idx.extend(indices)

        print(f"  Split sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

        return self._split_indices_to_dataframes(
            df,
            np.array(train_idx),
            np.array(val_idx),
            np.array(test_idx)
        )


class GCStratifiedSplitter(RNASplitter):
    """
    Stratified split by GC content.

    Ensures balanced GC content distribution across train/val/test.
    Important for avoiding GC-biased models.
    """

    def __init__(
        self,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
        random_state: Optional[int] = 42,
        n_bins: int = 5
    ):
        """
        Initialize GC stratified splitter.

        Args:
            n_bins: Number of GC content bins
        """
        super().__init__(train_size, val_size, test_size, random_state)
        self.n_bins = n_bins

    def _compute_gc_content(self, seq: str) -> float:
        """Compute GC content of sequence."""
        seq = seq.upper().replace('T', 'U')
        if len(seq) == 0:
            return 0.5
        gc_count = seq.count('G') + seq.count('C')
        return gc_count / len(seq)

    def split(
        self,
        df: pd.DataFrame,
        seq_col: str = 'seq_a'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split with stratification by GC content."""
        print(f"Stratifying by GC content ({seq_col})...")

        # Compute GC content
        gc_values = df[seq_col].apply(self._compute_gc_content).values

        print(f"  GC range: [{gc_values.min():.2%}, {gc_values.max():.2%}]")
        print(f"  Mean GC: {gc_values.mean():.2%}")

        # Create bins
        bins = np.percentile(gc_values, np.linspace(0, 100, self.n_bins + 1))
        bin_labels = np.digitize(gc_values, bins[1:-1])

        print(f"  Created {self.n_bins} GC bins")

        # Split within each bin
        train_idx, val_idx, test_idx = [], [], []

        for bin_id in range(self.n_bins):
            bin_mask = bin_labels == bin_id
            bin_indices = np.where(bin_mask)[0]

            if len(bin_indices) < 3:
                train_idx.extend(bin_indices)
                continue

            n_bin = len(bin_indices)
            n_train = int(n_bin * self.train_size)
            n_val = int(n_bin * self.val_size)

            np.random.seed(self.random_state + bin_id)
            np.random.shuffle(bin_indices)

            train_idx.extend(bin_indices[:n_train])
            val_idx.extend(bin_indices[n_train:n_train + n_val])
            test_idx.extend(bin_indices[n_train + n_val:])

        print(f"  Split sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

        return self._split_indices_to_dataframes(
            df,
            np.array(train_idx),
            np.array(val_idx),
            np.array(test_idx)
        )


class LengthStratifiedSplitter(RNASplitter):
    """
    Stratified split by sequence length.

    Ensures balanced length distribution, important when sequences
    have varying lengths (e.g., UTRs of different sizes).
    """

    def __init__(
        self,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
        random_state: Optional[int] = 42,
        n_bins: int = 5
    ):
        super().__init__(train_size, val_size, test_size, random_state)
        self.n_bins = n_bins

    def split(
        self,
        df: pd.DataFrame,
        seq_col: str = 'seq_a'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split with stratification by sequence length."""
        print(f"Stratifying by sequence length ({seq_col})...")

        lengths = df[seq_col].str.len().values

        print(f"  Length range: [{lengths.min()}, {lengths.max()}]")
        print(f"  Mean length: {lengths.mean():.1f}")

        # Create bins
        bins = np.percentile(lengths, np.linspace(0, 100, self.n_bins + 1))
        bin_labels = np.digitize(lengths, bins[1:-1])

        # Split within each bin
        train_idx, val_idx, test_idx = [], [], []

        for bin_id in range(self.n_bins):
            bin_mask = bin_labels == bin_id
            bin_indices = np.where(bin_mask)[0]

            if len(bin_indices) < 3:
                train_idx.extend(bin_indices)
                continue

            n_bin = len(bin_indices)
            n_train = int(n_bin * self.train_size)
            n_val = int(n_bin * self.val_size)

            np.random.seed(self.random_state + bin_id)
            np.random.shuffle(bin_indices)

            train_idx.extend(bin_indices[:n_train])
            val_idx.extend(bin_indices[n_train:n_train + n_val])
            test_idx.extend(bin_indices[n_train + n_val:])

        print(f"  Split sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

        return self._split_indices_to_dataframes(
            df,
            np.array(train_idx),
            np.array(val_idx),
            np.array(test_idx)
        )


class PositionSplitter(RNASplitter):
    """
    Split by edit position within the sequence.

    Tests whether the model learns position-invariant edit effects
    or memorizes position-specific patterns.

    Example: Train on positions 0-3 (5' end), test on positions 4-6 (3' end)
    """

    def __init__(
        self,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
        random_state: Optional[int] = 42,
        position_col: str = 'edit_positions_str',
        seq_col: str = 'loop_A',
        train_positions: Optional[List[int]] = None,
        test_positions: Optional[List[int]] = None,
        split_point: Optional[float] = 0.5  # Split at middle by default
    ):
        """
        Initialize position splitter.

        Args:
            position_col: Column containing edit position(s)
            seq_col: Column containing sequence (for length normalization)
            train_positions: Explicit list of positions for training
            test_positions: Explicit list of positions for testing
            split_point: If train/test_positions not specified, split at this
                        relative position (0.5 = middle)
        """
        super().__init__(train_size, val_size, test_size, random_state)
        self.position_col = position_col
        self.seq_col = seq_col
        self.train_positions = train_positions
        self.test_positions = test_positions
        self.split_point = split_point

    def _get_position(self, row) -> int:
        """Extract primary edit position from row."""
        pos_str = str(row[self.position_col])
        # Handle multiple positions (take first)
        if ',' in pos_str:
            pos_str = pos_str.split(',')[0]
        try:
            return int(pos_str)
        except ValueError:
            return -1

    def _get_relative_position(self, row) -> float:
        """Get position as fraction of sequence length."""
        pos = self._get_position(row)
        seq_len = len(row[self.seq_col]) if self.seq_col in row else 7
        if pos < 0 or seq_len == 0:
            return -1
        return pos / seq_len

    def split(
        self,
        df: pd.DataFrame,
        seq_col: str = 'seq_a'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split by edit position.

        Strategy:
        1. Extract edit position for each row
        2. Assign to train/test based on position criteria
        3. Split validation from training set
        """
        print(f"Splitting by edit position ({self.position_col})...")

        # Get positions
        positions = df.apply(self._get_position, axis=1).values

        # If explicit positions not given, use relative split
        if self.train_positions is None or self.test_positions is None:
            seq_lens = df[self.seq_col].str.len().values if self.seq_col in df.columns else np.full(len(df), 7)
            rel_positions = positions / np.maximum(seq_lens, 1)

            # Determine split point
            train_mask = rel_positions < self.split_point
            test_mask = rel_positions >= self.split_point
        else:
            train_mask = np.isin(positions, self.train_positions)
            test_mask = np.isin(positions, self.test_positions)

        # Handle invalid positions
        valid_mask = positions >= 0
        train_mask = train_mask & valid_mask
        test_mask = test_mask & valid_mask

        train_indices = np.where(train_mask)[0]
        test_indices = np.where(test_mask)[0]

        print(f"  Positions distribution:")
        print(f"    Train region: {train_mask.sum()} samples")
        print(f"    Test region: {test_mask.sum()} samples")

        # Split validation from training
        np.random.seed(self.random_state)
        np.random.shuffle(train_indices)

        n_train = len(train_indices)
        n_val = int(n_train * self.val_size / (self.train_size + self.val_size))

        val_indices = train_indices[:n_val]
        train_indices = train_indices[n_val:]

        print(f"  Final split: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")

        return self._split_indices_to_dataframes(
            df,
            train_indices,
            val_indices,
            test_indices
        )


class NeighborhoodContextSplitter(RNASplitter):
    """
    Split by sequence context around the edit site.

    Tests whether the model learns intrinsic edit effects or just
    memorizes context-specific patterns (e.g., "C-rich = good").

    Example: Train on A/U-rich contexts, test on G/C-rich contexts
    """

    def __init__(
        self,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
        random_state: Optional[int] = 42,
        upstream_col: str = 'context_3nt_upstream',
        downstream_col: str = 'context_3nt_downstream',
        split_by: str = 'gc_content',  # 'gc_content', 'kmer', or 'nucleotide'
        gc_threshold: float = 0.5,  # For gc_content mode
        train_contexts: Optional[List[str]] = None,  # For explicit context lists
        test_contexts: Optional[List[str]] = None
    ):
        """
        Initialize neighborhood context splitter.

        Args:
            upstream_col: Column with upstream context
            downstream_col: Column with downstream context
            split_by: How to split contexts
                - 'gc_content': Split by GC content of neighborhood
                - 'kmer': Split by specific k-mer patterns
                - 'nucleotide': Split by nucleotide at specific position
            gc_threshold: GC content threshold for splitting
            train_contexts: Explicit list of context patterns for training
            test_contexts: Explicit list of context patterns for testing
        """
        super().__init__(train_size, val_size, test_size, random_state)
        self.upstream_col = upstream_col
        self.downstream_col = downstream_col
        self.split_by = split_by
        self.gc_threshold = gc_threshold
        self.train_contexts = train_contexts
        self.test_contexts = test_contexts

    def _get_context_gc(self, row) -> float:
        """Compute GC content of neighborhood context."""
        upstream = str(row.get(self.upstream_col, '')).upper()
        downstream = str(row.get(self.downstream_col, '')).upper()
        context = upstream + downstream

        if len(context) == 0:
            return 0.5

        gc = sum(1 for c in context if c in 'GC')
        return gc / len(context)

    def _get_context_signature(self, row) -> str:
        """Get context signature for splitting."""
        upstream = str(row.get(self.upstream_col, '')).upper()
        downstream = str(row.get(self.downstream_col, '')).upper()
        return f"{upstream}_{downstream}"

    def split(
        self,
        df: pd.DataFrame,
        seq_col: str = 'seq_a'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split by neighborhood context.
        """
        print(f"Splitting by neighborhood context ({self.split_by})...")

        if self.split_by == 'gc_content':
            # Split by GC content of context
            gc_values = df.apply(self._get_context_gc, axis=1).values

            print(f"  Context GC range: [{gc_values.min():.2f}, {gc_values.max():.2f}]")
            print(f"  Mean GC: {gc_values.mean():.2f}")

            train_mask = gc_values < self.gc_threshold
            test_mask = gc_values >= self.gc_threshold

            print(f"  Low GC (train): {train_mask.sum()} samples")
            print(f"  High GC (test): {test_mask.sum()} samples")

        elif self.split_by == 'kmer':
            # Split by explicit k-mer patterns
            if self.train_contexts is None or self.test_contexts is None:
                raise ValueError("kmer split requires train_contexts and test_contexts")

            signatures = df.apply(self._get_context_signature, axis=1)

            train_mask = signatures.isin(self.train_contexts)
            test_mask = signatures.isin(self.test_contexts)

        else:
            raise ValueError(f"Unknown split_by: {self.split_by}")

        train_indices = np.where(train_mask)[0]
        test_indices = np.where(test_mask)[0]

        # Split validation from training
        np.random.seed(self.random_state)
        np.random.shuffle(train_indices)

        n_train = len(train_indices)
        n_val = int(n_train * self.val_size / (self.train_size + self.val_size))

        val_indices = train_indices[:n_val]
        train_indices = train_indices[n_val:]

        print(f"  Final split: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")

        return self._split_indices_to_dataframes(
            df,
            train_indices,
            val_indices,
            test_indices
        )


class ExperimentalContextSplitter(RNASplitter):
    """
    Split by experimental context/condition.

    Tests cross-context generalization: train on one experimental
    condition (e.g., A_vs_A), test on another (e.g., M_vs_M).

    This is the "cross-context" split used in the m6A experiments.
    """

    def __init__(
        self,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
        random_state: Optional[int] = 42,
        context_col: str = 'pair_type',
        train_contexts: List[str] = None,
        test_contexts: List[str] = None
    ):
        """
        Initialize experimental context splitter.

        Args:
            context_col: Column containing experimental context
            train_contexts: List of contexts for training (e.g., ['A_vs_A'])
            test_contexts: List of contexts for testing (e.g., ['M_vs_M'])
        """
        super().__init__(train_size, val_size, test_size, random_state)
        self.context_col = context_col
        self.train_contexts = train_contexts or ['A_vs_A']
        self.test_contexts = test_contexts or ['M_vs_M']

    def split(
        self,
        df: pd.DataFrame,
        seq_col: str = 'seq_a'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split by experimental context.
        """
        print(f"Splitting by experimental context ({self.context_col})...")

        contexts = df[self.context_col].values
        unique_contexts = np.unique(contexts)

        print(f"  Available contexts: {unique_contexts}")
        print(f"  Train contexts: {self.train_contexts}")
        print(f"  Test contexts: {self.test_contexts}")

        train_mask = np.isin(contexts, self.train_contexts)
        test_mask = np.isin(contexts, self.test_contexts)

        for ctx in unique_contexts:
            count = (contexts == ctx).sum()
            print(f"    {ctx}: {count} samples")

        train_indices = np.where(train_mask)[0]
        test_indices = np.where(test_mask)[0]

        # Split validation from training
        np.random.seed(self.random_state)
        np.random.shuffle(train_indices)

        n_train = len(train_indices)
        n_val = int(n_train * self.val_size / (self.train_size + self.val_size))

        val_indices = train_indices[:n_val]
        train_indices = train_indices[n_val:]

        print(f"  Final split: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")

        return self._split_indices_to_dataframes(
            df,
            train_indices,
            val_indices,
            test_indices
        )


class EffectMagnitudeSplitter(RNASplitter):
    """
    Split by magnitude of effect (target value).

    Tests whether the model can extrapolate to effects larger than
    those seen during training.

    Example: Train on small effects, test on large effects
    """

    def __init__(
        self,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
        random_state: Optional[int] = 42,
        target_col: str = 'delta_intensity_median',
        train_percentile: float = 75,  # Train on bottom 75%
        test_percentile: float = 75,   # Test on top 25%
        absolute: bool = True  # Use absolute value
    ):
        """
        Initialize effect magnitude splitter.

        Args:
            target_col: Column containing effect values
            train_percentile: Train on effects below this percentile
            test_percentile: Test on effects above this percentile
            absolute: If True, use absolute value of effect
        """
        super().__init__(train_size, val_size, test_size, random_state)
        self.target_col = target_col
        self.train_percentile = train_percentile
        self.test_percentile = test_percentile
        self.absolute = absolute

    def split(
        self,
        df: pd.DataFrame,
        seq_col: str = 'seq_a'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split by effect magnitude.
        """
        print(f"Splitting by effect magnitude ({self.target_col})...")

        values = df[self.target_col].values
        if self.absolute:
            magnitudes = np.abs(values)
        else:
            magnitudes = values

        train_threshold = np.percentile(magnitudes, self.train_percentile)
        test_threshold = np.percentile(magnitudes, self.test_percentile)

        print(f"  Effect range: [{magnitudes.min():.2f}, {magnitudes.max():.2f}]")
        print(f"  Train threshold (p{self.train_percentile}): {train_threshold:.2f}")
        print(f"  Test threshold (p{self.test_percentile}): {test_threshold:.2f}")

        train_mask = magnitudes <= train_threshold
        test_mask = magnitudes > test_threshold

        print(f"  Small effects (train): {train_mask.sum()} samples")
        print(f"  Large effects (test): {test_mask.sum()} samples")

        train_indices = np.where(train_mask)[0]
        test_indices = np.where(test_mask)[0]

        # Split validation from training
        np.random.seed(self.random_state)
        np.random.shuffle(train_indices)

        n_train = len(train_indices)
        n_val = int(n_train * self.val_size / (self.train_size + self.val_size))

        val_indices = train_indices[:n_val]
        train_indices = train_indices[n_val:]

        print(f"  Final split: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")

        return self._split_indices_to_dataframes(
            df,
            train_indices,
            val_indices,
            test_indices
        )


class NucleotideChangeSplitter(RNASplitter):
    """
    Split by type of nucleotide change.

    Tests whether the model learns transferable edit effects across
    different mutation types.

    Example: Train on A→C edits, test on A→G edits
    """

    def __init__(
        self,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
        random_state: Optional[int] = 42,
        edit_col: str = 'edit_description',
        train_changes: Optional[List[str]] = None,  # e.g., ['A>C', 'A>G']
        test_changes: Optional[List[str]] = None,   # e.g., ['A>U']
        group_by: str = 'exact'  # 'exact', 'transition_transversion', 'from_nucleotide'
    ):
        """
        Initialize nucleotide change splitter.

        Args:
            edit_col: Column containing edit description (e.g., "0:A>C")
            train_changes: List of nucleotide changes for training
            test_changes: List of nucleotide changes for testing
            group_by: How to group changes
                - 'exact': Use exact change (A>C, A>G, etc.)
                - 'transition_transversion': Group by type
                - 'from_nucleotide': Group by original nucleotide
        """
        super().__init__(train_size, val_size, test_size, random_state)
        self.edit_col = edit_col
        self.train_changes = train_changes
        self.test_changes = test_changes
        self.group_by = group_by

    def _extract_change(self, edit_desc: str) -> str:
        """Extract nucleotide change from edit description."""
        # Format: "0:A>C" or just "A>C"
        edit_desc = str(edit_desc)
        if ':' in edit_desc:
            edit_desc = edit_desc.split(':')[1]
        return edit_desc.upper()

    def _classify_change(self, change: str) -> str:
        """Classify the nucleotide change."""
        if '>' not in change:
            return 'unknown'

        parts = change.split('>')
        if len(parts) != 2:
            return 'unknown'

        from_nuc, to_nuc = parts[0][-1], parts[1][0]  # Handle multi-char

        if self.group_by == 'exact':
            return f"{from_nuc}>{to_nuc}"
        elif self.group_by == 'from_nucleotide':
            return from_nuc
        elif self.group_by == 'transition_transversion':
            purines = {'A', 'G'}
            pyrimidines = {'C', 'U', 'T'}
            if (from_nuc in purines and to_nuc in purines) or \
               (from_nuc in pyrimidines and to_nuc in pyrimidines):
                return 'transition'
            else:
                return 'transversion'
        else:
            return f"{from_nuc}>{to_nuc}"

    def split(
        self,
        df: pd.DataFrame,
        seq_col: str = 'seq_a'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split by nucleotide change type.
        """
        print(f"Splitting by nucleotide change ({self.group_by})...")

        # Extract and classify changes
        changes = df[self.edit_col].apply(self._extract_change)
        classified = changes.apply(self._classify_change)

        # Show distribution
        change_counts = classified.value_counts()
        print(f"  Change distribution:")
        for change, count in change_counts.items():
            print(f"    {change}: {count}")

        # Determine train/test split
        if self.train_changes is None or self.test_changes is None:
            # Auto-split: most common to train, least common to test
            sorted_changes = change_counts.index.tolist()
            n_train = max(1, int(len(sorted_changes) * 0.7))
            self.train_changes = sorted_changes[:n_train]
            self.test_changes = sorted_changes[n_train:]

        print(f"  Train changes: {self.train_changes}")
        print(f"  Test changes: {self.test_changes}")

        train_mask = classified.isin(self.train_changes).values
        test_mask = classified.isin(self.test_changes).values

        train_indices = np.where(train_mask)[0]
        test_indices = np.where(test_mask)[0]

        # Split validation from training
        np.random.seed(self.random_state)
        np.random.shuffle(train_indices)

        n_train = len(train_indices)
        n_val = int(n_train * self.val_size / (self.train_size + self.val_size))

        val_indices = train_indices[:n_val]
        train_indices = train_indices[n_val:]

        print(f"  Final split: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")

        return self._split_indices_to_dataframes(
            df,
            train_indices,
            val_indices,
            test_indices
        )


class GeneSplitter(RNASplitter):
    """
    Split by holding out entire genes for testing.

    Tests whether the model generalizes to unseen genes/transcripts.
    This is crucial for RNA modification prediction where nearby sites
    within a gene may share confounding features.

    Applicable to any RNA task with genomic position information.

    Example: Train on genes A, B, C; test on genes D, E
    """

    def __init__(
        self,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
        random_state: Optional[int] = 42,
        gene_col: str = 'gene_id',
        chrom_col: str = 'chrom',
        position_col: str = 'position',
        strand_col: str = 'strand',
        gtf_file: Optional[str] = None,
        min_samples_per_gene: int = 5
    ):
        """
        Initialize gene splitter.

        Args:
            gene_col: Column containing gene ID (if available)
            chrom_col: Column with chromosome name
            position_col: Column with genomic position
            strand_col: Column with strand (+/-)
            gtf_file: Path to GTF file for gene mapping (if gene_col not in data)
            min_samples_per_gene: Minimum samples per gene to include
        """
        super().__init__(train_size, val_size, test_size, random_state)
        self.gene_col = gene_col
        self.chrom_col = chrom_col
        self.position_col = position_col
        self.strand_col = strand_col
        self.gtf_file = gtf_file
        self.min_samples_per_gene = min_samples_per_gene

    def _load_gene_annotations(self, gtf_file: str) -> pd.DataFrame:
        """Load gene boundaries from GTF file."""
        genes = []
        with open(gtf_file) as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split('\t')
                if len(parts) >= 9 and parts[2] in ('gene', 'transcript'):
                    chrom = parts[0].replace('chr', '')
                    start = int(parts[3])
                    end = int(parts[4])
                    strand = parts[6]
                    attrs = parts[8]
                    gene_id = None
                    for attr in attrs.split(';'):
                        attr = attr.strip()
                        if attr.startswith('gene_id'):
                            gene_id = attr.split('"')[1]
                            break
                    if gene_id:
                        genes.append({
                            'chrom': chrom, 'start': start, 'end': end,
                            'strand': strand, 'gene_id': gene_id
                        })

        genes_df = pd.DataFrame(genes)
        if len(genes_df) > 0:
            # Aggregate by gene (merge overlapping transcripts)
            genes_df = genes_df.groupby(['chrom', 'gene_id', 'strand']).agg({
                'start': 'min', 'end': 'max'
            }).reset_index()
        return genes_df

    def _map_positions_to_genes(
        self, df: pd.DataFrame, gene_bounds: pd.DataFrame
    ) -> pd.Series:
        """Map genomic positions to gene IDs."""
        gene_ids = []
        for _, row in df.iterrows():
            chrom = str(row[self.chrom_col])
            pos = row[self.position_col]
            strand = row.get(self.strand_col, '+')

            mask = (gene_bounds['chrom'] == chrom)
            if self.strand_col in df.columns:
                mask = mask & (gene_bounds['strand'] == strand)
            cand = gene_bounds[mask]
            hits = cand[(cand['start'] <= pos) & (cand['end'] >= pos)]

            if len(hits) > 0:
                gene_ids.append(hits.iloc[0]['gene_id'])
            else:
                # Intergenic: group by 10kb region
                gene_ids.append(f"intergenic_{chrom}_{pos // 10000}")

        return pd.Series(gene_ids, index=df.index)

    def split(
        self,
        df: pd.DataFrame,
        seq_col: str = 'seq_a'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split by holding out genes.
        """
        print(f"Splitting by gene holdout...")

        # Get gene IDs
        if self.gene_col in df.columns:
            gene_ids = df[self.gene_col]
            print(f"  Using existing gene column: {self.gene_col}")
        elif self.gtf_file is not None:
            print(f"  Mapping positions to genes from GTF: {self.gtf_file}")
            gene_bounds = self._load_gene_annotations(self.gtf_file)
            print(f"  Loaded {len(gene_bounds)} genes from GTF")
            gene_ids = self._map_positions_to_genes(df, gene_bounds)
        else:
            # Fall back to chromosome-based grouping
            print(f"  No gene info available, falling back to chromosome split")
            gene_ids = df[self.chrom_col].astype(str)

        # Get unique genes and their sample counts
        gene_counts = gene_ids.value_counts()
        valid_genes = gene_counts[gene_counts >= self.min_samples_per_gene].index.tolist()

        print(f"  Total unique genes: {len(gene_counts)}")
        print(f"  Genes with >= {self.min_samples_per_gene} samples: {len(valid_genes)}")

        # Shuffle and split genes
        np.random.seed(self.random_state)
        np.random.shuffle(valid_genes)

        n_test = max(1, int(len(valid_genes) * self.test_size))
        n_val = max(1, int(len(valid_genes) * self.val_size))

        test_genes = set(valid_genes[:n_test])
        val_genes = set(valid_genes[n_test:n_test + n_val])
        train_genes = set(valid_genes[n_test + n_val:])

        # Assign samples
        train_mask = gene_ids.isin(train_genes)
        val_mask = gene_ids.isin(val_genes)
        test_mask = gene_ids.isin(test_genes)

        train_indices = np.where(train_mask)[0]
        val_indices = np.where(val_mask)[0]
        test_indices = np.where(test_mask)[0]

        print(f"  Gene split: {len(train_genes)} train, {len(val_genes)} val, {len(test_genes)} test genes")
        print(f"  Sample split: {len(train_indices)} train, {len(val_indices)} val, {len(test_indices)} test")

        return self._split_indices_to_dataframes(df, train_indices, val_indices, test_indices)


class TrinucleotideContextSplitter(RNASplitter):
    """
    Split by trinucleotide context around the target site.

    Tests whether the model learns context-independent modification rules
    or just memorizes context-specific patterns.

    Widely applicable to:
    - RNA editing (ADAR: A-to-I in UAG context)
    - RNA modification (m6A: DRACH motif)
    - Any nucleotide-level prediction task

    Example: Train on non-UAG contexts, test on UAG (hardest for ADAR)
    """

    def __init__(
        self,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
        random_state: Optional[int] = 42,
        seq_col: str = 'sequence',
        center_idx: Optional[int] = None,
        test_contexts: Optional[List[str]] = None,
        train_contexts: Optional[List[str]] = None,
        context_col: Optional[str] = None,
        left_feature_prefix: str = 'left_1_',
        right_feature_prefix: str = 'right_1_'
    ):
        """
        Initialize trinucleotide context splitter.

        Args:
            seq_col: Column containing sequence (for extracting context)
            center_idx: Index of center position in sequence (None = middle)
            test_contexts: List of contexts to hold out for testing (e.g., ['UAG'])
            train_contexts: List of contexts for training (None = all except test)
            context_col: Column with pre-computed context (optional)
            left_feature_prefix: Prefix for left neighbor one-hot features
            right_feature_prefix: Prefix for right neighbor one-hot features
        """
        super().__init__(train_size, val_size, test_size, random_state)
        self.seq_col = seq_col
        self.center_idx = center_idx
        self.test_contexts = test_contexts or ['UAG']
        self.train_contexts = train_contexts
        self.context_col = context_col
        self.left_feature_prefix = left_feature_prefix
        self.right_feature_prefix = right_feature_prefix

    def _extract_context_from_sequence(self, seq: str, center_idx: int) -> str:
        """Extract XNY trinucleotide context from sequence."""
        seq = seq.upper().replace('T', 'U')
        if center_idx < 1 or center_idx >= len(seq) - 1:
            return 'NNN'
        return seq[center_idx - 1:center_idx + 2]

    def _extract_context_from_features(self, row: pd.Series) -> str:
        """Extract context from one-hot encoded neighbor features."""
        left_nuc = 'N'
        right_nuc = 'N'
        center_nuc = 'A'  # Default for editing prediction

        for nuc in ['A', 'C', 'G', 'U']:
            left_col = f'{self.left_feature_prefix}{nuc}'
            right_col = f'{self.right_feature_prefix}{nuc}'
            if left_col in row and row[left_col] == 1:
                left_nuc = nuc
            if right_col in row and row[right_col] == 1:
                right_nuc = nuc

        return f"{left_nuc}{center_nuc}{right_nuc}"

    def split(
        self,
        df: pd.DataFrame,
        seq_col: str = 'seq_a'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split by trinucleotide context.
        """
        print(f"Splitting by trinucleotide context...")
        print(f"  Test contexts: {self.test_contexts}")

        # Extract contexts
        if self.context_col and self.context_col in df.columns:
            contexts = df[self.context_col]
            print(f"  Using pre-computed context column: {self.context_col}")
        elif self.seq_col in df.columns:
            center = self.center_idx or len(df[self.seq_col].iloc[0]) // 2
            contexts = df[self.seq_col].apply(
                lambda s: self._extract_context_from_sequence(s, center)
            )
            print(f"  Extracted context from sequence (center={center})")
        else:
            # Try to extract from one-hot features
            contexts = df.apply(self._extract_context_from_features, axis=1)
            print(f"  Extracted context from one-hot features")

        # Show distribution
        context_counts = contexts.value_counts()
        print(f"  Context distribution (top 10):")
        for ctx, count in context_counts.head(10).items():
            marker = " <-- TEST" if ctx in self.test_contexts else ""
            print(f"    {ctx}: {count} ({count/len(df)*100:.1f}%){marker}")

        # Split by context
        test_mask = contexts.isin(self.test_contexts)

        if self.train_contexts is not None:
            train_val_mask = contexts.isin(self.train_contexts)
        else:
            train_val_mask = ~test_mask

        test_indices = np.where(test_mask)[0]
        train_val_indices = np.where(train_val_mask)[0]

        # Split train/val
        np.random.seed(self.random_state)
        np.random.shuffle(train_val_indices)

        n_val = int(len(train_val_indices) * self.val_size / (self.train_size + self.val_size))
        val_indices = train_val_indices[:n_val]
        train_indices = train_val_indices[n_val:]

        print(f"  Final split: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")

        return self._split_indices_to_dataframes(df, train_indices, val_indices, test_indices)


class CoverageExtrapolationSplitter(RNASplitter):
    """
    Split by sequencing coverage to test extrapolation.

    Tests whether the model generalizes across coverage regimes
    or learns coverage as a spurious feature.

    Applicable to any RNA-seq based prediction where coverage
    correlates with positive labels (common bias!).

    Example: Train on low/medium coverage, test on high coverage
    """

    def __init__(
        self,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
        random_state: Optional[int] = 42,
        coverage_col: str = 'coverage',
        direction: str = 'low_to_high',
        test_percentile: float = 75
    ):
        """
        Initialize coverage extrapolation splitter.

        Args:
            coverage_col: Column containing coverage values
            direction: Extrapolation direction
                - 'low_to_high': Train on low/medium, test on high (most common)
                - 'high_to_low': Train on medium/high, test on low
            test_percentile: Percentile threshold for test set
                - For 'low_to_high': test on samples above this percentile
                - For 'high_to_low': test on samples below (100 - percentile)
        """
        super().__init__(train_size, val_size, test_size, random_state)
        self.coverage_col = coverage_col
        self.direction = direction
        self.test_percentile = test_percentile

    def split(
        self,
        df: pd.DataFrame,
        seq_col: str = 'seq_a'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split by coverage for extrapolation testing.
        """
        print(f"Splitting by coverage extrapolation ({self.direction})...")

        coverage = df[self.coverage_col].values

        print(f"  Coverage range: {coverage.min():.0f} - {coverage.max():.0f}")
        print(f"  Coverage mean: {coverage.mean():.1f}, median: {np.median(coverage):.1f}")

        if self.direction == 'low_to_high':
            threshold = np.percentile(coverage, self.test_percentile)
            test_mask = coverage >= threshold
            print(f"  Test: coverage >= {threshold:.0f} (top {100-self.test_percentile:.0f}%)")
        else:  # high_to_low
            threshold = np.percentile(coverage, 100 - self.test_percentile)
            test_mask = coverage <= threshold
            print(f"  Test: coverage <= {threshold:.0f} (bottom {100-self.test_percentile:.0f}%)")

        train_val_mask = ~test_mask

        test_indices = np.where(test_mask)[0]
        train_val_indices = np.where(train_val_mask)[0]

        # Split train/val
        np.random.seed(self.random_state)
        np.random.shuffle(train_val_indices)

        n_val = int(len(train_val_indices) * self.val_size / (self.train_size + self.val_size))
        val_indices = train_val_indices[:n_val]
        train_indices = train_val_indices[n_val:]

        # Report coverage stats per split
        print(f"  Train coverage: mean={coverage[train_indices].mean():.1f}")
        print(f"  Test coverage: mean={coverage[test_indices].mean():.1f}")
        print(f"  Final split: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")

        return self._split_indices_to_dataframes(df, train_indices, val_indices, test_indices)


class StructureStateSplitter(RNASplitter):
    """
    Split by RNA secondary structure state at the target site.

    Tests whether the model learns structure-independent rules
    or relies on structural context.

    Applicable to RNA modification/editing where structure matters:
    - ADAR editing requires dsRNA
    - m6A often in unstructured regions
    - RNA-protein binding depends on accessibility

    Example: Train on paired sites, test on unpaired (or vice versa)
    """

    def __init__(
        self,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
        random_state: Optional[int] = 42,
        structure_col: str = 'structure',
        paired_col: Optional[str] = 'center_is_paired',
        center_idx: Optional[int] = None,
        direction: str = 'paired_to_unpaired'
    ):
        """
        Initialize structure state splitter.

        Args:
            structure_col: Column with dot-bracket structure
            paired_col: Column with pre-computed paired state (0/1)
            center_idx: Index of center position (None = middle)
            direction: Split direction
                - 'paired_to_unpaired': Train on paired, test on unpaired
                - 'unpaired_to_paired': Train on unpaired, test on paired
        """
        super().__init__(train_size, val_size, test_size, random_state)
        self.structure_col = structure_col
        self.paired_col = paired_col
        self.center_idx = center_idx
        self.direction = direction

    def _is_paired(self, structure: str, center_idx: int) -> bool:
        """Check if center position is paired."""
        if center_idx < 0 or center_idx >= len(structure):
            return False
        return structure[center_idx] in '()'

    def split(
        self,
        df: pd.DataFrame,
        seq_col: str = 'seq_a'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split by structure state.
        """
        print(f"Splitting by structure state ({self.direction})...")

        # Get paired state
        if self.paired_col and self.paired_col in df.columns:
            is_paired = df[self.paired_col].values == 1
            print(f"  Using pre-computed paired column: {self.paired_col}")
        elif self.structure_col in df.columns:
            center = self.center_idx or len(df[self.structure_col].iloc[0]) // 2
            is_paired = df[self.structure_col].apply(
                lambda s: self._is_paired(s, center)
            ).values
            print(f"  Computed paired state from structure (center={center})")
        else:
            raise ValueError(f"Need either {self.paired_col} or {self.structure_col} column")

        n_paired = is_paired.sum()
        n_unpaired = (~is_paired).sum()
        print(f"  Paired: {n_paired} ({n_paired/len(df)*100:.1f}%)")
        print(f"  Unpaired: {n_unpaired} ({n_unpaired/len(df)*100:.1f}%)")

        if self.direction == 'paired_to_unpaired':
            train_val_mask = is_paired
            test_mask = ~is_paired
        else:  # unpaired_to_paired
            train_val_mask = ~is_paired
            test_mask = is_paired

        test_indices = np.where(test_mask)[0]
        train_val_indices = np.where(train_val_mask)[0]

        # Split train/val
        np.random.seed(self.random_state)
        np.random.shuffle(train_val_indices)

        n_val = int(len(train_val_indices) * self.val_size / (self.train_size + self.val_size))
        val_indices = train_val_indices[:n_val]
        train_indices = train_val_indices[n_val:]

        print(f"  Final split: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")

        return self._split_indices_to_dataframes(df, train_indices, val_indices, test_indices)


def get_rna_splitter(
    split_type: str,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: Optional[int] = 42,
    **kwargs
) -> RNASplitter:
    """
    Factory function to get an RNA splitter by name.

    Args:
        split_type: One of:
            Basic splits:
            - 'random': Random baseline split
            - 'sequence_similarity': K-mer based sequence clustering
            - 'motif': Split by regulatory motif presence
            - 'edit_type': Split by SNV/insertion/deletion type
            - 'gc_stratified': Stratified by GC content
            - 'length_stratified': Stratified by sequence length

            Generalization splits (for testing model generalization):
            - 'position': Split by edit position (5' vs 3')
            - 'neighborhood_context': Split by context around edit (GC-rich vs AT-rich)
            - 'experimental_context': Split by experimental condition (A_vs_A vs M_vs_M)
            - 'effect_magnitude': Split by effect size (small vs large effects)
            - 'nucleotide_change': Split by mutation type (A>C vs A>G, etc.)

            Stress-test splits (for exposing model shortcuts):
            - 'gene': Hold out entire genes (cross-gene generalization)
            - 'trinucleotide_context': Hold out sequence contexts (e.g., UAG for ADAR)
            - 'coverage_extrapolation': Train on low coverage, test on high (or vice versa)
            - 'structure_state': Train on paired sites, test on unpaired (or vice versa)

        train_size, val_size, test_size: Split fractions
        random_state: Random seed
        **kwargs: Additional arguments for specific splitters
            - sequence_similarity: kmer_size, similarity_threshold
            - motif: motif_types
            - edit_type: position_bins
            - gc_stratified: n_bins
            - length_stratified: n_bins
            - position: position_col, seq_col, train_positions, test_positions, split_point
            - neighborhood_context: upstream_col, downstream_col, split_by, gc_threshold
            - experimental_context: context_col, train_contexts, test_contexts
            - effect_magnitude: target_col, train_percentile, test_percentile, absolute
            - nucleotide_change: edit_col, train_changes, test_changes, group_by
            - gene: gene_col, chrom_col, position_col, strand_col, gtf_file
            - trinucleotide_context: seq_col, center_idx, test_contexts, context_col
            - coverage_extrapolation: coverage_col, direction, test_percentile
            - structure_state: structure_col, paired_col, center_idx, direction

    Returns:
        RNASplitter instance

    Example:
        >>> # Basic splits
        >>> splitter = get_rna_splitter('sequence_similarity', kmer_size=5)
        >>> train, val, test = splitter.split(df, seq_col='seq_a')

        >>> # Generalization splits
        >>> splitter = get_rna_splitter('position', train_positions=[0,1,2], test_positions=[4,5,6])
        >>> train, val, test = splitter.split(df)

        >>> # Stress-test splits
        >>> splitter = get_rna_splitter('gene', gtf_file='genes.gtf')
        >>> train, val, test = splitter.split(df)

        >>> splitter = get_rna_splitter('trinucleotide_context', test_contexts=['UAG', 'AAG'])
        >>> train, val, test = splitter.split(df)

        >>> splitter = get_rna_splitter('coverage_extrapolation', direction='low_to_high')
        >>> train, val, test = splitter.split(df)
    """
    splitters = {
        # Basic splits
        'random': RandomRNASplitter,
        'sequence_similarity': SequenceSimilaritySplitter,
        'motif': MotifSplitter,
        'edit_type': EditTypeSplitter,
        'gc_stratified': GCStratifiedSplitter,
        'length_stratified': LengthStratifiedSplitter,
        # Generalization splits
        'position': PositionSplitter,
        'neighborhood_context': NeighborhoodContextSplitter,
        'experimental_context': ExperimentalContextSplitter,
        'effect_magnitude': EffectMagnitudeSplitter,
        'nucleotide_change': NucleotideChangeSplitter,
        # Stress-test splits (NEW)
        'gene': GeneSplitter,
        'trinucleotide_context': TrinucleotideContextSplitter,
        'coverage_extrapolation': CoverageExtrapolationSplitter,
        'structure_state': StructureStateSplitter,
    }

    if split_type not in splitters:
        raise ValueError(
            f"Unknown split_type '{split_type}'. "
            f"Available: {list(splitters.keys())}"
        )

    splitter_class = splitters[split_type]
    return splitter_class(
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
        **kwargs
    )


class GeneralizationBenchmark:
    """
    Benchmark for systematically testing model generalization across multiple splits.

    This class runs a model through all generalization splits and reports performance
    on each, making it easy to identify where a model generalizes well vs. poorly.

    Example:
        >>> benchmark = GeneralizationBenchmark()
        >>> results = benchmark.run(df, train_fn, eval_fn)
        >>> benchmark.print_summary(results)
    """

    # Default split configurations for RNA modification/editing prediction
    DEFAULT_SPLITS = {
        # Baseline
        'random': {
            'description': 'Random baseline (no distribution shift)',
            'kwargs': {}
        },

        # Position-based splits
        'position_5prime_to_3prime': {
            'description': 'Train on 5\' positions, test on 3\' positions',
            'split_type': 'position',
            'kwargs': {'split_point': 0.5}
        },
        'position_explicit': {
            'description': 'Train on pos 0-2, test on pos 4-6',
            'split_type': 'position',
            'kwargs': {'train_positions': [0, 1, 2], 'test_positions': [4, 5, 6]}
        },

        # Neighborhood context splits
        'context_gc': {
            'description': 'Train on AU-rich context, test on GC-rich context',
            'split_type': 'neighborhood_context',
            'kwargs': {'split_by': 'gc_content', 'gc_threshold': 0.5}
        },

        # Experimental context splits
        'experimental_A_to_M': {
            'description': 'Train on A_vs_A, test on M_vs_M (cross-context)',
            'split_type': 'experimental_context',
            'kwargs': {'train_contexts': ['A_vs_A'], 'test_contexts': ['M_vs_M']}
        },
        'experimental_M_to_A': {
            'description': 'Train on M_vs_M, test on A_vs_A (cross-context)',
            'split_type': 'experimental_context',
            'kwargs': {'train_contexts': ['M_vs_M'], 'test_contexts': ['A_vs_A']}
        },

        # Effect magnitude splits
        'effect_small_to_large': {
            'description': 'Train on small effects, test on large effects',
            'split_type': 'effect_magnitude',
            'kwargs': {'train_percentile': 75, 'test_percentile': 75}
        },

        # Nucleotide change splits
        'nucleotide_holdout': {
            'description': 'Train on most changes, test on held-out change',
            'split_type': 'nucleotide_change',
            'kwargs': {'group_by': 'exact'}
        },
        'transition_to_transversion': {
            'description': 'Train on transitions, test on transversions',
            'split_type': 'nucleotide_change',
            'kwargs': {
                'group_by': 'transition_transversion',
                'train_changes': ['transition'],
                'test_changes': ['transversion']
            }
        },
        'transversion_to_transition': {
            'description': 'Train on transversions, test on transitions',
            'split_type': 'nucleotide_change',
            'kwargs': {
                'group_by': 'transition_transversion',
                'train_changes': ['transversion'],
                'test_changes': ['transition']
            }
        },

        # === NEW STRESS-TEST SPLITS ===

        # Gene-level generalization
        'gene_holdout': {
            'description': 'Hold out entire genes for testing',
            'split_type': 'gene',
            'kwargs': {'min_samples_per_gene': 5}
        },

        # Trinucleotide context holdout (for ADAR, m6A, etc.)
        'context_holdout_preferred': {
            'description': 'Hold out preferred motif context (e.g., UAG for ADAR)',
            'split_type': 'trinucleotide_context',
            'kwargs': {'test_contexts': ['UAG']}
        },
        'context_holdout_multiple': {
            'description': 'Hold out multiple preferred contexts',
            'split_type': 'trinucleotide_context',
            'kwargs': {'test_contexts': ['UAG', 'AAG']}
        },

        # Coverage extrapolation (common bias!)
        'coverage_low_to_high': {
            'description': 'Train on low/medium coverage, test on high',
            'split_type': 'coverage_extrapolation',
            'kwargs': {'direction': 'low_to_high', 'test_percentile': 75}
        },
        'coverage_high_to_low': {
            'description': 'Train on medium/high coverage, test on low',
            'split_type': 'coverage_extrapolation',
            'kwargs': {'direction': 'high_to_low', 'test_percentile': 75}
        },

        # Structure state splits
        'structure_paired_to_unpaired': {
            'description': 'Train on paired sites, test on unpaired',
            'split_type': 'structure_state',
            'kwargs': {'direction': 'paired_to_unpaired'}
        },
        'structure_unpaired_to_paired': {
            'description': 'Train on unpaired sites, test on paired',
            'split_type': 'structure_state',
            'kwargs': {'direction': 'unpaired_to_paired'}
        },
    }

    def __init__(
        self,
        splits: Optional[Dict] = None,
        random_state: int = 42
    ):
        """
        Initialize benchmark.

        Args:
            splits: Dictionary of split configurations. If None, uses DEFAULT_SPLITS.
            random_state: Random seed for reproducibility
        """
        self.splits = splits or self.DEFAULT_SPLITS
        self.random_state = random_state

    def get_split(self, split_name: str, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Get train/val/test split for a specific split configuration.

        Args:
            split_name: Name of the split configuration
            df: DataFrame to split

        Returns:
            train_df, val_df, test_df
        """
        if split_name not in self.splits:
            raise ValueError(f"Unknown split: {split_name}. Available: {list(self.splits.keys())}")

        config = self.splits[split_name]
        split_type = config.get('split_type', split_name)
        kwargs = config.get('kwargs', {})

        splitter = get_rna_splitter(
            split_type,
            random_state=self.random_state,
            **kwargs
        )

        return splitter.split(df)

    def run(
        self,
        df: pd.DataFrame,
        train_fn,
        eval_fn,
        splits_to_run: Optional[List[str]] = None,
        verbose: bool = True
    ) -> Dict[str, Dict]:
        """
        Run benchmark across all splits.

        Args:
            df: Full dataset DataFrame
            train_fn: Function(train_df, val_df) -> model
            eval_fn: Function(model, test_df) -> dict of metrics
            splits_to_run: List of split names to run (None = all)
            verbose: Print progress

        Returns:
            Dictionary mapping split_name -> {
                'description': str,
                'train_size': int,
                'val_size': int,
                'test_size': int,
                'metrics': dict
            }
        """
        results = {}
        splits_to_run = splits_to_run or list(self.splits.keys())

        for split_name in splits_to_run:
            if verbose:
                print(f"\n{'='*70}")
                print(f"Running split: {split_name}")
                print(f"{'='*70}")

            config = self.splits[split_name]

            try:
                train_df, val_df, test_df = self.get_split(split_name, df)

                if len(train_df) == 0 or len(test_df) == 0:
                    print(f"  Skipping: empty train or test set")
                    continue

                # Train model
                model = train_fn(train_df, val_df)

                # Evaluate
                metrics = eval_fn(model, test_df)

                results[split_name] = {
                    'description': config.get('description', split_name),
                    'train_size': len(train_df),
                    'val_size': len(val_df),
                    'test_size': len(test_df),
                    'metrics': metrics
                }

                if verbose:
                    print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
                    for metric, value in metrics.items():
                        print(f"  {metric}: {value:.4f}")

            except Exception as e:
                print(f"  Error: {e}")
                results[split_name] = {
                    'description': config.get('description', split_name),
                    'error': str(e)
                }

        return results

    def print_summary(
        self,
        results: Dict[str, Dict],
        metric: str = 'r2',
        sort_by_metric: bool = True
    ):
        """
        Print summary table of benchmark results.

        Args:
            results: Results from run()
            metric: Primary metric to display
            sort_by_metric: Sort by metric value (descending)
        """
        print("\n" + "=" * 80)
        print("GENERALIZATION BENCHMARK SUMMARY")
        print("=" * 80)

        rows = []
        for split_name, result in results.items():
            if 'error' in result:
                rows.append({
                    'split': split_name,
                    'description': result['description'][:40],
                    'train': '-',
                    'test': '-',
                    metric: 'ERROR'
                })
            else:
                metric_val = result['metrics'].get(metric, np.nan)
                rows.append({
                    'split': split_name,
                    'description': result['description'][:40],
                    'train': result['train_size'],
                    'test': result['test_size'],
                    metric: metric_val
                })

        # Sort if requested
        if sort_by_metric:
            rows = sorted(rows, key=lambda x: x[metric] if isinstance(x[metric], (int, float)) else -999, reverse=True)

        # Print table
        print(f"\n{'Split':<30} {'Description':<42} {'Train':>8} {'Test':>8} {metric:>10}")
        print("-" * 100)

        for row in rows:
            metric_str = f"{row[metric]:.4f}" if isinstance(row[metric], (int, float)) else row[metric]
            print(f"{row['split']:<30} {row['description']:<42} {row['train']:>8} {row['test']:>8} {metric_str:>10}")

        print("=" * 80)

    def to_dataframe(self, results: Dict[str, Dict]) -> pd.DataFrame:
        """Convert results to a DataFrame for further analysis."""
        rows = []
        for split_name, result in results.items():
            row = {
                'split': split_name,
                'description': result.get('description', ''),
                'train_size': result.get('train_size', 0),
                'val_size': result.get('val_size', 0),
                'test_size': result.get('test_size', 0),
            }
            if 'metrics' in result:
                row.update(result['metrics'])
            if 'error' in result:
                row['error'] = result['error']
            rows.append(row)

        return pd.DataFrame(rows)
