"""
Simple baseline RNA embedder using nucleotide features.

Provides interpretable features without requiring pretrained models:
- One-hot encoding per position (padded/pooled)
- K-mer frequency vectors
- GC content and other sequence statistics
- Optional: secondary structure features (requires ViennaRNA)
"""

import numpy as np
from typing import Union, List, Optional
from collections import Counter
from itertools import product
from .base import RNAEmbedder


class NucleotideEmbedder(RNAEmbedder):
    """
    Simple baseline RNA embedder using nucleotide-level features.

    This embedder creates fixed-dimensional representations by combining:
    1. Positional one-hot encoding (mean-pooled across sequence)
    2. K-mer frequency vectors (3-mers, 4-mers, etc.)
    3. Global sequence statistics (GC content, length, etc.)
    4. Optional: secondary structure features

    Advantages:
    - No GPU required
    - Fast computation
    - Interpretable features
    - Works on any sequence length

    Args:
        include_onehot: Include mean-pooled one-hot encoding (4 features)
        include_kmers: Include k-mer frequency vectors
        kmer_sizes: List of k-mer sizes to compute (default: [3, 4])
        include_stats: Include global statistics (GC content, length, etc.)
        include_structure: Include secondary structure features (requires ViennaRNA)
        include_positional: Include positional encoding (fixed-length binned features)
        num_position_bins: Number of position bins for positional features
        normalize_kmers: Normalize k-mer frequencies to sum to 1

    Example:
        >>> embedder = NucleotideEmbedder(kmer_sizes=[3, 4, 5])
        >>> emb = embedder.encode("AUGCAUGCAUGC")
        >>> print(emb.shape)  # Will be (1152,) for default settings
    """

    def __init__(
        self,
        include_onehot: bool = True,
        include_kmers: bool = True,
        kmer_sizes: List[int] = None,
        include_stats: bool = True,
        include_structure: bool = False,
        include_positional: bool = True,
        num_position_bins: int = 10,
        normalize_kmers: bool = True
    ):
        self.include_onehot = include_onehot
        self.include_kmers = include_kmers
        self.kmer_sizes = kmer_sizes if kmer_sizes is not None else [3, 4]
        self.include_stats = include_stats
        self.include_structure = include_structure
        self.include_positional = include_positional
        self.num_position_bins = num_position_bins
        self.normalize_kmers = normalize_kmers

        # Nucleotide vocabulary
        self.nucleotides = ['A', 'C', 'G', 'U']
        self.nuc_to_idx = {n: i for i, n in enumerate(self.nucleotides)}

        # Precompute k-mer vocabularies
        self.kmer_vocabs = {}
        for k in self.kmer_sizes:
            kmers = [''.join(p) for p in product(self.nucleotides, repeat=k)]
            self.kmer_vocabs[k] = {kmer: i for i, kmer in enumerate(kmers)}

        # Calculate embedding dimension
        self._embedding_dim = self._calculate_embedding_dim()

        # Check structure availability
        if self.include_structure:
            try:
                import RNA  # ViennaRNA
                self._vienna_available = True
            except ImportError:
                print("Warning: ViennaRNA not installed. Structure features disabled.")
                self.include_structure = False
                self._vienna_available = False
                self._embedding_dim = self._calculate_embedding_dim()

    def _calculate_embedding_dim(self) -> int:
        """Calculate total embedding dimension."""
        dim = 0

        if self.include_onehot:
            dim += 4  # Mean-pooled one-hot

        if self.include_kmers:
            for k in self.kmer_sizes:
                dim += 4 ** k  # All possible k-mers

        if self.include_stats:
            dim += 8  # GC content, length, etc.

        if self.include_structure:
            dim += 5  # MFE, paired fraction, etc.

        if self.include_positional:
            dim += 4 * self.num_position_bins  # Binned positional one-hot

        return dim

    def encode(self, sequences: Union[str, List[str]]) -> np.ndarray:
        """
        Encode RNA sequence(s) to embedding vectors.

        Args:
            sequences: Single RNA sequence or list of sequences

        Returns:
            Embedding array of shape (embedding_dim,) or (n_sequences, embedding_dim)
        """
        if isinstance(sequences, str):
            sequences_list = [sequences]
            return_single = True
        else:
            sequences_list = list(sequences)
            return_single = False

        embeddings = []
        for seq in sequences_list:
            # Validate and normalize
            seq = self.validate_sequence(seq)

            # Compute features
            features = []

            if self.include_onehot:
                features.append(self._onehot_features(seq))

            if self.include_kmers:
                features.append(self._kmer_features(seq))

            if self.include_stats:
                features.append(self._stat_features(seq))

            if self.include_structure:
                features.append(self._structure_features(seq))

            if self.include_positional:
                features.append(self._positional_features(seq))

            embedding = np.concatenate(features)
            embeddings.append(embedding)

        result = np.array(embeddings, dtype=np.float32)

        if return_single:
            return result[0]
        return result

    def _onehot_features(self, seq: str) -> np.ndarray:
        """
        Mean-pooled one-hot encoding.

        Returns 4-dimensional vector representing nucleotide frequencies.
        """
        counts = np.zeros(4, dtype=np.float32)
        for nuc in seq:
            if nuc in self.nuc_to_idx:
                counts[self.nuc_to_idx[nuc]] += 1

        # Normalize to frequencies
        if len(seq) > 0:
            counts /= len(seq)

        return counts

    def _kmer_features(self, seq: str) -> np.ndarray:
        """
        K-mer frequency vectors for each k in kmer_sizes.
        """
        all_kmer_features = []

        for k in self.kmer_sizes:
            vocab = self.kmer_vocabs[k]
            counts = np.zeros(len(vocab), dtype=np.float32)

            # Count k-mers
            for i in range(len(seq) - k + 1):
                kmer = seq[i:i+k]
                if kmer in vocab:
                    counts[vocab[kmer]] += 1

            # Normalize
            if self.normalize_kmers and counts.sum() > 0:
                counts /= counts.sum()

            all_kmer_features.append(counts)

        return np.concatenate(all_kmer_features)

    def _stat_features(self, seq: str) -> np.ndarray:
        """
        Global sequence statistics.

        Features:
        - GC content
        - AU content
        - Length (log-scaled)
        - Purine/pyrimidine ratio
        - Mono-nucleotide runs (max run length)
        - Dinucleotide bias (CpG, UpA)
        """
        n = len(seq)
        if n == 0:
            return np.zeros(8, dtype=np.float32)

        # Count nucleotides
        counts = Counter(seq)
        a_count = counts.get('A', 0)
        c_count = counts.get('C', 0)
        g_count = counts.get('G', 0)
        u_count = counts.get('U', 0)

        # Basic content
        gc_content = (g_count + c_count) / n
        au_content = (a_count + u_count) / n

        # Length (log-scaled for better distribution)
        log_length = np.log1p(n) / 10.0  # Normalized

        # Purine/pyrimidine ratio
        purines = a_count + g_count
        pyrimidines = c_count + u_count
        pur_pyr_ratio = purines / max(pyrimidines, 1)
        pur_pyr_ratio = min(pur_pyr_ratio, 5.0) / 5.0  # Clip and normalize

        # Max mono-nucleotide run
        max_run = self._max_run_length(seq) / min(n, 20)  # Normalized

        # Dinucleotide frequencies
        dinuc_counts = Counter(seq[i:i+2] for i in range(len(seq) - 1))
        total_dinucs = max(sum(dinuc_counts.values()), 1)

        cpg_freq = dinuc_counts.get('CG', 0) / total_dinucs
        upa_freq = dinuc_counts.get('UA', 0) / total_dinucs

        # A-richness in first 10nt (Kozak-relevant)
        first_10 = seq[:10] if len(seq) >= 10 else seq
        a_first_10 = first_10.count('A') / len(first_10) if first_10 else 0

        return np.array([
            gc_content,
            au_content,
            log_length,
            pur_pyr_ratio,
            max_run,
            cpg_freq,
            upa_freq,
            a_first_10
        ], dtype=np.float32)

    def _max_run_length(self, seq: str) -> int:
        """Find maximum run of identical nucleotides."""
        if not seq:
            return 0

        max_run = 1
        current_run = 1

        for i in range(1, len(seq)):
            if seq[i] == seq[i-1]:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 1

        return max_run

    def _structure_features(self, seq: str) -> np.ndarray:
        """
        Secondary structure features using ViennaRNA.

        Features:
        - Minimum free energy (normalized)
        - Paired fraction
        - Mean base pair probability
        - Structure complexity (number of loops)
        - Accessibility (unpaired probability in first 30nt)
        """
        try:
            import RNA

            # Fold sequence
            structure, mfe = RNA.fold(seq)

            # Normalize MFE
            mfe_normalized = mfe / len(seq) if len(seq) > 0 else 0
            mfe_normalized = max(min(mfe_normalized, 0), -2) / 2  # Clip to [-2, 0], normalize

            # Paired fraction
            paired = structure.count('(') + structure.count(')')
            paired_fraction = paired / len(structure) if structure else 0

            # Base pair probabilities
            fc = RNA.fold_compound(seq)
            fc.pf()
            bpp = fc.bpp()

            # Mean base pair probability
            n = len(seq)
            total_bpp = 0
            count = 0
            for i in range(1, n + 1):
                for j in range(i + 1, n + 1):
                    if bpp[i][j] > 0.01:
                        total_bpp += bpp[i][j]
                        count += 1
            mean_bpp = total_bpp / max(count, 1)

            # Structure complexity (number of loops/hairpins)
            loops = structure.count('(.')  # Simplified
            complexity = min(loops, 10) / 10

            # 5' accessibility (important for translation)
            first_30 = min(30, n)
            unpaired_5 = sum(1 for i in range(first_30) if structure[i] == '.')
            accessibility = unpaired_5 / first_30 if first_30 > 0 else 1.0

            return np.array([
                -mfe_normalized,  # Positive = more stable
                paired_fraction,
                mean_bpp,
                complexity,
                accessibility
            ], dtype=np.float32)

        except Exception as e:
            # Return zeros if structure prediction fails
            return np.zeros(5, dtype=np.float32)

    def _positional_features(self, seq: str) -> np.ndarray:
        """
        Positional features: binned nucleotide composition.

        Divides sequence into num_position_bins bins and computes
        nucleotide frequencies in each bin.
        """
        n = len(seq)
        if n == 0:
            return np.zeros(4 * self.num_position_bins, dtype=np.float32)

        features = np.zeros((self.num_position_bins, 4), dtype=np.float32)

        # Compute bin boundaries
        bin_size = n / self.num_position_bins

        for i, nuc in enumerate(seq):
            if nuc in self.nuc_to_idx:
                bin_idx = min(int(i / bin_size), self.num_position_bins - 1)
                features[bin_idx, self.nuc_to_idx[nuc]] += 1

        # Normalize each bin
        for b in range(self.num_position_bins):
            bin_sum = features[b].sum()
            if bin_sum > 0:
                features[b] /= bin_sum

        return features.flatten()

    @property
    def embedding_dim(self) -> int:
        """Return the dimensionality of embeddings."""
        return self._embedding_dim

    @property
    def name(self) -> str:
        """Return the name of this embedding method."""
        kmer_str = '_'.join(str(k) for k in self.kmer_sizes)
        components = []

        if self.include_onehot:
            components.append('onehot')
        if self.include_kmers:
            components.append(f'kmer{kmer_str}')
        if self.include_stats:
            components.append('stats')
        if self.include_structure:
            components.append('struct')
        if self.include_positional:
            components.append(f'pos{self.num_position_bins}')

        return f"nucleotide_{'_'.join(components)}_{self._embedding_dim}"
