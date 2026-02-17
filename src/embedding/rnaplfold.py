"""
RNAplfold secondary structure predictor.

Uses ViennaRNA package to compute local base-pairing probabilities
and accessibility scores for RNA secondary structure prediction.

This is a non-trainable predictor based on dynamic programming algorithms.
"""

import numpy as np
from typing import List, Optional, Dict, Tuple, Union
import warnings


class RNAplfoldPredictor:
    """
    RNA secondary structure predictor using RNAplfold from ViennaRNA.

    Computes:
    - Base-pairing probabilities for each position
    - Accessibility scores (probability of being unpaired)
    - Local folding energy estimates
    - Structure entropy at each position

    This is non-trainable - it uses thermodynamic parameters from ViennaRNA.
    """

    def __init__(
        self,
        window_size: int = 70,
        max_bp_span: int = 40,
        unpaired_length: int = 4,
        temperature: float = 37.0,
        cache_enabled: bool = True
    ):
        """
        Initialize RNAplfold predictor.

        Args:
            window_size: Window size for local folding (W parameter)
            max_bp_span: Maximum base pair span (L parameter)
            unpaired_length: Length for accessibility calculation (u parameter)
            temperature: Temperature in Celsius for energy calculations
            cache_enabled: Whether to cache results for repeated sequences
        """
        self.window_size = window_size
        self.max_bp_span = max_bp_span
        self.unpaired_length = unpaired_length
        self.temperature = temperature
        self.cache_enabled = cache_enabled
        self._cache: Dict[str, Dict] = {}

        # Try to import ViennaRNA
        self._vienna_available = False
        try:
            import RNA
            self._RNA = RNA
            self._vienna_available = True
        except ImportError:
            warnings.warn(
                "ViennaRNA package not found. Install with: "
                "conda install -c bioconda viennarna"
            )

    @property
    def is_available(self) -> bool:
        """Check if ViennaRNA is available."""
        return self._vienna_available

    def _normalize_sequence(self, sequence: str) -> str:
        """Normalize sequence to RNA format."""
        return sequence.upper().replace('T', 'U').replace(' ', '')

    def _get_cached(self, sequence: str) -> Optional[Dict]:
        """Get cached results if available."""
        if self.cache_enabled and sequence in self._cache:
            return self._cache[sequence]
        return None

    def _set_cached(self, sequence: str, results: Dict):
        """Cache results for a sequence."""
        if self.cache_enabled:
            self._cache[sequence] = results

    def predict_structure(self, sequence: str) -> Dict[str, np.ndarray]:
        """
        Predict secondary structure features for a sequence.

        Args:
            sequence: RNA sequence (A, C, G, U)

        Returns:
            Dictionary with:
                - pairing_prob: Base-pairing probability at each position [0, 1]
                - accessibility: Accessibility (unpaired probability) at each position [0, 1]
                - entropy: Structure entropy at each position
                - mfe: Minimum free energy (scalar)
                - ensemble_energy: Ensemble free energy (scalar)
        """
        sequence = self._normalize_sequence(sequence)

        # Check cache
        cached = self._get_cached(sequence)
        if cached is not None:
            return cached

        if not self._vienna_available:
            # Return dummy values if ViennaRNA not available
            n = len(sequence)
            return {
                'pairing_prob': np.zeros(n),
                'accessibility': np.ones(n),
                'entropy': np.zeros(n),
                'mfe': 0.0,
                'ensemble_energy': 0.0
            }

        RNA = self._RNA
        n = len(sequence)

        # Set model parameters
        md = RNA.md()
        md.temperature = self.temperature
        md.window_size = min(self.window_size, n)
        md.max_bp_span = min(self.max_bp_span, n)

        # Create fold compound
        fc = RNA.fold_compound(sequence, md)

        # Compute MFE structure
        mfe_structure, mfe = fc.mfe()

        # Compute partition function and base pair probabilities
        _, pf_energy = fc.pf()

        # Get base pair probability matrix
        # ViennaRNA's bpp() returns (n+1, n+1) matrix with 1-indexed positions
        bpp_raw = np.array(fc.bpp())
        # Extract the n x n submatrix (skip index 0)
        bpp = bpp_raw[1:n+1, 1:n+1]

        # Compute per-position pairing probability (sum of BP probs)
        pairing_prob = np.sum(bpp, axis=1)
        pairing_prob = np.clip(pairing_prob, 0, 1)

        # Accessibility = 1 - pairing probability
        accessibility = 1.0 - pairing_prob

        # Compute positional entropy from BP probabilities
        # H = -sum(p * log(p)) where p are the BP probabilities
        entropy = np.zeros(n)
        for i in range(n):
            probs = bpp[i, :]
            probs = probs[probs > 1e-10]  # Filter out zeros
            if len(probs) > 0:
                # Include unpaired probability
                unpaired = max(0, 1.0 - np.sum(probs))
                if unpaired > 1e-10:
                    probs = np.append(probs, unpaired)
                entropy[i] = -np.sum(probs * np.log2(probs + 1e-10))

        results = {
            'pairing_prob': pairing_prob,
            'accessibility': accessibility,
            'entropy': entropy,
            'mfe': mfe,
            'ensemble_energy': pf_energy,
            'bpp_matrix': bpp  # Full BP probability matrix
        }

        self._set_cached(sequence, results)
        return results

    def predict_batch(
        self,
        sequences: List[str],
        show_progress: bool = False
    ) -> List[Dict[str, np.ndarray]]:
        """
        Predict structure features for multiple sequences.

        Args:
            sequences: List of RNA sequences
            show_progress: Show progress bar

        Returns:
            List of result dictionaries
        """
        if show_progress:
            try:
                from tqdm import tqdm
                sequences = tqdm(sequences, desc="RNAplfold")
            except ImportError:
                pass

        return [self.predict_structure(seq) for seq in sequences]

    def compute_delta_structure(
        self,
        seq_before: str,
        seq_after: str,
        edit_position: int
    ) -> Dict[str, np.ndarray]:
        """
        Compute structure changes due to an edit.

        Args:
            seq_before: Original sequence
            seq_after: Mutated sequence
            edit_position: Position of the edit (0-indexed)

        Returns:
            Dictionary with:
                - delta_pairing: Change in pairing probability at each position
                - delta_accessibility: Change in accessibility at each position
                - delta_entropy: Change in structure entropy
                - delta_mfe: Change in minimum free energy
                - delta_local_pairing: Change in pairing prob in local window
        """
        struct_before = self.predict_structure(seq_before)
        struct_after = self.predict_structure(seq_after)

        delta_pairing = struct_after['pairing_prob'] - struct_before['pairing_prob']
        delta_accessibility = struct_after['accessibility'] - struct_before['accessibility']
        delta_entropy = struct_after['entropy'] - struct_before['entropy']
        delta_mfe = struct_after['mfe'] - struct_before['mfe']

        # Compute local delta (around edit position)
        window = 10
        n = len(seq_before)
        start = max(0, edit_position - window)
        end = min(n, edit_position + window + 1)

        delta_local_pairing = delta_pairing[start:end].mean() if end > start else 0.0

        return {
            'delta_pairing': delta_pairing,
            'delta_accessibility': delta_accessibility,
            'delta_entropy': delta_entropy,
            'delta_mfe': delta_mfe,
            'delta_local_pairing': delta_local_pairing,
            # Include raw structures for more detailed analysis
            'struct_before': struct_before,
            'struct_after': struct_after
        }

    def get_local_features(
        self,
        sequence: str,
        position: int,
        window: int = 10
    ) -> np.ndarray:
        """
        Get structure features around a specific position.

        Args:
            sequence: RNA sequence
            position: Center position (0-indexed)
            window: Window size on each side

        Returns:
            Feature vector with local structure information
        """
        struct = self.predict_structure(sequence)
        n = len(sequence)

        start = max(0, position - window)
        end = min(n, position + window + 1)

        # Extract local features
        local_pairing = struct['pairing_prob'][start:end]
        local_accessibility = struct['accessibility'][start:end]
        local_entropy = struct['entropy'][start:end]

        # Pad if needed
        expected_len = 2 * window + 1
        if len(local_pairing) < expected_len:
            pad_left = position - start
            pad_right = expected_len - len(local_pairing) - (window - pad_left)
            local_pairing = np.pad(local_pairing, (max(0, window - pad_left), max(0, pad_right)))
            local_accessibility = np.pad(local_accessibility, (max(0, window - pad_left), max(0, pad_right)))
            local_entropy = np.pad(local_entropy, (max(0, window - pad_left), max(0, pad_right)))

        # Aggregate statistics
        features = np.array([
            # Position-specific
            struct['pairing_prob'][position],
            struct['accessibility'][position],
            struct['entropy'][position],
            # Local statistics
            local_pairing.mean(),
            local_pairing.std(),
            local_accessibility.mean(),
            local_entropy.mean(),
            # Global
            struct['mfe'],
        ])

        return features

    def clear_cache(self):
        """Clear the structure prediction cache."""
        self._cache.clear()


class RNAfoldPredictor:
    """
    Full sequence folding using RNAfold from ViennaRNA.

    For global structure prediction (vs local with RNAplfold).
    """

    def __init__(self, temperature: float = 37.0):
        """
        Initialize RNAfold predictor.

        Args:
            temperature: Temperature in Celsius
        """
        self.temperature = temperature
        self._vienna_available = False

        try:
            import RNA
            self._RNA = RNA
            self._vienna_available = True
        except ImportError:
            warnings.warn(
                "ViennaRNA package not found. Install with: "
                "conda install -c bioconda viennarna"
            )

    @property
    def is_available(self) -> bool:
        return self._vienna_available

    def predict_mfe(self, sequence: str) -> Tuple[str, float]:
        """
        Predict minimum free energy structure.

        Args:
            sequence: RNA sequence

        Returns:
            Tuple of (dot-bracket structure, MFE in kcal/mol)
        """
        if not self._vienna_available:
            return '.' * len(sequence), 0.0

        sequence = sequence.upper().replace('T', 'U')
        RNA = self._RNA

        md = RNA.md()
        md.temperature = self.temperature
        fc = RNA.fold_compound(sequence, md)

        structure, mfe = fc.mfe()
        return structure, mfe

    def predict_ensemble(self, sequence: str) -> Tuple[str, float, float]:
        """
        Predict ensemble properties.

        Args:
            sequence: RNA sequence

        Returns:
            Tuple of (centroid structure, ensemble energy, mean BP distance)
        """
        if not self._vienna_available:
            return '.' * len(sequence), 0.0, 0.0

        sequence = sequence.upper().replace('T', 'U')
        RNA = self._RNA

        md = RNA.md()
        md.temperature = self.temperature
        fc = RNA.fold_compound(sequence, md)

        # Compute MFE first
        fc.mfe()

        # Partition function
        fc.pf()

        # Centroid structure
        centroid, dist = fc.centroid()

        # Ensemble energy
        fe = fc.mean_bp_distance()

        return centroid, fe, dist
