"""
EternaFold secondary structure predictor.

Uses the EternaFold package for RNA secondary structure prediction.
EternaFold uses CNN-learned energy parameters with dynamic programming.

This is a non-trainable wrapper - the DP algorithm is not differentiable.

Installation options:
1. Use arnie (recommended): pip install arnie
   - Also requires EternaFold binary compiled from https://github.com/eternagame/EternaFold
   - Set eternafold_PATH environment variable
2. Use ViennaRNA fallback: conda install -c bioconda viennarna

Reference: https://github.com/DasLab/arnie
"""

import numpy as np
from typing import List, Optional, Dict, Tuple
import warnings
import subprocess
import tempfile
import os


class EternaFoldPredictor:
    """
    RNA secondary structure predictor using EternaFold.

    EternaFold provides more accurate structure predictions than ViennaRNA
    for some RNA types by using learned energy parameters from Eterna data.

    This is non-trainable - we only use it for feature extraction.
    """

    def __init__(
        self,
        cache_enabled: bool = True,
        use_vienna_fallback: bool = True
    ):
        """
        Initialize EternaFold predictor.

        Args:
            cache_enabled: Whether to cache results
            use_vienna_fallback: Fall back to ViennaRNA if EternaFold unavailable
        """
        self.cache_enabled = cache_enabled
        self.use_vienna_fallback = use_vienna_fallback
        self._cache: Dict[str, Dict] = {}

        # Check for arnie (preferred interface)
        self._arnie_available = self._check_arnie()

        # Check EternaFold availability (direct or via arnie)
        self._eternafold_available = self._check_eternafold()

        # Vienna fallback
        self._vienna_fallback = None
        if not self._eternafold_available and not self._arnie_available and use_vienna_fallback:
            try:
                from .rnaplfold import RNAplfoldPredictor
                self._vienna_fallback = RNAplfoldPredictor()
                if self._vienna_fallback.is_available:
                    warnings.warn(
                        "EternaFold not found, falling back to ViennaRNA"
                    )
            except ImportError:
                pass

    def _check_arnie(self) -> bool:
        """Check if arnie is available."""
        try:
            import arnie
            return True
        except ImportError:
            return False

    def _check_eternafold(self) -> bool:
        """Check if EternaFold is available (directly or via arnie)."""
        # Check via arnie first
        if self._arnie_available:
            try:
                from arnie.mfe import mfe
                # Try a simple prediction
                test_seq = "GCGCGCGCGC"
                result = mfe(test_seq, package="eternafold")
                return result is not None
            except Exception:
                pass

        # Check direct eternafold package
        try:
            import eternafold
            return True
        except ImportError:
            pass

        # Check if eternafold CLI is available
        try:
            result = subprocess.run(
                ['eternafold', '--version'],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    @property
    def is_available(self) -> bool:
        """Check if EternaFold (via arnie or direct) or fallback is available."""
        return (
            self._arnie_available or
            self._eternafold_available or
            (self._vienna_fallback is not None and self._vienna_fallback.is_available)
        )

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

    def _predict_with_arnie(self, sequence: str) -> Dict:
        """Predict using arnie package (preferred method)."""
        from arnie.mfe import mfe
        from arnie.bpps import bpps

        n = len(sequence)

        # Get MFE structure
        structure = mfe(sequence, package="eternafold")
        if structure is None:
            structure = '.' * n

        # Get base pair probability matrix
        try:
            bpp_matrix = bpps(sequence, package="eternafold")
            # Sum over each row to get per-position pairing probability
            pairing_prob = np.sum(bpp_matrix, axis=1)
            pairing_prob = np.clip(pairing_prob, 0, 1)
        except Exception:
            # Fall back to MFE-based pairing
            pairing_prob = np.zeros(n)
            stack = []
            for i, c in enumerate(structure):
                if c == '(':
                    stack.append(i)
                elif c == ')' and stack:
                    j = stack.pop()
                    pairing_prob[i] = 1.0
                    pairing_prob[j] = 1.0

        accessibility = 1.0 - pairing_prob

        # Entropy from bpp matrix
        entropy = np.zeros(n)
        if 'bpp_matrix' in dir():
            for i in range(n):
                probs = bpp_matrix[i, :]
                probs = probs[probs > 1e-10]
                if len(probs) > 0:
                    unpaired = max(0, 1.0 - np.sum(probs))
                    if unpaired > 1e-10:
                        probs = np.append(probs, unpaired)
                    entropy[i] = -np.sum(probs * np.log2(probs + 1e-10))

        # Parse pairs from structure
        pairs = []
        stack = []
        for i, c in enumerate(structure):
            if c == '(':
                stack.append(i)
            elif c == ')' and stack:
                j = stack.pop()
                pairs.append((j, i))

        return {
            'structure': structure,
            'pairing_prob': pairing_prob,
            'accessibility': accessibility,
            'entropy': entropy,
            'mfe': 0.0,  # arnie.mfe returns structure, not energy
            'pairs': pairs
        }

    def _predict_with_eternafold_python(self, sequence: str) -> Dict:
        """Predict using EternaFold Python package (direct import)."""
        import eternafold

        # Get structure and energy
        result = eternafold.fold(sequence)

        # Parse result - eternafold returns (structure, mfe)
        if isinstance(result, tuple):
            structure, mfe = result
        else:
            structure = result
            mfe = 0.0

        n = len(sequence)

        # Convert dot-bracket to per-position features
        pairing_prob = np.zeros(n)
        in_stem = np.zeros(n)

        # Simple parse of dot-bracket
        stack = []
        pairs = []
        for i, c in enumerate(structure):
            if c == '(':
                stack.append(i)
            elif c == ')' and stack:
                j = stack.pop()
                pairs.append((j, i))
                pairing_prob[i] = 1.0
                pairing_prob[j] = 1.0
                in_stem[i] = 1.0
                in_stem[j] = 1.0

        accessibility = 1.0 - pairing_prob

        # Entropy estimate (simplified - based on structure diversity)
        # For MFE only, entropy is 0; with ensemble it would be computed
        entropy = np.zeros(n)

        return {
            'structure': structure,
            'pairing_prob': pairing_prob,
            'accessibility': accessibility,
            'entropy': entropy,
            'mfe': mfe,
            'pairs': pairs
        }

    def _predict_with_eternafold_cli(self, sequence: str) -> Dict:
        """Predict using EternaFold CLI."""
        n = len(sequence)

        # Write sequence to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fa', delete=False) as f:
            f.write(f">seq\n{sequence}\n")
            fasta_path = f.name

        try:
            # Run eternafold
            result = subprocess.run(
                ['eternafold', fasta_path],
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode != 0:
                raise RuntimeError(f"EternaFold failed: {result.stderr}")

            # Parse output
            lines = result.stdout.strip().split('\n')
            structure = None
            mfe = 0.0

            for line in lines:
                line = line.strip()
                if line and not line.startswith('>') and not line.startswith('#'):
                    if '(' in line or '.' in line:
                        parts = line.split()
                        structure = parts[0]
                        if len(parts) > 1:
                            try:
                                mfe = float(parts[1].strip('()'))
                            except ValueError:
                                pass
                        break

            if structure is None:
                structure = '.' * n

            # Convert to features
            pairing_prob = np.zeros(n)
            stack = []
            pairs = []
            for i, c in enumerate(structure):
                if c == '(':
                    stack.append(i)
                elif c == ')' and stack:
                    j = stack.pop()
                    pairs.append((j, i))
                    pairing_prob[i] = 1.0
                    pairing_prob[j] = 1.0

            return {
                'structure': structure,
                'pairing_prob': pairing_prob,
                'accessibility': 1.0 - pairing_prob,
                'entropy': np.zeros(n),
                'mfe': mfe,
                'pairs': pairs
            }

        finally:
            os.unlink(fasta_path)

    def predict_structure(self, sequence: str) -> Dict[str, np.ndarray]:
        """
        Predict secondary structure features for a sequence.

        Args:
            sequence: RNA sequence (A, C, G, U)

        Returns:
            Dictionary with:
                - structure: Dot-bracket notation
                - pairing_prob: Base-pairing probability at each position
                - accessibility: Accessibility (unpaired probability)
                - entropy: Structure entropy
                - mfe: Minimum free energy
        """
        sequence = self._normalize_sequence(sequence)

        # Check cache
        cached = self._get_cached(sequence)
        if cached is not None:
            return cached

        # Try arnie first (preferred)
        if self._arnie_available:
            try:
                results = self._predict_with_arnie(sequence)
                self._set_cached(sequence, results)
                return results
            except Exception:
                pass

        # Try direct EternaFold
        if self._eternafold_available:
            try:
                import eternafold
                results = self._predict_with_eternafold_python(sequence)
                self._set_cached(sequence, results)
                return results
            except Exception:
                pass

            try:
                results = self._predict_with_eternafold_cli(sequence)
                self._set_cached(sequence, results)
                return results
            except Exception:
                pass

        # Fallback to Vienna
        if self._vienna_fallback is not None and self._vienna_fallback.is_available:
            results = self._vienna_fallback.predict_structure(sequence)
            # Add structure field
            if 'structure' not in results:
                n = len(sequence)
                results['structure'] = '.' * n
            self._set_cached(sequence, results)
            return results

        # Return dummy if nothing available
        n = len(sequence)
        return {
            'structure': '.' * n,
            'pairing_prob': np.zeros(n),
            'accessibility': np.ones(n),
            'entropy': np.zeros(n),
            'mfe': 0.0
        }

    def predict_batch(
        self,
        sequences: List[str],
        show_progress: bool = False
    ) -> List[Dict]:
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
                sequences = tqdm(sequences, desc="EternaFold")
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
            Dictionary with delta features
        """
        struct_before = self.predict_structure(seq_before)
        struct_after = self.predict_structure(seq_after)

        delta_pairing = struct_after['pairing_prob'] - struct_before['pairing_prob']
        delta_accessibility = struct_after['accessibility'] - struct_before['accessibility']
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
            'delta_mfe': delta_mfe,
            'delta_local_pairing': delta_local_pairing,
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

        # Aggregate statistics
        features = np.array([
            # Position-specific
            struct['pairing_prob'][position],
            struct['accessibility'][position],
            # Local statistics
            local_pairing.mean(),
            local_pairing.std() if len(local_pairing) > 1 else 0,
            local_accessibility.mean(),
            # Global
            struct['mfe'],
        ])

        return features

    def clear_cache(self):
        """Clear the structure prediction cache."""
        self._cache.clear()


class CombinedStructurePredictor:
    """
    Combines multiple structure predictors for ensemble predictions.

    Uses both RNAplfold (local folding) and EternaFold (global MFE)
    to get complementary structure information.
    """

    def __init__(self, cache_enabled: bool = True):
        """Initialize combined predictor."""
        from .rnaplfold import RNAplfoldPredictor

        self.plfold = RNAplfoldPredictor(cache_enabled=cache_enabled)
        self.eternafold = EternaFoldPredictor(
            cache_enabled=cache_enabled,
            use_vienna_fallback=False  # plfold already uses Vienna
        )
        self.cache_enabled = cache_enabled

    @property
    def is_available(self) -> bool:
        """Check if at least one predictor is available."""
        return self.plfold.is_available or self.eternafold.is_available

    def predict_structure(self, sequence: str) -> Dict[str, np.ndarray]:
        """
        Predict structure using all available predictors.

        Returns combined features from RNAplfold and EternaFold.
        """
        results = {}

        # RNAplfold (local folding probabilities)
        if self.plfold.is_available:
            plfold_result = self.plfold.predict_structure(sequence)
            results['plfold_pairing_prob'] = plfold_result['pairing_prob']
            results['plfold_accessibility'] = plfold_result['accessibility']
            results['plfold_entropy'] = plfold_result['entropy']
            results['plfold_mfe'] = plfold_result.get('mfe', 0.0)

        # EternaFold (global MFE structure)
        if self.eternafold._eternafold_available:
            ef_result = self.eternafold.predict_structure(sequence)
            results['eternafold_pairing_prob'] = ef_result['pairing_prob']
            results['eternafold_accessibility'] = ef_result['accessibility']
            results['eternafold_mfe'] = ef_result['mfe']
            results['eternafold_structure'] = ef_result.get('structure', '')

        # Use available results
        if 'plfold_pairing_prob' in results:
            results['pairing_prob'] = results['plfold_pairing_prob']
            results['accessibility'] = results['plfold_accessibility']
            results['entropy'] = results['plfold_entropy']
            results['mfe'] = results.get('plfold_mfe', 0.0)
        elif 'eternafold_pairing_prob' in results:
            results['pairing_prob'] = results['eternafold_pairing_prob']
            results['accessibility'] = results['eternafold_accessibility']
            results['entropy'] = np.zeros_like(results['eternafold_pairing_prob'])
            results['mfe'] = results.get('eternafold_mfe', 0.0)
        else:
            n = len(sequence)
            results['pairing_prob'] = np.zeros(n)
            results['accessibility'] = np.ones(n)
            results['entropy'] = np.zeros(n)
            results['mfe'] = 0.0

        return results

    def compute_delta_structure(
        self,
        seq_before: str,
        seq_after: str,
        edit_position: int
    ) -> Dict[str, np.ndarray]:
        """Compute structure delta using combined predictors."""
        struct_before = self.predict_structure(seq_before)
        struct_after = self.predict_structure(seq_after)

        delta = {
            'delta_pairing': struct_after['pairing_prob'] - struct_before['pairing_prob'],
            'delta_accessibility': struct_after['accessibility'] - struct_before['accessibility'],
            'delta_entropy': struct_after['entropy'] - struct_before['entropy'],
            'delta_mfe': struct_after['mfe'] - struct_before['mfe'],
        }

        # Local delta
        window = 10
        n = len(seq_before)
        start = max(0, edit_position - window)
        end = min(n, edit_position + window + 1)
        delta['delta_local_pairing'] = delta['delta_pairing'][start:end].mean()

        return delta

    def clear_cache(self):
        """Clear caches."""
        self.plfold.clear_cache()
        self.eternafold.clear_cache()
