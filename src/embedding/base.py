"""
Base interface for RNA embedders.

Provides the abstract interface for RNA sequence embedding methods.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Union, List


class RNAEmbedder(ABC):
    """
    Abstract base class for RNA embedding methods.

    All RNA embedders should implement encode() to convert
    RNA sequences to fixed-dimensional vectors.

    This interface is designed to be compatible with the
    existing edit-chem framework, allowing RNA embedders
    to be used interchangeably with molecule embedders
    in the EditEffectPredictor and PropertyPredictor models.
    """

    @abstractmethod
    def encode(self, sequences: Union[str, List[str]]) -> np.ndarray:
        """
        Encode RNA sequence(s) to embedding vector(s).

        Args:
            sequences: Single RNA sequence string or list of sequences.
                      Sequences should contain only A, C, G, U (or T).
                      DNA sequences (with T) are automatically converted to RNA.

        Returns:
            Embedding vector(s) as numpy array
            - Single sequence: shape (embedding_dim,)
            - List of sequences: shape (n_sequences, embedding_dim)
        """
        pass

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the dimensionality of embeddings."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this embedding method."""
        pass

    @property
    def max_length(self) -> int:
        """
        Maximum sequence length supported by this embedder.

        Override in subclasses if there's a hard limit.
        Default is 512 nucleotides (suitable for most UTRs).
        """
        return 512

    @property
    def trainable(self) -> bool:
        """
        Whether this embedder has trainable parameters.

        Override in subclasses that support fine-tuning.
        """
        return False

    def validate_sequence(self, sequence: str) -> str:
        """
        Validate and normalize RNA sequence.

        - Converts to uppercase
        - Converts T to U (DNA to RNA)
        - Removes whitespace

        Args:
            sequence: Input sequence

        Returns:
            Normalized RNA sequence

        Raises:
            ValueError: If sequence contains invalid characters
        """
        # Normalize
        seq = sequence.upper().replace(' ', '').replace('\n', '')

        # Convert DNA to RNA
        seq = seq.replace('T', 'U')

        # Validate
        # Include M and other modified nucleotides supported by RNA-FM
        valid_chars = set('ACGUM')
        invalid_chars = set(seq) - valid_chars

        if invalid_chars:
            raise ValueError(
                f"Invalid characters in RNA sequence: {invalid_chars}. "
                f"Only A, C, G, U, M are allowed."
            )

        return seq

    def batch_encode(
        self,
        sequences: List[str],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Encode sequences in batches (useful for large datasets).

        Default implementation processes all at once.
        Override in subclasses for memory-efficient batching.

        Args:
            sequences: List of RNA sequences
            batch_size: Number of sequences per batch
            show_progress: Show progress bar

        Returns:
            Embeddings array of shape (n_sequences, embedding_dim)
        """
        if show_progress:
            try:
                from tqdm import tqdm
                sequences = tqdm(sequences, desc="Encoding")
            except ImportError:
                pass

        return self.encode(sequences)
