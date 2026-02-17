"""
RNA-FM (RNA Foundation Model) embedder.

RNA-FM is a large-scale pretrained model for RNA sequences that can
generate rich embeddings capturing sequence and structural information.

Paper: "Interpretable RNA Foundation Model from Unannotated Data for Highly Accurate RNA Structure and Function Predictions"
GitHub: https://github.com/ml4bio/RNA-FM

This embedder provides:
- 640-dimensional embeddings per sequence
- Pretrained representations from large RNA databases
- Optional fine-tuning capability
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Union, List, Optional
from .base import RNAEmbedder


class RNAFMEmbedder(nn.Module, RNAEmbedder):
    """
    RNA-FM embedder for RNA sequences.

    Uses the RNA-FM pretrained model to generate rich embeddings
    that capture both sequence and predicted structural information.

    Installation:
        # From GitHub (recommended)
        pip install git+https://github.com/ml4bio/RNA-FM.git

        # Or if packaged:
        pip install rna-fm

    Args:
        model_name: RNA-FM model variant:
            - "rna_fm_t12" (default): 12-layer model, 640-dim
            - "rna_fm_t6": 6-layer model, smaller but faster
        pooling: How to aggregate token embeddings:
            - "mean" (default): Mean of all token embeddings
            - "cls": Use [CLS] token embedding
            - "max": Max pooling across tokens
        device: Device to run on ("auto", "cuda", "cpu")
        batch_size: Batch size for encoding
        trainable: Whether to allow fine-tuning (default: False)

    Example:
        >>> embedder = RNAFMEmbedder()
        >>> emb = embedder.encode("AUGCAUGCAUGCAUGC")
        >>> print(emb.shape)  # (640,)
    """

    def __init__(
        self,
        model_name: str = "rna_fm_t12",
        pooling: str = "mean",
        device: str = "auto",
        batch_size: int = 32,
        trainable: bool = False
    ):
        super().__init__()

        self.model_name = model_name
        self.pooling = pooling
        self.batch_size = batch_size
        self._trainable = trainable

        # Auto-detect device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Try to load RNA-FM
        self._load_model()

    def _load_model(self):
        """Load RNA-FM model and tokenizer."""
        try:
            import fm

            print(f"Loading RNA-FM model: {self.model_name}")

            # Load pretrained model
            if self.model_name == "rna_fm_t12":
                self.model, self.alphabet = fm.pretrained.rna_fm_t12()
                self._embedding_dim = 640
            elif self.model_name == "rna_fm_t6":
                # Smaller model if available
                try:
                    self.model, self.alphabet = fm.pretrained.rna_fm_t6()
                    self._embedding_dim = 640
                except AttributeError:
                    print("rna_fm_t6 not available, falling back to t12")
                    self.model, self.alphabet = fm.pretrained.rna_fm_t12()
                    self._embedding_dim = 640
            else:
                raise ValueError(f"Unknown RNA-FM model: {self.model_name}")

            # Move to device
            self.model = self.model.to(self.device)

            # Create batch converter
            self.batch_converter = self.alphabet.get_batch_converter()

            # Set trainability
            if not self._trainable:
                self.model.eval()
                for param in self.model.parameters():
                    param.requires_grad = False
                print(f"  RNA-FM loaded on {self.device} (frozen)")
            else:
                self.model.train()
                print(f"  RNA-FM loaded on {self.device} (trainable)")

            self._model_loaded = True

        except ImportError as e:
            raise ImportError(
                "RNA-FM is not installed. Install from GitHub:\n"
                "pip install git+https://github.com/ml4bio/RNA-FM.git\n"
                f"Original error: {e}"
            )

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

        # Validate and normalize sequences
        normalized = []
        for seq in sequences_list:
            seq = self.validate_sequence(seq)
            normalized.append(seq)

        # Encode in batches
        all_embeddings = []

        for i in range(0, len(normalized), self.batch_size):
            batch_seqs = normalized[i:i + self.batch_size]
            batch_emb = self._encode_batch(batch_seqs)
            all_embeddings.append(batch_emb)

        embeddings = np.vstack(all_embeddings)

        if return_single:
            return embeddings[0]
        return embeddings

    def _encode_batch(self, sequences: List[str]) -> np.ndarray:
        """Encode a batch of sequences."""
        # Prepare data for RNA-FM
        # Format: list of (name, sequence) tuples
        data = [(f"seq_{i}", seq) for i, seq in enumerate(sequences)]

        # Convert to batch
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)

        # Get embeddings
        with torch.no_grad() if not self._trainable else torch.enable_grad():
            results = self.model(batch_tokens, repr_layers=[12])

            # Get final layer representations
            # Shape: [batch, seq_len, hidden_dim]
            token_embeddings = results["representations"][12]

            # Pool across sequence
            if self.pooling == "mean":
                # Mean over all tokens (excluding special tokens)
                # RNA-FM uses positions 1:-1 for actual sequence
                embeddings = token_embeddings[:, 1:-1, :].mean(dim=1)

            elif self.pooling == "cls":
                # Use first token (CLS-like)
                embeddings = token_embeddings[:, 0, :]

            elif self.pooling == "max":
                # Max over all tokens
                embeddings = token_embeddings[:, 1:-1, :].max(dim=1)[0]

            else:
                raise ValueError(f"Unknown pooling: {self.pooling}")

        return embeddings.cpu().numpy().astype(np.float32)

    def batch_encode(
        self,
        sequences: List[str],
        batch_size: Optional[int] = None,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Encode sequences in batches.

        Args:
            sequences: List of RNA sequences
            batch_size: Override default batch size
            show_progress: Show progress bar

        Returns:
            Embeddings array of shape (n_sequences, embedding_dim)
        """
        if batch_size is not None:
            old_batch_size = self.batch_size
            self.batch_size = batch_size

        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(range(0, len(sequences), self.batch_size),
                           desc="Encoding with RNA-FM")
        else:
            iterator = range(0, len(sequences), self.batch_size)

        # Normalize all sequences
        normalized = [self.validate_sequence(s) for s in sequences]

        all_embeddings = []
        for i in iterator:
            batch_seqs = normalized[i:i + self.batch_size]
            batch_emb = self._encode_batch(batch_seqs)
            all_embeddings.append(batch_emb)

        if batch_size is not None:
            self.batch_size = old_batch_size

        return np.vstack(all_embeddings)

    @property
    def embedding_dim(self) -> int:
        """Return the dimensionality of embeddings."""
        return self._embedding_dim

    @property
    def name(self) -> str:
        """Return the name of this embedding method."""
        trainable_str = "_trainable" if self._trainable else "_frozen"
        return f"rna_fm_{self.model_name}_{self.pooling}{trainable_str}"

    @property
    def max_length(self) -> int:
        """Maximum sequence length for RNA-FM."""
        return 1024  # RNA-FM can handle longer sequences

    @property
    def trainable(self) -> bool:
        """Whether the model has trainable parameters."""
        return self._trainable

    def freeze(self):
        """Freeze model parameters (disable fine-tuning)."""
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self._trainable = False
        print("RNA-FM model frozen")

    def unfreeze(self):
        """Unfreeze model parameters (enable fine-tuning)."""
        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = True
        self._trainable = True
        print("RNA-FM model unfrozen for fine-tuning")

    def get_attention_weights(
        self,
        sequence: str,
        layer: int = -1
    ) -> np.ndarray:
        """
        Get attention weights for a sequence (interpretability).

        Args:
            sequence: RNA sequence
            layer: Which layer's attention (-1 for last)

        Returns:
            Attention weights array
        """
        seq = self.validate_sequence(sequence)
        data = [("seq", seq)]
        _, _, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)

        with torch.no_grad():
            results = self.model(batch_tokens, need_head_weights=True)
            attention = results["attentions"]

            if layer == -1:
                layer = len(attention) - 1

            # Get attention for specified layer
            # Shape: [batch, heads, seq_len, seq_len]
            layer_attention = attention[layer][0]  # Remove batch dim

            # Average over heads
            avg_attention = layer_attention.mean(dim=0)

        return avg_attention.cpu().numpy()
