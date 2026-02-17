"""
RNABERT embedder for RNA sequences.

RNABERT is a BERT-style transformer pretrained on RNA sequences
with a focus on secondary structure awareness.

This provides an alternative to RNA-FM with different training objectives
and potentially different inductive biases.

Installation:
    pip install transformers torch
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Union, List, Optional
from .base import RNAEmbedder


class RNABERTEmbedder(nn.Module, RNAEmbedder):
    """
    RNABERT embedder for RNA sequences.

    Uses a BERT-style transformer pretrained on RNA sequences.
    Multiple RNABERT variants are available through HuggingFace.

    Installation:
        pip install transformers torch

    Args:
        model_name: HuggingFace model name:
            - "multimolecule/rnabert" (default): Standard RNABERT
            - "multimolecule/rnamsm": RNA masked sequence model
            - Custom model path
        pooling: How to aggregate token embeddings:
            - "mean" (default): Mean of all token embeddings
            - "cls": Use [CLS] token embedding
            - "max": Max pooling across tokens
        device: Device to run on ("auto", "cuda", "cpu")
        batch_size: Batch size for encoding
        trainable: Whether to allow fine-tuning (default: False)
        max_length: Maximum sequence length (default: 440 for RNABERT)

    Example:
        >>> embedder = RNABERTEmbedder()
        >>> emb = embedder.encode("AUGCAUGCAUGCAUGC")
        >>> print(emb.shape)  # (768,)
    """

    def __init__(
        self,
        model_name: str = "multimolecule/rnabert",
        pooling: str = "mean",
        device: str = "auto",
        batch_size: int = 32,
        trainable: bool = False,
        max_length: int = 440
    ):
        super().__init__()

        self.model_name = model_name
        self.pooling = pooling
        self.batch_size = batch_size
        self._trainable = trainable
        self._max_length = max_length

        # Auto-detect device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load model
        self._load_model()

    def _load_model(self):
        """Load RNABERT model and tokenizer from HuggingFace."""
        try:
            from transformers import AutoModel, AutoTokenizer

            print(f"Loading RNABERT model: {self.model_name}")

            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            self.model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            # Get embedding dimension from model config
            self._embedding_dim = self.model.config.hidden_size

            # Move to device
            self.model = self.model.to(self.device)

            # Set trainability
            if not self._trainable:
                self.model.eval()
                for param in self.model.parameters():
                    param.requires_grad = False
                print(f"  RNABERT loaded on {self.device} (frozen, {self._embedding_dim}-dim)")
            else:
                self.model.train()
                print(f"  RNABERT loaded on {self.device} (trainable, {self._embedding_dim}-dim)")

            self._model_loaded = True

        except ImportError as e:
            raise ImportError(
                "transformers is not installed. Install with:\n"
                "pip install transformers torch\n"
                f"Original error: {e}"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load RNABERT model '{self.model_name}':\n{e}\n"
                "Make sure the model name is correct and you have internet access."
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
        # Tokenize
        inputs = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._max_length
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get embeddings
        with torch.no_grad() if not self._trainable else torch.enable_grad():
            outputs = self.model(**inputs)

            # Get last hidden state
            # Shape: [batch, seq_len, hidden_dim]
            hidden_states = outputs.last_hidden_state

            # Get attention mask for proper pooling
            attention_mask = inputs['attention_mask']

            # Pool across sequence
            if self.pooling == "mean":
                # Mean over non-padding tokens
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                embeddings = sum_embeddings / sum_mask

            elif self.pooling == "cls":
                # Use [CLS] token (first token)
                embeddings = hidden_states[:, 0, :]

            elif self.pooling == "max":
                # Max over non-padding tokens
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
                hidden_states[mask_expanded == 0] = -1e9
                embeddings = torch.max(hidden_states, dim=1)[0]

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
                           desc="Encoding with RNABERT")
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
        model_short = self.model_name.split("/")[-1]
        trainable_str = "_trainable" if self._trainable else "_frozen"
        return f"rnabert_{model_short}_{self.pooling}{trainable_str}"

    @property
    def max_length(self) -> int:
        """Maximum sequence length."""
        return self._max_length

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
        print("RNABERT model frozen")

    def unfreeze(self):
        """Unfreeze model parameters (enable fine-tuning)."""
        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = True
        self._trainable = True
        print("RNABERT model unfrozen for fine-tuning")

    def get_token_embeddings(
        self,
        sequence: str
    ) -> tuple:
        """
        Get per-token embeddings (for fine-grained analysis).

        Args:
            sequence: RNA sequence

        Returns:
            (embeddings, tokens) tuple where:
            - embeddings: array of shape (seq_len, hidden_dim)
            - tokens: list of token strings
        """
        seq = self.validate_sequence(sequence)

        inputs = self.tokenizer(
            seq,
            return_tensors="pt",
            truncation=True,
            max_length=self._max_length
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            hidden_states = outputs.last_hidden_state[0]  # Remove batch dim

        # Get tokens
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

        return hidden_states.cpu().numpy(), tokens
