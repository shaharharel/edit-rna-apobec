"""RNA-FM encoder wrapper for extracting pre-trained RNA embeddings.

Wraps the RNA-FM foundation model (Chen et al., Nature Methods 2024) to extract
per-token contextual embeddings from RNA sequences. Supports frozen inference
and partial fine-tuning of the last N transformer layers.

RNA-FM produces 640-dimensional embeddings per nucleotide from a 12-layer
transformer trained on 23M+ non-coding RNA sequences.
"""

import torch
import torch.nn as nn


class RNAFMEncoder(nn.Module):
    """Wrapper around RNA-FM for extracting per-token RNA embeddings.

    Parameters
    ----------
    repr_layer : int
        Which transformer layer to extract representations from (1-12).
        Default 12 (final layer).
    finetune_last_n : int
        Number of final transformer layers to keep trainable. 0 = fully frozen.
    projection_dim : int or None
        If set, project embeddings to this dimension via a learned linear layer.
    """

    EMBEDDING_DIM = 640  # RNA-FM output dimension
    NUM_LAYERS = 12

    def __init__(
        self,
        repr_layer: int = 12,
        finetune_last_n: int = 0,
        projection_dim: int | None = None,
    ):
        super().__init__()
        self.repr_layer = repr_layer
        self.finetune_last_n = finetune_last_n

        # Load RNA-FM pretrained model
        import fm

        self.model, self.alphabet = fm.pretrained.rna_fm_t12()
        self.batch_converter = self.alphabet.get_batch_converter()

        # Freeze parameters
        self._setup_freezing()

        # Optional projection head
        self.projection = None
        out_dim = self.EMBEDDING_DIM
        if projection_dim is not None:
            self.projection = nn.Sequential(
                nn.Linear(self.EMBEDDING_DIM, projection_dim),
                nn.GELU(),
                nn.LayerNorm(projection_dim),
            )
            out_dim = projection_dim
        self.output_dim = out_dim

    def _setup_freezing(self):
        """Freeze all parameters except the last N transformer layers."""
        # First freeze everything
        for param in self.model.parameters():
            param.requires_grad = False

        if self.finetune_last_n > 0:
            # Unfreeze last N layers
            layers = self.model.layers
            total_layers = len(layers)
            for i in range(total_layers - self.finetune_last_n, total_layers):
                for param in layers[i].parameters():
                    param.requires_grad = True
            # Also unfreeze the final layer norm
            if hasattr(self.model, "emb_layer_norm_after"):
                for param in self.model.emb_layer_norm_after.parameters():
                    param.requires_grad = True

    def tokenize(self, sequences: list[str], names: list[str] | None = None) -> torch.Tensor:
        """Convert RNA sequences to token tensors.

        Parameters
        ----------
        sequences : list[str]
            RNA sequences (e.g., ["AUGCAUGC", "GCUAGCUA"]).
        names : list[str] or None
            Optional names for each sequence.

        Returns
        -------
        torch.Tensor
            Token indices, shape (batch, max_seq_len + 2) including BOS/EOS.
        """
        if names is None:
            names = [f"seq_{i}" for i in range(len(sequences))]
        data = list(zip(names, sequences))
        _, _, batch_tokens = self.batch_converter(data)
        return batch_tokens

    def forward(
        self,
        tokens: torch.Tensor | None = None,
        sequences: list[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Extract per-token embeddings from RNA-FM.

        Provide either pre-tokenized `tokens` or raw `sequences` (not both).

        Parameters
        ----------
        tokens : torch.Tensor or None
            Pre-tokenized input, shape (batch, seq_len).
        sequences : list[str] or None
            Raw RNA sequences to tokenize on the fly.

        Returns
        -------
        dict with keys:
            "embeddings": (batch, seq_len, dim) - per-token embeddings
                          (BOS/EOS tokens stripped)
            "pooled": (batch, dim) - mean-pooled sequence embedding
            "tokens": (batch, seq_len) - token ids used
        """
        if tokens is None and sequences is None:
            raise ValueError("Provide either tokens or sequences")

        if tokens is None:
            tokens = self.tokenize(sequences)
            tokens = tokens.to(next(self.model.parameters()).device)

        with torch.set_grad_enabled(self.finetune_last_n > 0 and self.training):
            results = self.model(tokens, repr_layers=[self.repr_layer])

        # Shape: (batch, seq_len_with_special, 640)
        embeddings = results["representations"][self.repr_layer]

        # Strip BOS (position 0) and EOS (last position) tokens
        # RNA-FM adds <cls> at start and <eos> at end
        embeddings = embeddings[:, 1:-1, :]

        # Apply optional projection
        if self.projection is not None:
            embeddings = self.projection(embeddings)

        # Mean pooling (ignoring padding)
        # Create mask from tokens (padding token is typically 1)
        # Strip special tokens from mask too
        mask = (tokens[:, 1:-1] != self.alphabet.padding_idx).unsqueeze(-1).float()
        pooled = (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        return {
            "embeddings": embeddings,
            "pooled": pooled,
            "tokens": tokens,
        }

    def get_embedding_dim(self) -> int:
        """Return the output embedding dimension."""
        return self.output_dim

    @torch.no_grad()
    def embed_sequences(self, sequences: list[str], batch_size: int = 32) -> torch.Tensor:
        """Convenience method: embed sequences in batches, return pooled embeddings.

        Parameters
        ----------
        sequences : list[str]
            RNA sequences.
        batch_size : int
            Processing batch size.

        Returns
        -------
        torch.Tensor
            Pooled embeddings, shape (n_sequences, dim).
        """
        self.eval()
        device = next(self.model.parameters()).device
        all_pooled = []

        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i : i + batch_size]
            tokens = self.tokenize(batch_seqs).to(device)
            result = self.forward(tokens=tokens)
            all_pooled.append(result["pooled"].cpu())

        return torch.cat(all_pooled, dim=0)
