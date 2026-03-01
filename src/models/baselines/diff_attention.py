"""Diff-Attention baseline: token diff -> TransformerEncoder -> pool -> MLP.

This baseline computes the token-level difference between edited and
original embeddings, processes it with a Transformer encoder, then
mean-pools and passes through an MLP. It tests whether a simple
subtraction at the token level, followed by self-attention over the
difference sequence, can capture editing effects.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class DiffAttentionConfig:
    """Configuration for the Diff-Attention baseline.

    Parameters
    ----------
    d_model : int
        Token embedding dimension (RNA-FM = 640).
    n_heads : int
        Number of attention heads in the TransformerEncoder.
    n_layers : int
        Number of TransformerEncoder layers.
    d_hidden : int
        Hidden dimension in the output MLP.
    dropout : float
        Dropout rate.
    """

    d_model: int = 640
    n_heads: int = 8
    n_layers: int = 2
    d_hidden: int = 256
    dropout: float = 0.3


class DiffAttentionBaseline(nn.Module):
    """Diff-Attention baseline for binary editing prediction.

    Computes the per-token difference (edited - original), processes
    it through a TransformerEncoder to capture inter-token dependencies
    in the difference signal, then mean-pools and classifies via MLP.

    Input keys from batch dict:
        - ``tokens_orig``: (B, L, d_model) original token embeddings
        - ``tokens_edited``: (B, L, d_model) edited token embeddings

    Returns:
        dict with ``"binary_logit"``: (B, 1)
    """

    def __init__(self, config: DiffAttentionConfig | None = None):
        super().__init__()
        cfg = config or DiffAttentionConfig()
        self.config = cfg

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * 4,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=cfg.n_layers
        )

        # Output MLP
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_hidden, 1),
        )

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        tokens_orig = batch["tokens_orig"]      # (B, L, d_model)
        tokens_edited = batch["tokens_edited"]  # (B, L, d_model)

        diff = tokens_edited - tokens_orig  # (B, L, d_model)

        encoded = self.transformer(diff)  # (B, L, d_model)

        # Mean pool over sequence length
        pooled = encoded.mean(dim=1)  # (B, d_model)

        logit = self.mlp(pooled)  # (B, 1)

        return {"binary_logit": logit}
