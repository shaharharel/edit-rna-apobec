"""Cross-Attention baseline: Q=original tokens, K,V=edited tokens -> pool -> MLP.

This baseline tests whether cross-attention between original and edited
token-level embeddings can capture editing signals without the structured
edit embedding. The original sequence tokens attend to the edited sequence
tokens, and the result is mean-pooled and passed through an MLP.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class CrossAttentionConfig:
    """Configuration for the Cross-Attention baseline.

    Parameters
    ----------
    d_model : int
        Token embedding dimension (RNA-FM = 640).
    n_heads : int
        Number of attention heads.
    n_layers : int
        Number of cross-attention layers.
    d_hidden : int
        Hidden dimension in the output MLP.
    dropout : float
        Dropout rate.
    """

    d_model: int = 640
    n_heads: int = 8
    n_layers: int = 1
    d_hidden: int = 256
    dropout: float = 0.3


class CrossAttentionBaseline(nn.Module):
    """Cross-attention baseline for binary editing prediction.

    Applies multi-head cross-attention where the original sequence tokens
    serve as queries and the edited sequence tokens serve as keys and values.
    The attended output is mean-pooled and passed through an MLP.

    Input keys from batch dict:
        - ``tokens_orig``: (B, L, d_model) original token embeddings
        - ``tokens_edited``: (B, L, d_model) edited token embeddings

    Returns:
        dict with ``"binary_logit"``: (B, 1)
    """

    def __init__(self, config: CrossAttentionConfig | None = None):
        super().__init__()
        cfg = config or CrossAttentionConfig()
        self.config = cfg

        # Cross-attention layers
        self.cross_attn_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(cfg.n_layers):
            self.cross_attn_layers.append(
                nn.MultiheadAttention(
                    embed_dim=cfg.d_model,
                    num_heads=cfg.n_heads,
                    dropout=cfg.dropout,
                    batch_first=True,
                )
            )
            self.norms.append(nn.LayerNorm(cfg.d_model))

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

        x = tokens_orig
        for attn, norm in zip(self.cross_attn_layers, self.norms):
            attended, _ = attn(query=x, key=tokens_edited, value=tokens_edited)
            x = norm(x + attended)

        # Mean pool over sequence length
        pooled = x.mean(dim=1)  # (B, d_model)

        logit = self.mlp(pooled)  # (B, 1)

        return {"binary_logit": logit}
