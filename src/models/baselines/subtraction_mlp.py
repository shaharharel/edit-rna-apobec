"""Subtraction baseline: RNA-FM(edited) - RNA-FM(original) -> MLP.

This baseline implements the subtraction approach from the causal edit
effect framework comparison: independently encode original and edited
sequences, subtract the pooled embeddings, and classify the difference.
This is the primary comparison point for the edit effect framework,
testing whether F(seq_after) - F(seq_before) is sufficient versus
learning directly from the edit embedding F(seq_before, edit).
"""

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class SubtractionMLPConfig:
    """Configuration for the Subtraction MLP baseline.

    Parameters
    ----------
    d_model : int
        Dimension of pooled embeddings (RNA-FM = 640).
    hidden_dims : tuple[int, ...]
        Hidden layer dimensions for the MLP.
    dropout : float
        Dropout rate between layers.
    """

    d_model: int = 640
    hidden_dims: tuple = (256, 128)
    dropout: float = 0.3


class SubtractionMLPBaseline(nn.Module):
    """Subtraction MLP baseline for binary editing prediction.

    Computes the difference between pooled edited and original sequence
    embeddings, then passes through a 3-layer MLP. This directly
    implements the subtraction baseline from the edit effect framework:
    F(seq_after) - F(seq_before) = delta_property.

    Input keys from batch dict:
        - ``pooled_orig``: (B, d_model) pooled original embedding
        - ``pooled_edited``: (B, d_model) pooled edited embedding

    Returns:
        dict with ``"binary_logit"``: (B, 1)
    """

    def __init__(self, config: SubtractionMLPConfig | None = None):
        super().__init__()
        cfg = config or SubtractionMLPConfig()
        self.config = cfg

        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dims[0], cfg.hidden_dims[1]),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dims[1], 1),
        )

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        pooled_orig = batch["pooled_orig"]      # (B, d_model)
        pooled_edited = batch["pooled_edited"]  # (B, d_model)

        diff = pooled_edited - pooled_orig  # (B, d_model)

        logit = self.mlp(diff)  # (B, 1)

        return {"binary_logit": logit}
