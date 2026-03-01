"""Pooled MLP baseline: RNA-FM pooled embedding only -> MLP (no edit info).

This baseline uses only the pooled original sequence embedding without
any edit information. It tests how much the model can predict from
sequence context alone, serving as a lower bound for edit-aware models.
If this baseline performs well, it suggests the task can be solved
primarily from sequence features rather than edit-specific signals.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class PooledMLPConfig:
    """Configuration for the Pooled MLP baseline.

    Parameters
    ----------
    d_model : int
        Dimension of pooled embedding (RNA-FM = 640).
    hidden_dims : tuple[int, ...]
        Hidden layer dimensions for the MLP.
    dropout : float
        Dropout rate between layers.
    """

    d_model: int = 640
    hidden_dims: tuple = (256, 128)
    dropout: float = 0.3


class PooledMLPBaseline(nn.Module):
    """Pooled MLP baseline for binary editing prediction.

    Uses only the pooled original sequence embedding (no edit information)
    as input to a 3-layer MLP. This is essentially a sequence-level
    classifier that ignores the edit entirely.

    Input keys from batch dict:
        - ``pooled_orig``: (B, d_model) pooled original embedding

    Returns:
        dict with ``"binary_logit"``: (B, 1)
    """

    def __init__(self, config: PooledMLPConfig | None = None):
        super().__init__()
        cfg = config or PooledMLPConfig()
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
        pooled_orig = batch["pooled_orig"]  # (B, d_model)

        logit = self.mlp(pooled_orig)  # (B, 1)

        return {"binary_logit": logit}
