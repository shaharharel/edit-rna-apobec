"""Concat+MLP baseline: concatenates RNA-FM(orig) ++ RNA-FM(edited) -> MLP.

This baseline tests whether simply concatenating the pooled representations
of the original and edited sequences (without explicit edit modeling) is
sufficient for predicting editing status. It serves as a comparison point
for the structured edit embedding approach.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ConcatMLPConfig:
    """Configuration for the Concat+MLP baseline.

    Parameters
    ----------
    d_input : int
        Dimension of each pooled embedding (RNA-FM = 640).
    hidden_dims : tuple[int, ...]
        Hidden layer dimensions for the MLP.
    dropout : float
        Dropout rate between layers.
    """

    d_input: int = 640
    hidden_dims: tuple = (512, 256)
    dropout: float = 0.3


class ConcatMLP(nn.Module):
    """Concat+MLP baseline for binary editing prediction.

    Concatenates pooled RNA-FM embeddings of the original and edited
    sequences, then passes through a 3-layer MLP.

    Input keys from batch dict:
        - ``pooled_orig``: (B, d_input) pooled original embedding
        - ``pooled_edited``: (B, d_input) pooled edited embedding

    Returns:
        dict with ``"binary_logit"``: (B, 1)
    """

    def __init__(self, config: ConcatMLPConfig | None = None):
        super().__init__()
        cfg = config or ConcatMLPConfig()
        self.config = cfg

        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_input * 2, cfg.hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dims[0], cfg.hidden_dims[1]),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dims[1], 1),
        )

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        pooled_orig = batch["pooled_orig"]      # (B, d_input)
        pooled_edited = batch["pooled_edited"]  # (B, d_input)

        x = torch.cat([pooled_orig, pooled_edited], dim=-1)  # (B, 2*d_input)
        logit = self.mlp(x)  # (B, 1)

        return {"binary_logit": logit}
