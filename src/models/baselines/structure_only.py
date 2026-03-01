"""Structure-Only baseline: ViennaRNA delta features only -> MLP.

This baseline uses only the 7-dimensional structure delta features
computed from RNAplfold (change in pairing probability, accessibility,
entropy, MFE, etc.) to predict editing status. It measures how much
structural information alone contributes to the prediction, without
any sequence embedding.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class StructureOnlyConfig:
    """Configuration for the Structure-Only baseline.

    Parameters
    ----------
    d_input : int
        Dimension of structure delta features (default 7).
    hidden_dims : tuple[int, ...]
        Hidden layer dimensions for the MLP.
    dropout : float
        Dropout rate between layers.
    """

    d_input: int = 7
    hidden_dims: tuple = (64, 32)
    dropout: float = 0.3


class StructureOnlyBaseline(nn.Module):
    """Structure-Only baseline for binary editing prediction.

    Uses only the ViennaRNA structure delta features (7-dim vector
    capturing changes in pairing, accessibility, entropy, and MFE
    due to the C-to-U edit) as input to a small MLP.

    Input keys from batch dict:
        - ``structure_delta``: (B, 7) structure delta features

    Returns:
        dict with ``"binary_logit"``: (B, 1)
    """

    def __init__(self, config: StructureOnlyConfig | None = None):
        super().__init__()
        cfg = config or StructureOnlyConfig()
        self.config = cfg

        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_input, cfg.hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dims[0], cfg.hidden_dims[1]),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dims[1], 1),
        )

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        structure_delta = batch["structure_delta"]  # (B, 7)

        logit = self.mlp(structure_delta)  # (B, 1)

        return {"binary_logit": logit}
