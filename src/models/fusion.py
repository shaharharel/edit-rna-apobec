"""Cross-attention fusion module for multi-modal representations.

Fuses representations from multiple modalities:
- Primary RNA encoder (RNA-FM, 640-dim)
- Edit embedding (APOBEC-specific, d_edit-dim)
- Optional secondary encoder (UTR-LM, 128-dim)
- Optional GNN structure encoder (128-dim)

Uses gated attention fusion: each modality is projected to a common dimension,
weighted by learned softmax gates, then concatenated and projected to d_fused.
"""

import torch
import torch.nn as nn
from typing import Optional


class GatedModalityFusion(nn.Module):
    """Gated attention fusion for multi-modal representations.

    Each modality is projected to a common dimension, then weighted by
    a learned softmax gate before concatenation and final projection.
    This allows the model to adaptively weight the contribution of each
    modality depending on the input.

    Parameters
    ----------
    d_model : int
        Dimension of primary encoder output.
    d_edit : int
        Dimension of edit embedding.
    d_fused : int
        Output fused dimension.
    d_model_secondary : int
        Dimension of secondary encoder (0 if not used).
    d_gnn : int
        Dimension of GNN output (0 if not used).
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        d_model: int,
        d_edit: int,
        d_fused: int,
        d_model_secondary: int = 0,
        d_gnn: int = 0,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_edit = d_edit
        self.d_model_secondary = d_model_secondary
        self.d_gnn = d_gnn

        proj_dim = d_fused // 2

        self.primary_proj = nn.Sequential(
            nn.Linear(d_model, proj_dim),
            nn.GELU(),
            nn.LayerNorm(proj_dim),
        )

        self.edit_proj = nn.Sequential(
            nn.Linear(d_edit, proj_dim),
            nn.GELU(),
            nn.LayerNorm(proj_dim),
        )

        n_modalities = 2
        concat_dim = proj_dim * 2

        if d_model_secondary > 0:
            self.secondary_proj = nn.Sequential(
                nn.Linear(d_model_secondary, proj_dim),
                nn.GELU(),
                nn.LayerNorm(proj_dim),
            )
            n_modalities += 1
            concat_dim += proj_dim

        if d_gnn > 0:
            self.gnn_proj = nn.Sequential(
                nn.Linear(d_gnn, proj_dim),
                nn.GELU(),
                nn.LayerNorm(proj_dim),
            )
            n_modalities += 1
            concat_dim += proj_dim

        self.gate = nn.Sequential(
            nn.Linear(concat_dim, n_modalities),
            nn.Softmax(dim=-1),
        )

        self.output_proj = nn.Sequential(
            nn.Linear(concat_dim, d_fused),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_fused),
        )

        self._n_modalities = n_modalities
        self._proj_dim = proj_dim

    def forward(
        self,
        primary_pooled: torch.Tensor,
        edit_emb: torch.Tensor,
        secondary_pooled: Optional[torch.Tensor] = None,
        gnn_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fuse multi-modal representations.

        Parameters
        ----------
        primary_pooled : (batch, d_model)
        edit_emb : (batch, d_edit)
        secondary_pooled : (batch, d_model_secondary) or None
        gnn_emb : (batch, d_gnn) or None

        Returns
        -------
        fused : (batch, d_fused)
        """
        proj_primary = self.primary_proj(primary_pooled)
        proj_edit = self.edit_proj(edit_emb)
        parts = [proj_primary, proj_edit]

        if self.d_model_secondary > 0 and secondary_pooled is not None:
            parts.append(self.secondary_proj(secondary_pooled))

        if self.d_gnn > 0 and gnn_emb is not None:
            parts.append(self.gnn_proj(gnn_emb))

        concat = torch.cat(parts, dim=-1)

        gate_weights = self.gate(concat)  # (B, n_modalities)

        gated = []
        for i, p in enumerate(parts):
            gated.append(p * gate_weights[:, i : i + 1])

        gated_concat = torch.cat(gated, dim=-1)
        return self.output_proj(gated_concat)

    @property
    def n_modalities(self) -> int:
        return self._n_modalities

    def get_gate_weights(
        self,
        primary_pooled: torch.Tensor,
        edit_emb: torch.Tensor,
        secondary_pooled: Optional[torch.Tensor] = None,
        gnn_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return the per-modality gate weights for interpretability.

        Returns (batch, n_modalities) softmax weights.
        """
        proj_primary = self.primary_proj(primary_pooled)
        proj_edit = self.edit_proj(edit_emb)
        parts = [proj_primary, proj_edit]

        if self.d_model_secondary > 0 and secondary_pooled is not None:
            parts.append(self.secondary_proj(secondary_pooled))
        if self.d_gnn > 0 and gnn_emb is not None:
            parts.append(self.gnn_proj(gnn_emb))

        concat = torch.cat(parts, dim=-1)
        return self.gate(concat)


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion between sequence and edit representations.

    Uses bidirectional cross-attention: sequence attends to edit and
    edit attends to sequence. This enables richer interaction than
    gated concatenation.

    Parameters
    ----------
    d_seq : int
        Sequence representation dimension.
    d_edit : int
        Edit embedding dimension.
    d_fused : int
        Output dimension.
    n_heads : int
        Number of attention heads.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        d_seq: int,
        d_edit: int,
        d_fused: int,
        n_heads: int = 8,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.d_common = d_fused

        # Project to common dim
        self.seq_proj = nn.Linear(d_seq, d_fused)
        self.edit_proj = nn.Linear(d_edit, d_fused)

        # Seq -> Edit cross-attention
        self.seq_to_edit = nn.MultiheadAttention(
            embed_dim=d_fused,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_seq = nn.LayerNorm(d_fused)

        # Edit -> Seq cross-attention
        self.edit_to_seq = nn.MultiheadAttention(
            embed_dim=d_fused,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_edit = nn.LayerNorm(d_fused)

        # Final projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_fused * 2, d_fused),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_fused),
        )

    def forward(
        self,
        seq_repr: torch.Tensor,
        edit_repr: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse sequence and edit representations via cross-attention.

        Parameters
        ----------
        seq_repr : (batch, d_seq)
            Pooled sequence representation.
        edit_repr : (batch, d_edit)
            Edit embedding.

        Returns
        -------
        fused : (batch, d_fused)
        """
        seq_h = self.seq_proj(seq_repr).unsqueeze(1)  # (B, 1, d_fused)
        edit_h = self.edit_proj(edit_repr).unsqueeze(1)  # (B, 1, d_fused)

        # Bidirectional cross-attention
        seq_attended, _ = self.seq_to_edit(seq_h, edit_h, edit_h)
        seq_out = self.norm_seq(seq_attended.squeeze(1) + seq_h.squeeze(1))

        edit_attended, _ = self.edit_to_seq(edit_h, seq_h, seq_h)
        edit_out = self.norm_edit(edit_attended.squeeze(1) + edit_h.squeeze(1))

        combined = torch.cat([seq_out, edit_out], dim=-1)
        return self.output_proj(combined)
