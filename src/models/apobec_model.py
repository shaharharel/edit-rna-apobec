"""Full APOBEC3A C-to-U editing prediction model.

Assembles the complete architecture:
    RNA Encoder(s)  -->  APOBECEditEmbedding  -->  Multi-modal Fusion  -->  Multi-task Heads

Supports:
  - RNA-FM (640-dim, 12-layer) as primary encoder
  - UTR-LM (128-dim, 6-layer) as optional secondary encoder
  - Optional GNN branch on RNA secondary structure graphs
  - APOBECEditEmbedding for structured C-to-U edit representations
  - APOBECMultiTaskHead with 6 prediction heads
  - APOBECMultiTaskLoss with uncertainty weighting and focal loss

The model is compatible with the APOBECDataset pipeline from
``src.data.apobec_dataset``.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .apobec_edit_embedding import (
    APOBECEditEmbedding,
    APOBECMultiTaskHead,
    APOBECMultiTaskLoss,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class APOBECModelConfig:
    """Full model configuration."""

    # Primary encoder
    primary_encoder: str = "rnafm"  # "rnafm" or "utrlm"
    d_model: int = 640  # 640 for RNA-FM, 128 for UTR-LM
    finetune_last_n: int = 0  # layers to fine-tune (0 = frozen)

    # Secondary encoder (optional)
    use_dual_encoder: bool = False
    secondary_encoder: str = "utrlm"
    d_model_secondary: int = 128

    # Edit embedding
    d_edit: int = 256
    edit_n_heads: int = 8
    use_structure_delta: bool = True

    # GNN branch (optional)
    use_gnn: bool = False
    gnn_hidden_dim: int = 128
    gnn_output_dim: int = 128
    gnn_layers: int = 4
    gnn_conv_type: str = "gat"

    # Multi-modal fusion
    d_fused: int = 512
    fusion_dropout: float = 0.2
    n_fusion_heads: int = 8

    # Multi-task heads
    n_tissues: int = 54
    head_dropout: float = 0.2

    # Loss
    focal_gamma: float = 2.0
    focal_alpha: float = 0.75

    # Training
    learning_rate: float = 1e-4
    encoder_lr_factor: float = 0.1  # encoder LR = lr * factor
    weight_decay: float = 1e-5


# ---------------------------------------------------------------------------
# Multi-modal fusion module
# ---------------------------------------------------------------------------

class MultiModalFusion(nn.Module):
    """Fuse encoder representations into a single vector.

    Combines:
    - Pooled sequence embedding from primary encoder
    - Edit embedding from APOBECEditEmbedding
    - (Optional) Secondary encoder embedding
    - (Optional) GNN structure embedding

    Uses gated attention fusion: each modality gets a learned gate weight
    before concatenation + projection.

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

        # Individual projections to a common dimension
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

        # Gating: per-modality scalar attention weights
        self.gate = nn.Sequential(
            nn.Linear(concat_dim, n_modalities),
            nn.Softmax(dim=-1),
        )

        # Final projection after gated fusion
        self.output_proj = nn.Sequential(
            nn.Linear(concat_dim, d_fused),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_fused),
        )

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

        # Gated attention (soft weighting of modalities)
        gate_weights = self.gate(concat)  # (B, n_modalities)

        # Weight each projected modality
        gated = []
        for i, p in enumerate(parts):
            gated.append(p * gate_weights[:, i : i + 1])

        gated_concat = torch.cat(gated, dim=-1)
        return self.output_proj(gated_concat)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class APOBECModel(nn.Module):
    """End-to-end APOBEC3A C-to-U editing prediction model.

    Architecture:
        1. RNA encoder(s) produce per-token embeddings
        2. APOBECEditEmbedding builds structured edit representations
        3. Multi-modal fusion combines encoder + edit + GNN embeddings
        4. Multi-task heads produce predictions

    The model accepts batched output from ``apobec_collate_fn`` and
    returns a prediction dict compatible with ``APOBECMultiTaskLoss``.

    Parameters
    ----------
    config : APOBECModelConfig
        Model configuration.
    primary_encoder : nn.Module or None
        Pre-loaded primary RNA encoder (RNAFMEncoder or UTRLMEmbedder).
        If None, will be created from config.
    secondary_encoder : nn.Module or None
        Pre-loaded secondary encoder (UTRLMEmbedder).
    gnn_encoder : nn.Module or None
        Pre-loaded GNN encoder (RNAStructureGNN).
    """

    def __init__(
        self,
        config: Optional[APOBECModelConfig] = None,
        primary_encoder: Optional[nn.Module] = None,
        secondary_encoder: Optional[nn.Module] = None,
        gnn_encoder: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.config = config or APOBECModelConfig()

        # --- Primary encoder ---
        if primary_encoder is not None:
            self.primary_encoder = primary_encoder
        else:
            self.primary_encoder = self._build_primary_encoder()

        # --- Secondary encoder ---
        self.secondary_encoder = None
        if self.config.use_dual_encoder:
            if secondary_encoder is not None:
                self.secondary_encoder = secondary_encoder
            else:
                self.secondary_encoder = self._build_secondary_encoder()

        # --- GNN encoder ---
        self.gnn_encoder = None
        if self.config.use_gnn:
            if gnn_encoder is not None:
                self.gnn_encoder = gnn_encoder
            else:
                self.gnn_encoder = self._build_gnn_encoder()

        # --- Edit embedding ---
        self.edit_embedding = APOBECEditEmbedding(
            d_model=self.config.d_model,
            d_edit=self.config.d_edit,
            n_heads=self.config.edit_n_heads,
            dropout=self.config.fusion_dropout,
            use_structure_delta=self.config.use_structure_delta,
            use_dual_encoder=self.config.use_dual_encoder,
            d_model_secondary=self.config.d_model_secondary,
        )

        # --- Multi-modal fusion ---
        gnn_dim = self.config.gnn_output_dim if self.config.use_gnn else 0
        sec_dim = self.config.d_model_secondary if self.config.use_dual_encoder else 0

        self.fusion = MultiModalFusion(
            d_model=self.config.d_model,
            d_edit=self.config.d_edit,
            d_fused=self.config.d_fused,
            d_model_secondary=sec_dim,
            d_gnn=gnn_dim,
            dropout=self.config.fusion_dropout,
        )

        # --- Multi-task prediction heads ---
        self.heads = APOBECMultiTaskHead(
            d_fused=self.config.d_fused,
            d_edit=self.config.d_edit,
            n_tissues=self.config.n_tissues,
            dropout=self.config.head_dropout,
        )

        # --- Multi-task loss ---
        self.loss_fn = APOBECMultiTaskLoss(
            focal_gamma=self.config.focal_gamma,
            focal_alpha=self.config.focal_alpha,
        )

    def _build_primary_encoder(self) -> nn.Module:
        """Build the primary RNA encoder from config."""
        if self.config.primary_encoder == "rnafm":
            from ..embedding.rna_fm_encoder import RNAFMEncoder
            return RNAFMEncoder(
                finetune_last_n=self.config.finetune_last_n,
            )
        elif self.config.primary_encoder == "utrlm":
            from ..embedding.utrlm import load_utrlm
            return load_utrlm(
                trainable=(self.config.finetune_last_n > 0),
            )
        else:
            raise ValueError(f"Unknown primary encoder: {self.config.primary_encoder}")

    def _build_secondary_encoder(self) -> nn.Module:
        """Build the secondary encoder from config."""
        if self.config.secondary_encoder == "utrlm":
            from ..embedding.utrlm import load_utrlm
            return load_utrlm(trainable=False)
        else:
            raise ValueError(
                f"Unknown secondary encoder: {self.config.secondary_encoder}"
            )

    def _build_gnn_encoder(self) -> nn.Module:
        """Build the GNN encoder from config."""
        from .structure_gnn import RNAStructureGNN
        return RNAStructureGNN(
            hidden_dim=self.config.gnn_hidden_dim,
            output_dim=self.config.gnn_output_dim,
            num_layers=self.config.gnn_layers,
            conv_type=self.config.gnn_conv_type,
        )

    def _encode_primary(self, sequences: List[str]) -> Dict[str, torch.Tensor]:
        """Run the primary encoder on a batch of sequences.

        Returns per-token embeddings and a pooled representation.
        """
        if self.config.primary_encoder == "rnafm":
            result = self.primary_encoder(sequences=sequences)
            return {
                "tokens": result["embeddings"],  # (B, L, 640)
                "pooled": result["pooled"],       # (B, 640)
            }
        else:
            # UTR-LM path
            tokens = self.primary_encoder(sequences, return_all_tokens=True)
            # Mean pool for pooled representation
            pooled = tokens.mean(dim=1)
            return {
                "tokens": tokens,
                "pooled": pooled,
            }

    def _encode_secondary(self, sequences: List[str]) -> Dict[str, torch.Tensor]:
        """Run the secondary encoder."""
        tokens = self.secondary_encoder(sequences, return_all_tokens=True)
        pooled = tokens.mean(dim=1)
        return {
            "tokens": tokens,
            "pooled": pooled,
        }

    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """Full forward pass.

        Parameters
        ----------
        batch : dict
            Output from ``apobec_collate_fn`` with keys:
            - sequences : list[str], length B
            - edit_pos : (B,) long tensor
            - flanking_context : (B,) long tensor
            - concordance_features : (B, 5) float tensor
            - structure_delta : (B, 7) float tensor
            - targets : dict of label tensors (during training)
            - site_ids : list[str]

        Returns
        -------
        dict with:
            - predictions : dict from APOBECMultiTaskHead
            - edit_embedding : (B, d_edit) for analysis
            - fused : (B, d_fused) fused representation
        """
        sequences = batch["sequences"]
        edit_pos = batch["edit_pos"]
        flanking_context = batch["flanking_context"]
        concordance_features = batch["concordance_features"]
        structure_delta = batch["structure_delta"]

        device = edit_pos.device

        # --- Step 1: Encode sequences ---
        primary_out = self._encode_primary(sequences)
        f_background = primary_out["tokens"]   # (B, L, d_model)
        primary_pooled = primary_out["pooled"]  # (B, d_model)

        # Create edited sequence representations (C -> U at edit_pos)
        # For the causal framework, we encode both original and edited sequences
        edited_sequences = []
        for seq, pos in zip(sequences, edit_pos.tolist()):
            seq_list = list(seq)
            p = min(pos, len(seq_list) - 1)
            if seq_list[p].upper() == "C":
                seq_list[p] = "U"
            edited_sequences.append("".join(seq_list))

        edited_out = self._encode_primary(edited_sequences)
        f_edited = edited_out["tokens"]  # (B, L_edited, d_model)

        # Ensure f_edited matches f_background length (they should for SNV)
        min_len = min(f_background.shape[1], f_edited.shape[1])
        f_background = f_background[:, :min_len, :]
        f_edited = f_edited[:, :min_len, :]

        # Sequence mask (all True for fixed-length windows, could be padded)
        seq_mask = torch.ones(
            f_background.shape[0], f_background.shape[1],
            dtype=torch.bool, device=device
        )

        # --- Step 2: Secondary encoder (optional) ---
        f_background_secondary = None
        secondary_pooled = None
        if self.secondary_encoder is not None:
            sec_out = self._encode_secondary(sequences)
            f_background_secondary = sec_out["tokens"]
            secondary_pooled = sec_out["pooled"]
            # Match length
            sec_len = min(f_background_secondary.shape[1], min_len)
            f_background_secondary = f_background_secondary[:, :sec_len, :]

        # --- Step 3: GNN (optional) ---
        gnn_emb = None
        if self.gnn_encoder is not None and "graph_data" in batch:
            gnn_emb = self.gnn_encoder(batch["graph_data"])

        # --- Step 4: APOBEC edit embedding ---
        edit_emb = self.edit_embedding(
            f_background=f_background,
            f_edited=f_edited,
            edit_pos=edit_pos,
            flanking_context=flanking_context,
            structure_delta=structure_delta,
            concordance_features=concordance_features,
            f_background_secondary=f_background_secondary,
            seq_mask=seq_mask,
        )

        # --- Step 5: Multi-modal fusion ---
        fused = self.fusion(
            primary_pooled=primary_pooled,
            edit_emb=edit_emb,
            secondary_pooled=secondary_pooled,
            gnn_emb=gnn_emb,
        )

        # --- Step 6: Multi-task prediction ---
        predictions = self.heads(fused, edit_emb)

        return {
            "predictions": predictions,
            "edit_embedding": edit_emb,
            "fused": fused,
        }

    def compute_loss(
        self,
        model_output: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> tuple:
        """Compute the multi-task loss.

        Parameters
        ----------
        model_output : dict from forward()
        targets : dict of target tensors from the batch

        Returns
        -------
        total_loss : scalar tensor
        task_losses : dict of per-task losses
        """
        return self.loss_fn(model_output["predictions"], targets)

    def get_parameter_groups(self) -> List[Dict]:
        """Get parameter groups with different learning rates.

        Returns optimizer parameter groups:
        - Encoder parameters: lr * encoder_lr_factor (slow)
        - Everything else: lr (normal)
        """
        cfg = self.config

        encoder_params = []
        other_params = []

        # Collect encoder parameters
        encoder_modules = [self.primary_encoder]
        if self.secondary_encoder is not None:
            encoder_modules.append(self.secondary_encoder)

        encoder_param_ids = set()
        for module in encoder_modules:
            for p in module.parameters():
                if p.requires_grad:
                    encoder_params.append(p)
                    encoder_param_ids.add(id(p))

        # Everything else
        for p in self.parameters():
            if p.requires_grad and id(p) not in encoder_param_ids:
                other_params.append(p)

        groups = [
            {
                "params": other_params,
                "lr": cfg.learning_rate,
                "weight_decay": cfg.weight_decay,
            },
        ]

        if encoder_params:
            groups.append({
                "params": encoder_params,
                "lr": cfg.learning_rate * cfg.encoder_lr_factor,
                "weight_decay": cfg.weight_decay,
            })

        return groups

    def count_parameters(self) -> Dict[str, int]:
        """Count trainable and total parameters by component."""
        counts = {}
        for name, module in [
            ("primary_encoder", self.primary_encoder),
            ("secondary_encoder", self.secondary_encoder),
            ("gnn_encoder", self.gnn_encoder),
            ("edit_embedding", self.edit_embedding),
            ("fusion", self.fusion),
            ("heads", self.heads),
            ("loss_fn", self.loss_fn),
        ]:
            if module is None:
                continue
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            counts[name] = {"total": total, "trainable": trainable}

        counts["model_total"] = {
            "total": sum(p.numel() for p in self.parameters()),
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }
        return counts


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def create_rnafm_model(
    finetune_last_n: int = 0,
    use_dual_encoder: bool = False,
    use_gnn: bool = False,
    use_structure_delta: bool = True,
    **kwargs,
) -> APOBECModel:
    """Create an APOBEC model with RNA-FM as primary encoder.

    This is the recommended configuration for best performance.

    Args:
        finetune_last_n: Number of RNA-FM layers to fine-tune (0=frozen).
        use_dual_encoder: Whether to add UTR-LM as secondary encoder.
        use_gnn: Whether to add structure GNN branch.
        use_structure_delta: Whether to use RNAplfold structure delta features.
        **kwargs: Additional config overrides.

    Returns:
        Configured APOBECModel.
    """
    config = APOBECModelConfig(
        primary_encoder="rnafm",
        d_model=640,
        finetune_last_n=finetune_last_n,
        use_dual_encoder=use_dual_encoder,
        secondary_encoder="utrlm",
        d_model_secondary=128,
        use_gnn=use_gnn,
        use_structure_delta=use_structure_delta,
        **{k: v for k, v in kwargs.items() if hasattr(APOBECModelConfig, k)},
    )
    return APOBECModel(config)


def create_utrlm_model(
    use_structure_delta: bool = True,
    **kwargs,
) -> APOBECModel:
    """Create an APOBEC model with UTR-LM as primary encoder.

    Lighter weight alternative to RNA-FM. Useful for rapid prototyping
    and when 5' UTR context is the primary focus.

    Args:
        use_structure_delta: Whether to use structure delta features.
        **kwargs: Additional config overrides.

    Returns:
        Configured APOBECModel.
    """
    config = APOBECModelConfig(
        primary_encoder="utrlm",
        d_model=128,
        finetune_last_n=0,
        use_dual_encoder=False,
        use_gnn=False,
        use_structure_delta=use_structure_delta,
        d_edit=128,
        d_fused=256,
        edit_n_heads=4,
        **{k: v for k, v in kwargs.items() if hasattr(APOBECModelConfig, k)},
    )
    return APOBECModel(config)
