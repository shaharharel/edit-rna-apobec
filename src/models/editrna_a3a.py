"""EditRNA-A3A: End-to-end APOBEC3A C-to-U editing prediction model.

Composes all modular components into the full architecture:

    Encoders  -->  EditEmbedding  -->  Fusion  -->  PredictionHeads

Supports flexible encoder configurations:
  - RNA-FM (640-dim, 12-layer) as primary encoder
  - UTR-LM (128-dim, 6-layer) as optional secondary encoder
  - Structure GNN on RNA secondary structure graphs
  - Contact Map ViT on base-pair probability matrices

The model accepts batched output from ``apobec_collate_fn`` and returns
a prediction dict compatible with ``APOBECMultiTaskLoss``.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .apobec_edit_embedding import APOBECEditEmbedding
from .fusion import GatedModalityFusion
from .prediction_heads import APOBECMultiTaskHead, APOBECMultiTaskLoss

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class EditRNAConfig:
    """Configuration for the EditRNA-A3A model.

    Groups all hyperparameters for encoders, edit embedding, fusion,
    prediction heads, loss, and training.
    """

    # Primary encoder
    primary_encoder: str = "rnafm"  # "rnafm", "utrlm", or "mock"
    d_model: int = 640  # 640 for RNA-FM, 128 for UTR-LM
    finetune_last_n: int = 0

    # Secondary encoder (optional)
    use_dual_encoder: bool = False
    secondary_encoder: str = "utrlm"
    d_model_secondary: int = 128

    # Edit embedding
    d_edit: int = 256
    edit_n_heads: int = 8
    use_structure_delta: bool = True

    # Structure GNN (optional)
    use_gnn: bool = False
    gnn_hidden_dim: int = 128
    gnn_output_dim: int = 128
    gnn_layers: int = 4
    gnn_conv_type: str = "gat"

    # Contact map ViT (optional)
    use_contact_vit: bool = False
    contact_max_len: int = 512
    contact_patch_size: int = 16
    contact_d_model: int = 256
    contact_n_heads: int = 8
    contact_n_layers: int = 4
    contact_output_dim: int = 128

    # Multi-modal fusion
    d_fused: int = 512
    fusion_dropout: float = 0.2
    n_fusion_heads: int = 8

    # Multi-task heads
    n_tissues: int = 54
    head_dropout: float = 0.2

    # Loss
    focal_gamma: float = 2.0
    focal_alpha_binary: float = 0.75
    focal_alpha_conservation: float = 0.85

    # Training
    learning_rate: float = 1e-4
    encoder_lr_factor: float = 0.1
    weight_decay: float = 1e-5


# ---------------------------------------------------------------------------
# Main Model
# ---------------------------------------------------------------------------


class EditRNA_A3A(nn.Module):
    """End-to-end APOBEC3A C-to-U editing prediction model.

    Architecture::

        1. RNA encoder(s) produce per-token embeddings
        2. APOBECEditEmbedding builds structured edit representations
        3. GatedModalityFusion combines encoder + edit + GNN + ViT embeddings
        4. APOBECMultiTaskHead produces 12 prediction outputs (11 tasks + effect)

    Parameters
    ----------
    config : EditRNAConfig or None
        Full model configuration. Uses defaults if None.
    primary_encoder : nn.Module or None
        Pre-loaded primary encoder. Built from config if None.
    secondary_encoder : nn.Module or None
        Pre-loaded secondary encoder.
    gnn_encoder : nn.Module or None
        Pre-loaded structure GNN encoder.
    contact_encoder : nn.Module or None
        Pre-loaded contact map ViT encoder.
    """

    def __init__(
        self,
        config: Optional[EditRNAConfig] = None,
        primary_encoder: Optional[nn.Module] = None,
        secondary_encoder: Optional[nn.Module] = None,
        gnn_encoder: Optional[nn.Module] = None,
        contact_encoder: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.config = config or EditRNAConfig()
        cfg = self.config

        # --- Primary encoder ---
        if primary_encoder is not None:
            self.primary_encoder = primary_encoder
        else:
            self.primary_encoder = self._build_primary_encoder()

        # --- Secondary encoder ---
        self.secondary_encoder = None
        if cfg.use_dual_encoder:
            self.secondary_encoder = secondary_encoder or self._build_secondary_encoder()

        # --- Structure GNN ---
        self.gnn_encoder = None
        if cfg.use_gnn:
            self.gnn_encoder = gnn_encoder or self._build_gnn_encoder()

        # --- Contact map ViT ---
        self.contact_encoder = None
        if cfg.use_contact_vit:
            self.contact_encoder = contact_encoder or self._build_contact_encoder()

        # --- Edit embedding ---
        self.edit_embedding = APOBECEditEmbedding(
            d_model=cfg.d_model,
            d_edit=cfg.d_edit,
            n_heads=cfg.edit_n_heads,
            dropout=cfg.fusion_dropout,
            use_structure_delta=cfg.use_structure_delta,
            use_dual_encoder=cfg.use_dual_encoder,
            d_model_secondary=cfg.d_model_secondary,
        )

        # --- Multi-modal fusion ---
        # Compute auxiliary modality dimensions
        gnn_dim = cfg.gnn_output_dim if cfg.use_gnn else 0
        contact_dim = cfg.contact_output_dim if cfg.use_contact_vit else 0
        sec_dim = cfg.d_model_secondary if cfg.use_dual_encoder else 0
        # GNN and contact map are both structure modalities; sum their dims
        # as d_gnn in the fusion module
        structure_dim = gnn_dim + contact_dim

        self.fusion = GatedModalityFusion(
            d_model=cfg.d_model,
            d_edit=cfg.d_edit,
            d_fused=cfg.d_fused,
            d_model_secondary=sec_dim,
            d_gnn=structure_dim,
            dropout=cfg.fusion_dropout,
        )

        # --- Multi-task prediction heads ---
        self.heads = APOBECMultiTaskHead(
            d_fused=cfg.d_fused,
            d_edit=cfg.d_edit,
            n_tissues=cfg.n_tissues,
            dropout=cfg.head_dropout,
        )

        # --- Multi-task loss ---
        self.loss_fn = APOBECMultiTaskLoss(
            focal_gamma=cfg.focal_gamma,
            focal_alpha_binary=cfg.focal_alpha_binary,
            focal_alpha_conservation=cfg.focal_alpha_conservation,
        )

    # -----------------------------------------------------------------
    # Encoder builders
    # -----------------------------------------------------------------

    def _build_primary_encoder(self) -> nn.Module:
        cfg = self.config
        if cfg.primary_encoder == "rnafm":
            from .encoders import RNAFMEncoderWrapper

            return RNAFMEncoderWrapper(finetune_last_n=cfg.finetune_last_n)
        elif cfg.primary_encoder == "utrlm":
            from ..embedding.utrlm import load_utrlm

            return load_utrlm(trainable=(cfg.finetune_last_n > 0))
        elif cfg.primary_encoder == "mock":
            from .encoders import MockRNAEncoder

            return MockRNAEncoder(d_model=cfg.d_model)
        elif cfg.primary_encoder == "cached":
            # CachedRNAEncoder must be passed as primary_encoder argument
            raise ValueError(
                "CachedRNAEncoder must be provided via the primary_encoder "
                "argument, not built automatically."
            )
        else:
            raise ValueError(f"Unknown primary encoder: {cfg.primary_encoder}")

    def _build_secondary_encoder(self) -> nn.Module:
        cfg = self.config
        if cfg.secondary_encoder == "utrlm":
            from ..embedding.utrlm import load_utrlm

            return load_utrlm(trainable=False)
        else:
            raise ValueError(f"Unknown secondary encoder: {cfg.secondary_encoder}")

    def _build_gnn_encoder(self) -> nn.Module:
        from .encoders import StructureGNNEncoder

        cfg = self.config
        return StructureGNNEncoder(
            hidden_dim=cfg.gnn_hidden_dim,
            output_dim=cfg.gnn_output_dim,
            num_layers=cfg.gnn_layers,
            conv_type=cfg.gnn_conv_type,
        )

    def _build_contact_encoder(self) -> nn.Module:
        from .encoders import ContactMapViT

        cfg = self.config
        return ContactMapViT(
            max_len=cfg.contact_max_len,
            patch_size=cfg.contact_patch_size,
            d_model=cfg.contact_d_model,
            n_heads=cfg.contact_n_heads,
            n_layers=cfg.contact_n_layers,
            output_dim=cfg.contact_output_dim,
        )

    # -----------------------------------------------------------------
    # Encoding helpers
    # -----------------------------------------------------------------

    def _encode_primary(
        self,
        sequences: list[str],
        site_ids: Optional[list[str]] = None,
        edited: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Run the primary encoder.

        Returns dict with "tokens" (per-token) and "pooled" keys.
        Supports cached encoder mode when site_ids are provided.
        """
        cfg = self.config
        # Cached encoder path
        if site_ids is not None and cfg.primary_encoder == "cached":
            result = self.primary_encoder(site_ids=site_ids, edited=edited)
            return {"tokens": result["embeddings"], "pooled": result["pooled"]}
        # Standard encoder path
        if cfg.primary_encoder in ("rnafm", "mock"):
            result = self.primary_encoder(sequences=sequences)
            return {"tokens": result["embeddings"], "pooled": result["pooled"]}
        else:
            tokens = self.primary_encoder(sequences, return_all_tokens=True)
            pooled = tokens.mean(dim=1)
            return {"tokens": tokens, "pooled": pooled}

    def _encode_secondary(self, sequences: list[str]) -> dict[str, torch.Tensor]:
        tokens = self.secondary_encoder(sequences, return_all_tokens=True)
        pooled = tokens.mean(dim=1)
        return {"tokens": tokens, "pooled": pooled}

    def _make_edited_sequences(
        self, sequences: list[str], edit_pos: torch.Tensor
    ) -> list[str]:
        """Apply C->U edits at specified positions."""
        edited = []
        for seq, pos in zip(sequences, edit_pos.tolist()):
            chars = list(seq)
            p = min(int(pos), len(chars) - 1)
            if chars[p].upper() == "C":
                chars[p] = "U"
            edited.append("".join(chars))
        return edited

    # -----------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
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
            - graph_data : PyG Batch (if use_gnn)
            - contact_maps : (B, L, L) float (if use_contact_vit)

        Returns
        -------
        dict with:
            "predictions"    : dict from APOBECMultiTaskHead
            "edit_embedding" : (B, d_edit) for analysis/interpretability
            "fused"          : (B, d_fused)
        """
        sequences = batch["sequences"]
        edit_pos = batch["edit_pos"]
        flanking_context = batch["flanking_context"]
        concordance_features = batch["concordance_features"]
        structure_delta = batch["structure_delta"]
        site_ids = batch.get("site_ids")
        device = edit_pos.device

        # --- Step 1: Encode original sequences ---
        primary_out = self._encode_primary(sequences, site_ids=site_ids, edited=False)
        f_background = primary_out["tokens"]   # (B, L, d_model)
        primary_pooled = primary_out["pooled"]  # (B, d_model)

        # --- Step 2: Encode edited sequences (C->U) ---
        edited_sequences = self._make_edited_sequences(sequences, edit_pos)
        edited_out = self._encode_primary(edited_sequences, site_ids=site_ids, edited=True)
        f_edited = edited_out["tokens"]

        # Align lengths (should match for single-nucleotide edits)
        min_len = min(f_background.shape[1], f_edited.shape[1])
        f_background = f_background[:, :min_len, :]
        f_edited = f_edited[:, :min_len, :]

        seq_mask = torch.ones(
            f_background.shape[0], min_len, dtype=torch.bool, device=device
        )

        # --- Step 3: Secondary encoder (optional) ---
        f_background_secondary = None
        secondary_pooled = None
        if self.secondary_encoder is not None:
            sec_out = self._encode_secondary(sequences)
            f_background_secondary = sec_out["tokens"]
            secondary_pooled = sec_out["pooled"]
            sec_len = min(f_background_secondary.shape[1], min_len)
            f_background_secondary = f_background_secondary[:, :sec_len, :]

        # --- Step 4: Structure GNN (optional) ---
        gnn_pooled = None
        if self.gnn_encoder is not None and "graph_data" in batch:
            gnn_out = self.gnn_encoder(batch["graph_data"])
            gnn_pooled = gnn_out["pooled"]

        # --- Step 5: Contact map ViT (optional) ---
        contact_pooled = None
        if self.contact_encoder is not None and "contact_maps" in batch:
            contact_out = self.contact_encoder(batch["contact_maps"])
            contact_pooled = contact_out["pooled"]

        # Merge structure modalities for fusion
        structure_pooled = None
        if gnn_pooled is not None and contact_pooled is not None:
            structure_pooled = torch.cat([gnn_pooled, contact_pooled], dim=-1)
        elif gnn_pooled is not None:
            structure_pooled = gnn_pooled
        elif contact_pooled is not None:
            structure_pooled = contact_pooled

        # --- Step 6: APOBEC edit embedding ---
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

        # --- Step 7: Multi-modal fusion ---
        fused = self.fusion(
            primary_pooled=primary_pooled,
            edit_emb=edit_emb,
            secondary_pooled=secondary_pooled,
            gnn_emb=structure_pooled,
        )

        # --- Step 8: Multi-task prediction ---
        predictions = self.heads(fused, edit_emb)

        return {
            "predictions": predictions,
            "edit_embedding": edit_emb,
            "fused": fused,
        }

    # -----------------------------------------------------------------
    # Loss computation
    # -----------------------------------------------------------------

    def compute_loss(
        self,
        model_output: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the multi-task loss.

        Returns (total_loss, per_task_losses_dict).
        """
        return self.loss_fn(model_output["predictions"], targets)

    # -----------------------------------------------------------------
    # Training utilities
    # -----------------------------------------------------------------

    def get_parameter_groups(self) -> list[dict]:
        """Return optimizer parameter groups with differential learning rates.

        Encoder parameters get ``lr * encoder_lr_factor`` (slow);
        everything else gets ``lr`` (normal).
        """
        cfg = self.config

        encoder_modules = [self.primary_encoder]
        if self.secondary_encoder is not None:
            encoder_modules.append(self.secondary_encoder)

        encoder_param_ids: set[int] = set()
        encoder_params: list[nn.Parameter] = []
        for module in encoder_modules:
            for p in module.parameters():
                if p.requires_grad:
                    encoder_params.append(p)
                    encoder_param_ids.add(id(p))

        other_params = [
            p for p in self.parameters()
            if p.requires_grad and id(p) not in encoder_param_ids
        ]

        groups = [
            {
                "params": other_params,
                "lr": cfg.learning_rate,
                "weight_decay": cfg.weight_decay,
            },
        ]
        if encoder_params:
            groups.append(
                {
                    "params": encoder_params,
                    "lr": cfg.learning_rate * cfg.encoder_lr_factor,
                    "weight_decay": cfg.weight_decay,
                }
            )
        return groups

    def count_parameters(self) -> dict[str, dict[str, int]]:
        """Count trainable and total parameters by component."""
        counts = {}
        components = [
            ("primary_encoder", self.primary_encoder),
            ("secondary_encoder", self.secondary_encoder),
            ("gnn_encoder", self.gnn_encoder),
            ("contact_encoder", self.contact_encoder),
            ("edit_embedding", self.edit_embedding),
            ("fusion", self.fusion),
            ("heads", self.heads),
            ("loss_fn", self.loss_fn),
        ]
        for name, module in components:
            if module is None:
                continue
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            counts[name] = {"total": total, "trainable": trainable}

        counts["model_total"] = {
            "total": sum(p.numel() for p in self.parameters()),
            "trainable": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
        }
        return counts


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def create_editrna_rnafm(
    finetune_last_n: int = 0,
    use_dual_encoder: bool = False,
    use_gnn: bool = False,
    use_contact_vit: bool = False,
    use_structure_delta: bool = True,
    **kwargs,
) -> EditRNA_A3A:
    """Create an EditRNA-A3A model with RNA-FM as primary encoder.

    This is the recommended configuration for best performance.
    """
    config = EditRNAConfig(
        primary_encoder="rnafm",
        d_model=640,
        finetune_last_n=finetune_last_n,
        use_dual_encoder=use_dual_encoder,
        use_gnn=use_gnn,
        use_contact_vit=use_contact_vit,
        use_structure_delta=use_structure_delta,
        **{k: v for k, v in kwargs.items() if hasattr(EditRNAConfig, k)},
    )
    return EditRNA_A3A(config)


def create_editrna_mock(
    d_model: int = 640,
    use_structure_delta: bool = True,
    **kwargs,
) -> EditRNA_A3A:
    """Create an EditRNA-A3A model with a mock encoder for testing.

    Allows full pipeline testing without downloading RNA-FM weights.
    """
    config = EditRNAConfig(
        primary_encoder="mock",
        d_model=d_model,
        finetune_last_n=0,
        use_dual_encoder=False,
        use_gnn=False,
        use_contact_vit=False,
        use_structure_delta=use_structure_delta,
        **{k: v for k, v in kwargs.items() if hasattr(EditRNAConfig, k)},
    )
    return EditRNA_A3A(config)


def create_editrna_utrlm(
    use_structure_delta: bool = True,
    **kwargs,
) -> EditRNA_A3A:
    """Create an EditRNA-A3A model with UTR-LM as primary encoder.

    Lighter weight alternative, useful for rapid prototyping.
    """
    config = EditRNAConfig(
        primary_encoder="utrlm",
        d_model=128,
        finetune_last_n=0,
        use_dual_encoder=False,
        use_gnn=False,
        use_contact_vit=False,
        use_structure_delta=use_structure_delta,
        d_edit=128,
        d_fused=256,
        edit_n_heads=4,
        **{k: v for k, v in kwargs.items() if hasattr(EditRNAConfig, k)},
    )
    return EditRNA_A3A(config)
