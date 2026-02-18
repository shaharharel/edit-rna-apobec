"""APOBEC3A-specific Edit Embedding Module.

Encodes C-to-U RNA edits as structured, context-aware intervention embeddings
tailored for APOBEC3A editing site prediction. Builds on the project's existing
StructuredRNAEditEmbedder and adds APOBEC-specific components:

1. Stem-loop aware context attention (attends to predicted stem-loop around edit)
2. Structure concordance encoding (mRNA vs pre-mRNA structure agreement)
3. TC/CC motif encoding (APOBEC3A's preferred dinucleotide context)
4. Multi-encoder support (RNA-FM + UTR-LM dual encoding)

The module works within the causal edit effect framework:
    F(seq_before, edit) -> delta_property
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# APOBEC-specific edit types
APOBEC_EDIT_TYPES = {
    "C_to_U": 0,  # primary APOBEC3A/3G edit
    "other": 1,
}

# Flanking dinucleotide context classes (position -1 relative to target C)
FLANKING_CONTEXT = {
    "TC": 0,  # APOBEC3A preferred
    "CC": 1,  # APOBEC3G preferred
    "AC": 2,
    "GC": 3,
}


class APOBECEditEmbedding(nn.Module):
    """Edit embedding module specialized for APOBEC3A C-to-U RNA editing.

    Produces a fixed-size embedding that captures the causal effect of a
    C-to-U edit in its full sequence and structural context.

    Parameters
    ----------
    d_model : int
        Dimension of the input sequence representations from the RNA encoder.
    d_edit : int
        Dimension of the output edit embedding.
    n_heads : int
        Number of attention heads for context cross-attention.
    dropout : float
        Dropout probability.
    use_structure_delta : bool
        Whether to encode structural changes caused by the edit.
    use_dual_encoder : bool
        Whether to accept embeddings from a second encoder (e.g., UTR-LM).
    d_model_secondary : int
        Dimension of the secondary encoder (only if use_dual_encoder=True).
    """

    def __init__(
        self,
        d_model: int = 640,
        d_edit: int = 256,
        n_heads: int = 8,
        dropout: float = 0.1,
        use_structure_delta: bool = True,
        use_dual_encoder: bool = False,
        d_model_secondary: int = 128,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_edit = d_edit
        self.use_structure_delta = use_structure_delta
        self.use_dual_encoder = use_dual_encoder

        # --- Component 1: Flanking motif embedding ---
        # Encodes the dinucleotide context (TC, CC, AC, GC)
        self.flanking_embed = nn.Embedding(len(FLANKING_CONTEXT), 32)

        # --- Component 2: Local difference encoding ---
        # Projects F(edited) - F(original) at the edit position
        self.local_diff_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        # --- Component 3: Stem-loop aware context attention ---
        # Edit position attends to surrounding sequence context
        self.context_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.context_norm = nn.LayerNorm(d_model)

        # --- Component 4: Structure delta encoding ---
        # Captures how C->U changes RNA structure (pairing probs, stem stability)
        if use_structure_delta:
            # Input: 7 structure delta features (from RNAplfold)
            # delta_pairing, delta_accessibility, delta_entropy, delta_mfe,
            # delta_local_pairing, delta_local_accessibility, local_pairing_std
            self.structure_delta_proj = nn.Sequential(
                nn.Linear(7, 64),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(64, 64),
                nn.LayerNorm(64),
            )
            structure_dim = 64
        else:
            structure_dim = 0

        # --- Component 5: Structure concordance encoding ---
        # Binary: does mRNA structure agree with pre-mRNA structure at this site?
        # Plus transition type encoding (e.g., dsRNA->InLoop)
        self.concordance_embed = nn.Sequential(
            nn.Linear(5, 32),  # 1 binary + 4 transition type one-hot
            nn.GELU(),
            nn.LayerNorm(32),
        )
        concordance_dim = 32

        # --- Component 6: Secondary encoder fusion (optional) ---
        if use_dual_encoder:
            self.secondary_proj = nn.Sequential(
                nn.Linear(d_model_secondary, d_model // 2),
                nn.GELU(),
                nn.LayerNorm(d_model // 2),
            )
            secondary_dim = d_model // 2
        else:
            secondary_dim = 0

        # --- Final fusion ---
        # Combines: local_diff+context (d_model) + flanking (32)
        #         + structure_delta (64) + concordance (32) + secondary (d_model//2)
        fusion_input_dim = d_model + 32 + structure_dim + concordance_dim + secondary_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, d_edit * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_edit * 2, d_edit),
            nn.LayerNorm(d_edit),
        )

        self.output_dim = d_edit

    def forward(
        self,
        f_background: torch.Tensor,
        f_edited: torch.Tensor,
        edit_pos: torch.Tensor,
        flanking_context: torch.Tensor,
        structure_delta: Optional[torch.Tensor] = None,
        concordance_features: Optional[torch.Tensor] = None,
        f_background_secondary: Optional[torch.Tensor] = None,
        seq_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the APOBEC edit embedding.

        Parameters
        ----------
        f_background : (batch, seq_len, d_model)
            Encoder output for the original sequence (with C).
        f_edited : (batch, seq_len, d_model)
            Encoder output for the edited sequence (with U).
        edit_pos : (batch,) int
            Position of the edited C within the sequence window.
        flanking_context : (batch,) int
            Flanking dinucleotide class index (0=TC, 1=CC, 2=AC, 3=GC).
        structure_delta : (batch, 7) float or None
            Structure change features from RNAplfold.
        concordance_features : (batch, 5) float or None
            Structure concordance: [is_concordant, trans_dsRNA_loop,
            trans_loop_dsRNA, trans_dsRNA_dsRNA, trans_loop_loop].
        f_background_secondary : (batch, seq_len, d_model_secondary) or None
            Embeddings from secondary encoder (UTR-LM).
        seq_mask : (batch, seq_len) bool or None
            True for valid (non-padding) positions.

        Returns
        -------
        edit_emb : (batch, d_edit)
        """
        batch_size = f_background.shape[0]
        device = f_background.device

        # Gather representations at edit position
        pos_idx = edit_pos.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, self.d_model)
        f_bg_at_pos = f_background.gather(1, pos_idx).squeeze(1)  # (B, d_model)
        f_ed_at_pos = f_edited.gather(1, pos_idx).squeeze(1)      # (B, d_model)

        # 1. Local difference
        local_delta = self.local_diff_proj(f_ed_at_pos - f_bg_at_pos)  # (B, d_model)

        # 2. Context attention: edit position queries surrounding context
        query = f_bg_at_pos.unsqueeze(1)  # (B, 1, d_model)
        key_padding_mask = ~seq_mask if seq_mask is not None else None

        context_out, _ = self.context_attention(
            query, f_background, f_background,
            key_padding_mask=key_padding_mask,
        )
        context_out = self.context_norm(
            context_out.squeeze(1) + f_bg_at_pos
        )  # (B, d_model)

        # Combine local delta and context
        combined_seq = local_delta + context_out  # (B, d_model)

        # 3. Flanking motif
        flanking_emb = self.flanking_embed(flanking_context)  # (B, 32)

        # 4. Structure delta
        components = [combined_seq, flanking_emb]

        if self.use_structure_delta and structure_delta is not None:
            struct_emb = self.structure_delta_proj(structure_delta)  # (B, 64)
            components.append(struct_emb)
        elif self.use_structure_delta:
            components.append(torch.zeros(batch_size, 64, device=device))

        # 5. Structure concordance
        if concordance_features is not None:
            conc_emb = self.concordance_embed(concordance_features)  # (B, 32)
        else:
            conc_emb = torch.zeros(batch_size, 32, device=device)
        components.append(conc_emb)

        # 6. Secondary encoder
        if self.use_dual_encoder and f_background_secondary is not None:
            pos_idx_sec = edit_pos.unsqueeze(-1).unsqueeze(-1).expand(
                -1, 1, f_background_secondary.shape[-1]
            )
            f_sec_at_pos = f_background_secondary.gather(1, pos_idx_sec).squeeze(1)
            sec_emb = self.secondary_proj(f_sec_at_pos)
            components.append(sec_emb)
        elif self.use_dual_encoder:
            components.append(
                torch.zeros(batch_size, self.d_model // 2, device=device)
            )

        # Fuse all components
        fused = torch.cat(components, dim=-1)
        edit_emb = self.fusion(fused)  # (B, d_edit)

        return edit_emb

    @property
    def embedding_dim(self) -> int:
        return self.output_dim


class APOBECMultiTaskHead(nn.Module):
    """Multi-task prediction head for APOBEC editing prediction.

    Produces predictions for:
    1. Binary editing classification (is this C edited by A3A?)
    2. Editing rate regression (what fraction of transcripts are edited?)
    3. Enzyme specificity (A3A-only / A3G-only / Both / Neither)
    4. Exonic function (synonymous / nonsynonymous / stopgain)
    5. Edit effect (delta in downstream property -- causal prediction)

    Parameters
    ----------
    d_fused : int
        Dimension of the fused multi-modal representation.
    d_edit : int
        Dimension of the edit embedding.
    n_tissues : int
        Number of tissues for per-tissue editing rate prediction.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        d_fused: int = 512,
        d_edit: int = 256,
        n_tissues: int = 54,
        dropout: float = 0.2,
    ):
        super().__init__()
        d_combined = d_fused + d_edit

        # Head 1: Binary classification (edited / not edited)
        self.binary_head = nn.Sequential(
            nn.Linear(d_combined, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

        # Head 2: Editing rate regression (0-100%)
        self.rate_head = nn.Sequential(
            nn.Linear(d_combined, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid(),  # constrain to [0, 1]
        )

        # Head 3: Enzyme specificity (4 classes: A3A, A3G, Both, Neither)
        self.enzyme_head = nn.Sequential(
            nn.Linear(d_combined, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 4),
        )

        # Head 4: Exonic function (3 classes: synonymous, nonsynonymous, stopgain)
        self.function_head = nn.Sequential(
            nn.Linear(d_combined, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3),
        )

        # Head 5: Per-tissue editing rates
        self.tissue_head = nn.Sequential(
            nn.Linear(d_combined, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_tissues),
            nn.Sigmoid(),
        )

        # Head 6: Edit effect (causal delta prediction, from edit embedding only)
        self.effect_head = nn.Sequential(
            nn.Linear(d_edit, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        fused_repr: torch.Tensor,
        edit_emb: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute all prediction heads.

        Parameters
        ----------
        fused_repr : (batch, d_fused)
            Multi-modal fused representation.
        edit_emb : (batch, d_edit)
            Edit embedding from APOBECEditEmbedding.

        Returns
        -------
        dict with keys:
            "binary_logit": (batch, 1)
            "editing_rate": (batch, 1)
            "enzyme_logits": (batch, 4)
            "function_logits": (batch, 3)
            "tissue_rates": (batch, n_tissues)
            "edit_effect": (batch, 1)
        """
        combined = torch.cat([fused_repr, edit_emb], dim=-1)

        return {
            "binary_logit": self.binary_head(combined),
            "editing_rate": self.rate_head(combined),
            "enzyme_logits": self.enzyme_head(combined),
            "function_logits": self.function_head(combined),
            "tissue_rates": self.tissue_head(combined),
            "edit_effect": self.effect_head(edit_emb),  # from edit embedding only
        }


class APOBECMultiTaskLoss(nn.Module):
    """Multi-task loss with learnable uncertainty weighting.

    Implements "Multi-Task Learning Using Uncertainty to Weigh Losses"
    (Kendall et al., CVPR 2018). Each task has a learnable log-variance
    parameter that automatically balances loss magnitudes.

    Parameters
    ----------
    task_names : list[str]
        Names of the tasks.
    focal_gamma : float
        Gamma parameter for focal loss on binary classification.
    focal_alpha : float
        Alpha parameter for focal loss (weight for positive class).
    """

    def __init__(
        self,
        task_names: Optional[list[str]] = None,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.75,
    ):
        super().__init__()
        if task_names is None:
            task_names = [
                "binary", "rate", "enzyme", "function", "tissue", "effect"
            ]
        self.task_names = task_names
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha

        # Learnable log-variance for each task (uncertainty weighting)
        self.log_vars = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(1)) for name in task_names
        })

    def focal_loss(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Focal loss for binary classification with class imbalance."""
        probs = torch.sigmoid(logits.squeeze(-1))
        targets = targets.float()

        # Focal modulation
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.focal_alpha * targets + (1 - self.focal_alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.focal_gamma

        bce = F.binary_cross_entropy_with_logits(
            logits.squeeze(-1), targets, reduction="none"
        )
        return (focal_weight * bce).mean()

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute weighted multi-task loss.

        Parameters
        ----------
        predictions : dict from APOBECMultiTaskHead.forward()
        targets : dict with matching keys and target tensors.
            Missing keys are skipped (for partially labeled data).

        Returns
        -------
        total_loss : scalar tensor
        task_losses : dict of individual task losses (for logging)
        """
        task_losses = {}
        total_loss = torch.tensor(0.0, device=next(iter(predictions.values())).device)

        # Binary classification (focal loss)
        if "binary" in targets and targets["binary"] is not None:
            mask = ~torch.isnan(targets["binary"])
            if mask.any():
                loss = self.focal_loss(
                    predictions["binary_logit"][mask],
                    targets["binary"][mask],
                )
                precision = torch.exp(-self.log_vars["binary"])
                total_loss = total_loss + precision * loss + self.log_vars["binary"]
                task_losses["binary"] = loss.detach()

        # Editing rate regression (Huber loss)
        if "rate" in targets and targets["rate"] is not None:
            mask = ~torch.isnan(targets["rate"])
            if mask.any():
                loss = F.huber_loss(
                    predictions["editing_rate"][mask].squeeze(-1),
                    targets["rate"][mask],
                )
                precision = torch.exp(-self.log_vars["rate"])
                total_loss = total_loss + precision * loss + self.log_vars["rate"]
                task_losses["rate"] = loss.detach()

        # Enzyme specificity (cross-entropy)
        if "enzyme" in targets and targets["enzyme"] is not None:
            mask = targets["enzyme"] >= 0  # -1 = unknown
            if mask.any():
                loss = F.cross_entropy(
                    predictions["enzyme_logits"][mask],
                    targets["enzyme"][mask].long(),
                )
                precision = torch.exp(-self.log_vars["enzyme"])
                total_loss = total_loss + precision * loss + self.log_vars["enzyme"]
                task_losses["enzyme"] = loss.detach()

        # Exonic function (cross-entropy)
        if "function" in targets and targets["function"] is not None:
            mask = targets["function"] >= 0
            if mask.any():
                loss = F.cross_entropy(
                    predictions["function_logits"][mask],
                    targets["function"][mask].long(),
                )
                precision = torch.exp(-self.log_vars["function"])
                total_loss = total_loss + precision * loss + self.log_vars["function"]
                task_losses["function"] = loss.detach()

        # Per-tissue rates (MSE with NaN masking)
        if "tissue" in targets and targets["tissue"] is not None:
            mask = ~torch.isnan(targets["tissue"])
            if mask.any():
                loss = F.mse_loss(
                    predictions["tissue_rates"][mask],
                    targets["tissue"][mask],
                )
                precision = torch.exp(-self.log_vars["tissue"])
                total_loss = total_loss + precision * loss + self.log_vars["tissue"]
                task_losses["tissue"] = loss.detach()

        # Edit effect (MSE)
        if "effect" in targets and targets["effect"] is not None:
            mask = ~torch.isnan(targets["effect"])
            if mask.any():
                loss = F.mse_loss(
                    predictions["edit_effect"][mask].squeeze(-1),
                    targets["effect"][mask],
                )
                precision = torch.exp(-self.log_vars["effect"])
                total_loss = total_loss + precision * loss + self.log_vars["effect"]
                task_losses["effect"] = loss.detach()

        return total_loss, task_losses
