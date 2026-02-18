"""Multi-task prediction heads for APOBEC editing prediction.

Provides independently testable prediction heads for 11 tasks organized
into three tiers plus auxiliary targets:

PRIMARY (always trained):
  1. Binary editing classification (is this C edited?)
  2. Editing rate regression (log2(max_rate + 0.01), range ~[-6.64, 6.62])
  3. APOBEC enzyme specificity (4-class: A3A=120, A3G=60, Both=178, Neither=206)

SECONDARY:
  4. Structure type (4-class: InLoop=368, dsRNA=135, ssRNA/Bulge=86, OpenssRNA=47)
  5. Tissue specificity (5-class: Blood=159, Ubiquitous=153, Testis=141,
     NonSpecific=110, Intestine=73)
  6. N tissues edited (log-transformed count regression)

TERTIARY:
  7. Functional impact (3-class CDS-only: syn=206, nonsyn=98, stopgain=19;
     mask 313 non-CDS sites)
  8. Conservation (binary: conserved=95, not=541)
  9. Cancer survival association (binary: yes=252, no=384)

AUXILIARY:
  10. Per-tissue editing rates (54 GTEx tissues, NaN-masked)
  11. HEK293 editing rate (available for 334/636 sites)

Also provides the multi-task loss with uncertainty weighting (Kendall et al. 2018).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ---------------------------------------------------------------------------
# Label encoding maps (for reference and data pipeline use)
# ---------------------------------------------------------------------------

ENZYME_CLASSES = {"APOBEC3A Only": 0, "APOBEC3G Only": 1, "Both": 2, "Neither": 3}
STRUCTURE_CLASSES = {"In Loop": 0, "dsRNA": 1, "ssRNA / Bulge": 2, "Open ssRNA": 3}
TISSUE_SPEC_CLASSES = {
    "Blood Specific": 0,
    "Ubiquitous": 1,
    "Testis Specific": 2,
    "Non-Specific": 3,
    "Intestine Specific": 4,
}
FUNCTION_CLASSES = {"synonymous": 0, "nonsynonymous": 1, "stopgain": 2}


# ---------------------------------------------------------------------------
# Individual prediction heads
# ---------------------------------------------------------------------------


class BinaryEditHead(nn.Module):
    """Binary classification head: is this C site edited?

    Parameters
    ----------
    d_in : int
        Input dimension (d_fused + d_edit).
    hidden_dim : int
        Hidden layer dimension.
    dropout : float
        Dropout probability.
    """

    def __init__(self, d_in: int, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (batch, 1) logits."""
        return self.head(x)


class EditingRateHead(nn.Module):
    """Editing rate regression head: predict log2(max_rate + 0.01).

    Output is unbounded (no sigmoid) since the target is log2-transformed.
    Range is approximately [-6.64, 6.62] for rates in [0%, 98.7%].

    Parameters
    ----------
    d_in : int
        Input dimension.
    hidden_dim : int
        Hidden layer dimension.
    dropout : float
        Dropout probability.
    """

    def __init__(self, d_in: int, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (batch, 1) log2-scale rate predictions."""
        return self.head(x)


class EnzymeSpecificityHead(nn.Module):
    """Enzyme specificity head: A3A-only / A3G-only / Both / Neither.

    72 sites have Unknown class and should be masked with -1.

    Parameters
    ----------
    d_in : int
        Input dimension.
    n_classes : int
        Number of enzyme classes.
    hidden_dim : int
        Hidden layer dimension.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self, d_in: int, n_classes: int = 4, hidden_dim: int = 128, dropout: float = 0.2
    ):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (batch, n_classes) logits."""
        return self.head(x)


class StructureTypeHead(nn.Module):
    """Structure type classification head.

    4-class: InLoop (368), dsRNA (135), ssRNA/Bulge (86), OpenssRNA (47).

    Parameters
    ----------
    d_in : int
        Input dimension.
    n_classes : int
        Number of structure classes.
    hidden_dim : int
        Hidden layer dimension.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self, d_in: int, n_classes: int = 4, hidden_dim: int = 128, dropout: float = 0.2
    ):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (batch, n_classes) logits."""
        return self.head(x)


class TissueSpecificityHead(nn.Module):
    """Tissue specificity classification head.

    5-class: Blood (159), Ubiquitous (153), Testis (141),
    NonSpecific (110), Intestine (73).

    Parameters
    ----------
    d_in : int
        Input dimension.
    n_classes : int
        Number of tissue specificity classes.
    hidden_dim : int
        Hidden layer dimension.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self, d_in: int, n_classes: int = 5, hidden_dim: int = 128, dropout: float = 0.2
    ):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (batch, n_classes) logits."""
        return self.head(x)


class NTissuesHead(nn.Module):
    """N tissues edited regression head (log-transformed).

    Predicts log2(n_tissues_edited) since distribution is highly skewed
    (median=1, max=52).

    Parameters
    ----------
    d_in : int
        Input dimension.
    hidden_dim : int
        Hidden layer dimension.
    dropout : float
        Dropout probability.
    """

    def __init__(self, d_in: int, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (batch, 1) log2(n_tissues) predictions."""
        return self.head(x)


class FunctionalImpactHead(nn.Module):
    """Exonic function head: synonymous / nonsynonymous / stopgain.

    CDS-only: 313 non-CDS sites should be masked with -1.

    Parameters
    ----------
    d_in : int
        Input dimension.
    n_classes : int
        Number of function classes.
    hidden_dim : int
        Hidden layer dimension.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self, d_in: int, n_classes: int = 3, hidden_dim: int = 128, dropout: float = 0.2
    ):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (batch, n_classes) logits."""
        return self.head(x)


class ConservationHead(nn.Module):
    """Conservation binary head: is this site conserved in mammals?

    Class balance: conserved=95, not_conserved=541 (highly imbalanced).

    Parameters
    ----------
    d_in : int
        Input dimension.
    hidden_dim : int
        Hidden layer dimension.
    dropout : float
        Dropout probability.
    """

    def __init__(self, d_in: int, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (batch, 1) logits."""
        return self.head(x)


class CancerSurvivalHead(nn.Module):
    """Cancer survival association binary head.

    Class balance: yes=252, no=384.

    Parameters
    ----------
    d_in : int
        Input dimension.
    hidden_dim : int
        Hidden layer dimension.
    dropout : float
        Dropout probability.
    """

    def __init__(self, d_in: int, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (batch, 1) logits."""
        return self.head(x)


class TissueRateHead(nn.Module):
    """Per-tissue editing rate prediction head (54 GTEx tissues).

    Output is in [0, 1] per tissue via sigmoid. Supports NaN masking
    for tissues where a site has no coverage.

    Parameters
    ----------
    d_in : int
        Input dimension.
    n_tissues : int
        Number of tissues to predict.
    hidden_dim : int
        Hidden layer dimension.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self, d_in: int, n_tissues: int = 54, hidden_dim: int = 256, dropout: float = 0.2
    ):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_tissues),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (batch, n_tissues) rates in [0, 1]."""
        return self.head(x)


class HEK293RateHead(nn.Module):
    """HEK293 editing rate head (auxiliary target).

    Available for 334/636 sites (mean=33%). Uses sigmoid output since
    rates are in [0, 100] and we normalize to [0, 1].

    Parameters
    ----------
    d_in : int
        Input dimension.
    hidden_dim : int
        Hidden layer dimension.
    dropout : float
        Dropout probability.
    """

    def __init__(self, d_in: int, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (batch, 1) rate in [0, 1]."""
        return self.head(x)


class EditEffectHead(nn.Module):
    """Causal edit effect head: predicts delta property from edit embedding only.

    This head operates on the edit embedding alone (not the fused representation),
    enforcing that it captures the causal effect of the C-to-U edit.

    Parameters
    ----------
    d_edit : int
        Edit embedding dimension.
    hidden_dim : int
        Hidden layer dimension.
    dropout : float
        Dropout probability.
    """

    def __init__(self, d_edit: int, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_edit, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, edit_emb: torch.Tensor) -> torch.Tensor:
        """Returns (batch, 1) edit effect prediction."""
        return self.head(edit_emb)


# ---------------------------------------------------------------------------
# Composite multi-task head
# ---------------------------------------------------------------------------


class APOBECMultiTaskHead(nn.Module):
    """Composite multi-task prediction head for APOBEC editing prediction.

    Composes all 11 individual heads and routes inputs appropriately.
    Most heads receive the concatenation of fused_repr and edit_emb.
    The edit_effect head receives only edit_emb (causal constraint).

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

        # PRIMARY
        self.binary_head = BinaryEditHead(d_combined, dropout=dropout)
        self.rate_head = EditingRateHead(d_combined, dropout=dropout)
        self.enzyme_head = EnzymeSpecificityHead(d_combined, dropout=dropout)

        # SECONDARY
        self.structure_head = StructureTypeHead(d_combined, dropout=dropout)
        self.tissue_spec_head = TissueSpecificityHead(d_combined, dropout=dropout)
        self.n_tissues_head = NTissuesHead(d_combined, dropout=dropout)

        # TERTIARY
        self.function_head = FunctionalImpactHead(d_combined, dropout=dropout)
        self.conservation_head = ConservationHead(d_combined, dropout=dropout)
        self.cancer_head = CancerSurvivalHead(d_combined, dropout=dropout)

        # AUXILIARY
        self.tissue_head = TissueRateHead(d_combined, n_tissues=n_tissues, dropout=dropout)
        self.hek293_head = HEK293RateHead(d_combined, dropout=dropout)

        # CAUSAL (from edit embedding only)
        self.effect_head = EditEffectHead(d_edit, dropout=dropout)

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
            "binary_logit"       : (batch, 1) -- is_edited logit
            "editing_rate"       : (batch, 1) -- log2(max_rate+0.01)
            "enzyme_logits"      : (batch, 4) -- A3A/A3G/Both/Neither
            "structure_logits"   : (batch, 4) -- InLoop/dsRNA/ssRNA/Open
            "tissue_spec_logits" : (batch, 5) -- Blood/Ubiq/Testis/NS/Intest
            "n_tissues"          : (batch, 1) -- log2(n_tissues_edited)
            "function_logits"    : (batch, 3) -- syn/nonsyn/stopgain
            "conservation_logit" : (batch, 1) -- is_conserved logit
            "cancer_logit"       : (batch, 1) -- has_survival logit
            "tissue_rates"       : (batch, n_tissues) -- per-tissue [0,1]
            "hek293_rate"        : (batch, 1) -- HEK293 rate [0,1]
            "edit_effect"        : (batch, 1) -- causal delta
        """
        combined = torch.cat([fused_repr, edit_emb], dim=-1)

        return {
            # PRIMARY
            "binary_logit": self.binary_head(combined),
            "editing_rate": self.rate_head(combined),
            "enzyme_logits": self.enzyme_head(combined),
            # SECONDARY
            "structure_logits": self.structure_head(combined),
            "tissue_spec_logits": self.tissue_spec_head(combined),
            "n_tissues": self.n_tissues_head(combined),
            # TERTIARY
            "function_logits": self.function_head(combined),
            "conservation_logit": self.conservation_head(combined),
            "cancer_logit": self.cancer_head(combined),
            # AUXILIARY
            "tissue_rates": self.tissue_head(combined),
            "hek293_rate": self.hek293_head(combined),
            # CAUSAL (edit embedding only)
            "edit_effect": self.effect_head(edit_emb),
        }


# ---------------------------------------------------------------------------
# Multi-task loss with uncertainty weighting
# ---------------------------------------------------------------------------

# All task names, grouped by tier
PRIMARY_TASKS = ["binary", "rate", "enzyme"]
SECONDARY_TASKS = ["structure", "tissue_spec", "n_tissues"]
TERTIARY_TASKS = ["function", "conservation", "cancer"]
AUXILIARY_TASKS = ["tissue_matrix", "hek293"]
CAUSAL_TASKS = ["effect"]
ALL_TASKS = PRIMARY_TASKS + SECONDARY_TASKS + TERTIARY_TASKS + AUXILIARY_TASKS + CAUSAL_TASKS


class APOBECMultiTaskLoss(nn.Module):
    """Multi-task loss with learnable uncertainty weighting.

    Implements "Multi-Task Learning Using Uncertainty to Weigh Losses"
    (Kendall et al., CVPR 2018). Each task has a learnable log-variance
    parameter that automatically balances loss magnitudes.

    Uses focal loss for binary classification tasks with class imbalance.

    Task-target key mapping
    -----------------------
    The ``targets`` dict uses these keys:
      - "binary"       : (B,) float {0, 1} or NaN
      - "rate"         : (B,) float log2(max_rate+0.01) or NaN
      - "enzyme"       : (B,) long {0,1,2,3} or -1 for Unknown
      - "structure"    : (B,) long {0,1,2,3} or -1
      - "tissue_spec"  : (B,) long {0,1,2,3,4} or -1
      - "n_tissues"    : (B,) float log2(n_tissues) or NaN
      - "function"     : (B,) long {0,1,2} or -1 for non-CDS
      - "conservation" : (B,) float {0, 1} or NaN
      - "cancer"       : (B,) float {0, 1} or NaN
      - "tissue_matrix": (B, 54) float in [0,1] with NaN for missing
      - "hek293"       : (B,) float in [0,1] or NaN
      - "effect"       : (B,) float or NaN

    Parameters
    ----------
    task_names : list[str] or None
        Names of the tasks. Defaults to ALL_TASKS.
    focal_gamma : float
        Gamma parameter for focal loss on binary classification.
    focal_alpha_binary : float
        Alpha for binary editing focal loss (weight for positive class).
    focal_alpha_conservation : float
        Alpha for conservation focal loss (weight for conserved class).
    """

    def __init__(
        self,
        task_names: Optional[list[str]] = None,
        focal_gamma: float = 2.0,
        focal_alpha_binary: float = 0.75,
        focal_alpha_conservation: float = 0.85,
    ):
        super().__init__()
        if task_names is None:
            task_names = list(ALL_TASKS)
        self.task_names = task_names
        self.focal_gamma = focal_gamma
        self.focal_alpha_binary = focal_alpha_binary
        self.focal_alpha_conservation = focal_alpha_conservation

        self.log_vars = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(1)) for name in task_names
        })

    def focal_loss(
        self, logits: torch.Tensor, targets: torch.Tensor, alpha: float
    ) -> torch.Tensor:
        """Focal loss for binary classification with class imbalance."""
        probs = torch.sigmoid(logits.squeeze(-1))
        targets = targets.float()

        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.focal_gamma

        bce = F.binary_cross_entropy_with_logits(
            logits.squeeze(-1), targets, reduction="none"
        )
        return (focal_weight * bce).mean()

    def _weighted_loss(
        self, task_name: str, raw_loss: torch.Tensor,
        total_loss: torch.Tensor, task_losses: dict,
    ) -> torch.Tensor:
        """Apply uncertainty weighting and accumulate."""
        precision = torch.exp(-self.log_vars[task_name])
        total_loss = total_loss + precision * raw_loss + self.log_vars[task_name]
        task_losses[task_name] = raw_loss.detach()
        return total_loss

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
        task_losses: dict[str, torch.Tensor] = {}
        total_loss = torch.tensor(0.0, device=next(iter(predictions.values())).device)

        # ---------------------------------------------------------------
        # PRIMARY TASKS
        # ---------------------------------------------------------------

        # 1. Binary editing (focal loss)
        if "binary" in targets and targets["binary"] is not None:
            mask = ~torch.isnan(targets["binary"])
            if mask.any():
                loss = self.focal_loss(
                    predictions["binary_logit"][mask],
                    targets["binary"][mask],
                    alpha=self.focal_alpha_binary,
                )
                total_loss = self._weighted_loss("binary", loss, total_loss, task_losses)

        # 2. Editing rate (MSE on log2 scale)
        if "rate" in targets and targets["rate"] is not None:
            mask = ~torch.isnan(targets["rate"])
            if mask.any():
                loss = F.mse_loss(
                    predictions["editing_rate"][mask].squeeze(-1),
                    targets["rate"][mask],
                )
                total_loss = self._weighted_loss("rate", loss, total_loss, task_losses)

        # 3. Enzyme specificity (cross-entropy, mask Unknown=-1)
        if "enzyme" in targets and targets["enzyme"] is not None:
            mask = targets["enzyme"] >= 0
            if mask.any():
                loss = F.cross_entropy(
                    predictions["enzyme_logits"][mask],
                    targets["enzyme"][mask].long(),
                )
                total_loss = self._weighted_loss("enzyme", loss, total_loss, task_losses)

        # ---------------------------------------------------------------
        # SECONDARY TASKS
        # ---------------------------------------------------------------

        # 4. Structure type (cross-entropy)
        if "structure" in targets and targets["structure"] is not None:
            mask = targets["structure"] >= 0
            if mask.any():
                loss = F.cross_entropy(
                    predictions["structure_logits"][mask],
                    targets["structure"][mask].long(),
                )
                total_loss = self._weighted_loss("structure", loss, total_loss, task_losses)

        # 5. Tissue specificity (cross-entropy)
        if "tissue_spec" in targets and targets["tissue_spec"] is not None:
            mask = targets["tissue_spec"] >= 0
            if mask.any():
                loss = F.cross_entropy(
                    predictions["tissue_spec_logits"][mask],
                    targets["tissue_spec"][mask].long(),
                )
                total_loss = self._weighted_loss("tissue_spec", loss, total_loss, task_losses)

        # 6. N tissues edited (MSE on log2 scale)
        if "n_tissues" in targets and targets["n_tissues"] is not None:
            mask = ~torch.isnan(targets["n_tissues"])
            if mask.any():
                loss = F.mse_loss(
                    predictions["n_tissues"][mask].squeeze(-1),
                    targets["n_tissues"][mask],
                )
                total_loss = self._weighted_loss("n_tissues", loss, total_loss, task_losses)

        # ---------------------------------------------------------------
        # TERTIARY TASKS
        # ---------------------------------------------------------------

        # 7. Functional impact (cross-entropy, mask non-CDS=-1)
        if "function" in targets and targets["function"] is not None:
            mask = targets["function"] >= 0
            if mask.any():
                loss = F.cross_entropy(
                    predictions["function_logits"][mask],
                    targets["function"][mask].long(),
                )
                total_loss = self._weighted_loss("function", loss, total_loss, task_losses)

        # 8. Conservation (focal loss, imbalanced: 95 vs 541)
        if "conservation" in targets and targets["conservation"] is not None:
            mask = ~torch.isnan(targets["conservation"])
            if mask.any():
                loss = self.focal_loss(
                    predictions["conservation_logit"][mask],
                    targets["conservation"][mask],
                    alpha=self.focal_alpha_conservation,
                )
                total_loss = self._weighted_loss("conservation", loss, total_loss, task_losses)

        # 9. Cancer survival (BCE)
        if "cancer" in targets and targets["cancer"] is not None:
            mask = ~torch.isnan(targets["cancer"])
            if mask.any():
                loss = F.binary_cross_entropy_with_logits(
                    predictions["cancer_logit"][mask].squeeze(-1),
                    targets["cancer"][mask],
                )
                total_loss = self._weighted_loss("cancer", loss, total_loss, task_losses)

        # ---------------------------------------------------------------
        # AUXILIARY TASKS
        # ---------------------------------------------------------------

        # 10. Per-tissue rates (MSE with per-element NaN masking)
        if "tissue_matrix" in targets and targets["tissue_matrix"] is not None:
            mask = ~torch.isnan(targets["tissue_matrix"])
            if mask.any():
                loss = F.mse_loss(
                    predictions["tissue_rates"][mask],
                    targets["tissue_matrix"][mask],
                )
                total_loss = self._weighted_loss("tissue_matrix", loss, total_loss, task_losses)

        # 11. HEK293 rate (MSE, available for 334/636 sites)
        if "hek293" in targets and targets["hek293"] is not None:
            mask = ~torch.isnan(targets["hek293"])
            if mask.any():
                loss = F.mse_loss(
                    predictions["hek293_rate"][mask].squeeze(-1),
                    targets["hek293"][mask],
                )
                total_loss = self._weighted_loss("hek293", loss, total_loss, task_losses)

        # ---------------------------------------------------------------
        # CAUSAL TASK
        # ---------------------------------------------------------------

        # 12. Edit effect (MSE from edit embedding only)
        if "effect" in targets and targets["effect"] is not None:
            mask = ~torch.isnan(targets["effect"])
            if mask.any():
                loss = F.mse_loss(
                    predictions["edit_effect"][mask].squeeze(-1),
                    targets["effect"][mask],
                )
                total_loss = self._weighted_loss("effect", loss, total_loss, task_losses)

        return total_loss, task_losses

    def get_task_weights(self) -> dict[str, float]:
        """Return the current effective task weights (precision = exp(-log_var)).

        Useful for logging and monitoring how the model balances tasks.
        """
        return {
            name: torch.exp(-self.log_vars[name]).item()
            for name in self.task_names
        }
