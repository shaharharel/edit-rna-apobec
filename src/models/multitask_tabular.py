"""Multi-task tabular MLP with uncertainty-weighted loss for APOBEC editing prediction.

This module implements the best-performing tabular model from Experiments 2b-4:
a 2-layer MLP with shared hidden representation and task-specific heads,
trained with learned uncertainty weighting (Kendall et al. 2018).

Key findings that shaped this architecture:
- Multi-task learning with MLP captures nonlinear feature interactions that RF/GB miss
- Rate prediction: max_rate alone achieves Spearman 0.938 via MLP (vs 0.547 via RF)
- Enzyme classification: tissue_specificity alone achieves 0.543 accuracy
- Feature groups are task-orthogonal: rate_stats for rate, tissue_breadth for enzyme/tissue,
  structure features for structure

Usage:
    model = MultiTaskTabularMLP(d_input=13, d_hidden=256)
    loss_fn = UncertaintyWeightedLoss(model)
    outputs = model(x_batch)  # returns dict with task predictions + 'hidden'
    loss = loss_fn(outputs, targets)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional


# 13 non-leaky features used in the verified model
DEFAULT_FEATURES = [
    "cat_cds",
    "cat_noncoding",
    "max_rate",
    "mean_rate",
    "std_rate",
    "n_tissues_with_rate",
    "tissue_specificity",
    "is_loop",
    "is_dsrna",
    "is_ssrna",
    "loop_length",
    "n_tissues_edited",
    "chrom_num",
]

# Feature groups for ablation studies
FEATURE_GROUPS = {
    "rate_stats": ["max_rate", "mean_rate", "std_rate"],
    "tissue_breadth": ["n_tissues_with_rate", "n_tissues_edited", "tissue_specificity"],
    "structure": ["is_loop", "is_dsrna", "is_ssrna", "loop_length"],
    "genomic": ["cat_cds", "cat_noncoding", "chrom_num"],
}

# Task definitions
TASK_CONFIGS = {
    "rate": {"type": "regression", "output_dim": 1},
    "enzyme": {"type": "classification", "output_dim": 4},
    "structure": {"type": "classification", "output_dim": 4},
    "tissue": {"type": "classification", "output_dim": 5},
    "function": {"type": "classification", "output_dim": 3},
    "n_tissues": {"type": "regression", "output_dim": 1},
    "conservation": {"type": "binary", "output_dim": 1},
    "cancer": {"type": "binary", "output_dim": 1},
    "hek293": {"type": "regression", "output_dim": 1},
}


class MultiTaskTabularMLP(nn.Module):
    """Multi-task MLP with shared backbone and task-specific heads.

    Architecture:
        Input -> [Linear -> GELU -> Dropout -> BN] x 2 -> Task Heads

    The shared backbone learns a joint representation that is then projected
    to task-specific predictions. Uncertainty weighting is applied during
    loss computation (see UncertaintyWeightedLoss).

    Args:
        d_input: Number of input features.
        d_hidden: Hidden dimension for shared backbone.
        dropout: Dropout rate.
        tasks: List of task names to include. Defaults to 5 core tasks.
    """

    def __init__(
        self,
        d_input: int,
        d_hidden: int = 256,
        dropout: float = 0.3,
        tasks: Optional[List[str]] = None,
    ):
        super().__init__()

        if tasks is None:
            tasks = ["rate", "enzyme", "structure", "tissue", "n_tissues"]

        self.tasks = tasks
        self.d_input = d_input
        self.d_hidden = d_hidden

        self.shared = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(d_hidden),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(d_hidden),
        )

        self.heads = nn.ModuleDict()
        for task in tasks:
            cfg = TASK_CONFIGS[task]
            if cfg["output_dim"] == 1:
                self.heads[task] = nn.Linear(d_hidden, 1)
            else:
                self.heads[task] = nn.Linear(d_hidden, cfg["output_dim"])

        self.log_vars = nn.ParameterDict(
            {task: nn.Parameter(torch.zeros(1)) for task in tasks}
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, d_input).

        Returns:
            Dict with task predictions and 'hidden' representation.
            Regression tasks return shape (batch,).
            Classification tasks return shape (batch, n_classes).
        """
        h = self.shared(x)

        outputs = {"hidden": h}
        for task in self.tasks:
            out = self.heads[task](h)
            cfg = TASK_CONFIGS[task]
            if cfg["output_dim"] == 1:
                outputs[task] = out.squeeze(-1)
            else:
                outputs[task] = out

        return outputs


class UncertaintyWeightedLoss(nn.Module):
    """Uncertainty-weighted multi-task loss (Kendall et al. 2018).

    For each task, the loss is:
        total += exp(-log_var) * task_loss + log_var

    This learns per-task weights that balance gradient magnitudes.

    Args:
        model: MultiTaskTabularMLP instance (uses its log_vars parameters).
    """

    def __init__(self, model: MultiTaskTabularMLP):
        super().__init__()
        self.model = model

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute uncertainty-weighted multi-task loss.

        Args:
            outputs: Model outputs dict.
            targets: Dict of target tensors. Classification targets use -1 for
                     invalid/missing samples. Binary targets use 0/1.

        Returns:
            Scalar loss tensor.
        """
        total = torch.tensor(0.0, device=next(iter(outputs.values())).device)

        for task in self.model.tasks:
            if task not in targets:
                continue

            cfg = TASK_CONFIGS[task]
            log_var = self.model.log_vars[task]

            if cfg["type"] == "regression":
                task_loss = F.mse_loss(outputs[task], targets[task])
            elif cfg["type"] == "classification":
                mask = targets[task] >= 0
                if mask.sum() == 0:
                    continue
                task_loss = F.cross_entropy(
                    outputs[task][mask], targets[task][mask]
                )
            elif cfg["type"] == "binary":
                mask = (targets[task] >= 0) & (targets[task] <= 1)
                if mask.sum() == 0:
                    continue
                task_loss = F.binary_cross_entropy_with_logits(
                    outputs[task][mask], targets[task][mask].float()
                )

            total = total + torch.exp(-log_var) * task_loss + log_var

        return total
