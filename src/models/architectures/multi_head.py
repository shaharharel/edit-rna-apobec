"""
Multi-task learning architecture with shared backbone and task-specific heads.

This architecture enables multi-task learning where:
1. A shared backbone processes input features (molecule + edit embeddings)
2. Task-specific heads make predictions for each property
3. All tasks are trained jointly, sharing representations

Benefits:
- Tasks share learned representations (transfer learning)
- Edit embeddings are refined by gradients from all tasks
- More parameter efficient than training separate models
- Can improve performance on tasks with limited data
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional
from collections import OrderedDict


class SharedBackbone(nn.Module):
    """
    Shared backbone network for multi-task learning.

    Processes concatenated (molecule, edit) embeddings into a shared
    representation that is used by all task-specific heads.

    Args:
        input_dim: Input dimension (mol_dim + edit_dim)
        hidden_dims: List of hidden layer dimensions
        dropout: Dropout probability
        activation: Activation function ('relu', 'elu', 'gelu')
        output_dim: Output dimension (shared representation size)
                   If None, uses last hidden_dim

    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        dropout: float = 0.2,
        activation: str = 'relu',
        output_dim: Optional[int] = None
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim if output_dim is not None else hidden_dims[-1]

        # Choose activation
        if activation == 'relu':
            act_fn = nn.ReLU
        elif activation == 'elu':
            act_fn = nn.ELU
        elif activation == 'gelu':
            act_fn = nn.GELU
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build network
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                act_fn(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # Output projection (if output_dim differs from last hidden)
        if self.output_dim != prev_dim:
            layers.append(nn.Linear(prev_dim, self.output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through shared backbone.

        Args:
            x: Input features [batch_size, input_dim]

        Returns:
            Shared representation [batch_size, output_dim]
        """
        return self.network(x)


class TaskHead(nn.Module):
    """
    Task-specific prediction head for multi-task learning.

    Takes shared representation and produces task-specific predictions.

    Args:
        input_dim: Dimension of shared representation
        hidden_dim: Optional single hidden layer dimension (legacy, use hidden_dims for multiple)
        hidden_dims: Optional list of hidden layer dimensions (e.g., [256, 256, 256, 128])
        dropout: Dropout probability (not applied to last layer)
        activation: Activation function

    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: Optional[int] = None,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        activation: str = 'relu'
    ):
        super().__init__()

        self.input_dim = input_dim

        # Choose activation
        if activation == 'relu':
            act_fn = nn.ReLU
        elif activation == 'elu':
            act_fn = nn.ELU
        elif activation == 'gelu':
            act_fn = nn.GELU
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build head
        if hidden_dims is not None:
            # Multi-layer head with configurable dims
            layers = []
            prev_dim = input_dim
            for i, dim in enumerate(hidden_dims):
                layers.append(nn.Linear(prev_dim, dim))
                layers.append(act_fn())
                # No dropout on last hidden layer (before output)
                if i < len(hidden_dims) - 1:
                    layers.append(nn.Dropout(dropout))
                prev_dim = dim
            layers.append(nn.Linear(prev_dim, 1))
            self.network = nn.Sequential(*layers)
        elif hidden_dim is not None:
            # Legacy two-layer head
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                act_fn(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            )
        else:
            # Single-layer head (direct projection)
            self.network = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through task head.

        Args:
            x: Shared representation [batch_size, input_dim]

        Returns:
            Task predictions [batch_size]
        """
        return self.network(x).squeeze(-1)


class MultiTaskHead(nn.Module):
    """
    Multi-task prediction head managing multiple task-specific heads.

    Routes shared representation to all task heads and collects predictions.

    Args:
        shared_dim: Dimension of shared representation from backbone
        task_names: List of task names (e.g., ['logP', 'QED', 'SAS'])
        head_hidden_dim: Optional single hidden dimension for task heads (legacy)
        head_hidden_dims: Optional list of hidden dims for task heads (e.g., [256, 256, 256, 128])
        dropout: Dropout probability for task heads (not applied to last layer)
        activation: Activation function for task heads

    """

    def __init__(
        self,
        shared_dim: int,
        task_names: List[str],
        head_hidden_dim: Optional[int] = None,
        head_hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        activation: str = 'relu'
    ):
        super().__init__()

        self.shared_dim = shared_dim
        self.task_names = task_names
        self.n_tasks = len(task_names)

        # Create task-specific heads
        self.heads = nn.ModuleDict({
            task_name: TaskHead(
                input_dim=shared_dim,
                hidden_dim=head_hidden_dim,
                hidden_dims=head_hidden_dims,
                dropout=dropout,
                activation=activation
            )
            for task_name in task_names
        })

    def forward(self, shared_repr: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through all task heads.

        Args:
            shared_repr: Shared representation [batch_size, shared_dim]

        Returns:
            Dictionary mapping task names to predictions
            {task_name: predictions [batch_size]}
        """
        predictions = {}
        for task_name, head in self.heads.items():
            predictions[task_name] = head(shared_repr)

        return predictions

    def forward_single_task(self, shared_repr: torch.Tensor, task_name: str) -> torch.Tensor:
        """
        Forward pass through a single task head.

        Args:
            shared_repr: Shared representation [batch_size, shared_dim]
            task_name: Name of task to predict

        Returns:
            Predictions for specified task [batch_size]
        """
        if task_name not in self.heads:
            raise ValueError(f"Unknown task: {task_name}. Available: {self.task_names}")

        return self.heads[task_name](shared_repr)


class MultiTaskNetwork(nn.Module):
    """
    Complete multi-task network combining backbone and task heads.

    This is a convenience wrapper that combines SharedBackbone and MultiTaskHead
    into a single module.

    Args:
        input_dim: Input dimension (mol_dim + edit_dim)
        task_names: List of task names
        backbone_hidden_dims: Hidden dimensions for shared backbone
        shared_dim: Dimension of shared representation
        head_hidden_dim: Optional single hidden dimension for task heads (legacy)
        head_hidden_dims: Optional list of hidden dims for task heads (e.g., [256, 256, 256, 128])
        dropout: Dropout probability
        activation: Activation function

    """

    def __init__(
        self,
        input_dim: int,
        task_names: List[str],
        backbone_hidden_dims: List[int],
        shared_dim: int,
        head_hidden_dim: Optional[int] = None,
        head_hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
        activation: str = 'relu'
    ):
        super().__init__()

        self.input_dim = input_dim
        self.task_names = task_names
        self.n_tasks = len(task_names)

        # Shared backbone
        self.backbone = SharedBackbone(
            input_dim=input_dim,
            hidden_dims=backbone_hidden_dims,
            dropout=dropout,
            activation=activation,
            output_dim=shared_dim
        )

        # Multi-task heads
        self.multi_head = MultiTaskHead(
            shared_dim=shared_dim,
            task_names=task_names,
            head_hidden_dim=head_hidden_dim,
            head_hidden_dims=head_hidden_dims,
            dropout=dropout,
            activation=activation
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through complete network.

        Args:
            x: Input features [batch_size, input_dim]

        Returns:
            Dictionary mapping task names to predictions
        """
        # Shared representation
        shared_repr = self.backbone(x)

        # Task-specific predictions
        predictions = self.multi_head(shared_repr)

        return predictions

    def forward_single_task(self, x: torch.Tensor, task_name: str) -> torch.Tensor:
        """
        Forward pass for a single task.

        Args:
            x: Input features [batch_size, input_dim]
            task_name: Task to predict

        Returns:
            Predictions for specified task [batch_size]
        """
        shared_repr = self.backbone(x)
        return self.multi_head.forward_single_task(shared_repr, task_name)
