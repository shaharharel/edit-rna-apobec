"""
Edit effect predictor: f(molecule, edit) → Δproperty

This is the CAUSAL model that predicts how an edit changes a property.
For baseline property prediction, use PropertyPredictor instead.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Optional, List, Union, Tuple
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# Enable Tensor Cores for L4 GPU
torch.set_float32_matmul_precision('high')


class EditEffectMLP(pl.LightningModule):
    """
    Multi-layer perceptron for edit effect prediction: f(molecule, edit) → ΔY

    Supports trainable edit embeddings and multi-task learning:
    - Trainable edits: Edit embeddings refined through backpropagation
    - Multi-task: Shared backbone + task-specific heads for multiple properties

    Architecture automatically scales down from input_dim to 1:
    [mol_dim + edit_dim] → [input_dim/2] → [input_dim/4] → ... → [1]

    Args:
        mol_dim: Molecule embedding dimension
        edit_dim: Edit embedding dimension
        hidden_dims: Optional list of hidden dimensions. If None, auto-generates halving layers
        head_hidden_dims: Optional list of hidden dims for task heads (e.g., [256, 256, 256, 128])
        dropout: Dropout probability
        learning_rate: Learning rate for Adam optimizer (MLP heads, default: 1e-3)
        activation: Activation function ('relu', 'elu', 'gelu')
        trainable_edit_layer: Optional TrainableEditEmbedder module
        n_tasks: Number of properties to predict (1 for single-task)
        task_names: Optional list of task names
        task_weights: Optional dict of task weights for loss
        gnn_learning_rate: Learning rate for GNN parameters (default: 1e-5)
                          Only used if mol_embedder is trainable
        mol_embedder: Reference to molecule embedder (for parameter grouping)
                     If provided and trainable, uses separate learning rates for GNN vs MLP
    """

    def __init__(
        self,
        mol_dim: int,
        edit_dim: int,
        hidden_dims: Optional[List[int]] = None,
        head_hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        activation: str = 'relu',
        trainable_edit_layer: Optional[nn.Module] = None,
        n_tasks: int = 1,
        task_names: Optional[List[str]] = None,
        task_weights: Optional[dict] = None,
        gnn_learning_rate: Optional[float] = None,  # Separate LR for GNN (if mol_embedder is trainable)
        mol_embedder: Optional[nn.Module] = None  # Reference to mol_embedder for parameter grouping
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['trainable_edit_layer', 'mol_embedder'])

        self.mol_dim = mol_dim
        self.edit_dim = edit_dim
        self.input_dim = mol_dim + edit_dim
        self.learning_rate = learning_rate
        self.gnn_learning_rate = gnn_learning_rate if gnn_learning_rate is not None else 1e-5
        self.n_tasks = n_tasks
        self.head_hidden_dims = head_hidden_dims

        # Trainable edit embedding layer
        self.trainable_edit_layer = trainable_edit_layer

        # Molecule embedder (for separate GNN learning rate)
        self.mol_embedder = mol_embedder

        # Task names
        if task_names is None:
            if n_tasks == 1:
                self.task_names = ['delta_property']
            else:
                self.task_names = [f'task_{i}' for i in range(n_tasks)]
        else:
            if len(task_names) != n_tasks:
                raise ValueError(f"Number of task names ({len(task_names)}) must match n_tasks ({n_tasks})")
            self.task_names = task_names

        # Task weights for loss
        if task_weights is None:
            self.task_weights = {name: 1.0 for name in self.task_names}
        else:
            self.task_weights = task_weights

        # Auto-generate hidden dims if not provided
        if hidden_dims is None:
            hidden_dims = []
            current_dim = self.input_dim

            # Halve until we reach 64 (or until we've halved 3 times max)
            min_hidden_dim = 64
            max_layers = 3

            for _ in range(max_layers):
                current_dim = current_dim // 2
                if current_dim < min_hidden_dim:
                    break
                hidden_dims.append(current_dim)

            # Ensure at least one hidden layer
            if len(hidden_dims) == 0:
                hidden_dims = [max(self.input_dim // 2, 64)]

        self.hidden_dims = hidden_dims

        # Choose activation
        if activation == 'relu':
            act_fn = nn.ReLU
        elif activation == 'elu':
            act_fn = nn.ELU
        elif activation == 'gelu':
            act_fn = nn.GELU
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build network based on number of tasks
        if n_tasks == 1:
            # Single-task network (backward compatible)
            layers = []
            prev_dim = self.input_dim

            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    act_fn(),
                    nn.Dropout(dropout)
                ])
                prev_dim = hidden_dim

            # Output layer (no activation for regression)
            layers.append(nn.Linear(prev_dim, 1))

            self.network = nn.Sequential(*layers)
            self.multi_task_network = None

        else:
            # Multi-task network
            from ..architectures.multi_head import MultiTaskNetwork

            # Shared dimension
            shared_dim = hidden_dims[-1] if hidden_dims else max(self.input_dim // 4, 64)

            # Head hidden dimension - only used if head_hidden_dims not provided
            head_hidden_dim = max(shared_dim // 2, 32) if head_hidden_dims is None else None

            self.multi_task_network = MultiTaskNetwork(
                input_dim=self.input_dim,
                task_names=self.task_names,
                backbone_hidden_dims=hidden_dims,
                shared_dim=shared_dim,
                head_hidden_dim=head_hidden_dim,
                head_hidden_dims=head_hidden_dims,
                dropout=dropout,
                activation=activation
            )
            self.network = None  # Not used in multi-task mode

    def forward(self, mol_emb, edit_emb_or_tuple):
        """
        Forward pass: (molecule_embedding, edit_embedding) → delta_property

        Args:
            mol_emb: Molecule A embedding tensor [batch_size, mol_dim]
                    (for trainable mode, this is redundant - mol_a_emb comes from tuple)
            edit_emb_or_tuple: Either:
                - Edit embedding tensor [batch_size, edit_dim] (non-trainable mode)
                - Tuple (mol_a_emb, mol_b_emb) for trainable Mode 1
                - Tuple (mol_a_emb, mol_b_emb, edit_frag_a_emb, edit_frag_b_emb) for trainable Mode 2

        Returns:
            Single-task: Predicted delta [batch_size]
            Multi-task: Dict {task_name: predictions [batch_size]}
        """
        # Handle trainable edit embeddings
        if self.trainable_edit_layer is not None:
            # Mode 2: 4-tuple (mol_a, mol_b, edit_fragment_a, edit_fragment_b)
            if isinstance(edit_emb_or_tuple, tuple) and len(edit_emb_or_tuple) == 4:
                mol_a_emb, mol_b_emb, edit_frag_a_emb, edit_frag_b_emb = edit_emb_or_tuple
                edit_emb = self.trainable_edit_layer(mol_a_emb, mol_b_emb, edit_frag_a_emb, edit_frag_b_emb)
            # Mode 1: 2-tuple (mol_a, mol_b)
            elif isinstance(edit_emb_or_tuple, tuple) and len(edit_emb_or_tuple) == 2:
                mol_a_emb, mol_b_emb = edit_emb_or_tuple
                edit_emb = self.trainable_edit_layer(mol_a_emb, mol_b_emb)
            else:
                raise ValueError(
                    f"For trainable edits, expected edit_emb_or_tuple to be a 2-tuple or 4-tuple, "
                    f"got {type(edit_emb_or_tuple)} with length {len(edit_emb_or_tuple) if isinstance(edit_emb_or_tuple, (tuple, list)) else 'N/A'}"
                )
            # Use mol_a_emb from the tuple (not the parameter)
        else:
            # Non-trainable mode: use parameter mol_emb and precomputed edit_emb
            mol_a_emb = mol_emb
            edit_emb = edit_emb_or_tuple

        # Concatenate molecule A and edit embeddings
        x = torch.cat([mol_a_emb, edit_emb], dim=-1)

        # Forward through network
        if self.n_tasks == 1:
            return self.network(x).squeeze(-1)
        else:
            return self.multi_task_network(x)

    def training_step(self, batch, batch_idx):
        # Handle trainable edit embeddings differently
        if self.trainable_edit_layer is not None:
            # Mode 2: batch is (mol_a_emb, mol_b_emb, edit_frag_a_emb, edit_frag_b_emb, delta_y)
            if len(batch) == 5:
                mol_a_emb, mol_b_emb, edit_frag_a_emb, edit_frag_b_emb, delta_y = batch
                mol_emb = mol_a_emb
                edit_emb_tuple = (mol_a_emb, mol_b_emb, edit_frag_a_emb, edit_frag_b_emb)
            # Mode 1: batch is (mol_a_emb, mol_b_emb, delta_y)
            elif len(batch) == 3:
                mol_a_emb, mol_b_emb, delta_y = batch
                mol_emb = mol_a_emb
                edit_emb_tuple = (mol_a_emb, mol_b_emb)
            else:
                raise ValueError(f"Expected batch to have 3 or 5 elements for trainable edits, got {len(batch)}")
        else:
            # For precomputed edits: batch is (mol_emb, edit_emb, delta_y)
            mol_emb, edit_emb_tuple, delta_y = batch

        if self.n_tasks == 1:
            # Single-task training
            delta_pred = self(mol_emb, edit_emb_tuple)
            loss = nn.functional.mse_loss(delta_pred, delta_y)

            # Log metrics
            self.log('train_loss', loss, prog_bar=True)
            self.log('train_mae', nn.functional.l1_loss(delta_pred, delta_y))

        else:
            # Multi-task training (handles sparse labels with NaN)
            delta_pred_dict = self(mol_emb, edit_emb_tuple)

            # Compute weighted loss across tasks (only for non-NaN labels)
            total_loss = 0.0
            total_weight = 0.0
            n_tasks_with_data = 0

            for i, task_name in enumerate(self.task_names):
                delta_task = delta_y[:, i]
                delta_pred_task = delta_pred_dict[task_name]

                # Filter out NaN values
                mask = ~torch.isnan(delta_task)
                if mask.sum() > 0:
                    delta_task_valid = delta_task[mask]
                    delta_pred_task_valid = delta_pred_task[mask]

                    task_loss = nn.functional.mse_loss(delta_pred_task_valid, delta_task_valid)
                    task_mae = nn.functional.l1_loss(delta_pred_task_valid, delta_task_valid)

                    weight = self.task_weights[task_name]
                    total_loss += weight * task_loss
                    total_weight += weight
                    n_tasks_with_data += 1

                    self.log(f'train_loss_{task_name}', task_loss)
                    self.log(f'train_mae_{task_name}', task_mae)

            # Average loss across tasks that have data
            if total_weight > 0:
                loss = total_loss / total_weight
            else:
                loss = torch.tensor(0.0, device=total_loss.device)

            self.log('train_loss', loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        # Handle trainable edit embeddings differently
        if self.trainable_edit_layer is not None:
            # Mode 2: batch is (mol_a_emb, mol_b_emb, edit_frag_a_emb, edit_frag_b_emb, delta_y)
            if len(batch) == 5:
                mol_a_emb, mol_b_emb, edit_frag_a_emb, edit_frag_b_emb, delta_y = batch
                mol_emb = mol_a_emb
                edit_emb_tuple = (mol_a_emb, mol_b_emb, edit_frag_a_emb, edit_frag_b_emb)
            # Mode 1: batch is (mol_a_emb, mol_b_emb, delta_y)
            elif len(batch) == 3:
                mol_a_emb, mol_b_emb, delta_y = batch
                mol_emb = mol_a_emb
                edit_emb_tuple = (mol_a_emb, mol_b_emb)
            else:
                raise ValueError(f"Expected batch to have 3 or 5 elements for trainable edits, got {len(batch)}")
        else:
            # For precomputed edits: batch is (mol_emb, edit_emb, delta_y)
            mol_emb, edit_emb_tuple, delta_y = batch

        if self.n_tasks == 1:
            # Single-task validation
            delta_pred = self(mol_emb, edit_emb_tuple)
            loss = nn.functional.mse_loss(delta_pred, delta_y)

            self.log('val_loss', loss, prog_bar=True)
            self.log('val_mae', nn.functional.l1_loss(delta_pred, delta_y), prog_bar=True)

        else:
            # Multi-task validation (handles sparse labels with NaN)
            delta_pred_dict = self(mol_emb, edit_emb_tuple)

            total_loss = 0.0
            total_weight = 0.0

            for i, task_name in enumerate(self.task_names):
                delta_task = delta_y[:, i]
                delta_pred_task = delta_pred_dict[task_name]

                # Filter out NaN values
                mask = ~torch.isnan(delta_task)
                if mask.sum() > 0:
                    delta_task_valid = delta_task[mask]
                    delta_pred_task_valid = delta_pred_task[mask]

                    task_loss = nn.functional.mse_loss(delta_pred_task_valid, delta_task_valid)
                    task_mae = nn.functional.l1_loss(delta_pred_task_valid, delta_task_valid)

                    weight = self.task_weights[task_name]
                    total_loss += weight * task_loss
                    total_weight += weight

                    self.log(f'val_loss_{task_name}', task_loss)
                    self.log(f'val_mae_{task_name}', task_mae, prog_bar=True)

            if total_weight > 0:
                loss = total_loss / total_weight
            else:
                loss = torch.tensor(0.0, device=total_loss.device)

            self.log('val_loss', loss, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        mol_emb, edit_emb, delta_y = batch

        if self.n_tasks == 1:
            # Single-task testing
            delta_pred = self(mol_emb, edit_emb)
            loss = nn.functional.mse_loss(delta_pred, delta_y)

            self.log('test_loss', loss)
            self.log('test_mae', nn.functional.l1_loss(delta_pred, delta_y))
            self.log('test_rmse', torch.sqrt(loss))

        else:
            # Multi-task testing
            delta_pred_dict = self(mol_emb, edit_emb)

            total_loss = 0.0
            for i, task_name in enumerate(self.task_names):
                delta_task = delta_y[:, i]
                delta_pred_task = delta_pred_dict[task_name]

                task_loss = nn.functional.mse_loss(delta_pred_task, delta_task)
                task_mae = nn.functional.l1_loss(delta_pred_task, delta_task)
                task_rmse = torch.sqrt(task_loss)

                weight = self.task_weights[task_name]
                total_loss += weight * task_loss

                self.log(f'test_loss_{task_name}', task_loss)
                self.log(f'test_mae_{task_name}', task_mae)
                self.log(f'test_rmse_{task_name}', task_rmse)

            total_weight = sum(self.task_weights.values())
            loss = total_loss / total_weight

            self.log('test_loss', loss)

        return loss

    def configure_optimizers(self):
        # Check if we need separate learning rates for GNN
        if self.mol_embedder is not None and hasattr(self.mol_embedder, 'trainable') and self.mol_embedder.trainable:
            # Separate learning rates for GNN and MLP heads
            param_groups = []

            # GNN parameters (lower learning rate)
            if hasattr(self.mol_embedder, 'message_passing'):
                gnn_params = list(self.mol_embedder.message_passing.parameters())
                gnn_trainable_params = [p for p in gnn_params if p.requires_grad]

                if gnn_trainable_params:
                    # Count GNN parameters
                    gnn_param_count = sum(p.numel() for p in gnn_trainable_params)

                    param_groups.append({
                        'params': gnn_trainable_params,
                        'lr': self.gnn_learning_rate,
                        'name': 'gnn'
                    })
                    print(f"\n{'='*70}")
                    print(f"OPTIMIZER SETUP:")
                    print(f"  → GNN: {len(gnn_trainable_params)} tensors, {gnn_param_count:,} params (lr={self.gnn_learning_rate})")
                else:
                    print(f"\n{'='*70}")
                    print(f"OPTIMIZER SETUP:")
                    print(f"  ⚠️  WARNING: GNN has NO trainable parameters (all frozen)!")

            # All other parameters (MLP heads, edit embedder, etc.) - higher learning rate
            mlp_params = []
            for name, param in self.named_parameters():
                if not param.requires_grad:
                    continue

                # Skip GNN parameters (already added)
                is_gnn_param = False
                if hasattr(self.mol_embedder, 'message_passing'):
                    for gnn_param in self.mol_embedder.message_passing.parameters():
                        if param is gnn_param:
                            is_gnn_param = True
                            break

                if not is_gnn_param:
                    mlp_params.append(param)

            if mlp_params:
                mlp_param_count = sum(p.numel() for p in mlp_params)
                param_groups.append({
                    'params': mlp_params,
                    'lr': self.learning_rate,
                    'name': 'mlp_heads'
                })
                print(f"  → MLP: {len(mlp_params)} tensors, {mlp_param_count:,} params (lr={self.learning_rate})")
                total_params = gnn_param_count + mlp_param_count if gnn_trainable_params else mlp_param_count
                print(f"  → TOTAL: {total_params:,} trainable parameters")
                print(f"{'='*70}\n")

            optimizer = torch.optim.Adam(param_groups)
        else:
            # Single learning rate for all parameters
            mlp_param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f"\n{'='*70}")
            print(f"OPTIMIZER SETUP:")
            print(f"  → MLP only: {mlp_param_count:,} params (lr={self.learning_rate})")
            print(f"  → GNN: FROZEN (not included in optimizer)")
            print(f"{'='*70}\n")
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        # Learning rate scheduler with plateau reduction
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }


class EditEffectPredictor:
    """
    High-level wrapper for causal edit effect prediction.

    This model learns F(m, e, c) = E[ΔY | m, e, c], the expected change
    in property Y when applying edit e to molecule m in context c.

    Supports:
    - Trainable edit embeddings (refined through backpropagation)
    - Multi-task learning (predict multiple property changes jointly)

    Single-task usage:
        >>> from src.embedding import RNAFMEmbedder, RNAEditEmbedder
        >>> from src.models.edit_effect_predictor import EditEffectPredictor
        >>>
        >>> # Create embedders
        >>> mol_embedder = FingerprintEmbedder(fp_type='morgan', radius=2, n_bits=512)
        >>> edit_embedder = EditEmbedder(mol_embedder)
        >>>
        >>> # Create predictor
        >>> predictor = EditEffectPredictor(
        ...     mol_embedder=mol_embedder,
        ...     edit_embedder=edit_embedder
        ... )
        >>>
        >>> # Train on paired data
        >>> smiles_a = ['CCO', 'c1ccccc1', ...]  # Before edit
        >>> smiles_b = ['CC(=O)O', 'c1ccncc1', ...]  # After edit
        >>> delta_y = [1.5, -0.8, ...]  # Property changes
        >>> predictor.fit(smiles_a, smiles_b, delta_y)
        >>>
        >>> # Predict edit effect
        >>> delta_pred = predictor.predict('CCO', 'CC(=O)O')

    Multi-task with trainable edits usage:
        >>> predictor = EditEffectPredictor(
        ...     mol_embedder=mol_embedder,
        ...     edit_embedder=edit_embedder,
        ...     trainable_edit_embeddings=True,
        ...     task_names=['logP', 'QED', 'SAS']
        ... )
        >>> delta_y = np.array([[1.5, 0.1, -0.3], [-0.8, -0.2, 0.5], ...])  # shape: (n, 3)
        >>> predictor.fit(smiles_a, smiles_b, delta_y)
        >>> delta_pred = predictor.predict('CCO', 'CC(=O)O')  # Returns dict
    """

    def __init__(
        self,
        mol_embedder,
        edit_embedder,
        hidden_dims: Optional[List[int]] = None,
        head_hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        gnn_learning_rate: Optional[float] = None,
        batch_size: int = 32,
        max_epochs: int = 100,
        device: Optional[str] = None,
        trainable_edit_embeddings: bool = False,
        trainable_edit_hidden_dims: Optional[List[int]] = None,
        trainable_edit_use_fragments: bool = False,
        task_names: Optional[List[str]] = None,
        task_weights: Optional[dict] = None
    ):
        """
        Initialize edit effect predictor.

        Args:
            mol_embedder: MoleculeEmbedder instance
            edit_embedder: EditEmbedder instance (not used if trainable_edit_embeddings=True)
            hidden_dims: Hidden layer dimensions (None for auto)
            head_hidden_dims: Hidden dims for task heads (e.g., [256, 256, 256, 128])
            dropout: Dropout probability
            learning_rate: Learning rate for MLP heads (default: 1e-3)
            gnn_learning_rate: Learning rate for GNN parameters (default: 1e-5)
                              Only used if mol_embedder has trainable GNN
            batch_size: Batch size
            max_epochs: Maximum training epochs
            device: 'cuda', 'cpu', or None (auto-detect)
            trainable_edit_embeddings: If True, use trainable edit embeddings (refine through backprop)
            trainable_edit_hidden_dims: Hidden dims for trainable edit embedder
            trainable_edit_use_fragments: If True, use edit_smiles fragments (Mode 2)
            task_names: List of task names for multi-task learning (None for single-task)
            task_weights: Dict of task weights for multi-task loss weighting
        """
        self.mol_embedder = mol_embedder
        self.edit_embedder = edit_embedder
        self.hidden_dims = hidden_dims
        self.head_hidden_dims = head_hidden_dims
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.gnn_learning_rate = gnn_learning_rate if gnn_learning_rate is not None else 1e-5
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.trainable_edit_embeddings = trainable_edit_embeddings
        self.trainable_edit_hidden_dims = trainable_edit_hidden_dims
        self.trainable_edit_use_fragments = trainable_edit_use_fragments
        self.task_names = task_names
        self.task_weights = task_weights

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.model = None
        self.trainer = None
        self.n_tasks = len(task_names) if task_names is not None else 1

    def fit(
        self,
        smiles_a: Union[List[str], np.ndarray] = None,
        smiles_b: Union[List[str], np.ndarray] = None,
        delta_y: Union[List[float], np.ndarray] = None,
        smiles_a_val: Optional[Union[List[str], np.ndarray]] = None,
        smiles_b_val: Optional[Union[List[str], np.ndarray]] = None,
        delta_y_val: Optional[Union[List[float], np.ndarray]] = None,
        verbose: bool = True,
        # Pre-computed embeddings (alternative to SMILES)
        mol_emb_a: Optional[np.ndarray] = None,
        mol_emb_b: Optional[np.ndarray] = None,
        mol_emb_a_val: Optional[np.ndarray] = None,
        mol_emb_b_val: Optional[np.ndarray] = None,
        # Edit fragment embeddings (Mode 2 only)
        edit_emb_a: Optional[np.ndarray] = None,
        edit_emb_b: Optional[np.ndarray] = None,
        edit_emb_a_val: Optional[np.ndarray] = None,
        edit_emb_b_val: Optional[np.ndarray] = None
    ):
        """
        Train the model on molecular pairs.

        Args:
            smiles_a: SMILES before edit (reactants) - required if mol_emb_a not provided
            smiles_b: SMILES after edit (products) - required if mol_emb_b not provided
            delta_y: Property changes (y_b - y_a)
                    Single-task: shape (n_samples,) or (n_samples, 1)
                    Multi-task: shape (n_samples, n_tasks)
            smiles_a_val: Optional validation reactants
            smiles_b_val: Optional validation products
            delta_y_val: Optional validation deltas
            verbose: Show training progress
            mol_emb_a: Pre-computed embeddings for smiles_a (alternative to computing from SMILES)
            mol_emb_b: Pre-computed embeddings for smiles_b (alternative to computing from SMILES)
            mol_emb_a_val: Pre-computed validation embeddings for smiles_a_val
            mol_emb_b_val: Pre-computed validation embeddings for smiles_b_val
            edit_emb_a: Pre-computed edit fragment reactant embeddings (Mode 2 only)
            edit_emb_b: Pre-computed edit fragment product embeddings (Mode 2 only)
            edit_emb_a_val: Pre-computed validation edit fragment reactant embeddings
            edit_emb_b_val: Pre-computed validation edit fragment product embeddings
        """
        # Get molecule embeddings (either pre-computed or compute from SMILES)
        if mol_emb_a is not None and mol_emb_b is not None:
            print(f"Using pre-computed embeddings for {len(mol_emb_a)} training pairs")
            mol_emb_train = mol_emb_a
            mol_emb_train_b = mol_emb_b
        else:
            if smiles_a is None or smiles_b is None:
                raise ValueError("Must provide either (smiles_a, smiles_b) or (mol_emb_a, mol_emb_b)")
            print(f"Embedding {len(smiles_a)} training molecule pairs...")
            mol_emb_train = self.mol_embedder.encode(smiles_a)
            mol_emb_train_b = self.mol_embedder.encode(smiles_b) if self.trainable_edit_embeddings else None

        # Handle trainable vs non-trainable edit embeddings
        if self.trainable_edit_embeddings:
            # For trainable edits, we need raw molecule embeddings (reactant, product)
            if mol_emb_train_b is None:
                mol_emb_train_b = self.mol_embedder.encode(smiles_b)

            # Mode 2: Use edit fragment embeddings
            if self.trainable_edit_use_fragments:
                if edit_emb_a is None or edit_emb_b is None:
                    raise ValueError(
                        "trainable_edit_use_fragments=True requires edit_emb_a and edit_emb_b. "
                        "Use scripts.prepare_edit_fragment_embeddings.embed_edit_fragments() to prepare them."
                    )
                # Store as tuple: (reactant_mol, product_mol, edit_fragment_reactant, edit_fragment_product)
                edit_emb_train = (mol_emb_train, mol_emb_train_b, edit_emb_a, edit_emb_b)
            else:
                # Mode 1: Use full molecule embeddings (original)
                edit_emb_train = (mol_emb_train, mol_emb_train_b)
        else:
            # Pre-compute edit embeddings (difference)
            if mol_emb_a is not None and mol_emb_b is not None:
                # Use pre-computed embeddings to compute edit
                edit_emb_train = mol_emb_b - mol_emb_a
            else:
                edit_emb_train = self.edit_embedder.encode_from_smiles(smiles_a, smiles_b)

        delta_y = np.array(delta_y, dtype=np.float32)

        # Handle multi-task labels
        if delta_y.ndim == 1:
            delta_y = delta_y.reshape(-1, 1) if self.n_tasks == 1 else delta_y.reshape(-1, 1)

        if delta_y.shape[1] != self.n_tasks:
            raise ValueError(f"delta_y has {delta_y.shape[1]} columns but expected {self.n_tasks} tasks")

        if self.n_tasks == 1:
            delta_y = delta_y.squeeze()

        # Convert to tensors
        mol_tensor = torch.FloatTensor(mol_emb_train)
        delta_tensor = torch.FloatTensor(delta_y)

        if self.trainable_edit_embeddings:
            # For trainable edits, store both reactant and product embeddings
            if self.trainable_edit_use_fragments:
                # Mode 2: 4-tuple (reactant, product, edit_fragment_react, edit_fragment_prod)
                reactant_tensor = torch.FloatTensor(edit_emb_train[0])
                product_tensor = torch.FloatTensor(edit_emb_train[1])
                edit_react_tensor = torch.FloatTensor(edit_emb_train[2])
                edit_prod_tensor = torch.FloatTensor(edit_emb_train[3])
                train_dataset = TensorDataset(reactant_tensor, product_tensor, edit_react_tensor, edit_prod_tensor, delta_tensor)
            else:
                # Mode 1: 2-tuple (reactant, product)
                reactant_tensor = torch.FloatTensor(edit_emb_train[0])
                product_tensor = torch.FloatTensor(edit_emb_train[1])
                train_dataset = TensorDataset(reactant_tensor, product_tensor, delta_tensor)
        else:
            # Non-trainable: pre-computed edit embeddings
            edit_tensor = torch.FloatTensor(edit_emb_train)
            train_dataset = TensorDataset(mol_tensor, edit_tensor, delta_tensor)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True  # Faster CPU→GPU transfer
        )

        # Validation data
        val_loader = None
        has_val = (smiles_a_val is not None or mol_emb_a_val is not None) and delta_y_val is not None

        if has_val:
            # Get validation embeddings (either pre-computed or compute from SMILES)
            if mol_emb_a_val is not None and mol_emb_b_val is not None:
                print(f"Using pre-computed embeddings for {len(mol_emb_a_val)} validation pairs")
                mol_emb_val = mol_emb_a_val
                mol_emb_val_b = mol_emb_b_val
            else:
                if smiles_a_val is None or smiles_b_val is None:
                    raise ValueError("Must provide either (smiles_a_val, smiles_b_val) or (mol_emb_a_val, mol_emb_b_val)")
                print(f"Embedding {len(smiles_a_val)} validation pairs...")
                mol_emb_val = self.mol_embedder.encode(smiles_a_val)
                mol_emb_val_b = self.mol_embedder.encode(smiles_b_val) if self.trainable_edit_embeddings else None

            if self.trainable_edit_embeddings:
                if mol_emb_val_b is None:
                    mol_emb_val_b = self.mol_embedder.encode(smiles_b_val)

                # Mode 2: Use edit fragment embeddings
                if self.trainable_edit_use_fragments:
                    if edit_emb_a_val is None or edit_emb_b_val is None:
                        raise ValueError(
                            "trainable_edit_use_fragments=True requires edit_emb_a_val and edit_emb_b_val"
                        )
                    edit_emb_val = (mol_emb_val, mol_emb_val_b, edit_emb_a_val, edit_emb_b_val)
                else:
                    # Mode 1: Use full molecule embeddings
                    edit_emb_val = (mol_emb_val, mol_emb_val_b)
            else:
                if mol_emb_a_val is not None and mol_emb_b_val is not None:
                    edit_emb_val = mol_emb_b_val - mol_emb_a_val
                else:
                    edit_emb_val = self.edit_embedder.encode_from_smiles(smiles_a_val, smiles_b_val)

            delta_y_val = np.array(delta_y_val, dtype=np.float32)

            if delta_y_val.ndim == 1:
                delta_y_val = delta_y_val.reshape(-1, 1) if self.n_tasks == 1 else delta_y_val.reshape(-1, 1)

            if delta_y_val.shape[1] != self.n_tasks:
                raise ValueError(f"delta_y_val has {delta_y_val.shape[1]} columns but expected {self.n_tasks} tasks")

            if self.n_tasks == 1:
                delta_y_val = delta_y_val.squeeze()

            mol_val_tensor = torch.FloatTensor(mol_emb_val)
            delta_val_tensor = torch.FloatTensor(delta_y_val)

            if self.trainable_edit_embeddings:
                if self.trainable_edit_use_fragments:
                    # Mode 2: 4-tuple
                    reactant_val_tensor = torch.FloatTensor(edit_emb_val[0])
                    product_val_tensor = torch.FloatTensor(edit_emb_val[1])
                    edit_react_val_tensor = torch.FloatTensor(edit_emb_val[2])
                    edit_prod_val_tensor = torch.FloatTensor(edit_emb_val[3])
                    val_dataset = TensorDataset(reactant_val_tensor, product_val_tensor, edit_react_val_tensor, edit_prod_val_tensor, delta_val_tensor)
                else:
                    # Mode 1: 2-tuple
                    reactant_val_tensor = torch.FloatTensor(edit_emb_val[0])
                    product_val_tensor = torch.FloatTensor(edit_emb_val[1])
                    val_dataset = TensorDataset(reactant_val_tensor, product_val_tensor, delta_val_tensor)
            else:
                edit_val_tensor = torch.FloatTensor(edit_emb_val)
                val_dataset = TensorDataset(mol_val_tensor, edit_val_tensor, delta_val_tensor)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                num_workers=4,
                pin_memory=True
            )

        # Initialize model
        mol_dim = mol_emb_train.shape[1]
        if self.trainable_edit_embeddings:
            edit_dim = mol_dim  # Edit dim same as mol dim for trainable embeddings
        else:
            edit_dim = edit_emb_train.shape[1]

        # Create trainable edit layer if requested
        trainable_edit_layer = None
        if self.trainable_edit_embeddings:
            try:
                from src.embedding.small_molecule.trainable_edit_embedder import TrainableEditEmbedder
            except ImportError:
                raise ImportError(
                    "TrainableEditEmbedder requires small molecule dependencies not available "
                    "in this RNA-only repo. Use RNA-specific edit embedders instead."
                )
            trainable_edit_layer = TrainableEditEmbedder(
                mol_dim=mol_dim,
                edit_dim=edit_dim,
                hidden_dims=self.trainable_edit_hidden_dims,
                dropout=self.dropout,
                use_edit_fragments=self.trainable_edit_use_fragments  # ← Pass fragment mode flag
            )

        self.model = EditEffectMLP(
            mol_dim=mol_dim,
            edit_dim=edit_dim,
            hidden_dims=self.hidden_dims,
            head_hidden_dims=self.head_hidden_dims,
            dropout=self.dropout,
            learning_rate=self.learning_rate,
            gnn_learning_rate=self.gnn_learning_rate,
            mol_embedder=self.mol_embedder,  # Pass embedder for parameter grouping
            trainable_edit_layer=trainable_edit_layer,
            n_tasks=self.n_tasks,
            task_names=self.task_names,
            task_weights=self.task_weights
        )

        # Determine embedder type for display
        embedder_type = "Unknown"
        if "morgan" in self.mol_embedder.name.lower():
            embedder_type = "Morgan Fingerprint"
        elif "chemberta" in self.mol_embedder.name.lower():
            embedder_type = "ChemBERTa Transformer"
        elif "chemprop" in self.mol_embedder.name.lower():
            embedder_type = "ChemProp D-MPNN"
        elif "rdkit" in self.mol_embedder.name.lower():
            embedder_type = "RDKit Fingerprint"
        elif "maccs" in self.mol_embedder.name.lower():
            embedder_type = "MACCS Keys"

        print(f"\nModel architecture:")
        print(f"  {'='*70}")
        print(f"  EMBEDDER: {embedder_type}")
        print(f"  Molecule embedding size: {mol_dim}")
        print(f"  {'='*70}")
        print(f"  Molecule embedder: {self.mol_embedder.name}")
        if self.trainable_edit_embeddings:
            print(f"  Edit embedder: TRAINABLE (learned)")
            print(f"    → Input: reactant({mol_dim}) + product({mol_dim})")
            print(f"    → Trainable layers: {self.trainable_edit_hidden_dims}")
            print(f"    → Output edit embedding: {edit_dim}")
        else:
            print(f"  Edit embedder: PRECOMPUTED (difference)")
            print(f"    → Edit embedding dim: {edit_dim}")
        print(f"  Combined predictor input: {mol_dim} + {edit_dim} = {mol_dim + edit_dim}")
        print(f"  Shared backbone: {self.model.hidden_dims}")
        if self.n_tasks == 1:
            print(f"  Output: 1 (single-task)")
        else:
            print(f"  Multi-task heads: {self.n_tasks} tasks")
            for task in self.task_names:
                print(f"    → {task}")
        print(f"  {'='*70}")
        print(f"  Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  {'='*70}")

        # Trainer with CSV logger for tracking metrics
        from pytorch_lightning.loggers import CSVLogger
        import tempfile
        import os

        # Create temporary directory for logs
        self.log_dir = tempfile.mkdtemp(prefix='edit_effect_')
        csv_logger = CSVLogger(self.log_dir, name='training')

        self.trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            accelerator='gpu' if self.device == 'cuda' else 'cpu',
            devices=1,
            enable_progress_bar=verbose,
            enable_model_summary=verbose,
            logger=csv_logger,
            enable_checkpointing=False
        )

        # Train
        print(f"\nTraining for up to {self.max_epochs} epochs...")
        self.trainer.fit(self.model, train_loader, val_loader)

        print("Training complete!")

    def get_training_history(self):
        """
        Get training history from CSV logger.

        Returns:
            pandas.DataFrame with training/validation metrics per epoch
        """
        import pandas as pd
        import os

        if not hasattr(self, 'log_dir') or not hasattr(self, 'trainer'):
            raise RuntimeError("Model has not been trained yet")

        # Find metrics CSV file
        csv_path = os.path.join(self.log_dir, 'training', 'version_0', 'metrics.csv')

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"No training metrics found at {csv_path}")

        # Load metrics
        df = pd.read_csv(csv_path)

        return df

    def predict(
        self,
        smiles_a: Union[str, List[str]],
        smiles_b: Union[str, List[str]],
        mol_emb_a: Optional[np.ndarray] = None,
        mol_emb_b: Optional[np.ndarray] = None,
        edit_frag_a_emb: Optional[np.ndarray] = None,
        edit_frag_b_emb: Optional[np.ndarray] = None
    ) -> Union[np.ndarray, dict]:
        """
        Predict edit effects.

        Args:
            smiles_a: Molecule before edit (single or list)
            smiles_b: Molecule after edit (single or list)
            mol_emb_a: Pre-computed embeddings for smiles_a (optional)
            mol_emb_b: Pre-computed embeddings for smiles_b (optional)
            edit_frag_a_emb: Pre-computed edit fragment A embeddings (Mode 2 only)
            edit_frag_b_emb: Pre-computed edit fragment B embeddings (Mode 2 only)

        Returns:
            Single-task: numpy array of predictions
            Multi-task: dict {task_name: predictions}
        """
        if self.model is None:
            raise RuntimeError("Model not trained yet!")

        is_single = isinstance(smiles_a, str)
        if is_single:
            smiles_a = [smiles_a]
            smiles_b = [smiles_b]

        # Get molecule embeddings (pre-computed or compute from SMILES)
        if mol_emb_a is not None:
            mol_emb_a_array = mol_emb_a
        else:
            mol_emb_a_array = self.mol_embedder.encode(smiles_a)

        # Handle trainable vs non-trainable edit embeddings
        if self.trainable_edit_embeddings:
            if mol_emb_b is not None:
                mol_emb_b_array = mol_emb_b
            else:
                mol_emb_b_array = self.mol_embedder.encode(smiles_b)

            reactant_tensor = torch.FloatTensor(mol_emb_a_array).to(self.device)
            product_tensor = torch.FloatTensor(mol_emb_b_array).to(self.device)

            # Mode 2: Use edit fragments if provided
            if self.trainable_edit_use_fragments and edit_frag_a_emb is not None and edit_frag_b_emb is not None:
                edit_frag_a_tensor = torch.FloatTensor(edit_frag_a_emb).to(self.device)
                edit_frag_b_tensor = torch.FloatTensor(edit_frag_b_emb).to(self.device)
                edit_input = (reactant_tensor, product_tensor, edit_frag_a_tensor, edit_frag_b_tensor)
            else:
                # Mode 1: Use full molecule embeddings
                edit_input = (reactant_tensor, product_tensor)
        else:
            edit_emb = self.edit_embedder.encode_from_smiles(smiles_a, smiles_b)
            edit_input = torch.FloatTensor(edit_emb).to(self.device)

        mol_tensor = torch.FloatTensor(mol_emb_a_array).to(self.device)

        # Predict
        self.model.eval()
        self.model.to(self.device)

        with torch.no_grad():
            if self.n_tasks == 1:
                # Single-task: return numpy array
                delta_pred = self.model(mol_tensor, edit_input).cpu().numpy()
                if is_single:
                    return delta_pred[0]
                return delta_pred
            else:
                # Multi-task: return dict
                delta_pred_dict = self.model(mol_tensor, edit_input)
                result = {}
                for task_name in self.task_names:
                    preds = delta_pred_dict[task_name].cpu().numpy()
                    if is_single:
                        result[task_name] = preds[0]
                    else:
                        result[task_name] = preds
                return result

    def predict_from_edit_smiles(
        self,
        smiles_a: Union[str, List[str]],
        edit_smiles: Union[str, List[str]]
    ) -> np.ndarray:
        """
        Predict edit effects using reaction SMILES format.

        Args:
            smiles_a: Starting molecule(s)
            edit_smiles: Edit in format "reactant>>product"

        Returns:
            Predicted delta (property change)
        """
        is_single = isinstance(smiles_a, str)

        if is_single:
            if '>>' not in edit_smiles:
                raise ValueError("edit_smiles must be in format 'reactant>>product'")
            _, prod = edit_smiles.split('>>')
            return self.predict(smiles_a, prod)
        else:
            smiles_b = []
            for edit in edit_smiles:
                if '>>' not in edit:
                    raise ValueError("edit_smiles must be in format 'reactant>>product'")
                _, prod = edit.split('>>')
                smiles_b.append(prod)

            return self.predict(smiles_a, smiles_b)

    def save_checkpoint(self, path: str):
        """
        Save model checkpoint to disk.

        Saves only the model state dict and hyperparameters, avoiding
        pickle issues with embedder objects.

        Args:
            path: Path to save checkpoint file

        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'hyperparameters': {
                'mol_dim': self.model.mol_dim,
                'edit_dim': self.model.edit_dim,
                'hidden_dims': self.hidden_dims,
                'dropout': self.dropout,
                'learning_rate': self.learning_rate,
                'trainable_edit_embeddings': self.trainable_edit_embeddings,
                'trainable_edit_hidden_dims': self.trainable_edit_hidden_dims,
                'task_names': self.task_names,
                'task_weights': self.task_weights,
                'n_tasks': self.n_tasks
            }
        }

        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(
        cls,
        path: str,
        mol_embedder,
        edit_embedder=None,
        device: Optional[str] = None
    ):
        """
        Load model checkpoint from disk.

        Args:
            path: Path to checkpoint file
            mol_embedder: MoleculeEmbedder instance (must match training)
            edit_embedder: EditEmbedder instance (required if trainable_edit_embeddings=False)
            device: Device to load model on ('cuda', 'cpu', or None for auto-detect)

        Returns:
            Loaded EditEffectPredictor instance

        """
        checkpoint = torch.load(path, map_location='cpu')
        hparams = checkpoint['hyperparameters']

        # Create predictor instance
        predictor = cls(
            mol_embedder=mol_embedder,
            edit_embedder=edit_embedder,
            hidden_dims=hparams['hidden_dims'],
            dropout=hparams['dropout'],
            learning_rate=hparams['learning_rate'],
            batch_size=32,  # Not saved, use default
            max_epochs=100,  # Not saved, use default
            device=device,
            trainable_edit_embeddings=hparams['trainable_edit_embeddings'],
            trainable_edit_hidden_dims=hparams['trainable_edit_hidden_dims'],
            task_names=hparams['task_names'],
            task_weights=hparams['task_weights']
        )

        # Recreate model structure
        mol_dim = hparams['mol_dim']
        edit_dim = hparams['edit_dim']

        # Create trainable edit layer if needed
        trainable_edit_layer = None
        if hparams['trainable_edit_embeddings']:
            try:
                from src.embedding.small_molecule.trainable_edit_embedder import TrainableEditEmbedder
            except ImportError:
                raise ImportError(
                    "TrainableEditEmbedder requires small molecule dependencies not available "
                    "in this RNA-only repo. Use RNA-specific edit embedders instead."
                )
            trainable_edit_layer = TrainableEditEmbedder(
                mol_dim=mol_dim,
                hidden_dims=hparams['trainable_edit_hidden_dims'],
                edit_dim=edit_dim
            )

        # Recreate PyTorch Lightning model
        predictor.model = EditEffectMLP(
            mol_dim=mol_dim,
            edit_dim=edit_dim,
            hidden_dims=hparams['hidden_dims'],
            dropout=hparams['dropout'],
            learning_rate=hparams['learning_rate'],
            trainable_edit_layer=trainable_edit_layer,
            n_tasks=hparams['n_tasks'],
            task_names=hparams['task_names'],
            task_weights=hparams['task_weights']
        )

        # Load state dict
        predictor.model.load_state_dict(checkpoint['model_state_dict'])
        predictor.model.to(predictor.device)
        predictor.model.eval()

        return predictor
