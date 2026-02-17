"""
End-to-end trainable edit effect predictor with trainable encoder backbone.

This predictor computes embeddings on-the-fly during training, allowing
gradients to flow back through the encoder (GNN or transformer) for joint optimization.

Unlike EditEffectPredictor which uses pre-computed embeddings, this class:
- Stores the embedder as part of the model
- Computes embeddings in the forward pass
- Supports separate learning rates for encoder and MLP

Supports any trainable encoder that implements:
- trainable: bool attribute
- encode_trainable(smiles) -> torch.Tensor method
- get_encoder_parameters() -> List[Parameter] method

Usage:
    from src.models.predictors.trainable_edit_effect_predictor import TrainableEditEffectPredictor

    embedder = create_embedder('chemprop_dmpnn', trainable_encoder=True)
    # or: embedder = create_embedder('chemberta', trainable_encoder=True)
    predictor = TrainableEditEffectPredictor(
        embedder=embedder,
        encoder_learning_rate=1e-5,  # Lower LR for encoder (GNN/transformer)
        mlp_learning_rate=1e-3,      # Higher LR for MLP heads
        use_edit_fragments=True,     # Optional: use fragment embeddings
        ...
    )
    predictor.fit(smiles_a_train, smiles_b_train, delta_y_train, ...)
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
from typing import Optional, List, Union, Tuple
from torch.utils.data import DataLoader, Dataset


class SMILESPairDataset(Dataset):
    """Dataset that stores SMILES pairs for on-the-fly embedding."""

    def __init__(
        self,
        smiles_a: List[str],
        smiles_b: List[str],
        delta_y: np.ndarray,
        edit_smiles_a: Optional[List[str]] = None,
        edit_smiles_b: Optional[List[str]] = None
    ):
        """
        Args:
            smiles_a: List of SMILES strings (before edit)
            smiles_b: List of SMILES strings (after edit)
            delta_y: Target deltas [n_samples] for single-task or [n_samples, n_tasks] for multi-task
            edit_smiles_a: Optional edit fragment SMILES (removed part)
            edit_smiles_b: Optional edit fragment SMILES (added part)
        """
        self.smiles_a = smiles_a
        self.smiles_b = smiles_b
        self.delta_y = torch.tensor(delta_y, dtype=torch.float32)
        self.edit_smiles_a = edit_smiles_a
        self.edit_smiles_b = edit_smiles_b
        self.use_fragments = edit_smiles_a is not None and edit_smiles_b is not None

    def __len__(self):
        return len(self.smiles_a)

    def __getitem__(self, idx):
        if self.use_fragments:
            return (
                self.smiles_a[idx],
                self.smiles_b[idx],
                self.edit_smiles_a[idx],
                self.edit_smiles_b[idx],
                self.delta_y[idx]
            )
        else:
            return (
                self.smiles_a[idx],
                self.smiles_b[idx],
                self.delta_y[idx]
            )


def smiles_pair_collate_fn(batch):
    """Custom collate function to batch SMILES pairs."""
    if len(batch[0]) == 5:
        # With fragments
        smiles_a = [item[0] for item in batch]
        smiles_b = [item[1] for item in batch]
        edit_smiles_a = [item[2] for item in batch]
        edit_smiles_b = [item[3] for item in batch]
        delta_y = torch.stack([item[4] for item in batch])
        return smiles_a, smiles_b, edit_smiles_a, edit_smiles_b, delta_y
    else:
        # Without fragments
        smiles_a = [item[0] for item in batch]
        smiles_b = [item[1] for item in batch]
        delta_y = torch.stack([item[2] for item in batch])
        return smiles_a, smiles_b, delta_y


class TrainableEditEffectMLP(pl.LightningModule):
    """
    End-to-end trainable edit effect predictor with encoder + MLP.

    The encoder (GNN or transformer) is part of the model and gets updated during training.
    Uses separate learning rates for encoder and MLP parameters.

    Supports two modes:
    - Mode 1: Edit embedding from (mol_a, mol_b) difference
    - Mode 2: Edit embedding from (mol_a, mol_b, frag_a, frag_b) with trainable combiner
    """

    def __init__(
        self,
        embedder: nn.Module,
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        head_hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
        encoder_learning_rate: float = 1e-5,
        mlp_learning_rate: float = 1e-3,
        n_tasks: int = 1,
        task_names: Optional[List[str]] = None,
        task_weights: Optional[dict] = None,
        use_edit_fragments: bool = False,
        trainable_edit_hidden_dims: Optional[List[int]] = None
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['embedder'])

        self.embedder = embedder
        self.input_dim = input_dim  # mol_dim
        self.encoder_learning_rate = encoder_learning_rate
        self.mlp_learning_rate = mlp_learning_rate
        self.n_tasks = n_tasks
        self.use_edit_fragments = use_edit_fragments

        # For trainable edit effects, edit_dim = mol_dim
        self.edit_dim = input_dim
        self.combined_dim = input_dim + self.edit_dim  # mol_a + edit

        # Task names
        if task_names is None:
            if n_tasks == 1:
                self.task_names = ['delta_property']
            else:
                self.task_names = [f'task_{i}' for i in range(n_tasks)]
        else:
            self.task_names = task_names

        # Task weights for loss
        if task_weights is None:
            self.task_weights = {name: 1.0 for name in self.task_names}
        else:
            self.task_weights = task_weights

        # Build trainable edit combiner if using fragments
        if use_edit_fragments:
            try:
                from src.embedding.small_molecule.trainable_edit_embedder import TrainableEditEmbedder
            except ImportError:
                raise ImportError(
                    "TrainableEditEmbedder requires small molecule dependencies not available "
                    "in this RNA-only repo. Use RNA-specific edit embedders instead."
                )
            self.trainable_edit_layer = TrainableEditEmbedder(
                mol_dim=input_dim,
                edit_dim=self.edit_dim,
                hidden_dims=trainable_edit_hidden_dims,
                dropout=dropout,
                use_edit_fragments=True
            )
        else:
            # For Mode 1, edit = mol_b - mol_a (no trainable layer needed)
            self.trainable_edit_layer = None

        # Auto-generate hidden dims if not provided
        if hidden_dims is None:
            hidden_dims = []
            current_dim = self.combined_dim
            for _ in range(3):
                current_dim = current_dim // 2
                if current_dim < 64:
                    break
                hidden_dims.append(current_dim)
            if len(hidden_dims) == 0:
                hidden_dims = [max(self.combined_dim // 2, 64)]

        self.hidden_dims = hidden_dims

        # Build network
        if n_tasks == 1:
            layers = []
            prev_dim = self.combined_dim

            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ])
                prev_dim = hidden_dim

            layers.append(nn.Linear(prev_dim, 1))
            self.network = nn.Sequential(*layers)
            self.multi_task_network = None
        else:
            from ..architectures.multi_head import MultiTaskNetwork

            shared_dim = hidden_dims[-1] if hidden_dims else max(self.combined_dim // 4, 64)
            head_hidden_dim = max(shared_dim // 2, 32) if head_hidden_dims is None else None

            self.multi_task_network = MultiTaskNetwork(
                input_dim=self.combined_dim,
                task_names=self.task_names,
                backbone_hidden_dims=hidden_dims,
                shared_dim=shared_dim,
                head_hidden_dim=head_hidden_dim,
                head_hidden_dims=head_hidden_dims,
                dropout=dropout
            )
            self.network = None

    def forward(
        self,
        smiles_a: List[str],
        smiles_b: List[str],
        edit_smiles_a: Optional[List[str]] = None,
        edit_smiles_b: Optional[List[str]] = None
    ):
        """
        Forward pass: (smiles_a, smiles_b) → embeddings → delta_predictions

        Args:
            smiles_a: List of SMILES strings (before edit)
            smiles_b: List of SMILES strings (after edit)
            edit_smiles_a: Optional edit fragment SMILES (removed part)
            edit_smiles_b: Optional edit fragment SMILES (added part)

        Returns:
            Single-task: Predicted delta [batch_size]
            Multi-task: Dict {task_name: predictions [batch_size]}
        """
        # Compute molecule embeddings on-the-fly with gradient tracking
        mol_a_emb = self.embedder.encode_trainable(smiles_a)
        mol_b_emb = self.embedder.encode_trainable(smiles_b)

        # Compute edit embedding
        if self.use_edit_fragments and edit_smiles_a is not None and edit_smiles_b is not None:
            # Mode 2: Use fragments with trainable combiner
            edit_frag_a_emb = self.embedder.encode_trainable(edit_smiles_a)
            edit_frag_b_emb = self.embedder.encode_trainable(edit_smiles_b)
            edit_emb = self.trainable_edit_layer(mol_a_emb, mol_b_emb, edit_frag_a_emb, edit_frag_b_emb)
        else:
            # Mode 1: Edit = difference of molecule embeddings
            edit_emb = mol_b_emb - mol_a_emb

        # Concatenate molecule A and edit embeddings
        x = torch.cat([mol_a_emb, edit_emb], dim=-1)

        # Forward through network
        if self.n_tasks == 1:
            return self.network(x).squeeze(-1)
        else:
            return self.multi_task_network(x)

    def training_step(self, batch, batch_idx):
        if len(batch) == 5:
            # With fragments
            smiles_a, smiles_b, edit_smiles_a, edit_smiles_b, delta_y = batch
        else:
            # Without fragments
            smiles_a, smiles_b, delta_y = batch
            edit_smiles_a, edit_smiles_b = None, None

        if self.n_tasks == 1:
            delta_pred = self(smiles_a, smiles_b, edit_smiles_a, edit_smiles_b)
            loss = nn.functional.mse_loss(delta_pred, delta_y)
            self.log('train_loss', loss, prog_bar=True)
            self.log('train_mae', nn.functional.l1_loss(delta_pred, delta_y))
        else:
            delta_pred_dict = self(smiles_a, smiles_b, edit_smiles_a, edit_smiles_b)

            total_loss = 0.0
            total_weight = 0.0

            for i, task_name in enumerate(self.task_names):
                delta_task = delta_y[:, i]
                delta_pred_task = delta_pred_dict[task_name]

                mask = ~torch.isnan(delta_task)
                if mask.sum() > 0:
                    delta_task_valid = delta_task[mask]
                    delta_pred_task_valid = delta_pred_task[mask]

                    task_loss = nn.functional.mse_loss(delta_pred_task_valid, delta_task_valid)
                    weight = self.task_weights[task_name]
                    total_loss += weight * task_loss
                    total_weight += weight

                    self.log(f'train_loss_{task_name}', task_loss)

            if total_weight > 0:
                loss = total_loss / total_weight
            else:
                loss = torch.tensor(0.0, device=self.device, requires_grad=True)

            self.log('train_loss', loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        if len(batch) == 5:
            smiles_a, smiles_b, edit_smiles_a, edit_smiles_b, delta_y = batch
        else:
            smiles_a, smiles_b, delta_y = batch
            edit_smiles_a, edit_smiles_b = None, None

        if self.n_tasks == 1:
            delta_pred = self(smiles_a, smiles_b, edit_smiles_a, edit_smiles_b)
            loss = nn.functional.mse_loss(delta_pred, delta_y)
            self.log('val_loss', loss, prog_bar=True)
            self.log('val_mae', nn.functional.l1_loss(delta_pred, delta_y), prog_bar=True)
        else:
            delta_pred_dict = self(smiles_a, smiles_b, edit_smiles_a, edit_smiles_b)

            total_loss = 0.0
            total_weight = 0.0

            for i, task_name in enumerate(self.task_names):
                delta_task = delta_y[:, i]
                delta_pred_task = delta_pred_dict[task_name]

                mask = ~torch.isnan(delta_task)
                if mask.sum() > 0:
                    delta_task_valid = delta_task[mask]
                    delta_pred_task_valid = delta_pred_task[mask]

                    task_loss = nn.functional.mse_loss(delta_pred_task_valid, delta_task_valid)
                    weight = self.task_weights[task_name]
                    total_loss += weight * task_loss
                    total_weight += weight

                    self.log(f'val_loss_{task_name}', task_loss)
                    self.log(f'val_mae_{task_name}',
                            nn.functional.l1_loss(delta_pred_task_valid, delta_task_valid))

            if total_weight > 0:
                loss = total_loss / total_weight
            else:
                loss = torch.tensor(0.0, device=self.device)

            self.log('val_loss', loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        """Configure optimizer with separate learning rates for encoder and MLP."""
        param_groups = []

        # Get encoder parameters using generalized interface
        encoder_params = []
        if hasattr(self.embedder, 'get_encoder_parameters'):
            encoder_params = self.embedder.get_encoder_parameters()
        elif hasattr(self.embedder, 'message_passing') and self.embedder.trainable:
            # Fallback for backward compatibility with older embedders
            encoder_params = list(self.embedder.message_passing.parameters())

        encoder_trainable = [p for p in encoder_params if p.requires_grad]

        if encoder_trainable:
            encoder_param_count = sum(p.numel() for p in encoder_trainable)
            param_groups.append({
                'params': encoder_trainable,
                'lr': self.encoder_learning_rate,
                'name': 'encoder'
            })
            print(f"\n{'='*70}")
            print(f"OPTIMIZER SETUP (End-to-End Trainable Edit Effect):")
            encoder_type = type(self.embedder).__name__
            print(f"  → Encoder ({encoder_type}): {len(encoder_trainable)} tensors, {encoder_param_count:,} params (lr={self.encoder_learning_rate})")

        # MLP parameters (everything except encoder)
        mlp_params = []
        encoder_param_ids = {id(p) for p in encoder_params}

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if id(param) not in encoder_param_ids:
                mlp_params.append(param)

        if mlp_params:
            mlp_param_count = sum(p.numel() for p in mlp_params)
            param_groups.append({
                'params': mlp_params,
                'lr': self.mlp_learning_rate,
                'name': 'mlp_heads'
            })
            if encoder_trainable:
                print(f"  → MLP: {len(mlp_params)} tensors, {mlp_param_count:,} params (lr={self.mlp_learning_rate})")
                total = sum(p.numel() for g in param_groups for p in g['params'])
                print(f"  → TOTAL: {total:,} trainable parameters")
                print(f"{'='*70}\n")

        optimizer = torch.optim.Adam(param_groups)

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
                'monitor': 'train_loss',  # Use train_loss to avoid issues when no val data
                'frequency': 1,
                'strict': False  # Don't raise error if metric not found
            }
        }


class TrainableEditEffectPredictor:
    """
    High-level API for end-to-end trainable edit effect prediction.

    Unlike EditEffectPredictor, this class:
    - Does NOT pre-compute embeddings
    - Computes embeddings on-the-fly during training
    - Allows gradients to flow back through the encoder (GNN or transformer)
    - Uses separate learning rates for encoder and MLP

    Supports two modes:
    - Mode 1 (default): Edit embedding = mol_b - mol_a
    - Mode 2 (use_edit_fragments=True): Edit embedding computed from fragments

    Usage:
        embedder = create_embedder('chemprop_dmpnn', trainable_encoder=True)
        # or: embedder = create_embedder('chemberta', trainable_encoder=True)
        predictor = TrainableEditEffectPredictor(
            embedder=embedder,
            encoder_learning_rate=1e-5,
            mlp_learning_rate=1e-3,
            ...
        )
        predictor.fit(smiles_a_train, smiles_b_train, delta_y_train, ...)
        delta_pred = predictor.predict(smiles_a_test, smiles_b_test)
    """

    def __init__(
        self,
        embedder,
        hidden_dims: Optional[List[int]] = None,
        head_hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
        encoder_learning_rate: float = 1e-5,
        mlp_learning_rate: float = 1e-3,
        batch_size: int = 32,
        max_epochs: int = 100,
        device: Optional[str] = None,
        task_names: Optional[List[str]] = None,
        task_weights: Optional[dict] = None,
        use_edit_fragments: bool = False,
        trainable_edit_hidden_dims: Optional[List[int]] = None
    ):
        self.embedder = embedder
        self.hidden_dims = hidden_dims
        self.head_hidden_dims = head_hidden_dims
        self.dropout = dropout
        self.encoder_learning_rate = encoder_learning_rate
        self.mlp_learning_rate = mlp_learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.task_names = task_names
        self.task_weights = task_weights
        self.use_edit_fragments = use_edit_fragments
        self.trainable_edit_hidden_dims = trainable_edit_hidden_dims

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.model = None
        self.trainer = None
        self.n_tasks = len(task_names) if task_names is not None else 1

    def fit(
        self,
        smiles_a_train: List[str],
        smiles_b_train: List[str],
        delta_y_train: Union[List[float], np.ndarray],
        smiles_a_val: Optional[List[str]] = None,
        smiles_b_val: Optional[List[str]] = None,
        delta_y_val: Optional[Union[List[float], np.ndarray]] = None,
        edit_smiles_a_train: Optional[List[str]] = None,
        edit_smiles_b_train: Optional[List[str]] = None,
        edit_smiles_a_val: Optional[List[str]] = None,
        edit_smiles_b_val: Optional[List[str]] = None,
        verbose: bool = True
    ):
        """
        Train the model end-to-end.

        Args:
            smiles_a_train: Training SMILES strings (before edit)
            smiles_b_train: Training SMILES strings (after edit)
            delta_y_train: Training property deltas
            smiles_a_val: Optional validation SMILES strings (before edit)
            smiles_b_val: Optional validation SMILES strings (after edit)
            delta_y_val: Optional validation property deltas
            edit_smiles_a_train: Optional training edit fragment SMILES (removed)
            edit_smiles_b_train: Optional training edit fragment SMILES (added)
            edit_smiles_a_val: Optional validation edit fragment SMILES (removed)
            edit_smiles_b_val: Optional validation edit fragment SMILES (added)
            verbose: Show training progress
        """
        print(f"\n{'='*70}")
        print("End-to-End Trainable Edit Effect Predictor - Training")
        print(f"{'='*70}\n")

        delta_y_train = np.array(delta_y_train, dtype=np.float32)

        # Handle multi-task labels
        if delta_y_train.ndim == 1:
            delta_y_train = delta_y_train.reshape(-1, 1) if self.n_tasks == 1 else delta_y_train.reshape(-1, 1)

        if self.n_tasks == 1:
            delta_y_train = delta_y_train.squeeze()

        # Create datasets
        train_dataset = SMILESPairDataset(
            smiles_a_train, smiles_b_train, delta_y_train,
            edit_smiles_a_train, edit_smiles_b_train
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=smiles_pair_collate_fn,
            num_workers=0  # SMILES processing is fast
        )

        val_loader = None
        if smiles_a_val is not None and smiles_b_val is not None and delta_y_val is not None:
            delta_y_val = np.array(delta_y_val, dtype=np.float32)
            if delta_y_val.ndim == 1:
                delta_y_val = delta_y_val.reshape(-1, 1) if self.n_tasks == 1 else delta_y_val.reshape(-1, 1)
            if self.n_tasks == 1:
                delta_y_val = delta_y_val.squeeze()

            val_dataset = SMILESPairDataset(
                smiles_a_val, smiles_b_val, delta_y_val,
                edit_smiles_a_val, edit_smiles_b_val
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=smiles_pair_collate_fn,
                num_workers=0
            )

        # Initialize model
        input_dim = self.embedder.embedding_dim
        self.model = TrainableEditEffectMLP(
            embedder=self.embedder,
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            head_hidden_dims=self.head_hidden_dims,
            dropout=self.dropout,
            encoder_learning_rate=self.encoder_learning_rate,
            mlp_learning_rate=self.mlp_learning_rate,
            n_tasks=self.n_tasks,
            task_names=self.task_names,
            task_weights=self.task_weights,
            use_edit_fragments=self.use_edit_fragments,
            trainable_edit_hidden_dims=self.trainable_edit_hidden_dims
        )

        # Move embedder to device (handle both GNN and transformer embedders)
        if hasattr(self.embedder, 'message_passing'):
            self.embedder.message_passing.to(self.device)
        if hasattr(self.embedder, 'aggregation'):
            self.embedder.aggregation.to(self.device)
        if hasattr(self.embedder, 'model'):  # For transformer-based embedders (ChemBERTa)
            self.embedder.model.to(self.device)

        encoder_type = type(self.embedder).__name__
        print(f"\nModel architecture:")
        print(f"  {'='*70}")
        print(f"  Encoder: {encoder_type} (trainable={self.embedder.trainable})")
        print(f"  Embedding dim: {input_dim}")
        print(f"  Edit mode: {'Fragments (Mode 2)' if self.use_edit_fragments else 'Difference (Mode 1)'}")
        print(f"  Combined input dim: {self.model.combined_dim}")
        print(f"  Shared backbone: {self.model.hidden_dims}")
        if self.n_tasks == 1:
            print(f"  Output: 1 (single-task)")
        else:
            print(f"  Multi-task heads: {self.n_tasks} tasks")
        print(f"  {'='*70}")
        print(f"  Encoder learning rate: {self.encoder_learning_rate}")
        print(f"  MLP learning rate: {self.mlp_learning_rate}")
        print(f"  {'='*70}")

        # Trainer
        from pytorch_lightning.loggers import CSVLogger
        import tempfile

        self.log_dir = tempfile.mkdtemp(prefix='trainable_edit_pred_')
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

        print(f"\nTraining for up to {self.max_epochs} epochs...")
        self.trainer.fit(self.model, train_loader, val_loader)

        print("Training complete!")

    def predict(
        self,
        smiles_a: Union[str, List[str]],
        smiles_b: Union[str, List[str]],
        edit_smiles_a: Optional[Union[str, List[str]]] = None,
        edit_smiles_b: Optional[Union[str, List[str]]] = None
    ) -> Union[np.ndarray, dict]:
        """
        Predict edit effects.

        Args:
            smiles_a: Single SMILES or list of SMILES (before edit)
            smiles_b: Single SMILES or list of SMILES (after edit)
            edit_smiles_a: Optional edit fragment SMILES (removed part)
            edit_smiles_b: Optional edit fragment SMILES (added part)

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
            if edit_smiles_a is not None:
                edit_smiles_a = [edit_smiles_a]
            if edit_smiles_b is not None:
                edit_smiles_b = [edit_smiles_b]

        self.model.eval()
        self.model.to(self.device)

        # Move embedder components to device (handle both GNN and transformer embedders)
        if hasattr(self.embedder, 'message_passing'):
            self.embedder.message_passing.to(self.device)
        if hasattr(self.embedder, 'aggregation'):
            self.embedder.aggregation.to(self.device)
        if hasattr(self.embedder, 'model'):  # For transformer-based embedders (ChemBERTa)
            self.embedder.model.to(self.device)

        with torch.no_grad():
            if self.n_tasks == 1:
                delta_pred = self.model(smiles_a, smiles_b, edit_smiles_a, edit_smiles_b).cpu().numpy()
                if is_single:
                    return delta_pred[0]
                return delta_pred
            else:
                delta_pred_dict = self.model(smiles_a, smiles_b, edit_smiles_a, edit_smiles_b)
                result = {}
                for task_name in self.task_names:
                    preds = delta_pred_dict[task_name].cpu().numpy()
                    if is_single:
                        result[task_name] = preds[0]
                    else:
                        result[task_name] = preds
                return result

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

        csv_path = os.path.join(self.log_dir, 'training', 'version_0', 'metrics.csv')

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"No training metrics found at {csv_path}")

        return pd.read_csv(csv_path)
