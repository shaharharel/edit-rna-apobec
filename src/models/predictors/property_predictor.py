"""
Property predictor: SMILES → property value.

This is the baseline (non-causal) model that predicts Y(molecule).
For causal edit prediction, use EditEffectPredictor instead.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Optional, List, Union
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


class PropertyPredictorMLP(pl.LightningModule):
    """
    Multi-layer perceptron for property prediction: f(molecule) → Y

    Supports both single-task and multi-task learning:
    - Single task (n_tasks=1): Standard regression to single value
    - Multi-task (n_tasks>1): Shared backbone + task-specific heads

    Architecture automatically scales down from input_dim to 1:
    [input_dim] → [input_dim/2] → [input_dim/4] → ... → [1]

    Args:
        input_dim: Input embedding dimension
        hidden_dims: Optional list of hidden dimensions. If None, auto-generates halving layers
        head_hidden_dims: Optional list of hidden dims for task heads (e.g., [256, 256, 256, 128])
        dropout: Dropout probability
        learning_rate: Learning rate for Adam optimizer
        activation: Activation function ('relu', 'elu', 'gelu')
        n_tasks: Number of properties to predict (1 for single-task)
        task_names: Optional list of task names (e.g., ['logP', 'QED', 'SAS'])
        task_weights: Optional dict of task weights for loss {task_name: weight}
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        head_hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        activation: str = 'relu',
        n_tasks: int = 1,
        task_names: Optional[List[str]] = None,
        task_weights: Optional[dict] = None
    ):
        super().__init__()
        self.save_hyperparameters()

        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.n_tasks = n_tasks

        # Task names
        if task_names is None:
            if n_tasks == 1:
                self.task_names = ['property']
            else:
                self.task_names = [f'task_{i}' for i in range(n_tasks)]
        else:
            if len(task_names) != n_tasks:
                raise ValueError(f"Number of task names ({len(task_names)}) must match n_tasks ({n_tasks})")
            self.task_names = task_names

        # Task weights for loss (default: equal weights)
        if task_weights is None:
            self.task_weights = {name: 1.0 for name in self.task_names}
        else:
            self.task_weights = task_weights

        # Auto-generate hidden dims if not provided
        if hidden_dims is None:
            hidden_dims = []
            current_dim = input_dim

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
                hidden_dims = [max(input_dim // 2, 64)]

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
            prev_dim = input_dim

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

            # Shared dimension (last hidden dim or input_dim/4)
            shared_dim = hidden_dims[-1] if hidden_dims else max(input_dim // 4, 64)

            # Head hidden dimension (smaller than shared) - only used if head_hidden_dims not provided
            head_hidden_dim = max(shared_dim // 2, 32) if head_hidden_dims is None else None

            self.multi_task_network = MultiTaskNetwork(
                input_dim=input_dim,
                task_names=self.task_names,
                backbone_hidden_dims=hidden_dims,
                shared_dim=shared_dim,
                head_hidden_dim=head_hidden_dim,
                head_hidden_dims=head_hidden_dims,
                dropout=dropout,
                activation=activation
            )
            self.network = None  # Not used in multi-task mode

    def forward(self, x):
        """
        Forward pass: embedding → property value(s)

        Args:
            x: Input embeddings [batch_size, input_dim]

        Returns:
            Single-task: predictions [batch_size]
            Multi-task: dict {task_name: predictions [batch_size]}
        """
        if self.n_tasks == 1:
            return self.network(x).squeeze(-1)
        else:
            return self.multi_task_network(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        if self.n_tasks == 1:
            # Single-task training
            y_pred = self(x)
            loss = nn.functional.mse_loss(y_pred, y)

            # Log metrics
            self.log('train_loss', loss, prog_bar=True)
            self.log('train_mae', nn.functional.l1_loss(y_pred, y))

        else:
            # Multi-task training (handles sparse labels with NaN)
            y_pred_dict = self(x)

            # Compute weighted loss across tasks (only for non-NaN labels)
            total_loss = 0.0
            total_weight = 0.0
            n_tasks_with_data = 0

            for i, task_name in enumerate(self.task_names):
                y_task = y[:, i]
                y_pred_task = y_pred_dict[task_name]

                # Filter out NaN values
                mask = ~torch.isnan(y_task)
                if mask.sum() > 0:
                    y_task_valid = y_task[mask]
                    y_pred_task_valid = y_pred_task[mask]

                    task_loss = nn.functional.mse_loss(y_pred_task_valid, y_task_valid)
                    task_mae = nn.functional.l1_loss(y_pred_task_valid, y_task_valid)

                    # Weight and accumulate
                    weight = self.task_weights[task_name]
                    total_loss += weight * task_loss
                    total_weight += weight
                    n_tasks_with_data += 1

                    # Log per-task metrics
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
        x, y = batch

        if self.n_tasks == 1:
            # Single-task validation
            y_pred = self(x)
            loss = nn.functional.mse_loss(y_pred, y)

            self.log('val_loss', loss, prog_bar=True)
            self.log('val_mae', nn.functional.l1_loss(y_pred, y), prog_bar=True)

        else:
            # Multi-task validation (handles sparse labels with NaN)
            y_pred_dict = self(x)

            total_loss = 0.0
            total_weight = 0.0

            for i, task_name in enumerate(self.task_names):
                y_task = y[:, i]
                y_pred_task = y_pred_dict[task_name]

                # Filter out NaN values
                mask = ~torch.isnan(y_task)
                if mask.sum() > 0:
                    y_task_valid = y_task[mask]
                    y_pred_task_valid = y_pred_task[mask]

                    task_loss = nn.functional.mse_loss(y_pred_task_valid, y_task_valid)
                    task_mae = nn.functional.l1_loss(y_pred_task_valid, y_task_valid)

                    weight = self.task_weights[task_name]
                    total_loss += weight * task_loss
                    total_weight += weight

                    self.log(f'val_loss_{task_name}', task_loss)
                    self.log(f'val_mae_{task_name}', task_mae, prog_bar=True)

            # Average loss across tasks that have data
            if total_weight > 0:
                loss = total_loss / total_weight
            else:
                loss = torch.tensor(0.0, device=total_loss.device)

            self.log('val_loss', loss, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch

        if self.n_tasks == 1:
            # Single-task testing
            y_pred = self(x)
            loss = nn.functional.mse_loss(y_pred, y)

            self.log('test_loss', loss)
            self.log('test_mae', nn.functional.l1_loss(y_pred, y))
            self.log('test_rmse', torch.sqrt(loss))

        else:
            # Multi-task testing
            y_pred_dict = self(x)

            total_loss = 0.0
            for i, task_name in enumerate(self.task_names):
                y_task = y[:, i]
                y_pred_task = y_pred_dict[task_name]

                task_loss = nn.functional.mse_loss(y_pred_task, y_task)
                task_mae = nn.functional.l1_loss(y_pred_task, y_task)
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


class PropertyPredictor:
    """
    High-level wrapper for property prediction.

    Supports both single-task and multi-task learning.

    Single-task usage:
        >>> from src.embedding import RNAFMEmbedder
        >>> from src.models.property_predictor import PropertyPredictor
        >>>
        >>> # Create embedder
        >>> embedder = FingerprintEmbedder(fp_type='morgan', radius=2, n_bits=512)
        >>>
        >>> # Create predictor
        >>> predictor = PropertyPredictor(embedder=embedder)
        >>>
        >>> # Train
        >>> smiles_train = ['CCO', 'c1ccccc1', ...]
        >>> y_train = [3.5, 7.2, ...]
        >>> predictor.fit(smiles_train, y_train)
        >>>
        >>> # Predict
        >>> y_pred = predictor.predict(['CCO'])

    Multi-task usage:
        >>> predictor = PropertyPredictor(
        ...     embedder=embedder,
        ...     task_names=['logP', 'QED', 'SAS']
        ... )
        >>> y_train = np.array([[3.5, 0.7, 2.1], [7.2, 0.4, 3.8], ...])  # shape: (n, 3)
        >>> predictor.fit(smiles_train, y_train)
        >>> y_pred = predictor.predict(['CCO'])  # Returns dict: {'logP': ..., 'QED': ..., 'SAS': ...}
    """

    def __init__(
        self,
        embedder,
        hidden_dims: Optional[List[int]] = None,
        head_hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        max_epochs: int = 100,
        device: Optional[str] = None,
        task_names: Optional[List[str]] = None,
        task_weights: Optional[dict] = None
    ):
        """
        Initialize property predictor.

        Args:
            embedder: MoleculeEmbedder instance (FingerprintEmbedder, ChemBERTaEmbedder, etc.)
            hidden_dims: Hidden layer dimensions (None for auto)
            head_hidden_dims: Hidden dims for task heads (e.g., [256, 256, 256, 128])
            dropout: Dropout probability
            learning_rate: Learning rate
            batch_size: Batch size
            max_epochs: Maximum training epochs
            device: 'cuda', 'cpu', or None (auto-detect)
            task_names: List of task names for multi-task learning (None for single-task)
            task_weights: Dict of task weights for multi-task loss weighting
        """
        self.embedder = embedder
        self.hidden_dims = hidden_dims
        self.head_hidden_dims = head_hidden_dims
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
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
        smiles_train: Optional[List[str]] = None,
        y_train: Union[List[float], np.ndarray] = None,
        smiles_val: Optional[List[str]] = None,
        y_val: Optional[Union[List[float], np.ndarray]] = None,
        verbose: bool = True,
        # Pre-computed embeddings (alternative to SMILES)
        mol_emb_train: Optional[np.ndarray] = None,
        mol_emb_val: Optional[np.ndarray] = None
    ):
        """
        Train the model.

        Args:
            smiles_train: Training SMILES - required if mol_emb_train not provided
            y_train: Training property values
                    Single-task: shape (n_samples,) or (n_samples, 1)
                    Multi-task: shape (n_samples, n_tasks)
            smiles_val: Optional validation SMILES
            y_val: Optional validation property values
            verbose: Show training progress
            mol_emb_train: Pre-computed embeddings for training (alternative to computing from SMILES)
            mol_emb_val: Pre-computed embeddings for validation (alternative to computing from SMILES)
        """
        # Get molecule embeddings (either pre-computed or compute from SMILES)
        if mol_emb_train is not None:
            print(f"Using pre-computed embeddings for {len(mol_emb_train)} training molecules")
            X_train = mol_emb_train
        else:
            if smiles_train is None:
                raise ValueError("Must provide either smiles_train or mol_emb_train")
            print(f"Embedding {len(smiles_train)} training molecules...")
            X_train = self.embedder.encode(smiles_train)
        y_train = np.array(y_train, dtype=np.float32)

        # Handle multi-task labels
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1) if self.n_tasks == 1 else y_train.reshape(-1, 1)

        # Verify shape matches n_tasks
        if y_train.shape[1] != self.n_tasks:
            raise ValueError(f"y_train has {y_train.shape[1]} columns but expected {self.n_tasks} tasks")

        # Squeeze for single-task (backward compatibility)
        if self.n_tasks == 1:
            y_train = y_train.squeeze()

        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        # Validation data
        val_loader = None
        has_val = (smiles_val is not None or mol_emb_val is not None) and y_val is not None

        if has_val:
            # Get validation embeddings (either pre-computed or compute from SMILES)
            if mol_emb_val is not None:
                print(f"Using pre-computed embeddings for {len(mol_emb_val)} validation molecules")
                X_val = mol_emb_val
            else:
                if smiles_val is None:
                    raise ValueError("Must provide either smiles_val or mol_emb_val")
                print(f"Embedding {len(smiles_val)} validation molecules...")
                X_val = self.embedder.encode(smiles_val)
            y_val = np.array(y_val, dtype=np.float32)

            # Handle multi-task labels for validation
            if y_val.ndim == 1:
                y_val = y_val.reshape(-1, 1) if self.n_tasks == 1 else y_val.reshape(-1, 1)

            if y_val.shape[1] != self.n_tasks:
                raise ValueError(f"y_val has {y_val.shape[1]} columns but expected {self.n_tasks} tasks")

            if self.n_tasks == 1:
                y_val = y_val.squeeze()

            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.FloatTensor(y_val)

            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                num_workers=4,
                pin_memory=True
            )

        # Initialize model
        input_dim = X_train.shape[1]
        self.model = PropertyPredictorMLP(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            head_hidden_dims=self.head_hidden_dims,
            dropout=self.dropout,
            learning_rate=self.learning_rate,
            n_tasks=self.n_tasks,
            task_names=self.task_names,
            task_weights=self.task_weights
        )

        # Determine embedder type for display
        embedder_type = "Unknown"
        if "morgan" in self.embedder.name.lower():
            embedder_type = "Morgan Fingerprint"
        elif "chemberta" in self.embedder.name.lower():
            embedder_type = "ChemBERTa Transformer"
        elif "chemprop" in self.embedder.name.lower():
            embedder_type = "ChemProp D-MPNN"
        elif "rdkit" in self.embedder.name.lower():
            embedder_type = "RDKit Fingerprint"
        elif "maccs" in self.embedder.name.lower():
            embedder_type = "MACCS Keys"

        print(f"\nModel architecture:")
        print(f"  {'='*70}")
        print(f"  EMBEDDER: {embedder_type}")
        print(f"  Molecule embedding size: {input_dim}")
        print(f"  {'='*70}")
        print(f"  Molecule embedder: {self.embedder.name}")
        print(f"  Predictor input: {input_dim}")
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
        self.log_dir = tempfile.mkdtemp(prefix='property_pred_')
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
        smiles: Union[str, List[str]],
        mol_emb: Optional[np.ndarray] = None
    ) -> Union[np.ndarray, dict]:
        """
        Predict property values.

        Args:
            smiles: Single SMILES or list of SMILES
            mol_emb: Pre-computed embeddings (optional)

        Returns:
            Single-task: numpy array of predictions
            Multi-task: dict {task_name: predictions}
        """
        if self.model is None:
            raise RuntimeError("Model not trained yet!")

        is_single = isinstance(smiles, str)
        if is_single:
            smiles = [smiles]

        # Get embeddings (pre-computed or compute from SMILES)
        if mol_emb is not None:
            X = mol_emb
        else:
            X = self.embedder.encode(smiles)
        X_tensor = torch.FloatTensor(X).to(self.device)

        # Predict
        self.model.eval()
        self.model.to(self.device)

        with torch.no_grad():
            if self.n_tasks == 1:
                # Single-task: return numpy array
                y_pred = self.model(X_tensor).cpu().numpy()
                if is_single:
                    return y_pred[0]
                return y_pred
            else:
                # Multi-task: return dict
                y_pred_dict = self.model(X_tensor)
                result = {}
                for task_name in self.task_names:
                    preds = y_pred_dict[task_name].cpu().numpy()
                    if is_single:
                        result[task_name] = preds[0]
                    else:
                        result[task_name] = preds
                return result

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
                'input_dim': self.model.input_dim,
                'hidden_dims': self.hidden_dims,
                'dropout': self.dropout,
                'learning_rate': self.learning_rate,
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
        embedder,
        device: Optional[str] = None
    ):
        """
        Load model checkpoint from disk.

        Args:
            path: Path to checkpoint file
            embedder: MoleculeEmbedder instance (must match training)
            device: Device to load model on ('cuda', 'cpu', or None for auto-detect)

        Returns:
            Loaded PropertyPredictor instance

        """
        checkpoint = torch.load(path, map_location='cpu')
        hparams = checkpoint['hyperparameters']

        # Create predictor instance
        predictor = cls(
            embedder=embedder,
            hidden_dims=hparams['hidden_dims'],
            dropout=hparams['dropout'],
            learning_rate=hparams['learning_rate'],
            batch_size=32,  # Not saved, use default
            max_epochs=100,  # Not saved, use default
            device=device,
            task_names=hparams['task_names'],
            task_weights=hparams['task_weights']
        )

        # Recreate PyTorch Lightning model
        predictor.model = PropertyPredictorMLP(
            input_dim=hparams['input_dim'],
            hidden_dims=hparams['hidden_dims'],
            dropout=hparams['dropout'],
            learning_rate=hparams['learning_rate'],
            n_tasks=hparams['n_tasks'],
            task_names=hparams['task_names'],
            task_weights=hparams['task_weights']
        )

        # Load state dict
        predictor.model.load_state_dict(checkpoint['model_state_dict'])
        predictor.model.to(predictor.device)
        predictor.model.eval()

        return predictor
