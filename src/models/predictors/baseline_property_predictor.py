"""
Baseline Property Predictor for delta prediction via subtraction.

This is the non-edit-aware baseline that:
1. Trains a property predictor on absolute values: f(molecule) → property
2. Predicts delta as: delta = f(mutant) - f(wild-type)

This approach works with any pre-computed embeddings for RNA sequences.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Optional, List, Union, Dict, Tuple
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats as scipy_stats

# Enable Tensor Cores for L4 GPU
torch.set_float32_matmul_precision('high')


class BaselinePropertyMLP(pl.LightningModule):
    """
    Multi-layer perceptron for property prediction: f(embedding) → Y

    This is a simpler version of PropertyPredictorMLP that works directly
    with pre-computed embeddings, without molecule-specific logic.

    Args:
        input_dim: Input embedding dimension
        hidden_dims: Optional list of hidden dimensions. If None, auto-generates
        dropout: Dropout probability
        learning_rate: Learning rate for optimizer
        activation: Activation function ('relu', 'elu', 'gelu')
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        activation: str = 'relu',
    ):
        super().__init__()
        self.save_hyperparameters()

        self.input_dim = input_dim
        self.learning_rate = learning_rate

        # Auto-generate hidden dims if not provided
        if hidden_dims is None:
            hidden_dims = []
            current_dim = input_dim
            min_hidden_dim = 64
            max_layers = 3

            for _ in range(max_layers):
                current_dim = current_dim // 2
                if current_dim < min_hidden_dim:
                    break
                hidden_dims.append(current_dim)

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

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: embedding → property value."""
        return self.network(x).squeeze(-1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = nn.functional.mse_loss(y_pred, y)

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_mae', nn.functional.l1_loss(y_pred, y))

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = nn.functional.mse_loss(y_pred, y)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_mae', nn.functional.l1_loss(y_pred, y), prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = nn.functional.mse_loss(y_pred, y)

        self.log('test_loss', loss)
        self.log('test_mae', nn.functional.l1_loss(y_pred, y))
        self.log('test_rmse', torch.sqrt(loss))

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val_loss'}
        }


class BaselinePropertyPredictor:
    """
    Baseline property predictor that learns f(molecule) and computes delta.

    This approach works with any pre-computed RNA embeddings:
    - RNA sequences → UTR-LM/RNA-FM/RNABERT → embedding

    Training:
        - Trains on absolute property values for both wild-type and mutant
        - Wild-type gets label 0 (reference), mutant gets label delta

    Inference:
        - delta_pred = f(mutant_embedding) - f(wt_embedding)

    Example:
        >>> # With pre-computed embeddings
        >>> predictor = BaselinePropertyPredictor(hidden_dims=[256, 128])
        >>> predictor.fit(
        ...     wt_emb_train=wt_embeddings,
        ...     mut_emb_train=mut_embeddings,
        ...     delta_train=delta_values,
        ...     wt_emb_val=wt_val_embeddings,
        ...     mut_emb_val=mut_val_embeddings,
        ...     delta_val=delta_val_values,
        ... )
        >>> delta_pred = predictor.predict(wt_emb_test, mut_emb_test)
    """

    def __init__(
        self,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        max_epochs: int = 100,
        patience: int = 10,
        device: Optional[str] = None,
    ):
        """
        Initialize baseline property predictor.

        Args:
            hidden_dims: Hidden layer dimensions (None for auto)
            dropout: Dropout probability
            learning_rate: Learning rate
            batch_size: Batch size
            max_epochs: Maximum training epochs
            patience: Early stopping patience
            device: 'cuda', 'mps', 'cpu', or None (auto-detect)
        """
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience

        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device

        self.model = None
        self.trainer = None
        self.input_dim = None

    def fit(
        self,
        wt_emb_train: Union[np.ndarray, torch.Tensor],
        mut_emb_train: Union[np.ndarray, torch.Tensor],
        delta_train: Union[np.ndarray, torch.Tensor],
        wt_emb_val: Optional[Union[np.ndarray, torch.Tensor]] = None,
        mut_emb_val: Optional[Union[np.ndarray, torch.Tensor]] = None,
        delta_val: Optional[Union[np.ndarray, torch.Tensor]] = None,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Train the baseline property predictor.

        The model learns f(embedding) → property_value, where:
        - Wild-type embeddings have label 0 (reference)
        - Mutant embeddings have label delta

        Args:
            wt_emb_train: Wild-type embeddings [N, D]
            mut_emb_train: Mutant embeddings [N, D]
            delta_train: Delta values [N]
            wt_emb_val: Optional validation wild-type embeddings
            mut_emb_val: Optional validation mutant embeddings
            delta_val: Optional validation delta values
            verbose: Show training progress

        Returns:
            Training history dict with 'train_loss' and 'val_loss' lists
        """
        # Convert to tensors
        wt_emb_train = self._to_tensor(wt_emb_train)
        mut_emb_train = self._to_tensor(mut_emb_train)
        delta_train = self._to_tensor(delta_train)

        # Combine wt and mut for training
        # wt has value 0 (reference), mut has value delta
        X_train = torch.cat([wt_emb_train, mut_emb_train], dim=0)
        y_train = torch.cat([
            torch.zeros(len(wt_emb_train)),
            delta_train,
        ], dim=0)

        self.input_dim = X_train.shape[1]

        # Create model
        self.model = BaselinePropertyMLP(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
            learning_rate=self.learning_rate,
        )
        self.model = self.model.to(self.device)

        print(f"\nBaseline Property Predictor:")
        print(f"  Input dim: {self.input_dim}")
        print(f"  Hidden dims: {self.model.hidden_dims}")
        print(f"  Training samples: {len(wt_emb_train)} pairs → {len(X_train)} total")
        print(f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        # Create dataloaders
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        val_loader = None
        if wt_emb_val is not None and mut_emb_val is not None and delta_val is not None:
            wt_emb_val = self._to_tensor(wt_emb_val)
            mut_emb_val = self._to_tensor(mut_emb_val)
            delta_val = self._to_tensor(delta_val)

            X_val = torch.cat([wt_emb_val, mut_emb_val], dim=0)
            y_val = torch.cat([
                torch.zeros(len(wt_emb_val)),
                delta_val,
            ], dim=0)

            val_dataset = TensorDataset(X_val, y_val)
            val_loader = DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=False
            )

        # Manual training loop (for flexibility and MPS support)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0

        for epoch in range(self.max_epochs):
            # Training
            self.model.train()
            train_losses = []
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                pred = self.model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            train_loss = np.mean(train_losses)
            history['train_loss'].append(train_loss)

            # Validation
            if val_loader is not None:
                self.model.eval()
                val_losses = []
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x = batch_x.to(self.device)
                        batch_y = batch_y.to(self.device)
                        pred = self.model(batch_x)
                        loss = criterion(pred, batch_y)
                        val_losses.append(loss.item())

                val_loss = np.mean(val_losses)
                history['val_loss'].append(val_loss)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.patience:
                    if verbose:
                        print(f"  Early stopping at epoch {epoch + 1}")
                    break
            else:
                history['val_loss'].append(train_loss)

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model = self.model.to(self.device)

        if verbose:
            final_train = history['train_loss'][-1]
            final_val = history['val_loss'][-1] if history['val_loss'] else None
            print(f"  Training complete: train_loss={final_train:.4f}" +
                  (f", val_loss={final_val:.4f}" if final_val else ""))

        return history

    def predict(
        self,
        wt_emb: Union[np.ndarray, torch.Tensor],
        mut_emb: Union[np.ndarray, torch.Tensor],
    ) -> np.ndarray:
        """
        Predict delta values.

        Computes: delta_pred = f(mut_emb) - f(wt_emb)

        Args:
            wt_emb: Wild-type embeddings [N, D]
            mut_emb: Mutant embeddings [N, D]

        Returns:
            Predicted delta values [N]
        """
        if self.model is None:
            raise RuntimeError("Model not trained yet!")

        wt_emb = self._to_tensor(wt_emb).to(self.device)
        mut_emb = self._to_tensor(mut_emb).to(self.device)

        self.model.eval()
        with torch.no_grad():
            pred_wt = self.model(wt_emb).cpu().numpy()
            pred_mut = self.model(mut_emb).cpu().numpy()

        return pred_mut - pred_wt

    def evaluate(
        self,
        wt_emb: Union[np.ndarray, torch.Tensor],
        mut_emb: Union[np.ndarray, torch.Tensor],
        delta_true: Union[np.ndarray, torch.Tensor],
    ) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
        """
        Evaluate model and return metrics.

        Args:
            wt_emb: Wild-type embeddings [N, D]
            mut_emb: Mutant embeddings [N, D]
            delta_true: True delta values [N]

        Returns:
            Tuple of (metrics_dict, y_true, y_pred)
        """
        delta_true = self._to_numpy(delta_true)
        delta_pred = self.predict(wt_emb, mut_emb)

        # Compute metrics
        mae = np.mean(np.abs(delta_pred - delta_true))
        mse = np.mean((delta_pred - delta_true) ** 2)
        rmse = np.sqrt(mse)

        if np.std(delta_pred) < 1e-6 or np.std(delta_true) < 1e-6:
            pearson = 0.0
            spearman = 0.0
        else:
            pearson, _ = scipy_stats.pearsonr(delta_pred, delta_true)
            spearman, _ = scipy_stats.spearmanr(delta_pred, delta_true)

        ss_res = np.sum((delta_true - delta_pred) ** 2)
        ss_tot = np.sum((delta_true - np.mean(delta_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        metrics = {
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'pearson': float(pearson),
            'spearman': float(spearman),
            'r2': float(r2),
        }

        return metrics, delta_true, delta_pred

    def _to_tensor(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Convert to float tensor."""
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).float()
        return x.float()

    def _to_numpy(self, x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Convert to numpy array."""
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy()
        return x

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'hyperparameters': {
                'input_dim': self.input_dim,
                'hidden_dims': self.hidden_dims,
                'dropout': self.dropout,
                'learning_rate': self.learning_rate,
            }
        }
        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(cls, path: str, device: Optional[str] = None) -> 'BaselinePropertyPredictor':
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location='cpu')
        hparams = checkpoint['hyperparameters']

        predictor = cls(
            hidden_dims=hparams['hidden_dims'],
            dropout=hparams['dropout'],
            learning_rate=hparams['learning_rate'],
            device=device,
        )

        predictor.input_dim = hparams['input_dim']
        predictor.model = BaselinePropertyMLP(
            input_dim=hparams['input_dim'],
            hidden_dims=hparams['hidden_dims'],
            dropout=hparams['dropout'],
            learning_rate=hparams['learning_rate'],
        )
        predictor.model.load_state_dict(checkpoint['model_state_dict'])
        predictor.model.to(predictor.device)
        predictor.model.eval()

        return predictor
