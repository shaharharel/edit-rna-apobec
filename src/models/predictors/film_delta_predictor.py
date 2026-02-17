"""
FiLM-conditioned delta predictor for RNA/molecule edit effect prediction.

FiLM (Feature-wise Linear Modulation) conditions the prediction network on the
edit representation, allowing the network to learn edit-specific transformations.

Architecture:
    - Sequence embeddings: emb_a, emb_b from RNA/molecule embedder
    - Delta/Edit: emb_b - emb_a (or learned transformation)
    - FiLM conditioning: γ, β = MLP(delta), then h' = γ * h + β
    - Prediction: f(emb_b | delta) - f(emb_a | delta)

Variants:
    - FiLMDelta: Standard FiLM with learnable γ, β from delta
    - SpectralFiLM: FiLM with spectral normalization for stability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Union, Dict, Tuple
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats as scipy_stats


# Enable Tensor Cores for GPU
torch.set_float32_matmul_precision('high')


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation layer.

    Applies: h' = γ * h + β
    where γ, β are generated from a conditioning input.

    Args:
        hidden_dim: Dimension of features to modulate
        cond_dim: Dimension of conditioning input (delta)
        spectral: If True, apply spectral normalization
    """

    def __init__(
        self,
        hidden_dim: int,
        cond_dim: int,
        spectral: bool = False
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cond_dim = cond_dim
        self.spectral = spectral

        # Generate γ and β from conditioning input
        gamma_layer = nn.Linear(cond_dim, hidden_dim)
        beta_layer = nn.Linear(cond_dim, hidden_dim)

        # Initialize before spectral norm wrapping
        # Use small weights for both, biases set for identity (gamma=1, beta=0)
        nn.init.xavier_uniform_(gamma_layer.weight, gain=0.1)
        nn.init.constant_(gamma_layer.bias, 1.0)  # Start with gamma ≈ 1
        nn.init.xavier_uniform_(beta_layer.weight, gain=0.1)  # Small but non-zero
        nn.init.zeros_(beta_layer.bias)  # Start with beta = 0

        if spectral:
            self.gamma_proj = nn.utils.spectral_norm(gamma_layer)
            self.beta_proj = nn.utils.spectral_norm(beta_layer)
        else:
            self.gamma_proj = gamma_layer
            self.beta_proj = beta_layer

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Apply FiLM modulation.

        Args:
            h: Features to modulate [batch, hidden_dim]
            cond: Conditioning input [batch, cond_dim]

        Returns:
            Modulated features [batch, hidden_dim]
        """
        gamma = self.gamma_proj(cond)  # [batch, hidden_dim]
        beta = self.beta_proj(cond)    # [batch, hidden_dim]

        # FiLM: γ * h + β
        return gamma * h + beta


class FiLMBlock(nn.Module):
    """
    FiLM-conditioned MLP block.

    Architecture options:
    - Default: Linear → ReLU → FiLM → Dropout
    - With BatchNorm (V5 style): Linear → BatchNorm → FiLM → ReLU → Dropout

    Args:
        input_dim: Input dimension
        hidden_dim: Output dimension
        cond_dim: Conditioning dimension (delta)
        dropout: Dropout probability
        spectral: Apply spectral normalization
        use_batchnorm: If True, use BatchNorm before FiLM (like V5 architecture)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        cond_dim: int,
        dropout: float = 0.2,
        spectral: bool = False,
        use_batchnorm: bool = False
    ):
        super().__init__()
        self.use_batchnorm = use_batchnorm

        linear = nn.Linear(input_dim, hidden_dim)
        if spectral:
            self.linear = nn.utils.spectral_norm(linear)
        else:
            self.linear = linear

        if use_batchnorm:
            self.batchnorm = nn.BatchNorm1d(hidden_dim)

        self.activation = nn.ReLU()
        self.film = FiLMLayer(hidden_dim, cond_dim, spectral=spectral)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Forward with FiLM conditioning.

        Args:
            x: Input features [batch, input_dim]
            cond: Conditioning (delta) [batch, cond_dim]

        Returns:
            Output features [batch, hidden_dim]
        """
        h = self.linear(x)
        if self.use_batchnorm:
            # V5 style: Linear → BatchNorm → FiLM → ReLU → Dropout
            h = self.batchnorm(h)
            h = self.film(h, cond)
            h = self.activation(h)
        else:
            # Default: Linear → ReLU → FiLM → Dropout
            h = self.activation(h)
            h = self.film(h, cond)
        h = self.dropout(h)
        return h


class FiLMDeltaMLP(nn.Module):
    """
    FiLM-conditioned MLP for delta prediction.

    The network predicts f(B|delta) and f(A|delta), where the network
    is conditioned on delta = emb_b - emb_a via FiLM layers.

    This allows the network to learn different transformations depending
    on the type/magnitude of edit, potentially improving generalization.

    Args:
        input_dim: Embedding dimension (same for A and B)
        hidden_dims: List of hidden layer dimensions
        dropout: Dropout probability
        spectral: If True, use spectral normalization (SpectralFiLM)
        modulation_strength: Scale factor for FiLM modulation (0.0-1.0)
        use_batchnorm: If True, use BatchNorm in FiLM blocks (V5 style architecture)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
        spectral: bool = False,
        modulation_strength: float = 1.0,
        use_batchnorm: bool = False
    ):
        super().__init__()

        self.input_dim = input_dim
        self.spectral = spectral
        self.modulation_strength = modulation_strength
        self.use_batchnorm = use_batchnorm

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

        # Conditioning dimension (delta is same as input)
        cond_dim = input_dim

        # Delta encoder: transform delta before using for conditioning
        delta_hidden = max(input_dim // 2, 64)
        delta_linear = nn.Linear(input_dim, delta_hidden)
        if spectral:
            delta_linear = nn.utils.spectral_norm(delta_linear)

        self.delta_encoder = nn.Sequential(
            delta_linear,
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # FiLM blocks
        self.blocks = nn.ModuleList()
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            block = FiLMBlock(
                input_dim=prev_dim,
                hidden_dim=hidden_dim,
                cond_dim=delta_hidden,
                dropout=dropout,
                spectral=spectral,
                use_batchnorm=use_batchnorm
            )
            self.blocks.append(block)
            prev_dim = hidden_dim

        # Output layer
        output_linear = nn.Linear(prev_dim, 1)
        if spectral:
            self.output = nn.utils.spectral_norm(output_linear)
        else:
            self.output = output_linear

    def forward_single(
        self,
        x: torch.Tensor,
        delta_cond: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for a single sequence embedding.

        Args:
            x: Sequence embedding [batch, input_dim]
            delta_cond: Encoded delta for conditioning [batch, cond_dim]

        Returns:
            Prediction [batch]
        """
        h = x
        for block in self.blocks:
            h = block(h, delta_cond)

        return self.output(h).squeeze(-1)

    def forward(
        self,
        emb_a: torch.Tensor,
        emb_b: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass: predict delta from embeddings.

        Computes f(B|delta) - f(A|delta) where delta = B - A.

        Args:
            emb_a: Wild-type/reference embeddings [batch, input_dim]
            emb_b: Mutant/variant embeddings [batch, input_dim]

        Returns:
            Predicted delta [batch]
        """
        # Compute delta (edit representation)
        delta = emb_b - emb_a

        # Encode delta for conditioning
        delta_cond = self.delta_encoder(delta)

        # Apply modulation strength
        if self.modulation_strength < 1.0:
            # Blend with zero conditioning (residual connection style)
            zero_cond = torch.zeros_like(delta_cond)
            delta_cond = (
                self.modulation_strength * delta_cond +
                (1 - self.modulation_strength) * zero_cond
            )

        # Predict for both sequences, conditioned on delta
        pred_a = self.forward_single(emb_a, delta_cond)
        pred_b = self.forward_single(emb_b, delta_cond)

        # Delta prediction
        return pred_b - pred_a


class FiLMDeltaPredictor:
    """
    High-level FiLM-conditioned delta predictor.

    Uses FiLM (Feature-wise Linear Modulation) to condition the prediction
    network on the edit representation, allowing edit-specific transformations.

    Two variants:
        - FiLMDelta (spectral=False): Standard FiLM conditioning
        - SpectralFiLM (spectral=True): FiLM with spectral normalization

    The spectral variant applies spectral normalization to all linear layers,
    which can improve training stability and generalization.

    Example:
        >>> from src.embedding import RNAFMEmbedder
        >>> from src.models.predictors import FiLMDeltaPredictor
        >>>
        >>> # Get embeddings
        >>> embedder = RNAFMEmbedder()
        >>> emb_a = embedder.encode(["AUGCAUGC", "GCUAGCUA"])
        >>> emb_b = embedder.encode(["AUGGAUGC", "GCUGGCUA"])
        >>>
        >>> # Create predictor (standard FiLM)
        >>> predictor = FiLMDeltaPredictor(spectral=False)
        >>> predictor.fit(emb_a_train, emb_b_train, delta_train)
        >>>
        >>> # Or SpectralFiLM variant
        >>> predictor = FiLMDeltaPredictor(spectral=True, dropout=0.3)

    Args:
        hidden_dims: Hidden layer dimensions (None for auto)
        dropout: Dropout probability
        spectral: If True, use spectral normalization (SpectralFiLM)
        modulation_strength: FiLM modulation strength (0.0-1.0)
        use_batchnorm: If True, use BatchNorm in FiLM blocks (V5 architecture)
        learning_rate: Learning rate
        batch_size: Batch size
        max_epochs: Maximum training epochs
        patience: Early stopping patience
        device: 'cuda', 'mps', 'cpu', or None (auto-detect)
    """

    def __init__(
        self,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
        spectral: bool = False,
        modulation_strength: float = 1.0,
        use_batchnorm: bool = False,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        max_epochs: int = 100,
        patience: int = 15,
        device: Optional[str] = None
    ):
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.spectral = spectral
        self.modulation_strength = modulation_strength
        self.use_batchnorm = use_batchnorm
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
        self.input_dim = None

    def fit(
        self,
        emb_a_train: Union[np.ndarray, torch.Tensor],
        emb_b_train: Union[np.ndarray, torch.Tensor],
        delta_train: Union[np.ndarray, torch.Tensor],
        emb_a_val: Optional[Union[np.ndarray, torch.Tensor]] = None,
        emb_b_val: Optional[Union[np.ndarray, torch.Tensor]] = None,
        delta_val: Optional[Union[np.ndarray, torch.Tensor]] = None,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the FiLM-conditioned predictor.

        Args:
            emb_a_train: Reference/WT embeddings [N, D]
            emb_b_train: Variant/mutant embeddings [N, D]
            delta_train: Delta values [N]
            emb_a_val: Optional validation reference embeddings
            emb_b_val: Optional validation variant embeddings
            delta_val: Optional validation delta values
            verbose: Show training progress

        Returns:
            Training history dict with 'train_loss' and 'val_loss' lists
        """
        # Convert to tensors
        emb_a_train = self._to_tensor(emb_a_train)
        emb_b_train = self._to_tensor(emb_b_train)
        delta_train = self._to_tensor(delta_train)

        self.input_dim = emb_a_train.shape[1]

        # Create model
        self.model = FiLMDeltaMLP(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
            spectral=self.spectral,
            modulation_strength=self.modulation_strength,
            use_batchnorm=self.use_batchnorm
        )
        self.model = self.model.to(self.device)

        variant_name = "SpectralFiLM" if self.spectral else "FiLMDelta"
        if self.use_batchnorm:
            variant_name += " (V5/BatchNorm)"
        if verbose:
            print(f"\n{variant_name} Predictor:")
            print(f"  Input dim: {self.input_dim}")
            print(f"  Hidden dims: {self.model.hidden_dims}")
            print(f"  Spectral norm: {self.spectral}")
            print(f"  BatchNorm: {self.use_batchnorm}")
            print(f"  Modulation strength: {self.modulation_strength}")
            print(f"  Training samples: {len(delta_train)}")
            print(f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        # Create dataloaders
        train_dataset = TensorDataset(emb_a_train, emb_b_train, delta_train)
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        val_loader = None
        if emb_a_val is not None and emb_b_val is not None and delta_val is not None:
            emb_a_val = self._to_tensor(emb_a_val)
            emb_b_val = self._to_tensor(emb_b_val)
            delta_val = self._to_tensor(delta_val)

            val_dataset = TensorDataset(emb_a_val, emb_b_val, delta_val)
            val_loader = DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=False
            )

        # Training loop
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        criterion = nn.MSELoss()

        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0

        for epoch in range(self.max_epochs):
            # Training
            self.model.train()
            train_losses = []

            for batch_a, batch_b, batch_y in train_loader:
                batch_a = batch_a.to(self.device)
                batch_b = batch_b.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                pred = self.model(batch_a, batch_b)
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
                    for batch_a, batch_b, batch_y in val_loader:
                        batch_a = batch_a.to(self.device)
                        batch_b = batch_b.to(self.device)
                        batch_y = batch_y.to(self.device)

                        pred = self.model(batch_a, batch_b)
                        loss = criterion(pred, batch_y)
                        val_losses.append(loss.item())

                val_loss = np.mean(val_losses)
                history['val_loss'].append(val_loss)

                # Learning rate scheduling
                scheduler.step(val_loss)

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
        emb_a: Union[np.ndarray, torch.Tensor],
        emb_b: Union[np.ndarray, torch.Tensor]
    ) -> np.ndarray:
        """
        Predict delta values.

        Args:
            emb_a: Reference/WT embeddings [N, D]
            emb_b: Variant/mutant embeddings [N, D]

        Returns:
            Predicted delta values [N]
        """
        if self.model is None:
            raise RuntimeError("Model not trained yet!")

        emb_a = self._to_tensor(emb_a).to(self.device)
        emb_b = self._to_tensor(emb_b).to(self.device)

        self.model.eval()
        with torch.no_grad():
            pred = self.model(emb_a, emb_b).cpu().numpy()

        return pred

    def evaluate(
        self,
        emb_a: Union[np.ndarray, torch.Tensor],
        emb_b: Union[np.ndarray, torch.Tensor],
        delta_true: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
        """
        Evaluate model and return metrics.

        Args:
            emb_a: Reference embeddings [N, D]
            emb_b: Variant embeddings [N, D]
            delta_true: True delta values [N]

        Returns:
            Tuple of (metrics_dict, y_true, y_pred)
        """
        delta_true = self._to_numpy(delta_true)
        delta_pred = self.predict(emb_a, emb_b)

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
            'r2': float(r2)
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
                'spectral': self.spectral,
                'modulation_strength': self.modulation_strength,
                'learning_rate': self.learning_rate
            }
        }
        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(
        cls,
        path: str,
        device: Optional[str] = None
    ) -> 'FiLMDeltaPredictor':
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location='cpu')
        hparams = checkpoint['hyperparameters']

        predictor = cls(
            hidden_dims=hparams['hidden_dims'],
            dropout=hparams['dropout'],
            spectral=hparams['spectral'],
            modulation_strength=hparams.get('modulation_strength', 1.0),
            learning_rate=hparams['learning_rate'],
            device=device
        )

        predictor.input_dim = hparams['input_dim']
        predictor.model = FiLMDeltaMLP(
            input_dim=hparams['input_dim'],
            hidden_dims=hparams['hidden_dims'],
            dropout=hparams['dropout'],
            spectral=hparams['spectral'],
            modulation_strength=hparams.get('modulation_strength', 1.0)
        )
        predictor.model.load_state_dict(checkpoint['model_state_dict'])
        predictor.model.to(predictor.device)
        predictor.model.eval()

        return predictor

    @property
    def name(self) -> str:
        """Return model name."""
        base = "SpectralFiLM" if self.spectral else "FiLMDelta"
        if self.modulation_strength < 1.0:
            return f"{base}_mod{self.modulation_strength}"
        if self.spectral and self.dropout != 0.2:
            return f"{base}_dp{self.dropout}"
        return base
