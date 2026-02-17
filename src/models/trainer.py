"""
Training utilities for edit prediction models.
"""

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from typing import Optional, Dict, List
from tqdm import tqdm
import numpy as np


class Trainer:
    """
    Trainer for edit prediction models.

    Handles training loop, validation, early stopping, and learning rate scheduling.

    Args:
        model: PyTorch model
        device: Device to train on
        learning_rate: Learning rate (default: 1e-3)
        weight_decay: L2 regularization (default: 1e-5)
        patience: Early stopping patience (default: 10)
        lr_scheduler: LR scheduler type ('plateau', 'cosine', None)

    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        patience: int = 10,
        lr_scheduler: Optional[str] = 'plateau'
    ):
        self.model = model.to(device)
        self.device = device

        # Loss and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Learning rate scheduler (FIXED: removed 'verbose')
        self.lr_scheduler = None
        if lr_scheduler == 'plateau':
            self.lr_scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        elif lr_scheduler == 'cosine':
            self.lr_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=50,
                eta_min=1e-7
            )

        # Early stopping
        self.patience = patience
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_model_state = None

    def train_epoch(self, train_loader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device).view(-1, 1)

            # Forward pass
            self.optimizer.zero_grad()
            y_pred = self.model(X_batch)
            loss = self.criterion(y_pred, y_batch)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    def validate(self, val_loader) -> Dict[str, float]:
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        predictions = []
        targets = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device).view(-1, 1)

                y_pred = self.model(X_batch)
                loss = self.criterion(y_pred, y_batch)

                total_loss += loss.item()
                n_batches += 1

                predictions.extend(y_pred.cpu().numpy().flatten())
                targets.extend(y_batch.cpu().numpy().flatten())

        val_loss = total_loss / n_batches

        # Calculate metrics
        predictions = np.array(predictions)
        targets = np.array(targets)

        mae = np.mean(np.abs(predictions - targets))
        rmse = np.sqrt(np.mean((predictions - targets) ** 2))

        # R² score
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        return {
            'val_loss': val_loss,
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }

    def fit(
        self,
        train_loader,
        val_loader,
        epochs: int = 100,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the model.

        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            epochs: Number of epochs
            verbose: Whether to print progress

        Returns:
            Training history dict with losses and metrics
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'mae': [],
            'rmse': [],
            'r2': [],
            'lr': []
        }

        if verbose:
            pbar = tqdm(range(epochs), desc="Training")
        else:
            pbar = range(epochs)

        for epoch in pbar:
            # Train
            train_loss = self.train_epoch(train_loader)

            # Validate
            val_metrics = self.validate(val_loader)

            # Record history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_metrics['val_loss'])
            history['mae'].append(val_metrics['mae'])
            history['rmse'].append(val_metrics['rmse'])
            history['r2'].append(val_metrics['r2'])
            history['lr'].append(self.optimizer.param_groups[0]['lr'])

            # Learning rate scheduling
            if self.lr_scheduler is not None:
                if isinstance(self.lr_scheduler, ReduceLROnPlateau):
                    self.lr_scheduler.step(val_metrics['val_loss'])
                else:
                    self.lr_scheduler.step()

            # Early stopping
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.patience_counter = 0
                self.best_model_state = self.model.state_dict().copy()
            else:
                self.patience_counter += 1

            # Update progress bar
            if verbose:
                pbar.set_postfix({
                    'train_loss': f"{train_loss:.4f}",
                    'val_loss': f"{val_metrics['val_loss']:.4f}",
                    'mae': f"{val_metrics['mae']:.4f}",
                    'r2': f"{val_metrics['r2']:.3f}",
                    'patience': f"{self.patience_counter}/{self.patience}"
                })

            # Check early stopping
            if self.patience_counter >= self.patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                break

        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            if verbose:
                print(f"Restored best model (val_loss={self.best_val_loss:.4f})")

        return history

    def predict(self, data_loader) -> np.ndarray:
        """
        Make predictions on a dataset.

        Args:
            data_loader: DataLoader

        Returns:
            Predictions as numpy array
        """
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, tuple):
                    X_batch = batch[0]
                else:
                    X_batch = batch

                X_batch = X_batch.to(self.device)
                y_pred = self.model(X_batch)
                predictions.extend(y_pred.cpu().numpy().flatten())

        return np.array(predictions)

    def save_model(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss
        }, path)

    def load_model(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
