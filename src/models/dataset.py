"""
PyTorch datasets for edit prediction with pre-computed embeddings.

This module supports:
1. Pre-computed embeddings (recommended for experiments)
2. On-the-fly embedding (for prototyping)
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Union, Tuple


class EditDataset(Dataset):
    """
    Dataset for edit-based property prediction.

    Supports two modes:
    1. Pre-computed embeddings (recommended)
    2. On-the-fly embedding (slower, but flexible)

    Args:
        edit_embeddings: Pre-computed edit embeddings (n_samples, embed_dim)
                        If provided, embeddings are used directly (fast!)
        pairs_df: DataFrame with mol_a, mol_b columns (only if edit_embeddings=None)
        targets: Target values (property changes)
        edit_embedder: EditEmbedder instance (only if edit_embeddings=None)

    """

    def __init__(
        self,
        edit_embeddings: Optional[np.ndarray] = None,
        pairs_df: Optional[pd.DataFrame] = None,
        targets: Optional[np.ndarray] = None,
        edit_embedder=None
    ):
        if edit_embeddings is not None:
            # Mode 1: Pre-computed embeddings (FAST)
            self.mode = 'precomputed'
            self.edit_embeddings = torch.FloatTensor(edit_embeddings)
            self.targets = torch.FloatTensor(targets) if targets is not None else None

        elif pairs_df is not None and edit_embedder is not None:
            # Mode 2: On-the-fly embedding (SLOWER)
            self.mode = 'on_the_fly'
            self.pairs_df = pairs_df
            self.edit_embedder = edit_embedder
            self.targets = torch.FloatTensor(targets) if targets is not None else None

        else:
            raise ValueError(
                "Must provide either:\n"
                "  1. edit_embeddings (pre-computed), or\n"
                "  2. pairs_df + edit_embedder (on-the-fly)"
            )

    def __len__(self):
        if self.mode == 'precomputed':
            return len(self.edit_embeddings)
        else:
            return len(self.pairs_df)

    def __getitem__(self, idx):
        if self.mode == 'precomputed':
            # Use pre-computed embeddings
            X = self.edit_embeddings[idx]
        else:
            # Compute embedding on-the-fly
            row = self.pairs_df.iloc[idx]
            mol_a = row['mol_a']
            mol_b = row['mol_b']
            edit_vec = self.edit_embedder.encode_from_smiles(mol_a, mol_b)
            X = torch.FloatTensor(edit_vec)

        if self.targets is not None:
            y = self.targets[idx]
            return X, y
        else:
            return X


def create_dataloaders(
    train_dataset: EditDataset,
    val_dataset: Optional[EditDataset] = None,
    batch_size: int = 32,
    num_workers: int = 0,
    shuffle_train: bool = True
) -> Union[DataLoader, Tuple[DataLoader, DataLoader]]:
    """
    Create DataLoader(s) from dataset(s).

    Args:
        train_dataset: Training dataset
        val_dataset: Optional validation dataset
        batch_size: Batch size
        num_workers: Number of workers for data loading
        shuffle_train: Whether to shuffle training data

    Returns:
        train_loader (or tuple of train_loader, val_loader if val_dataset provided)

    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        return train_loader, val_loader
    else:
        return train_loader


# Convenience function for creating datasets from pre-computed embeddings
def create_datasets_from_embeddings(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None
) -> Union[EditDataset, Tuple[EditDataset, EditDataset]]:
    """
    Create dataset(s) from pre-computed embeddings.

    Args:
        X_train: Training edit embeddings (n_train, embed_dim)
        y_train: Training targets (n_train,)
        X_val: Optional validation embeddings
        y_val: Optional validation targets

    Returns:
        train_dataset (or tuple of train_dataset, val_dataset)

    """
    train_dataset = EditDataset(edit_embeddings=X_train, targets=y_train)

    if X_val is not None and y_val is not None:
        val_dataset = EditDataset(edit_embeddings=X_val, targets=y_val)
        return train_dataset, val_dataset
    else:
        return train_dataset
