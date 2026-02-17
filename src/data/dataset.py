"""
PyTorch datasets for RNA edit effect prediction.

Provides datasets compatible with the edit-chem training pipeline.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path


class RNAPairDataset(Dataset):
    """
    PyTorch Dataset for RNA sequence pairs with Δ-labels.

    Loads pairs from long-format CSV and provides:
    - Pre-computed embeddings (if provided)
    - On-the-fly embedding computation (if embedder provided)
    - Multi-task support (multiple properties per pair)



    Args:
        pairs_df: DataFrame with pairs in long format
        embeddings: Optional dict of pre-computed embeddings
            {'seq_a': np.array, 'seq_b': np.array, 'edit': np.array}
        embedder: Optional RNA embedder for on-the-fly embedding
        property_name: Property to predict (or None for all)
        task_names: List of task names for multi-task learning

    Example:
        >>> dataset = RNAPairDataset(pairs_df, embeddings=precomputed)
        >>> seq_a_emb, seq_b_emb, edit_emb, delta = dataset[0]
    """

    def __init__(
        self,
        pairs_df: pd.DataFrame,
        embeddings: Optional[Dict[str, np.ndarray]] = None,
        embedder=None,
        property_name: Optional[str] = None,
        task_names: Optional[List[str]] = None
    ):
        self.embedder = embedder

        # Filter by property if specified
        if property_name is not None:
            pairs_df = pairs_df[pairs_df['property_name'] == property_name].copy()

        self.pairs_df = pairs_df.reset_index(drop=True)

        # Handle multi-task
        if task_names is not None:
            self.task_names = task_names
            self.multi_task = True
            self._prepare_multi_task()
        else:
            self.task_names = None
            self.multi_task = False

        # Store embeddings
        if embeddings is not None:
            self.seq_a_emb = embeddings['seq_a']
            self.seq_b_emb = embeddings['seq_b']
            self.edit_emb = embeddings.get('edit')
            self.precomputed = True
        else:
            self.seq_a_emb = None
            self.seq_b_emb = None
            self.edit_emb = None
            self.precomputed = False

    def _prepare_multi_task(self):
        """Prepare data for multi-task learning."""
        # Pivot to get one row per (seq_a, seq_b) pair with all properties
        self.pairs_df = self.pairs_df.pivot_table(
            index=['seq_a', 'seq_b'],
            columns='property_name',
            values='delta',
            aggfunc='first'
        ).reset_index()

        # Ensure all task columns exist
        for task in self.task_names:
            if task not in self.pairs_df.columns:
                self.pairs_df[task] = np.nan

    def __len__(self) -> int:
        return len(self.pairs_df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        row = self.pairs_df.iloc[idx]

        # Get embeddings
        if self.precomputed:
            seq_a_emb = torch.FloatTensor(self.seq_a_emb[idx])
            seq_b_emb = torch.FloatTensor(self.seq_b_emb[idx])

            if self.edit_emb is not None:
                edit_emb = torch.FloatTensor(self.edit_emb[idx])
            else:
                edit_emb = seq_b_emb - seq_a_emb

        elif self.embedder is not None:
            seq_a = row['seq_a']
            seq_b = row['seq_b']

            seq_a_emb = torch.FloatTensor(self.embedder.encode(seq_a))
            seq_b_emb = torch.FloatTensor(self.embedder.encode(seq_b))
            edit_emb = seq_b_emb - seq_a_emb

        else:
            raise ValueError("No embeddings or embedder provided")

        # Get delta(s)
        if self.multi_task:
            deltas = []
            for task in self.task_names:
                delta = row.get(task, np.nan)
                deltas.append(float(delta) if not pd.isna(delta) else float('nan'))
            delta = torch.FloatTensor(deltas)
        else:
            delta = torch.FloatTensor([row['delta']])

        return seq_a_emb, seq_b_emb, edit_emb, delta


class RNASequenceDataset(Dataset):
    """
    PyTorch Dataset for single RNA sequences (baseline property prediction).

    Used for training PropertyPredictor: f(sequence) → property.

    Args:
        sequences_df: DataFrame with sequences and property values
        embeddings: Optional pre-computed embeddings
        embedder: Optional RNA embedder for on-the-fly embedding
        seq_col: Column name for sequence
        value_col: Column name for property value
    """

    def __init__(
        self,
        sequences_df: pd.DataFrame,
        embeddings: Optional[np.ndarray] = None,
        embedder=None,
        seq_col: str = 'sequence',
        value_col: str = 'value'
    ):
        self.sequences_df = sequences_df.reset_index(drop=True)
        self.seq_col = seq_col
        self.value_col = value_col
        self.embedder = embedder
        self.embeddings = embeddings
        self.precomputed = embeddings is not None

    def __len__(self) -> int:
        return len(self.sequences_df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.sequences_df.iloc[idx]

        # Get embedding
        if self.precomputed:
            emb = torch.FloatTensor(self.embeddings[idx])
        elif self.embedder is not None:
            seq = row[self.seq_col]
            emb = torch.FloatTensor(self.embedder.encode(seq))
        else:
            raise ValueError("No embeddings or embedder provided")

        # Get value
        value = torch.FloatTensor([row[self.value_col]])

        return emb, value


class EmbeddingCache:
    """
    Cache for pre-computed RNA embeddings.

    Computes embeddings once and stores them for efficient reuse.

    Args:
        embedder: RNA embedder to use
        cache_dir: Optional directory for disk caching
    """

    def __init__(self, embedder, cache_dir: Optional[Path] = None):
        self.embedder = embedder
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._memory_cache = {}

    def get_embeddings(
        self,
        sequences: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Get embeddings for sequences (with caching).

        Args:
            sequences: List of RNA sequences
            batch_size: Batch size for embedding
            show_progress: Show progress bar

        Returns:
            Embeddings array of shape (n_sequences, embedding_dim)
        """
        # Check memory cache first
        cache_key = hash(tuple(sequences))
        if cache_key in self._memory_cache:
            return self._memory_cache[cache_key]

        # Check disk cache
        if self.cache_dir is not None:
            cache_file = self.cache_dir / f"embeddings_{cache_key}.npy"
            if cache_file.exists():
                embeddings = np.load(cache_file)
                self._memory_cache[cache_key] = embeddings
                return embeddings

        # Compute embeddings
        if hasattr(self.embedder, 'batch_encode'):
            embeddings = self.embedder.batch_encode(
                sequences,
                batch_size=batch_size,
                show_progress=show_progress
            )
        else:
            if show_progress:
                from tqdm import tqdm
                iterator = tqdm(sequences, desc="Computing embeddings")
            else:
                iterator = sequences

            embeddings = np.array([
                self.embedder.encode(seq) for seq in iterator
            ])

        # Cache
        self._memory_cache[cache_key] = embeddings

        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(cache_file, embeddings)

        return embeddings

    def get_pair_embeddings(
        self,
        pairs_df: pd.DataFrame,
        batch_size: int = 32,
        show_progress: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Get embeddings for pair sequences.

        Args:
            pairs_df: DataFrame with 'seq_a' and 'seq_b' columns
            batch_size: Batch size
            show_progress: Show progress

        Returns:
            Dict with 'seq_a', 'seq_b', 'edit' embeddings
        """
        # Get unique sequences
        all_seqs_a = pairs_df['seq_a'].tolist()
        all_seqs_b = pairs_df['seq_b'].tolist()

        unique_seqs = list(set(all_seqs_a + all_seqs_b))

        # Compute embeddings for unique sequences
        seq_to_idx = {seq: i for i, seq in enumerate(unique_seqs)}
        unique_embs = self.get_embeddings(
            unique_seqs,
            batch_size=batch_size,
            show_progress=show_progress
        )

        # Map back to pairs
        seq_a_emb = np.array([unique_embs[seq_to_idx[s]] for s in all_seqs_a])
        seq_b_emb = np.array([unique_embs[seq_to_idx[s]] for s in all_seqs_b])
        edit_emb = seq_b_emb - seq_a_emb

        return {
            'seq_a': seq_a_emb,
            'seq_b': seq_b_emb,
            'edit': edit_emb
        }

    def clear_cache(self):
        """Clear memory cache."""
        self._memory_cache.clear()


def create_data_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
    embedder=None,
    embeddings_train: Optional[Dict] = None,
    embeddings_val: Optional[Dict] = None,
    embeddings_test: Optional[Dict] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    task_names: Optional[List[str]] = None
) -> Dict[str, torch.utils.data.DataLoader]:
    """
    Create DataLoaders for training RNA models.

    Args:
        train_df: Training pairs DataFrame
        val_df: Validation pairs DataFrame
        test_df: Optional test pairs DataFrame
        embedder: RNA embedder (if embeddings not pre-computed)
        embeddings_*: Pre-computed embeddings dicts
        batch_size: Batch size
        num_workers: Number of data loading workers
        task_names: Task names for multi-task learning

    Returns:
        Dict with 'train', 'val', and optionally 'test' DataLoaders
    """
    from torch.utils.data import DataLoader

    # Create datasets
    train_dataset = RNAPairDataset(
        train_df,
        embeddings=embeddings_train,
        embedder=embedder,
        task_names=task_names
    )

    val_dataset = RNAPairDataset(
        val_df,
        embeddings=embeddings_val,
        embedder=embedder,
        task_names=task_names
    )

    loaders = {
        'train': DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        ),
        'val': DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    }

    if test_df is not None:
        test_dataset = RNAPairDataset(
            test_df,
            embeddings=embeddings_test,
            embedder=embedder,
            task_names=task_names
        )
        loaders['test'] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

    return loaders
