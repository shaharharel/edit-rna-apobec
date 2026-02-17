"""
Label normalization utilities for cross-dataset experiments.

When training on one dataset and testing on another, labels may have different
scales (e.g., enrichment scores vs Kd values). These utilities normalize labels
to make them comparable across datasets.
"""

import numpy as np
from typing import Tuple, Optional, Union, List
import torch


def zscore_normalize(
    values: Union[np.ndarray, torch.Tensor],
    mean: Optional[float] = None,
    std: Optional[float] = None,
) -> Tuple[Union[np.ndarray, torch.Tensor], float, float]:
    """
    Apply z-score normalization: (x - mean) / std.

    Args:
        values: Array of values to normalize
        mean: Pre-computed mean (if None, computed from values)
        std: Pre-computed std (if None, computed from values)

    Returns:
        normalized: Normalized values
        mean: Mean used for normalization
        std: Std used for normalization
    """
    if isinstance(values, torch.Tensor):
        if mean is None:
            mean = values.float().mean().item()
        if std is None:
            std = values.float().std().item()
        std = max(std, 1e-8)  # Avoid division by zero
        normalized = (values - mean) / std
    else:
        if mean is None:
            mean = float(np.mean(values))
        if std is None:
            std = float(np.std(values))
        std = max(std, 1e-8)
        normalized = (values - mean) / std

    return normalized, mean, std


def minmax_normalize(
    values: Union[np.ndarray, torch.Tensor],
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
) -> Tuple[Union[np.ndarray, torch.Tensor], float, float]:
    """
    Apply min-max normalization: (x - min) / (max - min).

    Args:
        values: Array of values to normalize
        min_val: Pre-computed min (if None, computed from values)
        max_val: Pre-computed max (if None, computed from values)

    Returns:
        normalized: Normalized values in [0, 1]
        min_val: Min used for normalization
        max_val: Max used for normalization
    """
    if isinstance(values, torch.Tensor):
        if min_val is None:
            min_val = values.float().min().item()
        if max_val is None:
            max_val = values.float().max().item()
        range_val = max(max_val - min_val, 1e-8)
        normalized = (values - min_val) / range_val
    else:
        if min_val is None:
            min_val = float(np.min(values))
        if max_val is None:
            max_val = float(np.max(values))
        range_val = max(max_val - min_val, 1e-8)
        normalized = (values - min_val) / range_val

    return normalized, min_val, max_val


def rank_normalize(
    values: Union[np.ndarray, torch.Tensor],
) -> Union[np.ndarray, torch.Tensor]:
    """
    Apply rank normalization: convert values to their percentile ranks.

    This is useful when you only care about relative ordering, not magnitudes.
    Output is in [0, 1] where 0 is the lowest value and 1 is the highest.

    Args:
        values: Array of values to normalize

    Returns:
        Normalized values in [0, 1] based on rank
    """
    if isinstance(values, torch.Tensor):
        n = len(values)
        ranks = values.argsort().argsort().float()
        return ranks / (n - 1) if n > 1 else torch.zeros_like(values)
    else:
        from scipy.stats import rankdata
        n = len(values)
        ranks = rankdata(values, method='average')
        return (ranks - 1) / (n - 1) if n > 1 else np.zeros_like(values)


def normalize_for_cross_dataset(
    train_labels: Union[np.ndarray, torch.Tensor],
    test_labels: Union[np.ndarray, torch.Tensor],
    method: str = 'zscore',
) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
    """
    Normalize train and test labels independently for cross-dataset evaluation.

    Each dataset is normalized using its own statistics, which allows for fair
    comparison when datasets have different label scales.

    Args:
        train_labels: Training set labels
        test_labels: Test set labels
        method: Normalization method ('zscore', 'minmax', 'rank')

    Returns:
        train_normalized: Normalized training labels
        test_normalized: Normalized test labels
    """
    if method == 'zscore':
        train_norm, _, _ = zscore_normalize(train_labels)
        test_norm, _, _ = zscore_normalize(test_labels)
    elif method == 'minmax':
        train_norm, _, _ = minmax_normalize(train_labels)
        test_norm, _, _ = minmax_normalize(test_labels)
    elif method == 'rank':
        train_norm = rank_normalize(train_labels)
        test_norm = rank_normalize(test_labels)
    else:
        raise ValueError(f"Unknown normalization method: {method}. "
                         f"Use 'zscore', 'minmax', or 'rank'.")

    return train_norm, test_norm


class CrossDatasetNormalizer:
    """
    Normalizer for cross-dataset experiments.

    Stores normalization statistics for reproducibility and supports
    applying the same normalization to new data.
    """

    def __init__(self, method: str = 'zscore'):
        """
        Initialize normalizer.

        Args:
            method: Normalization method ('zscore', 'minmax', 'rank')
        """
        self.method = method
        self.train_stats = {}
        self.test_stats = {}

    def fit_transform(
        self,
        train_labels: np.ndarray,
        test_labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit normalizer and transform labels.

        Args:
            train_labels: Training labels
            test_labels: Test labels

        Returns:
            train_normalized, test_normalized
        """
        if self.method == 'zscore':
            train_norm, train_mean, train_std = zscore_normalize(train_labels)
            test_norm, test_mean, test_std = zscore_normalize(test_labels)
            self.train_stats = {'mean': train_mean, 'std': train_std}
            self.test_stats = {'mean': test_mean, 'std': test_std}

        elif self.method == 'minmax':
            train_norm, train_min, train_max = minmax_normalize(train_labels)
            test_norm, test_min, test_max = minmax_normalize(test_labels)
            self.train_stats = {'min': train_min, 'max': train_max}
            self.test_stats = {'min': test_min, 'max': test_max}

        elif self.method == 'rank':
            train_norm = rank_normalize(train_labels)
            test_norm = rank_normalize(test_labels)
            self.train_stats = {'n': len(train_labels)}
            self.test_stats = {'n': len(test_labels)}

        else:
            raise ValueError(f"Unknown method: {self.method}")

        return train_norm, test_norm

    def transform_train(self, values: np.ndarray) -> np.ndarray:
        """Apply training set normalization to new values."""
        if self.method == 'zscore':
            return (values - self.train_stats['mean']) / self.train_stats['std']
        elif self.method == 'minmax':
            range_val = self.train_stats['max'] - self.train_stats['min']
            return (values - self.train_stats['min']) / max(range_val, 1e-8)
        else:
            return rank_normalize(values)

    def transform_test(self, values: np.ndarray) -> np.ndarray:
        """Apply test set normalization to new values."""
        if self.method == 'zscore':
            return (values - self.test_stats['mean']) / self.test_stats['std']
        elif self.method == 'minmax':
            range_val = self.test_stats['max'] - self.test_stats['min']
            return (values - self.test_stats['min']) / max(range_val, 1e-8)
        else:
            return rank_normalize(values)


def compute_spearman_correlation(
    predictions: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
) -> float:
    """
    Compute Spearman correlation coefficient.

    Spearman correlation is rank-based and therefore invariant to monotonic
    transformations, making it ideal for cross-dataset evaluation.

    Args:
        predictions: Predicted values
        targets: True values

    Returns:
        Spearman correlation coefficient
    """
    from scipy.stats import spearmanr

    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Flatten if needed
    predictions = predictions.flatten()
    targets = targets.flatten()

    corr, _ = spearmanr(predictions, targets)
    return float(corr) if not np.isnan(corr) else 0.0


def compute_direction_accuracy(
    predictions: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
) -> float:
    """
    Compute direction accuracy: fraction of pairs with correct relative ordering.

    This metric evaluates whether the model correctly predicts which of two
    variants has the higher value, without requiring absolute values to match.

    Args:
        predictions: Predicted values
        targets: True values

    Returns:
        Direction accuracy in [0, 1]
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    predictions = predictions.flatten()
    targets = targets.flatten()

    n = len(predictions)
    if n < 2:
        return 1.0

    # Sample pairs for efficiency (avoid O(n^2) for large datasets)
    max_pairs = 10000
    if n * (n - 1) // 2 > max_pairs:
        # Random sampling
        np.random.seed(42)
        idx = np.random.choice(n, size=min(n, 200), replace=False)
    else:
        idx = np.arange(n)

    correct = 0
    total = 0
    for i in range(len(idx)):
        for j in range(i + 1, len(idx)):
            ii, jj = idx[i], idx[j]
            pred_diff = predictions[ii] - predictions[jj]
            true_diff = targets[ii] - targets[jj]

            if true_diff == 0:
                continue  # Skip ties in ground truth

            total += 1
            if np.sign(pred_diff) == np.sign(true_diff):
                correct += 1

    return correct / max(total, 1)
