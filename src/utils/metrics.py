"""
Comprehensive evaluation metrics for molecular property prediction.

Includes standard regression metrics plus domain-specific metrics for
chemistry and drug discovery applications.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)
from scipy.stats import pearsonr, spearmanr, kendalltau
import warnings


class RegressionMetrics:
    """Compute comprehensive regression metrics for property prediction."""

    @staticmethod
    def compute_all(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sample_weights: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute all regression metrics.

        Args:
            y_true: True values [n_samples]
            y_pred: Predicted values [n_samples]
            sample_weights: Optional sample weights [n_samples]

        Returns:
            Dictionary of metric name -> value
        """
        # Remove NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        if not mask.any():
            return {metric: np.nan for metric in [
                'mse', 'rmse', 'mae', 'mape', 'r2', 'pearson_r', 'pearson_p',
                'spearman_r', 'spearman_p', 'kendall_tau', 'kendall_p',
                'max_error', 'mean_residual', 'std_residual'
            ]}

        y_true = y_true[mask]
        y_pred = y_pred[mask]
        if sample_weights is not None:
            sample_weights = sample_weights[mask]

        metrics = {}

        # Basic metrics
        metrics['mse'] = mean_squared_error(y_true, y_pred, sample_weight=sample_weights)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred, sample_weight=sample_weights)

        # MAPE (handle division by zero)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                metrics['mape'] = mean_absolute_percentage_error(
                    y_true, y_pred, sample_weight=sample_weights
                )
            except:
                metrics['mape'] = np.nan

        # R²
        metrics['r2'] = r2_score(y_true, y_pred, sample_weight=sample_weights)

        # Correlation metrics
        try:
            pearson_r, pearson_p = pearsonr(y_true, y_pred)
            metrics['pearson_r'] = pearson_r
            metrics['pearson_p'] = pearson_p
        except:
            metrics['pearson_r'] = np.nan
            metrics['pearson_p'] = np.nan

        try:
            spearman_r, spearman_p = spearmanr(y_true, y_pred)
            metrics['spearman_r'] = spearman_r
            metrics['spearman_p'] = spearman_p
        except:
            metrics['spearman_r'] = np.nan
            metrics['spearman_p'] = np.nan

        try:
            kendall_tau, kendall_p = kendalltau(y_true, y_pred)
            metrics['kendall_tau'] = kendall_tau
            metrics['kendall_p'] = kendall_p
        except:
            metrics['kendall_tau'] = np.nan
            metrics['kendall_p'] = np.nan

        # Residual analysis
        residuals = y_pred - y_true
        metrics['max_error'] = np.abs(residuals).max()
        metrics['mean_residual'] = residuals.mean()
        metrics['std_residual'] = residuals.std()

        # Additional metrics
        metrics['n_samples'] = len(y_true)
        metrics['y_mean'] = y_true.mean()
        metrics['y_std'] = y_true.std()
        metrics['y_range'] = y_true.max() - y_true.min()

        return metrics

    @staticmethod
    def compute_per_bin(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_bins: int = 5,
        metric: str = 'mae'
    ) -> Dict[str, Union[np.ndarray, List]]:
        """
        Compute metrics stratified by true value bins.

        Useful for understanding where model performs well/poorly.

        Args:
            y_true: True values
            y_pred: Predicted values
            n_bins: Number of bins for stratification
            metric: Metric to compute ('mae', 'mse', 'r2')

        Returns:
            Dictionary with bin edges, centers, and metric values
        """
        # Remove NaN
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        # Create bins
        bin_edges = np.percentile(y_true, np.linspace(0, 100, n_bins + 1))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_labels = np.digitize(y_true, bin_edges[1:-1])

        # Compute metric per bin
        metric_values = []
        bin_counts = []

        for bin_id in range(n_bins):
            bin_mask = bin_labels == bin_id
            if bin_mask.sum() == 0:
                metric_values.append(np.nan)
                bin_counts.append(0)
                continue

            y_true_bin = y_true[bin_mask]
            y_pred_bin = y_pred[bin_mask]

            if metric == 'mae':
                val = mean_absolute_error(y_true_bin, y_pred_bin)
            elif metric == 'mse':
                val = mean_squared_error(y_true_bin, y_pred_bin)
            elif metric == 'r2':
                val = r2_score(y_true_bin, y_pred_bin)
            else:
                raise ValueError(f"Unknown metric: {metric}")

            metric_values.append(val)
            bin_counts.append(bin_mask.sum())

        return {
            'bin_edges': bin_edges,
            'bin_centers': bin_centers,
            'metric_values': np.array(metric_values),
            'bin_counts': np.array(bin_counts),
            'metric_name': metric
        }


class MultiTaskMetrics:
    """Metrics for multi-task learning."""

    @staticmethod
    def compute_all_tasks(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        task_names: List[str],
        sample_weights: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Compute metrics for each task in multi-task setting.

        Args:
            y_true: True values [n_samples, n_tasks] (can have NaN)
            y_pred: Predictions [n_samples, n_tasks]
            task_names: Names of tasks
            sample_weights: Optional weights [n_samples]

        Returns:
            DataFrame with metrics per task
        """
        n_tasks = y_true.shape[1]
        results = []

        for i, task_name in enumerate(task_names):
            y_true_task = y_true[:, i]
            y_pred_task = y_pred[:, i]

            # Filter NaN (sparse labels)
            mask = ~np.isnan(y_true_task)
            if mask.sum() == 0:
                continue

            weights = sample_weights[mask] if sample_weights is not None else None

            metrics = RegressionMetrics.compute_all(
                y_true_task[mask],
                y_pred_task[mask],
                sample_weights=weights
            )

            metrics['task'] = task_name
            results.append(metrics)

        df = pd.DataFrame(results)

        # Reorder columns
        cols = ['task', 'n_samples', 'mse', 'rmse', 'mae', 'r2',
                'pearson_r', 'spearman_r', 'max_error']
        cols = [c for c in cols if c in df.columns]
        other_cols = [c for c in df.columns if c not in cols]
        df = df[cols + other_cols]

        return df

    @staticmethod
    def compute_macro_metrics(metrics_df: pd.DataFrame) -> Dict[str, float]:
        """
        Compute macro-averaged metrics across tasks.

        Args:
            metrics_df: DataFrame from compute_all_tasks()

        Returns:
            Dictionary of averaged metrics
        """
        numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns
        macro_metrics = {}

        for col in numeric_cols:
            if col != 'n_samples':
                macro_metrics[f'macro_{col}'] = metrics_df[col].mean()

        macro_metrics['total_samples'] = metrics_df['n_samples'].sum()
        macro_metrics['n_tasks'] = len(metrics_df)

        return macro_metrics


class RankingMetrics:
    """
    Ranking-based metrics for molecular screening applications.

    In drug discovery, we often care about ranking compounds correctly
    (e.g., enriching top candidates) more than exact value prediction.
    """

    @staticmethod
    def top_k_accuracy(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        k: int = 100,
        threshold: Optional[float] = None
    ) -> float:
        """
        Fraction of true top-k compounds correctly predicted in top-k.

        Args:
            y_true: True values
            y_pred: Predicted values
            k: Number of top compounds to consider
            threshold: If provided, define "active" as y_true >= threshold
                      instead of using top-k

        Returns:
            Top-k accuracy (0-1)
        """
        if threshold is None:
            # Top-k by value
            true_top_k = set(np.argsort(y_true)[-k:])
            pred_top_k = set(np.argsort(y_pred)[-k:])
        else:
            # Active compounds
            true_top_k = set(np.where(y_true >= threshold)[0])
            pred_top_k = set(np.argsort(y_pred)[-k:])

        if len(true_top_k) == 0:
            return np.nan

        overlap = len(true_top_k & pred_top_k)
        return overlap / len(true_top_k)

    @staticmethod
    def enrichment_factor(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        threshold: float,
        top_percent: float = 0.01
    ) -> float:
        """
        Enrichment factor: ratio of actives in top predictions vs random.

        EF = (actives_in_top / total_in_top) / (total_actives / total_compounds)

        Args:
            y_true: True values
            y_pred: Predicted values
            threshold: Threshold for defining "active" compounds
            top_percent: Fraction of top predictions to consider (e.g., 0.01 = 1%)

        Returns:
            Enrichment factor (>1 means better than random)
        """
        n = len(y_true)
        n_top = max(1, int(n * top_percent))

        # Actives
        is_active = y_true >= threshold
        n_actives = is_active.sum()

        if n_actives == 0:
            return np.nan

        # Top predictions
        top_indices = np.argsort(y_pred)[-n_top:]
        n_actives_in_top = is_active[top_indices].sum()

        # Enrichment
        actual_rate = n_actives_in_top / n_top
        random_rate = n_actives / n
        ef = actual_rate / random_rate if random_rate > 0 else np.nan

        return ef

    @staticmethod
    def ndcg_score(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        k: Optional[int] = None
    ) -> float:
        """
        Normalized Discounted Cumulative Gain at k.

        Measures ranking quality, with higher weight on top positions.

        Args:
            y_true: True relevance scores
            y_pred: Predicted scores
            k: Consider top-k only (None = all)

        Returns:
            NDCG score (0-1)
        """
        if k is None:
            k = len(y_true)

        # Sort by predicted scores
        order = np.argsort(y_pred)[::-1][:k]
        y_true_sorted = y_true[order]

        # DCG
        gains = 2 ** y_true_sorted - 1
        discounts = np.log2(np.arange(len(y_true_sorted)) + 2)
        dcg = np.sum(gains / discounts)

        # Ideal DCG
        ideal_order = np.argsort(y_true)[::-1][:k]
        y_true_ideal = y_true[ideal_order]
        ideal_gains = 2 ** y_true_ideal - 1
        idcg = np.sum(ideal_gains / discounts)

        if idcg == 0:
            return 0.0

        return dcg / idcg


class ChemistryMetrics:
    """Domain-specific metrics for chemistry applications."""

    @staticmethod
    def prediction_interval_coverage(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_std: np.ndarray,
        confidence: float = 0.95
    ) -> float:
        """
        Coverage of prediction intervals (for uncertainty quantification).

        Args:
            y_true: True values
            y_pred: Predicted mean values
            y_std: Predicted standard deviations
            confidence: Confidence level (e.g., 0.95 for 95%)

        Returns:
            Fraction of true values within confidence interval
        """
        from scipy.stats import norm

        z = norm.ppf((1 + confidence) / 2)
        lower = y_pred - z * y_std
        upper = y_pred + z * y_std

        in_interval = (y_true >= lower) & (y_true <= upper)
        coverage = in_interval.mean()

        return coverage

    @staticmethod
    def activity_cliff_detection(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        similarity: np.ndarray,
        similarity_threshold: float = 0.85,
        activity_threshold: float = 2.0
    ) -> Dict[str, float]:
        """
        Detect activity cliffs (similar molecules with large activity difference).

        Activity cliffs are challenging cases for ML models.

        Args:
            y_true: True activities [n_samples]
            y_pred: Predicted activities [n_samples]
            similarity: Pairwise similarity matrix [n_samples, n_samples]
            similarity_threshold: Molecules above this are "similar"
            activity_threshold: Activity difference above this is a "cliff"

        Returns:
            Dictionary with cliff statistics
        """
        n = len(y_true)
        cliffs = []
        cliff_errors = []

        for i in range(n):
            for j in range(i + 1, n):
                # Check similarity
                if similarity[i, j] < similarity_threshold:
                    continue

                # Check activity difference
                activity_diff = abs(y_true[i] - y_true[j])
                if activity_diff < activity_threshold:
                    continue

                # This is an activity cliff
                pred_diff = abs(y_pred[i] - y_pred[j])
                error = abs(activity_diff - pred_diff)

                cliffs.append((i, j, activity_diff, pred_diff))
                cliff_errors.append(error)

        if len(cliffs) == 0:
            return {
                'n_cliffs': 0,
                'cliff_mae': np.nan,
                'cliff_captured_rate': np.nan
            }

        cliff_errors = np.array(cliff_errors)
        true_diffs = np.array([c[2] for c in cliffs])
        pred_diffs = np.array([c[3] for c in cliffs])

        # How many cliffs were captured (pred_diff > threshold)?
        captured = (pred_diffs >= activity_threshold).sum()

        return {
            'n_cliffs': len(cliffs),
            'cliff_mae': cliff_errors.mean(),
            'cliff_captured_rate': captured / len(cliffs)
        }


def print_metrics_summary(
    metrics: Dict[str, float],
    title: str = "Metrics Summary"
):
    """
    Pretty print metrics dictionary.

    Args:
        metrics: Dictionary of metric name -> value
        title: Title for summary
    """
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    # Group by category
    basic = ['n_samples', 'mse', 'rmse', 'mae', 'mape', 'r2']
    correlation = ['pearson_r', 'pearson_p', 'spearman_r', 'spearman_p']
    residual = ['max_error', 'mean_residual', 'std_residual']

    for category, metric_names in [
        ("Basic Metrics", basic),
        ("Correlation", correlation),
        ("Residuals", residual)
    ]:
        print(f"\n{category}:")
        print("-" * 70)
        for name in metric_names:
            if name in metrics:
                val = metrics[name]
                if isinstance(val, (int, np.integer)):
                    print(f"  {name:20s}: {val:>12}")
                else:
                    print(f"  {name:20s}: {val:>12.4f}")

    print("=" * 70 + "\n")
