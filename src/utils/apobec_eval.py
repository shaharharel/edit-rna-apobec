"""Evaluation and interpretability utilities for APOBEC editing models.

Provides:
- Binary classification metrics (AUROC, AUPRC, F1, calibration)
- Multi-task evaluation across all prediction heads
- Edit embedding analysis (UMAP/t-SNE, clustering quality, geometry)
- Feature importance via permutation and attention
- Misclassification analysis (what does the model get wrong?)
- Comparison framework across model variants
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def binary_classification_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> Dict[str, float]:
    """Compute comprehensive binary classification metrics.

    Args:
        y_true: Binary labels (0/1).
        y_score: Predicted scores (probabilities or logits).

    Returns:
        Dict with auroc, auprc, f1, precision, recall, accuracy,
        optimal_threshold, n_positive, n_negative, brier_score,
        expected_calibration_error.
    """
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        brier_score_loss,
        f1_score,
        precision_recall_curve,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    mask = ~(np.isnan(y_true) | np.isnan(y_score))
    y_true = y_true[mask].astype(int)
    y_score = y_score[mask]

    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return {k: float("nan") for k in [
            "auroc", "auprc", "f1", "precision", "recall", "accuracy",
            "optimal_threshold", "n_positive", "n_negative",
            "brier_score", "ece",
        ]}

    metrics = {
        "auroc": roc_auc_score(y_true, y_score),
        "auprc": average_precision_score(y_true, y_score),
        "n_positive": int(y_true.sum()),
        "n_negative": int(len(y_true) - y_true.sum()),
    }

    # Optimal threshold (maximize F1)
    prec_arr, rec_arr, thresholds = precision_recall_curve(y_true, y_score)
    f1_arr = 2 * prec_arr[:-1] * rec_arr[:-1] / (prec_arr[:-1] + rec_arr[:-1] + 1e-8)
    best_idx = np.argmax(f1_arr)
    threshold = thresholds[best_idx] if len(thresholds) > 0 else 0.5

    y_pred = (y_score >= threshold).astype(int)
    metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)
    metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["optimal_threshold"] = float(threshold)

    # Calibration
    # Ensure scores are in [0, 1] for Brier score
    y_prob = _to_probability(y_score)
    metrics["brier_score"] = brier_score_loss(y_true, y_prob)
    metrics["ece"] = _expected_calibration_error(y_true, y_prob)

    return metrics


def _to_probability(y_score: np.ndarray) -> np.ndarray:
    """Convert scores to probabilities (apply sigmoid if needed)."""
    if np.any(y_score < 0) or np.any(y_score > 1):
        return 1.0 / (1.0 + np.exp(-y_score))
    return y_score


def _expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (ECE)."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        avg_pred = y_prob[mask].mean()
        avg_true = y_true[mask].mean()
        ece += mask.sum() * abs(avg_pred - avg_true)
    return ece / len(y_true)


# ---------------------------------------------------------------------------
# Multi-task evaluation
# ---------------------------------------------------------------------------

def evaluate_multitask(
    predictions: Dict[str, np.ndarray],
    targets: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, float]]:
    """Evaluate all task heads.

    Args:
        predictions: Dict of task_name -> predicted values.
        targets: Dict of task_name -> true values.

    Returns:
        Dict of task_name -> metrics dict.
    """
    from scipy.stats import spearmanr

    results = {}

    # Binary classification
    if "binary" in predictions and "binary" in targets:
        y_true = targets["binary"]
        y_score = predictions["binary"]
        results["binary"] = binary_classification_metrics(y_true, y_score)

    # Editing rate (regression on positive sites)
    if "rate" in predictions and "rate" in targets:
        y_true = targets["rate"]
        y_pred = predictions["rate"]
        mask = ~np.isnan(y_true) & (y_true > 0)
        if mask.sum() > 5:
            corr, pval = spearmanr(y_true[mask], y_pred[mask])
            mae = np.abs(y_true[mask] - y_pred[mask]).mean()
            results["rate"] = {
                "spearman_r": corr,
                "spearman_p": pval,
                "mae": mae,
                "n_samples": int(mask.sum()),
            }

    # Enzyme specificity (4-class)
    if "enzyme" in predictions and "enzyme" in targets:
        y_true = targets["enzyme"]
        y_pred = predictions["enzyme"]
        mask = y_true >= 0
        if mask.sum() > 0:
            from sklearn.metrics import accuracy_score, f1_score
            acc = accuracy_score(y_true[mask], y_pred[mask])
            f1 = f1_score(y_true[mask], y_pred[mask], average="macro", zero_division=0)
            results["enzyme"] = {
                "accuracy": acc,
                "f1_macro": f1,
                "n_samples": int(mask.sum()),
            }

    # Exonic function (3-class)
    if "function" in predictions and "function" in targets:
        y_true = targets["function"]
        y_pred = predictions["function"]
        mask = y_true >= 0
        if mask.sum() > 0:
            from sklearn.metrics import accuracy_score, f1_score
            acc = accuracy_score(y_true[mask], y_pred[mask])
            f1 = f1_score(y_true[mask], y_pred[mask], average="macro", zero_division=0)
            results["function"] = {
                "accuracy": acc,
                "f1_macro": f1,
                "n_samples": int(mask.sum()),
            }

    return results


# ---------------------------------------------------------------------------
# Embedding analysis
# ---------------------------------------------------------------------------

def analyze_edit_embeddings(
    embeddings: np.ndarray,
    labels: Dict[str, np.ndarray],
    output_dir: Optional[Path] = None,
) -> Dict[str, float]:
    """Analyze the geometry of edit embeddings.

    Computes:
    - Silhouette scores for various label groupings
    - Inter/intra-class distance ratios
    - UMAP coordinates (saved if output_dir provided)

    Args:
        embeddings: (N, d_edit) edit embedding matrix.
        labels: Dict of label_name -> array of labels (e.g., apobec_class, tissue, structure).
        output_dir: If provided, save UMAP coordinates and plots.

    Returns:
        Dict of metric_name -> value.
    """
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    results = {}
    emb = StandardScaler().fit_transform(embeddings)

    # Silhouette scores for each label grouping
    for label_name, label_arr in labels.items():
        valid = ~pd.isna(label_arr) if hasattr(label_arr, '__len__') else np.ones(len(embeddings), dtype=bool)
        if isinstance(label_arr[0], (str, np.str_)):
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            encoded = le.fit_transform(label_arr[valid])
        else:
            valid = valid & (label_arr >= 0)
            encoded = label_arr[valid].astype(int)

        n_classes = len(np.unique(encoded))
        if n_classes < 2 or n_classes > len(encoded) // 2:
            continue

        try:
            sil = silhouette_score(emb[valid], encoded, metric="cosine")
            results[f"silhouette_{label_name}"] = sil
        except Exception:
            pass

    # Inter vs intra class distances for binary label
    if "is_edited" in labels:
        pos_mask = labels["is_edited"] == 1
        neg_mask = labels["is_edited"] == 0
        if pos_mask.sum() > 0 and neg_mask.sum() > 0:
            pos_emb = emb[pos_mask]
            neg_emb = emb[neg_mask]

            # Intra-class: mean pairwise distance within positives
            intra_pos = _mean_pairwise_distance(pos_emb, max_pairs=5000)
            intra_neg = _mean_pairwise_distance(neg_emb, max_pairs=5000)
            intra_mean = (intra_pos + intra_neg) / 2

            # Inter-class: mean distance between positives and negatives
            inter = _mean_cross_distance(pos_emb, neg_emb, max_pairs=5000)

            results["intra_class_distance"] = intra_mean
            results["inter_class_distance"] = inter
            results["distance_ratio"] = inter / (intra_mean + 1e-8)

    # UMAP projection
    if output_dir is not None:
        _save_umap(emb, labels, output_dir)

    return results


def _mean_pairwise_distance(X: np.ndarray, max_pairs: int = 5000) -> float:
    """Compute mean pairwise cosine distance within a set of vectors."""
    n = len(X)
    if n < 2:
        return 0.0
    # Subsample if needed
    if n * (n - 1) // 2 > max_pairs:
        indices = np.random.choice(n, size=int(np.sqrt(2 * max_pairs)), replace=False)
        X = X[indices]
        n = len(X)

    from sklearn.metrics.pairwise import cosine_distances
    dist_matrix = cosine_distances(X)
    upper_tri = dist_matrix[np.triu_indices(n, k=1)]
    return upper_tri.mean()


def _mean_cross_distance(X1: np.ndarray, X2: np.ndarray, max_pairs: int = 5000) -> float:
    """Compute mean cosine distance between two sets."""
    if len(X1) * len(X2) > max_pairs:
        idx1 = np.random.choice(len(X1), size=min(100, len(X1)), replace=False)
        idx2 = np.random.choice(len(X2), size=min(100, len(X2)), replace=False)
        X1 = X1[idx1]
        X2 = X2[idx2]

    from sklearn.metrics.pairwise import cosine_distances
    return cosine_distances(X1, X2).mean()


def _save_umap(
    embeddings: np.ndarray,
    labels: Dict[str, np.ndarray],
    output_dir: Path,
):
    """Compute UMAP and save coordinates."""
    try:
        from umap import UMAP
    except ImportError:
        logger.info("UMAP not installed; skipping embedding visualization")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    coords = reducer.fit_transform(embeddings)

    # Save coordinates with labels
    df = pd.DataFrame({"umap_1": coords[:, 0], "umap_2": coords[:, 1]})
    for label_name, label_arr in labels.items():
        df[label_name] = label_arr

    df.to_csv(output_dir / "umap_coordinates.csv", index=False)
    logger.info("UMAP coordinates saved to %s", output_dir / "umap_coordinates.csv")


# ---------------------------------------------------------------------------
# Permutation feature importance
# ---------------------------------------------------------------------------

@torch.no_grad()
def permutation_importance(
    model: nn.Module,
    dataloader,
    feature_names: List[str],
    device: torch.device,
    n_repeats: int = 5,
    metric: str = "auroc",
) -> pd.DataFrame:
    """Compute permutation feature importance for the model.

    Shuffles each input feature one at a time and measures the drop in
    the specified metric.

    Args:
        model: Trained APOBECModel.
        dataloader: Validation DataLoader.
        feature_names: Names of features to permute.
        device: Torch device.
        n_repeats: Number of random permutation repeats.
        metric: Metric to measure (auroc, auprc, f1).

    Returns:
        DataFrame with columns: feature, importance_mean, importance_std.
    """
    model.eval()

    # Baseline performance
    baseline = _collect_predictions(model, dataloader, device)
    baseline_score = binary_classification_metrics(
        baseline["binary_targets"],
        baseline["binary_scores"],
    ).get(metric, 0)

    importances = []
    for feat_name in feature_names:
        scores = []
        for _ in range(n_repeats):
            permuted = _collect_predictions_with_permutation(
                model, dataloader, device, feat_name
            )
            perm_score = binary_classification_metrics(
                permuted["binary_targets"],
                permuted["binary_scores"],
            ).get(metric, 0)
            scores.append(baseline_score - perm_score)

        importances.append({
            "feature": feat_name,
            "importance_mean": np.mean(scores),
            "importance_std": np.std(scores),
        })

    df = pd.DataFrame(importances).sort_values("importance_mean", ascending=False)
    return df


def _collect_predictions(
    model: nn.Module,
    dataloader,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """Collect model predictions on a dataset."""
    all_targets = []
    all_scores = []

    for batch in dataloader:
        batch = _batch_to_device(batch, device)
        output = model(batch)
        preds = output["predictions"]

        binary_logits = preds["binary_logit"].squeeze(-1).cpu().numpy()
        binary_probs = 1.0 / (1.0 + np.exp(-binary_logits))
        all_scores.append(binary_probs)
        all_targets.append(batch["targets"]["binary"].cpu().numpy())

    return {
        "binary_targets": np.concatenate(all_targets),
        "binary_scores": np.concatenate(all_scores),
    }


def _collect_predictions_with_permutation(
    model: nn.Module,
    dataloader,
    device: torch.device,
    feature_name: str,
) -> Dict[str, np.ndarray]:
    """Collect predictions with a specific feature permuted."""
    all_targets = []
    all_scores = []

    for batch in dataloader:
        batch = _batch_to_device(batch, device)

        # Permute the specified feature
        if feature_name in batch:
            perm_idx = torch.randperm(batch[feature_name].shape[0])
            batch[feature_name] = batch[feature_name][perm_idx]
        elif feature_name in batch.get("targets", {}):
            pass  # Don't permute targets

        output = model(batch)
        preds = output["predictions"]

        binary_logits = preds["binary_logit"].squeeze(-1).cpu().numpy()
        binary_probs = 1.0 / (1.0 + np.exp(-binary_logits))
        all_scores.append(binary_probs)
        all_targets.append(batch["targets"]["binary"].cpu().numpy())

    return {
        "binary_targets": np.concatenate(all_targets),
        "binary_scores": np.concatenate(all_scores),
    }


# ---------------------------------------------------------------------------
# Misclassification analysis
# ---------------------------------------------------------------------------

def analyze_misclassifications(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metadata: pd.DataFrame,
    threshold: Optional[float] = None,
) -> Dict[str, pd.DataFrame]:
    """Analyze model errors to find patterns.

    Args:
        y_true: Binary labels.
        y_score: Predicted probabilities.
        metadata: DataFrame with metadata for each sample (site_id, gene, structure, etc.)
        threshold: Classification threshold. If None, uses optimal F1 threshold.

    Returns:
        Dict with DataFrames:
        - false_positives: FP samples with metadata
        - false_negatives: FN samples with metadata
        - error_by_category: Error rates grouped by metadata columns
    """
    if threshold is None:
        from sklearn.metrics import precision_recall_curve
        prec, rec, thresholds = precision_recall_curve(y_true, y_score)
        f1_arr = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-8)
        threshold = thresholds[np.argmax(f1_arr)] if len(thresholds) > 0 else 0.5

    y_pred = (y_score >= threshold).astype(int)

    # False positives and negatives
    fp_mask = (y_pred == 1) & (y_true == 0)
    fn_mask = (y_pred == 0) & (y_true == 1)

    results = {}

    fp_df = metadata[fp_mask].copy()
    fp_df["y_score"] = y_score[fp_mask]
    results["false_positives"] = fp_df

    fn_df = metadata[fn_mask].copy()
    fn_df["y_score"] = y_score[fn_mask]
    results["false_negatives"] = fn_df

    # Error rates by category
    category_cols = [c for c in metadata.columns if metadata[c].dtype == "object"
                     or metadata[c].nunique() < 20]

    error_analysis = []
    for col in category_cols:
        for val in metadata[col].dropna().unique():
            mask = metadata[col] == val
            n_total = mask.sum()
            if n_total < 5:
                continue
            n_fp = (fp_mask & mask).sum()
            n_fn = (fn_mask & mask).sum()
            n_correct = ((y_pred == y_true) & mask).sum()
            error_analysis.append({
                "feature": col,
                "value": val,
                "n_total": n_total,
                "n_fp": n_fp,
                "n_fn": n_fn,
                "accuracy": n_correct / n_total,
                "fp_rate": n_fp / max(1, (~y_true.astype(bool) & mask).sum()),
                "fn_rate": n_fn / max(1, (y_true.astype(bool) & mask).sum()),
            })

    results["error_by_category"] = pd.DataFrame(error_analysis)

    return results


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------

def compare_models(
    model_results: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    """Compare metrics across model variants.

    Args:
        model_results: Dict of model_name -> metrics_dict.

    Returns:
        DataFrame with models as rows and metrics as columns.
    """
    rows = []
    for model_name, metrics in model_results.items():
        row = {"model": model_name}
        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort by primary metric
    if "auroc" in df.columns:
        df = df.sort_values("auroc", ascending=False)

    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _batch_to_device(batch: Dict, device: torch.device) -> Dict:
    """Move batch tensors to device."""
    result = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            result[k] = v.to(device)
        elif isinstance(v, dict):
            result[k] = {kk: vv.to(device) if isinstance(vv, torch.Tensor) else vv
                         for kk, vv in v.items()}
        else:
            result[k] = v
    return result
