#!/usr/bin/env python
"""Rate prediction label normalization comparison experiment.

Compares TWO normalization strategies for editing rate prediction:

1. **Raw unified**: Use editing_rate_normalized as-is (all on 0-1 scale).
   Baysal sites have rates 0.15-0.73, advisor sites have 0.003-0.987.

2. **Z-score per dataset**: Compute z-score (mean=0, std=1) independently
   per dataset_source on the training set, then apply the same mean/std
   to val/test. Predictions are in z-score space during training; inverted
   back to original scale per-dataset for interpretability.

For each normalization, two models are trained:
  - PooledMLP: pooled_orig -> MLP (no edit info, sequence-only baseline)
  - SubtractionMLP: (pooled_edited - pooled_orig) -> MLP

All models use MSE loss and linear output for regression.

Metrics: Spearman correlation, Pearson correlation, MSE, MAE.

Usage:
    python experiments/apobec/exp_rate_normalization.py
"""

import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr, pearsonr
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

EMB_DIR = PROJECT_ROOT / "data" / "processed" / "embeddings"
SPLITS_CSV = PROJECT_ROOT / "data" / "processed" / "splits_expanded.csv"
OUTPUT_DIR = (
    PROJECT_ROOT / "experiments" / "apobec" / "outputs" / "rate_normalization"
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class NormExpConfig:
    """Configuration for the normalization experiment."""

    d_model: int = 640
    hidden_dims: tuple = (256, 128)
    dropout: float = 0.2
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 15
    max_grad_norm: float = 1.0
    seed: int = 42


# ---------------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------------

def compute_regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """Compute Spearman, Pearson, MSE, MAE from numpy arrays."""
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) < 3:
        return {
            "spearman": float("nan"),
            "pearson": float("nan"),
            "mse": float("nan"),
            "mae": float("nan"),
            "n_samples": 0,
        }

    sp_rho, _ = spearmanr(y_true, y_pred)
    pe_r, _ = pearsonr(y_true, y_pred)
    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    return {
        "spearman": float(sp_rho),
        "pearson": float(pe_r),
        "mse": mse,
        "mae": mae,
        "n_samples": int(len(y_true)),
    }


# ---------------------------------------------------------------------------
# Rate regression dataset
# ---------------------------------------------------------------------------

class RateDataset(Dataset):
    """Dataset serving pre-computed pooled embeddings with rate targets.

    Modes:
      - "pooled_mlp": input = pooled_orig
      - "subtraction_mlp": input = pooled_edited - pooled_orig
    """

    def __init__(
        self,
        site_ids: List[str],
        targets: np.ndarray,
        pooled_orig: Dict[str, torch.Tensor],
        pooled_edited: Dict[str, torch.Tensor],
        mode: str = "subtraction_mlp",
    ):
        self.site_ids = site_ids
        self.targets = targets.astype(np.float32)
        self.pooled_orig = pooled_orig
        self.pooled_edited = pooled_edited
        self.mode = mode

    def __len__(self) -> int:
        return len(self.site_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sid = self.site_ids[idx]
        orig = self.pooled_orig[sid]
        edited = self.pooled_edited[sid]

        if self.mode == "pooled_mlp":
            x = orig
        elif self.mode == "subtraction_mlp":
            x = edited - orig
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        return x, target


# ---------------------------------------------------------------------------
# Regression MLP
# ---------------------------------------------------------------------------

class RegressionMLP(nn.Module):
    """MLP for rate regression with linear output (no sigmoid)."""

    def __init__(self, d_input: int, hidden_dims: tuple = (256, 128),
                 dropout: float = 0.2):
        super().__init__()
        layers = []
        in_dim = d_input
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Z-score normalization helpers
# ---------------------------------------------------------------------------

def compute_zscore_params(
    df: pd.DataFrame, rate_col: str = "editing_rate_normalized"
) -> Dict[str, Dict[str, float]]:
    """Compute per-dataset z-score parameters (mean, std) from a training set.

    Returns a dict mapping dataset_source -> {"mean": float, "std": float}.
    """
    params = {}
    for ds, group in df.groupby("dataset_source"):
        rates = group[rate_col].values
        mean = float(np.mean(rates))
        std = float(np.std(rates))
        if std < 1e-8:
            std = 1.0  # avoid division by zero for single-value datasets
        params[ds] = {"mean": mean, "std": std}
    return params


def apply_zscore(
    df: pd.DataFrame,
    zscore_params: Dict[str, Dict[str, float]],
    rate_col: str = "editing_rate_normalized",
) -> np.ndarray:
    """Apply z-score normalization using pre-computed per-dataset params.

    Sites from datasets not present in zscore_params use global stats.
    """
    result = np.full(len(df), np.nan, dtype=np.float32)
    for i, (_, row) in enumerate(df.iterrows()):
        ds = row["dataset_source"]
        rate = row[rate_col]
        if pd.isna(rate):
            continue
        if ds in zscore_params:
            p = zscore_params[ds]
            result[i] = (rate - p["mean"]) / p["std"]
        else:
            # Fallback: use global mean/std from all params
            all_means = [v["mean"] for v in zscore_params.values()]
            all_stds = [v["std"] for v in zscore_params.values()]
            result[i] = (rate - np.mean(all_means)) / np.mean(all_stds)
    return result


def invert_zscore(
    predictions: np.ndarray,
    dataset_sources: np.ndarray,
    zscore_params: Dict[str, Dict[str, float]],
) -> np.ndarray:
    """Invert z-score predictions back to original editing_rate_normalized scale."""
    result = np.full(len(predictions), np.nan, dtype=np.float32)
    for i in range(len(predictions)):
        ds = dataset_sources[i]
        if ds in zscore_params:
            p = zscore_params[ds]
            result[i] = predictions[i] * p["std"] + p["mean"]
        else:
            # Fallback
            all_means = [v["mean"] for v in zscore_params.values()]
            all_stds = [v["std"] for v in zscore_params.values()]
            result[i] = predictions[i] * np.mean(all_stds) + np.mean(all_means)
    return result


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def train_and_evaluate(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_targets: np.ndarray,
    val_targets: np.ndarray,
    test_targets: np.ndarray,
    pooled_orig: Dict[str, torch.Tensor],
    pooled_edited: Dict[str, torch.Tensor],
    model_mode: str,
    config: NormExpConfig,
) -> Dict:
    """Train a regression MLP and evaluate on val/test.

    Parameters
    ----------
    model_mode : str
        "pooled_mlp" or "subtraction_mlp" -- determines input construction.
    train/val/test_targets : np.ndarray
        The target values (either raw or z-scored).

    Returns
    -------
    Dict with train/val/test metrics and raw predictions.
    """
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    d_input = config.d_model  # both modes use d_model (640)

    def make_loader(df, targets, shuffle=False):
        ds = RateDataset(
            site_ids=df["site_id"].tolist(),
            targets=targets,
            pooled_orig=pooled_orig,
            pooled_edited=pooled_edited,
            mode=model_mode,
        )
        return DataLoader(
            ds, batch_size=config.batch_size, shuffle=shuffle, num_workers=0
        )

    train_loader = make_loader(train_df, train_targets, shuffle=True)
    val_loader = make_loader(val_df, val_targets)
    test_loader = make_loader(test_df, test_targets)

    model = RegressionMLP(d_input, config.hidden_dims, config.dropout)
    optimizer = AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-7)

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    best_state = None

    for epoch in range(1, config.epochs + 1):
        # Train
        model.train()
        train_loss_sum = 0.0
        n_batches = 0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = F.mse_loss(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            train_loss_sum += loss.item()
            n_batches += 1
        scheduler.step()

        # Validate
        model.eval()
        val_preds_list, val_targets_list = [], []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                pred = model(x_batch)
                val_preds_list.append(pred.numpy())
                val_targets_list.append(y_batch.numpy())
        vp = np.concatenate(val_preds_list)
        vt = np.concatenate(val_targets_list)
        val_loss = float(np.mean((vp - vt) ** 2))

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if epoch % 20 == 0 or epoch == 1:
            logger.info(
                "  Epoch %3d  train_mse=%.5f  val_mse=%.5f  patience=%d/%d",
                epoch, train_loss_sum / max(n_batches, 1), val_loss,
                patience_counter, config.patience,
            )

        if patience_counter >= config.patience:
            logger.info(
                "  Early stopping at epoch %d (best=%d, val_mse=%.5f)",
                epoch, best_epoch, best_val_loss,
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    logger.info("  Best epoch: %d, best val MSE: %.5f", best_epoch, best_val_loss)

    # Evaluate on all splits
    model.eval()
    results = {"best_epoch": best_epoch, "best_val_mse": best_val_loss}

    for split_name, loader in [
        ("train", train_loader), ("val", val_loader), ("test", test_loader)
    ]:
        all_preds, all_tgts = [], []
        with torch.no_grad():
            for x_batch, y_batch in loader:
                pred = model(x_batch)
                all_preds.append(pred.numpy())
                all_tgts.append(y_batch.numpy())
        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_tgts)
        results[split_name] = compute_regression_metrics(y_true, y_pred)
        results[f"{split_name}_predictions"] = y_pred
        results[f"{split_name}_targets"] = y_true

    return results


# ---------------------------------------------------------------------------
# Per-dataset evaluation
# ---------------------------------------------------------------------------

def evaluate_per_dataset(
    test_df: pd.DataFrame,
    test_preds: np.ndarray,
    test_targets: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """Compute metrics per dataset_source on the test set."""
    results = {}
    ds_values = test_df["dataset_source"].values
    for ds in sorted(test_df["dataset_source"].unique()):
        mask = ds_values == ds
        if mask.sum() < 3:
            results[ds] = {
                "spearman": float("nan"), "pearson": float("nan"),
                "mse": float("nan"), "mae": float("nan"),
                "n_samples": int(mask.sum()),
            }
        else:
            results[ds] = compute_regression_metrics(
                test_targets[mask], test_preds[mask]
            )
    return results


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary_table(all_results: Dict):
    """Print a formatted comparison table of all model x normalization combos."""
    print("\n" + "=" * 110)
    print("RATE NORMALIZATION EXPERIMENT - TEST SET RESULTS")
    print("=" * 110)
    print(
        f"{'Normalization':<16} {'Model':<20} "
        f"{'Spearman':>10} {'Pearson':>10} {'MSE':>10} {'MAE':>10} "
        f"{'N':>8} {'BestEp':>8}"
    )
    print("-" * 110)

    for key, res in all_results.items():
        norm_name, model_name = key.split(" | ")
        tm = res.get("test", {})
        ep = res.get("best_epoch", 0)
        print(
            f"{norm_name:<16} {model_name:<20} "
            f"{tm.get('spearman', float('nan')):>10.4f} "
            f"{tm.get('pearson', float('nan')):>10.4f} "
            f"{tm.get('mse', float('nan')):>10.6f} "
            f"{tm.get('mae', float('nan')):>10.6f} "
            f"{tm.get('n_samples', 0):>8d} "
            f"{ep:>8d}"
        )

    print("=" * 110)


def print_inverted_table(inverted_results: Dict):
    """Print z-score models evaluated in original scale (after inversion)."""
    print("\n" + "=" * 110)
    print("Z-SCORE MODELS - INVERTED BACK TO ORIGINAL SCALE (TEST SET)")
    print("=" * 110)
    print(
        f"{'Model':<20} "
        f"{'Spearman':>10} {'Pearson':>10} {'MSE':>10} {'MAE':>10} "
        f"{'N':>8}"
    )
    print("-" * 110)

    for model_name, metrics in inverted_results.items():
        print(
            f"{model_name:<20} "
            f"{metrics.get('spearman', float('nan')):>10.4f} "
            f"{metrics.get('pearson', float('nan')):>10.4f} "
            f"{metrics.get('mse', float('nan')):>10.6f} "
            f"{metrics.get('mae', float('nan')):>10.6f} "
            f"{metrics.get('n_samples', 0):>8d}"
        )

    print("=" * 110)


def print_per_dataset_table(per_dataset_results: Dict):
    """Print per-dataset Spearman table for all model x normalization combos."""
    # Collect all unique datasets
    all_datasets = set()
    for _, per_ds in per_dataset_results.items():
        all_datasets.update(per_ds.keys())
    all_datasets = sorted(all_datasets)

    dataset_labels = {
        "advisor_c2t": "Levanon",
        "alqassim_2021": "Alqassim",
        "baysal_2016": "Baysal",
        "sharma_2015": "Sharma",
    }

    print("\n" + "=" * 110)
    print("PER-DATASET SPEARMAN CORRELATION (TEST SET)")
    print("=" * 110)
    header = f"{'Norm | Model':<36}"
    for ds in all_datasets:
        lbl = dataset_labels.get(ds, ds)
        header += f"{lbl:>18}"
    print(header)
    print("-" * 110)

    for key, per_ds in per_dataset_results.items():
        row = f"{key:<36}"
        for ds in all_datasets:
            v = per_ds.get(ds, {}).get("spearman", float("nan"))
            n = per_ds.get(ds, {}).get("n_samples", 0)
            if np.isnan(v):
                row += f"{'N/A':>18}"
            else:
                row += f"{v:>10.4f} (n={n:>3d})"
        print(row)

    print("=" * 110)


# ---------------------------------------------------------------------------
# JSON serializer
# ---------------------------------------------------------------------------

def _serialize(obj):
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return str(obj)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config = NormExpConfig()
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    t_global = time.time()

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    logger.info("Loading embeddings from %s ...", EMB_DIR)
    pooled_orig = torch.load(EMB_DIR / "rnafm_pooled.pt", weights_only=False)
    pooled_edited = torch.load(EMB_DIR / "rnafm_pooled_edited.pt", weights_only=False)
    logger.info("  Loaded %d pooled embeddings", len(pooled_orig))

    splits_df = pd.read_csv(SPLITS_CSV)
    logger.info("  Loaded %d sites from splits_expanded.csv", len(splits_df))

    # ------------------------------------------------------------------
    # Filter to positive sites with editing_rate > 0 and available embeddings
    # ------------------------------------------------------------------
    available_ids = set(pooled_orig.keys()) & set(pooled_edited.keys())
    rate_df = splits_df[
        (splits_df["label"] == 1)
        & (splits_df["editing_rate"] > 0)
        & (splits_df["editing_rate_normalized"].notna())
        & (splits_df["site_id"].isin(available_ids))
    ].copy()

    logger.info("Positive sites with editing_rate > 0 and embeddings: %d", len(rate_df))

    train_df = rate_df[rate_df["split"] == "train"].copy().reset_index(drop=True)
    val_df = rate_df[rate_df["split"] == "val"].copy().reset_index(drop=True)
    test_df = rate_df[rate_df["split"] == "test"].copy().reset_index(drop=True)
    logger.info("Train: %d, Val: %d, Test: %d", len(train_df), len(val_df), len(test_df))

    # ------------------------------------------------------------------
    # Per-dataset distribution summary
    # ------------------------------------------------------------------
    logger.info("\nPer-dataset rate distribution (editing_rate_normalized):")
    for ds in sorted(rate_df["dataset_source"].unique()):
        sub = rate_df[rate_df["dataset_source"] == ds]
        logger.info(
            "  %-15s  n=%4d  mean=%.4f  std=%.4f  range=[%.4f, %.4f]",
            ds, len(sub),
            sub["editing_rate_normalized"].mean(),
            sub["editing_rate_normalized"].std(),
            sub["editing_rate_normalized"].min(),
            sub["editing_rate_normalized"].max(),
        )

    # ------------------------------------------------------------------
    # Prepare targets for both normalization strategies
    # ------------------------------------------------------------------

    # Strategy 1: Raw unified (editing_rate_normalized as-is)
    raw_train_targets = train_df["editing_rate_normalized"].values.astype(np.float32)
    raw_val_targets = val_df["editing_rate_normalized"].values.astype(np.float32)
    raw_test_targets = test_df["editing_rate_normalized"].values.astype(np.float32)

    logger.info("\n--- Raw unified targets ---")
    logger.info(
        "  Train: mean=%.4f, std=%.4f",
        np.mean(raw_train_targets), np.std(raw_train_targets),
    )
    logger.info(
        "  Val:   mean=%.4f, std=%.4f",
        np.mean(raw_val_targets), np.std(raw_val_targets),
    )
    logger.info(
        "  Test:  mean=%.4f, std=%.4f",
        np.mean(raw_test_targets), np.std(raw_test_targets),
    )

    # Strategy 2: Z-score per dataset (computed on train set only)
    zscore_params = compute_zscore_params(train_df, "editing_rate_normalized")
    logger.info("\n--- Z-score per-dataset params (from train set) ---")
    for ds, params in zscore_params.items():
        logger.info("  %-15s  mean=%.4f  std=%.4f", ds, params["mean"], params["std"])

    zscore_train_targets = apply_zscore(train_df, zscore_params)
    zscore_val_targets = apply_zscore(val_df, zscore_params)
    zscore_test_targets = apply_zscore(test_df, zscore_params)

    logger.info("\n--- Z-scored targets ---")
    logger.info(
        "  Train: mean=%.4f, std=%.4f",
        np.nanmean(zscore_train_targets), np.nanstd(zscore_train_targets),
    )
    logger.info(
        "  Val:   mean=%.4f, std=%.4f",
        np.nanmean(zscore_val_targets), np.nanstd(zscore_val_targets),
    )
    logger.info(
        "  Test:  mean=%.4f, std=%.4f",
        np.nanmean(zscore_test_targets), np.nanstd(zscore_test_targets),
    )

    # ------------------------------------------------------------------
    # Train all model x normalization combinations
    # ------------------------------------------------------------------
    model_modes = ["pooled_mlp", "subtraction_mlp"]
    model_labels = {
        "pooled_mlp": "PooledMLP",
        "subtraction_mlp": "SubtractionMLP",
    }

    normalizations = {
        "raw_unified": {
            "train": raw_train_targets,
            "val": raw_val_targets,
            "test": raw_test_targets,
        },
        "zscore": {
            "train": zscore_train_targets,
            "val": zscore_val_targets,
            "test": zscore_test_targets,
        },
    }

    all_results = {}
    per_dataset_all = {}

    for norm_name, targets_dict in normalizations.items():
        for mode in model_modes:
            combo_key = f"{norm_name} | {model_labels[mode]}"

            logger.info("\n" + "=" * 70)
            logger.info("TRAINING: %s", combo_key)
            logger.info("=" * 70)

            t0 = time.time()
            results = train_and_evaluate(
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                train_targets=targets_dict["train"],
                val_targets=targets_dict["val"],
                test_targets=targets_dict["test"],
                pooled_orig=pooled_orig,
                pooled_edited=pooled_edited,
                model_mode=mode,
                config=config,
            )
            elapsed = time.time() - t0

            results["train_time_seconds"] = elapsed
            all_results[combo_key] = results

            logger.info("  Time: %.1fs", elapsed)
            tm = results["test"]
            logger.info(
                "  Test: Spearman=%.4f  Pearson=%.4f  MSE=%.6f  MAE=%.6f",
                tm["spearman"], tm["pearson"], tm["mse"], tm["mae"],
            )

            # Per-dataset evaluation
            per_ds = evaluate_per_dataset(
                test_df,
                results["test_predictions"],
                results["test_targets"],
            )
            per_dataset_all[combo_key] = per_ds

    # ------------------------------------------------------------------
    # Z-score inversion: evaluate z-score models in original scale
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("Z-SCORE INVERSION: evaluating in original rate scale")
    logger.info("=" * 70)

    inverted_results = {}
    inverted_per_dataset = {}

    for mode in model_modes:
        combo_key = f"zscore | {model_labels[mode]}"
        results = all_results[combo_key]

        # Invert predictions back to original scale
        test_preds_zspace = results["test_predictions"]
        test_ds_sources = test_df["dataset_source"].values
        inverted_preds = invert_zscore(test_preds_zspace, test_ds_sources, zscore_params)

        # Compare against raw targets (original scale)
        inv_metrics = compute_regression_metrics(raw_test_targets, inverted_preds)
        inverted_results[model_labels[mode]] = inv_metrics

        logger.info(
            "  %s (inverted): Spearman=%.4f  Pearson=%.4f  MSE=%.6f  MAE=%.6f",
            model_labels[mode],
            inv_metrics["spearman"], inv_metrics["pearson"],
            inv_metrics["mse"], inv_metrics["mae"],
        )

        # Per-dataset inverted evaluation
        inv_per_ds = evaluate_per_dataset(test_df, inverted_preds, raw_test_targets)
        inverted_per_dataset[f"zscore_inverted | {model_labels[mode]}"] = inv_per_ds

    # ------------------------------------------------------------------
    # Print summary tables
    # ------------------------------------------------------------------
    print_summary_table(all_results)
    print_inverted_table(inverted_results)
    print_per_dataset_table({**per_dataset_all, **inverted_per_dataset})

    # ------------------------------------------------------------------
    # Key findings
    # ------------------------------------------------------------------
    total_time = time.time() - t_global

    print("\n" + "=" * 110)
    print("KEY FINDINGS")
    print("=" * 110)

    # Compare raw vs z-score for each model
    for mode in model_modes:
        ml = model_labels[mode]
        raw_key = f"raw_unified | {ml}"
        zscore_key = f"zscore | {ml}"

        raw_sp = all_results[raw_key]["test"]["spearman"]
        zscore_sp = all_results[zscore_key]["test"]["spearman"]
        inv_sp = inverted_results[ml]["spearman"]

        print(f"\n  {ml}:")
        print(f"    Raw unified   Spearman = {raw_sp:.4f}")
        print(f"    Z-score       Spearman = {zscore_sp:.4f} (in z-space)")
        print(f"    Z-score (inv) Spearman = {inv_sp:.4f} (inverted to original scale)")

        # Note: Spearman is rank-based, so it should be identical for z-score
        # and z-score inverted. The difference is in MSE/MAE.
        raw_mse = all_results[raw_key]["test"]["mse"]
        inv_mse = inverted_results[ml]["mse"]
        print(f"    Raw MSE       = {raw_mse:.6f}")
        print(f"    Z-inv MSE     = {inv_mse:.6f}")
        if inv_mse < raw_mse:
            pct = (1.0 - inv_mse / raw_mse) * 100
            print(f"    --> Z-score inverted has {pct:.1f}% lower MSE")
        else:
            pct = (inv_mse / raw_mse - 1.0) * 100
            print(f"    --> Raw unified has {pct:.1f}% lower MSE")

    print(f"\n  Total experiment time: {total_time:.1f}s")
    print("=" * 110)

    # ------------------------------------------------------------------
    # Save JSON results
    # ------------------------------------------------------------------
    json_results = {
        "description": (
            "Rate normalization comparison experiment. "
            "Compares raw unified (editing_rate_normalized 0-1) vs "
            "z-score per dataset normalization for rate prediction. "
            "Models: PooledMLP (orig embedding only) and SubtractionMLP (edit diff). "
            "All models use MSE loss with linear output."
        ),
        "config": {
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "patience": config.patience,
            "hidden_dims": list(config.hidden_dims),
            "dropout": config.dropout,
            "seed": config.seed,
        },
        "data": {
            "n_train": len(train_df),
            "n_val": len(val_df),
            "n_test": len(test_df),
            "per_dataset_train_counts": {
                ds: int((train_df["dataset_source"] == ds).sum())
                for ds in sorted(train_df["dataset_source"].unique())
            },
            "per_dataset_test_counts": {
                ds: int((test_df["dataset_source"] == ds).sum())
                for ds in sorted(test_df["dataset_source"].unique())
            },
            "raw_target_stats": {
                "train_mean": float(np.mean(raw_train_targets)),
                "train_std": float(np.std(raw_train_targets)),
                "test_mean": float(np.mean(raw_test_targets)),
                "test_std": float(np.std(raw_test_targets)),
            },
            "zscore_params": zscore_params,
        },
        "models": {},
        "inverted_zscore": {},
        "per_dataset": {},
        "inverted_per_dataset": {},
        "total_time_seconds": total_time,
    }

    for combo_key, res in all_results.items():
        json_results["models"][combo_key] = {
            "best_epoch": res["best_epoch"],
            "best_val_mse": res["best_val_mse"],
            "train_time_seconds": res.get("train_time_seconds", 0),
            "train": res.get("train", {}),
            "val": res.get("val", {}),
            "test": res.get("test", {}),
        }

    for model_name, metrics in inverted_results.items():
        json_results["inverted_zscore"][model_name] = metrics

    for combo_key, per_ds in per_dataset_all.items():
        json_results["per_dataset"][combo_key] = per_ds

    for combo_key, per_ds in inverted_per_dataset.items():
        json_results["inverted_per_dataset"][combo_key] = per_ds

    results_path = OUTPUT_DIR / "rate_normalization_results.json"
    with open(results_path, "w") as f:
        json.dump(json_results, f, indent=2, default=_serialize)
    logger.info("\nResults saved to %s", results_path)

    logger.info("Done.")


if __name__ == "__main__":
    main()
