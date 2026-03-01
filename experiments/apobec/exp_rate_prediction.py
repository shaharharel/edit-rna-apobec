#!/usr/bin/env python
"""Editing RATE prediction (regression) evaluation across all DL architectures.

Evaluates how well each architecture can predict the continuous editing rate
(log2-transformed) from RNA-FM embeddings. Only positive sites with valid
editing_rate are used (train/val/test splits preserved).

Models evaluated:
  1. SubtractionMLP:  (pooled_edited - pooled_orig) -> MLP -> rate
  2. ConcatMLP:       [pooled_orig; pooled_edited] -> MLP -> rate
  3. OrigOnlyMLP:     pooled_orig -> MLP -> rate  (baseline, no edit info)
  4. EditRNA-A3A:     Full multi-task model, rate head output

Metrics: Spearman rho, Pearson r, MSE, R^2
Target:  log2(editing_rate + 0.01) after normalizing rates to [0,1]

Per-dataset evaluation is also performed for datasets with sufficient test data.

Usage:
    python experiments/apobec/exp_rate_prediction.py
"""

import json
import logging
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error, r2_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.apobec_dataset import (
    APOBECDataConfig,
    APOBECDataset,
    APOBECSiteSample,
    N_TISSUES,
    apobec_collate_fn,
    get_flanking_context,
)
from models.editrna_a3a import EditRNA_A3A, EditRNAConfig
from models.encoders import CachedRNAEncoder

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

EMB_DIR = PROJECT_ROOT / "data" / "processed" / "embeddings"
SPLITS_CSV = PROJECT_ROOT / "data" / "processed" / "splits_expanded.csv"
SEQ_JSON = PROJECT_ROOT / "data" / "processed" / "site_sequences.json"
STRUCT_CACHE = EMB_DIR / "vienna_structure_cache.npz"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "apobec" / "outputs" / "rate_prediction"

# Only datasets that have editing_rate data (asaoka_2019 has none)
DATASET_SOURCES = ["advisor_c2t", "alqassim_2021", "sharma_2015", "baysal_2016"]
DATASET_LABELS = {
    "advisor_c2t": "Levanon",
    "alqassim_2021": "Alqassim",
    "sharma_2015": "Sharma",
    "baysal_2016": "Baysal",
}


# ---------------------------------------------------------------------------
# Target transformation
# ---------------------------------------------------------------------------

def compute_log2_rate(editing_rate):
    """Convert raw editing rate to log2(rate + 0.01) after normalizing to [0,1].

    Rates > 1.0 are assumed to be percentages and divided by 100.
    Returns NaN for NaN / non-positive inputs.
    """
    if pd.isna(editing_rate) or editing_rate < 0:
        return float("nan")
    rate = float(editing_rate)
    if rate > 1.0:
        rate = rate / 100.0
    return np.log2(rate + 0.01)


# ---------------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------------

def compute_regression_metrics(y_true, y_pred):
    """Compute Spearman, Pearson, MSE, R^2 from numpy arrays.

    Returns a dict. Values are NaN if fewer than 3 valid pairs.
    """
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) < 3:
        return {
            "spearman": float("nan"),
            "pearson": float("nan"),
            "mse": float("nan"),
            "r2": float("nan"),
            "n_samples": 0,
        }

    sp_rho, _ = spearmanr(y_true, y_pred)
    pe_r, _ = pearsonr(y_true, y_pred)
    mse = float(mean_squared_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    return {
        "spearman": float(sp_rho),
        "pearson": float(pe_r),
        "mse": mse,
        "r2": r2,
        "n_samples": int(len(y_true)),
    }


# ---------------------------------------------------------------------------
# Dataset for pooled-embedding MLP models
# ---------------------------------------------------------------------------

class RateRegressionDataset(Dataset):
    """Dataset for rate regression from pre-computed pooled embeddings.

    Supports three input modes:
      - "subtraction": input = pooled_edited - pooled_orig
      - "concat":      input = [pooled_orig; pooled_edited]
      - "orig_only":   input = pooled_orig
    """

    def __init__(self, site_ids, targets, pooled_orig, pooled_edited, mode="subtraction"):
        self.site_ids = site_ids
        self.targets = targets  # numpy array of log2 rates
        self.pooled_orig = pooled_orig
        self.pooled_edited = pooled_edited
        self.mode = mode

    def __len__(self):
        return len(self.site_ids)

    def __getitem__(self, idx):
        sid = self.site_ids[idx]
        orig = self.pooled_orig[sid]
        edited = self.pooled_edited[sid]

        if self.mode == "subtraction":
            x = edited - orig
        elif self.mode == "concat":
            x = torch.cat([orig, edited], dim=-1)
        elif self.mode == "orig_only":
            x = orig
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        return x, target


# ---------------------------------------------------------------------------
# MLP regression model
# ---------------------------------------------------------------------------

class RateRegressionMLP(nn.Module):
    """MLP for rate regression (arbitrary input dim -> scalar output)."""

    def __init__(self, d_input, hidden=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_input, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# MLP training / evaluation
# ---------------------------------------------------------------------------

def train_eval_mlp(train_df, val_df, test_df, pooled_orig, pooled_edited,
                   mode="subtraction", d_model=640, epochs=80, lr=1e-3,
                   patience=15, seed=42):
    """Train and evaluate a rate-regression MLP.

    Parameters
    ----------
    mode : str
        "subtraction", "concat", or "orig_only"

    Returns
    -------
    dict with train/val/test metrics plus arrays of predictions.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if mode == "concat":
        d_input = d_model * 2
    else:
        d_input = d_model

    def make_loader(df, shuffle=False, batch_size=64):
        ds = RateRegressionDataset(
            df["site_id"].tolist(),
            df["log2_rate"].values.astype(np.float32),
            pooled_orig, pooled_edited,
            mode=mode,
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    train_loader = make_loader(train_df, shuffle=True)
    val_loader = make_loader(val_df)
    test_loader = make_loader(test_df)

    model = RateRegressionMLP(d_input)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = F.mse_loss(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # Validate
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                pred = model(x_batch)
                val_preds.append(pred.numpy())
                val_targets.append(y_batch.numpy())
        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)
        val_loss = float(np.mean((val_preds - val_targets) ** 2))

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break

    if best_state:
        model.load_state_dict(best_state)

    # Evaluate on all splits
    model.eval()
    results = {}
    for split_name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        all_preds, all_targets = [], []
        with torch.no_grad():
            for x_batch, y_batch in loader:
                pred = model(x_batch)
                all_preds.append(pred.numpy())
                all_targets.append(y_batch.numpy())
        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_targets)
        results[split_name] = compute_regression_metrics(y_true, y_pred)
        results[f"{split_name}_predictions"] = y_pred
        results[f"{split_name}_targets"] = y_true

    return results


# ---------------------------------------------------------------------------
# EditRNA-A3A: build samples for rate-only training
# ---------------------------------------------------------------------------

def build_rate_samples(df, sequences, structure_delta, window_size=100):
    """Build APOBECSiteSample list from a DataFrame of positive sites with rates.

    For rate regression, all samples are positive (label=1) with valid editing_rate.
    """
    samples = []
    for _, row in df.iterrows():
        site_id = str(row["site_id"])

        seq = sequences.get(site_id, "A" * (window_size * 2 + 1))
        edit_pos = min(window_size, len(seq) // 2)
        seq_list = list(seq)
        seq_list[edit_pos] = "C"
        seq = "".join(seq_list)

        flanking = get_flanking_context(seq, edit_pos)

        struct_d = structure_delta.get(site_id)
        if struct_d is not None:
            struct_d = np.array(struct_d, dtype=np.float32)

        log2_rate = float(row["log2_rate"])

        concordance = np.zeros(5, dtype=np.float32)

        sample = APOBECSiteSample(
            sequence=seq, edit_pos=edit_pos,
            is_edited=1.0,
            editing_rate_log2=log2_rate,
            apobec_class=-1, structure_type=-1, tissue_spec_class=-1,
            n_tissues_log2=float("nan"), exonic_function=-1,
            conservation=float("nan"), cancer_survival=float("nan"),
            tissue_rates=np.full(N_TISSUES, np.nan, dtype=np.float32),
            hek293_rate=float("nan"),
            flanking_context=flanking, concordance_features=concordance,
            structure_delta=struct_d, site_id=site_id,
            chrom=str(row.get("chr", "")),
            position=int(row.get("start", 0)),
            gene=str(row.get("gene", "")),
        )
        samples.append(sample)
    return samples


def batch_to_device(batch, device):
    """Move a batch dict (from apobec_collate_fn) to the specified device."""
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


def train_eval_editrna(train_df, val_df, test_df, sequences, structure_delta,
                       tokens_orig, pooled_orig, tokens_edited, pooled_edited,
                       epochs=60, lr=1e-4, patience=15, seed=42):
    """Train EditRNA-A3A with multi-task loss and evaluate rate regression.

    The model is trained with the full multi-task loss (binary + rate + others),
    but only positive sites with valid rates are used. Binary labels are all 1.
    Early stopping is based on validation rate MSE.

    Returns dict with train/val/test regression metrics.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")

    if len(train_df) < 10 or len(val_df) < 5:
        empty = {"spearman": float("nan"), "pearson": float("nan"),
                 "mse": float("nan"), "r2": float("nan"), "n_samples": 0}
        return {"train": empty, "val": empty, "test": empty}

    data_config = APOBECDataConfig(window_size=100)
    train_ds = APOBECDataset(build_rate_samples(train_df, sequences, structure_delta), data_config)
    val_ds = APOBECDataset(build_rate_samples(val_df, sequences, structure_delta), data_config)
    test_ds = APOBECDataset(build_rate_samples(test_df, sequences, structure_delta), data_config)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                              collate_fn=apobec_collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False,
                            collate_fn=apobec_collate_fn, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False,
                             collate_fn=apobec_collate_fn, num_workers=0)

    cached_encoder = CachedRNAEncoder(
        tokens_cache=tokens_orig, pooled_cache=pooled_orig,
        tokens_edited_cache=tokens_edited, pooled_edited_cache=pooled_edited,
        d_model=640,
    )
    config = EditRNAConfig(
        primary_encoder="cached", d_model=640, d_edit=256, d_fused=512,
        edit_n_heads=8, use_structure_delta=True,
        head_dropout=0.2, fusion_dropout=0.2,
        focal_gamma=2.0, focal_alpha_binary=0.75,
    )
    model = EditRNA_A3A(config=config, primary_encoder=cached_encoder).to(device)
    optimizer = AdamW(model.get_parameter_groups(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            batch = batch_to_device(batch, device)
            optimizer.zero_grad()
            output = model(batch)
            loss, _ = model.compute_loss(output, batch["targets"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # Validate on rate MSE
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch_to_device(batch, device)
                output = model(batch)
                rate_pred = output["predictions"]["editing_rate"].squeeze(-1).cpu().numpy()
                rate_target = batch["targets"]["rate"].cpu().numpy()
                mask = np.isfinite(rate_target)
                if mask.any():
                    val_preds.append(rate_pred[mask])
                    val_targets.append(rate_target[mask])

        if val_preds:
            val_preds_arr = np.concatenate(val_preds)
            val_targets_arr = np.concatenate(val_targets)
            val_loss = float(np.mean((val_preds_arr - val_targets_arr) ** 2))
        else:
            val_loss = float("inf")

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break

    if best_state:
        model.load_state_dict(best_state)

    # Evaluate on all splits
    model.eval()
    results = {}
    for split_name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in loader:
                batch = batch_to_device(batch, device)
                output = model(batch)
                rate_pred = output["predictions"]["editing_rate"].squeeze(-1).cpu().numpy()
                rate_target = batch["targets"]["rate"].cpu().numpy()
                mask = np.isfinite(rate_target)
                if mask.any():
                    all_preds.append(rate_pred[mask])
                    all_targets.append(rate_target[mask])

        if all_preds:
            y_pred = np.concatenate(all_preds)
            y_true = np.concatenate(all_targets)
            results[split_name] = compute_regression_metrics(y_true, y_pred)
            results[f"{split_name}_predictions"] = y_pred
            results[f"{split_name}_targets"] = y_true
        else:
            results[split_name] = {
                "spearman": float("nan"), "pearson": float("nan"),
                "mse": float("nan"), "r2": float("nan"), "n_samples": 0,
            }

    return results


# ---------------------------------------------------------------------------
# Per-dataset evaluation
# ---------------------------------------------------------------------------

def evaluate_per_dataset(test_df, test_preds, test_targets, datasets):
    """Compute metrics per dataset_source on the test set.

    Parameters
    ----------
    test_df : DataFrame with 'dataset_source' column (aligned with preds/targets)
    test_preds : numpy array of predictions (same length as test_df)
    test_targets : numpy array of targets
    datasets : list of dataset_source values to evaluate

    Returns dict of dataset -> metrics.
    """
    results = {}
    ds_values = test_df["dataset_source"].values
    for ds in datasets:
        mask = ds_values == ds
        if mask.sum() < 3:
            results[ds] = {
                "spearman": float("nan"), "pearson": float("nan"),
                "mse": float("nan"), "r2": float("nan"),
                "n_samples": int(mask.sum()),
            }
        else:
            y_true = test_targets[mask]
            y_pred = test_preds[mask]
            results[ds] = compute_regression_metrics(y_true, y_pred)
    return results


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_predictions_scatter(all_results, output_dir):
    """Generate scatter plots of predicted vs actual rates for each model on test set."""
    models = list(all_results.keys())
    n_models = len(models)

    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    for ax, model_name in zip(axes, models):
        res = all_results[model_name]
        if "test_predictions" not in res or "test_targets" not in res:
            ax.set_title(f"{model_name}\n(no data)")
            continue

        y_pred = res["test_predictions"]
        y_true = res["test_targets"]
        metrics = res["test"]

        ax.scatter(y_true, y_pred, alpha=0.3, s=10, color="steelblue")

        # Diagonal reference line
        lims = [
            min(y_true.min(), y_pred.min()) - 0.2,
            max(y_true.max(), y_pred.max()) + 0.2,
        ]
        ax.plot(lims, lims, "r--", alpha=0.7, linewidth=1)

        ax.set_xlabel("True log2(rate + 0.01)", fontsize=10)
        ax.set_ylabel("Predicted log2(rate + 0.01)", fontsize=10)
        ax.set_title(
            f"{model_name}\n"
            f"Spearman={metrics['spearman']:.3f}  Pearson={metrics['pearson']:.3f}\n"
            f"MSE={metrics['mse']:.3f}  R2={metrics['r2']:.3f}  n={metrics['n_samples']}",
            fontsize=9,
        )
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    plt.suptitle("Editing Rate Prediction: Predicted vs Actual (Test Set)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "rate_prediction_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_per_dataset_bars(per_dataset_results, output_dir):
    """Generate grouped bar chart of per-dataset Spearman correlation for each model."""
    models = list(per_dataset_results.keys())
    datasets = list(per_dataset_results[models[0]].keys())

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(datasets))
    width = 0.8 / len(models)
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))

    for i, model_name in enumerate(models):
        vals = []
        for ds in datasets:
            v = per_dataset_results[model_name].get(ds, {}).get("spearman", float("nan"))
            vals.append(v if np.isfinite(v) else 0.0)
        bars = ax.bar(x + i * width, vals, width, label=model_name, color=colors[i])
        # Annotate bars
        raw_vals = [per_dataset_results[model_name].get(ds, {}).get("spearman", float("nan"))
                    for ds in datasets]
        for bar, v_orig in zip(bars, raw_vals):
            if np.isfinite(v_orig):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{v_orig:.3f}", ha="center", va="bottom", fontsize=7, rotation=45)

    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels([DATASET_LABELS.get(ds, ds) for ds in datasets], fontsize=10)
    ax.set_ylabel("Spearman Correlation", fontsize=11)
    ax.set_title("Per-Dataset Rate Prediction (Test Set)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(bottom=-0.1)

    plt.tight_layout()
    plt.savefig(output_dir / "rate_prediction_per_dataset.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_model_comparison_bars(all_results, output_dir):
    """Generate bar chart comparing models on test-set metrics."""
    models = list(all_results.keys())
    metric_names = ["spearman", "pearson", "r2"]
    metric_labels = ["Spearman rho", "Pearson r", "R^2"]

    fig, axes = plt.subplots(1, len(metric_names), figsize=(5 * len(metric_names), 5))

    x = np.arange(len(models))
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))

    for ax, metric, label in zip(axes, metric_names, metric_labels):
        vals = []
        for m in models:
            v = all_results[m].get("test", {}).get(metric, float("nan"))
            vals.append(v if np.isfinite(v) else 0.0)

        bars = ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=9, rotation=20, ha="right")
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Model Comparison: Editing Rate Prediction (Test Set)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "rate_prediction_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)
    torch.manual_seed(42)

    t_global = time.time()

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    logger.info("Loading embeddings...")
    pooled_orig = torch.load(EMB_DIR / "rnafm_pooled.pt", weights_only=False)
    pooled_edited = torch.load(EMB_DIR / "rnafm_pooled_edited.pt", weights_only=False)
    tokens_path = EMB_DIR / "rnafm_tokens.pt"
    tokens_edited_path = EMB_DIR / "rnafm_tokens_edited.pt"
    has_tokens = tokens_path.exists() and tokens_edited_path.exists()
    if has_tokens:
        tokens_orig = torch.load(tokens_path, weights_only=False)
        tokens_edited = torch.load(tokens_edited_path, weights_only=False)
    else:
        tokens_orig = tokens_edited = None
        logger.warning("Token embeddings not found — EditRNA-A3A will be skipped")

    splits_df = pd.read_csv(SPLITS_CSV)
    logger.info("Loaded %d sites from splits_expanded.csv", len(splits_df))

    # Load sequences and structure
    sequences = {}
    if SEQ_JSON.exists():
        sequences = json.loads(SEQ_JSON.read_text())
        logger.info("Loaded %d sequences", len(sequences))

    structure_delta = {}
    if STRUCT_CACHE.exists():
        data = np.load(STRUCT_CACHE, allow_pickle=True)
        feat_key = "delta_features" if "delta_features" in data else "features"
        if "site_ids" in data and feat_key in data:
            sids = data["site_ids"]
            feats = data[feat_key]
            structure_delta = {str(sid): feats[i] for i, sid in enumerate(sids)}
        logger.info("Loaded %d structure deltas", len(structure_delta))

    # ------------------------------------------------------------------
    # Filter to positive sites with valid editing_rate and embeddings
    # ------------------------------------------------------------------
    available = set(pooled_orig.keys()) & set(pooled_edited.keys())
    rate_df = splits_df[
        (splits_df["label"] == 1) &
        (splits_df["editing_rate"].notna()) &
        (splits_df["site_id"].isin(available))
    ].copy()

    # Compute log2 target
    rate_df["log2_rate"] = rate_df["editing_rate"].apply(compute_log2_rate)
    rate_df = rate_df[rate_df["log2_rate"].notna()].copy()

    logger.info("Sites with valid rate + embeddings: %d", len(rate_df))

    train_df = rate_df[rate_df["split"] == "train"].copy()
    val_df = rate_df[rate_df["split"] == "val"].copy()
    test_df = rate_df[rate_df["split"] == "test"].copy()
    logger.info("Train: %d, Val: %d, Test: %d", len(train_df), len(val_df), len(test_df))

    # Log target distribution
    logger.info("Target log2(rate+0.01) stats:")
    logger.info("  Train: mean=%.3f, std=%.3f", train_df["log2_rate"].mean(), train_df["log2_rate"].std())
    logger.info("  Val:   mean=%.3f, std=%.3f", val_df["log2_rate"].mean(), val_df["log2_rate"].std())
    logger.info("  Test:  mean=%.3f, std=%.3f", test_df["log2_rate"].mean(), test_df["log2_rate"].std())

    # Per-dataset sample counts
    logger.info("\nPer-dataset test samples:")
    for ds in DATASET_SOURCES:
        n = (test_df["dataset_source"] == ds).sum()
        logger.info("  %s (%s): %d", DATASET_LABELS[ds], ds, n)

    # ------------------------------------------------------------------
    # Model 1: SubtractionMLP
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("Model 1: SubtractionMLP  (pooled_edited - pooled_orig -> MLP)")
    logger.info("=" * 70)
    t0 = time.time()
    results_sub = train_eval_mlp(
        train_df, val_df, test_df, pooled_orig, pooled_edited,
        mode="subtraction", d_model=640, epochs=80, lr=1e-3, patience=15, seed=42,
    )
    logger.info("  Time: %.1fs", time.time() - t0)
    logger.info("  Test: Spearman=%.4f  Pearson=%.4f  MSE=%.4f  R2=%.4f",
                results_sub["test"]["spearman"], results_sub["test"]["pearson"],
                results_sub["test"]["mse"], results_sub["test"]["r2"])

    # ------------------------------------------------------------------
    # Model 2: ConcatMLP
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("Model 2: ConcatMLP  ([pooled_orig; pooled_edited] -> MLP)")
    logger.info("=" * 70)
    t0 = time.time()
    results_concat = train_eval_mlp(
        train_df, val_df, test_df, pooled_orig, pooled_edited,
        mode="concat", d_model=640, epochs=80, lr=1e-3, patience=15, seed=42,
    )
    logger.info("  Time: %.1fs", time.time() - t0)
    logger.info("  Test: Spearman=%.4f  Pearson=%.4f  MSE=%.4f  R2=%.4f",
                results_concat["test"]["spearman"], results_concat["test"]["pearson"],
                results_concat["test"]["mse"], results_concat["test"]["r2"])

    # ------------------------------------------------------------------
    # Model 3: OrigOnlyMLP (baseline)
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("Model 3: OrigOnlyMLP  (pooled_orig -> MLP, NO edit info)")
    logger.info("=" * 70)
    t0 = time.time()
    results_orig = train_eval_mlp(
        train_df, val_df, test_df, pooled_orig, pooled_edited,
        mode="orig_only", d_model=640, epochs=80, lr=1e-3, patience=15, seed=42,
    )
    logger.info("  Time: %.1fs", time.time() - t0)
    logger.info("  Test: Spearman=%.4f  Pearson=%.4f  MSE=%.4f  R2=%.4f",
                results_orig["test"]["spearman"], results_orig["test"]["pearson"],
                results_orig["test"]["mse"], results_orig["test"]["r2"])

    # ------------------------------------------------------------------
    # Model 4: EditRNA-A3A
    # ------------------------------------------------------------------
    if has_tokens:
        logger.info("\n" + "=" * 70)
        logger.info("Model 4: EditRNA-A3A  (full multi-task, rate head)")
        logger.info("=" * 70)
        t0 = time.time()
        results_editrna = train_eval_editrna(
            train_df, val_df, test_df, sequences, structure_delta,
            tokens_orig, pooled_orig, tokens_edited, pooled_edited,
            epochs=60, lr=1e-4, patience=15, seed=42,
        )
        logger.info("  Time: %.1fs", time.time() - t0)
        logger.info("  Test: Spearman=%.4f  Pearson=%.4f  MSE=%.4f  R2=%.4f",
                    results_editrna["test"]["spearman"], results_editrna["test"]["pearson"],
                    results_editrna["test"]["mse"], results_editrna["test"]["r2"])
    else:
        logger.info("\n" + "=" * 70)
        logger.info("Model 4: EditRNA-A3A  — SKIPPED (no token embeddings)")
        logger.info("=" * 70)
        results_editrna = {"test": {"spearman": float("nan"), "pearson": float("nan"),
                                     "mse": float("nan"), "r2": float("nan")}}

    total_time = time.time() - t_global
    logger.info("\nTotal experiment time: %.1fs", total_time)

    # ------------------------------------------------------------------
    # Aggregate results
    # ------------------------------------------------------------------
    all_results = {
        "SubtractionMLP": results_sub,
        "ConcatMLP": results_concat,
        "OrigOnlyMLP": results_orig,
        "EditRNA-A3A": results_editrna,
    }

    # ------------------------------------------------------------------
    # Per-dataset evaluation
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("Per-Dataset Evaluation (Test Set)")
    logger.info("=" * 70)

    per_dataset_results = {}
    for model_name, res in all_results.items():
        if "test_predictions" in res and "test_targets" in res:
            per_ds = evaluate_per_dataset(
                test_df, res["test_predictions"], res["test_targets"], DATASET_SOURCES
            )
            per_dataset_results[model_name] = per_ds
            logger.info("\n  %s:", model_name)
            for ds in DATASET_SOURCES:
                m = per_ds[ds]
                logger.info("    %-12s  Spearman=%.4f  Pearson=%.4f  MSE=%.4f  R2=%.4f  n=%d",
                            DATASET_LABELS[ds], m["spearman"], m["pearson"],
                            m["mse"], m["r2"], m["n_samples"])

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 95)
    print("EDITING RATE PREDICTION - MODEL COMPARISON (TEST SET)")
    print("=" * 95)
    header = f"{'Model':<20}{'Spearman':>12}{'Pearson':>12}{'MSE':>12}{'R2':>12}{'N':>10}"
    print(header)
    print("-" * 95)
    for model_name in all_results:
        m = all_results[model_name].get("test", {})
        sp = m.get("spearman", float("nan"))
        pe = m.get("pearson", float("nan"))
        mse = m.get("mse", float("nan"))
        r2 = m.get("r2", float("nan"))
        n = m.get("n_samples", 0)
        print(f"{model_name:<20}{sp:>12.4f}{pe:>12.4f}{mse:>12.4f}{r2:>12.4f}{n:>10d}")
    print("=" * 95)

    # Per-dataset table
    print("\n" + "=" * 95)
    print("PER-DATASET SPEARMAN CORRELATION (TEST SET)")
    print("=" * 95)
    ds_labels = [DATASET_LABELS[ds] for ds in DATASET_SOURCES]
    header = f"{'Model':<20}" + "".join(f"{dl:>15}" for dl in ds_labels)
    print(header)
    print("-" * 95)
    for model_name in per_dataset_results:
        row = f"{model_name:<20}"
        for ds in DATASET_SOURCES:
            v = per_dataset_results[model_name].get(ds, {}).get("spearman", float("nan"))
            if np.isnan(v):
                row += f"{'N/A':>15}"
            else:
                row += f"{v:>15.4f}"
        print(row)
    print("=" * 95)

    # ------------------------------------------------------------------
    # Generate plots
    # ------------------------------------------------------------------
    logger.info("\nGenerating plots...")

    plot_predictions_scatter(all_results, OUTPUT_DIR)
    logger.info("  Saved scatter plot to %s", OUTPUT_DIR / "rate_prediction_scatter.png")

    plot_model_comparison_bars(all_results, OUTPUT_DIR)
    logger.info("  Saved model comparison to %s", OUTPUT_DIR / "rate_prediction_model_comparison.png")

    if per_dataset_results:
        plot_per_dataset_bars(per_dataset_results, OUTPUT_DIR)
        logger.info("  Saved per-dataset bars to %s", OUTPUT_DIR / "rate_prediction_per_dataset.png")

    # ------------------------------------------------------------------
    # Save results JSON
    # ------------------------------------------------------------------
    def serialize(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return str(obj)

    # Build JSON-safe results (exclude large numpy arrays)
    json_results = {
        "description": (
            "Editing rate prediction (regression) evaluation. "
            "Target: log2(editing_rate + 0.01) after normalizing rates to [0,1]. "
            "Models trained on positive sites with valid editing_rate only."
        ),
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "target_stats": {
            "train_mean": float(train_df["log2_rate"].mean()),
            "train_std": float(train_df["log2_rate"].std()),
            "test_mean": float(test_df["log2_rate"].mean()),
            "test_std": float(test_df["log2_rate"].std()),
        },
        "models": {},
        "per_dataset": {},
        "total_time_seconds": total_time,
    }

    for model_name in all_results:
        json_results["models"][model_name] = {}
        for split in ["train", "val", "test"]:
            metrics = all_results[model_name].get(split, {})
            # Keep only scalar metrics, not numpy arrays
            json_results["models"][model_name][split] = {
                k: v for k, v in metrics.items()
                if not isinstance(v, np.ndarray)
            }

    for model_name in per_dataset_results:
        json_results["per_dataset"][model_name] = per_dataset_results[model_name]

    with open(OUTPUT_DIR / "rate_prediction_results.json", "w") as f:
        json.dump(json_results, f, indent=2, default=serialize)
    logger.info("Saved results to %s", OUTPUT_DIR / "rate_prediction_results.json")

    # Save test predictions as .npz for further analysis
    pred_arrays = {}
    pred_arrays["site_ids"] = test_df["site_id"].values
    pred_arrays["dataset_sources"] = test_df["dataset_source"].values
    pred_arrays["true_log2_rates"] = test_df["log2_rate"].values
    for model_name in all_results:
        key = model_name.lower().replace("-", "_").replace(" ", "_")
        if "test_predictions" in all_results[model_name]:
            pred_arrays[f"pred_{key}"] = all_results[model_name]["test_predictions"]
    np.savez(OUTPUT_DIR / "rate_prediction_test_predictions.npz", **pred_arrays)
    logger.info("Saved test predictions to %s", OUTPUT_DIR / "rate_prediction_test_predictions.npz")

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
