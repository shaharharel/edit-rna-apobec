#!/usr/bin/env python
"""Editing RATE prediction with ALL 7 DL architectures + per-dataset cross-training.

Extends exp_rate_prediction.py to evaluate ALL architectures for rate regression:
  1. PooledMLP:       pooled_orig -> MLP  (baseline, no edit info)
  2. SubtractionMLP:  (pooled_edited - pooled_orig) -> MLP
  3. ConcatMLP:       [pooled_orig; pooled_edited] -> MLP
  4. CrossAttention:  Q=tokens_orig, K,V=tokens_edited -> pool -> MLP  (token-level)
  5. DiffAttention:   (tokens_edited - tokens_orig) -> Transformer -> pool -> MLP  (token-level)
  6. StructureOnly:   structure_delta (7-dim) -> MLP
  7. EditRNA-A3A:     Full multi-task model, rate head output

Also adds:
  - Per-dataset cross-training (train on each dataset, evaluate on all)
  - Save final rate predictions for all sites (for ClinVar scoring)

Metrics: Spearman rho, Pearson r, MSE, R^2
Target:  log2(editing_rate + 0.01) after normalizing rates to [0,1]

Usage:
    python experiments/apobec3a/exp_rate_all_architectures.py
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

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

DATASET_SOURCES = ["advisor_c2t", "alqassim_2021", "sharma_2015", "baysal_2016"]
DATASET_LABELS = {
    "advisor_c2t": "Levanon",
    "alqassim_2021": "Alqassim",
    "sharma_2015": "Sharma",
    "baysal_2016": "Baysal",
}


# ---------------------------------------------------------------------------
# Target transformation & metrics
# ---------------------------------------------------------------------------

def compute_log2_rate(editing_rate):
    if pd.isna(editing_rate) or editing_rate < 0:
        return float("nan")
    rate = float(editing_rate)
    if rate > 1.0:
        rate = rate / 100.0
    return np.log2(rate + 0.01)


def compute_regression_metrics(y_true, y_pred):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) < 3:
        return {"spearman": float("nan"), "pearson": float("nan"),
                "mse": float("nan"), "r2": float("nan"), "n_samples": 0}
    sp_rho, _ = spearmanr(y_true, y_pred)
    pe_r, _ = pearsonr(y_true, y_pred)
    mse = float(mean_squared_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"spearman": float(sp_rho), "pearson": float(pe_r),
            "mse": mse, "r2": r2, "n_samples": int(len(y_true))}


# ---------------------------------------------------------------------------
# Datasets for different input types
# ---------------------------------------------------------------------------

class PooledRateDataset(Dataset):
    """Rate regression from pre-computed pooled embeddings."""

    def __init__(self, site_ids, targets, pooled_orig, pooled_edited, mode="subtraction"):
        self.site_ids = site_ids
        self.targets = targets
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


class TokenRateDataset(Dataset):
    """Rate regression from token-level embeddings (for CrossAttention/DiffAttention)."""

    def __init__(self, site_ids, targets, tokens_orig, tokens_edited):
        self.site_ids = site_ids
        self.targets = targets
        self.tokens_orig = tokens_orig
        self.tokens_edited = tokens_edited

    def __len__(self):
        return len(self.site_ids)

    def __getitem__(self, idx):
        sid = self.site_ids[idx]
        t_orig = self.tokens_orig[sid]
        t_edited = self.tokens_edited[sid]
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        return {"tokens_orig": t_orig, "tokens_edited": t_edited}, target


def token_collate_fn(batch):
    """Collate function for token-level datasets."""
    items, targets = zip(*batch)
    max_len = max(item["tokens_orig"].shape[0] for item in items)
    d_model = items[0]["tokens_orig"].shape[1]

    tokens_orig = torch.zeros(len(items), max_len, d_model)
    tokens_edited = torch.zeros(len(items), max_len, d_model)

    for i, item in enumerate(items):
        L = item["tokens_orig"].shape[0]
        tokens_orig[i, :L] = item["tokens_orig"]
        tokens_edited[i, :L] = item["tokens_edited"]

    return {"tokens_orig": tokens_orig, "tokens_edited": tokens_edited}, torch.stack(targets)


class StructureRateDataset(Dataset):
    """Rate regression from structure delta features only."""

    def __init__(self, site_ids, targets, structure_delta):
        self.site_ids = site_ids
        self.targets = targets
        self.structure_delta = structure_delta

    def __len__(self):
        return len(self.site_ids)

    def __getitem__(self, idx):
        sid = self.site_ids[idx]
        struct = self.structure_delta.get(sid)
        if struct is not None:
            x = torch.tensor(struct, dtype=torch.float32)
        else:
            x = torch.zeros(7, dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        return x, target


# ---------------------------------------------------------------------------
# MLP regression model (for pooled and structure)
# ---------------------------------------------------------------------------

class RateRegressionMLP(nn.Module):
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
# Token-level rate regression models (adapted from classification baselines)
# ---------------------------------------------------------------------------

class CrossAttentionRate(nn.Module):
    """CrossAttention adapted for rate regression."""

    def __init__(self, d_model=640, n_heads=8, d_hidden=256, dropout=0.3):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, batch):
        tokens_orig = batch["tokens_orig"]
        tokens_edited = batch["tokens_edited"]
        attended, _ = self.cross_attn(query=tokens_orig, key=tokens_edited, value=tokens_edited)
        x = self.norm(tokens_orig + attended)
        pooled = x.mean(dim=1)
        return self.mlp(pooled).squeeze(-1)


class DiffAttentionRate(nn.Module):
    """DiffAttention adapted for rate regression."""

    def __init__(self, d_model=640, n_heads=8, n_layers=2, d_hidden=256, dropout=0.3):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, batch):
        diff = batch["tokens_edited"] - batch["tokens_orig"]
        encoded = self.transformer(diff)
        pooled = encoded.mean(dim=1)
        return self.mlp(pooled).squeeze(-1)


# ---------------------------------------------------------------------------
# Generic training / evaluation
# ---------------------------------------------------------------------------

def train_eval_generic(model, train_loader, val_loader, test_loader,
                       is_token_model=False, epochs=80, lr=1e-3,
                       patience=15, seed=42):
    """Train and evaluate a generic rate regression model."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        for batch_data, batch_target in train_loader:
            optimizer.zero_grad()
            if is_token_model:
                pred = model(batch_data)
            else:
                pred = model(batch_data)
            loss = F.mse_loss(pred, batch_target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # Validate
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch_data, batch_target in val_loader:
                pred = model(batch_data)
                val_preds.append(pred.numpy())
                val_targets.append(batch_target.numpy())
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

    # Evaluate all splits
    model.eval()
    results = {}
    for split_name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch_data, batch_target in loader:
                pred = model(batch_data)
                all_preds.append(pred.numpy())
                all_targets.append(batch_target.numpy())
        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_targets)
        results[split_name] = compute_regression_metrics(y_true, y_pred)
        results[f"{split_name}_predictions"] = y_pred
        results[f"{split_name}_targets"] = y_true

    return results


# ---------------------------------------------------------------------------
# EditRNA-A3A training (from exp_rate_prediction.py)
# ---------------------------------------------------------------------------

def build_rate_samples(df, sequences, structure_delta, window_size=100):
    """Build APOBECSiteSample list for rate regression (positive sites only)."""
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
            is_edited=1.0, editing_rate_log2=log2_rate,
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
    """Train EditRNA-A3A with multi-task loss and evaluate rate regression."""
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
    results = {}
    ds_values = test_df["dataset_source"].values
    for ds in datasets:
        mask = ds_values == ds
        if mask.sum() < 3:
            results[ds] = {"spearman": float("nan"), "pearson": float("nan"),
                           "mse": float("nan"), "r2": float("nan"),
                           "n_samples": int(mask.sum())}
        else:
            results[ds] = compute_regression_metrics(test_targets[mask], test_preds[mask])
    return results


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_all_models_comparison(all_results, output_dir):
    """Generate comprehensive comparison of all models."""
    models = list(all_results.keys())
    n_models = len(models)

    # Scatter plots
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    for i, model_name in enumerate(models):
        ax = axes[i] if i < len(axes) else None
        if ax is None:
            break
        res = all_results[model_name]
        if "test_predictions" not in res or "test_targets" not in res:
            ax.set_title(f"{model_name}\n(no data)")
            continue
        y_pred = res["test_predictions"]
        y_true = res["test_targets"]
        metrics = res["test"]
        ax.scatter(y_true, y_pred, alpha=0.3, s=10, color="steelblue")
        lims = [min(y_true.min(), y_pred.min()) - 0.2,
                max(y_true.max(), y_pred.max()) + 0.2]
        ax.plot(lims, lims, "r--", alpha=0.7, linewidth=1)
        ax.set_xlabel("True log2(rate + 0.01)", fontsize=8)
        ax.set_ylabel("Predicted", fontsize=8)
        ax.set_title(f"{model_name}\nSp={metrics['spearman']:.3f} Pe={metrics['pearson']:.3f}",
                     fontsize=9)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for j in range(n_models, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Rate Prediction: All 7 Architectures (Test Set)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "rate_all_architectures_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Bar chart comparison
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    metric_names = ["spearman", "pearson", "r2"]
    metric_labels = ["Spearman rho", "Pearson r", "R^2"]
    x = np.arange(n_models)
    colors = plt.cm.Set2(np.linspace(0, 1, n_models))

    for ax, metric, label in zip(axes, metric_names, metric_labels):
        vals = [all_results[m].get("test", {}).get(metric, float("nan")) for m in models]
        vals = [v if np.isfinite(v) else 0.0 for v in vals]
        bars = ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=7, rotation=35, ha="right")
        ax.set_ylabel(label, fontsize=10)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Rate Prediction: Model Comparison (Test Set)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "rate_all_architectures_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_per_dataset_heatmap(per_dataset_results, output_dir):
    """Generate heatmap of per-dataset Spearman for all models."""
    models = list(per_dataset_results.keys())
    datasets = DATASET_SOURCES

    data = np.zeros((len(models), len(datasets)))
    for i, m in enumerate(models):
        for j, ds in enumerate(datasets):
            v = per_dataset_results[m].get(ds, {}).get("spearman", float("nan"))
            data[i, j] = v if np.isfinite(v) else 0.0

    fig, ax = plt.subplots(figsize=(10, max(4, len(models) * 0.6)))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=-0.2, vmax=0.8)
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels([DATASET_LABELS[ds] for ds in datasets], fontsize=10)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=9)

    for i in range(len(models)):
        for j in range(len(datasets)):
            ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center", fontsize=8,
                    color="white" if data[i, j] < 0.1 else "black")

    plt.colorbar(im, ax=ax, label="Spearman rho", shrink=0.8)
    ax.set_title("Per-Dataset Rate Prediction (Spearman) - All Architectures",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "rate_per_dataset_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_cross_dataset_rate(cross_results, output_dir):
    """Plot cross-dataset training results."""
    train_datasets = list(cross_results.keys())
    test_datasets = DATASET_SOURCES

    data = np.zeros((len(train_datasets), len(test_datasets)))
    for i, td in enumerate(train_datasets):
        for j, te in enumerate(test_datasets):
            v = cross_results[td].get(te, {}).get("spearman", float("nan"))
            data[i, j] = v if np.isfinite(v) else 0.0

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=-0.3, vmax=0.7)
    ax.set_xticks(range(len(test_datasets)))
    ax.set_xticklabels([DATASET_LABELS[ds] for ds in test_datasets], fontsize=10)
    ax.set_yticks(range(len(train_datasets)))
    ax.set_yticklabels([DATASET_LABELS.get(ds, ds) for ds in train_datasets], fontsize=10)
    ax.set_xlabel("Test Dataset", fontsize=11)
    ax.set_ylabel("Train Dataset", fontsize=11)

    for i in range(len(train_datasets)):
        for j in range(len(test_datasets)):
            ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center", fontsize=9,
                    color="white" if data[i, j] < 0.1 else "black")

    plt.colorbar(im, ax=ax, label="Spearman rho", shrink=0.8)
    ax.set_title("Cross-Dataset Rate Prediction (SubtractionMLP)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "rate_cross_dataset_matrix.png", dpi=150, bbox_inches="tight")
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
        logger.warning("Token embeddings not found — token-dependent models will be skipped")

    splits_df = pd.read_csv(SPLITS_CSV)
    logger.info("Loaded %d sites from splits_expanded.csv", len(splits_df))

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
    # Filter to positive sites with valid editing_rate
    # ------------------------------------------------------------------
    available = set(pooled_orig.keys()) & set(pooled_edited.keys())
    rate_df = splits_df[
        (splits_df["label"] == 1) &
        (splits_df["editing_rate"].notna()) &
        (splits_df["site_id"].isin(available))
    ].copy()
    rate_df["log2_rate"] = rate_df["editing_rate"].apply(compute_log2_rate)
    rate_df = rate_df[rate_df["log2_rate"].notna()].copy()

    logger.info("Sites with valid rate + embeddings: %d", len(rate_df))

    train_df = rate_df[rate_df["split"] == "train"].copy()
    val_df = rate_df[rate_df["split"] == "val"].copy()
    test_df = rate_df[rate_df["split"] == "test"].copy()
    logger.info("Train: %d, Val: %d, Test: %d", len(train_df), len(val_df), len(test_df))

    # ------------------------------------------------------------------
    # Model 1: PooledMLP (orig only, no edit info)
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("Model 1: PooledMLP (pooled_orig -> MLP, NO edit info)")
    logger.info("=" * 70)
    t0 = time.time()

    def make_pooled_loaders(df_train, df_val, df_test, mode):
        tr = DataLoader(PooledRateDataset(df_train["site_id"].tolist(), df_train["log2_rate"].values.astype(np.float32),
                                          pooled_orig, pooled_edited, mode), batch_size=64, shuffle=True, num_workers=0)
        va = DataLoader(PooledRateDataset(df_val["site_id"].tolist(), df_val["log2_rate"].values.astype(np.float32),
                                          pooled_orig, pooled_edited, mode), batch_size=64, shuffle=False, num_workers=0)
        te = DataLoader(PooledRateDataset(df_test["site_id"].tolist(), df_test["log2_rate"].values.astype(np.float32),
                                          pooled_orig, pooled_edited, mode), batch_size=64, shuffle=False, num_workers=0)
        return tr, va, te

    loaders = make_pooled_loaders(train_df, val_df, test_df, "orig_only")
    model = RateRegressionMLP(d_input=640)
    results_pooled = train_eval_generic(model, *loaders, epochs=80, lr=1e-3, patience=15)
    logger.info("  Time: %.1fs  Test Spearman=%.4f", time.time() - t0, results_pooled["test"]["spearman"])

    # ------------------------------------------------------------------
    # Model 2: SubtractionMLP
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("Model 2: SubtractionMLP (pooled_edited - pooled_orig -> MLP)")
    logger.info("=" * 70)
    t0 = time.time()
    loaders = make_pooled_loaders(train_df, val_df, test_df, "subtraction")
    model = RateRegressionMLP(d_input=640)
    results_sub = train_eval_generic(model, *loaders, epochs=80, lr=1e-3, patience=15)
    logger.info("  Time: %.1fs  Test Spearman=%.4f", time.time() - t0, results_sub["test"]["spearman"])

    # ------------------------------------------------------------------
    # Model 3: ConcatMLP
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("Model 3: ConcatMLP ([pooled_orig; pooled_edited] -> MLP)")
    logger.info("=" * 70)
    t0 = time.time()
    loaders = make_pooled_loaders(train_df, val_df, test_df, "concat")
    model = RateRegressionMLP(d_input=1280)
    results_concat = train_eval_generic(model, *loaders, epochs=80, lr=1e-3, patience=15)
    logger.info("  Time: %.1fs  Test Spearman=%.4f", time.time() - t0, results_concat["test"]["spearman"])

    # ------------------------------------------------------------------
    # Model 4: CrossAttention (token-level)
    # ------------------------------------------------------------------
    nan_result = {"test": {"spearman": float("nan"), "pearson": float("nan"),
                           "mse": float("nan"), "r2": float("nan")}}
    if has_tokens:
        logger.info("\n" + "=" * 70)
        logger.info("Model 4: CrossAttention (Q=orig, K,V=edited tokens -> pool -> MLP)")
        logger.info("=" * 70)
        t0 = time.time()

        # Token-level available sites
        token_available = set(tokens_orig.keys()) & set(tokens_edited.keys())
        train_tok = train_df[train_df["site_id"].isin(token_available)]
        val_tok = val_df[val_df["site_id"].isin(token_available)]
        test_tok = test_df[test_df["site_id"].isin(token_available)]

        tr_tok = DataLoader(TokenRateDataset(train_tok["site_id"].tolist(), train_tok["log2_rate"].values.astype(np.float32),
                                             tokens_orig, tokens_edited), batch_size=16, shuffle=True,
                            collate_fn=token_collate_fn, num_workers=0)
        va_tok = DataLoader(TokenRateDataset(val_tok["site_id"].tolist(), val_tok["log2_rate"].values.astype(np.float32),
                                             tokens_orig, tokens_edited), batch_size=32, shuffle=False,
                            collate_fn=token_collate_fn, num_workers=0)
        te_tok = DataLoader(TokenRateDataset(test_tok["site_id"].tolist(), test_tok["log2_rate"].values.astype(np.float32),
                                             tokens_orig, tokens_edited), batch_size=32, shuffle=False,
                            collate_fn=token_collate_fn, num_workers=0)

        model = CrossAttentionRate()
        results_cross = train_eval_generic(model, tr_tok, va_tok, te_tok,
                                           is_token_model=True, epochs=60, lr=5e-4, patience=15)
        logger.info("  Time: %.1fs  Test Spearman=%.4f", time.time() - t0, results_cross["test"]["spearman"])
    else:
        logger.info("\n  Skipping Model 4: CrossAttention (no token embeddings)")
        results_cross = nan_result

    # ------------------------------------------------------------------
    # Model 5: DiffAttention (token-level)
    # ------------------------------------------------------------------
    if has_tokens:
        logger.info("\n" + "=" * 70)
        logger.info("Model 5: DiffAttention (token diff -> Transformer -> pool -> MLP)")
        logger.info("=" * 70)
        t0 = time.time()
        model = DiffAttentionRate()
        results_diff = train_eval_generic(model, tr_tok, va_tok, te_tok,
                                          is_token_model=True, epochs=60, lr=5e-4, patience=15)
        logger.info("  Time: %.1fs  Test Spearman=%.4f", time.time() - t0, results_diff["test"]["spearman"])
    else:
        logger.info("  Skipping Model 5: DiffAttention (no token embeddings)")
        results_diff = nan_result

    # ------------------------------------------------------------------
    # Model 6: StructureOnly
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("Model 6: StructureOnly (7-dim structure delta -> MLP)")
    logger.info("=" * 70)
    t0 = time.time()
    tr_struct = DataLoader(StructureRateDataset(train_df["site_id"].tolist(), train_df["log2_rate"].values.astype(np.float32),
                                                structure_delta), batch_size=64, shuffle=True, num_workers=0)
    va_struct = DataLoader(StructureRateDataset(val_df["site_id"].tolist(), val_df["log2_rate"].values.astype(np.float32),
                                                structure_delta), batch_size=64, shuffle=False, num_workers=0)
    te_struct = DataLoader(StructureRateDataset(test_df["site_id"].tolist(), test_df["log2_rate"].values.astype(np.float32),
                                                structure_delta), batch_size=64, shuffle=False, num_workers=0)
    model = RateRegressionMLP(d_input=7, hidden=64, dropout=0.3)
    results_struct = train_eval_generic(model, tr_struct, va_struct, te_struct,
                                        epochs=100, lr=5e-3, patience=20)
    logger.info("  Time: %.1fs  Test Spearman=%.4f", time.time() - t0, results_struct["test"]["spearman"])

    # ------------------------------------------------------------------
    # Model 7: EditRNA-A3A
    # ------------------------------------------------------------------
    if has_tokens:
        logger.info("\n" + "=" * 70)
        logger.info("Model 7: EditRNA-A3A (full multi-task, rate head)")
        logger.info("=" * 70)
        t0 = time.time()
        results_editrna = train_eval_editrna(
            train_df, val_df, test_df, sequences, structure_delta,
            tokens_orig, pooled_orig, tokens_edited, pooled_edited,
            epochs=60, lr=1e-4, patience=15, seed=42,
        )
        logger.info("  Time: %.1fs  Test Spearman=%.4f", time.time() - t0, results_editrna["test"]["spearman"])
    else:
        logger.info("\n  Skipping Model 7: EditRNA-A3A (no token embeddings)")
        results_editrna = nan_result

    # ------------------------------------------------------------------
    # Aggregate results
    # ------------------------------------------------------------------
    all_results = {
        "PooledMLP": results_pooled,
        "SubtractionMLP": results_sub,
        "ConcatMLP": results_concat,
        "CrossAttention": results_cross,
        "DiffAttention": results_diff,
        "StructureOnly": results_struct,
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
            # For token-level models, test_df may be subset
            if model_name in ("CrossAttention", "DiffAttention"):
                per_ds = evaluate_per_dataset(test_tok, res["test_predictions"],
                                              res["test_targets"], DATASET_SOURCES)
            else:
                per_ds = evaluate_per_dataset(test_df, res["test_predictions"],
                                              res["test_targets"], DATASET_SOURCES)
            per_dataset_results[model_name] = per_ds
            logger.info("\n  %s:", model_name)
            for ds in DATASET_SOURCES:
                m = per_ds[ds]
                logger.info("    %-12s  Spearman=%.4f  Pearson=%.4f  n=%d",
                            DATASET_LABELS[ds], m["spearman"], m["pearson"], m["n_samples"])

    # ------------------------------------------------------------------
    # Cross-dataset rate prediction (train on each, eval on all)
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("Cross-Dataset Rate Prediction (SubtractionMLP)")
    logger.info("=" * 70)

    cross_results = {}
    for train_ds in DATASET_SOURCES:
        tr_mask = rate_df["dataset_source"] == train_ds
        tr_split = rate_df[tr_mask & (rate_df["split"].isin(["train", "val"]))]
        # Split 80/20 for train/val
        n_val = max(5, len(tr_split) // 5)
        tr_split = tr_split.sample(frac=1, random_state=42)
        cd_val = tr_split.iloc[:n_val]
        cd_train = tr_split.iloc[n_val:]

        if len(cd_train) < 10 or len(cd_val) < 5:
            logger.info("  %s: too few samples (%d train), skipping", DATASET_LABELS[train_ds], len(cd_train))
            continue

        loaders = make_pooled_loaders(cd_train, cd_val, test_df, "subtraction")
        model = RateRegressionMLP(d_input=640)
        res = train_eval_generic(model, *loaders, epochs=80, lr=1e-3, patience=15)

        # Per-dataset test
        cross_results[train_ds] = evaluate_per_dataset(test_df, res["test_predictions"],
                                                       res["test_targets"], DATASET_SOURCES)
        logger.info("\n  Trained on %s:", DATASET_LABELS[train_ds])
        for te_ds in DATASET_SOURCES:
            m = cross_results[train_ds][te_ds]
            logger.info("    -> %-12s  Spearman=%.4f  n=%d",
                        DATASET_LABELS[te_ds], m["spearman"], m["n_samples"])

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("EDITING RATE PREDICTION - ALL 7 ARCHITECTURES (TEST SET)")
    print("=" * 100)
    header = f"{'Model':<20}{'Spearman':>12}{'Pearson':>12}{'MSE':>12}{'R2':>12}{'N':>10}"
    print(header)
    print("-" * 100)
    for model_name in all_results:
        m = all_results[model_name].get("test", {})
        sp = m.get("spearman", float("nan"))
        pe = m.get("pearson", float("nan"))
        mse = m.get("mse", float("nan"))
        r2 = m.get("r2", float("nan"))
        n = m.get("n_samples", 0)
        print(f"{model_name:<20}{sp:>12.4f}{pe:>12.4f}{mse:>12.4f}{r2:>12.4f}{n:>10d}")
    print("=" * 100)

    # Per-dataset Spearman table
    print("\n" + "=" * 100)
    print("PER-DATASET SPEARMAN CORRELATION (TEST SET)")
    print("=" * 100)
    ds_labels = [DATASET_LABELS[ds] for ds in DATASET_SOURCES]
    header = f"{'Model':<20}" + "".join(f"{dl:>15}" for dl in ds_labels)
    print(header)
    print("-" * 100)
    for model_name in per_dataset_results:
        row = f"{model_name:<20}"
        for ds in DATASET_SOURCES:
            v = per_dataset_results[model_name].get(ds, {}).get("spearman", float("nan"))
            row += f"{'N/A':>15}" if np.isnan(v) else f"{v:>15.4f}"
        print(row)
    print("=" * 100)

    # Cross-dataset table
    if cross_results:
        print("\n" + "=" * 100)
        print("CROSS-DATASET RATE PREDICTION: SPEARMAN (SubtractionMLP)")
        print("=" * 100)
        train_test_label = "Train / Test"
        header = f"{train_test_label:<15}" + "".join(f"{dl:>15}" for dl in ds_labels)
        print(header)
        print("-" * 100)
        for td in cross_results:
            row = f"{DATASET_LABELS[td]:<15}"
            for te in DATASET_SOURCES:
                v = cross_results[td].get(te, {}).get("spearman", float("nan"))
                row += f"{'N/A':>15}" if np.isnan(v) else f"{v:>15.4f}"
            print(row)
        print("=" * 100)

    total_time = time.time() - t_global
    logger.info("\nTotal experiment time: %.1fs", total_time)

    # ------------------------------------------------------------------
    # Generate plots
    # ------------------------------------------------------------------
    logger.info("\nGenerating plots...")
    plot_all_models_comparison(all_results, OUTPUT_DIR)
    if per_dataset_results:
        plot_per_dataset_heatmap(per_dataset_results, OUTPUT_DIR)
    if cross_results:
        plot_cross_dataset_rate(cross_results, OUTPUT_DIR)
    logger.info("  Saved all plots to %s", OUTPUT_DIR)

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

    json_results = {
        "description": "Rate prediction with all 7 DL architectures + cross-dataset training",
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "models": {},
        "per_dataset": {},
        "cross_dataset": {},
        "total_time_seconds": total_time,
    }

    for model_name in all_results:
        json_results["models"][model_name] = {}
        for split in ["train", "val", "test"]:
            metrics = all_results[model_name].get(split, {})
            json_results["models"][model_name][split] = {
                k: v for k, v in metrics.items() if not isinstance(v, np.ndarray)
            }

    json_results["per_dataset"] = {
        m: per_dataset_results[m] for m in per_dataset_results
    }
    json_results["cross_dataset"] = {
        DATASET_LABELS.get(td, td): {
            DATASET_LABELS.get(te, te): cross_results[td][te]
            for te in cross_results[td]
        }
        for td in cross_results
    }

    with open(OUTPUT_DIR / "rate_all_architectures_results.json", "w") as f:
        json.dump(json_results, f, indent=2, default=serialize)
    logger.info("Saved results to %s", OUTPUT_DIR / "rate_all_architectures_results.json")

    # Save test predictions for all models
    pred_arrays = {
        "site_ids": test_df["site_id"].values,
        "dataset_sources": test_df["dataset_source"].values,
        "true_log2_rates": test_df["log2_rate"].values,
    }
    for model_name in all_results:
        key = model_name.lower().replace("-", "_").replace(" ", "_")
        if "test_predictions" in all_results[model_name]:
            pred_arrays[f"pred_{key}"] = all_results[model_name]["test_predictions"]
    np.savez(OUTPUT_DIR / "rate_all_architectures_predictions.npz", **pred_arrays)
    logger.info("Saved predictions to %s", OUTPUT_DIR / "rate_all_architectures_predictions.npz")


if __name__ == "__main__":
    main()
