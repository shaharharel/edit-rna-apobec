#!/usr/bin/env python
"""Tissue-conditioned editing rate prediction experiment.

Predicts tissue-specific APOBEC3A editing rates by conditioning on both
the edit effect embedding (640d) and a learned tissue embedding (32d).
This enables answering: "What would the editing rate be at this site
in liver vs. brain vs. blood?"

Data sources:
    - GTEx tissue rates: data/processed/advisor/t1_gtex_editing_&_conservation.csv
    - Site metadata/splits: data/processed/splits_expanded.csv
    - Site ID mapping: data/processed/editing_sites_labels.csv
    - RNA-FM embeddings: data/processed/embeddings/rnafm_pooled{,_edited}.pt

Experiments:
    1. Tissue-conditioned MLP (edit_effect + tissue_embedding -> rate)
    2. Baselines: global tissue mean, per-site mean, edit-effect-only MLP
    3. Learned tissue embedding analysis (similarity heatmap, UMAP)
    4. Per-tissue performance breakdown
    5. Cross-tissue generalization (hold-out tissues)

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE /Users/shaharharel/miniconda3/envs/quris/bin/python \
        experiments/apobec3a/exp_tissue_conditioned_rate.py
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

EMB_DIR = PROJECT_ROOT / "data" / "processed" / "embeddings"
SPLITS_CSV = PROJECT_ROOT / "data" / "processed" / "splits_expanded.csv"
LABELS_CSV = PROJECT_ROOT / "data" / "processed" / "editing_sites_labels.csv"
GTEX_CSV = PROJECT_ROOT / "data" / "processed" / "advisor" / "t1_gtex_editing_&_conservation.csv"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "apobec3a" / "outputs" / "tissue_conditioned_rate"

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
})

# ---------------------------------------------------------------------------
# Data loading and preprocessing
# ---------------------------------------------------------------------------

def parse_tissue_rate(cell_value):
    """Parse a tissue cell 'edited_reads;total_reads;rate' -> rate.

    Returns NaN if unparseable or coverage < 10 reads.
    """
    if pd.isna(cell_value) or cell_value == "" or cell_value == "NA":
        return np.nan
    parts = str(cell_value).split(";")
    if len(parts) != 3:
        return np.nan
    try:
        coverage = float(parts[1])
        rate = float(parts[2])
        if coverage < 10:
            return np.nan
        return rate
    except (ValueError, IndexError):
        return np.nan


def load_tissue_data():
    """Load and parse per-tissue editing rates from the GTEx CSV.

    Returns:
        tissue_rates_df: DataFrame with columns [site_id, tissue_idx, tissue_name, rate]
        tissue_names: list of 54 tissue names (ordered)
        site_id_map: dict mapping (chr, start, end) -> site_id
    """
    gtex = pd.read_csv(GTEX_CSV)
    labels = pd.read_csv(LABELS_CSV)

    # Tissue columns are the last 54 columns (index 26 to 79 inclusive)
    tissue_cols = list(gtex.columns[26:])
    n_tissues = len(tissue_cols)
    logger.info("Found %d tissue columns in GTEx data", n_tissues)

    # Build (chr, start, end) -> site_id mapping from labels
    site_id_map = {}
    for _, row in labels.iterrows():
        key = (row["chr"], int(row["start"]), int(row["end"]))
        site_id_map[key] = row["site_id"]

    # Parse tissue rates for each site
    records = []
    for row_idx, row in gtex.iterrows():
        key = (row["Chr"], int(row["Start"]), int(row["End"]))
        site_id = site_id_map.get(key)
        if site_id is None:
            continue

        for tissue_idx, tissue_col in enumerate(tissue_cols):
            rate = parse_tissue_rate(row[tissue_col])
            if not np.isnan(rate):
                records.append({
                    "site_id": site_id,
                    "tissue_idx": tissue_idx,
                    "tissue_name": tissue_col,
                    "rate": rate,
                })

    tissue_rates_df = pd.DataFrame(records)
    return tissue_rates_df, tissue_cols, site_id_map


# ---------------------------------------------------------------------------
# Target transformation
# ---------------------------------------------------------------------------

def log2_transform(rate):
    """Convert editing rate (percentage, 0-100) to log2(rate/100 + 0.01).

    Rates are in percentage space. We normalize to [0,1] then apply log2.
    """
    rate_frac = rate / 100.0
    return np.log2(rate_frac + 0.01)


# ---------------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred):
    """Compute Spearman, Pearson, MSE, R2 from numpy arrays."""
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) < 3:
        return {
            "spearman": float("nan"), "pearson": float("nan"),
            "mse": float("nan"), "r2": float("nan"), "n_samples": 0,
        }

    sp_rho, _ = spearmanr(y_true, y_pred)
    pe_r, _ = pearsonr(y_true, y_pred)
    mse = float(mean_squared_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    return {
        "spearman": float(sp_rho), "pearson": float(pe_r),
        "mse": mse, "r2": r2, "n_samples": int(len(y_true)),
    }


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class TissueConditionedDataset(Dataset):
    """Dataset yielding (edit_effect_emb, tissue_idx) -> log2_rate.

    Each sample is a (site, tissue) pair with a valid editing rate.
    """

    def __init__(self, records_df, pooled_orig, pooled_edited):
        """
        Parameters
        ----------
        records_df : DataFrame with columns [site_id, tissue_idx, log2_rate]
        pooled_orig : dict site_id -> Tensor(640)
        pooled_edited : dict site_id -> Tensor(640)
        """
        self.site_ids = records_df["site_id"].values
        self.tissue_idxs = records_df["tissue_idx"].values.astype(np.int64)
        self.targets = records_df["log2_rate"].values.astype(np.float32)
        self.pooled_orig = pooled_orig
        self.pooled_edited = pooled_edited

    def __len__(self):
        return len(self.site_ids)

    def __getitem__(self, idx):
        sid = self.site_ids[idx]
        edit_effect = self.pooled_edited[sid] - self.pooled_orig[sid]
        tissue_idx = torch.tensor(self.tissue_idxs[idx], dtype=torch.long)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        return edit_effect, tissue_idx, target


class EditEffectOnlyDataset(Dataset):
    """Dataset for edit-effect-only baseline (no tissue conditioning).

    Each sample is a (site, tissue) pair but tissue is NOT provided to the model.
    The model must predict the tissue-specific rate from the site embedding alone.
    """

    def __init__(self, records_df, pooled_orig, pooled_edited):
        self.site_ids = records_df["site_id"].values
        self.targets = records_df["log2_rate"].values.astype(np.float32)
        self.pooled_orig = pooled_orig
        self.pooled_edited = pooled_edited

    def __len__(self):
        return len(self.site_ids)

    def __getitem__(self, idx):
        sid = self.site_ids[idx]
        edit_effect = self.pooled_edited[sid] - self.pooled_orig[sid]
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        return edit_effect, target


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class TissueConditionedMLP(nn.Module):
    """MLP that takes [edit_effect_emb; tissue_emb] -> predicted rate.

    Architecture:
        edit_effect (640d) || tissue_embedding (32d) -> 256 -> 128 -> 1
        with ReLU activations and Dropout.
    """

    def __init__(self, d_edit=640, n_tissues=54, d_tissue=32,
                 hidden1=256, hidden2=128, dropout=0.2):
        super().__init__()
        self.tissue_embedding = nn.Embedding(n_tissues, d_tissue)
        self.net = nn.Sequential(
            nn.Linear(d_edit + d_tissue, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden2, 1),
        )

    def forward(self, edit_effect, tissue_idx):
        tissue_emb = self.tissue_embedding(tissue_idx)
        x = torch.cat([edit_effect, tissue_emb], dim=-1)
        return self.net(x).squeeze(-1)

    def get_tissue_embeddings(self):
        """Return the learned tissue embedding matrix (n_tissues, d_tissue)."""
        return self.tissue_embedding.weight.detach().cpu().numpy()


class EditEffectOnlyMLP(nn.Module):
    """MLP baseline: edit_effect (640d) -> predicted rate, no tissue info."""

    def __init__(self, d_edit=640, hidden1=256, hidden2=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_edit, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden2, 1),
        )

    def forward(self, edit_effect):
        return self.net(edit_effect).squeeze(-1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_tissue_conditioned(train_df, val_df, pooled_orig, pooled_edited,
                             n_tissues=54, epochs=100, lr=1e-3, patience=20,
                             seed=42):
    """Train the tissue-conditioned MLP.

    Returns the trained model and training history.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds = TissueConditionedDataset(train_df, pooled_orig, pooled_edited)
    val_ds = TissueConditionedDataset(val_df, pooled_orig, pooled_edited)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=0)

    model = TissueConditionedMLP(d_edit=640, n_tissues=n_tissues, d_tissue=32)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        for edit_eff, tissue_idx, target in train_loader:
            optimizer.zero_grad()
            pred = model(edit_eff, tissue_idx)
            loss = F.mse_loss(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(loss.item())
        scheduler.step()

        train_loss = np.mean(epoch_losses)
        history["train_loss"].append(train_loss)

        # Validate
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for edit_eff, tissue_idx, target in val_loader:
                pred = model(edit_eff, tissue_idx)
                val_preds.append(pred.numpy())
                val_targets.append(target.numpy())
        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)
        val_loss = float(np.mean((val_preds - val_targets) ** 2))
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
        if patience_counter >= patience:
            logger.info("  Early stopping at epoch %d (best val_loss=%.5f)", epoch, best_val_loss)
            break

    if best_state:
        model.load_state_dict(best_state)

    return model, history


def train_edit_only(train_df, val_df, pooled_orig, pooled_edited,
                    epochs=100, lr=1e-3, patience=20, seed=42):
    """Train the edit-effect-only baseline MLP (no tissue info)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds = EditEffectOnlyDataset(train_df, pooled_orig, pooled_edited)
    val_ds = EditEffectOnlyDataset(val_df, pooled_orig, pooled_edited)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=0)

    model = EditEffectOnlyMLP(d_edit=640)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        for edit_eff, target in train_loader:
            optimizer.zero_grad()
            pred = model(edit_eff)
            loss = F.mse_loss(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for edit_eff, target in val_loader:
                pred = model(edit_eff)
                val_preds.append(pred.numpy())
                val_targets.append(target.numpy())
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
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_tissue_conditioned(model, df, pooled_orig, pooled_edited):
    """Evaluate the tissue-conditioned MLP. Returns predictions and metrics."""
    ds = TissueConditionedDataset(df, pooled_orig, pooled_edited)
    loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)

    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for edit_eff, tissue_idx, target in loader:
            pred = model(edit_eff, tissue_idx)
            all_preds.append(pred.numpy())
            all_targets.append(target.numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    metrics = compute_metrics(y_true, y_pred)
    return y_pred, y_true, metrics


def evaluate_edit_only(model, df, pooled_orig, pooled_edited):
    """Evaluate the edit-effect-only baseline."""
    ds = EditEffectOnlyDataset(df, pooled_orig, pooled_edited)
    loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)

    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for edit_eff, target in loader:
            pred = model(edit_eff)
            all_preds.append(pred.numpy())
            all_targets.append(target.numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    metrics = compute_metrics(y_true, y_pred)
    return y_pred, y_true, metrics


# ---------------------------------------------------------------------------
# Baseline models
# ---------------------------------------------------------------------------

def global_tissue_mean_baseline(train_df, test_df):
    """Baseline: predict the training set mean rate for each tissue.

    No site-specific information used -- just the global average
    editing rate per tissue from training data.
    """
    tissue_means = train_df.groupby("tissue_idx")["log2_rate"].mean().to_dict()
    global_mean = train_df["log2_rate"].mean()

    preds = test_df["tissue_idx"].map(tissue_means).fillna(global_mean).values
    targets = test_df["log2_rate"].values
    return preds.astype(np.float64), targets.astype(np.float64)


def per_site_mean_baseline(train_df, test_df, all_tissue_rates_df):
    """Baseline: predict the mean rate across tissues for each site.

    No tissue-specific information -- for each (site, tissue) test pair,
    predict the mean editing rate of that site across all tissues.
    Uses all available tissue data (not just training split) for the site means
    since this is a non-parametric baseline.
    For test sites unseen in training, we use the overall global mean.
    """
    # Compute per-site mean from ALL available data for that site
    site_means = all_tissue_rates_df.groupby("site_id")["log2_rate"].mean().to_dict()
    global_mean = all_tissue_rates_df["log2_rate"].mean()

    preds = test_df["site_id"].map(site_means).fillna(global_mean).values
    targets = test_df["log2_rate"].values
    return preds.astype(np.float64), targets.astype(np.float64)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_tissue_similarity_heatmap(tissue_emb_matrix, tissue_names, output_dir):
    """Plot cosine similarity heatmap of learned tissue embeddings."""
    # Normalize rows for cosine similarity
    norms = np.linalg.norm(tissue_emb_matrix, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    normed = tissue_emb_matrix / norms
    cos_sim = normed @ normed.T

    # Clean tissue names for display
    display_names = [n.replace("_", " ") for n in tissue_names]

    fig, ax = plt.subplots(figsize=(18, 16))
    im = ax.imshow(cos_sim, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Cosine Similarity")

    ax.set_xticks(range(len(display_names)))
    ax.set_yticks(range(len(display_names)))
    ax.set_xticklabels(display_names, rotation=90, fontsize=6, ha="center")
    ax.set_yticklabels(display_names, fontsize=6)
    ax.set_title("Learned Tissue Embedding Cosine Similarity", fontsize=14, fontweight="bold")

    plt.tight_layout()
    path = output_dir / "tissue_embedding_similarity.png"
    fig.savefig(path)
    plt.close(fig)
    logger.info("  Saved tissue similarity heatmap: %s", path)
    return cos_sim


def plot_ground_truth_tissue_correlation(tissue_rate_matrix, tissue_names, output_dir):
    """Plot ground-truth tissue correlation matrix from actual editing rates.

    Parameters
    ----------
    tissue_rate_matrix : ndarray (n_sites, n_tissues) with NaN for missing
    """
    # Compute pairwise Pearson correlation, ignoring NaN
    n_tissues = tissue_rate_matrix.shape[1]
    corr = np.full((n_tissues, n_tissues), np.nan)
    for i in range(n_tissues):
        for j in range(i, n_tissues):
            mask = np.isfinite(tissue_rate_matrix[:, i]) & np.isfinite(tissue_rate_matrix[:, j])
            if mask.sum() >= 5:
                r, _ = pearsonr(tissue_rate_matrix[mask, i], tissue_rate_matrix[mask, j])
                corr[i, j] = r
                corr[j, i] = r
            else:
                corr[i, j] = np.nan
                corr[j, i] = np.nan

    display_names = [n.replace("_", " ") for n in tissue_names]

    fig, ax = plt.subplots(figsize=(18, 16))
    # Replace NaN with 0 for display
    corr_display = np.nan_to_num(corr, nan=0.0)
    im = ax.imshow(corr_display, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Pearson Correlation")

    ax.set_xticks(range(len(display_names)))
    ax.set_yticks(range(len(display_names)))
    ax.set_xticklabels(display_names, rotation=90, fontsize=6, ha="center")
    ax.set_yticklabels(display_names, fontsize=6)
    ax.set_title("Ground-Truth Tissue Editing Rate Correlation", fontsize=14, fontweight="bold")

    plt.tight_layout()
    path = output_dir / "tissue_correlation_ground_truth.png"
    fig.savefig(path)
    plt.close(fig)
    logger.info("  Saved ground-truth tissue correlation: %s", path)
    return corr


def plot_tissue_umap(tissue_emb_matrix, tissue_names, output_dir):
    """UMAP projection of learned tissue embeddings. Color by tissue group."""
    try:
        from umap import UMAP
    except ImportError:
        logger.warning("  umap-learn not installed, falling back to PCA for tissue projection")
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2)
        coords = reducer.fit_transform(tissue_emb_matrix)
        method_name = "PCA"
    else:
        reducer = UMAP(n_components=2, n_neighbors=10, min_dist=0.3, random_state=42)
        coords = reducer.fit_transform(tissue_emb_matrix)
        method_name = "UMAP"

    # Define tissue groups for coloring
    tissue_groups = {}
    for name in tissue_names:
        if name.startswith("Brain_"):
            tissue_groups[name] = "Brain"
        elif name.startswith("Artery_"):
            tissue_groups[name] = "Artery"
        elif name.startswith("Heart_"):
            tissue_groups[name] = "Heart"
        elif name.startswith("Esophagus_"):
            tissue_groups[name] = "Esophagus"
        elif name.startswith("Colon_"):
            tissue_groups[name] = "Colon"
        elif name.startswith("Adipose_"):
            tissue_groups[name] = "Adipose"
        elif name.startswith("Skin_"):
            tissue_groups[name] = "Skin"
        elif name.startswith("Kidney_"):
            tissue_groups[name] = "Kidney"
        elif name.startswith("Cervix_"):
            tissue_groups[name] = "Cervix"
        elif name.startswith("Cells_"):
            tissue_groups[name] = "Cell Lines"
        else:
            tissue_groups[name] = "Other"

    group_colors = {
        "Brain": "#7c3aed", "Artery": "#f97316", "Heart": "#ec4899",
        "Esophagus": "#84cc16", "Colon": "#d97706", "Adipose": "#06b6d4",
        "Skin": "#a3e635", "Kidney": "#0891b2", "Cervix": "#f472b6",
        "Cell Lines": "#6b7280", "Other": "#9ca3af",
    }

    fig, ax = plt.subplots(figsize=(12, 10))

    groups_plotted = set()
    for i, name in enumerate(tissue_names):
        group = tissue_groups[name]
        color = group_colors.get(group, "#9ca3af")
        label = group if group not in groups_plotted else None
        groups_plotted.add(group)

        ax.scatter(coords[i, 0], coords[i, 1], s=60, c=color,
                   edgecolors="black", linewidths=0.5, zorder=3)
        # Label the point
        display_name = name.replace("_", " ")
        # Shorten long names
        if len(display_name) > 25:
            display_name = display_name[:22] + "..."
        ax.annotate(display_name, (coords[i, 0], coords[i, 1]),
                    fontsize=5.5, ha="left", va="bottom",
                    xytext=(4, 4), textcoords="offset points")

    # Create legend
    legend_handles = []
    for group in sorted(groups_plotted):
        color = group_colors.get(group, "#9ca3af")
        legend_handles.append(mpatches.Patch(color=color, label=group))
    ax.legend(handles=legend_handles, fontsize=8, loc="best", framealpha=0.9)

    ax.set_xlabel(f"{method_name} 1", fontsize=12)
    ax.set_ylabel(f"{method_name} 2", fontsize=12)
    ax.set_title(f"Learned Tissue Embeddings ({method_name} Projection)\n"
                 f"Do biologically similar tissues cluster?",
                 fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    path = output_dir / "tissue_embedding_umap.png"
    fig.savefig(path)
    plt.close(fig)
    logger.info("  Saved tissue %s: %s", method_name, path)
    return coords


def plot_per_tissue_performance(per_tissue_metrics, tissue_names, output_dir):
    """Bar chart of per-tissue Spearman correlation, sorted by performance."""
    tissues = []
    spearmans = []
    n_samples = []
    for tidx, tname in enumerate(tissue_names):
        m = per_tissue_metrics.get(tidx, {})
        sp = m.get("spearman", float("nan"))
        ns = m.get("n_samples", 0)
        if ns >= 5 and np.isfinite(sp):
            tissues.append(tname.replace("_", " "))
            spearmans.append(sp)
            n_samples.append(ns)

    if not tissues:
        logger.warning("  No tissues with enough samples for per-tissue plot")
        return

    # Sort by Spearman descending
    order = np.argsort(spearmans)[::-1]
    tissues = [tissues[i] for i in order]
    spearmans = [spearmans[i] for i in order]
    n_samples = [n_samples[i] for i in order]

    fig, ax = plt.subplots(figsize=(14, 8))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(tissues)))
    bars = ax.barh(range(len(tissues)), spearmans, color=colors[np.argsort(np.argsort(spearmans)[::-1])])

    ax.set_yticks(range(len(tissues)))
    ax.set_yticklabels(tissues, fontsize=7)
    ax.set_xlabel("Spearman Correlation (Test Set)", fontsize=12)
    ax.set_title("Per-Tissue Prediction Performance\n(Tissue-Conditioned MLP, Test Sites)",
                 fontsize=14, fontweight="bold")
    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.2, axis="x")

    # Annotate with n
    for i, (sp, ns) in enumerate(zip(spearmans, n_samples)):
        ax.text(max(sp + 0.01, 0.02), i, f"{sp:.3f} (n={ns})", va="center", fontsize=6)

    ax.invert_yaxis()
    plt.tight_layout()

    path = output_dir / "per_tissue_performance.png"
    fig.savefig(path)
    plt.close(fig)
    logger.info("  Saved per-tissue performance: %s", path)


def plot_model_comparison(all_metrics, output_dir):
    """Bar chart comparing all models on test set metrics."""
    models = list(all_metrics.keys())
    metric_names = ["spearman", "pearson", "r2"]
    metric_labels = ["Spearman rho", "Pearson r", "R-squared"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    x = np.arange(len(models))
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))

    for ax, metric, label in zip(axes, metric_names, metric_labels):
        vals = []
        for m in models:
            v = all_metrics[m].get(metric, float("nan"))
            vals.append(v if np.isfinite(v) else 0.0)

        bars = ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=8, rotation=25, ha="right")
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Model Comparison: Tissue-Conditioned Rate Prediction (Test Set)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = output_dir / "model_comparison.png"
    fig.savefig(path)
    plt.close(fig)
    logger.info("  Saved model comparison: %s", path)


def plot_cross_tissue_generalization(cross_tissue_results, output_dir):
    """Bar chart for cross-tissue held-out tissue performance."""
    tissues = []
    spearmans = []
    n_train = []
    n_test = []

    for tname, res in cross_tissue_results.items():
        sp = res.get("spearman", float("nan"))
        if np.isfinite(sp):
            tissues.append(tname.replace("_", " "))
            spearmans.append(sp)
            n_train.append(res.get("n_train_examples", 0))
            n_test.append(res.get("n_test_examples", 0))

    if not tissues:
        logger.warning("  No valid cross-tissue results to plot")
        return

    order = np.argsort(spearmans)[::-1]
    tissues = [tissues[i] for i in order]
    spearmans = [spearmans[i] for i in order]
    n_test = [n_test[i] for i in order]

    fig, ax = plt.subplots(figsize=(14, 8))
    colors = plt.cm.coolwarm(np.linspace(0.2, 0.8, len(tissues)))
    ax.barh(range(len(tissues)), spearmans,
            color=colors[np.argsort(np.argsort(spearmans)[::-1])])

    ax.set_yticks(range(len(tissues)))
    ax.set_yticklabels(tissues, fontsize=7)
    ax.set_xlabel("Spearman Correlation (Held-out Tissue)", fontsize=12)
    ax.set_title("Cross-Tissue Generalization\n(Train without tissue, predict its rates on test sites)",
                 fontsize=14, fontweight="bold")
    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.2, axis="x")

    for i, (sp, nt) in enumerate(zip(spearmans, n_test)):
        ax.text(max(sp + 0.01, 0.02), i, f"{sp:.3f} (n={nt})", va="center", fontsize=6)

    ax.invert_yaxis()
    plt.tight_layout()

    path = output_dir / "cross_tissue_generalization.png"
    fig.savefig(path)
    plt.close(fig)
    logger.info("  Saved cross-tissue generalization: %s", path)


def plot_embedding_vs_ground_truth(learned_sim, gt_corr, tissue_names, output_dir):
    """Scatter plot comparing learned embedding similarity to ground-truth correlation."""
    n = len(tissue_names)
    learned_vals = []
    gt_vals = []

    for i in range(n):
        for j in range(i + 1, n):
            if np.isfinite(learned_sim[i, j]) and np.isfinite(gt_corr[i, j]):
                learned_vals.append(learned_sim[i, j])
                gt_vals.append(gt_corr[i, j])

    learned_vals = np.array(learned_vals)
    gt_vals = np.array(gt_vals)

    if len(learned_vals) < 3:
        logger.warning("  Not enough tissue pairs for embedding vs ground-truth comparison")
        return

    sp_rho, _ = spearmanr(learned_vals, gt_vals)
    pe_r, _ = pearsonr(learned_vals, gt_vals)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(gt_vals, learned_vals, s=8, alpha=0.4, color="steelblue")
    ax.set_xlabel("Ground-Truth Tissue Correlation (Pearson)", fontsize=12)
    ax.set_ylabel("Learned Embedding Cosine Similarity", fontsize=12)
    ax.set_title(f"Learned vs Ground-Truth Tissue Relationships\n"
                 f"Spearman={sp_rho:.3f}, Pearson={pe_r:.3f} ({len(learned_vals)} tissue pairs)",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Reference line
    ax.plot([-1, 1], [-1, 1], "r--", alpha=0.5, linewidth=1)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal")

    plt.tight_layout()
    path = output_dir / "embedding_vs_ground_truth.png"
    fig.savefig(path)
    plt.close(fig)
    logger.info("  Saved embedding vs ground-truth: %s", path)

    return {"spearman": float(sp_rho), "pearson": float(pe_r),
            "n_pairs": len(learned_vals)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)
    torch.manual_seed(42)
    t_global = time.time()

    # ==================================================================
    # 1. Load data
    # ==================================================================
    logger.info("Loading tissue editing data...")
    tissue_rates_df, tissue_names, site_id_map = load_tissue_data()
    n_tissues = len(tissue_names)
    logger.info("  Tissue rates: %d records, %d tissues", len(tissue_rates_df), n_tissues)
    logger.info("  Unique sites in tissue data: %d", tissue_rates_df["site_id"].nunique())

    logger.info("Loading embeddings...")
    pooled_orig = torch.load(EMB_DIR / "rnafm_pooled.pt", weights_only=False)
    pooled_edited = torch.load(EMB_DIR / "rnafm_pooled_edited.pt", weights_only=False)
    available_emb = set(pooled_orig.keys()) & set(pooled_edited.keys())
    logger.info("  Embeddings available for %d sites", len(available_emb))

    logger.info("Loading splits...")
    splits_df = pd.read_csv(SPLITS_CSV)
    levanon = splits_df[splits_df["dataset_source"] == "advisor_c2t"].copy()
    levanon = levanon[levanon["site_id"].isin(available_emb)]
    logger.info("  Levanon sites with embeddings: %d", len(levanon))

    # Map site_id -> split
    site_split = dict(zip(levanon["site_id"], levanon["split"]))

    # ==================================================================
    # 2. Build training data: (site, tissue) pairs with valid rates
    # ==================================================================
    # Filter tissue_rates_df to sites with embeddings and known splits
    tissue_rates_df = tissue_rates_df[
        tissue_rates_df["site_id"].isin(available_emb) &
        tissue_rates_df["site_id"].isin(site_split)
    ].copy()

    # Apply log2 transform
    tissue_rates_df["log2_rate"] = tissue_rates_df["rate"].apply(log2_transform)
    tissue_rates_df = tissue_rates_df[tissue_rates_df["log2_rate"].notna()].copy()

    # Assign split based on SITE (not tissue) to avoid data leakage
    tissue_rates_df["split"] = tissue_rates_df["site_id"].map(site_split)

    # Split
    train_df = tissue_rates_df[tissue_rates_df["split"] == "train"].copy()
    val_df = tissue_rates_df[tissue_rates_df["split"] == "val"].copy()
    test_df = tissue_rates_df[tissue_rates_df["split"] == "test"].copy()

    logger.info("\nDataset sizes (site x tissue pairs):")
    logger.info("  Train: %d (%d unique sites)", len(train_df), train_df["site_id"].nunique())
    logger.info("  Val:   %d (%d unique sites)", len(val_df), val_df["site_id"].nunique())
    logger.info("  Test:  %d (%d unique sites)", len(test_df), test_df["site_id"].nunique())
    logger.info("  Total: %d", len(tissue_rates_df))

    logger.info("\nTarget log2(rate/100 + 0.01) stats:")
    logger.info("  Train: mean=%.3f, std=%.3f", train_df["log2_rate"].mean(), train_df["log2_rate"].std())
    logger.info("  Val:   mean=%.3f, std=%.3f", val_df["log2_rate"].mean(), val_df["log2_rate"].std())
    logger.info("  Test:  mean=%.3f, std=%.3f", test_df["log2_rate"].mean(), test_df["log2_rate"].std())

    # ==================================================================
    # 3. Build ground-truth tissue rate matrix for correlation analysis
    # ==================================================================
    logger.info("\nBuilding ground-truth tissue rate matrix...")
    all_site_ids = sorted(tissue_rates_df["site_id"].unique())
    site_id_to_idx = {sid: i for i, sid in enumerate(all_site_ids)}
    tissue_rate_matrix = np.full((len(all_site_ids), n_tissues), np.nan)
    for _, row in tissue_rates_df.iterrows():
        si = site_id_to_idx[row["site_id"]]
        ti = row["tissue_idx"]
        tissue_rate_matrix[si, ti] = row["log2_rate"]
    logger.info("  Matrix shape: %s, non-NaN: %d / %d",
                tissue_rate_matrix.shape,
                np.isfinite(tissue_rate_matrix).sum(),
                tissue_rate_matrix.size)

    # ==================================================================
    # Experiment 1: Tissue-Conditioned MLP
    # ==================================================================
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 1: Tissue-Conditioned MLP")
    logger.info("  Input: edit_effect(640d) + tissue_embedding(32d)")
    logger.info("=" * 70)

    t0 = time.time()
    tc_model, tc_history = train_tissue_conditioned(
        train_df, val_df, pooled_orig, pooled_edited,
        n_tissues=n_tissues, epochs=200, lr=5e-4, patience=30, seed=42,
    )
    tc_train_pred, tc_train_true, tc_train_metrics = evaluate_tissue_conditioned(
        tc_model, train_df, pooled_orig, pooled_edited)
    tc_val_pred, tc_val_true, tc_val_metrics = evaluate_tissue_conditioned(
        tc_model, val_df, pooled_orig, pooled_edited)
    tc_test_pred, tc_test_true, tc_test_metrics = evaluate_tissue_conditioned(
        tc_model, test_df, pooled_orig, pooled_edited)
    tc_time = time.time() - t0

    logger.info("  Time: %.1fs", tc_time)
    logger.info("  Train: Spearman=%.4f  Pearson=%.4f  R2=%.4f  n=%d",
                tc_train_metrics["spearman"], tc_train_metrics["pearson"],
                tc_train_metrics["r2"], tc_train_metrics["n_samples"])
    logger.info("  Val:   Spearman=%.4f  Pearson=%.4f  R2=%.4f  n=%d",
                tc_val_metrics["spearman"], tc_val_metrics["pearson"],
                tc_val_metrics["r2"], tc_val_metrics["n_samples"])
    logger.info("  Test:  Spearman=%.4f  Pearson=%.4f  R2=%.4f  n=%d",
                tc_test_metrics["spearman"], tc_test_metrics["pearson"],
                tc_test_metrics["r2"], tc_test_metrics["n_samples"])

    # ==================================================================
    # Experiment 2: Baselines
    # ==================================================================
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 2: Baselines")
    logger.info("=" * 70)

    # 2a. Global tissue mean
    logger.info("\n  2a. Global tissue mean (no site info)...")
    gtm_preds, gtm_targets = global_tissue_mean_baseline(train_df, test_df)
    gtm_metrics = compute_metrics(gtm_targets, gtm_preds)
    logger.info("      Test: Spearman=%.4f  Pearson=%.4f  R2=%.4f  n=%d",
                gtm_metrics["spearman"], gtm_metrics["pearson"],
                gtm_metrics["r2"], gtm_metrics["n_samples"])

    # 2b. Per-site mean
    logger.info("\n  2b. Per-site mean (no tissue info)...")
    psm_preds, psm_targets = per_site_mean_baseline(train_df, test_df, tissue_rates_df)
    psm_metrics = compute_metrics(psm_targets, psm_preds)
    logger.info("      Test: Spearman=%.4f  Pearson=%.4f  R2=%.4f  n=%d",
                psm_metrics["spearman"], psm_metrics["pearson"],
                psm_metrics["r2"], psm_metrics["n_samples"])

    # 2c. Site mean + tissue mean (additive)
    logger.info("\n  2c. Site mean + tissue mean (additive baseline)...")
    tissue_means = train_df.groupby("tissue_idx")["log2_rate"].mean()
    site_means_all = tissue_rates_df.groupby("site_id")["log2_rate"].mean()
    global_mean = train_df["log2_rate"].mean()
    stm_preds = []
    for _, row in test_df.iterrows():
        s_mean = site_means_all.get(row["site_id"], global_mean)
        t_mean = tissue_means.get(row["tissue_idx"], global_mean)
        stm_preds.append(s_mean + t_mean - global_mean)
    stm_preds = np.array(stm_preds, dtype=np.float64)
    stm_targets = test_df["log2_rate"].values.astype(np.float64)
    stm_metrics = compute_metrics(stm_targets, stm_preds)
    logger.info("      Test: Spearman=%.4f  Pearson=%.4f  R2=%.4f  n=%d",
                stm_metrics["spearman"], stm_metrics["pearson"],
                stm_metrics["r2"], stm_metrics["n_samples"])

    # 2d. Edit-effect-only MLP (no tissue conditioning)
    logger.info("\n  2d. Edit-effect-only MLP (no tissue info)...")
    t0 = time.time()
    eo_model = train_edit_only(
        train_df, val_df, pooled_orig, pooled_edited,
        epochs=200, lr=5e-4, patience=30, seed=42,
    )
    eo_test_pred, eo_test_true, eo_test_metrics = evaluate_edit_only(
        eo_model, test_df, pooled_orig, pooled_edited)
    eo_time = time.time() - t0
    logger.info("      Time: %.1fs", eo_time)
    logger.info("      Test: Spearman=%.4f  Pearson=%.4f  R2=%.4f  n=%d",
                eo_test_metrics["spearman"], eo_test_metrics["pearson"],
                eo_test_metrics["r2"], eo_test_metrics["n_samples"])

    # ==================================================================
    # Aggregate all model metrics
    # ==================================================================
    all_test_metrics = {
        "Tissue-Conditioned MLP": tc_test_metrics,
        "Edit-Effect-Only MLP": eo_test_metrics,
        "Site+Tissue Mean": stm_metrics,
        "Per-Site Mean": psm_metrics,
        "Global Tissue Mean": gtm_metrics,
    }

    # ==================================================================
    # Experiment 3: Learned tissue embedding analysis
    # ==================================================================
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 3: Tissue Embedding Analysis")
    logger.info("=" * 70)

    tissue_emb_matrix = tc_model.get_tissue_embeddings()
    logger.info("  Tissue embedding matrix shape: %s", tissue_emb_matrix.shape)

    # 3a. Cosine similarity heatmap
    logger.info("\n  3a. Tissue embedding similarity heatmap...")
    learned_sim = plot_tissue_similarity_heatmap(tissue_emb_matrix, tissue_names, OUTPUT_DIR)

    # 3b. Ground-truth tissue correlation
    logger.info("\n  3b. Ground-truth tissue correlation matrix...")
    gt_corr = plot_ground_truth_tissue_correlation(tissue_rate_matrix, tissue_names, OUTPUT_DIR)

    # 3c. Compare learned vs ground-truth
    logger.info("\n  3c. Comparing learned embeddings to ground-truth...")
    emb_vs_gt = plot_embedding_vs_ground_truth(learned_sim, gt_corr, tissue_names, OUTPUT_DIR)
    if emb_vs_gt:
        logger.info("      Learned vs GT -- Spearman=%.4f, Pearson=%.4f (%d pairs)",
                    emb_vs_gt["spearman"], emb_vs_gt["pearson"], emb_vs_gt["n_pairs"])

    # 3d. UMAP of tissue embeddings
    logger.info("\n  3d. Tissue embedding UMAP/PCA...")
    tissue_coords = plot_tissue_umap(tissue_emb_matrix, tissue_names, OUTPUT_DIR)

    # ==================================================================
    # Experiment 4: Per-tissue performance
    # ==================================================================
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 4: Per-Tissue Performance")
    logger.info("=" * 70)

    per_tissue_metrics = {}
    test_tissue_groups = test_df.groupby("tissue_idx")
    for tissue_idx, group in test_tissue_groups:
        if len(group) >= 3:
            # Get predictions for this tissue from the full test predictions
            idxs = group.index
            mask = test_df.index.isin(idxs)
            y_true = tc_test_true[test_df.index.get_indexer(idxs)]
            y_pred = tc_test_pred[test_df.index.get_indexer(idxs)]
            m = compute_metrics(y_true, y_pred)
            per_tissue_metrics[tissue_idx] = m
            per_tissue_metrics[tissue_idx]["tissue_name"] = tissue_names[tissue_idx]

    # For proper indexing, rebuild predictions aligned with test_df row order
    # Actually, the predictions are already aligned since we use the same dataset
    test_df_reset = test_df.reset_index(drop=True)
    per_tissue_metrics = {}
    for tissue_idx in range(n_tissues):
        mask = test_df_reset["tissue_idx"].values == tissue_idx
        if mask.sum() >= 3:
            y_true = tc_test_true[mask]
            y_pred = tc_test_pred[mask]
            m = compute_metrics(y_true, y_pred)
            m["tissue_name"] = tissue_names[tissue_idx]
            per_tissue_metrics[tissue_idx] = m

    logger.info("  Evaluated %d tissues with >= 3 test samples", len(per_tissue_metrics))

    # Log top 5 and bottom 5
    sorted_tissues = sorted(per_tissue_metrics.items(),
                           key=lambda x: x[1].get("spearman", float("-inf")),
                           reverse=True)
    logger.info("\n  Top 5 easiest tissues:")
    for tidx, m in sorted_tissues[:5]:
        logger.info("    %-40s Spearman=%.4f  n=%d",
                    tissue_names[tidx], m["spearman"], m["n_samples"])

    logger.info("\n  Bottom 5 hardest tissues:")
    for tidx, m in sorted_tissues[-5:]:
        logger.info("    %-40s Spearman=%.4f  n=%d",
                    tissue_names[tidx], m["spearman"], m["n_samples"])

    plot_per_tissue_performance(per_tissue_metrics, tissue_names, OUTPUT_DIR)

    # ==================================================================
    # Experiment 5: Cross-tissue generalization
    # ==================================================================
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 5: Cross-Tissue Generalization")
    logger.info("  Hold out entire tissues from training, predict their rates")
    logger.info("=" * 70)

    # Select a representative subset of tissues (one from each major group)
    # to keep runtime reasonable
    holdout_tissues = [
        "Whole_Blood", "Liver", "Lung", "Testis",
        "Brain_Cortex", "Heart_Left_Ventricle",
        "Skin_Sun_Exposed_Lower_leg", "Kidney_Cortex",
        "Pancreas", "Spleen",
    ]
    # Filter to tissues that actually exist
    holdout_tissues = [t for t in holdout_tissues if t in tissue_names]
    holdout_tissue_idxs = {t: tissue_names.index(t) for t in holdout_tissues}

    cross_tissue_results = {}
    for heldout_name in holdout_tissues:
        heldout_idx = holdout_tissue_idxs[heldout_name]
        logger.info("\n  Holding out: %s (idx=%d)", heldout_name, heldout_idx)

        # Train without this tissue
        train_subset = train_df[train_df["tissue_idx"] != heldout_idx].copy()
        val_subset = val_df[val_df["tissue_idx"] != heldout_idx].copy()

        # Test: only this tissue's data on test sites
        test_subset = test_df[test_df["tissue_idx"] == heldout_idx].copy()

        if len(train_subset) < 50 or len(test_subset) < 5:
            logger.info("    Skipping: insufficient data (train=%d, test=%d)",
                        len(train_subset), len(test_subset))
            continue

        # Train a new model without this tissue
        ct_model, _ = train_tissue_conditioned(
            train_subset, val_subset, pooled_orig, pooled_edited,
            n_tissues=n_tissues, epochs=80, lr=1e-3, patience=15, seed=42,
        )

        # Evaluate on held-out tissue
        ct_pred, ct_true, ct_metrics = evaluate_tissue_conditioned(
            ct_model, test_subset, pooled_orig, pooled_edited)

        cross_tissue_results[heldout_name] = {
            **ct_metrics,
            "n_train_examples": len(train_subset),
            "n_test_examples": len(test_subset),
        }

        logger.info("    Test: Spearman=%.4f  Pearson=%.4f  R2=%.4f  n=%d",
                    ct_metrics["spearman"], ct_metrics["pearson"],
                    ct_metrics["r2"], ct_metrics["n_samples"])

    if cross_tissue_results:
        plot_cross_tissue_generalization(cross_tissue_results, OUTPUT_DIR)

    # ==================================================================
    # Generate remaining plots
    # ==================================================================
    logger.info("\n" + "=" * 70)
    logger.info("Generating plots...")
    logger.info("=" * 70)

    plot_model_comparison(all_test_metrics, OUTPUT_DIR)

    # ==================================================================
    # Print summary
    # ==================================================================
    total_time = time.time() - t_global

    print("\n" + "=" * 95)
    print("TISSUE-CONDITIONED EDITING RATE PREDICTION - MODEL COMPARISON (TEST SET)")
    print("=" * 95)
    header = f"{'Model':<30}{'Spearman':>12}{'Pearson':>12}{'MSE':>12}{'R2':>12}{'N':>10}"
    print(header)
    print("-" * 95)
    for model_name, m in all_test_metrics.items():
        sp = m.get("spearman", float("nan"))
        pe = m.get("pearson", float("nan"))
        mse = m.get("mse", float("nan"))
        r2 = m.get("r2", float("nan"))
        n = m.get("n_samples", 0)
        print(f"{model_name:<30}{sp:>12.4f}{pe:>12.4f}{mse:>12.4f}{r2:>12.4f}{n:>10d}")
    print("=" * 95)

    if cross_tissue_results:
        print("\n" + "=" * 95)
        print("CROSS-TISSUE GENERALIZATION (HELD-OUT TISSUES, TEST SITES)")
        print("=" * 95)
        header = f"{'Held-out Tissue':<40}{'Spearman':>12}{'Pearson':>12}{'R2':>12}{'N test':>10}"
        print(header)
        print("-" * 95)
        for tname in holdout_tissues:
            if tname in cross_tissue_results:
                m = cross_tissue_results[tname]
                print(f"{tname:<40}{m['spearman']:>12.4f}{m['pearson']:>12.4f}"
                      f"{m['r2']:>12.4f}{m['n_samples']:>10d}")
        print("=" * 95)

    print(f"\nTotal experiment time: {total_time:.1f}s")

    # ==================================================================
    # Save JSON results
    # ==================================================================
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
        "description": (
            "Tissue-conditioned editing rate prediction. "
            "Predicts tissue-specific APOBEC3A editing rates by conditioning on "
            "edit effect embedding (640d) + learned tissue embedding (32d). "
            "Target: log2(rate/100 + 0.01). Site-level splits prevent data leakage."
        ),
        "data_summary": {
            "n_tissues": n_tissues,
            "n_train_pairs": len(train_df),
            "n_val_pairs": len(val_df),
            "n_test_pairs": len(test_df),
            "n_train_sites": int(train_df["site_id"].nunique()),
            "n_val_sites": int(val_df["site_id"].nunique()),
            "n_test_sites": int(test_df["site_id"].nunique()),
            "target_stats": {
                "train_mean": float(train_df["log2_rate"].mean()),
                "train_std": float(train_df["log2_rate"].std()),
                "test_mean": float(test_df["log2_rate"].mean()),
                "test_std": float(test_df["log2_rate"].std()),
            },
        },
        "models": {},
        "per_tissue_performance": {},
        "cross_tissue_generalization": {},
        "tissue_embedding_analysis": {},
        "total_time_seconds": total_time,
    }

    # Model metrics
    for model_name, m in all_test_metrics.items():
        json_results["models"][model_name] = {
            "test": {k: v for k, v in m.items() if not isinstance(v, np.ndarray)},
        }

    # Add train/val for the main model
    json_results["models"]["Tissue-Conditioned MLP"]["train"] = {
        k: v for k, v in tc_train_metrics.items() if not isinstance(v, np.ndarray)
    }
    json_results["models"]["Tissue-Conditioned MLP"]["val"] = {
        k: v for k, v in tc_val_metrics.items() if not isinstance(v, np.ndarray)
    }

    # Per-tissue performance
    for tidx, m in per_tissue_metrics.items():
        json_results["per_tissue_performance"][tissue_names[tidx]] = {
            k: v for k, v in m.items() if not isinstance(v, np.ndarray)
        }

    # Cross-tissue generalization
    for tname, m in cross_tissue_results.items():
        json_results["cross_tissue_generalization"][tname] = {
            k: v for k, v in m.items() if not isinstance(v, np.ndarray)
        }

    # Embedding analysis
    if emb_vs_gt:
        json_results["tissue_embedding_analysis"]["learned_vs_ground_truth"] = emb_vs_gt

    with open(OUTPUT_DIR / "tissue_conditioned_results.json", "w") as f:
        json.dump(json_results, f, indent=2, default=serialize)
    logger.info("Saved results to %s", OUTPUT_DIR / "tissue_conditioned_results.json")

    # Save tissue embedding matrix for further analysis
    np.savez(
        OUTPUT_DIR / "tissue_embeddings.npz",
        embedding_matrix=tissue_emb_matrix,
        tissue_names=np.array(tissue_names),
        learned_similarity=learned_sim,
        ground_truth_correlation=gt_corr,
    )
    logger.info("Saved tissue embeddings to %s", OUTPUT_DIR / "tissue_embeddings.npz")

    logger.info("\nDone. All outputs saved to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
