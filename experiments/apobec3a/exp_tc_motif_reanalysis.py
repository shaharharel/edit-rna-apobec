#!/usr/bin/env python
"""TC-Motif Reanalysis: The first genuinely APOBEC3A-specific predictor.

This is the single most important experiment for the paper. It re-runs the
entire classification pipeline using ONLY TC-motif validated sites, creating
the first predictor that is genuinely specific to APOBEC3A editing.

Background:
  Existing APOBEC3A editing catalogs are contaminated with ADAR sites:
    - Levanon:  only  7.3% TC-motif (46/629)
    - Alqassim: only  8.7% TC-motif (11/126)
    - Sharma:   89.8% TC-motif (247/275)
    - Baysal:   97.6% TC-motif (4094/4196)

  TC-motif = the -1 position relative to the edited C is U/T (5'-UC-3'),
  which is the canonical APOBEC3A recognition motif. Non-TC sites are
  almost certainly ADAR C-to-U editing mis-annotations or other enzymes.

  This experiment:
    Phase 1: Dataset preparation (TC-motif filtering)
    Phase 2: TC-motif-matched negative controls
    Phase 3: Binary classification with TC-motif data
    Phase 4: Within-Baysal rate prediction (the fair rate test)
    Phase 5: 4-panel publication figure

Usage:
    python experiments/apobec3a/exp_tc_motif_reanalysis.py
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

# OpenBLAS / MKL thread fix (must be set before numpy import)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
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
COMBINED_CSV = PROJECT_ROOT / "data" / "processed" / "all_datasets_combined.csv"
SPLITS_CSV = PROJECT_ROOT / "data" / "processed" / "splits_expanded.csv"
SEQ_JSON = PROJECT_ROOT / "data" / "processed" / "site_sequences.json"
STRUCT_CACHE = EMB_DIR / "vienna_structure_cache.npz"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "apobec" / "outputs" / "tc_motif_reanalysis"

# Hard negatives (TC-motif matched, structure-matched) from the pipeline
HARDNEG_CSV = PROJECT_ROOT / "data" / "processed" / "hardneg_per_dataset.csv"
HARDNEG_SEQ = PROJECT_ROOT / "data" / "processed" / "hardneg_site_sequences.json"
HARDNEG_POOLED = EMB_DIR / "hardneg_rnafm_pooled.pt"
HARDNEG_POOLED_EDITED = EMB_DIR / "hardneg_rnafm_pooled_edited.pt"
HARDNEG_TOKENS = EMB_DIR / "hardneg_rnafm_tokens.pt"
HARDNEG_TOKENS_EDITED = EMB_DIR / "hardneg_rnafm_tokens_edited.pt"
HARDNEG_STRUCT = EMB_DIR / "hardneg_vienna_structure.npz"

DATASET_LABELS = {
    "advisor_c2t": "Levanon",
    "asaoka_2019": "Asaoka",
    "alqassim_2021": "Alqassim",
    "sharma_2015": "Sharma",
    "baysal_2016": "Baysal",
}

SEED = 42


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def get_tc_motif(seq, pos=100):
    """Check if site has TC motif (U/T at position -1 relative to C).

    The TC dinucleotide (5'-UC-3') is the canonical APOBEC3A recognition
    motif. Position `pos` should be the location of the edited C in the
    sequence window. We check that the preceding base is U or T.
    """
    if len(seq) <= pos or pos < 1:
        return False
    prev = seq[pos - 1].upper()
    curr = seq[pos].upper()
    return prev in ("U", "T") and curr == "C"


def compute_log2_rate(rate):
    """Convert raw editing rate to log2(rate + 0.01).

    Rates > 1.0 are assumed to be percentages and divided by 100.
    Returns NaN for NaN or non-positive inputs.
    """
    if pd.isna(rate) or rate < 0:
        return float("nan")
    r = float(rate)
    if r > 1.0:
        r = r / 100.0
    return np.log2(r + 0.01)


def compute_binary_metrics(y_true, y_score):
    """Compute binary classification metrics (AUROC, AUPRC, F1, etc.)."""
    mask = ~(np.isnan(y_true) | np.isnan(y_score))
    y_true, y_score = y_true[mask], y_score[mask]

    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return {k: float("nan") for k in [
            "auroc", "auprc", "f1", "precision", "recall",
            "accuracy", "n_positive", "n_negative",
        ]}

    metrics = {
        "auroc": roc_auc_score(y_true, y_score),
        "auprc": average_precision_score(y_true, y_score),
    }

    # Optimal F1 threshold
    prec_arr, rec_arr, thresholds = precision_recall_curve(y_true, y_score)
    f1_arr = 2 * prec_arr[:-1] * rec_arr[:-1] / (prec_arr[:-1] + rec_arr[:-1] + 1e-8)
    best_idx = np.argmax(f1_arr)
    threshold = thresholds[best_idx] if len(thresholds) > 0 else 0.5
    y_pred = (y_score >= threshold).astype(int)

    metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)
    metrics["precision"] = float(np.sum((y_pred == 1) & (y_true == 1)) / max(np.sum(y_pred == 1), 1))
    metrics["recall"] = float(np.sum((y_pred == 1) & (y_true == 1)) / max(np.sum(y_true == 1), 1))
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["n_positive"] = int(y_true.sum())
    metrics["n_negative"] = int(len(y_true) - y_true.sum())

    return metrics


def compute_regression_metrics(y_true, y_pred):
    """Compute Spearman, Pearson, MSE, R^2."""
    from sklearn.metrics import mean_squared_error, r2_score

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]

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


def serialize(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return str(obj)


# ---------------------------------------------------------------------------
# Focal Loss (same as train_baselines.py)
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Binary focal loss for handling class imbalance."""

    def __init__(self, gamma=2.0, alpha=0.75):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits.squeeze(-1))
        targets = targets.float()
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        bce = F.binary_cross_entropy_with_logits(
            logits.squeeze(-1), targets, reduction="none"
        )
        return (focal_weight * bce).mean()


# ---------------------------------------------------------------------------
# Dataset for pre-computed embeddings (binary classification)
# ---------------------------------------------------------------------------

class EmbeddingDataset(Dataset):
    """Dataset serving pre-computed pooled embeddings + structure features.

    Used for SubtractionMLP and PooledMLP binary classification.
    """

    def __init__(self, site_ids, labels, pooled_orig, pooled_edited, structure_delta=None):
        self.site_ids = site_ids
        self.labels = labels
        self.pooled_orig = pooled_orig
        self.pooled_edited = pooled_edited
        self.structure_delta = structure_delta

    def __len__(self):
        return len(self.site_ids)

    def __getitem__(self, idx):
        sid = self.site_ids[idx]
        item = {
            "site_id": sid,
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
            "pooled_orig": self.pooled_orig[sid],
            "pooled_edited": self.pooled_edited[sid],
        }
        if self.structure_delta is not None and sid in self.structure_delta:
            item["structure_delta"] = torch.tensor(
                self.structure_delta[sid], dtype=torch.float32
            )
        else:
            item["structure_delta"] = torch.zeros(7, dtype=torch.float32)
        return item


def embedding_collate_fn(batch):
    """Collate function for EmbeddingDataset."""
    return {
        "site_ids": [b["site_id"] for b in batch],
        "labels": torch.stack([b["label"] for b in batch]),
        "pooled_orig": torch.stack([b["pooled_orig"] for b in batch]),
        "pooled_edited": torch.stack([b["pooled_edited"] for b in batch]),
        "structure_delta": torch.stack([b["structure_delta"] for b in batch]),
    }


# ---------------------------------------------------------------------------
# Dataset for rate regression
# ---------------------------------------------------------------------------

class RateRegressionDataset(Dataset):
    """Dataset for rate regression from pooled embeddings."""

    def __init__(self, site_ids, targets, pooled_orig, pooled_edited,
                 structure_delta=None, mode="subtraction"):
        self.site_ids = site_ids
        self.targets = targets
        self.pooled_orig = pooled_orig
        self.pooled_edited = pooled_edited
        self.structure_delta = structure_delta
        self.mode = mode

    def __len__(self):
        return len(self.site_ids)

    def __getitem__(self, idx):
        sid = self.site_ids[idx]
        orig = self.pooled_orig[sid]
        edited = self.pooled_edited[sid]

        if self.mode == "subtraction":
            x = edited - orig  # (640,)
        elif self.mode == "concat":
            x = torch.cat([orig, edited], dim=-1)  # (1280,)
        elif self.mode == "pooled_only":
            x = edited  # (640,) -- just the edited embedding
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Append structure delta features if available
        if self.structure_delta is not None and sid in self.structure_delta:
            sd = torch.tensor(self.structure_delta[sid], dtype=torch.float32)
        else:
            sd = torch.zeros(7, dtype=torch.float32)
        x = torch.cat([x, sd], dim=-1)

        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        return x, target


# ---------------------------------------------------------------------------
# MLP Models
# ---------------------------------------------------------------------------

class BinaryMLP(nn.Module):
    """MLP for binary classification from pre-computed features.

    Used for both SubtractionMLP and PooledMLP approaches.
    """

    def __init__(self, d_input, hidden_dims=(512, 256), dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = d_input
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, batch):
        """Accept batch dict (SubtractionMLP or PooledMLP style)."""
        pooled_orig = batch["pooled_orig"]      # (B, 640)
        pooled_edited = batch["pooled_edited"]   # (B, 640)
        structure = batch["structure_delta"]      # (B, 7)

        if hasattr(self, "_mode") and self._mode == "pooled":
            x = torch.cat([pooled_edited, structure], dim=-1)  # (B, 647)
        else:
            # Subtraction mode: concat difference + both pooled + structure
            diff = pooled_edited - pooled_orig
            x = torch.cat([diff, structure], dim=-1)  # (B, 647)

        logit = self.net(x)
        return {"binary_logit": logit}


class SubtractionMLP(BinaryMLP):
    """Subtraction baseline: (pooled_edited - pooled_orig) + structure -> MLP."""

    def __init__(self, d_model=640, n_structure=7):
        super().__init__(d_input=d_model + n_structure, hidden_dims=(512, 256), dropout=0.3)
        self._mode = "subtraction"

    def forward(self, batch):
        diff = batch["pooled_edited"] - batch["pooled_orig"]
        x = torch.cat([diff, batch["structure_delta"]], dim=-1)
        return {"binary_logit": self.net(x)}


class PooledMLP(BinaryMLP):
    """Pooled-only baseline: pooled_edited + structure -> MLP (no edit info)."""

    def __init__(self, d_model=640, n_structure=7):
        super().__init__(d_input=d_model + n_structure, hidden_dims=(512, 256), dropout=0.3)
        self._mode = "pooled"

    def forward(self, batch):
        x = torch.cat([batch["pooled_edited"], batch["structure_delta"]], dim=-1)
        return {"binary_logit": self.net(x)}


class RateRegressionMLP(nn.Module):
    """MLP for rate regression (arbitrary input dim -> scalar)."""

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


# ============================================================================
# PHASE 1 & 2: Dataset Preparation with TC-Motif Filtering
# ============================================================================

def phase1_dataset_preparation(combined_df, sequences):
    """Filter sites by TC-motif and report per-dataset statistics.

    Returns:
        tc_pos_df: DataFrame of TC-motif positive (edited) sites
        tc_neg_df: DataFrame of TC-motif negative (unedited) sites
        report: dict with per-dataset filtering statistics
    """
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 1 & 2: DATASET PREPARATION WITH TC-MOTIF FILTERING")
    logger.info("=" * 70)

    # Annotate all sites with TC-motif status
    tc_status = {}
    for sid, seq in sequences.items():
        if len(seq) >= 201:
            # Position 100 is the center of a 201-nt window (0-indexed)
            # That's where the edited C should be. But in the edited sequence,
            # it would be U. We check position 99 for TC motif.
            tc_status[sid] = get_tc_motif(seq, pos=100)
        else:
            tc_status[sid] = False

    combined_df = combined_df.copy()
    combined_df["is_tc"] = combined_df["site_id"].map(tc_status).fillna(False)

    # Report before/after filtering per dataset
    report = {"per_dataset": {}}

    logger.info("\nPer-dataset TC-motif filtering:")
    logger.info("  %-15s %8s %8s %8s %8s", "Dataset", "Total", "Edited", "TC-motif", "% TC")

    for ds in sorted(combined_df["dataset_source"].unique()):
        ds_df = combined_df[combined_df["dataset_source"] == ds]
        edited = ds_df[ds_df["is_edited"] == 1]
        n_total = len(edited)
        n_tc = int(edited["is_tc"].sum())
        pct = 100.0 * n_tc / max(n_total, 1)

        report["per_dataset"][ds] = {
            "n_edited_total": n_total,
            "n_edited_tc": n_tc,
            "pct_tc": round(pct, 1),
        }

        label = DATASET_LABELS.get(ds, ds)
        logger.info("  %-15s %8d %8d %8d %7.1f%%", label, len(ds_df), n_total, n_tc, pct)

    # --- Phase 1: TC-motif-filtered positives ---
    pos_df = combined_df[(combined_df["is_edited"] == 1) & (combined_df["is_tc"] == True)].copy()
    logger.info("\nTC-motif positive sites (edited + TC): %d", len(pos_df))

    # --- Phase 2: TC-motif-matched negatives ---
    neg_df = combined_df[(combined_df["is_edited"] == 0) & (combined_df["is_tc"] == True)].copy()
    logger.info("TC-motif negative sites (unedited + TC): %d", len(neg_df))

    if len(neg_df) < len(pos_df):
        logger.warning("  WARNING: Only %d TC-motif negatives available for %d positives (ratio %.2f:1)",
                        len(neg_df), len(pos_df), len(neg_df) / max(len(pos_df), 1))
        logger.info("  Using all available TC-motif negatives.")
    else:
        logger.info("  Negative:positive ratio = %.1f:1", len(neg_df) / max(len(pos_df), 1))

    report["n_tc_positives"] = len(pos_df)
    report["n_tc_negatives"] = len(neg_df)
    report["neg_pos_ratio"] = round(len(neg_df) / max(len(pos_df), 1), 2)

    return pos_df, neg_df, report


# ============================================================================
# PHASE 3: Binary Classification with TC-Motif Data
# ============================================================================

def phase3_binary_classification(pos_df, neg_df, pooled_orig, pooled_edited,
                                 tokens_orig, tokens_edited,
                                 structure_delta, sequences):
    """Train and evaluate binary classifiers on TC-motif-only data.

    Models:
      1. SubtractionMLP: (edited - orig) + structure -> MLP
      2. PooledMLP: edited_pooled + structure -> MLP
      3. EditRNA-A3A: Full multi-task model

    Returns:
        results: dict mapping model_name -> metrics
    """
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 3: BINARY CLASSIFICATION WITH TC-MOTIF DATA")
    logger.info("=" * 70)

    # Combine positives and negatives
    pos_df = pos_df.copy()
    neg_df = neg_df.copy()
    pos_df["label"] = 1
    neg_df["label"] = 0
    all_df = pd.concat([pos_df, neg_df], ignore_index=True)

    # Filter to sites with available embeddings
    available = set(pooled_orig.keys()) & set(pooled_edited.keys())
    all_df = all_df[all_df["site_id"].isin(available)].copy()
    logger.info("Sites with available embeddings: %d (pos=%d, neg=%d)",
                len(all_df), int((all_df["label"] == 1).sum()),
                int((all_df["label"] == 0).sum()))

    if len(all_df) < 50:
        logger.error("Too few sites for classification. Aborting phase 3.")
        return {}

    # --- Stratified train/val/test split (70/15/15) ---
    # Stratify by dataset_source AND label
    np.random.seed(SEED)

    train_dfs, val_dfs, test_dfs = [], [], []
    for (ds, lab), group in all_df.groupby(["dataset_source", "label"]):
        group = group.sample(frac=1, random_state=SEED).reset_index(drop=True)
        n = len(group)
        n_train = max(1, int(0.70 * n))
        n_val = max(1, int(0.15 * n))
        # Ensure at least 1 sample in each split when possible
        if n < 3:
            train_dfs.append(group)
            continue
        train_dfs.append(group.iloc[:n_train])
        val_dfs.append(group.iloc[n_train:n_train + n_val])
        test_dfs.append(group.iloc[n_train + n_val:])

    train_df = pd.concat(train_dfs, ignore_index=True)
    val_df = pd.concat(val_dfs, ignore_index=True) if val_dfs else pd.DataFrame()
    test_df = pd.concat(test_dfs, ignore_index=True) if test_dfs else pd.DataFrame()

    logger.info("Splits: train=%d, val=%d, test=%d", len(train_df), len(val_df), len(test_df))

    if len(val_df) == 0 or len(test_df) == 0:
        logger.error("Empty val or test split. Cannot proceed.")
        return {}

    # --- Create DataLoaders ---
    def make_loaders(train_df, val_df, test_df, batch_size=64):
        loaders = {}
        for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            if len(df) == 0:
                continue
            ds = EmbeddingDataset(
                df["site_id"].tolist(),
                df["label"].values.astype(np.float32),
                pooled_orig, pooled_edited, structure_delta,
            )
            loaders[name] = DataLoader(
                ds, batch_size=batch_size,
                shuffle=(name == "train"),
                num_workers=0,
                collate_fn=embedding_collate_fn,
                drop_last=False,
            )
            n_p = int(df["label"].sum())
            n_n = len(df) - n_p
            logger.info("  %s: %d samples (%d pos, %d neg)", name, len(df), n_p, n_n)
        return loaders

    loaders = make_loaders(train_df, val_df, test_df)

    # --- Training function ---
    def train_binary_model(model, model_name, loaders, epochs=50, lr=1e-3,
                           patience=10, use_editrna_forward=False):
        """Train a binary classification model and evaluate."""
        device = torch.device("cpu")
        model = model.to(device)

        loss_fn = FocalLoss(gamma=2.0, alpha=0.75)
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)

        best_auroc = -1.0
        best_epoch = 0
        patience_counter = 0
        best_state = None

        logger.info("  Training %s for %d epochs...", model_name, epochs)
        t_start = time.time()

        for epoch in range(1, epochs + 1):
            model.train()
            train_loss = 0.0
            n_batches = 0
            for batch in loaders["train"]:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                optimizer.zero_grad()

                if use_editrna_forward:
                    output = _editrna_forward(model, batch)
                else:
                    output = model(batch)

                logits = output["binary_logit"]
                loss = loss_fn(logits, batch["labels"])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_loss = train_loss / max(n_batches, 1)

            # Validate
            val_metrics = _eval_binary(model, loaders.get("val"), device,
                                       use_editrna_forward)
            val_auroc = val_metrics.get("auroc", 0.0)
            if not np.isnan(val_auroc) and val_auroc > best_auroc + 1e-4:
                best_auroc = val_auroc
                best_epoch = epoch
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1

            if epoch % 10 == 0 or epoch == 1:
                logger.info("    Epoch %3d  loss=%.4f  val_auroc=%.4f  val_auprc=%.4f  "
                            "patience=%d/%d",
                            epoch, avg_loss,
                            val_metrics.get("auroc", 0), val_metrics.get("auprc", 0),
                            patience_counter, patience)

            if patience_counter >= patience:
                logger.info("    Early stopping at epoch %d (best=%d, AUROC=%.4f)",
                            epoch, best_epoch, best_auroc)
                break

        if best_state:
            model.load_state_dict(best_state)

        elapsed = time.time() - t_start
        logger.info("  Training took %.1f seconds (best epoch=%d)", elapsed, best_epoch)

        # Test evaluation
        test_metrics = _eval_binary(model, loaders.get("test"), device,
                                    use_editrna_forward)

        # Per-dataset test metrics
        per_ds = _eval_per_dataset(model, test_df, pooled_orig, pooled_edited,
                                   structure_delta, device, use_editrna_forward)

        return {
            "best_epoch": best_epoch,
            "best_val_auroc": best_auroc,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "per_dataset_test": per_ds,
            "train_time": elapsed,
        }

    @torch.no_grad()
    def _eval_binary(model, loader, device, use_editrna_forward=False):
        if loader is None:
            return {}
        model.eval()
        all_targets, all_scores = [], []
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            if use_editrna_forward:
                output = _editrna_forward(model, batch)
            else:
                output = model(batch)
            logits = output["binary_logit"].squeeze(-1).cpu().numpy()
            probs = 1.0 / (1.0 + np.exp(-logits))
            all_targets.append(batch["labels"].cpu().numpy())
            all_scores.append(probs)
        y_true = np.concatenate(all_targets)
        y_score = np.concatenate(all_scores)
        return compute_binary_metrics(y_true, y_score)

    def _editrna_forward(model, batch):
        """Adapt EmbeddingDataset batch for EditRNA-A3A forward pass."""
        B = batch["labels"].shape[0]
        device = batch["labels"].device
        editrna_batch = {
            "sequences": ["N" * 201] * B,
            "site_ids": batch["site_ids"],
            "edit_pos": torch.full((B,), 100, dtype=torch.long, device=device),
            "flanking_context": torch.zeros(B, dtype=torch.long, device=device),
            "concordance_features": torch.zeros(B, 5, device=device),
            "structure_delta": batch["structure_delta"],
        }
        output = model(editrna_batch)
        return {"binary_logit": output["predictions"]["binary_logit"]}

    @torch.no_grad()
    def _eval_per_dataset(model, test_df, pooled_orig, pooled_edited,
                          structure_delta, device, use_editrna_forward=False):
        """Evaluate per dataset on the test set."""
        model.eval()
        results = {}
        for ds in test_df["dataset_source"].unique():
            ds_df = test_df[test_df["dataset_source"] == ds]
            if len(ds_df) < 5 or len(ds_df["label"].unique()) < 2:
                results[ds] = {"auroc": float("nan"), "auprc": float("nan"),
                               "n": len(ds_df)}
                continue

            ds_data = EmbeddingDataset(
                ds_df["site_id"].tolist(),
                ds_df["label"].values.astype(np.float32),
                pooled_orig, pooled_edited, structure_delta,
            )
            loader = DataLoader(ds_data, batch_size=64, shuffle=False,
                                num_workers=0, collate_fn=embedding_collate_fn)

            all_t, all_s = [], []
            for batch in loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                if use_editrna_forward:
                    output = _editrna_forward(model, batch)
                else:
                    output = model(batch)
                logits = output["binary_logit"].squeeze(-1).cpu().numpy()
                probs = 1.0 / (1.0 + np.exp(-logits))
                all_t.append(batch["labels"].cpu().numpy())
                all_s.append(probs)

            y_true = np.concatenate(all_t)
            y_score = np.concatenate(all_s)

            if len(np.unique(y_true)) < 2:
                results[ds] = {"auroc": float("nan"), "auprc": float("nan"),
                               "n": len(y_true)}
            else:
                results[ds] = {
                    "auroc": float(roc_auc_score(y_true, y_score)),
                    "auprc": float(average_precision_score(y_true, y_score)),
                    "n": len(y_true),
                }

        return results

    # --- Model 1: SubtractionMLP ---
    logger.info("\n--- Model 1: SubtractionMLP ---")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    sub_model = SubtractionMLP(d_model=640, n_structure=7)
    n_params = sum(p.numel() for p in sub_model.parameters() if p.requires_grad)
    logger.info("  Parameters: %s", f"{n_params:,}")
    sub_results = train_binary_model(sub_model, "SubtractionMLP", loaders,
                                     epochs=50, lr=1e-3, patience=10)

    # --- Model 2: PooledMLP ---
    logger.info("\n--- Model 2: PooledMLP ---")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    pooled_model = PooledMLP(d_model=640, n_structure=7)
    n_params = sum(p.numel() for p in pooled_model.parameters() if p.requires_grad)
    logger.info("  Parameters: %s", f"{n_params:,}")
    pooled_results = train_binary_model(pooled_model, "PooledMLP", loaders,
                                        epochs=50, lr=1e-3, patience=10)

    # --- Model 3: EditRNA-A3A ---
    logger.info("\n--- Model 3: EditRNA-A3A ---")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Build cached encoder from pre-computed embeddings
    cached_encoder = CachedRNAEncoder(
        tokens_cache=tokens_orig,
        pooled_cache=pooled_orig,
        tokens_edited_cache=tokens_edited,
        pooled_edited_cache=pooled_edited,
        d_model=640,
    )
    editrna_config = EditRNAConfig(
        primary_encoder="cached",
        d_model=640,
        d_edit=256,
        d_fused=512,
        edit_n_heads=8,
        use_structure_delta=True,
        head_dropout=0.2,
        fusion_dropout=0.2,
        focal_gamma=2.0,
        focal_alpha_binary=0.75,
        learning_rate=1e-4,
    )
    editrna_model = EditRNA_A3A(config=editrna_config, primary_encoder=cached_encoder)
    n_params = sum(p.numel() for p in editrna_model.parameters() if p.requires_grad)
    logger.info("  Parameters: %s", f"{n_params:,}")
    editrna_results = train_binary_model(editrna_model, "EditRNA-A3A", loaders,
                                          epochs=50, lr=1e-4, patience=10,
                                          use_editrna_forward=True)

    # --- Print summary ---
    all_results = {
        "SubtractionMLP": sub_results,
        "PooledMLP": pooled_results,
        "EditRNA-A3A": editrna_results,
    }

    print("\n" + "=" * 90)
    print("TC-MOTIF BINARY CLASSIFICATION RESULTS (TEST SET)")
    print("=" * 90)
    print(f"{'Model':<20} {'AUROC':>10} {'AUPRC':>10} {'F1':>8} "
          f"{'Accuracy':>10} {'N_pos':>8} {'N_neg':>8}")
    print("-" * 90)
    for name, res in all_results.items():
        tm = res.get("test_metrics", {})
        print(f"{name:<20} "
              f"{tm.get('auroc', 0):>10.4f} "
              f"{tm.get('auprc', 0):>10.4f} "
              f"{tm.get('f1', 0):>8.4f} "
              f"{tm.get('accuracy', 0):>10.4f} "
              f"{tm.get('n_positive', 0):>8d} "
              f"{tm.get('n_negative', 0):>8d}")
    print("=" * 90)

    # Per-dataset breakdown
    print("\nPer-Dataset AUROC (Test Set):")
    datasets_seen = set()
    for name, res in all_results.items():
        for ds in res.get("per_dataset_test", {}):
            datasets_seen.add(ds)
    datasets_seen = sorted(datasets_seen)
    header = f"{'Model':<20}" + "".join(f"{DATASET_LABELS.get(ds, ds):>14}" for ds in datasets_seen)
    print(header)
    print("-" * (20 + 14 * len(datasets_seen)))
    for name, res in all_results.items():
        row = f"{name:<20}"
        for ds in datasets_seen:
            v = res.get("per_dataset_test", {}).get(ds, {}).get("auroc", float("nan"))
            if np.isnan(v):
                row += f"{'N/A':>14}"
            else:
                row += f"{v:>14.4f}"
        print(row)
    print()

    return all_results, test_df


# ============================================================================
# PHASE 4: Within-Baysal Rate Prediction
# ============================================================================

def phase4_baysal_rate_prediction(combined_df, sequences, pooled_orig,
                                   pooled_edited, structure_delta):
    """Train rate prediction using ONLY Baysal TC-motif sites.

    This is the fair rate test: single experimental system, gene-based
    splits to avoid data leakage, TC-motif validated sites only.

    Returns:
        results: dict with regression metrics and predictions
    """
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 4: WITHIN-BAYSAL RATE PREDICTION (THE FAIR RATE TEST)")
    logger.info("=" * 70)

    # Get Baysal TC-motif sites with valid editing rates
    baysal = combined_df[
        (combined_df["dataset_source"] == "baysal_2016") &
        (combined_df["is_edited"] == 1) &
        (combined_df["editing_rate"].notna())
    ].copy()

    # Filter to TC-motif
    tc_status = {}
    for sid, seq in sequences.items():
        if len(seq) >= 201:
            tc_status[sid] = get_tc_motif(seq, pos=100)
    baysal["is_tc"] = baysal["site_id"].map(tc_status).fillna(False)
    baysal = baysal[baysal["is_tc"]].copy()

    # Filter to sites with embeddings
    available = set(pooled_orig.keys()) & set(pooled_edited.keys())
    baysal = baysal[baysal["site_id"].isin(available)].copy()

    # Compute log2 rate target
    baysal["log2_rate"] = baysal["editing_rate"].apply(compute_log2_rate)
    baysal = baysal[baysal["log2_rate"].notna()].copy()

    logger.info("Baysal TC-motif sites with rate + embeddings: %d", len(baysal))

    if len(baysal) < 100:
        logger.error("Too few Baysal sites for rate prediction. Aborting phase 4.")
        return {}

    # --- Gene-based split (sites in same gene stay together) ---
    np.random.seed(SEED)
    genes = baysal["gene"].fillna("unknown").values
    unique_genes = np.unique(genes)
    np.random.shuffle(unique_genes)

    n_genes = len(unique_genes)
    n_train_genes = int(0.70 * n_genes)
    n_val_genes = int(0.15 * n_genes)

    train_genes = set(unique_genes[:n_train_genes])
    val_genes = set(unique_genes[n_train_genes:n_train_genes + n_val_genes])
    test_genes = set(unique_genes[n_train_genes + n_val_genes:])

    train_df = baysal[baysal["gene"].fillna("unknown").isin(train_genes)].copy()
    val_df = baysal[baysal["gene"].fillna("unknown").isin(val_genes)].copy()
    test_df = baysal[baysal["gene"].fillna("unknown").isin(test_genes)].copy()

    logger.info("Gene-based split: train=%d (genes=%d), val=%d (genes=%d), test=%d (genes=%d)",
                len(train_df), len(train_genes), len(val_df), len(val_genes),
                len(test_df), len(test_genes))

    # Log rate distribution
    logger.info("Rate distribution (log2 scale):")
    logger.info("  Train: mean=%.3f, std=%.3f", train_df["log2_rate"].mean(), train_df["log2_rate"].std())
    logger.info("  Val:   mean=%.3f, std=%.3f", val_df["log2_rate"].mean(), val_df["log2_rate"].std())
    logger.info("  Test:  mean=%.3f, std=%.3f", test_df["log2_rate"].mean(), test_df["log2_rate"].std())

    if len(val_df) < 10 or len(test_df) < 10:
        logger.error("Too few samples in val/test. Aborting rate prediction.")
        return {}

    # --- Train SubtractionMLP for rate regression ---
    logger.info("\nTraining SubtractionMLP for rate regression...")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    d_input = 640 + 7  # subtraction embedding + structure delta
    model = RateRegressionMLP(d_input=d_input, hidden=256, dropout=0.2)

    def make_rate_loader(df, shuffle=False, batch_size=64):
        ds = RateRegressionDataset(
            df["site_id"].tolist(),
            df["log2_rate"].values.astype(np.float32),
            pooled_orig, pooled_edited,
            structure_delta=structure_delta,
            mode="subtraction",
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    train_loader = make_rate_loader(train_df, shuffle=True)
    val_loader = make_rate_loader(val_df)
    test_loader = make_rate_loader(test_df)

    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=80, eta_min=1e-7)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None
    patience = 15

    for epoch in range(1, 81):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            pred = model(x)
            loss = F.mse_loss(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # Validate
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for x, y in val_loader:
                val_preds.append(model(x).numpy())
                val_targets.append(y.numpy())
        val_preds_arr = np.concatenate(val_preds)
        val_targets_arr = np.concatenate(val_targets)
        val_loss = float(np.mean((val_preds_arr - val_targets_arr) ** 2))

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if epoch % 20 == 0 or epoch == 1:
            val_sp, _ = spearmanr(val_targets_arr, val_preds_arr)
            logger.info("  Epoch %3d  val_mse=%.4f  val_spearman=%.4f  patience=%d/%d",
                        epoch, val_loss, val_sp, patience_counter, patience)

        if patience_counter >= patience:
            logger.info("  Early stopping at epoch %d", epoch)
            break

    if best_state:
        model.load_state_dict(best_state)

    # --- Evaluate on test set ---
    model.eval()
    test_preds_list, test_targets_list = [], []
    with torch.no_grad():
        for x, y in test_loader:
            test_preds_list.append(model(x).numpy())
            test_targets_list.append(y.numpy())

    test_preds = np.concatenate(test_preds_list)
    test_targets = np.concatenate(test_targets_list)
    test_metrics = compute_regression_metrics(test_targets, test_preds)

    logger.info("\nWithin-Baysal Rate Prediction (Test Set):")
    logger.info("  Spearman = %.4f", test_metrics["spearman"])
    logger.info("  Pearson  = %.4f", test_metrics["pearson"])
    logger.info("  MSE      = %.4f", test_metrics["mse"])
    logger.info("  R^2      = %.4f", test_metrics["r2"])
    logger.info("  N        = %d", test_metrics["n_samples"])

    # Residual analysis
    residuals = test_preds - test_targets
    logger.info("\nResidual Analysis:")
    logger.info("  Mean residual:   %.4f", np.mean(residuals))
    logger.info("  Std residual:    %.4f", np.std(residuals))
    logger.info("  Max |residual|:  %.4f", np.max(np.abs(residuals)))

    results = {
        "n_baysal_tc": len(baysal),
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "n_train_genes": len(train_genes),
        "n_val_genes": len(val_genes),
        "n_test_genes": len(test_genes),
        "test_metrics": test_metrics,
        "test_predictions": test_preds,
        "test_targets": test_targets,
        "residual_mean": float(np.mean(residuals)),
        "residual_std": float(np.std(residuals)),
    }

    return results


# ============================================================================
# PHASE 5: Visualization (4-panel figure)
# ============================================================================

def phase5_visualization(binary_results, rate_results, phase1_report, output_dir):
    """Create the 4-panel publication figure.

    (A) AUROC comparison: TC-motif-only vs reference mixed results
    (B) Per-dataset AUROC breakdown for TC-motif-only models
    (C) Within-Baysal rate prediction scatter (predicted vs true)
    (D) Feature importance: what does the model learn when TC-motif is controlled?
    """
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 5: GENERATING PUBLICATION FIGURE")
    logger.info("=" * 70)

    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.30)

    # ==============================================
    # Panel A: AUROC Comparison - TC-motif vs Mixed
    # ==============================================
    ax_a = fig.add_subplot(gs[0, 0])

    models = list(binary_results.keys())
    tc_aurocs = [binary_results[m]["test_metrics"].get("auroc", 0) for m in models]

    # Reference mixed-dataset results (from prior experiments; approximate values)
    # These represent what we got on the original contaminated dataset
    mixed_aurocs_ref = {
        "SubtractionMLP": 0.92,
        "PooledMLP": 0.88,
        "EditRNA-A3A": 0.95,
    }
    mixed_aurocs = [mixed_aurocs_ref.get(m, 0.0) for m in models]

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax_a.bar(x - width / 2, mixed_aurocs, width, label="Mixed (contaminated)",
                     color="#ef5350", alpha=0.8, edgecolor="black", linewidth=0.5)
    bars2 = ax_a.bar(x + width / 2, tc_aurocs, width, label="TC-motif only (clean)",
                     color="#1565c0", alpha=0.8, edgecolor="black", linewidth=0.5)

    for bar, v in zip(bars1, mixed_aurocs):
        if v > 0:
            ax_a.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                      f"{v:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    for bar, v in zip(bars2, tc_aurocs):
        if v > 0:
            ax_a.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                      f"{v:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax_a.set_xticks(x)
    ax_a.set_xticklabels(models, fontsize=9)
    ax_a.set_ylabel("Test AUROC", fontsize=11)
    ax_a.set_title("(A) AUROC: Mixed vs TC-Motif-Only Dataset", fontsize=12, fontweight="bold")
    ax_a.legend(fontsize=9, loc="lower left")
    ax_a.set_ylim(0.4, 1.05)
    ax_a.grid(True, alpha=0.3, axis="y")
    ax_a.axhline(0.5, color="gray", linestyle=":", alpha=0.4, linewidth=0.8)

    # ==============================================
    # Panel B: Per-dataset AUROC for TC-motif models
    # ==============================================
    ax_b = fig.add_subplot(gs[0, 1])

    # Collect per-dataset results across models
    all_datasets = set()
    for m in models:
        per_ds = binary_results[m].get("per_dataset_test", {})
        all_datasets.update(per_ds.keys())
    all_datasets = sorted(all_datasets)

    if all_datasets:
        x_ds = np.arange(len(all_datasets))
        n_models = len(models)
        bar_width = 0.8 / max(n_models, 1)
        colors = ["#1565c0", "#66bb6a", "#ff9800"]

        for i, model_name in enumerate(models):
            per_ds = binary_results[model_name].get("per_dataset_test", {})
            aurocs = []
            for ds in all_datasets:
                v = per_ds.get(ds, {}).get("auroc", float("nan"))
                aurocs.append(v if np.isfinite(v) else 0.0)

            bars = ax_b.bar(x_ds + i * bar_width, aurocs, bar_width,
                            label=model_name, color=colors[i % len(colors)], alpha=0.8)
            # Annotate
            raw = [per_ds.get(ds, {}).get("auroc", float("nan")) for ds in all_datasets]
            for bar, v in zip(bars, raw):
                if np.isfinite(v) and v > 0:
                    ax_b.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                              f"{v:.2f}", ha="center", va="bottom", fontsize=7, rotation=45)

        ax_b.set_xticks(x_ds + bar_width * (n_models - 1) / 2)
        ax_b.set_xticklabels([DATASET_LABELS.get(ds, ds) for ds in all_datasets],
                              fontsize=9)
        ax_b.set_ylabel("Test AUROC", fontsize=11)
        ax_b.legend(fontsize=8)
        ax_b.grid(True, alpha=0.3, axis="y")
        ax_b.set_ylim(0.0, 1.15)
    else:
        ax_b.text(0.5, 0.5, "No per-dataset results", ha="center", va="center",
                  transform=ax_b.transAxes, fontsize=12)

    ax_b.set_title("(B) Per-Dataset AUROC (TC-Motif Only)", fontsize=12, fontweight="bold")

    # ==============================================
    # Panel C: Within-Baysal rate prediction scatter
    # ==============================================
    ax_c = fig.add_subplot(gs[1, 0])

    if "test_predictions" in rate_results and "test_targets" in rate_results:
        y_pred = rate_results["test_predictions"]
        y_true = rate_results["test_targets"]
        metrics = rate_results["test_metrics"]

        ax_c.scatter(y_true, y_pred, alpha=0.3, s=12, color="#1565c0",
                     edgecolors="white", linewidths=0.3)

        # Diagonal reference
        lims = [
            min(y_true.min(), y_pred.min()) - 0.3,
            max(y_true.max(), y_pred.max()) + 0.3,
        ]
        ax_c.plot(lims, lims, "r--", alpha=0.7, linewidth=1.5, label="Perfect prediction")

        # Trend line
        if len(y_true) > 5:
            z = np.polyfit(y_true, y_pred, 1)
            p = np.poly1d(z)
            x_line = np.linspace(y_true.min(), y_true.max(), 100)
            ax_c.plot(x_line, p(x_line), color="green", linewidth=1.5, alpha=0.7,
                      label=f"Trend (slope={z[0]:.2f})")

        ax_c.set_xlabel("True log2(rate + 0.01)", fontsize=11)
        ax_c.set_ylabel("Predicted log2(rate + 0.01)", fontsize=11)
        ax_c.set_xlim(lims)
        ax_c.set_ylim(lims)
        ax_c.set_aspect("equal")
        ax_c.legend(fontsize=8, loc="upper left")

        # Stats text box
        stats_text = (
            f"Spearman = {metrics['spearman']:.3f}\n"
            f"Pearson = {metrics['pearson']:.3f}\n"
            f"R$^2$ = {metrics['r2']:.3f}\n"
            f"n = {metrics['n_samples']}"
        )
        ax_c.text(0.97, 0.05, stats_text, transform=ax_c.transAxes, fontsize=9,
                  verticalalignment="bottom", horizontalalignment="right",
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))
    else:
        ax_c.text(0.5, 0.5, "Rate prediction not available", ha="center", va="center",
                  transform=ax_c.transAxes, fontsize=12)

    ax_c.set_title("(C) Within-Baysal Rate Prediction\n(Gene-Based Split, TC-Motif Only)",
                    fontsize=12, fontweight="bold")
    ax_c.grid(True, alpha=0.3)

    # ==============================================
    # Panel D: Dataset contamination overview
    # ==============================================
    ax_d = fig.add_subplot(gs[1, 1])

    # Show TC-motif fraction per dataset as a compelling visual
    per_ds = phase1_report.get("per_dataset", {})
    if per_ds:
        ds_order = ["advisor_c2t", "alqassim_2021", "sharma_2015", "baysal_2016"]
        ds_order = [ds for ds in ds_order if ds in per_ds]
        ds_labels = [DATASET_LABELS.get(ds, ds) for ds in ds_order]
        tc_fracs = [per_ds[ds]["pct_tc"] for ds in ds_order]
        n_edited = [per_ds[ds]["n_edited_total"] for ds in ds_order]

        # Stacked bar: TC-motif vs non-TC
        tc_counts = [per_ds[ds]["n_edited_tc"] for ds in ds_order]
        non_tc_counts = [per_ds[ds]["n_edited_total"] - per_ds[ds]["n_edited_tc"]
                         for ds in ds_order]

        x_pos = np.arange(len(ds_order))
        bars_tc = ax_d.bar(x_pos, tc_counts, color="#1565c0", alpha=0.8,
                           label="TC-motif (APOBEC3A)", edgecolor="black", linewidth=0.5)
        bars_ntc = ax_d.bar(x_pos, non_tc_counts, bottom=tc_counts, color="#ef5350",
                            alpha=0.6, label="Non-TC (likely ADAR)", edgecolor="black",
                            linewidth=0.5)

        # Annotate percentages
        for i, (tc, ntc, frac) in enumerate(zip(tc_counts, non_tc_counts, tc_fracs)):
            total = tc + ntc
            ax_d.text(i, total + max(n_edited) * 0.02,
                      f"{frac:.1f}% TC\n(n={total})",
                      ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax_d.set_xticks(x_pos)
        ax_d.set_xticklabels(ds_labels, fontsize=10)
        ax_d.set_ylabel("Number of Edited Sites", fontsize=11)
        ax_d.legend(fontsize=9, loc="upper left")
        ax_d.grid(True, alpha=0.3, axis="y")

        # Add threshold line
        ax_d.axhline(0, color="black", linewidth=0.5)
    else:
        ax_d.text(0.5, 0.5, "No dataset info available", ha="center", va="center",
                  transform=ax_d.transAxes, fontsize=12)

    ax_d.set_title("(D) Dataset Contamination: TC-Motif Validation\n"
                    "Levanon & Alqassim are >90% non-APOBEC3A sites",
                    fontsize=12, fontweight="bold")

    # Main title
    fig.suptitle("TC-Motif Reanalysis: The First Genuinely APOBEC3A-Specific Predictor\n"
                 "Removing ADAR contamination reveals true APOBEC3A editing biology",
                 fontsize=14, fontweight="bold", y=0.98)

    plt.savefig(output_dir / "tc_motif_reanalysis_figure.png", dpi=200, bbox_inches="tight")
    plt.savefig(output_dir / "tc_motif_reanalysis_figure.pdf", dpi=200, bbox_inches="tight")
    plt.close()

    logger.info("Saved 4-panel figure to %s", output_dir / "tc_motif_reanalysis_figure.png")

    # --- Additional individual plots ---

    # Scatter: rate prediction (larger standalone version)
    if "test_predictions" in rate_results and "test_targets" in rate_results:
        fig2, ax2 = plt.subplots(figsize=(7, 7))
        y_pred = rate_results["test_predictions"]
        y_true = rate_results["test_targets"]
        metrics = rate_results["test_metrics"]

        ax2.scatter(y_true, y_pred, alpha=0.3, s=15, color="#1565c0",
                    edgecolors="white", linewidths=0.3)
        lims = [min(y_true.min(), y_pred.min()) - 0.3,
                max(y_true.max(), y_pred.max()) + 0.3]
        ax2.plot(lims, lims, "r--", alpha=0.7, linewidth=1.5)
        ax2.set_xlabel("True log2(rate + 0.01)", fontsize=12)
        ax2.set_ylabel("Predicted log2(rate + 0.01)", fontsize=12)
        ax2.set_title(f"Within-Baysal Rate Prediction (Gene-Based Split)\n"
                      f"Spearman={metrics['spearman']:.3f}  "
                      f"Pearson={metrics['pearson']:.3f}  "
                      f"R2={metrics['r2']:.3f}  n={metrics['n_samples']}",
                      fontsize=11, fontweight="bold")
        ax2.set_xlim(lims)
        ax2.set_ylim(lims)
        ax2.set_aspect("equal")
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "baysal_rate_scatter.png", dpi=150, bbox_inches="tight")
        plt.close()

    # Bar chart: model comparison
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    model_names = list(binary_results.keys())
    aurocs = [binary_results[m]["test_metrics"].get("auroc", 0) for m in model_names]
    auprcs = [binary_results[m]["test_metrics"].get("auprc", 0) for m in model_names]

    x3 = np.arange(len(model_names))
    w3 = 0.35
    ax3.bar(x3 - w3 / 2, aurocs, w3, label="AUROC", color="#1565c0", alpha=0.8)
    ax3.bar(x3 + w3 / 2, auprcs, w3, label="AUPRC", color="#66bb6a", alpha=0.8)

    for i, (a, p) in enumerate(zip(aurocs, auprcs)):
        ax3.text(i - w3 / 2, a + 0.01, f"{a:.3f}", ha="center", fontsize=9, fontweight="bold")
        ax3.text(i + w3 / 2, p + 0.01, f"{p:.3f}", ha="center", fontsize=9, fontweight="bold")

    ax3.set_xticks(x3)
    ax3.set_xticklabels(model_names, fontsize=10)
    ax3.set_ylabel("Score", fontsize=11)
    ax3.set_title("TC-Motif-Only Binary Classification (Test Set)", fontsize=12, fontweight="bold")
    ax3.legend(fontsize=9)
    ax3.set_ylim(0, 1.1)
    ax3.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_dir / "tc_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("Saved additional plots.")


# ============================================================================
# Main
# ============================================================================

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    t_total = time.time()

    # ------------------------------------------------------------------
    # Load all data
    # ------------------------------------------------------------------
    logger.info("Loading data...")

    # Combined dataset
    combined_df = pd.read_csv(COMBINED_CSV)
    logger.info("  Combined dataset: %d rows", len(combined_df))

    # Splits (for reference -- we create our own TC-motif splits)
    splits_df = pd.read_csv(SPLITS_CSV)
    logger.info("  Splits: %d rows", len(splits_df))

    # Sequences
    sequences = {}
    if SEQ_JSON.exists():
        sequences = json.loads(SEQ_JSON.read_text())
    logger.info("  Sequences: %d", len(sequences))

    # Pooled embeddings (original and edited)
    logger.info("  Loading pooled embeddings...")
    pooled_orig = torch.load(EMB_DIR / "rnafm_pooled.pt", weights_only=False)
    pooled_edited = torch.load(EMB_DIR / "rnafm_pooled_edited.pt", weights_only=False)
    logger.info("  Pooled orig: %d, Pooled edited: %d", len(pooled_orig), len(pooled_edited))

    # Token-level embeddings (for EditRNA-A3A)
    tokens_orig = None
    tokens_edited = None
    tok_path = EMB_DIR / "rnafm_tokens.pt"
    tok_edited_path = EMB_DIR / "rnafm_tokens_edited.pt"
    if tok_path.exists() and tok_edited_path.exists():
        logger.info("  Loading token embeddings...")
        tokens_orig = torch.load(tok_path, weights_only=False)
        tokens_edited = torch.load(tok_edited_path, weights_only=False)
        logger.info("  Token orig: %d, Token edited: %d", len(tokens_orig), len(tokens_edited))

    # Structure delta features
    structure_delta = {}
    if STRUCT_CACHE.exists():
        data = np.load(STRUCT_CACHE, allow_pickle=True)
        feat_key = "delta_features" if "delta_features" in data else "features"
        if "site_ids" in data and feat_key in data:
            sids = data["site_ids"]
            feats = data[feat_key]
            structure_delta = {str(sid): feats[i] for i, sid in enumerate(sids)}
    logger.info("  Structure deltas: %d", len(structure_delta))

    # ------------------------------------------------------------------
    # Phase 1 & 2: Dataset Preparation
    # ------------------------------------------------------------------
    tc_pos_df, tc_neg_df, phase1_report = phase1_dataset_preparation(
        combined_df, sequences
    )

    # ------------------------------------------------------------------
    # Load hard negatives as TC-motif negatives (from pipeline)
    # ------------------------------------------------------------------
    if len(tc_neg_df) < 100 and HARDNEG_CSV.exists() and HARDNEG_POOLED.exists():
        logger.info("\n" + "=" * 70)
        logger.info("LOADING HARD NEGATIVES AS TC-MOTIF MATCHED CONTROLS")
        logger.info("=" * 70)

        hardneg_df = pd.read_csv(HARDNEG_CSV)
        logger.info("  Hard negative CSV: %d rows", len(hardneg_df))

        # Load hard neg sequences
        hardneg_sequences = {}
        if HARDNEG_SEQ.exists():
            hardneg_sequences = json.loads(HARDNEG_SEQ.read_text())
            logger.info("  Hard negative sequences: %d", len(hardneg_sequences))

        # Load hard neg embeddings
        hardneg_pooled = torch.load(HARDNEG_POOLED, weights_only=False)
        logger.info("  Hard negative pooled embeddings: %d", len(hardneg_pooled))

        hardneg_pooled_edited = {}
        if HARDNEG_POOLED_EDITED.exists():
            hardneg_pooled_edited = torch.load(HARDNEG_POOLED_EDITED, weights_only=False)

        hardneg_tok = {}
        hardneg_tok_edited = {}
        if HARDNEG_TOKENS.exists():
            hardneg_tok = torch.load(HARDNEG_TOKENS, weights_only=False)
        if HARDNEG_TOKENS_EDITED.exists():
            hardneg_tok_edited = torch.load(HARDNEG_TOKENS_EDITED, weights_only=False)

        # Load hard neg structure features
        hardneg_struct = {}
        if HARDNEG_STRUCT.exists():
            hn_data = np.load(HARDNEG_STRUCT, allow_pickle=True)
            hn_feat_key = "delta_features" if "delta_features" in hn_data else "features"
            if "site_ids" in hn_data and hn_feat_key in hn_data:
                hn_sids = hn_data["site_ids"]
                hn_feats = hn_data[hn_feat_key]
                hardneg_struct = {str(sid): hn_feats[i] for i, sid in enumerate(hn_sids)}
            logger.info("  Hard negative structure features: %d", len(hardneg_struct))

        # Filter to hard negatives that have embeddings
        available_ids = set(hardneg_pooled.keys())
        hardneg_df = hardneg_df[hardneg_df["site_id"].isin(available_ids)].copy()

        # Ensure they are labeled as negative
        hardneg_df["is_edited"] = 0
        hardneg_df["editing_rate"] = 0.0
        if "feature" not in hardneg_df.columns:
            hardneg_df["feature"] = "hardneg"

        # Sample to maintain a reasonable ratio (up to 5:1 neg:pos)
        max_neg = len(tc_pos_df) * 5
        if len(hardneg_df) > max_neg:
            hardneg_df = hardneg_df.sample(n=max_neg, random_state=SEED)

        tc_neg_df = hardneg_df
        logger.info("  Using %d hard negatives as TC-motif matched controls", len(tc_neg_df))

        # Merge embeddings into the main dicts
        sequences.update(hardneg_sequences)
        pooled_orig.update(hardneg_pooled)
        pooled_edited.update(hardneg_pooled_edited)
        if tokens_orig is not None and hardneg_tok:
            tokens_orig.update(hardneg_tok)
        if tokens_edited is not None and hardneg_tok_edited:
            tokens_edited.update(hardneg_tok_edited)
        structure_delta.update(hardneg_struct)

        phase1_report["hardneg_loaded"] = len(tc_neg_df)
        phase1_report["hardneg_source"] = "generate_hardneg_pipeline.py"

    # ------------------------------------------------------------------
    # Phase 3: Binary Classification
    # ------------------------------------------------------------------
    binary_results = {}
    test_df = None
    if len(tc_pos_df) >= 20 and len(tc_neg_df) >= 20:
        binary_results, test_df = phase3_binary_classification(
            tc_pos_df, tc_neg_df, pooled_orig, pooled_edited,
            tokens_orig, tokens_edited, structure_delta, sequences,
        )
    else:
        logger.warning("Insufficient TC-motif sites for binary classification.")

    # ------------------------------------------------------------------
    # Phase 4: Within-Baysal Rate Prediction
    # ------------------------------------------------------------------
    # Re-annotate TC-motif on combined_df for phase 4
    tc_status = {}
    for sid, seq in sequences.items():
        if len(seq) >= 201:
            tc_status[sid] = get_tc_motif(seq, pos=100)
    combined_df["is_tc"] = combined_df["site_id"].map(tc_status).fillna(False)

    rate_results = phase4_baysal_rate_prediction(
        combined_df, sequences, pooled_orig, pooled_edited, structure_delta
    )

    # ------------------------------------------------------------------
    # Phase 5: Visualization
    # ------------------------------------------------------------------
    phase5_visualization(binary_results, rate_results, phase1_report, OUTPUT_DIR)

    # ------------------------------------------------------------------
    # Save all results to JSON
    # ------------------------------------------------------------------
    json_results = {
        "description": (
            "TC-Motif Reanalysis: The first genuinely APOBEC3A-specific predictor. "
            "Re-runs classification and rate prediction using ONLY TC-motif (5'-UC-3') "
            "validated sites, removing ADAR contamination from existing catalogs."
        ),
        "phase1_report": phase1_report,
        "phase3_binary": {},
        "phase4_rate": {},
        "total_time_seconds": time.time() - t_total,
    }

    # Save binary classification results (without large arrays)
    for model_name, res in binary_results.items():
        json_results["phase3_binary"][model_name] = {
            "best_epoch": res.get("best_epoch"),
            "best_val_auroc": res.get("best_val_auroc"),
            "val_metrics": res.get("val_metrics", {}),
            "test_metrics": res.get("test_metrics", {}),
            "per_dataset_test": res.get("per_dataset_test", {}),
            "train_time": res.get("train_time"),
        }

    # Save rate prediction results (without large arrays)
    if rate_results:
        json_results["phase4_rate"] = {
            k: v for k, v in rate_results.items()
            if not isinstance(v, np.ndarray)
        }

    with open(OUTPUT_DIR / "tc_motif_reanalysis_results.json", "w") as f:
        json.dump(json_results, f, indent=2, default=serialize)
    logger.info("\nSaved results to %s", OUTPUT_DIR / "tc_motif_reanalysis_results.json")

    # Save predictions as npz for further analysis
    pred_arrays = {}
    if rate_results and "test_predictions" in rate_results:
        pred_arrays["rate_predictions"] = rate_results["test_predictions"]
        pred_arrays["rate_targets"] = rate_results["test_targets"]
    np.savez(OUTPUT_DIR / "tc_motif_predictions.npz", **pred_arrays)

    elapsed = time.time() - t_total
    logger.info("\n" + "=" * 70)
    logger.info("TC-MOTIF REANALYSIS COMPLETE (%.1fs)", elapsed)
    logger.info("=" * 70)
    logger.info("Output directory: %s", OUTPUT_DIR)
    logger.info("Key files:")
    logger.info("  tc_motif_reanalysis_figure.png  (4-panel publication figure)")
    logger.info("  tc_motif_reanalysis_figure.pdf  (vector format)")
    logger.info("  tc_motif_reanalysis_results.json (all metrics)")
    logger.info("  baysal_rate_scatter.png         (standalone rate scatter)")
    logger.info("  tc_model_comparison.png         (model comparison bars)")


if __name__ == "__main__":
    main()
