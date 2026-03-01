#!/usr/bin/env python
"""Multi-task comparison: binary-only vs regression-only vs multi-task.

Compares three training modes for APOBEC C-to-U editing prediction using
the subtraction approach (diff = pooled_edited - pooled_orig, 640-dim):

1. binary-only:      MLP with focal loss, outputs P(edited)
2. regression-only:  MLP with MSE loss on log2(editing_rate), positive sites only
3. multi-task:       Shared backbone MLP with 2 heads (binary + regression), joint loss

For EACH mode, evaluates on BOTH tasks:
  - Binary metrics: AUROC, AUPRC, F1
  - Regression metrics: Pearson r, Spearman rho on positive test sites with rates

Usage:
    python experiments/apobec3a/exp_multitask_comparison.py
"""

import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logger = logging.getLogger(__name__)

EMB_DIR = PROJECT_ROOT / "data" / "processed" / "embeddings"
SPLITS_CSV = PROJECT_ROOT / "data" / "processed" / "splits_expanded.csv"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "apobec" / "outputs" / "multitask_comparison"

DATASET_LABELS = {
    "advisor_c2t": "Levanon",
    "asaoka_2019": "Asaoka",
    "sharma_2015": "Sharma",
    "alqassim_2021": "Alqassim",
    "baysal_2016": "Baysal",
}

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class BinaryPredictor(nn.Module):
    """MLP for binary editing prediction from subtraction embedding."""

    def __init__(self, d_model=640, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, diff):
        return self.net(diff).squeeze(-1)


class RegressionPredictor(nn.Module):
    """MLP for editing rate prediction from subtraction embedding."""

    def __init__(self, d_model=640, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, diff):
        return self.net(diff).squeeze(-1)


class MultiTaskPredictor(nn.Module):
    """Shared backbone with binary and regression heads."""

    def __init__(self, d_model=640, hidden=256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.binary_head = nn.Linear(hidden // 2, 1)
        self.rate_head = nn.Linear(hidden // 2, 1)

    def forward(self, diff):
        features = self.backbone(diff)
        binary_logit = self.binary_head(features).squeeze(-1)
        rate_pred = self.rate_head(features).squeeze(-1)
        return binary_logit, rate_pred


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.75):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        targets = targets.float()
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        return (focal_weight * bce).mean()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MultiTaskDataset(Dataset):
    """Dataset that serves subtraction diffs with binary labels and optional rates."""

    def __init__(self, site_ids, labels, rates, pooled_orig, pooled_edited):
        self.site_ids = site_ids
        self.labels = labels
        self.rates = rates  # NaN for negatives or sites without rate
        self.pooled_orig = pooled_orig
        self.pooled_edited = pooled_edited

    def __len__(self):
        return len(self.site_ids)

    def __getitem__(self, idx):
        sid = self.site_ids[idx]
        diff = self.pooled_edited[sid] - self.pooled_orig[sid]
        return {
            "diff": diff,
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
            "rate": torch.tensor(self.rates[idx], dtype=torch.float32),
            "has_rate": torch.tensor(not np.isnan(self.rates[idx]), dtype=torch.bool),
        }


def collate_fn(batch):
    return {
        "diff": torch.stack([b["diff"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "rate": torch.stack([b["rate"] for b in batch]),
        "has_rate": torch.stack([b["has_rate"] for b in batch]),
    }


# ---------------------------------------------------------------------------
# Training functions
# ---------------------------------------------------------------------------

def train_binary(model, train_loader, val_loader, epochs=50, lr=1e-3, patience=10):
    """Train binary-only model with focal loss."""
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)
    loss_fn = FocalLoss()

    best_auroc = -1.0
    patience_counter = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            logits = model(batch["diff"])
            loss = loss_fn(logits, batch["label"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # Validate
        val_auroc = _eval_binary_auroc(model, val_loader, mode="binary")
        if val_auroc > best_auroc + 1e-4:
            best_auroc = val_auroc
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break

    if best_state:
        model.load_state_dict(best_state)
    return model


def train_regression(model, train_loader, val_loader, epochs=50, lr=1e-3, patience=10):
    """Train regression-only model on positive sites with valid rates."""
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            mask = batch["has_rate"]
            if mask.sum() == 0:
                continue
            optimizer.zero_grad()
            pred = model(batch["diff"][mask])
            loss = F.mse_loss(pred, batch["rate"][mask])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # Validate
        val_loss = _eval_regression_loss(model, val_loader)
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


def train_multitask(model, train_loader, val_loader, lambda_rate=0.5,
                    epochs=50, lr=1e-3, patience=10):
    """Train multi-task model with joint binary + regression loss."""
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)
    focal_loss = FocalLoss()

    best_auroc = -1.0
    patience_counter = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            binary_logit, rate_pred = model(batch["diff"])

            # Binary loss on all samples
            b_loss = focal_loss(binary_logit, batch["label"])

            # Regression loss on positive sites with valid rates only
            rate_mask = batch["has_rate"]
            if rate_mask.sum() > 0:
                r_loss = F.mse_loss(rate_pred[rate_mask], batch["rate"][rate_mask])
            else:
                r_loss = torch.tensor(0.0)

            loss = b_loss + lambda_rate * r_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # Validate on binary AUROC
        val_auroc = _eval_binary_auroc(model, val_loader, mode="multitask")
        if val_auroc > best_auroc + 1e-4:
            best_auroc = val_auroc
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
# Evaluation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def _eval_binary_auroc(model, loader, mode="binary"):
    model.eval()
    all_labels, all_scores = [], []
    for batch in loader:
        if mode == "binary":
            logits = model(batch["diff"])
        elif mode == "multitask":
            logits, _ = model(batch["diff"])
        elif mode == "regression":
            logits = model(batch["diff"])
        probs = torch.sigmoid(logits).cpu().numpy()
        all_labels.append(batch["label"].numpy())
        all_scores.append(probs)
    y_true = np.concatenate(all_labels)
    y_score = np.concatenate(all_scores)
    if len(np.unique(y_true)) < 2:
        return 0.0
    return roc_auc_score(y_true, y_score)


@torch.no_grad()
def _eval_regression_loss(model, loader):
    model.eval()
    total_loss, n = 0.0, 0
    for batch in loader:
        mask = batch["has_rate"]
        if mask.sum() == 0:
            continue
        pred = model(batch["diff"][mask])
        total_loss += F.mse_loss(pred, batch["rate"][mask], reduction="sum").item()
        n += mask.sum().item()
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate_all(model, test_loader, mode, test_df_with_rates):
    """Evaluate a model on both binary and regression tasks.

    Args:
        model: trained model
        test_loader: DataLoader for full test set (positives + negatives)
        mode: "binary", "regression", or "multitask"
        test_df_with_rates: DataFrame of positive test sites with valid rates
    """
    model.eval()

    # Collect predictions on entire test set
    all_labels, all_binary_scores, all_rate_preds = [], [], []
    all_diffs = []

    for batch in test_loader:
        diff = batch["diff"]
        if mode == "binary":
            logits = model(diff)
            binary_scores = torch.sigmoid(logits).cpu().numpy()
            rate_preds = binary_scores  # use P(edited) as rate proxy
        elif mode == "regression":
            preds = model(diff)
            rate_preds = preds.cpu().numpy()
            binary_scores = preds.cpu().numpy()  # use predicted rate as binary score
        elif mode == "multitask":
            binary_logit, rate_pred = model(diff)
            binary_scores = torch.sigmoid(binary_logit).cpu().numpy()
            rate_preds = rate_pred.cpu().numpy()

        all_labels.append(batch["label"].numpy())
        all_binary_scores.append(binary_scores)
        all_rate_preds.append(rate_preds)

    y_true = np.concatenate(all_labels)
    y_binary_score = np.concatenate(all_binary_scores)
    y_rate_pred = np.concatenate(all_rate_preds)

    # Binary metrics on full test set
    binary_metrics = _compute_binary_metrics(y_true, y_binary_score)

    # Regression metrics on positive test sites with valid rates
    regression_metrics = {"pearson_r": float("nan"), "spearman_r": float("nan"),
                          "pearson_p": float("nan"), "spearman_p": float("nan"),
                          "n_rate_samples": 0}

    if len(test_df_with_rates) >= 5:
        # We need to get rate predictions for the specific positive sites
        # Re-run model on rate-only subset
        rate_site_ids = set(test_df_with_rates["site_id"].values)
        rate_true = []
        rate_pred = []

        for batch_idx, batch in enumerate(test_loader):
            pass  # we already collected all preds

        # Use the full-test predictions but filter to rate sites
        # Need to rebuild the mapping: collect site_ids from loader
        all_site_ids = []
        for batch in test_loader:
            # We don't have site_ids in our collate, so reconstruct from dataset
            pass

        # Simpler approach: run directly on the rate subset
        rate_preds_sub, rate_true_sub = _predict_rates_for_subset(
            model, mode, test_df_with_rates,
            test_loader.dataset.pooled_orig,
            test_loader.dataset.pooled_edited,
        )

        if len(rate_true_sub) >= 5:
            pr, pp = pearsonr(rate_true_sub, rate_preds_sub)
            sr, sp = spearmanr(rate_true_sub, rate_preds_sub)
            regression_metrics = {
                "pearson_r": float(pr),
                "pearson_p": float(pp),
                "spearman_r": float(sr),
                "spearman_p": float(sp),
                "rmse": float(np.sqrt(np.mean((rate_true_sub - rate_preds_sub) ** 2))),
                "n_rate_samples": len(rate_true_sub),
            }

    return {**binary_metrics, **regression_metrics}


@torch.no_grad()
def _predict_rates_for_subset(model, mode, df, pooled_orig, pooled_edited):
    """Get predictions for a subset of sites defined by df."""
    model.eval()
    true_rates = []
    pred_rates = []

    for _, row in df.iterrows():
        sid = row["site_id"]
        if sid not in pooled_orig or sid not in pooled_edited:
            continue
        diff = (pooled_edited[sid] - pooled_orig[sid]).unsqueeze(0)
        if mode == "binary":
            logit = model(diff)
            pred = torch.sigmoid(logit).item()
        elif mode == "regression":
            pred = model(diff).item()
        elif mode == "multitask":
            _, rate_pred = model(diff)
            pred = rate_pred.item()

        true_rates.append(row["log2_rate"])
        pred_rates.append(pred)

    return np.array(pred_rates), np.array(true_rates)


def _compute_binary_metrics(y_true, y_score):
    mask = ~(np.isnan(y_true) | np.isnan(y_score))
    y_true, y_score = y_true[mask], y_score[mask]

    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan"), "f1": float("nan")}

    auroc = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)

    prec_arr, rec_arr, thresholds = precision_recall_curve(y_true, y_score)
    f1_arr = 2 * prec_arr[:-1] * rec_arr[:-1] / (prec_arr[:-1] + rec_arr[:-1] + 1e-8)
    best_idx = np.argmax(f1_arr)
    threshold = thresholds[best_idx] if len(thresholds) > 0 else 0.5
    y_pred = (y_score >= threshold).astype(int)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        "auroc": float(auroc),
        "auprc": float(auprc),
        "f1": float(f1),
        "n_test": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "n_negative": int(len(y_true) - y_true.sum()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
})

ALL_DATASETS = ["advisor_c2t", "asaoka_2019", "alqassim_2021", "sharma_2015", "baysal_2016"]


def _evaluate_per_dataset(model, mode, df, pooled_orig, pooled_edited):
    """Evaluate a model per dataset source. Returns dict of {dataset_label: metrics}."""
    per_ds = {}
    for ds_key, ds_label in DATASET_LABELS.items():
        ds_test = df[(df["split"] == "test") & (df["dataset_source"] == ds_key)]
        if len(ds_test) < 5:
            continue

        test_ds = MultiTaskDataset(
            site_ids=ds_test["site_id"].tolist(),
            labels=ds_test["label"].values.astype(np.float32),
            rates=ds_test["log2_rate"].values.astype(np.float32),
            pooled_orig=pooled_orig,
            pooled_edited=pooled_edited,
        )
        loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0,
                            collate_fn=collate_fn, drop_last=False)

        all_labels, all_scores = [], []
        model.eval()
        with torch.no_grad():
            for batch in loader:
                if mode == "binary":
                    logits = model(batch["diff"])
                    scores = torch.sigmoid(logits).cpu().numpy()
                elif mode == "regression":
                    scores = model(batch["diff"]).cpu().numpy()
                elif mode == "multitask":
                    logits, _ = model(batch["diff"])
                    scores = torch.sigmoid(logits).cpu().numpy()
                all_labels.append(batch["label"].numpy())
                all_scores.append(scores)

        y_true = np.concatenate(all_labels)
        y_score = np.concatenate(all_scores)
        metrics = _compute_binary_metrics(y_true, y_score)

        # Regression metrics for this dataset
        ds_rate_df = ds_test[(ds_test["label"] == 1) & ds_test["log2_rate"].notna()]
        if len(ds_rate_df) >= 5:
            rate_preds, rate_true = _predict_rates_for_subset(
                model, mode, ds_rate_df, pooled_orig, pooled_edited
            )
            if len(rate_true) >= 5:
                pr, _ = pearsonr(rate_true, rate_preds)
                sr, _ = spearmanr(rate_true, rate_preds)
                metrics["pearson_r"] = float(pr)
                metrics["spearman_r"] = float(sr)
                metrics["n_rate_samples"] = len(rate_true)

        per_ds[ds_label] = metrics
    return per_ds


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    torch.manual_seed(42)

    # Load data (pooled only - no token embeddings needed)
    logger.info("Loading pooled embeddings...")
    pooled_orig = torch.load(EMB_DIR / "rnafm_pooled.pt", weights_only=False)
    pooled_edited = torch.load(EMB_DIR / "rnafm_pooled_edited.pt", weights_only=False)
    splits_df = pd.read_csv(SPLITS_CSV)

    available_ids = set(pooled_orig.keys()) & set(pooled_edited.keys())
    splits_df = splits_df[splits_df["site_id"].isin(available_ids)].copy()

    # Prepare log2_rate for all sites (NaN for negatives/missing)
    splits_df["editing_rate"] = pd.to_numeric(splits_df["editing_rate"], errors="coerce")
    splits_df["log2_rate"] = np.nan
    rate_mask = (splits_df["label"] == 1) & (splits_df["editing_rate"] > 0) & splits_df["editing_rate"].notna()
    splits_df.loc[rate_mask, "log2_rate"] = np.log2(splits_df.loc[rate_mask, "editing_rate"].clip(lower=0.01))

    logger.info("Total sites: %d, with valid rates: %d", len(splits_df), rate_mask.sum())

    # Apply neg ratio (5:1)
    pos_df = splits_df[splits_df["label"] == 1]
    neg_df = splits_df[splits_df["label"] == 0]
    max_neg = len(pos_df) * 5
    if len(neg_df) > max_neg:
        neg_df = neg_df.sample(n=max_neg, random_state=42)
    df = pd.concat([pos_df, neg_df], ignore_index=True)

    # Create dataloaders
    def make_loader(split_name, shuffle=False):
        subset = df[df["split"] == split_name]
        ds = MultiTaskDataset(
            site_ids=subset["site_id"].tolist(),
            labels=subset["label"].values.astype(np.float32),
            rates=subset["log2_rate"].values.astype(np.float32),
            pooled_orig=pooled_orig,
            pooled_edited=pooled_edited,
        )
        return DataLoader(ds, batch_size=64, shuffle=shuffle, num_workers=0,
                          collate_fn=collate_fn, drop_last=False)

    train_loader = make_loader("train", shuffle=True)
    val_loader = make_loader("val")
    test_loader = make_loader("test")

    # Positive test sites with valid rates for regression eval
    test_df = df[(df["split"] == "test") & (df["label"] == 1) & df["log2_rate"].notna()].copy()
    logger.info("Test: %d total, %d positive with rates",
                len(df[df["split"] == "test"]), len(test_df))

    all_results = {}

    # ===================================================================
    # Mode 1: Binary-only
    # ===================================================================
    logger.info("\n" + "=" * 70)
    logger.info("MODE 1: Binary-only")
    logger.info("=" * 70)

    model_bin = BinaryPredictor()
    n_params = sum(p.numel() for p in model_bin.parameters())
    logger.info("Parameters: %d", n_params)

    t0 = time.time()
    model_bin = train_binary(model_bin, train_loader, val_loader)
    train_time = time.time() - t0
    logger.info("Training time: %.1fs", train_time)

    metrics_bin = evaluate_all(model_bin, test_loader, "binary", test_df)
    per_ds_bin = _evaluate_per_dataset(model_bin, "binary", df, pooled_orig, pooled_edited)
    logger.info("Binary-only results: AUROC=%.4f, AUPRC=%.4f, F1=%.4f, Pearson=%.4f, Spearman=%.4f",
                metrics_bin["auroc"], metrics_bin["auprc"], metrics_bin["f1"],
                metrics_bin.get("pearson_r", float("nan")),
                metrics_bin.get("spearman_r", float("nan")))
    all_results["binary_only"] = {
        "n_params": n_params,
        "train_time_seconds": train_time,
        "metrics": metrics_bin,
        "per_dataset": per_ds_bin,
    }
    del model_bin
    gc.collect()

    # ===================================================================
    # Mode 2: Regression-only
    # ===================================================================
    logger.info("\n" + "=" * 70)
    logger.info("MODE 2: Regression-only")
    logger.info("=" * 70)

    model_reg = RegressionPredictor()
    n_params = sum(p.numel() for p in model_reg.parameters())
    logger.info("Parameters: %d", n_params)

    t0 = time.time()
    model_reg = train_regression(model_reg, train_loader, val_loader)
    train_time = time.time() - t0
    logger.info("Training time: %.1fs", train_time)

    metrics_reg = evaluate_all(model_reg, test_loader, "regression", test_df)
    per_ds_reg = _evaluate_per_dataset(model_reg, "regression", df, pooled_orig, pooled_edited)
    logger.info("Regression-only results: AUROC=%.4f, AUPRC=%.4f, F1=%.4f, Pearson=%.4f, Spearman=%.4f",
                metrics_reg["auroc"], metrics_reg["auprc"], metrics_reg["f1"],
                metrics_reg.get("pearson_r", float("nan")),
                metrics_reg.get("spearman_r", float("nan")))
    all_results["regression_only"] = {
        "n_params": n_params,
        "train_time_seconds": train_time,
        "metrics": metrics_reg,
        "per_dataset": per_ds_reg,
    }
    del model_reg
    gc.collect()

    # ===================================================================
    # Mode 3: Multi-task (sweep lambda_rate)
    # ===================================================================
    logger.info("\n" + "=" * 70)
    logger.info("MODE 3: Multi-task (sweeping lambda_rate)")
    logger.info("=" * 70)

    lambda_values = [0.1, 0.5, 1.0]
    best_mt_auroc = -1.0
    best_lambda = None

    for lam in lambda_values:
        logger.info("\n--- lambda_rate=%.1f ---", lam)
        torch.manual_seed(42)
        model_mt = MultiTaskPredictor()
        n_params_mt = sum(p.numel() for p in model_mt.parameters())

        t0 = time.time()
        model_mt = train_multitask(model_mt, train_loader, val_loader, lambda_rate=lam)
        train_time = time.time() - t0
        logger.info("Training time: %.1fs", train_time)

        metrics_mt = evaluate_all(model_mt, test_loader, "multitask", test_df)
        per_ds_mt = _evaluate_per_dataset(model_mt, "multitask", df, pooled_orig, pooled_edited)
        logger.info("  lambda=%.1f: AUROC=%.4f, AUPRC=%.4f, F1=%.4f, Pearson=%.4f, Spearman=%.4f",
                    lam, metrics_mt["auroc"], metrics_mt["auprc"], metrics_mt["f1"],
                    metrics_mt.get("pearson_r", float("nan")),
                    metrics_mt.get("spearman_r", float("nan")))

        all_results[f"multitask_lambda{lam}"] = {
            "n_params": n_params_mt,
            "lambda_rate": lam,
            "train_time_seconds": train_time,
            "metrics": metrics_mt,
            "per_dataset": per_ds_mt,
        }

        if metrics_mt["auroc"] > best_mt_auroc:
            best_mt_auroc = metrics_mt["auroc"]
            best_lambda = lam

        del model_mt
        gc.collect()

    all_results["best_multitask_lambda"] = best_lambda

    # ===================================================================
    # Print summary
    # ===================================================================
    mode_keys = ["binary_only", "regression_only"] + [f"multitask_lambda{l}" for l in lambda_values]

    print("\n" + "=" * 110)
    print("MULTI-TASK COMPARISON RESULTS")
    print("=" * 110)
    print(f"{'Mode':<25} {'AUROC':>8} {'AUPRC':>8} {'F1':>8} {'Pearson':>8} {'Spearman':>8} {'N_rate':>7} {'Time(s)':>8}")
    print("-" * 110)

    for key in mode_keys:
        res = all_results[key]
        m = res["metrics"]
        label = key
        if key.startswith("multitask_lambda"):
            lam = res["lambda_rate"]
            label = f"multitask (lam={lam})"
        print(f"{label:<25} {m.get('auroc', float('nan')):>8.4f} {m.get('auprc', float('nan')):>8.4f} "
              f"{m.get('f1', float('nan')):>8.4f} {m.get('pearson_r', float('nan')):>8.4f} "
              f"{m.get('spearman_r', float('nan')):>8.4f} {m.get('n_rate_samples', 0):>7d} "
              f"{res['train_time_seconds']:>8.1f}")

    print("=" * 110)
    print(f"\nBest multi-task lambda: {best_lambda}")

    # Save results
    def serialize(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return str(obj)

    with open(OUTPUT_DIR / "multitask_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=serialize)

    logger.info("Results saved to %s", OUTPUT_DIR)

    # ===================================================================
    # Visualizations
    # ===================================================================
    mode_labels = []
    for key in mode_keys:
        if key == "binary_only":
            mode_labels.append("Binary")
        elif key == "regression_only":
            mode_labels.append("Regression")
        elif key.startswith("multitask_lambda"):
            lam = all_results[key]["lambda_rate"]
            mode_labels.append(f"MT lam={lam}")

    # --- Plot 1: Grouped bar chart for AUROC, AUPRC, F1 ---
    fig, ax = plt.subplots(figsize=(12, 6))
    metric_names = ["auroc", "auprc", "f1"]
    metric_display = ["AUROC", "AUPRC", "F1"]
    x = np.arange(len(mode_labels))
    bar_width = 0.25
    colors_bar = ["#2563eb", "#16a34a", "#dc2626"]

    for i, (mname, mdisp) in enumerate(zip(metric_names, metric_display)):
        values = []
        for key in mode_keys:
            v = all_results[key]["metrics"].get(mname, float("nan"))
            values.append(v if np.isfinite(v) else 0.0)
        bars = ax.bar(x + i * bar_width, values, bar_width, label=mdisp, color=colors_bar[i], alpha=0.85)
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax.set_xlabel("Training Mode")
    ax.set_ylabel("Score")
    ax.set_title("Multi-Task Comparison: Binary Metrics", fontweight="bold")
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(mode_labels, fontsize=9)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "multitask_binary_metrics.png")
    plt.close(fig)
    logger.info("Saved binary metrics bar chart")

    # --- Plot 2: Correlation metrics (Pearson, Spearman) ---
    fig, ax = plt.subplots(figsize=(10, 6))
    corr_metrics = ["pearson_r", "spearman_r"]
    corr_display = ["Pearson r", "Spearman rho"]
    corr_colors = ["#7c3aed", "#d97706"]
    bar_width = 0.3

    for i, (cname, cdisp) in enumerate(zip(corr_metrics, corr_display)):
        values = []
        for key in mode_keys:
            v = all_results[key]["metrics"].get(cname, float("nan"))
            values.append(v if np.isfinite(v) else 0.0)
        bars = ax.bar(x + i * bar_width, values, bar_width, label=cdisp, color=corr_colors[i], alpha=0.85)
        for bar, val in zip(bars, values):
            if val != 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax.set_xlabel("Training Mode")
    ax.set_ylabel("Correlation")
    ax.set_title("Multi-Task Comparison: Rate Correlation Metrics\n(on positive test sites with editing rates)",
                 fontweight="bold")
    ax.set_xticks(x + bar_width / 2)
    ax.set_xticklabels(mode_labels, fontsize=9)
    ax.legend(fontsize=10)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "multitask_correlation_metrics.png")
    plt.close(fig)
    logger.info("Saved correlation metrics bar chart")

    # --- Plot 3: Per-dataset AUROC breakdown ---
    ds_names_available = []
    for ds_key in ALL_DATASETS:
        dl = DATASET_LABELS[ds_key]
        if any(dl in all_results[mk].get("per_dataset", {}) for mk in mode_keys):
            ds_names_available.append(dl)

    if ds_names_available:
        fig, ax = plt.subplots(figsize=(14, 6))
        n_ds = len(ds_names_available)
        n_modes = len(mode_keys)
        bar_width = 0.8 / n_modes
        x_ds = np.arange(n_ds)
        mode_colors = ["#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed"]

        for mi, (mk, ml) in enumerate(zip(mode_keys, mode_labels)):
            per_ds = all_results[mk].get("per_dataset", {})
            values = []
            for dl in ds_names_available:
                v = per_ds.get(dl, {}).get("auroc", float("nan"))
                values.append(v if np.isfinite(v) else 0.0)
            ax.bar(x_ds + mi * bar_width, values, bar_width,
                   label=ml, color=mode_colors[mi % len(mode_colors)], alpha=0.85)

        ax.set_xlabel("Test Dataset")
        ax.set_ylabel("AUROC")
        ax.set_title("Per-Dataset AUROC by Training Mode", fontweight="bold")
        ax.set_xticks(x_ds + bar_width * (n_modes - 1) / 2)
        ax.set_xticklabels(ds_names_available, fontsize=10)
        ax.legend(fontsize=8, ncol=2)
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        fig.savefig(OUTPUT_DIR / "multitask_per_dataset_auroc.png")
        plt.close(fig)
        logger.info("Saved per-dataset AUROC chart")

    # --- Plot 4: Radar/overview chart - all metrics for each mode ---
    all_metric_names = ["auroc", "auprc", "f1", "pearson_r", "spearman_r"]
    all_metric_display = ["AUROC", "AUPRC", "F1", "Pearson", "Spearman"]

    fig, ax = plt.subplots(figsize=(12, 6))
    x_met = np.arange(len(all_metric_names))
    bar_width = 0.8 / len(mode_keys)
    mode_colors = ["#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed"]

    for mi, (mk, ml) in enumerate(zip(mode_keys, mode_labels)):
        values = []
        for mname in all_metric_names:
            v = all_results[mk]["metrics"].get(mname, float("nan"))
            values.append(v if np.isfinite(v) else 0.0)
        ax.bar(x_met + mi * bar_width, values, bar_width,
               label=ml, color=mode_colors[mi % len(mode_colors)], alpha=0.85)

    ax.set_xlabel("Metric")
    ax.set_ylabel("Value")
    ax.set_title("Multi-Task Comparison: All Metrics Overview", fontweight="bold")
    ax.set_xticks(x_met + bar_width * (len(mode_keys) - 1) / 2)
    ax.set_xticklabels(all_metric_display, fontsize=10)
    ax.legend(fontsize=8, ncol=2)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "multitask_all_metrics_overview.png")
    plt.close(fig)
    logger.info("Saved all-metrics overview chart")

    logger.info("\nAll outputs saved to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
