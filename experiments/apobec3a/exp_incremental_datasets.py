#!/usr/bin/env python
"""Incremental dataset addition experiment.

Shows how performance changes as more datasets are added to training,
using a fixed combined test set for fair comparison.

Training order: Levanon → +Asaoka → +Alqassim → +Sharma → +Baysal
(ordered by dataset size / historical importance)

Usage:
    python experiments/apobec3a/exp_incremental_datasets.py
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
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logger = logging.getLogger(__name__)

EMB_DIR = PROJECT_ROOT / "data" / "processed" / "embeddings"
SPLITS_CSV = PROJECT_ROOT / "data" / "processed" / "splits_expanded.csv"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "apobec3a" / "outputs" / "incremental_datasets"

# Order of dataset addition
DATASET_ORDER = ["advisor_c2t", "asaoka_2019", "alqassim_2021", "sharma_2015", "baysal_2016"]
DATASET_LABELS = {
    "advisor_c2t": "Levanon",
    "asaoka_2019": "Asaoka",
    "alqassim_2021": "Alqassim",
    "sharma_2015": "Sharma",
    "baysal_2016": "Baysal",
}
NEGATIVE_SOURCES = ["tier2_negative", "tier3_negative"]

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
})


class SubtractionMLP(nn.Module):
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


class DiffDataset(Dataset):
    def __init__(self, site_ids, labels, pooled_orig, pooled_edited):
        self.site_ids = site_ids
        self.labels = labels
        self.pooled_orig = pooled_orig
        self.pooled_edited = pooled_edited

    def __len__(self):
        return len(self.site_ids)

    def __getitem__(self, idx):
        sid = self.site_ids[idx]
        diff = self.pooled_edited[sid] - self.pooled_orig[sid]
        return diff, torch.tensor(self.labels[idx], dtype=torch.float32)


def train_and_evaluate(train_df, val_df, test_df, pooled_orig, pooled_edited,
                       epochs=50, lr=1e-3, patience=10, seed=42):
    """Train subtraction MLP, return test AUROC and per-dataset AUROCs."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    def make_loader(df, shuffle=False):
        ds = DiffDataset(
            df["site_id"].tolist(),
            df["label"].values.astype(np.float32),
            pooled_orig, pooled_edited,
        )
        return DataLoader(ds, batch_size=64, shuffle=shuffle, num_workers=0)

    train_loader = make_loader(train_df, shuffle=True)
    val_loader = make_loader(val_df)
    test_loader = make_loader(test_df)

    model = SubtractionMLP()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)
    loss_fn = FocalLoss()

    best_auroc = -1.0
    patience_counter = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        for diffs, labels in train_loader:
            optimizer.zero_grad()
            logits = model(diffs)
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        all_y, all_s = [], []
        with torch.no_grad():
            for diffs, labels in val_loader:
                logits = model(diffs)
                probs = torch.sigmoid(logits).numpy()
                all_y.append(labels.numpy())
                all_s.append(probs)
        y_true = np.concatenate(all_y)
        y_score = np.concatenate(all_s)
        if len(np.unique(y_true)) >= 2:
            val_auroc = roc_auc_score(y_true, y_score)
        else:
            val_auroc = 0.0

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

    # Evaluate on test
    model.eval()
    all_y, all_s, all_ds = [], [], []
    with torch.no_grad():
        for diffs, labels in test_loader:
            logits = model(diffs)
            probs = torch.sigmoid(logits).numpy()
            all_y.append(labels.numpy())
            all_s.append(probs)

    y_true = np.concatenate(all_y)
    y_score = np.concatenate(all_s)

    overall_auroc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) >= 2 else float("nan")

    # Per-dataset AUROC on test set
    # Merge predictions back with test_df to get dataset_source
    test_site_ids = test_df["site_id"].values
    test_sources = test_df["dataset_source"].values

    per_ds_auroc = {}
    for ds_name, ds_label in DATASET_LABELS.items():
        ds_mask = test_sources == ds_name
        if ds_mask.sum() < 3:
            continue
        ds_y = y_true[ds_mask]
        ds_s = y_score[ds_mask]
        # Need both classes for AUROC
        if len(np.unique(ds_y)) >= 2:
            per_ds_auroc[ds_label] = float(roc_auc_score(ds_y, ds_s))

    # If we only have positives from this dataset, compute on positives + all negatives
    # Re-evaluate per positive dataset (pos from DS + all neg)
    neg_mask = test_df["dataset_source"].isin(NEGATIVE_SOURCES).values
    for ds_name, ds_label in DATASET_LABELS.items():
        if ds_label in per_ds_auroc:
            continue
        pos_mask = (test_sources == ds_name) & (test_df["label"].values == 1)
        combined_mask = pos_mask | neg_mask
        if combined_mask.sum() < 5 or pos_mask.sum() < 2:
            continue
        ds_y = y_true[combined_mask]
        ds_s = y_score[combined_mask]
        if len(np.unique(ds_y)) >= 2:
            per_ds_auroc[ds_label] = float(roc_auc_score(ds_y, ds_s))

    return overall_auroc, per_ds_auroc


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading embeddings...")
    pooled_orig = torch.load(EMB_DIR / "rnafm_pooled.pt", weights_only=False)
    pooled_edited = torch.load(EMB_DIR / "rnafm_pooled_edited.pt", weights_only=False)
    splits_df = pd.read_csv(SPLITS_CSV)

    available_ids = set(pooled_orig.keys()) & set(pooled_edited.keys())
    splits_df = splits_df[splits_df["site_id"].isin(available_ids)].copy()

    # Fixed combined test set = all test split sites
    test_df = splits_df[splits_df["split"] == "test"].copy()
    logger.info("Fixed test set: %d sites (%d pos, %d neg)",
                len(test_df), (test_df["label"] == 1).sum(), (test_df["label"] == 0).sum())

    # Negatives for training/val
    neg_train = splits_df[
        (splits_df["dataset_source"].isin(NEGATIVE_SOURCES)) &
        (splits_df["split"] == "train")
    ]
    neg_val = splits_df[
        (splits_df["dataset_source"].isin(NEGATIVE_SOURCES)) &
        (splits_df["split"] == "val")
    ]

    results = []
    cumulative_datasets = []

    for step, ds_name in enumerate(DATASET_ORDER, 1):
        cumulative_datasets.append(ds_name)
        ds_labels = [DATASET_LABELS[d] for d in cumulative_datasets]
        config_name = " + ".join(ds_labels)

        logger.info("\n" + "=" * 70)
        logger.info("STEP %d: Training on %s", step, config_name)
        logger.info("=" * 70)

        # Gather train positives from all cumulative datasets
        train_pos = splits_df[
            (splits_df["dataset_source"].isin(cumulative_datasets)) &
            (splits_df["label"] == 1) &
            (splits_df["split"] == "train")
        ]
        val_pos = splits_df[
            (splits_df["dataset_source"].isin(cumulative_datasets)) &
            (splits_df["label"] == 1) &
            (splits_df["split"] == "val")
        ]

        train_data = pd.concat([train_pos, neg_train], ignore_index=True)
        val_data = pd.concat([val_pos, neg_val], ignore_index=True)

        n_train_pos = (train_data["label"] == 1).sum()
        n_train_neg = (train_data["label"] == 0).sum()
        n_val = len(val_data)

        logger.info("Train: %d pos + %d neg = %d, Val: %d",
                    n_train_pos, n_train_neg, len(train_data), n_val)

        overall_auroc, per_ds = train_and_evaluate(
            train_data, val_data, test_df, pooled_orig, pooled_edited
        )

        logger.info("Overall AUROC: %.4f", overall_auroc)
        for dl, auc in per_ds.items():
            logger.info("  %s: %.4f", dl, auc)

        results.append({
            "step": step,
            "datasets": ds_labels,
            "config_name": config_name,
            "n_train_pos": int(n_train_pos),
            "n_train_neg": int(n_train_neg),
            "n_train_total": int(len(train_data)),
            "overall_auroc": float(overall_auroc),
            "per_dataset_auroc": per_ds,
        })

    # Print summary
    print("\n" + "=" * 90)
    print("INCREMENTAL DATASET ADDITION RESULTS")
    print("=" * 90)
    print(f"{'Step':<5} {'Training Datasets':<45} {'Train Pos':>10} {'AUROC':>8}")
    print("-" * 90)
    for r in results:
        print(f"{r['step']:<5} {r['config_name']:<45} {r['n_train_pos']:>10} {r['overall_auroc']:>8.4f}")
    print("=" * 90)

    # Generate line plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    steps = [r["step"] for r in results]
    aurocs = [r["overall_auroc"] for r in results]
    labels_list = [r["config_name"].split(" + ")[-1] for r in results]

    # Panel 1: Overall AUROC progression
    ax1.plot(steps, aurocs, "o-", color="#2563eb", linewidth=2.5, markersize=8, zorder=5)
    for i, (s, a, l) in enumerate(zip(steps, aurocs, labels_list)):
        ax1.annotate(f"+{l}\n{a:.3f}", (s, a), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=8, fontweight="bold")
    ax1.set_xlabel("Number of Training Datasets")
    ax1.set_ylabel("Test AUROC")
    ax1.set_title("Overall Test AUROC vs Training Datasets", fontweight="bold")
    ax1.set_xticks(steps)
    ax1.set_xticklabels([str(s) for s in steps])
    ax1.set_ylim(min(aurocs) - 0.03, max(aurocs) + 0.04)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Per-dataset AUROC progression
    ds_colors = {
        "Levanon": "#2563eb", "Asaoka": "#16a34a", "Sharma": "#dc2626",
        "Alqassim": "#d97706", "Baysal": "#7c3aed",
    }
    all_ds_labels = list(DATASET_LABELS.values())
    for dl in all_ds_labels:
        ds_aurocs = []
        ds_steps = []
        for r in results:
            if dl in r["per_dataset_auroc"]:
                ds_steps.append(r["step"])
                ds_aurocs.append(r["per_dataset_auroc"][dl])
        if ds_aurocs:
            ax2.plot(ds_steps, ds_aurocs, "o-", color=ds_colors.get(dl, "#666"),
                    label=dl, linewidth=1.8, markersize=6)

    ax2.set_xlabel("Number of Training Datasets")
    ax2.set_ylabel("AUROC")
    ax2.set_title("Per-Dataset AUROC vs Training Datasets", fontweight="bold")
    ax2.set_xticks(steps)
    ax2.legend(fontsize=8, loc="best")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "incremental_datasets.png")
    plt.close(fig)
    logger.info("Plot saved to %s", OUTPUT_DIR / "incremental_datasets.png")

    # Save results
    output = {"results": results}
    with open(OUTPUT_DIR / "incremental_results.json", "w") as f:
        json.dump(output, f, indent=2)

    logger.info("Results saved to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
