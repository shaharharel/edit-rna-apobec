#!/usr/bin/env python
"""Per-dataset train/test matrix experiment.

For each (train_dataset, test_dataset) pair, trains a subtraction_mlp and
evaluates AUROC. Also includes "All" as a training config (all 5 datasets).
Generates a 6x5 heatmap of AUROC values.

Training data: positives from train_ds + all tier2/tier3 negatives (train+val splits)
Test data: positives from test_ds + all tier2/tier3 negatives (test split)

Usage:
    python experiments/apobec/exp_dataset_matrix.py
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
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "apobec" / "outputs" / "dataset_matrix"

DATASET_SOURCES = ["advisor_c2t", "asaoka_2019", "alqassim_2021", "sharma_2015", "baysal_2016"]
DATASET_LABELS = {
    "advisor_c2t": "Levanon",
    "asaoka_2019": "Asaoka",
    "alqassim_2021": "Alqassim",
    "sharma_2015": "Sharma",
    "baysal_2016": "Baysal",
}
NEGATIVE_SOURCES = ["tier2_negative", "tier3_negative"]


# ---------------------------------------------------------------------------
# Model and Loss
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_and_evaluate(train_df, test_df, pooled_orig, pooled_edited,
                       epochs=50, lr=1e-3, patience=10, seed=42):
    """Train subtraction_mlp on train_df, evaluate on test_df. Returns AUROC."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Split train_df into train/val (80/20)
    train_split = train_df[train_df["split"] == "train"]
    val_split = train_df[train_df["split"] == "val"]

    if len(train_split) < 10 or len(val_split) < 5:
        return float("nan")

    def make_loader(df, shuffle=False):
        ds = DiffDataset(
            df["site_id"].tolist(),
            df["label"].values.astype(np.float32),
            pooled_orig, pooled_edited,
        )
        return DataLoader(ds, batch_size=64, shuffle=shuffle, num_workers=0, drop_last=False)

    train_loader = make_loader(train_split, shuffle=True)
    val_loader = make_loader(val_split)
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

        # Validate
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
        if len(np.unique(y_true)) < 2:
            val_auroc = 0.0
        else:
            val_auroc = roc_auc_score(y_true, y_score)

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

    # Test
    model.eval()
    all_y, all_s = [], []
    with torch.no_grad():
        for diffs, labels in test_loader:
            logits = model(diffs)
            probs = torch.sigmoid(logits).numpy()
            all_y.append(labels.numpy())
            all_s.append(probs)
    y_true = np.concatenate(all_y)
    y_score = np.concatenate(all_s)

    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    torch.manual_seed(42)

    # Load data
    logger.info("Loading data...")
    pooled_orig = torch.load(EMB_DIR / "rnafm_pooled.pt", weights_only=False)
    pooled_edited = torch.load(EMB_DIR / "rnafm_pooled_edited.pt", weights_only=False)
    splits_df = pd.read_csv(SPLITS_CSV)

    available_ids = set(pooled_orig.keys()) & set(pooled_edited.keys())
    splits_df = splits_df[splits_df["site_id"].isin(available_ids)].copy()

    # Separate negatives (shared across all configs)
    neg_df = splits_df[splits_df["dataset_source"].isin(NEGATIVE_SOURCES)]
    neg_train_val = neg_df[neg_df["split"].isin(["train", "val"])]
    neg_test = neg_df[neg_df["split"] == "test"]
    logger.info("Negatives: %d train+val, %d test", len(neg_train_val), len(neg_test))

    # Training configs: 5 individual datasets + "All"
    train_configs = DATASET_SOURCES + ["All"]
    train_labels = [DATASET_LABELS[ds] for ds in DATASET_SOURCES] + ["All"]
    test_labels = [DATASET_LABELS[ds] for ds in DATASET_SOURCES]

    # Build the matrix
    n_train = len(train_configs)
    n_test = len(DATASET_SOURCES)
    matrix = np.full((n_train, n_test), float("nan"))

    logger.info("\nRunning %d x %d = %d experiments...\n", n_train, n_test, n_train * n_test)
    t_start = time.time()

    for i, train_cfg in enumerate(train_configs):
        # Assemble training positives
        if train_cfg == "All":
            train_pos = splits_df[
                (splits_df["label"] == 1) &
                (splits_df["split"].isin(["train", "val"]))
            ]
        else:
            train_pos = splits_df[
                (splits_df["dataset_source"] == train_cfg) &
                (splits_df["label"] == 1) &
                (splits_df["split"].isin(["train", "val"]))
            ]

        # Training data = positives + all negatives (train+val)
        train_data = pd.concat([train_pos, neg_train_val], ignore_index=True)

        for j, test_ds in enumerate(DATASET_SOURCES):
            # Test positives from specific dataset
            test_pos = splits_df[
                (splits_df["dataset_source"] == test_ds) &
                (splits_df["label"] == 1) &
                (splits_df["split"] == "test")
            ]

            # Test data = positives + all negatives (test)
            test_data = pd.concat([test_pos, neg_test], ignore_index=True)

            if len(test_pos) < 3:
                logger.info("  [%s -> %s] Skipping: only %d test positives",
                            train_labels[i], test_labels[j], len(test_pos))
                continue

            auroc = train_and_evaluate(train_data, test_data, pooled_orig, pooled_edited)
            matrix[i, j] = auroc

            n_train_pos = (train_data["label"] == 1).sum()
            n_test_pos = (test_data["label"] == 1).sum()
            logger.info("  [%s -> %s] AUROC=%.4f  (train: %d pos + %d neg, test: %d pos + %d neg)",
                        train_labels[i], test_labels[j], auroc,
                        n_train_pos, len(train_data) - n_train_pos,
                        n_test_pos, len(test_data) - n_test_pos)

    elapsed = time.time() - t_start
    logger.info("\nTotal time: %.1fs (%.1fs per experiment)", elapsed, elapsed / (n_train * n_test))

    # ===================================================================
    # Print matrix
    # ===================================================================
    print("\n" + "=" * 90)
    print("DATASET TRAIN/TEST MATRIX (AUROC)")
    print("=" * 90)

    train_test_label = "Train \\ Test"
    header = f"{train_test_label:<15}" + "".join(f"{tl:>12}" for tl in test_labels)
    print(header)
    print("-" * 90)

    for i, tl in enumerate(train_labels):
        row = f"{tl:<15}"
        for j in range(n_test):
            val = matrix[i, j]
            if np.isnan(val):
                row += f"{'N/A':>12}"
            else:
                row += f"{val:>12.4f}"
        print(row)

    print("=" * 90)

    # ===================================================================
    # Generate heatmap
    # ===================================================================
    fig, ax = plt.subplots(figsize=(10, 7))

    # Mask NaN values
    masked = np.ma.masked_invalid(matrix)
    im = ax.imshow(masked, cmap="YlOrRd", aspect="auto", vmin=0.5, vmax=1.0)

    ax.set_xticks(range(n_test))
    ax.set_xticklabels(test_labels, fontsize=11)
    ax.set_yticks(range(n_train))
    ax.set_yticklabels(train_labels, fontsize=11)
    ax.set_xlabel("Test Dataset", fontsize=13)
    ax.set_ylabel("Training Dataset", fontsize=13)
    ax.set_title("Cross-Dataset AUROC (Subtraction MLP)", fontsize=14, fontweight="bold")

    # Annotate cells
    for i in range(n_train):
        for j in range(n_test):
            val = matrix[i, j]
            if not np.isnan(val):
                color = "white" if val > 0.85 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=10, fontweight="bold", color=color)

    plt.colorbar(im, ax=ax, label="AUROC", shrink=0.8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "dataset_matrix_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Heatmap saved to %s", OUTPUT_DIR / "dataset_matrix_heatmap.png")

    # ===================================================================
    # Save results
    # ===================================================================
    results = {
        "train_configs": train_labels,
        "test_datasets": test_labels,
        "auroc_matrix": matrix.tolist(),
        "total_time_seconds": elapsed,
    }

    # Also save as per-pair dict for easy lookup
    results["per_pair"] = {}
    for i, tl in enumerate(train_labels):
        for j, te in enumerate(test_labels):
            val = matrix[i, j]
            results["per_pair"][f"{tl}->{te}"] = float(val) if not np.isnan(val) else None

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

    with open(OUTPUT_DIR / "dataset_matrix_results.json", "w") as f:
        json.dump(results, f, indent=2, default=serialize)

    logger.info("Results saved to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
