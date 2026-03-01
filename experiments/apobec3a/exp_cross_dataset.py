#!/usr/bin/env python
"""Cross-dataset generalization experiment.

Evaluates how models trained on one dataset generalize to others.
Uses subtraction_mlp (fast, good baseline) for the NxN matrix and
the trained EditRNA-A3A for the final evaluation.

Experiment designs:
1. Train on Levanon → test on Asaoka, Sharma, Alqassim (+ Levanon test)
2. NxN generalization matrix (each pair)
3. Combined training experiments (Levanon+each, all)

Usage:
    python experiments/apobec3a/exp_cross_dataset.py
"""

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

EMB_DIR = PROJECT_ROOT / "data" / "processed" / "embeddings"
SPLITS_CSV = PROJECT_ROOT / "data" / "processed" / "splits_expanded.csv"
STRUCT_CACHE = EMB_DIR / "vienna_structure_cache.npz"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "apobec" / "outputs" / "cross_dataset"

DATASETS = ["advisor_c2t", "asaoka_2019", "sharma_2015", "alqassim_2021"]
DATASET_LABELS = {
    "advisor_c2t": "Levanon",
    "asaoka_2019": "Asaoka",
    "sharma_2015": "Sharma",
    "alqassim_2021": "Alqassim",
    "baysal_2016": "Baysal",
}

# Import baseline and train_baselines utilities
from experiments.apobec.train_baselines import (
    BaselineConfig,
    EmbeddingDataset,
    FocalLoss,
    build_model,
    compute_metrics,
    embedding_collate_fn,
)


def load_embeddings():
    """Load pooled embedding caches."""
    pooled_orig = torch.load(EMB_DIR / "rnafm_pooled.pt", weights_only=False)
    pooled_edited = torch.load(EMB_DIR / "rnafm_pooled_edited.pt", weights_only=False)
    logger.info("Loaded %d pooled embeddings", len(pooled_orig))

    # Structure delta
    structure_delta = None
    if STRUCT_CACHE.exists():
        data = np.load(STRUCT_CACHE, allow_pickle=True)
        feat_key = "delta_features" if "delta_features" in data else "features"
        if "site_ids" in data and feat_key in data:
            sids = data["site_ids"]
            feats = data[feat_key]
            structure_delta = {str(sid): feats[i] for i, sid in enumerate(sids)}
            logger.info("Loaded %d structure delta features", len(structure_delta))

    return pooled_orig, pooled_edited, structure_delta


def create_dataset_splits(splits_df, pooled_orig, train_datasets, test_datasets,
                          neg_ratio=5, seed=42):
    """Create train/test splits for cross-dataset evaluation.

    Train: positives from train_datasets + negatives (sampled)
    Test: positives from test_datasets + negatives (sampled)
    """
    rng = np.random.RandomState(seed)
    available_ids = set(pooled_orig.keys())

    # Filter to available embeddings
    df = splits_df[splits_df["site_id"].isin(available_ids)].copy()

    # Training set
    train_pos = df[
        (df["dataset_source"].isin(train_datasets)) & (df["label"] == 1)
    ]
    train_neg = df[
        (df["dataset_source"].isin(["tier2_negative", "tier3_negative"])) &
        (df["split"] == "train")
    ]

    # Sample negatives
    n_train_pos = len(train_pos)
    max_train_neg = n_train_pos * neg_ratio
    if len(train_neg) > max_train_neg:
        train_neg = train_neg.sample(n=max_train_neg, random_state=rng)

    # Test set
    test_pos = df[
        (df["dataset_source"].isin(test_datasets)) & (df["label"] == 1)
    ]
    test_neg = df[
        (df["dataset_source"].isin(["tier2_negative", "tier3_negative"])) &
        (df["split"] == "test")
    ]

    n_test_pos = len(test_pos)
    max_test_neg = max(n_test_pos * 3, 100)
    if len(test_neg) > max_test_neg:
        test_neg = test_neg.sample(n=max_test_neg, random_state=rng)

    train_df = pd.concat([train_pos, train_neg], ignore_index=True)
    test_df = pd.concat([test_pos, test_neg], ignore_index=True)

    return train_df, test_df


def train_and_evaluate(model_name, train_df, test_df,
                       pooled_orig, pooled_edited, structure_delta,
                       epochs=30, lr=1e-3, patience=8, seed=42):
    """Train a model on train_df and evaluate on test_df."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    config = BaselineConfig(model_name=model_name, d_model=640, seed=seed)

    # Build datasets
    train_ids = train_df["site_id"].tolist()
    train_labels = train_df["label"].values.astype(np.float32)
    test_ids = test_df["site_id"].tolist()
    test_labels = test_df["label"].values.astype(np.float32)

    train_ds = EmbeddingDataset(
        train_ids, train_labels, pooled_orig, pooled_edited,
        structure_delta=structure_delta,
    )
    test_ds = EmbeddingDataset(
        test_ids, test_labels, pooled_orig, pooled_edited,
        structure_delta=structure_delta,
    )

    train_loader = DataLoader(
        train_ds, batch_size=64, shuffle=True, num_workers=0,
        collate_fn=embedding_collate_fn,
    )
    test_loader = DataLoader(
        test_ds, batch_size=64, shuffle=False, num_workers=0,
        collate_fn=embedding_collate_fn,
    )

    # Build and train model
    model = build_model(model_name, config)
    device = torch.device("cpu")
    model = model.to(device)

    loss_fn = FocalLoss(gamma=2.0, alpha=0.75)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        n_batches = 0
        for batch in train_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            optimizer.zero_grad()
            output = model(batch)
            logits = output["binary_logit"]
            loss = loss_fn(logits, batch["labels"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        if avg_loss < best_loss - 1e-4:
            best_loss = avg_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break

    if best_state:
        model.load_state_dict(best_state)

    # Evaluate
    model.eval()
    all_targets = []
    all_scores = []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            output = model(batch)
            logits = output["binary_logit"].squeeze(-1).cpu().numpy()
            probs = 1.0 / (1.0 + np.exp(-logits))
            all_targets.append(batch["labels"].cpu().numpy())
            all_scores.append(probs)

    y_true = np.concatenate(all_targets)
    y_score = np.concatenate(all_scores)
    metrics = compute_metrics(y_true, y_score)

    return metrics


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    splits_df = pd.read_csv(SPLITS_CSV)
    pooled_orig, pooled_edited, structure_delta = load_embeddings()

    logger.info("Sites per dataset:")
    for ds in DATASETS:
        n = (splits_df["dataset_source"] == ds).sum()
        logger.info("  %s: %d", ds, n)

    # ===================================================================
    # Experiment 1: Levanon → each dataset
    # ===================================================================
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 1: Levanon trained → cross-dataset evaluation")
    logger.info("=" * 70)

    levanon_results = {}
    for test_ds in DATASETS:
        train_df, test_df = create_dataset_splits(
            splits_df,
            pooled_orig,
            train_datasets=["advisor_c2t"],
            test_datasets=[test_ds],
        )
        logger.info("\nTrain: Levanon (%d pos + %d neg) → Test: %s (%d pos + %d neg)",
                    (train_df["label"] == 1).sum(), (train_df["label"] == 0).sum(),
                    DATASET_LABELS[test_ds],
                    (test_df["label"] == 1).sum(), (test_df["label"] == 0).sum())

        metrics = train_and_evaluate(
            "subtraction_mlp", train_df, test_df,
            pooled_orig, pooled_edited, structure_delta,
        )
        levanon_results[test_ds] = metrics
        logger.info("  AUROC=%.4f  AUPRC=%.4f  F1=%.4f",
                    metrics["auroc"], metrics["auprc"], metrics["f1"])

    # ===================================================================
    # Experiment 2: NxN generalization matrix
    # ===================================================================
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 2: NxN generalization matrix (subtraction_mlp)")
    logger.info("=" * 70)

    nxn_matrix = {}
    for train_ds in DATASETS:
        for test_ds in DATASETS:
            key = f"{DATASET_LABELS[train_ds]}→{DATASET_LABELS[test_ds]}"
            train_df, test_df = create_dataset_splits(
                splits_df,
                pooled_orig,
                train_datasets=[train_ds],
                test_datasets=[test_ds],
            )

            n_train_pos = (train_df["label"] == 1).sum()
            n_test_pos = (test_df["label"] == 1).sum()
            if n_train_pos < 10 or n_test_pos < 10:
                logger.warning("  %s: insufficient data (train=%d, test=%d)",
                              key, n_train_pos, n_test_pos)
                nxn_matrix[key] = {"auroc": float("nan"), "auprc": float("nan")}
                continue

            metrics = train_and_evaluate(
                "subtraction_mlp", train_df, test_df,
                pooled_orig, pooled_edited, structure_delta,
            )
            nxn_matrix[key] = metrics
            logger.info("  %s: AUROC=%.4f  AUPRC=%.4f",
                        key, metrics["auroc"], metrics["auprc"])

    # ===================================================================
    # Experiment 3: Combined training
    # ===================================================================
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 3: Combined training experiments")
    logger.info("=" * 70)

    combined_configs = [
        ("Levanon only", ["advisor_c2t"]),
        ("Levanon + Asaoka", ["advisor_c2t", "asaoka_2019"]),
        ("Levanon + Sharma", ["advisor_c2t", "sharma_2015"]),
        ("Levanon + Alqassim", ["advisor_c2t", "alqassim_2021"]),
        ("All datasets", DATASETS),
    ]

    combined_results = {}
    for config_name, train_datasets in combined_configs:
        logger.info("\nTraining on: %s", config_name)
        per_dataset_metrics = {}

        for test_ds in DATASETS:
            train_df, test_df = create_dataset_splits(
                splits_df,
                pooled_orig,
                train_datasets=train_datasets,
                test_datasets=[test_ds],
            )

            n_test_pos = (test_df["label"] == 1).sum()
            if n_test_pos < 10:
                per_dataset_metrics[test_ds] = {"auroc": float("nan")}
                continue

            metrics = train_and_evaluate(
                "subtraction_mlp", train_df, test_df,
                pooled_orig, pooled_edited, structure_delta,
            )
            per_dataset_metrics[test_ds] = metrics
            logger.info("  → %s: AUROC=%.4f  AUPRC=%.4f",
                        DATASET_LABELS[test_ds],
                        metrics["auroc"], metrics["auprc"])

        combined_results[config_name] = per_dataset_metrics

    # ===================================================================
    # Print summary tables
    # ===================================================================
    print("\n" + "=" * 80)
    print("CROSS-DATASET GENERALIZATION RESULTS")
    print("=" * 80)

    # NxN matrix
    print("\n--- NxN Generalization Matrix (AUROC, subtraction_mlp) ---")
    train_test_label = "Train \\ Test"
    header = f"{train_test_label:<18}"
    for ds in DATASETS:
        header += f" {DATASET_LABELS[ds]:>10}"
    print(header)
    print("-" * len(header))

    for train_ds in DATASETS:
        row = f"{DATASET_LABELS[train_ds]:<18}"
        for test_ds in DATASETS:
            key = f"{DATASET_LABELS[train_ds]}→{DATASET_LABELS[test_ds]}"
            auroc = nxn_matrix.get(key, {}).get("auroc", float("nan"))
            if np.isnan(auroc):
                row += f" {'N/A':>10}"
            else:
                diag = "*" if train_ds == test_ds else " "
                row += f" {auroc:>9.4f}{diag}"
        print(row)

    # Combined training table
    print("\n--- Combined Training Results (AUROC, subtraction_mlp) ---")
    header = f"{'Training Config':<25}"
    for ds in DATASETS:
        header += f" {DATASET_LABELS[ds]:>10}"
    print(header)
    print("-" * len(header))

    for config_name, per_ds in combined_results.items():
        row = f"{config_name:<25}"
        for ds in DATASETS:
            auroc = per_ds.get(ds, {}).get("auroc", float("nan"))
            if np.isnan(auroc):
                row += f" {'N/A':>10}"
            else:
                row += f" {auroc:>10.4f}"
        print(row)

    print("=" * 80)

    # Save all results
    all_results = {
        "levanon_cross_dataset": {k: v for k, v in levanon_results.items()},
        "nxn_matrix": nxn_matrix,
        "combined_training": {
            name: {ds: m for ds, m in metrics.items()}
            for name, metrics in combined_results.items()
        },
    }

    def serialize(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    with open(OUTPUT_DIR / "cross_dataset_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=serialize)

    logger.info("\nResults saved to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
