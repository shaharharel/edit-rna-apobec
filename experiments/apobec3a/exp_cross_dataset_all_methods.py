#!/usr/bin/env python
"""Cross-dataset generalization with ALL model architectures.

Extends exp_cross_dataset.py to evaluate all 10 architectures
(same as DL Architecture Comparison) in a cross-dataset setting:

  1. NxN matrix: train on one dataset, test on all others
  2. Combined training: train on dataset combinations, test on each

Architecture ordering (simplest to most complex):
  Majority Class, StructureOnly, GB_HandFeatures, GB_AllFeatures,
  PooledMLP, SubtractionMLP, ConcatMLP, CrossAttention,
  DiffAttention, EditRNA-A3A

Key decisions:
  - Sharma (6 positives): skip as training source, keep as test target
  - Trainable datasets: Levanon (636), Asaoka (4933), Alqassim (128)
  - Negatives: tier2/tier3, sampled at 5:1 ratio
  - Validation: 20% of training positives held out for early stopping

Usage:
    python experiments/apobec3a/exp_cross_dataset_all_methods.py
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.apobec3a.train_baselines import (
    BaselineConfig,
    EmbeddingDataset,
    FocalLoss,
    build_model,
    compute_metrics,
    embedding_collate_fn,
    _forward_baseline,
)
from experiments.apobec3a.train_gradient_boosting import (
    extract_motif_features,
    extract_structure_delta_features,
    extract_loop_position_features,
    extract_embedding_delta_features,
    compute_binary_metrics,
    get_classifier,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMB_DIR = PROJECT_ROOT / "data" / "processed" / "embeddings"
SPLITS_CSV = PROJECT_ROOT / "data" / "processed" / "splits_expanded.csv"
STRUCT_CACHE = EMB_DIR / "vienna_structure_cache.npz"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "apobec3a" / "outputs" / "cross_dataset_all_methods"

DATASETS = ["advisor_c2t", "asaoka_2019", "sharma_2015", "alqassim_2021"]
DATASET_LABELS = {
    "advisor_c2t": "Levanon",
    "asaoka_2019": "Asaoka",
    "sharma_2015": "Sharma",
    "alqassim_2021": "Alqassim",
}

# Datasets with enough positives to train on
TRAINABLE_DATASETS = ["advisor_c2t", "asaoka_2019", "alqassim_2021"]

# All DL model names
DL_MODELS = [
    "structure_only",
    "pooled_mlp",
    "subtraction_mlp",
    "concat_mlp",
    "cross_attention",
    "diff_attention",
    "editrna",
]

ALL_MODEL_NAMES = [
    "majority_class",
    "structure_only",
    "gb_hand",
    "gb_all",
    "pooled_mlp",
    "subtraction_mlp",
    "concat_mlp",
    "cross_attention",
    "diff_attention",
    "editrna",
]

MODEL_DISPLAY = {
    "majority_class": "Majority Class",
    "structure_only": "StructureOnly",
    "gb_hand": "GB_Hand",
    "gb_all": "GB_All",
    "pooled_mlp": "PooledMLP",
    "subtraction_mlp": "SubtractionMLP",
    "concat_mlp": "ConcatMLP",
    "cross_attention": "CrossAttention",
    "diff_attention": "DiffAttention",
    "editrna": "EditRNA",
}

NEG_RATIO = 5
VAL_FRACTION = 0.2
SEED = 42


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_embeddings():
    """Load pooled and token embedding caches."""
    pooled_orig = torch.load(EMB_DIR / "rnafm_pooled.pt", weights_only=False)
    pooled_edited = torch.load(EMB_DIR / "rnafm_pooled_edited.pt", weights_only=False)

    tokens_orig = None
    tokens_edited = None
    tok_path = EMB_DIR / "rnafm_tokens.pt"
    if tok_path.exists():
        tokens_orig = torch.load(tok_path, weights_only=False)
        tokens_edited = torch.load(EMB_DIR / "rnafm_tokens_edited.pt", weights_only=False)
        logger.info("Loaded %d token embeddings", len(tokens_orig))

    # Structure delta
    structure_delta = None
    if STRUCT_CACHE.exists():
        data = np.load(STRUCT_CACHE, allow_pickle=True)
        feat_key = "delta_features" if "delta_features" in data else "features"
        if "site_ids" in data and feat_key in data:
            sids = data["site_ids"]
            feats = data[feat_key]
            structure_delta = {str(sid): feats[i] for i, sid in enumerate(sids)}

    logger.info("Loaded %d pooled embeddings", len(pooled_orig))
    return pooled_orig, pooled_edited, tokens_orig, tokens_edited, structure_delta


def create_train_val_test_splits(splits_df, pooled_orig, train_datasets, test_datasets,
                                  neg_ratio=5, val_fraction=0.2, seed=42):
    """Create train/val/test splits for cross-dataset evaluation.

    Train: positives from train_datasets + negatives (sampled)
    Val: 20% of train positives held out + proportional negatives
    Test: positives from test_datasets + negatives (sampled)
    """
    rng = np.random.RandomState(seed)
    available_ids = set(pooled_orig.keys())
    df = splits_df[splits_df["site_id"].isin(available_ids)].copy()

    # Training positives
    train_pos = df[
        (df["dataset_source"].isin(train_datasets)) & (df["label"] == 1)
    ].copy()

    # Hold out validation set from training positives
    n_val = max(int(len(train_pos) * val_fraction), 1)
    train_pos_shuffled = train_pos.sample(frac=1, random_state=rng)
    val_pos = train_pos_shuffled.iloc[:n_val]
    train_pos = train_pos_shuffled.iloc[n_val:]

    # Training negatives from tier2/tier3
    all_neg = df[
        df["dataset_source"].isin(["tier2_negative", "tier3_negative"])
    ]
    train_neg_pool = all_neg[all_neg["split"] == "train"]
    test_neg_pool = all_neg[all_neg["split"] == "test"]
    val_neg_pool = all_neg[all_neg["split"] == "val"]

    # Sample negatives proportionally
    max_train_neg = len(train_pos) * neg_ratio
    if len(train_neg_pool) > max_train_neg:
        train_neg = train_neg_pool.sample(n=max_train_neg, random_state=rng)
    else:
        train_neg = train_neg_pool

    max_val_neg = len(val_pos) * neg_ratio
    if len(val_neg_pool) > max_val_neg:
        val_neg = val_neg_pool.sample(n=max_val_neg, random_state=rng)
    else:
        val_neg = val_neg_pool

    # Test set
    test_pos = df[
        (df["dataset_source"].isin(test_datasets)) & (df["label"] == 1)
    ]
    max_test_neg = max(len(test_pos) * 3, 100)
    if len(test_neg_pool) > max_test_neg:
        test_neg = test_neg_pool.sample(n=max_test_neg, random_state=rng)
    else:
        test_neg = test_neg_pool

    train_df = pd.concat([train_pos, train_neg], ignore_index=True)
    val_df = pd.concat([val_pos, val_neg], ignore_index=True)
    test_df = pd.concat([test_pos, test_neg], ignore_index=True)

    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# DL Model Training
# ---------------------------------------------------------------------------

def train_dl_model(model_name, train_df, val_df, test_df,
                   pooled_orig, pooled_edited, tokens_orig, tokens_edited,
                   structure_delta, epochs=30, lr=1e-3, patience=8, seed=42):
    """Train a DL model and evaluate on test set."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    config = BaselineConfig(model_name=model_name, d_model=640, seed=seed)

    needs_tokens = model_name in ("cross_attention", "diff_attention", "editrna")

    def make_dataset(df):
        ids = df["site_id"].tolist()
        labels = df["label"].values.astype(np.float32)
        return EmbeddingDataset(
            ids, labels, pooled_orig, pooled_edited,
            tokens_orig=tokens_orig if needs_tokens else None,
            tokens_edited=tokens_edited if needs_tokens else None,
            structure_delta=structure_delta,
        )

    train_ds = make_dataset(train_df)
    val_ds = make_dataset(val_df)
    test_ds = make_dataset(test_df)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,
                              num_workers=0, collate_fn=embedding_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False,
                            num_workers=0, collate_fn=embedding_collate_fn)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False,
                             num_workers=0, collate_fn=embedding_collate_fn)

    model = build_model(model_name, config)
    device = torch.device("cpu")
    model = model.to(device)

    loss_fn = FocalLoss(gamma=2.0, alpha=0.75)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_auroc = -1.0
    patience_counter = 0
    best_state = None

    t_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            optimizer.zero_grad()
            output = _forward_baseline(model, batch, model_name)
            logits = output["binary_logit"]
            loss = loss_fn(logits, batch["labels"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Validate
        val_metrics = _evaluate_dl(model, val_loader, model_name, device)
        val_auroc = val_metrics.get("auroc", 0.0)
        if not np.isnan(val_auroc) and val_auroc > best_auroc + 1e-4:
            best_auroc = val_auroc
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    if best_state:
        model.load_state_dict(best_state)

    elapsed = time.time() - t_start

    # Test
    test_metrics = _evaluate_dl(model, test_loader, model_name, device)
    test_metrics["train_time_seconds"] = elapsed

    return test_metrics


@torch.no_grad()
def _evaluate_dl(model, loader, model_name, device):
    """Evaluate a DL model on a DataLoader."""
    model.eval()
    all_targets = []
    all_scores = []

    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        output = _forward_baseline(model, batch, model_name)
        logits = output["binary_logit"].squeeze(-1).cpu().numpy()
        probs = 1.0 / (1.0 + np.exp(-logits))
        all_targets.append(batch["labels"].cpu().numpy())
        all_scores.append(probs)

    y_true = np.concatenate(all_targets)
    y_score = np.concatenate(all_scores)
    return compute_metrics(y_true, y_score)


# ---------------------------------------------------------------------------
# Gradient Boosting Training
# ---------------------------------------------------------------------------

def train_gb_model(variant, train_df, val_df, test_df, splits_df):
    """Train a gradient boosting model and evaluate.

    variant: 'gb_hand' or 'gb_all'
    """
    # Extract features for all sites
    motif_df = extract_motif_features(splits_df)
    struct_df = extract_structure_delta_features(splits_df)
    loop_df = extract_loop_position_features(splits_df)

    motif_cols = [c for c in motif_df.columns if c != "site_id"]
    struct_cols = [c for c in struct_df.columns if c != "site_id"]
    loop_cols = [c for c in loop_df.columns if c != "site_id"]

    hand_feature_cols = motif_cols + struct_cols + loop_cols

    if variant == "gb_all":
        emb_df = extract_embedding_delta_features(splits_df)
        emb_cols = [c for c in emb_df.columns if c != "site_id"]
        feature_cols = hand_feature_cols + emb_cols
    else:
        emb_df = None
        feature_cols = hand_feature_cols

    # Build site_id -> feature dict for fast lookup
    feature_lookup = {}
    for _, row in splits_df.iterrows():
        sid = str(row["site_id"])
        features = []
        # Find matching rows in each feature df
        motif_row = motif_df[motif_df["site_id"] == sid]
        struct_row = struct_df[struct_df["site_id"] == sid]
        loop_row = loop_df[loop_df["site_id"] == sid]

        if len(motif_row) > 0:
            features.extend([motif_row.iloc[0].get(c, 0.0) for c in motif_cols])
        else:
            features.extend([0.0] * len(motif_cols))

        if len(struct_row) > 0:
            features.extend([struct_row.iloc[0].get(c, 0.0) for c in struct_cols])
        else:
            features.extend([0.0] * len(struct_cols))

        if len(loop_row) > 0:
            features.extend([loop_row.iloc[0].get(c, 0.0) for c in loop_cols])
        else:
            features.extend([0.0] * len(loop_cols))

        if emb_df is not None:
            emb_row = emb_df[emb_df["site_id"] == sid]
            if len(emb_row) > 0:
                features.extend([emb_row.iloc[0].get(c, 0.0) for c in emb_cols])
            else:
                features.extend([0.0] * len(emb_cols))

        feature_lookup[sid] = features

    def df_to_arrays(df):
        sids = df["site_id"].tolist()
        X = np.array([feature_lookup.get(str(sid), [0.0] * len(feature_cols)) for sid in sids],
                     dtype=np.float32)
        y = df["label"].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X, y

    X_train, y_train = df_to_arrays(train_df)
    X_val, y_val = df_to_arrays(val_df)
    X_test, y_test = df_to_arrays(test_df)

    # Train
    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    scale_pos_weight = n_neg / max(n_pos, 1)

    clf, backend = get_classifier(len(feature_cols))

    if backend == "xgboost":
        clf.set_params(scale_pos_weight=scale_pos_weight)
    elif backend == "lightgbm":
        clf.set_params(scale_pos_weight=scale_pos_weight)

    t_start = time.time()

    if backend == "sklearn":
        sample_weights = np.where(y_train == 1, scale_pos_weight, 1.0)
        clf.fit(X_train, y_train, sample_weight=sample_weights)
    elif backend == "xgboost":
        clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    else:
        clf.fit(X_train, y_train)

    elapsed = time.time() - t_start

    # Evaluate
    test_proba = clf.predict_proba(X_test)[:, 1]
    test_metrics = compute_binary_metrics(y_test, test_proba)
    test_metrics["train_time_seconds"] = elapsed
    test_metrics["n_positive"] = int(y_test.sum())
    test_metrics["n_negative"] = int(len(y_test) - y_test.sum())

    return test_metrics


# ---------------------------------------------------------------------------
# Majority Class Baseline
# ---------------------------------------------------------------------------

def majority_class_metrics(test_df):
    """Compute metrics for a majority-class (all positive) baseline."""
    y_true = test_df["label"].values.astype(np.float32)
    y_score = np.ones_like(y_true)
    metrics = compute_metrics(y_true, y_score)
    metrics["train_time_seconds"] = 0.0
    return metrics


# ---------------------------------------------------------------------------
# Main Experiment
# ---------------------------------------------------------------------------

def run_single_experiment(model_name, train_df, val_df, test_df, splits_df,
                          pooled_orig, pooled_edited, tokens_orig, tokens_edited,
                          structure_delta):
    """Run a single model on the given train/test split."""
    if model_name == "majority_class":
        return majority_class_metrics(test_df)
    elif model_name in ("gb_hand", "gb_all"):
        return train_gb_model(model_name, train_df, val_df, test_df, splits_df)
    else:
        return train_dl_model(
            model_name, train_df, val_df, test_df,
            pooled_orig, pooled_edited, tokens_orig, tokens_edited,
            structure_delta,
        )


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


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    splits_df = pd.read_csv(SPLITS_CSV)
    pooled_orig, pooled_edited, tokens_orig, tokens_edited, structure_delta = load_embeddings()

    logger.info("Sites per dataset:")
    for ds in DATASETS:
        n_pos = ((splits_df["dataset_source"] == ds) & (splits_df["label"] == 1)).sum()
        logger.info("  %s: %d positives", DATASET_LABELS.get(ds, ds), n_pos)

    # =================================================================
    # Experiment 1: NxN Generalization Matrix (all methods)
    # =================================================================
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 1: NxN generalization matrix (ALL methods)")
    logger.info("=" * 70)

    # For the NxN matrix, we only use models that are fast enough:
    # SubtractionMLP as the representative DL model (fast, good baseline)
    # Plus GB models. The full set takes too long per cell.
    # We do a focused matrix with a subset of representative models.
    MATRIX_MODELS = [
        "majority_class",
        "structure_only",
        "gb_hand",
        "subtraction_mlp",
        "diff_attention",
        "editrna",
    ]

    nxn_results = {}

    for train_ds in TRAINABLE_DATASETS:
        for test_ds in DATASETS:
            train_label = DATASET_LABELS[train_ds]
            test_label = DATASET_LABELS[test_ds]
            pair_key = f"{train_label}->{test_label}"

            logger.info("\n--- %s ---", pair_key)

            train_df, val_df, test_df = create_train_val_test_splits(
                splits_df, pooled_orig,
                train_datasets=[train_ds],
                test_datasets=[test_ds],
            )

            n_train_pos = (train_df["label"] == 1).sum()
            n_test_pos = (test_df["label"] == 1).sum()
            logger.info("  Train: %d pos + %d neg | Test: %d pos + %d neg",
                        n_train_pos, len(train_df) - n_train_pos,
                        n_test_pos, len(test_df) - n_test_pos)

            pair_results = {}

            for model_name in MATRIX_MODELS:
                try:
                    metrics = run_single_experiment(
                        model_name, train_df, val_df, test_df, splits_df,
                        pooled_orig, pooled_edited, tokens_orig, tokens_edited,
                        structure_delta,
                    )
                    pair_results[model_name] = metrics
                    logger.info("  %s: AUROC=%.4f AUPRC=%.4f F1=%.4f",
                                MODEL_DISPLAY[model_name],
                                metrics.get("auroc", 0),
                                metrics.get("auprc", 0),
                                metrics.get("f1", 0))
                except Exception as e:
                    logger.error("  %s: FAILED - %s", model_name, e)
                    pair_results[model_name] = {"auroc": float("nan"), "error": str(e)}

            nxn_results[pair_key] = pair_results

    # =================================================================
    # Experiment 2: Combined training (all methods)
    # =================================================================
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 2: Combined training (ALL methods)")
    logger.info("=" * 70)

    combined_configs = [
        ("Levanon", ["advisor_c2t"]),
        ("Asaoka", ["asaoka_2019"]),
        ("Levanon+Asaoka", ["advisor_c2t", "asaoka_2019"]),
        ("Levanon+Alqassim", ["advisor_c2t", "alqassim_2021"]),
        ("All", ["advisor_c2t", "asaoka_2019", "alqassim_2021"]),
    ]

    combined_results = {}

    for config_name, train_datasets in combined_configs:
        logger.info("\n=== Training on: %s ===", config_name)
        config_results = {}

        for test_ds in DATASETS:
            test_label = DATASET_LABELS[test_ds]
            logger.info("\n  Testing on: %s", test_label)

            train_df, val_df, test_df = create_train_val_test_splits(
                splits_df, pooled_orig,
                train_datasets=train_datasets,
                test_datasets=[test_ds],
            )

            n_test_pos = (test_df["label"] == 1).sum()
            logger.info("    Train: %d | Val: %d | Test: %d (pos=%d)",
                        len(train_df), len(val_df), len(test_df), n_test_pos)

            per_model = {}
            for model_name in ALL_MODEL_NAMES:
                try:
                    metrics = run_single_experiment(
                        model_name, train_df, val_df, test_df, splits_df,
                        pooled_orig, pooled_edited, tokens_orig, tokens_edited,
                        structure_delta,
                    )
                    per_model[model_name] = metrics
                    logger.info("    %s: AUROC=%.4f",
                                MODEL_DISPLAY[model_name],
                                metrics.get("auroc", 0))
                except Exception as e:
                    logger.error("    %s: FAILED - %s", model_name, e)
                    per_model[model_name] = {"auroc": float("nan"), "error": str(e)}

            config_results[test_ds] = per_model

        combined_results[config_name] = config_results

    # =================================================================
    # Print Summary
    # =================================================================
    print("\n" + "=" * 120)
    print("CROSS-DATASET GENERALIZATION — ALL METHODS")
    print("=" * 120)

    # NxN matrix per model (AUROC)
    for model_name in MATRIX_MODELS:
        print(f"\n--- {MODEL_DISPLAY[model_name]} NxN Matrix (AUROC) ---")
        header = f"{'Train \\ Test':<18}"
        for ds in DATASETS:
            header += f" {DATASET_LABELS[ds]:>10}"
        print(header)
        print("-" * len(header))

        for train_ds in TRAINABLE_DATASETS:
            train_label = DATASET_LABELS[train_ds]
            row = f"{train_label:<18}"
            for test_ds in DATASETS:
                test_label = DATASET_LABELS[test_ds]
                pair_key = f"{train_label}->{test_label}"
                auroc = nxn_results.get(pair_key, {}).get(model_name, {}).get("auroc", float("nan"))
                if np.isnan(auroc):
                    row += f" {'N/A':>10}"
                else:
                    row += f" {auroc:>10.4f}"
            print(row)

    # Combined training summary
    print("\n--- Combined Training: Best Model per Config (AUROC) ---")
    header = f"{'Config':<20}"
    for ds in DATASETS:
        header += f" {DATASET_LABELS[ds]:>10}"
    header += f" {'Avg':>10}"
    print(header)
    print("-" * len(header))

    for config_name, config_results in combined_results.items():
        # Find best model across all test datasets for this config
        best_per_test = {}
        for test_ds in DATASETS:
            per_model = config_results.get(test_ds, {})
            best_auroc = -1
            for mn, metrics in per_model.items():
                auroc = metrics.get("auroc", 0)
                if not np.isnan(auroc) and auroc > best_auroc:
                    best_auroc = auroc
            best_per_test[test_ds] = best_auroc

        row = f"{config_name:<20}"
        vals = []
        for ds in DATASETS:
            v = best_per_test.get(ds, float("nan"))
            if np.isnan(v) or v < 0:
                row += f" {'N/A':>10}"
            else:
                row += f" {v:>10.4f}"
                vals.append(v)
        avg = np.mean(vals) if vals else float("nan")
        row += f" {avg:>10.4f}" if not np.isnan(avg) else f" {'N/A':>10}"
        print(row)

    print("=" * 120)

    # =================================================================
    # Save Results
    # =================================================================

    # Build structured output for NxN matrix
    nxn_output = {
        "train_datasets": [DATASET_LABELS[ds] for ds in TRAINABLE_DATASETS],
        "test_datasets": [DATASET_LABELS[ds] for ds in DATASETS],
        "models": [MODEL_DISPLAY[m] for m in MATRIX_MODELS],
        "matrix": {},
    }
    for pair_key, pair_data in nxn_results.items():
        nxn_output["matrix"][pair_key] = {
            MODEL_DISPLAY[mn]: metrics
            for mn, metrics in pair_data.items()
        }

    # Build structured output for combined training
    combined_output = {}
    for config_name, config_data in combined_results.items():
        combined_output[config_name] = {}
        for test_ds, per_model in config_data.items():
            combined_output[config_name][DATASET_LABELS.get(test_ds, test_ds)] = {
                MODEL_DISPLAY[mn]: metrics
                for mn, metrics in per_model.items()
            }

    all_results = {
        "experiment": "cross_dataset_all_methods",
        "matrix_models": [MODEL_DISPLAY[m] for m in MATRIX_MODELS],
        "all_models": [MODEL_DISPLAY[m] for m in ALL_MODEL_NAMES],
        "nxn_matrix": nxn_output,
        "combined_training": combined_output,
    }

    results_path = OUTPUT_DIR / "cross_dataset_all_methods_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=serialize)

    logger.info("\nResults saved to %s", results_path)
    return all_results


if __name__ == "__main__":
    main()
