#!/usr/bin/env python
"""Incremental dataset addition: Levanon test set performance.

Shows how test performance on the Levanon (advisor_c2t) test split changes
as we incrementally add training datasets. The test set is ALWAYS the
Levanon test split (positives + shared negatives).

Training configurations (in order of increasing data):
  1. Levanon only
  2. Levanon + Asaoka
  3. Levanon + Baysal
  4. Levanon + Asaoka + Baysal
  5. Levanon + Asaoka + Alqassim
  6. Levanon + Asaoka + Baysal + Alqassim (All)

For EACH configuration, trains ALL model types:
  - StructureOnly       (GB on 16-dim: 7 structure delta + 9 loop features)
  - GB_HandFeatures     (GB on ~40 hand-crafted features)
  - GB_AllFeatures      (GB on hand-crafted + 640-dim embedding delta)
  - PooledMLP           (MLP on 640-dim pooled original embedding)
  - SubtractionMLP      (MLP on 640-dim pooled embedding delta)
  - EditRNA_A3A         (gated fusion, pooled-only mode)

Data approach:
  - Positives: is_edited == 1 from specified dataset_sources
  - Negatives: tier2_negative + tier3_negative rows from train+val / test splits
  - Test: Levanon positives from test split + negatives from test split
  - Training: positives from specified datasets (train+val splits) + negatives (train+val)

Usage:
    conda activate quris
    python experiments/apobec/exp_incremental_levanon_test.py
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "apobec"))

from exp_cross_dataset_full import (
    extract_structure_delta_for_ids,
    extract_loop_features_for_ids,
    extract_hand_features,
    extract_all_features,
    extract_embedding_delta_for_ids,
    train_gb_model,
    build_dl_model,
    train_dl_model,
    EmbeddingDataset,
    embedding_collate_fn,
    FocalLoss,
    load_structure_delta_dict,
    load_sequences,
    load_loop_position_df,
    _forward_model,
    _serialize,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
EMB_DIR = PROJECT_ROOT / "data" / "processed" / "embeddings"
SPLITS_CSV = PROJECT_ROOT / "data" / "processed" / "splits_expanded.csv"
OUTPUT_DIR = (
    PROJECT_ROOT / "experiments" / "apobec" / "outputs" / "incremental_levanon"
)

NEGATIVE_SOURCES = ["tier2_negative", "tier3_negative"]

SEED = 42

# ---------------------------------------------------------------------------
# Training configurations (in order of increasing data)
# ---------------------------------------------------------------------------
TRAIN_CONFIGS = [
    {
        "name": "Levanon",
        "datasets": ["advisor_c2t"],
    },
    {
        "name": "Levanon + Asaoka",
        "datasets": ["advisor_c2t", "asaoka_2019"],
    },
    {
        "name": "Levanon + Baysal",
        "datasets": ["advisor_c2t", "baysal_2016"],
    },
    {
        "name": "Levanon + Asaoka + Baysal",
        "datasets": ["advisor_c2t", "asaoka_2019", "baysal_2016"],
    },
    {
        "name": "Levanon + Asaoka + Alqassim",
        "datasets": ["advisor_c2t", "asaoka_2019", "alqassim_2021"],
    },
    {
        "name": "Levanon + Asaoka + Baysal + Alqassim (All)",
        "datasets": ["advisor_c2t", "asaoka_2019", "baysal_2016", "alqassim_2021"],
    },
]

MODEL_ORDER = [
    "StructureOnly",
    "GB_HandFeatures",
    "GB_AllFeatures",
    "PooledMLP",
    "SubtractionMLP",
    "EditRNA_A3A",
]


# ---------------------------------------------------------------------------
# Extended GB training that returns both AUROC and AUPRC
# ---------------------------------------------------------------------------
def train_gb_model_extended(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    """Train XGBoost classifier and return test AUROC and AUPRC."""
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan")}

    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    scale_pos_weight = n_neg / max(n_pos, 1)

    try:
        import xgboost as xgb
        clf = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            scale_pos_weight=scale_pos_weight,
            random_state=SEED,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=1,
            verbosity=0,
        )
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        clf = GradientBoostingClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=SEED,
        )

    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)[:, 1]
    auroc = float(roc_auc_score(y_test, probs))
    auprc = float(average_precision_score(y_test, probs))
    return {"auroc": auroc, "auprc": auprc}


# ---------------------------------------------------------------------------
# Extended DL training that returns both AUROC and AUPRC
# ---------------------------------------------------------------------------
def train_dl_model_extended(
    model,
    model_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    pooled_orig: Dict,
    pooled_edited: Dict,
    structure_delta: Optional[Dict],
    epochs: int = 50,
    lr: float = 1e-3,
    patience: int = 10,
    batch_size: int = 64,
) -> Dict[str, float]:
    """Train a DL model on train_df, evaluate on test_df. Returns dict with AUROC and AUPRC."""
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.utils.data import DataLoader

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    train_split = train_df[train_df["split"] == "train"]
    val_split = train_df[train_df["split"] == "val"]

    if len(train_split) < 10 or len(val_split) < 5:
        return {"auroc": float("nan"), "auprc": float("nan")}

    def make_loader(df, shuffle=False):
        ds = EmbeddingDataset(
            df["site_id"].tolist(),
            df["label"].values.astype(np.float32),
            pooled_orig, pooled_edited, structure_delta,
        )
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=0, drop_last=False, collate_fn=embedding_collate_fn,
        )

    train_loader = make_loader(train_split, shuffle=True)
    val_loader = make_loader(val_split)
    test_loader = make_loader(test_df)

    device = torch.device("cpu")
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)
    loss_fn = FocalLoss()

    best_auroc = -1.0
    patience_counter = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            optimizer.zero_grad()
            logits = _forward_model(model, batch, model_name)
            loss = loss_fn(logits, batch["labels"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # Validate
        model.eval()
        all_y, all_s = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                logits = _forward_model(model, batch, model_name)
                probs = torch.sigmoid(logits.squeeze(-1)).numpy()
                all_y.append(batch["labels"].numpy())
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
        for batch in test_loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            logits = _forward_model(model, batch, model_name)
            probs = torch.sigmoid(logits.squeeze(-1)).numpy()
            all_y.append(batch["labels"].numpy())
            all_s.append(probs)

    y_true = np.concatenate(all_y)
    y_score = np.concatenate(all_s)
    if len(np.unique(y_true)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan")}

    auroc = float(roc_auc_score(y_true, y_score))
    auprc = float(average_precision_score(y_true, y_score))
    return {"auroc": auroc, "auprc": auprc}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # ==================================================================
    # Load all data
    # ==================================================================
    logger.info("Loading data...")

    # Splits
    splits_df = pd.read_csv(SPLITS_CSV)

    # Pooled embeddings
    pooled_orig = torch.load(EMB_DIR / "rnafm_pooled.pt", weights_only=False)
    pooled_edited = torch.load(EMB_DIR / "rnafm_pooled_edited.pt", weights_only=False)

    # Filter to sites with embeddings
    available_ids = set(pooled_orig.keys()) & set(pooled_edited.keys())
    splits_df = splits_df[splits_df["site_id"].isin(available_ids)].copy()
    logger.info("Splits after embedding filter: %d sites", len(splits_df))

    # Structure delta
    structure_delta_dict = load_structure_delta_dict()
    logger.info("Structure delta dict: %d sites", len(structure_delta_dict))

    # Sequences & loop features (for GB models)
    sequences = load_sequences()
    loop_df = load_loop_position_df()
    logger.info("Sequences: %d, Loop features: %d rows",
                len(sequences), len(loop_df))

    # ==================================================================
    # Separate negatives (shared across all configs)
    # ==================================================================
    neg_df = splits_df[splits_df["dataset_source"].isin(NEGATIVE_SOURCES)]
    neg_train_val = neg_df[neg_df["split"].isin(["train", "val"])]
    neg_test = neg_df[neg_df["split"] == "test"]
    logger.info("Negatives: %d train+val, %d test", len(neg_train_val), len(neg_test))

    # ==================================================================
    # Build fixed Levanon test set
    # ==================================================================
    levanon_test_pos = splits_df[
        (splits_df["dataset_source"] == "advisor_c2t")
        & (splits_df["label"] == 1)
        & (splits_df["split"] == "test")
    ]
    test_df = pd.concat([levanon_test_pos, neg_test], ignore_index=True)
    test_n_pos = int((test_df["label"] == 1).sum())
    test_n_neg = int((test_df["label"] == 0).sum())
    logger.info("Fixed Levanon test set: %d pos + %d neg = %d total",
                test_n_pos, test_n_neg, len(test_df))

    # ==================================================================
    # Build training DataFrames for each configuration
    # ==================================================================
    train_dfs = {}
    for cfg in TRAIN_CONFIGS:
        train_pos = splits_df[
            (splits_df["dataset_source"].isin(cfg["datasets"]))
            & (splits_df["label"] == 1)
            & (splits_df["split"].isin(["train", "val"]))
        ]
        train_data = pd.concat([train_pos, neg_train_val], ignore_index=True)
        train_dfs[cfg["name"]] = train_data
        n_p = int((train_data["label"] == 1).sum())
        n_n = int((train_data["label"] == 0).sum())
        logger.info("  Train [%s]: %d pos + %d neg = %d total",
                     cfg["name"], n_p, n_n, len(train_data))

    # ==================================================================
    # Run experiments
    # ==================================================================
    configs_results = []
    t_total_start = time.time()

    for cfg in TRAIN_CONFIGS:
        cfg_name = cfg["name"]
        train_data = train_dfs[cfg_name]
        n_train_pos = int((train_data["label"] == 1).sum())
        n_train_neg = int((train_data["label"] == 0).sum())

        logger.info("\n" + "=" * 80)
        logger.info("CONFIG: %s (%d pos + %d neg)", cfg_name, n_train_pos, n_train_neg)
        logger.info("=" * 80)

        models_results = {}

        for model_name in MODEL_ORDER:
            t_model_start = time.time()
            logger.info("  Training %s...", model_name)

            # ---- GB models ----
            if model_name in ("StructureOnly", "GB_HandFeatures", "GB_AllFeatures"):
                train_ids = train_data["site_id"].tolist()
                y_train = train_data["label"].values.astype(np.float32)
                test_ids = test_df["site_id"].tolist()
                y_test = test_df["label"].values.astype(np.float32)

                if model_name == "StructureOnly":
                    X_train = np.hstack([
                        extract_structure_delta_for_ids(train_ids, structure_delta_dict),
                        extract_loop_features_for_ids(train_ids, loop_df),
                    ])
                    X_train = np.nan_to_num(X_train, nan=0.0)
                    X_test = np.hstack([
                        extract_structure_delta_for_ids(test_ids, structure_delta_dict),
                        extract_loop_features_for_ids(test_ids, loop_df),
                    ])
                    X_test = np.nan_to_num(X_test, nan=0.0)
                elif model_name == "GB_HandFeatures":
                    X_train = extract_hand_features(
                        train_ids, sequences, structure_delta_dict, loop_df
                    )
                    X_test = extract_hand_features(
                        test_ids, sequences, structure_delta_dict, loop_df
                    )
                else:  # GB_AllFeatures
                    X_train = extract_all_features(
                        train_ids, sequences, structure_delta_dict, loop_df,
                        pooled_orig, pooled_edited,
                    )
                    X_test = extract_all_features(
                        test_ids, sequences, structure_delta_dict, loop_df,
                        pooled_orig, pooled_edited,
                    )

                result = train_gb_model_extended(X_train, y_train, X_test, y_test)

            # ---- DL models ----
            else:
                model = build_dl_model(model_name, pooled_orig, pooled_edited)
                result = train_dl_model_extended(
                    model=model,
                    model_name=model_name,
                    train_df=train_data,
                    test_df=test_df,
                    pooled_orig=pooled_orig,
                    pooled_edited=pooled_edited,
                    structure_delta=structure_delta_dict,
                    epochs=50,
                    lr=1e-3,
                    patience=10,
                    batch_size=64,
                )

            t_elapsed = time.time() - t_model_start
            models_results[model_name] = result
            logger.info("    %s -> AUROC=%.4f  AUPRC=%.4f  (%.1fs)",
                        model_name, result["auroc"], result["auprc"], t_elapsed)

        configs_results.append({
            "name": cfg_name,
            "datasets": cfg["datasets"],
            "n_train_pos": n_train_pos,
            "n_train_neg": n_train_neg,
            "models": models_results,
        })

    t_total = time.time() - t_total_start
    logger.info("\nTotal time: %.1fs", t_total)

    # ==================================================================
    # Print summary table
    # ==================================================================
    print("\n" + "=" * 120)
    print("INCREMENTAL LEVANON TEST -- AUROC Summary")
    print(f"Test set: Levanon ({test_n_pos} pos + {test_n_neg} neg)")
    print("=" * 120)

    header = f"{'Training Config':<45}" + "".join(f"{m:>16}" for m in MODEL_ORDER)
    print(header)
    print("-" * 120)

    for cfg_result in configs_results:
        row = f"{cfg_result['name']:<45}"
        for m in MODEL_ORDER:
            auroc = cfg_result["models"].get(m, {}).get("auroc", float("nan"))
            if isinstance(auroc, float) and np.isnan(auroc):
                row += f"{'N/A':>16}"
            else:
                row += f"{auroc:>16.4f}"
        print(row)

    print("=" * 120)

    print("\n" + "=" * 120)
    print("INCREMENTAL LEVANON TEST -- AUPRC Summary")
    print(f"Test set: Levanon ({test_n_pos} pos + {test_n_neg} neg)")
    print("=" * 120)

    header = f"{'Training Config':<45}" + "".join(f"{m:>16}" for m in MODEL_ORDER)
    print(header)
    print("-" * 120)

    for cfg_result in configs_results:
        row = f"{cfg_result['name']:<45}"
        for m in MODEL_ORDER:
            auprc = cfg_result["models"].get(m, {}).get("auprc", float("nan"))
            if isinstance(auprc, float) and np.isnan(auprc):
                row += f"{'N/A':>16}"
            else:
                row += f"{auprc:>16.4f}"
        print(row)

    print("=" * 120)

    # ==================================================================
    # Save results
    # ==================================================================
    output = {
        "experiment": "incremental_levanon_test",
        "test_dataset": "Levanon",
        "test_n_pos": test_n_pos,
        "test_n_neg": test_n_neg,
        "configs": configs_results,
        "total_time_seconds": t_total,
    }

    results_path = OUTPUT_DIR / "incremental_levanon_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=_serialize)

    logger.info("Results saved to %s", results_path)
    logger.info("All outputs saved to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
