#!/usr/bin/env python
"""Per-dataset train/test matrix using full EditRNA-A3A model.

Same experiment as exp_dataset_matrix.py but with the full EditRNA-A3A
architecture (cached RNA-FM encoder + edit embedding + gated fusion + multi-task heads).

Usage:
    python experiments/apobec3a/exp_dataset_matrix_editrna.py
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
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

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

EMB_DIR = PROJECT_ROOT / "data" / "processed" / "embeddings"
SPLITS_CSV = PROJECT_ROOT / "data" / "processed" / "splits_expanded.csv"
SEQ_JSON = PROJECT_ROOT / "data" / "processed" / "site_sequences.json"
STRUCT_CACHE = EMB_DIR / "vienna_structure_cache.npz"
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


def load_all_data():
    """Load all shared data once."""
    logger.info("Loading embeddings (this takes a moment)...")
    tokens_orig = torch.load(EMB_DIR / "rnafm_tokens.pt", weights_only=False)
    pooled_orig = torch.load(EMB_DIR / "rnafm_pooled.pt", weights_only=False)
    tokens_edited = torch.load(EMB_DIR / "rnafm_tokens_edited.pt", weights_only=False)
    pooled_edited = torch.load(EMB_DIR / "rnafm_pooled_edited.pt", weights_only=False)
    logger.info("  Loaded %d orig + %d edited embeddings", len(tokens_orig), len(tokens_edited))

    splits_df = pd.read_csv(SPLITS_CSV)

    # Filter to available embeddings
    available = set(tokens_orig.keys()) & set(tokens_edited.keys())
    splits_df = splits_df[splits_df["site_id"].isin(available)].copy()
    logger.info("  %d sites with embeddings", len(splits_df))

    # Load sequences
    sequences = {}
    if SEQ_JSON.exists():
        sequences = json.loads(SEQ_JSON.read_text())
        logger.info("  Loaded %d sequences", len(sequences))

    # Load structure cache
    structure_delta = {}
    if STRUCT_CACHE.exists():
        data = np.load(STRUCT_CACHE, allow_pickle=True)
        feat_key = "delta_features" if "delta_features" in data else "features"
        if "site_ids" in data and feat_key in data:
            sids = data["site_ids"]
            feats = data[feat_key]
            structure_delta = {str(sid): feats[i] for i, sid in enumerate(sids)}
        logger.info("  Loaded %d structure deltas", len(structure_delta))

    return (tokens_orig, pooled_orig, tokens_edited, pooled_edited,
            splits_df, sequences, structure_delta)


def build_samples(df, sequences, structure_delta, window_size=100):
    """Build APOBECSiteSample list from a DataFrame subset."""
    samples = []
    for _, row in df.iterrows():
        site_id = str(row["site_id"])
        label = int(row["label"])

        seq = sequences.get(site_id)
        if seq is None:
            # Generate placeholder
            seq = "A" * (window_size * 2 + 1)

        edit_pos = min(window_size, len(seq) // 2)
        seq_list = list(seq)
        seq_list[edit_pos] = "C"
        seq = "".join(seq_list)

        flanking = get_flanking_context(seq, edit_pos)

        # Structure delta
        struct_d = structure_delta.get(site_id)
        if struct_d is not None:
            struct_d = np.array(struct_d, dtype=np.float32)

        # Editing rate
        editing_rate = row.get("editing_rate", np.nan)
        if pd.notna(editing_rate) and editing_rate > 0:
            if editing_rate > 1.0:
                editing_rate = editing_rate / 100.0
            log2_rate = np.log2(editing_rate + 0.01)
        else:
            log2_rate = float("nan")

        # Concordance: default
        concordance = np.zeros(5, dtype=np.float32)
        if label == 0:
            concordance[0] = 1.0

        sample = APOBECSiteSample(
            sequence=seq,
            edit_pos=edit_pos,
            is_edited=float(label),
            editing_rate_log2=log2_rate if label == 1 else float("nan"),
            apobec_class=-1,
            structure_type=-1,
            tissue_spec_class=-1,
            n_tissues_log2=float("nan"),
            exonic_function=-1,
            conservation=float("nan"),
            cancer_survival=float("nan"),
            tissue_rates=np.full(N_TISSUES, np.nan, dtype=np.float32),
            hek293_rate=float("nan"),
            flanking_context=flanking,
            concordance_features=concordance,
            structure_delta=struct_d,
            site_id=site_id,
            chrom=str(row.get("chr", "")),
            position=int(row.get("start", 0)),
            gene=str(row.get("gene", "")),
        )
        samples.append(sample)
    return samples


def build_model(tokens_orig, pooled_orig, tokens_edited, pooled_edited):
    """Build EditRNA-A3A with cached encoder."""
    cached_encoder = CachedRNAEncoder(
        tokens_cache=tokens_orig,
        pooled_cache=pooled_orig,
        tokens_edited_cache=tokens_edited,
        pooled_edited_cache=pooled_edited,
        d_model=640,
    )
    config = EditRNAConfig(
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
    )
    model = EditRNA_A3A(config=config, primary_encoder=cached_encoder)
    return model


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


def train_and_evaluate(train_df, test_df, sequences, structure_delta,
                       tokens_orig, pooled_orig, tokens_edited, pooled_edited,
                       epochs=40, lr=1e-4, patience=10, seed=42):
    """Train EditRNA-A3A on train_df, evaluate AUROC on test_df."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")

    # Split train_df into train/val
    train_split = train_df[train_df["split"] == "train"]
    val_split = train_df[train_df["split"] == "val"]

    if len(train_split) < 10 or len(val_split) < 5:
        return float("nan")

    # Build sample lists
    data_config = APOBECDataConfig(window_size=100)
    train_samples = build_samples(train_split, sequences, structure_delta)
    val_samples = build_samples(val_split, sequences, structure_delta)
    test_samples = build_samples(test_df, sequences, structure_delta)

    train_ds = APOBECDataset(train_samples, data_config)
    val_ds = APOBECDataset(val_samples, data_config)
    test_ds = APOBECDataset(test_samples, data_config)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                              collate_fn=apobec_collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False,
                            collate_fn=apobec_collate_fn, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False,
                             collate_fn=apobec_collate_fn, num_workers=0)

    # Build model
    model = build_model(tokens_orig, pooled_orig, tokens_edited, pooled_edited)
    model = model.to(device)

    optimizer = AdamW(model.get_parameter_groups(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)

    best_auroc = -1.0
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

        # Validate
        model.eval()
        all_y, all_s = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch_to_device(batch, device)
                output = model(batch)
                logits = output["predictions"]["binary_logit"].squeeze(-1)
                probs = torch.sigmoid(logits).cpu().numpy()
                targets = batch["targets"]["binary"].cpu().numpy()
                all_y.append(targets)
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
            batch = batch_to_device(batch, device)
            output = model(batch)
            logits = output["predictions"]["binary_logit"].squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
            targets = batch["targets"]["binary"].cpu().numpy()
            all_y.append(targets)
            all_s.append(probs)
    y_true = np.concatenate(all_y)
    y_score = np.concatenate(all_s)

    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    torch.manual_seed(42)

    # Load all data once
    (tokens_orig, pooled_orig, tokens_edited, pooled_edited,
     splits_df, sequences, structure_delta) = load_all_data()

    # Separate negatives
    neg_df = splits_df[splits_df["dataset_source"].isin(NEGATIVE_SOURCES)]
    neg_train_val = neg_df[neg_df["split"].isin(["train", "val"])]
    neg_test = neg_df[neg_df["split"] == "test"]
    logger.info("Negatives: %d train+val, %d test", len(neg_train_val), len(neg_test))

    # Training configs
    train_configs = DATASET_SOURCES + ["All"]
    train_labels = [DATASET_LABELS[ds] for ds in DATASET_SOURCES] + ["All"]
    test_labels = [DATASET_LABELS[ds] for ds in DATASET_SOURCES]

    n_train = len(train_configs)
    n_test = len(DATASET_SOURCES)
    matrix = np.full((n_train, n_test), float("nan"))

    logger.info("\nRunning %d x %d = %d experiments (EditRNA-A3A)...\n",
                n_train, n_test, n_train * n_test)
    t_start = time.time()

    for i, train_cfg in enumerate(train_configs):
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

        train_data = pd.concat([train_pos, neg_train_val], ignore_index=True)

        for j, test_ds in enumerate(DATASET_SOURCES):
            test_pos = splits_df[
                (splits_df["dataset_source"] == test_ds) &
                (splits_df["label"] == 1) &
                (splits_df["split"] == "test")
            ]
            test_data = pd.concat([test_pos, neg_test], ignore_index=True)

            if len(test_pos) < 3:
                logger.info("  [%s -> %s] Skipping: only %d test positives",
                            train_labels[i], test_labels[j], len(test_pos))
                continue

            t0 = time.time()
            auroc = train_and_evaluate(
                train_data, test_data, sequences, structure_delta,
                tokens_orig, pooled_orig, tokens_edited, pooled_edited,
            )
            matrix[i, j] = auroc
            elapsed_pair = time.time() - t0

            n_train_pos = (train_data["label"] == 1).sum()
            logger.info("  [%s -> %s] AUROC=%.4f  (%d pos, %.0fs)",
                        train_labels[i], test_labels[j], auroc,
                        n_train_pos, elapsed_pair)

    elapsed = time.time() - t_start
    logger.info("\nTotal time: %.1fs (%.1fs per experiment)", elapsed, elapsed / (n_train * n_test))

    # Print matrix
    print("\n" + "=" * 90)
    print("DATASET TRAIN/TEST MATRIX (AUROC) — EditRNA-A3A")
    print("=" * 90)
    header = f"{'Train / Test':<15}" + "".join(f"{tl:>12}" for tl in test_labels)
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

    # Generate heatmap
    fig, ax = plt.subplots(figsize=(10, 7))
    masked = np.ma.masked_invalid(matrix)
    im = ax.imshow(masked, cmap="YlOrRd", aspect="auto", vmin=0.5, vmax=1.0)

    ax.set_xticks(range(n_test))
    ax.set_xticklabels(test_labels, fontsize=11)
    ax.set_yticks(range(n_train))
    ax.set_yticklabels(train_labels, fontsize=11)
    ax.set_xlabel("Test Dataset", fontsize=13)
    ax.set_ylabel("Training Dataset", fontsize=13)
    ax.set_title("Cross-Dataset AUROC (EditRNA-A3A)", fontsize=14, fontweight="bold")

    for i in range(n_train):
        for j in range(n_test):
            val = matrix[i, j]
            if not np.isnan(val):
                color = "white" if val > 0.85 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=10, fontweight="bold", color=color)

    plt.colorbar(im, ax=ax, label="AUROC", shrink=0.8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "dataset_matrix_editrna_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Save results
    results = {
        "model": "EditRNA-A3A",
        "train_configs": train_labels,
        "test_datasets": test_labels,
        "auroc_matrix": matrix.tolist(),
        "total_time_seconds": elapsed,
        "per_pair": {},
    }
    for i, tl in enumerate(train_labels):
        for j, te in enumerate(test_labels):
            val = matrix[i, j]
            results["per_pair"][f"{tl}->{te}"] = float(val) if not np.isnan(val) else None

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

    with open(OUTPUT_DIR / "dataset_matrix_editrna_results.json", "w") as f:
        json.dump(results, f, indent=2, default=serialize)

    logger.info("Saved results and heatmap to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
