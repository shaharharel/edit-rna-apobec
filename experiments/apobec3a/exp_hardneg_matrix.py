#!/usr/bin/env python
"""Cross-dataset matrix with structure-matched HARD negatives.

Uses only the 631 structure-matched negatives (|delta_MFE| <= 0.1) instead of
all Tier2/Tier3 negatives. This removes the trivial structural signature that
inflated previous cross-dataset AUROC scores.

Runs both SubtractionMLP and EditRNA-A3A for comparison.

Usage:
    python experiments/apobec3a/exp_hardneg_matrix.py
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
from sklearn.metrics import roc_auc_score, average_precision_score
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

EMB_DIR = PROJECT_ROOT / "data" / "processed" / "embeddings"
SPLITS_CSV = PROJECT_ROOT / "data" / "processed" / "splits_expanded.csv"
HARDNEG_CSV = PROJECT_ROOT / "data" / "processed" / "hardneg_per_dataset.csv"
HARDNEG_SEQ_JSON = PROJECT_ROOT / "data" / "processed" / "hardneg_site_sequences.json"
HARDNEG_STRUCT_CACHE = EMB_DIR / "hardneg_vienna_structure.npz"
SEQ_JSON = PROJECT_ROOT / "data" / "processed" / "site_sequences.json"
STRUCT_CACHE = EMB_DIR / "vienna_structure_cache.npz"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "apobec3a" / "outputs" / "hardneg_matrix"

DATASET_SOURCES = ["advisor_c2t", "asaoka_2019", "alqassim_2021", "sharma_2015", "baysal_2016"]
DATASET_LABELS = {
    "advisor_c2t": "Levanon",
    "asaoka_2019": "Asaoka",
    "alqassim_2021": "Alqassim",
    "sharma_2015": "Sharma",
    "baysal_2016": "Baysal",
}


# ---------------------------------------------------------------------------
# SubtractionMLP model
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
# SubtractionMLP training
# ---------------------------------------------------------------------------

def train_eval_subtraction(train_df, test_df, pooled_orig, pooled_edited,
                           epochs=50, lr=1e-3, patience=10, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_split = train_df[train_df["split"] == "train"]
    val_split = train_df[train_df["split"] == "val"]
    if len(train_split) < 10 or len(val_split) < 5:
        return float("nan"), float("nan")

    def make_loader(df, shuffle=False):
        ds = DiffDataset(df["site_id"].tolist(), df["label"].values.astype(np.float32),
                         pooled_orig, pooled_edited)
        return DataLoader(ds, batch_size=64, shuffle=shuffle, num_workers=0)

    train_loader = make_loader(train_split, shuffle=True)
    val_loader = make_loader(val_split)
    test_loader = make_loader(test_df)

    model = SubtractionMLP()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)
    loss_fn = FocalLoss()

    best_auroc, patience_counter, best_state = -1.0, 0, None

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
                probs = torch.sigmoid(model(diffs)).numpy()
                all_y.append(labels.numpy())
                all_s.append(probs)
        y_true, y_score = np.concatenate(all_y), np.concatenate(all_s)
        val_auroc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) >= 2 else 0.0

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

    model.eval()
    all_y, all_s = [], []
    with torch.no_grad():
        for diffs, labels in test_loader:
            probs = torch.sigmoid(model(diffs)).numpy()
            all_y.append(labels.numpy())
            all_s.append(probs)
    y_true, y_score = np.concatenate(all_y), np.concatenate(all_s)

    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")
    auroc = float(roc_auc_score(y_true, y_score))
    auprc = float(average_precision_score(y_true, y_score))
    return auroc, auprc


# ---------------------------------------------------------------------------
# EditRNA-A3A training
# ---------------------------------------------------------------------------

def build_samples(df, sequences, structure_delta, window_size=100):
    samples = []
    for _, row in df.iterrows():
        site_id = str(row["site_id"])
        label = int(row["label"])

        seq = sequences.get(site_id, "A" * (window_size * 2 + 1))
        edit_pos = min(window_size, len(seq) // 2)
        seq_list = list(seq)
        seq_list[edit_pos] = "C"
        seq = "".join(seq_list)

        flanking = get_flanking_context(seq, edit_pos)

        struct_d = structure_delta.get(site_id)
        if struct_d is not None:
            struct_d = np.array(struct_d, dtype=np.float32)

        editing_rate = row.get("editing_rate", np.nan)
        if pd.notna(editing_rate) and editing_rate > 0:
            if editing_rate > 1.0:
                editing_rate = editing_rate / 100.0
            log2_rate = np.log2(editing_rate + 0.01)
        else:
            log2_rate = float("nan")

        concordance = np.zeros(5, dtype=np.float32)
        if label == 0:
            concordance[0] = 1.0

        sample = APOBECSiteSample(
            sequence=seq, edit_pos=edit_pos,
            is_edited=float(label),
            editing_rate_log2=log2_rate if label == 1 else float("nan"),
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


def train_eval_editrna(train_df, test_df, sequences, structure_delta,
                       tokens_orig, pooled_orig, tokens_edited, pooled_edited,
                       epochs=40, lr=1e-4, patience=10, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")

    train_split = train_df[train_df["split"] == "train"]
    val_split = train_df[train_df["split"] == "val"]
    if len(train_split) < 10 or len(val_split) < 5:
        return float("nan"), float("nan")

    data_config = APOBECDataConfig(window_size=100)
    train_ds = APOBECDataset(build_samples(train_split, sequences, structure_delta), data_config)
    val_ds = APOBECDataset(build_samples(val_split, sequences, structure_delta), data_config)
    test_ds = APOBECDataset(build_samples(test_df, sequences, structure_delta), data_config)

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

    best_auroc, patience_counter, best_state = -1.0, 0, None

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
        all_y, all_s = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch_to_device(batch, device)
                output = model(batch)
                logits = output["predictions"]["binary_logit"].squeeze(-1)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_y.append(batch["targets"]["binary"].cpu().numpy())
                all_s.append(probs)
        y_true, y_score = np.concatenate(all_y), np.concatenate(all_s)
        val_auroc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) >= 2 else 0.0

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

    model.eval()
    all_y, all_s = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch_to_device(batch, device)
            output = model(batch)
            logits = output["predictions"]["binary_logit"].squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_y.append(batch["targets"]["binary"].cpu().numpy())
            all_s.append(probs)
    y_true, y_score = np.concatenate(all_y), np.concatenate(all_s)

    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")
    auroc = float(roc_auc_score(y_true, y_score))
    auprc = float(average_precision_score(y_true, y_score))
    return auroc, auprc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)
    torch.manual_seed(42)

    # Load data
    logger.info("Loading embeddings...")
    pooled_orig = torch.load(EMB_DIR / "rnafm_pooled.pt", weights_only=False)
    pooled_edited = torch.load(EMB_DIR / "rnafm_pooled_edited.pt", weights_only=False)

    # Check if hardneg token embeddings exist before loading tokens
    has_hardneg_tokens = (EMB_DIR / "hardneg_rnafm_tokens.pt").exists()
    run_editrna = has_hardneg_tokens and (EMB_DIR / "rnafm_tokens.pt").exists()
    tokens_orig = None
    tokens_edited = None
    if run_editrna:
        tokens_orig = torch.load(EMB_DIR / "rnafm_tokens.pt", weights_only=False)
        tokens_edited = torch.load(EMB_DIR / "rnafm_tokens_edited.pt", weights_only=False)
    else:
        logger.warning("Skipping EditRNA (hardneg token embeddings not available)")

    # Merge hard-negative embeddings
    merge_targets = [("pooled", pooled_orig), ("pooled_edited", pooled_edited)]
    if run_editrna:
        merge_targets.extend([("tokens", tokens_orig), ("tokens_edited", tokens_edited)])
    for name, target in merge_targets:
        hn_path = EMB_DIR / f"hardneg_rnafm_{name}.pt"
        if hn_path.exists():
            hn_emb = torch.load(hn_path, weights_only=False)
            target.update(hn_emb)
            logger.info("  Merged %d hard-negative %s embeddings", len(hn_emb), name)

    splits_df = pd.read_csv(SPLITS_CSV)
    if not HARDNEG_CSV.exists():
        logger.error("Hard-negative CSV not found: %s", HARDNEG_CSV)
        logger.error("Run scripts/apobec3a/generate_hardneg_pipeline.py first")
        sys.exit(1)
    hardneg_df = pd.read_csv(HARDNEG_CSV)

    # Load sequences and structure (both regular + hard-negative)
    sequences = {}
    if SEQ_JSON.exists():
        sequences = json.loads(SEQ_JSON.read_text())
    if HARDNEG_SEQ_JSON.exists():
        hn_seq = json.loads(HARDNEG_SEQ_JSON.read_text())
        sequences.update(hn_seq)
        logger.info("  Merged %d hard-negative sequences", len(hn_seq))

    structure_delta = {}
    for cache_path in [STRUCT_CACHE, HARDNEG_STRUCT_CACHE]:
        if cache_path.exists():
            data = np.load(cache_path, allow_pickle=True)
            feat_key = "delta_features" if "delta_features" in data else "features"
            if "site_ids" in data and feat_key in data:
                sids = data["site_ids"]
                feats = data[feat_key]
                for i, sid in enumerate(sids):
                    structure_delta[str(sid)] = feats[i]

    # Filter to available embeddings
    available = set(pooled_orig.keys()) & set(pooled_edited.keys())
    splits_df = splits_df[splits_df["site_id"].isin(available)].copy()

    # Per-dataset structure-matched hard negatives (splits already assigned in CSV)
    hardneg_df = hardneg_df[hardneg_df["site_id"].isin(available)].copy()
    neg_train_val = hardneg_df[hardneg_df["split"].isin(["train", "val"])].copy()
    neg_test = hardneg_df[hardneg_df["split"] == "test"].copy()
    logger.info("Hard negatives: %d train+val, %d test", len(neg_train_val), len(neg_test))

    # Run matrix
    train_configs = DATASET_SOURCES + ["All"]
    train_labels = [DATASET_LABELS[ds] for ds in DATASET_SOURCES] + ["All"]
    test_labels = [DATASET_LABELS[ds] for ds in DATASET_SOURCES]

    n_train = len(train_configs)
    n_test = len(DATASET_SOURCES)

    # Two matrices: SubtractionMLP and EditRNA-A3A
    matrix_sub = np.full((n_train, n_test), float("nan"))
    matrix_edit = np.full((n_train, n_test), float("nan"))
    matrix_sub_auprc = np.full((n_train, n_test), float("nan"))
    matrix_edit_auprc = np.full((n_train, n_test), float("nan"))

    logger.info("\nRunning %d x %d = %d experiments × 2 models...\n", n_train, n_test, n_train * n_test)
    t_start = time.time()

    for i, train_cfg in enumerate(train_configs):
        if train_cfg == "All":
            train_pos = splits_df[
                (splits_df["label"] == 1) & (splits_df["split"].isin(["train", "val"]))
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

            # SubtractionMLP
            auroc_sub, auprc_sub = train_eval_subtraction(
                train_data, test_data, pooled_orig, pooled_edited
            )
            matrix_sub[i, j] = auroc_sub
            matrix_sub_auprc[i, j] = auprc_sub

            # EditRNA-A3A (only if token embeddings available)
            if run_editrna:
                auroc_edit, auprc_edit = train_eval_editrna(
                    train_data, test_data, sequences, structure_delta,
                    tokens_orig, pooled_orig, tokens_edited, pooled_edited,
                )
                matrix_edit[i, j] = auroc_edit
                matrix_edit_auprc[i, j] = auprc_edit
            else:
                auroc_edit = float("nan")

            elapsed_pair = time.time() - t0
            logger.info("  [%s -> %s] Sub=%.4f  Edit=%.4f  (%d pos + %d neg test, %.0fs)",
                        train_labels[i], test_labels[j],
                        auroc_sub, auroc_edit,
                        len(test_pos), len(neg_test), elapsed_pair)

    elapsed = time.time() - t_start
    logger.info("\nTotal time: %.1fs", elapsed)

    # ===================================================================
    # Print matrices
    # ===================================================================
    for name, matrix in [("SubtractionMLP", matrix_sub), ("EditRNA-A3A", matrix_edit)]:
        print(f"\n{'=' * 90}")
        print(f"HARD-NEGATIVE MATRIX (AUROC) — {name}")
        print(f"{'=' * 90}")
        header = f"{'Train / Test':<15}" + "".join(f"{tl:>12}" for tl in test_labels)
        print(header)
        print("-" * 90)
        for i, tl in enumerate(train_labels):
            row = f"{tl:<15}"
            for j in range(n_test):
                val = matrix[i, j]
                row += f"{'N/A':>12}" if np.isnan(val) else f"{val:>12.4f}"
            print(row)
        print("=" * 90)

    # Comparison: hard-neg vs original (for reference)
    print(f"\n{'=' * 90}")
    print("COMPARISON: AUROC drop with hard negatives")
    print(f"{'=' * 90}")
    print("(Positive values = harder negatives reduced AUROC, as expected)")

    # ===================================================================
    # Generate heatmaps
    # ===================================================================
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    for ax, matrix_data, title in [
        (axes[0], matrix_sub, "SubtractionMLP"),
        (axes[1], matrix_edit, "EditRNA-A3A"),
    ]:
        masked = np.ma.masked_invalid(matrix_data)
        im = ax.imshow(masked, cmap="YlOrRd", aspect="auto", vmin=0.5, vmax=1.0)
        ax.set_xticks(range(n_test))
        ax.set_xticklabels(test_labels, fontsize=10)
        ax.set_yticks(range(n_train))
        ax.set_yticklabels(train_labels, fontsize=10)
        ax.set_xlabel("Test Dataset", fontsize=12)
        ax.set_ylabel("Training Dataset", fontsize=12)
        ax.set_title(f"{title} — Hard Negatives", fontsize=13, fontweight="bold")

        for ii in range(n_train):
            for jj in range(n_test):
                val = matrix_data[ii, jj]
                if not np.isnan(val):
                    color = "white" if val > 0.85 else "black"
                    ax.text(jj, ii, f"{val:.3f}", ha="center", va="center",
                            fontsize=9, fontweight="bold", color=color)
        plt.colorbar(im, ax=ax, label="AUROC", shrink=0.8)

    plt.suptitle("Cross-Dataset AUROC with Structure-Matched Hard Negatives",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "hardneg_matrix_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved heatmap to %s", OUTPUT_DIR / "hardneg_matrix_heatmap.png")

    # ===================================================================
    # Save results
    # ===================================================================
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

    results = {
        "description": "Cross-dataset matrix with structure-matched hard negatives (|delta_MFE| <= 0.1)",
        "n_hardneg_train_val": int(len(neg_train_val)),
        "n_hardneg_test": int(len(neg_test)),
        "subtraction_mlp": {
            "auroc_matrix": matrix_sub.tolist(),
            "auprc_matrix": matrix_sub_auprc.tolist(),
        },
        "editrna_a3a": {
            "auroc_matrix": matrix_edit.tolist(),
            "auprc_matrix": matrix_edit_auprc.tolist(),
        },
        "train_configs": train_labels,
        "test_datasets": test_labels,
        "total_time_seconds": elapsed,
    }

    with open(OUTPUT_DIR / "hardneg_matrix_results.json", "w") as f:
        json.dump(results, f, indent=2, default=serialize)

    logger.info("Saved results to %s", OUTPUT_DIR / "hardneg_matrix_results.json")


if __name__ == "__main__":
    main()
