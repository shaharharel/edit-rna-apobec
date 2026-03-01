#!/usr/bin/env python
"""Evaluate all 7 baselines with structure-matched hard negatives.

Compares classification performance using:
  - Original tier2/tier3 negatives (structurally easy, delta-MFE ~ 0.4-1.0)
  - Structure-matched hard negatives (|delta-MFE| <= 0.1, matching positives)

Key question: does the architecture ranking change when negatives are harder?
Expected: structure_only model drops most (relies on delta-MFE).

Prerequisites:
  - Run generate_hardneg_pipeline.py (produces hardneg_per_dataset.csv)
  - Generate hard negative embeddings (produces hardneg_rnafm_*.pt)

Usage:
    python experiments/apobec/exp_hardneg_baselines.py
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
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logger = logging.getLogger(__name__)

EMB_DIR = PROJECT_ROOT / "data" / "processed" / "embeddings"
SPLITS_CSV = PROJECT_ROOT / "data" / "processed" / "splits_expanded.csv"
SEQ_JSON = PROJECT_ROOT / "data" / "processed" / "site_sequences.json"
STRUCT_CACHE = EMB_DIR / "vienna_structure_cache.npz"
HARDNEG_CSV = PROJECT_ROOT / "data" / "processed" / "hardneg_per_dataset.csv"
HARDNEG_SEQ_JSON = PROJECT_ROOT / "data" / "processed" / "hardneg_site_sequences.json"
HARDNEG_STRUCT = EMB_DIR / "hardneg_vienna_structure.npz"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "apobec" / "outputs" / "hardneg_baselines"


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
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
# Dataset
# ---------------------------------------------------------------------------

class EmbeddingDataset(Dataset):
    """Dataset for pre-computed embeddings."""

    def __init__(self, site_ids, labels, pooled_orig, pooled_edited,
                 tokens_orig=None, tokens_edited=None, structure_delta=None):
        self.site_ids = site_ids
        self.labels = labels
        self.pooled_orig = pooled_orig
        self.pooled_edited = pooled_edited
        self.tokens_orig = tokens_orig
        self.tokens_edited = tokens_edited
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
        if self.tokens_orig is not None and sid in self.tokens_orig:
            item["tokens_orig"] = self.tokens_orig[sid]
            item["tokens_edited"] = self.tokens_edited[sid]
        if self.structure_delta is not None and sid in self.structure_delta:
            item["structure_delta"] = torch.tensor(
                self.structure_delta[sid], dtype=torch.float32
            )
        else:
            item["structure_delta"] = torch.zeros(7, dtype=torch.float32)
        return item


def embedding_collate_fn(batch):
    result = {
        "site_ids": [b["site_id"] for b in batch],
        "labels": torch.stack([b["label"] for b in batch]),
        "pooled_orig": torch.stack([b["pooled_orig"] for b in batch]),
        "pooled_edited": torch.stack([b["pooled_edited"] for b in batch]),
        "structure_delta": torch.stack([b["structure_delta"] for b in batch]),
    }
    if "tokens_orig" in batch[0]:
        max_len = max(b["tokens_orig"].shape[0] for b in batch)
        d = batch[0]["tokens_orig"].shape[1]
        tok_orig = torch.zeros(len(batch), max_len, d)
        tok_edited = torch.zeros(len(batch), max_len, d)
        for i, b in enumerate(batch):
            L = b["tokens_orig"].shape[0]
            tok_orig[i, :L] = b["tokens_orig"]
            tok_edited[i, :L] = b["tokens_edited"]
        result["tokens_orig"] = tok_orig
        result["tokens_edited"] = tok_edited
    return result


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_score):
    mask = ~(np.isnan(y_true) | np.isnan(y_score))
    y_true, y_score = y_true[mask], y_score[mask]

    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return {k: float("nan") for k in [
            "auroc", "auprc", "f1", "precision", "recall", "accuracy",
            "n_positive", "n_negative"
        ]}

    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score,
        recall_score, precision_recall_curve,
    )

    metrics = {
        "auroc": roc_auc_score(y_true, y_score),
        "auprc": average_precision_score(y_true, y_score),
    }

    prec_arr, rec_arr, thresholds = precision_recall_curve(y_true, y_score)
    f1_arr = 2 * prec_arr[:-1] * rec_arr[:-1] / (prec_arr[:-1] + rec_arr[:-1] + 1e-8)
    best_idx = np.argmax(f1_arr)
    threshold = thresholds[best_idx] if len(thresholds) > 0 else 0.5
    y_pred = (y_score >= threshold).astype(int)

    metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)
    metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["n_positive"] = int(y_true.sum())
    metrics["n_negative"] = int(len(y_true) - y_true.sum())

    return metrics


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_pooled_model(d_model=640):
    return nn.Sequential(
        nn.Linear(d_model, 256), nn.GELU(), nn.Dropout(0.3),
        nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.15),
        nn.Linear(128, 1),
    )


def build_subtraction_model(d_model=640):
    return nn.Sequential(
        nn.Linear(d_model, 256), nn.GELU(), nn.Dropout(0.3),
        nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.15),
        nn.Linear(128, 1),
    )


def build_concat_model(d_model=640):
    return nn.Sequential(
        nn.Linear(d_model * 2, 512), nn.GELU(), nn.Dropout(0.3),
        nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.15),
        nn.Linear(256, 1),
    )


def build_structure_model():
    return nn.Sequential(
        nn.Linear(7, 64), nn.GELU(), nn.Dropout(0.3),
        nn.Linear(64, 32), nn.GELU(), nn.Dropout(0.15),
        nn.Linear(32, 1),
    )


class CrossAttentionModel(nn.Module):
    def __init__(self, d_model=640, n_heads=8, d_hidden=256, dropout=0.3):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden // 2), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(d_hidden // 2, 1),
        )

    def forward(self, batch):
        attended, _ = self.cross_attn(
            query=batch["tokens_orig"],
            key=batch["tokens_edited"],
            value=batch["tokens_edited"],
        )
        x = self.norm(batch["tokens_orig"] + attended)
        pooled = x.mean(dim=1)
        return self.mlp(pooled)


class DiffAttentionModel(nn.Module):
    def __init__(self, d_model=640, n_heads=8, n_layers=2, d_hidden=256, dropout=0.3):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden // 2), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(d_hidden // 2, 1),
        )

    def forward(self, batch):
        diff = batch["tokens_edited"] - batch["tokens_orig"]
        encoded = self.transformer(diff)
        pooled = encoded.mean(dim=1)
        return self.mlp(pooled)


MODEL_CONFIGS = {
    "pooled_mlp": {"needs_tokens": False, "needs_structure": False},
    "subtraction_mlp": {"needs_tokens": False, "needs_structure": False},
    "concat_mlp": {"needs_tokens": False, "needs_structure": False},
    "cross_attention": {"needs_tokens": True, "needs_structure": False},
    "diff_attention": {"needs_tokens": True, "needs_structure": False},
    "structure_only": {"needs_tokens": False, "needs_structure": True},
    "editrna": {"needs_tokens": True, "needs_structure": True},
}


def get_model_prediction(model_name, model, batch):
    """Get prediction logits from a model."""
    if model_name == "pooled_mlp":
        return model(batch["pooled_orig"]).squeeze(-1)
    elif model_name == "subtraction_mlp":
        x = batch["pooled_edited"] - batch["pooled_orig"]
        return model(x).squeeze(-1)
    elif model_name == "concat_mlp":
        x = torch.cat([batch["pooled_orig"], batch["pooled_edited"]], dim=-1)
        return model(x).squeeze(-1)
    elif model_name == "structure_only":
        return model(batch["structure_delta"]).squeeze(-1)
    elif model_name in ("cross_attention", "diff_attention"):
        return model(batch).squeeze(-1)
    elif model_name == "editrna":
        output = model(batch)
        return output["predictions"]["binary_logit"].squeeze(-1)
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_and_evaluate(model_name, model, train_loader, val_loader, test_loader,
                       epochs=50, lr=1e-3, patience=10, seed=42):
    """Train and evaluate a single model."""
    torch.manual_seed(seed)

    # EditRNA uses multi-task loss
    is_editrna = model_name == "editrna"

    if is_editrna:
        optimizer = AdamW(model.get_parameter_groups(), lr=1e-4)
    else:
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)
    criterion = FocalLoss(gamma=2.0, alpha=0.75)

    best_val_auroc = -1.0
    patience_counter = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()

            if is_editrna:
                from experiments.apobec.exp_dataset_matrix_editrna import batch_to_device
                batch = batch_to_device(batch, torch.device("cpu"))
                output = model(batch)
                loss, _ = model.compute_loss(output, batch["targets"])
            else:
                logits = get_model_prediction(model_name, model, batch)
                loss = criterion(logits, batch["labels"])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # Validate
        model.eval()
        all_y, all_s = [], []
        with torch.no_grad():
            for batch in val_loader:
                if is_editrna:
                    from experiments.apobec.exp_dataset_matrix_editrna import batch_to_device
                    batch = batch_to_device(batch, torch.device("cpu"))
                logits = get_model_prediction(model_name, model, batch)
                probs = torch.sigmoid(logits).cpu().numpy()
                if is_editrna:
                    labels = batch["targets"]["binary"].cpu().numpy()
                else:
                    labels = batch["labels"].cpu().numpy()
                all_y.append(labels)
                all_s.append(probs)

        y_true = np.concatenate(all_y)
        y_score = np.concatenate(all_s)
        if len(np.unique(y_true)) < 2:
            val_auroc = 0.0
        else:
            val_auroc = roc_auc_score(y_true, y_score)

        if val_auroc > best_val_auroc + 1e-4:
            best_val_auroc = val_auroc
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
            if is_editrna:
                from experiments.apobec.exp_dataset_matrix_editrna import batch_to_device
                batch = batch_to_device(batch, torch.device("cpu"))
            logits = get_model_prediction(model_name, model, batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            if is_editrna:
                labels = batch["targets"]["binary"].cpu().numpy()
            else:
                labels = batch["labels"].cpu().numpy()
            all_y.append(labels)
            all_s.append(probs)

    y_true = np.concatenate(all_y)
    y_score = np.concatenate(all_s)
    metrics = compute_metrics(y_true, y_score)

    return metrics


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_embeddings(load_tokens=False):
    """Load all embeddings (original + hard negative)."""
    logger.info("Loading original embeddings...")
    pooled_orig = torch.load(EMB_DIR / "rnafm_pooled.pt", weights_only=False)
    pooled_edited = torch.load(EMB_DIR / "rnafm_pooled_edited.pt", weights_only=False)
    logger.info("  %d original embeddings", len(pooled_orig))

    tokens_orig = None
    tokens_edited = None
    if load_tokens:
        tok_path = EMB_DIR / "rnafm_tokens.pt"
        if tok_path.exists():
            tokens_orig = torch.load(tok_path, weights_only=False)
            tokens_edited = torch.load(EMB_DIR / "rnafm_tokens_edited.pt", weights_only=False)
            logger.info("  %d original token embeddings", len(tokens_orig))

    # Load hard negative embeddings
    hn_pooled_path = EMB_DIR / "hardneg_rnafm_pooled.pt"
    if hn_pooled_path.exists():
        hn_pooled_orig = torch.load(hn_pooled_path, weights_only=False)
        hn_pooled_edited = torch.load(EMB_DIR / "hardneg_rnafm_pooled_edited.pt",
                                       weights_only=False)
        logger.info("  %d hard negative pooled embeddings", len(hn_pooled_orig))
        pooled_orig.update(hn_pooled_orig)
        pooled_edited.update(hn_pooled_edited)
    else:
        logger.warning("Hard negative pooled embeddings not found at %s", hn_pooled_path)

    if load_tokens:
        hn_tok_path = EMB_DIR / "hardneg_rnafm_tokens.pt"
        if hn_tok_path.exists():
            hn_tokens_orig = torch.load(hn_tok_path, weights_only=False)
            hn_tokens_edited = torch.load(EMB_DIR / "hardneg_rnafm_tokens_edited.pt",
                                           weights_only=False)
            logger.info("  %d hard negative token embeddings", len(hn_tokens_orig))
            if tokens_orig is not None:
                tokens_orig.update(hn_tokens_orig)
                tokens_edited.update(hn_tokens_edited)

    # Structure deltas
    structure_delta = {}
    if STRUCT_CACHE.exists():
        data = np.load(STRUCT_CACHE, allow_pickle=True)
        feat_key = "delta_features" if "delta_features" in data else "features"
        if "site_ids" in data and feat_key in data:
            sids = data["site_ids"]
            feats = data[feat_key]
            structure_delta = {str(sid): feats[i] for i, sid in enumerate(sids)}
    if HARDNEG_STRUCT.exists():
        data = np.load(HARDNEG_STRUCT, allow_pickle=True)
        feat_key = "delta_features" if "delta_features" in data else "features"
        if "site_ids" in data and feat_key in data:
            sids = data["site_ids"]
            feats = data[feat_key]
            for i, sid in enumerate(sids):
                structure_delta[str(sid)] = feats[i]
    logger.info("  %d total structure deltas", len(structure_delta))

    return pooled_orig, pooled_edited, tokens_orig, tokens_edited, structure_delta


def prepare_splits(splits_df, hardneg_df, pooled_orig, neg_ratio=5, seed=42):
    """Create train/val/test splits with hard negatives.

    Positive sites come from splits_expanded.csv.
    Hard negatives come from hardneg_per_dataset.csv.
    We maintain the same positive splits and randomly assign hard negatives.
    """
    rng = np.random.RandomState(seed)

    # Positives
    pos_df = splits_df[splits_df["label"] == 1].copy()
    pos_train = pos_df[pos_df["split"] == "train"]
    pos_val = pos_df[pos_df["split"] == "val"]
    pos_test = pos_df[pos_df["split"] == "test"]

    # Hard negatives
    available_hn = hardneg_df[hardneg_df["site_id"].isin(pooled_orig.keys())].copy()
    if len(available_hn) == 0:
        logger.error("No hard negatives have embeddings. Cannot proceed.")
        return None, None, None

    logger.info("Available hard negatives with embeddings: %d", len(available_hn))

    # Limit negatives
    n_neg_target = min(len(available_hn), len(pos_df) * neg_ratio)
    if len(available_hn) > n_neg_target:
        available_hn = available_hn.sample(n=n_neg_target, random_state=seed)

    # Split negatives proportionally
    n_total = len(available_hn)
    n_test = max(1, int(n_total * 0.15))
    n_val = max(1, int(n_total * 0.15))
    n_train = n_total - n_test - n_val

    idx = rng.permutation(len(available_hn))
    available_hn = available_hn.iloc[idx].copy()

    hn_test = available_hn.iloc[:n_test].copy()
    hn_val = available_hn.iloc[n_test:n_test + n_val].copy()
    hn_train = available_hn.iloc[n_test + n_val:].copy()

    hn_train["label"] = 0
    hn_val["label"] = 0
    hn_test["label"] = 0

    # Combine
    train_df = pd.concat([
        pos_train[["site_id", "label"]],
        hn_train[["site_id", "label"]]
    ]).reset_index(drop=True)
    val_df = pd.concat([
        pos_val[["site_id", "label"]],
        hn_val[["site_id", "label"]]
    ]).reset_index(drop=True)
    test_df = pd.concat([
        pos_test[["site_id", "label"]],
        hn_test[["site_id", "label"]]
    ]).reset_index(drop=True)

    logger.info("Hard neg splits: train=%d (pos=%d, neg=%d), val=%d, test=%d",
                len(train_df), len(pos_train), len(hn_train),
                len(val_df), len(test_df))

    return train_df, val_df, test_df


def prepare_easy_splits(splits_df, pooled_orig):
    """Use the original tier2/tier3 negatives (easy negatives)."""
    available = splits_df[splits_df["site_id"].isin(pooled_orig.keys())].copy()

    train_df = available[available["split"] == "train"][["site_id", "label"]].reset_index(drop=True)
    val_df = available[available["split"] == "val"][["site_id", "label"]].reset_index(drop=True)
    test_df = available[available["split"] == "test"][["site_id", "label"]].reset_index(drop=True)

    logger.info("Easy neg splits: train=%d (pos=%d, neg=%d), val=%d, test=%d",
                len(train_df), train_df["label"].sum(), len(train_df) - train_df["label"].sum(),
                len(val_df), len(test_df))

    return train_df, val_df, test_df


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

    t_total = time.time()

    # Check prerequisites
    if not HARDNEG_CSV.exists():
        logger.error("Hard negatives not found: %s", HARDNEG_CSV)
        logger.error("Run: python scripts/apobec/generate_hardneg_pipeline.py")
        return

    hardneg_df = pd.read_csv(HARDNEG_CSV)
    logger.info("Loaded %d hard negatives", len(hardneg_df))
    logger.info("Hard neg delta_MFE stats: mean=%.4f, std=%.4f, max=%.4f",
                hardneg_df["delta_mfe"].abs().mean(),
                hardneg_df["delta_mfe"].abs().std(),
                hardneg_df["delta_mfe"].abs().max())

    splits_df = pd.read_csv(SPLITS_CSV)

    # Check which models we can run (token-level models need token embeddings)
    has_tokens = (EMB_DIR / "rnafm_tokens.pt").exists()
    has_hn_pooled = (EMB_DIR / "hardneg_rnafm_pooled.pt").exists()

    if not has_hn_pooled:
        logger.error("Hard negative pooled embeddings not found.")
        logger.error("Run embedding generation for hard negatives first.")
        return

    # Determine which models to run
    pooled_models = ["pooled_mlp", "subtraction_mlp", "concat_mlp", "structure_only"]
    token_models = ["cross_attention", "diff_attention"]

    models_to_run = pooled_models.copy()
    if has_tokens and (EMB_DIR / "hardneg_rnafm_tokens.pt").exists():
        models_to_run.extend(token_models)
        # EditRNA needs special handling
        models_to_run.append("editrna")

    logger.info("Models to evaluate: %s", models_to_run)

    # Load data
    need_tokens = any(MODEL_CONFIGS[m]["needs_tokens"] for m in models_to_run)
    (pooled_orig, pooled_edited, tokens_orig, tokens_edited,
     structure_delta) = load_all_embeddings(load_tokens=need_tokens)

    # Prepare splits
    hard_train, hard_val, hard_test = prepare_splits(
        splits_df, hardneg_df, pooled_orig, neg_ratio=5
    )
    easy_train, easy_val, easy_test = prepare_easy_splits(splits_df, pooled_orig)

    if hard_train is None:
        return

    # ---------------------------------------------------------------------------
    # Train and evaluate all models with both negative types
    # ---------------------------------------------------------------------------

    all_results = {}

    for neg_type, (train_df, val_df, test_df) in [
        ("easy", (easy_train, easy_val, easy_test)),
        ("hard", (hard_train, hard_val, hard_test)),
    ]:
        logger.info("\n" + "=" * 70)
        logger.info("EVALUATING WITH %s NEGATIVES", neg_type.upper())
        logger.info("=" * 70)

        for model_name in models_to_run:
            logger.info("\n--- %s (%s negatives) ---", model_name, neg_type)
            t0 = time.time()

            needs_tokens = MODEL_CONFIGS[model_name]["needs_tokens"]
            if needs_tokens and tokens_orig is None:
                logger.warning("  Skipping %s (no token embeddings)", model_name)
                continue

            # Build model
            if model_name == "pooled_mlp":
                model = build_pooled_model()
            elif model_name == "subtraction_mlp":
                model = build_subtraction_model()
            elif model_name == "concat_mlp":
                model = build_concat_model()
            elif model_name == "structure_only":
                model = build_structure_model()
            elif model_name == "cross_attention":
                model = CrossAttentionModel()
            elif model_name == "diff_attention":
                model = DiffAttentionModel()
            elif model_name == "editrna":
                from data.apobec_dataset import (
                    APOBECDataConfig, APOBECDataset, APOBECSiteSample,
                    N_TISSUES, apobec_collate_fn, get_flanking_context,
                )
                from models.editrna_a3a import EditRNA_A3A, EditRNAConfig
                from models.encoders import CachedRNAEncoder

                cached_encoder = CachedRNAEncoder(
                    tokens_cache=tokens_orig,
                    pooled_cache=pooled_orig,
                    tokens_edited_cache=tokens_edited,
                    pooled_edited_cache=pooled_edited,
                    d_model=640,
                )
                config = EditRNAConfig(
                    primary_encoder="cached", d_model=640, d_edit=256, d_fused=512,
                    edit_n_heads=8, use_structure_delta=True,
                    head_dropout=0.2, fusion_dropout=0.2,
                    focal_gamma=2.0, focal_alpha_binary=0.75,
                )
                model = EditRNA_A3A(config=config, primary_encoder=cached_encoder)

            # Create dataloaders
            if model_name == "editrna":
                # EditRNA uses APOBECDataset with apobec_collate_fn
                sequences = {}
                if SEQ_JSON.exists():
                    sequences.update(json.loads(SEQ_JSON.read_text()))
                if HARDNEG_SEQ_JSON.exists():
                    sequences.update(json.loads(HARDNEG_SEQ_JSON.read_text()))

                from experiments.apobec.exp_dataset_matrix_editrna import build_samples

                data_config = APOBECDataConfig(window_size=100)

                def make_editrna_df(df):
                    """Add columns needed by build_samples."""
                    df = df.copy()
                    if "editing_rate" not in df.columns:
                        df["editing_rate"] = np.nan
                    if "chr" not in df.columns:
                        df["chr"] = ""
                    if "start" not in df.columns:
                        df["start"] = 0
                    if "gene" not in df.columns:
                        df["gene"] = ""
                    return df

                train_samples = build_samples(
                    make_editrna_df(train_df), sequences, structure_delta
                )
                val_samples = build_samples(
                    make_editrna_df(val_df), sequences, structure_delta
                )
                test_samples = build_samples(
                    make_editrna_df(test_df), sequences, structure_delta
                )

                train_loader = DataLoader(
                    APOBECDataset(train_samples, data_config),
                    batch_size=32, shuffle=True,
                    collate_fn=apobec_collate_fn, num_workers=0,
                )
                val_loader = DataLoader(
                    APOBECDataset(val_samples, data_config),
                    batch_size=64, shuffle=False,
                    collate_fn=apobec_collate_fn, num_workers=0,
                )
                test_loader = DataLoader(
                    APOBECDataset(test_samples, data_config),
                    batch_size=64, shuffle=False,
                    collate_fn=apobec_collate_fn, num_workers=0,
                )
            else:
                def make_loader(df, shuffle=False, batch_size=64):
                    ds = EmbeddingDataset(
                        df["site_id"].tolist(),
                        df["label"].values.astype(np.float32),
                        pooled_orig, pooled_edited,
                        tokens_orig if needs_tokens else None,
                        tokens_edited if needs_tokens else None,
                        structure_delta,
                    )
                    return DataLoader(
                        ds, batch_size=batch_size, shuffle=shuffle,
                        collate_fn=embedding_collate_fn, num_workers=0,
                    )

                train_loader = make_loader(train_df, shuffle=True)
                val_loader = make_loader(val_df)
                test_loader = make_loader(test_df)

            # Train and evaluate
            metrics = train_and_evaluate(
                model_name, model, train_loader, val_loader, test_loader,
                epochs=50, lr=1e-3, patience=10,
            )

            elapsed = time.time() - t0
            logger.info("  %s (%s): AUROC=%.4f, AUPRC=%.4f, F1=%.4f (%.1fs)",
                        model_name, neg_type, metrics["auroc"], metrics["auprc"],
                        metrics["f1"], elapsed)

            key = f"{model_name}_{neg_type}"
            all_results[key] = metrics

    # ---------------------------------------------------------------------------
    # Comparison visualization
    # ---------------------------------------------------------------------------

    logger.info("\n" + "=" * 70)
    logger.info("COMPARISON: EASY vs HARD NEGATIVES")
    logger.info("=" * 70)

    models_compared = []
    for model_name in models_to_run:
        easy_key = f"{model_name}_easy"
        hard_key = f"{model_name}_hard"
        if easy_key in all_results and hard_key in all_results:
            easy_auroc = all_results[easy_key]["auroc"]
            hard_auroc = all_results[hard_key]["auroc"]
            drop = easy_auroc - hard_auroc if np.isfinite(easy_auroc) and np.isfinite(hard_auroc) else float("nan")
            logger.info("  %-20s  Easy AUROC=%.4f  Hard AUROC=%.4f  Drop=%.4f",
                        model_name, easy_auroc, hard_auroc, drop)
            models_compared.append(model_name)

    # Bar chart comparison
    if models_compared:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # AUROC comparison
        x = np.arange(len(models_compared))
        width = 0.35
        easy_aurocs = [all_results[f"{m}_easy"]["auroc"] for m in models_compared]
        hard_aurocs = [all_results[f"{m}_hard"]["auroc"] for m in models_compared]

        axes[0].bar(x - width / 2, easy_aurocs, width, label="Easy negatives",
                    color="#81c784", alpha=0.8)
        axes[0].bar(x + width / 2, hard_aurocs, width, label="Hard negatives",
                    color="#ef5350", alpha=0.8)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(models_compared, rotation=45, ha="right", fontsize=8)
        axes[0].set_ylabel("AUROC")
        axes[0].set_title("Classification with Easy vs Hard Negatives")
        axes[0].legend()
        axes[0].set_ylim(0, 1.05)
        axes[0].grid(True, alpha=0.3, axis="y")

        # AUROC annotation
        for i, (e, h) in enumerate(zip(easy_aurocs, hard_aurocs)):
            if np.isfinite(e):
                axes[0].text(i - width / 2, e + 0.01, f"{e:.3f}", ha="center", fontsize=7)
            if np.isfinite(h):
                axes[0].text(i + width / 2, h + 0.01, f"{h:.3f}", ha="center", fontsize=7)

        # AUROC drop
        drops = [e - h if np.isfinite(e) and np.isfinite(h) else 0
                 for e, h in zip(easy_aurocs, hard_aurocs)]
        colors = ["#d32f2f" if d > 0.1 else "#ff9800" if d > 0.05 else "#4caf50"
                  for d in drops]
        axes[1].bar(x, drops, color=colors, alpha=0.8)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(models_compared, rotation=45, ha="right", fontsize=8)
        axes[1].set_ylabel("AUROC Drop (Easy - Hard)")
        axes[1].set_title("AUROC Drop with Structure-Matched Negatives")
        axes[1].axhline(0, color="black", linewidth=0.5)
        axes[1].grid(True, alpha=0.3, axis="y")

        for i, d in enumerate(drops):
            axes[1].text(i, d + 0.005, f"{d:.3f}", ha="center", fontsize=8)

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "easy_vs_hard_comparison.png", dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Saved comparison plot to easy_vs_hard_comparison.png")

    # Save results
    with open(OUTPUT_DIR / "hardneg_baselines_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    elapsed = time.time() - t_total
    logger.info("\nTotal time: %.1fs", elapsed)
    logger.info("Results saved to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
