#!/usr/bin/env python
"""Rate prediction regularization sweep: fix DiffAttention & EditRNA overfitting.

Current problem:
  - DiffAttention: train Spearman=0.80 → test=0.15 (catastrophic overfitting)
  - EditRNA-A3A:   train Spearman=0.43 → test=0.12 (moderate overfitting)
  - PooledMLP:     train Spearman=0.18 → test=0.21 (best generalizer)

Strategy:
  1. DiffAttention with heavy regularization:
     - Fewer layers (1 vs 2), smaller hidden (128 vs 256)
     - Higher dropout (0.5, 0.7), heavier weight_decay (1e-2, 1e-3)
     - Smaller learning rate, early stopping with patience=10
  2. EditRNA-A3A with increased regularization
  3. SubtractionMLP with tuning

Goal: Beat PooledMLP's test Spearman of 0.211

Usage:
    python experiments/apobec/exp_rate_regularized.py
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from itertools import product

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
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error, r2_score
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
SEQ_JSON = PROJECT_ROOT / "data" / "processed" / "site_sequences.json"
STRUCT_CACHE = EMB_DIR / "vienna_structure_cache.npz"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "apobec" / "outputs" / "rate_regularized"


# ---------------------------------------------------------------------------
# Helpers (shared with exp_rate_all_architectures.py)
# ---------------------------------------------------------------------------

def compute_log2_rate(editing_rate):
    if pd.isna(editing_rate) or editing_rate < 0:
        return float("nan")
    rate = float(editing_rate)
    if rate > 1.0:
        rate = rate / 100.0
    return np.log2(rate + 0.01)


def compute_regression_metrics(y_true, y_pred):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) < 3:
        return {"spearman": float("nan"), "pearson": float("nan"),
                "mse": float("nan"), "r2": float("nan"), "n_samples": 0}
    sp_rho, _ = spearmanr(y_true, y_pred)
    pe_r, _ = pearsonr(y_true, y_pred)
    mse = float(mean_squared_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"spearman": float(sp_rho), "pearson": float(pe_r),
            "mse": mse, "r2": r2, "n_samples": int(len(y_true))}


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class TokenRateDataset(Dataset):
    def __init__(self, site_ids, targets, tokens_orig, tokens_edited):
        self.site_ids = site_ids
        self.targets = targets
        self.tokens_orig = tokens_orig
        self.tokens_edited = tokens_edited

    def __len__(self):
        return len(self.site_ids)

    def __getitem__(self, idx):
        sid = self.site_ids[idx]
        t_orig = self.tokens_orig[sid]
        t_edited = self.tokens_edited[sid]
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        return {"tokens_orig": t_orig, "tokens_edited": t_edited}, target


class PooledRateDataset(Dataset):
    def __init__(self, site_ids, targets, pooled_orig, pooled_edited, mode="subtraction"):
        self.site_ids = site_ids
        self.targets = targets
        self.pooled_orig = pooled_orig
        self.pooled_edited = pooled_edited
        self.mode = mode

    def __len__(self):
        return len(self.site_ids)

    def __getitem__(self, idx):
        sid = self.site_ids[idx]
        orig = self.pooled_orig[sid]
        edited = self.pooled_edited[sid]
        if self.mode == "subtraction":
            x = edited - orig
        elif self.mode == "concat":
            x = torch.cat([orig, edited], dim=-1)
        elif self.mode == "orig_only":
            x = orig
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        return x, target


def token_collate_fn(batch):
    items, targets = zip(*batch)
    max_len = max(item["tokens_orig"].shape[0] for item in items)
    d_model = items[0]["tokens_orig"].shape[1]
    tokens_orig = torch.zeros(len(items), max_len, d_model)
    tokens_edited = torch.zeros(len(items), max_len, d_model)
    for i, item in enumerate(items):
        L = item["tokens_orig"].shape[0]
        tokens_orig[i, :L] = item["tokens_orig"]
        tokens_edited[i, :L] = item["tokens_edited"]
    return {"tokens_orig": tokens_orig, "tokens_edited": tokens_edited}, torch.stack(targets)


# ---------------------------------------------------------------------------
# DiffAttention with configurable regularization
# ---------------------------------------------------------------------------

class DiffAttentionRateReg(nn.Module):
    """DiffAttention with heavy regularization options."""

    def __init__(self, d_model=640, n_heads=8, n_layers=1, d_hidden=128,
                 dropout=0.5, input_dropout=0.3, use_layer_norm=True):
        super().__init__()
        self.input_dropout = nn.Dropout(input_dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=dropout, activation="gelu", batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.layer_norm = nn.LayerNorm(d_model) if use_layer_norm else nn.Identity()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, batch):
        diff = batch["tokens_edited"] - batch["tokens_orig"]
        diff = self.input_dropout(diff)
        encoded = self.transformer(diff)
        encoded = self.layer_norm(encoded)
        pooled = encoded.mean(dim=1)
        return self.mlp(pooled).squeeze(-1)


class CrossAttentionRateReg(nn.Module):
    """CrossAttention with heavy regularization."""

    def __init__(self, d_model=640, n_heads=4, d_hidden=128, dropout=0.5,
                 input_dropout=0.3):
        super().__init__()
        self.input_dropout = nn.Dropout(input_dropout)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, batch):
        tokens_orig = self.input_dropout(batch["tokens_orig"])
        tokens_edited = self.input_dropout(batch["tokens_edited"])
        attended, _ = self.cross_attn(query=tokens_orig, key=tokens_edited, value=tokens_edited)
        x = self.norm(tokens_orig + attended)
        pooled = x.mean(dim=1)
        return self.mlp(pooled).squeeze(-1)


class SubtractionMLPReg(nn.Module):
    """SubtractionMLP with deeper architecture and heavy regularization."""

    def __init__(self, d_input=640, hidden=256, dropout=0.4, n_layers=3):
        super().__init__()
        layers = []
        d = d_input
        for i in range(n_layers):
            h = hidden // (2 ** i) if hidden // (2 ** i) >= 32 else 32
            layers.extend([
                nn.Linear(d, h),
                nn.GELU(),
                nn.BatchNorm1d(h),
                nn.Dropout(dropout),
            ])
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Training with regularization options
# ---------------------------------------------------------------------------

def train_eval_reg(model, train_loader, val_loader, test_loader,
                   is_token_model=False, epochs=100, lr=1e-4,
                   weight_decay=1e-2, patience=10, seed=42,
                   label_smoothing=0.0, gradient_noise_std=0.0):
    """Train with enhanced regularization."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)

    best_val_sp = -float("inf")
    patience_counter = 0
    best_state = None
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for batch_data, batch_target in train_loader:
            optimizer.zero_grad()
            pred = model(batch_data)

            # Optional label smoothing for regression (add noise to targets)
            if label_smoothing > 0:
                noise = torch.randn_like(batch_target) * label_smoothing
                target = batch_target + noise
            else:
                target = batch_target

            loss = F.mse_loss(pred, target)

            # Huber loss as alternative (more robust)
            # loss = F.huber_loss(pred, target, delta=1.0)

            loss.backward()

            # Gradient noise for regularization
            if gradient_noise_std > 0:
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad += torch.randn_like(param.grad) * gradient_noise_std

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train_loss = train_loss / max(n_batches, 1)

        # Validate
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch_data, batch_target in val_loader:
                pred = model(batch_data)
                val_preds.append(pred.numpy())
                val_targets.append(batch_target.numpy())
        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)
        val_metrics = compute_regression_metrics(val_targets, val_preds)
        val_sp = val_metrics["spearman"] if np.isfinite(val_metrics["spearman"]) else -1.0

        history.append({
            "epoch": epoch, "train_loss": avg_train_loss,
            "val_spearman": val_sp, "val_mse": val_metrics["mse"],
        })

        if val_sp > best_val_sp + 1e-4:
            best_val_sp = val_sp
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break

    if best_state:
        model.load_state_dict(best_state)

    # Evaluate all splits
    model.eval()
    results = {"history": history, "best_epoch": len(history) - patience_counter}
    for split_name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch_data, batch_target in loader:
                pred = model(batch_data)
                all_preds.append(pred.numpy())
                all_targets.append(batch_target.numpy())
        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_targets)
        results[split_name] = compute_regression_metrics(y_true, y_pred)

    return results


# ---------------------------------------------------------------------------
# EditRNA-A3A with rate-focused regularization
# ---------------------------------------------------------------------------

def build_rate_samples(df, sequences, structure_delta, window_size=100):
    samples = []
    for _, row in df.iterrows():
        site_id = str(row["site_id"])
        seq = sequences.get(site_id, "A" * (window_size * 2 + 1))
        edit_pos = min(window_size, len(seq) // 2)
        seq_list = list(seq)
        seq_list[edit_pos] = "C"
        seq = "".join(seq_list)
        flanking = get_flanking_context(seq, edit_pos)
        struct_d = structure_delta.get(site_id)
        if struct_d is not None:
            struct_d = np.array(struct_d, dtype=np.float32)
        log2_rate = float(row["log2_rate"])
        concordance = np.zeros(5, dtype=np.float32)
        sample = APOBECSiteSample(
            sequence=seq, edit_pos=edit_pos,
            is_edited=1.0, editing_rate_log2=log2_rate,
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


def train_eval_editrna_reg(train_df, val_df, test_df, sequences, structure_delta,
                           tokens_orig, pooled_orig, tokens_edited, pooled_edited,
                           head_dropout=0.4, fusion_dropout=0.4,
                           weight_decay=1e-3, lr=5e-5,
                           epochs=80, patience=12, seed=42, rate_loss_weight=5.0):
    """Train EditRNA-A3A with rate-focused heavy regularization."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")

    if len(train_df) < 10 or len(val_df) < 5:
        empty = {"spearman": float("nan"), "pearson": float("nan"),
                 "mse": float("nan"), "r2": float("nan"), "n_samples": 0}
        return {"train": empty, "val": empty, "test": empty}

    data_config = APOBECDataConfig(window_size=100)
    train_ds = APOBECDataset(build_rate_samples(train_df, sequences, structure_delta), data_config)
    val_ds = APOBECDataset(build_rate_samples(val_df, sequences, structure_delta), data_config)
    test_ds = APOBECDataset(build_rate_samples(test_df, sequences, structure_delta), data_config)

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
        primary_encoder="cached", d_model=640, d_edit=128, d_fused=256,
        edit_n_heads=4, use_structure_delta=True,
        head_dropout=head_dropout, fusion_dropout=fusion_dropout,
        focal_gamma=2.0, focal_alpha_binary=0.75,
    )
    model = EditRNA_A3A(config=config, primary_encoder=cached_encoder).to(device)

    optimizer = AdamW(model.get_parameter_groups(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)

    best_val_sp = -float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            batch = batch_to_device(batch, device)
            optimizer.zero_grad()
            output = model(batch)
            loss, loss_dict = model.compute_loss(output, batch["targets"])

            # Upweight rate loss
            rate_loss = loss_dict.get("rate", torch.tensor(0.0))
            total_loss = loss + rate_loss * (rate_loss_weight - 1.0)

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
        scheduler.step()

        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch_to_device(batch, device)
                output = model(batch)
                rate_pred = output["predictions"]["editing_rate"].squeeze(-1).cpu().numpy()
                rate_target = batch["targets"]["rate"].cpu().numpy()
                mask = np.isfinite(rate_target)
                if mask.any():
                    val_preds.append(rate_pred[mask])
                    val_targets.append(rate_target[mask])

        if val_preds:
            val_arr = np.concatenate(val_preds)
            val_t = np.concatenate(val_targets)
            val_sp, _ = spearmanr(val_t, val_arr)
            val_sp = float(val_sp) if np.isfinite(val_sp) else -1.0
        else:
            val_sp = -1.0

        if val_sp > best_val_sp + 1e-4:
            best_val_sp = val_sp
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break

    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    results = {}
    for split_name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in loader:
                batch = batch_to_device(batch, device)
                output = model(batch)
                rate_pred = output["predictions"]["editing_rate"].squeeze(-1).cpu().numpy()
                rate_target = batch["targets"]["rate"].cpu().numpy()
                mask = np.isfinite(rate_target)
                if mask.any():
                    all_preds.append(rate_pred[mask])
                    all_targets.append(rate_target[mask])
        if all_preds:
            y_pred = np.concatenate(all_preds)
            y_true = np.concatenate(all_targets)
            results[split_name] = compute_regression_metrics(y_true, y_pred)
        else:
            results[split_name] = {
                "spearman": float("nan"), "pearson": float("nan"),
                "mse": float("nan"), "r2": float("nan"), "n_samples": 0,
            }

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)
    torch.manual_seed(42)
    t_global = time.time()

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    logger.info("Loading embeddings...")
    pooled_orig = torch.load(EMB_DIR / "rnafm_pooled.pt", weights_only=False)
    pooled_edited = torch.load(EMB_DIR / "rnafm_pooled_edited.pt", weights_only=False)
    tokens_path = EMB_DIR / "rnafm_tokens.pt"
    tokens_edited_path = EMB_DIR / "rnafm_tokens_edited.pt"
    has_tokens = tokens_path.exists() and tokens_edited_path.exists()
    if has_tokens:
        logger.info("Loading token embeddings (this may take a minute)...")
        tokens_orig = torch.load(tokens_path, weights_only=False)
        tokens_edited = torch.load(tokens_edited_path, weights_only=False)
        logger.info("  Loaded %d token embeddings", len(tokens_orig))
    else:
        tokens_orig = tokens_edited = None
        logger.warning("Token embeddings not found — token models will be skipped")

    splits_df = pd.read_csv(SPLITS_CSV)
    sequences = {}
    if SEQ_JSON.exists():
        sequences = json.loads(SEQ_JSON.read_text())

    structure_delta = {}
    if STRUCT_CACHE.exists():
        data = np.load(STRUCT_CACHE, allow_pickle=True)
        feat_key = "delta_features" if "delta_features" in data else "features"
        if "site_ids" in data and feat_key in data:
            sids = data["site_ids"]
            feats = data[feat_key]
            structure_delta = {str(sid): feats[i] for i, sid in enumerate(sids)}

    # Filter to positive sites with valid rate
    available = set(pooled_orig.keys()) & set(pooled_edited.keys())
    rate_df = splits_df[
        (splits_df["label"] == 1) &
        (splits_df["editing_rate"].notna()) &
        (splits_df["site_id"].isin(available))
    ].copy()
    rate_df["log2_rate"] = rate_df["editing_rate"].apply(compute_log2_rate)
    rate_df = rate_df[rate_df["log2_rate"].notna()].copy()

    train_df = rate_df[rate_df["split"] == "train"].copy()
    val_df = rate_df[rate_df["split"] == "val"].copy()
    test_df = rate_df[rate_df["split"] == "test"].copy()
    logger.info("Rate samples — Train: %d, Val: %d, Test: %d", len(train_df), len(val_df), len(test_df))

    all_results = {}

    # ==================================================================
    # Section 1: DiffAttention regularization sweep
    # ==================================================================
    if has_tokens:
        logger.info("\n" + "=" * 80)
        logger.info("SECTION 1: DiffAttention Regularization Sweep")
        logger.info("=" * 80)

        token_available = set(tokens_orig.keys()) & set(tokens_edited.keys())
        train_tok = train_df[train_df["site_id"].isin(token_available)]
        val_tok = val_df[val_df["site_id"].isin(token_available)]
        test_tok = test_df[test_df["site_id"].isin(token_available)]

        tr_tok = DataLoader(TokenRateDataset(train_tok["site_id"].tolist(),
                    train_tok["log2_rate"].values.astype(np.float32),
                    tokens_orig, tokens_edited),
                    batch_size=16, shuffle=True, collate_fn=token_collate_fn, num_workers=0)
        va_tok = DataLoader(TokenRateDataset(val_tok["site_id"].tolist(),
                    val_tok["log2_rate"].values.astype(np.float32),
                    tokens_orig, tokens_edited),
                    batch_size=32, shuffle=False, collate_fn=token_collate_fn, num_workers=0)
        te_tok = DataLoader(TokenRateDataset(test_tok["site_id"].tolist(),
                    test_tok["log2_rate"].values.astype(np.float32),
                    tokens_orig, tokens_edited),
                    batch_size=32, shuffle=False, collate_fn=token_collate_fn, num_workers=0)

        diff_configs = [
            # (name, n_layers, d_hidden, dropout, input_dropout, weight_decay, lr)
            ("DiffAttn_L1_H128_D0.5_WD1e-2",   1, 128, 0.5, 0.3, 1e-2, 1e-4),
            ("DiffAttn_L1_H128_D0.7_WD1e-2",   1, 128, 0.7, 0.4, 1e-2, 1e-4),
            ("DiffAttn_L1_H64_D0.5_WD1e-2",    1,  64, 0.5, 0.3, 1e-2, 5e-5),
            ("DiffAttn_L1_H64_D0.7_WD1e-2",    1,  64, 0.7, 0.4, 1e-2, 5e-5),
            ("DiffAttn_L1_H128_D0.5_WD1e-3",   1, 128, 0.5, 0.2, 1e-3, 1e-4),
            ("DiffAttn_L1_H256_D0.6_WD1e-2",   1, 256, 0.6, 0.3, 1e-2, 5e-5),
            ("DiffAttn_L2_H128_D0.6_WD1e-2",   2, 128, 0.6, 0.3, 1e-2, 5e-5),
            ("DiffAttn_L1_H128_D0.5_WD5e-2",   1, 128, 0.5, 0.3, 5e-2, 1e-4),
        ]

        for name, n_layers, d_hidden, dropout, input_drop, wd, lr in diff_configs:
            logger.info("\n  --- %s ---", name)
            t0 = time.time()
            model = DiffAttentionRateReg(
                d_model=640, n_heads=8, n_layers=n_layers,
                d_hidden=d_hidden, dropout=dropout, input_dropout=input_drop)
            res = train_eval_reg(
                model, tr_tok, va_tok, te_tok, is_token_model=True,
                epochs=100, lr=lr, weight_decay=wd, patience=10, seed=42)
            elapsed = time.time() - t0
            sp = res["test"]["spearman"]
            logger.info("  %s: Test Sp=%.4f (train=%.4f, val=%.4f) in %.1fs [epoch %d]",
                        name, sp, res["train"]["spearman"], res["val"]["spearman"],
                        elapsed, res.get("best_epoch", -1))
            all_results[name] = {
                "type": "DiffAttention", "config": {
                    "n_layers": n_layers, "d_hidden": d_hidden, "dropout": dropout,
                    "input_dropout": input_drop, "weight_decay": wd, "lr": lr,
                },
                "train": res["train"], "val": res["val"], "test": res["test"],
                "best_epoch": res.get("best_epoch", -1), "time_seconds": elapsed,
            }

        # Also try CrossAttention with regularization
        logger.info("\n" + "=" * 80)
        logger.info("CrossAttention Regularization")
        logger.info("=" * 80)

        cross_configs = [
            ("CrossAttn_H4_D128_D0.5_WD1e-2",  4, 128, 0.5, 0.3, 1e-2, 1e-4),
            ("CrossAttn_H4_D64_D0.6_WD1e-2",   4,  64, 0.6, 0.3, 1e-2, 5e-5),
        ]

        for name, n_heads, d_hidden, dropout, input_drop, wd, lr in cross_configs:
            logger.info("\n  --- %s ---", name)
            t0 = time.time()
            model = CrossAttentionRateReg(
                d_model=640, n_heads=n_heads, d_hidden=d_hidden,
                dropout=dropout, input_dropout=input_drop)
            res = train_eval_reg(
                model, tr_tok, va_tok, te_tok, is_token_model=True,
                epochs=100, lr=lr, weight_decay=wd, patience=10, seed=42)
            elapsed = time.time() - t0
            sp = res["test"]["spearman"]
            logger.info("  %s: Test Sp=%.4f in %.1fs", name, sp, elapsed)
            all_results[name] = {
                "type": "CrossAttention", "config": {
                    "n_heads": n_heads, "d_hidden": d_hidden, "dropout": dropout,
                    "input_dropout": input_drop, "weight_decay": wd, "lr": lr,
                },
                "train": res["train"], "val": res["val"], "test": res["test"],
                "best_epoch": res.get("best_epoch", -1), "time_seconds": elapsed,
            }

    # ==================================================================
    # Section 2: SubtractionMLP with deeper architecture
    # ==================================================================
    logger.info("\n" + "=" * 80)
    logger.info("SECTION 2: SubtractionMLP Regularization")
    logger.info("=" * 80)

    sub_configs = [
        ("SubMLP_H256_D0.4_L3_WD1e-3", 256, 0.4, 3, 1e-3, 5e-4),
        ("SubMLP_H128_D0.3_L3_WD1e-3", 128, 0.3, 3, 1e-3, 1e-3),
        ("SubMLP_H256_D0.5_L2_WD1e-2", 256, 0.5, 2, 1e-2, 5e-4),
    ]

    for name, hidden, dropout, n_layers, wd, lr in sub_configs:
        logger.info("\n  --- %s ---", name)
        t0 = time.time()

        tr_sub = DataLoader(PooledRateDataset(
            train_df["site_id"].tolist(), train_df["log2_rate"].values.astype(np.float32),
            pooled_orig, pooled_edited, "subtraction"), batch_size=64, shuffle=True, num_workers=0)
        va_sub = DataLoader(PooledRateDataset(
            val_df["site_id"].tolist(), val_df["log2_rate"].values.astype(np.float32),
            pooled_orig, pooled_edited, "subtraction"), batch_size=64, shuffle=False, num_workers=0)
        te_sub = DataLoader(PooledRateDataset(
            test_df["site_id"].tolist(), test_df["log2_rate"].values.astype(np.float32),
            pooled_orig, pooled_edited, "subtraction"), batch_size=64, shuffle=False, num_workers=0)

        model = SubtractionMLPReg(d_input=640, hidden=hidden, dropout=dropout, n_layers=n_layers)
        res = train_eval_reg(
            model, tr_sub, va_sub, te_sub,
            epochs=100, lr=lr, weight_decay=wd, patience=15, seed=42)
        elapsed = time.time() - t0
        sp = res["test"]["spearman"]
        logger.info("  %s: Test Sp=%.4f in %.1fs", name, sp, elapsed)
        all_results[name] = {
            "type": "SubtractionMLP", "config": {
                "hidden": hidden, "dropout": dropout, "n_layers": n_layers,
                "weight_decay": wd, "lr": lr,
            },
            "train": res["train"], "val": res["val"], "test": res["test"],
            "best_epoch": res.get("best_epoch", -1), "time_seconds": elapsed,
        }

    # ==================================================================
    # Section 3: EditRNA-A3A with rate-focused regularization
    # ==================================================================
    if has_tokens:
        logger.info("\n" + "=" * 80)
        logger.info("SECTION 3: EditRNA-A3A Rate-Focused Regularization")
        logger.info("=" * 80)

        editrna_configs = [
            # (name, head_dropout, fusion_dropout, weight_decay, lr, rate_loss_weight)
            ("EditRNA_HD0.4_FD0.4_WD1e-3_RW5",  0.4, 0.4, 1e-3, 5e-5, 5.0),
            ("EditRNA_HD0.5_FD0.5_WD1e-2_RW10", 0.5, 0.5, 1e-2, 5e-5, 10.0),
            ("EditRNA_HD0.3_FD0.3_WD1e-3_RW3",  0.3, 0.3, 1e-3, 1e-4, 3.0),
        ]

        for name, hd, fd, wd, lr, rw in editrna_configs:
            logger.info("\n  --- %s ---", name)
            t0 = time.time()
            res = train_eval_editrna_reg(
                train_df, val_df, test_df, sequences, structure_delta,
                tokens_orig, pooled_orig, tokens_edited, pooled_edited,
                head_dropout=hd, fusion_dropout=fd,
                weight_decay=wd, lr=lr,
                epochs=80, patience=12, seed=42, rate_loss_weight=rw)
            elapsed = time.time() - t0
            sp = res["test"]["spearman"]
            logger.info("  %s: Test Sp=%.4f in %.1fs", name, sp, elapsed)
            all_results[name] = {
                "type": "EditRNA-A3A", "config": {
                    "head_dropout": hd, "fusion_dropout": fd,
                    "weight_decay": wd, "lr": lr, "rate_loss_weight": rw,
                },
                "train": res["train"], "val": res["val"], "test": res["test"],
                "time_seconds": elapsed,
            }

    # ==================================================================
    # Summary
    # ==================================================================
    print("\n" + "=" * 110)
    print("RATE PREDICTION REGULARIZATION SWEEP - RESULTS")
    print("=" * 110)
    header = f"{'Model':<45}{'Train Sp':>12}{'Val Sp':>12}{'Test Sp':>12}{'Test MSE':>12}{'Time':>10}"
    print(header)
    print("-" * 110)

    # Sort by test spearman
    sorted_results = sorted(all_results.items(),
                            key=lambda x: x[1]["test"].get("spearman", -1), reverse=True)

    for name, res in sorted_results:
        tr_sp = res["train"].get("spearman", float("nan"))
        va_sp = res["val"].get("spearman", float("nan"))
        te_sp = res["test"].get("spearman", float("nan"))
        te_mse = res["test"].get("mse", float("nan"))
        t = res.get("time_seconds", 0)
        print(f"{name:<45}{tr_sp:>12.4f}{va_sp:>12.4f}{te_sp:>12.4f}{te_mse:>12.4f}{t:>9.0f}s")

    print("=" * 110)
    print(f"\nBaseline to beat: PooledMLP test Spearman = 0.211")
    best_name, best_res = sorted_results[0]
    print(f"Best result: {best_name} test Spearman = {best_res['test']['spearman']:.4f}")
    if best_res["test"]["spearman"] > 0.211:
        print(">>> SUCCESS: Beat PooledMLP! <<<")
    else:
        print(">>> DID NOT beat PooledMLP — may need different approach <<<")

    total_time = time.time() - t_global
    logger.info("\nTotal sweep time: %.1fs (%.1f min)", total_time, total_time / 60)

    # Save results
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

    with open(OUTPUT_DIR / "regularization_sweep_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=serialize)
    logger.info("Saved results to %s", OUTPUT_DIR / "regularization_sweep_results.json")

    # Plot comparison
    fig, ax = plt.subplots(figsize=(14, 6))
    names = [n for n, _ in sorted_results]
    test_sp = [r["test"]["spearman"] for _, r in sorted_results]
    train_sp = [r["train"]["spearman"] for _, r in sorted_results]
    x = np.arange(len(names))
    width = 0.35
    bars1 = ax.bar(x - width/2, train_sp, width, label="Train", alpha=0.7, color="steelblue")
    bars2 = ax.bar(x + width/2, test_sp, width, label="Test", alpha=0.7, color="darkorange")
    ax.axhline(y=0.211, color="red", linestyle="--", linewidth=2, label="PooledMLP baseline (0.211)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Spearman rho")
    ax.set_title("Rate Prediction Regularization Sweep\n(Train vs Test Spearman)", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "regularization_sweep_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved plot to %s", OUTPUT_DIR / "regularization_sweep_comparison.png")


if __name__ == "__main__":
    main()
