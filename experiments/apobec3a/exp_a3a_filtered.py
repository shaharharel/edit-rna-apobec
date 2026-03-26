#!/usr/bin/env python
"""Re-run key experiments on APOBEC3A-only filtered dataset.

Filtering:
  - Levanon/Advisor: Only "APOBEC3A Only" annotated sites (120 of 636)
  - Asaoka, Alqassim, Sharma: All kept (A3A by study design)
  - Negatives: All kept (TC-motif matched)

Experiments:
  1. Binary classification: All 7 architectures (pooled-based only to save RAM)
  2. Rate prediction: All pooled architectures + DiffAttention with regularization
  3. TC motif statistics validation

Uses splits_expanded_a3a.csv instead of splits_expanded.csv.
All embeddings are pre-computed and shared - just filtering by site_id.

Usage:
    python experiments/apobec3a/exp_a3a_filtered.py
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
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, accuracy_score, mean_squared_error, r2_score,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.apobec_dataset import (
    APOBECDataConfig, APOBECDataset, APOBECSiteSample,
    N_TISSUES, apobec_collate_fn, get_flanking_context,
)
from models.editrna_a3a import EditRNA_A3A, EditRNAConfig
from models.encoders import CachedRNAEncoder

logger = logging.getLogger(__name__)

EMB_DIR = PROJECT_ROOT / "data" / "processed" / "embeddings"
SPLITS_CSV = PROJECT_ROOT / "data" / "processed" / "splits_expanded_a3a.csv"
SEQ_JSON = PROJECT_ROOT / "data" / "processed" / "site_sequences.json"
STRUCT_CACHE = EMB_DIR / "vienna_structure_cache.npz"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "apobec3a" / "outputs" / "a3a_filtered"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_binary_metrics(y_true, y_score, threshold=None):
    auroc = float(roc_auc_score(y_true, y_score))
    auprc = float(average_precision_score(y_true, y_score))
    if threshold is None:
        from sklearn.metrics import precision_recall_curve
        prec, rec, thresholds = precision_recall_curve(y_true, y_score)
        f1_scores = 2 * prec * rec / (prec + rec + 1e-10)
        threshold = float(thresholds[np.argmax(f1_scores)])
    y_pred = (np.array(y_score) >= threshold).astype(int)
    return {
        "auroc": auroc, "auprc": auprc,
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "optimal_threshold": threshold,
        "n_positive": int(np.sum(y_true == 1)),
        "n_negative": int(np.sum(y_true == 0)),
    }


def compute_regression_metrics(y_true, y_pred):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) < 3:
        return {"spearman": float("nan"), "pearson": float("nan"),
                "mse": float("nan"), "r2": float("nan"), "n_samples": 0}
    sp, _ = spearmanr(y_true, y_pred)
    pe, _ = pearsonr(y_true, y_pred)
    return {"spearman": float(sp), "pearson": float(pe),
            "mse": float(mean_squared_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
            "n_samples": int(len(y_true))}


def compute_log2_rate(editing_rate):
    if pd.isna(editing_rate) or editing_rate < 0:
        return float("nan")
    rate = float(editing_rate)
    if rate > 1.0:
        rate = rate / 100.0
    return np.log2(rate + 0.01)


# ---------------------------------------------------------------------------
# Binary classification datasets
# ---------------------------------------------------------------------------

class BinaryPooledDataset(Dataset):
    def __init__(self, site_ids, labels, pooled_orig, pooled_edited, mode="subtraction"):
        self.site_ids = site_ids
        self.labels = labels
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
        return x, torch.tensor(self.labels[idx], dtype=torch.float32)


class StructureBinaryDataset(Dataset):
    def __init__(self, site_ids, labels, structure_delta):
        self.site_ids = site_ids
        self.labels = labels
        self.structure_delta = structure_delta

    def __len__(self):
        return len(self.site_ids)

    def __getitem__(self, idx):
        sid = self.site_ids[idx]
        struct = self.structure_delta.get(sid)
        x = torch.tensor(struct, dtype=torch.float32) if struct is not None else torch.zeros(7)
        return x, torch.tensor(self.labels[idx], dtype=torch.float32)


class TokenBinaryDataset(Dataset):
    def __init__(self, site_ids, labels, tokens_orig, tokens_edited):
        self.site_ids = site_ids
        self.labels = labels
        self.tokens_orig = tokens_orig
        self.tokens_edited = tokens_edited

    def __len__(self):
        return len(self.site_ids)

    def __getitem__(self, idx):
        sid = self.site_ids[idx]
        return {"tokens_orig": self.tokens_orig[sid],
                "tokens_edited": self.tokens_edited[sid]}, \
               torch.tensor(self.labels[idx], dtype=torch.float32)


def token_collate_fn(batch):
    items, targets = zip(*batch)
    max_len = max(item["tokens_orig"].shape[0] for item in items)
    d = items[0]["tokens_orig"].shape[1]
    t_orig = torch.zeros(len(items), max_len, d)
    t_edit = torch.zeros(len(items), max_len, d)
    for i, item in enumerate(items):
        L = item["tokens_orig"].shape[0]
        t_orig[i, :L] = item["tokens_orig"]
        t_edit[i, :L] = item["tokens_edited"]
    return {"tokens_orig": t_orig, "tokens_edited": t_edit}, torch.stack(targets)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class BinaryMLP(nn.Module):
    def __init__(self, d_input, hidden=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_input, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class DiffAttentionBinary(nn.Module):
    def __init__(self, d_model=640, n_heads=8, n_layers=2, d_hidden=256, dropout=0.3):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, batch):
        diff = batch["tokens_edited"] - batch["tokens_orig"]
        encoded = self.transformer(diff)
        return self.mlp(encoded.mean(dim=1)).squeeze(-1)


class CrossAttentionBinary(nn.Module):
    def __init__(self, d_model=640, n_heads=8, d_hidden=256, dropout=0.3):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, batch):
        attended, _ = self.cross_attn(
            query=batch["tokens_orig"], key=batch["tokens_edited"],
            value=batch["tokens_edited"])
        x = self.norm(batch["tokens_orig"] + attended)
        return self.mlp(x.mean(dim=1)).squeeze(-1)


# ---------------------------------------------------------------------------
# Generic training
# ---------------------------------------------------------------------------

def train_eval_binary(model, train_loader, val_loader, test_loader,
                      is_token_model=False, epochs=50, lr=1e-3,
                      weight_decay=1e-4, patience=10, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)

    best_val_auroc = 0.0
    patience_counter = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        for batch_data, batch_target in train_loader:
            optimizer.zero_grad()
            pred = model(batch_data)
            loss = F.binary_cross_entropy_with_logits(pred, batch_target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for bd, bt in val_loader:
                val_preds.append(torch.sigmoid(model(bd)).numpy())
                val_targets.append(bt.numpy())
        vp = np.concatenate(val_preds)
        vt = np.concatenate(val_targets)
        val_auroc = roc_auc_score(vt, vp) if len(np.unique(vt)) > 1 else 0.0

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

    model.eval()
    results = {"best_epoch": max(1, epoch - patience_counter)}
    for split_name, loader in [("val", val_loader), ("test", test_loader)]:
        all_preds, all_targets = [], []
        with torch.no_grad():
            for bd, bt in loader:
                all_preds.append(torch.sigmoid(model(bd)).numpy())
                all_targets.append(bt.numpy())
        yp = np.concatenate(all_preds)
        yt = np.concatenate(all_targets)
        results[split_name] = compute_binary_metrics(yt, yp)

    return results


# ---------------------------------------------------------------------------
# Rate prediction
# ---------------------------------------------------------------------------

class RatePooledDataset(Dataset):
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
        return x, torch.tensor(self.targets[idx], dtype=torch.float32)


class RateMLP(nn.Module):
    def __init__(self, d_input, hidden=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_input, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_eval_rate(model, train_loader, val_loader, test_loader,
                    epochs=80, lr=1e-3, weight_decay=1e-4, patience=15, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)

    best_val_sp = -float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        for bx, bt in train_loader:
            optimizer.zero_grad()
            loss = F.mse_loss(model(bx), bt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        vp, vt = [], []
        with torch.no_grad():
            for bx, bt in val_loader:
                vp.append(model(bx).numpy())
                vt.append(bt.numpy())
        vp = np.concatenate(vp)
        vt_arr = np.concatenate(vt)
        val_sp, _ = spearmanr(vt_arr, vp)
        val_sp = float(val_sp) if np.isfinite(val_sp) else -1.0

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
        ap, at = [], []
        with torch.no_grad():
            for bx, bt in loader:
                ap.append(model(bx).numpy())
                at.append(bt.numpy())
        results[split_name] = compute_regression_metrics(
            np.concatenate(at), np.concatenate(ap))
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
    logger.info("Loading A3A-filtered splits...")
    splits_df = pd.read_csv(SPLITS_CSV)
    logger.info("  Total: %d (pos=%d, neg=%d)",
                len(splits_df), (splits_df["label"] == 1).sum(), (splits_df["label"] == 0).sum())

    logger.info("Loading pooled embeddings...")
    pooled_orig = torch.load(EMB_DIR / "rnafm_pooled.pt", weights_only=False)
    pooled_edited = torch.load(EMB_DIR / "rnafm_pooled_edited.pt", weights_only=False)

    structure_delta = {}
    if STRUCT_CACHE.exists():
        data = np.load(STRUCT_CACHE, allow_pickle=True)
        feat_key = "delta_features" if "delta_features" in data else "features"
        if "site_ids" in data and feat_key in data:
            sids = data["site_ids"]
            feats = data[feat_key]
            structure_delta = {str(sid): feats[i] for i, sid in enumerate(sids)}
        logger.info("Loaded %d structure deltas", len(structure_delta))

    sequences = {}
    if SEQ_JSON.exists():
        sequences = json.loads(SEQ_JSON.read_text())

    # Filter to sites with embeddings
    available = set(pooled_orig.keys()) & set(pooled_edited.keys())
    df = splits_df[splits_df["site_id"].isin(available)].copy()
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]
    logger.info("Train: %d, Val: %d, Test: %d", len(train_df), len(val_df), len(test_df))

    all_results = {"dataset_info": {
        "filter": "APOBEC3A Only (strict)",
        "n_total": len(df),
        "n_positive": int((df["label"] == 1).sum()),
        "n_negative": int((df["label"] == 0).sum()),
        "n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df),
    }}

    # ==================================================================
    # TC Motif Validation
    # ==================================================================
    logger.info("\n" + "=" * 70)
    logger.info("TC MOTIF VALIDATION (A3A-filtered)")
    logger.info("=" * 70)

    pos_df = df[df["label"] == 1]
    tc_count = 0
    for sid in pos_df["site_id"]:
        seq = sequences.get(str(sid), "")
        if len(seq) >= 201 and seq[99].upper() in ("U", "T") and seq[100].upper() == "C":
            tc_count += 1
    tc_pct = 100.0 * tc_count / max(len(pos_df), 1)
    logger.info("  A3A-filtered positives: %d/%d (%.1f%%) have TC motif",
                tc_count, len(pos_df), tc_pct)

    per_ds_tc = {}
    for ds in sorted(pos_df["dataset_source"].unique()):
        ds_df = pos_df[pos_df["dataset_source"] == ds]
        ds_tc = sum(1 for sid in ds_df["site_id"]
                    if len(sequences.get(str(sid), "")) >= 201
                    and sequences[str(sid)][99].upper() in ("U", "T")
                    and sequences[str(sid)][100].upper() == "C")
        pct = 100.0 * ds_tc / max(len(ds_df), 1)
        per_ds_tc[ds] = {"n": len(ds_df), "tc": ds_tc, "pct": round(pct, 1)}
        logger.info("  %s: %d/%d (%.1f%%) TC", ds, ds_tc, len(ds_df), pct)

    all_results["tc_motif"] = {
        "overall_tc_pct": round(tc_pct, 1),
        "per_dataset": per_ds_tc,
    }

    # ==================================================================
    # Binary Classification (pooled models)
    # ==================================================================
    logger.info("\n" + "=" * 70)
    logger.info("BINARY CLASSIFICATION (A3A-filtered, pooled models)")
    logger.info("=" * 70)

    binary_results = {}

    def make_binary_loaders(mode):
        tr = DataLoader(BinaryPooledDataset(
            train_df["site_id"].tolist(), train_df["label"].values.astype(np.float32),
            pooled_orig, pooled_edited, mode), batch_size=64, shuffle=True, num_workers=0)
        va = DataLoader(BinaryPooledDataset(
            val_df["site_id"].tolist(), val_df["label"].values.astype(np.float32),
            pooled_orig, pooled_edited, mode), batch_size=64, shuffle=False, num_workers=0)
        te = DataLoader(BinaryPooledDataset(
            test_df["site_id"].tolist(), test_df["label"].values.astype(np.float32),
            pooled_orig, pooled_edited, mode), batch_size=64, shuffle=False, num_workers=0)
        return tr, va, te

    # PooledMLP
    logger.info("\n  --- PooledMLP ---")
    t0 = time.time()
    model = BinaryMLP(d_input=640)
    res = train_eval_binary(model, *make_binary_loaders("orig_only"), epochs=50, lr=1e-3)
    logger.info("  PooledMLP: AUROC=%.4f, AUPRC=%.4f (%.1fs)",
                res["test"]["auroc"], res["test"]["auprc"], time.time() - t0)
    binary_results["PooledMLP"] = res

    # SubtractionMLP
    logger.info("\n  --- SubtractionMLP ---")
    t0 = time.time()
    model = BinaryMLP(d_input=640)
    res = train_eval_binary(model, *make_binary_loaders("subtraction"), epochs=50, lr=1e-3)
    logger.info("  SubtractionMLP: AUROC=%.4f (%.1fs)", res["test"]["auroc"], time.time() - t0)
    binary_results["SubtractionMLP"] = res

    # ConcatMLP
    logger.info("\n  --- ConcatMLP ---")
    t0 = time.time()
    model = BinaryMLP(d_input=1280)
    res = train_eval_binary(model, *make_binary_loaders("concat"), epochs=50, lr=1e-3)
    logger.info("  ConcatMLP: AUROC=%.4f (%.1fs)", res["test"]["auroc"], time.time() - t0)
    binary_results["ConcatMLP"] = res

    # StructureOnly
    logger.info("\n  --- StructureOnly ---")
    t0 = time.time()
    tr_s = DataLoader(StructureBinaryDataset(
        train_df["site_id"].tolist(), train_df["label"].values.astype(np.float32),
        structure_delta), batch_size=64, shuffle=True, num_workers=0)
    va_s = DataLoader(StructureBinaryDataset(
        val_df["site_id"].tolist(), val_df["label"].values.astype(np.float32),
        structure_delta), batch_size=64, shuffle=False, num_workers=0)
    te_s = DataLoader(StructureBinaryDataset(
        test_df["site_id"].tolist(), test_df["label"].values.astype(np.float32),
        structure_delta), batch_size=64, shuffle=False, num_workers=0)
    model = BinaryMLP(d_input=7, hidden=64, dropout=0.3)
    res = train_eval_binary(model, tr_s, va_s, te_s, epochs=50, lr=5e-3)
    logger.info("  StructureOnly: AUROC=%.4f (%.1fs)", res["test"]["auroc"], time.time() - t0)
    binary_results["StructureOnly"] = res

    # Token-level models (load tokens only if needed, then free)
    tokens_path = EMB_DIR / "rnafm_tokens.pt"
    tokens_edited_path = EMB_DIR / "rnafm_tokens_edited.pt"
    has_tokens = tokens_path.exists() and tokens_edited_path.exists()

    if has_tokens:
        logger.info("\n  Loading token embeddings for DiffAttention and CrossAttention...")
        tokens_orig = torch.load(tokens_path, weights_only=False)
        tokens_edited = torch.load(tokens_edited_path, weights_only=False)

        token_available = set(tokens_orig.keys()) & set(tokens_edited.keys())
        train_tok = train_df[train_df["site_id"].isin(token_available)]
        val_tok = val_df[val_df["site_id"].isin(token_available)]
        test_tok = test_df[test_df["site_id"].isin(token_available)]

        tr_t = DataLoader(TokenBinaryDataset(
            train_tok["site_id"].tolist(), train_tok["label"].values.astype(np.float32),
            tokens_orig, tokens_edited), batch_size=16, shuffle=True,
            collate_fn=token_collate_fn, num_workers=0)
        va_t = DataLoader(TokenBinaryDataset(
            val_tok["site_id"].tolist(), val_tok["label"].values.astype(np.float32),
            tokens_orig, tokens_edited), batch_size=32, shuffle=False,
            collate_fn=token_collate_fn, num_workers=0)
        te_t = DataLoader(TokenBinaryDataset(
            test_tok["site_id"].tolist(), test_tok["label"].values.astype(np.float32),
            tokens_orig, tokens_edited), batch_size=32, shuffle=False,
            collate_fn=token_collate_fn, num_workers=0)

        # DiffAttention
        logger.info("\n  --- DiffAttention ---")
        t0 = time.time()
        model = DiffAttentionBinary()
        res = train_eval_binary(model, tr_t, va_t, te_t, is_token_model=True,
                                epochs=30, lr=5e-4, weight_decay=1e-4, patience=8)
        logger.info("  DiffAttention: AUROC=%.4f (%.1fs)", res["test"]["auroc"], time.time() - t0)
        binary_results["DiffAttention"] = res

        # CrossAttention
        logger.info("\n  --- CrossAttention ---")
        t0 = time.time()
        model = CrossAttentionBinary()
        res = train_eval_binary(model, tr_t, va_t, te_t, is_token_model=True,
                                epochs=30, lr=5e-4, weight_decay=1e-4, patience=8)
        logger.info("  CrossAttention: AUROC=%.4f (%.1fs)", res["test"]["auroc"], time.time() - t0)
        binary_results["CrossAttention"] = res

        # EditRNA-A3A
        logger.info("\n  --- EditRNA-A3A ---")
        t0 = time.time()
        # Build samples for EditRNA
        def build_samples(df_split):
            samples = []
            for _, row in df_split.iterrows():
                sid = str(row["site_id"])
                seq = sequences.get(sid, "A" * 201)
                edit_pos = min(100, len(seq) // 2)
                seq_list = list(seq)
                seq_list[edit_pos] = "C"
                seq = "".join(seq_list)
                flanking = get_flanking_context(seq, edit_pos)
                struct_d = structure_delta.get(sid)
                if struct_d is not None:
                    struct_d = np.array(struct_d, dtype=np.float32)
                concordance = np.zeros(5, dtype=np.float32)
                sample = APOBECSiteSample(
                    sequence=seq, edit_pos=edit_pos,
                    is_edited=float(row["label"]),
                    editing_rate_log2=compute_log2_rate(row.get("editing_rate", float("nan"))),
                    apobec_class=-1, structure_type=-1, tissue_spec_class=-1,
                    n_tissues_log2=float("nan"), exonic_function=-1,
                    conservation=float("nan"), cancer_survival=float("nan"),
                    tissue_rates=np.full(N_TISSUES, np.nan, dtype=np.float32),
                    hek293_rate=float("nan"),
                    flanking_context=flanking, concordance_features=concordance,
                    structure_delta=struct_d, site_id=sid,
                    chrom=str(row.get("chr", "")),
                    position=int(row.get("start", 0)),
                    gene=str(row.get("gene", "")),
                )
                samples.append(sample)
            return samples

        data_config = APOBECDataConfig(window_size=100)
        train_ds = APOBECDataset(build_samples(train_df), data_config)
        val_ds = APOBECDataset(build_samples(val_df), data_config)
        test_ds = APOBECDataset(build_samples(test_df), data_config)

        tr_e = DataLoader(train_ds, batch_size=32, shuffle=True,
                          collate_fn=apobec_collate_fn, num_workers=0)
        va_e = DataLoader(val_ds, batch_size=64, shuffle=False,
                          collate_fn=apobec_collate_fn, num_workers=0)
        te_e = DataLoader(test_ds, batch_size=64, shuffle=False,
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
        editrna = EditRNA_A3A(config=config, primary_encoder=cached_encoder)
        optimizer = AdamW(editrna.get_parameter_groups(), lr=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-7)

        best_val_auroc = 0.0
        patience_counter = 0
        best_state = None
        for epoch in range(1, 31):
            editrna.train()
            for batch in tr_e:
                batch = {k: v.to("cpu") if isinstance(v, torch.Tensor) else
                         ({kk: vv.to("cpu") if isinstance(vv, torch.Tensor) else vv
                           for kk, vv in v.items()} if isinstance(v, dict) else v)
                         for k, v in batch.items()}
                optimizer.zero_grad()
                output = editrna(batch)
                loss, _ = editrna.compute_loss(output, batch["targets"])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(editrna.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            editrna.eval()
            vp, vt = [], []
            with torch.no_grad():
                for batch in va_e:
                    batch = {k: v.to("cpu") if isinstance(v, torch.Tensor) else
                             ({kk: vv.to("cpu") if isinstance(vv, torch.Tensor) else vv
                               for kk, vv in v.items()} if isinstance(v, dict) else v)
                             for k, v in batch.items()}
                    out = editrna(batch)
                    vp.append(torch.sigmoid(out["predictions"]["binary_logit"].squeeze(-1)).cpu().numpy())
                    vt.append(batch["targets"]["binary"].cpu().numpy())
            vp_arr = np.concatenate(vp)
            vt_arr = np.concatenate(vt)
            val_auroc = roc_auc_score(vt_arr, vp_arr)
            if val_auroc > best_val_auroc + 1e-4:
                best_val_auroc = val_auroc
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in editrna.state_dict().items()}
            else:
                patience_counter += 1
            if patience_counter >= 8:
                break

        if best_state:
            editrna.load_state_dict(best_state)
        editrna.eval()
        tp, tt = [], []
        with torch.no_grad():
            for batch in te_e:
                batch = {k: v.to("cpu") if isinstance(v, torch.Tensor) else
                         ({kk: vv.to("cpu") if isinstance(vv, torch.Tensor) else vv
                           for kk, vv in v.items()} if isinstance(v, dict) else v)
                         for k, v in batch.items()}
                out = editrna(batch)
                tp.append(torch.sigmoid(out["predictions"]["binary_logit"].squeeze(-1)).cpu().numpy())
                tt.append(batch["targets"]["binary"].cpu().numpy())
        tp_arr = np.concatenate(tp)
        tt_arr = np.concatenate(tt)
        editrna_metrics = compute_binary_metrics(tt_arr, tp_arr)
        logger.info("  EditRNA-A3A: AUROC=%.4f (%.1fs)", editrna_metrics["auroc"], time.time() - t0)
        binary_results["EditRNA-A3A"] = {"test": editrna_metrics}

        # Free token memory before rate prediction
        del tokens_orig, tokens_edited
        import gc; gc.collect()
        logger.info("  Freed token memory")

    all_results["binary"] = binary_results

    # ==================================================================
    # Rate Prediction (pooled only - no tokens needed)
    # ==================================================================
    logger.info("\n" + "=" * 70)
    logger.info("RATE PREDICTION (A3A-filtered, pooled models)")
    logger.info("=" * 70)

    rate_df = df[(df["label"] == 1) & (df["editing_rate"].notna())].copy()
    rate_df["log2_rate"] = rate_df["editing_rate"].apply(compute_log2_rate)
    rate_df = rate_df[rate_df["log2_rate"].notna()].copy()
    r_train = rate_df[rate_df["split"] == "train"]
    r_val = rate_df[rate_df["split"] == "val"]
    r_test = rate_df[rate_df["split"] == "test"]
    logger.info("  Rate samples: train=%d, val=%d, test=%d", len(r_train), len(r_val), len(r_test))

    rate_results = {}

    def make_rate_loaders(mode):
        tr = DataLoader(RatePooledDataset(
            r_train["site_id"].tolist(), r_train["log2_rate"].values.astype(np.float32),
            pooled_orig, pooled_edited, mode), batch_size=64, shuffle=True, num_workers=0)
        va = DataLoader(RatePooledDataset(
            r_val["site_id"].tolist(), r_val["log2_rate"].values.astype(np.float32),
            pooled_orig, pooled_edited, mode), batch_size=64, shuffle=False, num_workers=0)
        te = DataLoader(RatePooledDataset(
            r_test["site_id"].tolist(), r_test["log2_rate"].values.astype(np.float32),
            pooled_orig, pooled_edited, mode), batch_size=64, shuffle=False, num_workers=0)
        return tr, va, te

    for model_name, mode, d_input in [
        ("PooledMLP", "orig_only", 640),
        ("SubtractionMLP", "subtraction", 640),
        ("ConcatMLP", "concat", 1280),
    ]:
        logger.info("\n  --- %s ---", model_name)
        t0 = time.time()
        model = RateMLP(d_input=d_input)
        res = train_eval_rate(model, *make_rate_loaders(mode))
        logger.info("  %s: Test Sp=%.4f (train=%.4f) in %.1fs",
                     model_name, res["test"]["spearman"], res["train"]["spearman"], time.time() - t0)
        rate_results[model_name] = res

    all_results["rate"] = rate_results

    # ==================================================================
    # Summary
    # ==================================================================
    print("\n" + "=" * 100)
    print("A3A-FILTERED EXPERIMENT RESULTS")
    print("=" * 100)

    print("\nBINARY CLASSIFICATION (Test Set):")
    print(f"{'Model':<20}{'AUROC':>10}{'AUPRC':>10}{'F1':>10}{'Precision':>12}{'Recall':>10}")
    print("-" * 72)
    for m in sorted(binary_results, key=lambda x: -binary_results[x].get("test", {}).get("auroc", 0)):
        t = binary_results[m].get("test", {})
        print(f"{m:<20}{t.get('auroc', 0):>10.4f}{t.get('auprc', 0):>10.4f}"
              f"{t.get('f1', 0):>10.4f}{t.get('precision', 0):>12.4f}{t.get('recall', 0):>10.4f}")

    print(f"\nRATE PREDICTION (Test Set):")
    print(f"{'Model':<20}{'Train Sp':>12}{'Val Sp':>12}{'Test Sp':>12}{'Test R2':>10}")
    print("-" * 66)
    for m in sorted(rate_results, key=lambda x: -rate_results[x]["test"].get("spearman", -1)):
        tr = rate_results[m]["train"]
        va = rate_results[m]["val"]
        te = rate_results[m]["test"]
        print(f"{m:<20}{tr['spearman']:>12.4f}{va['spearman']:>12.4f}"
              f"{te['spearman']:>12.4f}{te['r2']:>10.4f}")

    print(f"\nTC MOTIF VALIDATION:")
    print(f"  Overall: {tc_count}/{len(pos_df)} ({tc_pct:.1f}%) positives have TC motif")
    for ds, info in per_ds_tc.items():
        print(f"  {ds}: {info['tc']}/{info['n']} ({info['pct']}%)")

    total_time = time.time() - t_global
    logger.info("\nTotal time: %.1fs", total_time)

    # Save
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

    with open(OUTPUT_DIR / "a3a_filtered_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=serialize)
    logger.info("Saved results to %s", OUTPUT_DIR / "a3a_filtered_results.json")


if __name__ == "__main__":
    main()
