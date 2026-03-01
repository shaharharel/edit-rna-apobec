#!/usr/bin/env python
"""Dual-Pooled Architecture for Rate Prediction.

Key idea: Use TWO pooled embeddings instead of token-level attention:
  1. Global pooled: RNA-FM pooled embedding of the full 201nt sequence
  2. Local pooled:  Mean-pool of RNA-FM token embeddings in a window around the edit site

This captures both global sequence context and local edit context without the
heavy parameterization of full token-level attention (which overfits on n=523).

Architectures tested:
  A. DualSubtraction: [global_diff; local_diff] -> MLP
  B. DualConcat:      [global_orig; global_edit; local_orig; local_edit] -> MLP
  C. DualGated:       Learned gate fusing global and local diffs -> MLP
  D. DualBilinear:    Bilinear interaction between global and local -> MLP
  E. DualHierarchical: local_diff -> project; concat with global_diff -> MLP

Window sizes tested: 5, 10, 20, 40 tokens around edit site

Usage:
    python experiments/apobec/exp_dual_pooled_rate.py
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
from sklearn.metrics import mean_squared_error, r2_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logger = logging.getLogger(__name__)

EMB_DIR = PROJECT_ROOT / "data" / "processed" / "embeddings"
SPLITS_CSV = PROJECT_ROOT / "data" / "processed" / "splits_expanded.csv"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "apobec" / "outputs" / "dual_pooled_rate"


# ---------------------------------------------------------------------------
# Helpers
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
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) < 3:
        return {"spearman": float("nan"), "pearson": float("nan"),
                "mse": float("nan"), "r2": float("nan"), "n_samples": 0}
    sp_rho, _ = spearmanr(y_true, y_pred)
    pe_r, _ = pearsonr(y_true, y_pred)
    return {"spearman": float(sp_rho), "pearson": float(pe_r),
            "mse": float(mean_squared_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
            "n_samples": int(len(y_true))}


def extract_local_pooled(tokens_dict, site_ids, window_size, edit_pos=100):
    """Extract local pooled embeddings by mean-pooling tokens around edit site.

    Args:
        tokens_dict: {site_id: tensor(L, 640)}
        site_ids: list of site_ids
        window_size: half-window (extract edit_pos-w to edit_pos+w+1)
        edit_pos: position of edit in token sequence (default 100 for 201nt window)

    Returns:
        {site_id: tensor(640,)} local pooled embeddings
    """
    local_pooled = {}
    for sid in site_ids:
        tokens = tokens_dict[sid]  # (L, 640)
        L = tokens.shape[0]
        center = min(edit_pos, L - 1)
        start = max(0, center - window_size)
        end = min(L, center + window_size + 1)
        local_pooled[sid] = tokens[start:end].mean(dim=0)  # (640,)
    return local_pooled


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DualPooledDataset(Dataset):
    """Dataset with global + local pooled embeddings."""

    def __init__(self, site_ids, targets, global_orig, global_edited,
                 local_orig, local_edited, mode="dual_subtraction"):
        self.site_ids = site_ids
        self.targets = targets
        self.global_orig = global_orig
        self.global_edited = global_edited
        self.local_orig = local_orig
        self.local_edited = local_edited
        self.mode = mode

    def __len__(self):
        return len(self.site_ids)

    def __getitem__(self, idx):
        sid = self.site_ids[idx]
        g_orig = self.global_orig[sid]
        g_edit = self.global_edited[sid]
        l_orig = self.local_orig[sid]
        l_edit = self.local_edited[sid]

        if self.mode == "dual_subtraction":
            # [global_diff; local_diff] -> 1280-dim
            x = torch.cat([g_edit - g_orig, l_edit - l_orig], dim=-1)
        elif self.mode == "dual_concat":
            # [g_orig; g_edit; l_orig; l_edit] -> 2560-dim
            x = torch.cat([g_orig, g_edit, l_orig, l_edit], dim=-1)
        elif self.mode == "dual_components":
            # Return all 4 components separately for gated/bilinear models
            x = {
                "global_diff": g_edit - g_orig,
                "local_diff": l_edit - l_orig,
                "global_orig": g_orig,
                "local_orig": l_orig,
            }
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        return x, target


def component_collate_fn(batch):
    """Collate for component mode (dict inputs)."""
    items, targets = zip(*batch)
    if isinstance(items[0], dict):
        batched = {}
        for key in items[0]:
            batched[key] = torch.stack([item[key] for item in items])
        return batched, torch.stack(targets)
    else:
        return torch.stack(items), torch.stack(targets)


# ---------------------------------------------------------------------------
# Architectures
# ---------------------------------------------------------------------------

class DualSubtractionMLP(nn.Module):
    """[global_diff; local_diff] -> MLP"""
    def __init__(self, d_emb=640, hidden=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_emb * 2, hidden),
            nn.GELU(),
            nn.BatchNorm1d(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class DualConcatMLP(nn.Module):
    """[g_orig; g_edit; l_orig; l_edit] -> MLP"""
    def __init__(self, d_emb=640, hidden=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_emb * 4, hidden),
            nn.GELU(),
            nn.BatchNorm1d(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class DualGatedFusion(nn.Module):
    """Learned gating between global and local diffs."""
    def __init__(self, d_emb=640, hidden=256, dropout=0.3):
        super().__init__()
        # Project global and local to same space
        self.global_proj = nn.Linear(d_emb, hidden)
        self.local_proj = nn.Linear(d_emb, hidden)
        # Gate
        self.gate = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, batch):
        g_diff = self.global_proj(batch["global_diff"])
        l_diff = self.local_proj(batch["local_diff"])
        gate_val = self.gate(torch.cat([g_diff, l_diff], dim=-1))
        fused = gate_val * g_diff + (1 - gate_val) * l_diff
        return self.head(fused).squeeze(-1)


class DualBilinear(nn.Module):
    """Bilinear interaction between global and local."""
    def __init__(self, d_emb=640, d_proj=128, hidden=128, dropout=0.3):
        super().__init__()
        self.global_proj = nn.Linear(d_emb, d_proj)
        self.local_proj = nn.Linear(d_emb, d_proj)
        self.bilinear = nn.Bilinear(d_proj, d_proj, hidden)
        self.head = nn.Sequential(
            nn.GELU(),
            nn.BatchNorm1d(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, batch):
        g = self.global_proj(batch["global_diff"])
        l = self.local_proj(batch["local_diff"])
        interaction = self.bilinear(g, l)
        return self.head(interaction).squeeze(-1)


class DualHierarchical(nn.Module):
    """local_diff -> compress -> concat with global_diff -> MLP"""
    def __init__(self, d_emb=640, d_local_proj=128, hidden=256, dropout=0.3):
        super().__init__()
        self.local_compress = nn.Sequential(
            nn.Linear(d_emb, d_local_proj),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(d_emb + d_local_proj, hidden),
            nn.GELU(),
            nn.BatchNorm1d(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, batch):
        g_diff = batch["global_diff"]
        l_compressed = self.local_compress(batch["local_diff"])
        combined = torch.cat([g_diff, l_compressed], dim=-1)
        return self.head(combined).squeeze(-1)


class DualCrossAttend(nn.Module):
    """Cross-attention between global and local (as 1-token sequences)."""
    def __init__(self, d_emb=640, d_proj=256, n_heads=4, hidden=128, dropout=0.3):
        super().__init__()
        self.global_proj = nn.Linear(d_emb, d_proj)
        self.local_proj = nn.Linear(d_emb, d_proj)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_proj, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_proj)
        self.head = nn.Sequential(
            nn.Linear(d_proj, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, batch):
        # Treat global and local as 2-token sequences
        g_diff = self.global_proj(batch["global_diff"]).unsqueeze(1)  # (B, 1, d)
        l_diff = self.local_proj(batch["local_diff"]).unsqueeze(1)    # (B, 1, d)
        kv = torch.cat([g_diff, l_diff], dim=1)  # (B, 2, d)
        attended, _ = self.cross_attn(query=g_diff, key=kv, value=kv)
        out = self.norm(g_diff + attended).squeeze(1)
        return self.head(out).squeeze(-1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_eval(model, train_loader, val_loader, test_loader,
               is_component_model=False, epochs=100, lr=5e-4,
               weight_decay=1e-3, patience=12, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)

    best_val_sp = -float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        for batch_data, batch_target in train_loader:
            optimizer.zero_grad()
            pred = model(batch_data)
            loss = F.mse_loss(pred, batch_target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # Validate using Spearman (our actual metric)
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch_data, batch_target in val_loader:
                pred = model(batch_data)
                val_preds.append(pred.numpy())
                val_targets.append(batch_target.numpy())
        vp = np.concatenate(val_preds)
        vt = np.concatenate(val_targets)
        val_sp, _ = spearmanr(vt, vp)
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
    results = {"best_epoch": max(1, epoch - patience_counter)}
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
    logger.info("Loading pooled embeddings...")
    pooled_orig = torch.load(EMB_DIR / "rnafm_pooled.pt", weights_only=False)
    pooled_edited = torch.load(EMB_DIR / "rnafm_pooled_edited.pt", weights_only=False)

    logger.info("Loading token embeddings for local pooling...")
    tokens_path = EMB_DIR / "rnafm_tokens.pt"
    tokens_edited_path = EMB_DIR / "rnafm_tokens_edited.pt"
    if not tokens_path.exists() or not tokens_edited_path.exists():
        logger.error("Token embeddings required for local pooling. Exiting.")
        return
    tokens_orig = torch.load(tokens_path, weights_only=False)
    tokens_edited = torch.load(tokens_edited_path, weights_only=False)
    logger.info("  Loaded %d token embeddings", len(tokens_orig))

    splits_df = pd.read_csv(SPLITS_CSV)

    # Filter to positive sites with valid rate
    available = set(pooled_orig.keys()) & set(pooled_edited.keys()) & \
                set(tokens_orig.keys()) & set(tokens_edited.keys())
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
    logger.info("Rate samples — Train: %d, Val: %d, Test: %d",
                len(train_df), len(val_df), len(test_df))

    all_site_ids = list(available & set(rate_df["site_id"].values))

    # ------------------------------------------------------------------
    # Pre-compute local pooled embeddings for different window sizes
    # ------------------------------------------------------------------
    WINDOW_SIZES = [5, 10, 20, 40]
    local_pooled_cache = {}

    for w in WINDOW_SIZES:
        logger.info("Pre-computing local pooled embeddings (window=%d)...", w)
        t0 = time.time()
        local_pooled_cache[w] = {
            "orig": extract_local_pooled(tokens_orig, all_site_ids, w),
            "edited": extract_local_pooled(tokens_edited, all_site_ids, w),
        }
        logger.info("  Done in %.1fs", time.time() - t0)

    # Free token memory
    del tokens_orig, tokens_edited
    import gc; gc.collect()
    logger.info("Freed token embedding memory")

    # ------------------------------------------------------------------
    # Run all architectures × window sizes
    # ------------------------------------------------------------------
    all_results = {}

    for w in WINDOW_SIZES:
        local_orig = local_pooled_cache[w]["orig"]
        local_edited = local_pooled_cache[w]["edited"]

        logger.info("\n" + "=" * 80)
        logger.info("WINDOW SIZE = %d (±%d tokens around edit)", 2 * w + 1, w)
        logger.info("=" * 80)

        # Helper to create data loaders
        def make_loaders(mode, collate=None):
            tr = DataLoader(DualPooledDataset(
                train_df["site_id"].tolist(), train_df["log2_rate"].values.astype(np.float32),
                pooled_orig, pooled_edited, local_orig, local_edited, mode),
                batch_size=64, shuffle=True, num_workers=0,
                collate_fn=collate or torch.utils.data.dataloader.default_collate)
            va = DataLoader(DualPooledDataset(
                val_df["site_id"].tolist(), val_df["log2_rate"].values.astype(np.float32),
                pooled_orig, pooled_edited, local_orig, local_edited, mode),
                batch_size=64, shuffle=False, num_workers=0,
                collate_fn=collate or torch.utils.data.dataloader.default_collate)
            te = DataLoader(DualPooledDataset(
                test_df["site_id"].tolist(), test_df["log2_rate"].values.astype(np.float32),
                pooled_orig, pooled_edited, local_orig, local_edited, mode),
                batch_size=64, shuffle=False, num_workers=0,
                collate_fn=collate or torch.utils.data.dataloader.default_collate)
            return tr, va, te

        # Architecture A: DualSubtraction
        name = f"DualSub_w{w}"
        logger.info("\n  --- %s ---", name)
        t0 = time.time()
        loaders = make_loaders("dual_subtraction")
        model = DualSubtractionMLP(d_emb=640, hidden=256, dropout=0.3)
        res = train_eval(model, *loaders, epochs=100, lr=5e-4, weight_decay=1e-3, patience=12)
        elapsed = time.time() - t0
        logger.info("  %s: Test Sp=%.4f (train=%.4f) in %.1fs",
                     name, res["test"]["spearman"], res["train"]["spearman"], elapsed)
        all_results[name] = {"architecture": "DualSubtraction", "window": w,
                             **{k: v for k, v in res.items()}, "time_seconds": elapsed}

        # Architecture B: DualConcat
        name = f"DualConcat_w{w}"
        logger.info("\n  --- %s ---", name)
        t0 = time.time()
        loaders = make_loaders("dual_concat")
        model = DualConcatMLP(d_emb=640, hidden=256, dropout=0.3)
        res = train_eval(model, *loaders, epochs=100, lr=5e-4, weight_decay=1e-3, patience=12)
        elapsed = time.time() - t0
        logger.info("  %s: Test Sp=%.4f in %.1fs", name, res["test"]["spearman"], elapsed)
        all_results[name] = {"architecture": "DualConcat", "window": w,
                             **{k: v for k, v in res.items()}, "time_seconds": elapsed}

        # Architecture C: DualGated
        name = f"DualGated_w{w}"
        logger.info("\n  --- %s ---", name)
        t0 = time.time()
        loaders = make_loaders("dual_components", collate=component_collate_fn)
        model = DualGatedFusion(d_emb=640, hidden=256, dropout=0.3)
        res = train_eval(model, *loaders, epochs=100, lr=5e-4, weight_decay=1e-3, patience=12)
        elapsed = time.time() - t0
        logger.info("  %s: Test Sp=%.4f in %.1fs", name, res["test"]["spearman"], elapsed)
        all_results[name] = {"architecture": "DualGated", "window": w,
                             **{k: v for k, v in res.items()}, "time_seconds": elapsed}

        # Architecture D: DualBilinear
        name = f"DualBilinear_w{w}"
        logger.info("\n  --- %s ---", name)
        t0 = time.time()
        loaders = make_loaders("dual_components", collate=component_collate_fn)
        model = DualBilinear(d_emb=640, d_proj=128, hidden=128, dropout=0.3)
        res = train_eval(model, *loaders, epochs=100, lr=5e-4, weight_decay=1e-3, patience=12)
        elapsed = time.time() - t0
        logger.info("  %s: Test Sp=%.4f in %.1fs", name, res["test"]["spearman"], elapsed)
        all_results[name] = {"architecture": "DualBilinear", "window": w,
                             **{k: v for k, v in res.items()}, "time_seconds": elapsed}

        # Architecture E: DualHierarchical
        name = f"DualHier_w{w}"
        logger.info("\n  --- %s ---", name)
        t0 = time.time()
        loaders = make_loaders("dual_components", collate=component_collate_fn)
        model = DualHierarchical(d_emb=640, d_local_proj=128, hidden=256, dropout=0.3)
        res = train_eval(model, *loaders, epochs=100, lr=5e-4, weight_decay=1e-3, patience=12)
        elapsed = time.time() - t0
        logger.info("  %s: Test Sp=%.4f in %.1fs", name, res["test"]["spearman"], elapsed)
        all_results[name] = {"architecture": "DualHierarchical", "window": w,
                             **{k: v for k, v in res.items()}, "time_seconds": elapsed}

        # Architecture F: DualCrossAttend
        name = f"DualCross_w{w}"
        logger.info("\n  --- %s ---", name)
        t0 = time.time()
        loaders = make_loaders("dual_components", collate=component_collate_fn)
        model = DualCrossAttend(d_emb=640, d_proj=256, n_heads=4, hidden=128, dropout=0.3)
        res = train_eval(model, *loaders, epochs=100, lr=5e-4, weight_decay=1e-3, patience=12)
        elapsed = time.time() - t0
        logger.info("  %s: Test Sp=%.4f in %.1fs", name, res["test"]["spearman"], elapsed)
        all_results[name] = {"architecture": "DualCrossAttend", "window": w,
                             **{k: v for k, v in res.items()}, "time_seconds": elapsed}

    # ==================================================================
    # Summary
    # ==================================================================
    print("\n" + "=" * 110)
    print("DUAL-POOLED RATE PREDICTION - ALL RESULTS")
    print("=" * 110)
    header = f"{'Model':<30}{'Window':>8}{'Train Sp':>12}{'Val Sp':>12}{'Test Sp':>12}{'Test R2':>10}{'Time':>8}"
    print(header)
    print("-" * 110)

    sorted_results = sorted(all_results.items(),
                            key=lambda x: x[1]["test"].get("spearman", -1), reverse=True)

    for name, res in sorted_results:
        w = res.get("window", "?")
        tr = res["train"].get("spearman", float("nan"))
        va = res["val"].get("spearman", float("nan"))
        te = res["test"].get("spearman", float("nan"))
        r2 = res["test"].get("r2", float("nan"))
        t = res.get("time_seconds", 0)
        print(f"{name:<30}{w:>8}{tr:>12.4f}{va:>12.4f}{te:>12.4f}{r2:>10.4f}{t:>7.0f}s")

    print("=" * 110)
    print(f"\nBaselines to beat:")
    print(f"  PooledMLP (global only, no edit):   Test Spearman = 0.211")
    print(f"  SubtractionMLP (global diff):       Test Spearman = 0.038")
    print(f"  ConcatMLP (global concat):          Test Spearman = 0.198")
    print(f"  DiffAttention (token, overfits):    Test Spearman = 0.155")

    best_name, best_res = sorted_results[0]
    print(f"\nBest dual-pooled: {best_name} Test Spearman = {best_res['test']['spearman']:.4f}")

    # Best per architecture
    print("\nBest per architecture:")
    arch_best = {}
    for name, res in sorted_results:
        arch = res["architecture"]
        if arch not in arch_best:
            arch_best[arch] = (name, res)
    for arch, (name, res) in arch_best.items():
        print(f"  {arch:<25} {name:<30} Sp={res['test']['spearman']:.4f} (w={res['window']})")

    # Best per window
    print("\nBest per window size:")
    win_best = {}
    for name, res in sorted_results:
        w = res["window"]
        if w not in win_best:
            win_best[w] = (name, res)
    for w in sorted(win_best):
        name, res = win_best[w]
        print(f"  w={w:<5} {name:<30} Sp={res['test']['spearman']:.4f}")

    total_time = time.time() - t_global
    logger.info("\nTotal time: %.1fs (%.1f min)", total_time, total_time / 60)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
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

    with open(OUTPUT_DIR / "dual_pooled_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=serialize)
    logger.info("Saved results to %s", OUTPUT_DIR / "dual_pooled_results.json")

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------
    # Heatmap: architecture × window size
    architectures = ["DualSubtraction", "DualConcat", "DualGated",
                     "DualBilinear", "DualHierarchical", "DualCrossAttend"]

    data_matrix = np.full((len(architectures), len(WINDOW_SIZES)), np.nan)
    for i, arch in enumerate(architectures):
        for j, w in enumerate(WINDOW_SIZES):
            for name, res in all_results.items():
                if res["architecture"] == arch and res["window"] == w:
                    data_matrix[i, j] = res["test"]["spearman"]

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(data_matrix, aspect="auto", cmap="RdYlGn", vmin=-0.1, vmax=0.4)
    ax.set_xticks(range(len(WINDOW_SIZES)))
    ax.set_xticklabels([f"±{w}\n({2*w+1} tok)" for w in WINDOW_SIZES])
    ax.set_yticks(range(len(architectures)))
    ax.set_yticklabels(architectures, fontsize=9)
    ax.set_xlabel("Local Window Size", fontsize=11)
    ax.set_ylabel("Architecture", fontsize=11)

    for i in range(len(architectures)):
        for j in range(len(WINDOW_SIZES)):
            v = data_matrix[i, j]
            if np.isfinite(v):
                color = "white" if v < 0.1 else "black"
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9, color=color)

    plt.colorbar(im, ax=ax, label="Test Spearman", shrink=0.8)

    # Add baseline lines
    ax.axhline(y=-0.5, color="red", linestyle="--", linewidth=0)  # dummy for legend
    fig.text(0.02, 0.02, "Baselines: PooledMLP=0.211 | ConcatMLP=0.198 | DiffAttn=0.155",
             fontsize=8, color="red", style="italic")

    ax.set_title("Dual-Pooled Rate Prediction: Architecture × Window Size\n(Test Spearman)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "dual_pooled_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Train vs Test gap plot
    fig, ax = plt.subplots(figsize=(14, 6))
    names_sorted = [n for n, _ in sorted_results]
    test_vals = [r["test"]["spearman"] for _, r in sorted_results]
    train_vals = [r["train"]["spearman"] for _, r in sorted_results]
    x = np.arange(len(names_sorted))
    ax.bar(x - 0.18, train_vals, 0.35, label="Train", alpha=0.7, color="steelblue")
    ax.bar(x + 0.18, test_vals, 0.35, label="Test", alpha=0.7, color="darkorange")
    ax.axhline(y=0.211, color="red", linestyle="--", linewidth=2, label="PooledMLP (0.211)")
    ax.set_xticks(x)
    ax.set_xticklabels(names_sorted, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Spearman rho")
    ax.set_title("Dual-Pooled: Train vs Test (sorted by test Spearman)", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "dual_pooled_train_vs_test.png", dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("Saved plots to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
