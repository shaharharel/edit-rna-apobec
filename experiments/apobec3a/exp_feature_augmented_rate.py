#!/usr/bin/env python
"""Feature-augmented rate prediction experiment.

Tests whether giving NN models access to the same hand-crafted features
that make Gradient Boosting strong can close the gap for rate prediction.

Strategies:
  A. SubtractionMLP + features (simple concat)
  B. EditRNA + features via gated fusion (d_gnn slot)
  C. DiffAttention + features (post-pool concat)
  D. ConcatMLP + features
  E. Multiple regularization configs (dropout, weight_decay, lr sweeps)
  F. Multiple seeds for stability

Models (each run with and without features):
  1. SubtractionMLP baseline vs SubtractionMLP+Features
  2. EditRNA_rate baseline vs EditRNA+Features (gated fusion)
  3. DiffAttention baseline vs DiffAttention+Features
  4. ConcatMLP baseline vs ConcatMLP+Features
  5. GB_HandFeatures and GB_AllFeatures (reference baselines)

Usage:
    python experiments/apobec3a/exp_feature_augmented_rate.py
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
import torch.nn.functional as F
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error, r2_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Paths
SPLITS_CSV = PROJECT_ROOT / "data" / "processed" / "splits_expanded.csv"
EMB_DIR = PROJECT_ROOT / "data" / "processed" / "embeddings"
STRUCTURE_CACHE = EMB_DIR / "vienna_structure_cache.npz"
SEQUENCES_JSON = PROJECT_ROOT / "data" / "processed" / "site_sequences.json"
LOOP_POS_CSV = (
    PROJECT_ROOT / "experiments" / "apobec" / "outputs"
    / "loop_position" / "loop_position_per_site.csv"
)
OUTPUT_DIR = (
    PROJECT_ROOT / "experiments" / "apobec" / "outputs" / "feature_augmented_rate"
)

DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_rate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) < 3:
        return {"spearman": float("nan"), "pearson": float("nan"),
                "mse": float("nan"), "r2": float("nan")}
    sp, _ = spearmanr(y_true, y_pred)
    pe, _ = pearsonr(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"spearman": float(sp), "pearson": float(pe),
            "mse": float(mse), "r2": float(r2)}


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_rate_data():
    df = pd.read_csv(SPLITS_CSV)
    df = df[df["editing_rate_normalized"].notna()].copy()
    df["target"] = np.log2(df["editing_rate_normalized"].values + 0.01)
    split_dfs = {s: df[df["split"] == s].copy() for s in ["train", "val", "test"]}
    logger.info("Rate data: train=%d, val=%d, test=%d",
                len(split_dfs["train"]), len(split_dfs["val"]), len(split_dfs["test"]))
    return df, split_dfs


def load_embeddings(load_tokens=False):
    logger.info("Loading pooled embeddings...")
    pooled_orig = torch.load(EMB_DIR / "rnafm_pooled.pt", weights_only=False)
    pooled_edited = torch.load(EMB_DIR / "rnafm_pooled_edited.pt", weights_only=False)
    tokens_orig = tokens_edited = None
    if load_tokens:
        logger.info("Loading token-level embeddings...")
        tokens_orig = torch.load(EMB_DIR / "rnafm_tokens.pt", weights_only=False)
        tokens_edited = torch.load(EMB_DIR / "rnafm_tokens_edited.pt", weights_only=False)
    return pooled_orig, pooled_edited, tokens_orig, tokens_edited


def load_structure_deltas():
    data = np.load(str(STRUCTURE_CACHE), allow_pickle=True)
    return {str(sid): data["delta_features"][i] for i, sid in enumerate(data["site_ids"])}


def load_sequences():
    with open(SEQUENCES_JSON) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Hand-Crafted Feature Extraction (same as binary exp)
# ---------------------------------------------------------------------------

def extract_motif_features(sequences: Dict[str, str], site_ids: List[str]) -> np.ndarray:
    """24-dim motif features: 5' dinuc (4) + 3' dinuc (4) + trinuc (16)."""
    TRINUC_MAP = {}
    bases = ["A", "C", "G", "U"]
    idx = 0
    for b1 in bases:
        for b2 in bases:
            TRINUC_MAP[b1 + b2] = idx
            idx += 1

    features = []
    for sid in site_ids:
        seq = sequences.get(sid, "N" * 201).upper().replace("T", "U")
        edit_pos = 100

        upstream = seq[edit_pos - 1] if edit_pos > 0 else "N"
        downstream = seq[edit_pos + 1] if edit_pos < len(seq) - 1 else "N"

        # 5' dinuc one-hot (4)
        motif_5p = upstream + "C"
        feat_5p = [1 if motif_5p == m else 0 for m in ["UC", "CC", "AC", "GC"]]

        # 3' dinuc one-hot (4)
        motif_3p = "C" + downstream
        feat_3p = [1 if motif_3p == m else 0 for m in ["CA", "CG", "CU", "CC"]]

        # Upstream trinuc one-hot (8 = 2 positions x 4 bases simplified)
        trinuc_up = [0] * 8
        for offset, base_idx_offset in [(-2, 0), (-1, 4)]:
            pos = edit_pos + offset
            if 0 <= pos < len(seq):
                b = seq[pos]
                for bi, base in enumerate(bases):
                    if b == base:
                        trinuc_up[base_idx_offset + bi] = 1

        # Downstream trinuc one-hot (8)
        trinuc_down = [0] * 8
        for offset, base_idx_offset in [(1, 0), (2, 4)]:
            pos = edit_pos + offset
            if 0 <= pos < len(seq):
                b = seq[pos]
                for bi, base in enumerate(bases):
                    if b == base:
                        trinuc_down[base_idx_offset + bi] = 1

        features.append(feat_5p + feat_3p + trinuc_up + trinuc_down)

    return np.array(features, dtype=np.float32)


def extract_loop_features(loop_df, site_ids: List[str]) -> np.ndarray:
    """9-dim loop position features."""
    loop_cols = [
        "is_unpaired", "loop_size", "dist_to_junction", "dist_to_apex",
        "relative_loop_position", "left_stem_length", "right_stem_length",
        "max_adjacent_stem_length", "local_unpaired_fraction",
    ]
    features = []
    for sid in site_ids:
        if sid in loop_df.index:
            row = loop_df.loc[sid, loop_cols]
            features.append(row.values.astype(np.float32))
        else:
            features.append(np.full(len(loop_cols), 0.0, dtype=np.float32))
    return np.array(features, dtype=np.float32)


def extract_structure_delta(structure_delta: Dict, site_ids: List[str]) -> np.ndarray:
    return np.array([structure_delta.get(sid, np.zeros(7)) for sid in site_ids],
                    dtype=np.float32)


def build_hand_features(site_ids, sequences, structure_delta, loop_df):
    """Build full hand feature matrix (motif + structure + loop)."""
    motif = extract_motif_features(sequences, site_ids)
    struct = extract_structure_delta(structure_delta, site_ids)
    loop = extract_loop_features(loop_df, site_ids)
    hand = np.concatenate([motif, struct, loop], axis=1)
    hand = np.nan_to_num(hand, nan=0.0)
    return hand


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class RateAugmentedDataset(Dataset):
    """Rate regression dataset with hand features + embeddings."""

    def __init__(self, site_ids, targets, pooled_orig, pooled_edited,
                 structure_delta, hand_features):
        self.site_ids = site_ids
        self.targets = targets
        self.pooled_orig = pooled_orig
        self.pooled_edited = pooled_edited
        self.structure_delta = structure_delta
        self.hand_features = hand_features

    def __len__(self):
        return len(self.site_ids)

    def __getitem__(self, idx):
        sid = self.site_ids[idx]
        return {
            "site_id": sid,
            "target": torch.tensor(self.targets[idx], dtype=torch.float32),
            "pooled_orig": self.pooled_orig[sid],
            "pooled_edited": self.pooled_edited[sid],
            "structure_delta": torch.tensor(
                self.structure_delta.get(sid, np.zeros(7)), dtype=torch.float32),
            "hand_features": torch.tensor(self.hand_features[idx], dtype=torch.float32),
        }


def rate_collate(batch):
    return {
        "site_ids": [b["site_id"] for b in batch],
        "targets": torch.stack([b["target"] for b in batch]),
        "pooled_orig": torch.stack([b["pooled_orig"] for b in batch]),
        "pooled_edited": torch.stack([b["pooled_edited"] for b in batch]),
        "structure_delta": torch.stack([b["structure_delta"] for b in batch]),
        "hand_features": torch.stack([b["hand_features"] for b in batch]),
    }


def create_loaders(split_dfs, pooled_orig, pooled_edited, structure_delta,
                   sequences, loop_df, batch_size=32):
    loaders = {}
    for split_name, sdf in split_dfs.items():
        available = set(pooled_orig.keys()) & set(pooled_edited.keys())
        sdf_filt = sdf[sdf["site_id"].isin(available)]
        site_ids = sdf_filt["site_id"].tolist()
        targets = sdf_filt["target"].values.astype(np.float32)
        hand = build_hand_features(site_ids, sequences, structure_delta, loop_df)

        ds = RateAugmentedDataset(
            site_ids=site_ids, targets=targets,
            pooled_orig=pooled_orig, pooled_edited=pooled_edited,
            structure_delta=structure_delta, hand_features=hand,
        )
        loaders[split_name] = DataLoader(
            ds, batch_size=batch_size, shuffle=(split_name == "train"),
            num_workers=0, collate_fn=rate_collate, drop_last=False,
        )
        logger.info("  %s: %d samples, hand_features=%d dims", split_name, len(ds), hand.shape[1])
    return loaders


# ---------------------------------------------------------------------------
# NN Model Architectures for Rate Regression
# ---------------------------------------------------------------------------

class SubtractionMLPRate(nn.Module):
    """Embedding delta MLP for rate regression."""
    def __init__(self, d_model=640, d_hand=0, hidden=256, dropout=0.3):
        super().__init__()
        d_in = d_model + d_hand
        self.mlp = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, batch):
        diff = batch["pooled_edited"] - batch["pooled_orig"]
        if "hand_features" in batch and self.mlp[0].in_features > 640:
            diff = torch.cat([diff, batch["hand_features"]], dim=-1)
        return {"rate_pred": self.mlp(diff)}


class ConcatMLPRate(nn.Module):
    """[before, after, features] → MLP for rate regression."""
    def __init__(self, d_model=640, d_hand=0, hidden=512, dropout=0.3):
        super().__init__()
        d_in = d_model * 2 + d_hand
        self.mlp = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, batch):
        parts = [batch["pooled_orig"], batch["pooled_edited"]]
        if "hand_features" in batch and self.mlp[0].in_features > 1280:
            parts.append(batch["hand_features"])
        return {"rate_pred": self.mlp(torch.cat(parts, dim=-1))}


class GatedFusionRateModel(nn.Module):
    """Gated fusion of edit embedding + hand features for rate regression.

    Architecture:
      primary_pooled (640) → proj → gated
      edit_delta (640)      → proj → gated
      hand_features (D)     → proj → gated
      → concat gated → output_proj → MLP → rate
    """
    def __init__(self, d_model=640, d_hand=33, d_proj=128, dropout=0.3):
        super().__init__()
        self.primary_proj = nn.Sequential(
            nn.Linear(d_model, d_proj), nn.GELU(), nn.LayerNorm(d_proj))
        self.edit_proj = nn.Sequential(
            nn.Linear(d_model, d_proj), nn.GELU(), nn.LayerNorm(d_proj))
        self.hand_proj = nn.Sequential(
            nn.Linear(d_hand, d_proj), nn.GELU(), nn.LayerNorm(d_proj))

        self.gate = nn.Linear(d_proj * 3, 3)
        self.fuse_proj = nn.Sequential(
            nn.Linear(d_proj * 3, d_proj * 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(d_proj * 2, d_proj),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_proj, 1),
        )

    def forward(self, batch):
        primary = batch["pooled_orig"]
        edit_delta = batch["pooled_edited"] - batch["pooled_orig"]
        hand = batch["hand_features"]

        p = self.primary_proj(primary)
        e = self.edit_proj(edit_delta)
        h = self.hand_proj(hand)

        concat = torch.cat([p, e, h], dim=-1)
        gates = torch.softmax(self.gate(concat), dim=-1)  # (B, 3)

        gated = torch.cat([
            gates[:, 0:1] * p,
            gates[:, 1:2] * e,
            gates[:, 2:3] * h,
        ], dim=-1)

        fused = self.fuse_proj(gated)
        return {"rate_pred": self.head(fused)}


class EditRNARateAugmented(nn.Module):
    """EditRNA-A3A with hand features via gated fusion for rate prediction."""

    def __init__(self, d_model=640, d_hand=33, d_edit=128, d_fused=256,
                 cached_encoder=None, dropout=0.3):
        super().__init__()
        from models.editrna_a3a import EditRNA_A3A, EditRNAConfig
        from models.fusion import GatedModalityFusion

        config = EditRNAConfig(
            primary_encoder="cached",
            d_model=d_model,
            d_edit=d_edit,
            d_fused=d_fused,
            pooled_only=True,
            use_gnn=False,
            use_structure_delta=True,
            head_dropout=dropout,
            fusion_dropout=dropout,
        )
        self.editrna = EditRNA_A3A(config=config, primary_encoder=cached_encoder)

        # Replace fusion with one that accepts hand features via d_gnn
        self.editrna.fusion = GatedModalityFusion(
            d_model=d_model,
            d_edit=d_edit,
            d_fused=d_fused,
            d_model_secondary=0,
            d_gnn=d_hand,
            dropout=dropout,
        )
        self.d_hand = d_hand

    def forward(self, batch):
        B = batch["targets"].shape[0]
        device = batch["targets"].device
        base = self.editrna
        cfg = base.config

        # Encode
        primary_out = base._encode_primary(
            ["N" * 201] * B, site_ids=batch["site_ids"], edited=False)
        primary_pooled = primary_out["pooled"]
        f_background = primary_pooled.unsqueeze(1)

        edited_seqs = base._make_edited_sequences(["N" * 201] * B,
            torch.full((B,), 100, dtype=torch.long, device=device))
        edited_out = base._encode_primary(
            edited_seqs, site_ids=batch["site_ids"], edited=True)
        f_edited = edited_out["pooled"].unsqueeze(1)

        edit_pos = torch.zeros(B, dtype=torch.long, device=device)
        seq_mask = torch.ones(B, 1, dtype=torch.bool, device=device)

        edit_emb = base.edit_embedding(
            f_background=f_background,
            f_edited=f_edited,
            edit_pos=edit_pos,
            flanking_context=torch.zeros(B, dtype=torch.long, device=device),
            structure_delta=batch["structure_delta"],
            concordance_features=torch.zeros(B, 5, device=device),
            seq_mask=seq_mask,
        )

        # Fuse with hand features
        fused = base.fusion(
            primary_pooled=primary_pooled,
            edit_emb=edit_emb,
            gnn_emb=batch["hand_features"],
        )

        # Rate head
        predictions = base.heads(fused, edit_emb)
        rate_pred = predictions["editing_rate"]
        return {"rate_pred": rate_pred}


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train_rate_model(
    model: nn.Module,
    loaders: Dict[str, DataLoader],
    model_name: str,
    epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    patience: int = 25,
    use_huber: bool = False,
) -> Dict:
    """Train NN model for rate regression with early stopping on val Spearman."""
    model = model.to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)

    best_spearman = -float("inf")
    best_epoch = 0
    patience_counter = 0
    best_state = None
    n_params = sum(p.numel() for p in model.parameters())

    logger.info("Training %s (params=%s, lr=%.0e, wd=%.0e, epochs=%d, patience=%d)...",
                model_name, f"{n_params:,}", lr, weight_decay, epochs, patience)
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for batch in loaders["train"]:
            batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            optimizer.zero_grad()
            output = model(batch)
            pred = output["rate_pred"].squeeze(-1)
            if use_huber:
                loss = F.huber_loss(pred, batch["targets"], delta=1.0)
            else:
                loss = F.mse_loss(pred, batch["targets"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1
        scheduler.step()

        val_metrics = evaluate_model(model, loaders.get("val"))
        val_sp = val_metrics.get("spearman", -999)

        if not np.isnan(val_sp) and val_sp > best_spearman + 1e-5:
            best_spearman = val_sp
            best_epoch = epoch
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if epoch % 20 == 0 or epoch == 1 or patience_counter >= patience:
            logger.info("  Epoch %3d  loss=%.4f  val_sp=%.4f  best_sp=%.4f  pat=%d/%d",
                        epoch, train_loss / max(n_batches, 1),
                        val_sp if not np.isnan(val_sp) else 0.0,
                        best_spearman, patience_counter, patience)

        if patience_counter >= patience:
            logger.info("  Early stop at epoch %d (best=%d, sp=%.4f)", epoch, best_epoch, best_spearman)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    elapsed = time.time() - t_start
    results = {"best_epoch": best_epoch, "time_seconds": round(elapsed, 1), "n_params": n_params}
    for split in ["train", "val", "test"]:
        if split in loaders:
            results[split] = evaluate_model(model, loaders[split])
    return results


@torch.no_grad()
def evaluate_model(model, loader):
    if loader is None:
        return {}
    model.eval()
    all_t, all_p = [], []
    for batch in loader:
        batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        output = model(batch)
        all_t.append(batch["targets"].cpu().numpy())
        all_p.append(output["rate_pred"].squeeze(-1).cpu().numpy())
    return compute_rate_metrics(np.concatenate(all_t), np.concatenate(all_p))


# ---------------------------------------------------------------------------
# Gradient Boosting Baselines
# ---------------------------------------------------------------------------

def train_gb_rate(split_dfs, sequences, structure_delta, loop_df,
                  pooled_orig, pooled_edited):
    """Train GB baselines for rate prediction."""
    from xgboost import XGBRegressor

    results = {}
    for variant, include_emb in [("GB_HandFeatures", False), ("GB_AllFeatures", True)]:
        logger.info("  Training %s...", variant)
        t0 = time.time()

        data = {}
        for split_name, sdf in split_dfs.items():
            site_ids = sdf["site_id"].tolist()
            targets = sdf["target"].values.astype(np.float32)
            hand = build_hand_features(site_ids, sequences, structure_delta, loop_df)

            if include_emb:
                emb_delta = np.array([
                    (pooled_edited[sid] - pooled_orig[sid]).numpy()
                    if sid in pooled_orig and sid in pooled_edited
                    else np.zeros(640, dtype=np.float32)
                    for sid in site_ids
                ], dtype=np.float32)
                feats = np.concatenate([hand, emb_delta], axis=1)
            else:
                feats = hand

            feats = np.nan_to_num(feats, nan=0.0)
            data[split_name] = (feats, targets)

        gb = XGBRegressor(
            n_estimators=500, max_depth=5, learning_rate=0.01,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            early_stopping_rounds=20, random_state=42,
        )
        gb.fit(data["train"][0], data["train"][1],
               eval_set=[(data["val"][0], data["val"][1])], verbose=False)

        elapsed = time.time() - t0
        model_results = {"time_seconds": round(elapsed, 1), "n_features": data["train"][0].shape[1]}
        for split_name in ["train", "val", "test"]:
            y_pred = gb.predict(data[split_name][0])
            model_results[split_name] = compute_rate_metrics(data[split_name][1], y_pred)
        results[variant] = model_results
        logger.info("    %s test: sp=%.4f, pearson=%.4f",
                    variant, model_results["test"]["spearman"], model_results["test"]["pearson"])

    return results


# ---------------------------------------------------------------------------
# Model Configuration Registry
# ---------------------------------------------------------------------------

def get_model_configs(d_hand: int):
    """Return all model configurations to train.

    Each config: (name, model_factory, training_kwargs)
    """
    configs = []

    # --- SubtractionMLP variants ---
    configs.append(("Sub_baseline", lambda: SubtractionMLPRate(d_hand=0, dropout=0.3),
                    {"lr": 1e-3, "weight_decay": 1e-3, "patience": 25}))
    configs.append(("Sub+Features", lambda: SubtractionMLPRate(d_hand=d_hand, dropout=0.3),
                    {"lr": 1e-3, "weight_decay": 1e-3, "patience": 25}))
    configs.append(("Sub+Feat_heavyreg", lambda: SubtractionMLPRate(d_hand=d_hand, dropout=0.5),
                    {"lr": 5e-4, "weight_decay": 1e-2, "patience": 30}))
    configs.append(("Sub+Feat_huber", lambda: SubtractionMLPRate(d_hand=d_hand, dropout=0.4),
                    {"lr": 1e-3, "weight_decay": 5e-3, "patience": 25, "use_huber": True}))

    # --- ConcatMLP variants ---
    configs.append(("Concat_baseline", lambda: ConcatMLPRate(d_hand=0, dropout=0.3),
                    {"lr": 1e-3, "weight_decay": 1e-3, "patience": 25}))
    configs.append(("Concat+Features", lambda: ConcatMLPRate(d_hand=d_hand, dropout=0.3),
                    {"lr": 1e-3, "weight_decay": 1e-3, "patience": 25}))
    configs.append(("Concat+Feat_heavyreg", lambda: ConcatMLPRate(d_hand=d_hand, dropout=0.5),
                    {"lr": 5e-4, "weight_decay": 1e-2, "patience": 30}))

    # --- Gated Fusion (new architecture) ---
    configs.append(("GatedFusion_small", lambda: GatedFusionRateModel(d_hand=d_hand, d_proj=64, dropout=0.3),
                    {"lr": 1e-3, "weight_decay": 5e-3, "patience": 25}))
    configs.append(("GatedFusion_med", lambda: GatedFusionRateModel(d_hand=d_hand, d_proj=128, dropout=0.3),
                    {"lr": 1e-3, "weight_decay": 5e-3, "patience": 25}))
    configs.append(("GatedFusion_heavyreg", lambda: GatedFusionRateModel(d_hand=d_hand, d_proj=128, dropout=0.5),
                    {"lr": 5e-4, "weight_decay": 1e-2, "patience": 30}))
    configs.append(("GatedFusion_huber", lambda: GatedFusionRateModel(d_hand=d_hand, d_proj=128, dropout=0.4),
                    {"lr": 1e-3, "weight_decay": 5e-3, "patience": 25, "use_huber": True}))

    return configs


def get_editrna_configs(d_hand, cached_encoder):
    """EditRNA configs (separate because they need cached_encoder)."""
    configs = []

    # Baseline EditRNA (no hand features)
    def make_editrna_baseline():
        from models.editrna_a3a import EditRNA_A3A, EditRNAConfig
        config = EditRNAConfig(
            primary_encoder="cached", d_model=640, d_edit=128, d_fused=256,
            pooled_only=True, use_structure_delta=True,
            head_dropout=0.3, fusion_dropout=0.3,
        )
        model = EditRNA_A3A(config=config, primary_encoder=cached_encoder)

        class Wrapper(nn.Module):
            def __init__(self, base):
                super().__init__()
                self.base = base
            def forward(self, batch):
                B = batch["targets"].shape[0]
                device = batch["targets"].device
                eb = {
                    "sequences": ["N" * 201] * B,
                    "site_ids": batch["site_ids"],
                    "edit_pos": torch.full((B,), 100, dtype=torch.long, device=device),
                    "flanking_context": torch.zeros(B, dtype=torch.long, device=device),
                    "concordance_features": torch.zeros(B, 5, device=device),
                    "structure_delta": batch["structure_delta"],
                }
                out = self.base(eb)
                return {"rate_pred": out["predictions"]["editing_rate"]}
        return Wrapper(model)

    configs.append(("EditRNA_baseline", make_editrna_baseline,
                    {"lr": 1e-3, "weight_decay": 1e-2, "patience": 25}))
    configs.append(("EditRNA_baseline_heavyreg", make_editrna_baseline,
                    {"lr": 5e-4, "weight_decay": 5e-2, "patience": 30}))

    # EditRNA + hand features via gated fusion
    def make_editrna_augmented(dropout=0.3):
        return EditRNARateAugmented(
            d_hand=d_hand, d_edit=128, d_fused=256,
            cached_encoder=cached_encoder, dropout=dropout,
        )

    configs.append(("EditRNA+Features", lambda: make_editrna_augmented(0.3),
                    {"lr": 1e-3, "weight_decay": 1e-2, "patience": 25}))
    configs.append(("EditRNA+Feat_heavyreg", lambda: make_editrna_augmented(0.5),
                    {"lr": 5e-4, "weight_decay": 5e-2, "patience": 30}))
    configs.append(("EditRNA+Feat_huber", lambda: make_editrna_augmented(0.4),
                    {"lr": 1e-3, "weight_decay": 1e-2, "patience": 25, "use_huber": True}))

    return configs


# ---------------------------------------------------------------------------
# Multi-Seed Runner
# ---------------------------------------------------------------------------

def run_with_seeds(model_factory, model_name, loaders, train_kwargs, seeds=(42, 123, 456)):
    """Run model with multiple seeds, return per-seed and aggregate results."""
    seed_results = []
    for seed in seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = model_factory()
        result = train_rate_model(model, loaders, f"{model_name}_s{seed}", **train_kwargs)
        seed_results.append(result)
        logger.info("  %s seed=%d: test_sp=%.4f, val_sp=%.4f",
                    model_name, seed, result["test"]["spearman"], result["val"]["spearman"])

    # Aggregate: mean and std of test metrics across seeds
    test_sps = [r["test"]["spearman"] for r in seed_results if not np.isnan(r["test"]["spearman"])]
    test_pes = [r["test"]["pearson"] for r in seed_results if not np.isnan(r["test"]["pearson"])]

    aggregate = {
        "mean_test_spearman": float(np.mean(test_sps)) if test_sps else float("nan"),
        "std_test_spearman": float(np.std(test_sps)) if len(test_sps) > 1 else 0.0,
        "mean_test_pearson": float(np.mean(test_pes)) if test_pes else float("nan"),
        "std_test_pearson": float(np.std(test_pes)) if len(test_pes) > 1 else 0.0,
        "seeds": list(seeds),
        "per_seed": seed_results,
    }
    return aggregate


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment():
    t_total = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("FEATURE-AUGMENTED RATE PREDICTION EXPERIMENT")
    logger.info("=" * 70)

    # Load data
    full_df, split_dfs = load_rate_data()
    pooled_orig, pooled_edited, _, _ = load_embeddings(load_tokens=False)
    structure_delta = load_structure_deltas()
    sequences = load_sequences()

    loop_df = pd.read_csv(LOOP_POS_CSV).set_index("site_id") if LOOP_POS_CSV.exists() else pd.DataFrame()
    logger.info("Loop features: %d sites", len(loop_df))

    # Create dataloaders
    loaders = create_loaders(split_dfs, pooled_orig, pooled_edited,
                             structure_delta, sequences, loop_df, batch_size=32)

    d_hand = loaders["train"].dataset.hand_features.shape[1]
    logger.info("Hand feature dimension: %d", d_hand)

    all_results = {
        "experiment": "feature_augmented_rate",
        "description": "NN + hand features for rate prediction (multi-seed, multi-config)",
        "target_transform": "log2(editing_rate_normalized + 0.01)",
        "n_train": len(split_dfs["train"]),
        "n_val": len(split_dfs["val"]),
        "n_test": len(split_dfs["test"]),
        "d_hand": d_hand,
        "models": {},
    }

    # ------------------------------------------------------------------
    # 1. GB Baselines
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("GRADIENT BOOSTING BASELINES")
    logger.info("=" * 70)

    gb_results = train_gb_rate(split_dfs, sequences, structure_delta, loop_df,
                               pooled_orig, pooled_edited)
    all_results["models"].update(gb_results)

    # ------------------------------------------------------------------
    # 2. NN Models (non-EditRNA)
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("NN MODELS (SubtractionMLP, ConcatMLP, GatedFusion)")
    logger.info("=" * 70)

    SEEDS = (42, 123, 456)
    model_configs = get_model_configs(d_hand)

    for name, factory, train_kwargs in model_configs:
        logger.info("\n--- %s ---", name)
        train_kw = {"epochs": 200, **train_kwargs}
        result = run_with_seeds(factory, name, loaders, train_kw, seeds=SEEDS)
        all_results["models"][name] = result
        logger.info("  >> %s: mean_test_sp=%.4f +/- %.4f",
                    name, result["mean_test_spearman"], result["std_test_spearman"])

    # ------------------------------------------------------------------
    # 3. EditRNA Models
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("EDITRNA MODELS (with and without features)")
    logger.info("=" * 70)

    from models.encoders import CachedRNAEncoder
    cached_encoder = CachedRNAEncoder(
        tokens_cache=None,
        pooled_cache=pooled_orig,
        tokens_edited_cache=None,
        pooled_edited_cache=pooled_edited,
        d_model=640,
    )

    editrna_configs = get_editrna_configs(d_hand, cached_encoder)

    for name, factory, train_kwargs in editrna_configs:
        logger.info("\n--- %s ---", name)
        train_kw = {"epochs": 200, **train_kwargs}
        result = run_with_seeds(factory, name, loaders, train_kw, seeds=SEEDS)
        all_results["models"][name] = result
        logger.info("  >> %s: mean_test_sp=%.4f +/- %.4f",
                    name, result["mean_test_spearman"], result["std_test_spearman"])

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total_time = time.time() - t_total
    all_results["total_time_seconds"] = round(total_time, 1)

    # Print summary table
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY (sorted by test Spearman)")
    logger.info("=" * 70)
    logger.info("%-30s  %10s  %10s  %10s", "Model", "Test Sp", "Val Sp", "Params")
    logger.info("-" * 65)

    summary = []
    for name, mdata in all_results["models"].items():
        if "mean_test_spearman" in mdata:
            test_sp = mdata["mean_test_spearman"]
            val_sp = np.mean([r["val"]["spearman"] for r in mdata["per_seed"]
                             if not np.isnan(r["val"]["spearman"])])
            n_p = mdata["per_seed"][0].get("n_params", "?")
        else:
            test_sp = mdata.get("test", {}).get("spearman", float("nan"))
            val_sp = mdata.get("val", {}).get("spearman", float("nan"))
            n_p = mdata.get("n_features", "?")
        summary.append((name, test_sp, val_sp, n_p))

    for name, tsp, vsp, np_ in sorted(summary, key=lambda x: -x[1] if not np.isnan(x[1]) else -999):
        logger.info("%-30s  %10.4f  %10.4f  %10s", name, tsp, vsp, str(np_))

    # Save
    out_path = OUTPUT_DIR / "feature_augmented_rate_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("\nResults saved to %s", out_path)
    logger.info("Total time: %.1f minutes", total_time / 60)


if __name__ == "__main__":
    run_experiment()
