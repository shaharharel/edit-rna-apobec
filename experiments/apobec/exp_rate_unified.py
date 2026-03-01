#!/usr/bin/env python
"""Unified rate prediction experiment with all architectures.

Tests ALL model architectures for editing rate regression using
NORMALIZED rates with log2(editing_rate_normalized + 0.01) target transform.

Models (ordered by complexity):
  1. Mean Baseline           - predicts training set mean
  2. StructureOnly           - MLP on 7-dim structure delta
  3. GB_HandFeatures         - XGBoost with hand-crafted features only
  4. GB_AllFeatures          - XGBoost with hand-crafted + embedding features
  5. PooledMLP               - MLP on pooled before-edit embedding
  6. SubtractionMLP          - MLP on (after - before) pooled embedding
  7. ConcatMLP               - MLP on concatenated [before, after] pooled
  8. CrossAttention_reg      - Cross-attention over token-level embeddings
  9. DiffAttention_reg       - Token diff + TransformerEncoder
 10. EditRNA_rate            - Full EditRNA-A3A model

Usage:
    python experiments/apobec/exp_rate_unified.py
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
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
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
    PROJECT_ROOT / "experiments" / "apobec" / "outputs" / "rate_unified"
)

SEED = 42
DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_rate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute regression metrics for rate prediction."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

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

def load_rate_data() -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Load and prepare rate prediction data.

    Returns splits_df (full with rate info) and split_dfs dict.
    """
    df = pd.read_csv(SPLITS_CSV)
    # Keep only rows with valid editing_rate_normalized
    df = df[df["editing_rate_normalized"].notna()].copy()
    # Apply log2 transform
    df["target"] = np.log2(df["editing_rate_normalized"].values + 0.01)

    split_dfs = {}
    for split in ["train", "val", "test"]:
        split_dfs[split] = df[df["split"] == split].copy()

    logger.info(
        "Rate data: train=%d, val=%d, test=%d",
        len(split_dfs["train"]), len(split_dfs["val"]), len(split_dfs["test"]),
    )
    return df, split_dfs


def load_embeddings(load_tokens: bool = False):
    """Load pre-computed embeddings."""
    logger.info("Loading pooled embeddings...")
    pooled_orig = torch.load(EMB_DIR / "rnafm_pooled.pt", weights_only=False)
    pooled_edited = torch.load(EMB_DIR / "rnafm_pooled_edited.pt", weights_only=False)
    logger.info("  %d pooled embeddings loaded", len(pooled_orig))

    tokens_orig = None
    tokens_edited = None
    if load_tokens:
        logger.info("Loading token-level embeddings...")
        tokens_orig = torch.load(EMB_DIR / "rnafm_tokens.pt", weights_only=False)
        tokens_edited = torch.load(EMB_DIR / "rnafm_tokens_edited.pt", weights_only=False)
        logger.info("  %d token embeddings loaded", len(tokens_orig))

    return pooled_orig, pooled_edited, tokens_orig, tokens_edited


def load_structure_deltas() -> Dict[str, np.ndarray]:
    """Load structure delta features from ViennaRNA cache."""
    data = np.load(str(STRUCTURE_CACHE), allow_pickle=True)
    sids = data["site_ids"]
    feats = data["delta_features"]
    structure_delta = {str(sid): feats[i] for i, sid in enumerate(sids)}
    logger.info("  %d structure deltas loaded", len(structure_delta))
    return structure_delta


# ---------------------------------------------------------------------------
# PyTorch Dataset for Rate Regression
# ---------------------------------------------------------------------------

class RateDataset(Dataset):
    """Dataset for rate regression with pre-computed embeddings."""

    def __init__(
        self,
        site_ids: List[str],
        targets: np.ndarray,
        pooled_orig: Dict[str, torch.Tensor],
        pooled_edited: Dict[str, torch.Tensor],
        structure_delta: Dict[str, np.ndarray],
        tokens_orig: Optional[Dict[str, torch.Tensor]] = None,
        tokens_edited: Optional[Dict[str, torch.Tensor]] = None,
    ):
        self.site_ids = site_ids
        self.targets = targets
        self.pooled_orig = pooled_orig
        self.pooled_edited = pooled_edited
        self.structure_delta = structure_delta
        self.tokens_orig = tokens_orig
        self.tokens_edited = tokens_edited

    def __len__(self) -> int:
        return len(self.site_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sid = self.site_ids[idx]
        item = {
            "site_id": sid,
            "target": torch.tensor(self.targets[idx], dtype=torch.float32),
            "pooled_orig": self.pooled_orig[sid],
            "pooled_edited": self.pooled_edited[sid],
        }
        if sid in self.structure_delta:
            item["structure_delta"] = torch.tensor(
                self.structure_delta[sid], dtype=torch.float32
            )
        else:
            item["structure_delta"] = torch.zeros(7, dtype=torch.float32)

        if self.tokens_orig is not None and sid in self.tokens_orig:
            item["tokens_orig"] = self.tokens_orig[sid]
            item["tokens_edited"] = self.tokens_edited[sid]

        return item


def rate_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for RateDataset."""
    result = {
        "site_ids": [b["site_id"] for b in batch],
        "targets": torch.stack([b["target"] for b in batch]),
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


def create_rate_dataloaders(
    split_dfs: Dict[str, pd.DataFrame],
    pooled_orig, pooled_edited,
    structure_delta,
    tokens_orig=None, tokens_edited=None,
    batch_size: int = 32,
) -> Dict[str, DataLoader]:
    """Create train/val/test DataLoaders for rate regression."""
    loaders = {}
    for split_name, sdf in split_dfs.items():
        available = set(pooled_orig.keys()) & set(pooled_edited.keys())
        sdf_filt = sdf[sdf["site_id"].isin(available)]
        site_ids = sdf_filt["site_id"].tolist()
        targets = sdf_filt["target"].values.astype(np.float32)

        ds = RateDataset(
            site_ids=site_ids,
            targets=targets,
            pooled_orig=pooled_orig,
            pooled_edited=pooled_edited,
            structure_delta=structure_delta,
            tokens_orig=tokens_orig,
            tokens_edited=tokens_edited,
        )
        loaders[split_name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            num_workers=0,
            collate_fn=rate_collate_fn,
            drop_last=False,
        )
        logger.info("  %s: %d samples", split_name, len(ds))
    return loaders


# ---------------------------------------------------------------------------
# Regression Model Wrappers (adapt classification baselines to regression)
# ---------------------------------------------------------------------------

class StructureOnlyRegressor(nn.Module):
    """MLP on 7-dim structure delta for rate regression."""

    def __init__(self, dropout=0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(7, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, batch):
        return {"rate_pred": self.mlp(batch["structure_delta"])}


class PooledMLPRegressor(nn.Module):
    """MLP on pooled before-edit embedding for rate regression."""

    def __init__(self, d_model=640, dropout=0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, batch):
        return {"rate_pred": self.mlp(batch["pooled_orig"])}


class SubtractionMLPRegressor(nn.Module):
    """MLP on (after - before) pooled embedding for rate regression."""

    def __init__(self, d_model=640, dropout=0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, batch):
        diff = batch["pooled_edited"] - batch["pooled_orig"]
        return {"rate_pred": self.mlp(diff)}


class ConcatMLPRegressor(nn.Module):
    """MLP on concatenated [before, after] pooled embeddings for rate regression."""

    def __init__(self, d_model=640, dropout=0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, batch):
        x = torch.cat([batch["pooled_orig"], batch["pooled_edited"]], dim=-1)
        return {"rate_pred": self.mlp(x)}


class CrossAttentionRegressor(nn.Module):
    """Cross-attention over token-level embeddings for rate regression."""

    def __init__(self, d_model=640, n_heads=4, d_hidden=128, dropout=0.5):
        super().__init__()
        # Project down to d_hidden to reduce parameters
        self.proj = nn.Linear(d_model, d_hidden)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_hidden,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_hidden)
        self.mlp = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, batch):
        tokens_orig = self.proj(batch["tokens_orig"])
        tokens_edited = self.proj(batch["tokens_edited"])

        attended, _ = self.cross_attn(
            query=tokens_orig, key=tokens_edited, value=tokens_edited
        )
        x = self.norm(tokens_orig + attended)
        pooled = x.mean(dim=1)
        return {"rate_pred": self.mlp(pooled)}


class DiffAttentionRegressor(nn.Module):
    """Token diff + TransformerEncoder for rate regression."""

    def __init__(self, d_model=640, n_heads=4, d_hidden=128, dropout=0.5):
        super().__init__()
        self.proj = nn.Linear(d_model, d_hidden)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_hidden,
            nhead=n_heads,
            dim_feedforward=d_hidden * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.mlp = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, batch):
        tokens_orig = self.proj(batch["tokens_orig"])
        tokens_edited = self.proj(batch["tokens_edited"])
        diff = tokens_edited - tokens_orig
        encoded = self.transformer(diff)
        pooled = encoded.mean(dim=1)
        return {"rate_pred": self.mlp(pooled)}


# ---------------------------------------------------------------------------
# Neural Training Loop
# ---------------------------------------------------------------------------

def train_nn_model(
    model: nn.Module,
    loaders: Dict[str, DataLoader],
    model_name: str,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    patience: int = 15,
) -> Dict:
    """Train a neural network model for rate regression.

    Early stopping on validation Spearman correlation.
    """
    model = model.to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)

    best_spearman = -float("inf")
    best_epoch = 0
    patience_counter = 0
    best_state = None

    logger.info("Training %s for up to %d epochs (patience=%d)...", model_name, epochs, patience)
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        n_batches = 0
        for batch in loaders["train"]:
            batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            optimizer.zero_grad()
            output = model(batch)
            pred = output["rate_pred"].squeeze(-1)
            loss = F.mse_loss(pred, batch["targets"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1
        scheduler.step()
        avg_loss = train_loss / max(n_batches, 1)

        # Validate
        val_metrics = evaluate_nn_model(model, loaders.get("val"), model_name)
        val_sp = val_metrics.get("spearman", -999)

        if not np.isnan(val_sp) and val_sp > best_spearman + 1e-5:
            best_spearman = val_sp
            best_epoch = epoch
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1 or patience_counter >= patience:
            logger.info(
                "  Epoch %3d  loss=%.4f  val_spearman=%.4f  patience=%d/%d",
                epoch, avg_loss, val_sp if not np.isnan(val_sp) else 0.0,
                patience_counter, patience,
            )

        if patience_counter >= patience:
            logger.info(
                "  Early stopping at epoch %d (best=%d, spearman=%.4f)",
                epoch, best_epoch, best_spearman,
            )
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    elapsed = time.time() - t_start

    # Evaluate on all splits
    results = {"best_epoch": best_epoch, "train_time_seconds": round(elapsed, 1)}
    for split_name in ["train", "val", "test"]:
        if split_name in loaders:
            metrics = evaluate_nn_model(model, loaders[split_name], model_name)
            results[split_name] = metrics

    return results


@torch.no_grad()
def evaluate_nn_model(
    model: nn.Module,
    loader: Optional[DataLoader],
    model_name: str,
) -> Dict[str, float]:
    """Evaluate a neural model on a DataLoader."""
    if loader is None:
        return {}
    model.eval()
    all_targets = []
    all_preds = []
    for batch in loader:
        batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        output = model(batch)
        pred = output["rate_pred"].squeeze(-1).cpu().numpy()
        all_targets.append(batch["targets"].cpu().numpy())
        all_preds.append(pred)
    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    return compute_rate_metrics(y_true, y_pred)


# ---------------------------------------------------------------------------
# Gradient Boosting Features
# ---------------------------------------------------------------------------

def extract_motif_features(sequences: Dict[str, str], site_ids: List[str]) -> np.ndarray:
    """Extract motif features from 201nt sequences with edit at position 100.

    Returns array of shape (n_sites, n_features) with:
    - 5' motif one-hot: TC, CC, AC, GC (4 features)
    - 3' motif one-hot: CA, CG, CU, CC (4 features)
    - Trinucleotide context encoded (upstream + downstream)
    - Local nucleotide composition (20nt window)
    """
    features = []
    for sid in site_ids:
        seq = sequences.get(sid, "N" * 201)
        seq = seq.upper().replace("T", "U")
        edit_pos = 100

        # 5' dinucleotide (position -1 and edit)
        upstream = seq[edit_pos - 1] if edit_pos > 0 else "N"
        motif_5p = upstream + "C"
        tc = 1 if motif_5p == "UC" else 0
        cc = 1 if motif_5p == "CC" else 0
        ac = 1 if motif_5p == "AC" else 0
        gc = 1 if motif_5p == "GC" else 0

        # 3' dinucleotide (edit and position +1)
        downstream = seq[edit_pos + 1] if edit_pos < len(seq) - 1 else "N"
        motif_3p = "C" + downstream
        ca = 1 if motif_3p == "CA" else 0
        cg = 1 if motif_3p == "CG" else 0
        cu = 1 if motif_3p == "CU" else 0
        cc_3p = 1 if motif_3p == "CC" else 0

        # Local composition in 20nt window centered on edit
        w_start = max(0, edit_pos - 10)
        w_end = min(len(seq), edit_pos + 11)
        window = seq[w_start:w_end]
        w_len = max(len(window), 1)
        frac_a = window.count("A") / w_len
        frac_c = window.count("C") / w_len
        frac_g = window.count("G") / w_len
        frac_u = window.count("U") / w_len

        # GC content
        gc_content = (window.count("G") + window.count("C")) / w_len

        features.append([
            tc, cc, ac, gc,     # 5' motif
            ca, cg, cu, cc_3p,  # 3' motif
            frac_a, frac_c, frac_g, frac_u,  # composition
            gc_content,
        ])

    return np.array(features, dtype=np.float32)


def build_gb_features(
    split_dfs: Dict[str, pd.DataFrame],
    sequences: Dict[str, str],
    structure_delta: Dict[str, np.ndarray],
    pooled_orig: Dict[str, torch.Tensor],
    pooled_edited: Dict[str, torch.Tensor],
    include_embeddings: bool = True,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Build feature matrices for gradient boosting."""
    # Load loop position features
    loop_df = None
    if LOOP_POS_CSV.exists():
        loop_df = pd.read_csv(LOOP_POS_CSV)
        loop_df = loop_df.set_index("site_id")
        loop_features = [
            "is_unpaired", "loop_size", "dist_to_left_boundary",
            "dist_to_right_boundary", "dist_to_nearest_stem",
            "relative_loop_position", "dist_to_apex",
            "left_stem_length", "right_stem_length",
            "max_adjacent_stem_length", "dist_to_junction",
            "local_unpaired_fraction", "mfe",
        ]
        logger.info("  Loaded loop position features: %d sites, %d features",
                     len(loop_df), len(loop_features))
    else:
        loop_features = []
        logger.warning("  Loop position file not found at %s", LOOP_POS_CSV)

    result = {}
    for split_name, sdf in split_dfs.items():
        site_ids = sdf["site_id"].tolist()
        targets = sdf["target"].values.astype(np.float32)

        # Motif features
        motif_feats = extract_motif_features(sequences, site_ids)

        # Structure delta features (7 dims)
        struct_feats = np.array([
            structure_delta.get(sid, np.zeros(7)) for sid in site_ids
        ], dtype=np.float32)

        # Loop position features
        if loop_df is not None:
            lp_feats = []
            for sid in site_ids:
                if sid in loop_df.index:
                    row = loop_df.loc[sid, loop_features]
                    lp_feats.append(row.values.astype(np.float32))
                else:
                    lp_feats.append(np.full(len(loop_features), np.nan, dtype=np.float32))
            lp_feats = np.array(lp_feats, dtype=np.float32)
        else:
            lp_feats = np.empty((len(site_ids), 0), dtype=np.float32)

        # Hand features
        hand_feats = np.concatenate([motif_feats, struct_feats, lp_feats], axis=1)

        if include_embeddings:
            # Embedding delta (after - before pooled, 640 dims)
            emb_delta = []
            for sid in site_ids:
                if sid in pooled_orig and sid in pooled_edited:
                    delta = (pooled_edited[sid] - pooled_orig[sid]).numpy()
                else:
                    delta = np.zeros(640, dtype=np.float32)
                emb_delta.append(delta)
            emb_delta = np.array(emb_delta, dtype=np.float32)
            all_feats = np.concatenate([hand_feats, emb_delta], axis=1)
        else:
            all_feats = hand_feats

        result[split_name] = (all_feats, targets)

    return result


# ---------------------------------------------------------------------------
# EditRNA-A3A Rate Regression Wrapper
# ---------------------------------------------------------------------------

class EditRNARateWrapper(nn.Module):
    """Wrapper that runs EditRNA-A3A and extracts rate prediction."""

    def __init__(self, editrna_model):
        super().__init__()
        self.editrna = editrna_model

    def forward(self, batch):
        B = batch["targets"].shape[0]
        device = batch["targets"].device

        editrna_batch = {
            "sequences": ["N" * 201] * B,
            "site_ids": batch["site_ids"],
            "edit_pos": torch.full((B,), 100, dtype=torch.long, device=device),
            "flanking_context": torch.zeros(B, dtype=torch.long, device=device),
            "concordance_features": torch.zeros(B, 5, device=device),
            "structure_delta": batch["structure_delta"],
        }
        output = self.editrna(editrna_batch)
        # Use the editing_rate prediction head
        rate_pred = output["predictions"]["editing_rate"]
        return {"rate_pred": rate_pred}


def build_editrna_rate_model(
    pooled_orig, pooled_edited, tokens_orig, tokens_edited,
):
    """Build EditRNA-A3A configured for rate regression."""
    from models.editrna_a3a import EditRNA_A3A, EditRNAConfig
    from models.encoders import CachedRNAEncoder

    cached_encoder = CachedRNAEncoder(
        tokens_cache=tokens_orig,
        pooled_cache=pooled_orig,
        tokens_edited_cache=tokens_edited,
        pooled_edited_cache=pooled_edited,
        d_model=640,
    )

    # Use a smaller architecture with heavy regularization for rate
    config = EditRNAConfig(
        primary_encoder="cached",
        d_model=640,
        d_edit=128,
        d_fused=256,
        edit_n_heads=4,
        pooled_only=(tokens_orig is None),
        use_structure_delta=True,
        head_dropout=0.3,
        fusion_dropout=0.3,
        weight_decay=1e-2,
        learning_rate=1e-3,
    )
    editrna = EditRNA_A3A(config=config, primary_encoder=cached_encoder)
    return EditRNARateWrapper(editrna)


def train_editrna_rate(
    model: nn.Module,
    loaders: Dict[str, DataLoader],
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    patience: int = 15,
    rate_weight: float = 10.0,
) -> Dict:
    """Train EditRNA-A3A with multitask loss (binary + rate).

    Uses the rate prediction from the multi-task head with heavy rate weighting.
    """
    model = model.to(DEVICE)
    editrna = model.editrna

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)

    best_spearman = -float("inf")
    best_epoch = 0
    patience_counter = 0
    best_state = None

    logger.info(
        "Training EditRNA_rate for up to %d epochs (patience=%d, rate_weight=%.1f)...",
        epochs, patience, rate_weight,
    )
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0

        for batch in loaders["train"]:
            batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            B = batch["targets"].shape[0]
            device = batch["targets"].device

            editrna_batch = {
                "sequences": ["N" * 201] * B,
                "site_ids": batch["site_ids"],
                "edit_pos": torch.full((B,), 100, dtype=torch.long, device=device),
                "flanking_context": torch.zeros(B, dtype=torch.long, device=device),
                "concordance_features": torch.zeros(B, 5, device=device),
                "structure_delta": batch["structure_delta"],
            }
            output = editrna(editrna_batch)
            predictions = output["predictions"]

            # Rate loss (MSE on editing_rate head)
            rate_pred = predictions["editing_rate"].squeeze(-1)
            rate_loss = F.mse_loss(rate_pred, batch["targets"])

            # Total loss: focus on rate
            loss = rate_weight * rate_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = train_loss / max(n_batches, 1)

        # Validate
        val_metrics = evaluate_nn_model(model, loaders.get("val"), "EditRNA_rate")
        val_sp = val_metrics.get("spearman", -999)

        if not np.isnan(val_sp) and val_sp > best_spearman + 1e-5:
            best_spearman = val_sp
            best_epoch = epoch
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1 or patience_counter >= patience:
            logger.info(
                "  Epoch %3d  loss=%.4f  val_spearman=%.4f  patience=%d/%d",
                epoch, avg_loss, val_sp if not np.isnan(val_sp) else 0.0,
                patience_counter, patience,
            )

        if patience_counter >= patience:
            logger.info(
                "  Early stopping at epoch %d (best=%d, spearman=%.4f)",
                epoch, best_epoch, best_spearman,
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    elapsed = time.time() - t_start
    results = {"best_epoch": best_epoch, "train_time_seconds": round(elapsed, 1)}
    for split_name in ["train", "val", "test"]:
        if split_name in loaders:
            results[split_name] = evaluate_nn_model(model, loaders[split_name], "EditRNA_rate")
    return results


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment():
    """Run the unified rate prediction experiment."""
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("=" * 70)
    logger.info("UNIFIED RATE PREDICTION EXPERIMENT")
    logger.info("=" * 70)

    full_df, split_dfs = load_rate_data()
    structure_delta = load_structure_deltas()

    # Load sequences for GB motif features
    with open(SEQUENCES_JSON) as f:
        sequences = json.load(f)
    logger.info("  %d sequences loaded", len(sequences))

    all_results = {
        "description": "Unified rate prediction with all architectures (normalized rates)",
        "target_transform": "log2(editing_rate_normalized + 0.01)",
        "n_train": len(split_dfs["train"]),
        "n_val": len(split_dfs["val"]),
        "n_test": len(split_dfs["test"]),
        "models": {},
    }

    # ------------------------------------------------------------------
    # 1. Mean Baseline
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("MODEL 1: Mean Baseline")
    logger.info("=" * 70)

    train_mean = split_dfs["train"]["target"].mean()
    logger.info("  Training set mean target: %.4f", train_mean)

    mean_results = {}
    for split_name, sdf in split_dfs.items():
        y_true = sdf["target"].values
        y_pred = np.full_like(y_true, train_mean)
        mean_results[split_name] = compute_rate_metrics(y_true, y_pred)

    all_results["models"]["Mean Baseline"] = mean_results
    logger.info("  Test: spearman=%.4f, mse=%.4f",
                mean_results["test"]["spearman"], mean_results["test"]["mse"])

    # ------------------------------------------------------------------
    # 2. StructureOnly
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("MODEL 2: StructureOnly")
    logger.info("=" * 70)

    pooled_orig, pooled_edited, _, _ = load_embeddings(load_tokens=False)
    loaders = create_rate_dataloaders(
        split_dfs, pooled_orig, pooled_edited, structure_delta,
        batch_size=32,
    )

    struct_model = StructureOnlyRegressor(dropout=0.3)
    n_params = sum(p.numel() for p in struct_model.parameters())
    logger.info("  Parameters: %s", f"{n_params:,}")

    struct_results = train_nn_model(
        struct_model, loaders, "StructureOnly",
        epochs=100, lr=1e-3, weight_decay=1e-3, patience=15,
    )
    all_results["models"]["StructureOnly"] = {
        k: v for k, v in struct_results.items() if k in ("train", "val", "test")
    }
    all_results["models"]["StructureOnly"]["n_params"] = n_params

    # ------------------------------------------------------------------
    # 3-4. Gradient Boosting (Hand Features and All Features)
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("MODEL 3-4: Gradient Boosting")
    logger.info("=" * 70)

    try:
        from xgboost import XGBRegressor

        for variant, include_emb in [("GB_HandFeatures", False), ("GB_AllFeatures", True)]:
            logger.info("\n  --- %s ---", variant)
            gb_data = build_gb_features(
                split_dfs, sequences, structure_delta,
                pooled_orig, pooled_edited,
                include_embeddings=include_emb,
            )
            X_train, y_train = gb_data["train"]
            X_val, y_val = gb_data["val"]
            X_test, y_test = gb_data["test"]

            logger.info("  Feature matrix: train=%s, val=%s, test=%s",
                        X_train.shape, X_val.shape, X_test.shape)

            # Replace NaN with 0 for XGBoost
            X_train = np.nan_to_num(X_train, nan=0.0)
            X_val = np.nan_to_num(X_val, nan=0.0)
            X_test = np.nan_to_num(X_test, nan=0.0)

            gb_model = XGBRegressor(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=SEED,
                n_jobs=1,
                early_stopping_rounds=30,
                verbosity=0,
            )
            gb_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            gb_results = {}
            for split_name, X, y in [("train", X_train, y_train),
                                      ("val", X_val, y_val),
                                      ("test", X_test, y_test)]:
                y_pred = gb_model.predict(X)
                gb_results[split_name] = compute_rate_metrics(y, y_pred)

            all_results["models"][variant] = gb_results
            all_results["models"][variant]["n_features"] = X_train.shape[1]
            all_results["models"][variant]["best_iteration"] = int(gb_model.best_iteration) if hasattr(gb_model, 'best_iteration') else -1
            logger.info("  Test: spearman=%.4f, mse=%.4f",
                        gb_results["test"]["spearman"], gb_results["test"]["mse"])

    except ImportError:
        logger.warning("xgboost not installed, skipping GB models")
        all_results["models"]["GB_HandFeatures"] = {"error": "xgboost not installed"}
        all_results["models"]["GB_AllFeatures"] = {"error": "xgboost not installed"}

    # ------------------------------------------------------------------
    # 5. PooledMLP
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("MODEL 5: PooledMLP")
    logger.info("=" * 70)

    pooled_model = PooledMLPRegressor(d_model=640, dropout=0.3)
    n_params = sum(p.numel() for p in pooled_model.parameters())
    logger.info("  Parameters: %s", f"{n_params:,}")

    pooled_results = train_nn_model(
        pooled_model, loaders, "PooledMLP",
        epochs=100, lr=1e-3, weight_decay=1e-3, patience=15,
    )
    all_results["models"]["PooledMLP"] = {
        k: v for k, v in pooled_results.items() if k in ("train", "val", "test")
    }
    all_results["models"]["PooledMLP"]["n_params"] = n_params

    # ------------------------------------------------------------------
    # 6. SubtractionMLP
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("MODEL 6: SubtractionMLP")
    logger.info("=" * 70)

    sub_model = SubtractionMLPRegressor(d_model=640, dropout=0.3)
    n_params = sum(p.numel() for p in sub_model.parameters())
    logger.info("  Parameters: %s", f"{n_params:,}")

    sub_results = train_nn_model(
        sub_model, loaders, "SubtractionMLP",
        epochs=100, lr=1e-3, weight_decay=1e-3, patience=15,
    )
    all_results["models"]["SubtractionMLP"] = {
        k: v for k, v in sub_results.items() if k in ("train", "val", "test")
    }
    all_results["models"]["SubtractionMLP"]["n_params"] = n_params

    # ------------------------------------------------------------------
    # 7. ConcatMLP
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("MODEL 7: ConcatMLP")
    logger.info("=" * 70)

    concat_model = ConcatMLPRegressor(d_model=640, dropout=0.3)
    n_params = sum(p.numel() for p in concat_model.parameters())
    logger.info("  Parameters: %s", f"{n_params:,}")

    concat_results = train_nn_model(
        concat_model, loaders, "ConcatMLP",
        epochs=100, lr=1e-3, weight_decay=1e-3, patience=15,
    )
    all_results["models"]["ConcatMLP"] = {
        k: v for k, v in concat_results.items() if k in ("train", "val", "test")
    }
    all_results["models"]["ConcatMLP"]["n_params"] = n_params

    # Save intermediate results before token-level models
    _save_results(all_results)
    logger.info("\nIntermediate results saved. Starting token-level models...")

    # ------------------------------------------------------------------
    # Load token-level embeddings for attention models
    # ------------------------------------------------------------------
    logger.info("\nLoading token-level embeddings for attention models...")
    _, _, tokens_orig, tokens_edited = load_embeddings(load_tokens=True)

    loaders_tok = create_rate_dataloaders(
        split_dfs, pooled_orig, pooled_edited, structure_delta,
        tokens_orig=tokens_orig, tokens_edited=tokens_edited,
        batch_size=32,
    )

    # ------------------------------------------------------------------
    # 8. CrossAttention
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("MODEL 8: CrossAttention_reg")
    logger.info("=" * 70)

    cross_model = CrossAttentionRegressor(
        d_model=640, n_heads=4, d_hidden=128, dropout=0.5,
    )
    n_params = sum(p.numel() for p in cross_model.parameters())
    logger.info("  Parameters: %s", f"{n_params:,}")

    cross_results = train_nn_model(
        cross_model, loaders_tok, "CrossAttention_reg",
        epochs=100, lr=1e-3, weight_decay=1e-2, patience=15,
    )
    all_results["models"]["CrossAttention_reg"] = {
        k: v for k, v in cross_results.items() if k in ("train", "val", "test")
    }
    all_results["models"]["CrossAttention_reg"]["n_params"] = n_params

    _save_results(all_results)

    # ------------------------------------------------------------------
    # 9. DiffAttention
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("MODEL 9: DiffAttention_reg")
    logger.info("=" * 70)

    diff_model = DiffAttentionRegressor(
        d_model=640, n_heads=4, d_hidden=128, dropout=0.5,
    )
    n_params = sum(p.numel() for p in diff_model.parameters())
    logger.info("  Parameters: %s", f"{n_params:,}")

    diff_results = train_nn_model(
        diff_model, loaders_tok, "DiffAttention_reg",
        epochs=100, lr=1e-3, weight_decay=1e-2, patience=15,
    )
    all_results["models"]["DiffAttention_reg"] = {
        k: v for k, v in diff_results.items() if k in ("train", "val", "test")
    }
    all_results["models"]["DiffAttention_reg"]["n_params"] = n_params

    _save_results(all_results)

    # ------------------------------------------------------------------
    # 10. EditRNA-A3A (Rate)
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("MODEL 10: EditRNA_rate")
    logger.info("=" * 70)

    try:
        editrna_model = build_editrna_rate_model(
            pooled_orig, pooled_edited, tokens_orig, tokens_edited,
        )
        n_params = sum(p.numel() for p in editrna_model.parameters() if p.requires_grad)
        logger.info("  Trainable parameters: %s", f"{n_params:,}")

        editrna_results = train_editrna_rate(
            editrna_model, loaders_tok,
            epochs=100, lr=1e-3, weight_decay=1e-2, patience=15,
            rate_weight=10.0,
        )
        all_results["models"]["EditRNA_rate"] = {
            k: v for k, v in editrna_results.items() if k in ("train", "val", "test")
        }
        all_results["models"]["EditRNA_rate"]["n_params"] = n_params

    except Exception as e:
        logger.error("EditRNA_rate FAILED: %s", e, exc_info=True)
        all_results["models"]["EditRNA_rate"] = {"error": str(e)}

    # ------------------------------------------------------------------
    # Save final results and print summary
    # ------------------------------------------------------------------
    _save_results(all_results)
    _print_summary(all_results)

    logger.info("\nResults saved to %s", OUTPUT_DIR / "rate_unified_results.json")
    return all_results


def _save_results(results: Dict):
    """Save results to JSON."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "rate_unified_results.json", "w") as f:
        json.dump(results, f, indent=2, default=_serialize)


def _print_summary(results: Dict):
    """Print summary table sorted by test Spearman."""
    print("\n" + "=" * 120)
    print("UNIFIED RATE PREDICTION - ALL ARCHITECTURES (normalized rates)")
    print(f"Target: {results['target_transform']}")
    print(f"Data: train={results['n_train']}, val={results['n_val']}, test={results['n_test']}")
    print("=" * 120)
    print(
        f"{'Model':<25} "
        f"{'Train Sp':>10} {'Val Sp':>10} {'Test Sp':>10} "
        f"{'Train Pe':>10} {'Val Pe':>10} {'Test Pe':>10} "
        f"{'Test MSE':>10} {'Test R2':>10}"
    )
    print("-" * 120)

    # Sort by test Spearman
    model_rows = []
    for model_name, model_data in results["models"].items():
        if "error" in model_data:
            model_rows.append((model_name, float("-inf"), model_data))
        else:
            test_sp = model_data.get("test", {}).get("spearman", float("nan"))
            model_rows.append((model_name, test_sp if not np.isnan(test_sp) else float("-inf"), model_data))

    model_rows.sort(key=lambda x: x[1], reverse=True)

    for model_name, _, model_data in model_rows:
        if "error" in model_data:
            print(f"{model_name:<25} FAILED: {model_data['error']}")
            continue

        tr = model_data.get("train", {})
        va = model_data.get("val", {})
        te = model_data.get("test", {})

        def fmt(v):
            return f"{v:>10.4f}" if v is not None and not np.isnan(v) else f"{'N/A':>10}"

        print(
            f"{model_name:<25} "
            f"{fmt(tr.get('spearman'))} {fmt(va.get('spearman'))} {fmt(te.get('spearman'))} "
            f"{fmt(tr.get('pearson'))} {fmt(va.get('pearson'))} {fmt(te.get('pearson'))} "
            f"{fmt(te.get('mse'))} {fmt(te.get('r2'))}"
        )

    print("=" * 120)


def _serialize(obj):
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


if __name__ == "__main__":
    run_experiment()
