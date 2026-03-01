#!/usr/bin/env python
"""Feature-Augmented DL Architecture Experiment ("EditRNA-A3A+").

Goal: Beat GB_AllFeatures (0.959 AUROC) with a DL model that combines
hand-crafted features with learned representations.

Hand features (33 dims, NOT including structure delta which is already in the model):
  - Motif features (24 dims): TC motif, upstream/downstream nucleotide one-hot,
    trinucleotide one-hot
  - Loop position features (9 dims): is_unpaired, loop_size, dist_to_junction,
    dist_to_apex, relative_loop_position, left/right stem length,
    max_adjacent_stem_length, local_unpaired_fraction

Integration strategies:
  A. Modality-level fusion: inject hand_features via d_gnn slot in GatedModalityFusion
  B. Edit embedding level: concat hand_features with edit embedding input
  C. Both A + B combined

Models compared:
  1. SubtractionMLP vs SubtractionMLP+Features
  2. DiffAttention vs DiffAttention+Features
  3. EditRNA vs EditRNA+Features (Strategy A)
  4. EditRNA+Features (Strategy B) - ablation
  5. GB_HandFeatures and GB_AllFeatures as reference

Ablations:
  - Feature group: motif-only vs loop-only vs both

Usage:
    python experiments/apobec/exp_feature_augmented.py
"""

import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F_torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.apobec.train_baselines import (
    BaselineConfig,
    EmbeddingDataset,
    FocalLoss,
    build_model,
    compute_metrics,
    embedding_collate_fn,
    _forward_baseline,
    load_data,
    create_dataloaders,
)
from experiments.apobec.train_gradient_boosting import (
    extract_motif_features,
    extract_loop_position_features,
    extract_structure_delta_features,
    extract_embedding_delta_features,
    get_classifier,
    compute_binary_metrics,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMB_DIR = PROJECT_ROOT / "data" / "processed" / "embeddings"
SPLITS_CSV = PROJECT_ROOT / "data" / "processed" / "splits_expanded.csv"
STRUCT_CACHE = EMB_DIR / "vienna_structure_cache.npz"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "apobec" / "outputs" / "feature_augmented"

SEED = 42
D_MODEL = 640


# ---------------------------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------------------------

def extract_hand_features(splits_df: pd.DataFrame) -> Tuple[Dict[str, np.ndarray], List[str], List[str]]:
    """Extract 33-dim hand-crafted features for each site.

    Returns:
        feature_dict: {site_id: np.ndarray(33,)}
        motif_cols: list of motif feature column names
        loop_cols: list of loop feature column names
    """
    motif_df = extract_motif_features(splits_df)
    loop_df = extract_loop_position_features(splits_df)

    motif_cols = [c for c in motif_df.columns if c != "site_id"]
    loop_cols = [c for c in loop_df.columns if c != "site_id"]

    logger.info("Hand features: %d motif + %d loop = %d total",
                len(motif_cols), len(loop_cols), len(motif_cols) + len(loop_cols))

    # Index DataFrames by site_id for O(1) lookup
    motif_indexed = motif_df.set_index("site_id")
    loop_indexed = loop_df.set_index("site_id")

    motif_zeros = [0.0] * len(motif_cols)
    loop_zeros = [0.0] * len(loop_cols)

    feature_dict = {}
    for _, row in splits_df.iterrows():
        sid = str(row["site_id"])

        if sid in motif_indexed.index:
            motif_row = motif_indexed.loc[sid]
            motif_vals = [float(motif_row.get(c, 0.0)) for c in motif_cols]
        else:
            motif_vals = motif_zeros

        if sid in loop_indexed.index:
            loop_row = loop_indexed.loc[sid]
            loop_vals = [float(loop_row.get(c, 0.0)) for c in loop_cols]
        else:
            loop_vals = loop_zeros

        feature_dict[sid] = np.array(motif_vals + loop_vals, dtype=np.float32)

    return feature_dict, motif_cols, loop_cols


# ---------------------------------------------------------------------------
# Augmented Dataset
# ---------------------------------------------------------------------------

class AugmentedEmbeddingDataset(Dataset):
    """EmbeddingDataset extended with hand-crafted features tensor."""

    def __init__(
        self,
        site_ids: List[str],
        labels: np.ndarray,
        pooled_orig: Dict[str, torch.Tensor],
        pooled_edited: Dict[str, torch.Tensor],
        tokens_orig: Optional[Dict[str, torch.Tensor]] = None,
        tokens_edited: Optional[Dict[str, torch.Tensor]] = None,
        structure_delta: Optional[Dict[str, np.ndarray]] = None,
        hand_features: Optional[Dict[str, np.ndarray]] = None,
        feature_mask: Optional[np.ndarray] = None,
        d_hand: int = 33,
    ):
        self.site_ids = site_ids
        self.labels = labels
        self.pooled_orig = pooled_orig
        self.pooled_edited = pooled_edited
        self.tokens_orig = tokens_orig
        self.tokens_edited = tokens_edited
        self.structure_delta = structure_delta
        self.hand_features = hand_features
        self.feature_mask = feature_mask  # boolean mask for feature ablation
        self.d_hand = d_hand

    def __len__(self) -> int:
        return len(self.site_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
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

        if self.hand_features is not None and sid in self.hand_features:
            hf = self.hand_features[sid].copy()
            if self.feature_mask is not None:
                hf = hf * self.feature_mask
            item["hand_features"] = torch.tensor(hf, dtype=torch.float32)
        else:
            item["hand_features"] = torch.zeros(self.d_hand, dtype=torch.float32)

        return item


def augmented_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for AugmentedEmbeddingDataset."""
    result = {
        "site_ids": [b["site_id"] for b in batch],
        "labels": torch.stack([b["label"] for b in batch]),
        "pooled_orig": torch.stack([b["pooled_orig"] for b in batch]),
        "pooled_edited": torch.stack([b["pooled_edited"] for b in batch]),
        "structure_delta": torch.stack([b["structure_delta"] for b in batch]),
        "hand_features": torch.stack([b["hand_features"] for b in batch]),
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
# Feature-Augmented Models
# ---------------------------------------------------------------------------

class FeatureAugmentedSubtractionMLP(nn.Module):
    """SubtractionMLP with hand-crafted features concatenated before MLP."""

    def __init__(self, d_model: int = 640, d_hand: int = 33, dropout: float = 0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model + d_hand, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        diff = batch["pooled_edited"] - batch["pooled_orig"]
        hand = batch["hand_features"]
        combined = torch.cat([diff, hand], dim=-1)
        logit = self.mlp(combined)
        return {"binary_logit": logit}


class FeatureAugmentedDiffAttention(nn.Module):
    """DiffAttention with hand features concatenated after pooling."""

    def __init__(self, d_model: int = 640, d_hand: int = 33,
                 n_heads: int = 8, n_layers: int = 2, dropout: float = 0.3):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        self.mlp = nn.Sequential(
            nn.Linear(d_model + d_hand, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        diff = batch["tokens_edited"] - batch["tokens_orig"]
        encoded = self.transformer(diff)
        pooled = encoded.mean(dim=1)
        hand = batch["hand_features"]
        combined = torch.cat([pooled, hand], dim=-1)
        logit = self.mlp(combined)
        return {"binary_logit": logit}


class FeatureAugmentedEditRNA(nn.Module):
    """EditRNA-A3A with hand features injected via GatedModalityFusion's d_gnn slot.

    Strategy A: Hand features are treated as a third modality alongside
    primary (RNA-FM pooled) and edit (APOBEC edit embedding).
    """

    def __init__(self, d_model: int = 640, d_hand: int = 33,
                 d_edit: int = 256, d_fused: int = 512,
                 pooled_only: bool = True, cached_encoder=None):
        super().__init__()

        from models.editrna_a3a import EditRNA_A3A, EditRNAConfig
        from models.fusion import GatedModalityFusion
        from models.apobec_edit_embedding import APOBECEditEmbedding

        config = EditRNAConfig(
            primary_encoder="cached",
            d_model=d_model,
            d_edit=d_edit,
            d_fused=d_fused,
            pooled_only=pooled_only,
            use_gnn=False,
        )

        # Build the base EditRNA model
        self.base_model = EditRNA_A3A(config=config, primary_encoder=cached_encoder)

        # Replace the fusion module with one that has d_gnn=d_hand
        self.base_model.fusion = GatedModalityFusion(
            d_model=d_model,
            d_edit=d_edit,
            d_fused=d_fused,
            d_model_secondary=0,
            d_gnn=d_hand,
            dropout=config.fusion_dropout,
        )

        self.d_hand = d_hand

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        """Forward with hand features injected as gnn_emb in fusion."""
        sequences = batch["sequences"]
        edit_pos = batch["edit_pos"]
        flanking_context = batch["flanking_context"]
        concordance_features = batch["concordance_features"]
        structure_delta = batch["structure_delta"]
        hand_features = batch["hand_features"]
        site_ids = batch.get("site_ids")
        device = edit_pos.device

        base = self.base_model
        cfg = base.config

        # Encode original
        primary_out = base._encode_primary(sequences, site_ids=site_ids, edited=False)
        primary_pooled = primary_out["pooled"]

        if cfg.pooled_only:
            f_background = primary_pooled.unsqueeze(1)
            edited_sequences = base._make_edited_sequences(sequences, edit_pos)
            edited_out = base._encode_primary(edited_sequences, site_ids=site_ids, edited=True)
            f_edited = edited_out["tokens"][:, :1, :] if edited_out["tokens"].dim() == 3 else edited_out["pooled"].unsqueeze(1)
            edit_pos_adj = torch.zeros_like(edit_pos)
            seq_mask = torch.ones(f_background.shape[0], 1, dtype=torch.bool, device=device)
        else:
            f_background = primary_out["tokens"]
            edited_sequences = base._make_edited_sequences(sequences, edit_pos)
            edited_out = base._encode_primary(edited_sequences, site_ids=site_ids, edited=True)
            f_edited = edited_out["tokens"]
            min_len = min(f_background.shape[1], f_edited.shape[1])
            f_background = f_background[:, :min_len, :]
            f_edited = f_edited[:, :min_len, :]
            edit_pos_adj = edit_pos
            seq_mask = torch.ones(f_background.shape[0], min_len, dtype=torch.bool, device=device)

        # Edit embedding
        edit_emb = base.edit_embedding(
            f_background=f_background,
            f_edited=f_edited,
            edit_pos=edit_pos_adj if cfg.pooled_only else edit_pos,
            flanking_context=flanking_context,
            structure_delta=structure_delta,
            concordance_features=concordance_features,
            seq_mask=seq_mask,
        )

        # Fusion with hand features as gnn_emb
        fused = base.fusion(
            primary_pooled=primary_pooled,
            edit_emb=edit_emb,
            gnn_emb=hand_features,
        )

        # Prediction heads
        predictions = base.heads(fused, edit_emb)

        return {
            "predictions": predictions,
            "edit_embedding": edit_emb,
            "fused": fused,
        }


# ---------------------------------------------------------------------------
# Forward helpers
# ---------------------------------------------------------------------------

def _forward_augmented(model, batch, model_name):
    """Forward pass dispatcher for augmented and baseline models.

    Handles three cases:
    1. editrna_features_*: FeatureAugmentedEditRNA — needs adapted batch with hand_features
    2. editrna: Baseline EditRNA_A3A — needs adapted batch (same as _forward_baseline)
    3. Everything else: Direct model(batch) — SubtractionMLP, DiffAttention, etc.
    """
    if model_name.startswith("editrna_features"):
        # FeatureAugmentedEditRNA: needs EditRNA-format batch + hand_features
        B = batch["labels"].shape[0]
        device = batch["labels"].device
        editrna_batch = {
            "sequences": ["N" * 201] * B,
            "site_ids": batch["site_ids"],
            "edit_pos": torch.full((B,), 100, dtype=torch.long, device=device),
            "flanking_context": torch.zeros(B, dtype=torch.long, device=device),
            "concordance_features": torch.zeros(B, 5, device=device),
            "structure_delta": batch["structure_delta"],
            "hand_features": batch["hand_features"],
        }
        output = model(editrna_batch)
        return {"binary_logit": output["predictions"]["binary_logit"]}

    elif model_name == "editrna":
        # Baseline EditRNA_A3A with CachedRNAEncoder: needs adapted batch
        B = batch["labels"].shape[0]
        device = batch["labels"].device
        editrna_batch = {
            "sequences": ["N" * 201] * B,
            "site_ids": batch["site_ids"],
            "edit_pos": torch.full((B,), 100, dtype=torch.long, device=device),
            "flanking_context": torch.zeros(B, dtype=torch.long, device=device),
            "concordance_features": torch.zeros(B, 5, device=device),
            "structure_delta": batch["structure_delta"],
        }
        output = model(editrna_batch)
        return {"binary_logit": output["predictions"]["binary_logit"]}

    else:
        # SubtractionMLP, DiffAttention, PooledMLP, etc. — read from batch directly
        return model(batch)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_and_evaluate_augmented(
    model, model_name, train_loader, val_loader, test_loader,
    epochs=50, lr=1e-3, patience=10, seed=42,
):
    """Train an augmented model and return test metrics."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cpu")
    model = model.to(device)

    loss_fn = FocalLoss(gamma=2.0, alpha=0.75)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)

    best_auroc = -1.0
    patience_counter = 0
    best_state = None
    best_epoch = 0

    t_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            optimizer.zero_grad()
            output = _forward_augmented(model, batch, model_name)
            logits = output["binary_logit"]
            loss = loss_fn(logits, batch["labels"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Validate
        val_metrics = _evaluate_augmented(model, val_loader, model_name, device)
        val_auroc = val_metrics.get("auroc", 0.0)

        if not np.isnan(val_auroc) and val_auroc > best_auroc + 1e-4:
            best_auroc = val_auroc
            best_epoch = epoch
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            logger.info("  Epoch %3d  loss=%.4f  val_auroc=%.4f  patience=%d/%d",
                        epoch, train_loss / max(n_batches, 1),
                        val_auroc, patience_counter, patience)

        if patience_counter >= patience:
            logger.info("  Early stopping at epoch %d (best=%d, AUROC=%.4f)",
                        epoch, best_epoch, best_auroc)
            break

    if best_state:
        model.load_state_dict(best_state)

    elapsed = time.time() - t_start

    # Test
    test_metrics = _evaluate_augmented(model, test_loader, model_name, device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "test_metrics": test_metrics,
        "val_metrics": val_metrics,
        "best_epoch": best_epoch,
        "best_val_auroc": best_auroc,
        "train_time_seconds": elapsed,
        "n_params": n_params,
    }


@torch.no_grad()
def _evaluate_augmented(model, loader, model_name, device):
    """Evaluate an augmented model."""
    if loader is None:
        return {}

    model.eval()
    all_targets = []
    all_scores = []

    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        output = _forward_augmented(model, batch, model_name)
        logits = output["binary_logit"].squeeze(-1).cpu().numpy()
        probs = 1.0 / (1.0 + np.exp(-logits))
        all_targets.append(batch["labels"].cpu().numpy())
        all_scores.append(probs)

    y_true = np.concatenate(all_targets)
    y_score = np.concatenate(all_scores)
    return compute_metrics(y_true, y_score)


# ---------------------------------------------------------------------------
# GB Reference Models
# ---------------------------------------------------------------------------

def train_gb_reference(splits_df, variant="gb_all"):
    """Train a GB model as reference and return test metrics."""
    motif_df = extract_motif_features(splits_df)
    struct_df = extract_structure_delta_features(splits_df)
    loop_df = extract_loop_position_features(splits_df)

    motif_cols = [c for c in motif_df.columns if c != "site_id"]
    struct_cols = [c for c in struct_df.columns if c != "site_id"]
    loop_cols = [c for c in loop_df.columns if c != "site_id"]
    hand_cols = motif_cols + struct_cols + loop_cols

    if variant == "gb_all":
        emb_df = extract_embedding_delta_features(splits_df)
        emb_cols = [c for c in emb_df.columns if c != "site_id"]
        feature_cols = hand_cols + emb_cols
    else:
        emb_df = None
        feature_cols = hand_cols

    # Build feature matrix aligned with splits_df
    X_motif = motif_df[[c for c in motif_cols]].values
    X_struct = struct_df[[c for c in struct_cols]].values
    X_loop = loop_df[[c for c in loop_cols]].values

    if emb_df is not None:
        X_emb = emb_df[[c for c in emb_cols]].values
        X = np.concatenate([X_motif, X_struct, X_loop, X_emb], axis=1)
    else:
        X = np.concatenate([X_motif, X_struct, X_loop], axis=1)

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = splits_df["label"].values
    splits = splits_df["split"].values

    train_mask = splits == "train"
    val_mask = splits == "val"
    test_mask = splits == "test"

    # Class imbalance
    n_pos = int(y[train_mask].sum())
    n_neg = int(train_mask.sum() - n_pos)
    scale_pos_weight = n_neg / max(n_pos, 1)

    clf, backend = get_classifier(len(feature_cols))
    if backend in ("xgboost", "lightgbm"):
        clf.set_params(scale_pos_weight=scale_pos_weight)

    t_start = time.time()
    if backend == "sklearn":
        sample_weights = np.where(y[train_mask] == 1, scale_pos_weight, 1.0)
        clf.fit(X[train_mask], y[train_mask], sample_weight=sample_weights)
    elif backend == "xgboost":
        clf.fit(X[train_mask], y[train_mask],
                eval_set=[(X[val_mask], y[val_mask])], verbose=False)
    else:
        clf.fit(X[train_mask], y[train_mask])
    elapsed = time.time() - t_start

    test_proba = clf.predict_proba(X[test_mask])[:, 1]
    test_metrics = compute_binary_metrics(y[test_mask], test_proba)
    test_metrics["n_positive"] = int(y[test_mask].sum())
    test_metrics["n_negative"] = int(test_mask.sum() - y[test_mask].sum())

    return {
        "test_metrics": test_metrics,
        "train_time_seconds": elapsed,
        "n_features": len(feature_cols),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def serialize(obj):
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

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # ===================================================================
    # Load data
    # ===================================================================
    logger.info("Loading data...")
    config = BaselineConfig(neg_ratio=5, seed=SEED)
    (pooled_orig, pooled_edited, tokens_orig, tokens_edited,
     structure_delta, splits_df) = load_data(config, load_tokens=True)

    # Extract hand features
    hand_features, motif_cols, loop_cols = extract_hand_features(splits_df)
    d_motif = len(motif_cols)
    d_loop = len(loop_cols)
    d_hand = d_motif + d_loop

    logger.info("Hand features: %d dims (motif=%d, loop=%d)",
                d_hand, d_motif, d_loop)

    # Create feature masks for ablation
    motif_mask = np.array([1.0] * d_motif + [0.0] * d_loop, dtype=np.float32)
    loop_mask = np.array([0.0] * d_motif + [1.0] * d_loop, dtype=np.float32)
    both_mask = np.ones(d_hand, dtype=np.float32)

    # ===================================================================
    # Create DataLoaders
    # ===================================================================
    available_ids = set(pooled_orig.keys()) & set(pooled_edited.keys())
    df = splits_df[splits_df["site_id"].isin(available_ids)].copy()

    # Apply negative ratio
    pos_df = df[df["label"] == 1]
    neg_df = df[df["label"] == 0]
    max_neg = len(pos_df) * config.neg_ratio
    if len(neg_df) > max_neg:
        neg_df = neg_df.sample(n=max_neg, random_state=SEED)
    df = pd.concat([pos_df, neg_df], ignore_index=True)

    def make_loaders(feature_mask=None, needs_tokens=False):
        """Create train/val/test loaders with optional feature masking."""
        loaders = {}
        for split_name in ["train", "val", "test"]:
            subset = df[df["split"] == split_name]
            if len(subset) == 0:
                continue

            ds = AugmentedEmbeddingDataset(
                site_ids=subset["site_id"].tolist(),
                labels=subset["label"].values.astype(np.float32),
                pooled_orig=pooled_orig,
                pooled_edited=pooled_edited,
                tokens_orig=tokens_orig if needs_tokens else None,
                tokens_edited=tokens_edited if needs_tokens else None,
                structure_delta=structure_delta,
                hand_features=hand_features,
                feature_mask=feature_mask,
                d_hand=d_hand,
            )

            loaders[split_name] = DataLoader(
                ds, batch_size=64, shuffle=(split_name == "train"),
                num_workers=0, collate_fn=augmented_collate_fn,
            )
        return loaders

    # ===================================================================
    # Experiment 1: Baseline DL models (without features)
    # ===================================================================
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 1: Baseline DL models (no hand features)")
    logger.info("=" * 70)

    all_results = {}

    # SubtractionMLP baseline
    logger.info("\n--- SubtractionMLP (baseline) ---")
    config_sub = BaselineConfig(model_name="subtraction_mlp", seed=SEED)
    sub_model = build_model("subtraction_mlp", config_sub)
    loaders = make_loaders(feature_mask=None, needs_tokens=False)
    r = train_and_evaluate_augmented(sub_model, "subtraction_mlp",
                                      loaders["train"], loaders.get("val"),
                                      loaders["test"])
    all_results["SubtractionMLP"] = r
    logger.info("  Test AUROC=%.4f", r["test_metrics"].get("auroc", 0))

    # DiffAttention baseline
    logger.info("\n--- DiffAttention (baseline) ---")
    config_diff = BaselineConfig(model_name="diff_attention", seed=SEED)
    diff_model = build_model("diff_attention", config_diff)
    loaders_tok = make_loaders(feature_mask=None, needs_tokens=True)
    r = train_and_evaluate_augmented(diff_model, "diff_attention",
                                      loaders_tok["train"], loaders_tok.get("val"),
                                      loaders_tok["test"])
    all_results["DiffAttention"] = r
    logger.info("  Test AUROC=%.4f", r["test_metrics"].get("auroc", 0))

    # EditRNA baseline
    logger.info("\n--- EditRNA (baseline) ---")
    config_editrna = BaselineConfig(model_name="editrna", seed=SEED)
    editrna_model = build_model("editrna", config_editrna)
    # Need to use _forward_baseline for editrna
    loaders_editrna = make_loaders(feature_mask=None, needs_tokens=False)
    r = train_and_evaluate_augmented(editrna_model, "editrna",
                                      loaders_editrna["train"], loaders_editrna.get("val"),
                                      loaders_editrna["test"])
    all_results["EditRNA"] = r
    logger.info("  Test AUROC=%.4f", r["test_metrics"].get("auroc", 0))

    # ===================================================================
    # Experiment 2: Feature-augmented DL models (all features)
    # ===================================================================
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 2: Feature-augmented DL models (+33 hand features)")
    logger.info("=" * 70)

    # SubtractionMLP + Features
    logger.info("\n--- SubtractionMLP + Features ---")
    sub_aug = FeatureAugmentedSubtractionMLP(d_model=D_MODEL, d_hand=d_hand)
    loaders = make_loaders(feature_mask=both_mask, needs_tokens=False)
    r = train_and_evaluate_augmented(sub_aug, "subtraction_features",
                                      loaders["train"], loaders.get("val"),
                                      loaders["test"])
    all_results["SubtractionMLP+Features"] = r
    logger.info("  Test AUROC=%.4f", r["test_metrics"].get("auroc", 0))

    # DiffAttention + Features
    logger.info("\n--- DiffAttention + Features ---")
    diff_aug = FeatureAugmentedDiffAttention(d_model=D_MODEL, d_hand=d_hand)
    loaders_tok = make_loaders(feature_mask=both_mask, needs_tokens=True)
    r = train_and_evaluate_augmented(diff_aug, "diff_features",
                                      loaders_tok["train"], loaders_tok.get("val"),
                                      loaders_tok["test"])
    all_results["DiffAttention+Features"] = r
    logger.info("  Test AUROC=%.4f", r["test_metrics"].get("auroc", 0))

    # EditRNA + Features (Strategy A: modality fusion)
    logger.info("\n--- EditRNA + Features (Strategy A: modality fusion) ---")
    cached_encoder = editrna_model.primary_encoder if hasattr(editrna_model, 'primary_encoder') else None
    if cached_encoder is None:
        # Rebuild cached encoder
        from models.encoders import CachedRNAEncoder
        cached_encoder = CachedRNAEncoder(
            tokens_cache=tokens_orig,
            pooled_cache=pooled_orig,
            tokens_edited_cache=tokens_edited,
            pooled_edited_cache=pooled_edited,
            d_model=D_MODEL,
        )

    editrna_aug = FeatureAugmentedEditRNA(
        d_model=D_MODEL, d_hand=d_hand,
        d_edit=256, d_fused=512,
        pooled_only=True,
        cached_encoder=cached_encoder,
    )
    loaders = make_loaders(feature_mask=both_mask, needs_tokens=False)
    r = train_and_evaluate_augmented(editrna_aug, "editrna_features_a",
                                      loaders["train"], loaders.get("val"),
                                      loaders["test"])
    all_results["EditRNA+Features_A"] = r
    logger.info("  Test AUROC=%.4f", r["test_metrics"].get("auroc", 0))

    # ===================================================================
    # Experiment 3: Feature ablation (motif-only vs loop-only)
    # ===================================================================
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 3: Feature ablation")
    logger.info("=" * 70)

    # SubtractionMLP + Motif only
    logger.info("\n--- SubtractionMLP + Motif Only (%d dims) ---", d_motif)
    sub_motif = FeatureAugmentedSubtractionMLP(d_model=D_MODEL, d_hand=d_hand)
    loaders = make_loaders(feature_mask=motif_mask, needs_tokens=False)
    r = train_and_evaluate_augmented(sub_motif, "sub_motif",
                                      loaders["train"], loaders.get("val"),
                                      loaders["test"])
    all_results["SubtractionMLP+Motif"] = r
    logger.info("  Test AUROC=%.4f", r["test_metrics"].get("auroc", 0))

    # SubtractionMLP + Loop only
    logger.info("\n--- SubtractionMLP + Loop Only (%d dims) ---", d_loop)
    sub_loop = FeatureAugmentedSubtractionMLP(d_model=D_MODEL, d_hand=d_hand)
    loaders = make_loaders(feature_mask=loop_mask, needs_tokens=False)
    r = train_and_evaluate_augmented(sub_loop, "sub_loop",
                                      loaders["train"], loaders.get("val"),
                                      loaders["test"])
    all_results["SubtractionMLP+Loop"] = r
    logger.info("  Test AUROC=%.4f", r["test_metrics"].get("auroc", 0))

    # EditRNA + Motif only
    logger.info("\n--- EditRNA + Motif Only ---")
    editrna_motif = FeatureAugmentedEditRNA(
        d_model=D_MODEL, d_hand=d_hand,
        d_edit=256, d_fused=512,
        pooled_only=True,
        cached_encoder=cached_encoder,
    )
    loaders = make_loaders(feature_mask=motif_mask, needs_tokens=False)
    r = train_and_evaluate_augmented(editrna_motif, "editrna_features_a",
                                      loaders["train"], loaders.get("val"),
                                      loaders["test"])
    all_results["EditRNA+Motif"] = r
    logger.info("  Test AUROC=%.4f", r["test_metrics"].get("auroc", 0))

    # EditRNA + Loop only
    logger.info("\n--- EditRNA + Loop Only ---")
    editrna_loop = FeatureAugmentedEditRNA(
        d_model=D_MODEL, d_hand=d_hand,
        d_edit=256, d_fused=512,
        pooled_only=True,
        cached_encoder=cached_encoder,
    )
    loaders = make_loaders(feature_mask=loop_mask, needs_tokens=False)
    r = train_and_evaluate_augmented(editrna_loop, "editrna_features_a",
                                      loaders["train"], loaders.get("val"),
                                      loaders["test"])
    all_results["EditRNA+Loop"] = r
    logger.info("  Test AUROC=%.4f", r["test_metrics"].get("auroc", 0))

    # ===================================================================
    # Experiment 4: GB Reference Models
    # ===================================================================
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 4: Gradient Boosting reference models")
    logger.info("=" * 70)

    logger.info("\n--- GB_HandFeatures ---")
    r = train_gb_reference(splits_df, variant="gb_hand")
    all_results["GB_HandFeatures"] = r
    logger.info("  Test AUROC=%.4f", r["test_metrics"].get("auroc", 0))

    logger.info("\n--- GB_AllFeatures ---")
    r = train_gb_reference(splits_df, variant="gb_all")
    all_results["GB_AllFeatures"] = r
    logger.info("  Test AUROC=%.4f", r["test_metrics"].get("auroc", 0))

    # ===================================================================
    # Summary
    # ===================================================================
    print("\n" + "=" * 110)
    print("FEATURE-AUGMENTED ARCHITECTURE COMPARISON")
    print("=" * 110)
    print(f"{'Model':<30} {'AUROC':>8} {'AUPRC':>8} {'F1':>8} {'Prec':>8} "
          f"{'Recall':>8} {'Params':>10} {'Time(s)':>8}")
    print("-" * 110)

    # Sort by AUROC
    sorted_results = sorted(
        all_results.items(),
        key=lambda x: x[1].get("test_metrics", {}).get("auroc", 0),
        reverse=True,
    )

    for name, r in sorted_results:
        tm = r.get("test_metrics", {})
        n_params = r.get("n_params", r.get("n_features", "N/A"))
        elapsed = r.get("train_time_seconds", 0)
        print(f"{name:<30} "
              f"{tm.get('auroc', 0):>8.4f} "
              f"{tm.get('auprc', 0):>8.4f} "
              f"{tm.get('f1', 0):>8.4f} "
              f"{tm.get('precision', 0):>8.4f} "
              f"{tm.get('recall', 0):>8.4f} "
              f"{n_params if isinstance(n_params, str) else f'{n_params:>10,}'} "
              f"{elapsed:>8.1f}")

    print("=" * 110)

    # Determine best model
    best_name = sorted_results[0][0]
    best_auroc = sorted_results[0][1].get("test_metrics", {}).get("auroc", 0)
    print(f"\nBest model: {best_name} (AUROC={best_auroc:.4f})")

    gb_auroc = all_results.get("GB_AllFeatures", {}).get("test_metrics", {}).get("auroc", 0)
    if best_auroc > gb_auroc:
        print(f"  -> Beats GB_AllFeatures ({gb_auroc:.4f}) by +{best_auroc - gb_auroc:.4f}")
    else:
        print(f"  -> Does NOT beat GB_AllFeatures ({gb_auroc:.4f}), gap: {gb_auroc - best_auroc:.4f}")

    # ===================================================================
    # Save Results
    # ===================================================================
    output = {
        "experiment": "feature_augmented",
        "d_hand": d_hand,
        "d_motif": d_motif,
        "d_loop": d_loop,
        "motif_features": motif_cols,
        "loop_features": loop_cols,
        "best_model": best_name,
        "results": {},
    }

    for name, r in all_results.items():
        output["results"][name] = {
            "test_metrics": r.get("test_metrics", {}),
            "val_metrics": r.get("val_metrics", {}),
            "best_epoch": r.get("best_epoch"),
            "best_val_auroc": r.get("best_val_auroc"),
            "train_time_seconds": r.get("train_time_seconds"),
            "n_params": r.get("n_params", r.get("n_features")),
        }

    results_path = OUTPUT_DIR / "feature_augmented_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=serialize)

    logger.info("\nResults saved to %s", results_path)
    return output


if __name__ == "__main__":
    main()
