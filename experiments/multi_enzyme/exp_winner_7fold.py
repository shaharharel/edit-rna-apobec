#!/usr/bin/env python
"""7-fold CV for top 5 winners from the architecture screen.

Runs 7-fold StratifiedKFold CV (seed=42) on the 5 best combinations from Phase 1.
Waits for PID 42342 (the winner screen) to finish before starting.

Top 5 winners (Phase 1 3-fold):
  1. A8+T1+H4 (mean PE AUROC=0.9066)
  2. A8+T6+H4 (0.9046)
  3. A8+T6+H1 (0.9046)
  4. A8+T1+H1 (0.9022)
  5. A8+T4+H1 (0.9013)

Output: experiments/multi_enzyme/outputs/architecture_screen/winner_7fold_results.json
Log:    experiments/multi_enzyme/outputs/architecture_screen/winner_7fold.log
Run:    nohup /opt/miniconda3/envs/quris/bin/python -u experiments/multi_enzyme/exp_winner_7fold.py \
        > experiments/multi_enzyme/outputs/architecture_screen/winner_7fold.log 2>&1 &
"""

import gc
import json
import logging
import math
import os
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.apobec_feature_extraction import build_hand_features

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "multi_enzyme" / "outputs" / "architecture_screen"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_PATH = OUTPUT_DIR / "winner_7fold_results.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

DEVICE = (
    torch.device("mps")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else torch.device("cpu")
)
logger.info(f"Using device: {DEVICE}")

ENZYME_CLASSES = ["A3A", "A3B", "A3G", "A3A_A3G", "Neither", "Unknown"]
ENZYME_TO_IDX = {e: i for i, e in enumerate(ENZYME_CLASSES)}
N_ENZYMES = len(ENZYME_CLASSES)
CENTER = 100
SEED = 42
N_FOLDS = 7

PER_ENZYME_HEADS = ["A3A", "A3B", "A3G", "A3A_A3G", "Neither"]
N_PER_ENZYME = len(PER_ENZYME_HEADS)

BATCH_SIZE = 64
BATCH_SIZE_LARGE = 256

# Stage 1 config (T1 baseline)
STAGE1_EPOCHS = 10
STAGE1_LR = 1e-3
STAGE1_WEIGHT_DECAY = 1e-4
STAGE1_ENZYME_HEAD_WEIGHT = 0.3
STAGE1_ENZYME_CLS_WEIGHT = 0.1

# Stage 2 config
STAGE2_EPOCHS = 20
STAGE2_LR = 5e-4
STAGE2_WEIGHT_DECAY = 1e-4

# Reduced epochs for T4 (v4-large)
STAGE1_EPOCHS_LARGE = 5
STAGE2_EPOCHS_LARGE = 10

# Pretext pretraining epochs (T6)
PRETEXT_EPOCHS = 5

# Feature dimensions
D_RNAFM = 640
D_EDIT_DELTA = 640
D_HAND = 40
D_SHARED = 128


# ---------------------------------------------------------------------------
# Head Mixins
# ---------------------------------------------------------------------------


class H1Mixin:
    """H1: Binary head on all data + 5 per-enzyme adapters."""

    def _init_heads(self):
        self.binary_head = nn.Linear(D_SHARED, 1)
        self.enzyme_adapters = nn.ModuleDict({
            enz: nn.Sequential(
                nn.Linear(D_SHARED, 32),
                nn.GELU(),
                nn.Linear(32, 1),
            )
            for enz in PER_ENZYME_HEADS
        })
        self.enzyme_classifier = nn.Sequential(
            nn.Linear(D_SHARED, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, N_ENZYMES),
        )

    def _apply_heads(self, shared):
        binary_logit = self.binary_head(shared).squeeze(-1)
        per_enzyme_logits = [
            self.enzyme_adapters[enz](shared).squeeze(-1)
            for enz in PER_ENZYME_HEADS
        ]
        enzyme_cls_logits = self.enzyme_classifier(shared)
        return binary_logit, per_enzyme_logits, enzyme_cls_logits


class H4Mixin:
    """H4: Shared encoder -> 128dim, private encoders per enzyme -> concat -> adapter."""

    def _init_heads(self):
        self.binary_head = nn.Linear(D_SHARED, 1)
        self.enzyme_classifier = nn.Sequential(
            nn.Linear(D_SHARED, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, N_ENZYMES),
        )
        self.private_encoders = nn.ModuleDict({
            enz: nn.Sequential(
                nn.Linear(D_SHARED, 32),
                nn.GELU(),
                nn.Dropout(0.1),
            )
            for enz in PER_ENZYME_HEADS
        })
        self.enzyme_adapters = nn.ModuleDict({
            enz: nn.Linear(D_SHARED + 32, 1)
            for enz in PER_ENZYME_HEADS
        })

    def _apply_heads(self, shared):
        binary_logit = self.binary_head(shared).squeeze(-1)
        per_enzyme_logits = []
        for enz in PER_ENZYME_HEADS:
            private = self.private_encoders[enz](shared)
            combined = torch.cat([shared, private], dim=-1)
            per_enzyme_logits.append(self.enzyme_adapters[enz](combined).squeeze(-1))
        enzyme_cls_logits = self.enzyme_classifier(shared)
        return binary_logit, per_enzyme_logits, enzyme_cls_logits


# ---------------------------------------------------------------------------
# Architecture: A8 (HierarchicalAttention)
# ---------------------------------------------------------------------------


def _make_a8_encoder():
    """Build A8 encoder components."""
    d_local = 41
    local_proj = nn.Linear(d_local, 64)
    local_pos_enc = nn.Parameter(torch.randn(41, 64) * 0.02)
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=64, nhead=4, dim_feedforward=128,
        dropout=0.1, activation="gelu", batch_first=True,
    )
    local_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
    local_pool_attn = nn.Linear(64, 1)
    cross_q = nn.Linear(64, 64)
    cross_k = nn.Linear(D_RNAFM, 64)
    cross_v = nn.Linear(D_RNAFM, 64)
    d_fused = 64 + 64 + D_RNAFM + D_HAND
    encoder = nn.Sequential(
        nn.Linear(d_fused, 256), nn.GELU(), nn.Dropout(0.3), nn.LayerNorm(256),
        nn.Linear(256, D_SHARED), nn.GELU(), nn.Dropout(0.2),
    )
    return local_proj, local_pos_enc, local_transformer, local_pool_attn, cross_q, cross_k, cross_v, encoder


def _a8_encode(model, batch):
    """Shared encoding logic for A8."""
    bp = batch["bp_submatrix"].squeeze(1)  # [B, 41, 41]
    rnafm = batch["rnafm"]
    hand = batch["hand_feat"]
    local_in = model.local_proj(bp) + model.local_pos_enc.unsqueeze(0)
    local_out = model.local_transformer(local_in)
    attn_w = torch.softmax(model.local_pool_attn(local_out), dim=1)
    local_repr = (local_out * attn_w).sum(dim=1)
    q = model.cross_q(local_repr).unsqueeze(1)
    k = model.cross_k(rnafm).unsqueeze(1)
    v = model.cross_v(rnafm).unsqueeze(1)
    attn_scores = (q * k).sum(-1) / math.sqrt(64)
    attn_weights = torch.softmax(attn_scores, dim=-1)
    cross_repr = (attn_weights.unsqueeze(-1) * v).squeeze(1)
    fused = torch.cat([local_repr, cross_repr, rnafm, hand], dim=-1)
    return model.encoder(fused)


def _a8_get_shared_params(model):
    return (
        list(model.local_proj.parameters()) + [model.local_pos_enc] +
        list(model.local_transformer.parameters()) +
        list(model.local_pool_attn.parameters()) +
        list(model.cross_q.parameters()) + list(model.cross_k.parameters()) +
        list(model.cross_v.parameters()) + list(model.encoder.parameters())
    )


class HierarchicalAttention_H1(nn.Module, H1Mixin):
    name = "A8_H1"

    def __init__(self):
        super().__init__()
        (self.local_proj, self.local_pos_enc, self.local_transformer,
         self.local_pool_attn, self.cross_q, self.cross_k, self.cross_v,
         self.encoder) = _make_a8_encoder()
        self._init_heads()

    def get_shared_params(self):
        return _a8_get_shared_params(self)

    def get_adapter_params(self):
        params = list(self.binary_head.parameters())
        for enz in PER_ENZYME_HEADS:
            params.extend(self.enzyme_adapters[enz].parameters())
        params.extend(self.enzyme_classifier.parameters())
        return params

    def forward(self, batch):
        shared = _a8_encode(self, batch)
        return self._apply_heads(shared)

    def encode(self, batch):
        return _a8_encode(self, batch)


class HierarchicalAttention_H4(nn.Module, H4Mixin):
    name = "A8_H4"

    def __init__(self):
        super().__init__()
        (self.local_proj, self.local_pos_enc, self.local_transformer,
         self.local_pool_attn, self.cross_q, self.cross_k, self.cross_v,
         self.encoder) = _make_a8_encoder()
        self._init_heads()

    def get_shared_params(self):
        return _a8_get_shared_params(self)

    def get_adapter_params(self):
        params = list(self.binary_head.parameters())
        for enz in PER_ENZYME_HEADS:
            params.extend(self.private_encoders[enz].parameters())
            params.extend(self.enzyme_adapters[enz].parameters())
        params.extend(self.enzyme_classifier.parameters())
        return params

    def forward(self, batch):
        shared = _a8_encode(self, batch)
        return self._apply_heads(shared)

    def encode(self, batch):
        return _a8_encode(self, batch)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class ScreenDataset(Dataset):
    """Standard dataset for pre-computed feature architectures."""

    def __init__(self, indices, data):
        self.indices = indices
        self.data = data

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        return {
            "rnafm": torch.from_numpy(self.data["rnafm_features"][i]),
            "edit_delta": torch.from_numpy(self.data["edit_delta_features"][i]),
            "bp_submatrix": torch.from_numpy(self.data["bp_submatrices"][i]).unsqueeze(0),
            "hand_feat": torch.from_numpy(self.data["hand_features"][i]),
            "label_binary": torch.tensor(self.data["labels_binary"][i]),
            "label_enzyme": torch.tensor(self.data["labels_enzyme"][i]),
            "per_enzyme_labels": torch.from_numpy(self.data["per_enzyme_labels"][i]),
            "index": i,
        }


class PretextDataset(Dataset):
    """Dataset for structure pretext task: predict if center position is unpaired."""

    def __init__(self, indices, data):
        self.indices = indices
        self.data = data

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        return {
            "rnafm": torch.from_numpy(self.data["rnafm_features"][i]),
            "edit_delta": torch.from_numpy(self.data["edit_delta_features"][i]),
            "bp_submatrix": torch.from_numpy(self.data["bp_submatrices"][i]).unsqueeze(0),
            "hand_feat": torch.from_numpy(self.data["hand_features"][i]),
            "pretext_label": torch.tensor(self.data["pretext_labels"][i], dtype=torch.float32),
            "index": i,
        }


def standard_collate(batch_list):
    """Collate: stack all tensors."""
    result = {}
    for key in batch_list[0]:
        vals = [b[key] for b in batch_list]
        if isinstance(vals[0], torch.Tensor):
            result[key] = torch.stack(vals)
        else:
            result[key] = vals
    return result


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------


def load_v3_data() -> Dict:
    """Load v3 dataset with all pre-computed features."""
    logger.info("=" * 70)
    logger.info("Loading V3 dataset + pre-computed embeddings")
    logger.info("=" * 70)

    splits_path = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "splits_multi_enzyme_v3_with_negatives.csv"
    df = pd.read_csv(splits_path)
    logger.info(f"  Loaded {len(df)} sites from v3 splits")

    seq_path = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "multi_enzyme_sequences_v3_with_negatives.json"
    with open(seq_path) as f:
        sequences = json.load(f)

    loop_path = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "loop_position_per_site_v3.csv"
    loop_df = pd.read_csv(loop_path)
    loop_df = loop_df.drop_duplicates(subset=["site_id"]).set_index("site_id")

    struct_path = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "structure_cache_multi_enzyme_v3.npz"
    struct_data = np.load(struct_path, allow_pickle=True)
    struct_ids = list(struct_data["site_ids"])
    struct_deltas = struct_data["delta_features"]
    structure_delta = {sid: struct_deltas[i] for i, sid in enumerate(struct_ids)}
    logger.info(f"  Structure cache: {len(structure_delta)} sites")

    rnafm_path = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "embeddings" / "rnafm_pooled_v3.pt"
    rnafm_emb = torch.load(rnafm_path, map_location="cpu", weights_only=False)

    rnafm_edited_path = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "embeddings" / "rnafm_pooled_edited_v3.pt"
    rnafm_edited_emb = torch.load(rnafm_edited_path, map_location="cpu", weights_only=False)

    # BP submatrix cache
    bp_cache_path = OUTPUT_DIR / "bp_submatrices_v3.npz"
    if not bp_cache_path.exists():
        raise FileNotFoundError(f"BP cache not found at {bp_cache_path}. Run exp_architecture_screen.py first.")
    bp_data = np.load(bp_cache_path, allow_pickle=True)
    bp_submatrices_v3 = bp_data["bp_submatrices"]
    bp_site_ids_v3 = list(bp_data["site_ids"])
    logger.info(f"  BP cache: {bp_submatrices_v3.shape}")

    # Dot-bracket structures (for pretext labels)
    db_cache_path = OUTPUT_DIR / "dot_brackets_v3.json"
    dot_brackets = {}
    if db_cache_path.exists():
        with open(db_cache_path) as f:
            dot_brackets = json.load(f)
        logger.info(f"  Dot-bracket cache: {len(dot_brackets)} structures")

    # Filter to sites with sequences
    site_ids = df["site_id"].tolist()
    valid_mask = [sid in sequences for sid in site_ids]
    df = df[valid_mask].reset_index(drop=True)
    site_ids = df["site_id"].tolist()
    n = len(site_ids)
    logger.info(f"  {n} sites with sequences")

    # Hand features (40-dim)
    hand_features = build_hand_features(site_ids, sequences, structure_delta, loop_df)
    logger.info(f"  Hand features shape: {hand_features.shape}")

    # RNA-FM features
    rnafm_matrix = np.zeros((n, D_RNAFM), dtype=np.float32)
    rnafm_edited_matrix = np.zeros((n, D_RNAFM), dtype=np.float32)
    n_rnafm_found = 0
    for i, sid in enumerate(site_ids):
        if sid in rnafm_emb:
            rnafm_matrix[i] = rnafm_emb[sid].numpy()
            n_rnafm_found += 1
        if sid in rnafm_edited_emb:
            rnafm_edited_matrix[i] = rnafm_edited_emb[sid].numpy()
    edit_delta_matrix = rnafm_edited_matrix - rnafm_matrix
    logger.info(f"  RNA-FM coverage: {n_rnafm_found}/{n} ({100 * n_rnafm_found / n:.1f}%)")

    del rnafm_emb, rnafm_edited_emb
    gc.collect()

    # BP submatrices — align to current site_ids
    if bp_site_ids_v3 == site_ids:
        bp_sub = bp_submatrices_v3
    else:
        bp_sid_to_idx = {sid: i for i, sid in enumerate(bp_site_ids_v3)}
        bp_sub = np.zeros((n, 41, 41), dtype=np.float32)
        for i, sid in enumerate(site_ids):
            if sid in bp_sid_to_idx:
                bp_sub[i] = bp_submatrices_v3[bp_sid_to_idx[sid]]
    logger.info(f"  BP submatrices: {bp_sub.shape}")

    # Pretext labels: 1 if center position is unpaired
    pretext_labels = np.zeros(n, dtype=np.float32)
    n_unpaired = 0
    for i, sid in enumerate(site_ids):
        if sid in dot_brackets:
            db = dot_brackets[sid]
            if len(db) > CENTER:
                if db[CENTER] == ".":
                    pretext_labels[i] = 1.0
                    n_unpaired += 1
            else:
                pretext_labels[i] = 0.5
        else:
            pretext_labels[i] = 0.5
    logger.info(f"  Pretext labels: {n_unpaired} unpaired at center")

    # Labels
    labels_binary = df["is_edited"].values.astype(np.float32)
    labels_enzyme = np.array([ENZYME_TO_IDX.get(e, 5) for e in df["enzyme"].values], dtype=np.int64)

    per_enzyme_labels = np.full((n, N_PER_ENZYME), -1, dtype=np.float32)
    for head_idx, enz_name in enumerate(PER_ENZYME_HEADS):
        enz_idx = ENZYME_TO_IDX[enz_name]
        pos_mask = (labels_binary == 1) & (labels_enzyme == enz_idx)
        per_enzyme_labels[pos_mask, head_idx] = 1.0
        neg_mask = (labels_binary == 0) & (labels_enzyme == enz_idx)
        per_enzyme_labels[neg_mask, head_idx] = 0.0

    logger.info("Per-enzyme head sample counts:")
    for head_idx, enz_name in enumerate(PER_ENZYME_HEADS):
        n_pos = int((per_enzyme_labels[:, head_idx] == 1).sum())
        n_neg = int((per_enzyme_labels[:, head_idx] == 0).sum())
        logger.info(f"  {enz_name}: {n_pos} pos, {n_neg} neg")

    logger.info(f"  Total: {int(labels_binary.sum())} pos, {int((labels_binary == 0).sum())} neg")
    logger.info("V3 data loading complete.")

    return {
        "site_ids": site_ids,
        "df": df,
        "sequences": sequences,
        "hand_features": hand_features,
        "rnafm_features": rnafm_matrix,
        "edit_delta_features": edit_delta_matrix,
        "bp_submatrices": bp_sub,
        "pretext_labels": pretext_labels,
        "labels_binary": labels_binary,
        "labels_enzyme": labels_enzyme,
        "per_enzyme_labels": per_enzyme_labels,
        "structure_delta": structure_delta,
        "loop_df": loop_df,
    }


def load_v4_large(v3_data: Dict) -> Dict:
    """Load v4-large dataset (385K sites, 1:50 ratio)."""
    logger.info("  Loading v4-large negatives...")

    splits_path = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "splits_v4_large_negatives.csv"
    df_v4 = pd.read_csv(splits_path, low_memory=False)

    df_v4_neg = df_v4[df_v4["is_edited"] == 0].reset_index(drop=True)
    n_neg = len(df_v4_neg)
    logger.info(f"  v4-large negatives: {n_neg}")

    seq_path = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "multi_enzyme_sequences_v4_large.json"
    with open(seq_path) as f:
        v4_sequences = json.load(f)

    rnafm_v4_path = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "embeddings" / "rnafm_pooled_v4_negatives.pt"
    rnafm_v4 = torch.load(rnafm_v4_path, map_location="cpu", weights_only=False)

    rnafm_edited_v4_path = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "embeddings" / "rnafm_pooled_edited_v4_negatives.pt"
    rnafm_edited_v4 = torch.load(rnafm_edited_v4_path, map_location="cpu", weights_only=False)
    logger.info(f"  v4 RNA-FM embeddings: {len(rnafm_v4)} orig, {len(rnafm_edited_v4)} edited")

    neg_site_ids = df_v4_neg["site_id"].tolist()

    # RNA-FM features for negatives
    neg_rnafm = np.zeros((n_neg, D_RNAFM), dtype=np.float32)
    neg_rnafm_edited = np.zeros((n_neg, D_RNAFM), dtype=np.float32)
    n_found = 0
    for i, sid in enumerate(neg_site_ids):
        if sid in rnafm_v4:
            neg_rnafm[i] = rnafm_v4[sid].numpy()
            n_found += 1
        if sid in rnafm_edited_v4:
            neg_rnafm_edited[i] = rnafm_edited_v4[sid].numpy()
    neg_edit_delta = neg_rnafm_edited - neg_rnafm
    logger.info(f"  v4 RNA-FM coverage: {n_found}/{n_neg}")

    # Hand features
    all_neg_sequences = {}
    for sid in neg_site_ids:
        if sid in v4_sequences:
            all_neg_sequences[sid] = v4_sequences[sid]
        else:
            all_neg_sequences[sid] = "N" * 201

    neg_hand_features = build_hand_features(
        neg_site_ids, all_neg_sequences,
        v3_data["structure_delta"], v3_data["loop_df"]
    )

    # BP submatrices: zeros for v4 negatives
    neg_bp = np.zeros((n_neg, 41, 41), dtype=np.float32)

    # Labels
    neg_labels_binary = np.zeros(n_neg, dtype=np.float32)
    neg_labels_enzyme = np.full(n_neg, 5, dtype=np.int64)  # Unknown
    neg_per_enzyme_labels = np.full((n_neg, N_PER_ENZYME), -1, dtype=np.float32)

    # Concatenate v3 + v4 negatives
    n_v3 = len(v3_data["site_ids"])

    combined = {
        "site_ids": v3_data["site_ids"] + neg_site_ids,
        "hand_features": np.concatenate([v3_data["hand_features"], neg_hand_features], axis=0),
        "rnafm_features": np.concatenate([v3_data["rnafm_features"], neg_rnafm], axis=0),
        "edit_delta_features": np.concatenate([v3_data["edit_delta_features"], neg_edit_delta], axis=0),
        "bp_submatrices": np.concatenate([v3_data["bp_submatrices"], neg_bp], axis=0),
        "labels_binary": np.concatenate([v3_data["labels_binary"], neg_labels_binary], axis=0),
        "labels_enzyme": np.concatenate([v3_data["labels_enzyme"], neg_labels_enzyme], axis=0),
        "per_enzyme_labels": np.concatenate([v3_data["per_enzyme_labels"], neg_per_enzyme_labels], axis=0),
        "n_v3": n_v3,
    }

    n_total = len(combined["site_ids"])
    logger.info(f"  Combined v4-large: {n_total} sites ({int(combined['labels_binary'].sum())} pos, {int((combined['labels_binary'] == 0).sum())} neg)")

    del rnafm_v4, rnafm_edited_v4
    gc.collect()

    return combined


# ---------------------------------------------------------------------------
# Loss / Training / Evaluation (copied from winner_screen)
# ---------------------------------------------------------------------------


def compute_loss(binary_logit, per_enzyme_logits, enzyme_cls_logits, batch):
    """Compute combined loss: binary + per-enzyme + enzyme classifier."""
    label_binary = batch["label_binary"]
    label_enzyme = batch["label_enzyme"]
    per_enzyme_labels = batch["per_enzyme_labels"]

    loss_binary = F.binary_cross_entropy_with_logits(binary_logit, label_binary)

    loss_enzyme_heads = torch.tensor(0.0, device=binary_logit.device)
    n_valid_heads = 0
    for head_idx in range(N_PER_ENZYME):
        enz_labels = per_enzyme_labels[:, head_idx]
        mask = enz_labels >= 0
        if mask.sum() > 0:
            head_loss = F.binary_cross_entropy_with_logits(
                per_enzyme_logits[head_idx][mask], enz_labels[mask]
            )
            loss_enzyme_heads = loss_enzyme_heads + head_loss
            n_valid_heads += 1
    if n_valid_heads > 0:
        loss_enzyme_heads = loss_enzyme_heads / n_valid_heads

    pos_mask = label_binary > 0.5
    loss_cls = torch.tensor(0.0, device=binary_logit.device)
    if pos_mask.sum() > 0:
        loss_cls = F.cross_entropy(enzyme_cls_logits[pos_mask], label_enzyme[pos_mask])

    total = loss_binary + STAGE1_ENZYME_HEAD_WEIGHT * loss_enzyme_heads + STAGE1_ENZYME_CLS_WEIGHT * loss_cls
    return total, loss_binary.item()


def train_one_epoch(model, loader, optimizer, device):
    """Train one epoch. Returns (avg_loss, avg_binary_loss)."""
    model.train()
    total_loss, total_binary, n_batches = 0.0, 0.0, 0

    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        optimizer.zero_grad()
        binary_logit, per_enzyme_logits, enzyme_cls_logits = model(batch)
        loss, binary_loss = compute_loss(binary_logit, per_enzyme_logits, enzyme_cls_logits, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_binary += binary_loss
        n_batches += 1

    return total_loss / max(n_batches, 1), total_binary / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate: returns overall AUROC and per-enzyme AUROCs."""
    model.eval()
    all_probs, all_labels = [], []
    all_per_enzyme_probs = [[] for _ in range(N_PER_ENZYME)]
    all_per_enzyme_labels_list = [[] for _ in range(N_PER_ENZYME)]

    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        binary_logit, per_enzyme_logits, _ = model(batch)
        probs = torch.sigmoid(binary_logit).cpu().numpy()
        labels = batch["label_binary"].cpu().numpy()
        per_enz_labels = batch["per_enzyme_labels"].cpu().numpy()

        all_probs.extend(probs)
        all_labels.extend(labels)

        for h_idx in range(N_PER_ENZYME):
            enz_probs = torch.sigmoid(per_enzyme_logits[h_idx]).cpu().numpy()
            mask = per_enz_labels[:, h_idx] >= 0
            if mask.any():
                all_per_enzyme_probs[h_idx].extend(enz_probs[mask])
                all_per_enzyme_labels_list[h_idx].extend(per_enz_labels[mask, h_idx])

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    try:
        overall_auroc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        overall_auroc = 0.5

    per_enzyme_aurocs = {}
    for h_idx, enz_name in enumerate(PER_ENZYME_HEADS):
        if len(all_per_enzyme_probs[h_idx]) > 10:
            y_true = np.array(all_per_enzyme_labels_list[h_idx])
            y_pred = np.array(all_per_enzyme_probs[h_idx])
            if len(np.unique(y_true)) > 1:
                try:
                    per_enzyme_aurocs[enz_name] = roc_auc_score(y_true, y_pred)
                except ValueError:
                    per_enzyme_aurocs[enz_name] = 0.5
            else:
                per_enzyme_aurocs[enz_name] = float("nan")
        else:
            per_enzyme_aurocs[enz_name] = float("nan")

    return overall_auroc, per_enzyme_aurocs


# ---------------------------------------------------------------------------
# Pretext pretraining (T6)
# ---------------------------------------------------------------------------


class PretextHead(nn.Module):
    """Simple binary head for pretext prediction."""

    def __init__(self):
        super().__init__()
        self.head = nn.Linear(D_SHARED, 1)

    def forward(self, shared):
        return self.head(shared).squeeze(-1)


def pretrain_pretext(model, pretext_head, train_loader, device, n_epochs=5, lr=1e-3):
    """Pretrain the shared encoder on the structure pretext task."""
    model.train()
    pretext_head.train()

    optimizer = torch.optim.Adam(
        list(model.get_shared_params()) + list(pretext_head.parameters()),
        lr=lr, weight_decay=1e-4,
    )

    for epoch in range(n_epochs):
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            optimizer.zero_grad()

            shared = model.encode(batch)
            logits = pretext_head(shared)
            labels = batch["pretext_label"]

            mask = (labels != 0.5)
            if mask.sum() == 0:
                continue

            loss = F.binary_cross_entropy_with_logits(logits[mask], labels[mask])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(pretext_head.parameters()), 1.0
            )
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(f"    Pretext epoch {epoch + 1:2d} | loss={avg_loss:.4f}")


# ---------------------------------------------------------------------------
# Training methods for 7-fold CV
# ---------------------------------------------------------------------------


def train_t1_7fold(model_cls, data):
    """T1: Standard v3 training with 7-fold CV."""
    n = len(data["labels_binary"])
    strat_key = data["labels_enzyme"] * 2 + data["labels_binary"].astype(np.int64)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    fold_results = []
    total_t0 = time.time()

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(n), strat_key)):
        logger.info(f"\n--- Fold {fold + 1}/{N_FOLDS} ({len(train_idx)} train, {len(val_idx)} val) ---")
        fold_t0 = time.time()

        fold_model = model_cls().to(DEVICE)

        train_ds = ScreenDataset(train_idx, data)
        val_ds = ScreenDataset(val_idx, data)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  collate_fn=standard_collate, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                collate_fn=standard_collate, num_workers=0)

        # Stage 1: Joint training
        optimizer = torch.optim.Adam([
            {"params": fold_model.get_shared_params(), "lr": STAGE1_LR},
            {"params": fold_model.get_adapter_params(), "lr": STAGE1_LR},
        ], weight_decay=STAGE1_WEIGHT_DECAY)

        for epoch in range(STAGE1_EPOCHS):
            avg_loss, avg_binary = train_one_epoch(fold_model, train_loader, optimizer, DEVICE)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                val_auroc, val_per_enz = evaluate(fold_model, val_loader, DEVICE)
                enz_str = " ".join(f"{e}={v:.3f}" for e, v in val_per_enz.items() if not np.isnan(v))
                logger.info(
                    f"    Epoch {epoch + 1:2d} | loss={avg_loss:.4f} bce={avg_binary:.4f} "
                    f"| val_auroc={val_auroc:.4f} | {enz_str}"
                )

        # Stage 2: Adapter-only
        for p in fold_model.get_shared_params():
            p.requires_grad = False

        optimizer2 = torch.optim.Adam(
            fold_model.get_adapter_params(),
            lr=STAGE2_LR, weight_decay=STAGE2_WEIGHT_DECAY,
        )

        for epoch in range(STAGE2_EPOCHS):
            avg_loss, avg_binary = train_one_epoch(fold_model, train_loader, optimizer2, DEVICE)
            if (epoch + 1) % 5 == 0 or epoch == STAGE2_EPOCHS - 1:
                val_auroc, val_per_enz = evaluate(fold_model, val_loader, DEVICE)
                enz_str = " ".join(f"{e}={v:.3f}" for e, v in val_per_enz.items() if not np.isnan(v))
                logger.info(
                    f"    Epoch {STAGE1_EPOCHS + epoch + 1:2d} | loss={avg_loss:.4f} "
                    f"| val_auroc={val_auroc:.4f} | {enz_str}"
                )

        final_auroc, final_per_enz = evaluate(fold_model, val_loader, DEVICE)
        fold_time = time.time() - fold_t0
        logger.info(f"  Fold {fold + 1} FINAL: overall_auroc={final_auroc:.4f} | time={fold_time:.0f}s")
        for enz, auroc in final_per_enz.items():
            logger.info(f"    {enz}: {auroc:.4f}")

        fold_results.append({
            "fold": fold + 1,
            "overall_auroc": final_auroc,
            "per_enzyme_aurocs": final_per_enz,
            "time_s": fold_time,
        })

        del fold_model, optimizer, optimizer2, train_loader, val_loader, train_ds, val_ds
        gc.collect()
        if DEVICE.type == "mps":
            torch.mps.empty_cache()

    total_time = time.time() - total_t0
    return _aggregate_fold_results(fold_results, total_time)


def train_t4_7fold(model_cls, v3_data, v4_data):
    """T4: Train on v4-large, eval on v3 val fold. 7-fold CV."""
    n = len(v4_data["labels_binary"])
    n_v3 = v4_data.get("n_v3", n)

    strat_key = v4_data["labels_binary"].astype(np.int64)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    fold_results = []
    total_t0 = time.time()

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(n), strat_key)):
        logger.info(f"\n--- Fold {fold + 1}/{N_FOLDS} ({len(train_idx)} train, {len(val_idx)} val) ---")
        fold_t0 = time.time()

        fold_model = model_cls().to(DEVICE)

        train_ds = ScreenDataset(train_idx, v4_data)
        val_ds = ScreenDataset(val_idx, v4_data)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE_LARGE, shuffle=True,
                                  collate_fn=standard_collate, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE_LARGE, shuffle=False,
                                collate_fn=standard_collate, num_workers=0)

        # Eval on v3-matched validation subset for per-enzyme AUROCs
        v3_val_idx = val_idx[val_idx < n_v3]
        if len(v3_val_idx) > 50:
            eval_ds = ScreenDataset(v3_val_idx, v4_data)
            eval_loader = DataLoader(eval_ds, batch_size=BATCH_SIZE_LARGE, shuffle=False,
                                     collate_fn=standard_collate, num_workers=0)
        else:
            eval_loader = val_loader

        # Stage 1: Joint (5 epochs)
        optimizer = torch.optim.Adam([
            {"params": fold_model.get_shared_params(), "lr": STAGE1_LR},
            {"params": fold_model.get_adapter_params(), "lr": STAGE1_LR},
        ], weight_decay=STAGE1_WEIGHT_DECAY)

        for epoch in range(STAGE1_EPOCHS_LARGE):
            avg_loss, avg_binary = train_one_epoch(fold_model, train_loader, optimizer, DEVICE)
            if (epoch + 1) % 3 == 0 or epoch == 0:
                val_auroc, _ = evaluate(fold_model, val_loader, DEVICE)
                logger.info(
                    f"    Epoch {epoch + 1:2d} | loss={avg_loss:.4f} bce={avg_binary:.4f} "
                    f"| val_auroc={val_auroc:.4f}"
                )

        # Stage 2: Adapter-only (10 epochs)
        for p in fold_model.get_shared_params():
            p.requires_grad = False

        optimizer2 = torch.optim.Adam(
            fold_model.get_adapter_params(),
            lr=STAGE2_LR, weight_decay=STAGE2_WEIGHT_DECAY,
        )

        for epoch in range(STAGE2_EPOCHS_LARGE):
            avg_loss, avg_binary = train_one_epoch(fold_model, train_loader, optimizer2, DEVICE)
            if (epoch + 1) % 5 == 0 or epoch == STAGE2_EPOCHS_LARGE - 1:
                val_auroc, _ = evaluate(fold_model, val_loader, DEVICE)
                logger.info(
                    f"    Epoch {STAGE1_EPOCHS_LARGE + epoch + 1:2d} | loss={avg_loss:.4f} "
                    f"| val_auroc={val_auroc:.4f}"
                )

        # Final eval on v3-matched subset
        final_auroc, _ = evaluate(fold_model, val_loader, DEVICE)
        _, final_per_enz = evaluate(fold_model, eval_loader, DEVICE)

        fold_time = time.time() - fold_t0
        logger.info(f"  Fold {fold + 1} FINAL: overall_auroc={final_auroc:.4f} | time={fold_time:.0f}s")
        for enz, auroc in final_per_enz.items():
            logger.info(f"    {enz}: {auroc:.4f}")

        fold_results.append({
            "fold": fold + 1,
            "overall_auroc": final_auroc,
            "per_enzyme_aurocs": final_per_enz,
            "time_s": fold_time,
        })

        del fold_model, optimizer, optimizer2, train_loader, val_loader
        gc.collect()
        if DEVICE.type == "mps":
            torch.mps.empty_cache()

    total_time = time.time() - total_t0
    return _aggregate_fold_results(fold_results, total_time)


def train_t6_7fold(model_cls, data):
    """T6: 5 epochs pretext pretraining, then standard T1 schedule. 7-fold CV."""
    n = len(data["labels_binary"])
    strat_key = data["labels_enzyme"] * 2 + data["labels_binary"].astype(np.int64)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    fold_results = []
    total_t0 = time.time()

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(n), strat_key)):
        logger.info(f"\n--- Fold {fold + 1}/{N_FOLDS} ({len(train_idx)} train, {len(val_idx)} val) ---")
        fold_t0 = time.time()

        fold_model = model_cls().to(DEVICE)
        pretext_head = PretextHead().to(DEVICE)

        # Stage 0: Pretext pretraining
        logger.info(f"  Stage 0: Pretext pretraining ({PRETEXT_EPOCHS} epochs)")
        pretext_ds = PretextDataset(train_idx, data)
        pretext_loader = DataLoader(pretext_ds, batch_size=BATCH_SIZE, shuffle=True,
                                    collate_fn=standard_collate, num_workers=0)
        pretrain_pretext(fold_model, pretext_head, pretext_loader, DEVICE,
                         n_epochs=PRETEXT_EPOCHS, lr=1e-3)

        del pretext_head
        gc.collect()

        # Re-initialize heads (shared encoder is pretrained)
        fold_model._init_heads()
        fold_model = fold_model.to(DEVICE)

        # Stage 1: Fine-tune joint
        train_ds = ScreenDataset(train_idx, data)
        val_ds = ScreenDataset(val_idx, data)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  collate_fn=standard_collate, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                collate_fn=standard_collate, num_workers=0)

        optimizer = torch.optim.Adam([
            {"params": fold_model.get_shared_params(), "lr": STAGE1_LR},
            {"params": fold_model.get_adapter_params(), "lr": STAGE1_LR},
        ], weight_decay=STAGE1_WEIGHT_DECAY)

        for epoch in range(STAGE1_EPOCHS):
            avg_loss, avg_binary = train_one_epoch(fold_model, train_loader, optimizer, DEVICE)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                val_auroc, val_per_enz = evaluate(fold_model, val_loader, DEVICE)
                enz_str = " ".join(f"{e}={v:.3f}" for e, v in val_per_enz.items() if not np.isnan(v))
                logger.info(
                    f"    Epoch {epoch + 1:2d} | loss={avg_loss:.4f} bce={avg_binary:.4f} "
                    f"| val_auroc={val_auroc:.4f} | {enz_str}"
                )

        # Stage 2: Adapter-only
        for p in fold_model.get_shared_params():
            p.requires_grad = False

        optimizer2 = torch.optim.Adam(
            fold_model.get_adapter_params(),
            lr=STAGE2_LR, weight_decay=STAGE2_WEIGHT_DECAY,
        )

        for epoch in range(STAGE2_EPOCHS):
            avg_loss, avg_binary = train_one_epoch(fold_model, train_loader, optimizer2, DEVICE)
            if (epoch + 1) % 5 == 0 or epoch == STAGE2_EPOCHS - 1:
                val_auroc, val_per_enz = evaluate(fold_model, val_loader, DEVICE)
                enz_str = " ".join(f"{e}={v:.3f}" for e, v in val_per_enz.items() if not np.isnan(v))
                logger.info(
                    f"    Epoch {STAGE1_EPOCHS + epoch + 1:2d} | loss={avg_loss:.4f} "
                    f"| val_auroc={val_auroc:.4f} | {enz_str}"
                )

        final_auroc, final_per_enz = evaluate(fold_model, val_loader, DEVICE)
        fold_time = time.time() - fold_t0
        logger.info(f"  Fold {fold + 1} FINAL: overall_auroc={final_auroc:.4f} | time={fold_time:.0f}s")
        for enz, auroc in final_per_enz.items():
            logger.info(f"    {enz}: {auroc:.4f}")

        fold_results.append({
            "fold": fold + 1,
            "overall_auroc": final_auroc,
            "per_enzyme_aurocs": final_per_enz,
            "time_s": fold_time,
        })

        del fold_model, optimizer, optimizer2, train_loader, val_loader
        gc.collect()
        if DEVICE.type == "mps":
            torch.mps.empty_cache()

    total_time = time.time() - total_t0
    return _aggregate_fold_results(fold_results, total_time)


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def _aggregate_fold_results(fold_results, total_time):
    """Aggregate fold results into summary statistics."""
    overall_aurocs = [r["overall_auroc"] for r in fold_results]
    mean_overall = np.mean(overall_aurocs)
    std_overall = np.std(overall_aurocs)

    per_enz_means = {}
    per_enz_stds = {}
    for enz in PER_ENZYME_HEADS:
        vals = [r["per_enzyme_aurocs"].get(enz, float("nan")) for r in fold_results]
        vals = [v for v in vals if not np.isnan(v)]
        if vals:
            per_enz_means[enz] = np.mean(vals)
            per_enz_stds[enz] = np.std(vals)

    if per_enz_means:
        mean_per_enz_auroc = np.mean(list(per_enz_means.values()))
    else:
        mean_per_enz_auroc = 0.5

    return {
        "overall_auroc_mean": round(float(mean_overall), 4),
        "overall_auroc_std": round(float(std_overall), 4),
        "per_enzyme_auroc_mean": {k: round(float(v), 4) for k, v in per_enz_means.items()},
        "per_enzyme_auroc_std": {k: round(float(v), 4) for k, v in per_enz_stds.items()},
        "mean_per_enzyme_auroc": round(float(mean_per_enz_auroc), 4),
        "total_time_s": round(total_time, 1),
        "fold_results": fold_results,
    }


# ---------------------------------------------------------------------------
# Results management
# ---------------------------------------------------------------------------


def save_results(results: Dict):
    """Save results incrementally."""
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)


def load_results() -> Dict:
    """Load existing results if available."""
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# PID wait
# ---------------------------------------------------------------------------


def wait_for_pid(pid: int, poll_interval: int = 30):
    """Wait for a process to finish by polling os.kill(pid, 0)."""
    logger.info(f"Waiting for PID {pid} to finish...")
    while True:
        try:
            os.kill(pid, 0)
            logger.info(f"  PID {pid} still running, waiting {poll_interval}s...")
            time.sleep(poll_interval)
        except ProcessLookupError:
            logger.info(f"  PID {pid} has finished (process not found).")
            return
        except PermissionError:
            # Process exists but we don't have permission — treat as still running
            logger.info(f"  PID {pid} still running (permission check), waiting {poll_interval}s...")
            time.sleep(poll_interval)


# ---------------------------------------------------------------------------
# Top 5 winners (hardcoded from Phase 1 3-fold results)
# ---------------------------------------------------------------------------

TOP5_WINNERS = [
    ("A8_T1_H4", HierarchicalAttention_H4, "T1"),
    ("A8_T6_H4", HierarchicalAttention_H4, "T6"),
    ("A8_T6_H1", HierarchicalAttention_H1, "T6"),
    ("A8_T1_H1", HierarchicalAttention_H1, "T1"),
    ("A8_T4_H1", HierarchicalAttention_H1, "T4"),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    logger.info("=" * 80)
    logger.info("7-FOLD CV FOR TOP 5 WINNERS")
    logger.info(f"N_FOLDS={N_FOLDS}, SEED={SEED}")
    logger.info("=" * 80)

    # Wait for PID 42342 (winner screen) to finish
    wait_for_pid(42342, poll_interval=30)

    # Load existing results for incremental saving
    results = load_results()
    results["config"] = {
        "n_folds": N_FOLDS,
        "seed": SEED,
        "top5": [name for name, _, _ in TOP5_WINNERS],
    }

    # Load data
    v3_data = load_v3_data()

    # Check if any combo needs v4 data
    needs_v4 = any(tm == "T4" for _, _, tm in TOP5_WINNERS)
    v4_data = None
    if needs_v4:
        v4_data = load_v4_large(v3_data)

    # Run 7-fold CV for each winner
    for rank, (combo_name, model_cls, training_method) in enumerate(TOP5_WINNERS, 1):
        if combo_name in results and "error" not in results.get(combo_name, {}):
            # Check if already completed
            existing = results.get(combo_name, {})
            if "fold_results" in existing and len(existing["fold_results"]) == N_FOLDS:
                logger.info(f"\n[{rank}/5] {combo_name} already complete, skipping")
                continue

        logger.info(f"\n{'=' * 70}")
        logger.info(f"[{rank}/5] {combo_name} (arch={model_cls.name}, training={training_method})")
        logger.info(f"{'=' * 70}")

        try:
            if training_method == "T1":
                result = train_t1_7fold(model_cls, v3_data)
            elif training_method == "T4":
                result = train_t4_7fold(model_cls, v3_data, v4_data)
            elif training_method == "T6":
                result = train_t6_7fold(model_cls, v3_data)
            else:
                raise ValueError(f"Unknown training method: {training_method}")

            result["combo_name"] = combo_name
            result["architecture"] = model_cls.name
            result["training_method"] = training_method
            results[combo_name] = result

            logger.info(f"\n  {combo_name} SUMMARY ({N_FOLDS}-fold CV):")
            logger.info(f"    Overall AUROC: {result['overall_auroc_mean']:.4f} +/- {result['overall_auroc_std']:.4f}")
            logger.info(f"    Mean per-enzyme AUROC: {result['mean_per_enzyme_auroc']:.4f}")
            for enz in PER_ENZYME_HEADS:
                if enz in result["per_enzyme_auroc_mean"]:
                    logger.info(f"    {enz}: {result['per_enzyme_auroc_mean'][enz]:.4f} +/- {result['per_enzyme_auroc_std'].get(enz, 0):.4f}")
            logger.info(f"    Time: {result['total_time_s']:.0f}s")

        except Exception as e:
            logger.error(f"FAILED: {combo_name}: {e}")
            logger.error(traceback.format_exc())
            results[combo_name] = {"error": str(e)}

        save_results(results)
        gc.collect()
        if DEVICE.type == "mps":
            torch.mps.empty_cache()

    # Print final comparison table
    logger.info("\n" + "=" * 100)
    logger.info(f"FINAL RESULTS — {N_FOLDS}-Fold CV, Ranked by Mean Per-Enzyme AUROC")
    logger.info("=" * 100)

    rows = []
    for combo_name, _, _ in TOP5_WINNERS:
        r = results.get(combo_name, {})
        if "error" in r or "mean_per_enzyme_auroc" not in r:
            continue
        rows.append((combo_name, r["mean_per_enzyme_auroc"], r))

    rows.sort(key=lambda x: x[1], reverse=True)

    header = f"{'Rank':<5} {'Combo':<18} {'Overall':>12} {'MeanPerEnz':>12}"
    for enz in PER_ENZYME_HEADS:
        header += f" {enz:>10}"
    header += f" {'Time':>8}"
    logger.info(header)
    logger.info("-" * 110)

    for rank, (name, mean_pe, r) in enumerate(rows, 1):
        ov = f"{r['overall_auroc_mean']:.4f}+/-{r['overall_auroc_std']:.4f}"
        row = f"{rank:<5} {name:<18} {ov:>12} {mean_pe:>12.4f}"
        for enz in PER_ENZYME_HEADS:
            val = r.get("per_enzyme_auroc_mean", {}).get(enz, float("nan"))
            if np.isnan(val):
                row += f" {'N/A':>10}"
            else:
                row += f" {val:>10.4f}"
        row += f" {r.get('total_time_s', 0):>7.0f}s"
        logger.info(row)

    logger.info("=" * 110)
    logger.info(f"\nResults saved to: {RESULTS_PATH}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
