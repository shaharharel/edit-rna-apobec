#!/usr/bin/env python
"""Winner screen for RNA editing prediction.

Phase 1: 3-fold CV on 12 combinations (2 architectures x 3 training methods x 2 heads).
Phase 2: Top 5 winners -> 5-fold CV + challenge splits (gene holdout, chr holdout, enzyme LOO).

Output: experiments/multi_enzyme/outputs/architecture_screen/winner_screen_results.json
Log:    experiments/multi_enzyme/outputs/architecture_screen/winner_screen.log
Run:    nohup /opt/miniconda3/envs/quris/bin/python -u experiments/multi_enzyme/exp_winner_screen.py \
        > experiments/multi_enzyme/outputs/architecture_screen/winner_screen.log 2>&1 &
"""

import gc
import json
import logging
import os
import sys
import time
import traceback
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, GroupKFold
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

RESULTS_PATH = OUTPUT_DIR / "winner_screen_results.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(OUTPUT_DIR / "winner_screen.log", mode="w"),
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
    """H1: Binary head on all data + 5 per-enzyme adapters. Standard two-stage training."""

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
    """H4: Shared encoder -> 128dim. Each enzyme gets: shared(128) concat with
    private_encoder(Linear(128,32)->GELU) -> adapter head on 160-dim. Binary head on shared 128-dim."""

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
            enz: nn.Linear(D_SHARED + 32, 1)  # shared(128) + private(32) -> 1
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
# Architecture: A6 (Conv2DBP) — all head variants
# ---------------------------------------------------------------------------


def _make_a6_encoder():
    """Build A6 encoder components."""
    conv = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(4),
    )
    conv_fc = nn.Linear(32 * 4 * 4, 128)
    d_fused = 128 + D_RNAFM + D_HAND
    encoder = nn.Sequential(
        nn.Linear(d_fused, 256), nn.GELU(), nn.Dropout(0.3), nn.LayerNorm(256),
        nn.Linear(256, D_SHARED), nn.GELU(), nn.Dropout(0.2),
    )
    return conv, conv_fc, encoder


def _a6_encode(model, batch):
    """Shared encoding logic for A6."""
    bp = batch["bp_submatrix"]
    conv_out = model.conv(bp).flatten(1)
    conv_out = model.conv_fc(conv_out)
    fused = torch.cat([conv_out, batch["rnafm"], batch["hand_feat"]], dim=-1)
    return model.encoder(fused)


def _a6_get_shared_params(model):
    return list(model.conv.parameters()) + list(model.conv_fc.parameters()) + list(model.encoder.parameters())


class Conv2DBP_H1(nn.Module, H1Mixin):
    name = "A6_H1"

    def __init__(self):
        super().__init__()
        self.conv, self.conv_fc, self.encoder = _make_a6_encoder()
        self._init_heads()

    def get_shared_params(self):
        return _a6_get_shared_params(self)

    def get_adapter_params(self):
        params = list(self.binary_head.parameters())
        for enz in PER_ENZYME_HEADS:
            params.extend(self.enzyme_adapters[enz].parameters())
        params.extend(self.enzyme_classifier.parameters())
        return params

    def forward(self, batch):
        shared = _a6_encode(self, batch)
        return self._apply_heads(shared)

    def encode(self, batch):
        return _a6_encode(self, batch)


class Conv2DBP_H4(nn.Module, H4Mixin):
    name = "A6_H4"

    def __init__(self):
        super().__init__()
        self.conv, self.conv_fc, self.encoder = _make_a6_encoder()
        self._init_heads()

    def get_shared_params(self):
        return _a6_get_shared_params(self)

    def get_adapter_params(self):
        params = list(self.binary_head.parameters())
        for enz in PER_ENZYME_HEADS:
            params.extend(self.private_encoders[enz].parameters())
            params.extend(self.enzyme_adapters[enz].parameters())
        params.extend(self.enzyme_classifier.parameters())
        return params

    def forward(self, batch):
        shared = _a6_encode(self, batch)
        return self._apply_heads(shared)

    def encode(self, batch):
        return _a6_encode(self, batch)


# ---------------------------------------------------------------------------
# Architecture: A8 (HierarchicalAttention) — all head variants
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
# Dataset classes
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
    """Load v4-large dataset (385K sites, 1:50 ratio).

    Positives are the SAME as v3. Negatives come from v4-large splits.
    V4 negatives get RNA-FM + motif but ZERO structure/loop/BP features.
    """
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

    # Hand features: motif from sequence, loop+struct = zeros (not cached)
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
# Loss / Training / Evaluation
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


def _aggregate_fold_results(fold_results: List[Dict], total_time: float) -> Dict:
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

    # Compute mean per-enzyme AUROC (the ranking metric)
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
# Training methods
# ---------------------------------------------------------------------------


def train_t1_baseline(model_cls, data, n_folds=3, eval_data=None):
    """T1: Standard v3 training. Stage 1 (10 ep joint) -> Stage 2 (20 ep adapters)."""
    n = len(data["labels_binary"])
    strat_key = data["labels_enzyme"] * 2 + data["labels_binary"].astype(np.int64)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    fold_results = []
    total_t0 = time.time()

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(n), strat_key)):
        logger.info(f"\n--- Fold {fold + 1}/{n_folds} ({len(train_idx)} train, {len(val_idx)} val) ---")
        fold_t0 = time.time()

        fold_model = model_cls().to(DEVICE)

        train_ds = ScreenDataset(train_idx, data)
        val_ds = ScreenDataset(val_idx, data)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  collate_fn=standard_collate, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                collate_fn=standard_collate, num_workers=0)

        # Use eval_data val subset if provided (for T4)
        if eval_data is not None:
            eval_loader = val_loader  # Default fallback
        else:
            eval_loader = val_loader

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

        final_auroc, final_per_enz = evaluate(fold_model, eval_loader, DEVICE)
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


def train_t4_v4large(model_cls, v3_data, v4_data, n_folds=3):
    """T4: Train on v4-large, eval on v3 val fold. Reduced epochs, batch 256."""
    n = len(v4_data["labels_binary"])
    n_v3 = v4_data.get("n_v3", n)

    strat_key = v4_data["labels_binary"].astype(np.int64)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    fold_results = []
    total_t0 = time.time()

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(n), strat_key)):
        logger.info(f"\n--- Fold {fold + 1}/{n_folds} ({len(train_idx)} train, {len(val_idx)} val) ---")
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


def train_t6_pretext(model_cls, data, n_folds=3):
    """T6: 5 epochs pretext pretraining (predict unpaired), then standard T1 schedule."""
    n = len(data["labels_binary"])
    strat_key = data["labels_enzyme"] * 2 + data["labels_binary"].astype(np.int64)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    fold_results = []
    total_t0 = time.time()

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(n), strat_key)):
        logger.info(f"\n--- Fold {fold + 1}/{n_folds} ({len(train_idx)} train, {len(val_idx)} val) ---")
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
# Phase 2: Challenge splits
# ---------------------------------------------------------------------------


def train_and_eval_single_split(model_cls, data, train_idx, val_idx, batch_size=BATCH_SIZE):
    """Train on train_idx, evaluate on val_idx. Returns (overall_auroc, per_enzyme_aurocs)."""
    fold_model = model_cls().to(DEVICE)

    train_ds = ScreenDataset(train_idx, data)
    val_ds = ScreenDataset(val_idx, data)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=standard_collate, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=standard_collate, num_workers=0)

    # Stage 1: Joint training
    optimizer = torch.optim.Adam([
        {"params": fold_model.get_shared_params(), "lr": STAGE1_LR},
        {"params": fold_model.get_adapter_params(), "lr": STAGE1_LR},
    ], weight_decay=STAGE1_WEIGHT_DECAY)

    for epoch in range(STAGE1_EPOCHS):
        train_one_epoch(fold_model, train_loader, optimizer, DEVICE)

    # Stage 2: Adapter-only
    for p in fold_model.get_shared_params():
        p.requires_grad = False

    optimizer2 = torch.optim.Adam(
        fold_model.get_adapter_params(),
        lr=STAGE2_LR, weight_decay=STAGE2_WEIGHT_DECAY,
    )

    for epoch in range(STAGE2_EPOCHS):
        train_one_epoch(fold_model, train_loader, optimizer2, DEVICE)

    final_auroc, final_per_enz = evaluate(fold_model, val_loader, DEVICE)

    del fold_model, optimizer, optimizer2, train_loader, val_loader
    gc.collect()
    if DEVICE.type == "mps":
        torch.mps.empty_cache()

    return final_auroc, final_per_enz


def run_5fold_cv(model_cls, data, combo_name):
    """Standard 5-fold StratifiedKFold CV."""
    logger.info(f"\n  5-Fold CV for {combo_name}")
    n = len(data["labels_binary"])
    strat_key = data["labels_enzyme"] * 2 + data["labels_binary"].astype(np.int64)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    fold_results = []
    t0 = time.time()

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(n), strat_key)):
        logger.info(f"    Fold {fold + 1}/5 ({len(train_idx)} train, {len(val_idx)} val)")
        auroc, per_enz = train_and_eval_single_split(model_cls, data, train_idx, val_idx)
        enz_str = " ".join(f"{e}={v:.3f}" for e, v in per_enz.items() if not np.isnan(v))
        logger.info(f"      AUROC={auroc:.4f} | {enz_str}")
        fold_results.append({
            "fold": fold + 1,
            "overall_auroc": auroc,
            "per_enzyme_aurocs": per_enz,
        })

    total_time = time.time() - t0
    result = _aggregate_fold_results(fold_results, total_time)
    logger.info(f"    5-Fold mean AUROC: {result['overall_auroc_mean']:.4f} +/- {result['overall_auroc_std']:.4f}")
    return result


def run_gene_holdout(model_cls, data):
    """Gene holdout: split by chromosome (as proxy for gene grouping)."""
    logger.info(f"\n  Gene holdout (chromosome-based 5-fold)")
    df = data["df"]
    chromosomes = df["chr"].values

    # Get unique chromosomes and assign to 5 groups
    unique_chr = sorted(df["chr"].unique())
    np.random.seed(SEED)
    np.random.shuffle(unique_chr)
    chr_groups = {}
    for i, c in enumerate(unique_chr):
        chr_groups[c] = i % 5

    groups = np.array([chr_groups[c] for c in chromosomes])

    fold_results = []
    t0 = time.time()

    for fold_id in range(5):
        val_mask = groups == fold_id
        train_mask = ~val_mask
        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]

        if len(val_idx) < 50:
            logger.info(f"    Gene fold {fold_id + 1}: too few test sites ({len(val_idx)}), skipping")
            continue

        logger.info(f"    Gene fold {fold_id + 1}/5 ({len(train_idx)} train, {len(val_idx)} val)")
        auroc, per_enz = train_and_eval_single_split(model_cls, data, train_idx, val_idx)
        enz_str = " ".join(f"{e}={v:.3f}" for e, v in per_enz.items() if not np.isnan(v))
        logger.info(f"      AUROC={auroc:.4f} | {enz_str}")
        fold_results.append({
            "fold": fold_id + 1,
            "overall_auroc": auroc,
            "per_enzyme_aurocs": per_enz,
        })

    total_time = time.time() - t0
    if fold_results:
        result = _aggregate_fold_results(fold_results, total_time)
        logger.info(f"    Gene holdout mean AUROC: {result['overall_auroc_mean']:.4f} +/- {result['overall_auroc_std']:.4f}")
        return result
    return {"error": "not enough folds"}


def run_chr_holdout(model_cls, data):
    """Chromosome holdout: train on chr1-17, test on chr18-22+X."""
    logger.info(f"\n  Chromosome holdout (chr1-17 train, chr18-22+X test)")
    df = data["df"]

    test_chrs = {"chr18", "chr19", "chr20", "chr21", "chr22", "chrX"}
    test_mask = df["chr"].isin(test_chrs).values
    train_mask = ~test_mask

    train_idx = np.where(train_mask)[0]
    val_idx = np.where(test_mask)[0]

    logger.info(f"    Train: {len(train_idx)}, Test: {len(val_idx)}")

    if len(val_idx) < 200:
        logger.info(f"    Too few test sites ({len(val_idx)}), skipping")
        return {"error": f"only {len(val_idx)} test sites"}

    t0 = time.time()
    auroc, per_enz = train_and_eval_single_split(model_cls, data, train_idx, val_idx)
    total_time = time.time() - t0

    enz_str = " ".join(f"{e}={v:.3f}" for e, v in per_enz.items() if not np.isnan(v))
    logger.info(f"    Chr holdout AUROC={auroc:.4f} | {enz_str} | {total_time:.0f}s")

    return {
        "overall_auroc": round(float(auroc), 4),
        "per_enzyme_aurocs": {k: round(float(v), 4) for k, v in per_enz.items()},
        "n_train": int(len(train_idx)),
        "n_test": int(len(val_idx)),
        "total_time_s": round(total_time, 1),
    }


def run_enzyme_loo(model_cls, data):
    """Enzyme LOO: for each of 5 enzymes, train on other 4, test on held-out."""
    logger.info(f"\n  Enzyme LOO")
    df = data["df"]
    labels_enzyme = data["labels_enzyme"]

    loo_results = {}
    t0 = time.time()

    for enz_name in PER_ENZYME_HEADS:
        enz_idx = ENZYME_TO_IDX[enz_name]
        test_mask = labels_enzyme == enz_idx
        train_mask = ~test_mask

        # Also exclude Unknown from training
        unknown_mask = labels_enzyme == ENZYME_TO_IDX["Unknown"]
        train_mask = train_mask & ~unknown_mask

        train_idx = np.where(train_mask)[0]
        val_idx = np.where(test_mask)[0]

        if len(val_idx) < 20:
            logger.info(f"    {enz_name}: too few test sites ({len(val_idx)}), skipping")
            loo_results[enz_name] = {"error": f"only {len(val_idx)} test sites"}
            continue

        # Check that we have both classes in val
        val_labels = data["labels_binary"][val_idx]
        if len(np.unique(val_labels)) < 2:
            logger.info(f"    {enz_name}: only one class in test set, skipping")
            loo_results[enz_name] = {"error": "only one class"}
            continue

        logger.info(f"    {enz_name}: {len(train_idx)} train, {len(val_idx)} test "
                     f"(pos={int(val_labels.sum())}, neg={int((val_labels == 0).sum())})")

        auroc, per_enz = train_and_eval_single_split(model_cls, data, train_idx, val_idx)
        logger.info(f"      {enz_name} holdout AUROC={auroc:.4f}")

        loo_results[enz_name] = {
            "overall_auroc": round(float(auroc), 4),
            "per_enzyme_aurocs": {k: round(float(v), 4) for k, v in per_enz.items()},
            "n_train": int(len(train_idx)),
            "n_test": int(len(val_idx)),
        }

    total_time = time.time() - t0
    logger.info(f"    Enzyme LOO total time: {total_time:.0f}s")
    return loo_results


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
# Phase 1: 3-Fold CV on 12 Combinations
# ---------------------------------------------------------------------------


# All 12 combinations: (combo_name, architecture_cls, training_method)
COMBO_REGISTRY = {
    # A6 + H4
    "A6_T1_H4": (Conv2DBP_H4, "T1"),
    "A6_T4_H4": (Conv2DBP_H4, "T4"),
    "A6_T6_H4": (Conv2DBP_H4, "T6"),
    # A8 + H4
    "A8_T1_H4": (HierarchicalAttention_H4, "T1"),
    "A8_T4_H4": (HierarchicalAttention_H4, "T4"),
    "A8_T6_H4": (HierarchicalAttention_H4, "T6"),
    # A6 + H1
    "A6_T1_H1": (Conv2DBP_H1, "T1"),
    "A6_T4_H1": (Conv2DBP_H1, "T4"),
    "A6_T6_H1": (Conv2DBP_H1, "T6"),
    # A8 + H1
    "A8_T1_H1": (HierarchicalAttention_H1, "T1"),
    "A8_T4_H1": (HierarchicalAttention_H1, "T4"),
    "A8_T6_H1": (HierarchicalAttention_H1, "T6"),
}


def run_phase1(v3_data, v4_data, results):
    """Phase 1: 3-fold CV on all 12 combinations."""
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 1: 3-Fold CV on 12 Combinations")
    logger.info("=" * 80)

    if "phase1" not in results:
        results["phase1"] = {}

    for combo_idx, (combo_name, (model_cls, training_method)) in enumerate(COMBO_REGISTRY.items()):
        if combo_name in results["phase1"] and "error" not in results["phase1"][combo_name]:
            logger.info(f"\n[{combo_idx + 1}/12] {combo_name} — already done, skipping")
            continue

        logger.info(f"\n{'=' * 70}")
        logger.info(f"[{combo_idx + 1}/12] {combo_name} (arch={model_cls.name}, training={training_method})")
        logger.info(f"{'=' * 70}")

        try:
            if training_method == "T1":
                result = train_t1_baseline(model_cls, v3_data, n_folds=3)
            elif training_method == "T4":
                result = train_t4_v4large(model_cls, v3_data, v4_data, n_folds=3)
            elif training_method == "T6":
                result = train_t6_pretext(model_cls, v3_data, n_folds=3)
            else:
                raise ValueError(f"Unknown training method: {training_method}")

            result["combo_name"] = combo_name
            result["architecture"] = model_cls.name
            result["training_method"] = training_method
            results["phase1"][combo_name] = result

            logger.info(f"\n  {combo_name} SUMMARY:")
            logger.info(f"    Overall AUROC: {result['overall_auroc_mean']:.4f} +/- {result['overall_auroc_std']:.4f}")
            logger.info(f"    Mean per-enzyme AUROC: {result['mean_per_enzyme_auroc']:.4f}")
            for enz in PER_ENZYME_HEADS:
                if enz in result["per_enzyme_auroc_mean"]:
                    logger.info(f"    {enz}: {result['per_enzyme_auroc_mean'][enz]:.4f} +/- {result['per_enzyme_auroc_std'].get(enz, 0):.4f}")
            logger.info(f"    Time: {result['total_time_s']:.0f}s")

        except Exception as e:
            logger.error(f"FAILED: {combo_name}: {e}")
            logger.error(traceback.format_exc())
            results["phase1"][combo_name] = {"error": str(e)}

        save_results(results)
        gc.collect()
        if DEVICE.type == "mps":
            torch.mps.empty_cache()

    # Print Phase 1 comparison table
    _print_phase1_table(results)

    return results


def _print_phase1_table(results):
    """Print Phase 1 comparison table sorted by mean per-enzyme AUROC."""
    logger.info("\n" + "=" * 100)
    logger.info("PHASE 1 RESULTS — Ranked by Mean Per-Enzyme AUROC")
    logger.info("=" * 100)

    rows = []
    for combo_name, r in results.get("phase1", {}).items():
        if "error" in r:
            continue
        mean_pe = r.get("mean_per_enzyme_auroc", 0)
        rows.append((combo_name, mean_pe, r))

    rows.sort(key=lambda x: x[1], reverse=True)

    header = f"{'Rank':<5} {'Combo':<18} {'Overall':>10} {'MeanPerEnz':>12}"
    for enz in PER_ENZYME_HEADS:
        header += f" {enz:>10}"
    header += f" {'Time':>8}"
    logger.info(header)
    logger.info("-" * 100)

    for rank, (name, mean_pe, r) in enumerate(rows, 1):
        row = f"{rank:<5} {name:<18} {r['overall_auroc_mean']:>10.4f} {mean_pe:>12.4f}"
        for enz in PER_ENZYME_HEADS:
            val = r.get("per_enzyme_auroc_mean", {}).get(enz, float("nan"))
            if np.isnan(val):
                row += f" {'N/A':>10}"
            else:
                row += f" {val:>10.4f}"
        row += f" {r.get('total_time_s', 0):>7.0f}s"
        logger.info(row)

    logger.info("=" * 100)


# ---------------------------------------------------------------------------
# Phase 2: Top 5 Winners -> 5-Fold + Challenge Splits
# ---------------------------------------------------------------------------


def select_top5(results) -> List[Tuple[str, type]]:
    """Select top 5 combos by mean per-enzyme AUROC from Phase 1."""
    rows = []
    for combo_name, r in results.get("phase1", {}).items():
        if "error" in r:
            continue
        mean_pe = r.get("mean_per_enzyme_auroc", 0)
        rows.append((combo_name, mean_pe))

    rows.sort(key=lambda x: x[1], reverse=True)
    top5 = rows[:5]

    logger.info("\nTop 5 winners for Phase 2:")
    for rank, (name, score) in enumerate(top5, 1):
        logger.info(f"  #{rank}: {name} (mean_per_enzyme_auroc={score:.4f})")

    return [(name, COMBO_REGISTRY[name][0]) for name, _ in top5]


def run_phase2(v3_data, results):
    """Phase 2: 5-fold CV + challenge splits for top 5."""
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 2: Top 5 Winners — 5-Fold CV + Challenge Splits")
    logger.info("=" * 80)

    if "phase2" not in results:
        results["phase2"] = {}

    top5 = select_top5(results)

    for rank, (combo_name, model_cls) in enumerate(top5, 1):
        logger.info(f"\n{'=' * 70}")
        logger.info(f"[{rank}/5] Phase 2: {combo_name}")
        logger.info(f"{'=' * 70}")

        if combo_name not in results["phase2"]:
            results["phase2"][combo_name] = {}

        combo_results = results["phase2"][combo_name]

        # 1. Standard 5-fold CV
        if "5fold_cv" not in combo_results:
            try:
                combo_results["5fold_cv"] = run_5fold_cv(model_cls, v3_data, combo_name)
            except Exception as e:
                logger.error(f"FAILED 5-fold CV for {combo_name}: {e}")
                logger.error(traceback.format_exc())
                combo_results["5fold_cv"] = {"error": str(e)}
            save_results(results)
        else:
            logger.info(f"  5-Fold CV already done for {combo_name}")

        # 2. Gene holdout (chromosome-based)
        if "gene_holdout" not in combo_results:
            try:
                combo_results["gene_holdout"] = run_gene_holdout(model_cls, v3_data)
            except Exception as e:
                logger.error(f"FAILED gene holdout for {combo_name}: {e}")
                logger.error(traceback.format_exc())
                combo_results["gene_holdout"] = {"error": str(e)}
            save_results(results)
        else:
            logger.info(f"  Gene holdout already done for {combo_name}")

        # 3. Chromosome holdout
        if "chr_holdout" not in combo_results:
            try:
                combo_results["chr_holdout"] = run_chr_holdout(model_cls, v3_data)
            except Exception as e:
                logger.error(f"FAILED chr holdout for {combo_name}: {e}")
                logger.error(traceback.format_exc())
                combo_results["chr_holdout"] = {"error": str(e)}
            save_results(results)
        else:
            logger.info(f"  Chr holdout already done for {combo_name}")

        # 4. Enzyme LOO
        if "enzyme_loo" not in combo_results:
            try:
                combo_results["enzyme_loo"] = run_enzyme_loo(model_cls, v3_data)
            except Exception as e:
                logger.error(f"FAILED enzyme LOO for {combo_name}: {e}")
                logger.error(traceback.format_exc())
                combo_results["enzyme_loo"] = {"error": str(e)}
            save_results(results)
        else:
            logger.info(f"  Enzyme LOO already done for {combo_name}")

        gc.collect()
        if DEVICE.type == "mps":
            torch.mps.empty_cache()

    # Print Phase 2 summary
    _print_phase2_table(results)

    return results


def _print_phase2_table(results):
    """Print Phase 2 summary table."""
    logger.info("\n" + "=" * 120)
    logger.info("PHASE 2 RESULTS SUMMARY")
    logger.info("=" * 120)

    header = f"{'Combo':<18} {'5FoldAUROC':>12} {'GeneHoldout':>13} {'ChrHoldout':>12} {'EnzLOO_A3A':>12} {'EnzLOO_A3B':>12} {'EnzLOO_A3G':>12} {'EnzLOO_Both':>13} {'EnzLOO_Neit':>13}"
    logger.info(header)
    logger.info("-" * 120)

    for combo_name, combo_r in results.get("phase2", {}).items():
        # 5-fold
        r5 = combo_r.get("5fold_cv", {})
        auroc_5f = r5.get("overall_auroc_mean", float("nan"))

        # Gene holdout
        rg = combo_r.get("gene_holdout", {})
        auroc_gene = rg.get("overall_auroc_mean", float("nan"))

        # Chr holdout
        rc = combo_r.get("chr_holdout", {})
        auroc_chr = rc.get("overall_auroc", float("nan"))

        # Enzyme LOO
        re = combo_r.get("enzyme_loo", {})
        enz_loo = {}
        for enz in PER_ENZYME_HEADS:
            er = re.get(enz, {})
            enz_loo[enz] = er.get("overall_auroc", float("nan"))

        row = f"{combo_name:<18} {auroc_5f:>12.4f} {auroc_gene:>13.4f} {auroc_chr:>12.4f}"
        for enz in PER_ENZYME_HEADS:
            val = enz_loo.get(enz, float("nan"))
            if isinstance(val, float) and np.isnan(val):
                row += f" {'N/A':>12}"
            else:
                row += f" {val:>12.4f}"
        logger.info(row)

    logger.info("=" * 120)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    logger.info("=" * 80)
    logger.info("WINNER SCREEN FOR RNA EDITING PREDICTION")
    logger.info(f"Device: {DEVICE} | Seed: {SEED}")
    logger.info(f"Phase 1: 12 combos x 3-fold CV")
    logger.info(f"Phase 2: Top 5 -> 5-fold + gene holdout + chr holdout + enzyme LOO")
    logger.info("=" * 80)

    start_time = time.time()

    # Load existing results for incremental resumption
    results = load_results()

    # Load v3 data (always needed)
    v3_data = load_v3_data()

    # Load v4-large data (needed for T4 combos)
    logger.info("\nLoading v4-large dataset for T4 training...")
    v4_data = load_v4_large(v3_data)

    # Phase 1
    results = run_phase1(v3_data, v4_data, results)

    # Free v4 data after Phase 1 (only needed for T4 training)
    del v4_data
    gc.collect()

    # Phase 2
    results = run_phase2(v3_data, results)

    total_time = time.time() - start_time
    results["total_time_s"] = round(total_time, 1)
    save_results(results)

    logger.info(f"\n{'=' * 80}")
    logger.info(f"WINNER SCREEN COMPLETE")
    logger.info(f"Total time: {total_time / 60:.1f} min")
    logger.info(f"Results: {RESULTS_PATH}")
    logger.info(f"{'=' * 80}")


if __name__ == "__main__":
    main()
