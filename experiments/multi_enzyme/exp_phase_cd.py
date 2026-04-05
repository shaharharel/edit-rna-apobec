#!/usr/bin/env python
"""Phase C (head configurations) and Phase D (5-fold + challenge splits) for RNA editing architecture screen.

Phase C: Test 4 head configurations (H1-H4) on A8+T1 (2-fold CV).
Phase D: 5-fold CV on top combinations + challenge splits (gene-holdout, chr-holdout, enzyme-holdout).

Output: experiments/multi_enzyme/outputs/architecture_screen/phase_cd_results.json
Run: /opt/miniconda3/envs/quris/bin/python -u experiments/multi_enzyme/exp_phase_cd.py
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(OUTPUT_DIR / "phase_cd.log", mode="w"),
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

# Stage 1 config
STAGE1_EPOCHS = 10
STAGE1_LR = 1e-3
STAGE1_WEIGHT_DECAY = 1e-4
STAGE1_ENZYME_HEAD_WEIGHT = 0.3
STAGE1_ENZYME_CLS_WEIGHT = 0.1

# Stage 2 config
STAGE2_EPOCHS = 20
STAGE2_LR = 5e-4
STAGE2_WEIGHT_DECAY = 1e-4

# Feature dimensions
D_RNAFM = 640
D_EDIT_DELTA = 640
D_HAND = 40
D_SHARED = 128


# ---------------------------------------------------------------------------
# Data Loading (copied from exp_architecture_screen.py)
# ---------------------------------------------------------------------------


def load_data() -> Dict:
    """Load v3 dataset with all pre-computed features."""
    logger.info("=" * 70)
    logger.info("Loading V3 dataset + pre-computed embeddings")
    logger.info("=" * 70)

    splits_path = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "splits_multi_enzyme_v3_with_negatives.csv"
    df = pd.read_csv(splits_path)
    logger.info(f"  Loaded {len(df)} sites from splits CSV")

    seq_path = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "multi_enzyme_sequences_v3_with_negatives.json"
    with open(seq_path) as f:
        sequences = json.load(f)
    logger.info(f"  Loaded {len(sequences)} sequences")

    loop_path = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "loop_position_per_site_v3.csv"
    loop_df = pd.read_csv(loop_path)
    loop_df = loop_df.drop_duplicates(subset=["site_id"]).set_index("site_id")

    struct_path = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "structure_cache_multi_enzyme_v3.npz"
    struct_data = np.load(struct_path, allow_pickle=True)
    struct_ids = list(struct_data["site_ids"])
    struct_deltas = struct_data["delta_features"]
    structure_delta = {sid: struct_deltas[i] for i, sid in enumerate(struct_ids)}
    logger.info(f"  Loaded structure cache: {len(structure_delta)} sites")

    rnafm_path = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "embeddings" / "rnafm_pooled_v3.pt"
    rnafm_emb = torch.load(rnafm_path, map_location="cpu", weights_only=False)
    logger.info(f"  Loaded RNA-FM embeddings: {len(rnafm_emb)} sites")

    rnafm_edited_path = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "embeddings" / "rnafm_pooled_edited_v3.pt"
    rnafm_edited_emb = torch.load(rnafm_edited_path, map_location="cpu", weights_only=False)
    logger.info(f"  Loaded RNA-FM edited embeddings: {len(rnafm_edited_emb)} sites")

    site_ids = df["site_id"].tolist()
    valid_mask = [sid in sequences for sid in site_ids]
    df = df[valid_mask].reset_index(drop=True)
    site_ids = df["site_id"].tolist()
    n = len(site_ids)
    logger.info(f"  {n} sites with sequences")

    hand_features = build_hand_features(site_ids, sequences, structure_delta, loop_df)
    logger.info(f"  Hand features shape: {hand_features.shape}")

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

    # BP submatrices (cached)
    bp_cache_path = OUTPUT_DIR / "bp_submatrices_v3.npz"
    if bp_cache_path.exists():
        logger.info(f"  Loading cached BP submatrices from {bp_cache_path}")
        bp_data = np.load(bp_cache_path, allow_pickle=True)
        bp_submatrices = bp_data["bp_submatrices"]
        bp_site_ids = list(bp_data["site_ids"])
        if bp_site_ids == site_ids:
            logger.info(f"  BP cache aligned: {bp_submatrices.shape}")
        else:
            logger.info("  BP cache site_ids differ, rebuilding index map...")
            bp_sid_to_idx = {sid: i for i, sid in enumerate(bp_site_ids)}
            aligned = np.zeros((n, 41, 41), dtype=np.float32)
            for i, sid in enumerate(site_ids):
                if sid in bp_sid_to_idx:
                    aligned[i] = bp_submatrices[bp_sid_to_idx[sid]]
            bp_submatrices = aligned
            logger.info(f"  BP realigned: {bp_submatrices.shape}")
    else:
        raise FileNotFoundError(f"BP submatrices cache not found at {bp_cache_path}. Run exp_architecture_screen.py first.")

    # Dot-bracket structures (for pretext labels)
    db_cache_path = OUTPUT_DIR / "dot_brackets_v3.json"
    dot_brackets = {}
    if db_cache_path.exists():
        with open(db_cache_path) as f:
            dot_brackets = json.load(f)
        logger.info(f"  Loaded {len(dot_brackets)} dot-bracket structures")

    # Pretext labels: 1 if center position is unpaired
    pretext_labels = np.zeros(n, dtype=np.float32)
    for i, sid in enumerate(site_ids):
        if sid in dot_brackets:
            db = dot_brackets[sid]
            if len(db) > CENTER:
                if db[CENTER] == ".":
                    pretext_labels[i] = 1.0
                else:
                    pretext_labels[i] = 0.0
            else:
                pretext_labels[i] = 0.5
        else:
            pretext_labels[i] = 0.5

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

    logger.info("Data loading complete.")
    return {
        "site_ids": site_ids,
        "df": df,
        "sequences": sequences,
        "hand_features": hand_features,
        "rnafm_features": rnafm_matrix,
        "rnafm_edited_features": rnafm_edited_matrix,
        "edit_delta_features": edit_delta_matrix,
        "bp_submatrices": bp_submatrices,
        "dot_brackets": dot_brackets,
        "pretext_labels": pretext_labels,
        "labels_binary": labels_binary,
        "labels_enzyme": labels_enzyme,
        "per_enzyme_labels": per_enzyme_labels,
    }


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
    """Dataset for structure pretext task."""

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
# Base HeadsMixin (H1 — current reference)
# ---------------------------------------------------------------------------


class HeadsMixin:
    """Mixin providing binary head, per-enzyme adapters, and enzyme classifier (H1)."""

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


# ---------------------------------------------------------------------------
# H2: Per-enzyme only (no binary head)
# ---------------------------------------------------------------------------


class H2Mixin:
    """Per-enzyme adapters only, no binary head. Binary score = max over adapters."""

    def _init_heads(self):
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
        per_enzyme_logits = [
            self.enzyme_adapters[enz](shared).squeeze(-1)
            for enz in PER_ENZYME_HEADS
        ]
        # Binary logit = max over enzyme adapter logits
        binary_logit = torch.stack(per_enzyme_logits, dim=-1).max(dim=-1).values
        enzyme_cls_logits = self.enzyme_classifier(shared)
        return binary_logit, per_enzyme_logits, enzyme_cls_logits


# ---------------------------------------------------------------------------
# H3: Hierarchical heads (3-stage training)
# ---------------------------------------------------------------------------


class H3Mixin:
    """Hierarchical heads: binary -> enzyme classifier -> per-enzyme adapters."""

    def _init_heads(self):
        self.binary_head = nn.Linear(D_SHARED, 1)
        self.enzyme_classifier = nn.Sequential(
            nn.Linear(D_SHARED, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, N_ENZYMES),
        )
        self.enzyme_adapters = nn.ModuleDict({
            enz: nn.Sequential(
                nn.Linear(D_SHARED, 32),
                nn.GELU(),
                nn.Linear(32, 1),
            )
            for enz in PER_ENZYME_HEADS
        })

    def _apply_heads(self, shared):
        binary_logit = self.binary_head(shared).squeeze(-1)
        per_enzyme_logits = [
            self.enzyme_adapters[enz](shared).squeeze(-1)
            for enz in PER_ENZYME_HEADS
        ]
        enzyme_cls_logits = self.enzyme_classifier(shared)
        return binary_logit, per_enzyme_logits, enzyme_cls_logits


# ---------------------------------------------------------------------------
# H4: Shared + private adapters
# ---------------------------------------------------------------------------


class H4Mixin:
    """Shared encoder + per-enzyme private encoders."""

    def _init_heads(self):
        self.binary_head = nn.Linear(D_SHARED, 1)
        self.enzyme_classifier = nn.Sequential(
            nn.Linear(D_SHARED, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, N_ENZYMES),
        )
        # Per-enzyme: private encoder (128 -> 32) + adapter head (32 -> 1)
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
# Architecture classes: A6 and A8 (copied from exp_architecture_screen.py)
# ---------------------------------------------------------------------------


class Conv2DBP_H1(nn.Module, HeadsMixin):
    """A6 with H1 (reference)."""
    name = "A6_H1"

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.conv_fc = nn.Linear(32 * 4 * 4, 128)
        d_fused = 128 + D_RNAFM + D_HAND
        self.encoder = nn.Sequential(
            nn.Linear(d_fused, 256), nn.GELU(), nn.Dropout(0.3), nn.LayerNorm(256),
            nn.Linear(256, D_SHARED), nn.GELU(), nn.Dropout(0.2),
        )
        self._init_heads()

    def get_shared_params(self):
        return list(self.conv.parameters()) + list(self.conv_fc.parameters()) + list(self.encoder.parameters())

    def get_adapter_params(self):
        params = list(self.binary_head.parameters())
        for enz in PER_ENZYME_HEADS:
            params.extend(self.enzyme_adapters[enz].parameters())
        params.extend(self.enzyme_classifier.parameters())
        return params

    def forward(self, batch):
        bp = batch["bp_submatrix"]
        conv_out = self.conv(bp).flatten(1)
        conv_out = self.conv_fc(conv_out)
        fused = torch.cat([conv_out, batch["rnafm"], batch["hand_feat"]], dim=-1)
        shared = self.encoder(fused)
        return self._apply_heads(shared)

    def encode(self, batch):
        bp = batch["bp_submatrix"]
        conv_out = self.conv(bp).flatten(1)
        conv_out = self.conv_fc(conv_out)
        fused = torch.cat([conv_out, batch["rnafm"], batch["hand_feat"]], dim=-1)
        return self.encoder(fused)


def _make_a8_encoder():
    """Build A8 encoder components (shared by all head variants)."""
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
    """Shared encoding logic for A8 variants."""
    bp = batch["bp_submatrix"].squeeze(1)
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


class HierarchicalAttention_H1(nn.Module, HeadsMixin):
    """A8 with H1 (reference)."""
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


class HierarchicalAttention_H2(nn.Module, H2Mixin):
    """A8 with H2 (per-enzyme only, no binary head)."""
    name = "A8_H2"

    def __init__(self):
        super().__init__()
        (self.local_proj, self.local_pos_enc, self.local_transformer,
         self.local_pool_attn, self.cross_q, self.cross_k, self.cross_v,
         self.encoder) = _make_a8_encoder()
        self._init_heads()

    def get_shared_params(self):
        return _a8_get_shared_params(self)

    def get_adapter_params(self):
        params = []
        for enz in PER_ENZYME_HEADS:
            params.extend(self.enzyme_adapters[enz].parameters())
        params.extend(self.enzyme_classifier.parameters())
        return params

    def forward(self, batch):
        shared = _a8_encode(self, batch)
        return self._apply_heads(shared)

    def encode(self, batch):
        return _a8_encode(self, batch)


class HierarchicalAttention_H3(nn.Module, H3Mixin):
    """A8 with H3 (hierarchical heads — 3-stage training)."""
    name = "A8_H3"

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
    """A8 with H4 (shared + private adapters)."""
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


# A6 variants for Phase D
class Conv2DBP_H2(nn.Module, H2Mixin):
    name = "A6_H2"
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(4),
        )
        self.conv_fc = nn.Linear(32 * 4 * 4, 128)
        d_fused = 128 + D_RNAFM + D_HAND
        self.encoder = nn.Sequential(
            nn.Linear(d_fused, 256), nn.GELU(), nn.Dropout(0.3), nn.LayerNorm(256),
            nn.Linear(256, D_SHARED), nn.GELU(), nn.Dropout(0.2),
        )
        self._init_heads()

    def get_shared_params(self):
        return list(self.conv.parameters()) + list(self.conv_fc.parameters()) + list(self.encoder.parameters())

    def get_adapter_params(self):
        params = []
        for enz in PER_ENZYME_HEADS:
            params.extend(self.enzyme_adapters[enz].parameters())
        params.extend(self.enzyme_classifier.parameters())
        return params

    def forward(self, batch):
        bp = batch["bp_submatrix"]
        conv_out = self.conv(bp).flatten(1)
        conv_out = self.conv_fc(conv_out)
        fused = torch.cat([conv_out, batch["rnafm"], batch["hand_feat"]], dim=-1)
        shared = self.encoder(fused)
        return self._apply_heads(shared)

    def encode(self, batch):
        bp = batch["bp_submatrix"]
        conv_out = self.conv(bp).flatten(1)
        conv_out = self.conv_fc(conv_out)
        fused = torch.cat([conv_out, batch["rnafm"], batch["hand_feat"]], dim=-1)
        return self.encoder(fused)


class Conv2DBP_H3(nn.Module, H3Mixin):
    name = "A6_H3"
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(4),
        )
        self.conv_fc = nn.Linear(32 * 4 * 4, 128)
        d_fused = 128 + D_RNAFM + D_HAND
        self.encoder = nn.Sequential(
            nn.Linear(d_fused, 256), nn.GELU(), nn.Dropout(0.3), nn.LayerNorm(256),
            nn.Linear(256, D_SHARED), nn.GELU(), nn.Dropout(0.2),
        )
        self._init_heads()

    def get_shared_params(self):
        return list(self.conv.parameters()) + list(self.conv_fc.parameters()) + list(self.encoder.parameters())

    def get_adapter_params(self):
        params = list(self.binary_head.parameters())
        for enz in PER_ENZYME_HEADS:
            params.extend(self.enzyme_adapters[enz].parameters())
        params.extend(self.enzyme_classifier.parameters())
        return params

    def forward(self, batch):
        bp = batch["bp_submatrix"]
        conv_out = self.conv(bp).flatten(1)
        conv_out = self.conv_fc(conv_out)
        fused = torch.cat([conv_out, batch["rnafm"], batch["hand_feat"]], dim=-1)
        shared = self.encoder(fused)
        return self._apply_heads(shared)

    def encode(self, batch):
        bp = batch["bp_submatrix"]
        conv_out = self.conv(bp).flatten(1)
        conv_out = self.conv_fc(conv_out)
        fused = torch.cat([conv_out, batch["rnafm"], batch["hand_feat"]], dim=-1)
        return self.encoder(fused)


class Conv2DBP_H4(nn.Module, H4Mixin):
    name = "A6_H4"
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(4),
        )
        self.conv_fc = nn.Linear(32 * 4 * 4, 128)
        d_fused = 128 + D_RNAFM + D_HAND
        self.encoder = nn.Sequential(
            nn.Linear(d_fused, 256), nn.GELU(), nn.Dropout(0.3), nn.LayerNorm(256),
            nn.Linear(256, D_SHARED), nn.GELU(), nn.Dropout(0.2),
        )
        self._init_heads()

    def get_shared_params(self):
        return list(self.conv.parameters()) + list(self.conv_fc.parameters()) + list(self.encoder.parameters())

    def get_adapter_params(self):
        params = list(self.binary_head.parameters())
        for enz in PER_ENZYME_HEADS:
            params.extend(self.private_encoders[enz].parameters())
            params.extend(self.enzyme_adapters[enz].parameters())
        params.extend(self.enzyme_classifier.parameters())
        return params

    def forward(self, batch):
        bp = batch["bp_submatrix"]
        conv_out = self.conv(bp).flatten(1)
        conv_out = self.conv_fc(conv_out)
        fused = torch.cat([conv_out, batch["rnafm"], batch["hand_feat"]], dim=-1)
        shared = self.encoder(fused)
        return self._apply_heads(shared)

    def encode(self, batch):
        bp = batch["bp_submatrix"]
        conv_out = self.conv(bp).flatten(1)
        conv_out = self.conv_fc(conv_out)
        fused = torch.cat([conv_out, batch["rnafm"], batch["hand_feat"]], dim=-1)
        return self.encoder(fused)


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------


def compute_loss(binary_logit, per_enzyme_logits, enzyme_cls_logits, batch,
                 binary_weight=1.0, enzyme_head_weight=STAGE1_ENZYME_HEAD_WEIGHT,
                 enzyme_cls_weight=STAGE1_ENZYME_CLS_WEIGHT):
    """Compute combined loss."""
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

    total = binary_weight * loss_binary + enzyme_head_weight * loss_enzyme_heads + enzyme_cls_weight * loss_cls
    return total, loss_binary.item()


def compute_loss_h2(binary_logit, per_enzyme_logits, enzyme_cls_logits, batch):
    """H2 loss: no binary head, only per-enzyme + enzyme classifier."""
    label_enzyme = batch["label_enzyme"]
    per_enzyme_labels = batch["per_enzyme_labels"]

    loss_enzyme_heads = torch.tensor(0.0, device=per_enzyme_logits[0].device)
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

    label_binary = batch["label_binary"]
    pos_mask = label_binary > 0.5
    loss_cls = torch.tensor(0.0, device=per_enzyme_logits[0].device)
    if pos_mask.sum() > 0:
        loss_cls = F.cross_entropy(enzyme_cls_logits[pos_mask], label_enzyme[pos_mask])

    total = loss_enzyme_heads + 0.1 * loss_cls
    return total, loss_enzyme_heads.item()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_one_epoch(model, loader, optimizer, device, loss_fn=None):
    """Train one epoch."""
    model.train()
    total_loss, total_binary, n_batches = 0.0, 0.0, 0
    if loss_fn is None:
        loss_fn = compute_loss

    for batch in loader:
        batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        optimizer.zero_grad()
        binary_logit, per_enzyme_logits, enzyme_cls_logits = model(batch)
        loss, binary_loss = loss_fn(binary_logit, per_enzyme_logits, enzyme_cls_logits, batch)
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
        batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
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

    return {
        "overall_auroc_mean": round(float(mean_overall), 4),
        "overall_auroc_std": round(float(std_overall), 4),
        "per_enzyme_auroc_mean": {k: round(float(v), 4) for k, v in per_enz_means.items()},
        "per_enzyme_auroc_std": {k: round(float(v), 4) for k, v in per_enz_stds.items()},
        "total_time_s": round(total_time, 1),
        "fold_results": fold_results,
    }


# ---------------------------------------------------------------------------
# Standard 2-stage training (H1, H4)
# ---------------------------------------------------------------------------


def train_standard(model_cls, data, n_folds=2, loss_fn=None):
    """Standard 2-stage training with n-fold CV."""
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

        # Stage 1: Joint training
        logger.info(f"  Stage 1: Joint training ({STAGE1_EPOCHS} epochs)")
        optimizer = torch.optim.Adam([
            {"params": fold_model.get_shared_params(), "lr": STAGE1_LR},
            {"params": fold_model.get_adapter_params(), "lr": STAGE1_LR},
        ], weight_decay=STAGE1_WEIGHT_DECAY)

        for epoch in range(STAGE1_EPOCHS):
            avg_loss, avg_binary = train_one_epoch(fold_model, train_loader, optimizer, DEVICE, loss_fn=loss_fn)
            val_auroc, val_per_enz = evaluate(fold_model, val_loader, DEVICE)
            enz_str = " ".join(f"{e}={v:.3f}" for e, v in val_per_enz.items() if not np.isnan(v))
            logger.info(
                f"    Epoch {epoch + 1:2d} | loss={avg_loss:.4f} bce={avg_binary:.4f} "
                f"| val_auroc={val_auroc:.4f} | {enz_str}"
            )

        # Stage 2: Adapter-only training
        logger.info(f"  Stage 2: Adapter-only ({STAGE2_EPOCHS} epochs)")
        for p in fold_model.get_shared_params():
            p.requires_grad = False

        optimizer2 = torch.optim.Adam(
            fold_model.get_adapter_params(),
            lr=STAGE2_LR, weight_decay=STAGE2_WEIGHT_DECAY,
        )

        for epoch in range(STAGE2_EPOCHS):
            avg_loss, avg_binary = train_one_epoch(fold_model, train_loader, optimizer2, DEVICE, loss_fn=loss_fn)
            val_auroc, val_per_enz = evaluate(fold_model, val_loader, DEVICE)
            if (epoch + 1) % 5 == 0 or epoch == STAGE2_EPOCHS - 1:
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
# H3: Hierarchical 3-stage training
# ---------------------------------------------------------------------------


def train_h3_hierarchical(model_cls, data, n_folds=2):
    """H3: 3-stage hierarchical training."""
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

        # Stage 1: Binary head on all data (10 epochs)
        logger.info(f"  H3 Stage 1: Binary head training (10 epochs)")
        optimizer = torch.optim.Adam([
            {"params": fold_model.get_shared_params(), "lr": STAGE1_LR},
            {"params": list(fold_model.binary_head.parameters()), "lr": STAGE1_LR},
        ], weight_decay=STAGE1_WEIGHT_DECAY)

        for epoch in range(10):
            fold_model.train()
            for batch in train_loader:
                batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                optimizer.zero_grad()
                binary_logit, _, _ = fold_model(batch)
                loss = F.binary_cross_entropy_with_logits(binary_logit, batch["label_binary"])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(fold_model.parameters(), 1.0)
                optimizer.step()

            val_auroc, _ = evaluate(fold_model, val_loader, DEVICE)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(f"    Epoch {epoch + 1:2d} | val_auroc={val_auroc:.4f}")

        # Stage 2: Freeze binary head, train enzyme classifier on positives (10 epochs)
        logger.info(f"  H3 Stage 2: Enzyme classifier on positives (10 epochs)")
        for p in fold_model.binary_head.parameters():
            p.requires_grad = False
        for p in fold_model.get_shared_params():
            p.requires_grad = False

        optimizer2 = torch.optim.Adam(
            list(fold_model.enzyme_classifier.parameters()),
            lr=STAGE2_LR, weight_decay=STAGE2_WEIGHT_DECAY,
        )

        for epoch in range(10):
            fold_model.train()
            for batch in train_loader:
                batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                optimizer2.zero_grad()
                _, _, enzyme_cls_logits = fold_model(batch)
                pos_mask = batch["label_binary"] > 0.5
                if pos_mask.sum() > 0:
                    loss = F.cross_entropy(enzyme_cls_logits[pos_mask], batch["label_enzyme"][pos_mask])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(fold_model.parameters(), 1.0)
                    optimizer2.step()

            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(f"    Epoch {epoch + 1:2d} | enzyme classifier training")

        # Stage 3: Freeze everything, train per-enzyme adapters (10 epochs)
        logger.info(f"  H3 Stage 3: Per-enzyme adapters (10 epochs)")
        for p in fold_model.enzyme_classifier.parameters():
            p.requires_grad = False

        adapter_params = []
        for enz in PER_ENZYME_HEADS:
            adapter_params.extend(fold_model.enzyme_adapters[enz].parameters())

        optimizer3 = torch.optim.Adam(adapter_params, lr=STAGE2_LR, weight_decay=STAGE2_WEIGHT_DECAY)

        for epoch in range(10):
            fold_model.train()
            for batch in train_loader:
                batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                optimizer3.zero_grad()
                _, per_enzyme_logits, _ = fold_model(batch)
                per_enzyme_labels_batch = batch["per_enzyme_labels"]
                loss = torch.tensor(0.0, device=DEVICE)
                n_valid = 0
                for head_idx in range(N_PER_ENZYME):
                    enz_labels = per_enzyme_labels_batch[:, head_idx]
                    mask = enz_labels >= 0
                    if mask.sum() > 0:
                        head_loss = F.binary_cross_entropy_with_logits(
                            per_enzyme_logits[head_idx][mask], enz_labels[mask]
                        )
                        loss = loss + head_loss
                        n_valid += 1
                if n_valid > 0:
                    loss = loss / n_valid
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(fold_model.parameters(), 1.0)
                    optimizer3.step()

            val_auroc, val_per_enz = evaluate(fold_model, val_loader, DEVICE)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                enz_str = " ".join(f"{e}={v:.3f}" for e, v in val_per_enz.items() if not np.isnan(v))
                logger.info(f"    Epoch {epoch + 1:2d} | val_auroc={val_auroc:.4f} | {enz_str}")

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

        del fold_model, optimizer, optimizer2, optimizer3, train_loader, val_loader
        gc.collect()
        if DEVICE.type == "mps":
            torch.mps.empty_cache()

    total_time = time.time() - total_t0
    return _aggregate_fold_results(fold_results, total_time)


# ---------------------------------------------------------------------------
# T6: Structure Pretext Pretraining
# ---------------------------------------------------------------------------


class PretextHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(D_SHARED, 1)

    def forward(self, shared):
        return self.head(shared).squeeze(-1)


def pretrain_pretext(model, pretext_head, train_loader, device, n_epochs=10, lr=1e-3):
    """Pretrain the shared encoder on structure pretext task."""
    model.train()
    pretext_head.train()
    optimizer = torch.optim.Adam(
        list(model.get_shared_params()) + list(pretext_head.parameters()),
        lr=lr, weight_decay=1e-4,
    )

    for epoch in range(n_epochs):
        total_loss, n_batches, n_correct, n_total = 0.0, 0, 0, 0
        for batch in train_loader:
            batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
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
            preds = (torch.sigmoid(logits[mask]) > 0.5).float()
            n_correct += (preds == labels[mask]).sum().item()
            n_total += mask.sum().item()

        avg_loss = total_loss / max(n_batches, 1)
        acc = n_correct / max(n_total, 1)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(f"    Pretext epoch {epoch + 1:2d} | loss={avg_loss:.4f} | acc={acc:.3f}")


def train_with_pretext(model_cls, data, n_folds=2, loss_fn=None):
    """T6: Pretext pretraining -> standard fine-tune."""
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
        logger.info(f"  Stage 0: Pretext pretraining (10 epochs)")
        pretext_ds = PretextDataset(train_idx, data)
        pretext_loader = DataLoader(pretext_ds, batch_size=BATCH_SIZE, shuffle=True,
                                    collate_fn=standard_collate, num_workers=0)
        pretrain_pretext(fold_model, pretext_head, pretext_loader, DEVICE, n_epochs=10)

        del pretext_head
        gc.collect()

        # Re-initialize heads
        fold_model._init_heads()
        fold_model = fold_model.to(DEVICE)

        # Stage 1: Fine-tune joint
        train_ds = ScreenDataset(train_idx, data)
        val_ds = ScreenDataset(val_idx, data)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  collate_fn=standard_collate, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                collate_fn=standard_collate, num_workers=0)

        logger.info(f"  Stage 1: Fine-tune joint ({STAGE1_EPOCHS} epochs)")
        optimizer = torch.optim.Adam([
            {"params": fold_model.get_shared_params(), "lr": STAGE1_LR},
            {"params": fold_model.get_adapter_params(), "lr": STAGE1_LR},
        ], weight_decay=STAGE1_WEIGHT_DECAY)

        for epoch in range(STAGE1_EPOCHS):
            avg_loss, avg_binary = train_one_epoch(fold_model, train_loader, optimizer, DEVICE, loss_fn=loss_fn)
            val_auroc, val_per_enz = evaluate(fold_model, val_loader, DEVICE)
            enz_str = " ".join(f"{e}={v:.3f}" for e, v in val_per_enz.items() if not np.isnan(v))
            logger.info(
                f"    Epoch {epoch + 1:2d} | loss={avg_loss:.4f} bce={avg_binary:.4f} "
                f"| val_auroc={val_auroc:.4f} | {enz_str}"
            )

        # Stage 2: Adapter-only
        logger.info(f"  Stage 2: Adapter-only ({STAGE2_EPOCHS} epochs)")
        for p in fold_model.get_shared_params():
            p.requires_grad = False

        optimizer2 = torch.optim.Adam(
            fold_model.get_adapter_params(),
            lr=STAGE2_LR, weight_decay=STAGE2_WEIGHT_DECAY,
        )

        for epoch in range(STAGE2_EPOCHS):
            avg_loss, avg_binary = train_one_epoch(fold_model, train_loader, optimizer2, DEVICE, loss_fn=loss_fn)
            val_auroc, val_per_enz = evaluate(fold_model, val_loader, DEVICE)
            if (epoch + 1) % 5 == 0 or epoch == STAGE2_EPOCHS - 1:
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
# XGB per-enzyme baseline
# ---------------------------------------------------------------------------


def run_xgb_per_enzyme(data, n_folds=5):
    """XGB per-enzyme: train a separate XGB for each enzyme on its matched pos+neg subset."""
    from xgboost import XGBClassifier

    logger.info(f"\n{'=' * 70}")
    logger.info(f"XGB Per-Enzyme Baseline ({n_folds}-fold CV)")
    logger.info(f"{'=' * 70}")

    n = len(data["labels_binary"])
    hand_features = data["hand_features"]
    labels_binary = data["labels_binary"]
    labels_enzyme = data["labels_enzyme"]
    per_enzyme_labels = data["per_enzyme_labels"]

    strat_key = labels_enzyme * 2 + labels_binary.astype(np.int64)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    fold_results = []
    total_t0 = time.time()

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(n), strat_key)):
        logger.info(f"\n--- Fold {fold + 1}/{n_folds} ---")
        fold_t0 = time.time()

        per_enzyme_aurocs = {}

        for h_idx, enz_name in enumerate(PER_ENZYME_HEADS):
            # Get enzyme-matched indices
            enz_train_mask = per_enzyme_labels[train_idx, h_idx] >= 0
            enz_val_mask = per_enzyme_labels[val_idx, h_idx] >= 0

            train_enz_idx = train_idx[enz_train_mask]
            val_enz_idx = val_idx[enz_val_mask]

            if len(train_enz_idx) < 10 or len(val_enz_idx) < 10:
                per_enzyme_aurocs[enz_name] = float("nan")
                continue

            X_train = hand_features[train_enz_idx]
            y_train = per_enzyme_labels[train_enz_idx, h_idx]
            X_val = hand_features[val_enz_idx]
            y_val = per_enzyme_labels[val_enz_idx, h_idx]

            if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
                per_enzyme_aurocs[enz_name] = float("nan")
                continue

            xgb = XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                random_state=SEED, eval_metric="logloss",
                verbosity=0,
            )
            xgb.fit(X_train, y_train)
            y_pred = xgb.predict_proba(X_val)[:, 1]

            try:
                per_enzyme_aurocs[enz_name] = roc_auc_score(y_val, y_pred)
            except ValueError:
                per_enzyme_aurocs[enz_name] = 0.5

        # Overall AUROC: use max enzyme prediction per sample
        all_probs = np.zeros(len(val_idx))
        for h_idx, enz_name in enumerate(PER_ENZYME_HEADS):
            enz_val_mask = per_enzyme_labels[val_idx, h_idx] >= 0
            val_enz_idx = val_idx[enz_val_mask]
            if len(val_enz_idx) < 10:
                continue
            X_val = hand_features[val_enz_idx]
            enz_train_mask = per_enzyme_labels[train_idx, h_idx] >= 0
            train_enz_idx = train_idx[enz_train_mask]
            if len(train_enz_idx) < 10:
                continue
            X_train = hand_features[train_enz_idx]
            y_train = per_enzyme_labels[train_enz_idx, h_idx]
            if len(np.unique(y_train)) < 2:
                continue
            xgb = XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                random_state=SEED, eval_metric="logloss", verbosity=0,
            )
            xgb.fit(X_train, y_train)
            # Score ALL val samples with this enzyme's model
            probs = xgb.predict_proba(hand_features[val_idx])[:, 1]
            all_probs = np.maximum(all_probs, probs)

        val_labels = labels_binary[val_idx]
        try:
            overall_auroc = roc_auc_score(val_labels, all_probs)
        except ValueError:
            overall_auroc = 0.5

        fold_time = time.time() - fold_t0
        enz_str = " ".join(f"{e}={v:.3f}" for e, v in per_enzyme_aurocs.items() if not np.isnan(v))
        logger.info(f"  Fold {fold + 1} | overall={overall_auroc:.4f} | {enz_str} | time={fold_time:.0f}s")

        fold_results.append({
            "fold": fold + 1,
            "overall_auroc": overall_auroc,
            "per_enzyme_aurocs": per_enzyme_aurocs,
            "time_s": fold_time,
        })

    total_time = time.time() - total_t0
    return _aggregate_fold_results(fold_results, total_time)


# ---------------------------------------------------------------------------
# Challenge Splits
# ---------------------------------------------------------------------------


def run_challenge_gene_holdout(model_cls, data, loss_fn=None):
    """Gene-holdout: split genes into 5 groups, train on 4, test on 1."""
    logger.info(f"\n{'=' * 70}")
    logger.info(f"Challenge: Gene Holdout — {model_cls.__name__}")
    logger.info(f"{'=' * 70}")

    df = data["df"]
    # Extract gene from site_id (format: gene_chr_pos or similar)
    # Try to get gene from the dataframe
    if "gene" in df.columns:
        genes = df["gene"].values
    else:
        # Parse gene from site_id: typically "GENE_chrN_POS_..."
        genes = np.array([sid.split("_")[0] for sid in data["site_ids"]])

    unique_genes = np.unique(genes)
    logger.info(f"  {len(unique_genes)} unique genes")

    # Shuffle and split genes into 5 groups
    rng = np.random.RandomState(SEED)
    gene_perm = rng.permutation(len(unique_genes))
    gene_groups = np.array_split(gene_perm, 5)

    fold_results = []
    total_t0 = time.time()

    for fold_idx, test_gene_indices in enumerate(gene_groups):
        test_genes = set(unique_genes[test_gene_indices])
        test_mask = np.array([g in test_genes for g in genes])
        train_mask = ~test_mask

        train_idx = np.where(train_mask)[0]
        val_idx = np.where(test_mask)[0]

        if len(val_idx) < 10:
            logger.info(f"  Fold {fold_idx + 1}: too few test samples ({len(val_idx)}), skipping")
            continue

        logger.info(f"\n--- Gene Fold {fold_idx + 1}/5 ({len(train_idx)} train, {len(val_idx)} val, {len(test_genes)} test genes) ---")
        fold_t0 = time.time()

        fold_model = model_cls().to(DEVICE)

        train_ds = ScreenDataset(train_idx, data)
        val_ds = ScreenDataset(val_idx, data)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  collate_fn=standard_collate, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                collate_fn=standard_collate, num_workers=0)

        # Stage 1: Joint
        optimizer = torch.optim.Adam([
            {"params": fold_model.get_shared_params(), "lr": STAGE1_LR},
            {"params": fold_model.get_adapter_params(), "lr": STAGE1_LR},
        ], weight_decay=STAGE1_WEIGHT_DECAY)

        for epoch in range(STAGE1_EPOCHS):
            train_one_epoch(fold_model, train_loader, optimizer, DEVICE, loss_fn=loss_fn)

        # Stage 2: Adapter-only
        for p in fold_model.get_shared_params():
            p.requires_grad = False
        optimizer2 = torch.optim.Adam(fold_model.get_adapter_params(), lr=STAGE2_LR, weight_decay=STAGE2_WEIGHT_DECAY)
        for epoch in range(STAGE2_EPOCHS):
            train_one_epoch(fold_model, train_loader, optimizer2, DEVICE, loss_fn=loss_fn)

        final_auroc, final_per_enz = evaluate(fold_model, val_loader, DEVICE)
        fold_time = time.time() - fold_t0
        enz_str = " ".join(f"{e}={v:.3f}" for e, v in final_per_enz.items() if not np.isnan(v))
        logger.info(f"  Gene Fold {fold_idx + 1} | overall={final_auroc:.4f} | {enz_str} | time={fold_time:.0f}s")

        fold_results.append({
            "fold": fold_idx + 1,
            "overall_auroc": final_auroc,
            "per_enzyme_aurocs": final_per_enz,
            "time_s": fold_time,
        })

        del fold_model, optimizer, optimizer2
        gc.collect()
        if DEVICE.type == "mps":
            torch.mps.empty_cache()

    total_time = time.time() - total_t0
    return _aggregate_fold_results(fold_results, total_time)


def run_challenge_chr_holdout(model_cls, data, loss_fn=None):
    """Chromosome holdout: train on chr1-17, test on chr18-22+X."""
    logger.info(f"\n{'=' * 70}")
    logger.info(f"Challenge: Chromosome Holdout — {model_cls.__name__}")
    logger.info(f"{'=' * 70}")

    df = data["df"]
    if "chrom" in df.columns:
        chroms = df["chrom"].values
    else:
        # Parse from site_id
        chroms = np.array([
            next((part for part in sid.split("_") if part.startswith("chr")), "chrUn")
            for sid in data["site_ids"]
        ])

    test_chroms = {"chr18", "chr19", "chr20", "chr21", "chr22", "chrX"}
    test_mask = np.array([c in test_chroms for c in chroms])
    train_mask = ~test_mask

    train_idx = np.where(train_mask)[0]
    val_idx = np.where(test_mask)[0]

    logger.info(f"  Train: {len(train_idx)} (chr1-17), Test: {len(val_idx)} (chr18-22+X)")

    fold_t0 = time.time()
    fold_model = model_cls().to(DEVICE)

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
        train_one_epoch(fold_model, train_loader, optimizer, DEVICE, loss_fn=loss_fn)

    for p in fold_model.get_shared_params():
        p.requires_grad = False
    optimizer2 = torch.optim.Adam(fold_model.get_adapter_params(), lr=STAGE2_LR, weight_decay=STAGE2_WEIGHT_DECAY)
    for epoch in range(STAGE2_EPOCHS):
        train_one_epoch(fold_model, train_loader, optimizer2, DEVICE, loss_fn=loss_fn)

    final_auroc, final_per_enz = evaluate(fold_model, val_loader, DEVICE)
    fold_time = time.time() - fold_t0
    enz_str = " ".join(f"{e}={v:.3f}" for e, v in final_per_enz.items() if not np.isnan(v))
    logger.info(f"  Chr Holdout | overall={final_auroc:.4f} | {enz_str} | time={fold_time:.0f}s")

    del fold_model, optimizer, optimizer2
    gc.collect()
    if DEVICE.type == "mps":
        torch.mps.empty_cache()

    return {
        "overall_auroc_mean": round(float(final_auroc), 4),
        "per_enzyme_auroc_mean": {k: round(float(v), 4) for k, v in final_per_enz.items()},
        "total_time_s": round(fold_time, 1),
    }


def run_challenge_enzyme_holdout(model_cls, data, loss_fn=None):
    """Enzyme LOO holdout: for each enzyme, train on the other 4, predict this one."""
    logger.info(f"\n{'=' * 70}")
    logger.info(f"Challenge: Enzyme LOO Holdout — {model_cls.__name__}")
    logger.info(f"{'=' * 70}")

    labels_enzyme = data["labels_enzyme"]
    labels_binary = data["labels_binary"]

    enzyme_results = {}
    total_t0 = time.time()

    for enz_name in PER_ENZYME_HEADS:
        enz_idx = ENZYME_TO_IDX[enz_name]
        test_mask = labels_enzyme == enz_idx
        train_mask = ~test_mask

        train_idx = np.where(train_mask)[0]
        val_idx = np.where(test_mask)[0]

        n_pos_test = int(labels_binary[val_idx].sum())
        n_neg_test = len(val_idx) - n_pos_test

        if n_pos_test < 5 or n_neg_test < 5:
            logger.info(f"  {enz_name}: too few test samples (pos={n_pos_test}, neg={n_neg_test}), skipping")
            enzyme_results[enz_name] = {"auroc": float("nan"), "n_test": len(val_idx)}
            continue

        logger.info(f"\n--- Enzyme LOO: {enz_name} ({len(train_idx)} train, {len(val_idx)} test: {n_pos_test} pos, {n_neg_test} neg) ---")
        fold_t0 = time.time()

        fold_model = model_cls().to(DEVICE)

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
            train_one_epoch(fold_model, train_loader, optimizer, DEVICE, loss_fn=loss_fn)

        for p in fold_model.get_shared_params():
            p.requires_grad = False
        optimizer2 = torch.optim.Adam(fold_model.get_adapter_params(), lr=STAGE2_LR, weight_decay=STAGE2_WEIGHT_DECAY)
        for epoch in range(STAGE2_EPOCHS):
            train_one_epoch(fold_model, train_loader, optimizer2, DEVICE, loss_fn=loss_fn)

        final_auroc, final_per_enz = evaluate(fold_model, val_loader, DEVICE)
        fold_time = time.time() - fold_t0

        logger.info(f"  {enz_name} holdout AUROC: {final_auroc:.4f} | time={fold_time:.0f}s")
        enzyme_results[enz_name] = {
            "auroc": round(float(final_auroc), 4),
            "n_test": len(val_idx),
            "time_s": round(fold_time, 1),
        }

        del fold_model, optimizer, optimizer2
        gc.collect()
        if DEVICE.type == "mps":
            torch.mps.empty_cache()

    total_time = time.time() - total_t0
    return {"enzyme_results": enzyme_results, "total_time_s": round(total_time, 1)}


# ---------------------------------------------------------------------------
# Results saving
# ---------------------------------------------------------------------------


def save_results(results, path):
    """Save results to JSON, handling NaN."""
    def clean(obj):
        if isinstance(obj, float) and np.isnan(obj):
            return None
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean(v) for v in obj]
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        return obj

    with open(path, "w") as f:
        json.dump(clean(results), f, indent=2)
    logger.info(f"Results saved to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    logger.info("=" * 70)
    logger.info("PHASE C + D: HEAD CONFIGURATIONS AND 5-FOLD CV + CHALLENGE SPLITS")
    logger.info(f"Device: {DEVICE} | Seed: {SEED}")
    logger.info("=" * 70)

    data = load_data()
    n = len(data["labels_binary"])
    logger.info(f"\nTotal sites: {n}")
    logger.info(f"  Positive: {int(data['labels_binary'].sum())}")
    logger.info(f"  Negative: {int((data['labels_binary'] == 0).sum())}")

    all_results = {}
    results_path = OUTPUT_DIR / "phase_cd_results.json"

    # ===================================================================
    # PHASE C: Head Configurations (A8 + T1, 2-fold CV)
    # ===================================================================
    logger.info("\n" + "=" * 70)
    logger.info("PHASE C: HEAD CONFIGURATIONS (A8 + T1, 2-fold CV)")
    logger.info("=" * 70)

    phase_c_results = {}

    # H1: Binary + per-enzyme adapters (reference)
    try:
        logger.info(f"\n{'=' * 70}")
        logger.info("H1: Binary + Per-enzyme Adapters (reference)")
        logger.info(f"{'=' * 70}")
        result = train_standard(HierarchicalAttention_H1, data, n_folds=2)
        phase_c_results["H1_binary_adapters"] = {"architecture": "A8_H1", **result}
        logger.info(f"  H1 Overall: {result['overall_auroc_mean']:.4f} +/- {result['overall_auroc_std']:.4f}")
    except Exception as e:
        logger.error(f"H1 FAILED: {e}\n{traceback.format_exc()}")
        phase_c_results["H1_binary_adapters"] = {"error": str(e)}

    save_results({"phase_c": phase_c_results}, results_path)

    # H2: Per-enzyme only (no binary head)
    try:
        logger.info(f"\n{'=' * 70}")
        logger.info("H2: Per-enzyme Only (no binary head)")
        logger.info(f"{'=' * 70}")
        result = train_standard(HierarchicalAttention_H2, data, n_folds=2, loss_fn=compute_loss_h2)
        phase_c_results["H2_per_enzyme_only"] = {"architecture": "A8_H2", **result}
        logger.info(f"  H2 Overall: {result['overall_auroc_mean']:.4f} +/- {result['overall_auroc_std']:.4f}")
    except Exception as e:
        logger.error(f"H2 FAILED: {e}\n{traceback.format_exc()}")
        phase_c_results["H2_per_enzyme_only"] = {"error": str(e)}

    save_results({"phase_c": phase_c_results}, results_path)

    # H3: Hierarchical heads (3-stage training)
    try:
        logger.info(f"\n{'=' * 70}")
        logger.info("H3: Hierarchical Heads (3-stage training)")
        logger.info(f"{'=' * 70}")
        result = train_h3_hierarchical(HierarchicalAttention_H3, data, n_folds=2)
        phase_c_results["H3_hierarchical"] = {"architecture": "A8_H3", **result}
        logger.info(f"  H3 Overall: {result['overall_auroc_mean']:.4f} +/- {result['overall_auroc_std']:.4f}")
    except Exception as e:
        logger.error(f"H3 FAILED: {e}\n{traceback.format_exc()}")
        phase_c_results["H3_hierarchical"] = {"error": str(e)}

    save_results({"phase_c": phase_c_results}, results_path)

    # H4: Shared + private adapters
    try:
        logger.info(f"\n{'=' * 70}")
        logger.info("H4: Shared + Private Adapters")
        logger.info(f"{'=' * 70}")
        result = train_standard(HierarchicalAttention_H4, data, n_folds=2)
        phase_c_results["H4_shared_private"] = {"architecture": "A8_H4", **result}
        logger.info(f"  H4 Overall: {result['overall_auroc_mean']:.4f} +/- {result['overall_auroc_std']:.4f}")
    except Exception as e:
        logger.error(f"H4 FAILED: {e}\n{traceback.format_exc()}")
        phase_c_results["H4_shared_private"] = {"error": str(e)}

    save_results({"phase_c": phase_c_results}, results_path)

    # Phase C comparison table
    logger.info("\n" + "=" * 90)
    logger.info("PHASE C — HEAD CONFIGURATION COMPARISON (A8 + T1, 2-fold)")
    logger.info("=" * 90)
    header = f"{'Head Config':<30} {'Overall':>10} {'A3A':>8} {'A3B':>8} {'A3G':>8} {'A3A_A3G':>8} {'Neither':>8} {'Time':>8}"
    logger.info(header)
    logger.info("-" * 90)

    for name, r in phase_c_results.items():
        if "error" in r:
            logger.info(f"{name:<30} {'FAILED':>10}")
            continue
        overall = f"{r['overall_auroc_mean']:.4f}"
        pe = r.get("per_enzyme_auroc_mean", {})
        vals = []
        for enz in PER_ENZYME_HEADS:
            v = pe.get(enz)
            vals.append(f"{v:.3f}" if v is not None else "  N/A")
        time_str = f"{r['total_time_s']:.0f}s"
        logger.info(f"{name:<30} {overall:>10} {vals[0]:>8} {vals[1]:>8} {vals[2]:>8} {vals[3]:>8} {vals[4]:>8} {time_str:>8}")
    logger.info("=" * 90)

    # Determine best head config
    best_head = None
    best_auroc = -1
    for name, r in phase_c_results.items():
        if "error" not in r and r.get("overall_auroc_mean", 0) > best_auroc:
            best_auroc = r["overall_auroc_mean"]
            best_head = name

    logger.info(f"\nBest head configuration: {best_head} (AUROC={best_auroc:.4f})")

    # Map best head to model classes
    # For simplicity, we use the head index to select the right variant
    HEAD_TO_A8_CLS = {
        "H1_binary_adapters": HierarchicalAttention_H1,
        "H2_per_enzyme_only": HierarchicalAttention_H2,
        "H3_hierarchical": HierarchicalAttention_H3,
        "H4_shared_private": HierarchicalAttention_H4,
    }
    HEAD_TO_A6_CLS = {
        "H1_binary_adapters": Conv2DBP_H1,
        "H2_per_enzyme_only": Conv2DBP_H2,
        "H3_hierarchical": Conv2DBP_H3,
        "H4_shared_private": Conv2DBP_H4,
    }
    HEAD_TO_LOSS = {
        "H1_binary_adapters": None,
        "H2_per_enzyme_only": compute_loss_h2,
        "H3_hierarchical": None,
        "H4_shared_private": None,
    }

    best_a8_cls = HEAD_TO_A8_CLS.get(best_head, HierarchicalAttention_H1)
    best_a6_cls = HEAD_TO_A6_CLS.get(best_head, Conv2DBP_H1)
    best_loss_fn = HEAD_TO_LOSS.get(best_head, None)
    is_h3 = best_head == "H3_hierarchical"

    # ===================================================================
    # PHASE D: 5-Fold CV + Challenge Splits
    # ===================================================================
    logger.info("\n" + "=" * 70)
    logger.info(f"PHASE D: 5-FOLD CV + CHALLENGE SPLITS (best head: {best_head})")
    logger.info("=" * 70)

    phase_d_results = {"best_head": best_head}

    # --------------- 5-Fold CV ---------------
    logger.info("\n" + "=" * 70)
    logger.info("PHASE D.1: 5-FOLD STRATIFIED CV")
    logger.info("=" * 70)

    fivefold_results = {}

    # 1. A8 + T1 (baseline) + best head
    try:
        logger.info(f"\n{'=' * 70}")
        logger.info(f"5-Fold: A8 + T1 + {best_head}")
        logger.info(f"{'=' * 70}")
        if is_h3:
            result = train_h3_hierarchical(best_a8_cls, data, n_folds=5)
        else:
            result = train_standard(best_a8_cls, data, n_folds=5, loss_fn=best_loss_fn)
        fivefold_results["A8_T1"] = result
        logger.info(f"  A8+T1 Overall: {result['overall_auroc_mean']:.4f} +/- {result['overall_auroc_std']:.4f}")
    except Exception as e:
        logger.error(f"A8+T1 5-fold FAILED: {e}\n{traceback.format_exc()}")
        fivefold_results["A8_T1"] = {"error": str(e)}

    save_results({"phase_c": phase_c_results, "phase_d": {"best_head": best_head, "fivefold": fivefold_results}}, results_path)

    # 2. A8 + T6 (pretext) + best head
    try:
        logger.info(f"\n{'=' * 70}")
        logger.info(f"5-Fold: A8 + T6 (pretext) + {best_head}")
        logger.info(f"{'=' * 70}")
        result = train_with_pretext(best_a8_cls, data, n_folds=5, loss_fn=best_loss_fn)
        fivefold_results["A8_T6"] = result
        logger.info(f"  A8+T6 Overall: {result['overall_auroc_mean']:.4f} +/- {result['overall_auroc_std']:.4f}")
    except Exception as e:
        logger.error(f"A8+T6 5-fold FAILED: {e}\n{traceback.format_exc()}")
        fivefold_results["A8_T6"] = {"error": str(e)}

    save_results({"phase_c": phase_c_results, "phase_d": {"best_head": best_head, "fivefold": fivefold_results}}, results_path)

    # 3. A6 + T1 (baseline) + best head
    try:
        logger.info(f"\n{'=' * 70}")
        logger.info(f"5-Fold: A6 + T1 + {best_head}")
        logger.info(f"{'=' * 70}")
        if is_h3:
            result = train_h3_hierarchical(best_a6_cls, data, n_folds=5)
        else:
            result = train_standard(best_a6_cls, data, n_folds=5, loss_fn=best_loss_fn)
        fivefold_results["A6_T1"] = result
        logger.info(f"  A6+T1 Overall: {result['overall_auroc_mean']:.4f} +/- {result['overall_auroc_std']:.4f}")
    except Exception as e:
        logger.error(f"A6+T1 5-fold FAILED: {e}\n{traceback.format_exc()}")
        fivefold_results["A6_T1"] = {"error": str(e)}

    save_results({"phase_c": phase_c_results, "phase_d": {"best_head": best_head, "fivefold": fivefold_results}}, results_path)

    # 4. A6 + T6 (pretext) + best head
    try:
        logger.info(f"\n{'=' * 70}")
        logger.info(f"5-Fold: A6 + T6 (pretext) + {best_head}")
        logger.info(f"{'=' * 70}")
        result = train_with_pretext(best_a6_cls, data, n_folds=5, loss_fn=best_loss_fn)
        fivefold_results["A6_T6"] = result
        logger.info(f"  A6+T6 Overall: {result['overall_auroc_mean']:.4f} +/- {result['overall_auroc_std']:.4f}")
    except Exception as e:
        logger.error(f"A6+T6 5-fold FAILED: {e}\n{traceback.format_exc()}")
        fivefold_results["A6_T6"] = {"error": str(e)}

    save_results({"phase_c": phase_c_results, "phase_d": {"best_head": best_head, "fivefold": fivefold_results}}, results_path)

    # 5. XGB per-enzyme
    try:
        logger.info(f"\n{'=' * 70}")
        logger.info(f"5-Fold: XGB Per-Enzyme (40d hand features)")
        logger.info(f"{'=' * 70}")
        result = run_xgb_per_enzyme(data, n_folds=5)
        fivefold_results["XGB_per_enzyme"] = result
        logger.info(f"  XGB Overall: {result['overall_auroc_mean']:.4f} +/- {result['overall_auroc_std']:.4f}")
    except Exception as e:
        logger.error(f"XGB 5-fold FAILED: {e}\n{traceback.format_exc()}")
        fivefold_results["XGB_per_enzyme"] = {"error": str(e)}

    save_results({"phase_c": phase_c_results, "phase_d": {"best_head": best_head, "fivefold": fivefold_results}}, results_path)

    phase_d_results["fivefold"] = fivefold_results

    # 5-fold comparison table
    logger.info("\n" + "=" * 100)
    logger.info("PHASE D.1 — 5-FOLD CV COMPARISON")
    logger.info("=" * 100)
    header = f"{'Combination':<25} {'Overall':>12} {'A3A':>10} {'A3B':>10} {'A3G':>10} {'A3A_A3G':>10} {'Neither':>10} {'Time':>8}"
    logger.info(header)
    logger.info("-" * 100)

    for name, r in fivefold_results.items():
        if "error" in r:
            logger.info(f"{name:<25} {'FAILED':>12}")
            continue
        overall = f"{r['overall_auroc_mean']:.4f}+/-{r['overall_auroc_std']:.4f}"
        pe_m = r.get("per_enzyme_auroc_mean", {})
        pe_s = r.get("per_enzyme_auroc_std", {})
        vals = []
        for enz in PER_ENZYME_HEADS:
            m = pe_m.get(enz)
            s = pe_s.get(enz)
            if m is not None and s is not None:
                vals.append(f"{m:.3f}+/-{s:.3f}")
            else:
                vals.append("    N/A   ")
        time_str = f"{r['total_time_s']:.0f}s"
        logger.info(f"{name:<25} {overall:>12} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10} {vals[3]:>10} {vals[4]:>10} {time_str:>8}")
    logger.info("=" * 100)

    # --------------- Challenge Splits (top 2 NN models) ---------------
    logger.info("\n" + "=" * 70)
    logger.info("PHASE D.2: CHALLENGE SPLITS")
    logger.info("=" * 70)

    # Determine top 2 NN models from 5-fold
    nn_models = [(name, r) for name, r in fivefold_results.items()
                 if name != "XGB_per_enzyme" and "error" not in r]
    nn_models.sort(key=lambda x: x[1].get("overall_auroc_mean", 0), reverse=True)
    top2_nn = nn_models[:2]

    challenge_results = {}

    for model_name, _ in top2_nn:
        if "A8" in model_name:
            model_cls = best_a8_cls
        else:
            model_cls = best_a6_cls

        # Determine loss function and training approach
        challenge_loss_fn = best_loss_fn
        use_pretext = "T6" in model_name

        # For pretext models, we still use standard challenge training
        # (pretext adds complexity that may not help with challenge splits)
        # Use standard training for challenge splits even for T6 models

        challenge_results[model_name] = {}

        # Gene holdout
        try:
            logger.info(f"\n{'=' * 70}")
            logger.info(f"Challenge: Gene Holdout — {model_name}")
            logger.info(f"{'=' * 70}")
            result = run_challenge_gene_holdout(model_cls, data, loss_fn=challenge_loss_fn)
            challenge_results[model_name]["gene_holdout"] = result
            logger.info(f"  {model_name} Gene Holdout: {result['overall_auroc_mean']:.4f} +/- {result['overall_auroc_std']:.4f}")
        except Exception as e:
            logger.error(f"Gene holdout FAILED for {model_name}: {e}\n{traceback.format_exc()}")
            challenge_results[model_name]["gene_holdout"] = {"error": str(e)}

        save_results({"phase_c": phase_c_results, "phase_d": {**phase_d_results, "challenge": challenge_results}}, results_path)

        # Chromosome holdout
        try:
            logger.info(f"\n{'=' * 70}")
            logger.info(f"Challenge: Chromosome Holdout — {model_name}")
            logger.info(f"{'=' * 70}")
            result = run_challenge_chr_holdout(model_cls, data, loss_fn=challenge_loss_fn)
            challenge_results[model_name]["chr_holdout"] = result
            logger.info(f"  {model_name} Chr Holdout: {result['overall_auroc_mean']:.4f}")
        except Exception as e:
            logger.error(f"Chr holdout FAILED for {model_name}: {e}\n{traceback.format_exc()}")
            challenge_results[model_name]["chr_holdout"] = {"error": str(e)}

        save_results({"phase_c": phase_c_results, "phase_d": {**phase_d_results, "challenge": challenge_results}}, results_path)

        # Enzyme LOO holdout
        try:
            logger.info(f"\n{'=' * 70}")
            logger.info(f"Challenge: Enzyme LOO Holdout — {model_name}")
            logger.info(f"{'=' * 70}")
            result = run_challenge_enzyme_holdout(model_cls, data, loss_fn=challenge_loss_fn)
            challenge_results[model_name]["enzyme_holdout"] = result
            for enz, info in result["enzyme_results"].items():
                auroc = info.get("auroc", "N/A")
                logger.info(f"  {model_name} Enzyme LOO {enz}: {auroc}")
        except Exception as e:
            logger.error(f"Enzyme holdout FAILED for {model_name}: {e}\n{traceback.format_exc()}")
            challenge_results[model_name]["enzyme_holdout"] = {"error": str(e)}

        save_results({"phase_c": phase_c_results, "phase_d": {**phase_d_results, "challenge": challenge_results}}, results_path)

    phase_d_results["challenge"] = challenge_results

    # Challenge comparison table
    logger.info("\n" + "=" * 90)
    logger.info("PHASE D.2 — CHALLENGE SPLIT COMPARISON")
    logger.info("=" * 90)

    for model_name, results in challenge_results.items():
        logger.info(f"\n  {model_name}:")
        for split_name, r in results.items():
            if "error" in r:
                logger.info(f"    {split_name}: FAILED")
            elif split_name == "enzyme_holdout":
                for enz, info in r.get("enzyme_results", {}).items():
                    logger.info(f"    {split_name} {enz}: AUROC={info.get('auroc', 'N/A')}")
            else:
                auroc = r.get("overall_auroc_mean", "N/A")
                logger.info(f"    {split_name}: AUROC={auroc}")

    logger.info("=" * 90)

    # Save final results
    final_results = {
        "phase_c": phase_c_results,
        "phase_d": phase_d_results,
    }
    save_results(final_results, results_path)

    logger.info("\n" + "=" * 70)
    logger.info("PHASE C + D COMPLETE")
    logger.info(f"Results saved to {results_path}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
