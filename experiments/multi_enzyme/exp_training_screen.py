#!/usr/bin/env python
"""Training methods screen (Phase B) for RNA editing architecture search.

Tests training methods T1-T5 on architectures A1, A2, A5, A6, A8:
  T1. Baseline (v3, 1:1)           — results from Phase A
  T2. v4-random (1:10 ratio)       — 75,640 random exonic C negatives
  T3. v4-hard (TC+loop, 1:5)       — 37,820 TC+loop negatives
  T4. v4-large (1:50 mixed)         — 378,200 mixed negatives
  T5. Hard negative curriculum      — random→mix→hard over 30 epochs

Output: experiments/multi_enzyme/outputs/architecture_screen/
Run: /opt/miniconda3/envs/quris/bin/python -u experiments/multi_enzyme/exp_training_screen.py
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
from torch.utils.data import DataLoader, Dataset, Subset

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "multi_enzyme"))

from data.apobec_feature_extraction import build_hand_features

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "multi_enzyme" / "outputs" / "architecture_screen"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(OUTPUT_DIR / "training_screen.log", mode="w"),
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

N_FOLDS = 2
BATCH_SIZE = 64
BATCH_SIZE_LARGE = 256  # For v4-large

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

# Reduced epochs for v4-large
STAGE1_EPOCHS_LARGE = 5
STAGE2_EPOCHS_LARGE = 10

# Feature dimensions
D_RNAFM = 640
D_EDIT_DELTA = 640
D_HAND = 40
D_SHARED = 128


# ---------------------------------------------------------------------------
# Architecture classes — copied from exp_architecture_screen.py
# ---------------------------------------------------------------------------


class HeadsMixin:
    """Mixin providing binary head, per-enzyme adapters, and enzyme classifier."""

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
        """shared: [B, 128] -> binary_logit, per_enzyme_logits, enzyme_cls_logits."""
        binary_logit = self.binary_head(shared).squeeze(-1)
        per_enzyme_logits = [
            self.enzyme_adapters[enz](shared).squeeze(-1)
            for enz in PER_ENZYME_HEADS
        ]
        enzyme_cls_logits = self.enzyme_classifier(shared)
        return binary_logit, per_enzyme_logits, enzyme_cls_logits


class Phase3Baseline(nn.Module, HeadsMixin):
    """A1: [rnafm(640) | delta(640) | hand(40)] = 1320 -> MLP -> heads."""

    name = "A1_Phase3Baseline"

    def __init__(self):
        super().__init__()
        d_in = D_RNAFM + D_EDIT_DELTA + D_HAND
        self.encoder = nn.Sequential(
            nn.Linear(d_in, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.LayerNorm(256),
            nn.Linear(256, D_SHARED),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self._init_heads()

    def get_shared_params(self):
        return list(self.encoder.parameters())

    def get_adapter_params(self):
        params = list(self.binary_head.parameters())
        for enz in PER_ENZYME_HEADS:
            params.extend(self.enzyme_adapters[enz].parameters())
        params.extend(self.enzyme_classifier.parameters())
        return params

    def forward(self, batch):
        x = torch.cat([batch["rnafm"], batch["edit_delta"], batch["hand_feat"]], dim=-1)
        shared = self.encoder(x)
        return self._apply_heads(shared)


class NoDelta(nn.Module, HeadsMixin):
    """A2: [rnafm(640) | hand(40)] = 680 -> MLP -> heads."""

    name = "A2_NoDelta"

    def __init__(self):
        super().__init__()
        d_in = D_RNAFM + D_HAND
        self.encoder = nn.Sequential(
            nn.Linear(d_in, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.LayerNorm(256),
            nn.Linear(256, D_SHARED),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self._init_heads()

    def get_shared_params(self):
        return list(self.encoder.parameters())

    def get_adapter_params(self):
        params = list(self.binary_head.parameters())
        for enz in PER_ENZYME_HEADS:
            params.extend(self.enzyme_adapters[enz].parameters())
        params.extend(self.enzyme_classifier.parameters())
        return params

    def forward(self, batch):
        x = torch.cat([batch["rnafm"], batch["hand_feat"]], dim=-1)
        shared = self.encoder(x)
        return self._apply_heads(shared)


class GatedFusion(nn.Module, HeadsMixin):
    """A5: Gated fusion: hand gates RNA-FM, RNA-FM gates delta."""

    name = "A5_GatedFusion"

    def __init__(self):
        super().__init__()
        self.gate_rnafm = nn.Linear(D_HAND, D_RNAFM)
        self.gate_delta = nn.Linear(D_RNAFM, D_EDIT_DELTA)
        self.hand_proj = nn.Linear(D_HAND, D_SHARED)

        d_fused = D_RNAFM + D_EDIT_DELTA + D_SHARED
        self.encoder = nn.Sequential(
            nn.Linear(d_fused, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.LayerNorm(256),
            nn.Linear(256, D_SHARED),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self._init_heads()

    def get_shared_params(self):
        return (
            list(self.gate_rnafm.parameters()) + list(self.gate_delta.parameters()) +
            list(self.hand_proj.parameters()) + list(self.encoder.parameters())
        )

    def get_adapter_params(self):
        params = list(self.binary_head.parameters())
        for enz in PER_ENZYME_HEADS:
            params.extend(self.enzyme_adapters[enz].parameters())
        params.extend(self.enzyme_classifier.parameters())
        return params

    def forward(self, batch):
        rnafm = batch["rnafm"]
        delta = batch["edit_delta"]
        hand = batch["hand_feat"]

        g_rnafm = torch.sigmoid(self.gate_rnafm(hand))
        g_delta = torch.sigmoid(self.gate_delta(rnafm))

        gated_rnafm = g_rnafm * rnafm
        gated_delta = g_delta * delta
        hand_out = self.hand_proj(hand)

        fused = torch.cat([gated_rnafm, gated_delta, hand_out], dim=-1)
        shared = self.encoder(fused)
        return self._apply_heads(shared)


class Conv2DBP(nn.Module, HeadsMixin):
    """A6: Conv2D on 41x41 BP probability matrix + rnafm + hand -> heads."""

    name = "A6_Conv2DBP"

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
            nn.Linear(d_fused, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.LayerNorm(256),
            nn.Linear(256, D_SHARED),
            nn.GELU(),
            nn.Dropout(0.2),
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
        conv_out = self.conv(bp)
        conv_out = conv_out.flatten(1)
        conv_out = self.conv_fc(conv_out)

        fused = torch.cat([conv_out, batch["rnafm"], batch["hand_feat"]], dim=-1)
        shared = self.encoder(fused)
        return self._apply_heads(shared)


class HierarchicalAttention(nn.Module, HeadsMixin):
    """A8: Two-level attention: local transformer on BP + cross-attention to RNA-FM."""

    name = "A8_HierarchicalAttention"

    def __init__(self):
        super().__init__()
        d_local = 41
        self.local_proj = nn.Linear(d_local, 64)
        self.local_pos_enc = nn.Parameter(torch.randn(41, 64) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128,
            dropout=0.1, activation="gelu", batch_first=True,
        )
        self.local_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.local_pool_attn = nn.Linear(64, 1)

        self.cross_q = nn.Linear(64, 64)
        self.cross_k = nn.Linear(D_RNAFM, 64)
        self.cross_v = nn.Linear(D_RNAFM, 64)

        d_fused = 64 + 64 + D_RNAFM + D_HAND
        self.encoder = nn.Sequential(
            nn.Linear(d_fused, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.LayerNorm(256),
            nn.Linear(256, D_SHARED),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self._init_heads()

    def get_shared_params(self):
        return (
            list(self.local_proj.parameters()) + [self.local_pos_enc] +
            list(self.local_transformer.parameters()) +
            list(self.local_pool_attn.parameters()) +
            list(self.cross_q.parameters()) + list(self.cross_k.parameters()) +
            list(self.cross_v.parameters()) + list(self.encoder.parameters())
        )

    def get_adapter_params(self):
        params = list(self.binary_head.parameters())
        for enz in PER_ENZYME_HEADS:
            params.extend(self.enzyme_adapters[enz].parameters())
        params.extend(self.enzyme_classifier.parameters())
        return params

    def forward(self, batch):
        bp = batch["bp_submatrix"].squeeze(1)  # [B, 41, 41]
        rnafm = batch["rnafm"]
        hand = batch["hand_feat"]

        local_in = self.local_proj(bp) + self.local_pos_enc.unsqueeze(0)
        local_out = self.local_transformer(local_in)

        attn_w = torch.softmax(self.local_pool_attn(local_out), dim=1)
        local_repr = (local_out * attn_w).sum(dim=1)

        q = self.cross_q(local_repr).unsqueeze(1)
        k = self.cross_k(rnafm).unsqueeze(1)
        v = self.cross_v(rnafm).unsqueeze(1)

        attn_scores = (q * k).sum(-1) / math.sqrt(64)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        cross_repr = (attn_weights.unsqueeze(-1) * v).squeeze(1)

        fused = torch.cat([local_repr, cross_repr, rnafm, hand], dim=-1)
        shared = self.encoder(fused)
        return self._apply_heads(shared)


ARCHITECTURE_CLASSES = {
    "A1_Phase3Baseline": Phase3Baseline,
    "A2_NoDelta": NoDelta,
    "A5_GatedFusion": GatedFusion,
    "A6_Conv2DBP": Conv2DBP,
    "A8_HierarchicalAttention": HierarchicalAttention,
}


# ---------------------------------------------------------------------------
# Dataset class
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


def standard_collate(batch_list):
    """Collate for ScreenDataset: stack all tensors."""
    result = {}
    for key in batch_list[0]:
        vals = [b[key] for b in batch_list]
        if isinstance(vals[0], torch.Tensor):
            result[key] = torch.stack(vals)
        else:
            result[key] = vals
    return result


# ---------------------------------------------------------------------------
# Data loading — v3 base + v4 variants
# ---------------------------------------------------------------------------


def load_v3_data() -> Dict:
    """Load the v3 base dataset (same as Phase A)."""
    logger.info("=" * 70)
    logger.info("Loading V3 base dataset")
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

    # RNA-FM embeddings
    rnafm_path = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "embeddings" / "rnafm_pooled_v3.pt"
    rnafm_emb = torch.load(rnafm_path, map_location="cpu", weights_only=False)

    rnafm_edited_path = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "embeddings" / "rnafm_pooled_edited_v3.pt"
    rnafm_edited_emb = torch.load(rnafm_edited_path, map_location="cpu", weights_only=False)

    # BP submatrix cache
    bp_cache_path = OUTPUT_DIR / "bp_submatrices_v3.npz"
    bp_submatrices_v3 = None
    bp_site_ids_v3 = None
    if bp_cache_path.exists():
        bp_data = np.load(bp_cache_path, allow_pickle=True)
        bp_submatrices_v3 = bp_data["bp_submatrices"]
        bp_site_ids_v3 = list(bp_data["site_ids"])
        logger.info(f"  BP cache: {bp_submatrices_v3.shape}")

    # Filter to sites with sequences
    site_ids = df["site_id"].tolist()
    valid_mask = [sid in sequences for sid in site_ids]
    df = df[valid_mask].reset_index(drop=True)
    site_ids = df["site_id"].tolist()
    n = len(site_ids)
    logger.info(f"  {n} v3 sites with sequences")

    # Hand features (40-dim)
    hand_features = build_hand_features(site_ids, sequences, structure_delta, loop_df)

    # RNA-FM features
    rnafm_matrix = np.zeros((n, D_RNAFM), dtype=np.float32)
    rnafm_edited_matrix = np.zeros((n, D_RNAFM), dtype=np.float32)
    for i, sid in enumerate(site_ids):
        if sid in rnafm_emb:
            rnafm_matrix[i] = rnafm_emb[sid].numpy()
        if sid in rnafm_edited_emb:
            rnafm_edited_matrix[i] = rnafm_edited_emb[sid].numpy()
    edit_delta_matrix = rnafm_edited_matrix - rnafm_matrix

    # BP submatrices — align to current site_ids
    bp_sub = np.zeros((n, 41, 41), dtype=np.float32)
    if bp_submatrices_v3 is not None and bp_site_ids_v3 is not None:
        if bp_site_ids_v3 == site_ids:
            bp_sub = bp_submatrices_v3
        else:
            bp_sid_to_idx = {sid: i for i, sid in enumerate(bp_site_ids_v3)}
            for i, sid in enumerate(site_ids):
                if sid in bp_sid_to_idx:
                    bp_sub[i] = bp_submatrices_v3[bp_sid_to_idx[sid]]
    logger.info(f"  BP submatrices: {bp_sub.shape}")

    # Labels
    labels_binary = df["is_edited"].values.astype(np.float32)
    labels_enzyme = np.array([ENZYME_TO_IDX.get(e, 5) for e in df["enzyme"].values], dtype=np.int64)

    # Per-enzyme labels
    per_enzyme_labels = np.full((n, N_PER_ENZYME), -1, dtype=np.float32)
    for head_idx, enz_name in enumerate(PER_ENZYME_HEADS):
        enz_idx = ENZYME_TO_IDX[enz_name]
        pos_mask = (labels_binary == 1) & (labels_enzyme == enz_idx)
        per_enzyme_labels[pos_mask, head_idx] = 1.0
        neg_mask = (labels_binary == 0) & (labels_enzyme == enz_idx)
        per_enzyme_labels[neg_mask, head_idx] = 0.0

    del rnafm_emb, rnafm_edited_emb
    gc.collect()

    logger.info(f"  Positives: {int(labels_binary.sum())}, Negatives: {int((labels_binary == 0).sum())}")

    return {
        "site_ids": site_ids,
        "df": df,
        "sequences": sequences,
        "hand_features": hand_features,
        "rnafm_features": rnafm_matrix,
        "edit_delta_features": edit_delta_matrix,
        "bp_submatrices": bp_sub,
        "labels_binary": labels_binary,
        "labels_enzyme": labels_enzyme,
        "per_enzyme_labels": per_enzyme_labels,
        "structure_delta": structure_delta,
        "loop_df": loop_df,
    }


def load_v4_variant(variant: str, v3_data: Dict) -> Dict:
    """Load a v4 dataset variant (random, hard, or large).

    Positives are the SAME as v3. Negatives come from v4 splits.
    Features for v4 negatives: RNA-FM from v4_negatives.pt, motif from sequence,
    structure/loop/BP set to zeros (not cached for v4 negatives).

    Args:
        variant: One of "random", "hard", "large"
        v3_data: The loaded v3 data dict (for reusing positives and shared resources)

    Returns:
        Data dict in the same format as v3, but with v4 negatives appended.
    """
    logger.info(f"  Loading v4-{variant} negatives...")

    variant_map = {
        "random": "splits_v4_random_negatives.csv",
        "hard": "splits_v4_hard_negatives.csv",
        "large": "splits_v4_large_negatives.csv",
    }
    seq_map = {
        "random": "multi_enzyme_sequences_v4_random.json",
        "hard": "multi_enzyme_sequences_v4_hard.json",
        "large": "multi_enzyme_sequences_v4_large.json",
    }

    splits_path = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / variant_map[variant]
    df_v4 = pd.read_csv(splits_path, low_memory=False)

    # Extract only the negatives from v4
    df_v4_neg = df_v4[df_v4["is_edited"] == 0].reset_index(drop=True)
    n_neg = len(df_v4_neg)
    logger.info(f"  v4-{variant} negatives: {n_neg}")

    # Load v4 sequences (negatives only)
    seq_path = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / seq_map[variant]
    with open(seq_path) as f:
        v4_sequences = json.load(f)

    # Load v4 RNA-FM embeddings (shared across all v4 variants)
    rnafm_v4_path = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "embeddings" / "rnafm_pooled_v4_negatives.pt"
    rnafm_v4 = torch.load(rnafm_v4_path, map_location="cpu", weights_only=False)

    rnafm_edited_v4_path = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "embeddings" / "rnafm_pooled_edited_v4_negatives.pt"
    rnafm_edited_v4 = torch.load(rnafm_edited_v4_path, map_location="cpu", weights_only=False)
    logger.info(f"  v4 RNA-FM embeddings: {len(rnafm_v4)} orig, {len(rnafm_edited_v4)} edited")

    # --- Build feature arrays for v4 negatives ---
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

    # Hand features for negatives: motif from sequence (24-dim), loop+struct = zeros
    # Build combined sequence dict for hand feature extraction
    all_neg_sequences = {}
    for sid in neg_site_ids:
        if sid in v4_sequences:
            all_neg_sequences[sid] = v4_sequences[sid]
        else:
            all_neg_sequences[sid] = "N" * 201

    # Build hand features — structure_delta and loop_df will miss v4 negatives,
    # so those dims will be zeros (which is the correct fallback)
    neg_hand_features = build_hand_features(
        neg_site_ids, all_neg_sequences,
        v3_data["structure_delta"], v3_data["loop_df"]
    )

    # BP submatrices: zeros for v4 negatives
    neg_bp = np.zeros((n_neg, 41, 41), dtype=np.float32)

    # Labels: all negatives
    neg_labels_binary = np.zeros(n_neg, dtype=np.float32)
    # Enzyme label: use a special "all" index — map to Unknown (5) for enzyme classifier
    neg_labels_enzyme = np.full(n_neg, 5, dtype=np.int64)

    # Per-enzyme labels: -1 for all v4 negatives (excluded from per-enzyme loss)
    neg_per_enzyme_labels = np.full((n_neg, N_PER_ENZYME), -1, dtype=np.float32)

    # --- Concatenate v3 positives + v3 negatives + v4 negatives ---
    # v3 data already has all v3 sites (pos + neg)
    n_v3 = len(v3_data["site_ids"])

    combined_site_ids = v3_data["site_ids"] + neg_site_ids
    combined_rnafm = np.concatenate([v3_data["rnafm_features"], neg_rnafm], axis=0)
    combined_delta = np.concatenate([v3_data["edit_delta_features"], neg_edit_delta], axis=0)
    combined_hand = np.concatenate([v3_data["hand_features"], neg_hand_features], axis=0)
    combined_bp = np.concatenate([v3_data["bp_submatrices"], neg_bp], axis=0)
    combined_binary = np.concatenate([v3_data["labels_binary"], neg_labels_binary], axis=0)
    combined_enzyme = np.concatenate([v3_data["labels_enzyme"], neg_labels_enzyme], axis=0)
    combined_per_enzyme = np.concatenate([v3_data["per_enzyme_labels"], neg_per_enzyme_labels], axis=0)

    n_total = len(combined_site_ids)
    logger.info(f"  Combined dataset: {n_total} sites ({int(combined_binary.sum())} pos, {int((combined_binary == 0).sum())} neg)")

    del rnafm_v4, rnafm_edited_v4
    gc.collect()

    return {
        "site_ids": combined_site_ids,
        "hand_features": combined_hand,
        "rnafm_features": combined_rnafm,
        "edit_delta_features": combined_delta,
        "bp_submatrices": combined_bp,
        "labels_binary": combined_binary,
        "labels_enzyme": combined_enzyme,
        "per_enzyme_labels": combined_per_enzyme,
        "n_v3": n_v3,  # Index boundary: [0, n_v3) are v3 sites, [n_v3, n_total) are v4 negatives
    }


# ---------------------------------------------------------------------------
# Training / Evaluation (same logic as Phase A)
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
    all_probs, all_labels, all_enzymes = [], [], []
    all_per_enzyme_probs = [[] for _ in range(N_PER_ENZYME)]
    all_per_enzyme_labels_list = [[] for _ in range(N_PER_ENZYME)]

    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        binary_logit, per_enzyme_logits, _ = model(batch)
        probs = torch.sigmoid(binary_logit).cpu().numpy()
        labels = batch["label_binary"].cpu().numpy()
        enzymes = batch["label_enzyme"].cpu().numpy()
        per_enz_labels = batch["per_enzyme_labels"].cpu().numpy()

        all_probs.extend(probs)
        all_labels.extend(labels)
        all_enzymes.extend(enzymes)

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
# Training method runners
# ---------------------------------------------------------------------------


def run_standard_training(
    arch_cls, arch_name: str, data: Dict,
    stage1_epochs: int = STAGE1_EPOCHS,
    stage2_epochs: int = STAGE2_EPOCHS,
    batch_size: int = BATCH_SIZE,
    eval_data: Optional[Dict] = None,
) -> Dict:
    """Standard 2-fold CV training (used by T2, T3, T4).

    Trains on `data`, evaluates on `eval_data` if provided (for per-enzyme AUROCs
    on v3-matched subsets), otherwise evaluates on the validation fold of `data`.
    """
    n = len(data["labels_binary"])

    # Stratify on binary label only (v4 negatives have label_enzyme=5)
    strat_key = data["labels_binary"].astype(np.int64)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    fold_results = []
    total_t0 = time.time()

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(n), strat_key)):
        logger.info(f"\n--- Fold {fold + 1}/{N_FOLDS} ({len(train_idx)} train, {len(val_idx)} val) ---")
        fold_t0 = time.time()

        fold_model = arch_cls().to(DEVICE)

        train_ds = ScreenDataset(train_idx, data)
        val_ds = ScreenDataset(val_idx, data)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  collate_fn=standard_collate, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                collate_fn=standard_collate, num_workers=0)

        # For per-enzyme eval, build a loader from the v3-matched validation subset
        if eval_data is not None:
            n_v3 = data.get("n_v3", n)
            # Validation indices that fall within v3 range
            v3_val_idx = val_idx[val_idx < n_v3]
            if len(v3_val_idx) > 0:
                eval_ds = ScreenDataset(v3_val_idx, data)
                eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False,
                                         collate_fn=standard_collate, num_workers=0)
            else:
                eval_loader = val_loader
        else:
            eval_loader = val_loader

        # Stage 1: Joint training
        logger.info(f"  Stage 1: Joint training ({stage1_epochs} epochs, lr={STAGE1_LR})")
        optimizer = torch.optim.Adam([
            {"params": fold_model.get_shared_params(), "lr": STAGE1_LR},
            {"params": fold_model.get_adapter_params(), "lr": STAGE1_LR},
        ], weight_decay=STAGE1_WEIGHT_DECAY)

        for epoch in range(stage1_epochs):
            avg_loss, avg_binary = train_one_epoch(fold_model, train_loader, optimizer, DEVICE)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                val_auroc, val_per_enz = evaluate(fold_model, val_loader, DEVICE)
                enz_str = " ".join(f"{e}={v:.3f}" for e, v in val_per_enz.items() if not np.isnan(v))
                logger.info(
                    f"    Epoch {epoch + 1:2d} | loss={avg_loss:.4f} bce={avg_binary:.4f} "
                    f"| val_auroc={val_auroc:.4f} | {enz_str}"
                )

        # Stage 2: Adapter-only training
        logger.info(f"  Stage 2: Adapter-only ({stage2_epochs} epochs, lr={STAGE2_LR})")
        for p in fold_model.get_shared_params():
            p.requires_grad = False

        optimizer2 = torch.optim.Adam(
            fold_model.get_adapter_params(),
            lr=STAGE2_LR, weight_decay=STAGE2_WEIGHT_DECAY,
        )

        for epoch in range(stage2_epochs):
            avg_loss, avg_binary = train_one_epoch(fold_model, train_loader, optimizer2, DEVICE)
            if (epoch + 1) % 5 == 0 or epoch == stage2_epochs - 1:
                val_auroc, val_per_enz = evaluate(fold_model, val_loader, DEVICE)
                enz_str = " ".join(f"{e}={v:.3f}" for e, v in val_per_enz.items() if not np.isnan(v))
                logger.info(
                    f"    Epoch {stage1_epochs + epoch + 1:2d} | loss={avg_loss:.4f} "
                    f"| val_auroc={val_auroc:.4f} | {enz_str}"
                )

        # Final evaluation — overall on full val set, per-enzyme on v3-matched subset
        final_auroc, _ = evaluate(fold_model, val_loader, DEVICE)
        _, final_per_enz = evaluate(fold_model, eval_loader, DEVICE)

        fold_time = time.time() - fold_t0
        logger.info(
            f"  Fold {fold + 1} FINAL: overall_auroc={final_auroc:.4f} | time={fold_time:.0f}s"
        )
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


def run_curriculum_training(
    arch_cls, arch_name: str,
    data_random: Dict, data_hard: Dict, v3_data: Dict,
) -> Dict:
    """T5: Hard negative curriculum training.

    Epoch 1-5:   train on v4-random negatives (easy)
    Epoch 6-10:  train on 50/50 mix of random + hard
    Epoch 11-30: train on v4-hard negatives only

    Total: 30 epochs (10 stage1 + 20 stage2), curriculum applied to both stages.
    """
    n_random = len(data_random["labels_binary"])
    n_hard = len(data_hard["labels_binary"])
    n_v3_random = data_random.get("n_v3", n_random)
    n_v3_hard = data_hard.get("n_v3", n_hard)

    # We stratify on the random dataset (it's used for fold splitting)
    strat_key = data_random["labels_binary"].astype(np.int64)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    # Also need consistent fold splitting for hard dataset
    strat_key_hard = data_hard["labels_binary"].astype(np.int64)
    skf_hard = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    fold_results = []
    total_t0 = time.time()

    folds_random = list(skf.split(np.zeros(n_random), strat_key))
    folds_hard = list(skf_hard.split(np.zeros(n_hard), strat_key_hard))

    for fold in range(N_FOLDS):
        train_idx_random, val_idx_random = folds_random[fold]
        train_idx_hard, val_idx_hard = folds_hard[fold]

        logger.info(f"\n--- Fold {fold + 1}/{N_FOLDS} ---")
        logger.info(f"  Random: {len(train_idx_random)} train, {len(val_idx_random)} val")
        logger.info(f"  Hard: {len(train_idx_hard)} train, {len(val_idx_hard)} val")
        fold_t0 = time.time()

        fold_model = arch_cls().to(DEVICE)

        # Create datasets
        train_ds_random = ScreenDataset(train_idx_random, data_random)
        train_ds_hard = ScreenDataset(train_idx_hard, data_hard)

        # Validation: evaluate on v3-matched subset from hard dataset
        v3_val_idx_hard = val_idx_hard[val_idx_hard < n_v3_hard]
        if len(v3_val_idx_hard) > 0:
            eval_ds = ScreenDataset(v3_val_idx_hard, data_hard)
        else:
            eval_ds = ScreenDataset(val_idx_hard, data_hard)
        eval_loader = DataLoader(eval_ds, batch_size=BATCH_SIZE, shuffle=False,
                                 collate_fn=standard_collate, num_workers=0)

        # Full val loader for overall AUROC (use hard dataset val set)
        full_val_ds = ScreenDataset(val_idx_hard, data_hard)
        full_val_loader = DataLoader(full_val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                     collate_fn=standard_collate, num_workers=0)

        # Stage 1: Joint training (10 epochs with curriculum)
        total_epochs = STAGE1_EPOCHS + STAGE2_EPOCHS  # 30 total
        logger.info(f"  Curriculum: epochs 1-5 random, 6-10 mix, 11-30 hard-only")

        optimizer = torch.optim.Adam([
            {"params": fold_model.get_shared_params(), "lr": STAGE1_LR},
            {"params": fold_model.get_adapter_params(), "lr": STAGE1_LR},
        ], weight_decay=STAGE1_WEIGHT_DECAY)

        global_epoch = 0

        # Phase A: epochs 1-5, train on random
        for epoch in range(5):
            global_epoch += 1
            loader = DataLoader(train_ds_random, batch_size=BATCH_SIZE, shuffle=True,
                                collate_fn=standard_collate, num_workers=0)
            avg_loss, avg_binary = train_one_epoch(fold_model, loader, optimizer, DEVICE)
            if epoch == 0 or epoch == 4:
                val_auroc, _ = evaluate(fold_model, full_val_loader, DEVICE)
                logger.info(
                    f"    Epoch {global_epoch:2d} [random] | loss={avg_loss:.4f} "
                    f"| val_auroc={val_auroc:.4f}"
                )

        # Phase B: epochs 6-10, train on 50/50 mix
        for epoch in range(5):
            global_epoch += 1
            # Create mixed dataset by sampling equal amounts
            n_per_source = min(len(train_ds_random), len(train_ds_hard))
            random_indices = np.random.RandomState(SEED + global_epoch).choice(
                len(train_ds_random), size=n_per_source, replace=False
            )
            hard_indices = np.random.RandomState(SEED + global_epoch + 100).choice(
                len(train_ds_hard), size=n_per_source, replace=False
            )
            mixed_ds = MixedDataset(train_ds_random, train_ds_hard, random_indices, hard_indices)
            loader = DataLoader(mixed_ds, batch_size=BATCH_SIZE, shuffle=True,
                                collate_fn=standard_collate, num_workers=0)
            avg_loss, avg_binary = train_one_epoch(fold_model, loader, optimizer, DEVICE)
            if epoch == 0 or epoch == 4:
                val_auroc, _ = evaluate(fold_model, full_val_loader, DEVICE)
                logger.info(
                    f"    Epoch {global_epoch:2d} [mixed] | loss={avg_loss:.4f} "
                    f"| val_auroc={val_auroc:.4f}"
                )

        # Stage 2: Adapter-only, epochs 11-30, hard negatives only
        logger.info(f"  Stage 2: Adapter-only on hard negatives (20 epochs)")
        for p in fold_model.get_shared_params():
            p.requires_grad = False

        optimizer2 = torch.optim.Adam(
            fold_model.get_adapter_params(),
            lr=STAGE2_LR, weight_decay=STAGE2_WEIGHT_DECAY,
        )

        for epoch in range(STAGE2_EPOCHS):
            global_epoch += 1
            loader = DataLoader(train_ds_hard, batch_size=BATCH_SIZE, shuffle=True,
                                collate_fn=standard_collate, num_workers=0)
            avg_loss, avg_binary = train_one_epoch(fold_model, loader, optimizer2, DEVICE)
            if (epoch + 1) % 5 == 0 or epoch == STAGE2_EPOCHS - 1:
                val_auroc, val_per_enz = evaluate(fold_model, full_val_loader, DEVICE)
                enz_str = " ".join(f"{e}={v:.3f}" for e, v in val_per_enz.items() if not np.isnan(v))
                logger.info(
                    f"    Epoch {global_epoch:2d} [hard] | loss={avg_loss:.4f} "
                    f"| val_auroc={val_auroc:.4f} | {enz_str}"
                )

        # Final evaluation
        final_auroc, _ = evaluate(fold_model, full_val_loader, DEVICE)
        _, final_per_enz = evaluate(fold_model, eval_loader, DEVICE)

        fold_time = time.time() - fold_t0
        logger.info(
            f"  Fold {fold + 1} FINAL: overall_auroc={final_auroc:.4f} | time={fold_time:.0f}s"
        )
        for enz, auroc in final_per_enz.items():
            logger.info(f"    {enz}: {auroc:.4f}")

        fold_results.append({
            "fold": fold + 1,
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


class MixedDataset(Dataset):
    """Dataset that interleaves samples from two source datasets."""

    def __init__(self, ds_a, ds_b, indices_a, indices_b):
        self.ds_a = ds_a
        self.ds_b = ds_b
        self.indices_a = indices_a
        self.indices_b = indices_b
        self.n_a = len(indices_a)
        self.n_b = len(indices_b)

    def __len__(self):
        return self.n_a + self.n_b

    def __getitem__(self, idx):
        if idx < self.n_a:
            return self.ds_a[self.indices_a[idx]]
        else:
            return self.ds_b[self.indices_b[idx - self.n_a]]


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
# Main
# ---------------------------------------------------------------------------


def load_t1_results() -> Dict:
    """Load T1 (baseline v3) results from Phase A architecture screen."""
    results_path = OUTPUT_DIR / "architecture_screen_results.json"
    if not results_path.exists():
        logger.warning(f"Phase A results not found at {results_path}")
        return {}

    with open(results_path) as f:
        phase_a = json.load(f)

    t1_results = {}
    for r in phase_a:
        arch = r["architecture"]
        if r.get("overall_auroc_mean") is not None:
            t1_results[arch] = {
                "overall_auroc_mean": r["overall_auroc_mean"],
                "overall_auroc_std": r["overall_auroc_std"],
                "per_enzyme_auroc_mean": r.get("per_enzyme_auroc_mean", {}),
                "per_enzyme_auroc_std": r.get("per_enzyme_auroc_std", {}),
                "total_time_s": r.get("total_time_s", 0),
                "fold_results": r.get("fold_results", []),
            }
    return t1_results


def main():
    logger.info("=" * 70)
    logger.info("TRAINING METHODS SCREEN (Phase B)")
    logger.info(f"Device: {DEVICE} | Folds: {N_FOLDS} | Seed: {SEED}")
    logger.info(f"Architectures: A1, A2, A5, A6, A8")
    logger.info(f"Training methods: T1 (baseline), T2 (random 1:10), T3 (hard 1:5), T4 (large 1:50), T5 (curriculum)")
    logger.info("=" * 70)

    # Load T1 results from Phase A
    t1_results = load_t1_results()
    logger.info(f"\nT1 (Phase A) results loaded for: {list(t1_results.keys())}")

    # Load v3 base data (shared across all methods)
    v3_data = load_v3_data()

    # Load v4 variants
    logger.info("\n" + "=" * 70)
    logger.info("Loading v4 dataset variants")
    logger.info("=" * 70)

    data_random = load_v4_variant("random", v3_data)
    data_hard = load_v4_variant("hard", v3_data)
    data_large = load_v4_variant("large", v3_data)

    # Architecture list
    arch_list = [
        ("A1_Phase3Baseline", Phase3Baseline),
        ("A2_NoDelta", NoDelta),
        ("A5_GatedFusion", GatedFusion),
        ("A6_Conv2DBP", Conv2DBP),
        ("A8_HierarchicalAttention", HierarchicalAttention),
    ]

    training_methods = ["T1_baseline", "T2_v4random", "T3_v4hard", "T4_v4large", "T5_curriculum"]

    # Results matrix: {arch_name: {method: result_dict}}
    all_results = {}

    # Load existing incremental results if available
    results_path = OUTPUT_DIR / "training_screen_results.json"
    if results_path.exists():
        try:
            with open(results_path) as f:
                all_results = json.load(f)
            logger.info(f"\nLoaded {len(all_results)} existing results from {results_path}")
        except Exception:
            all_results = {}

    # Populate T1 results
    for arch_name, _ in arch_list:
        if arch_name not in all_results:
            all_results[arch_name] = {}
        if arch_name in t1_results:
            all_results[arch_name]["T1_baseline"] = t1_results[arch_name]

    # Save after loading T1
    _save_results(all_results, results_path)

    # Run T2-T5 for each architecture
    for arch_name, arch_cls in arch_list:
        if arch_name not in all_results:
            all_results[arch_name] = {}

        # T2: v4-random (1:10)
        method = "T2_v4random"
        if method not in all_results[arch_name]:
            logger.info(f"\n{'=' * 70}")
            logger.info(f"Running {arch_name} + {method}")
            logger.info(f"{'=' * 70}")
            try:
                result = run_standard_training(
                    arch_cls, arch_name, data_random,
                    stage1_epochs=STAGE1_EPOCHS,
                    stage2_epochs=STAGE2_EPOCHS,
                    batch_size=BATCH_SIZE,
                    eval_data=v3_data,
                )
                result["architecture"] = arch_name
                result["training_method"] = method
                all_results[arch_name][method] = result
                _log_result(arch_name, method, result)
            except Exception as e:
                logger.error(f"FAILED: {arch_name} + {method}: {e}")
                logger.error(traceback.format_exc())
                all_results[arch_name][method] = {"error": str(e)}
            _save_results(all_results, results_path)
            gc.collect()
            if DEVICE.type == "mps":
                torch.mps.empty_cache()
        else:
            logger.info(f"Skipping {arch_name} + {method} (already done)")

        # T3: v4-hard (1:5)
        method = "T3_v4hard"
        if method not in all_results[arch_name]:
            logger.info(f"\n{'=' * 70}")
            logger.info(f"Running {arch_name} + {method}")
            logger.info(f"{'=' * 70}")
            try:
                result = run_standard_training(
                    arch_cls, arch_name, data_hard,
                    stage1_epochs=STAGE1_EPOCHS,
                    stage2_epochs=STAGE2_EPOCHS,
                    batch_size=BATCH_SIZE,
                    eval_data=v3_data,
                )
                result["architecture"] = arch_name
                result["training_method"] = method
                all_results[arch_name][method] = result
                _log_result(arch_name, method, result)
            except Exception as e:
                logger.error(f"FAILED: {arch_name} + {method}: {e}")
                logger.error(traceback.format_exc())
                all_results[arch_name][method] = {"error": str(e)}
            _save_results(all_results, results_path)
            gc.collect()
            if DEVICE.type == "mps":
                torch.mps.empty_cache()
        else:
            logger.info(f"Skipping {arch_name} + {method} (already done)")

        # T4: v4-large (1:50)
        method = "T4_v4large"
        if method not in all_results[arch_name]:
            logger.info(f"\n{'=' * 70}")
            logger.info(f"Running {arch_name} + {method}")
            logger.info(f"{'=' * 70}")
            try:
                result = run_standard_training(
                    arch_cls, arch_name, data_large,
                    stage1_epochs=STAGE1_EPOCHS_LARGE,
                    stage2_epochs=STAGE2_EPOCHS_LARGE,
                    batch_size=BATCH_SIZE_LARGE,
                    eval_data=v3_data,
                )
                result["architecture"] = arch_name
                result["training_method"] = method
                all_results[arch_name][method] = result
                _log_result(arch_name, method, result)
            except Exception as e:
                logger.error(f"FAILED: {arch_name} + {method}: {e}")
                logger.error(traceback.format_exc())
                all_results[arch_name][method] = {"error": str(e)}
            _save_results(all_results, results_path)
            gc.collect()
            if DEVICE.type == "mps":
                torch.mps.empty_cache()
        else:
            logger.info(f"Skipping {arch_name} + {method} (already done)")

        # T5: Curriculum (random -> mix -> hard)
        method = "T5_curriculum"
        if method not in all_results[arch_name]:
            logger.info(f"\n{'=' * 70}")
            logger.info(f"Running {arch_name} + {method}")
            logger.info(f"{'=' * 70}")
            try:
                result = run_curriculum_training(
                    arch_cls, arch_name,
                    data_random, data_hard, v3_data,
                )
                result["architecture"] = arch_name
                result["training_method"] = method
                all_results[arch_name][method] = result
                _log_result(arch_name, method, result)
            except Exception as e:
                logger.error(f"FAILED: {arch_name} + {method}: {e}")
                logger.error(traceback.format_exc())
                all_results[arch_name][method] = {"error": str(e)}
            _save_results(all_results, results_path)
            gc.collect()
            if DEVICE.type == "mps":
                torch.mps.empty_cache()
        else:
            logger.info(f"Skipping {arch_name} + {method} (already done)")

    # ---------------------------------------------------------------------------
    # Summary comparison table
    # ---------------------------------------------------------------------------
    _print_summary_table(all_results, arch_list, training_methods)

    logger.info(f"\nResults saved to {results_path}")
    logger.info("\nTraining methods screen complete.")


def _save_results(all_results: Dict, path: Path):
    """Save results incrementally."""
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)


def _log_result(arch_name: str, method: str, result: Dict):
    """Log a single result."""
    if "error" in result:
        logger.info(f"  {arch_name} + {method}: FAILED — {result['error']}")
        return
    overall = result.get("overall_auroc_mean", "N/A")
    pe = result.get("per_enzyme_auroc_mean", {})
    pe_str = " ".join(f"{e}={v:.3f}" for e, v in pe.items())
    time_s = result.get("total_time_s", 0)
    logger.info(f"  {arch_name} + {method}: overall={overall:.4f} | {pe_str} | {time_s:.0f}s")


def _print_summary_table(all_results: Dict, arch_list: List, training_methods: List):
    """Print comparison matrices: overall AUROC and per-enzyme AUROCs."""
    logger.info("\n" + "=" * 100)
    logger.info("TRAINING METHODS SCREEN — OVERALL AUROC")
    logger.info("=" * 100)

    # Header
    header = f"{'Architecture':<28}"
    for method in training_methods:
        header += f" {method:>14}"
    logger.info(header)
    logger.info("-" * 100)

    for arch_name, _ in arch_list:
        row = f"{arch_name:<28}"
        for method in training_methods:
            r = all_results.get(arch_name, {}).get(method, {})
            if "error" in r:
                row += f" {'FAIL':>14}"
            elif "overall_auroc_mean" in r and r["overall_auroc_mean"] is not None:
                val = r["overall_auroc_mean"]
                std = r.get("overall_auroc_std", 0)
                row += f" {val:.4f}+{std:.4f}"
            else:
                row += f" {'N/A':>14}"
        logger.info(row)

    logger.info("=" * 100)

    # Per-enzyme tables
    for enz in PER_ENZYME_HEADS:
        logger.info(f"\n{'=' * 100}")
        logger.info(f"PER-ENZYME AUROC: {enz}")
        logger.info(f"{'=' * 100}")

        header = f"{'Architecture':<28}"
        for method in training_methods:
            header += f" {method:>14}"
        logger.info(header)
        logger.info("-" * 100)

        for arch_name, _ in arch_list:
            row = f"{arch_name:<28}"
            for method in training_methods:
                r = all_results.get(arch_name, {}).get(method, {})
                pe = r.get("per_enzyme_auroc_mean", {})
                if "error" in r:
                    row += f" {'FAIL':>14}"
                elif enz in pe:
                    val = pe[enz]
                    std = r.get("per_enzyme_auroc_std", {}).get(enz, 0)
                    row += f" {val:.4f}+{std:.4f}"
                else:
                    row += f" {'N/A':>14}"
            logger.info(row)

        logger.info("=" * 100)

    # Best method per architecture
    logger.info(f"\n{'=' * 100}")
    logger.info("BEST TRAINING METHOD PER ARCHITECTURE (by overall AUROC)")
    logger.info("=" * 100)
    for arch_name, _ in arch_list:
        best_method = None
        best_auroc = -1
        for method in training_methods:
            r = all_results.get(arch_name, {}).get(method, {})
            auroc = r.get("overall_auroc_mean", -1)
            if auroc is not None and auroc > best_auroc:
                best_auroc = auroc
                best_method = method
        if best_method:
            logger.info(f"  {arch_name:<28} -> {best_method:<16} (AUROC={best_auroc:.4f})")
        else:
            logger.info(f"  {arch_name:<28} -> no valid results")
    logger.info("=" * 100)


if __name__ == "__main__":
    main()
