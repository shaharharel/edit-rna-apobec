#!/usr/bin/env python
"""V4Deep replication: train 3 A8 neural models, score TCGA/ClinVar/cross-species/gnomAD.

Models:
  1. A8+T1+H4 — HierarchicalAttention, T1 baseline, H4 shared+private heads
  2. A8+T6+H4 — same + 5-epoch structure pretext pretraining
  3. A8+T4+H1 — HierarchicalAttention, T4 v4-large training, H1 standard heads

Steps:
  1. Train each model on FULL training data (no CV — all data)
  2. Run 5-fold CV for classification metrics
  3. Score TCGA (10 cancers + COADREAD) with all 3 models
  4. Score ClinVar (1.69M) with all 3 models
  5. Score cross-species (557 orthologs) with all 3 models
  6. Compute gnomAD gene-level correlations

Output: experiments/multi_enzyme/outputs/v4deep/v4deep_results.json
Run:
  nohup /opt/miniconda3/envs/quris/bin/python -u experiments/multi_enzyme/exp_v4deep_replication.py \
    > experiments/multi_enzyme/outputs/v4deep/replication.log 2>&1 &
"""

import gc
import json
import logging
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
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

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "multi_enzyme" / "outputs" / "v4deep"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_PATH = OUTPUT_DIR / "v4deep_results.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(OUTPUT_DIR / "replication.log", mode="w"),
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
BATCH_SIZE_SCORE = 2048  # For inference

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

# TCGA cancers
TCGA_CANCERS = ["blca", "brca", "cesc", "lusc", "hnsc", "esca", "stad", "lihc", "skcm", "coadread"]


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
    """H4: Shared encoder -> 128dim. Private encoder per enzyme -> concat -> adapter."""

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
# Pretext Head (T6)
# ---------------------------------------------------------------------------


class PretextHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(D_SHARED, 1)

    def forward(self, shared):
        return self.head(shared).squeeze(-1)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


class ScreenDataset(Dataset):
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


class ScoringDataset(Dataset):
    """Lightweight dataset for inference — no labels needed."""
    def __init__(self, rnafm, hand_feat, bp_sub=None):
        self.rnafm = rnafm  # [N, 640]
        self.hand_feat = hand_feat  # [N, 40]
        self.bp_sub = bp_sub  # [N, 41, 41] or None (use zeros)
        self.n = rnafm.shape[0]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        bp = self.bp_sub[idx] if self.bp_sub is not None else np.zeros((41, 41), dtype=np.float32)
        return {
            "rnafm": torch.from_numpy(self.rnafm[idx]),
            "bp_submatrix": torch.from_numpy(bp).unsqueeze(0),
            "hand_feat": torch.from_numpy(self.hand_feat[idx]),
        }


def standard_collate(batch_list):
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

ARCH_SCREEN_DIR = PROJECT_ROOT / "experiments" / "multi_enzyme" / "outputs" / "architecture_screen"


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

    bp_cache_path = ARCH_SCREEN_DIR / "bp_submatrices_v3.npz"
    if not bp_cache_path.exists():
        raise FileNotFoundError(f"BP cache not found at {bp_cache_path}")
    bp_data = np.load(bp_cache_path, allow_pickle=True)
    bp_submatrices_v3 = bp_data["bp_submatrices"]
    bp_site_ids_v3 = list(bp_data["site_ids"])
    logger.info(f"  BP cache: {bp_submatrices_v3.shape}")

    db_cache_path = ARCH_SCREEN_DIR / "dot_brackets_v3.json"
    dot_brackets = {}
    if db_cache_path.exists():
        with open(db_cache_path) as f:
            dot_brackets = json.load(f)
        logger.info(f"  Dot-bracket cache: {len(dot_brackets)} structures")

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
    logger.info(f"  RNA-FM coverage: {n_rnafm_found}/{n}")

    del rnafm_emb, rnafm_edited_emb
    gc.collect()

    if bp_site_ids_v3 == site_ids:
        bp_sub = bp_submatrices_v3
    else:
        bp_sid_to_idx = {sid: i for i, sid in enumerate(bp_site_ids_v3)}
        bp_sub = np.zeros((n, 41, 41), dtype=np.float32)
        for i, sid in enumerate(site_ids):
            if sid in bp_sid_to_idx:
                bp_sub[i] = bp_submatrices_v3[bp_sid_to_idx[sid]]
    logger.info(f"  BP submatrices: {bp_sub.shape}")

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
    """Load v4-large dataset (385K sites)."""
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

    neg_bp = np.zeros((n_neg, 41, 41), dtype=np.float32)
    neg_labels_binary = np.zeros(n_neg, dtype=np.float32)
    neg_labels_enzyme = np.full(n_neg, 5, dtype=np.int64)  # Unknown
    neg_per_enzyme_labels = np.full((n_neg, N_PER_ENZYME), -1, dtype=np.float32)

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
    logger.info(f"  Combined v4-large: {n_total} sites ({int(combined['labels_binary'].sum())} pos, "
                f"{int((combined['labels_binary'] == 0).sum())} neg)")

    del rnafm_v4, rnafm_edited_v4
    gc.collect()

    return combined


# ---------------------------------------------------------------------------
# Loss / Training / Evaluation
# ---------------------------------------------------------------------------


def compute_loss(binary_logit, per_enzyme_logits, enzyme_cls_logits, batch):
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
# Inference: score external data with a trained model
# ---------------------------------------------------------------------------


@torch.no_grad()
def score_dataset(model, dataset, device, batch_size=BATCH_SIZE_SCORE):
    """Score a ScoringDataset. Returns dict: binary_probs, per_enzyme_probs."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=standard_collate, num_workers=0)

    all_binary = []
    all_per_enzyme = {enz: [] for enz in PER_ENZYME_HEADS}

    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        binary_logit, per_enzyme_logits, _ = model(batch)
        all_binary.append(torch.sigmoid(binary_logit).cpu().numpy())
        for h_idx, enz in enumerate(PER_ENZYME_HEADS):
            all_per_enzyme[enz].append(torch.sigmoid(per_enzyme_logits[h_idx]).cpu().numpy())

    return {
        "binary": np.concatenate(all_binary),
        **{enz: np.concatenate(all_per_enzyme[enz]) for enz in PER_ENZYME_HEADS},
    }


# ---------------------------------------------------------------------------
# Enrichment helpers
# ---------------------------------------------------------------------------


def compute_or_fisher(mut_scores, ctrl_scores, percentile, pooled_scores=None):
    """Compute OR using fisher_exact at a given percentile threshold."""
    if pooled_scores is None:
        pooled_scores = np.concatenate([mut_scores, ctrl_scores])
    thresh = np.percentile(pooled_scores, percentile)
    ma = int((mut_scores >= thresh).sum())
    mb = int((mut_scores < thresh).sum())
    ca = int((ctrl_scores >= thresh).sum())
    cb = int((ctrl_scores < thresh).sum())
    if all(x > 0 for x in [ma, mb, ca, cb]):
        OR, pv = stats.fisher_exact([[ma, mb], [ca, cb]])
    else:
        OR, pv = float("nan"), 1.0
    return {
        "OR": float(OR), "p": float(pv), "threshold": float(thresh),
        "mut_above": ma, "mut_below": mb, "ctrl_above": ca, "ctrl_below": cb,
    }


def compute_enrichment_full(mut_scores, ctrl_scores):
    """Compute enrichment at p50, p75, p90, p95."""
    pooled = np.concatenate([mut_scores, ctrl_scores])
    result = {}
    for pct in [50, 75, 90, 95]:
        result[f"p{pct}"] = compute_or_fisher(mut_scores, ctrl_scores, pct, pooled_scores=pooled)
    result["n_mut"] = int(len(mut_scores))
    result["n_ctrl"] = int(len(ctrl_scores))
    result["mean_mut"] = float(np.mean(mut_scores))
    result["mean_ctrl"] = float(np.mean(ctrl_scores))
    return result


# ---------------------------------------------------------------------------
# Training: full data (no CV)
# ---------------------------------------------------------------------------


def train_full_t1(model_cls, data, name="T1"):
    """Train T1 on ALL data. Returns trained model."""
    logger.info(f"\n{'='*70}")
    logger.info(f"Training {name} on FULL data ({len(data['labels_binary'])} sites)")
    logger.info(f"{'='*70}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = model_cls().to(DEVICE)

    all_idx = np.arange(len(data["labels_binary"]))
    ds = ScreenDataset(all_idx, data)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                        collate_fn=standard_collate, num_workers=0)

    # Stage 1: Joint
    optimizer = torch.optim.Adam([
        {"params": model.get_shared_params(), "lr": STAGE1_LR},
        {"params": model.get_adapter_params(), "lr": STAGE1_LR},
    ], weight_decay=STAGE1_WEIGHT_DECAY)

    t0 = time.time()
    for epoch in range(STAGE1_EPOCHS):
        avg_loss, avg_binary = train_one_epoch(model, loader, optimizer, DEVICE)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(f"  Stage1 Epoch {epoch+1:2d} | loss={avg_loss:.4f} bce={avg_binary:.4f}")

    # Stage 2: Adapter-only
    for p in model.get_shared_params():
        p.requires_grad = False
    optimizer2 = torch.optim.Adam(
        model.get_adapter_params(), lr=STAGE2_LR, weight_decay=STAGE2_WEIGHT_DECAY,
    )

    for epoch in range(STAGE2_EPOCHS):
        avg_loss, avg_binary = train_one_epoch(model, loader, optimizer2, DEVICE)
        if (epoch + 1) % 5 == 0 or epoch == STAGE2_EPOCHS - 1:
            logger.info(f"  Stage2 Epoch {STAGE1_EPOCHS+epoch+1:2d} | loss={avg_loss:.4f}")

    elapsed = time.time() - t0
    logger.info(f"  Training complete in {elapsed:.0f}s")

    # Save weights
    model_path = OUTPUT_DIR / f"model_{name}.pt"
    torch.save(model.state_dict(), model_path)
    logger.info(f"  Saved model to {model_path}")

    return model


def train_full_t6(model_cls, data, name="T6"):
    """Train T6: pretext + T1."""
    logger.info(f"\n{'='*70}")
    logger.info(f"Training {name} on FULL data (pretext + T1)")
    logger.info(f"{'='*70}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = model_cls().to(DEVICE)
    pretext_head = PretextHead().to(DEVICE)

    all_idx = np.arange(len(data["labels_binary"]))

    # Stage 0: Pretext
    logger.info(f"  Stage 0: Pretext ({PRETEXT_EPOCHS} epochs)")
    pretext_ds = PretextDataset(all_idx, data)
    pretext_loader = DataLoader(pretext_ds, batch_size=BATCH_SIZE, shuffle=True,
                                collate_fn=standard_collate, num_workers=0)

    pretext_opt = torch.optim.Adam(
        list(model.get_shared_params()) + list(pretext_head.parameters()),
        lr=1e-3, weight_decay=1e-4,
    )

    t0 = time.time()
    for epoch in range(PRETEXT_EPOCHS):
        model.train()
        pretext_head.train()
        total_loss, n_batches = 0.0, 0
        for batch in pretext_loader:
            batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            pretext_opt.zero_grad()
            shared = model.encode(batch)
            logits = pretext_head(shared)
            labels = batch["pretext_label"]
            mask = (labels != 0.5)
            if mask.sum() == 0:
                continue
            loss = F.binary_cross_entropy_with_logits(logits[mask], labels[mask])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(pretext_head.parameters()), 1.0)
            pretext_opt.step()
            total_loss += loss.item()
            n_batches += 1
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(f"    Pretext epoch {epoch+1} | loss={total_loss/max(n_batches,1):.4f}")

    del pretext_head
    gc.collect()

    # Re-init heads
    model._init_heads()
    model = model.to(DEVICE)

    # Stage 1: Joint
    ds = ScreenDataset(all_idx, data)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                        collate_fn=standard_collate, num_workers=0)

    optimizer = torch.optim.Adam([
        {"params": model.get_shared_params(), "lr": STAGE1_LR},
        {"params": model.get_adapter_params(), "lr": STAGE1_LR},
    ], weight_decay=STAGE1_WEIGHT_DECAY)

    for epoch in range(STAGE1_EPOCHS):
        avg_loss, avg_binary = train_one_epoch(model, loader, optimizer, DEVICE)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(f"  Stage1 Epoch {epoch+1:2d} | loss={avg_loss:.4f}")

    # Stage 2: Adapter-only
    for p in model.get_shared_params():
        p.requires_grad = False
    optimizer2 = torch.optim.Adam(
        model.get_adapter_params(), lr=STAGE2_LR, weight_decay=STAGE2_WEIGHT_DECAY,
    )
    for epoch in range(STAGE2_EPOCHS):
        avg_loss, _ = train_one_epoch(model, loader, optimizer2, DEVICE)
        if (epoch + 1) % 5 == 0 or epoch == STAGE2_EPOCHS - 1:
            logger.info(f"  Stage2 Epoch {STAGE1_EPOCHS+epoch+1:2d} | loss={avg_loss:.4f}")

    elapsed = time.time() - t0
    logger.info(f"  Training complete in {elapsed:.0f}s")

    model_path = OUTPUT_DIR / f"model_{name}.pt"
    torch.save(model.state_dict(), model_path)
    logger.info(f"  Saved model to {model_path}")

    return model


def train_full_t4(model_cls, v3_data, v4_data, name="T4"):
    """Train T4 on v4-large, all data."""
    logger.info(f"\n{'='*70}")
    logger.info(f"Training {name} on v4-large FULL data ({len(v4_data['labels_binary'])} sites)")
    logger.info(f"{'='*70}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = model_cls().to(DEVICE)

    all_idx = np.arange(len(v4_data["labels_binary"]))
    ds = ScreenDataset(all_idx, v4_data)
    loader = DataLoader(ds, batch_size=BATCH_SIZE_LARGE, shuffle=True,
                        collate_fn=standard_collate, num_workers=0)

    # Stage 1: Joint (5 epochs)
    optimizer = torch.optim.Adam([
        {"params": model.get_shared_params(), "lr": STAGE1_LR},
        {"params": model.get_adapter_params(), "lr": STAGE1_LR},
    ], weight_decay=STAGE1_WEIGHT_DECAY)

    t0 = time.time()
    for epoch in range(STAGE1_EPOCHS_LARGE):
        avg_loss, avg_binary = train_one_epoch(model, loader, optimizer, DEVICE)
        logger.info(f"  Stage1 Epoch {epoch+1:2d} | loss={avg_loss:.4f} bce={avg_binary:.4f}")

    # Stage 2: Adapter-only (10 epochs)
    for p in model.get_shared_params():
        p.requires_grad = False
    optimizer2 = torch.optim.Adam(
        model.get_adapter_params(), lr=STAGE2_LR, weight_decay=STAGE2_WEIGHT_DECAY,
    )
    for epoch in range(STAGE2_EPOCHS_LARGE):
        avg_loss, _ = train_one_epoch(model, loader, optimizer2, DEVICE)
        if (epoch + 1) % 5 == 0 or epoch == STAGE2_EPOCHS_LARGE - 1:
            logger.info(f"  Stage2 Epoch {STAGE1_EPOCHS_LARGE+epoch+1:2d} | loss={avg_loss:.4f}")

    elapsed = time.time() - t0
    logger.info(f"  Training complete in {elapsed:.0f}s")

    model_path = OUTPUT_DIR / f"model_{name}.pt"
    torch.save(model.state_dict(), model_path)
    logger.info(f"  Saved model to {model_path}")

    return model


# ---------------------------------------------------------------------------
# Step 2: 5-fold CV
# ---------------------------------------------------------------------------


def run_5fold_cv(model_cls, data, training_fn, training_kwargs, combo_name):
    """5-fold StratifiedKFold CV. Returns per-fold AUROCs."""
    logger.info(f"\n{'='*70}")
    logger.info(f"5-Fold CV for {combo_name}")
    logger.info(f"{'='*70}")

    n = len(data["labels_binary"])
    strat_key = data["labels_enzyme"] * 2 + data["labels_binary"].astype(np.int64)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    fold_results = []
    t0 = time.time()

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(n), strat_key)):
        logger.info(f"  Fold {fold+1}/5 ({len(train_idx)} train, {len(val_idx)} val)")
        fold_t0 = time.time()

        torch.manual_seed(SEED + fold)
        fold_model = model_cls().to(DEVICE)

        train_ds = ScreenDataset(train_idx, data)
        val_ds = ScreenDataset(val_idx, data)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  collate_fn=standard_collate, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                collate_fn=standard_collate, num_workers=0)

        # For T6: pretext first
        if "pretext" in combo_name.lower() or "t6" in combo_name.lower():
            pretext_head = PretextHead().to(DEVICE)
            pretext_ds = PretextDataset(train_idx, data)
            pretext_loader = DataLoader(pretext_ds, batch_size=BATCH_SIZE, shuffle=True,
                                        collate_fn=standard_collate, num_workers=0)
            pretext_opt = torch.optim.Adam(
                list(fold_model.get_shared_params()) + list(pretext_head.parameters()),
                lr=1e-3, weight_decay=1e-4,
            )
            for epoch in range(PRETEXT_EPOCHS):
                fold_model.train()
                pretext_head.train()
                for batch in pretext_loader:
                    batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    pretext_opt.zero_grad()
                    shared = fold_model.encode(batch)
                    logits = pretext_head(shared)
                    labels = batch["pretext_label"]
                    mask = (labels != 0.5)
                    if mask.sum() == 0:
                        continue
                    loss = F.binary_cross_entropy_with_logits(logits[mask], labels[mask])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(list(fold_model.parameters()) + list(pretext_head.parameters()), 1.0)
                    pretext_opt.step()
            del pretext_head
            gc.collect()
            fold_model._init_heads()
            fold_model = fold_model.to(DEVICE)

        # Stage 1: Joint
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
            fold_model.get_adapter_params(), lr=STAGE2_LR, weight_decay=STAGE2_WEIGHT_DECAY,
        )
        for epoch in range(STAGE2_EPOCHS):
            train_one_epoch(fold_model, train_loader, optimizer2, DEVICE)

        auroc, per_enz = evaluate(fold_model, val_loader, DEVICE)
        fold_time = time.time() - fold_t0
        enz_str = " ".join(f"{e}={v:.3f}" for e, v in per_enz.items() if not np.isnan(v))
        logger.info(f"    AUROC={auroc:.4f} | {enz_str} | {fold_time:.0f}s")

        fold_results.append({
            "fold": fold + 1,
            "overall_auroc": round(float(auroc), 4),
            "per_enzyme_aurocs": {k: round(float(v), 4) for k, v in per_enz.items()},
        })

        del fold_model, optimizer, optimizer2
        gc.collect()
        if DEVICE.type == "mps":
            torch.mps.empty_cache()

    total_time = time.time() - t0

    # Aggregate
    overall_aurocs = [r["overall_auroc"] for r in fold_results]
    per_enz_means = {}
    for enz in PER_ENZYME_HEADS:
        vals = [r["per_enzyme_aurocs"].get(enz, float("nan")) for r in fold_results]
        vals = [v for v in vals if not np.isnan(v)]
        if vals:
            per_enz_means[enz] = round(float(np.mean(vals)), 4)

    summary = {
        "overall_auroc_mean": round(float(np.mean(overall_aurocs)), 4),
        "overall_auroc_std": round(float(np.std(overall_aurocs)), 4),
        "per_enzyme_auroc_mean": per_enz_means,
        "total_time_s": round(total_time, 1),
        "fold_results": fold_results,
    }
    logger.info(f"  5-Fold mean AUROC: {summary['overall_auroc_mean']:.4f} +/- {summary['overall_auroc_std']:.4f}")
    return summary


def run_5fold_cv_t4(model_cls, v4_data, combo_name):
    """5-fold CV for T4 (v4-large training)."""
    logger.info(f"\n{'='*70}")
    logger.info(f"5-Fold CV for {combo_name} (v4-large)")
    logger.info(f"{'='*70}")

    n = len(v4_data["labels_binary"])
    n_v3 = v4_data.get("n_v3", n)
    strat_key = v4_data["labels_binary"].astype(np.int64)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    fold_results = []
    t0 = time.time()

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(n), strat_key)):
        logger.info(f"  Fold {fold+1}/5 ({len(train_idx)} train, {len(val_idx)} val)")
        fold_t0 = time.time()

        torch.manual_seed(SEED + fold)
        fold_model = model_cls().to(DEVICE)

        train_ds = ScreenDataset(train_idx, v4_data)
        val_ds = ScreenDataset(val_idx, v4_data)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE_LARGE, shuffle=True,
                                  collate_fn=standard_collate, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE_LARGE, shuffle=False,
                                collate_fn=standard_collate, num_workers=0)

        # Eval on v3-matched subset
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
            train_one_epoch(fold_model, train_loader, optimizer, DEVICE)

        # Stage 2: Adapter-only (10 epochs)
        for p in fold_model.get_shared_params():
            p.requires_grad = False
        optimizer2 = torch.optim.Adam(
            fold_model.get_adapter_params(), lr=STAGE2_LR, weight_decay=STAGE2_WEIGHT_DECAY,
        )
        for epoch in range(STAGE2_EPOCHS_LARGE):
            train_one_epoch(fold_model, train_loader, optimizer2, DEVICE)

        _, per_enz = evaluate(fold_model, eval_loader, DEVICE)
        auroc, _ = evaluate(fold_model, val_loader, DEVICE)
        fold_time = time.time() - fold_t0
        enz_str = " ".join(f"{e}={v:.3f}" for e, v in per_enz.items() if not np.isnan(v))
        logger.info(f"    AUROC={auroc:.4f} | {enz_str} | {fold_time:.0f}s")

        fold_results.append({
            "fold": fold + 1,
            "overall_auroc": round(float(auroc), 4),
            "per_enzyme_aurocs": {k: round(float(v), 4) for k, v in per_enz.items()},
        })

        del fold_model, optimizer, optimizer2
        gc.collect()
        if DEVICE.type == "mps":
            torch.mps.empty_cache()

    total_time = time.time() - t0
    overall_aurocs = [r["overall_auroc"] for r in fold_results]
    per_enz_means = {}
    for enz in PER_ENZYME_HEADS:
        vals = [r["per_enzyme_aurocs"].get(enz, float("nan")) for r in fold_results]
        vals = [v for v in vals if not np.isnan(v)]
        if vals:
            per_enz_means[enz] = round(float(np.mean(vals)), 4)

    summary = {
        "overall_auroc_mean": round(float(np.mean(overall_aurocs)), 4),
        "overall_auroc_std": round(float(np.std(overall_aurocs)), 4),
        "per_enzyme_auroc_mean": per_enz_means,
        "total_time_s": round(total_time, 1),
        "fold_results": fold_results,
    }
    logger.info(f"  5-Fold mean AUROC: {summary['overall_auroc_mean']:.4f} +/- {summary['overall_auroc_std']:.4f}")
    return summary


# ---------------------------------------------------------------------------
# Step 3: TCGA Scoring
# ---------------------------------------------------------------------------


def score_tcga(models: Dict[str, nn.Module], results: Dict):
    """Score all TCGA cancers with all models."""
    logger.info(f"\n{'='*70}")
    logger.info("TCGA Scoring")
    logger.info(f"{'='*70}")

    tcga_results = {}
    emb_dir = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "embeddings"
    hand_dir = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "tcga_hand_features"

    for cancer in TCGA_CANCERS:
        logger.info(f"\n--- {cancer.upper()} ---")
        cancer_t0 = time.time()

        try:
            # Load data
            emb_path = emb_dir / f"rnafm_tcga_{cancer}.pt"
            if not emb_path.exists():
                logger.warning(f"  No embeddings for {cancer}, skipping")
                continue

            emb_data = torch.load(emb_path, map_location="cpu", weights_only=False)
            n_mut = emb_data["n_mut"]
            n_ctrl = emb_data["n_ctrl"]
            n_total = emb_data["n_total"]
            pooled_orig = emb_data["pooled_orig"].numpy()
            pooled_edited = emb_data["pooled_edited"].numpy()
            logger.info(f"  n_mut={n_mut}, n_ctrl={n_ctrl}, n_total={n_total}")

            # Hand features
            aligned_path = hand_dir / f"{cancer}_hand40_aligned.npy"
            unaligned_path = hand_dir / f"{cancer}_hand40.npy"
            if aligned_path.exists():
                hand_feat = np.load(str(aligned_path))
                logger.info(f"  Using aligned hand features: {hand_feat.shape}")
            elif unaligned_path.exists():
                hand_feat = np.load(str(unaligned_path))
                logger.info(f"  Using unaligned hand features: {hand_feat.shape}")
            else:
                logger.warning(f"  No hand features for {cancer}, using zeros")
                hand_feat = np.zeros((n_total, D_HAND), dtype=np.float32)

            if hand_feat.shape[0] != n_total:
                logger.warning(f"  Hand feature size mismatch: {hand_feat.shape[0]} vs {n_total}, truncating/padding")
                if hand_feat.shape[0] > n_total:
                    hand_feat = hand_feat[:n_total]
                else:
                    pad = np.zeros((n_total - hand_feat.shape[0], D_HAND), dtype=np.float32)
                    hand_feat = np.concatenate([hand_feat, pad])

            # RNA-FM as input (A8 uses rnafm directly, not delta)
            rnafm = pooled_orig.astype(np.float32)
            hand_feat = hand_feat.astype(np.float32)

            # Derive TC context from hand features (col 0 = UC/TC dinucleotide)
            tc_mask = hand_feat[:, 0] == 1.0  # UC = TC context

            mut_mask = np.zeros(n_total, dtype=bool)
            mut_mask[:n_mut] = True
            ctrl_mask = ~mut_mask

            # Create scoring dataset (no BP submatrices for TCGA)
            dataset = ScoringDataset(rnafm, hand_feat, bp_sub=None)

            cancer_result = {"n_mut": n_mut, "n_ctrl": n_ctrl}

            for model_name, model in models.items():
                logger.info(f"  Scoring with {model_name}...")
                scores = score_dataset(model, dataset, DEVICE)

                model_result = {}

                # Binary head enrichment
                binary_scores = scores["binary"]
                model_result["binary"] = compute_enrichment_full(
                    binary_scores[mut_mask], binary_scores[ctrl_mask]
                )

                # Per-enzyme head enrichment
                for enz in PER_ENZYME_HEADS:
                    enz_scores = scores[enz]
                    model_result[f"adapter_{enz}"] = compute_enrichment_full(
                        enz_scores[mut_mask], enz_scores[ctrl_mask]
                    )

                # TC-stratified
                tc_mut = mut_mask & tc_mask
                tc_ctrl = ctrl_mask & tc_mask
                nontc_mut = mut_mask & ~tc_mask
                nontc_ctrl = ctrl_mask & ~tc_mask

                if tc_mut.sum() > 100 and tc_ctrl.sum() > 100:
                    model_result["binary_tc_only"] = compute_enrichment_full(
                        binary_scores[tc_mut], binary_scores[tc_ctrl]
                    )
                if nontc_mut.sum() > 100 and nontc_ctrl.sum() > 100:
                    model_result["binary_nontc_only"] = compute_enrichment_full(
                        binary_scores[nontc_mut], binary_scores[nontc_ctrl]
                    )

                cancer_result[model_name] = model_result

            del emb_data, pooled_orig, pooled_edited, rnafm, hand_feat
            gc.collect()

            tcga_results[cancer] = cancer_result
            logger.info(f"  {cancer} done in {time.time()-cancer_t0:.0f}s")

        except Exception as e:
            logger.error(f"  ERROR on {cancer}: {e}")
            logger.error(traceback.format_exc())
            tcga_results[cancer] = {"error": str(e)}

    results["tcga"] = tcga_results
    save_results(results)
    return results


# ---------------------------------------------------------------------------
# Step 4: ClinVar Scoring
# ---------------------------------------------------------------------------


def score_clinvar(models: Dict[str, nn.Module], results: Dict):
    """Score ClinVar with all models."""
    logger.info(f"\n{'='*70}")
    logger.info("ClinVar Scoring (1.69M variants)")
    logger.info(f"{'='*70}")

    try:
        # Load ClinVar embeddings
        emb_path = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "embeddings" / "rnafm_clinvar.pt"
        emb_data = torch.load(emb_path, map_location="cpu", weights_only=False)
        clinvar_rnafm = emb_data["pooled_orig"].numpy().astype(np.float32)
        clinvar_site_ids = emb_data["site_ids"]
        n_clinvar = clinvar_rnafm.shape[0]
        logger.info(f"  ClinVar embeddings: {n_clinvar}")

        # Load hand features
        feat_path = PROJECT_ROOT / "data" / "processed" / "clinvar_features_cache.npz"
        feat_data = np.load(feat_path, allow_pickle=True)
        clinvar_hand = feat_data["hand_46"][:, :40].astype(np.float32)  # First 40 cols
        feat_site_ids = list(feat_data["site_ids"])
        logger.info(f"  ClinVar hand features: {clinvar_hand.shape}")

        # Align hand features to embeddings
        if len(feat_site_ids) == n_clinvar:
            # Check if same order
            if feat_site_ids[:5] == list(clinvar_site_ids[:5]):
                logger.info("  Hand features aligned")
            else:
                logger.info("  Reindexing hand features...")
                feat_idx = {sid: i for i, sid in enumerate(feat_site_ids)}
                aligned = np.zeros((n_clinvar, 40), dtype=np.float32)
                for i, sid in enumerate(clinvar_site_ids):
                    if sid in feat_idx:
                        aligned[i] = clinvar_hand[feat_idx[sid]]
                clinvar_hand = aligned
        else:
            logger.warning(f"  Size mismatch: emb={n_clinvar}, feat={len(feat_site_ids)}, reindexing...")
            feat_idx = {sid: i for i, sid in enumerate(feat_site_ids)}
            aligned = np.zeros((n_clinvar, 40), dtype=np.float32)
            for i, sid in enumerate(clinvar_site_ids):
                if sid in feat_idx:
                    aligned[i] = clinvar_hand[feat_idx[sid]]
            clinvar_hand = aligned

        # Load ClinVar labels
        labels_path = PROJECT_ROOT / "experiments" / "apobec3a" / "outputs" / "clinvar_prediction" / "clinvar_all_scores.csv"
        clinvar_df = pd.read_csv(labels_path)
        logger.info(f"  ClinVar labels: {len(clinvar_df)} rows")

        # Align labels to embeddings
        label_idx = {sid: i for i, sid in enumerate(clinvar_df["site_id"].tolist())}

        # Create scoring dataset
        dataset = ScoringDataset(clinvar_rnafm, clinvar_hand, bp_sub=None)

        clinvar_results = {"n_total": n_clinvar}

        for model_name, model in models.items():
            logger.info(f"  Scoring with {model_name}...")
            scores = score_dataset(model, dataset, DEVICE, batch_size=4096)
            binary_scores = scores["binary"]

            # Map scores to ClinVar labels
            sig_simple = []
            gene_list = []
            score_mapped = []
            for i, sid in enumerate(clinvar_site_ids):
                if sid in label_idx:
                    row_idx = label_idx[sid]
                    sig_simple.append(clinvar_df.iloc[row_idx]["significance_simple"])
                    gene_list.append(clinvar_df.iloc[row_idx]["gene"])
                    score_mapped.append(binary_scores[i])

            sig_simple = np.array(sig_simple)
            score_mapped = np.array(score_mapped)
            gene_list = np.array(gene_list)

            pathogenic = (sig_simple == "Pathogenic") | (sig_simple == "Likely_pathogenic")
            benign = (sig_simple == "Benign") | (sig_simple == "Likely_benign")
            logger.info(f"    {model_name}: {pathogenic.sum()} pathogenic, {benign.sum()} benign")

            model_result = {}
            if pathogenic.sum() > 100 and benign.sum() > 100:
                model_result["binary"] = compute_enrichment_full(
                    score_mapped[pathogenic], score_mapped[benign]
                )

            # Per-enzyme adapter scores
            for enz in PER_ENZYME_HEADS:
                enz_scores = scores[enz]
                enz_mapped = []
                for i, sid in enumerate(clinvar_site_ids):
                    if sid in label_idx:
                        enz_mapped.append(enz_scores[i])
                enz_mapped = np.array(enz_mapped)
                if pathogenic.sum() > 100 and benign.sum() > 100:
                    model_result[f"adapter_{enz}"] = compute_enrichment_full(
                        enz_mapped[pathogenic], enz_mapped[benign]
                    )

            # GI-gene stratified for Neither adapter
            gi_genes = set()
            try:
                # Common GI-tract genes
                gi_keywords = ["MUC", "CDH", "APC", "MLH", "MSH", "EPCAM", "SMAD",
                               "BRAF", "KRAS", "TP53", "PIK3CA", "PTEN", "CTNNB1"]
                for g in gene_list:
                    if isinstance(g, str):
                        for kw in gi_keywords:
                            if kw in g.upper():
                                gi_genes.add(g)
            except Exception:
                pass

            if gi_genes:
                gi_mask = np.array([g in gi_genes for g in gene_list])
                if gi_mask.sum() > 50:
                    neither_scores = scores["Neither"]
                    neither_mapped = []
                    for i, sid in enumerate(clinvar_site_ids):
                        if sid in label_idx:
                            neither_mapped.append(neither_scores[i])
                    neither_mapped = np.array(neither_mapped)

                    gi_path = pathogenic & gi_mask
                    gi_ben = benign & gi_mask
                    if gi_path.sum() > 10 and gi_ben.sum() > 10:
                        model_result["neither_gi_genes"] = compute_enrichment_full(
                            neither_mapped[gi_path], neither_mapped[gi_ben]
                        )

            clinvar_results[model_name] = model_result

        del clinvar_rnafm, clinvar_hand, emb_data
        gc.collect()

        results["clinvar"] = clinvar_results
        save_results(results)

    except Exception as e:
        logger.error(f"ClinVar ERROR: {e}")
        logger.error(traceback.format_exc())
        results["clinvar"] = {"error": str(e)}

    return results


# ---------------------------------------------------------------------------
# Step 5: Cross-species Scoring
# ---------------------------------------------------------------------------


def score_cross_species(models: Dict[str, nn.Module], v3_data: Dict, results: Dict):
    """Score cross-species orthologs."""
    logger.info(f"\n{'='*70}")
    logger.info("Cross-species Scoring (557 orthologs)")
    logger.info(f"{'='*70}")

    try:
        emb_path = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "embeddings" / "rnafm_cross_species.pt"
        emb_data = torch.load(emb_path, map_location="cpu", weights_only=False)

        site_ids = emb_data["site_ids"]
        human_pooled = emb_data["human_pooled"].numpy().astype(np.float32)
        chimp_pooled = emb_data["chimp_pooled"].numpy().astype(np.float32)
        n_sites = len(site_ids)
        logger.info(f"  {n_sites} ortholog pairs")

        # Get hand features from v3 data
        v3_sid_to_idx = {sid: i for i, sid in enumerate(v3_data["site_ids"])}
        human_hand = np.zeros((n_sites, D_HAND), dtype=np.float32)
        for i, sid in enumerate(site_ids):
            if sid in v3_sid_to_idx:
                human_hand[i] = v3_data["hand_features"][v3_sid_to_idx[sid]]

        # Use same hand features for chimp (structural features should be similar for orthologs)
        chimp_hand = human_hand.copy()

        # Get BP submatrices from v3 data for human
        human_bp = np.zeros((n_sites, 41, 41), dtype=np.float32)
        for i, sid in enumerate(site_ids):
            if sid in v3_sid_to_idx:
                human_bp[i] = v3_data["bp_submatrices"][v3_sid_to_idx[sid]]

        human_ds = ScoringDataset(human_pooled, human_hand, bp_sub=human_bp)
        chimp_ds = ScoringDataset(chimp_pooled, chimp_hand, bp_sub=None)  # No BP for chimp

        cross_results = {"n_sites": n_sites}

        for model_name, model in models.items():
            logger.info(f"  Scoring with {model_name}...")
            human_scores = score_dataset(model, human_ds, DEVICE)
            chimp_scores = score_dataset(model, chimp_ds, DEVICE)

            model_result = {}

            # Spearman r between human and chimp scores
            for head_name in ["binary"] + PER_ENZYME_HEADS:
                h = human_scores[head_name]
                c = chimp_scores[head_name]
                if np.std(h) > 0 and np.std(c) > 0:
                    r, p = stats.spearmanr(h, c)
                    model_result[head_name] = {
                        "spearman_r": round(float(r), 4),
                        "spearman_p": float(p),
                        "human_mean": round(float(np.mean(h)), 4),
                        "chimp_mean": round(float(np.mean(c)), 4),
                        "human_std": round(float(np.std(h)), 4),
                        "chimp_std": round(float(np.std(c)), 4),
                    }
                else:
                    model_result[head_name] = {"spearman_r": float("nan")}

            cross_results[model_name] = model_result

        results["cross_species"] = cross_results
        save_results(results)

    except Exception as e:
        logger.error(f"Cross-species ERROR: {e}")
        logger.error(traceback.format_exc())
        results["cross_species"] = {"error": str(e)}

    return results


# ---------------------------------------------------------------------------
# Step 6: gnomAD gene-level
# ---------------------------------------------------------------------------


def score_gnomad(models: Dict[str, nn.Module], v3_data: Dict, results: Dict):
    """Gene-level gnomAD constraint correlations."""
    logger.info(f"\n{'='*70}")
    logger.info("gnomAD Gene-Level Correlations")
    logger.info(f"{'='*70}")

    try:
        # Score v3 training data
        all_idx = np.arange(len(v3_data["labels_binary"]))
        ds = ScoringDataset(
            v3_data["rnafm_features"],
            v3_data["hand_features"],
            bp_sub=v3_data["bp_submatrices"],
        )

        gnomad_path = PROJECT_ROOT / "data" / "raw" / "gnomad" / "gnomad_v4.1_constraint.tsv"
        gnomad_df = pd.read_csv(gnomad_path, sep="\t")
        # Keep canonical/level=2 only
        gnomad_df = gnomad_df[
            ((gnomad_df["canonical"] == True) | (gnomad_df["level"] == 2.0))
        ].drop_duplicates(subset=["gene"]).set_index("gene")
        logger.info(f"  gnomAD genes: {len(gnomad_df)}")

        df = v3_data["df"]
        pos_mask = v3_data["labels_binary"] == 1

        gnomad_results = {}

        for model_name, model in models.items():
            logger.info(f"  Scoring with {model_name}...")
            scores = score_dataset(model, ds, DEVICE)

            model_result = {}

            for head_name in ["binary"] + PER_ENZYME_HEADS:
                head_scores = scores[head_name]

                # Per-gene mean score (positives only)
                gene_scores = {}
                for i in range(len(df)):
                    if not pos_mask[i]:
                        continue
                    gene = df.iloc[i].get("gene", None)
                    if gene and isinstance(gene, str):
                        if gene not in gene_scores:
                            gene_scores[gene] = []
                        gene_scores[gene].append(head_scores[i])

                gene_means = {g: np.mean(s) for g, s in gene_scores.items() if len(s) >= 2}

                # Correlate with LOEUF and missense Z
                common_genes = set(gene_means.keys()) & set(gnomad_df.index)
                if len(common_genes) < 20:
                    model_result[head_name] = {"error": f"only {len(common_genes)} common genes"}
                    continue

                gs = np.array([gene_means[g] for g in common_genes])

                for metric in ["lof.oe_ci.upper", "mis.z_score"]:
                    metric_vals = []
                    gs_filtered = []
                    for g in common_genes:
                        v = gnomad_df.loc[g, metric]
                        if pd.notna(v):
                            try:
                                metric_vals.append(float(v))
                                gs_filtered.append(gene_means[g])
                            except (ValueError, TypeError):
                                continue

                    if len(metric_vals) >= 20:
                        r, p = stats.spearmanr(gs_filtered, metric_vals)
                        metric_key = "LOEUF" if "lof" in metric else "missense_Z"
                        model_result.setdefault(head_name, {})[metric_key] = {
                            "spearman_r": round(float(r), 4),
                            "spearman_p": float(p),
                            "n_genes": len(metric_vals),
                        }

            gnomad_results[model_name] = model_result

        results["gnomad"] = gnomad_results
        save_results(results)

    except Exception as e:
        logger.error(f"gnomAD ERROR: {e}")
        logger.error(traceback.format_exc())
        results["gnomad"] = {"error": str(e)}

    return results


# ---------------------------------------------------------------------------
# Results I/O
# ---------------------------------------------------------------------------


def save_results(results):
    """Save results incrementally."""

    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            v = float(obj)
            if np.isnan(v) or np.isinf(v):
                return None
            return v
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        return obj

    with open(RESULTS_PATH, "w") as f:
        json.dump(_clean(results), f, indent=2)
    logger.info(f"  Results saved to {RESULTS_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    total_t0 = time.time()
    results = {}

    logger.info("=" * 70)
    logger.info("V4Deep Replication Experiment")
    logger.info(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------
    v3_data = load_v3_data()

    logger.info("\nLoading v4-large data...")
    v4_data = load_v4_large(v3_data)

    # -----------------------------------------------------------------------
    # Step 1: Train 3 models on FULL data
    # -----------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("STEP 1: Train 3 models on FULL data")
    logger.info("=" * 70)

    # Model 1: A8+T1+H4
    model_t1_h4 = train_full_t1(HierarchicalAttention_H4, v3_data, name="A8_T1_H4")

    # Model 2: A8+T6+H4
    model_t6_h4 = train_full_t6(HierarchicalAttention_H4, v3_data, name="A8_T6_H4")

    # Model 3: A8+T4+H1
    model_t4_h1 = train_full_t4(HierarchicalAttention_H1, v3_data, v4_data, name="A8_T4_H1")

    models = {
        "A8_T1_H4": model_t1_h4,
        "A8_T6_H4": model_t6_h4,
        "A8_T4_H1": model_t4_h1,
    }

    # -----------------------------------------------------------------------
    # Step 2: 5-fold CV
    # -----------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: 5-Fold CV Classification")
    logger.info("=" * 70)

    try:
        cv_t1_h4 = run_5fold_cv(HierarchicalAttention_H4, v3_data,
                                 training_fn=None, training_kwargs={},
                                 combo_name="A8_T1_H4")
        results.setdefault("classification_5fold", {})["A8_T1_H4"] = cv_t1_h4
        save_results(results)
    except Exception as e:
        logger.error(f"CV A8_T1_H4 failed: {e}")
        logger.error(traceback.format_exc())

    try:
        cv_t6_h4 = run_5fold_cv(HierarchicalAttention_H4, v3_data,
                                 training_fn=None, training_kwargs={},
                                 combo_name="A8_T6_H4")
        results.setdefault("classification_5fold", {})["A8_T6_H4"] = cv_t6_h4
        save_results(results)
    except Exception as e:
        logger.error(f"CV A8_T6_H4 failed: {e}")
        logger.error(traceback.format_exc())

    try:
        cv_t4_h1 = run_5fold_cv_t4(HierarchicalAttention_H1, v4_data, combo_name="A8_T4_H1")
        results.setdefault("classification_5fold", {})["A8_T4_H1"] = cv_t4_h1
        save_results(results)
    except Exception as e:
        logger.error(f"CV A8_T4_H1 failed: {e}")
        logger.error(traceback.format_exc())

    # -----------------------------------------------------------------------
    # Step 3: TCGA Scoring
    # -----------------------------------------------------------------------
    try:
        results = score_tcga(models, results)
    except Exception as e:
        logger.error(f"TCGA scoring failed: {e}")
        logger.error(traceback.format_exc())

    # -----------------------------------------------------------------------
    # Step 4: ClinVar Scoring
    # -----------------------------------------------------------------------
    try:
        results = score_clinvar(models, results)
    except Exception as e:
        logger.error(f"ClinVar scoring failed: {e}")
        logger.error(traceback.format_exc())

    # -----------------------------------------------------------------------
    # Step 5: Cross-species
    # -----------------------------------------------------------------------
    try:
        results = score_cross_species(models, v3_data, results)
    except Exception as e:
        logger.error(f"Cross-species scoring failed: {e}")
        logger.error(traceback.format_exc())

    # -----------------------------------------------------------------------
    # Step 6: gnomAD
    # -----------------------------------------------------------------------
    try:
        results = score_gnomad(models, v3_data, results)
    except Exception as e:
        logger.error(f"gnomAD scoring failed: {e}")
        logger.error(traceback.format_exc())

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------
    total_time = time.time() - total_t0
    results["total_time_minutes"] = round(total_time / 60, 1)
    save_results(results)

    logger.info(f"\n{'='*70}")
    logger.info(f"V4Deep Replication COMPLETE in {total_time/60:.1f} minutes")
    logger.info(f"Results: {RESULTS_PATH}")
    logger.info(f"{'='*70}")

    # Print summary
    if "classification_5fold" in results:
        logger.info("\nClassification 5-fold summary:")
        for name, cv in results["classification_5fold"].items():
            if isinstance(cv, dict) and "overall_auroc_mean" in cv:
                logger.info(f"  {name}: AUROC={cv['overall_auroc_mean']:.4f} +/- {cv.get('overall_auroc_std', 0):.4f}")
                if "per_enzyme_auroc_mean" in cv:
                    for enz, auc in cv["per_enzyme_auroc_mean"].items():
                        logger.info(f"    {enz}: {auc:.4f}")

    if "tcga" in results:
        logger.info("\nTCGA summary (binary p90 OR):")
        for cancer, cr in results["tcga"].items():
            if isinstance(cr, dict) and "error" not in cr:
                for mn in models:
                    if mn in cr and "binary" in cr[mn]:
                        p90 = cr[mn]["binary"].get("p90", {})
                        logger.info(f"  {cancer}/{mn}: OR={p90.get('OR', 'N/A'):.3f} p={p90.get('p', 1.0):.2e}")

    if "clinvar" in results:
        logger.info("\nClinVar summary (binary p50 OR):")
        for mn in models:
            if mn in results.get("clinvar", {}):
                p50 = results["clinvar"][mn].get("binary", {}).get("p50", {})
                logger.info(f"  {mn}: OR={p50.get('OR', 'N/A'):.3f} p={p50.get('p', 1.0):.2e}")


if __name__ == "__main__":
    main()
