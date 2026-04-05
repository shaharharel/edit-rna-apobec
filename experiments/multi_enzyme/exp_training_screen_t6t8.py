#!/usr/bin/env python
"""Training methods screen T6-T8 for RNA editing architecture search.

Tests advanced training methods on top 2 architectures (A6_Conv2DBP, A8_HierarchicalAttention):
  T6. Structure pretext pretraining  — predict unpaired positions, then fine-tune
  T7. Contrastive pretraining        — NT-Xent on orig/edited embedding pairs
  T8. Meta-learning (prototypical)   — episode-based training with enzyme tasks

Output: experiments/multi_enzyme/outputs/architecture_screen/training_screen_t6t8_results.json
Run: /opt/miniconda3/envs/quris/bin/python -u experiments/multi_enzyme/exp_training_screen_t6t8.py
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
        logging.FileHandler(OUTPUT_DIR / "training_screen_t6t8.log", mode="w"),
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
# Architecture classes — copied from exp_training_screen.py
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

    def encode(self, batch):
        """Return shared embedding without heads."""
        bp = batch["bp_submatrix"]
        conv_out = self.conv(bp)
        conv_out = conv_out.flatten(1)
        conv_out = self.conv_fc(conv_out)
        fused = torch.cat([conv_out, batch["rnafm"], batch["hand_feat"]], dim=-1)
        shared = self.encoder(fused)
        return shared


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

    def encode(self, batch):
        """Return shared embedding without heads."""
        bp = batch["bp_submatrix"].squeeze(1)
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
        return shared


ARCHITECTURE_CLASSES = {
    "A6_Conv2DBP": Conv2DBP,
    "A8_HierarchicalAttention": HierarchicalAttention,
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


class ContrastiveDataset(Dataset):
    """Dataset that provides both orig and edited RNA-FM embeddings for contrastive learning."""

    def __init__(self, indices, data):
        self.indices = indices
        self.data = data

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        return {
            "rnafm": torch.from_numpy(self.data["rnafm_features"][i]),
            "rnafm_edited": torch.from_numpy(self.data["rnafm_edited_features"][i]),
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
    """Load the v3 base dataset with both orig and edited RNA-FM embeddings."""
    logger.info("=" * 70)
    logger.info("Loading V3 dataset for T6-T8")
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

    # RNA-FM embeddings (original + edited)
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
    logger.info(f"  {n} v3 sites with sequences")

    # Hand features (40-dim)
    hand_features = build_hand_features(site_ids, sequences, structure_delta, loop_df)

    # RNA-FM features: original, edited, delta
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

    # Pretext labels: 1 if center position is unpaired (dot in structure), 0 if paired
    pretext_labels = np.zeros(n, dtype=np.float32)
    n_unpaired = 0
    for i, sid in enumerate(site_ids):
        if sid in dot_brackets:
            db = dot_brackets[sid]
            if len(db) > CENTER:
                if db[CENTER] == ".":
                    pretext_labels[i] = 1.0
                    n_unpaired += 1
                # else: paired (0.0), which is default
            else:
                pretext_labels[i] = 0.5  # unknown
        else:
            pretext_labels[i] = 0.5  # unknown
    logger.info(f"  Pretext labels: {n_unpaired} unpaired, {n - n_unpaired} paired/unknown at center")

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

    logger.info(f"  Positives: {int(labels_binary.sum())}, Negatives: {int((labels_binary == 0).sum())}")

    return {
        "site_ids": site_ids,
        "df": df,
        "sequences": sequences,
        "hand_features": hand_features,
        "rnafm_features": rnafm_matrix,
        "rnafm_edited_features": rnafm_edited_matrix,
        "edit_delta_features": edit_delta_matrix,
        "bp_submatrices": bp_sub,
        "dot_brackets": dot_brackets,
        "pretext_labels": pretext_labels,
        "labels_binary": labels_binary,
        "labels_enzyme": labels_enzyme,
        "per_enzyme_labels": per_enzyme_labels,
    }


# ---------------------------------------------------------------------------
# Standard training / evaluation (same as exp_training_screen.py)
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
# T6: Structure Pretext Pretraining
# ---------------------------------------------------------------------------


class PretextHead(nn.Module):
    """Simple binary head for pretext prediction (replaces task heads during pretraining)."""

    def __init__(self):
        super().__init__()
        self.head = nn.Linear(D_SHARED, 1)

    def forward(self, shared):
        return self.head(shared).squeeze(-1)


def pretrain_pretext(model, pretext_head, train_loader, device, n_epochs=10, lr=1e-3):
    """Pretrain the shared encoder on the structure pretext task."""
    model.train()
    pretext_head.train()

    # Optimize shared params + pretext head
    optimizer = torch.optim.Adam(
        list(model.get_shared_params()) + list(pretext_head.parameters()),
        lr=lr, weight_decay=1e-4,
    )

    for epoch in range(n_epochs):
        total_loss = 0.0
        n_batches = 0
        n_correct = 0
        n_total = 0

        for batch in train_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            optimizer.zero_grad()

            # Get shared embedding via encode method
            shared = model.encode(batch)
            logits = pretext_head(shared)
            labels = batch["pretext_label"]

            # Skip ambiguous labels (0.5)
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

            # Accuracy
            preds = (torch.sigmoid(logits[mask]) > 0.5).float()
            n_correct += (preds == labels[mask]).sum().item()
            n_total += mask.sum().item()

        avg_loss = total_loss / max(n_batches, 1)
        acc = n_correct / max(n_total, 1)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(f"    Pretext epoch {epoch + 1:2d} | loss={avg_loss:.4f} | acc={acc:.3f}")


def run_t6_pretext_pretraining(arch_cls, arch_name: str, data: Dict) -> Dict:
    """T6: Structure pretext pretraining -> fine-tune on editing classification."""
    logger.info(f"\n{'=' * 70}")
    logger.info(f"T6: Structure Pretext Pretraining — {arch_name}")
    logger.info(f"{'=' * 70}")

    n = len(data["labels_binary"])
    strat_key = data["labels_binary"].astype(np.int64)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    fold_results = []
    total_t0 = time.time()

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(n), strat_key)):
        logger.info(f"\n--- Fold {fold + 1}/{N_FOLDS} ({len(train_idx)} train, {len(val_idx)} val) ---")
        fold_t0 = time.time()

        fold_model = arch_cls().to(DEVICE)
        pretext_head = PretextHead().to(DEVICE)

        # Stage 0: Pretext pretraining on ALL training data
        logger.info(f"  Stage 0: Pretext pretraining (10 epochs)")
        pretext_ds = PretextDataset(train_idx, data)
        pretext_loader = DataLoader(pretext_ds, batch_size=BATCH_SIZE, shuffle=True,
                                    collate_fn=standard_collate, num_workers=0)
        pretrain_pretext(fold_model, pretext_head, pretext_loader, DEVICE,
                         n_epochs=10, lr=1e-3)

        del pretext_head  # No longer needed
        gc.collect()

        # Re-initialize heads (pretext head was separate, task heads are fresh)
        # The shared encoder is pretrained, heads are randomly initialized
        fold_model._init_heads()
        fold_model = fold_model.to(DEVICE)

        # Stage 1: Fine-tune joint (10 epochs)
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
            avg_loss, avg_binary = train_one_epoch(fold_model, train_loader, optimizer, DEVICE)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                val_auroc, val_per_enz = evaluate(fold_model, val_loader, DEVICE)
                enz_str = " ".join(f"{e}={v:.3f}" for e, v in val_per_enz.items() if not np.isnan(v))
                logger.info(
                    f"    Epoch {epoch + 1:2d} | loss={avg_loss:.4f} bce={avg_binary:.4f} "
                    f"| val_auroc={val_auroc:.4f} | {enz_str}"
                )

        # Stage 2: Adapter-only fine-tuning (20 epochs)
        logger.info(f"  Stage 2: Adapter-only ({STAGE2_EPOCHS} epochs)")
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

        # Final evaluation
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
# T7: Contrastive Pretraining (NT-Xent / InfoNCE)
# ---------------------------------------------------------------------------


class ProjectionHead(nn.Module):
    """Projection head for contrastive learning: shared_embed -> 64-dim."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(D_SHARED, 64),
            nn.GELU(),
            nn.Linear(64, 64),
        )

    def forward(self, x):
        return F.normalize(self.proj(x), dim=-1)


def nt_xent_loss(z_orig, z_edited, temperature=0.07):
    """NT-Xent (InfoNCE) contrastive loss.

    Positive pairs: (z_orig[i], z_edited[i])
    Negative pairs: all other combinations in the batch.
    """
    B = z_orig.size(0)
    if B < 2:
        return torch.tensor(0.0, device=z_orig.device)

    # Concatenate: [z_orig; z_edited] -> [2B, D]
    z = torch.cat([z_orig, z_edited], dim=0)  # [2B, D]

    # Similarity matrix [2B, 2B]
    sim = torch.mm(z, z.t()) / temperature  # [2B, 2B]

    # Mask out self-similarity
    mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, -1e9)

    # Positive pairs: (i, i+B) and (i+B, i)
    labels = torch.cat([
        torch.arange(B, 2 * B, device=z.device),
        torch.arange(0, B, device=z.device),
    ])  # [2B]

    loss = F.cross_entropy(sim, labels)
    return loss


def pretrain_contrastive(model, proj_head, train_loader, device, n_epochs=10, temperature=0.07, lr=1e-3):
    """Contrastive pretraining: learn to match orig/edited embedding pairs."""
    model.train()
    proj_head.train()

    optimizer = torch.optim.Adam(
        list(model.get_shared_params()) + list(proj_head.parameters()),
        lr=lr, weight_decay=1e-4,
    )

    for epoch in range(n_epochs):
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            optimizer.zero_grad()

            # Encode original
            shared_orig = model.encode(batch)
            z_orig = proj_head(shared_orig)

            # Encode edited: swap rnafm with rnafm_edited
            batch_edited = dict(batch)
            batch_edited["rnafm"] = batch["rnafm_edited"]
            # Recompute edit_delta as zero (since we're using the edited embedding as rnafm)
            # Actually, for the edited view, we use the edited embedding directly
            shared_edited = model.encode(batch_edited)
            z_edited = proj_head(shared_edited)

            loss = nt_xent_loss(z_orig, z_edited, temperature=temperature)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(proj_head.parameters()), 1.0
            )
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(f"    Contrastive epoch {epoch + 1:2d} | loss={avg_loss:.4f}")


def run_t7_contrastive_pretraining(arch_cls, arch_name: str, data: Dict) -> Dict:
    """T7: Contrastive pretraining on orig/edited pairs -> fine-tune on classification."""
    logger.info(f"\n{'=' * 70}")
    logger.info(f"T7: Contrastive Pretraining — {arch_name}")
    logger.info(f"{'=' * 70}")

    n = len(data["labels_binary"])
    strat_key = data["labels_binary"].astype(np.int64)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    fold_results = []
    total_t0 = time.time()

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(n), strat_key)):
        logger.info(f"\n--- Fold {fold + 1}/{N_FOLDS} ({len(train_idx)} train, {len(val_idx)} val) ---")
        fold_t0 = time.time()

        fold_model = arch_cls().to(DEVICE)
        proj_head = ProjectionHead().to(DEVICE)

        # Stage 0: Contrastive pretraining
        logger.info(f"  Stage 0: Contrastive pretraining (10 epochs, temperature=0.07)")
        contrast_ds = ContrastiveDataset(train_idx, data)
        contrast_loader = DataLoader(contrast_ds, batch_size=BATCH_SIZE, shuffle=True,
                                     collate_fn=standard_collate, num_workers=0)
        pretrain_contrastive(fold_model, proj_head, contrast_loader, DEVICE,
                             n_epochs=10, temperature=0.07, lr=1e-3)

        del proj_head  # No longer needed
        gc.collect()

        # Re-initialize heads for fine-tuning
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
            avg_loss, avg_binary = train_one_epoch(fold_model, train_loader, optimizer, DEVICE)
            if (epoch + 1) % 5 == 0 or epoch == 0:
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
            avg_loss, avg_binary = train_one_epoch(fold_model, train_loader, optimizer2, DEVICE)
            if (epoch + 1) % 5 == 0 or epoch == STAGE2_EPOCHS - 1:
                val_auroc, val_per_enz = evaluate(fold_model, val_loader, DEVICE)
                enz_str = " ".join(f"{e}={v:.3f}" for e, v in val_per_enz.items() if not np.isnan(v))
                logger.info(
                    f"    Epoch {STAGE1_EPOCHS + epoch + 1:2d} | loss={avg_loss:.4f} "
                    f"| val_auroc={val_auroc:.4f} | {enz_str}"
                )

        # Final evaluation
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
# T8: Meta-Learning (Prototypical Networks)
# ---------------------------------------------------------------------------


class ProtoNet(nn.Module):
    """Wraps an architecture encoder for prototypical network training.

    During meta-training, we compute prototypes from support sets and
    classify query examples by distance to prototypes.
    At evaluation time, we use ALL training data to compute prototypes.
    """

    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def encode(self, batch):
        return self.base_model.encode(batch)


def sample_episode(data, train_idx, enzyme_labels, rng, n_support=5, n_query=10):
    """Sample a prototypical network episode.

    Returns:
        support_indices: list of (enzyme_idx, [indices]) for 3-4 enzyme tasks
        query_indices: list of (enzyme_idx, [indices]) for the same tasks
    """
    # Group training indices by enzyme (only enzymes with enough samples)
    enzyme_groups = {}
    for idx in train_idx:
        enz = enzyme_labels[idx]
        if enz not in enzyme_groups:
            enzyme_groups[enz] = []
        enzyme_groups[enz].append(idx)

    # Filter to enzymes with enough samples (need support + query for both pos and neg)
    min_samples = n_support + n_query
    valid_enzymes = []
    for enz, indices in enzyme_groups.items():
        if enz >= len(ENZYME_CLASSES):
            continue
        # Check that we have both positives and negatives
        pos_indices = [i for i in indices if data["labels_binary"][i] == 1]
        neg_indices = [i for i in indices if data["labels_binary"][i] == 0]
        if len(pos_indices) >= min_samples and len(neg_indices) >= min_samples:
            valid_enzymes.append(enz)

    if len(valid_enzymes) < 2:
        return None, None

    # Sample 3-4 enzyme tasks (or however many are available)
    n_tasks = min(rng.randint(3, 5), len(valid_enzymes))
    selected_enzymes = rng.choice(valid_enzymes, size=n_tasks, replace=False)

    support_data = []
    query_data = []

    for enz in selected_enzymes:
        pos_indices = [i for i in enzyme_groups[enz] if data["labels_binary"][i] == 1]
        neg_indices = [i for i in enzyme_groups[enz] if data["labels_binary"][i] == 0]

        # Sample support and query from positives
        pos_perm = rng.permutation(len(pos_indices))
        pos_support = [pos_indices[j] for j in pos_perm[:n_support]]
        pos_query = [pos_indices[j] for j in pos_perm[n_support:n_support + n_query]]

        # Sample support and query from negatives
        neg_perm = rng.permutation(len(neg_indices))
        neg_support = [neg_indices[j] for j in neg_perm[:n_support]]
        neg_query = [neg_indices[j] for j in neg_perm[n_support:n_support + n_query]]

        support_data.append({
            "enzyme": enz,
            "pos_indices": pos_support,
            "neg_indices": neg_support,
        })
        query_data.append({
            "enzyme": enz,
            "pos_indices": pos_query,
            "neg_indices": neg_query,
        })

    return support_data, query_data


def proto_episode_loss(model, data, support_data, query_data, device):
    """Compute prototypical network loss for one episode.

    For each enzyme task:
    - Compute prototype_pos = mean(encode(support_pos))
    - Compute prototype_neg = mean(encode(support_neg))
    - For each query: compute distance to both prototypes
    - Loss: cross-entropy over prototype distances
    """
    total_loss = torch.tensor(0.0, device=device)
    n_correct = 0
    n_total = 0

    for s_data, q_data in zip(support_data, query_data):
        # Encode support positives
        s_pos_batch = _make_batch(data, s_data["pos_indices"], device)
        s_pos_emb = model.encode(s_pos_batch)
        proto_pos = s_pos_emb.mean(dim=0)  # [D_SHARED]

        # Encode support negatives
        s_neg_batch = _make_batch(data, s_data["neg_indices"], device)
        s_neg_emb = model.encode(s_neg_batch)
        proto_neg = s_neg_emb.mean(dim=0)  # [D_SHARED]

        # Encode query examples
        q_pos_indices = q_data["pos_indices"]
        q_neg_indices = q_data["neg_indices"]
        all_q_indices = q_pos_indices + q_neg_indices
        q_labels = torch.tensor(
            [1.0] * len(q_pos_indices) + [0.0] * len(q_neg_indices),
            device=device
        )

        if len(all_q_indices) == 0:
            continue

        q_batch = _make_batch(data, all_q_indices, device)
        q_emb = model.encode(q_batch)  # [Q, D_SHARED]

        # Distances to prototypes: negative squared Euclidean distance
        dist_pos = -torch.sum((q_emb - proto_pos.unsqueeze(0)) ** 2, dim=-1)  # [Q]
        dist_neg = -torch.sum((q_emb - proto_neg.unsqueeze(0)) ** 2, dim=-1)  # [Q]

        # Logits: [Q, 2] where class 0 = negative, class 1 = positive
        logits = torch.stack([dist_neg, dist_pos], dim=-1)  # [Q, 2]
        targets = q_labels.long()

        loss = F.cross_entropy(logits, targets)
        total_loss = total_loss + loss

        # Accuracy
        preds = logits.argmax(dim=-1)
        n_correct += (preds == targets).sum().item()
        n_total += len(targets)

    n_tasks = len(support_data)
    if n_tasks > 0:
        total_loss = total_loss / n_tasks

    return total_loss, n_correct, n_total


def _make_batch(data, indices, device):
    """Create a batch dict from data and indices, moved to device."""
    batch = {
        "rnafm": torch.from_numpy(data["rnafm_features"][indices]).to(device),
        "edit_delta": torch.from_numpy(data["edit_delta_features"][indices]).to(device),
        "bp_submatrix": torch.from_numpy(data["bp_submatrices"][indices]).unsqueeze(1).to(device),
        "hand_feat": torch.from_numpy(data["hand_features"][indices]).to(device),
    }
    return batch


def run_t8_metalearning(arch_cls, arch_name: str, data: Dict) -> Dict:
    """T8: Meta-learning with prototypical networks.

    Episode-based training, then evaluate with full-data prototypes.
    Also evaluates enzyme-holdout generalization.
    """
    logger.info(f"\n{'=' * 70}")
    logger.info(f"T8: Meta-Learning (Prototypical Networks) — {arch_name}")
    logger.info(f"{'=' * 70}")

    n = len(data["labels_binary"])
    strat_key = data["labels_binary"].astype(np.int64)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    fold_results = []
    enzyme_holdout_results = []
    total_t0 = time.time()

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(n), strat_key)):
        logger.info(f"\n--- Fold {fold + 1}/{N_FOLDS} ({len(train_idx)} train, {len(val_idx)} val) ---")
        fold_t0 = time.time()

        base_model = arch_cls().to(DEVICE)
        proto_net = ProtoNet(base_model).to(DEVICE)

        # Meta-training: 200 episodes per epoch, 20 epochs
        n_episodes_per_epoch = 200
        n_meta_epochs = 20
        rng = np.random.RandomState(SEED + fold)

        optimizer = torch.optim.Adam(
            proto_net.parameters(), lr=1e-3, weight_decay=1e-4,
        )

        logger.info(f"  Meta-training: {n_meta_epochs} epochs x {n_episodes_per_epoch} episodes")

        for epoch in range(n_meta_epochs):
            proto_net.train()
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            n_valid_episodes = 0

            for ep in range(n_episodes_per_epoch):
                support_data_ep, query_data_ep = sample_episode(
                    data, train_idx, data["labels_enzyme"],
                    rng, n_support=5, n_query=10,
                )
                if support_data_ep is None:
                    continue

                optimizer.zero_grad()
                loss, nc, nt = proto_episode_loss(
                    proto_net, data, support_data_ep, query_data_ep, DEVICE
                )
                if loss.requires_grad:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(proto_net.parameters(), 1.0)
                    optimizer.step()

                epoch_loss += loss.item()
                epoch_correct += nc
                epoch_total += nt
                n_valid_episodes += 1

            avg_loss = epoch_loss / max(n_valid_episodes, 1)
            acc = epoch_correct / max(epoch_total, 1)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(
                    f"    Meta-epoch {epoch + 1:2d} | loss={avg_loss:.4f} | acc={acc:.3f} "
                    f"| valid_episodes={n_valid_episodes}"
                )

        # --- Evaluate using prototypes computed from full training data ---
        logger.info(f"  Computing prototypes from training data...")
        proto_net.eval()

        # Compute prototypes per enzyme (pos and neg)
        with torch.no_grad():
            # Encode all training data in batches
            train_embeddings = []
            for start in range(0, len(train_idx), BATCH_SIZE):
                end = min(start + BATCH_SIZE, len(train_idx))
                batch_indices = train_idx[start:end]
                batch = _make_batch(data, batch_indices, DEVICE)
                emb = proto_net.encode(batch)
                train_embeddings.append(emb.cpu())
            train_embeddings = torch.cat(train_embeddings, dim=0)  # [N_train, D_SHARED]

            # Compute prototype for positive and negative per enzyme
            proto_pos_per_enz = {}
            proto_neg_per_enz = {}
            for enz_idx, enz_name in enumerate(ENZYME_CLASSES):
                train_enz_mask = data["labels_enzyme"][train_idx] == enz_idx
                train_pos_mask = train_enz_mask & (data["labels_binary"][train_idx] == 1)
                train_neg_mask = train_enz_mask & (data["labels_binary"][train_idx] == 0)

                if train_pos_mask.sum() > 0:
                    proto_pos_per_enz[enz_name] = train_embeddings[train_pos_mask].mean(dim=0)
                if train_neg_mask.sum() > 0:
                    proto_neg_per_enz[enz_name] = train_embeddings[train_neg_mask].mean(dim=0)

            # Global prototypes (across all enzymes)
            global_pos_mask = data["labels_binary"][train_idx] == 1
            global_neg_mask = data["labels_binary"][train_idx] == 0
            global_proto_pos = train_embeddings[global_pos_mask].mean(dim=0)
            global_proto_neg = train_embeddings[global_neg_mask].mean(dim=0)

            # --- Evaluate: classify val set by nearest prototype ---
            val_embeddings = []
            for start in range(0, len(val_idx), BATCH_SIZE):
                end = min(start + BATCH_SIZE, len(val_idx))
                batch_indices = val_idx[start:end]
                batch = _make_batch(data, batch_indices, DEVICE)
                emb = proto_net.encode(batch)
                val_embeddings.append(emb.cpu())
            val_embeddings = torch.cat(val_embeddings, dim=0)  # [N_val, D_SHARED]

            # Global binary classification: distance to pos vs neg prototype
            dist_pos = -torch.sum((val_embeddings - global_proto_pos.unsqueeze(0)) ** 2, dim=-1)
            dist_neg = -torch.sum((val_embeddings - global_proto_neg.unsqueeze(0)) ** 2, dim=-1)
            # Score: softmax probability of positive
            proto_scores = torch.softmax(torch.stack([dist_neg, dist_pos], dim=-1), dim=-1)[:, 1]
            proto_scores = proto_scores.numpy()

            val_labels = data["labels_binary"][val_idx]

            try:
                overall_auroc = roc_auc_score(val_labels, proto_scores)
            except ValueError:
                overall_auroc = 0.5

            # Per-enzyme evaluation using enzyme-specific prototypes
            per_enzyme_aurocs = {}
            for h_idx, enz_name in enumerate(PER_ENZYME_HEADS):
                enz_idx = ENZYME_TO_IDX[enz_name]
                enz_mask = data["labels_enzyme"][val_idx] == enz_idx
                if enz_mask.sum() < 10:
                    per_enzyme_aurocs[enz_name] = float("nan")
                    continue

                enz_labels = data["labels_binary"][val_idx][enz_mask]
                if len(np.unique(enz_labels)) < 2:
                    per_enzyme_aurocs[enz_name] = float("nan")
                    continue

                # Use enzyme-specific prototypes if available, else global
                if enz_name in proto_pos_per_enz and enz_name in proto_neg_per_enz:
                    pp = proto_pos_per_enz[enz_name]
                    pn = proto_neg_per_enz[enz_name]
                else:
                    pp = global_proto_pos
                    pn = global_proto_neg

                enz_emb = val_embeddings[enz_mask]
                dp = -torch.sum((enz_emb - pp.unsqueeze(0)) ** 2, dim=-1)
                dn = -torch.sum((enz_emb - pn.unsqueeze(0)) ** 2, dim=-1)
                enz_scores = torch.softmax(torch.stack([dn, dp], dim=-1), dim=-1)[:, 1].numpy()

                try:
                    per_enzyme_aurocs[enz_name] = roc_auc_score(enz_labels, enz_scores)
                except ValueError:
                    per_enzyme_aurocs[enz_name] = 0.5

        fold_time = time.time() - fold_t0
        logger.info(f"  Fold {fold + 1} FINAL: overall_auroc={overall_auroc:.4f} | time={fold_time:.0f}s")
        for enz, auroc in per_enzyme_aurocs.items():
            if not np.isnan(auroc):
                logger.info(f"    {enz}: {auroc:.4f}")

        fold_results.append({
            "fold": fold + 1,
            "overall_auroc": overall_auroc,
            "per_enzyme_aurocs": per_enzyme_aurocs,
            "time_s": fold_time,
        })

        # --- Enzyme-holdout evaluation ---
        logger.info(f"  Enzyme-holdout evaluation...")
        holdout_results_fold = {}
        for holdout_enz_idx, holdout_enz in enumerate(PER_ENZYME_HEADS):
            enz_idx_val = ENZYME_TO_IDX[holdout_enz]

            # Train prototypes from all enzymes EXCEPT holdout
            other_pos_embs = []
            other_neg_embs = []
            for enz_idx2, enz_name2 in enumerate(ENZYME_CLASSES):
                if enz_idx2 == enz_idx_val:
                    continue
                m_pos = (data["labels_enzyme"][train_idx] == enz_idx2) & (data["labels_binary"][train_idx] == 1)
                m_neg = (data["labels_enzyme"][train_idx] == enz_idx2) & (data["labels_binary"][train_idx] == 0)
                if m_pos.sum() > 0:
                    other_pos_embs.append(train_embeddings[m_pos])
                if m_neg.sum() > 0:
                    other_neg_embs.append(train_embeddings[m_neg])

            if not other_pos_embs or not other_neg_embs:
                holdout_results_fold[holdout_enz] = float("nan")
                continue

            holdout_proto_pos = torch.cat(other_pos_embs).mean(dim=0)
            holdout_proto_neg = torch.cat(other_neg_embs).mean(dim=0)

            # Evaluate on holdout enzyme in val set
            holdout_mask = data["labels_enzyme"][val_idx] == enz_idx_val
            if holdout_mask.sum() < 10:
                holdout_results_fold[holdout_enz] = float("nan")
                continue

            holdout_labels = data["labels_binary"][val_idx][holdout_mask]
            if len(np.unique(holdout_labels)) < 2:
                holdout_results_fold[holdout_enz] = float("nan")
                continue

            h_emb = val_embeddings[holdout_mask]
            dp = -torch.sum((h_emb - holdout_proto_pos.unsqueeze(0)) ** 2, dim=-1)
            dn = -torch.sum((h_emb - holdout_proto_neg.unsqueeze(0)) ** 2, dim=-1)
            h_scores = torch.softmax(torch.stack([dn, dp], dim=-1), dim=-1)[:, 1].numpy()

            try:
                holdout_results_fold[holdout_enz] = roc_auc_score(holdout_labels, h_scores)
            except ValueError:
                holdout_results_fold[holdout_enz] = 0.5

        enzyme_holdout_results.append(holdout_results_fold)
        for enz, auroc in holdout_results_fold.items():
            if not np.isnan(auroc):
                logger.info(f"    Holdout {enz}: {auroc:.4f}")

        del proto_net, base_model, optimizer
        gc.collect()
        if DEVICE.type == "mps":
            torch.mps.empty_cache()

    total_time = time.time() - total_t0
    result = _aggregate_fold_results(fold_results, total_time)

    # Add enzyme-holdout results
    holdout_means = {}
    for enz in PER_ENZYME_HEADS:
        vals = [r.get(enz, float("nan")) for r in enzyme_holdout_results]
        vals = [v for v in vals if not np.isnan(v)]
        if vals:
            holdout_means[enz] = round(float(np.mean(vals)), 4)
    result["enzyme_holdout_aurocs"] = holdout_means

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save_results(all_results: Dict, path: Path):
    """Save results incrementally."""
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)


def _log_result(arch_name: str, method: str, result: Dict):
    """Log a single result."""
    if "error" in result:
        logger.info(f"  {arch_name} + {method}: FAILED -- {result['error']}")
        return
    overall = result.get("overall_auroc_mean", "N/A")
    pe = result.get("per_enzyme_auroc_mean", {})
    pe_str = " ".join(f"{e}={v:.3f}" for e, v in pe.items())
    time_s = result.get("total_time_s", 0)
    logger.info(f"  {arch_name} + {method}: overall={overall:.4f} | {pe_str} | {time_s:.0f}s")


def _print_summary_table(all_results: Dict, arch_list: List, training_methods: List):
    """Print comparison matrices."""
    logger.info("\n" + "=" * 100)
    logger.info("TRAINING SCREEN T6-T8 -- OVERALL AUROC")
    logger.info("=" * 100)

    header = f"{'Architecture':<28}"
    for method in training_methods:
        header += f" {method:>22}"
    logger.info(header)
    logger.info("-" * 100)

    for arch_name, _ in arch_list:
        row = f"{arch_name:<28}"
        for method in training_methods:
            r = all_results.get(arch_name, {}).get(method, {})
            if "error" in r:
                row += f" {'FAIL':>22}"
            elif "overall_auroc_mean" in r and r["overall_auroc_mean"] is not None:
                val = r["overall_auroc_mean"]
                std = r.get("overall_auroc_std", 0)
                row += f" {val:.4f}+/-{std:.4f}".rjust(22)
            else:
                row += f" {'N/A':>22}"
        logger.info(row)

    logger.info("=" * 100)

    # Per-enzyme tables
    for enz in PER_ENZYME_HEADS:
        logger.info(f"\n{'=' * 100}")
        logger.info(f"PER-ENZYME AUROC: {enz}")
        logger.info(f"{'=' * 100}")

        header = f"{'Architecture':<28}"
        for method in training_methods:
            header += f" {method:>22}"
        logger.info(header)
        logger.info("-" * 100)

        for arch_name, _ in arch_list:
            row = f"{arch_name:<28}"
            for method in training_methods:
                r = all_results.get(arch_name, {}).get(method, {})
                pe = r.get("per_enzyme_auroc_mean", {})
                if "error" in r:
                    row += f" {'FAIL':>22}"
                elif enz in pe:
                    val = pe[enz]
                    std = r.get("per_enzyme_auroc_std", {}).get(enz, 0)
                    row += f" {val:.4f}+/-{std:.4f}".rjust(22)
                else:
                    row += f" {'N/A':>22}"
            logger.info(row)

        logger.info("=" * 100)

    # Enzyme-holdout for T8
    logger.info(f"\n{'=' * 100}")
    logger.info("T8 ENZYME-HOLDOUT AUROC (train on 4 enzymes, predict 5th)")
    logger.info("=" * 100)
    for arch_name, _ in arch_list:
        r = all_results.get(arch_name, {}).get("T8_metalearning", {})
        holdout = r.get("enzyme_holdout_aurocs", {})
        if holdout:
            h_str = " ".join(f"{e}={v:.3f}" for e, v in holdout.items())
            logger.info(f"  {arch_name}: {h_str}")
        else:
            logger.info(f"  {arch_name}: N/A")
    logger.info("=" * 100)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    logger.info("=" * 70)
    logger.info("TRAINING METHODS SCREEN T6-T8")
    logger.info(f"Device: {DEVICE} | Folds: {N_FOLDS} | Seed: {SEED}")
    logger.info(f"Architectures: A6_Conv2DBP, A8_HierarchicalAttention")
    logger.info(f"Training methods: T6 (pretext), T7 (contrastive), T8 (prototypical)")
    logger.info("=" * 70)

    # Load v3 data
    data = load_v3_data()

    # Architecture list (top 2 only)
    arch_list = [
        ("A6_Conv2DBP", Conv2DBP),
        ("A8_HierarchicalAttention", HierarchicalAttention),
    ]

    training_methods = ["T6_pretext", "T7_contrastive", "T8_metalearning"]

    # Results matrix
    all_results = {}

    # Load existing incremental results
    results_path = OUTPUT_DIR / "training_screen_t6t8_results.json"
    if results_path.exists():
        try:
            with open(results_path) as f:
                all_results = json.load(f)
            logger.info(f"\nLoaded {len(all_results)} existing results from {results_path}")
        except Exception:
            all_results = {}

    # Run T6, T7, T8 for each architecture
    for arch_name, arch_cls in arch_list:
        if arch_name not in all_results:
            all_results[arch_name] = {}

        # T6: Structure pretext pretraining
        method = "T6_pretext"
        if method not in all_results[arch_name]:
            logger.info(f"\n{'=' * 70}")
            logger.info(f"Running {arch_name} + {method}")
            logger.info(f"{'=' * 70}")
            try:
                result = run_t6_pretext_pretraining(arch_cls, arch_name, data)
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

        # T7: Contrastive pretraining
        method = "T7_contrastive"
        if method not in all_results[arch_name]:
            logger.info(f"\n{'=' * 70}")
            logger.info(f"Running {arch_name} + {method}")
            logger.info(f"{'=' * 70}")
            try:
                result = run_t7_contrastive_pretraining(arch_cls, arch_name, data)
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

        # T8: Meta-learning (prototypical networks)
        method = "T8_metalearning"
        if method not in all_results[arch_name]:
            logger.info(f"\n{'=' * 70}")
            logger.info(f"Running {arch_name} + {method}")
            logger.info(f"{'=' * 70}")
            try:
                result = run_t8_metalearning(arch_cls, arch_name, data)
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

    # Summary table
    _print_summary_table(all_results, arch_list, training_methods)

    logger.info(f"\nResults saved to {results_path}")
    logger.info("\nTraining methods screen T6-T8 complete.")


if __name__ == "__main__":
    main()
