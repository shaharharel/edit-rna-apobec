#!/usr/bin/env python
"""Phase 3: Definitive multi-stream fusion architecture for RNA editing prediction.

Architecture:
  Input streams (all pre-computed):
    1. RNA-FM frozen pretrained → 640-dim pooled embedding per site
    2. ViennaRNA encoders:
       a. Conv2D on 41×41 BP probability submatrix → 128-dim
       b. Structure delta features → 16-dim (7 delta + 9 loop geometry)
       c. Edit delta embedding = rnafm_edited - rnafm_original → 640-dim
    3. 40-dim hand features (motif 24 + struct delta 7 + loop 9)

  Fusion:
    Concat [RNA-FM(640) + Conv2D(128) + edit_delta(640) + hand(40)] = 1448-dim
    → Linear(1448, 256) → GELU → Dropout(0.3) → LayerNorm
    → Linear(256, 128) → GELU → Dropout(0.2)
    → 128-dim shared embedding

  Heads:
    Binary head: Linear(128, 1) → sigmoid (all data)
    Per-enzyme adapters: Linear(128, 32) → GELU → Linear(32, 1) → sigmoid
      (trained only on enzyme-matched positives + negatives)
    Enzyme classifier: Linear(128, 6) → softmax (positives only)

Training:
  Stage 1 (10 epochs): Train everything jointly (lr=1e-3)
    Total loss = binary + 0.3 * sum(enzyme_losses) + 0.1 * classifier_loss
  Stage 2 (20 epochs): Freeze shared encoder, train only adapters (lr=5e-4)

Evaluation: 2-fold CV, overall + per-enzyme AUROC + enzyme classification accuracy.

Output: experiments/multi_enzyme/outputs/deep_architectures/
"""

import gc
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
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

from data.apobec_feature_extraction import build_hand_features, LOOP_FEATURE_COLS

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "multi_enzyme" / "outputs" / "deep_architectures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = OUTPUT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

log_file = LOG_DIR / "phase3_training.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, mode="w"),
    ],
)
logger = logging.getLogger(__name__)

DEVICE = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("mps")
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
D_CONV2D_OUT = 128
D_EDIT_DELTA = 640
D_HAND = 40
D_FUSION_IN = D_RNAFM + D_CONV2D_OUT + D_EDIT_DELTA + D_HAND  # 1448
D_SHARED = 128

# XGBoost baselines for comparison
XGBOOST_BASELINES = {
    "A3A": 0.907,
    "A3B": 0.831,
    "A3G": 0.931,
    "A3A_A3G": 0.941,
    "Neither": 0.840,
}
PHASE2B_BASELINES = {
    "A3A": 0.762,
    "A3B": 0.767,
    "A3G": 0.936,
}
UNIFIED_V1_BASELINES = {
    "A3A": 0.900,
    "A3G": 0.952,
}


# ---------------------------------------------------------------------------
# ViennaRNA folding (parallel) -- for BP probability submatrix
# ---------------------------------------------------------------------------


def _fold_and_bpp_worker(args):
    """Worker for parallel folding: returns (site_id, 41x41 submatrix)."""
    import RNA
    site_id, seq = args
    try:
        seq = seq.upper().replace("T", "U")
        n = len(seq)
        md = RNA.md()
        md.temperature = 37.0
        fc = RNA.fold_compound(seq, md)
        fc.mfe()
        fc.pf()
        bpp_raw = np.array(fc.bpp())
        bpp = bpp_raw[1:n + 1, 1:n + 1].astype(np.float32)
        # Extract 41x41 submatrix centered on edit site
        sub = bpp[80:121, 80:121].copy()
        # Zero out diagonal band (|i-j| < 3)
        for r in range(41):
            for c in range(41):
                if abs(r - c) < 3:
                    sub[r, c] = 0.0
        return site_id, sub
    except Exception:
        return site_id, np.zeros((41, 41), dtype=np.float32)


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------


def load_data() -> Dict:
    """Load v3 dataset with all pre-computed features."""
    logger.info("=" * 60)
    logger.info("Loading V3 dataset + pre-computed embeddings")
    logger.info("=" * 60)

    # --- Splits ---
    splits_path = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "splits_multi_enzyme_v3_with_negatives.csv"
    df = pd.read_csv(splits_path)
    logger.info(f"  Loaded {len(df)} sites from splits CSV")

    # --- Sequences ---
    seq_path = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "multi_enzyme_sequences_v3_with_negatives.json"
    with open(seq_path) as f:
        sequences = json.load(f)
    logger.info(f"  Loaded {len(sequences)} sequences")

    # --- Loop geometry ---
    loop_path = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "loop_position_per_site_v3.csv"
    loop_df = pd.read_csv(loop_path)
    loop_df = loop_df.drop_duplicates(subset=["site_id"]).set_index("site_id")

    # --- Structure delta ---
    struct_path = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "structure_cache_multi_enzyme_v3.npz"
    struct_data = np.load(struct_path, allow_pickle=True)
    struct_ids = list(struct_data["site_ids"])
    struct_deltas = struct_data["delta_features"]
    structure_delta = {sid: struct_deltas[i] for i, sid in enumerate(struct_ids)}
    logger.info(f"  Loaded structure cache: {len(structure_delta)} sites")

    # --- RNA-FM pooled embeddings (640-dim) ---
    rnafm_path = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "embeddings" / "rnafm_pooled_v3.pt"
    rnafm_emb = torch.load(rnafm_path, map_location="cpu")
    logger.info(f"  Loaded RNA-FM embeddings: {len(rnafm_emb)} sites")

    # --- RNA-FM edited embeddings (640-dim) ---
    rnafm_edited_path = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "embeddings" / "rnafm_pooled_edited_v3.pt"
    rnafm_edited_emb = torch.load(rnafm_edited_path, map_location="cpu")
    logger.info(f"  Loaded RNA-FM edited embeddings: {len(rnafm_edited_emb)} sites")

    # --- Filter to sites with sequences ---
    site_ids = df["site_id"].tolist()
    valid_mask = [sid in sequences for sid in site_ids]
    df = df[valid_mask].reset_index(drop=True)
    site_ids = df["site_id"].tolist()
    n = len(site_ids)
    logger.info(f"  {n} sites with sequences")

    # --- Hand features (40-dim) ---
    hand_features = build_hand_features(site_ids, sequences, structure_delta, loop_df)
    logger.info(f"  Hand features shape: {hand_features.shape}")

    # --- RNA-FM features: original, edited, delta (640-dim each) ---
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
    logger.info(f"  RNA-FM coverage: {n_rnafm_found}/{n} ({100*n_rnafm_found/n:.1f}%)")

    # Free memory
    del rnafm_emb, rnafm_edited_emb
    gc.collect()

    # --- Compute Conv2D BP submatrices (41x41 per site) via ViennaRNA ---
    logger.info("Computing ViennaRNA BP probability submatrices...")
    work_items = [(sid, sequences[sid]) for sid in site_ids]
    n_workers = min(14, os.cpu_count() or 4)
    bp_submatrices = np.zeros((n, 41, 41), dtype=np.float32)
    sid_to_idx = {sid: i for i, sid in enumerate(site_ids)}

    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_fold_and_bpp_worker, item): item[0] for item in work_items}
        for future in as_completed(futures):
            sid, sub = future.result()
            idx = sid_to_idx[sid]
            bp_submatrices[idx] = sub
            done += 1
            if done % 3000 == 0:
                logger.info(f"    Folded {done}/{n} ({time.time()-t0:.0f}s)")
    logger.info(f"  Folding complete: {done} sequences in {time.time()-t0:.0f}s")

    # --- Labels ---
    labels_binary = df["is_edited"].values.astype(np.float32)
    labels_enzyme = np.array([ENZYME_TO_IDX.get(e, 5) for e in df["enzyme"].values], dtype=np.int64)

    # --- Per-enzyme labels ---
    # Each enzyme head sees: (enzyme X positives, label=1) + (enzyme X negatives, label=0)
    # In v3, negatives already have enzyme assignments matching their positive counterparts
    per_enzyme_labels = np.full((n, N_PER_ENZYME), -1, dtype=np.float32)  # -1 = ignore
    for head_idx, enz_name in enumerate(PER_ENZYME_HEADS):
        enz_idx = ENZYME_TO_IDX[enz_name]
        # Positives for this enzyme
        pos_mask = (labels_binary == 1) & (labels_enzyme == enz_idx)
        per_enzyme_labels[pos_mask, head_idx] = 1.0
        # Negatives for this enzyme (negatives carry the enzyme they were generated for)
        neg_mask = (labels_binary == 0) & (labels_enzyme == enz_idx)
        per_enzyme_labels[neg_mask, head_idx] = 0.0

    logger.info("Per-enzyme head sample counts (enzyme-matched negatives):")
    for head_idx, enz_name in enumerate(PER_ENZYME_HEADS):
        n_pos = int((per_enzyme_labels[:, head_idx] == 1).sum())
        n_neg = int((per_enzyme_labels[:, head_idx] == 0).sum())
        logger.info(f"  {enz_name}: {n_pos} pos, {n_neg} neg")

    logger.info("Data loading complete.")
    return {
        "site_ids": site_ids,
        "df": df,
        "hand_features": hand_features,
        "rnafm_features": rnafm_matrix,
        "edit_delta_features": edit_delta_matrix,
        "bp_submatrices": bp_submatrices,
        "labels_binary": labels_binary,
        "labels_enzyme": labels_enzyme,
        "per_enzyme_labels": per_enzyme_labels,
    }


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class Phase3Dataset(Dataset):
    """Dataset for Phase 3 multi-stream model."""

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
            "bp_submatrix": torch.from_numpy(self.data["bp_submatrices"][i]).unsqueeze(0),  # [1, 41, 41]
            "hand_feat": torch.from_numpy(self.data["hand_features"][i]),
            "label_binary": torch.tensor(self.data["labels_binary"][i]),
            "label_enzyme": torch.tensor(self.data["labels_enzyme"][i]),
            "per_enzyme_labels": torch.from_numpy(self.data["per_enzyme_labels"][i]),
        }


def collate_fn(batch_list):
    """Standard collate: stack all tensors."""
    result = {}
    for key in batch_list[0]:
        result[key] = torch.stack([b[key] for b in batch_list])
    return result


# ---------------------------------------------------------------------------
# Model: Phase 3 Multi-Stream Fusion
# ---------------------------------------------------------------------------


class Conv2DEncoder(nn.Module):
    """Conv2D encoder on 41x41 base-pair probability submatrix → 128-dim."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.AdaptiveMaxPool2d(1),
        )
        self.fc = nn.Linear(64, D_CONV2D_OUT)

    def forward(self, bp_sub):
        """bp_sub: [B, 1, 41, 41] → [B, 128]"""
        x = self.conv(bp_sub).squeeze(-1).squeeze(-1)  # [B, 64]
        return self.fc(x)  # [B, 128]


class Phase3FusionModel(nn.Module):
    """Multi-stream fusion architecture.

    Streams:
      1. RNA-FM (640-dim) — pre-computed frozen embedding
      2. Conv2D on BP probability (41x41) → 128-dim
      3. Edit delta (640-dim) — rnafm_edited - rnafm_original
      4. Hand features (40-dim)

    Fusion: concat → 1448-dim → shared encoder → 128-dim

    Heads:
      - Binary: all data
      - Per-enzyme adapters: enzyme-matched data only
      - Enzyme classifier: positives only
    """

    def __init__(self):
        super().__init__()

        # Conv2D encoder for BP probability submatrix
        self.conv2d_encoder = Conv2DEncoder()

        # Shared fusion encoder
        self.shared_encoder = nn.Sequential(
            nn.Linear(D_FUSION_IN, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.LayerNorm(256),
            nn.Linear(256, D_SHARED),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        # Binary head (all data)
        self.binary_head = nn.Linear(D_SHARED, 1)

        # Per-enzyme adapter heads (enzyme-specific)
        self.enzyme_adapters = nn.ModuleDict({
            enz: nn.Sequential(
                nn.Linear(D_SHARED, 32),
                nn.GELU(),
                nn.Linear(32, 1),
            )
            for enz in PER_ENZYME_HEADS
        })

        # Enzyme classifier (6-class, positives only)
        self.enzyme_classifier = nn.Sequential(
            nn.Linear(D_SHARED, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, N_ENZYMES),
        )

    def get_shared_params(self):
        """Return parameters of the shared encoder + Conv2D encoder."""
        params = list(self.conv2d_encoder.parameters()) + list(self.shared_encoder.parameters())
        return params

    def get_adapter_params(self):
        """Return parameters of adapter heads + binary head + enzyme classifier."""
        params = list(self.binary_head.parameters())
        for enz in PER_ENZYME_HEADS:
            params.extend(self.enzyme_adapters[enz].parameters())
        params.extend(self.enzyme_classifier.parameters())
        return params

    def forward(self, batch):
        """
        Returns:
            binary_logit: [B]
            per_enzyme_logits: list of [B] tensors, one per enzyme head
            enzyme_cls_logits: [B, 6]
            shared_repr: [B, 128] (for analysis)
        """
        # Stream 1: RNA-FM (640-dim)
        rnafm = batch["rnafm"]  # [B, 640]

        # Stream 2: Conv2D on BP matrix (41x41 → 128)
        conv2d_out = self.conv2d_encoder(batch["bp_submatrix"])  # [B, 128]

        # Stream 3: Edit delta (640-dim)
        edit_delta = batch["edit_delta"]  # [B, 640]

        # Stream 4: Hand features (40-dim)
        hand = batch["hand_feat"]  # [B, 40]

        # Fusion: concat all streams → 1448-dim
        fused = torch.cat([rnafm, conv2d_out, edit_delta, hand], dim=-1)  # [B, 1448]

        # Shared encoder → 128-dim
        shared = self.shared_encoder(fused)  # [B, 128]

        # Binary head
        binary_logit = self.binary_head(shared).squeeze(-1)  # [B]

        # Per-enzyme adapter heads
        per_enzyme_logits = []
        for enz in PER_ENZYME_HEADS:
            logit = self.enzyme_adapters[enz](shared).squeeze(-1)  # [B]
            per_enzyme_logits.append(logit)

        # Enzyme classifier
        enzyme_cls_logits = self.enzyme_classifier(shared)  # [B, 6]

        return binary_logit, per_enzyme_logits, enzyme_cls_logits, shared


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------


def compute_loss(binary_logit, per_enzyme_logits, enzyme_cls_logits, batch,
                 enzyme_head_weight, enzyme_cls_weight):
    """Compute combined multi-head loss.

    Total = binary_loss + enzyme_head_weight * mean(per_enzyme_losses) + enzyme_cls_weight * cls_loss
    """
    device = binary_logit.device

    # Binary loss (all samples)
    loss_binary = F.binary_cross_entropy_with_logits(binary_logit, batch["label_binary"])

    # Per-enzyme head losses (enzyme-matched data only)
    per_enzyme_labels = batch["per_enzyme_labels"]  # [B, N_PER_ENZYME]
    loss_per_enzyme = torch.tensor(0.0, device=device)
    n_active_heads = 0

    for head_idx in range(N_PER_ENZYME):
        mask = per_enzyme_labels[:, head_idx] >= 0
        if mask.sum() < 2:
            continue
        head_logits = per_enzyme_logits[head_idx][mask]
        head_labels = per_enzyme_labels[:, head_idx][mask]
        loss_per_enzyme = loss_per_enzyme + F.binary_cross_entropy_with_logits(head_logits, head_labels)
        n_active_heads += 1

    if n_active_heads > 0:
        loss_per_enzyme = loss_per_enzyme / n_active_heads

    # Enzyme classifier loss (positives only)
    pos_mask = batch["label_binary"] == 1
    loss_enzyme_cls = torch.tensor(0.0, device=device)
    if pos_mask.sum() > 0:
        loss_enzyme_cls = F.cross_entropy(
            enzyme_cls_logits[pos_mask],
            batch["label_enzyme"][pos_mask],
        )

    total = loss_binary + enzyme_head_weight * loss_per_enzyme + enzyme_cls_weight * loss_enzyme_cls
    return total, {
        "binary": loss_binary.item(),
        "per_enzyme": loss_per_enzyme.item(),
        "enzyme_cls": loss_enzyme_cls.item(),
        "total": total.item(),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_one_epoch(model, loader, optimizer, enzyme_head_weight, enzyme_cls_weight):
    """Train one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        optimizer.zero_grad()

        binary_logit, per_enzyme_logits, enzyme_cls_logits, _ = model(batch)
        loss, _ = compute_loss(
            binary_logit, per_enzyme_logits, enzyme_cls_logits, batch,
            enzyme_head_weight, enzyme_cls_weight,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_model(model, loader):
    """Evaluate: overall AUROC, per-enzyme AUROC (adapter + binary head), enzyme accuracy."""
    model.eval()
    all_binary_probs = []
    all_per_enzyme_probs = [[] for _ in range(N_PER_ENZYME)]
    all_enzyme_cls = []
    all_lbl_binary = []
    all_lbl_enzyme = []
    all_per_enzyme_labels = []

    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        binary_logit, per_enzyme_logits, enzyme_cls_logits, _ = model(batch)

        all_binary_probs.append(torch.sigmoid(binary_logit).cpu().numpy())
        for h in range(N_PER_ENZYME):
            all_per_enzyme_probs[h].append(torch.sigmoid(per_enzyme_logits[h]).cpu().numpy())
        all_enzyme_cls.append(enzyme_cls_logits.cpu().numpy())
        all_lbl_binary.append(batch["label_binary"].cpu().numpy())
        all_lbl_enzyme.append(batch["label_enzyme"].cpu().numpy())
        all_per_enzyme_labels.append(batch["per_enzyme_labels"].cpu().numpy())

    binary_probs = np.concatenate(all_binary_probs)
    per_enzyme_probs = [np.concatenate(p) for p in all_per_enzyme_probs]
    enzyme_cls_all = np.concatenate(all_enzyme_cls)
    lbl_binary = np.concatenate(all_lbl_binary)
    lbl_enzyme = np.concatenate(all_lbl_enzyme)
    per_enzyme_labels = np.concatenate(all_per_enzyme_labels)

    # Overall binary AUROC
    try:
        overall_auroc = roc_auc_score(lbl_binary, binary_probs)
    except ValueError:
        overall_auroc = 0.5

    # Per-enzyme AUROC using dedicated adapter heads
    per_enzyme_auroc_adapter = {}
    for head_idx, enz_name in enumerate(PER_ENZYME_HEADS):
        mask = per_enzyme_labels[:, head_idx] >= 0
        if mask.sum() < 2:
            continue
        labels = per_enzyme_labels[:, head_idx][mask]
        probs = per_enzyme_probs[head_idx][mask]
        if len(np.unique(labels)) < 2:
            continue
        try:
            per_enzyme_auroc_adapter[enz_name] = float(roc_auc_score(labels, probs))
        except ValueError:
            pass

    # Per-enzyme AUROC using the unified binary head (for comparison)
    per_enzyme_auroc_binary = {}
    for enz_name in PER_ENZYME_HEADS:
        enz_idx = ENZYME_TO_IDX[enz_name]
        # Enzyme X positives + enzyme X negatives
        enz_mask = lbl_enzyme == enz_idx
        if enz_mask.sum() < 2:
            continue
        labels_sub = lbl_binary[enz_mask]
        probs_sub = binary_probs[enz_mask]
        if len(np.unique(labels_sub)) < 2:
            continue
        try:
            per_enzyme_auroc_binary[enz_name] = float(roc_auc_score(labels_sub, probs_sub))
        except ValueError:
            pass

    # Enzyme classification accuracy (positives only)
    pos_mask = lbl_binary == 1
    enzyme_acc = 0.0
    if pos_mask.sum() > 0:
        enzyme_acc = accuracy_score(lbl_enzyme[pos_mask], enzyme_cls_all[pos_mask].argmax(axis=1))

    return {
        "overall_auroc": float(overall_auroc),
        "per_enzyme_auroc_adapter": per_enzyme_auroc_adapter,
        "per_enzyme_auroc_binary": per_enzyme_auroc_binary,
        "enzyme_accuracy": float(enzyme_acc),
        "binary_probs": binary_probs,
        "per_enzyme_probs": per_enzyme_probs,
        "lbl_binary": lbl_binary,
        "lbl_enzyme": lbl_enzyme,
    }


# ---------------------------------------------------------------------------
# Two-stage training orchestrator
# ---------------------------------------------------------------------------


def run_two_stage_cv(data):
    """Run 2-fold CV with two-stage training."""
    logger.info("\n" + "=" * 70)
    logger.info("Phase 3: Two-Stage Multi-Stream Fusion Model")
    logger.info("=" * 70)

    n = len(data["site_ids"])
    skf = StratifiedKFold(n_splits=max(N_FOLDS, 2), shuffle=True, random_state=SEED)
    folds = list(skf.split(np.arange(n), data["labels_binary"]))[:N_FOLDS]

    fold_results = []
    fold_histories = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        torch.manual_seed(SEED + fold_idx)
        np.random.seed(SEED + fold_idx)

        logger.info(f"\n{'='*60}")
        logger.info(f"Fold {fold_idx}/{N_FOLDS}")
        logger.info(f"  Train: {len(train_idx)}, Val: {len(val_idx)}")
        logger.info(f"{'='*60}")
        t0_fold = time.time()

        model = Phase3FusionModel().to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if fold_idx == 0:
            logger.info(f"  Model parameters: {n_params:,}")
            logger.info(f"  Fusion input dim: {D_FUSION_IN}")

        train_ds = Phase3Dataset(list(train_idx), data)
        val_ds = Phase3Dataset(list(val_idx), data)

        n_loader_workers = 0 if DEVICE.type == "mps" else 4
        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True,
            collate_fn=collate_fn, num_workers=n_loader_workers, pin_memory=False,
        )
        val_loader = DataLoader(
            val_ds, batch_size=BATCH_SIZE, shuffle=False,
            collate_fn=collate_fn, num_workers=n_loader_workers, pin_memory=False,
        )

        history = {"stage": [], "epoch": [], "train_loss": [], "val_overall_auroc": []}
        best_auroc = 0.0
        best_state = None

        # ===================================================================
        # Stage 1: Train EVERYTHING jointly (10 epochs)
        # ===================================================================
        logger.info(f"\n--- Stage 1: Joint training ({STAGE1_EPOCHS} epochs, lr={STAGE1_LR}) ---")

        optimizer_s1 = torch.optim.AdamW(
            model.parameters(), lr=STAGE1_LR, weight_decay=STAGE1_WEIGHT_DECAY,
        )
        scheduler_s1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_s1, T_max=STAGE1_EPOCHS)

        for epoch in range(STAGE1_EPOCHS):
            loss = train_one_epoch(
                model, train_loader, optimizer_s1,
                enzyme_head_weight=STAGE1_ENZYME_HEAD_WEIGHT,
                enzyme_cls_weight=STAGE1_ENZYME_CLS_WEIGHT,
            )
            scheduler_s1.step()

            metrics = evaluate_model(model, val_loader)
            history["stage"].append(1)
            history["epoch"].append(epoch)
            history["train_loss"].append(loss)
            history["val_overall_auroc"].append(metrics["overall_auroc"])

            if metrics["overall_auroc"] > best_auroc:
                best_auroc = metrics["overall_auroc"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            enz_str = ", ".join(
                f"{e}={metrics['per_enzyme_auroc_adapter'].get(e, 0):.3f}"
                for e in PER_ENZYME_HEADS
            )
            logger.info(
                f"  [S1] Epoch {epoch+1}/{STAGE1_EPOCHS}: loss={loss:.4f}, "
                f"overall={metrics['overall_auroc']:.4f}, enz_acc={metrics['enzyme_accuracy']:.3f}, "
                f"{enz_str}"
            )

        # ===================================================================
        # Stage 2: Freeze shared encoder, train adapters only (20 epochs)
        # ===================================================================
        logger.info(f"\n--- Stage 2: Adapter-only training ({STAGE2_EPOCHS} epochs, lr={STAGE2_LR}) ---")

        # Freeze shared encoder + conv2d encoder
        for p in model.conv2d_encoder.parameters():
            p.requires_grad = False
        for p in model.shared_encoder.parameters():
            p.requires_grad = False

        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"  Trainable parameters (adapters only): {n_trainable:,}")

        optimizer_s2 = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=STAGE2_LR, weight_decay=STAGE2_WEIGHT_DECAY,
        )
        scheduler_s2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_s2, T_max=STAGE2_EPOCHS)

        for epoch in range(STAGE2_EPOCHS):
            loss = train_one_epoch(
                model, train_loader, optimizer_s2,
                enzyme_head_weight=STAGE1_ENZYME_HEAD_WEIGHT,
                enzyme_cls_weight=STAGE1_ENZYME_CLS_WEIGHT,
            )
            scheduler_s2.step()

            metrics = evaluate_model(model, val_loader)
            history["stage"].append(2)
            history["epoch"].append(STAGE1_EPOCHS + epoch)
            history["train_loss"].append(loss)
            history["val_overall_auroc"].append(metrics["overall_auroc"])

            if metrics["overall_auroc"] > best_auroc:
                best_auroc = metrics["overall_auroc"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if (epoch + 1) % 5 == 0:
                enz_str = ", ".join(
                    f"{e}={metrics['per_enzyme_auroc_adapter'].get(e, 0):.3f}"
                    for e in PER_ENZYME_HEADS
                )
                logger.info(
                    f"  [S2] Epoch {epoch+1}/{STAGE2_EPOCHS}: loss={loss:.4f}, "
                    f"overall={metrics['overall_auroc']:.4f}, enz_acc={metrics['enzyme_accuracy']:.3f}, "
                    f"{enz_str}"
                )

        # ===================================================================
        # Final evaluation with best model
        # ===================================================================

        # Unfreeze for state loading
        for p in model.parameters():
            p.requires_grad = True

        if best_state is not None:
            model.load_state_dict(best_state)

        final_metrics = evaluate_model(model, val_loader)

        elapsed = time.time() - t0_fold
        logger.info(f"\n  Fold {fold_idx} complete in {elapsed:.0f}s")
        logger.info(f"  Best overall AUROC: {final_metrics['overall_auroc']:.4f}")
        logger.info(f"  Enzyme classification accuracy: {final_metrics['enzyme_accuracy']:.3f}")
        logger.info(f"  Per-enzyme AUROC (adapter head / binary head):")
        for enz in PER_ENZYME_HEADS:
            adapter_val = final_metrics["per_enzyme_auroc_adapter"].get(enz, 0)
            binary_val = final_metrics["per_enzyme_auroc_binary"].get(enz, 0)
            logger.info(f"    {enz}: adapter={adapter_val:.3f}, binary={binary_val:.3f}")

        fold_results.append(final_metrics)
        fold_histories.append(history)

        # Save checkpoint
        save_path = OUTPUT_DIR / f"phase3_fold{fold_idx}.pt"
        torch.save(model.state_dict(), save_path)
        logger.info(f"  Saved checkpoint: {save_path}")

        del model, optimizer_s1, optimizer_s2, scheduler_s1, scheduler_s2
        if DEVICE.type == "mps":
            torch.mps.empty_cache()
        gc.collect()

    return fold_results, fold_histories


# ---------------------------------------------------------------------------
# Summary and comparison
# ---------------------------------------------------------------------------


def print_summary(fold_results, fold_histories):
    """Print aggregate results and comparison to baselines."""
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 3 RESULTS SUMMARY")
    logger.info("=" * 70)

    # Aggregate across folds
    mean_overall = np.mean([r["overall_auroc"] for r in fold_results])
    std_overall = np.std([r["overall_auroc"] for r in fold_results])
    mean_enz_acc = np.mean([r["enzyme_accuracy"] for r in fold_results])

    logger.info(f"\nOverall binary AUROC: {mean_overall:.4f} +/- {std_overall:.4f}")
    logger.info(f"Enzyme classification accuracy: {mean_enz_acc:.3f}")

    logger.info(f"\nPer-enzyme AUROC (adapter heads):")
    logger.info(f"  {'Enzyme':<12} {'Phase3':>8} {'XGBoost':>8} {'Phase2b':>8} {'Unified_V1':>10} {'Delta':>8}")
    logger.info(f"  {'-'*58}")

    per_enzyme_summary = {}
    for enz in PER_ENZYME_HEADS:
        vals = [r["per_enzyme_auroc_adapter"].get(enz, 0) for r in fold_results]
        mean_val = np.mean(vals)
        std_val = np.std(vals)
        per_enzyme_summary[enz] = {"mean": mean_val, "std": std_val}

        xgb = XGBOOST_BASELINES.get(enz, 0)
        p2b = PHASE2B_BASELINES.get(enz, 0)
        uv1 = UNIFIED_V1_BASELINES.get(enz, 0)
        delta = mean_val - xgb if xgb > 0 else 0

        logger.info(
            f"  {enz:<12} {mean_val:>7.3f} {xgb:>8.3f} "
            f"{p2b:>8.3f} {uv1:>10.3f} {delta:>+8.3f}"
        )

    logger.info(f"\nPer-enzyme AUROC (binary head, for comparison):")
    for enz in PER_ENZYME_HEADS:
        vals = [r["per_enzyme_auroc_binary"].get(enz, 0) for r in fold_results]
        mean_val = np.mean(vals)
        logger.info(f"  {enz:<12} {mean_val:.3f}")

    # Save results JSON
    results_json = {
        "model": "Phase3_MultiStreamFusion",
        "n_folds": N_FOLDS,
        "stages": {"stage1_epochs": STAGE1_EPOCHS, "stage2_epochs": STAGE2_EPOCHS},
        "overall_auroc": {"mean": float(mean_overall), "std": float(std_overall)},
        "enzyme_accuracy": float(mean_enz_acc),
        "per_enzyme_auroc_adapter": {
            enz: {"mean": float(per_enzyme_summary[enz]["mean"]),
                  "std": float(per_enzyme_summary[enz]["std"])}
            for enz in PER_ENZYME_HEADS
        },
        "per_enzyme_auroc_binary": {
            enz: {"mean": float(np.mean([r["per_enzyme_auroc_binary"].get(enz, 0) for r in fold_results])),
                  "std": float(np.std([r["per_enzyme_auroc_binary"].get(enz, 0) for r in fold_results]))}
            for enz in PER_ENZYME_HEADS
        },
        "baselines": {
            "xgboost": XGBOOST_BASELINES,
            "phase2b_conv2d": PHASE2B_BASELINES,
            "unified_v1": UNIFIED_V1_BASELINES,
        },
    }

    results_path = OUTPUT_DIR / "phase3_results.json"
    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2)
    logger.info(f"\nResults saved to {results_path}")

    return results_json


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_training_curves(fold_histories):
    """Plot training loss and validation AUROC across stages."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for fold_idx, hist in enumerate(fold_histories):
        epochs = list(range(len(hist["train_loss"])))
        stages = hist["stage"]

        # Color by stage
        s1_mask = [s == 1 for s in stages]
        s2_mask = [s == 2 for s in stages]

        s1_epochs = [e for e, m in zip(epochs, s1_mask) if m]
        s1_loss = [l for l, m in zip(hist["train_loss"], s1_mask) if m]
        s1_auroc = [a for a, m in zip(hist["val_overall_auroc"], s1_mask) if m]

        s2_epochs = [e for e, m in zip(epochs, s2_mask) if m]
        s2_loss = [l for l, m in zip(hist["train_loss"], s2_mask) if m]
        s2_auroc = [a for a, m in zip(hist["val_overall_auroc"], s2_mask) if m]

        # Loss
        axes[0].plot(s1_epochs, s1_loss, 'o-', color=f"C{fold_idx}", markersize=3,
                     label=f"Fold {fold_idx} S1")
        axes[0].plot(s2_epochs, s2_loss, 's--', color=f"C{fold_idx}", markersize=3,
                     label=f"Fold {fold_idx} S2")

        # AUROC
        axes[1].plot(s1_epochs, s1_auroc, 'o-', color=f"C{fold_idx}", markersize=3,
                     label=f"Fold {fold_idx} S1")
        axes[1].plot(s2_epochs, s2_auroc, 's--', color=f"C{fold_idx}", markersize=3,
                     label=f"Fold {fold_idx} S2")

    # Stage boundary
    axes[0].axvline(x=STAGE1_EPOCHS - 0.5, color="gray", linestyle=":", alpha=0.5)
    axes[1].axvline(x=STAGE1_EPOCHS - 0.5, color="gray", linestyle=":", alpha=0.5)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training Loss")
    axes[0].set_title("Training Loss (S1=joint, S2=adapters-only)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation AUROC")
    axes[1].set_title("Validation Overall AUROC")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plot_path = OUTPUT_DIR / "phase3_training_curves.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logger.info(f"Training curves saved to {plot_path}")


def plot_comparison_bar(results_json):
    """Bar chart comparing Phase 3 adapter AUROC to baselines per enzyme."""
    enzymes = PER_ENZYME_HEADS
    phase3_vals = [results_json["per_enzyme_auroc_adapter"][e]["mean"] for e in enzymes]
    phase3_stds = [results_json["per_enzyme_auroc_adapter"][e]["std"] for e in enzymes]
    xgb_vals = [XGBOOST_BASELINES.get(e, 0) for e in enzymes]

    x = np.arange(len(enzymes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, phase3_vals, width, yerr=phase3_stds,
                   label="Phase 3 (adapter)", color="#4C72B0", capsize=3)
    bars2 = ax.bar(x + width / 2, xgb_vals, width,
                   label="XGBoost baseline", color="#DD8452", capsize=3)

    ax.set_ylabel("AUROC")
    ax.set_title("Phase 3 Multi-Stream Fusion vs XGBoost Baselines")
    ax.set_xticks(x)
    ax.set_xticklabels(enzymes)
    ax.legend()
    ax.set_ylim(0.5, 1.0)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels
    for bar_group in [bars1, bars2]:
        for bar in bar_group:
            height = bar.get_height()
            ax.annotate(f"{height:.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plot_path = OUTPUT_DIR / "phase3_comparison.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logger.info(f"Comparison plot saved to {plot_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    t_start = time.time()
    logger.info("Phase 3: Definitive Multi-Stream Fusion Architecture")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Config: {N_FOLDS} folds, Stage1={STAGE1_EPOCHS}ep, Stage2={STAGE2_EPOCHS}ep")
    logger.info(f"Fusion: RNA-FM({D_RNAFM}) + Conv2D({D_CONV2D_OUT}) + EditDelta({D_EDIT_DELTA}) + Hand({D_HAND}) = {D_FUSION_IN}")

    # Load all data
    data = load_data()

    # Run two-stage CV
    fold_results, fold_histories = run_two_stage_cv(data)

    # Summary and comparison
    results_json = print_summary(fold_results, fold_histories)

    # Plots
    plot_training_curves(fold_histories)
    plot_comparison_bar(results_json)

    total_time = time.time() - t_start
    logger.info(f"\nTotal runtime: {total_time:.0f}s ({total_time/60:.1f}min)")
    logger.info("Phase 3 complete.")


if __name__ == "__main__":
    main()
