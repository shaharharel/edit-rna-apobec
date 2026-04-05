#!/usr/bin/env python3
"""Phase 3 TRUE Neural Validation: Train the actual Phase3FusionModel and score externally.

Previous validation used XGBoost on 1280/1320-dim features as a proxy.
This script trains the REAL Phase 3 multi-stream fusion neural network
(RNA-FM 640 + edit_delta 640 + hand 40 = 1320-dim, NO Conv2D for TCGA scoring)
and applies it to:
  1. 5-fold CV on v3 training data (per-enzyme classification)
  2. TCGA somatic mutation enrichment (8 cancers)
  3. Cross-species conservation (human vs chimp)
  4. gnomAD gene-level constraint correlation

Architecture (simplified for external scoring — no Conv2D since TCGA lacks BPP matrices):
  Input: RNA-FM(640) + edit_delta(640) + hand(40) = 1320-dim
  → Linear(1320, 256) → GELU → Dropout(0.3) → LayerNorm
  → Linear(256, 128) → GELU → Dropout(0.2) → 128-dim shared embedding
  → Binary head: Linear(128, 1)
  → Per-enzyme adapters: Linear(128, 32) → GELU → Linear(32, 1) × 5 enzymes

Two-stage training: Stage 1 (10 epochs joint) → Stage 2 (20 epochs adapters only)

Usage:
    conda run -n quris python experiments/multi_enzyme/exp_phase3_neural_true_validation.py
"""

import gc
import gzip
import json
import logging
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.apobec_feature_extraction import (
    build_hand_features,
    extract_motif_from_seq,
    LOOP_FEATURE_COLS,
)

# ============================================================================
# Output
# ============================================================================
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "multi_enzyme" / "outputs" / "phase3_neural_true_validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

log_file = OUTPUT_DIR / "run.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, mode="w"),
    ],
)
logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================
DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else torch.device("cpu")
)
logger.info(f"Device: {DEVICE}")

SEED = 42
N_FOLDS = 5
BATCH_SIZE = 64
ENZYMES = ["A3A", "A3B", "A3G", "A3A_A3G", "Neither"]
ENZYME_CLASSES = ["A3A", "A3B", "A3G", "A3A_A3G", "Neither", "Unknown"]
ENZYME_TO_IDX = {e: i for i, e in enumerate(ENZYME_CLASSES)}
N_ENZYMES = len(ENZYME_CLASSES)
N_PER_ENZYME = len(ENZYMES)
CANCERS = ["lihc", "esca", "hnsc", "lusc", "brca", "cesc", "blca", "stad"]

# Feature dims
D_RNAFM = 640
D_EDIT_DELTA = 640
D_HAND = 40
D_INPUT = D_RNAFM + D_EDIT_DELTA + D_HAND  # 1320
D_SHARED = 128

# Training
STAGE1_EPOCHS = 10
STAGE1_LR = 1e-3
STAGE2_EPOCHS = 20
STAGE2_LR = 5e-4
WEIGHT_DECAY = 1e-4
ENZYME_HEAD_WEIGHT = 0.3
ENZYME_CLS_WEIGHT = 0.1

# Paths
DATA_DIR = PROJECT_ROOT / "data"
EMB_DIR = DATA_DIR / "processed" / "multi_enzyme" / "embeddings"
MULTI_SPLITS = DATA_DIR / "processed" / "multi_enzyme" / "splits_multi_enzyme_v3_with_negatives.csv"
MULTI_SEQS = DATA_DIR / "processed" / "multi_enzyme" / "multi_enzyme_sequences_v3_with_negatives.json"
LOOP_CSV = DATA_DIR / "processed" / "multi_enzyme" / "loop_position_per_site_v3.csv"
STRUCT_CACHE = DATA_DIR / "processed" / "multi_enzyme" / "structure_cache_multi_enzyme_v3.npz"
RAW_SCORES_DIR = PROJECT_ROOT / "experiments" / "multi_enzyme" / "outputs" / "tcga_gnomad" / "raw_scores"


# ============================================================================
# Model Definition (no Conv2D — portable for external scoring)
# ============================================================================

class Phase3Model(nn.Module):
    """Phase 3 fusion model without Conv2D (for external scoring portability).

    Input: RNA-FM(640) + edit_delta(640) + hand(40) = 1320-dim
    → shared encoder → 128-dim
    → binary head + per-enzyme adapter heads + enzyme classifier
    """

    def __init__(self):
        super().__init__()
        self.shared_encoder = nn.Sequential(
            nn.Linear(D_INPUT, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.LayerNorm(256),
            nn.Linear(256, D_SHARED),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.binary_head = nn.Linear(D_SHARED, 1)
        self.enzyme_adapters = nn.ModuleDict({
            enz: nn.Sequential(
                nn.Linear(D_SHARED, 32),
                nn.GELU(),
                nn.Linear(32, 1),
            )
            for enz in ENZYMES
        })
        self.enzyme_classifier = nn.Sequential(
            nn.Linear(D_SHARED, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, N_ENZYMES),
        )

    def forward(self, x):
        """x: [B, 1320] → binary_logit, per_enzyme_logits, enzyme_cls_logits, shared_repr"""
        shared = self.shared_encoder(x)
        binary_logit = self.binary_head(shared).squeeze(-1)
        per_enzyme_logits = []
        for enz in ENZYMES:
            logit = self.enzyme_adapters[enz](shared).squeeze(-1)
            per_enzyme_logits.append(logit)
        enzyme_cls_logits = self.enzyme_classifier(shared)
        return binary_logit, per_enzyme_logits, enzyme_cls_logits, shared


# ============================================================================
# Dataset
# ============================================================================

class SimpleDataset(Dataset):
    def __init__(self, X, labels_binary, labels_enzyme, per_enzyme_labels):
        self.X = torch.from_numpy(X).float()
        self.labels_binary = torch.from_numpy(labels_binary).float()
        self.labels_enzyme = torch.from_numpy(labels_enzyme).long()
        self.per_enzyme_labels = torch.from_numpy(per_enzyme_labels).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            "x": self.X[idx],
            "label_binary": self.labels_binary[idx],
            "label_enzyme": self.labels_enzyme[idx],
            "per_enzyme_labels": self.per_enzyme_labels[idx],
        }


# ============================================================================
# Loss
# ============================================================================

def compute_loss(binary_logit, per_enzyme_logits, enzyme_cls_logits, batch):
    device = binary_logit.device
    loss_binary = F.binary_cross_entropy_with_logits(binary_logit, batch["label_binary"])

    per_enzyme_labels = batch["per_enzyme_labels"]
    loss_per_enzyme = torch.tensor(0.0, device=device)
    n_active = 0
    for head_idx in range(N_PER_ENZYME):
        mask = per_enzyme_labels[:, head_idx] >= 0
        if mask.sum() < 2:
            continue
        head_logits = per_enzyme_logits[head_idx][mask]
        head_labels = per_enzyme_labels[:, head_idx][mask]
        loss_per_enzyme = loss_per_enzyme + F.binary_cross_entropy_with_logits(head_logits, head_labels)
        n_active += 1
    if n_active > 0:
        loss_per_enzyme = loss_per_enzyme / n_active

    pos_mask = batch["label_binary"] == 1
    loss_cls = torch.tensor(0.0, device=device)
    if pos_mask.sum() > 0:
        loss_cls = F.cross_entropy(enzyme_cls_logits[pos_mask], batch["label_enzyme"][pos_mask])

    total = loss_binary + ENZYME_HEAD_WEIGHT * loss_per_enzyme + ENZYME_CLS_WEIGHT * loss_cls
    return total


# ============================================================================
# Training
# ============================================================================

def train_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0
    n = 0
    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        optimizer.zero_grad()
        binary_logit, per_enzyme_logits, enzyme_cls_logits, _ = model(batch["x"])
        loss = compute_loss(binary_logit, per_enzyme_logits, enzyme_cls_logits, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_binary_probs = []
    all_per_enzyme_probs = [[] for _ in range(N_PER_ENZYME)]
    all_lbl_binary = []
    all_lbl_enzyme = []
    all_per_enzyme_labels = []

    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        binary_logit, per_enzyme_logits, enzyme_cls_logits, _ = model(batch["x"])
        all_binary_probs.append(torch.sigmoid(binary_logit).cpu().numpy())
        for h in range(N_PER_ENZYME):
            all_per_enzyme_probs[h].append(torch.sigmoid(per_enzyme_logits[h]).cpu().numpy())
        all_lbl_binary.append(batch["label_binary"].cpu().numpy())
        all_lbl_enzyme.append(batch["label_enzyme"].cpu().numpy())
        all_per_enzyme_labels.append(batch["per_enzyme_labels"].cpu().numpy())

    binary_probs = np.concatenate(all_binary_probs)
    per_enzyme_probs = [np.concatenate(p) for p in all_per_enzyme_probs]
    lbl_binary = np.concatenate(all_lbl_binary)
    lbl_enzyme = np.concatenate(all_lbl_enzyme)
    per_enzyme_labels = np.concatenate(all_per_enzyme_labels)

    try:
        overall_auroc = roc_auc_score(lbl_binary, binary_probs)
    except ValueError:
        overall_auroc = 0.5

    per_enzyme_auroc = {}
    for head_idx, enz in enumerate(ENZYMES):
        mask = per_enzyme_labels[:, head_idx] >= 0
        if mask.sum() < 2:
            continue
        labels = per_enzyme_labels[:, head_idx][mask]
        probs = per_enzyme_probs[head_idx][mask]
        if len(np.unique(labels)) < 2:
            continue
        try:
            per_enzyme_auroc[enz] = float(roc_auc_score(labels, probs))
        except ValueError:
            pass

    return {
        "overall_auroc": float(overall_auroc),
        "per_enzyme_auroc": per_enzyme_auroc,
        "binary_probs": binary_probs,
        "per_enzyme_probs": per_enzyme_probs,
        "lbl_binary": lbl_binary,
        "lbl_enzyme": lbl_enzyme,
        "per_enzyme_labels": per_enzyme_labels,
    }


def train_two_stage(model, train_ds, val_ds=None):
    """Two-stage training. Returns best model state dict and val metrics."""
    n_workers = 0
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=n_workers)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=n_workers) if val_ds else None

    best_auroc = 0.0
    best_state = None

    # Stage 1: train everything
    optimizer = torch.optim.AdamW(model.parameters(), lr=STAGE1_LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STAGE1_EPOCHS)

    for epoch in range(STAGE1_EPOCHS):
        loss = train_epoch(model, train_loader, optimizer)
        scheduler.step()
        if val_loader:
            metrics = evaluate(model, val_loader)
            if metrics["overall_auroc"] > best_auroc:
                best_auroc = metrics["overall_auroc"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if (epoch + 1) % 5 == 0:
                enz_str = ", ".join(f"{e}={metrics['per_enzyme_auroc'].get(e,0):.3f}" for e in ENZYMES)
                logger.info(f"  [S1] Ep {epoch+1}: loss={loss:.4f}, auroc={metrics['overall_auroc']:.4f}, {enz_str}")
        else:
            if (epoch + 1) % 5 == 0:
                logger.info(f"  [S1] Ep {epoch+1}: loss={loss:.4f}")

    # Stage 2: freeze shared, train adapters only
    for p in model.shared_encoder.parameters():
        p.requires_grad = False

    adapter_params = list(model.binary_head.parameters())
    for enz in ENZYMES:
        adapter_params.extend(model.enzyme_adapters[enz].parameters())
    adapter_params.extend(model.enzyme_classifier.parameters())

    optimizer2 = torch.optim.AdamW(adapter_params, lr=STAGE2_LR, weight_decay=WEIGHT_DECAY)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=STAGE2_EPOCHS)

    for epoch in range(STAGE2_EPOCHS):
        loss = train_epoch(model, train_loader, optimizer2)
        scheduler2.step()
        if val_loader:
            metrics = evaluate(model, val_loader)
            if metrics["overall_auroc"] > best_auroc:
                best_auroc = metrics["overall_auroc"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if (epoch + 1) % 10 == 0:
                enz_str = ", ".join(f"{e}={metrics['per_enzyme_auroc'].get(e,0):.3f}" for e in ENZYMES)
                logger.info(f"  [S2] Ep {epoch+1}: loss={loss:.4f}, auroc={metrics['overall_auroc']:.4f}, {enz_str}")
        else:
            if (epoch + 1) % 10 == 0:
                logger.info(f"  [S2] Ep {epoch+1}: loss={loss:.4f}")

    # Unfreeze all
    for p in model.parameters():
        p.requires_grad = True

    if best_state is not None:
        model.load_state_dict(best_state)

    return best_state, best_auroc


# ============================================================================
# Data Loading
# ============================================================================

def load_v3_data():
    """Load v3 splits + all features, build 1320-dim input matrix."""
    logger.info("Loading v3 data...")

    df = pd.read_csv(MULTI_SPLITS)
    with open(MULTI_SEQS) as f:
        seqs = json.load(f)

    orig_emb = torch.load(EMB_DIR / "rnafm_pooled_v3.pt", weights_only=False)
    edited_emb = torch.load(EMB_DIR / "rnafm_pooled_edited_v3.pt", weights_only=False)

    loop_df = pd.read_csv(LOOP_CSV).drop_duplicates(subset=["site_id"]).set_index("site_id")

    sc = np.load(STRUCT_CACHE, allow_pickle=True)
    struct_map = {sid: sc["delta_features"][i] for i, sid in enumerate(sc["site_ids"])}

    # Filter to sites with sequences
    site_ids = df["site_id"].tolist()
    valid = [sid in seqs for sid in site_ids]
    df = df[valid].reset_index(drop=True)
    site_ids = df["site_id"].tolist()
    n = len(site_ids)

    logger.info(f"  {n} sites with sequences")

    # Build hand features (40-dim)
    hand = build_hand_features(site_ids, seqs, struct_map, loop_df)

    # RNA-FM features
    rnafm_orig = np.zeros((n, D_RNAFM), dtype=np.float32)
    rnafm_delta = np.zeros((n, D_EDIT_DELTA), dtype=np.float32)
    n_found = 0
    for i, sid in enumerate(site_ids):
        if sid in orig_emb and sid in edited_emb:
            o = orig_emb[sid].numpy() if isinstance(orig_emb[sid], torch.Tensor) else orig_emb[sid]
            e = edited_emb[sid].numpy() if isinstance(edited_emb[sid], torch.Tensor) else edited_emb[sid]
            rnafm_orig[i] = o
            rnafm_delta[i] = e - o
            n_found += 1

    logger.info(f"  RNA-FM coverage: {n_found}/{n}")

    # Concat: [rnafm_orig(640) | rnafm_delta(640) | hand(40)] = 1320-dim
    X = np.concatenate([rnafm_orig, rnafm_delta, hand], axis=1).astype(np.float32)
    logger.info(f"  Feature matrix: {X.shape}")

    # Labels
    labels_binary = df["is_edited"].values.astype(np.float32)
    labels_enzyme = np.array([ENZYME_TO_IDX.get(e, 5) for e in df["enzyme"].values], dtype=np.int64)

    # Per-enzyme labels (-1 = ignore)
    per_enzyme_labels = np.full((n, N_PER_ENZYME), -1, dtype=np.float32)
    for head_idx, enz in enumerate(ENZYMES):
        enz_idx = ENZYME_TO_IDX[enz]
        pos_mask = (labels_binary == 1) & (labels_enzyme == enz_idx)
        neg_mask = (labels_binary == 0) & (labels_enzyme == enz_idx)
        per_enzyme_labels[pos_mask, head_idx] = 1.0
        per_enzyme_labels[neg_mask, head_idx] = 0.0

    for head_idx, enz in enumerate(ENZYMES):
        n_pos = int((per_enzyme_labels[:, head_idx] == 1).sum())
        n_neg = int((per_enzyme_labels[:, head_idx] == 0).sum())
        logger.info(f"  {enz}: {n_pos} pos, {n_neg} neg")

    del orig_emb, edited_emb
    gc.collect()

    return {
        "df": df, "X": X, "site_ids": site_ids,
        "labels_binary": labels_binary, "labels_enzyme": labels_enzyme,
        "per_enzyme_labels": per_enzyme_labels,
        "seqs": seqs, "struct_map": struct_map, "loop_df": loop_df,
    }


# ============================================================================
# Step 1: 5-fold CV
# ============================================================================

def run_cv(data):
    """5-fold CV with Phase 3 neural model."""
    logger.info("\n" + "=" * 70)
    logger.info("STEP 1: 5-fold CV on v3 training data (Phase 3 Neural)")
    logger.info("=" * 70)

    X = data["X"]
    labels_binary = data["labels_binary"]
    labels_enzyme = data["labels_enzyme"]
    per_enzyme_labels = data["per_enzyme_labels"]

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, labels_binary)):
        torch.manual_seed(SEED + fold)
        np.random.seed(SEED + fold)
        t0 = time.time()

        logger.info(f"\n--- Fold {fold+1}/{N_FOLDS} (train={len(train_idx)}, val={len(val_idx)}) ---")

        train_ds = SimpleDataset(X[train_idx], labels_binary[train_idx],
                                 labels_enzyme[train_idx], per_enzyme_labels[train_idx])
        val_ds = SimpleDataset(X[val_idx], labels_binary[val_idx],
                               labels_enzyme[val_idx], per_enzyme_labels[val_idx])

        model = Phase3Model().to(DEVICE)
        if fold == 0:
            n_params = sum(p.numel() for p in model.parameters())
            logger.info(f"  Model params: {n_params:,}")

        _, best_auroc = train_two_stage(model, train_ds, val_ds)

        # Final evaluation
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        metrics = evaluate(model, val_loader)

        elapsed = time.time() - t0
        logger.info(f"  Fold {fold+1} done in {elapsed:.0f}s, overall AUROC={metrics['overall_auroc']:.4f}")
        for enz in ENZYMES:
            v = metrics["per_enzyme_auroc"].get(enz, 0)
            logger.info(f"    {enz}: {v:.3f}")

        fold_results.append(metrics)

        # Save fold checkpoint
        torch.save(model.state_dict(), OUTPUT_DIR / f"neural_fold{fold}.pt")

        del model, train_ds, val_ds
        if DEVICE.type == "mps":
            torch.mps.empty_cache()
        gc.collect()

    # Aggregate
    cv_summary = {
        "overall_auroc": {
            "mean": float(np.mean([r["overall_auroc"] for r in fold_results])),
            "std": float(np.std([r["overall_auroc"] for r in fold_results])),
        },
        "per_enzyme_auroc": {},
    }
    for enz in ENZYMES:
        vals = [r["per_enzyme_auroc"].get(enz, 0) for r in fold_results]
        cv_summary["per_enzyme_auroc"][enz] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "folds": [float(v) for v in vals],
        }

    logger.info(f"\n--- CV Summary ---")
    logger.info(f"Overall AUROC: {cv_summary['overall_auroc']['mean']:.4f} +/- {cv_summary['overall_auroc']['std']:.4f}")
    for enz in ENZYMES:
        m = cv_summary["per_enzyme_auroc"][enz]["mean"]
        s = cv_summary["per_enzyme_auroc"][enz]["std"]
        logger.info(f"  {enz}: {m:.3f} +/- {s:.3f}")

    with open(OUTPUT_DIR / "cv_results.json", "w") as f:
        json.dump(cv_summary, f, indent=2)

    return cv_summary


# ============================================================================
# Train full model on all data
# ============================================================================

def train_full_model(data):
    """Train on ALL v3 data for external scoring."""
    logger.info("\n" + "=" * 70)
    logger.info("Training full model on all v3 data")
    logger.info("=" * 70)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    X = data["X"]
    full_ds = SimpleDataset(X, data["labels_binary"], data["labels_enzyme"], data["per_enzyme_labels"])

    model = Phase3Model().to(DEVICE)
    train_two_stage(model, full_ds, val_ds=None)

    save_path = OUTPUT_DIR / "phase3_neural_full.pt"
    torch.save(model.state_dict(), save_path)
    logger.info(f"Saved full model to {save_path}")

    # Also get per-site scores for gnomAD analysis
    model.eval()
    loader = DataLoader(full_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    all_binary_probs = []
    all_per_enzyme_probs = [[] for _ in range(N_PER_ENZYME)]

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            binary_logit, per_enzyme_logits, _, _ = model(batch["x"])
            all_binary_probs.append(torch.sigmoid(binary_logit).cpu().numpy())
            for h in range(N_PER_ENZYME):
                all_per_enzyme_probs[h].append(torch.sigmoid(per_enzyme_logits[h]).cpu().numpy())

    binary_probs = np.concatenate(all_binary_probs)
    per_enzyme_probs = [np.concatenate(p) for p in all_per_enzyme_probs]

    return model, binary_probs, per_enzyme_probs


# ============================================================================
# Step 2: TCGA Scoring
# ============================================================================

@torch.no_grad()
def score_tcga(model, data):
    """Score TCGA mutations/controls with neural model."""
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: TCGA Somatic Mutation Enrichment (Neural Model)")
    logger.info("=" * 70)

    model.eval()
    tcga_results = {}

    for cancer in CANCERS:
        t0 = time.time()
        emb_path = EMB_DIR / f"rnafm_tcga_{cancer}.pt"
        if not emb_path.exists():
            logger.warning(f"  Missing {emb_path}")
            continue

        logger.info(f"\n--- {cancer.upper()} ---")
        d = torch.load(emb_path, weights_only=False)
        n_mut = d["n_mut"]
        n_ctrl = d["n_ctrl"]
        pooled_orig = d["pooled_orig"].numpy()
        pooled_edited = d["pooled_edited"].numpy()
        logger.info(f"  {n_mut} mutations, {n_ctrl} controls")

        # Build 1320-dim features: [rnafm_orig(640) | delta(640) | hand(40)]
        # For hand features, zero them out since we don't have sequence/structure for TCGA
        delta = pooled_edited - pooled_orig
        hand_zeros = np.zeros((len(pooled_orig), D_HAND), dtype=np.float32)
        X_tcga = np.concatenate([pooled_orig, delta, hand_zeros], axis=1).astype(np.float32)

        # Score in batches
        all_binary_probs = []
        all_per_enzyme_probs = [[] for _ in range(N_PER_ENZYME)]

        for start in range(0, len(X_tcga), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(X_tcga))
            x_batch = torch.from_numpy(X_tcga[start:end]).to(DEVICE)
            binary_logit, per_enzyme_logits, enzyme_cls_logits, _ = model(x_batch)
            all_binary_probs.append(torch.sigmoid(binary_logit).cpu().numpy())
            for h in range(N_PER_ENZYME):
                all_per_enzyme_probs[h].append(torch.sigmoid(per_enzyme_logits[h]).cpu().numpy())

        binary_probs = np.concatenate(all_binary_probs)
        per_enzyme_probs = [np.concatenate(p) for p in all_per_enzyme_probs]

        # Load TC context from existing raw scores
        scores_path = RAW_SCORES_DIR / f"{cancer}_scores.csv"
        if scores_path.exists():
            raw_df = pd.read_csv(scores_path)
            types = raw_df["type"].values
            tc_context = raw_df["tc_context"].values
        else:
            types = np.array(["mutation"] * n_mut + ["control"] * n_ctrl)
            tc_context = None

        mut_mask = types == "mutation"
        ctrl_mask = types == "control"

        # Enrichment stats for binary head
        cancer_result = {
            "cancer": cancer,
            "n_mutations": int(n_mut),
            "n_controls": int(n_ctrl),
            "binary_head": _enrichment_stats(binary_probs, mut_mask, ctrl_mask, tc_context),
        }

        # Per-enzyme adapter enrichment
        for h, enz in enumerate(ENZYMES):
            cancer_result[f"adapter_{enz}"] = _enrichment_stats(
                per_enzyme_probs[h], mut_mask, ctrl_mask, tc_context)

        logger.info(f"  Binary: mut={cancer_result['binary_head']['mean_mut']:.4f}, "
                    f"ctrl={cancer_result['binary_head']['mean_ctrl']:.4f}, "
                    f"delta={cancer_result['binary_head']['delta']:.4f}, "
                    f"p={cancer_result['binary_head']['mw_p']:.2e}")
        for enz in ENZYMES:
            r = cancer_result[f"adapter_{enz}"]
            logger.info(f"  {enz}: delta={r['delta']:.4f}, p={r['mw_p']:.2e}")

        # Save per-sample scores
        out_df = pd.DataFrame({"type": types, "neural_binary": binary_probs})
        if tc_context is not None:
            out_df["tc_context"] = tc_context
        for h, enz in enumerate(ENZYMES):
            out_df[f"neural_{enz}"] = per_enzyme_probs[h]
        out_df.to_csv(OUTPUT_DIR / f"tcga_{cancer}_neural_scores.csv", index=False)

        elapsed = time.time() - t0
        logger.info(f"  Elapsed: {elapsed:.0f}s")
        tcga_results[cancer] = cancer_result

        del d, pooled_orig, pooled_edited, X_tcga
        gc.collect()

    with open(OUTPUT_DIR / "tcga_results.json", "w") as f:
        json.dump(tcga_results, f, indent=2)

    return tcga_results


def _enrichment_stats(scores, mut_mask, ctrl_mask, tc_context=None):
    """Compute enrichment stats for mutation vs control."""
    mut_scores = scores[mut_mask]
    ctrl_scores = scores[ctrl_mask]

    result = {
        "mean_mut": float(np.mean(mut_scores)),
        "mean_ctrl": float(np.mean(ctrl_scores)),
        "delta": float(np.mean(mut_scores) - np.mean(ctrl_scores)),
    }

    if len(mut_scores) > 0 and len(ctrl_scores) > 0:
        U, p = stats.mannwhitneyu(mut_scores, ctrl_scores, alternative="greater")
        result["mw_p"] = float(p)
    else:
        result["mw_p"] = 1.0

    # TC-stratified
    if tc_context is not None:
        for ctx_name, ctx_val in [("TC", 1), ("nonTC", 0)]:
            ctx = tc_context == ctx_val
            ctx_mut = mut_mask & ctx
            ctx_ctrl = ctrl_mask & ctx
            if ctx_mut.sum() > 10 and ctx_ctrl.sum() > 10:
                result[f"{ctx_name}_delta"] = float(np.mean(scores[ctx_mut]) - np.mean(scores[ctx_ctrl]))
                _, p2 = stats.mannwhitneyu(scores[ctx_mut], scores[ctx_ctrl], alternative="greater")
                result[f"{ctx_name}_p"] = float(p2)

    # Percentile threshold enrichment
    thresholds = {}
    for pct in [50, 75, 90, 95]:
        thr = np.percentile(scores, pct)
        mut_above = int((mut_scores >= thr).sum())
        ctrl_above = int((ctrl_scores >= thr).sum())
        mut_below = len(mut_scores) - mut_above
        ctrl_below = len(ctrl_scores) - ctrl_above

        if ctrl_above > 0 and ctrl_below > 0:
            table = [[mut_above, ctrl_above], [mut_below, ctrl_below]]
            OR, fisher_p = stats.fisher_exact(table, alternative="greater")
        else:
            OR, fisher_p = float("nan"), 1.0

        thresholds[f"p{pct}"] = {
            "threshold": round(float(thr), 4),
            "OR": round(float(OR), 4) if not np.isnan(OR) else None,
            "p": float(fisher_p),
            "mut_above": mut_above,
            "ctrl_above": ctrl_above,
        }
    result["thresholds"] = thresholds
    return result


# ============================================================================
# Step 3: Cross-species
# ============================================================================

@torch.no_grad()
def score_cross_species(model):
    """Score human vs chimp editing sites."""
    logger.info("\n" + "=" * 70)
    logger.info("STEP 3: Cross-Species Conservation (Human vs Chimp)")
    logger.info("=" * 70)

    cs_path = EMB_DIR / "rnafm_cross_species.pt"
    if not cs_path.exists():
        logger.warning(f"Missing {cs_path}")
        return {}

    d = torch.load(cs_path, weights_only=False)
    site_ids = d["site_ids"]
    n = d["n_sites"]

    human_pooled = d["human_pooled"].numpy()
    human_edited = d["human_edited"].numpy()
    chimp_pooled = d["chimp_pooled"].numpy()
    chimp_edited = d["chimp_edited"].numpy()

    logger.info(f"  {n} conserved editing sites")

    model.eval()

    def score_batch(orig, edited):
        delta = edited - orig
        hand_zeros = np.zeros((len(orig), D_HAND), dtype=np.float32)
        X = np.concatenate([orig, delta, hand_zeros], axis=1).astype(np.float32)
        all_probs = []
        all_per_enzyme = [[] for _ in range(N_PER_ENZYME)]
        for start in range(0, len(X), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(X))
            x_batch = torch.from_numpy(X[start:end]).to(DEVICE)
            bl, pel, _, _ = model(x_batch)
            all_probs.append(torch.sigmoid(bl).cpu().numpy())
            for h in range(N_PER_ENZYME):
                all_per_enzyme[h].append(torch.sigmoid(pel[h]).cpu().numpy())
        return np.concatenate(all_probs), [np.concatenate(p) for p in all_per_enzyme]

    human_binary, human_enzyme = score_batch(human_pooled, human_edited)
    chimp_binary, chimp_enzyme = score_batch(chimp_pooled, chimp_edited)

    # Spearman correlation
    sp_binary = stats.spearmanr(human_binary, chimp_binary)
    pe_binary = stats.pearsonr(human_binary, chimp_binary)

    logger.info(f"  Binary head: Spearman r={sp_binary.statistic:.4f} (p={sp_binary.pvalue:.2e}), "
                f"Pearson r={pe_binary.statistic:.4f}")

    cs_results = {
        "n_sites": int(n),
        "binary_head": {
            "spearman_r": float(sp_binary.statistic),
            "spearman_p": float(sp_binary.pvalue),
            "pearson_r": float(pe_binary.statistic),
            "pearson_p": float(pe_binary.pvalue),
            "human_mean": float(np.mean(human_binary)),
            "chimp_mean": float(np.mean(chimp_binary)),
        },
        "per_enzyme_heads": {},
    }

    for h, enz in enumerate(ENZYMES):
        sp = stats.spearmanr(human_enzyme[h], chimp_enzyme[h])
        pe = stats.pearsonr(human_enzyme[h], chimp_enzyme[h])
        cs_results["per_enzyme_heads"][enz] = {
            "spearman_r": float(sp.statistic),
            "spearman_p": float(sp.pvalue),
            "pearson_r": float(pe.statistic),
        }
        logger.info(f"  {enz} adapter: Spearman r={sp.statistic:.4f} (p={sp.pvalue:.2e})")

    # Save per-site scores
    cs_df = pd.DataFrame({
        "site_id": site_ids,
        "human_binary": human_binary,
        "chimp_binary": chimp_binary,
    })
    for h, enz in enumerate(ENZYMES):
        cs_df[f"human_{enz}"] = human_enzyme[h]
        cs_df[f"chimp_{enz}"] = chimp_enzyme[h]
    cs_df.to_csv(OUTPUT_DIR / "cross_species_scores.csv", index=False)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(human_binary, chimp_binary, alpha=0.3, s=10, c="steelblue")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3)
    axes[0].set_xlabel("Human score")
    axes[0].set_ylabel("Chimp score")
    axes[0].set_title(f"Binary head: Spearman r={sp_binary.statistic:.3f}")
    axes[0].grid(alpha=0.3)

    # Per-enzyme correlations bar chart
    enz_names = ENZYMES
    enz_rs = [cs_results["per_enzyme_heads"][e]["spearman_r"] for e in enz_names]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
    axes[1].bar(enz_names, enz_rs, color=colors)
    axes[1].set_ylabel("Spearman r")
    axes[1].set_title("Cross-species Spearman by adapter head")
    axes[1].grid(axis="y", alpha=0.3)
    for i, (name, r) in enumerate(zip(enz_names, enz_rs)):
        axes[1].annotate(f"{r:.3f}", (i, r), ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cross_species.png", dpi=150)
    plt.close()

    with open(OUTPUT_DIR / "cross_species_results.json", "w") as f:
        json.dump(cs_results, f, indent=2)

    return cs_results


# ============================================================================
# Step 4: gnomAD gene-level constraint
# ============================================================================

def _map_sites_to_genes(site_ids, refgene_path):
    """Map chr:pos:strand site IDs to gene names using refGene.

    Builds an interval lookup from refGene transcripts (txStart-txEnd)
    and maps each site to the gene whose transcript it falls within.
    """
    from collections import defaultdict

    # Parse refGene: fields are tab-separated
    # col 2=name, col 3=chrom, col 4=strand, col 5=txStart, col 6=txEnd, col 12=gene
    gene_intervals = defaultdict(list)  # chrom -> list of (start, end, gene)
    with open(refgene_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 13:
                continue
            chrom = parts[2]
            tx_start = int(parts[4])
            tx_end = int(parts[5])
            gene = parts[12]
            gene_intervals[chrom].append((tx_start, tx_end, gene))

    # Sort intervals for binary search
    for chrom in gene_intervals:
        gene_intervals[chrom].sort()

    # Map each site
    site_to_gene = {}
    for sid in site_ids:
        parts = str(sid).split(":")
        if len(parts) < 3:
            continue
        chrom = parts[0]
        try:
            pos = int(parts[1])
        except ValueError:
            continue

        # Linear scan (fast enough for ~7k positives)
        best_gene = None
        for start, end, gene in gene_intervals.get(chrom, []):
            if start <= pos <= end:
                best_gene = gene
                break  # Take first match
            if start > pos:
                break

        if best_gene:
            site_to_gene[sid] = best_gene

    return site_to_gene


def run_gnomad_analysis(data, binary_probs, per_enzyme_probs):
    """Correlate neural editability scores with gnomAD constraint."""
    logger.info("\n" + "=" * 70)
    logger.info("STEP 4: gnomAD Gene-Level Constraint Analysis")
    logger.info("=" * 70)

    gnomad_path = PROJECT_ROOT / "data" / "raw" / "gnomad" / "gnomad_v4.1_constraint.tsv"
    if not gnomad_path.exists():
        logger.warning(f"Missing {gnomad_path}")
        return {}

    # Load gnomAD constraint
    gdf = pd.read_csv(gnomad_path, sep="\t")
    # Keep canonical transcripts
    gdf = gdf[gdf["canonical"] == True].copy()
    # LOEUF = lof.oe_ci.upper
    gdf["LOEUF"] = pd.to_numeric(gdf["lof.oe_ci.upper"], errors="coerce")
    gdf["pLI"] = pd.to_numeric(gdf["lof.pLI"], errors="coerce")
    gdf["missense_z"] = pd.to_numeric(gdf["mis.z_score"], errors="coerce")
    gdf = gdf.dropna(subset=["LOEUF"])
    gdf = gdf.drop_duplicates(subset=["gene"])
    logger.info(f"  gnomAD: {len(gdf)} genes with LOEUF")

    # Map site_ids to genes using refGene (hg38)
    df = data["df"]
    pos_mask = df["is_edited"] == 1

    refgene_path = PROJECT_ROOT / "data" / "raw" / "genomes" / "refGene.txt"
    if not refgene_path.exists():
        logger.warning(f"Missing {refgene_path}. Skipping gnomAD analysis.")
        return {}

    site_to_gene = _map_sites_to_genes(df[pos_mask]["site_id"].tolist(), refgene_path)
    logger.info(f"  Mapped {len(site_to_gene)}/{pos_mask.sum()} positive sites to genes")

    if len(site_to_gene) < 50:
        logger.warning("Too few sites mapped to genes. Skipping gnomAD analysis.")
        return {}

    pos_df = df[pos_mask].copy()
    pos_df["neural_score"] = binary_probs[pos_mask.values]
    for h, enz in enumerate(ENZYMES):
        pos_df[f"neural_{enz}"] = per_enzyme_probs[h][pos_mask.values]

    pos_df["gene"] = pos_df["site_id"].map(site_to_gene)
    pos_df = pos_df.dropna(subset=["gene"])

    # Compute per-gene mean neural editability
    gene_scores = pos_df.groupby("gene").agg(
        mean_neural=("neural_score", "mean"),
        n_sites=("neural_score", "count"),
    ).reset_index()

    logger.info(f"  {len(gene_scores)} genes with editing sites")

    # Merge with gnomAD
    merged = gene_scores.merge(gdf[["gene", "LOEUF", "pLI", "missense_z"]], on="gene", how="inner")
    logger.info(f"  {len(merged)} genes matched to gnomAD")

    if len(merged) < 10:
        logger.warning("Too few matched genes for meaningful analysis")
        return {}

    results = {"n_genes_matched": len(merged)}

    # Correlations
    for metric in ["LOEUF", "pLI", "missense_z"]:
        vals = merged[metric].dropna()
        idx = vals.index
        sp = stats.spearmanr(merged.loc[idx, "mean_neural"], vals)
        pe = stats.pearsonr(merged.loc[idx, "mean_neural"], vals)
        results[f"neural_vs_{metric}"] = {
            "spearman_r": float(sp.statistic),
            "spearman_p": float(sp.pvalue),
            "pearson_r": float(pe.statistic),
            "pearson_p": float(pe.pvalue),
        }
        logger.info(f"  neural vs {metric}: Spearman r={sp.statistic:.4f} (p={sp.pvalue:.2e})")

    # n_sites vs constraint
    sp_nsites = stats.spearmanr(merged["n_sites"], merged["LOEUF"])
    results["n_sites_vs_LOEUF"] = {
        "spearman_r": float(sp_nsites.statistic),
        "spearman_p": float(sp_nsites.pvalue),
    }
    logger.info(f"  n_sites vs LOEUF: Spearman r={sp_nsites.statistic:.4f} (p={sp_nsites.pvalue:.2e})")

    # Editing genes vs non-editing genes (like original analysis)
    editing_genes = set(merged["gene"])
    gdf_edit = gdf[gdf["gene"].isin(editing_genes)]
    gdf_noedit = gdf[~gdf["gene"].isin(editing_genes)]

    for metric in ["LOEUF", "pLI"]:
        v1 = gdf_edit[metric].dropna()
        v2 = gdf_noedit[metric].dropna()
        U, p = stats.mannwhitneyu(v1, v2, alternative="two-sided")
        results[f"editing_vs_nonediting_{metric}"] = {
            "editing_mean": float(v1.mean()),
            "nonediting_mean": float(v2.mean()),
            "editing_median": float(v1.median()),
            "nonediting_median": float(v2.median()),
            "mw_p": float(p),
            "n_editing": len(v1),
            "n_nonediting": len(v2),
        }
        logger.info(f"  Editing vs non-editing {metric}: "
                    f"{v1.mean():.3f} vs {v2.mean():.3f} (p={p:.2e})")

    # High-pLI enrichment
    high_pli_thresh = 0.9
    n_edit_high = (gdf_edit["pLI"].dropna() >= high_pli_thresh).sum()
    n_edit_total = len(gdf_edit["pLI"].dropna())
    n_noedit_high = (gdf_noedit["pLI"].dropna() >= high_pli_thresh).sum()
    n_noedit_total = len(gdf_noedit["pLI"].dropna())

    frac_edit = n_edit_high / max(n_edit_total, 1)
    frac_noedit = n_noedit_high / max(n_noedit_total, 1)

    table = [[n_edit_high, n_noedit_high],
             [n_edit_total - n_edit_high, n_noedit_total - n_noedit_high]]
    OR, fisher_p = stats.fisher_exact(table, alternative="greater")

    results["high_pli_enrichment"] = {
        "frac_editing": float(frac_edit),
        "frac_nonediting": float(frac_noedit),
        "fisher_OR": float(OR),
        "fisher_p": float(fisher_p),
    }
    logger.info(f"  High pLI enrichment: {frac_edit:.3f} vs {frac_noedit:.3f}, OR={OR:.2f}, p={fisher_p:.2e}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # LOEUF distribution
    axes[0].hist(gdf_edit["LOEUF"].dropna(), bins=30, alpha=0.6, label=f"Editing ({len(gdf_edit)})", density=True)
    axes[0].hist(gdf_noedit["LOEUF"].dropna(), bins=30, alpha=0.4, label=f"Non-editing ({len(gdf_noedit)})", density=True)
    axes[0].set_xlabel("LOEUF")
    axes[0].set_title("LOEUF Distribution")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Neural score vs LOEUF
    axes[1].scatter(merged["mean_neural"], merged["LOEUF"], alpha=0.2, s=5, c="steelblue")
    sp_val = results["neural_vs_LOEUF"]["spearman_r"]
    axes[1].set_xlabel("Mean Neural Editability")
    axes[1].set_ylabel("LOEUF")
    axes[1].set_title(f"Neural Score vs LOEUF (r={sp_val:.3f})")
    axes[1].grid(alpha=0.3)

    # n_sites vs LOEUF
    axes[2].scatter(merged["n_sites"], merged["LOEUF"], alpha=0.2, s=5, c="darkorange")
    axes[2].set_xlabel("Number of editing sites")
    axes[2].set_ylabel("LOEUF")
    axes[2].set_title(f"n_sites vs LOEUF (r={sp_nsites.statistic:.3f})")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "gnomad_constraint.png", dpi=150)
    plt.close()

    with open(OUTPUT_DIR / "gnomad_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


# ============================================================================
# Comparison Report
# ============================================================================

def generate_report(cv_results, tcga_results, cs_results, gnomad_results, runtime_min):
    """Generate comprehensive comparison report."""
    logger.info("\n" + "=" * 70)
    logger.info("Generating comparison report...")
    logger.info("=" * 70)

    # Load existing XGBoost proxy results for comparison
    xgb_proxy_cv = {}
    xgb_proxy_path = PROJECT_ROOT / "experiments" / "multi_enzyme" / "outputs" / "phase3_neural_validation" / "cv_results.json"
    if xgb_proxy_path.exists():
        with open(xgb_proxy_path) as f:
            xgb_proxy_cv = json.load(f)

    # Load original 40d TCGA results
    tcga_40d = {}
    tcga_40d_path = PROJECT_ROOT / "experiments" / "multi_enzyme" / "outputs" / "tcga_gnomad" / "tcga_full_model_results.json"
    if tcga_40d_path.exists():
        with open(tcga_40d_path) as f:
            tcga_40d = json.load(f)

    # Load original gnomAD results
    gnomad_orig = {}
    gnomad_orig_path = PROJECT_ROOT / "experiments" / "multi_enzyme" / "outputs" / "tcga_gnomad" / "gnomad_constraint_results.json"
    if gnomad_orig_path.exists():
        with open(gnomad_orig_path) as f:
            gnomad_orig = json.load(f)

    lines = []
    lines.append("# Phase 3 TRUE Neural Model Validation")
    lines.append("")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"Runtime: {runtime_min:.1f} minutes")
    lines.append(f"Device: {DEVICE}")
    lines.append("")
    lines.append("## Model Architecture")
    lines.append("")
    lines.append("```")
    lines.append(f"Input: RNA-FM(640) + edit_delta(640) + hand(40) = {D_INPUT}-dim")
    lines.append("-> Linear(1320, 256) -> GELU -> Dropout(0.3) -> LayerNorm")
    lines.append("-> Linear(256, 128) -> GELU -> Dropout(0.2) -> 128-dim shared")
    lines.append("-> Binary head: Linear(128, 1)")
    lines.append("-> Per-enzyme adapters: Linear(128, 32) -> GELU -> Linear(32, 1) x 5")
    lines.append("Two-stage: S1(10ep joint) -> S2(20ep adapters-only)")
    lines.append("```")
    lines.append("")

    # ---- CV Results ----
    lines.append("## 1. Classification (5-fold CV)")
    lines.append("")
    lines.append("| Enzyme | XGB 40d | XGB 1280d | XGB 1320d | Phase3 Neural | Delta vs XGB_40d |")
    lines.append("|--------|---------|-----------|-----------|---------------|-----------------|")

    for enz in ENZYMES:
        neural_m = cv_results.get("per_enzyme_auroc", {}).get(enz, {}).get("mean", 0)
        neural_s = cv_results.get("per_enzyme_auroc", {}).get(enz, {}).get("std", 0)

        xgb_40 = xgb_proxy_cv.get(enz, {}).get("XGB_HandFeatures_40d", {}).get("mean_auroc", 0)
        xgb_1280 = xgb_proxy_cv.get(enz, {}).get("XGB_RNAFM_1280d", {}).get("mean_auroc", 0)
        xgb_1320 = xgb_proxy_cv.get(enz, {}).get("XGB_RNAFM+Hand_1320d", {}).get("mean_auroc", 0)

        delta = neural_m - xgb_40 if xgb_40 > 0 else 0
        lines.append(f"| {enz} | {xgb_40:.3f} | {xgb_1280:.3f} | {xgb_1320:.3f} | "
                     f"{neural_m:.3f}+/-{neural_s:.3f} | {delta:+.3f} |")

    overall_m = cv_results.get("overall_auroc", {}).get("mean", 0)
    overall_s = cv_results.get("overall_auroc", {}).get("std", 0)
    lines.append(f"| **Overall** | --- | --- | --- | {overall_m:.3f}+/-{overall_s:.3f} | --- |")
    lines.append("")

    # ---- TCGA Results ----
    lines.append("## 2. TCGA Somatic Mutation Enrichment")
    lines.append("")
    lines.append("| Cancer | n_mut | XGB_40d delta | XGB_1280d delta | Neural delta | Neural p | Neural TC delta | Neural nonTC delta |")
    lines.append("|--------|-------|---------------|-----------------|-------------|----------|-----------------|-------------------|")

    xgb_proxy_tcga_path = PROJECT_ROOT / "experiments" / "multi_enzyme" / "outputs" / "phase3_neural_validation" / "all_results.json"
    xgb_proxy_tcga = {}
    if xgb_proxy_tcga_path.exists():
        with open(xgb_proxy_tcga_path) as f:
            xgb_proxy_tcga = json.load(f).get("tcga_results", {})

    for cancer in CANCERS:
        if cancer not in tcga_results:
            continue
        tr = tcga_results[cancer]
        n_mut = tr["n_mutations"]
        bh = tr.get("binary_head", {})

        # Original 40d
        d40 = "---"
        if cancer in tcga_40d:
            od = tcga_40d[cancer]
            d40_val = od.get("delta", od.get("mean_score_mutations", 0) - od.get("mean_score_controls", 0))
            d40 = f"{d40_val:.4f}"

        # XGB 1280d proxy
        d1280 = "---"
        if cancer in xgb_proxy_tcga:
            enr = xgb_proxy_tcga[cancer].get("binary_A3A_1280d", {})
            d1280 = f"{enr.get('delta', 0):.4f}"

        neural_d = f"{bh.get('delta', 0):.4f}"
        neural_p = f"{bh.get('mw_p', 1.0):.2e}"
        tc_d = f"{bh.get('TC_delta', 0):.4f}" if "TC_delta" in bh else "---"
        ntc_d = f"{bh.get('nonTC_delta', 0):.4f}" if "nonTC_delta" in bh else "---"

        lines.append(f"| {cancer.upper()} | {n_mut:,} | {d40} | {d1280} | {neural_d} | {neural_p} | {tc_d} | {ntc_d} |")

    lines.append("")

    # TCGA per-enzyme adapter results
    lines.append("### Per-Enzyme Adapter TCGA Enrichment (delta = mut - ctrl)")
    lines.append("")
    lines.append("| Cancer | A3A delta (p) | A3B delta (p) | A3G delta (p) | A3A_A3G delta (p) | Neither delta (p) |")
    lines.append("|--------|---------------|---------------|---------------|-------------------|--------------------|")

    for cancer in CANCERS:
        if cancer not in tcga_results:
            continue
        tr = tcga_results[cancer]
        row = f"| {cancer.upper()} |"
        for enz in ENZYMES:
            r = tr.get(f"adapter_{enz}", {})
            d = r.get("delta", 0)
            p = r.get("mw_p", 1.0)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            row += f" {d:+.4f}{sig} |"
        lines.append(row)
    lines.append("")

    # ---- Cross-species ----
    lines.append("## 3. Cross-Species Conservation (Human vs Chimp)")
    lines.append("")
    if cs_results:
        bh = cs_results.get("binary_head", {})
        lines.append(f"- **{cs_results.get('n_sites', 0)} conserved editing sites**")
        lines.append(f"- Binary head: Spearman r={bh.get('spearman_r', 0):.4f} (p={bh.get('spearman_p', 1):.2e}), "
                     f"Pearson r={bh.get('pearson_r', 0):.4f}")
        lines.append("")
        lines.append("| Adapter | Spearman r | p-value |")
        lines.append("|---------|-----------|---------|")
        for enz in ENZYMES:
            er = cs_results.get("per_enzyme_heads", {}).get(enz, {})
            lines.append(f"| {enz} | {er.get('spearman_r', 0):.4f} | {er.get('spearman_p', 1):.2e} |")
        lines.append("")
        lines.append("Previous result (XGBoost proxy): Spearman r=0.632")
        lines.append("")
    else:
        lines.append("Cross-species data not available.")
        lines.append("")

    # ---- gnomAD ----
    lines.append("## 4. gnomAD Gene-Level Constraint")
    lines.append("")
    if gnomad_results:
        lines.append(f"- **{gnomad_results.get('n_genes_matched', 0)} genes** matched to gnomAD")
        lines.append("")

        for metric in ["LOEUF", "pLI", "missense_z"]:
            key = f"neural_vs_{metric}"
            if key in gnomad_results:
                r = gnomad_results[key]
                lines.append(f"- Neural score vs {metric}: Spearman r={r['spearman_r']:.4f} (p={r['spearman_p']:.2e})")

        lines.append("")

        for metric in ["LOEUF", "pLI"]:
            key = f"editing_vs_nonediting_{metric}"
            if key in gnomad_results:
                r = gnomad_results[key]
                lines.append(f"- Editing vs non-editing genes ({metric}): "
                             f"{r['editing_mean']:.3f} vs {r['nonediting_mean']:.3f} (p={r['mw_p']:.2e})")

        lines.append("")

        if "high_pli_enrichment" in gnomad_results:
            hpe = gnomad_results["high_pli_enrichment"]
            lines.append(f"- High pLI enrichment: editing={hpe['frac_editing']:.3f}, "
                         f"non-editing={hpe['frac_nonediting']:.3f}, "
                         f"OR={hpe['fisher_OR']:.2f} (p={hpe['fisher_p']:.2e})")

        lines.append("")

        # Compare to original
        if gnomad_orig:
            lines.append("### Comparison to original XGBoost 40d analysis")
            lines.append("")
            orig_loeuf = gnomad_orig.get("tests", {}).get("LOEUF", {})
            if orig_loeuf:
                lines.append(f"- Original: editing LOEUF mean={orig_loeuf.get('mean_with_editing', '?'):.3f}, "
                             f"non-editing={orig_loeuf.get('mean_without_editing', '?'):.3f} "
                             f"(p={orig_loeuf.get('mann_whitney_p', '?'):.2e})")
                if "editing_vs_nonediting_LOEUF" in gnomad_results:
                    r = gnomad_results["editing_vs_nonediting_LOEUF"]
                    lines.append(f"- Neural: editing LOEUF mean={r['editing_mean']:.3f}, "
                                 f"non-editing={r['nonediting_mean']:.3f} (p={r['mw_p']:.2e})")
            lines.append("")
    else:
        lines.append("gnomAD data not available or no gene column in splits.")
        lines.append("")

    # ---- Summary ----
    lines.append("## Summary")
    lines.append("")
    lines.append("### Key Findings")
    lines.append("")

    # CV comparison
    lines.append("**Classification:**")
    for enz in ENZYMES:
        neural_m = cv_results.get("per_enzyme_auroc", {}).get(enz, {}).get("mean", 0)
        xgb_40 = xgb_proxy_cv.get(enz, {}).get("XGB_HandFeatures_40d", {}).get("mean_auroc", 0)
        xgb_1320 = xgb_proxy_cv.get(enz, {}).get("XGB_RNAFM+Hand_1320d", {}).get("mean_auroc", 0)
        if neural_m > 0:
            vs40 = neural_m - xgb_40 if xgb_40 > 0 else 0
            vs1320 = neural_m - xgb_1320 if xgb_1320 > 0 else 0
            lines.append(f"- {enz}: Neural={neural_m:.3f}, vs XGB_40d: {vs40:+.3f}, vs XGB_1320d: {vs1320:+.3f}")
    lines.append("")

    # TCGA summary
    lines.append("**TCGA Enrichment:**")
    n_enriched = sum(1 for c in tcga_results.values()
                     if c.get("binary_head", {}).get("delta", 0) > 0)
    lines.append(f"- {n_enriched}/{len(tcga_results)} cancers show mutation enrichment")
    for cancer in CANCERS:
        if cancer in tcga_results:
            bh = tcga_results[cancer].get("binary_head", {})
            d = bh.get("delta", 0)
            p = bh.get("mw_p", 1)
            sig = " ***" if p < 1e-10 else " **" if p < 0.01 else " *" if p < 0.05 else ""
            lines.append(f"  - {cancer.upper()}: delta={d:+.4f}{sig}")
    lines.append("")

    report = "\n".join(lines)
    report_path = OUTPUT_DIR / "comparison_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    logger.info(f"Report saved to {report_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    start_time = time.time()
    logger.info("Phase 3 TRUE Neural Model Validation")
    logger.info(f"Device: {DEVICE}")

    # Load data
    data = load_v3_data()

    # Step 1: 5-fold CV
    cv_results = run_cv(data)

    # Train full model for external scoring
    model, binary_probs, per_enzyme_probs = train_full_model(data)

    # Step 2: TCGA scoring
    tcga_results = score_tcga(model, data)

    # Step 3: Cross-species
    cs_results = score_cross_species(model)

    # Step 4: gnomAD
    gnomad_results = run_gnomad_analysis(data, binary_probs, per_enzyme_probs)

    # Generate report
    runtime_min = (time.time() - start_time) / 60.0
    generate_report(cv_results, tcga_results, cs_results, gnomad_results, runtime_min)

    logger.info(f"\nTotal runtime: {runtime_min:.1f} minutes")
    logger.info(f"All results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
