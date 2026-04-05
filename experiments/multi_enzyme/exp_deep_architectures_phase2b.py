#!/usr/bin/env python
"""Phase 2b: Per-enzyme head architecture comparison.

Key changes from Phase 2:
  1. Per-enzyme binary heads (A3A, A3B, A3G, A3A_A3G, Neither) alongside overall binary head
  2. Unified V1 baseline included for comparison
  3. 1-2 folds only for faster screening
  4. Evaluate BOTH overall AND per-enzyme AUROC

Head structure (all models):
  Shared encoder -> 64-dim representation
    -> Binary head: P(edited by any enzyme) -- sigmoid, BCE, all samples
    -> A3A head: P(edited by A3A) -- sigmoid, only A3A pos + A3A neg
    -> A3B head: P(edited by A3B) -- sigmoid, only A3B pos + A3B neg
    -> A3G head: P(edited by A3G) -- sigmoid, only A3G pos + A3G neg
    -> APOBEC1 head: P(edited by APOBEC1/"Neither") -- sigmoid
    -> Enzyme classifier: 6-class softmax, positives only

  Total loss = binary_loss + 0.2 * sum(per_enzyme_losses) + 0.1 * enzyme_classifier_loss

Models on v3 (15,352 sites):
  1. Conv2D_BP (Phase 1 best, 0.793 overall)
  2. DualPath_Residual (Phase 1 2nd, 0.790)
  3. AdarEdit_GAT (Phase 2 new)
  4. Unified_V1 (MLP on 40-dim hand + RNA-FM embeddings)

Best v3 model retrained on v4-large (385K sites).
XGBoost per-enzyme on v3 for reference.

Output: experiments/multi_enzyme/outputs/deep_architectures/
"""

import gc
import json
import logging
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

log_file = LOG_DIR / "phase2b_training.log"
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

# Per-enzyme head enzymes (exclude Unknown -- too few samples)
PER_ENZYME_HEADS = ["A3A", "A3B", "A3G", "A3A_A3G", "Neither"]
PER_ENZYME_HEAD_INDICES = [ENZYME_TO_IDX[e] for e in PER_ENZYME_HEADS]
N_PER_ENZYME = len(PER_ENZYME_HEADS)

# Screening: 2 folds
N_FOLDS = 2
V3_BATCH_SIZE = 64
V3_N_EPOCHS = 50
V3_LR = 1e-3
V3_WEIGHT_DECAY = 1e-4

V4_BATCH_SIZE = 512
V4_N_EPOCHS = 30
V4_LR = 1e-3

ENZYME_HEAD_WEIGHT = 0.2  # Weight for per-enzyme head losses
ENZYME_CLS_WEIGHT = 0.1   # Weight for enzyme classifier loss


# ---------------------------------------------------------------------------
# ViennaRNA folding (parallel)
# ---------------------------------------------------------------------------


def _fold_worker(args):
    """Worker for parallel folding: returns (site_id, dot_bracket)."""
    import RNA
    site_id, seq = args
    try:
        seq = seq.upper().replace("T", "U")
        md = RNA.md()
        md.temperature = 37.0
        fc = RNA.fold_compound(seq, md)
        structure, _ = fc.mfe()
        return site_id, structure
    except Exception:
        return site_id, "." * len(seq)


def _fold_and_bpp_worker(args):
    """Worker for parallel folding: returns (site_id, dot_bracket, 41x41 submatrix)."""
    import RNA
    site_id, seq = args
    try:
        seq = seq.upper().replace("T", "U")
        n = len(seq)
        md = RNA.md()
        md.temperature = 37.0
        fc = RNA.fold_compound(seq, md)
        structure, _ = fc.mfe()
        fc.pf()
        bpp_raw = np.array(fc.bpp())
        bpp = bpp_raw[1:n + 1, 1:n + 1].astype(np.float32)
        sub = bpp[80:121, 80:121].copy()
        for r in range(41):
            for c in range(41):
                if abs(r - c) < 3:
                    sub[r, c] = 0.0
        return site_id, structure, sub
    except Exception:
        return site_id, "." * len(seq), np.zeros((41, 41), dtype=np.float32)


def _match_base_pairs(dot_bracket: str) -> np.ndarray:
    """Parse dot-bracket to build base-pair indicator matrix."""
    n = len(dot_bracket)
    bp_matrix = np.zeros((n, n), dtype=np.float32)
    stack = []
    for i, c in enumerate(dot_bracket):
        if c == "(":
            stack.append(i)
        elif c == ")" and stack:
            j = stack.pop()
            bp_matrix[i, j] = 1.0
            bp_matrix[j, i] = 1.0
    return bp_matrix


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------


def load_v3_data() -> Dict:
    """Load v3 dataset (15,342 sites)."""
    logger.info("=" * 60)
    logger.info("Loading V3 dataset")
    logger.info("=" * 60)

    splits_path = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "splits_multi_enzyme_v3_with_negatives.csv"
    df = pd.read_csv(splits_path)
    logger.info(f"  Loaded {len(df)} sites")

    seq_path = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "multi_enzyme_sequences_v3_with_negatives.json"
    with open(seq_path) as f:
        sequences = json.load(f)

    loop_path = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "loop_position_per_site_v3.csv"
    loop_df = pd.read_csv(loop_path)
    db_lookup = {}
    for _, row in loop_df.iterrows():
        if pd.notna(row.get("dot_bracket")):
            db_lookup[str(row["site_id"])] = row["dot_bracket"]
    loop_df = loop_df.drop_duplicates(subset=["site_id"]).set_index("site_id")

    struct_path = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "structure_cache_multi_enzyme_v3.npz"
    struct_data = np.load(struct_path, allow_pickle=True)
    struct_ids = list(struct_data["site_ids"])
    struct_deltas = struct_data["delta_features"]
    structure_delta = {sid: struct_deltas[i] for i, sid in enumerate(struct_ids)}

    site_ids = df["site_id"].tolist()
    valid_mask = [sid in sequences for sid in site_ids]
    df = df[valid_mask].reset_index(drop=True)
    site_ids = df["site_id"].tolist()
    n = len(site_ids)
    logger.info(f"  {n} sites with sequences")

    hand_features = build_hand_features(site_ids, sequences, structure_delta, loop_df)
    logger.info(f"  Hand features shape: {hand_features.shape}")

    # One-hot encode sequences
    base_map = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3}
    onehot_seqs = np.zeros((n, 4, 201), dtype=np.float32)
    for i, sid in enumerate(site_ids):
        seq = sequences[sid].upper().replace("T", "U")
        for j, base in enumerate(seq[:201]):
            if base in base_map:
                onehot_seqs[i, base_map[base], j] = 1.0

    # Fold sequences for structures and BPP
    logger.info("Computing ViennaRNA structures and BPP submatrices...")
    work_items = [(sid, sequences[sid]) for sid in site_ids]
    n_workers = min(14, os.cpu_count() or 4)
    dot_brackets = {}
    bp_submatrices = np.zeros((n, 41, 41), dtype=np.float32)
    sid_to_idx = {sid: i for i, sid in enumerate(site_ids)}

    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_fold_and_bpp_worker, item): item[0] for item in work_items}
        for future in as_completed(futures):
            sid, structure, sub = future.result()
            dot_brackets[sid] = db_lookup.get(sid, structure)
            idx = sid_to_idx[sid]
            bp_submatrices[idx] = sub
            done += 1
            if done % 3000 == 0:
                logger.info(f"    Folded {done}/{n} ({time.time()-t0:.0f}s)")

    logger.info(f"  Folding complete: {done} sequences in {time.time()-t0:.0f}s")

    struct_onehot = np.zeros((n, 3, 201), dtype=np.float32)
    dot_bracket_list = []
    for i, sid in enumerate(site_ids):
        db = dot_brackets.get(sid, "." * 201)
        dot_bracket_list.append(db)
        for j, c in enumerate(db[:201]):
            if c == "(":
                struct_onehot[i, 0, j] = 1.0
            elif c == ")":
                struct_onehot[i, 1, j] = 1.0
            else:
                struct_onehot[i, 2, j] = 1.0

    labels_binary = df["is_edited"].values.astype(np.float32)
    labels_enzyme = np.array([ENZYME_TO_IDX.get(e, 5) for e in df["enzyme"].values], dtype=np.int64)

    onehot_edited = onehot_seqs.copy()
    onehot_edited[:, 1, CENTER] = 0.0
    onehot_edited[:, 3, CENTER] = 1.0

    # Build per-enzyme labels: for each enzyme, 1=positive for that enzyme, 0=negative for that enzyme, -1=not relevant
    # A sample is relevant to an enzyme if it's a positive of that enzyme OR a negative assigned to that enzyme.
    # In our v3 dataset, negatives are shared across enzymes. So we treat:
    #   - positive for enzyme X: label=1 for head X
    #   - negative (any enzyme neg): label=0 for ALL per-enzyme heads
    # This means each per-enzyme head sees: (enzyme X positives, label=1) + (all negatives, label=0)
    per_enzyme_labels = np.full((n, N_PER_ENZYME), -1, dtype=np.float32)  # -1 = ignore
    for head_idx, enz_name in enumerate(PER_ENZYME_HEADS):
        enz_idx = ENZYME_TO_IDX[enz_name]
        # Positives for this enzyme
        pos_mask = (labels_binary == 1) & (labels_enzyme == enz_idx)
        per_enzyme_labels[pos_mask, head_idx] = 1.0
        # Negatives: all negatives are valid negatives for each enzyme head
        neg_mask = labels_binary == 0
        per_enzyme_labels[neg_mask, head_idx] = 0.0

    logger.info("Per-enzyme head sample counts:")
    for head_idx, enz_name in enumerate(PER_ENZYME_HEADS):
        n_pos = (per_enzyme_labels[:, head_idx] == 1).sum()
        n_neg = (per_enzyme_labels[:, head_idx] == 0).sum()
        logger.info(f"  {enz_name}: {n_pos} pos, {n_neg} neg")

    logger.info("V3 data loading complete.")
    return {
        "site_ids": site_ids,
        "df": df,
        "hand_features": hand_features,
        "onehot_seqs": onehot_seqs,
        "onehot_edited": onehot_edited,
        "struct_onehot": struct_onehot,
        "bp_submatrices": bp_submatrices,
        "dot_bracket_list": dot_bracket_list,
        "labels_binary": labels_binary,
        "labels_enzyme": labels_enzyme,
        "per_enzyme_labels": per_enzyme_labels,
        "sequences": sequences,
    }


def load_v4_data() -> Dict:
    """Load v4-large dataset (385,764 sites)."""
    logger.info("=" * 60)
    logger.info("Loading V4-large dataset")
    logger.info("=" * 60)

    splits_path = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "splits_v4_large_negatives.csv"
    df = pd.read_csv(splits_path, low_memory=False)
    logger.info(f"  Loaded {len(df)} sites")
    logger.info(f"  Positives: {(df['is_edited']==1).sum()}, Negatives: {(df['is_edited']==0).sum()}")

    seq_path = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "multi_enzyme_sequences_v4_large.json"
    with open(seq_path) as f:
        sequences = json.load(f)

    loop_path = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "loop_position_per_site_v3.csv"
    loop_df = pd.read_csv(loop_path)
    db_lookup = {}
    for _, row in loop_df.iterrows():
        if pd.notna(row.get("dot_bracket")):
            db_lookup[str(row["site_id"])] = row["dot_bracket"]
    loop_df = loop_df.drop_duplicates(subset=["site_id"]).set_index("site_id")

    struct_path = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "structure_cache_multi_enzyme_v3.npz"
    struct_data = np.load(struct_path, allow_pickle=True)
    struct_ids = list(struct_data["site_ids"])
    struct_deltas = struct_data["delta_features"]
    structure_delta = {sid: struct_deltas[i] for i, sid in enumerate(struct_ids)}

    site_ids = df["site_id"].tolist()
    valid_mask = [sid in sequences for sid in site_ids]
    df = df[valid_mask].reset_index(drop=True)
    site_ids = df["site_id"].tolist()
    n = len(site_ids)
    logger.info(f"  {n} sites with sequences")

    hand_features = build_hand_features(site_ids, sequences, structure_delta, loop_df)

    base_map = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3}
    onehot_seqs = np.zeros((n, 4, 201), dtype=np.float32)
    for i, sid in enumerate(site_ids):
        seq = sequences[sid].upper().replace("T", "U")
        for j, base in enumerate(seq[:201]):
            if base in base_map:
                onehot_seqs[i, base_map[base], j] = 1.0

    # Fold only positives (negatives get zeros for BPP)
    pos_mask = df["is_edited"].values == 1
    pos_site_ids = [sid for sid, m in zip(site_ids, pos_mask) if m]
    logger.info(f"Computing ViennaRNA structures for {len(pos_site_ids)} positive sites...")

    work_items = [(sid, sequences[sid]) for sid in pos_site_ids]
    n_workers = min(14, os.cpu_count() or 4)
    dot_brackets = {}
    bp_submatrices = np.zeros((n, 41, 41), dtype=np.float32)
    sid_to_idx = {sid: i for i, sid in enumerate(site_ids)}

    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_fold_and_bpp_worker, item): item[0] for item in work_items}
        for future in as_completed(futures):
            sid, structure, sub = future.result()
            dot_brackets[sid] = db_lookup.get(sid, structure)
            idx = sid_to_idx[sid]
            bp_submatrices[idx] = sub
            done += 1
            if done % 2000 == 0:
                logger.info(f"    Folded {done}/{len(pos_site_ids)} positives ({time.time()-t0:.0f}s)")

    logger.info(f"  Positive folding complete: {done} in {time.time()-t0:.0f}s")

    neg_site_ids = [sid for sid, m in zip(site_ids, pos_mask) if not m]
    logger.info(f"Computing MFE structures for {len(neg_site_ids)} negative sites...")
    neg_work = [(sid, sequences[sid]) for sid in neg_site_ids]
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_fold_worker, item): item[0] for item in neg_work}
        for future in as_completed(futures):
            sid, structure = future.result()
            dot_brackets[sid] = structure
            done += 1
            if done % 50000 == 0:
                logger.info(f"    Folded {done}/{len(neg_site_ids)} negatives ({time.time()-t0:.0f}s)")

    logger.info(f"  Negative folding complete: {done} in {time.time()-t0:.0f}s")

    struct_onehot = np.zeros((n, 3, 201), dtype=np.float32)
    dot_bracket_list = []
    for i, sid in enumerate(site_ids):
        db = dot_brackets.get(sid, "." * 201)
        dot_bracket_list.append(db)
        for j, c in enumerate(db[:201]):
            if c == "(":
                struct_onehot[i, 0, j] = 1.0
            elif c == ")":
                struct_onehot[i, 1, j] = 1.0
            else:
                struct_onehot[i, 2, j] = 1.0

    labels_binary = df["is_edited"].values.astype(np.float32)
    labels_enzyme = np.array([ENZYME_TO_IDX.get(e, 5) for e in df["enzyme"].values], dtype=np.int64)

    onehot_edited = onehot_seqs.copy()
    onehot_edited[:, 1, CENTER] = 0.0
    onehot_edited[:, 3, CENTER] = 1.0

    # Per-enzyme labels
    per_enzyme_labels = np.full((n, N_PER_ENZYME), -1, dtype=np.float32)
    for head_idx, enz_name in enumerate(PER_ENZYME_HEADS):
        enz_idx = ENZYME_TO_IDX[enz_name]
        pos_enz_mask = (labels_binary == 1) & (labels_enzyme == enz_idx)
        per_enzyme_labels[pos_enz_mask, head_idx] = 1.0
        neg_mask = labels_binary == 0
        per_enzyme_labels[neg_mask, head_idx] = 0.0

    logger.info("V4-large data loading complete.")
    return {
        "site_ids": site_ids,
        "df": df,
        "hand_features": hand_features,
        "onehot_seqs": onehot_seqs,
        "onehot_edited": onehot_edited,
        "struct_onehot": struct_onehot,
        "bp_submatrices": bp_submatrices,
        "dot_bracket_list": dot_bracket_list,
        "labels_binary": labels_binary,
        "labels_enzyme": labels_enzyme,
        "per_enzyme_labels": per_enzyme_labels,
        "sequences": sequences,
    }


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


class EditingDataset(Dataset):
    """Standard dataset for Conv2D_BP, DualPath, Unified V1."""

    def __init__(self, indices, data):
        self.indices = indices
        self.data = data

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        return {
            "onehot_seq": torch.from_numpy(self.data["onehot_seqs"][i]),
            "onehot_edited": torch.from_numpy(self.data["onehot_edited"][i]),
            "hand_feat": torch.from_numpy(self.data["hand_features"][i]),
            "struct_onehot": torch.from_numpy(self.data["struct_onehot"][i]),
            "bp_submatrix": torch.from_numpy(self.data["bp_submatrices"][i]).unsqueeze(0),
            "label_binary": torch.tensor(self.data["labels_binary"][i]),
            "label_enzyme": torch.tensor(self.data["labels_enzyme"][i]),
            "per_enzyme_labels": torch.from_numpy(self.data["per_enzyme_labels"][i]),
        }


class GATDataset(Dataset):
    """Dataset for GAT model. Builds graph per sample from dot-bracket structure."""

    def __init__(self, indices, data, window=41):
        self.indices = indices
        self.data = data
        self.window = window
        self.half_w = window // 2

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        seq = self.data["onehot_seqs"][i]
        hand_feat = self.data["hand_features"][i]
        db = self.data["dot_bracket_list"][i]

        start = CENTER - self.half_w
        end = CENTER + self.half_w + 1
        local_seq = seq[:, start:end]
        local_db = db[start:end]

        n_nodes = self.window
        node_feats = np.zeros((n_nodes, 22), dtype=np.float32)

        full_seq_arr = seq
        for j in range(n_nodes):
            pos = start + j
            if 0 <= pos < 201:
                node_feats[j, :4] = full_seq_arr[:, pos]
            if 0 <= pos - 1 < 201:
                node_feats[j, 4:8] = full_seq_arr[:, pos - 1]
            if 0 <= pos + 1 < 201:
                node_feats[j, 8:12] = full_seq_arr[:, pos + 1]
            if j < len(local_db):
                node_feats[j, 12] = 1.0 if local_db[j] == "." else 0.0
            w5_start = max(0, j - 2)
            w5_end = min(len(local_db), j + 3)
            local_region = local_db[w5_start:w5_end]
            node_feats[j, 13] = sum(1 for c in local_region if c == ".") / max(len(local_region), 1)
            p = j / n_nodes
            node_feats[j, 14] = np.sin(p * np.pi)
            node_feats[j, 15] = np.cos(p * np.pi)
            node_feats[j, 16] = np.sin(2 * p * np.pi)
            node_feats[j, 17] = np.cos(2 * p * np.pi)
            node_feats[j, 18] = 0.0 if (j < len(local_db) and local_db[j] == ".") else 1.0
            node_feats[j, 19] = abs(j - self.half_w) / self.half_w
            node_feats[j, 20] = 1.0 if j == self.half_w else 0.0
            if j < len(local_db):
                if local_db[j] == "(":
                    node_feats[j, 21] = 1.0
                elif local_db[j] == ")":
                    node_feats[j, 21] = -1.0

        edge_src, edge_dst, edge_type = [], [], []
        for j in range(n_nodes - 1):
            edge_src.extend([j, j + 1])
            edge_dst.extend([j + 1, j])
            edge_type.extend([0, 0])

        stack = []
        for j, c in enumerate(local_db):
            if c == "(":
                stack.append(j)
            elif c == ")" and stack:
                partner = stack.pop()
                pos_j = start + j
                pos_p = start + partner
                base_j = np.argmax(full_seq_arr[:, pos_j]) if 0 <= pos_j < 201 else -1
                base_p = np.argmax(full_seq_arr[:, pos_p]) if 0 <= pos_p < 201 else -1
                is_wobble = (base_j == 2 and base_p == 3) or (base_j == 3 and base_p == 2)
                etype = 2 if is_wobble else 1
                edge_src.extend([j, partner])
                edge_dst.extend([partner, j])
                edge_type.extend([etype, etype])

        if len(edge_src) == 0:
            edge_src = [self.half_w]
            edge_dst = [self.half_w]
            edge_type = [0]

        return {
            "node_feats": torch.from_numpy(node_feats),
            "edge_index": torch.tensor([edge_src, edge_dst], dtype=torch.long),
            "edge_type": torch.tensor(edge_type, dtype=torch.long),
            "onehot_seq": torch.from_numpy(self.data["onehot_seqs"][i]),
            "hand_feat": torch.from_numpy(hand_feat),
            "label_binary": torch.tensor(self.data["labels_binary"][i]),
            "label_enzyme": torch.tensor(self.data["labels_enzyme"][i]),
            "per_enzyme_labels": torch.from_numpy(self.data["per_enzyme_labels"][i]),
        }


# ---------------------------------------------------------------------------
# Per-Enzyme Multi-Head Mixin
# ---------------------------------------------------------------------------


class PerEnzymeHeads(nn.Module):
    """Per-enzyme binary heads + enzyme classifier on top of a 64-dim representation."""

    def __init__(self, d_repr=64, n_per_enzyme=N_PER_ENZYME, n_enzymes=N_ENZYMES, dropout=0.2):
        super().__init__()
        self.binary_head = nn.Sequential(
            nn.Linear(d_repr, 32), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        # Per-enzyme heads: one head per enzyme
        self.enzyme_binary_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_repr, 32), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(32, 1),
            )
            for _ in range(n_per_enzyme)
        ])
        # Enzyme classifier (6-class)
        self.enzyme_classifier = nn.Sequential(
            nn.Linear(d_repr, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, n_enzymes),
        )

    def forward(self, repr_64):
        """Returns (binary_logit, list_of_per_enzyme_logits, enzyme_class_logits)."""
        binary_logit = self.binary_head(repr_64).squeeze(-1)
        per_enzyme_logits = [head(repr_64).squeeze(-1) for head in self.enzyme_binary_heads]
        enzyme_cls_logits = self.enzyme_classifier(repr_64)
        return binary_logit, per_enzyme_logits, enzyme_cls_logits


# ---------------------------------------------------------------------------
# Model 1: Conv2D_BP with per-enzyme heads
# ---------------------------------------------------------------------------


class Conv2DBPModel(nn.Module):
    """Conv2D on base-pair probability matrix + Conv1D sequence + hand features."""

    def __init__(self):
        super().__init__()
        self.seq_conv = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=7, padding=3), nn.ReLU(), nn.BatchNorm1d(32), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.ReLU(), nn.BatchNorm1d(64), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm1d(128),
            nn.AdaptiveAvgPool1d(1),
        )
        self.seq_fc = nn.Linear(128, 128)

        self.bp_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2), nn.ReLU(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.AdaptiveMaxPool2d(1),
        )
        self.bp_fc = nn.Linear(64, 128)

        self.hand_fc = nn.Sequential(nn.Linear(40, 64), nn.ReLU())

        self.fusion = nn.Sequential(
            nn.LayerNorm(128 + 128 + 64),
            nn.Linear(128 + 128 + 64, 256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 64), nn.GELU(),
        )
        self.heads = PerEnzymeHeads(d_repr=64)

    def forward(self, batch):
        seq_out = self.seq_conv(batch["onehot_seq"]).squeeze(-1)
        seq_out = self.seq_fc(seq_out)
        bp_out = self.bp_conv(batch["bp_submatrix"]).squeeze(-1).squeeze(-1)
        bp_out = self.bp_fc(bp_out)
        hand_out = self.hand_fc(batch["hand_feat"])
        fused = self.fusion(torch.cat([seq_out, bp_out, hand_out], dim=-1))
        return self.heads(fused)


# ---------------------------------------------------------------------------
# Model 2: DualPath_Residual with per-enzyme heads
# ---------------------------------------------------------------------------


class DualPathResidual(nn.Module):
    """Dual-path model with residual connection preserving hand features."""

    def __init__(self):
        super().__init__()
        self.path_a = nn.Sequential(nn.Linear(40, 128), nn.GELU(), nn.Linear(128, 64))
        self.path_b_conv = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=7, padding=3), nn.ReLU(), nn.BatchNorm1d(32), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.ReLU(), nn.BatchNorm1d(64), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm1d(128),
            nn.AdaptiveAvgPool1d(1),
        )
        self.path_b_fc = nn.Linear(128, 64)
        self.fusion = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 64))
        self.heads = PerEnzymeHeads(d_repr=64)

    def forward(self, batch):
        path_a = self.path_a(batch["hand_feat"])
        path_b = self.path_b_conv(batch["onehot_seq"]).squeeze(-1)
        path_b = self.path_b_fc(path_b)
        fused = self.fusion(torch.cat([path_a, path_b], dim=-1)) + path_a
        return self.heads(fused)


# ---------------------------------------------------------------------------
# Model 3: AdarEdit GAT with per-enzyme heads
# ---------------------------------------------------------------------------


class ManualGATLayer(nn.Module):
    """Manual Graph Attention layer."""

    def __init__(self, in_dim, out_dim, n_heads=4, edge_embed_dim=16, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = out_dim // n_heads
        assert out_dim % n_heads == 0

        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = nn.Parameter(torch.randn(n_heads, self.head_dim))
        self.a_dst = nn.Parameter(torch.randn(n_heads, self.head_dim))
        self.a_edge = nn.Parameter(torch.randn(n_heads, edge_embed_dim))
        self.edge_embed = nn.Embedding(3, edge_embed_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim)

        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_normal_(self.a_src.unsqueeze(0))
        nn.init.xavier_normal_(self.a_dst.unsqueeze(0))

    def forward(self, x, edge_index, edge_type):
        N = x.size(0)
        h = self.W(x)
        h_heads = h.view(N, self.n_heads, self.head_dim)

        src, dst = edge_index[0], edge_index[1]

        h_src = h_heads[src]
        h_dst = h_heads[dst]
        e_embed = self.edge_embed(edge_type)

        attn = (h_src * self.a_src.unsqueeze(0)).sum(-1) + \
               (h_dst * self.a_dst.unsqueeze(0)).sum(-1) + \
               (e_embed.unsqueeze(1) * self.a_edge.unsqueeze(0)).sum(-1)
        attn = self.leaky_relu(attn)

        attn_max = torch.zeros(N, self.n_heads, device=x.device).fill_(-1e9)
        attn_max.scatter_reduce_(0, dst.unsqueeze(1).expand(-1, self.n_heads), attn, reduce="amax")
        attn = attn - attn_max[dst]
        attn_exp = torch.exp(attn)
        attn_sum = torch.zeros(N, self.n_heads, device=x.device)
        attn_sum.scatter_add_(0, dst.unsqueeze(1).expand(-1, self.n_heads), attn_exp)
        attn_norm = attn_exp / (attn_sum[dst] + 1e-10)
        attn_norm = self.dropout(attn_norm)

        weighted = h_src * attn_norm.unsqueeze(-1)
        out = torch.zeros(N, self.n_heads, self.head_dim, device=x.device)
        out.scatter_add_(0, dst.unsqueeze(1).unsqueeze(2).expand(-1, self.n_heads, self.head_dim), weighted)

        out = out.reshape(N, -1)
        out = self.norm(out + h)
        return out


class AdarEditGAT(nn.Module):
    """AdarEdit-adapted GAT with per-enzyme heads.

    Stream A: GAT on 41-node local RNA structure graph
    Stream B: Parallel 1D CNN (3-mer + 5-mer)
    Stream C: Hand features (40-dim)
    Fusion: [64 + 64 + 40] -> 64
    """

    def __init__(self, node_feat_dim=22, gat_hidden=64, n_heads=4, n_gat_layers=3,
                 cnn_channels=32, hand_dim=40):
        super().__init__()

        self.node_proj = nn.Linear(node_feat_dim, gat_hidden)
        self.gat_layers = nn.ModuleList()
        for _ in range(n_gat_layers):
            self.gat_layers.append(ManualGATLayer(gat_hidden, gat_hidden, n_heads=n_heads, dropout=0.1))
        self.gat_out = nn.Linear(gat_hidden, gat_hidden)

        self.cnn_k3 = nn.Sequential(
            nn.Conv1d(4, cnn_channels, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm1d(cnn_channels),
        )
        self.cnn_k5 = nn.Sequential(
            nn.Conv1d(4, cnn_channels, kernel_size=5, padding=2), nn.ReLU(), nn.BatchNorm1d(cnn_channels),
        )
        self.cnn_merge = nn.Sequential(
            nn.Conv1d(cnn_channels * 2, gat_hidden, kernel_size=3, padding=1), nn.ReLU(),
            nn.BatchNorm1d(gat_hidden), nn.AdaptiveMaxPool1d(1),
        )
        self.cnn_out = nn.Linear(gat_hidden, gat_hidden)

        fusion_dim = gat_hidden + gat_hidden + hand_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, gat_hidden), nn.GELU(), nn.Dropout(0.3),
        )
        self.heads = PerEnzymeHeads(d_repr=gat_hidden)

    def forward(self, batch):
        node_feats = batch["node_feats"]
        edge_index = batch["edge_index"]
        edge_type = batch["edge_type"]

        B = node_feats.size(0)
        n_nodes = node_feats.size(1)
        center_node = n_nodes // 2

        gat_embeddings = []
        for b in range(B):
            x = self.node_proj(node_feats[b])
            ei = edge_index[b].to(x.device)
            et = edge_type[b].to(x.device)
            for gat_layer in self.gat_layers:
                x = F.gelu(gat_layer(x, ei, et))
            gat_embeddings.append(x[center_node])

        gat_out = torch.stack(gat_embeddings)
        gat_out = self.gat_out(gat_out)

        seq = batch["onehot_seq"]
        cnn_3 = self.cnn_k3(seq)
        cnn_5 = self.cnn_k5(seq)
        cnn_cat = torch.cat([cnn_3, cnn_5], dim=1)
        cnn_pooled = self.cnn_merge(cnn_cat).squeeze(-1)
        cnn_out = self.cnn_out(cnn_pooled)

        hand_feat = batch["hand_feat"]

        fused = torch.cat([gat_out, cnn_out, hand_feat], dim=-1)
        fused = self.fusion(fused)
        return self.heads(fused)


# ---------------------------------------------------------------------------
# Model 4: Unified V1 (MLP on hand features + RNA-FM embeddings, NO RNA-FM for Phase 2b)
# The original Unified V1 uses RNA-FM pooled embeddings (640-dim) which require
# pre-computed embeddings. For fair comparison with other models that only use
# sequence + hand features, we replicate the architecture but use hand features only.
# This is the "hand features MLP" baseline with per-enzyme heads.
# ---------------------------------------------------------------------------


class UnifiedV1(nn.Module):
    """Unified V1: MLP on 40-dim hand features with per-enzyme heads.

    Replicates the architecture from exp_unified_network_v1.py but without
    RNA-FM embeddings (which other models don't use either).
    Input: 40-dim hand features -> shared MLP -> 64-dim -> per-enzyme heads.
    """

    def __init__(self, d_hand=40, dropout=0.3):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(d_hand, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Dropout(dropout),
        )
        self.heads = PerEnzymeHeads(d_repr=64)

    def forward(self, batch):
        hand = batch["hand_feat"]
        shared = self.shared(hand)
        return self.heads(shared)


# ---------------------------------------------------------------------------
# Collate functions
# ---------------------------------------------------------------------------


def collate_fn(batch_list):
    """Standard collate."""
    result = {}
    for key in batch_list[0]:
        result[key] = torch.stack([b[key] for b in batch_list])
    return result


def gat_collate_fn(batch_list):
    """Collate for GAT: keeps edge_index/edge_type as lists."""
    result = {}
    for key in batch_list[0]:
        if key in ("edge_index", "edge_type"):
            result[key] = [b[key] for b in batch_list]
        else:
            result[key] = torch.stack([b[key] for b in batch_list])
    return result


# ---------------------------------------------------------------------------
# Training with per-enzyme head loss
# ---------------------------------------------------------------------------


def compute_multi_head_loss(binary_logit, per_enzyme_logits, enzyme_cls_logits,
                            batch, use_focal=False, pos_weight_binary=None):
    """Compute combined loss: binary + per-enzyme heads + enzyme classifier.

    Total loss = binary_loss + 0.2 * sum(per_enzyme_losses) + 0.1 * enzyme_cls_loss
    """
    device = binary_logit.device

    # Binary loss (all samples)
    if use_focal:
        bce = F.binary_cross_entropy_with_logits(binary_logit, batch["label_binary"], reduction="none")
        probs = torch.sigmoid(binary_logit)
        pt = torch.where(batch["label_binary"] == 1, probs, 1 - probs)
        focal_weight = 0.25 * (1 - pt) ** 2
        loss_binary = (focal_weight * bce).mean()
    elif pos_weight_binary is not None:
        pw = torch.tensor([pos_weight_binary], device=device)
        loss_binary = F.binary_cross_entropy_with_logits(
            binary_logit, batch["label_binary"],
            pos_weight=pw.expand_as(binary_logit),
        )
    else:
        loss_binary = F.binary_cross_entropy_with_logits(binary_logit, batch["label_binary"])

    # Per-enzyme head losses
    per_enzyme_labels = batch["per_enzyme_labels"]  # [B, N_PER_ENZYME]
    loss_per_enzyme = torch.tensor(0.0, device=device)
    n_active_heads = 0

    for head_idx in range(N_PER_ENZYME):
        # Mask: only samples where label != -1 (i.e., relevant to this enzyme)
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
    pos_mask = batch["label_enzyme"] < N_ENZYMES  # all are valid (0..5)
    # But we only want positives for enzyme classification
    binary_pos = batch["label_binary"] == 1
    enzyme_mask = binary_pos
    loss_enzyme_cls = torch.tensor(0.0, device=device)
    if enzyme_mask.sum() > 0:
        loss_enzyme_cls = F.cross_entropy(
            enzyme_cls_logits[enzyme_mask],
            batch["label_enzyme"][enzyme_mask],
        )

    total_loss = loss_binary + ENZYME_HEAD_WEIGHT * loss_per_enzyme + ENZYME_CLS_WEIGHT * loss_enzyme_cls
    return total_loss


def train_one_epoch(model, loader, optimizer, is_gat=False, use_focal=False, pos_weight_binary=None):
    """Train one epoch with multi-head loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        if is_gat:
            batch_device = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch_device[k] = v.to(DEVICE)
                else:
                    batch_device[k] = v
            batch = batch_device
        else:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

        optimizer.zero_grad()
        binary_logit, per_enzyme_logits, enzyme_cls_logits = model(batch)
        loss = compute_multi_head_loss(
            binary_logit, per_enzyme_logits, enzyme_cls_logits, batch,
            use_focal=use_focal, pos_weight_binary=pos_weight_binary,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_model(model, loader, is_gat=False):
    """Evaluate: overall AUROC, per-enzyme AUROC, enzyme classification accuracy."""
    model.eval()
    all_binary_probs = []
    all_per_enzyme_probs = [[] for _ in range(N_PER_ENZYME)]
    all_enzyme_cls = []
    all_lbl_binary = []
    all_lbl_enzyme = []
    all_per_enzyme_labels = []

    for batch in loader:
        if is_gat:
            batch_device = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch_device[k] = v.to(DEVICE)
                else:
                    batch_device[k] = v
            batch = batch_device
        else:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

        binary_logit, per_enzyme_logits, enzyme_cls_logits = model(batch)

        all_binary_probs.append(torch.sigmoid(binary_logit).cpu().numpy())
        for h in range(N_PER_ENZYME):
            all_per_enzyme_probs[h].append(torch.sigmoid(per_enzyme_logits[h]).cpu().numpy())
        all_enzyme_cls.append(enzyme_cls_logits.cpu().numpy())
        all_lbl_binary.append(batch["label_binary"].cpu().numpy())
        all_lbl_enzyme.append(batch["label_enzyme"].cpu().numpy())
        all_per_enzyme_labels.append(batch["per_enzyme_labels"].cpu().numpy())

    binary_probs = np.concatenate(all_binary_probs)
    per_enzyme_probs = [np.concatenate(p) for p in all_per_enzyme_probs]
    enzyme_cls_logits = np.concatenate(all_enzyme_cls)
    lbl_binary = np.concatenate(all_lbl_binary)
    lbl_enzyme = np.concatenate(all_lbl_enzyme)
    per_enzyme_labels = np.concatenate(all_per_enzyme_labels)

    # Overall binary AUROC
    try:
        overall_auroc = roc_auc_score(lbl_binary, binary_probs)
    except ValueError:
        overall_auroc = 0.5

    # Per-enzyme AUROC (using per-enzyme head outputs)
    per_enzyme_auroc = {}
    for head_idx, enz_name in enumerate(PER_ENZYME_HEADS):
        mask = per_enzyme_labels[:, head_idx] >= 0
        if mask.sum() < 2:
            continue
        labels = per_enzyme_labels[:, head_idx][mask]
        probs = per_enzyme_probs[head_idx][mask]
        if len(np.unique(labels)) < 2:
            continue
        try:
            per_enzyme_auroc[enz_name] = float(roc_auc_score(labels, probs))
        except ValueError:
            pass

    # Also compute per-enzyme AUROC using the BINARY head (for comparison)
    # This is the "unified binary" evaluation: enzyme-X positives vs all negatives, scored by binary head
    per_enzyme_auroc_binary = {}
    for enz_name in PER_ENZYME_HEADS:
        enz_idx = ENZYME_TO_IDX[enz_name]
        # Mask: positives of this enzyme + all negatives
        enz_pos = (lbl_binary == 1) & (lbl_enzyme == enz_idx)
        neg = lbl_binary == 0
        mask = enz_pos | neg
        if mask.sum() < 2:
            continue
        labels_sub = lbl_binary[mask]
        probs_sub = binary_probs[mask]
        if len(np.unique(labels_sub)) < 2:
            continue
        try:
            per_enzyme_auroc_binary[enz_name] = float(roc_auc_score(labels_sub, probs_sub))
        except ValueError:
            pass

    # Enzyme classification accuracy (positives only)
    pos_mask = lbl_binary == 1
    if pos_mask.sum() > 0:
        enzyme_acc = accuracy_score(lbl_enzyme[pos_mask], enzyme_cls_logits[pos_mask].argmax(axis=1))
    else:
        enzyme_acc = 0.0

    return {
        "overall_auroc": float(overall_auroc),
        "per_enzyme_auroc": per_enzyme_auroc,
        "per_enzyme_auroc_binary": per_enzyme_auroc_binary,
        "enzyme_accuracy": float(enzyme_acc),
    }


# ---------------------------------------------------------------------------
# Model creation
# ---------------------------------------------------------------------------


def create_model(model_name):
    if model_name == "Conv2D_BP":
        return Conv2DBPModel()
    elif model_name == "DualPath_Residual":
        return DualPathResidual()
    elif model_name == "AdarEdit_GAT":
        return AdarEditGAT()
    elif model_name == "Unified_V1":
        return UnifiedV1()
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ---------------------------------------------------------------------------
# Training orchestrator
# ---------------------------------------------------------------------------


def run_model_cv(model_name, data, dataset_label, batch_size, n_epochs, lr,
                 n_folds=N_FOLDS, use_focal=False, pos_weight_binary=None):
    """Run n_folds CV for a model. Returns results dict."""
    is_gat = model_name == "AdarEdit_GAT"
    logger.info(f"\n{'='*60}")
    logger.info(f"Training {model_name} on {dataset_label} ({n_folds} folds)")
    logger.info(f"{'='*60}")

    n = len(data["site_ids"])
    skf = StratifiedKFold(n_splits=max(n_folds, 2), shuffle=True, random_state=SEED)
    folds = list(skf.split(np.arange(n), data["labels_binary"]))[:n_folds]

    fold_results = []
    fold_histories = []
    n_params = 0

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        torch.manual_seed(SEED + fold_idx)
        logger.info(f"\n--- {model_name} Fold {fold_idx} ---")
        t0 = time.time()

        model = create_model(model_name).to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if fold_idx == 0:
            logger.info(f"  Model parameters: {n_params:,}")

        n_loader_workers = 0 if DEVICE.type == "mps" else 4
        if is_gat:
            train_ds = GATDataset(list(train_idx), data)
            val_ds = GATDataset(list(val_idx), data)
            cfn = gat_collate_fn
        else:
            train_ds = EditingDataset(list(train_idx), data)
            val_ds = EditingDataset(list(val_idx), data)
            cfn = collate_fn

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  collate_fn=cfn, num_workers=n_loader_workers, pin_memory=False)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                collate_fn=cfn, num_workers=n_loader_workers, pin_memory=False)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=V3_WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

        history = {"train_loss": [], "val_auroc": []}
        best_auroc = 0.0
        best_state = None

        for epoch in range(n_epochs):
            loss = train_one_epoch(model, train_loader, optimizer, is_gat=is_gat,
                                   use_focal=use_focal, pos_weight_binary=pos_weight_binary)
            scheduler.step()

            metrics = evaluate_model(model, val_loader, is_gat=is_gat)
            history["train_loss"].append(loss)
            history["val_auroc"].append(metrics["overall_auroc"])

            if metrics["overall_auroc"] > best_auroc:
                best_auroc = metrics["overall_auroc"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if (epoch + 1) % 10 == 0:
                enz_str = ", ".join(f"{e}={metrics['per_enzyme_auroc'].get(e, 0):.3f}"
                                    for e in PER_ENZYME_HEADS)
                logger.info(
                    f"  [{model_name}] Fold {fold_idx} Epoch {epoch+1}/{n_epochs}: "
                    f"loss={loss:.4f}, overall={metrics['overall_auroc']:.4f}, "
                    f"enz_acc={metrics['enzyme_accuracy']:.3f}, {enz_str}"
                )

        # Load best model and final eval
        if best_state is not None:
            model.load_state_dict(best_state)
        final = evaluate_model(model, val_loader, is_gat=is_gat)

        elapsed = time.time() - t0
        logger.info(f"\n  Fold {fold_idx} done in {elapsed:.0f}s")
        logger.info(f"  Overall AUROC: {final['overall_auroc']:.4f}")
        logger.info(f"  Enzyme accuracy: {final['enzyme_accuracy']:.3f}")
        logger.info(f"  Per-enzyme AUROC (dedicated heads):")
        for enz in PER_ENZYME_HEADS:
            val = final["per_enzyme_auroc"].get(enz, 0)
            val_bin = final["per_enzyme_auroc_binary"].get(enz, 0)
            logger.info(f"    {enz}: head={val:.3f}, binary={val_bin:.3f}")

        fold_results.append(final)
        fold_histories.append(history)

        # Save model
        save_name = f"phase2b_{model_name}_{dataset_label}_fold{fold_idx}.pt"
        torch.save(model.state_dict(), OUTPUT_DIR / save_name)

        del model
        if DEVICE.type == "mps":
            torch.mps.empty_cache()
        gc.collect()

    # Aggregate
    mean_overall = np.mean([r["overall_auroc"] for r in fold_results])
    std_overall = np.std([r["overall_auroc"] for r in fold_results])
    mean_enz_acc = np.mean([r["enzyme_accuracy"] for r in fold_results])

    per_enzyme_means = {}
    per_enzyme_binary_means = {}
    for enz in PER_ENZYME_HEADS:
        vals = [r["per_enzyme_auroc"].get(enz, 0) for r in fold_results]
        per_enzyme_means[enz] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        vals_bin = [r["per_enzyme_auroc_binary"].get(enz, 0) for r in fold_results]
        per_enzyme_binary_means[enz] = {"mean": float(np.mean(vals_bin)), "std": float(np.std(vals_bin))}

    result = {
        "mean_overall_auroc": float(mean_overall),
        "std_overall_auroc": float(std_overall),
        "mean_enzyme_accuracy": float(mean_enz_acc),
        "per_enzyme_auroc": per_enzyme_means,
        "per_enzyme_auroc_binary": per_enzyme_binary_means,
        "fold_results": fold_results,
        "n_params": n_params,
        "n_folds": n_folds,
    }

    logger.info(f"\n{model_name} on {dataset_label} SUMMARY:")
    logger.info(f"  Overall AUROC: {mean_overall:.4f} +/- {std_overall:.4f}")
    logger.info(f"  Enzyme acc: {mean_enz_acc:.3f}")
    logger.info(f"  Per-enzyme AUROC (dedicated head / binary head):")
    for enz in PER_ENZYME_HEADS:
        h = per_enzyme_means[enz]["mean"]
        b = per_enzyme_binary_means[enz]["mean"]
        logger.info(f"    {enz}: {h:.3f} / {b:.3f}")

    return result, fold_histories


# ---------------------------------------------------------------------------
# XGBoost per-enzyme baseline
# ---------------------------------------------------------------------------


def train_xgboost_per_enzyme(data, dataset_label="v3"):
    """Train per-enzyme XGBoost classifiers for reference."""
    from xgboost import XGBClassifier

    logger.info(f"\n{'='*60}")
    logger.info(f"XGBoost per-enzyme on {dataset_label}")
    logger.info(f"{'='*60}")

    X = data["hand_features"]
    y = data["labels_binary"]
    enzyme_labels = data["labels_enzyme"]
    per_enzyme_labels = data["per_enzyme_labels"]
    n = len(y)

    skf = StratifiedKFold(n_splits=max(N_FOLDS, 2), shuffle=True, random_state=SEED)
    folds = list(skf.split(np.arange(n), y))[:N_FOLDS]

    # Overall binary
    overall_aurocs = []
    per_enzyme_aurocs = {enz: [] for enz in PER_ENZYME_HEADS}

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Overall binary classifier
        n_pos = (y_train == 1).sum()
        n_neg = (y_train == 0).sum()
        spw = float(n_neg / n_pos) if n_pos > 0 else 1.0

        clf = XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=spw, eval_metric="logloss",
            use_label_encoder=False, random_state=SEED,
            tree_method="hist", n_jobs=-1,
        )
        clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        probs = clf.predict_proba(X_val)[:, 1]
        try:
            overall_aurocs.append(roc_auc_score(y_val, probs))
        except ValueError:
            overall_aurocs.append(0.5)

        # Per-enzyme classifiers
        for head_idx, enz_name in enumerate(PER_ENZYME_HEADS):
            pe_labels = per_enzyme_labels[:, head_idx]
            train_mask = pe_labels[train_idx] >= 0
            val_mask = pe_labels[val_idx] >= 0

            if train_mask.sum() < 10 or val_mask.sum() < 10:
                per_enzyme_aurocs[enz_name].append(0.5)
                continue

            X_tr = X_train[train_mask]
            y_tr = pe_labels[train_idx][train_mask]
            X_vl = X_val[val_mask]
            y_vl = pe_labels[val_idx][val_mask]

            n_p = (y_tr == 1).sum()
            n_n = (y_tr == 0).sum()
            spw_e = float(n_n / n_p) if n_p > 0 else 1.0

            clf_e = XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=spw_e, eval_metric="logloss",
                use_label_encoder=False, random_state=SEED,
                tree_method="hist", n_jobs=-1,
            )
            clf_e.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)
            probs_e = clf_e.predict_proba(X_vl)[:, 1]
            try:
                per_enzyme_aurocs[enz_name].append(roc_auc_score(y_vl, probs_e))
            except ValueError:
                per_enzyme_aurocs[enz_name].append(0.5)

        logger.info(f"  XGBoost Fold {fold_idx}: overall={overall_aurocs[-1]:.4f}, "
                     + ", ".join(f"{e}={per_enzyme_aurocs[e][-1]:.3f}" for e in PER_ENZYME_HEADS))

    result = {
        "mean_overall_auroc": float(np.mean(overall_aurocs)),
        "std_overall_auroc": float(np.std(overall_aurocs)),
        "per_enzyme_auroc": {
            enz: {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
            for enz, vals in per_enzyme_aurocs.items()
        },
        "n_params": "N/A (tree-based)",
        "n_folds": N_FOLDS,
    }

    logger.info(f"\nXGBoost {dataset_label} SUMMARY:")
    logger.info(f"  Overall AUROC: {result['mean_overall_auroc']:.4f} +/- {result['std_overall_auroc']:.4f}")
    for enz in PER_ENZYME_HEADS:
        m = result["per_enzyme_auroc"][enz]["mean"]
        s = result["per_enzyme_auroc"][enz]["std"]
        logger.info(f"  {enz}: {m:.3f} +/- {s:.3f}")

    return result


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def print_summary_table(all_results):
    """Print a comprehensive comparison table."""
    logger.info("\n" + "=" * 120)
    logger.info("PHASE 2B: COMPREHENSIVE RESULTS")
    logger.info("=" * 120)

    # Header
    header = f"{'Model':<30} {'Overall':>8} "
    for enz in PER_ENZYME_HEADS:
        header += f" {enz:>8}"
    header += f" {'EnzAcc':>8} {'Params':>10}"
    logger.info(header)
    logger.info("-" * 120)

    # Reference: XGBoost per-enzyme from CLAUDE.md
    ref_line = f"{'XGB Ref (per-enzyme)':<30} {'---':>8} "
    ref_vals = {"A3A": 0.923, "A3B": 0.831, "A3G": 0.929, "A3A_A3G": 0.941, "Neither": 0.840}
    for enz in PER_ENZYME_HEADS:
        ref_line += f" {ref_vals.get(enz, 0):8.3f}"
    ref_line += f" {'---':>8} {'---':>10}"
    logger.info(ref_line)

    # Unified V1 reference from CLAUDE.md
    uv1_line = f"{'Unified V1 Ref':<30} {'---':>8} "
    uv1_vals = {"A3A": 0.900, "A3B": 0.914, "A3G": 0.952, "A3A_A3G": 0.0, "Neither": 0.0}
    for enz in PER_ENZYME_HEADS:
        uv1_line += f" {uv1_vals.get(enz, 0):8.3f}"
    uv1_line += f" {'---':>8} {'---':>10}"
    logger.info(uv1_line)
    logger.info("-" * 120)

    for key, res in all_results.items():
        overall = res.get("mean_overall_auroc", 0)
        line = f"{key:<30} {overall:8.4f} "
        pe = res.get("per_enzyme_auroc", {})
        for enz in PER_ENZYME_HEADS:
            val = pe.get(enz, {})
            if isinstance(val, dict):
                m = val.get("mean", 0)
            else:
                m = val
            line += f" {m:8.3f}"
        enz_acc = res.get("mean_enzyme_accuracy", 0)
        n_params = res.get("n_params", "?")
        if isinstance(n_params, int):
            line += f" {enz_acc:8.3f} {n_params:>10,}"
        else:
            line += f" {enz_acc:8.3f} {str(n_params):>10}"
        logger.info(line)

    logger.info("=" * 120)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    t_start = time.time()
    all_results = {}
    all_histories = {}

    # =====================================================================
    # Load v3 data
    # =====================================================================
    v3_data = load_v3_data()

    # =====================================================================
    # 1. XGBoost per-enzyme baseline on v3
    # =====================================================================
    xgb_result = train_xgboost_per_enzyme(v3_data, "v3")
    all_results["XGBoost_v3"] = xgb_result

    # =====================================================================
    # 2. Unified V1 (hand features MLP) on v3
    # =====================================================================
    result, histories = run_model_cv(
        "Unified_V1", v3_data, "v3",
        batch_size=V3_BATCH_SIZE, n_epochs=V3_N_EPOCHS, lr=V3_LR,
    )
    all_results["Unified_V1_v3"] = result
    all_histories["Unified_V1_v3"] = histories

    # =====================================================================
    # 3. Conv2D_BP on v3
    # =====================================================================
    result, histories = run_model_cv(
        "Conv2D_BP", v3_data, "v3",
        batch_size=V3_BATCH_SIZE, n_epochs=V3_N_EPOCHS, lr=V3_LR,
    )
    all_results["Conv2D_BP_v3"] = result
    all_histories["Conv2D_BP_v3"] = histories

    # =====================================================================
    # 4. DualPath_Residual on v3
    # =====================================================================
    result, histories = run_model_cv(
        "DualPath_Residual", v3_data, "v3",
        batch_size=V3_BATCH_SIZE, n_epochs=V3_N_EPOCHS, lr=V3_LR,
    )
    all_results["DualPath_Residual_v3"] = result
    all_histories["DualPath_Residual_v3"] = histories

    # =====================================================================
    # 5. AdarEdit GAT on v3
    # =====================================================================
    result, histories = run_model_cv(
        "AdarEdit_GAT", v3_data, "v3",
        batch_size=V3_BATCH_SIZE, n_epochs=V3_N_EPOCHS, lr=V3_LR,
    )
    all_results["AdarEdit_GAT_v3"] = result
    all_histories["AdarEdit_GAT_v3"] = histories

    # Free v3 data
    del v3_data
    gc.collect()

    # =====================================================================
    # 6. Find best v3 model and retrain on v4-large
    # =====================================================================
    logger.info("\n" + "=" * 70)
    logger.info("Selecting best v3 model for v4-large retraining...")
    logger.info("=" * 70)

    v3_models = {k: v for k, v in all_results.items() if k.endswith("_v3") and "XGBoost" not in k}
    best_v3_name = max(v3_models, key=lambda k: v3_models[k]["mean_overall_auroc"])
    best_v3_model = best_v3_name.replace("_v3", "")
    logger.info(f"Best v3 model: {best_v3_name} (AUROC={v3_models[best_v3_name]['mean_overall_auroc']:.4f})")

    v4_data = load_v4_data()

    # Compute pos_weight for v4 class imbalance
    n_pos = (v4_data["labels_binary"] == 1).sum()
    n_neg = (v4_data["labels_binary"] == 0).sum()
    pos_weight_v4 = float(n_neg / n_pos) if n_pos > 0 else 1.0
    logger.info(f"v4-large: {len(v4_data['site_ids'])} sites, pos_weight={pos_weight_v4:.2f}")

    result, histories = run_model_cv(
        best_v3_model, v4_data, "v4large",
        batch_size=V4_BATCH_SIZE, n_epochs=V4_N_EPOCHS, lr=V4_LR,
        use_focal=True,
    )
    all_results[f"{best_v3_model}_v4large"] = result
    all_histories[f"{best_v3_model}_v4large"] = histories

    # Also XGBoost on v4-large
    xgb_v4 = train_xgboost_per_enzyme(v4_data, "v4large")
    all_results["XGBoost_v4large"] = xgb_v4

    del v4_data
    gc.collect()

    # =====================================================================
    # Summary
    # =====================================================================
    print_summary_table(all_results)

    # Save results
    # Convert numpy types for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(OUTPUT_DIR / "phase2b_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=convert_for_json)
    logger.info(f"\nSaved results to {OUTPUT_DIR / 'phase2b_results.json'}")

    elapsed = time.time() - t_start
    logger.info(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    logger.info("Phase 2b complete.")


if __name__ == "__main__":
    main()
