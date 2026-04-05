#!/usr/bin/env python
"""Final architecture comparison: Phase 3 + Conv2D, Phase 3 + GAT, v4 datasets, hard-neg curriculum.

Experiments:
  1. Phase 3 + Conv2D BP probability matrix (1448-dim fusion)
  2. Phase 3 + AdarEdit GAT (1384-dim fusion)
  3. Best neural winner on v4 datasets (hand features only, 40-dim)
  4. Hard negative curriculum (P7) on best architecture
  5. Comprehensive comparison table

All experiments use 2-fold CV for speed. Compares to XGBoost and Phase 3 baselines.

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

log_file = LOG_DIR / "final_comparison.log"
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
D_GAT_OUT = 64

# XGBoost baselines (from existing results)
XGBOOST_BASELINES = {
    "A3A": 0.907,
    "A3B": 0.831,
    "A3G": 0.931,
    "A3A_A3G": 0.941,
    "Neither": 0.840,
}
# Phase 3 baselines (from phase3_results.json)
PHASE3_BASELINES = {
    "A3A": 0.877,
    "A3B": 0.808,
    "A3G": 0.909,
    "A3A_A3G": 0.972,
    "Neither": 0.872,
}

# GAT hyperparameters
GAT_HIDDEN = 64
GAT_HEADS = 4
GAT_LAYERS = 3


# ---------------------------------------------------------------------------
# ViennaRNA folding (parallel) -- for BP probability submatrix
# ---------------------------------------------------------------------------


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
        # Extract 41x41 submatrix centered on edit site
        sub = bpp[80:121, 80:121].copy()
        # Zero out diagonal band (|i-j| < 3)
        for r in range(41):
            for c in range(41):
                if abs(r - c) < 3:
                    sub[r, c] = 0.0
        return site_id, structure, sub
    except Exception:
        return site_id, "." * len(seq), np.zeros((41, 41), dtype=np.float32)


def _fold_worker(args):
    """Worker for MFE folding only (no BPP): returns (site_id, dot_bracket)."""
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


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------


def load_v3_data() -> Dict:
    """Load v3 dataset with all pre-computed features (RNA-FM, BPP, hand features)."""
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
    db_lookup = {}
    for _, row in loop_df.iterrows():
        if pd.notna(row.get("dot_bracket")):
            db_lookup[str(row["site_id"])] = row["dot_bracket"]
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

    del rnafm_emb, rnafm_edited_emb
    gc.collect()

    # --- Compute Conv2D BP submatrices and dot-brackets via ViennaRNA ---
    logger.info("Computing ViennaRNA BP probability submatrices...")
    work_items = [(sid, sequences[sid]) for sid in site_ids]
    n_workers = min(14, os.cpu_count() or 4)
    bp_submatrices = np.zeros((n, 41, 41), dtype=np.float32)
    dot_brackets = {}
    sid_to_idx = {sid: i for i, sid in enumerate(site_ids)}

    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_fold_and_bpp_worker, item): item[0] for item in work_items}
        for future in as_completed(futures):
            sid, structure, sub = future.result()
            idx = sid_to_idx[sid]
            bp_submatrices[idx] = sub
            dot_brackets[sid] = db_lookup.get(sid, structure)
            done += 1
            if done % 3000 == 0:
                logger.info(f"    Folded {done}/{n} ({time.time()-t0:.0f}s)")
    logger.info(f"  Folding complete: {done} sequences in {time.time()-t0:.0f}s")

    # --- Build dot_bracket_list for GAT ---
    dot_bracket_list = [dot_brackets.get(sid, "." * 201) for sid in site_ids]

    # --- One-hot encode sequences ---
    base_map = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3}
    onehot_seqs = np.zeros((n, 4, 201), dtype=np.float32)
    for i, sid in enumerate(site_ids):
        seq = sequences[sid].upper().replace("T", "U")
        for j, base in enumerate(seq[:201]):
            if base in base_map:
                onehot_seqs[i, base_map[base], j] = 1.0

    # --- Labels ---
    labels_binary = df["is_edited"].values.astype(np.float32)
    labels_enzyme = np.array([ENZYME_TO_IDX.get(e, 5) for e in df["enzyme"].values], dtype=np.int64)

    # --- Per-enzyme labels ---
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

    logger.info("V3 data loading complete.")
    return {
        "site_ids": site_ids,
        "df": df,
        "hand_features": hand_features,
        "rnafm_features": rnafm_matrix,
        "edit_delta_features": edit_delta_matrix,
        "bp_submatrices": bp_submatrices,
        "dot_bracket_list": dot_bracket_list,
        "onehot_seqs": onehot_seqs,
        "labels_binary": labels_binary,
        "labels_enzyme": labels_enzyme,
        "per_enzyme_labels": per_enzyme_labels,
        "sequences": sequences,
    }


def load_v4_data(variant: str) -> Dict:
    """Load v4 dataset (hand features only, no RNA-FM).

    Args:
        variant: one of 'random', 'hard', 'large'
    """
    logger.info("=" * 60)
    logger.info(f"Loading V4-{variant} dataset (hand features only)")
    logger.info("=" * 60)

    splits_path = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / f"splits_v4_{variant}_negatives.csv"
    df = pd.read_csv(splits_path, low_memory=False)
    logger.info(f"  Loaded {len(df)} sites")
    logger.info(f"  Positives: {(df['is_edited']==1).sum()}, Negatives: {(df['is_edited']==0).sum()}")

    seq_path = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / f"multi_enzyme_sequences_v4_{variant}.json"
    with open(seq_path) as f:
        sequences = json.load(f)
    logger.info(f"  Loaded {len(sequences)} sequences")

    # Load v3 loop and structure data (covers positives)
    loop_path = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "loop_position_per_site_v3.csv"
    loop_df = pd.read_csv(loop_path)
    loop_df = loop_df.drop_duplicates(subset=["site_id"]).set_index("site_id")

    struct_path = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "structure_cache_multi_enzyme_v3.npz"
    struct_data = np.load(struct_path, allow_pickle=True)
    struct_ids = list(struct_data["site_ids"])
    struct_deltas = struct_data["delta_features"]
    structure_delta = {sid: struct_deltas[i] for i, sid in enumerate(struct_ids)}

    # Filter to sites with sequences
    site_ids = df["site_id"].tolist()
    valid_mask = [sid in sequences for sid in site_ids]
    df = df[valid_mask].reset_index(drop=True)
    site_ids = df["site_id"].tolist()
    n = len(site_ids)
    logger.info(f"  {n} sites with sequences")

    # Build hand features (40-dim)
    hand_features = build_hand_features(site_ids, sequences, structure_delta, loop_df)
    logger.info(f"  Hand features shape: {hand_features.shape}")

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

    logger.info(f"V4-{variant} data loading complete.")
    return {
        "site_ids": site_ids,
        "df": df,
        "hand_features": hand_features,
        "labels_binary": labels_binary,
        "labels_enzyme": labels_enzyme,
        "per_enzyme_labels": per_enzyme_labels,
        "sequences": sequences,
    }


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


class Phase3Dataset(Dataset):
    """Dataset for Phase 3 models with RNA-FM + Conv2D + hand features."""

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


class GATPhase3Dataset(Dataset):
    """Dataset for Phase 3 + GAT model. Builds graph per sample from dot-bracket."""

    def __init__(self, indices, data, window=41):
        self.indices = indices
        self.data = data
        self.window = window
        self.half_w = window // 2

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        seq_onehot = self.data["onehot_seqs"][i]  # [4, 201]
        db = self.data["dot_bracket_list"][i]

        # Extract local window around edit site
        start = CENTER - self.half_w
        end = CENTER + self.half_w + 1
        local_db = db[start:end]

        # Build node features (22-dim per node)
        n_nodes = self.window
        node_feats = np.zeros((n_nodes, 22), dtype=np.float32)

        for j in range(n_nodes):
            pos = start + j
            # Base identity (4)
            if 0 <= pos < 201:
                node_feats[j, :4] = seq_onehot[:, pos]
            # Trinucleotide left context (4)
            if 0 <= pos - 1 < 201:
                node_feats[j, 4:8] = seq_onehot[:, pos - 1]
            # Trinucleotide right context (4)
            if 0 <= pos + 1 < 201:
                node_feats[j, 8:12] = seq_onehot[:, pos + 1]
            # Is unpaired
            if j < len(local_db):
                node_feats[j, 12] = 1.0 if local_db[j] == "." else 0.0
            # Local unpaired fraction (window of 5)
            w5_start = max(0, j - 2)
            w5_end = min(len(local_db), j + 3)
            local_region = local_db[w5_start:w5_end]
            node_feats[j, 13] = sum(1 for c in local_region if c == ".") / max(len(local_region), 1)
            # Sinusoidal positional encoding (4-dim)
            p = j / n_nodes
            node_feats[j, 14] = np.sin(p * np.pi)
            node_feats[j, 15] = np.cos(p * np.pi)
            node_feats[j, 16] = np.sin(2 * p * np.pi)
            node_feats[j, 17] = np.cos(2 * p * np.pi)
            # Pairing energy proxy
            node_feats[j, 18] = 0.0 if (j < len(local_db) and local_db[j] == ".") else 1.0
            # Distance from center (normalized)
            node_feats[j, 19] = abs(j - self.half_w) / self.half_w
            # Is center
            node_feats[j, 20] = 1.0 if j == self.half_w else 0.0
            # Structure channel
            if j < len(local_db):
                if local_db[j] == "(":
                    node_feats[j, 21] = 1.0
                elif local_db[j] == ")":
                    node_feats[j, 21] = -1.0

        # Build edges: sequential + base-pair
        edge_src, edge_dst, edge_type = [], [], []

        # Sequential edges (type 0)
        for j in range(n_nodes - 1):
            edge_src.extend([j, j + 1])
            edge_dst.extend([j + 1, j])
            edge_type.extend([0, 0])

        # Base-pair edges from dot-bracket
        stack = []
        for j, c in enumerate(local_db):
            if c == "(":
                stack.append(j)
            elif c == ")" and stack:
                partner = stack.pop()
                pos_j = start + j
                pos_p = start + partner
                base_j = np.argmax(seq_onehot[:, pos_j]) if 0 <= pos_j < 201 else -1
                base_p = np.argmax(seq_onehot[:, pos_p]) if 0 <= pos_p < 201 else -1
                is_wobble = (base_j == 2 and base_p == 3) or (base_j == 3 and base_p == 2)
                etype = 2 if is_wobble else 1
                edge_src.extend([j, partner])
                edge_dst.extend([partner, j])
                edge_type.extend([etype, etype])

        if len(edge_src) == 0:
            edge_src = [self.half_w]
            edge_dst = [self.half_w]
            edge_type = [0]

        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        edge_type_tensor = torch.tensor(edge_type, dtype=torch.long)

        return {
            "rnafm": torch.from_numpy(self.data["rnafm_features"][i]),
            "edit_delta": torch.from_numpy(self.data["edit_delta_features"][i]),
            "hand_feat": torch.from_numpy(self.data["hand_features"][i]),
            "node_feats": torch.from_numpy(node_feats),        # [41, 22]
            "edge_index": edge_index,                            # [2, E]
            "edge_type": edge_type_tensor,                       # [E]
            "label_binary": torch.tensor(self.data["labels_binary"][i]),
            "label_enzyme": torch.tensor(self.data["labels_enzyme"][i]),
            "per_enzyme_labels": torch.from_numpy(self.data["per_enzyme_labels"][i]),
        }


class HandFeatOnlyDataset(Dataset):
    """Dataset for v4 experiments: hand features only (40-dim)."""

    def __init__(self, indices, data):
        self.indices = indices
        self.data = data

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        return {
            "hand_feat": torch.from_numpy(self.data["hand_features"][i]),
            "label_binary": torch.tensor(self.data["labels_binary"][i]),
            "label_enzyme": torch.tensor(self.data["labels_enzyme"][i]),
            "per_enzyme_labels": torch.from_numpy(self.data["per_enzyme_labels"][i]),
        }


# ---------------------------------------------------------------------------
# Collate functions
# ---------------------------------------------------------------------------


def collate_fn(batch_list):
    """Standard collate: stack all tensors."""
    result = {}
    for key in batch_list[0]:
        result[key] = torch.stack([b[key] for b in batch_list])
    return result


def gat_collate_fn(batch_list):
    """Collate for GAT: keeps edge_index/edge_type as lists (variable size)."""
    result = {}
    for key in batch_list[0]:
        if key in ("edge_index", "edge_type"):
            result[key] = [b[key] for b in batch_list]
        else:
            result[key] = torch.stack([b[key] for b in batch_list])
    return result


# ---------------------------------------------------------------------------
# Model: Conv2D Encoder
# ---------------------------------------------------------------------------


class Conv2DEncoder(nn.Module):
    """Conv2D encoder on 41x41 base-pair probability submatrix -> 128-dim."""

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
        """bp_sub: [B, 1, 41, 41] -> [B, 128]"""
        x = self.conv(bp_sub).squeeze(-1).squeeze(-1)  # [B, 64]
        return self.fc(x)  # [B, 128]


# ---------------------------------------------------------------------------
# Model: ManualGATLayer (from Phase 2)
# ---------------------------------------------------------------------------


class ManualGATLayer(nn.Module):
    """Manual Graph Attention layer (no torch_geometric dependency)."""

    def __init__(self, in_dim, out_dim, n_heads=4, edge_embed_dim=16, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = out_dim // n_heads
        assert out_dim % n_heads == 0

        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = nn.Parameter(torch.randn(n_heads, self.head_dim))
        self.a_dst = nn.Parameter(torch.randn(n_heads, self.head_dim))
        self.a_edge = nn.Parameter(torch.randn(n_heads, edge_embed_dim))
        self.edge_embed = nn.Embedding(3, edge_embed_dim)  # 3 edge types
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim)

        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_normal_(self.a_src.unsqueeze(0))
        nn.init.xavier_normal_(self.a_dst.unsqueeze(0))

    def forward(self, x, edge_index, edge_type):
        """x: [N, in_dim], edge_index: [2, E], edge_type: [E] -> [N, out_dim]"""
        N = x.size(0)
        h = self.W(x)  # [N, out_dim]
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


# ---------------------------------------------------------------------------
# Model: Phase 3 + Conv2D (Experiment 1)
# ---------------------------------------------------------------------------


class Phase3Conv2DModel(nn.Module):
    """Phase 3 with Conv2D BP probability encoder.

    Fusion: RNA-FM(640) + Conv2D(128) + EditDelta(640) + Hand(40) = 1448-dim
    """

    def __init__(self):
        super().__init__()
        d_fusion = D_RNAFM + D_CONV2D_OUT + D_EDIT_DELTA + D_HAND  # 1448

        self.conv2d_encoder = Conv2DEncoder()

        self.shared_encoder = nn.Sequential(
            nn.Linear(d_fusion, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        self.binary_head = nn.Linear(128, 1)
        self.enzyme_adapters = nn.ModuleDict({
            enz: nn.Sequential(nn.Linear(128, 32), nn.GELU(), nn.Linear(32, 1))
            for enz in PER_ENZYME_HEADS
        })
        self.enzyme_classifier = nn.Sequential(
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, N_ENZYMES),
        )

    def get_shared_params(self):
        return list(self.conv2d_encoder.parameters()) + list(self.shared_encoder.parameters())

    def get_adapter_params(self):
        params = list(self.binary_head.parameters())
        for enz in PER_ENZYME_HEADS:
            params.extend(self.enzyme_adapters[enz].parameters())
        params.extend(self.enzyme_classifier.parameters())
        return params

    def forward(self, batch):
        rnafm = batch["rnafm"]
        conv2d_out = self.conv2d_encoder(batch["bp_submatrix"])
        edit_delta = batch["edit_delta"]
        hand = batch["hand_feat"]

        fused = torch.cat([rnafm, conv2d_out, edit_delta, hand], dim=-1)
        shared = self.shared_encoder(fused)

        binary_logit = self.binary_head(shared).squeeze(-1)
        per_enzyme_logits = [self.enzyme_adapters[enz](shared).squeeze(-1) for enz in PER_ENZYME_HEADS]
        enzyme_cls_logits = self.enzyme_classifier(shared)

        return binary_logit, per_enzyme_logits, enzyme_cls_logits, shared


# ---------------------------------------------------------------------------
# Model: Phase 3 + GAT (Experiment 2)
# ---------------------------------------------------------------------------


class Phase3GATModel(nn.Module):
    """Phase 3 with AdarEdit GAT encoder replacing Conv2D.

    Fusion: RNA-FM(640) + GAT(64) + EditDelta(640) + Hand(40) = 1384-dim
    """

    def __init__(self):
        super().__init__()
        d_fusion = D_RNAFM + D_GAT_OUT + D_EDIT_DELTA + D_HAND  # 1384

        # GAT encoder
        self.node_proj = nn.Linear(22, GAT_HIDDEN)
        self.gat_layers = nn.ModuleList([
            ManualGATLayer(GAT_HIDDEN, GAT_HIDDEN, n_heads=GAT_HEADS, dropout=0.1)
            for _ in range(GAT_LAYERS)
        ])
        self.gat_out = nn.Linear(GAT_HIDDEN, D_GAT_OUT)

        self.shared_encoder = nn.Sequential(
            nn.Linear(d_fusion, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        self.binary_head = nn.Linear(128, 1)
        self.enzyme_adapters = nn.ModuleDict({
            enz: nn.Sequential(nn.Linear(128, 32), nn.GELU(), nn.Linear(32, 1))
            for enz in PER_ENZYME_HEADS
        })
        self.enzyme_classifier = nn.Sequential(
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, N_ENZYMES),
        )

    def get_shared_params(self):
        params = list(self.node_proj.parameters())
        for layer in self.gat_layers:
            params.extend(layer.parameters())
        params.extend(self.gat_out.parameters())
        params.extend(self.shared_encoder.parameters())
        return params

    def get_adapter_params(self):
        params = list(self.binary_head.parameters())
        for enz in PER_ENZYME_HEADS:
            params.extend(self.enzyme_adapters[enz].parameters())
        params.extend(self.enzyme_classifier.parameters())
        return params

    def forward(self, batch):
        rnafm = batch["rnafm"]
        edit_delta = batch["edit_delta"]
        hand = batch["hand_feat"]
        node_feats = batch["node_feats"]
        edge_index_list = batch["edge_index"]
        edge_type_list = batch["edge_type"]

        B = rnafm.size(0)
        n_nodes = node_feats.size(1)
        center_node = n_nodes // 2

        # Process GAT per sample (variable edge counts)
        gat_embeddings = []
        for b in range(B):
            x = self.node_proj(node_feats[b])  # [41, 64]
            ei = edge_index_list[b].to(x.device)
            et = edge_type_list[b].to(x.device)
            for gat_layer in self.gat_layers:
                x = F.gelu(gat_layer(x, ei, et))
            gat_embeddings.append(x[center_node])

        gat_out = torch.stack(gat_embeddings)  # [B, 64]
        gat_out = self.gat_out(gat_out)

        fused = torch.cat([rnafm, gat_out, edit_delta, hand], dim=-1)
        shared = self.shared_encoder(fused)

        binary_logit = self.binary_head(shared).squeeze(-1)
        per_enzyme_logits = [self.enzyme_adapters[enz](shared).squeeze(-1) for enz in PER_ENZYME_HEADS]
        enzyme_cls_logits = self.enzyme_classifier(shared)

        return binary_logit, per_enzyme_logits, enzyme_cls_logits, shared


# ---------------------------------------------------------------------------
# Model: Hand Features Only MLP (for v4 experiments)
# ---------------------------------------------------------------------------


class HandFeatMLPModel(nn.Module):
    """MLP on 40-dim hand features with per-enzyme adapters.

    Used for v4 experiments where RNA-FM is not available for negatives.
    """

    def __init__(self, d_input=D_HAND):
        super().__init__()
        self.shared_encoder = nn.Sequential(
            nn.Linear(d_input, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        self.binary_head = nn.Linear(128, 1)
        self.enzyme_adapters = nn.ModuleDict({
            enz: nn.Sequential(nn.Linear(128, 32), nn.GELU(), nn.Linear(32, 1))
            for enz in PER_ENZYME_HEADS
        })
        self.enzyme_classifier = nn.Sequential(
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, N_ENZYMES),
        )

    def get_shared_params(self):
        return list(self.shared_encoder.parameters())

    def get_adapter_params(self):
        params = list(self.binary_head.parameters())
        for enz in PER_ENZYME_HEADS:
            params.extend(self.enzyme_adapters[enz].parameters())
        params.extend(self.enzyme_classifier.parameters())
        return params

    def forward(self, batch):
        hand = batch["hand_feat"]
        shared = self.shared_encoder(hand)

        binary_logit = self.binary_head(shared).squeeze(-1)
        per_enzyme_logits = [self.enzyme_adapters[enz](shared).squeeze(-1) for enz in PER_ENZYME_HEADS]
        enzyme_cls_logits = self.enzyme_classifier(shared)

        return binary_logit, per_enzyme_logits, enzyme_cls_logits, shared


# ---------------------------------------------------------------------------
# Loss computation (shared across all models)
# ---------------------------------------------------------------------------


def compute_loss(binary_logit, per_enzyme_logits, enzyme_cls_logits, batch,
                 enzyme_head_weight, enzyme_cls_weight):
    """Compute combined multi-head loss."""
    device = binary_logit.device

    loss_binary = F.binary_cross_entropy_with_logits(binary_logit, batch["label_binary"])

    per_enzyme_labels = batch["per_enzyme_labels"]
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

    pos_mask = batch["label_binary"] == 1
    loss_enzyme_cls = torch.tensor(0.0, device=device)
    if pos_mask.sum() > 0:
        loss_enzyme_cls = F.cross_entropy(
            enzyme_cls_logits[pos_mask], batch["label_enzyme"][pos_mask],
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


def train_one_epoch(model, loader, optimizer, enzyme_head_weight, enzyme_cls_weight, is_gat=False):
    """Train one epoch. Returns average loss."""
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
def evaluate_model(model, loader, is_gat=False):
    """Evaluate: overall AUROC, per-enzyme AUROC (adapter), enzyme accuracy."""
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

    try:
        overall_auroc = roc_auc_score(lbl_binary, binary_probs)
    except ValueError:
        overall_auroc = 0.5

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

    pos_mask = lbl_binary == 1
    enzyme_acc = 0.0
    if pos_mask.sum() > 0:
        enzyme_acc = accuracy_score(lbl_enzyme[pos_mask], enzyme_cls_all[pos_mask].argmax(axis=1))

    return {
        "overall_auroc": float(overall_auroc),
        "per_enzyme_auroc": per_enzyme_auroc,
        "enzyme_accuracy": float(enzyme_acc),
        "binary_probs": binary_probs,
        "lbl_binary": lbl_binary,
        "lbl_enzyme": lbl_enzyme,
    }


# ---------------------------------------------------------------------------
# Two-stage training
# ---------------------------------------------------------------------------


def run_two_stage_training(model, train_loader, val_loader, is_gat=False, tag=""):
    """Run two-stage training on a model. Returns best metrics and history."""
    history = {"stage": [], "epoch": [], "train_loss": [], "val_overall_auroc": []}
    best_auroc = 0.0
    best_state = None

    # Stage 1: Train everything jointly
    logger.info(f"  [{tag}] Stage 1: Joint training ({STAGE1_EPOCHS} epochs, lr={STAGE1_LR})")
    optimizer_s1 = torch.optim.AdamW(model.parameters(), lr=STAGE1_LR, weight_decay=STAGE1_WEIGHT_DECAY)
    scheduler_s1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_s1, T_max=STAGE1_EPOCHS)

    for epoch in range(STAGE1_EPOCHS):
        loss = train_one_epoch(
            model, train_loader, optimizer_s1,
            enzyme_head_weight=STAGE1_ENZYME_HEAD_WEIGHT,
            enzyme_cls_weight=STAGE1_ENZYME_CLS_WEIGHT,
            is_gat=is_gat,
        )
        scheduler_s1.step()

        metrics = evaluate_model(model, val_loader, is_gat=is_gat)
        history["stage"].append(1)
        history["epoch"].append(epoch)
        history["train_loss"].append(loss)
        history["val_overall_auroc"].append(metrics["overall_auroc"])

        if metrics["overall_auroc"] > best_auroc:
            best_auroc = metrics["overall_auroc"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        enz_str = ", ".join(
            f"{e}={metrics['per_enzyme_auroc'].get(e, 0):.3f}" for e in PER_ENZYME_HEADS
        )
        logger.info(
            f"    [S1] Epoch {epoch+1}/{STAGE1_EPOCHS}: loss={loss:.4f}, "
            f"overall={metrics['overall_auroc']:.4f}, {enz_str}"
        )

    # Stage 2: Freeze shared encoder, train adapters only
    logger.info(f"  [{tag}] Stage 2: Adapter-only training ({STAGE2_EPOCHS} epochs, lr={STAGE2_LR})")

    for p in model.get_shared_params():
        p.requires_grad = False

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"    Trainable parameters (adapters only): {n_trainable:,}")

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
            is_gat=is_gat,
        )
        scheduler_s2.step()

        metrics = evaluate_model(model, val_loader, is_gat=is_gat)
        history["stage"].append(2)
        history["epoch"].append(STAGE1_EPOCHS + epoch)
        history["train_loss"].append(loss)
        history["val_overall_auroc"].append(metrics["overall_auroc"])

        if metrics["overall_auroc"] > best_auroc:
            best_auroc = metrics["overall_auroc"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 5 == 0:
            enz_str = ", ".join(
                f"{e}={metrics['per_enzyme_auroc'].get(e, 0):.3f}" for e in PER_ENZYME_HEADS
            )
            logger.info(
                f"    [S2] Epoch {epoch+1}/{STAGE2_EPOCHS}: loss={loss:.4f}, "
                f"overall={metrics['overall_auroc']:.4f}, {enz_str}"
            )

    # Unfreeze and load best state
    for p in model.parameters():
        p.requires_grad = True

    if best_state is not None:
        model.load_state_dict(best_state)

    final_metrics = evaluate_model(model, val_loader, is_gat=is_gat)
    return final_metrics, history


# ---------------------------------------------------------------------------
# XGBoost baseline
# ---------------------------------------------------------------------------


def run_xgboost_baseline(features, labels_binary, labels_enzyme, per_enzyme_labels,
                         folds, tag="XGBoost"):
    """Run XGBoost as baseline comparison."""
    from xgboost import XGBClassifier

    logger.info(f"\n{'='*60}")
    logger.info(f"{tag}: XGBoost on {features.shape[1]}-dim features")
    logger.info(f"{'='*60}")

    fold_results = []
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        X_train, X_val = features[train_idx], features[val_idx]
        y_train, y_val = labels_binary[train_idx], labels_binary[val_idx]

        # Compute class weight
        n_pos = (y_train == 1).sum()
        n_neg = (y_train == 0).sum()
        scale_pos = float(n_neg / max(n_pos, 1))

        clf = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos,
            tree_method="hist",
            random_state=SEED + fold_idx,
            n_jobs=-1,
            eval_metric="logloss",
        )
        clf.fit(X_train, y_train, verbose=False)
        probs = clf.predict_proba(X_val)[:, 1]

        try:
            overall_auroc = roc_auc_score(y_val, probs)
        except ValueError:
            overall_auroc = 0.5

        # Per-enzyme AUROC
        per_enzyme_auroc = {}
        val_per_enz = per_enzyme_labels[val_idx]
        for head_idx, enz_name in enumerate(PER_ENZYME_HEADS):
            mask = val_per_enz[:, head_idx] >= 0
            if mask.sum() < 2:
                continue
            labels_sub = val_per_enz[:, head_idx][mask]
            probs_sub = probs[mask]
            if len(np.unique(labels_sub)) < 2:
                continue
            try:
                per_enzyme_auroc[enz_name] = float(roc_auc_score(labels_sub, probs_sub))
            except ValueError:
                pass

        enz_str = ", ".join(f"{e}={per_enzyme_auroc.get(e, 0):.3f}" for e in PER_ENZYME_HEADS)
        logger.info(f"  Fold {fold_idx}: overall={overall_auroc:.4f}, {enz_str}")

        fold_results.append({
            "overall_auroc": overall_auroc,
            "per_enzyme_auroc": per_enzyme_auroc,
        })

    return fold_results


# ---------------------------------------------------------------------------
# Experiment 1: Phase 3 + Conv2D
# ---------------------------------------------------------------------------


def experiment1_phase3_conv2d(data, folds):
    """Experiment 1: Phase 3 + Conv2D BP probability matrix."""
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 1: Phase 3 + Conv2D BP Probability Matrix")
    logger.info("  Fusion: RNA-FM(640) + Conv2D(128) + EditDelta(640) + Hand(40) = 1448-dim")
    logger.info("=" * 70)

    n = len(data["site_ids"])
    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        torch.manual_seed(SEED + fold_idx)
        np.random.seed(SEED + fold_idx)

        logger.info(f"\n  --- Fold {fold_idx} (train={len(train_idx)}, val={len(val_idx)}) ---")
        t0 = time.time()

        model = Phase3Conv2DModel().to(DEVICE)
        if fold_idx == 0:
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"  Model parameters: {n_params:,}")

        train_ds = Phase3Dataset(list(train_idx), data)
        val_ds = Phase3Dataset(list(val_idx), data)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  collate_fn=collate_fn, num_workers=0, pin_memory=False)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                collate_fn=collate_fn, num_workers=0, pin_memory=False)

        metrics, history = run_two_stage_training(model, train_loader, val_loader, tag="Phase3+Conv2D")

        elapsed = time.time() - t0
        enz_str = ", ".join(f"{e}={metrics['per_enzyme_auroc'].get(e, 0):.3f}" for e in PER_ENZYME_HEADS)
        logger.info(f"  Fold {fold_idx} done in {elapsed:.0f}s: overall={metrics['overall_auroc']:.4f}, {enz_str}")

        fold_results.append(metrics)

        del model
        if DEVICE.type == "mps":
            torch.mps.empty_cache()
        gc.collect()

    return fold_results


# ---------------------------------------------------------------------------
# Experiment 2: Phase 3 + GAT
# ---------------------------------------------------------------------------


def experiment2_phase3_gat(data, folds):
    """Experiment 2: Phase 3 + AdarEdit GAT encoder."""
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 2: Phase 3 + AdarEdit GAT Encoder")
    logger.info("  Fusion: RNA-FM(640) + GAT(64) + EditDelta(640) + Hand(40) = 1384-dim")
    logger.info("=" * 70)

    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        torch.manual_seed(SEED + fold_idx)
        np.random.seed(SEED + fold_idx)

        logger.info(f"\n  --- Fold {fold_idx} (train={len(train_idx)}, val={len(val_idx)}) ---")
        t0 = time.time()

        model = Phase3GATModel().to(DEVICE)
        if fold_idx == 0:
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"  Model parameters: {n_params:,}")

        train_ds = GATPhase3Dataset(list(train_idx), data)
        val_ds = GATPhase3Dataset(list(val_idx), data)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  collate_fn=gat_collate_fn, num_workers=0, pin_memory=False)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                collate_fn=gat_collate_fn, num_workers=0, pin_memory=False)

        metrics, history = run_two_stage_training(model, train_loader, val_loader, is_gat=True, tag="Phase3+GAT")

        elapsed = time.time() - t0
        enz_str = ", ".join(f"{e}={metrics['per_enzyme_auroc'].get(e, 0):.3f}" for e in PER_ENZYME_HEADS)
        logger.info(f"  Fold {fold_idx} done in {elapsed:.0f}s: overall={metrics['overall_auroc']:.4f}, {enz_str}")

        fold_results.append(metrics)

        del model
        if DEVICE.type == "mps":
            torch.mps.empty_cache()
        gc.collect()

    return fold_results


# ---------------------------------------------------------------------------
# Experiment 3: Best architecture on v4 datasets (hand features only)
# ---------------------------------------------------------------------------


def experiment3_v4_datasets(best_model_name: str):
    """Experiment 3: Retrain best architecture on v4 datasets with hand features only."""
    logger.info("\n" + "=" * 70)
    logger.info(f"EXPERIMENT 3: {best_model_name} on v4 datasets (40-dim hand features only)")
    logger.info("=" * 70)

    v4_results = {}

    for variant in ["random", "hard", "large"]:
        logger.info(f"\n--- v4-{variant} ---")
        data = load_v4_data(variant)
        n = len(data["site_ids"])

        skf = StratifiedKFold(n_splits=max(N_FOLDS, 2), shuffle=True, random_state=SEED)
        folds = list(skf.split(np.arange(n), data["labels_binary"]))[:N_FOLDS]

        # XGBoost baseline on this v4 dataset
        xgb_results = run_xgboost_baseline(
            data["hand_features"], data["labels_binary"],
            data["labels_enzyme"], data["per_enzyme_labels"],
            folds, tag=f"XGBoost_v4_{variant}",
        )

        # Neural model (hand features MLP)
        logger.info(f"\n  Training HandFeat MLP on v4-{variant}...")
        neural_results = []

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            torch.manual_seed(SEED + fold_idx)
            np.random.seed(SEED + fold_idx)

            logger.info(f"\n  --- Fold {fold_idx} (train={len(train_idx)}, val={len(val_idx)}) ---")
            t0 = time.time()

            model = HandFeatMLPModel(d_input=D_HAND).to(DEVICE)
            if fold_idx == 0:
                n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                logger.info(f"  Model parameters: {n_params:,}")

            # For large v4, increase batch size
            batch_size = 512 if variant == "large" else BATCH_SIZE

            train_ds = HandFeatOnlyDataset(list(train_idx), data)
            val_ds = HandFeatOnlyDataset(list(val_idx), data)
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                      collate_fn=collate_fn, num_workers=0, pin_memory=False)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                    collate_fn=collate_fn, num_workers=0, pin_memory=False)

            metrics, _ = run_two_stage_training(model, train_loader, val_loader,
                                                 tag=f"HandMLP_v4_{variant}")

            elapsed = time.time() - t0
            enz_str = ", ".join(f"{e}={metrics['per_enzyme_auroc'].get(e, 0):.3f}" for e in PER_ENZYME_HEADS)
            logger.info(f"  Fold {fold_idx} done in {elapsed:.0f}s: overall={metrics['overall_auroc']:.4f}, {enz_str}")

            neural_results.append(metrics)

            del model
            if DEVICE.type == "mps":
                torch.mps.empty_cache()
            gc.collect()

        v4_results[variant] = {
            "xgboost": xgb_results,
            "neural": neural_results,
        }

        del data
        gc.collect()

    return v4_results


# ---------------------------------------------------------------------------
# Experiment 4: Hard Negative Curriculum (P7)
# ---------------------------------------------------------------------------


def experiment4_hard_neg_curriculum(data, folds):
    """Experiment 4: Hard negative curriculum training.

    1. Train normally for 15 epochs
    2. Score all negatives with the trained model
    3. Select negatives in 80th-95th percentile of model confidence, >1000bp from any positive
    4. Add selected hard negatives to training set
    5. Continue training for 15 more epochs
    """
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 4: Hard Negative Curriculum (P7)")
    logger.info("  Phase A: 15 epochs normal training")
    logger.info("  Phase B: mine hard negatives, continue 15 epochs")
    logger.info("=" * 70)

    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        torch.manual_seed(SEED + fold_idx)
        np.random.seed(SEED + fold_idx)

        logger.info(f"\n  --- Fold {fold_idx} (train={len(train_idx)}, val={len(val_idx)}) ---")
        t0 = time.time()

        model = HandFeatMLPModel(d_input=D_HAND).to(DEVICE)

        # Use hand features only for curriculum (simpler, faster)
        train_ds = HandFeatOnlyDataset(list(train_idx), data)
        val_ds = HandFeatOnlyDataset(list(val_idx), data)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  collate_fn=collate_fn, num_workers=0, pin_memory=False)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                collate_fn=collate_fn, num_workers=0, pin_memory=False)

        # Phase A: Normal training (15 epochs, all joint)
        logger.info(f"  [{fold_idx}] Phase A: Normal training (15 epochs)")
        optimizer_a = torch.optim.AdamW(model.parameters(), lr=STAGE1_LR, weight_decay=STAGE1_WEIGHT_DECAY)
        scheduler_a = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_a, T_max=15)
        best_auroc_a = 0.0
        best_state_a = None

        for epoch in range(15):
            loss = train_one_epoch(
                model, train_loader, optimizer_a,
                enzyme_head_weight=STAGE1_ENZYME_HEAD_WEIGHT,
                enzyme_cls_weight=STAGE1_ENZYME_CLS_WEIGHT,
            )
            scheduler_a.step()
            metrics = evaluate_model(model, val_loader)
            if metrics["overall_auroc"] > best_auroc_a:
                best_auroc_a = metrics["overall_auroc"]
                best_state_a = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if (epoch + 1) % 5 == 0:
                logger.info(f"    [A] Epoch {epoch+1}/15: loss={loss:.4f}, overall={metrics['overall_auroc']:.4f}")

        # Score training negatives to find hard ones
        logger.info(f"  [{fold_idx}] Mining hard negatives from training set...")
        if best_state_a is not None:
            model.load_state_dict(best_state_a)

        # Get predictions on all training samples
        model.eval()
        train_probs = []
        with torch.no_grad():
            score_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False,
                                      collate_fn=collate_fn, num_workers=0)
            for batch in score_loader:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                binary_logit, _, _, _ = model(batch)
                train_probs.append(torch.sigmoid(binary_logit).cpu().numpy())
        train_probs = np.concatenate(train_probs)

        # Find negatives in 80th-95th percentile of model confidence
        train_labels = data["labels_binary"][list(train_idx)]
        neg_mask = train_labels == 0
        neg_probs = train_probs[neg_mask]
        neg_indices = np.array(list(train_idx))[neg_mask]

        p80 = np.percentile(neg_probs, 80)
        p95 = np.percentile(neg_probs, 95)
        hard_neg_mask_in_negs = (neg_probs >= p80) & (neg_probs <= p95)

        # Distance filtering: skip negatives within 1000bp of any positive
        # Use site_id parsing (format: chr:pos:strand or similar)
        df = data["df"]
        pos_positions = {}
        for idx in train_idx:
            if data["labels_binary"][idx] == 1:
                sid = data["site_ids"][idx]
                parts = sid.split(":")
                if len(parts) >= 2:
                    chrom = parts[0]
                    try:
                        pos = int(parts[1])
                        if chrom not in pos_positions:
                            pos_positions[chrom] = []
                        pos_positions[chrom].append(pos)
                    except ValueError:
                        pass

        # Convert to sorted arrays for fast proximity check
        for chrom in pos_positions:
            pos_positions[chrom] = np.sort(pos_positions[chrom])

        selected_hard_neg_indices = []
        for i, is_hard in enumerate(hard_neg_mask_in_negs):
            if not is_hard:
                continue
            orig_idx = neg_indices[i]
            sid = data["site_ids"][orig_idx]
            parts = sid.split(":")
            too_close = False
            if len(parts) >= 2:
                chrom = parts[0]
                try:
                    pos = int(parts[1])
                    if chrom in pos_positions:
                        closest = pos_positions[chrom]
                        dists = np.abs(closest - pos)
                        if dists.min() < 1000:
                            too_close = True
                except ValueError:
                    pass
            if not too_close:
                selected_hard_neg_indices.append(orig_idx)

        n_hard = len(selected_hard_neg_indices)
        logger.info(f"    Found {n_hard} hard negatives (80th-95th pctl, >1000bp from pos)")
        logger.info(f"    p80={p80:.3f}, p95={p95:.3f}")

        # Phase B: Continue training with augmented set
        augmented_train_idx = list(train_idx) + selected_hard_neg_indices
        logger.info(f"  [{fold_idx}] Phase B: Curriculum training ({len(augmented_train_idx)} samples, 15 epochs)")

        aug_train_ds = HandFeatOnlyDataset(augmented_train_idx, data)
        aug_train_loader = DataLoader(aug_train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                      collate_fn=collate_fn, num_workers=0, pin_memory=False)

        optimizer_b = torch.optim.AdamW(model.parameters(), lr=STAGE2_LR, weight_decay=STAGE2_WEIGHT_DECAY)
        scheduler_b = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_b, T_max=15)
        best_auroc_b = best_auroc_a
        best_state_b = best_state_a

        for epoch in range(15):
            loss = train_one_epoch(
                model, aug_train_loader, optimizer_b,
                enzyme_head_weight=STAGE1_ENZYME_HEAD_WEIGHT,
                enzyme_cls_weight=STAGE1_ENZYME_CLS_WEIGHT,
            )
            scheduler_b.step()
            metrics = evaluate_model(model, val_loader)
            if metrics["overall_auroc"] > best_auroc_b:
                best_auroc_b = metrics["overall_auroc"]
                best_state_b = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if (epoch + 1) % 5 == 0:
                logger.info(f"    [B] Epoch {epoch+1}/15: loss={loss:.4f}, overall={metrics['overall_auroc']:.4f}")

        # Final eval
        if best_state_b is not None:
            model.load_state_dict(best_state_b)
        final_metrics = evaluate_model(model, val_loader)

        elapsed = time.time() - t0
        enz_str = ", ".join(f"{e}={final_metrics['per_enzyme_auroc'].get(e, 0):.3f}" for e in PER_ENZYME_HEADS)
        logger.info(
            f"  Fold {fold_idx} done in {elapsed:.0f}s: "
            f"pre-curriculum={best_auroc_a:.4f}, post-curriculum={final_metrics['overall_auroc']:.4f}, "
            f"{enz_str}"
        )

        final_metrics["pre_curriculum_auroc"] = float(best_auroc_a)
        final_metrics["n_hard_negatives"] = n_hard
        fold_results.append(final_metrics)

        del model
        if DEVICE.type == "mps":
            torch.mps.empty_cache()
        gc.collect()

    return fold_results


# ---------------------------------------------------------------------------
# Experiment 5: Comprehensive comparison
# ---------------------------------------------------------------------------


def aggregate_results(fold_results):
    """Aggregate fold results into mean/std summary."""
    overall = [r["overall_auroc"] for r in fold_results]
    per_enzyme = {}
    for enz in PER_ENZYME_HEADS:
        vals = [r["per_enzyme_auroc"].get(enz, 0) for r in fold_results]
        per_enzyme[enz] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    return {
        "overall": {"mean": float(np.mean(overall)), "std": float(np.std(overall))},
        "per_enzyme": per_enzyme,
    }


def experiment5_comparison(all_results):
    """Print comprehensive comparison table."""
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 5: COMPREHENSIVE COMPARISON TABLE")
    logger.info("=" * 70)

    # Header
    header = f"{'Model':<28} {'Dataset':<12} {'Overall':>8} {'A3A':>8} {'A3B':>8} {'A3G':>8} {'Both':>8} {'Neither':>8}"
    logger.info(header)
    logger.info("-" * len(header))

    comparison_table = []

    for model_name, dataset_name, results_summary in all_results:
        overall = results_summary["overall"]["mean"]
        row_vals = [results_summary["per_enzyme"].get(enz, {}).get("mean", 0) for enz in PER_ENZYME_HEADS]
        row = (
            f"{model_name:<28} {dataset_name:<12} "
            f"{overall:>8.3f} "
            + " ".join(f"{v:>8.3f}" for v in row_vals)
        )
        logger.info(row)
        comparison_table.append({
            "model": model_name,
            "dataset": dataset_name,
            "overall_auroc": overall,
            **{enz: results_summary["per_enzyme"].get(enz, {}).get("mean", 0) for enz in PER_ENZYME_HEADS},
        })

    logger.info("-" * len(header))
    return comparison_table


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_final_comparison(comparison_table):
    """Plot comprehensive comparison bar chart."""
    fig, ax = plt.subplots(figsize=(16, 8))

    # Filter to v3 models for main comparison
    v3_models = [r for r in comparison_table if r["dataset"] == "v3"]
    if not v3_models:
        v3_models = comparison_table

    model_names = [r["model"] for r in v3_models]
    enzymes = PER_ENZYME_HEADS
    n_models = len(model_names)
    n_enzymes = len(enzymes)

    x = np.arange(n_enzymes)
    width = 0.8 / n_models
    colors = plt.cm.tab10(np.linspace(0, 1, n_models))

    for i, (model_data, color) in enumerate(zip(v3_models, colors)):
        vals = [model_data.get(enz, 0) for enz in enzymes]
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=model_data["model"], color=color, alpha=0.8)

    ax.set_ylabel("AUROC", fontsize=12)
    ax.set_title("Final Architecture Comparison (2-Fold CV)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(enzymes, fontsize=11)
    ax.set_ylim(0.5, 1.0)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plot_path = OUTPUT_DIR / "final_comparison.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logger.info(f"Comparison plot saved to {plot_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    t_start = time.time()
    logger.info("=" * 70)
    logger.info("FINAL ARCHITECTURE COMPARISON")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Config: {N_FOLDS} folds, Stage1={STAGE1_EPOCHS}ep, Stage2={STAGE2_EPOCHS}ep")
    logger.info("=" * 70)

    # =====================================================================
    # Load v3 data (shared across Experiments 1, 2, 4)
    # =====================================================================
    data = load_v3_data()
    n = len(data["site_ids"])

    skf = StratifiedKFold(n_splits=max(N_FOLDS, 2), shuffle=True, random_state=SEED)
    folds = list(skf.split(np.arange(n), data["labels_binary"]))[:N_FOLDS]

    all_comparison_entries = []

    # =====================================================================
    # XGBoost baselines on v3 (hand features 40-dim and full 1320-dim)
    # =====================================================================

    # XGBoost on 40-dim hand features
    xgb_hand_results = run_xgboost_baseline(
        data["hand_features"], data["labels_binary"],
        data["labels_enzyme"], data["per_enzyme_labels"],
        folds, tag="XGBoost_Hand(40d)",
    )
    xgb_hand_summary = aggregate_results(xgb_hand_results)
    all_comparison_entries.append(("XGBoost_Hand(40d)", "v3", xgb_hand_summary))

    # XGBoost on full 1320-dim (RNA-FM + EditDelta + Hand)
    full_features = np.concatenate([
        data["rnafm_features"], data["edit_delta_features"], data["hand_features"]
    ], axis=1)
    logger.info(f"Full feature matrix: {full_features.shape}")

    xgb_full_results = run_xgboost_baseline(
        full_features, data["labels_binary"],
        data["labels_enzyme"], data["per_enzyme_labels"],
        folds, tag="XGBoost_Full(1320d)",
    )
    xgb_full_summary = aggregate_results(xgb_full_results)
    all_comparison_entries.append(("XGBoost_Full(1320d)", "v3", xgb_full_summary))

    del full_features
    gc.collect()

    # =====================================================================
    # Add Phase 3 baseline from existing results
    # =====================================================================
    phase3_summary = {
        "overall": {"mean": 0.798, "std": 0.003},
        "per_enzyme": {
            "A3A": {"mean": 0.877, "std": 0.002},
            "A3B": {"mean": 0.808, "std": 0.002},
            "A3G": {"mean": 0.909, "std": 0.009},
            "A3A_A3G": {"mean": 0.972, "std": 0.002},
            "Neither": {"mean": 0.872, "std": 0.013},
        },
    }
    all_comparison_entries.append(("Phase3_baseline", "v3", phase3_summary))

    # =====================================================================
    # Experiment 1: Phase 3 + Conv2D
    # =====================================================================
    exp1_results = experiment1_phase3_conv2d(data, folds)
    exp1_summary = aggregate_results(exp1_results)
    all_comparison_entries.append(("Phase3+Conv2D", "v3", exp1_summary))

    # =====================================================================
    # Experiment 2: Phase 3 + GAT
    # =====================================================================
    exp2_results = experiment2_phase3_gat(data, folds)
    exp2_summary = aggregate_results(exp2_results)
    all_comparison_entries.append(("Phase3+GAT", "v3", exp2_summary))

    # =====================================================================
    # Determine best architecture from Exp 1-2 + Phase 3 baseline
    # =====================================================================
    candidate_scores = {
        "Phase3_baseline": phase3_summary["overall"]["mean"],
        "Phase3+Conv2D": exp1_summary["overall"]["mean"],
        "Phase3+GAT": exp2_summary["overall"]["mean"],
    }
    best_model_name = max(candidate_scores, key=candidate_scores.get)
    logger.info(f"\nBest neural architecture: {best_model_name} (overall={candidate_scores[best_model_name]:.4f})")

    # =====================================================================
    # Experiment 3: Best on v4 datasets (hand features only)
    # =====================================================================
    v4_results = experiment3_v4_datasets(best_model_name)

    for variant in ["random", "hard", "large"]:
        # XGBoost on v4
        xgb_v4_summary = aggregate_results(v4_results[variant]["xgboost"])
        all_comparison_entries.append((f"XGBoost_Hand(40d)", f"v4-{variant}", xgb_v4_summary))

        # Neural on v4
        neural_v4_summary = aggregate_results(v4_results[variant]["neural"])
        all_comparison_entries.append((f"HandMLP_neural", f"v4-{variant}", neural_v4_summary))

    # =====================================================================
    # Experiment 4: Hard Negative Curriculum
    # =====================================================================
    exp4_results = experiment4_hard_neg_curriculum(data, folds)
    exp4_summary = aggregate_results(exp4_results)
    all_comparison_entries.append(("P7_Curriculum", "v3", exp4_summary))

    # Report pre/post curriculum improvement
    pre_aurocs = [r.get("pre_curriculum_auroc", 0) for r in exp4_results]
    post_aurocs = [r["overall_auroc"] for r in exp4_results]
    logger.info(f"\nP7 Curriculum: pre={np.mean(pre_aurocs):.4f} -> post={np.mean(post_aurocs):.4f}")

    # =====================================================================
    # Experiment 5: Comprehensive comparison
    # =====================================================================
    comparison_table = experiment5_comparison(all_comparison_entries)

    # Plot
    plot_final_comparison(comparison_table)

    # =====================================================================
    # Save all results
    # =====================================================================
    all_results_json = {
        "experiments": {
            "exp1_phase3_conv2d": {
                "model": "Phase3+Conv2D",
                "dataset": "v3",
                "fusion_dim": 1448,
                "results": exp1_summary,
            },
            "exp2_phase3_gat": {
                "model": "Phase3+GAT",
                "dataset": "v3",
                "fusion_dim": 1384,
                "results": exp2_summary,
            },
            "exp3_v4": {
                variant: {
                    "xgboost": aggregate_results(v4_results[variant]["xgboost"]),
                    "neural": aggregate_results(v4_results[variant]["neural"]),
                }
                for variant in ["random", "hard", "large"]
            },
            "exp4_curriculum": {
                "model": "P7_Curriculum",
                "dataset": "v3",
                "results": exp4_summary,
                "pre_curriculum_mean": float(np.mean(pre_aurocs)),
                "post_curriculum_mean": float(np.mean(post_aurocs)),
                "n_hard_negatives": [r.get("n_hard_negatives", 0) for r in exp4_results],
            },
        },
        "baselines": {
            "xgboost_hand_40d": xgb_hand_summary,
            "xgboost_full_1320d": xgb_full_summary,
            "phase3_baseline": phase3_summary,
        },
        "comparison_table": comparison_table,
        "best_neural_architecture": best_model_name,
        "config": {
            "n_folds": N_FOLDS,
            "stage1_epochs": STAGE1_EPOCHS,
            "stage2_epochs": STAGE2_EPOCHS,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
        },
    }

    results_path = OUTPUT_DIR / "final_comparison_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results_json, f, indent=2)
    logger.info(f"\nAll results saved to {results_path}")

    total_time = time.time() - t_start
    logger.info(f"\nTotal runtime: {total_time:.0f}s ({total_time/60:.1f}min)")
    logger.info("Final comparison complete.")


if __name__ == "__main__":
    main()
