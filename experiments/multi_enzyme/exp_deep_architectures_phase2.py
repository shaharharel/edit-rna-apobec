#!/usr/bin/env python
"""Phase 2: Deep architecture comparison with v4-large dataset and AdarEdit-adapted GAT.

Tasks:
  1. Retrain Conv2D_BP and DualPath_Residual on v4-large (385K sites, 1:50 ratio)
  2. Train XGBoost on v4-large for fair comparison
  3. Implement and train AdarEdit-adapted GAT (Graph Attention + parallel CNN + hand features)
  4. Compare all results (v3 Phase 1 vs v4-large Phase 2)

Output: experiments/multi_enzyme/outputs/deep_architectures/
"""

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

from data.apobec_feature_extraction import (
    build_hand_features, extract_motif_from_seq, LOOP_FEATURE_COLS,
    _extract_loop_geometry,
)

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "multi_enzyme" / "outputs" / "deep_architectures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = OUTPUT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Set up logging to both file and stdout
log_file = LOG_DIR / "phase2_training.log"
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
CENTER = 100
SEED = 42
N_FOLDS = 5

# v4-large hyperparameters (larger batches, fewer epochs for 385K samples)
V4_BATCH_SIZE = 512
V4_N_EPOCHS = 30
V4_LR = 1e-3
V4_WEIGHT_DECAY = 1e-4
ENZYME_LOSS_WEIGHT = 0.3

# v3 hyperparameters (same as Phase 1)
V3_BATCH_SIZE = 64
V3_N_EPOCHS = 50
V3_LR = 1e-3

# GAT hyperparameters
GAT_HIDDEN = 64
GAT_HEADS = 4
GAT_LAYERS = 3
CNN_CHANNELS = 32


# ---------------------------------------------------------------------------
# ViennaRNA folding (parallel) - reused from Phase 1
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
    """Load v3 dataset (15,342 sites). Same as Phase 1."""
    logger.info("=" * 60)
    logger.info("Loading V3 dataset (15,342 sites)")
    logger.info("=" * 60)

    splits_path = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "splits_multi_enzyme_v3_with_negatives.csv"
    df = pd.read_csv(splits_path)
    logger.info(f"  Loaded {len(df)} sites")

    seq_path = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "multi_enzyme_sequences_v3_with_negatives.json"
    with open(seq_path) as f:
        sequences = json.load(f)
    logger.info(f"  Loaded {len(sequences)} sequences")

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

    # One-hot encode sequences
    base_map = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3}
    onehot_seqs = np.zeros((n, 4, 201), dtype=np.float32)
    for i, sid in enumerate(site_ids):
        seq = sequences[sid].upper().replace("T", "U")
        for j, base in enumerate(seq[:201]):
            if base in base_map:
                onehot_seqs[i, base_map[base], j] = 1.0

    # Fold sequences for structures and BPP
    logger.info("Computing ViennaRNA structures and BPP submatrices for v3...")
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

    # Encode dot-bracket
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
        "sequences": sequences,
    }


def load_v4_data() -> Dict:
    """Load v4-large dataset (385,764 sites, 1:50 ratio).

    For the v4-large dataset, we compute hand features (motif only for sites
    without loop/structure data) and fold structures only for positives
    (negatives get zeros for BPP to save time).
    """
    logger.info("=" * 60)
    logger.info("Loading V4-large dataset (385,764 sites)")
    logger.info("=" * 60)

    splits_path = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "splits_v4_large_negatives.csv"
    df = pd.read_csv(splits_path, low_memory=False)
    logger.info(f"  Loaded {len(df)} sites")
    logger.info(f"  Positives: {(df['is_edited']==1).sum()}, Negatives: {(df['is_edited']==0).sum()}")

    seq_path = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "multi_enzyme_sequences_v4_large.json"
    with open(seq_path) as f:
        sequences = json.load(f)
    logger.info(f"  Loaded {len(sequences)} sequences")

    # Load v3 loop and structure data (covers positives)
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
    logger.info(f"  Loaded {len(structure_delta)} structure delta features from v3 cache")

    # Filter to sites with sequences
    site_ids = df["site_id"].tolist()
    valid_mask = [sid in sequences for sid in site_ids]
    df = df[valid_mask].reset_index(drop=True)
    site_ids = df["site_id"].tolist()
    n = len(site_ids)
    logger.info(f"  {n} sites with sequences")

    # Build hand features (40-dim) -- motif computed for all, loop/struct from v3 cache
    logger.info("Building 40-dim hand features for v4-large...")
    hand_features = build_hand_features(site_ids, sequences, structure_delta, loop_df)
    logger.info(f"  Hand features shape: {hand_features.shape}")

    # One-hot encode sequences
    logger.info("One-hot encoding sequences...")
    base_map = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3}
    onehot_seqs = np.zeros((n, 4, 201), dtype=np.float32)
    for i, sid in enumerate(site_ids):
        seq = sequences[sid].upper().replace("T", "U")
        for j, base in enumerate(seq[:201]):
            if base in base_map:
                onehot_seqs[i, base_map[base], j] = 1.0

    # For v4-large, fold only positives and sites that need structure.
    # Negatives (378K) would take too long; they get zero BPP submatrices.
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

    logger.info(f"  Positive folding complete: {done} sequences in {time.time()-t0:.0f}s")

    # For negatives, compute MFE structures (no BPP, much faster)
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

    logger.info(f"  Negative folding complete: {done} sequences in {time.time()-t0:.0f}s")

    # Encode dot-bracket
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
        "sequences": sequences,
    }


# ---------------------------------------------------------------------------
# Dataset classes
# ---------------------------------------------------------------------------


class EditingDataset(Dataset):
    """Standard dataset for Conv2D_BP and DualPath models."""

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
        seq = self.data["onehot_seqs"][i]  # [4, 201]
        hand_feat = self.data["hand_features"][i]  # [40]
        db = self.data["dot_bracket_list"][i]

        # Extract local window around edit site
        start = CENTER - self.half_w
        end = CENTER + self.half_w + 1
        local_seq = seq[:, start:end]  # [4, window]
        local_db = db[start:end]

        # Build node features (22-dim per node):
        #   base identity (4) + trinucleotide left (4) + trinucleotide right (4)
        #   + is_unpaired (1) + local_unpaired_frac (1) + positional_encoding (4)
        #   + pairing_energy_proxy (1) + distance_from_center (1) + is_center (1)
        #   + struct_channel (1) = 22
        n_nodes = self.window
        node_feats = np.zeros((n_nodes, 22), dtype=np.float32)

        full_seq_arr = seq  # [4, 201]
        for j in range(n_nodes):
            pos = start + j
            # Base identity (4)
            node_feats[j, :4] = full_seq_arr[:, pos] if 0 <= pos < 201 else 0

            # Trinucleotide left context (4)
            if 0 <= pos - 1 < 201:
                node_feats[j, 4:8] = full_seq_arr[:, pos - 1]

            # Trinucleotide right context (4)
            if 0 <= pos + 1 < 201:
                node_feats[j, 8:12] = full_seq_arr[:, pos + 1]

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

            # Pairing energy proxy: 1 if in stem, 0 if unpaired
            node_feats[j, 18] = 0.0 if (j < len(local_db) and local_db[j] == ".") else 1.0

            # Distance from center (normalized)
            node_feats[j, 19] = abs(j - self.half_w) / self.half_w

            # Is center
            node_feats[j, 20] = 1.0 if j == self.half_w else 0.0

            # Structure channel (open=1, close=-1, unpaired=0)
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

        # Base-pair edges from dot-bracket (type 1 = canonical, type 2 for wobble approximation)
        stack = []
        for j, c in enumerate(local_db):
            if c == "(":
                stack.append(j)
            elif c == ")" and stack:
                partner = stack.pop()
                # Determine edge type: check if G-U wobble pair
                pos_j = start + j
                pos_p = start + partner
                base_j = np.argmax(full_seq_arr[:, pos_j]) if 0 <= pos_j < 201 else -1
                base_p = np.argmax(full_seq_arr[:, pos_p]) if 0 <= pos_p < 201 else -1
                # G-U wobble: bases 2,3 or 3,2
                is_wobble = (base_j == 2 and base_p == 3) or (base_j == 3 and base_p == 2)
                etype = 2 if is_wobble else 1
                edge_src.extend([j, partner])
                edge_dst.extend([partner, j])
                edge_type.extend([etype, etype])

        if len(edge_src) == 0:
            # Fallback: self-loop on center
            edge_src = [self.half_w]
            edge_dst = [self.half_w]
            edge_type = [0]

        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        edge_type_tensor = torch.tensor(edge_type, dtype=torch.long)

        return {
            "node_feats": torch.from_numpy(node_feats),       # [41, 22]
            "edge_index": edge_index,                           # [2, E]
            "edge_type": edge_type_tensor,                      # [E]
            "onehot_seq": torch.from_numpy(self.data["onehot_seqs"][i]),  # [4, 201]
            "hand_feat": torch.from_numpy(hand_feat),           # [40]
            "label_binary": torch.tensor(self.data["labels_binary"][i]),
            "label_enzyme": torch.tensor(self.data["labels_enzyme"][i]),
        }


# ---------------------------------------------------------------------------
# Model: Conv2D_BP (same as Phase 1)
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
        self.binary_head = nn.Linear(64, 1)
        self.enzyme_head = nn.Linear(64, 6)

    def forward(self, batch):
        seq_out = self.seq_conv(batch["onehot_seq"]).squeeze(-1)
        seq_out = self.seq_fc(seq_out)
        bp_out = self.bp_conv(batch["bp_submatrix"]).squeeze(-1).squeeze(-1)
        bp_out = self.bp_fc(bp_out)
        hand_out = self.hand_fc(batch["hand_feat"])
        fused = self.fusion(torch.cat([seq_out, bp_out, hand_out], dim=-1))
        return self.binary_head(fused).squeeze(-1), self.enzyme_head(fused)


# ---------------------------------------------------------------------------
# Model: DualPath_Residual (same as Phase 1)
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
        self.binary_head = nn.Linear(64, 1)
        self.enzyme_head = nn.Linear(64, 6)

    def forward(self, batch):
        path_a = self.path_a(batch["hand_feat"])
        path_b = self.path_b_conv(batch["onehot_seq"]).squeeze(-1)
        path_b = self.path_b_fc(path_b)
        fused = self.fusion(torch.cat([path_a, path_b], dim=-1)) + path_a
        return self.binary_head(fused).squeeze(-1), self.enzyme_head(fused)


# ---------------------------------------------------------------------------
# Model 6: AdarEdit-adapted GAT
# ---------------------------------------------------------------------------


class ManualGATLayer(nn.Module):
    """Manual Graph Attention layer (no torch_geometric dependency for forward pass).

    Uses multi-head attention over adjacency defined by edge_index.
    """

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
        """
        x: [N, in_dim]
        edge_index: [2, E]
        edge_type: [E]
        Returns: [N, out_dim]
        """
        N = x.size(0)
        h = self.W(x)  # [N, out_dim]
        h_heads = h.view(N, self.n_heads, self.head_dim)  # [N, heads, head_dim]

        src, dst = edge_index[0], edge_index[1]
        E = src.size(0)

        # Compute attention scores
        h_src = h_heads[src]  # [E, heads, head_dim]
        h_dst = h_heads[dst]  # [E, heads, head_dim]
        e_embed = self.edge_embed(edge_type)  # [E, edge_embed_dim]

        # Attention: a_src * h_src + a_dst * h_dst + a_edge * e_embed
        attn = (h_src * self.a_src.unsqueeze(0)).sum(-1) + \
               (h_dst * self.a_dst.unsqueeze(0)).sum(-1) + \
               (e_embed.unsqueeze(1) * self.a_edge.unsqueeze(0)).sum(-1)  # [E, heads]
        attn = self.leaky_relu(attn)

        # Softmax over neighbors
        # Build sparse attention using scatter
        attn_max = torch.zeros(N, self.n_heads, device=x.device).fill_(-1e9)
        attn_max.scatter_reduce_(0, dst.unsqueeze(1).expand(-1, self.n_heads), attn, reduce="amax")
        attn = attn - attn_max[dst]
        attn_exp = torch.exp(attn)
        attn_sum = torch.zeros(N, self.n_heads, device=x.device)
        attn_sum.scatter_add_(0, dst.unsqueeze(1).expand(-1, self.n_heads), attn_exp)
        attn_norm = attn_exp / (attn_sum[dst] + 1e-10)
        attn_norm = self.dropout(attn_norm)

        # Aggregate
        weighted = h_src * attn_norm.unsqueeze(-1)  # [E, heads, head_dim]
        out = torch.zeros(N, self.n_heads, self.head_dim, device=x.device)
        out.scatter_add_(0, dst.unsqueeze(1).unsqueeze(2).expand(-1, self.n_heads, self.head_dim), weighted)

        out = out.reshape(N, -1)  # [N, out_dim]
        out = self.norm(out + h)  # residual + layer norm
        return out


class AdarEditGAT(nn.Module):
    """AdarEdit-adapted GAT for C-to-U editing prediction.

    Stream A: GAT on RNA structure graph (41-node local window)
      - 22-dim node features -> 64-dim via projection
      - 3 GAT layers with 4 heads
      - Extract center node embedding -> 64-dim

    Stream B: Parallel 1D CNN (3-mer + 5-mer filters)
      - One-hot sequence [4, 201]
      - Two parallel Conv1D branches -> concat -> pool -> 64-dim

    Stream C: Hand features (40-dim)

    Fusion: [64 + 64 + 40] -> 168 -> 64 -> heads
    """

    def __init__(self, node_feat_dim=22, gat_hidden=64, n_heads=4, n_gat_layers=3,
                 cnn_channels=32, hand_dim=40, n_enzymes=6):
        super().__init__()

        # Stream A: GAT
        self.node_proj = nn.Linear(node_feat_dim, gat_hidden)
        self.gat_layers = nn.ModuleList()
        for _ in range(n_gat_layers):
            self.gat_layers.append(ManualGATLayer(gat_hidden, gat_hidden, n_heads=n_heads, dropout=0.1))
        self.gat_out = nn.Linear(gat_hidden, gat_hidden)

        # Stream B: Parallel CNN
        self.cnn_k3 = nn.Sequential(
            nn.Conv1d(4, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_channels),
        )
        self.cnn_k5 = nn.Sequential(
            nn.Conv1d(4, cnn_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_channels),
        )
        self.cnn_merge = nn.Sequential(
            nn.Conv1d(cnn_channels * 2, gat_hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(gat_hidden),
            nn.AdaptiveMaxPool1d(1),
        )
        self.cnn_out = nn.Linear(gat_hidden, gat_hidden)

        # Fusion
        fusion_dim = gat_hidden + gat_hidden + hand_dim  # 64+64+40 = 168
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, gat_hidden),
            nn.GELU(),
            nn.Dropout(0.3),
        )
        self.binary_head = nn.Linear(gat_hidden, 1)
        self.enzyme_head = nn.Linear(gat_hidden, n_enzymes)

    def forward(self, batch):
        # Stream A: GAT
        node_feats = batch["node_feats"]  # [B, 41, 22]
        edge_index = batch["edge_index"]  # list of [2, E_i] or batched
        edge_type = batch["edge_type"]    # list of [E_i] or batched

        B = node_feats.size(0)
        n_nodes = node_feats.size(1)
        center_node = n_nodes // 2

        # Process each sample individually (variable edge counts)
        gat_embeddings = []
        for b in range(B):
            x = self.node_proj(node_feats[b])  # [41, 64]
            ei = edge_index[b].to(x.device)    # [2, E]
            et = edge_type[b].to(x.device)     # [E]
            for gat_layer in self.gat_layers:
                x = F.gelu(gat_layer(x, ei, et))
            gat_embeddings.append(x[center_node])  # [64]

        gat_out = torch.stack(gat_embeddings)  # [B, 64]
        gat_out = self.gat_out(gat_out)

        # Stream B: Parallel CNN
        seq = batch["onehot_seq"]  # [B, 4, 201]
        cnn_3 = self.cnn_k3(seq)   # [B, 32, 201]
        cnn_5 = self.cnn_k5(seq)   # [B, 32, 201]
        cnn_cat = torch.cat([cnn_3, cnn_5], dim=1)  # [B, 64, 201]
        cnn_pooled = self.cnn_merge(cnn_cat).squeeze(-1)  # [B, 64]
        cnn_out = self.cnn_out(cnn_pooled)

        # Stream C: Hand features
        hand_feat = batch["hand_feat"]  # [B, 40]

        # Fusion
        fused = torch.cat([gat_out, cnn_out, hand_feat], dim=-1)  # [B, 168]
        fused = self.fusion(fused)
        return self.binary_head(fused).squeeze(-1), self.enzyme_head(fused)


# ---------------------------------------------------------------------------
# Collate functions
# ---------------------------------------------------------------------------


def collate_fn(batch_list):
    """Standard collate for Conv2D/DualPath."""
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
# Training with class weighting / focal loss
# ---------------------------------------------------------------------------


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        focal_weight = alpha_t * (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()


def compute_class_weight(labels):
    """Compute pos_weight for BCEWithLogitsLoss from label array."""
    n_pos = (labels == 1).sum()
    n_neg = (labels == 0).sum()
    if n_pos == 0:
        return 1.0
    return float(n_neg / n_pos)


def train_one_epoch(model, loader, optimizer, binary_loss_fn, is_gat=False):
    """Train one epoch with custom binary loss function."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        if is_gat:
            # Move non-list items to device
            batch_device = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch_device[k] = v.to(DEVICE)
                else:
                    batch_device[k] = v  # lists stay on CPU, moved in forward
            batch = batch_device
        else:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

        optimizer.zero_grad()
        binary_logit, enzyme_logit = model(batch)
        loss_binary = binary_loss_fn(binary_logit, batch["label_binary"])
        loss_enzyme = F.cross_entropy(enzyme_logit, batch["label_enzyme"])
        loss = loss_binary + ENZYME_LOSS_WEIGHT * loss_enzyme
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, is_gat=False):
    """Evaluate model."""
    model.eval()
    all_probs, all_enzyme, all_lbl_b, all_lbl_e = [], [], [], []

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

        binary_logit, enzyme_logit = model(batch)
        all_probs.append(torch.sigmoid(binary_logit).cpu().numpy())
        all_enzyme.append(enzyme_logit.cpu().numpy())
        all_lbl_b.append(batch["label_binary"].cpu().numpy())
        all_lbl_e.append(batch["label_enzyme"].cpu().numpy())

    probs = np.concatenate(all_probs)
    enzyme_logits = np.concatenate(all_enzyme)
    lbl_b = np.concatenate(all_lbl_b)
    lbl_e = np.concatenate(all_lbl_e)

    try:
        auroc = roc_auc_score(lbl_b, probs)
    except ValueError:
        auroc = 0.5
    enzyme_acc = accuracy_score(lbl_e, enzyme_logits.argmax(axis=1))
    return {"auroc": auroc, "enzyme_acc": enzyme_acc}


# ---------------------------------------------------------------------------
# Model training orchestrator
# ---------------------------------------------------------------------------


def create_model(model_name):
    """Create a fresh model instance by name."""
    if model_name == "Conv2D_BP":
        return Conv2DBPModel()
    elif model_name == "DualPath_Residual":
        return DualPathResidual()
    elif model_name == "AdarEdit_GAT":
        return AdarEditGAT()
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_model_fold(model_name, data, train_idx, val_idx, fold,
                     batch_size, n_epochs, lr, use_focal=False, pos_weight=None):
    """Train a model for one fold."""
    is_gat = model_name == "AdarEdit_GAT"
    model = create_model(model_name).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if fold == 0:
        logger.info(f"  Model parameters: {n_params:,}")

    # Build loss function
    if use_focal:
        binary_loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
    elif pos_weight is not None:
        pw = torch.tensor([pos_weight], device=DEVICE)
        binary_loss_fn = lambda logits, targets: F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pw.expand_as(logits)
        )
    else:
        binary_loss_fn = F.binary_cross_entropy_with_logits

    n_loader_workers = 0 if DEVICE.type == "mps" else 4

    if is_gat:
        train_ds = GATDataset(train_idx, data)
        val_ds = GATDataset(val_idx, data)
        cfn = gat_collate_fn
    else:
        train_ds = EditingDataset(train_idx, data)
        val_ds = EditingDataset(val_idx, data)
        cfn = collate_fn

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=cfn, num_workers=n_loader_workers, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=cfn, num_workers=n_loader_workers, pin_memory=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=V4_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    history = {"train_loss": [], "val_auroc": [], "val_enzyme_acc": []}
    best_auroc = 0.0
    best_state = None

    for epoch in range(n_epochs):
        loss = train_one_epoch(model, train_loader, optimizer, binary_loss_fn, is_gat=is_gat)
        scheduler.step()
        metrics = evaluate(model, val_loader, is_gat=is_gat)

        history["train_loss"].append(loss)
        history["val_auroc"].append(metrics["auroc"])
        history["val_enzyme_acc"].append(metrics["enzyme_acc"])

        if metrics["auroc"] > best_auroc:
            best_auroc = metrics["auroc"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 5 == 0:
            logger.info(
                f"  [{model_name}] Fold {fold} Epoch {epoch+1}/{n_epochs}: "
                f"loss={loss:.4f}, AUROC={metrics['auroc']:.4f}, "
                f"enzyme_acc={metrics['enzyme_acc']:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history, best_auroc, n_params


# ---------------------------------------------------------------------------
# XGBoost training
# ---------------------------------------------------------------------------


def train_xgboost(data, dataset_label="v3"):
    """Train XGBoost on hand features with 5-fold CV."""
    from xgboost import XGBClassifier

    logger.info(f"\n{'='*60}")
    logger.info(f"Training XGBoost on {dataset_label}")
    logger.info(f"{'='*60}")

    X = data["hand_features"]
    y = data["labels_binary"]
    n = len(y)

    # Compute scale_pos_weight for imbalance
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    spw = float(n_neg / n_pos) if n_pos > 0 else 1.0
    logger.info(f"  N={n}, pos={n_pos}, neg={n_neg}, scale_pos_weight={spw:.2f}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_aurocs = []
    fold_enzyme_accs = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.arange(n), y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        clf = XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=spw,
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=SEED,
            tree_method="hist",
            n_jobs=-1,
        )
        clf.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        probs = clf.predict_proba(X_val)[:, 1]
        try:
            auroc = roc_auc_score(y_val, probs)
        except ValueError:
            auroc = 0.5

        # Enzyme accuracy (train separate multi-class)
        enzyme_train = data["labels_enzyme"][train_idx]
        enzyme_val = data["labels_enzyme"][val_idx]
        from xgboost import XGBClassifier as XGBMCls
        enz_clf = XGBMCls(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            objective="multi:softmax",
            num_class=6,
            use_label_encoder=False,
            random_state=SEED,
            tree_method="hist",
            n_jobs=-1,
        )
        enz_clf.fit(X_train, enzyme_train, verbose=False)
        enz_pred = enz_clf.predict(X_val)
        enz_acc = accuracy_score(enzyme_val, enz_pred)

        fold_aurocs.append(auroc)
        fold_enzyme_accs.append(enz_acc)
        logger.info(f"  XGBoost Fold {fold_idx}: AUROC={auroc:.4f}, enzyme_acc={enz_acc:.4f}")

    result = {
        "fold_aurocs": [float(x) for x in fold_aurocs],
        "fold_enzyme_accs": [float(x) for x in fold_enzyme_accs],
        "mean_auroc": float(np.mean(fold_aurocs)),
        "std_auroc": float(np.std(fold_aurocs)),
        "mean_enzyme_acc": float(np.mean(fold_enzyme_accs)),
        "std_enzyme_acc": float(np.std(fold_enzyme_accs)),
        "n_params": "N/A (tree-based)",
    }
    logger.info(
        f"  XGBoost {dataset_label} SUMMARY: AUROC={result['mean_auroc']:.4f} "
        f"+/- {result['std_auroc']:.4f}, enzyme_acc={result['mean_enzyme_acc']:.4f}"
    )
    return result


# ---------------------------------------------------------------------------
# Run a deep model across 5 folds
# ---------------------------------------------------------------------------


def run_model_cv(model_name, data, dataset_label, batch_size, n_epochs, lr,
                 use_focal=False, pos_weight=None):
    """Run 5-fold CV for a deep model."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Training {model_name} on {dataset_label}")
    logger.info(f"{'='*60}")

    n = len(data["site_ids"])
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = list(skf.split(np.arange(n), data["labels_binary"]))

    fold_aurocs = []
    fold_enzyme_accs = []
    fold_histories = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        logger.info(f"\n--- {model_name} Fold {fold_idx} ---")
        t0 = time.time()

        model, history, best_auroc, n_params = train_model_fold(
            model_name, data, list(train_idx), list(val_idx), fold_idx,
            batch_size=batch_size, n_epochs=n_epochs, lr=lr,
            use_focal=use_focal, pos_weight=pos_weight,
        )

        # Final evaluation
        is_gat = model_name == "AdarEdit_GAT"
        n_loader_workers = 0 if DEVICE.type == "mps" else 4
        if is_gat:
            val_ds = GATDataset(list(val_idx), data)
            cfn = gat_collate_fn
        else:
            val_ds = EditingDataset(list(val_idx), data)
            cfn = collate_fn
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                collate_fn=cfn, num_workers=n_loader_workers)
        final_metrics = evaluate(model, val_loader, is_gat=is_gat)

        elapsed = time.time() - t0
        logger.info(
            f"  Fold {fold_idx} done in {elapsed:.0f}s: "
            f"AUROC={final_metrics['auroc']:.4f}, enzyme_acc={final_metrics['enzyme_acc']:.4f}"
        )

        fold_aurocs.append(final_metrics["auroc"])
        fold_enzyme_accs.append(final_metrics["enzyme_acc"])
        fold_histories.append(history)

        # Save model
        save_name = f"{model_name}_{dataset_label}_fold{fold_idx}.pt"
        torch.save(model.state_dict(), OUTPUT_DIR / save_name)

        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    result = {
        "fold_aurocs": [float(x) for x in fold_aurocs],
        "fold_enzyme_accs": [float(x) for x in fold_enzyme_accs],
        "mean_auroc": float(np.mean(fold_aurocs)),
        "std_auroc": float(np.std(fold_aurocs)),
        "mean_enzyme_acc": float(np.mean(fold_enzyme_accs)),
        "std_enzyme_acc": float(np.std(fold_enzyme_accs)),
        "n_params": n_params,
    }

    logger.info(
        f"\n{model_name} {dataset_label} SUMMARY: "
        f"AUROC={result['mean_auroc']:.4f} +/- {result['std_auroc']:.4f}, "
        f"enzyme_acc={result['mean_enzyme_acc']:.4f} +/- {result['std_enzyme_acc']:.4f}"
    )

    return result, fold_histories


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_phase2_comparison(all_results, output_dir):
    """Plot comprehensive comparison: Phase 1 v3 vs Phase 2 v4-large + GAT."""
    fig, ax = plt.subplots(figsize=(14, 7))

    # Group results
    model_labels = []
    aurocs = []
    stds = []
    colors_list = []

    color_map = {
        "v3_phase1": "#90CAF9",    # light blue
        "v3_phase2": "#2196F3",    # blue
        "v4_large": "#FF9800",     # orange
    }

    for key, res in sorted(all_results.items()):
        model_labels.append(key.replace("_", "\n"))
        aurocs.append(res["mean_auroc"])
        stds.append(res["std_auroc"])
        if "v4" in key.lower():
            colors_list.append(color_map["v4_large"])
        elif "phase1" in key.lower() or "v3_P1" in key:
            colors_list.append(color_map["v3_phase1"])
        else:
            colors_list.append(color_map["v3_phase2"])

    x = np.arange(len(model_labels))
    bars = ax.bar(x, aurocs, yerr=stds, capsize=4,
                  color=colors_list, edgecolor="black", linewidth=0.5, width=0.7)

    for bar, mean, std in zip(bars, aurocs, stds):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + std + 0.003,
                f"{mean:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, fontsize=8)
    ax.set_ylabel("Binary AUROC", fontsize=12)
    ax.set_title("Phase 2: Deep Architecture Comparison (v3 vs v4-large)", fontsize=14)
    ax.set_ylim(0.5, 1.0)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color_map["v3_phase1"], label="v3 (Phase 1)"),
        Patch(facecolor=color_map["v3_phase2"], label="v3 (Phase 2)"),
        Patch(facecolor=color_map["v4_large"], label="v4-large"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    plt.savefig(output_dir / "phase2_comparison.png", dpi=150)
    plt.close()
    logger.info(f"  Saved phase2_comparison.png")


def plot_training_curves(all_histories, output_dir):
    """Plot per-model training curves."""
    for model_key, fold_histories in all_histories.items():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        for fold_idx, hist in enumerate(fold_histories):
            ax1.plot(hist["train_loss"], alpha=0.6, label=f"Fold {fold_idx}")
            ax2.plot(hist["val_auroc"], alpha=0.6, label=f"Fold {fold_idx}")
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("Training Loss")
        ax1.set_title(f"{model_key} - Training Loss"); ax1.legend(fontsize=8)
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("Validation AUROC")
        ax2.set_title(f"{model_key} - Validation AUROC"); ax2.legend(fontsize=8)
        plt.tight_layout()
        safe_name = model_key.replace(" ", "_").replace("/", "_")
        plt.savefig(output_dir / f"training_curves_{safe_name}.png", dpi=120)
        plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    all_results = {}
    all_histories = {}

    # Load Phase 1 results for comparison
    phase1_path = OUTPUT_DIR / "architecture_comparison.json"
    if phase1_path.exists():
        with open(phase1_path) as f:
            phase1_results = json.load(f)
        for model_name in ["Conv2D_BP", "DualPath_Residual"]:
            if model_name in phase1_results:
                key = f"{model_name}_v3_P1"
                all_results[key] = phase1_results[model_name]
        logger.info(f"Loaded Phase 1 results: {list(phase1_results.keys())}")

    # =====================================================================
    # TASK 1: Retrain Conv2D_BP and DualPath_Residual on v4-large
    # =====================================================================
    logger.info("\n" + "=" * 70)
    logger.info("TASK 1: Retrain top models on v4-large (385K sites, 1:50 ratio)")
    logger.info("=" * 70)

    v4_data = load_v4_data()
    n_v4 = len(v4_data["site_ids"])
    pos_weight_v4 = compute_class_weight(v4_data["labels_binary"])
    logger.info(f"v4-large: {n_v4} sites, pos_weight={pos_weight_v4:.2f}")

    # Conv2D_BP on v4-large with focal loss
    result, histories = run_model_cv(
        "Conv2D_BP", v4_data, "v4large",
        batch_size=V4_BATCH_SIZE, n_epochs=V4_N_EPOCHS, lr=V4_LR,
        use_focal=True,
    )
    all_results["Conv2D_BP_v4large"] = result
    all_histories["Conv2D_BP_v4large"] = histories

    # DualPath_Residual on v4-large with focal loss
    result, histories = run_model_cv(
        "DualPath_Residual", v4_data, "v4large",
        batch_size=V4_BATCH_SIZE, n_epochs=V4_N_EPOCHS, lr=V4_LR,
        use_focal=True,
    )
    all_results["DualPath_Residual_v4large"] = result
    all_histories["DualPath_Residual_v4large"] = histories

    # =====================================================================
    # TASK 2a: XGBoost on v4-large
    # =====================================================================
    logger.info("\n" + "=" * 70)
    logger.info("TASK 2a: XGBoost on v4-large")
    logger.info("=" * 70)

    xgb_v4_result = train_xgboost(v4_data, "v4large")
    all_results["XGBoost_v4large"] = xgb_v4_result

    # Free v4-large memory before loading v3
    logger.info("Freeing v4-large data from memory...")
    del v4_data
    import gc
    gc.collect()

    # =====================================================================
    # TASK 2b: XGBoost on v3 (for fair comparison)
    # =====================================================================
    logger.info("\n" + "=" * 70)
    logger.info("TASK 2b: XGBoost on v3 + AdarEdit GAT on v3 and v4-large")
    logger.info("=" * 70)

    v3_data = load_v3_data()
    n_v3 = len(v3_data["site_ids"])
    logger.info(f"v3: {n_v3} sites")

    xgb_v3_result = train_xgboost(v3_data, "v3")
    all_results["XGBoost_v3"] = xgb_v3_result

    # =====================================================================
    # TASK 3: AdarEdit GAT on v3
    # =====================================================================
    logger.info("\n" + "=" * 70)
    logger.info("TASK 3: AdarEdit GAT on v3")
    logger.info("=" * 70)

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
    # TASK 3b: AdarEdit GAT on v4-large
    # =====================================================================
    logger.info("\n" + "=" * 70)
    logger.info("TASK 3b: AdarEdit GAT on v4-large")
    logger.info("=" * 70)

    v4_data = load_v4_data()
    result, histories = run_model_cv(
        "AdarEdit_GAT", v4_data, "v4large",
        batch_size=V4_BATCH_SIZE, n_epochs=V4_N_EPOCHS, lr=V4_LR,
        use_focal=True,
    )
    all_results["AdarEdit_GAT_v4large"] = result
    all_histories["AdarEdit_GAT_v4large"] = histories

    del v4_data
    gc.collect()

    # =====================================================================
    # TASK 4: Save and compare all results
    # =====================================================================
    logger.info("\n" + "=" * 70)
    logger.info("FINAL RESULTS COMPARISON")
    logger.info("=" * 70)

    # Save results
    with open(OUTPUT_DIR / "phase2_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Saved results to {OUTPUT_DIR / 'phase2_results.json'}")

    # Print comparison table
    logger.info("\n" + "-" * 100)
    logger.info(f"{'Model':<35} {'Dataset':<10} {'AUROC':>16} {'Enzyme Acc':>16} {'Params':>12}")
    logger.info("-" * 100)

    # Sort for readability
    order = [
        "Conv2D_BP_v3_P1", "Conv2D_BP_v4large",
        "DualPath_Residual_v3_P1", "DualPath_Residual_v4large",
        "XGBoost_v3", "XGBoost_v4large",
        "AdarEdit_GAT_v3", "AdarEdit_GAT_v4large",
    ]
    for key in order:
        if key not in all_results:
            continue
        r = all_results[key]
        parts = key.rsplit("_", 1)
        model = key
        ds = "v4-large" if "v4" in key else "v3"
        n_params_str = f"{r['n_params']:>10,}" if isinstance(r.get("n_params"), int) else str(r.get("n_params", "?"))
        logger.info(
            f"{model:<35} {ds:<10} "
            f"{r['mean_auroc']:.4f}+/-{r['std_auroc']:.4f}   "
            f"{r['mean_enzyme_acc']:.4f}+/-{r['std_enzyme_acc']:.4f}   "
            f"{n_params_str}"
        )
    logger.info("-" * 100)

    # Key comparisons
    logger.info("\n--- KEY COMPARISONS ---")
    for model_base in ["Conv2D_BP", "DualPath_Residual", "XGBoost", "AdarEdit_GAT"]:
        v3_key = f"{model_base}_v3_P1" if f"{model_base}_v3_P1" in all_results else f"{model_base}_v3"
        v4_key = f"{model_base}_v4large"
        if v3_key in all_results and v4_key in all_results:
            v3_auroc = all_results[v3_key]["mean_auroc"]
            v4_auroc = all_results[v4_key]["mean_auroc"]
            delta = v4_auroc - v3_auroc
            direction = "+" if delta > 0 else ""
            logger.info(f"  {model_base}: v3={v3_auroc:.4f} -> v4-large={v4_auroc:.4f} ({direction}{delta:.4f})")

    # Plotting
    plot_phase2_comparison(all_results, OUTPUT_DIR)
    plot_training_curves(all_histories, OUTPUT_DIR)

    logger.info(f"\nAll outputs saved to {OUTPUT_DIR}")
    logger.info(f"Log file: {log_file}")
    logger.info("Phase 2 complete!")


if __name__ == "__main__":
    main()
