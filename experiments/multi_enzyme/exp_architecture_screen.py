#!/usr/bin/env python
"""Comprehensive architecture screen for RNA editing prediction.

Evaluates 8 architectures on the v3 multi-enzyme dataset:
  A1. Phase3Baseline     — [rnafm|delta|hand] concat → MLP
  A2. NoDelta            — [rnafm|hand] concat → MLP (no edit delta)
  A3. AdarEditGAT        — GAT on RNA structure graph + global features
  A4a. RNAFMLoRA         — RNA-FM with LoRA fine-tuning
  A4b. RNAFMLoRA+Feat    — RNA-FM LoRA + hand features
  A5. GatedFusion        — hand features gate RNA-FM, RNA-FM gates delta
  A6. Conv2DBP           — Conv2D on 41×41 BP probability matrix + global
  A8. HierarchicalAttn   — local transformer on BP + cross-attention to RNA-FM

Each architecture is trained with 2-stage training (joint → adapter-only),
evaluated with 2-fold stratified CV, and scored on overall + per-enzyme AUROC.

Output: experiments/multi_enzyme/outputs/architecture_screen/
Run: /opt/miniconda3/envs/quris/bin/python -u experiments/multi_enzyme/exp_architecture_screen.py
"""

import gc
import json
import logging
import os
import sys
import time
import traceback
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
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
        logging.FileHandler(OUTPUT_DIR / "architecture_screen.log", mode="w"),
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
BATCH_SIZE_LORA = 16

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
# ViennaRNA BP matrix computation (parallel)
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
        sub = bpp[80:121, 80:121].copy()
        for r in range(41):
            for c in range(41):
                if abs(r - c) < 3:
                    sub[r, c] = 0.0
        return site_id, sub
    except Exception:
        return site_id, np.zeros((41, 41), dtype=np.float32)


def _fold_dotbracket_worker(args):
    """Worker: returns (site_id, dot_bracket_string)."""
    import RNA
    site_id, seq = args
    try:
        seq = seq.upper().replace("T", "U")
        fc = RNA.fold_compound(seq)
        struct, _ = fc.mfe()
        return site_id, struct
    except Exception:
        return site_id, "." * len(seq)


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------


def load_data() -> Dict:
    """Load v3 dataset with all pre-computed features."""
    logger.info("=" * 70)
    logger.info("Loading V3 dataset + pre-computed embeddings")
    logger.info("=" * 70)

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
    rnafm_emb = torch.load(rnafm_path, map_location="cpu", weights_only=False)
    logger.info(f"  Loaded RNA-FM embeddings: {len(rnafm_emb)} sites")

    # --- RNA-FM edited embeddings (640-dim) ---
    rnafm_edited_path = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "embeddings" / "rnafm_pooled_edited_v3.pt"
    rnafm_edited_emb = torch.load(rnafm_edited_path, map_location="cpu", weights_only=False)
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
    logger.info(f"  RNA-FM coverage: {n_rnafm_found}/{n} ({100 * n_rnafm_found / n:.1f}%)")

    del rnafm_emb, rnafm_edited_emb
    gc.collect()

    # --- Compute Conv2D BP submatrices (41x41 per site) via ViennaRNA ---
    bp_cache_path = OUTPUT_DIR / "bp_submatrices_v3.npz"
    if bp_cache_path.exists():
        logger.info(f"  Loading cached BP submatrices from {bp_cache_path}")
        bp_data = np.load(bp_cache_path, allow_pickle=True)
        bp_submatrices = bp_data["bp_submatrices"]
        bp_site_ids = list(bp_data["site_ids"])
        # Align to current site_ids
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
        logger.info("  Computing ViennaRNA BP probability submatrices (this takes ~30-60 min)...")
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
                    logger.info(f"    Folded {done}/{n} ({time.time() - t0:.0f}s)")
        logger.info(f"  Folding complete: {done} sequences in {time.time() - t0:.0f}s")

        # Cache for reuse
        np.savez_compressed(bp_cache_path,
                            bp_submatrices=bp_submatrices,
                            site_ids=np.array(site_ids, dtype=object))
        logger.info(f"  Saved BP cache to {bp_cache_path}")

    # --- Compute dot-bracket structures for GAT ---
    db_cache_path = OUTPUT_DIR / "dot_brackets_v3.json"
    if db_cache_path.exists():
        logger.info(f"  Loading cached dot-bracket structures from {db_cache_path}")
        with open(db_cache_path) as f:
            dot_brackets = json.load(f)
        logger.info(f"  Loaded {len(dot_brackets)} dot-bracket structures")
    else:
        # First check loop_df for dot_bracket column
        dot_brackets = {}
        if "dot_bracket" in loop_df.columns:
            for sid in site_ids:
                if sid in loop_df.index:
                    db = loop_df.loc[sid, "dot_bracket"]
                    if isinstance(db, str) and len(db) == 201:
                        dot_brackets[sid] = db

        missing = [sid for sid in site_ids if sid not in dot_brackets]
        if missing:
            logger.info(f"  Computing dot-bracket for {len(missing)} sites via ViennaRNA...")
            work_items = [(sid, sequences[sid]) for sid in missing if sid in sequences]
            n_workers = min(14, os.cpu_count() or 4)
            t0 = time.time()
            done = 0
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {executor.submit(_fold_dotbracket_worker, item): item[0] for item in work_items}
                for future in as_completed(futures):
                    sid, db = future.result()
                    dot_brackets[sid] = db
                    done += 1
                    if done % 3000 == 0:
                        logger.info(f"    Folded {done}/{len(missing)} ({time.time() - t0:.0f}s)")
            logger.info(f"  Folding complete: {done} structures in {time.time() - t0:.0f}s")

        with open(db_cache_path, "w") as f:
            json.dump(dot_brackets, f)
        logger.info(f"  Saved dot-bracket cache to {db_cache_path}")

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

    # --- Raw sequences for LoRA models ---
    raw_sequences = [sequences.get(sid, "N" * 201) for sid in site_ids]

    logger.info("Data loading complete.")
    return {
        "site_ids": site_ids,
        "df": df,
        "sequences": sequences,
        "raw_sequences": raw_sequences,
        "hand_features": hand_features,
        "rnafm_features": rnafm_matrix,
        "edit_delta_features": edit_delta_matrix,
        "bp_submatrices": bp_submatrices,
        "dot_brackets": dot_brackets,
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


class LoRADataset(Dataset):
    """Dataset that also provides raw sequences for RNA-FM tokenization."""

    def __init__(self, indices, data):
        self.indices = indices
        self.data = data

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        return {
            "raw_seq": self.data["raw_sequences"][i],
            "hand_feat": torch.from_numpy(self.data["hand_features"][i]),
            "label_binary": torch.tensor(self.data["labels_binary"][i]),
            "label_enzyme": torch.tensor(self.data["labels_enzyme"][i]),
            "per_enzyme_labels": torch.from_numpy(self.data["per_enzyme_labels"][i]),
            "index": i,
        }


class GATDataset(Dataset):
    """Dataset for GAT architecture with graph construction."""

    def __init__(self, indices, data):
        self.indices = indices
        self.data = data

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        sid = self.data["site_ids"][i]
        return {
            "rnafm": torch.from_numpy(self.data["rnafm_features"][i]),
            "hand_feat": torch.from_numpy(self.data["hand_features"][i]),
            "dot_bracket": self.data["dot_brackets"].get(sid, "." * 201),
            "raw_seq": self.data["raw_sequences"][i],
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
# Base model mixin with shared heads
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
        """shared: [B, 128] → binary_logit, per_enzyme_logits, enzyme_cls_logits."""
        binary_logit = self.binary_head(shared).squeeze(-1)
        per_enzyme_logits = [
            self.enzyme_adapters[enz](shared).squeeze(-1)
            for enz in PER_ENZYME_HEADS
        ]
        enzyme_cls_logits = self.enzyme_classifier(shared)
        return binary_logit, per_enzyme_logits, enzyme_cls_logits


# ---------------------------------------------------------------------------
# A1. Phase3Baseline
# ---------------------------------------------------------------------------


class Phase3Baseline(nn.Module, HeadsMixin):
    """[rnafm(640) | delta(640) | hand(40)] = 1320 → MLP → heads."""

    name = "A1_Phase3Baseline"

    def __init__(self):
        super().__init__()
        d_in = D_RNAFM + D_EDIT_DELTA + D_HAND  # 1320
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


# ---------------------------------------------------------------------------
# A2. NoDelta
# ---------------------------------------------------------------------------


class NoDelta(nn.Module, HeadsMixin):
    """[rnafm(640) | hand(40)] = 680 → MLP → heads."""

    name = "A2_NoDelta"

    def __init__(self):
        super().__init__()
        d_in = D_RNAFM + D_HAND  # 680
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


# ---------------------------------------------------------------------------
# A3. AdarEditGAT (using torch_geometric)
# ---------------------------------------------------------------------------


def build_graph_from_dotbracket(dot_bracket: str, seq: str):
    """Build edge_index from dot-bracket: backbone + base-pair edges.

    Returns edge_index [2, E] and node features [N, 21] (4 one-hot + 1 is_center + 16 pos_enc).
    """
    n = len(dot_bracket)
    edges_src, edges_dst = [], []

    # Backbone edges (i, i+1)
    for i in range(n - 1):
        edges_src.extend([i, i + 1])
        edges_dst.extend([i + 1, i])

    # Base-pair edges from dot-bracket
    stack = []
    for i, c in enumerate(dot_bracket):
        if c == "(":
            stack.append(i)
        elif c == ")" and stack:
            j = stack.pop()
            edges_src.extend([i, j])
            edges_dst.extend([j, i])

    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)

    # Node features: one-hot nucleotide (4) + is_center (1) + sinusoidal position (16)
    base_map = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3, "N": 0}
    seq_upper = seq.upper().replace("T", "U")
    node_feat = torch.zeros(n, 21, dtype=torch.float32)
    for i in range(n):
        b = base_map.get(seq_upper[i], 0)
        node_feat[i, b] = 1.0
    node_feat[CENTER, 4] = 1.0  # is_center

    # Sinusoidal position encoding (16-dim)
    positions = torch.arange(n, dtype=torch.float32)
    div_term = torch.exp(torch.arange(0, 16, 2, dtype=torch.float32) * (-math.log(10000.0) / 16))
    node_feat[:, 5::2] = torch.sin(positions.unsqueeze(1) * div_term)
    node_feat[:, 6::2] = torch.cos(positions.unsqueeze(1) * div_term)

    return node_feat, edge_index


try:
    from torch_geometric.nn import GATConv, global_add_pool
    from torch_geometric.data import Data, Batch as PyGBatch
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


class AdarEditGAT(nn.Module, HeadsMixin):
    """GAT on RNA structure graph + global RNA-FM and hand features."""

    name = "A3_AdarEditGAT"

    def __init__(self):
        super().__init__()
        if not HAS_PYG:
            raise ImportError("torch_geometric required for A3_AdarEditGAT")

        d_node = 21  # 4 one-hot + 1 is_center + 16 pos_enc
        self.gat1 = GATConv(d_node, 64, heads=4, concat=False, dropout=0.1)
        self.gat2 = GATConv(64, 64, heads=4, concat=False, dropout=0.1)
        self.gat3 = GATConv(64, 64, heads=4, concat=False, dropout=0.1)

        # Attention-weighted pooling
        self.pool_attn = nn.Linear(64, 1)

        # Fusion: graph(64) + rnafm(640) + hand(40) = 744
        d_fused = 64 + D_RNAFM + D_HAND
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
        params = (
            list(self.gat1.parameters()) + list(self.gat2.parameters()) +
            list(self.gat3.parameters()) + list(self.pool_attn.parameters()) +
            list(self.encoder.parameters())
        )
        return params

    def get_adapter_params(self):
        params = list(self.binary_head.parameters())
        for enz in PER_ENZYME_HEADS:
            params.extend(self.enzyme_adapters[enz].parameters())
        params.extend(self.enzyme_classifier.parameters())
        return params

    def forward(self, batch):
        # batch is a dict with "pyg_batch" (PyG Batch), "rnafm", "hand_feat", labels
        pyg = batch["pyg_batch"]
        x, edge_index, batch_idx = pyg.x, pyg.edge_index, pyg.batch

        x = F.gelu(self.gat1(x, edge_index))
        x = F.gelu(self.gat2(x, edge_index))
        x = F.gelu(self.gat3(x, edge_index))

        # Attention-weighted pooling
        attn_w = torch.softmax(self.pool_attn(x), dim=0)
        # Scatter: weighted sum per graph
        num_graphs = batch_idx.max().item() + 1
        pooled = torch.zeros(num_graphs, 64, device=x.device)
        weighted = x * attn_w
        pooled.scatter_add_(0, batch_idx.unsqueeze(1).expand_as(weighted), weighted)

        fused = torch.cat([pooled, batch["rnafm"], batch["hand_feat"]], dim=-1)
        shared = self.encoder(fused)
        return self._apply_heads(shared)


def gat_collate(batch_list):
    """Custom collate for GAT: builds PyG batch from dot-bracket structures."""
    graphs = []
    rnafm_list, hand_list = [], []
    label_binary_list, label_enzyme_list, per_enzyme_list = [], [], []
    index_list = []

    for b in batch_list:
        node_feat, edge_index = build_graph_from_dotbracket(b["dot_bracket"], b["raw_seq"])
        graphs.append(Data(x=node_feat, edge_index=edge_index))
        rnafm_list.append(b["rnafm"])
        hand_list.append(b["hand_feat"])
        label_binary_list.append(b["label_binary"])
        label_enzyme_list.append(b["label_enzyme"])
        per_enzyme_list.append(b["per_enzyme_labels"])
        index_list.append(b["index"])

    pyg_batch = PyGBatch.from_data_list(graphs)
    return {
        "pyg_batch": pyg_batch,
        "rnafm": torch.stack(rnafm_list),
        "hand_feat": torch.stack(hand_list),
        "label_binary": torch.stack(label_binary_list),
        "label_enzyme": torch.stack(label_enzyme_list),
        "per_enzyme_labels": torch.stack(per_enzyme_list),
        "index": index_list,
    }


# ---------------------------------------------------------------------------
# A4a/A4b. RNAFMLoRA
# ---------------------------------------------------------------------------


class LoRALayer(nn.Module):
    """Low-rank adaptation wrapper for a linear layer.

    Exposes weight/bias as properties so that code accessing layer.weight
    (like RNA-FM's multi_head_attention_forward) gets the LoRA-modified weight.
    """

    def __init__(self, orig_layer: nn.Linear, rank: int = 4):
        super().__init__()
        self.orig = orig_layer
        self.in_features = orig_layer.in_features
        self.out_features = orig_layer.out_features
        self.lora_A = nn.Parameter(torch.randn(self.in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, self.out_features))
        # Freeze original
        self.orig.weight.requires_grad = False
        if self.orig.bias is not None:
            self.orig.bias.requires_grad = False

    @property
    def weight(self):
        """Return LoRA-modified weight so F.multi_head_attention_forward sees it."""
        return self.orig.weight + (self.lora_A @ self.lora_B).T

    @property
    def bias(self):
        return self.orig.bias

    def forward(self, x):
        return self.orig(x) + x @ self.lora_A @ self.lora_B


def apply_lora_to_rnafm(rnafm_model, rank=4):
    """Apply LoRA to all Q, K, V projections in RNA-FM transformer layers."""
    lora_params = []
    for layer in rnafm_model.layers:
        # RNA-FM uses self_attn with q_proj, k_proj, v_proj
        attn = layer.self_attn
        for proj_name in ["q_proj", "k_proj", "v_proj"]:
            if hasattr(attn, proj_name):
                orig = getattr(attn, proj_name)
                if isinstance(orig, nn.Linear):
                    lora = LoRALayer(orig, rank=rank)
                    setattr(attn, proj_name, lora)
                    lora_params.extend([lora.lora_A, lora.lora_B])
    return lora_params


class RNAFMLoRA(nn.Module, HeadsMixin):
    """RNA-FM with LoRA fine-tuning → CLS token → heads."""

    name = "A4a_RNAFMLoRA"

    def __init__(self, with_features=False):
        super().__init__()
        self.with_features = with_features
        if with_features:
            self.name = "A4b_RNAFMLoRA_Features"

        import fm as fm_module
        self.rnafm_model, self.alphabet = fm_module.pretrained.rna_fm_t12()
        self.batch_converter = self.alphabet.get_batch_converter()

        # Freeze everything first
        for param in self.rnafm_model.parameters():
            param.requires_grad = False

        # Apply LoRA
        self.lora_params = apply_lora_to_rnafm(self.rnafm_model, rank=4)

        # Heads
        d_in = D_RNAFM + (D_HAND if with_features else 0)
        self.projection = nn.Sequential(
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
        return self.lora_params + list(self.projection.parameters())

    def get_adapter_params(self):
        params = list(self.binary_head.parameters())
        for enz in PER_ENZYME_HEADS:
            params.extend(self.enzyme_adapters[enz].parameters())
        params.extend(self.enzyme_classifier.parameters())
        return params

    def tokenize(self, sequences: List[str], device):
        """Tokenize sequences for RNA-FM (T→U conversion)."""
        data = [(f"seq_{i}", seq.upper().replace("T", "U")) for i, seq in enumerate(sequences)]
        _, _, tokens = self.batch_converter(data)
        return tokens.to(device)

    def forward(self, batch):
        device = next(self.parameters()).device
        tokens = self.tokenize(batch["raw_seq"], device)

        # Forward through RNA-FM with LoRA
        results = self.rnafm_model(tokens, repr_layers=[12])
        # CLS token is at position 0
        cls_repr = results["representations"][12][:, 0, :]  # [B, 640]

        if self.with_features:
            x = torch.cat([cls_repr, batch["hand_feat"]], dim=-1)
        else:
            x = cls_repr

        shared = self.projection(x)
        return self._apply_heads(shared)


# ---------------------------------------------------------------------------
# A5. GatedFusion
# ---------------------------------------------------------------------------


class GatedFusion(nn.Module, HeadsMixin):
    """Gated fusion: hand gates RNA-FM, RNA-FM gates delta."""

    name = "A5_GatedFusion"

    def __init__(self):
        super().__init__()
        # Gates
        self.gate_rnafm = nn.Linear(D_HAND, D_RNAFM)       # hand → gate for RNA-FM
        self.gate_delta = nn.Linear(D_RNAFM, D_EDIT_DELTA)  # rnafm → gate for delta
        self.hand_proj = nn.Linear(D_HAND, D_SHARED)        # hand → 128

        # Encoder: gated_rnafm(640) + gated_delta(640) + hand_proj(128) = 1408
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
        params = (
            list(self.gate_rnafm.parameters()) + list(self.gate_delta.parameters()) +
            list(self.hand_proj.parameters()) + list(self.encoder.parameters())
        )
        return params

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


# ---------------------------------------------------------------------------
# A6. Conv2DBP
# ---------------------------------------------------------------------------


class Conv2DBP(nn.Module, HeadsMixin):
    """Conv2D on 41×41 BP probability matrix + rnafm + hand → heads."""

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

        # conv_out(128) + rnafm(640) + hand(40) = 808
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
        bp = batch["bp_submatrix"]  # [B, 1, 41, 41]
        conv_out = self.conv(bp)  # [B, 32, 4, 4]
        conv_out = conv_out.flatten(1)  # [B, 512]
        conv_out = self.conv_fc(conv_out)  # [B, 128]

        fused = torch.cat([conv_out, batch["rnafm"], batch["hand_feat"]], dim=-1)
        shared = self.encoder(fused)
        return self._apply_heads(shared)


# ---------------------------------------------------------------------------
# A8. HierarchicalAttention
# ---------------------------------------------------------------------------


class HierarchicalAttention(nn.Module, HeadsMixin):
    """Two-level attention: local transformer on BP + cross-attention to RNA-FM."""

    name = "A8_HierarchicalAttention"

    def __init__(self):
        super().__init__()
        # Level 1: Local transformer on BP rows (41 positions, each with 41-dim BP features)
        d_local = 41  # Each row of the 41×41 BP matrix
        self.local_proj = nn.Linear(d_local, 64)

        # Position encoding for 41 positions
        self.local_pos_enc = nn.Parameter(torch.randn(41, 64) * 0.02)

        # Small transformer: 2 layers, 4 heads, dim=64
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128,
            dropout=0.1, activation="gelu", batch_first=True,
        )
        self.local_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.local_pool_attn = nn.Linear(64, 1)

        # Level 2: Cross-attention — local(64) queries global(640)
        self.cross_q = nn.Linear(64, 64)
        self.cross_k = nn.Linear(D_RNAFM, 64)
        self.cross_v = nn.Linear(D_RNAFM, 64)

        # Fusion: local(64) + cross(64) + rnafm(640) + hand(40) = 808
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
        params = (
            list(self.local_proj.parameters()) + [self.local_pos_enc] +
            list(self.local_transformer.parameters()) +
            list(self.local_pool_attn.parameters()) +
            list(self.cross_q.parameters()) + list(self.cross_k.parameters()) +
            list(self.cross_v.parameters()) + list(self.encoder.parameters())
        )
        return params

    def get_adapter_params(self):
        params = list(self.binary_head.parameters())
        for enz in PER_ENZYME_HEADS:
            params.extend(self.enzyme_adapters[enz].parameters())
        params.extend(self.enzyme_classifier.parameters())
        return params

    def forward(self, batch):
        bp = batch["bp_submatrix"]  # [B, 1, 41, 41]
        bp = bp.squeeze(1)  # [B, 41, 41]
        rnafm = batch["rnafm"]  # [B, 640]
        hand = batch["hand_feat"]  # [B, 40]

        # Level 1: Local transformer on BP rows
        local_in = self.local_proj(bp) + self.local_pos_enc.unsqueeze(0)  # [B, 41, 64]
        local_out = self.local_transformer(local_in)  # [B, 41, 64]

        # Attention-weighted pooling over 41 positions
        attn_w = torch.softmax(self.local_pool_attn(local_out), dim=1)  # [B, 41, 1]
        local_repr = (local_out * attn_w).sum(dim=1)  # [B, 64]

        # Level 2: Cross-attention (local queries global)
        q = self.cross_q(local_repr).unsqueeze(1)  # [B, 1, 64]
        k = self.cross_k(rnafm).unsqueeze(1)  # [B, 1, 64]
        v = self.cross_v(rnafm).unsqueeze(1)  # [B, 1, 64]

        attn_scores = (q * k).sum(-1) / math.sqrt(64)  # [B, 1]
        attn_weights = torch.softmax(attn_scores, dim=-1)  # [B, 1]
        cross_repr = (attn_weights.unsqueeze(-1) * v).squeeze(1)  # [B, 64]

        # Fusion
        fused = torch.cat([local_repr, cross_repr, rnafm, hand], dim=-1)
        shared = self.encoder(fused)
        return self._apply_heads(shared)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def compute_loss(binary_logit, per_enzyme_logits, enzyme_cls_logits, batch):
    """Compute combined loss: binary + per-enzyme + enzyme classifier."""
    label_binary = batch["label_binary"]
    label_enzyme = batch["label_enzyme"]
    per_enzyme_labels = batch["per_enzyme_labels"]

    # Binary BCE
    loss_binary = F.binary_cross_entropy_with_logits(binary_logit, label_binary)

    # Per-enzyme BCE (only on enzyme-matched samples, ignore label=-1)
    loss_enzyme_heads = torch.tensor(0.0, device=binary_logit.device)
    n_valid_heads = 0
    for head_idx in range(N_PER_ENZYME):
        enz_labels = per_enzyme_labels[:, head_idx]
        mask = enz_labels >= 0  # Only samples assigned to this enzyme
        if mask.sum() > 0:
            head_loss = F.binary_cross_entropy_with_logits(
                per_enzyme_logits[head_idx][mask], enz_labels[mask]
            )
            loss_enzyme_heads = loss_enzyme_heads + head_loss
            n_valid_heads += 1
    if n_valid_heads > 0:
        loss_enzyme_heads = loss_enzyme_heads / n_valid_heads

    # Enzyme classifier CE (positives only)
    pos_mask = label_binary > 0.5
    loss_cls = torch.tensor(0.0, device=binary_logit.device)
    if pos_mask.sum() > 0:
        loss_cls = F.cross_entropy(enzyme_cls_logits[pos_mask], label_enzyme[pos_mask])

    total = loss_binary + STAGE1_ENZYME_HEAD_WEIGHT * loss_enzyme_heads + STAGE1_ENZYME_CLS_WEIGHT * loss_cls
    return total, loss_binary.item()


def train_one_epoch(model, loader, optimizer, device, is_gat=False, is_lora=False):
    """Train one epoch. Returns (avg_loss, avg_binary_loss)."""
    model.train()
    total_loss, total_binary, n_batches = 0.0, 0.0, 0

    for batch in loader:
        # Move to device
        if is_gat:
            batch["pyg_batch"] = batch["pyg_batch"].to(device)
            batch["rnafm"] = batch["rnafm"].to(device)
            batch["hand_feat"] = batch["hand_feat"].to(device)
            batch["label_binary"] = batch["label_binary"].to(device)
            batch["label_enzyme"] = batch["label_enzyme"].to(device)
            batch["per_enzyme_labels"] = batch["per_enzyme_labels"].to(device)
        elif is_lora:
            # raw_seq stays as list of strings
            batch["hand_feat"] = batch["hand_feat"].to(device)
            batch["label_binary"] = batch["label_binary"].to(device)
            batch["label_enzyme"] = batch["label_enzyme"].to(device)
            batch["per_enzyme_labels"] = batch["per_enzyme_labels"].to(device)
        else:
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
def evaluate(model, loader, data, device, is_gat=False, is_lora=False):
    """Evaluate: returns overall AUROC and per-enzyme AUROCs."""
    model.eval()
    all_probs, all_labels, all_enzymes = [], [], []
    all_per_enzyme_probs = [[] for _ in range(N_PER_ENZYME)]
    all_per_enzyme_labels_list = [[] for _ in range(N_PER_ENZYME)]

    for batch in loader:
        if is_gat:
            batch["pyg_batch"] = batch["pyg_batch"].to(device)
            batch["rnafm"] = batch["rnafm"].to(device)
            batch["hand_feat"] = batch["hand_feat"].to(device)
            batch["label_binary"] = batch["label_binary"].to(device)
            batch["label_enzyme"] = batch["label_enzyme"].to(device)
            batch["per_enzyme_labels"] = batch["per_enzyme_labels"].to(device)
        elif is_lora:
            batch["hand_feat"] = batch["hand_feat"].to(device)
            batch["label_binary"] = batch["label_binary"].to(device)
            batch["label_enzyme"] = batch["label_enzyme"].to(device)
            batch["per_enzyme_labels"] = batch["per_enzyme_labels"].to(device)
        else:
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
    all_enzymes = np.array(all_enzymes)

    # Overall AUROC
    try:
        overall_auroc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        overall_auroc = 0.5

    # Per-enzyme AUROC (using adapter heads)
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


def train_and_evaluate(model, data, arch_name, is_gat=False, is_lora=False):
    """Full 2-fold CV training and evaluation for one architecture."""
    logger.info(f"\n{'=' * 70}")
    logger.info(f"Training: {arch_name}")
    logger.info(f"{'=' * 70}")

    n = len(data["labels_binary"])
    bs = BATCH_SIZE_LORA if is_lora else BATCH_SIZE

    # Stratify on enzyme+label
    strat_key = data["labels_enzyme"] * 2 + data["labels_binary"].astype(np.int64)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    fold_results = []
    total_t0 = time.time()

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(n), strat_key)):
        logger.info(f"\n--- Fold {fold + 1}/{N_FOLDS} ({len(train_idx)} train, {len(val_idx)} val) ---")
        fold_t0 = time.time()

        # Create fresh model
        if is_lora:
            # Need to recreate model for each fold (RNA-FM is heavy)
            if "Features" in arch_name or "Feat" in arch_name:
                fold_model = RNAFMLoRA(with_features=True).to(DEVICE)
            else:
                fold_model = RNAFMLoRA(with_features=False).to(DEVICE)
        elif is_gat:
            fold_model = AdarEditGAT().to(DEVICE)
        else:
            fold_model = model.__class__().to(DEVICE)

        # Create dataloaders
        if is_gat:
            train_ds = GATDataset(train_idx, data)
            val_ds = GATDataset(val_idx, data)
            train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                                      collate_fn=gat_collate, num_workers=0)
            val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                                    collate_fn=gat_collate, num_workers=0)
        elif is_lora:
            train_ds = LoRADataset(train_idx, data)
            val_ds = LoRADataset(val_idx, data)
            train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                                      collate_fn=standard_collate, num_workers=0)
            val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                                    collate_fn=standard_collate, num_workers=0)
        else:
            train_ds = ScreenDataset(train_idx, data)
            val_ds = ScreenDataset(val_idx, data)
            train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                                      collate_fn=standard_collate, num_workers=0)
            val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                                    collate_fn=standard_collate, num_workers=0)

        # Stage 1: Joint training
        logger.info(f"  Stage 1: Joint training ({STAGE1_EPOCHS} epochs, lr={STAGE1_LR})")
        if is_lora:
            # Separate LR for LoRA params vs head params
            optimizer = torch.optim.Adam([
                {"params": fold_model.lora_params, "lr": 1e-4},
                {"params": list(fold_model.projection.parameters()), "lr": STAGE1_LR},
                {"params": fold_model.get_adapter_params(), "lr": STAGE1_LR},
            ], weight_decay=STAGE1_WEIGHT_DECAY)
        else:
            optimizer = torch.optim.Adam([
                {"params": fold_model.get_shared_params(), "lr": STAGE1_LR},
                {"params": fold_model.get_adapter_params(), "lr": STAGE1_LR},
            ], weight_decay=STAGE1_WEIGHT_DECAY)

        for epoch in range(STAGE1_EPOCHS):
            avg_loss, avg_binary = train_one_epoch(
                fold_model, train_loader, optimizer, DEVICE,
                is_gat=is_gat, is_lora=is_lora,
            )
            val_auroc, val_per_enz = evaluate(
                fold_model, val_loader, data, DEVICE,
                is_gat=is_gat, is_lora=is_lora,
            )
            enz_str = " ".join(f"{e}={v:.3f}" for e, v in val_per_enz.items() if not np.isnan(v))
            logger.info(
                f"    Epoch {epoch + 1:2d} | loss={avg_loss:.4f} bce={avg_binary:.4f} "
                f"| val_auroc={val_auroc:.4f} | {enz_str}"
            )

        # Stage 2: Adapter-only training
        logger.info(f"  Stage 2: Adapter-only ({STAGE2_EPOCHS} epochs, lr={STAGE2_LR})")
        # Freeze shared params
        for p in fold_model.get_shared_params():
            p.requires_grad = False

        optimizer2 = torch.optim.Adam(
            fold_model.get_adapter_params(),
            lr=STAGE2_LR, weight_decay=STAGE2_WEIGHT_DECAY,
        )

        for epoch in range(STAGE2_EPOCHS):
            avg_loss, avg_binary = train_one_epoch(
                fold_model, train_loader, optimizer2, DEVICE,
                is_gat=is_gat, is_lora=is_lora,
            )
            val_auroc, val_per_enz = evaluate(
                fold_model, val_loader, data, DEVICE,
                is_gat=is_gat, is_lora=is_lora,
            )
            if (epoch + 1) % 5 == 0 or epoch == STAGE2_EPOCHS - 1:
                enz_str = " ".join(f"{e}={v:.3f}" for e, v in val_per_enz.items() if not np.isnan(v))
                logger.info(
                    f"    Epoch {STAGE1_EPOCHS + epoch + 1:2d} | loss={avg_loss:.4f} "
                    f"| val_auroc={val_auroc:.4f} | {enz_str}"
                )

        # Final evaluation
        final_auroc, final_per_enz = evaluate(
            fold_model, val_loader, data, DEVICE,
            is_gat=is_gat, is_lora=is_lora,
        )
        fold_time = time.time() - fold_t0
        logger.info(
            f"  Fold {fold + 1} FINAL: overall_auroc={final_auroc:.4f} "
            f"| time={fold_time:.0f}s"
        )
        for enz, auroc in final_per_enz.items():
            logger.info(f"    {enz}: {auroc:.4f}")

        fold_results.append({
            "fold": fold + 1,
            "overall_auroc": final_auroc,
            "per_enzyme_aurocs": final_per_enz,
            "time_s": fold_time,
        })

        # Cleanup
        del fold_model, optimizer, train_loader, val_loader, train_ds, val_ds
        if is_lora:
            del optimizer2
        gc.collect()
        if DEVICE.type == "mps":
            torch.mps.empty_cache()

    total_time = time.time() - total_t0

    # Aggregate across folds
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

    logger.info(f"\n  {arch_name} SUMMARY:")
    logger.info(f"    Overall AUROC: {mean_overall:.4f} +/- {std_overall:.4f}")
    for enz in PER_ENZYME_HEADS:
        if enz in per_enz_means:
            logger.info(f"    {enz}: {per_enz_means[enz]:.4f} +/- {per_enz_stds[enz]:.4f}")
    logger.info(f"    Total time: {total_time:.0f}s")

    return {
        "architecture": arch_name,
        "overall_auroc_mean": round(mean_overall, 4),
        "overall_auroc_std": round(std_overall, 4),
        "per_enzyme_auroc_mean": {k: round(v, 4) for k, v in per_enz_means.items()},
        "per_enzyme_auroc_std": {k: round(v, 4) for k, v in per_enz_stds.items()},
        "total_time_s": round(total_time, 1),
        "fold_results": fold_results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    logger.info("=" * 70)
    logger.info("ARCHITECTURE SCREEN FOR RNA EDITING PREDICTION")
    logger.info(f"Device: {DEVICE} | Folds: {N_FOLDS} | Seed: {SEED}")
    logger.info(f"Stage 1: {STAGE1_EPOCHS} epochs, lr={STAGE1_LR}")
    logger.info(f"Stage 2: {STAGE2_EPOCHS} epochs, lr={STAGE2_LR}")
    logger.info("=" * 70)

    # Load data once
    data = load_data()
    n = len(data["labels_binary"])
    logger.info(f"\nTotal sites: {n}")
    logger.info(f"  Positive: {int(data['labels_binary'].sum())}")
    logger.info(f"  Negative: {int((data['labels_binary'] == 0).sum())}")

    # Define architectures to screen
    architectures = [
        ("A1_Phase3Baseline", lambda: Phase3Baseline(), False, False),
        ("A2_NoDelta", lambda: NoDelta(), False, False),
        ("A5_GatedFusion", lambda: GatedFusion(), False, False),
        ("A6_Conv2DBP", lambda: Conv2DBP(), False, False),
        ("A8_HierarchicalAttention", lambda: HierarchicalAttention(), False, False),
    ]

    # Conditionally add GAT (requires torch_geometric)
    if HAS_PYG:
        architectures.insert(2, ("A3_AdarEditGAT", lambda: AdarEditGAT(), True, False))
    else:
        logger.warning("torch_geometric not available — skipping A3_AdarEditGAT")

    # Add LoRA architectures
    architectures.append(("A4a_RNAFMLoRA", lambda: RNAFMLoRA(with_features=False), False, True))
    architectures.append(("A4b_RNAFMLoRA_Features", lambda: RNAFMLoRA(with_features=True), False, True))

    all_results = []

    for arch_name, model_fn, is_gat, is_lora in architectures:
        try:
            model = model_fn()
            n_params = sum(p.numel() for p in model.parameters())
            n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"\n{arch_name}: {n_params:,} total params, {n_trainable:,} trainable")

            result = train_and_evaluate(model, data, arch_name, is_gat=is_gat, is_lora=is_lora)
            all_results.append(result)

            # Save intermediate results
            with open(OUTPUT_DIR / "architecture_screen_results.json", "w") as f:
                json.dump(all_results, f, indent=2, default=str)

            del model
            gc.collect()
            if DEVICE.type == "mps":
                torch.mps.empty_cache()

        except Exception as e:
            logger.error(f"\n{'!' * 70}")
            logger.error(f"FAILED: {arch_name}")
            logger.error(f"Error: {e}")
            logger.error(traceback.format_exc())
            logger.error(f"{'!' * 70}")
            all_results.append({
                "architecture": arch_name,
                "overall_auroc_mean": None,
                "overall_auroc_std": None,
                "per_enzyme_auroc_mean": {},
                "per_enzyme_auroc_std": {},
                "total_time_s": 0,
                "error": str(e),
            })
            gc.collect()
            if DEVICE.type == "mps":
                torch.mps.empty_cache()

    # ---------------------------------------------------------------------------
    # Summary comparison table
    # ---------------------------------------------------------------------------
    logger.info("\n" + "=" * 90)
    logger.info("ARCHITECTURE SCREEN — SUMMARY COMPARISON")
    logger.info("=" * 90)

    header = f"{'Architecture':<30} {'Overall':>10} {'A3A':>8} {'A3B':>8} {'A3G':>8} {'A3A_A3G':>8} {'Neither':>8} {'Time':>8}"
    logger.info(header)
    logger.info("-" * 90)

    for r in all_results:
        name = r["architecture"]
        if r["overall_auroc_mean"] is None:
            logger.info(f"{name:<30} {'FAILED':>10}")
            continue

        overall = f"{r['overall_auroc_mean']:.4f}"
        pe = r.get("per_enzyme_auroc_mean", {})
        a3a = f"{pe.get('A3A', float('nan')):.3f}" if "A3A" in pe else "  N/A"
        a3b = f"{pe.get('A3B', float('nan')):.3f}" if "A3B" in pe else "  N/A"
        a3g = f"{pe.get('A3G', float('nan')):.3f}" if "A3G" in pe else "  N/A"
        both = f"{pe.get('A3A_A3G', float('nan')):.3f}" if "A3A_A3G" in pe else "  N/A"
        neither = f"{pe.get('Neither', float('nan')):.3f}" if "Neither" in pe else "  N/A"
        time_str = f"{r['total_time_s']:.0f}s"
        logger.info(f"{name:<30} {overall:>10} {a3a:>8} {a3b:>8} {a3g:>8} {both:>8} {neither:>8} {time_str:>8}")

    logger.info("=" * 90)

    # Save final results
    results_path = OUTPUT_DIR / "architecture_screen_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nResults saved to {results_path}")

    logger.info("\nArchitecture screen complete.")


if __name__ == "__main__":
    main()
