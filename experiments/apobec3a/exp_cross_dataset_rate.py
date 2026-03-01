#!/usr/bin/env python
"""Cross-dataset rate prediction experiment.

Trains on each dataset independently and evaluates on all other datasets
(leave-one-dataset-out cross-dataset generalization). Only 3 models:
  1. GB_HandFeatures   - XGBRegressor on 40-dim hand features
  2. EditRNA_rate      - Full EditRNA-A3A with CachedRNAEncoder
  3. 4Way_heavyreg     - 4-way gated fusion (primary + edit_delta + hand + GNN)

Datasets (from dataset_source column):
  - baysal_2016  (~4208 sites)
  - advisor_c2t  (~636 sites)
  - alqassim_2021 (~128 sites)
  - sharma_2015   (~6 sites, test-only)

This gives a 3x4 matrix per model (3 training sets x 4 test sets).

Usage:
    python experiments/apobec3a/exp_cross_dataset_rate.py
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import gc
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error, r2_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "apobec"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SPLITS_CSV = PROJECT_ROOT / "data" / "processed" / "splits_expanded.csv"
EMB_DIR = PROJECT_ROOT / "data" / "processed" / "embeddings"
STRUCTURE_CACHE = EMB_DIR / "vienna_structure_cache.npz"
SEQUENCES_JSON = PROJECT_ROOT / "data" / "processed" / "site_sequences.json"
LOOP_POS_CSV = (
    PROJECT_ROOT / "experiments" / "apobec" / "outputs"
    / "loop_position" / "loop_position_per_site.csv"
)
OUTPUT_DIR = (
    PROJECT_ROOT / "experiments" / "apobec" / "outputs" / "cross_dataset_rate"
)

SEED = 42
DEVICE = torch.device("cpu")

# Datasets to train on (sharma_2015 is test-only due to only 6 sites)
TRAIN_DATASETS = ["baysal_2016", "advisor_c2t", "alqassim_2021"]
ALL_DATASETS = ["baysal_2016", "advisor_c2t", "alqassim_2021", "sharma_2015"]

MODEL_NAMES = ["GB_HandFeatures", "EditRNA_rate", "4Way_heavyreg"]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_rate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute regression metrics for rate prediction."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) < 3:
        return {"spearman": float("nan"), "pearson": float("nan"),
                "mse": float("nan"), "r2": float("nan")}
    sp, _ = spearmanr(y_true, y_pred)
    pe, _ = pearsonr(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"spearman": float(sp), "pearson": float(pe),
            "mse": float(mse), "r2": float(r2)}


# ---------------------------------------------------------------------------
# Data Loading (all at once)
# ---------------------------------------------------------------------------

def load_all_data():
    """Load all data once: splits, embeddings, tokens, structures, sequences,
    loop features, pairing profiles.  Memory-filtered to rate-annotated sites."""
    logger.info("Loading data...")

    # Splits -- positives only (NEVER include negatives in rate regression)
    df = pd.read_csv(SPLITS_CSV)
    df = df[(df["editing_rate_normalized"].notna()) & (df["is_edited"] == 1)].copy()
    assert (df["is_edited"] == 1).all(), "Rate prediction must use positives only!"
    df["target"] = np.log2(df["editing_rate_normalized"].values + 0.01)
    logger.info("  Total rate-annotated positive sites: %d", len(df))
    # Log dataset composition
    if "dataset_source" in df.columns:
        for src, cnt in df["dataset_source"].value_counts().items():
            logger.info("    %s: %d sites", src, cnt)

    needed_sids = set(df["site_id"].astype(str).tolist())

    # Pooled embeddings (filtered)
    _po = torch.load(EMB_DIR / "rnafm_pooled.pt", weights_only=False)
    pooled_orig = {k: v for k, v in _po.items() if str(k) in needed_sids}
    del _po; gc.collect()

    _pe = torch.load(EMB_DIR / "rnafm_pooled_edited.pt", weights_only=False)
    pooled_edited = {k: v for k, v in _pe.items() if str(k) in needed_sids}
    del _pe; gc.collect()
    logger.info("  %d pooled embeddings loaded (filtered)", len(pooled_orig))

    # Token-level embeddings (filtered)
    _to = torch.load(EMB_DIR / "rnafm_tokens.pt", weights_only=False)
    tokens_orig = {k: v for k, v in _to.items() if str(k) in needed_sids}
    del _to; gc.collect()

    _te = torch.load(EMB_DIR / "rnafm_tokens_edited.pt", weights_only=False)
    tokens_edited = {k: v for k, v in _te.items() if str(k) in needed_sids}
    del _te; gc.collect()
    logger.info("  %d token embeddings loaded (filtered)", len(tokens_orig))

    # Structure cache
    struct_data = np.load(str(STRUCTURE_CACHE), allow_pickle=True)
    struct_sids = [str(s) for s in struct_data["site_ids"]]
    sid_to_idx = {sid: i for i, sid in enumerate(struct_sids) if sid in needed_sids}
    idx_arr = np.array(sorted(sid_to_idx.values()))
    idx_sids = [struct_sids[i] for i in idx_arr]

    structure_delta = dict(zip(idx_sids, struct_data["delta_features"][idx_arr]))

    pairing_probs = dict(zip(idx_sids, struct_data["pairing_probs"][idx_arr]))
    pairing_probs_edited = dict(zip(idx_sids, struct_data["pairing_probs_edited"][idx_arr]))
    accessibilities = dict(zip(idx_sids, struct_data["accessibilities"][idx_arr]))
    accessibilities_edited = dict(zip(idx_sids, struct_data["accessibilities_edited"][idx_arr]))
    mfes = {sid: float(struct_data["mfes"][i]) for sid, i in sid_to_idx.items()}
    mfes_edited = {sid: float(struct_data["mfes_edited"][i]) for sid, i in sid_to_idx.items()}

    del struct_data; gc.collect()
    logger.info("  %d structure caches loaded (filtered)", len(pairing_probs))

    # Sequences
    with open(SEQUENCES_JSON) as f:
        _seq = json.load(f)
    sequences = {k: v for k, v in _seq.items() if str(k) in needed_sids}
    del _seq
    logger.info("  %d sequences loaded (filtered)", len(sequences))

    # Loop features
    loop_df = pd.DataFrame()
    if LOOP_POS_CSV.exists():
        loop_df = pd.read_csv(LOOP_POS_CSV).set_index("site_id")
        loop_df = loop_df[loop_df.index.isin(needed_sids)]
        logger.info("  %d loop features loaded (filtered)", len(loop_df))

    return {
        "df": df,
        "pooled_orig": pooled_orig, "pooled_edited": pooled_edited,
        "tokens_orig": tokens_orig, "tokens_edited": tokens_edited,
        "structure_delta": structure_delta,
        "pairing_probs": pairing_probs, "pairing_probs_edited": pairing_probs_edited,
        "accessibilities": accessibilities, "accessibilities_edited": accessibilities_edited,
        "mfes": mfes, "mfes_edited": mfes_edited,
        "sequences": sequences, "loop_df": loop_df,
    }


# ---------------------------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------------------------

def extract_motif_features(sequences: Dict[str, str], site_ids: List[str]) -> np.ndarray:
    """24-dim motif features: 5' dinuc (4) + 3' dinuc (4) + trinuc_up (8) + trinuc_down (8)."""
    bases = ["A", "C", "G", "U"]
    features = []
    for sid in site_ids:
        seq = sequences.get(sid, "N" * 201).upper().replace("T", "U")
        ep = 100
        up = seq[ep - 1] if ep > 0 else "N"
        down = seq[ep + 1] if ep < len(seq) - 1 else "N"
        feat_5p = [1 if up + "C" == m else 0 for m in ["UC", "CC", "AC", "GC"]]
        feat_3p = [1 if "C" + down == m else 0 for m in ["CA", "CG", "CU", "CC"]]
        trinuc_up = [0] * 8
        for offset, bo in [(-2, 0), (-1, 4)]:
            pos = ep + offset
            if 0 <= pos < len(seq):
                for bi, b in enumerate(bases):
                    if seq[pos] == b:
                        trinuc_up[bo + bi] = 1
        trinuc_down = [0] * 8
        for offset, bo in [(1, 0), (2, 4)]:
            pos = ep + offset
            if 0 <= pos < len(seq):
                for bi, b in enumerate(bases):
                    if seq[pos] == b:
                        trinuc_down[bo + bi] = 1
        features.append(feat_5p + feat_3p + trinuc_up + trinuc_down)
    return np.array(features, dtype=np.float32)


def extract_loop_features(loop_df: pd.DataFrame, site_ids: List[str]) -> np.ndarray:
    """9-dim loop position features."""
    cols = [
        "is_unpaired", "loop_size", "dist_to_junction", "dist_to_apex",
        "relative_loop_position", "left_stem_length", "right_stem_length",
        "max_adjacent_stem_length", "local_unpaired_fraction",
    ]
    features = []
    for sid in site_ids:
        if sid in loop_df.index:
            features.append(loop_df.loc[sid, cols].values.astype(np.float32))
        else:
            features.append(np.zeros(len(cols), dtype=np.float32))
    return np.array(features, dtype=np.float32)


def extract_structure_delta(structure_delta: Dict, site_ids: List[str]) -> np.ndarray:
    """7-dim structure delta features."""
    return np.array([structure_delta.get(sid, np.zeros(7)) for sid in site_ids],
                    dtype=np.float32)


def build_hand_features(site_ids, sequences, structure_delta, loop_df) -> np.ndarray:
    """Build 40-dim hand feature matrix (motif 24 + struct_delta 7 + loop 9)."""
    motif = extract_motif_features(sequences, site_ids)
    struct = extract_structure_delta(structure_delta, site_ids)
    loop = extract_loop_features(loop_df, site_ids)
    hand = np.concatenate([motif, struct, loop], axis=1)
    return np.nan_to_num(hand, nan=0.0)


# ---------------------------------------------------------------------------
# GNN graph building and embedding pre-computation
# ---------------------------------------------------------------------------

def build_structure_graph_from_pairing(pp, sequence, edit_pos=100, threshold=0.3):
    """Build a PyG graph from pairing probabilities."""
    from torch_geometric.data import Data

    n = len(sequence)
    seq = sequence.upper().replace("T", "U")

    nuc_to_idx = {"A": 0, "C": 1, "G": 2, "U": 3, "N": 4}
    x = torch.zeros(n, 12)
    for i, nuc in enumerate(seq):
        x[i, nuc_to_idx.get(nuc, 4)] = 1.0
    x[:, 5] = torch.linspace(-1, 1, n)
    x[:, 6] = torch.abs(torch.arange(n, dtype=torch.float32) - edit_pos) / 100.0
    x[edit_pos, 7] = 1.0
    x[:, 8] = torch.tensor(pp[:n], dtype=torch.float32)
    x[:, 9] = (torch.tensor(pp[:n]) > 0.5).float()
    for i in range(n):
        s, e = max(0, i - 5), min(n, i + 6)
        x[i, 10] = float(pp[s:e].mean())
    for i in range(n):
        s, e = max(0, i - 5), min(n, i + 6)
        win = seq[s:e]
        x[i, 11] = float((win.count("G") + win.count("C")) / max(len(win), 1))

    edges_src, edges_dst = [], []
    for i in range(n - 1):
        edges_src.extend([i, i + 1])
        edges_dst.extend([i + 1, i])
    stack = []
    for i in range(n):
        if pp[i] > threshold:
            if i > 0 and pp[i] > pp[i - 1]:
                stack.append(i)
            elif stack:
                j = stack.pop()
                edges_src.extend([i, j])
                edges_dst.extend([j, i])
    for i in range(n):
        if pp[i] > 0.7:
            for j in range(i + 4, min(i + 100, n)):
                if pp[j] > 0.7 and abs(pp[i] - pp[j]) < 0.2:
                    b1, b2 = seq[i], seq[j]
                    if (b1, b2) in {("A", "U"), ("U", "A"), ("G", "C"), ("C", "G"),
                                     ("G", "U"), ("U", "G")}:
                        edges_src.extend([i, j])
                        edges_dst.extend([j, i])
                        break
    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    return Data(x=x, edge_index=edge_index)


class SmallStructureGNN(nn.Module):
    """Compact GNN for structure-aware embedding."""
    def __init__(self, node_features=12, hidden_dim=64, output_dim=64,
                 num_layers=3, heads=4, dropout=0.3):
        super().__init__()
        from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool

        self.node_embed = nn.Sequential(
            nn.Linear(node_features, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GATConv(hidden_dim, hidden_dim // heads,
                                      heads=heads, concat=True, dropout=dropout))
            self.norms.append(nn.LayerNorm(hidden_dim))
        self.dropout = nn.Dropout(dropout)
        self.mean_pool = global_mean_pool
        self.max_pool = global_max_pool
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, "batch") else torch.zeros(
            x.size(0), dtype=torch.long, device=x.device)
        x = self.node_embed(x)
        for conv, norm in zip(self.convs, self.norms):
            x_new = conv(x, edge_index)
            x_new = norm(x_new)
            x_new = F.gelu(x_new)
            x_new = self.dropout(x_new)
            x = x + x_new
        x_mean = self.mean_pool(x, batch)
        x_max = self.max_pool(x, batch)
        return self.output_proj(torch.cat([x_mean, x_max], dim=-1))


def precompute_gnn_embeddings(gnn_model, site_ids, sequences, pairing_probs,
                               batch_size=64):
    """Pre-compute GNN graph embeddings for all sites."""
    from torch_geometric.data import Batch

    gnn_model.eval()
    embeddings = {}
    batched_ids = [site_ids[i:i + batch_size] for i in range(0, len(site_ids), batch_size)]
    for batch_ids in batched_ids:
        graphs, valid_ids = [], []
        for sid in batch_ids:
            seq = sequences.get(sid, "N" * 201)
            pp = pairing_probs.get(sid, np.zeros(201))
            if len(seq) >= 201 and len(pp) >= 201:
                graphs.append(build_structure_graph_from_pairing(pp, seq[:201]))
                valid_ids.append(sid)
        if not graphs:
            continue
        batch_data = Batch.from_data_list(graphs)
        with torch.no_grad():
            emb = gnn_model(batch_data)
        for i, sid in enumerate(valid_ids):
            embeddings[sid] = emb[i].numpy()
    return embeddings


# ---------------------------------------------------------------------------
# NN Model Architectures
# ---------------------------------------------------------------------------

class EditRNARateWrapper(nn.Module):
    """Wrapper that runs EditRNA-A3A and extracts rate prediction."""
    def __init__(self, editrna_model):
        super().__init__()
        self.editrna = editrna_model
    def forward(self, batch):
        B = batch["targets"].shape[0]
        device = batch["targets"].device
        editrna_batch = {
            "sequences": ["N" * 201] * B,
            "site_ids": batch["site_ids"],
            "edit_pos": torch.full((B,), 100, dtype=torch.long, device=device),
            "flanking_context": torch.zeros(B, dtype=torch.long, device=device),
            "concordance_features": torch.zeros(B, 5, device=device),
            "structure_delta": batch["structure_delta"],
        }
        output = self.editrna(editrna_batch)
        return {"rate_pred": output["predictions"]["editing_rate"]}


class FourWayGatedFusion(nn.Module):
    """4-way gated fusion: primary + edit_delta + hand + GNN -> rate."""
    def __init__(self, d_primary=640, d_hand=40, d_gnn=64, d_proj=128, dropout=0.3):
        super().__init__()
        self.primary_proj = nn.Sequential(
            nn.Linear(d_primary, d_proj), nn.GELU(), nn.LayerNorm(d_proj))
        self.edit_proj = nn.Sequential(
            nn.Linear(d_primary, d_proj), nn.GELU(), nn.LayerNorm(d_proj))
        self.hand_proj = nn.Sequential(
            nn.Linear(d_hand, d_proj), nn.GELU(), nn.LayerNorm(d_proj))
        self.gnn_proj = nn.Sequential(
            nn.Linear(d_gnn, d_proj), nn.GELU(), nn.LayerNorm(d_proj))
        n_mod = 4
        self.gate = nn.Linear(d_proj * n_mod, n_mod)
        self.fuse = nn.Sequential(
            nn.Linear(d_proj * n_mod, d_proj * 2), nn.GELU(), nn.Dropout(dropout))
        self.head = nn.Sequential(
            nn.Linear(d_proj * 2, d_proj), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_proj, 1))
    def forward(self, batch):
        p = self.primary_proj(batch["pooled_orig"])
        e = self.edit_proj(batch["pooled_edited"] - batch["pooled_orig"])
        h = self.hand_proj(batch["hand_features"])
        g = self.gnn_proj(batch["gnn_emb"])
        concat = torch.cat([p, e, h, g], dim=-1)
        gates = torch.softmax(self.gate(concat), dim=-1)
        gated = torch.cat([
            gates[:, 0:1] * p, gates[:, 1:2] * e,
            gates[:, 2:3] * h, gates[:, 3:4] * g,
        ], dim=-1)
        return {"rate_pred": self.head(self.fuse(gated))}


# ---------------------------------------------------------------------------
# PyTorch Datasets and Collate Functions
# ---------------------------------------------------------------------------

class RateDataset(Dataset):
    """Dataset for rate regression with pre-computed embeddings."""
    def __init__(self, site_ids, targets, pooled_orig, pooled_edited,
                 structure_delta, tokens_orig=None, tokens_edited=None):
        self.site_ids = site_ids
        self.targets = targets
        self.pooled_orig = pooled_orig
        self.pooled_edited = pooled_edited
        self.structure_delta = structure_delta
        self.tokens_orig = tokens_orig
        self.tokens_edited = tokens_edited

    def __len__(self):
        return len(self.site_ids)

    def __getitem__(self, idx):
        sid = self.site_ids[idx]
        item = {
            "site_id": sid,
            "target": torch.tensor(self.targets[idx], dtype=torch.float32),
            "pooled_orig": self.pooled_orig[sid],
            "pooled_edited": self.pooled_edited[sid],
            "structure_delta": torch.tensor(
                self.structure_delta.get(sid, np.zeros(7)), dtype=torch.float32),
        }
        if self.tokens_orig is not None and sid in self.tokens_orig:
            item["tokens_orig"] = self.tokens_orig[sid]
            item["tokens_edited"] = self.tokens_edited[sid]
        return item


def rate_collate_fn(batch):
    result = {
        "site_ids": [b["site_id"] for b in batch],
        "targets": torch.stack([b["target"] for b in batch]),
        "pooled_orig": torch.stack([b["pooled_orig"] for b in batch]),
        "pooled_edited": torch.stack([b["pooled_edited"] for b in batch]),
        "structure_delta": torch.stack([b["structure_delta"] for b in batch]),
    }
    if "tokens_orig" in batch[0]:
        max_len = max(b["tokens_orig"].shape[0] for b in batch)
        d = batch[0]["tokens_orig"].shape[1]
        tok_orig = torch.zeros(len(batch), max_len, d)
        tok_edited = torch.zeros(len(batch), max_len, d)
        for i, b in enumerate(batch):
            L = b["tokens_orig"].shape[0]
            tok_orig[i, :L] = b["tokens_orig"]
            tok_edited[i, :L] = b["tokens_edited"]
        result["tokens_orig"] = tok_orig
        result["tokens_edited"] = tok_edited
    return result


class MultiModalRateDataset(Dataset):
    """Dataset with pooled embeddings + hand features + optional GNN embeddings."""
    def __init__(self, site_ids, targets, pooled_orig, pooled_edited,
                 structure_delta, hand_features, gnn_embeddings=None):
        self.site_ids = site_ids
        self.targets = targets
        self.pooled_orig = pooled_orig
        self.pooled_edited = pooled_edited
        self.structure_delta = structure_delta
        self.hand_features = hand_features
        self.gnn_embeddings = gnn_embeddings

    def __len__(self):
        return len(self.site_ids)

    def __getitem__(self, idx):
        sid = self.site_ids[idx]
        item = {
            "site_id": sid,
            "target": torch.tensor(self.targets[idx], dtype=torch.float32),
            "pooled_orig": self.pooled_orig[sid],
            "pooled_edited": self.pooled_edited[sid],
            "structure_delta": torch.tensor(
                self.structure_delta.get(sid, np.zeros(7)), dtype=torch.float32),
            "hand_features": torch.tensor(self.hand_features[idx], dtype=torch.float32),
        }
        if self.gnn_embeddings is not None:
            emb = self.gnn_embeddings.get(sid)
            if emb is not None:
                item["gnn_emb"] = torch.tensor(emb, dtype=torch.float32)
            else:
                d = next(iter(self.gnn_embeddings.values())).shape[0]
                item["gnn_emb"] = torch.zeros(d, dtype=torch.float32)
        return item


def mm_collate(batch):
    result = {
        "site_ids": [b["site_id"] for b in batch],
        "targets": torch.stack([b["target"] for b in batch]),
        "pooled_orig": torch.stack([b["pooled_orig"] for b in batch]),
        "pooled_edited": torch.stack([b["pooled_edited"] for b in batch]),
        "structure_delta": torch.stack([b["structure_delta"] for b in batch]),
        "hand_features": torch.stack([b["hand_features"] for b in batch]),
    }
    if "gnn_emb" in batch[0]:
        result["gnn_emb"] = torch.stack([b["gnn_emb"] for b in batch])
    return result


# ---------------------------------------------------------------------------
# Training Utilities
# ---------------------------------------------------------------------------

def train_nn_model(
    model: nn.Module,
    loaders: Dict[str, DataLoader],
    model_name: str,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    patience: int = 15,
) -> nn.Module:
    """Train a NN model with early stopping on val Spearman. Returns trained model."""
    model = model.to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)

    best_spearman = -float("inf")
    best_epoch = 0
    patience_counter = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for batch in loaders["train"]:
            batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            optimizer.zero_grad()
            output = model(batch)
            pred = output["rate_pred"].squeeze(-1)
            loss = F.mse_loss(pred, batch["targets"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1
        scheduler.step()

        val_metrics = evaluate_nn_model(model, loaders.get("val"))
        val_sp = val_metrics.get("spearman", -999)
        if not np.isnan(val_sp) and val_sp > best_spearman + 1e-5:
            best_spearman = val_sp
            best_epoch = epoch
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    logger.info("    %s: best epoch=%d, best val spearman=%.4f",
                model_name, best_epoch, best_spearman)
    return model


def train_editrna_rate(
    model: nn.Module,
    loaders: Dict[str, DataLoader],
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    patience: int = 15,
    rate_weight: float = 10.0,
) -> nn.Module:
    """Train EditRNA-A3A with rate_weight on MSE loss. Returns trained model."""
    model = model.to(DEVICE)
    editrna = model.editrna

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)

    best_spearman = -float("inf")
    best_epoch = 0
    patience_counter = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in loaders["train"]:
            batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            B = batch["targets"].shape[0]
            device = batch["targets"].device
            editrna_batch = {
                "sequences": ["N" * 201] * B,
                "site_ids": batch["site_ids"],
                "edit_pos": torch.full((B,), 100, dtype=torch.long, device=device),
                "flanking_context": torch.zeros(B, dtype=torch.long, device=device),
                "concordance_features": torch.zeros(B, 5, device=device),
                "structure_delta": batch["structure_delta"],
            }
            output = editrna(editrna_batch)
            rate_pred = output["predictions"]["editing_rate"].squeeze(-1)
            loss = rate_weight * F.mse_loss(rate_pred, batch["targets"])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        val_metrics = evaluate_nn_model(model, loaders.get("val"))
        val_sp = val_metrics.get("spearman", -999)
        if not np.isnan(val_sp) and val_sp > best_spearman + 1e-5:
            best_spearman = val_sp
            best_epoch = epoch
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    logger.info("    EditRNA_rate: best epoch=%d, best val spearman=%.4f",
                best_epoch, best_spearman)
    return model


@torch.no_grad()
def evaluate_nn_model(model, loader):
    """Evaluate a neural model on a DataLoader."""
    if loader is None:
        return {}
    model.eval()
    all_t, all_p = [], []
    for batch in loader:
        batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        output = model(batch)
        all_t.append(batch["targets"].cpu().numpy())
        all_p.append(output["rate_pred"].squeeze(-1).cpu().numpy())
    return compute_rate_metrics(np.concatenate(all_t), np.concatenate(all_p))


# ---------------------------------------------------------------------------
# DataLoader creation helpers
# ---------------------------------------------------------------------------

def make_token_loaders_from_splits(
    train_sids, train_targets,
    val_sids, val_targets,
    data, batch_size=32,
) -> Dict[str, DataLoader]:
    """Create train/val DataLoaders with token-level embeddings."""
    loaders = {}
    for name, sids, tgts in [("train", train_sids, train_targets),
                              ("val", val_sids, val_targets)]:
        ds = RateDataset(
            site_ids=sids, targets=tgts,
            pooled_orig=data["pooled_orig"], pooled_edited=data["pooled_edited"],
            structure_delta=data["structure_delta"],
            tokens_orig=data["tokens_orig"], tokens_edited=data["tokens_edited"],
        )
        loaders[name] = DataLoader(
            ds, batch_size=batch_size, shuffle=(name == "train"),
            num_workers=0, collate_fn=rate_collate_fn, drop_last=False)
    return loaders


def make_eval_token_loader(sids, targets, data, batch_size=64) -> DataLoader:
    """Create a single eval DataLoader with token-level embeddings."""
    ds = RateDataset(
        site_ids=sids, targets=targets,
        pooled_orig=data["pooled_orig"], pooled_edited=data["pooled_edited"],
        structure_delta=data["structure_delta"],
        tokens_orig=data["tokens_orig"], tokens_edited=data["tokens_edited"],
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=0, collate_fn=rate_collate_fn, drop_last=False)


def make_mm_loaders_from_splits(
    train_sids, train_targets, train_hand,
    val_sids, val_targets, val_hand,
    data, gnn_embeddings=None, batch_size=32,
) -> Dict[str, DataLoader]:
    """Create train/val multi-modal DataLoaders."""
    loaders = {}
    for name, sids, tgts, hand in [("train", train_sids, train_targets, train_hand),
                                    ("val", val_sids, val_targets, val_hand)]:
        ds = MultiModalRateDataset(
            site_ids=sids, targets=tgts,
            pooled_orig=data["pooled_orig"], pooled_edited=data["pooled_edited"],
            structure_delta=data["structure_delta"],
            hand_features=hand, gnn_embeddings=gnn_embeddings,
        )
        loaders[name] = DataLoader(
            ds, batch_size=batch_size, shuffle=(name == "train"),
            num_workers=0, collate_fn=mm_collate, drop_last=False)
    return loaders


def make_eval_mm_loader(sids, targets, hand, data, gnn_embeddings=None,
                        batch_size=64) -> DataLoader:
    """Create a single eval multi-modal DataLoader."""
    ds = MultiModalRateDataset(
        site_ids=sids, targets=targets,
        pooled_orig=data["pooled_orig"], pooled_edited=data["pooled_edited"],
        structure_delta=data["structure_delta"],
        hand_features=hand, gnn_embeddings=gnn_embeddings,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=0, collate_fn=mm_collate, drop_last=False)


# ---------------------------------------------------------------------------
# Main Experiment
# ---------------------------------------------------------------------------

def run_experiment():
    t_total = time.time()
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("CROSS-DATASET RATE PREDICTION EXPERIMENT")
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # Load ALL data once
    # ------------------------------------------------------------------
    data = load_all_data()
    df = data["df"]

    # Filter to sites with embeddings
    available = set(data["pooled_orig"].keys()) & set(data["pooled_edited"].keys())
    df = df[df["site_id"].isin(available)].copy().reset_index(drop=True)
    logger.info("Sites with embeddings: %d", len(df))

    # Ensure site_id is string for consistent lookups
    df["site_id"] = df["site_id"].astype(str)

    n_total = len(df)

    # Build per-dataset dataframes
    dataset_dfs = {}
    dataset_counts = {}
    for ds_name in ALL_DATASETS:
        ds_df = df[df["dataset_source"] == ds_name].copy().reset_index(drop=True)
        dataset_dfs[ds_name] = ds_df
        dataset_counts[ds_name] = len(ds_df)
        logger.info("  Dataset '%s': %d sites", ds_name, len(ds_df))

    # ------------------------------------------------------------------
    # Pre-compute hand features for ALL sites (keyed by dataset)
    # ------------------------------------------------------------------
    dataset_sids = {}
    dataset_targets = {}
    dataset_hand = {}
    for ds_name in ALL_DATASETS:
        ds_df = dataset_dfs[ds_name]
        sids = ds_df["site_id"].tolist()
        targets = ds_df["target"].values.astype(np.float32)
        hand = build_hand_features(sids, data["sequences"],
                                    data["structure_delta"], data["loop_df"])
        dataset_sids[ds_name] = sids
        dataset_targets[ds_name] = targets
        dataset_hand[ds_name] = hand

    # ------------------------------------------------------------------
    # Pre-compute GNN embeddings (random, once for all)
    # ------------------------------------------------------------------
    gnn_embeddings = None
    d_gnn = 64
    try:
        from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
        logger.info("Pre-computing random GNN embeddings for 4Way_heavyreg...")
        torch.manual_seed(SEED)
        gnn = SmallStructureGNN(12, 64, 64, 3, 4, 0.3)
        all_sids_list = df["site_id"].tolist()
        gnn_embeddings = precompute_gnn_embeddings(
            gnn, all_sids_list, data["sequences"], data["pairing_probs"])
        logger.info("  GNN embeddings: %d sites, %d-dim",
                    len(gnn_embeddings), d_gnn)
        del gnn
    except ImportError:
        logger.warning("torch_geometric not available, skipping 4Way_heavyreg model")

    # ------------------------------------------------------------------
    # Results accumulators
    # ------------------------------------------------------------------
    model_results = {}
    for model_name in MODEL_NAMES:
        model_results[model_name] = {
            "matrix_spearman": {ds: {} for ds in TRAIN_DATASETS},
            "matrix_r2": {ds: {} for ds in TRAIN_DATASETS},
        }

    # ------------------------------------------------------------------
    # Run cross-dataset experiments
    # ------------------------------------------------------------------
    for train_ds in TRAIN_DATASETS:
        logger.info("\n" + "=" * 70)
        logger.info("TRAINING ON: %s (%d sites)", train_ds, dataset_counts[train_ds])
        logger.info("=" * 70)

        train_sids_all = dataset_sids[train_ds]
        train_targets_all = dataset_targets[train_ds]
        train_hand_all = dataset_hand[train_ds]
        n_train_total = len(train_sids_all)

        # 80/20 train/val split
        rng = np.random.RandomState(SEED)
        perm = rng.permutation(n_train_total)
        n_val = int(n_train_total * 0.2)
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]

        train_sids = [train_sids_all[i] for i in train_idx]
        val_sids = [train_sids_all[i] for i in val_idx]
        train_targets = train_targets_all[train_idx]
        val_targets = train_targets_all[val_idx]
        train_hand = train_hand_all[train_idx]
        val_hand = train_hand_all[val_idx]

        logger.info("  train=%d, val=%d", len(train_sids), len(val_sids))

        # ==============================================================
        # 1. GB_HandFeatures
        # ==============================================================
        logger.info("  [1/3] GB_HandFeatures")
        try:
            from xgboost import XGBRegressor

            X_train_h = np.nan_to_num(train_hand, nan=0.0)
            X_val_h = np.nan_to_num(val_hand, nan=0.0)

            gb_hand = XGBRegressor(
                n_estimators=500, max_depth=5, learning_rate=0.05,
                subsample=0.8, min_child_weight=10,
                random_state=SEED, n_jobs=1,
                early_stopping_rounds=30, verbosity=0,
            )
            gb_hand.fit(X_train_h, train_targets,
                        eval_set=[(X_val_h, val_targets)], verbose=False)

            # Evaluate on ALL datasets
            for test_ds in ALL_DATASETS:
                test_hand_ds = np.nan_to_num(dataset_hand[test_ds], nan=0.0)
                y_pred = gb_hand.predict(test_hand_ds)
                y_true = dataset_targets[test_ds]
                metrics = compute_rate_metrics(y_true, y_pred)
                model_results["GB_HandFeatures"]["matrix_spearman"][train_ds][test_ds] = metrics["spearman"]
                model_results["GB_HandFeatures"]["matrix_r2"][train_ds][test_ds] = metrics["r2"]
                logger.info("    GB_HandFeatures [%s -> %s]: sp=%.4f, r2=%.4f",
                           train_ds, test_ds, metrics["spearman"], metrics["r2"])
            del gb_hand
        except ImportError:
            logger.warning("xgboost not available, skipping GB_HandFeatures")
            for test_ds in ALL_DATASETS:
                model_results["GB_HandFeatures"]["matrix_spearman"][train_ds][test_ds] = float("nan")
                model_results["GB_HandFeatures"]["matrix_r2"][train_ds][test_ds] = float("nan")

        # ==============================================================
        # 2. EditRNA_rate
        # ==============================================================
        logger.info("  [2/3] EditRNA_rate")
        try:
            torch.manual_seed(SEED)
            from models.editrna_a3a import EditRNA_A3A, EditRNAConfig
            from models.encoders import CachedRNAEncoder

            cached_encoder = CachedRNAEncoder(
                tokens_cache=data["tokens_orig"],
                pooled_cache=data["pooled_orig"],
                tokens_edited_cache=data["tokens_edited"],
                pooled_edited_cache=data["pooled_edited"],
                d_model=640,
            )
            config = EditRNAConfig(
                primary_encoder="cached", d_model=640, d_edit=128, d_fused=256,
                edit_n_heads=4,
                pooled_only=(data["tokens_orig"] is None),
                use_structure_delta=True,
                head_dropout=0.3, fusion_dropout=0.3,
                weight_decay=1e-2, learning_rate=1e-3,
            )
            editrna = EditRNA_A3A(config=config, primary_encoder=cached_encoder)
            editrna_wrapper = EditRNARateWrapper(editrna)

            # Create train/val loaders
            token_loaders = make_token_loaders_from_splits(
                train_sids, train_targets, val_sids, val_targets, data)

            editrna_wrapper = train_editrna_rate(
                editrna_wrapper, token_loaders,
                epochs=100, lr=1e-3, weight_decay=1e-2,
                patience=15, rate_weight=10.0)

            # Evaluate on ALL datasets
            for test_ds in ALL_DATASETS:
                test_loader = make_eval_token_loader(
                    dataset_sids[test_ds], dataset_targets[test_ds], data)
                metrics = evaluate_nn_model(editrna_wrapper, test_loader)
                model_results["EditRNA_rate"]["matrix_spearman"][train_ds][test_ds] = metrics["spearman"]
                model_results["EditRNA_rate"]["matrix_r2"][train_ds][test_ds] = metrics["r2"]
                logger.info("    EditRNA_rate [%s -> %s]: sp=%.4f, r2=%.4f",
                           train_ds, test_ds, metrics["spearman"], metrics["r2"])
            del editrna_wrapper, editrna, cached_encoder, token_loaders
        except Exception as e:
            logger.error("EditRNA_rate FAILED for train=%s: %s", train_ds, e)
            import traceback; traceback.print_exc()
            for test_ds in ALL_DATASETS:
                model_results["EditRNA_rate"]["matrix_spearman"][train_ds][test_ds] = float("nan")
                model_results["EditRNA_rate"]["matrix_r2"][train_ds][test_ds] = float("nan")

        # ==============================================================
        # 3. 4Way_heavyreg
        # ==============================================================
        logger.info("  [3/3] 4Way_heavyreg")
        if gnn_embeddings is not None:
            try:
                torch.manual_seed(SEED)
                mm_loaders = make_mm_loaders_from_splits(
                    train_sids, train_targets, train_hand,
                    val_sids, val_targets, val_hand,
                    data, gnn_embeddings=gnn_embeddings)

                d_hand = train_hand.shape[1]
                four_model = FourWayGatedFusion(
                    d_primary=640, d_hand=d_hand, d_gnn=d_gnn,
                    d_proj=128, dropout=0.5)
                four_model = train_nn_model(
                    four_model, mm_loaders, "4Way_heavyreg",
                    epochs=200, lr=5e-4, weight_decay=1e-2, patience=30)

                # Evaluate on ALL datasets
                for test_ds in ALL_DATASETS:
                    test_loader = make_eval_mm_loader(
                        dataset_sids[test_ds], dataset_targets[test_ds],
                        dataset_hand[test_ds], data,
                        gnn_embeddings=gnn_embeddings)
                    metrics = evaluate_nn_model(four_model, test_loader)
                    model_results["4Way_heavyreg"]["matrix_spearman"][train_ds][test_ds] = metrics["spearman"]
                    model_results["4Way_heavyreg"]["matrix_r2"][train_ds][test_ds] = metrics["r2"]
                    logger.info("    4Way_heavyreg [%s -> %s]: sp=%.4f, r2=%.4f",
                               train_ds, test_ds, metrics["spearman"], metrics["r2"])
                del four_model, mm_loaders
            except Exception as e:
                logger.error("4Way_heavyreg FAILED for train=%s: %s", train_ds, e)
                import traceback; traceback.print_exc()
                for test_ds in ALL_DATASETS:
                    model_results["4Way_heavyreg"]["matrix_spearman"][train_ds][test_ds] = float("nan")
                    model_results["4Way_heavyreg"]["matrix_r2"][train_ds][test_ds] = float("nan")
        else:
            logger.warning("Skipping 4Way_heavyreg (no GNN embeddings)")
            for test_ds in ALL_DATASETS:
                model_results["4Way_heavyreg"]["matrix_spearman"][train_ds][test_ds] = float("nan")
                model_results["4Way_heavyreg"]["matrix_r2"][train_ds][test_ds] = float("nan")

        gc.collect()

    # ------------------------------------------------------------------
    # Build and save results
    # ------------------------------------------------------------------
    total_time = time.time() - t_total

    results = {
        "experiment": "cross_dataset_rate",
        "n_total_sites": n_total,
        "target_transform": "log2(editing_rate_normalized + 0.01)",
        "datasets": dataset_counts,
        "models": {},
        "total_time_seconds": round(total_time, 1),
    }

    for model_name in MODEL_NAMES:
        results["models"][model_name] = {
            "matrix_spearman": model_results[model_name]["matrix_spearman"],
            "matrix_r2": model_results[model_name]["matrix_r2"],
        }

    out_path = OUTPUT_DIR / "cross_dataset_rate_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=_serialize)
    logger.info("\nResults saved to %s", out_path)

    # ------------------------------------------------------------------
    # Print summary tables
    # ------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("CROSS-DATASET RATE PREDICTION RESULTS")
    print(f"Target: {results['target_transform']}")
    print(f"Total sites: {n_total}")
    print(f"Datasets: {dataset_counts}")
    print("=" * 100)

    for model_name in MODEL_NAMES:
        print(f"\n--- {model_name} (Spearman / R2) ---")
        header_label = "Train \\ Test"
        print(f"{header_label:<20}", end="")
        for test_ds in ALL_DATASETS:
            print(f"  {test_ds:>15}", end="")
        print()
        print("-" * (20 + 17 * len(ALL_DATASETS)))

        mat_sp = model_results[model_name]["matrix_spearman"]
        mat_r2 = model_results[model_name]["matrix_r2"]

        for train_ds in TRAIN_DATASETS:
            print(f"{train_ds:<20}", end="")
            for test_ds in ALL_DATASETS:
                sp = mat_sp[train_ds].get(test_ds, float("nan"))
                if np.isnan(sp):
                    print(f"  {'N/A':>15}", end="")
                else:
                    r2 = mat_r2[train_ds].get(test_ds, float("nan"))
                    print(f"  {sp:>6.3f}/{r2:>6.3f}", end="")
            print()

    print("\n" + "=" * 100)
    print(f"Total time: {total_time:.1f} seconds ({total_time / 60:.1f} minutes)")

    return results


def _serialize(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return str(obj)


if __name__ == "__main__":
    run_experiment()
