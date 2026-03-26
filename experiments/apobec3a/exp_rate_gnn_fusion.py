#!/usr/bin/env python
"""Rate prediction with GNN structure embeddings + multi-modal fusion.

Goal: Beat GB_AllFeatures (Spearman=0.727) by combining:
  - RNA-FM pooled embeddings (640-dim)
  - Edit delta (pooled_edited - pooled_orig, 640-dim)
  - Hand-crafted features (40-dim: motif + structure delta + loop)
  - GNN structure embeddings from RNA 2D structure graph
  - Per-position pairing probability features

Approaches:
  1. GNN embeddings as extra features for GB (GNN → GB)
  2. 4-way Gated Fusion: primary + edit_delta + hand + GNN → gate → MLP
  3. Pairing profile features for GB (201-dim pairing probs → PCA/pooled)
  4. Ensemble of top models (average predictions)
  5. K-fold CV for stable estimates

Usage:
    python experiments/apobec3a/exp_rate_gnn_fusion.py
"""

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
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SPLITS_CSV = PROJECT_ROOT / "data" / "processed" / "splits_expanded.csv"
EMB_DIR = PROJECT_ROOT / "data" / "processed" / "embeddings"
STRUCTURE_CACHE = EMB_DIR / "vienna_structure_cache.npz"
SEQUENCES_JSON = PROJECT_ROOT / "data" / "processed" / "site_sequences.json"
LOOP_POS_CSV = (
    PROJECT_ROOT / "experiments" / "apobec3a" / "outputs"
    / "loop_position" / "loop_position_per_site.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "apobec3a" / "outputs" / "rate_gnn_fusion"
DEVICE = torch.device("cpu")
SEED = 42


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_rate_metrics(y_true, y_pred):
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) < 3:
        return {"spearman": float("nan"), "pearson": float("nan"),
                "mse": float("nan"), "r2": float("nan")}
    sp, _ = spearmanr(y_true, y_pred)
    pe, _ = pearsonr(y_true, y_pred)
    return {"spearman": float(sp), "pearson": float(pe),
            "mse": float(mean_squared_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred))}


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_all_data():
    """Load everything: splits, embeddings, structures, sequences, features.
    Memory-optimized: only keeps data for rate-prediction sites (~8K of 77K)."""
    logger.info("Loading data...")

    # Splits
    df = pd.read_csv(SPLITS_CSV)
    df = df[df["editing_rate_normalized"].notna()].copy()
    df["target"] = np.log2(df["editing_rate_normalized"].values + 0.01)
    split_dfs = {s: df[df["split"] == s].copy() for s in ["train", "val", "test"]}
    logger.info("  Rate splits: train=%d, val=%d, test=%d",
                len(split_dfs["train"]), len(split_dfs["val"]), len(split_dfs["test"]))

    # Collect needed site IDs to filter large dicts
    needed_sids = set(df["site_id"].astype(str).tolist())
    logger.info("  Filtering to %d rate-prediction sites", len(needed_sids))

    import gc

    # Embeddings (filter to needed sites only, with explicit GC)
    _pooled_orig = torch.load(EMB_DIR / "rnafm_pooled.pt", weights_only=False)
    pooled_orig = {k: v for k, v in _pooled_orig.items() if str(k) in needed_sids}
    del _pooled_orig
    gc.collect()
    _pooled_edited = torch.load(EMB_DIR / "rnafm_pooled_edited.pt", weights_only=False)
    pooled_edited = {k: v for k, v in _pooled_edited.items() if str(k) in needed_sids}
    del _pooled_edited
    gc.collect()
    logger.info("  %d pooled embeddings (filtered)", len(pooled_orig))

    # Structure cache - use array-based access (avoid per-element dicts for memory)
    struct_data = np.load(str(STRUCTURE_CACHE), allow_pickle=True)
    struct_sids = [str(s) for s in struct_data["site_ids"]]
    sid_to_idx = {sid: i for i, sid in enumerate(struct_sids) if sid in needed_sids}
    idx_arr = np.array(sorted(sid_to_idx.values()))  # sorted indices for bulk slicing
    idx_sids = [struct_sids[i] for i in idx_arr]
    logger.info("  Loading structure arrays for %d sites (bulk)...", len(idx_arr))

    # Bulk extract needed rows (single array access per field)
    _pp_all = struct_data["pairing_probs"]
    pairing_probs = dict(zip(idx_sids, _pp_all[idx_arr]))
    del _pp_all

    _ppe_all = struct_data["pairing_probs_edited"]
    pairing_probs_edited = dict(zip(idx_sids, _ppe_all[idx_arr]))
    del _ppe_all

    _acc_all = struct_data["accessibilities"]
    accessibilities = dict(zip(idx_sids, _acc_all[idx_arr]))
    del _acc_all

    _acce_all = struct_data["accessibilities_edited"]
    accessibilities_edited = dict(zip(idx_sids, _acce_all[idx_arr]))
    del _acce_all

    _sd_all = struct_data["delta_features"]
    structure_delta = dict(zip(idx_sids, _sd_all[idx_arr]))
    del _sd_all

    _m = struct_data["mfes"]
    mfes = {sid: float(_m[i]) for sid, i in sid_to_idx.items()}
    del _m

    _me = struct_data["mfes_edited"]
    mfes_edited = {sid: float(_me[i]) for sid, i in sid_to_idx.items()}
    del _me

    del struct_data, struct_sids, sid_to_idx, idx_arr, idx_sids
    gc.collect()
    logger.info("  %d structure caches loaded (filtered)", len(pairing_probs))

    # Sequences (filter to needed sites)
    with open(SEQUENCES_JSON) as f:
        _sequences = json.load(f)
    sequences = {k: v for k, v in _sequences.items() if str(k) in needed_sids}
    del _sequences
    logger.info("  %d sequences (filtered)", len(sequences))

    # Loop features
    loop_df = pd.DataFrame()
    if LOOP_POS_CSV.exists():
        loop_df = pd.read_csv(LOOP_POS_CSV).set_index("site_id")
        loop_df = loop_df[loop_df.index.isin(needed_sids)]
        logger.info("  %d loop features (filtered)", len(loop_df))

    return {
        "df": df, "split_dfs": split_dfs,
        "pooled_orig": pooled_orig, "pooled_edited": pooled_edited,
        "pairing_probs": pairing_probs, "pairing_probs_edited": pairing_probs_edited,
        "accessibilities": accessibilities, "accessibilities_edited": accessibilities_edited,
        "structure_delta": structure_delta,
        "mfes": mfes, "mfes_edited": mfes_edited,
        "sequences": sequences, "loop_df": loop_df,
    }


# ---------------------------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------------------------

def extract_motif_features(sequences, site_ids):
    """24-dim motif features."""
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


def extract_loop_features(loop_df, site_ids):
    """9-dim loop position features."""
    cols = ["is_unpaired", "loop_size", "dist_to_junction", "dist_to_apex",
            "relative_loop_position", "left_stem_length", "right_stem_length",
            "max_adjacent_stem_length", "local_unpaired_fraction"]
    features = []
    for sid in site_ids:
        if sid in loop_df.index:
            features.append(loop_df.loc[sid, cols].values.astype(np.float32))
        else:
            features.append(np.zeros(len(cols), dtype=np.float32))
    return np.array(features, dtype=np.float32)


def extract_structure_delta(structure_delta, site_ids):
    return np.array([structure_delta.get(sid, np.zeros(7)) for sid in site_ids], dtype=np.float32)


def extract_pairing_profile_features(pairing_probs, pairing_probs_edited,
                                      accessibilities, accessibilities_edited,
                                      mfes, mfes_edited, site_ids):
    """Extract rich structure profile features from ViennaRNA cache.

    For each site, creates:
    - Local window pairing prob stats (orig + edited + delta): 15 features
    - Local window accessibility stats: 15 features
    - Center pairing prob / accessibility: 6 features
    - MFE and MFE delta: 3 features
    - Pairing prob change profile (binned): 10 features
    Total: ~49 features
    """
    features = []
    for sid in site_ids:
        pp = pairing_probs.get(sid, np.zeros(201))
        pp_ed = pairing_probs_edited.get(sid, np.zeros(201))
        acc = accessibilities.get(sid, np.zeros(201))
        acc_ed = accessibilities_edited.get(sid, np.zeros(201))
        mfe = mfes.get(sid, 0.0)
        mfe_ed = mfes_edited.get(sid, 0.0)

        ep = 100  # edit position
        feat = []

        # Local window pairing prob stats (windows: 5, 11, 21)
        for w in [5, 11, 21]:
            s, e = max(0, ep - w // 2), min(201, ep + w // 2 + 1)
            win_pp = pp[s:e]
            win_pp_ed = pp_ed[s:e]
            win_delta = win_pp_ed - win_pp
            feat.extend([win_pp.mean(), win_pp.std(),
                         win_pp_ed.mean(), win_pp_ed.std(),
                         win_delta.mean()])

        # Local window accessibility stats
        for w in [5, 11, 21]:
            s, e = max(0, ep - w // 2), min(201, ep + w // 2 + 1)
            win_acc = acc[s:e]
            win_acc_ed = acc_ed[s:e]
            win_delta = win_acc_ed - win_acc
            feat.extend([win_acc.mean(), win_acc.std(),
                         win_acc_ed.mean(), win_acc_ed.std(),
                         win_delta.mean()])

        # Center position features
        feat.extend([
            pp[ep], pp_ed[ep], pp_ed[ep] - pp[ep],
            acc[ep], acc_ed[ep], acc_ed[ep] - acc[ep],
        ])

        # MFE features
        feat.extend([mfe, mfe_ed, mfe_ed - mfe])

        # Pairing prob change profile: bin the 201-pos delta into 10 bins
        delta_pp = pp_ed - pp
        binned = []
        for i in range(10):
            s = i * 20
            e = min(201, (i + 1) * 20)
            binned.append(delta_pp[s:e].mean())
        # Plus center bin (positions 90-110)
        binned.append(delta_pp[90:111].mean())
        feat.extend(binned)

        features.append(feat)

    return np.array(features, dtype=np.float32)


def build_all_features(site_ids, data, include_emb=True, include_pairing=True):
    """Build the full feature matrix combining all feature types."""
    parts = []
    part_names = []

    # Motif (24)
    motif = extract_motif_features(data["sequences"], site_ids)
    parts.append(motif)
    part_names.append(f"motif({motif.shape[1]})")

    # Structure delta (7)
    struct = extract_structure_delta(data["structure_delta"], site_ids)
    parts.append(struct)
    part_names.append(f"struct_delta({struct.shape[1]})")

    # Loop (9)
    loop = extract_loop_features(data["loop_df"], site_ids)
    parts.append(loop)
    part_names.append(f"loop({loop.shape[1]})")

    # Pairing profile features (50)
    if include_pairing:
        pairing = extract_pairing_profile_features(
            data["pairing_probs"], data["pairing_probs_edited"],
            data["accessibilities"], data["accessibilities_edited"],
            data["mfes"], data["mfes_edited"], site_ids)
        parts.append(pairing)
        part_names.append(f"pairing_profile({pairing.shape[1]})")

    # Embedding delta (640)
    if include_emb:
        emb_delta = np.array([
            (data["pooled_edited"][sid] - data["pooled_orig"][sid]).numpy()
            if sid in data["pooled_orig"] and sid in data["pooled_edited"]
            else np.zeros(640, dtype=np.float32)
            for sid in site_ids
        ], dtype=np.float32)
        parts.append(emb_delta)
        part_names.append(f"emb_delta({emb_delta.shape[1]})")

    all_feats = np.concatenate(parts, axis=1)
    all_feats = np.nan_to_num(all_feats, nan=0.0)
    logger.info("  Features: %s = %d dims", " + ".join(part_names), all_feats.shape[1])
    return all_feats


# ---------------------------------------------------------------------------
# GNN Graph Construction + Embedding
# ---------------------------------------------------------------------------

def build_structure_graph_from_pairing(pp, sequence, edit_pos=100, threshold=0.3):
    """Build a PyG graph from pairing probabilities (soft structure).

    Nodes = nucleotides, edges = backbone + structure.
    Structure edges connect positions with high mutual pairing probability.
    """
    from torch_geometric.data import Data

    n = len(sequence)
    seq = sequence.upper().replace("T", "U")

    # Node features (12-dim)
    nuc_to_idx = {"A": 0, "C": 1, "G": 2, "U": 3, "N": 4}
    x = torch.zeros(n, 12)
    for i, nuc in enumerate(seq):
        x[i, nuc_to_idx.get(nuc, 4)] = 1.0  # one-hot (5)
    # Relative position
    x[:, 5] = torch.linspace(-1, 1, n)
    # Distance from edit site
    x[:, 6] = torch.abs(torch.arange(n, dtype=torch.float32) - edit_pos) / 100.0
    # Is edit site
    x[edit_pos, 7] = 1.0
    # Pairing probability at each position
    x[:, 8] = torch.tensor(pp[:n], dtype=torch.float32)
    # Paired vs unpaired (threshold)
    x[:, 9] = (torch.tensor(pp[:n]) > 0.5).float()
    # Local pairing density (11nt window)
    for i in range(n):
        s, e = max(0, i - 5), min(n, i + 6)
        x[i, 10] = float(pp[s:e].mean())
    # GC content in local window
    for i in range(n):
        s, e = max(0, i - 5), min(n, i + 6)
        win = seq[s:e]
        x[i, 11] = float((win.count("G") + win.count("C")) / max(len(win), 1))

    # Edges
    edges_src, edges_dst = [], []

    # Backbone edges (i <-> i+1)
    for i in range(n - 1):
        edges_src.extend([i, i + 1])
        edges_dst.extend([i + 1, i])

    # Structure edges from pairing probabilities
    # Use a simple heuristic: positions with high pairing prob likely pair with
    # complementary positions. We connect positions within stems.
    # For simplicity, connect positions that are both highly paired and
    # at complementary distance from the stem boundary.
    # Alternative: threshold on pairing probability pattern
    paired_positions = np.where(pp[:n] > threshold)[0]
    # Connect paired positions that form plausible base pairs
    # (opposite sides of a stem: i paired with j where j > i)
    stack = []
    dot_bracket = []
    for i in range(n):
        if pp[i] > threshold:
            # Heuristic: if pairing prob rises then falls, it's a stem
            if i > 0 and pp[i] > pp[i - 1]:
                stack.append(i)
                dot_bracket.append("(")
            elif stack:
                j = stack.pop()
                edges_src.extend([i, j])
                edges_dst.extend([j, i])
                dot_bracket.append(")")
            else:
                dot_bracket.append(".")
        else:
            dot_bracket.append(".")

    # Also add long-range contacts for positions with very high pairing prob
    # Connect positions symmetric around loops
    for i in range(n):
        if pp[i] > 0.7:
            # Find closest position with similar pairing on opposite side
            for j in range(i + 4, min(i + 100, n)):
                if pp[j] > 0.7 and abs(pp[i] - pp[j]) < 0.2:
                    # Check if they could form a base pair
                    b1, b2 = seq[i], seq[j]
                    if (b1, b2) in {("A", "U"), ("U", "A"), ("G", "C"), ("C", "G"),
                                     ("G", "U"), ("U", "G")}:
                        edges_src.extend([i, j])
                        edges_dst.extend([j, i])
                        break

    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)

    return Data(x=x, edge_index=edge_index)


def precompute_gnn_embeddings(gnn_model, site_ids, sequences, pairing_probs,
                               batch_size=64):
    """Pre-compute GNN graph embeddings for all sites."""
    from torch_geometric.data import Batch

    gnn_model.eval()
    embeddings = {}
    batched_ids = [site_ids[i:i + batch_size] for i in range(0, len(site_ids), batch_size)]

    for batch_ids in batched_ids:
        graphs = []
        valid_ids = []
        for sid in batch_ids:
            seq = sequences.get(sid, "N" * 201)
            pp = pairing_probs.get(sid, np.zeros(201))
            if len(seq) >= 201 and len(pp) >= 201:
                g = build_structure_graph_from_pairing(pp, seq[:201])
                graphs.append(g)
                valid_ids.append(sid)

        if not graphs:
            continue

        batch_data = Batch.from_data_list(graphs)
        with torch.no_grad():
            emb = gnn_model(batch_data)  # (batch_size, output_dim)

        for i, sid in enumerate(valid_ids):
            embeddings[sid] = emb[i].numpy()

    return embeddings


# ---------------------------------------------------------------------------
# GNN Model (small, for rate prediction)
# ---------------------------------------------------------------------------

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
            nn.Linear(hidden_dim * 2, output_dim),  # mean + max pool
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

        # Mean + max pooling
        x_mean = self.mean_pool(x, batch)
        x_max = self.max_pool(x, batch)
        pooled = torch.cat([x_mean, x_max], dim=-1)
        return self.output_proj(pooled)


# ---------------------------------------------------------------------------
# Multi-modal Fusion Models
# ---------------------------------------------------------------------------

class FourWayGatedFusion(nn.Module):
    """4-way gated fusion: primary + edit_delta + hand + GNN → rate.

    Inspired by AdarEdit's multi-modal approach:
    - Primary: RNA-FM pooled embedding (background context)
    - Edit delta: (edited - orig) embedding (edit effect signal)
    - Hand features: motif + structure + loop (explicit biology)
    - GNN: structure graph embedding (2D structure context)
    """
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
            nn.Linear(d_proj * n_mod, d_proj * 2),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(d_proj * 2, d_proj),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_proj, 1),
        )

    def forward(self, batch):
        p = self.primary_proj(batch["pooled_orig"])
        e = self.edit_proj(batch["pooled_edited"] - batch["pooled_orig"])
        h = self.hand_proj(batch["hand_features"])
        g = self.gnn_proj(batch["gnn_emb"])

        concat = torch.cat([p, e, h, g], dim=-1)
        gates = torch.softmax(self.gate(concat), dim=-1)

        gated = torch.cat([
            gates[:, 0:1] * p,
            gates[:, 1:2] * e,
            gates[:, 2:3] * h,
            gates[:, 3:4] * g,
        ], dim=-1)

        fused = self.fuse(gated)
        return {"rate_pred": self.head(fused)}


class ThreeWayGatedFusion(nn.Module):
    """3-way gated fusion: edit_delta + hand + pairing_profile → rate."""
    def __init__(self, d_edit=640, d_hand=40, d_pairing=50, d_proj=128, dropout=0.3):
        super().__init__()
        self.edit_proj = nn.Sequential(
            nn.Linear(d_edit, d_proj), nn.GELU(), nn.LayerNorm(d_proj))
        self.hand_proj = nn.Sequential(
            nn.Linear(d_hand, d_proj), nn.GELU(), nn.LayerNorm(d_proj))
        self.pairing_proj = nn.Sequential(
            nn.Linear(d_pairing, d_proj), nn.GELU(), nn.LayerNorm(d_proj))

        n_mod = 3
        self.gate = nn.Linear(d_proj * n_mod, n_mod)
        self.fuse = nn.Sequential(
            nn.Linear(d_proj * n_mod, d_proj * 2),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(d_proj * 2, d_proj),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_proj, 1),
        )

    def forward(self, batch):
        e = self.edit_proj(batch["edit_delta"])
        h = self.hand_proj(batch["hand_features"])
        p = self.pairing_proj(batch["pairing_features"])

        concat = torch.cat([e, h, p], dim=-1)
        gates = torch.softmax(self.gate(concat), dim=-1)
        gated = torch.cat([gates[:, i:i+1] * x for i, x in enumerate([e, h, p])], dim=-1)
        return {"rate_pred": self.head(self.fuse(gated))}


# ---------------------------------------------------------------------------
# Dataset & DataLoader
# ---------------------------------------------------------------------------

class MultiModalRateDataset(Dataset):
    def __init__(self, site_ids, targets, pooled_orig, pooled_edited,
                 hand_features, pairing_features=None, gnn_embeddings=None):
        self.site_ids = site_ids
        self.targets = targets
        self.pooled_orig = pooled_orig
        self.pooled_edited = pooled_edited
        self.hand_features = hand_features
        self.pairing_features = pairing_features
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
            "hand_features": torch.tensor(self.hand_features[idx], dtype=torch.float32),
        }
        if self.pairing_features is not None:
            item["pairing_features"] = torch.tensor(
                self.pairing_features[idx], dtype=torch.float32)
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
        "hand_features": torch.stack([b["hand_features"] for b in batch]),
    }
    if "pairing_features" in batch[0]:
        result["pairing_features"] = torch.stack([b["pairing_features"] for b in batch])
    if "gnn_emb" in batch[0]:
        result["gnn_emb"] = torch.stack([b["gnn_emb"] for b in batch])
    result["edit_delta"] = result["pooled_edited"] - result["pooled_orig"]
    return result


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_nn(model, loaders, model_name, epochs=200, lr=1e-3,
             weight_decay=1e-2, patience=25):
    model = model.to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)
    n_params = sum(p.numel() for p in model.parameters())

    best_sp, best_epoch, pat = -float("inf"), 0, 0
    best_state = None

    logger.info("Training %s (params=%s)...", model_name, f"{n_params:,}")
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, nb = 0, 0
        for batch in loaders["train"]:
            batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            optimizer.zero_grad()
            pred = model(batch)["rate_pred"].squeeze(-1)
            loss = F.mse_loss(pred, batch["targets"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            nb += 1
        scheduler.step()

        val_m = eval_nn(model, loaders.get("val"))
        val_sp = val_m.get("spearman", -999)
        if not np.isnan(val_sp) and val_sp > best_sp + 1e-5:
            best_sp, best_epoch, pat = val_sp, epoch, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            pat += 1
        if epoch % 20 == 0 or pat >= patience:
            logger.info("  E%3d loss=%.4f val_sp=%.4f best=%.4f pat=%d/%d",
                        epoch, total_loss / max(nb, 1), val_sp if not np.isnan(val_sp) else 0,
                        best_sp, pat, patience)
        if pat >= patience:
            break

    if best_state:
        model.load_state_dict(best_state)

    results = {"best_epoch": best_epoch, "time_seconds": round(time.time() - t0, 1),
               "n_params": n_params}
    for s in ["train", "val", "test"]:
        if s in loaders:
            results[s] = eval_nn(model, loaders[s])
    return results


@torch.no_grad()
def eval_nn(model, loader):
    if not loader:
        return {}
    model.eval()
    at, ap = [], []
    for batch in loader:
        batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        at.append(batch["targets"].numpy())
        ap.append(model(batch)["rate_pred"].squeeze(-1).numpy())
    return compute_rate_metrics(np.concatenate(at), np.concatenate(ap))


def run_with_seeds(factory, name, loaders, kw, seeds=(42, 123, 456)):
    results = []
    for seed in seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        r = train_nn(factory(), loaders, f"{name}_s{seed}", **kw)
        results.append(r)
        logger.info("  %s seed=%d: test_sp=%.4f", name, seed, r["test"]["spearman"])

    sps = [r["test"]["spearman"] for r in results if not np.isnan(r["test"]["spearman"])]
    return {
        "mean_test_spearman": float(np.mean(sps)) if sps else float("nan"),
        "std_test_spearman": float(np.std(sps)) if len(sps) > 1 else 0.0,
        "per_seed": results,
    }


# ---------------------------------------------------------------------------
# GB with enriched features
# ---------------------------------------------------------------------------

def train_gb_with_features(feature_name, X_train, y_train, X_val, y_val, X_test, y_test):
    from xgboost import XGBRegressor
    t0 = time.time()
    gb = XGBRegressor(
        n_estimators=500, max_depth=5, learning_rate=0.01,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        early_stopping_rounds=20, random_state=42,
    )
    gb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    elapsed = time.time() - t0

    result = {"time_seconds": round(elapsed, 1), "n_features": X_train.shape[1]}
    for name, X, y in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
        result[name] = compute_rate_metrics(y, gb.predict(X))
    logger.info("  %s: test_sp=%.4f (n_feat=%d)", feature_name,
                result["test"]["spearman"], X_train.shape[1])
    return result, gb


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------

def ensemble_predictions(predictions_list, y_test):
    """Average predictions from multiple models."""
    avg_pred = np.mean(predictions_list, axis=0)
    return compute_rate_metrics(y_test, avg_pred)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment():
    t_total = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    logger.info("=" * 70)
    logger.info("RATE PREDICTION: GNN FUSION EXPERIMENT")
    logger.info("=" * 70)

    data = load_all_data()
    split_dfs = data["split_dfs"]
    all_results = {
        "experiment": "rate_gnn_fusion",
        "n_train": len(split_dfs["train"]),
        "n_val": len(split_dfs["val"]),
        "n_test": len(split_dfs["test"]),
        "models": {},
    }

    # Filter to sites with embeddings
    available = set(data["pooled_orig"].keys()) & set(data["pooled_edited"].keys())
    for s in split_dfs:
        split_dfs[s] = split_dfs[s][split_dfs[s]["site_id"].isin(available)].copy()
    logger.info("After embedding filter: train=%d, val=%d, test=%d",
                len(split_dfs["train"]), len(split_dfs["val"]), len(split_dfs["test"]))

    # ==================================================================
    # Phase 1: Feature-based models (GB with various feature combos)
    # ==================================================================
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 1: GRADIENT BOOSTING WITH ENRICHED FEATURES")
    logger.info("=" * 70)

    gb_predictions = {}  # Store test predictions for ensemble

    for feat_name, include_emb, include_pairing in [
        ("GB_Hand", False, False),
        ("GB_Hand+Pairing", False, True),
        ("GB_All", True, False),
        ("GB_All+Pairing", True, True),
    ]:
        logger.info("\n--- %s ---", feat_name)
        splits_data = {}
        for sn, sdf in split_dfs.items():
            sids = sdf["site_id"].tolist()
            targets = sdf["target"].values.astype(np.float32)
            feats = build_all_features(sids, data, include_emb=include_emb,
                                       include_pairing=include_pairing)
            splits_data[sn] = (feats, targets, sids)

        result, gb_model = train_gb_with_features(
            feat_name,
            splits_data["train"][0], splits_data["train"][1],
            splits_data["val"][0], splits_data["val"][1],
            splits_data["test"][0], splits_data["test"][1],
        )
        all_results["models"][feat_name] = result
        gb_predictions[feat_name] = gb_model.predict(splits_data["test"][0])

    # ==================================================================
    # Phase 2: GNN embeddings
    # ==================================================================
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 2: GNN STRUCTURE EMBEDDINGS")
    logger.info("=" * 70)

    gnn_embeddings = None
    try:
        from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool

        # Train GNN end-to-end on rate prediction first, then extract embeddings
        gnn = SmallStructureGNN(node_features=12, hidden_dim=64, output_dim=64,
                                num_layers=3, heads=4, dropout=0.3)
        logger.info("GNN parameters: %s", f"{sum(p.numel() for p in gnn.parameters()):,}")

        # Pre-compute GNN embeddings (untrained - just structural features)
        all_sids = []
        for sdf in split_dfs.values():
            all_sids.extend(sdf["site_id"].tolist())
        all_sids = list(set(all_sids))

        logger.info("Pre-computing GNN embeddings for %d sites...", len(all_sids))
        gnn_embeddings = precompute_gnn_embeddings(
            gnn, all_sids, data["sequences"], data["pairing_probs"])
        logger.info("  GNN embeddings: %d sites, %d-dim",
                    len(gnn_embeddings), next(iter(gnn_embeddings.values())).shape[0])

        # GB with GNN embeddings
        for feat_name, include_emb in [("GB_All+GNN", True), ("GB_Hand+GNN", False)]:
            logger.info("\n--- %s ---", feat_name)
            splits_data = {}
            for sn, sdf in split_dfs.items():
                sids = sdf["site_id"].tolist()
                targets = sdf["target"].values.astype(np.float32)
                base_feats = build_all_features(sids, data, include_emb=include_emb,
                                                include_pairing=True)
                gnn_feats = np.array([gnn_embeddings.get(sid, np.zeros(64))
                                      for sid in sids], dtype=np.float32)
                feats = np.concatenate([base_feats, gnn_feats], axis=1)
                splits_data[sn] = (feats, targets, sids)

            result, gb_model = train_gb_with_features(
                feat_name,
                splits_data["train"][0], splits_data["train"][1],
                splits_data["val"][0], splits_data["val"][1],
                splits_data["test"][0], splits_data["test"][1],
            )
            all_results["models"][feat_name] = result
            gb_predictions[feat_name] = gb_model.predict(splits_data["test"][0])

    except ImportError:
        logger.warning("torch_geometric not available, skipping GNN models")

    # ==================================================================
    # Phase 3: Neural fusion models
    # ==================================================================
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 3: NEURAL FUSION MODELS")
    logger.info("=" * 70)

    SEEDS = (42, 123, 456)

    # Build features for DataLoaders
    hand_features = {}
    pairing_features = {}
    for sn, sdf in split_dfs.items():
        sids = sdf["site_id"].tolist()
        motif = extract_motif_features(data["sequences"], sids)
        struct = extract_structure_delta(data["structure_delta"], sids)
        loop = extract_loop_features(data["loop_df"], sids)
        hand_features[sn] = np.nan_to_num(np.concatenate([motif, struct, loop], axis=1))

        pair = extract_pairing_profile_features(
            data["pairing_probs"], data["pairing_probs_edited"],
            data["accessibilities"], data["accessibilities_edited"],
            data["mfes"], data["mfes_edited"], sids)
        pairing_features[sn] = np.nan_to_num(pair)

    d_hand = hand_features["train"].shape[1]
    d_pairing = pairing_features["train"].shape[1]
    d_gnn = 64 if gnn_embeddings else 0
    logger.info("d_hand=%d, d_pairing=%d, d_gnn=%d", d_hand, d_pairing, d_gnn)

    def make_loaders(use_gnn=False, use_pairing=False, batch_size=32):
        loaders = {}
        for sn, sdf in split_dfs.items():
            sids = sdf["site_id"].tolist()
            targets = sdf["target"].values.astype(np.float32)
            ds = MultiModalRateDataset(
                site_ids=sids, targets=targets,
                pooled_orig=data["pooled_orig"], pooled_edited=data["pooled_edited"],
                hand_features=hand_features[sn],
                pairing_features=pairing_features[sn] if use_pairing else None,
                gnn_embeddings=gnn_embeddings if use_gnn else None,
            )
            loaders[sn] = DataLoader(ds, batch_size=batch_size,
                                     shuffle=(sn == "train"), num_workers=0,
                                     collate_fn=mm_collate)
        return loaders

    nn_predictions = {}
    nn_configs = []

    # 3-way: edit_delta + hand + pairing
    loaders_3way = make_loaders(use_pairing=True)
    nn_configs.append(("3Way_small", loaders_3way,
                       lambda: ThreeWayGatedFusion(d_hand=d_hand, d_pairing=d_pairing,
                                                    d_proj=64, dropout=0.3),
                       {"lr": 1e-3, "weight_decay": 5e-3, "patience": 25}))
    nn_configs.append(("3Way_med", loaders_3way,
                       lambda: ThreeWayGatedFusion(d_hand=d_hand, d_pairing=d_pairing,
                                                    d_proj=128, dropout=0.3),
                       {"lr": 1e-3, "weight_decay": 5e-3, "patience": 25}))
    nn_configs.append(("3Way_heavyreg", loaders_3way,
                       lambda: ThreeWayGatedFusion(d_hand=d_hand, d_pairing=d_pairing,
                                                    d_proj=128, dropout=0.5),
                       {"lr": 5e-4, "weight_decay": 1e-2, "patience": 30}))

    # 4-way: primary + edit_delta + hand + GNN
    if gnn_embeddings:
        loaders_4way = make_loaders(use_gnn=True)
        nn_configs.append(("4Way_small", loaders_4way,
                           lambda: FourWayGatedFusion(d_hand=d_hand, d_gnn=d_gnn,
                                                       d_proj=64, dropout=0.3),
                           {"lr": 1e-3, "weight_decay": 5e-3, "patience": 25}))
        nn_configs.append(("4Way_med", loaders_4way,
                           lambda: FourWayGatedFusion(d_hand=d_hand, d_gnn=d_gnn,
                                                       d_proj=128, dropout=0.3),
                           {"lr": 1e-3, "weight_decay": 5e-3, "patience": 25}))
        nn_configs.append(("4Way_heavyreg", loaders_4way,
                           lambda: FourWayGatedFusion(d_hand=d_hand, d_gnn=d_gnn,
                                                       d_proj=128, dropout=0.5),
                           {"lr": 5e-4, "weight_decay": 1e-2, "patience": 30}))

    for name, loaders, factory, kw in nn_configs:
        logger.info("\n--- %s ---", name)
        result = run_with_seeds(factory, name, loaders, {"epochs": 200, **kw}, seeds=SEEDS)
        all_results["models"][name] = result
        logger.info(">> %s: mean_sp=%.4f +/- %.4f",
                    name, result["mean_test_spearman"], result["std_test_spearman"])

    # ==================================================================
    # Phase 4: Ensemble
    # ==================================================================
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 4: ENSEMBLE")
    logger.info("=" * 70)

    test_targets = split_dfs["test"]["target"].values.astype(np.float32)

    # Collect GB predictions
    ensemble_configs = [
        ("Ensemble_GB2", ["GB_All", "GB_All+Pairing"]),
        ("Ensemble_GB3", ["GB_All", "GB_All+Pairing", "GB_Hand+Pairing"]),
    ]
    if "GB_All+GNN" in gb_predictions:
        ensemble_configs.append(("Ensemble_GB4",
            ["GB_All", "GB_All+Pairing", "GB_All+GNN", "GB_Hand+GNN"]))

    for ens_name, model_names in ensemble_configs:
        preds = [gb_predictions[m] for m in model_names if m in gb_predictions]
        if len(preds) >= 2:
            result = ensemble_predictions(preds, test_targets)
            all_results["models"][ens_name] = {"test": result, "n_models": len(preds),
                                               "components": model_names}
            logger.info("  %s: test_sp=%.4f (from %d models)",
                        ens_name, result["spearman"], len(preds))

    # ==================================================================
    # Summary
    # ==================================================================
    total_time = time.time() - t_total
    all_results["total_time_seconds"] = round(total_time, 1)

    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info("%-30s  %10s  %10s", "Model", "Test Sp", "Params/Feats")
    logger.info("-" * 55)

    summary = []
    for name, m in all_results["models"].items():
        if "mean_test_spearman" in m:
            sp = m["mean_test_spearman"]
            info = m["per_seed"][0].get("n_params", "?")
        else:
            sp = m.get("test", {}).get("spearman", float("nan"))
            info = m.get("n_features", m.get("n_models", "?"))
        summary.append((name, sp, info))

    for n, sp, info in sorted(summary, key=lambda x: -x[1] if not np.isnan(x[1]) else -999):
        logger.info("%-30s  %10.4f  %10s", n, sp, str(info))

    out_path = OUTPUT_DIR / "rate_gnn_fusion_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("\nResults saved to %s", out_path)
    logger.info("Total time: %.1f minutes", total_time / 60)


if __name__ == "__main__":
    run_experiment()
