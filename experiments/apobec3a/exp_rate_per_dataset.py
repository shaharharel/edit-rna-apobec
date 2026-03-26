#!/usr/bin/env python
"""Within-dataset 5-fold CV + cross-dataset generalization for rate prediction.

Part 1: Within-Dataset CV
For each dataset (baysal_2016, advisor_c2t, alqassim_2021):
  - Filter to that dataset's positives only
  - Run 5-fold KFold CV (shuffle=True, random_state=42)
  - Target: log2(editing_rate_normalized + 0.01) -- NOT Z-scored
  - Run 6 models (skip neural models for small datasets n<500)

Part 2: Cross-Dataset Generalization
  - Train on each dataset, test on every other dataset
  - Models: GB_HandFeatures, EditRNA_rate
  - Reports Spearman, Pearson, R-squared

Data sources:
  - baysal_2016: from splits_expanded.csv (4,208 A3A sites with rates)
  - advisor_c2t: from splits_expanded_a3a.csv (120 A3A-only sites with rates)
  - alqassim_2021: from splits_expanded_a3a.csv (128 sites with rates)

Models:
  1. Mean Baseline       (all datasets)
  2. StructureOnly       (all datasets)
  3. GB_HandFeatures     (all datasets)
  4. GB_AllFeatures      (all datasets)
  5. EditRNA_rate        (baysal_2016 only, n>=500)
  6. 4Way_heavyreg       (baysal_2016 only, n>=500)

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE python experiments/apobec3a/exp_rate_per_dataset.py
"""

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
from sklearn.model_selection import KFold
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data" / "processed"
EMB_DIR = DATA_DIR / "embeddings"
SPLITS_CSV = DATA_DIR / "splits_expanded.csv"
SPLITS_A3A_CSV = DATA_DIR / "splits_expanded_a3a.csv"
STRUCTURE_CACHE = EMB_DIR / "vienna_structure_cache.npz"
SEQUENCES_JSON = DATA_DIR / "site_sequences.json"
LOOP_POS_CSV = (
    PROJECT_ROOT / "experiments" / "apobec3a" / "outputs"
    / "loop_position" / "loop_position_per_site.csv"
)
OUTPUT_DIR = Path(__file__).parent / "outputs" / "rate_per_dataset"

SEED = 42
DEVICE = torch.device("cpu")

# Datasets to evaluate (skip sharma_2015 with only 6 sites)
DATASETS = ["baysal_2016", "advisor_c2t", "alqassim_2021"]
# Run neural models on all datasets (including small ones for completeness)
NEURAL_DATASETS = {"baysal_2016", "advisor_c2t", "alqassim_2021"}


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
    loop features, pairing profiles.  Memory-filtered to rate-annotated sites.
    Target: log2(editing_rate_normalized + 0.01) -- NOT Z-scored.

    Data sources:
      - baysal_2016: from splits_expanded.csv (4,208 A3A sites)
      - advisor_c2t: from splits_expanded_a3a.csv (120 A3A-only sites)
      - alqassim_2021: from splits_expanded_a3a.csv (128 sites)
    """
    logger.info("Loading data...")

    # Load baysal from full splits (not in A3A file, but confirmed A3A)
    df_full = pd.read_csv(SPLITS_CSV)
    df_baysal = df_full[
        (df_full["dataset_source"] == "baysal_2016")
        & (df_full["editing_rate_normalized"].notna())
        & (df_full["is_edited"] == 1)
    ].copy()
    logger.info("  baysal_2016 from splits_expanded.csv: %d sites", len(df_baysal))

    # Load advisor_c2t and alqassim from A3A-filtered splits
    df_a3a = pd.read_csv(SPLITS_A3A_CSV)
    df_advisor = df_a3a[
        (df_a3a["dataset_source"] == "advisor_c2t")
        & (df_a3a["editing_rate_normalized"].notna())
        & (df_a3a["is_edited"] == 1)
    ].copy()
    df_alqassim = df_a3a[
        (df_a3a["dataset_source"] == "alqassim_2021")
        & (df_a3a["editing_rate_normalized"].notna())
        & (df_a3a["is_edited"] == 1)
    ].copy()
    logger.info("  advisor_c2t from splits_expanded_a3a.csv: %d sites", len(df_advisor))
    logger.info("  alqassim_2021 from splits_expanded_a3a.csv: %d sites", len(df_alqassim))

    # Merge
    df = pd.concat([df_baysal, df_advisor, df_alqassim], ignore_index=True)
    del df_full, df_a3a, df_baysal, df_advisor, df_alqassim

    # Positives only (NEVER include negatives in rate regression)
    assert (df["is_edited"] == 1).all(), "Rate prediction must use positives only!"

    # Raw log2 rate -- no Z-scoring (within a single dataset Spearman is rank-invariant)
    df["target"] = np.log2(df["editing_rate_normalized"].values + 0.01)

    logger.info("  Total rate-annotated positive sites: %d", len(df))
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


def extract_pairing_profile_features(data, site_ids) -> np.ndarray:
    """Extract ~50-dim pairing profile features from structure cache."""
    features = []
    for sid in site_ids:
        pp = data["pairing_probs"].get(sid, np.zeros(201))
        pp_ed = data["pairing_probs_edited"].get(sid, np.zeros(201))
        acc = data["accessibilities"].get(sid, np.zeros(201))
        acc_ed = data["accessibilities_edited"].get(sid, np.zeros(201))
        mfe = data["mfes"].get(sid, 0.0)
        mfe_ed = data["mfes_edited"].get(sid, 0.0)
        ep = 100
        feat = []
        for w in [5, 11, 21]:
            s, e = max(0, ep - w // 2), min(201, ep + w // 2 + 1)
            win_pp = pp[s:e]; win_pp_ed = pp_ed[s:e]
            feat.extend([win_pp.mean(), win_pp.std(),
                         win_pp_ed.mean(), win_pp_ed.std(),
                         (win_pp_ed - win_pp).mean()])
        for w in [5, 11, 21]:
            s, e = max(0, ep - w // 2), min(201, ep + w // 2 + 1)
            win_acc = acc[s:e]; win_acc_ed = acc_ed[s:e]
            feat.extend([win_acc.mean(), win_acc.std(),
                         win_acc_ed.mean(), win_acc_ed.std(),
                         (win_acc_ed - win_acc).mean()])
        feat.extend([pp[ep], pp_ed[ep], pp_ed[ep] - pp[ep],
                     acc[ep], acc_ed[ep], acc_ed[ep] - acc[ep]])
        feat.extend([mfe, mfe_ed, mfe_ed - mfe])
        delta_pp = pp_ed - pp
        for i in range(10):
            s = i * 20; e = min(201, (i + 1) * 20)
            feat.append(delta_pp[s:e].mean())
        feat.append(delta_pp[90:111].mean())
        features.append(feat)
    return np.array(features, dtype=np.float32)


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

class StructureOnlyRegressor(nn.Module):
    """MLP on 7-dim structure delta for rate regression."""
    def __init__(self, dropout=0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(7, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
    def forward(self, batch):
        return {"rate_pred": self.mlp(batch["structure_delta"])}


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
) -> Dict:
    """Train a NN model with early stopping on val Spearman."""
    model = model.to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)

    best_spearman = -float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
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
        scheduler.step()

        val_metrics = evaluate_nn_model(model, loaders.get("val"))
        val_sp = val_metrics.get("spearman", -999)
        if not np.isnan(val_sp) and val_sp > best_spearman + 1e-5:
            best_spearman = val_sp
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate_nn_model(model, loaders.get("test"))
    return test_metrics


def train_editrna_rate(
    model: nn.Module,
    loaders: Dict[str, DataLoader],
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    patience: int = 15,
    rate_weight: float = 10.0,
) -> Dict:
    """Train EditRNA-A3A with rate_weight on MSE loss."""
    model = model.to(DEVICE)
    editrna = model.editrna

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)

    best_spearman = -float("inf")
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
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return evaluate_nn_model(model, loaders.get("test"))


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
# Fold Data Preparation Helpers
# ---------------------------------------------------------------------------

def make_token_loaders(
    train_sids, val_sids, test_sids,
    train_targets, val_targets, test_targets,
    data, batch_size=16,
) -> Dict[str, DataLoader]:
    """Create DataLoaders with token-level embeddings for a fold."""
    loaders = {}
    for name, sids, tgts in [("train", train_sids, train_targets),
                              ("val", val_sids, val_targets),
                              ("test", test_sids, test_targets)]:
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


def make_pooled_loaders(
    train_sids, val_sids, test_sids,
    train_targets, val_targets, test_targets,
    data, batch_size=64,
) -> Dict[str, DataLoader]:
    """Create DataLoaders with pooled-only embeddings for a fold."""
    loaders = {}
    for name, sids, tgts in [("train", train_sids, train_targets),
                              ("val", val_sids, val_targets),
                              ("test", test_sids, test_targets)]:
        ds = RateDataset(
            site_ids=sids, targets=tgts,
            pooled_orig=data["pooled_orig"], pooled_edited=data["pooled_edited"],
            structure_delta=data["structure_delta"],
        )
        loaders[name] = DataLoader(
            ds, batch_size=batch_size, shuffle=(name == "train"),
            num_workers=0, collate_fn=rate_collate_fn, drop_last=False)
    return loaders


def make_mm_loaders(
    train_sids, val_sids, test_sids,
    train_targets, val_targets, test_targets,
    train_hand, val_hand, test_hand,
    data, gnn_embeddings=None, batch_size=64,
) -> Dict[str, DataLoader]:
    """Create multi-modal DataLoaders (pooled + hand + optional GNN) for a fold."""
    loaders = {}
    for name, sids, tgts, hand in [("train", train_sids, train_targets, train_hand),
                                    ("val", val_sids, val_targets, val_hand),
                                    ("test", test_sids, test_targets, test_hand)]:
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


# ---------------------------------------------------------------------------
# Per-Dataset Fold Runner
# ---------------------------------------------------------------------------

def run_dataset_cv(dataset_name, df_dataset, data, gnn_embeddings, d_gnn=64):
    """Run 5-fold CV for a single dataset. Returns dict of model -> fold_results."""
    n_sites = len(df_dataset)
    logger.info("=" * 70)
    logger.info("DATASET: %s  (n=%d)", dataset_name, n_sites)
    logger.info("=" * 70)

    # Filter to sites with embeddings
    available = set(data["pooled_orig"].keys()) & set(data["pooled_edited"].keys())
    df_dataset = df_dataset[df_dataset["site_id"].isin(available)].copy().reset_index(drop=True)
    n_sites = len(df_dataset)
    logger.info("  Sites with embeddings: %d", n_sites)

    all_site_ids = df_dataset["site_id"].values
    all_targets = df_dataset["target"].values.astype(np.float32)

    # Determine which models to run
    run_neural = dataset_name in NEURAL_DATASETS
    MODEL_NAMES = ["Mean Baseline", "StructureOnly", "GB_HandFeatures", "GB_AllFeatures"]
    if run_neural:
        MODEL_NAMES.extend(["EditRNA_rate", "4Way_heavyreg"])
    model_fold_results = {name: [] for name in MODEL_NAMES}
    gb_hand_importances = []

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

    for fold_idx, (remain_idx, test_idx) in enumerate(kf.split(all_site_ids)):
        fold_seed = SEED + fold_idx
        np.random.seed(fold_seed)
        torch.manual_seed(fold_seed)

        logger.info("\n  FOLD %d / 5 (%s)", fold_idx + 1, dataset_name)

        # Inner split: 80% train, 20% val from remaining
        n_remain = len(remain_idx)
        inner_perm = np.random.RandomState(fold_seed).permutation(n_remain)
        n_val = int(n_remain * 0.2)
        val_inner_idx = inner_perm[:n_val]
        train_inner_idx = inner_perm[n_val:]

        train_idx = remain_idx[train_inner_idx]
        val_idx = remain_idx[val_inner_idx]

        train_sids = [all_site_ids[i] for i in train_idx]
        val_sids = [all_site_ids[i] for i in val_idx]
        test_sids = [all_site_ids[i] for i in test_idx]
        train_targets = all_targets[train_idx]
        val_targets = all_targets[val_idx]
        test_targets = all_targets[test_idx]

        logger.info("    train=%d, val=%d, test=%d",
                    len(train_sids), len(val_sids), len(test_sids))

        # Pre-compute hand features
        train_hand = build_hand_features(train_sids, data["sequences"],
                                          data["structure_delta"], data["loop_df"])
        val_hand = build_hand_features(val_sids, data["sequences"],
                                        data["structure_delta"], data["loop_df"])
        test_hand = build_hand_features(test_sids, data["sequences"],
                                         data["structure_delta"], data["loop_df"])
        d_hand = train_hand.shape[1]

        # ==============================================================
        # 1. Mean Baseline
        # ==============================================================
        logger.info("    [1] Mean Baseline")
        train_mean = train_targets.mean()
        y_pred_mean = np.full_like(test_targets, train_mean)
        model_fold_results["Mean Baseline"].append(
            compute_rate_metrics(test_targets, y_pred_mean))

        # ==============================================================
        # 2. StructureOnly
        # ==============================================================
        logger.info("    [2] StructureOnly")
        torch.manual_seed(fold_seed)
        pooled_loaders = make_pooled_loaders(
            train_sids, val_sids, test_sids,
            train_targets, val_targets, test_targets, data)
        struct_model = StructureOnlyRegressor(dropout=0.3)
        struct_metrics = train_nn_model(
            struct_model, pooled_loaders, "StructureOnly",
            epochs=100, lr=1e-3, weight_decay=1e-3, patience=15)
        model_fold_results["StructureOnly"].append(struct_metrics)
        del struct_model

        # ==============================================================
        # 3. GB_HandFeatures
        # ==============================================================
        logger.info("    [3] GB_HandFeatures")
        try:
            from xgboost import XGBRegressor

            X_train_h = np.nan_to_num(train_hand, nan=0.0)
            X_val_h = np.nan_to_num(val_hand, nan=0.0)
            X_test_h = np.nan_to_num(test_hand, nan=0.0)

            gb_hand = XGBRegressor(
                n_estimators=500, max_depth=5, learning_rate=0.05,
                subsample=0.8, min_child_weight=10, reg_alpha=0.1, reg_lambda=1.0,
                random_state=fold_seed, n_jobs=1,
                early_stopping_rounds=30, verbosity=0,
            )
            gb_hand.fit(X_train_h, train_targets,
                        eval_set=[(X_val_h, val_targets)], verbose=False)
            y_pred_gb_hand = gb_hand.predict(X_test_h)
            model_fold_results["GB_HandFeatures"].append(
                compute_rate_metrics(test_targets, y_pred_gb_hand))
            gb_hand_importances.append(gb_hand.feature_importances_.copy())
            del gb_hand
        except ImportError:
            logger.warning("xgboost not available, skipping GB models")
            model_fold_results["GB_HandFeatures"].append(
                {"spearman": float("nan"), "pearson": float("nan"),
                 "mse": float("nan"), "r2": float("nan")})

        # ==============================================================
        # 4. GB_AllFeatures
        # ==============================================================
        logger.info("    [4] GB_AllFeatures")
        try:
            from xgboost import XGBRegressor

            train_pairing = extract_pairing_profile_features(data, train_sids)
            val_pairing = extract_pairing_profile_features(data, val_sids)
            test_pairing = extract_pairing_profile_features(data, test_sids)

            def _emb_delta(sids):
                return np.array([
                    (data["pooled_edited"][sid] - data["pooled_orig"][sid]).numpy()
                    if sid in data["pooled_orig"] and sid in data["pooled_edited"]
                    else np.zeros(640, dtype=np.float32)
                    for sid in sids
                ], dtype=np.float32)

            train_emb_delta = _emb_delta(train_sids)
            val_emb_delta = _emb_delta(val_sids)
            test_emb_delta = _emb_delta(test_sids)

            X_train_all = np.nan_to_num(np.concatenate(
                [train_hand, train_pairing, train_emb_delta], axis=1), nan=0.0)
            X_val_all = np.nan_to_num(np.concatenate(
                [val_hand, val_pairing, val_emb_delta], axis=1), nan=0.0)
            X_test_all = np.nan_to_num(np.concatenate(
                [test_hand, test_pairing, test_emb_delta], axis=1), nan=0.0)

            gb_all = XGBRegressor(
                n_estimators=500, max_depth=5, learning_rate=0.05,
                subsample=0.8, min_child_weight=10, reg_alpha=0.1, reg_lambda=1.0,
                random_state=fold_seed, n_jobs=1,
                early_stopping_rounds=30, verbosity=0,
            )
            gb_all.fit(X_train_all, train_targets,
                       eval_set=[(X_val_all, val_targets)], verbose=False)
            y_pred_gb_all = gb_all.predict(X_test_all)
            model_fold_results["GB_AllFeatures"].append(
                compute_rate_metrics(test_targets, y_pred_gb_all))
            del gb_all, train_pairing, val_pairing, test_pairing
            del train_emb_delta, val_emb_delta, test_emb_delta
        except ImportError:
            model_fold_results["GB_AllFeatures"].append(
                {"spearman": float("nan"), "pearson": float("nan"),
                 "mse": float("nan"), "r2": float("nan")})

        del pooled_loaders

        # ==============================================================
        # 5. EditRNA_rate (neural -- skip for small datasets)
        # ==============================================================
        if run_neural:
            logger.info("    [5] EditRNA_rate")
            token_loaders = make_token_loaders(
                train_sids, val_sids, test_sids,
                train_targets, val_targets, test_targets, data, batch_size=16)
            try:
                torch.manual_seed(fold_seed)
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

                editrna_metrics = train_editrna_rate(
                    editrna_wrapper, token_loaders,
                    epochs=100, lr=1e-3, weight_decay=1e-2,
                    patience=15, rate_weight=10.0)
                model_fold_results["EditRNA_rate"].append(editrna_metrics)
                del editrna_wrapper, editrna, cached_encoder
            except Exception as e:
                logger.error("EditRNA_rate FAILED fold %d: %s", fold_idx + 1, e)
                model_fold_results["EditRNA_rate"].append(
                    {"spearman": float("nan"), "pearson": float("nan"),
                     "mse": float("nan"), "r2": float("nan")})
            del token_loaders

            # ==============================================================
            # 6. 4Way_heavyreg (needs pooled + hand + GNN embeddings)
            # ==============================================================
            logger.info("    [6] 4Way_heavyreg")
            if gnn_embeddings is not None:
                torch.manual_seed(fold_seed)
                mm4_loaders = make_mm_loaders(
                    train_sids, val_sids, test_sids,
                    train_targets, val_targets, test_targets,
                    train_hand, val_hand, test_hand,
                    data, gnn_embeddings=gnn_embeddings)
                four_model = FourWayGatedFusion(
                    d_primary=640, d_hand=d_hand, d_gnn=d_gnn,
                    d_proj=128, dropout=0.5)
                four_metrics = train_nn_model(
                    four_model, mm4_loaders, "4Way_heavyreg",
                    epochs=200, lr=5e-4, weight_decay=1e-2, patience=30)
                model_fold_results["4Way_heavyreg"].append(four_metrics)
                del four_model, mm4_loaders
            else:
                model_fold_results["4Way_heavyreg"].append(
                    {"spearman": float("nan"), "pearson": float("nan"),
                     "mse": float("nan"), "r2": float("nan")})

        # Log fold summary
        logger.info("\n    Fold %d summary (%s):", fold_idx + 1, dataset_name)
        for name in MODEL_NAMES:
            fr = model_fold_results[name][-1]
            sp = fr.get("spearman", float("nan"))
            pe = fr.get("pearson", float("nan"))
            logger.info("      %-25s  sp=%.4f  pe=%.4f",
                        name,
                        sp if not np.isnan(sp) else 0.0,
                        pe if not np.isnan(pe) else 0.0)

        gc.collect()

    # Aggregate results for this dataset
    dataset_result = {
        "n_sites": n_sites,
        "models": {},
    }
    for name in MODEL_NAMES:
        fold_results = model_fold_results[name]
        sps = [fr["spearman"] for fr in fold_results
               if not np.isnan(fr.get("spearman", float("nan")))]
        pes = [fr["pearson"] for fr in fold_results
               if not np.isnan(fr.get("pearson", float("nan")))]
        mses = [fr["mse"] for fr in fold_results
                if not np.isnan(fr.get("mse", float("nan")))]
        r2s = [fr["r2"] for fr in fold_results
               if not np.isnan(fr.get("r2", float("nan")))]

        dataset_result["models"][name] = {
            "fold_results": fold_results,
            "mean_spearman": float(np.mean(sps)) if sps else float("nan"),
            "std_spearman": float(np.std(sps)) if len(sps) > 1 else 0.0,
            "mean_pearson": float(np.mean(pes)) if pes else float("nan"),
            "std_pearson": float(np.std(pes)) if len(pes) > 1 else 0.0,
            "mean_mse": float(np.mean(mses)) if mses else float("nan"),
            "mean_r2": float(np.mean(r2s)) if r2s else float("nan"),
        }

    # Save per-dataset feature importance for GB_HandFeatures
    if gb_hand_importances:
        hand_names = []
        for ctx in ["UC", "CC", "AC", "GC"]:
            hand_names.append(f"5p_{ctx}")
        for ctx in ["CA", "CG", "CU", "CC"]:
            hand_names.append(f"3p_{ctx}")
        for label in ["m2", "m1"]:
            for b in ["A", "C", "G", "U"]:
                hand_names.append(f"trinuc_up_{label}_{b}")
        for label in ["p1", "p2"]:
            for b in ["A", "C", "G", "U"]:
                hand_names.append(f"trinuc_down_{label}_{b}")
        for n in ["delta_pairing_center", "delta_accessibility_center",
                   "delta_entropy_center", "delta_mfe",
                   "mean_delta_pairing_window", "mean_delta_accessibility_window",
                   "std_delta_pairing_window"]:
            hand_names.append(n)
        for n in ["is_unpaired", "loop_size", "dist_to_junction", "dist_to_apex",
                   "relative_loop_position", "left_stem_length", "right_stem_length",
                   "max_adjacent_stem_length", "local_unpaired_fraction"]:
            hand_names.append(n)

        def _categorize_feature(name):
            if name.startswith("5p_") or name.startswith("3p_") or "trinuc" in name:
                return "Motif"
            if "delta" in name or "mfe" in name.lower():
                return "Structure Delta"
            if any(k in name for k in ["unpaired", "loop", "stem", "junction", "apex"]):
                return "Loop Geometry"
            return "Other"

        imp_mean = np.mean(gb_hand_importances, axis=0)
        imp_std = np.std(gb_hand_importances, axis=0)
        rows = []
        for j, name in enumerate(hand_names):
            rows.append({
                "feature_name": name,
                "mean_importance": float(imp_mean[j]),
                "std_importance": float(imp_std[j]),
                "category": _categorize_feature(name),
            })
        csv_path = OUTPUT_DIR / f"feature_importance_{dataset_name}_gb_hand.csv"
        pd.DataFrame(rows).sort_values("mean_importance", ascending=False).to_csv(
            csv_path, index=False)
        logger.info("Saved %s feature importance to %s", dataset_name, csv_path)

    return dataset_result


# ---------------------------------------------------------------------------
# Cross-Dataset Generalization
# ---------------------------------------------------------------------------

def run_cross_dataset(df, data, gnn_embeddings, d_gnn=64):
    """Train on each dataset, test on every other dataset.

    Models: GB_HandFeatures, EditRNA_rate (EditRNA only when train set >= 500).
    Returns dict of {train_ds: {test_ds: {model: metrics}}}.
    """
    logger.info("\n" + "=" * 70)
    logger.info("CROSS-DATASET GENERALIZATION")
    logger.info("=" * 70)

    available = set(data["pooled_orig"].keys()) & set(data["pooled_edited"].keys())
    cross_results = {}

    for train_ds in DATASETS:
        df_train_full = df[
            (df["dataset_source"] == train_ds) & df["site_id"].isin(available)
        ].copy().reset_index(drop=True)
        if len(df_train_full) < 10:
            continue

        cross_results[train_ds] = {}

        # 80/20 inner split for val
        n = len(df_train_full)
        perm = np.random.RandomState(SEED).permutation(n)
        n_val = int(n * 0.2)
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]

        train_sids = df_train_full["site_id"].values[train_idx].tolist()
        val_sids = df_train_full["site_id"].values[val_idx].tolist()
        train_targets = df_train_full["target"].values[train_idx].astype(np.float32)
        val_targets = df_train_full["target"].values[val_idx].astype(np.float32)

        # Hand features for training
        train_hand = build_hand_features(
            train_sids, data["sequences"], data["structure_delta"], data["loop_df"])
        val_hand = build_hand_features(
            val_sids, data["sequences"], data["structure_delta"], data["loop_df"])

        # Train GB_HandFeatures
        logger.info("\nTrain: %s (n=%d) — GB_HandFeatures", train_ds, len(df_train_full))
        from xgboost import XGBRegressor
        gb = XGBRegressor(
            n_estimators=500, max_depth=5, learning_rate=0.05,
            subsample=0.8, min_child_weight=10, reg_alpha=0.1, reg_lambda=1.0,
            random_state=SEED, n_jobs=1, early_stopping_rounds=30, verbosity=0,
        )
        gb.fit(np.nan_to_num(train_hand, nan=0.0), train_targets,
               eval_set=[(np.nan_to_num(val_hand, nan=0.0), val_targets)], verbose=False)

        # Train EditRNA_rate if large enough
        editrna_wrapper = None
        if len(df_train_full) >= 500:
            logger.info("  Training EditRNA_rate on %s...", train_ds)
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
                    edit_n_heads=4, pooled_only=False, use_structure_delta=True,
                    head_dropout=0.3, fusion_dropout=0.3,
                    weight_decay=1e-2, learning_rate=1e-3,
                )
                editrna = EditRNA_A3A(config=config, primary_encoder=cached_encoder)
                editrna_wrapper = EditRNARateWrapper(editrna)

                token_loaders = make_token_loaders(
                    train_sids, val_sids, val_sids,  # val as "test" for training
                    train_targets, val_targets, val_targets, data, batch_size=16)
                train_editrna_rate(
                    editrna_wrapper, token_loaders,
                    epochs=100, lr=1e-3, weight_decay=1e-2,
                    patience=15, rate_weight=10.0)
                del token_loaders
            except Exception as e:
                logger.error("EditRNA_rate training failed on %s: %s", train_ds, e)
                editrna_wrapper = None

        # Test on every other dataset
        for test_ds in DATASETS:
            if test_ds == train_ds:
                continue

            df_test = df[
                (df["dataset_source"] == test_ds) & df["site_id"].isin(available)
            ].copy().reset_index(drop=True)
            if len(df_test) < 3:
                continue

            test_sids = df_test["site_id"].values.tolist()
            test_targets = df_test["target"].values.astype(np.float32)
            test_hand = build_hand_features(
                test_sids, data["sequences"], data["structure_delta"], data["loop_df"])

            model_results = {}

            # GB_HandFeatures
            y_pred = gb.predict(np.nan_to_num(test_hand, nan=0.0))
            model_results["GB_HandFeatures"] = compute_rate_metrics(test_targets, y_pred)
            logger.info("  Test: %s (n=%d) — GB sp=%.4f", test_ds, len(df_test),
                        model_results["GB_HandFeatures"]["spearman"])

            # EditRNA_rate
            if editrna_wrapper is not None:
                test_loader = make_token_loaders(
                    test_sids, test_sids, test_sids,
                    test_targets, test_targets, test_targets, data, batch_size=16
                )["test"]
                er_metrics = evaluate_nn_model(editrna_wrapper, test_loader)
                model_results["EditRNA_rate"] = er_metrics
                logger.info("  Test: %s (n=%d) — EditRNA sp=%.4f", test_ds, len(df_test),
                            er_metrics["spearman"])
                del test_loader

            cross_results[train_ds][test_ds] = model_results

        # Cleanup
        del gb
        if editrna_wrapper is not None:
            del editrna_wrapper, editrna, cached_encoder
        gc.collect()

    return cross_results


# ---------------------------------------------------------------------------
# Main Experiment
# ---------------------------------------------------------------------------

def run_experiment():
    t_total = time.time()
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("WITHIN-DATASET 5-FOLD CV RATE PREDICTION (PER DATASET)")
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # Load ALL data once
    # ------------------------------------------------------------------
    data = load_all_data()
    df = data["df"]

    # ------------------------------------------------------------------
    # Pre-compute GNN embeddings (random init, once for all datasets/folds)
    # ------------------------------------------------------------------
    gnn_embeddings = None
    d_gnn = 64
    try:
        from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
        logger.info("Pre-computing random GNN embeddings for 4Way_heavyreg...")
        torch.manual_seed(SEED)
        gnn = SmallStructureGNN(12, 64, 64, 3, 4, 0.3)
        all_sids_for_gnn = list(
            set(data["pooled_orig"].keys()) & set(data["pooled_edited"].keys()))
        gnn_embeddings = precompute_gnn_embeddings(
            gnn, all_sids_for_gnn, data["sequences"], data["pairing_probs"])
        logger.info("  GNN embeddings: %d sites, %d-dim",
                    len(gnn_embeddings), d_gnn)
        del gnn
    except ImportError:
        logger.warning("torch_geometric not available, skipping 4Way_heavyreg model")

    # ------------------------------------------------------------------
    # Run per-dataset CV
    # ------------------------------------------------------------------
    results = {
        "experiment": "rate_per_dataset",
        "target_transform": "log2(editing_rate_normalized + 0.01)",
        "datasets": {},
    }

    for ds_name in DATASETS:
        df_ds = df[df["dataset_source"] == ds_name].copy()
        if len(df_ds) == 0:
            logger.warning("No sites found for dataset %s, skipping", ds_name)
            continue

        ds_result = run_dataset_cv(ds_name, df_ds, data, gnn_embeddings, d_gnn=d_gnn)
        results["datasets"][ds_name] = ds_result

    # ------------------------------------------------------------------
    # Cross-Dataset Generalization
    # ------------------------------------------------------------------
    cross_results = run_cross_dataset(df, data, gnn_embeddings, d_gnn=d_gnn)
    results["cross_dataset"] = cross_results

    # Save cross-dataset results separately too
    cross_out_path = OUTPUT_DIR / "rate_cross_dataset_results.json"
    with open(cross_out_path, "w") as f:
        json.dump({"cross_dataset": cross_results}, f, indent=2, default=_serialize)
    logger.info("Cross-dataset results saved to %s", cross_out_path)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    total_time = time.time() - t_total
    results["total_time_seconds"] = round(total_time, 1)

    out_path = OUTPUT_DIR / "rate_per_dataset_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=_serialize)
    logger.info("\nResults saved to %s", out_path)

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 110)
    print("WITHIN-DATASET 5-FOLD CV RATE PREDICTION RESULTS")
    print(f"Target: {results['target_transform']}")
    print("=" * 110)

    for ds_name in DATASETS:
        ds = results["datasets"].get(ds_name)
        if ds is None:
            continue
        print(f"\n--- {ds_name} (n={ds['n_sites']}) ---")
        print(f"  {'Model':<25} {'Spearman':>18} {'Pearson':>18} {'MSE':>10} {'R2':>10}")
        print("  " + "-" * 85)

        sorted_models = sorted(
            ds["models"].items(),
            key=lambda x: x[1]["mean_spearman"] if not np.isnan(x[1]["mean_spearman"]) else -999,
            reverse=True)

        for name, m in sorted_models:
            sp = m["mean_spearman"]
            sp_std = m["std_spearman"]
            pe = m["mean_pearson"]
            pe_std = m["std_pearson"]
            mse_val = m["mean_mse"]
            r2_val = m["mean_r2"]

            sp_str = f"{sp:.4f}+/-{sp_std:.4f}" if not np.isnan(sp) else "N/A"
            pe_str = f"{pe:.4f}+/-{pe_std:.4f}" if not np.isnan(pe) else "N/A"
            mse_str = f"{mse_val:.4f}" if not np.isnan(mse_val) else "N/A"
            r2_str = f"{r2_val:.4f}" if not np.isnan(r2_val) else "N/A"

            print(f"  {name:<25} {sp_str:>18} {pe_str:>18} {mse_str:>10} {r2_str:>10}")

    # Cross-dataset summary
    if cross_results:
        print("\n" + "=" * 110)
        print("CROSS-DATASET GENERALIZATION")
        print("=" * 110)
        for train_ds, test_dict in cross_results.items():
            for test_ds, model_dict in test_dict.items():
                for model_name, metrics in model_dict.items():
                    sp = metrics.get("spearman", float("nan"))
                    pe = metrics.get("pearson", float("nan"))
                    sp_str = f"{sp:.4f}" if not np.isnan(sp) else "N/A"
                    pe_str = f"{pe:.4f}" if not np.isnan(pe) else "N/A"
                    print(f"  Train: {train_ds:<15} -> Test: {test_ds:<15}  "
                          f"{model_name:<20}  sp={sp_str}  pe={pe_str}")

    print("\n" + "=" * 110)
    print(f"Total time: {total_time:.1f} seconds ({total_time / 60:.1f} minutes)")
    print("=" * 110)

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
