#!/usr/bin/env python
"""5-fold cross-validation for 13 classification models on A3A-filtered data.

Runs KFold(n_splits=5, shuffle=True, random_state=42) over all sites in
splits_expanded_a3a.csv.  For each fold the remaining 80% is split 80/20
into train/val (inner split seed=42+fold_idx).

Models (ordered by complexity):
   1. Majority Class        - predicts all positive
   2. StructureOnly         - MLP on 7-dim structure delta
   3. GB_HandFeatures       - XGBClassifier on 40-dim hand features
   4. GB_AllFeatures        - XGBClassifier on all features (~680 dim)
   5. PooledMLP             - MLP on 640-dim pooled orig
   6. SubtractionMLP        - MLP on 640-dim pooled delta
   7. ConcatMLP             - MLP on 1280-dim concat
   8. CrossAttention        - Token cross-attention
   9. DiffAttention          - Token diff + TransformerEncoder
  10. EditRNA-A3A            - Full EditRNA
  11. EditRNA+Features       - EditRNA + 33-dim hand features
  12. SubtractionMLP+Features - SubtractionMLP + 33-dim hand
  13. DiffAttention+Features  - DiffAttention + 33-dim hand

Usage:
    python experiments/apobec3a/exp_classification_a3a_5fold.py
"""

import gc
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, accuracy_score,
    precision_recall_curve,
)
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
SPLITS_CSV = PROJECT_ROOT / "data" / "processed" / "splits_expanded_a3a.csv"
EMB_DIR = PROJECT_ROOT / "data" / "processed" / "embeddings"
STRUCTURE_CACHE = EMB_DIR / "vienna_structure_cache.npz"
SEQUENCES_JSON = PROJECT_ROOT / "data" / "processed" / "site_sequences.json"
LOOP_POS_CSV = (
    PROJECT_ROOT / "experiments" / "apobec" / "outputs"
    / "loop_position" / "loop_position_per_site.csv"
)
OUTPUT_DIR = (
    PROJECT_ROOT / "experiments" / "apobec" / "outputs" / "classification_a3a_5fold"
)

SEED = 42
DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Focal Loss for binary classification (handles class imbalance)."""
    def __init__(self, gamma=2.0, alpha=0.75):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal = alpha_t * (1 - pt) ** self.gamma * bce
        return focal.mean()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_binary_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    """Compute binary classification metrics."""
    if len(np.unique(y_true)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan"), "f1": float("nan"),
                "precision": float("nan"), "recall": float("nan")}
    auroc = float(roc_auc_score(y_true, y_score))
    auprc = float(average_precision_score(y_true, y_score))
    # Find optimal threshold
    prec, rec, thresholds = precision_recall_curve(y_true, y_score)
    f1_scores = 2 * prec * rec / (prec + rec + 1e-10)
    best_idx = np.argmax(f1_scores)
    threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
    y_pred = (np.array(y_score) >= threshold).astype(int)
    return {
        "auroc": auroc, "auprc": auprc,
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


# ---------------------------------------------------------------------------
# Feature Extraction (shared with rate experiment)
# ---------------------------------------------------------------------------

def extract_motif_features(sequences: Dict[str, str], site_ids: List[str]) -> np.ndarray:
    """24-dim motif features."""
    features = []
    for sid in site_ids:
        seq = sequences.get(str(sid), "N" * 201).upper().replace("T", "U")
        ep = 100
        up = seq[ep - 1] if ep > 0 else "N"
        down = seq[ep + 1] if ep < len(seq) - 1 else "N"
        feat_5p = [1 if up + "C" == m else 0 for m in ["UC", "CC", "AC", "GC"]]
        feat_3p = [1 if "C" + down == m else 0 for m in ["CA", "CG", "CU", "CC"]]
        trinuc_up = [0] * 8
        bases = ["A", "C", "G", "U"]
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
        if str(sid) in loop_df.index:
            features.append(loop_df.loc[str(sid), cols].values.astype(np.float32))
        else:
            features.append(np.zeros(len(cols), dtype=np.float32))
    return np.array(features, dtype=np.float32)


def extract_structure_delta(structure_delta: Dict, site_ids: List[str]) -> np.ndarray:
    """7-dim structure delta features."""
    return np.array([structure_delta.get(str(sid), np.zeros(7)) for sid in site_ids],
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
        sid = str(sid)
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
# Data Loading
# ---------------------------------------------------------------------------

def load_all_data():
    """Load splits, embeddings, structures, sequences, loop features."""
    logger.info("Loading data...")

    # Splits — A3A only, no Baysal
    df = pd.read_csv(SPLITS_CSV)
    assert "baysal" not in df["dataset_source"].str.lower().unique(), \
        "Classification must NOT include Baysal sites!"
    logger.info("  Total A3A sites: %d (pos=%d, neg=%d)",
                len(df), (df["label"] == 1).sum(), (df["label"] == 0).sum())
    if "dataset_source" in df.columns:
        for src, cnt in df["dataset_source"].value_counts().items():
            logger.info("    %s: %d sites", src, cnt)

    needed_sids = set(df["site_id"].astype(str).tolist())

    # Pooled embeddings
    _po = torch.load(EMB_DIR / "rnafm_pooled.pt", weights_only=False)
    pooled_orig = {k: v for k, v in _po.items() if str(k) in needed_sids}
    del _po; gc.collect()
    _pe = torch.load(EMB_DIR / "rnafm_pooled_edited.pt", weights_only=False)
    pooled_edited = {k: v for k, v in _pe.items() if str(k) in needed_sids}
    del _pe; gc.collect()
    logger.info("  %d pooled embeddings loaded", len(pooled_orig))

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
    logger.info("  %d structure caches loaded", len(structure_delta))

    # Sequences
    with open(SEQUENCES_JSON) as f:
        _seq = json.load(f)
    sequences = {k: v for k, v in _seq.items() if str(k) in needed_sids}
    del _seq
    logger.info("  %d sequences loaded", len(sequences))

    # Loop features
    loop_df = pd.DataFrame()
    if LOOP_POS_CSV.exists():
        loop_df = pd.read_csv(LOOP_POS_CSV)
        loop_df["site_id"] = loop_df["site_id"].astype(str)
        loop_df = loop_df.set_index("site_id")
        loop_df = loop_df[loop_df.index.isin(needed_sids)]
        logger.info("  %d loop features loaded", len(loop_df))

    return {
        "df": df,
        "pooled_orig": pooled_orig, "pooled_edited": pooled_edited,
        "structure_delta": structure_delta,
        "pairing_probs": pairing_probs, "pairing_probs_edited": pairing_probs_edited,
        "accessibilities": accessibilities, "accessibilities_edited": accessibilities_edited,
        "mfes": mfes, "mfes_edited": mfes_edited,
        "sequences": sequences, "loop_df": loop_df,
    }


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class BinaryPooledDataset(Dataset):
    """Dataset for binary classification with pooled embeddings."""
    def __init__(self, site_ids, labels, pooled_orig, pooled_edited, mode="subtraction",
                 hand_features=None):
        self.site_ids = site_ids
        self.labels = labels
        self.pooled_orig = pooled_orig
        self.pooled_edited = pooled_edited
        self.mode = mode
        self.hand_features = hand_features

    def __len__(self):
        return len(self.site_ids)

    def __getitem__(self, idx):
        sid = self.site_ids[idx]
        orig = self.pooled_orig[sid]
        edited = self.pooled_edited[sid]
        if self.mode == "subtraction":
            x = edited - orig
        elif self.mode == "concat":
            x = torch.cat([orig, edited], dim=-1)
        elif self.mode == "orig_only":
            x = orig
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        if self.hand_features is not None:
            hand = torch.tensor(self.hand_features[idx], dtype=torch.float32)
            x = torch.cat([x, hand], dim=-1)
        return x, torch.tensor(self.labels[idx], dtype=torch.float32)


class StructureBinaryDataset(Dataset):
    def __init__(self, site_ids, labels, structure_delta):
        self.site_ids = site_ids
        self.labels = labels
        self.structure_delta = structure_delta

    def __len__(self):
        return len(self.site_ids)

    def __getitem__(self, idx):
        sid = self.site_ids[idx]
        struct = self.structure_delta.get(str(sid))
        x = torch.tensor(struct, dtype=torch.float32) if struct is not None else torch.zeros(7)
        return x, torch.tensor(self.labels[idx], dtype=torch.float32)


class TokenBinaryDataset(Dataset):
    def __init__(self, site_ids, labels, tokens_orig, tokens_edited,
                 hand_features=None):
        self.site_ids = site_ids
        self.labels = labels
        self.tokens_orig = tokens_orig
        self.tokens_edited = tokens_edited
        self.hand_features = hand_features

    def __len__(self):
        return len(self.site_ids)

    def __getitem__(self, idx):
        sid = self.site_ids[idx]
        item = {
            "tokens_orig": self.tokens_orig[sid],
            "tokens_edited": self.tokens_edited[sid],
        }
        if self.hand_features is not None:
            item["hand_features"] = torch.tensor(self.hand_features[idx], dtype=torch.float32)
        return item, torch.tensor(self.labels[idx], dtype=torch.float32)


def token_collate_fn(batch):
    items, targets = zip(*batch)
    max_len = max(item["tokens_orig"].shape[0] for item in items)
    d = items[0]["tokens_orig"].shape[1]
    t_orig = torch.zeros(len(items), max_len, d)
    t_edit = torch.zeros(len(items), max_len, d)
    for i, item in enumerate(items):
        L = item["tokens_orig"].shape[0]
        t_orig[i, :L] = item["tokens_orig"]
        t_edit[i, :L] = item["tokens_edited"]
    result = {"tokens_orig": t_orig, "tokens_edited": t_edit}
    if "hand_features" in items[0]:
        result["hand_features"] = torch.stack([it["hand_features"] for it in items])
    return result, torch.stack(targets)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class BinaryMLP(nn.Module):
    def __init__(self, d_input, hidden=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_input, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class DiffAttentionBinary(nn.Module):
    def __init__(self, d_model=640, n_heads=8, n_layers=2, d_hidden=256, dropout=0.3,
                 d_hand=0):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.d_hand = d_hand
        self.mlp = nn.Sequential(
            nn.Linear(d_model + d_hand, d_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, batch):
        diff = batch["tokens_edited"] - batch["tokens_orig"]
        encoded = self.transformer(diff)
        pooled = encoded.mean(dim=1)
        if self.d_hand > 0 and "hand_features" in batch:
            pooled = torch.cat([pooled, batch["hand_features"]], dim=-1)
        return self.mlp(pooled).squeeze(-1)


class CrossAttentionBinary(nn.Module):
    def __init__(self, d_model=640, n_heads=8, d_hidden=256, dropout=0.3):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, batch):
        attended, _ = self.cross_attn(
            query=batch["tokens_orig"], key=batch["tokens_edited"],
            value=batch["tokens_edited"])
        x = self.norm(batch["tokens_orig"] + attended)
        return self.mlp(x.mean(dim=1)).squeeze(-1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_eval_binary(model, train_loader, val_loader, test_loader,
                      is_token_model=False, epochs=50, lr=1e-3,
                      weight_decay=1e-4, patience=10, seed=42):
    """Train binary classifier with FocalLoss and early stopping on val AUROC."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)
    focal = FocalLoss(gamma=2.0, alpha=0.75)

    best_val_auroc = 0.0
    patience_counter = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        for batch_data, batch_target in train_loader:
            optimizer.zero_grad()
            if is_token_model:
                pred = model(batch_data)
            else:
                pred = model(batch_data)
            loss = focal(pred, batch_target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for bd, bt in val_loader:
                if is_token_model:
                    val_preds.append(torch.sigmoid(model(bd)).numpy())
                else:
                    val_preds.append(torch.sigmoid(model(bd)).numpy())
                val_targets.append(bt.numpy())
        vp = np.concatenate(val_preds)
        vt = np.concatenate(val_targets)
        vp = np.nan_to_num(vp, nan=0.5)
        val_auroc = roc_auc_score(vt, vp) if len(np.unique(vt)) > 1 else 0.0

        if val_auroc > best_val_auroc + 1e-4:
            best_val_auroc = val_auroc
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break

    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for bd, bt in test_loader:
            if is_token_model:
                all_preds.append(torch.sigmoid(model(bd)).numpy())
            else:
                all_preds.append(torch.sigmoid(model(bd)).numpy())
            all_targets.append(bt.numpy())
    yp = np.concatenate(all_preds)
    yp = np.nan_to_num(yp, nan=0.5)
    yt = np.concatenate(all_targets)
    return compute_binary_metrics(yt, yp)


# ---------------------------------------------------------------------------
# Main Experiment
# ---------------------------------------------------------------------------

def run_experiment():
    t_total = time.time()
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("5-FOLD CV CLASSIFICATION (A3A-FILTERED)")
    logger.info("=" * 70)

    # Load all data
    data = load_all_data()
    df = data["df"]

    # Filter to sites with embeddings
    available = set(data["pooled_orig"].keys()) & set(data["pooled_edited"].keys())
    df = df[df["site_id"].isin(available)].copy().reset_index(drop=True)
    logger.info("Sites with embeddings: %d", len(df))

    all_site_ids = df["site_id"].values
    all_labels = df["label"].values.astype(np.float32)
    n_total = len(df)
    n_pos = int((all_labels == 1).sum())
    n_neg = int((all_labels == 0).sum())
    logger.info("Total: %d (pos=%d, neg=%d, ratio=1:%.1f)",
                n_total, n_pos, n_neg, n_neg / max(n_pos, 1))

    # ------------------------------------------------------------------
    # 5-fold CV
    # ------------------------------------------------------------------
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

    MODEL_NAMES = [
        "Majority Class",
        "StructureOnly",
        "GB_HandFeatures",
        "GB_AllFeatures",
        "PooledMLP",
        "SubtractionMLP",
        "ConcatMLP",
        "CrossAttention",
        "DiffAttention",
        "EditRNA-A3A",
        "EditRNA+Features",
        "SubtractionMLP+Features",
        "DiffAttention+Features",
    ]
    model_fold_results = {name: [] for name in MODEL_NAMES}

    # Feature importance accumulators
    gb_hand_importances = []
    gb_all_importances = []

    for fold_idx, (remain_idx, test_idx) in enumerate(kf.split(all_site_ids)):
        fold_seed = SEED + fold_idx
        np.random.seed(fold_seed)
        torch.manual_seed(fold_seed)

        logger.info("\n" + "=" * 70)
        logger.info("FOLD %d / 5", fold_idx + 1)
        logger.info("=" * 70)

        # Inner split: 80% train, 20% val
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
        train_labels = all_labels[train_idx]
        val_labels = all_labels[val_idx]
        test_labels = all_labels[test_idx]

        logger.info("  train=%d (pos=%d), val=%d (pos=%d), test=%d (pos=%d)",
                    len(train_sids), int(train_labels.sum()),
                    len(val_sids), int(val_labels.sum()),
                    len(test_sids), int(test_labels.sum()))

        # Pre-compute hand features (motif 24 + struct 7 + loop 9 = 40)
        train_hand = build_hand_features(train_sids, data["sequences"],
                                          data["structure_delta"], data["loop_df"])
        val_hand = build_hand_features(val_sids, data["sequences"],
                                        data["structure_delta"], data["loop_df"])
        test_hand = build_hand_features(test_sids, data["sequences"],
                                         data["structure_delta"], data["loop_df"])

        # Feature-augmented hand features (motif 24 + loop 9 = 33, no struct delta)
        train_hand_aug = np.nan_to_num(np.concatenate([
            extract_motif_features(data["sequences"], train_sids),
            extract_loop_features(data["loop_df"], train_sids)
        ], axis=1), nan=0.0)
        val_hand_aug = np.nan_to_num(np.concatenate([
            extract_motif_features(data["sequences"], val_sids),
            extract_loop_features(data["loop_df"], val_sids)
        ], axis=1), nan=0.0)
        test_hand_aug = np.nan_to_num(np.concatenate([
            extract_motif_features(data["sequences"], test_sids),
            extract_loop_features(data["loop_df"], test_sids)
        ], axis=1), nan=0.0)
        d_hand_aug = train_hand_aug.shape[1]

        # ==============================================================
        # 1. Majority Class (predict all positive)
        # ==============================================================
        logger.info("  [1/13] Majority Class")
        y_majority = np.ones_like(test_labels)
        model_fold_results["Majority Class"].append(
            compute_binary_metrics(test_labels, y_majority))

        # ==============================================================
        # 2. StructureOnly
        # ==============================================================
        logger.info("  [2/13] StructureOnly")
        torch.manual_seed(fold_seed)
        tr_s = DataLoader(StructureBinaryDataset(train_sids, train_labels, data["structure_delta"]),
                          batch_size=64, shuffle=True, num_workers=0)
        va_s = DataLoader(StructureBinaryDataset(val_sids, val_labels, data["structure_delta"]),
                          batch_size=64, shuffle=False, num_workers=0)
        te_s = DataLoader(StructureBinaryDataset(test_sids, test_labels, data["structure_delta"]),
                          batch_size=64, shuffle=False, num_workers=0)
        model = BinaryMLP(d_input=7, hidden=64, dropout=0.3)
        metrics = train_eval_binary(model, tr_s, va_s, te_s, epochs=50, lr=5e-3, seed=fold_seed)
        model_fold_results["StructureOnly"].append(metrics)
        del model

        # ==============================================================
        # 3. GB_HandFeatures
        # ==============================================================
        logger.info("  [3/13] GB_HandFeatures")
        try:
            from xgboost import XGBClassifier

            X_train_h = np.nan_to_num(train_hand, nan=0.0)
            X_val_h = np.nan_to_num(val_hand, nan=0.0)
            X_test_h = np.nan_to_num(test_hand, nan=0.0)

            gb_hand = XGBClassifier(
                n_estimators=500, max_depth=5, learning_rate=0.05,
                subsample=0.8, min_child_weight=10, scale_pos_weight=n_neg / max(n_pos, 1),
                random_state=fold_seed, n_jobs=1,
                early_stopping_rounds=30, verbosity=0,
                eval_metric="logloss",
            )
            gb_hand.fit(X_train_h, train_labels,
                        eval_set=[(X_val_h, val_labels)], verbose=False)
            y_score = gb_hand.predict_proba(X_test_h)[:, 1]
            model_fold_results["GB_HandFeatures"].append(
                compute_binary_metrics(test_labels, y_score))
            gb_hand_importances.append(gb_hand.feature_importances_.copy())
            del gb_hand
        except ImportError:
            logger.warning("xgboost not available")
            model_fold_results["GB_HandFeatures"].append(
                {"auroc": float("nan"), "auprc": float("nan"), "f1": float("nan"),
                 "precision": float("nan"), "recall": float("nan")})

        # ==============================================================
        # 4. GB_AllFeatures
        # ==============================================================
        logger.info("  [4/13] GB_AllFeatures")
        try:
            from xgboost import XGBClassifier

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

            train_emb = _emb_delta(train_sids)
            val_emb = _emb_delta(val_sids)
            test_emb = _emb_delta(test_sids)

            X_train_all = np.nan_to_num(np.concatenate(
                [train_hand, train_pairing, train_emb], axis=1), nan=0.0)
            X_val_all = np.nan_to_num(np.concatenate(
                [val_hand, val_pairing, val_emb], axis=1), nan=0.0)
            X_test_all = np.nan_to_num(np.concatenate(
                [test_hand, test_pairing, test_emb], axis=1), nan=0.0)

            gb_all = XGBClassifier(
                n_estimators=500, max_depth=5, learning_rate=0.05,
                subsample=0.8, min_child_weight=10, scale_pos_weight=n_neg / max(n_pos, 1),
                random_state=fold_seed, n_jobs=1,
                early_stopping_rounds=30, verbosity=0,
                eval_metric="logloss",
            )
            gb_all.fit(X_train_all, train_labels,
                       eval_set=[(X_val_all, val_labels)], verbose=False)
            y_score = gb_all.predict_proba(X_test_all)[:, 1]
            model_fold_results["GB_AllFeatures"].append(
                compute_binary_metrics(test_labels, y_score))
            gb_all_importances.append(gb_all.feature_importances_.copy())
            del gb_all, train_pairing, val_pairing, test_pairing
            del train_emb, val_emb, test_emb
        except ImportError:
            model_fold_results["GB_AllFeatures"].append(
                {"auroc": float("nan"), "auprc": float("nan"), "f1": float("nan"),
                 "precision": float("nan"), "recall": float("nan")})

        # ==============================================================
        # 5. PooledMLP
        # ==============================================================
        logger.info("  [5/13] PooledMLP")
        torch.manual_seed(fold_seed)

        def _make_pooled_loaders(mode, hand_feat=None, bs=64):
            loaders = {}
            for name, sids, labs, hf in [
                ("train", train_sids, train_labels, train_hand_aug if hand_feat else None),
                ("val", val_sids, val_labels, val_hand_aug if hand_feat else None),
                ("test", test_sids, test_labels, test_hand_aug if hand_feat else None),
            ]:
                ds = BinaryPooledDataset(sids, labs, data["pooled_orig"],
                                         data["pooled_edited"], mode, hf)
                loaders[name] = DataLoader(ds, batch_size=bs, shuffle=(name == "train"),
                                           num_workers=0)
            return loaders

        loaders = _make_pooled_loaders("orig_only")
        model = BinaryMLP(d_input=640)
        metrics = train_eval_binary(model, loaders["train"], loaders["val"], loaders["test"],
                                    epochs=50, lr=1e-3, seed=fold_seed)
        model_fold_results["PooledMLP"].append(metrics)
        del model

        # ==============================================================
        # 6. SubtractionMLP
        # ==============================================================
        logger.info("  [6/13] SubtractionMLP")
        torch.manual_seed(fold_seed)
        loaders = _make_pooled_loaders("subtraction")
        model = BinaryMLP(d_input=640)
        metrics = train_eval_binary(model, loaders["train"], loaders["val"], loaders["test"],
                                    epochs=50, lr=1e-3, seed=fold_seed)
        model_fold_results["SubtractionMLP"].append(metrics)
        del model

        # ==============================================================
        # 7. ConcatMLP
        # ==============================================================
        logger.info("  [7/13] ConcatMLP")
        torch.manual_seed(fold_seed)
        loaders = _make_pooled_loaders("concat")
        model = BinaryMLP(d_input=1280)
        metrics = train_eval_binary(model, loaders["train"], loaders["val"], loaders["test"],
                                    epochs=50, lr=1e-3, seed=fold_seed)
        model_fold_results["ConcatMLP"].append(metrics)
        del model, loaders

        # ==============================================================
        # Token-level models (8-13) - load tokens
        # ==============================================================
        logger.info("  Loading token embeddings...")
        _to = torch.load(EMB_DIR / "rnafm_tokens.pt", weights_only=False)
        tokens_orig = {k: v for k, v in _to.items() if str(k) in set(str(s) for s in all_site_ids)}
        del _to; gc.collect()
        _te = torch.load(EMB_DIR / "rnafm_tokens_edited.pt", weights_only=False)
        tokens_edited = {k: v for k, v in _te.items() if str(k) in set(str(s) for s in all_site_ids)}
        del _te; gc.collect()

        def _make_token_loaders(hand_feat=False, bs=16):
            loaders = {}
            for name, sids, labs, hf in [
                ("train", train_sids, train_labels, train_hand_aug if hand_feat else None),
                ("val", val_sids, val_labels, val_hand_aug if hand_feat else None),
                ("test", test_sids, test_labels, test_hand_aug if hand_feat else None),
            ]:
                ds = TokenBinaryDataset(sids, labs, tokens_orig, tokens_edited, hf)
                loaders[name] = DataLoader(ds, batch_size=bs, shuffle=(name == "train"),
                                           collate_fn=token_collate_fn, num_workers=0)
            return loaders

        # ==============================================================
        # 8. CrossAttention
        # ==============================================================
        logger.info("  [8/13] CrossAttention")
        torch.manual_seed(fold_seed)
        tok_loaders = _make_token_loaders(hand_feat=False, bs=16)
        model = CrossAttentionBinary()
        metrics = train_eval_binary(model, tok_loaders["train"], tok_loaders["val"],
                                    tok_loaders["test"], is_token_model=True,
                                    epochs=30, lr=5e-4, weight_decay=1e-4, patience=8,
                                    seed=fold_seed)
        model_fold_results["CrossAttention"].append(metrics)
        del model

        # ==============================================================
        # 9. DiffAttention
        # ==============================================================
        logger.info("  [9/13] DiffAttention")
        torch.manual_seed(fold_seed)
        model = DiffAttentionBinary()
        metrics = train_eval_binary(model, tok_loaders["train"], tok_loaders["val"],
                                    tok_loaders["test"], is_token_model=True,
                                    epochs=30, lr=5e-4, weight_decay=1e-4, patience=8,
                                    seed=fold_seed)
        model_fold_results["DiffAttention"].append(metrics)
        del model

        # ==============================================================
        # 10. EditRNA-A3A
        # ==============================================================
        logger.info("  [10/13] EditRNA-A3A")
        try:
            torch.manual_seed(fold_seed)
            from data.apobec_dataset import (
                APOBECDataConfig, APOBECDataset, APOBECSiteSample,
                N_TISSUES, apobec_collate_fn, get_flanking_context,
            )
            from models.editrna_a3a import EditRNA_A3A, EditRNAConfig
            from models.encoders import CachedRNAEncoder

            def _build_editrna_samples(sids, labels):
                samples = []
                for sid, label in zip(sids, labels):
                    sid_str = str(sid)
                    seq = data["sequences"].get(sid_str, "A" * 201)
                    edit_pos = min(100, len(seq) // 2)
                    seq_list = list(seq)
                    seq_list[edit_pos] = "C"
                    seq = "".join(seq_list)
                    flanking = get_flanking_context(seq, edit_pos)
                    struct_d = data["structure_delta"].get(sid_str)
                    if struct_d is not None:
                        struct_d = np.array(struct_d, dtype=np.float32)
                    concordance = np.zeros(5, dtype=np.float32)
                    sample = APOBECSiteSample(
                        sequence=seq, edit_pos=edit_pos,
                        is_edited=float(label),
                        editing_rate_log2=float("nan"),
                        apobec_class=-1, structure_type=-1, tissue_spec_class=-1,
                        n_tissues_log2=float("nan"), exonic_function=-1,
                        conservation=float("nan"), cancer_survival=float("nan"),
                        tissue_rates=np.full(N_TISSUES, np.nan, dtype=np.float32),
                        hek293_rate=float("nan"),
                        flanking_context=flanking, concordance_features=concordance,
                        structure_delta=struct_d, site_id=sid_str,
                        chrom="", position=0, gene="",
                    )
                    samples.append(sample)
                return samples

            data_config = APOBECDataConfig(window_size=100)
            train_ds = APOBECDataset(_build_editrna_samples(train_sids, train_labels), data_config)
            val_ds = APOBECDataset(_build_editrna_samples(val_sids, val_labels), data_config)
            test_ds = APOBECDataset(_build_editrna_samples(test_sids, test_labels), data_config)

            tr_e = DataLoader(train_ds, batch_size=32, shuffle=True,
                              collate_fn=apobec_collate_fn, num_workers=0)
            va_e = DataLoader(val_ds, batch_size=64, shuffle=False,
                              collate_fn=apobec_collate_fn, num_workers=0)
            te_e = DataLoader(test_ds, batch_size=64, shuffle=False,
                              collate_fn=apobec_collate_fn, num_workers=0)

            cached_encoder = CachedRNAEncoder(
                tokens_cache=tokens_orig, pooled_cache=data["pooled_orig"],
                tokens_edited_cache=tokens_edited, pooled_edited_cache=data["pooled_edited"],
                d_model=640,
            )
            config = EditRNAConfig(
                primary_encoder="cached", d_model=640, d_edit=256, d_fused=512,
                edit_n_heads=8, use_structure_delta=True,
                head_dropout=0.2, fusion_dropout=0.2,
                focal_gamma=2.0, focal_alpha_binary=0.75,
            )
            editrna = EditRNA_A3A(config=config, primary_encoder=cached_encoder)
            optimizer = AdamW(editrna.get_parameter_groups(), lr=1e-4)
            scheduler = CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-7)
            focal = FocalLoss(gamma=2.0, alpha=0.75)

            best_val_auroc = 0.0
            patience_counter = 0
            best_state = None
            for epoch in range(1, 31):
                editrna.train()
                for batch in tr_e:
                    batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else
                             ({kk: vv.to(DEVICE) if isinstance(vv, torch.Tensor) else vv
                               for kk, vv in v.items()} if isinstance(v, dict) else v)
                             for k, v in batch.items()}
                    optimizer.zero_grad()
                    output = editrna(batch)
                    loss, _ = editrna.compute_loss(output, batch["targets"])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(editrna.parameters(), 1.0)
                    optimizer.step()
                scheduler.step()

                editrna.eval()
                vp, vt = [], []
                with torch.no_grad():
                    for batch in va_e:
                        batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else
                                 ({kk: vv.to(DEVICE) if isinstance(vv, torch.Tensor) else vv
                                   for kk, vv in v.items()} if isinstance(v, dict) else v)
                                 for k, v in batch.items()}
                        out = editrna(batch)
                        vp.append(torch.sigmoid(out["predictions"]["binary_logit"].squeeze(-1)).cpu().numpy())
                        vt.append(batch["targets"]["binary"].cpu().numpy())
                vp_arr = np.concatenate(vp)
                vt_arr = np.concatenate(vt)
                val_auroc = roc_auc_score(vt_arr, vp_arr) if len(np.unique(vt_arr)) > 1 else 0.0
                if val_auroc > best_val_auroc + 1e-4:
                    best_val_auroc = val_auroc
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in editrna.state_dict().items()}
                else:
                    patience_counter += 1
                if patience_counter >= 8:
                    break

            if best_state:
                editrna.load_state_dict(best_state)
            editrna.eval()
            tp, tt = [], []
            with torch.no_grad():
                for batch in te_e:
                    batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else
                             ({kk: vv.to(DEVICE) if isinstance(vv, torch.Tensor) else vv
                               for kk, vv in v.items()} if isinstance(v, dict) else v)
                             for k, v in batch.items()}
                    out = editrna(batch)
                    tp.append(torch.sigmoid(out["predictions"]["binary_logit"].squeeze(-1)).cpu().numpy())
                    tt.append(batch["targets"]["binary"].cpu().numpy())
            tp_arr = np.concatenate(tp)
            tt_arr = np.concatenate(tt)
            model_fold_results["EditRNA-A3A"].append(compute_binary_metrics(tt_arr, tp_arr))
            del editrna, cached_encoder
        except Exception as e:
            logger.error("EditRNA-A3A FAILED fold %d: %s", fold_idx + 1, e, exc_info=True)
            model_fold_results["EditRNA-A3A"].append(
                {"auroc": float("nan"), "auprc": float("nan"), "f1": float("nan"),
                 "precision": float("nan"), "recall": float("nan")})

        # ==============================================================
        # 11. EditRNA+Features (EditRNA + 33-dim hand features via d_gnn slot)
        # ==============================================================
        logger.info("  [11/13] EditRNA+Features")
        try:
            torch.manual_seed(fold_seed)
            from models.editrna_a3a import EditRNA_A3A, EditRNAConfig
            from models.encoders import CachedRNAEncoder
            from models.fusion import GatedModalityFusion

            cached_encoder = CachedRNAEncoder(
                tokens_cache=tokens_orig, pooled_cache=data["pooled_orig"],
                tokens_edited_cache=tokens_edited, pooled_edited_cache=data["pooled_edited"],
                d_model=640,
            )
            config = EditRNAConfig(
                primary_encoder="cached", d_model=640, d_edit=256, d_fused=512,
                edit_n_heads=8, use_structure_delta=True,
                head_dropout=0.2, fusion_dropout=0.2,
                focal_gamma=2.0, focal_alpha_binary=0.75,
            )
            editrna_feat = EditRNA_A3A(config=config, primary_encoder=cached_encoder)
            editrna_feat.fusion = GatedModalityFusion(
                d_model=640, d_edit=256, d_fused=512,
                d_model_secondary=0, d_gnn=d_hand_aug, dropout=0.2,
            )

            # Build hand feature lookup: site_id -> tensor
            hand_feat_map = {}
            for i, sid in enumerate(train_sids):
                hand_feat_map[str(sid)] = torch.tensor(train_hand_aug[i], dtype=torch.float32)
            for i, sid in enumerate(val_sids):
                hand_feat_map[str(sid)] = torch.tensor(val_hand_aug[i], dtype=torch.float32)
            for i, sid in enumerate(test_sids):
                hand_feat_map[str(sid)] = torch.tensor(test_hand_aug[i], dtype=torch.float32)

            def _inject_hand_features(batch):
                """Inject hand features into batch for fusion gnn_emb slot."""
                batch_sids = batch["site_ids"]
                hand_batch = torch.stack([hand_feat_map.get(str(sid), torch.zeros(d_hand_aug))
                                         for sid in batch_sids]).to(DEVICE)
                batch["hand_features"] = hand_batch
                return batch

            optimizer = AdamW(editrna_feat.get_parameter_groups(), lr=1e-4)
            scheduler = CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-7)

            best_val_auroc = 0.0
            patience_counter = 0
            best_state = None

            for epoch in range(1, 31):
                editrna_feat.train()
                for batch in tr_e:
                    batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else
                             ({kk: vv.to(DEVICE) if isinstance(vv, torch.Tensor) else vv
                               for kk, vv in v.items()} if isinstance(v, dict) else v)
                             for k, v in batch.items()}
                    batch = _inject_hand_features(batch)
                    optimizer.zero_grad()
                    output = editrna_feat(batch)
                    loss, _ = editrna_feat.compute_loss(output, batch["targets"])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(editrna_feat.parameters(), 1.0)
                    optimizer.step()
                scheduler.step()

                editrna_feat.eval()
                vp, vt = [], []
                with torch.no_grad():
                    for batch in va_e:
                        batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else
                                 ({kk: vv.to(DEVICE) if isinstance(vv, torch.Tensor) else vv
                                   for kk, vv in v.items()} if isinstance(v, dict) else v)
                                 for k, v in batch.items()}
                        batch = _inject_hand_features(batch)
                        out = editrna_feat(batch)
                        vp.append(torch.sigmoid(out["predictions"]["binary_logit"].squeeze(-1)).cpu().numpy())
                        vt.append(batch["targets"]["binary"].cpu().numpy())
                vp_arr = np.concatenate(vp)
                vt_arr = np.concatenate(vt)
                val_auroc = roc_auc_score(vt_arr, vp_arr) if len(np.unique(vt_arr)) > 1 else 0.0
                if val_auroc > best_val_auroc + 1e-4:
                    best_val_auroc = val_auroc
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in editrna_feat.state_dict().items()}
                else:
                    patience_counter += 1
                if patience_counter >= 8:
                    break

            if best_state:
                editrna_feat.load_state_dict(best_state)
            editrna_feat.eval()
            tp, tt = [], []
            with torch.no_grad():
                for batch in te_e:
                    batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else
                             ({kk: vv.to(DEVICE) if isinstance(vv, torch.Tensor) else vv
                               for kk, vv in v.items()} if isinstance(v, dict) else v)
                             for k, v in batch.items()}
                    batch = _inject_hand_features(batch)
                    out = editrna_feat(batch)
                    tp.append(torch.sigmoid(out["predictions"]["binary_logit"].squeeze(-1)).cpu().numpy())
                    tt.append(batch["targets"]["binary"].cpu().numpy())
            tp_arr = np.concatenate(tp)
            tt_arr = np.concatenate(tt)
            model_fold_results["EditRNA+Features"].append(compute_binary_metrics(tt_arr, tp_arr))
            del editrna_feat, cached_encoder, hand_feat_map
        except Exception as e:
            logger.error("EditRNA+Features FAILED fold %d: %s", fold_idx + 1, e, exc_info=True)
            model_fold_results["EditRNA+Features"].append(
                {"auroc": float("nan"), "auprc": float("nan"), "f1": float("nan"),
                 "precision": float("nan"), "recall": float("nan")})

        # ==============================================================
        # 12. SubtractionMLP+Features
        # ==============================================================
        logger.info("  [12/13] SubtractionMLP+Features")
        torch.manual_seed(fold_seed)
        loaders = _make_pooled_loaders("subtraction", hand_feat=True)
        model = BinaryMLP(d_input=640 + d_hand_aug)
        metrics = train_eval_binary(model, loaders["train"], loaders["val"], loaders["test"],
                                    epochs=50, lr=1e-3, seed=fold_seed)
        model_fold_results["SubtractionMLP+Features"].append(metrics)
        del model

        # ==============================================================
        # 13. DiffAttention+Features
        # ==============================================================
        logger.info("  [13/13] DiffAttention+Features")
        torch.manual_seed(fold_seed)
        tok_loaders_feat = _make_token_loaders(hand_feat=True, bs=16)
        model = DiffAttentionBinary(d_hand=d_hand_aug)
        metrics = train_eval_binary(model, tok_loaders_feat["train"], tok_loaders_feat["val"],
                                    tok_loaders_feat["test"], is_token_model=True,
                                    epochs=30, lr=5e-4, weight_decay=1e-4, patience=8,
                                    seed=fold_seed)
        model_fold_results["DiffAttention+Features"].append(metrics)
        del model

        # Free token memory
        del tokens_orig, tokens_edited, tok_loaders, tok_loaders_feat
        gc.collect()

        # Log fold summary
        logger.info("\n  Fold %d summary:", fold_idx + 1)
        for name in MODEL_NAMES:
            fr = model_fold_results[name][-1]
            auroc = fr.get("auroc", float("nan"))
            auprc = fr.get("auprc", float("nan"))
            logger.info("    %-30s  AUROC=%.4f  AUPRC=%.4f",
                        name,
                        auroc if not np.isnan(auroc) else 0.0,
                        auprc if not np.isnan(auprc) else 0.0)

        gc.collect()

    # ------------------------------------------------------------------
    # Aggregate Results
    # ------------------------------------------------------------------
    total_time = time.time() - t_total

    results = {
        "experiment": "classification_a3a_5fold",
        "data_file": "splits_expanded_a3a.csv",
        "n_folds": 5,
        "n_total_sites": n_total,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "models": {},
        "total_time_seconds": round(total_time, 1),
    }

    METRICS = ["auroc", "auprc", "f1", "precision", "recall"]
    for name in MODEL_NAMES:
        fold_results = model_fold_results[name]
        model_stats = {"fold_results": fold_results}
        for metric in METRICS:
            vals = [fr[metric] for fr in fold_results
                    if not np.isnan(fr.get(metric, float("nan")))]
            model_stats[f"mean_{metric}"] = float(np.mean(vals)) if vals else float("nan")
            model_stats[f"std_{metric}"] = float(np.std(vals)) if len(vals) > 1 else 0.0
        results["models"][name] = model_stats

    # ------------------------------------------------------------------
    # Save feature importance CSVs
    # ------------------------------------------------------------------
    def _build_hand_feature_names():
        names = []
        for ctx in ["UC", "CC", "AC", "GC"]:
            names.append(f"5p_{ctx}")
        for ctx in ["CA", "CG", "CU", "CC"]:
            names.append(f"3p_{ctx}")
        for offset, label in [(-2, "m2"), (-1, "m1")]:
            for b in ["A", "C", "G", "U"]:
                names.append(f"trinuc_up_{label}_{b}")
        for offset, label in [(1, "p1"), (2, "p2")]:
            for b in ["A", "C", "G", "U"]:
                names.append(f"trinuc_down_{label}_{b}")
        for n in ["delta_pairing_center", "delta_accessibility_center",
                   "delta_entropy_center", "delta_mfe",
                   "mean_delta_pairing_window", "mean_delta_accessibility_window",
                   "std_delta_pairing_window"]:
            names.append(n)
        for n in ["is_unpaired", "loop_size", "dist_to_junction", "dist_to_apex",
                   "relative_loop_position", "left_stem_length", "right_stem_length",
                   "max_adjacent_stem_length", "local_unpaired_fraction"]:
            names.append(n)
        return names

    def _categorize_feature(name):
        if name.startswith("5p_") or name.startswith("3p_") or "trinuc" in name:
            return "Motif"
        if "delta" in name or "mfe" in name.lower():
            return "Structure Delta"
        if any(k in name for k in ["unpaired", "loop", "stem", "junction", "apex"]):
            return "Loop Geometry"
        if "emb_delta_" in name:
            return "Embedding Delta"
        return "Other"

    hand_names = _build_hand_feature_names()
    if gb_hand_importances:
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
        pd.DataFrame(rows).sort_values("mean_importance", ascending=False).to_csv(
            OUTPUT_DIR / "feature_importance_cls_gb_hand.csv", index=False)

    if gb_all_importances:
        n_hand = len(hand_names)
        pp_names = []
        for w in [5, 11, 21]:
            for m in ["pp_mean", "pp_std", "pp_ed_mean", "pp_ed_std", "pp_delta_mean"]:
                pp_names.append(f"w{w}_{m}")
        for w in [5, 11, 21]:
            for m in ["acc_mean", "acc_std", "acc_ed_mean", "acc_ed_std", "acc_delta_mean"]:
                pp_names.append(f"w{w}_{m}")
        pp_names += ["center_pp", "center_pp_ed", "center_pp_delta",
                     "center_acc", "center_acc_ed", "center_acc_delta",
                     "mfe_orig", "mfe_edited", "mfe_delta"]
        for i in range(10):
            pp_names.append(f"pp_bin{i}")
        pp_names.append("pp_center_window")
        emb_names = [f"emb_delta_{i}" for i in range(640)]
        all_names = hand_names + pp_names + emb_names
        n_actual = len(gb_all_importances[0])
        while len(all_names) < n_actual:
            all_names.append(f"feature_{len(all_names)}")
        all_names = all_names[:n_actual]

        imp_mean = np.mean(gb_all_importances, axis=0)
        imp_std = np.std(gb_all_importances, axis=0)
        rows = []
        for j, name in enumerate(all_names):
            rows.append({
                "feature_name": name,
                "mean_importance": float(imp_mean[j]),
                "std_importance": float(imp_std[j]),
                "category": _categorize_feature(name),
            })
        pd.DataFrame(rows).sort_values("mean_importance", ascending=False).to_csv(
            OUTPUT_DIR / "feature_importance_cls_gb_all.csv", index=False)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    out_path = OUTPUT_DIR / "classification_a3a_5fold_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=_serialize)
    logger.info("\nResults saved to %s", out_path)

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("5-FOLD CV CLASSIFICATION RESULTS (A3A-FILTERED)")
    print(f"Total sites: {n_total} (pos={n_pos}, neg={n_neg})")
    print("=" * 100)
    print(f"{'Model':<30} {'AUROC':>16} {'AUPRC':>16} {'F1':>12} {'Prec':>12} {'Recall':>12}")
    print("-" * 100)

    sorted_models = sorted(
        results["models"].items(),
        key=lambda x: x[1]["mean_auroc"] if not np.isnan(x[1]["mean_auroc"]) else -999,
        reverse=True)

    for name, m in sorted_models:
        auroc = m["mean_auroc"]
        auroc_std = m["std_auroc"]
        auprc = m["mean_auprc"]
        auprc_std = m["std_auprc"]
        f1_val = m["mean_f1"]
        prec = m["mean_precision"]
        rec = m["mean_recall"]

        def _fmt(v, s):
            if np.isnan(v):
                return "N/A"
            return f"{v:.4f}+/-{s:.4f}"

        print(f"{name:<30} {_fmt(auroc, auroc_std):>16} {_fmt(auprc, auprc_std):>16} "
              f"{f1_val:>12.4f} {prec:>12.4f} {rec:>12.4f}")

    print("=" * 100)
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
