#!/usr/bin/env python
"""V5 dataset architecture screen: A6 (Conv2DBP) and A8 (HierarchicalAttention).

Runs 4 combinations on the V5 dataset (17,790 sites: 10,002 pos + 7,788 neg):
  1. A6 + baseline (standard 2-stage training)
  2. A6 + pretext pretraining (predict unpaired center, then fine-tune)
  3. A8 + baseline
  4. A8 + pretext pretraining

Also runs XGB per-enzyme baseline on V5 for comparison.

Output: experiments/multi_enzyme/outputs/architecture_screen/v5_screen_results.json
Log:    experiments/multi_enzyme/outputs/architecture_screen/v5_screen.log
Run:    /opt/miniconda3/envs/quris/bin/python -u experiments/multi_enzyme/exp_v5_screen.py
"""

import gc
import json
import logging
import math
import os
import sys
import time
import traceback
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
        logging.FileHandler(OUTPUT_DIR / "v5_screen.log", mode="w"),
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
# ViennaRNA workers (parallel)
# ---------------------------------------------------------------------------


def _fold_and_bpp_worker(args):
    """Worker: returns (site_id, 41x41 BP probability submatrix)."""
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
# Data Loading for V5
# ---------------------------------------------------------------------------


def load_v5_data() -> Dict:
    """Load V5 dataset with all pre-computed features, computing missing ones."""
    logger.info("=" * 70)
    logger.info("Loading V5 dataset")
    logger.info("=" * 70)

    # --- Splits ---
    splits_path = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "splits_multi_enzyme_v5.csv"
    df = pd.read_csv(splits_path)
    logger.info(f"  Loaded {len(df)} sites from V5 splits CSV")

    # --- Sequences (V5 has its own file) ---
    seq_path = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "multi_enzyme_sequences_v5.json"
    with open(seq_path) as f:
        sequences = json.load(f)
    logger.info(f"  Loaded {len(sequences)} V5 sequences")

    # --- Filter to sites with sequences ---
    site_ids = df["site_id"].tolist()
    valid_mask = [sid in sequences for sid in site_ids]
    df = df[valid_mask].reset_index(drop=True)
    site_ids = df["site_id"].tolist()
    n = len(site_ids)
    logger.info(f"  {n} sites with sequences")

    # --- Loop geometry (v3 + compute missing) ---
    loop_path = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "loop_position_per_site_v3.csv"
    loop_df = pd.read_csv(loop_path)
    loop_df = loop_df.drop_duplicates(subset=["site_id"]).set_index("site_id")
    n_loop_found = sum(1 for sid in site_ids if sid in loop_df.index)
    logger.info(f"  Loop CSV coverage: {n_loop_found}/{n}")

    # --- Structure delta (merge v3 + A3A caches) ---
    struct_path_v3 = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "structure_cache_multi_enzyme_v3.npz"
    struct_data_v3 = np.load(struct_path_v3, allow_pickle=True)
    struct_ids_v3 = list(struct_data_v3["site_ids"])
    struct_deltas_v3 = struct_data_v3["delta_features"]
    structure_delta = {sid: struct_deltas_v3[i] for i, sid in enumerate(struct_ids_v3)}

    struct_path_a3a = PROJECT_ROOT / "data" / "processed" / "embeddings" / "vienna_structure_cache.npz"
    if struct_path_a3a.exists():
        struct_data_a3a = np.load(struct_path_a3a, allow_pickle=True)
        struct_ids_a3a = list(struct_data_a3a["site_ids"])
        struct_deltas_a3a = struct_data_a3a["delta_features"]
        for i, sid in enumerate(struct_ids_a3a):
            if sid not in structure_delta:
                structure_delta[sid] = struct_deltas_a3a[i]
    n_struct = sum(1 for sid in site_ids if sid in structure_delta)
    logger.info(f"  Structure delta coverage: {n_struct}/{n}")

    # --- Hand features (40-dim) ---
    hand_features = build_hand_features(site_ids, sequences, structure_delta, loop_df)
    logger.info(f"  Hand features shape: {hand_features.shape}")

    # --- RNA-FM pooled embeddings (640-dim) ---
    rnafm_path = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "embeddings" / "rnafm_pooled_v3.pt"
    rnafm_emb = torch.load(rnafm_path, map_location="cpu", weights_only=False)
    logger.info(f"  Loaded RNA-FM embeddings: {len(rnafm_emb)} entries")

    rnafm_edited_path = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "embeddings" / "rnafm_pooled_edited_v3.pt"
    rnafm_edited_emb = torch.load(rnafm_edited_path, map_location="cpu", weights_only=False)
    logger.info(f"  Loaded RNA-FM edited embeddings: {len(rnafm_edited_emb)} entries")

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

    # --- BP submatrices: load v3 cache, compute missing via ViennaRNA ---
    bp_cache_path = OUTPUT_DIR / "bp_submatrices_v5.npz"
    if bp_cache_path.exists():
        logger.info(f"  Loading cached V5 BP submatrices from {bp_cache_path}")
        bp_data = np.load(bp_cache_path, allow_pickle=True)
        bp_submatrices = bp_data["bp_submatrices"]
        bp_site_ids = list(bp_data["site_ids"])
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
        # Load v3 cache for sites already computed
        bp_v3_path = OUTPUT_DIR / "bp_submatrices_v3.npz"
        bp_submatrices = np.zeros((n, 41, 41), dtype=np.float32)
        n_from_cache = 0
        if bp_v3_path.exists():
            bp_v3 = np.load(bp_v3_path, allow_pickle=True)
            bp_v3_ids = list(bp_v3["site_ids"])
            bp_v3_data = bp_v3["bp_submatrices"]
            bp_v3_map = {sid: i for i, sid in enumerate(bp_v3_ids)}
            for i, sid in enumerate(site_ids):
                if sid in bp_v3_map:
                    bp_submatrices[i] = bp_v3_data[bp_v3_map[sid]]
                    n_from_cache += 1
            del bp_v3, bp_v3_data
            gc.collect()
        logger.info(f"  BP from v3 cache: {n_from_cache}/{n}")

        # Compute missing
        missing_bp = [(sid, sequences[sid]) for i, sid in enumerate(site_ids)
                      if bp_submatrices[i].sum() == 0 and sid in sequences]
        if missing_bp:
            logger.info(f"  Computing BP submatrices for {len(missing_bp)} missing sites...")
            sid_to_idx = {sid: i for i, sid in enumerate(site_ids)}
            n_workers = min(14, os.cpu_count() or 4)
            t0 = time.time()
            done = 0
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {executor.submit(_fold_and_bpp_worker, item): item[0] for item in missing_bp}
                for future in as_completed(futures):
                    sid, sub = future.result()
                    if sid in sid_to_idx:
                        bp_submatrices[sid_to_idx[sid]] = sub
                    done += 1
                    if done % 1000 == 0:
                        logger.info(f"    Folded {done}/{len(missing_bp)} ({time.time() - t0:.0f}s)")
            logger.info(f"  BP folding complete: {done} sequences in {time.time() - t0:.0f}s")

        # Save V5 cache
        np.savez_compressed(bp_cache_path,
                            bp_submatrices=bp_submatrices,
                            site_ids=np.array(site_ids, dtype=object))
        logger.info(f"  Saved V5 BP cache to {bp_cache_path}")

    # --- Dot-bracket structures (for pretext labels) ---
    db_cache_path = OUTPUT_DIR / "dot_brackets_v5.json"
    if db_cache_path.exists():
        logger.info(f"  Loading cached V5 dot-bracket structures from {db_cache_path}")
        with open(db_cache_path) as f:
            dot_brackets = json.load(f)
        logger.info(f"  Loaded {len(dot_brackets)} dot-bracket structures")
    else:
        # Load v3 cache
        dot_brackets = {}
        db_v3_path = OUTPUT_DIR / "dot_brackets_v3.json"
        if db_v3_path.exists():
            with open(db_v3_path) as f:
                dot_brackets = json.load(f)
            logger.info(f"  Loaded {len(dot_brackets)} dot-brackets from v3 cache")

        # Also try loop_df for dot_bracket column
        if "dot_bracket" in loop_df.columns:
            for sid in site_ids:
                if sid not in dot_brackets and sid in loop_df.index:
                    db = loop_df.loc[sid, "dot_bracket"]
                    if isinstance(db, str) and len(db) == 201:
                        dot_brackets[sid] = db

        missing_db = [sid for sid in site_ids if sid not in dot_brackets]
        if missing_db:
            logger.info(f"  Computing dot-bracket for {len(missing_db)} missing sites...")
            work_items = [(sid, sequences[sid]) for sid in missing_db if sid in sequences]
            n_workers = min(14, os.cpu_count() or 4)
            t0 = time.time()
            done = 0
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {executor.submit(_fold_dotbracket_worker, item): item[0] for item in work_items}
                for future in as_completed(futures):
                    sid, db = future.result()
                    dot_brackets[sid] = db
                    done += 1
                    if done % 1000 == 0:
                        logger.info(f"    Folded {done}/{len(missing_db)} ({time.time() - t0:.0f}s)")
            logger.info(f"  Dot-bracket folding complete: {done} in {time.time() - t0:.0f}s")

        with open(db_cache_path, "w") as f:
            json.dump(dot_brackets, f)
        logger.info(f"  Saved V5 dot-bracket cache to {db_cache_path}")

    # --- Pretext labels: 1 if center is unpaired, 0 if paired, 0.5 if unknown ---
    pretext_labels = np.zeros(n, dtype=np.float32)
    n_unpaired = 0
    for i, sid in enumerate(site_ids):
        if sid in dot_brackets:
            db = dot_brackets[sid]
            if len(db) > CENTER:
                if db[CENTER] == ".":
                    pretext_labels[i] = 1.0
                    n_unpaired += 1
            else:
                pretext_labels[i] = 0.5
        else:
            pretext_labels[i] = 0.5
    logger.info(f"  Pretext labels: {n_unpaired} unpaired, {n - n_unpaired} paired/unknown")

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

    logger.info(f"  Total: {int(labels_binary.sum())} positives, {int((labels_binary == 0).sum())} negatives")
    logger.info("V5 data loading complete.")

    return {
        "site_ids": site_ids,
        "df": df,
        "sequences": sequences,
        "hand_features": hand_features,
        "rnafm_features": rnafm_matrix,
        "rnafm_edited_features": rnafm_edited_matrix,
        "edit_delta_features": edit_delta_matrix,
        "bp_submatrices": bp_submatrices,
        "dot_brackets": dot_brackets,
        "pretext_labels": pretext_labels,
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


class PretextDataset(Dataset):
    """Dataset for structure pretext task: predict if center position is unpaired."""

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
            "pretext_label": torch.tensor(self.data["pretext_labels"][i], dtype=torch.float32),
            "index": i,
        }


def standard_collate(batch_list):
    """Collate: stack all tensors."""
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
        """shared: [B, 128] -> binary_logit, per_enzyme_logits, enzyme_cls_logits."""
        binary_logit = self.binary_head(shared).squeeze(-1)
        per_enzyme_logits = [
            self.enzyme_adapters[enz](shared).squeeze(-1)
            for enz in PER_ENZYME_HEADS
        ]
        enzyme_cls_logits = self.enzyme_classifier(shared)
        return binary_logit, per_enzyme_logits, enzyme_cls_logits


# ---------------------------------------------------------------------------
# A6. Conv2DBP
# ---------------------------------------------------------------------------


class Conv2DBP(nn.Module, HeadsMixin):
    """Conv2D on 41x41 BP probability matrix + rnafm + hand -> heads."""

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

        d_fused = 128 + D_RNAFM + D_HAND  # 808
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
        conv_out = self.conv(bp)
        conv_out = conv_out.flatten(1)
        conv_out = self.conv_fc(conv_out)
        fused = torch.cat([conv_out, batch["rnafm"], batch["hand_feat"]], dim=-1)
        shared = self.encoder(fused)
        return self._apply_heads(shared)

    def encode(self, batch):
        """Return shared embedding without heads (for pretext)."""
        bp = batch["bp_submatrix"]
        conv_out = self.conv(bp)
        conv_out = conv_out.flatten(1)
        conv_out = self.conv_fc(conv_out)
        fused = torch.cat([conv_out, batch["rnafm"], batch["hand_feat"]], dim=-1)
        return self.encoder(fused)


# ---------------------------------------------------------------------------
# A8. HierarchicalAttention
# ---------------------------------------------------------------------------


class HierarchicalAttention(nn.Module, HeadsMixin):
    """Two-level attention: local transformer on BP + cross-attention to RNA-FM."""

    name = "A8_HierarchicalAttention"

    def __init__(self):
        super().__init__()
        d_local = 41
        self.local_proj = nn.Linear(d_local, 64)
        self.local_pos_enc = nn.Parameter(torch.randn(41, 64) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128,
            dropout=0.1, activation="gelu", batch_first=True,
        )
        self.local_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.local_pool_attn = nn.Linear(64, 1)

        self.cross_q = nn.Linear(64, 64)
        self.cross_k = nn.Linear(D_RNAFM, 64)
        self.cross_v = nn.Linear(D_RNAFM, 64)

        d_fused = 64 + 64 + D_RNAFM + D_HAND  # 808
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
        return (
            list(self.local_proj.parameters()) + [self.local_pos_enc] +
            list(self.local_transformer.parameters()) +
            list(self.local_pool_attn.parameters()) +
            list(self.cross_q.parameters()) + list(self.cross_k.parameters()) +
            list(self.cross_v.parameters()) + list(self.encoder.parameters())
        )

    def get_adapter_params(self):
        params = list(self.binary_head.parameters())
        for enz in PER_ENZYME_HEADS:
            params.extend(self.enzyme_adapters[enz].parameters())
        params.extend(self.enzyme_classifier.parameters())
        return params

    def forward(self, batch):
        bp = batch["bp_submatrix"].squeeze(1)  # [B, 41, 41]
        rnafm = batch["rnafm"]
        hand = batch["hand_feat"]

        local_in = self.local_proj(bp) + self.local_pos_enc.unsqueeze(0)
        local_out = self.local_transformer(local_in)

        attn_w = torch.softmax(self.local_pool_attn(local_out), dim=1)
        local_repr = (local_out * attn_w).sum(dim=1)

        q = self.cross_q(local_repr).unsqueeze(1)
        k = self.cross_k(rnafm).unsqueeze(1)
        v = self.cross_v(rnafm).unsqueeze(1)

        attn_scores = (q * k).sum(-1) / math.sqrt(64)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        cross_repr = (attn_weights.unsqueeze(-1) * v).squeeze(1)

        fused = torch.cat([local_repr, cross_repr, rnafm, hand], dim=-1)
        shared = self.encoder(fused)
        return self._apply_heads(shared)

    def encode(self, batch):
        """Return shared embedding without heads (for pretext)."""
        bp = batch["bp_submatrix"].squeeze(1)
        rnafm = batch["rnafm"]
        hand = batch["hand_feat"]

        local_in = self.local_proj(bp) + self.local_pos_enc.unsqueeze(0)
        local_out = self.local_transformer(local_in)

        attn_w = torch.softmax(self.local_pool_attn(local_out), dim=1)
        local_repr = (local_out * attn_w).sum(dim=1)

        q = self.cross_q(local_repr).unsqueeze(1)
        k = self.cross_k(rnafm).unsqueeze(1)
        v = self.cross_v(rnafm).unsqueeze(1)

        attn_scores = (q * k).sum(-1) / math.sqrt(64)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        cross_repr = (attn_weights.unsqueeze(-1) * v).squeeze(1)

        fused = torch.cat([local_repr, cross_repr, rnafm, hand], dim=-1)
        return self.encoder(fused)


ARCHITECTURE_CLASSES = {
    "A6_Conv2DBP": Conv2DBP,
    "A8_HierarchicalAttention": HierarchicalAttention,
}


# ---------------------------------------------------------------------------
# Pretext head
# ---------------------------------------------------------------------------


class PretextHead(nn.Module):
    """Simple binary head for pretext prediction (predict unpaired center)."""

    def __init__(self):
        super().__init__()
        self.head = nn.Linear(D_SHARED, 1)

    def forward(self, shared):
        return self.head(shared).squeeze(-1)


# ---------------------------------------------------------------------------
# Training functions
# ---------------------------------------------------------------------------


def compute_loss(binary_logit, per_enzyme_logits, enzyme_cls_logits, batch):
    """Compute combined loss: binary + per-enzyme + enzyme classifier."""
    label_binary = batch["label_binary"]
    label_enzyme = batch["label_enzyme"]
    per_enzyme_labels = batch["per_enzyme_labels"]

    loss_binary = F.binary_cross_entropy_with_logits(binary_logit, label_binary)

    loss_enzyme_heads = torch.tensor(0.0, device=binary_logit.device)
    n_valid_heads = 0
    for head_idx in range(N_PER_ENZYME):
        enz_labels = per_enzyme_labels[:, head_idx]
        mask = enz_labels >= 0
        if mask.sum() > 0:
            head_loss = F.binary_cross_entropy_with_logits(
                per_enzyme_logits[head_idx][mask], enz_labels[mask]
            )
            loss_enzyme_heads = loss_enzyme_heads + head_loss
            n_valid_heads += 1
    if n_valid_heads > 0:
        loss_enzyme_heads = loss_enzyme_heads / n_valid_heads

    pos_mask = label_binary > 0.5
    loss_cls = torch.tensor(0.0, device=binary_logit.device)
    if pos_mask.sum() > 0:
        loss_cls = F.cross_entropy(enzyme_cls_logits[pos_mask], label_enzyme[pos_mask])

    total = loss_binary + STAGE1_ENZYME_HEAD_WEIGHT * loss_enzyme_heads + STAGE1_ENZYME_CLS_WEIGHT * loss_cls
    return total, loss_binary.item()


def train_one_epoch(model, loader, optimizer, device):
    """Train one epoch. Returns (avg_loss, avg_binary_loss)."""
    model.train()
    total_loss, total_binary, n_batches = 0.0, 0.0, 0

    for batch in loader:
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
def evaluate(model, loader, device):
    """Evaluate: returns overall AUROC and per-enzyme AUROCs."""
    model.eval()
    all_probs, all_labels, all_enzymes = [], [], []
    all_per_enzyme_probs = [[] for _ in range(N_PER_ENZYME)]
    all_per_enzyme_labels_list = [[] for _ in range(N_PER_ENZYME)]

    for batch in loader:
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

    try:
        overall_auroc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        overall_auroc = 0.5

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


def _aggregate_fold_results(fold_results, total_time):
    """Aggregate fold results into summary dict."""
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

    return {
        "overall_auroc_mean": round(float(mean_overall), 4),
        "overall_auroc_std": round(float(std_overall), 4),
        "per_enzyme_auroc_mean": {k: round(float(v), 4) for k, v in per_enz_means.items()},
        "per_enzyme_auroc_std": {k: round(float(v), 4) for k, v in per_enz_stds.items()},
        "total_time_s": round(total_time, 1),
        "fold_results": fold_results,
    }


# ---------------------------------------------------------------------------
# Training method: T1 Baseline (standard 2-stage)
# ---------------------------------------------------------------------------


def run_baseline(arch_cls, arch_name: str, data: Dict) -> Dict:
    """T1: Standard 2-stage training (joint + adapter-only)."""
    logger.info(f"\n{'=' * 70}")
    logger.info(f"T1 Baseline: {arch_name} on V5")
    logger.info(f"{'=' * 70}")

    n = len(data["labels_binary"])
    strat_key = data["labels_enzyme"] * 2 + data["labels_binary"].astype(np.int64)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    fold_results = []
    total_t0 = time.time()

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(n), strat_key)):
        logger.info(f"\n--- Fold {fold + 1}/{N_FOLDS} ({len(train_idx)} train, {len(val_idx)} val) ---")
        fold_t0 = time.time()

        fold_model = arch_cls().to(DEVICE)

        train_ds = ScreenDataset(train_idx, data)
        val_ds = ScreenDataset(val_idx, data)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  collate_fn=standard_collate, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                collate_fn=standard_collate, num_workers=0)

        # Stage 1: Joint training
        logger.info(f"  Stage 1: Joint training ({STAGE1_EPOCHS} epochs, lr={STAGE1_LR})")
        optimizer = torch.optim.Adam([
            {"params": fold_model.get_shared_params(), "lr": STAGE1_LR},
            {"params": fold_model.get_adapter_params(), "lr": STAGE1_LR},
        ], weight_decay=STAGE1_WEIGHT_DECAY)

        for epoch in range(STAGE1_EPOCHS):
            avg_loss, avg_binary = train_one_epoch(fold_model, train_loader, optimizer, DEVICE)
            val_auroc, val_per_enz = evaluate(fold_model, val_loader, DEVICE)
            enz_str = " ".join(f"{e}={v:.3f}" for e, v in val_per_enz.items() if not np.isnan(v))
            logger.info(
                f"    Epoch {epoch + 1:2d} | loss={avg_loss:.4f} bce={avg_binary:.4f} "
                f"| val_auroc={val_auroc:.4f} | {enz_str}"
            )

        # Stage 2: Adapter-only
        logger.info(f"  Stage 2: Adapter-only ({STAGE2_EPOCHS} epochs, lr={STAGE2_LR})")
        for p in fold_model.get_shared_params():
            p.requires_grad = False

        optimizer2 = torch.optim.Adam(
            fold_model.get_adapter_params(),
            lr=STAGE2_LR, weight_decay=STAGE2_WEIGHT_DECAY,
        )

        for epoch in range(STAGE2_EPOCHS):
            avg_loss, avg_binary = train_one_epoch(fold_model, train_loader, optimizer2, DEVICE)
            val_auroc, val_per_enz = evaluate(fold_model, val_loader, DEVICE)
            if (epoch + 1) % 5 == 0 or epoch == STAGE2_EPOCHS - 1:
                enz_str = " ".join(f"{e}={v:.3f}" for e, v in val_per_enz.items() if not np.isnan(v))
                logger.info(
                    f"    Epoch {STAGE1_EPOCHS + epoch + 1:2d} | loss={avg_loss:.4f} "
                    f"| val_auroc={val_auroc:.4f} | {enz_str}"
                )

        # Final evaluation
        final_auroc, final_per_enz = evaluate(fold_model, val_loader, DEVICE)
        fold_time = time.time() - fold_t0
        logger.info(f"  Fold {fold + 1} FINAL: overall_auroc={final_auroc:.4f} | time={fold_time:.0f}s")
        for enz, auroc in final_per_enz.items():
            logger.info(f"    {enz}: {auroc:.4f}")

        fold_results.append({
            "fold": fold + 1,
            "overall_auroc": final_auroc,
            "per_enzyme_aurocs": final_per_enz,
            "time_s": fold_time,
        })

        del fold_model, optimizer, optimizer2, train_loader, val_loader, train_ds, val_ds
        gc.collect()
        if DEVICE.type == "mps":
            torch.mps.empty_cache()

    total_time = time.time() - total_t0
    result = _aggregate_fold_results(fold_results, total_time)
    result["architecture"] = arch_name
    result["training_method"] = "T1_baseline"

    logger.info(f"\n  {arch_name} T1_baseline SUMMARY:")
    logger.info(f"    Overall AUROC: {result['overall_auroc_mean']:.4f} +/- {result['overall_auroc_std']:.4f}")
    for enz in PER_ENZYME_HEADS:
        if enz in result["per_enzyme_auroc_mean"]:
            logger.info(f"    {enz}: {result['per_enzyme_auroc_mean'][enz]:.4f}")
    logger.info(f"    Total time: {total_time:.0f}s")

    return result


# ---------------------------------------------------------------------------
# Training method: T6 Pretext pretraining
# ---------------------------------------------------------------------------


def pretrain_pretext(model, pretext_head, train_loader, device, n_epochs=10, lr=1e-3):
    """Pretrain the shared encoder on the structure pretext task."""
    model.train()
    pretext_head.train()

    optimizer = torch.optim.Adam(
        list(model.get_shared_params()) + list(pretext_head.parameters()),
        lr=lr, weight_decay=1e-4,
    )

    for epoch in range(n_epochs):
        total_loss = 0.0
        n_batches = 0
        n_correct = 0
        n_total = 0

        for batch in train_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            optimizer.zero_grad()

            shared = model.encode(batch)
            logits = pretext_head(shared)
            labels = batch["pretext_label"]

            # Skip ambiguous labels (0.5)
            mask = (labels != 0.5)
            if mask.sum() == 0:
                continue

            loss = F.binary_cross_entropy_with_logits(logits[mask], labels[mask])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(pretext_head.parameters()), 1.0
            )
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            preds = (torch.sigmoid(logits[mask]) > 0.5).float()
            n_correct += (preds == labels[mask]).sum().item()
            n_total += mask.sum().item()

        avg_loss = total_loss / max(n_batches, 1)
        acc = n_correct / max(n_total, 1)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(f"    Pretext epoch {epoch + 1:2d} | loss={avg_loss:.4f} | acc={acc:.3f}")


def run_pretext(arch_cls, arch_name: str, data: Dict) -> Dict:
    """T6: Structure pretext pretraining -> fine-tune on editing classification."""
    logger.info(f"\n{'=' * 70}")
    logger.info(f"T6 Pretext: {arch_name} on V5")
    logger.info(f"{'=' * 70}")

    n = len(data["labels_binary"])
    strat_key = data["labels_binary"].astype(np.int64)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    fold_results = []
    total_t0 = time.time()

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(n), strat_key)):
        logger.info(f"\n--- Fold {fold + 1}/{N_FOLDS} ({len(train_idx)} train, {len(val_idx)} val) ---")
        fold_t0 = time.time()

        fold_model = arch_cls().to(DEVICE)
        pretext_head = PretextHead().to(DEVICE)

        # Stage 0: Pretext pretraining
        logger.info("  Stage 0: Pretext pretraining (10 epochs)")
        pretext_ds = PretextDataset(train_idx, data)
        pretext_loader = DataLoader(pretext_ds, batch_size=BATCH_SIZE, shuffle=True,
                                    collate_fn=standard_collate, num_workers=0)
        pretrain_pretext(fold_model, pretext_head, pretext_loader, DEVICE, n_epochs=10, lr=1e-3)

        del pretext_head
        gc.collect()

        # Re-initialize heads (shared encoder is pretrained, heads are fresh)
        fold_model._init_heads()
        fold_model = fold_model.to(DEVICE)

        # Stage 1: Fine-tune joint
        train_ds = ScreenDataset(train_idx, data)
        val_ds = ScreenDataset(val_idx, data)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  collate_fn=standard_collate, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                collate_fn=standard_collate, num_workers=0)

        logger.info(f"  Stage 1: Fine-tune joint ({STAGE1_EPOCHS} epochs)")
        optimizer = torch.optim.Adam([
            {"params": fold_model.get_shared_params(), "lr": STAGE1_LR},
            {"params": fold_model.get_adapter_params(), "lr": STAGE1_LR},
        ], weight_decay=STAGE1_WEIGHT_DECAY)

        for epoch in range(STAGE1_EPOCHS):
            avg_loss, avg_binary = train_one_epoch(fold_model, train_loader, optimizer, DEVICE)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                val_auroc, val_per_enz = evaluate(fold_model, val_loader, DEVICE)
                enz_str = " ".join(f"{e}={v:.3f}" for e, v in val_per_enz.items() if not np.isnan(v))
                logger.info(
                    f"    Epoch {epoch + 1:2d} | loss={avg_loss:.4f} bce={avg_binary:.4f} "
                    f"| val_auroc={val_auroc:.4f} | {enz_str}"
                )

        # Stage 2: Adapter-only
        logger.info(f"  Stage 2: Adapter-only ({STAGE2_EPOCHS} epochs)")
        for p in fold_model.get_shared_params():
            p.requires_grad = False

        optimizer2 = torch.optim.Adam(
            fold_model.get_adapter_params(),
            lr=STAGE2_LR, weight_decay=STAGE2_WEIGHT_DECAY,
        )

        for epoch in range(STAGE2_EPOCHS):
            avg_loss, avg_binary = train_one_epoch(fold_model, train_loader, optimizer2, DEVICE)
            if (epoch + 1) % 5 == 0 or epoch == STAGE2_EPOCHS - 1:
                val_auroc, val_per_enz = evaluate(fold_model, val_loader, DEVICE)
                enz_str = " ".join(f"{e}={v:.3f}" for e, v in val_per_enz.items() if not np.isnan(v))
                logger.info(
                    f"    Epoch {STAGE1_EPOCHS + epoch + 1:2d} | loss={avg_loss:.4f} "
                    f"| val_auroc={val_auroc:.4f} | {enz_str}"
                )

        # Final evaluation
        final_auroc, final_per_enz = evaluate(fold_model, val_loader, DEVICE)
        fold_time = time.time() - fold_t0
        logger.info(f"  Fold {fold + 1} FINAL: overall_auroc={final_auroc:.4f} | time={fold_time:.0f}s")
        for enz, auroc in final_per_enz.items():
            logger.info(f"    {enz}: {auroc:.4f}")

        fold_results.append({
            "fold": fold + 1,
            "overall_auroc": final_auroc,
            "per_enzyme_aurocs": final_per_enz,
            "time_s": fold_time,
        })

        del fold_model, optimizer, optimizer2, train_loader, val_loader, train_ds, val_ds
        gc.collect()
        if DEVICE.type == "mps":
            torch.mps.empty_cache()

    total_time = time.time() - total_t0
    result = _aggregate_fold_results(fold_results, total_time)
    result["architecture"] = arch_name
    result["training_method"] = "T6_pretext"

    logger.info(f"\n  {arch_name} T6_pretext SUMMARY:")
    logger.info(f"    Overall AUROC: {result['overall_auroc_mean']:.4f} +/- {result['overall_auroc_std']:.4f}")
    for enz in PER_ENZYME_HEADS:
        if enz in result["per_enzyme_auroc_mean"]:
            logger.info(f"    {enz}: {result['per_enzyme_auroc_mean'][enz]:.4f}")
    logger.info(f"    Total time: {total_time:.0f}s")

    return result


# ---------------------------------------------------------------------------
# XGB per-enzyme baseline on V5
# ---------------------------------------------------------------------------


def run_xgb_baseline(data: Dict) -> Dict:
    """XGB per-enzyme baseline: 40-dim hand features, 2-fold CV."""
    from xgboost import XGBClassifier

    logger.info(f"\n{'=' * 70}")
    logger.info("XGB per-enzyme baseline on V5")
    logger.info(f"{'=' * 70}")

    n = len(data["labels_binary"])
    X = data["hand_features"]
    y = data["labels_binary"]
    enzymes = data["labels_enzyme"]

    strat_key = enzymes * 2 + y.astype(np.int64)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    fold_results = []
    total_t0 = time.time()

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(n), strat_key)):
        logger.info(f"\n--- XGB Fold {fold + 1}/{N_FOLDS} ({len(train_idx)} train, {len(val_idx)} val) ---")
        fold_t0 = time.time()

        # Overall binary model
        clf = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=SEED,
            eval_metric="logloss", verbosity=0,
        )
        clf.fit(X[train_idx], y[train_idx])
        overall_probs = clf.predict_proba(X[val_idx])[:, 1]
        overall_auroc = roc_auc_score(y[val_idx], overall_probs)
        logger.info(f"  Overall AUROC: {overall_auroc:.4f}")

        # Per-enzyme models
        per_enzyme_aurocs = {}
        for enz_name in PER_ENZYME_HEADS:
            enz_idx = ENZYME_TO_IDX[enz_name]
            enz_train_mask = enzymes[train_idx] == enz_idx
            enz_val_mask = enzymes[val_idx] == enz_idx

            if enz_train_mask.sum() < 20 or enz_val_mask.sum() < 10:
                per_enzyme_aurocs[enz_name] = float("nan")
                continue

            X_enz_train = X[train_idx][enz_train_mask]
            y_enz_train = y[train_idx][enz_train_mask]
            X_enz_val = X[val_idx][enz_val_mask]
            y_enz_val = y[val_idx][enz_val_mask]

            if len(np.unique(y_enz_train)) < 2 or len(np.unique(y_enz_val)) < 2:
                per_enzyme_aurocs[enz_name] = float("nan")
                continue

            clf_enz = XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, random_state=SEED,
                eval_metric="logloss", verbosity=0,
            )
            clf_enz.fit(X_enz_train, y_enz_train)
            enz_probs = clf_enz.predict_proba(X_enz_val)[:, 1]
            try:
                enz_auroc = roc_auc_score(y_enz_val, enz_probs)
            except ValueError:
                enz_auroc = float("nan")
            per_enzyme_aurocs[enz_name] = enz_auroc
            logger.info(f"    {enz_name}: AUROC={enz_auroc:.4f} (n={enz_val_mask.sum()})")

        fold_time = time.time() - fold_t0
        fold_results.append({
            "fold": fold + 1,
            "overall_auroc": overall_auroc,
            "per_enzyme_aurocs": per_enzyme_aurocs,
            "time_s": fold_time,
        })

    total_time = time.time() - total_t0
    result = _aggregate_fold_results(fold_results, total_time)
    result["model"] = "XGB_40d"
    result["dataset"] = "v5"
    result["n_samples"] = n

    logger.info(f"\n  XGB V5 SUMMARY:")
    logger.info(f"    Overall AUROC: {result['overall_auroc_mean']:.4f} +/- {result['overall_auroc_std']:.4f}")
    for enz in PER_ENZYME_HEADS:
        if enz in result["per_enzyme_auroc_mean"]:
            logger.info(f"    {enz}: {result['per_enzyme_auroc_mean'][enz]:.4f}")
    logger.info(f"    Total time: {total_time:.0f}s")

    return result


# ---------------------------------------------------------------------------
# Comparison with V3 results
# ---------------------------------------------------------------------------


def load_v3_results() -> Dict:
    """Load V3 architecture screen results for comparison."""
    v3_results = {}

    # Architecture screen (T1 baseline)
    arch_path = OUTPUT_DIR / "architecture_screen_results.json"
    if arch_path.exists():
        with open(arch_path) as f:
            arch_data = json.load(f)
        for entry in arch_data:
            name = entry.get("architecture", "")
            if name in ("A6_Conv2DBP", "A8_HierarchicalAttention"):
                v3_results[f"{name}_T1_baseline"] = {
                    "overall_auroc_mean": entry.get("overall_auroc_mean"),
                    "per_enzyme_auroc_mean": entry.get("per_enzyme_auroc_mean", {}),
                }

    # T6 pretext
    t6_path = OUTPUT_DIR / "training_screen_t6t8_results.json"
    if t6_path.exists():
        with open(t6_path) as f:
            t6_data = json.load(f)
        for arch_name in ("A6_Conv2DBP", "A8_HierarchicalAttention"):
            if arch_name in t6_data and "T6_pretext" in t6_data[arch_name]:
                entry = t6_data[arch_name]["T6_pretext"]
                v3_results[f"{arch_name}_T6_pretext"] = {
                    "overall_auroc_mean": entry.get("overall_auroc_mean"),
                    "per_enzyme_auroc_mean": entry.get("per_enzyme_auroc_mean", {}),
                }

    # XGB baseline
    xgb_path = OUTPUT_DIR / "xgb_v4_baselines.json"
    if xgb_path.exists():
        with open(xgb_path) as f:
            xgb_data = json.load(f)
        if "v3" in xgb_data:
            v3_results["XGB_40d"] = {
                "overall_auroc_mean": xgb_data["v3"]["overall_auroc"]["mean"],
                "per_enzyme_auroc_mean": {k: v["mean"] for k, v in xgb_data["v3"]["per_enzyme_auroc"].items()},
            }

    return v3_results


def print_comparison_table(v5_results: Dict, v3_results: Dict):
    """Print comparison table of V5 vs V3 results."""
    logger.info("\n" + "=" * 90)
    logger.info("COMPARISON TABLE: V5 (17,790 sites) vs V3 (15,342 sites)")
    logger.info("=" * 90)

    # Header
    header = f"{'Combination':<35} {'V5 Overall':>12} {'V3 Overall':>12} {'Delta':>8}"
    for enz in PER_ENZYME_HEADS:
        header += f" {enz:>8}"
    logger.info(header)
    logger.info("-" * len(header))

    # Neural model rows
    for combo_key, v5_entry in v5_results.items():
        if combo_key == "XGB_40d":
            continue
        v5_auroc = v5_entry.get("overall_auroc_mean", 0)

        v3_key = combo_key
        v3_auroc = v3_results.get(v3_key, {}).get("overall_auroc_mean")
        delta = f"{v5_auroc - v3_auroc:+.4f}" if v3_auroc is not None else "  N/A"
        v3_str = f"{v3_auroc:.4f}" if v3_auroc is not None else "  N/A"

        row = f"{combo_key:<35} {v5_auroc:>12.4f} {v3_str:>12} {delta:>8}"
        for enz in PER_ENZYME_HEADS:
            v5_enz = v5_entry.get("per_enzyme_auroc_mean", {}).get(enz, float("nan"))
            if not np.isnan(v5_enz):
                row += f" {v5_enz:>8.4f}"
            else:
                row += f" {'N/A':>8}"
        logger.info(row)

    # XGB row
    if "XGB_40d" in v5_results:
        v5_xgb = v5_results["XGB_40d"]
        v5_auroc = v5_xgb.get("overall_auroc_mean", 0)
        v3_auroc = v3_results.get("XGB_40d", {}).get("overall_auroc_mean")
        delta = f"{v5_auroc - v3_auroc:+.4f}" if v3_auroc is not None else "  N/A"
        v3_str = f"{v3_auroc:.4f}" if v3_auroc is not None else "  N/A"

        row = f"{'XGB_40d':<35} {v5_auroc:>12.4f} {v3_str:>12} {delta:>8}"
        for enz in PER_ENZYME_HEADS:
            v5_enz = v5_xgb.get("per_enzyme_auroc_mean", {}).get(enz, float("nan"))
            if not np.isnan(v5_enz):
                row += f" {v5_enz:>8.4f}"
            else:
                row += f" {'N/A':>8}"
        logger.info(row)

    logger.info("=" * 90)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    logger.info("=" * 70)
    logger.info("V5 ARCHITECTURE SCREEN: A6 (Conv2DBP) + A8 (HierarchicalAttention)")
    logger.info(f"Device: {DEVICE} | Folds: {N_FOLDS} | Seed: {SEED}")
    logger.info(f"Stage 1: {STAGE1_EPOCHS} epochs, lr={STAGE1_LR}")
    logger.info(f"Stage 2: {STAGE2_EPOCHS} epochs, lr={STAGE2_LR}")
    logger.info("4 neural combinations + XGB baseline")
    logger.info("=" * 70)

    # Load V5 data
    data = load_v5_data()
    n = len(data["labels_binary"])
    logger.info(f"\nTotal V5 sites: {n}")
    logger.info(f"  Positive: {int(data['labels_binary'].sum())}")
    logger.info(f"  Negative: {int((data['labels_binary'] == 0).sum())}")

    # Results storage
    results_path = OUTPUT_DIR / "v5_screen_results.json"
    all_results = {}

    # Load existing partial results if any
    if results_path.exists():
        with open(results_path) as f:
            all_results = json.load(f)
        logger.info(f"\nLoaded {len(all_results)} existing results from {results_path}")

    def save_results():
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        logger.info(f"  Saved results to {results_path}")

    # --- 4 Neural combinations ---
    combinations = [
        ("A6_Conv2DBP", Conv2DBP, "T1_baseline", run_baseline),
        ("A6_Conv2DBP", Conv2DBP, "T6_pretext", run_pretext),
        ("A8_HierarchicalAttention", HierarchicalAttention, "T1_baseline", run_baseline),
        ("A8_HierarchicalAttention", HierarchicalAttention, "T6_pretext", run_pretext),
    ]

    for arch_name, arch_cls, training_name, train_fn in combinations:
        combo_key = f"{arch_name}_{training_name}"

        if combo_key in all_results:
            logger.info(f"\nSkipping {combo_key} (already computed)")
            continue

        try:
            n_params = sum(p.numel() for p in arch_cls().parameters())
            logger.info(f"\n{combo_key}: {n_params:,} params")

            result = train_fn(arch_cls, arch_name, data)
            all_results[combo_key] = result
            save_results()

        except Exception as e:
            logger.error(f"ERROR in {combo_key}: {e}")
            logger.error(traceback.format_exc())
            all_results[combo_key] = {"error": str(e)}
            save_results()

    # --- XGB baseline ---
    if "XGB_40d" not in all_results:
        try:
            xgb_result = run_xgb_baseline(data)
            all_results["XGB_40d"] = xgb_result
            save_results()
        except Exception as e:
            logger.error(f"ERROR in XGB: {e}")
            logger.error(traceback.format_exc())
            all_results["XGB_40d"] = {"error": str(e)}
            save_results()
    else:
        logger.info("\nSkipping XGB (already computed)")

    # --- Comparison table ---
    v3_results = load_v3_results()

    # Build v5 summary for comparison
    v5_summary = {}
    for combo_key, entry in all_results.items():
        if "error" in entry:
            continue
        v5_summary[combo_key] = entry

    print_comparison_table(v5_summary, v3_results)

    logger.info(f"\nAll results saved to: {results_path}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
