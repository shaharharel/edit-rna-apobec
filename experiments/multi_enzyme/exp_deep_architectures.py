#!/usr/bin/env python
"""Deep learning architecture comparison for APOBEC RNA editing prediction.

Trains and evaluates 5 deep learning architectures on the multi-enzyme v3 dataset:
  1. Conv2D BP Matrix + Sequence + Hand Features
  2. Dual-Path Residual (Hand Features + Sequence)
  3. Structure-Biased Transformer
  4. EditRNA Fixed (simplified existing approach)
  5. FiLM-Conditioned Sequence Model

All models share the same data pipeline, folds, and evaluation.
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

from data.apobec_feature_extraction import build_hand_features, LOOP_FEATURE_COLS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "multi_enzyme" / "outputs" / "deep_architectures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("mps")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else torch.device("cpu")
)
logger.info(f"Using device: {DEVICE}")

# Training hyperparameters
N_FOLDS = 5
N_EPOCHS = 50
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
ENZYME_LOSS_WEIGHT = 0.3
SEED = 42

ENZYME_CLASSES = ["A3A", "A3B", "A3G", "A3A_A3G", "Neither", "Unknown"]
ENZYME_TO_IDX = {e: i for i, e in enumerate(ENZYME_CLASSES)}

CENTER = 100  # Edit site position in 201-nt window


# ---------------------------------------------------------------------------
# ViennaRNA folding (parallel)
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
        # Extract 41x41 submatrix around edit site
        sub = bpp[80:121, 80:121].copy()
        # Mask diagonal |i-j| < 3
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


def load_all_data() -> Dict:
    """Load and preprocess all data for training.

    Returns dict with numpy arrays, keeping memory-heavy items (bp_indicator)
    computed lazily per-batch in the Dataset class.
    """
    logger.info("Loading splits...")
    splits_path = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "splits_multi_enzyme_v3_with_negatives.csv"
    df = pd.read_csv(splits_path)
    logger.info(f"  Loaded {len(df)} sites")

    logger.info("Loading sequences...")
    seq_path = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "multi_enzyme_sequences_v3_with_negatives.json"
    with open(seq_path) as f:
        sequences = json.load(f)
    logger.info(f"  Loaded {len(sequences)} sequences")

    logger.info("Loading loop positions...")
    loop_path = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "loop_position_per_site_v3.csv"
    loop_df = pd.read_csv(loop_path)
    db_lookup = {}
    for _, row in loop_df.iterrows():
        if pd.notna(row.get("dot_bracket")):
            db_lookup[str(row["site_id"])] = row["dot_bracket"]
    loop_df = loop_df.drop_duplicates(subset=["site_id"]).set_index("site_id")
    logger.info(f"  {len(db_lookup)} sites have pre-computed dot-bracket")

    logger.info("Loading structure cache...")
    struct_path = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "structure_cache_multi_enzyme_v3.npz"
    struct_data = np.load(struct_path, allow_pickle=True)
    struct_ids = list(struct_data["site_ids"])
    struct_deltas = struct_data["delta_features"]
    structure_delta = {sid: struct_deltas[i] for i, sid in enumerate(struct_ids)}
    logger.info(f"  Loaded {len(structure_delta)} structure delta features")

    # Filter to sites with sequences
    site_ids = df["site_id"].tolist()
    valid_mask = [sid in sequences for sid in site_ids]
    df = df[valid_mask].reset_index(drop=True)
    site_ids = df["site_id"].tolist()
    n = len(site_ids)
    logger.info(f"  {n} sites with sequences")

    # Build hand features (40-dim)
    logger.info("Building 40-dim hand features...")
    hand_features = build_hand_features(site_ids, sequences, structure_delta, loop_df)
    logger.info(f"  Hand features shape: {hand_features.shape}")

    # One-hot encode sequences (4 x 201)
    logger.info("One-hot encoding sequences...")
    base_map = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3}
    onehot_seqs = np.zeros((n, 4, 201), dtype=np.float32)
    for i, sid in enumerate(site_ids):
        seq = sequences[sid].upper().replace("T", "U")
        for j, base in enumerate(seq[:201]):
            if base in base_map:
                onehot_seqs[i, base_map[base], j] = 1.0

    # Fold all sequences to get dot-brackets and BP probability submatrices
    logger.info("Computing ViennaRNA structures and 41x41 BP submatrices...")
    # Prepare work items (need folding for all sites to get BPP)
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
            if sid not in db_lookup:
                dot_brackets[sid] = structure
            else:
                dot_brackets[sid] = db_lookup[sid]
            idx = sid_to_idx[sid]
            bp_submatrices[idx] = sub
            done += 1
            if done % 3000 == 0:
                elapsed = time.time() - t0
                logger.info(f"    Folded {done}/{n} ({elapsed:.0f}s)")

    elapsed = time.time() - t0
    logger.info(f"  Folding complete: {done} sequences in {elapsed:.0f}s")

    # Encode dot-bracket structures (3 channels: open, close, unpaired)
    logger.info("Encoding dot-bracket structures...")
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

    # Labels
    labels_binary = df["is_edited"].values.astype(np.float32)
    labels_enzyme = np.array([ENZYME_TO_IDX.get(e, 5) for e in df["enzyme"].values], dtype=np.int64)

    # Edited sequence one-hot (C->U at center)
    logger.info("Creating edited sequence one-hots...")
    onehot_edited = onehot_seqs.copy()
    onehot_edited[:, 1, CENTER] = 0.0
    onehot_edited[:, 3, CENTER] = 1.0

    logger.info("Data loading complete.")
    return {
        "site_ids": site_ids,
        "df": df,
        "hand_features": hand_features,          # [N, 40]
        "onehot_seqs": onehot_seqs,              # [N, 4, 201]
        "onehot_edited": onehot_edited,          # [N, 4, 201]
        "struct_onehot": struct_onehot,           # [N, 3, 201]
        "bp_submatrices": bp_submatrices,         # [N, 41, 41]
        "dot_bracket_list": dot_bracket_list,     # list of str
        "labels_binary": labels_binary,           # [N]
        "labels_enzyme": labels_enzyme,           # [N]
    }


# ---------------------------------------------------------------------------
# Custom Dataset with lazy bp_indicator for Transformer
# ---------------------------------------------------------------------------


class EditingDataset(Dataset):
    """Dataset that lazily computes bp_indicator matrices per sample."""

    def __init__(self, indices, data):
        self.indices = indices
        self.data = data

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        result = {
            "onehot_seq": torch.from_numpy(self.data["onehot_seqs"][i]),
            "onehot_edited": torch.from_numpy(self.data["onehot_edited"][i]),
            "hand_feat": torch.from_numpy(self.data["hand_features"][i]),
            "struct_onehot": torch.from_numpy(self.data["struct_onehot"][i]),
            "bp_submatrix": torch.from_numpy(self.data["bp_submatrices"][i]).unsqueeze(0),
            "label_binary": torch.tensor(self.data["labels_binary"][i]),
            "label_enzyme": torch.tensor(self.data["labels_enzyme"][i]),
        }
        # Compute bp_indicator lazily (201x201 only when needed)
        db = self.data["dot_bracket_list"][i]
        bp = _match_base_pairs(db)
        result["bp_indicator"] = torch.from_numpy(bp)
        return result


# ---------------------------------------------------------------------------
# Model 1: Conv2D BP Matrix + Sequence + Hand Features
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
# Model 2: Dual-Path Residual
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
# Model 3: Structure-Biased Transformer
# ---------------------------------------------------------------------------


class StructureBiasedTransformer(nn.Module):
    """Transformer with structure-biased attention via explicit BP context injection."""

    def __init__(self, d_model=128, n_heads=4, n_layers=4, d_ff=256):
        super().__init__()
        self.d_model = d_model
        # Token embedding: seq(4) + struct(3) + edit_indicator(1) = 8
        self.token_embed = nn.Linear(8, d_model)

        # Structure bias: project bp_indicator row into d_model-dim context
        self.struct_proj = nn.Linear(201, d_model)
        self.struct_scale = nn.Parameter(torch.tensor(0.1))

        # Positional encoding (sinusoidal)
        pe = torch.zeros(201, d_model)
        position = torch.arange(0, 201, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, 201, d_model]

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=0.1, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.post_fc = nn.Sequential(nn.Linear(d_model + 40, 64), nn.GELU())
        self.binary_head = nn.Linear(64, 1)
        self.enzyme_head = nn.Linear(64, 6)

    def forward(self, batch):
        onehot_seq = batch["onehot_seq"]      # [B, 4, 201]
        struct_onehot = batch["struct_onehot"] # [B, 3, 201]
        bp_indicator = batch["bp_indicator"]   # [B, 201, 201]
        hand_feat = batch["hand_feat"]         # [B, 40]
        B = onehot_seq.shape[0]

        # Build token embeddings
        edit_ind = torch.zeros(B, 1, 201, device=onehot_seq.device)
        edit_ind[:, 0, CENTER] = 1.0
        tokens = torch.cat([onehot_seq, struct_onehot, edit_ind], dim=1)  # [B, 8, 201]
        tokens = tokens.permute(0, 2, 1)  # [B, 201, 8]
        tokens = self.token_embed(tokens)  # [B, 201, d_model]

        # Add structure context: each position gets a weighted sum of its paired partners
        struct_ctx = self.struct_proj(bp_indicator)  # [B, 201, d_model]
        tokens = tokens + self.struct_scale * struct_ctx

        # Add positional encoding
        tokens = tokens + self.pe

        # Transformer
        out = self.transformer(tokens)  # [B, 201, d_model]
        site_repr = out[:, CENTER, :]  # [B, d_model]

        combined = torch.cat([site_repr, hand_feat], dim=-1)
        combined = self.post_fc(combined)
        return self.binary_head(combined).squeeze(-1), self.enzyme_head(combined)


# ---------------------------------------------------------------------------
# Model 4: EditRNA Fixed (simplified)
# ---------------------------------------------------------------------------


class EditRNAFixed(nn.Module):
    """Simplified EditRNA: sequence + edit delta + structure features."""

    def __init__(self):
        super().__init__()
        self.seq_encoder = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=7, padding=3), nn.ReLU(), nn.BatchNorm1d(32), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.ReLU(), nn.BatchNorm1d(64), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm1d(128),
            nn.AdaptiveAvgPool1d(1),
        )
        self.delta_encoder = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=7, padding=3), nn.ReLU(), nn.BatchNorm1d(32), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.ReLU(), nn.BatchNorm1d(64), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm1d(128),
            nn.AdaptiveAvgPool1d(1),
        )
        self.struct_fc = nn.Sequential(nn.Linear(16, 64), nn.ReLU())
        self.fusion = nn.Sequential(
            nn.Linear(320, 128), nn.GELU(), nn.Dropout(0.2), nn.Linear(128, 64),
        )
        self.binary_head = nn.Linear(64, 1)
        self.enzyme_head = nn.Linear(64, 6)

    def forward(self, batch):
        seq_out = self.seq_encoder(batch["onehot_seq"]).squeeze(-1)
        delta = batch["onehot_edited"] - batch["onehot_seq"]
        delta_out = self.delta_encoder(delta).squeeze(-1)
        struct_feats = batch["hand_feat"][:, 24:]  # struct_delta(7) + loop(9) = 16
        struct_out = self.struct_fc(struct_feats)
        combined = torch.cat([seq_out, delta_out, struct_out], dim=-1)
        fused = self.fusion(combined)
        return self.binary_head(fused).squeeze(-1), self.enzyme_head(fused)


# ---------------------------------------------------------------------------
# Model 5: FiLM-Conditioned Sequence Model
# ---------------------------------------------------------------------------


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation."""

    def __init__(self, cond_dim, feat_dim):
        super().__init__()
        self.gamma_fc = nn.Linear(cond_dim, feat_dim)
        self.beta_fc = nn.Linear(cond_dim, feat_dim)
        nn.init.ones_(self.gamma_fc.bias)
        nn.init.zeros_(self.gamma_fc.weight)
        nn.init.zeros_(self.beta_fc.bias)
        nn.init.zeros_(self.beta_fc.weight)

    def forward(self, x, cond):
        gamma = self.gamma_fc(cond).unsqueeze(-1)
        beta = self.beta_fc(cond).unsqueeze(-1)
        return gamma * x + beta


class FiLMConditionedModel(nn.Module):
    """Sequence model conditioned on hand features via FiLM."""

    def __init__(self):
        super().__init__()
        self.cond_fc = nn.Sequential(nn.Linear(40, 128), nn.GELU())
        self.conv1 = nn.Conv1d(4, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.film1 = FiLMLayer(128, 64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.film2 = FiLMLayer(128, 128)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.local_fc = nn.Linear(128, 128)
        self.fusion = nn.Sequential(nn.Linear(256, 64), nn.GELU(), nn.Dropout(0.2))
        self.binary_head = nn.Linear(64, 1)
        self.enzyme_head = nn.Linear(64, 6)

    def forward(self, batch):
        cond = self.cond_fc(batch["hand_feat"])
        x = F.relu(self.bn1(self.conv1(batch["onehot_seq"])))
        x = self.film1(x, cond)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.film2(x, cond)
        global_feat = self.pool(x).squeeze(-1)
        local_feat = self.local_fc(x[:, :, CENTER])
        fused = self.fusion(torch.cat([global_feat, local_feat], dim=-1))
        return self.binary_head(fused).squeeze(-1), self.enzyme_head(fused)


# ---------------------------------------------------------------------------
# Training and Evaluation
# ---------------------------------------------------------------------------


def collate_fn(batch_list):
    """Custom collate that handles variable-size bp_indicator."""
    result = {}
    for key in batch_list[0]:
        result[key] = torch.stack([b[key] for b in batch_list])
    return result


def train_one_epoch(model, loader, optimizer):
    """Train one epoch, returns average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        optimizer.zero_grad()
        binary_logit, enzyme_logit = model(batch)
        loss_binary = F.binary_cross_entropy_with_logits(binary_logit, batch["label_binary"])
        loss_enzyme = F.cross_entropy(enzyme_logit, batch["label_enzyme"])
        loss = loss_binary + ENZYME_LOSS_WEIGHT * loss_enzyme
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader):
    """Evaluate model on a DataLoader, return metrics dict."""
    model.eval()
    all_probs, all_enzyme, all_lbl_b, all_lbl_e = [], [], [], []

    for batch in loader:
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


def create_model(model_name):
    """Create a fresh model instance by name."""
    if model_name == "Conv2D_BP":
        return Conv2DBPModel()
    elif model_name == "DualPath_Residual":
        return DualPathResidual()
    elif model_name == "StructBiased_Transformer":
        return StructureBiasedTransformer()
    elif model_name == "EditRNA_Fixed":
        return EditRNAFixed()
    elif model_name == "FiLM_Conditioned":
        return FiLMConditionedModel()
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_model(model_name, data, train_idx, val_idx, fold):
    """Train a model for one fold, return model, history, best AUROC."""
    model = create_model(model_name).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if fold == 0:
        logger.info(f"  Model parameters: {n_params:,}")

    # Use num_workers=0 on MPS to avoid multiprocessing issues
    n_loader_workers = 0 if DEVICE.type == "mps" else 4

    train_ds = EditingDataset(train_idx, data)
    val_ds = EditingDataset(val_idx, data)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=n_loader_workers,
                              pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=collate_fn, num_workers=n_loader_workers,
                            pin_memory=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

    history = {"train_loss": [], "val_auroc": [], "val_enzyme_acc": []}
    best_auroc = 0.0
    best_state = None

    for epoch in range(N_EPOCHS):
        loss = train_one_epoch(model, train_loader, optimizer)
        scheduler.step()
        metrics = evaluate(model, val_loader)

        history["train_loss"].append(loss)
        history["val_auroc"].append(metrics["auroc"])
        history["val_enzyme_acc"].append(metrics["enzyme_acc"])

        if metrics["auroc"] > best_auroc:
            best_auroc = metrics["auroc"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            logger.info(
                f"  [{model_name}] Fold {fold} Epoch {epoch+1}/{N_EPOCHS}: "
                f"loss={loss:.4f}, AUROC={metrics['auroc']:.4f}, "
                f"enzyme_acc={metrics['enzyme_acc']:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history, best_auroc


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_comparison(results, output_dir):
    """Plot AUROC comparison bar chart."""
    model_names = list(results.keys())
    mean_aurocs = [results[m]["mean_auroc"] for m in model_names]
    std_aurocs = [results[m]["std_auroc"] for m in model_names]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(model_names))
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]
    bars = ax.bar(x, mean_aurocs, yerr=std_aurocs, capsize=5,
                  color=colors[:len(model_names)], edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in model_names], fontsize=10)
    ax.set_ylabel("Binary AUROC", fontsize=12)
    ax.set_title("Deep Architecture Comparison (5-Fold CV AUROC)", fontsize=14)
    ax.set_ylim(0.5, 1.0)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random")
    for bar, mean, std in zip(bars, mean_aurocs, std_aurocs):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + std + 0.005,
                f"{mean:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "architecture_comparison.png", dpi=150)
    plt.close()


def plot_training_curves(all_histories, output_dir):
    """Plot per-model training curves."""
    for model_name, fold_histories in all_histories.items():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        for fold_idx, hist in enumerate(fold_histories):
            ax1.plot(hist["train_loss"], alpha=0.6, label=f"Fold {fold_idx}")
            ax2.plot(hist["val_auroc"], alpha=0.6, label=f"Fold {fold_idx}")
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("Training Loss")
        ax1.set_title(f"{model_name} - Training Loss"); ax1.legend(fontsize=8)
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("Validation AUROC")
        ax2.set_title(f"{model_name} - Validation AUROC"); ax2.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(output_dir / f"training_curves_{model_name}.png", dpi=120)
        plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    data = load_all_data()
    n = len(data["site_ids"])
    logger.info(f"Total sites: {n}")

    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = list(skf.split(np.arange(n), data["labels_binary"]))

    model_names = [
        "Conv2D_BP",
        "DualPath_Residual",
        "StructBiased_Transformer",
        "EditRNA_Fixed",
        "FiLM_Conditioned",
    ]

    results = {}
    all_histories = {}

    for model_name in model_names:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {model_name}")
        logger.info(f"{'='*60}")

        fold_aurocs = []
        fold_enzyme_accs = []
        fold_histories = []

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            logger.info(f"\n--- {model_name} Fold {fold_idx} ---")
            t0 = time.time()

            model, history, best_auroc = train_model(
                model_name, data, list(train_idx), list(val_idx), fold_idx,
            )

            # Final evaluation with best model
            n_loader_workers = 0 if DEVICE.type == "mps" else 4
            val_ds = EditingDataset(list(val_idx), data)
            val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                    collate_fn=collate_fn, num_workers=n_loader_workers)
            final_metrics = evaluate(model, val_loader)

            elapsed = time.time() - t0
            logger.info(
                f"  Fold {fold_idx} done in {elapsed:.0f}s: "
                f"AUROC={final_metrics['auroc']:.4f}, "
                f"enzyme_acc={final_metrics['enzyme_acc']:.4f}"
            )

            fold_aurocs.append(final_metrics["auroc"])
            fold_enzyme_accs.append(final_metrics["enzyme_acc"])
            fold_histories.append(history)

            torch.save(model.state_dict(), OUTPUT_DIR / f"{model_name}_fold{fold_idx}.pt")
            # Free GPU memory
            del model
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

        results[model_name] = {
            "fold_aurocs": [float(x) for x in fold_aurocs],
            "fold_enzyme_accs": [float(x) for x in fold_enzyme_accs],
            "mean_auroc": float(np.mean(fold_aurocs)),
            "std_auroc": float(np.std(fold_aurocs)),
            "mean_enzyme_acc": float(np.mean(fold_enzyme_accs)),
            "std_enzyme_acc": float(np.std(fold_enzyme_accs)),
            "n_params": sum(p.numel() for p in create_model(model_name).parameters()),
        }
        all_histories[model_name] = fold_histories

        logger.info(
            f"\n{model_name} SUMMARY: "
            f"AUROC={results[model_name]['mean_auroc']:.4f} +/- {results[model_name]['std_auroc']:.4f}, "
            f"enzyme_acc={results[model_name]['mean_enzyme_acc']:.4f} +/- {results[model_name]['std_enzyme_acc']:.4f}"
        )

    # Save results
    logger.info(f"\n{'='*60}")
    logger.info("FINAL RESULTS")
    logger.info(f"{'='*60}")

    with open(OUTPUT_DIR / "architecture_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to {OUTPUT_DIR / 'architecture_comparison.json'}")

    logger.info("\n" + "-" * 80)
    logger.info(f"{'Model':<30} {'AUROC':>16} {'Enzyme Acc':>16} {'Params':>12}")
    logger.info("-" * 80)
    for m in model_names:
        r = results[m]
        logger.info(
            f"{m:<30} {r['mean_auroc']:.4f}+/-{r['std_auroc']:.4f}   "
            f"{r['mean_enzyme_acc']:.4f}+/-{r['std_enzyme_acc']:.4f}   "
            f"{r['n_params']:>10,}"
        )
    logger.info("-" * 80)

    plot_comparison(results, OUTPUT_DIR)
    plot_training_curves(all_histories, OUTPUT_DIR)
    logger.info(f"\nAll outputs saved to {OUTPUT_DIR}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
