#!/usr/bin/env python
"""Comprehensive cross-dataset generalization matrix experiment.

For each model architecture (ordered by complexity), trains on each dataset
and evaluates on ALL other datasets. Outputs an NxN AUROC matrix per model.

Models (ordered by complexity):
  1. StructureOnly    - XGBoost on 16-dim structure features (7 delta + 9 loop)
  2. GB_HandFeatures  - XGBoost on ~40 hand-crafted features (motif + structure + loop)
  3. GB_AllFeatures   - XGBoost on hand-crafted + 640-dim embedding delta
  4. PooledMLP        - MLP on pooled embedding (original only, no edit signal)
  5. SubtractionMLP   - MLP on pooled embedding delta (edited - original)
  6. EditRNA_A3A      - Full gated fusion model (pooled-only mode)

Training datasets (rows): Levanon, Asaoka, Alqassim, Baysal, All Combined
Test datasets (columns):  Levanon, Asaoka, Alqassim, Baysal

Data approach (matches exp_dataset_matrix.py):
  - Positives from each dataset_source in splits_expanded.csv
  - Negatives: tier2_negative + tier3_negative rows (shared across all configs)
  - Training: positives from train_dataset (train+val splits) + all negatives (train+val splits)
  - Testing: positives from test_dataset (test split) + all negatives (test split)
  - Sharma (6 sites) is skipped as training source

Usage:
    conda activate quris
    python experiments/apobec3a/exp_cross_dataset_full.py
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"  # XGBoost safety on macOS

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
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "apobec"))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
EMB_DIR = PROJECT_ROOT / "data" / "processed" / "embeddings"
SPLITS_CSV = PROJECT_ROOT / "data" / "processed" / "splits_expanded.csv"
SPLITS_A3A_CSV = PROJECT_ROOT / "data" / "processed" / "splits_expanded_a3a.csv"
SEQUENCES_JSON = PROJECT_ROOT / "data" / "processed" / "site_sequences.json"
STRUCTURE_CACHE = EMB_DIR / "vienna_structure_cache.npz"
LOOP_POSITION_CSV = (
    PROJECT_ROOT / "experiments" / "apobec" / "outputs" / "loop_position"
    / "loop_position_per_site.csv"
)
OUTPUT_DIR = (
    PROJECT_ROOT / "experiments" / "apobec" / "outputs" / "cross_dataset_full"
)

DATASET_SOURCES = ["advisor_c2t", "asaoka_2019", "alqassim_2021"]
DATASET_LABELS = {
    "advisor_c2t": "Levanon",
    "asaoka_2019": "Asaoka",
    "alqassim_2021": "Alqassim",
}
NEGATIVE_SOURCES = ["tier2_negative", "tier3_negative"]

MODEL_ORDER = [
    "StructureOnly",
    "GB_HandFeatures",
    "GB_AllFeatures",
    "PooledMLP",
    "SubtractionMLP",
    "EditRNA_A3A",
]

SEED = 42


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.75):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        logits = logits.squeeze(-1)
        probs = torch.sigmoid(logits)
        targets = targets.float()
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        return (focal_weight * bce).mean()


# ---------------------------------------------------------------------------
# Embedding Dataset for DL models
# ---------------------------------------------------------------------------

class EmbeddingDataset(Dataset):
    """Dataset serving pre-computed embeddings and structure delta."""

    def __init__(
        self,
        site_ids: List[str],
        labels: np.ndarray,
        pooled_orig: Dict[str, torch.Tensor],
        pooled_edited: Dict[str, torch.Tensor],
        structure_delta: Optional[Dict[str, np.ndarray]] = None,
    ):
        self.site_ids = site_ids
        self.labels = labels
        self.pooled_orig = pooled_orig
        self.pooled_edited = pooled_edited
        self.structure_delta = structure_delta

    def __len__(self):
        return len(self.site_ids)

    def __getitem__(self, idx):
        sid = self.site_ids[idx]
        item = {
            "site_id": sid,
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
            "pooled_orig": self.pooled_orig[sid],
            "pooled_edited": self.pooled_edited[sid],
        }
        if self.structure_delta is not None and sid in self.structure_delta:
            item["structure_delta"] = torch.tensor(
                self.structure_delta[sid], dtype=torch.float32
            )
        else:
            item["structure_delta"] = torch.zeros(7, dtype=torch.float32)
        return item


def embedding_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    return {
        "site_ids": [b["site_id"] for b in batch],
        "labels": torch.stack([b["label"] for b in batch]),
        "pooled_orig": torch.stack([b["pooled_orig"] for b in batch]),
        "pooled_edited": torch.stack([b["pooled_edited"] for b in batch]),
        "structure_delta": torch.stack([b["structure_delta"] for b in batch]),
    }


# ---------------------------------------------------------------------------
# DL training / evaluation helpers
# ---------------------------------------------------------------------------

def _forward_model(model, batch, model_name):
    """Run forward pass for a given model type."""
    if model_name == "EditRNA_A3A":
        B = batch["labels"].shape[0]
        device = batch["labels"].device
        editrna_batch = {
            "sequences": ["N" * 201] * B,
            "site_ids": batch["site_ids"],
            "edit_pos": torch.full((B,), 100, dtype=torch.long, device=device),
            "flanking_context": torch.zeros(B, dtype=torch.long, device=device),
            "concordance_features": torch.zeros(B, 5, device=device),
            "structure_delta": batch["structure_delta"],
        }
        output = model(editrna_batch)
        return output["predictions"]["binary_logit"]
    else:
        output = model(batch)
        return output["binary_logit"]


def train_dl_model(
    model: nn.Module,
    model_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    pooled_orig: Dict,
    pooled_edited: Dict,
    structure_delta: Optional[Dict],
    epochs: int = 50,
    lr: float = 1e-3,
    patience: int = 10,
    batch_size: int = 64,
) -> float:
    """Train a DL model on train_df, evaluate on test_df. Returns test AUROC."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    train_split = train_df[train_df["split"] == "train"]
    val_split = train_df[train_df["split"] == "val"]

    if len(train_split) < 10 or len(val_split) < 5:
        return float("nan")

    def make_loader(df, shuffle=False):
        ds = EmbeddingDataset(
            df["site_id"].tolist(),
            df["label"].values.astype(np.float32),
            pooled_orig, pooled_edited, structure_delta,
        )
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=0, drop_last=False, collate_fn=embedding_collate_fn,
        )

    train_loader = make_loader(train_split, shuffle=True)
    val_loader = make_loader(val_split)
    test_loader = make_loader(test_df)

    device = torch.device("cpu")
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)
    loss_fn = FocalLoss()

    best_auroc = -1.0
    patience_counter = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            optimizer.zero_grad()
            logits = _forward_model(model, batch, model_name)
            loss = loss_fn(logits, batch["labels"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # Validate
        model.eval()
        all_y, all_s = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                logits = _forward_model(model, batch, model_name)
                probs = torch.sigmoid(logits.squeeze(-1)).numpy()
                all_y.append(batch["labels"].numpy())
                all_s.append(probs)

        y_true = np.concatenate(all_y)
        y_score = np.concatenate(all_s)
        if len(np.unique(y_true)) < 2:
            val_auroc = 0.0
        else:
            val_auroc = roc_auc_score(y_true, y_score)

        if val_auroc > best_auroc + 1e-4:
            best_auroc = val_auroc
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break

    if best_state:
        model.load_state_dict(best_state)

    # Test
    model.eval()
    all_y, all_s = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            logits = _forward_model(model, batch, model_name)
            probs = torch.sigmoid(logits.squeeze(-1)).numpy()
            all_y.append(batch["labels"].numpy())
            all_s.append(probs)

    y_true = np.concatenate(all_y)
    y_score = np.concatenate(all_s)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


# ---------------------------------------------------------------------------
# GB feature extraction (reuse from train_gradient_boosting.py)
# ---------------------------------------------------------------------------

def load_structure_delta_dict() -> Dict[str, np.ndarray]:
    """Load structure delta dict from vienna_structure_cache.npz."""
    if not STRUCTURE_CACHE.exists():
        logger.warning("Structure cache not found: %s", STRUCTURE_CACHE)
        return {}
    data = np.load(STRUCTURE_CACHE, allow_pickle=True)
    site_ids = data["site_ids"]
    delta_features = data["delta_features"]
    return {str(sid): delta_features[i] for i, sid in enumerate(site_ids)}


def load_sequences() -> Dict[str, str]:
    """Load site sequences from JSON."""
    with open(SEQUENCES_JSON) as f:
        return json.load(f)


def load_loop_position_df() -> pd.DataFrame:
    """Load loop position features."""
    if not LOOP_POSITION_CSV.exists():
        logger.warning("Loop position CSV not found: %s", LOOP_POSITION_CSV)
        return pd.DataFrame(columns=["site_id"])
    return pd.read_csv(LOOP_POSITION_CSV)


def extract_motif_features_for_ids(
    site_ids: List[str], sequences: Dict[str, str]
) -> np.ndarray:
    """Extract motif features for a list of site IDs. Returns (N, n_motif_features)."""
    top_motifs = ["UCG", "UCA", "UCC", "UCU", "ACA", "GCA", "CCA",
                  "ACG", "GCG", "CCG", "ACU", "GCU", "CCU", "ACC", "GCC"]
    # Deduplicate while preserving order
    seen = set()
    unique_motifs = []
    for m in top_motifs:
        if m not in seen:
            seen.add(m)
            unique_motifs.append(m)
    top_motifs = unique_motifs

    n_motif_cols = 1 + 4 + 4 + len(top_motifs)  # TC + upstream + downstream + trinuc
    X = np.zeros((len(site_ids), n_motif_cols), dtype=np.float32)

    for i, sid in enumerate(site_ids):
        if sid not in sequences:
            continue
        seq = sequences[sid]
        if len(seq) < 102:
            continue
        trinuc = seq[99:102]
        col = 0
        # TC motif
        X[i, col] = 1.0 if (seq[99] == "U" and seq[100] == "C") else 0.0
        col += 1
        # Upstream nucleotide
        for nuc in ["A", "C", "G", "U"]:
            X[i, col] = 1.0 if seq[99] == nuc else 0.0
            col += 1
        # Downstream nucleotide
        for nuc in ["A", "C", "G", "U"]:
            X[i, col] = 1.0 if seq[101] == nuc else 0.0
            col += 1
        # Trinucleotide one-hot
        for motif in top_motifs:
            X[i, col] = 1.0 if trinuc == motif else 0.0
            col += 1

    return X


def extract_structure_delta_for_ids(
    site_ids: List[str], struct_dict: Dict[str, np.ndarray]
) -> np.ndarray:
    """Extract 7-dim structure delta features for a list of site IDs."""
    X = np.zeros((len(site_ids), 7), dtype=np.float32)
    for i, sid in enumerate(site_ids):
        if sid in struct_dict:
            X[i] = struct_dict[sid]
    return X


def extract_loop_features_for_ids(
    site_ids: List[str], loop_df: pd.DataFrame
) -> np.ndarray:
    """Extract loop position features for a list of site IDs."""
    loop_feature_cols = [
        "is_unpaired", "loop_size", "dist_to_junction", "dist_to_apex",
        "relative_loop_position", "left_stem_length", "right_stem_length",
        "max_adjacent_stem_length", "local_unpaired_fraction",
    ]
    n_cols = len(loop_feature_cols)
    X = np.zeros((len(site_ids), n_cols), dtype=np.float32)

    if loop_df.empty or "site_id" not in loop_df.columns:
        return X

    loop_dict = {}
    available_cols = [c for c in loop_feature_cols if c in loop_df.columns]
    for _, row in loop_df.iterrows():
        sid = str(row["site_id"])
        vals = np.zeros(n_cols, dtype=np.float32)
        for j, col in enumerate(loop_feature_cols):
            if col in available_cols:
                v = row.get(col, 0.0)
                vals[j] = float(v) if pd.notna(v) else 0.0
        loop_dict[sid] = vals

    for i, sid in enumerate(site_ids):
        if sid in loop_dict:
            X[i] = loop_dict[sid]

    return X


def extract_hand_features(
    site_ids: List[str],
    sequences: Dict[str, str],
    struct_dict: Dict[str, np.ndarray],
    loop_df: pd.DataFrame,
) -> np.ndarray:
    """Extract all hand-crafted features (motif + structure + loop)."""
    motif = extract_motif_features_for_ids(site_ids, sequences)
    struct = extract_structure_delta_for_ids(site_ids, struct_dict)
    loop = extract_loop_features_for_ids(site_ids, loop_df)
    X = np.hstack([motif, struct, loop])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def extract_embedding_delta_for_ids(
    site_ids: List[str],
    pooled_orig: Dict[str, torch.Tensor],
    pooled_edited: Dict[str, torch.Tensor],
) -> np.ndarray:
    """Extract 640-dim embedding delta (edited - original)."""
    X = np.zeros((len(site_ids), 640), dtype=np.float32)
    for i, sid in enumerate(site_ids):
        if sid in pooled_orig and sid in pooled_edited:
            X[i] = (pooled_edited[sid] - pooled_orig[sid]).numpy()
    return X


def extract_all_features(
    site_ids: List[str],
    sequences: Dict[str, str],
    struct_dict: Dict[str, np.ndarray],
    loop_df: pd.DataFrame,
    pooled_orig: Dict[str, torch.Tensor],
    pooled_edited: Dict[str, torch.Tensor],
) -> np.ndarray:
    """Extract hand-crafted + 640-dim embedding delta features."""
    hand = extract_hand_features(site_ids, sequences, struct_dict, loop_df)
    emb_delta = extract_embedding_delta_for_ids(site_ids, pooled_orig, pooled_edited)
    X = np.hstack([hand, emb_delta])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


# ---------------------------------------------------------------------------
# GB training
# ---------------------------------------------------------------------------

def train_gb_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    """Train XGBoost classifier and return test AUROC."""
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return float("nan")

    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    scale_pos_weight = n_neg / max(n_pos, 1)

    try:
        import xgboost as xgb
        clf = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            scale_pos_weight=scale_pos_weight,
            random_state=SEED,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=1,
            verbosity=0,
        )
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        clf = GradientBoostingClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=SEED,
        )

    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)[:, 1]
    return float(roc_auc_score(y_test, probs))


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_dl_model(
    model_name: str,
    pooled_orig: Dict[str, torch.Tensor],
    pooled_edited: Dict[str, torch.Tensor],
) -> nn.Module:
    """Build a DL model by name."""
    if model_name == "StructureOnly":
        from models.baselines.structure_only import StructureOnlyBaseline
        return StructureOnlyBaseline()

    elif model_name == "PooledMLP":
        from models.baselines.pooled_mlp import PooledMLPBaseline
        return PooledMLPBaseline()

    elif model_name == "SubtractionMLP":
        from models.baselines.subtraction_mlp import SubtractionMLPBaseline
        return SubtractionMLPBaseline()

    elif model_name == "EditRNA_A3A":
        from models.editrna_a3a import EditRNA_A3A, EditRNAConfig
        from models.encoders import CachedRNAEncoder

        cached_encoder = CachedRNAEncoder(
            tokens_cache=None,
            pooled_cache=pooled_orig,
            tokens_edited_cache=None,
            pooled_edited_cache=pooled_edited,
            d_model=640,
        )
        model_config = EditRNAConfig(
            primary_encoder="cached",
            d_model=640,
            pooled_only=True,
            learning_rate=1e-3,
        )
        return EditRNA_A3A(config=model_config, primary_encoder=cached_encoder)

    else:
        raise ValueError(f"Unknown DL model: {model_name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # ==================================================================
    # Load all data
    # ==================================================================
    logger.info("Loading data...")

    # Use A3A-only splits (no Baysal — different enzyme)
    splits_df = pd.read_csv(SPLITS_A3A_CSV)
    logger.info("Loaded A3A splits: %d total sites", len(splits_df))

    # Pooled embeddings
    pooled_orig = torch.load(EMB_DIR / "rnafm_pooled.pt", weights_only=False)
    pooled_edited = torch.load(EMB_DIR / "rnafm_pooled_edited.pt", weights_only=False)

    # Filter to sites with embeddings
    available_ids = set(pooled_orig.keys()) & set(pooled_edited.keys())
    splits_df = splits_df[splits_df["site_id"].isin(available_ids)].copy()
    logger.info("Splits after embedding filter: %d sites", len(splits_df))

    # Structure delta
    structure_delta_dict = load_structure_delta_dict()
    logger.info("Structure delta dict: %d sites", len(structure_delta_dict))

    # Sequences & loop features (for GB models)
    sequences = load_sequences()
    loop_df = load_loop_position_df()
    logger.info("Sequences: %d, Loop features: %d rows",
                len(sequences), len(loop_df))

    # ==================================================================
    # Separate negatives (shared across all train/test configs)
    # ==================================================================
    neg_df = splits_df[splits_df["dataset_source"].isin(NEGATIVE_SOURCES)]
    neg_train_val = neg_df[neg_df["split"].isin(["train", "val"])]
    neg_test = neg_df[neg_df["split"] == "test"]
    logger.info("Negatives: %d train+val, %d test", len(neg_train_val), len(neg_test))

    # ==================================================================
    # Define train/test configs
    # ==================================================================
    train_configs = DATASET_SOURCES + ["All"]
    train_labels = [DATASET_LABELS[ds] for ds in DATASET_SOURCES] + ["All"]
    test_labels = [DATASET_LABELS[ds] for ds in DATASET_SOURCES]

    n_train = len(train_configs)
    n_test = len(DATASET_SOURCES)

    # ==================================================================
    # Precompute train/test DataFrames for each config
    # ==================================================================
    train_dfs = {}
    for train_cfg in train_configs:
        if train_cfg == "All":
            train_pos = splits_df[
                (splits_df["label"] == 1)
                & (splits_df["split"].isin(["train", "val"]))
            ]
        else:
            train_pos = splits_df[
                (splits_df["dataset_source"] == train_cfg)
                & (splits_df["label"] == 1)
                & (splits_df["split"].isin(["train", "val"]))
            ]
        train_data = pd.concat([train_pos, neg_train_val], ignore_index=True)
        train_dfs[train_cfg] = train_data

    test_dfs = {}
    for test_ds in DATASET_SOURCES:
        test_pos = splits_df[
            (splits_df["dataset_source"] == test_ds)
            & (splits_df["label"] == 1)
            & (splits_df["split"] == "test")
        ]
        test_data = pd.concat([test_pos, neg_test], ignore_index=True)
        test_dfs[test_ds] = test_data

    # Log dataset sizes
    for cfg in train_configs:
        tdf = train_dfs[cfg]
        n_p = (tdf["label"] == 1).sum()
        n_n = (tdf["label"] == 0).sum()
        label = DATASET_LABELS[cfg] if cfg != "All" else "All"
        logger.info("  Train [%s]: %d pos + %d neg = %d total",
                     label, n_p, n_n, len(tdf))
    for ds in DATASET_SOURCES:
        tdf = test_dfs[ds]
        n_p = (tdf["label"] == 1).sum()
        n_n = (tdf["label"] == 0).sum()
        logger.info("  Test  [%s]: %d pos + %d neg = %d total",
                     DATASET_LABELS[ds], n_p, n_n, len(tdf))

    # ==================================================================
    # Run experiments: for each model, build the NxN matrix
    # ==================================================================
    all_results = {}
    t_total_start = time.time()

    for model_name in MODEL_ORDER:
        logger.info("\n" + "=" * 80)
        logger.info("MODEL: %s", model_name)
        logger.info("=" * 80)

        matrix = {}  # {train_label: {test_label: auroc}}
        t_model_start = time.time()

        for i, train_cfg in enumerate(train_configs):
            train_label = train_labels[i]
            train_data = train_dfs[train_cfg]
            n_train_pos = (train_data["label"] == 1).sum()

            if n_train_pos < 20:
                logger.info("  Skipping train=%s (%d positives < 20)",
                            train_label, n_train_pos)
                matrix[train_label] = {tl: float("nan") for tl in test_labels}
                continue

            matrix[train_label] = {}

            # ---- GB models: extract features once per train config ----
            if model_name in ("StructureOnly", "GB_HandFeatures", "GB_AllFeatures"):
                train_ids = train_data["site_id"].tolist()
                y_train = train_data["label"].values.astype(np.float32)

                if model_name == "StructureOnly":
                    # 16-dim: 7 structure delta + 9 loop position
                    X_train = np.hstack([
                        extract_structure_delta_for_ids(train_ids, structure_delta_dict),
                        extract_loop_features_for_ids(train_ids, loop_df),
                    ])
                    X_train = np.nan_to_num(X_train, nan=0.0)
                elif model_name == "GB_HandFeatures":
                    X_train = extract_hand_features(
                        train_ids, sequences, structure_delta_dict, loop_df
                    )
                else:
                    X_train = extract_all_features(
                        train_ids, sequences, structure_delta_dict, loop_df,
                        pooled_orig, pooled_edited,
                    )

                for j, test_ds in enumerate(DATASET_SOURCES):
                    test_label = test_labels[j]
                    test_data = test_dfs[test_ds]
                    test_ids = test_data["site_id"].tolist()
                    y_test = test_data["label"].values.astype(np.float32)

                    n_test_pos = (test_data["label"] == 1).sum()
                    if n_test_pos < 3:
                        logger.info("    [%s -> %s] Skipping: %d test positives",
                                    train_label, test_label, n_test_pos)
                        matrix[train_label][test_label] = float("nan")
                        continue

                    if model_name == "StructureOnly":
                        X_test = np.hstack([
                            extract_structure_delta_for_ids(test_ids, structure_delta_dict),
                            extract_loop_features_for_ids(test_ids, loop_df),
                        ])
                        X_test = np.nan_to_num(X_test, nan=0.0)
                    elif model_name == "GB_HandFeatures":
                        X_test = extract_hand_features(
                            test_ids, sequences, structure_delta_dict, loop_df
                        )
                    else:
                        X_test = extract_all_features(
                            test_ids, sequences, structure_delta_dict, loop_df,
                            pooled_orig, pooled_edited,
                        )

                    auroc = train_gb_model(X_train, y_train, X_test, y_test)
                    matrix[train_label][test_label] = auroc
                    logger.info("    [%s -> %s] AUROC=%.4f", train_label, test_label, auroc)

            # ---- DL models ----
            else:
                for j, test_ds in enumerate(DATASET_SOURCES):
                    test_label = test_labels[j]
                    test_data = test_dfs[test_ds]

                    n_test_pos = (test_data["label"] == 1).sum()
                    if n_test_pos < 3:
                        logger.info("    [%s -> %s] Skipping: %d test positives",
                                    train_label, test_label, n_test_pos)
                        matrix[train_label][test_label] = float("nan")
                        continue

                    # Build a fresh model for each (train, test) pair
                    model = build_dl_model(model_name, pooled_orig, pooled_edited)

                    auroc = train_dl_model(
                        model=model,
                        model_name=model_name,
                        train_df=train_data,
                        test_df=test_data,
                        pooled_orig=pooled_orig,
                        pooled_edited=pooled_edited,
                        structure_delta=structure_delta_dict,
                        epochs=50,
                        lr=1e-3,
                        patience=10,
                        batch_size=64,
                    )
                    matrix[train_label][test_label] = auroc
                    logger.info("    [%s -> %s] AUROC=%.4f", train_label, test_label, auroc)

        t_model_elapsed = time.time() - t_model_start
        all_results[model_name] = matrix
        logger.info("  %s completed in %.1fs", model_name, t_model_elapsed)

        # Print matrix for this model
        _print_matrix(model_name, matrix, train_labels, test_labels)

    t_total = time.time() - t_total_start
    logger.info("\nTotal time: %.1fs", t_total)

    # ==================================================================
    # Print combined summary
    # ==================================================================
    print("\n" + "=" * 100)
    print("COMPREHENSIVE CROSS-DATASET GENERALIZATION MATRIX")
    print("=" * 100)

    for model_name in MODEL_ORDER:
        _print_matrix(model_name, all_results[model_name], train_labels, test_labels)

    # ==================================================================
    # Save results
    # ==================================================================
    output = {
        "experiment": "cross_dataset_full",
        "train_datasets": train_labels,
        "test_datasets": test_labels,
        "models": {},
        "total_time_seconds": t_total,
    }

    for model_name in MODEL_ORDER:
        output["models"][model_name] = all_results[model_name]

    results_path = OUTPUT_DIR / "cross_dataset_full_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=_serialize)

    logger.info("Results saved to %s", results_path)

    # ==================================================================
    # Generate heatmaps
    # ==================================================================
    _generate_heatmaps(all_results, train_labels, test_labels)

    logger.info("All outputs saved to %s", OUTPUT_DIR)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _print_matrix(
    model_name: str,
    matrix: Dict[str, Dict[str, float]],
    train_labels: List[str],
    test_labels: List[str],
):
    """Print an AUROC matrix for a single model."""
    print(f"\n--- {model_name} ---")
    train_test_label = "Train \\ Test"
    header = f"{train_test_label:<15}" + "".join(f"{tl:>12}" for tl in test_labels)
    print(header)
    print("-" * (15 + 12 * len(test_labels)))
    for tl in train_labels:
        row = f"{tl:<15}"
        for te in test_labels:
            val = matrix.get(tl, {}).get(te, float("nan"))
            if val is None or (isinstance(val, float) and np.isnan(val)):
                row += f"{'N/A':>12}"
            else:
                row += f"{val:>12.4f}"
        print(row)
    print()


def _generate_heatmaps(
    all_results: Dict[str, Dict],
    train_labels: List[str],
    test_labels: List[str],
):
    """Generate and save heatmap PNGs for each model."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping heatmaps")
        return

    n_models = len(MODEL_ORDER)
    n_train = len(train_labels)
    n_test = len(test_labels)

    # Individual heatmaps per model
    for model_name in MODEL_ORDER:
        matrix = all_results[model_name]
        data = np.full((n_train, n_test), float("nan"))
        for i, tl in enumerate(train_labels):
            for j, te in enumerate(test_labels):
                val = matrix.get(tl, {}).get(te, float("nan"))
                if val is not None:
                    data[i, j] = val

        fig, ax = plt.subplots(figsize=(8, 6))
        masked = np.ma.masked_invalid(data)
        im = ax.imshow(masked, cmap="YlOrRd", aspect="auto", vmin=0.5, vmax=1.0)

        ax.set_xticks(range(n_test))
        ax.set_xticklabels(test_labels, fontsize=10)
        ax.set_yticks(range(n_train))
        ax.set_yticklabels(train_labels, fontsize=10)
        ax.set_xlabel("Test Dataset", fontsize=12)
        ax.set_ylabel("Training Dataset", fontsize=12)
        ax.set_title(f"Cross-Dataset AUROC: {model_name}", fontsize=13, fontweight="bold")

        for i in range(n_train):
            for j in range(n_test):
                val = data[i, j]
                if not np.isnan(val):
                    color = "white" if val > 0.85 else "black"
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            fontsize=9, fontweight="bold", color=color)

        plt.colorbar(im, ax=ax, label="AUROC", shrink=0.8)
        plt.tight_layout()
        safe_name = model_name.replace(" ", "_").lower()
        plt.savefig(
            OUTPUT_DIR / f"heatmap_{safe_name}.png", dpi=150, bbox_inches="tight"
        )
        plt.close()

    # Combined figure with all models
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    axes_flat = axes.flatten()

    for idx, model_name in enumerate(MODEL_ORDER):
        ax = axes_flat[idx]
        matrix = all_results[model_name]
        data = np.full((n_train, n_test), float("nan"))
        for i, tl in enumerate(train_labels):
            for j, te in enumerate(test_labels):
                val = matrix.get(tl, {}).get(te, float("nan"))
                if val is not None:
                    data[i, j] = val

        masked = np.ma.masked_invalid(data)
        im = ax.imshow(masked, cmap="YlOrRd", aspect="auto", vmin=0.5, vmax=1.0)

        ax.set_xticks(range(n_test))
        ax.set_xticklabels(test_labels, fontsize=8, rotation=30, ha="right")
        ax.set_yticks(range(n_train))
        ax.set_yticklabels(train_labels, fontsize=8)
        ax.set_title(model_name, fontsize=11, fontweight="bold")

        for i in range(n_train):
            for j in range(n_test):
                val = data[i, j]
                if not np.isnan(val):
                    color = "white" if val > 0.85 else "black"
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            fontsize=7, fontweight="bold", color=color)

    fig.suptitle("Cross-Dataset Generalization: All Models", fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=axes_flat, label="AUROC", shrink=0.6, pad=0.02)
    plt.tight_layout(rect=[0, 0, 0.95, 0.96])
    plt.savefig(OUTPUT_DIR / "heatmap_combined.png", dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("Heatmaps saved to %s", OUTPUT_DIR)


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
    main()
