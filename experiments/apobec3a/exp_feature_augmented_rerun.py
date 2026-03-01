#!/usr/bin/env python
"""Re-run feature-augmented classification models on the EXACT same data/splits as baselines.

The original exp_feature_augmented.py produced test set sizes of 867+216=1083,
while the baselines (train_baselines.py) produce 855+209=1064. This script
ensures a fair comparison by using the identical data pipeline from
train_baselines.py (same splits_csv, neg_ratio, seed, and available_ids
filtering) so the test set is guaranteed to match.

Models trained:
  1. SubtractionMLP+Features  - SubtractionMLP baseline + 33-dim hand features
  2. DiffAttention+Features   - DiffAttention baseline + 33-dim hand features
  3. EditRNA+Features         - EditRNA-A3A with hand features via GatedModalityFusion d_gnn

Hand features (33 dims):
  - Motif features (24 dims): TC motif, upstream/downstream nucleotide one-hot,
    trinucleotide one-hot
  - Loop position features (9 dims): is_unpaired, loop_size, dist_to_junction,
    dist_to_apex, relative_loop_position, left/right stem length,
    max_adjacent_stem_length, local_unpaired_fraction

Usage:
    python experiments/apobec3a/exp_feature_augmented_rerun.py
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
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.apobec.train_baselines import (
    BaselineConfig,
    FocalLoss,
    compute_metrics,
    load_data,
    create_dataloaders,
)
from experiments.apobec.train_gradient_boosting import (
    extract_motif_features,
    extract_loop_position_features,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_DIR = (
    PROJECT_ROOT / "experiments" / "apobec" / "outputs" / "feature_augmented_rerun"
)
SEED = 42
D_MODEL = 640
D_HAND = 33  # 24 motif + 9 loop


# ---------------------------------------------------------------------------
# Feature Extraction (reused from exp_feature_augmented.py)
# ---------------------------------------------------------------------------

def extract_hand_features(
    splits_df: pd.DataFrame,
) -> Tuple[Dict[str, np.ndarray], List[str], List[str]]:
    """Extract 33-dim hand-crafted features for each site.

    Returns:
        feature_dict: {site_id: np.ndarray(33,)}
        motif_cols: list of motif feature column names
        loop_cols: list of loop feature column names
    """
    motif_df = extract_motif_features(splits_df)
    loop_df = extract_loop_position_features(splits_df)

    motif_cols = [c for c in motif_df.columns if c != "site_id"]
    loop_cols = [c for c in loop_df.columns if c != "site_id"]

    logger.info(
        "Hand features: %d motif + %d loop = %d total",
        len(motif_cols),
        len(loop_cols),
        len(motif_cols) + len(loop_cols),
    )

    motif_indexed = motif_df.set_index("site_id")
    loop_indexed = loop_df.set_index("site_id")

    motif_zeros = [0.0] * len(motif_cols)
    loop_zeros = [0.0] * len(loop_cols)

    feature_dict = {}
    for _, row in splits_df.iterrows():
        sid = str(row["site_id"])

        if sid in motif_indexed.index:
            motif_row = motif_indexed.loc[sid]
            motif_vals = [float(motif_row.get(c, 0.0)) for c in motif_cols]
        else:
            motif_vals = motif_zeros

        if sid in loop_indexed.index:
            loop_row = loop_indexed.loc[sid]
            loop_vals = [float(loop_row.get(c, 0.0)) for c in loop_cols]
        else:
            loop_vals = loop_zeros

        feature_dict[sid] = np.array(motif_vals + loop_vals, dtype=np.float32)

    return feature_dict, motif_cols, loop_cols


# ---------------------------------------------------------------------------
# Augmented Dataset & Collate
# ---------------------------------------------------------------------------

class AugmentedEmbeddingDataset(Dataset):
    """EmbeddingDataset extended with hand-crafted features tensor."""

    def __init__(
        self,
        site_ids: List[str],
        labels: np.ndarray,
        pooled_orig: Dict[str, torch.Tensor],
        pooled_edited: Dict[str, torch.Tensor],
        tokens_orig: Optional[Dict[str, torch.Tensor]] = None,
        tokens_edited: Optional[Dict[str, torch.Tensor]] = None,
        structure_delta: Optional[Dict[str, np.ndarray]] = None,
        hand_features: Optional[Dict[str, np.ndarray]] = None,
        d_hand: int = 33,
    ):
        self.site_ids = site_ids
        self.labels = labels
        self.pooled_orig = pooled_orig
        self.pooled_edited = pooled_edited
        self.tokens_orig = tokens_orig
        self.tokens_edited = tokens_edited
        self.structure_delta = structure_delta
        self.hand_features = hand_features
        self.d_hand = d_hand

    def __len__(self) -> int:
        return len(self.site_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sid = self.site_ids[idx]
        item = {
            "site_id": sid,
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
            "pooled_orig": self.pooled_orig[sid],
            "pooled_edited": self.pooled_edited[sid],
        }

        if self.tokens_orig is not None and sid in self.tokens_orig:
            item["tokens_orig"] = self.tokens_orig[sid]
            item["tokens_edited"] = self.tokens_edited[sid]

        if self.structure_delta is not None and sid in self.structure_delta:
            item["structure_delta"] = torch.tensor(
                self.structure_delta[sid], dtype=torch.float32
            )
        else:
            item["structure_delta"] = torch.zeros(7, dtype=torch.float32)

        if self.hand_features is not None and sid in self.hand_features:
            item["hand_features"] = torch.tensor(
                self.hand_features[sid], dtype=torch.float32
            )
        else:
            item["hand_features"] = torch.zeros(self.d_hand, dtype=torch.float32)

        return item


def augmented_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for AugmentedEmbeddingDataset."""
    result = {
        "site_ids": [b["site_id"] for b in batch],
        "labels": torch.stack([b["label"] for b in batch]),
        "pooled_orig": torch.stack([b["pooled_orig"] for b in batch]),
        "pooled_edited": torch.stack([b["pooled_edited"] for b in batch]),
        "structure_delta": torch.stack([b["structure_delta"] for b in batch]),
        "hand_features": torch.stack([b["hand_features"] for b in batch]),
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


# ---------------------------------------------------------------------------
# Feature-Augmented Models (replicated from exp_feature_augmented.py)
# ---------------------------------------------------------------------------

class FeatureAugmentedSubtractionMLP(nn.Module):
    """SubtractionMLP with hand-crafted features concatenated before MLP."""

    def __init__(self, d_model: int = 640, d_hand: int = 33, dropout: float = 0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model + d_hand, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, batch: dict) -> dict:
        diff = batch["pooled_edited"] - batch["pooled_orig"]
        hand = batch["hand_features"]
        combined = torch.cat([diff, hand], dim=-1)
        logit = self.mlp(combined)
        return {"binary_logit": logit}


class FeatureAugmentedDiffAttention(nn.Module):
    """DiffAttention with hand features concatenated after pooling."""

    def __init__(
        self,
        d_model: int = 640,
        d_hand: int = 33,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        self.mlp = nn.Sequential(
            nn.Linear(d_model + d_hand, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, batch: dict) -> dict:
        diff = batch["tokens_edited"] - batch["tokens_orig"]
        encoded = self.transformer(diff)
        pooled = encoded.mean(dim=1)
        hand = batch["hand_features"]
        combined = torch.cat([pooled, hand], dim=-1)
        logit = self.mlp(combined)
        return {"binary_logit": logit}


class FeatureAugmentedEditRNA(nn.Module):
    """EditRNA-A3A with hand features injected via GatedModalityFusion's d_gnn slot.

    Hand features are treated as a third modality alongside
    primary (RNA-FM pooled) and edit (APOBEC edit embedding).
    """

    def __init__(
        self,
        d_model: int = 640,
        d_hand: int = 33,
        d_edit: int = 256,
        d_fused: int = 512,
        pooled_only: bool = True,
        cached_encoder=None,
    ):
        super().__init__()

        from models.editrna_a3a import EditRNA_A3A, EditRNAConfig
        from models.fusion import GatedModalityFusion

        config = EditRNAConfig(
            primary_encoder="cached",
            d_model=d_model,
            d_edit=d_edit,
            d_fused=d_fused,
            pooled_only=pooled_only,
            use_gnn=False,
        )

        # Build the base EditRNA model
        self.base_model = EditRNA_A3A(config=config, primary_encoder=cached_encoder)

        # Replace the fusion module with one that has d_gnn=d_hand
        self.base_model.fusion = GatedModalityFusion(
            d_model=d_model,
            d_edit=d_edit,
            d_fused=d_fused,
            d_model_secondary=0,
            d_gnn=d_hand,
            dropout=config.fusion_dropout,
        )

        self.d_hand = d_hand

    def forward(self, batch: dict) -> dict:
        """Forward with hand features injected as gnn_emb in fusion."""
        sequences = batch["sequences"]
        edit_pos = batch["edit_pos"]
        flanking_context = batch["flanking_context"]
        concordance_features = batch["concordance_features"]
        structure_delta = batch["structure_delta"]
        hand_features = batch["hand_features"]
        site_ids = batch.get("site_ids")
        device = edit_pos.device

        base = self.base_model
        cfg = base.config

        # Encode original
        primary_out = base._encode_primary(sequences, site_ids=site_ids, edited=False)
        primary_pooled = primary_out["pooled"]

        if cfg.pooled_only:
            f_background = primary_pooled.unsqueeze(1)
            edited_sequences = base._make_edited_sequences(sequences, edit_pos)
            edited_out = base._encode_primary(
                edited_sequences, site_ids=site_ids, edited=True
            )
            f_edited = (
                edited_out["tokens"][:, :1, :]
                if edited_out["tokens"].dim() == 3
                else edited_out["pooled"].unsqueeze(1)
            )
            edit_pos_adj = torch.zeros_like(edit_pos)
            seq_mask = torch.ones(
                f_background.shape[0], 1, dtype=torch.bool, device=device
            )
        else:
            f_background = primary_out["tokens"]
            edited_sequences = base._make_edited_sequences(sequences, edit_pos)
            edited_out = base._encode_primary(
                edited_sequences, site_ids=site_ids, edited=True
            )
            f_edited = edited_out["tokens"]
            min_len = min(f_background.shape[1], f_edited.shape[1])
            f_background = f_background[:, :min_len, :]
            f_edited = f_edited[:, :min_len, :]
            edit_pos_adj = edit_pos
            seq_mask = torch.ones(
                f_background.shape[0], min_len, dtype=torch.bool, device=device
            )

        # Edit embedding
        edit_emb = base.edit_embedding(
            f_background=f_background,
            f_edited=f_edited,
            edit_pos=edit_pos_adj if cfg.pooled_only else edit_pos,
            flanking_context=flanking_context,
            structure_delta=structure_delta,
            concordance_features=concordance_features,
            seq_mask=seq_mask,
        )

        # Fusion with hand features as gnn_emb
        fused = base.fusion(
            primary_pooled=primary_pooled,
            edit_emb=edit_emb,
            gnn_emb=hand_features,
        )

        # Prediction heads
        predictions = base.heads(fused, edit_emb)

        return {
            "predictions": predictions,
            "edit_embedding": edit_emb,
            "fused": fused,
        }


# ---------------------------------------------------------------------------
# Forward helper
# ---------------------------------------------------------------------------

def _forward_augmented(model, batch, model_name):
    """Forward pass dispatcher for augmented models.

    Handles:
    1. editrna_features: FeatureAugmentedEditRNA - needs adapted batch
    2. Everything else: direct model(batch) - SubtractionMLP+Features, DiffAttention+Features
    """
    if model_name == "editrna_features":
        B = batch["labels"].shape[0]
        device = batch["labels"].device
        editrna_batch = {
            "sequences": ["N" * 201] * B,
            "site_ids": batch["site_ids"],
            "edit_pos": torch.full((B,), 100, dtype=torch.long, device=device),
            "flanking_context": torch.zeros(B, dtype=torch.long, device=device),
            "concordance_features": torch.zeros(B, 5, device=device),
            "structure_delta": batch["structure_delta"],
            "hand_features": batch["hand_features"],
        }
        output = model(editrna_batch)
        return {"binary_logit": output["predictions"]["binary_logit"]}

    return model(batch)


# ---------------------------------------------------------------------------
# Training & Evaluation
# ---------------------------------------------------------------------------

def train_and_evaluate(
    model: nn.Module,
    model_name: str,
    loaders: Dict[str, DataLoader],
    config: BaselineConfig,
) -> Dict:
    """Train a feature-augmented model using the same loop as train_baselines.py."""
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    device = torch.device("cpu")
    model = model.to(device)

    loss_fn = FocalLoss(gamma=config.focal_gamma, alpha=config.focal_alpha)
    optimizer = AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-7)

    best_auroc = -1.0
    best_epoch = 0
    patience_counter = 0
    best_state = None

    logger.info("Training %s for %d epochs...", model_name, config.epochs)
    t_start = time.time()

    for epoch in range(1, config.epochs + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        n_batches = 0

        for batch in loaders["train"]:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            optimizer.zero_grad()
            output = _forward_augmented(model, batch, model_name)
            logits = output["binary_logit"]
            loss = loss_fn(logits, batch["labels"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train_loss = train_loss / max(n_batches, 1)

        # --- Validate ---
        val_metrics = _evaluate(model, loaders.get("val"), model_name, device)

        # Early stopping
        val_auroc = val_metrics.get("auroc", 0.0)
        if not np.isnan(val_auroc) and val_auroc > best_auroc + 1e-4:
            best_auroc = val_auroc
            best_epoch = epoch
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                "  Epoch %3d  loss=%.4f  val_auroc=%.4f  val_f1=%.4f  patience=%d/%d",
                epoch,
                avg_train_loss,
                val_metrics.get("auroc", 0),
                val_metrics.get("f1", 0),
                patience_counter,
                config.patience,
            )

        if patience_counter >= config.patience:
            logger.info(
                "  Early stopping at epoch %d (best=%d, AUROC=%.4f)",
                epoch,
                best_epoch,
                best_auroc,
            )
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    elapsed = time.time() - t_start
    logger.info("  Training took %.1f seconds", elapsed)

    # --- Evaluate on val and test ---
    val_metrics = _evaluate(model, loaders.get("val"), model_name, device)
    test_metrics = _evaluate(model, loaders.get("test"), model_name, device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "test_metrics": test_metrics,
        "val_metrics": val_metrics,
        "best_epoch": best_epoch,
        "best_val_auroc": float(best_auroc),
        "train_time_seconds": elapsed,
        "n_params": n_params,
    }


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: Optional[DataLoader],
    model_name: str,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate a model on a DataLoader."""
    if loader is None:
        return {}

    model.eval()
    all_targets = []
    all_scores = []

    for batch in loader:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        output = _forward_augmented(model, batch, model_name)
        logits = output["binary_logit"].squeeze(-1).cpu().numpy()
        probs = 1.0 / (1.0 + np.exp(-logits))

        all_targets.append(batch["labels"].cpu().numpy())
        all_scores.append(probs)

    y_true = np.concatenate(all_targets)
    y_score = np.concatenate(all_scores)
    return compute_metrics(y_true, y_score)


# ---------------------------------------------------------------------------
# DataLoader creation (using same filtering as train_baselines.create_dataloaders)
# ---------------------------------------------------------------------------

def create_augmented_dataloaders(
    splits_df: pd.DataFrame,
    pooled_orig: Dict[str, torch.Tensor],
    pooled_edited: Dict[str, torch.Tensor],
    tokens_orig: Optional[Dict[str, torch.Tensor]],
    tokens_edited: Optional[Dict[str, torch.Tensor]],
    structure_delta: Optional[Dict[str, np.ndarray]],
    hand_features: Dict[str, np.ndarray],
    config: BaselineConfig,
    needs_tokens: bool = False,
) -> Dict[str, DataLoader]:
    """Create train/val/test DataLoaders with hand features.

    Uses the EXACT same filtering logic as train_baselines.create_dataloaders
    to ensure identical splits and sample counts.
    """
    available_ids = set(pooled_orig.keys()) & set(pooled_edited.keys())
    df = splits_df[splits_df["site_id"].isin(available_ids)].copy()

    # Apply negative ratio (same logic as train_baselines.create_dataloaders)
    pos_df = df[df["label"] == 1]
    neg_df = df[df["label"] == 0]
    n_pos = len(pos_df)
    max_neg = n_pos * config.neg_ratio
    if len(neg_df) > max_neg:
        neg_df = neg_df.sample(n=max_neg, random_state=config.seed)
    df = pd.concat([pos_df, neg_df], ignore_index=True)

    loaders = {}
    for split_name in ["train", "val", "test"]:
        subset = df[df["split"] == split_name]
        if len(subset) == 0:
            logger.warning("No samples for split '%s'", split_name)
            continue

        site_ids = subset["site_id"].tolist()
        labels = subset["label"].values.astype(np.float32)

        ds = AugmentedEmbeddingDataset(
            site_ids=site_ids,
            labels=labels,
            pooled_orig=pooled_orig,
            pooled_edited=pooled_edited,
            tokens_orig=tokens_orig if needs_tokens else None,
            tokens_edited=tokens_edited if needs_tokens else None,
            structure_delta=structure_delta,
            hand_features=hand_features,
            d_hand=D_HAND,
        )

        loaders[split_name] = DataLoader(
            ds,
            batch_size=config.batch_size,
            shuffle=(split_name == "train"),
            num_workers=0,
            collate_fn=augmented_collate_fn,
            drop_last=False,
        )

        n_p = int(labels.sum())
        n_n = len(labels) - n_p
        logger.info(
            "  %s: %d samples (%d pos, %d neg, ratio=1:%.1f)",
            split_name,
            len(labels),
            n_p,
            n_n,
            n_n / max(n_p, 1),
        )

    return loaders


# ---------------------------------------------------------------------------
# Serialization helper
# ---------------------------------------------------------------------------

def _serialize(obj):
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return str(obj)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Use default BaselineConfig to match train_baselines.py exactly
    config = BaselineConfig()

    # ==================================================================
    # Load data (same as train_baselines.py)
    # ==================================================================
    logger.info("Loading data with load_tokens=True ...")
    (
        pooled_orig,
        pooled_edited,
        tokens_orig,
        tokens_edited,
        structure_delta,
        splits_df,
    ) = load_data(config, load_tokens=True)

    # ==================================================================
    # Extract hand features from splits_df
    # ==================================================================
    logger.info("Extracting hand features...")
    hand_features, motif_cols, loop_cols = extract_hand_features(splits_df)
    d_motif = len(motif_cols)
    d_loop = len(loop_cols)
    d_hand = d_motif + d_loop
    logger.info(
        "Hand features: %d dims (motif=%d, loop=%d)", d_hand, d_motif, d_loop
    )
    assert d_hand == D_HAND, (
        f"Expected {D_HAND} hand features, got {d_hand}"
    )

    # ==================================================================
    # Create augmented DataLoaders (with tokens for DiffAttention)
    # ==================================================================
    logger.info("Creating augmented dataloaders...")
    loaders_with_tokens = create_augmented_dataloaders(
        splits_df,
        pooled_orig,
        pooled_edited,
        tokens_orig,
        tokens_edited,
        structure_delta,
        hand_features,
        config,
        needs_tokens=True,
    )

    # Also create loaders without tokens (for SubtractionMLP and EditRNA)
    loaders_no_tokens = create_augmented_dataloaders(
        splits_df,
        pooled_orig,
        pooled_edited,
        tokens_orig,
        tokens_edited,
        structure_delta,
        hand_features,
        config,
        needs_tokens=False,
    )

    # Verify test set sizes
    test_loader = loaders_no_tokens["test"]
    n_test = len(test_loader.dataset)
    n_test_pos = int(test_loader.dataset.labels.sum())
    n_test_neg = n_test - n_test_pos
    logger.info(
        "Test set: %d samples (%d pos, %d neg)", n_test, n_test_pos, n_test_neg
    )

    # ==================================================================
    # Build CachedRNAEncoder for EditRNA+Features
    # ==================================================================
    from models.encoders import CachedRNAEncoder

    cached_encoder = CachedRNAEncoder(
        tokens_cache=tokens_orig,
        pooled_cache=pooled_orig,
        tokens_edited_cache=tokens_edited,
        pooled_edited_cache=pooled_edited,
        d_model=D_MODEL,
    )

    # ==================================================================
    # Train 3 feature-augmented models
    # ==================================================================
    all_results = {}

    # --- 1. SubtractionMLP+Features ---
    logger.info("\n" + "=" * 70)
    logger.info("MODEL 1: SubtractionMLP+Features")
    logger.info("=" * 70)

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    sub_model = FeatureAugmentedSubtractionMLP(
        d_model=D_MODEL, d_hand=d_hand, dropout=0.3
    )
    n_params = sum(p.numel() for p in sub_model.parameters() if p.requires_grad)
    logger.info("  Trainable parameters: %s", f"{n_params:,}")

    r = train_and_evaluate(
        sub_model, "subtraction_features", loaders_no_tokens, config
    )
    all_results["SubtractionMLP+Features"] = r
    logger.info(
        "  Test AUROC=%.4f  AUPRC=%.4f  F1=%.4f",
        r["test_metrics"].get("auroc", 0),
        r["test_metrics"].get("auprc", 0),
        r["test_metrics"].get("f1", 0),
    )

    # --- 2. DiffAttention+Features ---
    logger.info("\n" + "=" * 70)
    logger.info("MODEL 2: DiffAttention+Features")
    logger.info("=" * 70)

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    diff_model = FeatureAugmentedDiffAttention(
        d_model=D_MODEL, d_hand=d_hand, n_heads=8, n_layers=2, dropout=0.3
    )
    n_params = sum(p.numel() for p in diff_model.parameters() if p.requires_grad)
    logger.info("  Trainable parameters: %s", f"{n_params:,}")

    r = train_and_evaluate(
        diff_model, "diff_features", loaders_with_tokens, config
    )
    all_results["DiffAttention+Features"] = r
    logger.info(
        "  Test AUROC=%.4f  AUPRC=%.4f  F1=%.4f",
        r["test_metrics"].get("auroc", 0),
        r["test_metrics"].get("auprc", 0),
        r["test_metrics"].get("f1", 0),
    )

    # --- 3. EditRNA+Features ---
    logger.info("\n" + "=" * 70)
    logger.info("MODEL 3: EditRNA+Features")
    logger.info("=" * 70)

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    editrna_model = FeatureAugmentedEditRNA(
        d_model=D_MODEL,
        d_hand=d_hand,
        d_edit=256,
        d_fused=512,
        pooled_only=True,
        cached_encoder=cached_encoder,
    )
    n_params = sum(p.numel() for p in editrna_model.parameters() if p.requires_grad)
    logger.info("  Trainable parameters: %s", f"{n_params:,}")

    r = train_and_evaluate(
        editrna_model, "editrna_features", loaders_no_tokens, config
    )
    all_results["EditRNA+Features"] = r
    logger.info(
        "  Test AUROC=%.4f  AUPRC=%.4f  F1=%.4f",
        r["test_metrics"].get("auroc", 0),
        r["test_metrics"].get("auprc", 0),
        r["test_metrics"].get("f1", 0),
    )

    # ==================================================================
    # Summary
    # ==================================================================
    print("\n" + "=" * 110)
    print("FEATURE-AUGMENTED RERUN RESULTS (same splits as baselines)")
    print("=" * 110)
    print(
        f"{'Model':<30} {'AUROC':>8} {'AUPRC':>8} {'F1':>8} "
        f"{'Prec':>8} {'Recall':>8} {'P@90R':>8} {'ECE':>8} "
        f"{'Params':>10} {'Epoch':>6}"
    )
    print("-" * 110)

    sorted_results = sorted(
        all_results.items(),
        key=lambda x: x[1].get("test_metrics", {}).get("auroc", 0),
        reverse=True,
    )

    for name, r in sorted_results:
        tm = r.get("test_metrics", {})
        print(
            f"{name:<30} "
            f"{tm.get('auroc', 0):>8.4f} "
            f"{tm.get('auprc', 0):>8.4f} "
            f"{tm.get('f1', 0):>8.4f} "
            f"{tm.get('precision', 0):>8.4f} "
            f"{tm.get('recall', 0):>8.4f} "
            f"{tm.get('precision_at_90recall', 0):>8.4f} "
            f"{tm.get('ece', 0):>8.4f} "
            f"{r.get('n_params', 'N/A'):>10,} "
            f"{r.get('best_epoch', 0):>6}"
        )

    print("=" * 110)

    # Verify test set consistency
    for name, r in all_results.items():
        tm = r.get("test_metrics", {})
        n_pos = tm.get("n_positive", 0)
        n_neg = tm.get("n_negative", 0)
        print(f"  {name}: test set = {n_pos} pos + {n_neg} neg = {n_pos + n_neg} total")

    # ==================================================================
    # Save results
    # ==================================================================
    output = {}
    for name, r in all_results.items():
        output[name] = {
            "test_metrics": r["test_metrics"],
            "val_metrics": r["val_metrics"],
            "best_epoch": r["best_epoch"],
            "n_params": r["n_params"],
        }

    results_path = OUTPUT_DIR / "feature_augmented_rerun_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=_serialize)

    logger.info("\nResults saved to %s", results_path)

    return output


if __name__ == "__main__":
    main()
