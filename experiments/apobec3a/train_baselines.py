#!/usr/bin/env python
"""Unified training and evaluation script for all baseline models.

Trains baseline models for binary APOBEC C-to-U editing site classification
using pre-computed RNA-FM embeddings (CPU-trainable). Supports all baselines
plus the full EditRNA-A3A model for comparison.

Supported models:
  pooled_mlp      - RNA-FM pooled only (no edit info)
  subtraction_mlp - RNA-FM(edited) - RNA-FM(original)
  concat_mlp      - RNA-FM(orig) ++ RNA-FM(edited) concatenation
  cross_attention  - Cross-attention over token-level embeddings
  diff_attention   - Token diff + TransformerEncoder
  structure_only   - ViennaRNA structure delta features only
  editrna          - Full EditRNA-A3A (ours)

Usage:
    # Train a single model
    python experiments/apobec3a/train_baselines.py --model concat_mlp --epochs 50

    # Train all baselines
    python experiments/apobec3a/train_baselines.py --model all --epochs 50

    # Different negative ratio
    python experiments/apobec3a/train_baselines.py --model all --neg-ratio 5

    # Evaluate existing checkpoint
    python experiments/apobec3a/train_baselines.py --model concat_mlp --evaluate-only
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BaselineConfig:
    """Configuration for baseline training."""

    # Model
    model_name: str = "concat_mlp"
    d_model: int = 640  # RNA-FM embedding dimension

    # Data
    embeddings_dir: str = str(PROJECT_ROOT / "data" / "processed" / "embeddings")
    splits_csv: str = str(PROJECT_ROOT / "data" / "processed" / "splits_expanded.csv")
    labels_csv: str = str(PROJECT_ROOT / "data" / "processed" / "editing_sites_labels.csv")
    structure_cache: str = str(
        PROJECT_ROOT / "data" / "processed" / "embeddings" / "vienna_structure_cache.npz"
    )
    neg_ratio: int = 5  # max negatives per positive

    # Training
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 10
    focal_gamma: float = 2.0
    focal_alpha: float = 0.75
    max_grad_norm: float = 1.0
    seed: int = 42

    # Output
    output_dir: str = str(
        PROJECT_ROOT / "experiments" / "apobec3a" / "outputs" / "baselines"
    )


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Binary focal loss for handling class imbalance."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits.squeeze(-1))
        targets = targets.float()
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        bce = F.binary_cross_entropy_with_logits(
            logits.squeeze(-1), targets, reduction="none"
        )
        return (focal_weight * bce).mean()


# ---------------------------------------------------------------------------
# Embedding-based Dataset
# ---------------------------------------------------------------------------

class EmbeddingDataset(Dataset):
    """Dataset that serves pre-computed embeddings for baseline training.

    Loads RNA-FM pooled/token embeddings from .pt caches and serves them
    alongside labels and optional structure features.
    """

    def __init__(
        self,
        site_ids: List[str],
        labels: np.ndarray,
        pooled_orig: Dict[str, torch.Tensor],
        pooled_edited: Dict[str, torch.Tensor],
        tokens_orig: Optional[Dict[str, torch.Tensor]] = None,
        tokens_edited: Optional[Dict[str, torch.Tensor]] = None,
        structure_delta: Optional[Dict[str, np.ndarray]] = None,
    ):
        self.site_ids = site_ids
        self.labels = labels
        self.pooled_orig = pooled_orig
        self.pooled_edited = pooled_edited
        self.tokens_orig = tokens_orig
        self.tokens_edited = tokens_edited
        self.structure_delta = structure_delta

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

        return item


def embedding_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for EmbeddingDataset."""
    result = {
        "site_ids": [b["site_id"] for b in batch],
        "labels": torch.stack([b["label"] for b in batch]),
        "pooled_orig": torch.stack([b["pooled_orig"] for b in batch]),
        "pooled_edited": torch.stack([b["pooled_edited"] for b in batch]),
        "structure_delta": torch.stack([b["structure_delta"] for b in batch]),
    }

    if "tokens_orig" in batch[0]:
        # Pad token sequences to same length
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
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    """Compute binary classification metrics."""
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        precision_recall_curve,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    mask = ~(np.isnan(y_true) | np.isnan(y_score))
    y_true, y_score = y_true[mask], y_score[mask]

    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return {k: float("nan") for k in [
            "auroc", "auprc", "f1", "precision", "recall",
            "accuracy", "optimal_threshold", "n_positive", "n_negative",
            "precision_at_90recall", "ece",
        ]}

    metrics = {
        "auroc": roc_auc_score(y_true, y_score),
        "auprc": average_precision_score(y_true, y_score),
    }

    # Optimal F1 threshold
    prec_arr, rec_arr, thresholds = precision_recall_curve(y_true, y_score)
    f1_arr = 2 * prec_arr[:-1] * rec_arr[:-1] / (prec_arr[:-1] + rec_arr[:-1] + 1e-8)
    best_idx = np.argmax(f1_arr)
    threshold = thresholds[best_idx] if len(thresholds) > 0 else 0.5
    y_pred = (y_score >= threshold).astype(int)

    metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)
    metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["optimal_threshold"] = float(threshold)
    metrics["n_positive"] = int(y_true.sum())
    metrics["n_negative"] = int(len(y_true) - y_true.sum())

    # Precision at 90% recall
    p90 = float("nan")
    for i in range(len(rec_arr) - 1):
        if rec_arr[i] >= 0.90:
            p90 = prec_arr[i]
            break
    metrics["precision_at_90recall"] = p90

    # Expected calibration error (ECE)
    metrics["ece"] = _compute_ece(y_true, y_score)

    return metrics


def _compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute expected calibration error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += mask.sum() / len(y_true) * abs(bin_acc - bin_conf)
    return float(ece)


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_model(model_name: str, config: BaselineConfig) -> nn.Module:
    """Build a baseline model by name."""
    d = config.d_model

    if model_name == "pooled_mlp":
        from models.baselines.pooled_mlp import PooledMLPBaseline
        return PooledMLPBaseline()

    elif model_name == "subtraction_mlp":
        from models.baselines.subtraction_mlp import SubtractionMLPBaseline
        return SubtractionMLPBaseline()

    elif model_name == "concat_mlp":
        from models.baselines.concat_mlp import ConcatMLP
        return ConcatMLP()

    elif model_name == "cross_attention":
        from models.baselines.cross_attention import CrossAttentionBaseline
        return CrossAttentionBaseline()

    elif model_name == "diff_attention":
        from models.baselines.diff_attention import DiffAttentionBaseline
        return DiffAttentionBaseline()

    elif model_name == "structure_only":
        from models.baselines.structure_only import StructureOnlyBaseline
        return StructureOnlyBaseline()

    elif model_name == "editrna":
        from models.editrna_a3a import EditRNA_A3A, EditRNAConfig
        from models.encoders import CachedRNAEncoder

        emb_dir = Path(config.embeddings_dir)
        pooled_cache = torch.load(emb_dir / "rnafm_pooled.pt", weights_only=False)
        pooled_edited = torch.load(emb_dir / "rnafm_pooled_edited.pt", weights_only=False)

        tokens_path = emb_dir / "rnafm_tokens.pt"
        if tokens_path.exists():
            tokens_cache = torch.load(tokens_path, weights_only=False)
            tokens_edited = torch.load(emb_dir / "rnafm_tokens_edited.pt", weights_only=False)
        else:
            logger.warning("Token embeddings not found — EditRNA-A3A will use pooled-only mode")
            tokens_cache = None
            tokens_edited = None

        cached_encoder = CachedRNAEncoder(
            tokens_cache=tokens_cache,
            pooled_cache=pooled_cache,
            tokens_edited_cache=tokens_edited,
            pooled_edited_cache=pooled_edited,
            d_model=config.d_model,
        )
        model_config = EditRNAConfig(
            primary_encoder="cached",
            d_model=config.d_model,
            learning_rate=config.learning_rate,
            pooled_only=(tokens_cache is None),
        )
        return EditRNA_A3A(config=model_config, primary_encoder=cached_encoder)

    else:
        raise ValueError(f"Unknown model: {model_name}")


ALL_BASELINE_MODELS = [
    "pooled_mlp",
    "subtraction_mlp",
    "concat_mlp",
    "cross_attention",
    "diff_attention",
    "structure_only",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(config: BaselineConfig, load_tokens: bool = False) -> Tuple[
    Dict[str, torch.Tensor], Dict[str, torch.Tensor],
    Optional[Dict[str, torch.Tensor]], Optional[Dict[str, torch.Tensor]],
    Optional[Dict[str, np.ndarray]],
    pd.DataFrame,
]:
    """Load embeddings, structure features, and splits."""
    emb_dir = Path(config.embeddings_dir)

    logger.info("Loading embeddings from %s ...", emb_dir)
    pooled_orig = torch.load(emb_dir / "rnafm_pooled.pt", weights_only=False)
    pooled_edited = torch.load(emb_dir / "rnafm_pooled_edited.pt", weights_only=False)
    logger.info("  Loaded %d pooled embeddings", len(pooled_orig))

    tokens_orig = None
    tokens_edited = None
    if load_tokens:
        tok_path = emb_dir / "rnafm_tokens.pt"
        if tok_path.exists():
            tokens_orig = torch.load(tok_path, weights_only=False)
            tokens_edited = torch.load(
                emb_dir / "rnafm_tokens_edited.pt", weights_only=False
            )
            logger.info("  Loaded %d token embeddings", len(tokens_orig))

    # Structure delta features
    structure_delta = None
    struct_path = Path(config.structure_cache)
    if struct_path.exists():
        data = np.load(struct_path, allow_pickle=True)
        feat_key = "delta_features" if "delta_features" in data else "features"
        if "site_ids" in data and feat_key in data:
            sids = data["site_ids"]
            feats = data[feat_key]
            structure_delta = {str(sid): feats[i] for i, sid in enumerate(sids)}
            logger.info("  Loaded %d structure delta features", len(structure_delta))

    # Splits
    splits_df = pd.read_csv(config.splits_csv)
    logger.info("  Loaded splits: %d sites", len(splits_df))

    return pooled_orig, pooled_edited, tokens_orig, tokens_edited, structure_delta, splits_df


def create_dataloaders(
    splits_df: pd.DataFrame,
    pooled_orig: Dict[str, torch.Tensor],
    pooled_edited: Dict[str, torch.Tensor],
    tokens_orig: Optional[Dict[str, torch.Tensor]],
    tokens_edited: Optional[Dict[str, torch.Tensor]],
    structure_delta: Optional[Dict[str, np.ndarray]],
    config: BaselineConfig,
) -> Dict[str, DataLoader]:
    """Create train/val/test DataLoaders from splits."""
    available_ids = set(pooled_orig.keys()) & set(pooled_edited.keys())
    splits_df = splits_df[splits_df["site_id"].isin(available_ids)].copy()

    # Apply negative ratio
    pos_df = splits_df[splits_df["label"] == 1]
    neg_df = splits_df[splits_df["label"] == 0]
    n_pos = len(pos_df)
    max_neg = n_pos * config.neg_ratio
    if len(neg_df) > max_neg:
        neg_df = neg_df.sample(n=max_neg, random_state=config.seed)
    splits_df = pd.concat([pos_df, neg_df], ignore_index=True)

    loaders = {}
    for split_name in ["train", "val", "test"]:
        subset = splits_df[splits_df["split"] == split_name]
        if len(subset) == 0:
            logger.warning("No samples for split '%s'", split_name)
            continue

        site_ids = subset["site_id"].tolist()
        labels = subset["label"].values.astype(np.float32)

        ds = EmbeddingDataset(
            site_ids=site_ids,
            labels=labels,
            pooled_orig=pooled_orig,
            pooled_edited=pooled_edited,
            tokens_orig=tokens_orig,
            tokens_edited=tokens_edited,
            structure_delta=structure_delta,
        )

        loaders[split_name] = DataLoader(
            ds,
            batch_size=config.batch_size,
            shuffle=(split_name == "train"),
            num_workers=0,
            collate_fn=embedding_collate_fn,
            drop_last=False,
        )

        n_p = int(labels.sum())
        n_n = len(labels) - n_p
        logger.info(
            "  %s: %d samples (%d pos, %d neg, ratio=1:%.1f)",
            split_name, len(labels), n_p, n_n,
            n_n / max(n_p, 1),
        )

    return loaders


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    model: nn.Module,
    loaders: Dict[str, DataLoader],
    config: BaselineConfig,
    model_name: str,
) -> Dict[str, any]:
    """Train a baseline model and return results."""
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
    history = []

    logger.info("Training %s for %d epochs...", model_name, config.epochs)
    t_start = time.time()

    for epoch in range(1, config.epochs + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        n_batches = 0
        for batch in loaders["train"]:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            optimizer.zero_grad()
            output = _forward_baseline(model, batch, model_name)
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

        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })

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
                "  Epoch %3d  loss=%.4f  val_auroc=%.4f  val_auprc=%.4f  "
                "val_f1=%.4f  patience=%d/%d",
                epoch, avg_train_loss, val_metrics.get("auroc", 0),
                val_metrics.get("auprc", 0), val_metrics.get("f1", 0),
                patience_counter, config.patience,
            )

        if patience_counter >= config.patience:
            logger.info("  Early stopping at epoch %d (best=%d, AUROC=%.4f)",
                        epoch, best_epoch, best_auroc)
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    elapsed = time.time() - t_start
    logger.info("  Training took %.1f seconds", elapsed)

    # --- Test ---
    test_metrics = _evaluate(model, loaders.get("test"), model_name, device)

    return {
        "model": model_name,
        "best_epoch": best_epoch,
        "best_val_auroc": best_auroc,
        "train_time_seconds": elapsed,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "history": history,
    }


def _forward_baseline(
    model: nn.Module, batch: Dict[str, torch.Tensor], model_name: str
) -> Dict[str, torch.Tensor]:
    """Run forward pass for a baseline model."""
    if model_name == "editrna":
        # Adapt batch for EditRNA-A3A with cached embeddings.
        # The CachedRNAEncoder path uses site_ids directly,
        # but the forward method still reads some fields unconditionally.
        B = batch["labels"].shape[0]
        device = batch["labels"].device
        editrna_batch = {
            "sequences": ["N" * 201] * B,  # dummy, not used by CachedRNAEncoder
            "site_ids": batch["site_ids"],
            "edit_pos": torch.full((B,), 100, dtype=torch.long, device=device),
            "flanking_context": torch.zeros(B, dtype=torch.long, device=device),
            "concordance_features": torch.zeros(B, 5, device=device),
            "structure_delta": batch["structure_delta"],
        }
        output = model(editrna_batch)
        return {"binary_logit": output["predictions"]["binary_logit"]}

    return model(batch)


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
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        output = _forward_baseline(model, batch, model_name)
        logits = output["binary_logit"].squeeze(-1).cpu().numpy()
        probs = 1.0 / (1.0 + np.exp(-logits))

        all_targets.append(batch["labels"].cpu().numpy())
        all_scores.append(probs)

    y_true = np.concatenate(all_targets)
    y_score = np.concatenate(all_scores)
    return compute_metrics(y_true, y_score)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train baseline models for APOBEC editing prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, default="all",
                        help="Model to train (or 'all' for all baselines)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--neg-ratio", type=int, default=5,
                        help="Max negatives per positive")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embeddings-dir", type=str,
                        default=str(PROJECT_ROOT / "data" / "processed" / "embeddings"))
    parser.add_argument("--splits-csv", type=str,
                        default=str(PROJECT_ROOT / "data" / "processed" / "splits_expanded.csv"))
    parser.add_argument("--output-dir", type=str,
                        default=str(PROJECT_ROOT / "experiments" / "apobec3a" / "outputs" / "baselines"))
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = BaselineConfig(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        neg_ratio=args.neg_ratio,
        patience=args.patience,
        seed=args.seed,
        embeddings_dir=args.embeddings_dir,
        splits_csv=args.splits_csv,
        output_dir=args.output_dir,
    )

    # Set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Determine which models to train
    if args.model == "all":
        models_to_train = ALL_BASELINE_MODELS
    else:
        models_to_train = [args.model]

    # Check if any model needs token-level embeddings
    needs_tokens = any(
        m in ("cross_attention", "diff_attention", "editrna")
        for m in models_to_train
    )

    # Load data
    (pooled_orig, pooled_edited, tokens_orig, tokens_edited,
     structure_delta, splits_df) = load_data(config, load_tokens=needs_tokens)

    # Create DataLoaders
    loaders = create_dataloaders(
        splits_df, pooled_orig, pooled_edited,
        tokens_orig, tokens_edited, structure_delta, config,
    )

    # Train each model
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for model_name in models_to_train:
        logger.info("\n" + "=" * 70)
        logger.info("MODEL: %s", model_name)
        logger.info("=" * 70)

        try:
            model = build_model(model_name, config)
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info("  Trainable parameters: %s", f"{n_params:,}")

            results = train_model(model, loaders, config, model_name)
            all_results.append(results)

            # Save per-model results
            model_dir = output_dir / model_name
            model_dir.mkdir(parents=True, exist_ok=True)
            with open(model_dir / "results.json", "w") as f:
                json.dump(results, f, indent=2, default=_serialize)
            torch.save(model.state_dict(), model_dir / "best_model.pt")

        except Exception as e:
            logger.error("  FAILED: %s", e, exc_info=True)
            all_results.append({"model": model_name, "error": str(e)})

    # Print summary table
    _print_summary(all_results)

    # Save combined results
    with open(output_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=_serialize)

    logger.info("\nResults saved to %s", output_dir)


def _print_summary(results: List[Dict]):
    """Print a summary comparison table."""
    print("\n" + "=" * 100)
    print("BASELINE COMPARISON TABLE")
    print("=" * 100)
    print(f"{'Model':<22} {'Val AUROC':>10} {'Val AUPRC':>10} {'Val F1':>8} "
          f"{'Test AUROC':>10} {'Test AUPRC':>10} {'Test F1':>8} "
          f"{'P@90R':>8} {'ECE':>8}")
    print("-" * 100)

    for r in results:
        if "error" in r:
            print(f"{r['model']:<22} FAILED: {r['error']}")
            continue

        vm = r.get("val_metrics", {})
        tm = r.get("test_metrics", {})
        print(
            f"{r['model']:<22} "
            f"{vm.get('auroc', 0):>10.4f} {vm.get('auprc', 0):>10.4f} "
            f"{vm.get('f1', 0):>8.4f} "
            f"{tm.get('auroc', 0):>10.4f} {tm.get('auprc', 0):>10.4f} "
            f"{tm.get('f1', 0):>8.4f} "
            f"{tm.get('precision_at_90recall', 0):>8.4f} "
            f"{tm.get('ece', 0):>8.4f}"
        )

    print("=" * 100)


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


if __name__ == "__main__":
    main()
