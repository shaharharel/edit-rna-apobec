#!/usr/bin/env python
"""End-to-end training pipeline for APOBEC C-to-U editing site prediction.

Trains the APOBECModel (RNA encoder + edit embedding + multi-task heads)
on the curated editing site dataset. Supports:

- Config-driven experiment setup (YAML or CLI overrides)
- Pre-computed gene-stratified train/val/test splits
- Classification metrics: AUROC, AUPRC, F1 at optimal threshold
- Multi-task training with uncertainty-weighted loss
- Differential learning rates (encoder vs. downstream)
- Gradient accumulation for effective batch sizes
- Checkpoint saving and experiment logging
- Synthetic data mode for pipeline testing

Usage:
    # Train with default config (uses synthetic data if no sequences available)
    python experiments/apobec/train_editrna.py

    # Train with RNA-FM encoder and real sequences
    python experiments/apobec/train_editrna.py \\
        --sequences_json data/sequences/site_sequences.json \\
        --encoder rnafm --epochs 50

    # Quick pipeline test with mock encoder
    python experiments/apobec/train_editrna.py --mock --epochs 3 --batch_size 8
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.apobec_dataset import (
    APOBECDataConfig,
    APOBECDataset,
    APOBECDatasetBuilder,
    APOBECSiteSample,
    N_TISSUES,
    apobec_collate_fn,
    create_apobec_dataloaders,
    encode_apobec_class,
    encode_exonic_function,
    encode_structure_type,
    encode_tissue_spec,
    get_flanking_context,
    compute_concordance_features,
)
from models.editrna_a3a import EditRNA_A3A, EditRNAConfig, create_editrna_mock
from models.encoders import CachedRNAEncoder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Full experiment configuration."""

    # Experiment identity
    name: str = "apobec_editrna_v1"
    output_dir: str = str(PROJECT_ROOT / "experiments" / "apobec" / "outputs")

    # Data
    labels_csv: str = str(PROJECT_ROOT / "data" / "processed" / "editing_sites_labels.csv")
    negatives_csv: str = str(PROJECT_ROOT / "data" / "processed" / "advisor" / "negative_controls_ct.csv")
    splits_csv: str = str(PROJECT_ROOT / "data" / "processed" / "splits.csv")
    hard_negatives_csv: str = str(PROJECT_ROOT / "data" / "processed" / "hard_negatives.csv")
    sequences_json: str = ""  # path to {site_id: sequence} JSON
    combined_csv: str = str(PROJECT_ROOT / "data" / "processed" / "advisor" / "positive_negative_combined.csv")

    # Model
    encoder: str = "rnafm"  # "rnafm", "utrlm", "mock", "cached"
    embeddings_dir: str = ""  # path to pre-computed embedding caches
    d_model: int = 640
    d_edit: int = 256
    d_fused: int = 512
    edit_n_heads: int = 8
    use_structure_delta: bool = True
    use_dual_encoder: bool = False
    use_gnn: bool = False
    finetune_last_n: int = 0
    head_dropout: float = 0.2
    fusion_dropout: float = 0.2

    # Loss
    focal_gamma: float = 2.0
    focal_alpha_binary: float = 0.75
    focal_alpha_conservation: float = 0.85

    # Training
    epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 1e-4
    encoder_lr_factor: float = 0.1
    weight_decay: float = 1e-5
    grad_accumulation_steps: int = 2
    max_grad_norm: float = 1.0
    warmup_epochs: int = 3
    lr_scheduler: str = "cosine_warm"  # "plateau", "cosine_warm", "cosine"

    # Early stopping
    patience: int = 15
    min_delta: float = 1e-4

    # Evaluation
    eval_every: int = 1  # evaluate every N epochs
    primary_metric: str = "auroc"  # metric for early stopping / model selection ("auroc", "multitask_score")

    # Misc
    seed: int = 42
    num_workers: int = 0
    use_mock_encoder: bool = False  # use mock encoder for testing
    window_size: int = 100  # nucleotides on each side of edit
    expanded: bool = False  # use expanded multi-dataset splits
    positive_only: bool = False  # train on positive sites only (no negatives)


# ---------------------------------------------------------------------------
# Mock encoder for pipeline testing
# ---------------------------------------------------------------------------

class MockRNAEncoder(nn.Module):
    """Lightweight mock encoder for pipeline testing without RNA-FM weights."""

    def __init__(self, d_model: int = 640, max_len: int = 512):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(5, d_model)  # A, C, G, U, pad
        self.proj = nn.Linear(d_model, d_model)
        self.nuc_to_idx = {"A": 0, "C": 1, "G": 2, "U": 3}

    def _tokenize(self, sequences: List[str]) -> torch.Tensor:
        max_len = max(len(s) for s in sequences)
        tokens = torch.full((len(sequences), max_len), 4, dtype=torch.long)
        for i, seq in enumerate(sequences):
            for j, c in enumerate(seq.upper()):
                tokens[i, j] = self.nuc_to_idx.get(c, 4)
        return tokens

    def forward(
        self,
        sequences: Optional[List[str]] = None,
        tokens: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if tokens is None:
            tokens = self._tokenize(sequences)
        device = next(self.parameters()).device
        tokens = tokens.to(device)
        emb = self.embed(tokens)
        emb = self.proj(emb)
        pooled = emb.mean(dim=1)
        return {"embeddings": emb, "pooled": pooled, "tokens": tokens}


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def compute_classification_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> Dict[str, float]:
    """Compute binary classification metrics.

    Args:
        y_true: Binary labels (0/1).
        y_score: Predicted probabilities or logits.

    Returns:
        Dict with auroc, auprc, f1, precision, recall, accuracy, threshold.
    """
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        precision_recall_curve,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    metrics = {}

    # Filter NaN
    mask = ~(np.isnan(y_true) | np.isnan(y_score))
    y_true = y_true[mask]
    y_score = y_score[mask]

    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return {k: float("nan") for k in [
            "auroc", "auprc", "f1", "precision", "recall", "accuracy",
            "optimal_threshold", "n_positive", "n_negative",
        ]}

    # AUROC
    metrics["auroc"] = roc_auc_score(y_true, y_score)

    # AUPRC
    metrics["auprc"] = average_precision_score(y_true, y_score)

    # Find optimal threshold (max F1)
    precision_arr, recall_arr, thresholds = precision_recall_curve(y_true, y_score)
    f1_arr = 2 * precision_arr[:-1] * recall_arr[:-1] / (
        precision_arr[:-1] + recall_arr[:-1] + 1e-8
    )
    best_idx = np.argmax(f1_arr)
    optimal_threshold = thresholds[best_idx] if len(thresholds) > 0 else 0.5

    y_pred = (y_score >= optimal_threshold).astype(int)
    metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)
    metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["optimal_threshold"] = float(optimal_threshold)

    metrics["n_positive"] = int(y_true.sum())
    metrics["n_negative"] = int(len(y_true) - y_true.sum())

    return metrics


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def build_datasets_from_csvs(
    config: ExperimentConfig,
) -> Tuple[APOBECDataset, APOBECDataset, APOBECDataset]:
    """Build train/val/test datasets from the prepared CSV files.

    Uses:
    - editing_sites_labels.csv: positive sites with full labels
    - negative_controls_ct.csv: CT mismatch negative controls
    - splits.csv: pre-computed gene-stratified splits
    - sequences_json: optional pre-extracted sequences

    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    # Load data
    labels_df = pd.read_csv(config.labels_csv)
    negatives_df = pd.read_csv(config.negatives_csv)
    splits_df = pd.read_csv(config.splits_csv)

    # Load sequences if available
    sequences = None
    if config.sequences_json and Path(config.sequences_json).exists():
        with open(config.sequences_json) as f:
            sequences = json.load(f)
        logger.info("Loaded %d pre-extracted sequences", len(sequences))

    # For cached mode, filter to sites that have cached embeddings
    if config.encoder == "cached" and config.embeddings_dir:
        emb_ids_path = Path(config.embeddings_dir) / "rnafm_site_ids.json"
        if emb_ids_path.exists():
            with open(emb_ids_path) as f:
                cached_ids = set(json.load(f))
            n_before = len(labels_df)
            labels_df = labels_df[labels_df["site_id"].isin(cached_ids)]
            n_before_neg = len(negatives_df)
            negatives_df = negatives_df[negatives_df["site_id"].isin(cached_ids)]
            logger.info("Filtered to cached sites: %d->%d pos, %d->%d neg",
                        n_before, len(labels_df), n_before_neg, len(negatives_df))

    # Merge splits
    labels_df = labels_df.merge(splits_df, on="site_id", how="left")
    negatives_df = negatives_df.merge(splits_df[splits_df["site_id"].str.startswith("NEG")],
                                       on="site_id", how="left")

    # Build positive samples
    positive_samples = _build_positive_samples(labels_df, sequences, config.window_size)
    logger.info("Positive samples: %d", len(positive_samples))

    # Build negative samples (skip in positive_only mode)
    if config.positive_only:
        negative_samples = []
        logger.info("Positive-only mode: skipping negative samples")
    else:
        negative_samples = _build_negative_samples(negatives_df, sequences, config.window_size)
        logger.info("Negative samples: %d", len(negative_samples))

    # Assign splits
    all_samples = positive_samples + negative_samples
    site_to_split = dict(zip(splits_df["site_id"], splits_df["split"]))

    train_samples, val_samples, test_samples = [], [], []
    for s in all_samples:
        split = site_to_split.get(s.site_id, "train")
        if split == "val":
            val_samples.append(s)
        elif split == "test":
            test_samples.append(s)
        else:
            train_samples.append(s)

    data_config = APOBECDataConfig(window_size=config.window_size)

    logger.info(
        "Split sizes: train=%d, val=%d, test=%d",
        len(train_samples), len(val_samples), len(test_samples),
    )

    return (
        APOBECDataset(train_samples, data_config),
        APOBECDataset(val_samples, data_config),
        APOBECDataset(test_samples, data_config),
    )


def _build_positive_samples(
    df: pd.DataFrame,
    sequences: Optional[Dict[str, str]],
    window_size: int,
) -> List[APOBECSiteSample]:
    """Build positive editing site samples from labels DataFrame."""
    samples = []
    for _, row in df.iterrows():
        site_id = str(row["site_id"])

        # Get sequence
        seq = None
        if sequences and site_id in sequences:
            seq = sequences[site_id]
        else:
            seq = _generate_synthetic_sequence(window_size * 2 + 1)

        edit_pos = min(window_size, len(seq) // 2)

        # Ensure the edit position has a C
        seq_list = list(seq)
        seq_list[edit_pos] = "C"
        seq = "".join(seq_list)

        # --- PRIMARY ---
        # Editing rate: use pre-computed log2 column if available
        log2_rate = row.get("log2_max_rate", np.nan)
        if pd.isna(log2_rate):
            raw_rate = float(row.get("max_gtex_rate", np.nan))
            if not np.isnan(raw_rate):
                if raw_rate > 1.0:
                    raw_rate = raw_rate / 100.0
                log2_rate = np.log2(raw_rate + 0.01)
            else:
                log2_rate = np.nan
        else:
            log2_rate = float(log2_rate)

        apobec_class = encode_apobec_class(row.get("apobec_class", ""))

        # --- SECONDARY ---
        structure_type = encode_structure_type(row.get("structure_type", ""))
        tissue_spec_class = encode_tissue_spec(row.get("tissue_class", ""))

        n_raw = row.get("n_tissues_edited", np.nan)
        n_tissues_log2 = np.log2(float(n_raw)) if not pd.isna(n_raw) and float(n_raw) > 0 else np.nan

        # --- TERTIARY ---
        exonic_func = encode_exonic_function(row.get("exonic_function", ""))

        cv = row.get("any_mammalian_conservation", np.nan)
        if pd.isna(cv) or str(cv).strip() == "":
            conservation = float("nan")
        else:
            conservation = 1.0 if str(cv).strip().lower() in ("true", "1", "yes") else 0.0

        cs = row.get("has_survival_association", np.nan)
        if pd.isna(cs) or str(cs).strip() == "":
            cancer_survival = float("nan")
        else:
            cancer_survival = 1.0 if str(cs).strip().lower() in ("true", "1", "yes") else 0.0

        # --- AUXILIARY ---
        tissue_rates = np.full(N_TISSUES, np.nan, dtype=np.float32)

        hek_raw = row.get("hek293_rate", np.nan)
        if pd.isna(hek_raw):
            hek293_rate = float("nan")
        else:
            hek293_rate = float(hek_raw)
            if hek293_rate > 1.0:
                hek293_rate = hek293_rate / 100.0

        # --- Features ---
        flanking = get_flanking_context(seq, edit_pos)
        concordance = compute_concordance_features(
            row.get("structure_type_mRNA", ""),
            row.get("structure_type_premRNA", ""),
        )

        sample = APOBECSiteSample(
            sequence=seq,
            edit_pos=edit_pos,
            is_edited=1.0,
            editing_rate_log2=log2_rate,
            apobec_class=apobec_class,
            structure_type=structure_type,
            tissue_spec_class=tissue_spec_class,
            n_tissues_log2=n_tissues_log2,
            exonic_function=exonic_func,
            conservation=conservation,
            cancer_survival=cancer_survival,
            tissue_rates=tissue_rates,
            hek293_rate=hek293_rate,
            flanking_context=flanking,
            concordance_features=concordance,
            site_id=site_id,
            chrom=str(row.get("chr", "")),
            position=int(row.get("start", 0)),
            gene=str(row.get("gene_name", "")),
        )
        samples.append(sample)

    return samples


def _build_negative_samples(
    df: pd.DataFrame,
    sequences: Optional[Dict[str, str]],
    window_size: int,
) -> List[APOBECSiteSample]:
    """Build negative control samples from the CT mismatch DataFrame."""
    samples = []

    for _, row in df.iterrows():
        site_id = str(row["site_id"])

        # Get sequence
        seq = None
        if sequences and site_id in sequences:
            seq = sequences[site_id]
        else:
            seq = _generate_synthetic_sequence(window_size * 2 + 1)

        edit_pos = min(window_size, len(seq) // 2)

        # Ensure C at center
        seq_list = list(seq)
        seq_list[edit_pos] = "C"
        seq = "".join(seq_list)

        flanking = get_flanking_context(seq, edit_pos)

        # Concordance defaults for negatives
        concordance = np.zeros(5, dtype=np.float32)
        concordance[0] = 1.0  # assume concordant

        sample = APOBECSiteSample(
            sequence=seq,
            edit_pos=edit_pos,
            is_edited=0.0,
            editing_rate_log2=float("nan"),
            apobec_class=3,  # Neither
            structure_type=-1,
            tissue_spec_class=-1,
            n_tissues_log2=float("nan"),
            exonic_function=-1,
            conservation=float("nan"),
            cancer_survival=float("nan"),
            tissue_rates=np.full(N_TISSUES, np.nan, dtype=np.float32),
            hek293_rate=float("nan"),
            flanking_context=flanking,
            concordance_features=concordance,
            site_id=site_id,
            chrom=str(row.get("chr", "")),
            position=int(row.get("start", 0)),
            gene="",
        )
        samples.append(sample)

    return samples


def _generate_synthetic_sequence(length: int = 201) -> str:
    """Generate a synthetic RNA sequence for pipeline testing."""
    rng = np.random.RandomState(None)
    nucs = ["A", "C", "G", "U"]
    seq = "".join(rng.choice(nucs, size=length))
    # Ensure a C at the center
    center = length // 2
    seq_list = list(seq)
    seq_list[center] = "C"
    return "".join(seq_list)


def build_datasets_expanded(
    config: ExperimentConfig,
) -> Tuple[APOBECDataset, APOBECDataset, APOBECDataset]:
    """Build train/val/test datasets from expanded multi-dataset splits.

    Uses splits_expanded.csv which contains sites from all datasets
    (Levanon, Alqassim, Asaoka, Sharma) plus tiered negatives.

    For Levanon sites: merges with editing_sites_labels.csv for rich multi-task labels.
    For other sites: uses basic labels only (binary editing, editing rate).
    The multi-task loss handles NaN labels via masking.
    """
    splits_path = Path(config.splits_csv)
    if not splits_path.exists():
        raise FileNotFoundError(f"Expanded splits not found: {splits_path}")

    splits_df = pd.read_csv(splits_path)
    logger.info("Loaded expanded splits: %d sites", len(splits_df))

    # Load sequences
    sequences = None
    seq_path = Path(config.sequences_json) if config.sequences_json else (
        PROJECT_ROOT / "data" / "processed" / "site_sequences.json"
    )
    if seq_path.exists():
        with open(seq_path) as f:
            sequences = json.load(f)
        logger.info("Loaded %d sequences", len(sequences))

    # For cached mode, filter to sites with cached embeddings
    if config.encoder == "cached" and config.embeddings_dir:
        emb_ids_path = Path(config.embeddings_dir) / "rnafm_site_ids.json"
        if emb_ids_path.exists():
            with open(emb_ids_path) as f:
                cached_ids = set(json.load(f))
            n_before = len(splits_df)
            splits_df = splits_df[splits_df["site_id"].isin(cached_ids)]
            logger.info("Filtered to cached sites: %d->%d", n_before, len(splits_df))

    # Load rich labels for Levanon sites
    levanon_labels = {}
    if Path(config.labels_csv).exists():
        labels_df = pd.read_csv(config.labels_csv)
        levanon_labels = {row["site_id"]: row for _, row in labels_df.iterrows()}
        logger.info("Loaded %d Levanon site labels for enrichment", len(levanon_labels))

    # Build samples
    data_config = APOBECDataConfig(window_size=config.window_size)
    train_samples, val_samples, test_samples = [], [], []

    for _, row in splits_df.iterrows():
        site_id = str(row["site_id"])
        label = int(row["label"])
        split = row.get("split", "train")

        # Get sequence
        seq = None
        if sequences and site_id in sequences:
            seq = sequences[site_id]
        else:
            seq = _generate_synthetic_sequence(config.window_size * 2 + 1)

        edit_pos = min(config.window_size, len(seq) // 2)
        seq_list = list(seq)
        seq_list[edit_pos] = "C"
        seq = "".join(seq_list)

        flanking = get_flanking_context(seq, edit_pos)

        if label == 1 and site_id in levanon_labels:
            # Rich labels from Levanon dataset
            lrow = levanon_labels[site_id]
            sample = _build_enriched_positive(
                site_id, seq, edit_pos, flanking, row, lrow, config.window_size
            )
        elif label == 1:
            # Basic positive from other datasets
            editing_rate = row.get("editing_rate", np.nan)
            if pd.notna(editing_rate) and editing_rate > 0:
                if editing_rate > 1.0:
                    editing_rate = editing_rate / 100.0
                log2_rate = np.log2(editing_rate + 0.01)
            else:
                log2_rate = float("nan")

            sample = APOBECSiteSample(
                sequence=seq,
                edit_pos=edit_pos,
                is_edited=1.0,
                editing_rate_log2=log2_rate,
                apobec_class=-1,  # Unknown for non-Levanon
                structure_type=-1,
                tissue_spec_class=-1,
                n_tissues_log2=float("nan"),
                exonic_function=-1,
                conservation=float("nan"),
                cancer_survival=float("nan"),
                tissue_rates=np.full(N_TISSUES, np.nan, dtype=np.float32),
                hek293_rate=float("nan"),
                flanking_context=flanking,
                concordance_features=np.zeros(5, dtype=np.float32),
                site_id=site_id,
                chrom=str(row.get("chr", "")),
                position=int(row.get("start", 0)),
                gene=str(row.get("gene", "")),
            )
        else:
            # Negative sample
            concordance = np.zeros(5, dtype=np.float32)
            concordance[0] = 1.0

            sample = APOBECSiteSample(
                sequence=seq,
                edit_pos=edit_pos,
                is_edited=0.0,
                editing_rate_log2=float("nan"),
                apobec_class=3,  # Neither
                structure_type=-1,
                tissue_spec_class=-1,
                n_tissues_log2=float("nan"),
                exonic_function=-1,
                conservation=float("nan"),
                cancer_survival=float("nan"),
                tissue_rates=np.full(N_TISSUES, np.nan, dtype=np.float32),
                hek293_rate=float("nan"),
                flanking_context=flanking,
                concordance_features=concordance,
                site_id=site_id,
                chrom=str(row.get("chr", "")),
                position=int(row.get("start", 0)),
                gene=str(row.get("gene", "")),
            )

        if split == "val":
            val_samples.append(sample)
        elif split == "test":
            test_samples.append(sample)
        else:
            train_samples.append(sample)

    logger.info("Expanded dataset built: train=%d, val=%d, test=%d",
                len(train_samples), len(val_samples), len(test_samples))

    return (
        APOBECDataset(train_samples, data_config),
        APOBECDataset(val_samples, data_config),
        APOBECDataset(test_samples, data_config),
    )


def _build_enriched_positive(
    site_id: str,
    seq: str,
    edit_pos: int,
    flanking: np.ndarray,
    row: pd.Series,
    lrow: pd.Series,
    window_size: int,
) -> APOBECSiteSample:
    """Build a positive sample with full Levanon labels."""
    log2_rate = lrow.get("log2_max_rate", np.nan)
    if pd.isna(log2_rate):
        raw_rate = float(lrow.get("max_gtex_rate", np.nan))
        if not np.isnan(raw_rate):
            if raw_rate > 1.0:
                raw_rate = raw_rate / 100.0
            log2_rate = np.log2(raw_rate + 0.01)
        else:
            log2_rate = np.nan
    else:
        log2_rate = float(log2_rate)

    apobec_class = encode_apobec_class(lrow.get("apobec_class", ""))
    structure_type = encode_structure_type(lrow.get("structure_type", ""))
    tissue_spec_class = encode_tissue_spec(lrow.get("tissue_class", ""))

    n_raw = lrow.get("n_tissues_edited", np.nan)
    n_tissues_log2 = np.log2(float(n_raw)) if not pd.isna(n_raw) and float(n_raw) > 0 else np.nan

    exonic_func = encode_exonic_function(lrow.get("exonic_function", ""))

    cv = lrow.get("any_mammalian_conservation", np.nan)
    if pd.isna(cv) or str(cv).strip() == "":
        conservation = float("nan")
    else:
        conservation = 1.0 if str(cv).strip().lower() in ("true", "1", "yes") else 0.0

    cs = lrow.get("has_survival_association", np.nan)
    if pd.isna(cs) or str(cs).strip() == "":
        cancer_survival = float("nan")
    else:
        cancer_survival = 1.0 if str(cs).strip().lower() in ("true", "1", "yes") else 0.0

    tissue_rates = np.full(N_TISSUES, np.nan, dtype=np.float32)

    hek_raw = lrow.get("hek293_rate", np.nan)
    if pd.isna(hek_raw):
        hek293_rate = float("nan")
    else:
        hek293_rate = float(hek_raw)
        if hek293_rate > 1.0:
            hek293_rate = hek293_rate / 100.0

    concordance = compute_concordance_features(
        lrow.get("structure_type_mRNA", ""),
        lrow.get("structure_type_premRNA", ""),
    )

    return APOBECSiteSample(
        sequence=seq,
        edit_pos=edit_pos,
        is_edited=1.0,
        editing_rate_log2=log2_rate,
        apobec_class=apobec_class,
        structure_type=structure_type,
        tissue_spec_class=tissue_spec_class,
        n_tissues_log2=n_tissues_log2,
        exonic_function=exonic_func,
        conservation=conservation,
        cancer_survival=cancer_survival,
        tissue_rates=tissue_rates,
        hek293_rate=hek293_rate,
        flanking_context=flanking,
        concordance_features=concordance,
        site_id=site_id,
        chrom=str(row.get("chr", "")),
        position=int(row.get("start", 0)),
        gene=str(lrow.get("gene_name", "")),
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

class APOBECTrainer:
    """Training manager for the APOBEC editing prediction model.

    Handles:
    - Training with gradient accumulation
    - Validation with classification metrics (AUROC, AUPRC)
    - Early stopping on primary metric
    - Checkpoint saving
    - Experiment logging
    """

    def __init__(
        self,
        model: EditRNA_A3A,
        config: ExperimentConfig,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device

        # Optimizer with differential learning rates
        self.optimizer = AdamW(
            model.get_parameter_groups(),
            lr=config.learning_rate,  # default group lr
        )

        # LR scheduler
        if config.lr_scheduler == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="max" if config.primary_metric in ("auroc", "auprc", "f1") else "min",
                factor=0.5,
                patience=5,
                min_lr=1e-7,
            )
        elif config.lr_scheduler == "cosine_warm":
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=10,
                T_mult=2,
                eta_min=1e-7,
            )
        elif config.lr_scheduler == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=config.epochs,
                eta_min=1e-7,
            )
        else:
            self.scheduler = None

        # Early stopping state
        self.best_metric = -float("inf")
        self.best_epoch = 0
        self.patience_counter = 0
        self.best_state = None

        # Output directory
        self.output_dir = Path(config.output_dir) / config.name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # History
        self.history: Dict[str, List[float]] = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "val_auroc": [],
            "val_auprc": [],
            "val_f1": [],
            "val_rate_spearman": [],
            "val_enzyme_acc": [],
            "val_tissue_acc": [],
            "val_multitask_score": [],
            "lr": [],
        }

    def train(
        self,
        train_loader,
        val_loader,
    ) -> Dict[str, List[float]]:
        """Run the full training loop."""
        logger.info("Starting training: %s", self.config.name)
        logger.info("  Epochs: %d", self.config.epochs)
        logger.info("  Batch size: %d x %d accumulation = %d effective",
                     self.config.batch_size,
                     self.config.grad_accumulation_steps,
                     self.config.batch_size * self.config.grad_accumulation_steps)
        logger.info("  Learning rate: %.2e (encoder: %.2e)",
                     self.config.learning_rate,
                     self.config.learning_rate * self.config.encoder_lr_factor)
        logger.info("  Output: %s", self.output_dir)

        for epoch in range(1, self.config.epochs + 1):
            t0 = time.time()

            # Train one epoch
            train_metrics = self._train_epoch(train_loader, epoch)

            # Validate
            val_metrics = {}
            if epoch % self.config.eval_every == 0:
                val_metrics = self._validate(val_loader)

            # LR scheduling
            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    metric_val = val_metrics.get(self.config.primary_metric, 0)
                    self.scheduler.step(metric_val)
                else:
                    self.scheduler.step()

            # Record history
            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["val_loss"].append(val_metrics.get("loss", float("nan")))
            self.history["val_auroc"].append(val_metrics.get("auroc", float("nan")))
            self.history["val_auprc"].append(val_metrics.get("auprc", float("nan")))
            self.history["val_f1"].append(val_metrics.get("f1", float("nan")))
            self.history["val_rate_spearman"].append(val_metrics.get("rate_spearman", float("nan")))
            self.history["val_enzyme_acc"].append(val_metrics.get("enzyme_acc", float("nan")))
            self.history["val_tissue_acc"].append(val_metrics.get("tissue_acc", float("nan")))
            self.history["val_multitask_score"].append(val_metrics.get("multitask_score", float("nan")))
            self.history["lr"].append(current_lr)

            # Early stopping check
            primary_val = val_metrics.get(self.config.primary_metric, float("nan"))
            if not np.isnan(primary_val):
                if primary_val > self.best_metric + self.config.min_delta:
                    self.best_metric = primary_val
                    self.best_epoch = epoch
                    self.patience_counter = 0
                    self.best_state = {
                        k: v.cpu().clone() for k, v in self.model.state_dict().items()
                    }
                    self._save_checkpoint(epoch, is_best=True)
                else:
                    self.patience_counter += 1

            elapsed = time.time() - t0

            # Log
            log_parts = [f"Epoch {epoch:3d}/{self.config.epochs}"]
            log_parts.append(f"loss={train_metrics['loss']:.4f}")
            if val_metrics:
                log_parts.append(f"val_loss={val_metrics.get('loss', 0):.4f}")
                if self.config.positive_only:
                    log_parts.append(f"rate_rho={val_metrics.get('rate_spearman', 0):.4f}")
                    log_parts.append(f"enz_acc={val_metrics.get('enzyme_acc', 0):.4f}")
                    log_parts.append(f"tis_acc={val_metrics.get('tissue_acc', 0):.4f}")
                    log_parts.append(f"mt_score={val_metrics.get('multitask_score', 0):.4f}")
                else:
                    log_parts.append(f"AUROC={val_metrics.get('auroc', 0):.4f}")
                    log_parts.append(f"AUPRC={val_metrics.get('auprc', 0):.4f}")
                    log_parts.append(f"F1={val_metrics.get('f1', 0):.4f}")
            log_parts.append(f"lr={current_lr:.2e}")
            log_parts.append(f"patience={self.patience_counter}/{self.config.patience}")
            log_parts.append(f"{elapsed:.1f}s")
            logger.info("  ".join(log_parts))

            # Check early stopping
            if self.patience_counter >= self.config.patience:
                logger.info("Early stopping at epoch %d (best=%d, %s=%.4f)",
                            epoch, self.best_epoch,
                            self.config.primary_metric, self.best_metric)
                break

        # Restore best model
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
            logger.info("Restored best model from epoch %d (%s=%.4f)",
                        self.best_epoch, self.config.primary_metric, self.best_metric)

        # Save training history
        self._save_history()

        return self.history

    def _train_epoch(self, train_loader, epoch: int) -> Dict[str, float]:
        """Train for one epoch with gradient accumulation."""
        self.model.train()
        total_loss = 0.0
        task_loss_accum = {}
        n_batches = 0

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            # Move tensors to device
            batch = _batch_to_device(batch, self.device)

            # Forward
            output = self.model(batch)
            loss, task_losses = self.model.compute_loss(output, batch["targets"])

            # Scale for gradient accumulation
            loss = loss / self.config.grad_accumulation_steps
            loss.backward()

            total_loss += loss.item() * self.config.grad_accumulation_steps
            for k, v in task_losses.items():
                task_loss_accum[k] = task_loss_accum.get(k, 0) + v.item()
            n_batches += 1

            # Optimizer step every N accumulation steps
            if (batch_idx + 1) % self.config.grad_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()
                self.optimizer.zero_grad()

        # Final optimizer step for remaining gradients
        if n_batches % self.config.grad_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )
            self.optimizer.step()
            self.optimizer.zero_grad()

        avg_loss = total_loss / max(n_batches, 1)
        result = {"loss": avg_loss}
        for k, v in task_loss_accum.items():
            result[f"task_{k}"] = v / max(n_batches, 1)

        return result

    @torch.no_grad()
    def _validate(self, val_loader) -> Dict[str, float]:
        """Validate and compute classification/multi-task metrics."""
        self.model.eval()

        total_loss = 0.0
        all_binary_targets = []
        all_binary_scores = []
        all_rate_targets = []
        all_rate_preds = []
        all_enzyme_targets = []
        all_enzyme_logits = []
        all_tissue_targets = []
        all_tissue_logits = []
        n_batches = 0

        for batch in val_loader:
            batch = _batch_to_device(batch, self.device)
            output = self.model(batch)
            loss, task_losses = self.model.compute_loss(output, batch["targets"])

            total_loss += loss.item()
            n_batches += 1

            # Collect predictions for metrics
            preds = output["predictions"]

            # Binary classification
            binary_targets = batch["targets"]["binary"].cpu().numpy()
            binary_logits = preds["binary_logit"].squeeze(-1).cpu().numpy()
            binary_probs = 1.0 / (1.0 + np.exp(-binary_logits))  # sigmoid
            all_binary_targets.append(binary_targets)
            all_binary_scores.append(binary_probs)

            # Rate regression
            rate_targets = batch["targets"]["rate"].cpu().numpy()
            rate_preds = preds["editing_rate"].squeeze(-1).cpu().numpy()
            all_rate_targets.append(rate_targets)
            all_rate_preds.append(rate_preds)

            # Enzyme specificity
            all_enzyme_targets.append(batch["targets"]["enzyme"].cpu().numpy())
            all_enzyme_logits.append(preds["enzyme_logits"].cpu().numpy())

            # Tissue specificity
            all_tissue_targets.append(batch["targets"]["tissue_spec"].cpu().numpy())
            all_tissue_logits.append(preds["tissue_spec_logits"].cpu().numpy())

        # Concatenate
        all_binary_targets = np.concatenate(all_binary_targets)
        all_binary_scores = np.concatenate(all_binary_scores)
        all_rate_targets = np.concatenate(all_rate_targets)
        all_rate_preds = np.concatenate(all_rate_preds)
        all_enzyme_targets = np.concatenate(all_enzyme_targets)
        all_enzyme_logits = np.concatenate(all_enzyme_logits)
        all_tissue_targets = np.concatenate(all_tissue_targets)
        all_tissue_logits = np.concatenate(all_tissue_logits)

        # Classification metrics
        cls_metrics = compute_classification_metrics(all_binary_targets, all_binary_scores)

        # Rate regression metrics (only for positive sites with valid rates)
        rate_mask = ~np.isnan(all_rate_targets)
        rate_corr = float("nan")
        if rate_mask.sum() > 5:
            from scipy.stats import spearmanr
            corr, _ = spearmanr(all_rate_targets[rate_mask], all_rate_preds[rate_mask])
            rate_corr = corr

        # Enzyme accuracy (valid targets only)
        from sklearn.metrics import accuracy_score, f1_score
        enzyme_mask = all_enzyme_targets >= 0
        enzyme_acc = float("nan")
        if enzyme_mask.sum() > 0:
            enzyme_pred = np.argmax(all_enzyme_logits[enzyme_mask], axis=1)
            enzyme_acc = accuracy_score(all_enzyme_targets[enzyme_mask], enzyme_pred)

        # Tissue spec accuracy
        tissue_mask = all_tissue_targets >= 0
        tissue_acc = float("nan")
        if tissue_mask.sum() > 0:
            tissue_pred = np.argmax(all_tissue_logits[tissue_mask], axis=1)
            tissue_acc = accuracy_score(all_tissue_targets[tissue_mask], tissue_pred)

        result = {
            "loss": total_loss / max(n_batches, 1),
            "auroc": cls_metrics["auroc"],
            "auprc": cls_metrics["auprc"],
            "f1": cls_metrics["f1"],
            "precision": cls_metrics["precision"],
            "recall": cls_metrics["recall"],
            "accuracy": cls_metrics["accuracy"],
            "rate_spearman": rate_corr,
            "enzyme_acc": enzyme_acc,
            "tissue_acc": tissue_acc,
        }

        # Composite multi-task score for positive-only training
        # Average of normalized task metrics (each in ~[0,1])
        scores = []
        if not np.isnan(rate_corr):
            scores.append(max(0, rate_corr))  # Spearman can be negative
        if not np.isnan(enzyme_acc):
            scores.append(enzyme_acc)
        if not np.isnan(tissue_acc):
            scores.append(tissue_acc)
        result["multitask_score"] = float(np.mean(scores)) if scores else float("nan")

        return result

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_metric": self.best_metric,
            "config": asdict(self.config),
        }
        path = self.output_dir / f"checkpoint_epoch{epoch}.pt"
        torch.save(ckpt, path)

        if is_best:
            best_path = self.output_dir / "best_model.pt"
            torch.save(ckpt, best_path)

    def _save_history(self):
        """Save training history as JSON and CSV."""
        history_path = self.output_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, float)) else x)

        # Also save as CSV for easy plotting
        csv_path = self.output_dir / "training_history.csv"
        pd.DataFrame(self.history).to_csv(csv_path, index=False)

    @torch.no_grad()
    def evaluate(self, test_loader) -> Dict[str, float]:
        """Final evaluation on test set."""
        logger.info("=== Final Evaluation on Test Set ===")
        metrics = self._validate(test_loader)

        # Log results
        logger.info("  AUROC:     %.4f", metrics["auroc"])
        logger.info("  AUPRC:     %.4f", metrics["auprc"])
        logger.info("  F1:        %.4f", metrics["f1"])
        logger.info("  Precision: %.4f", metrics["precision"])
        logger.info("  Recall:    %.4f", metrics["recall"])
        logger.info("  Accuracy:  %.4f", metrics["accuracy"])
        logger.info("  Rate rho:  %.4f", metrics["rate_spearman"])

        # Save test results
        results_path = self.output_dir / "test_results.json"
        with open(results_path, "w") as f:
            json.dump(metrics, f, indent=2, default=lambda x: float(x))

        return metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _batch_to_device(batch: Dict, device: torch.device) -> Dict:
    """Move a collated batch to the specified device."""
    result = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            result[k] = v.to(device)
        elif isinstance(v, dict):
            result[k] = {kk: vv.to(device) if isinstance(vv, torch.Tensor) else vv
                         for kk, vv in v.items()}
        else:
            result[k] = v  # lists (sequences, site_ids)
    return result


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(config: ExperimentConfig, device: torch.device) -> EditRNA_A3A:
    """Build the EditRNA_A3A model from experiment config."""
    if config.use_mock_encoder or config.encoder == "mock":
        model = create_editrna_mock(
            d_model=config.d_model,
            d_edit=config.d_edit,
            d_fused=config.d_fused,
            head_dropout=config.head_dropout,
        )
    elif config.encoder == "cached":
        # Load pre-computed RNA-FM embeddings
        emb_dir = Path(config.embeddings_dir)
        logger.info("Loading cached embeddings from %s", emb_dir)
        tokens_cache = torch.load(emb_dir / "rnafm_tokens.pt", weights_only=False)
        pooled_cache = torch.load(emb_dir / "rnafm_pooled.pt", weights_only=False)
        tokens_edited = torch.load(emb_dir / "rnafm_tokens_edited.pt", weights_only=False)
        pooled_edited = torch.load(emb_dir / "rnafm_pooled_edited.pt", weights_only=False)
        logger.info("  Loaded %d original + %d edited embeddings",
                     len(tokens_cache), len(tokens_edited))

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
            d_edit=config.d_edit,
            d_fused=config.d_fused,
            edit_n_heads=config.edit_n_heads,
            use_structure_delta=config.use_structure_delta,
            use_dual_encoder=config.use_dual_encoder,
            use_gnn=config.use_gnn,
            finetune_last_n=0,
            head_dropout=config.head_dropout,
            fusion_dropout=config.fusion_dropout,
            focal_gamma=config.focal_gamma,
            focal_alpha_binary=config.focal_alpha_binary,
            focal_alpha_conservation=config.focal_alpha_conservation,
            learning_rate=config.learning_rate,
            encoder_lr_factor=config.encoder_lr_factor,
            weight_decay=config.weight_decay,
        )
        model = EditRNA_A3A(config=model_config, primary_encoder=cached_encoder)
    else:
        model_config = EditRNAConfig(
            primary_encoder=config.encoder,
            d_model=config.d_model,
            d_edit=config.d_edit,
            d_fused=config.d_fused,
            edit_n_heads=config.edit_n_heads,
            use_structure_delta=config.use_structure_delta,
            use_dual_encoder=config.use_dual_encoder,
            use_gnn=config.use_gnn,
            finetune_last_n=config.finetune_last_n,
            head_dropout=config.head_dropout,
            fusion_dropout=config.fusion_dropout,
            focal_gamma=config.focal_gamma,
            focal_alpha_binary=config.focal_alpha_binary,
            focal_alpha_conservation=config.focal_alpha_conservation,
            learning_rate=config.learning_rate,
            encoder_lr_factor=config.encoder_lr_factor,
            weight_decay=config.weight_decay,
        )
        model = EditRNA_A3A(config=model_config)

    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> ExperimentConfig:
    """Parse CLI arguments into ExperimentConfig."""
    parser = argparse.ArgumentParser(
        description="Train APOBEC C-to-U editing prediction model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Key args
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--encoder", type=str, default="rnafm",
                        choices=["rnafm", "utrlm", "mock", "cached"])
    parser.add_argument("--mock", action="store_true",
                        help="Use mock encoder for pipeline testing")
    parser.add_argument("--cached", action="store_true",
                        help="Use pre-computed RNA-FM embeddings for fast training")
    parser.add_argument("--embeddings_dir", type=str,
                        default=str(PROJECT_ROOT / "data" / "processed" / "embeddings"),
                        help="Directory with cached rnafm_*.pt files")
    parser.add_argument("--sequences_json", type=str, default="")
    parser.add_argument("--negatives_csv", type=str, default="")
    parser.add_argument("--splits_csv", type=str, default="")
    parser.add_argument("--expanded", action="store_true",
                        help="Use expanded multi-dataset splits (splits_expanded.csv)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--finetune_last_n", type=int, default=0)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--d_edit", type=int, default=256)
    parser.add_argument("--d_fused", type=int, default=512)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--focal_alpha_binary", type=float, default=0.75)
    parser.add_argument("--focal_alpha_conservation", type=float, default=0.85)
    parser.add_argument("--grad_accumulation_steps", type=int, default=2)
    parser.add_argument("--lr_scheduler", type=str, default="cosine_warm",
                        choices=["plateau", "cosine_warm", "cosine"])
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--positive_only", action="store_true",
                        help="Train on positive sites only (no negatives)")
    parser.add_argument("--primary_metric", type=str, default=None,
                        choices=["auroc", "auprc", "f1", "multitask_score"],
                        help="Metric for early stopping / model selection")

    args = parser.parse_args()

    config = ExperimentConfig()

    # Apply CLI overrides
    if args.cached:
        config.encoder = "cached"
        config.embeddings_dir = args.embeddings_dir
    elif args.mock:
        config.use_mock_encoder = True
        config.encoder = "mock"
    if args.encoder and not args.cached and not args.mock:
        config.encoder = args.encoder
    if args.embeddings_dir:
        config.embeddings_dir = args.embeddings_dir
    if args.name:
        config.name = args.name
    elif args.cached:
        config.name = "exp1_rnafm_cached"
    elif args.mock:
        config.name = "apobec_mock_test"
    config.sequences_json = args.sequences_json
    if args.negatives_csv:
        config.negatives_csv = args.negatives_csv
    if args.splits_csv:
        config.splits_csv = args.splits_csv
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.finetune_last_n = args.finetune_last_n
    config.patience = args.patience
    config.seed = args.seed
    config.d_edit = args.d_edit
    config.d_fused = args.d_fused
    config.focal_gamma = args.focal_gamma
    config.focal_alpha_binary = args.focal_alpha_binary
    config.focal_alpha_conservation = args.focal_alpha_conservation
    config.grad_accumulation_steps = args.grad_accumulation_steps
    config.lr_scheduler = args.lr_scheduler
    config.expanded = args.expanded
    config.positive_only = args.positive_only
    if args.positive_only and args.primary_metric is None:
        config.primary_metric = "multitask_score"
    if args.primary_metric is not None:
        config.primary_metric = args.primary_metric
    if args.expanded and not args.splits_csv:
        config.splits_csv = str(PROJECT_ROOT / "data" / "processed" / "splits_expanded.csv")

    if args.d_model is not None:
        config.d_model = args.d_model
    elif config.encoder == "utrlm":
        config.d_model = 128
    elif config.encoder == "mock":
        config.d_model = 640

    if args.output_dir:
        config.output_dir = args.output_dir

    return config


def main():
    """Main entry point."""
    config = parse_args()
    set_seed(config.seed)

    # Create output directory first
    out_dir = Path(config.output_dir) / config.name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(out_dir / "train.log", mode="w"),
        ],
    )

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Save config
    config_path = out_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(asdict(config), f, indent=2)
    logger.info("Config saved to %s", config_path)

    # Build datasets
    logger.info("Building datasets...")
    if config.expanded:
        logger.info("Using expanded multi-dataset splits")
        train_dataset, val_dataset, test_dataset = build_datasets_expanded(config)
    else:
        train_dataset, val_dataset, test_dataset = build_datasets_from_csvs(config)

    logger.info("  Train: %d samples (%d pos, %d neg, ratio=%.1f)",
                len(train_dataset), train_dataset.n_positive, train_dataset.n_negative,
                train_dataset.class_ratio)
    logger.info("  Val:   %d samples (%d pos, %d neg)",
                len(val_dataset), val_dataset.n_positive, val_dataset.n_negative)
    logger.info("  Test:  %d samples (%d pos, %d neg)",
                len(test_dataset), test_dataset.n_positive, test_dataset.n_negative)

    # Create DataLoaders
    loaders = create_apobec_dataloaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    # Build model
    logger.info("Building model (encoder=%s)...", config.encoder)
    model = build_model(config, device)

    # Parameter counts
    param_counts = model.count_parameters()
    for component, counts in param_counts.items():
        logger.info("  %s: %s total, %s trainable",
                     component,
                     f"{counts['total']:,}",
                     f"{counts['trainable']:,}")

    # Train
    trainer = APOBECTrainer(model, config, device)
    history = trainer.train(loaders["train"], loaders["val"])

    # Final evaluation on test set
    test_metrics = trainer.evaluate(loaders["test"])

    logger.info("Training complete. Results saved to %s", out_dir)

    return test_metrics


if __name__ == "__main__":
    main()
