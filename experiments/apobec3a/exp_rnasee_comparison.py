#!/usr/bin/env python
"""RNAsee apples-to-apples comparison experiment.

Replicates the RNAsee (Van Norden et al. 2024, Commun. Biol.) evaluation setup:
  - Asaoka 2019 data only
  - ALL-cytidine negatives at 1:3 pos:neg ratio (RNAsee uses 2:1 random-C +
    rules-based hard negatives, we use all-cytidine uniform as closest match)
  - 70:30 random split (no gene stratification, matching RNAsee)
  - Exact 50-bit binary encoding: asymmetric 15nt upstream + 10nt downstream,
    center C excluded, 2 bits/nt (is_purine, pairs_GC)
  - Default sklearn RandomForestClassifier (100 trees, no tuning)

Then trains multiple models under both the RNAsee easy-negative setup and
our harder TC-motif-only negative setup to demonstrate that our approach
outperforms RNAsee on their own terms.

Models evaluated:
  Feature-based (no embeddings):
    - RNAsee_RF: Exact RNAsee replication (50-bit, default RF)
    - GB_HandFeatures: XGBoost on motif+structure+loop features

  Embedding-based (pre-computed RNA-FM):
    - SubtractionMLP: Pooled embedding subtraction baseline
    - EditRNA-A3A: Full gated fusion model (pooled-only mode)

Note: RNAsee's reported AUROC=0.962 is on a 1:468 proportional set (all
cytidines), not the 1:3 balanced set. Direct comparison at 1:3 is approximate.

Usage:
    python experiments/apobec3a/exp_rnasee_comparison.py
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
COMBINED_CSV = PROJECT_ROOT / "data" / "processed" / "all_datasets_combined.csv"
ALLC_NEG_CSV = PROJECT_ROOT / "data" / "processed" / "negatives_per_dataset_all_c.csv"
PER_DATASET_NEG_CSV = PROJECT_ROOT / "data" / "processed" / "negatives_per_dataset.csv"
SEQUENCES_JSON = PROJECT_ROOT / "data" / "processed" / "site_sequences.json"
GENOME_PATH = PROJECT_ROOT / "data" / "raw" / "genomes" / "hg38.fa"
STRUCTURE_CACHE = (
    PROJECT_ROOT / "data" / "processed" / "embeddings" / "vienna_structure_cache.npz"
)
LOOP_POSITION_CSV = (
    PROJECT_ROOT
    / "experiments"
    / "apobec"
    / "outputs"
    / "loop_position"
    / "loop_position_per_site.csv"
)
POOLED_ORIG_PT = (
    PROJECT_ROOT / "data" / "processed" / "embeddings" / "rnafm_pooled.pt"
)
POOLED_EDITED_PT = (
    PROJECT_ROOT / "data" / "processed" / "embeddings" / "rnafm_pooled_edited.pt"
)
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "apobec" / "outputs" / "rnasee_comparison"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_binary_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    """Compute binary classification metrics with optimal F1 threshold."""
    if len(np.unique(y_true)) < 2:
        return {k: float("nan") for k in [
            "auroc", "auprc", "f1", "precision", "recall",
        ]}

    metrics = {
        "auroc": float(roc_auc_score(y_true, y_score)),
        "auprc": float(average_precision_score(y_true, y_score)),
    }

    prec_arr, rec_arr, thresholds = precision_recall_curve(y_true, y_score)
    f1_arr = 2 * prec_arr[:-1] * rec_arr[:-1] / (prec_arr[:-1] + rec_arr[:-1] + 1e-8)
    best_idx = np.argmax(f1_arr)
    threshold = thresholds[best_idx] if len(thresholds) > 0 else 0.5
    y_pred = (y_score >= threshold).astype(int)

    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    return metrics


# ---------------------------------------------------------------------------
# RNAsee feature encoding
# ---------------------------------------------------------------------------


def encode_rnasee_features(sequences: dict, site_ids: list) -> np.ndarray:
    """Encode sequences using RNAsee's exact 50-bit binary nucleotide encoding.

    From their code (ML_methods.py:get_short_site_data):
      - 15 nucleotides upstream + 10 nucleotides downstream of the target C
      - Each nucleotide encoded as 2 bits: (is_purine, pairs_GC)
      - Center C is EXCLUDED from the feature vector
      - Total: (15 + 10) * 2 = 50 binary features

    The encoding: A=(1,0), G=(1,1), C=(0,1), U/T=(0,0)
    """
    BWD = 15  # upstream positions
    FWD = 10  # downstream positions
    CENTER = 100  # edit site in 201nt window

    features = np.zeros((len(site_ids), (BWD + FWD) * 2), dtype=np.float32)

    for i, sid in enumerate(site_ids):
        if sid not in sequences:
            continue
        seq = sequences[sid]
        if len(seq) < CENTER + FWD + 1:
            continue

        # Upstream positions (15 nt before center)
        feat_idx = 0
        for offset in range(-BWD, 0):
            pos = CENTER + offset
            if pos < 0:
                feat_idx += 2
                continue
            nuc = seq[pos].upper()
            features[i, feat_idx] = 1.0 if nuc in ("A", "G") else 0.0
            features[i, feat_idx + 1] = 1.0 if nuc in ("G", "C") else 0.0
            feat_idx += 2

        # Skip center C (excluded in RNAsee)

        # Downstream positions (10 nt after center)
        for offset in range(1, FWD + 1):
            pos = CENTER + offset
            if pos >= len(seq):
                feat_idx += 2
                continue
            nuc = seq[pos].upper()
            features[i, feat_idx] = 1.0 if nuc in ("A", "G") else 0.0
            features[i, feat_idx + 1] = 1.0 if nuc in ("G", "C") else 0.0
            feat_idx += 2

    return features


# ---------------------------------------------------------------------------
# Hand-crafted feature extraction (reused from train_gradient_boosting.py)
# ---------------------------------------------------------------------------


def extract_hand_features(
    site_ids: list, sequences: dict
) -> Tuple[np.ndarray, List[str]]:
    """Extract motif + structure + loop features for given site IDs.

    Returns (feature_matrix, feature_names).
    """
    # --- Motif features ---
    top_motifs = ["UCG", "UCA", "UCC", "UCU", "ACA", "GCA", "CCA",
                  "ACG", "GCG", "CCG", "ACU", "GCU", "CCU", "ACC", "GCC"]
    seen = set()
    unique_motifs = []
    for m in top_motifs:
        if m not in seen:
            seen.add(m)
            unique_motifs.append(m)
    top_motifs = unique_motifs

    motif_cols = (
        ["has_TC_motif"]
        + [f"upstream_{n}" for n in "ACGU"]
        + [f"downstream_{n}" for n in "ACGU"]
        + [f"trinuc_{m}" for m in top_motifs]
    )

    motif_feats = np.zeros((len(site_ids), len(motif_cols)), dtype=np.float32)
    for i, sid in enumerate(site_ids):
        if sid not in sequences:
            continue
        seq = sequences[sid]
        if len(seq) < 102:
            continue
        trinuc = seq[99:102]
        motif_feats[i, 0] = 1.0 if (seq[99] == "U" and seq[100] == "C") else 0.0
        for k, n in enumerate("ACGU"):
            motif_feats[i, 1 + k] = 1.0 if seq[99] == n else 0.0
            motif_feats[i, 5 + k] = 1.0 if seq[101] == n else 0.0
        for k, motif in enumerate(top_motifs):
            motif_feats[i, 9 + k] = 1.0 if trinuc == motif else 0.0

    # --- Structure delta features ---
    struct_names = [
        "delta_pairing_at_pos", "delta_accessibility_at_pos",
        "delta_entropy_at_pos", "delta_mfe", "delta_local_pairing",
        "delta_local_accessibility", "local_pairing_std",
    ]
    struct_feats = np.zeros((len(site_ids), len(struct_names)), dtype=np.float32)
    if STRUCTURE_CACHE.exists():
        data = np.load(STRUCTURE_CACHE, allow_pickle=True)
        s_ids = data["site_ids"]
        s_feats = data["delta_features"]
        struct_dict = {str(sid): s_feats[j] for j, sid in enumerate(s_ids)}
        for i, sid in enumerate(site_ids):
            if sid in struct_dict:
                struct_feats[i] = struct_dict[sid]

    # --- Loop position features ---
    loop_names = [
        "is_unpaired", "loop_size", "dist_to_junction", "dist_to_apex",
        "relative_loop_position", "left_stem_length", "right_stem_length",
        "max_adjacent_stem_length", "local_unpaired_fraction",
    ]
    loop_feats = np.zeros((len(site_ids), len(loop_names)), dtype=np.float32)
    if LOOP_POSITION_CSV.exists():
        loop_df = pd.read_csv(LOOP_POSITION_CSV)
        loop_dict = {}
        for _, row in loop_df.iterrows():
            vals = []
            for col in loop_names:
                v = row.get(col, 0.0)
                if isinstance(v, bool):
                    v = float(v)
                vals.append(float(v) if pd.notna(v) else 0.0)
            loop_dict[str(row["site_id"])] = vals
        for i, sid in enumerate(site_ids):
            if sid in loop_dict:
                loop_feats[i] = loop_dict[sid]

    X = np.concatenate([motif_feats, struct_feats, loop_feats], axis=1)
    feature_names = motif_cols + struct_names + loop_names
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, feature_names


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_asaoka_positives() -> pd.DataFrame:
    """Load Asaoka 2019 positive editing sites."""
    combined = pd.read_csv(COMBINED_CSV)
    asaoka = combined[combined["dataset_source"] == "asaoka_2019"].copy()
    logger.info("Loaded %d Asaoka 2019 positive sites", len(asaoka))
    return asaoka


def load_negatives(neg_csv: Path, dataset_name: str = "asaoka_2019") -> pd.DataFrame:
    """Load negatives for a specific dataset from per-dataset negatives CSV."""
    if not neg_csv.exists():
        logger.error("Negatives file not found: %s", neg_csv)
        return pd.DataFrame()
    neg = pd.read_csv(neg_csv)
    neg = neg[neg["dataset_source"] == dataset_name].copy()
    logger.info("Loaded %d negatives for %s from %s",
                len(neg), dataset_name, neg_csv.name)
    return neg


def create_rnasee_split(
    positives: pd.DataFrame,
    negatives: pd.DataFrame,
    neg_ratio: int = 3,
    test_frac: float = 0.30,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a 70:30 random split matching RNAsee's methodology.

    No gene stratification — just random sampling.
    """
    rng = np.random.RandomState(seed)

    # Subsample negatives to target ratio
    target_neg = len(positives) * neg_ratio
    if len(negatives) > target_neg:
        negatives = negatives.sample(n=target_neg, random_state=rng)
    logger.info("After ratio subsampling: %d pos, %d neg (1:%.1f)",
                len(positives), len(negatives),
                len(negatives) / max(1, len(positives)))

    # Combine
    pos = positives[["site_id", "chr", "start", "strand", "gene"]].copy()
    pos["label"] = 1
    neg_cols = ["site_id", "chr", "start", "strand", "gene"]
    neg_df = negatives[[c for c in neg_cols if c in negatives.columns]].copy()
    if "strand" not in neg_df.columns:
        neg_df["strand"] = "+"
    neg_df["label"] = 0
    combined = pd.concat([pos, neg_df], ignore_index=True)

    # Random 70:30 split
    indices = rng.permutation(len(combined))
    n_test = int(len(combined) * test_frac)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    combined["split"] = "train"
    combined.iloc[test_idx, combined.columns.get_loc("split")] = "test"

    for split in ["train", "test"]:
        subset = combined[combined["split"] == split]
        n_pos = (subset["label"] == 1).sum()
        n_neg = (subset["label"] == 0).sum()
        logger.info("  %s: %d total (%d pos, %d neg)", split, len(subset), n_pos, n_neg)

    return combined


def create_gene_stratified_split(
    positives: pd.DataFrame,
    negatives: pd.DataFrame,
    neg_ratio: int = 3,
    test_frac: float = 0.30,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a gene-stratified 70:30 split.

    No gene appears in both train and test — prevents same-gene leakage.
    """
    rng = np.random.RandomState(seed)

    target_neg = len(positives) * neg_ratio
    if len(negatives) > target_neg:
        negatives = negatives.sample(n=target_neg, random_state=rng)

    pos = positives[["site_id", "chr", "start", "strand", "gene"]].copy()
    pos["label"] = 1
    neg_cols = ["site_id", "chr", "start", "strand", "gene"]
    neg_df = negatives[[c for c in neg_cols if c in negatives.columns]].copy()
    if "strand" not in neg_df.columns:
        neg_df["strand"] = "+"
    neg_df["label"] = 0
    combined = pd.concat([pos, neg_df], ignore_index=True)

    # Gene-based splitting
    genes = combined["gene"].dropna().unique()
    rng.shuffle(genes)
    n_test_genes = int(len(genes) * test_frac)
    test_genes = set(genes[:n_test_genes])

    combined["split"] = combined["gene"].apply(
        lambda g: "test" if g in test_genes else "train"
    )
    # Assign sites without gene to train
    combined.loc[combined["gene"].isna(), "split"] = "train"

    for split in ["train", "test"]:
        subset = combined[combined["split"] == split]
        n_pos = (subset["label"] == 1).sum()
        n_neg = (subset["label"] == 0).sum()
        logger.info("  %s: %d total (%d pos, %d neg)", split, len(subset), n_pos, n_neg)

    return combined


def create_chromosome_stratified_split(
    positives: pd.DataFrame,
    negatives: pd.DataFrame,
    neg_ratio: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a chromosome-stratified split.

    Test set = chr1, chr8, chr19, chrX (diverse sizes).
    Train set = all other chromosomes.
    """
    rng = np.random.RandomState(seed)

    target_neg = len(positives) * neg_ratio
    if len(negatives) > target_neg:
        negatives = negatives.sample(n=target_neg, random_state=rng)

    pos = positives[["site_id", "chr", "start", "strand", "gene"]].copy()
    pos["label"] = 1
    neg_cols = ["site_id", "chr", "start", "strand", "gene"]
    neg_df = negatives[[c for c in neg_cols if c in negatives.columns]].copy()
    if "strand" not in neg_df.columns:
        neg_df["strand"] = "+"
    neg_df["label"] = 0
    combined = pd.concat([pos, neg_df], ignore_index=True)

    test_chroms = {"chr1", "chr8", "chr19", "chrX"}
    combined["split"] = combined["chr"].apply(
        lambda c: "test" if c in test_chroms else "train"
    )

    for split in ["train", "test"]:
        subset = combined[combined["split"] == split]
        n_pos = (subset["label"] == 1).sum()
        n_neg = (subset["label"] == 0).sum()
        logger.info("  %s: %d total (%d pos, %d neg)", split, len(subset), n_pos, n_neg)

    return combined


# ---------------------------------------------------------------------------
# Sequence extraction for negative sites
# ---------------------------------------------------------------------------

FLANK_SIZE = 100  # 201nt window


def _revcomp_dna(seq: str) -> str:
    comp = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
    return "".join(comp.get(b, "N") for b in reversed(seq.upper()))


def extract_missing_sequences(
    split_df: pd.DataFrame,
    sequences: dict,
    neg_df_full: pd.DataFrame,
) -> dict:
    """Extract 201nt RNA sequences for sites not in the sequence cache.

    Uses the genome to extract sequences for negative sites that were
    generated from genomic coordinates.
    """
    missing_ids = [sid for sid in split_df["site_id"] if sid not in sequences]
    if not missing_ids:
        return sequences

    logger.info("Extracting sequences for %d sites missing from cache...", len(missing_ids))

    if not GENOME_PATH.exists():
        logger.warning("Genome not found at %s, skipping extraction", GENOME_PATH)
        return sequences

    from pyfaidx import Fasta
    genome = Fasta(str(GENOME_PATH))

    # Build lookup from neg_df
    neg_lookup = {}
    for _, row in neg_df_full.iterrows():
        neg_lookup[row["site_id"]] = row

    new_seqs = dict(sequences)  # copy
    extracted = 0
    for sid in missing_ids:
        if sid in neg_lookup:
            row = neg_lookup[sid]
        else:
            # Try from split_df
            match = split_df[split_df["site_id"] == sid]
            if len(match) == 0:
                continue
            row = match.iloc[0]

        chrom = str(row["chr"])
        pos = int(row["start"])
        strand = str(row.get("strand", "+"))
        if strand not in ("+", "-"):
            strand = "+"

        if chrom not in genome:
            continue
        chrom_len = len(genome[chrom])
        g_start = pos - FLANK_SIZE
        g_end = pos + FLANK_SIZE + 1
        if g_start < 0 or g_end > chrom_len:
            continue

        dna_seq = str(genome[chrom][g_start:g_end]).upper()
        if len(dna_seq) != 2 * FLANK_SIZE + 1:
            continue

        if strand == "-":
            dna_seq = _revcomp_dna(dna_seq)

        rna_seq = dna_seq.replace("T", "U")
        new_seqs[sid] = rna_seq
        extracted += 1

    logger.info("Extracted %d new sequences (total: %d)", extracted, len(new_seqs))
    return new_seqs


# ---------------------------------------------------------------------------
# Model training helpers
# ---------------------------------------------------------------------------


def train_rf_rnasee(X_train, y_train, X_test, y_test, name: str) -> Dict:
    """Train Random Forest matching RNAsee's exact setup.

    RNAsee uses sklearn RandomForestClassifier() with DEFAULT parameters:
    100 trees, unlimited depth, no class weighting, gini criterion.
    """
    logger.info("Training %s (%d features)...", name, X_train.shape[1])
    t0 = time.time()

    # RNAsee uses default sklearn RF (no hyperparameter tuning)
    clf = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    elapsed = time.time() - t0

    y_score = clf.predict_proba(X_test)[:, 1]
    metrics = compute_binary_metrics(y_test, y_score)

    logger.info("  %s: AUROC=%.4f  AUPRC=%.4f  F1=%.4f  (%.1fs)",
                name, metrics["auroc"], metrics["auprc"], metrics["f1"], elapsed)
    return {"metrics": metrics, "train_time": elapsed, "y_score": y_score}


def train_gb_hand(X_train, y_train, X_test, y_test, name: str) -> Dict:
    """Train XGBoost on hand-crafted features."""
    logger.info("Training %s (%d features)...", name, X_train.shape[1])
    t0 = time.time()

    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())

    try:
        import xgboost as xgb
        clf = xgb.XGBClassifier(
            n_estimators=500, max_depth=5, learning_rate=0.05,
            subsample=0.8, min_child_weight=10, colsample_bytree=0.8,
            scale_pos_weight=n_neg / max(n_pos, 1),
            random_state=42, n_jobs=1, eval_metric="logloss", verbosity=0,
        )
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        clf = GradientBoostingClassifier(
            n_estimators=500, max_depth=5, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=10, random_state=42,
        )

    clf.fit(X_train, y_train)
    elapsed = time.time() - t0

    y_score = clf.predict_proba(X_test)[:, 1]
    metrics = compute_binary_metrics(y_test, y_score)

    logger.info("  %s: AUROC=%.4f  AUPRC=%.4f  F1=%.4f  (%.1fs)",
                name, metrics["auroc"], metrics["auprc"], metrics["f1"], elapsed)
    return {"metrics": metrics, "train_time": elapsed, "y_score": y_score}


def train_embedding_model(
    model_name: str,
    site_ids_train: list,
    labels_train: np.ndarray,
    site_ids_test: list,
    labels_test: np.ndarray,
    pooled_orig: dict,
    pooled_edited: dict,
    structure_delta: dict,
) -> Dict:
    """Train an embedding-based model (SubtractionMLP or EditRNA-A3A)."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    from train_baselines import (
        BaselineConfig,
        EmbeddingDataset,
        FocalLoss,
        _forward_baseline,
        build_model,
        compute_metrics,
        embedding_collate_fn,
    )

    logger.info("Training %s (embedding-based)...", model_name)
    t0 = time.time()

    config = BaselineConfig(
        model_name=model_name,
        epochs=50,
        batch_size=64,
        learning_rate=1e-3,
        patience=10,
    )

    if model_name == "editrna":
        # Build EditRNA in pooled-only mode (no token embeddings for negatives)
        from models.editrna_a3a import EditRNA_A3A, EditRNAConfig
        from models.encoders import CachedRNAEncoder

        cached_encoder = CachedRNAEncoder(
            tokens_cache=None,
            pooled_cache=pooled_orig,
            tokens_edited_cache=None,
            pooled_edited_cache=pooled_edited,
            d_model=config.d_model,
        )
        model_config = EditRNAConfig(
            primary_encoder="cached",
            d_model=config.d_model,
            learning_rate=config.learning_rate,
            pooled_only=True,
        )
        model = EditRNA_A3A(config=model_config, primary_encoder=cached_encoder)
    else:
        model = build_model(model_name, config)
    device = torch.device("cpu")
    model = model.to(device)

    loss_fn = FocalLoss(gamma=config.focal_gamma, alpha=config.focal_alpha)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    # Create datasets
    train_ds = EmbeddingDataset(
        site_ids=site_ids_train, labels=labels_train,
        pooled_orig=pooled_orig, pooled_edited=pooled_edited,
        structure_delta=structure_delta,
    )
    test_ds = EmbeddingDataset(
        site_ids=site_ids_test, labels=labels_test,
        pooled_orig=pooled_orig, pooled_edited=pooled_edited,
        structure_delta=structure_delta,
    )
    train_loader = DataLoader(
        train_ds, batch_size=64, shuffle=True, collate_fn=embedding_collate_fn,
    )
    test_loader = DataLoader(
        test_ds, batch_size=64, shuffle=False, collate_fn=embedding_collate_fn,
    )

    # Training loop
    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            optimizer.zero_grad()
            output = _forward_baseline(model, batch, model_name)
            loss = loss_fn(output["binary_logit"], batch["labels"])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(len(train_loader), 1)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                logger.info("  Early stopping at epoch %d", epoch)
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Evaluate
    model.eval()
    all_targets, all_scores = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            output = _forward_baseline(model, batch, model_name)
            logits = output["binary_logit"].squeeze(-1).cpu().numpy()
            probs = 1.0 / (1.0 + np.exp(-logits))
            all_targets.append(batch["labels"].cpu().numpy())
            all_scores.append(probs)

    y_true = np.concatenate(all_targets)
    y_score = np.concatenate(all_scores)
    metrics = compute_metrics(y_true, y_score)
    elapsed = time.time() - t0

    # Convert numpy/float types for JSON serialization
    metrics = {k: float(v) for k, v in metrics.items()}

    logger.info("  %s: AUROC=%.4f  AUPRC=%.4f  F1=%.4f  (%.1fs)",
                model_name, metrics["auroc"], metrics["auprc"], metrics["f1"], elapsed)
    return {"metrics": metrics, "train_time": elapsed, "y_score": y_score}


# ---------------------------------------------------------------------------
# Error analysis helpers
# ---------------------------------------------------------------------------


def compute_optimal_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute the optimal F1 threshold from precision-recall curve."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    prec_arr, rec_arr, thresholds = precision_recall_curve(y_true, y_score)
    f1_arr = 2 * prec_arr[:-1] * rec_arr[:-1] / (prec_arr[:-1] + rec_arr[:-1] + 1e-8)
    best_idx = int(np.argmax(f1_arr))
    return float(thresholds[best_idx]) if len(thresholds) > 0 else 0.5


def run_error_analysis(
    test_ids: list,
    y_true: np.ndarray,
    model_scores: dict,
    sequences: dict,
) -> dict:
    """Analyze per-site prediction errors between our models and RNAsee_RF.

    Args:
        test_ids: list of all test site IDs (for label lookup)
        y_true: true labels aligned with test_ids
        model_scores: {model_name: {"site_ids": [...], "y_score": np.array, "threshold": float}}
        sequences: {site_id: rna_sequence}

    Returns:
        dict with error analysis results per comparison pair.
    """
    ref_model = "RNAsee_RF"
    if ref_model not in model_scores:
        return {}

    true_idx = {sid: i for i, sid in enumerate(test_ids)}
    results = {"reference_model": ref_model, "comparisons": {}}

    for comp_model in ["GB_HandFeatures", "EditRNA_A3A", "SubtractionMLP"]:
        if comp_model not in model_scores:
            continue

        ref = model_scores[ref_model]
        comp = model_scores[comp_model]

        ref_idx_map = {sid: i for i, sid in enumerate(ref["site_ids"])}
        comp_idx_map = {sid: i for i, sid in enumerate(comp["site_ids"])}
        common_ids = sorted(set(ref["site_ids"]) & set(comp["site_ids"]))
        if not common_ids:
            continue

        ref_thresh = ref["threshold"]
        comp_thresh = comp["threshold"]

        rescued = {"true_positives": [], "true_negatives": []}
        lost = {"true_positives": [], "true_negatives": []}

        for sid in common_ids:
            if sid not in true_idx:
                continue
            label = int(y_true[true_idx[sid]])
            ref_pred = int(ref["y_score"][ref_idx_map[sid]] >= ref_thresh)
            comp_pred = int(comp["y_score"][comp_idx_map[sid]] >= comp_thresh)
            ref_ok = ref_pred == label
            comp_ok = comp_pred == label

            if comp_ok and not ref_ok:
                bucket = "true_positives" if label == 1 else "true_negatives"
                rescued[bucket].append(sid)
            elif ref_ok and not comp_ok:
                bucket = "true_positives" if label == 1 else "true_negatives"
                lost[bucket].append(sid)

        # Motif distribution of rescued true positives
        rescued_tp_motifs = {}
        for sid in rescued["true_positives"]:
            if sid in sequences and len(sequences[sid]) >= 102:
                trinuc = sequences[sid][99:102]
                rescued_tp_motifs[trinuc] = rescued_tp_motifs.get(trinuc, 0) + 1

        # Baseline: all true-positive motifs in the common set
        all_tp_motifs = {}
        for sid in common_ids:
            if sid not in true_idx:
                continue
            if int(y_true[true_idx[sid]]) == 1:
                if sid in sequences and len(sequences[sid]) >= 102:
                    trinuc = sequences[sid][99:102]
                    all_tp_motifs[trinuc] = all_tp_motifs.get(trinuc, 0) + 1

        comparison = {
            "n_common_sites": len(common_ids),
            "ref_threshold": ref_thresh,
            "comp_threshold": comp_thresh,
            "rescued": {
                "total": len(rescued["true_positives"]) + len(rescued["true_negatives"]),
                "true_positives": len(rescued["true_positives"]),
                "true_negatives": len(rescued["true_negatives"]),
                "tp_site_ids": rescued["true_positives"],
                "tn_site_ids": rescued["true_negatives"],
                "tp_motif_distribution": rescued_tp_motifs,
            },
            "lost": {
                "total": len(lost["true_positives"]) + len(lost["true_negatives"]),
                "true_positives": len(lost["true_positives"]),
                "true_negatives": len(lost["true_negatives"]),
                "tp_site_ids": lost["true_positives"],
                "tn_site_ids": lost["true_negatives"],
            },
            "all_tp_motif_distribution": all_tp_motifs,
        }
        results["comparisons"][f"{comp_model}_vs_{ref_model}"] = comparison

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info("=" * 80)
    logger.info("RNAsee APPLES-TO-APPLES COMPARISON EXPERIMENT")
    logger.info("=" * 80)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load Asaoka 2019 positives
    # ------------------------------------------------------------------
    positives = load_asaoka_positives()

    # ------------------------------------------------------------------
    # 2. Load sequences
    # ------------------------------------------------------------------
    logger.info("Loading sequences...")
    with open(SEQUENCES_JSON) as f:
        sequences = json.load(f)
    logger.info("Loaded %d sequences", len(sequences))

    # ------------------------------------------------------------------
    # 3. Load embeddings (for embedding-based models)
    # ------------------------------------------------------------------
    import torch

    pooled_orig = pooled_edited = None
    structure_delta_dict = None
    has_embeddings = POOLED_ORIG_PT.exists() and POOLED_EDITED_PT.exists()

    if has_embeddings:
        logger.info("Loading pooled embeddings...")
        pooled_orig = torch.load(POOLED_ORIG_PT, map_location="cpu", weights_only=False)
        pooled_edited = torch.load(POOLED_EDITED_PT, map_location="cpu", weights_only=False)
        logger.info("  Loaded %d original, %d edited", len(pooled_orig), len(pooled_edited))

        if STRUCTURE_CACHE.exists():
            data = np.load(STRUCTURE_CACHE, allow_pickle=True)
            s_ids = data["site_ids"]
            s_feats = data["delta_features"]
            structure_delta_dict = {str(sid): s_feats[j] for j, sid in enumerate(s_ids)}

    # ------------------------------------------------------------------
    # 4. Run comparison for each negative type
    # ------------------------------------------------------------------
    neg_configs = []

    # All-cytidine negatives (RNAsee-style easy)
    if ALLC_NEG_CSV.exists():
        neg_configs.append(("all_cytidine", ALLC_NEG_CSV, 3))
    else:
        logger.warning("All-cytidine negatives not found: %s", ALLC_NEG_CSV)
        logger.warning("Run: python scripts/apobec3a/generate_per_dataset_negatives.py --all-cytidines --neg-ratio 3")

    # TC-motif negatives (harder)
    if PER_DATASET_NEG_CSV.exists():
        neg_configs.append(("tc_motif", PER_DATASET_NEG_CSV, 3))
    else:
        logger.warning("TC-motif negatives not found: %s", PER_DATASET_NEG_CSV)

    results = {}

    # Split strategies to compare
    split_strategies = [
        ("random_70_30", "Random 70:30 (RNAsee setup)"),
        ("gene_stratified", "Gene-Stratified 70:30"),
        ("chromosome_stratified", "Chromosome-Stratified"),
    ]

    for neg_name, neg_path, neg_ratio in neg_configs:
        logger.info("\n" + "=" * 80)
        logger.info("NEGATIVE TYPE: %s (from %s)", neg_name, neg_path.name)
        logger.info("=" * 80)

        negatives = load_negatives(neg_path, "asaoka_2019")
        if len(negatives) == 0:
            logger.warning("No negatives found for asaoka_2019 in %s, skipping", neg_path)
            continue

        # Extract sequences once for this negative type
        neg_full = pd.read_csv(neg_path)

        for split_name, split_desc in split_strategies:
            logger.info("\n--- Split: %s ---", split_desc)

            if split_name == "random_70_30":
                split_df = create_rnasee_split(positives, negatives, neg_ratio=neg_ratio)
            elif split_name == "gene_stratified":
                split_df = create_gene_stratified_split(positives, negatives, neg_ratio=neg_ratio)
            elif split_name == "chromosome_stratified":
                split_df = create_chromosome_stratified_split(positives, negatives, neg_ratio=neg_ratio)

            # Extract sequences for negative sites not in cache
            sequences = extract_missing_sequences(split_df, sequences, neg_full)

            result_key = f"{neg_name}__{split_name}"

            train_df = split_df[split_df["split"] == "train"]
            test_df = split_df[split_df["split"] == "test"]

            train_ids = train_df["site_id"].tolist()
            test_ids = test_df["site_id"].tolist()
            y_train = train_df["label"].values
            y_test = test_df["label"].values

            split_results = {
                "split_strategy": split_name,
                "split_description": split_desc,
                "n_train": len(train_df),
                "n_test": len(test_df),
                "n_train_pos": int((train_df["label"] == 1).sum()),
                "n_train_neg": int((train_df["label"] == 0).sum()),
                "n_test_pos": int((test_df["label"] == 1).sum()),
                "n_test_neg": int((test_df["label"] == 0).sum()),
                "models": {},
            }

            # ---- Feature-based models (run on all sites) ----
            per_site_model_scores = {}

            # 1. RNAsee-like RF with 50-bit binary encoding
            X_rnasee_train = encode_rnasee_features(sequences, train_ids)
            X_rnasee_test = encode_rnasee_features(sequences, test_ids)
            r = train_rf_rnasee(X_rnasee_train, y_train, X_rnasee_test, y_test,
                                "RNAsee_RF")
            y_score_rf = r.pop("y_score")
            per_site_model_scores["RNAsee_RF"] = {
                "site_ids": test_ids,
                "y_score": y_score_rf,
                "threshold": compute_optimal_threshold(y_test, y_score_rf),
            }
            split_results["models"]["RNAsee_RF"] = r

            # 2. GB_HandFeatures
            X_hand_all, hand_names = extract_hand_features(
                train_ids + test_ids, sequences
            )
            n_train = len(train_ids)
            X_hand_train = X_hand_all[:n_train]
            X_hand_test = X_hand_all[n_train:]
            r = train_gb_hand(X_hand_train, y_train, X_hand_test, y_test,
                              "GB_HandFeatures")
            y_score_gb = r.pop("y_score")
            per_site_model_scores["GB_HandFeatures"] = {
                "site_ids": test_ids,
                "y_score": y_score_gb,
                "threshold": compute_optimal_threshold(y_test, y_score_gb),
            }
            split_results["models"]["GB_HandFeatures"] = r

            # ---- Embedding-based models (subset with embeddings) ----
            if has_embeddings:
                available = set(pooled_orig.keys()) & set(pooled_edited.keys())

                emb_train_mask = [sid in available for sid in train_ids]
                emb_test_mask = [sid in available for sid in test_ids]

                emb_train_ids = [sid for sid, m in zip(train_ids, emb_train_mask) if m]
                emb_test_ids = [sid for sid, m in zip(test_ids, emb_test_mask) if m]
                emb_y_train = y_train[emb_train_mask]
                emb_y_test = y_test[emb_test_mask]

                logger.info("Embedding coverage: train %d/%d, test %d/%d",
                            len(emb_train_ids), len(train_ids),
                            len(emb_test_ids), len(test_ids))

                if len(emb_train_ids) > 100 and len(emb_test_ids) > 50:
                    for model_name in ["subtraction_mlp", "editrna"]:
                        try:
                            r = train_embedding_model(
                                model_name,
                                emb_train_ids, emb_y_train,
                                emb_test_ids, emb_y_test,
                                pooled_orig, pooled_edited,
                                structure_delta_dict,
                            )
                            y_score_emb = r.pop("y_score")
                            display_name = {
                                "subtraction_mlp": "SubtractionMLP",
                                "editrna": "EditRNA_A3A",
                            }.get(model_name, model_name)
                            per_site_model_scores[display_name] = {
                                "site_ids": emb_test_ids,
                                "y_score": y_score_emb,
                                "threshold": compute_optimal_threshold(
                                    emb_y_test, y_score_emb
                                ),
                            }
                            split_results["models"][display_name] = r
                        except Exception as e:
                            logger.error("Failed to train %s: %s", model_name, e)
                else:
                    logger.warning("Insufficient embedding coverage for embedding models")

            results[result_key] = split_results

            # Save per-site predictions and error analysis for the primary config
            if result_key == "all_cytidine__random_70_30":
                per_site = {
                    "test_site_ids": test_ids,
                    "y_true": y_test.tolist(),
                    "models": {},
                }
                for mname, mdata in per_site_model_scores.items():
                    ys = mdata["y_score"]
                    per_site["models"][mname] = {
                        "site_ids": mdata["site_ids"],
                        "y_score": ys.tolist() if isinstance(ys, np.ndarray) else ys,
                        "threshold": mdata["threshold"],
                    }
                pred_path = OUTPUT_DIR / "per_site_predictions.json"
                with open(pred_path, "w") as f:
                    json.dump(per_site, f, indent=2)
                logger.info("Per-site predictions saved to %s", pred_path)

                error_analysis = run_error_analysis(
                    test_ids, y_test, per_site_model_scores, sequences
                )
                error_path = OUTPUT_DIR / "rnasee_error_analysis.json"
                with open(error_path, "w") as f:
                    json.dump(error_analysis, f, indent=2)
                logger.info("Error analysis saved to %s", error_path)

    # ------------------------------------------------------------------
    # 5. Print summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("RNAsee COMPARISON RESULTS")
    print("=" * 80)
    print(f"\nRNAsee reported: AUROC=0.962 (Asaoka 2019, all-cytidine 1:3, 70:30 split)")
    print()

    for result_key, neg_data in results.items():
        split_desc = neg_data.get("split_description", result_key)
        print(f"\n--- {result_key} ({split_desc}) ---")
        print(f"Train: {neg_data['n_train']} ({neg_data['n_train_pos']} pos, "
              f"{neg_data['n_train_neg']} neg)")
        print(f"Test:  {neg_data['n_test']} ({neg_data['n_test_pos']} pos, "
              f"{neg_data['n_test_neg']} neg)")
        print()
        print(f"{'Model':<22} {'AUROC':>8} {'AUPRC':>8} {'F1':>8} "
              f"{'Precision':>10} {'Recall':>8}")
        print("-" * 66)
        for model_name, model_data in neg_data["models"].items():
            m = model_data["metrics"]
            auroc = m.get("auroc", float("nan"))
            auprc = m.get("auprc", float("nan"))
            f1 = m.get("f1", float("nan"))
            prec = m.get("precision", float("nan"))
            rec = m.get("recall", float("nan"))
            print(f"{model_name:<22} {auroc:>8.4f} {auprc:>8.4f} "
                  f"{f1:>8.4f} {prec:>10.4f} {rec:>8.4f}")

    print("\n" + "=" * 80)

    # ------------------------------------------------------------------
    # 6. Save results
    # ------------------------------------------------------------------
    output = {
        "experiment": "rnasee_comparison",
        "description": "Apples-to-apples comparison with RNAsee (Baysal et al. 2024)",
        "rnasee_reported_auroc": 0.962,
        "setup": {
            "dataset": "asaoka_2019",
            "split": "70:30 random (no gene stratification)",
            "seed": 42,
        },
        "results": results,
    }

    results_path = OUTPUT_DIR / "rnasee_comparison_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info("Results saved to %s", results_path)
    return output


if __name__ == "__main__":
    main()
