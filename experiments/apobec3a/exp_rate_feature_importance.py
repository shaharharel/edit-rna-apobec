#!/usr/bin/env python
"""Feature importance analysis for editing rate regression.

Part 1: GB_HandFeatures with XGBRegressor feature importance extraction
Part 2: DL model performance comparison (EditRNA, DiffAttention, 3Way, GNN_Only, 4Way)
Part 3: Gate weight analysis for gated fusion models (3Way, 4Way)

Output directory: outputs/rate_feature_importance/
Output files:
  - feature_importance_rate_hand.csv  (feature, importance, rank)
  - gate_weights_3way.json            (modality-level gate weights)
  - gate_weights_4way.json            (modality-level gate weights)
  - results.json                      (summary: GB importance + DL metrics + gate weights)

Usage:
    python experiments/apobec3a/exp_rate_feature_importance.py
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error, r2_score

# Avoid OpenMP conflicts between xgboost and torch_geometric
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "apobec"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SPLITS_CSV = PROJECT_ROOT / "data" / "processed" / "splits_expanded.csv"
SEQUENCES_JSON = PROJECT_ROOT / "data" / "processed" / "site_sequences.json"
EMB_DIR = PROJECT_ROOT / "data" / "processed" / "embeddings"
STRUCTURE_CACHE = EMB_DIR / "vienna_structure_cache.npz"
LOOP_POS_CSV = (
    PROJECT_ROOT / "experiments" / "apobec" / "outputs"
    / "loop_position" / "loop_position_per_site.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "apobec" / "outputs" / "rate_feature_importance"

SEED = 42
DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_rate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute regression metrics for rate prediction."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

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
# Data loading (rate-annotated sites only)
# ---------------------------------------------------------------------------

def load_rate_data() -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Load splits_expanded.csv, filter to rate-annotated sites, apply log2 transform."""
    df = pd.read_csv(SPLITS_CSV)
    df = df[df["editing_rate_normalized"].notna()].copy()
    df["target"] = np.log2(df["editing_rate_normalized"].values + 0.01)

    split_dfs = {}
    for split in ["train", "val", "test"]:
        split_dfs[split] = df[df["split"] == split].copy()

    logger.info(
        "Rate data: train=%d, val=%d, test=%d",
        len(split_dfs["train"]), len(split_dfs["val"]), len(split_dfs["test"]),
    )
    return df, split_dfs


# ---------------------------------------------------------------------------
# Feature extraction for GB
# ---------------------------------------------------------------------------

def extract_motif_features(sequences: Dict[str, str], site_ids: List[str]) -> Tuple[np.ndarray, List[str]]:
    """Extract motif features from 201nt sequences with edit at position 100."""
    feature_names = [
        "motif_TC", "motif_CC", "motif_AC", "motif_GC",
        "motif_CA", "motif_CG", "motif_CU", "motif_CC_3p",
        "frac_A", "frac_C", "frac_G", "frac_U",
        "gc_content",
    ]

    features = []
    for sid in site_ids:
        seq = sequences.get(sid, "N" * 201)
        seq = seq.upper().replace("T", "U")
        edit_pos = 100

        upstream = seq[edit_pos - 1] if edit_pos > 0 else "N"
        motif_5p = upstream + "C"
        tc = 1 if motif_5p == "UC" else 0
        cc = 1 if motif_5p == "CC" else 0
        ac = 1 if motif_5p == "AC" else 0
        gc = 1 if motif_5p == "GC" else 0

        downstream = seq[edit_pos + 1] if edit_pos < len(seq) - 1 else "N"
        motif_3p = "C" + downstream
        ca = 1 if motif_3p == "CA" else 0
        cg = 1 if motif_3p == "CG" else 0
        cu = 1 if motif_3p == "CU" else 0
        cc_3p = 1 if motif_3p == "CC" else 0

        w_start = max(0, edit_pos - 10)
        w_end = min(len(seq), edit_pos + 11)
        window = seq[w_start:w_end]
        w_len = max(len(window), 1)
        frac_a = window.count("A") / w_len
        frac_c = window.count("C") / w_len
        frac_g = window.count("G") / w_len
        frac_u = window.count("U") / w_len
        gc_content = (window.count("G") + window.count("C")) / w_len

        features.append([
            tc, cc, ac, gc,
            ca, cg, cu, cc_3p,
            frac_a, frac_c, frac_g, frac_u,
            gc_content,
        ])

    return np.array(features, dtype=np.float32), feature_names


def extract_structure_delta_features(site_ids: List[str]) -> Tuple[np.ndarray, List[str]]:
    """Extract 7-dim structure delta features from vienna_structure_cache.npz."""
    logger.info("Loading structure deltas from %s ...", STRUCTURE_CACHE)
    data = np.load(str(STRUCTURE_CACHE), allow_pickle=True)
    sids = data["site_ids"]
    delta_features = data["delta_features"]
    struct_dict = {str(sid): delta_features[i] for i, sid in enumerate(sids)}

    feature_names = [
        "delta_pairing_at_pos", "delta_accessibility_at_pos", "delta_entropy_at_pos",
        "delta_mfe", "delta_local_pairing", "delta_local_accessibility", "local_pairing_std",
    ]

    feats = np.array([
        struct_dict.get(sid, np.zeros(7)) for sid in site_ids
    ], dtype=np.float32)

    logger.info("  %d structure deltas loaded, %d matched",
                len(struct_dict), sum(1 for sid in site_ids if sid in struct_dict))
    return feats, feature_names


def extract_loop_position_features(site_ids: List[str]) -> Tuple[np.ndarray, List[str]]:
    """Extract loop position features from loop_position_per_site.csv."""
    logger.info("Loading loop position features from %s ...", LOOP_POS_CSV)

    loop_feature_names = [
        "is_unpaired", "loop_size", "dist_to_left_boundary",
        "dist_to_right_boundary", "dist_to_nearest_stem",
        "relative_loop_position", "dist_to_apex",
        "left_stem_length", "right_stem_length",
        "max_adjacent_stem_length", "dist_to_junction",
        "local_unpaired_fraction", "mfe",
    ]

    if not LOOP_POS_CSV.exists():
        logger.warning("  Loop position file not found, using zeros")
        return np.zeros((len(site_ids), len(loop_feature_names)), dtype=np.float32), loop_feature_names

    loop_df = pd.read_csv(LOOP_POS_CSV)
    loop_df = loop_df.set_index("site_id")
    available_cols = [c for c in loop_feature_names if c in loop_df.columns]

    feats = []
    for sid in site_ids:
        if sid in loop_df.index:
            row = loop_df.loc[sid, available_cols] if available_cols else pd.Series(dtype=float)
            vals = row.values.astype(np.float32)
            if len(vals) < len(loop_feature_names):
                vals = np.concatenate([vals, np.zeros(len(loop_feature_names) - len(vals), dtype=np.float32)])
            feats.append(vals)
        else:
            feats.append(np.full(len(loop_feature_names), np.nan, dtype=np.float32))

    logger.info("  Loop position features: %d columns, %d matched",
                len(loop_feature_names), sum(1 for sid in site_ids if sid in loop_df.index))
    return np.array(feats, dtype=np.float32), loop_feature_names


# ---------------------------------------------------------------------------
# Build hand feature matrices
# ---------------------------------------------------------------------------

def build_hand_feature_matrices(
    split_dfs: Dict[str, pd.DataFrame],
    sequences: Dict[str, str],
) -> Dict:
    """Build hand feature matrices for each split."""
    all_site_ids = []
    for split in ["train", "val", "test"]:
        all_site_ids.extend(split_dfs[split]["site_id"].tolist())

    logger.info("\n--- Extracting hand features for %d rate-annotated sites ---", len(all_site_ids))

    motif_feats, motif_names = extract_motif_features(sequences, all_site_ids)
    struct_feats, struct_names = extract_structure_delta_features(all_site_ids)
    loop_feats, loop_names = extract_loop_position_features(all_site_ids)

    hand_feature_names = motif_names + struct_names + loop_names
    X_hand = np.concatenate([motif_feats, struct_feats, loop_feats], axis=1)

    all_targets = []
    for split in ["train", "val", "test"]:
        all_targets.extend(split_dfs[split]["target"].values.tolist())
    all_targets = np.array(all_targets, dtype=np.float32)

    logger.info("  Hand features: %d dims", len(hand_feature_names))

    result = {}
    offset = 0
    for split in ["train", "val", "test"]:
        n = len(split_dfs[split])
        result[split] = {
            "hand_X": np.nan_to_num(X_hand[offset:offset + n], nan=0.0),
            "y": all_targets[offset:offset + n],
        }
        offset += n

    result["hand_feature_names"] = hand_feature_names
    return result


# ---------------------------------------------------------------------------
# Train GB and extract feature importances
# ---------------------------------------------------------------------------

def train_and_extract_importance(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    feature_names: List[str],
    variant_name: str,
) -> Dict:
    """Train XGBRegressor for rate regression, extract feature importances."""
    from xgboost import XGBRegressor

    logger.info("\n--- Training %s (%d features) ---", variant_name, len(feature_names))
    logger.info("  Train: %d, Val: %d, Test: %d", len(y_train), len(y_val), len(y_test))

    model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=SEED,
        n_jobs=1,
        early_stopping_rounds=30,
        verbosity=0,
    )

    t_start = time.time()
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    elapsed = time.time() - t_start
    logger.info("  Training took %.1f seconds", elapsed)

    metrics = {}
    for split_name, X, y in [("train", X_train, y_train),
                              ("val", X_val, y_val),
                              ("test", X_test, y_test)]:
        y_pred = model.predict(X)
        metrics[split_name] = compute_rate_metrics(y, y_pred)

    logger.info("  Val:  spearman=%.4f, pearson=%.4f, mse=%.4f",
                metrics["val"]["spearman"], metrics["val"]["pearson"], metrics["val"]["mse"])
    logger.info("  Test: spearman=%.4f, pearson=%.4f, mse=%.4f",
                metrics["test"]["spearman"], metrics["test"]["pearson"], metrics["test"]["mse"])

    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    importance_df["rank"] = range(1, len(importance_df) + 1)

    best_iteration = int(model.best_iteration) if hasattr(model, "best_iteration") else -1

    return {
        "variant": variant_name,
        "n_features": len(feature_names),
        "best_iteration": best_iteration,
        "train_time_seconds": round(elapsed, 1),
        "metrics": metrics,
        "importance_df": importance_df,
    }


# ---------------------------------------------------------------------------
# DL Model training and evaluation (imports from existing experiments)
# ---------------------------------------------------------------------------

def run_dl_models(split_dfs: Dict[str, pd.DataFrame]) -> Dict:
    """Train all 5 DL models and return their metrics."""
    from exp_rate_unified import (
        RateDataset, rate_collate_fn, create_rate_dataloaders,
        DiffAttentionRegressor, EditRNARateWrapper, build_editrna_rate_model,
        train_nn_model, train_editrna_rate, evaluate_nn_model,
        load_embeddings, load_structure_deltas,
    )
    from exp_rate_gnn_fusion import (
        ThreeWayGatedFusion, MultiModalRateDataset, mm_collate,
        load_all_data,
        extract_motif_features as extract_motif_gf,
        extract_loop_features as extract_loop_gf,
        extract_structure_delta as extract_struct_gf,
        extract_pairing_profile_features,
        train_nn as train_nn_fusion,
        eval_nn as eval_nn_fusion,
    )
    from exp_rate_gnn_endtoend import (
        GNNOnlyRate, FourWayE2E, SmallStructureGNN,
        GraphRateDataset, graph_collate_fn, prebuild_all_graphs,
        train_e2e, eval_e2e,
    )
    from torch.utils.data import DataLoader

    dl_results = {}

    # ---------------------------------------------------------------
    # Load shared data
    # ---------------------------------------------------------------
    logger.info("\n--- Loading data for DL models ---")
    data = load_all_data()
    gf_split_dfs = data["split_dfs"]

    # Filter to sites with embeddings
    available = set(data["pooled_orig"].keys()) & set(data["pooled_edited"].keys())
    for s in gf_split_dfs:
        gf_split_dfs[s] = gf_split_dfs[s][gf_split_dfs[s]["site_id"].isin(available)].copy()
    logger.info("After embedding filter: train=%d, val=%d, test=%d",
                len(gf_split_dfs["train"]), len(gf_split_dfs["val"]), len(gf_split_dfs["test"]))

    structure_delta = data["structure_delta"]

    # Load token-level embeddings for EditRNA and DiffAttention
    logger.info("Loading token-level embeddings...")
    pooled_orig, pooled_edited, tokens_orig, tokens_edited = load_embeddings(load_tokens=True)

    # Create dataloaders for token-level models (EditRNA, DiffAttention)
    loaders_tok = create_rate_dataloaders(
        gf_split_dfs, pooled_orig, pooled_edited, structure_delta,
        tokens_orig=tokens_orig, tokens_edited=tokens_edited,
        batch_size=32,
    )

    # Build hand + pairing features for fusion models
    hand_features = {}
    pairing_features = {}
    for sn, sdf in gf_split_dfs.items():
        sids = sdf["site_id"].tolist()
        motif = extract_motif_gf(data["sequences"], sids)
        struct = extract_struct_gf(data["structure_delta"], sids)
        loop = extract_loop_gf(data["loop_df"], sids)
        hand_features[sn] = np.nan_to_num(np.concatenate([motif, struct, loop], axis=1))

        pair = extract_pairing_profile_features(
            data["pairing_probs"], data["pairing_probs_edited"],
            data["accessibilities"], data["accessibilities_edited"],
            data["mfes"], data["mfes_edited"], sids)
        pairing_features[sn] = np.nan_to_num(pair)

    d_hand = hand_features["train"].shape[1]
    d_pairing = pairing_features["train"].shape[1]

    # ---------------------------------------------------------------
    # Model 1: EditRNA_rate
    # ---------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("DL MODEL 1: EditRNA_rate")
    logger.info("=" * 70)

    try:
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        editrna_model = build_editrna_rate_model(
            pooled_orig, pooled_edited, tokens_orig, tokens_edited,
        )
        n_params = sum(p.numel() for p in editrna_model.parameters() if p.requires_grad)
        logger.info("  Trainable parameters: %s", f"{n_params:,}")

        editrna_results = train_editrna_rate(
            editrna_model, loaders_tok,
            epochs=100, lr=1e-3, weight_decay=1e-2, patience=15,
            rate_weight=10.0,
        )
        dl_results["EditRNA_rate"] = {
            "metrics": {k: v for k, v in editrna_results.items()
                        if k in ("train", "val", "test")},
            "n_params": n_params,
        }
        logger.info("  Test spearman: %.4f",
                     editrna_results.get("test", {}).get("spearman", float("nan")))
    except Exception as e:
        logger.error("EditRNA_rate FAILED: %s", e, exc_info=True)
        dl_results["EditRNA_rate"] = {"error": str(e)}

    # ---------------------------------------------------------------
    # Model 2: DiffAttention_reg
    # ---------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("DL MODEL 2: DiffAttention_reg")
    logger.info("=" * 70)

    try:
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        diff_model = DiffAttentionRegressor(
            d_model=640, n_heads=4, d_hidden=128, dropout=0.5,
        )
        n_params = sum(p.numel() for p in diff_model.parameters())
        logger.info("  Parameters: %s", f"{n_params:,}")

        diff_results = train_nn_model(
            diff_model, loaders_tok, "DiffAttention_reg",
            epochs=100, lr=1e-3, weight_decay=1e-2, patience=15,
        )
        dl_results["DiffAttention_reg"] = {
            "metrics": {k: v for k, v in diff_results.items()
                        if k in ("train", "val", "test")},
            "n_params": n_params,
        }
        logger.info("  Test spearman: %.4f",
                     diff_results.get("test", {}).get("spearman", float("nan")))
    except Exception as e:
        logger.error("DiffAttention_reg FAILED: %s", e, exc_info=True)
        dl_results["DiffAttention_reg"] = {"error": str(e)}

    # ---------------------------------------------------------------
    # Model 3: 3Way_heavyreg (with gate weight extraction)
    # ---------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("DL MODEL 3: 3Way_heavyreg")
    logger.info("=" * 70)

    try:
        # Create 3-way loaders
        loaders_3way = {}
        for sn, sdf in gf_split_dfs.items():
            sids = sdf["site_id"].tolist()
            targets = sdf["target"].values.astype(np.float32)
            ds = MultiModalRateDataset(
                site_ids=sids, targets=targets,
                pooled_orig=data["pooled_orig"], pooled_edited=data["pooled_edited"],
                hand_features=hand_features[sn],
                pairing_features=pairing_features[sn],
            )
            loaders_3way[sn] = DataLoader(ds, batch_size=32,
                                           shuffle=(sn == "train"), num_workers=0,
                                           collate_fn=mm_collate)

        np.random.seed(SEED)
        torch.manual_seed(SEED)
        model_3way = ThreeWayGatedFusion(
            d_hand=d_hand, d_pairing=d_pairing,
            d_proj=128, dropout=0.5,
        )
        n_params = sum(p.numel() for p in model_3way.parameters())
        logger.info("  Parameters: %s", f"{n_params:,}")

        result_3way = train_nn_fusion(
            model_3way, loaders_3way, "3Way_heavyreg",
            epochs=200, lr=5e-4, weight_decay=1e-2, patience=30,
        )

        # Extract gate weights
        gate_weights_3way = extract_3way_gate_weights(model_3way, loaders_3way["test"])
        logger.info("  Gate weights: %s", gate_weights_3way)

        dl_results["3Way_heavyreg"] = {
            "metrics": {k: v for k, v in result_3way.items()
                        if k in ("train", "val", "test")},
            "n_params": n_params,
            "gate_weights": gate_weights_3way,
        }
        logger.info("  Test spearman: %.4f",
                     result_3way.get("test", {}).get("spearman", float("nan")))
    except Exception as e:
        logger.error("3Way_heavyreg FAILED: %s", e, exc_info=True)
        dl_results["3Way_heavyreg"] = {"error": str(e)}

    # ---------------------------------------------------------------
    # Model 4: GNN_Only
    # ---------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("DL MODEL 4: GNN_Only")
    logger.info("=" * 70)

    try:
        # Pre-build PyG graphs
        split_graphs = {}
        for sn, sdf in gf_split_dfs.items():
            sids = sdf["site_id"].tolist()
            split_graphs[sn] = prebuild_all_graphs(
                sids, data["sequences"], data["pairing_probs"]
            )

        # Create graph loaders
        loaders_graph = {}
        for sn, sdf in gf_split_dfs.items():
            sids = sdf["site_id"].tolist()
            targets = sdf["target"].values.astype(np.float32)
            ds = GraphRateDataset(
                site_ids=sids, targets=targets,
                graphs=split_graphs[sn],
                pooled_orig=data["pooled_orig"],
                pooled_edited=data["pooled_edited"],
                hand_features=hand_features[sn],
            )
            loaders_graph[sn] = DataLoader(
                ds, batch_size=64,
                shuffle=(sn == "train"), num_workers=0,
                collate_fn=graph_collate_fn,
            )

        np.random.seed(SEED)
        torch.manual_seed(SEED)
        gnn_model = GNNOnlyRate(
            SmallStructureGNN(12, 64, 64, 3, 4, 0.3),
        )
        n_params = sum(p.numel() for p in gnn_model.parameters())
        logger.info("  Parameters: %s", f"{n_params:,}")

        result_gnn, _ = train_e2e(
            gnn_model, loaders_graph, "GNN_Only",
            epochs=150, lr=1e-3, weight_decay=5e-3, patience=20,
        )
        dl_results["GNN_Only"] = {
            "metrics": {k: v for k, v in result_gnn.items()
                        if k in ("train", "val", "test")},
            "n_params": n_params,
        }
        logger.info("  Test spearman: %.4f",
                     result_gnn.get("test", {}).get("spearman", float("nan")))
    except Exception as e:
        logger.error("GNN_Only FAILED: %s", e, exc_info=True)
        dl_results["GNN_Only"] = {"error": str(e)}

    # ---------------------------------------------------------------
    # Model 5: 4Way_e2e_heavy (with gate weight extraction)
    # ---------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("DL MODEL 5: 4Way_e2e_heavy")
    logger.info("=" * 70)

    try:
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        model_4way = FourWayE2E(
            SmallStructureGNN(12, 64, 64, 3, 4, 0.4),
            d_hand=d_hand, d_proj=128, dropout=0.5,
        )
        n_params = sum(p.numel() for p in model_4way.parameters())
        logger.info("  Parameters: %s", f"{n_params:,}")

        result_4way, trained_4way = train_e2e(
            model_4way, loaders_graph, "4Way_e2e_heavy",
            epochs=150, lr=5e-4, weight_decay=2e-2, patience=25,
            use_huber=True,
        )

        # Extract gate weights
        gate_weights_4way = extract_4way_gate_weights(trained_4way, loaders_graph["test"])
        logger.info("  Gate weights: %s", gate_weights_4way)

        dl_results["4Way_e2e_heavy"] = {
            "metrics": {k: v for k, v in result_4way.items()
                        if k in ("train", "val", "test")},
            "n_params": n_params,
            "gate_weights": gate_weights_4way,
        }
        logger.info("  Test spearman: %.4f",
                     result_4way.get("test", {}).get("spearman", float("nan")))
    except Exception as e:
        logger.error("4Way_e2e_heavy FAILED: %s", e, exc_info=True)
        dl_results["4Way_e2e_heavy"] = {"error": str(e)}

    return dl_results


# ---------------------------------------------------------------------------
# Gate weight extraction
# ---------------------------------------------------------------------------

def _move_batch(batch):
    """Move batch to DEVICE, handling PyG Batch separately."""
    out = {}
    for k, v in batch.items():
        if k == "graph_batch":
            out[k] = v.to(DEVICE)
        elif isinstance(v, torch.Tensor):
            out[k] = v.to(DEVICE)
        else:
            out[k] = v
    return out


@torch.no_grad()
def extract_3way_gate_weights(model, loader):
    """Extract average gate weights from ThreeWayGatedFusion model."""
    model.eval()
    all_gates = []
    for batch in loader:
        batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        e = model.edit_proj(batch["edit_delta"])
        h = model.hand_proj(batch["hand_features"])
        p = model.pairing_proj(batch["pairing_features"])
        concat = torch.cat([e, h, p], dim=-1)
        gates = torch.softmax(model.gate(concat), dim=-1)
        all_gates.append(gates.cpu().numpy())
    avg_gates = np.concatenate(all_gates).mean(axis=0)
    return {
        "edit_delta": float(avg_gates[0]),
        "hand": float(avg_gates[1]),
        "pairing": float(avg_gates[2]),
    }


@torch.no_grad()
def extract_4way_gate_weights(model, loader):
    """Extract average gate weights from FourWayE2E model."""
    model.eval()
    all_gates = []
    for batch in loader:
        batch = _move_batch(batch)
        gnn_emb = model.gnn(batch["graph_batch"])
        p = model.primary_proj(batch["pooled_orig"])
        e = model.edit_proj(batch["edit_delta"])
        h = model.hand_proj(batch["hand_features"])
        g = model.gnn_proj(gnn_emb)
        concat = torch.cat([p, e, h, g], dim=-1)
        gates = torch.softmax(model.gate(concat), dim=-1)
        all_gates.append(gates.cpu().numpy())
    avg_gates = np.concatenate(all_gates).mean(axis=0)
    return {
        "primary": float(avg_gates[0]),
        "edit_delta": float(avg_gates[1]),
        "hand": float(avg_gates[2]),
        "gnn": float(avg_gates[3]),
    }


# ---------------------------------------------------------------------------
# JSON serializer
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logger.info("=" * 80)
    logger.info("RATE FEATURE IMPORTANCE ANALYSIS")
    logger.info("Target: log2(editing_rate_normalized + 0.01)")
    logger.info("=" * 80)

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load rate-annotated data
    full_df, split_dfs = load_rate_data()

    # Load sequences
    with open(SEQUENCES_JSON) as f:
        sequences = json.load(f)
    logger.info("  %d sequences loaded", len(sequences))

    # ==================================================================
    # Part 1: GB Feature Importance
    # ==================================================================
    logger.info("\n" + "=" * 80)
    logger.info("PART 1: GB FEATURE IMPORTANCE")
    logger.info("=" * 80)

    data = build_hand_feature_matrices(split_dfs, sequences)
    hand_names = data["hand_feature_names"]

    r_hand = train_and_extract_importance(
        data["train"]["hand_X"], data["train"]["y"],
        data["val"]["hand_X"], data["val"]["y"],
        data["test"]["hand_X"], data["test"]["y"],
        hand_names,
        "GB_HandFeatures",
    )

    # Save importance CSV
    hand_csv_path = OUTPUT_DIR / "feature_importance_rate_hand.csv"
    r_hand["importance_df"].to_csv(hand_csv_path, index=False)
    logger.info("Saved %s", hand_csv_path)

    # ==================================================================
    # Part 2 + 3: DL Models + Gate Weights
    # ==================================================================
    logger.info("\n" + "=" * 80)
    logger.info("PART 2: DL MODEL PERFORMANCE COMPARISON")
    logger.info("PART 3: GATE WEIGHT ANALYSIS")
    logger.info("=" * 80)

    dl_results = run_dl_models(split_dfs)

    # Save gate weight files
    if "3Way_heavyreg" in dl_results and "gate_weights" in dl_results["3Way_heavyreg"]:
        gw3_path = OUTPUT_DIR / "gate_weights_3way.json"
        with open(gw3_path, "w") as f:
            json.dump(dl_results["3Way_heavyreg"]["gate_weights"], f, indent=2)
        logger.info("Saved %s", gw3_path)

    if "4Way_e2e_heavy" in dl_results and "gate_weights" in dl_results["4Way_e2e_heavy"]:
        gw4_path = OUTPUT_DIR / "gate_weights_4way.json"
        with open(gw4_path, "w") as f:
            json.dump(dl_results["4Way_e2e_heavy"]["gate_weights"], f, indent=2)
        logger.info("Saved %s", gw4_path)

    # ==================================================================
    # Build results.json
    # ==================================================================
    top_n = 20
    imp_df = r_hand["importance_df"]
    top_features = imp_df.head(top_n)[["feature", "importance"]].to_dict(orient="records")

    results_json = {
        "description": "Feature importance analysis for editing rate regression",
        "target_transform": "log2(editing_rate_normalized + 0.01)",
        "seed": SEED,
        "n_train": len(split_dfs["train"]),
        "n_val": len(split_dfs["val"]),
        "n_test": len(split_dfs["test"]),
        "variants": {
            "GB_HandFeatures": {
                "n_features": r_hand["n_features"],
                "best_iteration": r_hand["best_iteration"],
                "train_time_seconds": r_hand["train_time_seconds"],
                "metrics": r_hand["metrics"],
                "top_features": top_features,
            },
        },
        "dl_models": dl_results,
    }

    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2, default=_serialize)
    logger.info("Saved %s", results_path)

    # ==================================================================
    # Print summary
    # ==================================================================
    print("\n" + "=" * 80)
    print("RATE FEATURE IMPORTANCE - SUMMARY")
    print("=" * 80)

    # GB summary
    tm = r_hand["metrics"]["test"]
    print(f"\n--- GB_HandFeatures ({r_hand['n_features']} features) ---")
    print(f"  Test Spearman: {tm['spearman']:.4f}")
    print(f"  Test Pearson:  {tm['pearson']:.4f}")
    print(f"  Test MSE:      {tm['mse']:.4f}")
    print(f"  Test R2:       {tm['r2']:.4f}")
    print(f"\n  Top {top_n} features by importance:")
    print(f"  {'Rank':<6} {'Feature':<40} {'Importance':>12}")
    print(f"  {'-' * 58}")
    for _, row in r_hand["importance_df"].head(top_n).iterrows():
        print(f"  {int(row['rank']):<6} {row['feature']:<40} {row['importance']:>12.6f}")

    # DL model summary
    print(f"\n{'=' * 80}")
    print("DL MODEL COMPARISON")
    print(f"{'=' * 80}")
    print(f"  {'Model':<25} {'Test Sp':>10} {'Test Pe':>10} {'Test MSE':>10} {'Test R2':>10}")
    print(f"  {'-' * 70}")
    for model_name, model_data in dl_results.items():
        if "error" in model_data:
            print(f"  {model_name:<25} FAILED: {model_data['error'][:40]}")
        else:
            te = model_data.get("metrics", {}).get("test", {})
            def fmt(v):
                return f"{v:>10.4f}" if v is not None and not np.isnan(v) else f"{'N/A':>10}"
            print(f"  {model_name:<25} {fmt(te.get('spearman'))} {fmt(te.get('pearson'))} "
                  f"{fmt(te.get('mse'))} {fmt(te.get('r2'))}")

    # Gate weight summary
    print(f"\n{'=' * 80}")
    print("GATE WEIGHT ANALYSIS")
    print(f"{'=' * 80}")
    if "3Way_heavyreg" in dl_results and "gate_weights" in dl_results["3Way_heavyreg"]:
        gw = dl_results["3Way_heavyreg"]["gate_weights"]
        print(f"\n  3Way_heavyreg gate weights:")
        for k, v in gw.items():
            print(f"    {k:<20}: {v:.4f}")

    if "4Way_e2e_heavy" in dl_results and "gate_weights" in dl_results["4Way_e2e_heavy"]:
        gw = dl_results["4Way_e2e_heavy"]["gate_weights"]
        print(f"\n  4Way_e2e_heavy gate weights:")
        for k, v in gw.items():
            print(f"    {k:<20}: {v:.4f}")

    print("\n" + "=" * 80)
    print(f"Results saved to {OUTPUT_DIR}")
    print("=" * 80)

    return results_json


if __name__ == "__main__":
    main()
