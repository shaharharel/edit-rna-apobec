#!/usr/bin/env python3
"""Phase 3 Neural Validation: XGBoost on RNA-FM (1280-dim) features.

Uses pre-computed RNA-FM embeddings (640-dim orig + 640-dim edit delta = 1280-dim)
as features for XGBoost, serving as a Phase 3 neural model proxy.

Experiments:
  1. 5-fold CV on v3 training data (per-enzyme binary classification)
  2. TCGA somatic mutation enrichment (8 cancers with RNA-FM embeddings)
  3. Multi-class enzyme TCGA scoring (7-class)

Comparison: original XGBoost 40-dim hand features.

Usage:
    conda run -n quris python experiments/multi_enzyme/exp_phase3_neural_validation.py
"""

import gc
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.apobec_feature_extraction import (
    extract_motif_from_seq,
    LOOP_FEATURE_COLS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# =============================================================================
# Paths
# =============================================================================
DATA_DIR = PROJECT_ROOT / "data"
EMB_DIR = DATA_DIR / "processed/multi_enzyme/embeddings"
MULTI_SPLITS = DATA_DIR / "processed/multi_enzyme/splits_multi_enzyme_v3_with_negatives.csv"
MULTI_SEQS = DATA_DIR / "processed/multi_enzyme/multi_enzyme_sequences_v3_with_negatives.json"
LOOP_CSV = DATA_DIR / "processed/multi_enzyme/loop_position_per_site_v3.csv"
STRUCT_CACHE = DATA_DIR / "processed/multi_enzyme/structure_cache_multi_enzyme_v3.npz"

# TCGA
RAW_SCORES_DIR = PROJECT_ROOT / "experiments/multi_enzyme/outputs/tcga_gnomad/raw_scores"

# Output
OUTPUT_DIR = PROJECT_ROOT / "experiments/multi_enzyme/outputs/phase3_neural_validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
CANCERS = ["lihc", "esca", "hnsc", "lusc", "brca", "cesc", "blca", "stad"]
ENZYMES = ["A3A", "A3B", "A3G", "A3A_A3G", "Neither"]


def make_xgb(n_features, seed=SEED, multiclass=False, n_classes=None):
    """Create XGBoost with hyperparams tuned to feature dimensionality."""
    # For high-dim: fewer trees, more regularization, lower learning rate
    if n_features > 500:
        params = dict(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.7,
            colsample_bytree=0.3,  # Sample 30% of 1280 cols = ~384
            reg_alpha=1.0,
            reg_lambda=5.0,
            min_child_weight=3,
        )
    else:
        params = dict(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
        )

    params.update(
        random_state=seed,
        use_label_encoder=False,
        tree_method="hist",
    )

    if multiclass:
        params["objective"] = "multi:softprob"
        params["eval_metric"] = "mlogloss"
        params["num_class"] = n_classes
    else:
        params["eval_metric"] = "logloss"

    return XGBClassifier(**params)


def load_v3_data():
    """Load v3 splits + sequences + RNA-FM embeddings + hand features."""
    logger.info("Loading v3 data...")
    df = pd.read_csv(MULTI_SPLITS)
    logger.info(f"  {len(df)} sites, enzymes: {df['enzyme'].value_counts().to_dict()}")

    with open(MULTI_SEQS) as f:
        seqs = json.load(f)

    orig_emb = torch.load(EMB_DIR / "rnafm_pooled_v3.pt", weights_only=False)
    edited_emb = torch.load(EMB_DIR / "rnafm_pooled_edited_v3.pt", weights_only=False)

    loop_df = pd.read_csv(LOOP_CSV).set_index("site_id")

    sc = np.load(STRUCT_CACHE, allow_pickle=True)
    struct_map = {sid: sc["delta_features"][i]
                  for i, sid in enumerate(sc["site_ids"])}

    logger.info(f"  Loaded: {len(seqs)} seqs, {len(orig_emb)} embeddings, "
                f"{len(loop_df)} loop, {len(struct_map)} struct")

    return df, seqs, orig_emb, edited_emb, loop_df, struct_map


def build_features_batch(site_ids, seqs, orig_emb, edited_emb, loop_df, struct_map,
                         include_hand=True, include_rnafm=True):
    """Build feature matrix using vectorized operations where possible."""
    n = len(site_ids)
    parts_list = []
    valid = np.ones(n, dtype=bool)

    if include_rnafm:
        orig_arr = np.zeros((n, 640), dtype=np.float32)
        delta_arr = np.zeros((n, 640), dtype=np.float32)
        for i, sid in enumerate(site_ids):
            if sid in orig_emb and sid in edited_emb:
                o = orig_emb[sid]
                e = edited_emb[sid]
                if isinstance(o, torch.Tensor):
                    o = o.numpy()
                    e = e.numpy()
                orig_arr[i] = o
                delta_arr[i] = e - o
            else:
                valid[i] = False
        parts_list.append(orig_arr)
        parts_list.append(delta_arr)

    if include_hand:
        motif_arr = np.zeros((n, 24), dtype=np.float32)
        loop_arr = np.zeros((n, len(LOOP_FEATURE_COLS)), dtype=np.float32)
        struct_arr = np.zeros((n, 7), dtype=np.float32)

        for i, sid in enumerate(site_ids):
            seq = seqs.get(sid, "")
            if len(seq) >= 201:
                motif_arr[i] = extract_motif_from_seq(seq[:201])

            if sid in loop_df.index:
                loop_arr[i] = loop_df.loc[sid, LOOP_FEATURE_COLS].values.astype(np.float32)

            if sid in struct_map:
                struct_arr[i] = struct_map[sid]

        parts_list.append(motif_arr)
        parts_list.append(struct_arr)
        parts_list.append(loop_arr)

    X = np.concatenate(parts_list, axis=1)
    X_valid = X[valid]
    return X_valid, valid


def run_cv_experiment(df, seqs, orig_emb, edited_emb, loop_df, struct_map):
    """Run 5-fold CV per enzyme with different feature sets."""
    logger.info("=" * 70)
    logger.info("EXPERIMENT 1: 5-fold CV on v3 training data")
    logger.info("=" * 70)

    results = {}
    feature_configs = {
        "XGB_HandFeatures_40d": {"include_hand": True, "include_rnafm": False},
        "XGB_RNAFM_1280d": {"include_hand": False, "include_rnafm": True},
        "XGB_RNAFM+Hand_1320d": {"include_hand": True, "include_rnafm": True},
    }

    for enzyme in ENZYMES:
        logger.info(f"\n--- Enzyme: {enzyme} ---")
        mask = df["enzyme"] == enzyme
        edf = df[mask].copy()

        if len(edf) == 0:
            continue

        site_ids = edf["site_id"].values
        labels = edf["is_edited"].values
        n_pos = (labels == 1).sum()
        n_neg = (labels == 0).sum()
        logger.info(f"  {len(edf)} sites ({n_pos} pos, {n_neg} neg)")

        if n_pos < 10 or n_neg < 10:
            continue

        results[enzyme] = {}

        for config_name, config in feature_configs.items():
            t0 = time.time()
            logger.info(f"  Model: {config_name}")
            X, valid = build_features_batch(site_ids, seqs, orig_emb, edited_emb,
                                            loop_df, struct_map, **config)
            valid_labels = labels[valid]

            if len(X) < 20:
                continue

            logger.info(f"    Features: {X.shape[1]}-dim, {len(X)} samples")

            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
            fold_metrics = []

            for fold, (train_idx, test_idx) in enumerate(skf.split(X, valid_labels)):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = valid_labels[train_idx], valid_labels[test_idx]

                clf = make_xgb(X.shape[1], seed=SEED + fold)
                clf.fit(X_train, y_train, verbose=False)
                probs = clf.predict_proba(X_test)[:, 1]

                auroc = roc_auc_score(y_test, probs)
                auprc = average_precision_score(y_test, probs)
                preds = (probs >= 0.5).astype(int)
                f1 = f1_score(y_test, preds)
                fold_metrics.append({"auroc": auroc, "auprc": auprc, "f1": f1})

            mean_auroc = np.mean([m["auroc"] for m in fold_metrics])
            std_auroc = np.std([m["auroc"] for m in fold_metrics])
            mean_auprc = np.mean([m["auprc"] for m in fold_metrics])
            mean_f1 = np.mean([m["f1"] for m in fold_metrics])
            elapsed = time.time() - t0

            logger.info(f"    AUROC: {mean_auroc:.3f} +/- {std_auroc:.3f} ({elapsed:.0f}s)")
            logger.info(f"    AUPRC: {mean_auprc:.3f}, F1: {mean_f1:.3f}")

            results[enzyme][config_name] = {
                "mean_auroc": round(mean_auroc, 4),
                "std_auroc": round(std_auroc, 4),
                "mean_auprc": round(mean_auprc, 4),
                "mean_f1": round(mean_f1, 4),
                "n_samples": len(X),
                "n_features": X.shape[1],
                "fold_metrics": [{k: round(v, 4) for k, v in m.items()} for m in fold_metrics],
                "elapsed_sec": round(elapsed, 1),
            }

    return results


def train_full_models(df, seqs, orig_emb, edited_emb, loop_df, struct_map):
    """Train full models on all v3 data for external scoring."""
    logger.info("\nTraining full models for external scoring...")
    models = {}

    # Per-enzyme binary models (1280-dim RNA-FM only, for TCGA scoring)
    for enzyme in ENZYMES:
        mask = df["enzyme"] == enzyme
        edf = df[mask].copy()
        if len(edf) == 0:
            continue

        site_ids = edf["site_id"].values
        labels = edf["is_edited"].values
        if (labels == 1).sum() < 10 or (labels == 0).sum() < 10:
            continue

        X, valid = build_features_batch(site_ids, seqs, orig_emb, edited_emb,
                                        loop_df, struct_map,
                                        include_hand=False, include_rnafm=True)
        y = labels[valid]

        clf = make_xgb(X.shape[1], seed=SEED)
        clf.fit(X, y, verbose=False)
        models[f"binary_{enzyme}_1280d"] = clf
        logger.info(f"  Trained binary {enzyme} 1280d: {len(X)} samples")

    # Multi-class model (positives only, enzyme as label)
    pos_mask = df["is_edited"] == 1
    pos_df = df[pos_mask].copy()
    enzyme_to_label = {e: i for i, e in enumerate(ENZYMES)}
    enzyme_to_label["Unknown"] = len(ENZYMES)
    inv_map = {v: k for k, v in enzyme_to_label.items()}

    mc_mask = pos_df["enzyme"].isin(enzyme_to_label.keys())
    mc_df = pos_df[mc_mask].copy()

    site_ids = mc_df["site_id"].values
    mc_labels = mc_df["enzyme"].map(enzyme_to_label).values

    X_mc, valid = build_features_batch(site_ids, seqs, orig_emb, edited_emb,
                                       loop_df, struct_map,
                                       include_hand=False, include_rnafm=True)
    mc_y = mc_labels[valid]

    clf_mc = make_xgb(X_mc.shape[1], seed=SEED, multiclass=True,
                       n_classes=len(enzyme_to_label))
    clf_mc.fit(X_mc, mc_y, verbose=False)
    models["multiclass_1280d"] = clf_mc
    models["multiclass_label_map"] = enzyme_to_label
    models["multiclass_inv_map"] = inv_map
    logger.info(f"  Trained multi-class 1280d: {len(X_mc)} samples, {len(enzyme_to_label)} classes")

    return models


def compute_enrichment_stats(scores, types, tc_context=None):
    """Compute enrichment statistics for mutation vs control scores."""
    mut_mask = types == "mutation"
    ctrl_mask = types == "control"
    mut_scores = scores[mut_mask]
    ctrl_scores = scores[ctrl_mask]

    result = {
        "mean_mut": float(np.mean(mut_scores)),
        "mean_ctrl": float(np.mean(ctrl_scores)),
        "delta": float(np.mean(mut_scores) - np.mean(ctrl_scores)),
    }

    # Mann-Whitney
    if len(mut_scores) > 0 and len(ctrl_scores) > 0:
        U, p = stats.mannwhitneyu(mut_scores, ctrl_scores, alternative="greater")
        result["mw_p"] = float(p)
    else:
        result["mw_p"] = 1.0

    # TC-stratified
    if tc_context is not None:
        for ctx_name, ctx_val in [("TC", 1), ("nonTC", 0)]:
            ctx_mask = tc_context == ctx_val
            ctx_mut = mut_mask & ctx_mask
            ctx_ctrl = ctrl_mask & ctx_mask
            if ctx_mut.sum() > 10 and ctx_ctrl.sum() > 10:
                result[f"{ctx_name}_delta"] = float(
                    np.mean(scores[ctx_mut]) - np.mean(scores[ctx_ctrl]))
                _, p2 = stats.mannwhitneyu(scores[ctx_mut], scores[ctx_ctrl],
                                           alternative="greater")
                result[f"{ctx_name}_p"] = float(p2)

    # Percentile threshold enrichment
    thresholds = {}
    for pct in [50, 75, 90, 95]:
        thr = np.percentile(scores, pct)
        mut_above = (mut_scores >= thr).sum()
        ctrl_above = (ctrl_scores >= thr).sum()
        mut_below = len(mut_scores) - mut_above
        ctrl_below = len(ctrl_scores) - ctrl_above

        if ctrl_above > 0 and ctrl_below > 0:
            table = [[mut_above, ctrl_above], [mut_below, ctrl_below]]
            OR, fisher_p = stats.fisher_exact(table, alternative="greater")
        else:
            OR, fisher_p = float("nan"), 1.0

        thresholds[f"p{pct}"] = {
            "threshold": round(float(thr), 4),
            "OR": round(float(OR), 4),
            "p": float(fisher_p),
            "mut_above": int(mut_above),
            "ctrl_above": int(ctrl_above),
        }
    result["thresholds"] = thresholds

    return result


def score_tcga(models):
    """Score TCGA mutations and controls with Phase 3 RNA-FM models."""
    logger.info("=" * 70)
    logger.info("EXPERIMENT 2: TCGA Enrichment with Phase 3 RNA-FM")
    logger.info("=" * 70)

    tcga_results = {}

    for cancer in CANCERS:
        t0 = time.time()
        logger.info(f"\n--- Cancer: {cancer.upper()} ---")

        emb_path = EMB_DIR / f"rnafm_tcga_{cancer}.pt"
        if not emb_path.exists():
            continue

        d = torch.load(emb_path, weights_only=False)
        n_mut = d["n_mut"]
        n_ctrl = d["n_ctrl"]
        pooled_orig = d["pooled_orig"].numpy()
        pooled_edited = d["pooled_edited"].numpy()

        logger.info(f"  {n_mut} mutations, {n_ctrl} controls")

        # Build 1280-dim features
        delta = pooled_edited - pooled_orig
        X_tcga = np.concatenate([pooled_orig, delta], axis=1).astype(np.float32)

        # Load tc_context from existing raw scores
        scores_path = RAW_SCORES_DIR / f"{cancer}_scores.csv"
        if scores_path.exists():
            raw_df = pd.read_csv(scores_path)
            tc_context = raw_df["tc_context"].values
            types = raw_df["type"].values
        else:
            types = np.array(["mutation"] * n_mut + ["control"] * n_ctrl)
            tc_context = None

        cancer_results = {
            "cancer": cancer,
            "n_mutations": int(n_mut),
            "n_controls": int(n_ctrl),
        }

        # Score with A3A binary model
        model_key = "binary_A3A_1280d"
        if model_key in models:
            clf = models[model_key]
            scores = clf.predict_proba(X_tcga)[:, 1]

            enrichment = compute_enrichment_stats(scores, types, tc_context)
            cancer_results["binary_A3A_1280d"] = enrichment

            logger.info(f"  A3A binary: mut={enrichment['mean_mut']:.4f}, "
                        f"ctrl={enrichment['mean_ctrl']:.4f}, "
                        f"delta={enrichment['delta']:.4f}, p={enrichment['mw_p']:.2e}")

            # Save raw scores
            raw_out = pd.DataFrame({
                "type": types,
                "score_rnafm_1280d": scores,
                "tc_context": tc_context if tc_context is not None else -1,
            })
            raw_out.to_csv(OUTPUT_DIR / f"tcga_{cancer}_rnafm_scores.csv", index=False)
            del scores

        elapsed = time.time() - t0
        logger.info(f"  Elapsed: {elapsed:.0f}s")
        tcga_results[cancer] = cancer_results

        del d, pooled_orig, pooled_edited, X_tcga
        gc.collect()

    return tcga_results


def score_tcga_multiclass(models):
    """Score TCGA with multi-class enzyme model."""
    logger.info("=" * 70)
    logger.info("EXPERIMENT 3: Multi-class Enzyme TCGA Scoring")
    logger.info("=" * 70)

    inv_map = models.get("multiclass_inv_map", {})
    label_map = models.get("multiclass_label_map", {})
    clf = models.get("multiclass_1280d")
    if clf is None:
        logger.warning("No multi-class model available")
        return {}

    mc_results = {}

    for cancer in CANCERS:
        t0 = time.time()
        logger.info(f"\n--- Cancer: {cancer.upper()} (multi-class) ---")
        emb_path = EMB_DIR / f"rnafm_tcga_{cancer}.pt"
        if not emb_path.exists():
            continue

        d = torch.load(emb_path, weights_only=False)
        n_mut = d["n_mut"]
        n_ctrl = d["n_ctrl"]
        pooled_orig = d["pooled_orig"].numpy()
        pooled_edited = d["pooled_edited"].numpy()
        delta = pooled_edited - pooled_orig
        X = np.concatenate([pooled_orig, delta], axis=1).astype(np.float32)

        scores_path = RAW_SCORES_DIR / f"{cancer}_scores.csv"
        if scores_path.exists():
            raw_df = pd.read_csv(scores_path)
            types = raw_df["type"].values
            tc_context = raw_df["tc_context"].values
        else:
            types = np.array(["mutation"] * n_mut + ["control"] * n_ctrl)
            tc_context = None

        mut_mask = types == "mutation"
        ctrl_mask = types == "control"

        probs = clf.predict_proba(X)

        cancer_mc = {"cancer": cancer, "n_mut": int(n_mut), "n_ctrl": int(n_ctrl)}

        for class_idx, enzyme_name in inv_map.items():
            if enzyme_name == "Unknown":
                continue
            p_enzyme = probs[:, class_idx]
            mut_mean = float(np.mean(p_enzyme[mut_mask]))
            ctrl_mean = float(np.mean(p_enzyme[ctrl_mask]))
            U, p_val = stats.mannwhitneyu(p_enzyme[mut_mask], p_enzyme[ctrl_mask],
                                           alternative="greater")

            cancer_mc[f"P({enzyme_name})_mut"] = round(mut_mean, 5)
            cancer_mc[f"P({enzyme_name})_ctrl"] = round(ctrl_mean, 5)
            cancer_mc[f"P({enzyme_name})_delta"] = round(mut_mean - ctrl_mean, 5)
            cancer_mc[f"P({enzyme_name})_p"] = float(p_val)

            logger.info(f"  P({enzyme_name}): mut={mut_mean:.4f}, ctrl={ctrl_mean:.4f}, "
                        f"delta={mut_mean - ctrl_mean:.4f}, p={p_val:.2e}")

        # TC-stratified P(Neither) for GI cancers
        if tc_context is not None and "Neither" in label_map:
            neither_idx = label_map["Neither"]
            p_neither = probs[:, neither_idx]
            tc_mask = tc_context == 1
            for ctx_name, ctx_val in [("TC", True), ("nonTC", False)]:
                ctx_mask = tc_mask if ctx_val else ~tc_mask
                ctx_mut = mut_mask & ctx_mask
                ctx_ctrl = ctrl_mask & ctx_mask
                if ctx_mut.sum() > 10 and ctx_ctrl.sum() > 10:
                    cancer_mc[f"P(Neither)_{ctx_name}_mut"] = round(float(np.mean(p_neither[ctx_mut])), 5)
                    cancer_mc[f"P(Neither)_{ctx_name}_ctrl"] = round(float(np.mean(p_neither[ctx_ctrl])), 5)

        # Save per-sample probabilities for stad and lihc only (GI cancers)
        if cancer in ["stad", "lihc"]:
            mc_df = pd.DataFrame({"type": types})
            if tc_context is not None:
                mc_df["tc_context"] = tc_context
            for class_idx, enzyme_name in inv_map.items():
                mc_df[f"P_{enzyme_name}"] = probs[:, class_idx]
            mc_df.to_csv(OUTPUT_DIR / f"tcga_{cancer}_multiclass_probs.csv", index=False)

        elapsed = time.time() - t0
        logger.info(f"  Elapsed: {elapsed:.0f}s")
        mc_results[cancer] = cancer_mc

        del d, pooled_orig, pooled_edited, X, probs
        gc.collect()

    return mc_results


def load_original_results():
    """Load original 40-dim XGBoost results for comparison."""
    original = {}

    tcga_path = PROJECT_ROOT / "experiments/multi_enzyme/outputs/tcga_gnomad/tcga_full_model_results.json"
    if tcga_path.exists():
        original["tcga"] = json.load(open(tcga_path))

    p3_path = PROJECT_ROOT / "experiments/multi_enzyme/outputs/phase3_replication/all_results_combined.json"
    if p3_path.exists():
        original["phase3_40d"] = json.load(open(p3_path))

    cv_path = PROJECT_ROOT / "experiments/multi_enzyme/outputs/phase3_replication/cv_results.json"
    if cv_path.exists():
        original["cv_40d"] = json.load(open(cv_path))

    return original


def generate_report(cv_results, tcga_results, mc_results, original, runtime_min):
    """Generate comparison report."""
    lines = []
    lines.append("# Phase 3 Neural Validation: RNA-FM XGBoost Results")
    lines.append("")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"Runtime: {runtime_min:.1f} minutes")
    lines.append("")

    # --- Overview ---
    lines.append("## Overview")
    lines.append("")
    lines.append("This experiment tests XGBoost on RNA-FM embeddings (1280-dim) as a Phase 3 proxy.")
    lines.append("RNA-FM captures learned RNA structure/function; edit delta captures the intervention effect.")
    lines.append("")
    lines.append("Feature configurations:")
    lines.append("- **XGB_40d**: motif (24) + struct delta (7) + loop (9) = 40-dim hand features")
    lines.append("- **XGB_RNAFM_1280d**: RNA-FM orig (640) + edit delta (640) = 1280-dim")
    lines.append("- **XGB_RNAFM+Hand_1320d**: all combined = 1320-dim")
    lines.append("")
    lines.append("XGBoost hyperparams for 1280d+: n_estimators=150, max_depth=5, colsample_bytree=0.3, reg_alpha=1, reg_lambda=5")
    lines.append("")

    # --- CV Results ---
    lines.append("## 1. Classification (5-fold CV on v3 data)")
    lines.append("")
    lines.append("| Enzyme | n | XGB_40d | XGB_RNAFM_1280d | XGB_1320d | Phase3 Neural* |")
    lines.append("|--------|---|---------|-----------------|-----------|----------------|")

    phase3_neural = {"A3A": 0.885, "A3B": 0.821, "A3G": 0.906, "A3A_A3G": 0.959, "Neither": 0.870}

    for enzyme in ENZYMES:
        if enzyme not in cv_results:
            continue
        er = cv_results[enzyme]
        n = next((v.get("n_samples", "?") for v in er.values()), "?")

        vals = {}
        for cfg in ["XGB_HandFeatures_40d", "XGB_RNAFM_1280d", "XGB_RNAFM+Hand_1320d"]:
            v = er.get(cfg, {}).get("mean_auroc")
            s = er.get(cfg, {}).get("std_auroc", 0)
            vals[cfg] = f"{v:.3f}+/-{s:.3f}" if v else "---"

        neural = phase3_neural.get(enzyme, "---")
        neural_str = f"{neural:.3f}" if isinstance(neural, float) else neural

        lines.append(f"| {enzyme} | {n} | {vals['XGB_HandFeatures_40d']} | "
                     f"{vals['XGB_RNAFM_1280d']} | {vals['XGB_RNAFM+Hand_1320d']} | {neural_str} |")

    lines.append("")
    lines.append("*Phase3 Neural = shared encoder + per-enzyme adapter heads (prior experiment, different training)")
    lines.append("")

    # AUPRC comparison
    lines.append("### AUPRC Comparison")
    lines.append("")
    lines.append("| Enzyme | XGB_40d | XGB_RNAFM_1280d | XGB_1320d |")
    lines.append("|--------|---------|-----------------|-----------|")
    for enzyme in ENZYMES:
        if enzyme not in cv_results:
            continue
        er = cv_results[enzyme]
        row = f"| {enzyme} |"
        for cfg in ["XGB_HandFeatures_40d", "XGB_RNAFM_1280d", "XGB_RNAFM+Hand_1320d"]:
            v = er.get(cfg, {}).get("mean_auprc")
            row += f" {v:.3f} |" if v else " --- |"
        lines.append(row)
    lines.append("")

    # --- TCGA Results ---
    lines.append("## 2. TCGA Somatic Mutation Enrichment (A3A Binary)")
    lines.append("")
    lines.append("Positive delta = mutations score higher than controls (enrichment).")
    lines.append("")
    lines.append("| Cancer | n_mut | n_ctrl | XGB_40d delta | RNAFM delta | RNAFM p | RNAFM TC delta | RNAFM nonTC delta |")
    lines.append("|--------|-------|--------|---------------|-------------|---------|----------------|-------------------|")

    orig_tcga = original.get("tcga", {})

    for cancer in CANCERS:
        if cancer not in tcga_results:
            continue
        tr = tcga_results[cancer]
        n_mut = tr["n_mutations"]
        n_ctrl = tr["n_controls"]

        # Original 40d
        orig_delta = "---"
        if cancer in orig_tcga:
            od = orig_tcga[cancer]
            d40 = od.get("delta", od.get("mean_score_mutations", 0) - od.get("mean_score_controls", 0))
            orig_delta = f"{d40:.4f}"

        # New RNA-FM
        enr = tr.get("binary_A3A_1280d", {})
        rnafm_delta = f"{enr.get('delta', 0):.4f}"
        rnafm_p = f"{enr.get('mw_p', 1.0):.2e}"
        tc = enr.get("TC_delta")
        ntc = enr.get("nonTC_delta")
        tc_str = f"{tc:.4f}" if tc is not None else "---"
        ntc_str = f"{ntc:.4f}" if ntc is not None else "---"

        lines.append(f"| {cancer.upper()} | {n_mut:,} | {n_ctrl:,} | {orig_delta} | "
                     f"{rnafm_delta} | {rnafm_p} | {tc_str} | {ntc_str} |")

    lines.append("")

    # Threshold ORs
    lines.append("### Percentile Threshold Odds Ratios")
    lines.append("")
    lines.append("| Cancer | p50 OR (p) | p75 OR (p) | p90 OR (p) | p95 OR (p) |")
    lines.append("|--------|------------|------------|------------|------------|")

    for cancer in CANCERS:
        if cancer not in tcga_results:
            continue
        enr = tcga_results[cancer].get("binary_A3A_1280d", {})
        thresholds = enr.get("thresholds", {})

        cells = []
        for pct in ["p50", "p75", "p90", "p95"]:
            t = thresholds.get(pct, {})
            o = t.get("OR", float("nan"))
            p = t.get("p", 1.0)
            if not np.isnan(o):
                sig = "**" if p < 0.001 else "*" if p < 0.05 else ""
                cells.append(f"{o:.3f}{sig}")
            else:
                cells.append("---")

        lines.append(f"| {cancer.upper()} | {' | '.join(cells)} |")

    lines.append("")
    lines.append("** p<0.001, * p<0.05")
    lines.append("")

    # --- Multi-class ---
    lines.append("## 3. Multi-class Enzyme TCGA Scoring (1280d)")
    lines.append("")
    lines.append("Mean P(enzyme) difference: mutations - controls")
    lines.append("")
    lines.append("| Cancer | dP(A3A) | dP(A3B) | dP(A3G) | dP(Both) | dP(Neither) |")
    lines.append("|--------|---------|---------|---------|----------|-------------|")

    for cancer in CANCERS:
        if cancer not in mc_results:
            continue
        mr = mc_results[cancer]
        row = f"| {cancer.upper()} |"
        for e in ENZYMES:
            d = mr.get(f"P({e})_delta")
            p = mr.get(f"P({e})_p", 1.0)
            if d is not None:
                sig = "**" if p < 0.001 else "*" if p < 0.05 else ""
                row += f" {d:+.4f}{sig} |"
            else:
                row += " --- |"
        lines.append(row)

    lines.append("")
    lines.append("** p<0.001, * p<0.05")
    lines.append("")

    # P(Neither) in GI cancers
    lines.append("### P(Neither) in GI Cancers")
    lines.append("")
    for cancer in ["stad", "lihc"]:
        if cancer not in mc_results:
            continue
        mr = mc_results[cancer]
        lines.append(f"**{cancer.upper()}:**")
        for key in sorted(mr.keys()):
            if "Neither" in key and key != f"P(Neither)_p":
                val = mr[key]
                lines.append(f"- {key}: {val:.5f}" if isinstance(val, (int, float)) else f"- {key}: {val}")
        lines.append("")

    # --- Summary ---
    lines.append("## Summary of Key Findings")
    lines.append("")

    for enzyme in ENZYMES:
        if enzyme in cv_results:
            er = cv_results[enzyme]
            hand = er.get("XGB_HandFeatures_40d", {}).get("mean_auroc")
            rnafm = er.get("XGB_RNAFM_1280d", {}).get("mean_auroc")
            both = er.get("XGB_RNAFM+Hand_1320d", {}).get("mean_auroc")
            if hand and rnafm:
                diff = rnafm - hand
                lines.append(f"- **{enzyme}**: 40d={hand:.3f}, 1280d={rnafm:.3f} "
                             f"({'better' if diff > 0 else 'worse'} by {abs(diff):.3f}), "
                             f"1320d={both:.3f}" if both else "")

    lines.append("")

    # TCGA summary
    n_enriched = sum(1 for c in tcga_results.values()
                     if c.get("binary_A3A_1280d", {}).get("delta", 0) > 0)
    lines.append(f"- TCGA: {n_enriched}/{len(tcga_results)} cancers show mutation enrichment with RNA-FM model")
    lines.append("")

    report = "\n".join(lines)
    report_path = OUTPUT_DIR / "comparison_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    logger.info(f"\nReport saved to {report_path}")
    return report


def main():
    start_time = time.time()

    df, seqs, orig_emb, edited_emb, loop_df, struct_map = load_v3_data()

    # Experiment 1: 5-fold CV
    cv_results = run_cv_experiment(df, seqs, orig_emb, edited_emb, loop_df, struct_map)

    # Save intermediate CV results
    with open(OUTPUT_DIR / "cv_results.json", "w") as f:
        json.dump(cv_results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)
    logger.info("CV results saved")

    # Train full models
    models = train_full_models(df, seqs, orig_emb, edited_emb, loop_df, struct_map)

    # Experiment 2: TCGA enrichment
    tcga_results = score_tcga(models)

    # Experiment 3: Multi-class TCGA
    mc_results = score_tcga_multiclass(models)

    # Load original results
    original = load_original_results()

    # Save all results
    all_results = {
        "cv_results": cv_results,
        "tcga_results": tcga_results,
        "multiclass_tcga_results": mc_results,
    }

    with open(OUTPUT_DIR / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2,
                  default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)

    runtime_min = (time.time() - start_time) / 60.0
    generate_report(cv_results, tcga_results, mc_results, original, runtime_min)

    logger.info(f"\nTotal runtime: {runtime_min:.1f} minutes")
    logger.info(f"All results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
