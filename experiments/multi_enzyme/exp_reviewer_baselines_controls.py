#!/usr/bin/env python3
"""Reviewer-requested analyses: Logistic Regression baseline, StructOnly TCGA ablation,
and per-sample APOBEC signature stratification for BRCA.

Three analyses:
  1. Logistic Regression vs XGBoost on 40-dim hand features (A3A, A3B, A3G)
  2. StructOnly (16-dim) XGBoost TCGA enrichment across 6 cancers
  3. BRCA per-sample APOBEC signature stratification

Usage:
    conda run -n quris python experiments/multi_enzyme/exp_reviewer_baselines_controls.py
"""

import gc
import gzip
import json
import logging
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.apobec_feature_extraction import (
    extract_motif_from_seq,
    LOOP_FEATURE_COLS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ===========================================================================
# Paths
# ===========================================================================
DATA_DIR = PROJECT_ROOT / "data"
MULTI_SPLITS = DATA_DIR / "processed/multi_enzyme/splits_multi_enzyme_v3_with_negatives.csv"
MULTI_SEQS = DATA_DIR / "processed/multi_enzyme/multi_enzyme_sequences_v3_with_negatives.json"
LOOP_CSV = DATA_DIR / "processed/multi_enzyme/loop_position_per_site_v3.csv"
STRUCT_CACHE = DATA_DIR / "processed/multi_enzyme/structure_cache_multi_enzyme_v3.npz"

VIENNA_CACHE_DIR = PROJECT_ROOT / "experiments/multi_enzyme/outputs/tcga_gnomad/vienna_cache"
RAW_SCORES_DIR = PROJECT_ROOT / "experiments/multi_enzyme/outputs/tcga_gnomad/raw_scores"
TCGA_DIR = DATA_DIR / "raw/tcga"

OUTPUT_DIR = PROJECT_ROOT / "experiments/multi_enzyme/outputs/tcga_gnomad/baselines_and_controls"

SEED = 42
N_FOLDS = 5
CANCERS = ["blca", "brca", "cesc", "lusc", "hnsc", "skcm"]


# ===========================================================================
# Data loading
# ===========================================================================

def load_training_data():
    """Load multi-enzyme v3 data and build feature matrices."""
    logger.info("Loading training data...")
    splits = pd.read_csv(MULTI_SPLITS)
    with open(MULTI_SEQS) as f:
        sequences = json.load(f)

    loop_df = pd.read_csv(LOOP_CSV)
    loop_df["site_id"] = loop_df["site_id"].astype(str)
    loop_df = loop_df.set_index("site_id")

    struct_data = None
    if STRUCT_CACHE.exists():
        struct_data = np.load(str(STRUCT_CACHE), allow_pickle=True)

    site_ids = splits["site_id"].astype(str).tolist()
    labels = splits["is_edited"].values
    enzymes = splits["enzyme"].values

    # Motif features (24-dim)
    motif_feats = np.array(
        [extract_motif_from_seq(sequences.get(sid, "N" * 201)) for sid in site_ids],
        dtype=np.float32,
    )

    # Structure delta features (7-dim)
    struct_feats = np.zeros((len(site_ids), 7), dtype=np.float32)
    if struct_data is not None:
        struct_sids = [str(s) for s in struct_data["site_ids"]]
        sid_to_idx = {s: i for i, s in enumerate(struct_sids)}
        for i, sid in enumerate(site_ids):
            if sid in sid_to_idx:
                struct_feats[i] = struct_data["delta_features"][sid_to_idx[sid]]
        del struct_data
        gc.collect()

    # Loop geometry features (9-dim)
    loop_feats = np.zeros((len(site_ids), len(LOOP_FEATURE_COLS)), dtype=np.float32)
    for i, sid in enumerate(site_ids):
        if sid in loop_df.index:
            for j, col in enumerate(LOOP_FEATURE_COLS):
                if col in loop_df.columns:
                    val = loop_df.loc[sid, col]
                    if not isinstance(val, (int, float)):
                        val = val.iloc[0] if hasattr(val, "iloc") else 0.0
                    loop_feats[i, j] = float(val) if pd.notna(val) else 0.0

    # Full 40-dim
    X_full = np.concatenate([motif_feats, struct_feats, loop_feats], axis=1)
    X_full = np.nan_to_num(X_full, nan=0.0)

    # Motif only (24-dim)
    X_motif = motif_feats.copy()

    # Struct only (16-dim = 7 delta + 9 loop)
    X_struct = np.concatenate([struct_feats, loop_feats], axis=1)
    X_struct = np.nan_to_num(X_struct, nan=0.0)

    logger.info(f"  Loaded {len(labels)} sites, {int(labels.sum())} positives")
    logger.info(f"  Feature dims: full={X_full.shape[1]}, motif={X_motif.shape[1]}, struct={X_struct.shape[1]}")

    return {
        "splits": splits,
        "labels": labels,
        "enzymes": enzymes,
        "X_full": X_full,
        "X_motif": X_motif,
        "X_struct": X_struct,
        "sequences": sequences,
        "site_ids": site_ids,
    }


# ===========================================================================
# Analysis 1: Logistic Regression Baseline
# ===========================================================================

def run_logistic_regression_baseline(data):
    """5-fold CV comparison: LogReg vs XGBoost on 40-dim, MotifOnly, StructOnly."""
    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS 1: Logistic Regression Baseline Comparison")
    logger.info("=" * 70)

    results = {}

    for enzyme in ["A3A", "A3B", "A3G"]:
        mask = data["enzymes"] == enzyme
        if mask.sum() == 0:
            continue

        y = data["labels"][mask]
        n_pos = int(y.sum())
        n_neg = int(len(y) - n_pos)
        logger.info(f"\n--- {enzyme}: {n_pos} pos, {n_neg} neg ---")

        enzyme_results = {}
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

        # Model configs: (name, X_key, model_factory)
        configs = [
            ("LogReg_40dim", "X_full", lambda: LogisticRegression(
                max_iter=2000, C=1.0, random_state=SEED, solver="lbfgs")),
            ("XGB_40dim", "X_full", lambda: XGBClassifier(
                n_estimators=500, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                random_state=SEED, eval_metric="logloss")),
            ("XGB_MotifOnly", "X_motif", lambda: XGBClassifier(
                n_estimators=500, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                random_state=SEED, eval_metric="logloss")),
            ("XGB_StructOnly", "X_struct", lambda: XGBClassifier(
                n_estimators=500, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                random_state=SEED, eval_metric="logloss")),
        ]

        for model_name, x_key, model_factory in configs:
            X = data[x_key][mask]
            fold_aurocs = []
            fold_auprcs = []

            for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                model = model_factory()

                # Scale features for logistic regression
                if "LogReg" in model_name:
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)

                model.fit(X_train, y_train)
                y_prob = model.predict_proba(X_test)[:, 1]

                auroc = roc_auc_score(y_test, y_prob)
                auprc = average_precision_score(y_test, y_prob)
                fold_aurocs.append(auroc)
                fold_auprcs.append(auprc)

            mean_auroc = np.mean(fold_aurocs)
            std_auroc = np.std(fold_aurocs)
            mean_auprc = np.mean(fold_auprcs)
            std_auprc = np.std(fold_auprcs)

            enzyme_results[model_name] = {
                "auroc_mean": mean_auroc,
                "auroc_std": std_auroc,
                "auprc_mean": mean_auprc,
                "auprc_std": std_auprc,
                "fold_aurocs": fold_aurocs,
                "fold_auprcs": fold_auprcs,
            }

            logger.info(f"  {model_name:20s}: AUROC={mean_auroc:.4f}+/-{std_auroc:.4f}  "
                         f"AUPRC={mean_auprc:.4f}+/-{std_auprc:.4f}")

        results[enzyme] = enzyme_results

    return results


def plot_logistic_regression(results, output_dir):
    """Create bar chart comparing LogReg vs XGBoost across enzymes."""
    enzymes = list(results.keys())
    model_names = ["LogReg_40dim", "XGB_40dim", "XGB_MotifOnly", "XGB_StructOnly"]
    display_names = ["LogReg (40d)", "XGB (40d)", "XGB Motif (24d)", "XGB Struct (16d)"]
    colors = ["#4477AA", "#EE6677", "#CCBB44", "#66CCEE"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for metric_idx, (metric, metric_label) in enumerate([("auroc", "AUROC"), ("auprc", "AUPRC")]):
        ax = axes[metric_idx]
        x = np.arange(len(enzymes))
        width = 0.18

        for j, (mname, dname, color) in enumerate(zip(model_names, display_names, colors)):
            means = [results[e][mname][f"{metric}_mean"] for e in enzymes]
            stds = [results[e][mname][f"{metric}_std"] for e in enzymes]
            ax.bar(x + j * width, means, width, yerr=stds, label=dname,
                   color=color, edgecolor="white", capsize=3, alpha=0.85)

        ax.set_xlabel("Enzyme")
        ax.set_ylabel(metric_label)
        ax.set_title(f"5-Fold CV {metric_label}")
        ax.set_xticks(x + 1.5 * width)
        ax.set_xticklabels(enzymes)
        ax.legend(fontsize=8)
        ax.set_ylim(0.5, 1.0)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig_path = output_dir / "analysis1_logreg_vs_xgboost.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved figure: {fig_path}")


# ===========================================================================
# Analysis 2: StructOnly TCGA Ablation
# ===========================================================================

def derive_features_from_fold(fold_result):
    """Derive 16-dim structure features (7 delta + 9 loop) from cached fold data."""
    if fold_result is None:
        return np.zeros(16, dtype=np.float32)

    center = 100

    pair_wt = fold_result["bpp_wt_center"]
    pair_ed = fold_result["bpp_ed_center"]
    delta_pairing = pair_ed - pair_wt
    delta_accessibility = -delta_pairing
    delta_mfe = fold_result["mfe_ed"] - fold_result["mfe_wt"]

    bpp_wt_w = fold_result["bpp_wt_window"]
    bpp_ed_w = fold_result["bpp_ed_window"]

    def _entropy_from_p(p_val):
        if p_val <= 0 or p_val >= 1:
            return 0.0
        return -(p_val * np.log2(p_val + 1e-10) + (1 - p_val) * np.log2(1 - p_val + 1e-10))

    delta_entropy = _entropy_from_p(pair_ed) - _entropy_from_p(pair_wt)
    delta_window = np.array(bpp_ed_w) - np.array(bpp_wt_w)
    mean_delta_pairing = float(np.mean(delta_window)) if len(delta_window) > 0 else 0.0
    std_delta_pairing = float(np.std(delta_window)) if len(delta_window) > 0 else 0.0

    struct_delta = np.array([
        delta_pairing, delta_accessibility, delta_entropy,
        delta_mfe, mean_delta_pairing, std_delta_pairing, -mean_delta_pairing
    ], dtype=np.float32)

    # Loop geometry (9-dim)
    struct_wt = fold_result["struct_wt"]
    is_unpaired = 1.0 if struct_wt[center] == "." else 0.0

    loop_size = 0.0
    dist_to_junction = 0.0
    dist_to_apex = 0.0
    relative_loop_position = 0.5
    left_stem = 0.0
    right_stem = 0.0
    max_adj_stem = 0.0

    if is_unpaired:
        left = center
        while left > 0 and struct_wt[left] == ".":
            left -= 1
        right = center
        while right < len(struct_wt) - 1 and struct_wt[right] == ".":
            right += 1

        loop_size = float(right - left - 1)
        if loop_size > 0:
            pos_in_loop = center - left - 1
            relative_loop_position = pos_in_loop / max(loop_size - 1, 1)
            dist_to_apex = abs(pos_in_loop - (loop_size - 1) / 2)

        ls = 0
        i = left
        while i >= 0 and struct_wt[i] in "()":
            ls += 1
            i -= 1
        left_stem = float(ls)

        rs = 0
        i = right
        while i < len(struct_wt) and struct_wt[i] in "()":
            rs += 1
            i += 1
        right_stem = float(rs)
        max_adj_stem = max(left_stem, right_stem)

    local_region = struct_wt[max(0, center - 10):min(len(struct_wt), center + 11)]
    local_unpaired = sum(1 for ch in local_region if ch == ".") / max(len(local_region), 1)

    loop_features = np.array([
        is_unpaired, loop_size, dist_to_junction, dist_to_apex,
        relative_loop_position, left_stem, right_stem,
        max_adj_stem, local_unpaired
    ], dtype=np.float32)

    return np.concatenate([struct_delta, loop_features])


def train_struct_only_model(data):
    """Train StructOnly XGBoost on A3A data (16-dim: 7 delta + 9 loop)."""
    mask = data["enzymes"] == "A3A"
    X = data["X_struct"][mask]
    y = data["labels"][mask]

    model = XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=SEED, eval_metric="logloss",
    )
    model.fit(X, y)
    train_auc = roc_auc_score(y, model.predict_proba(X)[:, 1])
    logger.info(f"  StructOnly A3A training AUC: {train_auc:.4f}")
    return model


def train_full_model(data):
    """Train full 40-dim XGBoost on A3A data for comparison."""
    mask = data["enzymes"] == "A3A"
    X = data["X_full"][mask]
    y = data["labels"][mask]

    model = XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=SEED, eval_metric="logloss",
    )
    model.fit(X, y)
    train_auc = roc_auc_score(y, model.predict_proba(X)[:, 1])
    logger.info(f"  Full 40-dim A3A training AUC: {train_auc:.4f}")
    return model


def compute_enrichment(scores, types, tc_flags, thresholds):
    """Compute TC-stratified enrichment (odds ratio) at each threshold.

    Returns dict: threshold -> {OR_all, p_all, OR_tc, p_tc, OR_nontc, p_nontc, ...}
    """
    results = {}
    for t in thresholds:
        high = scores >= t
        low = ~high

        for subset_name, subset_mask in [("all", np.ones(len(scores), dtype=bool)),
                                          ("tc", tc_flags == 1),
                                          ("nontc", tc_flags == 0)]:
            m = subset_mask
            a = ((types == "mutation") & high & m).sum()
            b = ((types == "control") & high & m).sum()
            c = ((types == "mutation") & low & m).sum()
            d = ((types == "control") & low & m).sum()

            if b > 0 and c > 0:
                or_val = (a * d) / (b * c) if (b * c) > 0 else float("inf")
            else:
                or_val = float("nan")

            # Fisher's exact test on 2x2 table
            try:
                _, p_val = stats.fisher_exact([[a, b], [c, d]])
            except Exception:
                p_val = 1.0

            results.setdefault(t, {})[f"OR_{subset_name}"] = or_val
            results.setdefault(t, {})[f"p_{subset_name}"] = p_val
            results.setdefault(t, {})[f"n_mut_high_{subset_name}"] = int(a)
            results.setdefault(t, {})[f"n_ctrl_high_{subset_name}"] = int(b)

    return results


def run_struct_only_tcga(data):
    """Score TCGA mutations with StructOnly model and compute enrichment."""
    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS 2: StructOnly (16-dim) TCGA Enrichment Ablation")
    logger.info("=" * 70)

    struct_model = train_struct_only_model(data)
    full_model = train_full_model(data)

    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    all_results = {}

    for cancer in CANCERS:
        logger.info(f"\n--- {cancer.upper()} ---")

        # Load vienna cache
        cache_path = VIENNA_CACHE_DIR / f"{cancer}_vienna_raw.json.gz"
        if not cache_path.exists():
            logger.warning(f"  No vienna cache for {cancer}, skipping")
            continue

        t0 = time.time()
        with gzip.open(str(cache_path), "rt") as f:
            cache = json.load(f)
        logger.info(f"  Loaded vienna cache: {cache['n_sequences']:,} sequences ({time.time()-t0:.1f}s)")

        # Load raw_scores for tc_context
        scores_path = RAW_SCORES_DIR / f"{cancer}_scores.csv"
        raw_df = pd.read_csv(scores_path)
        assert len(raw_df) == len(cache["fold_results"]), \
            f"Row mismatch: raw_scores={len(raw_df)}, vienna={len(cache['fold_results'])}"

        types = raw_df["type"].values
        tc_flags = raw_df["tc_context"].values
        original_scores = raw_df["score"].values

        # Derive 16-dim structure features from fold results
        logger.info("  Deriving 16-dim structure features from fold results...")
        n = len(cache["fold_results"])
        struct_features = np.zeros((n, 16), dtype=np.float32)
        for i, fr in enumerate(cache["fold_results"]):
            struct_features[i] = derive_features_from_fold(fr)
        struct_features = np.nan_to_num(struct_features, nan=0.0)

        # Score with StructOnly model
        struct_scores = struct_model.predict_proba(struct_features)[:, 1]

        # Also build full 40-dim features for full model re-scoring
        # (motif features require sequences which we don't have; use original scores)
        logger.info(f"  StructOnly scores: mean={struct_scores.mean():.4f}, "
                     f"median={np.median(struct_scores):.4f}")

        # Compute enrichment for StructOnly
        struct_enrich = compute_enrichment(struct_scores, types, tc_flags, thresholds)

        # Compute enrichment for original full model (from raw_scores)
        full_enrich = compute_enrichment(original_scores, types, tc_flags, thresholds)

        cancer_results = {
            "n_mut": int((types == "mutation").sum()),
            "n_ctrl": int((types == "control").sum()),
            "struct_enrichment": struct_enrich,
            "full_enrichment": full_enrich,
            "struct_score_stats": {
                "mean": float(struct_scores.mean()),
                "median": float(np.median(struct_scores)),
                "std": float(struct_scores.std()),
            },
        }

        # Log key results
        for t in [0.3, 0.5]:
            se = struct_enrich.get(t, {})
            fe = full_enrich.get(t, {})
            logger.info(f"  t={t}: StructOnly OR_tc={se.get('OR_tc', 'N/A'):.3f} (p={se.get('p_tc', 1):.2e}), "
                         f"Full OR_tc={fe.get('OR_tc', 'N/A'):.3f} (p={fe.get('p_tc', 1):.2e})")

        all_results[cancer] = cancer_results

        # Free memory
        del cache, struct_features, struct_scores
        gc.collect()

    return all_results


def plot_struct_only_tcga(results, output_dir):
    """Plot StructOnly vs Full enrichment comparison across cancers."""
    cancers = [c for c in CANCERS if c in results]
    if not cancers:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: OR_tc at threshold=0.5 for StructOnly vs Full
    ax = axes[0]
    x = np.arange(len(cancers))
    width = 0.35
    struct_ors = []
    full_ors = []
    for c in cancers:
        se = results[c]["struct_enrichment"].get(0.5, {})
        fe = results[c]["full_enrichment"].get(0.5, {})
        struct_ors.append(se.get("OR_tc", 1.0))
        full_ors.append(fe.get("OR_tc", 1.0))

    ax.bar(x - width / 2, struct_ors, width, label="StructOnly (16d)", color="#66CCEE", edgecolor="white")
    ax.bar(x + width / 2, full_ors, width, label="Full (40d)", color="#EE6677", edgecolor="white")
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Cancer Type")
    ax.set_ylabel("Odds Ratio (TC-context, t=0.5)")
    ax.set_title("TCGA Enrichment: StructOnly vs Full Model")
    ax.set_xticks(x)
    ax.set_xticklabels([c.upper() for c in cancers])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Panel B: OR_tc across thresholds for BRCA (or first cancer with data)
    ax = axes[1]
    target_cancer = "brca" if "brca" in results else cancers[0]
    thresholds = sorted(results[target_cancer]["struct_enrichment"].keys())
    struct_ors_t = [results[target_cancer]["struct_enrichment"][t].get("OR_tc", 1.0) for t in thresholds]
    full_ors_t = [results[target_cancer]["full_enrichment"][t].get("OR_tc", 1.0) for t in thresholds]

    ax.plot(thresholds, struct_ors_t, "o-", color="#66CCEE", label="StructOnly (16d)", linewidth=2)
    ax.plot(thresholds, full_ors_t, "s-", color="#EE6677", label="Full (40d)", linewidth=2)
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Score Threshold")
    ax.set_ylabel("Odds Ratio (TC-context)")
    ax.set_title(f"Enrichment vs Threshold ({target_cancer.upper()})")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig_path = output_dir / "analysis2_structonly_tcga_enrichment.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved figure: {fig_path}")


# ===========================================================================
# Analysis 3: BRCA Per-Sample APOBEC Signature Stratification
# ===========================================================================

def run_brca_apobec_stratification(data):
    """Stratify BRCA tumors by APOBEC signature strength and recompute enrichment."""
    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS 3: BRCA Per-Sample APOBEC Signature Stratification")
    logger.info("=" * 70)

    maf_path = TCGA_DIR / "brca_tcga_pan_can_atlas_2018_mutations.txt"
    if not maf_path.exists():
        logger.error(f"  BRCA MAF not found: {maf_path}")
        return {}

    # Step 1: Parse all C>T SNPs from BRCA MAF with sample barcodes
    logger.info("  Step 1: Parsing BRCA MAF for C>T mutations per sample...")
    sample_ct_total = {}   # sample -> total C>T count
    sample_tc_count = {}   # sample -> TC-context C>T count
    ct_mutations = []      # list of (chrom, pos, strand, sample)

    for chunk in pd.read_csv(maf_path, sep="\t", comment="#", low_memory=False, chunksize=100000):
        if "Variant_Type" in chunk.columns:
            chunk = chunk[chunk["Variant_Type"] == "SNP"]

        # C>T on + strand
        ct_plus = (chunk["Reference_Allele"] == "C") & (chunk["Tumor_Seq_Allele2"] == "T")
        # G>A on + strand = C>T on - strand
        ga_minus = (chunk["Reference_Allele"] == "G") & (chunk["Tumor_Seq_Allele2"] == "A")

        sub = chunk[ct_plus | ga_minus].copy()

        for _, row in sub.iterrows():
            sample = row["Tumor_Sample_Barcode"]
            ref = row["Reference_Allele"]

            sample_ct_total[sample] = sample_ct_total.get(sample, 0) + 1

            # Check TC context from the CONTEXT field if available
            context = str(row.get("CONTEXT", ""))
            if ref == "C" and len(context) >= 3:
                # Context is typically 5nt centered on the mutation
                center_idx = len(context) // 2
                if center_idx > 0 and context[center_idx - 1].upper() == "T":
                    sample_tc_count[sample] = sample_tc_count.get(sample, 0) + 1
            elif ref == "G" and len(context) >= 3:
                # Reverse complement: G>A means C>T on opposite strand
                # TC context on - strand = GA context on + strand
                center_idx = len(context) // 2
                if center_idx < len(context) - 1 and context[center_idx + 1].upper() == "A":
                    sample_tc_count[sample] = sample_tc_count.get(sample, 0) + 1

            chrom = str(row["Chromosome"])
            if not chrom.startswith("chr"):
                chrom = "chr" + chrom
            pos = int(row["Start_Position"]) - 1  # 0-based
            strand = "+" if ref == "C" else "-"
            ct_mutations.append((chrom, pos, strand, sample))

    # Compute TC fraction per sample
    sample_tc_frac = {}
    for sample in sample_ct_total:
        total = sample_ct_total[sample]
        tc = sample_tc_count.get(sample, 0)
        if total >= 5:  # minimum mutations to compute fraction
            sample_tc_frac[sample] = tc / total

    tc_fracs = np.array(list(sample_tc_frac.values()))
    median_tc = np.median(tc_fracs) if len(tc_fracs) > 0 else 0.5

    apobec_high = {s for s, f in sample_tc_frac.items() if f >= median_tc}
    apobec_low = {s for s, f in sample_tc_frac.items() if f < median_tc}

    logger.info(f"  Found {len(sample_tc_frac)} samples with >=5 C>T mutations")
    logger.info(f"  TC fraction: median={median_tc:.3f}, "
                 f"mean={tc_fracs.mean():.3f}, std={tc_fracs.std():.3f}")
    logger.info(f"  APOBEC-high: {len(apobec_high)} samples, APOBEC-low: {len(apobec_low)} samples")

    # Step 2: Build mutation -> sample mapping and match to raw_scores
    logger.info("  Step 2: Matching mutations to raw_scores via vienna cache index...")

    # The raw_scores CSV has type/score/tc_context aligned with vienna cache.
    # The first n_mut rows are mutations, next n_ctrl are controls.
    # We need to know which mutation row corresponds to which sample.
    # The mutations in the vienna cache were generated in the same order as the MAF parsing.
    # We need to re-parse the MAF the same way as the original scoring script.

    # Load raw_scores and vienna cache metadata
    raw_df = pd.read_csv(RAW_SCORES_DIR / "brca_scores.csv")
    with gzip.open(str(VIENNA_CACHE_DIR / "brca_vienna_raw.json.gz"), "rt") as f:
        cache_meta = json.load(f)

    n_mut = cache_meta["n_mut"]
    n_ctrl = cache_meta["n_ctrl"]
    logger.info(f"  Raw scores: {n_mut} mutations, {n_ctrl} controls")

    # Re-parse BRCA MAF to get C>T SNPs in order (matching the vienna cache)
    # The original script used parse_ct_mutations which deduplicates by (chrom, pos)
    mutation_samples = []
    seen_positions = set()
    sample_per_position = {}

    for chunk in pd.read_csv(maf_path, sep="\t", comment="#", low_memory=False, chunksize=100000):
        if "Variant_Type" in chunk.columns:
            chunk = chunk[chunk["Variant_Type"] == "SNP"]

        ct_plus = (chunk["Reference_Allele"] == "C") & (chunk["Tumor_Seq_Allele2"] == "T")
        ga_minus = (chunk["Reference_Allele"] == "G") & (chunk["Tumor_Seq_Allele2"] == "A")
        sub = chunk[ct_plus | ga_minus].copy()

        for _, row in sub.iterrows():
            chrom = str(row["Chromosome"])
            if not chrom.startswith("chr"):
                chrom = "chr" + chrom
            pos = int(row["Start_Position"]) - 1
            key = (chrom, pos)
            if key not in seen_positions:
                seen_positions.add(key)
                sample_per_position[key] = row["Tumor_Sample_Barcode"]

    # Now reconstruct the order of mutations in the vienna cache
    # The original parsing used pd.read_csv with drop_duplicates, keeping first occurrence
    # We parse again in the same way to get ordered unique mutations
    mutation_rows = []
    for chunk in pd.read_csv(maf_path, sep="\t", comment="#", low_memory=False, chunksize=100000):
        if "Variant_Type" in chunk.columns:
            chunk = chunk[chunk["Variant_Type"] == "SNP"]
        ct_plus = (chunk["Reference_Allele"] == "C") & (chunk["Tumor_Seq_Allele2"] == "T")
        ga_minus = (chunk["Reference_Allele"] == "G") & (chunk["Tumor_Seq_Allele2"] == "A")
        sub = chunk[ct_plus | ga_minus].copy()
        if len(sub) > 0:
            chrom = sub["Chromosome"].astype(str)
            if not chrom.str.startswith("chr").any():
                chrom = "chr" + chrom
            sub = sub.copy()
            sub["chrom"] = chrom
            sub["pos"] = sub["Start_Position"].astype(int) - 1
            mutation_rows.append(sub[["chrom", "pos", "Tumor_Sample_Barcode"]].copy())

    if mutation_rows:
        all_muts = pd.concat(mutation_rows, ignore_index=True)
        all_muts_dedup = all_muts.drop_duplicates(subset=["chrom", "pos"], keep="first")
        mutation_samples = all_muts_dedup["Tumor_Sample_Barcode"].values
        logger.info(f"  Parsed {len(mutation_samples)} unique C>T mutations from MAF")
        logger.info(f"  Expected {n_mut} mutations in vienna cache")

        # They should match
        if len(mutation_samples) != n_mut:
            logger.warning(f"  Mismatch: MAF has {len(mutation_samples)} unique C>T, "
                           f"cache has {n_mut}. Using min({len(mutation_samples)}, {n_mut})")
            n_use = min(len(mutation_samples), n_mut)
            mutation_samples = mutation_samples[:n_use]
    else:
        logger.error("  No C>T mutations found in MAF")
        return {}

    # Step 3: Assign each mutation row in raw_scores to APOBEC-high or low
    scores = raw_df["score"].values
    types = raw_df["type"].values
    tc_flags = raw_df["tc_context"].values

    # Mutation rows are first n_mut rows
    mut_in_high = np.zeros(n_mut, dtype=bool)
    mut_in_low = np.zeros(n_mut, dtype=bool)
    n_matched = 0
    for i in range(min(len(mutation_samples), n_mut)):
        sample = mutation_samples[i]
        if sample in apobec_high:
            mut_in_high[i] = True
            n_matched += 1
        elif sample in apobec_low:
            mut_in_low[i] = True
            n_matched += 1

    logger.info(f"  Matched {n_matched}/{n_mut} mutations to APOBEC-high/low")
    logger.info(f"  APOBEC-high mutations: {mut_in_high.sum()}")
    logger.info(f"  APOBEC-low mutations: {mut_in_low.sum()}")

    # Step 4: Compute enrichment separately for APOBEC-high and APOBEC-low
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    results = {
        "n_samples": len(sample_tc_frac),
        "median_tc_frac": float(median_tc),
        "mean_tc_frac": float(tc_fracs.mean()),
        "n_apobec_high_samples": len(apobec_high),
        "n_apobec_low_samples": len(apobec_low),
        "n_apobec_high_muts": int(mut_in_high.sum()),
        "n_apobec_low_muts": int(mut_in_low.sum()),
        "enrichment_high": {},
        "enrichment_low": {},
        "enrichment_all": {},
    }

    # Controls are the last n_ctrl rows
    ctrl_mask = types == "control"
    ctrl_scores = scores[ctrl_mask]
    ctrl_tc = tc_flags[ctrl_mask]

    for subset_name, mut_mask in [("high", mut_in_high), ("low", mut_in_low)]:
        for t in thresholds:
            for tc_label, tc_val in [("all", None), ("tc", 1), ("nontc", 0)]:
                # Mutations in this subset
                mut_scores_sub = scores[:n_mut][mut_mask]
                mut_tc_sub = tc_flags[:n_mut][mut_mask]

                if tc_val is not None:
                    mut_sub = mut_scores_sub[mut_tc_sub == tc_val]
                    ctrl_sub = ctrl_scores[ctrl_tc == tc_val]
                else:
                    mut_sub = mut_scores_sub
                    ctrl_sub = ctrl_scores

                a = (mut_sub >= t).sum()
                b = (ctrl_sub >= t).sum()
                c = (mut_sub < t).sum()
                d = (ctrl_sub < t).sum()

                if b > 0 and c > 0:
                    or_val = (a * d) / (b * c)
                else:
                    or_val = float("nan")

                try:
                    _, p_val = stats.fisher_exact([[int(a), int(b)], [int(c), int(d)]])
                except Exception:
                    p_val = 1.0

                results[f"enrichment_{subset_name}"].setdefault(t, {})[f"OR_{tc_label}"] = float(or_val)
                results[f"enrichment_{subset_name}"].setdefault(t, {})[f"p_{tc_label}"] = float(p_val)
                results[f"enrichment_{subset_name}"].setdefault(t, {})[f"n_mut_high_{tc_label}"] = int(a)

    # Also compute combined enrichment for reference
    for t in thresholds:
        for tc_label, tc_val in [("all", None), ("tc", 1), ("nontc", 0)]:
            mut_scores_all = scores[:n_mut]
            mut_tc_all = tc_flags[:n_mut]

            if tc_val is not None:
                mut_sub = mut_scores_all[mut_tc_all == tc_val]
                ctrl_sub = ctrl_scores[ctrl_tc == tc_val]
            else:
                mut_sub = mut_scores_all
                ctrl_sub = ctrl_scores

            a = (mut_sub >= t).sum()
            b = (ctrl_sub >= t).sum()
            c = (mut_sub < t).sum()
            d = (ctrl_sub < t).sum()

            if b > 0 and c > 0:
                or_val = (a * d) / (b * c)
            else:
                or_val = float("nan")

            try:
                _, p_val = stats.fisher_exact([[int(a), int(b)], [int(c), int(d)]])
            except Exception:
                p_val = 1.0

            results["enrichment_all"].setdefault(t, {})[f"OR_{tc_label}"] = float(or_val)
            results["enrichment_all"].setdefault(t, {})[f"p_{tc_label}"] = float(p_val)
            results["enrichment_all"].setdefault(t, {})[f"n_mut_high_{tc_label}"] = int(a)

    # Log key results
    for t in [0.3, 0.5]:
        eh = results["enrichment_high"].get(t, {})
        el = results["enrichment_low"].get(t, {})
        ea = results["enrichment_all"].get(t, {})
        logger.info(f"  t={t}  TC-context:")
        logger.info(f"    All:         OR={ea.get('OR_tc', 'N/A'):.3f} (p={ea.get('p_tc', 1):.2e})")
        logger.info(f"    APOBEC-high: OR={eh.get('OR_tc', 'N/A'):.3f} (p={eh.get('p_tc', 1):.2e})")
        logger.info(f"    APOBEC-low:  OR={el.get('OR_tc', 'N/A'):.3f} (p={el.get('p_tc', 1):.2e})")

    return results


def plot_brca_stratification(results, output_dir):
    """Plot BRCA APOBEC-high vs APOBEC-low enrichment."""
    if not results or "enrichment_high" not in results:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: TC fraction distribution
    ax = axes[0]
    # We don't store individual fractions, so show summary stats
    ax.text(0.5, 0.7, f"n_samples = {results['n_samples']}",
            ha="center", transform=ax.transAxes, fontsize=12)
    ax.text(0.5, 0.55, f"Median TC frac = {results['median_tc_frac']:.3f}",
            ha="center", transform=ax.transAxes, fontsize=12)
    ax.text(0.5, 0.4, f"APOBEC-high: {results['n_apobec_high_samples']} samples, "
            f"{results['n_apobec_high_muts']} muts",
            ha="center", transform=ax.transAxes, fontsize=10)
    ax.text(0.5, 0.25, f"APOBEC-low: {results['n_apobec_low_samples']} samples, "
            f"{results['n_apobec_low_muts']} muts",
            ha="center", transform=ax.transAxes, fontsize=10)
    ax.set_title("BRCA Sample Stratification")
    ax.axis("off")

    # Panel B: OR_tc across thresholds
    ax = axes[1]
    thresholds = sorted(results["enrichment_high"].keys())
    or_high = [results["enrichment_high"][t].get("OR_tc", 1.0) for t in thresholds]
    or_low = [results["enrichment_low"][t].get("OR_tc", 1.0) for t in thresholds]
    or_all = [results["enrichment_all"][t].get("OR_tc", 1.0) for t in thresholds]

    ax.plot(thresholds, or_high, "o-", color="#EE6677", label="APOBEC-high", linewidth=2)
    ax.plot(thresholds, or_low, "s-", color="#4477AA", label="APOBEC-low", linewidth=2)
    ax.plot(thresholds, or_all, "^--", color="gray", label="All", linewidth=1.5)
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Score Threshold")
    ax.set_ylabel("Odds Ratio (TC-context)")
    ax.set_title("BRCA: APOBEC-high vs APOBEC-low")
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel C: OR comparison at t=0.5 (all vs tc vs nontc)
    ax = axes[2]
    t = 0.5
    categories = ["all", "tc", "nontc"]
    cat_labels = ["All C>T", "TC>TT", "non-TC"]
    x = np.arange(len(categories))
    width = 0.25

    for j, (subset, color, label) in enumerate([
        ("high", "#EE6677", "APOBEC-high"),
        ("low", "#4477AA", "APOBEC-low"),
        ("all", "gray", "All"),
    ]):
        ors = [results[f"enrichment_{subset}"].get(t, {}).get(f"OR_{c}", 1.0) for c in categories]
        ax.bar(x + j * width, ors, width, color=color, label=label, edgecolor="white", alpha=0.85)

    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xticks(x + width)
    ax.set_xticklabels(cat_labels)
    ax.set_ylabel("Odds Ratio")
    ax.set_title(f"BRCA Enrichment at t={t}")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig_path = output_dir / "analysis3_brca_apobec_stratification.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved figure: {fig_path}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    t_start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data once
    data = load_training_data()

    # Analysis 1: Logistic Regression Baseline
    lr_results = run_logistic_regression_baseline(data)
    plot_logistic_regression(lr_results, OUTPUT_DIR)

    # Analysis 2: StructOnly TCGA Ablation
    struct_results = run_struct_only_tcga(data)
    plot_struct_only_tcga(struct_results, OUTPUT_DIR)

    # Analysis 3: BRCA Per-Sample APOBEC Stratification
    brca_results = run_brca_apobec_stratification(data)
    plot_brca_stratification(brca_results, OUTPUT_DIR)

    # Save all results
    all_results = {
        "analysis1_logreg_baseline": {},
        "analysis2_structonly_tcga": {},
        "analysis3_brca_stratification": brca_results,
    }

    # Serialize analysis 1
    for enzyme, models in lr_results.items():
        all_results["analysis1_logreg_baseline"][enzyme] = {}
        for mname, mdata in models.items():
            all_results["analysis1_logreg_baseline"][enzyme][mname] = {
                "auroc_mean": mdata["auroc_mean"],
                "auroc_std": mdata["auroc_std"],
                "auprc_mean": mdata["auprc_mean"],
                "auprc_std": mdata["auprc_std"],
                "fold_aurocs": mdata["fold_aurocs"],
                "fold_auprcs": mdata["fold_auprcs"],
            }

    # Serialize analysis 2
    for cancer, cdata in struct_results.items():
        serializable = {
            "n_mut": cdata["n_mut"],
            "n_ctrl": cdata["n_ctrl"],
            "struct_score_stats": cdata["struct_score_stats"],
            "struct_enrichment": {str(k): v for k, v in cdata["struct_enrichment"].items()},
            "full_enrichment": {str(k): v for k, v in cdata["full_enrichment"].items()},
        }
        all_results["analysis2_structonly_tcga"][cancer] = serializable

    # Serialize analysis 3 (convert threshold keys to strings)
    for key in ["enrichment_high", "enrichment_low", "enrichment_all"]:
        if key in brca_results:
            brca_results[key] = {str(k): v for k, v in brca_results[key].items()}

    results_path = OUTPUT_DIR / "reviewer_baselines_controls.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    elapsed = time.time() - t_start
    logger.info(f"\n{'='*70}")
    logger.info(f"ALL ANALYSES COMPLETE ({elapsed/60:.1f} min)")
    logger.info(f"Results: {results_path}")
    logger.info(f"Figures: {OUTPUT_DIR}")
    logger.info(f"{'='*70}")

    # Print summary table
    logger.info("\n=== SUMMARY ===\n")

    logger.info("Analysis 1: Logistic Regression vs XGBoost (5-fold CV AUROC)")
    logger.info(f"{'Enzyme':<8} {'LogReg(40d)':<18} {'XGB(40d)':<18} {'XGB_Motif(24d)':<18} {'XGB_Struct(16d)':<18}")
    for enzyme in lr_results:
        row = []
        for m in ["LogReg_40dim", "XGB_40dim", "XGB_MotifOnly", "XGB_StructOnly"]:
            d = lr_results[enzyme][m]
            row.append(f"{d['auroc_mean']:.4f}+/-{d['auroc_std']:.4f}")
        logger.info(f"{enzyme:<8} {row[0]:<18} {row[1]:<18} {row[2]:<18} {row[3]:<18}")

    logger.info("\nAnalysis 2: StructOnly vs Full TCGA Enrichment (OR_tc at t=0.5)")
    logger.info(f"{'Cancer':<8} {'StructOnly OR':<16} {'Full OR':<16} {'StructOnly p':<16} {'Full p':<16}")
    for cancer in struct_results:
        se = struct_results[cancer]["struct_enrichment"].get(0.5, {})
        fe = struct_results[cancer]["full_enrichment"].get(0.5, {})
        logger.info(f"{cancer.upper():<8} {se.get('OR_tc', 0):<16.3f} {fe.get('OR_tc', 0):<16.3f} "
                     f"{se.get('p_tc', 1):<16.2e} {fe.get('p_tc', 1):<16.2e}")

    if brca_results:
        logger.info(f"\nAnalysis 3: BRCA APOBEC Stratification (OR_tc at t=0.5)")
        for subset in ["all", "high", "low"]:
            e = brca_results.get(f"enrichment_{subset}", {}).get(0.5, brca_results.get(f"enrichment_{subset}", {}).get("0.5", {}))
            label = {"all": "All tumors", "high": "APOBEC-high", "low": "APOBEC-low"}[subset]
            logger.info(f"  {label:<15}: OR_tc={e.get('OR_tc', 'N/A'):.3f} (p={e.get('p_tc', 1):.2e})")


if __name__ == "__main__":
    main()
