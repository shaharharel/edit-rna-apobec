#!/usr/bin/env python
"""Levanon-internal confound analysis: do enzyme categories show the same
motif-structure signatures within the 636 Levanon sites only?

This controls for dataset-of-origin confound. If A3A vs A3G differences are
driven by Kockler vs Dang data differences (different sequence lengths, cell
types), the Levanon-internal analysis (all from the same source/methodology)
would NOT show those differences.

Steps:
1. Load all 636 Levanon sites with their enzyme categories
2. For each category, compute: TC%, CC%, in-loop%, relative_loop_position mean
3. Run classification within Levanon-only data (5-fold CV per category)
4. Compare signatures to the full multi-enzyme dataset
5. Statistical tests (Kruskal-Wallis) across categories

Usage:
    conda run -n quris python experiments/multi_enzyme/exp_levanon_confound.py
"""

import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.apobec_feature_extraction import (
    extract_motif_from_seq,
    build_hand_features,
    compute_vienna_features,
    LOOP_FEATURE_COLS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Paths
_ME_DIR = PROJECT_ROOT / "data" / "processed" / "multi_enzyme"
LEVANON_CSV = _ME_DIR / "levanon_all_categories.csv"
SPLITS_CSV = _ME_DIR / "splits_multi_enzyme_v3_with_negatives.csv"
SEQ_JSON_ME = _ME_DIR / "multi_enzyme_sequences_v3_with_negatives.json"
SEQ_JSON_MAIN = PROJECT_ROOT / "data" / "processed" / "site_sequences.json"
LOOP_CSV = _ME_DIR / "loop_position_per_site_v3.csv"
LOOP_CSV_UNIFIED = _ME_DIR / "loop_position_per_site_v3_unified.csv"
STRUCT_CACHE = _ME_DIR / "structure_cache_multi_enzyme_v3.npz"
STRUCT_CACHE_MAIN = PROJECT_ROOT / "data" / "processed" / "embeddings" / "vienna_structure_cache.npz"

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "multi_enzyme" / "outputs" / "levanon_confound"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
CENTER = 100
CATEGORIES = ["A3A", "A3G", "A3A_A3G", "Neither", "Unknown"]


def load_all_data():
    """Load Levanon sites, sequences, loop features, structure deltas."""
    logger.info("Loading data...")

    # Levanon sites
    lev = pd.read_csv(LEVANON_CSV)
    logger.info("  Levanon sites: %d", len(lev))
    logger.info("  Categories: %s", lev["enzyme_category"].value_counts().to_dict())

    # Sequences (merge both JSONs to get all 636 sites)
    with open(SEQ_JSON_ME) as f:
        seqs_me = json.load(f)
    seqs_main = {}
    if SEQ_JSON_MAIN.exists():
        with open(SEQ_JSON_MAIN) as f:
            seqs_main = json.load(f)
    all_seqs = {**seqs_me, **seqs_main}
    n_found = sum(1 for sid in lev["site_id"] if str(sid) in all_seqs)
    logger.info("  Sequences found: %d / %d", n_found, len(lev))

    # Loop features - try unified CSV first (has C2U_ IDs), then fall back to v3
    loop_df = pd.DataFrame()
    for lp in [LOOP_CSV_UNIFIED, LOOP_CSV]:
        if lp.exists():
            tmp = pd.read_csv(lp)
            tmp["site_id"] = tmp["site_id"].astype(str)
            tmp = tmp.set_index("site_id")
            if loop_df.empty:
                loop_df = tmp
            else:
                # Merge, preferring existing entries
                new_ids = tmp.index.difference(loop_df.index)
                if len(new_ids) > 0:
                    loop_df = pd.concat([loop_df, tmp.loc[new_ids]])
            logger.info("  Loop features from %s: %d sites (total: %d)",
                        lp.name, len(tmp), len(loop_df))

    # Check coverage of Levanon sites
    lev_sids = set(lev["site_id"].astype(str).tolist())
    lev_in_loop = sum(1 for sid in lev_sids if sid in loop_df.index)
    logger.info("  Levanon sites with loop data: %d / %d", lev_in_loop, len(lev))

    # Compute loop features on-the-fly for any missing sites
    missing_sids = [sid for sid in lev_sids if sid not in loop_df.index and str(sid) in all_seqs]
    if missing_sids:
        logger.info("  Computing ViennaRNA loop features for %d missing sites...", len(missing_sids))
        new_rows = []
        new_deltas = {}
        for i, sid in enumerate(missing_sids):
            seq = all_seqs[str(sid)].upper().replace("T", "U")
            try:
                sd, lf, _ = compute_vienna_features(seq)
                new_deltas[sid] = sd
                row = {"site_id": sid}
                for j, col in enumerate(LOOP_FEATURE_COLS):
                    row[col] = float(lf[j])
                new_rows.append(row)
            except Exception as e:
                logger.warning("  ViennaRNA failed for %s: %s", sid, e)
                row = {"site_id": sid}
                for col in LOOP_FEATURE_COLS:
                    row[col] = 0.0
                new_rows.append(row)
            if (i + 1) % 20 == 0:
                logger.info("    Computed %d/%d", i + 1, len(missing_sids))

        if new_rows:
            new_df = pd.DataFrame(new_rows).set_index("site_id")
            loop_df = pd.concat([loop_df, new_df])
            struct_delta_extra = new_deltas
            logger.info("  After computing: Levanon sites with loop data: %d / %d",
                        sum(1 for sid in lev_sids if sid in loop_df.index), len(lev))
    else:
        struct_delta_extra = {}

    # Structure delta features
    struct_delta = {}
    for cache_path in [STRUCT_CACHE, STRUCT_CACHE_MAIN]:
        if cache_path.exists():
            data = np.load(str(cache_path), allow_pickle=True)
            if "site_ids" in data and "delta_features" in data:
                sids = list(data["site_ids"])
                deltas = data["delta_features"]
                for i, sid in enumerate(sids):
                    struct_delta[str(sid)] = deltas[i]
                logger.info("  Structure deltas from %s: %d", cache_path.name, len(sids))

    # Merge any newly computed structure deltas
    struct_delta.update(struct_delta_extra)

    return lev, all_seqs, loop_df, struct_delta


def compute_motif_stats(lev, all_seqs):
    """Compute motif statistics for each category."""
    logger.info("Computing motif statistics...")

    results = {}
    for cat in CATEGORIES:
        cat_sites = lev[lev["enzyme_category"] == cat]
        n = len(cat_sites)
        tc_count = 0
        cc_count = 0
        dinuc_counts = Counter()

        for _, row in cat_sites.iterrows():
            sid = str(row["site_id"])
            seq = all_seqs.get(sid)
            if seq is None or len(seq) < CENTER + 2:
                continue
            seq = seq.upper().replace("T", "U")
            up = seq[CENTER - 1] if CENTER > 0 else "N"
            dinuc = up + "C"
            dinuc_counts[dinuc] += 1
            if up == "U":
                tc_count += 1
            elif up == "C":
                cc_count += 1

        results[cat] = {
            "n_sites": n,
            "tc_pct": 100 * tc_count / max(n, 1),
            "cc_pct": 100 * cc_count / max(n, 1),
            "dinuc_counts": dict(dinuc_counts),
        }
        logger.info("  %s (n=%d): TC=%.1f%%, CC=%.1f%%",
                     cat, n, results[cat]["tc_pct"], results[cat]["cc_pct"])

    return results


def compute_structure_stats(lev, loop_df):
    """Compute structure statistics for each category."""
    logger.info("Computing structure statistics...")

    results = {}
    for cat in CATEGORIES:
        cat_sites = lev[lev["enzyme_category"] == cat]
        sids = [str(sid) for sid in cat_sites["site_id"]]

        in_loop_count = 0
        rlp_values = []
        loop_size_values = []
        unpaired_frac_values = []

        for sid in sids:
            if sid in loop_df.index:
                row = loop_df.loc[sid]
                is_unpaired = float(row.get("is_unpaired", 0))
                rlp = float(row.get("relative_loop_position", 0))
                lsize = float(row.get("loop_size", 0))
                upf = float(row.get("local_unpaired_fraction", 0))

                if is_unpaired > 0.5:
                    in_loop_count += 1
                if not np.isnan(rlp):
                    rlp_values.append(rlp)
                if not np.isnan(lsize):
                    loop_size_values.append(lsize)
                if not np.isnan(upf):
                    unpaired_frac_values.append(upf)

        n = len(sids)
        n_with_loop = len(rlp_values)
        results[cat] = {
            "n_sites": n,
            "n_with_loop_data": n_with_loop,
            "in_loop_pct": 100 * in_loop_count / max(n_with_loop, 1),
            "mean_rlp": float(np.mean(rlp_values)) if rlp_values else 0,
            "std_rlp": float(np.std(rlp_values)) if rlp_values else 0,
            "mean_loop_size": float(np.mean(loop_size_values)) if loop_size_values else 0,
            "mean_unpaired_frac": float(np.mean(unpaired_frac_values)) if unpaired_frac_values else 0,
            "rlp_values": rlp_values,  # keep for stat tests
            "loop_size_values": loop_size_values,
            "unpaired_frac_values": unpaired_frac_values,
        }
        logger.info("  %s (n=%d, loop_data=%d): in_loop=%.1f%%, mean_rlp=%.3f",
                     cat, n, n_with_loop, results[cat]["in_loop_pct"],
                     results[cat]["mean_rlp"])

    return results


def run_statistical_tests(motif_stats, struct_stats, lev, all_seqs, loop_df):
    """Run Kruskal-Wallis tests across categories for each feature."""
    logger.info("Running statistical tests...")

    results = {}

    # 1. TC% test: chi-square on TC vs non-TC counts
    tc_table = []
    for cat in CATEGORIES:
        n = motif_stats[cat]["n_sites"]
        tc = motif_stats[cat]["dinuc_counts"].get("UC", 0)
        tc_table.append([tc, n - tc])
    tc_table = np.array(tc_table)
    chi2, p_tc, dof, _ = stats.chi2_contingency(tc_table)
    results["tc_chi2"] = {"chi2": float(chi2), "p": float(p_tc), "dof": int(dof)}
    logger.info("  TC%% chi-square: chi2=%.2f, p=%.2e", chi2, p_tc)

    # 2. CC% test
    cc_table = []
    for cat in CATEGORIES:
        n = motif_stats[cat]["n_sites"]
        cc = motif_stats[cat]["dinuc_counts"].get("CC", 0)
        cc_table.append([cc, n - cc])
    cc_table = np.array(cc_table)
    chi2, p_cc, dof, _ = stats.chi2_contingency(cc_table)
    results["cc_chi2"] = {"chi2": float(chi2), "p": float(p_cc), "dof": int(dof)}
    logger.info("  CC%% chi-square: chi2=%.2f, p=%.2e", chi2, p_cc)

    # 3. Kruskal-Wallis on relative_loop_position
    rlp_groups = [struct_stats[cat]["rlp_values"] for cat in CATEGORIES
                  if len(struct_stats[cat]["rlp_values"]) > 0]
    rlp_labels = [cat for cat in CATEGORIES
                  if len(struct_stats[cat]["rlp_values"]) > 0]
    if len(rlp_groups) >= 2:
        kw_stat, kw_p = stats.kruskal(*rlp_groups)
        results["rlp_kruskal_wallis"] = {
            "H": float(kw_stat), "p": float(kw_p),
            "groups": rlp_labels,
        }
        logger.info("  RLP Kruskal-Wallis: H=%.2f, p=%.2e", kw_stat, kw_p)

    # 4. In-loop% test: chi-square (only categories with loop data)
    loop_table = []
    loop_cats = []
    for cat in CATEGORIES:
        s = struct_stats[cat]
        n = s["n_with_loop_data"]
        if n > 0:
            in_loop = int(round(s["in_loop_pct"] * n / 100))
            loop_table.append([in_loop, n - in_loop])
            loop_cats.append(cat)
    if len(loop_table) >= 2:
        loop_table = np.array(loop_table)
        try:
            chi2, p_loop, dof, _ = stats.chi2_contingency(loop_table)
            results["in_loop_chi2"] = {
                "chi2": float(chi2), "p": float(p_loop), "dof": int(dof),
                "categories": loop_cats,
            }
            logger.info("  In-loop%% chi-square: chi2=%.2f, p=%.2e (cats: %s)",
                        chi2, p_loop, loop_cats)
        except ValueError as e:
            logger.warning("  In-loop chi-square failed: %s", e)

    # 5. Kruskal-Wallis on loop_size
    ls_groups = [struct_stats[cat]["loop_size_values"] for cat in CATEGORIES
                 if len(struct_stats[cat]["loop_size_values"]) > 0]
    ls_labels = [cat for cat in CATEGORIES
                 if len(struct_stats[cat]["loop_size_values"]) > 0]
    if len(ls_groups) >= 2:
        kw_stat, kw_p = stats.kruskal(*ls_groups)
        results["loop_size_kruskal_wallis"] = {
            "H": float(kw_stat), "p": float(kw_p),
            "groups": ls_labels,
        }
        logger.info("  Loop size Kruskal-Wallis: H=%.2f, p=%.2e", kw_stat, kw_p)

    # 6. Pairwise Mann-Whitney for key comparisons
    pairwise = {}
    pairs = [("A3A", "A3G"), ("A3A", "A3A_A3G"), ("A3G", "A3A_A3G"),
             ("A3A", "Neither"), ("A3G", "Neither")]
    for cat_a, cat_b in pairs:
        pw = {}
        # TC%: Fisher exact
        ms_a, ms_b = motif_stats[cat_a], motif_stats[cat_b]
        tc_a = ms_a["dinuc_counts"].get("UC", 0)
        tc_b = ms_b["dinuc_counts"].get("UC", 0)
        n_a, n_b = ms_a["n_sites"], ms_b["n_sites"]
        tab = np.array([[tc_a, n_a - tc_a], [tc_b, n_b - tc_b]])
        odr, fp = stats.fisher_exact(tab)
        pw["tc_fisher"] = {"or": float(odr), "p": float(fp)}

        # CC%: Fisher exact
        cc_a = ms_a["dinuc_counts"].get("CC", 0)
        cc_b = ms_b["dinuc_counts"].get("CC", 0)
        tab = np.array([[cc_a, n_a - cc_a], [cc_b, n_b - cc_b]])
        odr, fp = stats.fisher_exact(tab)
        pw["cc_fisher"] = {"or": float(odr), "p": float(fp)}

        # RLP: Mann-Whitney
        rlp_a = struct_stats[cat_a]["rlp_values"]
        rlp_b = struct_stats[cat_b]["rlp_values"]
        if len(rlp_a) > 5 and len(rlp_b) > 5:
            u_stat, u_p = stats.mannwhitneyu(rlp_a, rlp_b, alternative="two-sided")
            pw["rlp_mannwhitney"] = {"U": float(u_stat), "p": float(u_p)}

        pairwise[f"{cat_a}_vs_{cat_b}"] = pw

    results["pairwise"] = pairwise

    return results


def run_classification_levanon_only(lev, all_seqs, struct_delta, loop_df):
    """Run classification within Levanon-only data.

    For each category with enough sites (>=20), generate negatives by
    sampling from the multi-enzyme negatives matched to levanon_advisor source,
    or generate simple shuffle negatives. Then run 5-fold CV.
    """
    logger.info("=== Classification within Levanon-only data ===")

    # Load the full splits to get negatives
    splits = pd.read_csv(SPLITS_CSV)

    # For each category, we create a binary dataset: positives are the Levanon
    # sites of that category, negatives are randomly sampled cytidines nearby.
    # To keep it simple and fair, we use the multi-enzyme negatives that were
    # generated for the same enzyme type.
    classification_results = {}

    for cat in CATEGORIES:
        cat_sites = lev[lev["enzyme_category"] == cat]
        n_pos = len(cat_sites)
        if n_pos < 20:
            logger.info("  Skipping %s (n=%d < 20)", cat, n_pos)
            classification_results[cat] = {"skipped": True, "n_pos": n_pos}
            continue

        pos_ids = set(cat_sites["site_id"].astype(str).tolist())

        # Get negatives for this enzyme from splits
        # Map category names to enzyme labels in splits
        enzyme_label = cat  # Most match directly
        neg_rows = splits[(splits["enzyme"] == enzyme_label) & (splits["is_edited"] == 0)]

        if len(neg_rows) < n_pos:
            # Fall back: use negatives from "A3A" as generic negatives
            logger.info("  %s: only %d negatives, supplementing with A3A negatives", cat, len(neg_rows))
            extra = splits[(splits["enzyme"] == "A3A") & (splits["is_edited"] == 0)]
            neg_rows = pd.concat([neg_rows, extra]).drop_duplicates(subset="site_id")

        # Sample negatives to match positive count
        neg_sample = neg_rows.sample(n=min(len(neg_rows), n_pos), random_state=SEED)
        neg_ids = neg_sample["site_id"].astype(str).tolist()

        # Build feature matrix
        all_ids = list(pos_ids) + neg_ids
        labels = np.array([1] * len(pos_ids) + [0] * len(neg_ids))

        X = build_hand_features(all_ids, all_seqs, struct_delta, loop_df)
        logger.info("  %s: %d pos + %d neg = %d total, features=%s",
                     cat, len(pos_ids), len(neg_ids), len(all_ids), X.shape)

        # 5-fold CV (or LOO for small categories)
        n_splits = 5 if len(all_ids) >= 50 else 3
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

        fold_aurocs = []
        fold_auprcs = []
        all_y_true = []
        all_y_score = []

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, labels)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]

            model = XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                random_state=SEED, eval_metric="logloss",
                use_label_encoder=False,
            )
            model.fit(X_train, y_train)
            y_score = model.predict_proba(X_test)[:, 1]

            if len(np.unique(y_test)) >= 2:
                fold_aurocs.append(roc_auc_score(y_test, y_score))
                fold_auprcs.append(average_precision_score(y_test, y_score))

            all_y_true.extend(y_test.tolist())
            all_y_score.extend(y_score.tolist())

        # Aggregate
        overall_auroc = roc_auc_score(all_y_true, all_y_score) if len(np.unique(all_y_true)) >= 2 else float("nan")
        overall_auprc = average_precision_score(all_y_true, all_y_score) if len(np.unique(all_y_true)) >= 2 else float("nan")

        # Feature importance from last fold
        fi = model.feature_importances_
        feat_names = (
            [f"motif_{i}" for i in range(24)] +
            [f"struct_delta_{i}" for i in range(7)] +
            LOOP_FEATURE_COLS
        )
        fi_dict = {name: float(fi[i]) for i, name in enumerate(feat_names)}
        fi_sorted = sorted(fi_dict.items(), key=lambda x: -x[1])

        classification_results[cat] = {
            "n_pos": len(pos_ids),
            "n_neg": len(neg_ids),
            "n_folds": n_splits,
            "fold_aurocs": [float(x) for x in fold_aurocs],
            "fold_auprcs": [float(x) for x in fold_auprcs],
            "mean_auroc": float(np.mean(fold_aurocs)) if fold_aurocs else float("nan"),
            "std_auroc": float(np.std(fold_aurocs)) if fold_aurocs else float("nan"),
            "pooled_auroc": float(overall_auroc),
            "pooled_auprc": float(overall_auprc),
            "top_features": fi_sorted[:10],
        }

        logger.info("  %s: AUROC=%.3f (pooled=%.3f), AUPRC=%.3f",
                     cat, classification_results[cat]["mean_auroc"],
                     overall_auroc, overall_auprc)
        logger.info("    Top 3 features: %s",
                     ", ".join(f"{n}={v:.3f}" for n, v in fi_sorted[:3]))

    return classification_results


def compare_with_full_dataset(motif_stats, struct_stats):
    """Compare Levanon-internal signatures with the full multi-enzyme dataset.

    Expected signatures from full dataset:
    - A3A: high TC (86%), moderate in-loop
    - A3G: high CC (65%), extreme 3' tetraloop
    - A3A_A3G: high CC (65%), A3G-like
    - Neither: random motif, intestine-specific
    """
    logger.info("Comparing Levanon-internal with full dataset signatures...")

    # Full dataset reference values (from MEMORY.md and prior experiments)
    full_dataset = {
        "A3A": {"tc_pct": 86.1, "cc_pct": 5.0, "note": "Full A3A dataset (8153 sites)"},
        "A3G": {"tc_pct": 25.0, "cc_pct": 65.0, "note": "Full A3G dataset"},
        "A3A_A3G": {"tc_pct": 30.0, "cc_pct": 65.0, "note": "Both category"},
        "Neither": {"tc_pct": 25.0, "cc_pct": 25.0, "note": "Random motif"},
    }

    comparison = {}
    for cat in CATEGORIES:
        if cat not in motif_stats:
            continue
        lev_tc = motif_stats[cat]["tc_pct"]
        lev_cc = motif_stats[cat]["cc_pct"]
        lev_inloop = struct_stats[cat]["in_loop_pct"]
        lev_rlp = struct_stats[cat]["mean_rlp"]

        entry = {
            "levanon_tc_pct": lev_tc,
            "levanon_cc_pct": lev_cc,
            "levanon_in_loop_pct": lev_inloop,
            "levanon_mean_rlp": lev_rlp,
        }
        if cat in full_dataset:
            entry["full_dataset_tc_pct"] = full_dataset[cat]["tc_pct"]
            entry["full_dataset_cc_pct"] = full_dataset[cat]["cc_pct"]
            entry["tc_consistent"] = (
                (lev_tc > 50 and full_dataset[cat]["tc_pct"] > 50) or
                (lev_tc <= 50 and full_dataset[cat]["tc_pct"] <= 50)
            )
            entry["cc_consistent"] = (
                (lev_cc > 30 and full_dataset[cat]["cc_pct"] > 30) or
                (lev_cc <= 30 and full_dataset[cat]["cc_pct"] <= 30)
            )

        comparison[cat] = entry
        logger.info("  %s: TC=%.1f%% (full: %.1f%%), CC=%.1f%% (full: %.1f%%)",
                     cat, lev_tc, full_dataset.get(cat, {}).get("tc_pct", 0),
                     lev_cc, full_dataset.get(cat, {}).get("cc_pct", 0))

    return comparison


def generate_plots(motif_stats, struct_stats, classification_results):
    """Generate summary plots."""
    logger.info("Generating plots...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: TC% and CC% by category
    ax = axes[0, 0]
    cats = CATEGORIES
    tc_vals = [motif_stats[c]["tc_pct"] for c in cats]
    cc_vals = [motif_stats[c]["cc_pct"] for c in cats]
    x = np.arange(len(cats))
    width = 0.35
    bars1 = ax.bar(x - width/2, tc_vals, width, label="TC%", color="#e41a1c", alpha=0.7)
    bars2 = ax.bar(x + width/2, cc_vals, width, label="CC%", color="#377eb8", alpha=0.7)
    ax.set_xlabel("Enzyme Category")
    ax.set_ylabel("Percentage")
    ax.set_title("Levanon-Internal: Dinucleotide Motif by Category")
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=30, ha="right")
    ax.legend()
    # Add value labels
    for bar, val in zip(bars1, tc_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.0f}%", ha="center", fontsize=8)
    for bar, val in zip(bars2, cc_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.0f}%", ha="center", fontsize=8)

    # Plot 2: In-loop% and mean RLP by category
    ax = axes[0, 1]
    inloop_vals = [struct_stats[c]["in_loop_pct"] for c in cats]
    rlp_vals = [struct_stats[c]["mean_rlp"] * 100 for c in cats]  # scale for visibility
    bars1 = ax.bar(x - width/2, inloop_vals, width, label="In-loop %", color="#4daf4a", alpha=0.7)
    bars2 = ax.bar(x + width/2, rlp_vals, width, label="Mean RLP x100", color="#ff7f00", alpha=0.7)
    ax.set_xlabel("Enzyme Category")
    ax.set_ylabel("Value")
    ax.set_title("Levanon-Internal: Structure Features by Category")
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=30, ha="right")
    ax.legend()

    # Plot 3: Classification AUROC by category
    ax = axes[1, 0]
    auroc_cats = [c for c in cats if c in classification_results
                  and not classification_results[c].get("skipped", False)]
    auroc_vals = [classification_results[c]["pooled_auroc"] for c in auroc_cats]
    n_vals = [classification_results[c]["n_pos"] for c in auroc_cats]
    colors = ["#e41a1c", "#4daf4a", "#ff7f00", "#984ea3", "#999999"]
    bars = ax.bar(range(len(auroc_cats)), auroc_vals,
                  color=colors[:len(auroc_cats)], alpha=0.7)
    ax.set_xlabel("Enzyme Category")
    ax.set_ylabel("Pooled AUROC")
    ax.set_title("Levanon-Internal Classification (5-fold CV)")
    ax.set_xticks(range(len(auroc_cats)))
    ax.set_xticklabels(auroc_cats, rotation=30, ha="right")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Random")
    ax.set_ylim(0.4, 1.0)
    for i, (bar, val, n) in enumerate(zip(bars, auroc_vals, n_vals)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}\n(n={n})", ha="center", fontsize=8)
    ax.legend()

    # Plot 4: Dinucleotide distribution stacked bar
    ax = axes[1, 1]
    dinuc_order = ["UC", "CC", "AC", "GC"]
    dinuc_labels = ["TC", "CC", "AC", "GC"]
    dinuc_colors = ["#e41a1c", "#377eb8", "#4daf4a", "#ff7f00"]
    bottom = np.zeros(len(cats))
    for di, di_label, color in zip(dinuc_order, dinuc_labels, dinuc_colors):
        vals = []
        for c in cats:
            n = motif_stats[c]["n_sites"]
            count = motif_stats[c]["dinuc_counts"].get(di, 0)
            vals.append(100 * count / max(n, 1))
        ax.bar(x, vals, width=0.6, bottom=bottom, label=di_label, color=color, alpha=0.7)
        bottom += np.array(vals)
    ax.set_xlabel("Enzyme Category")
    ax.set_ylabel("Percentage")
    ax.set_title("Levanon-Internal: Full Dinucleotide Distribution")
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=30, ha="right")
    ax.legend(loc="upper right")

    plt.tight_layout()
    plot_path = OUTPUT_DIR / "levanon_confound_analysis.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  Saved plot: %s", plot_path)


def main():
    t_start = time.time()

    # Load data
    lev, all_seqs, loop_df, struct_delta = load_all_data()

    # 1. Motif statistics
    motif_stats = compute_motif_stats(lev, all_seqs)

    # 2. Structure statistics
    struct_stats = compute_structure_stats(lev, loop_df)

    # 3. Statistical tests
    stat_tests = run_statistical_tests(motif_stats, struct_stats, lev, all_seqs, loop_df)

    # 4. Classification within Levanon-only
    classification_results = run_classification_levanon_only(
        lev, all_seqs, struct_delta, loop_df)

    # 5. Compare with full dataset
    comparison = compare_with_full_dataset(motif_stats, struct_stats)

    # 6. Plots
    # Clean struct_stats for serialization (remove raw value lists)
    struct_stats_clean = {}
    for cat in CATEGORIES:
        s = struct_stats[cat].copy()
        s.pop("rlp_values", None)
        s.pop("loop_size_values", None)
        s.pop("unpaired_frac_values", None)
        struct_stats_clean[cat] = s

    generate_plots(motif_stats, struct_stats_clean, classification_results)

    # 7. Save results
    all_results = {
        "motif_stats": motif_stats,
        "structure_stats": struct_stats_clean,
        "statistical_tests": stat_tests,
        "classification": classification_results,
        "comparison_with_full_dataset": comparison,
        "runtime_minutes": (time.time() - t_start) / 60,
    }

    results_path = OUTPUT_DIR / "levanon_confound_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("Saved results: %s", results_path)

    # Summary CSV
    summary_rows = []
    for cat in CATEGORIES:
        row = {
            "category": cat,
            "n_sites": motif_stats[cat]["n_sites"],
            "tc_pct": motif_stats[cat]["tc_pct"],
            "cc_pct": motif_stats[cat]["cc_pct"],
            "in_loop_pct": struct_stats_clean[cat]["in_loop_pct"],
            "mean_rlp": struct_stats_clean[cat]["mean_rlp"],
            "mean_loop_size": struct_stats_clean[cat]["mean_loop_size"],
        }
        if cat in classification_results and not classification_results[cat].get("skipped", False):
            row["auroc"] = classification_results[cat]["pooled_auroc"]
            row["auprc"] = classification_results[cat]["pooled_auprc"]
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUTPUT_DIR / "levanon_confound_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info("Saved summary: %s", summary_path)

    elapsed = time.time() - t_start
    logger.info("=== COMPLETE in %.1f sec ===", elapsed)

    # Print summary
    print("\n" + "=" * 70)
    print("LEVANON-INTERNAL CONFOUND ANALYSIS")
    print("=" * 70)

    print("\n--- Motif Signatures (Levanon 636 sites only) ---")
    print(f"{'Category':<12} {'N':>4} {'TC%':>6} {'CC%':>6} {'In-loop%':>9} {'Mean RLP':>9}")
    print("-" * 50)
    for cat in CATEGORIES:
        m = motif_stats[cat]
        s = struct_stats_clean[cat]
        print(f"{cat:<12} {m['n_sites']:>4} {m['tc_pct']:>5.1f}% {m['cc_pct']:>5.1f}% "
              f"{s['in_loop_pct']:>8.1f}% {s['mean_rlp']:>8.3f}")

    print("\n--- Statistical Tests ---")
    print(f"  TC%% chi-square: p = {stat_tests['tc_chi2']['p']:.2e}")
    print(f"  CC%% chi-square: p = {stat_tests['cc_chi2']['p']:.2e}")
    if "rlp_kruskal_wallis" in stat_tests:
        print(f"  RLP Kruskal-Wallis: H = {stat_tests['rlp_kruskal_wallis']['H']:.2f}, "
              f"p = {stat_tests['rlp_kruskal_wallis']['p']:.2e}")

    print("\n--- Classification (Levanon-internal, 5-fold CV) ---")
    for cat in CATEGORIES:
        cr = classification_results.get(cat, {})
        if cr.get("skipped"):
            print(f"  {cat}: skipped (n={cr.get('n_pos', '?')})")
        else:
            print(f"  {cat} (n={cr['n_pos']}): AUROC={cr['pooled_auroc']:.3f}, "
                  f"AUPRC={cr['pooled_auprc']:.3f}")

    print("\n--- Key Pairwise Comparisons ---")
    for pair_name, pw in stat_tests.get("pairwise", {}).items():
        tc_p = pw.get("tc_fisher", {}).get("p", None)
        cc_p = pw.get("cc_fisher", {}).get("p", None)
        rlp_p = pw.get("rlp_mannwhitney", {}).get("p", None)
        print(f"  {pair_name}: TC p={tc_p:.2e}, CC p={cc_p:.2e}"
              + (f", RLP p={rlp_p:.2e}" if rlp_p else ""))

    print(f"\n--- Conclusion ---")
    # Check if A3A shows TC enrichment even within Levanon
    a3a_tc = motif_stats["A3A"]["tc_pct"]
    a3g_cc = motif_stats["A3G"]["cc_pct"]
    a3a_a3g_pair = stat_tests.get("pairwise", {}).get("A3A_vs_A3G", {})
    tc_p = a3a_a3g_pair.get("tc_fisher", {}).get("p", 1.0)

    if a3a_tc > 50 and a3g_cc > 30 and tc_p < 0.05:
        print("  CONFIRMED: A3A vs A3G motif differences are REAL (not dataset confound)")
        print(f"  Within Levanon only: A3A TC={a3a_tc:.1f}%, A3G CC={a3g_cc:.1f}%, "
              f"Fisher p={tc_p:.2e}")
    else:
        print("  INCONCLUSIVE: Motif differences may be driven by dataset-of-origin")
        print(f"  Within Levanon only: A3A TC={a3a_tc:.1f}%, A3G CC={a3g_cc:.1f}%, "
              f"Fisher p={tc_p:.2e}")

    print(f"\nRuntime: {elapsed:.1f} sec")
    print("=" * 70)


if __name__ == "__main__":
    main()
