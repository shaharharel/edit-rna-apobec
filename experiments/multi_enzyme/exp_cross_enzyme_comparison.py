#!/usr/bin/env python
"""Cross-enzyme comparison analysis.

Compares APOBEC enzyme editing site preferences across enzymes:
1. Motif radar chart: TC fraction, CC fraction, GC content, trinucleotide diversity
2. Structure comparison: in-loop fraction, mean loop length, dsRNA fraction
3. Pairwise classifier: can we distinguish sites from different enzymes? (AUROC)
4. Feature importance comparison: which features matter for each enzyme?

Requires per-enzyme experiment outputs:
  - experiments/multi_enzyme/outputs/motif_analysis/per_enzyme_motif_results.json
  - experiments/multi_enzyme/outputs/structure_analysis/per_enzyme_structure_results.json
  - experiments/multi_enzyme/outputs/classification/per_enzyme_classification_results.json

Input:  data/processed/multi_enzyme/splits_multi_enzyme_v2.csv
        data/processed/multi_enzyme/multi_enzyme_sequences_v2.json
        data/processed/multi_enzyme/loop_position_per_site_v2.csv
        data/processed/multi_enzyme/structure_cache_multi_enzyme_v2.npz
Output: experiments/multi_enzyme/outputs/cross_enzyme/

Usage:
    conda run -n quris python experiments/multi_enzyme/exp_cross_enzyme_comparison.py
"""

import json
import logging
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SPLITS_CSV = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "splits_multi_enzyme_v2.csv"
SEQ_JSON = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "multi_enzyme_sequences_v2.json"
LOOP_POS_CSV = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "loop_position_per_site_v2.csv"
STRUCT_CACHE = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "structure_cache_multi_enzyme_v2.npz"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "multi_enzyme" / "outputs" / "cross_enzyme"

# Per-enzyme results (from prior experiments)
MOTIF_RESULTS = PROJECT_ROOT / "experiments" / "multi_enzyme" / "outputs" / "motif_analysis" / "per_enzyme_motif_results.json"
STRUCT_RESULTS = PROJECT_ROOT / "experiments" / "multi_enzyme" / "outputs" / "structure_analysis" / "per_enzyme_structure_results.json"
CLASS_RESULTS = PROJECT_ROOT / "experiments" / "multi_enzyme" / "outputs" / "classification" / "per_enzyme_classification_results.json"

# A3A reference for comparison
A3A_RESULTS_DIR = PROJECT_ROOT / "experiments" / "apobec3a" / "outputs"

SEED = 42
MIN_SITES_FOR_PAIRWISE = 50  # Minimum sites per enzyme for pairwise classifier

ENZYME_COLORS = {
    "A3A": "#2563eb",
    "A3B": "#dc2626",
    "A3G": "#16a34a",
    "A1":  "#d97706",
    "A3H": "#7c3aed",
    "A4":  "#6b7280",
}

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
})


# ---------------------------------------------------------------------------
# Feature extraction (same as classification script)
# ---------------------------------------------------------------------------

def extract_motif_features(sequences: Dict[str, str], site_ids: List[str]) -> np.ndarray:
    """24-dim motif features."""
    features = []
    bases = ["A", "C", "G", "U"]
    for sid in site_ids:
        seq = sequences.get(str(sid), "N" * 201).upper().replace("T", "U")
        ep = 100
        up = seq[ep - 1] if ep > 0 else "N"
        down = seq[ep + 1] if ep < len(seq) - 1 else "N"
        feat_5p = [1 if up + "C" == m else 0 for m in ["UC", "CC", "AC", "GC"]]
        feat_3p = [1 if "C" + down == m else 0 for m in ["CA", "CG", "CU", "CC"]]
        trinuc_up = [0] * 8
        for offset, bo in [(-2, 0), (-1, 4)]:
            pos = ep + offset
            if 0 <= pos < len(seq):
                for bi, b in enumerate(bases):
                    if seq[pos] == b:
                        trinuc_up[bo + bi] = 1
        trinuc_down = [0] * 8
        for offset, bo in [(1, 0), (2, 4)]:
            pos = ep + offset
            if 0 <= pos < len(seq):
                for bi, b in enumerate(bases):
                    if seq[pos] == b:
                        trinuc_down[bo + bi] = 1
        features.append(feat_5p + feat_3p + trinuc_up + trinuc_down)
    return np.array(features, dtype=np.float32)


def extract_loop_features(loop_df: pd.DataFrame, site_ids: List[str]) -> np.ndarray:
    """9-dim loop position features."""
    cols = [
        "is_unpaired", "loop_size", "dist_to_junction", "dist_to_apex",
        "relative_loop_position", "left_stem_length", "right_stem_length",
        "max_adjacent_stem_length", "local_unpaired_fraction",
    ]
    features = []
    for sid in site_ids:
        if str(sid) in loop_df.index:
            features.append(loop_df.loc[str(sid), cols].values.astype(np.float32))
        else:
            features.append(np.zeros(len(cols), dtype=np.float32))
    return np.array(features, dtype=np.float32)


def extract_structure_delta(structure_delta: Dict, site_ids: List[str]) -> np.ndarray:
    """7-dim structure delta features."""
    return np.array(
        [structure_delta.get(str(sid), np.zeros(7)) for sid in site_ids],
        dtype=np.float32,
    )


def build_hand_features(site_ids, sequences, structure_delta, loop_df) -> np.ndarray:
    """Build 40-dim hand feature matrix."""
    motif = extract_motif_features(sequences, site_ids)
    struct = extract_structure_delta(structure_delta, site_ids)
    loop = extract_loop_features(loop_df, site_ids)
    hand = np.concatenate([motif, struct, loop], axis=1)
    return np.nan_to_num(hand, nan=0.0)


FEATURE_NAMES = (
    ["motif_UC", "motif_CC", "motif_AC", "motif_GC",
     "motif_CA", "motif_CG", "motif_CU", "motif_CC_3p",
     "trinuc_m2_A", "trinuc_m2_C", "trinuc_m2_G", "trinuc_m2_U",
     "trinuc_m1_A", "trinuc_m1_C", "trinuc_m1_G", "trinuc_m1_U",
     "trinuc_p1_A", "trinuc_p1_C", "trinuc_p1_G", "trinuc_p1_U",
     "trinuc_p2_A", "trinuc_p2_C", "trinuc_p2_G", "trinuc_p2_U"] +
    ["delta_pairing_center", "delta_accessibility_center", "delta_entropy_center",
     "delta_mfe", "mean_delta_pairing_window", "std_delta_pairing_window",
     "mean_delta_accessibility_window"] +
    ["is_unpaired", "loop_size", "dist_to_junction", "dist_to_apex",
     "relative_loop_position", "left_stem_length", "right_stem_length",
     "max_adjacent_stem_length", "local_unpaired_fraction"]
)


# ---------------------------------------------------------------------------
# Analysis 1: Motif radar chart data
# ---------------------------------------------------------------------------

def compute_motif_radar_data(sequences, pos_df, enzymes):
    """Compute radar chart data for each enzyme."""
    radar_data = {}

    for enzyme in enzymes:
        enz_df = pos_df[pos_df["enzyme"] == enzyme]
        seqs = []
        for sid in enz_df["site_id"]:
            seq = sequences.get(str(sid))
            if seq and len(seq) >= 201:
                seqs.append(seq.upper().replace("T", "U"))

        if not seqs:
            continue

        n = len(seqs)
        center = 100

        # TC fraction
        n_tc = sum(1 for s in seqs if center > 0 and s[center - 1] == "U")
        tc_frac = n_tc / n

        # CC fraction
        n_cc = sum(1 for s in seqs if center > 0 and s[center - 1] == "C")
        cc_frac = n_cc / n

        # GC content (in +-10 window)
        gc_fracs = []
        for s in seqs:
            window = s[max(0, center - 10):min(len(s), center + 11)]
            gc = sum(1 for c in window if c in ("G", "C")) / len(window)
            gc_fracs.append(gc)
        gc_content = np.mean(gc_fracs)

        # Trinucleotide diversity (Shannon entropy)
        from collections import Counter
        tri_counter = Counter()
        for s in seqs:
            if center >= 1 and center + 1 < len(s):
                tri = s[center - 1:center + 2]
                tri_counter[tri] += 1
        total = sum(tri_counter.values())
        if total > 0:
            probs = np.array(list(tri_counter.values())) / total
            tri_diversity = -np.sum(probs * np.log2(probs + 1e-10))
        else:
            tri_diversity = 0

        # AC fraction
        n_ac = sum(1 for s in seqs if center > 0 and s[center - 1] == "A")
        ac_frac = n_ac / n

        # Strict CC: C at both -2 and -1 (A3B UCC signature)
        n_strict_cc = sum(
            1 for s in seqs
            if center >= 2 and s[center - 1] == "C" and s[center - 2] == "C"
        )
        strict_cc_frac = n_strict_cc / n

        radar_data[enzyme] = {
            "n_sites": n,
            "tc_fraction": tc_frac,
            "cc_fraction": cc_frac,
            "ac_fraction": ac_frac,
            "strict_cc_m2m1_fraction": strict_cc_frac,
            "gc_content": gc_content,
            "trinuc_diversity": tri_diversity,
        }

    return radar_data


def plot_motif_radar(radar_data, output_dir):
    """Radar chart comparing motif properties across enzymes."""
    enzymes = sorted(radar_data.keys())
    if len(enzymes) < 2:
        logger.warning("Need >= 2 enzymes for radar chart")
        return

    categories = ["TC frac", "CC frac", "AC frac", "GC content", "Tri diversity\n(norm)"]
    n_cats = len(categories)

    # Normalize tri_diversity to 0-1 range
    max_tri_div = max(radar_data[e]["trinuc_diversity"] for e in enzymes) or 1

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="polar"))
    angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    for enzyme in enzymes:
        d = radar_data[enzyme]
        values = [
            d["tc_fraction"],
            d["cc_fraction"],
            d["ac_fraction"],
            d["gc_content"],
            d["trinuc_diversity"] / max_tri_div,
        ]
        values += values[:1]
        color = ENZYME_COLORS.get(enzyme, "#999999")
        ax.plot(angles, values, "o-", linewidth=1.5, label=f"{enzyme} (n={d['n_sites']})",
                color=color, alpha=0.7)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title("Motif Preference Radar Chart by Enzyme", fontsize=13, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "motif_radar_chart.png")
    plt.close(fig)
    logger.info("Saved motif_radar_chart.png")


# ---------------------------------------------------------------------------
# Analysis 2: Structure comparison
# ---------------------------------------------------------------------------

def compute_structure_comparison(pos_df, loop_df, enzymes):
    """Compute structure comparison data per enzyme."""
    struct_data = {}

    for enzyme in enzymes:
        enz_df = pos_df[pos_df["enzyme"] == enzyme]
        enz_ids = set(enz_df["site_id"].astype(str))

        if len(loop_df) > 0:
            enz_loop = loop_df[loop_df.index.isin(enz_ids)]
        else:
            enz_loop = pd.DataFrame()

        n = len(enz_df)
        if n == 0:
            continue

        result = {"n_sites": n}

        if len(enz_loop) > 0 and "is_unpaired" in enz_loop.columns:
            result["inloop_fraction"] = float(enz_loop["is_unpaired"].mean())

            unpaired = enz_loop[enz_loop["is_unpaired"] == True]
            if len(unpaired) > 0 and "loop_size" in unpaired.columns:
                result["mean_loop_size"] = float(unpaired["loop_size"].mean())
                result["median_loop_size"] = float(unpaired["loop_size"].median())
            else:
                result["mean_loop_size"] = None
                result["median_loop_size"] = None

            # dsRNA fraction (sites in stem / paired)
            result["dsrna_fraction"] = 1.0 - result["inloop_fraction"]

            if "local_unpaired_fraction" in enz_loop.columns:
                result["mean_local_unpaired"] = float(enz_loop["local_unpaired_fraction"].mean())
        else:
            result["inloop_fraction"] = None
            result["dsrna_fraction"] = None
            result["mean_loop_size"] = None
            result["mean_local_unpaired"] = None

        struct_data[enzyme] = result

    return struct_data


def plot_structure_comparison(struct_data, output_dir):
    """Grouped bar chart comparing structural properties."""
    enzymes = sorted(struct_data.keys())
    if not enzymes:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: In-loop fraction
    ax = axes[0]
    vals = [struct_data[e].get("inloop_fraction", 0) or 0 for e in enzymes]
    colors = [ENZYME_COLORS.get(e, "#999") for e in enzymes]
    bars = ax.bar(range(len(enzymes)), vals, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xticks(range(len(enzymes)))
    ax.set_xticklabels(enzymes)
    ax.set_ylabel("In-Loop Fraction")
    ax.set_title("Loop Preference (Positive Sites)")
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", fontsize=8)

    # Panel 2: Mean loop size
    ax = axes[1]
    vals = [struct_data[e].get("mean_loop_size", 0) or 0 for e in enzymes]
    bars = ax.bar(range(len(enzymes)), vals, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(enzymes)))
    ax.set_xticklabels(enzymes)
    ax.set_ylabel("Mean Loop Size (nt)")
    ax.set_title("Loop Size (Unpaired Positives)")
    for bar, val in zip(bars, vals):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                    f"{val:.1f}", ha="center", fontsize=8)

    # Panel 3: dsRNA fraction
    ax = axes[2]
    vals = [struct_data[e].get("dsrna_fraction", 0) or 0 for e in enzymes]
    bars = ax.bar(range(len(enzymes)), vals, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xticks(range(len(enzymes)))
    ax.set_xticklabels(enzymes)
    ax.set_ylabel("dsRNA Fraction (in stem)")
    ax.set_title("dsRNA Preference")
    ax.set_ylim(0, 1)

    plt.suptitle("Structural Preferences by Enzyme", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_dir / "structure_comparison.png")
    plt.close(fig)
    logger.info("Saved structure_comparison.png")


# ---------------------------------------------------------------------------
# Analysis 3: Pairwise classifier
# ---------------------------------------------------------------------------

def run_pairwise_classifiers(pos_df, sequences, structure_delta, loop_df, enzymes):
    """Train pairwise classifiers: can we distinguish enzyme A sites from enzyme B?"""
    pairwise_results = {}

    # Filter to enzymes with enough sites
    valid_enzymes = []
    for e in enzymes:
        n = len(pos_df[pos_df["enzyme"] == e])
        if n >= MIN_SITES_FOR_PAIRWISE:
            valid_enzymes.append(e)
        else:
            logger.info("  Skipping %s for pairwise (n=%d < %d)", e, n, MIN_SITES_FOR_PAIRWISE)

    for e1, e2 in combinations(valid_enzymes, 2):
        logger.info("  Pairwise: %s vs %s", e1, e2)

        df1 = pos_df[pos_df["enzyme"] == e1]
        df2 = pos_df[pos_df["enzyme"] == e2]

        site_ids = df1["site_id"].tolist() + df2["site_id"].tolist()
        y = np.array([0] * len(df1) + [1] * len(df2))

        X = build_hand_features(site_ids, sequences, structure_delta, loop_df)

        # 3-fold CV (smaller folds since some enzyme pairs may be small)
        n_splits = min(3, min(len(df1), len(df2)))
        if n_splits < 2:
            logger.warning("  Too few sites for %s vs %s, skipping", e1, e2)
            continue

        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        fold_aurocs = []

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                random_state=SEED + fold_idx,
                eval_metric="logloss",
            )
            model.fit(X_train, y_train, verbose=False)
            y_score = model.predict_proba(X_test)[:, 1]

            if len(np.unique(y_test)) >= 2:
                auroc = float(roc_auc_score(y_test, y_score))
                fold_aurocs.append(auroc)

        if fold_aurocs:
            mean_auroc = float(np.mean(fold_aurocs))
            std_auroc = float(np.std(fold_aurocs))

            # Get feature importance from last model
            importance = model.feature_importances_
            top_feats = sorted(
                zip(FEATURE_NAMES, importance.tolist()),
                key=lambda x: -x[1],
            )[:5]

            pairwise_results[f"{e1}_vs_{e2}"] = {
                "enzyme_1": e1,
                "enzyme_2": e2,
                "n_enzyme_1": len(df1),
                "n_enzyme_2": len(df2),
                "auroc": mean_auroc,
                "auroc_std": std_auroc,
                "n_folds": len(fold_aurocs),
                "distinguishable": mean_auroc > 0.75,
                "top_discriminating_features": [
                    {"feature": name, "importance": imp} for name, imp in top_feats
                ],
            }

            logger.info("    AUROC: %.4f +/- %.4f (%s)",
                        mean_auroc, std_auroc,
                        "DISTINGUISHABLE" if mean_auroc > 0.75 else "similar")

    return pairwise_results


def plot_pairwise_heatmap(pairwise_results, enzymes, output_dir):
    """Heatmap of pairwise AUROC values."""
    valid_enzymes = sorted(set(
        e for pair_data in pairwise_results.values()
        for e in [pair_data["enzyme_1"], pair_data["enzyme_2"]]
    ))

    if len(valid_enzymes) < 2:
        return

    n = len(valid_enzymes)
    matrix = np.full((n, n), 0.5)  # diagonal = 0.5 (random)

    for pair_key, pair_data in pairwise_results.items():
        i = valid_enzymes.index(pair_data["enzyme_1"])
        j = valid_enzymes.index(pair_data["enzyme_2"])
        matrix[i, j] = pair_data["auroc"]
        matrix[j, i] = pair_data["auroc"]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, cmap="RdYlGn_r", vmin=0.4, vmax=1.0)
    ax.set_xticks(range(n))
    ax.set_xticklabels(valid_enzymes, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(valid_enzymes)
    ax.set_title("Pairwise Enzyme Site Distinguishability (AUROC)")
    plt.colorbar(im, ax=ax, shrink=0.8, label="AUROC")

    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            color = "white" if val > 0.75 else "black"
            if i == j:
                ax.text(j, i, "---", ha="center", va="center", fontsize=10, color="gray")
            else:
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=9, color=color)

    fig.tight_layout()
    fig.savefig(output_dir / "pairwise_auroc_heatmap.png")
    plt.close(fig)
    logger.info("Saved pairwise_auroc_heatmap.png")


# ---------------------------------------------------------------------------
# Analysis 4: Feature importance comparison
# ---------------------------------------------------------------------------

def plot_feature_importance_comparison(class_results, output_dir):
    """Side-by-side feature importance comparison across enzymes."""
    if not class_results:
        # Try loading from file
        if CLASS_RESULTS.exists():
            with open(CLASS_RESULTS) as f:
                class_results = json.load(f)

    enzymes = sorted(
        e for e, r in class_results.items()
        if not r.get("skipped", True) and "top_features" in r
    )

    if len(enzymes) < 2:
        logger.warning("Need >= 2 enzymes with classification results for comparison")
        return

    # Collect top features across all enzymes
    all_features = set()
    for e in enzymes:
        for feat in class_results[e]["top_features"][:10]:
            all_features.add(feat["feature"])

    feature_list = sorted(all_features)
    if not feature_list:
        return

    # Build importance matrix
    imp_matrix = np.zeros((len(enzymes), len(feature_list)))
    for i, e in enumerate(enzymes):
        feat_dict = {f["feature"]: f["importance"] for f in class_results[e]["top_features"]}
        for j, feat in enumerate(feature_list):
            imp_matrix[i, j] = feat_dict.get(feat, 0)

    fig, ax = plt.subplots(figsize=(max(12, len(feature_list) * 0.8), max(4, len(enzymes) * 0.8)))
    im = ax.imshow(imp_matrix, aspect="auto", cmap="YlOrRd", vmin=0)
    ax.set_yticks(range(len(enzymes)))
    ax.set_yticklabels(enzymes)
    ax.set_xticks(range(len(feature_list)))
    ax.set_xticklabels(feature_list, rotation=45, ha="right", fontsize=8)
    ax.set_title("Feature Importance Comparison Across Enzymes")
    plt.colorbar(im, ax=ax, shrink=0.7, label="Importance")

    for i in range(len(enzymes)):
        for j in range(len(feature_list)):
            val = imp_matrix[i, j]
            if val > 0.01:
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=7, color="white" if val > 0.1 else "black")

    fig.tight_layout()
    fig.savefig(output_dir / "feature_importance_comparison.png")
    plt.close(fig)
    logger.info("Saved feature_importance_comparison.png")

    return imp_matrix, enzymes, feature_list


# ---------------------------------------------------------------------------
# Analysis 5: Enzyme signature table
# ---------------------------------------------------------------------------

def build_signature_table(radar_data, struct_comparison, enzymes):
    """Build a signature table: TC%, CC%, in-loop fraction, mean loop size per enzyme.

    This provides a concise summary of each enzyme's editing site preferences
    for the key discriminating features identified in the literature.
    """
    rows = []
    for enzyme in enzymes:
        row = {"enzyme": enzyme}

        # Motif fractions from radar data
        if enzyme in radar_data:
            rd = radar_data[enzyme]
            row["n_sites"] = rd["n_sites"]
            row["tc_pct"] = rd["tc_fraction"] * 100
            row["cc_pct"] = rd["cc_fraction"] * 100
            row["strict_cc_pct"] = rd.get("strict_cc_m2m1_fraction", 0) * 100
        else:
            row["n_sites"] = 0
            row["tc_pct"] = None
            row["cc_pct"] = None
            row["strict_cc_pct"] = None

        # Structure from struct_comparison
        if enzyme in struct_comparison:
            sc = struct_comparison[enzyme]
            row["inloop_fraction"] = sc.get("inloop_fraction")
            row["mean_loop_size"] = sc.get("mean_loop_size")
        else:
            row["inloop_fraction"] = None
            row["mean_loop_size"] = None

        # Determine dominant motif
        if row["tc_pct"] is not None and row["cc_pct"] is not None:
            if row["tc_pct"] > row["cc_pct"] + 10:
                row["dominant_motif"] = "TC"
            elif row["cc_pct"] > row["tc_pct"] + 10:
                row["dominant_motif"] = "CC"
            else:
                row["dominant_motif"] = "mixed"
        else:
            row["dominant_motif"] = "N/A"

        # Determine loop preference
        if row["inloop_fraction"] is not None:
            if row["inloop_fraction"] > 0.55:
                row["loop_preference"] = "LOOP"
            elif row["inloop_fraction"] < 0.45:
                row["loop_preference"] = "STEM"
            else:
                row["loop_preference"] = "none"
        else:
            row["loop_preference"] = "N/A"

        rows.append(row)

    return rows


def print_signature_table(signature_rows):
    """Print the signature table in a formatted manner."""
    print("\n--- Enzyme Signature Table ---")
    header = (
        f"  {'Enzyme':<8} {'n':>6}  {'TC%':>6}  {'CC%':>6}  {'CC-2%':>6}  "
        f"{'InLoop':>7}  {'LoopSz':>7}  {'Motif':<6}  {'LoopPref':<8}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for row in sorted(signature_rows, key=lambda r: r["enzyme"]):
        tc = f"{row['tc_pct']:.1f}" if row["tc_pct"] is not None else "N/A"
        cc = f"{row['cc_pct']:.1f}" if row["cc_pct"] is not None else "N/A"
        scc = f"{row['strict_cc_pct']:.1f}" if row["strict_cc_pct"] is not None else "N/A"
        inl = f"{row['inloop_fraction']:.3f}" if row["inloop_fraction"] is not None else "N/A"
        lsz = f"{row['mean_loop_size']:.1f}" if row["mean_loop_size"] is not None else "N/A"

        print(
            f"  {row['enzyme']:<8} {row['n_sites']:>6}  {tc:>6}  {cc:>6}  {scc:>6}  "
            f"{inl:>7}  {lsz:>7}  {row['dominant_motif']:<6}  {row['loop_preference']:<8}"
        )


# ---------------------------------------------------------------------------
# Analysis 6: Pairwise Fisher tests (in-loop fraction vs A3A)
# ---------------------------------------------------------------------------

def compute_pairwise_fisher_vs_a3a(pos_df, loop_df, enzymes):
    """For each enzyme, test whether in-loop fraction differs from A3A.

    Uses Fisher's exact test on a 2x2 contingency table:
        [[A3A_inloop, A3A_stem], [enzymeX_inloop, enzymeX_stem]]

    Returns dict with test results per enzyme.
    """
    if loop_df is None or len(loop_df) == 0 or "is_unpaired" not in loop_df.columns:
        logger.warning("No loop data available for Fisher tests vs A3A")
        return {}

    if "A3A" not in enzymes:
        logger.warning("A3A not in enzyme list; cannot compute Fisher tests vs A3A")
        return {}

    # Get A3A in-loop counts
    a3a_ids = set(pos_df[pos_df["enzyme"] == "A3A"]["site_id"].astype(str))
    a3a_loop = loop_df[loop_df.index.isin(a3a_ids)]
    if len(a3a_loop) < 10:
        logger.warning("Too few A3A sites with loop data (%d)", len(a3a_loop))
        return {}

    a3a_inloop = int(a3a_loop["is_unpaired"].sum())
    a3a_stem = len(a3a_loop) - a3a_inloop
    a3a_frac = a3a_inloop / len(a3a_loop)

    fisher_results = {}
    for enzyme in enzymes:
        if enzyme == "A3A":
            continue

        enz_ids = set(pos_df[pos_df["enzyme"] == enzyme]["site_id"].astype(str))
        enz_loop = loop_df[loop_df.index.isin(enz_ids)]

        if len(enz_loop) < 10:
            logger.info("  Skipping %s (n=%d < 10)", enzyme, len(enz_loop))
            continue

        enz_inloop = int(enz_loop["is_unpaired"].sum())
        enz_stem = len(enz_loop) - enz_inloop
        enz_frac = enz_inloop / len(enz_loop)

        # 2x2 contingency table: [[A3A_inloop, A3A_stem], [enz_inloop, enz_stem]]
        table = [[a3a_inloop, a3a_stem], [enz_inloop, enz_stem]]
        odds_ratio, p_value = fisher_exact(table, alternative="two-sided")

        fisher_results[enzyme] = {
            "enzyme": enzyme,
            "a3a_inloop": a3a_inloop,
            "a3a_stem": a3a_stem,
            "a3a_fraction": a3a_frac,
            "enzyme_inloop": enz_inloop,
            "enzyme_stem": enz_stem,
            "enzyme_fraction": enz_frac,
            "odds_ratio": float(odds_ratio),
            "p_value": float(p_value),
            "significant": p_value < 0.05,
            "direction": "higher" if enz_frac > a3a_frac else "lower",
        }

    return fisher_results


def print_fisher_vs_a3a(fisher_results):
    """Print Fisher test results comparing each enzyme to A3A."""
    if not fisher_results:
        print("\n--- Fisher Tests vs A3A (In-Loop Fraction): No data ---")
        return

    print("\n--- Fisher Tests vs A3A (In-Loop Fraction) ---")
    print(f"  A3A reference: {list(fisher_results.values())[0]['a3a_fraction']:.1%} in-loop "
          f"({list(fisher_results.values())[0]['a3a_inloop']}/{list(fisher_results.values())[0]['a3a_inloop'] + list(fisher_results.values())[0]['a3a_stem']})")
    print()

    for enzyme in sorted(fisher_results.keys()):
        r = fisher_results[enzyme]
        sig = "***" if r["p_value"] < 0.001 else "**" if r["p_value"] < 0.01 else "*" if r["p_value"] < 0.05 else "ns"
        n_total = r["enzyme_inloop"] + r["enzyme_stem"]
        print(f"  {enzyme} vs A3A: {r['enzyme_fraction']:.1%} in-loop ({r['enzyme_inloop']}/{n_total}), "
              f"OR={r['odds_ratio']:.2f}, p={r['p_value']:.2e} {sig} "
              f"({'DIFFERENT' if r['significant'] else 'NOT different'} from A3A)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    if not SPLITS_CSV.exists():
        logger.error("Splits CSV not found: %s", SPLITS_CSV)
        sys.exit(1)
    if not SEQ_JSON.exists():
        logger.error("Sequences JSON not found: %s", SEQ_JSON)
        sys.exit(1)

    df = pd.read_csv(SPLITS_CSV)
    logger.info("Loaded %d sites", len(df))

    with open(SEQ_JSON) as f:
        sequences = json.load(f)
    logger.info("Loaded %d sequences", len(sequences))

    if "enzyme" not in df.columns:
        logger.error("No 'enzyme' column")
        sys.exit(1)

    if "is_edited" in df.columns:
        label_col = "is_edited"
    elif "label" in df.columns:
        label_col = "label"
    else:
        label_col = None
        df = df.copy()
        df["_all_positive"] = 1
        logger.info("No is_edited/label column — treating all sites as positive")
    _lc = label_col if label_col else "_all_positive"
    pos_df = df[df[_lc] == 1]
    enzymes = sorted(pos_df["enzyme"].unique())
    logger.info("Enzymes: %s", enzymes)

    # Load loop position data — HARD FAIL if missing.
    # The pairwise classifier (Analysis 3), structure comparison (Analysis 2),
    # and Fisher tests (Analysis 6) all depend on loop features.
    if not LOOP_POS_CSV.exists():
        logger.error(
            "FATAL: Loop position CSV not found: %s\n"
            "Run loop_position analysis first! Without it, pairwise classifier AUROC "
            "and structure comparisons will be wrong (all loop features = zero).",
            LOOP_POS_CSV,
        )
        sys.exit(1)
    loop_df = pd.read_csv(LOOP_POS_CSV)
    loop_df["site_id"] = loop_df["site_id"].astype(str)
    loop_df = loop_df.set_index("site_id")
    logger.info("Loaded %d loop features", len(loop_df))

    # Load structure cache — HARD FAIL if missing.
    if not STRUCT_CACHE.exists():
        logger.error(
            "FATAL: Structure cache not found: %s\n"
            "Run structure cache generation first! Without it, all 7 structure delta "
            "features in the pairwise classifier will be zero-vectors.",
            STRUCT_CACHE,
        )
        sys.exit(1)
    struct_data = np.load(str(STRUCT_CACHE), allow_pickle=True)
    struct_sids = [str(s) for s in struct_data["site_ids"]]
    delta_features = struct_data["delta_features"]
    structure_delta = dict(zip(struct_sids, delta_features))
    logger.info("Loaded %d structure delta entries", len(structure_delta))

    all_results = {}

    # ===================================================================
    # Analysis 1: Motif radar chart
    # ===================================================================
    logger.info("=" * 60)
    logger.info("ANALYSIS 1: Motif Radar Chart")
    logger.info("=" * 60)

    radar_data = compute_motif_radar_data(sequences, pos_df, enzymes)
    all_results["motif_radar"] = radar_data
    plot_motif_radar(radar_data, OUTPUT_DIR)

    print("\n--- Motif Radar Data ---")
    for e in sorted(radar_data.keys()):
        d = radar_data[e]
        print(f"  {e} (n={d['n_sites']}): TC={d['tc_fraction']:.3f}, "
              f"CC={d['cc_fraction']:.3f}, GC={d['gc_content']:.3f}, "
              f"TriDiv={d['trinuc_diversity']:.2f}")

    # ===================================================================
    # Analysis 2: Structure comparison
    # ===================================================================
    logger.info("=" * 60)
    logger.info("ANALYSIS 2: Structure Comparison")
    logger.info("=" * 60)

    struct_comparison = compute_structure_comparison(pos_df, loop_df, enzymes)
    all_results["structure_comparison"] = struct_comparison
    plot_structure_comparison(struct_comparison, OUTPUT_DIR)

    print("\n--- Structure Comparison ---")
    for e in sorted(struct_comparison.keys()):
        d = struct_comparison[e]
        inloop = d.get("inloop_fraction")
        loop_sz = d.get("mean_loop_size")
        print(f"  {e} (n={d['n_sites']}): "
              f"InLoop={f'{inloop:.3f}' if inloop is not None else 'N/A'}, "
              f"LoopSize={f'{loop_sz:.1f}' if loop_sz is not None else 'N/A'}")

    # ===================================================================
    # Analysis 3: Pairwise classifier
    # ===================================================================
    logger.info("=" * 60)
    logger.info("ANALYSIS 3: Pairwise Classifier (Enzyme A vs B)")
    logger.info("=" * 60)

    pairwise_results = run_pairwise_classifiers(
        pos_df, sequences, structure_delta, loop_df, enzymes
    )
    all_results["pairwise_classifiers"] = pairwise_results
    plot_pairwise_heatmap(pairwise_results, enzymes, OUTPUT_DIR)

    print("\n--- Pairwise Classifier Results ---")
    for pair, data in sorted(pairwise_results.items()):
        status = "DISTINGUISHABLE" if data["distinguishable"] else "similar"
        print(f"  {pair}: AUROC={data['auroc']:.4f} +/- {data['auroc_std']:.4f} ({status})")
        if data["top_discriminating_features"]:
            top = data["top_discriminating_features"][0]
            print(f"    Top discriminating feature: {top['feature']} ({top['importance']:.4f})")

    # ===================================================================
    # Analysis 4: Feature importance comparison
    # ===================================================================
    logger.info("=" * 60)
    logger.info("ANALYSIS 4: Feature Importance Comparison")
    logger.info("=" * 60)

    # Load per-enzyme classification results
    class_results = {}
    if CLASS_RESULTS.exists():
        with open(CLASS_RESULTS) as f:
            class_results = json.load(f)
        plot_feature_importance_comparison(class_results, OUTPUT_DIR)
    else:
        logger.warning("Classification results not found: %s", CLASS_RESULTS)
        logger.warning("Run exp_per_enzyme_classification.py first.")

    # ===================================================================
    # Statistical tests across enzymes
    # ===================================================================
    logger.info("=" * 60)
    logger.info("Statistical Tests: Cross-Enzyme")
    logger.info("=" * 60)

    # Mann-Whitney on in-loop fraction between enzyme pairs
    structural_tests = {}
    if len(loop_df) > 0 and "is_unpaired" in loop_df.columns:
        for e1, e2 in combinations(enzymes, 2):
            ids1 = set(pos_df[pos_df["enzyme"] == e1]["site_id"].astype(str))
            ids2 = set(pos_df[pos_df["enzyme"] == e2]["site_id"].astype(str))

            lf1 = loop_df[loop_df.index.isin(ids1)]["is_unpaired"].values.astype(float)
            lf2 = loop_df[loop_df.index.isin(ids2)]["is_unpaired"].values.astype(float)

            if len(lf1) >= 10 and len(lf2) >= 10:
                u, p = mannwhitneyu(lf1, lf2, alternative="two-sided")
                structural_tests[f"{e1}_vs_{e2}_inloop"] = {
                    "enzyme_1": e1,
                    "enzyme_2": e2,
                    f"{e1}_mean": float(np.mean(lf1)),
                    f"{e2}_mean": float(np.mean(lf2)),
                    "mannwhitney_U": float(u),
                    "p_value": float(p),
                    "significant": p < 0.05,
                }

    all_results["structural_tests"] = structural_tests

    if structural_tests:
        print("\n--- Pairwise In-Loop Fraction Tests (Mann-Whitney) ---")
        for key, data in sorted(structural_tests.items()):
            sig = "***" if data["p_value"] < 0.001 else "**" if data["p_value"] < 0.01 else "*" if data["p_value"] < 0.05 else "ns"
            print(f"  {key}: p={data['p_value']:.2e} {sig}")

    # ===================================================================
    # Analysis 5: Enzyme Signature Table
    # ===================================================================
    logger.info("=" * 60)
    logger.info("ANALYSIS 5: Enzyme Signature Table")
    logger.info("=" * 60)

    signature_rows = build_signature_table(radar_data, struct_comparison, enzymes)
    all_results["signature_table"] = signature_rows
    print_signature_table(signature_rows)

    # ===================================================================
    # Analysis 6: Pairwise Fisher Tests (In-Loop Fraction vs A3A)
    # ===================================================================
    logger.info("=" * 60)
    logger.info("ANALYSIS 6: Pairwise Fisher Tests vs A3A (In-Loop)")
    logger.info("=" * 60)

    fisher_vs_a3a = compute_pairwise_fisher_vs_a3a(pos_df, loop_df, enzymes)
    all_results["fisher_inloop_vs_a3a"] = fisher_vs_a3a
    print_fisher_vs_a3a(fisher_vs_a3a)

    # ===================================================================
    # Save results
    # ===================================================================
    def serialize(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return str(obj)

    with open(OUTPUT_DIR / "cross_enzyme_comparison_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=serialize)
    logger.info("Results saved to %s", OUTPUT_DIR / "cross_enzyme_comparison_results.json")

    # ===================================================================
    # Final summary
    # ===================================================================
    print("\n" + "=" * 80)
    print("CROSS-ENZYME COMPARISON SUMMARY")
    print("=" * 80)

    print("\nMotif preferences:")
    for e in sorted(radar_data.keys()):
        d = radar_data[e]
        dominant = "TC" if d["tc_fraction"] > d["cc_fraction"] else "CC"
        print(f"  {e}: {dominant} dominant ({d['tc_fraction']:.0%} TC, {d['cc_fraction']:.0%} CC)")

    print("\nStructural context:")
    for e in sorted(struct_comparison.keys()):
        d = struct_comparison[e]
        if d.get("inloop_fraction") is not None:
            pref = "LOOP" if d["inloop_fraction"] > 0.5 else "STEM"
            print(f"  {e}: {pref} preference ({d['inloop_fraction']:.0%} in-loop)")

    print("\nPairwise distinguishability:")
    n_distinguishable = sum(1 for d in pairwise_results.values() if d["distinguishable"])
    n_total = len(pairwise_results)
    print(f"  {n_distinguishable}/{n_total} enzyme pairs are distinguishable (AUROC > 0.75)")

    print("\nA3B stem-loop hypothesis (Zhang 2023):")
    if "A3B" in fisher_vs_a3a:
        r = fisher_vs_a3a["A3B"]
        if r["significant"]:
            print(f"  A3B in-loop fraction ({r['enzyme_fraction']:.1%}) is SIGNIFICANTLY "
                  f"different from A3A ({r['a3a_fraction']:.1%}), p={r['p_value']:.2e}")
            if r["enzyme_fraction"] < 0.55:
                print("  -> CONSISTENT with Zhang 2023: A3B lacks stem-loop preference")
            else:
                print("  -> INCONSISTENT with Zhang 2023: A3B shows loop preference")
        else:
            print(f"  A3B in-loop fraction ({r['enzyme_fraction']:.1%}) is NOT significantly "
                  f"different from A3A ({r['a3a_fraction']:.1%}), p={r['p_value']:.2e}")
    else:
        print("  A3B not available or insufficient data for Fisher test")

    print("=" * 80)


if __name__ == "__main__":
    main()
