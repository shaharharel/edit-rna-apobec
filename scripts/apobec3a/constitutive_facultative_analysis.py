#!/usr/bin/env python
"""Constitutive vs. Facultative RNA editing analysis for APOBEC3A sites.

Classifies the 636 Levanon sites by tissue breadth:
  - Constitutive (edited in >= 30 tissues): intrinsic editability
  - Intermediate (6-29 tissues)
  - Facultative (1-5 tissues): requires tissue-specific factors

Analyses:
  1. Classification and breadth distribution
  2. Embedding space visualization (UMAP/tSNE)
  3. Sequence motif comparison (+/-5nt context)
  4. Structure comparison (ViennaRNA delta features)
  5. Editing rate patterns across tissues
  6. ML prediction of tissue breadth from embeddings
  7. Conservation analysis
  8. Gene / genomic category analysis

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE python scripts/apobec3a/constitutive_facultative_analysis.py
"""

import os
import sys
import json
import logging
import warnings
from pathlib import Path
from collections import Counter

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import torch
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
EMB_DIR = PROJECT_ROOT / "data" / "processed" / "embeddings"
SPLITS_CSV = PROJECT_ROOT / "data" / "processed" / "splits_expanded.csv"
LABELS_CSV = PROJECT_ROOT / "data" / "processed" / "editing_sites_labels.csv"
T1_CSV = PROJECT_ROOT / "data" / "processed" / "advisor" / "t1_gtex_editing_&_conservation.csv"
SEQ_JSON = PROJECT_ROOT / "data" / "processed" / "site_sequences.json"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "apobec" / "outputs" / "constitutive_facultative"

# GTEx tissue list (54 tissues)
GTEX_TISSUES = [
    "Adipose_Subcutaneous", "Adipose_Visceral_Omentum", "Adrenal_Gland",
    "Artery_Aorta", "Artery_Coronary", "Artery_Tibial", "Bladder",
    "Brain_Amygdala", "Brain_Anterior_cingulate_cortex_BA24",
    "Brain_Caudate_basal_ganglia", "Brain_Cerebellar_Hemisphere",
    "Brain_Cerebellum", "Brain_Cortex", "Brain_Frontal_Cortex_BA9",
    "Brain_Hippocampus", "Brain_Hypothalamus",
    "Brain_Nucleus_accumbens_basal_ganglia", "Brain_Putamen_basal_ganglia",
    "Brain_Spinal_cord_cervical_c-1", "Brain_Substantia_nigra",
    "Breast_Mammary_Tissue", "Cells_Cultured_fibroblasts",
    "Cells_EBV-transformed_lymphocytes", "Cervix_Ectocervix",
    "Cervix_Endocervix", "Colon_Sigmoid", "Colon_Transverse",
    "Esophagus_Gastroesophageal_Junction", "Esophagus_Mucosa",
    "Esophagus_Muscularis", "Fallopian_Tube", "Heart_Atrial_Appendage",
    "Heart_Left_Ventricle", "Kidney_Cortex", "Kidney_Medulla", "Liver",
    "Lung", "Minor_Salivary_Gland", "Muscle_Skeletal", "Nerve_Tibial",
    "Ovary", "Pancreas", "Pituitary", "Prostate",
    "Skin_Not_Sun_Exposed_Suprapubic", "Skin_Sun_Exposed_Lower_leg",
    "Small_Intestine_Terminal_Ileum", "Spleen", "Stomach", "Testis",
    "Thyroid", "Uterus", "Vagina", "Whole_Blood",
]

# Breadth category thresholds
CONSTITUTIVE_THRESH = 30
FACULTATIVE_MAX = 5

# Colors for categories
CAT_COLORS = {
    "Constitutive": "#e63946",
    "Intermediate": "#457b9d",
    "Facultative": "#a8dadc",
}


def parse_tissue_rate(val):
    """Parse 'count;total;rate' string -> editing rate (float). Returns NaN if invalid."""
    if pd.isna(val) or val == "" or val is None:
        return np.nan
    parts = str(val).split(";")
    if len(parts) == 3:
        try:
            return float(parts[2])
        except ValueError:
            return np.nan
    return np.nan


def compute_tissue_breadth(gtex_df, threshold_pct=1.0):
    """Compute tissue breadth = number of tissues with editing rate >= threshold_pct."""
    breadths = []
    rate_matrix = []
    for _, row in gtex_df.iterrows():
        rates = []
        for tissue in GTEX_TISSUES:
            rate = parse_tissue_rate(row.get(tissue, np.nan))
            rates.append(rate)
        rates_arr = np.array(rates)
        n_edited = np.nansum(rates_arr >= threshold_pct)
        breadths.append(int(n_edited))
        rate_matrix.append(rates_arr)
    return np.array(breadths), np.array(rate_matrix)


def classify_breadth(breadth):
    """Classify a breadth value into constitutive/intermediate/facultative."""
    if breadth >= CONSTITUTIVE_THRESH:
        return "Constitutive"
    elif breadth <= FACULTATIVE_MAX:
        return "Facultative"
    else:
        return "Intermediate"


# ===========================================================================
# Analysis 1: Classification
# ===========================================================================
def analysis_classification(labels_df, breadths, categories, output_dir):
    """Compute and visualize tissue breadth classification."""
    logger.info("=== Analysis 1: Classification ===")

    results = {}
    for cat in ["Constitutive", "Intermediate", "Facultative"]:
        mask = categories == cat
        results[cat] = {
            "count": int(mask.sum()),
            "fraction": float(mask.sum() / len(categories)),
            "mean_breadth": float(breadths[mask].mean()) if mask.any() else 0,
            "breadth_range": [int(breadths[mask].min()), int(breadths[mask].max())] if mask.any() else [0, 0],
        }
        logger.info(f"  {cat}: n={results[cat]['count']}, "
                     f"mean_breadth={results[cat]['mean_breadth']:.1f}, "
                     f"range={results[cat]['breadth_range']}")

    # Distribution histogram
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram of breadth
    ax = axes[0]
    bins = np.arange(0, 55, 2)
    for cat, color in CAT_COLORS.items():
        mask = categories == cat
        ax.hist(breadths[mask], bins=bins, alpha=0.7, color=color, label=cat, edgecolor="white")
    ax.set_xlabel("Tissue Breadth (# tissues with editing >= 1%)", fontsize=11)
    ax.set_ylabel("Number of Sites", fontsize=11)
    ax.set_title("Distribution of Tissue Breadth", fontsize=13)
    ax.legend(fontsize=10)
    ax.axvline(FACULTATIVE_MAX + 0.5, color="gray", ls="--", alpha=0.5)
    ax.axvline(CONSTITUTIVE_THRESH - 0.5, color="gray", ls="--", alpha=0.5)

    # Pie chart
    ax = axes[1]
    counts = [results[c]["count"] for c in ["Constitutive", "Intermediate", "Facultative"]]
    colors = [CAT_COLORS[c] for c in ["Constitutive", "Intermediate", "Facultative"]]
    labels_pie = [f"{c}\n(n={n})" for c, n in zip(["Constitutive", "Intermediate", "Facultative"], counts)]
    ax.pie(counts, labels=labels_pie, colors=colors, autopct="%1.1f%%",
           startangle=90, textprops={"fontsize": 10})
    ax.set_title("Breadth Category Distribution", fontsize=13)

    plt.tight_layout()
    fig.savefig(output_dir / "01_classification.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved 01_classification.png")

    return results


# ===========================================================================
# Analysis 2: Embedding Space Visualization
# ===========================================================================
def analysis_embedding_visualization(labels_df, breadths, categories, output_dir):
    """UMAP and tSNE of edit effect embeddings colored by breadth."""
    logger.info("=== Analysis 2: Embedding Space Visualization ===")

    # Load embeddings
    wt_emb = torch.load(EMB_DIR / "rnafm_pooled.pt", map_location="cpu", weights_only=False)
    ed_emb = torch.load(EMB_DIR / "rnafm_pooled_edited.pt", map_location="cpu", weights_only=False)

    site_ids = labels_df["site_id"].values
    edit_effects = []
    valid_idx = []
    for i, sid in enumerate(site_ids):
        if sid in wt_emb and sid in ed_emb:
            diff = ed_emb[sid] - wt_emb[sid]
            edit_effects.append(diff.numpy())
            valid_idx.append(i)

    if len(edit_effects) < 10:
        logger.warning("Too few valid embeddings for visualization, skipping.")
        return {}

    X = np.stack(edit_effects)
    valid_breadths = breadths[valid_idx]
    valid_cats = categories[valid_idx]
    logger.info(f"  {len(X)} sites with valid edit effect embeddings")

    # Standardize
    from sklearn.preprocessing import StandardScaler
    X_scaled = StandardScaler().fit_transform(X)

    # tSNE
    from sklearn.manifold import TSNE
    logger.info("  Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X) - 1))
    coords_tsne = tsne.fit_transform(X_scaled)

    # UMAP (if available and safe)
    coords_umap = None
    try:
        import umap
        logger.info("  Running UMAP...")
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(X) - 1))
        coords_umap = reducer.fit_transform(X_scaled)
    except ImportError:
        logger.info("  UMAP not available, using only t-SNE")
    except Exception as e:
        logger.warning(f"  UMAP failed: {e}, using only t-SNE")

    n_methods = 2 if coords_umap is not None else 1

    # Plot: categorical coloring
    fig, axes = plt.subplots(n_methods, 2, figsize=(14, 6 * n_methods))
    if n_methods == 1:
        axes = axes[np.newaxis, :]

    def _scatter_cat(ax, coords, cats, title):
        for cat in ["Facultative", "Intermediate", "Constitutive"]:
            mask = cats == cat
            if mask.any():
                ax.scatter(coords[mask, 0], coords[mask, 1], c=CAT_COLORS[cat],
                           label=cat, s=15, alpha=0.7, edgecolors="none")
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=9, markerscale=2)
        ax.set_xticks([]); ax.set_yticks([])

    def _scatter_cont(ax, coords, values, title):
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=values, cmap="viridis",
                        s=15, alpha=0.7, edgecolors="none")
        plt.colorbar(sc, ax=ax, shrink=0.8, label="Tissue Breadth")
        ax.set_title(title, fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])

    _scatter_cat(axes[0, 0], coords_tsne, valid_cats, "t-SNE: Breadth Category")
    _scatter_cont(axes[0, 1], coords_tsne, valid_breadths, "t-SNE: Continuous Breadth")

    if coords_umap is not None:
        _scatter_cat(axes[1, 0], coords_umap, valid_cats, "UMAP: Breadth Category")
        _scatter_cont(axes[1, 1], coords_umap, valid_breadths, "UMAP: Continuous Breadth")

    plt.tight_layout()
    fig.savefig(output_dir / "02_embedding_visualization.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved 02_embedding_visualization.png")

    return {
        "n_sites_visualized": len(X),
        "methods": ["tSNE"] + (["UMAP"] if coords_umap is not None else []),
    }


# ===========================================================================
# Analysis 3: Sequence Motif Comparison
# ===========================================================================
def analysis_sequence_motifs(labels_df, breadths, categories, output_dir):
    """Compare extended motifs (+-5nt) for constitutive vs facultative sites."""
    logger.info("=== Analysis 3: Sequence Motif Comparison ===")

    # Load sequences
    with open(SEQ_JSON) as f:
        sequences = json.load(f)

    site_ids = labels_df["site_id"].values
    center = 100  # 201nt window, edit site at position 100

    # Collect motifs by category
    motifs_by_cat = {cat: [] for cat in ["Constitutive", "Intermediate", "Facultative"]}
    for i, sid in enumerate(site_ids):
        if sid not in sequences:
            continue
        seq = sequences[sid].upper()
        if len(seq) != 201:
            continue
        # Extract +-5nt context (11nt total)
        start = max(0, center - 5)
        end = min(len(seq), center + 6)
        motif = seq[start:end]
        if len(motif) == 11:
            motifs_by_cat[categories[i]].append(motif)

    results = {}
    # Compute position frequency matrices
    nuc_order = ["A", "C", "G", "U", "T"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    positions = list(range(-5, 6))

    for ax_idx, cat in enumerate(["Constitutive", "Intermediate", "Facultative"]):
        motifs = motifs_by_cat[cat]
        if not motifs:
            axes[ax_idx].set_title(f"{cat} (n=0)")
            continue

        n = len(motifs)
        freq = np.zeros((4, 11))  # A, C/U (edited), G, U/T
        nuc_map = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3}

        for motif in motifs:
            for pos_idx, nt in enumerate(motif):
                if nt in nuc_map:
                    freq[nuc_map[nt], pos_idx] += 1

        freq_norm = freq / n
        results[cat] = {
            "n_sites": n,
            "position_freq": {f"pos_{p}": {nt: float(freq_norm[j, i])
                              for j, nt in enumerate(["A", "C", "G", "U"])}
                              for i, p in enumerate(positions)},
        }

        # Stacked bar plot
        nuc_colors = {"A": "#4CAF50", "C": "#2196F3", "G": "#FF9800", "U": "#F44336"}
        bottom = np.zeros(11)
        for j, nt in enumerate(["A", "C", "G", "U"]):
            axes[ax_idx].bar(positions, freq_norm[j], bottom=bottom, color=nuc_colors[nt],
                             label=nt if ax_idx == 0 else "", width=0.8, edgecolor="white", linewidth=0.5)
            bottom += freq_norm[j]

        axes[ax_idx].set_ylabel("Frequency", fontsize=10)
        axes[ax_idx].set_title(f"{cat} (n={n})", fontsize=12)
        axes[ax_idx].set_ylim(0, 1.05)
        axes[ax_idx].axvline(0, color="black", ls="--", alpha=0.3)

    axes[0].legend(loc="upper right", fontsize=9, ncol=4)
    axes[-1].set_xlabel("Position relative to edit site", fontsize=11)
    axes[-1].set_xticks(positions)

    plt.suptitle("Nucleotide Composition Around Edit Site (+-5nt)", fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(output_dir / "03_sequence_motifs.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved 03_sequence_motifs.png")

    # TC dinucleotide context: check -1 position (should be T/U for APOBEC TC motif)
    logger.info("  TC motif context at -1 position:")
    for cat in ["Constitutive", "Intermediate", "Facultative"]:
        motifs = motifs_by_cat[cat]
        if not motifs:
            continue
        tc_count = sum(1 for m in motifs if m[4] in ("T", "U"))  # position -1
        results[cat]["tc_motif_fraction"] = float(tc_count / len(motifs))
        logger.info(f"    {cat}: T/U at -1 = {tc_count}/{len(motifs)} = {tc_count/len(motifs):.3f}")

    # Upstream (-2) and downstream (+1) context differences
    logger.info("  Context at -2 and +1 positions:")
    for cat in ["Constitutive", "Intermediate", "Facultative"]:
        if cat not in results or results[cat]["n_sites"] == 0:
            continue
        motifs = motifs_by_cat[cat]
        minus2 = Counter(m[3] for m in motifs)
        plus1 = Counter(m[6] for m in motifs)
        results[cat]["pos_minus2"] = {k: v/len(motifs) for k, v in minus2.most_common()}
        results[cat]["pos_plus1"] = {k: v/len(motifs) for k, v in plus1.most_common()}
        logger.info(f"    {cat} -2: {dict(minus2.most_common())}")
        logger.info(f"    {cat} +1: {dict(plus1.most_common())}")

    return results


# ===========================================================================
# Analysis 4: Structure Comparison
# ===========================================================================
def analysis_structure_comparison(labels_df, breadths, categories, output_dir):
    """Compare ViennaRNA delta features between constitutive and facultative sites."""
    logger.info("=== Analysis 4: Structure Comparison ===")

    vienna = np.load(EMB_DIR / "vienna_structure_cache.npz", allow_pickle=True)
    vienna_ids = list(vienna["site_ids"])
    delta_features = vienna["delta_features"]  # (N, 7)
    mfes = vienna["mfes"]
    mfes_edited = vienna["mfes_edited"]
    pairing_probs = vienna["pairing_probs"]  # (N, 201)
    pairing_probs_edited = vienna["pairing_probs_edited"]

    # Feature names for delta_features
    delta_names = ["delta_MFE", "delta_pairing_center", "delta_accessibility_center",
                   "delta_entropy_center", "delta_pairing_mean", "delta_accessibility_mean",
                   "delta_entropy_mean"]

    # Map labels to vienna indices
    site_ids = labels_df["site_id"].values
    vienna_id_set = {v: i for i, v in enumerate(vienna_ids)}

    cat_features = {cat: [] for cat in ["Constitutive", "Intermediate", "Facultative"]}
    cat_delta_mfe = {cat: [] for cat in ["Constitutive", "Intermediate", "Facultative"]}
    cat_delta_pairing_center = {cat: [] for cat in ["Constitutive", "Intermediate", "Facultative"]}

    for i, sid in enumerate(site_ids):
        if sid not in vienna_id_set:
            continue
        vi = vienna_id_set[sid]
        cat = categories[i]
        cat_features[cat].append(delta_features[vi])
        cat_delta_mfe[cat].append(float(mfes_edited[vi] - mfes[vi]))
        # Delta pairing at center (position 100)
        cat_delta_pairing_center[cat].append(
            float(pairing_probs_edited[vi, 100] - pairing_probs[vi, 100]))

    results = {}
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))

    for feat_idx in range(min(7, delta_features.shape[1])):
        ax = axes[feat_idx // 4, feat_idx % 4]
        data_for_box = []
        box_labels = []
        for cat in ["Constitutive", "Intermediate", "Facultative"]:
            feats = cat_features[cat]
            if feats:
                vals = [f[feat_idx] for f in feats]
                data_for_box.append(vals)
                box_labels.append(f"{cat}\n(n={len(vals)})")
                results.setdefault(cat, {})[delta_names[feat_idx]] = {
                    "mean": float(np.mean(vals)),
                    "median": float(np.median(vals)),
                    "std": float(np.std(vals)),
                }

        if data_for_box:
            bp = ax.boxplot(data_for_box, labels=box_labels, patch_artist=True, widths=0.6)
            colors_list = [CAT_COLORS[c] for c in ["Constitutive", "Intermediate", "Facultative"]
                           if cat_features[c]]
            for patch, color in zip(bp["boxes"], colors_list):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        ax.set_title(delta_names[feat_idx], fontsize=10)
        ax.tick_params(axis="x", labelsize=8)

    # Hide the 8th subplot if only 7 features
    if delta_features.shape[1] < 8:
        axes[1, 3].set_visible(False)

    plt.suptitle("ViennaRNA Delta Structure Features by Breadth Category", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_dir / "04_structure_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved 04_structure_comparison.png")

    # Statistical tests: constitutive vs facultative
    stat_tests = {}
    for feat_idx, fname in enumerate(delta_names[:delta_features.shape[1]]):
        c_vals = [f[feat_idx] for f in cat_features["Constitutive"]]
        f_vals = [f[feat_idx] for f in cat_features["Facultative"]]
        if len(c_vals) >= 3 and len(f_vals) >= 3:
            t_stat, p_val = stats.mannwhitneyu(c_vals, f_vals, alternative="two-sided")
            stat_tests[fname] = {"U_statistic": float(t_stat), "p_value": float(p_val)}
            logger.info(f"  {fname}: U={t_stat:.1f}, p={p_val:.4f}")

    results["statistical_tests"] = stat_tests
    return results


# ===========================================================================
# Analysis 5: Editing Rate Patterns
# ===========================================================================
def analysis_editing_rate_patterns(labels_df, breadths, categories, rate_matrix, output_dir):
    """Mean editing rate across tissues for constitutive vs. facultative."""
    logger.info("=== Analysis 5: Editing Rate Patterns ===")

    results = {}

    # Mean rate per category across tissues
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: mean rate profile across tissues
    ax = axes[0]
    for cat, color in CAT_COLORS.items():
        mask = categories == cat
        if not mask.any():
            continue
        cat_rates = rate_matrix[mask]
        mean_per_tissue = np.nanmean(cat_rates, axis=0)
        se_per_tissue = np.nanstd(cat_rates, axis=0) / np.sqrt(np.sum(mask))
        tissue_idx = np.arange(len(GTEX_TISSUES))

        ax.plot(tissue_idx, mean_per_tissue, color=color, label=f"{cat} (n={mask.sum()})",
                linewidth=1.5, alpha=0.8)
        ax.fill_between(tissue_idx, mean_per_tissue - se_per_tissue,
                        mean_per_tissue + se_per_tissue, color=color, alpha=0.15)

        results[cat] = {
            "n_sites": int(mask.sum()),
            "overall_mean_rate": float(np.nanmean(cat_rates)),
            "overall_median_rate": float(np.nanmedian(cat_rates)),
            "mean_across_max_rate": float(np.nanmean(np.nanmax(cat_rates, axis=1))),
        }
        logger.info(f"  {cat}: overall_mean_rate={results[cat]['overall_mean_rate']:.3f}, "
                     f"mean_max_rate={results[cat]['mean_across_max_rate']:.2f}")

    ax.set_xlabel("Tissue Index", fontsize=11)
    ax.set_ylabel("Mean Editing Rate (%)", fontsize=11)
    ax.set_title("Mean Editing Rate Across 54 Tissues", fontsize=13)
    ax.legend(fontsize=10)

    # Right: scatter of breadth vs mean editing rate
    ax = axes[1]
    mean_rates = np.nanmean(rate_matrix, axis=1)
    max_rates = np.nanmax(rate_matrix, axis=1)
    for cat, color in CAT_COLORS.items():
        mask = categories == cat
        ax.scatter(breadths[mask], mean_rates[mask], c=color, s=20, alpha=0.6,
                   label=cat, edgecolors="none")
    # Fit line
    valid = ~np.isnan(mean_rates)
    if valid.sum() > 10:
        r, p = stats.spearmanr(breadths[valid], mean_rates[valid])
        ax.set_title(f"Breadth vs Mean Rate (Spearman r={r:.3f}, p={p:.2e})", fontsize=12)
        results["breadth_vs_mean_rate"] = {"spearman_r": float(r), "spearman_p": float(p)}

    ax.set_xlabel("Tissue Breadth", fontsize=11)
    ax.set_ylabel("Mean Editing Rate (%)", fontsize=11)
    ax.legend(fontsize=10)

    plt.tight_layout()
    fig.savefig(output_dir / "05_editing_rate_patterns.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved 05_editing_rate_patterns.png")

    # Heatmap: top constitutive vs top facultative tissue profiles
    fig, ax = plt.subplots(figsize=(16, 8))
    # Sort by breadth descending, take top and bottom
    order = np.argsort(-breadths)
    n_show = min(30, len(breadths))
    top_idx = order[:n_show]
    bottom_idx = order[-n_show:]
    show_idx = np.concatenate([top_idx, bottom_idx])

    heatmap_data = rate_matrix[show_idx]
    # Replace NaN with 0 for display
    heatmap_data = np.nan_to_num(heatmap_data, nan=0.0)
    # Clip for visualization
    heatmap_data = np.clip(heatmap_data, 0, 20)

    im = ax.imshow(heatmap_data, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Editing Rate (%, clipped at 20)")
    ax.set_xlabel("Tissue", fontsize=11)
    ax.set_ylabel("Sites (top constitutive | bottom facultative)", fontsize=11)
    ax.set_title("Tissue Editing Profiles: Most vs Least Broadly Edited", fontsize=13)
    ax.axhline(n_show - 0.5, color="white", linewidth=2)
    ax.set_yticks([n_show // 2, n_show + n_show // 2])
    ax.set_yticklabels(["Constitutive/\nBroad", "Facultative/\nNarrow"])

    # Shortened tissue labels on x-axis
    short_labels = [t.split("_")[0][:6] for t in GTEX_TISSUES]
    ax.set_xticks(range(len(GTEX_TISSUES)))
    ax.set_xticklabels(short_labels, rotation=90, fontsize=6)

    plt.tight_layout()
    fig.savefig(output_dir / "05b_rate_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved 05b_rate_heatmap.png")

    return results


# ===========================================================================
# Analysis 6: ML Prediction of Tissue Breadth
# ===========================================================================
def analysis_ml_prediction(labels_df, breadths, categories, output_dir):
    """Train MLP on edit effect embeddings to predict tissue breadth."""
    logger.info("=== Analysis 6: ML Prediction of Tissue Breadth ===")

    from sklearn.model_selection import cross_val_predict, KFold
    from sklearn.neural_network import MLPRegressor
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import r2_score, mean_absolute_error
    from sklearn.preprocessing import StandardScaler

    # Load embeddings
    wt_emb = torch.load(EMB_DIR / "rnafm_pooled.pt", map_location="cpu", weights_only=False)
    ed_emb = torch.load(EMB_DIR / "rnafm_pooled_edited.pt", map_location="cpu", weights_only=False)

    site_ids = labels_df["site_id"].values
    X_list, y_list, cat_list = [], [], []
    for i, sid in enumerate(site_ids):
        if sid in wt_emb and sid in ed_emb:
            diff = (ed_emb[sid] - wt_emb[sid]).numpy()
            X_list.append(diff)
            y_list.append(breadths[i])
            cat_list.append(categories[i])

    if len(X_list) < 30:
        logger.warning("Too few samples for ML prediction, skipping.")
        return {}

    X = np.stack(X_list)
    y = np.array(y_list)
    cats = np.array(cat_list)
    logger.info(f"  ML dataset: {X.shape[0]} sites, {X.shape[1]} features")

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Also try with combined: edit effect + original embedding
    X_orig = []
    for i, sid in enumerate(site_ids):
        if sid in wt_emb and sid in ed_emb:
            X_orig.append(wt_emb[sid].numpy())
    X_orig = np.stack(X_orig)
    X_combined = np.hstack([X_scaled, StandardScaler().fit_transform(X_orig)])

    results = {}
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "Ridge (edit effect)": (Ridge(alpha=1.0), X_scaled),
        "MLP (edit effect)": (MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500,
                                           random_state=42, early_stopping=True), X_scaled),
        "GBR (edit effect)": (GradientBoostingRegressor(n_estimators=100, max_depth=4,
                                                         random_state=42), X_scaled),
        "Ridge (combined)": (Ridge(alpha=1.0), X_combined),
        "MLP (combined)": (MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500,
                                        random_state=42, early_stopping=True), X_combined),
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes_flat = axes.flatten()

    for idx, (name, (model, X_input)) in enumerate(models.items()):
        try:
            y_pred = cross_val_predict(model, X_input, y, cv=cv)
            r2 = r2_score(y, y_pred)
            mae = mean_absolute_error(y, y_pred)
            r_spearman, p_spearman = stats.spearmanr(y, y_pred)

            results[name] = {
                "r2": float(r2),
                "mae": float(mae),
                "spearman_r": float(r_spearman),
                "spearman_p": float(p_spearman),
            }
            logger.info(f"  {name}: R2={r2:.3f}, MAE={mae:.2f}, "
                         f"Spearman_r={r_spearman:.3f} (p={p_spearman:.2e})")

            # Scatter plot
            ax = axes_flat[idx]
            for cat, color in CAT_COLORS.items():
                mask = cats == cat
                ax.scatter(y[mask], y_pred[mask], c=color, s=15, alpha=0.6,
                           label=cat, edgecolors="none")
            ax.plot([0, max(y)], [0, max(y)], "k--", alpha=0.3)
            ax.set_xlabel("True Breadth", fontsize=10)
            ax.set_ylabel("Predicted Breadth", fontsize=10)
            ax.set_title(f"{name}\nR2={r2:.3f}, rho={r_spearman:.3f}", fontsize=10)
            ax.legend(fontsize=8, markerscale=2)
        except Exception as e:
            logger.warning(f"  {name} failed: {e}")
            results[name] = {"error": str(e)}

    # Hide unused subplot
    for idx in range(len(models), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.suptitle("ML Prediction of Tissue Breadth from Embeddings", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_dir / "06_ml_prediction.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved 06_ml_prediction.png")

    # Classification: predict category
    from sklearn.metrics import classification_report, accuracy_score
    from sklearn.neural_network import MLPClassifier
    logger.info("  --- Classification: Predicting category ---")
    clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42,
                        early_stopping=True)
    cat_encoded = np.array([{"Constitutive": 2, "Intermediate": 1, "Facultative": 0}[c]
                            for c in cats])
    try:
        y_pred_cls = cross_val_predict(clf, X_scaled, cat_encoded, cv=cv)
        acc = accuracy_score(cat_encoded, y_pred_cls)
        results["classification_accuracy"] = float(acc)
        logger.info(f"  Classification accuracy: {acc:.3f}")
    except Exception as e:
        logger.warning(f"  Classification failed: {e}")

    return results


# ===========================================================================
# Analysis 7: Conservation Analysis
# ===========================================================================
def analysis_conservation(labels_df, breadths, categories, output_dir):
    """Compare conservation of constitutive vs facultative sites."""
    logger.info("=== Analysis 7: Conservation Analysis ===")

    results = {}

    # Conservation data from labels
    has_conservation = "conservation_level" in labels_df.columns and "any_mammalian_conservation" in labels_df.columns
    if not has_conservation:
        logger.info("  No conservation columns found, skipping.")
        return results

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Conservation level by category
    ax = axes[0]
    for cat in ["Constitutive", "Intermediate", "Facultative"]:
        mask = categories == cat
        if not mask.any():
            continue
        cons_vals = labels_df.loc[mask, "conservation_level"].dropna()
        results.setdefault(cat, {})["mean_conservation_level"] = float(cons_vals.mean()) if len(cons_vals) > 0 else None
        results[cat]["n_with_conservation"] = int(len(cons_vals))

    cons_data = []
    cons_labels = []
    for cat in ["Constitutive", "Intermediate", "Facultative"]:
        mask = categories == cat
        vals = labels_df.loc[mask, "conservation_level"].dropna().values
        if len(vals) > 0:
            cons_data.append(vals)
            cons_labels.append(f"{cat}\n(n={len(vals)})")

    if cons_data:
        bp = ax.boxplot(cons_data, labels=cons_labels, patch_artist=True, widths=0.6)
        colors_list = [CAT_COLORS[c] for c in ["Constitutive", "Intermediate", "Facultative"]
                       if len(labels_df.loc[categories == c, "conservation_level"].dropna()) > 0]
        for patch, color in zip(bp["boxes"], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    ax.set_title("Conservation Level by Category", fontsize=12)
    ax.set_ylabel("Conservation Level")

    # Mammalian conservation fraction
    ax = axes[1]
    cat_cons_frac = {}
    for cat in ["Constitutive", "Intermediate", "Facultative"]:
        mask = categories == cat
        if not mask.any():
            continue
        mam = labels_df.loc[mask, "any_mammalian_conservation"]
        frac = mam.sum() / len(mam) if len(mam) > 0 else 0
        cat_cons_frac[cat] = float(frac)
        results.setdefault(cat, {})["mammalian_conservation_fraction"] = float(frac)

    if cat_cons_frac:
        bars = ax.bar(list(cat_cons_frac.keys()),
                      list(cat_cons_frac.values()),
                      color=[CAT_COLORS[c] for c in cat_cons_frac.keys()],
                      alpha=0.7, edgecolor="white")
        ax.set_ylabel("Fraction with Mammalian Conservation")
        ax.set_title("Mammalian Conservation by Category", fontsize=12)
        ax.set_ylim(0, 1)
        for bar, val in zip(bars, cat_cons_frac.values()):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{val:.2f}", ha="center", fontsize=10)

    # Primate editing fraction
    ax = axes[2]
    cat_prim_frac = {}
    for cat in ["Constitutive", "Intermediate", "Facultative"]:
        mask = categories == cat
        if not mask.any():
            continue
        prim = labels_df.loc[mask, "any_primate_editing"]
        frac = prim.sum() / len(prim) if len(prim) > 0 else 0
        cat_prim_frac[cat] = float(frac)
        results.setdefault(cat, {})["primate_editing_fraction"] = float(frac)

    if cat_prim_frac:
        bars = ax.bar(list(cat_prim_frac.keys()),
                      list(cat_prim_frac.values()),
                      color=[CAT_COLORS[c] for c in cat_prim_frac.keys()],
                      alpha=0.7, edgecolor="white")
        ax.set_ylabel("Fraction with Primate Editing")
        ax.set_title("Primate Editing by Category", fontsize=12)
        ax.set_ylim(0, 1)
        for bar, val in zip(bars, cat_prim_frac.values()):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{val:.2f}", ha="center", fontsize=10)

    # Stat tests
    c_mask = categories == "Constitutive"
    f_mask = categories == "Facultative"
    c_cons = labels_df.loc[c_mask, "conservation_level"].dropna()
    f_cons = labels_df.loc[f_mask, "conservation_level"].dropna()
    if len(c_cons) >= 3 and len(f_cons) >= 3:
        u_stat, p_val = stats.mannwhitneyu(c_cons, f_cons, alternative="two-sided")
        results["conservation_test"] = {"U_statistic": float(u_stat), "p_value": float(p_val)}
        logger.info(f"  Conservation: Constitutive vs Facultative U={u_stat:.1f}, p={p_val:.4f}")

    plt.tight_layout()
    fig.savefig(output_dir / "07_conservation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved 07_conservation.png")

    return results


# ===========================================================================
# Analysis 8: Gene / Genomic Category Analysis
# ===========================================================================
def analysis_gene_analysis(labels_df, breadths, categories, output_dir):
    """Analyze whether constitutive sites are in housekeeping genes vs tissue-specific genes."""
    logger.info("=== Analysis 8: Gene / Genomic Category Analysis ===")

    results = {}

    # Genomic category distribution by breadth category
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # 1) Genomic category distribution
    ax = axes[0, 0]
    gen_cats = labels_df["genomic_category"].fillna("Unknown").values
    unique_gcats = sorted(set(gen_cats))

    cat_gcat_counts = {}
    for cat in ["Constitutive", "Intermediate", "Facultative"]:
        mask = categories == cat
        gc = gen_cats[mask]
        counts = Counter(gc)
        total = len(gc)
        cat_gcat_counts[cat] = {g: counts.get(g, 0) / total if total > 0 else 0
                                for g in unique_gcats}

    x_pos = np.arange(len(unique_gcats))
    width = 0.25
    for i, (cat, color) in enumerate(CAT_COLORS.items()):
        vals = [cat_gcat_counts.get(cat, {}).get(g, 0) for g in unique_gcats]
        ax.bar(x_pos + i * width, vals, width, color=color, label=cat, alpha=0.7)
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels(unique_gcats, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Fraction of Sites")
    ax.set_title("Genomic Category Distribution", fontsize=12)
    ax.legend(fontsize=9)

    results["genomic_category_distribution"] = cat_gcat_counts

    # 2) Exonic function distribution
    ax = axes[0, 1]
    exon_funcs = labels_df["exonic_function"].fillna("non-coding/other").values
    unique_efuncs = sorted(set(exon_funcs))

    cat_efunc_counts = {}
    for cat in ["Constitutive", "Intermediate", "Facultative"]:
        mask = categories == cat
        ef = exon_funcs[mask]
        counts = Counter(ef)
        total = len(ef)
        cat_efunc_counts[cat] = {e: counts.get(e, 0) / total if total > 0 else 0
                                  for e in unique_efuncs}

    x_pos = np.arange(len(unique_efuncs))
    for i, (cat, color) in enumerate(CAT_COLORS.items()):
        vals = [cat_efunc_counts.get(cat, {}).get(e, 0) for e in unique_efuncs]
        ax.bar(x_pos + i * width, vals, width, color=color, label=cat, alpha=0.7)
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels(unique_efuncs, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Fraction of Sites")
    ax.set_title("Exonic Function Distribution", fontsize=12)
    ax.legend(fontsize=9)

    results["exonic_function_distribution"] = cat_efunc_counts

    # 3) Structure type distribution
    ax = axes[1, 0]
    if "structure_type" in labels_df.columns:
        struct_types = labels_df["structure_type"].fillna("Unknown").values
        unique_structs = sorted(set(struct_types))

        cat_struct_counts = {}
        for cat in ["Constitutive", "Intermediate", "Facultative"]:
            mask = categories == cat
            st = struct_types[mask]
            counts = Counter(st)
            total = len(st)
            cat_struct_counts[cat] = {s: counts.get(s, 0) / total if total > 0 else 0
                                       for s in unique_structs}

        x_pos = np.arange(len(unique_structs))
        for i, (cat, color) in enumerate(CAT_COLORS.items()):
            vals = [cat_struct_counts.get(cat, {}).get(s, 0) for s in unique_structs]
            ax.bar(x_pos + i * width, vals, width, color=color, label=cat, alpha=0.7)
        ax.set_xticks(x_pos + width)
        ax.set_xticklabels(unique_structs, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Fraction of Sites")
        ax.set_title("RNA Structure Type Distribution", fontsize=12)
        ax.legend(fontsize=9)
        results["structure_type_distribution"] = cat_struct_counts
    else:
        ax.set_visible(False)

    # 4) Genes shared across categories vs unique
    ax = axes[1, 1]
    cat_genes = {}
    for cat in ["Constitutive", "Intermediate", "Facultative"]:
        mask = categories == cat
        genes = set(labels_df.loc[mask, "gene_name"].dropna().values)
        cat_genes[cat] = genes
        results.setdefault("gene_counts", {})[cat] = {
            "n_unique_genes": len(genes),
            "top_genes": Counter(labels_df.loc[mask, "gene_name"].dropna().values).most_common(10),
        }
        logger.info(f"  {cat}: {len(genes)} unique genes")

    # Gene overlap
    const_genes = cat_genes.get("Constitutive", set())
    fac_genes = cat_genes.get("Facultative", set())
    inter_genes = cat_genes.get("Intermediate", set())
    overlap_cf = const_genes & fac_genes
    results["gene_overlap"] = {
        "constitutive_only": len(const_genes - fac_genes - inter_genes),
        "facultative_only": len(fac_genes - const_genes - inter_genes),
        "shared_const_fac": len(overlap_cf),
    }
    logger.info(f"  Gene overlap (Const & Fac): {len(overlap_cf)}")

    # Bar chart of top genes by breadth
    top_genes_const = Counter(labels_df.loc[categories == "Constitutive", "gene_name"].dropna().values)
    top_genes_fac = Counter(labels_df.loc[categories == "Facultative", "gene_name"].dropna().values)

    all_genes_sorted = sorted(set(list(top_genes_const.keys()) + list(top_genes_fac.keys())),
                              key=lambda g: top_genes_const.get(g, 0) + top_genes_fac.get(g, 0),
                              reverse=True)[:15]

    x_pos = np.arange(len(all_genes_sorted))
    c_vals = [top_genes_const.get(g, 0) for g in all_genes_sorted]
    f_vals = [top_genes_fac.get(g, 0) for g in all_genes_sorted]
    ax.bar(x_pos - 0.2, c_vals, 0.35, color=CAT_COLORS["Constitutive"], label="Constitutive", alpha=0.7)
    ax.bar(x_pos + 0.2, f_vals, 0.35, color=CAT_COLORS["Facultative"], label="Facultative", alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(all_genes_sorted, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Number of Sites")
    ax.set_title("Top Genes by Category", fontsize=12)
    ax.legend(fontsize=9)

    # APOBEC class
    if "apobec_class" in labels_df.columns:
        logger.info("  APOBEC class distribution:")
        for cat in ["Constitutive", "Intermediate", "Facultative"]:
            mask = categories == cat
            ac = Counter(labels_df.loc[mask, "apobec_class"].fillna("Unknown").values)
            results.setdefault("apobec_class", {})[cat] = dict(ac.most_common())
            logger.info(f"    {cat}: {dict(ac.most_common())}")

    plt.tight_layout()
    fig.savefig(output_dir / "08_gene_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved 08_gene_analysis.png")

    return results


# ===========================================================================
# Summary Figure
# ===========================================================================
def create_summary_figure(all_results, labels_df, breadths, categories, rate_matrix, output_dir):
    """Create a multi-panel summary figure."""
    logger.info("=== Creating Summary Figure ===")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # Panel 1: Breadth distribution
    ax = axes[0, 0]
    bins = np.arange(0, 55, 2)
    for cat, color in CAT_COLORS.items():
        mask = categories == cat
        ax.hist(breadths[mask], bins=bins, alpha=0.7, color=color, label=cat, edgecolor="white")
    ax.set_xlabel("Tissue Breadth")
    ax.set_ylabel("Count")
    ax.set_title("A. Tissue Breadth Distribution")
    ax.legend(fontsize=8)
    ax.axvline(FACULTATIVE_MAX + 0.5, color="gray", ls="--", alpha=0.5)
    ax.axvline(CONSTITUTIVE_THRESH - 0.5, color="gray", ls="--", alpha=0.5)

    # Panel 2: Breadth vs mean rate
    ax = axes[0, 1]
    mean_rates = np.nanmean(rate_matrix, axis=1)
    for cat, color in CAT_COLORS.items():
        mask = categories == cat
        ax.scatter(breadths[mask], mean_rates[mask], c=color, s=15, alpha=0.6,
                   label=cat, edgecolors="none")
    valid = ~np.isnan(mean_rates)
    r, p = stats.spearmanr(breadths[valid], mean_rates[valid])
    ax.set_xlabel("Tissue Breadth")
    ax.set_ylabel("Mean Editing Rate (%)")
    ax.set_title(f"B. Breadth vs Rate (rho={r:.3f})")
    ax.legend(fontsize=8)

    # Panel 3: Conservation by category
    ax = axes[0, 2]
    cons_data, cons_labels_list = [], []
    for cat in ["Constitutive", "Intermediate", "Facultative"]:
        mask = categories == cat
        vals = labels_df.loc[mask, "conservation_level"].dropna().values
        if len(vals) > 0:
            cons_data.append(vals)
            cons_labels_list.append(cat)
    if cons_data:
        bp = ax.boxplot(cons_data, labels=cons_labels_list, patch_artist=True, widths=0.6)
        for patch, cat in zip(bp["boxes"], cons_labels_list):
            patch.set_facecolor(CAT_COLORS[cat])
            patch.set_alpha(0.7)
    ax.set_title("C. Conservation Level")
    ax.set_ylabel("Conservation Level")

    # Panel 4: Mean rate profile across tissues
    ax = axes[1, 0]
    for cat, color in CAT_COLORS.items():
        mask = categories == cat
        if not mask.any():
            continue
        cat_rates = rate_matrix[mask]
        mean_per_tissue = np.nanmean(cat_rates, axis=0)
        ax.plot(range(len(GTEX_TISSUES)), mean_per_tissue, color=color,
                label=f"{cat} (n={mask.sum()})", linewidth=1.5, alpha=0.8)
    ax.set_xlabel("Tissue Index")
    ax.set_ylabel("Mean Rate (%)")
    ax.set_title("D. Tissue Rate Profiles")
    ax.legend(fontsize=8)

    # Panel 5: ML prediction best result
    ax = axes[1, 1]
    ml_res = all_results.get("ml_prediction", {})
    model_names = [k for k in ml_res.keys() if isinstance(ml_res[k], dict) and "r2" in ml_res[k]]
    if model_names:
        r2_vals = [ml_res[k]["r2"] for k in model_names]
        rho_vals = [ml_res[k]["spearman_r"] for k in model_names]
        x_pos = np.arange(len(model_names))
        bars1 = ax.bar(x_pos - 0.15, r2_vals, 0.3, label="R2", color="#457b9d", alpha=0.7)
        bars2 = ax.bar(x_pos + 0.15, rho_vals, 0.3, label="Spearman rho", color="#e63946", alpha=0.7)
        ax.set_xticks(x_pos)
        short_names = [n.replace(" (edit effect)", "\n(EE)").replace(" (combined)", "\n(Comb)") for n in model_names]
        ax.set_xticklabels(short_names, fontsize=7, rotation=30, ha="right")
        ax.legend(fontsize=8)
        ax.set_ylabel("Score")
        ax.axhline(0, color="gray", ls="-", alpha=0.3)
    ax.set_title("E. ML Prediction Performance")

    # Panel 6: Genomic category
    ax = axes[1, 2]
    if "genomic_category" in labels_df.columns:
        gen_cats = labels_df["genomic_category"].fillna("Unknown").values
        for cat, color in CAT_COLORS.items():
            mask = categories == cat
            gc = gen_cats[mask]
            counts = Counter(gc)
            total = len(gc)
            fracs = {g: counts.get(g, 0) / total for g in sorted(set(gen_cats))}
            logger.info(f"  {cat} genomic: {fracs}")

        unique_gcats = sorted(set(gen_cats))
        x_pos = np.arange(len(unique_gcats))
        width = 0.25
        for i, (cat, color) in enumerate(CAT_COLORS.items()):
            mask = categories == cat
            gc = gen_cats[mask]
            counts = Counter(gc)
            total = len(gc)
            vals = [counts.get(g, 0) / total if total > 0 else 0 for g in unique_gcats]
            ax.bar(x_pos + i * width, vals, width, color=color, label=cat, alpha=0.7)
        ax.set_xticks(x_pos + width)
        ax.set_xticklabels(unique_gcats, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Fraction")
        ax.legend(fontsize=8)
    ax.set_title("F. Genomic Category")

    plt.suptitle("Constitutive vs. Facultative APOBEC3A Editing Analysis", fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_dir / "00_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved 00_summary.png")


# ===========================================================================
# Main
# ===========================================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {OUTPUT_DIR}")

    # Load data
    logger.info("Loading data...")
    labels_df = pd.read_csv(LABELS_CSV)
    gtex_df = pd.read_csv(T1_CSV)
    splits_df = pd.read_csv(SPLITS_CSV)

    logger.info(f"  Labels: {len(labels_df)} sites")
    logger.info(f"  GTEx: {len(gtex_df)} sites with 54-tissue profiles")
    logger.info(f"  Splits: {len(splits_df)} total entries")

    # The labels file already has the 636 Levanon sites with n_tissues_edited
    # Use the pre-computed n_tissues_edited from labels as the primary breadth measure
    # But also compute our own from the GTEx rate data for cross-validation
    breadths_precomputed = labels_df["n_tissues_edited"].values.astype(int)

    # Also compute from GTEx data for the rate matrix
    breadths_computed, rate_matrix = compute_tissue_breadth(gtex_df, threshold_pct=1.0)

    # Use precomputed breadths (from labels) as primary
    breadths = breadths_precomputed
    logger.info(f"  Breadth stats: mean={breadths.mean():.1f}, median={np.median(breadths):.0f}, "
                f"max={breadths.max()}, min={breadths.min()}")

    # Cross-check: correlation between precomputed and our computation
    # The GTEx file and labels file are aligned by row order (both 636 rows)
    r_check, _ = stats.spearmanr(breadths_precomputed, breadths_computed)
    logger.info(f"  Breadth cross-check (precomputed vs computed): Spearman r={r_check:.4f}")

    # Classify
    categories = np.array([classify_breadth(b) for b in breadths])
    for cat in ["Constitutive", "Intermediate", "Facultative"]:
        logger.info(f"  {cat}: {(categories == cat).sum()} sites")

    # Run all analyses
    all_results = {}

    all_results["classification"] = analysis_classification(
        labels_df, breadths, categories, OUTPUT_DIR)

    all_results["embedding_visualization"] = analysis_embedding_visualization(
        labels_df, breadths, categories, OUTPUT_DIR)

    all_results["sequence_motifs"] = analysis_sequence_motifs(
        labels_df, breadths, categories, OUTPUT_DIR)

    all_results["structure_comparison"] = analysis_structure_comparison(
        labels_df, breadths, categories, OUTPUT_DIR)

    all_results["editing_rate_patterns"] = analysis_editing_rate_patterns(
        labels_df, breadths, categories, rate_matrix, OUTPUT_DIR)

    all_results["ml_prediction"] = analysis_ml_prediction(
        labels_df, breadths, categories, OUTPUT_DIR)

    all_results["conservation"] = analysis_conservation(
        labels_df, breadths, categories, OUTPUT_DIR)

    all_results["gene_analysis"] = analysis_gene_analysis(
        labels_df, breadths, categories, OUTPUT_DIR)

    # Summary figure
    create_summary_figure(all_results, labels_df, breadths, categories, rate_matrix, OUTPUT_DIR)

    # Add metadata
    all_results["metadata"] = {
        "n_total_sites": int(len(labels_df)),
        "n_constitutive": int((categories == "Constitutive").sum()),
        "n_intermediate": int((categories == "Intermediate").sum()),
        "n_facultative": int((categories == "Facultative").sum()),
        "constitutive_threshold": CONSTITUTIVE_THRESH,
        "facultative_max": FACULTATIVE_MAX,
        "breadth_threshold_pct": 1.0,
        "breadth_cross_check_spearman": float(r_check),
    }

    # Make results JSON-serializable
    def make_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, set):
            return list(obj)
        return obj

    all_results = make_serializable(all_results)

    # Save results JSON
    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"Saved results to {results_path}")

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS COMPLETE - KEY FINDINGS")
    logger.info("=" * 70)
    logger.info(f"Total Levanon sites: {len(labels_df)}")
    logger.info(f"  Constitutive (>={CONSTITUTIVE_THRESH} tissues): {(categories == 'Constitutive').sum()}")
    logger.info(f"  Intermediate (6-29 tissues): {(categories == 'Intermediate').sum()}")
    logger.info(f"  Facultative (1-5 tissues): {(categories == 'Facultative').sum()}")

    ml_res = all_results.get("ml_prediction", {})
    best_r2 = max((v.get("r2", -999) for v in ml_res.values() if isinstance(v, dict) and "r2" in v), default=None)
    best_rho = max((v.get("spearman_r", -999) for v in ml_res.values() if isinstance(v, dict) and "spearman_r" in v), default=None)
    if best_r2 is not None:
        logger.info(f"  Best ML R2: {best_r2:.3f}")
    if best_rho is not None:
        logger.info(f"  Best ML Spearman rho: {best_rho:.3f}")

    rate_res = all_results.get("editing_rate_patterns", {})
    if "breadth_vs_mean_rate" in rate_res:
        logger.info(f"  Breadth vs Rate Spearman: r={rate_res['breadth_vs_mean_rate']['spearman_r']:.3f}")

    logger.info(f"\nOutputs saved to: {OUTPUT_DIR}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
