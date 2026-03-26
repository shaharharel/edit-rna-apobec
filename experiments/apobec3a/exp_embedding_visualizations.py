#!/usr/bin/env python
"""Edit embedding space visualization with PCA, t-SNE, and UMAP.

Generates 2D scatter plots of the edit effect embedding space,
colored by various biological and technical annotations:
  - Dataset source
  - Binary label (positive/negative)
  - Editing rate (continuous gradient)
  - TC-motif status
  - Delta MFE (structure change)
  - Positive sites by source dataset
  - Genomic feature category
  - Density contours

Produces individual per-method grids and a combined mega-grid.

Usage:
    python experiments/apobec3a/exp_embedding_visualizations.py
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gc
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import torch
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logger = logging.getLogger(__name__)

EMB_DIR = PROJECT_ROOT / "data" / "processed" / "embeddings"
SPLITS_CSV = PROJECT_ROOT / "data" / "processed" / "splits_expanded.csv"
STRUCT_CACHE = EMB_DIR / "vienna_structure_cache.npz"
SEQ_JSON = PROJECT_ROOT / "data" / "processed" / "site_sequences.json"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "apobec3a" / "outputs" / "embedding_viz"

DATASET_LABELS = {
    "advisor_c2t": "Levanon",
    "asaoka_2019": "Asaoka",
    "sharma_2015": "Sharma",
    "alqassim_2021": "Alqassim",
    "baysal_2016": "Baysal",
    "tier2_negative": "Tier2 Neg",
    "tier3_negative": "Tier3 Neg",
}

DATASET_COLORS = {
    "Levanon": "#2563eb",
    "Asaoka": "#16a34a",
    "Sharma": "#dc2626",
    "Alqassim": "#d97706",
    "Baysal": "#7c3aed",
    "Tier2 Neg": "#6b7280",
    "Tier3 Neg": "#374151",
}

FEATURE_COLORS = {
    "synonymous": "#2563eb",
    "nonsynonymous": "#dc2626",
    "stopgain": "#f97316",
    "intronic": "#16a34a",
    "ncRNA_exonic": "#7c3aed",
    "UTR3": "#0891b2",
    "UTR5": "#be185d",
    "intergenic": "#6b7280",
}

MAX_SAMPLES = 3000  # Subsample for t-SNE/UMAP

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
})


def load_data():
    """Load embeddings and metadata."""
    pooled_orig = torch.load(EMB_DIR / "rnafm_pooled.pt", weights_only=False)
    pooled_edited = torch.load(EMB_DIR / "rnafm_pooled_edited.pt", weights_only=False)
    logger.info("Loaded %d pooled embeddings", len(pooled_orig))

    splits_df = pd.read_csv(SPLITS_CSV)
    available = set(pooled_orig.keys()) & set(pooled_edited.keys())
    splits_df = splits_df[splits_df["site_id"].isin(available)].copy()
    logger.info("Sites with embeddings: %d", len(splits_df))

    # Compute edit diff embeddings
    site_ids = splits_df["site_id"].tolist()
    diff_embs = []
    valid_ids = []
    for sid in site_ids:
        if sid in pooled_orig and sid in pooled_edited:
            diff = (pooled_edited[sid] - pooled_orig[sid]).numpy()
            diff_embs.append(diff)
            valid_ids.append(sid)

    diff_matrix = np.array(diff_embs)
    logger.info("Diff embedding matrix: %s", diff_matrix.shape)

    # Free large embedding dicts
    del pooled_orig, pooled_edited
    gc.collect()

    # Build metadata
    meta_df = splits_df.set_index("site_id").loc[valid_ids].reset_index()
    meta_df["dataset_label"] = meta_df["dataset_source"].map(DATASET_LABELS).fillna(meta_df["dataset_source"])
    meta_df["editing_rate"] = pd.to_numeric(meta_df["editing_rate"], errors="coerce")

    # TC-motif
    sequences = {}
    if SEQ_JSON.exists():
        sequences = json.loads(SEQ_JSON.read_text())

    tc_motif = []
    for sid in valid_ids:
        if sid in sequences:
            seq = sequences[sid]
            center = len(seq) // 2
            is_tc = center >= 1 and seq[center - 1] in "Uu"
            tc_motif.append(is_tc)
        else:
            tc_motif.append(False)
    meta_df["tc_motif"] = tc_motif

    # Genomic feature category
    if "feature" in meta_df.columns:
        meta_df["feature_label"] = meta_df["feature"].fillna("Unknown")
    else:
        meta_df["feature_label"] = "Unknown"

    # Structure delta
    structure_delta = {}
    if STRUCT_CACHE.exists():
        data = np.load(STRUCT_CACHE, allow_pickle=True)
        feat_key = "delta_features" if "delta_features" in data else "features"
        if "site_ids" in data and feat_key in data:
            sids = data["site_ids"]
            feats = data[feat_key]
            structure_delta = {str(sid): feats[i] for i, sid in enumerate(sids)}

    delta_mfe = []
    for sid in valid_ids:
        if sid in structure_delta:
            delta_mfe.append(float(structure_delta[sid][3]))
        else:
            delta_mfe.append(np.nan)
    meta_df["delta_mfe"] = delta_mfe

    return diff_matrix, meta_df, valid_ids


def run_dimensionality_reduction(diff_matrix, n_subsample=MAX_SAMPLES):
    """Run PCA, t-SNE, and UMAP on edit diff embeddings."""
    results = {}

    # PCA on full data
    logger.info("Running PCA...")
    pca = PCA(n_components=50)
    pca_50 = pca.fit_transform(diff_matrix)
    results["pca_2d"] = pca_50[:, :2]
    results["pca_variance"] = pca.explained_variance_ratio_
    logger.info("PCA variance explained (top 10): %s",
                [f"{v:.3f}" for v in pca.explained_variance_ratio_[:10]])

    # Subsample for t-SNE and UMAP
    n = len(diff_matrix)
    if n > n_subsample:
        idx = np.random.RandomState(42).choice(n, n_subsample, replace=False)
        idx = np.sort(idx)
    else:
        idx = np.arange(n)
    results["subsample_idx"] = idx

    pca_sub = pca_50[idx]

    # t-SNE
    logger.info("Running t-SNE on %d samples...", len(idx))
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42, n_jobs=1)
    results["tsne_2d"] = tsne.fit_transform(pca_sub[:, :30])

    # UMAP
    try:
        import umap
        logger.info("Running UMAP on %d samples...", len(idx))
        reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.3, random_state=42)
        results["umap_2d"] = reducer.fit_transform(pca_sub[:, :30])
    except ImportError:
        logger.warning("UMAP not available (pip install umap-learn). Skipping.")
        results["umap_2d"] = None

    return results


def add_density_contours(ax, x, y, n_levels=5):
    """Add KDE density contours to an axis."""
    try:
        # Subsample if too many points for KDE performance
        n = len(x)
        if n > 2000:
            rng = np.random.RandomState(42)
            idx = rng.choice(n, 2000, replace=False)
            x_sub, y_sub = x[idx], y[idx]
        else:
            x_sub, y_sub = x, y

        xy = np.vstack([x_sub, y_sub])
        kde = gaussian_kde(xy, bw_method=0.3)

        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
        pad_x = (xmax - xmin) * 0.05
        pad_y = (ymax - ymin) * 0.05
        xx, yy = np.mgrid[xmin - pad_x:xmax + pad_x:100j,
                           ymin - pad_y:ymax + pad_y:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        z = kde(positions).reshape(xx.shape)

        ax.contour(xx, yy, z, levels=n_levels, colors="black",
                   linewidths=0.5, alpha=0.3)
    except Exception:
        pass  # Silently skip if KDE fails


def make_scatter(ax, x, y, colors, title, cmap=None, labels=None, alpha=0.4, s=8,
                 color_map=None, density=False):
    """Make a scatter plot on given axis.

    Parameters
    ----------
    color_map : dict or None
        If provided and labels are given, map label -> color.
    density : bool
        If True, add KDE density contours.
    """
    if labels is not None:
        # Categorical
        unique_labels = sorted(set(labels))
        for label in unique_labels:
            mask = np.array(labels) == label
            if color_map and label in color_map:
                color = color_map[label]
            else:
                color = DATASET_COLORS.get(label, None)
            ax.scatter(x[mask], y[mask], s=s, alpha=alpha, label=label,
                       c=color, edgecolors="none")
        ax.legend(fontsize=7, markerscale=2, loc="best", framealpha=0.8)
    else:
        # Continuous
        sc = ax.scatter(x, y, s=s, alpha=alpha, c=colors, cmap=cmap, edgecolors="none")
        plt.colorbar(sc, ax=ax, shrink=0.8)

    if density:
        add_density_contours(ax, x, y)

    ax.set_title(title, fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])


def generate_visualization_grid(coords_2d, meta_df, method_name, filename_prefix):
    """Generate an extended grid of scatter plots with different colorings.

    Layout: 3 rows x 3 cols = 9 panels.
    """
    fig, axes = plt.subplots(3, 3, figsize=(20, 17))

    x, y = coords_2d[:, 0], coords_2d[:, 1]

    # ---- Row 0 ----

    # 1. Color by dataset
    make_scatter(axes[0, 0], x, y, None, f"{method_name} -- Dataset",
                 labels=meta_df["dataset_label"].values, density=True)

    # 2. Color by label (pos/neg)
    axes[0, 1].scatter(x[meta_df["label"].values == 0], y[meta_df["label"].values == 0],
                       s=8, alpha=0.4, c="#dc2626", label="Negative", edgecolors="none")
    axes[0, 1].scatter(x[meta_df["label"].values == 1], y[meta_df["label"].values == 1],
                       s=8, alpha=0.3, c="#2563eb", label="Positive", edgecolors="none")
    axes[0, 1].legend(fontsize=9, markerscale=2)
    axes[0, 1].set_title(f"{method_name} -- Label")
    axes[0, 1].set_xticks([])
    axes[0, 1].set_yticks([])
    add_density_contours(axes[0, 1], x, y)

    # 3. Color by editing rate (log scale)
    rates = meta_df["editing_rate"].values.copy()
    valid_rate = ~np.isnan(rates) & (rates > 0)
    log_rates = np.full_like(rates, np.nan)
    log_rates[valid_rate] = np.log10(np.clip(rates[valid_rate], 0.01, None))

    if valid_rate.sum() > 10:
        sc = axes[0, 2].scatter(x[valid_rate], y[valid_rate], s=8, alpha=0.5,
                                c=log_rates[valid_rate], cmap="viridis", edgecolors="none")
        plt.colorbar(sc, ax=axes[0, 2], shrink=0.8, label="log10(rate %)")
        axes[0, 2].scatter(x[~valid_rate], y[~valid_rate], s=4, alpha=0.15,
                           c="gray", edgecolors="none")
    axes[0, 2].set_title(f"{method_name} -- Editing Rate")
    axes[0, 2].set_xticks([])
    axes[0, 2].set_yticks([])

    # ---- Row 1 ----

    # 4. Color by TC-motif
    tc = meta_df["tc_motif"].values
    axes[1, 0].scatter(x[~tc], y[~tc], s=8, alpha=0.3, c="#94a3b8",
                       label="Non-TC", edgecolors="none")
    axes[1, 0].scatter(x[tc], y[tc], s=8, alpha=0.4, c="#dc2626",
                       label="TC-motif", edgecolors="none")
    axes[1, 0].legend(fontsize=9, markerscale=2)
    axes[1, 0].set_title(f"{method_name} -- TC-Motif")
    axes[1, 0].set_xticks([])
    axes[1, 0].set_yticks([])

    # 5. Color by delta MFE
    dmfe = meta_df["delta_mfe"].values.copy()
    valid_mfe = ~np.isnan(dmfe)
    if valid_mfe.sum() > 10:
        dmfe_clipped = np.clip(dmfe[valid_mfe], -2, 3)
        sc = axes[1, 1].scatter(x[valid_mfe], y[valid_mfe], s=8, alpha=0.5,
                                c=dmfe_clipped, cmap="coolwarm", edgecolors="none")
        plt.colorbar(sc, ax=axes[1, 1], shrink=0.8, label="Delta MFE (kcal/mol)")
        axes[1, 1].scatter(x[~valid_mfe], y[~valid_mfe], s=4, alpha=0.15,
                           c="gray", edgecolors="none")
    axes[1, 1].set_title(f"{method_name} -- Delta MFE")
    axes[1, 1].set_xticks([])
    axes[1, 1].set_yticks([])

    # 6. Color by positive dataset only (ignore negatives)
    pos_mask = meta_df["label"].values == 1
    pos_labels = meta_df["dataset_label"].values.copy()
    neg_mask = ~pos_mask
    axes[1, 2].scatter(x[neg_mask], y[neg_mask], s=4, alpha=0.15,
                       c="gray", label="Negative", edgecolors="none")
    for ds_label in ["Levanon", "Asaoka", "Sharma", "Alqassim", "Baysal"]:
        ds_mask = pos_mask & (pos_labels == ds_label)
        if ds_mask.sum() > 0:
            axes[1, 2].scatter(x[ds_mask], y[ds_mask], s=10, alpha=0.5,
                               c=DATASET_COLORS.get(ds_label, "#666"),
                               label=ds_label, edgecolors="none")
    axes[1, 2].legend(fontsize=7, markerscale=2, loc="best", framealpha=0.8)
    axes[1, 2].set_title(f"{method_name} -- Positive Sites by Source")
    axes[1, 2].set_xticks([])
    axes[1, 2].set_yticks([])

    # ---- Row 2 (NEW) ----

    # 7. Color by genomic feature category
    feat_labels = meta_df["feature_label"].values
    unique_feats = sorted(set(feat_labels))
    # Assign colors from the feature color map or a fallback palette
    fallback_palette = plt.cm.tab20(np.linspace(0, 1, 20))
    feat_color_map = {}
    fallback_idx = 0
    for f in unique_feats:
        if f in FEATURE_COLORS:
            feat_color_map[f] = FEATURE_COLORS[f]
        else:
            feat_color_map[f] = mcolors.to_hex(fallback_palette[fallback_idx % 20])
            fallback_idx += 1

    make_scatter(axes[2, 0], x, y, None, f"{method_name} -- Genomic Feature",
                 labels=feat_labels, color_map=feat_color_map)

    # 8. Density-only plot using hexbin
    axes[2, 1].hexbin(x, y, gridsize=40, cmap="YlOrRd", mincnt=1)
    axes[2, 1].set_title(f"{method_name} -- Density (hexbin)")
    axes[2, 1].set_xticks([])
    axes[2, 1].set_yticks([])
    add_density_contours(axes[2, 1], x, y, n_levels=8)

    # 9. Chromosome as color
    if "chr" in meta_df.columns:
        chr_vals = meta_df["chr"].values
        # Map chr to numeric for coloring
        unique_chr = sorted(set(chr_vals), key=lambda c: (
            int(c.replace("chr", "")) if c.replace("chr", "").isdigit() else 100 + ord(c[-1])
        ))
        chr_to_num = {c: i for i, c in enumerate(unique_chr)}
        chr_nums = np.array([chr_to_num.get(c, -1) for c in chr_vals], dtype=float)

        sc = axes[2, 2].scatter(x, y, s=6, alpha=0.4, c=chr_nums,
                                cmap="tab20", edgecolors="none")
        # Don't add colorbar since labels would be numeric; add a few reference ticks
        cbar = plt.colorbar(sc, ax=axes[2, 2], shrink=0.8)
        tick_positions = np.linspace(0, len(unique_chr) - 1, min(10, len(unique_chr)))
        cbar.set_ticks(tick_positions)
        cbar.set_ticklabels([unique_chr[int(t)] for t in tick_positions])
        cbar.ax.tick_params(labelsize=6)
    axes[2, 2].set_title(f"{method_name} -- Chromosome")
    axes[2, 2].set_xticks([])
    axes[2, 2].set_yticks([])

    plt.suptitle(f"Edit Effect Embedding Space -- {method_name}", fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{filename_prefix}.png")
    plt.close(fig)
    logger.info("Saved %s.png", filename_prefix)


def plot_pca_variance(variance_ratios):
    """Plot PCA explained variance."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    n = min(30, len(variance_ratios))
    ax1.bar(range(1, n + 1), variance_ratios[:n] * 100, color="#2563eb", alpha=0.8)
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Variance Explained (%)")
    ax1.set_title("Individual Variance per PC")

    cumvar = np.cumsum(variance_ratios[:50]) * 100
    ax2.plot(range(1, len(cumvar) + 1), cumvar, "o-", color="#2563eb", markersize=4)
    ax2.axhline(80, color="gray", linestyle="--", alpha=0.5, label="80%")
    ax2.axhline(90, color="gray", linestyle=":", alpha=0.5, label="90%")
    ax2.set_xlabel("Number of PCs")
    ax2.set_ylabel("Cumulative Variance (%)")
    ax2.set_title("Cumulative Variance Explained")
    ax2.legend()

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "pca_variance.png")
    plt.close(fig)
    logger.info("Saved pca_variance.png")


def generate_combined_mega_grid(dr_results, meta_df):
    """Generate a combined grid: 3 methods x 4 key coloring schemes.

    Rows: PCA, t-SNE, UMAP
    Cols: Dataset, Label, Editing Rate, Density
    """
    idx = dr_results["subsample_idx"]
    sub_meta = meta_df.iloc[idx].reset_index(drop=True)

    methods = []
    coords_list = []

    # PCA (subsample to match)
    methods.append("PCA")
    coords_list.append(dr_results["pca_2d"][idx])

    methods.append("t-SNE")
    coords_list.append(dr_results["tsne_2d"])

    if dr_results["umap_2d"] is not None:
        methods.append("UMAP")
        coords_list.append(dr_results["umap_2d"])

    n_methods = len(methods)
    n_cols = 4

    fig, axes = plt.subplots(n_methods, n_cols, figsize=(n_cols * 5.5, n_methods * 5))
    if n_methods == 1:
        axes = axes.reshape(1, -1)

    for row, (method, coords) in enumerate(zip(methods, coords_list)):
        x, y = coords[:, 0], coords[:, 1]

        # Col 0: Dataset
        make_scatter(axes[row, 0], x, y, None, f"{method} -- Dataset",
                     labels=sub_meta["dataset_label"].values, density=True)

        # Col 1: Label
        lab_vals = sub_meta["label"].values
        axes[row, 1].scatter(x[lab_vals == 0], y[lab_vals == 0],
                             s=8, alpha=0.4, c="#dc2626", label="Negative", edgecolors="none")
        axes[row, 1].scatter(x[lab_vals == 1], y[lab_vals == 1],
                             s=8, alpha=0.3, c="#2563eb", label="Positive", edgecolors="none")
        axes[row, 1].legend(fontsize=8, markerscale=2)
        axes[row, 1].set_title(f"{method} -- Label")
        axes[row, 1].set_xticks([])
        axes[row, 1].set_yticks([])
        add_density_contours(axes[row, 1], x, y)

        # Col 2: Editing rate
        rates = sub_meta["editing_rate"].values.copy()
        valid_rate = ~np.isnan(rates) & (rates > 0)
        if valid_rate.sum() > 10:
            log_r = np.log10(np.clip(rates[valid_rate], 0.01, None))
            sc = axes[row, 2].scatter(x[valid_rate], y[valid_rate], s=8, alpha=0.5,
                                      c=log_r, cmap="viridis", edgecolors="none")
            plt.colorbar(sc, ax=axes[row, 2], shrink=0.8, label="log10(rate %)")
            axes[row, 2].scatter(x[~valid_rate], y[~valid_rate], s=4, alpha=0.15,
                                 c="gray", edgecolors="none")
        axes[row, 2].set_title(f"{method} -- Editing Rate")
        axes[row, 2].set_xticks([])
        axes[row, 2].set_yticks([])

        # Col 3: Density hexbin
        axes[row, 3].hexbin(x, y, gridsize=40, cmap="YlOrRd", mincnt=1)
        axes[row, 3].set_title(f"{method} -- Density")
        axes[row, 3].set_xticks([])
        axes[row, 3].set_yticks([])
        add_density_contours(axes[row, 3], x, y, n_levels=8)

    plt.suptitle("Edit Effect Embedding Space -- Combined Methods", fontsize=16, y=1.01)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "combined_mega_grid.png")
    plt.close(fig)
    logger.info("Saved combined_mega_grid.png")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    diff_matrix, meta_df, valid_ids = load_data()

    dr_results = run_dimensionality_reduction(diff_matrix, n_subsample=MAX_SAMPLES)

    # PCA variance plot
    plot_pca_variance(dr_results["pca_variance"])

    # PCA visualization (full data)
    logger.info("Generating PCA plots...")
    generate_visualization_grid(
        dr_results["pca_2d"], meta_df, "PCA", "pca_grid"
    )

    # t-SNE visualization (subsampled)
    idx = dr_results["subsample_idx"]
    sub_meta = meta_df.iloc[idx].reset_index(drop=True)

    logger.info("Generating t-SNE plots...")
    generate_visualization_grid(
        dr_results["tsne_2d"], sub_meta, "t-SNE", "tsne_grid"
    )

    # UMAP visualization (subsampled)
    if dr_results["umap_2d"] is not None:
        logger.info("Generating UMAP plots...")
        generate_visualization_grid(
            dr_results["umap_2d"], sub_meta, "UMAP", "umap_grid"
        )

    # Combined mega-grid
    logger.info("Generating combined mega grid...")
    generate_combined_mega_grid(dr_results, meta_df)

    gc.collect()
    logger.info("All visualizations saved to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
