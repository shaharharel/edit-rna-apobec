#!/usr/bin/env python3
"""Embedding visualization: UMAP + t-SNE with multiple colorings and cluster metrics.

Generates publication-quality 2D scatter plots with coloring by:
- Dataset source (categorical)
- Label (positive/negative)
- Editing rate (continuous gradient)
- Delta MFE (coolwarm gradient)
- Genomic feature (categorical)
- Tissue breadth (continuous)

Also computes cluster separation scores (silhouette, Davies-Bouldin, Calinski-Harabasz).

Outputs saved to: experiments/apobec3a/outputs/embedding_viz_v2/
"""

import base64
import io
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
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logger = logging.getLogger(__name__)

EMB_DIR = PROJECT_ROOT / "data" / "processed" / "embeddings"
SPLITS_CSV = PROJECT_ROOT / "data" / "processed" / "splits_expanded.csv"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "apobec3a" / "outputs" / "embedding_viz_v2"

# Colors
DATASET_COLORS = {
    "advisor_c2t": "#1f77b4",  # Blue (Levanon)
    "asaoka_2019": "#2ca02c",  # Green
    "sharma_2015": "#d62728",  # Red
    "alqassim_2021": "#ff7f0e",  # Orange
    "baysal_2016": "#9467bd",  # Purple
    "tier2_negative": "#aaaaaa",  # Gray
    "tier3_negative": "#cccccc",  # Light gray
}

DATASET_LABELS = {
    "advisor_c2t": "Levanon",
    "asaoka_2019": "Asaoka",
    "sharma_2015": "Sharma",
    "alqassim_2021": "Alqassim",
    "baysal_2016": "Baysal",
    "tier2_negative": "Tier2 Neg",
    "tier3_negative": "Tier3 Neg",
}

FEATURE_COLORS = {
    "synonymous": "#1f77b4",
    "nonsynonymous": "#d62728",
    "stopgain": "#ff7f0e",
    "intronic": "#2ca02c",
    "ncRNA_exonic": "#9467bd",
    "UTR3": "#17becf",
    "UTR5": "#e377c2",
    "intergenic": "#7f7f7f",
}


def load_data():
    """Load embeddings and metadata."""
    logger.info("Loading embeddings and metadata...")
    pooled_orig = torch.load(EMB_DIR / "rnafm_pooled.pt", weights_only=False)
    pooled_edited = torch.load(EMB_DIR / "rnafm_pooled_edited.pt", weights_only=False)
    splits_df = pd.read_csv(SPLITS_CSV)

    # Load structure features
    struct_data = {}
    struct_path = EMB_DIR / "vienna_structure_cache.npz"
    if struct_path.exists():
        npz = np.load(struct_path, allow_pickle=True)
        if "site_ids" in npz and "delta_features" in npz:
            for i, sid in enumerate(npz["site_ids"]):
                struct_data[str(sid)] = npz["delta_features"][i]

    return pooled_orig, pooled_edited, splits_df, struct_data


def compute_embeddings(pooled_orig, pooled_edited, site_ids):
    """Compute subtraction and pooled embedding matrices."""
    pooled_embs = []
    sub_embs = []
    valid_ids = []

    for sid in site_ids:
        if sid in pooled_orig and sid in pooled_edited:
            po = pooled_orig[sid].numpy()
            pe = pooled_edited[sid].numpy()
            pooled_embs.append(po)
            sub_embs.append(pe - po)
            valid_ids.append(sid)

    return np.array(pooled_embs), np.array(sub_embs), valid_ids


def compute_dim_reductions(X, random_state=42):
    """Compute UMAP and t-SNE on embeddings (with PCA pre-reduction for speed)."""
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    # PCA pre-reduction: 640d -> 50d (standard practice, much faster for t-SNE/UMAP)
    n_pca = min(50, X.shape[1])
    logger.info("PCA pre-reduction: %dd -> %dd...", X.shape[1], n_pca)
    pca = PCA(n_components=n_pca, random_state=random_state)
    X_pca = pca.fit_transform(X)
    logger.info("  PCA variance explained (top 50): %.1f%%", pca.explained_variance_ratio_.sum() * 100)

    logger.info("Computing t-SNE (perplexity=30) on %dd PCA output...", n_pca)
    tsne = TSNE(n_components=2, perplexity=30, random_state=random_state, max_iter=1000,
                n_jobs=1)  # n_jobs=1 avoids OpenBLAS issues
    X_tsne = tsne.fit_transform(X_pca)

    logger.info("Computing UMAP (n_neighbors=15) on %dd PCA output...", n_pca)
    try:
        import umap
        reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                            random_state=random_state)
        X_umap = reducer.fit_transform(X_pca)
    except ImportError:
        logger.warning("umap-learn not installed, using t-SNE with perplexity=50 as fallback")
        tsne2 = TSNE(n_components=2, perplexity=50, random_state=random_state, max_iter=1000,
                     n_jobs=1)
        X_umap = tsne2.fit_transform(X_pca)

    return X_tsne, X_umap


def compute_cluster_metrics(X, labels, label_name="label"):
    """Compute cluster separation metrics."""
    unique_labels = np.unique(labels[~pd.isna(labels)])
    if len(unique_labels) < 2:
        return {}

    # Convert to integer labels
    label_map = {l: i for i, l in enumerate(unique_labels)}
    int_labels = np.array([label_map.get(l, -1) for l in labels])
    valid = int_labels >= 0
    X_valid = X[valid]
    int_labels_valid = int_labels[valid]

    if len(np.unique(int_labels_valid)) < 2:
        return {}

    # Subsample if too large for silhouette
    n = len(X_valid)
    if n > 5000:
        idx = np.random.RandomState(42).choice(n, 5000, replace=False)
        X_sub = X_valid[idx]
        labels_sub = int_labels_valid[idx]
    else:
        X_sub = X_valid
        labels_sub = int_labels_valid

    try:
        sil = silhouette_score(X_sub, labels_sub)
    except Exception:
        sil = None
    try:
        db = davies_bouldin_score(X_valid, int_labels_valid)
    except Exception:
        db = None
    try:
        ch = calinski_harabasz_score(X_valid, int_labels_valid)
    except Exception:
        ch = None

    return {
        "label_name": label_name,
        "n_clusters": int(len(unique_labels)),
        "n_samples": int(len(X_valid)),
        "silhouette": float(sil) if sil is not None else None,
        "davies_bouldin": float(db) if db is not None else None,
        "calinski_harabasz": float(ch) if ch is not None else None,
        "cluster_labels": {str(l): int(np.sum(int_labels_valid == i)) for l, i in label_map.items()},
    }


def fig_to_base64(fig, dpi=150):
    """Convert matplotlib figure to base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return b64


def plot_scatter_categorical(coords, labels, colors_map, labels_map, title, method_name,
                             figsize=(8, 6), alpha=0.4, s=8, neg_behind=True):
    """Create scatter plot with categorical coloring."""
    fig, ax = plt.subplots(figsize=figsize)

    unique_labels = list(colors_map.keys())

    # Draw negatives behind positives
    if neg_behind:
        order = [l for l in unique_labels if "neg" in l.lower()] + \
                [l for l in unique_labels if "neg" not in l.lower()]
    else:
        order = unique_labels

    for label in order:
        mask = labels == label
        if mask.sum() == 0:
            continue
        display_name = labels_map.get(label, label)
        color = colors_map.get(label, "#999999")
        ax.scatter(coords[mask, 0], coords[mask, 1], c=color, label=display_name,
                   s=s, alpha=alpha, edgecolors="none")

    ax.set_xlabel(f"{method_name} 1", fontsize=10)
    ax.set_ylabel(f"{method_name} 2", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(markerscale=3, fontsize=8, loc="upper right", framealpha=0.9)
    ax.tick_params(labelsize=8)

    return fig


def plot_scatter_continuous(coords, values, title, method_name, cmap="viridis",
                            figsize=(8, 6), alpha=0.5, s=8, vmin=None, vmax=None,
                            cbar_label=""):
    """Create scatter plot with continuous coloring."""
    fig, ax = plt.subplots(figsize=figsize)

    valid = ~np.isnan(values)
    # Draw NaN points first in gray
    if (~valid).sum() > 0:
        ax.scatter(coords[~valid, 0], coords[~valid, 1], c="#dddddd",
                   s=s * 0.5, alpha=0.1, edgecolors="none", label=f"N/A ({(~valid).sum()})")

    sc = ax.scatter(coords[valid, 0], coords[valid, 1], c=values[valid],
                    cmap=cmap, s=s, alpha=alpha, edgecolors="none",
                    vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
    cbar.set_label(cbar_label, fontsize=9)

    ax.set_xlabel(f"{method_name} 1", fontsize=10)
    ax.set_ylabel(f"{method_name} 2", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.tick_params(labelsize=8)
    if (~valid).sum() > 0:
        ax.legend(markerscale=3, fontsize=8, loc="upper right")

    return fig


def plot_multi_panel(coords_dict, labels_list, title_prefix, save_prefix, output_dir):
    """Create multi-panel figures for all colorings."""
    figures = {}

    for method_name, coords in coords_dict.items():
        for coloring_name, (values, plot_type, kwargs) in labels_list.items():
            key = f"{save_prefix}_{method_name}_{coloring_name}"
            fname = f"{key}.png"

            if plot_type == "categorical":
                fig = plot_scatter_categorical(
                    coords, values, title=f"{title_prefix}: {coloring_name} ({method_name})",
                    method_name=method_name, **kwargs)
            else:
                fig = plot_scatter_continuous(
                    coords, values, title=f"{title_prefix}: {coloring_name} ({method_name})",
                    method_name=method_name, **kwargs)

            fig.savefig(output_dir / fname, dpi=150, bbox_inches="tight", facecolor="white")
            figures[key] = fig_to_base64(fig)

    return figures


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    pooled_orig, pooled_edited, splits_df, struct_data = load_data()
    site_ids = splits_df["site_id"].tolist()
    pooled_X, sub_X, valid_ids = compute_embeddings(pooled_orig, pooled_edited, site_ids)

    logger.info("Valid embeddings: %d sites", len(valid_ids))

    # Build metadata for valid sites
    df = splits_df.set_index("site_id").loc[valid_ids].reset_index()

    # Compute delta MFE values
    delta_mfe = np.array([
        struct_data.get(sid, np.full(7, np.nan))[0] if sid in struct_data else np.nan
        for sid in valid_ids
    ])
    df["delta_mfe"] = delta_mfe

    # Compute editing rate (log2 scale for visualization)
    rates = df["editing_rate"].values.copy().astype(float)
    rates_present = ~np.isnan(rates)
    log2_rates = np.full_like(rates, np.nan)
    valid_rates = rates[rates_present]
    if (valid_rates > 1).any():
        valid_rates = np.where(valid_rates > 1, valid_rates / 100.0, valid_rates)
    log2_rates[rates_present] = np.log2(valid_rates + 0.01)
    df["log2_rate"] = log2_rates

    # Scale embeddings
    scaler = StandardScaler()
    sub_X_scaled = scaler.fit_transform(sub_X)

    # Compute dimensionality reductions on subtraction embeddings
    logger.info("Computing dimensionality reductions on subtraction embeddings (%d x %d)...",
                sub_X_scaled.shape[0], sub_X_scaled.shape[1])
    X_tsne, X_umap = compute_dim_reductions(sub_X_scaled)

    # Save coordinates
    coord_df = pd.DataFrame({
        "site_id": valid_ids,
        "tsne_1": X_tsne[:, 0],
        "tsne_2": X_tsne[:, 1],
        "umap_1": X_umap[:, 0],
        "umap_2": X_umap[:, 1],
        "dataset_source": df["dataset_source"].values,
        "label": df["label"].values,
        "editing_rate": df["editing_rate"].values,
        "log2_rate": df["log2_rate"].values,
        "delta_mfe": delta_mfe,
        "feature": df["feature"].values if "feature" in df.columns else "unknown",
    })
    coord_df.to_csv(OUTPUT_DIR / "embedding_coordinates.csv", index=False)

    # --- Generate all plots ---
    coords_dict = {"t-SNE": X_tsne, "UMAP": X_umap}
    all_figures = {}  # key -> base64

    datasets = df["dataset_source"].values
    labels = df["label"].values.astype(str)
    label_colors = {"1": "#1f77b4", "0": "#d62728"}
    label_names = {"1": "Positive (edited)", "0": "Negative"}

    features = df["feature"].values if "feature" in df.columns else np.full(len(valid_ids), "unknown")

    for method_name, coords in coords_dict.items():
        # 1. Dataset coloring
        fig = plot_scatter_categorical(
            coords, datasets, DATASET_COLORS, DATASET_LABELS,
            f"By Dataset ({method_name})", method_name)
        key = f"sub_{method_name}_dataset"
        fig.savefig(OUTPUT_DIR / f"{key}.png", dpi=150, bbox_inches="tight", facecolor="white")
        all_figures[key] = fig_to_base64(fig)

        # 2. Label coloring
        fig = plot_scatter_categorical(
            coords, labels, label_colors, label_names,
            f"Positive vs Negative ({method_name})", method_name, neg_behind=True)
        key = f"sub_{method_name}_label"
        fig.savefig(OUTPUT_DIR / f"{key}.png", dpi=150, bbox_inches="tight", facecolor="white")
        all_figures[key] = fig_to_base64(fig)

        # 3. Editing rate
        fig = plot_scatter_continuous(
            coords, log2_rates, f"Editing Rate ({method_name})", method_name,
            cmap="plasma", cbar_label="log2(rate + 0.01)")
        key = f"sub_{method_name}_rate"
        fig.savefig(OUTPUT_DIR / f"{key}.png", dpi=150, bbox_inches="tight", facecolor="white")
        all_figures[key] = fig_to_base64(fig)

        # 4. Delta MFE
        fig = plot_scatter_continuous(
            coords, delta_mfe, f"Delta MFE ({method_name})", method_name,
            cmap="coolwarm", cbar_label="Delta MFE (kcal/mol)",
            vmin=-2, vmax=2)
        key = f"sub_{method_name}_mfe"
        fig.savefig(OUTPUT_DIR / f"{key}.png", dpi=150, bbox_inches="tight", facecolor="white")
        all_figures[key] = fig_to_base64(fig)

        # 5. Genomic feature
        fig = plot_scatter_categorical(
            coords, features, FEATURE_COLORS,
            {k: k for k in FEATURE_COLORS},
            f"Genomic Feature ({method_name})", method_name, neg_behind=False)
        key = f"sub_{method_name}_feature"
        fig.savefig(OUTPUT_DIR / f"{key}.png", dpi=150, bbox_inches="tight", facecolor="white")
        all_figures[key] = fig_to_base64(fig)

    # --- Cluster separation metrics ---
    logger.info("Computing cluster separation metrics...")
    cluster_metrics = {}

    # By label (pos vs neg) on subtraction embeddings
    label_int = df["label"].values.astype(int)
    cluster_metrics["pos_vs_neg_sub"] = compute_cluster_metrics(
        sub_X_scaled, label_int.astype(str), "Positive vs Negative (subtraction)")

    # By dataset on subtraction embeddings
    cluster_metrics["by_dataset_sub"] = compute_cluster_metrics(
        sub_X_scaled, datasets, "By Dataset (subtraction)")

    # By label on t-SNE
    cluster_metrics["pos_vs_neg_tsne"] = compute_cluster_metrics(
        X_tsne, label_int.astype(str), "Positive vs Negative (t-SNE 2D)")

    # By dataset on t-SNE
    cluster_metrics["by_dataset_tsne"] = compute_cluster_metrics(
        X_tsne, datasets, "By Dataset (t-SNE 2D)")

    # By label on UMAP
    cluster_metrics["pos_vs_neg_umap"] = compute_cluster_metrics(
        X_umap, label_int.astype(str), "Positive vs Negative (UMAP 2D)")

    # K-means clustering quality
    for k in [2, 3, 5, 7]:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km_labels = km.fit_predict(sub_X_scaled)
        cluster_metrics[f"kmeans_k{k}_sub"] = compute_cluster_metrics(
            sub_X_scaled, km_labels.astype(str), f"KMeans (k={k}, subtraction)")

    # --- Additional bar chart / summary figures ---

    # Dataset size bar chart
    fig, ax = plt.subplots(figsize=(8, 4))
    ds_counts = df["dataset_source"].value_counts()
    ds_order = list(DATASET_COLORS.keys())
    ds_order = [d for d in ds_order if d in ds_counts.index]
    counts = [ds_counts.get(d, 0) for d in ds_order]
    colors = [DATASET_COLORS.get(d, "#999") for d in ds_order]
    names = [DATASET_LABELS.get(d, d) for d in ds_order]
    bars = ax.barh(range(len(ds_order)), counts, color=colors)
    ax.set_yticks(range(len(ds_order)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Number of Sites")
    ax.set_title("Dataset Sizes", fontweight="bold")
    for i, (bar, c) in enumerate(zip(bars, counts)):
        ax.text(bar.get_width() + 20, bar.get_y() + bar.get_height() / 2,
                str(c), va="center", fontsize=9)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "dataset_sizes.png", dpi=150, bbox_inches="tight", facecolor="white")
    all_figures["dataset_sizes"] = fig_to_base64(fig)

    # Editing rate distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    valid_log2 = log2_rates[~np.isnan(log2_rates)]
    ax.hist(valid_log2, bins=50, color="#1f77b4", alpha=0.7, edgecolor="white")
    ax.axvline(np.median(valid_log2), color="red", ls="--", label=f"Median: {np.median(valid_log2):.2f}")
    ax.set_xlabel("log2(editing rate + 0.01)")
    ax.set_ylabel("Count")
    ax.set_title("Editing Rate Distribution", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "rate_distribution.png", dpi=150, bbox_inches="tight", facecolor="white")
    all_figures["rate_distribution"] = fig_to_base64(fig)

    # Delta MFE distribution by label
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    pos_mfe = delta_mfe[label_int == 1]
    neg_mfe = delta_mfe[label_int == 0]
    pos_mfe = pos_mfe[~np.isnan(pos_mfe)]
    neg_mfe = neg_mfe[~np.isnan(neg_mfe)]
    axes[0].hist(pos_mfe, bins=50, color="#1f77b4", alpha=0.7, edgecolor="white")
    axes[0].set_title("Positive Sites", fontweight="bold")
    axes[0].set_xlabel("Delta MFE (kcal/mol)")
    axes[0].axvline(np.median(pos_mfe), color="red", ls="--",
                     label=f"Median: {np.median(pos_mfe):.3f}")
    axes[0].legend()
    axes[1].hist(neg_mfe, bins=50, color="#d62728", alpha=0.7, edgecolor="white")
    axes[1].set_title("Negative Sites", fontweight="bold")
    axes[1].set_xlabel("Delta MFE (kcal/mol)")
    axes[1].axvline(np.median(neg_mfe), color="red", ls="--",
                     label=f"Median: {np.median(neg_mfe):.3f}")
    axes[1].legend()
    fig.suptitle("Delta MFE Distribution by Label", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "mfe_by_label.png", dpi=150, bbox_inches="tight", facecolor="white")
    all_figures["mfe_by_label"] = fig_to_base64(fig)

    # --- Save results ---
    results = {
        "n_sites": len(valid_ids),
        "n_positive": int((label_int == 1).sum()),
        "n_negative": int((label_int == 0).sum()),
        "n_with_rates": int(rates_present.sum()),
        "embedding_type": "subtraction (edited - original, 640d)",
        "dim_reduction": {"tsne_perplexity": 30, "umap_n_neighbors": 15},
        "cluster_metrics": cluster_metrics,
        "figures": {k: f"data:image/png;base64,{v[:50]}..." for k, v in all_figures.items()},
    }
    with open(OUTPUT_DIR / "embedding_viz_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Save base64 figures separately for HTML embedding
    with open(OUTPUT_DIR / "figures_base64.json", "w") as f:
        json.dump(all_figures, f)

    logger.info("Done. Saved %d figures and cluster metrics to %s", len(all_figures), OUTPUT_DIR)
    logger.info("Cluster separation metrics:")
    for name, metrics in cluster_metrics.items():
        if metrics:
            sil = metrics.get("silhouette")
            db = metrics.get("davies_bouldin")
            ch = metrics.get("calinski_harabasz")
            logger.info("  %s: silhouette=%.4f, DB=%.4f, CH=%.1f",
                        name, sil or 0, db or 0, ch or 0)


if __name__ == "__main__":
    main()
