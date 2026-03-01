#!/usr/bin/env python
"""Tissue-colored UMAP and t-SNE visualizations of edit effect embeddings.

Loads RNA-FM pooled embeddings (original and edited), computes edit effect
embeddings (edited - original) for positive sites, then runs UMAP and t-SNE
with multiple tissue-related colorings:

    a. Max tissue editing rate (continuous colormap)
    b. Dominant tissue (categorical - which tissue has highest editing rate)
    c. Number of tissues with editing > 1% (tissue breadth)
    d. Tissue module (cluster sites by their tissue editing profile)
    e. Dataset source (which published dataset)

Saves all plots to experiments/apobec3a/outputs/embedding_viz/tissue_*.png
and coordinates + metadata to a CSV for later use.

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE python scripts/apobec3a/tissue_embedding_viz.py
"""

import os
import sys
import logging
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
EMB_DIR = PROJECT_ROOT / "data" / "processed" / "embeddings"
SPLITS_CSV = PROJECT_ROOT / "data" / "processed" / "splits_expanded.csv"
LABELS_CSV = PROJECT_ROOT / "data" / "processed" / "editing_sites_labels.csv"
T1_CSV = PROJECT_ROOT / "data" / "processed" / "advisor" / "t1_gtex_editing_&_conservation.csv"
EXCEL_PATH = PROJECT_ROOT / "data" / "raw" / "C2TFinalSites.DB.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "apobec" / "outputs" / "embedding_viz"

# ---------------------------------------------------------------------------
# GTEx tissue list (54 tissues, same order as in T1)
# ---------------------------------------------------------------------------
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

# Tissue groups for dominant tissue assignment
TISSUE_GROUPS = {
    "Brain": [t for t in GTEX_TISSUES if t.startswith("Brain_")],
    "Heart": [t for t in GTEX_TISSUES if t.startswith("Heart_")],
    "Kidney": [t for t in GTEX_TISSUES if t.startswith("Kidney_")],
    "Skin": [t for t in GTEX_TISSUES if t.startswith("Skin_")],
    "Artery": [t for t in GTEX_TISSUES if t.startswith("Artery_")],
    "Esophagus": [t for t in GTEX_TISSUES if t.startswith("Esophagus_")],
    "Colon": [t for t in GTEX_TISSUES if t.startswith("Colon_")],
    "Adipose": [t for t in GTEX_TISSUES if t.startswith("Adipose_")],
    "Cervix": [t for t in GTEX_TISSUES if t.startswith("Cervix_")],
}

DOMINANT_TISSUE_COLORS = {
    "Whole_Blood": "#dc2626",
    "Testis":      "#2563eb",
    "Brain":       "#7c3aed",
    "Liver":       "#b45309",
    "Lung":        "#059669",
    "Heart":       "#ec4899",
    "Kidney":      "#0891b2",
    "Spleen":      "#84cc16",
    "Artery":      "#f97316",
    "Colon":       "#8b5cf6",
    "Esophagus":   "#14b8a6",
    "Adipose":     "#eab308",
    "Other":       "#9ca3af",
}

# Tissue module definitions for hierarchical clustering coloring
TISSUE_MODULE_DEFS = {
    "Brain": [t for t in GTEX_TISSUES if t.startswith("Brain_")],
    "Blood/Immune": [t for t in GTEX_TISSUES if any(x in t for x in ["Blood", "Spleen", "lymphocyte"])],
    "Cardiovascular": [t for t in GTEX_TISSUES if any(x in t for x in ["Heart", "Artery"])],
    "GI/Digestive": [t for t in GTEX_TISSUES if any(x in t for x in ["Colon", "Esophagus", "Stomach", "Small_Intestine", "Liver", "Pancreas"])],
    "Reproductive": [t for t in GTEX_TISSUES if any(x in t for x in ["Ovary", "Testis", "Uterus", "Vagina", "Fallopian", "Cervix", "Prostate", "Breast"])],
    "Skin": [t for t in GTEX_TISSUES if "Skin" in t],
    "Other": [t for t in GTEX_TISSUES if not any(
        x in t for x in [
            "Brain_", "Blood", "Spleen", "lymphocyte", "Heart", "Artery",
            "Colon", "Esophagus", "Stomach", "Small_Intestine", "Liver",
            "Pancreas", "Ovary", "Testis", "Uterus", "Vagina", "Fallopian",
            "Cervix", "Prostate", "Breast", "Skin",
        ]
    )],
}

TISSUE_MODULE_COLORS = {
    "Brain":          "#7c3aed",
    "Blood/Immune":   "#dc2626",
    "Cardiovascular": "#ec4899",
    "GI/Digestive":   "#d97706",
    "Reproductive":   "#2563eb",
    "Skin":           "#eab308",
    "Other":          "#9ca3af",
    "Mixed":          "#6b7280",
}

# Style
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
})


# ---------------------------------------------------------------------------
# Tissue data loading from T1 CSV
# ---------------------------------------------------------------------------

def parse_tissue_rate(val):
    """Extract editing rate from 'mismatch;coverage;rate' string."""
    if pd.isna(val) or val == "" or val == "nan" or val == "NA":
        return np.nan
    parts = str(val).split(";")
    if len(parts) == 3:
        try:
            coverage = float(parts[1])
            rate = float(parts[2])
            if coverage < 10:
                return np.nan
            return rate
        except (ValueError, IndexError):
            pass
    return np.nan


def load_tissue_rates_from_t1_csv():
    """Load per-tissue editing rates from the processed T1 CSV.

    Returns a DataFrame indexed by (chr, start, end) with one column per tissue.
    """
    logger.info("Loading tissue rates from T1 CSV: %s", T1_CSV)
    t1 = pd.read_csv(T1_CSV)

    # Parse rates for all 54 GTEx tissues
    rate_data = {}
    for tissue in GTEX_TISSUES:
        matching = [c for c in t1.columns if tissue in c.replace(" ", "_")]
        if matching:
            rate_data[tissue] = t1[matching[0]].apply(parse_tissue_rate)
        else:
            rate_data[tissue] = pd.Series([np.nan] * len(t1))

    rates_df = pd.DataFrame(rate_data)
    rates_df["chr"] = t1["Chr"]
    rates_df["start"] = t1["Start"]
    rates_df["end"] = t1["End"]

    return rates_df


def get_dominant_tissue(tissue_rates_dict):
    """Given a dict {tissue: rate}, return the dominant tissue group name."""
    if not tissue_rates_dict:
        return "Other"

    # Build inverse mapping: individual tissue -> group
    tissue_to_group = {}
    for group, members in TISSUE_GROUPS.items():
        for member in members:
            tissue_to_group[member] = group

    # Aggregate by group: take max rate within each group
    group_rates = {}
    for tissue, rate in tissue_rates_dict.items():
        if np.isnan(rate) or rate <= 0:
            continue
        group = tissue_to_group.get(tissue, tissue)
        if group not in group_rates or rate > group_rates[group]:
            group_rates[group] = rate

    if not group_rates:
        return "Other"

    dominant = max(group_rates, key=group_rates.get)
    if dominant not in DOMINANT_TISSUE_COLORS:
        return "Other"
    return dominant


def get_dominant_tissue_module(tissue_rates_dict):
    """Given a dict {tissue: rate}, return the dominant tissue module based on
    which module has the highest mean editing rate."""
    if not tissue_rates_dict:
        return "Other"

    module_means = {}
    for module, members in TISSUE_MODULE_DEFS.items():
        rates = [tissue_rates_dict.get(m, np.nan) for m in members]
        rates = [r for r in rates if not np.isnan(r) and r > 0]
        if rates:
            module_means[module] = np.mean(rates)

    if not module_means:
        return "Other"

    # If top two modules are close (within 20% relative), call it Mixed
    sorted_modules = sorted(module_means.items(), key=lambda x: -x[1])
    if len(sorted_modules) >= 2:
        top_val = sorted_modules[0][1]
        second_val = sorted_modules[1][1]
        if top_val > 0 and second_val / top_val > 0.8:
            return "Mixed"

    return sorted_modules[0][0]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load site metadata from splits_expanded.csv
    # ------------------------------------------------------------------
    logger.info("Loading splits from %s", SPLITS_CSV)
    splits_df = pd.read_csv(SPLITS_CSV)
    # Focus on positive sites only (label == 1) since edit effects
    # are meaningful only for sites that actually undergo editing.
    # Also include all dataset sources to enable coloring by source.
    positive_df = splits_df[splits_df["label"] == 1].copy()
    logger.info("  Total positive sites: %d", len(positive_df))

    # ------------------------------------------------------------------
    # 2. Load embeddings and compute edit effects
    # ------------------------------------------------------------------
    logger.info("Loading RNA-FM pooled embeddings...")
    pooled_orig = torch.load(EMB_DIR / "rnafm_pooled.pt", weights_only=False, map_location="cpu")
    pooled_edited = torch.load(EMB_DIR / "rnafm_pooled_edited.pt", weights_only=False, map_location="cpu")

    available = set(pooled_orig.keys()) & set(pooled_edited.keys())
    positive_df = positive_df[positive_df["site_id"].isin(available)].copy()
    logger.info("  Positive sites with embeddings: %d", len(positive_df))

    site_ids = positive_df["site_id"].tolist()
    diff_embs = []
    valid_ids = []
    for sid in site_ids:
        diff = (pooled_edited[sid] - pooled_orig[sid]).numpy()
        diff_embs.append(diff)
        valid_ids.append(sid)

    diff_matrix = np.array(diff_embs)  # (N, 640)
    meta_df = positive_df.set_index("site_id").loc[valid_ids].reset_index()
    meta_df["editing_rate"] = pd.to_numeric(meta_df["editing_rate"], errors="coerce")
    logger.info("  Edit effect matrix shape: %s", diff_matrix.shape)

    # ------------------------------------------------------------------
    # 3. Load Levanon tissue editing rates
    # ------------------------------------------------------------------
    logger.info("Loading tissue editing rates...")
    # Load labels CSV which has site_id -> (chr, start, end) mapping for Levanon sites
    labels_df = pd.read_csv(LABELS_CSV)
    labels_lookup = {
        row["site_id"]: (row["chr"], int(row["start"]), int(row["end"]))
        for _, row in labels_df.iterrows()
    }

    # Load T1 tissue rates
    tissue_rates_df = load_tissue_rates_from_t1_csv()
    # Build a lookup: (chr, start, end) -> {tissue: rate}
    tissue_lookup = {}
    for idx, row in tissue_rates_df.iterrows():
        locus = (str(row["chr"]), int(row["start"]), int(row["end"]))
        rates = {}
        for tissue in GTEX_TISSUES:
            val = row.get(tissue, np.nan)
            if not np.isnan(val):
                rates[tissue] = val
        tissue_lookup[locus] = rates

    logger.info("  T1 tissue rate entries: %d", len(tissue_lookup))

    # ------------------------------------------------------------------
    # 4. Build tissue metadata for each positive site
    # ------------------------------------------------------------------
    BREADTH_THRESHOLD = 1.0  # percent

    max_tissue_rates = []
    dominant_tissues = []
    tissue_breadths = []
    tissue_modules = []
    tissue_rate_vectors = []  # for module clustering

    for _, row in meta_df.iterrows():
        sid = row["site_id"]
        locus = labels_lookup.get(sid)
        rates_dict = tissue_lookup.get(locus, {}) if locus else {}

        if rates_dict:
            rates_vals = [r for r in rates_dict.values() if not np.isnan(r)]
            max_rate = max(rates_vals) if rates_vals else 0.0
            n_above = sum(1 for r in rates_vals if r >= BREADTH_THRESHOLD)
            dominant = get_dominant_tissue(rates_dict)
            module = get_dominant_tissue_module(rates_dict)

            # Build rate vector for clustering
            rate_vec = np.array([rates_dict.get(t, 0.0) for t in GTEX_TISSUES])
        else:
            max_rate = 0.0
            n_above = 0
            dominant = "Other"
            module = "Other"
            rate_vec = np.zeros(len(GTEX_TISSUES))

        max_tissue_rates.append(max_rate)
        dominant_tissues.append(dominant)
        tissue_breadths.append(n_above)
        tissue_modules.append(module)
        tissue_rate_vectors.append(rate_vec)

    meta_df["max_tissue_rate"] = max_tissue_rates
    meta_df["dominant_tissue"] = dominant_tissues
    meta_df["tissue_breadth"] = tissue_breadths
    meta_df["tissue_module"] = tissue_modules

    # Also add dataset source labels
    logger.info("  Dominant tissue distribution:")
    for dt, count in meta_df["dominant_tissue"].value_counts().items():
        logger.info("    %s: %d", dt, count)
    logger.info("  Tissue module distribution:")
    for tm, count in meta_df["tissue_module"].value_counts().items():
        logger.info("    %s: %d", tm, count)

    # ------------------------------------------------------------------
    # 5. Dimensionality reduction: UMAP + t-SNE
    # ------------------------------------------------------------------
    # PCA pre-reduction for speed and stability
    from sklearn.decomposition import PCA

    n_pca = min(50, diff_matrix.shape[0] - 1, diff_matrix.shape[1])
    logger.info("Running PCA (%d components) for pre-reduction...", n_pca)
    pca = PCA(n_components=n_pca, random_state=42)
    pca_embs = pca.fit_transform(diff_matrix)
    logger.info("  PCA variance explained (top 2): %.1f%%, %.1f%%",
                pca.explained_variance_ratio_[0] * 100,
                pca.explained_variance_ratio_[1] * 100)

    # t-SNE
    logger.info("Running t-SNE (perplexity=30)...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    tsne_coords = tsne.fit_transform(pca_embs)
    logger.info("  t-SNE complete.")

    # UMAP
    logger.info("Running UMAP (n_neighbors=30, min_dist=0.3)...")
    import umap
    reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.3, random_state=42)
    umap_coords = reducer.fit_transform(pca_embs)
    logger.info("  UMAP complete.")

    meta_df["umap_1"] = umap_coords[:, 0]
    meta_df["umap_2"] = umap_coords[:, 1]
    meta_df["tsne_1"] = tsne_coords[:, 0]
    meta_df["tsne_2"] = tsne_coords[:, 1]

    # ------------------------------------------------------------------
    # 6. Plotting helpers
    # ------------------------------------------------------------------

    def _save_scatter_continuous(coords, values, cmap, label, title, filename,
                                 log_scale=False, vmin=None, vmax=None):
        """Save a scatter plot with continuous colormap."""
        fig, ax = plt.subplots(figsize=(10, 8))
        plot_vals = values.copy().astype(float)
        valid = ~np.isnan(plot_vals) & (plot_vals > 0) if log_scale else ~np.isnan(plot_vals)

        if log_scale:
            plot_vals[valid] = np.log10(np.clip(plot_vals[valid], 0.01, None))

        if valid.sum() > 0:
            sc = ax.scatter(coords[valid, 0], coords[valid, 1],
                            c=plot_vals[valid], cmap=cmap, alpha=0.65, s=14,
                            edgecolors="none", vmin=vmin, vmax=vmax)
            plt.colorbar(sc, ax=ax, label=label, shrink=0.85)
        if (~valid).sum() > 0:
            ax.scatter(coords[~valid, 0], coords[~valid, 1],
                       s=6, alpha=0.15, c="gray", edgecolors="none", label="No data")
            ax.legend(fontsize=9, markerscale=2, framealpha=0.8)

        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Dimension 1", fontsize=11)
        ax.set_ylabel("Dimension 2", fontsize=11)
        ax.tick_params(labelsize=9)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / filename, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info("  Saved %s", filename)

    def _save_scatter_categorical(coords, categories, color_map, title, filename,
                                   order=None):
        """Save a scatter plot with categorical coloring."""
        fig, ax = plt.subplots(figsize=(10, 8))
        if order is None:
            order = sorted(set(categories))

        auto_cmap = matplotlib.colormaps.get_cmap("tab20").resampled(max(len(order), 1))
        for i, cat in enumerate(order):
            mask = np.array(categories) == cat
            n_points = mask.sum()
            if n_points == 0:
                continue
            color = color_map.get(cat, auto_cmap(i))
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=[color], alpha=0.6, s=14, edgecolors="none",
                       label=f"{cat} ({n_points})")

        ax.legend(fontsize=8, markerscale=2, framealpha=0.8, loc="best",
                  ncol=max(1, len(order) // 8))
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Dimension 1", fontsize=11)
        ax.set_ylabel("Dimension 2", fontsize=11)
        ax.tick_params(labelsize=9)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / filename, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info("  Saved %s", filename)

    # ------------------------------------------------------------------
    # 7. Generate all plots
    # ------------------------------------------------------------------
    logger.info("Generating tissue-colored embedding visualizations...")

    # (a) Max tissue editing rate (continuous)
    max_rates_arr = meta_df["max_tissue_rate"].values

    _save_scatter_continuous(
        umap_coords, max_rates_arr, "viridis",
        "log10(max editing rate %)", "UMAP -- Max Tissue Editing Rate",
        "tissue_umap_max_rate.png", log_scale=True,
    )
    _save_scatter_continuous(
        tsne_coords, max_rates_arr, "viridis",
        "log10(max editing rate %)", "t-SNE -- Max Tissue Editing Rate",
        "tissue_tsne_max_rate.png", log_scale=True,
    )

    # (b) Dominant tissue (categorical)
    dominant_tissue_list = meta_df["dominant_tissue"].tolist()
    dom_order = [k for k in DOMINANT_TISSUE_COLORS if k in set(dominant_tissue_list)]
    # Ensure "Other" is last
    if "Other" in dom_order:
        dom_order.remove("Other")
        dom_order.append("Other")

    _save_scatter_categorical(
        umap_coords, dominant_tissue_list, DOMINANT_TISSUE_COLORS,
        "UMAP -- Dominant Tissue (highest editing rate)",
        "tissue_umap_dominant_tissue.png", order=dom_order,
    )
    _save_scatter_categorical(
        tsne_coords, dominant_tissue_list, DOMINANT_TISSUE_COLORS,
        "t-SNE -- Dominant Tissue (highest editing rate)",
        "tissue_tsne_dominant_tissue.png", order=dom_order,
    )

    # (c) Tissue breadth (number of tissues with editing > 1%)
    breadth_arr = meta_df["tissue_breadth"].values.astype(float)

    _save_scatter_continuous(
        umap_coords, breadth_arr, "YlOrRd",
        f"# Tissues with rate >= {BREADTH_THRESHOLD}%",
        "UMAP -- Tissue Breadth (# tissues with editing > 1%)",
        "tissue_umap_breadth.png",
    )
    _save_scatter_continuous(
        tsne_coords, breadth_arr, "YlOrRd",
        f"# Tissues with rate >= {BREADTH_THRESHOLD}%",
        "t-SNE -- Tissue Breadth (# tissues with editing > 1%)",
        "tissue_tsne_breadth.png",
    )

    # (d) Tissue module (based on tissue profile)
    module_list = meta_df["tissue_module"].tolist()
    mod_order = [k for k in TISSUE_MODULE_COLORS if k in set(module_list)]

    _save_scatter_categorical(
        umap_coords, module_list, TISSUE_MODULE_COLORS,
        "UMAP -- Tissue Module (dominant editing module)",
        "tissue_umap_module.png", order=mod_order,
    )
    _save_scatter_categorical(
        tsne_coords, module_list, TISSUE_MODULE_COLORS,
        "t-SNE -- Tissue Module (dominant editing module)",
        "tissue_tsne_module.png", order=mod_order,
    )

    # (e) Dataset source
    source_list = meta_df["dataset_source"].tolist()
    source_colors = {
        "advisor_c2t":    "#228833",
        "sharma_2015":    "#ee6677",
        "baysal_2016":    "#4477aa",
        "asaoka_2019":    "#ccbb44",
        "alqassim_2021":  "#aa3377",
    }
    source_order = [k for k in source_colors if k in set(source_list)]
    # Add any sources not in our predefined colors
    for s in sorted(set(source_list)):
        if s not in source_order:
            source_order.append(s)

    _save_scatter_categorical(
        umap_coords, source_list, source_colors,
        "UMAP -- Dataset Source",
        "tissue_umap_dataset_source.png", order=source_order,
    )
    _save_scatter_categorical(
        tsne_coords, source_list, source_colors,
        "t-SNE -- Dataset Source",
        "tissue_tsne_dataset_source.png", order=source_order,
    )

    # ------------------------------------------------------------------
    # 8. Combined multi-panel figure (UMAP)
    # ------------------------------------------------------------------
    logger.info("Generating combined multi-panel figure...")

    fig, axes = plt.subplots(2, 3, figsize=(24, 14))

    # Row 1: UMAP panels
    # Panel (0,0): Max rate
    ax = axes[0, 0]
    valid = max_rates_arr > 0
    log_rates = np.full_like(max_rates_arr, np.nan, dtype=float)
    log_rates[valid] = np.log10(np.clip(max_rates_arr[valid], 0.01, None))
    if valid.sum() > 0:
        sc = ax.scatter(umap_coords[valid, 0], umap_coords[valid, 1],
                        c=log_rates[valid], cmap="viridis", alpha=0.65, s=12,
                        edgecolors="none")
        plt.colorbar(sc, ax=ax, shrink=0.8, label="log10(max rate %)")
    if (~valid).sum() > 0:
        ax.scatter(umap_coords[~valid, 0], umap_coords[~valid, 1],
                   s=5, alpha=0.15, c="gray", edgecolors="none")
    ax.set_title("A) UMAP -- Max Editing Rate", fontsize=12, fontweight="bold")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    # Panel (0,1): Dominant tissue
    ax = axes[0, 1]
    for cat in dom_order:
        mask = np.array(dominant_tissue_list) == cat
        n_pts = mask.sum()
        if n_pts == 0:
            continue
        color = DOMINANT_TISSUE_COLORS.get(cat, "#9ca3af")
        ax.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                   c=[color], alpha=0.6, s=12, edgecolors="none",
                   label=f"{cat} ({n_pts})")
    ax.legend(fontsize=7, markerscale=2, framealpha=0.8, loc="best", ncol=2)
    ax.set_title("B) UMAP -- Dominant Tissue", fontsize=12, fontweight="bold")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    # Panel (0,2): Tissue breadth
    ax = axes[0, 2]
    sc = ax.scatter(umap_coords[:, 0], umap_coords[:, 1],
                    c=breadth_arr, cmap="YlOrRd", alpha=0.65, s=12,
                    edgecolors="none")
    plt.colorbar(sc, ax=ax, shrink=0.8, label="# Tissues (rate >= 1%)")
    ax.set_title("C) UMAP -- Tissue Breadth", fontsize=12, fontweight="bold")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    # Row 2: t-SNE panels
    # Panel (1,0): Tissue module
    ax = axes[1, 0]
    for cat in mod_order:
        mask = np.array(module_list) == cat
        n_pts = mask.sum()
        if n_pts == 0:
            continue
        color = TISSUE_MODULE_COLORS.get(cat, "#9ca3af")
        ax.scatter(tsne_coords[mask, 0], tsne_coords[mask, 1],
                   c=[color], alpha=0.6, s=12, edgecolors="none",
                   label=f"{cat} ({n_pts})")
    ax.legend(fontsize=7, markerscale=2, framealpha=0.8, loc="best", ncol=2)
    ax.set_title("D) t-SNE -- Tissue Module", fontsize=12, fontweight="bold")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")

    # Panel (1,1): Dataset source
    ax = axes[1, 1]
    for cat in source_order:
        mask = np.array(source_list) == cat
        n_pts = mask.sum()
        if n_pts == 0:
            continue
        auto_cmap = matplotlib.colormaps.get_cmap("tab10").resampled(len(source_order))
        color = source_colors.get(cat, auto_cmap(source_order.index(cat)))
        ax.scatter(tsne_coords[mask, 0], tsne_coords[mask, 1],
                   c=[color], alpha=0.6, s=12, edgecolors="none",
                   label=f"{cat} ({n_pts})")
    ax.legend(fontsize=7, markerscale=2, framealpha=0.8, loc="best")
    ax.set_title("E) t-SNE -- Dataset Source", fontsize=12, fontweight="bold")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")

    # Panel (1,2): Max rate on t-SNE
    ax = axes[1, 2]
    if valid.sum() > 0:
        sc = ax.scatter(tsne_coords[valid, 0], tsne_coords[valid, 1],
                        c=log_rates[valid], cmap="viridis", alpha=0.65, s=12,
                        edgecolors="none")
        plt.colorbar(sc, ax=ax, shrink=0.8, label="log10(max rate %)")
    if (~valid).sum() > 0:
        ax.scatter(tsne_coords[~valid, 0], tsne_coords[~valid, 1],
                   s=5, alpha=0.15, c="gray", edgecolors="none")
    ax.set_title("F) t-SNE -- Max Editing Rate", fontsize=12, fontweight="bold")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")

    plt.suptitle(
        "Tissue-Colored Edit Effect Embeddings (Positive Sites Only)\n"
        "RNA-FM pooled, edit = edited - original",
        fontsize=14, y=1.01,
    )
    plt.tight_layout()
    combined_path = OUTPUT_DIR / "tissue_combined_panel.png"
    fig.savefig(combined_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved %s", combined_path)

    # ------------------------------------------------------------------
    # 9. Save coordinates + metadata to CSV
    # ------------------------------------------------------------------
    output_csv = OUTPUT_DIR / "tissue_embedding_coordinates.csv"
    export_cols = [
        "site_id", "chr", "start", "end", "gene", "dataset_source",
        "editing_rate", "max_tissue_rate", "dominant_tissue",
        "tissue_breadth", "tissue_module",
        "umap_1", "umap_2", "tsne_1", "tsne_2",
    ]
    # Only export columns that exist
    export_cols = [c for c in export_cols if c in meta_df.columns]
    meta_df[export_cols].to_csv(output_csv, index=False)
    logger.info("Saved coordinates CSV: %s", output_csv)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 60)
    logger.info("Tissue Embedding Visualization Complete")
    logger.info("=" * 60)
    logger.info("  Total positive sites visualized: %d", len(meta_df))

    # Count sites with tissue data (Levanon sites)
    n_with_tissue = (meta_df["max_tissue_rate"] > 0).sum()
    logger.info("  Sites with tissue rate data: %d", n_with_tissue)
    logger.info("  Sites without tissue data: %d", len(meta_df) - n_with_tissue)

    logger.info("")
    logger.info("  Output files:")
    output_files = list(OUTPUT_DIR.glob("tissue_*.png"))
    for f in sorted(output_files):
        logger.info("    %s", f.name)
    logger.info("    %s", output_csv.name)
    logger.info("")
    logger.info("Done.")


if __name__ == "__main__":
    main()
