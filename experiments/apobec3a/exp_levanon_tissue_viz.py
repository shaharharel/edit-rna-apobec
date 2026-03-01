#!/usr/bin/env python
"""Embedding visualization of Levanon dataset sites colored by tissue editing patterns and cancer associations.

Generates a multi-panel figure with PCA (and optionally t-SNE) projections of edit effect
embeddings, colored by dominant tissue, tissue breadth, editing rate, and TCGA survival
associations.

Data sources:
    - Embeddings: data/processed/embeddings/rnafm_pooled.pt, rnafm_pooled_edited.pt
    - Site metadata: data/processed/splits_expanded.csv
    - Tissue editing: data/raw/C2TFinalSites.DB.xlsx (T1-GTEx Editing & Conservation)
    - Cancer survival: data/raw/C2TFinalSites.DB.xlsx (T5 -TCGA Survival)

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE /Users/shaharharel/miniconda3/envs/quris/bin/python \\
        experiments/apobec3a/exp_levanon_tissue_viz.py
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

EMB_DIR = PROJECT_ROOT / "data" / "processed" / "embeddings"
SPLITS_CSV = PROJECT_ROOT / "data" / "processed" / "splits_expanded.csv"
EXCEL_PATH = PROJECT_ROOT / "data" / "raw" / "C2TFinalSites.DB.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "apobec" / "outputs" / "embedding_viz"

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
})

# Tissue groups: map fine-grained GTEx tissues into broader categories for coloring.
# The tissue classification column in T1 already provides a clean categorization.
TISSUE_CLASS_COLORS = {
    "Whole Blood":       "#dc2626",  # red
    "Testis":            "#2563eb",  # blue
    "Brain":             "#7c3aed",  # purple
    "Ubiquitous":        "#16a34a",  # green
    "Intestine":         "#d97706",  # orange
    "Non-Specific":      "#6b7280",  # gray
}

# Colors for the tissue classification categories as stored in the Excel
TISSUE_CLASSIFICATION_COLORS = {
    "Blood Specific":      "#dc2626",
    "Testis Specific":     "#2563eb",
    "Ubiquitous":          "#16a34a",
    "Non-Specific":        "#6b7280",
    "Intestine Specific":  "#d97706",
}

# Top tissues to consider for "dominant tissue" coloring
# These are grouped to simplify the visualization
TISSUE_GROUPS = {
    "Brain": [
        "Brain Amygdala", "Brain Anterior cingulate cortex BA24",
        "Brain Caudate basal ganglia", "Brain Cerebellar Hemisphere",
        "Brain Cerebellum", "Brain Cortex", "Brain Frontal Cortex BA9",
        "Brain Hippocampus", "Brain Hypothalamus",
        "Brain Nucleus accumbens basal ganglia",
        "Brain Putamen basal ganglia", "Brain Spinal cord cervical c-1",
        "Brain Substantia nigra",
    ],
    "Heart": ["Heart Atrial Appendage", "Heart Left Ventricle"],
    "Kidney": ["Kidney Cortex", "Kidney Medulla"],
    "Skin": ["Skin Not Sun Exposed Suprapubic", "Skin Sun Exposed Lower leg"],
    "Artery": ["Artery Aorta", "Artery Coronary", "Artery Tibial"],
    "Esophagus": ["Esophagus Gastroesophageal Junction", "Esophagus Mucosa", "Esophagus Muscularis"],
    "Colon": ["Colon Sigmoid", "Colon Transverse"],
    "Adipose": ["Adipose Subcutaneous", "Adipose Visceral Omentum"],
}

# Colors for dominant tissue group (top-level)
DOMINANT_TISSUE_COLORS = {
    "Whole Blood":  "#dc2626",
    "Testis":       "#2563eb",
    "Brain":        "#7c3aed",
    "Liver":        "#b45309",
    "Lung":         "#059669",
    "Heart":        "#ec4899",
    "Kidney":       "#0891b2",
    "Spleen":       "#84cc16",
    "Artery":       "#f97316",
    "Other":        "#9ca3af",
}


def parse_tissue_editing_rate(cell_value):
    """Parse a tissue cell value like '346;2270;15.2' -> editing rate (15.2).
    Returns NaN if the value can't be parsed or coverage is too low."""
    if pd.isna(cell_value) or cell_value == "" or cell_value == "NA":
        return np.nan
    parts = str(cell_value).split(";")
    if len(parts) != 3:
        return np.nan
    try:
        mismatched = float(parts[0])
        coverage = float(parts[1])
        rate = float(parts[2])
        # Require minimum coverage
        if coverage < 10:
            return np.nan
        return rate
    except (ValueError, IndexError):
        return np.nan


def load_t1_tissue_data(excel_path):
    """Load tissue editing rates from the T1 sheet.

    Returns:
        tissue_data: dict of (chr, start, end) -> dict with keys:
            'tissue_rates': {tissue_name: editing_rate},
            'n_tissues': int,
            'tissue_classification': str,
            'max_rate': float,
            'mean_rate': float,
    """
    t1 = pd.read_excel(excel_path, sheet_name="T1-GTEx Editing & Conservation", header=None)

    # Row 1 has column headers; tissue columns start at col 26 through col 79
    tissue_names = []
    for col_idx in range(26, min(80, t1.shape[1])):
        name = t1.iloc[1, col_idx]
        if pd.notna(name):
            tissue_names.append((col_idx, str(name)))

    tissue_data = {}
    for row_idx in range(2, t1.shape[0]):
        chr_val = str(t1.iloc[row_idx, 0])
        start = int(t1.iloc[row_idx, 1])
        end = int(t1.iloc[row_idx, 2])

        n_tissues = t1.iloc[row_idx, 7]
        tissue_class = t1.iloc[row_idx, 9]
        max_rate = t1.iloc[row_idx, 11]
        mean_rate = t1.iloc[row_idx, 12]

        # Parse per-tissue editing rates
        tissue_rates = {}
        for col_idx, tissue_name in tissue_names:
            rate = parse_tissue_editing_rate(t1.iloc[row_idx, col_idx])
            if not np.isnan(rate):
                tissue_rates[tissue_name] = rate

        tissue_data[(chr_val, start, end)] = {
            "tissue_rates": tissue_rates,
            "n_tissues": int(n_tissues) if pd.notna(n_tissues) else 0,
            "tissue_classification": str(tissue_class) if pd.notna(tissue_class) else "Unknown",
            "max_rate": float(max_rate) if pd.notna(max_rate) else 0.0,
            "mean_rate": float(mean_rate) if pd.notna(mean_rate) else 0.0,
        }

    return tissue_data


def load_t5_survival_data(excel_path):
    """Load TCGA survival association data from the T5 sheet.

    Returns:
        survival_data: dict of (chr, start, end) -> dict with keys:
            'cancers': str (semicolon-separated cancer types),
            'n_cancers': int,
    """
    t5 = pd.read_excel(excel_path, sheet_name="T5 -TCGA Survival", header=None)

    survival_data = {}
    for row_idx in range(2, t5.shape[0]):
        chr_val = t5.iloc[row_idx, 0]
        start_val = t5.iloc[row_idx, 1]
        end_val = t5.iloc[row_idx, 2]
        # Skip blank rows
        if pd.isna(chr_val) or pd.isna(start_val) or pd.isna(end_val):
            continue
        chr_val = str(chr_val)
        start = int(start_val)
        end = int(end_val)
        cancers = t5.iloc[row_idx, 13]
        n_cancers = t5.iloc[row_idx, 14]

        survival_data[(chr_val, start, end)] = {
            "cancers": str(cancers) if pd.notna(cancers) else "",
            "n_cancers": int(n_cancers) if pd.notna(n_cancers) else 0,
        }

    return survival_data


def get_dominant_tissue_group(tissue_rates):
    """Given a dict of {tissue: rate}, return the dominant tissue group name."""
    if not tissue_rates:
        return "Other"

    # Build inverse mapping: individual tissue -> group name
    tissue_to_group = {}
    for group, members in TISSUE_GROUPS.items():
        for member in members:
            tissue_to_group[member] = group

    # Aggregate by group: take max rate within each group
    group_rates = {}
    for tissue, rate in tissue_rates.items():
        group = tissue_to_group.get(tissue, tissue)  # ungrouped tissues are their own group
        if group not in group_rates or rate > group_rates[group]:
            group_rates[group] = rate

    # Return the group with the highest rate
    dominant = max(group_rates, key=group_rates.get)

    # Simplify: if not in our predefined color map, call it "Other"
    if dominant not in DOMINANT_TISSUE_COLORS:
        return "Other"
    return dominant


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load Levanon sites from splits_expanded.csv
    # ------------------------------------------------------------------
    print("Loading site metadata...")
    splits_df = pd.read_csv(SPLITS_CSV)
    levanon_df = splits_df[splits_df["dataset_source"] == "advisor_c2t"].copy()
    print(f"  Levanon sites: {len(levanon_df)}")

    # ------------------------------------------------------------------
    # 2. Load embeddings and compute edit effects
    # ------------------------------------------------------------------
    print("Loading embeddings...")
    pooled_orig = torch.load(EMB_DIR / "rnafm_pooled.pt", weights_only=False)
    pooled_edited = torch.load(EMB_DIR / "rnafm_pooled_edited.pt", weights_only=False)

    available = set(pooled_orig.keys()) & set(pooled_edited.keys())
    levanon_df = levanon_df[levanon_df["site_id"].isin(available)].copy()
    print(f"  Levanon sites with embeddings: {len(levanon_df)}")

    site_ids = levanon_df["site_id"].tolist()
    diff_embs = []
    valid_ids = []
    for sid in site_ids:
        diff = (pooled_edited[sid] - pooled_orig[sid]).numpy()
        diff_embs.append(diff)
        valid_ids.append(sid)

    diff_matrix = np.array(diff_embs)
    meta_df = levanon_df.set_index("site_id").loc[valid_ids].reset_index()
    meta_df["editing_rate"] = pd.to_numeric(meta_df["editing_rate"], errors="coerce")
    print(f"  Edit effect matrix shape: {diff_matrix.shape}")

    # ------------------------------------------------------------------
    # 3. Load tissue data from Excel T1
    # ------------------------------------------------------------------
    print("Loading tissue editing data from Excel T1...")
    tissue_data = load_t1_tissue_data(EXCEL_PATH)
    print(f"  T1 entries: {len(tissue_data)}")

    # Match by (chr, start, end) locus
    dominant_tissues = []
    tissue_classifications = []
    n_tissues_edited = []
    max_rates = []
    tissue_breadth = []  # number of tissues with editing rate > threshold

    BREADTH_THRESHOLD = 1.0  # percent editing rate threshold

    for _, row in meta_df.iterrows():
        locus = (row["chr"], int(row["start"]), int(row["end"]))
        td = tissue_data.get(locus)
        if td is not None:
            dominant_tissues.append(get_dominant_tissue_group(td["tissue_rates"]))
            tissue_classifications.append(td["tissue_classification"])
            n_tissues_edited.append(td["n_tissues"])
            max_rates.append(td["max_rate"])
            # Count tissues above threshold
            n_above = sum(1 for r in td["tissue_rates"].values() if r >= BREADTH_THRESHOLD)
            tissue_breadth.append(n_above)
        else:
            dominant_tissues.append("Other")
            tissue_classifications.append("Unknown")
            n_tissues_edited.append(0)
            max_rates.append(0.0)
            tissue_breadth.append(0)

    meta_df["dominant_tissue"] = dominant_tissues
    meta_df["tissue_classification"] = tissue_classifications
    meta_df["n_tissues_edited"] = n_tissues_edited
    meta_df["max_rate"] = max_rates
    meta_df["tissue_breadth"] = tissue_breadth

    print(f"  Tissue classification distribution:")
    for tc, count in meta_df["tissue_classification"].value_counts().items():
        print(f"    {tc}: {count}")

    # ------------------------------------------------------------------
    # 4. Load cancer/survival data from Excel T5
    # ------------------------------------------------------------------
    print("Loading TCGA survival data from Excel T5...")
    survival_data = load_t5_survival_data(EXCEL_PATH)
    print(f"  T5 entries: {len(survival_data)}")

    has_survival = []
    n_cancers = []
    cancer_types = []
    for _, row in meta_df.iterrows():
        locus = (row["chr"], int(row["start"]), int(row["end"]))
        sd = survival_data.get(locus)
        if sd is not None and sd["n_cancers"] > 0:
            has_survival.append(True)
            n_cancers.append(sd["n_cancers"])
            cancer_types.append(sd["cancers"])
        else:
            has_survival.append(False)
            n_cancers.append(0)
            cancer_types.append("")

    meta_df["has_survival_assoc"] = has_survival
    meta_df["n_cancers"] = n_cancers
    meta_df["cancer_types"] = cancer_types

    n_with_surv = sum(has_survival)
    print(f"  Sites with survival association: {n_with_surv} / {len(meta_df)}")

    # ------------------------------------------------------------------
    # 5. Dimensionality reduction: PCA + t-SNE
    # ------------------------------------------------------------------
    print("Running PCA...")
    pca = PCA(n_components=min(50, diff_matrix.shape[0], diff_matrix.shape[1]))
    pca_all = pca.fit_transform(diff_matrix)
    pca_2d = pca_all[:, :2]
    var_explained = pca.explained_variance_ratio_[:2]
    print(f"  PCA variance explained: PC1={var_explained[0]:.1%}, PC2={var_explained[1]:.1%}")

    print("Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000, n_jobs=1)
    tsne_2d = tsne.fit_transform(pca_all[:, :30])
    print("  t-SNE complete.")

    # ------------------------------------------------------------------
    # 6. Generate multi-panel figure
    # ------------------------------------------------------------------
    print("Generating visualization...")

    fig, axes = plt.subplots(2, 4, figsize=(24, 11))

    # ---- Row 1: PCA panels ----
    pca_x, pca_y = pca_2d[:, 0], pca_2d[:, 1]

    # Panel 1: PCA colored by tissue classification (from Levanon paper)
    ax = axes[0, 0]
    for tc_label, color in TISSUE_CLASSIFICATION_COLORS.items():
        mask = meta_df["tissue_classification"].values == tc_label
        if mask.sum() > 0:
            ax.scatter(pca_x[mask], pca_y[mask], s=12, alpha=0.6, c=color,
                       label=tc_label, edgecolors="none")
    ax.legend(fontsize=7, markerscale=2, loc="best", framealpha=0.8)
    ax.set_xlabel(f"PC1 ({var_explained[0]:.1%})")
    ax.set_ylabel(f"PC2 ({var_explained[1]:.1%})")
    ax.set_title("PCA -- Tissue Classification")

    # Panel 2: PCA colored by tissue breadth (number of tissues with editing > threshold)
    ax = axes[0, 1]
    breadth_vals = meta_df["tissue_breadth"].values.astype(float)
    sc = ax.scatter(pca_x, pca_y, s=12, alpha=0.6, c=breadth_vals,
                    cmap="YlOrRd", edgecolors="none")
    plt.colorbar(sc, ax=ax, shrink=0.8, label=f"# Tissues (rate >= {BREADTH_THRESHOLD}%)")
    ax.set_xlabel(f"PC1 ({var_explained[0]:.1%})")
    ax.set_ylabel(f"PC2 ({var_explained[1]:.1%})")
    ax.set_title("PCA -- Tissue Breadth")

    # Panel 3: PCA colored by max editing rate across tissues
    ax = axes[0, 2]
    rates = meta_df["max_rate"].values.copy()
    valid_rate = rates > 0
    log_rates = np.full_like(rates, np.nan)
    log_rates[valid_rate] = np.log10(np.clip(rates[valid_rate], 0.01, None))
    sc = ax.scatter(pca_x[valid_rate], pca_y[valid_rate], s=12, alpha=0.6,
                    c=log_rates[valid_rate], cmap="viridis", edgecolors="none")
    plt.colorbar(sc, ax=ax, shrink=0.8, label="log10(max editing rate %)")
    if (~valid_rate).sum() > 0:
        ax.scatter(pca_x[~valid_rate], pca_y[~valid_rate], s=6, alpha=0.2,
                   c="gray", edgecolors="none", label="No rate")
    ax.set_xlabel(f"PC1 ({var_explained[0]:.1%})")
    ax.set_ylabel(f"PC2 ({var_explained[1]:.1%})")
    ax.set_title("PCA -- Max Editing Rate")

    # Panel 4: PCA colored by cancer/survival association
    ax = axes[0, 3]
    surv_mask = np.array(meta_df["has_survival_assoc"].values, dtype=bool)
    ax.scatter(pca_x[~surv_mask], pca_y[~surv_mask], s=10, alpha=0.3,
               c="#d1d5db", label="No survival assoc.", edgecolors="none")
    if surv_mask.sum() > 0:
        n_canc = meta_df["n_cancers"].values[surv_mask].astype(float)
        sc = ax.scatter(pca_x[surv_mask], pca_y[surv_mask], s=18, alpha=0.7,
                        c=n_canc, cmap="Reds", edgecolors="black", linewidths=0.3)
        plt.colorbar(sc, ax=ax, shrink=0.8, label="# Cancer types")
    ax.legend(fontsize=8, markerscale=2, loc="best", framealpha=0.8)
    ax.set_xlabel(f"PC1 ({var_explained[0]:.1%})")
    ax.set_ylabel(f"PC2 ({var_explained[1]:.1%})")
    ax.set_title("PCA -- TCGA Survival Association")

    # ---- Row 2: t-SNE panels (same colorings) ----
    tsne_x, tsne_y = tsne_2d[:, 0], tsne_2d[:, 1]

    # Panel 5: t-SNE colored by tissue classification
    ax = axes[1, 0]
    for tc_label, color in TISSUE_CLASSIFICATION_COLORS.items():
        mask = meta_df["tissue_classification"].values == tc_label
        if mask.sum() > 0:
            ax.scatter(tsne_x[mask], tsne_y[mask], s=12, alpha=0.6, c=color,
                       label=tc_label, edgecolors="none")
    ax.legend(fontsize=7, markerscale=2, loc="best", framealpha=0.8)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("t-SNE -- Tissue Classification")

    # Panel 6: t-SNE colored by tissue breadth
    ax = axes[1, 1]
    sc = ax.scatter(tsne_x, tsne_y, s=12, alpha=0.6, c=breadth_vals,
                    cmap="YlOrRd", edgecolors="none")
    plt.colorbar(sc, ax=ax, shrink=0.8, label=f"# Tissues (rate >= {BREADTH_THRESHOLD}%)")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("t-SNE -- Tissue Breadth")

    # Panel 7: t-SNE colored by max editing rate
    ax = axes[1, 2]
    sc = ax.scatter(tsne_x[valid_rate], tsne_y[valid_rate], s=12, alpha=0.6,
                    c=log_rates[valid_rate], cmap="viridis", edgecolors="none")
    plt.colorbar(sc, ax=ax, shrink=0.8, label="log10(max editing rate %)")
    if (~valid_rate).sum() > 0:
        ax.scatter(tsne_x[~valid_rate], tsne_y[~valid_rate], s=6, alpha=0.2,
                   c="gray", edgecolors="none")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("t-SNE -- Max Editing Rate")

    # Panel 8: t-SNE colored by cancer/survival association
    ax = axes[1, 3]
    ax.scatter(tsne_x[~surv_mask], tsne_y[~surv_mask], s=10, alpha=0.3,
               c="#d1d5db", label="No survival assoc.", edgecolors="none")
    if surv_mask.sum() > 0:
        sc = ax.scatter(tsne_x[surv_mask], tsne_y[surv_mask], s=18, alpha=0.7,
                        c=n_canc, cmap="Reds", edgecolors="black", linewidths=0.3)
        plt.colorbar(sc, ax=ax, shrink=0.8, label="# Cancer types")
    ax.legend(fontsize=8, markerscale=2, loc="best", framealpha=0.8)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("t-SNE -- TCGA Survival Association")

    # ------------------------------------------------------------------
    # Final layout
    # ------------------------------------------------------------------
    plt.suptitle(
        "Levanon C-to-U Editing Sites: Edit Effect Embedding Space\n"
        "(RNA-FM pooled embeddings, edit = edited - original)",
        fontsize=14, y=1.02,
    )
    plt.tight_layout()

    output_path = OUTPUT_DIR / "levanon_tissue_viz.png"
    fig.savefig(output_path)
    plt.close(fig)
    print(f"\nSaved: {output_path}")

    # ------------------------------------------------------------------
    # Bonus: Dominant tissue group PCA plot (standalone)
    # ------------------------------------------------------------------
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 7))
    for tissue_group in ["Whole Blood", "Testis", "Brain", "Liver", "Lung",
                         "Heart", "Kidney", "Spleen", "Artery", "Other"]:
        mask = meta_df["dominant_tissue"].values == tissue_group
        if mask.sum() > 0:
            ax2.scatter(pca_x[mask], pca_y[mask], s=15, alpha=0.6,
                        c=DOMINANT_TISSUE_COLORS[tissue_group],
                        label=f"{tissue_group} ({mask.sum()})", edgecolors="none")
    ax2.legend(fontsize=8, markerscale=2, loc="best", framealpha=0.9, ncol=2)
    ax2.set_xlabel(f"PC1 ({var_explained[0]:.1%})")
    ax2.set_ylabel(f"PC2 ({var_explained[1]:.1%})")
    ax2.set_title("Edit Effect PCA -- Dominant Tissue Group\n(tissue with highest editing rate per site)")
    plt.tight_layout()
    output_path2 = OUTPUT_DIR / "levanon_dominant_tissue_pca.png"
    fig2.savefig(output_path2)
    plt.close(fig2)
    print(f"Saved: {output_path2}")

    print("\nDone.")


if __name__ == "__main__":
    main()
