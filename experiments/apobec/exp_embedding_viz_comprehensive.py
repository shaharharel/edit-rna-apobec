#!/usr/bin/env python
"""Comprehensive embedding visualization with 12 coloring schemes.

Loads pre-computed 512-dim contextual edit embeddings from trained EditRNA-A3A
and generates UMAP visualizations colored by many semantically interesting
properties to uncover structure in the latent space.

Usage:
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    KMP_DUPLICATE_LIB_OK=TRUE \
    python experiments/apobec/exp_embedding_viz_comprehensive.py
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import logging
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EMB_PATH = (
    PROJECT_ROOT
    / "experiments/apobec/outputs/embedding_trained/contextual_edit_embeddings.pt"
)
SPLITS_CSV = PROJECT_ROOT / "data/processed/splits_expanded_a3a.csv"
SEQUENCES_JSON = PROJECT_ROOT / "data/processed/site_sequences.json"
STRUCT_CACHE = PROJECT_ROOT / "data/processed/embeddings/vienna_structure_cache.npz"
OUTPUT_DIR = (
    PROJECT_ROOT / "experiments/apobec/outputs/embedding_viz_comprehensive"
)

MAX_SITES = 3000

DATASET_LABELS = {
    "advisor_c2t": "Advisor",
    "asaoka_2019": "Asaoka",
    "sharma_2015": "Sharma",
    "alqassim_2021": "Alqassim",
    "tier2_negative": "Tier2 Neg",
    "tier3_negative": "Tier3 Neg",
}

DATASET_COLORS = {
    "Advisor": "#2563eb",
    "Asaoka": "#16a34a",
    "Sharma": "#dc2626",
    "Alqassim": "#d97706",
    "Tier2 Neg": "#6b7280",
    "Tier3 Neg": "#374151",
}

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
    }
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_all_data():
    """Load embeddings, metadata, sequences, and structure cache."""

    # 1. Embeddings
    logger.info("Loading embeddings from %s", EMB_PATH)
    emb_data = torch.load(EMB_PATH, weights_only=False)
    site_ids_emb = list(emb_data["site_ids"])
    fused_matrix = emb_data["fused"]
    if isinstance(fused_matrix, torch.Tensor):
        fused_matrix = fused_matrix.numpy()
    logger.info(
        "Loaded %d embeddings, dim=%d", len(site_ids_emb), fused_matrix.shape[1]
    )

    # 2. Splits CSV
    logger.info("Loading splits from %s", SPLITS_CSV)
    splits_df = pd.read_csv(SPLITS_CSV)
    emb_set = set(site_ids_emb)
    splits_df = splits_df[splits_df["site_id"].isin(emb_set)].copy()
    logger.info("Sites in both embeddings and splits: %d", len(splits_df))

    # Build a site_id -> row index mapping in the embedding arrays
    emb_id_to_idx = {sid: i for i, sid in enumerate(site_ids_emb)}

    # Reorder splits_df to match embedding order and drop sites not in splits
    splits_df["_emb_idx"] = splits_df["site_id"].map(emb_id_to_idx)
    splits_df = splits_df.dropna(subset=["_emb_idx"]).copy()
    splits_df["_emb_idx"] = splits_df["_emb_idx"].astype(int)
    splits_df = splits_df.sort_values("_emb_idx").reset_index(drop=True)

    # Extract the fused matrix rows that match
    emb_indices = splits_df["_emb_idx"].values
    fused_aligned = fused_matrix[emb_indices]
    site_ids_aligned = splits_df["site_id"].tolist()
    logger.info("Aligned matrix shape: %s", fused_aligned.shape)

    # 3. Sequences
    sequences = {}
    if SEQUENCES_JSON.exists():
        logger.info("Loading sequences from %s", SEQUENCES_JSON)
        with open(SEQUENCES_JSON) as f:
            sequences = json.load(f)
        logger.info("Loaded %d sequences", len(sequences))
    else:
        logger.warning("Sequences file not found: %s", SEQUENCES_JSON)

    # 4. Structure cache
    struct_data = {}
    if STRUCT_CACHE.exists():
        logger.info("Loading structure cache from %s", STRUCT_CACHE)
        npz = np.load(STRUCT_CACHE, allow_pickle=True)
        struct_sids = [str(s) for s in npz["site_ids"]]
        struct_pairing = npz["pairing_probs"]  # (N, 201)
        struct_delta = npz["delta_features"]  # (N, 7)
        struct_mfes = npz["mfes"]  # (N,)
        struct_mfes_edited = npz["mfes_edited"]  # (N,)
        struct_data = {
            "site_ids": struct_sids,
            "pairing_probs": struct_pairing,
            "delta_features": struct_delta,
            "mfes": struct_mfes,
            "mfes_edited": struct_mfes_edited,
            "id_to_idx": {sid: i for i, sid in enumerate(struct_sids)},
        }
        logger.info("Structure cache: %d sites", len(struct_sids))
    else:
        logger.warning("Structure cache not found: %s", STRUCT_CACHE)

    return fused_aligned, splits_df, site_ids_aligned, sequences, struct_data


# ---------------------------------------------------------------------------
# Coloring scheme builders
# ---------------------------------------------------------------------------


def compute_loop_size(pairing_probs_row, center=100, threshold=0.3):
    """Compute loop size by counting consecutive unpaired positions around center."""
    if pairing_probs_row[center] >= threshold:
        return 0  # Center is paired
    # Expand left
    left = center - 1
    while left >= 0 and pairing_probs_row[left] < threshold:
        left -= 1
    # Expand right
    right = center + 1
    while right < len(pairing_probs_row) and pairing_probs_row[right] < threshold:
        right += 1
    return right - left - 1


def build_coloring_schemes(splits_df, site_ids, sequences, struct_data):
    """Build all 12 coloring scheme arrays aligned with site_ids."""
    n = len(site_ids)
    schemes = {}

    # Helper: struct lookup
    struct_idx = struct_data.get("id_to_idx", {})

    # 1. Label (is_edited)
    label_map = dict(zip(splits_df["site_id"], splits_df["is_edited"]))
    labels = [
        "Positive" if label_map.get(sid, 0) == 1 else "Negative" for sid in site_ids
    ]
    schemes["label"] = {
        "values": labels,
        "type": "categorical",
        "categories": ["Positive", "Negative"],
        "colors": {"Positive": "#2563eb", "Negative": "#dc2626"},
        "title": "Editing Label",
    }

    # 2. Dataset source
    ds_map = dict(zip(splits_df["site_id"], splits_df["dataset_source"]))
    ds_labels = [DATASET_LABELS.get(ds_map.get(sid, ""), "Unknown") for sid in site_ids]
    schemes["dataset"] = {
        "values": ds_labels,
        "type": "categorical",
        "categories": list(DATASET_COLORS.keys()),
        "colors": DATASET_COLORS,
        "title": "Dataset Source",
    }

    # 3. Editing rate (continuous, log10 scale)
    rate_map = dict(
        zip(
            splits_df["site_id"],
            pd.to_numeric(splits_df["editing_rate_normalized"], errors="coerce"),
        )
    )
    rates = np.array([rate_map.get(sid, np.nan) for sid in site_ids])
    schemes["editing_rate"] = {
        "values": rates,
        "type": "continuous",
        "title": "Editing Rate (log10)",
        "cmap": "viridis",
        "colorbar_label": "log10(rate)",
    }

    # 4. Rate z-score (per-dataset z-score of log2(rate + 0.01))
    rate_zscore = np.full(n, np.nan)
    edited_mask_map = dict(zip(splits_df["site_id"], splits_df["is_edited"]))
    sid_to_idx = {sid: i for i, sid in enumerate(site_ids)}
    for ds_name in splits_df["dataset_source"].unique():
        ds_rows = splits_df[
            (splits_df["dataset_source"] == ds_name)
            & (splits_df["is_edited"] == 1)
        ]
        ds_rates = pd.to_numeric(
            ds_rows["editing_rate_normalized"], errors="coerce"
        )
        valid = ds_rates.notna() & (ds_rates > 0)
        if valid.sum() < 2:
            continue
        log_rates = np.log2(ds_rates[valid].values + 0.01)
        mu, sigma = log_rates.mean(), log_rates.std()
        if sigma < 1e-10:
            continue
        for sid, rate_val in zip(
            ds_rows.loc[valid.index[valid], "site_id"],
            ds_rates[valid],
        ):
            idx_in_list = sid_to_idx.get(sid)
            if idx_in_list is not None:
                rate_zscore[idx_in_list] = (np.log2(rate_val + 0.01) - mu) / sigma

    schemes["rate_zscore"] = {
        "values": rate_zscore,
        "type": "continuous",
        "title": "Editing Rate Z-score",
        "cmap": "RdBu_r",
        "colorbar_label": "Z-score",
        "center_zero": True,
    }

    # 5. Trinucleotide motif (positions 99-100 in 0-indexed 201-nt sequence)
    trinuc = []
    for sid in site_ids:
        seq = sequences.get(sid, "")
        if len(seq) >= 101:
            dinuc = seq[99:101].upper().replace("U", "T")
            if dinuc == "TC":
                trinuc.append("TC")
            elif dinuc == "CC":
                trinuc.append("CC")
            elif dinuc == "AC":
                trinuc.append("AC")
            elif dinuc == "GC":
                trinuc.append("GC")
            else:
                trinuc.append("Other")
        else:
            trinuc.append("Other")

    motif_cats = ["TC", "CC", "AC", "GC", "Other"]
    motif_colors = {
        "TC": "#dc2626",
        "CC": "#2563eb",
        "AC": "#16a34a",
        "GC": "#d97706",
        "Other": "#9ca3af",
    }
    schemes["trinuc_motif"] = {
        "values": trinuc,
        "type": "categorical",
        "categories": motif_cats,
        "colors": motif_colors,
        "title": "Upstream Dinucleotide Motif",
    }

    # 6. Genomic feature (group rare into Other)
    feat_map = dict(
        zip(splits_df["site_id"], splits_df["feature"].fillna("Unknown"))
    )
    keep_features = {"exonic", "UTR3", "UTR5", "intronic"}
    # Also keep sub-exonic if present
    raw_feats = [feat_map.get(sid, "Unknown") for sid in site_ids]
    grouped_feats = []
    for f in raw_feats:
        if f in keep_features:
            grouped_feats.append(f)
        elif f in ("synonymous", "nonsynonymous", "stopgain"):
            grouped_feats.append("exonic")
        else:
            grouped_feats.append("Other")

    feat_cats = ["exonic", "UTR3", "UTR5", "intronic", "Other"]
    feat_colors = {
        "exonic": "#2563eb",
        "UTR3": "#0891b2",
        "UTR5": "#be185d",
        "intronic": "#16a34a",
        "Other": "#9ca3af",
    }
    schemes["genomic_feature"] = {
        "values": grouped_feats,
        "type": "categorical",
        "categories": feat_cats,
        "colors": feat_colors,
        "title": "Genomic Feature",
    }

    # 7. Delta MFE (from delta_features[:, 3])
    delta_mfe_vals = np.full(n, np.nan)
    for i, sid in enumerate(site_ids):
        if sid in struct_idx:
            sidx = struct_idx[sid]
            delta_mfe_vals[i] = struct_data["delta_features"][sidx, 3]
    schemes["delta_mfe"] = {
        "values": delta_mfe_vals,
        "type": "continuous",
        "title": "Delta MFE (edited - original)",
        "cmap": "coolwarm",
        "colorbar_label": "delta MFE (kcal/mol)",
        "center_zero": True,
    }

    # 8. Pairing probability at center
    pairing_center = np.full(n, np.nan)
    for i, sid in enumerate(site_ids):
        if sid in struct_idx:
            sidx = struct_idx[sid]
            pairing_center[i] = struct_data["pairing_probs"][sidx, 100]
    schemes["pairing_prob_center"] = {
        "values": pairing_center,
        "type": "continuous",
        "title": "Pairing Prob. at Edit Site",
        "cmap": "viridis",
        "colorbar_label": "P(paired)",
    }

    # 9. Loop size
    loop_sizes = np.full(n, np.nan)
    for i, sid in enumerate(site_ids):
        if sid in struct_idx:
            sidx = struct_idx[sid]
            loop_sizes[i] = compute_loop_size(struct_data["pairing_probs"][sidx])
    schemes["loop_size"] = {
        "values": loop_sizes,
        "type": "continuous",
        "title": "Loop Size (unpaired stretch)",
        "cmap": "viridis",
        "colorbar_label": "Loop size (nt)",
    }

    # 10. Upstream 5nt (positions 95-99)
    upstream_5nt = []
    for sid in site_ids:
        seq = sequences.get(sid, "")
        if len(seq) >= 100:
            upstream_5nt.append(seq[95:100].upper())
        else:
            upstream_5nt.append("?????")
    # Group into top-8 most common + Other
    counts = Counter(upstream_5nt)
    top8 = [motif for motif, _ in counts.most_common(8)]
    upstream_grouped = [m if m in top8 else "Other" for m in upstream_5nt]
    upstream_cats = top8 + ["Other"]

    # Generate distinct colors for upstream motifs
    cmap9 = plt.cm.get_cmap("tab10", 9)
    upstream_colors = {
        cat: mcolors.to_hex(cmap9(i)) for i, cat in enumerate(upstream_cats)
    }
    schemes["upstream_5nt"] = {
        "values": upstream_grouped,
        "type": "categorical",
        "categories": upstream_cats,
        "colors": upstream_colors,
        "title": "5nt Upstream Motif",
    }

    # 11. Chromosome grouping
    chr_map = dict(zip(splits_df["site_id"], splits_df["chr"]))

    def chr_group(c):
        if not isinstance(c, str):
            return "Other"
        c = c.replace("chr", "")
        if c == "X":
            return "chrX"
        try:
            num = int(c)
        except ValueError:
            return "Other"
        if 1 <= num <= 5:
            return "chr1-5"
        elif 6 <= num <= 10:
            return "chr6-10"
        elif 11 <= num <= 15:
            return "chr11-15"
        elif 16 <= num <= 22:
            return "chr16-22"
        return "Other"

    chr_groups = [chr_group(chr_map.get(sid, "")) for sid in site_ids]
    chr_cats = ["chr1-5", "chr6-10", "chr11-15", "chr16-22", "chrX", "Other"]
    chr_colors = {
        "chr1-5": "#2563eb",
        "chr6-10": "#16a34a",
        "chr11-15": "#d97706",
        "chr16-22": "#7c3aed",
        "chrX": "#dc2626",
        "Other": "#9ca3af",
    }
    schemes["chromosome"] = {
        "values": chr_groups,
        "type": "categorical",
        "categories": chr_cats,
        "colors": chr_colors,
        "title": "Chromosome Group",
    }

    # 12. Negative tier
    neg_tier = []
    for sid in site_ids:
        ds = ds_map.get(sid, "")
        edited = edited_mask_map.get(sid, 0)
        if edited == 1:
            neg_tier.append("Positive")
        elif ds == "tier2_negative":
            neg_tier.append("Tier2 Neg")
        elif ds == "tier3_negative":
            neg_tier.append("Tier3 Neg")
        else:
            neg_tier.append("Positive")

    schemes["negative_tier"] = {
        "values": neg_tier,
        "type": "categorical",
        "categories": ["Positive", "Tier2 Neg", "Tier3 Neg"],
        "colors": {
            "Positive": "lightgray",
            "Tier2 Neg": "#2563eb",
            "Tier3 Neg": "#dc2626",
        },
        "title": "Negative Tier",
        "background_category": "Positive",
    }

    return schemes


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_embedding(
    coords,
    values,
    title,
    output_path,
    cmap=None,
    categorical=False,
    categories=None,
    colors=None,
    alpha=0.4,
    s=8,
    show_legend=True,
    background_mask=None,
    background_color="lightgray",
    colorbar_label=None,
    center_zero=False,
):
    """Generic embedding scatter plot.

    Parameters
    ----------
    coords : ndarray (N, 2)
    values : list or ndarray
    title : str
    output_path : Path
    cmap : str, optional  (for continuous)
    categorical : bool
    categories : list of str (ordering for legend)
    colors : dict  category -> color hex
    alpha : float
    s : float  marker size
    show_legend : bool
    background_mask : ndarray bool  (True = background point)
    background_color : str
    colorbar_label : str
    center_zero : bool  (center colormap at 0 for diverging)
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    if categorical:
        # If there is a background category, plot those first
        if background_mask is not None:
            bg_idx = np.where(background_mask)[0]
            if len(bg_idx) > 0:
                ax.scatter(
                    coords[bg_idx, 0],
                    coords[bg_idx, 1],
                    c=background_color,
                    s=s,
                    alpha=0.2,
                    label=None,
                    edgecolors="none",
                    rasterized=True,
                )

        for cat in categories:
            mask = np.array([v == cat for v in values])
            if background_mask is not None:
                mask = mask & ~background_mask
            idx = np.where(mask)[0]
            if len(idx) == 0:
                continue
            color = colors.get(cat, "#9ca3af") if colors else None
            ax.scatter(
                coords[idx, 0],
                coords[idx, 1],
                c=color,
                s=s,
                alpha=alpha,
                label=f"{cat} ({len(idx)})",
                edgecolors="none",
                rasterized=True,
            )
        if background_mask is not None:
            bg_count = int(background_mask.sum())
            bg_cat = "Background"
            # Find background category name from values
            bg_vals = set(np.array(values)[background_mask])
            if len(bg_vals) == 1:
                bg_cat = list(bg_vals)[0]
            ax.scatter(
                [],
                [],
                c=background_color,
                s=s,
                alpha=0.3,
                label=f"{bg_cat} ({bg_count})",
                edgecolors="none",
            )
        if show_legend:
            ax.legend(
                loc="best",
                fontsize=8,
                framealpha=0.8,
                markerscale=2,
                handletextpad=0.3,
            )
    else:
        # Continuous
        vals = np.asarray(values, dtype=float)
        valid_mask = np.isfinite(vals)
        # Plot missing as gray background
        missing_idx = np.where(~valid_mask)[0]
        if len(missing_idx) > 0:
            ax.scatter(
                coords[missing_idx, 0],
                coords[missing_idx, 1],
                c="lightgray",
                s=s * 0.5,
                alpha=0.15,
                edgecolors="none",
                rasterized=True,
            )

        valid_idx = np.where(valid_mask)[0]
        if len(valid_idx) == 0:
            logger.warning("No valid values for %s, skipping", title)
            plt.close(fig)
            return

        plot_vals = vals[valid_idx]

        # For editing rate, use log10 scale
        if colorbar_label and "log10" in colorbar_label:
            with np.errstate(divide="ignore", invalid="ignore"):
                plot_vals = np.log10(np.clip(plot_vals, 1e-6, None))
            plot_vals = np.nan_to_num(plot_vals, nan=-6.0)

        vmin, vmax = np.nanpercentile(plot_vals, [2, 98])
        if center_zero:
            abs_max = max(abs(vmin), abs(vmax))
            vmin, vmax = -abs_max, abs_max

        sc = ax.scatter(
            coords[valid_idx, 0],
            coords[valid_idx, 1],
            c=plot_vals,
            cmap=cmap or "viridis",
            s=s,
            alpha=alpha,
            edgecolors="none",
            vmin=vmin,
            vmax=vmax,
            rasterized=True,
        )
        cbar = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
        if colorbar_label:
            cbar.set_label(colorbar_label, fontsize=10)

        # Annotate missing count
        n_missing = int((~valid_mask).sum())
        if n_missing > 0:
            ax.text(
                0.02,
                0.02,
                f"{n_missing} missing (gray)",
                transform=ax.transAxes,
                fontsize=8,
                color="gray",
            )

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", output_path)


# ---------------------------------------------------------------------------
# Separation metrics
# ---------------------------------------------------------------------------


def compute_separation_metric(coords, values, scheme_type):
    """Compute a separation metric for a coloring scheme.

    For categorical: silhouette score (subsample to 5000).
    For continuous: max |Spearman| between values and each UMAP axis.
    """
    if scheme_type == "categorical":
        labels_arr = np.array(values)
        unique_labels = np.unique(labels_arr)
        if len(unique_labels) < 2:
            return np.nan
        # Subsample
        n = len(labels_arr)
        if n > 5000:
            rng = np.random.RandomState(42)
            idx = rng.choice(n, 5000, replace=False)
            coords_sub = coords[idx]
            labels_sub = labels_arr[idx]
        else:
            coords_sub = coords
            labels_sub = labels_arr
        # Encode labels as integers
        label_to_int = {lab: i for i, lab in enumerate(unique_labels)}
        int_labels = np.array([label_to_int[l] for l in labels_sub])
        try:
            return float(silhouette_score(coords_sub, int_labels))
        except Exception:
            return np.nan
    else:
        # Continuous
        vals = np.asarray(values, dtype=float)
        valid = np.isfinite(vals)
        if valid.sum() < 10:
            return np.nan
        rho1, _ = spearmanr(coords[valid, 0], vals[valid])
        rho2, _ = spearmanr(coords[valid, 1], vals[valid])
        return float(max(abs(rho1), abs(rho2)))


# ---------------------------------------------------------------------------
# Summary grid
# ---------------------------------------------------------------------------


def create_summary_grid(umap_coords, schemes, subsample_idx, output_path):
    """Create a 3x4 grid of UMAP mini-plots for all 12 schemes."""
    scheme_names = list(schemes.keys())
    assert len(scheme_names) == 12, f"Expected 12 schemes, got {len(scheme_names)}"

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes_flat = axes.flatten()

    for ax_idx, name in enumerate(scheme_names):
        ax = axes_flat[ax_idx]
        scheme = schemes[name]
        vals = scheme["values"]

        if isinstance(vals, np.ndarray):
            vals_sub = vals[subsample_idx]
        else:
            vals_sub = [vals[i] for i in subsample_idx]

        if scheme["type"] == "categorical":
            categories = scheme.get("categories", sorted(set(vals_sub)))
            colors = scheme.get("colors", {})
            bg_cat = scheme.get("background_category", None)

            # Plot background category first
            if bg_cat is not None:
                bg_mask = np.array([v == bg_cat for v in vals_sub])
                bg_idx = np.where(bg_mask)[0]
                if len(bg_idx) > 0:
                    ax.scatter(
                        umap_coords[bg_idx, 0],
                        umap_coords[bg_idx, 1],
                        c="lightgray",
                        s=1,
                        alpha=0.15,
                        edgecolors="none",
                        rasterized=True,
                    )

            for cat in categories:
                mask = np.array([v == cat for v in vals_sub])
                if bg_cat is not None and cat == bg_cat:
                    continue
                idx = np.where(mask)[0]
                if len(idx) == 0:
                    continue
                color = colors.get(cat, "#9ca3af")
                ax.scatter(
                    umap_coords[idx, 0],
                    umap_coords[idx, 1],
                    c=color,
                    s=1,
                    alpha=0.3,
                    edgecolors="none",
                    rasterized=True,
                )
        else:
            float_vals = np.asarray(vals_sub, dtype=float)
            valid = np.isfinite(float_vals)
            missing_idx = np.where(~valid)[0]
            if len(missing_idx) > 0:
                ax.scatter(
                    umap_coords[missing_idx, 0],
                    umap_coords[missing_idx, 1],
                    c="lightgray",
                    s=0.5,
                    alpha=0.1,
                    edgecolors="none",
                    rasterized=True,
                )
            valid_idx = np.where(valid)[0]
            if len(valid_idx) > 0:
                plot_vals = float_vals[valid_idx]
                cmap_name = scheme.get("cmap", "viridis")
                center = scheme.get("center_zero", False)

                if scheme.get("colorbar_label", "") and "log10" in scheme.get(
                    "colorbar_label", ""
                ):
                    with np.errstate(divide="ignore", invalid="ignore"):
                        plot_vals = np.log10(np.clip(plot_vals, 1e-6, None))
                    plot_vals = np.nan_to_num(plot_vals, nan=-6.0)

                vmin, vmax = np.nanpercentile(plot_vals, [2, 98])
                if center:
                    abs_max = max(abs(vmin), abs(vmax))
                    vmin, vmax = -abs_max, abs_max

                ax.scatter(
                    umap_coords[valid_idx, 0],
                    umap_coords[valid_idx, 1],
                    c=plot_vals,
                    cmap=cmap_name,
                    s=1,
                    alpha=0.3,
                    edgecolors="none",
                    vmin=vmin,
                    vmax=vmax,
                    rasterized=True,
                )

        ax.set_title(scheme["title"], fontsize=9, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Embedding Space: 12 Coloring Schemes (UMAP)", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved summary grid: %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Load data ----
    fused_matrix, splits_df, site_ids, sequences, struct_data = load_all_data()
    n_total = len(site_ids)
    logger.info("Total sites for visualization: %d", n_total)

    # ---- Build coloring schemes ----
    logger.info("Building 12 coloring schemes...")
    schemes = build_coloring_schemes(splits_df, site_ids, sequences, struct_data)
    logger.info("Built %d coloring schemes", len(schemes))

    # ---- Dimensionality reduction ----
    logger.info("Running PCA (n_components=50)...")
    pca = PCA(n_components=50)
    pca_50 = pca.fit_transform(fused_matrix)
    logger.info(
        "PCA variance explained (top 10): %s",
        [f"{v:.3f}" for v in pca.explained_variance_ratio_[:10]],
    )

    # Subsample
    rng = np.random.RandomState(42)
    if n_total > MAX_SITES:
        subsample_idx = np.sort(rng.choice(n_total, MAX_SITES, replace=False))
    else:
        subsample_idx = np.arange(n_total)
    logger.info("Subsampled %d sites for UMAP/t-SNE", len(subsample_idx))

    pca_sub = pca_50[subsample_idx, :30]

    # UMAP
    logger.info("Running UMAP (n_neighbors=30, min_dist=0.3)...")
    import umap

    reducer = umap.UMAP(
        n_components=2, n_neighbors=30, min_dist=0.3, random_state=42
    )
    umap_coords = reducer.fit_transform(pca_sub)
    logger.info("UMAP complete, shape: %s", umap_coords.shape)

    # t-SNE
    logger.info("Running t-SNE (perplexity=30, max_iter=1000)...")
    tsne = TSNE(
        n_components=2, perplexity=30, max_iter=1000, random_state=42, n_jobs=1
    )
    tsne_coords = tsne.fit_transform(pca_sub)
    logger.info("t-SNE complete, shape: %s", tsne_coords.shape)

    # Save coordinates
    np.savez(
        OUTPUT_DIR / "coordinates.npz",
        umap=umap_coords,
        tsne=tsne_coords,
        subsample_idx=subsample_idx,
        site_ids=np.array([site_ids[i] for i in subsample_idx]),
    )
    logger.info("Saved coordinates to %s", OUTPUT_DIR / "coordinates.npz")

    # ---- Generate plots ----
    results = {}
    tsne_schemes = {"label", "dataset"}  # Only generate t-SNE for these two

    for name, scheme in schemes.items():
        logger.info("Plotting scheme: %s (%s)", name, scheme["type"])

        # Get subsampled values
        vals = scheme["values"]
        if isinstance(vals, np.ndarray):
            vals_sub = vals[subsample_idx]
        else:
            vals_sub = [vals[i] for i in subsample_idx]

        is_cat = scheme["type"] == "categorical"
        bg_cat = scheme.get("background_category", None)
        bg_mask = None
        if bg_cat is not None:
            bg_mask = np.array([v == bg_cat for v in vals_sub])

        # UMAP plot
        plot_embedding(
            coords=umap_coords,
            values=vals_sub,
            title=f"UMAP: {scheme['title']}",
            output_path=OUTPUT_DIR / f"umap_{name}.png",
            cmap=scheme.get("cmap"),
            categorical=is_cat,
            categories=scheme.get("categories"),
            colors=scheme.get("colors"),
            alpha=0.4,
            s=8,
            show_legend=True,
            background_mask=bg_mask,
            background_color="lightgray",
            colorbar_label=scheme.get("colorbar_label"),
            center_zero=scheme.get("center_zero", False),
        )

        # t-SNE plot (only for label and dataset)
        if name in tsne_schemes:
            plot_embedding(
                coords=tsne_coords,
                values=vals_sub,
                title=f"t-SNE: {scheme['title']}",
                output_path=OUTPUT_DIR / f"tsne_{name}.png",
                cmap=scheme.get("cmap"),
                categorical=is_cat,
                categories=scheme.get("categories"),
                colors=scheme.get("colors"),
                alpha=0.4,
                s=8,
                show_legend=True,
                background_mask=bg_mask,
                background_color="lightgray",
                colorbar_label=scheme.get("colorbar_label"),
                center_zero=scheme.get("center_zero", False),
            )

        # Compute separation metric
        metric = compute_separation_metric(umap_coords, vals_sub, scheme["type"])
        metric_name = (
            "silhouette" if is_cat else "max_abs_spearman"
        )

        results[name] = {
            "title": scheme["title"],
            "type": scheme["type"],
            "metric_name": metric_name,
            "metric_value": float(metric) if np.isfinite(metric) else None,
        }

        # Count stats
        if is_cat:
            counts = Counter(vals_sub)
            results[name]["category_counts"] = dict(counts)
        else:
            float_vals = np.asarray(vals_sub, dtype=float)
            valid = np.isfinite(float_vals)
            results[name]["n_valid"] = int(valid.sum())
            results[name]["n_missing"] = int((~valid).sum())
            if valid.sum() > 0:
                results[name]["value_range"] = [
                    float(np.nanmin(float_vals[valid])),
                    float(np.nanmax(float_vals[valid])),
                ]

    # ---- Summary grid ----
    logger.info("Creating summary grid...")
    create_summary_grid(
        umap_coords, schemes, subsample_idx, OUTPUT_DIR / "embedding_summary_grid.png"
    )

    # ---- Save results JSON ----
    output_json = {
        "n_total_sites": n_total,
        "n_subsampled": len(subsample_idx),
        "embedding_dim": fused_matrix.shape[1],
        "pca_components": 50,
        "umap_input_dims": 30,
        "umap_params": {"n_neighbors": 30, "min_dist": 0.3, "random_state": 42},
        "tsne_params": {"perplexity": 30, "max_iter": 1000, "random_state": 42},
        "pca_variance_explained_top10": [
            float(v) for v in pca.explained_variance_ratio_[:10]
        ],
        "schemes": results,
    }
    with open(OUTPUT_DIR / "embedding_viz_comprehensive_results.json", "w") as f:
        json.dump(output_json, f, indent=2)
    logger.info(
        "Saved results to %s",
        OUTPUT_DIR / "embedding_viz_comprehensive_results.json",
    )

    # ---- Print summary table ----
    print("\n" + "=" * 75)
    print("EMBEDDING VISUALIZATION SUMMARY")
    print("=" * 75)
    print(f"{'Scheme':<22} {'Type':<14} {'Metric':<20} {'Value':>8}")
    print("-" * 75)
    for name, res in results.items():
        val_str = (
            f"{res['metric_value']:.4f}"
            if res["metric_value"] is not None
            else "N/A"
        )
        print(
            f"{name:<22} {res['type']:<14} {res['metric_name']:<20} {val_str:>8}"
        )
    print("=" * 75)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"Total plots: {len(schemes) + len(tsne_schemes) + 1} files")
    print("Done.")


if __name__ == "__main__":
    main()
