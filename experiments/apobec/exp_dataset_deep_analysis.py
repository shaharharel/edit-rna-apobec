#!/usr/bin/env python
"""Deep dataset analysis with statistics and visualizations.

Generates comprehensive per-dataset statistics, rate distributions,
gene overlap analysis, sequence context distributions, chromosome
distributions, genomic category breakdowns, site overlap UpSet-style
plots, and positive vs negative comparisons.

Usage:
    python experiments/apobec/exp_dataset_deep_analysis.py
"""

import gc
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logger = logging.getLogger(__name__)

COMBINED_CSV = PROJECT_ROOT / "data" / "processed" / "all_datasets_combined.csv"
SPLITS_CSV = PROJECT_ROOT / "data" / "processed" / "splits_expanded.csv"
SEQ_JSON = PROJECT_ROOT / "data" / "processed" / "site_sequences.json"
STRUCT_CACHE = PROJECT_ROOT / "data" / "processed" / "embeddings" / "vienna_structure_cache.npz"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "apobec" / "outputs" / "dataset_analysis"

DATASET_LABELS = {
    "advisor_c2t": "Levanon",
    "asaoka_2019": "Asaoka",
    "sharma_2015": "Sharma",
    "alqassim_2021": "Alqassim",
    "baysal_2016": "Baysal",
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

# Canonical chromosome order
CHROM_ORDER = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]

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
    """Load all datasets."""
    combined_df = pd.read_csv(COMBINED_CSV)
    splits_df = pd.read_csv(SPLITS_CSV)

    sequences = {}
    if SEQ_JSON.exists():
        sequences = json.loads(SEQ_JSON.read_text())

    structure_delta = {}
    if STRUCT_CACHE.exists():
        data = np.load(STRUCT_CACHE, allow_pickle=True)
        feat_key = "delta_features" if "delta_features" in data else "features"
        if "site_ids" in data and feat_key in data:
            sids = data["site_ids"]
            feats = data[feat_key]
            structure_delta = {str(sid): feats[i] for i, sid in enumerate(sids)}

    return combined_df, splits_df, sequences, structure_delta


def compute_dataset_statistics(combined_df, splits_df, sequences, structure_delta):
    """Compute comprehensive per-dataset statistics."""
    results = {}

    for ds_name, ds_label in DATASET_LABELS.items():
        ds_df = combined_df[combined_df["dataset_source"] == ds_name]
        if len(ds_df) == 0:
            continue

        stats = {
            "name": ds_label,
            "n_sites": len(ds_df),
            "n_genes": ds_df["gene"].nunique(),
            "n_chromosomes": ds_df["chr"].nunique(),
        }

        # Editing rates
        rates = pd.to_numeric(ds_df["editing_rate"], errors="coerce").dropna()
        rates = rates[rates > 0]
        if len(rates) > 0:
            stats["n_with_rates"] = int(len(rates))
            stats["rate_min"] = float(rates.min())
            stats["rate_max"] = float(rates.max())
            stats["rate_mean"] = float(rates.mean())
            stats["rate_median"] = float(rates.median())
            stats["rate_std"] = float(rates.std())
            stats["rate_q25"] = float(rates.quantile(0.25))
            stats["rate_q75"] = float(rates.quantile(0.75))
        else:
            stats["n_with_rates"] = 0

        # Strand distribution
        if "strand" in ds_df.columns:
            strand_counts = ds_df["strand"].value_counts().to_dict()
            stats["strand_plus"] = strand_counts.get("+", 0)
            stats["strand_minus"] = strand_counts.get("-", 0)

        # Feature types (genomic category)
        if "feature" in ds_df.columns:
            feat_counts = ds_df["feature"].value_counts().to_dict()
            stats["feature_types"] = {str(k): int(v) for k, v in feat_counts.items()}

        # Chromosome distribution
        if "chr" in ds_df.columns:
            chr_counts = ds_df["chr"].value_counts().to_dict()
            stats["chr_distribution"] = {str(k): int(v) for k, v in chr_counts.items()}

        # TC-motif analysis from sequences
        tc_count = 0
        trinuc_counter = Counter()
        site_ids_in_ds = set(ds_df["site_id"].values)
        for sid, seq in sequences.items():
            if sid not in site_ids_in_ds:
                continue
            center = len(seq) // 2
            if center >= 1 and center < len(seq):
                trinuc = seq[max(0, center - 1):center + 2]
                if len(trinuc) == 3:
                    trinuc_counter[trinuc] += 1
                if center >= 1 and seq[center - 1] in "Uu":
                    tc_count += 1

        if len(site_ids_in_ds & set(sequences.keys())) > 0:
            n_with_seq = len(site_ids_in_ds & set(sequences.keys()))
            stats["tc_motif_fraction"] = tc_count / max(n_with_seq, 1)
            stats["top_trinucleotides"] = trinuc_counter.most_common(10)

        # Structure delta stats
        ds_deltas = []
        for sid in ds_df["site_id"]:
            if sid in structure_delta:
                ds_deltas.append(structure_delta[sid])
        if ds_deltas:
            ds_deltas = np.array(ds_deltas)
            stats["n_with_structure"] = len(ds_deltas)
            stats["delta_mfe_mean"] = float(ds_deltas[:, 3].mean())
            stats["delta_mfe_std"] = float(ds_deltas[:, 3].std())
            stats["delta_pair_mean"] = float(ds_deltas[:, 0].mean())

        # Overlap with other datasets
        ds_genes = set(ds_df["gene"].dropna())
        ds_coords = set(zip(ds_df["chr"], ds_df["start"]))
        stats["gene_set_size"] = len(ds_genes)

        results[ds_label] = stats

    # Compute gene overlaps between datasets
    gene_sets = {}
    coord_sets = {}
    for ds_name, ds_label in DATASET_LABELS.items():
        ds_df = combined_df[combined_df["dataset_source"] == ds_name]
        gene_sets[ds_label] = set(ds_df["gene"].dropna())
        coord_sets[ds_label] = set(zip(ds_df["chr"], ds_df["start"]))

    overlap_matrix = {}
    for d1 in gene_sets:
        overlap_matrix[d1] = {}
        for d2 in gene_sets:
            shared_genes = len(gene_sets[d1] & gene_sets[d2])
            shared_coords = len(coord_sets[d1] & coord_sets[d2])
            overlap_matrix[d1][d2] = {
                "shared_genes": shared_genes,
                "shared_coordinates": shared_coords,
            }

    # Negative tier stats
    for tier_name, tier_label in [("tier2_negative", "Tier2 Neg"), ("tier3_negative", "Tier3 Neg")]:
        tier_df = splits_df[splits_df["dataset_source"] == tier_name]
        if len(tier_df) > 0:
            results[tier_label] = {
                "name": tier_label,
                "n_sites": len(tier_df),
                "n_genes": tier_df["gene"].nunique() if "gene" in tier_df.columns else 0,
            }

    return results, overlap_matrix, coord_sets


def plot_rate_distributions(combined_df):
    """Plot combined editing rate overlay: raw vs log2-transformed training target."""
    ds_with_rates = []
    for ds_name, ds_label in DATASET_LABELS.items():
        ds_df = combined_df[combined_df["dataset_source"] == ds_name]
        raw = pd.to_numeric(ds_df["editing_rate"], errors="coerce").dropna()
        raw = raw[raw > 0]
        if len(raw) > 0:
            ds_with_rates.append((ds_name, ds_label))

    if not ds_with_rates:
        logger.warning("No datasets with rate data, skipping rate distributions")
        return

    fig, (ax_raw, ax_target) = plt.subplots(1, 2, figsize=(14, 5))

    # LEFT: editing rates normalised to 0-1 (Levanon / 100, others as-is)
    _PERCENT_SCALE = {"advisor_c2t"}
    for ds_name, ds_label in ds_with_rates:
        ds_df = combined_df[combined_df["dataset_source"] == ds_name]
        rates = pd.to_numeric(ds_df["editing_rate"], errors="coerce").dropna()
        rates = rates[rates > 0]
        if ds_name in _PERCENT_SCALE:
            rates = rates / 100.0
        if len(rates) > 0:
            ax_raw.hist(rates, bins=60, alpha=0.55,
                        label=f"{ds_label} (n={len(rates)})",
                        color=DATASET_COLORS.get(ds_label, "#666"),
                        edgecolor="white", linewidth=0.3)
    ax_raw.set_yscale("log")
    ax_raw.set_title("Editing Rates (0–1 scale)", fontsize=13, fontweight="bold")
    ax_raw.set_xlabel("Editing Rate (fraction)", fontsize=10)
    ax_raw.set_ylabel("Count (log scale)", fontsize=11)
    ax_raw.legend(fontsize=9)
    ax_raw.set_ylim(bottom=0.8)

    # RIGHT: per-dataset Z-scored log2 rates (each dataset standardized independently)
    for ds_name, ds_label in ds_with_rates:
        ds_df = combined_df[combined_df["dataset_source"] == ds_name]
        norm = pd.to_numeric(ds_df["editing_rate_normalized"], errors="coerce").dropna()
        norm = norm[norm > 0]
        if len(norm) > 0:
            log_rates = np.log2(norm.values + 0.01)
            mu, sigma = log_rates.mean(), log_rates.std()
            if sigma > 0:
                z = (log_rates - mu) / sigma
            else:
                z = log_rates - mu
            ax_target.hist(z, bins=60, alpha=0.55,
                           label=f"{ds_label} (n={len(norm)})",
                           color=DATASET_COLORS.get(ds_label, "#666"),
                           edgecolor="white", linewidth=0.3)
    ax_target.set_yscale("log")
    ax_target.set_title("Per-Dataset Z-scored log₂ Rates", fontsize=13, fontweight="bold")
    ax_target.set_xlabel("Z-score (per-dataset standardized)", fontsize=10)
    ax_target.set_ylabel("Count (log scale)", fontsize=11)
    ax_target.legend(fontsize=9)
    ax_target.set_ylim(bottom=0.8)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "rate_distributions.png")
    plt.close(fig)
    logger.info("Saved rate_distributions.png")


def plot_gene_overlap_matrix(overlap_matrix):
    """Plot gene overlap heatmap between datasets."""
    ds_names = [d for d in DATASET_LABELS.values() if d in overlap_matrix]
    n = len(ds_names)

    gene_matrix = np.zeros((n, n))
    coord_matrix = np.zeros((n, n))
    for i, d1 in enumerate(ds_names):
        for j, d2 in enumerate(ds_names):
            gene_matrix[i, j] = overlap_matrix[d1][d2]["shared_genes"]
            coord_matrix[i, j] = overlap_matrix[d1][d2]["shared_coordinates"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Gene overlap
    im1 = ax1.imshow(gene_matrix, cmap="Blues", aspect="auto")
    ax1.set_xticks(range(n))
    ax1.set_yticks(range(n))
    ax1.set_xticklabels(ds_names, rotation=45, ha="right")
    ax1.set_yticklabels(ds_names)
    ax1.set_title("Shared Genes")
    for i in range(n):
        for j in range(n):
            ax1.text(j, i, f"{int(gene_matrix[i, j])}", ha="center", va="center",
                     fontsize=9, color="white" if gene_matrix[i, j] > gene_matrix.max() * 0.6 else "black")
    fig.colorbar(im1, ax=ax1, shrink=0.8)

    # Coordinate overlap
    im2 = ax2.imshow(coord_matrix, cmap="Reds", aspect="auto")
    ax2.set_xticks(range(n))
    ax2.set_yticks(range(n))
    ax2.set_xticklabels(ds_names, rotation=45, ha="right")
    ax2.set_yticklabels(ds_names)
    ax2.set_title("Shared Coordinates")
    for i in range(n):
        for j in range(n):
            ax2.text(j, i, f"{int(coord_matrix[i, j])}", ha="center", va="center",
                     fontsize=9, color="white" if coord_matrix[i, j] > coord_matrix.max() * 0.6 else "black")
    fig.colorbar(im2, ax=ax2, shrink=0.8)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "gene_overlap_matrix.png")
    plt.close(fig)
    logger.info("Saved gene_overlap_matrix.png")


def plot_dataset_overview(stats):
    """Plot dataset overview comparison chart."""
    ds_names = [s["name"] for s in stats.values() if s["name"] in DATASET_COLORS and s["name"] not in ("Tier2 Neg", "Tier3 Neg")]
    n_sites = [stats[d]["n_sites"] for d in ds_names]
    n_genes = [stats[d]["n_genes"] for d in ds_names]
    colors = [DATASET_COLORS.get(d, "#666") for d in ds_names]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Sites per dataset
    bars = axes[0].bar(ds_names, n_sites, color=colors, edgecolor="white", linewidth=0.5)
    axes[0].set_title("Sites per Dataset")
    axes[0].set_ylabel("Number of Sites")
    for bar, val in zip(bars, n_sites):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{val:,}", ha="center", va="bottom", fontsize=9)
    axes[0].tick_params(axis="x", rotation=30)

    # Genes per dataset
    bars = axes[1].bar(ds_names, n_genes, color=colors, edgecolor="white", linewidth=0.5)
    axes[1].set_title("Unique Genes per Dataset")
    axes[1].set_ylabel("Number of Genes")
    for bar, val in zip(bars, n_genes):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{val:,}", ha="center", va="bottom", fontsize=9)
    axes[1].tick_params(axis="x", rotation=30)

    # Mean editing rate
    mean_rates = []
    for d in ds_names:
        r = stats[d].get("rate_mean", 0)
        mean_rates.append(r if r else 0)

    bars = axes[2].bar(ds_names, mean_rates, color=colors, edgecolor="white", linewidth=0.5)
    axes[2].set_title("Mean Editing Rate (%)")
    axes[2].set_ylabel("Rate (%)")
    for bar, val in zip(bars, mean_rates):
        label = f"{val:.2f}%" if val > 0 else "N/A"
        axes[2].text(bar.get_x() + bar.get_width() / 2, max(bar.get_height(), 0.01),
                     label, ha="center", va="bottom", fontsize=9)
    axes[2].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "dataset_overview.png")
    plt.close(fig)
    logger.info("Saved dataset_overview.png")


def plot_structure_delta_comparison(combined_df, splits_df, structure_delta):
    """Plot structure delta distributions by dataset."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    all_labels = {**DATASET_LABELS, "tier2_negative": "Tier2 Neg", "tier3_negative": "Tier3 Neg"}
    all_dfs = {**{k: combined_df[combined_df["dataset_source"] == k] for k in DATASET_LABELS},
               **{k: splits_df[splits_df["dataset_source"] == k] for k in ["tier2_negative", "tier3_negative"]}}

    ds_order = ["Levanon", "Asaoka", "Sharma", "Alqassim", "Baysal", "Tier2 Neg", "Tier3 Neg"]
    delta_mfe_data = []
    delta_pair_data = []
    labels = []

    for ds_name, ds_label in all_labels.items():
        if ds_label not in ds_order:
            continue
        df = all_dfs.get(ds_name, pd.DataFrame())
        deltas = []
        for sid in df["site_id"]:
            if sid in structure_delta:
                deltas.append(structure_delta[sid])
        if deltas:
            deltas = np.array(deltas)
            delta_mfe_data.append(deltas[:, 3])
            delta_pair_data.append(deltas[:, 0])
            labels.append(ds_label)

    if delta_mfe_data:
        colors = [DATASET_COLORS.get(l, "#666") for l in labels]
        bp1 = axes[0].boxplot(delta_mfe_data, labels=labels, patch_artist=True, showfliers=False)
        for patch, color in zip(bp1["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[0].set_title("Delta MFE by Dataset")
        axes[0].set_ylabel("Delta MFE (kcal/mol)")
        axes[0].tick_params(axis="x", rotation=30)
        axes[0].axhline(0, color="gray", linestyle="--", alpha=0.5)

        bp2 = axes[1].boxplot(delta_pair_data, labels=labels, patch_artist=True, showfliers=False)
        for patch, color in zip(bp2["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[1].set_title("Delta Pairing Probability at Edit Site")
        axes[1].set_ylabel("Delta Pairing Prob")
        axes[1].tick_params(axis="x", rotation=30)
        axes[1].axhline(0, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "structure_delta_comparison.png")
    plt.close(fig)
    logger.info("Saved structure_delta_comparison.png")


# -----------------------------------------------------------------------
# NEW: Site overlap UpSet-style bar chart
# -----------------------------------------------------------------------


def plot_site_overlap_upset(combined_df):
    """Plot an UpSet-style bar chart showing site overlaps between datasets.

    Each site is identified by (chr, start). We enumerate all combinations of
    datasets that share sites and display the intersection sizes.
    """
    from itertools import combinations

    ds_labels_ordered = [v for v in DATASET_LABELS.values()]
    coord_to_datasets = {}
    for ds_name, ds_label in DATASET_LABELS.items():
        ds_df = combined_df[combined_df["dataset_source"] == ds_name]
        for _, row in ds_df.iterrows():
            coord = (row["chr"], row["start"])
            coord_to_datasets.setdefault(coord, set()).add(ds_label)

    # Count occurrences of each combination
    combo_counts = Counter()
    for coord, ds_set in coord_to_datasets.items():
        key = frozenset(ds_set)
        combo_counts[key] += 1

    # Sort by count descending, take top 20
    sorted_combos = combo_counts.most_common(20)
    if not sorted_combos:
        return

    combo_labels = []
    combo_sizes = []
    combo_members = []
    for fs, count in sorted_combos:
        combo_labels.append(" & ".join(sorted(fs)))
        combo_sizes.append(count)
        combo_members.append(fs)

    n_combos = len(combo_labels)
    n_ds = len(ds_labels_ordered)

    fig = plt.figure(figsize=(max(12, n_combos * 0.6), 7))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1.2], hspace=0.05)
    ax_bar = fig.add_subplot(gs[0])
    ax_dots = fig.add_subplot(gs[1])

    # Bar chart of intersection sizes
    bar_colors = []
    for members in combo_members:
        if len(members) == 1:
            bar_colors.append(DATASET_COLORS.get(list(members)[0], "#999"))
        else:
            bar_colors.append("#475569")

    ax_bar.bar(range(n_combos), combo_sizes, color=bar_colors, edgecolor="white", linewidth=0.5)
    for i, v in enumerate(combo_sizes):
        ax_bar.text(i, v + max(combo_sizes) * 0.01, str(v), ha="center", va="bottom", fontsize=8)
    ax_bar.set_ylabel("Number of Sites")
    ax_bar.set_title("Site Overlap Between Datasets (UpSet-style)")
    ax_bar.set_xlim(-0.5, n_combos - 0.5)
    ax_bar.set_xticks([])

    # Dot matrix showing set membership
    for i, members in enumerate(combo_members):
        active_rows = []
        for j, ds_label in enumerate(ds_labels_ordered):
            if ds_label in members:
                ax_dots.scatter(i, j, s=60, c="#1e293b", zorder=3)
                active_rows.append(j)
            else:
                ax_dots.scatter(i, j, s=30, c="#cbd5e1", zorder=2)
        if len(active_rows) > 1:
            ax_dots.plot([i, i], [min(active_rows), max(active_rows)],
                         c="#1e293b", linewidth=2, zorder=2)

    ax_dots.set_yticks(range(n_ds))
    ax_dots.set_yticklabels(ds_labels_ordered, fontsize=9)
    ax_dots.set_xlim(-0.5, n_combos - 0.5)
    ax_dots.set_xticks([])
    ax_dots.invert_yaxis()
    ax_dots.set_xlabel("Intersection")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "site_overlap_upset.png")
    plt.close(fig)
    logger.info("Saved site_overlap_upset.png")


# -----------------------------------------------------------------------
# NEW: Chromosome distribution per dataset
# -----------------------------------------------------------------------


def plot_chromosome_distribution(combined_df, splits_df):
    """Plot chromosome distribution as a grouped stacked bar chart."""
    all_labels = {**DATASET_LABELS, "tier2_negative": "Tier2 Neg", "tier3_negative": "Tier3 Neg"}
    all_dfs = {**{k: combined_df[combined_df["dataset_source"] == k] for k in DATASET_LABELS},
               **{k: splits_df[splits_df["dataset_source"] == k] for k in ["tier2_negative", "tier3_negative"]}}

    ds_order = ["Levanon", "Asaoka", "Sharma", "Alqassim", "Baysal", "Tier2 Neg", "Tier3 Neg"]

    # Build chromosome counts per dataset (fraction)
    chr_data = {}
    for ds_name, ds_label in all_labels.items():
        if ds_label not in ds_order:
            continue
        df = all_dfs.get(ds_name, pd.DataFrame())
        if len(df) == 0:
            continue
        counts = df["chr"].value_counts()
        total = counts.sum()
        chr_data[ds_label] = {c: counts.get(c, 0) / total * 100 for c in CHROM_ORDER}

    if not chr_data:
        return

    fig, ax = plt.subplots(figsize=(16, 6))
    x = np.arange(len(CHROM_ORDER))
    n_ds = len(chr_data)
    width = 0.8 / n_ds

    for i, (ds_label, counts) in enumerate(chr_data.items()):
        vals = [counts.get(c, 0) for c in CHROM_ORDER]
        offset = (i - n_ds / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=ds_label,
               color=DATASET_COLORS.get(ds_label, "#999"), alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(CHROM_ORDER, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Fraction of Sites (%)")
    ax.set_title("Chromosome Distribution per Dataset")
    ax.legend(fontsize=8, ncol=3, loc="upper right")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "chromosome_distribution.png")
    plt.close(fig)
    logger.info("Saved chromosome_distribution.png")


# -----------------------------------------------------------------------
# NEW: Genomic category (feature) stacked bar chart
# -----------------------------------------------------------------------


def plot_genomic_category_breakdown(combined_df):
    """Plot stacked bar chart of genomic category fractions per dataset."""
    if "feature" not in combined_df.columns:
        logger.warning("No 'feature' column found, skipping genomic category plot")
        return

    category_sets = {}
    for ds_name, ds_label in DATASET_LABELS.items():
        ds_df = combined_df[combined_df["dataset_source"] == ds_name]
        if len(ds_df) == 0:
            continue
        feat_counts = ds_df["feature"].fillna("Unknown").value_counts()
        category_sets[ds_label] = feat_counts

    if not category_sets:
        return

    all_categories = set()
    for counts in category_sets.values():
        all_categories.update(counts.index.tolist())
    all_categories = sorted(all_categories)

    cat_colors = plt.cm.Set3(np.linspace(0, 1, max(len(all_categories), 3)))

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ds_labels_plot = list(category_sets.keys())
    x = np.arange(len(ds_labels_plot))
    bottom = np.zeros(len(category_sets))
    for ci, cat in enumerate(all_categories):
        vals_raw = [category_sets[d].get(cat, 0) for d in ds_labels_plot]
        totals = [category_sets[d].sum() for d in ds_labels_plot]
        vals = [v / t * 100 if t > 0 else 0 for v, t in zip(vals_raw, totals)]
        ax.bar(x, vals, bottom=bottom, label=cat, color=cat_colors[ci % len(cat_colors)],
               edgecolor="white", linewidth=0.5)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(ds_labels_plot, rotation=30, ha="right")
    ax.set_ylabel("Fraction (%)")
    ax.set_title("Genomic Category (Fraction)")
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5))

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "genomic_category_breakdown.png")
    plt.close(fig)
    logger.info("Saved genomic_category_breakdown.png")


# -----------------------------------------------------------------------
# NEW: TC-motif prevalence comparison
# -----------------------------------------------------------------------


def plot_tc_motif_comparison(stats):
    """Plot TC-motif prevalence per dataset as bar chart."""
    ds_names = []
    tc_fracs = []
    for ds_label in DATASET_LABELS.values():
        if ds_label in stats and "tc_motif_fraction" in stats[ds_label]:
            ds_names.append(ds_label)
            tc_fracs.append(stats[ds_label]["tc_motif_fraction"] * 100)

    if not ds_names:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [DATASET_COLORS.get(d, "#666") for d in ds_names]
    bars = ax.bar(ds_names, tc_fracs, color=colors, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, tc_fracs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("TC-Motif Prevalence (%)")
    ax.set_title("TC-Motif Prevalence per Dataset")
    ax.set_ylim(0, max(tc_fracs) * 1.15)
    ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "tc_motif_comparison.png")
    plt.close(fig)
    logger.info("Saved tc_motif_comparison.png")


# -----------------------------------------------------------------------
# NEW: Positive vs Negative comparison
# -----------------------------------------------------------------------


def plot_positive_vs_negative(splits_df, sequences, structure_delta):
    """Compare feature distributions between positive and negative sites."""
    pos_df = splits_df[splits_df["label"] == 1].copy()
    neg_df = splits_df[splits_df["label"] == 0].copy()

    if len(pos_df) == 0 or len(neg_df) == 0:
        logger.warning("Empty positive or negative set, skipping pos vs neg plot")
        return

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # 1. Editing rate distribution (positive only, since negatives have no rate)
    pos_rates = pd.to_numeric(pos_df["editing_rate"], errors="coerce").dropna()
    pos_rates = pos_rates[pos_rates > 0]
    if len(pos_rates) > 0:
        axes[0, 0].hist(pos_rates, bins=50, color="#2563eb", alpha=0.8,
                        edgecolor="white", linewidth=0.5)
        axes[0, 0].axvline(pos_rates.median(), color="red", linestyle="--",
                           linewidth=1.5, label=f"Median={pos_rates.median():.1f}%")
        axes[0, 0].legend(fontsize=9)
    axes[0, 0].set_title(f"Editing Rate (Positive Sites, n={len(pos_rates)})")
    axes[0, 0].set_xlabel("Editing Rate (%)")
    axes[0, 0].set_ylabel("Count")

    # 2. Dataset source distribution (positive vs negative stacked)
    pos_sources = pos_df["dataset_source"].map(
        {**DATASET_LABELS, "tier2_negative": "Tier2 Neg", "tier3_negative": "Tier3 Neg"}
    ).fillna("Other").value_counts()
    neg_sources = neg_df["dataset_source"].map(
        {**DATASET_LABELS, "tier2_negative": "Tier2 Neg", "tier3_negative": "Tier3 Neg"}
    ).fillna("Other").value_counts()

    all_sources = sorted(set(pos_sources.index) | set(neg_sources.index))
    x = np.arange(len(all_sources))
    pos_vals = [pos_sources.get(s, 0) for s in all_sources]
    neg_vals = [neg_sources.get(s, 0) for s in all_sources]

    axes[0, 1].bar(x - 0.2, pos_vals, 0.35, label="Positive", color="#2563eb", alpha=0.8)
    axes[0, 1].bar(x + 0.2, neg_vals, 0.35, label="Negative", color="#dc2626", alpha=0.8)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(all_sources, rotation=45, ha="right", fontsize=8)
    axes[0, 1].set_title("Sites by Dataset Source")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].legend(fontsize=9)

    # 3. TC-motif prevalence (positive vs negative)
    seq_keys = set(sequences.keys())
    tc_pos = 0
    n_pos_seq = 0
    for sid in pos_df["site_id"]:
        if sid in seq_keys:
            n_pos_seq += 1
            seq = sequences[sid]
            center = len(seq) // 2
            if center >= 1 and seq[center - 1] in "Uu":
                tc_pos += 1

    tc_neg = 0
    n_neg_seq = 0
    for sid in neg_df["site_id"]:
        if sid in seq_keys:
            n_neg_seq += 1
            seq = sequences[sid]
            center = len(seq) // 2
            if center >= 1 and seq[center - 1] in "Uu":
                tc_neg += 1

    tc_frac_pos = tc_pos / max(n_pos_seq, 1) * 100
    tc_frac_neg = tc_neg / max(n_neg_seq, 1) * 100

    bars = axes[0, 2].bar(["Positive", "Negative"], [tc_frac_pos, tc_frac_neg],
                          color=["#2563eb", "#dc2626"], edgecolor="white")
    for bar, val in zip(bars, [tc_frac_pos, tc_frac_neg]):
        axes[0, 2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{val:.1f}%", ha="center", va="bottom", fontsize=10)
    axes[0, 2].set_title("TC-Motif Prevalence")
    axes[0, 2].set_ylabel("Fraction (%)")

    # 4. Structure delta MFE (positive vs negative)
    pos_mfe = []
    neg_mfe = []
    for sid in pos_df["site_id"]:
        if sid in structure_delta:
            pos_mfe.append(float(structure_delta[sid][3]))
    for sid in neg_df["site_id"]:
        if sid in structure_delta:
            neg_mfe.append(float(structure_delta[sid][3]))

    if pos_mfe and neg_mfe:
        bp = axes[1, 0].boxplot([pos_mfe, neg_mfe], labels=["Positive", "Negative"],
                                patch_artist=True, showfliers=False)
        bp["boxes"][0].set_facecolor("#2563eb")
        bp["boxes"][0].set_alpha(0.7)
        bp["boxes"][1].set_facecolor("#dc2626")
        bp["boxes"][1].set_alpha(0.7)
        axes[1, 0].axhline(0, color="gray", linestyle="--", alpha=0.5)
        axes[1, 0].set_title(f"Delta MFE (pos n={len(pos_mfe)}, neg n={len(neg_mfe)})")
        axes[1, 0].set_ylabel("Delta MFE (kcal/mol)")
    else:
        axes[1, 0].text(0.5, 0.5, "No structure data", transform=axes[1, 0].transAxes,
                        ha="center", fontsize=12, color="gray")

    # 5. Structure delta pairing probability (positive vs negative)
    pos_pair = []
    neg_pair = []
    for sid in pos_df["site_id"]:
        if sid in structure_delta:
            pos_pair.append(float(structure_delta[sid][0]))
    for sid in neg_df["site_id"]:
        if sid in structure_delta:
            neg_pair.append(float(structure_delta[sid][0]))

    if pos_pair and neg_pair:
        bp = axes[1, 1].boxplot([pos_pair, neg_pair], labels=["Positive", "Negative"],
                                patch_artist=True, showfliers=False)
        bp["boxes"][0].set_facecolor("#2563eb")
        bp["boxes"][0].set_alpha(0.7)
        bp["boxes"][1].set_facecolor("#dc2626")
        bp["boxes"][1].set_alpha(0.7)
        axes[1, 1].axhline(0, color="gray", linestyle="--", alpha=0.5)
        axes[1, 1].set_title("Delta Pairing Probability")
        axes[1, 1].set_ylabel("Delta Pairing Prob")
    else:
        axes[1, 1].text(0.5, 0.5, "No structure data", transform=axes[1, 1].transAxes,
                        ha="center", fontsize=12, color="gray")

    # 6. Chromosome distribution (positive vs negative)
    pos_chr = pos_df["chr"].value_counts()
    neg_chr = neg_df["chr"].value_counts()
    chroms = [c for c in CHROM_ORDER if c in pos_chr.index or c in neg_chr.index]
    if chroms:
        x = np.arange(len(chroms))
        pos_v = [pos_chr.get(c, 0) / len(pos_df) * 100 for c in chroms]
        neg_v = [neg_chr.get(c, 0) / len(neg_df) * 100 for c in chroms]
        axes[1, 2].bar(x - 0.2, pos_v, 0.35, label="Positive", color="#2563eb", alpha=0.8)
        axes[1, 2].bar(x + 0.2, neg_v, 0.35, label="Negative", color="#dc2626", alpha=0.8)
        axes[1, 2].set_xticks(x)
        axes[1, 2].set_xticklabels(chroms, rotation=60, ha="right", fontsize=7)
        axes[1, 2].set_title("Chromosome Distribution")
        axes[1, 2].set_ylabel("Fraction (%)")
        axes[1, 2].legend(fontsize=8)

    plt.suptitle("Positive vs Negative Site Comparison", fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "positive_vs_negative.png")
    plt.close(fig)
    logger.info("Saved positive_vs_negative.png")


# -----------------------------------------------------------------------
# NEW: Per-dataset statistics table as figure
# -----------------------------------------------------------------------


def plot_statistics_table(stats):
    """Render per-dataset statistics as a matplotlib table figure."""
    ds_order = [v for v in DATASET_LABELS.values() if v in stats]
    ds_order += [t for t in ["Tier2 Neg", "Tier3 Neg"] if t in stats]

    columns = ["Dataset", "Sites", "Genes", "Mean Rate", "Median Rate", "Std Rate",
               "TC-Motif %", "Delta MFE"]

    rows = []
    for ds in ds_order:
        s = stats[ds]
        rows.append([
            s["name"],
            f"{s['n_sites']:,}",
            str(s.get("n_genes", "-")),
            f"{s['rate_mean']:.2f}%" if "rate_mean" in s else "-",
            f"{s['rate_median']:.2f}%" if "rate_median" in s else "-",
            f"{s['rate_std']:.2f}" if "rate_std" in s else "-",
            f"{s['tc_motif_fraction']*100:.1f}%" if "tc_motif_fraction" in s else "-",
            f"{s['delta_mfe_mean']:.3f}" if "delta_mfe_mean" in s else "-",
        ])

    fig, ax = plt.subplots(figsize=(14, 1.5 + 0.5 * len(rows)))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.auto_set_column_width(range(len(columns)))

    # Color header
    for j in range(len(columns)):
        table[(0, j)].set_facecolor("#e2e8f0")
        table[(0, j)].set_text_props(fontweight="bold")

    # Color dataset rows
    for i, ds in enumerate(ds_order):
        color = DATASET_COLORS.get(ds, "#ffffff")
        table[(i + 1, 0)].set_facecolor(color + "33")  # light version

    ax.set_title("Per-Dataset Statistics Summary", fontsize=13, pad=20)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "statistics_table.png")
    plt.close(fig)
    logger.info("Saved statistics_table.png")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading data...")
    combined_df, splits_df, sequences, structure_delta = load_data()
    logger.info("Combined: %d sites, Splits: %d sites, Sequences: %d, Structure: %d",
                len(combined_df), len(splits_df), len(sequences), len(structure_delta))

    logger.info("Computing statistics...")
    stats, overlap_matrix, coord_sets = compute_dataset_statistics(
        combined_df, splits_df, sequences, structure_delta
    )

    # Print summary to console
    logger.info("=" * 70)
    logger.info("DATASET STATISTICS SUMMARY")
    logger.info("=" * 70)
    for ds_label, s in stats.items():
        if ds_label in ("Tier2 Neg", "Tier3 Neg"):
            logger.info("  %s: %d sites", ds_label, s["n_sites"])
        else:
            rate_str = f"mean_rate={s.get('rate_mean', 'N/A')}"
            tc_str = f"TC-motif={s.get('tc_motif_fraction', 0)*100:.1f}%" if "tc_motif_fraction" in s else ""
            logger.info("  %s: %d sites, %d genes, %s %s",
                        ds_label, s["n_sites"], s["n_genes"], rate_str, tc_str)

    logger.info("Generating visualizations...")

    # Original plots
    plot_rate_distributions(combined_df)
    plot_gene_overlap_matrix(overlap_matrix)
    plot_dataset_overview(stats)
    plot_structure_delta_comparison(combined_df, splits_df, structure_delta)

    # New plots
    plot_site_overlap_upset(combined_df)
    plot_chromosome_distribution(combined_df, splits_df)
    plot_genomic_category_breakdown(combined_df)
    plot_tc_motif_comparison(stats)
    plot_positive_vs_negative(splits_df, sequences, structure_delta)
    plot_statistics_table(stats)

    gc.collect()

    # Save JSON results
    def serialize(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, tuple):
            return list(obj)
        return str(obj)

    with open(OUTPUT_DIR / "dataset_statistics.json", "w") as f:
        json.dump({"statistics": stats, "overlap_matrix": overlap_matrix}, f, indent=2, default=serialize)

    logger.info("Results saved to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
