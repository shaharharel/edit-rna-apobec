#!/usr/bin/env python
"""RNA structure effect analysis for APOBEC3A editing.

Analyzes how RNA secondary structure features relate to:
1. Editing site identification (positive vs negative)
2. Editing rate prediction
3. Structure change before vs after C→U edit
4. How structure is encoded in the edit embedding space

Generates figures for HTML report section on structure effects.

Usage:
    python experiments/apobec3a/exp_structure_analysis.py
"""

import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, mannwhitneyu

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logger = logging.getLogger(__name__)

EMB_DIR = PROJECT_ROOT / "data" / "processed" / "embeddings"
SPLITS_CSV = PROJECT_ROOT / "data" / "processed" / "splits_expanded_a3a.csv"
STRUCT_CACHE = EMB_DIR / "vienna_structure_cache.npz"
SEQ_JSON = PROJECT_ROOT / "data" / "processed" / "site_sequences.json"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "apobec" / "outputs" / "structure_analysis"

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

FEATURE_NAMES = [
    "Delta Pairing (edit site)",
    "Delta Accessibility (edit site)",
    "Delta Entropy (edit site)",
    "Delta MFE (kcal/mol)",
    "Mean Delta Pairing (±10nt)",
    "Mean Delta Accessibility (±10nt)",
    "Std Delta Pairing (±10nt)",
]

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
})


def load_data():
    """Load structure cache, embeddings, and metadata."""
    splits_df = pd.read_csv(SPLITS_CSV)
    splits_df["dataset_label"] = splits_df["dataset_source"].map(DATASET_LABELS).fillna(splits_df["dataset_source"])
    splits_df["editing_rate"] = pd.to_numeric(splits_df["editing_rate"], errors="coerce")

    # Load structure cache
    struct_data = np.load(STRUCT_CACHE, allow_pickle=True)
    struct_sids = [str(s) for s in struct_data["site_ids"]]
    delta_features = struct_data["delta_features"]
    mfes_orig = struct_data["mfes"]
    mfes_edited = struct_data["mfes_edited"]
    pairing_orig = struct_data["pairing_probs"]
    pairing_edited = struct_data["pairing_probs_edited"]
    entropy_orig = struct_data["entropies"]
    entropy_edited = struct_data["entropies_edited"]

    # Build structure dict
    struct_dict = {}
    for i, sid in enumerate(struct_sids):
        struct_dict[sid] = {
            "delta_features": delta_features[i],
            "mfe_orig": float(mfes_orig[i]),
            "mfe_edited": float(mfes_edited[i]),
            "pairing_orig": pairing_orig[i],
            "pairing_edited": pairing_edited[i],
            "entropy_orig": entropy_orig[i],
            "entropy_edited": entropy_edited[i],
        }

    # Merge structure with metadata
    available = set(struct_sids) & set(splits_df["site_id"].values)
    df = splits_df[splits_df["site_id"].isin(available)].copy()

    # Add structure features
    for j, fname in enumerate(FEATURE_NAMES):
        col = f"struct_{j}"
        df[col] = df["site_id"].map(lambda sid, idx=j: struct_dict.get(sid, {}).get("delta_features", np.zeros(7))[idx])

    df["mfe_orig"] = df["site_id"].map(lambda sid: struct_dict.get(sid, {}).get("mfe_orig", np.nan))
    df["mfe_edited"] = df["site_id"].map(lambda sid: struct_dict.get(sid, {}).get("mfe_edited", np.nan))
    df["delta_mfe"] = df["mfe_edited"] - df["mfe_orig"]

    # Center position pairing/entropy
    center = 100  # 201-nt window, center at 100
    df["pairing_orig_center"] = df["site_id"].map(
        lambda sid: struct_dict.get(sid, {}).get("pairing_orig", np.zeros(201))[center])
    df["pairing_edited_center"] = df["site_id"].map(
        lambda sid: struct_dict.get(sid, {}).get("pairing_edited", np.zeros(201))[center])
    df["entropy_orig_center"] = df["site_id"].map(
        lambda sid: struct_dict.get(sid, {}).get("entropy_orig", np.zeros(201))[center])
    df["entropy_edited_center"] = df["site_id"].map(
        lambda sid: struct_dict.get(sid, {}).get("entropy_edited", np.zeros(201))[center])

    logger.info("Loaded %d sites with structure data", len(df))
    return df, struct_dict


def plot_structure_before_after(df):
    """Plot structure properties before vs after C→U edit."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    pos = df[df["label"] == 1]
    neg = df[df["label"] == 0]

    # Panel 1: MFE before vs after
    ax = axes[0, 0]
    ax.scatter(pos["mfe_orig"], pos["mfe_edited"], s=3, alpha=0.2, c="#2563eb", label=f"Positive (n={len(pos)})")
    ax.scatter(neg["mfe_orig"], neg["mfe_edited"], s=3, alpha=0.2, c="#dc2626", label=f"Negative (n={len(neg)})")
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, "k--", alpha=0.3, lw=1)
    ax.set_xlabel("MFE before edit (kcal/mol)")
    ax.set_ylabel("MFE after C→U edit (kcal/mol)")
    ax.set_title("Stability Before vs After Edit")
    ax.legend(fontsize=8, markerscale=3)

    # Panel 2: Pairing probability at edit site before vs after
    ax = axes[0, 1]
    ax.scatter(pos["pairing_orig_center"], pos["pairing_edited_center"],
               s=3, alpha=0.2, c="#2563eb", label="Positive")
    ax.scatter(neg["pairing_orig_center"], neg["pairing_edited_center"],
               s=3, alpha=0.2, c="#dc2626", label="Negative")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, lw=1)
    ax.set_xlabel("Pairing Prob (before edit)")
    ax.set_ylabel("Pairing Prob (after C→U)")
    ax.set_title("Base Pairing at Edit Site")
    ax.legend(fontsize=8, markerscale=3)

    # Panel 3: Entropy at edit site before vs after
    ax = axes[1, 0]
    ax.scatter(pos["entropy_orig_center"], pos["entropy_edited_center"],
               s=3, alpha=0.2, c="#2563eb", label="Positive")
    ax.scatter(neg["entropy_orig_center"], neg["entropy_edited_center"],
               s=3, alpha=0.2, c="#dc2626", label="Negative")
    lims = [0, max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, "k--", alpha=0.3, lw=1)
    ax.set_xlabel("Structure Entropy (before edit)")
    ax.set_ylabel("Structure Entropy (after C→U)")
    ax.set_title("Structural Entropy at Edit Site")
    ax.legend(fontsize=8, markerscale=3)

    # Panel 4: Delta MFE distribution by label
    ax = axes[1, 1]
    bins = np.linspace(-3, 3, 60)
    ax.hist(pos["delta_mfe"].dropna().clip(-3, 3), bins=bins, alpha=0.6,
            color="#2563eb", label=f"Positive (mean={pos['delta_mfe'].mean():.3f})", density=True)
    ax.hist(neg["delta_mfe"].dropna().clip(-3, 3), bins=bins, alpha=0.6,
            color="#dc2626", label=f"Negative (mean={neg['delta_mfe'].mean():.3f})", density=True)
    ax.axvline(0, color="black", lw=1, ls="--", alpha=0.5)
    ax.set_xlabel("Delta MFE (kcal/mol)")
    ax.set_ylabel("Density")
    ax.set_title("Stability Change Distribution")
    ax.legend(fontsize=8)

    plt.suptitle("RNA Structure: Before vs After C→U Edit", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "structure_before_after.png")
    plt.close(fig)
    logger.info("Saved structure_before_after.png")


def plot_7dim_features(df):
    """Plot all 7 structure delta features: positive vs negative distributions."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    pos = df[df["label"] == 1]
    neg = df[df["label"] == 0]

    for j in range(7):
        ax = axes[j]
        col = f"struct_{j}"
        p_vals = pos[col].dropna()
        n_vals = neg[col].dropna()

        # Clip for visualization
        lo, hi = np.percentile(np.concatenate([p_vals, n_vals]), [1, 99])
        bins = np.linspace(lo, hi, 50)

        ax.hist(p_vals.clip(lo, hi), bins=bins, alpha=0.6, color="#2563eb",
                label="Positive", density=True)
        ax.hist(n_vals.clip(lo, hi), bins=bins, alpha=0.6, color="#dc2626",
                label="Negative", density=True)
        ax.axvline(0, color="black", lw=0.8, ls="--", alpha=0.4)

        # Mann-Whitney test
        if len(p_vals) > 10 and len(n_vals) > 10:
            stat, pval = mannwhitneyu(p_vals, n_vals, alternative="two-sided")
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
            ax.set_title(f"{FEATURE_NAMES[j]}\n(p={pval:.1e}, {sig})", fontsize=9)
        else:
            ax.set_title(FEATURE_NAMES[j], fontsize=9)

        ax.set_xlabel("")
        ax.set_ylabel("Density" if j % 4 == 0 else "")
        if j == 0:
            ax.legend(fontsize=7)

    # Remove empty subplot
    axes[7].axis("off")

    plt.suptitle("7-Dim Structure Delta Features: Positive vs Negative", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "structure_7dim_features.png")
    plt.close(fig)
    logger.info("Saved structure_7dim_features.png")


def plot_structure_vs_rate(df):
    """Plot structure features vs editing rate for positive sites."""
    pos = df[(df["label"] == 1) & df["editing_rate"].notna() & (df["editing_rate"] > 0)].copy()
    pos["log_rate"] = np.log10(pos["editing_rate"].clip(0.01))

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # Key structure features to plot vs rate
    features = [
        ("delta_mfe", "Delta MFE (kcal/mol)"),
        ("struct_0", "Delta Pairing at Edit Site"),
        ("struct_2", "Delta Entropy at Edit Site"),
        ("mfe_orig", "Original MFE (kcal/mol)"),
        ("pairing_orig_center", "Original Pairing Prob"),
        ("entropy_orig_center", "Original Entropy"),
    ]

    for idx, (col, label) in enumerate(features):
        ax = axes[idx // 3, idx % 3]
        vals = pos[col].dropna()
        rates = pos.loc[vals.index, "log_rate"]

        # Color by dataset
        for ds_label, color in DATASET_COLORS.items():
            if ds_label in ("Tier2 Neg", "Tier3 Neg"):
                continue
            mask = pos.loc[vals.index, "dataset_label"] == ds_label
            if mask.sum() > 0:
                ax.scatter(vals[mask], rates[mask], s=5, alpha=0.3, c=color, label=ds_label)

        # Correlation
        valid = vals.notna() & rates.notna()
        if valid.sum() > 10:
            r, p = spearmanr(vals[valid], rates[valid])
            ax.set_title(f"{label}\nSpearman ρ={r:.3f} (p={p:.1e})", fontsize=10)
        else:
            ax.set_title(label, fontsize=10)

        ax.set_xlabel(label if idx >= 3 else "")
        ax.set_ylabel("log10(Editing Rate %)" if idx % 3 == 0 else "")

    axes[0, 0].legend(fontsize=7, markerscale=2, loc="best")

    plt.suptitle("Structure Features vs Editing Rate (Positive Sites)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "structure_vs_rate.png")
    plt.close(fig)
    logger.info("Saved structure_vs_rate.png")


def plot_structure_per_dataset(df):
    """Compare structure features across datasets."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    features = [
        ("delta_mfe", "Delta MFE (kcal/mol)"),
        ("struct_0", "Delta Pairing (edit site)"),
        ("struct_2", "Delta Entropy (edit site)"),
        ("mfe_orig", "Original MFE (kcal/mol)"),
        ("pairing_orig_center", "Original Pairing Prob"),
        ("entropy_orig_center", "Original Entropy"),
    ]

    ds_order = ["Advisor", "Asaoka", "Alqassim", "Sharma", "Tier2 Neg", "Tier3 Neg"]

    for idx, (col, label) in enumerate(features):
        ax = axes[idx // 3, idx % 3]
        data_groups = []
        labels_list = []
        colors = []

        for ds in ds_order:
            vals = df[df["dataset_label"] == ds][col].dropna()
            if len(vals) > 5:
                data_groups.append(vals.values)
                labels_list.append(ds)
                colors.append(DATASET_COLORS.get(ds, "#999"))

        if data_groups:
            bp = ax.boxplot(data_groups, tick_labels=labels_list, patch_artist=True, showfliers=False)
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.5)
            ax.set_title(label, fontsize=10)
            ax.tick_params(axis="x", rotation=45)

    plt.suptitle("Structure Features by Dataset", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "structure_per_dataset.png")
    plt.close(fig)
    logger.info("Saved structure_per_dataset.png")


def plot_structure_in_embedding(df):
    """Show how structure features correlate with edit embedding PCA components."""
    import torch
    from sklearn.decomposition import PCA

    pooled_orig = torch.load(EMB_DIR / "rnafm_pooled.pt", weights_only=False)
    pooled_edited = torch.load(EMB_DIR / "rnafm_pooled_edited.pt", weights_only=False)

    # Compute diff embeddings
    site_ids = df["site_id"].tolist()
    diffs = []
    valid_ids = []
    for sid in site_ids:
        if sid in pooled_orig and sid in pooled_edited:
            diff = (pooled_edited[sid] - pooled_orig[sid]).numpy()
            diffs.append(diff)
            valid_ids.append(sid)

    diff_matrix = np.array(diffs)
    pca = PCA(n_components=10)
    pca_coords = pca.fit_transform(diff_matrix)

    meta = df.set_index("site_id").loc[valid_ids].reset_index()

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    struct_cols = [
        ("delta_mfe", "Delta MFE"),
        ("struct_0", "Delta Pairing"),
        ("struct_2", "Delta Entropy"),
        ("mfe_orig", "Original MFE"),
        ("pairing_orig_center", "Original Pairing"),
        ("entropy_orig_center", "Original Entropy"),
    ]

    for idx, (col, label) in enumerate(struct_cols):
        ax = axes[idx // 3, idx % 3]
        vals = meta[col].values
        valid = ~np.isnan(vals)

        if valid.sum() > 50:
            vmin, vmax = np.percentile(vals[valid], [5, 95])
            sc = ax.scatter(pca_coords[valid, 0], pca_coords[valid, 1],
                           s=4, alpha=0.3, c=vals[valid], cmap="coolwarm",
                           vmin=vmin, vmax=vmax, edgecolors="none")
            plt.colorbar(sc, ax=ax, shrink=0.8)

        ax.set_title(f"PCA colored by {label}", fontsize=10)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2" if idx % 3 == 0 else "")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle("Structure Features in Edit Embedding Space (PCA)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "structure_in_embedding.png")
    plt.close(fig)
    logger.info("Saved structure_in_embedding.png")

    # Also compute correlations between PCs and structure features
    correlations = {}
    for col, label in struct_cols:
        vals = meta[col].values
        valid = ~np.isnan(vals)
        if valid.sum() > 50:
            pc_corrs = []
            for pc_i in range(10):
                r, p = spearmanr(pca_coords[valid, pc_i], vals[valid])
                pc_corrs.append({"pc": pc_i + 1, "spearman_r": float(r), "p_value": float(p)})
            correlations[label] = pc_corrs
    return correlations


def plot_stability_effect(df):
    """Does C→U stabilize or destabilize? And does direction matter for editing?"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    pos = df[df["label"] == 1].copy()
    neg = df[df["label"] == 0].copy()

    # Panel 1: Stabilizing vs destabilizing by label
    ax = axes[0]
    for label_val, color, name in [(1, "#2563eb", "Positive"), (0, "#dc2626", "Negative")]:
        subset = df[df["label"] == label_val]
        stabilizing = (subset["delta_mfe"] < -0.01).sum()
        destabilizing = (subset["delta_mfe"] > 0.01).sum()
        neutral = len(subset) - stabilizing - destabilizing
        total = len(subset)
        bars = ax.bar(
            [f"{name}\nStabilizing", f"{name}\nNeutral", f"{name}\nDestabilizing"],
            [stabilizing / total * 100, neutral / total * 100, destabilizing / total * 100],
            color=[color] * 3, alpha=0.7,
        )
        for bar, count in zip(bars, [stabilizing, neutral, destabilizing]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"n={count}", ha="center", fontsize=8)
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Stabilizing vs Destabilizing Edits")

    # Panel 2: Rate vs stability direction (positive sites only)
    ax = axes[1]
    rate_pos = pos[pos["editing_rate"].notna() & (pos["editing_rate"] > 0)].copy()
    rate_pos["stability_class"] = pd.cut(
        rate_pos["delta_mfe"],
        bins=[-np.inf, -0.5, -0.01, 0.01, 0.5, np.inf],
        labels=["Strong\nstabilize", "Mild\nstabilize", "Neutral", "Mild\ndestabilize", "Strong\ndestabilize"]
    )
    group_means = rate_pos.groupby("stability_class", observed=True)["editing_rate"].agg(["mean", "median", "count"])
    if len(group_means) > 0:
        colors_stab = ["#2563eb", "#60a5fa", "#94a3b8", "#f97316", "#dc2626"]
        bars = ax.bar(range(len(group_means)), group_means["mean"], color=colors_stab[:len(group_means)], alpha=0.8)
        ax.set_xticks(range(len(group_means)))
        ax.set_xticklabels(group_means.index, fontsize=8)
        for bar, (_, row) in zip(bars, group_means.iterrows()):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"n={int(row['count'])}", ha="center", fontsize=7)
    ax.set_ylabel("Mean Editing Rate (%)")
    ax.set_title("Editing Rate by Stability Effect")

    # Panel 3: Does higher original stability → more editing?
    ax = axes[2]
    for ds_label, color in [("Advisor", "#2563eb"), ("Asaoka", "#16a34a"), ("Sharma", "#dc2626")]:
        subset = pos[(pos["dataset_label"] == ds_label) & pos["editing_rate"].notna() & (pos["editing_rate"] > 0)]
        if len(subset) > 10:
            ax.scatter(subset["mfe_orig"], np.log10(subset["editing_rate"].clip(0.01)),
                      s=5, alpha=0.3, c=color, label=ds_label)
    ax.set_xlabel("Original MFE (kcal/mol)")
    ax.set_ylabel("log10(Editing Rate %)")
    ax.set_title("Original Stability vs Editing Rate")
    ax.legend(fontsize=8, markerscale=3)

    plt.suptitle("RNA Stability and APOBEC3A Editing", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "stability_effect.png")
    plt.close(fig)
    logger.info("Saved stability_effect.png")


def compute_statistics(df):
    """Compute summary statistics for the report."""
    pos = df[df["label"] == 1]
    neg = df[df["label"] == 0]

    stats = {
        "n_sites_with_structure": len(df),
        "n_positive": len(pos),
        "n_negative": len(neg),
    }

    # Delta MFE stats
    for name, subset in [("positive", pos), ("negative", neg)]:
        dmfe = subset["delta_mfe"].dropna()
        stats[f"{name}_delta_mfe_mean"] = float(dmfe.mean())
        stats[f"{name}_delta_mfe_median"] = float(dmfe.median())
        stats[f"{name}_delta_mfe_std"] = float(dmfe.std())
        stats[f"{name}_stabilizing_pct"] = float((dmfe < -0.01).sum() / len(dmfe) * 100)
        stats[f"{name}_destabilizing_pct"] = float((dmfe > 0.01).sum() / len(dmfe) * 100)

    # Per-feature Mann-Whitney
    feature_tests = {}
    for j, fname in enumerate(FEATURE_NAMES):
        col = f"struct_{j}"
        p_vals = pos[col].dropna()
        n_vals = neg[col].dropna()
        if len(p_vals) > 10 and len(n_vals) > 10:
            stat, pval = mannwhitneyu(p_vals, n_vals, alternative="two-sided")
            feature_tests[fname] = {
                "pos_mean": float(p_vals.mean()),
                "neg_mean": float(n_vals.mean()),
                "difference": float(p_vals.mean() - n_vals.mean()),
                "mann_whitney_p": float(pval),
                "significant": pval < 0.001,
            }
    stats["feature_tests"] = feature_tests

    # Structure vs rate correlation
    rate_pos = pos[pos["editing_rate"].notna() & (pos["editing_rate"] > 0)]
    if len(rate_pos) > 20:
        log_rate = np.log10(rate_pos["editing_rate"].clip(0.01))
        rate_corrs = {}
        for col, label in [("delta_mfe", "Delta MFE"), ("struct_0", "Delta Pairing"),
                           ("struct_2", "Delta Entropy"), ("mfe_orig", "Original MFE"),
                           ("pairing_orig_center", "Original Pairing")]:
            vals = rate_pos[col].dropna()
            idx = vals.index
            if len(vals) > 20:
                r, p = spearmanr(vals, log_rate.loc[idx])
                rate_corrs[label] = {"spearman_r": float(r), "p_value": float(p)}
        stats["structure_rate_correlations"] = rate_corrs

    # Per-dataset delta MFE
    ds_stats = {}
    for ds_label in ["Advisor", "Asaoka", "Sharma", "Alqassim", "Tier2 Neg", "Tier3 Neg"]:
        subset = df[df["dataset_label"] == ds_label]
        dmfe = subset["delta_mfe"].dropna()
        if len(dmfe) > 5:
            ds_stats[ds_label] = {
                "n": int(len(dmfe)),
                "mean_delta_mfe": float(dmfe.mean()),
                "median_delta_mfe": float(dmfe.median()),
                "mean_orig_mfe": float(subset["mfe_orig"].mean()),
                "mean_pairing": float(subset["pairing_orig_center"].mean()),
            }
    stats["per_dataset"] = ds_stats

    return stats


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df, struct_dict = load_data()

    logger.info("Generating structure before/after plots...")
    plot_structure_before_after(df)

    logger.info("Generating 7-dim feature distributions...")
    plot_7dim_features(df)

    logger.info("Generating structure vs rate plots...")
    plot_structure_vs_rate(df)

    logger.info("Generating per-dataset structure comparison...")
    plot_structure_per_dataset(df)

    logger.info("Generating stability effect analysis...")
    plot_stability_effect(df)

    logger.info("Generating structure in embedding space...")
    pc_corrs = plot_structure_in_embedding(df)

    logger.info("Computing statistics...")
    stats = compute_statistics(df)
    stats["pc_structure_correlations"] = pc_corrs

    def serialize(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    with open(OUTPUT_DIR / "structure_analysis.json", "w") as f:
        json.dump(stats, f, indent=2, default=serialize)

    logger.info("\nAll figures saved to %s", OUTPUT_DIR)

    # Print summary
    print("\n" + "=" * 70)
    print("STRUCTURE ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"\nSites with structure: {stats['n_sites_with_structure']}")
    print(f"Positive delta MFE: {stats['positive_delta_mfe_mean']:.4f} ± {stats['positive_delta_mfe_std']:.4f}")
    print(f"Negative delta MFE: {stats['negative_delta_mfe_mean']:.4f} ± {stats['negative_delta_mfe_std']:.4f}")
    print(f"\nPositive: {stats['positive_stabilizing_pct']:.1f}% stabilizing, {stats['positive_destabilizing_pct']:.1f}% destabilizing")
    print(f"Negative: {stats['negative_stabilizing_pct']:.1f}% stabilizing, {stats['negative_destabilizing_pct']:.1f}% destabilizing")
    print(f"\n7-dim feature significance (positive vs negative):")
    for fname, info in stats["feature_tests"].items():
        sig = "***" if info["mann_whitney_p"] < 0.001 else "ns"
        print(f"  {fname}: pos={info['pos_mean']:.4f}, neg={info['neg_mean']:.4f}, p={info['mann_whitney_p']:.1e} {sig}")
    print("=" * 70)


if __name__ == "__main__":
    main()
