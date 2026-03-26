#!/usr/bin/env python
"""Enhanced RNA structure analysis for APOBEC3A editing sites.

Compares RAW structural features (before-edit values) between positive and
negative editing sites, visualises structural motif preferences, and analyses
loop-size distributions on the full dataset (no subsampling).

Outputs:
    experiments/apobec3a/outputs/structure_enhanced/
        pairing_accessibility_comparison.png
        structural_context_profiles.png
        loop_size_distribution.png
        loop_type_comparison.png
        stem_length_comparison.png
        structural_motif_summary.png
        structure_enhanced_results.json

Usage:
    python experiments/apobec3a/exp_structure_enhanced.py
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
from scipy.stats import mannwhitneyu

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
EMB_DIR = PROJECT_ROOT / "data" / "processed" / "embeddings"
SPLITS_CSV = PROJECT_ROOT / "data" / "processed" / "splits_expanded_a3a.csv"
STRUCT_CACHE = EMB_DIR / "vienna_structure_cache.npz"
LOOP_CSV = (
    PROJECT_ROOT
    / "experiments"
    / "apobec"
    / "outputs"
    / "loop_position"
    / "loop_position_per_site.csv"
)
OUTPUT_DIR = (
    PROJECT_ROOT / "experiments" / "apobec3a" / "outputs" / "structure_enhanced"
)

CENTER = 100  # edit position in 201-nt window
LOCAL_HALF = 10  # ±10 nt context

# ---------------------------------------------------------------------------
# Matplotlib style
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
    }
)

COLOR_POS = "#3b82f6"  # blue
COLOR_NEG = "#ef4444"  # red


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data():
    """Load splits, structure cache, and loop-position table.

    Returns
    -------
    df : pd.DataFrame
        Per-site table with label, raw structural features at edit site, and
        delta MFE.
    struct : dict
        Raw numpy arrays from the structure cache, keyed by array name.
    loop_df : pd.DataFrame
        Loop position data for every site.
    """
    # ---- splits ----
    splits_df = pd.read_csv(SPLITS_CSV)
    splits_df["editing_rate"] = pd.to_numeric(
        splits_df["editing_rate"], errors="coerce"
    )

    # ---- structure cache ----
    struct = dict(np.load(STRUCT_CACHE, allow_pickle=True))
    site_ids = np.array([str(s) for s in struct["site_ids"]])
    sid_to_idx = {sid: i for i, sid in enumerate(site_ids)}

    # Build per-site feature table aligned with splits_df
    available = set(site_ids) & set(splits_df["site_id"].values)
    df = splits_df[splits_df["site_id"].isin(available)].copy()
    df = df.reset_index(drop=True)

    # Map site_id -> struct index
    idx_arr = df["site_id"].map(sid_to_idx).values.astype(int)

    # Raw (before-edit) features at edit position
    df["pairing_prob_at_edit"] = struct["pairing_probs"][idx_arr, CENTER]
    df["accessibility_at_edit"] = struct["accessibilities"][idx_arr, CENTER]
    df["entropy_at_edit"] = struct["entropies"][idx_arr, CENTER]

    # Local context (mean over ±10 nt)
    sl = slice(CENTER - LOCAL_HALF, CENTER + LOCAL_HALF + 1)  # 90:111
    df["local_pairing_mean"] = struct["pairing_probs"][idx_arr, sl].mean(axis=1)
    df["local_accessibility_mean"] = struct["accessibilities"][idx_arr, sl].mean(
        axis=1
    )

    # Delta MFE (edited - original)
    df["delta_mfe"] = (
        struct["mfes_edited"][idx_arr] - struct["mfes"][idx_arr]
    ).astype(float)

    # Keep full profiles for the profile plot
    df_profiles = {
        "pairing_pos": struct["pairing_probs"][
            idx_arr[df["label"].values == 1]
        ],
        "pairing_neg": struct["pairing_probs"][
            idx_arr[df["label"].values == 0]
        ],
    }

    # ---- loop position ----
    loop_df = pd.read_csv(LOOP_CSV)

    logger.info(
        "Loaded %d sites (%d pos, %d neg) with structure data",
        len(df),
        (df["label"] == 1).sum(),
        (df["label"] == 0).sum(),
    )
    return df, df_profiles, loop_df


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------
def run_feature_tests(df):
    """Mann-Whitney U tests for each raw feature: positive vs negative."""
    pos = df[df["label"] == 1]
    neg = df[df["label"] == 0]

    features = [
        ("Pairing Prob (edit site)", "pairing_prob_at_edit"),
        ("Accessibility (edit site)", "accessibility_at_edit"),
        ("Entropy (edit site)", "entropy_at_edit"),
        ("Local Pairing Mean (+/-10nt)", "local_pairing_mean"),
        ("Local Accessibility Mean (+/-10nt)", "local_accessibility_mean"),
        ("Delta MFE (kcal/mol)", "delta_mfe"),
    ]

    results = {}
    for name, col in features:
        pv = pos[col].dropna().values
        nv = neg[col].dropna().values
        if len(pv) > 10 and len(nv) > 10:
            stat, p = mannwhitneyu(pv, nv, alternative="two-sided")
            results[name] = {
                "pos_mean": float(np.mean(pv)),
                "neg_mean": float(np.mean(nv)),
                "difference": float(np.mean(pv) - np.mean(nv)),
                "mann_whitney_p": float(p),
                "significant": bool(p < 0.05),
            }
    return results


def compute_loop_stats(loop_df):
    """Compute loop size / type statistics for positive and negative sites."""
    stats = {}
    for lbl, name in [(1, "positive"), (0, "negative")]:
        sub = loop_df[loop_df["label"] == lbl]
        unpaired = sub[sub["is_unpaired"] == True]  # noqa: E712
        stats[name] = {
            "n_total": int(len(sub)),
            "n_in_loops": int(len(unpaired)),
            "pct_in_loops": float(
                len(unpaired) / len(sub) * 100 if len(sub) > 0 else 0
            ),
            "mean_loop_size": float(unpaired["loop_size"].mean())
            if len(unpaired) > 0
            else None,
            "median_loop_size": float(unpaired["loop_size"].median())
            if len(unpaired) > 0
            else None,
            "loop_type_counts": (
                unpaired["loop_type"]
                .value_counts()
                .to_dict()
            )
            if len(unpaired) > 0
            else {},
        }
    return stats


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def _pval_text(p):
    if p < 0.001:
        return f"p = {p:.1e} ***"
    elif p < 0.01:
        return f"p = {p:.1e} **"
    elif p < 0.05:
        return f"p = {p:.3f} *"
    else:
        return f"p = {p:.3f} ns"


def plot_pairing_accessibility_comparison(df, feature_tests):
    """Side-by-side violin plots for pairing prob and accessibility at edit."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    pos = df[df["label"] == 1]
    neg = df[df["label"] == 0]

    for ax_idx, (col, title) in enumerate(
        [
            ("pairing_prob_at_edit", "Pairing Probability at Edit Site"),
            ("accessibility_at_edit", "Accessibility at Edit Site"),
        ]
    ):
        ax = axes[ax_idx]
        data_pos = pos[col].dropna().values
        data_neg = neg[col].dropna().values

        parts = ax.violinplot(
            [data_pos, data_neg],
            positions=[1, 2],
            showmeans=True,
            showmedians=True,
            showextrema=False,
        )
        # Color the violins
        for i, body in enumerate(parts["bodies"]):
            body.set_facecolor(COLOR_POS if i == 0 else COLOR_NEG)
            body.set_alpha(0.6)
        parts["cmeans"].set_color("black")
        parts["cmedians"].set_color("gray")

        # Annotate means
        ax.text(
            1,
            np.mean(data_pos) + 0.02,
            f"mean={np.mean(data_pos):.3f}",
            ha="center",
            fontsize=9,
            color=COLOR_POS,
            fontweight="bold",
        )
        ax.text(
            2,
            np.mean(data_neg) + 0.02,
            f"mean={np.mean(data_neg):.3f}",
            ha="center",
            fontsize=9,
            color=COLOR_NEG,
            fontweight="bold",
        )

        # p-value annotation
        lookup = (
            "Pairing Prob (edit site)"
            if ax_idx == 0
            else "Accessibility (edit site)"
        )
        if lookup in feature_tests:
            p = feature_tests[lookup]["mann_whitney_p"]
            ax.text(
                1.5,
                ax.get_ylim()[1] * 0.95,
                _pval_text(p),
                ha="center",
                fontsize=10,
                fontstyle="italic",
            )

        ax.set_xticks([1, 2])
        ax.set_xticklabels(
            [f"Positive (n={len(data_pos)})", f"Negative (n={len(data_neg)})"]
        )
        ax.set_title(title)
        ax.set_ylabel(col.replace("_", " ").title())

    plt.suptitle(
        "Raw Structural Features: Positive vs Negative Editing Sites",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "pairing_accessibility_comparison.png")
    plt.close(fig)
    logger.info("Saved pairing_accessibility_comparison.png")


def plot_structural_context_profiles(df_profiles):
    """Mean pairing probability profiles across the 201-nt window."""
    fig, ax = plt.subplots(figsize=(12, 5))

    positions = np.arange(201) - CENTER  # -100 to +100

    mean_pos = df_profiles["pairing_pos"].mean(axis=0)
    mean_neg = df_profiles["pairing_neg"].mean(axis=0)

    # Smooth with a rolling mean for clearer visualization
    window = 5
    kernel = np.ones(window) / window
    mean_pos_smooth = np.convolve(mean_pos, kernel, mode="same")
    mean_neg_smooth = np.convolve(mean_neg, kernel, mode="same")

    # Standard error bands
    n_pos = df_profiles["pairing_pos"].shape[0]
    n_neg = df_profiles["pairing_neg"].shape[0]
    se_pos = df_profiles["pairing_pos"].std(axis=0) / np.sqrt(n_pos)
    se_neg = df_profiles["pairing_neg"].std(axis=0) / np.sqrt(n_neg)
    se_pos_smooth = np.convolve(se_pos, kernel, mode="same")
    se_neg_smooth = np.convolve(se_neg, kernel, mode="same")

    ax.plot(positions, mean_pos_smooth, color=COLOR_POS, lw=2,
            label=f"Positive (n={n_pos})")
    ax.fill_between(
        positions,
        mean_pos_smooth - 1.96 * se_pos_smooth,
        mean_pos_smooth + 1.96 * se_pos_smooth,
        color=COLOR_POS,
        alpha=0.15,
    )

    ax.plot(positions, mean_neg_smooth, color=COLOR_NEG, lw=2,
            label=f"Negative (n={n_neg})")
    ax.fill_between(
        positions,
        mean_neg_smooth - 1.96 * se_neg_smooth,
        mean_neg_smooth + 1.96 * se_neg_smooth,
        color=COLOR_NEG,
        alpha=0.15,
    )

    ax.axvline(0, color="black", ls="--", lw=1, alpha=0.5, label="Edit site")
    ax.axvspan(-LOCAL_HALF, LOCAL_HALF, color="gray", alpha=0.07,
               label=f"+/-{LOCAL_HALF}nt context")

    ax.set_xlabel("Position relative to edit site (nt)")
    ax.set_ylabel("Mean pairing probability")
    ax.set_title(
        "Pairing Probability Profile Around Edit Site",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=10)
    ax.set_xlim(-100, 100)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "structural_context_profiles.png")
    plt.close(fig)
    logger.info("Saved structural_context_profiles.png")

    # --- Raw (unsmoothed) version ---
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(positions, mean_pos, color=COLOR_POS, lw=1.5,
            label=f"Positive (n={n_pos})")
    ax.fill_between(
        positions,
        mean_pos - 1.96 * se_pos,
        mean_pos + 1.96 * se_pos,
        color=COLOR_POS,
        alpha=0.15,
    )

    ax.plot(positions, mean_neg, color=COLOR_NEG, lw=1.5,
            label=f"Negative (n={n_neg})")
    ax.fill_between(
        positions,
        mean_neg - 1.96 * se_neg,
        mean_neg + 1.96 * se_neg,
        color=COLOR_NEG,
        alpha=0.15,
    )

    ax.axvline(0, color="black", ls="--", lw=1, alpha=0.5, label="Edit site")
    ax.axvspan(-LOCAL_HALF, LOCAL_HALF, color="gray", alpha=0.07,
               label=f"+/-{LOCAL_HALF}nt context")

    ax.set_xlabel("Position relative to edit site (nt)")
    ax.set_ylabel("Mean pairing probability")
    ax.set_title(
        "Pairing Probability Profile Around Edit Site (Raw)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=10)
    ax.set_xlim(-100, 100)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "structural_context_profiles_raw.png")
    plt.close(fig)
    logger.info("Saved structural_context_profiles_raw.png")


def plot_loop_size_distribution(loop_df):
    """Loop size histogram for ALL unpaired sites (full dataset)."""
    unpaired = loop_df[loop_df["is_unpaired"] == True].copy()  # noqa: E712
    pos = unpaired[unpaired["label"] == 1]
    neg = unpaired[unpaired["label"] == 0]

    fig, ax = plt.subplots(figsize=(10, 5))

    max_size = int(min(unpaired["loop_size"].quantile(0.99), 30))
    bins = np.arange(0.5, max_size + 1.5, 1)

    ax.hist(
        pos["loop_size"].values,
        bins=bins,
        alpha=0.6,
        color=COLOR_POS,
        label=f"Positive (n={len(pos)})",
        edgecolor="white",
        linewidth=0.5,
    )
    ax.hist(
        neg["loop_size"].values,
        bins=bins,
        alpha=0.6,
        color=COLOR_NEG,
        label=f"Negative (n={len(neg)})",
        edgecolor="white",
        linewidth=0.5,
    )

    ax.set_xlabel("Loop Size (nt)")
    ax.set_ylabel("Count")
    ax.set_title(
        "Loop Size Distribution (All Unpaired Sites, Full Dataset)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=11)

    # Annotate means
    ax.axvline(
        pos["loop_size"].mean(),
        color=COLOR_POS,
        ls="--",
        lw=1.5,
        label=f"Pos mean={pos['loop_size'].mean():.1f}",
    )
    ax.axvline(
        neg["loop_size"].mean(),
        color=COLOR_NEG,
        ls="--",
        lw=1.5,
        label=f"Neg mean={neg['loop_size'].mean():.1f}",
    )
    ax.legend(fontsize=10)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "loop_size_distribution.png")
    plt.close(fig)
    logger.info("Saved loop_size_distribution.png")


def plot_loop_type_comparison(loop_df):
    """Bar chart comparing loop type distribution between pos and neg."""
    unpaired = loop_df[loop_df["is_unpaired"] == True].copy()  # noqa: E712
    pos = unpaired[unpaired["label"] == 1]
    neg = unpaired[unpaired["label"] == 0]

    # Normalised loop type counts
    loop_types = ["hairpin", "bulge_left", "bulge_right", "multiloop"]
    nice_names = ["Hairpin", "Bulge (left)", "Bulge (right)", "Multibranch"]

    pos_counts = pos["loop_type"].value_counts()
    neg_counts = neg["loop_type"].value_counts()

    pos_pct = np.array(
        [pos_counts.get(lt, 0) / len(pos) * 100 for lt in loop_types]
    )
    neg_pct = np.array(
        [neg_counts.get(lt, 0) / len(neg) * 100 for lt in loop_types]
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    x = np.arange(len(loop_types))
    width = 0.35

    bars_pos = ax.bar(
        x - width / 2,
        pos_pct,
        width,
        label=f"Positive (n={len(pos)})",
        color=COLOR_POS,
        alpha=0.8,
    )
    bars_neg = ax.bar(
        x + width / 2,
        neg_pct,
        width,
        label=f"Negative (n={len(neg)})",
        color=COLOR_NEG,
        alpha=0.8,
    )

    # Annotate counts
    for bar, pct in zip(bars_pos, pos_pct):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{pct:.1f}%",
            ha="center",
            fontsize=9,
        )
    for bar, pct in zip(bars_neg, neg_pct):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{pct:.1f}%",
            ha="center",
            fontsize=9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(nice_names, fontsize=11)
    ax.set_ylabel("Percentage of Unpaired Sites (%)")
    ax.set_title(
        "Loop Type Distribution: Positive vs Negative",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=11)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "loop_type_comparison.png")
    plt.close(fig)
    logger.info("Saved loop_type_comparison.png")


def plot_stem_length_comparison(loop_df):
    """Violin / box plot comparing stem lengths between pos and neg."""
    pos = loop_df[loop_df["label"] == 1]
    neg = loop_df[loop_df["label"] == 0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax_idx, col, title in [
        (0, "left_stem_length", "Left Stem Length"),
        (1, "right_stem_length", "Right Stem Length"),
    ]:
        ax = axes[ax_idx]
        data_pos = pos[col].dropna().values
        data_neg = neg[col].dropna().values

        parts = ax.violinplot(
            [data_pos, data_neg],
            positions=[1, 2],
            showmeans=True,
            showmedians=True,
            showextrema=False,
        )
        for i, body in enumerate(parts["bodies"]):
            body.set_facecolor(COLOR_POS if i == 0 else COLOR_NEG)
            body.set_alpha(0.6)
        parts["cmeans"].set_color("black")
        parts["cmedians"].set_color("gray")

        # Overlay box plots for IQR
        bp = ax.boxplot(
            [data_pos, data_neg],
            positions=[1, 2],
            widths=0.15,
            patch_artist=True,
            showfliers=False,
            zorder=3,
        )
        for patch, color in zip(bp["boxes"], [COLOR_POS, COLOR_NEG]):
            patch.set_facecolor(color)
            patch.set_alpha(0.3)

        # p-value
        if len(data_pos) > 10 and len(data_neg) > 10:
            _, p = mannwhitneyu(data_pos, data_neg, alternative="two-sided")
            ax.text(
                1.5,
                ax.get_ylim()[1] * 0.92,
                _pval_text(p),
                ha="center",
                fontsize=10,
                fontstyle="italic",
            )

        ax.set_xticks([1, 2])
        ax.set_xticklabels(
            [f"Positive\n(n={len(data_pos)})", f"Negative\n(n={len(data_neg)})"]
        )
        ax.set_title(title)
        ax.set_ylabel("Stem length (bp)")

    plt.suptitle(
        "Adjacent Stem Lengths: Positive vs Negative",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "stem_length_comparison.png")
    plt.close(fig)
    logger.info("Saved stem_length_comparison.png")


def plot_structural_motif_summary(df, feature_tests, loop_df, df_profiles):
    """Multi-panel (3x2) summary figure combining key structural features."""
    fig = plt.figure(figsize=(16, 18))
    gs = gridspec.GridSpec(3, 2, hspace=0.35, wspace=0.30)

    pos = df[df["label"] == 1]
    neg = df[df["label"] == 0]

    # ------------------------------------------------------------------
    # Panel (0, 0): Pairing probability violin
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    data_pos = pos["pairing_prob_at_edit"].dropna().values
    data_neg = neg["pairing_prob_at_edit"].dropna().values
    parts = ax.violinplot(
        [data_pos, data_neg],
        positions=[1, 2],
        showmeans=True,
        showmedians=True,
        showextrema=False,
    )
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(COLOR_POS if i == 0 else COLOR_NEG)
        body.set_alpha(0.6)
    parts["cmeans"].set_color("black")
    parts["cmedians"].set_color("gray")
    lookup = "Pairing Prob (edit site)"
    if lookup in feature_tests:
        p = feature_tests[lookup]["mann_whitney_p"]
        ax.set_title(f"Pairing Prob at Edit Site\n{_pval_text(p)}", fontsize=11)
    else:
        ax.set_title("Pairing Prob at Edit Site", fontsize=11)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Positive", "Negative"])
    ax.set_ylabel("Pairing probability")
    ax.text(0.02, 0.95, "A", transform=ax.transAxes, fontsize=16,
            fontweight="bold", va="top")

    # ------------------------------------------------------------------
    # Panel (0, 1): Accessibility violin
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    data_pos = pos["accessibility_at_edit"].dropna().values
    data_neg = neg["accessibility_at_edit"].dropna().values
    parts = ax.violinplot(
        [data_pos, data_neg],
        positions=[1, 2],
        showmeans=True,
        showmedians=True,
        showextrema=False,
    )
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(COLOR_POS if i == 0 else COLOR_NEG)
        body.set_alpha(0.6)
    parts["cmeans"].set_color("black")
    parts["cmedians"].set_color("gray")
    lookup = "Accessibility (edit site)"
    if lookup in feature_tests:
        p = feature_tests[lookup]["mann_whitney_p"]
        ax.set_title(f"Accessibility at Edit Site\n{_pval_text(p)}", fontsize=11)
    else:
        ax.set_title("Accessibility at Edit Site", fontsize=11)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Positive", "Negative"])
    ax.set_ylabel("Accessibility")
    ax.text(0.02, 0.95, "B", transform=ax.transAxes, fontsize=16,
            fontweight="bold", va="top")

    # ------------------------------------------------------------------
    # Panel (1, 0): Pairing profile across 201-nt window
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    positions = np.arange(201) - CENTER
    mean_pos_prof = df_profiles["pairing_pos"].mean(axis=0)
    mean_neg_prof = df_profiles["pairing_neg"].mean(axis=0)
    window = 5
    kernel = np.ones(window) / window
    ax.plot(positions, np.convolve(mean_pos_prof, kernel, mode="same"),
            color=COLOR_POS, lw=1.5, label="Positive")
    ax.plot(positions, np.convolve(mean_neg_prof, kernel, mode="same"),
            color=COLOR_NEG, lw=1.5, label="Negative")
    ax.axvline(0, color="black", ls="--", lw=1, alpha=0.5)
    ax.set_xlabel("Position relative to edit (nt)")
    ax.set_ylabel("Mean pairing prob")
    ax.set_title("Pairing Profile (+/-100 nt)", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_xlim(-100, 100)
    ax.text(0.02, 0.95, "C", transform=ax.transAxes, fontsize=16,
            fontweight="bold", va="top")

    # ------------------------------------------------------------------
    # Panel (1, 1): Loop type bar chart
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    unpaired = loop_df[loop_df["is_unpaired"] == True]  # noqa: E712
    lp = unpaired[unpaired["label"] == 1]
    ln = unpaired[unpaired["label"] == 0]
    loop_types = ["hairpin", "bulge_left", "bulge_right", "multiloop"]
    nice_names = ["Hairpin", "Bulge (L)", "Bulge (R)", "Multi"]
    pos_pct = [lp["loop_type"].value_counts().get(lt, 0) / len(lp) * 100
               for lt in loop_types]
    neg_pct = [ln["loop_type"].value_counts().get(lt, 0) / len(ln) * 100
               for lt in loop_types]
    x = np.arange(len(loop_types))
    w = 0.35
    ax.bar(x - w / 2, pos_pct, w, color=COLOR_POS, alpha=0.8, label="Positive")
    ax.bar(x + w / 2, neg_pct, w, color=COLOR_NEG, alpha=0.8, label="Negative")
    ax.set_xticks(x)
    ax.set_xticklabels(nice_names, fontsize=10)
    ax.set_ylabel("% of unpaired sites")
    ax.set_title("Loop Type Distribution", fontsize=11)
    ax.legend(fontsize=9)
    ax.text(0.02, 0.95, "D", transform=ax.transAxes, fontsize=16,
            fontweight="bold", va="top")

    # ------------------------------------------------------------------
    # Panel (2, 0): Loop size histogram
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[2, 0])
    max_size = int(min(unpaired["loop_size"].quantile(0.99), 30))
    bins = np.arange(0.5, max_size + 1.5, 1)
    ax.hist(lp["loop_size"].values, bins=bins, alpha=0.6, color=COLOR_POS,
            edgecolor="white", linewidth=0.5, label="Positive")
    ax.hist(ln["loop_size"].values, bins=bins, alpha=0.6, color=COLOR_NEG,
            edgecolor="white", linewidth=0.5, label="Negative")
    ax.set_xlabel("Loop size (nt)")
    ax.set_ylabel("Count")
    ax.set_title("Loop Size Distribution", fontsize=11)
    ax.legend(fontsize=9)
    ax.text(0.02, 0.95, "E", transform=ax.transAxes, fontsize=16,
            fontweight="bold", va="top")

    # ------------------------------------------------------------------
    # Panel (2, 1): Stem length violin
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[2, 1])
    lp_all = loop_df[loop_df["label"] == 1]
    ln_all = loop_df[loop_df["label"] == 0]
    data = [
        lp_all["left_stem_length"].dropna().values,
        ln_all["left_stem_length"].dropna().values,
        lp_all["right_stem_length"].dropna().values,
        ln_all["right_stem_length"].dropna().values,
    ]
    parts = ax.violinplot(
        data,
        positions=[1, 2, 3.5, 4.5],
        showmeans=True,
        showmedians=True,
        showextrema=False,
    )
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(COLOR_POS if i % 2 == 0 else COLOR_NEG)
        body.set_alpha(0.6)
    parts["cmeans"].set_color("black")
    parts["cmedians"].set_color("gray")
    ax.set_xticks([1, 2, 3.5, 4.5])
    ax.set_xticklabels(["Pos\n(left)", "Neg\n(left)", "Pos\n(right)",
                         "Neg\n(right)"], fontsize=9)
    ax.set_ylabel("Stem length (bp)")
    ax.set_title("Adjacent Stem Lengths", fontsize=11)
    ax.text(0.02, 0.95, "F", transform=ax.transAxes, fontsize=16,
            fontweight="bold", va="top")

    fig.suptitle(
        "Structural Motif Summary: APOBEC3A Editing Sites",
        fontsize=16,
        fontweight="bold",
        y=0.99,
    )
    fig.savefig(OUTPUT_DIR / "structural_motif_summary.png")
    plt.close(fig)
    logger.info("Saved structural_motif_summary.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading data...")
    df, df_profiles, loop_df = load_data()

    # ---- Statistical tests ----
    logger.info("Running feature tests...")
    feature_tests = run_feature_tests(df)
    loop_stats = compute_loop_stats(loop_df)

    # ---- Plots ----
    logger.info("Generating pairing/accessibility comparison...")
    plot_pairing_accessibility_comparison(df, feature_tests)

    logger.info("Generating structural context profiles...")
    plot_structural_context_profiles(df_profiles)

    logger.info("Generating loop size distribution...")
    plot_loop_size_distribution(loop_df)

    logger.info("Generating loop type comparison...")
    plot_loop_type_comparison(loop_df)

    logger.info("Generating stem length comparison...")
    plot_stem_length_comparison(loop_df)

    logger.info("Generating structural motif summary...")
    plot_structural_motif_summary(df, feature_tests, loop_df, df_profiles)

    # ---- Output JSON ----
    n_pos = int((df["label"] == 1).sum())
    n_neg = int((df["label"] == 0).sum())
    results = {
        "n_positive": n_pos,
        "n_negative": n_neg,
        "feature_tests": feature_tests,
        "loop_stats": loop_stats,
    }

    def _serialize(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    out_path = OUTPUT_DIR / "structure_enhanced_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=_serialize)
    logger.info("Saved results to %s", out_path)

    # ---- Print summary ----
    print("\n" + "=" * 72)
    print("ENHANCED STRUCTURE ANALYSIS SUMMARY")
    print("=" * 72)
    print(f"\nTotal sites: {n_pos + n_neg}  (positive={n_pos}, negative={n_neg})")
    print("\n--- Raw Feature Tests (positive vs negative) ---")
    for name, info in feature_tests.items():
        sig = "***" if info["mann_whitney_p"] < 0.001 else (
            "**" if info["mann_whitney_p"] < 0.01 else (
                "*" if info["mann_whitney_p"] < 0.05 else "ns"
            )
        )
        print(
            f"  {name:40s}  pos={info['pos_mean']:+.4f}  "
            f"neg={info['neg_mean']:+.4f}  diff={info['difference']:+.4f}  "
            f"p={info['mann_whitney_p']:.1e} {sig}"
        )

    print("\n--- Loop Statistics ---")
    for label_name in ["positive", "negative"]:
        ls = loop_stats[label_name]
        print(
            f"  {label_name.capitalize():10s}: "
            f"{ls['n_in_loops']}/{ls['n_total']} in loops ({ls['pct_in_loops']:.1f}%)  "
            f"mean loop size={ls['mean_loop_size']:.1f}  "
            f"median={ls['median_loop_size']:.1f}"
        )
        if ls["loop_type_counts"]:
            for lt, cnt in sorted(
                ls["loop_type_counts"].items(), key=lambda x: -x[1]
            ):
                print(f"    {lt:20s}: {cnt}")

    print("\n--- Figures saved to ---")
    for p in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"  {p}")
    print("=" * 72)


if __name__ == "__main__":
    main()
