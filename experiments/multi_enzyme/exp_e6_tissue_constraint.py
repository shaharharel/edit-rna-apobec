#!/usr/bin/env python
"""
E6: Tissue-Specific Evolutionary Constraint

Hypothesis: Editing sites active in many tissues (ubiquitous) are under stronger
evolutionary constraint than tissue-specific sites.

Analyses:
1. Compute tissue breadth per site (number of tissues with editing rate > threshold)
2. Correlate tissue breadth with cross-species conservation (human-chimp)
3. Correlate tissue breadth with gnomAD gene-level constraint (LOEUF, missense Z)
4. Correlate tissue breadth with editability scores
5. Break down by enzyme category
6. Visualize tissue breadth vs conservation metrics
"""

import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT / "experiments" / "multi_enzyme" / "outputs" / "evolutionary" / "e6_tissue_constraint"
OUT.mkdir(parents=True, exist_ok=True)

TISSUE_RATES = PROJECT / "data" / "processed" / "multi_enzyme" / "levanon_tissue_rates.csv"
LEVANON_CAT = PROJECT / "data" / "processed" / "multi_enzyme" / "levanon_all_categories.csv"
CROSS_SPECIES = PROJECT / "experiments" / "multi_enzyme" / "outputs" / "cross_species" / "gb_scoring_human_vs_chimp.csv"
CROSS_SPECIES_SITES = PROJECT / "experiments" / "multi_enzyme" / "outputs" / "cross_species" / "editing_sites_cross_species.csv"
GNOMAD = PROJECT / "data" / "raw" / "gnomad" / "gnomad_v4.1_constraint.tsv"

TISSUE_COLS = None  # will be set after loading

ENZYME_COLORS = {
    "A3A": "#2196F3",
    "A3G": "#4CAF50",
    "A3A_A3G": "#FF9800",
    "Neither": "#9C27B0",
    "Unknown": "#757575",
}


def load_tissue_rates():
    """Load tissue rates and compute tissue breadth metrics."""
    df = pd.read_csv(TISSUE_RATES)
    global TISSUE_COLS
    TISSUE_COLS = [c for c in df.columns if c not in ("site_id", "enzyme_category")]

    # Replace empty strings / missing with NaN, then to numeric
    rate_df = df[TISSUE_COLS].apply(pd.to_numeric, errors="coerce")

    # Tissue breadth: count tissues with rate > 0
    df["tissue_breadth"] = (rate_df > 0).sum(axis=1)
    # Also count tissues with rate > 1% (more stringent)
    df["tissue_breadth_1pct"] = (rate_df > 1.0).sum(axis=1)
    # Mean editing rate across tissues (non-zero only)
    df["mean_rate"] = rate_df.where(rate_df > 0).mean(axis=1)
    # Max editing rate
    df["max_rate"] = rate_df.max(axis=1)
    # Coefficient of variation across tissues
    row_mean = rate_df.where(rate_df > 0).mean(axis=1)
    row_std = rate_df.where(rate_df > 0).std(axis=1)
    df["rate_cv"] = row_std / row_mean

    print(f"Loaded tissue rates: {len(df)} sites, {len(TISSUE_COLS)} tissues")
    print(f"Tissue breadth range: {df['tissue_breadth'].min()}-{df['tissue_breadth'].max()}")
    return df


def load_levanon_categories():
    """Load Levanon site categories with genomic info."""
    df = pd.read_csv(LEVANON_CAT)
    print(f"Loaded Levanon categories: {len(df)} sites")
    print(f"  Enzyme categories: {df['enzyme_category'].value_counts().to_dict()}")
    return df


def load_cross_species():
    """Load cross-species conservation data and merge scoring + site info."""
    scoring = pd.read_csv(CROSS_SPECIES)
    sites = pd.read_csv(CROSS_SPECIES_SITES)

    # Merge scoring and site info on site_id
    cs = scoring.merge(sites[["site_id", "human_chr", "human_pos", "human_strand",
                               "conserved", "sub_rate"]], on="site_id", how="left")

    # Build a coordinate-based key for joining to Levanon (0-based start)
    # human_pos in cross-species is 0-based (matches Levanon 'start')
    cs["coord_key"] = cs["human_chr"] + ":" + cs["human_pos"].astype(str) + ":" + cs["human_strand"]
    print(f"Loaded cross-species data: {len(cs)} ortholog pairs")
    return cs


def load_gnomad_constraint():
    """Load gnomAD gene-level constraint (canonical transcripts only)."""
    df = pd.read_csv(GNOMAD, sep="\t", low_memory=False)
    # Keep canonical transcripts only to avoid duplicates
    df_canon = df[df["canonical"] == True].copy()
    if len(df_canon) == 0:
        df_canon = df[df["canonical"] == "true"].copy()
    if len(df_canon) == 0:
        # Try string comparison
        df_canon = df[df["canonical"].astype(str).str.lower() == "true"].copy()

    # Select key columns
    cols = ["gene", "lof.oe_ci.upper", "lof.pLI", "mis.z_score", "mis.oe", "syn.z_score"]
    existing = [c for c in cols if c in df_canon.columns]
    gnomad = df_canon[existing].copy()
    # Rename for convenience
    gnomad = gnomad.rename(columns={
        "lof.oe_ci.upper": "LOEUF",
        "lof.pLI": "pLI",
        "mis.z_score": "mis_z",
        "mis.oe": "mis_oe",
        "syn.z_score": "syn_z",
    })
    # Convert to numeric
    for c in ["LOEUF", "pLI", "mis_z", "mis_oe", "syn_z"]:
        if c in gnomad.columns:
            gnomad[c] = pd.to_numeric(gnomad[c], errors="coerce")

    gnomad = gnomad.dropna(subset=["LOEUF"])
    print(f"Loaded gnomAD constraint: {len(gnomad)} genes (canonical)")
    return gnomad


def build_merged_dataset(tissue_df, levanon_df, cs_df, gnomad_df):
    """Build the main analysis dataframe: one row per Levanon site."""
    # Start with tissue data
    merged = tissue_df.copy()

    # Merge Levanon categories (drop enzyme_category from levanon since tissue_df already has it)
    cat_cols = ["site_id", "chr", "start", "end", "gene_refseq",
                "genomic_category", "exonic_function", "strand",
                "edited_in_num_tissues", "tissue_classification"]
    cat_cols = [c for c in cat_cols if c in levanon_df.columns]
    merged = merged.merge(levanon_df[cat_cols], on="site_id", how="left")

    # Build cross-species join key using Levanon's 0-based start coordinate
    merged["cs_key"] = merged["chr"] + ":" + merged["start"].astype(str) + ":" + merged["strand"]

    # Merge cross-species (scoring + conservation) via coord_key (0-based)
    cs_cols = ["coord_key", "human_gb_score", "chimp_gb_score", "score_diff", "score_ratio",
               "conserved", "sub_rate", "motif_preserved"]
    cs_merge = cs_df[cs_cols].rename(columns={"coord_key": "cs_key"})
    merged = merged.merge(cs_merge, on="cs_key", how="left")

    n_cs = merged["human_gb_score"].notna().sum()
    print(f"Cross-species match: {n_cs}/{len(merged)} sites have ortholog data")

    # Merge gnomAD on gene name — drop duplicates in gnomAD first (keep first per gene)
    gnomad_dedup = gnomad_df.drop_duplicates(subset=["gene"], keep="first")
    merged = merged.merge(gnomad_dedup, left_on="gene_refseq", right_on="gene", how="left")
    n_gn = merged["LOEUF"].notna().sum()
    print(f"gnomAD match: {n_gn}/{len(merged)} sites have gene constraint data")

    # Editability conservation: abs(score_diff), score_ratio
    merged["abs_score_diff"] = merged["score_diff"].abs()

    return merged


# ── Analysis Functions ─────────────────────────────────────────────────────

def analysis_1_tissue_breadth_distribution(df):
    """Characterize tissue breadth across sites and enzyme categories."""
    print("\n" + "=" * 70)
    print("ANALYSIS 1: Tissue Breadth Distribution")
    print("=" * 70)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1a: Overall distribution
    ax = axes[0]
    ax.hist(df["tissue_breadth"], bins=30, color="#2196F3", edgecolor="white", alpha=0.8)
    ax.set_xlabel("Tissue Breadth (# tissues with rate > 0)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Tissue Breadth")
    ax.axvline(df["tissue_breadth"].median(), color="red", ls="--", label=f"Median={df['tissue_breadth'].median():.0f}")
    ax.legend()

    # 1b: By enzyme category
    ax = axes[1]
    cats = [c for c in ["A3A", "A3G", "A3A_A3G", "Neither", "Unknown"]
            if c in df["enzyme_category"].unique()]
    data_by_cat = [df[df["enzyme_category"] == c]["tissue_breadth"].values for c in cats]
    bp = ax.boxplot(data_by_cat, labels=cats, patch_artist=True)
    for patch, cat in zip(bp["boxes"], cats):
        patch.set_facecolor(ENZYME_COLORS.get(cat, "#999"))
        patch.set_alpha(0.7)
    ax.set_ylabel("Tissue Breadth")
    ax.set_title("Tissue Breadth by Enzyme")
    ax.tick_params(axis="x", rotation=30)

    # 1c: Breadth vs mean rate
    ax = axes[2]
    for cat in cats:
        sub = df[df["enzyme_category"] == cat]
        ax.scatter(sub["tissue_breadth"], sub["mean_rate"], alpha=0.5, s=20,
                   color=ENZYME_COLORS.get(cat, "#999"), label=cat)
    ax.set_xlabel("Tissue Breadth")
    ax.set_ylabel("Mean Editing Rate (%)")
    ax.set_title("Breadth vs Mean Rate")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(OUT / "tissue_breadth_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Summary stats
    for cat in cats:
        sub = df[df["enzyme_category"] == cat]
        print(f"  {cat}: n={len(sub)}, median breadth={sub['tissue_breadth'].median():.0f}, "
              f"mean breadth={sub['tissue_breadth'].mean():.1f}")

    # Kruskal-Wallis test
    if len(data_by_cat) >= 2:
        h_stat, p_val = stats.kruskal(*[d for d in data_by_cat if len(d) > 0])
        print(f"  Kruskal-Wallis: H={h_stat:.2f}, p={p_val:.2e}")


def _safe_spearman(x, y):
    """Spearman correlation with NaN guard."""
    mask = x.notna() & y.notna()
    if mask.sum() < 5:
        return float("nan"), float("nan")
    return stats.spearmanr(x[mask], y[mask])


def analysis_2_breadth_vs_conservation(df):
    """Correlate tissue breadth with cross-species conservation."""
    print("\n" + "=" * 70)
    print("ANALYSIS 2: Tissue Breadth vs Cross-Species Conservation")
    print("=" * 70)

    cs_df = df.dropna(subset=["human_gb_score"])
    n_cs = len(cs_df)
    print(f"  Sites with cross-species data: {n_cs}")

    if n_cs < 10:
        print("  WARNING: Too few sites with cross-species data for robust analysis.")

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))

    def _scatter_corr(ax, data, xcol, ycol, xlabel, ylabel, color):
        sub = data.dropna(subset=[ycol])
        if len(sub) < 3:
            ax.set_title(f"{ylabel}: n={len(sub)}, too few")
            return
        ax.scatter(sub[xcol], sub[ycol], alpha=0.5, s=25, color=color)
        r, p = _safe_spearman(sub[xcol], sub[ycol])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"Breadth vs {ylabel}\nSpearman r={r:.3f}, p={p:.2e}" if not np.isnan(r)
                      else f"Breadth vs {ylabel}\nn={len(sub)}")
        print(f"  Breadth vs {ycol}: Spearman r={r:.3f}, p={p:.2e}, n={len(sub)}")

    _scatter_corr(axes[0, 0], cs_df, "tissue_breadth", "sub_rate",
                  "Tissue Breadth", "Substitution Rate", "#2196F3")
    _scatter_corr(axes[0, 1], cs_df, "tissue_breadth", "score_ratio",
                  "Tissue Breadth", "Editability Ratio (chimp/human)", "#4CAF50")
    _scatter_corr(axes[0, 2], cs_df, "tissue_breadth", "abs_score_diff",
                  "Tissue Breadth", "|Score Diff|", "#FF9800")

    # 2d: Motif preservation by breadth tertile
    ax = axes[1, 0]
    if n_cs >= 10:
        cs_df_motif = cs_df.copy()
        try:
            cs_df_motif["breadth_tertile"] = pd.qcut(cs_df_motif["tissue_breadth"], q=3,
                                                      labels=["Low", "Mid", "High"],
                                                      duplicates="drop")
            motif_by_tertile = cs_df_motif.groupby("breadth_tertile")["motif_preserved"].mean()
            colors_t = ["#EF5350", "#FFC107", "#66BB6A"][:len(motif_by_tertile)]
            bars = ax.bar(motif_by_tertile.index, motif_by_tertile.values, color=colors_t)
            ax.set_ylabel("Fraction Motif Preserved")
            ax.set_xlabel("Tissue Breadth Tertile")
            ax.set_title("Motif Conservation by Breadth")
            for bar, val in zip(bars, motif_by_tertile.values):
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.002, f"{val:.3f}",
                        ha="center", va="bottom", fontsize=10)
            print(f"  Motif preservation by tertile: {motif_by_tertile.to_dict()}")
        except ValueError:
            ax.set_title("Motif Conservation: insufficient variation")
    else:
        ax.set_title("Motif Conservation: too few sites")

    # 2e: Editability score (human) vs breadth
    _scatter_corr(axes[1, 1], cs_df, "tissue_breadth", "human_gb_score",
                  "Tissue Breadth", "Human GB Editability Score", "#9C27B0")

    # 2f: Conservation (binary) vs breadth
    ax = axes[1, 2]
    sub = cs_df.dropna(subset=["conserved"])
    if len(sub) >= 5:
        sub = sub.copy()
        sub["conserved_label"] = sub["conserved"].map({True: "Conserved", False: "Diverged"})
        for label, color in [("Conserved", "#66BB6A"), ("Diverged", "#EF5350")]:
            vals = sub[sub["conserved_label"] == label]["tissue_breadth"]
            if len(vals) > 0:
                ax.hist(vals, bins=15, alpha=0.6, label=f"{label} (n={len(vals)})", color=color)
        ax.set_xlabel("Tissue Breadth")
        ax.set_ylabel("Count")
        ax.set_title("Breadth: Conserved vs Diverged Sites")
        ax.legend()
        conserved_breadth = sub[sub["conserved"] == True]["tissue_breadth"]
        diverged_breadth = sub[sub["conserved"] == False]["tissue_breadth"]
        if len(conserved_breadth) > 1 and len(diverged_breadth) > 1:
            u, p = stats.mannwhitneyu(conserved_breadth, diverged_breadth, alternative="two-sided")
            print(f"  Conserved vs Diverged breadth: Mann-Whitney p={p:.2e}, "
                  f"median conserved={conserved_breadth.median():.0f} vs diverged={diverged_breadth.median():.0f}")
    else:
        ax.set_title("Conservation comparison: too few sites")

    plt.tight_layout()
    plt.savefig(OUT / "breadth_vs_conservation.png", dpi=150, bbox_inches="tight")
    plt.close()


def analysis_3_breadth_vs_gnomad(df):
    """Correlate tissue breadth with gnomAD gene-level constraint."""
    print("\n" + "=" * 70)
    print("ANALYSIS 3: Tissue Breadth vs gnomAD Gene Constraint")
    print("=" * 70)

    gn_df = df.dropna(subset=["LOEUF"])
    print(f"  Sites with gnomAD data: {len(gn_df)}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    metrics = [("LOEUF", "LOEUF (lower = more constrained)"),
               ("pLI", "pLI (probability of LoF intolerance)"),
               ("mis_z", "Missense Z-score (higher = more constrained)")]

    for ax, (col, label) in zip(axes, metrics):
        sub = gn_df.dropna(subset=[col])
        if len(sub) < 10:
            ax.set_title(f"{col}: too few points")
            continue
        ax.scatter(sub["tissue_breadth"], sub[col], alpha=0.4, s=15, color="#2196F3")
        r, p = stats.spearmanr(sub["tissue_breadth"], sub[col])
        ax.set_xlabel("Tissue Breadth")
        ax.set_ylabel(label)
        ax.set_title(f"Breadth vs {col}\nSpearman r={r:.3f}, p={p:.2e}")
        print(f"  Breadth vs {col}: Spearman r={r:.3f}, p={p:.2e} (n={len(sub)})")

    plt.tight_layout()
    plt.savefig(OUT / "breadth_vs_gnomad.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Tertile analysis
    gn_df = gn_df.copy()
    gn_df["breadth_tertile"] = pd.qcut(gn_df["tissue_breadth"], q=3,
                                         labels=["Low", "Mid", "High"],
                                         duplicates="drop")
    print("\n  gnomAD constraint by tissue breadth tertile:")
    for metric in ["LOEUF", "pLI", "mis_z"]:
        if metric in gn_df.columns:
            summary = gn_df.groupby("breadth_tertile")[metric].agg(["median", "mean", "count"])
            print(f"\n  {metric}:")
            print(summary.to_string(index=True))


def analysis_4_enzyme_breakdown(df):
    """Break down breadth-conservation relationship by enzyme category."""
    print("\n" + "=" * 70)
    print("ANALYSIS 4: Enzyme-Specific Breadth-Conservation Patterns")
    print("=" * 70)

    cs_df = df.dropna(subset=["human_gb_score"])
    cats = [c for c in ["A3A", "A3G", "A3A_A3G", "Neither", "Unknown"]
            if c in cs_df["enzyme_category"].unique()]

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))

    results = {}
    for idx, cat in enumerate(cats):
        row, col = divmod(idx, 3)
        ax = axes[row, col]
        sub = cs_df[cs_df["enzyme_category"] == cat]
        if len(sub) < 5:
            ax.set_title(f"{cat}: n={len(sub)}, too few")
            continue

        ax.scatter(sub["tissue_breadth"], sub["score_ratio"], alpha=0.5, s=25,
                   color=ENZYME_COLORS.get(cat, "#999"))

        r, p = stats.spearmanr(sub["tissue_breadth"], sub["score_ratio"])
        ax.set_xlabel("Tissue Breadth")
        ax.set_ylabel("Editability Ratio (chimp/human)")
        ax.set_title(f"{cat} (n={len(sub)})\nSpearman r={r:.3f}, p={p:.2e}")
        results[cat] = {"n": len(sub), "spearman_r": float(r), "spearman_p": float(p),
                        "median_breadth": float(sub["tissue_breadth"].median())}
        print(f"  {cat}: n={len(sub)}, breadth vs score_ratio: r={r:.3f}, p={p:.2e}")

    # Use remaining subplot for summary bar chart
    if len(cats) < 6:
        ax = axes[1, 2] if len(cats) <= 5 else axes[1, 2]
        cat_names = list(results.keys())
        r_vals = [results[c]["spearman_r"] for c in cat_names]
        colors = [ENZYME_COLORS.get(c, "#999") for c in cat_names]
        bars = ax.bar(cat_names, r_vals, color=colors, alpha=0.8)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_ylabel("Spearman r (breadth vs editability ratio)")
        ax.set_title("Breadth-Conservation Correlation\nby Enzyme")
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(OUT / "enzyme_breakdown.png", dpi=150, bbox_inches="tight")
    plt.close()

    return results


def analysis_5_breadth_bins_comprehensive(df):
    """Comprehensive comparison: ubiquitous vs tissue-specific sites."""
    print("\n" + "=" * 70)
    print("ANALYSIS 5: Ubiquitous vs Tissue-Specific Sites")
    print("=" * 70)

    # Define bins
    df = df.copy()
    q_low = df["tissue_breadth"].quantile(0.25)
    q_high = df["tissue_breadth"].quantile(0.75)
    df["breadth_group"] = "Mid"
    df.loc[df["tissue_breadth"] <= q_low, "breadth_group"] = "Tissue-specific"
    df.loc[df["tissue_breadth"] >= q_high, "breadth_group"] = "Ubiquitous"

    print(f"  Tissue-specific (breadth <= {q_low:.0f}): n={len(df[df['breadth_group']=='Tissue-specific'])}")
    print(f"  Mid ({q_low:.0f} < breadth < {q_high:.0f}): n={len(df[df['breadth_group']=='Mid'])}")
    print(f"  Ubiquitous (breadth >= {q_high:.0f}): n={len(df[df['breadth_group']=='Ubiquitous'])}")

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    group_colors = {"Tissue-specific": "#EF5350", "Mid": "#FFC107", "Ubiquitous": "#66BB6A"}
    groups = ["Tissue-specific", "Mid", "Ubiquitous"]

    # 5a: Editability score
    ax = axes[0, 0]
    data = [df[df["breadth_group"] == g]["human_gb_score"].dropna().values for g in groups]
    bp = ax.boxplot(data, labels=groups, patch_artist=True)
    for patch, g in zip(bp["boxes"], groups):
        patch.set_facecolor(group_colors[g])
        patch.set_alpha(0.7)
    ax.set_ylabel("Human GB Editability Score")
    ax.set_title("Editability by Breadth Group")
    if len(data[0]) > 0 and len(data[2]) > 0:
        u, p = stats.mannwhitneyu(data[0], data[2], alternative="two-sided")
        ax.text(0.5, 0.95, f"Spec. vs Ubiq.: p={p:.2e}", transform=ax.transAxes,
                ha="center", va="top", fontsize=9)
        print(f"  Editability: tissue-spec median={np.nanmedian(data[0]):.3f}, "
              f"ubiq median={np.nanmedian(data[2]):.3f}, MW p={p:.2e}")

    # 5b: Score ratio (editability conservation)
    ax = axes[0, 1]
    data = [df[df["breadth_group"] == g]["score_ratio"].dropna().values for g in groups]
    bp = ax.boxplot(data, labels=groups, patch_artist=True)
    for patch, g in zip(bp["boxes"], groups):
        patch.set_facecolor(group_colors[g])
        patch.set_alpha(0.7)
    ax.set_ylabel("Editability Ratio (chimp/human)")
    ax.set_title("Editability Conservation")
    if len(data[0]) > 0 and len(data[2]) > 0:
        u, p = stats.mannwhitneyu(data[0], data[2], alternative="two-sided")
        ax.text(0.5, 0.95, f"Spec. vs Ubiq.: p={p:.2e}", transform=ax.transAxes,
                ha="center", va="top", fontsize=9)
        print(f"  Score ratio: tissue-spec median={np.nanmedian(data[0]):.3f}, "
              f"ubiq median={np.nanmedian(data[2]):.3f}, MW p={p:.2e}")

    # 5c: Substitution rate
    ax = axes[0, 2]
    data = [df[df["breadth_group"] == g]["sub_rate"].dropna().values for g in groups]
    bp = ax.boxplot(data, labels=groups, patch_artist=True)
    for patch, g in zip(bp["boxes"], groups):
        patch.set_facecolor(group_colors[g])
        patch.set_alpha(0.7)
    ax.set_ylabel("Substitution Rate (human-chimp)")
    ax.set_title("Sequence Divergence")
    if len(data[0]) > 0 and len(data[2]) > 0:
        u, p = stats.mannwhitneyu(data[0], data[2], alternative="two-sided")
        ax.text(0.5, 0.95, f"Spec. vs Ubiq.: p={p:.2e}", transform=ax.transAxes,
                ha="center", va="top", fontsize=9)
        print(f"  Sub rate: tissue-spec median={np.nanmedian(data[0]):.4f}, "
              f"ubiq median={np.nanmedian(data[2]):.4f}, MW p={p:.2e}")

    # 5d: LOEUF
    ax = axes[1, 0]
    data = [df[df["breadth_group"] == g]["LOEUF"].dropna().values for g in groups]
    bp = ax.boxplot(data, labels=groups, patch_artist=True)
    for patch, g in zip(bp["boxes"], groups):
        patch.set_facecolor(group_colors[g])
        patch.set_alpha(0.7)
    ax.set_ylabel("LOEUF (lower = more constrained)")
    ax.set_title("LoF Constraint")
    if len(data[0]) > 0 and len(data[2]) > 0:
        u, p = stats.mannwhitneyu(data[0], data[2], alternative="two-sided")
        ax.text(0.5, 0.95, f"Spec. vs Ubiq.: p={p:.2e}", transform=ax.transAxes,
                ha="center", va="top", fontsize=9)
        print(f"  LOEUF: tissue-spec median={np.nanmedian(data[0]):.3f}, "
              f"ubiq median={np.nanmedian(data[2]):.3f}, MW p={p:.2e}")

    # 5e: Missense Z
    ax = axes[1, 1]
    data = [df[df["breadth_group"] == g]["mis_z"].dropna().values for g in groups]
    bp = ax.boxplot(data, labels=groups, patch_artist=True)
    for patch, g in zip(bp["boxes"], groups):
        patch.set_facecolor(group_colors[g])
        patch.set_alpha(0.7)
    ax.set_ylabel("Missense Z-score")
    ax.set_title("Missense Constraint")
    if len(data[0]) > 0 and len(data[2]) > 0:
        u, p = stats.mannwhitneyu(data[0], data[2], alternative="two-sided")
        ax.text(0.5, 0.95, f"Spec. vs Ubiq.: p={p:.2e}", transform=ax.transAxes,
                ha="center", va="top", fontsize=9)
        print(f"  Missense Z: tissue-spec median={np.nanmedian(data[0]):.3f}, "
              f"ubiq median={np.nanmedian(data[2]):.3f}, MW p={p:.2e}")

    # 5f: Enzyme composition
    ax = axes[1, 2]
    enzyme_comp = df.groupby(["breadth_group", "enzyme_category"]).size().unstack(fill_value=0)
    enzyme_comp_pct = enzyme_comp.div(enzyme_comp.sum(axis=1), axis=0)
    enzyme_comp_pct = enzyme_comp_pct.reindex(groups)
    enzyme_cols_ordered = [c for c in ["A3A", "A3G", "A3A_A3G", "Neither", "Unknown"] if c in enzyme_comp_pct.columns]
    enzyme_comp_pct[enzyme_cols_ordered].plot(kind="bar", stacked=True, ax=ax,
                                              color=[ENZYME_COLORS[c] for c in enzyme_cols_ordered])
    ax.set_ylabel("Fraction")
    ax.set_title("Enzyme Composition")
    ax.legend(fontsize=7, loc="upper right")
    ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(OUT / "ubiquitous_vs_specific.png", dpi=150, bbox_inches="tight")
    plt.close()

    return df


def analysis_6_heatmap_summary(df):
    """Correlation heatmap: all breadth + conservation + constraint metrics."""
    print("\n" + "=" * 70)
    print("ANALYSIS 6: Correlation Heatmap")
    print("=" * 70)

    cols = ["tissue_breadth", "tissue_breadth_1pct", "mean_rate", "max_rate", "rate_cv",
            "human_gb_score", "score_ratio", "abs_score_diff", "sub_rate",
            "LOEUF", "pLI", "mis_z"]
    cols = [c for c in cols if c in df.columns]
    sub = df[cols].dropna(how="all")

    # Spearman correlation
    corr_matrix = sub.corr(method="spearman")

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, ax=ax, square=True)
    ax.set_title("Spearman Correlations: Tissue Breadth, Conservation & Constraint")
    plt.tight_layout()
    plt.savefig(OUT / "correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Print key correlations
    for c in ["score_ratio", "sub_rate", "LOEUF", "mis_z", "human_gb_score"]:
        if c in corr_matrix.columns:
            r = corr_matrix.loc["tissue_breadth", c]
            print(f"  tissue_breadth vs {c}: Spearman r={r:.3f}")


def save_results(df, enzyme_results):
    """Save results to JSON and CSV."""
    # Summary CSV
    out_cols = ["site_id", "enzyme_category", "tissue_breadth", "tissue_breadth_1pct",
                "mean_rate", "max_rate", "rate_cv",
                "human_gb_score", "chimp_gb_score", "score_ratio", "abs_score_diff",
                "sub_rate", "conserved", "motif_preserved",
                "LOEUF", "pLI", "mis_z", "gene_refseq"]
    out_cols = [c for c in out_cols if c in df.columns]
    df[out_cols].to_csv(OUT / "e6_tissue_constraint_data.csv", index=False)

    # Key correlations
    cs_df = df.dropna(subset=["score_ratio"])
    gn_df = df.dropna(subset=["LOEUF"])

    results = {
        "experiment": "E6_tissue_constraint",
        "n_sites_total": len(df),
        "n_sites_cross_species": int(df["human_gb_score"].notna().sum()),
        "n_sites_gnomad": int(df["LOEUF"].notna().sum()),
        "correlations": {},
        "enzyme_breakdown": enzyme_results,
    }

    for col in ["score_ratio", "sub_rate", "abs_score_diff", "human_gb_score"]:
        sub = df.dropna(subset=[col])
        if len(sub) >= 10:
            r, p = stats.spearmanr(sub["tissue_breadth"], sub[col])
            results["correlations"][f"breadth_vs_{col}"] = {
                "spearman_r": float(r), "p_value": float(p), "n": len(sub)
            }

    for col in ["LOEUF", "pLI", "mis_z"]:
        sub = df.dropna(subset=[col])
        if len(sub) >= 10:
            r, p = stats.spearmanr(sub["tissue_breadth"], sub[col])
            results["correlations"][f"breadth_vs_{col}"] = {
                "spearman_r": float(r), "p_value": float(p), "n": len(sub)
            }

    # Ubiquitous vs tissue-specific comparison
    q_low = df["tissue_breadth"].quantile(0.25)
    q_high = df["tissue_breadth"].quantile(0.75)
    spec = df[df["tissue_breadth"] <= q_low]
    ubiq = df[df["tissue_breadth"] >= q_high]

    group_tests = {}
    for col in ["score_ratio", "LOEUF", "mis_z", "human_gb_score", "sub_rate"]:
        s = spec[col].dropna()
        u = ubiq[col].dropna()
        if len(s) >= 5 and len(u) >= 5:
            stat, p = stats.mannwhitneyu(s, u, alternative="two-sided")
            group_tests[col] = {
                "tissue_specific_median": float(s.median()),
                "ubiquitous_median": float(u.median()),
                "mann_whitney_p": float(p),
                "n_specific": len(s),
                "n_ubiquitous": len(u),
            }
    results["ubiquitous_vs_specific"] = group_tests

    with open(OUT / "e6_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Saved results to {OUT / 'e6_results.json'}")
    print(f"  Saved data to {OUT / 'e6_tissue_constraint_data.csv'}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("E6: Tissue-Specific Evolutionary Constraint")
    print("=" * 70)

    # Load data
    tissue_df = load_tissue_rates()
    levanon_df = load_levanon_categories()
    cs_df = load_cross_species()
    gnomad_df = load_gnomad_constraint()

    # Build merged dataset
    df = build_merged_dataset(tissue_df, levanon_df, cs_df, gnomad_df)

    # Run analyses
    analysis_1_tissue_breadth_distribution(df)
    analysis_2_breadth_vs_conservation(df)
    analysis_3_breadth_vs_gnomad(df)
    enzyme_results = analysis_4_enzyme_breakdown(df)
    analysis_5_breadth_bins_comprehensive(df)
    analysis_6_heatmap_summary(df)

    # Save
    save_results(df, enzyme_results)

    print("\n" + "=" * 70)
    print("E6 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
