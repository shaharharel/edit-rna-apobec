#!/usr/bin/env python
"""Replot ClinVar visualizations from saved scores CSV.

Generates improved visualizations without re-running the full 6-hour pipeline:
  1. Enrichment ratio bar chart (fold-enrichment of P>=0.5 per category vs background)
  2. Fraction above threshold bar chart (% of variants with P>=t per category)
  3. Rank percentile violin/box plots (rank all 1.68M variants, show distribution per category)
  4. Survival curves (1-CDF, Pathogenic line visually ABOVE Benign)
  5. Known editing sites bar chart (normalized by total per category)
  6. Updated score distributions
  7. Ranking metrics added to results JSON
  8. Calibrated OR comparison (original vs prior-calibrated thresholds)

Usage:
    conda run -n quris python experiments/apobec/replot_clinvar.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, rankdata, fisher_exact

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "clinvar_prediction"
SCORES_CSV = OUTPUT_DIR / "clinvar_all_scores.csv"
RESULTS_JSON = OUTPUT_DIR / "clinvar_prediction_results.json"
CALIBRATED_JSON = Path(__file__).resolve().parent / "outputs" / "clinvar_calibrated" / "calibrated_enrichment_results.json"
SPLITS_A3A = PROJECT_ROOT / "data" / "processed" / "splits_expanded_a3a.csv"
NEG_TIER1 = PROJECT_ROOT / "data" / "processed" / "negatives_tier1.csv"

CATEGORIES = ["Pathogenic", "Likely_pathogenic", "VUS", "Conflicting",
              "Likely_benign", "Benign"]
COLORS = {"Pathogenic": "#d32f2f", "Likely_pathogenic": "#ff7043",
          "VUS": "#ffb74d", "Conflicting": "#90a4ae",
          "Likely_benign": "#81c784", "Benign": "#43a047",
          "Other": "#bdbdbd"}
NICE_NAMES = {"Pathogenic": "Pathogenic", "Likely_pathogenic": "Likely\npathogenic",
              "VUS": "VUS", "Conflicting": "Conflicting",
              "Likely_benign": "Likely\nbenign", "Benign": "Benign"}


def main():
    print(f"Loading scores from {SCORES_CSV} ...")
    df = pd.read_csv(SCORES_CSV)
    print(f"  Loaded {len(df):,} variants")

    # Build cat_scores dict
    cat_scores = {}
    plot_cats = []
    for cat in CATEGORIES:
        mask = df["significance_simple"] == cat
        if mask.sum() > 0:
            cat_scores[cat] = {
                "gb": df.loc[mask, "p_edited_gb"].values,
                "rf": df.loc[mask, "p_edited_rnasee"].values,
            }
            plot_cats.append(cat)
            print(f"  {cat}: n={mask.sum():,}, GB mean={cat_scores[cat]['gb'].mean():.4f}, "
                  f"RF mean={cat_scores[cat]['rf'].mean():.4f}")

    known = df[df["is_known_editing_site"] == True].copy()
    print(f"  Known editing sites: {len(known)}")

    # --- Compute ranking metrics ---
    print("\nComputing rank percentiles ...")
    ranking_metrics = compute_ranking_metrics(df, cat_scores, plot_cats)

    # --- Generate all plots ---
    print("\nGenerating plots ...")

    plot_cdf_all(cat_scores, plot_cats)
    plot_cdf_path_vs_benign(cat_scores)
    plot_known_sites_normalized(known, df)

    # New visualizations
    or_results = plot_or_comparison(df)
    threshold_results = plot_enrichment_by_threshold(df)
    plot_path_ratio_by_method(df)

    # Calibrated OR comparison
    calibrated_results = plot_calibrated_or_comparison(df)

    # --- Update results JSON ---
    print("\nUpdating results JSON with ranking metrics ...")
    with open(RESULTS_JSON) as f:
        results = json.load(f)
    results["ranking_metrics"] = ranking_metrics
    results["or_comparison"] = or_results
    results["enrichment_by_threshold"] = threshold_results
    if calibrated_results:
        results["calibrated_or_comparison"] = calibrated_results
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Updated {RESULTS_JSON}")

    print("\nDone! All plots saved to", OUTPUT_DIR)


def compute_ranking_metrics(df, cat_scores, plot_cats):
    """Compute mean rank percentile per category for both models."""
    # Rank all variants (higher score = higher rank percentile)
    n = len(df)
    gb_ranks = rankdata(df["p_edited_gb"].values) / n * 100  # percentile
    rf_ranks = rankdata(df["p_edited_rnasee"].values) / n * 100

    metrics = {}
    for cat in plot_cats:
        mask = (df["significance_simple"] == cat).values
        metrics[cat] = {
            "n": int(mask.sum()),
            "gb_mean_rank_pctl": float(np.mean(gb_ranks[mask])),
            "gb_median_rank_pctl": float(np.median(gb_ranks[mask])),
            "rf_mean_rank_pctl": float(np.mean(rf_ranks[mask])),
            "rf_median_rank_pctl": float(np.median(rf_ranks[mask])),
        }
        print(f"  {cat} (n={mask.sum():,}): "
              f"GB mean pctl={metrics[cat]['gb_mean_rank_pctl']:.1f}, "
              f"RF mean pctl={metrics[cat]['rf_mean_rank_pctl']:.1f}")

    return metrics


def plot_enrichment_ratio(cat_scores, plot_cats, df):
    """Bar chart showing fold-enrichment of P>=0.5 per category vs background rate."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Background rate (all variants)
    bg_gb = (df["p_edited_gb"] >= 0.5).mean()
    bg_rf = (df["p_edited_rnasee"] >= 0.5).mean()

    for ax_idx, (model_key, bg_rate, model_name) in enumerate([
        ("gb", bg_gb, "GB_Full"), ("rf", bg_rf, "RNAsee_RF")
    ]):
        ratios = []
        cats_plot = []
        for cat in plot_cats:
            frac = (cat_scores[cat][model_key] >= 0.5).mean()
            ratio = frac / bg_rate if bg_rate > 0 else 0
            ratios.append(ratio)
            cats_plot.append(cat)

        x = np.arange(len(cats_plot))
        bars = axes[ax_idx].bar(x, ratios,
                                color=[COLORS.get(c, "#bdbdbd") for c in cats_plot],
                                edgecolor="white", linewidth=0.5)
        axes[ax_idx].axhline(1.0, color="black", linestyle="--", alpha=0.5, linewidth=1)
        axes[ax_idx].set_xticks(x)
        axes[ax_idx].set_xticklabels([NICE_NAMES.get(c, c) for c in cats_plot],
                                     fontsize=9)
        axes[ax_idx].set_ylabel("Fold Enrichment vs Background")
        axes[ax_idx].set_title(f"{model_name}: Editing Score Enrichment by Clinical Significance")
        # Annotate bars
        for i, (bar, ratio) in enumerate(zip(bars, ratios)):
            axes[ax_idx].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                              f"{ratio:.2f}x", ha="center", va="bottom", fontsize=9,
                              fontweight="bold")
        axes[ax_idx].set_ylim(0, max(ratios) * 1.2)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "enrichment_ratio.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved enrichment_ratio.png")


def plot_fraction_above_threshold(cat_scores, plot_cats):
    """Bar chart showing fraction of variants with P>=threshold per category."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax_idx, (model_key, model_name) in enumerate([("gb", "GB_Full"), ("rf", "RNAsee_RF")]):
        # Multiple thresholds
        thresholds = [0.3, 0.5, 0.7]
        x = np.arange(len(plot_cats))
        width = 0.25

        for t_idx, threshold in enumerate(thresholds):
            fracs = []
            for cat in plot_cats:
                frac = (cat_scores[cat][model_key] >= threshold).mean() * 100
                fracs.append(frac)
            bars = axes[ax_idx].bar(x + (t_idx - 1) * width, fracs, width,
                                    alpha=0.8, label=f"P ≥ {threshold}",
                                    edgecolor="white", linewidth=0.5)

        axes[ax_idx].set_xticks(x)
        axes[ax_idx].set_xticklabels([NICE_NAMES.get(c, c) for c in plot_cats], fontsize=9)
        axes[ax_idx].set_ylabel("% of Variants Above Threshold")
        axes[ax_idx].set_title(f"{model_name}: Fraction Above Threshold")
        axes[ax_idx].legend(fontsize=9)
        axes[ax_idx].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fraction_above_threshold.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved fraction_above_threshold.png")


def plot_rank_percentile(df, plot_cats):
    """Violin/box plots of rank percentiles per clinical significance category."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    n = len(df)
    gb_ranks = rankdata(df["p_edited_gb"].values) / n * 100
    rf_ranks = rankdata(df["p_edited_rnasee"].values) / n * 100

    for ax_idx, (ranks, model_name) in enumerate([(gb_ranks, "GB_Full"), (rf_ranks, "RNAsee_RF")]):
        data_list = []
        labels = []
        cat_colors = []
        for cat in plot_cats:
            mask = (df["significance_simple"] == cat).values
            data_list.append(ranks[mask])
            labels.append(NICE_NAMES.get(cat, cat))
            cat_colors.append(COLORS.get(cat, "#bdbdbd"))

        vp = axes[ax_idx].violinplot(data_list, positions=range(len(plot_cats)),
                                     showmeans=True, showmedians=True, showextrema=False)
        for i, body in enumerate(vp["bodies"]):
            body.set_facecolor(cat_colors[i])
            body.set_alpha(0.6)
        if "cmeans" in vp:
            vp["cmeans"].set_color("black")
            vp["cmeans"].set_linewidth(1.5)
        if "cmedians" in vp:
            vp["cmedians"].set_color("red")
            vp["cmedians"].set_linewidth(1.5)

        axes[ax_idx].axhline(50, color="gray", linestyle="--", alpha=0.5, linewidth=1)
        axes[ax_idx].set_xticks(range(len(plot_cats)))
        axes[ax_idx].set_xticklabels(labels, fontsize=9)
        axes[ax_idx].set_ylabel("Rank Percentile (among all 1.68M variants)")
        axes[ax_idx].set_title(f"{model_name}: Score Rank by Clinical Significance")
        axes[ax_idx].grid(axis="y", alpha=0.3)

        # Annotate means
        for i, cat in enumerate(plot_cats):
            mask = (df["significance_simple"] == cat).values
            mean_pctl = np.mean(ranks[mask])
            n_cat = mask.sum()
            axes[ax_idx].text(i, axes[ax_idx].get_ylim()[1] * 0.98,
                              f"μ={mean_pctl:.1f}\nn={n_cat:,}",
                              ha="center", va="top", fontsize=7, color="black")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rank_percentile.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved rank_percentile.png")


def plot_cdf_all(cat_scores, plot_cats):
    """CDF curves for all significance categories."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax_idx, (model_key, model_name) in enumerate([("gb", "GB_Full"), ("rf", "RNAsee_RF")]):
        for cat in plot_cats:
            sorted_vals = np.sort(cat_scores[cat][model_key])
            cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
            axes[ax_idx].plot(sorted_vals, cdf,
                              label=f"{cat} (n={len(sorted_vals):,})",
                              color=COLORS.get(cat, "#bdbdbd"), linewidth=1.5)

        axes[ax_idx].axvline(0.5, color="red", linestyle="--", alpha=0.5, label="P=0.5")
        axes[ax_idx].set_xlabel(f"{model_name} P(edited)")
        axes[ax_idx].set_ylabel("Cumulative Fraction")
        axes[ax_idx].set_title(f"{model_name}: Score CDF by Significance")
        axes[ax_idx].legend(fontsize=7, loc="lower right")
        axes[ax_idx].grid(True, alpha=0.3)
        axes[ax_idx].set_xlim(0, 1)
        axes[ax_idx].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pathogenicity_enrichment.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved pathogenicity_enrichment.png")


def plot_cdf_path_vs_benign(cat_scores):
    """CDF curves for Pathogenic vs Benign only, clearer comparison."""
    if "Pathogenic" not in cat_scores or "Benign" not in cat_scores:
        print("  SKIP pathogenicity_cdf_focused.png (missing categories)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    focus_cats = ["Pathogenic", "Benign"]

    for ax_idx, (model_key, model_name) in enumerate([("gb", "GB_Full"), ("rf", "RNAsee_RF")]):
        for cat in focus_cats:
            sorted_vals = np.sort(cat_scores[cat][model_key])
            cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
            axes[ax_idx].plot(sorted_vals, cdf,
                              label=f"{cat} (n={len(sorted_vals):,})",
                              color=COLORS[cat], linewidth=2.5)

        # Shade the gap between curves at P=0.5
        axes[ax_idx].axvline(0.5, color="red", linestyle="--", alpha=0.5, label="P=0.5")

        # Annotate the gap at P=0.5
        path_vals = cat_scores["Pathogenic"][model_key]
        ben_vals = cat_scores["Benign"][model_key]
        path_frac_below = (path_vals < 0.5).mean()
        ben_frac_below = (ben_vals < 0.5).mean()
        gap = abs(path_frac_below - ben_frac_below)
        axes[ax_idx].annotate(
            f"Δ = {gap:.3f}",
            xy=(0.5, (path_frac_below + ben_frac_below) / 2),
            xytext=(0.62, (path_frac_below + ben_frac_below) / 2),
            fontsize=11, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
        )

        axes[ax_idx].set_xlabel(f"{model_name} P(edited)")
        axes[ax_idx].set_ylabel("Cumulative Fraction")
        axes[ax_idx].set_title(f"{model_name}: Pathogenic vs Benign CDF")
        axes[ax_idx].legend(fontsize=10, loc="lower right")
        axes[ax_idx].grid(True, alpha=0.3)
        axes[ax_idx].set_xlim(0, 1)
        axes[ax_idx].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pathogenicity_cdf_focused.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved pathogenicity_cdf_focused.png")


def plot_known_sites_normalized(known, df):
    """Known editing sites with normalized clinical significance bar chart."""
    if len(known) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: P(edited) distributions for known sites
    axes[0].hist(known["p_edited_gb"], bins=30, alpha=0.6, color="#1976D2",
                 label="GB_Full", edgecolor="white")
    axes[0].hist(known["p_edited_rnasee"], bins=30, alpha=0.6, color="#F57C00",
                 label="RNAsee_RF", edgecolor="white")
    axes[0].axvline(0.5, color="red", linestyle="--", alpha=0.7, label="P=0.5")
    axes[0].set_xlabel("P(edited)")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Known Editing Sites (n={len(known)})")
    axes[0].legend()

    # Right: Normalized clinical significance breakdown
    sig_order = ["Pathogenic", "Likely_pathogenic", "VUS", "Conflicting",
                 "Likely_benign", "Benign", "Other"]
    nice_labels = ["Pathogenic", "Likely\npathogenic", "VUS", "Conflicting",
                   "Likely\nbenign", "Benign", "Other"]

    # Count known editing sites per category
    known_counts = known["significance_simple"].value_counts()
    # Count ALL ClinVar variants per category
    total_counts = df["significance_simple"].value_counts()

    fracs = []
    annot_texts = []
    for s in sig_order:
        n_known = known_counts.get(s, 0)
        n_total = total_counts.get(s, 0)
        frac = n_known / n_total * 100 if n_total > 0 else 0
        fracs.append(frac)
        annot_texts.append(f"{n_known}/{n_total:,}")

    sig_colors = [COLORS.get(s, "#bdbdbd") for s in sig_order]
    bars = axes[1].bar(range(len(sig_order)), fracs, color=sig_colors, edgecolor="white")
    axes[1].set_xticks(range(len(sig_order)))
    axes[1].set_xticklabels(nice_labels, fontsize=8)
    axes[1].set_ylabel("% of Category That Are Known Editing Sites")
    axes[1].set_title("Known Editing Sites: Fraction per ClinVar Category")

    # Annotate with raw counts
    for i, (bar, txt) in enumerate(zip(bars, annot_texts)):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     txt, ha="center", va="bottom", fontsize=7, color="gray")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "known_editing_sites_clinvar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved known_editing_sites_clinvar.png")


def plot_score_distributions(cat_scores, plot_cats):
    """Updated score distributions boxplots."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax_idx, (model_key, model_name) in enumerate([("gb", "GB_Full"), ("rf", "RNAsee_RF")]):
        plot_data = [cat_scores[c][model_key] for c in plot_cats]
        bp = axes[ax_idx].boxplot(plot_data, labels=[NICE_NAMES.get(c, c) for c in plot_cats],
                                  patch_artist=True, showfliers=False)
        for patch, cat in zip(bp["boxes"], plot_cats):
            patch.set_facecolor(COLORS.get(cat, "#bdbdbd"))
            patch.set_alpha(0.7)
        axes[ax_idx].set_ylabel("P(edited)")
        axes[ax_idx].set_title(f"{model_name}: Editing Probability by Clinical Significance")

        for i, cat in enumerate(plot_cats):
            n = len(cat_scores[cat][model_key])
            axes[ax_idx].text(i + 1, axes[ax_idx].get_ylim()[0] - 0.02,
                              f"n={n:,}", ha="center", fontsize=7, color="gray")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "score_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved score_distributions.png")


def _compute_or_ci(predicted_mask, df, combine_lp_lb=True):
    """Compute odds ratio with 95% CI for predicted vs not-predicted pathogenicity."""
    if combine_lp_lb:
        df_sub = df[df["significance_simple"].isin(
            ["Pathogenic", "Likely_pathogenic", "Benign", "Likely_benign"])].copy()
        path_col = df_sub["significance_simple"].isin(["Pathogenic", "Likely_pathogenic"])
    else:
        df_sub = df[df["significance_simple"].isin(["Pathogenic", "Benign"])].copy()
        path_col = df_sub["significance_simple"] == "Pathogenic"

    pred_mask_sub = predicted_mask.reindex(df_sub.index, fill_value=False)
    p_pred = path_col[pred_mask_sub].sum()
    b_pred = (~path_col)[pred_mask_sub].sum()
    p_rest = path_col[~pred_mask_sub].sum()
    b_rest = (~path_col)[~pred_mask_sub].sum()

    if (p_pred + b_pred) == 0 or (p_rest + b_rest) == 0:
        return None

    table = [[p_pred, b_pred], [p_rest, b_rest]]
    odds, pval = fisher_exact(table)

    # 95% CI for log(OR) using Woolf's method
    log_or = np.log(odds) if odds > 0 else 0
    se = np.sqrt(1/max(p_pred,1) + 1/max(b_pred,1) + 1/max(p_rest,1) + 1/max(b_rest,1))
    ci_low = np.exp(log_or - 1.96 * se)
    ci_high = np.exp(log_or + 1.96 * se)

    return {
        "or": round(odds, 4), "p": float(pval),
        "ci_low": round(ci_low, 4), "ci_high": round(ci_high, 4),
        "n_predicted": int(p_pred + b_pred),
        "path_rate": round(p_pred / (p_pred + b_pred) * 100, 1),
    }


def plot_or_comparison(df):
    """Bar chart of odds ratios with 95% CI for CDS non-syn subset (matching RNAsee methodology).

    Uses pre-computed CDS replication results for Rules, RF, and GB on the non-synonymous
    CDS-mapped subset (~434K variants), which is the correct comparison with RNAsee 2024.
    Falls back to all-variant computation if CDS results are unavailable.
    """
    # Try to load CDS replication results (the correct non-syn subset)
    cds_json = OUTPUT_DIR / "rnasee_cds_replication.json"
    cds_results = None
    if cds_json.exists():
        with open(cds_json) as f:
            cds_results = json.load(f)

    fig, ax = plt.subplots(figsize=(10, 5.5))

    or_results = {}
    labels = []
    ors = []
    ci_lows = []
    ci_highs = []
    bar_colors = []

    if cds_results and "fisher_tests_path_lp_vs_ben_lb" in cds_results:
        # Use CDS non-syn results (correct methodology)
        fisher_tests = cds_results["fisher_tests_path_lp_vs_ben_lb"]
        method_map = {
            "CDS Rules (score>9)": ("#9E9E9E", "CDS Rules\n(score>9)"),
            "RF (p>=0.5)": ("#F57C00", "RF\n(P\u22650.5)"),
            "GB_Full (p>=0.5)": ("#1976D2", "GB_Full\n(P\u22650.5)"),
            "CDS Rules OR RF": ("#7B1FA2", "Rules OR RF\n(P\u22650.5)"),
            "CDS Rules OR GB": ("#00796B", "Rules OR GB\n(P\u22650.5)"),
        }
        for test in fisher_tests:
            lbl = test["label"]
            if lbl not in method_map:
                continue
            color, nice_label = method_map[lbl]
            od = test["odds_ratio"]
            pval = test["p_value"]
            n = test["n_predicted"]
            # Compute 95% CI using Woolf's method
            log_or = np.log(od) if od > 0 else 0
            se = np.sqrt(1/max(test["n_path"],1) + 1/max(test["n_benign"],1)
                         + 1/max(n-test["n_path"],1) + 1/max(n-test["n_benign"],1))
            cl = np.exp(log_or - 1.96 * se)
            ch = np.exp(log_or + 1.96 * se)
            labels.append(nice_label)
            ors.append(od)
            ci_lows.append(cl)
            ci_highs.append(ch)
            bar_colors.append(color)
            or_results[nice_label.replace("\n", " ")] = {
                "or": od, "p": pval, "ci_low": round(cl, 4), "ci_high": round(ch, 4),
                "n_predicted": n, "path_rate": test["path_rate"],
                "source": "CDS non-syn subset",
            }
        subtitle = f"Non-synonymous CDS-mapped variants (n={cds_results.get('n_cds_center_c', '?'):,})"
    else:
        # Fallback: compute on all variants
        methods_data = [
            ("RF\n(P\u22650.5)", df["p_edited_rnasee"] >= 0.5),
            ("GB_Full\n(P\u22650.5)", df["p_edited_gb"] >= 0.5),
            ("RF OR GB\n(P\u22650.5)", (df["p_edited_rnasee"] >= 0.5) | (df["p_edited_gb"] >= 0.5)),
        ]
        for label, mask in methods_data:
            result = _compute_or_ci(mask, df, combine_lp_lb=True)
            if result is None:
                continue
            labels.append(label)
            ors.append(result["or"])
            ci_lows.append(result["ci_low"])
            ci_highs.append(result["ci_high"])
            or_results[label.replace("\n", " ")] = result
            if "GB" in label and "OR" not in label:
                bar_colors.append("#1976D2")
            elif "RF OR" in label:
                bar_colors.append("#7B1FA2")
            else:
                bar_colors.append("#F57C00")
        subtitle = f"All C>U variants (n={len(df):,})"

    x = np.arange(len(labels))
    bars = ax.bar(x, ors, color=bar_colors, edgecolor="white", linewidth=0.5, width=0.5, alpha=0.85)

    # Error bars for 95% CI
    yerr_low = [o - cl for o, cl in zip(ors, ci_lows)]
    yerr_high = [ch - o for o, ch in zip(ors, ci_highs)]
    ax.errorbar(x, ors, yerr=[yerr_low, yerr_high], fmt="none", ecolor="black",
                capsize=5, linewidth=1.5)

    # Reference line at OR=1.0
    ax.axhline(1.0, color="black", linestyle="--", alpha=0.5, linewidth=1)
    ax.text(len(labels) - 0.5, 1.01, "OR = 1.0 (no enrichment)", fontsize=8, alpha=0.6, ha="right")

    # Annotate bars with OR and p-value
    for i, (bar, od, p_label) in enumerate(zip(bars, ors, labels)):
        result = or_results[p_label.replace("\n", " ")]
        pval = result["p"]
        if pval < 1e-10:
            p_str = "p<1e-10"
        elif pval < 0.001:
            p_str = f"p={pval:.1e}"
        else:
            p_str = f"p={pval:.3f}"
        y_pos = max(od, ci_highs[i]) + 0.02
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                f"OR={od:.3f}\n{p_str}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Odds Ratio (Pathogenic+LP / Benign+LB)", fontsize=11)
    ax.set_title(f"Pathogenic Enrichment: Odds Ratios with 95% CI\n{subtitle}", fontsize=12)
    ax.set_ylim(0, max(ci_highs) * 1.35 if ci_highs else 2.0)
    ax.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "or_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved or_comparison.png")

    return or_results


def plot_enrichment_by_threshold(df):
    """Line plot of OR vs prediction threshold for GB vs RF."""
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    results = {"thresholds": thresholds, "gb": {}, "rf": {}}

    for ax_idx, (model_key, model_name, color) in enumerate([
        ("p_edited_gb", "GB_Full", "#1976D2"),
        ("p_edited_rnasee", "RNAsee_RF", "#F57C00"),
    ]):
        ors_list = []
        ci_lows_list = []
        ci_highs_list = []
        n_predicted_list = []

        for t in thresholds:
            mask = df[model_key] >= t
            result = _compute_or_ci(mask, df, combine_lp_lb=True)
            if result:
                ors_list.append(result["or"])
                ci_lows_list.append(result["ci_low"])
                ci_highs_list.append(result["ci_high"])
                n_predicted_list.append(result["n_predicted"])
            else:
                ors_list.append(np.nan)
                ci_lows_list.append(np.nan)
                ci_highs_list.append(np.nan)
                n_predicted_list.append(0)

        key = "gb" if "gb" in model_key else "rf"
        results[key] = {
            "ors": [round(x, 4) if not np.isnan(x) else None for x in ors_list],
            "n_predicted": n_predicted_list,
        }

        # Plot on both axes for comparison
        for ax in axes:
            ax.plot(thresholds, ors_list, "o-", color=color, label=model_name,
                    linewidth=2, markersize=5)
            ax.fill_between(thresholds, ci_lows_list, ci_highs_list,
                            alpha=0.15, color=color)

    for ax in axes:
        ax.axhline(1.0, color="black", linestyle="--", alpha=0.5, linewidth=1)
        ax.set_xlabel("Prediction Threshold (P\u2265t)", fontsize=11)
        ax.set_ylabel("Odds Ratio", fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.2)

    axes[0].set_title("OR by Threshold (Path+LP vs Ben+LB)", fontsize=13)
    axes[0].set_xlim(0.05, 0.95)

    # Right panel: zoomed to show GB advantage
    axes[1].set_title("OR by Threshold (zoomed, y-axis from 0.8)", fontsize=13)
    axes[1].set_ylim(0.8, None)
    axes[1].set_xlim(0.05, 0.95)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "enrichment_by_threshold.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved enrichment_by_threshold.png")

    return results


def plot_path_ratio_by_method(df):
    """Bar chart: Path/(Path+Benign) fraction by prediction method."""
    fig, ax = plt.subplots(figsize=(10, 5))

    definitive = df[df["significance_simple"].isin(
        ["Pathogenic", "Likely_pathogenic", "Benign", "Likely_benign"])].copy()
    is_path = definitive["significance_simple"].isin(["Pathogenic", "Likely_pathogenic"])

    methods = [
        ("Background\n(all)", pd.Series(True, index=definitive.index)),
        ("RF\n(P\u22650.5)", definitive["p_edited_rnasee"] >= 0.5),
        ("RF\n(P\u22650.8)", definitive["p_edited_rnasee"] >= 0.8),
        ("GB\n(P\u22650.5)", definitive["p_edited_gb"] >= 0.5),
        ("GB\n(P\u22650.8)", definitive["p_edited_gb"] >= 0.8),
        ("GB-only\n(P\u22650.5)", (definitive["p_edited_gb"] >= 0.5) & (definitive["p_edited_rnasee"] < 0.5)),
    ]

    labels = []
    fracs = []
    counts = []
    bar_colors = []
    color_map = {
        "Background": "#9E9E9E", "RF": "#F57C00", "GB": "#1976D2", "GB-only": "#7B1FA2"
    }

    for label, mask in methods:
        n_total = mask.sum()
        n_path = is_path[mask].sum()
        frac = n_path / n_total * 100 if n_total > 0 else 0
        labels.append(label)
        fracs.append(frac)
        counts.append(n_total)
        for key, col in color_map.items():
            if key in label.replace("\n", " "):
                bar_colors.append(col)
                break

    x = np.arange(len(labels))
    bars = ax.bar(x, fracs, color=bar_colors, edgecolor="white", linewidth=0.5, width=0.6, alpha=0.85)

    # Annotate with fraction and count
    for i, (bar, frac, n) in enumerate(zip(bars, fracs, counts)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{frac:.1f}%\n(n={n:,})", ha="center", va="bottom", fontsize=9)

    # Reference line for background rate
    bg_frac = fracs[0]
    ax.axhline(bg_frac, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.text(len(labels) - 0.5, bg_frac + 0.3, f"Background: {bg_frac:.1f}%",
            fontsize=9, alpha=0.6, ha="right")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Pathogenic+LP / (Path+LP + Ben+LB) %", fontsize=11)
    ax.set_title("Pathogenic Fraction Among Definitive Classifications", fontsize=13)
    ax.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pathogenic_ratio_by_method.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved pathogenic_ratio_by_method.png")


def _bayesian_recalibrate(p_model, pi_model, pi_real):
    """Apply Bayesian prior adjustment to model probabilities."""
    prior_ratio = pi_real / pi_model
    inv_prior_ratio = (1 - pi_real) / (1 - pi_model)
    return (p_model * prior_ratio) / (p_model * prior_ratio + (1 - p_model) * inv_prior_ratio)


def _calibrated_threshold(pi_model, pi_real):
    """Find model threshold where P_calibrated = 0.5."""
    a = pi_real / pi_model
    b = (1 - pi_real) / (1 - pi_model)
    return 0.5 * b / (a * 0.5 + 0.5 * b)


def plot_calibrated_or_comparison(df):
    """Bar chart comparing original (P>=0.5) vs calibrated (P_cal>=0.5) enrichment."""
    # Derive priors from data
    if SPLITS_A3A.exists() and NEG_TIER1.exists():
        splits = pd.read_csv(SPLITS_A3A)
        n_pos = (splits["is_edited"] == 1).sum()
        neg_t1 = pd.read_csv(NEG_TIER1)
        n_neg = len(neg_t1)
        pi_real = n_pos / (n_pos + n_neg)
    elif CALIBRATED_JSON.exists():
        with open(CALIBRATED_JSON) as f:
            cal_data = json.load(f)
        pi_real = cal_data["priors"]["pi_real_tier1"]
    else:
        print("  SKIP calibrated_or_comparison.png (missing prior data)")
        return None

    pi_model = 0.50  # balanced via scale_pos_weight
    t_model = _calibrated_threshold(pi_model, pi_real)

    fig, ax = plt.subplots(figsize=(10, 6))

    methods = []
    for model_key, model_name, color in [
        ("p_edited_gb", "GB_Full", "#1976D2"),
        ("p_edited_rnasee", "RNAsee_RF", "#F57C00"),
    ]:
        # Original P>=0.5
        mask_orig = df[model_key] >= 0.5
        res_orig = _compute_or_ci(mask_orig, df)
        if res_orig:
            methods.append((f"{model_name}\nP≥0.5\n(uncalibrated)", res_orig, color, 0.5))

        # Calibrated P_cal>=0.5 (equiv to P_model >= t_model)
        mask_cal = df[model_key] >= t_model
        res_cal = _compute_or_ci(mask_cal, df)
        if res_cal:
            methods.append((f"{model_name}\nP_cal≥0.5\n(calibrated)", res_cal, color, 1.0))

    if not methods:
        plt.close()
        return None

    x = np.arange(len(methods))
    ors = [m[1]["or"] for m in methods]
    ci_lows = [m[1]["ci_low"] for m in methods]
    ci_highs = [m[1]["ci_high"] for m in methods]
    alphas = [m[3] for m in methods]
    colors = [m[2] for m in methods]

    bars = ax.bar(x, ors, color=colors, edgecolor="white", linewidth=0.5, width=0.5)
    for bar, alpha in zip(bars, alphas):
        bar.set_alpha(alpha)

    yerr_low = [o - cl for o, cl in zip(ors, ci_lows)]
    yerr_high = [ch - o for o, ch in zip(ors, ci_highs)]
    ax.errorbar(x, ors, yerr=[yerr_low, yerr_high], fmt="none", ecolor="black",
                capsize=5, linewidth=1.5)

    ax.axhline(1.0, color="black", linestyle="--", alpha=0.5, linewidth=1)

    for i, m in enumerate(methods):
        res = m[1]
        pval = res["p"]
        p_str = "p<1e-10" if pval < 1e-10 else f"p={pval:.1e}" if pval < 0.001 else f"p={pval:.3f}"
        y_pos = max(ors[i], ci_highs[i]) + 0.02
        ax.text(x[i], y_pos,
                f"OR={ors[i]:.3f}\nn={res['n_predicted']:,}\n{p_str}",
                ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in methods], fontsize=9)
    ax.set_ylabel("Odds Ratio (Path+LP / Ben+LB)", fontsize=11)
    ax.set_title(f"Original vs Prior-Calibrated Pathogenic Enrichment\n"
                 f"π_real={pi_real:.4f} (1:{1/pi_real:.0f}), "
                 f"calibrated threshold: P_model≥{t_model:.3f}", fontsize=11)
    ax.set_ylim(0, max(ci_highs) * 1.4 if ci_highs else 2.0)
    ax.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "calibrated_or_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved calibrated_or_comparison.png")

    return {
        "pi_real": pi_real,
        "pi_model": pi_model,
        "t_model_for_pcal05": round(t_model, 4),
        "methods": {m[0].replace("\n", " "): m[1] for m in methods},
    }


if __name__ == "__main__":
    main()
