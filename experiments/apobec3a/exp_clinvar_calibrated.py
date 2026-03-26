#!/usr/bin/env python
"""Prior-calibrated ClinVar pathogenicity enrichment analysis.

Our editing prediction models (GB_Full, RF) were trained on artificially enriched
datasets (1:3 positive:negative with scale_pos_weight=3.0). The model's P=0.5
threshold does NOT correspond to a real-world 50% chance of being an editing site.

This script applies Bayesian recalibration to adjust model probabilities to
reflect true editing site prevalence, then re-analyzes pathogenic enrichment.

Real-world priors (derived from data):
  - Tier 1: 5,187 positives / 277,495 total → π_real = 1.87% (1:53)
  - ClinVar-specific: 413 known sites / 1,679,864 variants → π_clinvar = 0.025%

Calibration formula (Bayesian adjustment):
  P_cal = P_model × (π_real / π_model) /
          [P_model × (π_real / π_model) + (1 - P_model) × ((1 - π_real) / (1 - π_model))]

Usage:
    conda run -n quris python experiments/apobec3a/exp_clinvar_calibrated.py
"""

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

SCORES_CSV = PROJECT_ROOT / "experiments/apobec3a/outputs/clinvar_prediction/clinvar_all_scores.csv"
CDS_JSON = PROJECT_ROOT / "experiments/apobec3a/outputs/clinvar_prediction/rnasee_cds_replication.json"
SPLITS_A3A = PROJECT_ROOT / "data/processed/splits_expanded_a3a.csv"
NEG_TIER1 = PROJECT_ROOT / "data/processed/negatives_tier1.csv"
OUTPUT_DIR = PROJECT_ROOT / "experiments/apobec3a/outputs/clinvar_calibrated"

CATEGORIES = ["Pathogenic", "Likely_pathogenic", "VUS", "Conflicting",
              "Likely_benign", "Benign"]
COLORS = {"Pathogenic": "#d32f2f", "Likely_pathogenic": "#ff7043",
          "VUS": "#ffb74d", "Conflicting": "#90a4ae",
          "Likely_benign": "#81c784", "Benign": "#43a047"}


def bayesian_recalibrate(p_model, pi_model, pi_real):
    """Apply Bayesian prior adjustment to model probabilities.

    Adjusts from training prevalence (pi_model) to real-world prevalence (pi_real).
    """
    lr = p_model / np.maximum(1 - p_model, 1e-15)  # likelihood ratio
    prior_ratio = pi_real / pi_model
    inv_prior_ratio = (1 - pi_real) / (1 - pi_model)
    p_cal = (p_model * prior_ratio) / (p_model * prior_ratio + (1 - p_model) * inv_prior_ratio)
    return p_cal


def calibrated_threshold(pi_model, pi_real, p_cal_target=0.5):
    """Find model threshold where P_calibrated = p_cal_target.

    Solves: p_cal_target = p * (pi_real/pi_model) /
            [p * (pi_real/pi_model) + (1-p) * ((1-pi_real)/(1-pi_model))]
    """
    a = pi_real / pi_model
    b = (1 - pi_real) / (1 - pi_model)
    # At P_cal = 0.5: p*a = (1-p)*b → p = b/(a+b)
    # More generally: p_cal * [p*a + (1-p)*b] = p*a
    # p_cal * p*a + p_cal * (1-p)*b = p*a
    # p * [a - p_cal*a - p_cal*b] + p_cal*b ... wait let me just solve directly
    # p_cal_target = p*a / (p*a + (1-p)*b)
    # p_cal_target * (p*a + (1-p)*b) = p*a
    # p_cal_target * p * a + p_cal_target * b - p_cal_target * p * b = p * a
    # p * (p_cal_target * a - p_cal_target * b - a) = -p_cal_target * b
    # p = p_cal_target * b / (a - p_cal_target * a + p_cal_target * b)
    # p = p_cal_target * b / (a * (1 - p_cal_target) + p_cal_target * b)
    t = p_cal_target * b / (a * (1 - p_cal_target) + p_cal_target * b)
    return t


def fisher_enrichment(predicted_mask, df, combine_lp_lb=True):
    """Compute pathogenic enrichment with Fisher exact test."""
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

    # 95% CI via Woolf's method
    log_or = np.log(odds) if odds > 0 else 0
    se = np.sqrt(1/max(p_pred, 1) + 1/max(b_pred, 1) + 1/max(p_rest, 1) + 1/max(b_rest, 1))
    ci_low = np.exp(log_or - 1.96 * se)
    ci_high = np.exp(log_or + 1.96 * se)

    return {
        "or": round(odds, 4), "p": float(pval),
        "ci_low": round(ci_low, 4), "ci_high": round(ci_high, 4),
        "n_predicted": int(p_pred + b_pred),
        "n_path": int(p_pred), "n_benign": int(b_pred),
        "path_rate": round(p_pred / (p_pred + b_pred) * 100, 2),
        "bg_rate": round(p_rest / (p_rest + b_rest) * 100, 2) if (p_rest + b_rest) > 0 else 0,
    }


def plot_calibration_curve(pi_model_options, pi_real_options, output_dir):
    """Plot P_model vs P_calibrated for different priors."""
    p_model = np.linspace(0.001, 0.999, 500)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, pi_model in zip(axes, pi_model_options):
        for pi_real, label, color in pi_real_options:
            p_cal = bayesian_recalibrate(p_model, pi_model, pi_real)
            t_cal = calibrated_threshold(pi_model, pi_real)
            ax.plot(p_model, p_cal, color=color, linewidth=2, label=f"{label}")
            ax.axvline(t_cal, color=color, linestyle=":", alpha=0.6, linewidth=1)
            ax.annotate(f"t={t_cal:.3f}", xy=(t_cal, 0.5), fontsize=8,
                        color=color, ha="left", va="bottom",
                        xytext=(t_cal + 0.01, 0.52))

        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Uncalibrated")
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.3)
        ax.set_xlabel("Model P(edited)", fontsize=11)
        ax.set_ylabel("Calibrated P(edited)", fontsize=11)
        ax.set_title(f"Bayesian Calibration (π_model={pi_model:.2f})", fontsize=12)
        ax.legend(fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_dir / "calibration_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved calibration_curves.png")


def plot_enrichment_vs_calibrated_threshold(df, pi_model, pi_real, label, output_dir):
    """OR at each calibrated threshold for GB and RF."""
    p_cal_thresholds = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    results = {"pi_model": pi_model, "pi_real": pi_real, "label": label}

    for ax_idx, (model_key, model_name, color) in enumerate([
        ("p_edited_gb", "GB_Full", "#1976D2"),
        ("p_edited_rnasee", "RNAsee_RF", "#F57C00"),
    ]):
        p_cal = bayesian_recalibrate(df[model_key].values, pi_model, pi_real)

        ors_list = []
        n_pass_list = []
        ci_lows_list = []
        ci_highs_list = []
        valid_thresholds = []

        for t in p_cal_thresholds:
            mask = pd.Series(p_cal >= t, index=df.index)
            n_pass = mask.sum()
            if n_pass < 10:
                continue
            result = fisher_enrichment(mask, df)
            if result is None:
                continue
            ors_list.append(result["or"])
            ci_lows_list.append(result["ci_low"])
            ci_highs_list.append(result["ci_high"])
            n_pass_list.append(n_pass)
            valid_thresholds.append(t)

        key = "gb" if "gb" in model_key else "rf"
        results[key] = {
            "thresholds": valid_thresholds,
            "ors": ors_list,
            "n_pass": n_pass_list,
        }

        for ax in axes:
            ax.plot(valid_thresholds, ors_list, "o-", color=color, label=model_name,
                    linewidth=2, markersize=5)
            ax.fill_between(valid_thresholds, ci_lows_list, ci_highs_list,
                            alpha=0.15, color=color)

    for ax in axes:
        ax.axhline(1.0, color="black", linestyle="--", alpha=0.5, linewidth=1)
        ax.set_xlabel("Calibrated P(edited) threshold", fontsize=11)
        ax.set_ylabel("Odds Ratio (Path+LP / Ben+LB)", fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.2)
        ax.set_xscale("log")

    axes[0].set_title(f"OR by Calibrated Threshold ({label})", fontsize=12)
    axes[1].set_title(f"OR (zoomed, y from 0.8) ({label})", fontsize=12)
    axes[1].set_ylim(0.8, None)

    plt.tight_layout()
    safe_label = label.lower().replace(" ", "_").replace("=", "").replace("/", "_")
    plt.savefig(output_dir / f"enrichment_vs_calibrated_threshold_{safe_label}.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved enrichment_vs_calibrated_threshold_{safe_label}.png")

    return results


def plot_original_vs_calibrated_or(df, pi_model, pi_real, label, output_dir):
    """Side-by-side: original OR (P>=0.5) vs calibrated OR (P_cal>=0.5)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = []

    for model_key, model_name, color in [
        ("p_edited_gb", "GB_Full", "#1976D2"),
        ("p_edited_rnasee", "RNAsee_RF", "#F57C00"),
    ]:
        # Original: P_model >= 0.5
        mask_orig = df[model_key] >= 0.5
        res_orig = fisher_enrichment(mask_orig, df)

        # Calibrated: P_cal >= 0.5
        p_cal = bayesian_recalibrate(df[model_key].values, pi_model, pi_real)
        t_model = calibrated_threshold(pi_model, pi_real)
        mask_cal = pd.Series(df[model_key].values >= t_model, index=df.index)
        res_cal = fisher_enrichment(mask_cal, df)

        if res_orig:
            methods.append((f"{model_name}\nP≥0.5", res_orig, color, 0.5))
        if res_cal:
            methods.append((f"{model_name}\nP_cal≥0.5", res_cal, color, 1.0))

    if not methods:
        plt.close()
        return

    x = np.arange(len(methods))
    ors = [m[1]["or"] for m in methods]
    ci_lows = [m[1]["ci_low"] for m in methods]
    ci_highs = [m[1]["ci_high"] for m in methods]
    alphas = [m[3] for m in methods]
    colors = [m[2] for m in methods]
    labels = [m[0] for m in methods]

    bars = ax.bar(x, ors, color=colors, edgecolor="white", linewidth=0.5,
                  width=0.5)
    for bar, alpha in zip(bars, alphas):
        bar.set_alpha(alpha)

    # Error bars (clip to non-negative to avoid matplotlib error with degenerate CIs)
    yerr_low = [max(0.0, o - cl) for o, cl in zip(ors, ci_lows)]
    yerr_high = [max(0.0, ch - o) for o, ch in zip(ors, ci_highs)]
    ax.errorbar(x, ors, yerr=[yerr_low, yerr_high], fmt="none", ecolor="black",
                capsize=5, linewidth=1.5)

    ax.axhline(1.0, color="black", linestyle="--", alpha=0.5, linewidth=1)

    # Annotate
    for i, m in enumerate(methods):
        res = m[1]
        pval = res["p"]
        p_str = f"p<1e-10" if pval < 1e-10 else f"p={pval:.1e}" if pval < 0.001 else f"p={pval:.3f}"
        y_pos = max(ors[i], ci_highs[i]) + 0.02
        ax.text(x[i], y_pos,
                f"OR={ors[i]:.3f}\nn={res['n_predicted']:,}\n{p_str}",
                ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Odds Ratio (Path+LP / Ben+LB)", fontsize=11)
    ax.set_title(f"Original vs Calibrated Enrichment ({label})\n"
                 f"π_model={pi_model:.2f}, π_real={pi_real:.4f}", fontsize=12)
    finite_highs = [v for v in ci_highs if np.isfinite(v)]
    ax.set_ylim(0, max(finite_highs) * 1.4 if finite_highs else 2.0)
    ax.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    safe_label = label.lower().replace(" ", "_").replace("=", "").replace("/", "_")
    plt.savefig(output_dir / f"original_vs_calibrated_{safe_label}.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved original_vs_calibrated_{safe_label}.png")


def plot_calibrated_distribution(df, pi_model, pi_real, label, output_dir):
    """Histogram of P_calibrated for Pathogenic vs Benign variants."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, (model_key, model_name) in enumerate([
        ("p_edited_gb", "GB_Full"), ("p_edited_rnasee", "RNAsee_RF")
    ]):
        p_cal = bayesian_recalibrate(df[model_key].values, pi_model, pi_real)

        for cat, color in [("Pathogenic", "#d32f2f"), ("Benign", "#43a047")]:
            mask = df["significance_simple"] == cat
            if mask.sum() == 0:
                continue
            vals = p_cal[mask.values]
            # Use log-spaced bins for better visualization of small values
            bins = np.logspace(np.log10(max(vals.min(), 1e-8)), np.log10(max(vals.max(), 1e-6)), 50)
            axes[ax_idx].hist(vals, bins=bins, alpha=0.6, color=color,
                              label=f"{cat} (n={mask.sum():,})", edgecolor="white")

        axes[ax_idx].set_xscale("log")
        axes[ax_idx].set_xlabel(f"Calibrated P(edited) [{model_name}]", fontsize=11)
        axes[ax_idx].set_ylabel("Count", fontsize=11)
        axes[ax_idx].set_title(f"{model_name}: Calibrated Score Distribution\n({label})")
        axes[ax_idx].legend(fontsize=9)
        axes[ax_idx].grid(True, alpha=0.2)

        # Mark P_cal = 0.5 if visible
        t_model = calibrated_threshold(pi_model, pi_real)
        axes[ax_idx].axvline(0.5, color="red", linestyle="--", alpha=0.5, label="P_cal=0.5")

    plt.tight_layout()
    safe_label = label.lower().replace(" ", "_").replace("=", "").replace("/", "_")
    plt.savefig(output_dir / f"calibrated_distribution_{safe_label}.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved calibrated_distribution_{safe_label}.png")


def main():
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ===================================================================
    # Step 1: Derive priors from data
    # ===================================================================
    print("=" * 70)
    print("PRIOR-CALIBRATED CLINVAR ENRICHMENT ANALYSIS")
    print("=" * 70)

    print("\n--- Deriving priors from data ---")

    # Tier 1 prior
    splits = pd.read_csv(SPLITS_A3A)
    n_positives = (splits["is_edited"] == 1).sum()
    neg_tier1 = pd.read_csv(NEG_TIER1)
    n_tier1_neg = len(neg_tier1)
    pi_real_tier1 = n_positives / (n_positives + n_tier1_neg)

    print(f"  Positives (splits_expanded_a3a.csv): {n_positives:,}")
    print(f"  Tier 1 negatives: {n_tier1_neg:,}")
    print(f"  π_real (Tier 1): {pi_real_tier1:.4f} (1:{1/pi_real_tier1:.0f})")

    # ClinVar-specific prior
    scores_df = pd.read_csv(SCORES_CSV)
    n_clinvar = len(scores_df)
    n_known_clinvar = (scores_df["is_known_editing_site"] == True).sum()
    pi_real_clinvar = n_known_clinvar / n_clinvar

    print(f"  ClinVar variants: {n_clinvar:,}")
    print(f"  Known editing sites in ClinVar: {n_known_clinvar:,}")
    print(f"  π_real (ClinVar): {pi_real_clinvar:.6f} (1:{1/pi_real_clinvar:.0f})")

    # Model training prior
    # With scale_pos_weight=3 on 1:3 data, the effective prior is ~0.5
    # (loss treats positives as 3x, balancing the 1:3 ratio)
    pi_model_balanced = 0.50  # effective prior after weighting
    pi_model_raw = 0.25       # raw training prevalence

    print(f"\n  π_model (balanced, w/ scale_pos_weight=3): {pi_model_balanced}")
    print(f"  π_model (raw training prevalence): {pi_model_raw}")

    # Calibrated thresholds
    for pi_m_label, pi_m in [("balanced (0.50)", pi_model_balanced),
                              ("raw (0.25)", pi_model_raw)]:
        for pi_r_label, pi_r in [("Tier1", pi_real_tier1),
                                   ("ClinVar", pi_real_clinvar)]:
            t = calibrated_threshold(pi_m, pi_r)
            print(f"  t_cal (π_model={pi_m_label}, π_real={pi_r_label}): {t:.6f}")

    # ===================================================================
    # Step 2: Calibrate all scores
    # ===================================================================
    print("\n--- Calibrating model scores ---")

    # Primary analysis: π_model=0.50 (balanced), π_real=Tier1
    p_cal_gb_tier1 = bayesian_recalibrate(scores_df["p_edited_gb"].values,
                                           pi_model_balanced, pi_real_tier1)
    p_cal_rf_tier1 = bayesian_recalibrate(scores_df["p_edited_rnasee"].values,
                                           pi_model_balanced, pi_real_tier1)

    # Sensitivity: π_model=0.25 (raw)
    p_cal_gb_tier1_raw = bayesian_recalibrate(scores_df["p_edited_gb"].values,
                                               pi_model_raw, pi_real_tier1)

    # ClinVar-specific prior
    p_cal_gb_clinvar = bayesian_recalibrate(scores_df["p_edited_gb"].values,
                                             pi_model_balanced, pi_real_clinvar)

    scores_df["p_cal_gb_tier1"] = p_cal_gb_tier1
    scores_df["p_cal_rf_tier1"] = p_cal_rf_tier1
    scores_df["p_cal_gb_tier1_raw"] = p_cal_gb_tier1_raw
    scores_df["p_cal_gb_clinvar"] = p_cal_gb_clinvar

    # Sanity check: most calibrated scores should be << model scores
    print(f"\n  Sanity checks (π_model=0.50, π_real=Tier1):")
    print(f"    GB: median P_model={scores_df['p_edited_gb'].median():.4f} → "
          f"P_cal={np.median(p_cal_gb_tier1):.6f}")
    print(f"    RF: median P_model={scores_df['p_edited_rnasee'].median():.4f} → "
          f"P_cal={np.median(p_cal_rf_tier1):.6f}")
    print(f"    GB P_cal >= 0.5: {(p_cal_gb_tier1 >= 0.5).sum():,} variants "
          f"({(p_cal_gb_tier1 >= 0.5).mean()*100:.2f}%)")
    print(f"    RF P_cal >= 0.5: {(p_cal_rf_tier1 >= 0.5).sum():,} variants "
          f"({(p_cal_rf_tier1 >= 0.5).mean()*100:.2f}%)")

    t_model_tier1 = calibrated_threshold(pi_model_balanced, pi_real_tier1)
    print(f"    Model threshold for P_cal=0.5: {t_model_tier1:.4f}")
    print(f"    GB P_model >= {t_model_tier1:.4f}: "
          f"{(scores_df['p_edited_gb'] >= t_model_tier1).sum():,}")
    print(f"    RF P_model >= {t_model_tier1:.4f}: "
          f"{(scores_df['p_edited_rnasee'] >= t_model_tier1).sum():,}")

    # ===================================================================
    # Step 3: Enrichment at multiple calibrated thresholds
    # ===================================================================
    print("\n--- Enrichment at calibrated thresholds ---")

    p_cal_thresholds = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5]

    all_enrichment = {}

    for config_label, pi_m, pi_r in [
        ("Tier1_balanced", pi_model_balanced, pi_real_tier1),
        ("Tier1_raw", pi_model_raw, pi_real_tier1),
        ("ClinVar_balanced", pi_model_balanced, pi_real_clinvar),
    ]:
        print(f"\n  === {config_label} (π_model={pi_m}, π_real={pi_r:.6f}) ===")
        config_results = {"pi_model": pi_m, "pi_real": pi_r, "gb": [], "rf": []}

        for model_key, model_name in [("p_edited_gb", "GB_Full"),
                                       ("p_edited_rnasee", "RNAsee_RF")]:
            p_cal = bayesian_recalibrate(scores_df[model_key].values, pi_m, pi_r)

            print(f"\n    {model_name}:")
            key = "gb" if "gb" in model_key else "rf"

            for t in p_cal_thresholds:
                mask = pd.Series(p_cal >= t, index=scores_df.index)
                n_pass = mask.sum()
                if n_pass < 10:
                    print(f"      P_cal>={t}: {n_pass:,} variants (too few)")
                    continue

                res = fisher_enrichment(mask, scores_df)
                if res is None:
                    continue

                sig = "***" if res["p"] < 0.001 else "**" if res["p"] < 0.01 else "*" if res["p"] < 0.05 else "ns"
                print(f"      P_cal>={t}: n={n_pass:,} | OR={res['or']:.3f} "
                      f"({res['ci_low']:.3f}-{res['ci_high']:.3f}) | "
                      f"Path%={res['path_rate']:.1f}% (bg={res['bg_rate']:.1f}%) | "
                      f"p={res['p']:.2e} {sig}")

                config_results[key].append({
                    "threshold": t, "n_pass": n_pass, **res
                })

        all_enrichment[config_label] = config_results

    # ===================================================================
    # Step 4: Compare original vs calibrated
    # ===================================================================
    print("\n\n--- Original vs Calibrated comparison ---")

    comparison = {}
    for model_key, model_name in [("p_edited_gb", "GB_Full"),
                                   ("p_edited_rnasee", "RNAsee_RF")]:
        # Original P>=0.5
        mask_orig = scores_df[model_key] >= 0.5
        res_orig = fisher_enrichment(mask_orig, scores_df)

        # Calibrated P_cal>=0.5 (Tier1 balanced)
        t_m = calibrated_threshold(pi_model_balanced, pi_real_tier1)
        mask_cal = scores_df[model_key] >= t_m
        res_cal = fisher_enrichment(mask_cal, scores_df)

        comparison[model_name] = {
            "original_p05": res_orig,
            "calibrated_pcal05_tier1": res_cal,
            "t_model_for_pcal05": round(t_m, 6),
        }

        if res_orig and res_cal:
            print(f"\n  {model_name}:")
            print(f"    Original (P>=0.5):      n={res_orig['n_predicted']:,}, "
                  f"OR={res_orig['or']:.3f}, p={res_orig['p']:.2e}")
            print(f"    Calibrated (P_cal>=0.5): n={res_cal['n_predicted']:,}, "
                  f"OR={res_cal['or']:.3f}, p={res_cal['p']:.2e}")
            print(f"    Model threshold: P>={t_m:.4f}")

    # ===================================================================
    # Step 5: CDS non-syn subset analysis (if available)
    # ===================================================================
    cds_enrichment = None
    if CDS_JSON.exists():
        print("\n\n--- CDS non-syn subset (from rnasee_cds_replication.json) ---")
        with open(CDS_JSON) as f:
            cds_data = json.load(f)

        if "fisher_tests_path_lp_vs_ben_lb" in cds_data:
            cds_enrichment = {"original": cds_data["fisher_tests_path_lp_vs_ben_lb"]}
            print("  Loaded CDS non-syn Fisher test results (original thresholds)")
            print("  Note: CDS-level recalibration requires re-running on CDS subset scores")

    # ===================================================================
    # Step 6: Generate figures
    # ===================================================================
    print("\n\n--- Generating figures ---")

    # 1. Calibration curves
    pi_model_options = [pi_model_balanced, pi_model_raw]
    pi_real_options = [
        (pi_real_tier1, f"Tier1 (1:{1/pi_real_tier1:.0f})", "#1976D2"),
        (pi_real_clinvar, f"ClinVar (1:{1/pi_real_clinvar:.0f})", "#d32f2f"),
    ]
    plot_calibration_curve(pi_model_options, pi_real_options, OUTPUT_DIR)

    # 2. Enrichment vs calibrated threshold
    tier1_enrich = plot_enrichment_vs_calibrated_threshold(
        scores_df, pi_model_balanced, pi_real_tier1, "Tier1 prior", OUTPUT_DIR)

    # 3. Original vs calibrated OR
    plot_original_vs_calibrated_or(
        scores_df, pi_model_balanced, pi_real_tier1, "Tier1 prior", OUTPUT_DIR)

    # 4. Distribution shift
    plot_calibrated_distribution(
        scores_df, pi_model_balanced, pi_real_tier1, "Tier1 prior", OUTPUT_DIR)

    # ===================================================================
    # Step 7: Save results
    # ===================================================================
    elapsed = time.time() - t0

    results = {
        "description": "Prior-calibrated ClinVar pathogenicity enrichment analysis",
        "priors": {
            "n_positives": int(n_positives),
            "n_tier1_negatives": int(n_tier1_neg),
            "pi_real_tier1": round(pi_real_tier1, 6),
            "pi_real_tier1_ratio": f"1:{1/pi_real_tier1:.0f}",
            "n_clinvar_variants": int(n_clinvar),
            "n_known_clinvar": int(n_known_clinvar),
            "pi_real_clinvar": round(pi_real_clinvar, 8),
            "pi_real_clinvar_ratio": f"1:{1/pi_real_clinvar:.0f}",
            "pi_model_balanced": pi_model_balanced,
            "pi_model_raw": pi_model_raw,
        },
        "calibrated_thresholds": {
            "tier1_balanced": round(calibrated_threshold(pi_model_balanced, pi_real_tier1), 6),
            "tier1_raw": round(calibrated_threshold(pi_model_raw, pi_real_tier1), 6),
            "clinvar_balanced": round(calibrated_threshold(pi_model_balanced, pi_real_clinvar), 8),
        },
        "sanity_checks": {
            "gb_median_pmodel": round(float(scores_df["p_edited_gb"].median()), 4),
            "gb_median_pcal_tier1": round(float(np.median(p_cal_gb_tier1)), 6),
            "rf_median_pmodel": round(float(scores_df["p_edited_rnasee"].median()), 4),
            "rf_median_pcal_tier1": round(float(np.median(p_cal_rf_tier1)), 6),
            "gb_n_pcal_above_0.5_tier1": int((p_cal_gb_tier1 >= 0.5).sum()),
            "rf_n_pcal_above_0.5_tier1": int((p_cal_rf_tier1 >= 0.5).sum()),
        },
        "enrichment_by_config": all_enrichment,
        "original_vs_calibrated": comparison,
        "cds_nonsyn_enrichment": cds_enrichment,
        "total_time_seconds": round(elapsed, 1),
    }

    # Save JSON
    def serialize(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return str(obj)

    with open(OUTPUT_DIR / "calibrated_enrichment_results.json", "w") as f:
        json.dump(results, f, indent=2, default=serialize)
    print(f"\n  Saved calibrated_enrichment_results.json")

    # Save calibrated scores CSV (only calibrated columns + identifiers)
    cal_cols = scores_df[["site_id", "chr", "start", "gene", "significance_simple",
                           "is_known_editing_site", "p_edited_gb", "p_edited_rnasee",
                           "p_cal_gb_tier1", "p_cal_rf_tier1",
                           "p_cal_gb_tier1_raw", "p_cal_gb_clinvar"]].copy()
    cal_cols.to_csv(OUTPUT_DIR / "clinvar_calibrated_scores.csv", index=False)
    print(f"  Saved clinvar_calibrated_scores.csv ({len(cal_cols):,} variants)")

    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
