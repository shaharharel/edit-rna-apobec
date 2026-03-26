#!/usr/bin/env python
"""Editing rate analysis for APOBEC3B sites.

Analyzes rate distributions, structure-rate correlations, and motif-rate
relationships for A3B editing sites.

Usage:
    conda run -n quris python experiments/apobec3b/exp_rate_analysis_a3b.py
"""

import json
import logging
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SPLITS_CSV = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "splits_multi_enzyme_v2.csv"
SEQUENCES_JSON = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "multi_enzyme_sequences_v2.json"
LOOP_POS_CSV = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "loop_position_per_site_v2.csv"
STRUCT_CACHE = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "structure_cache_multi_enzyme_v2.npz"
OUTPUT_DIR = Path(__file__).parent / "outputs" / "rate_analysis"

CENTER = 100


def load_data():
    """Load A3B data with editing rates, loop geometry, and structure features."""
    df = pd.read_csv(SPLITS_CSV)
    df = df[df["enzyme"] == "A3B"].copy()
    logger.info("A3B sites: %d", len(df))

    # Filter to sites with valid editing rates
    df = df[df["editing_rate"].notna() & (df["editing_rate"] > 0)].copy()
    logger.info("A3B sites with editing rate > 0: %d", len(df))

    # Load sequences
    with open(SEQUENCES_JSON) as f:
        sequences = json.load(f)

    # Extract TC motif
    def is_tc(sid):
        seq = sequences.get(str(sid), "N" * 201).upper().replace("T", "U")
        return seq[CENTER - 1] == "U" if len(seq) > CENTER else False

    df["is_tc"] = df["site_id"].apply(is_tc)

    # Load loop geometry
    if LOOP_POS_CSV.exists():
        loop_df = pd.read_csv(LOOP_POS_CSV)
        loop_df["site_id"] = loop_df["site_id"].astype(str)
        loop_cols = ["site_id", "is_unpaired", "loop_size", "relative_loop_position",
                     "dist_to_apex", "local_unpaired_fraction", "loop_type"]
        available_cols = [c for c in loop_cols if c in loop_df.columns]
        loop_df = loop_df[available_cols]
        df["site_id"] = df["site_id"].astype(str)
        df = df.merge(loop_df, on="site_id", how="left")
        logger.info("Loop features merged: %d rows", len(df))

    # Load structure delta
    if STRUCT_CACHE.exists():
        data = np.load(STRUCT_CACHE, allow_pickle=True)
        sids = [str(s) for s in data["site_ids"]]
        deltas = data["delta_features"]
        mfes = data["mfes"]
        delta_map = {sid: deltas[i] for i, sid in enumerate(sids)}
        mfe_map = {sid: float(mfes[i]) for i, sid in enumerate(sids)}
        df["delta_mfe"] = df["site_id"].map(
            lambda x: delta_map.get(str(x), np.zeros(7))[3])
        df["mfe"] = df["site_id"].map(lambda x: mfe_map.get(str(x), np.nan))

    return df


def analyze_rates(df):
    """Compute rate statistics and correlations."""
    results = {
        "enzyme": "A3B",
        "n_sites": int(len(df)),
    }

    # Overall rate stats
    rates = df["editing_rate"].values
    results["rate_stats"] = {
        "mean": float(np.mean(rates)),
        "std": float(np.std(rates)),
        "median": float(np.median(rates)),
        "min": float(np.min(rates)),
        "max": float(np.max(rates)),
        "q25": float(np.percentile(rates, 25)),
        "q75": float(np.percentile(rates, 75)),
    }

    # Per-dataset rate stats
    results["per_dataset"] = {}
    for ds, grp in df.groupby("dataset_source"):
        r = grp["editing_rate"].values
        results["per_dataset"][ds] = {
            "n": int(len(grp)),
            "mean": float(np.mean(r)),
            "std": float(np.std(r)),
            "median": float(np.median(r)),
        }

    # Structure-rate correlations
    results["correlations"] = {}
    for feat in ["relative_loop_position", "loop_size", "local_unpaired_fraction",
                 "delta_mfe"]:
        if feat in df.columns:
            valid = df[[feat, "editing_rate"]].dropna()
            if len(valid) > 10:
                rho, p = spearmanr(valid[feat], valid["editing_rate"])
                results["correlations"][feat] = {
                    "spearman_rho": float(rho),
                    "p_value": float(p),
                    "n": int(len(valid)),
                }

    # TC motif vs rate
    if "is_tc" in df.columns:
        tc_rates = df[df["is_tc"]]["editing_rate"].values
        non_tc_rates = df[~df["is_tc"]]["editing_rate"].values
        results["tc_rate_comparison"] = {
            "tc_n": int(len(tc_rates)),
            "tc_mean_rate": float(np.mean(tc_rates)) if len(tc_rates) > 0 else None,
            "non_tc_n": int(len(non_tc_rates)),
            "non_tc_mean_rate": float(np.mean(non_tc_rates)) if len(non_tc_rates) > 0 else None,
        }
        if len(tc_rates) > 5 and len(non_tc_rates) > 5:
            stat, p = mannwhitneyu(tc_rates, non_tc_rates, alternative="two-sided")
            results["tc_rate_comparison"]["mannwhitney_stat"] = float(stat)
            results["tc_rate_comparison"]["mannwhitney_p"] = float(p)

    # Unpaired vs paired rate
    if "is_unpaired" in df.columns:
        unpaired = df[df["is_unpaired"] == True]["editing_rate"].values
        paired = df[df["is_unpaired"] == False]["editing_rate"].values
        results["unpaired_rate_comparison"] = {
            "unpaired_n": int(len(unpaired)),
            "unpaired_mean": float(np.mean(unpaired)) if len(unpaired) > 0 else None,
            "paired_n": int(len(paired)),
            "paired_mean": float(np.mean(paired)) if len(paired) > 0 else None,
        }
        if len(unpaired) > 5 and len(paired) > 5:
            stat, p = mannwhitneyu(unpaired, paired, alternative="two-sided")
            results["unpaired_rate_comparison"]["mannwhitney_p"] = float(p)

    return results


def make_figures(df):
    """Generate rate analysis figures."""

    # 1. Rate distribution by dataset
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    for ds, grp in df.groupby("dataset_source"):
        axes[0].hist(grp["editing_rate"], bins=50, alpha=0.6, label=f"{ds} (n={len(grp)})")
    axes[0].set_xlabel("Editing Rate")
    axes[0].set_ylabel("Count")
    axes[0].set_title("A3B Editing Rate Distribution by Dataset")
    axes[0].legend()

    # Log-scale histogram
    log_rates = np.log10(df["editing_rate"].clip(lower=1e-6))
    for ds, grp in df.groupby("dataset_source"):
        lr = np.log10(grp["editing_rate"].clip(lower=1e-6))
        axes[1].hist(lr, bins=50, alpha=0.6, label=f"{ds}")
    axes[1].set_xlabel("log10(Editing Rate)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("A3B Editing Rate Distribution (log scale)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rate_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 2. Rate correlations
    corr_features = []
    for feat in ["relative_loop_position", "loop_size", "local_unpaired_fraction", "delta_mfe"]:
        if feat in df.columns and df[feat].notna().sum() > 10:
            corr_features.append(feat)

    if corr_features:
        n_feats = len(corr_features)
        fig, axes = plt.subplots(1, n_feats, figsize=(5 * n_feats, 4.5))
        if n_feats == 1:
            axes = [axes]

        for i, feat in enumerate(corr_features):
            valid = df[[feat, "editing_rate"]].dropna()
            axes[i].scatter(valid[feat], valid["editing_rate"], alpha=0.2, s=10)
            rho, p = spearmanr(valid[feat], valid["editing_rate"])
            axes[i].set_xlabel(feat)
            axes[i].set_ylabel("Editing Rate")
            axes[i].set_title(f"rho={rho:.3f}, p={p:.2e}")

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "rate_correlations.png", dpi=150, bbox_inches="tight")
        plt.close()

    # 3. TC motif vs rate boxplot
    if "is_tc" in df.columns:
        fig, ax = plt.subplots(figsize=(6, 5))
        tc_data = [df[df["is_tc"]]["editing_rate"].values,
                    df[~df["is_tc"]]["editing_rate"].values]
        bp = ax.boxplot(tc_data, labels=["TC context", "Non-TC context"],
                        patch_artist=True)
        bp["boxes"][0].set_facecolor("#3b82f6")
        bp["boxes"][1].set_facecolor("#ef4444")
        ax.set_ylabel("Editing Rate")
        ax.set_title("A3B: Editing Rate by Motif Context")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "rate_by_motif.png", dpi=150, bbox_inches="tight")
        plt.close()

    logger.info("Figures saved to %s", OUTPUT_DIR)


def main():
    t_start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    results = analyze_rates(df)
    make_figures(df)

    out_path = OUTPUT_DIR / "rate_results_a3b.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", out_path)

    # Print summary
    logger.info("=" * 60)
    logger.info("A3B Rate Analysis Summary")
    logger.info("  Sites: %d", results["n_sites"])
    logger.info("  Rate: mean=%.4f, median=%.4f, std=%.4f",
                results["rate_stats"]["mean"],
                results["rate_stats"]["median"],
                results["rate_stats"]["std"])
    for feat, corr in results.get("correlations", {}).items():
        logger.info("  %s: Spearman rho=%.4f, p=%.2e",
                    feat, corr["spearman_rho"], corr["p_value"])

    elapsed = time.time() - t_start
    logger.info("Total time: %.1f seconds", elapsed)


if __name__ == "__main__":
    main()
