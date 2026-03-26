#!/usr/bin/env python
"""Rate analysis for APOBEC3G editing sites.

Analyzes editing rate distributions and correlations with structural features
for A3G sites from Dang 2019 (NK_Hyp and NK_Norm conditions).

Key analyses:
1. Overall rate distribution
2. NK_Hyp vs NK_Norm comparison (if condition data available)
3. Rate vs relative loop position (RLP) correlation
4. Rate vs loop size correlation
5. Rate vs CC motif correlation

Usage:
    conda run -n quris python experiments/apobec3g/exp_rate_analysis_a3g.py
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
SPLITS_CSV = PROJECT_ROOT / "data/processed/multi_enzyme/splits_multi_enzyme_v2.csv"
SEQUENCES_JSON = PROJECT_ROOT / "data/processed/multi_enzyme/multi_enzyme_sequences_v2.json"
LOOP_POS_CSV = PROJECT_ROOT / "data/processed/multi_enzyme/loop_position_per_site_v2.csv"
OUTPUT_DIR = Path(__file__).parent / "outputs" / "rate_analysis"
FIGURES_DIR = Path(__file__).parent / "outputs" / "figures"


def main():
    t_start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading data...")
    df = pd.read_csv(SPLITS_CSV)
    df_a3g = df[df["enzyme"] == "A3G"].copy()
    logger.info("  A3G sites: %d", len(df_a3g))

    # Load sequences for motif analysis
    with open(SEQUENCES_JSON) as f:
        sequences = json.load(f)

    # Load loop features
    loop_df = pd.DataFrame()
    if LOOP_POS_CSV.exists():
        loop_df = pd.read_csv(LOOP_POS_CSV)
        loop_df["site_id"] = loop_df["site_id"].astype(str)
        loop_df = loop_df.set_index("site_id")
        logger.info("  %d loop features loaded", len(loop_df))

    # Check for condition column
    has_condition = "condition" in df_a3g.columns
    logger.info("  Condition column available: %s", has_condition)

    # Check for rate data
    has_rate = "editing_rate" in df_a3g.columns
    if has_rate:
        rate_valid = df_a3g["editing_rate"].notna()
        n_with_rate = int(rate_valid.sum())
        logger.info("  Sites with editing rate: %d/%d", n_with_rate, len(df_a3g))
    else:
        n_with_rate = 0
    results = {
        "enzyme": "A3G",
        "n_total": len(df_a3g),
        "n_with_rate": n_with_rate,
    }

    if n_with_rate < 10:
        results["rate_analysis"] = "insufficient data (fewer than 10 sites with rates)"
        logger.warning("Fewer than 10 sites have editing rates. Skipping rate analysis.")
    else:
        rates = df_a3g.loc[rate_valid, "editing_rate"].values.astype(float)
        results["rate_distribution"] = {
            "mean": float(np.mean(rates)),
            "median": float(np.median(rates)),
            "std": float(np.std(rates)),
            "min": float(np.min(rates)),
            "max": float(np.max(rates)),
            "q25": float(np.percentile(rates, 25)),
            "q75": float(np.percentile(rates, 75)),
        }
        logger.info("  Rate distribution: mean=%.4f, median=%.4f, std=%.4f",
                    np.mean(rates), np.median(rates), np.std(rates))

        # --- Condition comparison (NK_Hyp vs NK_Norm) ---
        if has_condition:
            conditions = df_a3g["condition"].unique()
            logger.info("  Conditions found: %s", list(conditions))
            results["conditions"] = list(str(c) for c in conditions)

            cond_rates = {}
            for cond in conditions:
                mask = (df_a3g["condition"] == cond) & rate_valid
                cond_r = df_a3g.loc[mask, "editing_rate"].values.astype(float)
                if len(cond_r) > 0:
                    cond_rates[str(cond)] = {
                        "n": len(cond_r),
                        "mean": float(np.mean(cond_r)),
                        "median": float(np.median(cond_r)),
                        "std": float(np.std(cond_r)),
                    }

            if len(cond_rates) >= 2:
                cond_names = sorted(cond_rates.keys())
                c1, c2 = cond_names[0], cond_names[1]
                mask1 = (df_a3g["condition"] == c1) & rate_valid
                mask2 = (df_a3g["condition"] == c2) & rate_valid
                r1 = df_a3g.loc[mask1, "editing_rate"].values.astype(float)
                r2 = df_a3g.loc[mask2, "editing_rate"].values.astype(float)
                if len(r1) >= 5 and len(r2) >= 5:
                    stat, p = mannwhitneyu(r1, r2, alternative="two-sided")
                    cond_rates["comparison"] = {
                        "test": "Mann-Whitney U",
                        "groups": [c1, c2],
                        "statistic": float(stat),
                        "p_value": float(p),
                    }
                    logger.info("  %s vs %s: U=%.1f, p=%.4e", c1, c2, stat, p)

            results["condition_rates"] = cond_rates

        # --- Correlations with structural features ---
        site_ids_with_rate = df_a3g.loc[rate_valid, "site_id"].astype(str).values

        # RLP correlation
        rlp_values = []
        rate_values_rlp = []
        for i, sid in enumerate(site_ids_with_rate):
            if sid in loop_df.index and "relative_loop_position" in loop_df.columns:
                rlp = loop_df.loc[sid, "relative_loop_position"]
                if not np.isnan(rlp):
                    rlp_values.append(float(rlp))
                    idx = df_a3g.index[df_a3g["site_id"].astype(str) == sid][0]
                    rate_values_rlp.append(float(df_a3g.loc[idx, "editing_rate"]))

        if len(rlp_values) >= 5:
            rho, p = spearmanr(rate_values_rlp, rlp_values)
            results["rate_vs_rlp"] = {
                "n": len(rlp_values),
                "spearman_rho": float(rho),
                "p_value": float(p),
            }
            logger.info("  Rate vs RLP: rho=%.4f, p=%.4e (n=%d)", rho, p, len(rlp_values))

            # Generate figure
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(rlp_values, rate_values_rlp, alpha=0.6, s=30, c="#2563eb")
            ax.set_xlabel("Relative Loop Position (RLP)")
            ax.set_ylabel("Editing Rate")
            ax.set_title(f"A3G: Rate vs RLP (Spearman rho={rho:.3f}, p={p:.2e})")
            fig.tight_layout()
            fig.savefig(FIGURES_DIR / "rate_vs_rlp_a3g.png", dpi=150)
            plt.close(fig)
            logger.info("  Figure saved: rate_vs_rlp_a3g.png")
        else:
            results["rate_vs_rlp"] = "insufficient overlapping data"

        # Loop size correlation
        ls_values = []
        rate_values_ls = []
        for sid in site_ids_with_rate:
            if sid in loop_df.index and "loop_size" in loop_df.columns:
                ls = loop_df.loc[sid, "loop_size"]
                if not np.isnan(ls) and ls > 0:
                    ls_values.append(float(ls))
                    idx = df_a3g.index[df_a3g["site_id"].astype(str) == sid][0]
                    rate_values_ls.append(float(df_a3g.loc[idx, "editing_rate"]))

        if len(ls_values) >= 5:
            rho, p = spearmanr(rate_values_ls, ls_values)
            results["rate_vs_loop_size"] = {
                "n": len(ls_values),
                "spearman_rho": float(rho),
                "p_value": float(p),
            }
            logger.info("  Rate vs loop_size: rho=%.4f, p=%.4e (n=%d)", rho, p, len(ls_values))
        else:
            results["rate_vs_loop_size"] = "insufficient overlapping data"

        # CC motif correlation
        cc_flags = []
        rate_values_cc = []
        for sid in site_ids_with_rate:
            seq = sequences.get(str(sid), "N" * 201).upper().replace("T", "U")
            ep = 100
            is_cc = 1 if ep > 0 and seq[ep - 1] == "C" else 0
            cc_flags.append(is_cc)
            idx = df_a3g.index[df_a3g["site_id"].astype(str) == sid][0]
            rate_values_cc.append(float(df_a3g.loc[idx, "editing_rate"]))

        if len(cc_flags) >= 5:
            rho, p = spearmanr(rate_values_cc, cc_flags)
            results["rate_vs_cc_motif"] = {
                "n": len(cc_flags),
                "n_cc": int(sum(cc_flags)),
                "n_non_cc": int(len(cc_flags) - sum(cc_flags)),
                "spearman_rho": float(rho),
                "p_value": float(p),
            }
            logger.info("  Rate vs CC: rho=%.4f, p=%.4e (n_cc=%d, n_non_cc=%d)",
                        rho, p, sum(cc_flags), len(cc_flags) - sum(cc_flags))

            # Mann-Whitney: CC vs non-CC rates
            cc_rates = [r for r, f in zip(rate_values_cc, cc_flags) if f == 1]
            non_cc_rates = [r for r, f in zip(rate_values_cc, cc_flags) if f == 0]
            if len(cc_rates) >= 3 and len(non_cc_rates) >= 3:
                stat, p_mw = mannwhitneyu(cc_rates, non_cc_rates, alternative="two-sided")
                results["rate_vs_cc_motif"]["mann_whitney_U"] = float(stat)
                results["rate_vs_cc_motif"]["mann_whitney_p"] = float(p_mw)
                results["rate_vs_cc_motif"]["cc_rate_mean"] = float(np.mean(cc_rates))
                results["rate_vs_cc_motif"]["non_cc_rate_mean"] = float(np.mean(non_cc_rates))
                logger.info("  CC vs non-CC rates: U=%.1f, p=%.4e (CC mean=%.4f, non-CC mean=%.4f)",
                            stat, p_mw, np.mean(cc_rates), np.mean(non_cc_rates))

    # Save results
    out_path = OUTPUT_DIR / "rate_results_a3g.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    elapsed = time.time() - t_start
    logger.info("\nResults saved to %s (%.1f sec)", out_path, elapsed)


if __name__ == "__main__":
    main()
