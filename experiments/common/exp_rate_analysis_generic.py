#!/usr/bin/env python
"""Generic rate analysis for any enzyme category with GTEx tissue rates.

Analyzes editing rate distributions, tissue-specific patterns, and rate
prediction using the Levanon 54-tissue GTEx rates.

Usage:
    conda run -n quris python experiments/common/exp_rate_analysis_generic.py --enzyme A3A_A3G
    conda run -n quris python experiments/common/exp_rate_analysis_generic.py --enzyme Neither
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_ME_DIR = PROJECT_ROOT / "data/processed/multi_enzyme"
LEVANON_CSV = _ME_DIR / "levanon_all_categories.csv"
TISSUE_RATES = _ME_DIR / "levanon_tissue_rates.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--enzyme", required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    enzyme_dir_map = {
        "A3G": "apobec3g", "A3A_A3G": "apobec_both",
        "Neither": "apobec_neither", "Unknown": "apobec_unknown",
        "A3A": "apobec3a",
    }
    exp_dir = enzyme_dir_map.get(args.enzyme, f"apobec_{args.enzyme.lower()}")
    output_dir = args.output_dir or (PROJECT_ROOT / "experiments" / exp_dir / "outputs" / "rate_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    lev = pd.read_csv(LEVANON_CSV)
    tissue = pd.read_csv(TISSUE_RATES)
    enzyme = args.enzyme

    # Filter to enzyme category
    lev_sub = lev[lev["enzyme_category"] == enzyme].copy()
    tissue_sub = tissue[tissue["enzyme_category"] == enzyme].copy()
    logger.info("%s: %d sites", enzyme, len(lev_sub))

    if len(lev_sub) == 0:
        logger.error("No sites for %s", enzyme)
        return

    # Get tissue columns (excluding site_id and enzyme_category)
    tissue_cols = [c for c in tissue.columns if c not in ["site_id", "enzyme_category"]]

    # =========================================================================
    # 1. Rate distribution statistics
    # =========================================================================
    results = {"enzyme": enzyme, "n_sites": len(lev_sub)}

    if "editing_rate" in lev_sub.columns:
        rates = lev_sub["editing_rate"].dropna()
        results["rate_stats"] = {
            "mean": float(rates.mean()),
            "median": float(rates.median()),
            "std": float(rates.std()),
            "min": float(rates.min()),
            "max": float(rates.max()),
            "n_with_rate": len(rates),
        }
        logger.info("  Mean rate: %.4f, Median: %.4f", rates.mean(), rates.median())

    # =========================================================================
    # 2. Tissue-level analysis
    # =========================================================================
    tissue_means = {}
    tissue_coverage = {}
    for col in tissue_cols:
        vals = tissue_sub[col].dropna()
        if len(vals) > 0:
            tissue_means[col] = float(vals.mean())
            tissue_coverage[col] = len(vals)

    # Sort tissues by mean editing rate
    sorted_tissues = sorted(tissue_means.items(), key=lambda x: -x[1])
    results["top_tissues"] = [
        {"tissue": t, "mean_rate_pct": r, "n_sites_with_rate": tissue_coverage.get(t, 0)}
        for t, r in sorted_tissues[:20]
    ]
    results["bottom_tissues"] = [
        {"tissue": t, "mean_rate_pct": r, "n_sites_with_rate": tissue_coverage.get(t, 0)}
        for t, r in sorted_tissues[-10:]
    ]

    logger.info("  Top 5 tissues by editing rate:")
    for t, r in sorted_tissues[:5]:
        logger.info("    %s: %.2f%%", t, r)

    # =========================================================================
    # 3. Tissue classification distribution
    # =========================================================================
    if "tissue_classification" in lev_sub.columns:
        tc = lev_sub["tissue_classification"].value_counts()
        results["tissue_classification"] = tc.to_dict()
        logger.info("  Tissue classification: %s", tc.to_dict())

    # =========================================================================
    # 4. Compare to other enzyme categories
    # =========================================================================
    comparisons = {}
    all_categories = tissue["enzyme_category"].unique()
    for other in all_categories:
        if other == enzyme:
            continue
        other_sub = tissue[tissue["enzyme_category"] == other]
        if len(other_sub) < 5:
            continue

        # Compare mean editing rate across tissues
        enzyme_tissue_means = np.array([tissue_means.get(c, np.nan) for c in tissue_cols])
        other_tissue_means = np.array([
            float(other_sub[c].dropna().mean()) if len(other_sub[c].dropna()) > 0 else np.nan
            for c in tissue_cols
        ])

        # Remove NaN pairs
        valid = ~(np.isnan(enzyme_tissue_means) | np.isnan(other_tissue_means))
        if valid.sum() >= 10:
            r, p = stats.spearmanr(enzyme_tissue_means[valid], other_tissue_means[valid])
            comparisons[other] = {"spearman_r": float(r), "p_value": float(p),
                                   "n_tissues": int(valid.sum())}

    results["tissue_correlations"] = comparisons
    if comparisons:
        logger.info("  Tissue rate correlations with other enzymes:")
        for other, c in sorted(comparisons.items(), key=lambda x: -abs(x[1]["spearman_r"])):
            logger.info("    vs %s: r=%.3f (p=%.2e)", other, c["spearman_r"], c["p_value"])

    # =========================================================================
    # 5. Motif analysis
    # =========================================================================
    seqs_path = PROJECT_ROOT / "data/processed/site_sequences.json"
    if seqs_path.exists():
        with open(seqs_path) as f:
            seqs = json.load(f)

        tc_count = cc_count = ac_count = gc_count = 0
        for sid in lev_sub["site_id"]:
            seq = seqs.get(str(sid), "N" * 201)
            if len(seq) >= 102:
                up = seq[99].upper()
                if up in ("U", "T"): tc_count += 1
                elif up == "C": cc_count += 1
                elif up == "A": ac_count += 1
                elif up == "G": gc_count += 1

        n = len(lev_sub)
        results["motif"] = {
            "TC": tc_count, "CC": cc_count, "AC": ac_count, "GC": gc_count,
            "TC_pct": tc_count / n * 100 if n else 0,
            "CC_pct": cc_count / n * 100 if n else 0,
        }
        logger.info("  Motif: TC=%.1f%%, CC=%.1f%%",
                     tc_count / n * 100 if n else 0, cc_count / n * 100 if n else 0)

    # =========================================================================
    # 6. Genomic annotation
    # =========================================================================
    if "genomic_category" in lev_sub.columns:
        results["genomic_category"] = lev_sub["genomic_category"].value_counts().to_dict()
    if "exonic_function" in lev_sub.columns:
        results["exonic_function"] = lev_sub["exonic_function"].value_counts().to_dict()

    # =========================================================================
    # 7. Edited tissues breadth
    # =========================================================================
    if "edited_in_num_tissues" in lev_sub.columns:
        breadth = lev_sub["edited_in_num_tissues"].dropna()
        results["tissue_breadth"] = {
            "mean": float(breadth.mean()),
            "median": float(breadth.median()),
            "min": int(breadth.min()),
            "max": int(breadth.max()),
        }
        logger.info("  Tissue breadth: mean=%.1f, median=%.0f",
                     breadth.mean(), breadth.median())

    # Save
    with open(output_dir / "rate_analysis_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Saved results to %s", output_dir / "rate_analysis_results.json")

    # Save tissue rate summary
    if tissue_means:
        tissue_summary = pd.DataFrame([
            {"tissue": t, "mean_rate_pct": r, "n_sites": tissue_coverage.get(t, 0)}
            for t, r in sorted_tissues
        ])
        tissue_summary.to_csv(output_dir / "tissue_rate_summary.csv", index=False)

    logger.info("=== %s Rate Analysis Complete ===", enzyme)


if __name__ == "__main__":
    main()
