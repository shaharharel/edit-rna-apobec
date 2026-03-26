#!/usr/bin/env python
"""GTEx tissue clustering analysis across enzyme categories.

Clusters 54 GTEx tissues by editing rate profiles across 636 Levanon sites.
Clusters sites by tissue patterns. Generates data for heatmap figure.

Key predictions:
- Immune cluster (blood, spleen) driven by A3A
- GI cluster (intestine, liver) driven by "Neither" (APOBEC1?)
- Low-editing cluster (brain, muscle, adipose)
- A3G sites show distinct tissue pattern

Usage:
    conda run -n quris python experiments/common/exp_tissue_clustering.py
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist, squareform
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TISSUE_RATES = PROJECT_ROOT / "data/processed/multi_enzyme/levanon_tissue_rates.csv"
LEVANON_CSV = PROJECT_ROOT / "data/processed/multi_enzyme/levanon_all_categories.csv"
OUTPUT_DIR = PROJECT_ROOT / "experiments/common/outputs/tissue_clustering"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tissue = pd.read_csv(TISSUE_RATES)
    lev = pd.read_csv(LEVANON_CSV)

    tissue_cols = [c for c in tissue.columns if c not in ["site_id", "enzyme_category"]]
    logger.info("Loaded %d sites × %d tissues", len(tissue), len(tissue_cols))

    # =========================================================================
    # 1. Build rate matrix (sites × tissues)
    # =========================================================================
    rate_matrix = tissue[tissue_cols].values  # 636 × 54
    site_ids = tissue["site_id"].values
    enzyme_cats = tissue["enzyme_category"].values

    # Replace NaN with 0 for clustering
    rate_matrix_filled = np.nan_to_num(rate_matrix, nan=0.0)

    # =========================================================================
    # 2. Tissue clustering (columns)
    # =========================================================================
    logger.info("\n=== Tissue Clustering ===")

    # Spearman correlation between tissues (across sites)
    tissue_corr = np.zeros((len(tissue_cols), len(tissue_cols)))
    for i in range(len(tissue_cols)):
        for j in range(len(tissue_cols)):
            valid = ~(np.isnan(rate_matrix[:, i]) | np.isnan(rate_matrix[:, j]))
            if valid.sum() > 10:
                r, _ = stats.spearmanr(rate_matrix[valid, i], rate_matrix[valid, j])
                tissue_corr[i, j] = r if not np.isnan(r) else 0
            else:
                tissue_corr[i, j] = 0

    # Hierarchical clustering of tissues
    tissue_dist = 1 - tissue_corr
    np.fill_diagonal(tissue_dist, 0)
    tissue_dist = np.maximum(tissue_dist, 0)  # Ensure non-negative
    tissue_condensed = squareform(tissue_dist, checks=False)
    tissue_linkage = linkage(tissue_condensed, method="ward")

    # Cut into 4-5 clusters
    tissue_clusters = fcluster(tissue_linkage, t=4, criterion="maxclust")
    tissue_cluster_map = dict(zip(tissue_cols, tissue_clusters))

    logger.info("Tissue clusters (4 groups):")
    for cl in sorted(set(tissue_clusters)):
        members = [t for t, c in tissue_cluster_map.items() if c == cl]
        logger.info("  Cluster %d (%d tissues): %s", cl, len(members),
                     ", ".join(members[:5]) + ("..." if len(members) > 5 else ""))

    # =========================================================================
    # 3. Per-enzyme mean tissue profile
    # =========================================================================
    logger.info("\n=== Per-Enzyme Tissue Profiles ===")

    enzyme_tissue_profiles = {}
    for cat in ["A3A", "A3G", "A3A_A3G", "Neither", "Unknown"]:
        mask = enzyme_cats == cat
        if mask.sum() == 0:
            continue
        sub_matrix = rate_matrix[mask]
        # Mean rate per tissue (ignoring NaN)
        with np.errstate(all="ignore"):
            means = np.nanmean(sub_matrix, axis=0)
        enzyme_tissue_profiles[cat] = means.tolist()

        # Top 5 tissues
        valid_means = [(tissue_cols[i], means[i]) for i in range(len(means)) if not np.isnan(means[i])]
        sorted_tissues = sorted(valid_means, key=lambda x: -x[1])
        logger.info("  %s top 5: %s", cat,
                     ", ".join(f"{t}={r:.1f}%" for t, r in sorted_tissues[:5]))

    # =========================================================================
    # 4. Cross-enzyme tissue correlation matrix
    # =========================================================================
    logger.info("\n=== Cross-Enzyme Tissue Correlations ===")

    cats = list(enzyme_tissue_profiles.keys())
    corr_matrix = {}
    for i, cat1 in enumerate(cats):
        for j, cat2 in enumerate(cats):
            if i >= j:
                continue
            p1 = np.array(enzyme_tissue_profiles[cat1])
            p2 = np.array(enzyme_tissue_profiles[cat2])
            valid = ~(np.isnan(p1) | np.isnan(p2))
            if valid.sum() > 10:
                r, p = stats.spearmanr(p1[valid], p2[valid])
                logger.info("  %s vs %s: r=%.3f (p=%.2e)", cat1, cat2, r, p)
                corr_matrix[f"{cat1}_vs_{cat2}"] = {"r": float(r), "p": float(p)}

    # =========================================================================
    # 5. Site clustering (rows) — by tissue pattern
    # =========================================================================
    logger.info("\n=== Site Clustering by Tissue Pattern ===")

    # Use only sites with reasonable coverage (>10 non-NaN tissues)
    coverage = np.sum(~np.isnan(rate_matrix), axis=1)
    good_sites = coverage >= 10
    logger.info("Sites with ≥10 tissues: %d/%d", good_sites.sum(), len(coverage))

    good_matrix = rate_matrix_filled[good_sites]
    good_cats = enzyme_cats[good_sites]
    good_sids = site_ids[good_sites]

    if len(good_matrix) > 20:
        # Z-score normalize each site across tissues
        site_means = good_matrix.mean(axis=1, keepdims=True)
        site_stds = good_matrix.std(axis=1, keepdims=True)
        site_stds[site_stds == 0] = 1
        z_matrix = (good_matrix - site_means) / site_stds

        # Cluster sites
        site_dist = pdist(z_matrix, metric="correlation")
        site_dist = np.nan_to_num(site_dist, nan=1.0)
        site_linkage = linkage(site_dist, method="ward")
        site_clusters = fcluster(site_linkage, t=5, criterion="maxclust")

        # Which enzyme categories dominate each cluster?
        logger.info("Site clusters (5 groups) — enzyme composition:")
        for cl in sorted(set(site_clusters)):
            cl_mask = site_clusters == cl
            cl_cats = good_cats[cl_mask]
            cat_counts = pd.Series(cl_cats).value_counts()
            n = cl_mask.sum()
            top_cats = ", ".join(f"{c}={n}" for c, n in cat_counts.head(3).items())
            logger.info("  Cluster %d (n=%d): %s", cl, n, top_cats)

    # =========================================================================
    # 6. Tissue breadth analysis
    # =========================================================================
    logger.info("\n=== Tissue Breadth by Enzyme ===")

    if "edited_in_num_tissues" in lev.columns:
        for cat in ["A3A", "A3G", "A3A_A3G", "Neither", "Unknown"]:
            sub = lev[lev["enzyme_category"] == cat]
            breadth = sub["edited_in_num_tissues"].dropna()
            if len(breadth) > 0:
                logger.info("  %s: mean=%.1f, median=%.0f, max=%d",
                             cat, breadth.mean(), breadth.median(), breadth.max())

    # =========================================================================
    # Save everything
    # =========================================================================
    results = {
        "tissue_clusters": tissue_cluster_map,
        "enzyme_tissue_profiles": {k: list(v) for k, v in enzyme_tissue_profiles.items()},
        "tissue_cols": tissue_cols,
        "cross_enzyme_correlations": corr_matrix,
        "tissue_correlation_matrix": tissue_corr.tolist(),
        "tissue_linkage": tissue_linkage.tolist(),
    }

    with open(OUTPUT_DIR / "tissue_clustering_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Save enzyme tissue profile CSV for plotting
    profile_df = pd.DataFrame(enzyme_tissue_profiles, index=tissue_cols)
    profile_df.to_csv(OUTPUT_DIR / "enzyme_tissue_profiles.csv")

    # Save tissue correlation matrix
    corr_df = pd.DataFrame(tissue_corr, index=tissue_cols, columns=tissue_cols)
    corr_df.to_csv(OUTPUT_DIR / "tissue_correlation_matrix.csv")

    logger.info("\nSaved results to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
