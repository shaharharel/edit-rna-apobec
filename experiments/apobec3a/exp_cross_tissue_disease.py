#!/usr/bin/env python
"""Iteration 4: Cross-tissue editing profile analysis + disease associations.

Analyzes:
1. 54-tissue GTEx editing profile clustering (from Levanon T1)
2. Tissue module correlations with edit embeddings
3. Cancer survival associations (from Levanon T5)
4. Gene enrichment analysis on high-confidence editing sites
5. Levanon-specific biological characterization

Usage:
    python experiments/apobec3a/exp_cross_tissue_disease.py
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logger = logging.getLogger(__name__)

ADVISOR_DIR = PROJECT_ROOT / "data" / "processed" / "advisor"
EMB_DIR = PROJECT_ROOT / "data" / "processed" / "embeddings"
SPLITS_CSV = PROJECT_ROOT / "data" / "processed" / "splits_expanded.csv"
LABELS_CSV = PROJECT_ROOT / "data" / "processed" / "editing_sites_labels.csv"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "apobec3a" / "outputs" / "cross_tissue_disease"


def parse_tissue_rate(val):
    """Extract editing rate from 'reads;total_reads;rate' format."""
    if pd.isna(val) or val == "" or val == "nan":
        return np.nan
    parts = str(val).split(";")
    if len(parts) == 3:
        try:
            return float(parts[2])
        except ValueError:
            return np.nan
    return np.nan


def load_tissue_rate_matrix():
    """Load the 636×54 tissue rate matrix from Levanon T1."""
    t1_path = ADVISOR_DIR / "t1_gtex_editing_&_conservation.csv"
    if not t1_path.exists():
        logger.warning("T1 GTEx CSV not found at %s", t1_path)
        return None, None, None

    t1 = pd.read_csv(t1_path)
    logger.info("Loaded T1 GTEx sheet: %d sites, %d columns", len(t1), len(t1.columns))

    # Identify tissue columns (exclude known non-tissue columns)
    non_tissue_cols = {
        "Chr", "Start", "End", "Genomic_Category", "Gene_(RefSeq)",
        "mRNA_location_(RefSeq)", "Exonic_Function", "Edited_In_#_Tissues",
        "Edited_Tissues_(Z_score_≥_2)", "Tissue_Classification",
        "Affecting_Over_Expressed_APOBEC", "Max_GTEx_Editing_Rate",
        "Mean_GTEx_Editing_Rate", "GTEx_Editing_Rate_SD",
        "Any_Non-Primate_Editing", "Any_Non-Primate_Editing_≥_1%",
        "Any_Primate_Editing", "Any_Primate_Editing_≥_1%",
        "Any_Mammalian_Editing", "Any_Mammlian_Editing_≥_1%",
        "non-Boreoeutheria_(Primitve_mammals)",
        "Laurasiatheria_(non_rodent_or_primate_placental_mammals)",
        "Glires_(rodents_&_rabbits)",
        "non-Catarrhini_Primates_(new_world_monekys_and_lemurs)",
        "Cercopithecinae_(most_old_world_monkeys)",
        "Laurasiatherianon-Human_Homininae_(Apes)",
    }
    tissue_cols = [c for c in t1.columns if c not in non_tissue_cols]
    logger.info("Identified %d tissue columns", len(tissue_cols))

    # Build rate matrix
    rate_matrix = pd.DataFrame(index=t1.index)
    for tc in tissue_cols:
        rate_matrix[tc] = t1[tc].apply(parse_tissue_rate)

    # Build site ID mapping (using Chr:Start:End)
    site_info = t1[["Chr", "Start", "End"]].copy()
    if "Gene_(RefSeq)" in t1.columns:
        site_info["gene"] = t1["Gene_(RefSeq)"]

    return rate_matrix, tissue_cols, site_info


def run_tissue_clustering(rate_matrix, tissue_cols):
    """Cluster sites by their 54-tissue editing profile."""
    logger.info("Running tissue profile clustering...")

    # Fill NaN with 0 for clustering
    rm = rate_matrix[tissue_cols].fillna(0).values
    logger.info("Rate matrix shape: %s", rm.shape)

    # Remove sites with no editing across all tissues
    has_editing = rm.sum(axis=1) > 0
    rm_active = rm[has_editing]
    active_idx = np.where(has_editing)[0]
    logger.info("Sites with any tissue editing: %d / %d", len(active_idx), len(rm))

    if len(rm_active) < 10:
        logger.warning("Too few active sites for clustering")
        return None

    # Hierarchical clustering on sites
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import pdist

    # Normalize rows (per-site profile)
    row_sums = rm_active.sum(axis=1, keepdims=True)
    rm_norm = rm_active / np.maximum(row_sums, 1e-8)

    dist = pdist(rm_norm, metric="cosine")
    Z_sites = linkage(dist, method="average")

    # Cut at different levels
    site_clusters = {}
    for n_clust in [3, 5, 8]:
        labels = fcluster(Z_sites, n_clust, criterion="maxclust")
        site_clusters[n_clust] = labels

    # Tissue clustering
    rm_t = rm_active.T
    dist_t = pdist(rm_t, metric="cosine")
    Z_tissues = linkage(dist_t, method="average")
    tissue_labels = fcluster(Z_tissues, 5, criterion="maxclust")
    tissue_to_cluster = {tissue_cols[i]: int(tissue_labels[i]) for i in range(len(tissue_cols))}

    return {
        "active_indices": active_idx.tolist(),
        "site_clusters": {str(k): v.tolist() for k, v in site_clusters.items()},
        "tissue_clusters": tissue_to_cluster,
        "n_active": len(active_idx),
    }


def compute_tissue_correlations(rate_matrix, tissue_cols):
    """Compute pairwise tissue correlations and tissue modules."""
    logger.info("Computing tissue correlations...")

    # Tissue modules
    tissue_modules = {
        "Brain": [c for c in tissue_cols if "Brain" in c],
        "Blood/Immune": [c for c in tissue_cols if any(x in c for x in ["Blood", "Spleen", "lymphocyte"])],
        "Cardiovascular": [c for c in tissue_cols if any(x in c for x in ["Heart", "Artery"])],
        "GI": [c for c in tissue_cols if any(x in c for x in [
            "Colon", "Esophagus", "Stomach", "Small_Intestine", "Liver", "Pancreas"
        ])],
        "Reproductive": [c for c in tissue_cols if any(x in c for x in [
            "Ovary", "Testis", "Uterus", "Vagina", "Fallopian", "Cervix", "Prostate", "Breast"
        ])],
        "Skin": [c for c in tissue_cols if "Skin" in c],
        "Kidney": [c for c in tissue_cols if "Kidney" in c],
        "Lung": [c for c in tissue_cols if c == "Lung"],
        "Other": [c for c in tissue_cols if any(x in c for x in [
            "Nerve", "Muscle", "Pituitary", "Thyroid", "Adrenal", "Bladder", "Minor_Salivary"
        ])],
    }

    # Module-level correlations
    module_means = pd.DataFrame()
    for module, tissues in tissue_modules.items():
        if tissues:
            module_means[module] = rate_matrix[tissues].mean(axis=1)

    module_corr = module_means.corr(method="pearson")

    # Most and least correlated tissue pairs
    n = len(tissue_cols)
    all_corrs = []
    for i in range(n):
        for j in range(i + 1, n):
            vals_i = rate_matrix[tissue_cols[i]].values
            vals_j = rate_matrix[tissue_cols[j]].values
            mask = ~(np.isnan(vals_i) | np.isnan(vals_j))
            if mask.sum() < 10:
                continue
            r, p = pearsonr(vals_i[mask], vals_j[mask])
            all_corrs.append({
                "tissue_1": tissue_cols[i],
                "tissue_2": tissue_cols[j],
                "pearson_r": float(r),
                "p_value": float(p),
                "n_shared": int(mask.sum()),
            })

    all_corrs.sort(key=lambda x: x["pearson_r"], reverse=True)

    return {
        "tissue_modules": {k: v for k, v in tissue_modules.items()},
        "module_correlation_matrix": module_corr.to_dict(),
        "top_correlated_pairs": all_corrs[:10],
        "least_correlated_pairs": all_corrs[-10:],
    }


def analyze_cancer_survival(labels_df):
    """Analyze cancer survival associations from Levanon data."""
    logger.info("Analyzing cancer survival associations...")

    if "has_survival_association" not in labels_df.columns:
        return {}

    df = labels_df.dropna(subset=["has_survival_association"])
    n_with = (df["has_survival_association"] == True).sum()
    n_without = (df["has_survival_association"] == False).sum()
    logger.info("Sites with survival associations: %d / %d", n_with, n_with + n_without)

    # Parse cancer types
    cancer_counter = Counter()
    for cts in df["cancer_types_survival"].dropna():
        for ct in str(cts).split(";"):
            ct = ct.strip()
            if ct:
                cancer_counter[ct] += 1

    # Per-cancer enrichment analysis
    cancer_info = {}
    for cancer, count in cancer_counter.most_common(20):
        sites_with = df[df["cancer_types_survival"].str.contains(cancer, na=False)]
        if len(sites_with) > 0:
            # Characterize sites associated with this cancer
            info = {"n_sites": count}
            if "genomic_category" in sites_with.columns:
                info["genomic_cats"] = dict(Counter(sites_with["genomic_category"].dropna()))
            if "tissue_class" in sites_with.columns:
                info["tissue_classes"] = dict(Counter(sites_with["tissue_class"].dropna()))
            if "structure_type" in sites_with.columns:
                info["structure_types"] = dict(Counter(sites_with["structure_type"].dropna()))
            if "max_gtex_rate" in sites_with.columns:
                rates = pd.to_numeric(sites_with["max_gtex_rate"], errors="coerce").dropna()
                if len(rates) > 0:
                    info["mean_max_rate"] = float(rates.mean())
            cancer_info[cancer] = info

    return {
        "n_with_survival": int(n_with),
        "n_without_survival": int(n_without),
        "cancer_type_counts": dict(cancer_counter.most_common()),
        "cancer_details": cancer_info,
    }


def analyze_tissue_embedding_correlation(rate_matrix, tissue_cols, site_info, pooled_orig, pooled_edited):
    """Correlate tissue editing profiles with edit effect embeddings."""
    logger.info("Correlating tissue profiles with embeddings...")

    # Build mapping from T1 sites to site_ids in our embedding cache
    # T1 uses Chr:Start:End, our cache uses C2U_XXXX IDs
    # Need to cross-reference via splits_expanded.csv
    splits_df = pd.read_csv(SPLITS_CSV)
    levanon_sites = splits_df[splits_df["dataset_source"] == "advisor_c2t"]

    # Map by chr:start
    levanon_map = {}
    for _, row in levanon_sites.iterrows():
        key = (str(row["chr"]), int(row["start"]))
        levanon_map[key] = row["site_id"]

    # Map T1 rows to site_ids
    matched = 0
    site_id_to_t1_idx = {}
    for i, row in site_info.iterrows():
        key = (str(row["Chr"]).lower().replace("chr", "chr"), int(row["Start"]))
        # Also try with "chr" prefix
        key2 = ("chr" + str(row["Chr"]).lower().replace("chr", ""), int(row["Start"]))
        for k in [key, key2]:
            if k in levanon_map:
                sid = levanon_map[k]
                if sid in pooled_orig:
                    site_id_to_t1_idx[sid] = i
                    matched += 1
                    break

    logger.info("Matched %d T1 sites to embeddings", matched)

    if matched < 20:
        return {"n_matched": matched}

    # Compute edit diff embeddings for matched sites
    sids = list(site_id_to_t1_idx.keys())
    diffs = []
    for sid in sids:
        diff = pooled_edited[sid] - pooled_orig[sid]
        diffs.append(diff.numpy())
    diffs = np.array(diffs)

    # Get tissue rate profiles for matched sites
    t1_indices = [site_id_to_t1_idx[sid] for sid in sids]
    rates = rate_matrix.iloc[t1_indices][tissue_cols].fillna(0).values

    # PCA on embeddings
    from sklearn.decomposition import PCA
    pca = PCA(n_components=10, random_state=42)
    diffs_pca = pca.fit_transform(diffs)

    # Correlate each PC with tissue rates
    pc_tissue_corrs = {}
    for pc_idx in range(min(5, diffs_pca.shape[1])):
        pc_vals = diffs_pca[:, pc_idx]
        best_tissue = None
        best_r = 0
        tissue_rs = {}
        for tc_idx, tc in enumerate(tissue_cols):
            tc_rates = rates[:, tc_idx]
            mask = tc_rates > 0
            if mask.sum() < 10:
                continue
            r, p = pearsonr(pc_vals[mask], tc_rates[mask])
            tissue_rs[tc] = {"r": float(r), "p": float(p)}
            if abs(r) > abs(best_r):
                best_r = r
                best_tissue = tc

        pc_tissue_corrs[f"PC{pc_idx+1}"] = {
            "best_tissue": best_tissue,
            "best_r": float(best_r),
            "variance_explained": float(pca.explained_variance_ratio_[pc_idx]),
        }

    # Overall: mean editing rate vs embedding magnitude
    mean_rates = rates.mean(axis=1)
    emb_magnitude = np.linalg.norm(diffs, axis=1)
    r_mag, p_mag = pearsonr(mean_rates[mean_rates > 0], emb_magnitude[mean_rates > 0])

    return {
        "n_matched": matched,
        "pc_tissue_correlations": pc_tissue_corrs,
        "rate_vs_magnitude": {
            "pearson_r": float(r_mag),
            "p_value": float(p_mag),
        },
    }


def analyze_gene_enrichment(labels_df):
    """Analyze gene-level enrichment for edited sites."""
    logger.info("Analyzing gene enrichment...")

    if "gene_name" not in labels_df.columns:
        return {}

    # Gene with most editing sites
    gene_counts = Counter(labels_df["gene_name"].dropna())
    top_genes = gene_counts.most_common(30)

    # Functional categories
    func_cats = Counter(labels_df["exonic_function"].dropna()) if "exonic_function" in labels_df.columns else {}

    # APOBEC class distribution
    apobec_dist = Counter(labels_df["apobec_class"].dropna()) if "apobec_class" in labels_df.columns else {}

    return {
        "top_edited_genes": {g: c for g, c in top_genes},
        "exonic_function_dist": dict(func_cats),
        "apobec_class_dist": dict(apobec_dist),
        "n_unique_genes": len(gene_counts),
    }


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # ===================================================================
    # Part 1: Cross-tissue analysis from Levanon T1
    # ===================================================================
    logger.info("=" * 70)
    logger.info("PART 1: Cross-Tissue GTEx Editing Profile Analysis")
    logger.info("=" * 70)

    rate_matrix, tissue_cols, site_info = load_tissue_rate_matrix()
    if rate_matrix is not None:
        # 1a. Tissue statistics
        print("\n--- Top 10 Tissues by Mean Editing Rate ---")
        mean_rates = rate_matrix.mean().sort_values(ascending=False)
        for tissue, rate in mean_rates.head(10).items():
            n_sites = rate_matrix[tissue].notna().sum()
            print(f"  {tissue:50s} mean_rate={rate:7.3f}%  (n={n_sites})")

        all_results["tissue_mean_rates"] = {
            tissue: {"mean_rate": float(rate), "n_sites": int(rate_matrix[tissue].notna().sum())}
            for tissue, rate in mean_rates.items()
        }

        # 1b. Tissue correlations
        corr_results = compute_tissue_correlations(rate_matrix, tissue_cols)
        all_results["tissue_correlations"] = corr_results

        print("\n--- Top 5 Most Correlated Tissue Pairs ---")
        for pair in corr_results["top_correlated_pairs"][:5]:
            print(f"  {pair['tissue_1']:30s} <-> {pair['tissue_2']:30s} r={pair['pearson_r']:.3f}")

        print("\n--- Top 5 Least Correlated Tissue Pairs ---")
        for pair in corr_results["least_correlated_pairs"][:5]:
            print(f"  {pair['tissue_1']:30s} <-> {pair['tissue_2']:30s} r={pair['pearson_r']:.3f}")

        print("\n--- Module Correlation Matrix ---")
        mod_corr = corr_results["module_correlation_matrix"]
        modules = list(mod_corr.keys())
        header = f"{'':>20s}"
        for m in modules:
            header += f" {m[:8]:>8s}"
        print(header)
        for m1 in modules:
            row = f"{m1:>20s}"
            for m2 in modules:
                val = mod_corr[m1].get(m2, float("nan"))
                row += f" {val:>8.3f}"
            print(row)

        # 1c. Tissue clustering
        cluster_results = run_tissue_clustering(rate_matrix, tissue_cols)
        if cluster_results:
            all_results["tissue_clustering"] = cluster_results
            print(f"\n--- Tissue Clustering ---")
            print(f"Active sites (with editing): {cluster_results['n_active']}")
            print(f"Tissue cluster assignments:")
            for tissue, cl in sorted(cluster_results["tissue_clusters"].items(), key=lambda x: x[1]):
                print(f"  Cluster {cl}: {tissue}")

        # 1d. Tissue-embedding correlation
        pooled_orig = torch.load(EMB_DIR / "rnafm_pooled.pt", weights_only=False)
        pooled_edited = torch.load(EMB_DIR / "rnafm_pooled_edited.pt", weights_only=False)
        emb_corr = analyze_tissue_embedding_correlation(
            rate_matrix, tissue_cols, site_info, pooled_orig, pooled_edited
        )
        all_results["tissue_embedding_correlation"] = emb_corr
        if "pc_tissue_correlations" in emb_corr:
            print("\n--- Embedding PC ↔ Tissue Correlations ---")
            for pc, info in emb_corr["pc_tissue_correlations"].items():
                print(f"  {pc} (var={info['variance_explained']:.1%}): "
                      f"best_tissue={info['best_tissue']}, r={info['best_r']:.3f}")
    else:
        logger.warning("Skipping tissue analysis: T1 data not available")

    # ===================================================================
    # Part 2: Cancer survival analysis
    # ===================================================================
    logger.info("\n" + "=" * 70)
    logger.info("PART 2: Cancer Survival Associations")
    logger.info("=" * 70)

    labels_df = pd.read_csv(LABELS_CSV)
    cancer_results = analyze_cancer_survival(labels_df)
    all_results["cancer_survival"] = cancer_results

    if cancer_results:
        print("\n--- Cancer Survival Associations ---")
        print(f"Sites with survival associations: {cancer_results.get('n_with_survival', 0)}")
        print(f"Sites without: {cancer_results.get('n_without_survival', 0)}")
        print("\nTop cancers:")
        for cancer, count in list(cancer_results.get("cancer_type_counts", {}).items())[:10]:
            detail = cancer_results.get("cancer_details", {}).get(cancer, {})
            rate = detail.get("mean_max_rate", 0)
            print(f"  {cancer}: {count} sites, mean_max_rate={rate:.1f}%")

    # ===================================================================
    # Part 3: Gene enrichment analysis
    # ===================================================================
    logger.info("\n" + "=" * 70)
    logger.info("PART 3: Gene Enrichment Analysis")
    logger.info("=" * 70)

    gene_results = analyze_gene_enrichment(labels_df)
    all_results["gene_enrichment"] = gene_results

    if gene_results:
        print(f"\n--- Gene-Level Analysis ---")
        print(f"Unique genes with editing sites: {gene_results.get('n_unique_genes', 0)}")
        print("\nTop 15 most-edited genes:")
        for gene, count in list(gene_results.get("top_edited_genes", {}).items())[:15]:
            print(f"  {gene}: {count} sites")

        print("\nExonic function distribution:")
        for func, count in sorted(gene_results.get("exonic_function_dist", {}).items(),
                                   key=lambda x: -x[1]):
            print(f"  {func}: {count}")

        print("\nAPOBEC class distribution:")
        for cls, count in sorted(gene_results.get("apobec_class_dist", {}).items(),
                                  key=lambda x: -x[1]):
            print(f"  {cls}: {count}")

    # ===================================================================
    # Part 4: Levanon vs other datasets biological comparison
    # ===================================================================
    logger.info("\n" + "=" * 70)
    logger.info("PART 4: Cross-Dataset Biological Comparison")
    logger.info("=" * 70)

    splits_df = pd.read_csv(SPLITS_CSV)
    ds_stats = {}
    for ds in ["advisor_c2t", "asaoka_2019", "sharma_2015", "alqassim_2021"]:
        ds_df = splits_df[splits_df["dataset_source"] == ds]
        ds_label = {
            "advisor_c2t": "Levanon",
            "asaoka_2019": "Asaoka",
            "sharma_2015": "Sharma",
            "alqassim_2021": "Alqassim",
            "baysal_2016": "Baysal",
        }.get(ds, ds)
        rates = pd.to_numeric(ds_df["editing_rate"], errors="coerce").dropna()
        ds_stats[ds_label] = {
            "n_sites": len(ds_df),
            "n_genes": len(ds_df["gene"].dropna().unique()),
            "chromosomes": len(ds_df["chr"].unique()),
            "mean_rate": float(rates.mean()) if len(rates) > 0 else None,
            "median_rate": float(rates.median()) if len(rates) > 0 else None,
            "max_rate": float(rates.max()) if len(rates) > 0 else None,
        }
    all_results["dataset_statistics"] = ds_stats

    print("\n--- Dataset Statistics ---")
    print(f"{'Dataset':<15} {'Sites':>6} {'Genes':>6} {'Mean Rate':>10} {'Max Rate':>10}")
    print("-" * 50)
    for ds, stats in ds_stats.items():
        mr = f"{stats['mean_rate']:.2f}%" if stats["mean_rate"] else "N/A"
        mx = f"{stats['max_rate']:.2f}%" if stats["max_rate"] else "N/A"
        print(f"{ds:<15} {stats['n_sites']:>6d} {stats['n_genes']:>6d} {mr:>10s} {mx:>10s}")

    print("=" * 80)

    # Save all results
    def serialize(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, set):
            return list(obj)
        return str(obj)

    with open(OUTPUT_DIR / "cross_tissue_disease_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=serialize)

    logger.info("\nResults saved to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
