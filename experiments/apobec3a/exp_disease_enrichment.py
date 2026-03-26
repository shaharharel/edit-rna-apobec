#!/usr/bin/env python
"""Disease enrichment analysis for APOBEC3A editing sites.

Performs:
1. ClinVar overlap analysis (editing sites vs pathogenic variants)
2. Gene Ontology enrichment (edited genes vs background)
3. KEGG pathway analysis
4. DisGeNET disease-gene associations (via Enrichr)
5. Per-dataset and per-tissue enrichment comparison

Uses gseapy for enrichment analysis (Enrichr API-based).

Usage:
    python experiments/apobec3a/exp_disease_enrichment.py
"""

import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logger = logging.getLogger(__name__)

COMBINED_PATH = PROJECT_ROOT / "data" / "processed" / "all_datasets_combined.csv"
SPLITS_CSV = PROJECT_ROOT / "data" / "processed" / "splits_expanded.csv"
SPLITS_A3A = PROJECT_ROOT / "data" / "processed" / "splits_expanded_a3a.csv"
NEG_TIER1 = PROJECT_ROOT / "data" / "processed" / "negatives_tier1.csv"
CLINVAR_PATH = PROJECT_ROOT / "data" / "processed" / "clinvar_c2u_variants.csv"
CALIBRATED_SCORES = PROJECT_ROOT / "experiments" / "apobec3a" / "outputs" / "clinvar_calibrated" / "clinvar_calibrated_scores.csv"
LEVANON_PATH = PROJECT_ROOT / "data" / "processed" / "advisor" / "t1_gtex_editing_&_conservation.csv"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "apobec3a" / "outputs" / "disease_enrichment"

DATASET_LABELS = {
    "advisor_c2t": "Levanon",
    "asaoka_2019": "Asaoka",
    "alqassim_2021": "Alqassim",
    "sharma_2015": "Sharma",
    "baysal_2016": "Baysal",
}

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
})


# =====================================================================
# 1. ClinVar Overlap Analysis
# =====================================================================

def clinvar_overlap_analysis(combined_df, clinvar_df, output_dir):
    """Analyze overlap between known editing sites and ClinVar C>T variants."""
    logger.info("\n=== ClinVar Overlap Analysis ===")

    # Known editing site coordinates
    editing_coords = set(zip(combined_df["chr"], combined_df["start"]))
    editing_genes = set(combined_df["gene"].dropna().str.strip())
    editing_genes.discard("")

    # ClinVar C>T variants
    clinvar_coords = set(zip(clinvar_df["chr"], clinvar_df["start"]))

    # Direct coordinate overlap
    overlap_coords = editing_coords & clinvar_coords
    logger.info("Direct coordinate overlap: %d sites", len(overlap_coords))

    # Gene-level overlap
    clinvar_genes = set(clinvar_df["gene"].dropna().str.strip())
    clinvar_genes.discard("")
    clinvar_genes.discard("unknown")
    gene_overlap = editing_genes & clinvar_genes
    logger.info("Gene-level overlap: %d genes (of %d editing, %d ClinVar)",
                len(gene_overlap), len(editing_genes), len(clinvar_genes))

    # Pathogenicity of overlapping variants
    overlap_df = clinvar_df[clinvar_df["is_known_editing_site"] == True].copy()

    if len(overlap_df) > 0:
        sig_counts = overlap_df["clinical_significance"].value_counts()
        logger.info("\nPathogenicity of editing-site ClinVar variants:")
        for sig, count in sig_counts.items():
            logger.info("  %s: %d", sig, count)

        # Genes with both editing and pathogenic variants
        pathogenic_mask = overlap_df["clinical_significance"].str.contains(
            "Pathogenic|Likely_pathogenic", case=False, na=False
        )
        pathogenic_editing = overlap_df[pathogenic_mask]
        if len(pathogenic_editing) > 0:
            logger.info("\nKnown editing sites with pathogenic ClinVar classification:")
            for _, row in pathogenic_editing.iterrows():
                logger.info("  %s %s:%d - %s (%s)",
                            row["gene"], row["chr"], row["start"],
                            row["clinical_significance"], row["condition"])

    # ClinVar pathogenicity in editing genes vs non-editing genes
    clinvar_in_editing_genes = clinvar_df[clinvar_df["gene"].isin(editing_genes)]
    clinvar_not_in_editing = clinvar_df[~clinvar_df["gene"].isin(editing_genes)]

    def pathogenic_fraction(df):
        if len(df) == 0:
            return 0.0
        mask = df["clinical_significance"].str.contains(
            "Pathogenic|Likely_pathogenic", case=False, na=False
        )
        return mask.sum() / len(df)

    frac_in = pathogenic_fraction(clinvar_in_editing_genes)
    frac_out = pathogenic_fraction(clinvar_not_in_editing)
    logger.info("\nPathogenic fraction in editing genes: %.4f (%d variants)",
                frac_in, len(clinvar_in_editing_genes))
    logger.info("Pathogenic fraction in non-editing genes: %.4f (%d variants)",
                frac_out, len(clinvar_not_in_editing))

    # Visualization: pathogenicity distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: ClinVar significance of overlapping sites
    if len(overlap_df) > 0:
        sig_data = overlap_df["clinical_significance"].value_counts().head(8)
        colors = plt.cm.Set3(np.linspace(0, 1, len(sig_data)))
        axes[0].barh(range(len(sig_data)), sig_data.values, color=colors)
        axes[0].set_yticks(range(len(sig_data)))
        axes[0].set_yticklabels([s.replace("_", " ") for s in sig_data.index], fontsize=9)
        axes[0].set_xlabel("Count")
        axes[0].set_title(f"ClinVar Classification of\nKnown Editing Sites (n={len(overlap_df)})")

    # Panel 2: Pathogenic fraction comparison
    categories = ["Editing Genes", "Non-Editing Genes"]
    fractions = [frac_in, frac_out]
    counts = [len(clinvar_in_editing_genes), len(clinvar_not_in_editing)]
    bars = axes[1].bar(categories, fractions, color=["#e74c3c", "#95a5a6"], edgecolor="black")
    for bar, f, n in zip(bars, fractions, counts):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                     f"{f:.3f}\n(n={n:,})", ha="center", va="bottom", fontsize=9)
    axes[1].set_ylabel("Fraction Pathogenic/Likely Pathogenic")
    axes[1].set_title("Pathogenic Variant Fraction:\nEditing vs Non-Editing Genes")

    plt.tight_layout()
    plt.savefig(output_dir / "clinvar_overlap_analysis.png")
    plt.close()

    results = {
        "n_editing_sites": len(editing_coords),
        "n_clinvar_c2t": len(clinvar_df),
        "n_direct_overlap": len(overlap_coords),
        "n_editing_genes": len(editing_genes),
        "n_clinvar_genes": len(clinvar_genes),
        "n_gene_overlap": len(gene_overlap),
        "pathogenic_fraction_editing_genes": frac_in,
        "pathogenic_fraction_non_editing_genes": frac_out,
        "overlap_pathogenicity": sig_counts.to_dict() if len(overlap_df) > 0 else {},
    }

    if len(pathogenic_editing) > 0:
        results["pathogenic_editing_sites"] = [
            {"gene": row["gene"], "chr": row["chr"], "pos": row["start"],
             "significance": row["clinical_significance"], "condition": row["condition"]}
            for _, row in pathogenic_editing.iterrows()
        ]

    return results


# =====================================================================
# 2. Gene Ontology / KEGG / Disease Enrichment via Enrichr
# =====================================================================

def run_enrichr_analysis(gene_list, gene_set_label, output_dir, libraries=None):
    """Run Enrichr enrichment analysis for a gene list.

    Uses gseapy.enrichr() which calls the Enrichr REST API.
    """
    import gseapy as gp

    if libraries is None:
        libraries = [
            "GO_Biological_Process_2023",
            "GO_Molecular_Function_2023",
            "GO_Cellular_Component_2023",
            "KEGG_2021_Human",
            "DisGeNET",
            "Jensen_DISEASES",
            "OMIM_Disease",
        ]

    logger.info("\nRunning Enrichr for '%s' (%d genes)...", gene_set_label, len(gene_list))
    logger.info("  Libraries: %s", ", ".join(libraries))

    all_results = {}

    for lib in libraries:
        try:
            enr = gp.enrichr(
                gene_list=list(gene_list),
                gene_sets=lib,
                organism="human",
                outdir=None,  # Don't save individual files
                no_plot=True,
                verbose=False,
            )

            if enr.results is not None and len(enr.results) > 0:
                df = enr.results.copy()
                # Filter significant results
                sig = df[df["Adjusted P-value"] < 0.05]
                all_results[lib] = {
                    "n_total": len(df),
                    "n_significant": len(sig),
                    "top_terms": [],
                }

                for _, row in sig.head(20).iterrows():
                    all_results[lib]["top_terms"].append({
                        "term": row["Term"],
                        "p_value": float(row["P-value"]),
                        "adj_p_value": float(row["Adjusted P-value"]),
                        "odds_ratio": float(row["Odds Ratio"]) if "Odds Ratio" in row else None,
                        "combined_score": float(row["Combined Score"]),
                        "genes": row["Genes"],
                    })

                logger.info("  %s: %d significant / %d total terms",
                            lib, len(sig), len(df))

                if len(sig) > 0:
                    logger.info("    Top 3:")
                    for _, row in sig.head(3).iterrows():
                        logger.info("      %s (p=%.2e, adj_p=%.2e)",
                                    row["Term"], row["P-value"], row["Adjusted P-value"])
            else:
                all_results[lib] = {"n_total": 0, "n_significant": 0, "top_terms": []}
                logger.info("  %s: no results", lib)

        except Exception as e:
            logger.warning("  %s: FAILED - %s", lib, str(e))
            all_results[lib] = {"error": str(e)}

    return all_results


def plot_enrichment_dotplot(enrichment_results, gene_set_label, output_dir, max_terms=15):
    """Create dot plot visualization for enrichment results."""
    # Collect all significant terms across libraries
    all_terms = []
    for lib, res in enrichment_results.items():
        if isinstance(res, dict) and "top_terms" in res:
            for term in res["top_terms"][:max_terms]:
                all_terms.append({
                    "Library": lib.replace("_", " "),
                    "Term": term["term"][:60],
                    "neg_log10_p": -np.log10(max(term["adj_p_value"], 1e-300)),
                    "combined_score": term.get("combined_score", 0),
                })

    if not all_terms:
        logger.info("No significant terms to plot for '%s'", gene_set_label)
        return

    df = pd.DataFrame(all_terms)
    # Take top terms by significance
    df = df.nlargest(min(max_terms, len(df)), "neg_log10_p")

    fig, ax = plt.subplots(figsize=(12, max(4, len(df) * 0.4)))
    scatter = ax.scatter(
        df["neg_log10_p"], range(len(df)),
        s=df["combined_score"].clip(0, 500) * 0.5 + 20,
        c=df["neg_log10_p"],
        cmap="YlOrRd",
        alpha=0.8,
        edgecolors="black",
        linewidth=0.5,
    )

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["Term"].values, fontsize=8)
    ax.set_xlabel("-log10(Adjusted P-value)", fontsize=11)
    ax.set_title(f"Enrichment Analysis: {gene_set_label}", fontsize=13, fontweight="bold")
    ax.axvline(x=-np.log10(0.05), color="gray", linestyle="--", alpha=0.5, label="p=0.05")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis="x")

    plt.colorbar(scatter, ax=ax, label="-log10(adj. p)", shrink=0.6)
    plt.tight_layout()
    plt.savefig(output_dir / f"enrichment_{gene_set_label.lower().replace(' ', '_')}.png")
    plt.close()


# =====================================================================
# 3. Constitutive vs Facultative Gene Enrichment
# =====================================================================

def constitutive_vs_facultative_enrichment(combined_df, levanon_path, output_dir):
    """Compare GO enrichment between constitutive and facultative editing genes."""
    logger.info("\n=== Constitutive vs Facultative Gene Enrichment ===")

    if not levanon_path.exists():
        logger.warning("Levanon tissue data not found: %s", levanon_path)
        return {}

    levanon_df = pd.read_csv(levanon_path)

    # Detect gene column name (varies between files)
    gene_col = None
    for candidate in ["gene", "Gene_(RefSeq)", "Gene", "gene_name"]:
        if candidate in levanon_df.columns:
            gene_col = candidate
            break
    if gene_col is None:
        logger.warning("Could not find gene column in Levanon data. Columns: %s",
                        list(levanon_df.columns[:10]))
        return {}

    # Known GTEx tissue columns
    TISSUE_NAMES = [
        "Adipose_Subcutaneous", "Adipose_Visceral_Omentum", "Adrenal_Gland",
        "Artery_Aorta", "Artery_Coronary", "Artery_Tibial", "Bladder",
        "Brain_Amygdala", "Brain_Anterior_cingulate_cortex_BA24",
        "Brain_Caudate_basal_ganglia", "Brain_Cerebellar_Hemisphere",
        "Brain_Cerebellum", "Brain_Cortex", "Brain_Frontal_Cortex_BA9",
        "Brain_Hippocampus", "Brain_Hypothalamus",
        "Brain_Nucleus_accumbens_basal_ganglia", "Brain_Putamen_basal_ganglia",
        "Brain_Spinal_cord_cervical_c-1", "Brain_Substantia_nigra",
        "Breast_Mammary_Tissue", "Cells_Cultured_fibroblasts",
        "Cells_EBV-transformed_lymphocytes", "Cervix_Ectocervix",
        "Cervix_Endocervix", "Colon_Sigmoid", "Colon_Transverse",
        "Esophagus_Gastroesophageal_Junction", "Esophagus_Mucosa",
        "Esophagus_Muscularis", "Fallopian_Tube", "Heart_Atrial_Appendage",
        "Heart_Left_Ventricle", "Kidney_Cortex", "Kidney_Medulla", "Liver",
        "Lung", "Minor_Salivary_Gland", "Muscle_Skeletal", "Nerve_Tibial",
        "Ovary", "Pancreas", "Pituitary", "Prostate",
        "Skin_Not_Sun_Exposed_Suprapubic", "Skin_Sun_Exposed_Lower_leg",
        "Small_Intestine_Terminal_Ileum", "Spleen", "Stomach", "Testis",
        "Thyroid", "Uterus", "Vagina", "Whole_Blood",
    ]
    tissue_cols = [c for c in TISSUE_NAMES if c in levanon_df.columns]

    if not tissue_cols:
        logger.warning("Could not identify tissue columns in Levanon data")
        return {}

    logger.info("Found %d tissue columns in Levanon data", len(tissue_cols))

    # Parse tissue values: format is "mismatch;coverage;rate" — extract rate
    for tc in tissue_cols:
        levanon_df[tc] = levanon_df[tc].apply(
            lambda v: float(str(v).split(";")[2]) if pd.notna(v) and ";" in str(v) else (
                float(v) if pd.notna(v) and str(v).replace(".", "", 1).replace("-", "", 1).isdigit() else 0.0
            )
        )

    # Count tissues with editing per site
    tissue_breadth = (levanon_df[tissue_cols] > 0).sum(axis=1)
    levanon_df["tissue_breadth"] = tissue_breadth

    # Classify: constitutive (>40 tissues), intermediate (5-40), facultative (<5)
    constitutive_genes = set(levanon_df[tissue_breadth >= 40][gene_col].dropna().str.strip())
    facultative_genes = set(levanon_df[tissue_breadth <= 5][gene_col].dropna().str.strip())
    constitutive_genes.discard("")
    facultative_genes.discard("")

    logger.info("Constitutive genes (≥40 tissues): %d", len(constitutive_genes))
    logger.info("Facultative genes (≤5 tissues): %d", len(facultative_genes))

    results = {
        "n_constitutive_genes": len(constitutive_genes),
        "n_facultative_genes": len(facultative_genes),
        "constitutive_genes": sorted(list(constitutive_genes)),
        "facultative_genes_sample": sorted(list(facultative_genes))[:50],
    }

    # Run enrichment for each group
    if len(constitutive_genes) >= 5:
        const_enr = run_enrichr_analysis(
            list(constitutive_genes), "Constitutive Editing Genes", output_dir,
            libraries=["GO_Biological_Process_2023", "KEGG_2021_Human", "DisGeNET"]
        )
        results["constitutive_enrichment"] = const_enr
        plot_enrichment_dotplot(const_enr, "Constitutive Editing Genes", output_dir)

    if len(facultative_genes) >= 5:
        fac_enr = run_enrichr_analysis(
            list(facultative_genes), "Facultative Editing Genes", output_dir,
            libraries=["GO_Biological_Process_2023", "KEGG_2021_Human", "DisGeNET"]
        )
        results["facultative_enrichment"] = fac_enr
        plot_enrichment_dotplot(fac_enr, "Facultative Editing Genes", output_dir)

    return results


# =====================================================================
# 4. Per-Dataset Gene Enrichment
# =====================================================================

def per_dataset_enrichment(combined_df, output_dir):
    """Run enrichment separately for genes from each dataset."""
    logger.info("\n=== Per-Dataset Gene Enrichment ===")

    results = {}
    for ds_name, ds_label in DATASET_LABELS.items():
        ds_genes = set(
            combined_df[combined_df["dataset_source"] == ds_name]["gene"]
            .dropna().str.strip()
        )
        ds_genes.discard("")

        if len(ds_genes) < 5:
            logger.info("  %s: only %d genes, skipping", ds_label, len(ds_genes))
            continue

        logger.info("  %s: %d genes", ds_label, len(ds_genes))
        ds_enr = run_enrichr_analysis(
            list(ds_genes), f"{ds_label} Edited Genes", output_dir,
            libraries=["GO_Biological_Process_2023", "KEGG_2021_Human", "DisGeNET"]
        )
        results[ds_label] = ds_enr

    return results


# =====================================================================
# 5. High-Rate vs Low-Rate Gene Enrichment
# =====================================================================

def rate_stratified_enrichment(combined_df, output_dir):
    """Compare enrichment between high-rate and low-rate editing genes."""
    logger.info("\n=== Rate-Stratified Gene Enrichment ===")

    rates = combined_df[combined_df["editing_rate"].notna()].copy()
    if len(rates) < 20:
        logger.warning("Not enough sites with editing rates")
        return {}

    rate_vals = rates["editing_rate"].values.copy()
    # Normalize to [0,1] if needed
    if (rate_vals > 1).any():
        rate_vals = np.where(rate_vals > 1, rate_vals / 100, rate_vals)

    median_rate = np.median(rate_vals)
    rates["norm_rate"] = rate_vals

    high_rate_genes = set(rates[rates["norm_rate"] >= median_rate]["gene"].dropna().str.strip())
    low_rate_genes = set(rates[rates["norm_rate"] < median_rate]["gene"].dropna().str.strip())
    high_rate_genes.discard("")
    low_rate_genes.discard("")

    logger.info("High-rate genes (≥ median %.4f): %d", median_rate, len(high_rate_genes))
    logger.info("Low-rate genes (< median): %d", len(low_rate_genes))

    results = {"median_rate": float(median_rate)}

    if len(high_rate_genes) >= 5:
        high_enr = run_enrichr_analysis(
            list(high_rate_genes), "High-Rate Editing Genes", output_dir,
            libraries=["GO_Biological_Process_2023", "KEGG_2021_Human", "DisGeNET"]
        )
        results["high_rate_enrichment"] = high_enr
        plot_enrichment_dotplot(high_enr, "High-Rate Editing Genes", output_dir)

    if len(low_rate_genes) >= 5:
        low_enr = run_enrichr_analysis(
            list(low_rate_genes), "Low-Rate Editing Genes", output_dir,
            libraries=["GO_Biological_Process_2023", "KEGG_2021_Human", "DisGeNET"]
        )
        results["low_rate_enrichment"] = low_enr
        plot_enrichment_dotplot(low_enr, "Low-Rate Editing Genes", output_dir)

    return results


# =====================================================================
# 6. Model-Predicted Gene Enrichment (Calibrated Threshold)
# =====================================================================

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


def model_predicted_gene_enrichment(combined_df, output_dir):
    """Compare disease enrichment: known editing genes vs model-predicted editing genes.

    Uses calibrated threshold (P_cal >= 0.5) to define model-predicted editing sites,
    then runs Enrichr on genes containing predicted sites.
    """
    logger.info("\n=== Model-Predicted Gene Enrichment (Calibrated) ===")

    # Load calibrated scores (generated by exp_clinvar_calibrated.py)
    if CALIBRATED_SCORES.exists():
        cal_df = pd.read_csv(CALIBRATED_SCORES)
        logger.info("Loaded calibrated scores: %d variants", len(cal_df))
    else:
        logger.warning("Calibrated scores not found: %s", CALIBRATED_SCORES)
        logger.info("Falling back to raw scores with Bayesian recalibration")

        scores_path = PROJECT_ROOT / "experiments/apobec3a/outputs/clinvar_prediction/clinvar_all_scores.csv"
        if not scores_path.exists():
            logger.warning("ClinVar scores not found: %s", scores_path)
            return {}
        cal_df = pd.read_csv(scores_path)

        # Compute calibrated scores on the fly
        splits = pd.read_csv(SPLITS_A3A)
        n_pos = (splits["is_edited"] == 1).sum()
        neg_t1 = pd.read_csv(NEG_TIER1)
        n_neg = len(neg_t1)
        pi_real = n_pos / (n_pos + n_neg)
        pi_model = 0.50  # balanced via scale_pos_weight

        cal_df["p_cal_gb_tier1"] = _bayesian_recalibrate(
            cal_df["p_edited_gb"].values, pi_model, pi_real)
        cal_df["p_cal_rf_tier1"] = _bayesian_recalibrate(
            cal_df["p_edited_rnasee"].values, pi_model, pi_real)

    # Known editing genes (from combined dataset)
    known_genes = set(combined_df["gene"].dropna().str.strip())
    known_genes.discard("")

    # Model-predicted genes at calibrated threshold (P_cal >= 0.5)
    gb_predicted = cal_df[cal_df["p_cal_gb_tier1"] >= 0.5]
    gb_pred_genes = set(gb_predicted["gene"].dropna().str.strip())
    gb_pred_genes.discard("")

    # Also at lower threshold (P_cal >= 0.1) for sensitivity
    gb_sens = cal_df[cal_df["p_cal_gb_tier1"] >= 0.1]
    gb_sens_genes = set(gb_sens["gene"].dropna().str.strip())
    gb_sens_genes.discard("")

    # Overlap statistics
    overlap_strict = known_genes & gb_pred_genes
    novel_strict = gb_pred_genes - known_genes

    logger.info("Known editing genes: %d", len(known_genes))
    logger.info("GB-predicted genes (P_cal>=0.5): %d", len(gb_pred_genes))
    logger.info("  Overlap with known: %d", len(overlap_strict))
    logger.info("  Novel (not in known set): %d", len(novel_strict))
    logger.info("GB-predicted genes (P_cal>=0.1): %d", len(gb_sens_genes))

    results = {
        "n_known_genes": len(known_genes),
        "n_predicted_genes_pcal05": len(gb_pred_genes),
        "n_overlap": len(overlap_strict),
        "n_novel": len(novel_strict),
        "n_predicted_genes_pcal01": len(gb_sens_genes),
        "n_predicted_sites_pcal05": len(gb_predicted),
        "novel_genes_sample": sorted(list(novel_strict))[:50],
    }

    # Run enrichment on predicted genes
    if len(gb_pred_genes) >= 5:
        logger.info("\nRunning Enrichr on GB-predicted genes (P_cal>=0.5)...")
        pred_enr = run_enrichr_analysis(
            list(gb_pred_genes), "GB-Predicted Editing Genes (P_cal>=0.5)", output_dir,
            libraries=["GO_Biological_Process_2023", "KEGG_2021_Human", "DisGeNET"]
        )
        results["predicted_enrichment_pcal05"] = pred_enr
        plot_enrichment_dotplot(pred_enr, "GB-Predicted Editing Genes (Pcal05)", output_dir)

    # Run enrichment on novel genes only
    if len(novel_strict) >= 5:
        logger.info("\nRunning Enrichr on novel predicted genes (not in known set)...")
        novel_enr = run_enrichr_analysis(
            list(novel_strict), "Novel GB-Predicted Genes", output_dir,
            libraries=["GO_Biological_Process_2023", "KEGG_2021_Human", "DisGeNET"]
        )
        results["novel_genes_enrichment"] = novel_enr
        plot_enrichment_dotplot(novel_enr, "Novel GB-Predicted Genes", output_dir)

    # Visualization: Venn-style comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: Gene set sizes
    categories = ["Known\nEditing", "GB-Predicted\n(P_cal≥0.5)", "Overlap", "Novel\nPredicted"]
    counts = [len(known_genes), len(gb_pred_genes), len(overlap_strict), len(novel_strict)]
    colors_bar = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0"]
    bars = axes[0].bar(range(len(categories)), counts, color=colors_bar, edgecolor="white")
    for bar, c in zip(bars, counts):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                     f"{c:,}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    axes[0].set_xticks(range(len(categories)))
    axes[0].set_xticklabels(categories, fontsize=10)
    axes[0].set_ylabel("Number of Genes")
    axes[0].set_title("Known vs Model-Predicted Editing Genes")

    # Panel 2: ClinVar pathogenicity in predicted vs non-predicted genes
    if "gene" in cal_df.columns:
        pred_gene_clinvar = cal_df[cal_df["gene"].isin(gb_pred_genes)]
        rest_clinvar = cal_df[~cal_df["gene"].isin(gb_pred_genes)]

        def path_frac(df):
            if len(df) == 0 or "significance_simple" not in df.columns:
                return 0.0
            return df["significance_simple"].isin(
                ["Pathogenic", "Likely_pathogenic"]).sum() / len(df)

        frac_pred = path_frac(pred_gene_clinvar)
        frac_rest = path_frac(rest_clinvar)

        bars2 = axes[1].bar(["Predicted\nEditing Genes", "Other\nGenes"],
                            [frac_pred, frac_rest],
                            color=["#FF9800", "#9E9E9E"], edgecolor="black")
        for bar, f, n in zip(bars2, [frac_pred, frac_rest],
                              [len(pred_gene_clinvar), len(rest_clinvar)]):
            axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                         f"{f:.3f}\n(n={n:,})", ha="center", va="bottom", fontsize=9)
        axes[1].set_ylabel("Fraction Pathogenic/Likely Pathogenic")
        axes[1].set_title("ClinVar Pathogenicity:\nPredicted-Editing vs Other Genes")

    plt.tight_layout()
    plt.savefig(output_dir / "model_predicted_gene_enrichment.png")
    plt.close()

    return results


# =====================================================================
# Main
# =====================================================================

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # Load data
    logger.info("Loading datasets...")
    combined_df = pd.read_csv(COMBINED_PATH)
    logger.info("Combined dataset: %d sites", len(combined_df))

    # Load ClinVar
    clinvar_df = None
    if CLINVAR_PATH.exists():
        clinvar_df = pd.read_csv(CLINVAR_PATH)
        logger.info("ClinVar C>T variants: %d", len(clinvar_df))
    else:
        logger.warning("ClinVar data not found: %s", CLINVAR_PATH)

    # All editing genes
    all_editing_genes = set(combined_df["gene"].dropna().str.strip())
    all_editing_genes.discard("")
    logger.info("Total editing genes: %d", len(all_editing_genes))

    results = {
        "description": "Disease enrichment analysis for APOBEC3A editing sites",
        "n_editing_sites": len(combined_df),
        "n_editing_genes": len(all_editing_genes),
    }

    # 1. ClinVar overlap
    if clinvar_df is not None:
        results["clinvar_overlap"] = clinvar_overlap_analysis(
            combined_df, clinvar_df, OUTPUT_DIR
        )

    # 2. All editing genes enrichment
    logger.info("\n=== All Editing Genes Enrichment ===")
    all_enr = run_enrichr_analysis(
        list(all_editing_genes), "All APOBEC3A Editing Genes", OUTPUT_DIR
    )
    results["all_genes_enrichment"] = all_enr
    plot_enrichment_dotplot(all_enr, "All APOBEC3A Editing Genes", OUTPUT_DIR)

    # 3. Constitutive vs facultative
    results["constitutive_facultative"] = constitutive_vs_facultative_enrichment(
        combined_df, LEVANON_PATH, OUTPUT_DIR
    )

    # 4. Per-dataset enrichment
    results["per_dataset"] = per_dataset_enrichment(combined_df, OUTPUT_DIR)

    # 5. Rate-stratified enrichment
    results["rate_stratified"] = rate_stratified_enrichment(combined_df, OUTPUT_DIR)

    # 6. Model-predicted gene enrichment (calibrated threshold)
    results["model_predicted"] = model_predicted_gene_enrichment(combined_df, OUTPUT_DIR)

    # Summary
    elapsed = time.time() - t0
    results["total_time_seconds"] = elapsed

    logger.info("\n" + "=" * 70)
    logger.info("DISEASE ENRICHMENT ANALYSIS COMPLETE")
    logger.info("=" * 70)
    logger.info("Total time: %.1fs", elapsed)

    # Save results
    def serialize(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, set):
            return sorted(list(obj))
        return str(obj)

    with open(OUTPUT_DIR / "disease_enrichment_results.json", "w") as f:
        json.dump(results, f, indent=2, default=serialize)
    logger.info("Saved results to %s", OUTPUT_DIR / "disease_enrichment_results.json")


if __name__ == "__main__":
    main()
