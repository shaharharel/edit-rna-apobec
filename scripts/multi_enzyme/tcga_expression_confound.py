#!/usr/bin/env python3
"""C1: Gene expression confound control for TCGA editability enrichment.

Tests whether the TCGA somatic mutation enrichment at predicted editing sites
is confounded by gene expression levels. Highly expressed genes have more
sequencing coverage and thus more detected mutations. If enrichment persists
within expression-matched strata, it is not an expression artifact.

Approach:
  1. Download RSEM gene expression from cBioPortal for BLCA, BRCA, CESC, LUSC
  2. Compute median expression per gene across tumor samples
  3. Parse MAF files to get gene-level mutation counts (C>T/G>A only)
  4. For each cancer, split genes into expression quartiles
  5. Within each quartile, compare editability scores of mutations vs controls
  6. Key test: does enrichment persist within expression-matched strata?

Requires completed run of tcga_full_model_enrichment.py.

Usage:
    conda run -n quris python scripts/multi_enzyme/tcga_expression_confound.py
"""

import json
import logging
import os
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyfaidx import Fasta
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = PROJECT_ROOT / "data"
HG19_FA = DATA_DIR / "raw/genomes/hg19.fa"
REFGENE_HG19 = DATA_DIR / "raw/genomes/refGene_hg19.txt"
TCGA_CACHE = DATA_DIR / "raw/tcga"
SCORES_DIR = PROJECT_ROOT / "experiments/multi_enzyme/outputs/tcga_gnomad/raw_scores"
OUTPUT_DIR = PROJECT_ROOT / "experiments/multi_enzyme/outputs/tcga_gnomad/expression_confound"

SEED = 42
N_CONTROLS = 5

CANCER_STUDIES = {
    "blca": "blca_tcga_pan_can_atlas_2018",
    "brca": "brca_tcga_pan_can_atlas_2018",
    "cesc": "cesc_tcga_pan_can_atlas_2018",
    "lusc": "lusc_tcga_pan_can_atlas_2018",
}

CBIO_BASE = "https://media.githubusercontent.com/media/cBioPortal/datahub/master/public"


def download_expression(cancer):
    """Download RSEM gene expression data from cBioPortal. Returns local path."""
    TCGA_CACHE.mkdir(parents=True, exist_ok=True)
    study = CANCER_STUDIES[cancer]
    local = TCGA_CACHE / f"{study}_mrna_rsem.txt"
    if local.exists():
        logger.info(f"  Cached expression: {local} ({local.stat().st_size / 1e6:.1f} MB)")
        return local

    url = f"{CBIO_BASE}/{study}/data_mrna_seq_v2_rsem.txt"
    logger.info(f"  Downloading expression data: {url}")
    try:
        urllib.request.urlretrieve(url, local)
        logger.info(f"  Downloaded: {local.stat().st_size / 1e6:.1f} MB")
        return local
    except Exception as e:
        logger.error(f"  Download failed: {e}")
        return None


def load_expression(expr_path):
    """Load RSEM expression and compute median per gene.

    Returns dict: gene_name -> median_expression (across samples).
    """
    if expr_path is None or not expr_path.exists():
        return {}

    df = pd.read_csv(expr_path, sep="\t", comment="#", low_memory=False)

    # Typical columns: Hugo_Symbol, Entrez_Gene_Id, then sample columns
    gene_col = None
    for col in ["Hugo_Symbol", "HUGO_SYMBOL", "Gene Symbol"]:
        if col in df.columns:
            gene_col = col
            break

    if gene_col is None:
        # First column is often the gene name
        gene_col = df.columns[0]

    # Sample columns are everything after the first 1-2 identifier columns
    id_cols = [c for c in df.columns if c in [gene_col, "Entrez_Gene_Id", "ENTREZ_GENE_ID"]]
    sample_cols = [c for c in df.columns if c not in id_cols]

    if len(sample_cols) == 0:
        logger.warning("No sample columns found in expression data")
        return {}

    logger.info(f"  Expression: {len(df)} genes, {len(sample_cols)} samples")

    # Compute median expression per gene
    gene_expr = {}
    for _, row in df.iterrows():
        gene = str(row[gene_col])
        if gene in ["nan", "None", ""]:
            continue
        vals = pd.to_numeric(row[sample_cols], errors="coerce").dropna()
        if len(vals) > 0:
            gene_expr[gene] = float(vals.median())

    logger.info(f"  Computed median expression for {len(gene_expr):,} genes")
    return gene_expr


def parse_ct_mutations_with_gene(maf_path):
    """Parse C>T and G>A SNPs from MAF, keeping gene annotation.

    Returns DataFrame with chrom, pos, strand_inf, gene, sample.
    """
    rows = []
    for chunk in pd.read_csv(maf_path, sep="\t", comment="#", low_memory=False, chunksize=100000):
        if "Variant_Type" in chunk.columns:
            chunk = chunk[chunk["Variant_Type"] == "SNP"]

        ct = (chunk["Reference_Allele"] == "C") & (chunk["Tumor_Seq_Allele2"] == "T")
        ga = (chunk["Reference_Allele"] == "G") & (chunk["Tumor_Seq_Allele2"] == "A")
        sub = chunk[ct | ga].copy()

        if len(sub) > 0:
            chrom = sub["Chromosome"].astype(str)
            if not chrom.str.startswith("chr").any():
                chrom = "chr" + chrom
            sub["chrom"] = chrom
            sub["pos"] = sub["Start_Position"].astype(int) - 1
            sub["strand_inf"] = np.where(sub["Reference_Allele"] == "C", "+", "-")
            sub["gene"] = sub.get("Hugo_Symbol", "unknown")
            sub["sample"] = sub.get("Tumor_Sample_Barcode", "unknown")
            rows.append(sub[["chrom", "pos", "strand_inf", "gene", "sample"]].copy())

    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    df_dedup = df.drop_duplicates(subset=["chrom", "pos"]).copy()
    rec = df.groupby(["chrom", "pos"]).size().reset_index(name="recurrence")
    df_dedup = df_dedup.merge(rec, on=["chrom", "pos"], how="left")
    return df_dedup


def get_matched_controls(mutations_df, genome, exons_by_gene, n_controls=N_CONTROLS):
    """Generate matched controls — same as original pipeline."""
    rng = np.random.RandomState(SEED)
    controls = []

    for _, row in mutations_df.iterrows():
        gene = row["gene"]
        gene_exons = exons_by_gene.get(gene, [])
        if not gene_exons:
            continue

        c_positions = []
        for chrom, start, end, strand, _ in gene_exons:
            if chrom != row["chrom"]:
                continue
            try:
                exon_seq = str(genome[chrom][start:end]).upper()
                for i, base in enumerate(exon_seq):
                    pos = start + i
                    if pos == row["pos"]:
                        continue
                    if base == "C":
                        c_positions.append((chrom, pos, "+", gene))
                    elif base == "G":
                        c_positions.append((chrom, pos, "-", gene))
            except (KeyError, ValueError):
                continue

        if len(c_positions) >= n_controls:
            chosen = rng.choice(len(c_positions), n_controls, replace=False)
            for idx in chosen:
                ch, p, s, g = c_positions[idx]
                controls.append({"chrom": ch, "pos": p, "strand_inf": s, "gene": g})
        elif c_positions:
            for ch, p, s, g in c_positions[:n_controls]:
                controls.append({"chrom": ch, "pos": p, "strand_inf": s, "gene": g})

    return pd.DataFrame(controls) if controls else pd.DataFrame()


def extract_seq_valid(chrom, pos, strand, genome):
    """Check if a position yields a valid 201-nt sequence with C at center."""
    try:
        chrom_len = len(genome[chrom])
        start = pos - 100
        end = pos + 101
        if start < 0 or end > chrom_len:
            return False
        seq = str(genome[chrom][start:end]).upper()
        if strand == "-":
            comp = str.maketrans("ACGT", "TGCA")
            seq = seq.translate(comp)[::-1]
        return len(seq) == 201 and seq[100] == "C"
    except (KeyError, ValueError):
        return False


def compute_enrichment_percentile(mut_scores, ctrl_scores,
                                   percentiles=(50, 60, 70, 80, 90, 95)):
    """Compute enrichment ORs at percentile-based thresholds."""
    results = {}
    for pct in percentiles:
        if len(ctrl_scores) == 0:
            results[f"p{pct}"] = {"OR": float("nan"), "p": 1.0}
            continue
        threshold = float(np.percentile(ctrl_scores, pct))
        mut_above = int((mut_scores >= threshold).sum())
        mut_below = int((mut_scores < threshold).sum())
        ctrl_above = int((ctrl_scores >= threshold).sum())
        ctrl_below = int((ctrl_scores < threshold).sum())

        if mut_below > 0 and ctrl_above > 0 and ctrl_below > 0:
            table = [[mut_above, mut_below], [ctrl_above, ctrl_below]]
            or_val, p_val = stats.fisher_exact(table)
        else:
            or_val, p_val = float("nan"), 1.0

        results[f"p{pct}"] = {
            "percentile": pct,
            "threshold": round(threshold, 4),
            "OR": round(float(or_val), 4),
            "p": float(p_val),
            "n_mut_above": mut_above,
            "n_mut_total": len(mut_scores),
            "n_ctrl_above": ctrl_above,
            "n_ctrl_total": len(ctrl_scores),
        }
    return results


def main():
    t_start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load genome
    logger.info("Loading hg19 genome...")
    genome = Fasta(str(HG19_FA))

    # Parse exons
    logger.info("Parsing exonic regions...")
    exons_by_gene = defaultdict(list)
    with open(REFGENE_HG19) as f:
        seen = set()
        for line in f:
            fields = line.strip().split("\t")
            if len(fields) < 13:
                continue
            chrom, strand = fields[2], fields[3]
            cds_s, cds_e = int(fields[6]), int(fields[7])
            if cds_s == cds_e:
                continue
            gene = fields[12]
            for es_str, ee_str in zip(fields[9].rstrip(",").split(","),
                                       fields[10].rstrip(",").split(",")):
                if not es_str or not ee_str:
                    continue
                es, ee = max(int(es_str), cds_s), min(int(ee_str), cds_e)
                if es >= ee:
                    continue
                key = (chrom, es, ee)
                if key not in seen:
                    seen.add(key)
                    exons_by_gene[gene].append((chrom, es, ee, strand, gene))
    logger.info(f"  {len(exons_by_gene):,} genes with exonic regions")

    all_results = {}

    for cancer in ["blca", "brca", "cesc", "lusc"]:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {cancer.upper()}")

        # Step 1: Download and load expression data
        expr_path = download_expression(cancer)
        gene_expr = load_expression(expr_path)

        if len(gene_expr) == 0:
            logger.warning(f"  No expression data for {cancer}, skipping")
            continue

        # Step 2: Load raw scores
        scores_path = SCORES_DIR / f"{cancer}_scores.csv"
        if not scores_path.exists():
            logger.warning(f"  No scores file for {cancer}")
            continue
        scores_df = pd.read_csv(scores_path)

        # Step 3: Parse MAF to get gene annotation per mutation
        study = CANCER_STUDIES[cancer]
        maf_path = TCGA_CACHE / f"{study}_mutations.txt"
        if not maf_path.exists():
            logger.warning(f"  No MAF file for {cancer}")
            continue

        mut_df = parse_ct_mutations_with_gene(maf_path)
        if len(mut_df) == 0:
            continue
        logger.info(f"  {len(mut_df):,} unique C>T/G>A mutations")

        # Step 4: Generate matched controls (same seed as original)
        ctrl_df = get_matched_controls(mut_df, genome, exons_by_gene, N_CONTROLS)
        logger.info(f"  {len(ctrl_df):,} control positions")

        # Step 5: Filter valid sequences (same as original)
        mut_valid = []
        for _, row in mut_df.iterrows():
            valid = extract_seq_valid(row["chrom"], int(row["pos"]),
                                      row["strand_inf"], genome)
            mut_valid.append(valid)
        mut_valid = np.array(mut_valid, dtype=bool)

        ctrl_valid = []
        for _, row in ctrl_df.iterrows():
            valid = extract_seq_valid(row["chrom"], int(row["pos"]),
                                      row["strand_inf"], genome)
            ctrl_valid.append(valid)
        ctrl_valid = np.array(ctrl_valid, dtype=bool)

        valid_mut_df = mut_df[mut_valid].reset_index(drop=True)
        valid_ctrl_df = ctrl_df[ctrl_valid].reset_index(drop=True)

        n_mut = min(len(valid_mut_df), (scores_df["type"] == "mutation").sum())
        n_ctrl = min(len(valid_ctrl_df), (scores_df["type"] == "control").sum())

        logger.info(f"  Valid: {n_mut:,} mutations, {n_ctrl:,} controls")

        # Get gene names for each valid position
        mut_genes = valid_mut_df["gene"].values[:n_mut]
        ctrl_genes = valid_ctrl_df["gene"].values[:n_ctrl]

        # Get scores
        mut_scores = scores_df[scores_df["type"] == "mutation"]["score"].values[:n_mut]
        ctrl_scores = scores_df[scores_df["type"] == "control"]["score"].values[:n_ctrl]

        # Assign expression levels
        mut_expr = np.array([gene_expr.get(g, np.nan) for g in mut_genes])
        ctrl_expr = np.array([gene_expr.get(g, np.nan) for g in ctrl_genes])

        # Filter to positions with known expression
        mut_has_expr = ~np.isnan(mut_expr)
        ctrl_has_expr = ~np.isnan(ctrl_expr)
        logger.info(f"  With expression: {mut_has_expr.sum():,} mutations, {ctrl_has_expr.sum():,} controls")

        # Compute expression quartiles from the combined distribution
        all_expr = np.concatenate([mut_expr[mut_has_expr], ctrl_expr[ctrl_has_expr]])
        if len(all_expr) < 100:
            logger.warning(f"  Too few positions with expression data")
            continue

        quartile_edges = np.percentile(all_expr, [25, 50, 75])
        logger.info(f"  Expression quartile edges: {quartile_edges}")

        # Assign quartiles
        def get_quartile(expr_val):
            if np.isnan(expr_val):
                return -1
            if expr_val < quartile_edges[0]:
                return 0
            elif expr_val < quartile_edges[1]:
                return 1
            elif expr_val < quartile_edges[2]:
                return 2
            else:
                return 3

        mut_quartiles = np.array([get_quartile(e) for e in mut_expr])
        ctrl_quartiles = np.array([get_quartile(e) for e in ctrl_expr])

        cancer_results = {
            "cancer": cancer,
            "n_mut": int(n_mut),
            "n_ctrl": int(n_ctrl),
            "n_mut_with_expr": int(mut_has_expr.sum()),
            "n_ctrl_with_expr": int(ctrl_has_expr.sum()),
            "expression_quartile_edges": quartile_edges.tolist(),
        }

        # Overall enrichment (reference)
        cancer_results["all"] = compute_enrichment_percentile(mut_scores, ctrl_scores)
        if n_mut > 0 and n_ctrl > 0:
            _, mw_p = stats.mannwhitneyu(mut_scores, ctrl_scores, alternative="two-sided")
            cancer_results["all"]["mann_whitney_p"] = float(mw_p)
            cancer_results["all"]["mean_mut"] = round(float(np.mean(mut_scores)), 4)
            cancer_results["all"]["mean_ctrl"] = round(float(np.mean(ctrl_scores)), 4)
            cancer_results["all"]["delta"] = round(
                float(np.mean(mut_scores) - np.mean(ctrl_scores)), 4)

        # Per-quartile enrichment
        quartile_labels = ["Q1 (low)", "Q2", "Q3", "Q4 (high)"]
        for q in range(4):
            q_mut_mask = mut_quartiles == q
            q_ctrl_mask = ctrl_quartiles == q
            n_m = q_mut_mask.sum()
            n_c = q_ctrl_mask.sum()

            q_key = f"q{q+1}"
            if n_m > 50 and n_c > 50:
                logger.info(f"  Quartile {q+1} ({quartile_labels[q]}): "
                            f"{n_m:,} mutations, {n_c:,} controls")
                q_result = compute_enrichment_percentile(
                    mut_scores[q_mut_mask], ctrl_scores[q_ctrl_mask])
                _, mw_p = stats.mannwhitneyu(
                    mut_scores[q_mut_mask], ctrl_scores[q_ctrl_mask],
                    alternative="two-sided")
                q_result["mann_whitney_p"] = float(mw_p)
                q_result["mean_mut"] = round(float(np.mean(mut_scores[q_mut_mask])), 4)
                q_result["mean_ctrl"] = round(float(np.mean(ctrl_scores[q_ctrl_mask])), 4)
                q_result["delta"] = round(
                    float(np.mean(mut_scores[q_mut_mask]) - np.mean(ctrl_scores[q_ctrl_mask])), 4)
                q_result["n_mut"] = int(n_m)
                q_result["n_ctrl"] = int(n_c)
                q_result["expr_range"] = {
                    "min": round(float(np.min(all_expr[
                        (all_expr >= (quartile_edges[q-1] if q > 0 else -np.inf)) &
                        (all_expr < (quartile_edges[q] if q < 3 else np.inf))])) if n_m > 0 else 0, 1),
                    "max": round(float(np.max(all_expr[
                        (all_expr >= (quartile_edges[q-1] if q > 0 else -np.inf)) &
                        (all_expr < (quartile_edges[q] if q < 3 else np.inf))])) if n_m > 0 else 0, 1),
                }
                cancer_results[q_key] = q_result
            else:
                cancer_results[q_key] = {"note": f"Too few ({n_m} mut, {n_c} ctrl)"}

        all_results[cancer] = cancer_results

        # Log summary
        logger.info(f"\n  {cancer.upper()} SUMMARY:")
        for q in range(4):
            q_key = f"q{q+1}"
            r = cancer_results.get(q_key, {})
            if "p90" in r:
                p90 = r["p90"]
                logger.info(f"    {quartile_labels[q]:>12}: OR@p90={p90['OR']:.3f} "
                            f"(p={p90['p']:.2e}), delta={r.get('delta', 'N/A')}, "
                            f"n_mut={r.get('n_mut', 'N/A')}")

    # Save results
    with open(OUTPUT_DIR / "expression_confound_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # ---- Generate figures ----
    logger.info("\nGenerating figures...")
    cancers = [c for c in ["blca", "brca", "cesc", "lusc"] if c in all_results]

    if not cancers:
        logger.warning("No results to plot")
        return

    # Figure 1: OR at each percentile threshold, per quartile
    fig, axes = plt.subplots(1, len(cancers), figsize=(5 * len(cancers), 5), squeeze=False)

    quartile_colors = ["#60a5fa", "#3b82f6", "#1d4ed8", "#1e3a8a"]
    quartile_labels = ["Q1 (low expr)", "Q2", "Q3", "Q4 (high expr)"]

    for col, cancer in enumerate(cancers):
        ax = axes[0, col]
        r = all_results[cancer]

        # Plot overall
        if "all" in r and "p50" in r["all"]:
            pcts = [50, 60, 70, 80, 90, 95]
            ors = [r["all"].get(f"p{p}", {}).get("OR", float("nan")) for p in pcts]
            ax.plot(pcts, ors, marker="s", label="All (unstratified)", color="#374151",
                    linewidth=2, linestyle="--")

        # Plot each quartile
        for q in range(4):
            q_key = f"q{q+1}"
            if q_key not in r or "p50" not in r[q_key]:
                continue
            pcts = [50, 60, 70, 80, 90, 95]
            ors = [r[q_key].get(f"p{p}", {}).get("OR", float("nan")) for p in pcts]
            ax.plot(pcts, ors, marker="o", label=quartile_labels[q],
                    color=quartile_colors[q], linewidth=2)

        ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel("Control percentile threshold")
        ax.set_ylabel("Odds Ratio")
        ax.set_title(f"{cancer.upper()}")
        ax.legend(fontsize=7)
        ax.set_ylim(0, max(2.5, ax.get_ylim()[1]))

    plt.suptitle("Expression-Stratified TCGA Enrichment\n"
                 "Genes split into quartiles by median RSEM expression", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "expression_confound_enrichment.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Figure 2: Summary bar chart — OR at 90th percentile per quartile
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(cancers))
    width = 0.15
    all_color = "#374151"

    # All (unstratified)
    ors_all = []
    for cancer in cancers:
        r = all_results[cancer].get("all", {})
        ors_all.append(r.get("p90", {}).get("OR", float("nan")))
    ax.bar(x - 2 * width, ors_all, width, label="All", color=all_color, alpha=0.8)

    # Per quartile
    for q in range(4):
        ors_q = []
        for cancer in cancers:
            r = all_results[cancer].get(f"q{q+1}", {})
            ors_q.append(r.get("p90", {}).get("OR", float("nan")))
        ax.bar(x + (q - 1) * width, ors_q, width, label=quartile_labels[q],
               color=quartile_colors[q], alpha=0.8)

    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([c.upper() for c in cancers])
    ax.set_ylabel("Odds Ratio at 90th percentile")
    ax.set_title("Expression-Stratified Enrichment Summary")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "expression_confound_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Figure 3: Expression distribution of mutations vs controls
    fig, axes = plt.subplots(1, len(cancers), figsize=(5 * len(cancers), 4), squeeze=False)

    for col, cancer in enumerate(cancers):
        ax = axes[0, col]
        # Reload data for this cancer
        study = CANCER_STUDIES[cancer]
        maf_path = TCGA_CACHE / f"{study}_mutations.txt"
        expr_path = TCGA_CACHE / f"{study}_mrna_rsem.txt"

        gene_expr = load_expression(expr_path) if expr_path.exists() else {}
        mut_df = parse_ct_mutations_with_gene(maf_path)

        # Get expression for each mutation's gene
        mut_gene_expr = [gene_expr.get(g, np.nan) for g in mut_df["gene"]]
        mut_gene_expr = [x for x in mut_gene_expr if not np.isnan(x)]

        # Compare to all genes
        all_gene_expr = list(gene_expr.values())

        if mut_gene_expr and all_gene_expr:
            ax.hist(np.log2(np.array(all_gene_expr) + 1), bins=50, alpha=0.5,
                    density=True, label="All genes", color="#94a3b8")
            ax.hist(np.log2(np.array(mut_gene_expr) + 1), bins=50, alpha=0.5,
                    density=True, label="Mutated genes", color="#dc2626")
            ax.set_xlabel("log2(RSEM + 1)")
            ax.set_ylabel("Density")
            ax.set_title(f"{cancer.upper()}")
            ax.legend(fontsize=8)

    plt.suptitle("Gene Expression: Mutated vs All Genes", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "expression_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()

    elapsed = time.time() - t_start
    logger.info(f"\n{'='*60}")
    logger.info(f"COMPLETE ({elapsed / 60:.1f} min)")
    logger.info(f"Results: {OUTPUT_DIR / 'expression_confound_results.json'}")
    logger.info(f"Figures: {OUTPUT_DIR}")

    # Summary table
    logger.info(f"\n{'='*80}")
    logger.info("SUMMARY: OR at 90th percentile threshold by expression quartile")
    logger.info(f"{'Cancer':<8} {'Quartile':<15} {'OR':>8} {'p':>12} {'n_mut':>8} {'n_ctrl':>8}")
    logger.info("-" * 65)
    for cancer in cancers:
        for q_key, label in [("all", "All"), ("q1", "Q1 (low)"),
                              ("q2", "Q2"), ("q3", "Q3"), ("q4", "Q4 (high)")]:
            r = all_results[cancer].get(q_key, {})
            p90 = r.get("p90", {})
            if "OR" in p90:
                logger.info(f"{cancer.upper():<8} {label:<15} "
                            f"{p90['OR']:>8.3f} {p90['p']:>12.2e} "
                            f"{r.get('n_mut', p90.get('n_mut_total', 'N/A')):>8} "
                            f"{r.get('n_ctrl', p90.get('n_ctrl_total', 'N/A')):>8}")


if __name__ == "__main__":
    main()
