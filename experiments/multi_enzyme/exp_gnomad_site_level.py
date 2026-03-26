#!/usr/bin/env python3
"""B3: gnomAD Site-Level with Model Scores.

Gene-level pilot: correlate mean editability per gene from the exome map
with gnomAD LOEUF (loss-of-function observed/expected upper bound fraction).

Also attempts site-level analysis on chr22 by downloading gnomAD exome VCF
and checking C>T variant frequencies at scored positions.

Usage:
    conda run -n quris python experiments/multi_enzyme/exp_gnomad_site_level.py
"""

import gzip
import json
import logging
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = PROJECT_ROOT / "data"
GNOMAD_CONSTRAINT = DATA_DIR / "raw/gnomad/gnomad_v4.1_constraint.tsv"
EXOME_MAP = PROJECT_ROOT / "experiments/multi_enzyme/outputs/tcga_gnomad/exome_editability_map.csv.gz"
EXOME_CHR22 = PROJECT_ROOT / "experiments/multi_enzyme/outputs/tcga_gnomad/exome_editability_chr22.csv"
OUTPUT_DIR = PROJECT_ROOT / "experiments/multi_enzyme/outputs/gnomad_site_level"

# gnomAD chr22 VCF (for site-level pilot)
GNOMAD_VCF_URL = "https://storage.googleapis.com/gcp-public-data--gnomad/release/4.1/vcf/exomes/gnomad.exomes.v4.1.sites.chr22.vcf.bgz"
GNOMAD_VCF_LOCAL = DATA_DIR / "raw/gnomad/gnomad.exomes.v4.1.sites.chr22.vcf.bgz"


def load_exome_map():
    """Load exome editability map. Prefer full map, fall back to chr22."""
    if EXOME_MAP.exists():
        logger.info(f"Loading full exome map: {EXOME_MAP}")
        df = pd.read_csv(EXOME_MAP)
        logger.info(f"  {len(df):,} positions, {df['gene'].nunique():,} genes")
        return df
    elif EXOME_CHR22.exists():
        logger.info(f"Loading chr22 exome map: {EXOME_CHR22}")
        df = pd.read_csv(EXOME_CHR22)
        logger.info(f"  {len(df):,} positions, {df['gene'].nunique():,} genes")
        return df
    else:
        logger.error("No exome editability map found!")
        return None


def load_gnomad_constraint():
    """Load gnomAD constraint data (gene-level)."""
    if not GNOMAD_CONSTRAINT.exists():
        logger.error(f"gnomAD constraint file not found: {GNOMAD_CONSTRAINT}")
        return None

    df = pd.read_csv(GNOMAD_CONSTRAINT, sep="\t")
    logger.info(f"gnomAD constraint: {len(df):,} rows, {df['gene'].nunique():,} genes")

    # Keep MANE select or canonical transcripts (one per gene)
    if "mane_select" in df.columns:
        mane = df[df["mane_select"].notna()].copy()
        if len(mane) > 0:
            df = mane.drop_duplicates(subset="gene", keep="first")
        else:
            df = df.drop_duplicates(subset="gene", keep="first")
    else:
        df = df.drop_duplicates(subset="gene", keep="first")

    logger.info(f"  After dedup: {len(df):,} genes")
    return df


# ============================================================================
# Gene-level analysis
# ============================================================================

def gene_level_analysis(exome_df, constraint_df):
    """Correlate per-gene mean editability with gnomAD LOEUF."""
    logger.info("\n" + "=" * 60)
    logger.info("GENE-LEVEL: Mean editability vs gnomAD constraint")
    logger.info("=" * 60)

    # Compute per-gene editability statistics
    gene_stats = exome_df.groupby("gene").agg(
        n_positions=("editability_score", "size"),
        mean_score=("editability_score", "mean"),
        median_score=("editability_score", "median"),
        max_score=("editability_score", "max"),
        std_score=("editability_score", "std"),
        frac_high=("editability_score", lambda x: (x >= 0.5).mean()),
    ).reset_index()

    # Merge with constraint
    merged = gene_stats.merge(constraint_df[["gene", "lof.oe_ci.upper", "lof.pLI",
                                              "lof.obs", "lof.exp", "lof.z_score",
                                              "mis.oe", "mis.z_score",
                                              "syn.oe", "syn.z_score",
                                              "cds_length"]],
                               on="gene", how="inner")
    merged = merged.rename(columns={"lof.oe_ci.upper": "LOEUF"})

    # Drop genes with missing constraint
    merged = merged.dropna(subset=["LOEUF", "mean_score"])
    logger.info(f"  Genes with both editability and LOEUF: {len(merged):,}")

    # Filter to genes with enough positions
    merged_filt = merged[merged["n_positions"] >= 10].copy()
    logger.info(f"  After filter (>=10 positions): {len(merged_filt):,}")

    results = {"n_genes": int(len(merged_filt))}

    # Correlations
    for metric, col in [("mean_score", "mean_score"), ("frac_high", "frac_high"),
                         ("max_score", "max_score")]:
        for constraint, ccol in [("LOEUF", "LOEUF"), ("pLI", "lof.pLI"),
                                  ("lof_zscore", "lof.z_score"),
                                  ("mis_oe", "mis.oe"), ("mis_zscore", "mis.z_score")]:
            valid = merged_filt[[col, ccol]].dropna()
            if len(valid) < 30:
                continue

            rho, p = stats.spearmanr(valid[col], valid[ccol])
            r_pearson, p_pearson = stats.pearsonr(valid[col], valid[ccol])
            results[f"{metric}_vs_{constraint}"] = {
                "spearman_rho": float(rho),
                "spearman_p": float(p),
                "pearson_r": float(r_pearson),
                "pearson_p": float(p_pearson),
                "n": int(len(valid)),
            }
            if p < 0.05:
                logger.info(f"  {metric} vs {constraint}: rho={rho:.4f} (p={p:.2e}), "
                             f"r={r_pearson:.4f} (p={p_pearson:.2e})")

    # Decile analysis: LOEUF deciles vs mean editability
    merged_filt["loeuf_decile"] = pd.qcut(merged_filt["LOEUF"], 10, labels=False,
                                           duplicates="drop")
    decile_stats = merged_filt.groupby("loeuf_decile").agg(
        mean_editability=("mean_score", "mean"),
        mean_loeuf=("LOEUF", "mean"),
        n_genes=("gene", "size"),
    ).reset_index()

    results["decile_analysis"] = decile_stats.to_dict(orient="records")

    # Top constrained vs bottom constrained
    n_top = len(merged_filt) // 5
    top_constrained = merged_filt.nsmallest(n_top, "LOEUF")  # Low LOEUF = more constrained
    bottom_constrained = merged_filt.nlargest(n_top, "LOEUF")

    mw_stat, mw_p = stats.mannwhitneyu(
        top_constrained["mean_score"], bottom_constrained["mean_score"],
        alternative="two-sided"
    )
    results["top_vs_bottom_quintile"] = {
        "top_constrained_mean_score": float(top_constrained["mean_score"].mean()),
        "bottom_constrained_mean_score": float(bottom_constrained["mean_score"].mean()),
        "mann_whitney_p": float(mw_p),
        "n_per_group": int(n_top),
    }
    logger.info(f"\n  Top constrained (LOEUF low) mean editability: "
                 f"{top_constrained['mean_score'].mean():.4f}")
    logger.info(f"  Bottom constrained (LOEUF high) mean editability: "
                 f"{bottom_constrained['mean_score'].mean():.4f}")
    logger.info(f"  Mann-Whitney p={mw_p:.2e}")

    # Haploinsufficiency genes specifically
    hi_genes = merged_filt[merged_filt["lof.pLI"] >= 0.9]
    non_hi = merged_filt[merged_filt["lof.pLI"] < 0.1]
    if len(hi_genes) >= 10 and len(non_hi) >= 10:
        mw_stat2, mw_p2 = stats.mannwhitneyu(
            hi_genes["mean_score"], non_hi["mean_score"], alternative="two-sided"
        )
        results["haploinsufficient_vs_tolerant"] = {
            "hi_mean_score": float(hi_genes["mean_score"].mean()),
            "tolerant_mean_score": float(non_hi["mean_score"].mean()),
            "mann_whitney_p": float(mw_p2),
            "n_hi": int(len(hi_genes)),
            "n_tolerant": int(len(non_hi)),
        }
        logger.info(f"\n  Haploinsufficient (pLI>=0.9) mean: {hi_genes['mean_score'].mean():.4f} "
                     f"(n={len(hi_genes)})")
        logger.info(f"  LoF-tolerant (pLI<0.1) mean: {non_hi['mean_score'].mean():.4f} "
                     f"(n={len(non_hi)})")
        logger.info(f"  Mann-Whitney p={mw_p2:.2e}")

    # --- Figures ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Panel A: Scatter of mean editability vs LOEUF
    ax = axes[0, 0]
    ax.scatter(merged_filt["LOEUF"], merged_filt["mean_score"],
               alpha=0.1, s=3, color="steelblue")
    rho = results.get("mean_score_vs_LOEUF", {}).get("spearman_rho", 0)
    p = results.get("mean_score_vs_LOEUF", {}).get("spearman_p", 1)
    ax.set_xlabel("LOEUF (low = more constrained)")
    ax.set_ylabel("Mean editability score")
    ax.set_title(f"Gene Editability vs Constraint\n(rho={rho:.4f}, p={p:.2e})")

    # Add trend line
    z = np.polyfit(merged_filt["LOEUF"], merged_filt["mean_score"], 1)
    poly = np.poly1d(z)
    x_range = np.linspace(merged_filt["LOEUF"].min(), merged_filt["LOEUF"].max(), 100)
    ax.plot(x_range, poly(x_range), color="red", linewidth=2, alpha=0.7)

    # Panel B: LOEUF decile boxplot
    ax = axes[0, 1]
    decile_groups = [merged_filt[merged_filt["loeuf_decile"] == d]["mean_score"].values
                     for d in sorted(merged_filt["loeuf_decile"].unique())]
    bp = ax.boxplot(decile_groups, labels=[str(d) for d in sorted(merged_filt["loeuf_decile"].unique())])
    ax.set_xlabel("LOEUF decile (0=most constrained)")
    ax.set_ylabel("Mean editability score")
    ax.set_title("Editability by Constraint Decile")

    # Panel C: pLI distribution colored by editability
    ax = axes[1, 0]
    high_edit = merged_filt[merged_filt["mean_score"] >= merged_filt["mean_score"].quantile(0.9)]
    low_edit = merged_filt[merged_filt["mean_score"] <= merged_filt["mean_score"].quantile(0.1)]
    bins = np.linspace(0, 1, 50)
    ax.hist(low_edit["lof.pLI"].dropna(), bins=bins, alpha=0.6, label=f"Low editability (n={len(low_edit)})",
            color="steelblue", density=True)
    ax.hist(high_edit["lof.pLI"].dropna(), bins=bins, alpha=0.6, label=f"High editability (n={len(high_edit)})",
            color="firebrick", density=True)
    ax.set_xlabel("pLI")
    ax.set_ylabel("Density")
    ax.set_title("pLI Distribution by Editability")
    ax.legend(fontsize=8)

    # Panel D: Gene-level editability distribution
    ax = axes[1, 1]
    ax.hist(merged_filt["mean_score"], bins=50, color="steelblue", alpha=0.7, edgecolor="none")
    ax.set_xlabel("Mean editability score per gene")
    ax.set_ylabel("Number of genes")
    ax.set_title(f"Gene Editability Distribution (n={len(merged_filt):,})")
    ax.axvline(merged_filt["mean_score"].mean(), color="red", linestyle="--",
               label=f"Mean={merged_filt['mean_score'].mean():.3f}")
    ax.legend()

    plt.tight_layout()
    fig_path = OUTPUT_DIR / "gnomad_gene_level.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"\nFigure saved: {fig_path}")

    return results, merged_filt


# ============================================================================
# Site-level analysis (chr22 pilot)
# ============================================================================

def site_level_chr22_pilot(exome_df):
    """Download gnomAD chr22 VCF and check C>T variant frequencies at scored positions.

    This is a pilot to see if high-editability positions have more/less
    common C>T variants in the population.
    """
    logger.info("\n" + "=" * 60)
    logger.info("SITE-LEVEL: chr22 gnomAD C>T variant pilot")
    logger.info("=" * 60)

    # Filter exome map to chr22
    chr22 = exome_df[exome_df["chr"] == "chr22"].copy()
    if len(chr22) == 0:
        logger.warning("No chr22 data in exome map")
        return None

    logger.info(f"  chr22 exome positions: {len(chr22):,}")

    # Try to download gnomAD chr22 VCF
    if not GNOMAD_VCF_LOCAL.exists():
        logger.info(f"  gnomAD chr22 VCF not found locally")
        logger.info(f"  Attempting download (this is ~1.5 GB)...")

        GNOMAD_VCF_LOCAL.parent.mkdir(parents=True, exist_ok=True)
        try:
            import urllib.request
            urllib.request.urlretrieve(GNOMAD_VCF_URL, GNOMAD_VCF_LOCAL)
            logger.info(f"  Downloaded: {GNOMAD_VCF_LOCAL.stat().st_size / 1e9:.1f} GB")
        except Exception as e:
            logger.warning(f"  Download failed: {e}")
            logger.info("  Falling back to gene-level analysis only")
            return None

    if not GNOMAD_VCF_LOCAL.exists():
        logger.warning("  gnomAD VCF not available, skipping site-level")
        return None

    # Parse C>T variants from VCF (streaming, filter to C>T only)
    logger.info("  Parsing gnomAD chr22 VCF for C>T variants...")

    # Build set of scored positions for fast lookup
    scored_positions = set(zip(chr22["chr"], chr22["pos"]))
    logger.info(f"  {len(scored_positions):,} scored positions to check")

    ct_variants = {}  # pos -> allele frequency
    n_parsed = 0

    try:
        import pysam
        vcf = pysam.VariantFile(str(GNOMAD_VCF_LOCAL))
        for record in vcf:
            n_parsed += 1
            if n_parsed % 1000000 == 0:
                logger.info(f"    Parsed {n_parsed:,} VCF records, found {len(ct_variants):,} C>T variants")

            if record.ref == "C" and "T" in record.alts:
                pos_0based = record.pos - 1  # VCF is 1-based
                chrom = "chr22" if not record.chrom.startswith("chr") else record.chrom

                # Get allele frequency
                af = record.info.get("AF", [0])
                if isinstance(af, (list, tuple)):
                    af = af[0] if af else 0
                ct_variants[pos_0based] = float(af)

            elif record.ref == "G" and "A" in record.alts:
                pos_0based = record.pos - 1
                chrom = "chr22" if not record.chrom.startswith("chr") else record.chrom
                af = record.info.get("AF", [0])
                if isinstance(af, (list, tuple)):
                    af = af[0] if af else 0
                ct_variants[pos_0based] = float(af)

        vcf.close()
    except ImportError:
        logger.warning("  pysam not available, trying gzip parse...")
        # Fallback: parse bgzipped VCF manually
        try:
            with gzip.open(str(GNOMAD_VCF_LOCAL), "rt") as f:
                for line in f:
                    if line.startswith("#"):
                        continue
                    n_parsed += 1
                    if n_parsed % 1000000 == 0:
                        logger.info(f"    Parsed {n_parsed:,} lines, "
                                     f"found {len(ct_variants):,} C>T variants")

                    fields = line.strip().split("\t", 8)
                    if len(fields) < 8:
                        continue

                    ref = fields[3]
                    alt = fields[4]

                    if (ref == "C" and alt == "T") or (ref == "G" and alt == "A"):
                        pos_0based = int(fields[1]) - 1
                        # Parse AF from INFO
                        info = fields[7]
                        af = 0.0
                        for kv in info.split(";"):
                            if kv.startswith("AF="):
                                try:
                                    af = float(kv.split("=")[1].split(",")[0])
                                except (ValueError, IndexError):
                                    pass
                                break
                        ct_variants[pos_0based] = af
        except Exception as e:
            logger.error(f"  Failed to parse VCF: {e}")
            return None

    logger.info(f"  Total C>T/G>A variants found: {len(ct_variants):,}")

    # Match to exome map positions
    chr22["has_gnomad_variant"] = chr22["pos"].isin(ct_variants)
    chr22["gnomad_af"] = chr22["pos"].map(ct_variants).fillna(0.0)

    n_with_variant = chr22["has_gnomad_variant"].sum()
    logger.info(f"  Scored positions with gnomAD C>T variant: {n_with_variant:,} / {len(chr22):,} "
                 f"({100*n_with_variant/len(chr22):.1f}%)")

    # Analysis: is editability correlated with having a C>T variant?
    results = {
        "n_chr22_positions": int(len(chr22)),
        "n_gnomad_ct_variants": int(len(ct_variants)),
        "n_matched": int(n_with_variant),
        "frac_with_variant": float(n_with_variant / len(chr22)),
    }

    # Editability of positions with vs without gnomAD variant
    with_var = chr22[chr22["has_gnomad_variant"]]["editability_score"]
    without_var = chr22[~chr22["has_gnomad_variant"]]["editability_score"]

    if len(with_var) > 10 and len(without_var) > 10:
        mw_stat, mw_p = stats.mannwhitneyu(with_var, without_var, alternative="two-sided")
        results["editability_with_variant"] = float(with_var.mean())
        results["editability_without_variant"] = float(without_var.mean())
        results["mann_whitney_p"] = float(mw_p)
        logger.info(f"  Mean editability: with variant={with_var.mean():.4f}, "
                     f"without={without_var.mean():.4f}, MWU p={mw_p:.2e}")

    # Correlation between editability and AF (among sites with variants)
    sites_with_af = chr22[chr22["gnomad_af"] > 0]
    if len(sites_with_af) > 30:
        rho, p = stats.spearmanr(sites_with_af["editability_score"],
                                  sites_with_af["gnomad_af"])
        results["editability_vs_af_spearman"] = float(rho)
        results["editability_vs_af_p"] = float(p)
        results["n_sites_with_af"] = int(len(sites_with_af))
        logger.info(f"  Editability vs AF: rho={rho:.4f}, p={p:.2e} (n={len(sites_with_af):,})")

    # Decile analysis
    chr22["edit_decile"] = pd.qcut(chr22["editability_score"], 10, labels=False,
                                    duplicates="drop")
    decile_variant_rate = chr22.groupby("edit_decile")["has_gnomad_variant"].mean()
    results["decile_variant_rates"] = decile_variant_rate.to_dict()

    # Figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: Variant rate by editability decile
    ax = axes[0]
    deciles = sorted(decile_variant_rate.index)
    rates = [decile_variant_rate[d] for d in deciles]
    ax.bar(deciles, rates, color="steelblue", alpha=0.8)
    ax.set_xlabel("Editability decile (0=lowest)")
    ax.set_ylabel("Fraction with gnomAD C>T variant")
    ax.set_title("C>T Variant Rate by Editability")

    # Panel B: AF distribution for high vs low editability
    ax = axes[1]
    if len(sites_with_af) > 0:
        q90 = chr22["editability_score"].quantile(0.9)
        q10 = chr22["editability_score"].quantile(0.1)
        high_af = sites_with_af[sites_with_af["editability_score"] >= q90]["gnomad_af"]
        low_af = sites_with_af[sites_with_af["editability_score"] <= q10]["gnomad_af"]

        if len(high_af) > 0 and len(low_af) > 0:
            bins = np.logspace(-6, 0, 50)
            ax.hist(low_af, bins=bins, alpha=0.6, label=f"Low edit (n={len(low_af)})",
                    color="steelblue", density=True)
            ax.hist(high_af, bins=bins, alpha=0.6, label=f"High edit (n={len(high_af)})",
                    color="firebrick", density=True)
            ax.set_xscale("log")
            ax.set_xlabel("Allele Frequency")
            ax.set_ylabel("Density")
            ax.set_title("AF Distribution by Editability")
            ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No variants with AF", transform=ax.transAxes,
                ha="center", va="center")

    # Panel C: Editability distribution with/without variant
    ax = axes[2]
    bins = np.linspace(0, 1, 50)
    ax.hist(without_var, bins=bins, alpha=0.6, label=f"No variant (n={len(without_var):,})",
            color="steelblue", density=True)
    ax.hist(with_var, bins=bins, alpha=0.6, label=f"Has C>T variant (n={len(with_var):,})",
            color="firebrick", density=True)
    ax.set_xlabel("Editability score")
    ax.set_ylabel("Density")
    ax.set_title("Editability: Sites With vs Without gnomAD C>T")
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig_path = OUTPUT_DIR / "gnomad_site_level_chr22.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Figure saved: {fig_path}")

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    t_start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load exome map
    exome_df = load_exome_map()
    if exome_df is None:
        logger.error("Cannot proceed without exome map")
        return

    # Load gnomAD constraint
    constraint_df = load_gnomad_constraint()
    if constraint_df is None:
        logger.error("Cannot proceed without gnomAD constraint data")
        return

    # Gene-level analysis
    gene_results, merged_df = gene_level_analysis(exome_df, constraint_df)

    # Save merged data for downstream use
    merged_path = OUTPUT_DIR / "gene_editability_constraint.csv"
    merged_df.to_csv(merged_path, index=False)
    logger.info(f"Saved gene-level data: {merged_path}")

    # Site-level chr22 pilot (only if gnomAD VCF available or downloadable)
    site_results = site_level_chr22_pilot(exome_df)

    # Combine results
    all_results = {
        "gene_level": gene_results,
        "site_level_chr22": site_results,
    }

    with open(OUTPUT_DIR / "gnomad_analysis_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    elapsed = time.time() - t_start
    logger.info(f"\n{'='*60}")
    logger.info(f"COMPLETE ({elapsed:.0f}s)")
    logger.info(f"Results: {OUTPUT_DIR / 'gnomad_analysis_results.json'}")

    # Summary
    logger.info("\n--- SUMMARY ---")
    if gene_results:
        rho_loeuf = gene_results.get("mean_score_vs_LOEUF", {}).get("spearman_rho", "N/A")
        p_loeuf = gene_results.get("mean_score_vs_LOEUF", {}).get("spearman_p", "N/A")
        logger.info(f"Gene editability vs LOEUF: rho={rho_loeuf}, p={p_loeuf}")
        logger.info(f"Number of genes: {gene_results.get('n_genes', 0)}")

        hi = gene_results.get("haploinsufficient_vs_tolerant", {})
        if hi:
            logger.info(f"Haploinsufficient vs tolerant: {hi.get('hi_mean_score', 0):.4f} vs "
                         f"{hi.get('tolerant_mean_score', 0):.4f} (p={hi.get('mann_whitney_p', 1):.2e})")

    if site_results:
        logger.info(f"Site-level: {site_results.get('n_matched', 0):,} positions with gnomAD C>T")
        logger.info(f"  With variant editability: {site_results.get('editability_with_variant', 'N/A')}")
        logger.info(f"  Without variant: {site_results.get('editability_without_variant', 'N/A')}")


if __name__ == "__main__":
    main()
