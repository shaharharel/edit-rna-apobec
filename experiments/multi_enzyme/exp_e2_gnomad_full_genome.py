#!/usr/bin/env python3
"""E2: gnomAD Site-Level Full Genome Analysis.

For ALL chromosomes (1-22, X, Y):
- Load exome editability map (hg19) and liftover to hg38
- Parse gnomAD v4.1 exome VCFs (hg38) for C>T variants only
- Compute: Spearman(editability, AF), Mann-Whitney(with vs without variant),
  OR at editability percentiles, decile analysis, trinucleotide stratification,
  CpG control analysis

Usage:
    /opt/miniconda3/envs/quris/bin/python -u experiments/multi_enzyme/exp_e2_gnomad_full_genome.py
"""

import gzip
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
from pyliftover import LiftOver
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# --- Paths ---
EXOME_MAP_DIR = PROJECT_ROOT / "experiments/multi_enzyme/outputs/exome_map"
GNOMAD_DIR = PROJECT_ROOT / "data/raw/gnomad"
OUTPUT_DIR = PROJECT_ROOT / "experiments/multi_enzyme/outputs/evolutionary/e2_gnomad_full_genome"

CHROMOSOMES = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]


# =============================================================================
# Liftover: hg19 -> hg38
# =============================================================================

def liftover_exome_map(df, lo):
    """Liftover exome map positions from hg19 to hg38.

    Args:
        df: DataFrame with columns chr, pos (hg19 coordinates)
        lo: LiftOver object (hg19 -> hg38)

    Returns:
        DataFrame with added hg38_pos column. Rows that fail liftover are dropped.
    """
    # Extract arrays for fast iteration (avoid iterrows overhead)
    chroms = df["chr"].values
    positions = df["pos"].values.astype(int)

    hg38_positions = np.full(len(df), np.nan)
    n_failed = 0

    for i in range(len(positions)):
        result = lo.convert_coordinate(chroms[i], positions[i])
        if result and len(result) > 0:
            hg38_positions[i] = result[0][1]
        else:
            n_failed += 1

    df = df.copy()
    df["hg38_pos"] = hg38_positions
    n_before = len(df)
    df = df.dropna(subset=["hg38_pos"])
    df["hg38_pos"] = df["hg38_pos"].astype(int)
    n_after = len(df)
    if n_failed > 0:
        logger.info(f"  Liftover: {n_before:,} -> {n_after:,} ({n_failed} failed)")
    return df


# =============================================================================
# Parse gnomAD VCF for C>T variants
# =============================================================================

def _extract_af_from_info(info):
    """Fast AF extraction from gnomAD INFO string using string find."""
    # Find AF= (but not AC_AF= etc.)
    idx = info.find("AF=")
    if idx == -1:
        return 0.0
    # Make sure it's the start or preceded by ;
    if idx > 0 and info[idx - 1] != ';':
        # Search for ;AF=
        idx = info.find(";AF=")
        if idx == -1:
            return 0.0
        idx += 1  # skip the ;

    val_start = idx + 3  # skip "AF="
    val_end = info.find(";", val_start)
    if val_end == -1:
        val_end = len(info)
    val_str = info[val_start:val_end]
    # Handle multiallelic (take first)
    comma = val_str.find(",")
    if comma != -1:
        val_str = val_str[:comma]
    try:
        return float(val_str)
    except ValueError:
        return 0.0


def parse_gnomad_vcf(vcf_path, target_positions=None):
    """Parse gnomAD VCF for C>T (and G>A on - strand) SNVs.

    Args:
        vcf_path: Path to bgzipped VCF
        target_positions: optional set of positions (hg38, 1-based) to restrict to.
                         If None, collect all C>T variants.

    Returns:
        dict: pos (1-based) -> af (float)
    """
    variants = {}
    n_parsed = 0
    n_ct_total = 0

    with gzip.open(str(vcf_path), "rt") as f:
        for line in f:
            if line[0] == "#":
                continue
            n_parsed += 1

            if n_parsed % 5_000_000 == 0:
                logger.info(f"    Parsed {n_parsed:,} records, {len(variants):,} C>T at target")

            # Fast pre-filter: split only first 8 fields
            fields = line.split("\t", 8)
            if len(fields) < 8:
                continue

            ref = fields[3]
            alt = fields[4]

            # Only single-base C>T or G>A (complement strand)
            if not ((ref == "C" and alt == "T") or (ref == "G" and alt == "A")):
                continue

            n_ct_total += 1
            pos = int(fields[1])  # 1-based

            # If we have target positions, skip non-matching
            if target_positions is not None and pos not in target_positions:
                continue

            # Parse AF from INFO (fast extraction)
            af = _extract_af_from_info(fields[7])
            variants[pos] = af

    logger.info(f"    Done: {n_parsed:,} records, {n_ct_total:,} C>T/G>A total, "
                f"{len(variants):,} at target positions")
    return variants


# =============================================================================
# Per-chromosome analysis
# =============================================================================

def process_chromosome(chrom, lo):
    """Process a single chromosome: load exome map, liftover, parse gnomAD, match.

    Returns:
        DataFrame with columns: chr, pos_hg19, pos_hg38, strand, score_full, score_motif,
                                trinuc, gene, has_variant, af
    """
    t0 = time.time()

    # Load exome map
    exome_path = EXOME_MAP_DIR / f"exome_editability_{chrom}.csv.gz"
    if not exome_path.exists():
        logger.warning(f"  No exome map for {chrom}, skipping")
        return None

    exome_df = pd.read_csv(exome_path)
    logger.info(f"  {chrom}: {len(exome_df):,} exome positions (hg19)")

    # Liftover to hg38
    exome_df = liftover_exome_map(exome_df, lo)
    if len(exome_df) == 0:
        logger.warning(f"  {chrom}: no positions survived liftover")
        return None

    logger.info(f"  {chrom}: {len(exome_df):,} positions after liftover to hg38")

    # Build set of hg38 positions (1-based, to match VCF)
    # pyliftover returns 0-based, VCF is 1-based
    # So we need hg38_pos + 1 for matching to VCF POS
    target_positions_1based = set(exome_df["hg38_pos"].values + 1)

    # Parse gnomAD VCF
    vcf_path = GNOMAD_DIR / f"gnomad.exomes.v4.1.sites.{chrom}.vcf.bgz"
    if not vcf_path.exists():
        logger.warning(f"  gnomAD VCF not found for {chrom}")
        return None

    logger.info(f"  Parsing gnomAD VCF for {chrom}...")
    variants = parse_gnomad_vcf(vcf_path, target_positions=target_positions_1based)

    # Match: convert VCF 1-based back to 0-based for merging
    # exome_df.hg38_pos is 0-based (from pyliftover)
    # VCF POS is 1-based, so variant keys are 1-based
    # Match: exome hg38_pos + 1 == VCF POS
    variant_pos_set = set(variants.keys())
    exome_df["vcf_pos"] = exome_df["hg38_pos"] + 1
    exome_df["has_variant"] = exome_df["vcf_pos"].isin(variant_pos_set)
    exome_df["af"] = exome_df["vcf_pos"].map(variants).fillna(0.0)

    n_matched = exome_df["has_variant"].sum()
    elapsed = time.time() - t0
    logger.info(f"  {chrom}: {n_matched:,}/{len(exome_df):,} positions have gnomAD C>T variant "
                f"({100*n_matched/len(exome_df):.1f}%) [{elapsed:.0f}s]")

    return exome_df


# =============================================================================
# Aggregate analysis
# =============================================================================

def compute_statistics(all_data):
    """Compute all requested statistics on the combined dataset."""
    results = {}

    n_total = len(all_data)
    n_with_var = all_data["has_variant"].sum()
    n_without_var = n_total - n_with_var

    results["n_total_positions"] = int(n_total)
    results["n_with_variant"] = int(n_with_var)
    results["n_without_variant"] = int(n_without_var)
    results["frac_with_variant"] = float(n_with_var / n_total) if n_total > 0 else 0

    logger.info(f"\n{'='*60}")
    logger.info(f"AGGREGATE STATISTICS ({n_total:,} positions)")
    logger.info(f"{'='*60}")
    logger.info(f"Positions with gnomAD C>T variant: {n_with_var:,} ({100*n_with_var/n_total:.1f}%)")

    # --- 1. Mann-Whitney: editability WITH vs WITHOUT variant ---
    with_var = all_data[all_data["has_variant"]]["score_full"]
    without_var = all_data[~all_data["has_variant"]]["score_full"]

    mw_stat, mw_p = stats.mannwhitneyu(with_var, without_var, alternative="two-sided")
    results["mann_whitney"] = {
        "mean_with_variant": float(with_var.mean()),
        "mean_without_variant": float(without_var.mean()),
        "median_with_variant": float(with_var.median()),
        "median_without_variant": float(without_var.median()),
        "statistic": float(mw_stat),
        "p_value": float(mw_p),
    }
    logger.info(f"\nMann-Whitney: editability WITH vs WITHOUT variant")
    logger.info(f"  Mean with variant:    {with_var.mean():.6f}")
    logger.info(f"  Mean without variant: {without_var.mean():.6f}")
    logger.info(f"  p = {mw_p:.2e}")

    # --- 2. Spearman: editability vs AF (among sites with variants and AF > 0) ---
    sites_af = all_data[(all_data["has_variant"]) & (all_data["af"] > 0)].copy()
    if len(sites_af) > 30:
        rho, p = stats.spearmanr(sites_af["score_full"], sites_af["af"])
        results["spearman_editability_vs_af"] = {
            "rho": float(rho),
            "p_value": float(p),
            "n": int(len(sites_af)),
        }
        logger.info(f"\nSpearman: editability vs AF (n={len(sites_af):,})")
        logger.info(f"  rho = {rho:.6f}, p = {p:.2e}")

    # --- 3. Decile analysis: variant rate in each editability decile ---
    all_data["edit_decile"] = pd.qcut(all_data["score_full"], 10, labels=False, duplicates="drop")
    decile_stats = all_data.groupby("edit_decile").agg(
        n_positions=("score_full", "size"),
        mean_score=("score_full", "mean"),
        variant_rate=("has_variant", "mean"),
        mean_af_all=("af", "mean"),
    ).reset_index()

    # Mean AF among variants only per decile
    variants_only = all_data[all_data["has_variant"]].copy()
    if len(variants_only) > 0:
        decile_af = variants_only.groupby("edit_decile")["af"].mean()
        decile_stats["mean_af_variants"] = decile_stats["edit_decile"].map(decile_af)

    results["decile_analysis"] = decile_stats.to_dict(orient="records")
    logger.info(f"\nDecile analysis (variant rate by editability decile):")
    for _, row in decile_stats.iterrows():
        logger.info(f"  Decile {int(row['edit_decile'])}: score={row['mean_score']:.4f}, "
                     f"variant_rate={row['variant_rate']:.4f}, n={int(row['n_positions']):,}")

    # --- 4. OR at editability percentiles ---
    percentile_ors = {}
    for pct in [50, 75, 90, 95]:
        threshold = np.percentile(all_data["score_full"], pct)
        high = all_data[all_data["score_full"] >= threshold]
        low = all_data[all_data["score_full"] < threshold]

        a = high["has_variant"].sum()  # high edit + variant
        b = (~high["has_variant"]).sum()  # high edit + no variant
        c = low["has_variant"].sum()  # low edit + variant
        d = (~low["has_variant"]).sum()  # low edit + no variant

        if b > 0 and c > 0:
            odds_ratio = (a * d) / (b * c) if (b * c) > 0 else float("inf")
            # Fisher's exact test on 2x2 table (use chi2 for large N)
            chi2, chi2_p, _, _ = stats.chi2_contingency([[a, b], [c, d]])
            percentile_ors[f"p{pct}"] = {
                "threshold": float(threshold),
                "odds_ratio": float(odds_ratio),
                "chi2": float(chi2),
                "p_value": float(chi2_p),
                "n_high": int(len(high)),
                "n_low": int(len(low)),
                "variant_rate_high": float(a / len(high)) if len(high) > 0 else 0,
                "variant_rate_low": float(c / len(low)) if len(low) > 0 else 0,
            }
            logger.info(f"\nOR at p{pct} (threshold={threshold:.4f}):")
            logger.info(f"  OR = {odds_ratio:.4f}, p = {chi2_p:.2e}")
            logger.info(f"  Variant rate: high={a/len(high):.4f}, low={c/len(low):.4f}")

    results["percentile_odds_ratios"] = percentile_ors

    # --- 5. Trinucleotide stratification ---
    trinuc_results = {}
    for trinuc in ["TC", "CC"]:
        subset = all_data[all_data["trinuc"].str.contains(trinuc, case=False, na=False)]
        if len(subset) < 100:
            continue

        sub_with = subset[subset["has_variant"]]["score_full"]
        sub_without = subset[~subset["has_variant"]]["score_full"]

        trinuc_res = {
            "n_total": int(len(subset)),
            "n_with_variant": int(len(sub_with)),
            "frac_with_variant": float(len(sub_with) / len(subset)),
        }

        if len(sub_with) > 10 and len(sub_without) > 10:
            mw_s, mw_p = stats.mannwhitneyu(sub_with, sub_without, alternative="two-sided")
            trinuc_res["mean_with_variant"] = float(sub_with.mean())
            trinuc_res["mean_without_variant"] = float(sub_without.mean())
            trinuc_res["mann_whitney_p"] = float(mw_p)

        # Spearman within trinuc
        sub_af = subset[(subset["has_variant"]) & (subset["af"] > 0)]
        if len(sub_af) > 30:
            rho, p = stats.spearmanr(sub_af["score_full"], sub_af["af"])
            trinuc_res["spearman_rho"] = float(rho)
            trinuc_res["spearman_p"] = float(p)
            trinuc_res["n_for_spearman"] = int(len(sub_af))

        trinuc_results[trinuc] = trinuc_res
        logger.info(f"\nTrinucleotide: {trinuc} (n={len(subset):,})")
        if "mann_whitney_p" in trinuc_res:
            logger.info(f"  Mean with/without variant: {trinuc_res['mean_with_variant']:.4f} / "
                         f"{trinuc_res['mean_without_variant']:.4f}, p={trinuc_res['mann_whitney_p']:.2e}")
        if "spearman_rho" in trinuc_res:
            logger.info(f"  Spearman editability vs AF: rho={trinuc_res['spearman_rho']:.4f}, "
                         f"p={trinuc_res['spearman_p']:.2e}")

    # Also do "other" (non-TC, non-CC)
    other_mask = ~(all_data["trinuc"].str.contains("TC", case=False, na=False) |
                   all_data["trinuc"].str.contains("CC", case=False, na=False))
    other = all_data[other_mask]
    if len(other) >= 100:
        o_with = other[other["has_variant"]]["score_full"]
        o_without = other[~other["has_variant"]]["score_full"]
        other_res = {
            "n_total": int(len(other)),
            "n_with_variant": int(len(o_with)),
            "frac_with_variant": float(len(o_with) / len(other)),
        }
        if len(o_with) > 10 and len(o_without) > 10:
            mw_s, mw_p = stats.mannwhitneyu(o_with, o_without, alternative="two-sided")
            other_res["mean_with_variant"] = float(o_with.mean())
            other_res["mean_without_variant"] = float(o_without.mean())
            other_res["mann_whitney_p"] = float(mw_p)
        o_af = other[(other["has_variant"]) & (other["af"] > 0)]
        if len(o_af) > 30:
            rho, p = stats.spearmanr(o_af["score_full"], o_af["af"])
            other_res["spearman_rho"] = float(rho)
            other_res["spearman_p"] = float(p)
            other_res["n_for_spearman"] = int(len(o_af))
        trinuc_results["other"] = other_res
        logger.info(f"\nTrinucleotide: other (n={len(other):,})")
        if "mann_whitney_p" in other_res:
            logger.info(f"  Mean with/without variant: {other_res['mean_with_variant']:.4f} / "
                         f"{other_res['mean_without_variant']:.4f}, p={other_res['mann_whitney_p']:.2e}")

    results["trinucleotide_stratification"] = trinuc_results

    # --- 6. CpG context control ---
    # CpG sites have higher mutation rates -> control for this
    # Trinuc column: middle C. If trinuc is xCG -> CpG context
    all_data["is_cpg"] = all_data["trinuc"].str[1:].str.startswith("CG") | \
                          all_data["trinuc"].str[:2].str.endswith("CG")
    # More precise: center is C, trinuc[2] (position +1) is G => CpG
    # trinuc format: (m1)(center=C)(p1), so trinuc[2] == 'G' means CpG
    all_data["is_cpg"] = all_data["trinuc"].apply(
        lambda x: len(x) >= 3 and x[1] == "C" and x[2] == "G" if isinstance(x, str) else False
    )

    cpg_results = {}
    for cpg_label, mask in [("CpG", all_data["is_cpg"]), ("non-CpG", ~all_data["is_cpg"])]:
        subset = all_data[mask]
        if len(subset) < 100:
            continue
        sub_with = subset[subset["has_variant"]]["score_full"]
        sub_without = subset[~subset["has_variant"]]["score_full"]
        res = {
            "n_total": int(len(subset)),
            "n_with_variant": int(len(sub_with)),
            "frac_with_variant": float(len(sub_with) / len(subset)),
        }
        if len(sub_with) > 10 and len(sub_without) > 10:
            mw_s, mw_p = stats.mannwhitneyu(sub_with, sub_without, alternative="two-sided")
            res["mean_with_variant"] = float(sub_with.mean())
            res["mean_without_variant"] = float(sub_without.mean())
            res["mann_whitney_p"] = float(mw_p)
        sub_af = subset[(subset["has_variant"]) & (subset["af"] > 0)]
        if len(sub_af) > 30:
            rho, p = stats.spearmanr(sub_af["score_full"], sub_af["af"])
            res["spearman_rho"] = float(rho)
            res["spearman_p"] = float(p)
            res["n_for_spearman"] = int(len(sub_af))
        cpg_results[cpg_label] = res
        logger.info(f"\nCpG control: {cpg_label} (n={len(subset):,})")
        if "mann_whitney_p" in res:
            logger.info(f"  Mean with/without variant: {res['mean_with_variant']:.4f} / "
                         f"{res['mean_without_variant']:.4f}, p={res['mann_whitney_p']:.2e}")
        if "spearman_rho" in res:
            logger.info(f"  Spearman editability vs AF: rho={res['spearman_rho']:.4f}, "
                         f"p={res['spearman_p']:.2e}")

    results["cpg_control"] = cpg_results

    # --- 7. Within-trinucleotide control ---
    # For each trinucleotide, compute correlation between editability and variant presence
    # This controls for mutation rate differences between trinucleotides
    trinuc_controlled = {}
    all_trinucs = all_data["trinuc"].value_counts()
    logger.info(f"\nWithin-trinucleotide control (top 10 trinucs):")
    for trinuc in all_trinucs.head(16).index:
        subset = all_data[all_data["trinuc"] == trinuc]
        if len(subset) < 200:
            continue
        sub_with = subset[subset["has_variant"]]["score_full"]
        sub_without = subset[~subset["has_variant"]]["score_full"]
        if len(sub_with) < 10 or len(sub_without) < 10:
            continue
        mw_s, mw_p = stats.mannwhitneyu(sub_with, sub_without, alternative="two-sided")
        trinuc_controlled[trinuc] = {
            "n": int(len(subset)),
            "n_with_variant": int(len(sub_with)),
            "frac_with_variant": float(len(sub_with) / len(subset)),
            "mean_with": float(sub_with.mean()),
            "mean_without": float(sub_without.mean()),
            "mann_whitney_p": float(mw_p),
        }
        logger.info(f"  {trinuc}: n={len(subset):,}, with={sub_with.mean():.4f}, "
                     f"without={sub_without.mean():.4f}, p={mw_p:.2e}")

    results["within_trinucleotide_control"] = trinuc_controlled

    # --- 8. Per-chromosome summary ---
    chr_summary = all_data.groupby("chr").agg(
        n_positions=("score_full", "size"),
        n_with_variant=("has_variant", "sum"),
        mean_score=("score_full", "mean"),
    ).reset_index()
    chr_summary["frac_with_variant"] = chr_summary["n_with_variant"] / chr_summary["n_positions"]
    # Mean editability for positions with variants per chromosome
    chr_var_mean = all_data[all_data["has_variant"]].groupby("chr")["score_full"].mean()
    chr_summary["mean_score_with_var"] = chr_summary["chr"].map(chr_var_mean)
    results["per_chromosome"] = chr_summary.to_dict(orient="records")

    return results


def compare_to_chr22_pilot(results):
    """Compare full-genome results to chr22 pilot."""
    pilot_path = PROJECT_ROOT / "experiments/multi_enzyme/outputs/gnomad_site_level/gnomad_analysis_results.json"
    if not pilot_path.exists():
        logger.info("No chr22 pilot results found for comparison")
        return {}

    with open(pilot_path) as f:
        pilot = json.load(f)

    pilot_site = pilot.get("site_level_chr22", {})
    if not pilot_site:
        return {}

    comparison = {
        "pilot_chr22": {
            "spearman_rho": pilot_site.get("editability_vs_af_spearman"),
            "mw_p": pilot_site.get("mann_whitney_p"),
            "mean_with_variant": pilot_site.get("editability_with_variant"),
            "mean_without_variant": pilot_site.get("editability_without_variant"),
            "note": "Pilot used hg19 exome map positions matched directly to hg38 gnomAD without liftover",
        },
        "full_genome": {
            "spearman_rho": results.get("spearman_editability_vs_af", {}).get("rho"),
            "mw_p": results.get("mann_whitney", {}).get("p_value"),
            "mean_with_variant": results.get("mann_whitney", {}).get("mean_with_variant"),
            "mean_without_variant": results.get("mann_whitney", {}).get("mean_without_variant"),
        },
    }

    logger.info(f"\n{'='*60}")
    logger.info("COMPARISON TO CHR22 PILOT")
    logger.info(f"{'='*60}")
    logger.info(f"  Pilot (chr22, no liftover):  rho={comparison['pilot_chr22']['spearman_rho']:.4f}, "
                f"MWU p={comparison['pilot_chr22']['mw_p']:.2e}")
    logger.info(f"  Full genome (with liftover): rho={comparison['full_genome']['spearman_rho']:.4f}, "
                f"MWU p={comparison['full_genome']['mw_p']:.2e}")

    return comparison


# =============================================================================
# Figures
# =============================================================================

def make_figures(all_data, results):
    """Generate summary figures."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Panel A: Variant rate by editability decile
    ax = axes[0, 0]
    decile_df = pd.DataFrame(results["decile_analysis"])
    ax.bar(decile_df["edit_decile"], decile_df["variant_rate"], color="steelblue", alpha=0.8,
           edgecolor="white")
    ax.set_xlabel("Editability decile (0=lowest)")
    ax.set_ylabel("Fraction with gnomAD C>T variant")
    ax.set_title(f"C>T Variant Rate by Editability\n(n={results['n_total_positions']:,})")
    # Add trend line
    z = np.polyfit(decile_df["edit_decile"], decile_df["variant_rate"], 1)
    x = np.linspace(decile_df["edit_decile"].min(), decile_df["edit_decile"].max(), 100)
    ax.plot(x, np.poly1d(z)(x), color="red", linewidth=2, linestyle="--")

    # Panel B: Editability distribution WITH vs WITHOUT variant
    ax = axes[0, 1]
    with_var = all_data[all_data["has_variant"]]["score_full"]
    without_var = all_data[~all_data["has_variant"]]["score_full"]
    bins = np.linspace(0, 1, 80)
    ax.hist(without_var, bins=bins, alpha=0.6, density=True,
            label=f"No variant (n={len(without_var):,})", color="steelblue")
    ax.hist(with_var, bins=bins, alpha=0.6, density=True,
            label=f"Has C>T variant (n={len(with_var):,})", color="firebrick")
    ax.set_xlabel("Editability score")
    ax.set_ylabel("Density")
    mw_p = results["mann_whitney"]["p_value"]
    ax.set_title(f"Editability: With vs Without C>T\n(MWU p={mw_p:.1e})")
    ax.legend(fontsize=8)

    # Panel C: Editability vs AF scatter (subsampled)
    ax = axes[0, 2]
    sites_af = all_data[(all_data["has_variant"]) & (all_data["af"] > 0)].copy()
    if len(sites_af) > 50000:
        plot_df = sites_af.sample(50000, random_state=42)
    else:
        plot_df = sites_af
    ax.scatter(plot_df["score_full"], plot_df["af"], alpha=0.02, s=1, color="steelblue")
    ax.set_yscale("log")
    ax.set_xlabel("Editability score")
    ax.set_ylabel("Allele Frequency (log)")
    sp = results.get("spearman_editability_vs_af", {})
    rho = sp.get("rho", 0)
    p = sp.get("p_value", 1)
    ax.set_title(f"Editability vs AF\n(rho={rho:.4f}, p={p:.1e})")

    # Panel D: Trinucleotide stratification — variant rate by decile for TC vs CC vs other
    ax = axes[1, 0]
    for trinuc_label, color, marker in [("TC", "firebrick", "o"), ("CC", "steelblue", "s")]:
        mask = all_data["trinuc"].str.contains(trinuc_label, case=False, na=False)
        subset = all_data[mask].copy()
        if len(subset) < 100:
            continue
        subset["dec"] = pd.qcut(subset["score_full"], 10, labels=False, duplicates="drop")
        dec_rate = subset.groupby("dec")["has_variant"].mean()
        ax.plot(dec_rate.index, dec_rate.values, marker=marker, color=color,
                label=f"{trinuc_label} (n={len(subset):,})", linewidth=2, markersize=5)

    # Other
    other_mask = ~(all_data["trinuc"].str.contains("TC", case=False, na=False) |
                   all_data["trinuc"].str.contains("CC", case=False, na=False))
    other = all_data[other_mask].copy()
    if len(other) >= 100:
        other["dec"] = pd.qcut(other["score_full"], 10, labels=False, duplicates="drop")
        dec_rate = other.groupby("dec")["has_variant"].mean()
        ax.plot(dec_rate.index, dec_rate.values, marker="^", color="gray",
                label=f"Other (n={len(other):,})", linewidth=2, markersize=5)

    ax.set_xlabel("Editability decile (0=lowest)")
    ax.set_ylabel("Fraction with C>T variant")
    ax.set_title("Variant Rate by Trinuc Context")
    ax.legend(fontsize=8)

    # Panel E: OR at different percentile thresholds
    ax = axes[1, 1]
    por = results.get("percentile_odds_ratios", {})
    if por:
        pcts = sorted(por.keys(), key=lambda x: int(x[1:]))
        x_vals = [int(k[1:]) for k in pcts]
        or_vals = [por[k]["odds_ratio"] for k in pcts]
        ax.bar(range(len(x_vals)), or_vals, tick_label=[f"P{v}" for v in x_vals],
               color="steelblue", alpha=0.8, edgecolor="white")
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("Editability percentile threshold")
        ax.set_ylabel("Odds Ratio (high vs low editability)")
        ax.set_title("OR for C>T Variant at High Editability")

    # Panel F: CpG vs non-CpG comparison
    ax = axes[1, 2]
    cpg = results.get("cpg_control", {})
    if cpg:
        labels = list(cpg.keys())
        variant_rates = [cpg[l].get("frac_with_variant", 0) for l in labels]
        mean_with = [cpg[l].get("mean_with_variant", 0) for l in labels]
        mean_without = [cpg[l].get("mean_without_variant", 0) for l in labels]

        x = np.arange(len(labels))
        width = 0.35
        ax.bar(x - width/2, mean_with, width, label="With variant", color="firebrick", alpha=0.8)
        ax.bar(x + width/2, mean_without, width, label="Without variant", color="steelblue", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Mean editability score")
        ax.set_title("Editability by Variant Status\n(CpG Control)")
        ax.legend(fontsize=8)

        # Add p-values as text
        for i, l in enumerate(labels):
            p_val = cpg[l].get("mann_whitney_p", 1)
            ax.text(i, max(mean_with[i], mean_without[i]) + 0.005,
                    f"p={p_val:.1e}", ha="center", fontsize=7)

    plt.tight_layout()
    fig_path = OUTPUT_DIR / "e2_gnomad_full_genome.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"\nFigure saved: {fig_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    t_total = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize liftover
    logger.info("Initializing hg19 -> hg38 liftover...")
    lo = LiftOver("hg19", "hg38")
    logger.info("Liftover ready")

    # Process each chromosome
    all_dfs = []
    chr_timings = {}

    for chrom in CHROMOSOMES:
        t_chr = time.time()
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {chrom}")
        logger.info(f"{'='*60}")

        df = process_chromosome(chrom, lo)
        if df is not None:
            all_dfs.append(df)

        chr_timings[chrom] = time.time() - t_chr
        logger.info(f"  {chrom} done in {chr_timings[chrom]:.0f}s")

    if not all_dfs:
        logger.error("No data collected from any chromosome!")
        return

    # Combine
    logger.info(f"\nCombining {len(all_dfs)} chromosomes...")
    all_data = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"Total: {len(all_data):,} positions")

    # Save combined data (compressed)
    combined_path = OUTPUT_DIR / "gnomad_editability_matched.csv.gz"
    all_data.to_csv(combined_path, index=False, compression="gzip")
    logger.info(f"Saved combined data: {combined_path}")

    # Compute statistics
    results = compute_statistics(all_data)
    results["chr_timings"] = {k: float(v) for k, v in chr_timings.items()}
    results["total_time_seconds"] = float(time.time() - t_total)

    # Compare to chr22 pilot
    comparison = compare_to_chr22_pilot(results)
    results["comparison_to_pilot"] = comparison

    # Save results
    results_path = OUTPUT_DIR / "e2_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nResults saved: {results_path}")

    # Make figures
    logger.info("\nGenerating figures...")
    make_figures(all_data, results)

    # Summary
    elapsed = time.time() - t_total
    logger.info(f"\n{'='*60}")
    logger.info(f"E2 COMPLETE ({elapsed/60:.1f} min)")
    logger.info(f"{'='*60}")
    logger.info(f"Total positions: {results['n_total_positions']:,}")
    logger.info(f"With variant: {results['n_with_variant']:,} ({results['frac_with_variant']*100:.1f}%)")

    mw = results["mann_whitney"]
    logger.info(f"Mann-Whitney: mean with={mw['mean_with_variant']:.6f}, "
                f"without={mw['mean_without_variant']:.6f}, p={mw['p_value']:.2e}")

    sp = results.get("spearman_editability_vs_af", {})
    if sp:
        logger.info(f"Spearman editability vs AF: rho={sp['rho']:.6f}, p={sp['p_value']:.2e}")

    por = results.get("percentile_odds_ratios", {})
    for k in sorted(por.keys()):
        v = por[k]
        logger.info(f"OR at {k}: {v['odds_ratio']:.4f} (p={v['p_value']:.2e})")


if __name__ == "__main__":
    main()
