#!/usr/bin/env python
"""
Exome-wide ClinVar enrichment using the full XGB 40d exome editability map.

Matches ClinVar variants (hg38) to the pre-computed exome editability map (hg19)
via liftover, then compares editability scores of pathogenic vs benign variants.

Output: experiments/multi_enzyme/outputs/evolutionary/exome_clinvar_xgb/
"""

import glob
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pyliftover import LiftOver
from scipy import stats

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT = Path(__file__).resolve().parents[2]
EXOME_DIR = PROJECT / "experiments/multi_enzyme/outputs/exome_map"
CLINVAR_PATH = PROJECT / "experiments/apobec3a/outputs/clinvar_prediction/clinvar_all_scores.csv"
OUT_DIR = PROJECT / "experiments/multi_enzyme/outputs/evolutionary/exome_clinvar_xgb"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_exome_map():
    """Load all exome editability files into a single DataFrame."""
    files = sorted(glob.glob(str(EXOME_DIR / "exome_editability_chr*.csv.gz")))
    print(f"Loading {len(files)} exome map files...")
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)
        print(f"  {os.path.basename(f)}: {len(df):,} positions")
    exome = pd.concat(dfs, ignore_index=True)
    print(f"Total exome positions: {len(exome):,}")
    return exome


def load_clinvar():
    """Load ClinVar scored variants."""
    print(f"\nLoading ClinVar from {CLINVAR_PATH}...")
    cv = pd.read_csv(CLINVAR_PATH)
    print(f"Total ClinVar variants: {len(cv):,}")
    print(f"Significance distribution:\n{cv['significance_simple'].value_counts().to_string()}")
    return cv


def liftover_clinvar(cv):
    """Liftover ClinVar hg38 -> hg19 coordinates."""
    print("\nLifting over ClinVar hg38 → hg19...")
    lo = LiftOver("hg38", "hg19")

    hg19_chr = []
    hg19_pos = []
    failed = 0
    for _, row in cv.iterrows():
        result = lo.convert_coordinate(row["chr"], int(row["start"]))
        if result and len(result) > 0:
            hg19_chr.append(result[0][0])
            hg19_pos.append(int(result[0][1]))
        else:
            hg19_chr.append(None)
            hg19_pos.append(None)
            failed += 1

    cv = cv.copy()
    cv["hg19_chr"] = hg19_chr
    cv["hg19_pos"] = hg19_pos
    n_ok = cv["hg19_pos"].notna().sum()
    print(f"Liftover: {n_ok:,} succeeded, {failed:,} failed")
    cv = cv.dropna(subset=["hg19_pos"])
    cv["hg19_pos"] = cv["hg19_pos"].astype(int)
    return cv


def match_clinvar_to_exome(cv, exome):
    """Match ClinVar variants to exome map by (chr, pos)."""
    print("\nMatching ClinVar to exome map...")

    # Create lookup key
    exome_key = exome.assign(key=exome["chr"] + "_" + exome["pos"].astype(str))
    cv_key = cv.assign(key=cv["hg19_chr"] + "_" + cv["hg19_pos"].astype(str))

    exome_set = set(exome_key["key"])
    cv_matched_mask = cv_key["key"].isin(exome_set)
    print(f"ClinVar variants matching exome positions: {cv_matched_mask.sum():,} / {len(cv):,} ({100*cv_matched_mask.mean():.1f}%)")

    # Merge
    merged = cv_key[cv_matched_mask].merge(
        exome_key[["key", "score_full", "score_motif", "trinuc", "strand"]].rename(
            columns={"strand": "exome_strand"}
        ),
        on="key",
        how="inner",
    )
    # Deduplicate (some positions may appear on both strands)
    merged = merged.drop_duplicates(subset=["site_id"])
    print(f"After dedup: {len(merged):,} matched variants")
    return merged


def classify_significance(merged):
    """Create binary pathogenic/benign labels."""
    path_mask = merged["significance_simple"].isin(["Pathogenic", "Likely_pathogenic"])
    benign_mask = merged["significance_simple"].isin(["Benign", "Likely_benign"])
    merged = merged.copy()
    merged["label"] = None
    merged.loc[path_mask, "label"] = "Pathogenic"
    merged.loc[benign_mask, "label"] = "Benign"
    labeled = merged.dropna(subset=["label"])
    print(f"\nLabeled variants: {len(labeled):,}")
    print(f"  Pathogenic: {(labeled['label']=='Pathogenic').sum():,}")
    print(f"  Benign: {(labeled['label']=='Benign').sum():,}")
    return labeled


def compute_enrichment(labeled, score_col, tag, out_dir):
    """Compute enrichment statistics for a given score column."""
    results = {}
    path = labeled[labeled["label"] == "Pathogenic"][score_col].dropna()
    ben = labeled[labeled["label"] == "Benign"][score_col].dropna()

    # Basic stats
    results["n_pathogenic"] = len(path)
    results["n_benign"] = len(ben)
    results["mean_score_pathogenic"] = float(path.mean())
    results["mean_score_benign"] = float(ben.mean())
    results["median_score_pathogenic"] = float(path.median())
    results["median_score_benign"] = float(ben.median())

    # Mann-Whitney
    u_stat, u_pval = stats.mannwhitneyu(path, ben, alternative="greater")
    results["mannwhitney_U"] = float(u_stat)
    results["mannwhitney_pval"] = float(u_pval)

    # OR at percentile thresholds
    all_scores = labeled[score_col].dropna()
    for pct in [50, 75, 90, 95]:
        thresh = np.percentile(all_scores, pct)
        path_above = (path >= thresh).sum()
        path_below = (path < thresh).sum()
        ben_above = (ben >= thresh).sum()
        ben_below = (ben < thresh).sum()
        if ben_above > 0 and path_below > 0 and ben_below > 0:
            OR = (path_above / path_below) / (ben_above / ben_below)
            # Fisher exact
            table = [[path_above, path_below], [ben_above, ben_below]]
            fisher_or, fisher_p = stats.fisher_exact(table, alternative="greater")
        else:
            OR = float("nan")
            fisher_or, fisher_p = float("nan"), float("nan")
        results[f"OR_p{pct}"] = float(OR)
        results[f"fisher_OR_p{pct}"] = float(fisher_or)
        results[f"fisher_pval_p{pct}"] = float(fisher_p)
        results[f"threshold_p{pct}"] = float(thresh)
        results[f"n_path_above_p{pct}"] = int(path_above)
        results[f"n_ben_above_p{pct}"] = int(ben_above)

    print(f"\n{'='*60}")
    print(f"Enrichment: {tag} (score_col={score_col})")
    print(f"{'='*60}")
    print(f"N pathogenic: {results['n_pathogenic']:,}  |  N benign: {results['n_benign']:,}")
    print(f"Mean score — path: {results['mean_score_pathogenic']:.4f}  ben: {results['mean_score_benign']:.4f}")
    print(f"Median score — path: {results['median_score_pathogenic']:.4f}  ben: {results['median_score_benign']:.4f}")
    print(f"Mann-Whitney U={results['mannwhitney_U']:.0f}, p={results['mannwhitney_pval']:.2e} (path > ben)")
    for pct in [50, 75, 90, 95]:
        print(f"  p{pct}: thresh={results[f'threshold_p{pct}']:.4f}  "
              f"OR={results[f'OR_p{pct}']:.3f}  "
              f"Fisher OR={results[f'fisher_OR_p{pct}']:.3f}  "
              f"p={results[f'fisher_pval_p{pct}']:.2e}  "
              f"(path_above={results[f'n_path_above_p{pct}']:,}, ben_above={results[f'n_ben_above_p{pct}']:,})")

    return results


def trinucleotide_stratification(labeled, score_col):
    """Stratify enrichment by trinucleotide context."""
    print(f"\n{'='*60}")
    print(f"Trinucleotide stratification ({score_col})")
    print(f"{'='*60}")

    trinuc_results = {}
    # Group: TC (position -1 is T), CC (position -1 is C), other
    def get_trinuc_group(trinuc):
        if pd.isna(trinuc) or len(str(trinuc)) < 2:
            return "other"
        mid_context = str(trinuc)[0]  # first char is -1 position
        if mid_context == "T":
            return "TC"
        elif mid_context == "C":
            return "CC"
        else:
            return "other"

    labeled = labeled.copy()
    labeled["trinuc_group"] = labeled["trinuc"].apply(get_trinuc_group)

    for grp in ["TC", "CC", "other"]:
        sub = labeled[labeled["trinuc_group"] == grp]
        path = sub[sub["label"] == "Pathogenic"][score_col].dropna()
        ben = sub[sub["label"] == "Benign"][score_col].dropna()
        if len(path) < 10 or len(ben) < 10:
            print(f"  {grp}: too few variants (path={len(path)}, ben={len(ben)})")
            continue

        u_stat, u_pval = stats.mannwhitneyu(path, ben, alternative="greater")

        # OR at p75
        all_scores = sub[score_col].dropna()
        thresh = np.percentile(all_scores, 75)
        pa = (path >= thresh).sum()
        pb = (path < thresh).sum()
        ba = (ben >= thresh).sum()
        bb = (ben < thresh).sum()
        if ba > 0 and pb > 0 and bb > 0:
            OR = (pa / pb) / (ba / bb)
            fisher_or, fisher_p = stats.fisher_exact([[pa, pb], [ba, bb]], alternative="greater")
        else:
            OR = float("nan")
            fisher_or, fisher_p = float("nan"), float("nan")

        trinuc_results[grp] = {
            "n_path": len(path), "n_ben": len(ben),
            "mean_path": float(path.mean()), "mean_ben": float(ben.mean()),
            "MW_p": float(u_pval), "OR_p75": float(OR),
            "fisher_OR_p75": float(fisher_or), "fisher_p_p75": float(fisher_p),
        }
        print(f"  {grp}: n_path={len(path):,}, n_ben={len(ben):,}  "
              f"mean_path={path.mean():.4f}, mean_ben={ben.mean():.4f}  "
              f"MW p={u_pval:.2e}  OR@p75={OR:.3f} (Fisher p={fisher_p:.2e})")

    return trinuc_results


def within_gene_analysis(labeled, score_col):
    """For genes with both pathogenic and benign, compare within-gene scores."""
    print(f"\n{'='*60}")
    print(f"Within-gene analysis ({score_col})")
    print(f"{'='*60}")

    # Get genes with both labels
    gene_label_counts = labeled.groupby(["gene", "label"]).size().unstack(fill_value=0)
    both_genes = gene_label_counts[
        (gene_label_counts.get("Pathogenic", 0) >= 3)
        & (gene_label_counts.get("Benign", 0) >= 3)
    ].index.tolist()
    print(f"Genes with >=3 pathogenic AND >=3 benign: {len(both_genes):,}")

    if len(both_genes) < 10:
        print("Too few genes for within-gene analysis")
        return {}

    # For each gene: is mean(path) > mean(ben)?
    gene_diffs = []
    gene_wins = 0
    for gene in both_genes:
        gsub = labeled[labeled["gene"] == gene]
        gpath = gsub[gsub["label"] == "Pathogenic"][score_col]
        gben = gsub[gsub["label"] == "Benign"][score_col]
        diff = gpath.mean() - gben.mean()
        gene_diffs.append(diff)
        if diff > 0:
            gene_wins += 1

    gene_diffs = np.array(gene_diffs)
    frac_higher = gene_wins / len(gene_diffs)
    t_stat, t_pval = stats.ttest_1samp(gene_diffs, 0, alternative="greater")
    sign_test_p = stats.binom_test(gene_wins, len(gene_diffs), 0.5, alternative="greater") if hasattr(stats, 'binom_test') else stats.binomtest(gene_wins, len(gene_diffs), 0.5, alternative="greater").pvalue

    results = {
        "n_genes": len(both_genes),
        "frac_path_higher": float(frac_higher),
        "mean_diff": float(gene_diffs.mean()),
        "median_diff": float(np.median(gene_diffs)),
        "ttest_pval": float(t_pval),
        "sign_test_pval": float(sign_test_p),
    }
    print(f"Genes where mean(path) > mean(ben): {gene_wins}/{len(both_genes)} ({100*frac_higher:.1f}%)")
    print(f"Mean within-gene diff: {gene_diffs.mean():.4f}")
    print(f"Median within-gene diff: {np.median(gene_diffs):.4f}")
    print(f"One-sample t-test (diff > 0): t={t_stat:.3f}, p={t_pval:.2e}")
    print(f"Sign test: p={sign_test_p:.2e}")

    return results


def compare_with_direct_scoring(labeled):
    """Compare exome-map scores with direct ClinVar XGB scoring."""
    print(f"\n{'='*60}")
    print("Comparison: exome-map score_full vs direct p_edited_gb")
    print(f"{'='*60}")

    # Both should be present
    both = labeled.dropna(subset=["score_full", "p_edited_gb"])
    print(f"Variants with both scores: {len(both):,}")

    if len(both) < 100:
        print("Too few variants for comparison")
        return {}

    # Correlation
    r, p = stats.pearsonr(both["score_full"], both["p_edited_gb"])
    rho, rho_p = stats.spearmanr(both["score_full"], both["p_edited_gb"])
    print(f"Pearson r = {r:.4f} (p={p:.2e})")
    print(f"Spearman rho = {rho:.4f} (p={rho_p:.2e})")

    # Enrichment comparison
    path = both[both["label"] == "Pathogenic"]
    ben = both[both["label"] == "Benign"]
    for col, name in [("score_full", "exome_map_XGB"), ("p_edited_gb", "direct_XGB")]:
        u, up = stats.mannwhitneyu(path[col], ben[col], alternative="greater")
        thresh = np.percentile(both[col], 75)
        pa = (path[col] >= thresh).sum()
        pb = (path[col] < thresh).sum()
        ba = (ben[col] >= thresh).sum()
        bb = (ben[col] < thresh).sum()
        if ba > 0 and pb > 0 and bb > 0:
            OR = (pa / pb) / (ba / bb)
        else:
            OR = float("nan")
        print(f"  {name}: MW p={up:.2e}, OR@p75={OR:.3f}")

    return {
        "n_both": len(both),
        "pearson_r": float(r), "pearson_p": float(p),
        "spearman_rho": float(rho), "spearman_p": float(rho_p),
    }


def main():
    print("=" * 70)
    print("Exome-wide ClinVar Enrichment using XGB 40d Editability Map")
    print("=" * 70)

    # Load data
    exome = load_exome_map()
    cv = load_clinvar()

    # Liftover
    cv = liftover_clinvar(cv)

    # Match
    merged = match_clinvar_to_exome(cv, exome)

    # Save matched data
    merged.to_csv(OUT_DIR / "clinvar_exome_matched.csv.gz", index=False, compression="gzip")
    print(f"\nSaved matched data: {OUT_DIR / 'clinvar_exome_matched.csv.gz'}")

    # Classify
    labeled = classify_significance(merged)

    # ── Main enrichment: score_full ──
    full_results = compute_enrichment(labeled, "score_full", "XGB_Full_ExomeMap", OUT_DIR)

    # ── Motif-only enrichment ──
    motif_results = compute_enrichment(labeled, "score_motif", "MotifOnly_ExomeMap", OUT_DIR)

    # ── Trinucleotide stratification ──
    trinuc_results = trinucleotide_stratification(labeled, "score_full")

    # ── Within-gene analysis ──
    gene_results = within_gene_analysis(labeled, "score_full")

    # ── Compare with direct ClinVar scoring ──
    comparison = compare_with_direct_scoring(labeled)

    # ── Also run enrichment on direct p_edited_gb for matched subset ──
    direct_results = compute_enrichment(labeled, "p_edited_gb", "Direct_XGB_matched_subset", OUT_DIR)

    # ── Save all results ──
    all_results = {
        "exome_map_full": full_results,
        "exome_map_motif": motif_results,
        "trinucleotide_stratification": trinuc_results,
        "within_gene": gene_results,
        "direct_scoring_comparison": comparison,
        "direct_xgb_on_matched": direct_results,
    }

    with open(OUT_DIR / "enrichment_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results: {OUT_DIR / 'enrichment_results.json'}")

    # ── Summary table ──
    print(f"\n{'='*70}")
    print("SUMMARY TABLE")
    print(f"{'='*70}")
    print(f"{'Method':<30} {'MW p':>12} {'OR@p50':>8} {'OR@p75':>8} {'OR@p90':>8} {'OR@p95':>8}")
    print("-" * 78)
    for tag, res in [
        ("XGB Full (exome map)", full_results),
        ("Motif Only (exome map)", motif_results),
        ("Direct XGB (matched)", direct_results),
    ]:
        print(f"{tag:<30} {res['mannwhitney_pval']:>12.2e} "
              f"{res['OR_p50']:>8.3f} {res['OR_p75']:>8.3f} "
              f"{res['OR_p90']:>8.3f} {res['OR_p95']:>8.3f}")

    print(f"\nDone. All outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
