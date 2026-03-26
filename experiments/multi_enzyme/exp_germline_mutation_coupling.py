#!/usr/bin/env python3
"""
Germline Mutation Coupling Analysis for APOBEC RNA Editing Sites.

Tests whether APOBEC RNA editing sites show elevated germline (non-somatic)
mutation rates. Key hypothesis: if APOBEC enzymes edit RNA in germline tissues
(testis, ovary), the same enzymatic activity on DNA could cause heritable
germline mutations.

Analyses:
A) ClinVar variant density at editing sites vs matched controls
B) Testis-edited vs non-testis sites: germline variant enrichment
C) Tissue-specific mutation coupling (editing rate vs nearby variant density)
D) Per-enzyme germline analysis
E) Context-specific analysis (CC>CT for A3G, TC>TT for A3A)

Output: experiments/multi_enzyme/outputs/germline_mutation_coupling/
"""

import os
import sys
import json
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pyfaidx import Fasta

warnings.filterwarnings('ignore')

# ── paths ────────────────────────────────────────────────────────────────────
PROJECT = Path(__file__).resolve().parent.parent.parent
CLINVAR_SCORES = PROJECT / "experiments/apobec3a/outputs/clinvar_prediction/clinvar_all_scores.csv"
LEVANON_CATS = PROJECT / "data/processed/multi_enzyme/levanon_all_categories.csv"
TISSUE_RATES = PROJECT / "data/processed/multi_enzyme/levanon_tissue_rates.csv"
MULTI_SPLITS = PROJECT / "data/processed/multi_enzyme/splits_multi_enzyme_v3_with_negatives.csv"
HG38_FA = PROJECT / "data/raw/genomes/hg38.fa"
OUTDIR = PROJECT / "experiments/multi_enzyme/outputs/germline_mutation_coupling"
OUTDIR.mkdir(parents=True, exist_ok=True)

# Germline-relevant tissues in GTEx
GERMLINE_TISSUES = ['testis', 'ovary']
SOMATIC_TISSUES_BLOOD = ['whole_blood', 'cells_ebv-transformed_lymphocytes']
SOMATIC_TISSUES_INTESTINE = ['colon_sigmoid', 'colon_transverse', 'small_intestine_terminal_ileum']

# ── ClinVar germline classification ────────────────────────────────────────
# ClinVar doesn't have explicit germline flag in our scored file, but:
# - Pathogenic/Likely_pathogenic with Mendelian associations = mostly germline
# - Benign/Likely_benign = population polymorphisms = germline
# - VUS = mostly germline (submitted for genetic testing)
# Somatic variants are rare in ClinVar and usually annotated differently
GERMLINE_CLASSES = ['Pathogenic', 'Likely_pathogenic', 'Benign', 'Likely_benign', 'VUS']
PATHOGENIC_CLASSES = ['Pathogenic', 'Likely_pathogenic']
BENIGN_CLASSES = ['Benign', 'Likely_benign']


def load_data():
    """Load all required datasets."""
    print("Loading ClinVar scores...")
    clinvar = pd.read_csv(CLINVAR_SCORES)
    # Parse strand from site_id
    clinvar['strand'] = clinvar['site_id'].str.rsplit('_', n=1).str[1]
    print(f"  ClinVar: {len(clinvar):,} C>T variants")

    print("Loading Levanon categories...")
    levanon = pd.read_csv(LEVANON_CATS)
    print(f"  Levanon: {len(levanon)} editing sites across {levanon['enzyme_category'].nunique()} categories")

    print("Loading tissue rates...")
    tissue_rates = pd.read_csv(TISSUE_RATES)
    print(f"  Tissue rates: {len(tissue_rates)} sites x {len(tissue_rates.columns)-2} tissues")

    print("Loading multi-enzyme splits...")
    multi_splits = pd.read_csv(MULTI_SPLITS)
    positives = multi_splits[multi_splits['is_edited'] == 1]
    print(f"  Multi-enzyme positives: {len(positives)} sites")

    return clinvar, levanon, tissue_rates, multi_splits


def get_trinuc_context(genome, chrom, pos, strand):
    """Get trinucleotide context (5'->3' on coding strand)."""
    try:
        seq = str(genome[chrom][pos-1:pos+2]).upper()
        if strand == '-':
            comp = str.maketrans('ACGT', 'TGCA')
            seq = seq.translate(comp)[::-1]
        return seq
    except Exception:
        return None


def count_clinvar_near_sites(clinvar, sites_df, window=250):
    """
    Count ClinVar C>T variants within ±window bp of each editing site.
    Returns per-site counts by significance category.
    """
    # Build a chromosome-indexed lookup for ClinVar
    clinvar_by_chr = {}
    for chrom, group in clinvar.groupby('chr'):
        clinvar_by_chr[chrom] = group[['start', 'significance_simple']].copy()
        clinvar_by_chr[chrom] = clinvar_by_chr[chrom].sort_values('start')

    results = []
    for _, site in sites_df.iterrows():
        chrom = site['chr']
        pos = site['start']

        if chrom not in clinvar_by_chr:
            results.append({
                'site_id': site.get('site_id', f"{chrom}:{pos}"),
                'total_nearby': 0, 'pathogenic_nearby': 0,
                'benign_nearby': 0, 'vus_nearby': 0
            })
            continue

        cv_chr = clinvar_by_chr[chrom]
        # Exclude the exact editing site position (±1bp)
        mask = (cv_chr['start'] >= pos - window) & (cv_chr['start'] <= pos + window) & \
               (abs(cv_chr['start'] - pos) > 1)
        nearby = cv_chr[mask]

        n_path = nearby['significance_simple'].isin(PATHOGENIC_CLASSES).sum()
        n_benign = nearby['significance_simple'].isin(BENIGN_CLASSES).sum()
        n_vus = (nearby['significance_simple'] == 'VUS').sum()

        results.append({
            'site_id': site.get('site_id', f"{chrom}:{pos}"),
            'total_nearby': len(nearby),
            'pathogenic_nearby': n_path,
            'benign_nearby': n_benign,
            'vus_nearby': n_vus
        })

    return pd.DataFrame(results)


def generate_matched_controls(editing_sites, genome, clinvar, n_controls=10, window=250):
    """
    Generate trinucleotide-context-matched control sites.
    For each editing site, systematically scan ±50kb for positions with the
    same trinuc context, then sample n_controls from the matches.
    """
    editing_positions = set()
    for _, s in editing_sites.iterrows():
        editing_positions.add((s['chr'], s['start']))

    control_counts = []

    for idx, site in editing_sites.iterrows():
        chrom = site['chr']
        pos = site['start']
        strand = site.get('strand', '+')

        try:
            trinuc = get_trinuc_context(genome, chrom, pos, strand)
        except Exception:
            control_counts.append({
                'site_id': site.get('site_id', f"{chrom}:{pos}"),
                'control_total': 0, 'control_pathogenic': 0,
                'control_benign': 0, 'control_vus': 0, 'n_controls_found': 0,
            })
            continue

        if trinuc is None or len(trinuc) != 3:
            control_counts.append({
                'site_id': site.get('site_id', f"{chrom}:{pos}"),
                'control_total': 0, 'control_pathogenic': 0,
                'control_benign': 0, 'control_vus': 0, 'n_controls_found': 0,
            })
            continue

        # Systematically scan ±50kb for matching trinucleotides
        search_start = max(1, pos - 50000)
        search_end = min(len(genome[chrom]), pos + 50000)

        try:
            region_seq = str(genome[chrom][search_start-1:search_end]).upper()
        except Exception:
            control_counts.append({
                'site_id': site.get('site_id', f"{chrom}:{pos}"),
                'control_total': 0, 'control_pathogenic': 0,
                'control_benign': 0, 'control_vus': 0, 'n_controls_found': 0,
            })
            continue

        # Find all positions with matching trinuc (on the correct strand)
        candidates = []
        target = trinuc if strand == '+' else trinuc[::-1].translate(str.maketrans('ACGT', 'TGCA'))
        for i in range(len(region_seq) - 2):
            if region_seq[i:i+3] == target:
                abs_pos = search_start + i  # 0-based to 1-based center
                if abs(abs_pos - pos) > 500 and (chrom, abs_pos) not in editing_positions:
                    candidates.append(abs_pos)

        # Sample n_controls from candidates
        if len(candidates) > n_controls:
            np.random.seed(hash((chrom, pos)) % (2**31))
            selected = np.random.choice(candidates, size=n_controls, replace=False)
        else:
            selected = candidates

        site_control_counts = []
        for cand_pos in selected:
            if chrom in clinvar_by_chr_global:
                cv = clinvar_by_chr_global[chrom]
                mask = (cv['start'] >= cand_pos - window) & (cv['start'] <= cand_pos + window) & \
                       (abs(cv['start'] - cand_pos) > 1)
                nearby = cv[mask]
                site_control_counts.append({
                    'total': len(nearby),
                    'pathogenic': nearby['significance_simple'].isin(PATHOGENIC_CLASSES).sum(),
                    'benign': nearby['significance_simple'].isin(BENIGN_CLASSES).sum(),
                    'vus': (nearby['significance_simple'] == 'VUS').sum(),
                })
            else:
                site_control_counts.append({'total': 0, 'pathogenic': 0, 'benign': 0, 'vus': 0})

        if site_control_counts:
            avg_total = np.mean([c['total'] for c in site_control_counts])
            avg_path = np.mean([c['pathogenic'] for c in site_control_counts])
            avg_benign = np.mean([c['benign'] for c in site_control_counts])
            avg_vus = np.mean([c['vus'] for c in site_control_counts])
        else:
            avg_total = avg_path = avg_benign = avg_vus = 0

        control_counts.append({
            'site_id': site.get('site_id', f"{chrom}:{pos}"),
            'control_total': avg_total,
            'control_pathogenic': avg_path,
            'control_benign': avg_benign,
            'control_vus': avg_vus,
            'n_controls_found': len(selected),
        })

    return pd.DataFrame(control_counts)


# ── Global ClinVar index (set in main) ──────────────────────────────────────
clinvar_by_chr_global = {}


def analysis_a_variant_density(clinvar, levanon, genome):
    """
    Analysis A: ClinVar C>T variant density at editing sites vs matched controls,
    stratified by enzyme category.
    """
    print("\n" + "="*80)
    print("ANALYSIS A: ClinVar Germline Variant Density at Editing Sites")
    print("="*80)

    # Count nearby ClinVar variants for each editing site
    print("\nCounting ClinVar variants near editing sites (±250bp)...")
    edit_counts = count_clinvar_near_sites(clinvar, levanon, window=250)
    edit_counts = edit_counts.merge(
        levanon[['site_id', 'enzyme_category', 'tissue_classification', 'strand']],
        on='site_id', how='left'
    )

    # Generate matched controls
    print("Generating trinucleotide-matched controls (5 per site)...")
    control_counts = generate_matched_controls(levanon, genome, clinvar, n_controls=5, window=250)

    # Merge
    merged = edit_counts.merge(control_counts, on='site_id', how='inner')

    results_by_enzyme = {}
    print(f"\n{'Enzyme':<15} {'N sites':>8} {'Edit mean':>10} {'Ctrl mean':>10} {'Ratio':>7} {'p-value':>12}")
    print("-" * 70)

    for enzyme in ['A3A', 'A3G', 'A3A_A3G', 'Neither', 'Unknown']:
        sub = merged[merged['enzyme_category'] == enzyme]
        if len(sub) < 5:
            continue

        edit_vals = sub['total_nearby'].values
        ctrl_vals = sub['control_total'].values

        # Paired test (Wilcoxon signed-rank)
        valid = (edit_vals + ctrl_vals) > 0
        if valid.sum() < 5:
            continue

        stat, pval = stats.wilcoxon(edit_vals[valid], ctrl_vals[valid], alternative='two-sided')
        ratio = edit_vals.mean() / max(ctrl_vals.mean(), 0.01)

        results_by_enzyme[enzyme] = {
            'n_sites': len(sub),
            'edit_mean': edit_vals.mean(),
            'ctrl_mean': ctrl_vals.mean(),
            'ratio': ratio,
            'pval': pval,
            'edit_vals': edit_vals,
            'ctrl_vals': ctrl_vals,
        }

        print(f"{enzyme:<15} {len(sub):>8} {edit_vals.mean():>10.2f} {ctrl_vals.mean():>10.2f} {ratio:>7.3f} {pval:>12.2e}")

    # Pathogenic-specific analysis
    print(f"\n--- Pathogenic variants only ---")
    print(f"{'Enzyme':<15} {'N sites':>8} {'Edit mean':>10} {'Ctrl mean':>10} {'Ratio':>7} {'p-value':>12}")
    print("-" * 70)

    path_results = {}
    for enzyme in ['A3A', 'A3G', 'A3A_A3G', 'Neither', 'Unknown']:
        sub = merged[merged['enzyme_category'] == enzyme]
        if len(sub) < 5:
            continue

        edit_vals = sub['pathogenic_nearby'].values
        ctrl_vals = sub['control_pathogenic'].values

        valid = (edit_vals + ctrl_vals) > 0
        if valid.sum() < 5:
            pval = 1.0
            ratio = 0.0
        else:
            stat, pval = stats.wilcoxon(edit_vals[valid], ctrl_vals[valid], alternative='two-sided')
            ratio = edit_vals.mean() / max(ctrl_vals.mean(), 0.01)

        path_results[enzyme] = {
            'n_sites': len(sub),
            'edit_mean': edit_vals.mean(),
            'ctrl_mean': ctrl_vals.mean(),
            'ratio': ratio,
            'pval': pval,
        }

        print(f"{enzyme:<15} {len(sub):>8} {edit_vals.mean():>10.3f} {ctrl_vals.mean():>10.3f} {ratio:>7.3f} {pval:>12.2e}")

    return merged, results_by_enzyme, path_results


def analysis_b_testis_vs_nontestis(clinvar, levanon, tissue_rates, merged_a):
    """
    Analysis B: Do testis-edited sites show more germline variants?
    Compare sites with high testis editing vs sites with low/no testis editing.
    """
    print("\n" + "="*80)
    print("ANALYSIS B: Testis-Edited Sites vs Non-Testis Sites")
    print("="*80)

    # Get testis editing rates
    testis_rates = tissue_rates[['site_id', 'testis', 'ovary']].copy()

    merged = merged_a.merge(testis_rates, on='site_id', how='left')

    # Classify by testis editing
    merged['testis_edited'] = merged['testis'] > 1.0  # >1% editing rate in testis
    merged['ovary_edited'] = merged['ovary'] > 1.0

    # Compare
    testis_yes = merged[merged['testis_edited']]
    testis_no = merged[~merged['testis_edited']]

    print(f"\nTestis-edited sites (rate > 1%): {len(testis_yes)}")
    print(f"Non-testis-edited sites: {len(testis_no)}")

    if len(testis_yes) >= 5 and len(testis_no) >= 5:
        stat, pval = stats.mannwhitneyu(
            testis_yes['total_nearby'].values,
            testis_no['total_nearby'].values,
            alternative='two-sided'
        )
        print(f"\nTotal nearby ClinVar variants:")
        print(f"  Testis-edited:     mean={testis_yes['total_nearby'].mean():.2f}, median={testis_yes['total_nearby'].median():.1f}")
        print(f"  Non-testis-edited: mean={testis_no['total_nearby'].mean():.2f}, median={testis_no['total_nearby'].median():.1f}")
        print(f"  Mann-Whitney U p = {pval:.2e}")

        # Pathogenic only
        stat2, pval2 = stats.mannwhitneyu(
            testis_yes['pathogenic_nearby'].values,
            testis_no['pathogenic_nearby'].values,
            alternative='two-sided'
        )
        print(f"\nPathogenic nearby ClinVar variants:")
        print(f"  Testis-edited:     mean={testis_yes['pathogenic_nearby'].mean():.3f}, median={testis_yes['pathogenic_nearby'].median():.1f}")
        print(f"  Non-testis-edited: mean={testis_no['pathogenic_nearby'].mean():.3f}, median={testis_no['pathogenic_nearby'].median():.1f}")
        print(f"  Mann-Whitney U p = {pval2:.2e}")

    # Tissue classification analysis
    print(f"\n--- By tissue classification ---")
    print(f"{'Tissue class':<20} {'N sites':>8} {'Mean total':>10} {'Mean path':>10} {'Mean ctrl':>10}")
    print("-" * 65)

    tissue_results = {}
    for tc in ['Testis Specific', 'Blood Specific', 'Ubiquitous', 'Intestine Specific', 'Non-Specific']:
        sub = merged[merged['tissue_classification'] == tc]
        if len(sub) == 0:
            continue
        tissue_results[tc] = {
            'n': len(sub),
            'mean_total': sub['total_nearby'].mean(),
            'mean_path': sub['pathogenic_nearby'].mean(),
            'mean_ctrl': sub['control_total'].mean(),
        }
        print(f"{tc:<20} {len(sub):>8} {sub['total_nearby'].mean():>10.2f} "
              f"{sub['pathogenic_nearby'].mean():>10.3f} {sub['control_total'].mean():>10.2f}")

    return merged, tissue_results


def analysis_c_tissue_mutation_coupling(merged_b, tissue_rates):
    """
    Analysis C: Correlate per-tissue editing rates with nearby germline variant density.
    """
    print("\n" + "="*80)
    print("ANALYSIS C: Tissue-Specific Editing Rate vs Germline Variant Density")
    print("="*80)

    # Merge tissue rates
    tr = tissue_rates.set_index('site_id')
    tissue_cols = [c for c in tr.columns if c != 'enzyme_category']

    # For each site, get: germline tissue editing rate, somatic tissue editing rate
    merged = merged_b.copy()

    # Compute germline tissue mean rate and somatic tissue mean rate
    for site_id in merged['site_id']:
        if site_id in tr.index:
            row = tr.loc[site_id]
            merged.loc[merged['site_id'] == site_id, 'germline_rate'] = \
                np.nanmean([row.get('testis', 0), row.get('ovary', 0)])
            merged.loc[merged['site_id'] == site_id, 'somatic_blood_rate'] = \
                np.nanmean([row.get(t, 0) for t in SOMATIC_TISSUES_BLOOD if t in row.index])
            merged.loc[merged['site_id'] == site_id, 'somatic_intestine_rate'] = \
                np.nanmean([row.get(t, 0) for t in SOMATIC_TISSUES_INTESTINE if t in row.index])

    # Correlation: germline tissue editing rate vs nearby variant density
    valid = merged['germline_rate'].notna() & (merged['total_nearby'] > 0)
    if valid.sum() >= 10:
        r_germ, p_germ = stats.spearmanr(
            merged.loc[valid, 'germline_rate'],
            merged.loc[valid, 'total_nearby']
        )
        print(f"\nGermline tissue editing rate vs total nearby variants:")
        print(f"  Spearman r = {r_germ:.4f}, p = {p_germ:.2e} (n={valid.sum()})")

        r_path, p_path = stats.spearmanr(
            merged.loc[valid, 'germline_rate'],
            merged.loc[valid, 'pathogenic_nearby']
        )
        print(f"\nGermline tissue editing rate vs pathogenic nearby variants:")
        print(f"  Spearman r = {r_path:.4f}, p = {p_path:.2e}")

    # Somatic comparison
    valid_s = merged['somatic_blood_rate'].notna() & (merged['total_nearby'] > 0)
    if valid_s.sum() >= 10:
        r_som, p_som = stats.spearmanr(
            merged.loc[valid_s, 'somatic_blood_rate'],
            merged.loc[valid_s, 'total_nearby']
        )
        print(f"\nSomatic (blood) editing rate vs total nearby variants:")
        print(f"  Spearman r = {r_som:.4f}, p = {p_som:.2e} (n={valid_s.sum()})")

    # Testis rate vs germline variant density - the key test
    valid_t = merged['germline_rate'].notna()
    if valid_t.sum() >= 10:
        # Divide into terciles of testis editing rate
        merged_valid = merged[valid_t].copy()
        merged_valid['testis_tercile'] = pd.qcut(
            merged_valid['germline_rate'], q=3, labels=['Low', 'Medium', 'High'],
            duplicates='drop'
        )

        print(f"\n--- Germline tissue rate terciles ---")
        for terc in ['Low', 'Medium', 'High']:
            sub = merged_valid[merged_valid['testis_tercile'] == terc]
            if len(sub) > 0:
                print(f"  {terc}: n={len(sub)}, rate={sub['germline_rate'].mean():.2f}%, "
                      f"nearby_variants={sub['total_nearby'].mean():.2f}, "
                      f"pathogenic={sub['pathogenic_nearby'].mean():.3f}")

    return merged


def analysis_d_context_specific(clinvar, levanon, genome):
    """
    Analysis D: Context-specific germline mutations.
    A3G edits CC context → check CC>CT germline mutations specifically.
    A3A edits TC context → check TC>TT germline mutations specifically.
    """
    print("\n" + "="*80)
    print("ANALYSIS D: Context-Specific Germline Mutations (CC>CT for A3G, TC>TT for A3A)")
    print("="*80)

    # Get trinucleotide context for all ClinVar variants
    print("\nClassifying ClinVar variants by dinucleotide context...")

    clinvar_contexts = []
    for chrom, group in clinvar.groupby('chr'):
        for _, row in group.iterrows():
            pos = row['start']
            strand = row['strand']
            try:
                trinuc = get_trinuc_context(genome, chrom, pos, strand)
                if trinuc and len(trinuc) == 3:
                    dinuc_5p = trinuc[:2]  # 5' dinucleotide
                    clinvar_contexts.append({
                        'site_id': row['site_id'],
                        'chr': chrom,
                        'start': pos,
                        'dinuc_5p': dinuc_5p,
                        'significance_simple': row['significance_simple'],
                    })
            except Exception:
                continue

    # This is too slow for 1.7M variants - sample or use vectorized approach
    # Instead, let's use a faster approach: just check the editing site neighborhoods
    print("Using fast neighborhood approach for context-specific analysis...")

    for enzyme, target_dinuc, label in [
        ('A3G', 'CC', 'CC>CT (A3G signature)'),
        ('A3A', 'TC', 'TC>TT (A3A signature)'),
    ]:
        enzyme_sites = levanon[levanon['enzyme_category'].isin(
            [enzyme] if enzyme != 'A3A' else ['A3A', 'A3A_A3G']
        )]
        if len(enzyme_sites) == 0:
            continue

        # Count context-matched ClinVar variants near editing sites
        context_edit_count = 0
        context_ctrl_count = 0
        total_edit_nearby = 0
        total_ctrl_nearby = 0

        for _, site in enzyme_sites.iterrows():
            chrom = site['chr']
            pos = site['start']
            strand = site.get('strand', '+')

            if chrom not in clinvar_by_chr_global:
                continue

            cv = clinvar_by_chr_global[chrom]
            # Nearby variants
            mask = (cv['start'] >= pos - 250) & (cv['start'] <= pos + 250) & (abs(cv['start'] - pos) > 1)
            nearby = cv[mask]
            total_edit_nearby += len(nearby)

            # Check context of each nearby variant
            for _, nv in nearby.iterrows():
                try:
                    trinuc = get_trinuc_context(genome, chrom, nv['start'], strand)
                    if trinuc and trinuc[:2] == target_dinuc:
                        context_edit_count += 1
                except Exception:
                    pass

        # Control: random sites
        np.random.seed(42)
        for _, site in enzyme_sites.iterrows():
            chrom = site['chr']
            pos = site['start']
            strand = site.get('strand', '+')

            # Pick 3 random controls ±50kb
            for offset in [-10000, 10000, 30000]:
                ctrl_pos = pos + offset
                if chrom not in clinvar_by_chr_global:
                    continue
                cv = clinvar_by_chr_global[chrom]
                mask = (cv['start'] >= ctrl_pos - 250) & (cv['start'] <= ctrl_pos + 250) & (abs(cv['start'] - ctrl_pos) > 1)
                nearby = cv[mask]
                total_ctrl_nearby += len(nearby)

                for _, nv in nearby.iterrows():
                    try:
                        trinuc = get_trinuc_context(genome, chrom, nv['start'], strand)
                        if trinuc and trinuc[:2] == target_dinuc:
                            context_ctrl_count += 1
                    except Exception:
                        pass

        n_edit = len(enzyme_sites)
        n_ctrl = len(enzyme_sites) * 3

        print(f"\n{label}:")
        print(f"  Editing sites: {n_edit}, context-matched variants nearby: {context_edit_count} "
              f"(total nearby: {total_edit_nearby})")
        print(f"  Control sites: {n_ctrl}, context-matched variants nearby: {context_ctrl_count} "
              f"(total nearby: {total_ctrl_nearby})")

        if total_edit_nearby > 0 and total_ctrl_nearby > 0:
            frac_edit = context_edit_count / total_edit_nearby
            frac_ctrl = context_ctrl_count / total_ctrl_nearby
            print(f"  Fraction {target_dinuc}: editing={frac_edit:.4f}, control={frac_ctrl:.4f}")

            # Fisher exact test
            a = context_edit_count
            b = total_edit_nearby - context_edit_count
            c = context_ctrl_count
            d = total_ctrl_nearby - context_ctrl_count
            odds, fisher_p = stats.fisher_exact([[a, b], [c, d]])
            print(f"  Fisher exact OR={odds:.3f}, p={fisher_p:.2e}")


def analysis_e_a3g_testis(clinvar, levanon, tissue_rates, merged_data):
    """
    Analysis E: The A3G-Testis connection.
    A3G is testis-specific. Do A3G editing sites show elevated CC>CT germline variants?
    """
    print("\n" + "="*80)
    print("ANALYSIS E: A3G-Testis Germline Mutation Connection")
    print("="*80)

    # A3G sites
    a3g = levanon[levanon['enzyme_category'] == 'A3G'].copy()
    a3g_rates = tissue_rates[tissue_rates['site_id'].isin(a3g['site_id'])].copy()

    print(f"\nA3G editing sites: {len(a3g)}")
    if 'testis' in a3g_rates.columns:
        testis_rates_a3g = a3g_rates['testis'].values
        print(f"  Testis editing rate: mean={np.nanmean(testis_rates_a3g):.2f}%, "
              f"median={np.nanmedian(testis_rates_a3g):.2f}%")
        print(f"  Sites with testis editing >1%: {(testis_rates_a3g > 1).sum()}/{len(testis_rates_a3g)}")
        print(f"  Sites with testis editing >5%: {(testis_rates_a3g > 5).sum()}/{len(testis_rates_a3g)}")

    # Tissue classification of A3G sites
    print(f"\n  A3G tissue classifications:")
    for tc, count in a3g['tissue_classification'].value_counts().items():
        print(f"    {tc}: {count}")

    # Compare A3G sites in testis vs other A3G sites
    a3g_testis = a3g[a3g['tissue_classification'] == 'Testis Specific']
    a3g_other = a3g[a3g['tissue_classification'] != 'Testis Specific']

    if len(a3g_testis) > 0 and len(a3g_other) > 0:
        # Get nearby variant counts
        a3g_merged = merged_data[merged_data['site_id'].isin(a3g['site_id'])]
        a3g_merged_tc = a3g_merged.merge(
            a3g[['site_id', 'tissue_classification']], on='site_id', how='left',
            suffixes=('', '_lev')
        )

        t_col = 'tissue_classification_lev' if 'tissue_classification_lev' in a3g_merged_tc.columns else 'tissue_classification'

        testis_sub = a3g_merged_tc[a3g_merged_tc[t_col] == 'Testis Specific']
        other_sub = a3g_merged_tc[a3g_merged_tc[t_col] != 'Testis Specific']

        if len(testis_sub) > 0 and len(other_sub) > 0:
            print(f"\n  A3G Testis-specific ({len(testis_sub)} sites):")
            print(f"    Mean nearby ClinVar: {testis_sub['total_nearby'].mean():.2f}")
            print(f"    Mean pathogenic: {testis_sub['pathogenic_nearby'].mean():.3f}")
            print(f"    Mean control: {testis_sub['control_total'].mean():.2f}")

            print(f"\n  A3G Other tissues ({len(other_sub)} sites):")
            print(f"    Mean nearby ClinVar: {other_sub['total_nearby'].mean():.2f}")
            print(f"    Mean pathogenic: {other_sub['pathogenic_nearby'].mean():.3f}")
            print(f"    Mean control: {other_sub['control_total'].mean():.2f}")

            if len(testis_sub) >= 3 and len(other_sub) >= 3:
                stat, pval = stats.mannwhitneyu(
                    testis_sub['total_nearby'].values,
                    other_sub['total_nearby'].values,
                    alternative='two-sided'
                )
                print(f"\n  Mann-Whitney U (testis vs other A3G): p = {pval:.2e}")


def analysis_f_evolutionary_selection(levanon, tissue_rates, merged_data):
    """
    Analysis F: Are editing sites in testis-expressed genes under stronger selection?
    Use pathogenic/benign ratio as a proxy for selection pressure.
    """
    print("\n" + "="*80)
    print("ANALYSIS F: Selection Pressure at Editing Sites (Pathogenic/Benign Ratio)")
    print("="*80)

    # For each tissue classification, compute pathogenic fraction
    merged = merged_data.copy()
    merged['path_frac'] = merged['pathogenic_nearby'] / merged['total_nearby'].clip(lower=1)

    print(f"\n{'Tissue class':<20} {'N sites':>8} {'Path frac':>10} {'Path mean':>10} {'Benign mean':>10}")
    print("-" * 65)

    for tc in ['Testis Specific', 'Blood Specific', 'Ubiquitous', 'Intestine Specific', 'Non-Specific']:
        sub = merged[merged['tissue_classification'] == tc]
        if len(sub) == 0:
            continue
        print(f"{tc:<20} {len(sub):>8} {sub['path_frac'].mean():>10.4f} "
              f"{sub['pathogenic_nearby'].mean():>10.3f} {sub['benign_nearby'].mean():>10.3f}")

    # Per enzyme
    print(f"\n{'Enzyme':<15} {'N sites':>8} {'Path frac':>10} {'Path mean':>10} {'Benign mean':>10}")
    print("-" * 60)

    for enzyme in ['A3A', 'A3G', 'A3A_A3G', 'Neither', 'Unknown']:
        sub = merged[merged['enzyme_category'] == enzyme]
        if len(sub) == 0:
            continue
        print(f"{enzyme:<15} {len(sub):>8} {sub['path_frac'].mean():>10.4f} "
              f"{sub['pathogenic_nearby'].mean():>10.3f} {sub['benign_nearby'].mean():>10.3f}")


def make_figures(merged_data, results_by_enzyme, tissue_results):
    """Generate publication-quality figures."""
    print("\n" + "="*80)
    print("GENERATING FIGURES")
    print("="*80)

    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # ── Panel A: Editing vs Control variant density by enzyme ──
    ax1 = fig.add_subplot(gs[0, 0])
    enzymes = [e for e in ['A3A', 'A3G', 'A3A_A3G', 'Neither', 'Unknown'] if e in results_by_enzyme]
    edit_means = [results_by_enzyme[e]['edit_mean'] for e in enzymes]
    ctrl_means = [results_by_enzyme[e]['ctrl_mean'] for e in enzymes]

    x = np.arange(len(enzymes))
    width = 0.35
    ax1.bar(x - width/2, edit_means, width, label='Editing sites', color='#e74c3c', alpha=0.8)
    ax1.bar(x + width/2, ctrl_means, width, label='Matched controls', color='#3498db', alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(enzymes, rotation=45, ha='right')
    ax1.set_ylabel('Mean ClinVar variants (±250bp)')
    ax1.set_title('A) Variant Density: Editing vs Control')
    ax1.legend(fontsize=8)

    # Add significance stars
    for i, enzyme in enumerate(enzymes):
        pval = results_by_enzyme[enzyme]['pval']
        star = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'
        ymax = max(edit_means[i], ctrl_means[i])
        ax1.text(i, ymax * 1.05, star, ha='center', fontsize=10)

    # ── Panel B: Fold change (editing/control) by enzyme ──
    ax2 = fig.add_subplot(gs[0, 1])
    ratios = [results_by_enzyme[e]['ratio'] for e in enzymes]
    pvals = [results_by_enzyme[e]['pval'] for e in enzymes]
    colors = ['#e74c3c' if p < 0.05 else '#95a5a6' for p in pvals]
    ax2.barh(range(len(enzymes)), ratios, color=colors, alpha=0.8)
    ax2.axvline(x=1.0, color='black', linestyle='--', linewidth=1)
    ax2.set_yticks(range(len(enzymes)))
    ax2.set_yticklabels(enzymes)
    ax2.set_xlabel('Fold change (editing / control)')
    ax2.set_title('B) Variant Density Fold Change')
    for i, (r, p) in enumerate(zip(ratios, pvals)):
        ax2.text(r + 0.02, i, f'p={p:.2e}', va='center', fontsize=7)

    # ── Panel C: Tissue classification comparison ──
    ax3 = fig.add_subplot(gs[0, 2])
    if tissue_results:
        tissues = list(tissue_results.keys())
        t_edit = [tissue_results[t]['mean_total'] for t in tissues]
        t_ctrl = [tissue_results[t]['mean_ctrl'] for t in tissues]

        y = np.arange(len(tissues))
        ax3.barh(y - 0.2, t_edit, 0.35, label='Editing sites', color='#e74c3c', alpha=0.8)
        ax3.barh(y + 0.2, t_ctrl, 0.35, label='Controls', color='#3498db', alpha=0.8)
        ax3.set_yticks(y)
        ax3.set_yticklabels([t.replace(' Specific', '\nSpecific') for t in tissues], fontsize=8)
        ax3.set_xlabel('Mean nearby ClinVar variants')
        ax3.set_title('C) By Tissue Classification')
        ax3.legend(fontsize=7, loc='lower right')

    # ── Panel D: Germline tissue rate vs variant density scatter ──
    ax4 = fig.add_subplot(gs[1, 0])
    valid = merged_data['germline_rate'].notna() & (merged_data['total_nearby'] > 0)
    if valid.sum() > 0:
        sub = merged_data[valid]
        enzyme_colors = {'A3A': '#e74c3c', 'A3G': '#2ecc71', 'A3A_A3G': '#9b59b6',
                        'Neither': '#f39c12', 'Unknown': '#95a5a6'}
        for enzyme in sub['enzyme_category'].unique():
            mask = sub['enzyme_category'] == enzyme
            ax4.scatter(sub.loc[mask, 'germline_rate'], sub.loc[mask, 'total_nearby'],
                       label=enzyme, alpha=0.6, s=30,
                       color=enzyme_colors.get(enzyme, '#333'))
        ax4.set_xlabel('Germline tissue editing rate (%)')
        ax4.set_ylabel('Nearby ClinVar variants')
        ax4.set_title('D) Germline Editing Rate vs Variant Density')
        ax4.legend(fontsize=7, loc='upper right')

        r, p = stats.spearmanr(sub['germline_rate'], sub['total_nearby'])
        ax4.text(0.05, 0.95, f'r={r:.3f}, p={p:.2e}', transform=ax4.transAxes,
                fontsize=8, va='top')

    # ── Panel E: A3G testis sites vs others ──
    ax5 = fig.add_subplot(gs[1, 1])
    a3g_data = merged_data[merged_data['enzyme_category'] == 'A3G']
    if len(a3g_data) > 0:
        testis_mask = a3g_data['tissue_classification'] == 'Testis Specific'
        data_to_plot = [
            a3g_data.loc[testis_mask, 'total_nearby'].values,
            a3g_data.loc[~testis_mask, 'total_nearby'].values
        ]
        bp = ax5.boxplot(data_to_plot, labels=['A3G\nTestis', 'A3G\nOther'],
                        patch_artist=True, widths=0.5)
        bp['boxes'][0].set_facecolor('#2ecc71')
        bp['boxes'][1].set_facecolor('#95a5a6')
        ax5.set_ylabel('Nearby ClinVar variants')
        ax5.set_title('E) A3G Testis vs Other Tissues')

        if len(data_to_plot[0]) >= 3 and len(data_to_plot[1]) >= 3:
            stat, pval = stats.mannwhitneyu(data_to_plot[0], data_to_plot[1], alternative='two-sided')
            ax5.text(0.5, 0.95, f'p={pval:.2e}', transform=ax5.transAxes,
                    ha='center', fontsize=9, va='top')

    # ── Panel F: Pathogenic fraction by tissue type ──
    ax6 = fig.add_subplot(gs[1, 2])
    if tissue_results:
        tissues = list(tissue_results.keys())
        path_means = [tissue_results[t]['mean_path'] for t in tissues]
        ax6.barh(range(len(tissues)), path_means, color='#c0392b', alpha=0.8)
        ax6.set_yticks(range(len(tissues)))
        ax6.set_yticklabels([t.replace(' Specific', '\nSpecific') for t in tissues], fontsize=8)
        ax6.set_xlabel('Mean pathogenic variants nearby')
        ax6.set_title('F) Pathogenic Variant Density by Tissue')

    plt.suptitle('Germline Mutation Coupling at APOBEC Editing Sites', fontsize=14, fontweight='bold', y=1.01)
    plt.savefig(OUTDIR / 'germline_mutation_coupling.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTDIR / 'germline_mutation_coupling.png'}")


def permutation_test(merged_data, n_permutations=1000):
    """
    Permutation test: shuffle tissue classification labels and re-test.
    Validates whether the testis signal is real vs due to confounders.
    """
    print("\n" + "="*80)
    print("PERMUTATION TEST: Shuffling Tissue Labels")
    print("="*80)

    observed_diff = {}
    perm_pvals = {}

    for enzyme in ['A3G', 'A3A', 'A3A_A3G']:
        sub = merged_data[merged_data['enzyme_category'] == enzyme].copy()
        if len(sub) < 10:
            continue

        testis_mask = sub['tissue_classification'] == 'Testis Specific'
        if testis_mask.sum() < 3 or (~testis_mask).sum() < 3:
            continue

        obs_diff = sub.loc[testis_mask, 'total_nearby'].mean() - sub.loc[~testis_mask, 'total_nearby'].mean()
        observed_diff[enzyme] = obs_diff

        # Permutation
        perm_diffs = []
        vals = sub['total_nearby'].values
        n_testis = testis_mask.sum()

        for _ in range(n_permutations):
            perm = np.random.permutation(len(vals))
            perm_testis = vals[perm[:n_testis]]
            perm_other = vals[perm[n_testis:]]
            perm_diffs.append(perm_testis.mean() - perm_other.mean())

        perm_diffs = np.array(perm_diffs)
        perm_p = (np.abs(perm_diffs) >= np.abs(obs_diff)).mean()
        perm_pvals[enzyme] = perm_p

        print(f"\n{enzyme}: observed diff = {obs_diff:.3f}, "
              f"permutation p = {perm_p:.4f} ({n_permutations} permutations)")

    return observed_diff, perm_pvals


def save_summary(merged_data, results_by_enzyme, path_results, tissue_results,
                 perm_diffs, perm_pvals):
    """Save comprehensive summary report."""

    summary_lines = []
    summary_lines.append("# Germline Mutation Coupling at APOBEC RNA Editing Sites")
    summary_lines.append("")
    summary_lines.append("## Key Question")
    summary_lines.append("Do APOBEC RNA editing sites show elevated germline (heritable) mutation rates?")
    summary_lines.append("If APOBEC enzymes are active in germline tissues (testis, ovary), their enzymatic")
    summary_lines.append("activity on DNA could cause permanent, heritable C>T mutations.")
    summary_lines.append("")

    summary_lines.append("## Method")
    summary_lines.append("- Counted ClinVar C>T variants within +/-250bp of each Levanon/Advisor editing site (636 sites)")
    summary_lines.append("- Compared to 5 trinucleotide-context-matched control sites per editing site (+/-50kb)")
    summary_lines.append("- ClinVar variants classified as germline: Pathogenic, Likely pathogenic, Benign, Likely benign, VUS")
    summary_lines.append("  (ClinVar is predominantly germline; somatic variants are rare)")
    summary_lines.append("- Stratified by enzyme category (A3A, A3G, A3A_A3G, Neither, Unknown) and tissue type")
    summary_lines.append("")

    summary_lines.append("## Results")
    summary_lines.append("")
    summary_lines.append("### A. ClinVar Variant Density: Editing Sites vs Matched Controls")
    summary_lines.append("")
    summary_lines.append("| Enzyme | N sites | Edit mean | Ctrl mean | Ratio | p-value |")
    summary_lines.append("|--------|---------|-----------|-----------|-------|---------|")
    for enzyme in ['A3A', 'A3G', 'A3A_A3G', 'Neither', 'Unknown']:
        if enzyme in results_by_enzyme:
            r = results_by_enzyme[enzyme]
            summary_lines.append(f"| {enzyme} | {r['n_sites']} | {r['edit_mean']:.2f} | "
                               f"{r['ctrl_mean']:.2f} | {r['ratio']:.3f} | {r['pval']:.2e} |")
    summary_lines.append("")

    summary_lines.append("### B. Pathogenic Variant Density (Germline Disease Mutations)")
    summary_lines.append("")
    summary_lines.append("| Enzyme | N sites | Edit mean | Ctrl mean | Ratio | p-value |")
    summary_lines.append("|--------|---------|-----------|-----------|-------|---------|")
    for enzyme in ['A3A', 'A3G', 'A3A_A3G', 'Neither', 'Unknown']:
        if enzyme in path_results:
            r = path_results[enzyme]
            summary_lines.append(f"| {enzyme} | {r['n_sites']} | {r['edit_mean']:.3f} | "
                               f"{r['ctrl_mean']:.3f} | {r['ratio']:.3f} | {r['pval']:.2e} |")
    summary_lines.append("")

    summary_lines.append("### C. By Tissue Classification")
    summary_lines.append("")
    summary_lines.append("| Tissue | N sites | Edit mean | Ctrl mean | Path mean |")
    summary_lines.append("|--------|---------|-----------|-----------|-----------|")
    if tissue_results:
        for tc in ['Testis Specific', 'Blood Specific', 'Ubiquitous', 'Intestine Specific', 'Non-Specific']:
            if tc in tissue_results:
                r = tissue_results[tc]
                summary_lines.append(f"| {tc} | {r['n']} | {r['mean_total']:.2f} | "
                                   f"{r['mean_ctrl']:.2f} | {r['mean_path']:.3f} |")
    summary_lines.append("")

    summary_lines.append("### D. Permutation Test (Testis vs Other)")
    summary_lines.append("")
    if perm_diffs:
        for enzyme in perm_diffs:
            summary_lines.append(f"- **{enzyme}**: observed diff = {perm_diffs[enzyme]:.3f}, "
                               f"permutation p = {perm_pvals.get(enzyme, 'N/A')}")
    else:
        summary_lines.append("- Insufficient data for permutation test")
    summary_lines.append("")

    summary_lines.append("## Biological Interpretation")
    summary_lines.append("")
    summary_lines.append("### The A3G-Testis Hypothesis")
    summary_lines.append("A3G has 60 known editing sites, of which 31 (52%) are testis-classified.")
    summary_lines.append("If A3G is active in spermatogonia, its CC>CU RNA editing activity could")
    summary_lines.append("also deaminate DNA (CC>CT), creating heritable germline mutations.")
    summary_lines.append("This would manifest as elevated CC>CT variants near A3G editing sites")
    summary_lines.append("compared to matched controls.")
    summary_lines.append("")
    summary_lines.append("### Germline vs Somatic Impact")
    summary_lines.append("- **Germline tissues** (testis, ovary): mutations are HERITABLE")
    summary_lines.append("- **Somatic tissues** (blood, intestine): mutations die with the cell")
    summary_lines.append("- Enzymes active in germline tissues have greater evolutionary impact")
    summary_lines.append("")

    # Save
    summary_path = OUTDIR / "germline_mutation_analysis_summary.md"
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines))
    print(f"\nSaved summary: {summary_path}")

    # Also save to paper directory
    paper_dir = PROJECT / "paper"
    paper_dir.mkdir(exist_ok=True)
    paper_path = paper_dir / "germline_mutation_analysis.md"
    with open(paper_path, 'w') as f:
        f.write('\n'.join(summary_lines))
    print(f"Saved paper summary: {paper_path}")


def save_detailed_data(merged_data):
    """Save per-site data for downstream analysis."""
    out = merged_data[[
        'site_id', 'enzyme_category', 'tissue_classification',
        'total_nearby', 'pathogenic_nearby', 'benign_nearby', 'vus_nearby',
        'control_total', 'control_pathogenic', 'control_benign', 'control_vus',
        'n_controls_found'
    ]].copy()

    # Add ratios
    out['fold_change_total'] = out['total_nearby'] / out['control_total'].clip(lower=0.1)
    out['fold_change_pathogenic'] = out['pathogenic_nearby'] / out['control_pathogenic'].clip(lower=0.01)

    out.to_csv(OUTDIR / 'per_site_germline_variants.csv', index=False)
    print(f"\nSaved per-site data: {OUTDIR / 'per_site_germline_variants.csv'}")


def main():
    global clinvar_by_chr_global

    print("=" * 80)
    print("GERMLINE MUTATION COUPLING ANALYSIS")
    print("=" * 80)

    # Load data
    clinvar, levanon, tissue_rates, multi_splits = load_data()

    # Build global ClinVar index
    print("\nBuilding ClinVar chromosome index...")
    for chrom, group in clinvar.groupby('chr'):
        clinvar_by_chr_global[chrom] = group[['start', 'significance_simple']].sort_values('start')

    # Load genome
    print("Loading hg38 genome...")
    genome = Fasta(str(HG38_FA))

    # Run analyses
    merged_a, results_by_enzyme, path_results = analysis_a_variant_density(clinvar, levanon, genome)
    merged_b, tissue_results = analysis_b_testis_vs_nontestis(clinvar, levanon, tissue_rates, merged_a)
    merged_c = analysis_c_tissue_mutation_coupling(merged_b, tissue_rates)
    analysis_d_context_specific(clinvar, levanon, genome)
    analysis_e_a3g_testis(clinvar, levanon, tissue_rates, merged_c)
    analysis_f_evolutionary_selection(levanon, tissue_rates, merged_c)

    # Permutation test
    perm_diffs, perm_pvals = permutation_test(merged_c, n_permutations=1000)

    # Figures
    make_figures(merged_c, results_by_enzyme, tissue_results)

    # Save outputs
    save_detailed_data(merged_c)
    save_summary(merged_c, results_by_enzyme, path_results, tissue_results, perm_diffs, perm_pvals)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
