#!/usr/bin/env python3
"""
Mutation Coupling Analysis: Are APOBEC RNA editing sites more prone to DNA mutations?

Hypothesis: Sites where APOBEC enzymes edit RNA are in genomic contexts where APOBEC
is active, leading these regions to accumulate more APOBEC-signature DNA mutations.

Analysis levels:
  A) Site-level: exact editing position overlap with ClinVar C>T variants
  B) Window-level: C>T variant density in ±100/250/500bp windows around editing sites
  C) Gene-level: C>T variant density per kb by editing site count
  D) Pathogenicity coupling: pathogenic fraction among nearby C>T variants
  E) APOBEC-signature analysis: TCA/TCT>TTA/TTT enrichment near editing sites
  F) Per-enzyme comparison: A3A vs A3G vs A3B vs Neither

All tests use motif-matched control sites from the same genes.
"""

import sys
import os
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from scipy import stats
from pyfaidx import Fasta

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "multi_enzyme" / "outputs" / "mutation_coupling"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GENOME_PATH = PROJECT_ROOT / "data" / "raw" / "genomes" / "hg38.fa"
CLINVAR_PATH = PROJECT_ROOT / "experiments" / "apobec3a" / "outputs" / "clinvar_prediction" / "clinvar_all_scores.csv"
A3A_SPLITS_PATH = DATA_DIR / "splits_expanded_a3a.csv"
LEVANON_PATH = DATA_DIR / "multi_enzyme" / "levanon_all_categories.csv"

WINDOW_SIZES = [100, 250, 500]
N_CONTROLS_PER_SITE = 10
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)


def load_genome():
    """Load hg38 reference genome."""
    print("Loading hg38 genome...")
    return Fasta(str(GENOME_PATH))


def reverse_complement(seq):
    """Return reverse complement of a DNA sequence."""
    comp = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
    return ''.join(comp.get(b, 'N') for b in reversed(seq.upper()))


def get_trinucleotide_context(genome, chrom, pos, strand):
    """Get trinucleotide context around a position (on the mRNA strand)."""
    try:
        seq = str(genome[chrom][pos - 1:pos + 2]).upper()
        if len(seq) != 3:
            return None
        if strand == '-':
            seq = reverse_complement(seq)
        return seq
    except Exception:
        return None


def load_editing_sites():
    """Load all editing sites: A3A positives + Levanon multi-enzyme."""
    print("Loading editing sites...")

    # A3A sites (positives only)
    a3a = pd.read_csv(A3A_SPLITS_PATH)
    a3a_pos = a3a[a3a['is_edited'] == 1].copy()
    a3a_pos['enzyme_category'] = 'A3A'
    print(f"  A3A positives: {len(a3a_pos)}")

    # Levanon multi-enzyme sites
    lev = pd.read_csv(LEVANON_PATH)
    lev = lev[lev['is_edited'] == 1].copy()
    print(f"  Levanon sites: {len(lev)} ({lev['enzyme_category'].value_counts().to_dict()})")

    # Combine, using Levanon for enzyme labels
    # Remove A3A sites that are also in Levanon (to avoid double counting)
    lev_keys = set(zip(lev['chr'], lev['start']))
    a3a_unique = a3a_pos[~a3a_pos.apply(lambda r: (r['chr'], r['start']) in lev_keys, axis=1)]
    print(f"  A3A sites not in Levanon: {len(a3a_unique)}")

    # For A3A unique sites, keep enzyme_category='A3A'
    cols = ['site_id', 'chr', 'start', 'strand', 'gene', 'enzyme_category']

    a3a_df = a3a_unique[['site_id', 'chr', 'start', 'strand', 'gene']].copy()
    a3a_df['enzyme_category'] = 'A3A'

    lev_df = lev[['site_id', 'chr', 'start', 'strand']].copy()
    lev_df['gene'] = lev['gene_refseq']
    lev_df['enzyme_category'] = lev['enzyme_category']

    combined = pd.concat([a3a_df, lev_df], ignore_index=True)
    print(f"  Total editing sites: {len(combined)}")
    print(f"  Enzyme distribution: {combined['enzyme_category'].value_counts().to_dict()}")

    return combined


def load_clinvar():
    """Load ClinVar C>T variants as a position-indexed structure."""
    print("Loading ClinVar variants (1.69M)...")
    cv = pd.read_csv(CLINVAR_PATH, usecols=['site_id', 'chr', 'start', 'gene',
                                              'significance_simple', 'p_edited_gb'])

    # Parse strand from site_id
    cv['strand'] = cv['site_id'].str.split('_').str[-1]

    print(f"  Total ClinVar variants: {len(cv)}")
    print(f"  Significance: {cv['significance_simple'].value_counts().head(4).to_dict()}")

    return cv


def build_clinvar_index(cv):
    """Build spatial index: chr -> sorted positions for fast window queries."""
    print("Building ClinVar spatial index...")
    idx = {}
    for chrom, grp in cv.groupby('chr'):
        positions = np.sort(grp['start'].values)
        idx[chrom] = positions

    # Also build a detailed index with significance
    detail_idx = {}
    for chrom, grp in cv.groupby('chr'):
        detail_idx[chrom] = {
            'positions': np.sort(grp['start'].values),
            'significance': grp.set_index('start')['significance_simple'].to_dict(),
            'strand': grp.set_index('start')['strand'].to_dict(),
        }

    print(f"  Indexed {len(idx)} chromosomes")
    return idx, detail_idx


def count_variants_in_window(positions_array, center, window_size):
    """Count how many variants fall within [center-window, center+window]."""
    if positions_array is None or len(positions_array) == 0:
        return 0
    lo = np.searchsorted(positions_array, center - window_size, side='left')
    hi = np.searchsorted(positions_array, center + window_size, side='right')
    return hi - lo


def get_variants_in_window(positions_array, center, window_size):
    """Get variant positions within [center-window, center+window]."""
    if positions_array is None or len(positions_array) == 0:
        return np.array([], dtype=int)
    lo = np.searchsorted(positions_array, center - window_size, side='left')
    hi = np.searchsorted(positions_array, center + window_size, side='right')
    return positions_array[lo:hi]


def classify_trinucleotide_apobec(trinuc):
    """
    Classify if a trinucleotide context represents an APOBEC DNA mutation signature.
    APOBEC SBS2/SBS13: mutations at C in TCA or TCT context.
    """
    if trinuc is None or len(trinuc) != 3:
        return 'unknown'
    if trinuc[1] != 'C':
        return 'non_C_center'
    if trinuc[0] == 'T' and trinuc[2] in ('A', 'T'):
        return 'APOBEC_signature'  # TCA or TCT
    return 'other_C_context'


def generate_matched_controls(editing_sites, genome, n_controls=N_CONTROLS_PER_SITE):
    """
    For each editing site, find motif-matched control sites from the same gene region.
    Controls: same trinucleotide context, same gene, NOT an editing site.
    """
    print(f"Generating {n_controls} matched controls per editing site...")

    # Build set of editing positions for quick lookup
    editing_positions = set(zip(editing_sites['chr'], editing_sites['start']))

    controls = []
    sites_with_controls = 0
    sites_without_controls = 0

    for _, site in editing_sites.iterrows():
        chrom = site['chr']
        pos = site['start']
        strand = site['strand']

        # Get trinucleotide context of the editing site
        trinuc = get_trinucleotide_context(genome, chrom, pos, strand)
        if trinuc is None:
            sites_without_controls += 1
            continue

        # Search in a ±5kb window around the editing site for matching contexts
        search_window = 5000
        try:
            region_start = max(0, pos - search_window)
            region_end = pos + search_window
            region_seq = str(genome[chrom][region_start:region_end]).upper()
        except Exception:
            sites_without_controls += 1
            continue

        # Find all positions in the region with the same trinucleotide context
        candidates = []

        if strand == '+':
            # Look for the trinucleotide on plus strand
            target = trinuc
            for i in range(len(region_seq) - 2):
                if region_seq[i:i+3] == target:
                    candidate_pos = region_start + i + 1  # 1-based center
                    if (chrom, candidate_pos) not in editing_positions and candidate_pos != pos:
                        candidates.append(candidate_pos)
        else:
            # Look for the reverse complement on plus strand
            target_rc = reverse_complement(trinuc)
            for i in range(len(region_seq) - 2):
                if region_seq[i:i+3] == target_rc:
                    candidate_pos = region_start + i + 1  # center position
                    if (chrom, candidate_pos) not in editing_positions and candidate_pos != pos:
                        candidates.append(candidate_pos)

        if len(candidates) >= n_controls:
            selected = np.random.choice(candidates, size=n_controls, replace=False)
            sites_with_controls += 1
        elif len(candidates) > 0:
            selected = np.array(candidates)
            sites_with_controls += 1
        else:
            sites_without_controls += 1
            continue

        for ctrl_pos in selected:
            controls.append({
                'chr': chrom,
                'start': int(ctrl_pos),
                'strand': strand,
                'matched_to': site['site_id'],
                'trinuc': trinuc,
                'enzyme_category': site['enzyme_category'],
                'gene': site.get('gene', ''),
            })

    controls_df = pd.DataFrame(controls)
    print(f"  Sites with controls: {sites_with_controls}")
    print(f"  Sites without controls: {sites_without_controls}")
    print(f"  Total control sites: {len(controls_df)}")

    return controls_df


def analyze_trinucleotide_context(editing_sites, genome):
    """Analyze trinucleotide context distribution of editing sites."""
    print("\n=== Trinucleotide Context Analysis ===")

    trinucs = []
    for _, site in editing_sites.iterrows():
        trinuc = get_trinucleotide_context(genome, site['chr'], site['start'], site['strand'])
        if trinuc:
            trinucs.append({
                'site_id': site['site_id'],
                'trinuc': trinuc,
                'enzyme_category': site['enzyme_category'],
            })

    trinuc_df = pd.DataFrame(trinucs)
    print("\nOverall trinucleotide distribution:")
    print(trinuc_df['trinuc'].value_counts().head(10))

    print("\nPer-enzyme trinucleotide distribution:")
    for enz, grp in trinuc_df.groupby('enzyme_category'):
        top = grp['trinuc'].value_counts().head(5)
        print(f"\n  {enz} (n={len(grp)}):")
        for trinuc, count in top.items():
            pct = count / len(grp) * 100
            print(f"    {trinuc}: {count} ({pct:.1f}%)")

    # APOBEC signature classification
    trinuc_df['apobec_class'] = trinuc_df['trinuc'].apply(classify_trinucleotide_apobec)
    print("\n\nAPOBEC signature classification of editing sites:")
    print(trinuc_df['apobec_class'].value_counts())

    return trinuc_df


def site_level_analysis(editing_sites, clinvar_idx, detail_idx):
    """Test A: Exact site overlap between editing positions and ClinVar C>T variants."""
    print("\n" + "=" * 70)
    print("TEST A: Site-Level Enrichment (exact position overlap)")
    print("=" * 70)

    overlaps = 0
    total = 0
    overlap_details = []

    for _, site in editing_sites.iterrows():
        chrom = site['chr']
        pos = site['start']
        total += 1

        if chrom in clinvar_idx:
            positions = clinvar_idx[chrom]
            idx = np.searchsorted(positions, pos)
            if idx < len(positions) and positions[idx] == pos:
                overlaps += 1
                sig = detail_idx[chrom]['significance'].get(pos, 'unknown')
                overlap_details.append({
                    'site_id': site['site_id'],
                    'chr': chrom,
                    'pos': pos,
                    'enzyme': site['enzyme_category'],
                    'clinvar_significance': sig,
                })

    pct = overlaps / total * 100 if total > 0 else 0
    print(f"\n  Editing sites with exact ClinVar C>T overlap: {overlaps}/{total} ({pct:.2f}%)")

    if overlap_details:
        ol_df = pd.DataFrame(overlap_details)
        print(f"\n  Significance of overlapping variants:")
        print(f"  {ol_df['clinvar_significance'].value_counts().to_dict()}")
        print(f"\n  Per-enzyme overlap:")
        for enz, grp in ol_df.groupby('enzyme'):
            print(f"    {enz}: {len(grp)} overlaps")

    return {'overlaps': overlaps, 'total': total, 'pct': pct, 'details': overlap_details}


def window_level_analysis(editing_sites, controls, clinvar_idx, genome):
    """Test B: C>T variant density in windows around editing sites vs controls."""
    print("\n" + "=" * 70)
    print("TEST B: Window-Level Enrichment")
    print("=" * 70)

    results = {}

    for window_size in WINDOW_SIZES:
        print(f"\n  --- Window: ±{window_size}bp ---")

        # Count variants around editing sites
        edit_counts = []
        for _, site in editing_sites.iterrows():
            chrom = site['chr']
            pos = site['start']
            positions = clinvar_idx.get(chrom, np.array([]))
            count = count_variants_in_window(positions, pos, window_size)
            edit_counts.append(count)

        # Count variants around control sites
        ctrl_counts = []
        for _, ctrl in controls.iterrows():
            chrom = ctrl['chr']
            pos = ctrl['start']
            positions = clinvar_idx.get(chrom, np.array([]))
            count = count_variants_in_window(positions, pos, window_size)
            ctrl_counts.append(count)

        edit_counts = np.array(edit_counts)
        ctrl_counts = np.array(ctrl_counts)

        # Mann-Whitney U test
        u_stat, p_value = stats.mannwhitneyu(edit_counts, ctrl_counts, alternative='two-sided')

        # Effect size (rank-biserial correlation)
        n1, n2 = len(edit_counts), len(ctrl_counts)
        r_rb = 1 - (2 * u_stat) / (n1 * n2)

        # Means and medians
        edit_mean = np.mean(edit_counts)
        ctrl_mean = np.mean(ctrl_counts)
        edit_median = np.median(edit_counts)
        ctrl_median = np.median(ctrl_counts)
        ratio = edit_mean / ctrl_mean if ctrl_mean > 0 else float('inf')

        print(f"    Editing sites: mean={edit_mean:.2f}, median={edit_median:.0f} variants")
        print(f"    Control sites: mean={ctrl_mean:.2f}, median={ctrl_median:.0f} variants")
        print(f"    Ratio (edit/ctrl): {ratio:.3f}")
        print(f"    Mann-Whitney U p-value: {p_value:.2e}")
        print(f"    Rank-biserial r: {r_rb:.4f}")

        # Also compute using only APOBEC-signature contexts
        results[window_size] = {
            'edit_mean': edit_mean, 'ctrl_mean': ctrl_mean,
            'edit_median': float(edit_median), 'ctrl_median': float(ctrl_median),
            'ratio': ratio, 'p_value': p_value, 'r_rb': r_rb,
            'n_edit': len(edit_counts), 'n_ctrl': len(ctrl_counts),
        }

    return results


def apobec_signature_window_analysis(editing_sites, controls, clinvar_idx, detail_idx, genome):
    """
    Test B2: Among C>T variants near editing sites, count APOBEC-signature mutations
    (TCA>TTA or TCT>TTT) vs other C>T mutations.
    """
    print("\n" + "=" * 70)
    print("TEST B2: APOBEC Signature Variants in Windows")
    print("=" * 70)

    window_size = 250  # Use the middle window
    results = {}

    for label, sites_df in [('editing', editing_sites), ('control', controls)]:
        apobec_counts = []
        other_counts = []
        total_counts = []

        for _, site in sites_df.iterrows():
            chrom = site['chr']
            pos = site['start']
            positions_arr = clinvar_idx.get(chrom, np.array([]))
            nearby = get_variants_in_window(positions_arr, pos, window_size)

            n_apobec = 0
            n_other = 0

            for vpos in nearby:
                # Get strand of the variant
                vstrand = detail_idx.get(chrom, {}).get('strand', {}).get(vpos, '+')
                trinuc = get_trinucleotide_context(genome, chrom, int(vpos), vstrand)
                cls = classify_trinucleotide_apobec(trinuc)
                if cls == 'APOBEC_signature':
                    n_apobec += 1
                else:
                    n_other += 1

            apobec_counts.append(n_apobec)
            other_counts.append(n_other)
            total_counts.append(len(nearby))

        results[label] = {
            'apobec': np.array(apobec_counts),
            'other': np.array(other_counts),
            'total': np.array(total_counts),
        }

    # Compare APOBEC-signature variant counts
    for vtype in ['apobec', 'other', 'total']:
        edit_vals = results['editing'][vtype]
        ctrl_vals = results['control'][vtype]

        u_stat, p_val = stats.mannwhitneyu(edit_vals, ctrl_vals, alternative='two-sided')
        edit_mean = np.mean(edit_vals)
        ctrl_mean = np.mean(ctrl_vals)
        ratio = edit_mean / ctrl_mean if ctrl_mean > 0 else float('inf')

        print(f"\n  {vtype.upper()} variants (±{window_size}bp):")
        print(f"    Editing: mean={edit_mean:.3f}, Control: mean={ctrl_mean:.3f}")
        print(f"    Ratio: {ratio:.3f}, p={p_val:.2e}")

    # APOBEC fraction comparison
    edit_frac = np.mean(results['editing']['apobec']) / max(np.mean(results['editing']['total']), 0.001)
    ctrl_frac = np.mean(results['control']['apobec']) / max(np.mean(results['control']['total']), 0.001)
    print(f"\n  APOBEC signature fraction:")
    print(f"    Editing sites: {edit_frac:.3f}")
    print(f"    Control sites: {ctrl_frac:.3f}")

    return results


def gene_level_analysis(editing_sites, cv):
    """Test C: C>T variant density per kb by gene editing-site count."""
    print("\n" + "=" * 70)
    print("TEST C: Gene-Level Enrichment")
    print("=" * 70)

    # Count editing sites per gene
    gene_edit_counts = editing_sites.groupby('gene').size().reset_index(name='n_editing_sites')

    # Count ClinVar variants per gene
    gene_clinvar_counts = cv.groupby('gene').size().reset_index(name='n_clinvar')

    # Merge
    gene_df = gene_clinvar_counts.merge(gene_edit_counts, on='gene', how='left')
    gene_df['n_editing_sites'] = gene_df['n_editing_sites'].fillna(0).astype(int)

    # Bin genes by editing site count
    bins = [0, 1, 5, 20, 1000]
    labels = ['0', '1-5', '6-20', '20+']
    gene_df['edit_bin'] = pd.cut(gene_df['n_editing_sites'], bins=bins, labels=labels, right=True, include_lowest=True)

    # Handle the 0 bin: genes with no editing sites get the '0' label
    gene_df.loc[gene_df['n_editing_sites'] == 0, 'edit_bin'] = '0'

    print("\n  C>T variants per gene by editing site count:")
    for bin_label in ['0', '1-5', '6-20', '20+']:
        grp = gene_df[gene_df['edit_bin'] == bin_label]
        if len(grp) == 0:
            continue
        mean_clinvar = grp['n_clinvar'].mean()
        median_clinvar = grp['n_clinvar'].median()
        print(f"    {bin_label} editing sites: {len(grp)} genes, "
              f"mean={mean_clinvar:.1f}, median={median_clinvar:.0f} C>T variants")

    # Spearman correlation between editing site count and ClinVar count
    genes_with_edits = gene_df[gene_df['n_editing_sites'] > 0]
    if len(genes_with_edits) > 10:
        rho, p = stats.spearmanr(genes_with_edits['n_editing_sites'],
                                  genes_with_edits['n_clinvar'])
        print(f"\n  Spearman correlation (genes with edits): rho={rho:.3f}, p={p:.2e}")

    return gene_df


def pathogenicity_coupling(editing_sites, controls, clinvar_idx, detail_idx):
    """Test D: Are pathogenic variants over-represented near editing sites?"""
    print("\n" + "=" * 70)
    print("TEST D: Pathogenicity Coupling")
    print("=" * 70)

    window_size = 250

    results = {}
    for label, sites_df in [('editing', editing_sites), ('control', controls)]:
        pathogenic = 0
        benign = 0
        vus = 0
        total = 0

        for _, site in sites_df.iterrows():
            chrom = site['chr']
            pos = site['start']
            positions_arr = clinvar_idx.get(chrom, np.array([]))
            nearby = get_variants_in_window(positions_arr, pos, window_size)

            for vpos in nearby:
                sig = detail_idx.get(chrom, {}).get('significance', {}).get(int(vpos), 'unknown')
                total += 1
                if sig in ('Pathogenic', 'Likely_pathogenic'):
                    pathogenic += 1
                elif sig in ('Benign', 'Likely_benign'):
                    benign += 1
                elif sig == 'VUS':
                    vus += 1

        results[label] = {
            'pathogenic': pathogenic, 'benign': benign, 'vus': vus, 'total': total,
            'path_frac': pathogenic / total if total > 0 else 0,
        }

        print(f"\n  {label.upper()} ({window_size}bp window):")
        print(f"    Total nearby variants: {total}")
        print(f"    Pathogenic/Likely_path: {pathogenic} ({pathogenic/total*100:.2f}%)" if total > 0 else "    No variants")
        print(f"    Benign/Likely_benign: {benign} ({benign/total*100:.2f}%)" if total > 0 else "")
        print(f"    VUS: {vus} ({vus/total*100:.2f}%)" if total > 0 else "")

    # Fisher's exact test: pathogenic vs non-pathogenic, editing vs control
    edit_r = results['editing']
    ctrl_r = results['control']

    table = [
        [edit_r['pathogenic'], edit_r['total'] - edit_r['pathogenic']],
        [ctrl_r['pathogenic'], ctrl_r['total'] - ctrl_r['pathogenic']],
    ]

    or_val, p_val = stats.fisher_exact(table)
    print(f"\n  Fisher's exact test (pathogenic enrichment near editing sites):")
    print(f"    Odds ratio: {or_val:.3f}")
    print(f"    p-value: {p_val:.2e}")

    return results


def per_enzyme_analysis(editing_sites, controls, clinvar_idx):
    """Test F: Window-level enrichment broken down by enzyme category."""
    print("\n" + "=" * 70)
    print("TEST F: Per-Enzyme Window-Level Analysis (±250bp)")
    print("=" * 70)

    window_size = 250
    results = {}

    enzymes = editing_sites['enzyme_category'].unique()

    for enz in sorted(enzymes):
        enz_sites = editing_sites[editing_sites['enzyme_category'] == enz]
        enz_ctrls = controls[controls['enzyme_category'] == enz]

        if len(enz_sites) < 10 or len(enz_ctrls) < 10:
            continue

        edit_counts = []
        for _, site in enz_sites.iterrows():
            positions = clinvar_idx.get(site['chr'], np.array([]))
            count = count_variants_in_window(positions, site['start'], window_size)
            edit_counts.append(count)

        ctrl_counts = []
        for _, ctrl in enz_ctrls.iterrows():
            positions = clinvar_idx.get(ctrl['chr'], np.array([]))
            count = count_variants_in_window(positions, ctrl['start'], window_size)
            ctrl_counts.append(count)

        edit_counts = np.array(edit_counts)
        ctrl_counts = np.array(ctrl_counts)

        u_stat, p_val = stats.mannwhitneyu(edit_counts, ctrl_counts, alternative='two-sided')
        edit_mean = np.mean(edit_counts)
        ctrl_mean = np.mean(ctrl_counts)
        ratio = edit_mean / ctrl_mean if ctrl_mean > 0 else float('inf')

        print(f"\n  {enz} (n={len(enz_sites)} sites, n={len(enz_ctrls)} controls):")
        print(f"    Edit mean: {edit_mean:.2f}, Ctrl mean: {ctrl_mean:.2f}")
        print(f"    Ratio: {ratio:.3f}, p={p_val:.2e}")

        results[enz] = {
            'n_sites': len(enz_sites), 'n_controls': len(enz_ctrls),
            'edit_mean': edit_mean, 'ctrl_mean': ctrl_mean,
            'ratio': ratio, 'p_value': p_val,
        }

    return results


def tc_motif_depletion_test(editing_sites, genome):
    """
    Test E (proxy for evolutionary conservation):
    Is the TC/CC motif depleted at editing positions relative to the local genomic background?
    If editing sites are under negative selection, we might see motif depletion.
    """
    print("\n" + "=" * 70)
    print("TEST E: TC/CC Motif Frequency at Editing Sites vs Local Background")
    print("=" * 70)

    window = 1000  # ±1kb

    tc_at_site = 0
    tc_in_background = 0
    total_sites = 0
    total_bg_positions = 0

    for _, site in editing_sites.iterrows():
        chrom = site['chr']
        pos = site['start']
        strand = site['strand']

        # Get trinucleotide at site
        trinuc = get_trinucleotide_context(genome, chrom, pos, strand)
        if trinuc is None:
            continue

        total_sites += 1

        # Check if the site itself is TC
        if trinuc[0] == 'T' and trinuc[1] == 'C':
            tc_at_site += 1

        # Count TC dinucleotides in the local window
        try:
            region_start = max(0, pos - window)
            region_end = pos + window
            region_seq = str(genome[chrom][region_start:region_end]).upper()

            if strand == '-':
                region_seq = reverse_complement(region_seq)

            tc_count = sum(1 for i in range(len(region_seq) - 1)
                          if region_seq[i:i+2] == 'TC')
            total_positions = len(region_seq) - 1

            tc_in_background += tc_count
            total_bg_positions += total_positions
        except Exception:
            pass

    site_tc_frac = tc_at_site / total_sites if total_sites > 0 else 0
    bg_tc_frac = tc_in_background / total_bg_positions if total_bg_positions > 0 else 0
    enrichment = site_tc_frac / bg_tc_frac if bg_tc_frac > 0 else float('inf')

    print(f"\n  TC dinucleotide at editing sites: {tc_at_site}/{total_sites} ({site_tc_frac:.3f})")
    print(f"  TC dinucleotide in local ±{window}bp: {tc_in_background}/{total_bg_positions} ({bg_tc_frac:.4f})")
    print(f"  Enrichment (site vs background): {enrichment:.2f}x")

    # Binomial test: is TC frequency at editing sites different from background?
    if total_sites > 0:
        p_val = stats.binomtest(tc_at_site, total_sites, bg_tc_frac, alternative='two-sided').pvalue
        print(f"  Binomial test p-value: {p_val:.2e}")

    return {
        'site_tc_frac': site_tc_frac, 'bg_tc_frac': bg_tc_frac,
        'enrichment': enrichment, 'n_sites': total_sites,
    }


def permutation_validation(editing_sites, controls, clinvar_idx, n_permutations=1000):
    """Validate: repeat window analysis with permuted labels to confirm signal isn't spurious."""
    print("\n" + "=" * 70)
    print(f"VALIDATION: Permutation Test ({n_permutations} permutations)")
    print("=" * 70)

    window_size = 250

    # Get actual counts
    all_counts = []
    all_labels = []

    for _, site in editing_sites.iterrows():
        positions = clinvar_idx.get(site['chr'], np.array([]))
        count = count_variants_in_window(positions, site['start'], window_size)
        all_counts.append(count)
        all_labels.append(1)

    for _, ctrl in controls.iterrows():
        positions = clinvar_idx.get(ctrl['chr'], np.array([]))
        count = count_variants_in_window(positions, ctrl['start'], window_size)
        all_counts.append(count)
        all_labels.append(0)

    all_counts = np.array(all_counts)
    all_labels = np.array(all_labels)

    # Observed difference in means
    obs_diff = np.mean(all_counts[all_labels == 1]) - np.mean(all_counts[all_labels == 0])

    # Permutation test
    n_edit = np.sum(all_labels == 1)
    perm_diffs = []
    for _ in range(n_permutations):
        perm_labels = np.random.permutation(all_labels)
        perm_diff = np.mean(all_counts[perm_labels == 1]) - np.mean(all_counts[perm_labels == 0])
        perm_diffs.append(perm_diff)

    perm_diffs = np.array(perm_diffs)
    p_perm = np.mean(np.abs(perm_diffs) >= np.abs(obs_diff))

    print(f"\n  Observed mean difference: {obs_diff:.4f}")
    print(f"  Permutation mean diff: {np.mean(perm_diffs):.4f} ± {np.std(perm_diffs):.4f}")
    print(f"  Permutation p-value (two-sided): {p_perm:.4f}")
    print(f"  Z-score: {(obs_diff - np.mean(perm_diffs)) / max(np.std(perm_diffs), 1e-10):.2f}")

    return {
        'obs_diff': obs_diff,
        'perm_mean': float(np.mean(perm_diffs)),
        'perm_std': float(np.std(perm_diffs)),
        'p_perm': p_perm,
    }


def gc_content_control(editing_sites, controls, genome, clinvar_idx):
    """Additional control: check if GC content explains variant density differences."""
    print("\n" + "=" * 70)
    print("CONTROL: GC Content Comparison")
    print("=" * 70)

    window = 500

    def get_gc(chrom, pos):
        try:
            seq = str(genome[chrom][max(0, pos - window):pos + window]).upper()
            gc = (seq.count('G') + seq.count('C')) / len(seq)
            return gc
        except Exception:
            return None

    edit_gc = [get_gc(r['chr'], r['start']) for _, r in editing_sites.iterrows()]
    ctrl_gc = [get_gc(r['chr'], r['start']) for _, r in controls.iterrows()]

    edit_gc = np.array([x for x in edit_gc if x is not None])
    ctrl_gc = np.array([x for x in ctrl_gc if x is not None])

    u_stat, p_val = stats.mannwhitneyu(edit_gc, ctrl_gc, alternative='two-sided')

    print(f"\n  GC content (±{window}bp):")
    print(f"    Editing sites: mean={np.mean(edit_gc):.3f}")
    print(f"    Control sites: mean={np.mean(ctrl_gc):.3f}")
    print(f"    Mann-Whitney p-value: {p_val:.2e}")

    # After GC content check, do GC-stratified variant density analysis
    print(f"\n  GC-stratified variant density (±250bp):")

    # Combine all sites with their GC and variant counts
    all_gc = np.concatenate([edit_gc, ctrl_gc])
    gc_quartiles = np.percentile(all_gc, [25, 50, 75])

    edit_counts_250 = []
    for _, site in editing_sites.iterrows():
        positions = clinvar_idx.get(site['chr'], np.array([]))
        edit_counts_250.append(count_variants_in_window(positions, site['start'], 250))

    ctrl_counts_250 = []
    for _, ctrl in controls.iterrows():
        positions = clinvar_idx.get(ctrl['chr'], np.array([]))
        ctrl_counts_250.append(count_variants_in_window(positions, ctrl['start'], 250))

    edit_counts_250 = np.array(edit_counts_250[:len(edit_gc)])
    ctrl_counts_250 = np.array(ctrl_counts_250[:len(ctrl_gc)])

    for q_lo, q_hi, q_label in [(0, gc_quartiles[0], 'Q1'),
                                  (gc_quartiles[0], gc_quartiles[1], 'Q2'),
                                  (gc_quartiles[1], gc_quartiles[2], 'Q3'),
                                  (gc_quartiles[2], 1.0, 'Q4')]:
        edit_mask = (edit_gc >= q_lo) & (edit_gc < q_hi)
        ctrl_mask = (ctrl_gc >= q_lo) & (ctrl_gc < q_hi)

        if np.sum(edit_mask) < 5 or np.sum(ctrl_mask) < 5:
            continue

        e_mean = np.mean(edit_counts_250[edit_mask])
        c_mean = np.mean(ctrl_counts_250[ctrl_mask])
        ratio = e_mean / c_mean if c_mean > 0 else float('inf')

        try:
            _, p = stats.mannwhitneyu(edit_counts_250[edit_mask], ctrl_counts_250[ctrl_mask], alternative='two-sided')
        except ValueError:
            p = 1.0

        print(f"    {q_label} (GC {q_lo:.2f}-{q_hi:.2f}): edit={e_mean:.2f}, ctrl={c_mean:.2f}, "
              f"ratio={ratio:.3f}, p={p:.2e}, n_edit={np.sum(edit_mask)}, n_ctrl={np.sum(ctrl_mask)}")

    return {'edit_gc_mean': float(np.mean(edit_gc)), 'ctrl_gc_mean': float(np.mean(ctrl_gc)),
            'gc_p_value': p_val}


def clinvar_density_by_editing_rate(editing_sites, clinvar_idx, a3a_splits):
    """
    Test G: Do higher-edited sites have more nearby C>T variants?
    (Dose-response relationship)
    """
    print("\n" + "=" * 70)
    print("TEST G: Variant Density by Editing Rate (Dose-Response)")
    print("=" * 70)

    # Get editing rates for A3A sites
    a3a_pos = a3a_splits[a3a_splits['is_edited'] == 1].copy()
    rate_map = dict(zip(zip(a3a_pos['chr'], a3a_pos['start']), a3a_pos['editing_rate_normalized']))

    window_size = 250
    rows = []

    for _, site in editing_sites.iterrows():
        rate = rate_map.get((site['chr'], site['start']))
        if rate is None or pd.isna(rate) or rate <= 0:
            continue

        positions = clinvar_idx.get(site['chr'], np.array([]))
        count = count_variants_in_window(positions, site['start'], window_size)

        rows.append({'rate': rate, 'variant_count': count, 'log_rate': np.log10(rate + 0.001)})

    if len(rows) < 20:
        print("  Not enough sites with rates for dose-response analysis")
        return {}

    df = pd.DataFrame(rows)

    # Bin by editing rate quartiles
    df['rate_quartile'] = pd.qcut(df['rate'], q=4, labels=['Q1_low', 'Q2', 'Q3', 'Q4_high'])

    print(f"\n  Variant density (±{window_size}bp) by editing rate quartile:")
    for q in ['Q1_low', 'Q2', 'Q3', 'Q4_high']:
        grp = df[df['rate_quartile'] == q]
        print(f"    {q}: mean rate={grp['rate'].mean():.4f}, "
              f"mean variants={grp['variant_count'].mean():.2f}, n={len(grp)}")

    rho, p = stats.spearmanr(df['rate'], df['variant_count'])
    print(f"\n  Spearman correlation (rate vs variant count): rho={rho:.3f}, p={p:.2e}")

    return {'spearman_rho': rho, 'spearman_p': p, 'n_sites': len(df)}


def main():
    print("=" * 70)
    print("MUTATION COUPLING ANALYSIS")
    print("Are APOBEC RNA editing sites enriched for DNA C>T mutations?")
    print("=" * 70)

    # Load data
    genome = load_genome()
    editing_sites = load_editing_sites()
    cv = load_clinvar()
    clinvar_idx, detail_idx = build_clinvar_index(cv)
    a3a_splits = pd.read_csv(A3A_SPLITS_PATH)

    # Trinucleotide context analysis
    trinuc_df = analyze_trinucleotide_context(editing_sites, genome)

    # Generate matched controls
    controls = generate_matched_controls(editing_sites, genome)

    # ── Core Tests ──────────────────────────────────────────────────────────────
    all_results = {}

    # Test A: Site-level overlap
    all_results['site_level'] = site_level_analysis(editing_sites, clinvar_idx, detail_idx)

    # Test B: Window-level enrichment (MAIN TEST)
    all_results['window_level'] = window_level_analysis(editing_sites, controls, clinvar_idx, genome)

    # Test B2: APOBEC signature in windows
    all_results['apobec_signature'] = apobec_signature_window_analysis(
        editing_sites, controls, clinvar_idx, detail_idx, genome)

    # Test C: Gene-level enrichment
    all_results['gene_level'] = gene_level_analysis(editing_sites, cv)

    # Test D: Pathogenicity coupling
    all_results['pathogenicity'] = pathogenicity_coupling(editing_sites, controls, clinvar_idx, detail_idx)

    # Test E: TC motif enrichment
    all_results['motif_enrichment'] = tc_motif_depletion_test(editing_sites, genome)

    # Test F: Per-enzyme analysis
    all_results['per_enzyme'] = per_enzyme_analysis(editing_sites, controls, clinvar_idx)

    # Test G: Dose-response (editing rate vs variant density)
    all_results['dose_response'] = clinvar_density_by_editing_rate(editing_sites, clinvar_idx, a3a_splits)

    # ── Controls & Validation ───────────────────────────────────────────────────

    # GC content control
    all_results['gc_control'] = gc_content_control(editing_sites, controls, genome, clinvar_idx)

    # Permutation validation
    all_results['permutation'] = permutation_validation(editing_sites, controls, clinvar_idx, n_permutations=1000)

    # ── Save Results ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    # Save controls for reproducibility
    controls.to_csv(OUTPUT_DIR / "matched_controls.csv", index=False)
    print(f"  Saved matched controls: {OUTPUT_DIR / 'matched_controls.csv'}")

    # Save summary JSON (convert numpy types)
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return "DataFrame"
        elif isinstance(obj, list):
            return [make_serializable(x) for x in obj]
        return obj

    summary = make_serializable(all_results)
    with open(OUTPUT_DIR / "mutation_coupling_results.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Saved results JSON: {OUTPUT_DIR / 'mutation_coupling_results.json'}")

    # ── Final Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY OF KEY FINDINGS")
    print("=" * 70)

    print(f"\n  A) Site-level overlap: {all_results['site_level']['overlaps']}/{all_results['site_level']['total']} "
          f"editing sites have exact ClinVar C>T overlap ({all_results['site_level']['pct']:.2f}%)")

    print(f"\n  B) Window-level enrichment (MAIN TEST):")
    for ws, res in all_results['window_level'].items():
        sig = "***" if res['p_value'] < 0.001 else "**" if res['p_value'] < 0.01 else "*" if res['p_value'] < 0.05 else "ns"
        print(f"     ±{ws}bp: ratio={res['ratio']:.3f}, p={res['p_value']:.2e} {sig}")

    if 'gc_control' in all_results:
        print(f"\n  GC control: edit GC={all_results['gc_control']['edit_gc_mean']:.3f}, "
              f"ctrl GC={all_results['gc_control']['ctrl_gc_mean']:.3f}")

    if 'permutation' in all_results:
        perm = all_results['permutation']
        print(f"\n  Permutation validation: obs_diff={perm['obs_diff']:.4f}, p_perm={perm['p_perm']:.4f}")

    print(f"\n  Output directory: {OUTPUT_DIR}")
    print("\nDone!")


if __name__ == '__main__':
    main()
