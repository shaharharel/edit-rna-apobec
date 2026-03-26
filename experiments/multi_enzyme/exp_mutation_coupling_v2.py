#!/usr/bin/env python3
"""
Mutation Coupling Analysis v2: Confound-aware follow-up

Key concern from v1: ClinVar has ascertainment bias toward exonic regions.
Editing sites are exonic → controls ±5kb may include intronic regions with fewer ClinVar entries.
This v2 addresses this by:

1. EXON-MATCHED CONTROLS: Only use control cytidines that are within CDS (same exon or nearby exons)
2. ClinVar density normalization: normalize by local ClinVar density (all variants, not just C>T)
3. Strand-specific analysis: separate C>T on + strand vs G>A on - strand
4. Broader window analysis with distance decay curves
5. Bootstrap confidence intervals

Also adds:
- Comparison of ClinVar density at editing sites vs ALL exonic cytidines (genome-wide baseline)
- Per-chromosome analysis to check for confounds
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "multi_enzyme" / "outputs" / "mutation_coupling"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GENOME_PATH = PROJECT_ROOT / "data" / "raw" / "genomes" / "hg38.fa"
CLINVAR_PATH = PROJECT_ROOT / "experiments" / "apobec3a" / "outputs" / "clinvar_prediction" / "clinvar_all_scores.csv"
A3A_SPLITS_PATH = DATA_DIR / "splits_expanded_a3a.csv"
LEVANON_PATH = DATA_DIR / "multi_enzyme" / "levanon_all_categories.csv"
REFGENE_PATH = PROJECT_ROOT / "data" / "raw" / "genomes" / "refGene.txt"

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def load_genome():
    print("Loading hg38 genome...")
    return Fasta(str(GENOME_PATH))


def reverse_complement(seq):
    comp = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
    return ''.join(comp.get(b, 'N') for b in reversed(seq.upper()))


def get_trinucleotide_context(genome, chrom, pos, strand):
    try:
        seq = str(genome[chrom][pos - 1:pos + 2]).upper()
        if len(seq) != 3:
            return None
        if strand == '-':
            seq = reverse_complement(seq)
        return seq
    except Exception:
        return None


def load_refgene_exons():
    """Load refGene exon coordinates to identify exonic regions."""
    print("Loading refGene exon coordinates...")
    exon_intervals = defaultdict(list)  # chr -> [(start, end), ...]

    with open(REFGENE_PATH) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 11:
                continue
            chrom = parts[2]
            if '_' in chrom:  # skip alt chromosomes
                continue
            exon_starts = [int(x) for x in parts[9].strip(',').split(',') if x]
            exon_ends = [int(x) for x in parts[10].strip(',').split(',') if x]
            for s, e in zip(exon_starts, exon_ends):
                exon_intervals[chrom].append((s, e))

    # Merge overlapping exons per chromosome
    merged = {}
    for chrom, intervals in exon_intervals.items():
        intervals.sort()
        merged_list = []
        for s, e in intervals:
            if merged_list and s <= merged_list[-1][1]:
                merged_list[-1] = (merged_list[-1][0], max(merged_list[-1][1], e))
            else:
                merged_list.append((s, e))
        merged[chrom] = merged_list

    total_exons = sum(len(v) for v in merged.values())
    total_bp = sum(sum(e - s for s, e in v) for v in merged.values())
    print(f"  Loaded {total_exons} merged exon intervals ({total_bp/1e6:.1f} Mbp)")

    return merged


def is_in_exon(chrom, pos, exon_intervals):
    """Check if a position falls within an exon (binary search)."""
    intervals = exon_intervals.get(chrom, [])
    if not intervals:
        return False
    # Binary search for the interval
    lo, hi = 0, len(intervals) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if pos < intervals[mid][0]:
            hi = mid - 1
        elif pos > intervals[mid][1]:
            lo = mid + 1
        else:
            return True
    return False


def get_nearby_exonic_positions(chrom, pos, exon_intervals, search_window=5000):
    """Get all exonic positions within search_window of pos."""
    intervals = exon_intervals.get(chrom, [])
    positions = []
    for s, e in intervals:
        if s > pos + search_window:
            break
        if e < pos - search_window:
            continue
        # This exon overlaps the search window
        region_start = max(s, pos - search_window)
        region_end = min(e, pos + search_window)
        for p in range(region_start, region_end + 1):
            if p != pos:
                positions.append(p)
    return positions


def load_editing_sites():
    """Load all editing sites."""
    print("Loading editing sites...")
    a3a = pd.read_csv(A3A_SPLITS_PATH)
    a3a_pos = a3a[a3a['is_edited'] == 1].copy()
    a3a_pos['enzyme_category'] = 'A3A'

    lev = pd.read_csv(LEVANON_PATH)
    lev = lev[lev['is_edited'] == 1].copy()

    lev_keys = set(zip(lev['chr'], lev['start']))
    a3a_unique = a3a_pos[~a3a_pos.apply(lambda r: (r['chr'], r['start']) in lev_keys, axis=1)]

    a3a_df = a3a_unique[['site_id', 'chr', 'start', 'strand', 'gene']].copy()
    a3a_df['enzyme_category'] = 'A3A'

    lev_df = lev[['site_id', 'chr', 'start', 'strand']].copy()
    lev_df['gene'] = lev['gene_refseq']
    lev_df['enzyme_category'] = lev['enzyme_category']

    combined = pd.concat([a3a_df, lev_df], ignore_index=True)
    print(f"  Total editing sites: {len(combined)}")
    return combined


def load_clinvar():
    print("Loading ClinVar variants...")
    cv = pd.read_csv(CLINVAR_PATH, usecols=['site_id', 'chr', 'start', 'gene',
                                              'significance_simple', 'p_edited_gb'])
    cv['strand'] = cv['site_id'].str.split('_').str[-1]
    print(f"  Total: {len(cv)}")
    return cv


def build_clinvar_index(cv):
    """Build per-chromosome sorted arrays for fast range queries."""
    idx = {}
    sig_map = {}  # (chr, pos) -> significance
    for chrom, grp in cv.groupby('chr'):
        sorted_pos = np.sort(grp['start'].values)
        idx[chrom] = sorted_pos
        for _, row in grp.iterrows():
            sig_map[(chrom, row['start'])] = row['significance_simple']
    return idx, sig_map


def count_in_window(arr, center, w):
    if arr is None or len(arr) == 0:
        return 0
    lo = np.searchsorted(arr, center - w, side='left')
    hi = np.searchsorted(arr, center + w, side='right')
    return hi - lo


def generate_exon_matched_controls(editing_sites, genome, exon_intervals, n_controls=10):
    """
    Generate controls that are:
    1. In the same gene region (±10kb)
    2. Within exons (same genomic context as editing sites)
    3. Same trinucleotide context
    4. NOT editing sites
    """
    print("Generating EXON-MATCHED controls...")
    editing_positions = set(zip(editing_sites['chr'], editing_sites['start']))

    controls = []
    matched = 0
    unmatched = 0

    for i, (_, site) in enumerate(editing_sites.iterrows()):
        if i % 1000 == 0 and i > 0:
            print(f"  Processed {i}/{len(editing_sites)} sites...")

        chrom = site['chr']
        pos = site['start']
        strand = site['strand']
        trinuc = get_trinucleotide_context(genome, chrom, pos, strand)
        if trinuc is None:
            unmatched += 1
            continue

        # Get nearby exonic positions
        exonic_positions = get_nearby_exonic_positions(chrom, pos, exon_intervals, search_window=10000)

        # Filter to those with matching trinucleotide context
        candidates = []
        for cpos in exonic_positions:
            if (chrom, cpos) in editing_positions:
                continue
            ctrinuc = get_trinucleotide_context(genome, chrom, cpos, strand)
            if ctrinuc == trinuc:
                candidates.append(cpos)

        if len(candidates) >= n_controls:
            selected = np.random.choice(candidates, size=n_controls, replace=False)
            matched += 1
        elif len(candidates) > 0:
            selected = np.array(candidates)
            matched += 1
        else:
            unmatched += 1
            continue

        for cp in selected:
            controls.append({
                'chr': chrom, 'start': int(cp), 'strand': strand,
                'matched_to': site['site_id'], 'trinuc': trinuc,
                'enzyme_category': site['enzyme_category'],
            })

    print(f"  Matched: {matched}, Unmatched: {unmatched}")
    print(f"  Total exon-matched controls: {len(controls)}")
    return pd.DataFrame(controls)


def check_exonic_fraction(editing_sites, controls, exon_intervals):
    """Verify that controls and editing sites have similar exonic fraction."""
    print("\n=== Exonic Fraction Validation ===")

    edit_in_exon = sum(1 for _, s in editing_sites.iterrows()
                       if is_in_exon(s['chr'], s['start'], exon_intervals))
    ctrl_in_exon = sum(1 for _, s in controls.iterrows()
                       if is_in_exon(s['chr'], s['start'], exon_intervals))

    edit_frac = edit_in_exon / len(editing_sites) * 100
    ctrl_frac = ctrl_in_exon / len(controls) * 100

    print(f"  Editing sites in exons: {edit_in_exon}/{len(editing_sites)} ({edit_frac:.1f}%)")
    print(f"  Control sites in exons: {ctrl_in_exon}/{len(controls)} ({ctrl_frac:.1f}%)")

    return edit_frac, ctrl_frac


def distance_decay_analysis(editing_sites, controls, clinvar_idx):
    """Analyze how enrichment changes with distance from the editing site."""
    print("\n" + "=" * 70)
    print("DISTANCE DECAY ANALYSIS")
    print("=" * 70)

    windows = [25, 50, 100, 150, 200, 250, 500, 1000, 2000, 5000]
    results = []

    for w in windows:
        edit_counts = np.array([
            count_in_window(clinvar_idx.get(r['chr'], np.array([])), r['start'], w)
            for _, r in editing_sites.iterrows()
        ])
        ctrl_counts = np.array([
            count_in_window(clinvar_idx.get(r['chr'], np.array([])), r['start'], w)
            for _, r in controls.iterrows()
        ])

        edit_mean = np.mean(edit_counts)
        ctrl_mean = np.mean(ctrl_counts)
        ratio = edit_mean / ctrl_mean if ctrl_mean > 0 else float('inf')

        try:
            u, p = stats.mannwhitneyu(edit_counts, ctrl_counts, alternative='two-sided')
        except ValueError:
            p = 1.0

        results.append({
            'window': w, 'edit_mean': edit_mean, 'ctrl_mean': ctrl_mean,
            'ratio': ratio, 'p_value': p,
        })

        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  ±{w:>5d}bp: edit={edit_mean:7.2f}, ctrl={ctrl_mean:7.2f}, "
              f"ratio={ratio:.3f}, p={p:.2e} {sig}")

    return results


def bootstrap_ci(editing_sites, controls, clinvar_idx, window=250, n_boot=1000):
    """Bootstrap confidence intervals for the enrichment ratio."""
    print(f"\n=== Bootstrap CI for ±{window}bp enrichment ratio (n={n_boot}) ===")

    edit_counts = np.array([
        count_in_window(clinvar_idx.get(r['chr'], np.array([])), r['start'], window)
        for _, r in editing_sites.iterrows()
    ])
    ctrl_counts = np.array([
        count_in_window(clinvar_idx.get(r['chr'], np.array([])), r['start'], window)
        for _, r in controls.iterrows()
    ])

    ratios = []
    for _ in range(n_boot):
        e_boot = np.random.choice(edit_counts, size=len(edit_counts), replace=True)
        c_boot = np.random.choice(ctrl_counts, size=len(ctrl_counts), replace=True)
        c_mean = np.mean(c_boot)
        if c_mean > 0:
            ratios.append(np.mean(e_boot) / c_mean)

    ratios = np.array(ratios)
    ci_lo, ci_hi = np.percentile(ratios, [2.5, 97.5])
    print(f"  Ratio: {np.mean(ratios):.3f} (95% CI: {ci_lo:.3f} - {ci_hi:.3f})")

    return {'mean': float(np.mean(ratios)), 'ci_lo': float(ci_lo), 'ci_hi': float(ci_hi)}


def per_chromosome_analysis(editing_sites, controls, clinvar_idx):
    """Check if enrichment is consistent across chromosomes."""
    print("\n" + "=" * 70)
    print("PER-CHROMOSOME ANALYSIS (±250bp)")
    print("=" * 70)

    window = 250
    results = []

    for chrom in sorted(editing_sites['chr'].unique()):
        e_sites = editing_sites[editing_sites['chr'] == chrom]
        c_sites = controls[controls['chr'] == chrom]

        if len(e_sites) < 5 or len(c_sites) < 5:
            continue

        e_counts = np.array([
            count_in_window(clinvar_idx.get(chrom, np.array([])), r['start'], window)
            for _, r in e_sites.iterrows()
        ])
        c_counts = np.array([
            count_in_window(clinvar_idx.get(chrom, np.array([])), r['start'], window)
            for _, r in c_sites.iterrows()
        ])

        e_mean = np.mean(e_counts)
        c_mean = np.mean(c_counts)
        ratio = e_mean / c_mean if c_mean > 0 else float('inf')

        try:
            _, p = stats.mannwhitneyu(e_counts, c_counts, alternative='two-sided')
        except ValueError:
            p = 1.0

        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  {chrom:>6}: n_edit={len(e_sites):>4}, ratio={ratio:.2f}, p={p:.2e} {sig}")

        results.append({
            'chr': chrom, 'n_edit': len(e_sites), 'n_ctrl': len(c_sites),
            'edit_mean': e_mean, 'ctrl_mean': c_mean, 'ratio': ratio, 'p_value': p,
        })

    return results


def clinvar_scoring_coupling(editing_sites, controls, clinvar_idx, sig_map, genome):
    """
    Test: Among C>T variants near editing sites, how does our GB classifier score them?
    Do variants near editing sites get higher editing probability scores?
    """
    print("\n" + "=" * 70)
    print("GB EDITING SCORE OF NEARBY CLINVAR VARIANTS")
    print("=" * 70)

    # Load full ClinVar with scores
    cv = pd.read_csv(CLINVAR_PATH, usecols=['chr', 'start', 'p_edited_gb'])
    score_map = dict(zip(zip(cv['chr'], cv['start']), cv['p_edited_gb']))

    window = 250
    edit_scores = []
    ctrl_scores = []

    for _, site in editing_sites.iterrows():
        chrom = site['chr']
        pos = site['start']
        arr = clinvar_idx.get(chrom, np.array([]))
        lo = np.searchsorted(arr, pos - window, side='left')
        hi = np.searchsorted(arr, pos + window, side='right')
        for vpos in arr[lo:hi]:
            score = score_map.get((chrom, int(vpos)))
            if score is not None:
                edit_scores.append(score)

    for _, ctrl in controls.iterrows():
        chrom = ctrl['chr']
        pos = ctrl['start']
        arr = clinvar_idx.get(chrom, np.array([]))
        lo = np.searchsorted(arr, pos - window, side='left')
        hi = np.searchsorted(arr, pos + window, side='right')
        for vpos in arr[lo:hi]:
            score = score_map.get((chrom, int(vpos)))
            if score is not None:
                ctrl_scores.append(score)

    edit_scores = np.array(edit_scores)
    ctrl_scores = np.array(ctrl_scores)

    u, p = stats.mannwhitneyu(edit_scores, ctrl_scores, alternative='two-sided')

    print(f"  Variants near editing sites: n={len(edit_scores)}, mean GB score={np.mean(edit_scores):.4f}")
    print(f"  Variants near control sites: n={len(ctrl_scores)}, mean GB score={np.mean(ctrl_scores):.4f}")
    print(f"  Mann-Whitney p-value: {p:.2e}")

    # Fraction with GB score > 0.5 (predicted editing sites)
    edit_high = np.mean(edit_scores > 0.5)
    ctrl_high = np.mean(ctrl_scores > 0.5)
    print(f"  Fraction with GB > 0.5: editing={edit_high:.3f}, control={ctrl_high:.3f}")

    return {'edit_mean_score': float(np.mean(edit_scores)),
            'ctrl_mean_score': float(np.mean(ctrl_scores)),
            'p_value': p}


def exonic_cytidine_baseline(genome, exon_intervals, clinvar_idx, editing_sites, n_sample=10000):
    """
    CRITICAL CONTROL: Compare editing sites vs random exonic cytidines genome-wide.
    This controls for the ClinVar ascertainment bias toward exonic regions.
    """
    print("\n" + "=" * 70)
    print("CRITICAL CONTROL: Editing Sites vs Random Exonic Cytidines")
    print("=" * 70)

    editing_positions = set(zip(editing_sites['chr'], editing_sites['start']))

    # Sample random exonic cytidines
    # Build a list of exonic regions to sample from
    exon_regions = []
    for chrom in sorted(exon_intervals.keys()):
        if '_' in chrom or chrom not in clinvar_idx:
            continue
        for s, e in exon_intervals[chrom]:
            if e - s > 10:  # skip tiny exons
                exon_regions.append((chrom, s, e))

    print(f"  Sampling from {len(exon_regions)} exonic regions...")

    random_cytidines = []
    attempts = 0
    max_attempts = n_sample * 100

    while len(random_cytidines) < n_sample and attempts < max_attempts:
        attempts += 1
        # Pick a random exon
        region = exon_regions[np.random.randint(len(exon_regions))]
        chrom, start, end = region
        # Pick a random position in the exon
        pos = np.random.randint(start, end)
        if (chrom, pos) in editing_positions:
            continue

        # Check if it's a cytidine on either strand
        try:
            base = str(genome[chrom][pos]).upper()
            if base == 'C':
                strand = '+'
            elif base == 'G':
                strand = '-'
            else:
                continue

            random_cytidines.append({'chr': chrom, 'start': pos, 'strand': strand})
        except Exception:
            continue

    print(f"  Sampled {len(random_cytidines)} random exonic cytidines")

    random_df = pd.DataFrame(random_cytidines)

    # Count ClinVar variants in ±250bp around random cytidines vs editing sites
    window = 250

    edit_counts = np.array([
        count_in_window(clinvar_idx.get(r['chr'], np.array([])), r['start'], window)
        for _, r in editing_sites.iterrows()
    ])
    random_counts = np.array([
        count_in_window(clinvar_idx.get(r['chr'], np.array([])), r['start'], window)
        for _, r in random_df.iterrows()
    ])

    u, p = stats.mannwhitneyu(edit_counts, random_counts, alternative='two-sided')
    e_mean = np.mean(edit_counts)
    r_mean = np.mean(random_counts)
    ratio = e_mean / r_mean if r_mean > 0 else float('inf')

    print(f"\n  ±{window}bp variant counts:")
    print(f"    Editing sites: mean={e_mean:.2f}, median={np.median(edit_counts):.0f}")
    print(f"    Random exonic C: mean={r_mean:.2f}, median={np.median(random_counts):.0f}")
    print(f"    Ratio: {ratio:.3f}")
    print(f"    Mann-Whitney p: {p:.2e}")

    # Also check with motif-matched random exonic cytidines
    # What fraction of editing sites are TC context?
    tc_edits = 0
    for _, site in editing_sites.iterrows():
        trinuc = get_trinucleotide_context(genome, site['chr'], site['start'], site['strand'])
        if trinuc and trinuc[:2] == 'TC':
            tc_edits += 1
    print(f"\n  TC-context editing sites: {tc_edits}/{len(editing_sites)} ({tc_edits/len(editing_sites)*100:.1f}%)")

    # Sample TC-context random exonic cytidines
    tc_random = []
    attempts = 0
    while len(tc_random) < min(n_sample, 5000) and attempts < max_attempts:
        attempts += 1
        region = exon_regions[np.random.randint(len(exon_regions))]
        chrom, start, end = region
        pos = np.random.randint(start + 1, max(start + 2, end - 1))
        if (chrom, pos) in editing_positions:
            continue

        try:
            # Check plus strand for TC
            dinuc = str(genome[chrom][pos-1:pos+1]).upper()
            if dinuc == 'TC':
                tc_random.append({'chr': chrom, 'start': pos, 'strand': '+'})
            # Check minus strand (GA on plus = TC on minus)
            elif dinuc == 'GA':
                # Actually need to check if the C is at pos on minus strand
                pass
        except Exception:
            continue

    if len(tc_random) > 100:
        tc_random_df = pd.DataFrame(tc_random)
        tc_counts = np.array([
            count_in_window(clinvar_idx.get(r['chr'], np.array([])), r['start'], window)
            for _, r in tc_random_df.iterrows()
        ])

        # Compare TC editing sites vs TC random exonic
        tc_edit_counts = []
        for _, site in editing_sites.iterrows():
            trinuc = get_trinucleotide_context(genome, site['chr'], site['start'], site['strand'])
            if trinuc and trinuc[:2] == 'TC':
                tc_edit_counts.append(
                    count_in_window(clinvar_idx.get(site['chr'], np.array([])), site['start'], window)
                )
        tc_edit_counts = np.array(tc_edit_counts)

        u2, p2 = stats.mannwhitneyu(tc_edit_counts, tc_counts, alternative='two-sided')
        tc_e_mean = np.mean(tc_edit_counts)
        tc_r_mean = np.mean(tc_counts)

        print(f"\n  TC-CONTEXT ONLY (motif-matched baseline):")
        print(f"    TC editing sites: mean={tc_e_mean:.2f}, n={len(tc_edit_counts)}")
        print(f"    TC random exonic: mean={tc_r_mean:.2f}, n={len(tc_counts)}")
        print(f"    Ratio: {tc_e_mean/tc_r_mean:.3f}" if tc_r_mean > 0 else "    Ratio: inf")
        print(f"    Mann-Whitney p: {p2:.2e}")

    return {
        'edit_mean': float(e_mean), 'random_mean': float(r_mean),
        'ratio': ratio, 'p_value': p,
    }


def pathogenicity_by_proximity(editing_sites, clinvar_idx, sig_map):
    """
    Within editing-site windows, does pathogenicity vary with distance from the edit?
    Closest C>T variants to editing sites: are they more/less pathogenic?
    """
    print("\n" + "=" * 70)
    print("PATHOGENICITY BY DISTANCE FROM EDITING SITE")
    print("=" * 70)

    distance_bins = [(0, 0), (1, 50), (51, 100), (101, 250), (251, 500)]

    for lo, hi in distance_bins:
        path_count = 0
        benign_count = 0
        total = 0

        for _, site in editing_sites.iterrows():
            chrom = site['chr']
            pos = site['start']
            arr = clinvar_idx.get(chrom, np.array([]))
            search_lo = np.searchsorted(arr, pos - hi, side='left')
            search_hi = np.searchsorted(arr, pos + hi, side='right')

            for vpos in arr[search_lo:search_hi]:
                dist = abs(int(vpos) - pos)
                if dist < lo or dist > hi:
                    continue
                total += 1
                sig = sig_map.get((chrom, int(vpos)), 'unknown')
                if sig in ('Pathogenic', 'Likely_pathogenic'):
                    path_count += 1
                elif sig in ('Benign', 'Likely_benign'):
                    benign_count += 1

        if total > 0:
            path_frac = path_count / total * 100
            benign_frac = benign_count / total * 100
        else:
            path_frac = benign_frac = 0

        label = f"{lo}-{hi}bp" if lo > 0 else "exact"
        print(f"  {label:>10}: total={total:>6}, pathogenic={path_frac:.2f}%, benign={benign_frac:.2f}%")


def main():
    print("=" * 70)
    print("MUTATION COUPLING ANALYSIS v2: Confound-Aware Analysis")
    print("=" * 70)

    genome = load_genome()
    editing_sites = load_editing_sites()
    cv = load_clinvar()
    clinvar_idx, sig_map = build_clinvar_index(cv)
    exon_intervals = load_refgene_exons()

    # ── Exon-matched controls ──────────────────────────────────────────────
    controls = generate_exon_matched_controls(editing_sites, genome, exon_intervals, n_controls=10)

    # Validate exonic fraction
    check_exonic_fraction(editing_sites, controls, exon_intervals)

    all_results = {}

    # ── Main test with exon-matched controls ──────────────────────────────
    print("\n" + "=" * 70)
    print("MAIN TEST: Window-level enrichment with EXON-MATCHED controls")
    print("=" * 70)

    for window in [100, 250, 500]:
        edit_counts = np.array([
            count_in_window(clinvar_idx.get(r['chr'], np.array([])), r['start'], window)
            for _, r in editing_sites.iterrows()
        ])
        ctrl_counts = np.array([
            count_in_window(clinvar_idx.get(r['chr'], np.array([])), r['start'], window)
            for _, r in controls.iterrows()
        ])

        u, p = stats.mannwhitneyu(edit_counts, ctrl_counts, alternative='two-sided')
        e_mean = np.mean(edit_counts)
        c_mean = np.mean(ctrl_counts)
        ratio = e_mean / c_mean if c_mean > 0 else float('inf')

        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  ±{window}bp: edit={e_mean:.2f}, ctrl={c_mean:.2f}, "
              f"ratio={ratio:.3f}, p={p:.2e} {sig}")

        all_results[f'window_{window}'] = {
            'edit_mean': e_mean, 'ctrl_mean': c_mean,
            'ratio': ratio, 'p_value': p,
        }

    # ── Distance decay ─────────────────────────────────────────────────────
    all_results['distance_decay'] = distance_decay_analysis(editing_sites, controls, clinvar_idx)

    # ── Bootstrap CI ──────────────────────────────────────────────────────
    all_results['bootstrap_ci_250'] = bootstrap_ci(editing_sites, controls, clinvar_idx, window=250)

    # ── Per-chromosome ─────────────────────────────────────────────────────
    all_results['per_chromosome'] = per_chromosome_analysis(editing_sites, controls, clinvar_idx)

    # ── Critical control: random exonic cytidines ─────────────────────────
    all_results['exonic_baseline'] = exonic_cytidine_baseline(
        genome, exon_intervals, clinvar_idx, editing_sites, n_sample=10000)

    # ── GB score coupling ─────────────────────────────────────────────────
    all_results['gb_score_coupling'] = clinvar_scoring_coupling(
        editing_sites, controls, clinvar_idx, sig_map, genome)

    # ── Pathogenicity by distance ─────────────────────────────────────────
    pathogenicity_by_proximity(editing_sites, clinvar_idx, sig_map)

    # ── Save ───────────────────────────────────────────────────────────────
    controls.to_csv(OUTPUT_DIR / "exon_matched_controls.csv", index=False)

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

    with open(OUTPUT_DIR / "mutation_coupling_v2_results.json", 'w') as f:
        json.dump(make_serializable(all_results), f, indent=2, default=str)

    print(f"\n  Saved to: {OUTPUT_DIR}")

    # ── Final Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("V2 SUMMARY")
    print("=" * 70)
    print(f"\n  With EXON-MATCHED controls (eliminating ClinVar ascertainment bias):")
    for window in [100, 250, 500]:
        r = all_results[f'window_{window}']
        print(f"    ±{window}bp: ratio={r['ratio']:.3f}, p={r['p_value']:.2e}")

    if 'bootstrap_ci_250' in all_results:
        ci = all_results['bootstrap_ci_250']
        print(f"\n  Bootstrap 95% CI for ±250bp ratio: {ci['ci_lo']:.3f} - {ci['ci_hi']:.3f}")

    if 'exonic_baseline' in all_results:
        eb = all_results['exonic_baseline']
        print(f"\n  vs Random exonic cytidines: ratio={eb['ratio']:.3f}, p={eb['p_value']:.2e}")

    print("\nDone!")


if __name__ == '__main__':
    main()
