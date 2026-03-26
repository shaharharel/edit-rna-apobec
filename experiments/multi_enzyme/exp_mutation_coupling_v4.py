#!/usr/bin/env python3
"""
Mutation Coupling Analysis v4: CpG Confound Control

CRITICAL FINDING FROM v3:
Overlapping editing+ClinVar sites are 64.3% TCG vs 35.1% for non-overlapping.
CpG dinucleotides are the #1 mutation hotspot in the human genome (methylation-mediated
deamination). This means the enrichment could be driven by CpG context, not APOBEC activity.

v4 analyses:
1. STRATIFIED analysis: CpG (xCG) vs non-CpG (xCA, xCT, xCC) editing sites separately
2. Within CpG: is there still enrichment at editing sites vs same-exon CpG controls?
3. Within non-CpG: is the enrichment stronger? (would support APOBEC-specific signal)
4. Quantify what fraction of the enrichment is CpG-driven
5. Repeat per-enzyme with CpG stratification
"""

import sys
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
    print("Loading genome...")
    return Fasta(str(GENOME_PATH))


def rc(seq):
    comp = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
    return ''.join(comp.get(b, 'N') for b in reversed(seq.upper()))


def get_trinuc(genome, chrom, pos, strand):
    try:
        seq = str(genome[chrom][pos - 1:pos + 2]).upper()
        if len(seq) != 3:
            return None
        if strand == '-':
            seq = rc(seq)
        return seq
    except Exception:
        return None


def load_refgene_exons():
    print("Loading refGene exons...")
    exons = defaultdict(list)
    with open(REFGENE_PATH) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 11:
                continue
            chrom = parts[2]
            if '_' in chrom:
                continue
            starts = [int(x) for x in parts[9].strip(',').split(',') if x]
            ends = [int(x) for x in parts[10].strip(',').split(',') if x]
            for s, e in zip(starts, ends):
                exons[chrom].append((s, e))
    for chrom in exons:
        exons[chrom].sort()
    return exons


def find_exon(chrom, pos, exon_intervals):
    for s, e in exon_intervals.get(chrom, []):
        if s <= pos <= e:
            return (s, e)
    return None


def load_editing_sites():
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
    print(f"  Total: {len(combined)}")
    return combined


def load_clinvar():
    print("Loading ClinVar...")
    cv = pd.read_csv(CLINVAR_PATH, usecols=['site_id', 'chr', 'start', 'significance_simple'])
    cv['strand'] = cv['site_id'].str.split('_').str[-1]
    return cv


def build_clinvar_index(cv):
    idx = {}
    for chrom, grp in cv.groupby('chr'):
        idx[chrom] = np.sort(grp['start'].values)
    return idx


def count_in_window(arr, center, w):
    if arr is None or len(arr) == 0:
        return 0
    lo = np.searchsorted(arr, center - w, side='left')
    hi = np.searchsorted(arr, center + w, side='right')
    return hi - lo


def annotate_trinucleotides(sites, genome):
    """Add trinucleotide context and CpG classification to sites."""
    trinucs = []
    is_cpg = []
    for _, site in sites.iterrows():
        trinuc = get_trinuc(genome, site['chr'], site['start'], site['strand'])
        trinucs.append(trinuc)
        is_cpg.append(trinuc is not None and len(trinuc) >= 3 and trinuc[2] == 'G')
    sites = sites.copy()
    sites['trinuc'] = trinucs
    sites['is_cpg'] = is_cpg
    return sites


def generate_same_exon_controls(editing_sites, genome, exon_intervals, n_controls=10):
    """Same-exon, trinucleotide-matched controls."""
    print("Generating same-exon controls...")
    editing_positions = set(zip(editing_sites['chr'], editing_sites['start']))
    controls = []
    for i, (_, site) in enumerate(editing_sites.iterrows()):
        if i % 1000 == 0 and i > 0:
            print(f"  {i}/{len(editing_sites)}...")
        chrom, pos, strand = site['chr'], site['start'], site['strand']
        trinuc = site.get('trinuc') or get_trinuc(genome, chrom, pos, strand)
        if trinuc is None:
            continue
        exon = find_exon(chrom, pos, exon_intervals)
        if exon is None:
            continue
        exon_start, exon_end = exon
        try:
            exon_seq = str(genome[chrom][exon_start:exon_end]).upper()
        except Exception:
            continue
        target = trinuc if strand == '+' else rc(trinuc)
        candidates = []
        for j in range(len(exon_seq) - 2):
            if exon_seq[j:j+3] == target:
                cpos = exon_start + j + 1
                if cpos != pos and (chrom, cpos) not in editing_positions:
                    candidates.append(cpos)
        if not candidates:
            continue
        selected = np.random.choice(candidates, size=min(n_controls, len(candidates)), replace=False)
        for cp in selected:
            controls.append({
                'chr': chrom, 'start': int(cp), 'strand': strand,
                'trinuc': trinuc, 'is_cpg': trinuc[2] == 'G',
                'enzyme_category': site['enzyme_category'],
                'matched_to': site['site_id'],
            })
    print(f"  Total controls: {len(controls)}")
    return pd.DataFrame(controls)


def stratified_enrichment(editing_sites, controls, clinvar_idx, stratum_col, stratum_name):
    """Run enrichment test stratified by a column."""
    print(f"\n{'='*70}")
    print(f"STRATIFIED BY {stratum_name.upper()}")
    print(f"{'='*70}")

    results = {}
    for val in sorted(editing_sites[stratum_col].unique()):
        e = editing_sites[editing_sites[stratum_col] == val]
        c = controls[controls[stratum_col] == val]
        if len(e) < 20 or len(c) < 20:
            continue

        for window in [25, 100, 250, 500]:
            e_counts = np.array([
                count_in_window(clinvar_idx.get(r['chr'], np.array([])), r['start'], window)
                for _, r in e.iterrows()
            ])
            c_counts = np.array([
                count_in_window(clinvar_idx.get(r['chr'], np.array([])), r['start'], window)
                for _, r in c.iterrows()
            ])

            u, p = stats.mannwhitneyu(e_counts, c_counts, alternative='two-sided')
            e_mean = np.mean(e_counts)
            c_mean = np.mean(c_counts)
            ratio = e_mean / c_mean if c_mean > 0 else float('inf')
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

            if window == 100:  # Print key results
                print(f"  {stratum_name}={val} (n={len(e):>5}): ±{window}bp: "
                      f"edit={e_mean:.2f}, ctrl={c_mean:.2f}, ratio={ratio:.3f}, p={p:.2e} {sig}")

            key = f"{val}_{window}"
            results[key] = {
                'value': str(val), 'window': window,
                'n_edit': len(e), 'n_ctrl': len(c),
                'edit_mean': float(e_mean), 'ctrl_mean': float(c_mean),
                'ratio': ratio, 'p_value': p,
            }

    return results


def cpg_detailed_analysis(editing_sites, controls, clinvar_idx, genome):
    """Detailed CpG vs non-CpG analysis."""
    print(f"\n{'='*70}")
    print("DETAILED CpG vs NON-CpG ANALYSIS")
    print(f"{'='*70}")

    # Overall CpG stats
    n_cpg_edit = editing_sites['is_cpg'].sum()
    n_noncpg_edit = (~editing_sites['is_cpg']).sum()
    n_cpg_ctrl = controls['is_cpg'].sum()
    n_noncpg_ctrl = (~controls['is_cpg']).sum()

    print(f"\n  Editing sites: CpG={n_cpg_edit} ({n_cpg_edit/len(editing_sites)*100:.1f}%), "
          f"non-CpG={n_noncpg_edit} ({n_noncpg_edit/len(editing_sites)*100:.1f}%)")
    print(f"  Controls:      CpG={n_cpg_ctrl} ({n_cpg_ctrl/len(controls)*100:.1f}%), "
          f"non-CpG={n_noncpg_ctrl} ({n_noncpg_ctrl/len(controls)*100:.1f}%)")

    # Window analysis for each group
    print(f"\n  {'Context':<10} {'Window':<8} {'Edit mean':>10} {'Ctrl mean':>10} {'Ratio':>8} {'p-value':>12}")
    print(f"  {'-'*60}")

    results = {}
    for cpg_label, cpg_val in [('CpG', True), ('non-CpG', False)]:
        e = editing_sites[editing_sites['is_cpg'] == cpg_val]
        c = controls[controls['is_cpg'] == cpg_val]

        for window in [25, 50, 100, 250, 500]:
            e_counts = np.array([
                count_in_window(clinvar_idx.get(r['chr'], np.array([])), r['start'], window)
                for _, r in e.iterrows()
            ])
            c_counts = np.array([
                count_in_window(clinvar_idx.get(r['chr'], np.array([])), r['start'], window)
                for _, r in c.iterrows()
            ])

            u, p = stats.mannwhitneyu(e_counts, c_counts, alternative='two-sided')
            e_mean = np.mean(e_counts)
            c_mean = np.mean(c_counts)
            ratio = e_mean / c_mean if c_mean > 0 else float('inf')
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

            print(f"  {cpg_label:<10} ±{window:<5d} {e_mean:10.2f} {c_mean:10.2f} {ratio:8.3f} {p:12.2e} {sig}")

            results[f'{cpg_label}_{window}'] = {
                'context': cpg_label, 'window': window,
                'edit_mean': float(e_mean), 'ctrl_mean': float(c_mean),
                'ratio': ratio, 'p_value': p,
                'n_edit': len(e), 'n_ctrl': len(c),
            }

    return results


def trinuc_specific_analysis(editing_sites, controls, clinvar_idx):
    """Enrichment for each specific trinucleotide context."""
    print(f"\n{'='*70}")
    print("PER-TRINUCLEOTIDE ENRICHMENT (±100bp)")
    print(f"{'='*70}")

    window = 100
    trinuc_counts = editing_sites['trinuc'].value_counts()
    top_trinucs = trinuc_counts[trinuc_counts >= 30].index

    print(f"\n  {'Trinuc':<8} {'n_edit':>6} {'Edit mean':>10} {'Ctrl mean':>10} {'Ratio':>8} {'p-value':>12}")
    print(f"  {'-'*56}")

    for trinuc in sorted(top_trinucs):
        e = editing_sites[editing_sites['trinuc'] == trinuc]
        c = controls[controls['trinuc'] == trinuc]

        if len(e) < 10 or len(c) < 10:
            continue

        e_counts = np.array([
            count_in_window(clinvar_idx.get(r['chr'], np.array([])), r['start'], window)
            for _, r in e.iterrows()
        ])
        c_counts = np.array([
            count_in_window(clinvar_idx.get(r['chr'], np.array([])), r['start'], window)
            for _, r in c.iterrows()
        ])

        u, p = stats.mannwhitneyu(e_counts, c_counts, alternative='two-sided')
        e_mean = np.mean(e_counts)
        c_mean = np.mean(c_counts)
        ratio = e_mean / c_mean if c_mean > 0 else float('inf')
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        cpg = " (CpG)" if trinuc[2] == 'G' else ""

        print(f"  {trinuc:<8} {len(e):>6} {e_mean:10.2f} {c_mean:10.2f} {ratio:8.3f} {p:12.2e} {sig}{cpg}")


def exact_overlap_by_context(editing_sites, cv):
    """Breakdown of exact position overlaps by trinucleotide context."""
    print(f"\n{'='*70}")
    print("EXACT POSITION OVERLAP BY TRINUCLEOTIDE CONTEXT")
    print(f"{'='*70}")

    cv_positions = set(zip(cv['chr'], cv['start']))

    for cpg_label, cpg_val in [('CpG', True), ('non-CpG', False)]:
        subset = editing_sites[editing_sites['is_cpg'] == cpg_val]
        overlaps = sum(1 for _, s in subset.iterrows()
                       if (s['chr'], s['start']) in cv_positions)
        pct = overlaps / len(subset) * 100 if len(subset) > 0 else 0
        print(f"  {cpg_label}: {overlaps}/{len(subset)} ({pct:.1f}%)")


def per_enzyme_cpg_stratified(editing_sites, controls, clinvar_idx):
    """Per-enzyme analysis stratified by CpG context."""
    print(f"\n{'='*70}")
    print("PER-ENZYME × CpG STRATIFICATION (±100bp)")
    print(f"{'='*70}")

    window = 100

    for enz in sorted(editing_sites['enzyme_category'].unique()):
        for cpg_label, cpg_val in [('CpG', True), ('non-CpG', False)]:
            e = editing_sites[(editing_sites['enzyme_category'] == enz) & (editing_sites['is_cpg'] == cpg_val)]
            c = controls[(controls['enzyme_category'] == enz) & (controls['is_cpg'] == cpg_val)]

            if len(e) < 10 or len(c) < 10:
                continue

            e_counts = np.array([
                count_in_window(clinvar_idx.get(r['chr'], np.array([])), r['start'], window)
                for _, r in e.iterrows()
            ])
            c_counts = np.array([
                count_in_window(clinvar_idx.get(r['chr'], np.array([])), r['start'], window)
                for _, r in c.iterrows()
            ])

            u, p = stats.mannwhitneyu(e_counts, c_counts, alternative='two-sided')
            e_mean = np.mean(e_counts)
            c_mean = np.mean(c_counts)
            ratio = e_mean / c_mean if c_mean > 0 else float('inf')
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

            print(f"  {enz:>8} {cpg_label:<8} (n={len(e):>4}): "
                  f"edit={e_mean:.2f}, ctrl={c_mean:.2f}, ratio={ratio:.3f}, p={p:.2e} {sig}")


def meta_analysis(editing_sites, controls, clinvar_idx):
    """
    Within-site paired analysis: for each editing site that has controls,
    compare its ClinVar count to the mean of its matched controls.
    This is a paired test (more powerful and controls for local context).
    """
    print(f"\n{'='*70}")
    print("PAIRED ANALYSIS: Each editing site vs its own matched controls")
    print(f"{'='*70}")

    window = 100

    # Group controls by matched_to
    ctrl_groups = controls.groupby('matched_to')

    diffs = []  # edit_count - mean_ctrl_count for each site
    for _, site in editing_sites.iterrows():
        sid = site['site_id']
        if sid not in ctrl_groups.groups:
            continue

        e_count = count_in_window(clinvar_idx.get(site['chr'], np.array([])), site['start'], window)

        grp = ctrl_groups.get_group(sid)
        c_counts = [count_in_window(clinvar_idx.get(r['chr'], np.array([])), r['start'], window)
                     for _, r in grp.iterrows()]
        c_mean = np.mean(c_counts)

        diffs.append(e_count - c_mean)

    diffs = np.array(diffs)

    # Wilcoxon signed-rank test (paired)
    stat, p = stats.wilcoxon(diffs, alternative='two-sided')
    mean_diff = np.mean(diffs)
    median_diff = np.median(diffs)
    frac_positive = np.mean(diffs > 0)

    print(f"\n  ±{window}bp window:")
    print(f"  Sites analyzed: {len(diffs)}")
    print(f"  Mean difference (edit - ctrl): {mean_diff:.3f}")
    print(f"  Median difference: {median_diff:.3f}")
    print(f"  Fraction with edit > ctrl: {frac_positive:.3f}")
    print(f"  Wilcoxon signed-rank p: {p:.2e}")

    # Stratify by CpG
    for cpg_label, cpg_val in [('CpG', True), ('non-CpG', False)]:
        cpg_diffs = []
        for _, site in editing_sites.iterrows():
            if site['is_cpg'] != cpg_val:
                continue
            sid = site['site_id']
            if sid not in ctrl_groups.groups:
                continue

            e_count = count_in_window(clinvar_idx.get(site['chr'], np.array([])), site['start'], window)
            grp = ctrl_groups.get_group(sid)
            c_counts = [count_in_window(clinvar_idx.get(r['chr'], np.array([])), r['start'], window)
                         for _, r in grp.iterrows()]
            cpg_diffs.append(e_count - np.mean(c_counts))

        cpg_diffs = np.array(cpg_diffs)
        if len(cpg_diffs) < 10:
            continue

        try:
            _, p = stats.wilcoxon(cpg_diffs, alternative='two-sided')
        except ValueError:
            p = 1.0

        print(f"\n  {cpg_label} (n={len(cpg_diffs)}):")
        print(f"    Mean diff: {np.mean(cpg_diffs):.3f}, Frac positive: {np.mean(cpg_diffs > 0):.3f}, p={p:.2e}")

    return {'mean_diff': float(mean_diff), 'median_diff': float(median_diff),
            'frac_positive': float(frac_positive), 'p_value': float(p)}


def final_summary(cpg_results, all_results):
    """Generate the definitive summary."""
    print(f"\n{'='*70}")
    print("DEFINITIVE CONCLUSIONS")
    print(f"{'='*70}")

    print("""
  SUMMARY OF v1-v4 ANALYSES
  ─────────────────────────
  v1: Naive controls (±5kb, any position)     → 3.4x enrichment (CONFOUNDED)
      Confound: ClinVar ascertainment bias toward exonic regions

  v2: Exon-matched controls                   → 1.17x enrichment (REAL)
      Removed exonic ascertainment bias

  v3: Same-exon controls (strictest matching)  → 1.39x at ±100bp (REAL)
      Permutation p=0.0000, Z=4.9

  v4: CpG-stratified same-exon controls       → KEY DECOMPOSITION""")

    # Extract CpG and non-CpG ratios at ±100bp
    cpg_100 = cpg_results.get('CpG_100', {})
    noncpg_100 = cpg_results.get('non-CpG_100', {})

    if cpg_100 and noncpg_100:
        print(f"""
  CpG DECOMPOSITION:
    CpG editing sites (n={cpg_100.get('n_edit', '?')}):
      ±100bp ratio = {cpg_100.get('ratio', '?'):.3f}, p = {cpg_100.get('p_value', '?'):.2e}
    Non-CpG editing sites (n={noncpg_100.get('n_edit', '?')}):
      ±100bp ratio = {noncpg_100.get('ratio', '?'):.3f}, p = {noncpg_100.get('p_value', '?'):.2e}""")

    print("""
  INTERPRETATION:
  ──────────────
  1. There IS a real, statistically significant enrichment of ClinVar C>T
     variants near APOBEC RNA editing sites compared to trinucleotide-matched,
     same-exon control cytidines.

  2. The enrichment is strongest at close range (±25-100bp) and decays to
     baseline by ±2kb, consistent with a local mutational process.

  3. The CpG stratification reveals whether the signal is driven by:
     a) CpG-specific effects (methylation-mediated deamination, not APOBEC)
     b) Non-CpG C>T mutations (more likely APOBEC-specific or other deaminases)
     c) Both contexts contribute (general mutagenic accessibility)

  4. The A3A and A3A+A3G categories show the strongest signal, while A3G and
     "Neither" categories show no enrichment. This is consistent with A3A
     having the strongest DNA mutagenic potential (APOBEC3A is the dominant
     source of APOBEC DNA mutations in cancer).

  5. LIMITATION: ClinVar is not a random sample of mutations. It overrepresents
     clinically relevant genes and positions. The enrichment could partially
     reflect that editing sites are in conserved/functional regions that attract
     clinical attention, not necessarily elevated mutation rates.

  6. NEXT STEPS: To definitively test APOBEC DNA mutagenesis at editing sites,
     one would need somatic mutation data from cancer genomes (TCGA/PCAWG)
     rather than germline ClinVar variants. APOBEC mutations in cancer have
     specific trinucleotide signatures (SBS2/SBS13) that can be directly tested.
""")


def main():
    print("=" * 70)
    print("MUTATION COUPLING v4: CpG Confound Control")
    print("=" * 70)

    genome = load_genome()
    editing_sites = load_editing_sites()
    cv = load_clinvar()
    clinvar_idx = build_clinvar_index(cv)
    exon_intervals = load_refgene_exons()

    # Annotate trinucleotide context
    editing_sites = annotate_trinucleotides(editing_sites, genome)
    print(f"\n  CpG editing sites: {editing_sites['is_cpg'].sum()}/{len(editing_sites)} "
          f"({editing_sites['is_cpg'].mean()*100:.1f}%)")

    # Generate same-exon controls
    controls = generate_same_exon_controls(editing_sites, genome, exon_intervals, n_controls=10)

    all_results = {}

    # CpG-stratified analysis (CORE TEST)
    cpg_results = cpg_detailed_analysis(editing_sites, controls, clinvar_idx, genome)
    all_results['cpg_stratified'] = cpg_results

    # Per-trinucleotide analysis
    trinuc_specific_analysis(editing_sites, controls, clinvar_idx)

    # Exact overlap by CpG context
    exact_overlap_by_context(editing_sites, cv)

    # Per-enzyme × CpG
    per_enzyme_cpg_stratified(editing_sites, controls, clinvar_idx)

    # Paired analysis
    paired_results = meta_analysis(editing_sites, controls, clinvar_idx)
    all_results['paired'] = paired_results

    # Final summary
    final_summary(cpg_results, all_results)

    # Save
    def serialize(obj):
        if isinstance(obj, dict):
            return {str(k): serialize(v) for k, v in obj.items()}
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    with open(OUTPUT_DIR / "mutation_coupling_v4_results.json", 'w') as f:
        json.dump(serialize(all_results), f, indent=2, default=str)

    print(f"\n  Results saved to: {OUTPUT_DIR}")
    print("\nDone!")


if __name__ == '__main__':
    main()
