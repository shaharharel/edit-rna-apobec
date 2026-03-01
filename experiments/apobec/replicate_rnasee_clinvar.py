#!/usr/bin/env python
"""Replicate RNAsee 2024 ClinVar pathogenicity enrichment analysis.

Implements the rules-based stem-loop detector from Van Norden et al. 2024
and tests pathogenicity enrichment among predicted editing sites.

Usage:
    conda run -n quris python experiments/apobec/replicate_rnasee_clinvar.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "clinvar_prediction"


def wc_pair(a, b):
    """Check Watson-Crick (+ wobble GU) base pair."""
    pairs = {('A','U'), ('U','A'), ('G','C'), ('C','G'),
             ('A','T'), ('T','A'), ('G','U'), ('U','G')}
    return (a.upper(), b.upper()) in pairs


def score_stemloop(seq, center, loop_size):
    """Score a stem-loop with C at 3' end of loop.

    Mimics RNAsee scoring.py logic:
    - loop_size 3 or 4 (tri/tetra loop)
    - C is at the rightmost position of the loop
    - Stem extends outward from loop boundaries
    """
    n = len(seq)
    loop_start = center - (loop_size - 1)
    if loop_start < 1 or center + 1 >= n:
        return -1, 0

    loop_nucs = seq[loop_start:center + 1].upper()

    score = 0
    # +1 position purine bonus
    if center + 1 < n and seq[center + 1].upper() in ('A', 'G'):
        score += 2
    # U in loop bonus
    if 'U' in loop_nucs or 'T' in loop_nucs:
        score += 2
    # G in loop penalty
    if 'G' in loop_nucs:
        score -= 2

    # Extend stem
    right_pos = center + 1
    left_pos = loop_start - 1
    stem_len = 0
    bulge_used = False

    while right_pos < n and left_pos >= 0:
        a = seq[left_pos].upper()
        b = seq[right_pos].upper()
        if wc_pair(a, b):
            if a in 'GC' and b in 'GC':
                score += 3
            else:
                score += 1
            stem_len += 1
            right_pos += 1
            left_pos -= 1
        else:
            # Single bulge at -2 position (first mismatch only)
            if not bulge_used and left_pos - 1 >= 0:
                a2 = seq[left_pos - 1].upper()
                if wc_pair(a2, b):
                    if a2 in 'GC' and b in 'GC':
                        score += 3
                    else:
                        score += 1
                    stem_len += 1
                    right_pos += 1
                    left_pos -= 2
                    bulge_used = True
                    continue
            break

    if stem_len < 2:
        return -1, 0
    return score, stem_len


def rules_based_check(seq, center=100, threshold=9):
    """Check if center C passes RNAsee rules-based filter."""
    if pd.isna(seq) or len(seq) < 201:
        return False, 0
    if seq[center].upper() != 'C':
        return False, 0
    if seq[center - 1].upper() not in ('U', 'C', 'T'):
        return False, 0

    best_score = -1
    # Tetra-loop (4nt loop, C at 3' end)
    s4, _ = score_stemloop(seq, center, 4)
    if s4 > best_score:
        best_score = s4
    # Tri-loop (3nt loop, C at 3' end)
    s3, _ = score_stemloop(seq, center, 3)
    if s3 > best_score:
        best_score = s3

    return best_score > threshold, best_score


def main():
    print("Loading data...")
    clinvar = pd.read_csv(PROJECT_ROOT / "data/processed/clinvar_c2u_variants.csv", low_memory=False)
    scores = pd.read_csv(OUTPUT_DIR / "clinvar_all_scores.csv")
    merged = scores.merge(clinvar[['site_id', 'molecular_consequence', 'sequence']],
                          on='site_id', how='left')

    # Non-synonymous filter
    def is_nonsyn(mc):
        if pd.isna(mc) or mc == 'unknown':
            return False
        s = str(mc)
        return 'missense_variant' in s or 'nonsense' in s or 'stop_gained' in s

    merged['is_nonsyn'] = merged['molecular_consequence'].apply(is_nonsyn)
    nonsyn = merged[merged['is_nonsyn']].copy()
    print(f'Non-synonymous: {len(nonsyn):,}')

    # Check center is C
    def center_is_c(seq):
        if pd.isna(seq) or len(seq) < 201:
            return False
        return seq[100].upper() == 'C'

    nonsyn['center_c'] = nonsyn['sequence'].apply(center_is_c)
    nonsyn_c = nonsyn[nonsyn['center_c']].copy()
    print(f'Non-syn with center C: {len(nonsyn_c):,}')

    # Apply rules-based filter
    print('Applying rules-based stem-loop filter...')
    rules_results = []
    seqs = nonsyn_c['sequence'].values
    for i, seq in enumerate(seqs):
        rules_results.append(rules_based_check(seq))
        if (i + 1) % 100000 == 0:
            print(f'  {i+1:,}/{len(seqs):,}')

    nonsyn_c['rules_pass'] = [r[0] for r in rules_results]
    nonsyn_c['rules_score'] = [r[1] for r in rules_results]

    # Report results
    cats = ['Pathogenic', 'Likely_pathogenic', 'VUS', 'Conflicting', 'Likely_benign', 'Benign']

    def report(label, subset):
        t = len(subset)
        if t == 0:
            print(f'\n{label}: 0 sites')
            return
        print(f'\n=== {label} (n={t:,}) ===')
        for s in cats:
            c = (subset['significance_simple'] == s).sum()
            print(f'  {s}: {c:,} ({c/t*100:.1f}%)')
        path = subset['significance_simple'].isin(['Pathogenic', 'Likely_pathogenic']).sum()
        ben = subset['significance_simple'].isin(['Benign', 'Likely_benign']).sum()
        print(f'  --- Path+LP: {path:,} ({path/t*100:.1f}%) | Ben+LB: {ben:,} ({ben/t*100:.1f}%)')

    # Background
    report("BACKGROUND (non-syn, center=C)", nonsyn_c)

    # Rules-based only
    rules_sites = nonsyn_c[nonsyn_c['rules_pass']]
    report("RULES-BASED (score>9)", rules_sites)

    # RF only
    rf_sites = nonsyn_c[nonsyn_c['p_edited_rnasee'] >= 0.5]
    report("RF P>=0.5", rf_sites)

    # Union
    union = nonsyn_c[nonsyn_c['rules_pass'] | (nonsyn_c['p_edited_rnasee'] >= 0.5)]
    report("UNION (rules OR RF>=0.5)", union)

    # Intersection
    inter = nonsyn_c[nonsyn_c['rules_pass'] & (nonsyn_c['p_edited_rnasee'] >= 0.5)]
    report("INTERSECTION (rules AND RF>=0.5)", inter)

    # Rules threshold sweep
    print(f'\n=== Rules threshold sweep ===')
    for thresh in [3, 5, 7, 9, 11, 13, 15, 17, 20]:
        sites = nonsyn_c[nonsyn_c['rules_score'] > thresh]
        t = len(sites)
        if t == 0:
            continue
        p = sites['significance_simple'].isin(['Pathogenic', 'Likely_pathogenic']).sum()
        b = sites['significance_simple'].isin(['Benign', 'Likely_benign']).sum()
        frac = t / len(nonsyn_c) * 100
        print(f'  score>{thresh}: n={t:,} ({frac:.1f}%) | Path+LP={p/t*100:.1f}% | Ben+LB={b/t*100:.1f}%')

    # Now test OUR model (GB_Full) on the same subsets
    print('\n' + '='*70)
    print('=== COMPARISON: GB_Full on same subsets ===')
    print('='*70)

    gb_sites = nonsyn_c[nonsyn_c['p_edited_gb'] >= 0.5]
    report("GB_Full P>=0.5", gb_sites)

    # GB union with rules
    gb_union = nonsyn_c[nonsyn_c['rules_pass'] | (nonsyn_c['p_edited_gb'] >= 0.5)]
    report("GB UNION (rules OR GB>=0.5)", gb_union)

    # GB-only (sites GB finds but RF misses)
    gb_only = nonsyn_c[(nonsyn_c['p_edited_gb'] >= 0.5) & (nonsyn_c['p_edited_rnasee'] < 0.5)]
    report("GB-ONLY (GB>=0.5, RF<0.5)", gb_only)

    # Save results
    results = {
        "description": "RNAsee 2024 ClinVar replication",
        "n_nonsyn": int(len(nonsyn)),
        "n_nonsyn_center_c": int(len(nonsyn_c)),
        "n_rules_pass": int(nonsyn_c['rules_pass'].sum()),
        "rules_frac": float(nonsyn_c['rules_pass'].mean()),
    }
    with open(OUTPUT_DIR / "rnasee_replication_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved results to {OUTPUT_DIR / "rnasee_replication_results.json"}')


if __name__ == "__main__":
    main()
