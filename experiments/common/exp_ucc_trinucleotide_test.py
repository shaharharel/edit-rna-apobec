#!/usr/bin/env python
"""Test whether "Both" (A3A_A3G) sites are enriched for UCC/TCC trinucleotide.

Hypothesis: UCC is simultaneously TC context for A3A (reading C at pos 0)
and CC context for A3G (reading the same C). If "Both" sites are enriched
for UCC relative to A3A-only and A3G-only, this confirms the shared motif.

Usage:
    conda run -n quris python experiments/common/exp_ucc_trinucleotide_test.py
"""

import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

LEVANON_CSV = PROJECT_ROOT / "data/processed/multi_enzyme/levanon_all_categories.csv"
SEQS_JSON = PROJECT_ROOT / "data/processed/site_sequences.json"
OUTPUT_DIR = PROJECT_ROOT / "experiments/common/outputs/ucc_trinucleotide"


def get_trinucleotide(seq, center=100):
    """Extract -1, 0, +1 trinucleotide context around edit site."""
    seq = seq.upper().replace("T", "U")
    if len(seq) < center + 2:
        return "NNN"
    return seq[center - 1] + seq[center] + seq[center + 1]


def get_pentanucleotide(seq, center=100):
    """Extract -2 to +2 pentanucleotide context."""
    seq = seq.upper().replace("T", "U")
    if len(seq) < center + 3:
        return "NNNNN"
    return seq[center - 2:center + 3]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    lev = pd.read_csv(LEVANON_CSV)
    with open(SEQS_JSON) as f:
        seqs = json.load(f)

    results = {}

    for category in ["A3A", "A3G", "A3A_A3G", "Neither", "Unknown"]:
        subset = lev[lev["enzyme_category"] == category]
        trinucs = []
        pentanucs = []
        dinuc_m1 = []  # -1 position (TC, CC, AC, GC)

        for sid in subset["site_id"]:
            seq = seqs.get(str(sid), "N" * 201)
            tri = get_trinucleotide(seq)
            penta = get_pentanucleotide(seq)
            trinucs.append(tri)
            pentanucs.append(penta)
            dinuc_m1.append(seq[99].upper().replace("T", "U") if len(seq) > 99 else "N")

        n = len(subset)
        tri_counts = Counter(trinucs)
        penta_counts = Counter(pentanucs)
        dinuc_counts = Counter(dinuc_m1)

        # Key trinucleotides
        ucc = tri_counts.get("UCC", 0)
        uca = tri_counts.get("UCA", 0)
        ucg = tri_counts.get("UCG", 0)
        ucu = tri_counts.get("UCU", 0)
        cca = tri_counts.get("CCA", 0)
        ccg = tri_counts.get("CCG", 0)
        ccu = tri_counts.get("CCU", 0)
        ccc = tri_counts.get("CCC", 0)

        tc_total = sum(1 for d in dinuc_m1 if d == "U")
        cc_total = sum(1 for d in dinuc_m1 if d == "C")

        cat_results = {
            "n_sites": n,
            "dinucleotide": {
                "UC": tc_total, "CC": cc_total,
                "AC": sum(1 for d in dinuc_m1 if d == "A"),
                "GC": sum(1 for d in dinuc_m1 if d == "G"),
                "UC_pct": tc_total / n * 100 if n else 0,
                "CC_pct": cc_total / n * 100 if n else 0,
            },
            "trinucleotide": {
                "UCC": ucc, "UCA": uca, "UCG": ucg, "UCU": ucu,
                "CCA": cca, "CCG": ccg, "CCU": ccu, "CCC": ccc,
                "UCC_pct": ucc / n * 100 if n else 0,
                "UCC_among_UC": ucc / tc_total * 100 if tc_total else 0,
                "UCC_among_CC": ucc / cc_total * 100 if cc_total else 0,
            },
            "top_trinucleotides": sorted(tri_counts.items(), key=lambda x: -x[1])[:10],
            "top_pentanucleotides": sorted(penta_counts.items(), key=lambda x: -x[1])[:10],
        }
        results[category] = cat_results

        logger.info("\n=== %s (n=%d) ===", category, n)
        logger.info("  Dinucleotide: UC=%.1f%%, CC=%.1f%%",
                     tc_total / n * 100 if n else 0, cc_total / n * 100 if n else 0)
        logger.info("  UCC: %d (%.1f%% of all, %.1f%% of UC, %.1f%% of CC)",
                     ucc, ucc / n * 100 if n else 0,
                     ucc / tc_total * 100 if tc_total else 0,
                     ucc / cc_total * 100 if cc_total else 0)
        logger.info("  Top trinucs: %s", sorted(tri_counts.items(), key=lambda x: -x[1])[:6])

    # =========================================================================
    # Statistical tests: Is UCC enriched in "Both" vs A3A-only and A3G-only?
    # =========================================================================
    logger.info("\n=== Statistical Tests ===")

    both = results["A3A_A3G"]
    a3a = results["A3A"]
    a3g = results["A3G"]

    # Fisher exact test: UCC vs non-UCC in Both vs A3A
    both_ucc = both["trinucleotide"]["UCC"]
    both_non_ucc = both["n_sites"] - both_ucc
    a3a_ucc = a3a["trinucleotide"]["UCC"]
    a3a_non_ucc = a3a["n_sites"] - a3a_ucc

    table_vs_a3a = [[both_ucc, both_non_ucc], [a3a_ucc, a3a_non_ucc]]
    or_a3a, p_a3a = stats.fisher_exact(table_vs_a3a)
    logger.info("UCC enrichment Both vs A3A: OR=%.2f, p=%.4f", or_a3a, p_a3a)
    logger.info("  Both: %d/%d (%.1f%%), A3A: %d/%d (%.1f%%)",
                both_ucc, both["n_sites"], both_ucc / both["n_sites"] * 100,
                a3a_ucc, a3a["n_sites"], a3a_ucc / a3a["n_sites"] * 100)

    # Fisher exact test: UCC vs non-UCC in Both vs A3G
    a3g_ucc = a3g["trinucleotide"]["UCC"]
    a3g_non_ucc = a3g["n_sites"] - a3g_ucc

    table_vs_a3g = [[both_ucc, both_non_ucc], [a3g_ucc, a3g_non_ucc]]
    or_a3g, p_a3g = stats.fisher_exact(table_vs_a3g)
    logger.info("UCC enrichment Both vs A3G: OR=%.2f, p=%.4f", or_a3g, p_a3g)
    logger.info("  Both: %d/%d (%.1f%%), A3G: %d/%d (%.1f%%)",
                both_ucc, both["n_sites"], both_ucc / both["n_sites"] * 100,
                a3g_ucc, a3g["n_sites"], a3g_ucc / a3g["n_sites"] * 100)

    # Also test: Is CC enrichment in Both driven by UCC specifically?
    # Compare UCC fraction among CC-context sites: Both vs A3G
    both_ucc_in_cc = both["trinucleotide"]["UCC"]
    both_other_cc = results["A3A_A3G"]["dinucleotide"]["CC"] - both_ucc_in_cc
    a3g_ucc_in_cc = a3g["trinucleotide"]["UCC"]
    a3g_other_cc = results["A3G"]["dinucleotide"]["CC"] - a3g_ucc_in_cc

    if both_other_cc >= 0 and a3g_other_cc >= 0:
        table_cc = [[both_ucc_in_cc, max(0, both_other_cc)],
                     [a3g_ucc_in_cc, max(0, a3g_other_cc)]]
        or_cc, p_cc = stats.fisher_exact(table_cc)
        logger.info("UCC among CC-context: Both vs A3G: OR=%.2f, p=%.4f", or_cc, p_cc)

    results["statistical_tests"] = {
        "ucc_both_vs_a3a": {"OR": float(or_a3a), "p": float(p_a3a)},
        "ucc_both_vs_a3g": {"OR": float(or_a3g), "p": float(p_a3g)},
    }

    # Save results
    with open(OUTPUT_DIR / "ucc_trinucleotide_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Save trinucleotide comparison table
    rows = []
    for cat in ["A3A", "A3G", "A3A_A3G", "Neither", "Unknown"]:
        r = results[cat]
        tri = r["trinucleotide"]
        rows.append({
            "category": cat, "n": r["n_sites"],
            "UC_pct": r["dinucleotide"]["UC_pct"],
            "CC_pct": r["dinucleotide"]["CC_pct"],
            "UCC": tri["UCC"], "UCC_pct": tri["UCC_pct"],
            "UCA": tri["UCA"], "UCG": tri["UCG"],
            "CCG": tri["CCG"], "CCA": tri["CCA"],
        })
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "trinucleotide_comparison.csv", index=False)

    logger.info("\nSaved results to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
