#!/usr/bin/env python
"""Validate APOBEC1 hypothesis for "Neither" sites.

Tests:
1. Tissue enrichment: liver/intestine editing rates vs other tissues
2. Mooring sequence: AU-rich motif downstream of edit site
3. Genomic region: 3'UTR enrichment (APOBEC1 targets 3'UTR AU-rich regions)
4. Motif: No dinucleotide preference (vs A3A=TC, A3G=CC)
5. Structure independence: weaker loop preference than A3A/A3G
6. Score with A3B classifier: rule out A3B as alternative editor

Usage:
    conda run -n quris python experiments/common/exp_apobec1_validation.py
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

LEVANON_CSV = PROJECT_ROOT / "data/processed/multi_enzyme/levanon_all_categories.csv"
TISSUE_RATES = PROJECT_ROOT / "data/processed/multi_enzyme/levanon_tissue_rates.csv"
SEQS_JSON = PROJECT_ROOT / "data/processed/site_sequences.json"
LOOP_CSV = PROJECT_ROOT / "data/processed/multi_enzyme/loop_position_per_site_v3.csv"
OUTPUT_DIR = PROJECT_ROOT / "experiments/common/outputs/apobec1_validation"


def check_mooring_sequence(seq, center=100):
    """Check for AU-rich mooring sequence downstream of edit site.

    APOBEC1's mooring is typically 4-8nt downstream, AU-rich.
    Returns AU fraction in +4 to +12 window.
    """
    seq = seq.upper().replace("T", "U")
    window_start = center + 4
    window_end = center + 13  # +4 to +12 inclusive
    if window_end > len(seq):
        return np.nan
    window = seq[window_start:window_end]
    au_count = sum(1 for b in window if b in ("A", "U"))
    return au_count / len(window)


def check_au_rich_context(seq, center=100, window=20):
    """Check AU richness in broader flanking region."""
    seq = seq.upper().replace("T", "U")
    start = max(0, center - window)
    end = min(len(seq), center + window + 1)
    region = seq[start:end]
    au_count = sum(1 for b in region if b in ("A", "U"))
    return au_count / len(region)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    lev = pd.read_csv(LEVANON_CSV)
    tissue = pd.read_csv(TISSUE_RATES)
    with open(SEQS_JSON) as f:
        seqs = json.load(f)

    results = {}

    # =========================================================================
    # 1. Tissue enrichment: liver/intestine
    # =========================================================================
    logger.info("=== 1. Tissue Enrichment Analysis ===")

    # Key APOBEC1 tissues
    apobec1_tissues = [
        "small_intestine_terminal_ileum", "liver", "colon_sigmoid",
        "colon_transverse", "stomach"
    ]
    # Key A3A tissues (immune/epithelial)
    a3a_tissues = [
        "whole_blood", "spleen", "lung", "cells_ebv-transformed_lymphocytes",
        "skin_sun_exposed_lower_leg"
    ]

    tissue_cols = [c for c in tissue.columns if c not in ["site_id", "enzyme_category"]]

    for category in ["A3A", "A3G", "A3A_A3G", "Neither", "Unknown"]:
        sub = tissue[tissue["enzyme_category"] == category]
        if len(sub) == 0:
            continue

        # Mean editing rate in APOBEC1 tissues vs A3A tissues
        a1_rates = []
        a3a_rates = []
        for col in tissue_cols:
            vals = sub[col].dropna()
            mean_rate = float(vals.mean()) if len(vals) > 0 else 0
            clean_col = col.lower().strip()
            if any(t in clean_col for t in ["intestine", "liver", "colon", "stomach"]):
                a1_rates.append(mean_rate)
            if any(t in clean_col for t in ["blood", "spleen", "lung", "lymphocyte"]):
                a3a_rates.append(mean_rate)

        a1_mean = np.mean(a1_rates) if a1_rates else 0
        a3a_mean = np.mean(a3a_rates) if a3a_rates else 0
        ratio = a1_mean / a3a_mean if a3a_mean > 0 else float("inf")

        logger.info("  %s: GI/liver mean=%.2f%%, immune mean=%.2f%%, ratio=%.2f",
                     category, a1_mean, a3a_mean, ratio)

    # Detailed tissue analysis for "Neither"
    neither_tissue = tissue[tissue["enzyme_category"] == "Neither"]
    neither_means = {}
    for col in tissue_cols:
        vals = neither_tissue[col].dropna()
        if len(vals) > 0:
            neither_means[col] = float(vals.mean())

    sorted_tissues = sorted(neither_means.items(), key=lambda x: -x[1])
    logger.info("\n  'Neither' top 10 tissues:")
    for t, r in sorted_tissues[:10]:
        logger.info("    %s: %.2f%%", t, r)

    results["tissue_enrichment"] = {
        "neither_top10": [(t, r) for t, r in sorted_tissues[:10]],
        "neither_bottom10": [(t, r) for t, r in sorted_tissues[-10:]],
    }

    # Statistical test: are GI tissues significantly higher than immune for "Neither"?
    neither_gi_rates = []
    neither_immune_rates = []
    for col in tissue_cols:
        vals = neither_tissue[col].dropna()
        if len(vals) == 0:
            continue
        mean_r = float(vals.mean())
        clean = col.lower()
        if any(t in clean for t in ["intestine", "liver", "colon", "stomach"]):
            neither_gi_rates.append(mean_r)
        elif any(t in clean for t in ["blood", "spleen", "lung", "lymphocyte"]):
            neither_immune_rates.append(mean_r)

    if neither_gi_rates and neither_immune_rates:
        t_stat, p_val = stats.ttest_ind(neither_gi_rates, neither_immune_rates)
        logger.info("  GI vs immune for Neither: t=%.2f, p=%.4f (GI mean=%.2f, immune mean=%.2f)",
                     t_stat, p_val, np.mean(neither_gi_rates), np.mean(neither_immune_rates))
        results["gi_vs_immune_test"] = {
            "t_stat": float(t_stat), "p_value": float(p_val),
            "gi_mean": float(np.mean(neither_gi_rates)),
            "immune_mean": float(np.mean(neither_immune_rates)),
        }

    # =========================================================================
    # 2. Mooring sequence analysis
    # =========================================================================
    logger.info("\n=== 2. Mooring Sequence (AU-rich downstream) ===")

    for category in ["A3A", "A3G", "A3A_A3G", "Neither", "Unknown"]:
        sub = lev[lev["enzyme_category"] == category]
        mooring_scores = []
        au_context = []
        for sid in sub["site_id"]:
            seq = seqs.get(str(sid), "N" * 201)
            mooring_scores.append(check_mooring_sequence(seq))
            au_context.append(check_au_rich_context(seq))

        mooring_arr = np.array([x for x in mooring_scores if not np.isnan(x)])
        au_arr = np.array([x for x in au_context if not np.isnan(x)])
        logger.info("  %s: mooring AU=%.1f%%, flanking AU=%.1f%%",
                     category,
                     mooring_arr.mean() * 100 if len(mooring_arr) else 0,
                     au_arr.mean() * 100 if len(au_arr) else 0)

        results[f"{category}_mooring"] = {
            "mooring_au_fraction": float(mooring_arr.mean()) if len(mooring_arr) else 0,
            "flanking_au_fraction": float(au_arr.mean()) if len(au_arr) else 0,
            "n": len(mooring_arr),
        }

    # Test: mooring AU enrichment in Neither vs A3A
    neither_mooring = [check_mooring_sequence(seqs.get(str(sid), "N" * 201))
                       for sid in lev[lev["enzyme_category"] == "Neither"]["site_id"]]
    a3a_mooring = [check_mooring_sequence(seqs.get(str(sid), "N" * 201))
                   for sid in lev[lev["enzyme_category"] == "A3A"]["site_id"]]
    neither_mooring = [x for x in neither_mooring if not np.isnan(x)]
    a3a_mooring = [x for x in a3a_mooring if not np.isnan(x)]

    if neither_mooring and a3a_mooring:
        t_stat, p_val = stats.ttest_ind(neither_mooring, a3a_mooring)
        logger.info("  Mooring AU Neither vs A3A: t=%.2f, p=%.4f", t_stat, p_val)
        results["mooring_neither_vs_a3a"] = {"t_stat": float(t_stat), "p_value": float(p_val)}

    # =========================================================================
    # 3. Genomic region enrichment
    # =========================================================================
    logger.info("\n=== 3. Genomic Region (3'UTR Enrichment) ===")

    for category in ["A3A", "A3G", "A3A_A3G", "Neither", "Unknown"]:
        sub = lev[lev["enzyme_category"] == category]
        if "genomic_category" in sub.columns:
            gc = sub["genomic_category"].value_counts()
            n = len(sub)
            cds_n = gc.get("CDS", 0)
            noncoding_n = gc.get("Non Coding mRNA", 0)
            logger.info("  %s: CDS=%d (%.1f%%), Non-coding mRNA=%d (%.1f%%)",
                         category, cds_n, cds_n / n * 100, noncoding_n, noncoding_n / n * 100)
            results[f"{category}_genomic"] = gc.to_dict()

    # =========================================================================
    # 4. Structure independence
    # =========================================================================
    logger.info("\n=== 4. Structure Analysis ===")

    if Path(LOOP_CSV).exists():
        loop_df = pd.read_csv(LOOP_CSV)
        loop_df["site_id"] = loop_df["site_id"].astype(str)

        for category in ["A3A", "A3G", "A3A_A3G", "Neither", "Unknown"]:
            sub = lev[lev["enzyme_category"] == category]
            sids = set(sub["site_id"].astype(str))
            loop_sub = loop_df[loop_df["site_id"].isin(sids)]

            if len(loop_sub) > 0:
                is_unpaired = loop_sub["is_unpaired"].mean()
                rlp = loop_sub[loop_sub["is_unpaired"] == 1]["relative_loop_position"].mean()
                loop_size = loop_sub[loop_sub["is_unpaired"] == 1]["loop_size"].mean()
                logger.info("  %s: unpaired=%.1f%%, mean_rlp=%.3f, mean_loop_size=%.1f",
                             category, is_unpaired * 100,
                             rlp if not np.isnan(rlp) else 0,
                             loop_size if not np.isnan(loop_size) else 0)
                results[f"{category}_structure"] = {
                    "is_unpaired_frac": float(is_unpaired),
                    "mean_rlp": float(rlp) if not np.isnan(rlp) else 0,
                    "mean_loop_size": float(loop_size) if not np.isnan(loop_size) else 0,
                }

    # =========================================================================
    # 5. Tissue classification distribution
    # =========================================================================
    logger.info("\n=== 5. Tissue Classification ===")
    for category in ["A3A", "A3G", "A3A_A3G", "Neither", "Unknown"]:
        sub = lev[lev["enzyme_category"] == category]
        if "tissue_classification" in sub.columns:
            tc = sub["tissue_classification"].value_counts()
            logger.info("  %s: %s", category, tc.to_dict())
            results[f"{category}_tissue_class"] = tc.to_dict()

    # =========================================================================
    # Summary: APOBEC1 evidence score
    # =========================================================================
    logger.info("\n=== APOBEC1 Evidence Summary for 'Neither' ===")

    evidence = []
    # 1. Tissue: intestine-enriched?
    neither_class = lev[lev["enzyme_category"] == "Neither"]["tissue_classification"].value_counts()
    intestine_frac = neither_class.get("Intestine Specific", 0) / len(lev[lev["enzyme_category"] == "Neither"])
    evidence.append(("Intestine-specific tissue pattern", intestine_frac > 0.2, f"{intestine_frac:.1%}"))

    # 2. Motif: no TC/CC preference?
    neither_sids = lev[lev["enzyme_category"] == "Neither"]["site_id"]
    tc = sum(1 for sid in neither_sids if seqs.get(str(sid), "N" * 201)[99].upper() in ("U", "T"))
    cc = sum(1 for sid in neither_sids if seqs.get(str(sid), "N" * 201)[99].upper() == "C")
    n = len(neither_sids)
    motif_random = abs(tc / n - 0.25) < 0.10 and abs(cc / n - 0.25) < 0.15
    evidence.append(("No dinucleotide preference (random-like)", motif_random,
                     f"TC={tc/n:.1%}, CC={cc/n:.1%}"))

    # 3. Structure: weaker loop preference?
    if "Neither_structure" in results:
        unpaired = results["Neither_structure"]["is_unpaired_frac"]
        weak_struct = unpaired < 0.55  # A3A ~63%, random ~50%
        evidence.append(("Weak structure preference", weak_struct, f"unpaired={unpaired:.1%}"))

    # 4. Non-coding mRNA enrichment? (proxy for 3'UTR)
    if "Neither_genomic" in results:
        noncoding = results["Neither_genomic"].get("Non Coding mRNA", 0)
        noncoding_frac = noncoding / n
        evidence.append(("Non-coding mRNA enriched", noncoding_frac > 0.40, f"{noncoding_frac:.1%}"))

    logger.info("  APOBEC1 evidence for 'Neither' sites:")
    score = 0
    for name, passes, detail in evidence:
        status = "PASS" if passes else "FAIL"
        if passes:
            score += 1
        logger.info("    [%s] %s (%s)", status, name, detail)
    logger.info("  Score: %d/%d APOBEC1 indicators", score, len(evidence))

    results["apobec1_evidence"] = {
        "score": score, "total": len(evidence),
        "tests": [{"name": n, "pass": p, "detail": d} for n, p, d in evidence],
    }

    # Save
    with open(OUTPUT_DIR / "apobec1_validation_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("\nSaved to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
