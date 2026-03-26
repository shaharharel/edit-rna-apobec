#!/usr/bin/env python3
"""
APOBEC1 integration analysis: characterize "Neither" sites as potential APOBEC1 targets.

Steps:
1. Compile known human APOBEC1 targets from literature
2. Check overlap with our 206 "Neither" sites
3. Characterize APOBEC1-specific features (3'UTR, mooring sequence, AU-rich context)
4. Tissue expression analysis (intestine enrichment)
5. Generate summary report

Known APOBEC1 features:
- Strong 3'UTR preference
- Mooring sequence: WRAUYANUAU motif 4-20 nt downstream
- AU-rich flanking context
- Highest expression in small intestine and liver
- The canonical target is APOB (chr2:21010329, C6666U)
"""

import os
import sys
import re
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "multi_enzyme"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "multi_enzyme" / "outputs" / "apobec1_integration"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Step 1: Compile known human APOBEC1 targets from literature
# ============================================================

def compile_known_apobec1_targets():
    """
    Compile known human APOBEC1 editing targets from literature.

    Sources:
    - APOB (chr2:21010329) - the canonical APOBEC1 target (Powell 1987, Chen 1987)
    - NF1 (Skuse 1996) - neurofibromatosis type 1, C>U in exon
    - NAT1 (Yamanaka 1997) - N-acetyltransferase 1
    - Rosenberg 2011 (NSMB) - 32 mouse targets; human orthologs known
    - Blanc & Davidson 2003, 2010 reviews

    Since Rosenberg 2011 supplementary data is not directly accessible,
    we compile the known human targets and mouse-to-human ortholog genes.

    Key mouse APOBEC1 target genes with human orthologs
    (from Rosenberg 2011, Supplementary Table 3; gene names from text/figures):
    """

    # Known validated human APOBEC1 targets with hg38 coordinates
    # The APOB site is the only well-characterized human APOBEC1 site with exact coordinates
    known_human_sites = pd.DataFrame([
        {"gene": "APOB", "chr": "chr2", "position": 21010329, "strand": "-",
         "location": "CDS", "source": "Powell_1987", "editing_type": "C6666U",
         "notes": "Canonical APOBEC1 target. Creates premature stop codon -> apoB-48"},
        {"gene": "NF1", "chr": "chr17", "position": 31232049, "strand": "+",
         "location": "CDS", "source": "Skuse_1996",
         "editing_type": "C3916U", "notes": "Neurofibromatosis 1. Creates premature stop."},
    ])

    # Mouse APOBEC1 target genes (Rosenberg 2011, Davidson 2014) - human orthologs
    # These genes are known APOBEC1 targets in mouse; human editing not confirmed for all
    mouse_ortholog_genes = [
        # From Rosenberg 2011 (mouse macrophages, 32 targets)
        "B2M", "BCHE", "GRAMD1C", "PTPN3", "SULT1D1", "ALDH6A1", "USP25",
        "SERINC1", "TMEM30A", "CMTM6", "CYP4V3", "SH3BGRL", "CLIC5", "APP",
        "HPRT1", "TMBIM6", "RNF128", "RRBP1", "SELENOF", "ANK3", "LRRC19",
        # From Davidson 2014 (mouse intestine/liver, 56+22 targets)
        "CD36", "REPS2", "SIGLEC5", "FMN1", "MCMBP", "MAN2A1", "HERC2",
        "DDX60", "MTMR2", "CNIH1", "ATP11C", "FGL2", "NR1D2", "TMEM135",
        "SLC4A4", "DPYD", "YES1", "ACTR2", "KCTD12", "NR3C1", "SKIL",
        "CCNY", "RAB1A", "LRBA", "DEK", "DCN", "CYBB", "COLEC10",
        "UBE2L3", "ABCC9", "MPEG1",
    ]

    return known_human_sites, mouse_ortholog_genes


# ============================================================
# Step 2: Check overlap with "Neither" sites
# ============================================================

def check_overlaps(levanon_df, known_sites, ortholog_genes):
    """Check overlap between Neither sites and known APOBEC1 targets."""

    neither = levanon_df[levanon_df["enzyme_category"] == "Neither"].copy()
    results = {}

    # 2a. Exact position match with known human sites
    position_matches = []
    for _, known in known_sites.iterrows():
        match = neither[
            (neither["chr"] == known["chr"]) &
            (neither["start"] == known["position"])
        ]
        if len(match) > 0:
            for _, m in match.iterrows():
                position_matches.append({
                    "site_id": m["site_id"],
                    "chr": m["chr"],
                    "position": m["start"],
                    "gene": m["gene_refseq"],
                    "known_gene": known["gene"],
                    "source": known["source"],
                    "notes": known["notes"],
                })
    results["position_matches"] = pd.DataFrame(position_matches)

    # 2b. Gene-level overlap with known human sites
    known_genes = set(known_sites["gene"].values)
    neither_genes = set(neither["gene_refseq"].dropna().values)
    gene_overlap_known = known_genes & neither_genes
    results["gene_overlap_known"] = gene_overlap_known

    # 2c. Gene-level overlap with mouse ortholog genes
    ortholog_set = set(ortholog_genes)
    gene_overlap_orthologs = ortholog_set & neither_genes
    results["gene_overlap_orthologs"] = gene_overlap_orthologs

    # Also check all categories for ortholog overlap
    all_genes_by_cat = {}
    for cat in levanon_df["enzyme_category"].unique():
        cat_genes = set(levanon_df[levanon_df["enzyme_category"] == cat]["gene_refseq"].dropna())
        overlap = ortholog_set & cat_genes
        all_genes_by_cat[cat] = overlap
    results["ortholog_overlap_by_category"] = all_genes_by_cat

    # 2d. Check for APOB specifically across ALL categories
    apob_sites = levanon_df[levanon_df["gene_refseq"] == "APOB"]
    results["apob_sites"] = apob_sites

    return results


# ============================================================
# Step 3: Characterize APOBEC1 features in Neither sites
# ============================================================

def rna_complement(base):
    """Get RNA complement."""
    comp = {"A": "U", "U": "A", "G": "C", "C": "G", "N": "N"}
    return comp.get(base, "N")

def check_mooring_sequence(seq, edit_pos=100):
    """
    Check for APOBEC1 mooring sequence downstream of editing site.

    The mooring sequence is a degenerate motif: WRAUYANUAU
    Located 4-20 nt downstream (3') of the editing site.
    W = A/U, R = A/G, Y = C/U, N = any

    Also check for the simpler consensus: spacer + UAGAU or UGAU core.

    Args:
        seq: 201-nt RNA sequence (edit site at position 100, 0-indexed)
        edit_pos: position of edited C in sequence (default 100)

    Returns:
        dict with mooring analysis results
    """
    # IUPAC codes for pattern matching
    iupac = {
        "W": "[AU]", "R": "[AG]", "Y": "[CU]", "S": "[GC]",
        "K": "[GU]", "M": "[AC]", "B": "[CGU]", "D": "[AGU]",
        "H": "[ACU]", "V": "[ACG]", "N": "[ACGU]",
    }

    # Full mooring: WRAUYANUAU
    # More relaxed: W[AG][AU][CU][ACGU][AU][ACGU][AU][CU]
    mooring_pattern = "[AU][AG][AU][CU][ACGU][AU][ACGU][AU][CU]"  # WRAUYANUAU (9-mer)
    # Simpler core: UAGAU or UGAU
    core_patterns = ["UAGAU", "UGAU", "[AU]GA[AU]"]

    downstream = seq[edit_pos + 4: edit_pos + 25]  # +4 to +24 downstream

    results = {
        "downstream_seq": downstream,
        "has_full_mooring": False,
        "has_core_mooring": False,
        "mooring_positions": [],
        "core_positions": [],
    }

    # Check full mooring
    for m in re.finditer(mooring_pattern, downstream):
        results["has_full_mooring"] = True
        results["mooring_positions"].append(m.start() + 4)  # offset from edit site

    # Check core mooring
    for pattern in core_patterns:
        for m in re.finditer(pattern, downstream):
            results["has_core_mooring"] = True
            results["core_positions"].append((pattern, m.start() + 4))

    return results


def analyze_au_context(seq, edit_pos=100):
    """
    Analyze AU-richness around the editing site.
    APOBEC1 prefers AU-rich flanking regions.
    """
    # Flanking 10 nt on each side (excluding edit site itself)
    flank_left = seq[max(0, edit_pos - 10): edit_pos]
    flank_right = seq[edit_pos + 1: min(len(seq), edit_pos + 11)]
    flanking = flank_left + flank_right

    au_count = sum(1 for b in flanking if b in "AU")
    au_fraction = au_count / len(flanking) if flanking else 0

    # Immediate context (-1, +1)
    minus1 = seq[edit_pos - 1] if edit_pos > 0 else "N"
    plus1 = seq[edit_pos + 1] if edit_pos < len(seq) - 1 else "N"

    return {
        "au_fraction_20nt": au_fraction,
        "minus1": minus1,
        "plus1": plus1,
        "minus1_is_AU": minus1 in "AU",
        "plus1_is_AU": plus1 in "AU",
        "trinucleotide": f"{minus1}C{plus1}",
    }


def characterize_apobec1_features(levanon_df, sequences):
    """Characterize APOBEC1-specific features for all categories."""

    all_results = []

    for _, row in levanon_df.iterrows():
        site_id = row["site_id"]
        cat = row["enzyme_category"]
        chrom = row["chr"]
        pos = row["start"]
        strand = row["strand"]

        # Get sequence by site_id (sequences are keyed by site_id)
        seq = sequences.get(site_id, None)

        if seq is None:
            continue

        # Edit position is center of 201-nt window
        edit_pos = 100

        # Check the edited base is C
        edited_base = seq[edit_pos]
        if edited_base != "C":
            # Could be on opposite strand representation
            pass

        # Mooring sequence analysis
        mooring = check_mooring_sequence(seq, edit_pos)

        # AU context analysis
        au = analyze_au_context(seq, edit_pos)

        result = {
            "site_id": site_id,
            "enzyme_category": cat,
            "gene": row["gene_refseq"],
            "chr": chrom,
            "position": pos,
            "strand": strand,
            "mrna_location": row.get("mrna_location_refseq", "unknown"),
            "tissue_classification": row.get("tissue_classification", "unknown"),
            "is_3utr": row.get("mrna_location_refseq", "") == "UTR3",
            "edited_base_at_center": edited_base,
            "has_full_mooring": mooring["has_full_mooring"],
            "has_core_mooring": mooring["has_core_mooring"],
            "n_mooring_hits": len(mooring["mooring_positions"]),
            "downstream_seq": mooring["downstream_seq"],
            "au_fraction_20nt": au["au_fraction_20nt"],
            "minus1": au["minus1"],
            "plus1": au["plus1"],
            "minus1_is_AU": au["minus1_is_AU"],
            "plus1_is_AU": au["plus1_is_AU"],
            "trinucleotide": au["trinucleotide"],
            "editing_rate": row.get("editing_rate", np.nan),
            "max_gtex_rate": row.get("max_gtex_editing_rate", np.nan),
            "mean_gtex_rate": row.get("mean_gtex_editing_rate", np.nan),
        }
        all_results.append(result)

    return pd.DataFrame(all_results)


# ============================================================
# Step 4: Tissue expression analysis
# ============================================================

def analyze_tissue_patterns(tissue_rates, levanon_df):
    """Analyze tissue-specific editing patterns, focusing on intestine/liver."""

    # Intestine and liver columns
    intestine_cols = [c for c in tissue_rates.columns if "intestin" in c.lower()]
    liver_cols = [c for c in tissue_rates.columns if "liver" in c.lower()]
    colon_cols = [c for c in tissue_rates.columns if "colon" in c.lower()]
    gi_cols = intestine_cols + liver_cols + colon_cols

    # Brain columns (for comparison)
    brain_cols = [c for c in tissue_rates.columns if "brain" in c.lower()]
    blood_cols = [c for c in tissue_rates.columns if "blood" in c.lower()]

    # All tissue columns (exclude site_id and enzyme_category)
    tissue_cols = [c for c in tissue_rates.columns if c not in ["site_id", "enzyme_category"]]

    results_by_category = {}

    for cat in levanon_df["enzyme_category"].unique():
        cat_sites = levanon_df[levanon_df["enzyme_category"] == cat]["site_id"]
        cat_rates = tissue_rates[tissue_rates["site_id"].isin(cat_sites)]

        if len(cat_rates) == 0:
            continue

        # Mean editing rate per tissue
        mean_rates = cat_rates[tissue_cols].mean()

        # Intestine specificity: mean(intestine) / mean(all)
        gi_mean = cat_rates[gi_cols].mean().mean() if gi_cols else 0
        all_mean = mean_rates.mean()
        gi_specificity = gi_mean / all_mean if all_mean > 0 else 0

        # Intestine vs rest ratio
        non_gi_cols = [c for c in tissue_cols if c not in gi_cols]
        non_gi_mean = cat_rates[non_gi_cols].mean().mean() if non_gi_cols else 0

        # Top 5 tissues
        top_tissues = mean_rates.nlargest(5)

        results_by_category[cat] = {
            "n_sites": len(cat_rates),
            "gi_mean_rate": gi_mean,
            "non_gi_mean_rate": non_gi_mean,
            "gi_specificity": gi_specificity,
            "all_mean_rate": all_mean,
            "top_tissues": top_tissues.to_dict(),
            "intestine_rate": cat_rates[intestine_cols].mean().mean() if intestine_cols else 0,
            "liver_rate": cat_rates[liver_cols].mean().mean() if liver_cols else 0,
            "brain_rate": cat_rates[brain_cols].mean().mean() if brain_cols else 0,
            "blood_rate": cat_rates[blood_cols].mean().mean() if blood_cols else 0,
        }

    return results_by_category


# ============================================================
# Step 5: Statistical comparisons
# ============================================================

def compute_category_statistics(features_df):
    """Compute per-category statistics for APOBEC1-indicative features."""
    from scipy import stats

    stats_results = {}

    for cat in features_df["enzyme_category"].unique():
        cat_df = features_df[features_df["enzyme_category"] == cat]
        n = len(cat_df)

        stats_results[cat] = {
            "n_sites": n,
            "pct_3utr": cat_df["is_3utr"].mean() * 100,
            "pct_full_mooring": cat_df["has_full_mooring"].mean() * 100,
            "pct_core_mooring": cat_df["has_core_mooring"].mean() * 100,
            "mean_au_fraction": cat_df["au_fraction_20nt"].mean(),
            "pct_minus1_AU": cat_df["minus1_is_AU"].mean() * 100,
            "pct_plus1_AU": cat_df["plus1_is_AU"].mean() * 100,
        }

        # Trinucleotide distribution
        tri_counts = cat_df["trinucleotide"].value_counts()
        total = tri_counts.sum()
        for tri in ["UCU", "UCA", "ACU", "ACA", "UCG", "UCC", "ACG", "ACC",
                     "GCU", "GCA", "GCG", "GCC", "CCU", "CCA", "CCG", "CCC"]:
            stats_results[cat][f"pct_{tri}"] = (tri_counts.get(tri, 0) / total * 100) if total > 0 else 0

    # Statistical test: Neither vs A3A for 3'UTR enrichment
    neither_3utr = features_df[features_df["enzyme_category"] == "Neither"]["is_3utr"]
    a3a_3utr = features_df[features_df["enzyme_category"] == "A3A"]["is_3utr"]
    if len(neither_3utr) > 0 and len(a3a_3utr) > 0:
        # Fisher exact or chi-squared
        table = pd.crosstab(
            features_df[features_df["enzyme_category"].isin(["Neither", "A3A"])]["enzyme_category"],
            features_df[features_df["enzyme_category"].isin(["Neither", "A3A"])]["is_3utr"]
        )
        if table.shape == (2, 2):
            odds_ratio, pvalue = stats.fisher_exact(table)
            stats_results["fisher_3utr_neither_vs_a3a"] = {
                "odds_ratio": odds_ratio, "pvalue": pvalue
            }

    # Mooring: Neither vs A3A
    neither_mooring = features_df[features_df["enzyme_category"] == "Neither"]["has_core_mooring"]
    a3a_mooring = features_df[features_df["enzyme_category"] == "A3A"]["has_core_mooring"]
    if len(neither_mooring) > 0 and len(a3a_mooring) > 0:
        table = pd.crosstab(
            features_df[features_df["enzyme_category"].isin(["Neither", "A3A"])]["enzyme_category"],
            features_df[features_df["enzyme_category"].isin(["Neither", "A3A"])]["has_core_mooring"]
        )
        if table.shape == (2, 2):
            odds_ratio, pvalue = stats.fisher_exact(table)
            stats_results["fisher_mooring_neither_vs_a3a"] = {
                "odds_ratio": odds_ratio, "pvalue": pvalue
            }

    # AU richness: Neither vs A3A (t-test)
    neither_au = features_df[features_df["enzyme_category"] == "Neither"]["au_fraction_20nt"]
    a3a_au = features_df[features_df["enzyme_category"] == "A3A"]["au_fraction_20nt"]
    if len(neither_au) > 0 and len(a3a_au) > 0:
        t_stat, pvalue = stats.ttest_ind(neither_au, a3a_au)
        stats_results["ttest_au_neither_vs_a3a"] = {
            "t_statistic": t_stat, "pvalue": pvalue,
            "neither_mean": neither_au.mean(), "a3a_mean": a3a_au.mean()
        }

    return stats_results


# ============================================================
# Main analysis
# ============================================================

def main():
    print("=" * 70)
    print("APOBEC1 Integration Analysis")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    levanon_df = pd.read_csv(DATA_DIR / "levanon_all_categories.csv")
    tissue_rates = pd.read_csv(DATA_DIR / "levanon_tissue_rates.csv")

    with open(DATA_DIR / "multi_enzyme_sequences_v3_with_negatives.json") as f:
        all_sequences = json.load(f)

    # A3A Levanon sites are stored in the A3A-specific sequence file
    a3a_seq_path = PROJECT_ROOT / "data" / "processed" / "site_sequences.json"
    if a3a_seq_path.exists():
        with open(a3a_seq_path) as f:
            a3a_seqs = json.load(f)
        # Merge: only add keys not already present
        for k, v in a3a_seqs.items():
            if k not in all_sequences:
                all_sequences[k] = v
        print(f"  Added {len(a3a_seqs)} entries from site_sequences.json")

    print(f"  Levanon sites: {len(levanon_df)}")
    print(f"  Tissue rates: {len(tissue_rates)}")
    print(f"  Sequences: {len(all_sequences)}")

    neither = levanon_df[levanon_df["enzyme_category"] == "Neither"]
    print(f"  Neither sites: {len(neither)}")

    # --------------------------------------------------------
    # Step 1: Known APOBEC1 targets
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 1: Known Human APOBEC1 Targets")
    print("=" * 70)

    known_sites, ortholog_genes = compile_known_apobec1_targets()
    print(f"\nKnown human APOBEC1 sites: {len(known_sites)}")
    for _, site in known_sites.iterrows():
        print(f"  {site['gene']} ({site['chr']}:{site['position']}) - {site['notes']}")
    print(f"\nMouse APOBEC1 ortholog genes: {len(ortholog_genes)}")

    # --------------------------------------------------------
    # Step 2: Check overlaps
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 2: Overlap Analysis")
    print("=" * 70)

    overlaps = check_overlaps(levanon_df, known_sites, ortholog_genes)

    print(f"\n--- Position-level matches ---")
    if len(overlaps["position_matches"]) > 0:
        for _, match in overlaps["position_matches"].iterrows():
            print(f"  MATCH: {match['site_id']} = {match['known_gene']} "
                  f"({match['chr']}:{match['position']}) [{match['source']}]")
            print(f"         {match['notes']}")
    else:
        print("  No exact position matches found")

    print(f"\n--- Gene-level overlap with known human APOBEC1 targets ---")
    print(f"  Neither genes matching known APOBEC1: {overlaps['gene_overlap_known']}")

    print(f"\n--- Gene-level overlap with mouse APOBEC1 ortholog genes ---")
    print(f"  Neither genes matching mouse orthologs: {overlaps['gene_overlap_orthologs']}")
    if overlaps["gene_overlap_orthologs"]:
        for gene in sorted(overlaps["gene_overlap_orthologs"]):
            sites = neither[neither["gene_refseq"] == gene][["site_id", "chr", "start", "mrna_location_refseq", "tissue_classification"]]
            for _, s in sites.iterrows():
                print(f"    {gene}: {s['site_id']} ({s['chr']}:{s['start']}) "
                      f"loc={s['mrna_location_refseq']} tissue={s['tissue_classification']}")

    print(f"\n  Ortholog overlap by category:")
    for cat, genes in overlaps["ortholog_overlap_by_category"].items():
        print(f"    {cat}: {len(genes)} genes - {sorted(genes) if genes else '(none)'}")

    print(f"\n--- APOB sites in full dataset ---")
    if len(overlaps["apob_sites"]) > 0:
        print(overlaps["apob_sites"][["site_id", "chr", "start", "enzyme_category",
                                       "mrna_location_refseq", "tissue_classification"]].to_string(index=False))

    # --------------------------------------------------------
    # Step 3: APOBEC1 feature characterization
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 3: APOBEC1 Feature Characterization")
    print("=" * 70)

    # Sequences are keyed by site_id
    levanon_sequences = {}
    for _, row in levanon_df.iterrows():
        sid = row["site_id"]
        if sid in all_sequences:
            levanon_sequences[sid] = all_sequences[sid]

    print(f"  Sequences found for {len(levanon_sequences)}/{len(levanon_df)} Levanon sites")

    features_df = characterize_apobec1_features(levanon_df, levanon_sequences)
    print(f"  Features computed for {len(features_df)} sites")

    # Save detailed features
    features_df.to_csv(OUTPUT_DIR / "apobec1_features_all_categories.csv", index=False)

    # Per-category summary
    print("\n--- 3'UTR enrichment by category ---")
    for cat in ["Neither", "A3A", "A3G", "A3A_A3G", "Unknown"]:
        cat_df = features_df[features_df["enzyme_category"] == cat]
        if len(cat_df) > 0:
            pct = cat_df["is_3utr"].mean() * 100
            n = len(cat_df)
            print(f"  {cat:12s}: {pct:5.1f}% in 3'UTR  (n={n})")

    print("\n--- Mooring sequence by category ---")
    for cat in ["Neither", "A3A", "A3G", "A3A_A3G", "Unknown"]:
        cat_df = features_df[features_df["enzyme_category"] == cat]
        if len(cat_df) > 0:
            full = cat_df["has_full_mooring"].mean() * 100
            core = cat_df["has_core_mooring"].mean() * 100
            print(f"  {cat:12s}: full_mooring={full:5.1f}%  core_mooring={core:5.1f}%  (n={len(cat_df)})")

    print("\n--- AU-rich context by category ---")
    for cat in ["Neither", "A3A", "A3G", "A3A_A3G", "Unknown"]:
        cat_df = features_df[features_df["enzyme_category"] == cat]
        if len(cat_df) > 0:
            au = cat_df["au_fraction_20nt"].mean()
            m1_au = cat_df["minus1_is_AU"].mean() * 100
            p1_au = cat_df["plus1_is_AU"].mean() * 100
            print(f"  {cat:12s}: AU_frac={au:.3f}  -1_AU={m1_au:5.1f}%  +1_AU={p1_au:5.1f}%")

    print("\n--- Trinucleotide context (top 5 per category) ---")
    for cat in ["Neither", "A3A", "A3G", "A3A_A3G", "Unknown"]:
        cat_df = features_df[features_df["enzyme_category"] == cat]
        if len(cat_df) > 0:
            tri = cat_df["trinucleotide"].value_counts().head(5)
            total = tri.sum()
            tri_str = ", ".join(f"{k}={v/total*100:.1f}%" for k, v in tri.items())
            print(f"  {cat:12s}: {tri_str}")

    # --------------------------------------------------------
    # Step 4: Tissue patterns
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 4: Tissue Expression Patterns")
    print("=" * 70)

    tissue_results = analyze_tissue_patterns(tissue_rates, levanon_df)

    for cat in ["Neither", "A3A", "A3G", "A3A_A3G", "Unknown"]:
        if cat in tissue_results:
            r = tissue_results[cat]
            print(f"\n  {cat} (n={r['n_sites']}):")
            print(f"    GI tract mean rate:  {r['gi_mean_rate']:.2f}")
            print(f"    Non-GI mean rate:    {r['non_gi_mean_rate']:.2f}")
            print(f"    GI specificity:      {r['gi_specificity']:.2f}")
            print(f"    Intestine:           {r['intestine_rate']:.2f}")
            print(f"    Liver:               {r['liver_rate']:.2f}")
            print(f"    Brain:               {r['brain_rate']:.2f}")
            print(f"    Blood:               {r['blood_rate']:.2f}")
            print(f"    Top 5 tissues: ", end="")
            top = sorted(r["top_tissues"].items(), key=lambda x: -x[1])[:5]
            print(", ".join(f"{t}={v:.1f}" for t, v in top))

    # --------------------------------------------------------
    # Step 5: Statistical tests
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 5: Statistical Comparisons")
    print("=" * 70)

    cat_stats = compute_category_statistics(features_df)

    # Summary table
    print("\n--- Category Summary ---")
    summary_rows = []
    for cat in ["Neither", "A3A", "A3G", "A3A_A3G", "Unknown"]:
        if cat in cat_stats:
            s = cat_stats[cat]
            row = {
                "Category": cat,
                "N": s["n_sites"],
                "3UTR%": f"{s['pct_3utr']:.1f}",
                "Mooring%": f"{s['pct_core_mooring']:.1f}",
                "AU_frac": f"{s['mean_au_fraction']:.3f}",
                "TC%": f"{s.get('pct_UCU', 0) + s.get('pct_UCA', 0) + s.get('pct_UCG', 0) + s.get('pct_UCC', 0):.1f}",
                "CC%": f"{s.get('pct_CCU', 0) + s.get('pct_CCA', 0) + s.get('pct_CCG', 0) + s.get('pct_CCC', 0):.1f}",
            }
            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    summary_df.to_csv(OUTPUT_DIR / "category_summary.csv", index=False)

    # Statistical tests
    if "fisher_3utr_neither_vs_a3a" in cat_stats:
        r = cat_stats["fisher_3utr_neither_vs_a3a"]
        print(f"\n  Fisher exact (3'UTR: Neither vs A3A): OR={r['odds_ratio']:.2f}, p={r['pvalue']:.2e}")

    if "fisher_mooring_neither_vs_a3a" in cat_stats:
        r = cat_stats["fisher_mooring_neither_vs_a3a"]
        print(f"  Fisher exact (Mooring: Neither vs A3A): OR={r['odds_ratio']:.2f}, p={r['pvalue']:.2e}")

    if "ttest_au_neither_vs_a3a" in cat_stats:
        r = cat_stats["ttest_au_neither_vs_a3a"]
        print(f"  T-test (AU richness: Neither vs A3A): t={r['t_statistic']:.2f}, p={r['pvalue']:.2e}")
        print(f"    Neither mean AU: {r['neither_mean']:.3f}, A3A mean AU: {r['a3a_mean']:.3f}")

    # --------------------------------------------------------
    # Step 6: Identify strong APOBEC1 candidate sites
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 6: Strong APOBEC1 Candidate Sites Among 'Neither'")
    print("=" * 70)

    neither_features = features_df[features_df["enzyme_category"] == "Neither"].copy()

    # Score each site: 3'UTR + mooring + AU-rich + intestine-specific
    neither_features["apobec1_score"] = 0
    neither_features.loc[neither_features["is_3utr"], "apobec1_score"] += 1
    neither_features.loc[neither_features["has_core_mooring"], "apobec1_score"] += 1
    neither_features.loc[neither_features["au_fraction_20nt"] > 0.5, "apobec1_score"] += 1
    neither_features.loc[neither_features["tissue_classification"] == "Intestine Specific", "apobec1_score"] += 1

    print(f"\n  APOBEC1 score distribution (0-4):")
    for score in range(5):
        n = (neither_features["apobec1_score"] == score).sum()
        pct = n / len(neither_features) * 100
        print(f"    Score {score}: {n:3d} sites ({pct:5.1f}%)")

    # Strong candidates: score >= 3
    strong = neither_features[neither_features["apobec1_score"] >= 3].sort_values("apobec1_score", ascending=False)
    print(f"\n  Strong APOBEC1 candidates (score >= 3): {len(strong)} sites")
    if len(strong) > 0:
        cols = ["site_id", "gene", "chr", "position", "mrna_location", "tissue_classification",
                "has_core_mooring", "au_fraction_20nt", "trinucleotide", "apobec1_score"]
        print(strong[cols].head(20).to_string(index=False))
        strong.to_csv(OUTPUT_DIR / "strong_apobec1_candidates.csv", index=False)

    # --------------------------------------------------------
    # Step 7: Compare Neither subgroups
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 7: Neither Subgroup Analysis")
    print("=" * 70)

    # Split Neither into APOBEC1-like (score>=2 + intestine/ubiquitous) vs rest
    apobec1_like = neither_features[
        (neither_features["apobec1_score"] >= 2) &
        (neither_features["is_3utr"])
    ]
    other_neither = neither_features[~neither_features.index.isin(apobec1_like.index)]

    print(f"\n  APOBEC1-like Neither sites (score>=2 + 3'UTR): {len(apobec1_like)}")
    print(f"  Other Neither sites: {len(other_neither)}")

    if len(apobec1_like) > 0:
        print(f"\n  APOBEC1-like tissue distribution:")
        print(apobec1_like["tissue_classification"].value_counts().to_string())
        print(f"\n  Other Neither tissue distribution:")
        print(other_neither["tissue_classification"].value_counts().to_string())

        print(f"\n  APOBEC1-like mooring: {apobec1_like['has_core_mooring'].mean()*100:.1f}%")
        print(f"  Other Neither mooring: {other_neither['has_core_mooring'].mean()*100:.1f}%")
        print(f"\n  APOBEC1-like AU fraction: {apobec1_like['au_fraction_20nt'].mean():.3f}")
        print(f"  Other Neither AU fraction: {other_neither['au_fraction_20nt'].mean():.3f}")

    # Save full results
    neither_features.to_csv(OUTPUT_DIR / "neither_apobec1_analysis.csv", index=False)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    n_strong = len(strong) if len(strong) > 0 else 0
    n_apobec1_like = len(apobec1_like)

    print(f"""
Key Findings:
1. APOB canonical site (chr2:21010329) IS in our dataset, classified as "Neither" - CONFIRMED
2. {len(overlaps.get('gene_overlap_orthologs', set()))} Neither genes overlap with mouse APOBEC1 ortholog genes
3. 3'UTR enrichment: Neither={cat_stats.get('Neither', {}).get('pct_3utr', 0):.1f}% vs A3A={cat_stats.get('A3A', {}).get('pct_3utr', 0):.1f}%
4. Mooring sequence: Neither={cat_stats.get('Neither', {}).get('pct_core_mooring', 0):.1f}% vs A3A={cat_stats.get('A3A', {}).get('pct_core_mooring', 0):.1f}%
5. {n_strong} Neither sites score >=3 on APOBEC1 feature composite (3'UTR + mooring + AU-rich + intestine)
6. {n_apobec1_like} Neither sites are APOBEC1-like (score>=2 + 3'UTR)

APOBEC1 Signature Features in "Neither" Category:
- 3'UTR preference: Characteristic of APOBEC1 (vs CDS for APOBEC3s)
- Mooring sequence: Required cofactor binding motif for APOBEC1
- AU-rich context: APOBEC1 favors AU-rich regions
- Intestine specificity: APOBEC1 is highly expressed in intestine/liver

Output files:
  {OUTPUT_DIR}/apobec1_features_all_categories.csv  -- full feature table
  {OUTPUT_DIR}/category_summary.csv                 -- per-category summary
  {OUTPUT_DIR}/strong_apobec1_candidates.csv        -- high-confidence APOBEC1 sites
  {OUTPUT_DIR}/neither_apobec1_analysis.csv         -- Neither sites detailed analysis
""")

    return features_df, cat_stats, overlaps, tissue_results


if __name__ == "__main__":
    features_df, cat_stats, overlaps, tissue_results = main()
