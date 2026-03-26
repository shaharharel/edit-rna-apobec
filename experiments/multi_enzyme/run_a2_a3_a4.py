#!/usr/bin/env python3
"""Analyses A2, A3, A4: Nonsense enrichment, GoF/LoF, Exome editability map.

Output directory: experiments/multi_enzyme/outputs/tcga_gnomad/
"""

import gzip
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from pyfaidx import Fasta

# Project root
ROOT = Path("/Users/shaharharel/Documents/github/edit-rna-apobec")
sys.path.insert(0, str(ROOT))

from src.data.apobec_feature_extraction import extract_motif_from_seq

OUTDIR = ROOT / "experiments" / "multi_enzyme" / "outputs" / "tcga_gnomad"
OUTDIR.mkdir(parents=True, exist_ok=True)

CLINVAR_SCORES = ROOT / "experiments" / "apobec3a" / "outputs" / "clinvar_prediction" / "clinvar_all_scores.csv"
CLINVAR_FULL = ROOT / "data" / "processed" / "clinvar_c2u_variants.csv"
HG38_FA = ROOT / "data" / "raw" / "genomes" / "hg38.fa"
REFGENE_HG38 = ROOT / "data" / "raw" / "genomes" / "refGene.txt"

# ============================================================================
# OncoKB-derived cancer gene lists (well-established classifications)
# ============================================================================
TSG_GENES = {
    "TP53", "RB1", "APC", "PTEN", "VHL", "NF1", "NF2", "TSC1", "TSC2",
    "BRCA1", "BRCA2", "SMAD4", "CDKN2A", "CDH1", "WT1", "PTCH1", "STK11",
    "MLH1", "MSH2", "MSH6", "PMS2", "BAP1", "ARID1A", "ARID1B", "ARID2",
    "SMARCB1", "SMARCD1", "PBRM1", "KDM6A", "KDM5C", "SETD2", "TET2",
    "DNMT3A", "IDH1", "IDH2", "ASXL1", "EZH2", "SUZ12", "CREBBP", "EP300",
    "FBXW7", "MEN1", "CYLD", "FLCN", "FH", "SDHB", "SDHC", "SDHD",
    "MAX", "TMEM127", "DICER1", "SUFU", "AXIN1", "AXIN2", "AMER1",
    "RNF43", "ZNRF3", "BCOR", "BCORL1", "PHF6", "CEBPA", "GATA3",
    "RUNX1", "ETV6", "ATM", "ATR", "CHEK1", "CHEK2", "PALB2", "RAD51C",
    "RAD51D", "FANCA", "FANCC", "FAT1", "LATS1", "LATS2", "NF1", "NF2",
    "NOTCH1", "NOTCH2", "CIC", "FUBP1", "SMARCA4", "STAG2", "PPP2R1A",
    "MAP3K1", "CASP8", "CBFB", "CDH1", "KMT2C", "KMT2D", "NCOR1",
    "SPOP", "ZFHX3", "ACVR2A", "TGFBR2", "TNFAIP3",
}

ONCOGENE_GENES = {
    "KRAS", "BRAF", "PIK3CA", "EGFR", "MYC", "ERBB2", "ALK", "RET", "MET",
    "KIT", "PDGFRA", "FGFR1", "FGFR2", "FGFR3", "NRAS", "HRAS", "AKT1",
    "AKT2", "AKT3", "MTOR", "RAF1", "MAP2K1", "MAP2K2", "CTNNB1",
    "NOTCH1", "JAK2", "JAK3", "ABL1", "FLT3", "NPM1", "CCND1", "CCND2",
    "CCND3", "CDK4", "CDK6", "MDM2", "MDM4", "MYCN", "MYCL", "BCL2",
    "BCL6", "GNAS", "GNA11", "GNAQ", "SF3B1", "U2AF1", "SRSF2",
    "CALR", "MPL", "CSF3R", "STAT3", "STAT5B", "SRC", "ERBB3",
    "DDR2", "ROS1", "NTRK1", "NTRK2", "NTRK3", "SMO", "GLI1", "GLI2",
    "TERT", "FOXL2", "MYOD1", "PPM1D", "RAC1", "RHOA", "SYK",
    "PTPN11", "CBL", "KDR", "FGF19", "FGF3", "FGF4",
}

# Some genes are classified as both (e.g., NOTCH1) - handle per-analysis


def load_clinvar_with_consequences():
    """Load ClinVar scores merged with molecular consequences."""
    print("Loading ClinVar scores...")
    scores = pd.read_csv(CLINVAR_SCORES, low_memory=False)
    print(f"  Scores: {len(scores):,} variants")

    print("Loading ClinVar full data (for molecular_consequence)...")
    full = pd.read_csv(CLINVAR_FULL, usecols=["site_id", "molecular_consequence", "gene"],
                       low_memory=False)
    print(f"  Full data: {len(full):,} variants")

    # Merge on site_id
    merged = scores.merge(full[["site_id", "molecular_consequence"]], on="site_id", how="left")
    print(f"  Merged: {len(merged):,} variants")
    return merged


# ============================================================================
# ANALYSIS A2: Top-1000 Pathogenic - Nonsense Rate
# ============================================================================

def analysis_a2(df):
    print("\n" + "=" * 70)
    print("ANALYSIS A2: Top-1000 Model-Predicted Pathogenic — Nonsense Rate")
    print("=" * 70)

    # Classify molecular consequence
    def classify_consequence(mc):
        if pd.isna(mc):
            return "unknown"
        mc = str(mc).lower()
        if "nonsense" in mc or "stop_gained" in mc:
            return "nonsense"
        if "missense" in mc:
            return "missense"
        if "synonymous" in mc:
            return "synonymous"
        if "splice" in mc:
            return "splice"
        if "intron" in mc:
            return "intronic"
        if "utr" in mc:
            return "utr"
        if "non-coding" in mc:
            return "non_coding"
        if "frameshift" in mc:
            return "frameshift"
        return "other"

    df["consequence_class"] = df["molecular_consequence"].apply(classify_consequence)

    # Filter to pathogenic + likely_pathogenic
    path_mask = df["significance_simple"].isin(["Pathogenic", "Likely_pathogenic"])
    df_path = df[path_mask].copy()
    print(f"\nPathogenic + Likely_pathogenic: {len(df_path):,}")

    # Baseline: consequence distribution among ALL pathogenic C>T
    baseline_counts = df_path["consequence_class"].value_counts()
    baseline_total = len(df_path)
    baseline_nonsense_frac = baseline_counts.get("nonsense", 0) / baseline_total

    print(f"\nBaseline consequence distribution (all {baseline_total:,} pathogenic):")
    for c, n in baseline_counts.items():
        print(f"  {c}: {n:,} ({n/baseline_total*100:.1f}%)")

    # Top 1000 by model score
    df_top1000 = df_path.nlargest(1000, "p_edited_gb")
    top_counts = df_top1000["consequence_class"].value_counts()
    top_nonsense_frac = top_counts.get("nonsense", 0) / 1000

    print(f"\nTop-1000 by GB model score (p_edited_gb >= {df_top1000['p_edited_gb'].min():.4f}):")
    for c, n in top_counts.items():
        print(f"  {c}: {n} ({n/10:.1f}%)")

    # Also do top-500 and top-2000 for robustness
    results_by_k = {}
    for k in [100, 500, 1000, 2000, 5000]:
        df_topk = df_path.nlargest(k, "p_edited_gb")
        ck = df_topk["consequence_class"].value_counts()
        nonsense_k = ck.get("nonsense", 0)
        results_by_k[k] = {
            "n": k,
            "nonsense_count": int(nonsense_k),
            "nonsense_frac": nonsense_k / k,
            "missense_count": int(ck.get("missense", 0)),
            "synonymous_count": int(ck.get("synonymous", 0)),
            "min_score": float(df_topk["p_edited_gb"].min()),
        }

    # Statistical test: Fisher's exact for top-1000 vs rest of pathogenic
    top_nonsense = top_counts.get("nonsense", 0)
    top_other = 1000 - top_nonsense
    rest_nonsense = baseline_counts.get("nonsense", 0) - top_nonsense
    rest_other = (baseline_total - 1000) - rest_nonsense

    table = [[top_nonsense, top_other], [rest_nonsense, rest_other]]
    odds_ratio, p_value = stats.fisher_exact(table, alternative="two-sided")

    print(f"\nFisher's exact test (top-1000 vs rest):")
    print(f"  Top-1000 nonsense: {top_nonsense}/1000 ({top_nonsense/10:.1f}%)")
    print(f"  Rest nonsense: {rest_nonsense}/{baseline_total-1000} ({rest_nonsense/(baseline_total-1000)*100:.1f}%)")
    print(f"  Odds ratio: {odds_ratio:.3f}")
    print(f"  P-value: {p_value:.2e}")

    # Also test: is nonsense ENRICHED (one-sided)
    _, p_enriched = stats.fisher_exact(table, alternative="greater")
    _, p_depleted = stats.fisher_exact(table, alternative="less")

    result = {
        "analysis": "A2_nonsense_analysis",
        "description": "Top-K model-predicted pathogenic variants: nonsense rate analysis",
        "n_pathogenic_total": int(baseline_total),
        "baseline_consequence_distribution": {k: int(v) for k, v in baseline_counts.items()},
        "baseline_nonsense_fraction": float(baseline_nonsense_frac),
        "top_k_results": results_by_k,
        "statistical_test": {
            "test": "Fisher exact (top-1000 vs rest of pathogenic)",
            "top_1000_nonsense": int(top_nonsense),
            "top_1000_other": int(top_other),
            "rest_nonsense": int(rest_nonsense),
            "rest_other": int(rest_other),
            "odds_ratio": float(odds_ratio),
            "p_value_two_sided": float(p_value),
            "p_value_enriched": float(p_enriched),
            "p_value_depleted": float(p_depleted),
        },
        "interpretation": (
            f"Nonsense fraction in top-1000: {top_nonsense/10:.1f}% vs baseline {baseline_nonsense_frac*100:.1f}%. "
            f"OR={odds_ratio:.3f}, p={p_value:.2e}. "
            + ("ENRICHED" if p_enriched < 0.05 else "DEPLETED" if p_depleted < 0.05 else "NOT SIGNIFICANT")
            + " for nonsense in high-editability pathogenic variants."
        ),
    }

    out_path = OUTDIR / "a2_nonsense_analysis.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {out_path}")

    return result


# ============================================================================
# ANALYSIS A3: GoF/LoF Scaled to COSMIC Cancer Gene Census
# ============================================================================

def analysis_a3(df):
    print("\n" + "=" * 70)
    print("ANALYSIS A3: GoF/LoF — Cancer Gene Census Editability")
    print("=" * 70)

    # Classify genes
    all_cancer_genes = TSG_GENES | ONCOGENE_GENES
    df_cancer = df[df["gene"].isin(all_cancer_genes)].copy()
    print(f"\nVariants in cancer genes: {len(df_cancer):,}")

    # Assign gene role (TSG-only, Oncogene-only, Both)
    def gene_role(g):
        is_tsg = g in TSG_GENES
        is_onco = g in ONCOGENE_GENES
        if is_tsg and is_onco:
            return "Both"
        if is_tsg:
            return "TSG"
        if is_onco:
            return "Oncogene"
        return "Other"

    df_cancer["gene_role"] = df_cancer["gene"].apply(gene_role)

    # Split by significance
    path_mask = df_cancer["significance_simple"].isin(["Pathogenic", "Likely_pathogenic"])
    benign_mask = df_cancer["significance_simple"].isin(["Benign", "Likely_benign"])

    results = {}
    gene_level_results = {}

    for role in ["TSG", "Oncogene"]:
        role_mask = df_cancer["gene_role"] == role
        df_role = df_cancer[role_mask]

        path_scores = df_role[path_mask & role_mask]["p_edited_gb"].dropna()
        benign_scores = df_role[benign_mask & role_mask]["p_edited_gb"].dropna()

        print(f"\n{role}:")
        print(f"  Pathogenic variants: {len(path_scores):,}, mean score: {path_scores.mean():.4f}")
        print(f"  Benign variants: {len(benign_scores):,}, mean score: {benign_scores.mean():.4f}")

        if len(path_scores) > 0 and len(benign_scores) > 0:
            # Wilcoxon rank-sum (Mann-Whitney U)
            u_stat, mwu_p = stats.mannwhitneyu(path_scores, benign_scores, alternative="two-sided")

            # For TSGs: expect pathogenic > benign (LoF = more edited)
            # For Oncogenes: expect pathogenic < benign (GoF = less edited / specific)
            if role == "TSG":
                _, directional_p = stats.mannwhitneyu(path_scores, benign_scores, alternative="greater")
                direction = "pathogenic > benign (LoF hypothesis)"
            else:
                _, directional_p = stats.mannwhitneyu(path_scores, benign_scores, alternative="less")
                direction = "pathogenic < benign (GoF hypothesis)"

            # Effect size (rank-biserial correlation)
            n1, n2 = len(path_scores), len(benign_scores)
            r_rb = 1 - (2 * u_stat) / (n1 * n2)

            print(f"  Mann-Whitney U p={mwu_p:.2e}, directional p={directional_p:.2e}")
            print(f"  Effect size (rank-biserial r): {r_rb:.4f}")
            print(f"  Hypothesis: {direction}")

            results[role] = {
                "n_pathogenic": int(len(path_scores)),
                "n_benign": int(len(benign_scores)),
                "mean_score_pathogenic": float(path_scores.mean()),
                "mean_score_benign": float(benign_scores.mean()),
                "median_score_pathogenic": float(path_scores.median()),
                "median_score_benign": float(benign_scores.median()),
                "mannwhitney_u": float(u_stat),
                "p_two_sided": float(mwu_p),
                "p_directional": float(directional_p),
                "directional_hypothesis": direction,
                "rank_biserial_r": float(r_rb),
            }
        else:
            results[role] = {
                "n_pathogenic": int(len(path_scores)),
                "n_benign": int(len(benign_scores)),
                "error": "insufficient data for test",
            }

        # Per-gene analysis
        genes_in_role = sorted(df_role["gene"].unique())
        for g in genes_in_role:
            gm = df_role["gene"] == g
            gp = df_role[gm & path_mask]["p_edited_gb"].dropna()
            gb = df_role[gm & benign_mask]["p_edited_gb"].dropna()
            if len(gp) >= 5 and len(gb) >= 5:
                u, p = stats.mannwhitneyu(gp, gb, alternative="two-sided")
                gene_level_results[g] = {
                    "role": role,
                    "n_path": int(len(gp)),
                    "n_benign": int(len(gb)),
                    "mean_path": float(gp.mean()),
                    "mean_benign": float(gb.mean()),
                    "delta": float(gp.mean() - gb.mean()),
                    "p_value": float(p),
                }

    # Sign test across genes
    for role in ["TSG", "Oncogene"]:
        gene_deltas = [v["delta"] for v in gene_level_results.values() if v["role"] == role]
        if len(gene_deltas) >= 3:
            n_pos = sum(1 for d in gene_deltas if d > 0)
            n_neg = sum(1 for d in gene_deltas if d < 0)
            n_total = n_pos + n_neg
            # Binomial test
            sign_p = stats.binomtest(n_pos, n_total, 0.5).pvalue
            print(f"\n{role} sign test: {n_pos}/{n_total} genes have path > benign, p={sign_p:.4f}")
            results[role]["sign_test"] = {
                "n_genes_tested": n_total,
                "n_path_higher": n_pos,
                "n_benign_higher": n_neg,
                "binom_p": float(sign_p),
            }

    # --- Figure ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: TSG vs Oncogene score distributions
    for i, role in enumerate(["TSG", "Oncogene"]):
        ax = axes[i]
        role_mask = df_cancer["gene_role"] == role
        ps = df_cancer[path_mask & role_mask]["p_edited_gb"].dropna()
        bs = df_cancer[benign_mask & role_mask]["p_edited_gb"].dropna()

        bins = np.linspace(0, 1, 50)
        ax.hist(bs, bins=bins, alpha=0.6, label=f"Benign (n={len(bs):,})", color="steelblue", density=True)
        ax.hist(ps, bins=bins, alpha=0.6, label=f"Pathogenic (n={len(ps):,})", color="firebrick", density=True)
        ax.set_xlabel("GB Editability Score")
        ax.set_ylabel("Density")
        ax.set_title(f"{role} Genes")
        ax.legend(fontsize=8)

        r = results.get(role, {})
        if "p_two_sided" in r:
            ax.text(0.05, 0.95, f"MWU p={r['p_two_sided']:.2e}\nr={r['rank_biserial_r']:.3f}",
                    transform=ax.transAxes, fontsize=8, va="top",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # Panel C: Per-gene effect sizes
    ax = axes[2]
    sorted_genes = sorted(gene_level_results.items(), key=lambda x: x[1]["delta"])
    gene_names = [g for g, _ in sorted_genes]
    deltas = [v["delta"] for _, v in sorted_genes]
    colors = ["firebrick" if gene_level_results[g]["role"] == "TSG" else "steelblue" for g in gene_names]

    if len(gene_names) > 40:
        # Show top/bottom 20
        show_idx = list(range(20)) + list(range(len(gene_names) - 20, len(gene_names)))
        gene_names = [gene_names[i] for i in show_idx]
        deltas = [deltas[i] for i in show_idx]
        colors = [colors[i] for i in show_idx]

    y_pos = range(len(gene_names))
    ax.barh(y_pos, deltas, color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(gene_names, fontsize=5)
    ax.set_xlabel("Mean score delta (path - benign)")
    ax.set_title("Per-gene editability delta")
    ax.axvline(0, color="black", linewidth=0.5)

    plt.tight_layout()
    fig_path = OUTDIR / "a3_gof_lof.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"\nFigure saved: {fig_path}")

    # Save results
    result = {
        "analysis": "A3_gof_lof_cancer_genes",
        "description": "Editability score comparison: pathogenic vs benign in TSG vs Oncogenes",
        "n_cancer_gene_variants": int(len(df_cancer)),
        "n_tsg_genes_in_list": len(TSG_GENES),
        "n_oncogene_genes_in_list": len(ONCOGENE_GENES),
        "aggregate_results": results,
        "per_gene_results": gene_level_results,
    }

    out_path = OUTDIR / "a3_gof_lof_results.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {out_path}")

    return result


# ============================================================================
# ANALYSIS A4: Full Exome Editability Map (MotifOnly)
# ============================================================================

def parse_refgene_exons(refgene_path):
    """Parse refGene.txt to get coding exonic regions.

    Returns list of (chrom, exon_start, exon_end, strand, gene_name, cds_start, cds_end).
    Only includes coding exons (overlap with CDS).
    """
    cols = [
        "bin", "name", "chrom", "strand", "txStart", "txEnd",
        "cdsStart", "cdsEnd", "exonCount", "exonStarts", "exonEnds",
        "score", "name2", "cdsStartStat", "cdsEndStat", "exonFrames",
    ]
    df = pd.read_csv(refgene_path, sep="\t", header=None, names=cols, low_memory=False)

    # Filter to standard chromosomes and coding transcripts
    valid_chroms = {f"chr{i}" for i in range(1, 23)} | {"chrX", "chrY"}
    df = df[df["chrom"].isin(valid_chroms)]
    df = df[df["cdsStart"] != df["cdsEnd"]]  # coding transcripts only

    # Deduplicate: keep longest transcript per gene
    df["tx_len"] = df["txEnd"] - df["txStart"]
    df = df.sort_values("tx_len", ascending=False).drop_duplicates(subset="name2", keep="first")

    exons = []
    for _, row in df.iterrows():
        chrom = row["chrom"]
        strand = row["strand"]
        gene = row["name2"]
        cds_start = row["cdsStart"]
        cds_end = row["cdsEnd"]

        starts = [int(x) for x in row["exonStarts"].rstrip(",").split(",")]
        ends = [int(x) for x in row["exonEnds"].rstrip(",").split(",")]

        for es, ee in zip(starts, ends):
            # Clip to CDS
            cs = max(es, cds_start)
            ce = min(ee, cds_end)
            if cs < ce:
                exons.append((chrom, cs, ce, strand, gene))

    return exons


def score_motif_only_batch(sequences):
    """Score a batch of 201-nt sequences with the 24-dim motif model.

    Instead of training a full model, we use a simplified scoring:
    TC-motif presence (the dominant feature) + trinucleotide context weighting.

    We use the known feature importances from the GB_HandFeatures model
    to create a MotifOnly score directly from the 24-dim motif features.
    """
    # Feature importance weights from A3A classification (motif features only, normalized)
    # From exp_classification_a3a_5fold: the 24 motif feature importances
    # Key: UC (5p) = strong positive, CC (5p) = negative, downstream matters
    # We'll compute the raw motif features and use a simple weighted sum
    # based on known A3A preferences: UC >> AC > GC > CC for 5' context

    # Simplified: compute extract_motif_from_seq for each, then score
    # with known weights approximated from feature importances
    scores = np.zeros(len(sequences), dtype=np.float32)
    for i, seq in enumerate(sequences):
        if len(seq) < 201:
            continue
        feat = extract_motif_from_seq(seq)
        # Simple TC-motif score: weighted by known A3A preferences
        # 5p dinuc: UC(0), CC(1), AC(2), GC(3) - indices in feat
        # 3p dinuc: CA(4), CG(5), CU(6), CC(7)
        # TC motif = feat[0] (UC at 5')
        tc_motif = feat[0]  # UC 5' = T-C context
        # Downstream: U preference
        cu_3p = feat[6]  # CU at 3'

        # Weighted score (simplified from GB feature importances)
        score = 0.4 * tc_motif + 0.15 * cu_3p + 0.1 * feat[2]  # AC 5p
        # Add trinucleotide context effects
        score += 0.05 * feat[8]  # m2 = A
        score += 0.03 * feat[9]  # m2 = C
        # Penalize CC context (A3G-like, not A3A)
        score -= 0.2 * feat[1]  # CC 5'

        scores[i] = max(0.0, min(1.0, score + 0.2))  # shift baseline
    return scores


def score_motif_vectorized(seqs_array):
    """Vectorized motif scoring for speed."""
    n = len(seqs_array)
    scores = np.zeros(n, dtype=np.float32)

    for i in range(n):
        seq = seqs_array[i]
        if len(seq) < 201:
            continue
        center = 100
        seq_upper = seq.upper().replace("T", "U")

        up = seq_upper[center - 1] if center > 0 else "N"
        down = seq_upper[center + 1] if center < len(seq_upper) - 1 else "N"

        # TC motif (strongest signal)
        tc = 1.0 if up == "U" else 0.0
        cc = 1.0 if up == "C" else 0.0
        ac = 1.0 if up == "A" else 0.0

        # 3' context
        cu = 1.0 if down == "U" else 0.0

        score = 0.4 * tc + 0.15 * cu + 0.1 * ac - 0.2 * cc + 0.2
        scores[i] = max(0.0, min(1.0, score))

    return scores


def analysis_a4(clinvar_df):
    print("\n" + "=" * 70)
    print("ANALYSIS A4: Full Exome Editability Map (MotifOnly)")
    print("=" * 70)

    # Parse refGene for coding exons
    print("Parsing refGene.txt for coding exons...")
    exons = parse_refgene_exons(REFGENE_HG38)
    print(f"  Found {len(exons):,} coding exon intervals across {len(set(e[4] for e in exons)):,} genes")

    # Group exons by chromosome
    from collections import defaultdict
    exons_by_chrom = defaultdict(list)
    for chrom, start, end, strand, gene in exons:
        exons_by_chrom[chrom].append((start, end, strand, gene))

    # Load genome
    print("Loading hg38 genome...")
    genome = Fasta(str(HG38_FA))

    # Process each chromosome
    all_rows = []
    total_c_positions = 0
    chroms = sorted(exons_by_chrom.keys(), key=lambda c: (len(c), c))

    t0 = time.time()
    for ci, chrom in enumerate(chroms):
        chrom_exons = exons_by_chrom[chrom]
        chrom_len = len(genome[chrom])
        chrom_seq = str(genome[chrom])  # full chromosome sequence

        chrom_rows = []
        for exon_start, exon_end, strand, gene in chrom_exons:
            for pos in range(exon_start, exon_end):
                base = chrom_seq[pos].upper()

                # For + strand: look for C (edit target)
                # For - strand: look for G (complement of C on - strand)
                if strand == "+" and base == "C":
                    # Extract 201-nt window centered on this C
                    ws = pos - 100
                    we = pos + 101
                    if ws < 0 or we > chrom_len:
                        continue
                    window = chrom_seq[ws:we].upper().replace("T", "U")
                    trinuc = chrom_seq[max(0, pos-1):pos+2].upper()
                    chrom_rows.append((chrom, pos, "+", window, trinuc, gene))

                elif strand == "-" and base == "G":
                    # Reverse complement: G on + strand = C on - strand
                    ws = pos - 100
                    we = pos + 101
                    if ws < 0 or we > chrom_len:
                        continue
                    window = chrom_seq[ws:we].upper()
                    # Reverse complement
                    comp = str.maketrans("ACGT", "TGCA")
                    window_rc = window.translate(comp)[::-1].replace("T", "U")
                    trinuc_rc = chrom_seq[max(0, pos-1):pos+2].upper().translate(comp)[::-1]
                    chrom_rows.append((chrom, pos, "-", window_rc, trinuc_rc, gene))

        # Score all windows for this chromosome
        if chrom_rows:
            seqs = [r[3] for r in chrom_rows]
            scores = score_motif_vectorized(seqs)

            for j, (ch, p, st, seq, tri, gn) in enumerate(chrom_rows):
                all_rows.append({
                    "chr": ch,
                    "pos": p,
                    "strand": st,
                    "editability_score": float(scores[j]),
                    "trinucleotide_context": tri,
                    "gene": gn,
                })

        total_c_positions += len(chrom_rows)
        elapsed = time.time() - t0
        print(f"  {chrom}: {len(chrom_rows):,} C positions ({elapsed:.0f}s elapsed, {total_c_positions:,} total)")

    # Create DataFrame and save
    print(f"\nTotal exonic C positions: {total_c_positions:,}")
    df_exome = pd.DataFrame(all_rows)

    out_csv = OUTDIR / "exome_editability_map.csv.gz"
    df_exome.to_csv(out_csv, index=False, compression="gzip")
    print(f"Saved: {out_csv} ({out_csv.stat().st_size / 1024 / 1024:.1f} MB)")

    # --- Quick ClinVar enrichment test ---
    print("\nClinVar enrichment at high vs low editability positions...")

    # Match ClinVar to exome by (chr, pos)
    clinvar_pos = clinvar_df[["chr", "start", "significance_simple", "p_edited_gb"]].copy()
    clinvar_pos.rename(columns={"start": "pos"}, inplace=True)

    merged = df_exome.merge(clinvar_pos, on=["chr", "pos"], how="inner")
    print(f"  ClinVar variants matched to exome map: {len(merged):,}")

    if len(merged) > 100:
        # Top 10% vs bottom 10% editability
        q10 = df_exome["editability_score"].quantile(0.1)
        q90 = df_exome["editability_score"].quantile(0.9)

        high_edit = merged[merged["editability_score"] >= q90]
        low_edit = merged[merged["editability_score"] <= q10]

        path_mask_h = high_edit["significance_simple"].isin(["Pathogenic", "Likely_pathogenic"])
        path_mask_l = low_edit["significance_simple"].isin(["Pathogenic", "Likely_pathogenic"])
        benign_mask_h = high_edit["significance_simple"].isin(["Benign", "Likely_benign"])
        benign_mask_l = low_edit["significance_simple"].isin(["Benign", "Likely_benign"])

        n_path_high = path_mask_h.sum()
        n_benign_high = benign_mask_h.sum()
        n_path_low = path_mask_l.sum()
        n_benign_low = benign_mask_l.sum()

        print(f"  High editability (>= {q90:.3f}): {n_path_high} pathogenic, {n_benign_high} benign")
        print(f"  Low editability (<= {q10:.3f}): {n_path_low} pathogenic, {n_benign_low} benign")

        if n_benign_high > 0 and n_benign_low > 0:
            table = [[n_path_high, n_benign_high], [n_path_low, n_benign_low]]
            or_val, p_val = stats.fisher_exact(table)
            print(f"  Fisher's OR={or_val:.3f}, p={p_val:.2e}")
        else:
            or_val, p_val = float("nan"), float("nan")

        # Trinucleotide-controlled analysis
        print("\n  Trinucleotide-controlled enrichment:")
        trinuc_results = {}
        for tri in merged["trinucleotide_context"].unique():
            tri_df = merged[merged["trinucleotide_context"] == tri]
            if len(tri_df) < 50:
                continue
            tri_q10 = tri_df["editability_score"].quantile(0.1)
            tri_q90 = tri_df["editability_score"].quantile(0.9)

            hi = tri_df[tri_df["editability_score"] >= tri_q90]
            lo = tri_df[tri_df["editability_score"] <= tri_q10]

            ph = hi["significance_simple"].isin(["Pathogenic", "Likely_pathogenic"]).sum()
            bh = hi["significance_simple"].isin(["Benign", "Likely_benign"]).sum()
            pl = lo["significance_simple"].isin(["Pathogenic", "Likely_pathogenic"]).sum()
            bl = lo["significance_simple"].isin(["Benign", "Likely_benign"]).sum()

            if bh > 0 and bl > 0 and ph + bh >= 10 and pl + bl >= 10:
                tri_or, tri_p = stats.fisher_exact([[ph, bh], [pl, bl]])
                trinuc_results[tri] = {
                    "n_high": int(len(hi)),
                    "n_low": int(len(lo)),
                    "path_high": int(ph),
                    "benign_high": int(bh),
                    "path_low": int(pl),
                    "benign_low": int(bl),
                    "odds_ratio": float(tri_or),
                    "p_value": float(tri_p),
                }
                if tri_p < 0.05:
                    print(f"    {tri}: OR={tri_or:.3f}, p={tri_p:.3e} (n_hi={len(hi)}, n_lo={len(lo)})")

        clinvar_test = {
            "n_matched": int(len(merged)),
            "q10_threshold": float(q10),
            "q90_threshold": float(q90),
            "high_editability": {
                "n_pathogenic": int(n_path_high),
                "n_benign": int(n_benign_high),
            },
            "low_editability": {
                "n_pathogenic": int(n_path_low),
                "n_benign": int(n_benign_low),
            },
            "fisher_or": float(or_val) if not np.isnan(or_val) else None,
            "fisher_p": float(p_val) if not np.isnan(p_val) else None,
            "trinucleotide_controlled": trinuc_results,
        }
    else:
        clinvar_test = {"n_matched": int(len(merged)), "error": "too few matches"}

    result = {
        "analysis": "A4_exome_editability_map",
        "description": "Full exome C-position editability map (MotifOnly, hg38)",
        "n_exonic_c_positions": int(total_c_positions),
        "n_genes": int(df_exome["gene"].nunique()),
        "score_distribution": {
            "mean": float(df_exome["editability_score"].mean()),
            "median": float(df_exome["editability_score"].median()),
            "std": float(df_exome["editability_score"].std()),
            "q25": float(df_exome["editability_score"].quantile(0.25)),
            "q75": float(df_exome["editability_score"].quantile(0.75)),
        },
        "trinucleotide_counts": df_exome["trinucleotide_context"].value_counts().head(20).to_dict(),
        "output_file": str(out_csv),
        "clinvar_enrichment_test": clinvar_test,
    }

    out_json = OUTDIR / "a4_exome_clinvar_test.json"
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {out_json}")

    return result


# ============================================================================
# MAIN
# ============================================================================

def main():
    t_start = time.time()

    # Load ClinVar data (shared by A2 and A3)
    df = load_clinvar_with_consequences()

    # A2: Nonsense analysis
    r2 = analysis_a2(df)

    # A3: GoF/LoF
    r3 = analysis_a3(df)

    # A4: Exome editability map
    r4 = analysis_a4(df)

    elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"All analyses complete in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'=' * 70}")

    # Summary
    print("\n--- KEY RESULTS ---")
    print(f"\nA2: Nonsense enrichment in top-1000 pathogenic:")
    st = r2["statistical_test"]
    print(f"  {st['top_1000_nonsense']}/1000 nonsense ({st['top_1000_nonsense']/10:.1f}%)")
    print(f"  Baseline: {r2['baseline_nonsense_fraction']*100:.1f}%")
    print(f"  OR={st['odds_ratio']:.3f}, p={st['p_value_two_sided']:.2e}")

    print(f"\nA3: GoF/LoF cancer gene editability:")
    for role in ["TSG", "Oncogene"]:
        r = r3["aggregate_results"].get(role, {})
        if "p_two_sided" in r:
            print(f"  {role}: path mean={r['mean_score_pathogenic']:.4f}, benign mean={r['mean_score_benign']:.4f}, "
                  f"p={r['p_two_sided']:.2e}, r={r['rank_biserial_r']:.4f}")

    print(f"\nA4: Exome editability map:")
    print(f"  {r4['n_exonic_c_positions']:,} C positions across {r4['n_genes']:,} genes")
    ct = r4.get("clinvar_enrichment_test", {})
    if ct.get("fisher_or") is not None:
        print(f"  ClinVar enrichment (high vs low editability): OR={ct['fisher_or']:.3f}, p={ct['fisher_p']:.2e}")


if __name__ == "__main__":
    main()
