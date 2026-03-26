#!/usr/bin/env python3
"""C2: CpG-stratified TCGA enrichment analysis.

Tests whether TCGA somatic mutation enrichment at predicted editing sites
survives when stratified by CpG context. CpG sites have elevated mutation
rates (deamination of 5-methylcytosine) independent of APOBEC editing,
so demonstrating enrichment in non-CpG context strengthens the APOBEC link.

Approach:
  1. Re-parse MAF files to get C>T/G>A mutations with CONTEXT column
  2. Determine CpG status from the 11-nt CONTEXT (center is at index 5)
  3. Reconstruct the exact same mutation set + matched controls as in
     tcga_full_model_enrichment.py (same seed, same filtering)
  4. Annotate each entry in the raw_scores CSV with CpG status
  5. Within CpG and non-CpG strata, compute enrichment at percentile thresholds

Requires completed run of tcga_full_model_enrichment.py.

Usage:
    conda run -n quris python scripts/multi_enzyme/tcga_cpg_stratified.py
"""

import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyfaidx import Fasta
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = PROJECT_ROOT / "data"
HG19_FA = DATA_DIR / "raw/genomes/hg19.fa"
REFGENE_HG19 = DATA_DIR / "raw/genomes/refGene_hg19.txt"
TCGA_CACHE = DATA_DIR / "raw/tcga"
SCORES_DIR = PROJECT_ROOT / "experiments/multi_enzyme/outputs/tcga_gnomad/raw_scores"
OUTPUT_DIR = PROJECT_ROOT / "experiments/multi_enzyme/outputs/tcga_gnomad/cpg_stratified"

SEED = 42
N_CONTROLS = 5

CANCER_STUDIES = {
    "blca": "blca_tcga_pan_can_atlas_2018",
    "brca": "brca_tcga_pan_can_atlas_2018",
    "cesc": "cesc_tcga_pan_can_atlas_2018",
    "lusc": "lusc_tcga_pan_can_atlas_2018",
}


def parse_ct_mutations(maf_path):
    """Parse C>T and G>A SNPs from MAF, preserving CONTEXT column.

    Returns DataFrame with chrom, pos, strand_inf, gene, sample, context_11nt.
    """
    rows = []
    for chunk in pd.read_csv(maf_path, sep="\t", comment="#", low_memory=False, chunksize=100000):
        if "Variant_Type" in chunk.columns:
            chunk = chunk[chunk["Variant_Type"] == "SNP"]

        ct = (chunk["Reference_Allele"] == "C") & (chunk["Tumor_Seq_Allele2"] == "T")
        ga = (chunk["Reference_Allele"] == "G") & (chunk["Tumor_Seq_Allele2"] == "A")
        sub = chunk[ct | ga].copy()

        if len(sub) > 0:
            chrom = sub["Chromosome"].astype(str)
            if not chrom.str.startswith("chr").any():
                chrom = "chr" + chrom
            sub["chrom"] = chrom
            sub["pos"] = sub["Start_Position"].astype(int) - 1  # 0-based
            sub["strand_inf"] = np.where(sub["Reference_Allele"] == "C", "+", "-")
            sub["gene"] = sub.get("Hugo_Symbol", "unknown")
            sub["sample"] = sub.get("Tumor_Sample_Barcode", "unknown")
            sub["context_11nt"] = sub.get("CONTEXT", "")
            rows.append(sub[["chrom", "pos", "strand_inf", "gene", "sample", "context_11nt"]].copy())

    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)

    # Deduplicate by position (keep unique genomic positions) — same as original
    df_dedup = df.drop_duplicates(subset=["chrom", "pos"]).copy()
    rec = df.groupby(["chrom", "pos"]).size().reset_index(name="recurrence")
    df_dedup = df_dedup.merge(rec, on=["chrom", "pos"], how="left")
    logger.info(f"  {len(df):,} C>T/G>A mutations -> {len(df_dedup):,} unique positions")
    return df_dedup


def get_matched_controls_with_cpg(mutations_df, genome, exons_by_gene, n_controls=N_CONTROLS):
    """Same as original get_matched_controls but also returns CpG status for each control.

    For controls, CpG is determined from the genome sequence at the control position.
    """
    rng = np.random.RandomState(SEED)
    controls = []

    for _, row in mutations_df.iterrows():
        gene = row["gene"]
        gene_exons = exons_by_gene.get(gene, [])
        if not gene_exons:
            continue

        c_positions = []
        for chrom, start, end, strand, _ in gene_exons:
            if chrom != row["chrom"]:
                continue
            try:
                exon_seq = str(genome[chrom][start:end]).upper()
                for i, base in enumerate(exon_seq):
                    pos = start + i
                    if pos == row["pos"]:
                        continue
                    if base == "C":
                        c_positions.append((chrom, pos, "+"))
                    elif base == "G":
                        c_positions.append((chrom, pos, "-"))
            except (KeyError, ValueError):
                continue

        if len(c_positions) >= n_controls:
            chosen = rng.choice(len(c_positions), n_controls, replace=False)
            for idx in chosen:
                ch, p, s = c_positions[idx]
                controls.append({"chrom": ch, "pos": p, "strand_inf": s,
                                 "gene": gene, "is_control": True})
        elif c_positions:
            for ch, p, s in c_positions[:n_controls]:
                controls.append({"chrom": ch, "pos": p, "strand_inf": s,
                                 "gene": gene, "is_control": True})

    return pd.DataFrame(controls) if controls else pd.DataFrame()


def determine_cpg_status_mutation(context_11nt, ref_allele_is_c):
    """Determine if a mutation position is in CpG context.

    For C>T mutations (+ strand): check if position+1 is G (i.e., context[6] == 'G')
    For G>A mutations (- strand): the C is on the other strand, check if position-1 is C
        In terms of the reported context (around G): context[4] == 'C' (position before G)
        This means xCG on the + strand.

    The CONTEXT column is 11-nt centered on the mutated base.
    """
    ctx = str(context_11nt).upper()
    if len(ctx) != 11:
        return False

    if ref_allele_is_c:
        # C at center (index 5), check if next base is G
        return ctx[6] == "G"
    else:
        # G at center (index 5), check if previous base is C
        return ctx[4] == "C"


def determine_cpg_status_control(chrom, pos, strand, genome):
    """Determine CpG status for a control position from genome sequence."""
    try:
        chrom_len = len(genome[chrom])
        if strand == "+":
            # Position is a C on + strand; check if next base is G
            if pos + 1 < chrom_len:
                return str(genome[chrom][pos + 1]).upper() == "G"
        else:
            # Position is a G on + strand (C on - strand); check if previous is C
            if pos - 1 >= 0:
                return str(genome[chrom][pos - 1]).upper() == "C"
    except (KeyError, ValueError):
        pass
    return False


def compute_enrichment_percentile(mut_scores, ctrl_scores,
                                   percentiles=(50, 60, 70, 80, 90, 95)):
    """Compute enrichment ORs at percentile-based thresholds of control distribution."""
    results = {}
    for pct in percentiles:
        if len(ctrl_scores) == 0:
            results[f"p{pct}"] = {"OR": float("nan"), "p": 1.0}
            continue
        threshold = float(np.percentile(ctrl_scores, pct))
        mut_above = int((mut_scores >= threshold).sum())
        mut_below = int((mut_scores < threshold).sum())
        ctrl_above = int((ctrl_scores >= threshold).sum())
        ctrl_below = int((ctrl_scores < threshold).sum())

        if mut_below > 0 and ctrl_above > 0 and ctrl_below > 0:
            table = [[mut_above, mut_below], [ctrl_above, ctrl_below]]
            or_val, p_val = stats.fisher_exact(table)
        else:
            or_val, p_val = float("nan"), 1.0

        results[f"p{pct}"] = {
            "percentile": pct,
            "threshold": round(threshold, 4),
            "OR": round(float(or_val), 4),
            "p": float(p_val),
            "n_mut_above": mut_above,
            "n_mut_total": len(mut_scores),
            "n_ctrl_above": ctrl_above,
            "n_ctrl_total": len(ctrl_scores),
            "frac_mut_above": round(mut_above / max(len(mut_scores), 1), 4),
            "frac_ctrl_above": round(ctrl_above / max(len(ctrl_scores), 1), 4),
        }
    return results


def main():
    t_start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load genome
    logger.info("Loading hg19 genome...")
    genome = Fasta(str(HG19_FA))

    # Parse exons (same as tcga_full_model_enrichment.py)
    logger.info("Parsing exonic regions...")
    exons_by_gene = defaultdict(list)
    with open(REFGENE_HG19) as f:
        seen = set()
        for line in f:
            fields = line.strip().split("\t")
            if len(fields) < 13:
                continue
            chrom, strand = fields[2], fields[3]
            cds_s, cds_e = int(fields[6]), int(fields[7])
            if cds_s == cds_e:
                continue
            gene = fields[12]
            for es_str, ee_str in zip(fields[9].rstrip(",").split(","),
                                       fields[10].rstrip(",").split(",")):
                if not es_str or not ee_str:
                    continue
                es, ee = max(int(es_str), cds_s), min(int(ee_str), cds_e)
                if es >= ee:
                    continue
                key = (chrom, es, ee)
                if key not in seen:
                    seen.add(key)
                    exons_by_gene[gene].append((chrom, es, ee, strand, gene))
    logger.info(f"  {len(exons_by_gene):,} genes with exonic regions")

    all_results = {}

    for cancer in ["blca", "brca", "cesc", "lusc"]:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {cancer.upper()}")

        # Load raw scores
        scores_path = SCORES_DIR / f"{cancer}_scores.csv"
        if not scores_path.exists():
            logger.warning(f"  No scores file for {cancer}")
            continue
        scores_df = pd.read_csv(scores_path)

        # Parse MAF to get mutation positions + CONTEXT
        study = CANCER_STUDIES[cancer]
        maf_path = TCGA_CACHE / f"{study}_mutations.txt"
        if not maf_path.exists():
            logger.warning(f"  No MAF file for {cancer}")
            continue

        mut_df = parse_ct_mutations(maf_path)
        if len(mut_df) == 0:
            continue

        # Generate matched controls (same seed as original)
        ctrl_df = get_matched_controls_with_cpg(mut_df, genome, exons_by_gene, N_CONTROLS)
        logger.info(f"  {len(ctrl_df):,} control positions")

        # Extract sequences to filter valid ones (same as original)
        mut_valid_mask = []
        for _, row in mut_df.iterrows():
            chrom = row["chrom"]
            pos = int(row["pos"])
            strand = row["strand_inf"]
            try:
                chrom_len = len(genome[chrom])
                start = pos - 100
                end = pos + 101
                if start < 0 or end > chrom_len:
                    mut_valid_mask.append(False)
                    continue
                seq = str(genome[chrom][start:end]).upper()
                if strand == "-":
                    comp = str.maketrans("ACGT", "TGCA")
                    seq = seq.translate(comp)[::-1]
                valid = len(seq) == 201 and seq[100] == "C"
                mut_valid_mask.append(valid)
            except (KeyError, ValueError):
                mut_valid_mask.append(False)

        ctrl_valid_mask = []
        for _, row in ctrl_df.iterrows():
            chrom = row["chrom"]
            pos = int(row["pos"])
            strand = row["strand_inf"]
            try:
                chrom_len = len(genome[chrom])
                start = pos - 100
                end = pos + 101
                if start < 0 or end > chrom_len:
                    ctrl_valid_mask.append(False)
                    continue
                seq = str(genome[chrom][start:end]).upper()
                if strand == "-":
                    comp = str.maketrans("ACGT", "TGCA")
                    seq = seq.translate(comp)[::-1]
                valid = len(seq) == 201 and seq[100] == "C"
                ctrl_valid_mask.append(valid)
            except (KeyError, ValueError):
                ctrl_valid_mask.append(False)

        mut_valid_mask = np.array(mut_valid_mask, dtype=bool)
        ctrl_valid_mask = np.array(ctrl_valid_mask, dtype=bool)

        n_mut_valid = mut_valid_mask.sum()
        n_ctrl_valid = ctrl_valid_mask.sum()
        logger.info(f"  Valid: {n_mut_valid:,} mutations, {n_ctrl_valid:,} controls")

        # Check alignment with scores CSV
        n_mut_in_csv = (scores_df["type"] == "mutation").sum()
        n_ctrl_in_csv = (scores_df["type"] == "control").sum()
        logger.info(f"  Scores CSV: {n_mut_in_csv:,} mutations, {n_ctrl_in_csv:,} controls")

        if n_mut_valid != n_mut_in_csv or n_ctrl_valid != n_ctrl_in_csv:
            logger.warning(f"  COUNT MISMATCH: valid ({n_mut_valid},{n_ctrl_valid}) vs "
                           f"CSV ({n_mut_in_csv},{n_ctrl_in_csv})")
            # Use the minimum to avoid index errors
            n_mut = min(n_mut_valid, n_mut_in_csv)
            n_ctrl = min(n_ctrl_valid, n_ctrl_in_csv)
        else:
            n_mut = n_mut_valid
            n_ctrl = n_ctrl_valid

        # Determine CpG status for valid mutations
        valid_mut_df = mut_df[mut_valid_mask].reset_index(drop=True)
        mut_cpg = []
        for _, row in valid_mut_df.iterrows():
            is_c_ref = row["strand_inf"] == "+"
            cpg = determine_cpg_status_mutation(row.get("context_11nt", ""), is_c_ref)
            mut_cpg.append(cpg)
        mut_cpg = np.array(mut_cpg[:n_mut], dtype=bool)

        # Determine CpG status for valid controls
        valid_ctrl_df = ctrl_df[ctrl_valid_mask].reset_index(drop=True)
        ctrl_cpg = []
        for _, row in valid_ctrl_df.iterrows():
            cpg = determine_cpg_status_control(row["chrom"], int(row["pos"]),
                                                row["strand_inf"], genome)
            ctrl_cpg.append(cpg)
        ctrl_cpg = np.array(ctrl_cpg[:n_ctrl], dtype=bool)

        # Get scores
        mut_scores = scores_df[scores_df["type"] == "mutation"]["score"].values[:n_mut]
        ctrl_scores = scores_df[scores_df["type"] == "control"]["score"].values[:n_ctrl]

        logger.info(f"  CpG mutations: {mut_cpg.sum():,}/{n_mut:,} ({100*mut_cpg.mean():.1f}%)")
        logger.info(f"  CpG controls: {ctrl_cpg.sum():,}/{n_ctrl:,} ({100*ctrl_cpg.mean():.1f}%)")

        cancer_results = {
            "cancer": cancer,
            "n_mut": int(n_mut),
            "n_ctrl": int(n_ctrl),
            "n_mut_cpg": int(mut_cpg.sum()),
            "n_mut_noncpg": int((~mut_cpg).sum()),
            "n_ctrl_cpg": int(ctrl_cpg.sum()),
            "n_ctrl_noncpg": int((~ctrl_cpg).sum()),
            "frac_mut_cpg": round(float(mut_cpg.mean()), 4),
            "frac_ctrl_cpg": round(float(ctrl_cpg.mean()), 4),
        }

        # Enrichment: unstratified (all)
        logger.info("  Computing enrichment: all...")
        cancer_results["all"] = compute_enrichment_percentile(mut_scores, ctrl_scores)
        if n_mut > 0 and n_ctrl > 0:
            _, mw_p = stats.mannwhitneyu(mut_scores, ctrl_scores, alternative="two-sided")
            cancer_results["all"]["mann_whitney_p"] = float(mw_p)
            cancer_results["all"]["mean_mut"] = round(float(np.mean(mut_scores)), 4)
            cancer_results["all"]["mean_ctrl"] = round(float(np.mean(ctrl_scores)), 4)
            cancer_results["all"]["delta"] = round(
                float(np.mean(mut_scores) - np.mean(ctrl_scores)), 4)

        # Enrichment: CpG only
        if mut_cpg.sum() > 100 and ctrl_cpg.sum() > 100:
            logger.info(f"  Computing enrichment: CpG only ({mut_cpg.sum()} mut, {ctrl_cpg.sum()} ctrl)...")
            cancer_results["cpg_only"] = compute_enrichment_percentile(
                mut_scores[mut_cpg], ctrl_scores[ctrl_cpg])
            _, mw_p = stats.mannwhitneyu(mut_scores[mut_cpg], ctrl_scores[ctrl_cpg],
                                          alternative="two-sided")
            cancer_results["cpg_only"]["mann_whitney_p"] = float(mw_p)
            cancer_results["cpg_only"]["mean_mut"] = round(float(np.mean(mut_scores[mut_cpg])), 4)
            cancer_results["cpg_only"]["mean_ctrl"] = round(float(np.mean(ctrl_scores[ctrl_cpg])), 4)
            cancer_results["cpg_only"]["delta"] = round(
                float(np.mean(mut_scores[mut_cpg]) - np.mean(ctrl_scores[ctrl_cpg])), 4)
            cancer_results["cpg_only"]["n_mut"] = int(mut_cpg.sum())
            cancer_results["cpg_only"]["n_ctrl"] = int(ctrl_cpg.sum())
        else:
            cancer_results["cpg_only"] = {"note": "Too few CpG sites for stratification"}

        # Enrichment: non-CpG only
        noncpg_mut = ~mut_cpg
        noncpg_ctrl = ~ctrl_cpg
        if noncpg_mut.sum() > 100 and noncpg_ctrl.sum() > 100:
            logger.info(f"  Computing enrichment: non-CpG ({noncpg_mut.sum()} mut, {noncpg_ctrl.sum()} ctrl)...")
            cancer_results["noncpg_only"] = compute_enrichment_percentile(
                mut_scores[noncpg_mut], ctrl_scores[noncpg_ctrl])
            _, mw_p = stats.mannwhitneyu(mut_scores[noncpg_mut], ctrl_scores[noncpg_ctrl],
                                          alternative="two-sided")
            cancer_results["noncpg_only"]["mann_whitney_p"] = float(mw_p)
            cancer_results["noncpg_only"]["mean_mut"] = round(float(np.mean(mut_scores[noncpg_mut])), 4)
            cancer_results["noncpg_only"]["mean_ctrl"] = round(float(np.mean(ctrl_scores[noncpg_ctrl])), 4)
            cancer_results["noncpg_only"]["delta"] = round(
                float(np.mean(mut_scores[noncpg_mut]) - np.mean(ctrl_scores[noncpg_ctrl])), 4)
            cancer_results["noncpg_only"]["n_mut"] = int(noncpg_mut.sum())
            cancer_results["noncpg_only"]["n_ctrl"] = int(noncpg_ctrl.sum())
        else:
            cancer_results["noncpg_only"] = {"note": "Too few non-CpG sites"}

        # Also stratify by TC + CpG (4-way)
        mut_tc = scores_df[scores_df["type"] == "mutation"]["tc_context"].values[:n_mut].astype(bool)
        ctrl_tc = scores_df[scores_df["type"] == "control"]["tc_context"].values[:n_ctrl].astype(bool)

        for tc_label, tc_mut_mask, tc_ctrl_mask in [
            ("tc_cpg", mut_tc & mut_cpg, ctrl_tc & ctrl_cpg),
            ("tc_noncpg", mut_tc & ~mut_cpg, ctrl_tc & ~ctrl_cpg),
            ("nontc_cpg", ~mut_tc & mut_cpg, ~ctrl_tc & ctrl_cpg),
            ("nontc_noncpg", ~mut_tc & ~mut_cpg, ~ctrl_tc & ~ctrl_cpg),
        ]:
            n_m = tc_mut_mask.sum()
            n_c = tc_ctrl_mask.sum()
            if n_m > 50 and n_c > 50:
                logger.info(f"  Computing enrichment: {tc_label} ({n_m} mut, {n_c} ctrl)...")
                cancer_results[tc_label] = compute_enrichment_percentile(
                    mut_scores[tc_mut_mask], ctrl_scores[tc_ctrl_mask])
                cancer_results[tc_label]["n_mut"] = int(n_m)
                cancer_results[tc_label]["n_ctrl"] = int(n_c)
                if n_m > 0 and n_c > 0:
                    _, mw_p = stats.mannwhitneyu(
                        mut_scores[tc_mut_mask], ctrl_scores[tc_ctrl_mask],
                        alternative="two-sided")
                    cancer_results[tc_label]["mann_whitney_p"] = float(mw_p)
                    cancer_results[tc_label]["mean_mut"] = round(float(np.mean(mut_scores[tc_mut_mask])), 4)
                    cancer_results[tc_label]["mean_ctrl"] = round(float(np.mean(ctrl_scores[tc_ctrl_mask])), 4)
            else:
                cancer_results[tc_label] = {"note": f"Too few ({n_m} mut, {n_c} ctrl)"}

        all_results[cancer] = cancer_results

        # Log summary
        logger.info(f"\n  {cancer.upper()} SUMMARY:")
        for strat in ["all", "cpg_only", "noncpg_only", "tc_noncpg"]:
            r = cancer_results.get(strat, {})
            if "p90" in r:
                p90 = r["p90"]
                logger.info(f"    {strat:>15}: OR@p90={p90['OR']:.3f} (p={p90['p']:.2e}), "
                            f"delta={r.get('delta', 'N/A')}")

    # Save results
    with open(OUTPUT_DIR / "cpg_stratified_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # ---- Generate figures ----
    logger.info("\nGenerating figures...")
    cancers = [c for c in ["blca", "brca", "cesc", "lusc"] if c in all_results]

    if not cancers:
        logger.warning("No results to plot")
        return

    # Figure 1: CpG vs non-CpG enrichment by percentile
    fig, axes = plt.subplots(1, len(cancers), figsize=(5 * len(cancers), 5), squeeze=False)

    for col, cancer in enumerate(cancers):
        ax = axes[0, col]
        r = all_results[cancer]

        for strat, color, label in [
            ("all", "#374151", "All"),
            ("cpg_only", "#dc2626", "CpG"),
            ("noncpg_only", "#2563eb", "Non-CpG"),
        ]:
            if strat not in r or "p50" not in r[strat]:
                continue
            pcts = [50, 60, 70, 80, 90, 95]
            ors = [r[strat].get(f"p{p}", {}).get("OR", float("nan")) for p in pcts]
            ax.plot(pcts, ors, marker="o", label=label, color=color, linewidth=2)

        ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel("Control percentile threshold")
        ax.set_ylabel("Odds Ratio")
        ax.set_title(f"{cancer.upper()}")
        ax.legend(fontsize=8)
        ax.set_ylim(0, max(2.5, ax.get_ylim()[1]))

    plt.suptitle("CpG-Stratified TCGA Enrichment at Predicted Editing Sites", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cpg_stratified_enrichment.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Figure 2: 4-way stratification (TC x CpG)
    fig, axes = plt.subplots(1, len(cancers), figsize=(5 * len(cancers), 5), squeeze=False)

    for col, cancer in enumerate(cancers):
        ax = axes[0, col]
        r = all_results[cancer]

        for strat, color, label in [
            ("tc_noncpg", "#2563eb", "TC + non-CpG"),
            ("tc_cpg", "#dc2626", "TC + CpG"),
            ("nontc_noncpg", "#6b7280", "non-TC + non-CpG"),
            ("nontc_cpg", "#f59e0b", "non-TC + CpG"),
        ]:
            if strat not in r or "p50" not in r[strat]:
                continue
            pcts = [50, 60, 70, 80, 90, 95]
            ors = [r[strat].get(f"p{p}", {}).get("OR", float("nan")) for p in pcts]
            ax.plot(pcts, ors, marker="o", label=label, color=color, linewidth=2)

        ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel("Control percentile threshold")
        ax.set_ylabel("Odds Ratio")
        ax.set_title(f"{cancer.upper()}")
        ax.legend(fontsize=7)
        ax.set_ylim(0, max(2.5, ax.get_ylim()[1]))

    plt.suptitle("4-Way Stratified TCGA Enrichment (TC x CpG)", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cpg_4way_stratified.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Figure 3: Summary bar chart — OR at 90th percentile for each stratum
    fig, ax = plt.subplots(figsize=(10, 6))
    strata = ["all", "cpg_only", "noncpg_only", "tc_noncpg"]
    strat_labels = ["All", "CpG", "Non-CpG", "TC+Non-CpG"]
    strat_colors = ["#374151", "#dc2626", "#2563eb", "#10b981"]
    x = np.arange(len(cancers))
    width = 0.2

    for i, (strat, label, color) in enumerate(zip(strata, strat_labels, strat_colors)):
        ors = []
        for cancer in cancers:
            r = all_results[cancer].get(strat, {})
            or_val = r.get("p90", {}).get("OR", float("nan"))
            ors.append(or_val)
        bars = ax.bar(x + i * width - 1.5 * width, ors, width, label=label,
                      color=color, alpha=0.8)
        for bar, or_val in zip(bars, ors):
            if not np.isnan(or_val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        f"{or_val:.2f}", ha="center", fontsize=7, rotation=45)

    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([c.upper() for c in cancers])
    ax.set_ylabel("Odds Ratio at 90th percentile")
    ax.set_title("CpG-Stratified Enrichment Summary")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cpg_summary_bar.png", dpi=150, bbox_inches="tight")
    plt.close()

    elapsed = time.time() - t_start
    logger.info(f"\n{'='*60}")
    logger.info(f"COMPLETE ({elapsed / 60:.1f} min)")
    logger.info(f"Results: {OUTPUT_DIR / 'cpg_stratified_results.json'}")
    logger.info(f"Figures: {OUTPUT_DIR}")

    # Print summary table
    logger.info(f"\n{'='*80}")
    logger.info("SUMMARY: OR at 90th percentile threshold")
    logger.info(f"{'Cancer':<8} {'Stratum':<15} {'OR':>8} {'p':>12} {'n_mut':>8} {'n_ctrl':>8}")
    logger.info("-" * 65)
    for cancer in cancers:
        for strat in ["all", "cpg_only", "noncpg_only", "tc_noncpg"]:
            r = all_results[cancer].get(strat, {})
            p90 = r.get("p90", {})
            logger.info(f"{cancer.upper():<8} {strat:<15} "
                        f"{p90.get('OR', 'N/A'):>8} {p90.get('p', 'N/A'):>12.2e} "
                        f"{r.get('n_mut', r.get('n_mut_total', 'N/A')):>8} "
                        f"{r.get('n_ctrl', r.get('n_ctrl_total', 'N/A')):>8}")


if __name__ == "__main__":
    main()
