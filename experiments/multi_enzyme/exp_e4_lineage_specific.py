#!/usr/bin/env python
"""E4: Lineage-Specific Editability Gain/Loss.

From 3,610 ortholog pairs scored with GB editability models, identify positions
where editability changed dramatically between human and chimp:
  - "Gained" = human HIGH, chimp LOW (top 10% delta)
  - "Lost"   = human LOW, chimp HIGH  (bottom 10% delta)
  - "Stable" = middle 80%

Characterize gained/lost/stable by:
  1. Gene identity and enzyme category
  2. Trinucleotide context
  3. Tissue editing profiles (Levanon data)
  4. gnomAD constraint (LOEUF, missense Z, pLI)
  5. Structural features comparison

Usage:
    /opt/miniconda3/envs/quris/bin/python -u experiments/multi_enzyme/exp_e4_lineage_specific.py
"""

import json
import logging
import sys
import warnings
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
SCORED_CSV = PROJECT_ROOT / "experiments/multi_enzyme/outputs/cross_species/gb_scoring_human_vs_chimp.csv"
LEVANON_CAT = PROJECT_ROOT / "data/processed/multi_enzyme/levanon_all_categories.csv"
LEVANON_TISSUE = PROJECT_ROOT / "data/processed/multi_enzyme/levanon_tissue_rates.csv"
SPLITS_A3A = PROJECT_ROOT / "data/processed/splits_expanded_a3a.csv"
SPLITS_ME = PROJECT_ROOT / "data/processed/multi_enzyme/splits_multi_enzyme_v3_with_negatives.csv"
GNOMAD = PROJECT_ROOT / "data/raw/gnomad/gnomad_v4.1_constraint.tsv"
REFGENE = PROJECT_ROOT / "data/raw/genomes/refGene.txt"
LOOP_POS = PROJECT_ROOT / "data/processed/multi_enzyme/loop_position_per_site_v3.csv"

OUT_DIR = PROJECT_ROOT / "experiments/multi_enzyme/outputs/evolutionary/e4_lineage_specific"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_refgene_map():
    """Build chr:pos -> gene_name mapping from refGene."""
    logger.info("Loading refGene annotations...")
    cols = ["bin", "name", "chrom", "strand", "txStart", "txEnd",
            "cdsStart", "cdsEnd", "exonCount", "exonStarts", "exonEnds",
            "score", "name2", "cdsStartStat", "cdsEndStat", "exonFrames"]
    rg = pd.read_csv(REFGENE, sep="\t", header=None, names=cols, low_memory=False)
    # Keep canonical (longest CDS per gene)
    rg["cds_len"] = rg["cdsEnd"] - rg["cdsStart"]
    rg = rg.sort_values("cds_len", ascending=False).drop_duplicates("name2", keep="first")
    return rg[["name2", "chrom", "txStart", "txEnd", "strand", "cdsStart", "cdsEnd"]].copy()


def site_to_gene(site_ids, refgene_df, levanon_df=None):
    """Map site_id (chr:pos:strand or C2U_XXXX) to gene name."""
    logger.info("Mapping %d sites to genes via refGene...", len(site_ids))

    # Build C2U -> gene mapping from Levanon data
    c2u_gene = {}
    c2u_chrpos = {}
    if levanon_df is not None:
        for _, row in levanon_df.iterrows():
            c2u_gene[row["site_id"]] = row.get("gene_refseq", "")
            c2u_chrpos[row["site_id"]] = (row["chr"], int(row["start"]))

    # Build interval lookup per chromosome
    gene_map = {}
    for chrom, grp in refgene_df.groupby("chrom"):
        intervals = list(zip(grp["txStart"], grp["txEnd"], grp["name2"]))
        gene_map[chrom] = sorted(intervals)

    result = {}
    for sid in site_ids:
        # Handle C2U_ format
        if sid.startswith("C2U_"):
            if sid in c2u_gene and c2u_gene[sid]:
                result[sid] = c2u_gene[sid]
                continue
            # Try refGene lookup via chrpos
            if sid in c2u_chrpos:
                chrom, pos = c2u_chrpos[sid]
                matches = []
                for start, end, gene in gene_map.get(chrom, []):
                    if start <= pos <= end:
                        matches.append(gene)
                result[sid] = ";".join(matches) if matches else "intergenic"
                continue
            result[sid] = "unknown"
            continue

        # Handle chr:pos:strand format
        parts = sid.split(":")
        if len(parts) < 2:
            result[sid] = "unknown"
            continue
        chrom, pos = parts[0], int(parts[1])
        matches = []
        for start, end, gene in gene_map.get(chrom, []):
            if start <= pos <= end:
                matches.append(gene)
        result[sid] = ";".join(matches) if matches else "intergenic"
    return result


def load_gnomad_constraint():
    """Load gnomAD constraint metrics (canonical transcripts only)."""
    logger.info("Loading gnomAD v4.1 constraint data...")
    gnomad = pd.read_csv(GNOMAD, sep="\t", low_memory=False)
    # Keep canonical transcripts
    gnomad_canon = gnomad[gnomad["canonical"] == True].copy()
    # If duplicates, keep MANE select
    mane = gnomad_canon[gnomad_canon["mane_select"] == True]
    non_mane = gnomad_canon[~gnomad_canon["gene"].isin(mane["gene"])]
    gnomad_canon = pd.concat([mane, non_mane]).drop_duplicates("gene", keep="first")

    # Extract key constraint columns
    constraint_cols = {
        "gene": "gene",
        "lof.oe_ci.upper": "loeuf",
        "lof.pLI": "pLI",
        "lof.z_score": "lof_z",
        "mis.z_score": "mis_z",
        "mis.oe": "mis_oe",
    }
    result = gnomad_canon[list(constraint_cols.keys())].rename(columns=constraint_cols)
    for c in ["loeuf", "pLI", "lof_z", "mis_z", "mis_oe"]:
        result[c] = pd.to_numeric(result[c], errors="coerce")
    logger.info("  gnomAD: %d genes with constraint data", len(result))
    return result


def extract_trinucleotide(site_id, splits_df):
    """Get trinucleotide context from flanking_seq in splits data."""
    row = splits_df[splits_df["site_id"] == site_id]
    if len(row) == 0 or pd.isna(row.iloc[0].get("flanking_seq", np.nan)):
        return None
    seq = row.iloc[0]["flanking_seq"]
    center = len(seq) // 2
    if center >= 1 and center < len(seq) - 1:
        return seq[center - 1:center + 2].upper()
    return None


def main():
    logger.info("=" * 70)
    logger.info("E4: Lineage-Specific Editability Gain/Loss")
    logger.info("=" * 70)

    # ── 1. Load scored orthologs ──────────────────────────────────────────
    logger.info("\n[Step 1] Loading scored orthologs...")
    df = pd.read_csv(SCORED_CSV)
    logger.info("  Loaded %d ortholog pairs", len(df))
    logger.info("  Columns: %s", list(df.columns))
    logger.info("  Score diff stats: mean=%.4f, std=%.4f, min=%.4f, max=%.4f",
                df["score_diff"].mean(), df["score_diff"].std(),
                df["score_diff"].min(), df["score_diff"].max())

    # score_diff = human_gb_score - chimp_gb_score  (already computed, but verify)
    # Actually from the CSV: score_diff looks like chimp - human based on signs.
    # Let's recompute to be sure
    df["delta"] = df["human_gb_score"] - df["chimp_gb_score"]
    logger.info("  Recomputed delta (human - chimp): mean=%.4f, std=%.4f",
                df["delta"].mean(), df["delta"].std())

    # ── 2. Define gained / lost / stable ──────────────────────────────────
    logger.info("\n[Step 2] Classifying sites into gained/lost/stable...")
    q10 = df["delta"].quantile(0.10)
    q90 = df["delta"].quantile(0.90)
    logger.info("  10th percentile delta: %.4f", q10)
    logger.info("  90th percentile delta: %.4f", q90)

    df["category"] = "stable"
    df.loc[df["delta"] >= q90, "category"] = "gained"   # human >> chimp
    df.loc[df["delta"] <= q10, "category"] = "lost"     # chimp >> human

    cat_counts = df["category"].value_counts()
    logger.info("  Category counts: %s", dict(cat_counts))

    # Also create a "strong" threshold (top/bottom 5%)
    q05 = df["delta"].quantile(0.05)
    q95 = df["delta"].quantile(0.95)
    df["category_strict"] = "stable"
    df.loc[df["delta"] >= q95, "category_strict"] = "gained_strict"
    df.loc[df["delta"] <= q05, "category_strict"] = "lost_strict"

    # ── 3. Map sites to genes ─────────────────────────────────────────────
    logger.info("\n[Step 3] Mapping sites to genes...")
    refgene = load_refgene_map()
    lev_for_genes = pd.read_csv(LEVANON_CAT) if LEVANON_CAT.exists() else None
    gene_map = site_to_gene(df["site_id"].tolist(), refgene, levanon_df=lev_for_genes)
    df["gene"] = df["site_id"].map(gene_map)
    logger.info("  Gene mapping: %d genic, %d intergenic",
                (df["gene"] != "intergenic").sum(), (df["gene"] == "intergenic").sum())

    # Supplement: also try A3A splits for chr:pos:strand sites
    if SPLITS_A3A.exists():
        a3a = pd.read_csv(SPLITS_A3A, low_memory=False)
        a3a_gene_map = {}
        for _, row in a3a.iterrows():
            if pd.notna(row.get("gene", np.nan)):
                a3a_gene_map[row["site_id"]] = row["gene"]
                sid_key = f"{row['chr']}:{row['start']}:{row['strand']}"
                a3a_gene_map[sid_key] = row["gene"]
        for idx, row in df.iterrows():
            if row["gene"] in ("intergenic", "unknown") and row["site_id"] in a3a_gene_map:
                g = a3a_gene_map[row["site_id"]]
                if pd.notna(g) and g:
                    df.at[idx, "gene"] = g
        logger.info("  After A3A supplement: %d genic, %d intergenic/unknown",
                    (~df["gene"].isin(["intergenic", "unknown"])).sum(),
                    df["gene"].isin(["intergenic", "unknown"]).sum())

    # ── 4. Enzyme category breakdown ──────────────────────────────────────
    logger.info("\n[Step 4] Enzyme category breakdown...")
    enzyme_by_cat = df.groupby(["category", "enzyme"]).size().unstack(fill_value=0)
    logger.info("\nEnzyme × Category:\n%s", enzyme_by_cat)

    # Chi-square test: is enzyme distribution different across categories?
    contingency = enzyme_by_cat.values
    if contingency.shape[0] >= 2 and contingency.shape[1] >= 2:
        chi2, p_chi, dof, expected = stats.chi2_contingency(contingency)
        logger.info("  Chi-square: chi2=%.2f, p=%.2e, dof=%d", chi2, p_chi, dof)
    else:
        chi2, p_chi = 0, 1.0

    # ── 5. Trinucleotide context ──────────────────────────────────────────
    logger.info("\n[Step 5] Trinucleotide context analysis...")
    # Load splits to get flanking sequences
    splits_me = pd.read_csv(SPLITS_ME, low_memory=False)
    # Build site_id -> flanking_seq map (both native ID and chr:pos:strand)
    seq_map = {}
    for _, row in splits_me.iterrows():
        if pd.notna(row.get("flanking_seq", np.nan)):
            seq_map[row["site_id"]] = row["flanking_seq"]
            sid = f"{row['chr']}:{row['start']}:{row['strand']}"
            seq_map[sid] = row["flanking_seq"]
    # Also build C2U -> chrpos mapping for matching
    if lev_for_genes is not None:
        for _, row in lev_for_genes.iterrows():
            chrpos = f"{row['chr']}:{row['start']}:{row['strand']}"
            if chrpos in seq_map:
                seq_map[row["site_id"]] = seq_map[chrpos]

    trinuc_data = {}
    for idx, row in df.iterrows():
        seq = seq_map.get(row["site_id"])
        if seq and len(seq) >= 3:
            center = len(seq) // 2
            tri = seq[center - 1:center + 2].upper()
            trinuc_data[row["site_id"]] = tri

    df["trinucleotide"] = df["site_id"].map(trinuc_data)
    logger.info("  Trinucleotide mapped for %d / %d sites", df["trinucleotide"].notna().sum(), len(df))

    trinuc_by_cat = df.groupby(["category", "trinucleotide"]).size().unstack(fill_value=0)
    if len(trinuc_by_cat) > 0:
        logger.info("\nTop trinucleotides by category:\n%s", trinuc_by_cat)

    # ── 6. Tissue editing profiles ────────────────────────────────────────
    logger.info("\n[Step 6] Tissue editing profiles...")
    tissue_data = {}
    if LEVANON_TISSUE.exists():
        tissue_df = pd.read_csv(LEVANON_TISSUE)
        # Map C2U site IDs to chr:pos:strand AND keep C2U IDs
        if LEVANON_CAT.exists():
            lev = pd.read_csv(LEVANON_CAT)
            c2u_to_chrpos = {}
            for _, row in lev.iterrows():
                c2u_to_chrpos[row["site_id"]] = f"{row['chr']}:{row['start']}:{row['strand']}"
            tissue_df["chrpos_id"] = tissue_df["site_id"].map(c2u_to_chrpos)
            tissue_cols = [c for c in tissue_df.columns if c not in ["site_id", "enzyme_category", "chrpos_id"]]

            for cat in ["gained", "lost", "stable"]:
                cat_sites = set(df[df["category"] == cat]["site_id"])
                # Match on both chrpos_id and original C2U site_id
                matched = tissue_df[tissue_df["chrpos_id"].isin(cat_sites) |
                                    tissue_df["site_id"].isin(cat_sites)]
                if len(matched) > 0:
                    mean_rates = matched[tissue_cols].mean()
                    tissue_data[cat] = {
                        "n_matched": len(matched),
                        "mean_editing_rate": float(mean_rates.mean()),
                        "top_5_tissues": dict(mean_rates.nlargest(5)),
                        "bottom_5_tissues": dict(mean_rates.nsmallest(5)),
                        "n_tissues_edited": int((mean_rates > 1.0).sum()),
                    }
                    logger.info("  %s: %d sites with tissue data, mean rate=%.2f%%, tissues>1%%=%d",
                                cat, len(matched), mean_rates.mean(), (mean_rates > 1.0).sum())

    # ── 7. gnomAD constraint comparison ───────────────────────────────────
    logger.info("\n[Step 7] gnomAD constraint analysis...")
    gnomad = load_gnomad_constraint()

    # Get primary gene (first gene if multiple)
    df["primary_gene"] = df["gene"].str.split(";").str[0]
    df_gnomad = df.merge(gnomad, left_on="primary_gene", right_on="gene", how="left", suffixes=("", "_gnomad"))

    matched_gnomad = df_gnomad["loeuf"].notna().sum()
    logger.info("  Matched %d / %d sites to gnomAD constraint", matched_gnomad, len(df))

    constraint_results = {}
    for metric in ["loeuf", "pLI", "lof_z", "mis_z"]:
        constraint_results[metric] = {}
        logger.info("\n  --- %s ---", metric)
        for cat in ["gained", "lost", "stable"]:
            vals = df_gnomad.loc[df_gnomad["category"] == cat, metric].dropna()
            constraint_results[metric][cat] = {
                "n": int(len(vals)),
                "mean": float(vals.mean()) if len(vals) > 0 else None,
                "median": float(vals.median()) if len(vals) > 0 else None,
                "std": float(vals.std()) if len(vals) > 0 else None,
            }
            logger.info("    %s: n=%d, mean=%.3f, median=%.3f",
                        cat, len(vals),
                        vals.mean() if len(vals) > 0 else 0,
                        vals.median() if len(vals) > 0 else 0)

        # Statistical tests: gained vs stable, lost vs stable
        gained_vals = df_gnomad.loc[df_gnomad["category"] == "gained", metric].dropna()
        lost_vals = df_gnomad.loc[df_gnomad["category"] == "lost", metric].dropna()
        stable_vals = df_gnomad.loc[df_gnomad["category"] == "stable", metric].dropna()

        if len(gained_vals) > 5 and len(stable_vals) > 5:
            u_stat, p_val = stats.mannwhitneyu(gained_vals, stable_vals, alternative="two-sided")
            constraint_results[metric]["gained_vs_stable_p"] = float(p_val)
            logger.info("    gained vs stable: U=%.0f, p=%.4e", u_stat, p_val)
        if len(lost_vals) > 5 and len(stable_vals) > 5:
            u_stat, p_val = stats.mannwhitneyu(lost_vals, stable_vals, alternative="two-sided")
            constraint_results[metric]["lost_vs_stable_p"] = float(p_val)
            logger.info("    lost vs stable:   U=%.0f, p=%.4e", u_stat, p_val)
        if len(gained_vals) > 5 and len(lost_vals) > 5:
            u_stat, p_val = stats.mannwhitneyu(gained_vals, lost_vals, alternative="two-sided")
            constraint_results[metric]["gained_vs_lost_p"] = float(p_val)
            logger.info("    gained vs lost:   U=%.0f, p=%.4e", u_stat, p_val)

    # Kruskal-Wallis across all 3 groups
    kw_results = {}
    for metric in ["loeuf", "pLI", "lof_z", "mis_z"]:
        groups = [df_gnomad.loc[df_gnomad["category"] == cat, metric].dropna()
                  for cat in ["gained", "lost", "stable"]]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) >= 2:
            h_stat, p_val = stats.kruskal(*groups)
            kw_results[metric] = {"H": float(h_stat), "p": float(p_val)}
            logger.info("  Kruskal-Wallis %s: H=%.2f, p=%.4e", metric, h_stat, p_val)

    # ── 8. Structural features comparison ─────────────────────────────────
    logger.info("\n[Step 8] Structural features comparison...")
    struct_results = {}
    if LOOP_POS.exists():
        loop_df = pd.read_csv(LOOP_POS)
        # Map loop site_ids to chr:pos:strand AND keep C2U IDs
        loop_site_map = {}  # multi-enzyme site_id -> chr:pos:strand
        me_df = splits_me  # already loaded
        for _, row in me_df.iterrows():
            sid = f"{row['chr']}:{row['start']}:{row['strand']}"
            loop_site_map[row["site_id"]] = sid

        loop_df["chrpos_id"] = loop_df["site_id"].map(loop_site_map)
        # Merge: try matching on chrpos_id first, then on original site_id (for C2U_ IDs)
        loop_merged = df[["site_id", "category", "delta"]].merge(
            loop_df, left_on="site_id", right_on="chrpos_id", how="left", suffixes=("", "_loop")
        )
        # For unmatched C2U_ IDs, try direct site_id match
        unmatched = loop_merged[loop_merged["is_unpaired"].isna() if "is_unpaired" in loop_merged.columns else loop_merged.iloc[:, 0].isna()]
        if len(unmatched) > 0:
            loop_merged2 = df[["site_id", "category", "delta"]].merge(
                loop_df, left_on="site_id", right_on="site_id", how="left", suffixes=("", "_loop2")
            )
            # Fill in missing values from second merge
            for col in loop_df.columns:
                if col in loop_merged.columns and col != "site_id":
                    mask = loop_merged[col].isna() & loop_merged2[col].notna()
                    loop_merged.loc[mask, col] = loop_merged2.loc[mask, col]

        struct_cols = ["is_unpaired", "relative_loop_position", "loop_size",
                       "dist_to_apex", "local_unpaired_fraction",
                       "left_stem_length", "right_stem_length"]
        available_cols = [c for c in struct_cols if c in loop_merged.columns]

        for col in available_cols:
            struct_results[col] = {}
            for cat in ["gained", "lost", "stable"]:
                vals = loop_merged.loc[loop_merged["category"] == cat, col].dropna()
                struct_results[col][cat] = {
                    "n": int(len(vals)),
                    "mean": float(vals.mean()) if len(vals) > 0 else None,
                    "median": float(vals.median()) if len(vals) > 0 else None,
                }

            # Test gained vs lost
            gv = loop_merged.loc[loop_merged["category"] == "gained", col].dropna()
            lv = loop_merged.loc[loop_merged["category"] == "lost", col].dropna()
            if len(gv) > 5 and len(lv) > 5:
                u, p = stats.mannwhitneyu(gv, lv, alternative="two-sided")
                struct_results[col]["gained_vs_lost_p"] = float(p)
                logger.info("  %s: gained=%.3f, lost=%.3f, p=%.4e",
                            col, gv.mean(), lv.mean(), p)

    # ── 9. Gene lists for gained / lost ───────────────────────────────────
    logger.info("\n[Step 9] Gene lists for gained/lost categories...")
    gene_lists = {}
    for cat in ["gained", "lost"]:
        genes = df.loc[df["category"] == cat, "primary_gene"].value_counts()
        gene_lists[cat] = dict(genes.head(30))
        logger.info("\n  Top 15 %s genes:", cat)
        for g, n in genes.head(15).items():
            logger.info("    %s: %d sites", g, n)

    # Unique genes in gained vs lost
    gained_genes = set(df.loc[df["category"] == "gained", "primary_gene"].dropna())
    lost_genes = set(df.loc[df["category"] == "lost", "primary_gene"].dropna())
    gained_only = gained_genes - lost_genes
    lost_only = lost_genes - gained_genes
    shared = gained_genes & lost_genes
    logger.info("\n  Gained-only genes: %d, Lost-only genes: %d, Shared: %d",
                len(gained_only), len(lost_only), len(shared))

    # ── 10. Score distribution analysis ───────────────────────────────────
    logger.info("\n[Step 10] Score distribution analysis...")
    score_stats = {}
    for cat in ["gained", "lost", "stable"]:
        sub = df[df["category"] == cat]
        score_stats[cat] = {
            "n": int(len(sub)),
            "human_score_mean": float(sub["human_gb_score"].mean()),
            "human_score_median": float(sub["human_gb_score"].median()),
            "chimp_score_mean": float(sub["chimp_gb_score"].mean()),
            "chimp_score_median": float(sub["chimp_gb_score"].median()),
            "delta_mean": float(sub["delta"].mean()),
            "delta_median": float(sub["delta"].median()),
            "delta_std": float(sub["delta"].std()),
            "delta_min": float(sub["delta"].min()),
            "delta_max": float(sub["delta"].max()),
        }
        logger.info("  %s: n=%d, human=%.3f±%.3f, chimp=%.3f±%.3f, delta=%.3f±%.3f",
                    cat, len(sub),
                    sub["human_gb_score"].mean(), sub["human_gb_score"].std(),
                    sub["chimp_gb_score"].mean(), sub["chimp_gb_score"].std(),
                    sub["delta"].mean(), sub["delta"].std())

    # ── 11. Motif preservation analysis ───────────────────────────────────
    logger.info("\n[Step 11] Motif preservation in gained/lost...")
    motif_by_cat = df.groupby("category")["motif_preserved"].value_counts(normalize=True).unstack(fill_value=0)
    logger.info("\nMotif preservation by category:\n%s", motif_by_cat)

    # ── 12. Levanon category enrichment (for Levanon sites only) ──────────
    logger.info("\n[Step 12] Levanon category enrichment...")
    levanon_enrich = {}
    if LEVANON_CAT.exists():
        lev = pd.read_csv(LEVANON_CAT)
        lev["chrpos_id"] = lev.apply(lambda r: f"{r['chr']}:{r['start']}:{r['strand']}", axis=1)
        lev_cols = ["chrpos_id", "site_id", "enzyme_category", "genomic_category",
                    "mrna_location_refseq", "exonic_function",
                    "tissue_classification", "mean_gtex_editing_rate"]
        lev_cols = [c for c in lev_cols if c in lev.columns]
        # Merge on chrpos_id first
        df_lev = df.merge(lev[lev_cols], left_on="site_id", right_on="chrpos_id",
                           how="left", suffixes=("", "_lev"))
        # For C2U_ IDs, also try matching on site_id directly
        unmatched_mask = df_lev["enzyme_category"].isna()
        if unmatched_mask.any():
            df_lev2 = df.merge(lev[lev_cols], left_on="site_id", right_on="site_id",
                                how="left", suffixes=("", "_lev2"))
            for col in ["enzyme_category", "genomic_category", "mrna_location_refseq",
                        "exonic_function", "tissue_classification", "mean_gtex_editing_rate"]:
                if col in df_lev2.columns:
                    mask = df_lev[col].isna() & df_lev2[col].notna()
                    df_lev.loc[mask, col] = df_lev2.loc[mask, col]

        for col in ["enzyme_category", "genomic_category", "tissue_classification", "exonic_function"]:
            matched = df_lev[df_lev[col].notna()]
            if len(matched) > 0:
                ct = matched.groupby(["category", col]).size().unstack(fill_value=0)
                levanon_enrich[col] = ct.to_dict()
                logger.info("\n  %s distribution:\n%s", col, ct)

        # Mean editing rate by category
        for cat in ["gained", "lost", "stable"]:
            rates = df_lev.loc[(df_lev["category"] == cat) &
                               df_lev["mean_gtex_editing_rate"].notna(), "mean_gtex_editing_rate"]
            if len(rates) > 0:
                logger.info("  %s mean GTEx rate: %.2f%% (n=%d)", cat, rates.mean(), len(rates))

    # ── 13. Figures ───────────────────────────────────────────────────────
    logger.info("\n[Step 13] Generating figures...")

    # Fig 1: Delta distribution with gained/lost highlighted
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    ax.hist(df["delta"], bins=80, color="gray", alpha=0.5, label="All")
    ax.hist(df.loc[df["category"] == "gained", "delta"], bins=30, color="red", alpha=0.7, label="Gained")
    ax.hist(df.loc[df["category"] == "lost", "delta"], bins=30, color="blue", alpha=0.7, label="Lost")
    ax.axvline(q10, color="blue", linestyle="--", alpha=0.5, label=f"Q10={q10:.3f}")
    ax.axvline(q90, color="red", linestyle="--", alpha=0.5, label=f"Q90={q90:.3f}")
    ax.set_xlabel("Delta (human - chimp)")
    ax.set_ylabel("Count")
    ax.set_title("Editability Score Difference Distribution")
    ax.legend(fontsize=8)

    # Fig 2: Human vs Chimp scatter
    ax = axes[1]
    colors = {"gained": "red", "lost": "blue", "stable": "gray"}
    for cat in ["stable", "gained", "lost"]:
        sub = df[df["category"] == cat]
        ax.scatter(sub["chimp_gb_score"], sub["human_gb_score"],
                   c=colors[cat], alpha=0.3, s=10, label=f"{cat} (n={len(sub)})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("Chimp GB Score")
    ax.set_ylabel("Human GB Score")
    ax.set_title("Human vs Chimp Editability")
    ax.legend(fontsize=8)

    # Fig 3: Enzyme category proportions
    ax = axes[2]
    enzyme_props = df.groupby("category")["enzyme"].value_counts(normalize=True).unstack(fill_value=0)
    enzyme_props.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title("Enzyme Category Proportions")
    ax.set_ylabel("Proportion")
    ax.legend(fontsize=7, loc="upper right")
    ax.tick_params(axis="x", rotation=0)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "e4_overview.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  Saved e4_overview.png")

    # Fig 4: gnomAD constraint boxplots
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for i, metric in enumerate(["loeuf", "pLI", "lof_z", "mis_z"]):
        ax = axes[i]
        data_to_plot = []
        labels = []
        for cat in ["gained", "stable", "lost"]:
            vals = df_gnomad.loc[df_gnomad["category"] == cat, metric].dropna()
            if len(vals) > 0:
                data_to_plot.append(vals.values)
                labels.append(f"{cat}\n(n={len(vals)})")
        if data_to_plot:
            bp = ax.boxplot(data_to_plot, labels=labels, showfliers=False)
            ax.set_title(metric)
            if metric in kw_results:
                ax.set_xlabel(f"KW p={kw_results[metric]['p']:.2e}")
    plt.suptitle("gnomAD Constraint by Editability Change Category", fontsize=14)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "e4_gnomad_constraint.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  Saved e4_gnomad_constraint.png")

    # Fig 5: Structural features comparison
    if struct_results:
        n_cols = min(len(struct_results), 4)
        n_rows = (len(struct_results) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = np.array(axes).flatten() if len(struct_results) > 1 else [axes]

        for i, col in enumerate(struct_results.keys()):
            if i >= len(axes):
                break
            ax = axes[i]
            data_to_plot = []
            labels = []
            for cat in ["gained", "stable", "lost"]:
                vals = loop_merged.loc[loop_merged["category"] == cat, col].dropna()
                if len(vals) > 0:
                    data_to_plot.append(vals.values)
                    labels.append(cat)
            if data_to_plot:
                ax.boxplot(data_to_plot, labels=labels, showfliers=False)
                ax.set_title(col)
                p_val = struct_results[col].get("gained_vs_lost_p", None)
                if p_val is not None:
                    ax.set_xlabel(f"G vs L p={p_val:.2e}")

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        plt.suptitle("Structural Features by Editability Change", fontsize=13)
        plt.tight_layout()
        fig.savefig(OUT_DIR / "e4_structural_features.png", dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("  Saved e4_structural_features.png")

    # ── 14. Compile and save results ──────────────────────────────────────
    logger.info("\n[Step 14] Saving results...")

    # Save annotated CSV
    save_cols = ["site_id", "chimp_id", "enzyme", "motif_preserved",
                 "human_gb_score", "chimp_gb_score", "delta", "category",
                 "category_strict", "gene", "primary_gene", "trinucleotide"]
    df[save_cols].to_csv(OUT_DIR / "e4_annotated_orthologs.csv", index=False)
    logger.info("  Saved e4_annotated_orthologs.csv")

    # Save gene lists
    gained_df = df[df["category"] == "gained"][["site_id", "gene", "primary_gene", "enzyme", "delta", "human_gb_score", "chimp_gb_score"]]
    gained_df.to_csv(OUT_DIR / "e4_gained_sites.csv", index=False)
    lost_df = df[df["category"] == "lost"][["site_id", "gene", "primary_gene", "enzyme", "delta", "human_gb_score", "chimp_gb_score"]]
    lost_df.to_csv(OUT_DIR / "e4_lost_sites.csv", index=False)
    logger.info("  Saved e4_gained_sites.csv and e4_lost_sites.csv")

    # JSON summary
    results = {
        "experiment": "E4: Lineage-Specific Editability Gain/Loss",
        "n_orthologs": int(len(df)),
        "thresholds": {
            "q10_lost": float(q10),
            "q90_gained": float(q90),
            "q05_strict_lost": float(q05),
            "q95_strict_gained": float(q95),
        },
        "category_counts": {k: int(v) for k, v in cat_counts.items()},
        "score_stats": score_stats,
        "enzyme_by_category": enzyme_by_cat.to_dict() if isinstance(enzyme_by_cat, pd.DataFrame) else {},
        "enzyme_chi2": {"chi2": float(chi2), "p": float(p_chi)},
        "trinucleotide_by_category": trinuc_by_cat.to_dict() if len(trinuc_by_cat) > 0 else {},
        "gnomad_constraint": constraint_results,
        "gnomad_kruskal_wallis": kw_results,
        "structural_features": struct_results,
        "tissue_data": tissue_data,
        "levanon_enrichment": {k: {str(k2): v2 for k2, v2 in v.items()} for k, v in levanon_enrich.items()},
        "gene_lists": gene_lists,
        "gene_set_sizes": {
            "gained_unique_genes": int(len(gained_only)),
            "lost_unique_genes": int(len(lost_only)),
            "shared_genes": int(len(shared)),
        },
        "motif_preservation": motif_by_cat.to_dict() if len(motif_by_cat) > 0 else {},
    }

    with open(OUT_DIR / "e4_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("  Saved e4_results.json")

    # ── Summary ───────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info("Total ortholog pairs: %d", len(df))
    logger.info("Gained (human > chimp, top 10%%): %d sites", cat_counts.get("gained", 0))
    logger.info("Lost (chimp > human, bottom 10%%): %d sites", cat_counts.get("lost", 0))
    logger.info("Stable (middle 80%%): %d sites", cat_counts.get("stable", 0))
    logger.info("Delta thresholds: lost <= %.4f, gained >= %.4f", q10, q90)
    logger.info("Enzyme chi2 p-value: %.4e", p_chi)
    for metric in ["loeuf", "pLI", "mis_z"]:
        if metric in kw_results:
            logger.info("gnomAD %s Kruskal-Wallis p=%.4e", metric, kw_results[metric]["p"])
    logger.info("Output dir: %s", OUT_DIR)
    logger.info("DONE.")


if __name__ == "__main__":
    main()
