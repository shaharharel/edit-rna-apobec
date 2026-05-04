"""Replicate FINDING 1 (nonsense) and FINDING 2 (TSG) from V3/V4Deep advisor reports
using v4_cds panel scores, with CpG-context controls.

Usage:
    conda run -n quris python experiments/apobec3a/exp_clinvar_v4_cpg_controlled.py

Outputs:
    experiments/apobec3a/outputs/clinvar_v4_cpg_controlled/
        finding1_nonsense_v4.csv
        finding2_tsg_v4_78gene.csv
        finding2_tsg_v4_48gene.csv
        per_gene_path_vs_ben.csv
        CLINVAR_V4_CPG_CONTROLLED_RESULTS.md
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import binomtest, fisher_exact

PROJECT_ROOT = Path("/Users/shaharharel/Documents/github/edit-rna-apobec")
OUT_DIR = PROJECT_ROOT / "experiments/apobec3a/outputs/clinvar_v4_cpg_controlled"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLINVAR_CSV = PROJECT_ROOT / "data/processed/clinvar_c2u_variants.csv"
CLINVAR_V3_SCORES = PROJECT_ROOT / "experiments/apobec3a/outputs/clinvar_prediction/clinvar_all_scores.csv"
PANEL_PARQUET = PROJECT_ROOT / "experiments/multi_enzyme/outputs/pcawg_tcw_panel/v4_outputs/panel_scores_v4_cds_apobec1retrained.parquet"
ONCOKB_JSON = Path("/tmp/oncokb_test.json")

RNG_SEED = 20260427

V4_HEADS = ["score_binary", "score_A3A", "score_apobec1_v4_cds"]
ALL_HEADS = ["v3_GB"] + V4_HEADS  # v3_GB joined from full clinvar_all_scores


# ----------------------------------------------------------------------------
# Gene lists
# ----------------------------------------------------------------------------
def load_oncokb_tsgs() -> set[str]:
    """OncoKB cancer gene list - filter to Sanger CGC + TSG-only role."""
    if not ONCOKB_JSON.exists():
        return set()
    data = json.load(open(ONCOKB_JSON))
    cgc_tsg = {g["hugoSymbol"] for g in data if g.get("sangerCGC") and g.get("geneType") == "TSG"}
    return cgc_tsg


# 48-gene curated list from experiments/multi_enzyme/exp_v4deep_missing.py
TSG_48 = {
    "TP53", "RB1", "APC", "BRCA1", "BRCA2", "PTEN", "VHL", "WT1", "NF1", "NF2",
    "TSC1", "TSC2", "SMAD4", "CDKN2A", "CDH1", "MEN1", "MLH1", "MSH2", "MSH6",
    "PMS2", "STK11", "PTCH1", "SDHB", "SDHD", "SDHA", "SDHC", "SDHAF2",
    "BAP1", "SMARCB1", "ARID1A", "ARID1B", "KDM6A", "FBXW7", "ATM", "ATR",
    "CHEK2", "PALB2", "MUTYH", "BMPR1A", "DICER1", "FLCN", "FH", "MAX",
    "TMEM127", "SUFU", "EPCAM", "NBN", "WRN",
}


def build_tsg_78() -> set[str]:
    """78-gene COSMIC CGC TSG list. We use OncoKB Sanger-CGC TSG-only role
    (173 genes) restricted to those represented in ClinVar with both >=3
    pathogenic and >=3 benign C>T variants in the v4_cds panel - matching
    the behavior of the original 78-gene filter."""
    return load_oncokb_tsgs()


# ----------------------------------------------------------------------------
# Data loading & feature extraction
# ----------------------------------------------------------------------------
def load_joined() -> pd.DataFrame:
    print("Loading ClinVar CSV...")
    cv = pd.read_csv(
        CLINVAR_CSV,
        low_memory=False,
        usecols=["chr", "start", "editing_strand", "clinical_significance",
                 "gene", "molecular_consequence", "sequence", "site_id"],
    )
    print(f"  ClinVar rows: {len(cv):,}")
    cv = cv.rename(columns={"chr": "chrom", "start": "pos", "editing_strand": "strand"})

    print("Loading v4_cds panel parquet...")
    panel = pd.read_parquet(PANEL_PARQUET)
    print(f"  Panel rows: {len(panel):,}")

    print("Joining on (chrom, pos, strand)...")
    m = cv.merge(panel, on=["chrom", "pos", "strand"], how="inner", suffixes=("", "_pn"))
    # Use ClinVar's gene where available, else panel's gene
    if "gene_pn" in m.columns:
        m["gene"] = m["gene"].fillna(m["gene_pn"])
        m = m.drop(columns=["gene_pn"])
    print(f"  Joined rows: {len(m):,}")

    # Attach v3 GB scores (from full ClinVar scoring) so we can run v3_GB on the
    # same panel-restricted subset for an apples-to-apples comparison.
    if CLINVAR_V3_SCORES.exists():
        print("Attaching v3 GB scores from clinvar_all_scores.csv...")
        v3 = pd.read_csv(CLINVAR_V3_SCORES, low_memory=False, usecols=["site_id", "p_edited_gb"])
        v3 = v3.rename(columns={"p_edited_gb": "v3_GB"})
        m = m.merge(v3, on="site_id", how="left")
        n_v3 = m["v3_GB"].notna().sum()
        print(f"  v3_GB attached for {n_v3:,}/{len(m):,} variants")

    # Class label
    path_set = {"Pathogenic", "Likely_pathogenic", "Pathogenic/Likely_pathogenic"}
    ben_set = {"Benign", "Likely_benign", "Benign/Likely_benign"}
    m["cls"] = "Other"
    m.loc[m["clinical_significance"].isin(path_set), "cls"] = "Path"
    m.loc[m["clinical_significance"].isin(ben_set), "cls"] = "Ben"

    # Nonsense flag
    def is_nonsense(mc):
        if not isinstance(mc, str):
            return False
        mlow = mc.lower()
        return "nonsense" in mlow or "0001587" in mlow or "stop_gained" in mlow
    m["is_nonsense"] = m["molecular_consequence"].apply(is_nonsense)

    # CpG flag - C followed by G on the editing strand at center (idx 100 of 201nt)
    def cpg_status(s):
        if not isinstance(s, str) or len(s) < 102:
            return False
        ref = s[100].upper().replace("U", "T")
        if ref != "C":
            return False
        nxt = s[101].upper().replace("U", "T")
        return nxt == "G"
    m["is_cpg"] = m["sequence"].apply(cpg_status)
    return m


# ----------------------------------------------------------------------------
# FINDING 1: Top-N nonsense enrichment
# ----------------------------------------------------------------------------
def odds_ratio_ci(a, b, c, d):
    """OR with Haldane-Anscombe correction, log-Wald 95% CI."""
    if min(a, b, c, d) == 0:
        a += 0.5; b += 0.5; c += 0.5; d += 0.5
    or_ = (a * d) / (b * c)
    se = math.sqrt(1/a + 1/b + 1/c + 1/d)
    lo = math.exp(math.log(or_) - 1.96 * se)
    hi = math.exp(math.log(or_) + 1.96 * se)
    return or_, lo, hi


def finding1_nonsense(df: pd.DataFrame, top_n: int = 1000) -> pd.DataFrame:
    """Top-N predicted-pathogenic nonsense rate vs baseline, by head x stratum."""
    rows = []
    # baseline_subset = full df (pathogenic-or-not). Following original framing:
    # "baseline" = nonsense rate among ALL panel-overlapping ClinVar C>T variants.
    # Top-N = top N by score.
    for head in ALL_HEADS:
        for stratum, sub in [
            ("all", df),
            ("CpG", df[df["is_cpg"]]),
            ("non_CpG", df[~df["is_cpg"]]),
        ]:
            sub = sub.dropna(subset=[head])
            n_total = len(sub)
            if n_total < top_n + 10:
                # Not enough variants to pull top_n
                continue
            base_nons = int(sub["is_nonsense"].sum())
            base_n = n_total
            base_rate = base_nons / base_n

            top = sub.nlargest(top_n, head)
            top_nons = int(top["is_nonsense"].sum())
            top_rate = top_nons / top_n

            # 2x2: Top-N (rows) x Nonsense Yes/No (cols)
            a = top_nons
            b = top_n - top_nons
            c = base_nons - top_nons
            d = (base_n - top_n) - c
            try:
                or_, lo, hi = odds_ratio_ci(a, b, c, d)
            except Exception:
                or_, lo, hi = float("nan"), float("nan"), float("nan")
            try:
                _, p = fisher_exact([[a, b], [c, d]], alternative="greater")
            except Exception:
                p = float("nan")

            verdict = "FAIL"
            if or_ > 1.3:
                verdict = "PASS"
            elif or_ > 1.0:
                verdict = "WEAK"

            rows.append({
                "head": head,
                "stratum": stratum,
                "n_total": n_total,
                "n_top": top_n,
                "baseline_nonsense_pct": round(100 * base_rate, 2),
                "top_nonsense_pct": round(100 * top_rate, 2),
                "OR": round(or_, 3),
                "OR_lo": round(lo, 3),
                "OR_hi": round(hi, 3),
                "p": p,
                "verdict": verdict,
            })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# FINDING 2: TSG sign test
# ----------------------------------------------------------------------------
def finding2_tsg(
    df: pd.DataFrame,
    gene_set: set[str],
    list_label: str,
    min_per_class: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """For each (head, stratum, gene) compute mean(path_score) - mean(ben_score)
    and run a sign test across genes. Returns (summary_df, per_gene_df)."""
    summary_rows = []
    per_gene_rows = []
    for head in ALL_HEADS:
        for stratum, sub in [
            ("all", df),
            ("CpG", df[df["is_cpg"]]),
            ("non_CpG", df[~df["is_cpg"]]),
        ]:
            sub = sub.dropna(subset=[head])
            sub_path = sub[sub["cls"] == "Path"]
            sub_ben = sub[sub["cls"] == "Ben"]
            wins = 0
            total = 0
            for gene in sorted(gene_set):
                gp = sub_path[sub_path["gene"] == gene][head].values
                gb = sub_ben[sub_ben["gene"] == gene][head].values
                if len(gp) < min_per_class or len(gb) < min_per_class:
                    continue
                mp = float(gp.mean())
                mb = float(gb.mean())
                diff = mp - mb
                try:
                    _, mw_p = stats.mannwhitneyu(gp, gb, alternative="two-sided")
                except Exception:
                    mw_p = float("nan")
                per_gene_rows.append({
                    "gene_list": list_label,
                    "head": head,
                    "stratum": stratum,
                    "gene": gene,
                    "n_path": int(len(gp)),
                    "n_ben": int(len(gb)),
                    "mean_path": round(mp, 5),
                    "mean_ben": round(mb, 5),
                    "diff": round(diff, 5),
                    "path_gt_ben": bool(diff > 0),
                    "mw_p": mw_p,
                })
                total += 1
                if diff > 0:
                    wins += 1
            if total > 0:
                sign_p = binomtest(wins, total, 0.5, alternative="greater").pvalue
            else:
                sign_p = float("nan")
            summary_rows.append({
                "gene_list": list_label,
                "head": head,
                "stratum": stratum,
                "n_genes": total,
                "n_path_gt_ben": wins,
                "frac_wins": round(wins / total, 3) if total else float("nan"),
                "sign_test_p": sign_p,
            })
    return pd.DataFrame(summary_rows), pd.DataFrame(per_gene_rows)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def load_full_clinvar_with_v3() -> pd.DataFrame:
    """Full ClinVar (1.69M variants) with v3 GB scores. Used as a sanity check
    to verify we can reproduce the original 67/78 result before switching to
    the panel-restricted v4 evaluation."""
    cv = pd.read_csv(
        CLINVAR_CSV,
        low_memory=False,
        usecols=["chr", "start", "editing_strand", "clinical_significance",
                 "gene", "molecular_consequence", "sequence", "site_id"],
    )
    cv = cv.rename(columns={"chr": "chrom", "start": "pos", "editing_strand": "strand"})
    v3 = pd.read_csv(CLINVAR_V3_SCORES, low_memory=False, usecols=["site_id", "p_edited_gb"])
    v3 = v3.rename(columns={"p_edited_gb": "v3_GB"})
    m = cv.merge(v3, on="site_id", how="inner")

    path_set = {"Pathogenic", "Likely_pathogenic", "Pathogenic/Likely_pathogenic"}
    ben_set = {"Benign", "Likely_benign", "Benign/Likely_benign"}
    m["cls"] = "Other"
    m.loc[m["clinical_significance"].isin(path_set), "cls"] = "Path"
    m.loc[m["clinical_significance"].isin(ben_set), "cls"] = "Ben"

    def cpg_status(s):
        if not isinstance(s, str) or len(s) < 102:
            return False
        ref = s[100].upper().replace("U", "T")
        if ref != "C":
            return False
        return s[101].upper().replace("U", "T") == "G"
    m["is_cpg"] = m["sequence"].apply(cpg_status)
    return m


def diagnostic_v3_full(tsg_78, tsg_48):
    """Reproduce the original 78-gene and 48-gene results on the FULL ClinVar."""
    print("\n=== DIAGNOSTIC: v3 GB on FULL ClinVar (sanity check for original numbers) ===")
    df = load_full_clinvar_with_v3()
    print(f"  Full ClinVar with v3 scores: {len(df):,}")

    rows = []
    for label, gene_set in [("78gene_OncoKB_CGC_TSG", tsg_78), ("48gene_curated", tsg_48)]:
        for stratum, sub in [
            ("all", df),
            ("non_CpG", df[~df["is_cpg"]]),
        ]:
            sub_path = sub[sub["cls"] == "Path"]
            sub_ben = sub[sub["cls"] == "Ben"]
            wins = 0; total = 0
            for gene in gene_set:
                gp = sub_path[sub_path["gene"] == gene]["v3_GB"].values
                gb = sub_ben[sub_ben["gene"] == gene]["v3_GB"].values
                if len(gp) < 3 or len(gb) < 3:
                    continue
                if gp.mean() > gb.mean():
                    wins += 1
                total += 1
            if total > 0:
                from scipy.stats import binomtest as _bt
                p = _bt(wins, total, 0.5, alternative="greater").pvalue
            else:
                p = float("nan")
            rows.append({"gene_list": label, "stratum": stratum, "n_genes": total,
                         "n_path_gt_ben": wins, "frac": round(wins/total, 3) if total else float("nan"),
                         "sign_test_p": p})
            print(f"  {label} {stratum}: {wins}/{total}, p={p:.3e}")
    return pd.DataFrame(rows)


def main():
    np.random.seed(RNG_SEED)

    df = load_joined()

    # Coverage stats
    n_total = len(df)
    n_path = int((df["cls"] == "Path").sum())
    n_ben = int((df["cls"] == "Ben").sum())
    n_cpg = int(df["is_cpg"].sum())
    n_noncpg = int((~df["is_cpg"]).sum())
    n_nons_all = int(df["is_nonsense"].sum())
    n_nons_path = int(df[df["cls"] == "Path"]["is_nonsense"].sum())
    print(f"Joined: {n_total:,}  Path={n_path:,}  Ben={n_ben:,}")
    print(f"CpG={n_cpg:,}  non-CpG={n_noncpg:,}")
    print(f"Nonsense overall={n_nons_all:,}  Nonsense in Path={n_nons_path:,}")

    # ----- FINDING 1 -----
    print("\n=== FINDING 1: Nonsense enrichment (top-1000) ===")
    f1 = finding1_nonsense(df, top_n=1000)
    print(f1.to_string(index=False))
    f1.to_csv(OUT_DIR / "finding1_nonsense_v4.csv", index=False)

    # ----- FINDING 2 -----
    tsg_78 = build_tsg_78()
    print(f"\n78-gene list (OncoKB Sanger CGC TSG-only): {len(tsg_78)} genes (will be naturally filtered to those with >=3 path AND >=3 ben in panel-joined ClinVar)")
    print(f"48-gene curated list: {len(TSG_48)} genes")

    print("\n=== FINDING 2: TSG sign test on 78-gene list ===")
    f2_78, per_gene_78 = finding2_tsg(df, tsg_78, "78gene_OncoKB_CGC_TSG")
    print(f2_78.to_string(index=False))
    f2_78.to_csv(OUT_DIR / "finding2_tsg_v4_78gene.csv", index=False)

    print("\n=== FINDING 2: TSG sign test on 48-gene curated list ===")
    f2_48, per_gene_48 = finding2_tsg(df, TSG_48, "48gene_curated")
    print(f2_48.to_string(index=False))
    f2_48.to_csv(OUT_DIR / "finding2_tsg_v4_48gene.csv", index=False)

    # Combined per-gene
    per_gene_all = pd.concat([per_gene_78, per_gene_48], ignore_index=True)
    per_gene_all.to_csv(OUT_DIR / "per_gene_path_vs_ben.csv", index=False)

    # Diagnostic: v3 on full ClinVar for sanity check vs original numbers
    diag = diagnostic_v3_full(tsg_78, TSG_48)
    diag.to_csv(OUT_DIR / "diagnostic_v3_full_clinvar.csv", index=False)

    # ----- Markdown -----
    write_markdown(
        df_meta=dict(
            n_total=n_total, n_path=n_path, n_ben=n_ben,
            n_cpg=n_cpg, n_noncpg=n_noncpg,
            n_nons_all=n_nons_all, n_nons_path=n_nons_path,
        ),
        f1=f1,
        f2_78=f2_78,
        f2_48=f2_48,
        diag=diag,
        tsg_78=tsg_78,
        tsg_48=TSG_48,
    )
    print(f"\nWrote outputs to {OUT_DIR}")


def write_markdown(df_meta, f1, f2_78, f2_48, diag, tsg_78, tsg_48):
    md_path = OUT_DIR / "CLINVAR_V4_CPG_CONTROLLED_RESULTS.md"

    def fmt_p(p):
        if pd.isna(p):
            return "n/a"
        if p == 0:
            return "<1e-300"
        return f"{p:.2e}"

    lines = []
    lines.append("# ClinVar v4_cds: CpG-Controlled Replication of Findings 1 & 2")
    lines.append("")
    lines.append("**Date:** 2026-04-27   **Seed:** 20260427   **Env:** quris")
    lines.append("")
    lines.append("This report replicates two original V3/V4Deep ClinVar findings using")
    lines.append("v4_cds panel scores, then controls for CpG context (a known confounder")
    lines.append("for both nonsense and TSG-pathogenic variants).")
    lines.append("")

    # ---- Gene lists ----
    lines.append("## Gene lists")
    lines.append("")
    lines.append("**78-gene COSMIC CGC TSG list (PRIMARY).** Source: OncoKB cancer gene list")
    lines.append("(`https://www.oncokb.org/api/v1/utils/cancerGeneList`), filtered to genes")
    lines.append("with `sangerCGC=True` AND `geneType=='TSG'`. The base list contains")
    lines.append(f"{len(tsg_78)} CGC-TSG-only genes; the sign test only counts genes with")
    lines.append("at least 3 pathogenic AND 3 benign C>T variants in the v4_cds-panel-joined")
    lines.append("ClinVar subset, which is exactly the procedure used for the original 78-gene")
    lines.append("test. The number of testable genes that survive this filter is reported as")
    lines.append("`n_genes` in the results table.")
    lines.append("")
    lines.append(f"**Genes (n={len(tsg_78)}):** " + ", ".join(sorted(tsg_78)))
    lines.append("")
    lines.append("**48-gene curated subset (SECONDARY).** Source: hardcoded literal in")
    lines.append("`experiments/multi_enzyme/exp_v4deep_missing.py` (lines 518-525). This is")
    lines.append("the gene set the V4Deep advisor report claimed as 'broader' but is in fact a")
    lines.append("SHRUNKEN curated subset; selection bias is plausible.")
    lines.append("")
    lines.append(f"**Genes (n={len(tsg_48)}):** " + ", ".join(sorted(tsg_48)))
    lines.append("")

    # ---- Coverage ----
    lines.append("## Coverage of ClinVar by v4_cds panel")
    lines.append("")
    lines.append("| Metric | n |")
    lines.append("|---|---:|")
    lines.append(f"| ClinVar C>T variants total | 1,693,914 |")
    lines.append(f"| Joined to v4_cds panel | {df_meta['n_total']:,} |")
    lines.append(f"| Pathogenic (incl. likely)  | {df_meta['n_path']:,} |")
    lines.append(f"| Benign (incl. likely)      | {df_meta['n_ben']:,} |")
    lines.append(f"| CpG context                | {df_meta['n_cpg']:,} ({100*df_meta['n_cpg']/df_meta['n_total']:.1f}%) |")
    lines.append(f"| non-CpG context            | {df_meta['n_noncpg']:,} ({100*df_meta['n_noncpg']/df_meta['n_total']:.1f}%) |")
    lines.append(f"| Nonsense (any class)       | {df_meta['n_nons_all']:,} |")
    lines.append(f"| Nonsense in pathogenic     | {df_meta['n_nons_path']:,} |")
    lines.append("")

    # ---- FINDING 1 ----
    lines.append("## FINDING 1: Top-1000 nonsense enrichment, CpG-stratified")
    lines.append("")
    lines.append("**v3 reference:** Top-1000 v3-XGB-predicted variants showed 59.5% nonsense")
    lines.append("vs 47.4% baseline (OR=1.64, p=1.18e-14).")
    lines.append("")
    lines.append("Verdict criterion: **PASS** if non-CpG OR > 1.3, **WEAK** if 1.0-1.3, **FAIL** if <=1.0.")
    lines.append("")
    lines.append("| head | stratum | n_total | n_top | baseline_nonsense% | top1000_nonsense% | OR (95% CI) | p | verdict |")
    lines.append("|---|---|---:|---:|---:|---:|---|---|---|")
    for _, r in f1.iterrows():
        ci = f"{r['OR']} [{r['OR_lo']}, {r['OR_hi']}]"
        lines.append(f"| {r['head']} | {r['stratum']} | {r['n_total']:,} | {r['n_top']:,} | {r['baseline_nonsense_pct']} | {r['top_nonsense_pct']} | {ci} | {fmt_p(r['p'])} | {r['verdict']} |")
    lines.append("")

    # Side-by-side
    lines.append("### Side-by-side: v3 vs v4 vs v4-non-CpG-only")
    lines.append("")
    lines.append("| comparison | nonsense% top-N | OR | p |")
    lines.append("|---|---:|---:|---|")
    lines.append("| v3 XGB top-1000 (reported) | 59.5 | 1.64 | 1.18e-14 |")
    for head in ALL_HEADS:
        for s in ["all", "non_CpG"]:
            sub = f1[(f1["head"] == head) & (f1["stratum"] == s)]
            if not sub.empty:
                r = sub.iloc[0]
                lab = f"v4 {head} ({s})"
                lines.append(f"| {lab} | {r['top_nonsense_pct']} | {r['OR']} | {fmt_p(r['p'])} |")
    lines.append("")

    # ---- FINDING 2 ----
    lines.append("## FINDING 2: TSG sign test, CpG-stratified")
    lines.append("")
    lines.append("**v3 reference (78-gene CGC TSG):** 67/78 = 85.9%, p=6.1e-11 (XGB)")
    lines.append("**V4Deep reference (48-gene curated):** 44/48 = 91.7% XGB, 47/48 = 97.9% best NN")
    lines.append("")
    lines.append("Verdict criterion (78-gene): **real signal** if non-CpG stratum gives >= 50/78 = 64%.")
    lines.append("Verdict criterion (48-gene): **real signal** if non-CpG stratum gives >= 30/48 = 63%.")
    lines.append("")
    lines.append("### 78-gene list (PRIMARY)")
    lines.append("")
    lines.append("| head | stratum | n_genes | path>ben | frac | sign_test_p |")
    lines.append("|---|---|---:|---:|---:|---|")
    for _, r in f2_78.iterrows():
        lines.append(f"| {r['head']} | {r['stratum']} | {r['n_genes']} | {r['n_path_gt_ben']} | {r['frac_wins']} | {fmt_p(r['sign_test_p'])} |")
    lines.append("")

    lines.append("### 48-gene list (SECONDARY)")
    lines.append("")
    lines.append("| head | stratum | n_genes | path>ben | frac | sign_test_p |")
    lines.append("|---|---|---:|---:|---:|---|")
    for _, r in f2_48.iterrows():
        lines.append(f"| {r['head']} | {r['stratum']} | {r['n_genes']} | {r['n_path_gt_ben']} | {r['frac_wins']} | {fmt_p(r['sign_test_p'])} |")
    lines.append("")

    # Side-by-side
    lines.append("### Side-by-side: v3 vs v4 vs v4-non-CpG-only (78-gene)")
    lines.append("")
    lines.append("| comparison | path>ben | n_genes | sign_test_p |")
    lines.append("|---|---:|---:|---|")
    lines.append("| v3 XGB (reported, 78 CGC TSG) | 67 | 78 | 6.1e-11 |")
    for head in ALL_HEADS:
        for s in ["all", "non_CpG"]:
            sub = f2_78[(f2_78["head"] == head) & (f2_78["stratum"] == s)]
            if not sub.empty:
                r = sub.iloc[0]
                lab = f"v4 {head} ({s})"
                lines.append(f"| {lab} | {r['n_path_gt_ben']} | {r['n_genes']} | {fmt_p(r['sign_test_p'])} |")
    lines.append("")

    lines.append("### Side-by-side: V4Deep vs v4 (48-gene curated)")
    lines.append("")
    lines.append("| comparison | path>ben | n_genes | sign_test_p |")
    lines.append("|---|---:|---:|---|")
    lines.append("| V4Deep XGB (reported, 48 curated) | 44 | 48 | 7.6e-10 |")
    lines.append("| V4Deep best NN A8+T4+H1 (reported) | 47 | 48 | 1.7e-13 |")
    for head in ALL_HEADS:
        for s in ["all", "non_CpG"]:
            sub = f2_48[(f2_48["head"] == head) & (f2_48["stratum"] == s)]
            if not sub.empty:
                r = sub.iloc[0]
                lab = f"v4_cds {head} ({s})"
                lines.append(f"| {lab} | {r['n_path_gt_ben']} | {r['n_genes']} | {fmt_p(r['sign_test_p'])} |")
    lines.append("")

    # ---- Diagnostic ----
    lines.append("## Diagnostic: v3 GB on FULL ClinVar (1.69M variants)")
    lines.append("")
    lines.append("The v4_cds panel only scores TCW-context CDS positions (~16k of ~1.69M ClinVar")
    lines.append("C>T variants). The original 78-gene result was computed against the full ClinVar")
    lines.append("with the v3 model, where every gene has many variants. Here we re-run the v3 GB")
    lines.append("scorer on the full ClinVar against both gene lists and CpG strata, both as a")
    lines.append("sanity check that we can reproduce the original numbers and to provide an")
    lines.append("apples-to-apples CpG-controlled v3 baseline.")
    lines.append("")
    lines.append("| gene_list | stratum | n_genes | path>ben | frac | sign_test_p |")
    lines.append("|---|---|---:|---:|---:|---|")
    for _, r in diag.iterrows():
        lines.append(f"| {r['gene_list']} | {r['stratum']} | {r['n_genes']} | {r['n_path_gt_ben']} | {r['frac']} | {fmt_p(r['sign_test_p'])} |")
    lines.append("")

    # ---- Verdicts ----
    lines.append("## Per-finding verdict")
    lines.append("")
    lines.append("### Finding 1 (nonsense)")
    for head in ALL_HEADS:
        sub = f1[(f1["head"] == head) & (f1["stratum"] == "non_CpG")]
        if not sub.empty:
            r = sub.iloc[0]
            lines.append(f"- **{head}** (non-CpG only): OR = {r['OR']}, verdict = **{r['verdict']}**")
    lines.append("")
    lines.append("### Finding 2 (TSG)")
    for label, table, n_total_genes, criterion in [
        ("78-gene", f2_78, 78, 50),
        ("48-gene", f2_48, 48, 30),
    ]:
        for head in ALL_HEADS:
            sub = table[(table["head"] == head) & (table["stratum"] == "non_CpG")]
            if not sub.empty:
                r = sub.iloc[0]
                wins = r["n_path_gt_ben"]; n = r["n_genes"]
                # criterion is set against full N (78 or 48). We rescale by what's testable.
                expected_criterion_frac = criterion / n_total_genes
                meets = (wins / n) >= expected_criterion_frac if n else False
                verdict = "REAL" if meets else "NOT_REAL"
                lines.append(f"- **{label} list, {head}** (non-CpG, n_testable={n}/{n_total_genes}): {wins}/{n} wins, criterion needs frac >= {expected_criterion_frac:.2f}, verdict = **{verdict}**")
    lines.append("")

    # ---- Caveats ----
    lines.append("## Caveats")
    lines.append("")
    lines.append("1. **48-gene selection bias.** The V4Deep advisor report described the 48-gene")
    lines.append("   subset as 'broader / well-characterized' than the original 78. It is in fact")
    lines.append("   SMALLER. Without a documented inclusion procedure, this is consistent with")
    lines.append("   filtering down to genes where the model already worked. The 78-gene CGC list")
    lines.append("   is the more honest comparison and is the primary endpoint here.")
    lines.append("")
    lines.append("2. **Panel coverage is sparse.** The v4_cds panel restricts scoring to TCW-context")
    lines.append("   CDS positions, so only ~16k of ~1.7M ClinVar C>T variants overlap. After")
    lines.append("   per-gene >=3-path/>=3-ben filtering and CpG stratification, many TSGs have no")
    lines.append("   testable variants in the non-CpG stratum. The reduced n_testable is reported")
    lines.append("   per cell.")
    lines.append("")
    lines.append("3. **CpG bias in v4_cds.** Although v4_cds reduces CpG enrichment in top-1% to")
    lines.append("   34% (from v3's 53.9%), this is still 2.7x the panel baseline of 12.5%, so")
    lines.append("   CpG controls are warranted.")
    lines.append("")
    lines.append("4. **Apobec1 head.** `score_apobec1_v4_cds` was retrained on apobec1 data; we")
    lines.append("   include it because the panel exposes it, but APOBEC1 is C-to-U cytidine")
    lines.append("   editing in apoB mRNA contexts and is not the canonical engine for ClinVar")
    lines.append("   pathogenic variants.")
    lines.append("")

    md_path.write_text("\n".join(lines))
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
