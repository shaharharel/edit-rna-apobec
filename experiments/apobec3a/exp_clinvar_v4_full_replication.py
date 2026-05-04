"""Re-run v3 advisor's two ClinVar findings on the FULL 1.69M ClinVar set
using the just-scored v4 heads.

Findings replicated:
  1. Nonsense top-1000 enrichment per (head x stratum) on pathogenic+likely_pathogenic.
  2. Per-TSG sign test (path > ben) per (gene_list x head x stratum), on three gene lists:
       - 48-gene curated (hardcoded)
       - 82-gene Tier 1 TSG proxy (Sondka 2018 Tier 1 TSGs)
       - 173-gene OncoKB sangerCGC TSG-only (live API fetch)

Inputs:
  - experiments/apobec3a/outputs/clinvar_v4_scored/clinvar_scores_v4.parquet  (1.69M, 6 v4 heads)
  - data/processed/clinvar_c2u_variants.csv                                   (significance, mol_consequence, gene)
  - experiments/apobec3a/outputs/clinvar_prediction/clinvar_all_scores.csv   (v3 GB scores)
  - data/raw/genomes/hg19.fa                                                  (CpG context)

Outputs (under experiments/apobec3a/outputs/clinvar_v4_full/):
  - finding1_nonsense_v4_full.csv
  - finding2_tsg_v4_full.csv
  - per_gene_path_vs_ben_full.csv
  - CLINVAR_V4_FULL_RESULTS.md
"""
from __future__ import annotations

import json
import ssl
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from pyfaidx import Fasta
from scipy import stats
from scipy.stats import binomtest, fisher_exact

PROJECT_ROOT = Path("/Users/shaharharel/Documents/github/edit-rna-apobec")
SEED = 20260427
np.random.seed(SEED)

OUT_DIR = PROJECT_ROOT / "experiments" / "apobec3a" / "outputs" / "clinvar_v4_full"
OUT_DIR.mkdir(parents=True, exist_ok=True)

V4_PARQUET = PROJECT_ROOT / "experiments" / "apobec3a" / "outputs" / "clinvar_v4_scored" / "clinvar_scores_v4.parquet"
CLINVAR_META = PROJECT_ROOT / "data" / "processed" / "clinvar_c2u_variants.csv"
V3_GB_CSV = PROJECT_ROOT / "experiments" / "apobec3a" / "outputs" / "clinvar_prediction" / "clinvar_all_scores.csv"
HG19 = PROJECT_ROOT / "data" / "raw" / "genomes" / "hg19.fa"

# Heads to evaluate (in order)
HEADS = [
    "v3_GB",
    "score_binary_v4",
    "score_A3A_v4",
    "score_A3B_v4",
    "score_A3A_A3G_v4",
    "score_apobec1_v4_cds",
]

# Significance categories
PATH = {"Pathogenic", "Likely_pathogenic"}
BEN = {"Benign", "Likely_benign"}

# ---------------------------------------------------------------------------
# Gene lists
# ---------------------------------------------------------------------------

CURATED_48 = {
    "TP53", "RB1", "APC", "BRCA1", "BRCA2", "PTEN", "VHL", "WT1", "NF1", "NF2",
    "TSC1", "TSC2", "SMAD4", "CDKN2A", "CDH1", "MEN1", "MLH1", "MSH2", "MSH6",
    "PMS2", "STK11", "PTCH1", "SDHB", "SDHD", "SDHA", "SDHC", "SDHAF2",
    "BAP1", "SMARCB1", "ARID1A", "ARID1B", "KDM6A", "FBXW7", "ATM", "ATR",
    "CHEK2", "PALB2", "MUTYH", "BMPR1A", "DICER1", "FLCN", "FH", "MAX",
    "TMEM127", "SUFU", "EPCAM", "NBN", "WRN",
}

# 82-gene Tier-1 TSG proxy (Sondka 2018 + standard literature lists)
TIER1_TSG = {
    "TP53","RB1","APC","BRCA1","BRCA2","PTEN","VHL","WT1","NF1","NF2","TSC1","TSC2",
    "SMAD4","CDKN2A","CDH1","MEN1","MLH1","MSH2","MSH6","PMS2","STK11","PTCH1","SDHB",
    "SDHD","SDHA","SDHC","SDHAF2","BAP1","SMARCB1","ARID1A","ARID1B","KDM6A","FBXW7",
    "ATM","ATR","CHEK2","PALB2","MUTYH","BMPR1A","DICER1","FLCN","FH","MAX","TMEM127",
    "SUFU","EPCAM","NBN","WRN",
    "AXIN1","AXIN2","BLM","CYLD","EXT1","EXT2","GATA3","HNF1A","HNF1B",
    "KEAP1","NOTCH1","NOTCH2","NPM1","PRDM1","RAD51C","RAD51D","RNF43",
    "RUNX1","SETD2","SMARCA4","SMARCE1","SOCS1","SPEN","TET2","TNFAIP3",
    "ZNRF3","ASXL1","BCOR","CIC","CREBBP","DAXX","DNMT3A","EP300","FAT1",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fetch_oncokb_tsg() -> set[str]:
    """Fetch OncoKB sangerCGC=True AND geneType=='TSG' (~173 genes)."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    try:
        req = urllib.request.Request(
            "https://www.oncokb.org/api/v1/utils/cancerGeneList",
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with urllib.request.urlopen(req, timeout=30, context=ctx) as r:
            data = json.loads(r.read())
        tsgs = {g["hugoSymbol"] for g in data if g.get("sangerCGC") and g.get("geneType") == "TSG"}
        return tsgs
    except Exception as e:
        print(f"[WARN] OncoKB fetch failed: {e}")
        return set()


def haldane_fisher(a: int, b: int, c: int, d: int):
    """Fisher's exact + OR with Haldane correction.

    2x2 table:
        path_topK    other
        nonsense        a       b
        non-nonsense    c       d
    OR = (a*d) / (b*c) with +0.5 to all cells if any zero.
    """
    if a == 0 or b == 0 or c == 0 or d == 0:
        a_, b_, c_, d_ = a + 0.5, b + 0.5, c + 0.5, d + 0.5
        or_val = (a_ * d_) / (b_ * c_)
    else:
        or_val = (a * d) / (b * c)
    # 95% CI on log(OR) via SE
    se = np.sqrt(1 / (a + 0.5) + 1 / (b + 0.5) + 1 / (c + 0.5) + 1 / (d + 0.5))
    log_or = np.log(or_val)
    ci_lo = float(np.exp(log_or - 1.96 * se))
    ci_hi = float(np.exp(log_or + 1.96 * se))
    _, p = fisher_exact([[a, b], [c, d]], alternative="greater")
    return float(or_val), ci_lo, ci_hi, float(p)


def annotate_cpg(df: pd.DataFrame) -> pd.Series:
    """Return is_cpg boolean. CpG = next base is G (on + strand) or prev base is C (on - strand).

    For C>T variants on + strand: CpG iff hg19[pos]=='C' and hg19[pos+1]=='G'.
    For G>A variants on - strand: editing is at C on opposite strand, so CpG iff hg19[pos]=='G' and hg19[pos-1]=='C'.
    """
    fa = Fasta(str(HG19), as_raw=True, sequence_always_upper=True)
    is_cpg = np.zeros(len(df), dtype=bool)
    chroms = df["chrom"].values
    poss = df["pos"].values.astype(int)
    strands = df["strand"].values

    # Build per-chrom indices to limit Fasta lookups
    # pyfaidx is 0-based in slicing but our pos is 1-based (VCF style)
    for i in range(len(df)):
        ch = chroms[i]
        p = poss[i]
        st = strands[i]
        try:
            seq = fa[ch][p - 2 : p + 1]  # 1-based pos -> 0-based slice [p-2 : p+1] gives [pos-1, pos, pos+1]
            # seq is length 3 typically: prev, this, next
            if len(seq) < 3:
                continue
            prev_b, this_b, next_b = seq[0], seq[1], seq[2]
            if st == "+":
                # editing at C: CpG = this=='C' and next=='G'
                if this_b == "C" and next_b == "G":
                    is_cpg[i] = True
            else:
                # editing at C on - strand: + strand has G at pos, C at pos-1
                if this_b == "G" and prev_b == "C":
                    is_cpg[i] = True
        except (KeyError, IndexError):
            continue
    return pd.Series(is_cpg, index=df.index)


# ---------------------------------------------------------------------------
# Step 1: Build unified table
# ---------------------------------------------------------------------------

print("=" * 80)
print("Step 1. Build unified per-variant table")
print("=" * 80)

print("Loading v4 parquet ...")
v4 = pd.read_parquet(V4_PARQUET)
print(f"  v4 rows: {len(v4):,}; cols: {list(v4.columns)}")

print("Loading ClinVar metadata ...")
meta = pd.read_csv(
    CLINVAR_META,
    usecols=["site_id", "clinical_significance", "molecular_consequence", "gene", "ref", "alt", "editing_strand", "chr", "start"],
)
print(f"  meta rows: {len(meta):,}")

# significance_simple mapping (mirror of v3 mapping)
def simplify_sig(s):
    if pd.isna(s):
        return "Other"
    if s in ("Pathogenic", "Pathogenic/Likely_pathogenic"):
        return "Pathogenic"
    if s in ("Likely_pathogenic",):
        return "Likely_pathogenic"
    if s in ("Benign", "Benign/Likely_benign"):
        return "Benign"
    if s == "Likely_benign":
        return "Likely_benign"
    if s == "Uncertain_significance":
        return "VUS"
    if s == "Conflicting_classifications_of_pathogenicity":
        return "Conflicting"
    return "Other"


meta["significance_simple"] = meta["clinical_significance"].map(simplify_sig)

# molecular consequence simplification (nonsense / missense / synonymous / other)
def simplify_consequence(c):
    if pd.isna(c) or c == "unknown":
        return "other"
    if "nonsense" in c:
        return "nonsense"
    if "missense" in c:
        return "missense"
    if "synonymous" in c:
        return "synonymous"
    if "splice_donor" in c or "splice_acceptor" in c:
        return "splice"
    return "other"


meta["consequence_simple"] = meta["molecular_consequence"].map(simplify_consequence)

print("Loading v3 GB scores ...")
v3 = pd.read_csv(V3_GB_CSV, usecols=["site_id", "p_edited_gb"])
v3 = v3.rename(columns={"p_edited_gb": "v3_GB"})
print(f"  v3 rows: {len(v3):,}")

# Join: v4 ⨝ meta ⨝ v3 by site_id
print("Joining ...")
df = v4.merge(meta, on="site_id", how="inner")
print(f"  v4 + meta: {len(df):,}")
df = df.merge(v3, on="site_id", how="left")
print(f"  v4 + meta + v3: {len(df):,}; v3 NaN: {df['v3_GB'].isna().sum():,}")

# Rename v4 score columns to canonical names
df = df.rename(
    columns={
        "score_binary": "score_binary_v4",
        "score_A3A": "score_A3A_v4",
        "score_A3B": "score_A3B_v4",
        "score_A3G": "score_A3G_v4",
        "score_A3A_A3G": "score_A3A_A3G_v4",
        # score_apobec1_v4_cds keeps name
    }
)

# CpG annotation
print("Annotating CpG context (this may take 1-3 min for 1.69M variants) ...")
df["is_cpg"] = annotate_cpg(df).values
print(f"  CpG: {df['is_cpg'].sum():,}; non-CpG: {(~df['is_cpg']).sum():,}")

# ---------------------------------------------------------------------------
# Coverage breakdown
# ---------------------------------------------------------------------------

print("\nCoverage breakdown:")
sig_counts = df["significance_simple"].value_counts()
print(sig_counts)

cov = {
    "total": len(df),
    "with_v3_GB": int(df["v3_GB"].notna().sum()),
    "pathogenic": int((df["significance_simple"] == "Pathogenic").sum()),
    "likely_pathogenic": int((df["significance_simple"] == "Likely_pathogenic").sum()),
    "benign": int((df["significance_simple"] == "Benign").sum()),
    "likely_benign": int((df["significance_simple"] == "Likely_benign").sum()),
    "cpg": int(df["is_cpg"].sum()),
    "non_cpg": int((~df["is_cpg"]).sum()),
    "nonsense": int((df["consequence_simple"] == "nonsense").sum()),
    "missense": int((df["consequence_simple"] == "missense").sum()),
}
print(json.dumps(cov, indent=2))

# ---------------------------------------------------------------------------
# Step 2: FINDING 1 — Nonsense top-1000 (per head x stratum)
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("Step 2. FINDING 1 — Nonsense top-1000 per head x stratum")
print("=" * 80)

# Restrict to pathogenic + likely_pathogenic (consistent with v3 advisor's analysis)
path_mask = df["significance_simple"].isin(PATH)
print(f"Path+LP universe: {path_mask.sum():,}")

# Strata definitions
strata = {
    "all": np.ones(len(df), dtype=bool),
    "CpG": df["is_cpg"].values,
    "non_CpG": (~df["is_cpg"]).values,
}

TOP_K = 1000
finding1_rows = []

for head in HEADS:
    if head not in df.columns:
        print(f"[skip] {head}: column missing")
        continue
    scores = df[head].values
    valid_score = ~np.isnan(scores)

    for stratum_name, stratum_mask in strata.items():
        sm = path_mask.values & stratum_mask & valid_score
        n_path_in_stratum = int(sm.sum())
        if n_path_in_stratum < 100:
            continue

        nonsense_mask = (df["consequence_simple"] == "nonsense").values
        # Baseline (within stratum, within path)
        baseline_nonsense_n = int((nonsense_mask & sm).sum())
        baseline_nonsense_rate = baseline_nonsense_n / n_path_in_stratum

        # Top-K within stratum
        idx_in_stratum = np.where(sm)[0]
        scores_strat = scores[idx_in_stratum]
        k = min(TOP_K, n_path_in_stratum)
        top_idx = idx_in_stratum[np.argsort(scores_strat)[-k:]]
        top_nonsense_n = int(nonsense_mask[top_idx].sum())
        top_nonsense_rate = top_nonsense_n / k

        # 2x2 contingency: nonsense among top-K vs nonsense among rest of stratum
        rest_mask = sm.copy()
        rest_mask[top_idx] = False
        rest_n = int(rest_mask.sum())
        rest_nonsense_n = int((nonsense_mask & rest_mask).sum())

        a = top_nonsense_n
        b = k - top_nonsense_n
        c = rest_nonsense_n
        d = rest_n - rest_nonsense_n
        or_val, ci_lo, ci_hi, p = haldane_fisher(a, b, c, d)

        finding1_rows.append({
            "head": head,
            "stratum": stratum_name,
            "n_in_stratum": n_path_in_stratum,
            "top_k": k,
            "baseline_nonsense_n": baseline_nonsense_n,
            "baseline_nonsense_rate": round(baseline_nonsense_rate, 4),
            "topk_nonsense_n": top_nonsense_n,
            "topk_nonsense_rate": round(top_nonsense_rate, 4),
            "OR": round(or_val, 3),
            "CI_lo": round(ci_lo, 3),
            "CI_hi": round(ci_hi, 3),
            "p_fisher": p,
        })
        print(f"  {head:30s} {stratum_name:8s} n={n_path_in_stratum:6d} k={k} top_ns={top_nonsense_n} OR={or_val:.3f} [{ci_lo:.2f},{ci_hi:.2f}] p={p:.2e}")

f1_df = pd.DataFrame(finding1_rows)
def verdict(row):
    # Verdict on the non_CpG row
    return "PASS" if row["OR"] > 1.3 else ("WEAK" if row["OR"] > 1.0 else "FAIL")

f1_df["verdict"] = f1_df.apply(verdict, axis=1)
f1_df.to_csv(OUT_DIR / "finding1_nonsense_v4_full.csv", index=False)
print(f"\nWrote {OUT_DIR / 'finding1_nonsense_v4_full.csv'}")

# ---------------------------------------------------------------------------
# Step 3: FINDING 2 — TSG sign test (gene_list x head x stratum)
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("Step 3. FINDING 2 — Per-TSG sign test (full ClinVar)")
print("=" * 80)

print("Fetching OncoKB CGC TSG list ...")
ONCOKB_TSG = fetch_oncokb_tsg()
print(f"  OncoKB sangerCGC TSG-only: {len(ONCOKB_TSG)}")

GENE_LISTS = {
    "curated_48": CURATED_48,
    "tier1_proxy_82": TIER1_TSG,
    "oncokb_cgc_tsg_173": ONCOKB_TSG,
}

# Path/Ben masks
path_mask_arr = df["significance_simple"].isin(PATH).values
ben_mask_arr = df["significance_simple"].isin(BEN).values
genes_arr = df["gene"].fillna("").values

per_gene_rows = []
finding2_rows = []

for gl_name, gl in GENE_LISTS.items():
    if not gl:
        continue
    for head in HEADS:
        if head not in df.columns:
            continue
        scores = df[head].values
        valid = ~np.isnan(scores)
        for stratum_name, stratum_mask in strata.items():
            in_path = path_mask_arr & stratum_mask & valid
            in_ben = ben_mask_arr & stratum_mask & valid

            n_testable = 0
            n_wins = 0
            for gene in gl:
                gm = (genes_arr == gene)
                gp = gm & in_path
                gb = gm & in_ben
                np_n = int(gp.sum())
                nb_n = int(gb.sum())
                if np_n < 3 or nb_n < 3:
                    continue
                ps = scores[gp]
                bs = scores[gb]
                mean_p = float(ps.mean())
                mean_b = float(bs.mean())
                diff = mean_p - mean_b
                try:
                    _, mwu_p = stats.mannwhitneyu(ps, bs, alternative="two-sided")
                except ValueError:
                    mwu_p = 1.0
                won = bool(diff > 0)
                if won:
                    n_wins += 1
                n_testable += 1

                # Save full per-gene row only for stratum=='all' (avoid 3x bloat)
                if stratum_name == "all":
                    per_gene_rows.append({
                        "gene_list": gl_name,
                        "head": head,
                        "gene": gene,
                        "n_path": np_n,
                        "n_ben": nb_n,
                        "mean_path": round(mean_p, 4),
                        "mean_ben": round(mean_b, 4),
                        "delta": round(diff, 4),
                        "path_gt_ben": won,
                        "mwu_p": float(mwu_p),
                    })

            if n_testable > 0:
                sign_p = binomtest(n_wins, n_testable, 0.5, alternative="greater").pvalue
            else:
                sign_p = 1.0

            finding2_rows.append({
                "gene_list": gl_name,
                "n_genes_in_list": len(gl),
                "head": head,
                "stratum": stratum_name,
                "n_genes_testable": n_testable,
                "n_path_gt_ben": n_wins,
                "frac_wins": round(n_wins / n_testable, 4) if n_testable > 0 else 0.0,
                "sign_test_p": float(sign_p),
            })
            print(f"  {gl_name:24s} {head:30s} {stratum_name:8s} testable={n_testable:4d} wins={n_wins:4d} p={sign_p:.2e}")

f2_df = pd.DataFrame(finding2_rows)
f2_df.to_csv(OUT_DIR / "finding2_tsg_v4_full.csv", index=False)
print(f"\nWrote {OUT_DIR / 'finding2_tsg_v4_full.csv'}")

per_gene_df = pd.DataFrame(per_gene_rows)
per_gene_df.to_csv(OUT_DIR / "per_gene_path_vs_ben_full.csv", index=False)
print(f"Wrote {OUT_DIR / 'per_gene_path_vs_ben_full.csv'} ({len(per_gene_df)} rows)")

# ---------------------------------------------------------------------------
# Step 4: Markdown report
# ---------------------------------------------------------------------------

md_lines = []
md_lines.append("# ClinVar v4 Heads — Full 1.69M Replication of v3 Advisor Findings")
md_lines.append("")
md_lines.append(f"Random seed: {SEED}")
md_lines.append("")
md_lines.append("## 1. Coverage")
md_lines.append("")
md_lines.append("| Slice | N |")
md_lines.append("|---|---|")
for k, v in cov.items():
    md_lines.append(f"| {k} | {v:,} |")
md_lines.append("")
md_lines.append("## 2. Finding 1 — Nonsense top-1000 enrichment (path+LP)")
md_lines.append("")
md_lines.append("Top-K = min(1000, N_in_stratum). 2x2 contingency: (nonsense in top-K) vs (nonsense in rest of stratum). Haldane-Anscombe correction. Verdict on each row's OR (PASS=>1.3, WEAK=1.0-1.3, FAIL<=1.0).")
md_lines.append("")
md_lines.append("| Head | Stratum | N path | top-K | top nonsense | rate | baseline rate | OR | 95% CI | p | Verdict |")
md_lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
for _, r in f1_df.iterrows():
    md_lines.append(
        f"| {r['head']} | {r['stratum']} | {r['n_in_stratum']:,} | {r['top_k']:,} | {r['topk_nonsense_n']:,} | {r['topk_nonsense_rate']:.4f} | {r['baseline_nonsense_rate']:.4f} | {r['OR']:.3f} | [{r['CI_lo']:.2f}, {r['CI_hi']:.2f}] | {r['p_fisher']:.2e} | {r['verdict']} |"
    )
md_lines.append("")

# Headline numbers
md_lines.append("### Finding 1 — Headline")
md_lines.append("")
for head in HEADS:
    sub = f1_df[f1_df["head"] == head]
    if sub.empty:
        continue
    top_or_row = sub.loc[sub["OR"].idxmax()]
    non_cpg = sub[sub["stratum"] == "non_CpG"]
    if len(non_cpg):
        nc = non_cpg.iloc[0]
        md_lines.append(
            f"- **{head}**: best OR={top_or_row['OR']:.3f} ({top_or_row['stratum']}); non-CpG OR={nc['OR']:.3f} (verdict {nc['verdict']})"
        )
md_lines.append("")

md_lines.append("## 3. Finding 2 — Per-TSG sign test")
md_lines.append("")
md_lines.append("Per gene: require >=3 path AND >=3 ben C>T variants in ClinVar. Sign test on (mean_path > mean_ben) wins. Binomial one-sided p (greater).")
md_lines.append("")
md_lines.append("| Gene list | List size | Head | Stratum | testable | path>ben | frac | sign-p |")
md_lines.append("|---|---|---|---|---|---|---|---|")
for _, r in f2_df.iterrows():
    md_lines.append(
        f"| {r['gene_list']} | {r['n_genes_in_list']} | {r['head']} | {r['stratum']} | {r['n_genes_testable']} | {r['n_path_gt_ben']} | {r['frac_wins']:.3f} | {r['sign_test_p']:.2e} |"
    )
md_lines.append("")

# v3 vs v4 comparison
md_lines.append("## 4. v3 (GB) vs v4 head-to-head")
md_lines.append("")
md_lines.append("Same gene lists, stratum=all only.")
md_lines.append("")
md_lines.append("| Gene list | Head | testable | wins | sign-p |")
md_lines.append("|---|---|---|---|---|")
for gl in GENE_LISTS:
    for head in HEADS:
        sub = f2_df[(f2_df["gene_list"] == gl) & (f2_df["head"] == head) & (f2_df["stratum"] == "all")]
        if sub.empty:
            continue
        r = sub.iloc[0]
        md_lines.append(f"| {gl} | {head} | {r['n_genes_testable']} | {r['n_path_gt_ben']} | {r['sign_test_p']:.2e} |")
md_lines.append("")

md_lines.append("## 5. Gene-list discussion")
md_lines.append("")
md_lines.append("- **48-gene curated** list is hand-picked: convenient but susceptible to selection bias because it cherry-picks high-prevalence TSGs (TP53, BRCA1, BRCA2, ATM) which dominate the path/ben statistics. Statistical wins on this list partly reflect the top-of-list genes' editing-friendly motif densities, not a true population-level effect.")
md_lines.append("- **Tier1 82-gene proxy** is broader (Sondka 2018 + literature). Less prone to cherry-picking but still biased toward well-studied, mutation-rich genes.")
md_lines.append("- **OncoKB sangerCGC TSG-only (173)** is the most defensible: it's an externally curated source (COSMIC CGC + OncoKB), independent of this project. The previous v3-advisor agent already used this list and reported 173 base / 128 testable.")
md_lines.append("")
md_lines.append("Use the **OncoKB 173** list as the primary; report 48 / 82 only as sensitivity checks to show the sign of the effect doesn't depend on the gene list.")
md_lines.append("")

# Verdicts
md_lines.append("## 6. Verdicts")
md_lines.append("")
# Finding 1: do any v4 heads PASS on non-CpG?
v4_pass = f1_df[(f1_df["head"].str.contains("v4")) & (f1_df["stratum"] == "non_CpG") & (f1_df["verdict"] == "PASS")]
v3_pass = f1_df[(f1_df["head"] == "v3_GB") & (f1_df["stratum"] == "non_CpG")]
v3_v = v3_pass.iloc[0]["verdict"] if len(v3_pass) else "N/A"
md_lines.append(f"- **Finding 1 (nonsense)**: v3 GB non-CpG verdict = `{v3_v}`. v4 heads with PASS on non-CpG: {len(v4_pass)} of {len(HEADS)-1}.")
oncokb_v3 = f2_df[(f2_df["gene_list"] == "oncokb_cgc_tsg_173") & (f2_df["head"] == "v3_GB") & (f2_df["stratum"] == "all")]
oncokb_v4 = f2_df[(f2_df["gene_list"] == "oncokb_cgc_tsg_173") & (f2_df["head"].str.contains("v4")) & (f2_df["stratum"] == "all")]
v3_p = float(oncokb_v3.iloc[0]["sign_test_p"]) if len(oncokb_v3) else 1.0
md_lines.append(f"- **Finding 2 (TSG sign)** on OncoKB-173 (stratum=all): v3 GB sign-p={v3_p:.2e}; v4 heads passing nominal 0.05: {(oncokb_v4['sign_test_p'] < 0.05).sum()} of {len(oncokb_v4)}.")
md_lines.append("")

# Save markdown
md_path = OUT_DIR / "CLINVAR_V4_FULL_RESULTS.md"
md_path.write_text("\n".join(md_lines))
print(f"\nWrote {md_path}")

print("\nDONE")
