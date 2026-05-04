"""TSG-specificity control for the v4 ClinVar TSG sign-test finding.

Background:
  exp_clinvar_v4_full_replication.py reports that 110/128 (86%) of OncoKB CGC
  TSGs have mean(pathogenic ClinVar score) > mean(benign ClinVar score), p=1.3e-17
  on v3 GB / v4 binary scores. We claim this means "pathogenic variants in TSGs
  sit at higher-editability positions than benign variants" --- implying
  APOBEC-targeted positions are also stress-vulnerable.

  This control tests whether the effect is TSG-specific or a general "model picks
  pathogenic-distinctive positions" pattern that would also appear in non-TSG
  genes (oncogenes, random non-cancer genes).

Design:
  Three gene lists:
    1. TSGs (already-analysed reference; reused from per_gene_path_vs_ben_full.csv)
    2. Oncogenes: OncoKB sangerCGC=True AND geneType=='ONCOGENE' (~205 base)
    3. Random non-cancer: 200-gene random sample drawn from genes NOT in any
       sangerCGC role (any of TSG / ONCOGENE / ONCOGENE_AND_TSG / etc.).
       Filter to genes with >=3 path AND >=3 ben in ClinVar.

  Heads tested:
    - v3_GB
    - score_binary_v4
    - score_A3A_v4

  Strata:
    - all
    - CpG
    - non_CpG

  Per cell (gene_list x head x stratum):
    - Per gene: require >=3 path+LP AND >=3 ben+LB ClinVar C>T variants
    - Sign test (binomial, one-sided greater): how many genes have
      mean(path_score) > mean(ben_score)?

Outputs:
  experiments/apobec3a/outputs/clinvar_v4_full/tsg_specificity_control.csv
  experiments/apobec3a/outputs/clinvar_v4_full/tsg_specificity_per_gene.csv
  experiments/apobec3a/outputs/clinvar_v4_full/TSG_SPECIFICITY_RESULTS.md
"""
from __future__ import annotations

import json
import ssl
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from pyfaidx import Fasta
from scipy import stats
from scipy.stats import binomtest

PROJECT_ROOT = Path("/Users/shaharharel/Documents/github/edit-rna-apobec")
SEED = 20260427
np.random.seed(SEED)

OUT_DIR = PROJECT_ROOT / "experiments" / "apobec3a" / "outputs" / "clinvar_v4_full"
OUT_DIR.mkdir(parents=True, exist_ok=True)

V4_PARQUET = PROJECT_ROOT / "experiments" / "apobec3a" / "outputs" / "clinvar_v4_scored" / "clinvar_scores_v4.parquet"
CLINVAR_META = PROJECT_ROOT / "data" / "processed" / "clinvar_c2u_variants.csv"
V3_GB_CSV = PROJECT_ROOT / "experiments" / "apobec3a" / "outputs" / "clinvar_prediction" / "clinvar_all_scores.csv"
HG19 = PROJECT_ROOT / "data" / "raw" / "genomes" / "hg19.fa"

# Heads tested
HEADS = ["v3_GB", "score_binary_v4", "score_A3A_v4"]

PATH = {"Pathogenic", "Likely_pathogenic"}
BEN = {"Benign", "Likely_benign"}

N_RANDOM_GENES = 200


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fetch_oncokb_full() -> list[dict]:
    """Fetch full OncoKB cancerGeneList payload."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(
        "https://www.oncokb.org/api/v1/utils/cancerGeneList",
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(req, timeout=30, context=ctx) as r:
        return json.loads(r.read())


def annotate_cpg(df: pd.DataFrame) -> pd.Series:
    """Return is_cpg boolean. CpG = next base is G (+ strand) or prev base is C (- strand)."""
    fa = Fasta(str(HG19), as_raw=True, sequence_always_upper=True)
    is_cpg = np.zeros(len(df), dtype=bool)
    chroms = df["chrom"].values
    poss = df["pos"].values.astype(int)
    strands = df["strand"].values
    for i in range(len(df)):
        ch = chroms[i]
        p = poss[i]
        st = strands[i]
        try:
            seq = fa[ch][p - 2 : p + 1]
            if len(seq) < 3:
                continue
            prev_b, this_b, next_b = seq[0], seq[1], seq[2]
            if st == "+":
                if this_b == "C" and next_b == "G":
                    is_cpg[i] = True
            else:
                if this_b == "G" and prev_b == "C":
                    is_cpg[i] = True
        except (KeyError, IndexError):
            continue
    return pd.Series(is_cpg, index=df.index)


def simplify_sig(s):
    if pd.isna(s):
        return "Other"
    if s in ("Pathogenic", "Pathogenic/Likely_pathogenic"):
        return "Pathogenic"
    if s == "Likely_pathogenic":
        return "Likely_pathogenic"
    if s in ("Benign", "Benign/Likely_benign"):
        return "Benign"
    if s == "Likely_benign":
        return "Likely_benign"
    return "Other"


# ---------------------------------------------------------------------------
# Step 1. Build unified table
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
    usecols=["site_id", "clinical_significance", "gene"],
)
meta["significance_simple"] = meta["clinical_significance"].map(simplify_sig)
print(f"  meta rows: {len(meta):,}")

print("Loading v3 GB scores ...")
v3 = pd.read_csv(V3_GB_CSV, usecols=["site_id", "p_edited_gb"])
v3 = v3.rename(columns={"p_edited_gb": "v3_GB"})
print(f"  v3 rows: {len(v3):,}")

print("Joining ...")
df = v4.merge(meta, on="site_id", how="inner")
df = df.merge(v3, on="site_id", how="left")
df = df.rename(columns={"score_binary": "score_binary_v4", "score_A3A": "score_A3A_v4"})
print(f"  joined rows: {len(df):,}; v3 NaN: {df['v3_GB'].isna().sum():,}")

# CpG annotation
print("Annotating CpG context ...")
df["is_cpg"] = annotate_cpg(df).values
print(f"  CpG: {df['is_cpg'].sum():,}; non-CpG: {(~df['is_cpg']).sum():,}")

# ---------------------------------------------------------------------------
# Step 2. Build gene lists
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("Step 2. Build gene lists (TSG / Oncogene / Random-non-cancer)")
print("=" * 80)

print("Fetching OncoKB cancerGeneList ...")
oncokb = fetch_oncokb_full()
print(f"  total entries: {len(oncokb):,}")

ONCOKB_TSG = {g["hugoSymbol"] for g in oncokb if g.get("sangerCGC") and g.get("geneType") == "TSG"}
ONCOKB_ONCO = {g["hugoSymbol"] for g in oncokb if g.get("sangerCGC") and g.get("geneType") == "ONCOGENE"}
# all sangerCGC genes (any role) -> exclusion universe for "random non-cancer"
ALL_CGC = {g["hugoSymbol"] for g in oncokb if g.get("sangerCGC")}
print(f"  OncoKB sangerCGC TSG-only:      {len(ONCOKB_TSG)}")
print(f"  OncoKB sangerCGC ONCOGENE-only: {len(ONCOKB_ONCO)}")
print(f"  OncoKB sangerCGC any-role:      {len(ALL_CGC)} (used as exclusion for random list)")

# Build "random non-cancer" gene list:
#   1. all genes in ClinVar metadata with >=3 path AND >=3 ben (testable universe)
#   2. exclude any sangerCGC gene
#   3. random sample 200 genes (seed=20260427)

genes_arr_meta = meta["gene"].fillna("")
gene_path_n = meta[meta["significance_simple"].isin(PATH)].groupby("gene").size()
gene_ben_n = meta[meta["significance_simple"].isin(BEN)].groupby("gene").size()
testable_genes = set(gene_path_n[gene_path_n >= 3].index) & set(gene_ben_n[gene_ben_n >= 3].index)
print(f"  ClinVar testable universe (>=3 path AND >=3 ben): {len(testable_genes):,}")

non_cgc_pool = sorted(testable_genes - ALL_CGC - {""})
print(f"  Non-CGC testable pool: {len(non_cgc_pool):,}")
rng = np.random.default_rng(SEED)
RANDOM_GENES = set(rng.choice(non_cgc_pool, size=min(N_RANDOM_GENES, len(non_cgc_pool)), replace=False).tolist())
print(f"  Random non-cancer sample (seed={SEED}): {len(RANDOM_GENES)}")

GENE_LISTS = {
    "oncokb_cgc_tsg_173":   ONCOKB_TSG,
    "oncokb_cgc_onco_205":  ONCOKB_ONCO,
    "random_non_cgc_200":   RANDOM_GENES,
}

# ---------------------------------------------------------------------------
# Step 3. Per-gene sign test (gene_list x head x stratum)
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("Step 3. Per-gene sign test")
print("=" * 80)

path_mask_arr = df["significance_simple"].isin(PATH).values
ben_mask_arr = df["significance_simple"].isin(BEN).values
genes_arr = df["gene"].fillna("").values

strata = {
    "all":     np.ones(len(df), dtype=bool),
    "CpG":     df["is_cpg"].values,
    "non_CpG": (~df["is_cpg"]).values,
}

cell_rows = []
per_gene_rows = []

for gl_name, gl in GENE_LISTS.items():
    if not gl:
        continue
    # Pre-restrict to genes in this list to speed up
    gl_set = set(gl)
    gl_mask_global = pd.Series(genes_arr).isin(gl_set).values

    for head in HEADS:
        if head not in df.columns:
            print(f"  [skip] {head}: column missing")
            continue
        scores = df[head].values
        valid = ~np.isnan(scores)

        for stratum_name, stratum_mask in strata.items():
            in_path = path_mask_arr & stratum_mask & valid & gl_mask_global
            in_ben  = ben_mask_arr  & stratum_mask & valid & gl_mask_global

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

                # Save per-gene detail only on stratum=='all' to avoid 3x bloat
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

            sign_p = binomtest(n_wins, n_testable, 0.5, alternative="greater").pvalue if n_testable > 0 else 1.0

            cell_rows.append({
                "gene_list":         gl_name,
                "head":              head,
                "stratum":           stratum_name,
                "n_genes_in_list":   len(gl),
                "n_genes_testable":  n_testable,
                "n_path_gt_ben":     n_wins,
                "frac_wins":         round(n_wins / n_testable, 4) if n_testable > 0 else 0.0,
                "sign_test_p":       float(sign_p),
            })
            print(
                f"  {gl_name:24s} {head:18s} {stratum_name:8s} "
                f"testable={n_testable:4d} wins={n_wins:4d} "
                f"frac={(n_wins/n_testable if n_testable else 0):.3f} p={sign_p:.2e}"
            )

control_df = pd.DataFrame(cell_rows)
control_df.to_csv(OUT_DIR / "tsg_specificity_control.csv", index=False)
print(f"\nWrote {OUT_DIR / 'tsg_specificity_control.csv'}")

per_gene_df = pd.DataFrame(per_gene_rows)
per_gene_df.to_csv(OUT_DIR / "tsg_specificity_per_gene.csv", index=False)
print(f"Wrote {OUT_DIR / 'tsg_specificity_per_gene.csv'} ({len(per_gene_df)} rows)")

# ---------------------------------------------------------------------------
# Step 4. Markdown report
# ---------------------------------------------------------------------------

def fmt_p(p: float) -> str:
    if p < 1e-4:
        return f"{p:.1e}"
    return f"{p:.3f}"


md = []
md.append("# TSG Specificity Control for v4 ClinVar Sign-Test Finding")
md.append("")
md.append(f"Random seed: {SEED}")
md.append("")
md.append("## Question")
md.append("")
md.append(
    "The headline TSG finding (110/128 = 86% of OncoKB CGC TSGs show mean(pathogenic ClinVar score) "
    "> mean(benign), p=1.3e-17 on v3 GB / v4 binary) was claimed as evidence that **APOBEC-targeted "
    "positions are stress-vulnerable in tumor suppressors**. This control tests whether the same "
    "win-rate appears in non-TSG genes, which would mean the effect is a generic "
    "'pathogenic variants concentrate at high-scoring positions in any gene' phenomenon "
    "rather than TSG biology."
)
md.append("")
md.append("## Gene lists")
md.append("")
md.append("| Gene list | Source | Base size |")
md.append("|---|---|---|")
md.append(f"| TSGs              | OncoKB sangerCGC=True AND geneType=='TSG'      | {len(ONCOKB_TSG)} |")
md.append(f"| Oncogenes         | OncoKB sangerCGC=True AND geneType=='ONCOGENE' | {len(ONCOKB_ONCO)} |")
md.append(f"| Random non-cancer | Random sample of genes NOT in any OncoKB sangerCGC role, restricted to >=3 path AND >=3 ben in ClinVar (pool size {len(non_cgc_pool):,}) | {len(RANDOM_GENES)} |")
md.append("")
md.append("Test: per gene, require >=3 path+LP AND >=3 ben+LB ClinVar C>T variants. "
          "Sign test = binomial one-sided greater on n(mean_path > mean_ben).")
md.append("")

# ----- Headline table: v3_GB stratum=all
md.append("## Headline: v3 GB, stratum=all")
md.append("")
md.append("| Gene set | n_in_list | testable | wins | frac | sign-p |")
md.append("|---|---|---|---|---|---|")
order = ["oncokb_cgc_tsg_173", "oncokb_cgc_onco_205", "random_non_cgc_200"]
labels = {
    "oncokb_cgc_tsg_173":  "TSGs (OncoKB CGC)",
    "oncokb_cgc_onco_205": "Oncogenes (OncoKB CGC)",
    "random_non_cgc_200":  "Random non-cancer",
}
for gl in order:
    sub = control_df[(control_df["gene_list"] == gl) & (control_df["head"] == "v3_GB") & (control_df["stratum"] == "all")]
    if sub.empty:
        continue
    r = sub.iloc[0]
    md.append(
        f"| {labels[gl]} | {r['n_genes_in_list']} | {r['n_genes_testable']} | "
        f"{r['n_path_gt_ben']} | {r['frac_wins']*100:.1f}% | {fmt_p(r['sign_test_p'])} |"
    )
md.append("")

# ----- Full table
md.append("## Full results: gene_list x head x stratum")
md.append("")
md.append("| gene_list | head | stratum | testable | wins | frac | sign-p |")
md.append("|---|---|---|---|---|---|---|")
for _, r in control_df.iterrows():
    md.append(
        f"| {r['gene_list']} | {r['head']} | {r['stratum']} | "
        f"{r['n_genes_testable']} | {r['n_path_gt_ben']} | {r['frac_wins']:.3f} | {fmt_p(r['sign_test_p'])} |"
    )
md.append("")

# ----- Verdict logic
md.append("## Verdict")
md.append("")

tsg_row = control_df[(control_df["gene_list"] == "oncokb_cgc_tsg_173") & (control_df["head"] == "v3_GB") & (control_df["stratum"] == "all")].iloc[0]
onco_row = control_df[(control_df["gene_list"] == "oncokb_cgc_onco_205") & (control_df["head"] == "v3_GB") & (control_df["stratum"] == "all")].iloc[0]
rand_row = control_df[(control_df["gene_list"] == "random_non_cgc_200") & (control_df["head"] == "v3_GB") & (control_df["stratum"] == "all")].iloc[0]

tsg_frac = float(tsg_row["frac_wins"])
onco_frac = float(onco_row["frac_wins"])
rand_frac = float(rand_row["frac_wins"])

md.append(f"- **TSG win-rate (v3 GB, all):**            {tsg_frac*100:.1f}% ({int(tsg_row['n_path_gt_ben'])}/{int(tsg_row['n_genes_testable'])}, p={fmt_p(float(tsg_row['sign_test_p']))})")
md.append(f"- **Oncogene win-rate (v3 GB, all):**       {onco_frac*100:.1f}% ({int(onco_row['n_path_gt_ben'])}/{int(onco_row['n_genes_testable'])}, p={fmt_p(float(onco_row['sign_test_p']))})")
md.append(f"- **Random-non-cancer win-rate (v3 GB):**   {rand_frac*100:.1f}% ({int(rand_row['n_path_gt_ben'])}/{int(rand_row['n_genes_testable'])}, p={fmt_p(float(rand_row['sign_test_p']))})")
md.append("")

# Decision rule
def classify(tsg, onco, rand):
    """
    TSG-specific: TSG ~86%, others ~50%
    General:      all ~80%+
    Partial:     TSG 86%, others 60-75%
    """
    if rand >= 0.75 and onco >= 0.75:
        return ("GENERAL",
                "All three gene sets show similar high win-rates. The effect is NOT TSG-specific. "
                "It reflects a generic phenomenon: pathogenic variants in any gene concentrate at "
                "positions the model scores higher than positions where benign variants land "
                "(likely conserved residues, CpG codons, etc.). The TSG framing is misleading.")
    if rand <= 0.60 and onco <= 0.60 and tsg >= 0.75:
        return ("TSG-SPECIFIC",
                "Only TSGs show the high win-rate. The effect is TSG-specific and consistent with "
                "the biological hypothesis that APOBEC-targeted positions are stress-vulnerable "
                "in tumor suppressors. The paper claim is supported.")
    return ("PARTIAL",
            "TSGs show the strongest signal but oncogenes and/or random genes also show "
            "non-trivial enrichment. Part of the effect is generic (model picks 'interesting' "
            "positions in any gene), part may be TSG-enriched. The paper claim should be "
            "softened to 'TSGs show the strongest signal' rather than 'TSGs are uniquely affected'.")

verdict, interpretation = classify(tsg_frac, onco_frac, rand_frac)
md.append(f"### Verdict: **{verdict}**")
md.append("")
md.append(interpretation)
md.append("")

# Implications
md.append("## Implications for the paper claim")
md.append("")
if verdict == "TSG-SPECIFIC":
    md.append(
        "- The claim 'pathogenic variants in TSGs sit at higher-editability positions than benign "
        "variants' stands. APOBEC editability is enriched at stress-vulnerable positions in tumor "
        "suppressors, consistent with the model's biological hypothesis.")
elif verdict == "GENERAL":
    md.append(
        "- The original framing OVERSTATES the TSG-specificity. The same effect appears in "
        "oncogenes and random non-cancer genes, so it is not unique to TSG biology. Reframe as: "
        "'the model assigns higher editability scores to positions where pathogenic ClinVar "
        "variants are observed, in any gene' --- a model-property statement, not a TSG-biology "
        "statement.")
    md.append(
        "- Plausible mechanism: pathogenic ClinVar C>T variants concentrate at evolutionarily "
        "conserved codons (where mutations are deleterious enough to be flagged as pathogenic), "
        "and these conserved positions also tend to have specific sequence/structure features "
        "that the editability model upweights (e.g., CpG context). The TSG-specific claim should "
        "be retracted or significantly softened.")
else:  # PARTIAL
    md.append(
        "- The TSG-specific framing is partially supported. TSGs do show the strongest signal, "
        "but a meaningful fraction of the effect is generic (also present in oncogenes / random "
        "genes). Reframe the paper claim as 'TSGs show the largest pathogenic-vs-benign separation "
        "on editability scores' rather than 'this effect is TSG-specific'.")
    md.append(
        "- Recommended: report the TSG / Oncogene / Random comparison in the paper as a control, "
        "and discuss the partial-generic-effect interpretation honestly.")
md.append("")

md.append("## Files produced")
md.append("")
md.append("- `experiments/apobec3a/outputs/clinvar_v4_full/tsg_specificity_control.csv` (cell summary)")
md.append("- `experiments/apobec3a/outputs/clinvar_v4_full/tsg_specificity_per_gene.csv` (per-gene detail, stratum=all)")
md.append("- `experiments/apobec3a/outputs/clinvar_v4_full/TSG_SPECIFICITY_RESULTS.md` (this report)")
md.append("")

(OUT_DIR / "TSG_SPECIFICITY_RESULTS.md").write_text("\n".join(md))
print(f"\nWrote {OUT_DIR / 'TSG_SPECIFICITY_RESULTS.md'}")
print("\nDONE")
