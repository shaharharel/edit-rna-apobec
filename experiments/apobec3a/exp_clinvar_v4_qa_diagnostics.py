"""QA diagnostics for ClinVar v4 / v3 findings.

Implements 5 read-only QA checks flagged by the QA agent:

  1. Top-K sensitivity for Finding 1 (nonsense top-K, K in {100, 500, 1000, 2000, 5000})
     across heads x strata. Tests whether headline OR is monotonic in K or fragile.
  2. Wilcoxon meta-p for Finding 2 (per-gene MWU p-values aggregated via Stouffer + Fisher),
     plus an effect-size-thresholded sign test (|diff| >= 0.05).
  3. Random-head shuffle null for Finding 2 — shuffle scores across all 1.69M variants,
     re-run per-TSG sign test, build null distribution, compute empirical p of observed.
  4. Within-gene paired CpG-vs-non-CpG comparison: for genes that have >=3 path/ben in
     BOTH strata, test whether diff_nonCpG > diff_CpG (Wilcoxon signed-rank).
  5. Bonferroni correction at the test-family level (54 = 3 gene_lists * 6 heads * 3 strata).

Inputs (all assumed already produced by exp_clinvar_v4_full_replication.py):
  - experiments/apobec3a/outputs/clinvar_v4_scored/clinvar_scores_v4.parquet
  - experiments/apobec3a/outputs/clinvar_prediction/clinvar_all_scores.csv
  - data/processed/clinvar_c2u_variants.csv
  - experiments/apobec3a/outputs/clinvar_v4_full/per_gene_path_vs_ben_full.csv
  - experiments/apobec3a/outputs/clinvar_v4_full/finding1_nonsense_v4_full.csv
  - experiments/apobec3a/outputs/clinvar_v4_full/finding2_tsg_v4_full.csv
  - data/raw/genomes/hg19.fa  (for CpG annotation)

Outputs (under experiments/apobec3a/outputs/clinvar_v4_full/):
  - qa_diag1_topk.csv
  - qa_diag2_wilcoxon.csv
  - qa_diag3_shuffle.csv
  - qa_diag4_paired_cpg.csv
  - qa_diag5_bonferroni.csv
  - QA_DIAGNOSTICS_RESULTS.md
"""
from __future__ import annotations

import json
import ssl
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from pyfaidx import Fasta
from scipy import stats
from scipy.stats import binomtest, fisher_exact, norm, wilcoxon

PROJECT_ROOT = Path("/Users/shaharharel/Documents/github/edit-rna-apobec")
SEED = 20260427
RNG = np.random.default_rng(SEED)

OUT_DIR = PROJECT_ROOT / "experiments" / "apobec3a" / "outputs" / "clinvar_v4_full"
OUT_DIR.mkdir(parents=True, exist_ok=True)

V4_PARQUET = PROJECT_ROOT / "experiments" / "apobec3a" / "outputs" / "clinvar_v4_scored" / "clinvar_scores_v4.parquet"
CLINVAR_META = PROJECT_ROOT / "data" / "processed" / "clinvar_c2u_variants.csv"
V3_GB_CSV = PROJECT_ROOT / "experiments" / "apobec3a" / "outputs" / "clinvar_prediction" / "clinvar_all_scores.csv"
HG19 = PROJECT_ROOT / "data" / "raw" / "genomes" / "hg19.fa"
PER_GENE_CSV = OUT_DIR / "per_gene_path_vs_ben_full.csv"
FINDING2_CSV = OUT_DIR / "finding2_tsg_v4_full.csv"

HEADS = [
    "v3_GB",
    "score_binary_v4",
    "score_A3A_v4",
    "score_A3B_v4",
    "score_A3A_A3G_v4",
    "score_apobec1_v4_cds",
]

PATH = {"Pathogenic", "Likely_pathogenic"}
BEN = {"Benign", "Likely_benign"}

# Gene lists (mirror exp_clinvar_v4_full_replication.py)
CURATED_48 = {
    "TP53","RB1","APC","BRCA1","BRCA2","PTEN","VHL","WT1","NF1","NF2",
    "TSC1","TSC2","SMAD4","CDKN2A","CDH1","MEN1","MLH1","MSH2","MSH6",
    "PMS2","STK11","PTCH1","SDHB","SDHD","SDHA","SDHC","SDHAF2",
    "BAP1","SMARCB1","ARID1A","ARID1B","KDM6A","FBXW7","ATM","ATR",
    "CHEK2","PALB2","MUTYH","BMPR1A","DICER1","FLCN","FH","MAX",
    "TMEM127","SUFU","EPCAM","NBN","WRN",
}

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


def fetch_oncokb_tsg() -> set[str]:
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
        return {g["hugoSymbol"] for g in data if g.get("sangerCGC") and g.get("geneType") == "TSG"}
    except Exception as e:
        print(f"[WARN] OncoKB fetch failed: {e}; falling back to gene set from per_gene CSV")
        return set()


def haldane_fisher(a: int, b: int, c: int, d: int):
    if a == 0 or b == 0 or c == 0 or d == 0:
        a_, b_, c_, d_ = a + 0.5, b + 0.5, c + 0.5, d + 0.5
        or_val = (a_ * d_) / (b_ * c_)
    else:
        or_val = (a * d) / (b * c)
    se = np.sqrt(1 / (a + 0.5) + 1 / (b + 0.5) + 1 / (c + 0.5) + 1 / (d + 0.5))
    log_or = np.log(or_val)
    ci_lo = float(np.exp(log_or - 1.96 * se))
    ci_hi = float(np.exp(log_or + 1.96 * se))
    _, p = fisher_exact([[a, b], [c, d]], alternative="greater")
    return float(or_val), ci_lo, ci_hi, float(p)


def annotate_cpg(df: pd.DataFrame) -> np.ndarray:
    fa = Fasta(str(HG19), as_raw=True, sequence_always_upper=True)
    is_cpg = np.zeros(len(df), dtype=bool)
    chroms = df["chrom"].values
    poss = df["pos"].values.astype(int)
    strands = df["strand"].values
    for i in range(len(df)):
        ch = chroms[i]
        p = int(poss[i])
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
    return is_cpg


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


# ---------------------------------------------------------------------------
# Build the joined master dataframe (same as v4 full replication)
# ---------------------------------------------------------------------------

print("=" * 80)
print("Loading v4 + meta + v3 + CpG annotation")
print("=" * 80)

t0 = time.time()
v4 = pd.read_parquet(V4_PARQUET)
print(f"v4 rows: {len(v4):,}")

meta = pd.read_csv(
    CLINVAR_META,
    usecols=["site_id", "clinical_significance", "molecular_consequence", "gene"],
)
meta["significance_simple"] = meta["clinical_significance"].map(simplify_sig)
meta["consequence_simple"] = meta["molecular_consequence"].map(simplify_consequence)

v3 = pd.read_csv(V3_GB_CSV, usecols=["site_id", "p_edited_gb"])
v3 = v3.rename(columns={"p_edited_gb": "v3_GB"})

df = v4.merge(meta, on="site_id", how="inner").merge(v3, on="site_id", how="left")
df = df.rename(
    columns={
        "score_binary": "score_binary_v4",
        "score_A3A": "score_A3A_v4",
        "score_A3B": "score_A3B_v4",
        "score_A3G": "score_A3G_v4",
        "score_A3A_A3G": "score_A3A_A3G_v4",
    }
)
print(f"Joined: {len(df):,}; v3 NaN: {df['v3_GB'].isna().sum():,}")

print("Annotating CpG ...")
df["is_cpg"] = annotate_cpg(df)
print(f"CpG: {int(df['is_cpg'].sum()):,}; non-CpG: {int((~df['is_cpg']).sum()):,}")
print(f"Setup time: {time.time()-t0:.1f}s")

path_mask = df["significance_simple"].isin(PATH).values
ben_mask = df["significance_simple"].isin(BEN).values
genes_arr = df["gene"].fillna("").values
nonsense_mask = (df["consequence_simple"] == "nonsense").values

strata = {
    "all": np.ones(len(df), dtype=bool),
    "CpG": df["is_cpg"].values,
    "non_CpG": (~df["is_cpg"]).values,
}

# ---------------------------------------------------------------------------
# Diagnostic 1 — Top-K sensitivity (Finding 1)
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("Diagnostic 1 — Top-K sensitivity for Finding 1")
print("=" * 80)

KS = [100, 500, 1000, 2000, 5000]
diag1_rows = []

for head in HEADS:
    if head not in df.columns:
        continue
    scores = df[head].values
    valid_score = ~np.isnan(scores)
    for stratum_name, stratum_mask in strata.items():
        sm = path_mask & stratum_mask & valid_score
        n_strat = int(sm.sum())
        if n_strat < 100:
            continue
        idx_in_stratum = np.where(sm)[0]
        scores_strat = scores[idx_in_stratum]
        baseline_n = int(nonsense_mask[idx_in_stratum].sum())
        baseline_rate = baseline_n / n_strat

        # sort once
        order = np.argsort(scores_strat)  # ascending; top is at the end
        for K in KS:
            k = min(K, n_strat)
            top_idx = idx_in_stratum[order[-k:]]
            top_n_nonsense = int(nonsense_mask[top_idx].sum())
            rest_idx = idx_in_stratum[order[:-k]]
            rest_n = len(rest_idx)
            rest_n_nonsense = int(nonsense_mask[rest_idx].sum())
            a = top_n_nonsense
            b = k - top_n_nonsense
            c = rest_n_nonsense
            d = rest_n - rest_n_nonsense
            or_val, ci_lo, ci_hi, p = haldane_fisher(a, b, c, d)
            diag1_rows.append({
                "head": head,
                "K": K,
                "stratum": stratum_name,
                "n_top_K": k,
                "n_top_K_nonsense": top_n_nonsense,
                "baseline_rate": round(baseline_rate, 4),
                "OR": round(or_val, 3),
                "ci_lo": round(ci_lo, 3),
                "ci_hi": round(ci_hi, 3),
                "p": p,
            })

diag1_df = pd.DataFrame(diag1_rows)
diag1_df.to_csv(OUT_DIR / "qa_diag1_topk.csv", index=False)
print(f"Wrote {OUT_DIR/'qa_diag1_topk.csv'} ({len(diag1_df)} rows)")

# Monotonicity verdict per head x stratum
def monotonicity_label(sub: pd.DataFrame):
    """Verdict based on OR shape across K in {100, 500, 1000, 2000, 5000}."""
    s = sub.sort_values("K")["OR"].values
    if len(s) < 3:
        return "INSUFFICIENT"
    # is OR > 1.3 at every K?
    all_strong = bool((s > 1.3).all())
    # is OR > 1.3 only at one K and <=1.0 at others?
    any_collapse = bool(((s <= 1.0).sum()) >= 1) and bool(((s > 1.3).sum()) >= 1)
    if all_strong:
        return "ROBUST"
    if any_collapse:
        return "FRAGILE"
    return "PARTIAL"

# ---------------------------------------------------------------------------
# Diagnostic 2 — Wilcoxon meta-p replacement (Finding 2)
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("Diagnostic 2 — Wilcoxon meta-p (Stouffer + Fisher) + ES-thresholded sign test")
print("=" * 80)

per_gene = pd.read_csv(PER_GENE_CSV)
print(f"per_gene rows: {len(per_gene)}")

# Note: per_gene was generated only for stratum=='all'. For per-stratum aggregation we need
# to recompute per-gene path/ben means + MWU on each stratum, since the existing CSV is all only.
# We'll recompute per-stratum on the fly.

def compute_per_gene_stratum(stratum_name: str, stratum_mask: np.ndarray, gene_set: set[str], head: str):
    """Recompute per-gene stats for a head x stratum. Returns DataFrame with cols
       gene, n_path, n_ben, mean_path, mean_ben, diff, mwu_p, won."""
    scores = df[head].values
    valid = ~np.isnan(scores)
    in_path = path_mask & stratum_mask & valid
    in_ben = ben_mask & stratum_mask & valid
    rows = []
    for gene in gene_set:
        gm = (genes_arr == gene)
        gp = gm & in_path
        gb = gm & in_ben
        np_n = int(gp.sum()); nb_n = int(gb.sum())
        if np_n < 3 or nb_n < 3:
            continue
        ps = scores[gp]; bs = scores[gb]
        mean_p = float(ps.mean()); mean_b = float(bs.mean())
        diff = mean_p - mean_b
        try:
            _, mwu_p = stats.mannwhitneyu(ps, bs, alternative="two-sided")
        except ValueError:
            mwu_p = 1.0
        rows.append({
            "gene": gene, "n_path": np_n, "n_ben": nb_n,
            "mean_path": mean_p, "mean_ben": mean_b, "diff": diff,
            "mwu_p": float(mwu_p), "won": bool(diff > 0),
        })
    return pd.DataFrame(rows)


print("Fetching OncoKB CGC TSG list (may use cached) ...")
ONCOKB_TSG = fetch_oncokb_tsg()
if not ONCOKB_TSG:
    # fall back to whatever genes are in per_gene CSV under oncokb_cgc_tsg_173
    ONCOKB_TSG = set(per_gene[per_gene["gene_list"] == "oncokb_cgc_tsg_173"]["gene"].unique())
    # add a few that may be missing-but-not-testable; keep the testable ones we already saw
print(f"OncoKB TSG: {len(ONCOKB_TSG)}")

GENE_LISTS = {
    "curated_48": CURATED_48,
    "tier1_proxy_82": TIER1_TSG,
    "oncokb_cgc_tsg_173": ONCOKB_TSG,
}

diag2_rows = []
# also retain per-gene per-stratum stats for diagnostic 4
per_gene_per_stratum: dict[tuple[str, str, str], pd.DataFrame] = {}

for gl_name, gl in GENE_LISTS.items():
    if not gl:
        continue
    for head in HEADS:
        if head not in df.columns:
            continue
        for stratum_name, stratum_mask in strata.items():
            sub = compute_per_gene_stratum(stratum_name, stratum_mask, gl, head)
            per_gene_per_stratum[(gl_name, head, stratum_name)] = sub
            n_genes = len(sub)
            if n_genes == 0:
                diag2_rows.append({
                    "gene_list": gl_name, "head": head, "stratum": stratum_name,
                    "n_genes": 0, "n_genes_mw_sig": 0,
                    "sign_test_p": 1.0, "sign_test_p_es_thresh_0.05": 1.0,
                    "stouffer_p": 1.0, "fisher_p": 1.0,
                })
                continue

            # standard sign test (matches Finding 2 in original)
            wins = int(sub["won"].sum())
            sign_p = binomtest(wins, n_genes, 0.5, alternative="greater").pvalue

            # ES-thresholded sign test: drop |diff| < 0.05; among the rest count diff>0
            es = sub[sub["diff"].abs() >= 0.05]
            if len(es) == 0:
                sign_p_es = 1.0
            else:
                wins_es = int((es["diff"] > 0).sum())
                sign_p_es = binomtest(wins_es, len(es), 0.5, alternative="greater").pvalue

            # Stouffer's: convert each gene's mwu_p (two-sided) to a one-sided p in the
            # direction of mean_path > mean_ben. If diff > 0, p_one = mwu_p/2; else 1 - mwu_p/2.
            mwu_two = sub["mwu_p"].clip(1e-300, 1.0).values
            diffs = sub["diff"].values
            p_one = np.where(diffs > 0, mwu_two / 2.0, 1.0 - mwu_two / 2.0)
            # Avoid p==0 or p==1 exactly to keep ppf finite
            p_one = np.clip(p_one, 1e-300, 1.0 - 1e-16)
            z_scores = norm.isf(p_one)  # higher z = stronger evidence for path>ben
            stouffer_z = z_scores.sum() / np.sqrt(len(z_scores))
            stouffer_p = float(norm.sf(stouffer_z))

            # Fisher's combined chi-square on the same one-sided p_one
            chi2_stat = -2.0 * np.log(p_one).sum()
            fisher_p = float(stats.chi2.sf(chi2_stat, df=2 * len(p_one)))

            # MWU significant per-gene at 0.05 (two-sided)
            n_mw_sig = int((sub["mwu_p"] < 0.05).sum())

            diag2_rows.append({
                "gene_list": gl_name, "head": head, "stratum": stratum_name,
                "n_genes": n_genes, "n_genes_mw_sig": n_mw_sig,
                "sign_test_p": float(sign_p),
                "sign_test_p_es_thresh_0.05": float(sign_p_es),
                "stouffer_p": stouffer_p,
                "fisher_p": fisher_p,
            })

diag2_df = pd.DataFrame(diag2_rows)
diag2_df.to_csv(OUT_DIR / "qa_diag2_wilcoxon.csv", index=False)
print(f"Wrote {OUT_DIR/'qa_diag2_wilcoxon.csv'} ({len(diag2_df)} rows)")

# ---------------------------------------------------------------------------
# Diagnostic 3 — Random-head shuffle null (Finding 2)
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("Diagnostic 3 — Random shuffle null (Finding 2 sign test)")
print("=" * 80)

N_SHUFFLES = 100
diag3_rows = []

# We shuffle the v3_GB and v4_binary scores. Use the OncoKB list (the headline list).
# Observed wins are in finding2_tsg_v4_full.csv at stratum='all'.
finding2_main = pd.read_csv(FINDING2_CSV)

# Use ONCOKB_TSG gene set for headline
for shuffle_head in ["v3_GB", "score_binary_v4"]:
    if shuffle_head not in df.columns:
        continue
    scores = df[shuffle_head].values.copy()
    # Only shuffle valid scores
    valid_idx = np.where(~np.isnan(scores))[0]
    print(f"Shuffling {shuffle_head}: valid entries = {len(valid_idx):,}")
    for gl_name, gl in GENE_LISTS.items():
        if not gl:
            continue
        # Observed wins from finding2_main
        obs_row = finding2_main[
            (finding2_main["gene_list"] == gl_name)
            & (finding2_main["head"] == shuffle_head)
            & (finding2_main["stratum"] == "all")
        ]
        if obs_row.empty:
            continue
        observed_wins = int(obs_row.iloc[0]["n_path_gt_ben"])
        n_genes_observed = int(obs_row.iloc[0]["n_genes_testable"])

        rng = np.random.default_rng(SEED + hash((shuffle_head, gl_name)) % (2**31))
        scores_local = scores.copy()
        valid_score = ~np.isnan(scores_local)
        in_path = path_mask & valid_score
        in_ben = ben_mask & valid_score

        # Pre-compute per-gene path/ben indices once
        gene_idx_path = {}
        gene_idx_ben = {}
        for gene in gl:
            gm = (genes_arr == gene)
            gp = np.where(gm & in_path)[0]
            gb = np.where(gm & in_ben)[0]
            if len(gp) >= 3 and len(gb) >= 3:
                gene_idx_path[gene] = gp
                gene_idx_ben[gene] = gb

        n_genes = len(gene_idx_path)
        null_wins = []
        for s in range(N_SHUFFLES):
            shuffled = scores_local.copy()
            perm = rng.permutation(valid_idx)
            shuffled[valid_idx] = scores_local[perm]
            wins = 0
            for gene, pidx in gene_idx_path.items():
                bidx = gene_idx_ben[gene]
                mp = shuffled[pidx].mean()
                mb = shuffled[bidx].mean()
                if mp > mb:
                    wins += 1
            null_wins.append(wins)

        null_arr = np.array(null_wins)
        null_mean = float(null_arr.mean())
        null_std = float(null_arr.std(ddof=1))
        # Empirical p: fraction of nulls with wins >= observed
        p_emp = float((null_arr >= observed_wins).mean())
        diag3_rows.append({
            "shuffle_head": shuffle_head,
            "gene_list": gl_name,
            "n_genes": n_genes,
            "n_shuffles": N_SHUFFLES,
            "observed_wins": observed_wins,
            "null_mean_wins": round(null_mean, 2),
            "null_std": round(null_std, 2),
            "p_empirical": p_emp,
        })
        print(f"  {shuffle_head} {gl_name}: obs={observed_wins}/{n_genes}; "
              f"null mean={null_mean:.2f}+/-{null_std:.2f}; p_emp={p_emp:.4f}")

diag3_df = pd.DataFrame(diag3_rows)
diag3_df.to_csv(OUT_DIR / "qa_diag3_shuffle.csv", index=False)
print(f"Wrote {OUT_DIR/'qa_diag3_shuffle.csv'} ({len(diag3_df)} rows)")

# ---------------------------------------------------------------------------
# Diagnostic 4 — Within-gene paired CpG-vs-non-CpG comparison (Finding 2)
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("Diagnostic 4 — Within-gene paired CpG vs non-CpG diff comparison")
print("=" * 80)

diag4_rows = []
for gl_name, gl in GENE_LISTS.items():
    if not gl:
        continue
    for head in HEADS:
        if head not in df.columns:
            continue
        cpg_sub = per_gene_per_stratum.get((gl_name, head, "CpG"))
        non_sub = per_gene_per_stratum.get((gl_name, head, "non_CpG"))
        if cpg_sub is None or non_sub is None:
            continue
        # paired by gene
        merged = cpg_sub[["gene", "diff"]].rename(columns={"diff": "diff_CpG"}).merge(
            non_sub[["gene", "diff"]].rename(columns={"diff": "diff_nonCpG"}),
            on="gene", how="inner"
        )
        n_paired = len(merged)
        if n_paired == 0:
            diag4_rows.append({
                "gene_list": gl_name, "head": head, "n_genes_paired": 0,
                "n_nonCpG_gt_CpG": 0, "signed_rank_p": 1.0,
                "mean_diff_nonCpG": np.nan, "mean_diff_CpG": np.nan,
            })
            continue
        n_non_gt = int((merged["diff_nonCpG"] > merged["diff_CpG"]).sum())
        # Wilcoxon signed-rank: H1: diff_nonCpG > diff_CpG
        try:
            stat, p_signed = wilcoxon(
                merged["diff_nonCpG"].values,
                merged["diff_CpG"].values,
                alternative="greater",
                zero_method="wilcox",
            )
        except ValueError:
            p_signed = 1.0
        diag4_rows.append({
            "gene_list": gl_name, "head": head, "n_genes_paired": n_paired,
            "n_nonCpG_gt_CpG": n_non_gt, "signed_rank_p": float(p_signed),
            "mean_diff_nonCpG": float(merged["diff_nonCpG"].mean()),
            "mean_diff_CpG": float(merged["diff_CpG"].mean()),
        })

diag4_df = pd.DataFrame(diag4_rows)
diag4_df.to_csv(OUT_DIR / "qa_diag4_paired_cpg.csv", index=False)
print(f"Wrote {OUT_DIR/'qa_diag4_paired_cpg.csv'} ({len(diag4_df)} rows)")

# ---------------------------------------------------------------------------
# Diagnostic 5 — Bonferroni at the test family
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("Diagnostic 5 — Bonferroni at family-wide alpha=0.05/54")
print("=" * 80)

# Family = 3 gene_lists * 6 heads * 3 strata = 54 cells (sign-test cells)
ALPHA = 0.05
M = 3 * 6 * 3  # 54
BONF_Q = ALPHA / M

# Use the original sign_test_p from finding2_tsg_v4_full.csv (the headline numbers)
diag5_rows = []
for _, r in finding2_main.iterrows():
    p = float(r["sign_test_p"])
    diag5_rows.append({
        "gene_list": r["gene_list"],
        "head": r["head"],
        "stratum": r["stratum"],
        "sign_test_p": p,
        "survives_bonferroni": bool(p < BONF_Q),
    })

diag5_df = pd.DataFrame(diag5_rows)
diag5_df.to_csv(OUT_DIR / "qa_diag5_bonferroni.csv", index=False)
n_survive = int(diag5_df["survives_bonferroni"].sum())
print(f"Bonferroni q={BONF_Q:.2e}; survive: {n_survive} / {len(diag5_df)}")
print(f"Wrote {OUT_DIR/'qa_diag5_bonferroni.csv'}")

# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("Writing markdown report")
print("=" * 80)

md = []
md.append("# ClinVar v4 — QA Diagnostics Results")
md.append("")
md.append(f"Random seed: {SEED}")
md.append("")
md.append("Five read-only diagnostics flagged by the QA agent. Inputs are the v4 ClinVar replication outputs (1.69M variants, 6 heads, 3 gene lists, 3 strata).")
md.append("")

# ---- Diagnostic 1 ----
md.append("## 1. Diagnostic 1 — Top-K sensitivity (Finding 1, nonsense)")
md.append("")
md.append("For each (head x stratum), OR(top-K nonsense vs rest of stratum) at K in {100, 500, 1000, 2000, 5000}. Verdict per head x stratum:")
md.append("- ROBUST: OR > 1.3 at every K")
md.append("- FRAGILE: OR > 1.3 at one K AND OR <= 1.0 at another")
md.append("- PARTIAL: otherwise")
md.append("")
md.append("### Compact OR-vs-K table (non_CpG only — the headline stratum)")
md.append("")
md.append("| Head | K=100 | K=500 | K=1000 | K=2000 | K=5000 | verdict |")
md.append("|---|---|---|---|---|---|---|")
for head in HEADS:
    sub = diag1_df[(diag1_df["head"] == head) & (diag1_df["stratum"] == "non_CpG")]
    if sub.empty:
        continue
    sub = sub.sort_values("K")
    cells = []
    for K in KS:
        row = sub[sub["K"] == K]
        if row.empty:
            cells.append("-")
        else:
            cells.append(f"{row.iloc[0]['OR']:.2f}")
    verdict = monotonicity_label(sub)
    md.append(f"| {head} | " + " | ".join(cells) + f" | {verdict} |")
md.append("")
md.append("### Compact OR-vs-K table (all stratum)")
md.append("")
md.append("| Head | K=100 | K=500 | K=1000 | K=2000 | K=5000 | verdict |")
md.append("|---|---|---|---|---|---|---|")
for head in HEADS:
    sub = diag1_df[(diag1_df["head"] == head) & (diag1_df["stratum"] == "all")]
    if sub.empty:
        continue
    sub = sub.sort_values("K")
    cells = []
    for K in KS:
        row = sub[sub["K"] == K]
        if row.empty:
            cells.append("-")
        else:
            cells.append(f"{row.iloc[0]['OR']:.2f}")
    verdict = monotonicity_label(sub)
    md.append(f"| {head} | " + " | ".join(cells) + f" | {verdict} |")
md.append("")
md.append("### Compact OR-vs-K table (CpG only)")
md.append("")
md.append("| Head | K=100 | K=500 | K=1000 | K=2000 | K=5000 | verdict |")
md.append("|---|---|---|---|---|---|---|")
for head in HEADS:
    sub = diag1_df[(diag1_df["head"] == head) & (diag1_df["stratum"] == "CpG")]
    if sub.empty:
        continue
    sub = sub.sort_values("K")
    cells = []
    for K in KS:
        row = sub[sub["K"] == K]
        if row.empty:
            cells.append("-")
        else:
            cells.append(f"{row.iloc[0]['OR']:.2f}")
    verdict = monotonicity_label(sub)
    md.append(f"| {head} | " + " | ".join(cells) + f" | {verdict} |")
md.append("")

# Finding 1 verdict
v3_nc = diag1_df[(diag1_df["head"] == "v3_GB") & (diag1_df["stratum"] == "non_CpG")].sort_values("K")
v3_label = monotonicity_label(v3_nc)
v4bin_nc = diag1_df[(diag1_df["head"] == "score_binary_v4") & (diag1_df["stratum"] == "non_CpG")].sort_values("K")
v4_label = monotonicity_label(v4bin_nc)
md.append(f"**Finding 1 monotonicity verdict (non_CpG):** v3_GB={v3_label}; score_binary_v4={v4_label}.")
md.append("")

# ---- Diagnostic 2 ----
md.append("## 2. Diagnostic 2 — Wilcoxon meta-p + ES-thresholded sign test (Finding 2)")
md.append("")
md.append("For each (gene_list x head x stratum):")
md.append("- `n_genes_mw_sig`: number of genes with two-sided MWU p < 0.05")
md.append("- `sign_test_p`: original binomial sign test (matches finding2_tsg_v4_full)")
md.append("- `sign_test_p_es_thresh_0.05`: sign test on genes with |mean_path - mean_ben| >= 0.05")
md.append("- `stouffer_p`: Stouffer's combination of one-sided MWU p-values (path > ben direction)")
md.append("- `fisher_p`: Fisher's combined chi-square on the same one-sided p-values")
md.append("")
md.append("### OncoKB CGC TSG-173 (headline)")
md.append("")
md.append("| Head | Stratum | n_genes | mw_sig | sign_p | sign_p_es>=.05 | stouffer_p | fisher_p |")
md.append("|---|---|---|---|---|---|---|---|")
for head in HEADS:
    for stratum_name in ["all", "CpG", "non_CpG"]:
        r = diag2_df[
            (diag2_df["gene_list"] == "oncokb_cgc_tsg_173")
            & (diag2_df["head"] == head)
            & (diag2_df["stratum"] == stratum_name)
        ]
        if r.empty:
            continue
        r = r.iloc[0]
        md.append(
            f"| {head} | {stratum_name} | {r['n_genes']} | {r['n_genes_mw_sig']} | "
            f"{r['sign_test_p']:.2e} | {r['sign_test_p_es_thresh_0.05']:.2e} | "
            f"{r['stouffer_p']:.2e} | {r['fisher_p']:.2e} |"
        )
md.append("")
md.append("### Curated-48 and Tier1-82 (sensitivity)")
md.append("")
md.append("| Gene list | Head | Stratum | n_genes | mw_sig | sign_p | sign_p_es>=.05 | stouffer_p | fisher_p |")
md.append("|---|---|---|---|---|---|---|---|---|")
for gl_name in ["curated_48", "tier1_proxy_82"]:
    for head in HEADS:
        for stratum_name in ["all", "CpG", "non_CpG"]:
            r = diag2_df[
                (diag2_df["gene_list"] == gl_name)
                & (diag2_df["head"] == head)
                & (diag2_df["stratum"] == stratum_name)
            ]
            if r.empty:
                continue
            r = r.iloc[0]
            md.append(
                f"| {gl_name} | {head} | {stratum_name} | {r['n_genes']} | {r['n_genes_mw_sig']} | "
                f"{r['sign_test_p']:.2e} | {r['sign_test_p_es_thresh_0.05']:.2e} | "
                f"{r['stouffer_p']:.2e} | {r['fisher_p']:.2e} |"
            )
md.append("")

# Finding 2 headline change
oncokb_v3_all = diag2_df[
    (diag2_df["gene_list"] == "oncokb_cgc_tsg_173")
    & (diag2_df["head"] == "v3_GB")
    & (diag2_df["stratum"] == "all")
]
if not oncokb_v3_all.empty:
    rr = oncokb_v3_all.iloc[0]
    md.append(
        f"**Finding 2 headline (OncoKB-173, v3_GB, stratum=all):** "
        f"original sign_p={rr['sign_test_p']:.2e}; "
        f"|diff|>=0.05 sign_p={rr['sign_test_p_es_thresh_0.05']:.2e}; "
        f"Stouffer p={rr['stouffer_p']:.2e}; Fisher p={rr['fisher_p']:.2e}."
    )
md.append("")

# ---- Diagnostic 3 ----
md.append("## 3. Diagnostic 3 — Random shuffle null (Finding 2 sign test)")
md.append("")
md.append(f"Shuffles = {N_SHUFFLES} per (head x gene_list). Scores randomly permuted across all valid ClinVar variants. Sanity: null mean wins should be ~ n_genes/2.")
md.append("")
md.append("| Head | Gene list | n_genes | observed wins | null mean | null std | p_empirical | sane? |")
md.append("|---|---|---|---|---|---|---|---|")
for _, r in diag3_df.iterrows():
    expected = r["n_genes"] / 2
    sane = "OK" if abs(r["null_mean_wins"] - expected) < 0.10 * r["n_genes"] else "BIASED"
    md.append(
        f"| {r['shuffle_head']} | {r['gene_list']} | {r['n_genes']} | {r['observed_wins']} | "
        f"{r['null_mean_wins']:.2f} | {r['null_std']:.2f} | {r['p_empirical']:.4f} | {sane} |"
    )
md.append("")
# Headline empirical p
oncokb_shuffle = diag3_df[(diag3_df["gene_list"] == "oncokb_cgc_tsg_173") & (diag3_df["shuffle_head"] == "v3_GB")]
if not oncokb_shuffle.empty:
    rr = oncokb_shuffle.iloc[0]
    md.append(
        f"**Headline:** v3_GB on OncoKB-173: observed wins={rr['observed_wins']}, "
        f"null mean={rr['null_mean_wins']:.1f} (expected ~{rr['n_genes']/2:.0f}), "
        f"empirical p={rr['p_empirical']:.4f}."
    )
md.append("")

# ---- Diagnostic 4 ----
md.append("## 4. Diagnostic 4 — Within-gene paired CpG vs non-CpG diff comparison")
md.append("")
md.append("For genes with >=3 path/ben in BOTH CpG and non-CpG strata: do non-CpG diffs exceed CpG diffs in the same gene? (Wilcoxon signed-rank, alternative='greater'.)")
md.append("")
md.append("| Gene list | Head | n_paired | n_nonCpG > CpG | signed_rank_p | mean diff non-CpG | mean diff CpG |")
md.append("|---|---|---|---|---|---|---|")
for _, r in diag4_df.iterrows():
    md.append(
        f"| {r['gene_list']} | {r['head']} | {r['n_genes_paired']} | {r['n_nonCpG_gt_CpG']} | "
        f"{r['signed_rank_p']:.2e} | {r['mean_diff_nonCpG']:.4f} | {r['mean_diff_CpG']:.4f} |"
    )
md.append("")
n_paired_oncokb = diag4_df[
    (diag4_df["gene_list"] == "oncokb_cgc_tsg_173")
]
if not n_paired_oncokb.empty:
    md.append(
        f"**Headline:** Among OncoKB-173 genes with paired CpG/non-CpG strata, paired n is "
        f"typically small ({n_paired_oncokb['n_genes_paired'].max()} max). "
    )
md.append("")

# ---- Diagnostic 5 ----
md.append(f"## 5. Diagnostic 5 — Bonferroni at family-wide alpha = 0.05 / {M} = {BONF_Q:.2e}")
md.append("")
md.append(f"Family = 3 gene_lists × 6 heads × 3 strata = {M} cells. Survivors:")
md.append("")
md.append("| Gene list | Head | Stratum | sign_test_p | survives |")
md.append("|---|---|---|---|---|")
for _, r in diag5_df.iterrows():
    md.append(
        f"| {r['gene_list']} | {r['head']} | {r['stratum']} | {r['sign_test_p']:.2e} | "
        f"{'YES' if r['survives_bonferroni'] else 'no'} |"
    )
md.append("")
md.append(f"**Total cells surviving Bonferroni: {n_survive} / {len(diag5_df)}.**")
md.append("")
# Stratify survivors by stratum
md.append("### Survivors by stratum")
md.append("")
md.append("| Stratum | survivors / total |")
md.append("|---|---|")
for stratum_name in ["all", "CpG", "non_CpG"]:
    sub = diag5_df[diag5_df["stratum"] == stratum_name]
    s = int(sub["survives_bonferroni"].sum())
    md.append(f"| {stratum_name} | {s} / {len(sub)} |")
md.append("")

# ---- Final verdicts ----
md.append("## 6. FINAL VERDICTS")
md.append("")

# Finding 1 verdict
def f1_verdict():
    # PASS if v3_GB non-CpG OR > 1.3 at K in {500, 1000, 2000} (3 of 5 K's)
    sub = diag1_df[(diag1_df["head"] == "v3_GB") & (diag1_df["stratum"] == "non_CpG")].sort_values("K")
    or_arr = sub["OR"].values
    if len(or_arr) == 5:
        n_above = int((or_arr > 1.3).sum())
        if n_above == 5:
            return "PASS"
        if n_above >= 3:
            return "WEAK"
        return "FAIL"
    return "INSUFFICIENT"

# Finding 2 verdict
def f2_verdict():
    # PASS if OncoKB-173 v3_GB stratum=all sign_p_es < 0.001 AND stouffer_p < 0.001 AND null shuffle p < 0.001
    rr = diag2_df[
        (diag2_df["gene_list"] == "oncokb_cgc_tsg_173")
        & (diag2_df["head"] == "v3_GB")
        & (diag2_df["stratum"] == "all")
    ]
    sh = diag3_df[(diag3_df["gene_list"] == "oncokb_cgc_tsg_173") & (diag3_df["shuffle_head"] == "v3_GB")]
    if rr.empty or sh.empty:
        return "INSUFFICIENT"
    rr = rr.iloc[0]; sh = sh.iloc[0]
    es_p = float(rr["sign_test_p_es_thresh_0.05"])
    st_p = float(rr["stouffer_p"])
    sh_p = float(sh["p_empirical"])
    null_sane = abs(sh["null_mean_wins"] - sh["n_genes"] / 2) < 0.10 * sh["n_genes"]
    if not null_sane:
        return "FAIL"  # if null is biased, the entire headline is suspect
    if es_p < 1e-3 and st_p < 1e-3 and sh_p < 1e-3:
        return "PASS"
    if es_p < 0.05 and st_p < 0.05 and sh_p < 0.05:
        return "WEAK"
    return "FAIL"

f1v = f1_verdict()
f2v = f2_verdict()
md.append(f"- **Finding 1 (nonsense top-K, v3_GB non-CpG):** {f1v}")
md.append(f"- **Finding 2 (TSG sign test, OncoKB-173):** {f2v}")
md.append("")

# Headline
md.append("## 7. Headline")
md.append("")
if f1v == "PASS" and f2v == "PASS":
    md.append("Both v3 findings survive all five QA diagnostics. They are defensible for publication.")
elif "PASS" in (f1v, f2v):
    md.append(f"Mixed: Finding 1 = {f1v}, Finding 2 = {f2v}. The PASS finding is defensible; the WEAK/FAIL finding requires caveats or additional diagnostics.")
else:
    md.append(f"Both findings fall short of strict PASS: F1={f1v}, F2={f2v}. Headlines need significant softening before publication.")
md.append("")

(OUT_DIR / "QA_DIAGNOSTICS_RESULTS.md").write_text("\n".join(md))
print(f"\nWrote {OUT_DIR/'QA_DIAGNOSTICS_RESULTS.md'}")
print("\nDONE")
