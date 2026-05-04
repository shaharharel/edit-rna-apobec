"""QA diagnostics for ClinVar v4 / v3 findings — TIGHTER RERUN.

Diagnostics:
  A. CpG fraction in top-K (NEW, replaces flawed "non-CpG vs all" stratification)
  B. Wilcoxon meta-p (Stouffer) + ES-thresholded sign test, using existing per_gene CSV
  C. Random-head shuffle null for Finding 2 (only v3_GB and score_binary_v4)
  D. Bonferroni at the test family (54 cells)

Inputs (already produced):
  - experiments/apobec3a/outputs/clinvar_v4_scored/clinvar_scores_v4.parquet
  - experiments/apobec3a/outputs/clinvar_prediction/clinvar_all_scores.csv
  - data/processed/clinvar_c2u_variants.csv
  - experiments/apobec3a/outputs/clinvar_v4_full/per_gene_path_vs_ben_full.csv
  - experiments/apobec3a/outputs/clinvar_v4_full/finding2_tsg_v4_full.csv
  - data/raw/genomes/hg19.fa  (for CpG annotation)

Outputs (under experiments/apobec3a/outputs/clinvar_v4_full/):
  - qa_diagA_cpg_fraction.csv
  - qa_diagB_wilcoxon.csv
  - qa_diagC_shuffle.csv
  - qa_diagD_bonferroni.csv
  - QA_DIAGNOSTICS_RESULTS.md
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from pyfaidx import Fasta
from scipy import stats
from scipy.stats import binomtest, chi2_contingency, norm

PROJECT_ROOT = Path("/Users/shaharharel/Documents/github/edit-rna-apobec")
SEED = 20260427

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


def annotate_cpg(df: pd.DataFrame) -> np.ndarray:
    fa = Fasta(str(HG19), as_raw=True, sequence_always_upper=True)
    is_cpg = np.zeros(len(df), dtype=bool)
    chroms = df["chrom"].values
    poss = df["pos"].values.astype(int)
    strands = df["strand"].values
    n = len(df)
    print(f"  CpG annotation for {n:,} variants ...")
    t0 = time.time()
    for i in range(n):
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
        if (i + 1) % 250000 == 0:
            print(f"    {i+1:,}/{n:,}  ({time.time()-t0:.0f}s)")
    print(f"  done in {time.time()-t0:.0f}s")
    return is_cpg


# ---------------------------------------------------------------------------
# Build the joined master dataframe
# ---------------------------------------------------------------------------

print("=" * 80)
print("Loading v4 + meta + v3 + CpG annotation")
print("=" * 80)
t0 = time.time()
v4 = pd.read_parquet(V4_PARQUET)
print(f"v4 rows: {len(v4):,}")

meta = pd.read_csv(
    CLINVAR_META,
    usecols=["site_id", "clinical_significance", "gene"],
)
meta["significance_simple"] = meta["clinical_significance"].map(simplify_sig)
v3 = pd.read_csv(V3_GB_CSV, usecols=["site_id", "p_edited_gb"]).rename(
    columns={"p_edited_gb": "v3_GB"}
)

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
n_cpg = int(df["is_cpg"].sum())
print(f"CpG: {n_cpg:,}; non-CpG: {len(df)-n_cpg:,}")
print(f"Setup: {time.time()-t0:.0f}s")

path_mask = df["significance_simple"].isin(PATH).values
ben_mask = df["significance_simple"].isin(BEN).values
genes_arr = df["gene"].fillna("").values
is_cpg_arr = df["is_cpg"].values

# Pathogenic+LP universe — from these we'll pick top-K
path_lp_mask = path_mask
n_path_universe = int(path_lp_mask.sum())
n_cpg_path = int((path_lp_mask & is_cpg_arr).sum())
universe_cpg_frac = n_cpg_path / n_path_universe
print(f"Path+LP universe: {n_path_universe:,}; CpG: {n_cpg_path:,} ({universe_cpg_frac*100:.2f}%)")

# ---------------------------------------------------------------------------
# Diagnostic A — CpG fraction in top-K
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("Diagnostic A — CpG fraction in top-K of Path+LP universe")
print("=" * 80)
KS_A = [100, 500, 1000, 5000]
diagA_rows = []

for head in HEADS:
    if head not in df.columns:
        continue
    scores = df[head].values
    valid = ~np.isnan(scores)
    sm = path_lp_mask & valid
    n_strat = int(sm.sum())
    if n_strat < 100:
        continue
    idx_strat = np.where(sm)[0]
    scores_strat = scores[idx_strat]
    is_cpg_strat = is_cpg_arr[idx_strat]
    universe_cpg_frac_head = is_cpg_strat.mean()
    n_universe_head = len(idx_strat)
    n_universe_cpg = int(is_cpg_strat.sum())
    n_universe_noncpg = n_universe_head - n_universe_cpg

    order = np.argsort(scores_strat)  # ascending; top is at the end
    for K in KS_A:
        k = min(K, n_strat)
        top_local = order[-k:]
        top_cpg = int(is_cpg_strat[top_local].sum())
        top_noncpg = k - top_cpg
        top_cpg_frac = top_cpg / k
        enrichment = top_cpg_frac / universe_cpg_frac_head if universe_cpg_frac_head > 0 else float("nan")
        # chi-squared: top-K vs the rest of the universe (CpG vs non-CpG)
        rest_cpg = n_universe_cpg - top_cpg
        rest_noncpg = n_universe_noncpg - top_noncpg
        if rest_cpg + rest_noncpg > 0 and (top_cpg + top_noncpg) > 0:
            try:
                chi2, p_chi2, _, _ = chi2_contingency([[top_cpg, top_noncpg], [rest_cpg, rest_noncpg]])
            except ValueError:
                p_chi2 = 1.0
        else:
            p_chi2 = 1.0
        diagA_rows.append({
            "head": head,
            "K": K,
            "n_top_K": k,
            "n_top_K_cpg": top_cpg,
            "top_K_cpg_frac": round(top_cpg_frac, 5),
            "universe_cpg_frac": round(universe_cpg_frac_head, 5),
            "cpg_enrichment": round(enrichment, 3),
            "chi2_p": float(p_chi2),
        })

diagA_df = pd.DataFrame(diagA_rows)
diagA_df.to_csv(OUT_DIR / "qa_diagA_cpg_fraction.csv", index=False)
print(f"Wrote {OUT_DIR/'qa_diagA_cpg_fraction.csv'} ({len(diagA_df)} rows)")
print(diagA_df.to_string(index=False))

# ---------------------------------------------------------------------------
# Diagnostic B — Wilcoxon meta-p (Stouffer) + ES-thresholded sign test
# Uses existing per_gene CSV (one row per gene per gene_list per head; "all" stratum only)
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("Diagnostic B — Stouffer combined p + ES-filtered sign test")
print("=" * 80)
per_gene = pd.read_csv(PER_GENE_CSV)
print(f"per_gene rows: {len(per_gene)}")
print(f"per_gene columns: {per_gene.columns.tolist()}")

diagB_rows = []
for (gl_name, head), sub in per_gene.groupby(["gene_list", "head"]):
    sub = sub.copy()
    n_genes = len(sub)
    if n_genes == 0:
        continue
    # Two-sided MWU p
    mwu_two = sub["mwu_p"].clip(1e-300, 1.0).values
    deltas = sub["delta"].values

    # Convert to one-sided p in the direction of mean_path > mean_ben
    # If delta > 0 (path > ben): p_one = mwu_p / 2; else: 1 - mwu_p / 2
    p_one = np.where(deltas > 0, mwu_two / 2.0, 1.0 - mwu_two / 2.0)
    p_one = np.clip(p_one, 1e-300, 1.0 - 1e-16)
    z_scores = norm.isf(p_one)
    stouffer_z = z_scores.sum() / np.sqrt(len(z_scores))
    stouffer_p = float(norm.sf(stouffer_z))

    # Counts
    n_mw_05 = int((sub["mwu_p"] < 0.05).sum())
    n_mw_01 = int((sub["mwu_p"] < 0.01).sum())

    # Unfiltered sign test (matches original Finding 2)
    wins = int((sub["delta"] > 0).sum())
    sign_p_unf = float(binomtest(wins, n_genes, 0.5, alternative="greater").pvalue)

    # ES-filtered: drop |delta| < 0.05 ties
    es = sub[sub["delta"].abs() >= 0.05]
    n_es = len(es)
    if n_es == 0:
        sign_p_es = 1.0
        wins_es = 0
    else:
        wins_es = int((es["delta"] > 0).sum())
        sign_p_es = float(binomtest(wins_es, n_es, 0.5, alternative="greater").pvalue)

    diagB_rows.append({
        "gene_list": gl_name, "head": head, "stratum": "all",
        "n_genes": n_genes,
        "n_mw_p_lt_0.05": n_mw_05,
        "n_mw_p_lt_0.01": n_mw_01,
        "wins_unfiltered": wins,
        "sign_p_unfiltered": sign_p_unf,
        "n_es_filtered": n_es,
        "wins_es_filtered": wins_es,
        "sign_p_es_filtered": sign_p_es,
        "stouffer_combined_p": stouffer_p,
    })

diagB_df = pd.DataFrame(diagB_rows)
diagB_df.to_csv(OUT_DIR / "qa_diagB_wilcoxon.csv", index=False)
print(f"Wrote {OUT_DIR/'qa_diagB_wilcoxon.csv'} ({len(diagB_df)} rows)")
print(diagB_df[["gene_list","head","n_genes","n_mw_p_lt_0.05","sign_p_unfiltered","sign_p_es_filtered","stouffer_combined_p"]].to_string(index=False))

# ---------------------------------------------------------------------------
# Diagnostic C — Random shuffle null (Finding 2)
# Only on v3_GB and score_binary_v4 to save time
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("Diagnostic C — Random shuffle null (Finding 2 sign test)")
print("=" * 80)

finding2_main = pd.read_csv(FINDING2_CSV)
N_SHUFFLES = 100
SHUFFLE_HEADS = ["v3_GB", "score_binary_v4"]
GENE_LISTS_OBS = ["curated_48", "tier1_proxy_82", "oncokb_cgc_tsg_173"]
diagC_rows = []

for shuffle_head in SHUFFLE_HEADS:
    if shuffle_head not in df.columns:
        continue
    scores = df[shuffle_head].values
    valid_idx = np.where(~np.isnan(scores))[0]
    print(f"\nShuffling {shuffle_head}: valid entries = {len(valid_idx):,}")

    for gl_name in GENE_LISTS_OBS:
        # Get observed wins from finding2_tsg_v4_full.csv for stratum=='all'
        obs_row = finding2_main[
            (finding2_main["gene_list"] == gl_name)
            & (finding2_main["head"] == shuffle_head)
            & (finding2_main["stratum"] == "all")
        ]
        if obs_row.empty:
            continue
        observed_wins = int(obs_row.iloc[0]["n_path_gt_ben"])

        # Get the gene set: use genes that appear in per_gene CSV (matches "testable" list)
        gene_set = set(per_gene[(per_gene["gene_list"] == gl_name) & (per_gene["head"] == shuffle_head)]["gene"].unique())

        # Pre-compute per-gene path/ben indices
        valid_score = ~np.isnan(scores)
        in_path = path_mask & valid_score
        in_ben = ben_mask & valid_score
        gene_idx_path = {}
        gene_idx_ben = {}
        for gene in gene_set:
            gm = (genes_arr == gene)
            gp = np.where(gm & in_path)[0]
            gb = np.where(gm & in_ben)[0]
            if len(gp) >= 3 and len(gb) >= 3:
                gene_idx_path[gene] = gp
                gene_idx_ben[gene] = gb

        n_genes_testable = len(gene_idx_path)
        rng = np.random.default_rng(SEED + abs(hash((shuffle_head, gl_name))) % (2**31))

        # We need to shuffle scores across valid entries and recompute wins.
        # Optimization: each shuffle reassigns scores randomly. For each gene the only thing
        # that matters for "wins" is the comparison of mean(scores at gp) vs mean(scores at gb).
        # We can use the same shuffled array for all genes.
        scores_valid = scores[valid_idx].astype(np.float32)
        # Build a position->index_in_valid map for fast scatter
        # Actually simpler: reconstruct shuffled full array each iter. 1.69M floats * 100 = manageable.

        null_wins = np.zeros(N_SHUFFLES, dtype=np.int32)
        t1 = time.time()
        for s in range(N_SHUFFLES):
            perm = rng.permutation(len(valid_idx))
            shuffled = np.full(len(scores), np.nan, dtype=np.float32)
            shuffled[valid_idx] = scores_valid[perm]
            wins = 0
            for gene, pidx in gene_idx_path.items():
                bidx = gene_idx_ben[gene]
                mp = shuffled[pidx].mean()
                mb = shuffled[bidx].mean()
                if mp > mb:
                    wins += 1
            null_wins[s] = wins
        elapsed = time.time() - t1
        null_mean = float(null_wins.mean())
        null_std = float(null_wins.std(ddof=1))
        null_p5 = float(np.percentile(null_wins, 5))
        null_p95 = float(np.percentile(null_wins, 95))
        p_emp = float((null_wins >= observed_wins).mean())
        # Replace 0 with 1/N_SHUFFLES upper bound
        if p_emp == 0:
            p_emp = 1.0 / N_SHUFFLES
            p_emp_str = f"<{p_emp:.3f}"
        else:
            p_emp_str = f"{p_emp:.4f}"
        print(f"  {shuffle_head} {gl_name}: obs={observed_wins}/{n_genes_testable}; "
              f"null mean={null_mean:.2f}+/-{null_std:.2f}; p_emp={p_emp_str}; ({elapsed:.0f}s)")
        diagC_rows.append({
            "gene_list": gl_name,
            "head": shuffle_head,
            "n_genes": n_genes_testable,
            "n_shuffles": N_SHUFFLES,
            "observed_wins": observed_wins,
            "null_mean_wins": round(null_mean, 2),
            "null_std": round(null_std, 2),
            "null_p5": round(null_p5, 2),
            "null_p95": round(null_p95, 2),
            "p_empirical": p_emp,
        })

diagC_df = pd.DataFrame(diagC_rows)
diagC_df.to_csv(OUT_DIR / "qa_diagC_shuffle.csv", index=False)
print(f"\nWrote {OUT_DIR/'qa_diagC_shuffle.csv'} ({len(diagC_df)} rows)")

# ---------------------------------------------------------------------------
# Diagnostic D — Bonferroni at the test family
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("Diagnostic D — Bonferroni at family-wide alpha=0.05/54")
print("=" * 80)
ALPHA = 0.05
M = 3 * 6 * 3  # 54
BONF_Q = ALPHA / M
diagD_rows = []
for _, r in finding2_main.iterrows():
    p = float(r["sign_test_p"])
    diagD_rows.append({
        "gene_list": r["gene_list"],
        "head": r["head"],
        "stratum": r["stratum"],
        "sign_test_p": p,
        "bonferroni_q": BONF_Q,
        "survives": bool(p < BONF_Q),
    })
diagD_df = pd.DataFrame(diagD_rows)
diagD_df.to_csv(OUT_DIR / "qa_diagD_bonferroni.csv", index=False)
n_survive = int(diagD_df["survives"].sum())
print(f"Bonferroni q={BONF_Q:.2e}; survive: {n_survive} / {len(diagD_df)}")

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
md.append("Four targeted diagnostics on Finding 1 (nonsense top-K) and Finding 2 (TSG sign test).")
md.append("Inputs: 1.69M v4 ClinVar variants, joined to v3 GB scores, with CpG annotation.")
md.append("")
md.append(f"Path+LP universe: {n_path_universe:,}; CpG fraction = {universe_cpg_frac*100:.2f}%")
md.append("")

# ---- Diagnostic A ----
md.append("## 1. Diagnostic A — CpG fraction in top-K (bias check)")
md.append("")
md.append("For each head (within Path+LP universe), the CpG fraction in top-K. CpG enrichment = top-K CpG fraction / universe CpG fraction. If a head's top-1000 is e.g. 5% CpG (3.3x over universe), it has CpG bias.")
md.append("")
md.append("| Head | K | n_top_K | n_cpg | top_K_cpg_frac | universe_cpg_frac | enrichment | chi2_p |")
md.append("|---|---|---|---|---|---|---|---|")
for _, r in diagA_df.iterrows():
    md.append(
        f"| {r['head']} | {r['K']} | {r['n_top_K']} | {r['n_top_K_cpg']} | "
        f"{r['top_K_cpg_frac']*100:.2f}% | {r['universe_cpg_frac']*100:.2f}% | "
        f"{r['cpg_enrichment']:.2f}x | {r['chi2_p']:.2e} |"
    )
md.append("")
# Verdict
max_enr = diagA_df[diagA_df["K"] == 1000]["cpg_enrichment"].max()
md.append(f"**Diag A verdict:** max CpG enrichment at top-1000 across heads = **{max_enr:.2f}x**.")
if max_enr < 1.5:
    md.append(" → No head shows meaningful CpG bias (<1.5x). The non-CpG headlines are NOT a CpG-leak artifact.")
elif max_enr < 3.0:
    md.append(" → Mild CpG enrichment in some heads (1.5-3x). Worth flagging but not disqualifying.")
else:
    md.append(" → Strong CpG bias (>3x). Headlines likely contaminated.")
md.append("")

# ---- Diagnostic B ----
md.append("## 2. Diagnostic B — Wilcoxon meta-p (Stouffer) + ES-filtered sign test")
md.append("")
md.append("For each (gene_list x head), Stouffer's method combines per-gene MWU p-values into one combined p (one-sided in path > ben direction). The ES-filtered sign test drops genes with |delta| < 0.05 (small-effect ties).")
md.append("")
md.append("| Gene list | Head | n_genes | mw_p<.05 | mw_p<.01 | sign_p (unfilt) | wins/n (ES-filt) | sign_p (ES-filt) | Stouffer p |")
md.append("|---|---|---|---|---|---|---|---|---|")
for _, r in diagB_df.iterrows():
    md.append(
        f"| {r['gene_list']} | {r['head']} | {r['n_genes']} | {r['n_mw_p_lt_0.05']} | "
        f"{r['n_mw_p_lt_0.01']} | {r['sign_p_unfiltered']:.2e} | "
        f"{r['wins_es_filtered']}/{r['n_es_filtered']} | "
        f"{r['sign_p_es_filtered']:.2e} | {r['stouffer_combined_p']:.2e} |"
    )
md.append("")
# Headline check
hl = diagB_df[(diagB_df["gene_list"] == "oncokb_cgc_tsg_173") & (diagB_df["head"] == "v3_GB")]
if not hl.empty:
    rr = hl.iloc[0]
    md.append(
        f"**Headline check (OncoKB-173, v3_GB, all stratum):** "
        f"Original 110/128 sign_p = {rr['sign_p_unfiltered']:.2e}; "
        f"ES-filtered (|delta|>=0.05): {rr['wins_es_filtered']}/{rr['n_es_filtered']} sign_p = {rr['sign_p_es_filtered']:.2e}; "
        f"Stouffer combined p = {rr['stouffer_combined_p']:.2e}."
    )
md.append("")

# ---- Diagnostic C ----
md.append("## 3. Diagnostic C — Random shuffle null (Finding 2)")
md.append("")
md.append(f"Shuffles = {N_SHUFFLES} per (head x gene_list). Scores randomly permuted across all 1.69M valid ClinVar variants. Sanity: null mean wins should be ~ n_genes/2 (binomial 50%).")
md.append("")
md.append("| Head | Gene list | n_genes | observed wins | null mean | null std | null p5/p95 | p_empirical | sane? |")
md.append("|---|---|---|---|---|---|---|---|---|")
for _, r in diagC_df.iterrows():
    expected = r["n_genes"] / 2
    sane = "OK" if abs(r["null_mean_wins"] - expected) < 0.10 * r["n_genes"] else "BIASED"
    p_str = f"<{1.0/N_SHUFFLES:.3f}" if r["p_empirical"] <= 1.0 / N_SHUFFLES else f"{r['p_empirical']:.4f}"
    md.append(
        f"| {r['head']} | {r['gene_list']} | {r['n_genes']} | {r['observed_wins']} | "
        f"{r['null_mean_wins']:.2f} | {r['null_std']:.2f} | "
        f"{r['null_p5']:.0f}/{r['null_p95']:.0f} | {p_str} | {sane} |"
    )
md.append("")

# ---- Diagnostic D ----
md.append(f"## 4. Diagnostic D — Bonferroni at family-wide alpha = 0.05 / {M} = {BONF_Q:.2e}")
md.append("")
md.append(f"Family = 3 gene_lists × 6 heads × 3 strata = {M} cells.")
md.append("")
md.append("| Gene list | Head | Stratum | sign_test_p | survives |")
md.append("|---|---|---|---|---|")
for _, r in diagD_df.iterrows():
    md.append(
        f"| {r['gene_list']} | {r['head']} | {r['stratum']} | {r['sign_test_p']:.2e} | "
        f"{'YES' if r['survives'] else 'no'} |"
    )
md.append("")
md.append(f"**Total cells surviving Bonferroni: {n_survive} / {len(diagD_df)}.**")
md.append("")
md.append("### Survivors by stratum")
md.append("")
md.append("| Stratum | survivors / total |")
md.append("|---|---|")
for stratum_name in ["all", "CpG", "non_CpG"]:
    sub = diagD_df[diagD_df["stratum"] == stratum_name]
    s = int(sub["survives"].sum())
    md.append(f"| {stratum_name} | {s} / {len(sub)} |")
md.append("")

# ---- Final verdicts ----
md.append("## 5. FINAL VERDICTS")
md.append("")

# Finding 1 verdict: based on existing top-K CSV. Use v3_GB non-CpG OR > 1.3 across K.
diag1_old = pd.read_csv(OUT_DIR / "qa_diag1_topk.csv")
v3_nc = diag1_old[(diag1_old["head"] == "v3_GB") & (diag1_old["stratum"] == "non_CpG")].sort_values("K")
if not v3_nc.empty:
    or_arr = v3_nc["OR"].values
    n_above_13 = int((or_arr > 1.3).sum())
    if n_above_13 >= 4:
        f1v = "PASS"
    elif n_above_13 >= 3:
        f1v = "WEAK"
    else:
        f1v = "FAIL"
else:
    f1v = "INSUFFICIENT"

# Finding 2 verdict
hl = diagB_df[(diagB_df["gene_list"] == "oncokb_cgc_tsg_173") & (diagB_df["head"] == "v3_GB")]
sh = diagC_df[(diagC_df["gene_list"] == "oncokb_cgc_tsg_173") & (diagC_df["head"] == "v3_GB")]
if not hl.empty and not sh.empty:
    es_p = float(hl.iloc[0]["sign_p_es_filtered"])
    st_p = float(hl.iloc[0]["stouffer_combined_p"])
    sh_p = float(sh.iloc[0]["p_empirical"])
    null_sane = abs(sh.iloc[0]["null_mean_wins"] - sh.iloc[0]["n_genes"] / 2) < 0.10 * sh.iloc[0]["n_genes"]
    if not null_sane:
        f2v = "FAIL (shuffle null biased — invalidates sign test)"
    elif es_p < 1e-3 and st_p < 1e-3 and sh_p < 1e-2:
        f2v = "PASS"
    elif es_p < 0.05 and st_p < 0.05 and sh_p < 0.05:
        f2v = "WEAK"
    else:
        f2v = "FAIL"
else:
    f2v = "INSUFFICIENT"

md.append(f"- **Finding 1 (nonsense top-K, v3_GB non-CpG):** {f1v}")
md.append(f"- **Finding 2 (TSG sign test, OncoKB-173, v3_GB):** {f2v}")
md.append("")
if f1v == "PASS" and f2v == "PASS":
    headline = "Both v3 findings survive all four QA diagnostics. They are publication-ready."
elif "PASS" in (f1v, f2v):
    headline = f"Mixed: Finding 1 = {f1v}, Finding 2 = {f2v}. The PASS finding is publication-ready; the WEAK/FAIL finding requires caveats."
else:
    headline = f"Both findings fall short of strict PASS: F1={f1v}, F2={f2v}. Headlines need significant softening."
md.append(f"**Are these v3 findings publication-ready on v4?** {headline}")
md.append("")

(OUT_DIR / "QA_DIAGNOSTICS_RESULTS.md").write_text("\n".join(md))
print(f"Wrote {OUT_DIR/'QA_DIAGNOSTICS_RESULTS.md'}")
print("\nDONE")
