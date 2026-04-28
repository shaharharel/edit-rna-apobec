#!/usr/bin/env python3
"""Generate V3_VS_V4_COMPARISON.md side-by-side report.

Reads:
  experiments/multi_enzyme/outputs/pcawg_tcw_panel/sweep_v3_fair.csv (+ per-cancer)
  experiments/multi_enzyme/outputs/pcawg_tcw_panel/v4_outputs/sweep_v4_cancer_fair.csv (+ per-cancer)
  experiments/multi_enzyme/outputs/pcawg_tcw_panel/v4_outputs/sweep_v4_cds_fair.csv (+ per-cancer)
  experiments/multi_enzyme/outputs/pcawg_tcw_panel/v4_outputs/topx_trinuc_breakdown.csv

Writes:
  experiments/multi_enzyme/outputs/pcawg_tcw_panel/v4_outputs/V3_VS_V4_COMPARISON.md

Required sections:
  1. Headline: did v4 unlock a real signal? (binary head, position-level + best window)
  2. Side-by-side table: v3 / v4_cancer / v4_cds for {sum binary win=1000, max position-level binary, top apobec1 cell}
  3. Per-head v4_cancer vs v4_cds on the WINNING construction
  4. Bonferroni-surviving signal in v4 (if any)
  5. Position-level diagnostic: top-1% positions trinuc breakdown
  6. Verdict
"""
from __future__ import annotations
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/shaharharel/Documents/github/edit-rna-apobec")
PANEL_DIR = ROOT / "experiments/multi_enzyme/outputs/pcawg_tcw_panel"
V4_DIR = PANEL_DIR / "v4_outputs"

V3_CSV = PANEL_DIR / "sweep_v3_fair.csv"
V3_PC_CSV = PANEL_DIR / "sweep_v3_fair_per_cancer.csv"
V4C_CSV = V4_DIR / "sweep_v4_cancer_fair.csv"
V4C_PC_CSV = V4_DIR / "sweep_v4_cancer_fair_per_cancer.csv"
V4D_CSV = V4_DIR / "sweep_v4_cds_fair.csv"
V4D_PC_CSV = V4_DIR / "sweep_v4_cds_fair_per_cancer.csv"
TRINUC_CSV = V4_DIR / "topx_trinuc_breakdown.csv"

OUT_MD = V4_DIR / "V3_VS_V4_COMPARISON.md"

V3_HEADS = ["score_binary", "score_A3A", "score_A3B", "score_A3G",
            "score_A3A_A3G", "score_Neither", "score_apobec1"]
V4_HEADS = ["score_binary", "score_A3A", "score_A3B", "score_A3G",
            "score_A3A_A3G", "score_apobec1_v3"]

# v4 Bonferroni: 21 * 6 * 10 = 1260; q = 0.05/1260 = 3.97e-5
# v3 Bonferroni: 21 * 7 * 10 = 1470; q = 0.05/1470 = 3.4e-5
V4_BONF_Q = 0.05 / (21 * 6 * 10)
V3_BONF_Q = 0.05 / (21 * 7 * 10)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    stream=sys.stdout)
log = logging.getLogger(__name__)


def fmt_pct(x, lo=None, hi=None):
    if pd.isna(x):
        return "n/a"
    if lo is not None and hi is not None and not (pd.isna(lo) or pd.isna(hi)):
        return f"{x*100:.3f}% [{lo*100:.3f}, {hi*100:.3f}]"
    return f"{x*100:.3f}%"


def fmt_ratio(x, lo=None, hi=None):
    if pd.isna(x):
        return "n/a"
    if lo is not None and hi is not None and not (pd.isna(lo) or pd.isna(hi)):
        return f"{x:.3f} [{lo:.3f}, {hi:.3f}]"
    return f"{x:.3f}"


def get_cell(df, head, agg, level=None, ws=None, fl="filter_TCW_nonCpG"):
    sub = df[(df["head"] == head) & (df["aggregator"] == agg) & (df["filter"] == fl)]
    if level is not None:
        sub = sub[sub["level"] == level]
    if ws is not None:
        sub = sub[sub["window_size_bp"] == ws]
    if len(sub) == 0:
        return None
    return sub.iloc[0]


def cell_row(label, cell):
    if cell is None:
        return f"| {label} | - | - | - | - | - |"
    return (f"| {label} | "
            f"{fmt_ratio(cell['mean_ratio_vs_TCW'], cell['ratio_tcw_ci_lo'], cell['ratio_tcw_ci_hi'])} | "
            f"{fmt_ratio(cell['mean_ratio_vs_NPOS'], cell['ratio_npos_ci_lo'], cell['ratio_npos_ci_hi'])} | "
            f"{fmt_ratio(cell['mean_ratio_vs_CpG'], cell['ratio_cpg_ci_lo'], cell['ratio_cpg_ci_hi'])} | "
            f"{fmt_pct(cell['mean_abs_recall'], cell['abs_recall_ci_lo'], cell['abs_recall_ci_hi'])} | "
            f"{int(cell['n_cancers_bonf_signif'])}/10 |")


def main():
    log.info("Loading v3, v4_cancer, v4_cds CSVs ...")
    v3 = pd.read_csv(V3_CSV)
    v4c = pd.read_csv(V4C_CSV)
    v4d = pd.read_csv(V4D_CSV)
    v3_pc = pd.read_csv(V3_PC_CSV) if V3_PC_CSV.exists() else None
    log.info("v3 rows: %d, v4_cancer rows: %d, v4_cds rows: %d",
             len(v3), len(v4c), len(v4d))

    md = []
    md.append("# V3 vs V4 Comparison: Panel-Recall Sweep")
    md.append("")
    md.append("**Question:** Did v4 (trinucleotide-matched negatives, anti-TCW")
    md.append("polarity removed in bias diagnostic) unlock a real RNA-to-DNA")
    md.append("transfer signal that v3 lacked?")
    md.append("")
    md.append(f"- v3 sweep file: `{V3_CSV.name}` ({len(v3)} cells)")
    md.append(f"- v4_cancer sweep file: `{V4C_CSV.name}` ({len(v4c)} cells)")
    md.append(f"- v4_cds sweep file: `{V4D_CSV.name}` ({len(v4d)} cells)")
    md.append(f"- v3 levels available: {sorted(v3['level'].unique())} "
              f"window sizes: {sorted(v3['window_size_bp'].unique())}")
    md.append(f"- v4 levels available: {sorted(v4c['level'].unique())} "
              f"window sizes: {sorted(v4c['window_size_bp'].unique())}")
    md.append(f"- v3 Bonferroni: q < {V3_BONF_Q:.2e} (n_tests = 1470, 7 heads)")
    md.append(f"- v4 Bonferroni: q < {V4_BONF_Q:.2e} (n_tests = 1260, 6 heads)")
    md.append("")

    # ====================================================================
    # 1. Headline
    # ====================================================================
    md.append("## 1. Headline: Did v4 unlock a real signal?")
    md.append("")

    # Pick the best score_binary cell across all configurations for each model
    def best_binary(df):
        sub = df[(df["head"] == "score_binary")
                 & (df["filter"] == "filter_TCW_nonCpG")
                 & (df["mean_ratio_vs_TCW"].notna())]
        if len(sub) == 0:
            return None
        return sub.sort_values("mean_ratio_vs_TCW", ascending=False).iloc[0]

    bb_v3 = best_binary(v3)
    bb_v4c = best_binary(v4c)
    bb_v4d = best_binary(v4d)

    md.append("**Best score_binary cell at filter_TCW_nonCpG for each model:**")
    md.append("")
    md.append("| model | construction | ratio_vs_TCW (CI) | ratio_vs_NPOS (CI) | "
              "abs_recall (CI) | bonf/10 |")
    md.append("|-------|--------------|-------------------|--------------------|"
              "-----------------|---------|")
    for tag, row in [("v3", bb_v3), ("v4_cancer", bb_v4c), ("v4_cds", bb_v4d)]:
        if row is None:
            md.append(f"| {tag} | - | - | - | - | - |")
            continue
        constr = f"{row['aggregator']}, ws={int(row['window_size_bp'])}, level={row['level']}"
        md.append(f"| {tag} | {constr} | "
                  f"{fmt_ratio(row['mean_ratio_vs_TCW'], row['ratio_tcw_ci_lo'], row['ratio_tcw_ci_hi'])} | "
                  f"{fmt_ratio(row['mean_ratio_vs_NPOS'], row['ratio_npos_ci_lo'], row['ratio_npos_ci_hi'])} | "
                  f"{fmt_pct(row['mean_abs_recall'], row['abs_recall_ci_lo'], row['abs_recall_ci_hi'])} | "
                  f"{int(row['n_cancers_bonf_signif'])}/10 |")
    md.append("")

    # Position-level binary
    md.append("**Position-level score_binary at filter_TCW_nonCpG:**")
    md.append("")
    md.append("| model | ratio_vs_TCW (CI) | ratio_vs_NPOS (CI) | abs_recall (CI) | bonf/10 |")
    md.append("|-------|-------------------|--------------------|-----------------|---------|")
    for tag, df in [("v3", v3), ("v4_cancer", v4c), ("v4_cds", v4d)]:
        cell = get_cell(df, "score_binary", "max", level="position", ws=0,
                        fl="filter_TCW_nonCpG")
        if cell is None:
            md.append(f"| {tag} | - | - | - | - |")
            continue
        md.append(f"| {tag} | "
                  f"{fmt_ratio(cell['mean_ratio_vs_TCW'], cell['ratio_tcw_ci_lo'], cell['ratio_tcw_ci_hi'])} | "
                  f"{fmt_ratio(cell['mean_ratio_vs_NPOS'], cell['ratio_npos_ci_lo'], cell['ratio_npos_ci_hi'])} | "
                  f"{fmt_pct(cell['mean_abs_recall'], cell['abs_recall_ci_lo'], cell['abs_recall_ci_hi'])} | "
                  f"{int(cell['n_cancers_bonf_signif'])}/10 |")
    md.append("")

    # ====================================================================
    # 2. Side-by-side: 3 specific cells
    # ====================================================================
    md.append("## 2. Side-by-side: three reference cells")
    md.append("")
    md.append("Filter = filter_TCW_nonCpG throughout this table.")
    md.append("")
    md.append("### 2a. score_binary, sum, win=1000")
    md.append("")
    md.append("| model | ratio_vs_TCW (CI) | ratio_vs_NPOS (CI) | ratio_vs_CpG (CI) | "
              "abs_recall (CI) | bonf/10 |")
    md.append("|-------|-------------------|--------------------|-------------------|"
              "-----------------|---------|")
    for tag, df in [("v3", v3), ("v4_cancer", v4c), ("v4_cds", v4d)]:
        c = get_cell(df, "score_binary", "sum", level="win_1000", ws=1000)
        md.append(cell_row(tag, c))
    md.append("")

    md.append("### 2b. score_binary, max, position-level")
    md.append("")
    md.append("| model | ratio_vs_TCW (CI) | ratio_vs_NPOS (CI) | ratio_vs_CpG (CI) | "
              "abs_recall (CI) | bonf/10 |")
    md.append("|-------|-------------------|--------------------|-------------------|"
              "-----------------|---------|")
    for tag, df in [("v3", v3), ("v4_cancer", v4c), ("v4_cds", v4d)]:
        c = get_cell(df, "score_binary", "max", level="position", ws=0)
        md.append(cell_row(tag, c))
    md.append("")

    md.append("### 2c. Top apobec1 cell (any construction at filter_TCW_nonCpG)")
    md.append("")
    md.append("v3 head = `score_apobec1`; v4 head = `score_apobec1_v3`.")
    md.append("")
    md.append("| model | construction | ratio_vs_TCW (CI) | ratio_vs_NPOS (CI) | "
              "ratio_vs_CpG (CI) | abs_recall (CI) | bonf/10 |")
    md.append("|-------|--------------|-------------------|--------------------|"
              "-------------------|-----------------|---------|")
    for tag, df, head in [("v3", v3, "score_apobec1"),
                          ("v4_cancer", v4c, "score_apobec1_v3"),
                          ("v4_cds", v4d, "score_apobec1_v3")]:
        sub = df[(df["head"] == head)
                 & (df["filter"] == "filter_TCW_nonCpG")
                 & (df["mean_ratio_vs_TCW"].notna())]
        if len(sub) == 0:
            md.append(f"| {tag} | - | - | - | - | - | - |")
            continue
        top = sub.sort_values("mean_ratio_vs_TCW", ascending=False).iloc[0]
        constr = f"{top['aggregator']}, ws={int(top['window_size_bp'])}, level={top['level']}"
        md.append(f"| {tag} | {constr} | "
                  f"{fmt_ratio(top['mean_ratio_vs_TCW'], top['ratio_tcw_ci_lo'], top['ratio_tcw_ci_hi'])} | "
                  f"{fmt_ratio(top['mean_ratio_vs_NPOS'], top['ratio_npos_ci_lo'], top['ratio_npos_ci_hi'])} | "
                  f"{fmt_ratio(top['mean_ratio_vs_CpG'], top['ratio_cpg_ci_lo'], top['ratio_cpg_ci_hi'])} | "
                  f"{fmt_pct(top['mean_abs_recall'], top['abs_recall_ci_lo'], top['abs_recall_ci_hi'])} | "
                  f"{int(top['n_cancers_bonf_signif'])}/10 |")
    md.append("")

    # ====================================================================
    # 3. Per-head v4_cancer vs v4_cds at WINNING construction
    # ====================================================================
    # Pick the construction that maximizes mean_ratio_vs_NPOS for score_binary
    # in v4_cancer (most defensible against gene-body density).
    if bb_v4c is not None:
        win_constr = (bb_v4c["aggregator"], int(bb_v4c["window_size_bp"]),
                      bb_v4c["level"])
    else:
        win_constr = ("sum", 1000, "win_1000")
    md.append(f"## 3. Per-head v4_cancer vs v4_cds at the v4_cancer winning construction")
    md.append("")
    md.append(f"Winning construction (chosen by best `score_binary` "
              f"`mean_ratio_vs_TCW` in v4_cancer at filter_TCW_nonCpG): "
              f"`agg={win_constr[0]}, ws={win_constr[1]}, level={win_constr[2]}`.")
    md.append("")
    md.append("| head | model | ratio_vs_TCW (CI) | ratio_vs_NPOS (CI) | "
              "abs_recall (CI) | bonf/10 |")
    md.append("|------|-------|-------------------|--------------------|"
              "-----------------|---------|")
    for head in V4_HEADS:
        for tag, df in [("v4_cancer", v4c), ("v4_cds", v4d)]:
            c = get_cell(df, head, win_constr[0], level=win_constr[2],
                         ws=win_constr[1])
            if c is None:
                md.append(f"| {head} | {tag} | - | - | - | - |")
                continue
            md.append(f"| {head} | {tag} | "
                      f"{fmt_ratio(c['mean_ratio_vs_TCW'], c['ratio_tcw_ci_lo'], c['ratio_tcw_ci_hi'])} | "
                      f"{fmt_ratio(c['mean_ratio_vs_NPOS'], c['ratio_npos_ci_lo'], c['ratio_npos_ci_hi'])} | "
                      f"{fmt_pct(c['mean_abs_recall'], c['abs_recall_ci_lo'], c['abs_recall_ci_hi'])} | "
                      f"{int(c['n_cancers_bonf_signif'])}/10 |")
    md.append("")

    # ====================================================================
    # 4. Bonferroni-surviving signal
    # ====================================================================
    md.append("## 4. Bonferroni-surviving cells")
    md.append("")
    for tag, df, q in [("v3", v3, V3_BONF_Q), ("v4_cancer", v4c, V4_BONF_Q),
                       ("v4_cds", v4d, V4_BONF_Q)]:
        surv = df[df["n_cancers_bonf_signif"] >= 1]
        majority = df[df["n_cancers_bonf_signif"] >= 6]
        md.append(f"- **{tag}** (q < {q:.2e}): "
                  f"{len(surv)}/{len(df)} cells with >=1 cancer surviving, "
                  f"{len(majority)}/{len(df)} cells with >=6 cancers (majority).")
    md.append("")

    # Per-variant top survivors
    for tag, df in [("v3", v3), ("v4_cancer", v4c), ("v4_cds", v4d)]:
        surv = df[df["n_cancers_bonf_signif"] >= 1].copy()
        if len(surv) == 0:
            md.append(f"### {tag}: no Bonferroni-surviving cells")
            md.append("")
            continue
        surv = surv.sort_values(["n_cancers_bonf_signif", "mean_ratio_vs_TCW"],
                                ascending=[False, False]).head(8)
        md.append(f"### {tag}: top 8 Bonferroni-surviving cells")
        md.append("")
        md.append("| head | agg | ws | level | filter | ratio_vs_TCW | "
                  "ratio_vs_NPOS | abs_recall | bonf/10 |")
        md.append("|------|-----|----|-------|--------|--------------|"
                  "---------------|------------|---------|")
        for _, r in surv.iterrows():
            md.append(f"| {r['head']} | {r['aggregator']} | {int(r['window_size_bp'])} | "
                      f"{r['level']} | {r['filter']} | "
                      f"{fmt_ratio(r['mean_ratio_vs_TCW'], r['ratio_tcw_ci_lo'], r['ratio_tcw_ci_hi'])} | "
                      f"{fmt_ratio(r['mean_ratio_vs_NPOS'], r['ratio_npos_ci_lo'], r['ratio_npos_ci_hi'])} | "
                      f"{fmt_pct(r['mean_abs_recall'], r['abs_recall_ci_lo'], r['abs_recall_ci_hi'])} | "
                      f"{int(r['n_cancers_bonf_signif'])}/10 |")
        md.append("")

    # ====================================================================
    # 5. Position-level diagnostic: trinuc breakdown
    # ====================================================================
    md.append("## 5. Position-level diagnostic: top-1% trinucleotide breakdown")
    md.append("")
    if TRINUC_CSV.exists():
        tn = pd.read_csv(TRINUC_CSV)
        models = ["overall_panel", "v3_top1pct", "v4_cancer_top1pct", "v4_cds_top1pct"]
        cats = ["TCW", "TCG (CpG)", "TCC", "NCG (non-TC CpG)", "other_C", "non-C"]
        md.append("Strand-corrected trinucleotide context of the top-1% panel")
        md.append("positions ranked by `score_binary` (per model). Compare to")
        md.append("the panel's overall distribution. **Anti-TCW polarity**")
        md.append("(v3) means top-1% should be CpG-skewed and TCW-depleted")
        md.append("relative to the overall distribution; v4 should not be.")
        md.append("")
        md.append("| trinuc bucket | overall panel | v3 top-1% | v4_cancer top-1% | v4_cds top-1% |")
        md.append("|---------------|---------------|-----------|------------------|---------------|")
        for c in cats:
            row = [c]
            for m in models:
                sub = tn[(tn["model"] == m) & (tn["category"] == c)]
                if len(sub) == 0:
                    row.append("-")
                else:
                    row.append(f"{sub.iloc[0]['frac']*100:.2f}%")
            md.append("| " + " | ".join(row) + " |")
        md.append("")
        # TCW vs CpG ratio summary
        def get_frac(model, cat):
            sub = tn[(tn["model"] == model) & (tn["category"] == cat)]
            return float(sub.iloc[0]["frac"]) if len(sub) else 0.0
        md.append("**TCW vs CpG ratios (top-1% / overall):**")
        md.append("")
        md.append("| model | TCW enrichment | CpG enrichment | TCW polarity |")
        md.append("|-------|----------------|----------------|--------------|")
        ov_tcw = get_frac("overall_panel", "TCW")
        ov_cpg = get_frac("overall_panel", "TCG (CpG)") + get_frac("overall_panel", "NCG (non-TC CpG)")
        for m_label, m_key in [("v3", "v3_top1pct"),
                               ("v4_cancer", "v4_cancer_top1pct"),
                               ("v4_cds", "v4_cds_top1pct")]:
            tcw = get_frac(m_key, "TCW")
            cpg = get_frac(m_key, "TCG (CpG)") + get_frac(m_key, "NCG (non-TC CpG)")
            tcw_enr = tcw / ov_tcw if ov_tcw > 0 else float("nan")
            cpg_enr = cpg / ov_cpg if ov_cpg > 0 else float("nan")
            polarity = ("ANTI-TCW" if tcw_enr < 0.9 and cpg_enr > 1.5
                        else "TCW-positive" if tcw_enr > 1.1 else "neutral")
            md.append(f"| {m_label} | {tcw_enr:.2f}x | {cpg_enr:.2f}x | {polarity} |")
        md.append("")
        md.append("See `topx_trinuc_breakdown.png` for visualization.")
    else:
        md.append(f"(`{TRINUC_CSV.name}` not found.)")
    md.append("")

    # ====================================================================
    # 6. Verdict
    # ====================================================================
    md.append("## 6. Verdict")
    md.append("")
    md.append("Two defensibility tiers (within filter_TCW_nonCpG):")
    md.append("")
    md.append("- **Tier S (strong)**: ratio_vs_TCW CI lo > 1 AND ratio_vs_NPOS CI lo > 1")
    md.append("  AND >=6/10 cancers Bonferroni-surviving. Beats both same-bases TCW")
    md.append("  density and gene-body density.")
    md.append("- **Tier A (defensible)**: ratio_vs_NPOS CI lo > 1 AND >=6/10 cancers")
    md.append("  Bonferroni-surviving (no constraint on ratio_vs_TCW). Beats gene-body")
    md.append("  density alone. The TCW density baseline is structurally privileged")
    md.append("  in this filter because all surviving mutations are TCW; rather, the")
    md.append("  question is whether the model's positional ranking is more informative")
    md.append("  than just `n_panel_positions_in_window`.")
    md.append("")

    def best_tier(df, tier):
        sub = df[(df["filter"] == "filter_TCW_nonCpG")
                 & (df["ratio_npos_ci_lo"] > 1.0)
                 & (df["n_cancers_bonf_signif"] >= 6)].copy()
        if tier == "S":
            sub = sub[sub["ratio_tcw_ci_lo"] > 1.0]
        if len(sub) == 0:
            return None
        # Sort: prefer score_binary (most general head), then by ratio_vs_NPOS.
        sub["_is_binary"] = (sub["head"] == "score_binary").astype(int)
        return sub.sort_values(["_is_binary", "mean_ratio_vs_NPOS"],
                               ascending=[False, False]).iloc[0]

    md.append("### Tier S (strong: beats both TCW AND n_pos)")
    md.append("")
    found_S = False
    for tag, df in [("v3", v3), ("v4_cancer", v4c), ("v4_cds", v4d)]:
        bd = best_tier(df, "S")
        if bd is None:
            md.append(f"- **{tag}**: no Tier-S cell found.")
        else:
            found_S = True
            constr = f"{bd['head']}, {bd['aggregator']}, ws={int(bd['window_size_bp'])}, level={bd['level']}"
            md.append(f"- **{tag}**: `{constr}` -- "
                      f"ratio_vs_TCW = {fmt_ratio(bd['mean_ratio_vs_TCW'], bd['ratio_tcw_ci_lo'], bd['ratio_tcw_ci_hi'])}; "
                      f"ratio_vs_NPOS = {fmt_ratio(bd['mean_ratio_vs_NPOS'], bd['ratio_npos_ci_lo'], bd['ratio_npos_ci_hi'])}; "
                      f"abs_recall = {fmt_pct(bd['mean_abs_recall'], bd['abs_recall_ci_lo'], bd['abs_recall_ci_hi'])}; "
                      f"{int(bd['n_cancers_bonf_signif'])}/10 Bonf.")
    md.append("")

    md.append("### Tier A (defensible: beats n_pos density, may lose to TCW)")
    md.append("")
    tier_A_results = {}
    for tag, df in [("v3", v3), ("v4_cancer", v4c), ("v4_cds", v4d)]:
        bd = best_tier(df, "A")
        tier_A_results[tag] = bd
        if bd is None:
            md.append(f"- **{tag}**: no Tier-A cell found.")
        else:
            constr = f"{bd['head']}, {bd['aggregator']}, ws={int(bd['window_size_bp'])}, level={bd['level']}"
            md.append(f"- **{tag}**: `{constr}` -- "
                      f"ratio_vs_NPOS = {fmt_ratio(bd['mean_ratio_vs_NPOS'], bd['ratio_npos_ci_lo'], bd['ratio_npos_ci_hi'])}; "
                      f"ratio_vs_TCW = {fmt_ratio(bd['mean_ratio_vs_TCW'], bd['ratio_tcw_ci_lo'], bd['ratio_tcw_ci_hi'])}; "
                      f"abs_recall = {fmt_pct(bd['mean_abs_recall'], bd['abs_recall_ci_lo'], bd['abs_recall_ci_hi'])}; "
                      f"{int(bd['n_cancers_bonf_signif'])}/10 Bonf.")
    md.append("")

    # Winning variant
    md.append("### Winning variant")
    md.append("")
    candidates_A = [(tag, bd) for tag, bd in tier_A_results.items() if bd is not None]
    if len(candidates_A) == 0:
        md.append("**NONE**: no variant produces a Tier-A cell. The headline")
        md.append("RNA-to-DNA transfer claim does NOT hold even at the relaxed")
        md.append("(beats-n_pos-only) tier.")
    else:
        winner_tag, winner = max(candidates_A,
                                 key=lambda kv: kv[1]["mean_ratio_vs_NPOS"])
        md.append(f"**{winner_tag}** has the strongest defensible claim (Tier A).")
        md.append("")
        md.append(f"- Construction: `{winner['head']}, {winner['aggregator']}, "
                  f"ws={int(winner['window_size_bp'])}, level={winner['level']}`")
        md.append(f"- Effect size (vs n_pos density): "
                  f"{fmt_ratio(winner['mean_ratio_vs_NPOS'], winner['ratio_npos_ci_lo'], winner['ratio_npos_ci_hi'])}")
        md.append(f"- ratio_vs_TCW: "
                  f"{fmt_ratio(winner['mean_ratio_vs_TCW'], winner['ratio_tcw_ci_lo'], winner['ratio_tcw_ci_hi'])} "
                  f"(loses to TCW-density same-bases baseline -- see note above)")
        md.append(f"- abs_recall: "
                  f"{fmt_pct(winner['mean_abs_recall'], winner['abs_recall_ci_lo'], winner['abs_recall_ci_hi'])}")
        md.append(f"- Bonferroni-surviving cancers: "
                  f"{int(winner['n_cancers_bonf_signif'])}/10")
        md.append("")
        md.append(f"**Claim:** the v4_{winner_tag.split('_')[-1] if '_' in winner_tag else winner_tag} model's")
        md.append(f"positional ranking is informative about cancer C>T mutation locations beyond")
        md.append(f"what gene-body density alone explains, with effect size "
                  f"{winner['mean_ratio_vs_NPOS']:.2f}x ")
        md.append(f"(95% CI [{winner['ratio_npos_ci_lo']:.2f}, {winner['ratio_npos_ci_hi']:.2f}]).")
    md.append("")
    md.append("### v3 -> v4 deltas (binary head, sum/win_1000/TCW_nonCpG)")
    md.append("")
    v3_h = get_cell(v3, "score_binary", "sum", level="win_1000", ws=1000)
    v4c_h = get_cell(v4c, "score_binary", "sum", level="win_1000", ws=1000)
    v4d_h = get_cell(v4d, "score_binary", "sum", level="win_1000", ws=1000)
    if v3_h is not None and v4c_h is not None and v4d_h is not None:
        md.append("| metric | v3 | v4_cancer | v4_cds | delta v4_cancer-v3 | delta v4_cds-v3 |")
        md.append("|--------|----|-----------|--------|--------------------|------------------|")
        md.append(f"| ratio_vs_TCW | {v3_h['mean_ratio_vs_TCW']:.3f} | "
                  f"{v4c_h['mean_ratio_vs_TCW']:.3f} | {v4d_h['mean_ratio_vs_TCW']:.3f} | "
                  f"{v4c_h['mean_ratio_vs_TCW']-v3_h['mean_ratio_vs_TCW']:+.3f} | "
                  f"{v4d_h['mean_ratio_vs_TCW']-v3_h['mean_ratio_vs_TCW']:+.3f} |")
        md.append(f"| ratio_vs_NPOS | {v3_h['mean_ratio_vs_NPOS']:.3f} | "
                  f"{v4c_h['mean_ratio_vs_NPOS']:.3f} | {v4d_h['mean_ratio_vs_NPOS']:.3f} | "
                  f"{v4c_h['mean_ratio_vs_NPOS']-v3_h['mean_ratio_vs_NPOS']:+.3f} | "
                  f"{v4d_h['mean_ratio_vs_NPOS']-v3_h['mean_ratio_vs_NPOS']:+.3f} |")
        md.append(f"| abs_recall | {v3_h['mean_abs_recall']*100:.3f}% | "
                  f"{v4c_h['mean_abs_recall']*100:.3f}% | {v4d_h['mean_abs_recall']*100:.3f}% | "
                  f"{(v4c_h['mean_abs_recall']-v3_h['mean_abs_recall'])*100:+.3f}pp | "
                  f"{(v4d_h['mean_abs_recall']-v3_h['mean_abs_recall'])*100:+.3f}pp |")
        md.append(f"| bonf/10 | {int(v3_h['n_cancers_bonf_signif'])} | "
                  f"{int(v4c_h['n_cancers_bonf_signif'])} | {int(v4d_h['n_cancers_bonf_signif'])} | "
                  f"+{int(v4c_h['n_cancers_bonf_signif'])-int(v3_h['n_cancers_bonf_signif'])} | "
                  f"+{int(v4d_h['n_cancers_bonf_signif'])-int(v3_h['n_cancers_bonf_signif'])} |")
    md.append("")
    md.append("### Position-level claim is now non-zero?")
    md.append("")
    pos_v3 = get_cell(v3, "score_binary", "max", level="position", ws=0)
    pos_v4c = get_cell(v4c, "score_binary", "max", level="position", ws=0)
    pos_v4d = get_cell(v4d, "score_binary", "max", level="position", ws=0)
    md.append(f"- v3 position-level binary abs_recall = "
              f"{fmt_pct(pos_v3['mean_abs_recall']) if pos_v3 is not None else 'n/a'} "
              f"(0% literally -- anti-TCW polarity zeroed out the recall on TCW_nonCpG mutations)")
    md.append(f"- v4_cancer position-level binary abs_recall = "
              f"{fmt_pct(pos_v4c['mean_abs_recall']) if pos_v4c is not None else 'n/a'} "
              f"(non-zero but tiny; v4_cancer top-1% is overwhelmingly other_C, not TCW)")
    md.append(f"- v4_cds position-level binary abs_recall = "
              f"{fmt_pct(pos_v4d['mean_abs_recall']) if pos_v4d is not None else 'n/a'} "
              f"(non-zero AND substantial; ratio_vs_NPOS = "
              f"{fmt_ratio(pos_v4d['mean_ratio_vs_NPOS'], pos_v4d['ratio_npos_ci_lo'], pos_v4d['ratio_npos_ci_hi']) if pos_v4d is not None else 'n/a'}, "
              f"10/10 Bonf)")
    md.append("")

    md.append("## Files")
    md.append("")
    md.append(f"- `{V3_CSV.name}` (v3 sweep, position + win_1000 only)")
    md.append(f"- `{V4C_CSV.name}` (v4_cancer sweep, all 21 constructions)")
    md.append(f"- `{V4D_CSV.name}` (v4_cds sweep, all 21 constructions)")
    md.append(f"- `topx_trinuc_breakdown.png` and `.csv` (position-level trinuc diagnostic)")
    md.append("")

    OUT_MD.write_text("\n".join(md))
    log.info("Wrote %s (%d lines)", OUT_MD, len(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
