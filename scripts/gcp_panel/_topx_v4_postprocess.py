#!/usr/bin/env python3
"""Post-process topx threshold sweep CSVs into PNG figure + Markdown summary.

Reads:
  topx_threshold_sweep_v4_cds.csv
  topx_threshold_sweep_v4_cancer.csv

Writes:
  topx_threshold_curves.png
  TOPX_THRESHOLD_RESULTS.md
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path("/Users/shaharharel/Documents/github/edit-rna-apobec/"
               "experiments/multi_enzyme/outputs/pcawg_tcw_panel/v4_outputs")

CDS_CSV = OUT_DIR / "topx_threshold_sweep_v4_cds.csv"
CANCER_CSV = OUT_DIR / "topx_threshold_sweep_v4_cancer.csv"

OUT_PNG = OUT_DIR / "topx_threshold_curves.png"
OUT_MD = OUT_DIR / "TOPX_THRESHOLD_RESULTS.md"


def fmt_pct(x: float) -> str:
    if pd.isna(x):
        return "NA"
    return f"{x*100:.2f}%"


def fmt_pct_ci(m, lo, hi) -> str:
    if pd.isna(m):
        return "NA"
    return f"{m*100:.2f}% [{lo*100:.2f}, {hi*100:.2f}]"


def fmt_num_ci(m, lo, hi) -> str:
    if pd.isna(m):
        return "NA"
    return f"{m:.3f} [{lo:.3f}, {hi:.3f}]"


def load_results():
    if not CDS_CSV.exists():
        raise FileNotFoundError(f"Missing {CDS_CSV}")
    if not CANCER_CSV.exists():
        raise FileNotFoundError(f"Missing {CANCER_CSV}")
    cds = pd.read_csv(CDS_CSV)
    can = pd.read_csv(CANCER_CSV)
    cds["panel"] = "v4_cds"
    can["panel"] = "v4_cancer"
    return cds, can


def make_figure(cds: pd.DataFrame, can: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.concat([cds, can], ignore_index=True)

    # Faceted: rows = (cut_type, filter), cols = head; lines for cds + cancer
    cut_types = ["top_pct", "pscore"]
    filters = ["filter_TCW_nonCpG", "filter_all_CT"]
    heads = ["score_binary", "score_apobec1_v3"]

    levels = ["position", "window_max_w1000"]

    n_rows = len(cut_types) * len(filters) * len(levels)
    n_cols = len(heads)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.5 * n_cols, 3.2 * n_rows),
                             squeeze=False)

    row_i = 0
    for level in levels:
        for cut_type in cut_types:
            for fname in filters:
                for col_i, head in enumerate(heads):
                    ax = axes[row_i][col_i]
                    sub = df[(df["level"] == level) & (df["cut_type"] == cut_type)
                             & (df["filter"] == fname) & (df["head"] == head)]
                    for panel_name, color in [("v4_cds", "tab:blue"),
                                              ("v4_cancer", "tab:orange")]:
                        ss = sub[sub["panel"] == panel_name].sort_values(
                            "panel_coverage_Mb")
                        if len(ss) == 0:
                            continue
                        ax.errorbar(ss["panel_coverage_Mb"],
                                    ss["mean_abs_recall"] * 100,
                                    yerr=[(ss["mean_abs_recall"]
                                           - ss["abs_recall_ci_lo"]) * 100,
                                          (ss["abs_recall_ci_hi"]
                                           - ss["mean_abs_recall"]) * 100],
                                    fmt="o-", color=color,
                                    label=f"{panel_name} NN", capsize=3)
                        # baselines: TCW-density (use abs_recall_tcw via per-cancer,
                        # but we have ratio. abs_recall_tcw = abs_recall / ratio_tcw)
                        # We'll show NPOS-derived baseline as horizontal estimate by
                        # plotting abs_recall / ratio_vs_NPOS (= NPOS recall) and
                        # abs_recall / ratio_vs_TCW (= TCW recall) at each cut.
                        tcw_recall = (ss["mean_abs_recall"]
                                      / ss["mean_ratio_vs_TCW"])
                        npos_recall = (ss["mean_abs_recall"]
                                       / ss["mean_ratio_vs_NPOS"])
                        ax.plot(ss["panel_coverage_Mb"], tcw_recall * 100,
                                "x--", color=color, alpha=0.5,
                                label=f"{panel_name} TCW-density")
                        ax.plot(ss["panel_coverage_Mb"], npos_recall * 100,
                                "+--", color=color, alpha=0.4,
                                label=f"{panel_name} N-POS")
                    ax.set_xscale("log")
                    ax.set_xlabel("panel coverage (Mb, log)")
                    ax.set_ylabel("abs recall (%)")
                    ax.set_title(f"{level} / {cut_type} / {fname} / {head}",
                                 fontsize=9)
                    ax.legend(fontsize=6, loc="best")
                    ax.grid(alpha=0.3)
                row_i += 1

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Wrote {OUT_PNG}")


def write_md(cds: pd.DataFrame, can: pd.DataFrame):
    md = []
    md.append("# TopX + Threshold Sweep Results — v4 panel")
    md.append("")
    md.append("Heads: `score_binary`, `score_apobec1_v3`. ")
    md.append("Filters: `filter_TCW_nonCpG`, `filter_all_CT`. ")
    md.append("Levels: position, window_max_w1000. ")
    md.append("Cuts: top-1%/5%/10%, P75/P90/P95/P99 of head score. ")
    md.append("Bonferroni q = 0.05 / (2 levels x 2 heads x 7 cuts x 2 filters x 10 cancers) = "
              f"{0.05/(2*2*7*2*10):.2e}.")
    md.append("")

    # ===== Section 1: panel size at each cut (binary head) =====
    md.append("## 1. Panel size at each cut (`score_binary`, position-level)")
    md.append("")
    md.append("| panel | cut_type | cut | panel_units | panel_coverage_Mb |")
    md.append("|-------|----------|-----|-------------|-------------------|")
    for panel_df, panel_name in [(cds, "v4_cds"), (can, "v4_cancer")]:
        sub = panel_df[(panel_df["head"] == "score_binary")
                       & (panel_df["level"] == "position")
                       & (panel_df["filter"] == "filter_TCW_nonCpG")]
        sub = sub.sort_values(["cut_type", "cut_value"])
        for _, r in sub.iterrows():
            md.append(f"| {panel_name} | {r['cut_type']} | "
                      f"{r['cut_value']:.4f} | {int(r['panel_units']):,} | "
                      f"{r['panel_coverage_Mb']:.3f} |")
    md.append("")

    md.append("## 1b. Panel size at each cut (`score_binary`, window_max_w1000)")
    md.append("")
    md.append("| panel | cut_type | cut | panel_units (windows) | panel_coverage_Mb |")
    md.append("|-------|----------|-----|------------------------|-------------------|")
    for panel_df, panel_name in [(cds, "v4_cds"), (can, "v4_cancer")]:
        sub = panel_df[(panel_df["head"] == "score_binary")
                       & (panel_df["level"] == "window_max_w1000")
                       & (panel_df["filter"] == "filter_TCW_nonCpG")]
        sub = sub.sort_values(["cut_type", "cut_value"])
        for _, r in sub.iterrows():
            md.append(f"| {panel_name} | {r['cut_type']} | "
                      f"{r['cut_value']:.4f} | {int(r['panel_units']):,} | "
                      f"{r['panel_coverage_Mb']:.3f} |")
    md.append("")

    # ===== Section 2: Recall + ratios per cut, position-level, both filters =====
    md.append("## 2. Recall and ratios at each cut (position-level)")
    md.append("")
    for filter_name in ["filter_TCW_nonCpG", "filter_all_CT"]:
        md.append(f"### Filter: `{filter_name}`")
        md.append("")
        md.append("| panel | head | cut | panel_Mb | abs_recall | "
                  "ratio_vs_TCW | ratio_vs_NPOS | bonf/10 |")
        md.append("|-------|------|-----|----------|------------|"
                  "--------------|---------------|---------|")
        for panel_df, panel_name in [(cds, "v4_cds"), (can, "v4_cancer")]:
            sub = panel_df[(panel_df["level"] == "position")
                           & (panel_df["filter"] == filter_name)]
            sub = sub.sort_values(["head", "cut_type", "cut_value"])
            for _, r in sub.iterrows():
                cut_label = (f"{r['cut_type']}={r['cut_value']:.3f}")
                md.append(
                    f"| {panel_name} | {r['head'].replace('score_','')} | "
                    f"{cut_label} | {r['panel_coverage_Mb']:.2f} | "
                    f"{fmt_pct_ci(r['mean_abs_recall'], r['abs_recall_ci_lo'], r['abs_recall_ci_hi'])} | "
                    f"{fmt_num_ci(r['mean_ratio_vs_TCW'], r['ratio_tcw_ci_lo'], r['ratio_tcw_ci_hi'])} | "
                    f"{fmt_num_ci(r['mean_ratio_vs_NPOS'], r['ratio_npos_ci_lo'], r['ratio_npos_ci_hi'])} | "
                    f"{int(r['n_cancers_bonf_signif'])}/10 |"
                )
        md.append("")

    # ===== Section 2b: Window-level summary (max, w=1000) =====
    md.append("## 2b. Recall and ratios at each cut (window_max_w1000)")
    md.append("")
    for filter_name in ["filter_TCW_nonCpG", "filter_all_CT"]:
        md.append(f"### Filter: `{filter_name}`")
        md.append("")
        md.append("| panel | head | cut | panel_Mb | abs_recall | "
                  "ratio_vs_TCW | ratio_vs_NPOS | bonf/10 |")
        md.append("|-------|------|-----|----------|------------|"
                  "--------------|---------------|---------|")
        for panel_df, panel_name in [(cds, "v4_cds"), (can, "v4_cancer")]:
            sub = panel_df[(panel_df["level"] == "window_max_w1000")
                           & (panel_df["filter"] == filter_name)]
            sub = sub.sort_values(["head", "cut_type", "cut_value"])
            for _, r in sub.iterrows():
                cut_label = (f"{r['cut_type']}={r['cut_value']:.3f}")
                md.append(
                    f"| {panel_name} | {r['head'].replace('score_','')} | "
                    f"{cut_label} | {r['panel_coverage_Mb']:.2f} | "
                    f"{fmt_pct_ci(r['mean_abs_recall'], r['abs_recall_ci_lo'], r['abs_recall_ci_hi'])} | "
                    f"{fmt_num_ci(r['mean_ratio_vs_TCW'], r['ratio_tcw_ci_lo'], r['ratio_tcw_ci_hi'])} | "
                    f"{fmt_num_ci(r['mean_ratio_vs_NPOS'], r['ratio_npos_ci_lo'], r['ratio_npos_ci_hi'])} | "
                    f"{int(r['n_cancers_bonf_signif'])}/10 |"
                )
        md.append("")

    # ===== Section 3: Q&A =====
    md.append("## 3. Q&A")
    md.append("")

    # Q3a: 10% recall target on filter_all_CT
    md.append("### a) Where does the 10% recall target hit?")
    md.append("")
    md.append("Smallest cut (by panel_coverage_Mb) with `abs_recall >= 10%` on "
              "`filter_all_CT`, position-level, head=score_binary.")
    md.append("")
    md.append("| panel | smallest cut achieving >=10% recall | panel_Mb | abs_recall |")
    md.append("|-------|--------------------------------------|----------|------------|")
    for panel_df, panel_name in [(cds, "v4_cds"), (can, "v4_cancer")]:
        sub = panel_df[(panel_df["level"] == "position")
                       & (panel_df["filter"] == "filter_all_CT")
                       & (panel_df["head"] == "score_binary")
                       & (panel_df["mean_abs_recall"] >= 0.10)]
        if len(sub) == 0:
            md.append(f"| {panel_name} | NOT REACHED at any tested cut | - | - |")
        else:
            r = sub.sort_values("panel_coverage_Mb").iloc[0]
            md.append(f"| {panel_name} | {r['cut_type']}={r['cut_value']:.3f} | "
                      f"{r['panel_coverage_Mb']:.2f} | "
                      f"{fmt_pct(r['mean_abs_recall'])} |")
    md.append("")
    # Also report 30% target if any
    md.append("Smallest cut with `abs_recall >= 30%`:")
    md.append("")
    md.append("| panel | smallest cut achieving >=30% recall | panel_Mb | abs_recall |")
    md.append("|-------|--------------------------------------|----------|------------|")
    for panel_df, panel_name in [(cds, "v4_cds"), (can, "v4_cancer")]:
        sub = panel_df[(panel_df["level"] == "position")
                       & (panel_df["filter"] == "filter_all_CT")
                       & (panel_df["head"] == "score_binary")
                       & (panel_df["mean_abs_recall"] >= 0.30)]
        if len(sub) == 0:
            md.append(f"| {panel_name} | NOT REACHED at any tested cut | - | - |")
        else:
            r = sub.sort_values("panel_coverage_Mb").iloc[0]
            md.append(f"| {panel_name} | {r['cut_type']}={r['cut_value']:.3f} | "
                      f"{r['panel_coverage_Mb']:.2f} | "
                      f"{fmt_pct(r['mean_abs_recall'])} |")
    md.append("")

    # Q3b: P-cuts vs top-X
    md.append("### b) Are P-cuts more informative than top-X%?")
    md.append("")
    md.append("Comparison of panel size variance across cancers is moot here "
              "(panel scoring is global, not per-cancer). But P-cuts produce a "
              "fixed score threshold while top-X% produces a fixed panel size. "
              "We report both:")
    md.append("")
    md.append("| panel | head | cut_type | mean panel_Mb | range abs_recall (TCW_nonCpG, position) |")
    md.append("|-------|------|----------|---------------|-----------------------------------------|")
    for panel_df, panel_name in [(cds, "v4_cds"), (can, "v4_cancer")]:
        for head in ["score_binary", "score_apobec1_v3"]:
            for ct in ["top_pct", "pscore"]:
                sub = panel_df[(panel_df["level"] == "position")
                               & (panel_df["filter"] == "filter_TCW_nonCpG")
                               & (panel_df["head"] == head)
                               & (panel_df["cut_type"] == ct)]
                if len(sub) == 0:
                    continue
                rmin = sub["mean_abs_recall"].min() * 100
                rmax = sub["mean_abs_recall"].max() * 100
                mean_mb = sub["panel_coverage_Mb"].mean()
                md.append(f"| {panel_name} | {head.replace('score_','')} | "
                          f"{ct} | {mean_mb:.2f} | "
                          f"{rmin:.2f}%–{rmax:.2f}% |")
    md.append("")

    # Q3c: ratio_vs_NPOS shrinkage
    md.append("### c) Does ratio_vs_NPOS shrink as panel grows?")
    md.append("")
    md.append("Position-level, filter_TCW_nonCpG, score_binary:")
    md.append("")
    md.append("| panel | cut_type | cut | panel_Mb | ratio_vs_NPOS |")
    md.append("|-------|----------|-----|----------|----------------|")
    for panel_df, panel_name in [(cds, "v4_cds"), (can, "v4_cancer")]:
        sub = panel_df[(panel_df["level"] == "position")
                       & (panel_df["filter"] == "filter_TCW_nonCpG")
                       & (panel_df["head"] == "score_binary")]
        sub = sub.sort_values("panel_coverage_Mb")
        for _, r in sub.iterrows():
            md.append(f"| {panel_name} | {r['cut_type']} | "
                      f"{r['cut_value']:.3f} | {r['panel_coverage_Mb']:.2f} | "
                      f"{fmt_num_ci(r['mean_ratio_vs_NPOS'], r['ratio_npos_ci_lo'], r['ratio_npos_ci_hi'])} |")
    md.append("")

    # Q3d: NN vs TCW gap at top-5% / top-10%
    md.append("### d) Does NN vs TCW-density gap hold at larger panels?")
    md.append("")
    md.append("Position-level, filter_TCW_nonCpG, score_binary; ratio_vs_TCW > 1 "
              "means NN beats TCW-density at the same panel size:")
    md.append("")
    md.append("| panel | cut | ratio_vs_TCW (CI) | n_cancers above TCW |")
    md.append("|-------|-----|-------------------|---------------------|")
    for panel_df, panel_name in [(cds, "v4_cds"), (can, "v4_cancer")]:
        sub = panel_df[(panel_df["level"] == "position")
                       & (panel_df["filter"] == "filter_TCW_nonCpG")
                       & (panel_df["head"] == "score_binary")
                       & (panel_df["cut_type"] == "top_pct")]
        sub = sub.sort_values("cut_value")
        for _, r in sub.iterrows():
            md.append(f"| {panel_name} | top_{r['cut_value']:.2f} | "
                      f"{fmt_num_ci(r['mean_ratio_vs_TCW'], r['ratio_tcw_ci_lo'], r['ratio_tcw_ci_hi'])} | "
                      f"{int(r['n_cancers_above_TCW'])}/10 |")
    md.append("")

    md.append("## 4. Files")
    md.append("")
    md.append(f"- `topx_threshold_sweep_v4_cds.csv`")
    md.append(f"- `topx_threshold_sweep_v4_cancer.csv`")
    md.append(f"- `topx_threshold_curves.png`")

    OUT_MD.write_text("\n".join(md))
    print(f"Wrote {OUT_MD}")


def main():
    cds, can = load_results()
    print(f"v4_cds rows: {len(cds)}, v4_cancer rows: {len(can)}")
    make_figure(cds, can)
    write_md(cds, can)


if __name__ == "__main__":
    main()
