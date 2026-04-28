#!/usr/bin/env python3
"""Post-process the all-heads topx sweep on v4_cds apobec1retrained panel.

Reads:
  topx_sweep_v4_cds_all_heads.csv

Writes:
  topx_sweep_v4_cds_all_heads.png
  PER_ENZYME_HEAD_RESULTS.md

Also computes (from the source parquet directly):
  - Jaccard index between top-1% sets of each head pair
  - Ensemble panels:
      * binary union apobec1_v4_cds at top-X%
      * union of all 6 heads at top-X%
    Then re-evaluates recall on the same TCGA+PCAWG combined coding MAFs
    (filter_all_CT and filter_TCW_nonCpG, position-level).

Inputs:
  experiments/multi_enzyme/outputs/pcawg_tcw_panel/v4_outputs/
    panel_scores_v4_cds_apobec1retrained.parquet
    topx_sweep_v4_cds_all_heads.csv
"""
from __future__ import annotations
import logging
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/shaharharel/Documents/github/edit-rna-apobec")
TCGA_DIR = ROOT / "data/raw/tcga"
PCAWG_DIR = ROOT / "data/raw/pcawg/by_cancer"
HG19 = ROOT / "data/raw/genomes/hg19.fa"
OUT_DIR = ROOT / "experiments/multi_enzyme/outputs/pcawg_tcw_panel/v4_outputs"
PANEL = OUT_DIR / "panel_scores_v4_cds_apobec1retrained.parquet"
CSV = OUT_DIR / "topx_sweep_v4_cds_all_heads.csv"
PNG = OUT_DIR / "topx_sweep_v4_cds_all_heads.png"
MD = OUT_DIR / "PER_ENZYME_HEAD_RESULTS.md"

CANCERS = ["blca", "brca", "cesc", "coadread", "esca", "hnsc", "lihc",
           "lusc", "skcm", "stad"]
HEADS = ["score_binary", "score_A3A", "score_A3B", "score_A3G",
         "score_A3A_A3G", "score_apobec1_v4_cds"]
HEAD_LABELS = {h: h.replace("score_", "") for h in HEADS}
TOP_PCTS = [0.01, 0.05, 0.10]
N_BOOT = 5000

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    stream=sys.stdout)
log = logging.getLogger(__name__)


def fmt_pct(x):
    if pd.isna(x):
        return "NA"
    return f"{x * 100:.2f}%"


def fmt_pct_ci(m, lo, hi):
    if pd.isna(m):
        return "NA"
    return f"{m * 100:.2f}% [{lo * 100:.2f}, {hi * 100:.2f}]"


def fmt_num_ci(m, lo, hi):
    if pd.isna(m):
        return "NA"
    return f"{m:.3f} [{lo:.3f}, {hi:.3f}]"


# --------------------------------------------------------------------- #
# MAF loading - mirrors compute_panel_recall_topx_v4.py
# --------------------------------------------------------------------- #

def _load_one_maf(path: Path, cancer: str, source: str):
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, sep="\t", low_memory=False)
    except Exception:
        return None
    if "Chromosome" not in df.columns or "Start_Position" not in df.columns:
        return None
    df = df[df.get("Variant_Type", "SNP") == "SNP"]
    ref = df.get("Reference_Allele")
    alt = df.get("Tumor_Seq_Allele2", df.get("Tumor_Seq_Allele"))
    if ref is None or alt is None:
        return None
    is_CT = (ref == "C") & (alt == "T")
    is_GA = (ref == "G") & (alt == "A")
    df = df[is_CT | is_GA].copy()
    df["strand"] = np.where(
        (df["Reference_Allele"] == "C") & (df["Tumor_Seq_Allele2"] == "T"),
        "+", "-")
    df["pos"] = df["Start_Position"].astype(int) - 1
    df["chrom"] = df["Chromosome"].astype(str)
    df.loc[~df["chrom"].str.startswith("chr"), "chrom"] = "chr" + df["chrom"]
    df["cancer"] = cancer
    df["source"] = source
    return df[["chrom", "pos", "strand", "cancer", "source"]]


def load_combined_maf():
    rows = []
    for cancer in CANCERS:
        d = _load_one_maf(PCAWG_DIR / f"{cancer}_pcawg_mutations.txt",
                          cancer, "pcawg")
        if d is not None:
            rows.append(d)
        d = _load_one_maf(TCGA_DIR / f"{cancer}_tcga_pan_can_atlas_2018_mutations.txt",
                          cancer, "tcga")
        if d is not None:
            rows.append(d)
    combined = pd.concat(rows, ignore_index=True)
    valid = set([f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"])
    return combined[combined["chrom"].isin(valid)].reset_index(drop=True)


def annotate_panel_tcw(panel: pd.DataFrame) -> pd.DataFrame:
    from pyfaidx import Fasta
    g = Fasta(str(HG19), as_raw=False, sequence_always_upper=True)
    n = len(panel)
    is_cpg = np.zeros(n, dtype=bool)
    is_tcw_c = np.zeros(n, dtype=bool)
    chroms = panel["chrom"].to_numpy()
    poses = panel["pos"].astype(int).to_numpy()
    strands = panel["strand"].to_numpy()
    idx_all = np.arange(n)
    for ch in pd.Series(chroms).unique():
        mask = chroms == ch
        idx = idx_all[mask]
        if len(idx) == 0:
            continue
        try:
            seq = np.frombuffer(str(g[ch][:]).upper().encode("ascii"),
                                dtype=np.uint8)
        except Exception:
            continue
        L = len(seq)
        ps = poses[mask]
        ss = strands[mask]
        ok = (ps >= 1) & (ps + 1 < L)
        v_idx = idx[ok]
        ps_ok = ps[ok]
        ss_ok = ss[ok]
        left = seq[ps_ok - 1]
        right = seq[ps_ok + 1]
        is_plus = ss_ok == "+"
        is_minus = ~is_plus
        is_cpg[v_idx[is_plus]] = right[is_plus] == ord("G")
        is_cpg[v_idx[is_minus]] = left[is_minus] == ord("C")
        right_AT = (right == ord("A")) | (right == ord("T"))
        left_AT = (left == ord("A")) | (left == ord("T"))
        is_tcw_c[v_idx[is_plus]] = (left[is_plus] == ord("T")) & right_AT[is_plus]
        is_tcw_c[v_idx[is_minus]] = (right[is_minus] == ord("A")) & left_AT[is_minus]
    out = panel.copy()
    out["is_cpg"] = is_cpg
    out["is_TCW_C"] = is_tcw_c
    return out


def annotate_maf_tcw(maf: pd.DataFrame) -> pd.DataFrame:
    from pyfaidx import Fasta
    g = Fasta(str(HG19), as_raw=False, sequence_always_upper=True)
    n = len(maf)
    is_tcw = np.zeros(n, dtype=bool)
    is_cpg = np.zeros(n, dtype=bool)
    chroms = maf["chrom"].to_numpy()
    poses = maf["pos"].astype(int).to_numpy()
    strands = maf["strand"].to_numpy()
    idx_all = np.arange(n)
    for ch in pd.Series(chroms).unique():
        mask = chroms == ch
        idx = idx_all[mask]
        if len(idx) == 0:
            continue
        try:
            seq = np.frombuffer(str(g[ch][:]).upper().encode("ascii"),
                                dtype=np.uint8)
        except Exception:
            continue
        L = len(seq)
        ps = poses[mask]
        ss = strands[mask]
        ok = (ps >= 1) & (ps + 1 < L)
        v_idx = idx[ok]
        ps_ok = ps[ok]
        ss_ok = ss[ok]
        left = seq[ps_ok - 1]
        center = seq[ps_ok]
        right = seq[ps_ok + 1]
        is_plus = ss_ok == "+"
        is_minus = ~is_plus
        plus_tcw = is_plus & (left == ord("T")) & (center == ord("C")) & (
            (right == ord("A")) | (right == ord("T")))
        minus_tcw = is_minus & (right == ord("A")) & (center == ord("G")) & (
            (left == ord("A")) | (left == ord("T")))
        plus_cpg = is_plus & (center == ord("C")) & (right == ord("G"))
        minus_cpg = is_minus & (center == ord("G")) & (left == ord("C"))
        is_tcw[v_idx] = plus_tcw | minus_tcw
        is_cpg[v_idx] = plus_cpg | minus_cpg
    out = maf.copy()
    out["is_TCW"] = is_tcw
    out["is_CpG"] = is_cpg
    out["is_TCW_nonCpG"] = is_tcw & ~is_cpg
    return out


# --------------------------------------------------------------------- #
# Per-cancer recall + boot CI given a binary panel mask
# --------------------------------------------------------------------- #

def panel_recall_per_cancer(panel_mask: np.ndarray,
                            mut_per_cancer: dict[str, np.ndarray]) -> dict:
    rec = {}
    abs_recalls = []
    for c, mut in mut_per_cancer.items():
        total = int(mut.sum())
        if total == 0:
            rec[c] = float("nan")
            continue
        r = float(mut[panel_mask].sum()) / total
        rec[c] = r
        abs_recalls.append(r)
    a = np.asarray(abs_recalls, dtype=float)
    if len(a) == 0:
        return {"per_cancer": rec, "mean": float("nan"),
                "lo": float("nan"), "hi": float("nan")}
    rng = np.random.default_rng(2026_04_27)
    n = len(a)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    boot = a[idx].mean(axis=1)
    return {"per_cancer": rec, "mean": float(a.mean()),
            "lo": float(np.percentile(boot, 2.5)),
            "hi": float(np.percentile(boot, 97.5)),
            "panel_size": int(panel_mask.sum())}


def baseline_npos_recall(n_units: int, k: int,
                         mut_per_cancer: dict[str, np.ndarray]) -> dict:
    """Random N-positions baseline: expected fraction = k/n_units (per cancer)."""
    frac = k / n_units
    abs_recalls = []
    for c, mut in mut_per_cancer.items():
        total = int(mut.sum())
        if total == 0:
            continue
        abs_recalls.append(frac)
    a = np.asarray(abs_recalls, dtype=float)
    return {"mean": float(a.mean()), "lo": float(a.mean()),
            "hi": float(a.mean()), "panel_size": k}


# --------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------- #

def make_figure(df: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = df[df["level"] == "position"].copy()
    filters = ["filter_TCW_nonCpG", "filter_all_CT"]
    n_rows = len(HEADS)
    n_cols = len(filters)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(7 * n_cols, 2.6 * n_rows),
                             squeeze=False)
    for r, head in enumerate(HEADS):
        for c, fn in enumerate(filters):
            ax = axes[r][c]
            sub = df[(df["head"] == head) & (df["filter"] == fn)]
            sub = sub.sort_values("panel_coverage_Mb")
            if len(sub) == 0:
                continue
            ax.errorbar(sub["panel_coverage_Mb"],
                        sub["mean_abs_recall"] * 100,
                        yerr=[(sub["mean_abs_recall"] - sub["abs_recall_ci_lo"]) * 100,
                              (sub["abs_recall_ci_hi"] - sub["mean_abs_recall"]) * 100],
                        fmt="o-", color="tab:blue",
                        label=f"{HEAD_LABELS[head]}", capsize=3)
            tcw_recall = sub["mean_abs_recall"] / sub["mean_ratio_vs_TCW"]
            npos_recall = sub["mean_abs_recall"] / sub["mean_ratio_vs_NPOS"]
            ax.plot(sub["panel_coverage_Mb"], tcw_recall * 100,
                    "x--", color="gray", alpha=0.7, label="TCW-density")
            ax.plot(sub["panel_coverage_Mb"], npos_recall * 100,
                    "+--", color="orange", alpha=0.7, label="N-positions (gene density)")
            ax.set_xscale("log")
            ax.set_xlabel("panel coverage (Mb, log)")
            ax.set_ylabel("abs recall (%)")
            ax.set_title(f"{HEAD_LABELS[head]} / {fn}", fontsize=9)
            ax.legend(fontsize=7, loc="best")
            ax.grid(alpha=0.3)
    fig.suptitle("v4_cds all heads — abs recall vs panel size (position-level)",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(PNG, dpi=130, bbox_inches="tight")
    plt.close()
    log.info("Wrote %s", PNG)


# --------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------- #

def main():
    log.info("Reading sweep CSV: %s", CSV)
    sweep = pd.read_csv(CSV)
    log.info("  %d rows", len(sweep))

    log.info("Reading panel parquet (this is the heavy step) ...")
    panel = pd.read_parquet(PANEL)
    log.info("  %d positions", len(panel))
    panel = annotate_panel_tcw(panel)

    log.info("Loading combined coding MAFs ...")
    maf = load_combined_maf()
    log.info("  %d C>T/G>A variants", len(maf))
    panel_set = set(zip(panel["chrom"].astype(str).values,
                        panel["pos"].astype(int).values))
    in_panel = np.array([(c, int(p)) in panel_set
                         for c, p in zip(maf["chrom"], maf["pos"])])
    maf = maf.iloc[np.where(in_panel)[0]].reset_index(drop=True)
    log.info("  in-panel variants: %d", len(maf))
    maf = annotate_maf_tcw(maf)

    # Build mut_per_cancer arrays at position level for both filters
    n = len(panel)
    panel_lookup = pd.DataFrame({"chrom": panel["chrom"].astype(str).values,
                                 "pos": panel["pos"].astype(int).values,
                                 "_uidx": np.arange(n)})

    def mut_arrays_for_filter(filtered_maf):
        m = filtered_maf[["chrom", "pos", "cancer"]].copy()
        m["pos"] = m["pos"].astype(int)
        m = m.merge(panel_lookup, on=["chrom", "pos"], how="inner")
        out = {}
        for cancer in CANCERS:
            sub = m[m["cancer"] == cancer]
            arr = np.zeros(n, dtype=np.int32)
            if len(sub) > 0:
                cnt = sub["_uidx"].value_counts()
                arr[cnt.index.astype(int).to_numpy()] = cnt.values.astype(np.int32)
            out[cancer] = arr
        return out

    mut_all_CT = mut_arrays_for_filter(maf)
    mut_TCW = mut_arrays_for_filter(maf[maf["is_TCW_nonCpG"]])

    # ---------------- Top-X% sets per head ----------------
    log.info("Building top-X%% index sets per head ...")
    top_sets = {}  # (head, pct) -> np.ndarray sorted indices
    for head in HEADS:
        scores = panel[head].to_numpy()
        for pct in TOP_PCTS:
            k = max(1, int(round(n * pct)))
            top_idx = np.argpartition(-scores, k - 1)[:k]
            top_sets[(head, pct)] = np.sort(top_idx)

    # ---------------- Jaccard at top-1% ----------------
    log.info("Computing Jaccard at top-1%% ...")
    jacc_rows = []
    for h1, h2 in combinations(HEADS, 2):
        s1 = top_sets[(h1, 0.01)]
        s2 = top_sets[(h2, 0.01)]
        inter = np.intersect1d(s1, s2, assume_unique=True)
        union = np.union1d(s1, s2)
        j = len(inter) / len(union) if len(union) > 0 else 0.0
        jacc_rows.append({"h1": HEAD_LABELS[h1], "h2": HEAD_LABELS[h2],
                          "size_h1": len(s1), "size_h2": len(s2),
                          "intersection": len(inter), "union": len(union),
                          "jaccard": j})
    jacc_df = pd.DataFrame(jacc_rows)
    log.info("Jaccard pairs:\n%s", jacc_df.to_string(index=False))

    # ---------------- Ensemble panels ----------------
    log.info("Building ensemble panels ...")
    ensemble_results = []  # list of dicts

    def eval_mask(label, mask, mut_dict):
        rec = panel_recall_per_cancer(mask, mut_dict)
        return {"label": label, "panel_size": int(mask.sum()),
                "panel_coverage_Mb": int(mask.sum()) * 1e-6,
                "mean": rec["mean"], "lo": rec["lo"], "hi": rec["hi"]}

    for pct in TOP_PCTS:
        # Single heads (for reference)
        for head in HEADS:
            mask = np.zeros(n, dtype=bool)
            mask[top_sets[(head, pct)]] = True
            for fn, md in [("filter_all_CT", mut_all_CT),
                           ("filter_TCW_nonCpG", mut_TCW)]:
                r = eval_mask(f"single:{HEAD_LABELS[head]}", mask, md)
                r.update({"ensemble_type": "single",
                          "head": HEAD_LABELS[head],
                          "pct": pct, "filter": fn})
                ensemble_results.append(r)
        # binary U apobec1_v4_cds
        b = top_sets[("score_binary", pct)]
        a1 = top_sets[("score_apobec1_v4_cds", pct)]
        union2 = np.union1d(b, a1)
        mask2 = np.zeros(n, dtype=bool); mask2[union2] = True
        for fn, md in [("filter_all_CT", mut_all_CT),
                       ("filter_TCW_nonCpG", mut_TCW)]:
            r = eval_mask("ensemble:binary_U_apobec1", mask2, md)
            r.update({"ensemble_type": "binary_U_apobec1",
                      "head": "binary_U_apobec1",
                      "pct": pct, "filter": fn})
            ensemble_results.append(r)
        # union of all 6
        all_idx = b
        for h in HEADS:
            all_idx = np.union1d(all_idx, top_sets[(h, pct)])
        mask6 = np.zeros(n, dtype=bool); mask6[all_idx] = True
        for fn, md in [("filter_all_CT", mut_all_CT),
                       ("filter_TCW_nonCpG", mut_TCW)]:
            r = eval_mask("ensemble:all6", mask6, md)
            r.update({"ensemble_type": "all6_union",
                      "head": "union_all_6",
                      "pct": pct, "filter": fn})
            ensemble_results.append(r)

    ens_df = pd.DataFrame(ensemble_results)

    # Best single head per pct/filter (for comparison vs ensemble)
    best_single_rows = []
    for pct in TOP_PCTS:
        for fn in ["filter_all_CT", "filter_TCW_nonCpG"]:
            sub = ens_df[(ens_df["ensemble_type"] == "single")
                         & (ens_df["pct"] == pct)
                         & (ens_df["filter"] == fn)]
            best = sub.loc[sub["mean"].idxmax()]
            best_single_rows.append({"pct": pct, "filter": fn,
                                     "best_head": best["head"],
                                     "best_recall": best["mean"]})

    # ---------------- Plot ----------------
    make_figure(sweep)

    # ---------------- Markdown ----------------
    md_lines = []
    md_lines.append("# Per-Enzyme Head Sweep — v4_cds APOBEC1retrained panel")
    md_lines.append("")
    md_lines.append(f"Panel: `{PANEL.name}` ({n:,} CDS positions).")
    md_lines.append(f"Heads: " + ", ".join(f"`{h}`" for h in HEADS))
    md_lines.append("Filters: `filter_TCW_nonCpG`, `filter_all_CT`. ")
    md_lines.append("Levels: position + window_max_w1000 (full sweep CSV); "
                    "summary tables and ensembles below are position-level.")
    md_lines.append("Cuts: top_pct in {0.01, 0.05, 0.10} + pscore P75/P90/P95/P99. "
                    "Permutation reps: 2000.")
    md_lines.append("")

    # 1. Comparison table at top-1%, top-5%, top-10%
    md_lines.append("## 1. Per-head recall + ratios (position-level)")
    md_lines.append("")
    for pct in TOP_PCTS:
        for fn in ["filter_all_CT", "filter_TCW_nonCpG"]:
            md_lines.append(f"### top_pct = {pct:.2f} | filter = `{fn}`")
            md_lines.append("")
            md_lines.append("| head | panel_Mb | abs_recall | ratio_vs_TCW | "
                            "ratio_vs_NPOS | bonf/10 |")
            md_lines.append("|------|----------|------------|--------------|"
                            "----------------|---------|")
            for head in HEADS:
                row = sweep[(sweep["level"] == "position")
                            & (sweep["filter"] == fn)
                            & (sweep["head"] == head)
                            & (sweep["cut_type"] == "top_pct")
                            & (np.isclose(sweep["cut_value"], pct))]
                if len(row) == 0:
                    md_lines.append(f"| {HEAD_LABELS[head]} | - | - | - | - | - |")
                    continue
                r = row.iloc[0]
                md_lines.append(
                    f"| {HEAD_LABELS[head]} | {r['panel_coverage_Mb']:.2f} | "
                    f"{fmt_pct_ci(r['mean_abs_recall'], r['abs_recall_ci_lo'], r['abs_recall_ci_hi'])} | "
                    f"{fmt_num_ci(r['mean_ratio_vs_TCW'], r['ratio_tcw_ci_lo'], r['ratio_tcw_ci_hi'])} | "
                    f"{fmt_num_ci(r['mean_ratio_vs_NPOS'], r['ratio_npos_ci_lo'], r['ratio_npos_ci_hi'])} | "
                    f"{int(r['n_cancers_bonf_signif'])}/10 |")
            md_lines.append("")

    # 2. Best head per panel size
    md_lines.append("## 2. Which head wins?")
    md_lines.append("")
    md_lines.append("| pct | filter | best by abs_recall | best by ratio_vs_NPOS |")
    md_lines.append("|-----|--------|--------------------|----------------------|")
    for pct in TOP_PCTS:
        for fn in ["filter_all_CT", "filter_TCW_nonCpG"]:
            sub = sweep[(sweep["level"] == "position")
                        & (sweep["filter"] == fn)
                        & (sweep["cut_type"] == "top_pct")
                        & (np.isclose(sweep["cut_value"], pct))]
            if len(sub) == 0:
                md_lines.append(f"| {pct:.2f} | {fn} | NO DATA | NO DATA |")
                continue
            best_abs = sub.loc[sub["mean_abs_recall"].idxmax()]
            best_ratio = sub.loc[sub["mean_ratio_vs_NPOS"].idxmax()]
            md_lines.append(
                f"| {pct:.2f} | {fn} | "
                f"{HEAD_LABELS[best_abs['head']]} ({fmt_pct(best_abs['mean_abs_recall'])}) | "
                f"{HEAD_LABELS[best_ratio['head']]} "
                f"({best_ratio['mean_ratio_vs_NPOS']:.3f}) |")
    md_lines.append("")

    # 3. Jaccard
    md_lines.append("## 3. Jaccard at top-1% — do heads pick the same positions?")
    md_lines.append("")
    md_lines.append(f"Top-1% panel size = {len(top_sets[('score_binary', 0.01)]):,} positions.")
    md_lines.append("")
    md_lines.append("| head_a | head_b | |A| | |B| | intersection | union | Jaccard |")
    md_lines.append("|--------|--------|-----|-----|--------------|-------|---------|")
    for _, r in jacc_df.iterrows():
        md_lines.append(
            f"| {r['h1']} | {r['h2']} | {int(r['size_h1']):,} | "
            f"{int(r['size_h2']):,} | {int(r['intersection']):,} | "
            f"{int(r['union']):,} | {r['jaccard']:.3f} |")
    md_lines.append("")

    # 4. Ensemble tests
    md_lines.append("## 4. Ensemble test — does union(binary, apobec1_v4_cds) beat binary alone?")
    md_lines.append("")
    md_lines.append("Same top-X% per head, then union; recall computed at the union's actual size.")
    md_lines.append("")
    md_lines.append("| pct | filter | binary alone (size, recall) | "
                    "binary U apobec1_v4_cds (size, recall) | "
                    "delta vs binary |")
    md_lines.append("|-----|--------|------------------------------|"
                    "----------------------------------------|"
                    "------------------|")
    for pct in TOP_PCTS:
        for fn in ["filter_all_CT", "filter_TCW_nonCpG"]:
            single = ens_df[(ens_df["ensemble_type"] == "single")
                            & (ens_df["head"] == "binary")
                            & (ens_df["pct"] == pct)
                            & (ens_df["filter"] == fn)].iloc[0]
            ens2 = ens_df[(ens_df["ensemble_type"] == "binary_U_apobec1")
                          & (ens_df["pct"] == pct)
                          & (ens_df["filter"] == fn)].iloc[0]
            delta = ens2["mean"] - single["mean"]
            md_lines.append(
                f"| {pct:.2f} | {fn} | "
                f"{int(single['panel_size']):,} ({fmt_pct(single['mean'])}) | "
                f"{int(ens2['panel_size']):,} ({fmt_pct(ens2['mean'])}) | "
                f"{delta * 100:+.2f} pp |")
    md_lines.append("")

    md_lines.append("## 5. Ensemble test 2 — union of all 6 heads")
    md_lines.append("")
    md_lines.append("| pct | filter | union size | recall (CI) | "
                    "best single (head, recall) | "
                    "delta union vs best single |")
    md_lines.append("|-----|--------|-----------|--------------|"
                    "------------------------------|"
                    "-----------------------------|")
    for pct in TOP_PCTS:
        for fn in ["filter_all_CT", "filter_TCW_nonCpG"]:
            ens6 = ens_df[(ens_df["ensemble_type"] == "all6_union")
                          & (ens_df["pct"] == pct)
                          & (ens_df["filter"] == fn)].iloc[0]
            single_sub = ens_df[(ens_df["ensemble_type"] == "single")
                                & (ens_df["pct"] == pct)
                                & (ens_df["filter"] == fn)]
            best = single_sub.loc[single_sub["mean"].idxmax()]
            delta = ens6["mean"] - best["mean"]
            md_lines.append(
                f"| {pct:.2f} | {fn} | {int(ens6['panel_size']):,} | "
                f"{fmt_pct_ci(ens6['mean'], ens6['lo'], ens6['hi'])} | "
                f"{best['head']} ({fmt_pct(best['mean'])}) | "
                f"{delta * 100:+.2f} pp |")
    md_lines.append("")

    # Final recommendation: derived from numbers
    # We pick the head/strategy with the best mean_abs_recall at top-5% on filter_all_CT
    target = ens_df[(ens_df["pct"] == 0.05) & (ens_df["filter"] == "filter_all_CT")]
    target_sorted = target.sort_values("mean", ascending=False)
    md_lines.append("## 6. Final recommendation")
    md_lines.append("")
    md_lines.append("Ranking at top-5% / filter_all_CT (production target panel size):")
    md_lines.append("")
    md_lines.append("| rank | strategy | panel size | recall |")
    md_lines.append("|------|----------|-----------|--------|")
    for i, (_, r) in enumerate(target_sorted.iterrows(), 1):
        md_lines.append(f"| {i} | {r['head']} | {int(r['panel_size']):,} | "
                        f"{fmt_pct(r['mean'])} |")
    md_lines.append("")
    md_lines.append("Final note: see Q4/Q5 above for whether the ensemble's gain over "
                    "binary alone justifies the larger panel size at fixed top-X% per head.")
    md_lines.append("")
    md_lines.append("## 7. Files")
    md_lines.append("")
    md_lines.append(f"- `{CSV.name}`")
    md_lines.append(f"- `{PNG.name}`")
    md_lines.append(f"- `{MD.name}`")

    MD.write_text("\n".join(md_lines))
    log.info("Wrote %s", MD)


if __name__ == "__main__":
    main()
