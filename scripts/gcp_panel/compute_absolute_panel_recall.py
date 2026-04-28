#!/usr/bin/env python3
"""Compute absolute panel-level recall curves.

Headline claim form: "X Mb of CDS captures Y% of APOBEC mutations across N cancers"

Compares the NN-ranked panel against four baselines:
  1. CpG-density-ranked windows
  2. TCW-motif-density-ranked windows
  3. Exon-only (single point: 100% of CDS)
  4. Random GC-matched windows (100 random sets)

Outputs:
  experiments/multi_enzyme/outputs/pcawg_tcw_panel/absolute_recall_curve.json
  experiments/multi_enzyme/outputs/pcawg_tcw_panel/absolute_recall_curve.png
  experiments/multi_enzyme/outputs/pcawg_tcw_panel/PANEL_LEVEL_CLAIM.md

Runs both TCGA+PCAWG and POG570 cohorts.
"""
from __future__ import annotations
import gzip
import json
import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Users/shaharharel/Documents/github/edit-rna-apobec")
PANEL = ROOT / "experiments/multi_enzyme/outputs/pcawg_tcw_panel/panel_scores_cds.parquet"
TCGA = ROOT / "data/raw/tcga"
PCAWG = ROOT / "data/raw/pcawg/by_cancer"
POG = ROOT / "data/raw/pog570/POG570_small_mutations.txt.gz"
OUTDIR = ROOT / "experiments/multi_enzyme/outputs/pcawg_tcw_panel"
OUTDIR.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE = 250
PERCENTILES = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20]
CANCERS = ["blca", "brca", "cesc", "coadread", "esca", "hnsc", "lihc", "lusc", "skcm", "stad"]
POG_COHORT_MAP = {"COLO": "coadread", "STAD": "stad", "BRCA": "brca", "LUNG": "lusc",
                  "SKCM": "skcm", "ESCA": "esca", "HNSC": "hnsc", "HCC": "lihc",
                  "BLCA": "blca", "CERV": "cesc"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def get_genome():
    from pyfaidx import Fasta
    return Fasta(str(ROOT / "data/raw/genomes/hg19.fa"), as_raw=False, sequence_always_upper=True)


def filter_tcw_in_panel(muts: pd.DataFrame, panel_pos_set, genome) -> pd.DataFrame:
    """muts has chrom, pos (0-based), ref, alt. Filter to in-panel + TCW-non-CpG."""
    panel_set = panel_pos_set
    keys = list(zip(muts["chrom"], muts["pos"].astype(int)))
    in_panel = np.array([k in panel_set for k in keys])
    muts = muts[in_panel].copy()
    if len(muts) == 0:
        return muts
    keep = []
    for chrom, pos, ref, alt in zip(muts["chrom"], muts["pos"], muts["ref"], muts["alt"]):
        try:
            ch = chrom if str(chrom).startswith("chr") else f"chr{chrom}"
            p = int(pos)
            base_at = str(genome[ch][p]).upper()
            base_left = str(genome[ch][p-1]).upper()
            base_right = str(genome[ch][p+1]).upper()
        except Exception:
            keep.append(False); continue
        if base_at != ref:
            keep.append(False); continue
        if ref == "C" and alt == "T":
            keep.append(base_left == "T" and base_right in ("A", "T"))
        elif ref == "G" and alt == "A":
            keep.append(base_right == "A" and base_left in ("A", "T"))
        else:
            keep.append(False)
    return muts[pd.Series(keep, index=muts.index)]


def build_windows(panel: pd.DataFrame, genome):
    """Build 250 bp windows with NN max score, CpG count, TCW motif count, GC content."""
    log.info("Building windows ...")
    p = panel[["chrom", "pos", "score_binary"]].copy()
    p["pos"] = p["pos"].astype(int)
    p["win_start"] = (p["pos"] // WINDOW_SIZE) * WINDOW_SIZE

    grp = p.groupby(["chrom", "win_start"])
    win = grp.agg(max_score=("score_binary", "max"), n_pos=("pos", "size")).reset_index()
    win["win_end"] = win["win_start"] + WINDOW_SIZE

    log.info("  computing per-position annotations from hg19 ...")
    chrom_arr = panel["chrom"].values
    pos_arr = panel["pos"].astype(int).values
    strand_arr = panel.get("strand", pd.Series(["+"] * len(panel))).values
    is_cpg = np.zeros(len(panel), dtype=bool)
    is_tcw_noncpg = np.zeros(len(panel), dtype=bool)
    chrom_seq_cache = {}
    for ch in pd.Series(chrom_arr).unique():
        try:
            chrom_seq_cache[ch] = str(genome[ch][:]).upper()
        except Exception:
            chrom_seq_cache[ch] = None
    for i in range(len(panel)):
        seq = chrom_seq_cache.get(chrom_arr[i])
        if seq is None:
            continue
        pi = int(pos_arr[i])
        s = strand_arr[i]
        try:
            if s == "+":
                # Position is C on forward strand
                left = seq[pi-1] if pi-1 >= 0 else "N"
                right = seq[pi+1] if pi+1 < len(seq) else "N"
                is_cpg[i] = (right == "G")
                is_tcw_noncpg[i] = (left == "T") and (right in ("A", "T"))
            else:
                # Position is G on forward, but is C on minus strand. Forward genome[pi-1:pi+2]
                # represents minus-strand TCN motif as (right, this(G), left) reverse-comp'd.
                # G on plus = C on minus. The 5' neighbor in minus-strand frame is at pi+1; the 3' is at pi-1.
                left_minus = seq[pi+1] if pi+1 < len(seq) else "N"      # minus 5' = plus 3' rev-comp
                right_minus = seq[pi-1] if pi-1 >= 0 else "N"           # minus 3' = plus 5' rev-comp
                # Map to minus-strand bases by complement
                comp = {"A":"T", "T":"A", "C":"G", "G":"C", "N":"N"}
                lm = comp.get(left_minus, "N")
                rm = comp.get(right_minus, "N")
                is_cpg[i] = (rm == "G")
                is_tcw_noncpg[i] = (lm == "T") and (rm in ("A", "T"))
        except Exception:
            pass
    p2 = panel[["chrom", "pos"]].copy()
    p2["pos"] = p2["pos"].astype(int)
    p2["win_start"] = (p2["pos"] // WINDOW_SIZE) * WINDOW_SIZE
    p2["is_cpg"] = is_cpg
    p2["is_tcw"] = is_tcw_noncpg
    cpg_per = p2.groupby(["chrom", "win_start"])["is_cpg"].sum().reset_index().rename(columns={"is_cpg": "cpg_count"})
    tcw_per = p2.groupby(["chrom", "win_start"])["is_tcw"].sum().reset_index().rename(columns={"is_tcw": "tcw_count"})
    win = win.merge(cpg_per, on=["chrom", "win_start"], how="left")
    win = win.merge(tcw_per, on=["chrom", "win_start"], how="left")
    win["cpg_count"] = win["cpg_count"].fillna(0).astype(int)
    win["tcw_count"] = win["tcw_count"].fillna(0).astype(int)

    # GC content per window — counted from genome sequence directly
    log.info("  computing GC content per window ...")
    gc_count = []
    win_chrom = win["chrom"].values
    win_start = win["win_start"].values
    for ch, ws in zip(win_chrom, win_start):
        seq = chrom_seq_cache.get(ch)
        if seq is None:
            gc_count.append(0); continue
        sub = seq[ws:ws+WINDOW_SIZE]
        gc = sum(1 for c in sub if c in "GC")
        gc_count.append(gc)
    win["gc_count"] = gc_count
    win["gc_content"] = win["gc_count"] / WINDOW_SIZE
    win["win_id"] = np.arange(len(win))
    log.info("  built %d windows", len(win))
    return win


def load_tcga_pcawg_all(panel_pos_set, genome):
    """Load TCGA + PCAWG mutations for all 10 cancers. Return DataFrame with cancer column."""
    out = []
    for cancer in CANCERS:
        log.info("Loading TCGA+PCAWG %s ...", cancer)
        for src, base in [("TCGA", TCGA), ("PCAWG", PCAWG)]:
            if src == "TCGA":
                path = base / f"{cancer}_tcga_pan_can_atlas_2018_mutations.txt"
            else:
                path = base / f"{cancer}_pcawg_mutations.txt"
            if not path.exists():
                continue
            df = pd.read_csv(path, sep="\t", low_memory=False, comment="#")
            df = df[df["Variant_Type"] == "SNP"]
            df = df[((df["Reference_Allele"] == "C") & (df["Tumor_Seq_Allele2"] == "T")) |
                    ((df["Reference_Allele"] == "G") & (df["Tumor_Seq_Allele2"] == "A"))]
            df = df.rename(columns={"Chromosome": "chrom", "Start_Position": "pos",
                                    "Reference_Allele": "ref", "Tumor_Seq_Allele2": "alt"})
            df["chrom"] = df["chrom"].astype(str).apply(lambda s: s if s.startswith("chr") else f"chr{s}")
            df["pos"] = pd.to_numeric(df["pos"], errors="coerce").astype("Int64").astype(int) - 1
            df["cancer"] = cancer
            df["source"] = src
            out.append(df[["chrom", "pos", "ref", "alt", "cancer", "source"]])
    if not out:
        return pd.DataFrame()
    df = pd.concat(out, ignore_index=True)
    log.info("  combined TCGA+PCAWG: %d C>T/G>A rows across cancers", len(df))
    df = filter_tcw_in_panel(df, panel_pos_set, genome)
    log.info("  in-panel TCW non-CpG: %d", len(df))
    return df


def load_pog570(panel_pos_set, genome):
    log.info("Loading POG570 ...")
    df = pd.read_csv(POG, sep="\t", compression="gzip", low_memory=False)
    df = df[(df["ref"].str.len() == 1) & (df["alt"].str.len() == 1)].copy()
    df = df[((df["ref"] == "C") & (df["alt"] == "T")) | ((df["ref"] == "G") & (df["alt"] == "A"))]
    df["cancer"] = df["analysis_cohort"].map(POG_COHORT_MAP)
    df = df.dropna(subset=["cancer"])
    df["chrom"] = "chr" + df["chrom"].astype(str)
    df["pos"] = pd.to_numeric(df["pos"], errors="coerce").astype("Int64").astype(int) - 1
    df = df[["chrom", "pos", "ref", "alt", "cancer"]]
    log.info("  POG570 C>T/G>A in target cohorts: %d", len(df))
    df = filter_tcw_in_panel(df, panel_pos_set, genome)
    log.info("  in-panel TCW non-CpG: %d", len(df))
    return df


def assign_to_windows(muts: pd.DataFrame, win_lookup):
    muts = muts.copy()
    muts["win_start"] = (muts["pos"].astype(int) // WINDOW_SIZE) * WINDOW_SIZE
    keys = list(zip(muts["chrom"], muts["win_start"]))
    muts["win_id"] = win_lookup.reindex(keys).values
    return muts.dropna(subset=["win_id"]).assign(win_id=lambda d: d["win_id"].astype(int))


def compute_recall_curve(win, muts_assigned, ranking_col, percentiles, label):
    """Sort windows by ranking_col descending, compute cumulative recall at each percentile."""
    n_win = len(win)
    sort_idx = np.argsort(-win[ranking_col].values, kind="stable")
    sorted_win_ids = win["win_id"].values[sort_idx]
    win_id_to_rank = np.empty(n_win, dtype=np.int64)
    win_id_to_rank[sorted_win_ids] = np.arange(n_win)

    rows = []
    total_muts = len(muts_assigned)
    if total_muts == 0:
        return pd.DataFrame()
    mut_window_ranks = win_id_to_rank[muts_assigned["win_id"].astype(int).values]
    # Coverage in Mb
    for pct in percentiles:
        n_top = max(1, int(round(n_win * pct)))
        n_captured = int((mut_window_ranks < n_top).sum())
        recall = n_captured / total_muts
        coverage_mb = n_top * WINDOW_SIZE / 1e6
        rows.append({
            "label": label,
            "percentile": pct,
            "n_top_windows": n_top,
            "coverage_mb": coverage_mb,
            "n_total_mutations": total_muts,
            "n_captured": n_captured,
            "recall": recall,
        })
    return pd.DataFrame(rows)


def compute_random_gc_matched(win, muts_assigned, percentiles, n_replicates=100, seed=42):
    """Random sets matched on GC content (decile-stratified). Return mean and std recall per percentile."""
    rng = np.random.default_rng(seed)
    win = win.copy()
    win["gc_decile"] = pd.qcut(win["gc_content"], 10, labels=False, duplicates="drop")
    n_win = len(win)
    total_muts = len(muts_assigned)
    if total_muts == 0:
        return pd.DataFrame()
    mut_window_set = set(muts_assigned["win_id"].astype(int).values)

    # Assign each window a GC decile bucket
    decile_to_winids = win.groupby("gc_decile")["win_id"].apply(list).to_dict()

    rows = []
    for pct in percentiles:
        n_top = max(1, int(round(n_win * pct)))
        # For each replicate: sample n_top windows preserving GC decile distribution (uniform proportional)
        # Simplification: sample n_top windows uniformly from each decile in proportion to total decile size
        recalls = []
        decile_sizes = win.groupby("gc_decile").size().to_dict()
        decile_quota = {d: max(1, int(round(n_top * sz / n_win))) for d, sz in decile_sizes.items()}
        for _ in range(n_replicates):
            picked = []
            for d, quota in decile_quota.items():
                pool = decile_to_winids[d]
                sample = rng.choice(pool, size=min(quota, len(pool)), replace=False)
                picked.extend(sample.tolist())
            # truncate to n_top
            picked = picked[:n_top]
            n_captured = sum(1 for w in picked if w in mut_window_set)
            recalls.append(n_captured / total_muts)
        rows.append({
            "label": "random_gc_matched",
            "percentile": pct,
            "n_top_windows": n_top,
            "coverage_mb": n_top * WINDOW_SIZE / 1e6,
            "n_total_mutations": total_muts,
            "n_captured_mean": float(np.mean(recalls) * total_muts),
            "n_captured_std": float(np.std(recalls) * total_muts),
            "recall_mean": float(np.mean(recalls)),
            "recall_std": float(np.std(recalls)),
        })
    return pd.DataFrame(rows)


def per_cancer_recall(win, muts_assigned, ranking_col, pct):
    """Per-cancer recall at one specific percentile."""
    n_win = len(win)
    n_top = max(1, int(round(n_win * pct)))
    sort_idx = np.argsort(-win[ranking_col].values, kind="stable")
    top_set = set(win["win_id"].values[sort_idx[:n_top]])
    rows = []
    for cancer, sub in muts_assigned.groupby("cancer"):
        n = len(sub)
        if n == 0:
            continue
        n_cap = int(sub["win_id"].astype(int).isin(top_set).sum())
        rows.append({
            "cancer": cancer,
            "percentile": pct,
            "n_total": n,
            "n_captured": n_cap,
            "recall": n_cap / n,
        })
    return pd.DataFrame(rows)


def make_plot(curves_dict, out_png, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 6))
    for label, df in curves_dict.items():
        if "recall_mean" in df.columns:
            ax.errorbar(df["coverage_mb"], df["recall_mean"], yerr=df["recall_std"],
                        marker="o", label=label, alpha=0.8)
        elif "recall" in df.columns:
            ax.plot(df["coverage_mb"], df["recall"], marker="o", label=label, alpha=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Panel coverage (Mb of CDS)")
    ax.set_ylabel("Recall (fraction of TCW-non-CpG mutations captured)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    panel = pd.read_parquet(PANEL)
    panel["pos"] = panel["pos"].astype(int)
    log.info("Panel loaded: %d positions", len(panel))
    panel_pos_set = set(zip(panel["chrom"].values, panel["pos"].astype(int).values))

    genome = get_genome()
    win = build_windows(panel, genome)
    win_lookup = win.set_index(["chrom", "win_start"])["win_id"]

    # Load mutations
    tcga_pcawg = load_tcga_pcawg_all(panel_pos_set, genome)
    pog = load_pog570(panel_pos_set, genome)

    tcga_pcawg_assigned = assign_to_windows(tcga_pcawg, win_lookup)
    pog_assigned = assign_to_windows(pog, win_lookup)
    log.info("TCGA+PCAWG assigned: %d  POG570 assigned: %d", len(tcga_pcawg_assigned), len(pog_assigned))

    results = {}
    for cohort_name, muts_assigned in [("tcga_pcawg", tcga_pcawg_assigned), ("pog570", pog_assigned)]:
        log.info("=== %s ===", cohort_name)
        nn_curve = compute_recall_curve(win, muts_assigned, "max_score", PERCENTILES, "NN")
        cpg_curve = compute_recall_curve(win, muts_assigned, "cpg_count", PERCENTILES, "CpG-density")
        tcw_curve = compute_recall_curve(win, muts_assigned, "tcw_count", PERCENTILES, "TCW-density")
        rand_curve = compute_random_gc_matched(win, muts_assigned, PERCENTILES, n_replicates=100)
        # exon-only is full panel: 100% recall by definition
        exon_curve = pd.DataFrame([{
            "label": "exon-only",
            "percentile": 1.0,
            "n_top_windows": len(win),
            "coverage_mb": len(win) * WINDOW_SIZE / 1e6,
            "n_total_mutations": len(muts_assigned),
            "n_captured": len(muts_assigned),
            "recall": 1.0,
        }])

        # Per-cancer breakdown at 1%
        per_cancer_nn_1 = per_cancer_recall(win, muts_assigned, "max_score", 0.01)
        per_cancer_cpg_1 = per_cancer_recall(win, muts_assigned, "cpg_count", 0.01)

        # Per-cancer breakdown at 5%
        per_cancer_nn_5 = per_cancer_recall(win, muts_assigned, "max_score", 0.05)

        results[cohort_name] = {
            "n_total_mutations": int(len(muts_assigned)),
            "n_windows_total": int(len(win)),
            "panel_total_coverage_mb": float(len(win) * WINDOW_SIZE / 1e6),
            "nn_curve": nn_curve.to_dict(orient="records"),
            "cpg_curve": cpg_curve.to_dict(orient="records"),
            "tcw_curve": tcw_curve.to_dict(orient="records"),
            "random_gc_curve": rand_curve.to_dict(orient="records"),
            "exon_only": exon_curve.to_dict(orient="records"),
            "per_cancer_nn_at_1pct": per_cancer_nn_1.to_dict(orient="records"),
            "per_cancer_cpg_at_1pct": per_cancer_cpg_1.to_dict(orient="records"),
            "per_cancer_nn_at_5pct": per_cancer_nn_5.to_dict(orient="records"),
        }

        # Plot
        out_png = OUTDIR / f"absolute_recall_curve_{cohort_name}.png"
        curves = {
            "NN (APOBEC)": nn_curve,
            "CpG density": cpg_curve,
            "TCW motif density": tcw_curve,
            "Random GC-matched (100 reps)": rand_curve,
        }
        make_plot(curves, out_png, f"Panel recall curve — {cohort_name}")
        log.info("Saved %s", out_png)

    out_json = OUTDIR / "absolute_recall_curve.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info("Wrote %s", out_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
