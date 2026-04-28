#!/usr/bin/env python3
"""Analysis D — POG570 independent WGS validation (v2: Phase-1.5-style baseline).

Identical to analysis_D_pog570_validation.py EXCEPT the CpG-density baseline:

    Phase 1.5 (Analysis A/B): cpg_density per window = seq.count("CG") over the
    250 bp hg19 window sequence (every CG dinucleotide in the literal genomic
    stretch, plus-strand only).
    POG570 v1 (Analysis D): cpg_count = sum(is_cpg) over panel positions only.

The two baselines pick only ~43% of the same top-1% windows (QA_FULL_AUDIT.md
§2.5). This script switches to Phase 1.5's definition for apples-to-apples.

Output: enrichment_primary_pog570_v2_baseline_fixed.json
        REPORT_v2_baseline_fixed.md

macOS-safe multiprocessing via concurrent.futures.ProcessPoolExecutor.
"""
from __future__ import annotations
import json
import logging
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control

ROOT = Path("/Users/shaharharel/Documents/github/edit-rna-apobec")
PANEL_PATH = ROOT / "experiments/multi_enzyme/outputs/pcawg_tcw_panel/panel_scores_cds.parquet"
POG570_PATH = ROOT / "data/raw/pog570/POG570_small_mutations.txt.gz"
OUT_DIR = ROOT / "experiments/multi_enzyme/outputs/pcawg_tcw_panel/analysis_D_pog570"
OUT_DIR.mkdir(parents=True, exist_ok=True)
HG19 = ROOT / "data/raw/genomes/hg19.fa"

COHORT_MAP = {
    "COLO": "coadread",
    "SKCM": "skcm",
    "BRCA": "brca",
    "LUNG": "lusc",   # POG570 doesn't separate LUSC vs LUAD
    "ESCA": "esca",
    "HNSC": "hnsc",
    "STAD": "stad",
    "HCC": "lihc",
    "BLCA": "blca",
    "CERV": "cesc",
}
TARGET_CANCERS = list(COHORT_MAP.values())
WINDOW_SIZE = 250
TOP_PCT = 0.01
PERM_REPS = 10000
SEED_BASE = 20260427

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_pog570(path: Path) -> pd.DataFrame:
    log.info("Loading POG570 small mutations from %s", path)
    df = pd.read_csv(path, sep="\t", compression="gzip", low_memory=False)
    log.info("  raw rows: %d", len(df))
    df = df[(df["ref"].str.len() == 1) & (df["alt"].str.len() == 1)].copy()
    df = df[((df["ref"] == "C") & (df["alt"] == "T")) | ((df["ref"] == "G") & (df["alt"] == "A"))]
    log.info("  C>T/G>A SNVs: %d", len(df))
    df["cancer"] = df["analysis_cohort"].map(COHORT_MAP)
    df = df.dropna(subset=["cancer"])
    log.info("  in target cohorts: %d", len(df))
    df["chrom"] = "chr" + df["chrom"].astype(str)
    df["pos"] = pd.to_numeric(df["pos"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["pos"])
    df["pos"] = df["pos"].astype(int) - 1   # 1-based VCF -> 0-based panel
    return df[["chrom", "pos", "ref", "alt", "cancer", "patient_id"]]


def load_panel(path: Path) -> pd.DataFrame:
    log.info("Loading panel scores from %s", path)
    df = pd.read_parquet(path)
    log.info("  rows: %d  cols: %s", len(df), list(df.columns)[:12])
    keep = ["chrom", "pos", "score_binary"]
    return df[[c for c in keep if c in df.columns]].copy()


def filter_tcw_non_cpg(mut_df: pd.DataFrame, panel_df: pd.DataFrame) -> pd.DataFrame:
    from pyfaidx import Fasta
    log.info("Loading hg19.fa for trinucleotide context check ...")
    genome = Fasta(str(HG19), as_raw=False, sequence_always_upper=True)

    log.info("Filtering POG570 mutations to those in-panel ...")
    panel_set = panel_df[["chrom", "pos"]].drop_duplicates(["chrom", "pos"]).copy()
    panel_set["pos"] = panel_set["pos"].astype(int)
    merged = mut_df.merge(panel_set, on=["chrom", "pos"], how="inner")
    log.info("  in-panel mutations: %d / %d (%.1f%%)",
             len(merged), len(mut_df), 100.0 * len(merged) / max(1, len(mut_df)))

    keep = []
    sanity_ref_mismatches = 0
    for chrom, pos, ref, alt in zip(merged["chrom"], merged["pos"], merged["ref"], merged["alt"]):
        try:
            chrom_str = chrom if str(chrom).startswith("chr") else f"chr{chrom}"
            p = int(pos)
            base_at = str(genome[chrom_str][p]).upper()
            base_left = str(genome[chrom_str][p - 1]).upper()
            base_right = str(genome[chrom_str][p + 1]).upper()
        except Exception:
            keep.append(False)
            continue
        if base_at != ref:
            sanity_ref_mismatches += 1
            keep.append(False)
            continue
        if ref == "C" and alt == "T":
            keep.append(base_left == "T" and base_right in ("A", "T"))
        elif ref == "G" and alt == "A":
            keep.append(base_right == "A" and base_left in ("A", "T"))
        else:
            keep.append(False)
    if sanity_ref_mismatches > 0:
        log.warning("  %d/%d positions had ref-base mismatch with hg19",
                    sanity_ref_mismatches, len(merged))
    merged = merged[pd.Series(keep, index=merged.index)]
    log.info("  TCW-non-CpG mutations: %d", len(merged))
    return merged


def build_windows_phase15_baseline(panel_df: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """Aggregate panel -> 250bp windows by max(score_binary), with CpG density
    computed as seq.count('CG') over the literal hg19 window sequence (Phase 1.5
    convention)."""
    from pyfaidx import Fasta
    log.info("Building %d bp non-overlapping windows over panel positions ...", window_size)
    p = panel_df[["chrom", "pos", "score_binary"]].copy()
    p["pos"] = p["pos"].astype(int)
    p["win_start"] = (p["pos"] // window_size) * window_size
    grp = p.groupby(["chrom", "win_start"])
    out = grp.agg(
        max_score_binary=("score_binary", "max"),
        n_pos=("pos", "size"),
    ).reset_index()
    out["win_end"] = out["win_start"] + window_size

    log.info("  Computing Phase-1.5 cpg_density = seq.count('CG') for %d windows ...", len(out))
    genome = Fasta(str(HG19), as_raw=False, sequence_always_upper=True)
    cpg_density = np.zeros(len(out), dtype=np.int32)
    chroms = out["chrom"].to_numpy()
    starts = out["win_start"].to_numpy()
    # Cache chr seq for a chunk of consecutive same-chrom rows
    cur_chrom, cur_seq = None, None
    for i in range(len(out)):
        ch = chroms[i]
        if ch != cur_chrom:
            try:
                cur_seq = str(genome[ch][:]).upper()
                cur_chrom = ch
            except Exception:
                cur_seq, cur_chrom = "", ch
        s = int(starts[i])
        e = s + window_size
        if e > len(cur_seq):
            e = len(cur_seq)
        if s >= len(cur_seq):
            cpg_density[i] = 0
            continue
        seq = cur_seq[s:e]
        cpg_density[i] = seq.count("CG")
    out["cpg_count"] = cpg_density
    log.info("  built %d windows; mean cpg_count=%.2f, max=%d",
             len(out), float(cpg_density.mean()), int(cpg_density.max()))
    return out


def compute_per_cancer_primary(args):
    """macOS-safe per-cancer worker: ratio + permutation null."""
    cancer, windows_df_arr, mut_pos_arr, n_top, perm_reps, seed = args
    rng = np.random.default_rng(seed)
    n_windows = windows_df_arr.shape[0]
    nn_top_idx = np.argpartition(-windows_df_arr[:, 1], n_top)[:n_top]
    cpg_top_idx = np.argpartition(-windows_df_arr[:, 2], n_top)[:n_top]
    mut_per_window = np.bincount(mut_pos_arr, minlength=n_windows)
    nn_recall = mut_per_window[nn_top_idx].sum() / max(1, mut_per_window.sum())
    cpg_recall = mut_per_window[cpg_top_idx].sum() / max(1, mut_per_window.sum())
    ratio = nn_recall / cpg_recall if cpg_recall > 0 else float("nan")
    null_ratios = []
    scores = windows_df_arr[:, 1].copy()
    for _ in range(perm_reps):
        perm = rng.permutation(scores)
        perm_top_idx = np.argpartition(-perm, n_top)[:n_top]
        perm_recall = mut_per_window[perm_top_idx].sum() / max(1, mut_per_window.sum())
        if cpg_recall > 0:
            null_ratios.append(perm_recall / cpg_recall)
    null_ratios = np.array(null_ratios) if null_ratios else np.array([np.nan])
    p_perm = ((null_ratios >= ratio).sum() + 1) / (len(null_ratios) + 1)
    return {
        "cancer": cancer,
        "n_mutations_in_panel_tcw_noncpg": int(mut_per_window.sum()),
        "n_windows": n_windows,
        "n_top": n_top,
        "nn_recall": float(nn_recall),
        "cpg_recall": float(cpg_recall),
        "ratio": float(ratio),
        "p_perm": float(p_perm),
        "null_mean": float(np.nanmean(null_ratios)) if not np.all(np.isnan(null_ratios)) else float("nan"),
        "null_std": float(np.nanstd(null_ratios)) if not np.all(np.isnan(null_ratios)) else float("nan"),
    }


def main():
    log.info("=== Analysis D v2 (baseline-fixed): POG570 with Phase-1.5 CpG baseline ===")
    panel = load_panel(PANEL_PATH)
    muts = load_pog570(POG570_PATH)
    tcw_muts = filter_tcw_non_cpg(muts, panel)

    if len(tcw_muts) == 0:
        log.error("No TCW-non-CpG mutations found in panel positions. Aborting.")
        return 1

    windows_df = build_windows_phase15_baseline(panel, WINDOW_SIZE)
    n_top = max(1, int(round(len(windows_df) * TOP_PCT)))
    log.info("Top-1%% of windows = %d", n_top)

    windows_df["win_id"] = np.arange(len(windows_df))
    windows_arr = windows_df[["win_id", "max_score_binary", "cpg_count"]].to_numpy()

    log.info("Mapping mutations to windows ...")
    tcw_muts["win_start"] = (tcw_muts["pos"].astype(int) // WINDOW_SIZE) * WINDOW_SIZE
    win_id_lookup = windows_df.set_index(["chrom", "win_start"])["win_id"]
    tcw_muts["win_id"] = tcw_muts.set_index(["chrom", "win_start"]).index.map(win_id_lookup)
    tcw_muts = tcw_muts.dropna(subset=["win_id"])
    tcw_muts["win_id"] = tcw_muts["win_id"].astype(int)
    log.info("  %d mutations mapped to windows", len(tcw_muts))

    cancers = sorted(tcw_muts["cancer"].unique())
    log.info("Cancers found: %s", cancers)

    args_list = []
    for i, cancer in enumerate(cancers):
        sub = tcw_muts[tcw_muts["cancer"] == cancer]
        if len(sub) < 100:
            log.warning("  %s has only %d mutations — including but flagging low power", cancer, len(sub))
        mut_arr = sub["win_id"].astype(int).to_numpy()
        seed = SEED_BASE + i * 1000
        args_list.append((cancer, windows_arr, mut_arr, n_top, PERM_REPS, seed))

    log.info("Running per-cancer in ProcessPoolExecutor (macOS-safe) ...")
    results = []
    n_workers = min(8, len(args_list))
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        fut_map = {ex.submit(compute_per_cancer_primary, a): a[0] for a in args_list}
        for fut in as_completed(fut_map):
            results.append(fut.result())

    # Sort results back into deterministic order (matches input cancers list)
    by_cancer = {r["cancer"]: r for r in results}
    results = [by_cancer[c] for c in cancers if c in by_cancer]

    p_vals = [r["p_perm"] for r in results]
    if p_vals:
        q_vals = false_discovery_control(p_vals, method="bh")
    else:
        q_vals = []
    for r, q in zip(results, q_vals):
        r["q_bh"] = float(q)
        r["reject_bh"] = bool(q < 0.025)

    ratios = [r["ratio"] for r in results if not np.isnan(r["ratio"])]
    summary = {
        "experiment": "Analysis D v2 — POG570 validation (Phase-1.5-style CpG baseline)",
        "panel_path": str(PANEL_PATH),
        "pog570_path": str(POG570_PATH),
        "window_size": WINDOW_SIZE,
        "aggregator": "max",
        "top_pct": TOP_PCT,
        "perm_reps": PERM_REPS,
        "filter": "tcw_not_cpg",
        "head": "binary",
        "baseline": "phase1_5_seq_count_CG",
        "mean_ratio": float(np.mean(ratios)) if ratios else float("nan"),
        "median_ratio": float(np.median(ratios)) if ratios else float("nan"),
        "n_cancers_above_1": int(sum(r > 1 for r in ratios)),
        "n_cancers_above_1_5": int(sum(r > 1.5 for r in ratios)),
        "n_bh_significant_p025": int(sum(r["reject_bh"] for r in results)),
        "primary_pass_a_mean_15": (float(np.mean(ratios)) if ratios else 0.0) >= 1.5,
        "primary_pass_b_bh_6_of_n": int(sum(r["reject_bh"] for r in results)) >= 6,
        "per_cancer": results,
        "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    out_json = OUT_DIR / "enrichment_primary_pog570_v2_baseline_fixed.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Wrote %s", out_json)

    # Quick report
    log.info("=== Summary (v2 baseline-fixed) ===")
    log.info("Mean ratio across %d cancers: %.2f×", len(ratios), summary["mean_ratio"])
    log.info("Cancers above 1.0×: %d / %d", summary["n_cancers_above_1"], len(results))
    log.info("Cancers above 1.5×: %d / %d", summary["n_cancers_above_1_5"], len(results))
    log.info("BH-significant (q<0.025): %d / %d", summary["n_bh_significant_p025"], len(results))
    log.info("Primary (a) mean>=1.5: %s", "PASS" if summary["primary_pass_a_mean_15"] else "FAIL")
    log.info("Primary (b) >=6/N BH<0.025: %s", "PASS" if summary["primary_pass_b_bh_6_of_n"] else "FAIL")
    log.info("Per-cancer:")
    for r in sorted(results, key=lambda x: -x["ratio"] if not np.isnan(x["ratio"]) else -999):
        log.info("  %-10s ratio=%6.2f  q=%.4f  rej=%s  n_mut=%d  nn_rec=%.4f  cpg_rec=%.4f",
                 r["cancer"], r["ratio"], r["q_bh"], r["reject_bh"],
                 r["n_mutations_in_panel_tcw_noncpg"], r["nn_recall"], r["cpg_recall"])

    # Write a brief REPORT
    report_path = OUT_DIR / "REPORT_v2_baseline_fixed.md"
    with open(report_path, "w") as f:
        f.write("# Analysis D v2 — POG570 (Phase-1.5 CpG baseline)\n\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write("## Baseline change vs v1\n\n")
        f.write("v1 used `sum(is_cpg)` per panel position; v2 uses Phase-1.5's `seq.count('CG')` over the 250 bp hg19 window sequence.\n\n")
        f.write(f"## Headline\n\n")
        f.write(f"- Mean ratio: **{summary['mean_ratio']:.2f}×** across {len(ratios)} testable cancers\n")
        f.write(f"- Cancers >1.0×: {summary['n_cancers_above_1']}/{len(results)}\n")
        f.write(f"- Cancers >1.5×: {summary['n_cancers_above_1_5']}/{len(results)}\n")
        f.write(f"- BH q<0.025: {summary['n_bh_significant_p025']}/{len(results)}\n\n")
        f.write("## Per-cancer (sorted by ratio desc)\n\n")
        f.write("| cancer | ratio | nn_recall | cpg_recall | total_mut | p_perm | q_bh | reject |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---|\n")
        for r in sorted(results, key=lambda x: -x["ratio"] if not np.isnan(x["ratio"]) else -999):
            ratio_str = f"{r['ratio']:.3f}" if not np.isnan(r["ratio"]) else "NaN"
            f.write(f"| {r['cancer']} | {ratio_str} | {r['nn_recall']:.4f} | "
                    f"{r['cpg_recall']:.4f} | {r['n_mutations_in_panel_tcw_noncpg']} | "
                    f"{r['p_perm']:.3e} | {r['q_bh']:.3g} | {'Y' if r['reject_bh'] else 'N'} |\n")
    log.info("Wrote %s", report_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
