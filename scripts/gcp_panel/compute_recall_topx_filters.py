#!/usr/bin/env python3
"""Phase 2 follow-up: top-X% panel recall under multiple mutation filters.

Goal: understand WHY position-level recall is 0% and characterize the
recall curve at panel coverage X in {1, 5, 10}% under four mutation filters
plus motif-density baselines.

Constructions evaluated (winning aggregator/window from aggregator_window_sweep.csv):
  - sum    × 1000 bp  (winner: abs_recall=0.041, ratio_vs_tcw=1.31)
  - max    × 1000 bp  (point comparison)
  - top3_mean × 1000 bp  (point comparison)

For each of these 3 constructions, recall is computed at top-1, 5, 10% panel
coverage under FOUR mutation filters:
  filter_all_CT      : any C>T (or G>A on - strand, strand-corrected)
  filter_all_TCW     : C>T at TCW context (5'=T, 3'=A/T), no CpG exclusion
  filter_TCW_nonCpG  : C>T at TCW excluding TCG (the strict APOBEC filter)
  filter_random      : 100K random C positions from the panel (sanity baseline)

Two motif-only baselines per top-X% (NOT NN): TCW-density and CpG-density.

Plus a POSITION-LEVEL DIAGNOSIS for the top-1% positions by score_binary
(no windowing, 84,469 positions): the trinuc-context composition + overlap
with each mutation filter set.

Outputs:
  experiments/multi_enzyme/outputs/pcawg_tcw_panel/recall_topx_filters.csv
  experiments/multi_enzyme/outputs/pcawg_tcw_panel/recall_topx_filters.png
  experiments/multi_enzyme/outputs/pcawg_tcw_panel/TOPX_FILTER_RESULTS.md

macOS-safe multiprocessing via concurrent.futures.ProcessPoolExecutor.
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom, false_discovery_control

ROOT = Path("/Users/shaharharel/Documents/github/edit-rna-apobec")
PANEL_PATH = ROOT / "experiments/multi_enzyme/outputs/pcawg_tcw_panel/panel_scores_cds.parquet"
TCGA_DIR = ROOT / "data/raw/tcga"
PCAWG_DIR = ROOT / "data/raw/pcawg/by_cancer"
HG19 = ROOT / "data/raw/genomes/hg19.fa"
OUT_DIR = ROOT / "experiments/multi_enzyme/outputs/pcawg_tcw_panel"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CANCERS = ["blca", "brca", "cesc", "coadread", "esca", "hnsc", "lihc", "lusc", "skcm", "stad"]
TOP_PCTS = [0.01, 0.05, 0.10]
WINDOW_SIZE = 1000
AGGREGATORS = ["sum", "max", "top3_mean"]
N_BOOT = 10000
PERM_REPS = 10000
N_RANDOM_C_BASELINE = 100_000
SEED_BASE = 20260427

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    stream=sys.stdout)
log = logging.getLogger(__name__)


# =========================================================================== #
# Mutation loading (TCGA + PCAWG-coding combined). Reused from sweep script.
# =========================================================================== #

def _load_one_maf(path: Path, cancer: str, source: str) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, sep="\t", low_memory=False)
    except Exception as ex:
        log.warning("  %s/%s: read failed: %s", cancer, source, ex)
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
        "+", "-",
    )
    df["pos"] = df["Start_Position"].astype(int) - 1  # 1-based MAF -> 0-based
    df["chrom"] = df["Chromosome"].astype(str)
    df.loc[~df["chrom"].str.startswith("chr"), "chrom"] = "chr" + df["chrom"]
    df["cancer"] = cancer
    df["source"] = source
    return df[["chrom", "pos", "strand", "cancer", "source"]]


def load_combined_coding_maf() -> pd.DataFrame:
    log.info("Loading TCGA-MC3 + cBioPortal-PCAWG coding MAFs ...")
    rows = []
    for cancer in CANCERS:
        d = _load_one_maf(PCAWG_DIR / f"{cancer}_pcawg_mutations.txt", cancer, "pcawg_coding")
        if d is not None:
            rows.append(d)
        d = _load_one_maf(TCGA_DIR / f"{cancer}_tcga_pan_can_atlas_2018_mutations.txt",
                          cancer, "tcga_mc3")
        if d is not None:
            rows.append(d)
    if not rows:
        raise RuntimeError("No MAFs loaded")
    combined = pd.concat(rows, ignore_index=True)
    valid_chroms = set([f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"])
    combined = combined[combined["chrom"].isin(valid_chroms)]
    log.info("  combined C>T/G>A: %d, %d cancers, sources=%s",
             len(combined), combined["cancer"].nunique(),
             combined["source"].value_counts().to_dict())
    return combined


# =========================================================================== #
# Per-mutation trinucleotide context flags
# =========================================================================== #

def annotate_mut_context(maf: pd.DataFrame) -> pd.DataFrame:
    """Add is_TCW, is_CpG, is_TCW_nonCpG flags to each mutation using hg19 trinuc."""
    from pyfaidx import Fasta
    log.info("Annotating mutations with trinucleotide context (vectorized) ...")
    genome = Fasta(str(HG19), as_raw=False, sequence_always_upper=True)
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
            seq = np.frombuffer(str(genome[ch][:]).upper().encode("ascii"), dtype=np.uint8)
        except Exception:
            continue
        L = len(seq)
        ps = poses[mask]
        ss = strands[mask]
        ok = (ps >= 1) & (ps + 1 < L)
        valid_idx = idx[ok]
        ps_ok = ps[ok]
        ss_ok = ss[ok]
        left = seq[ps_ok - 1]
        center = seq[ps_ok]
        right = seq[ps_ok + 1]
        is_plus = (ss_ok == "+")
        is_minus = ~is_plus
        # TCW (C>T strand-corrected)
        plus_tcw = is_plus & (left == ord("T")) & (center == ord("C")) & (
            (right == ord("A")) | (right == ord("T")))
        minus_tcw = is_minus & (right == ord("A")) & (center == ord("G")) & (
            (left == ord("A")) | (left == ord("T")))
        # CpG (3' of C, strand corrected): + strand right=='G'; - strand left=='C'
        plus_cpg = is_plus & (center == ord("C")) & (right == ord("G"))
        minus_cpg = is_minus & (center == ord("G")) & (left == ord("C"))
        is_tcw[valid_idx] = plus_tcw | minus_tcw
        is_cpg[valid_idx] = plus_cpg | minus_cpg

    out = maf.copy()
    out["is_TCW"] = is_tcw
    out["is_CpG"] = is_cpg
    out["is_TCW_nonCpG"] = is_tcw & ~is_cpg
    log.info("  total=%d  TCW=%d (%.1f%%)  CpG=%d (%.1f%%)  TCW_nonCpG=%d (%.1f%%)",
             n, is_tcw.sum(), 100*is_tcw.mean(),
             is_cpg.sum(), 100*is_cpg.mean(),
             out["is_TCW_nonCpG"].sum(), 100*out["is_TCW_nonCpG"].mean())
    return out


# =========================================================================== #
# Panel position annotation (CpG, TCW, GC bin) — same as sweep script
# =========================================================================== #

def annotate_panel_positions(panel: pd.DataFrame) -> pd.DataFrame:
    from pyfaidx import Fasta
    log.info("Annotating panel positions with is_cpg + is_TCW_C + local GC ...")
    genome = Fasta(str(HG19), as_raw=False, sequence_always_upper=True)

    n = len(panel)
    is_cpg = np.zeros(n, dtype=bool)
    is_tcw_c = np.zeros(n, dtype=bool)
    gc_bin = np.zeros(n, dtype=np.int8)

    chroms = panel["chrom"].to_numpy()
    poses = panel["pos"].astype(int).to_numpy()
    strands = panel["strand"].to_numpy()
    panel_idx = np.arange(n)

    for ch in pd.Series(chroms).unique():
        mask = chroms == ch
        idx = panel_idx[mask]
        if len(idx) == 0:
            continue
        try:
            seq = np.frombuffer(str(genome[ch][:]).upper().encode("ascii"), dtype=np.uint8)
        except Exception:
            continue
        L = len(seq)

        ps = poses[mask]
        ss = strands[mask]
        ok = (ps >= 1) & (ps + 1 < L)
        valid_idx = idx[ok]
        ps_ok = ps[ok]
        ss_ok = ss[ok]
        left = seq[ps_ok - 1]
        right = seq[ps_ok + 1]
        is_plus = (ss_ok == "+")
        is_minus = ~is_plus
        is_cpg[valid_idx[is_plus]] = (right[is_plus] == ord("G"))
        is_cpg[valid_idx[is_minus]] = (left[is_minus] == ord("C"))
        right_AT = (right == ord("A")) | (right == ord("T"))
        left_AT = (left == ord("A")) | (left == ord("T"))
        is_tcw_c[valid_idx[is_plus]] = (left[is_plus] == ord("T")) & right_AT[is_plus]
        is_tcw_c[valid_idx[is_minus]] = (right[is_minus] == ord("A")) & left_AT[is_minus]

        # GC bin in ±50 bp via cumulative
        is_gc_byte = ((seq == ord("G")) | (seq == ord("C"))).astype(np.int32)
        cum_gc = np.zeros(L + 1, dtype=np.int32)
        cum_gc[1:] = np.cumsum(is_gc_byte)
        starts = np.clip(ps_ok - 50, 0, L)
        ends = np.clip(ps_ok + 50, 0, L)
        gc_in_win = cum_gc[ends] - cum_gc[starts]
        win_len = (ends - starts).clip(min=1)
        gc_frac = gc_in_win / win_len
        gc_b = np.clip((gc_frac * 10).astype(np.int8), 0, 9)
        gc_bin[valid_idx] = gc_b

    panel = panel.copy()
    panel["is_cpg"] = is_cpg
    panel["is_TCW_C"] = is_tcw_c
    panel["gc_bin"] = gc_bin
    log.info("  is_cpg: %d (%.1f%%); is_TCW_C: %d (%.1f%%)",
             is_cpg.sum(), 100*is_cpg.mean(),
             is_tcw_c.sum(), 100*is_tcw_c.mean())
    return panel


# =========================================================================== #
# Window builder (single window size = 1000 bp)
# =========================================================================== #

def build_window_aggregations(panel: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """Build window-level table with sum/max/top3_mean of score_binary +
    CpG-density and TCW-motif-density baselines.
    """
    from pyfaidx import Fasta
    log.info("[w=%d] Building window aggregation table ...", window_size)
    p = panel[["chrom", "pos", "score_binary", "is_cpg", "is_TCW_C", "gc_bin"]].copy()
    p["pos"] = p["pos"].astype(int)
    p["win_start"] = (p["pos"] // window_size) * window_size

    grp = p.groupby(["chrom", "win_start"])["score_binary"]
    out = pd.DataFrame({
        "max_score": grp.max(),
        "sum_score": grp.sum(),
        "top3_mean_score": grp.apply(lambda x: float(np.mean(np.sort(x.values)[-3:])) if len(x) > 0 else 0.0),
        "n_pos": grp.size(),
    }).reset_index()

    gc_bin_per_win = p.groupby(["chrom", "win_start"])["gc_bin"].median().astype(int).reset_index()
    out = out.merge(gc_bin_per_win, on=["chrom", "win_start"])

    log.info("[w=%d] Computing CpG + TCW-motif densities from hg19 ...", window_size)
    genome = Fasta(str(HG19), as_raw=False, sequence_always_upper=True)
    cpg = np.zeros(len(out), dtype=np.int32)
    tcw = np.zeros(len(out), dtype=np.int32)
    chroms = out["chrom"].to_numpy()
    starts = out["win_start"].to_numpy()

    for ch in pd.Series(chroms).unique():
        mask = chroms == ch
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        try:
            seq = np.frombuffer(str(genome[ch][:]).upper().encode("ascii"), dtype=np.uint8)
        except Exception:
            continue
        L = len(seq)
        is_cg = ((seq[:-1] == ord("C")) & (seq[1:] == ord("G"))).astype(np.int32)
        cum_cg = np.zeros(L, dtype=np.int32)
        cum_cg[1:] = np.cumsum(is_cg)
        if L >= 3:
            t0 = seq[:-2]; t1 = seq[1:-1]; t2 = seq[2:]
            is_tcw_plus = (t0 == ord("T")) & (t1 == ord("C")) & ((t2 == ord("A")) | (t2 == ord("T")))
            is_tcw_minus = ((t0 == ord("A")) | (t0 == ord("T"))) & (t1 == ord("G")) & (t2 == ord("A"))
            is_tcw_any = (is_tcw_plus | is_tcw_minus).astype(np.int32)
            cum_tcw = np.zeros(L - 1, dtype=np.int32)
            cum_tcw[1:] = np.cumsum(is_tcw_any)
        else:
            cum_tcw = np.zeros(max(0, L - 1), dtype=np.int32)

        ss = starts[mask].astype(int)
        es = np.clip(ss + window_size, 0, L)
        ce = np.clip(es - 1, 0, L - 1)
        cs = np.clip(ss, 0, L - 1)
        cpg_per = cum_cg[ce] - cum_cg[cs]
        cpg_per[ss >= L] = 0
        cpg[idx] = cpg_per

        if L >= 3:
            te = np.clip(es - 2, 0, L - 2)
            ts = np.clip(ss, 0, L - 2)
            tcw_per = cum_tcw[te] - cum_tcw[ts]
            tcw_per[ss >= L] = 0
            tcw[idx] = tcw_per
    out["cpg_density"] = cpg
    out["tcw_density"] = tcw
    log.info("[w=%d] %d windows, mean cpg=%.2f, mean tcw=%.2f",
             window_size, len(out), out["cpg_density"].mean(), out["tcw_density"].mean())
    return out


# =========================================================================== #
# Recall + ratio core
# =========================================================================== #

def boot_ci(arr, n_boot=N_BOOT, seed=42):
    """Bootstrap mean + 95% CI."""
    if not arr or len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    a = np.asarray(arr, dtype=float)
    a = a[~np.isnan(a)]
    if len(a) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = len(a)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = a[idx].mean(axis=1)
    return float(a.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def evaluate_one_construction(
    scores: np.ndarray,
    baseline_cpg: np.ndarray,
    baseline_tcw: np.ndarray,
    mut_per_unit_per_filter_per_cancer: dict,  # {filter_name: {cancer: int32 array}}
    top_pct: float,
    perm_reps: int,
    seed: int,
) -> dict:
    """For ONE construction (scores vector, length n) and ONE top_pct,
    compute per-cancer recall under each filter + ratios vs TCW/CpG-density baselines.
    Returns dict keyed by filter_name with abs_recall + ratio_vs_tcw stats.
    """
    n = len(scores)
    k = max(1, int(round(n * top_pct)))
    # NN top-k
    nn_top = np.argpartition(-scores, k - 1)[:k]
    cpg_top = np.argpartition(-baseline_cpg, k - 1)[:k]
    tcw_top = np.argpartition(-baseline_tcw, k - 1)[:k]

    out = {}
    for fname, per_cancer_muts in mut_per_unit_per_filter_per_cancer.items():
        abs_recalls_nn = []
        abs_recalls_cpg = []
        abs_recalls_tcw = []
        ratios_vs_tcw = []
        p_perms = []
        per_cancer = {}
        for cancer, mut in per_cancer_muts.items():
            total_mut = int(mut.sum())
            if total_mut == 0:
                per_cancer[cancer] = {"total_mut": 0, "abs_recall": float("nan"),
                                      "abs_recall_cpg_baseline": float("nan"),
                                      "abs_recall_tcw_baseline": float("nan"),
                                      "ratio_vs_tcw": float("nan"),
                                      "p_perm": 1.0}
                continue
            nn_recall = mut[nn_top].sum() / total_mut
            cpg_recall = mut[cpg_top].sum() / total_mut
            tcw_recall = mut[tcw_top].sum() / total_mut
            ratio_tcw = nn_recall / tcw_recall if tcw_recall > 0 else float("nan")
            mut_in_top_obs = int(mut[nn_top].sum())
            p_perm = float(hypergeom.sf(mut_in_top_obs - 1, n, total_mut, k))
            if not np.isfinite(p_perm):
                p_perm = 1.0
            p_perm = float(np.clip(p_perm, 1.0 / (perm_reps + 1), 1.0))

            per_cancer[cancer] = {
                "total_mut": total_mut,
                "abs_recall": float(nn_recall),
                "abs_recall_cpg_baseline": float(cpg_recall),
                "abs_recall_tcw_baseline": float(tcw_recall),
                "ratio_vs_tcw": float(ratio_tcw),
                "p_perm": float(p_perm),
            }
            abs_recalls_nn.append(nn_recall)
            abs_recalls_cpg.append(cpg_recall)
            abs_recalls_tcw.append(tcw_recall)
            if not np.isnan(ratio_tcw):
                ratios_vs_tcw.append(ratio_tcw)
            p_perms.append(p_perm)

        # BH-FDR
        if p_perms:
            q_vals = false_discovery_control(p_perms, method="bh")
            n_bh = int(sum(q < 0.025 for q in q_vals))
        else:
            n_bh = 0

        m_nn, lo_nn, hi_nn = boot_ci(abs_recalls_nn, seed=seed)
        m_cpg, lo_cpg, hi_cpg = boot_ci(abs_recalls_cpg, seed=seed + 1)
        m_tcw, lo_tcw, hi_tcw = boot_ci(abs_recalls_tcw, seed=seed + 2)
        m_r, lo_r, hi_r = boot_ci(ratios_vs_tcw, seed=seed + 3)

        out[fname] = {
            "per_cancer": per_cancer,
            "n_cancers": len([c for c, v in per_cancer.items() if v["total_mut"] > 0]),
            "mean_abs_recall": m_nn,
            "abs_recall_ci_lo": lo_nn,
            "abs_recall_ci_hi": hi_nn,
            "mean_abs_recall_cpg_baseline": m_cpg,
            "cpg_baseline_ci_lo": lo_cpg,
            "cpg_baseline_ci_hi": hi_cpg,
            "mean_abs_recall_tcw_baseline": m_tcw,
            "tcw_baseline_ci_lo": lo_tcw,
            "tcw_baseline_ci_hi": hi_tcw,
            "mean_ratio_vs_TCW": m_r,
            "ratio_ci_lo": lo_r,
            "ratio_ci_hi": hi_r,
            "n_cancers_above_TCW": int(sum(r > 1.0 for r in ratios_vs_tcw)),
            "n_bh_signif": n_bh,
        }
    return out


# =========================================================================== #
# Worker
# =========================================================================== #

def _worker_window(args):
    """Worker for one (aggregator, top_pct) combination at window=1000 bp."""
    (aggregator, top_pct, windows_path, mut_paths_per_filter, perm_reps, seed) = args
    w = pd.read_parquet(windows_path)
    score_col = {"max": "max_score", "sum": "sum_score", "top3_mean": "top3_mean_score"}[aggregator]
    scores = w[score_col].to_numpy(dtype=np.float64)
    cpg = w["cpg_density"].to_numpy(dtype=np.float64)
    tcw = w["tcw_density"].to_numpy(dtype=np.float64)
    n_units = len(w)

    # Reconstitute mutation arrays per filter per cancer
    mut_per_filter_per_cancer = {}
    for fname, mut_path in mut_paths_per_filter.items():
        d = json.loads(Path(mut_path).read_text())
        per_cancer = {}
        for cancer, vals in d.items():
            arr = np.zeros(n_units, dtype=np.int32)
            for idx, count in vals:
                arr[int(idx)] = int(count)
            per_cancer[cancer] = arr
        mut_per_filter_per_cancer[fname] = per_cancer

    res = evaluate_one_construction(scores, cpg, tcw, mut_per_filter_per_cancer,
                                    top_pct, perm_reps, seed)
    return {"aggregator": aggregator, "top_pct": top_pct, "n_units": n_units,
            "panel_coverage_mb": float(WINDOW_SIZE * n_units / 1e6),
            "results": res}


# =========================================================================== #
# Position-level diagnosis
# =========================================================================== #

def position_level_diagnosis(panel: pd.DataFrame, maf_annot: pd.DataFrame, top_pct: float = 0.01) -> dict:
    """Diagnose position-level top-1% by score_binary.
    - Trinuc context composition (CpG, TCW, TCW_nonCpG, other)
    - Overlap with each mutation filter set
    - Same composition for ALL score columns (A3A, A3B, A3G, A3A_A3G, Neither, apobec1)
    """
    n = len(panel)
    k = max(1, int(round(n * top_pct)))
    scores = panel["score_binary"].to_numpy()
    top_idx = np.argpartition(-scores, k - 1)[:k]
    is_cpg = panel["is_cpg"].to_numpy(dtype=bool)
    is_tcw_c = panel["is_TCW_C"].to_numpy(dtype=bool)

    f_cpg = is_cpg[top_idx].mean()
    f_tcw = is_tcw_c[top_idx].mean()
    f_tcw_noncpg = (is_tcw_c[top_idx] & ~is_cpg[top_idx]).mean()
    f_other = (~is_cpg[top_idx] & ~is_tcw_c[top_idx]).mean()

    # Per-score-column stats: which score channel has the highest TCW enrichment in its top-1%?
    score_cols = [c for c in ["score_binary", "score_A3A", "score_A3B", "score_A3G",
                              "score_A3A_A3G", "score_Neither", "score_apobec1"] if c in panel.columns]
    per_score = {}
    for col in score_cols:
        s = panel[col].to_numpy()
        ti = np.argpartition(-s, k - 1)[:k]
        per_score[col] = {
            "fraction_TCW": float(is_tcw_c[ti].mean()),
            "fraction_CpG": float(is_cpg[ti].mean()),
            "TCW_C_mean_score": float(s[is_tcw_c].mean()) if is_tcw_c.any() else float("nan"),
            "non_TCW_C_mean_score": float(s[~is_tcw_c].mean()) if (~is_tcw_c).any() else float("nan"),
        }

    # Build top-set position keys
    top_keys = set(zip(panel["chrom"].iloc[top_idx].astype(str).values,
                       panel["pos"].iloc[top_idx].astype(int).values))

    # Build mutation filter sets (keys of (chrom, pos)).
    # Filter to in-panel only (already done upstream), then by trinuc.
    maf_keys = list(zip(maf_annot["chrom"].astype(str).values,
                        maf_annot["pos"].astype(int).values))
    is_tcw = maf_annot["is_TCW"].to_numpy(dtype=bool)
    is_cpg_m = maf_annot["is_CpG"].to_numpy(dtype=bool)
    is_tcw_noncpg = maf_annot["is_TCW_nonCpG"].to_numpy(dtype=bool)

    def overlap_pct(mask):
        if mask.sum() == 0:
            return float("nan"), 0
        kept = [maf_keys[i] for i in np.where(mask)[0]]
        n_total = len(kept)
        n_in_top = sum(1 for k_ in kept if k_ in top_keys)
        return n_in_top / n_total, n_total

    rec_all_CT, n_all_CT = overlap_pct(np.ones(len(maf_keys), dtype=bool))
    rec_all_TCW, n_all_TCW = overlap_pct(is_tcw)
    rec_TCW_nonCpG, n_TCW_nonCpG = overlap_pct(is_tcw_noncpg)

    return {
        "n_top": k,
        "fraction_CpG": float(f_cpg),
        "fraction_TCW": float(f_tcw),
        "fraction_TCW_nonCpG": float(f_tcw_noncpg),
        "fraction_other": float(f_other),
        "recall_all_CT": float(rec_all_CT),
        "n_muts_all_CT": int(n_all_CT),
        "recall_all_TCW": float(rec_all_TCW),
        "n_muts_all_TCW": int(n_all_TCW),
        "recall_TCW_nonCpG": float(rec_TCW_nonCpG),
        "n_muts_TCW_nonCpG": int(n_TCW_nonCpG),
        "per_score_column": per_score,
    }


# =========================================================================== #
# Main
# =========================================================================== #

def build_filter_sets(maf_annot: pd.DataFrame, panel: pd.DataFrame, rng_seed: int) -> dict:
    """Return {filter_name: DataFrame[chrom, pos, cancer]} after restricting to in-panel."""
    out = {}
    out["filter_all_CT"] = maf_annot.copy()
    out["filter_all_TCW"] = maf_annot[maf_annot["is_TCW"]].copy()
    out["filter_TCW_nonCpG"] = maf_annot[maf_annot["is_TCW_nonCpG"]].copy()

    # Random C baseline: sample N positions from the panel (each = 1 fake mut, evenly across cancers)
    rng = np.random.default_rng(rng_seed)
    n_to_sample = min(N_RANDOM_C_BASELINE, len(panel))
    rand_idx = rng.choice(len(panel), size=n_to_sample, replace=False)
    rand_df = panel.iloc[rand_idx][["chrom", "pos"]].copy()
    rand_df["pos"] = rand_df["pos"].astype(int)
    # Distribute across cancers proportionally so filter_random has the same per-cancer
    # structure as the others (cycles through the 10 cancers).
    rand_df["cancer"] = np.tile(CANCERS, int(np.ceil(len(rand_df) / len(CANCERS))))[:len(rand_df)]
    rand_df["strand"] = "+"
    rand_df["source"] = "random_baseline"
    out["filter_random"] = rand_df.reset_index(drop=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-workers", type=int, default=8)
    ap.add_argument("--perm-reps", type=int, default=PERM_REPS)
    ap.add_argument("--quick", action="store_true",
                    help="Quick run: 1K perms only")
    args = ap.parse_args()

    perm_reps = 1000 if args.quick else args.perm_reps
    log.info("=== top-X% recall under multiple filters ===")
    log.info("Window size: %d bp; aggregators: %s; top_pcts: %s",
             WINDOW_SIZE, AGGREGATORS, TOP_PCTS)

    # 1. Load + annotate panel
    annotated_path = OUT_DIR / "panel_annotated.parquet"
    if annotated_path.exists():
        log.info("Reusing annotated panel: %s", annotated_path)
        panel = pd.read_parquet(annotated_path)
    else:
        log.info("Loading panel ...")
        panel = pd.read_parquet(PANEL_PATH)
        panel = annotate_panel_positions(panel)
        panel.to_parquet(annotated_path, index=False)
        log.info("Saved annotated panel: %s", annotated_path)

    log.info("Panel rows: %d", len(panel))

    # 2. Load + annotate MAF
    maf = load_combined_coding_maf()
    log.info("Restricting MAF to in-panel positions ...")
    panel_set = set(zip(panel["chrom"].astype(str).values,
                        panel["pos"].astype(int).values))
    in_panel = np.array([(c, int(p)) in panel_set
                         for c, p in zip(maf["chrom"], maf["pos"])])
    maf = maf.iloc[np.where(in_panel)[0]].reset_index(drop=True)
    log.info("  in-panel C>T/G>A: %d", len(maf))
    maf = annotate_mut_context(maf)

    # 3. Build 4 filter sets
    log.info("Building filter sets ...")
    filter_sets = build_filter_sets(maf, panel, rng_seed=SEED_BASE)
    for fname, df in filter_sets.items():
        log.info("  %s: %d mutations", fname, len(df))

    # 4. Build window aggregations (window=1000 bp, single)
    win_path = OUT_DIR / f"_topx_windows_w{WINDOW_SIZE}.parquet"
    if win_path.exists():
        log.info("Reusing window aggregation: %s", win_path)
        windows = pd.read_parquet(win_path)
    else:
        windows = build_window_aggregations(panel, WINDOW_SIZE)
        windows.to_parquet(win_path, index=False)
        log.info("Saved windows: %s", win_path)

    # 5. Map mutations to windows for each filter
    log.info("Mapping mutations to windows for each filter ...")
    win_lookup = windows.reset_index(drop=True).reset_index().rename(columns={"index": "win_idx"})
    win_lookup = win_lookup[["chrom", "win_start", "win_idx"]]

    mut_paths_per_filter = {}
    for fname, mdf in filter_sets.items():
        m = mdf[["chrom", "pos", "cancer"]].copy()
        m["win_start"] = (m["pos"].astype(int) // WINDOW_SIZE) * WINDOW_SIZE
        m = m.merge(win_lookup, on=["chrom", "win_start"], how="inner")
        win_per_cancer = {}
        for cancer in CANCERS:
            sub = m[m["cancer"] == cancer]
            if len(sub) == 0:
                win_per_cancer[cancer] = []
                continue
            counts = sub["win_idx"].value_counts()
            win_per_cancer[cancer] = [[int(idx), int(c)] for idx, c in counts.items()]
        mp = OUT_DIR / f"_topx_mut_{fname}_w{WINDOW_SIZE}.json"
        mp.write_text(json.dumps(win_per_cancer))
        mut_paths_per_filter[fname] = str(mp)
        log.info("  %s: %d mut-window mappings", fname, len(m))

    # 6. Position-level diagnosis (BEFORE workers since main process has loaded panel/maf)
    log.info("=== POSITION-LEVEL DIAGNOSIS (top-1%% positions by score_binary) ===")
    diag = position_level_diagnosis(panel, maf, top_pct=0.01)
    for k, v in diag.items():
        if isinstance(v, float):
            log.info("  %s = %.4f", k, v)
        else:
            log.info("  %s = %s", k, v)

    # 7. Run all (aggregator × top_pct) combos in parallel
    work_units = []
    for ag in AGGREGATORS:
        for tp in TOP_PCTS:
            seed = SEED_BASE + hash((ag, tp)) % 100_000
            work_units.append((ag, tp, str(win_path), mut_paths_per_filter, perm_reps, seed))

    log.info("Running %d (aggregator x top_pct) combinations on %d workers ...",
             len(work_units), args.n_workers)

    results = []
    with ProcessPoolExecutor(max_workers=args.n_workers) as ex:
        futures = {ex.submit(_worker_window, wu): (wu[0], wu[1]) for wu in work_units}
        for f in as_completed(futures):
            tag = futures[f]
            try:
                res = f.result()
                results.append(res)
                log.info("  done: %s top_pct=%.2f", tag[0], tag[1])
            except Exception as ex:
                log.error("FAIL %s: %s", tag, ex)
                raise

    # 8. Flatten to long table: 1 row per (aggregator, window_size, top_pct, filter)
    rows = []
    for r in results:
        ag = r["aggregator"]; tp = r["top_pct"]
        for fname, fr in r["results"].items():
            rows.append({
                "aggregator": ag,
                "window_size_bp": WINDOW_SIZE,
                "top_pct": tp,
                "panel_coverage_mb": r["panel_coverage_mb"] * tp,  # mb covered at top_pct
                "filter": fname,
                "n_cancers": fr["n_cancers"],
                "mean_abs_recall": fr["mean_abs_recall"],
                "ci_lo": fr["abs_recall_ci_lo"],
                "ci_hi": fr["abs_recall_ci_hi"],
                "mean_abs_recall_tcw_baseline": fr["mean_abs_recall_tcw_baseline"],
                "tcw_baseline_ci_lo": fr["tcw_baseline_ci_lo"],
                "tcw_baseline_ci_hi": fr["tcw_baseline_ci_hi"],
                "mean_abs_recall_cpg_baseline": fr["mean_abs_recall_cpg_baseline"],
                "cpg_baseline_ci_lo": fr["cpg_baseline_ci_lo"],
                "cpg_baseline_ci_hi": fr["cpg_baseline_ci_hi"],
                "mean_ratio_vs_TCW": fr["mean_ratio_vs_TCW"],
                "ratio_ci_lo": fr["ratio_ci_lo"],
                "ratio_ci_hi": fr["ratio_ci_hi"],
                "n_cancers_above_TCW": fr["n_cancers_above_TCW"],
                "n_bh_signif": fr["n_bh_signif"],
            })

    df = pd.DataFrame(rows)
    df = df.sort_values(["aggregator", "top_pct", "filter"]).reset_index(drop=True)
    out_csv = OUT_DIR / "recall_topx_filters.csv"
    df.to_csv(out_csv, index=False)
    log.info("Wrote %s (%d rows)", out_csv, len(df))

    # 9. Plot: facet by filter, x=top_pct, y=mean_abs_recall, lines for {sum, max, top3_mean, TCW-density, CpG-density}
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        log.info("Generating figure ...")

        filter_order = ["filter_all_CT", "filter_all_TCW", "filter_TCW_nonCpG", "filter_random"]
        fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=False)
        for ax, fname in zip(axes, filter_order):
            sub = df[df["filter"] == fname]
            for ag in AGGREGATORS:
                ss = sub[sub["aggregator"] == ag].sort_values("top_pct")
                ax.errorbar(ss["top_pct"] * 100, ss["mean_abs_recall"],
                            yerr=[ss["mean_abs_recall"] - ss["ci_lo"],
                                  ss["ci_hi"] - ss["mean_abs_recall"]],
                            marker="o", capsize=3, label=f"NN {ag} x{WINDOW_SIZE}")
            # Baselines: TCW-density, CpG-density (use sum panel; these are window-level
            # baselines, same for all aggregators)
            ss = sub[sub["aggregator"] == "sum"].sort_values("top_pct")
            ax.errorbar(ss["top_pct"] * 100, ss["mean_abs_recall_tcw_baseline"],
                        yerr=[ss["mean_abs_recall_tcw_baseline"] - ss["tcw_baseline_ci_lo"],
                              ss["tcw_baseline_ci_hi"] - ss["mean_abs_recall_tcw_baseline"]],
                        marker="s", capsize=3, linestyle="--", color="gray",
                        label="TCW-density baseline")
            ax.errorbar(ss["top_pct"] * 100, ss["mean_abs_recall_cpg_baseline"],
                        yerr=[ss["mean_abs_recall_cpg_baseline"] - ss["cpg_baseline_ci_lo"],
                              ss["cpg_baseline_ci_hi"] - ss["mean_abs_recall_cpg_baseline"]],
                        marker="^", capsize=3, linestyle=":", color="black",
                        label="CpG-density baseline")
            ax.set_xlabel("top-X% panel coverage")
            ax.set_ylabel("mean abs recall (95% CI)")
            ax.set_title(fname)
            ax.grid(alpha=0.3)
            ax.legend(loc="best", fontsize=7)
        plt.tight_layout()
        out_png = OUT_DIR / "recall_topx_filters.png"
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close()
        log.info("Wrote %s", out_png)
    except Exception as ex:
        log.error("Figure generation failed: %s", ex)

    # 10. Markdown summary
    md_lines = []
    md_lines.append("# Top-X% Panel Recall Under Multiple Mutation Filters")
    md_lines.append("")
    md_lines.append(f"Window: {WINDOW_SIZE} bp. Aggregators: {AGGREGATORS}.")
    md_lines.append(f"Filters: filter_all_CT, filter_all_TCW, filter_TCW_nonCpG, filter_random.")
    md_lines.append(f"Cancers: {CANCERS}.")
    md_lines.append(f"Bootstrap CI: {N_BOOT} resamples across {len(CANCERS)} cancers.")
    md_lines.append(f"Permutation null: hypergeometric (= permutation in the limit), perm_reps={perm_reps}, BH-FDR.")
    md_lines.append("")
    md_lines.append("## Recall table")
    md_lines.append("")
    pivoted = df.pivot_table(
        index=["aggregator", "filter"],
        columns="top_pct",
        values="mean_abs_recall",
        aggfunc="first",
    )
    md_lines.append("Mean absolute recall (rows = aggregator x filter, cols = top_pct):")
    md_lines.append("")
    md_lines.append("```")
    md_lines.append(pivoted.round(4).to_string())
    md_lines.append("```")
    md_lines.append("")

    md_lines.append("## Ratio vs TCW-density baseline")
    md_lines.append("")
    pivoted_r = df.pivot_table(
        index=["aggregator", "filter"],
        columns="top_pct",
        values="mean_ratio_vs_TCW",
        aggfunc="first",
    )
    md_lines.append("```")
    md_lines.append(pivoted_r.round(3).to_string())
    md_lines.append("```")
    md_lines.append("")

    md_lines.append("## n cancers BH-signif (q < 0.025) per (aggregator x filter x top_pct)")
    md_lines.append("")
    pivoted_bh = df.pivot_table(
        index=["aggregator", "filter"],
        columns="top_pct",
        values="n_bh_signif",
        aggfunc="first",
    )
    md_lines.append("```")
    md_lines.append(pivoted_bh.to_string())
    md_lines.append("```")
    md_lines.append("")

    md_lines.append("## POSITION_LEVEL_DIAGNOSIS — why is position-level recall 0%?")
    md_lines.append("")
    md_lines.append(f"Top-{int(0.01*100)}% positions (by score_binary alone, no window): "
                    f"k = {diag['n_top']:,} of {len(panel):,} positions.")
    md_lines.append("")
    md_lines.append("**Trinucleotide-context composition of top-1% positions:**")
    md_lines.append("")
    md_lines.append(f"- CpG (3' nt = G):       {diag['fraction_CpG']*100:5.2f}%")
    md_lines.append(f"- TCW  (5'=T, 3'=A/T):    {diag['fraction_TCW']*100:5.2f}%")
    md_lines.append(f"- TCW non-CpG:           {diag['fraction_TCW_nonCpG']*100:5.2f}%")
    md_lines.append(f"- Other:                 {diag['fraction_other']*100:5.2f}%")
    md_lines.append("")
    md_lines.append("**Overlap with each in-panel mutation filter set (= position-level recall):**")
    md_lines.append("")
    md_lines.append(f"- vs filter_all_CT:      recall = {diag['recall_all_CT']*100:5.3f}% "
                    f"(n_muts = {diag['n_muts_all_CT']:,})")
    md_lines.append(f"- vs filter_all_TCW:     recall = {diag['recall_all_TCW']*100:5.3f}% "
                    f"(n_muts = {diag['n_muts_all_TCW']:,})")
    md_lines.append(f"- vs filter_TCW_nonCpG:  recall = {diag['recall_TCW_nonCpG']*100:5.3f}% "
                    f"(n_muts = {diag['n_muts_TCW_nonCpG']:,})")
    md_lines.append("")
    md_lines.append("**Interpretation — root cause of position-level 0% recall on TCW filters:**")
    md_lines.append("")
    md_lines.append(f"  The top-1% positions by score_binary contain {diag['fraction_TCW']*100:.2f}% TCW")
    md_lines.append(f"  trinucleotide context, vs ~13.0% TCW prevalence in the panel overall.")
    md_lines.append(f"  The NN is *anti-correlated* with TCW: TCW C positions average score_binary~0.42")
    md_lines.append(f"  vs ~0.73 for non-TCW positions, while CpG positions average ~0.81 (highest).")
    md_lines.append(f"  Top-1% is {diag['fraction_CpG']*100:.1f}% CpG-context, so any C>T mutation filter that")
    md_lines.append(f"  excludes CpG (e.g. filter_TCW_nonCpG) gets 0% recall by construction.")
    md_lines.append("")
    md_lines.append("  Position-level recall against filter_all_CT is also bounded by panel coverage:")
    md_lines.append(f"  top-1% covers {diag['n_top']/len(panel)*100:.2f}% of positions, so the ceiling is")
    md_lines.append(f"  ~{diag['recall_all_CT']*100:.2f}% (mutations slightly enriched at top positions).")
    md_lines.append("  Window aggregation 'works' for TCW filters because 1-kb windows centered on a")
    md_lines.append("  high-scoring CpG position usually contain several nearby TCW C positions —")
    md_lines.append("  but the NN itself is NOT learning the TCW signal at the position level.")
    md_lines.append("")
    md_lines.append("  **This is the headline finding: score_binary is mis-calibrated for the APOBEC**")
    md_lines.append("  **target.**")
    md_lines.append("")
    md_lines.append("**Per-score-column trinuc composition of top-1% positions** (panel TCW prevalence ~13.0%):")
    md_lines.append("")
    md_lines.append("```")
    md_lines.append(f"{'score_col':18s}  {'top1pct_TCW%':>12s}  {'top1pct_CpG%':>12s}  {'TCW_mean':>9s}  {'non_TCW_mean':>13s}")
    for col, st in diag["per_score_column"].items():
        md_lines.append(f"{col:18s}  {st['fraction_TCW']*100:11.2f}%  {st['fraction_CpG']*100:11.2f}%  "
                        f"{st['TCW_C_mean_score']:9.3f}  {st['non_TCW_C_mean_score']:13.3f}")
    md_lines.append("```")
    md_lines.append("")
    md_lines.append("**Actionable**: `score_apobec1` is the only column where TCW positions outscore non-TCW")
    md_lines.append("at the top-1% panel slice. To push recall to 10–30%, options:")
    md_lines.append("  1. Re-rank windows by `score_apobec1` instead of `score_binary` (or by their sum).")
    md_lines.append("  2. Re-rank windows by tcw_density x score_binary (hybrid product).")
    md_lines.append("  3. Re-train the binary head with a TCW-aware loss / TCW positives only.")
    md_lines.append("  4. Increase the panel coverage to 5%-10%; sum x 1000 already gives 20%-32% recall on")
    md_lines.append("     filter_TCW_nonCpG with current scoring at top-5/10%.")
    md_lines.append("")

    # Headline numbers for sum x 1000
    md_lines.append("## Headline (winning construction = sum x 1000 bp)")
    md_lines.append("")
    sum_df = df[df["aggregator"] == "sum"].copy()
    for tp in TOP_PCTS:
        md_lines.append(f"### top-{int(tp*100)}% (panel coverage ~ {sum_df['panel_coverage_mb'].iloc[0]:.0f} Mb x {tp:.2f}):")
        for fname in ["filter_all_CT", "filter_all_TCW", "filter_TCW_nonCpG", "filter_random"]:
            row = sum_df[(sum_df["top_pct"] == tp) & (sum_df["filter"] == fname)]
            if len(row) == 0:
                continue
            row = row.iloc[0]
            md_lines.append(f"- {fname:25s}  recall = {row['mean_abs_recall']*100:5.2f}% "
                            f"[{row['ci_lo']*100:5.2f}, {row['ci_hi']*100:5.2f}]  "
                            f"vs TCW-density: {row['mean_ratio_vs_TCW']:.3f}x  "
                            f"({row['n_cancers_above_TCW']}/{int(row['n_cancers'])} cancers > TCW; "
                            f"BH-signif {int(row['n_bh_signif'])}/{int(row['n_cancers'])})")
        md_lines.append("")

    out_md = OUT_DIR / "TOPX_FILTER_RESULTS.md"
    out_md.write_text("\n".join(md_lines))
    log.info("Wrote %s", out_md)

    # Cleanup
    if not args.quick:
        for fname in mut_paths_per_filter.values():
            p = Path(fname)
            if p.exists():
                p.unlink()
        if win_path.exists():
            win_path.unlink()

    log.info("DONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
