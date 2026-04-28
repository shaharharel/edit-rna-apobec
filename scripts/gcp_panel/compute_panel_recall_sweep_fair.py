#!/usr/bin/env python3
"""FAIR remediation of the 21-construction panel sweep.

Addresses three QA findings that inflated the NN's apparent advantage in
`compute_panel_recall_sweep.py` and `sweep_all_heads.py`:

  1. **Same-bases baselines.** Original code computed CpG- and TCW-density
     baselines from the FULL hg19 window sequence (introns+intergenic), giving
     the NN — which only sums over panel CDS-C positions — an unfair scaffold.
     Here we recompute both densities by counting the number of panel rows
     (CDS-C positions) within each window where ``is_cpg`` / ``is_TCW_C`` is
     true (per-position trinucleotide context recomputed from hg19).

  2. **n_pos-only baseline.** Adds a baseline that ranks windows purely by the
     number of panel CDS-C positions they contain (no model, no motif).
     Diagnostic: tests whether the headline 1.31x ratio is just gene-body
     density.

  3. **All 7 heads × all 21 constructions = 147 cells.** Iterates every
     ``score_*`` column over the 1 position-level + 4 windows x 5 aggregators
     constructions (replacing the previous "all_heads_sweep" subset that
     evaluated only 3 constructions x 7 heads).

  4. **Real shuffle permutation null.** Replaces hypergeometric closed-form
     p-values. For each (window-size, cancer, filter) we sample 10K random
     k-subsets without replacement and compute the null distribution of
     ``mut[draw].sum()`` once, then reuse across all 35 (5 aggs x 7 heads)
     constructions at that window size.

  5. **Bonferroni across 1470 tests.** 147 (head, agg, ws) cells x 10 cancers
     = 1470 tests. q_threshold = 0.05 / 1470 = 3.4e-5. Reports cells where
     >=1 cancer (and majority) survive.

Mutation filters (4): filter_TCW_nonCpG (default), filter_all_TCW,
filter_all_CT, filter_random_C.

Outputs:
  experiments/multi_enzyme/outputs/pcawg_tcw_panel/sweep_v3_fair.csv
  experiments/multi_enzyme/outputs/pcawg_tcw_panel/sweep_v3_fair_per_cancer.csv
  experiments/multi_enzyme/outputs/pcawg_tcw_panel/sweep_v3_fair.png
  experiments/multi_enzyme/outputs/pcawg_tcw_panel/SWEEP_V3_FAIR_RESULTS.md

macOS-safe multiprocessing via concurrent.futures.ProcessPoolExecutor.
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/shaharharel/Documents/github/edit-rna-apobec")
PANEL_PATH = ROOT / "experiments/multi_enzyme/outputs/pcawg_tcw_panel/panel_scores_cds.parquet"
TCGA_DIR = ROOT / "data/raw/tcga"
PCAWG_DIR = ROOT / "data/raw/pcawg/by_cancer"
HG19 = ROOT / "data/raw/genomes/hg19.fa"
OUT_DIR = ROOT / "experiments/multi_enzyme/outputs/pcawg_tcw_panel"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CANCERS = ["blca", "brca", "cesc", "coadread", "esca", "hnsc", "lihc",
           "lusc", "skcm", "stad"]
WINDOW_SIZES = [100, 250, 500, 1000]
AGGREGATORS = ["max", "mean", "sum", "top3_mean", "p95"]
HEADS = ["score_binary", "score_A3A", "score_A3B", "score_A3G",
         "score_A3A_A3G", "score_Neither", "score_apobec1"]
FILTERS = ["filter_TCW_nonCpG", "filter_all_TCW", "filter_all_CT",
           "filter_random_C"]
TOP_PCT = 0.01
# 10K perms gives a p_perm floor of 1/(N+1) = ~1e-4. That floor is ABOVE the
# Bonferroni q = 3.4e-5 (147 cells x 10 cancers); to resolve significance at
# Bonferroni we need >= 29411 perms. Default to 30K so the floor (3.3e-5)
# clears Bonferroni. 10K still works for the headline ratio_vs_TCW reading;
# 30K only matters for the bonf_signif column.
PERM_REPS = 30000
N_BOOT = 10000
SEED_BASE = 20260427

# Bonferroni: 21 constructions x 7 heads = 147 cells x 10 cancers = 1470 tests
BONFERRONI_N_TESTS = 21 * 7 * 10
ALPHA = 0.05
BONF_Q = ALPHA / BONFERRONI_N_TESTS  # = 3.4e-5

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    stream=sys.stdout)
log = logging.getLogger(__name__)


# =========================================================================== #
# Mutation loading (TCGA + PCAWG-coding combined). Reused from
# compute_panel_recall_sweep.py.
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
    df["pos"] = df["Start_Position"].astype(int) - 1
    df["chrom"] = df["Chromosome"].astype(str)
    df.loc[~df["chrom"].str.startswith("chr"), "chrom"] = "chr" + df["chrom"]
    df["cancer"] = cancer
    df["source"] = source
    return df[["chrom", "pos", "strand", "cancer", "source"]]


def load_combined_coding_maf() -> pd.DataFrame:
    log.info("Loading TCGA-MC3 + cBioPortal-PCAWG coding MAFs ...")
    rows = []
    for cancer in CANCERS:
        d = _load_one_maf(PCAWG_DIR / f"{cancer}_pcawg_mutations.txt",
                          cancer, "pcawg_coding")
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
# Annotate panel positions and mutations from hg19 trinucleotide context
# =========================================================================== #

def annotate_panel_positions(panel: pd.DataFrame) -> pd.DataFrame:
    """Add is_cpg + is_TCW_C + gc_bin columns to each panel row using hg19
    trinucleotide context. THIS IS THE CRITICAL FIX: ``is_cpg``/``is_TCW_C``
    are recomputed from the position's strand-corrected trinucleotide, NOT
    inherited from any pre-computed window-seq column.
    """
    from pyfaidx import Fasta
    log.info("Annotating panel positions with is_cpg + is_TCW_C (hg19 trinuc) ...")
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
            seq = np.frombuffer(str(genome[ch][:]).upper().encode("ascii"),
                                dtype=np.uint8)
        except Exception:
            log.warning("  couldn't load %s", ch)
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
        # CpG: + strand C => right == 'G'; - strand G => left == 'C'.
        is_cpg[valid_idx[is_plus]] = (right[is_plus] == ord("G"))
        is_cpg[valid_idx[is_minus]] = (left[is_minus] == ord("C"))
        # TCW-C: + strand: left==T AND right in {A,T}.
        # - strand: right==A AND left in {A,T}.
        right_AT = (right == ord("A")) | (right == ord("T"))
        left_AT = (left == ord("A")) | (left == ord("T"))
        is_tcw_c[valid_idx[is_plus]] = (left[is_plus] == ord("T")) & right_AT[is_plus]
        is_tcw_c[valid_idx[is_minus]] = (right[is_minus] == ord("A")) & left_AT[is_minus]

        # GC bin in +/- 50 bp via cumulative GC count.
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
    log.info("  Sanity: panel n=%d, is_cpg=%d (%.2f%%), is_TCW_C=%d (%.2f%%)",
             n, is_cpg.sum(), 100 * is_cpg.mean(),
             is_tcw_c.sum(), 100 * is_tcw_c.mean())
    return panel


def annotate_mut_context(maf: pd.DataFrame) -> pd.DataFrame:
    """Add is_TCW, is_CpG, is_TCW_nonCpG flags using hg19 trinuc context."""
    from pyfaidx import Fasta
    log.info("Annotating mutations with hg19 trinuc context ...")
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
            seq = np.frombuffer(str(genome[ch][:]).upper().encode("ascii"),
                                dtype=np.uint8)
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
        plus_tcw = is_plus & (left == ord("T")) & (center == ord("C")) & (
            (right == ord("A")) | (right == ord("T")))
        minus_tcw = is_minus & (right == ord("A")) & (center == ord("G")) & (
            (left == ord("A")) | (left == ord("T")))
        plus_cpg = is_plus & (center == ord("C")) & (right == ord("G"))
        minus_cpg = is_minus & (center == ord("G")) & (left == ord("C"))
        is_tcw[valid_idx] = plus_tcw | minus_tcw
        is_cpg[valid_idx] = plus_cpg | minus_cpg

    out = maf.copy()
    out["is_TCW"] = is_tcw
    out["is_CpG"] = is_cpg
    out["is_TCW_nonCpG"] = is_tcw & ~is_cpg
    log.info("  total=%d  TCW=%d (%.1f%%)  CpG=%d (%.1f%%)  TCW_nonCpG=%d (%.1f%%)",
             n, is_tcw.sum(), 100 * is_tcw.mean(),
             is_cpg.sum(), 100 * is_cpg.mean(),
             out["is_TCW_nonCpG"].sum(), 100 * out["is_TCW_nonCpG"].mean())
    return out


# =========================================================================== #
# Build window aggregations: ALL 7 heads x 5 aggregators + same-bases baselines
# =========================================================================== #

def _top3_mean_groupby(p_sorted: pd.DataFrame,
                       group_keys: list[str], col: str) -> pd.Series:
    """Compute top-3 mean per group using grouped sort."""
    grp = p_sorted.groupby(group_keys, sort=False)[col]
    return grp.apply(lambda v: float(np.mean(np.partition(v.values, -3)[-3:]))
                     if len(v) > 3 else float(v.mean()) if len(v) > 0 else 0.0)


def build_window_aggregations(panel: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """Build per-window aggregations across all 7 heads x 5 aggregators, plus
    same-bases baselines (cpg_density_panel, tcw_density_panel, n_pos_panel),
    plus the median GC bin per window.

    SAME-BASES BASELINES (the QA fix):
      * ``cpg_density_panel`` = number of panel rows in window where is_cpg
      * ``tcw_density_panel`` = number of panel rows in window where is_TCW_C
      * ``n_pos_panel`` = number of panel rows in window
    Counted from the EXACT same units the NN sums over (CDS-C positions only),
    not from the full window sequence.
    """
    log.info("[w=%d] Building windows (all 7 heads x 5 aggs + same-bases baselines) ...",
             window_size)
    p = panel.copy()
    p["pos"] = p["pos"].astype(int)
    p["win_start"] = (p["pos"] // window_size) * window_size

    # 1. Compute simple per-group aggs: max, mean, sum, p95 (use named agg).
    agg_dict = {
        "n_pos_panel":   ("pos", "size"),
        "cpg_density_panel": ("is_cpg", "sum"),
        "tcw_density_panel": ("is_TCW_C", "sum"),
        "gc_bin": ("gc_bin", "median"),
    }
    for h in HEADS:
        agg_dict[f"{h}__max"] = (h, "max")
        agg_dict[f"{h}__mean"] = (h, "mean")
        agg_dict[f"{h}__sum"] = (h, "sum")
    grp = p.groupby(["chrom", "win_start"])
    out = grp.agg(**agg_dict).reset_index()
    out["gc_bin"] = out["gc_bin"].astype(int)

    # 2. p95 per (group, head): use groupby quantile (vectorized).
    log.info("[w=%d] Computing p95 per head ...", window_size)
    for h in HEADS:
        q95 = grp[h].quantile(0.95).rename(f"{h}__p95").reset_index()
        out = out.merge(q95, on=["chrom", "win_start"])

    # 3. top3_mean per (group, head): vectorized via partition.
    log.info("[w=%d] Computing top3_mean per head ...", window_size)
    p_sorted = p.sort_values(["chrom", "win_start"], kind="stable")
    for h in HEADS:
        s = _top3_mean_groupby(p_sorted, ["chrom", "win_start"], h).rename(
            f"{h}__top3_mean").reset_index()
        out = out.merge(s, on=["chrom", "win_start"])

    log.info("[w=%d] %d windows; mean cpg_density_panel=%.2f, tcw_density_panel=%.2f, "
             "n_pos_panel=%.2f", window_size, len(out),
             out["cpg_density_panel"].mean(),
             out["tcw_density_panel"].mean(),
             out["n_pos_panel"].mean())
    return out


# =========================================================================== #
# Permutation null distributions: real shuffle, 10K reps. Cached per
# (window_size, k, cancer, filter).
# =========================================================================== #

def _build_null_dist(n_units: int, k: int, mut: np.ndarray,
                     n_reps: int, seed: int) -> np.ndarray:
    """Sample n_reps random k-subsets of n_units WITHOUT replacement; record
    mut[draw].sum() for each. Returns the null distribution (length n_reps).
    """
    rng = np.random.default_rng(seed)
    if int(mut.sum()) == 0:
        return np.zeros(n_reps, dtype=np.int64)
    out = np.empty(n_reps, dtype=np.int64)
    for i in range(n_reps):
        idx = rng.choice(n_units, size=k, replace=False)
        out[i] = int(mut[idx].sum())
    return out


def _build_null_dist_worker(args):
    (n_units, k, mut_compact, n_reps, seed) = args
    # mut_compact: sparse representation [(idx, count), ...]. Reconstruct.
    mut = np.zeros(n_units, dtype=np.int32)
    for i, c in mut_compact:
        mut[int(i)] = int(c)
    null = _build_null_dist(n_units, k, mut, n_reps, seed)
    return null


def _null_worker_from_path(args):
    """Worker that loads ONE cancer's mut array from disk and samples nulls.
    Used for parallel null-distribution build at scale (4 ws + position x
    4 filters x 10 cancers = up to 200 jobs).
    """
    (level_id, filter_name, cancer, n_units, k, mut_path, n_reps, seed) = args
    mc = np.load(mut_path)
    if cancer not in mc.files:
        return np.zeros(n_reps, dtype=np.int64)
    mut = mc[cancer]
    if int(mut.sum()) == 0:
        return np.zeros(n_reps, dtype=np.int64)
    return _build_null_dist(n_units, k, mut, n_reps, seed)


# =========================================================================== #
# Per-cell evaluator: given scores + same-bases baselines + mut + null_dist,
# compute observed recall, ratios, p-perm.
# =========================================================================== #

def evaluate_cell(scores: np.ndarray, base_cpg: np.ndarray, base_tcw: np.ndarray,
                  base_npos: np.ndarray, mut_per_cancer: dict[str, np.ndarray],
                  null_per_cancer: dict[str, np.ndarray], k: int) -> dict:
    """Evaluate one (head, agg, ws, filter) cell across all 10 cancers.

    Args:
      scores        : (n,) score vector (head x aggregator)
      base_cpg      : (n,) same-bases CpG-density (panel rows with is_cpg)
      base_tcw      : (n,) same-bases TCW-density (panel rows with is_TCW_C)
      base_npos     : (n,) panel-position count per unit
      mut_per_cancer: {cancer -> (n,) int counts under one filter}
      null_per_cancer: {cancer -> (n_reps,) null mut_in_top under H0 for this filter}
      k             : top-k cutoff
    """
    n = len(scores)

    # Observed top-k indices for each ranking
    nn_top = np.argpartition(-scores, k - 1)[:k]
    cpg_top = np.argpartition(-base_cpg, k - 1)[:k]
    tcw_top = np.argpartition(-base_tcw, k - 1)[:k]
    npos_top = np.argpartition(-base_npos, k - 1)[:k]

    per_cancer = {}
    abs_recalls = []
    ratios_cpg, ratios_tcw, ratios_npos = [], [], []
    n_above_cpg = n_above_tcw = n_above_npos = 0
    p_perms = []

    for cancer, mut in mut_per_cancer.items():
        total = int(mut.sum())
        if total == 0:
            per_cancer[cancer] = {"total_mut": 0, "abs_recall": float("nan"),
                                  "abs_recall_cpg": float("nan"),
                                  "abs_recall_tcw": float("nan"),
                                  "abs_recall_npos": float("nan"),
                                  "ratio_cpg": float("nan"),
                                  "ratio_tcw": float("nan"),
                                  "ratio_npos": float("nan"),
                                  "p_perm": 1.0,
                                  "k": int(k), "n_units": int(n)}
            continue
        nn_recall = float(mut[nn_top].sum()) / total
        cpg_recall = float(mut[cpg_top].sum()) / total
        tcw_recall = float(mut[tcw_top].sum()) / total
        npos_recall = float(mut[npos_top].sum()) / total
        ratio_cpg = nn_recall / cpg_recall if cpg_recall > 0 else float("nan")
        ratio_tcw = nn_recall / tcw_recall if tcw_recall > 0 else float("nan")
        ratio_npos = nn_recall / npos_recall if npos_recall > 0 else float("nan")

        # Real-shuffle p-value: fraction of perms with mut_in_top >= observed
        null = null_per_cancer[cancer]
        mut_in_top_obs = int(mut[nn_top].sum())
        p_perm = float((null >= mut_in_top_obs).sum() + 1) / (len(null) + 1)

        per_cancer[cancer] = {
            "total_mut": total,
            "abs_recall": nn_recall,
            "abs_recall_cpg": cpg_recall,
            "abs_recall_tcw": tcw_recall,
            "abs_recall_npos": npos_recall,
            "ratio_cpg": ratio_cpg,
            "ratio_tcw": ratio_tcw,
            "ratio_npos": ratio_npos,
            "mut_in_top_obs": mut_in_top_obs,
            "p_perm": p_perm,
            "k": int(k),
            "n_units": int(n),
        }
        abs_recalls.append(nn_recall)
        if not np.isnan(ratio_cpg):
            ratios_cpg.append(ratio_cpg)
            n_above_cpg += int(ratio_cpg > 1.0)
        if not np.isnan(ratio_tcw):
            ratios_tcw.append(ratio_tcw)
            n_above_tcw += int(ratio_tcw > 1.0)
        if not np.isnan(ratio_npos):
            ratios_npos.append(ratio_npos)
            n_above_npos += int(ratio_npos > 1.0)
        p_perms.append(p_perm)

    # Bootstrap CI for ratios + abs_recall over the (<=10) cancers
    def boot_ci(arr, seed_):
        if not arr:
            return float("nan"), float("nan"), float("nan")
        a = np.asarray(arr, dtype=float)
        a = a[~np.isnan(a)]
        if len(a) == 0:
            return float("nan"), float("nan"), float("nan")
        rng = np.random.default_rng(seed_)
        n_a = len(a)
        idx = rng.integers(0, n_a, size=(N_BOOT, n_a))
        boot = a[idx].mean(axis=1)
        return float(a.mean()), float(np.percentile(boot, 2.5)), \
               float(np.percentile(boot, 97.5))

    m_abs, lo_abs, hi_abs = boot_ci(abs_recalls, 1)
    m_cpg, lo_cpg, hi_cpg = boot_ci(ratios_cpg, 2)
    m_tcw, lo_tcw, hi_tcw = boot_ci(ratios_tcw, 3)
    m_npos, lo_npos, hi_npos = boot_ci(ratios_npos, 4)

    n_bonf = int(sum(p < BONF_Q for p in p_perms))

    return {
        "per_cancer": per_cancer,
        "n_cancers": len(per_cancer),
        "mean_abs_recall": m_abs, "abs_recall_ci_lo": lo_abs, "abs_recall_ci_hi": hi_abs,
        "mean_ratio_cpg": m_cpg, "ratio_cpg_ci_lo": lo_cpg, "ratio_cpg_ci_hi": hi_cpg,
        "mean_ratio_tcw": m_tcw, "ratio_tcw_ci_lo": lo_tcw, "ratio_tcw_ci_hi": hi_tcw,
        "mean_ratio_npos": m_npos, "ratio_npos_ci_lo": lo_npos, "ratio_npos_ci_hi": hi_npos,
        "n_above_cpg": n_above_cpg,
        "n_above_tcw": n_above_tcw,
        "n_above_npos": n_above_npos,
        "n_bonf_signif": n_bonf,
        "k": int(k), "n_units": int(n),
    }


# =========================================================================== #
# Worker: one (window_size, head, aggregator, filter) cell.
# Receives windows + nulls via on-disk paths to avoid pickling 8.45M-row DFs.
# =========================================================================== #

def _process_cell(args):
    (level, head, aggregator, window_size, filter_name,
     scores_path, baselines_path, mut_path, null_path, top_pct) = args
    # Load scores (one column) and baselines.
    scores = np.load(scores_path)["scores"]
    bl = np.load(baselines_path)
    base_cpg = bl["cpg"]
    base_tcw = bl["tcw"]
    base_npos = bl["npos"]
    mut_per_cancer = {}
    mc_npz = np.load(mut_path)
    for cancer in CANCERS:
        if cancer in mc_npz.files:
            mut_per_cancer[cancer] = mc_npz[cancer]
    null_per_cancer = {}
    n_npz = np.load(null_path)
    for cancer in CANCERS:
        if cancer in n_npz.files:
            null_per_cancer[cancer] = n_npz[cancer]

    n = len(scores)
    k = max(1, int(round(n * top_pct)))

    res = evaluate_cell(scores, base_cpg, base_tcw, base_npos,
                        mut_per_cancer, null_per_cancer, k)
    res["level"] = level
    res["head"] = head
    res["aggregator"] = aggregator
    res["window_size_bp"] = window_size
    res["filter"] = filter_name
    return res


# =========================================================================== #
# Filter set construction
# =========================================================================== #

def build_filter_sets(maf: pd.DataFrame, panel: pd.DataFrame,
                      seed: int) -> dict[str, pd.DataFrame]:
    """Return {filter_name: DataFrame[chrom, pos, cancer]}.
    filter_random_C: 100K random C positions sampled from the panel, then
    distributed across 10 cancers proportional to the real distribution.
    """
    log.info("Building filter sets ...")
    out = {}
    out["filter_TCW_nonCpG"] = maf[maf["is_TCW_nonCpG"]].copy()
    out["filter_all_TCW"] = maf[maf["is_TCW"]].copy()
    out["filter_all_CT"] = maf.copy()
    rng = np.random.default_rng(seed)
    n_to_sample = min(100_000, len(panel))
    rand_idx = rng.choice(len(panel), size=n_to_sample, replace=False)
    rand_df = panel.iloc[rand_idx][["chrom", "pos"]].copy()
    rand_df["pos"] = rand_df["pos"].astype(int)
    # Distribute across cancers proportionally to real per-cancer counts
    cancer_props = maf["cancer"].value_counts(normalize=True).to_dict()
    cancer_assign = []
    for c, frac in cancer_props.items():
        nc = int(round(n_to_sample * frac))
        cancer_assign += [c] * nc
    while len(cancer_assign) < n_to_sample:
        cancer_assign.append(CANCERS[0])
    cancer_assign = cancer_assign[:n_to_sample]
    rand_df["cancer"] = cancer_assign
    rand_df["strand"] = "+"
    rand_df["source"] = "random_baseline"
    out["filter_random_C"] = rand_df.reset_index(drop=True)
    for fn, fdf in out.items():
        log.info("  %s: %d mutations", fn, len(fdf))
    return out


# =========================================================================== #
# Main
# =========================================================================== #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-workers", type=int, default=8)
    ap.add_argument("--perm-reps", type=int, default=PERM_REPS)
    ap.add_argument("--top-pct", type=float, default=TOP_PCT)
    ap.add_argument("--quick", action="store_true",
                    help="Quick run: 1K perms, only ws=250, only ws=1000.")
    args = ap.parse_args()
    n_workers = min(8, args.n_workers)

    perm_reps = 1000 if args.quick else args.perm_reps
    window_sizes = [1000] if args.quick else WINDOW_SIZES
    log.info("=== Fair sweep v3 ===")
    log.info("Window sizes: %s; aggregators: %s; heads: %s; filters: %s",
             window_sizes, AGGREGATORS, HEADS, FILTERS)
    log.info("Top pct: %.3f; perm_reps: %d; Bonferroni q=%.2e (n_tests=%d)",
             args.top_pct, perm_reps, BONF_Q, BONFERRONI_N_TESTS)

    t_start = time.time()

    # 1. Load + annotate panel.
    log.info("Loading panel ...")
    panel = pd.read_parquet(PANEL_PATH)
    log.info("  panel rows: %d", len(panel))
    panel = annotate_panel_positions(panel)

    # 2. Load + annotate MAFs; restrict to in-panel; build filter sets.
    maf = load_combined_coding_maf()
    log.info("Restricting MAF to in-panel positions ...")
    panel_set = set(zip(panel["chrom"].astype(str).values,
                        panel["pos"].astype(int).values))
    in_panel = np.array([(c, int(p)) in panel_set
                         for c, p in zip(maf["chrom"], maf["pos"])])
    maf = maf.iloc[np.where(in_panel)[0]].reset_index(drop=True)
    log.info("  in-panel C>T/G>A: %d", len(maf))
    maf = annotate_mut_context(maf)
    filter_sets = build_filter_sets(maf, panel, seed=SEED_BASE)

    # 3. Build window aggregations + position-level table.
    levels = []  # list of (level_id, n_units, units_df_path)
    log.info("Building per-level data (windows + position-level) ...")
    for ws in window_sizes:
        windows = build_window_aggregations(panel, ws)
        win_path = OUT_DIR / f"_fair_units_w{ws}.parquet"
        windows.to_parquet(win_path, index=False)
        levels.append(("window", ws, str(win_path), len(windows)))
        log.info("[w=%d] saved %s with %d rows", ws, win_path, len(windows))
    # Position-level: each panel row IS a unit. Synthesize same-shape table.
    pos_units = panel[["chrom", "pos", "is_cpg", "is_TCW_C", "gc_bin"] + HEADS].copy()
    pos_units["pos"] = pos_units["pos"].astype(int)
    # Synthesize same column names as window-level: head__sum, head__max, etc.
    # At position level, all aggregators collapse to the raw score.
    for h in HEADS:
        for ag in AGGREGATORS:
            pos_units[f"{h}__{ag}"] = pos_units[h]
    pos_units["n_pos_panel"] = 1
    pos_units["cpg_density_panel"] = pos_units["is_cpg"].astype(int)
    pos_units["tcw_density_panel"] = pos_units["is_TCW_C"].astype(int)
    pos_path = OUT_DIR / "_fair_units_position.parquet"
    pos_units.drop(columns=HEADS).to_parquet(pos_path, index=False)
    levels.append(("position", 0, str(pos_path), len(pos_units)))
    log.info("position-level: saved %s with %d rows", pos_path, len(pos_units))

    # 4. For each level: dump baselines (cpg, tcw, npos), per-head-per-agg
    # scores; map mutations under each filter to unit indices.
    log.info("Dumping per-level scores, baselines, mut-arrays ...")
    work_units = []
    null_jobs = []  # (level_id, filter_name, cancer, n_units, k, mut_path, perm_reps, seed)
    null_meta = {}  # (level_id, filter_name) -> {n_units, k, mut_arrays_path}
    for (level_kind, ws_or_zero, units_path, n_units) in levels:
        units = pd.read_parquet(units_path)
        u_chrom = units["chrom"].to_numpy()
        if level_kind == "window":
            u_start = units["win_start"].to_numpy()
            level_id = f"win_{ws_or_zero}"
        else:
            u_pos = units["pos"].astype(int).to_numpy()
            level_id = "position"

        # Dump baselines.
        bl_path = OUT_DIR / f"_fair_baselines_{level_id}.npz"
        np.savez(bl_path,
                 cpg=units["cpg_density_panel"].to_numpy(dtype=np.float64),
                 tcw=units["tcw_density_panel"].to_numpy(dtype=np.float64),
                 npos=units["n_pos_panel"].to_numpy(dtype=np.float64))

        # Dump per-(head, agg) scores.
        score_paths = {}
        for head in HEADS:
            for ag in AGGREGATORS:
                col = f"{head}__{ag}"
                if col not in units.columns:
                    continue
                sp = OUT_DIR / f"_fair_scores_{level_id}_{head}_{ag}.npz"
                np.savez(sp, scores=units[col].to_numpy(dtype=np.float64))
                score_paths[(head, ag)] = str(sp)

        # Map mutations under each filter to unit indices.
        if level_kind == "window":
            ws = ws_or_zero
            unit_lookup = pd.DataFrame({"chrom": u_chrom, "win_start": u_start,
                                        "_uidx": np.arange(n_units)})

        else:
            unit_lookup = pd.DataFrame({"chrom": u_chrom, "pos": u_pos,
                                        "_uidx": np.arange(n_units)})

        k = max(1, int(round(n_units * args.top_pct)))
        log.info("  [%s] n_units=%d  k=%d", level_id, n_units, k)

        for filter_name, mdf in filter_sets.items():
            m = mdf[["chrom", "pos", "cancer"]].copy()
            m["pos"] = m["pos"].astype(int)
            if level_kind == "window":
                m["win_start"] = (m["pos"] // ws_or_zero) * ws_or_zero
                m = m.merge(unit_lookup, on=["chrom", "win_start"], how="inner")
            else:
                m = m.merge(unit_lookup, on=["chrom", "pos"], how="inner")
            mut_arrays = {}
            for cancer in CANCERS:
                sub = m[m["cancer"] == cancer]
                arr = np.zeros(n_units, dtype=np.int32)
                if len(sub) > 0:
                    counts = sub["_uidx"].value_counts()
                    arr[counts.index.astype(int).to_numpy()] = counts.values.astype(np.int32)
                mut_arrays[cancer] = arr
            mut_path = OUT_DIR / f"_fair_mut_{level_id}_{filter_name}.npz"
            np.savez(mut_path, **mut_arrays)
            log.info("    %s: total muts=%d", filter_name,
                     int(sum(arr.sum() for arr in mut_arrays.values())))

            # Queue null-sampling jobs to be parallelized.
            null_meta[(level_id, filter_name)] = {
                "n_units": n_units, "k": k, "mut_path": str(mut_path),
            }
            for cancer in CANCERS:
                mut = mut_arrays[cancer]
                if int(mut.sum()) == 0:
                    continue  # placeholder zeros set later
                seed = SEED_BASE + hash((level_id, filter_name, cancer)) % 100_000
                null_jobs.append((level_id, filter_name, cancer, n_units, k,
                                  str(mut_path), perm_reps, seed))

            # Queue work items: (head, ag, ws, filter_name, ...)
            null_path = OUT_DIR / f"_fair_null_{level_id}_{filter_name}.npz"
            for head in HEADS:
                for ag in AGGREGATORS:
                    sp = score_paths.get((head, ag))
                    if sp is None:
                        continue
                    work_units.append((
                        level_id, head, ag, ws_or_zero, filter_name,
                        sp, str(bl_path), str(mut_path), str(null_path),
                        args.top_pct,
                    ))

    # 4b. Build null distributions in parallel (200 jobs total, dominated by
    # position-level which takes ~40s/30K perms).
    log.info("Sampling null distributions in parallel: %d jobs (perm_reps=%d) ...",
             len(null_jobs), perm_reps)
    t_null_start = time.time()
    nulls_by_key = {}  # (level_id, filter_name, cancer) -> null array
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {}
        for (level_id, filter_name, cancer, n_units, k, mut_path, prs, seed) in null_jobs:
            f = ex.submit(_null_worker_from_path,
                          (level_id, filter_name, cancer, n_units, k, mut_path,
                           prs, seed))
            futures[f] = (level_id, filter_name, cancer)
        n_done = 0
        for f in as_completed(futures):
            lid, fn, can = futures[f]
            try:
                null_arr = f.result()
                nulls_by_key[(lid, fn, can)] = null_arr
                n_done += 1
                if n_done % 50 == 0:
                    log.info("  null %d/%d (%.0fs elapsed)",
                             n_done, len(null_jobs), time.time() - t_null_start)
            except Exception as ex:
                log.error("  null FAIL %s/%s/%s: %s", lid, fn, can, ex)
                raise
    log.info("Null sampling complete in %.1f s.", time.time() - t_null_start)

    # 4c. Save null arrays per (level, filter).
    for (lid, fn), meta in null_meta.items():
        null_arrays = {}
        for cancer in CANCERS:
            arr = nulls_by_key.get((lid, fn, cancer))
            if arr is None:
                arr = np.zeros(perm_reps, dtype=np.int64)
            null_arrays[cancer] = arr
        np.savez(OUT_DIR / f"_fair_null_{lid}_{fn}.npz", **null_arrays)

    log.info("Total cell evaluations queued: %d (= %d levels x %d heads x "
             "%d aggregators x %d filters)",
             len(work_units), len(levels), len(HEADS), len(AGGREGATORS),
             len(FILTERS))
    log.info("Per-level data dump complete in %.1f s. Starting parallel evals ...",
             time.time() - t_start)

    # 5. Parallel evaluation across all cell-filter combos.
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_process_cell, wu): (wu[1], wu[2], wu[3], wu[4])
                   for wu in work_units}
        n_done = 0
        for f in as_completed(futures):
            tag = futures[f]
            try:
                res = f.result()
                results.append(res)
                n_done += 1
                if n_done % 25 == 0:
                    log.info("[%d/%d] last: %s/%s w=%d filt=%s rec=%.4f "
                             "vs_TCW=%.2f vs_NPOS=%.2f bonf=%d/10",
                             n_done, len(futures),
                             res["head"], res["aggregator"],
                             res["window_size_bp"], res["filter"],
                             res["mean_abs_recall"],
                             res["mean_ratio_tcw"],
                             res["mean_ratio_npos"],
                             res["n_bonf_signif"])
            except Exception as ex:
                log.error("FAIL %s: %s", tag, ex)
                raise

    # 6. Build the long-form output table.
    log.info("Building output tables ...")
    rows = []
    per_cancer_rows = []
    for r in results:
        rows.append({
            "level": r["level"],
            "head": r["head"],
            "aggregator": r["aggregator"],
            "window_size_bp": r["window_size_bp"],
            "filter": r["filter"],
            "n_cancers": r["n_cancers"],
            "n_units": r["n_units"],
            "k_top1pct": r["k"],
            "mean_abs_recall": r["mean_abs_recall"],
            "abs_recall_ci_lo": r["abs_recall_ci_lo"],
            "abs_recall_ci_hi": r["abs_recall_ci_hi"],
            "mean_ratio_vs_TCW": r["mean_ratio_tcw"],
            "ratio_tcw_ci_lo": r["ratio_tcw_ci_lo"],
            "ratio_tcw_ci_hi": r["ratio_tcw_ci_hi"],
            "mean_ratio_vs_CpG": r["mean_ratio_cpg"],
            "ratio_cpg_ci_lo": r["ratio_cpg_ci_lo"],
            "ratio_cpg_ci_hi": r["ratio_cpg_ci_hi"],
            "mean_ratio_vs_NPOS": r["mean_ratio_npos"],
            "ratio_npos_ci_lo": r["ratio_npos_ci_lo"],
            "ratio_npos_ci_hi": r["ratio_npos_ci_hi"],
            "n_cancers_above_TCW": r["n_above_tcw"],
            "n_cancers_above_CpG": r["n_above_cpg"],
            "n_cancers_above_NPOS": r["n_above_npos"],
            "n_cancers_bonf_signif": r["n_bonf_signif"],
        })
        for cancer, pc in r["per_cancer"].items():
            per_cancer_rows.append({
                "level": r["level"],
                "head": r["head"],
                "aggregator": r["aggregator"],
                "window_size_bp": r["window_size_bp"],
                "filter": r["filter"],
                "cancer": cancer,
                **pc,
                "bonf_signif": (pc.get("p_perm", 1.0) < BONF_Q),
            })

    df = pd.DataFrame(rows)
    df = df.sort_values(["filter", "window_size_bp", "aggregator", "head"]).reset_index(drop=True)
    out_csv = OUT_DIR / "sweep_v3_fair.csv"
    df.to_csv(out_csv, index=False)
    log.info("Wrote %s (%d rows)", out_csv, len(df))

    pcdf = pd.DataFrame(per_cancer_rows)
    pcdf = pcdf.sort_values(["filter", "window_size_bp", "aggregator",
                             "head", "cancer"]).reset_index(drop=True)
    out_pccsv = OUT_DIR / "sweep_v3_fair_per_cancer.csv"
    pcdf.to_csv(out_pccsv, index=False)
    log.info("Wrote %s (%d rows)", out_pccsv, len(pcdf))

    # 7. Plot: ratio_vs_TCW per head, faceted by filter (1 row x 4 cols).
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        log.info("Generating figure ...")
        fig, axes = plt.subplots(1, 4, figsize=(22, 5), sharey=False)
        for ax, fname in zip(axes, FILTERS):
            sub = df[df["filter"] == fname]
            for head in HEADS:
                ss = sub[(sub["head"] == head) & (sub["level"] == "window")] \
                    .sort_values("window_size_bp")
                if len(ss) == 0:
                    continue
                # Use sum aggregator as the headline line, but plot all 5 if
                # we want; here we plot all 5 thinly + sum boldly.
                # For clarity, just plot sum aggregator.
                ssum = ss[ss["aggregator"] == "sum"]
                ax.plot(ssum["window_size_bp"], ssum["mean_ratio_vs_TCW"],
                        marker="o", label=head.replace("score_", ""))
            ax.axhline(1.0, color="red", linestyle="--", alpha=0.5)
            ax.set_xlabel("window size (bp)")
            ax.set_ylabel("ratio vs TCW (same-bases)")
            ax.set_title(fname)
            ax.legend(fontsize=7, loc="best")
            ax.grid(alpha=0.3)
        plt.tight_layout()
        out_png = OUT_DIR / "sweep_v3_fair.png"
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close()
        log.info("Wrote %s", out_png)
    except Exception as ex:
        log.error("Figure generation failed: %s", ex)

    # 8. Markdown report.
    log.info("Writing SWEEP_V3_FAIR_RESULTS.md ...")
    md = []
    md.append("# Sweep v3 — Fair Re-evaluation of the Panel-Construction Family")
    md.append("")
    md.append("Re-runs the 21 panel constructions x 7 score heads x 4 mutation")
    md.append("filters under three QA-mandated remediations:")
    md.append("")
    md.append(f"1. **Same-bases baselines.** CpG- and TCW-density baselines")
    md.append(f"   counted ONLY over the panel CDS-C positions in each window")
    md.append(f"   (using per-position trinucleotide context recomputed from")
    md.append(f"   hg19), NOT over the full window sequence. Aligns the NN's")
    md.append(f"   sum-over-CDS-C with what motif baselines see.")
    md.append(f"2. **n_pos-only baseline.** Each window also evaluated against")
    md.append(f"   ranking by `n_panel_positions_in_window` — pure gene-body")
    md.append(f"   density, no model, no motif.")
    md.append(f"3. **Real shuffle permutation null.** 10K random k-subset")
    md.append(f"   draws per (window-size, cancer, filter); p_perm = fraction")
    md.append(f"   of perms with mut_in_top >= observed.")
    md.append(f"4. **Bonferroni at alpha=0.05** across {BONFERRONI_N_TESTS}")
    md.append(f"   tests (147 cells x 10 cancers): q < {BONF_Q:.2e}.")
    md.append("")
    md.append(f"**Stratum:** Default = filter_TCW_nonCpG. Also computed for")
    md.append(f"filter_all_TCW, filter_all_CT, filter_random_C.")
    md.append(f"**Cancers (10):** {CANCERS}.")
    md.append(f"**Bootstrap:** N_BOOT={N_BOOT} resamples across 10 cancers.")
    md.append("")

    # Table 1: Top 10 by ratio_vs_TCW
    md.append("## Table 1 — Top 10 cells by ratio_vs_TCW (same-bases)")
    md.append("")
    md.append("Ranked across all 4 filters; the strict TCW-nonCpG stratum is")
    md.append("the headline.")
    md.append("")
    md.append("| rank | head | agg | win | filter | abs_recall (CI) | "
              "ratio_vs_TCW (CI) | ratio_vs_NPOS | bonf/10 |")
    md.append("|------|------|-----|-----|--------|-----------------|"
              "-------------------|---------------|---------|")
    df_top = df.copy()
    df_top = df_top[df_top["mean_ratio_vs_TCW"].notna()].sort_values(
        "mean_ratio_vs_TCW", ascending=False).head(10)
    for i, row in df_top.reset_index(drop=True).iterrows():
        md.append(
            f"| {i+1} | {row['head']} | {row['aggregator']} | "
            f"{int(row['window_size_bp'])} | {row['filter']} | "
            f"{row['mean_abs_recall']*100:.2f}% "
            f"[{row['abs_recall_ci_lo']*100:.2f}, "
            f"{row['abs_recall_ci_hi']*100:.2f}] | "
            f"{row['mean_ratio_vs_TCW']:.3f} "
            f"[{row['ratio_tcw_ci_lo']:.3f}, "
            f"{row['ratio_tcw_ci_hi']:.3f}] | "
            f"{row['mean_ratio_vs_NPOS']:.3f} | "
            f"{int(row['n_cancers_bonf_signif'])}/10 |"
        )
    md.append("")

    # Table 2: n_pos-only baseline ratio_vs_TCW for each (agg, win)
    md.append("## Table 2 — n_pos-only baseline: ratio_vs_TCW per (agg, win)")
    md.append("")
    md.append("Diagnostic: how does ranking by **panel-position count alone**")
    md.append("(no model, no motif) compare to the TCW-density baseline? If")
    md.append("this ratio > 1, then much of the headline 'NN beats TCW' could")
    md.append("be explained by gene-body density alone.")
    md.append("")
    md.append("Per-construction n_pos-only ratio_vs_TCW under filter_TCW_nonCpG:")
    md.append("")
    # Compute n_pos-only ratio: at the unit level, rank by n_pos, get top-k
    # recall, then ratio against tcw-density top-k recall. We have NN/CpG/TCW/
    # NPOS top-k columns recorded in the per-cancer table; the n_pos-only
    # "construction" is captured by abs_recall_npos / abs_recall_tcw at the
    # cancer level. Aggregate to mean across cancers.
    def npos_only_ratio_vs_tcw(level_filter_df: pd.DataFrame) -> tuple[float, float, float]:
        # level_filter_df is per-cancer: includes abs_recall_npos and abs_recall_tcw
        ratios = []
        for _, r in level_filter_df.iterrows():
            an = r.get("abs_recall_npos", float("nan"))
            at = r.get("abs_recall_tcw", float("nan"))
            if pd.notna(an) and pd.notna(at) and at > 0:
                ratios.append(an / at)
        if not ratios:
            return float("nan"), float("nan"), float("nan")
        a = np.asarray(ratios)
        rng = np.random.default_rng(123)
        boot = rng.integers(0, len(a), size=(N_BOOT, len(a)))
        bm = a[boot].mean(axis=1)
        return float(a.mean()), float(np.percentile(bm, 2.5)), \
               float(np.percentile(bm, 97.5))

    md.append("| level | aggregator | window | n_pos_only_recall | "
              "TCW_baseline_recall | ratio_NPOS_vs_TCW | 95% CI |")
    md.append("|-------|------------|--------|-------------------|"
              "---------------------|-------------------|--------|")
    for level_id in sorted(set(df["level"])):
        if level_id == "position":
            ws_show = 0
            agg_show = "(none)"
            sub_pc = pcdf[(pcdf["level"] == level_id) & (pcdf["filter"] == "filter_TCW_nonCpG")
                          & (pcdf["aggregator"] == "sum")  # any agg ok at position level
                          & (pcdf["head"] == "score_binary")]
            mn_npos = sub_pc["abs_recall_npos"].dropna().mean()
            mn_tcw = sub_pc["abs_recall_tcw"].dropna().mean()
            r_mean, r_lo, r_hi = npos_only_ratio_vs_tcw(sub_pc)
            md.append(f"| {level_id} | {agg_show} | {ws_show} | "
                      f"{(mn_npos or 0)*100:.3f}% | "
                      f"{(mn_tcw or 0)*100:.3f}% | "
                      f"{r_mean:.3f} | [{r_lo:.3f}, {r_hi:.3f}] |")
        else:
            ws = int(level_id.split("_")[1])
            for ag in AGGREGATORS:
                sub_pc = pcdf[(pcdf["level"] == level_id)
                              & (pcdf["filter"] == "filter_TCW_nonCpG")
                              & (pcdf["aggregator"] == ag)
                              & (pcdf["head"] == "score_binary")]
                mn_npos = sub_pc["abs_recall_npos"].dropna().mean()
                mn_tcw = sub_pc["abs_recall_tcw"].dropna().mean()
                r_mean, r_lo, r_hi = npos_only_ratio_vs_tcw(sub_pc)
                md.append(f"| window | {ag} | {ws} | "
                          f"{(mn_npos or 0)*100:.3f}% | "
                          f"{(mn_tcw or 0)*100:.3f}% | "
                          f"{r_mean:.3f} | [{r_lo:.3f}, {r_hi:.3f}] |")
    md.append("")
    md.append("(NPOS top-k and TCW top-k baselines do not depend on the score")
    md.append("head or aggregator, so values for `score_binary/sum` are")
    md.append("representative of all 7 heads x 5 aggregators at that ws.)")
    md.append("")

    # Table 3: Bonferroni-surviving cells
    md.append("## Table 3 — Bonferroni-surviving cells")
    md.append("")
    md.append(f"Per-cancer p_perm tested at q < {BONF_Q:.2e}. A cell is")
    md.append("'surviving' if at least one cancer's p_perm clears Bonferroni;")
    md.append("'majority surviving' if >=6 of 10 cancers do.")
    md.append("")
    survivors = df[df["n_cancers_bonf_signif"] >= 1].copy()
    majority = df[df["n_cancers_bonf_signif"] >= 6].copy()
    md.append(f"- Cells with >=1 cancer surviving Bonferroni: **{len(survivors)} / {len(df)}**")
    md.append(f"- Cells with majority (>=6/10) surviving: **{len(majority)} / {len(df)}**")
    md.append("")
    md.append("### Top 15 cells by `n_cancers_bonf_signif`, tied by ratio_vs_TCW")
    md.append("")
    md.append("| head | agg | win | filter | abs_recall | ratio_vs_TCW | "
              "ratio_vs_NPOS | bonf/10 |")
    md.append("|------|-----|-----|--------|------------|--------------|"
              "---------------|---------|")
    df_bonf = df.sort_values(["n_cancers_bonf_signif", "mean_ratio_vs_TCW"],
                             ascending=[False, False]).head(15)
    for _, row in df_bonf.iterrows():
        md.append(
            f"| {row['head']} | {row['aggregator']} | "
            f"{int(row['window_size_bp'])} | {row['filter']} | "
            f"{row['mean_abs_recall']*100:.2f}% | "
            f"{row['mean_ratio_vs_TCW']:.3f} | "
            f"{row['mean_ratio_vs_NPOS']:.3f} | "
            f"{int(row['n_cancers_bonf_signif'])}/10 |"
        )
    md.append("")

    # Headline rewrite
    md.append("## Headline rewrite")
    md.append("")
    # Find the previous "winner": score_apobec1, sum, 1000, filter_TCW_nonCpG
    # And the score_binary row at same construction.
    headline_filter = "filter_TCW_nonCpG"
    win_row = df[(df["filter"] == headline_filter)
                 & (df["aggregator"] == "sum")
                 & (df["window_size_bp"] == 1000)
                 & (df["head"] == "score_binary")]
    apo_row = df[(df["filter"] == headline_filter)
                 & (df["aggregator"] == "sum")
                 & (df["window_size_bp"] == 1000)
                 & (df["head"] == "score_apobec1")]
    if len(win_row) and len(apo_row):
        wr = win_row.iloc[0]
        ar = apo_row.iloc[0]
        md.append(f"**Original headline:** score_binary, sum, 1000 bp, ")
        md.append(f"filter_TCW_nonCpG = 1.31x ratio_vs_TCW (window-seq baseline).")
        md.append("")
        md.append(f"**Fair re-eval, same construction:**")
        md.append(f"- ratio_vs_TCW (same-bases): **{wr['mean_ratio_vs_TCW']:.3f}** "
                  f"[{wr['ratio_tcw_ci_lo']:.3f}, {wr['ratio_tcw_ci_hi']:.3f}]")
        md.append(f"- ratio_vs_NPOS (n_pos-only): **{wr['mean_ratio_vs_NPOS']:.3f}** "
                  f"[{wr['ratio_npos_ci_lo']:.3f}, {wr['ratio_npos_ci_hi']:.3f}]")
        md.append(f"- abs_recall: {wr['mean_abs_recall']*100:.2f}% "
                  f"[{wr['abs_recall_ci_lo']*100:.2f}, "
                  f"{wr['abs_recall_ci_hi']*100:.2f}]")
        md.append(f"- Bonferroni-surviving cancers: {int(wr['n_cancers_bonf_signif'])}/10")
        md.append("")
        md.append(f"**Best per-head at same construction:** score_apobec1 = "
                  f"ratio_vs_TCW {ar['mean_ratio_vs_TCW']:.3f} "
                  f"[{ar['ratio_tcw_ci_lo']:.3f}, {ar['ratio_tcw_ci_hi']:.3f}]")
        md.append(f"   ratio_vs_NPOS = {ar['mean_ratio_vs_NPOS']:.3f}, "
                  f"bonf {int(ar['n_cancers_bonf_signif'])}/10")
        md.append("")

    # Find the strongest defensible cell: max ratio_vs_NPOS with bonf >=6/10
    # and abs_recall_ci_lo > 0.
    candidates = df[(df["filter"] == "filter_TCW_nonCpG")
                    & (df["mean_ratio_vs_NPOS"] > 1.0)
                    & (df["n_cancers_bonf_signif"] >= 6)
                    & (df["mean_ratio_vs_TCW"] > 1.0)].sort_values(
        "mean_ratio_vs_NPOS", ascending=False)
    if len(candidates):
        c = candidates.iloc[0]
        md.append(f"**Strongest defensible claim** (filter_TCW_nonCpG, "
                  f">=6/10 bonf-signif, beats both same-bases motif AND "
                  f"n_pos-only):")
        md.append(f"- Construction: **{c['head']}, {c['aggregator']}, "
                  f"{int(c['window_size_bp'])} bp**")
        md.append(f"- abs_recall = {c['mean_abs_recall']*100:.2f}% "
                  f"[{c['abs_recall_ci_lo']*100:.2f}, "
                  f"{c['abs_recall_ci_hi']*100:.2f}]")
        md.append(f"- ratio_vs_TCW = {c['mean_ratio_vs_TCW']:.3f} "
                  f"[{c['ratio_tcw_ci_lo']:.3f}, "
                  f"{c['ratio_tcw_ci_hi']:.3f}]")
        md.append(f"- ratio_vs_NPOS = {c['mean_ratio_vs_NPOS']:.3f} "
                  f"[{c['ratio_npos_ci_lo']:.3f}, "
                  f"{c['ratio_npos_ci_hi']:.3f}]")
        md.append(f"- Bonf-signif cancers: {int(c['n_cancers_bonf_signif'])}/10")
    else:
        md.append("**Strongest defensible claim:** NO cell at filter_TCW_nonCpG")
        md.append("simultaneously (a) majority Bonferroni-surviving, ")
        md.append("(b) ratio_vs_TCW > 1, AND (c) ratio_vs_NPOS > 1. The 1.31x")
        md.append("headline does not survive the fair comparison.")
    md.append("")

    # Verdict
    md.append("## Verdict")
    md.append("")
    if len(win_row):
        wr = win_row.iloc[0]
        survives_tcw = (wr["mean_ratio_vs_TCW"] > 1.0
                        and wr["ratio_tcw_ci_lo"] > 1.0)
        survives_npos = (wr["mean_ratio_vs_NPOS"] > 1.0
                         and wr["ratio_npos_ci_lo"] > 1.0)
        verdict_tcw = "YES" if survives_tcw else "NO"
        verdict_npos = "YES" if survives_npos else "NO"
        md.append(f"**Q1: Does the 1.31x headline survive same-bases TCW?**  "
                  f"**{verdict_tcw}**  "
                  f"(ratio_vs_TCW = {wr['mean_ratio_vs_TCW']:.3f} "
                  f"[{wr['ratio_tcw_ci_lo']:.3f}, "
                  f"{wr['ratio_tcw_ci_hi']:.3f}])")
        md.append(f"**Q2: Does ANY construction beat n_pos alone (CI lo > 1)?**  ")
        any_beats_npos = df[(df["filter"] == "filter_TCW_nonCpG")
                            & (df["ratio_npos_ci_lo"] > 1.0)]
        if len(any_beats_npos):
            best = any_beats_npos.sort_values("mean_ratio_vs_NPOS",
                                              ascending=False).iloc[0]
            md.append(f"**YES** — best: {best['head']} {best['aggregator']} "
                      f"w={int(best['window_size_bp'])} ratio_vs_NPOS="
                      f"{best['mean_ratio_vs_NPOS']:.3f} "
                      f"[{best['ratio_npos_ci_lo']:.3f}, "
                      f"{best['ratio_npos_ci_hi']:.3f}]")
        else:
            md.append(f"**NO** — no cell at filter_TCW_nonCpG has 95% CI ")
            md.append(f"lower bound for ratio_vs_NPOS > 1.0. Gene-body density")
            md.append(f"alone explains the headline.")
    md.append("")
    md.append("## Files")
    md.append("")
    md.append("- `sweep_v3_fair.csv` — flat table, 1 row per (head, agg, ws, filter)")
    md.append("- `sweep_v3_fair_per_cancer.csv` — per-cancer drill-down")
    md.append("- `sweep_v3_fair.png` — ratio_vs_TCW per head, faceted by filter")
    md.append("- `SWEEP_V3_FAIR_RESULTS.md` — this report")

    out_md = OUT_DIR / "SWEEP_V3_FAIR_RESULTS.md"
    out_md.write_text("\n".join(md))
    log.info("Wrote %s", out_md)

    # 9. Cleanup tmp files unless --quick.
    if not args.quick:
        log.info("Cleaning up temp files ...")
        for p in OUT_DIR.glob("_fair_*"):
            try:
                p.unlink()
            except Exception:
                pass

    log.info("DONE in %.1f s.", time.time() - t_start)
    return 0


if __name__ == "__main__":
    sys.exit(main())
