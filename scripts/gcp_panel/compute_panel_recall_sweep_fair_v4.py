#!/usr/bin/env python3
"""FAIR remediation of the 21-construction panel sweep — V4 PARAMETERIZED.

Identical logic to ``compute_panel_recall_sweep_fair.py`` but accepts a
configurable panel parquet, output prefix, and head list. Default heads list
matches the v4 panel parquet column set (6 heads, no Neither, apobec1 renamed
to apobec1_v3).

Coordinate convention: panel parquet position is hg19 (same as v3).

Bonferroni count is computed dynamically: 21 constructions x N_heads x 10
cancers (= 1260 for 6 heads).

Outputs (with --out-prefix <prefix>):
  <prefix>.csv
  <prefix>_per_cancer.csv
  <prefix>.png
  <prefix>_RESULTS.md

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
TCGA_DIR = ROOT / "data/raw/tcga"
PCAWG_DIR = ROOT / "data/raw/pcawg/by_cancer"
HG19 = ROOT / "data/raw/genomes/hg19.fa"

CANCERS = ["blca", "brca", "cesc", "coadread", "esca", "hnsc", "lihc",
           "lusc", "skcm", "stad"]
WINDOW_SIZES = [100, 250, 500, 1000]
AGGREGATORS = ["max", "mean", "sum", "top3_mean", "p95"]
DEFAULT_HEADS_V4 = [
    "score_binary",
    "score_A3A",
    "score_A3B",
    "score_A3G",
    "score_A3A_A3G",
    "score_apobec1_v3",
]
FILTERS = ["filter_TCW_nonCpG", "filter_all_TCW", "filter_all_CT",
           "filter_random_C"]
TOP_PCT = 0.01
PERM_REPS = 30000
N_BOOT = 10000
SEED_BASE = 20260427
ALPHA = 0.05

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    stream=sys.stdout)
log = logging.getLogger(__name__)


# =========================================================================== #
# Mutation loading (TCGA + PCAWG-coding combined).
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
        is_cpg[valid_idx[is_plus]] = (right[is_plus] == ord("G"))
        is_cpg[valid_idx[is_minus]] = (left[is_minus] == ord("C"))
        right_AT = (right == ord("A")) | (right == ord("T"))
        left_AT = (left == ord("A")) | (left == ord("T"))
        is_tcw_c[valid_idx[is_plus]] = (left[is_plus] == ord("T")) & right_AT[is_plus]
        is_tcw_c[valid_idx[is_minus]] = (right[is_minus] == ord("A")) & left_AT[is_minus]

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
# Build window aggregations
# =========================================================================== #

def _top3_mean_groupby(p_sorted: pd.DataFrame,
                       group_keys: list[str], col: str) -> pd.Series:
    grp = p_sorted.groupby(group_keys, sort=False)[col]
    return grp.apply(lambda v: float(np.mean(np.partition(v.values, -3)[-3:]))
                     if len(v) > 3 else float(v.mean()) if len(v) > 0 else 0.0)


def build_window_aggregations(panel: pd.DataFrame, window_size: int,
                              heads: list[str]) -> pd.DataFrame:
    log.info("[w=%d] Building windows (%d heads x 5 aggs + same-bases baselines) ...",
             window_size, len(heads))
    p = panel.copy()
    p["pos"] = p["pos"].astype(int)
    p["win_start"] = (p["pos"] // window_size) * window_size

    agg_dict = {
        "n_pos_panel":   ("pos", "size"),
        "cpg_density_panel": ("is_cpg", "sum"),
        "tcw_density_panel": ("is_TCW_C", "sum"),
        "gc_bin": ("gc_bin", "median"),
    }
    for h in heads:
        agg_dict[f"{h}__max"] = (h, "max")
        agg_dict[f"{h}__mean"] = (h, "mean")
        agg_dict[f"{h}__sum"] = (h, "sum")
    grp = p.groupby(["chrom", "win_start"])
    out = grp.agg(**agg_dict).reset_index()
    out["gc_bin"] = out["gc_bin"].astype(int)

    log.info("[w=%d] Computing p95 per head ...", window_size)
    for h in heads:
        q95 = grp[h].quantile(0.95).rename(f"{h}__p95").reset_index()
        out = out.merge(q95, on=["chrom", "win_start"])

    log.info("[w=%d] Computing top3_mean per head ...", window_size)
    p_sorted = p.sort_values(["chrom", "win_start"], kind="stable")
    for h in heads:
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
# Permutation null distributions
# =========================================================================== #

def _build_null_dist(n_units: int, k: int, mut: np.ndarray,
                     n_reps: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if int(mut.sum()) == 0:
        return np.zeros(n_reps, dtype=np.int64)
    out = np.empty(n_reps, dtype=np.int64)
    for i in range(n_reps):
        idx = rng.choice(n_units, size=k, replace=False)
        out[i] = int(mut[idx].sum())
    return out


def _null_worker_from_path(args):
    (level_id, filter_name, cancer, n_units, k, mut_path, n_reps, seed) = args
    mc = np.load(mut_path)
    if cancer not in mc.files:
        return np.zeros(n_reps, dtype=np.int64)
    mut = mc[cancer]
    if int(mut.sum()) == 0:
        return np.zeros(n_reps, dtype=np.int64)
    return _build_null_dist(n_units, k, mut, n_reps, seed)


# =========================================================================== #
# Per-cell evaluator
# =========================================================================== #

def evaluate_cell(scores: np.ndarray, base_cpg: np.ndarray, base_tcw: np.ndarray,
                  base_npos: np.ndarray, mut_per_cancer: dict[str, np.ndarray],
                  null_per_cancer: dict[str, np.ndarray], k: int,
                  bonf_q: float) -> dict:
    n = len(scores)

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

    n_bonf = int(sum(p < bonf_q for p in p_perms))

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


def _process_cell(args):
    (level, head, aggregator, window_size, filter_name,
     scores_path, baselines_path, mut_path, null_path, top_pct,
     bonf_q) = args
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
                        mut_per_cancer, null_per_cancer, k, bonf_q)
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
    ap.add_argument("--panel", type=str, required=True,
                    help="Path to panel parquet with score_* columns")
    ap.add_argument("--out-prefix", type=str, required=True,
                    help="Output prefix path (no extension)")
    ap.add_argument("--heads", type=str,
                    default=",".join(DEFAULT_HEADS_V4),
                    help="Comma-separated list of score head columns to evaluate")
    ap.add_argument("--n-workers", type=int, default=8)
    ap.add_argument("--perm-reps", type=int, default=PERM_REPS)
    ap.add_argument("--top-pct", type=float, default=TOP_PCT)
    ap.add_argument("--quick", action="store_true",
                    help="Quick run: 1K perms, only ws=1000.")
    args = ap.parse_args()
    n_workers = min(8, args.n_workers)

    panel_path = Path(args.panel)
    out_prefix = Path(args.out_prefix)
    out_dir = out_prefix.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_stem = out_prefix.name
    heads = [h.strip() for h in args.heads.split(",") if h.strip()]
    n_heads = len(heads)

    n_constructions = 21  # 1 position + 4 ws x 5 aggs
    bonferroni_n_tests = n_constructions * n_heads * 10
    bonf_q = ALPHA / bonferroni_n_tests

    perm_reps = 1000 if args.quick else args.perm_reps
    window_sizes = [1000] if args.quick else WINDOW_SIZES
    log.info("=== Fair sweep V4 ===")
    log.info("Panel: %s", panel_path)
    log.info("Out prefix: %s", out_prefix)
    log.info("Heads: %s", heads)
    log.info("Window sizes: %s; aggregators: %s; filters: %s",
             window_sizes, AGGREGATORS, FILTERS)
    log.info("Top pct: %.3f; perm_reps: %d; Bonferroni q=%.2e (n_tests=%d)",
             args.top_pct, perm_reps, bonf_q, bonferroni_n_tests)

    t_start = time.time()

    # 1. Load + annotate panel.
    log.info("Loading panel ...")
    panel = pd.read_parquet(panel_path)
    log.info("  panel rows: %d", len(panel))
    # Verify all heads exist
    missing = [h for h in heads if h not in panel.columns]
    if missing:
        raise ValueError(f"Missing head columns in panel: {missing}")
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
    levels = []
    log.info("Building per-level data (windows + position-level) ...")
    tmp_prefix = out_dir / f"_{out_stem}_tmp"
    for ws in window_sizes:
        windows = build_window_aggregations(panel, ws, heads)
        win_path = out_dir / f"_{out_stem}_units_w{ws}.parquet"
        windows.to_parquet(win_path, index=False)
        levels.append(("window", ws, str(win_path), len(windows)))
        log.info("[w=%d] saved %s with %d rows", ws, win_path, len(windows))
    pos_units = panel[["chrom", "pos", "is_cpg", "is_TCW_C", "gc_bin"] + heads].copy()
    pos_units["pos"] = pos_units["pos"].astype(int)
    for h in heads:
        for ag in AGGREGATORS:
            pos_units[f"{h}__{ag}"] = pos_units[h]
    pos_units["n_pos_panel"] = 1
    pos_units["cpg_density_panel"] = pos_units["is_cpg"].astype(int)
    pos_units["tcw_density_panel"] = pos_units["is_TCW_C"].astype(int)
    pos_path = out_dir / f"_{out_stem}_units_position.parquet"
    pos_units.drop(columns=heads).to_parquet(pos_path, index=False)
    levels.append(("position", 0, str(pos_path), len(pos_units)))
    log.info("position-level: saved %s with %d rows", pos_path, len(pos_units))

    # 4. For each level: dump baselines, scores, mut arrays.
    log.info("Dumping per-level scores, baselines, mut-arrays ...")
    work_units = []
    null_jobs = []
    null_meta = {}
    for (level_kind, ws_or_zero, units_path, n_units) in levels:
        units = pd.read_parquet(units_path)
        u_chrom = units["chrom"].to_numpy()
        if level_kind == "window":
            u_start = units["win_start"].to_numpy()
            level_id = f"win_{ws_or_zero}"
        else:
            u_pos = units["pos"].astype(int).to_numpy()
            level_id = "position"

        bl_path = out_dir / f"_{out_stem}_baselines_{level_id}.npz"
        np.savez(bl_path,
                 cpg=units["cpg_density_panel"].to_numpy(dtype=np.float64),
                 tcw=units["tcw_density_panel"].to_numpy(dtype=np.float64),
                 npos=units["n_pos_panel"].to_numpy(dtype=np.float64))

        score_paths = {}
        for head in heads:
            for ag in AGGREGATORS:
                col = f"{head}__{ag}"
                if col not in units.columns:
                    continue
                sp = out_dir / f"_{out_stem}_scores_{level_id}_{head}_{ag}.npz"
                np.savez(sp, scores=units[col].to_numpy(dtype=np.float64))
                score_paths[(head, ag)] = str(sp)

        if level_kind == "window":
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
            mut_path = out_dir / f"_{out_stem}_mut_{level_id}_{filter_name}.npz"
            np.savez(mut_path, **mut_arrays)
            log.info("    %s: total muts=%d", filter_name,
                     int(sum(arr.sum() for arr in mut_arrays.values())))

            null_meta[(level_id, filter_name)] = {
                "n_units": n_units, "k": k, "mut_path": str(mut_path),
            }
            for cancer in CANCERS:
                mut = mut_arrays[cancer]
                if int(mut.sum()) == 0:
                    continue
                seed = SEED_BASE + hash((level_id, filter_name, cancer)) % 100_000
                null_jobs.append((level_id, filter_name, cancer, n_units, k,
                                  str(mut_path), perm_reps, seed))

            null_path = out_dir / f"_{out_stem}_null_{level_id}_{filter_name}.npz"
            for head in heads:
                for ag in AGGREGATORS:
                    sp = score_paths.get((head, ag))
                    if sp is None:
                        continue
                    work_units.append((
                        level_id, head, ag, ws_or_zero, filter_name,
                        sp, str(bl_path), str(mut_path), str(null_path),
                        args.top_pct, bonf_q,
                    ))

    # 4b. Build null distributions in parallel.
    log.info("Sampling null distributions in parallel: %d jobs (perm_reps=%d) ...",
             len(null_jobs), perm_reps)
    t_null_start = time.time()
    nulls_by_key = {}
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
        np.savez(out_dir / f"_{out_stem}_null_{lid}_{fn}.npz", **null_arrays)

    log.info("Total cell evaluations queued: %d (= %d levels x %d heads x "
             "%d aggregators x %d filters)",
             len(work_units), len(levels), n_heads, len(AGGREGATORS),
             len(FILTERS))
    log.info("Per-level data dump complete in %.1f s. Starting parallel evals ...",
             time.time() - t_start)

    # 5. Parallel evaluation.
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
                "bonf_signif": (pc.get("p_perm", 1.0) < bonf_q),
            })

    df = pd.DataFrame(rows)
    df = df.sort_values(["filter", "window_size_bp", "aggregator", "head"]).reset_index(drop=True)
    out_csv = out_dir / f"{out_stem}.csv"
    df.to_csv(out_csv, index=False)
    log.info("Wrote %s (%d rows)", out_csv, len(df))

    pcdf = pd.DataFrame(per_cancer_rows)
    pcdf = pcdf.sort_values(["filter", "window_size_bp", "aggregator",
                             "head", "cancer"]).reset_index(drop=True)
    out_pccsv = out_dir / f"{out_stem}_per_cancer.csv"
    pcdf.to_csv(out_pccsv, index=False)
    log.info("Wrote %s (%d rows)", out_pccsv, len(pcdf))

    # 7. Plot.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        log.info("Generating figure ...")
        fig, axes = plt.subplots(1, 4, figsize=(22, 5), sharey=False)
        for ax, fname in zip(axes, FILTERS):
            sub = df[df["filter"] == fname]
            for head in heads:
                ss = sub[(sub["head"] == head) & (sub["level"] == "window")] \
                    .sort_values("window_size_bp")
                if len(ss) == 0:
                    continue
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
        out_png = out_dir / f"{out_stem}.png"
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close()
        log.info("Wrote %s", out_png)
    except Exception as ex:
        log.error("Figure generation failed: %s", ex)

    # 8. Markdown summary report.
    log.info("Writing %s_RESULTS.md ...", out_stem)
    md = []
    md.append(f"# Sweep {out_stem} - Fair Re-evaluation")
    md.append("")
    md.append(f"Panel: `{panel_path}`")
    md.append(f"Heads ({n_heads}): {heads}")
    md.append(f"Window sizes: {window_sizes}; aggregators: {AGGREGATORS}; "
              f"filters: {FILTERS}")
    md.append(f"Top pct: {args.top_pct}; perm_reps: {perm_reps}; "
              f"Bonferroni n_tests = {bonferroni_n_tests}; q < {bonf_q:.2e}")
    md.append(f"Bootstrap N_BOOT = {N_BOOT} resamples")
    md.append("")

    # Top 10 by ratio_vs_TCW
    md.append("## Top 10 cells by ratio_vs_TCW (filter_TCW_nonCpG)")
    md.append("")
    md.append("| rank | head | agg | win | filter | abs_recall (CI) | "
              "ratio_vs_TCW (CI) | ratio_vs_NPOS (CI) | bonf/10 |")
    md.append("|------|------|-----|-----|--------|-----------------|"
              "-------------------|--------------------|---------|")
    df_top = df.copy()
    df_top = df_top[(df_top["mean_ratio_vs_TCW"].notna())
                    & (df_top["filter"] == "filter_TCW_nonCpG")] \
        .sort_values("mean_ratio_vs_TCW", ascending=False).head(10)
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
            f"{row['mean_ratio_vs_NPOS']:.3f} "
            f"[{row['ratio_npos_ci_lo']:.3f}, "
            f"{row['ratio_npos_ci_hi']:.3f}] | "
            f"{int(row['n_cancers_bonf_signif'])}/10 |"
        )
    md.append("")

    # Bonferroni summary
    survivors = df[df["n_cancers_bonf_signif"] >= 1].copy()
    majority = df[df["n_cancers_bonf_signif"] >= 6].copy()
    md.append(f"## Bonferroni (q < {bonf_q:.2e})")
    md.append("")
    md.append(f"- Cells with >=1 cancer surviving: **{len(survivors)} / {len(df)}**")
    md.append(f"- Cells with majority (>=6/10) surviving: **{len(majority)} / {len(df)}**")
    md.append("")

    # Headline
    md.append("## Headline (binary, sum, win=1000, filter_TCW_nonCpG)")
    md.append("")
    win_row = df[(df["filter"] == "filter_TCW_nonCpG")
                 & (df["aggregator"] == "sum")
                 & (df["window_size_bp"] == 1000)
                 & (df["head"] == "score_binary")]
    if len(win_row):
        wr = win_row.iloc[0]
        md.append(f"- abs_recall: {wr['mean_abs_recall']*100:.3f}% "
                  f"[{wr['abs_recall_ci_lo']*100:.3f}, "
                  f"{wr['abs_recall_ci_hi']*100:.3f}]")
        md.append(f"- ratio_vs_TCW: {wr['mean_ratio_vs_TCW']:.3f} "
                  f"[{wr['ratio_tcw_ci_lo']:.3f}, "
                  f"{wr['ratio_tcw_ci_hi']:.3f}]")
        md.append(f"- ratio_vs_NPOS: {wr['mean_ratio_vs_NPOS']:.3f} "
                  f"[{wr['ratio_npos_ci_lo']:.3f}, "
                  f"{wr['ratio_npos_ci_hi']:.3f}]")
        md.append(f"- bonf signif cancers: {int(wr['n_cancers_bonf_signif'])}/10")
    md.append("")

    md.append("## Files")
    md.append(f"- `{out_stem}.csv`")
    md.append(f"- `{out_stem}_per_cancer.csv`")
    md.append(f"- `{out_stem}.png`")

    out_md = out_dir / f"{out_stem}_RESULTS.md"
    out_md.write_text("\n".join(md))
    log.info("Wrote %s", out_md)

    # 9. Cleanup.
    if not args.quick:
        log.info("Cleaning up temp files ...")
        for p in out_dir.glob(f"_{out_stem}_*"):
            try:
                p.unlink()
            except Exception:
                pass

    log.info("DONE in %.1f s.", time.time() - t_start)
    return 0


if __name__ == "__main__":
    sys.exit(main())
