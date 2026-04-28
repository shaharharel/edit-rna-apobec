#!/usr/bin/env python3
"""Panel-size and absolute-threshold sweep on v4 panel scores.

Sweeps:
- top_pct in [0.01, 0.05, 0.10]
- pscore in [P75, P90, P95, P99] of each head's score distribution
- levels: position, window_max_w1000
- heads: score_binary, score_apobec1_v3 (focus heads)
- filters: filter_TCW_nonCpG, filter_all_CT

For each cell: panel_units, panel_coverage_Mb, abs_recall (mean+CI),
ratio_vs_TCW (same-bases) + CI, ratio_vs_NPOS + CI, n_cancers_bonf_signif.

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
DEFAULT_HEADS = ["score_binary", "score_apobec1_v3"]
HEADS = list(DEFAULT_HEADS)  # mutated by --heads at startup
FILTERS = ["filter_TCW_nonCpG", "filter_all_CT"]
WINDOW_SIZE = 1000  # window_max_w1000
TOP_PCTS = [0.01, 0.05, 0.10]
PSCORE_QS = [0.75, 0.90, 0.95, 0.99]
PERM_REPS = 10000
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
# Mutation loading (TCGA + PCAWG-coding combined). Same as fair_v4.
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
    log.info("  combined C>T/G>A: %d, %d cancers",
             len(combined), combined["cancer"].nunique())
    return combined


def annotate_panel_positions(panel: pd.DataFrame) -> pd.DataFrame:
    from pyfaidx import Fasta
    log.info("Annotating panel positions with is_cpg + is_TCW_C ...")
    genome = Fasta(str(HG19), as_raw=False, sequence_always_upper=True)

    n = len(panel)
    is_cpg = np.zeros(n, dtype=bool)
    is_tcw_c = np.zeros(n, dtype=bool)

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

    panel = panel.copy()
    panel["is_cpg"] = is_cpg
    panel["is_TCW_C"] = is_tcw_c
    log.info("  panel n=%d, is_TCW_C=%d (%.2f%%)", n, is_tcw_c.sum(),
             100 * is_tcw_c.mean())
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
    log.info("  total=%d  TCW_nonCpG=%d (%.1f%%)", n,
             out["is_TCW_nonCpG"].sum(), 100 * out["is_TCW_nonCpG"].mean())
    return out


# =========================================================================== #
# Window-level aggregation (max only, win=1000)
# =========================================================================== #

def build_window_max_w1000(panel: pd.DataFrame, heads: list[str],
                           window_size: int = 1000) -> pd.DataFrame:
    log.info("Building windows w=%d (max aggregator only)...", window_size)
    p = panel.copy()
    p["pos"] = p["pos"].astype(int)
    p["win_start"] = (p["pos"] // window_size) * window_size

    agg_dict = {
        "n_pos_panel":   ("pos", "size"),
        "tcw_density_panel": ("is_TCW_C", "sum"),
    }
    for h in heads:
        agg_dict[h] = (h, "max")
    grp = p.groupby(["chrom", "win_start"])
    out = grp.agg(**agg_dict).reset_index()
    log.info("  %d windows; mean tcw_density_panel=%.2f, mean n_pos=%.2f",
             len(out), out["tcw_density_panel"].mean(),
             out["n_pos_panel"].mean())
    return out


# =========================================================================== #
# Filter sets (TCW_nonCpG + all_CT only — no random_C, no all_TCW)
# =========================================================================== #

def build_filter_sets(maf: pd.DataFrame) -> dict[str, pd.DataFrame]:
    log.info("Building filter sets ...")
    out = {}
    out["filter_TCW_nonCpG"] = maf[maf["is_TCW_nonCpG"]].copy()
    out["filter_all_CT"] = maf.copy()
    for fn, fdf in out.items():
        log.info("  %s: %d mutations", fn, len(fdf))
    return out


# =========================================================================== #
# Permutation null
# =========================================================================== #

def _build_null_dist(n_units: int, k: int, mut: np.ndarray,
                     n_reps: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if int(mut.sum()) == 0 or k <= 0:
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
# Per-cell evaluator (selects top-k by score, baselines by tcw / npos)
# =========================================================================== #

def evaluate_cell_topk(scores: np.ndarray, base_tcw: np.ndarray,
                       base_npos: np.ndarray,
                       mut_per_cancer: dict[str, np.ndarray],
                       null_per_cancer: dict[str, np.ndarray],
                       k: int, bonf_q: float) -> dict:
    n = len(scores)
    if k <= 0 or k >= n:
        # degenerate: skip
        return None

    nn_top = np.argpartition(-scores, k - 1)[:k]
    tcw_top = np.argpartition(-base_tcw, k - 1)[:k]
    npos_top = np.argpartition(-base_npos, k - 1)[:k]

    per_cancer = {}
    abs_recalls = []
    ratios_tcw, ratios_npos = [], []
    n_above_tcw = n_above_npos = 0
    p_perms = []

    for cancer, mut in mut_per_cancer.items():
        total = int(mut.sum())
        if total == 0:
            per_cancer[cancer] = {"total_mut": 0, "abs_recall": float("nan"),
                                  "abs_recall_tcw": float("nan"),
                                  "abs_recall_npos": float("nan"),
                                  "ratio_tcw": float("nan"),
                                  "ratio_npos": float("nan"),
                                  "p_perm": 1.0,
                                  "k": int(k), "n_units": int(n)}
            continue
        nn_recall = float(mut[nn_top].sum()) / total
        tcw_recall = float(mut[tcw_top].sum()) / total
        npos_recall = float(mut[npos_top].sum()) / total
        ratio_tcw = nn_recall / tcw_recall if tcw_recall > 0 else float("nan")
        ratio_npos = nn_recall / npos_recall if npos_recall > 0 else float("nan")

        null = null_per_cancer.get(cancer, np.zeros(1, dtype=np.int64))
        mut_in_top_obs = int(mut[nn_top].sum())
        p_perm = float((null >= mut_in_top_obs).sum() + 1) / (len(null) + 1)

        per_cancer[cancer] = {
            "total_mut": total,
            "abs_recall": nn_recall,
            "abs_recall_tcw": tcw_recall,
            "abs_recall_npos": npos_recall,
            "ratio_tcw": ratio_tcw,
            "ratio_npos": ratio_npos,
            "mut_in_top_obs": mut_in_top_obs,
            "p_perm": p_perm,
            "k": int(k),
            "n_units": int(n),
        }
        abs_recalls.append(nn_recall)
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
    m_tcw, lo_tcw, hi_tcw = boot_ci(ratios_tcw, 3)
    m_npos, lo_npos, hi_npos = boot_ci(ratios_npos, 4)

    n_bonf = int(sum(p < bonf_q for p in p_perms))

    return {
        "per_cancer": per_cancer,
        "n_cancers": len(per_cancer),
        "mean_abs_recall": m_abs, "abs_recall_ci_lo": lo_abs,
        "abs_recall_ci_hi": hi_abs,
        "mean_ratio_tcw": m_tcw, "ratio_tcw_ci_lo": lo_tcw,
        "ratio_tcw_ci_hi": hi_tcw,
        "mean_ratio_npos": m_npos, "ratio_npos_ci_lo": lo_npos,
        "ratio_npos_ci_hi": hi_npos,
        "n_above_tcw": n_above_tcw,
        "n_above_npos": n_above_npos,
        "n_bonf_signif": n_bonf,
        "k": int(k), "n_units": int(n),
    }


def _process_cell(args):
    (level, head, cut_type, cut_value, filter_name,
     scores_path, baselines_path, mut_path, null_path, k_target,
     bonf_q, panel_units, panel_coverage_Mb) = args
    scores = np.load(scores_path)["scores"]
    bl = np.load(baselines_path)
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
    k = max(1, int(k_target))
    if k >= n:
        k = n - 1

    res = evaluate_cell_topk(scores, base_tcw, base_npos,
                             mut_per_cancer, null_per_cancer, k, bonf_q)
    if res is None:
        return None
    res["level"] = level
    res["head"] = head
    res["cut_type"] = cut_type
    res["cut_value"] = cut_value
    res["filter"] = filter_name
    res["panel_units"] = panel_units
    res["panel_coverage_Mb"] = panel_coverage_Mb
    return res


# =========================================================================== #
# Helpers
# =========================================================================== #

def panel_coverage_mb(level: str, k: int, window_size: int = 1000) -> float:
    """Genomic coverage in Mb for a panel of k units."""
    if level == "position":
        return k * 1e-6
    elif level == "window_max_w1000":
        return k * window_size * 1e-6
    else:
        return float("nan")


def k_from_threshold(scores: np.ndarray, threshold: float) -> int:
    """Number of units with score >= threshold."""
    return int((scores >= threshold).sum())


# =========================================================================== #
# Main
# =========================================================================== #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", type=str, required=True,
                    help="Path to panel parquet")
    ap.add_argument("--out-prefix", type=str, required=True,
                    help="Output prefix (no extension)")
    ap.add_argument("--n-workers", type=int, default=8)
    ap.add_argument("--perm-reps", type=int, default=PERM_REPS)
    ap.add_argument("--heads", type=str, default=",".join(DEFAULT_HEADS),
                    help="Comma-separated list of head columns to evaluate.")
    args = ap.parse_args()
    n_workers = min(8, args.n_workers)

    # Override globals based on --heads. Module globals are referenced from
    # multiple helper functions and ProcessPoolExecutor workers (which inherit
    # the parent's module state on macOS via 'fork'-then-spawn semantics).
    global HEADS
    HEADS = [h.strip() for h in args.heads.split(",") if h.strip()]
    log.info("Heads override: %s", HEADS)

    # Bonferroni now scales with the actual number of heads.
    # Recompute below in main using HEADS.

    panel_path = Path(args.panel)
    out_prefix = Path(args.out_prefix)
    out_dir = out_prefix.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_stem = out_prefix.name

    # Bonferroni: 2 levels x len(HEADS) x (3+4=7 cuts) x 2 filters x 10 cancers
    bonferroni_n_tests = 2 * len(HEADS) * 7 * 2 * 10
    bonf_q = ALPHA / bonferroni_n_tests

    perm_reps = args.perm_reps
    log.info("=== TopX Threshold Sweep V4 ===")
    log.info("Panel: %s", panel_path)
    log.info("Out prefix: %s", out_prefix)
    log.info("Heads: %s", HEADS)
    log.info("Filters: %s", FILTERS)
    log.info("Top pcts: %s; pscore qs: %s; perm_reps: %d; "
             "Bonferroni q=%.2e (n_tests=%d)",
             TOP_PCTS, PSCORE_QS, perm_reps, bonf_q, bonferroni_n_tests)

    t_start = time.time()

    # 1. Load + annotate panel.
    log.info("Loading panel ...")
    panel = pd.read_parquet(panel_path)
    log.info("  panel rows: %d", len(panel))
    missing = [h for h in HEADS if h not in panel.columns]
    if missing:
        raise ValueError(f"Missing head columns: {missing}")
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
    filter_sets = build_filter_sets(maf)

    # 3. Build window-level (max, w=1000) units.
    windows = build_window_max_w1000(panel, HEADS, WINDOW_SIZE)

    # 4. For each level (position, window_max_w1000) write baselines + scores.
    log.info("Dumping per-level data ...")
    levels = []  # (level_id, units_df)
    levels.append(("position", panel))
    levels.append(("window_max_w1000", windows))

    cell_specs = []  # work units
    null_specs = []  # null jobs
    null_meta = {}   # (level_id, filter, cut_id) -> {n_units, k, mut_path}

    for level_id, units in levels:
        u_chrom = units["chrom"].to_numpy()
        if level_id == "window_max_w1000":
            u_start = units["win_start"].to_numpy()
            unit_lookup = pd.DataFrame({"chrom": u_chrom, "win_start": u_start,
                                        "_uidx": np.arange(len(units))})
            tcw_dens = units["tcw_density_panel"].to_numpy(dtype=np.float64)
            npos = units["n_pos_panel"].to_numpy(dtype=np.float64)
        else:
            u_pos = units["pos"].astype(int).to_numpy()
            unit_lookup = pd.DataFrame({"chrom": u_chrom, "pos": u_pos,
                                        "_uidx": np.arange(len(units))})
            tcw_dens = units["is_TCW_C"].to_numpy(dtype=np.float64)
            npos = np.ones(len(units), dtype=np.float64)
        n_units = len(units)

        # baselines file (shared across heads/filters)
        bl_path = out_dir / f"_{out_stem}_baselines_{level_id}.npz"
        np.savez(bl_path, tcw=tcw_dens, npos=npos)

        # save scores per head + compute thresholds & k for each cut
        score_paths = {}
        cuts_per_head = {}  # head -> list[(cut_id, cut_type, cut_value, k, threshold)]
        for head in HEADS:
            scores = units[head].to_numpy(dtype=np.float64)
            sp = out_dir / f"_{out_stem}_scores_{level_id}_{head}.npz"
            np.savez(sp, scores=scores)
            score_paths[head] = str(sp)

            cuts = []
            for tp in TOP_PCTS:
                k = max(1, int(round(n_units * tp)))
                # threshold for top-k
                thr = float(np.partition(-scores, k - 1)[k - 1])
                # ^ this gives -kth largest; convert: top-k threshold is the kth largest
                # actually easier:
                thr = float(np.sort(scores)[-k])
                cuts.append({"cut_id": f"top_{tp:.2f}",
                             "cut_type": "top_pct",
                             "cut_value": tp,
                             "k": k, "threshold": thr})
            for q in PSCORE_QS:
                thr = float(np.quantile(scores, q))
                k = int((scores >= thr).sum())
                cuts.append({"cut_id": f"P{int(q*100)}",
                             "cut_type": "pscore",
                             "cut_value": q,
                             "k": k, "threshold": thr})
            cuts_per_head[head] = cuts
            log.info("  [%s/%s] cuts: %s",
                     level_id, head,
                     [(c["cut_id"], c["k"], f"thr={c['threshold']:.4f}") for c in cuts])

        # for each filter, build mut arrays
        for filter_name, mdf in filter_sets.items():
            m = mdf[["chrom", "pos", "cancer"]].copy()
            m["pos"] = m["pos"].astype(int)
            if level_id == "window_max_w1000":
                m["win_start"] = (m["pos"] // WINDOW_SIZE) * WINDOW_SIZE
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
            log.info("    [%s/%s] muts=%d", level_id, filter_name,
                     int(sum(arr.sum() for arr in mut_arrays.values())))

            # build cells, queue null jobs
            for head in HEADS:
                for c in cuts_per_head[head]:
                    cut_id = c["cut_id"]
                    k = c["k"]
                    if k <= 0 or k >= n_units:
                        log.warning("    skip degenerate cut %s/%s/%s/%s k=%d n=%d",
                                    level_id, head, filter_name, cut_id, k, n_units)
                        continue
                    cov_mb = panel_coverage_mb(level_id, k, WINDOW_SIZE)

                    # null path: depends only on (level, filter, k)
                    null_id = f"{level_id}_{filter_name}_k{k}"
                    null_path = out_dir / f"_{out_stem}_null_{null_id}.npz"

                    if (level_id, filter_name, k) not in null_meta:
                        null_meta[(level_id, filter_name, k)] = {
                            "n_units": n_units, "k": k,
                            "mut_path": str(mut_path),
                            "null_path": str(null_path),
                        }
                        for cancer in CANCERS:
                            mut = mut_arrays[cancer]
                            if int(mut.sum()) == 0:
                                continue
                            seed = SEED_BASE + hash(
                                (level_id, filter_name, cancer, k)) % 100_000
                            null_specs.append((level_id, filter_name, cancer,
                                               n_units, k, str(mut_path),
                                               perm_reps, seed))

                    cell_specs.append((
                        level_id, head, c["cut_type"], c["cut_value"],
                        filter_name, score_paths[head], str(bl_path),
                        str(mut_path), str(null_path), k, bonf_q,
                        k, cov_mb,
                    ))

    # 4b. Run null distributions in parallel.
    log.info("Sampling null distributions: %d jobs (perm_reps=%d) ...",
             len(null_specs), perm_reps)
    t_null = time.time()
    nulls_by_key = {}  # (level_id, filter_name, cancer, k) -> null_arr
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {}
        for spec in null_specs:
            (lid, fn, can, n_units, k, mp, prs, seed) = spec
            f = ex.submit(_null_worker_from_path, spec)
            futures[f] = (lid, fn, can, k)
        n_done = 0
        for f in as_completed(futures):
            (lid, fn, can, k) = futures[f]
            arr = f.result()
            nulls_by_key[(lid, fn, can, k)] = arr
            n_done += 1
            if n_done % 50 == 0:
                log.info("  null %d/%d (%.1fs)", n_done, len(null_specs),
                         time.time() - t_null)
    log.info("Null sampling complete in %.1fs.", time.time() - t_null)

    # 4c. Save null arrays per (level, filter, k).
    for (lid, fn, k), meta in null_meta.items():
        null_arrays = {}
        for cancer in CANCERS:
            arr = nulls_by_key.get((lid, fn, cancer, k))
            if arr is None:
                arr = np.zeros(perm_reps, dtype=np.int64)
            null_arrays[cancer] = arr
        np.savez(meta["null_path"], **null_arrays)

    # 5. Run cells in parallel.
    log.info("Running %d cell evaluations ...", len(cell_specs))
    t_cell = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_process_cell, cs): cs[:5] for cs in cell_specs}
        n_done = 0
        for f in as_completed(futures):
            res = f.result()
            if res is None:
                continue
            results.append(res)
            n_done += 1
            if n_done % 10 == 0:
                log.info("  [%d/%d] last: %s/%s/%s=%.3f filt=%s rec=%.4f "
                         "vs_TCW=%.2f vs_NPOS=%.2f bonf=%d/10",
                         n_done, len(cell_specs),
                         res["level"], res["head"], res["cut_type"],
                         res["cut_value"], res["filter"],
                         res["mean_abs_recall"], res["mean_ratio_tcw"],
                         res["mean_ratio_npos"], res["n_bonf_signif"])
    log.info("Cells complete in %.1fs.", time.time() - t_cell)

    # 6. Build output rows.
    rows = []
    for r in results:
        rows.append({
            "level": r["level"],
            "head": r["head"],
            "cut_type": r["cut_type"],
            "cut_value": r["cut_value"],
            "filter": r["filter"],
            "panel_units": r["panel_units"],
            "panel_coverage_Mb": r["panel_coverage_Mb"],
            "n_units_total": r["n_units"],
            "k": r["k"],
            "n_cancers": r["n_cancers"],
            "mean_abs_recall": r["mean_abs_recall"],
            "abs_recall_ci_lo": r["abs_recall_ci_lo"],
            "abs_recall_ci_hi": r["abs_recall_ci_hi"],
            "mean_ratio_vs_TCW": r["mean_ratio_tcw"],
            "ratio_tcw_ci_lo": r["ratio_tcw_ci_lo"],
            "ratio_tcw_ci_hi": r["ratio_tcw_ci_hi"],
            "mean_ratio_vs_NPOS": r["mean_ratio_npos"],
            "ratio_npos_ci_lo": r["ratio_npos_ci_lo"],
            "ratio_npos_ci_hi": r["ratio_npos_ci_hi"],
            "n_cancers_above_TCW": r["n_above_tcw"],
            "n_cancers_above_NPOS": r["n_above_npos"],
            "n_cancers_bonf_signif": r["n_bonf_signif"],
        })
    df = pd.DataFrame(rows)
    df = df.sort_values(["filter", "level", "head", "cut_type", "cut_value"]) \
        .reset_index(drop=True)
    out_csv = out_dir / f"{out_stem}.csv"
    df.to_csv(out_csv, index=False)
    log.info("Wrote %s (%d rows)", out_csv, len(df))

    # 7. Cleanup tmp files.
    log.info("Cleaning up temp files ...")
    for p in out_dir.glob(f"_{out_stem}_*"):
        try:
            p.unlink()
        except Exception:
            pass

    log.info("DONE in %.1fs.", time.time() - t_start)
    return 0


if __name__ == "__main__":
    sys.exit(main())
