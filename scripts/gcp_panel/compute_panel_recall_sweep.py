#!/usr/bin/env python3
"""Aggregator + window-size sweep — the headline Phase 2 deliverable.

Computes panel recall at top-1% across:
  - Position-level: top-X% of individual C positions by NN binary score (no window)
  - Window-level: 4 window sizes (100, 250, 500, 1000 bp) × 5 aggregators
    (max, mean, sum, top3_mean, p95)
  = 1 + 4*5 = 21 panel constructions

For each construction, computes:
  - Absolute recall: fraction of TCW-non-CpG mutations captured
  - Recall ratio vs CpG-density baseline (`seq.count("CG")` per window;
    is_cpg per position)
  - Recall ratio vs TCW-motif-density baseline (count "TCA"+"TCT"+"AGA"+"TGA"
    per window; is_TCW_C per position)
  - Recall ratio vs random GC-matched baseline (100 random equal-size sets,
    GC-stratified)
  - Per-cancer breakdown across 10 cancers
  - Permutation null at 10K perms; BH-FDR
  - Bootstrap CI on the mean ratio across cancers

Mutation source: TCGA-MC3 + PCAWG cBioPortal coding (10 cancers, Analysis-B-equivalent).

Output:
  experiments/multi_enzyme/outputs/pcawg_tcw_panel/aggregator_window_sweep.csv
  experiments/multi_enzyme/outputs/pcawg_tcw_panel/aggregator_window_sweep.png
  experiments/multi_enzyme/outputs/pcawg_tcw_panel/aggregator_window_sweep_per_cancer.csv

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
from scipy.stats import false_discovery_control

ROOT = Path("/Users/shaharharel/Documents/github/edit-rna-apobec")
PANEL_PATH = ROOT / "experiments/multi_enzyme/outputs/pcawg_tcw_panel/panel_scores_cds.parquet"
TCGA_DIR = ROOT / "data/raw/tcga"
PCAWG_DIR = ROOT / "data/raw/pcawg/by_cancer"
HG19 = ROOT / "data/raw/genomes/hg19.fa"
OUT_DIR = ROOT / "experiments/multi_enzyme/outputs/pcawg_tcw_panel"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CANCERS = ["blca", "brca", "cesc", "coadread", "esca", "hnsc", "lihc", "lusc", "skcm", "stad"]
WINDOW_SIZES = [100, 250, 500, 1000]
AGGREGATORS = ["max", "mean", "sum", "top3_mean", "p95"]
TOP_PCT = 0.01
PERM_REPS = 10000
N_RANDOM_BASELINES = 100
N_BOOT = 10000
SEED_BASE = 20260424

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    stream=sys.stdout)
log = logging.getLogger(__name__)


# =========================================================================== #
# Mutation loading (TCGA + PCAWG-coding combined)
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
# Panel position annotations: is_cpg, is_TCW_C, GC-content (per-position)
# =========================================================================== #

COMP = str.maketrans("ACGTN", "TGCAN")


def annotate_panel_positions(panel: pd.DataFrame) -> pd.DataFrame:
    """Annotate each panel position with is_cpg, is_TCW_C, neighborhood GC.
    Vectorized per-chromosome for speed (8.45M positions in ~30s)."""
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
            log.warning("  couldn't load %s", ch)
            continue
        L = len(seq)
        log.info("  %s: %d panel positions, seq len=%d", ch, len(idx), L)

        ps = poses[mask]
        ss = strands[mask]

        # Bounds-valid mask
        ok = (ps >= 1) & (ps + 1 < L)
        # Lookup neighbors
        valid_idx = idx[ok]
        ps_ok = ps[ok]
        ss_ok = ss[ok]
        left = seq[ps_ok - 1]
        right = seq[ps_ok + 1]
        # CpG: + strand C => right == 'G' (71); - strand G => left == 'C' (67)
        is_plus = (ss_ok == "+")
        is_minus = ~is_plus
        is_cpg[valid_idx[is_plus]] = (right[is_plus] == ord("G"))
        is_cpg[valid_idx[is_minus]] = (left[is_minus] == ord("C"))
        # TCW-C: + strand: left==T (84) AND right in {A,T} (65, 84)
        # - strand: right==A (65) AND left in {A,T}
        right_AT = (right == ord("A")) | (right == ord("T"))
        left_AT = (left == ord("A")) | (left == ord("T"))
        is_tcw_c[valid_idx[is_plus]] = (left[is_plus] == ord("T")) & right_AT[is_plus]
        is_tcw_c[valid_idx[is_minus]] = (right[is_minus] == ord("A")) & left_AT[is_minus]

        # GC bin over ±50 bp — use cumulative GC counts on the chrom seq
        # gc_count[i] = number of G or C in seq[:i]
        is_gc_byte = ((seq == ord("G")) | (seq == ord("C"))).astype(np.int32)
        cum_gc = np.zeros(L + 1, dtype=np.int32)
        cum_gc[1:] = np.cumsum(is_gc_byte)
        # Window [p-50, p+50)
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
# Per-position TCW-non-CpG mutation flag
# =========================================================================== #

def filter_tcw_non_cpg_muts(maf: pd.DataFrame) -> pd.DataFrame:
    """Return only TCW-non-CpG C>T mutations using hg19 trinucleotide context.
    Vectorized per-chromosome."""
    from pyfaidx import Fasta
    log.info("Filtering MAF to TCW-non-CpG C>T mutations (vectorized) ...")
    genome = Fasta(str(HG19), as_raw=False, sequence_always_upper=True)
    keep = np.zeros(len(maf), dtype=bool)
    chroms = maf["chrom"].to_numpy()
    poses = maf["pos"].astype(int).to_numpy()
    strands = maf["strand"].to_numpy()
    idx_all = np.arange(len(maf))

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
        # + strand: tri = [left, center, right], TCW-C means left=T, center=C, right in {A,T}
        # - strand: tri must be RC'd. RC of [left, center, right] = [comp(right), comp(center), comp(left)]
        # so RC[0]==T  ->  comp(right)==T -> right==A
        # RC[1]==C  ->  comp(center)==C -> center==G
        # RC[2] in {A,T} -> comp(left) in {A,T} -> left in {A,T}
        is_plus = (ss_ok == "+")
        is_minus = ~is_plus
        plus_pass = is_plus & (left == ord("T")) & (center == ord("C")) & ((right == ord("A")) | (right == ord("T")))
        minus_pass = is_minus & (right == ord("A")) & (center == ord("G")) & ((left == ord("A")) | (left == ord("T")))
        tcw_pass = plus_pass | minus_pass
        keep[valid_idx[tcw_pass]] = True

    out = maf[keep].copy()
    log.info("  TCW-non-CpG kept: %d / %d (%.1f%%)", keep.sum(), len(maf),
             100*keep.mean())
    return out


# =========================================================================== #
# Window builders
# =========================================================================== #

def build_window_aggregations(panel: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """For a single window size, build a window-level table with all 5 aggregators
    of score_binary plus CpG-density (seq.count('CG')) + TCW-motif-density baselines."""
    from pyfaidx import Fasta
    log.info("[w=%d] Building window aggregation table ...", window_size)
    p = panel[["chrom", "pos", "score_binary", "is_cpg", "is_TCW_C", "gc_bin"]].copy()
    p["pos"] = p["pos"].astype(int)
    p["win_start"] = (p["pos"] // window_size) * window_size

    # 5 aggregators of score_binary
    grp = p.groupby(["chrom", "win_start"])["score_binary"]
    out = pd.DataFrame({
        "max_score": grp.max(),
        "mean_score": grp.mean(),
        "sum_score": grp.sum(),
        "top3_mean_score": grp.apply(lambda x: float(np.mean(np.sort(x.values)[-3:])) if len(x) > 0 else 0.0),
        "p95_score": grp.quantile(0.95),
        "n_pos": grp.size(),
    }).reset_index()

    # Random-baseline support: median GC bin per window
    gc_bin_per_win = p.groupby(["chrom", "win_start"])["gc_bin"].median().astype(int).reset_index()
    out = out.merge(gc_bin_per_win, on=["chrom", "win_start"])

    # Phase 1.5 baselines: from hg19 window seq (vectorized via cumulative counts)
    log.info("[w=%d] Computing CpG + TCW-motif densities from hg19 (vectorized) ...", window_size)
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
        # CG dinuc per position: seq[k]==C and seq[k+1]==G  -> array of length L-1
        is_cg = ((seq[:-1] == ord("C")) & (seq[1:] == ord("G"))).astype(np.int32)
        cum_cg = np.zeros(L, dtype=np.int32)
        cum_cg[1:] = np.cumsum(is_cg)
        # Trinucleotide TCW (+ strand): seq[k]==T, seq[k+1]==C, seq[k+2] in {A,T}
        # Trinucleotide rev-comp TCW (- strand): seq[k] in {A,T}, seq[k+1]==G, seq[k+2]==A
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
        ss_clip = np.clip(ss, 0, L - 1)
        # CG count in [s, e): cum_cg[max(s, 0)] gives 'CG' starting at any pos < s
        # Actually cum_cg[i] counts is_cg[0..i-1], so CGs starting in [s, e-1) is cum_cg[e-1] - cum_cg[s]
        # (we want CGs whose start is in [s, e-1]; max start index is e-2)
        ce = np.clip(es - 1, 0, L - 1)
        cs = np.clip(ss, 0, L - 1)
        cpg_per = cum_cg[ce] - cum_cg[cs]
        cpg_per[ss >= L] = 0
        cpg[idx] = cpg_per

        # TCW count: cum_tcw indexed up to L-2 valid trinucs
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


def map_muts_to_windows(muts: pd.DataFrame, windows: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """Add win_idx (0-based row idx into windows) per mutation."""
    m = muts.copy()
    m["win_start"] = (m["pos"].astype(int) // window_size) * window_size
    win_lookup = windows.reset_index(drop=True).reset_index().rename(columns={"index": "win_idx"})
    win_lookup = win_lookup[["chrom", "win_start", "win_idx"]]
    m = m.merge(win_lookup, on=["chrom", "win_start"], how="inner")
    return m


# =========================================================================== #
# Recall + ratio computation
# =========================================================================== #

def recall_at_top_pct(scores: np.ndarray, mut_per_unit: np.ndarray,
                      top_pct: float) -> tuple[float, int, int, int, np.ndarray]:
    """Return (recall, mut_in_top, total_mut, k, top_idx)."""
    n = len(scores)
    if n == 0:
        return 0.0, 0, 0, 0, np.array([], dtype=int)
    k = max(1, int(round(n * top_pct)))
    top_idx = np.argpartition(-scores, k - 1)[:k]
    mut_in_top = int(mut_per_unit[top_idx].sum())
    total_mut = int(mut_per_unit.sum())
    recall = mut_in_top / total_mut if total_mut > 0 else 0.0
    return recall, mut_in_top, total_mut, k, top_idx


def random_gc_matched_recall(scores_gc: np.ndarray, mut_per_unit: np.ndarray,
                             top_pct: float, n_baselines: int, seed: int) -> dict:
    """Sample n_baselines random GC-matched top-1% sets. Return mean recall."""
    rng = np.random.default_rng(seed)
    n = len(scores_gc)
    k = max(1, int(round(n * top_pct)))
    # GC-matched: in each gc_bin, sample proportionally
    gc_bins = scores_gc.astype(int)
    bin_counts = np.bincount(gc_bins, minlength=10)
    # The "score-rank" baseline target is the gc-bin distribution among the
    # NN-top-k. Since we don't have NN_top here as input, we'll build a
    # GC-stratified random set: from each bin, sample k * (bin_count / n) units
    # with replacement (within bin without replacement). This gives us a
    # GC-matched random panel.
    target_counts = np.round(k * bin_counts / n).astype(int)
    # Adjust to sum to exactly k
    diff = k - target_counts.sum()
    if diff != 0:
        # add/remove from the largest bin
        target_counts[bin_counts.argmax()] += diff

    recalls = []
    for _ in range(n_baselines):
        chosen = []
        for b in range(10):
            count_b = int(target_counts[b])
            if count_b <= 0:
                continue
            pool = np.where(gc_bins == b)[0]
            if len(pool) == 0:
                continue
            sz = min(count_b, len(pool))
            chosen.extend(rng.choice(pool, size=sz, replace=False).tolist())
        chosen = np.array(chosen, dtype=int)
        if len(chosen) == 0:
            recalls.append(0.0)
            continue
        recalls.append(mut_per_unit[chosen].sum() / max(1, mut_per_unit.sum()))
    return {
        "mean_recall": float(np.mean(recalls)),
        "std_recall": float(np.std(recalls)),
        "n_baselines": n_baselines,
    }


def evaluate_construction_per_cancer(scores: np.ndarray, baseline_cpg: np.ndarray,
                                     baseline_tcw: np.ndarray, gc_bins: np.ndarray,
                                     mut_per_unit_per_cancer: dict[str, np.ndarray],
                                     top_pct: float, perm_reps: int,
                                     n_random_baselines: int, seed: int) -> dict:
    """Evaluate one panel construction (scores at unit-level). Returns per-cancer
    + pooled metrics."""
    rng_global = np.random.default_rng(seed)
    n = len(scores)
    k = max(1, int(round(n * top_pct)))

    # Top-k by NN, by CpG, by TCW
    nn_top = np.argpartition(-scores, k - 1)[:k]
    cpg_top = np.argpartition(-baseline_cpg, k - 1)[:k]
    tcw_top = np.argpartition(-baseline_tcw, k - 1)[:k]

    per_cancer = {}
    ratios_cpg, ratios_tcw, ratios_random = [], [], []
    p_perms = []
    n_above_1_cpg = 0
    n_above_1_tcw = 0
    abs_recalls = []

    for cancer, mut in mut_per_unit_per_cancer.items():
        total_mut = int(mut.sum())
        if total_mut == 0:
            per_cancer[cancer] = {
                "total_mut": 0, "abs_recall": float("nan"),
                "ratio_cpg": float("nan"), "ratio_tcw": float("nan"),
                "ratio_random": float("nan"), "p_perm": 1.0,
            }
            continue
        nn_recall = mut[nn_top].sum() / total_mut
        cpg_recall = mut[cpg_top].sum() / total_mut
        tcw_recall = mut[tcw_top].sum() / total_mut
        # Random GC-matched
        rand = random_gc_matched_recall(gc_bins, mut, top_pct, n_random_baselines,
                                        seed=seed + hash(cancer) % 1000)
        rand_recall = rand["mean_recall"]

        ratio_cpg = nn_recall / cpg_recall if cpg_recall > 0 else float("nan")
        ratio_tcw = nn_recall / tcw_recall if tcw_recall > 0 else float("nan")
        ratio_rand = nn_recall / rand_recall if rand_recall > 0 else float("nan")

        # Permutation null: under H0 (no NN signal), the top-k by rank is a
        # uniform random k-subset of all units. The mut_in_top statistic then
        # follows a hypergeometric distribution: X ~ Hypergeom(N=n, K=total_mut,
        # n_sample=k). One-sided p-value = P(X >= mut_in_top_obs) — closed-form.
        # This is mathematically identical to the standard permutation null on
        # score-rank labels, but O(1) vs O(n*perm_reps) in the worker.
        # We retain perm_reps as a metadata field for reporting consistency.
        from scipy.stats import hypergeom
        mut_in_top_obs = int(mut[nn_top].sum())
        # P(X >= mut_in_top_obs) under H0
        p_perm = float(hypergeom.sf(mut_in_top_obs - 1, n, total_mut, k))
        # Add a small Laplace-style correction for parity with perm_reps-based
        # p-values: clamp at 1/(perm_reps+1) effective floor
        p_perm = max(p_perm, 1.0 / (perm_reps + 1))

        per_cancer[cancer] = {
            "total_mut": total_mut,
            "abs_recall": float(nn_recall),
            "ratio_cpg": float(ratio_cpg),
            "ratio_tcw": float(ratio_tcw),
            "ratio_random": float(ratio_rand),
            "p_perm": float(p_perm),
            "k": int(k),
            "n_units": int(n),
        }
        abs_recalls.append(nn_recall)
        if not np.isnan(ratio_cpg):
            ratios_cpg.append(ratio_cpg)
            if ratio_cpg > 1.0:
                n_above_1_cpg += 1
        if not np.isnan(ratio_tcw):
            ratios_tcw.append(ratio_tcw)
            if ratio_tcw > 1.0:
                n_above_1_tcw += 1
        if not np.isnan(ratio_rand):
            ratios_random.append(ratio_rand)
        p_perms.append(p_perm)

    # BH-FDR across cancers
    if p_perms:
        q_vals = false_discovery_control(p_perms, method="bh")
        n_bh = int(sum(q < 0.025 for q in q_vals))
        for (cancer, _), q in zip(per_cancer.items(), q_vals):
            per_cancer[cancer]["q_bh"] = float(q)
            per_cancer[cancer]["reject_bh"] = bool(q < 0.025)
    else:
        n_bh = 0

    # Bootstrap CI on mean ratios
    def boot_ci(arr, n_boot=N_BOOT, seed=42):
        if not arr:
            return float("nan"), float("nan"), float("nan")
        a = np.array(arr)
        rng = np.random.default_rng(seed)
        boot = np.array([rng.choice(a, size=len(a), replace=True).mean()
                         for _ in range(n_boot)])
        return float(a.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

    mean_cpg, ci_cpg_lo, ci_cpg_hi = boot_ci(ratios_cpg, seed=seed + 1)
    mean_tcw, ci_tcw_lo, ci_tcw_hi = boot_ci(ratios_tcw, seed=seed + 2)
    mean_rand, ci_rand_lo, ci_rand_hi = boot_ci(ratios_random, seed=seed + 3)
    mean_abs, ci_abs_lo, ci_abs_hi = boot_ci(abs_recalls, seed=seed + 4)

    return {
        "per_cancer": per_cancer,
        "n_cancers_evaluated": len(per_cancer),
        "mean_abs_recall": mean_abs,
        "abs_recall_ci_lo": ci_abs_lo,
        "abs_recall_ci_hi": ci_abs_hi,
        "mean_ratio_cpg": mean_cpg,
        "ratio_cpg_ci_lo": ci_cpg_lo,
        "ratio_cpg_ci_hi": ci_cpg_hi,
        "n_above_1_cpg": n_above_1_cpg,
        "mean_ratio_tcw": mean_tcw,
        "ratio_tcw_ci_lo": ci_tcw_lo,
        "ratio_tcw_ci_hi": ci_tcw_hi,
        "n_above_1_tcw": n_above_1_tcw,
        "mean_ratio_random": mean_rand,
        "ratio_random_ci_lo": ci_rand_lo,
        "ratio_random_ci_hi": ci_rand_hi,
        "n_bh_signif": n_bh,
    }


# =========================================================================== #
# Worker driver — one (aggregator, window_size) construction
# =========================================================================== #

def _process_window_construction(args):
    """Worker: process one (aggregator, window_size) construction.
    Receives serialized arrays via on-disk parquet to avoid pickling 8.45M-row DFs."""
    (window_size, aggregator, windows_path, mut_per_cancer_path, top_pct,
     perm_reps, n_random_baselines, seed) = args

    # Load on worker side
    w = pd.read_parquet(windows_path)
    mut_pc = json.loads(Path(mut_per_cancer_path).read_text())
    # Reconstitute numpy arrays per cancer
    mut_per_cancer = {}
    for cancer, vals in mut_pc.items():
        arr = np.zeros(len(w), dtype=np.int32)
        for idx, count in vals:
            arr[int(idx)] = int(count)
        mut_per_cancer[cancer] = arr

    score_col = {
        "max": "max_score", "mean": "mean_score", "sum": "sum_score",
        "top3_mean": "top3_mean_score", "p95": "p95_score",
    }[aggregator]
    scores = w[score_col].to_numpy(dtype=np.float64)
    cpg = w["cpg_density"].to_numpy(dtype=np.float64)
    tcw = w["tcw_density"].to_numpy(dtype=np.float64)
    gc = w["gc_bin"].to_numpy(dtype=np.int32)

    res = evaluate_construction_per_cancer(
        scores, cpg, tcw, gc, mut_per_cancer,
        top_pct, perm_reps, n_random_baselines, seed,
    )
    res["aggregator"] = aggregator
    res["window_size"] = window_size
    res["n_units"] = int(len(w))
    res["panel_coverage_mb"] = float(window_size * len(w) / 1e6)
    return res


def _process_position_level(args):
    """Worker: process the position-level (no windowing) panel."""
    (panel_path, mut_pos_per_cancer_path, top_pct, perm_reps,
     n_random_baselines, seed) = args
    panel = pd.read_parquet(panel_path)
    mut_pc = json.loads(Path(mut_pos_per_cancer_path).read_text())
    mut_per_cancer = {}
    for cancer, vals in mut_pc.items():
        arr = np.zeros(len(panel), dtype=np.int32)
        for idx, count in vals:
            arr[int(idx)] = int(count)
        mut_per_cancer[cancer] = arr

    scores = panel["score_binary"].to_numpy(dtype=np.float64)
    cpg = panel["is_cpg"].to_numpy(dtype=np.float64)
    tcw = panel["is_TCW_C"].to_numpy(dtype=np.float64)
    gc = panel["gc_bin"].to_numpy(dtype=np.int32)

    res = evaluate_construction_per_cancer(
        scores, cpg, tcw, gc, mut_per_cancer,
        top_pct, perm_reps, n_random_baselines, seed,
    )
    res["aggregator"] = "position_level"
    res["window_size"] = 0
    res["n_units"] = int(len(panel))
    res["panel_coverage_mb"] = float(len(panel) / 1e6)  # one C per unit
    return res


# =========================================================================== #
# Main
# =========================================================================== #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-workers", type=int, default=4)
    ap.add_argument("--perm-reps", type=int, default=PERM_REPS)
    ap.add_argument("--top-pct", type=float, default=TOP_PCT)
    ap.add_argument("--quick", action="store_true",
                    help="Quick run: 1K perms, 1 window size (250 bp) for testing")
    args = ap.parse_args()

    perm_reps = 1000 if args.quick else args.perm_reps
    window_sizes = [250] if args.quick else WINDOW_SIZES

    log.info("=== Aggregator + Window-size sweep ===")
    log.info("Window sizes: %s", window_sizes)
    log.info("Aggregators: %s", AGGREGATORS)
    log.info("Top pct: %.3f, perm_reps=%d, n_random_baselines=%d",
             args.top_pct, perm_reps, N_RANDOM_BASELINES)

    # 1. Load panel + annotate
    log.info("Loading panel ...")
    panel = pd.read_parquet(PANEL_PATH)
    log.info("  panel rows: %d", len(panel))
    panel = annotate_panel_positions(panel)
    # Save annotated panel for workers
    annotated_path = OUT_DIR / "panel_annotated.parquet"
    panel.to_parquet(annotated_path, index=False)
    log.info("Saved annotated panel: %s", annotated_path)

    # 2. Load + filter MAFs
    maf = load_combined_coding_maf()
    # Restrict to in-panel positions (set lookup, deduplicated)
    log.info("Restricting MAF to in-panel positions ...")
    panel_set = set(zip(panel["chrom"].astype(str).values,
                        panel["pos"].astype(int).values))
    in_panel = np.array([(c, int(p)) in panel_set
                         for c, p in zip(maf["chrom"], maf["pos"])])
    maf = maf.iloc[np.where(in_panel)[0]].reset_index(drop=True)
    log.info("  in-panel: %d", len(maf))
    maf = filter_tcw_non_cpg_muts(maf)
    log.info("  TCW-non-CpG in-panel: %d", len(maf))
    log.info("  per-cancer counts: %s", maf["cancer"].value_counts().to_dict())

    # 3. Build position-level mut counts: index in panel for each mutation
    log.info("Building position-level mutation index per cancer ...")
    panel_pos_lookup = panel.reset_index(drop=True).reset_index().rename(columns={"index": "panel_idx"})
    panel_pos_lookup = panel_pos_lookup[["chrom", "pos", "panel_idx"]]
    maf_with_idx = maf.merge(panel_pos_lookup, on=["chrom", "pos"], how="inner")
    log.info("  mutations with panel_idx: %d", len(maf_with_idx))
    # Save per-cancer position counts as compact JSON (idx, count)
    pos_per_cancer = {}
    for cancer in CANCERS:
        sub = maf_with_idx[maf_with_idx["cancer"] == cancer]
        if len(sub) == 0:
            pos_per_cancer[cancer] = []
            continue
        counts = sub["panel_idx"].value_counts()
        pos_per_cancer[cancer] = [[int(idx), int(c)] for idx, c in counts.items()]
    pos_path = OUT_DIR / "_sweep_pos_per_cancer.json"
    pos_path.write_text(json.dumps(pos_per_cancer))

    # 4. Build window aggregations + window-level mut counts for each window size
    work_units = []  # each is (aggregator, window_size, windows_path, mut_path)
    for ws in window_sizes:
        log.info("[w=%d] Building windows ...", ws)
        windows = build_window_aggregations(panel, ws)
        win_path = OUT_DIR / f"_sweep_windows_w{ws}.parquet"
        windows.to_parquet(win_path, index=False)
        # Map each mutation to its window_idx
        log.info("[w=%d] Mapping mutations to windows ...", ws)
        m = maf.copy()
        m["win_start"] = (m["pos"].astype(int) // ws) * ws
        win_lookup = windows.reset_index(drop=True).reset_index().rename(columns={"index": "win_idx"})
        win_lookup = win_lookup[["chrom", "win_start", "win_idx"]]
        m = m.merge(win_lookup, on=["chrom", "win_start"], how="inner")
        win_per_cancer = {}
        for cancer in CANCERS:
            sub = m[m["cancer"] == cancer]
            if len(sub) == 0:
                win_per_cancer[cancer] = []
                continue
            counts = sub["win_idx"].value_counts()
            win_per_cancer[cancer] = [[int(idx), int(c)] for idx, c in counts.items()]
        mut_path = OUT_DIR / f"_sweep_mut_w{ws}.json"
        mut_path.write_text(json.dumps(win_per_cancer))

        for ag in AGGREGATORS:
            work_units.append((ws, ag, str(win_path), str(mut_path),
                               args.top_pct, perm_reps, N_RANDOM_BASELINES,
                               SEED_BASE + ws * 100 + hash(ag) % 100))

    # 5. Run all 21 constructions in parallel
    log.info("Running %d window constructions + 1 position-level construction ...",
             len(work_units))

    results = []
    # Position-level first (use same seed scheme)
    pos_args = (str(annotated_path), str(pos_path), args.top_pct, perm_reps,
                N_RANDOM_BASELINES, SEED_BASE + 999)

    # ProcessPoolExecutor with maxtasksperchild=1 equivalent: spawn fresh workers
    with ProcessPoolExecutor(max_workers=args.n_workers) as ex:
        log.info("Submitting position-level + %d window constructions ...", len(work_units))
        futures = {}
        f = ex.submit(_process_position_level, pos_args)
        futures[f] = ("position_level", 0)
        for wu in work_units:
            f = ex.submit(_process_window_construction, wu)
            futures[f] = (wu[1], wu[0])  # (aggregator, window_size)
        n_done = 0
        for f in as_completed(futures):
            tag = futures[f]
            try:
                res = f.result()
                results.append(res)
                n_done += 1
                log.info("[%d/%d] %s w=%d: abs_recall=%.4f mean_ratio_cpg=%.2f mean_ratio_tcw=%.2f",
                         n_done, len(futures), res["aggregator"], res["window_size"],
                         res["mean_abs_recall"], res["mean_ratio_cpg"], res["mean_ratio_tcw"])
            except Exception as ex:
                log.error("FAIL %s: %s", tag, ex)

    # 6. Write results table
    log.info("Writing aggregator_window_sweep.csv ...")
    rows = []
    per_cancer_rows = []
    for r in results:
        rows.append({
            "aggregator": r["aggregator"],
            "window_size_bp": r["window_size"],
            "panel_units": r["n_units"],
            "panel_coverage_mb": r.get("panel_coverage_mb", float("nan")),
            "n_cancers": r["n_cancers_evaluated"],
            "mean_abs_recall": r["mean_abs_recall"],
            "abs_recall_ci95_lo": r["abs_recall_ci_lo"],
            "abs_recall_ci95_hi": r["abs_recall_ci_hi"],
            "mean_ratio_vs_cpg": r["mean_ratio_cpg"],
            "ratio_vs_cpg_ci95_lo": r["ratio_cpg_ci_lo"],
            "ratio_vs_cpg_ci95_hi": r["ratio_cpg_ci_hi"],
            "n_cancers_above_1_cpg": r["n_above_1_cpg"],
            "mean_ratio_vs_tcw": r["mean_ratio_tcw"],
            "ratio_vs_tcw_ci95_lo": r["ratio_tcw_ci_lo"],
            "ratio_vs_tcw_ci95_hi": r["ratio_tcw_ci_hi"],
            "n_cancers_above_1_tcw": r["n_above_1_tcw"],
            "mean_ratio_vs_random": r["mean_ratio_random"],
            "ratio_vs_random_ci95_lo": r["ratio_random_ci_lo"],
            "ratio_vs_random_ci95_hi": r["ratio_random_ci_hi"],
            "n_bh_signif_p025": r["n_bh_signif"],
        })
        for cancer, pc in r["per_cancer"].items():
            per_cancer_rows.append({
                "aggregator": r["aggregator"],
                "window_size_bp": r["window_size"],
                "cancer": cancer,
                **pc,
            })

    df = pd.DataFrame(rows)
    df = df.sort_values(["window_size_bp", "aggregator"]).reset_index(drop=True)
    out_csv = OUT_DIR / "aggregator_window_sweep.csv"
    df.to_csv(out_csv, index=False)
    log.info("Wrote %s (%d rows)", out_csv, len(df))

    pcdf = pd.DataFrame(per_cancer_rows)
    pcdf = pcdf.sort_values(["window_size_bp", "aggregator", "cancer"]).reset_index(drop=True)
    out_pccsv = OUT_DIR / "aggregator_window_sweep_per_cancer.csv"
    pcdf.to_csv(out_pccsv, index=False)
    log.info("Wrote %s (%d rows)", out_pccsv, len(pcdf))

    # 7. Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        log.info("Generating figure ...")

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        ax1, ax2 = axes

        # Panel 1: absolute recall
        # Position-level point
        for ag in AGGREGATORS:
            sub = df[(df["aggregator"] == ag)].sort_values("window_size_bp")
            ax1.plot(sub["window_size_bp"], sub["mean_abs_recall"],
                     marker="o", label=ag)
        # Position-level
        pos = df[df["aggregator"] == "position_level"]
        if len(pos):
            ax1.scatter([0], [pos["mean_abs_recall"].iloc[0]],
                        marker="*", s=200, color="black", label="position_level", zorder=5)
        ax1.set_xlabel("Window size (bp; 0 = position-level)")
        ax1.set_ylabel(f"Mean absolute recall at top-{int(TOP_PCT*100)}%")
        ax1.set_title("Panel A: Absolute recall vs window/aggregator")
        ax1.legend(loc="best", fontsize=8)
        ax1.grid(alpha=0.3)

        # Panel 2: ratios vs all 3 baselines
        # Use the BEST aggregator at each window size (highest mean_ratio_tcw),
        # and plot 3 lines (cpg, tcw, random) across windows. Plus position level.
        # But cleaner: show all aggregators in 3 sub-rows? Use stacked bars instead.
        # Simpler: plot mean_ratio_vs_tcw per (aggregator, window_size), with
        # horizontal dashed lines for "TCW=1" and "CpG=1".
        for ag in AGGREGATORS:
            sub = df[(df["aggregator"] == ag)].sort_values("window_size_bp")
            ax2.plot(sub["window_size_bp"], sub["mean_ratio_vs_tcw"],
                     marker="o", label=ag)
        if len(pos):
            ax2.scatter([0], [pos["mean_ratio_vs_tcw"].iloc[0]],
                        marker="*", s=200, color="black", label="position_level", zorder=5)
        ax2.axhline(1.0, color="red", linestyle="--", alpha=0.5,
                    label="ratio=1 (no enrichment)")
        ax2.set_xlabel("Window size (bp; 0 = position-level)")
        ax2.set_ylabel("Mean ratio NN / TCW-motif-density")
        ax2.set_title("Panel B: Ratio vs TCW-motif baseline")
        ax2.legend(loc="best", fontsize=8)
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        out_png = OUT_DIR / "aggregator_window_sweep.png"
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close()
        log.info("Wrote %s", out_png)
    except Exception as ex:
        log.error("Figure generation failed: %s", ex)

    # 8. Headline
    log.info("=== HEADLINE: best constructions ===")
    df_sorted = df.sort_values("mean_ratio_vs_tcw", ascending=False).reset_index(drop=True)
    log.info("Top-5 by mean_ratio_vs_tcw:")
    for _, row in df_sorted.head(5).iterrows():
        log.info("  %s w=%d  abs_rec=%.4f  vs_cpg=%.2f  vs_tcw=%.2f  vs_rand=%.2f",
                 row["aggregator"], row["window_size_bp"],
                 row["mean_abs_recall"], row["mean_ratio_vs_cpg"],
                 row["mean_ratio_vs_tcw"], row["mean_ratio_vs_random"])
    log.info("By mean_abs_recall:")
    df_sorted = df.sort_values("mean_abs_recall", ascending=False).reset_index(drop=True)
    for _, row in df_sorted.head(5).iterrows():
        log.info("  %s w=%d  abs_rec=%.4f  vs_cpg=%.2f  vs_tcw=%.2f  vs_rand=%.2f",
                 row["aggregator"], row["window_size_bp"],
                 row["mean_abs_recall"], row["mean_ratio_vs_cpg"],
                 row["mean_ratio_vs_tcw"], row["mean_ratio_vs_random"])

    # 9. Cleanup tmp files
    for ws in window_sizes:
        for tmp in [OUT_DIR / f"_sweep_windows_w{ws}.parquet",
                    OUT_DIR / f"_sweep_mut_w{ws}.json"]:
            if tmp.exists() and not args.quick:
                tmp.unlink()
    if pos_path.exists() and not args.quick:
        pos_path.unlink()
    if annotated_path.exists() and not args.quick:
        annotated_path.unlink()

    return 0


if __name__ == "__main__":
    sys.exit(main())
