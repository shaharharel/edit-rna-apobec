#!/usr/bin/env python3
"""Analysis D v4 — POG570 independent WGS validation against the v4 panel.

This re-validates the v4_cds binary-head panel claim on the POG570 cohort
(BC Cancer Agency Personalised OncoGenomics, n~570 patients), independent of
TCGA/PCAWG.

Methodology mirrors the v4 PCAWG fair-sweep (compute_panel_recall_topx_v4.py)
at the position level, so PCAWG and POG570 are on the same footing:

  - Restrict POG570 to in-panel positions (CDS-C set; v4 panel has 8.45M
    positions across the 10 reference cancers).
  - Annotate trinucleotide context using hg19 (matching panel build).
  - Two mutation filters: TCW_nonCpG, all_CT.
  - Three panel sizes: top-1%, top-5%, top-10% by binary head.
  - Two same-bases baselines (TCW-density and n_pos), counted only over
    panel positions — avoids the gene-density artefact that confounded the
    v3 POG570 attempt.
  - Random baseline = uniform sampling of k positions (per-cancer mean
    recall over 10000 perms used as null and as random reference).
  - Per-cancer ratios with bootstrap 95% CI (10000 reps over cancers),
    matching the PCAWG sweep convention.

Output
------
  pog570_v4_validation/enrichment_v4_cds.csv         (one row per cell)
  pog570_v4_validation/enrichment_v4_cds_per_cancer.csv
  pog570_v4_validation/recall_curve_pog570_v4.png
  pog570_v4_validation/POG570_V4_RESULTS.md
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
HG19 = ROOT / "data/raw/genomes/hg19.fa"
POG570_PATH = ROOT / "data/raw/pog570/POG570_small_mutations.txt.gz"

DEFAULT_PANEL = (
    ROOT
    / "experiments/multi_enzyme/outputs/pcawg_tcw_panel/v4_outputs"
    / "panel_scores_v4_cds_apobec1retrained.parquet"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "experiments/multi_enzyme/outputs/pcawg_tcw_panel/v4_outputs"
    / "pog570_v4_validation"
)

# POG570 analysis_cohort -> reference cancer label used in PCAWG sweep.
COHORT_MAP = {
    "COLO": "coadread",
    "SKCM": "skcm",
    "BRCA": "brca",
    "LUNG": "lusc",  # POG570 doesn't separate LUSC vs LUAD
    "ESCA": "esca",
    "HNSC": "hnsc",
    "STAD": "stad",
    "HCC": "lihc",
    "BLCA": "blca",
    "CERV": "cesc",
}
TARGET_CANCERS = sorted(set(COHORT_MAP.values()))

TOP_PCTS = [0.01, 0.05, 0.10]
PERM_REPS = 10000
N_BOOT = 10000
SEED_BASE = 20260427
ALPHA = 0.05

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


# =========================================================================== #
# Loaders
# =========================================================================== #


def load_panel(path: Path, head_cols: list[str]) -> pd.DataFrame:
    log.info("Loading panel: %s", path)
    df = pd.read_parquet(path)
    keep = ["chrom", "pos", "strand"] + [c for c in head_cols if c in df.columns]
    log.info("  rows: %d  cols: %s", len(df), keep)
    return df[keep].copy()


def load_pog570(path: Path) -> pd.DataFrame:
    log.info("Loading POG570 small mutations from %s", path)
    df = pd.read_csv(path, sep="\t", compression="gzip", low_memory=False)
    log.info("  raw rows: %d", len(df))
    df = df[(df["ref"].str.len() == 1) & (df["alt"].str.len() == 1)].copy()
    df = df[
        ((df["ref"] == "C") & (df["alt"] == "T"))
        | ((df["ref"] == "G") & (df["alt"] == "A"))
    ]
    log.info("  C>T/G>A SNVs: %d", len(df))
    df["cancer"] = df["analysis_cohort"].map(COHORT_MAP)
    df = df.dropna(subset=["cancer"])
    log.info("  in target cohorts (10 cancers): %d", len(df))

    df["chrom"] = "chr" + df["chrom"].astype(str)
    df["pos"] = pd.to_numeric(df["pos"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["pos"])
    df["pos"] = df["pos"].astype(int) - 1  # 1-based VCF -> 0-based panel
    df["strand"] = np.where(df["ref"] == "C", "+", "-")
    return df[["chrom", "pos", "ref", "alt", "strand", "cancer", "patient_id",
               "analysis_cohort"]].copy()


def annotate_panel_positions(panel: pd.DataFrame) -> pd.DataFrame:
    """Add is_TCW_C and is_CpG to panel positions (used for TCW-density and
    same-bases baselines)."""
    from pyfaidx import Fasta

    log.info("Annotating panel positions with is_TCW_C + is_CpG ...")
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
            seq = np.frombuffer(
                str(genome[ch][:]).upper().encode("ascii"), dtype=np.uint8
            )
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
        is_plus = ss_ok == "+"
        is_minus = ~is_plus
        is_cpg[valid_idx[is_plus]] = right[is_plus] == ord("G")
        is_cpg[valid_idx[is_minus]] = left[is_minus] == ord("C")
        right_AT = (right == ord("A")) | (right == ord("T"))
        left_AT = (left == ord("A")) | (left == ord("T"))
        is_tcw_c[valid_idx[is_plus]] = (left[is_plus] == ord("T")) & right_AT[is_plus]
        is_tcw_c[valid_idx[is_minus]] = (right[is_minus] == ord("A")) & left_AT[is_minus]

    panel = panel.copy()
    panel["is_cpg"] = is_cpg
    panel["is_TCW_C"] = is_tcw_c
    log.info(
        "  panel n=%d, is_TCW_C=%d (%.2f%%), is_CpG=%d (%.2f%%)",
        n, is_tcw_c.sum(), 100 * is_tcw_c.mean(),
        is_cpg.sum(), 100 * is_cpg.mean(),
    )
    return panel


def annotate_mut_context(maf: pd.DataFrame) -> pd.DataFrame:
    """Add is_TCW, is_CpG, is_TCW_nonCpG to a MAF-like df with chrom/pos/strand."""
    from pyfaidx import Fasta

    log.info("Annotating mutations with hg19 trinuc context ...")
    genome = Fasta(str(HG19), as_raw=False, sequence_always_upper=True)
    n = len(maf)
    is_tcw = np.zeros(n, dtype=bool)
    is_cpg = np.zeros(n, dtype=bool)
    sanity_ref_mismatches = 0

    chroms = maf["chrom"].to_numpy()
    poses = maf["pos"].astype(int).to_numpy()
    strands = maf["strand"].to_numpy()
    refs = maf["ref"].to_numpy()
    idx_all = np.arange(n)

    for ch in pd.Series(chroms).unique():
        mask = chroms == ch
        idx = idx_all[mask]
        if len(idx) == 0:
            continue
        try:
            seq = np.frombuffer(
                str(genome[ch][:]).upper().encode("ascii"), dtype=np.uint8
            )
        except Exception:
            continue
        L = len(seq)
        ps = poses[mask]
        ss = strands[mask]
        rs = refs[mask]
        ok = (ps >= 1) & (ps + 1 < L)
        valid_idx = idx[ok]
        ps_ok = ps[ok]
        ss_ok = ss[ok]
        rs_ok = rs[ok]
        left = seq[ps_ok - 1]
        center = seq[ps_ok]
        right = seq[ps_ok + 1]
        is_plus = ss_ok == "+"
        is_minus = ~is_plus
        # Sanity: center should match ref (C for +, G for -)
        ref_byte = np.where(is_plus, ord("C"), ord("G"))
        sanity_ref_mismatches += int((center != ref_byte).sum())
        plus_tcw = (
            is_plus
            & (left == ord("T"))
            & (center == ord("C"))
            & ((right == ord("A")) | (right == ord("T")))
        )
        minus_tcw = (
            is_minus
            & (right == ord("A"))
            & (center == ord("G"))
            & ((left == ord("A")) | (left == ord("T")))
        )
        plus_cpg = is_plus & (center == ord("C")) & (right == ord("G"))
        minus_cpg = is_minus & (center == ord("G")) & (left == ord("C"))
        is_tcw[valid_idx] = plus_tcw | minus_tcw
        is_cpg[valid_idx] = plus_cpg | minus_cpg

    out = maf.copy()
    out["is_TCW"] = is_tcw
    out["is_CpG"] = is_cpg
    out["is_TCW_nonCpG"] = is_tcw & ~is_cpg
    if sanity_ref_mismatches > 0:
        log.warning(
            "  %d/%d positions had ref-base mismatch with hg19",
            sanity_ref_mismatches, n,
        )
    log.info(
        "  total=%d  TCW=%d (%.1f%%)  CpG=%d (%.1f%%)  TCW_nonCpG=%d (%.1f%%)",
        n,
        out["is_TCW"].sum(), 100 * out["is_TCW"].mean(),
        out["is_CpG"].sum(), 100 * out["is_CpG"].mean(),
        out["is_TCW_nonCpG"].sum(), 100 * out["is_TCW_nonCpG"].mean(),
    )
    return out


# =========================================================================== #
# Per-cell computation (position-level)
# =========================================================================== #


def evaluate_cell(
    scores: np.ndarray,
    base_tcw: np.ndarray,
    base_npos: np.ndarray,
    mut_per_cancer: dict[str, np.ndarray],
    null_per_cancer: dict[str, np.ndarray],
    k: int,
    bonf_q: float,
) -> dict:
    n = len(scores)
    if k <= 0 or k >= n:
        return None

    nn_top = np.argpartition(-scores, k - 1)[:k]
    tcw_top = np.argpartition(-base_tcw, k - 1)[:k]
    npos_top = np.argpartition(-base_npos, k - 1)[:k]

    per_cancer = {}
    abs_recalls = []
    ratios_tcw, ratios_npos, ratios_random = [], [], []
    n_above_tcw = n_above_npos = 0
    p_perms = []

    rand_baseline = float(k) / n  # uniform-prior expected recall

    for cancer, mut in mut_per_cancer.items():
        total = int(mut.sum())
        if total == 0:
            per_cancer[cancer] = {
                "total_mut": 0,
                "abs_recall": float("nan"),
                "abs_recall_tcw": float("nan"),
                "abs_recall_npos": float("nan"),
                "ratio_tcw": float("nan"),
                "ratio_npos": float("nan"),
                "ratio_random": float("nan"),
                "p_perm": 1.0,
                "k": int(k),
                "n_units": int(n),
            }
            continue
        nn_recall = float(mut[nn_top].sum()) / total
        tcw_recall = float(mut[tcw_top].sum()) / total
        npos_recall = float(mut[npos_top].sum()) / total
        ratio_tcw = nn_recall / tcw_recall if tcw_recall > 0 else float("nan")
        ratio_npos = nn_recall / npos_recall if npos_recall > 0 else float("nan")
        ratio_random = nn_recall / rand_baseline if rand_baseline > 0 else float("nan")

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
            "ratio_random": ratio_random,
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
        if not np.isnan(ratio_random):
            ratios_random.append(ratio_random)
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
        return (
            float(a.mean()),
            float(np.percentile(boot, 2.5)),
            float(np.percentile(boot, 97.5)),
        )

    m_abs, lo_abs, hi_abs = boot_ci(abs_recalls, 1)
    m_tcw, lo_tcw, hi_tcw = boot_ci(ratios_tcw, 3)
    m_npos, lo_npos, hi_npos = boot_ci(ratios_npos, 4)
    m_rand, lo_rand, hi_rand = boot_ci(ratios_random, 5)

    n_bonf = int(sum(p < bonf_q for p in p_perms))

    return {
        "per_cancer": per_cancer,
        "n_cancers": len(per_cancer),
        "mean_abs_recall": m_abs,
        "abs_recall_ci_lo": lo_abs,
        "abs_recall_ci_hi": hi_abs,
        "mean_ratio_tcw": m_tcw,
        "ratio_tcw_ci_lo": lo_tcw,
        "ratio_tcw_ci_hi": hi_tcw,
        "mean_ratio_npos": m_npos,
        "ratio_npos_ci_lo": lo_npos,
        "ratio_npos_ci_hi": hi_npos,
        "mean_ratio_random": m_rand,
        "ratio_random_ci_lo": lo_rand,
        "ratio_random_ci_hi": hi_rand,
        "n_above_tcw": n_above_tcw,
        "n_above_npos": n_above_npos,
        "n_bonf_signif": n_bonf,
        "k": int(k),
        "n_units": int(n),
    }


# =========================================================================== #
# Permutation null worker (macOS-safe)
# =========================================================================== #


def _null_worker(args):
    n_units, k, mut, n_reps, seed = args
    rng = np.random.default_rng(seed)
    if int(mut.sum()) == 0 or k <= 0:
        return np.zeros(n_reps, dtype=np.int64)
    out = np.empty(n_reps, dtype=np.int64)
    for i in range(n_reps):
        idx = rng.choice(n_units, size=k, replace=False)
        out[i] = int(mut[idx].sum())
    return out


# =========================================================================== #
# Plot
# =========================================================================== #


def plot_recall_curve(rows, out_path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.DataFrame(rows)
    df = df[df["filter"] == "filter_TCW_nonCpG"]
    df = df.sort_values("cut_value")

    fig, ax = plt.subplots(1, 1, figsize=(7, 5), constrained_layout=True)
    ax.plot(
        df["cut_value"] * 100, df["mean_abs_recall"], "-o", color="tab:red",
        label="NN binary head", linewidth=2,
    )
    ax.fill_between(
        df["cut_value"] * 100,
        df["abs_recall_ci_lo"], df["abs_recall_ci_hi"],
        color="tab:red", alpha=0.15,
    )

    # Baseline references at each panel size
    rand_recalls = df["cut_value"].values
    ax.plot(
        df["cut_value"] * 100, rand_recalls, "--", color="grey",
        label="Random (k/n)",
    )

    # Implicit baseline recalls per cell (back out from ratios)
    df["abs_recall_tcw"] = df["mean_abs_recall"] / df["mean_ratio_vs_TCW"]
    df["abs_recall_npos"] = df["mean_abs_recall"] / df["mean_ratio_vs_NPOS"]
    ax.plot(
        df["cut_value"] * 100, df["abs_recall_tcw"], "-s",
        color="tab:blue", label="TCW-density baseline (same-bases)",
    )
    ax.plot(
        df["cut_value"] * 100, df["abs_recall_npos"], "-^",
        color="tab:green", label="n_pos / gene-density baseline",
    )

    ax.set_xlabel("Panel size (% of CDS-C positions)")
    ax.set_ylabel("Mean per-cancer recall (TCW_nonCpG mutations)")
    ax.set_title("POG570 v4_cds binary-head recall vs baselines")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# =========================================================================== #
# Per-cohort breakdown of POG570 (optional, for the markdown report)
# =========================================================================== #


def per_pog_cohort_counts(maf: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    g = (
        maf.groupby("analysis_cohort")
        .agg(
            n_total=("ref", "size"),
            n_tcw_noncpg=("is_TCW_nonCpG", "sum"),
        )
        .sort_values("n_total", ascending=False)
        .head(top_n)
    )
    return g.reset_index()


# =========================================================================== #
# Main
# =========================================================================== #


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", type=str, default=str(DEFAULT_PANEL))
    ap.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--head", type=str, default="score_binary",
                    help="Head column in the panel parquet to evaluate")
    ap.add_argument("--n-workers", type=int, default=8)
    ap.add_argument("--perm-reps", type=int, default=PERM_REPS)
    args = ap.parse_args()

    panel_path = Path(args.panel)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    head = args.head
    n_workers = max(1, min(8, args.n_workers))
    perm_reps = args.perm_reps

    log.info("=== Analysis D v4 — POG570 validation on v4_cds panel ===")
    log.info("Panel: %s", panel_path)
    log.info("Out:   %s", out_dir)
    log.info("Head:  %s", head)
    log.info("Top pcts: %s; perm_reps=%d", TOP_PCTS, perm_reps)

    t0 = time.time()

    # 1. Load panel + annotate.
    panel = load_panel(panel_path, head_cols=[head])
    if head not in panel.columns:
        raise ValueError(f"Head '{head}' not found in panel columns")
    panel = annotate_panel_positions(panel)
    n_units = len(panel)
    log.info("Panel positions (n_units): %d", n_units)

    # 2. Load POG570 + restrict to in-panel positions.
    muts = load_pog570(POG570_PATH)
    log.info("Restricting POG570 mutations to in-panel positions ...")
    panel_set = set(
        zip(
            panel["chrom"].astype(str).values,
            panel["pos"].astype(int).values,
        )
    )
    in_panel = np.array(
        [(c, int(p)) in panel_set for c, p in zip(muts["chrom"], muts["pos"])]
    )
    muts_inpanel = muts.iloc[np.where(in_panel)[0]].reset_index(drop=True)
    log.info(
        "  in-panel POG570 C>T/G>A: %d / %d (%.2f%%)",
        len(muts_inpanel), len(muts), 100 * len(muts_inpanel) / max(1, len(muts)),
    )
    if len(muts_inpanel) == 0:
        log.error("No in-panel POG570 mutations. Aborting.")
        return 1

    # 3. Annotate trinuc context on POG570.
    muts_inpanel = annotate_mut_context(muts_inpanel)

    # Filter sets (TCW_nonCpG, all_CT)
    filter_sets = {
        "filter_TCW_nonCpG": muts_inpanel[muts_inpanel["is_TCW_nonCpG"]].copy(),
        "filter_all_CT": muts_inpanel.copy(),
    }
    for fn, fdf in filter_sets.items():
        log.info("  %s: %d in-panel mutations", fn, len(fdf))

    # 4. Map mutations to panel index per cancer.
    panel_lookup = pd.DataFrame(
        {
            "chrom": panel["chrom"].astype(str).values,
            "pos": panel["pos"].astype(int).values,
            "_uidx": np.arange(n_units),
        }
    )

    # We want per-cancer mut arrays per filter (n_units int32 vector).
    mut_per_filter_cancer: dict[tuple[str, str], np.ndarray] = {}
    for filter_name, fdf in filter_sets.items():
        m = fdf[["chrom", "pos", "cancer"]].copy()
        m["pos"] = m["pos"].astype(int)
        m = m.merge(panel_lookup, on=["chrom", "pos"], how="inner")
        for cancer in TARGET_CANCERS:
            sub = m[m["cancer"] == cancer]
            arr = np.zeros(n_units, dtype=np.int32)
            if len(sub) > 0:
                counts = sub["_uidx"].value_counts()
                arr[counts.index.astype(int).to_numpy()] = (
                    counts.values.astype(np.int32)
                )
            mut_per_filter_cancer[(filter_name, cancer)] = arr

    # 5. Build score / baseline arrays.
    scores = panel[head].to_numpy(dtype=np.float64)
    base_tcw = panel["is_TCW_C"].to_numpy(dtype=np.float64)
    base_npos = np.ones(n_units, dtype=np.float64)  # at position-level, 1 per unit

    # Determine k for each top-pct.
    cuts = []
    for tp in TOP_PCTS:
        k = max(1, int(round(n_units * tp)))
        cuts.append({"top_pct": tp, "k": k})
    log.info("Cuts: %s", [(c["top_pct"], c["k"]) for c in cuts])

    # 6. Build null distributions (per filter × cancer × k).
    log.info("Sampling null distributions ...")
    null_specs = []
    null_keys = []
    for c in cuts:
        k = c["k"]
        for filter_name in filter_sets.keys():
            for cancer in TARGET_CANCERS:
                mut = mut_per_filter_cancer[(filter_name, cancer)]
                if int(mut.sum()) == 0:
                    continue
                seed = SEED_BASE + abs(hash((filter_name, cancer, k))) % 100_000
                null_specs.append((n_units, k, mut, perm_reps, seed))
                null_keys.append((filter_name, cancer, k))

    nulls: dict[tuple[str, str, int], np.ndarray] = {}
    t_null = time.time()
    log.info("  %d null jobs (perm_reps=%d) ...", len(null_specs), perm_reps)
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = {
            ex.submit(_null_worker, spec): key
            for spec, key in zip(null_specs, null_keys)
        }
        n_done = 0
        for fut in as_completed(futs):
            arr = fut.result()
            nulls[futs[fut]] = arr
            n_done += 1
            if n_done % 10 == 0:
                log.info("  null %d/%d (%.1fs)", n_done, len(null_specs),
                         time.time() - t_null)
    log.info("  null sampling done in %.1fs", time.time() - t_null)

    # 7. Evaluate cells.
    bonf_n_tests = len(cuts) * len(filter_sets) * len(TARGET_CANCERS)
    bonf_q = ALPHA / bonf_n_tests
    log.info("Bonferroni q=%.2e (n_tests=%d)", bonf_q, bonf_n_tests)

    rows = []
    per_cancer_rows = []
    for c in cuts:
        k = c["k"]
        for filter_name in filter_sets.keys():
            mut_per_cancer = {}
            null_per_cancer = {}
            for cancer in TARGET_CANCERS:
                arr = mut_per_filter_cancer[(filter_name, cancer)]
                if int(arr.sum()) == 0:
                    continue
                mut_per_cancer[cancer] = arr
                null_per_cancer[cancer] = nulls.get(
                    (filter_name, cancer, k), np.zeros(1, dtype=np.int64)
                )
            res = evaluate_cell(
                scores, base_tcw, base_npos,
                mut_per_cancer, null_per_cancer,
                k, bonf_q,
            )
            if res is None:
                continue
            rows.append({
                "level": "position",
                "head": head,
                "cut_type": "top_pct",
                "cut_value": c["top_pct"],
                "filter": filter_name,
                "panel_units": k,
                "n_units_total": n_units,
                "n_cancers": res["n_cancers"],
                "mean_abs_recall": res["mean_abs_recall"],
                "abs_recall_ci_lo": res["abs_recall_ci_lo"],
                "abs_recall_ci_hi": res["abs_recall_ci_hi"],
                "mean_ratio_vs_TCW": res["mean_ratio_tcw"],
                "ratio_tcw_ci_lo": res["ratio_tcw_ci_lo"],
                "ratio_tcw_ci_hi": res["ratio_tcw_ci_hi"],
                "mean_ratio_vs_NPOS": res["mean_ratio_npos"],
                "ratio_npos_ci_lo": res["ratio_npos_ci_lo"],
                "ratio_npos_ci_hi": res["ratio_npos_ci_hi"],
                "mean_ratio_vs_RANDOM": res["mean_ratio_random"],
                "ratio_random_ci_lo": res["ratio_random_ci_lo"],
                "ratio_random_ci_hi": res["ratio_random_ci_hi"],
                "n_cancers_above_TCW": res["n_above_tcw"],
                "n_cancers_above_NPOS": res["n_above_npos"],
                "n_cancers_bonf_signif": res["n_bonf_signif"],
            })
            for cancer, pc in res["per_cancer"].items():
                per_cancer_rows.append({
                    "head": head,
                    "filter": filter_name,
                    "cut_value": c["top_pct"],
                    "cancer": cancer,
                    **pc,
                })

    df_out = pd.DataFrame(rows).sort_values(
        ["filter", "cut_value"]
    ).reset_index(drop=True)
    csv_path = out_dir / "enrichment_v4_cds.csv"
    df_out.to_csv(csv_path, index=False)
    log.info("Wrote %s (%d rows)", csv_path, len(df_out))

    df_pc = pd.DataFrame(per_cancer_rows)
    pc_path = out_dir / "enrichment_v4_cds_per_cancer.csv"
    df_pc.to_csv(pc_path, index=False)
    log.info("Wrote %s (%d rows)", pc_path, len(df_pc))

    # 8. Plot recall curve.
    plot_path = out_dir / "recall_curve_pog570_v4.png"
    plot_recall_curve(rows, plot_path)
    log.info("Wrote %s", plot_path)

    # 9. POG570 per-cohort counts (top 10 by total).
    pog_breakdown = per_pog_cohort_counts(muts_inpanel, top_n=10)

    # 10. Pull PCAWG comparison numbers from the v4 sweep CSV.
    pcawg_csv = (
        ROOT
        / "experiments/multi_enzyme/outputs/pcawg_tcw_panel/v4_outputs"
        / "topx_threshold_sweep_v4_cds.csv"
    )
    pcawg_ref = None
    if pcawg_csv.exists():
        pdf = pd.read_csv(pcawg_csv)
        pdf = pdf[
            (pdf["head"] == head)
            & (pdf["cut_type"] == "top_pct")
            & (pdf["level"] == "position")
        ]
        pcawg_ref = pdf

    # 11. Write markdown report.
    report_path = out_dir / "POG570_V4_RESULTS.md"

    def _fmt(x, prec=3):
        try:
            return f"{x:.{prec}f}"
        except Exception:
            return str(x)

    def _row(df, filt, tp):
        sub = df[(df["filter"] == filt) & (df["cut_value"].round(3) == round(tp, 3))]
        if len(sub) == 0:
            return None
        return sub.iloc[0]

    with open(report_path, "w") as f:
        f.write("# POG570 v4 validation — v4_cds binary head\n\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(f"- Panel: `{panel_path}`\n")
        f.write(f"- Panel positions (n_units): {n_units:,}\n")
        f.write(f"- Head: `{head}`\n")
        f.write(f"- POG570 source: `{POG570_PATH}`\n")
        f.write(f"- Random seed: {SEED_BASE}\n\n")

        f.write("## POG570 mutation counts (after filtering)\n\n")
        n_raw = len(muts)
        n_inp = len(muts_inpanel)
        n_tcw = int(muts_inpanel["is_TCW_nonCpG"].sum())
        n_ct = len(muts_inpanel)  # all_CT inside panel = all C>T/G>A SNVs in-panel
        f.write(f"- Raw POG570 C>T/G>A SNVs (10 cohorts mapping to PCAWG cancers): {n_raw:,}\n")
        f.write(f"- In-panel (CDS-C, v4_cds): {n_inp:,} ({100*n_inp/max(1,n_raw):.2f}%)\n")
        f.write(f"- In-panel TCW_nonCpG: {n_tcw:,}\n")
        f.write(f"- In-panel all_CT: {n_ct:,}\n\n")

        f.write("## Headline numbers (position-level binary head)\n\n")
        f.write(
            "| filter | top% | NN recall (CI) | ratio_vs_TCW (CI) | "
            "ratio_vs_NPOS (CI) | ratio_vs_random (CI) | bonf_sig/n |\n"
        )
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for filt in ["filter_TCW_nonCpG", "filter_all_CT"]:
            for tp in TOP_PCTS:
                r = _row(df_out, filt, tp)
                if r is None:
                    continue
                f.write(
                    f"| {filt} | {tp:.2f} | "
                    f"{_fmt(r['mean_abs_recall'])} "
                    f"({_fmt(r['abs_recall_ci_lo'])}–{_fmt(r['abs_recall_ci_hi'])}) | "
                    f"{_fmt(r['mean_ratio_vs_TCW'], 2)}× "
                    f"({_fmt(r['ratio_tcw_ci_lo'], 2)}–"
                    f"{_fmt(r['ratio_tcw_ci_hi'], 2)}) | "
                    f"{_fmt(r['mean_ratio_vs_NPOS'], 2)}× "
                    f"({_fmt(r['ratio_npos_ci_lo'], 2)}–"
                    f"{_fmt(r['ratio_npos_ci_hi'], 2)}) | "
                    f"{_fmt(r['mean_ratio_vs_RANDOM'], 2)}× "
                    f"({_fmt(r['ratio_random_ci_lo'], 2)}–"
                    f"{_fmt(r['ratio_random_ci_hi'], 2)}) | "
                    f"{int(r['n_cancers_bonf_signif'])}/{int(r['n_cancers'])} |\n"
                )

        f.write("\n## PCAWG vs POG570 (10-cancer aggregate, top-1%, position-level)\n\n")
        f.write(
            "| metric | PCAWG (v4_cds) | POG570 (v4_cds) | replicates? |\n"
        )
        f.write("|---|---:|---:|:---:|\n")
        # all_CT, top-1%, ratio_vs_TCW
        pog_act = _row(df_out, "filter_all_CT", 0.01)
        pog_tcw = _row(df_out, "filter_TCW_nonCpG", 0.01)
        if pcawg_ref is not None:
            p_act = pcawg_ref[
                (pcawg_ref["filter"] == "filter_all_CT")
                & (pcawg_ref["cut_value"].round(3) == 0.010)
            ]
            p_tcw = pcawg_ref[
                (pcawg_ref["filter"] == "filter_TCW_nonCpG")
                & (pcawg_ref["cut_value"].round(3) == 0.010)
            ]
            if len(p_act) and pog_act is not None:
                pcawg_v = float(p_act.iloc[0]["mean_ratio_vs_TCW"])
                pog_v = float(pog_act["mean_ratio_vs_TCW"])
                rep = "yes" if pog_v >= 1.5 and pog_v >= 0.5 * pcawg_v else "partial" if pog_v >= 1.0 else "no"
                f.write(
                    f"| ratio_vs_TCW (all_CT, top-1%) | {pcawg_v:.2f}× | "
                    f"{pog_v:.2f}× ({float(pog_act['ratio_tcw_ci_lo']):.2f}–"
                    f"{float(pog_act['ratio_tcw_ci_hi']):.2f}) | {rep} |\n"
                )
            if len(p_tcw) and pog_tcw is not None:
                pcawg_v = float(p_tcw.iloc[0]["mean_ratio_vs_NPOS"])
                pog_v = float(pog_tcw["mean_ratio_vs_NPOS"])
                rep = "yes" if pog_v >= 1.5 and pog_v >= 0.5 * pcawg_v else "partial" if pog_v >= 1.0 else "no"
                f.write(
                    f"| ratio_vs_NPOS (TCW_nonCpG, top-1%) | {pcawg_v:.2f}× | "
                    f"{pog_v:.2f}× ({float(pog_tcw['ratio_npos_ci_lo']):.2f}–"
                    f"{float(pog_tcw['ratio_npos_ci_hi']):.2f}) | {rep} |\n"
                )
                # also abs recall
                pcawg_a = float(p_tcw.iloc[0]["mean_abs_recall"])
                pog_a = float(pog_tcw["mean_abs_recall"])
                f.write(
                    f"| abs recall (TCW_nonCpG, top-1%) | {pcawg_a:.4f} | "
                    f"{pog_a:.4f} ({float(pog_tcw['abs_recall_ci_lo']):.4f}–"
                    f"{float(pog_tcw['abs_recall_ci_hi']):.4f}) | "
                    f"{'yes' if pog_a >= 0.5 * pcawg_a else 'no'} |\n"
                )

        # POG570 per-cohort breakdown
        f.write("\n## POG570 per-cohort breakdown (top 10 cohorts by mutation count)\n\n")
        f.write("| analysis_cohort | mapped_cancer | n_total_in_panel | n_TCW_nonCpG_in_panel |\n")
        f.write("|---|---|---:|---:|\n")
        for _, row in pog_breakdown.iterrows():
            mapped = COHORT_MAP.get(row["analysis_cohort"], "—")
            f.write(
                f"| {row['analysis_cohort']} | {mapped} | "
                f"{int(row['n_total']):,} | {int(row['n_tcw_noncpg']):,} |\n"
            )

        # Per-cancer ratio breakdown at top-1%, TCW_nonCpG
        f.write("\n## Per-cancer ratios at top-1%, TCW_nonCpG (binary head)\n\n")
        f.write(
            "| cancer | n_mut | NN_recall | TCW_recall | NPOS_recall | "
            "ratio_TCW | ratio_NPOS | p_perm |\n"
        )
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        sub = df_pc[
            (df_pc["filter"] == "filter_TCW_nonCpG")
            & (df_pc["cut_value"].round(3) == 0.010)
        ].sort_values("ratio_npos", ascending=False)
        for _, r in sub.iterrows():
            f.write(
                f"| {r['cancer']} | {int(r['total_mut'])} | "
                f"{_fmt(r['abs_recall'], 4)} | "
                f"{_fmt(r['abs_recall_tcw'], 4)} | "
                f"{_fmt(r['abs_recall_npos'], 4)} | "
                f"{_fmt(r['ratio_tcw'], 2)} | "
                f"{_fmt(r['ratio_npos'], 2)} | "
                f"{_fmt(r['p_perm'], 4)} |\n"
            )

        # Verdict
        f.write("\n## Verdict\n\n")
        if pog_tcw is not None and pog_act is not None:
            v_tcw = float(pog_tcw["mean_ratio_vs_NPOS"])
            v_act = float(pog_act["mean_ratio_vs_TCW"])
            v_tcw_lo = float(pog_tcw["ratio_npos_ci_lo"])
            v_act_lo = float(pog_act["ratio_tcw_ci_lo"])

            replicate_strong = v_tcw_lo > 1.0 and v_act_lo > 1.0 and v_tcw >= 1.5 and v_act >= 1.5
            replicate_partial = v_tcw_lo > 1.0 or v_act_lo > 1.0
            verdict = (
                "**REPLICATES**" if replicate_strong
                else "**PARTIAL REPLICATION**" if replicate_partial
                else "**FAILS TO REPLICATE**"
            )
            f.write(
                f"v4_cds binary head on POG570 (independent cohort):\n\n"
                f"- ratio_vs_NPOS at top-1% (TCW_nonCpG) = "
                f"{v_tcw:.2f}× (95% CI {v_tcw_lo:.2f}–"
                f"{float(pog_tcw['ratio_npos_ci_hi']):.2f}); CI lower bound "
                f"{'>' if v_tcw_lo > 1 else '≤'} 1.0\n"
                f"- ratio_vs_TCW at top-1% (all_CT) = "
                f"{v_act:.2f}× (95% CI {v_act_lo:.2f}–"
                f"{float(pog_act['ratio_tcw_ci_hi']):.2f}); CI lower bound "
                f"{'>' if v_act_lo > 1 else '≤'} 1.0\n\n"
                f"Verdict: {verdict}\n"
            )

        f.write("\n## Notes on baseline construction (v4 vs v3)\n\n")
        f.write(
            "- v3 POG570 was confounded by a baseline mismatch: PCAWG analysis\n"
            "  used `seq.count('CG')` over the literal hg19 window sequence,\n"
            "  whereas POG570 v1 used `sum(is_cpg)` over panel positions.\n"
            "  Different windows ranked highest by these two definitions.\n"
            "- v4 uses **same-bases baselines exclusively**: TCW-density and\n"
            "  n_pos counted *only over CDS-C panel positions*, matching the\n"
            "  v4 PCAWG fair sweep so PCAWG and POG570 are on the same footing.\n"
            "- Position-level (not 250 bp window) is the operational unit, \n"
            "  removing the gene-density artefact (windows that span large\n"
            "  CDS regions had inflated n_pos under any window-aggregator).\n"
        )

    log.info("Wrote %s", report_path)
    log.info("DONE in %.1fs", time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
