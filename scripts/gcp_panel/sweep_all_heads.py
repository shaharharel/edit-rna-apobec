#!/usr/bin/env python3
"""All-heads aggregator/window sweep.

Compares all 7 score heads (binary + 6 enzyme-specific) at the top three
panel constructions identified in `aggregator_window_sweep.csv`:

  * sum × 1000 bp  (binary winner: 4.09% recall, ratio_vs_TCW=1.31)
  * sum × 500 bp   (2.28% recall, 9/10 BH)
  * top3_mean × 500 bp (1.97% recall, 9/10 BH)

For each (head, construction) combination, computes per-cancer recall under
the TCW-non-CpG stratum, ratios vs TCW-motif-density and CpG-density, and
permutation null + BH-FDR across 10 cancers.

Also tests:
  * Top-1% panel disjointness between binary and apobec1 (Jaccard).
  * Combined "binary OR apobec1" panel built as top-1% by max(score_binary,
    score_apobec1).

Outputs:
  experiments/multi_enzyme/outputs/pcawg_tcw_panel/all_heads_sweep.csv
  experiments/multi_enzyme/outputs/pcawg_tcw_panel/all_heads_sweep.png
  experiments/multi_enzyme/outputs/pcawg_tcw_panel/ALL_HEADS_RESULTS.md

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
from scipy.stats import false_discovery_control, hypergeom

ROOT = Path("/Users/shaharharel/Documents/github/edit-rna-apobec")
PANEL_PATH = ROOT / "experiments/multi_enzyme/outputs/pcawg_tcw_panel/panel_scores_cds.parquet"
TCGA_DIR = ROOT / "data/raw/tcga"
PCAWG_DIR = ROOT / "data/raw/pcawg/by_cancer"
HG19 = ROOT / "data/raw/genomes/hg19.fa"
OUT_DIR = ROOT / "experiments/multi_enzyme/outputs/pcawg_tcw_panel"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CANCERS = ["blca", "brca", "cesc", "coadread", "esca", "hnsc", "lihc", "lusc",
           "skcm", "stad"]
HEADS = ["score_binary", "score_A3A", "score_A3B", "score_A3G",
         "score_A3A_A3G", "score_Neither", "score_apobec1"]
# (aggregator, window_size_bp) constructions to test
CONSTRUCTIONS = [
    ("sum",       1000),
    ("sum",        500),
    ("top3_mean",  500),
]
TOP_PCT = 0.01
PERM_REPS = 10000
N_BOOT = 10000
SEED_BASE = 20260427

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    stream=sys.stdout)
log = logging.getLogger(__name__)


# =========================================================================== #
# MAF loading (mirrors compute_panel_recall_sweep.py)
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
    df["pos"] = df["Start_Position"].astype(int) - 1  # 1-based MAF -> 0-based panel
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
    log.info("  combined C>T/G>A: %d, %d cancers", len(combined),
             combined["cancer"].nunique())
    return combined


# =========================================================================== #
# Panel annotation (is_TCW_C, is_cpg) — used both to filter MAF and for TCW
# baseline density at the per-window level.
# =========================================================================== #

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
            log.warning("  couldn't load %s", ch)
            continue
        L = len(seq)

        ps = poses[mask]; ss = strands[mask]
        ok = (ps >= 1) & (ps + 1 < L)
        valid_idx = idx[ok]
        ps_ok = ps[ok]; ss_ok = ss[ok]
        left = seq[ps_ok - 1]; right = seq[ps_ok + 1]
        is_plus = (ss_ok == "+"); is_minus = ~is_plus
        is_cpg[valid_idx[is_plus]] = (right[is_plus] == ord("G"))
        is_cpg[valid_idx[is_minus]] = (left[is_minus] == ord("C"))
        right_AT = (right == ord("A")) | (right == ord("T"))
        left_AT = (left == ord("A")) | (left == ord("T"))
        is_tcw_c[valid_idx[is_plus]] = (left[is_plus] == ord("T")) & right_AT[is_plus]
        is_tcw_c[valid_idx[is_minus]] = (right[is_minus] == ord("A")) & left_AT[is_minus]

    panel = panel.copy()
    panel["is_cpg"] = is_cpg
    panel["is_TCW_C"] = is_tcw_c
    log.info("  is_cpg: %.1f%%; is_TCW_C: %.1f%%",
             100*is_cpg.mean(), 100*is_tcw_c.mean())
    return panel


def filter_tcw_non_cpg_muts(maf: pd.DataFrame) -> pd.DataFrame:
    from pyfaidx import Fasta
    log.info("Filtering MAF to TCW-non-CpG C>T mutations ...")
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
            seq = np.frombuffer(str(genome[ch][:]).upper().encode("ascii"),
                                dtype=np.uint8)
        except Exception:
            continue
        L = len(seq)
        ps = poses[mask]; ss = strands[mask]
        ok = (ps >= 1) & (ps + 1 < L)
        valid_idx = idx[ok]
        ps_ok = ps[ok]; ss_ok = ss[ok]
        left = seq[ps_ok - 1]; center = seq[ps_ok]; right = seq[ps_ok + 1]
        is_plus = (ss_ok == "+"); is_minus = ~is_plus
        plus_pass = is_plus & (left == ord("T")) & (center == ord("C")) & \
                    ((right == ord("A")) | (right == ord("T")))
        minus_pass = is_minus & (right == ord("A")) & (center == ord("G")) & \
                     ((left == ord("A")) | (left == ord("T")))
        tcw_pass = plus_pass | minus_pass
        keep[valid_idx[tcw_pass]] = True

    out = maf[keep].copy()
    log.info("  TCW-non-CpG kept: %d / %d (%.1f%%)", keep.sum(), len(maf),
             100*keep.mean())
    return out


# =========================================================================== #
# Window aggregation for an arbitrary score head
# =========================================================================== #

def build_window_aggregations(panel: pd.DataFrame, window_size: int,
                              head_cols: list[str]) -> pd.DataFrame:
    """Build a window-level table containing, for each head in head_cols, the
    `sum` and `top3_mean` aggregations, plus CpG- and TCW-density baselines."""
    from pyfaidx import Fasta
    log.info("[w=%d] Building window aggregations (%d heads) ...",
             window_size, len(head_cols))
    p = panel.copy()
    p["pos"] = p["pos"].astype(int)
    p["win_start"] = (p["pos"] // window_size) * window_size

    grp = p.groupby(["chrom", "win_start"])
    agg_cols = {"n_pos": ("pos", "size")}
    for h in head_cols:
        agg_cols[f"{h}__sum"] = (h, "sum")
    out = grp.agg(**agg_cols).reset_index()

    # top3_mean is not a built-in groupby agg — compute via manual loop only for
    # the specific window+head combos we need. Cheaper than apply.
    log.info("[w=%d] Computing top3_mean per head (vectorized via sort) ...",
             window_size)
    # Sort within each (chrom, win_start), then take last 3 of each group
    p_sorted = p.sort_values(["chrom", "win_start"], kind="stable")
    grp_sorted = p_sorted.groupby(["chrom", "win_start"], sort=False)
    for h in head_cols:
        # top3_mean: mean of top-3 (or all if <3) values of head h in each window
        def _top3(s: pd.Series) -> float:
            v = s.values
            if len(v) <= 3:
                return float(v.mean()) if len(v) else 0.0
            # partial sort
            return float(np.mean(np.partition(v, -3)[-3:]))
        out_top3 = grp_sorted[h].apply(_top3).rename(f"{h}__top3_mean")
        out = out.merge(out_top3.reset_index(), on=["chrom", "win_start"])

    # Phase 1.5 baselines: CpG and TCW density from hg19 (vectorized)
    log.info("[w=%d] Computing CpG + TCW-motif densities ...", window_size)
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
            seq = np.frombuffer(str(genome[ch][:]).upper().encode("ascii"),
                                dtype=np.uint8)
        except Exception:
            continue
        L = len(seq)
        is_cg = ((seq[:-1] == ord("C")) & (seq[1:] == ord("G"))).astype(np.int32)
        cum_cg = np.zeros(L, dtype=np.int32)
        cum_cg[1:] = np.cumsum(is_cg)
        if L >= 3:
            t0 = seq[:-2]; t1 = seq[1:-1]; t2 = seq[2:]
            is_tcw_plus = (t0 == ord("T")) & (t1 == ord("C")) & \
                          ((t2 == ord("A")) | (t2 == ord("T")))
            is_tcw_minus = ((t0 == ord("A")) | (t0 == ord("T"))) & \
                           (t1 == ord("G")) & (t2 == ord("A"))
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
    log.info("[w=%d] %d windows", window_size, len(out))
    return out


# =========================================================================== #
# Per-construction evaluator (one head × one (aggregator, window))
# =========================================================================== #

def evaluate_construction(scores: np.ndarray, baseline_cpg: np.ndarray,
                          baseline_tcw: np.ndarray,
                          mut_per_cancer: dict[str, np.ndarray],
                          top_pct: float, perm_reps: int,
                          n_boot: int, seed: int) -> dict:
    n = len(scores)
    k = max(1, int(round(n * top_pct)))

    nn_top = np.argpartition(-scores, k - 1)[:k]
    cpg_top = np.argpartition(-baseline_cpg, k - 1)[:k]
    tcw_top = np.argpartition(-baseline_tcw, k - 1)[:k]

    per_cancer = {}
    abs_recalls = []
    ratios_cpg = []
    ratios_tcw = []
    p_perms = []
    n_above_1_cpg = 0
    n_above_1_tcw = 0

    for cancer, mut in mut_per_cancer.items():
        total = int(mut.sum())
        if total == 0:
            per_cancer[cancer] = {
                "total_mut": 0, "abs_recall": float("nan"),
                "ratio_cpg": float("nan"), "ratio_tcw": float("nan"),
                "p_perm": 1.0,
            }
            continue
        nn_recall = float(mut[nn_top].sum()) / total
        cpg_recall = float(mut[cpg_top].sum()) / total
        tcw_recall = float(mut[tcw_top].sum()) / total

        ratio_cpg = nn_recall / cpg_recall if cpg_recall > 0 else float("nan")
        ratio_tcw = nn_recall / tcw_recall if tcw_recall > 0 else float("nan")

        # Hypergeometric tail = exact 1-sided permutation null
        mut_in_top_obs = int(mut[nn_top].sum())
        p_perm = float(hypergeom.sf(mut_in_top_obs - 1, n, total, k))
        p_perm = max(p_perm, 1.0 / (perm_reps + 1))

        per_cancer[cancer] = {
            "total_mut": total,
            "abs_recall": nn_recall,
            "ratio_cpg": ratio_cpg,
            "ratio_tcw": ratio_tcw,
            "p_perm": p_perm,
            "k": k,
            "n_units": n,
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
        p_perms.append(p_perm)

    # BH-FDR
    if p_perms:
        q_vals = false_discovery_control(p_perms, method="bh")
        n_bh = int(sum(q < 0.025 for q in q_vals))
        for (cancer, _), q in zip(list(per_cancer.items()), q_vals):
            per_cancer[cancer]["q_bh"] = float(q)
            per_cancer[cancer]["reject_bh"] = bool(q < 0.025)
    else:
        n_bh = 0

    def boot_ci(arr, seed_):
        if not arr:
            return float("nan"), float("nan"), float("nan")
        a = np.array(arr)
        rng = np.random.default_rng(seed_)
        boot = np.array([rng.choice(a, size=len(a), replace=True).mean()
                         for _ in range(n_boot)])
        return float(a.mean()), float(np.percentile(boot, 2.5)), \
               float(np.percentile(boot, 97.5))

    mean_abs, ci_abs_lo, ci_abs_hi = boot_ci(abs_recalls, seed + 1)
    mean_cpg, ci_cpg_lo, ci_cpg_hi = boot_ci(ratios_cpg, seed + 2)
    mean_tcw, ci_tcw_lo, ci_tcw_hi = boot_ci(ratios_tcw, seed + 3)

    return {
        "per_cancer": per_cancer,
        "n_cancers": len(per_cancer),
        "mean_abs_recall": mean_abs,
        "ci_abs_lo": ci_abs_lo,
        "ci_abs_hi": ci_abs_hi,
        "mean_ratio_cpg": mean_cpg,
        "ratio_cpg_ci_lo": ci_cpg_lo,
        "ratio_cpg_ci_hi": ci_cpg_hi,
        "mean_ratio_tcw": mean_tcw,
        "ratio_tcw_ci_lo": ci_tcw_lo,
        "ratio_tcw_ci_hi": ci_tcw_hi,
        "n_cancers_above_TCW": n_above_1_tcw,
        "n_cancers_above_CpG": n_above_1_cpg,
        "n_bh_signif": n_bh,
        "k": k,
        "n_units": n,
    }


# =========================================================================== #
# Worker: load windows + evaluate one (head, aggregator, window) combo
# =========================================================================== #

def _process_one(args):
    (windows_path, mut_path, head, aggregator, window_size,
     top_pct, perm_reps, n_boot, seed) = args
    w = pd.read_parquet(windows_path)
    mut_pc = json.loads(Path(mut_path).read_text())
    mut_per_cancer = {}
    for cancer, vals in mut_pc.items():
        arr = np.zeros(len(w), dtype=np.int32)
        for idx, count in vals:
            arr[int(idx)] = int(count)
        mut_per_cancer[cancer] = arr

    score_col = f"{head}__{aggregator}"
    scores = w[score_col].to_numpy(dtype=np.float64)
    cpg = w["cpg_density"].to_numpy(dtype=np.float64)
    tcw = w["tcw_density"].to_numpy(dtype=np.float64)

    res = evaluate_construction(scores, cpg, tcw, mut_per_cancer,
                                top_pct, perm_reps, n_boot, seed)
    res["head"] = head
    res["aggregator"] = aggregator
    res["window_size_bp"] = window_size
    return res


# =========================================================================== #
# Main
# =========================================================================== #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-workers", type=int, default=8)
    ap.add_argument("--quick", action="store_true",
                    help="1K perms, fewer bootstraps for testing")
    args = ap.parse_args()
    n_workers = min(8, args.n_workers)
    perm_reps = 1000 if args.quick else PERM_REPS
    n_boot = 1000 if args.quick else N_BOOT

    log.info("=== All-heads sweep ===")
    log.info("Heads: %s", HEADS)
    log.info("Constructions: %s", CONSTRUCTIONS)
    log.info("perm_reps=%d, n_boot=%d, n_workers=%d", perm_reps, n_boot,
             n_workers)

    # 1. Load + annotate panel
    log.info("Loading panel ...")
    panel = pd.read_parquet(PANEL_PATH)
    log.info("  panel rows: %d", len(panel))
    panel = annotate_panel_positions(panel)

    # 2. Add a synthetic combined head: max(score_binary, score_apobec1)
    panel["score_binary_OR_apobec1"] = np.maximum(panel["score_binary"].values,
                                                  panel["score_apobec1"].values)

    # 3. Load + filter MAFs
    maf = load_combined_coding_maf()
    log.info("Restricting MAF to in-panel positions ...")
    panel_set = set(zip(panel["chrom"].astype(str).values,
                        panel["pos"].astype(int).values))
    in_panel = np.array([(c, int(p)) in panel_set
                         for c, p in zip(maf["chrom"], maf["pos"])])
    maf = maf.iloc[np.where(in_panel)[0]].reset_index(drop=True)
    log.info("  in-panel: %d", len(maf))
    maf = filter_tcw_non_cpg_muts(maf)
    log.info("  TCW-non-CpG in-panel: %d", len(maf))

    # 4. For each unique window size, build the window-level aggregation table
    #    containing all heads × needed aggregators.
    window_sizes_needed = sorted({ws for _, ws in CONSTRUCTIONS})
    aggregators_needed = sorted({ag for ag, _ in CONSTRUCTIONS})
    all_heads = HEADS + ["score_binary_OR_apobec1"]

    work_units = []  # (windows_path, mut_path, head, agg, ws, ...)
    for ws in window_sizes_needed:
        windows = build_window_aggregations(panel, ws, all_heads)
        win_path = OUT_DIR / f"_allheads_windows_w{ws}.parquet"
        windows.to_parquet(win_path, index=False)

        # Map mutations to windows
        log.info("[w=%d] Mapping mutations to windows ...", ws)
        m = maf.copy()
        m["win_start"] = (m["pos"].astype(int) // ws) * ws
        win_lookup = windows.reset_index(drop=True).reset_index().rename(
            columns={"index": "win_idx"})
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
        mut_path = OUT_DIR / f"_allheads_mut_w{ws}.json"
        mut_path.write_text(json.dumps(win_per_cancer))

        for ag, ws2 in CONSTRUCTIONS:
            if ws2 != ws:
                continue
            for head in HEADS:
                work_units.append((str(win_path), str(mut_path), head, ag, ws,
                                   TOP_PCT, perm_reps, n_boot,
                                   SEED_BASE + ws * 100 + hash(head + ag) % 100))

    # 5. Run all (head × construction) jobs in parallel
    log.info("Running %d (head × construction) evaluations ...", len(work_units))
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_process_one, wu): wu for wu in work_units}
        n_done = 0
        for f in as_completed(futures):
            try:
                res = f.result()
                results.append(res)
                n_done += 1
                log.info("[%d/%d] %s %s w=%d  abs_rec=%.4f  vs_TCW=%.2f  "
                         "vs_CpG=%.2f  bh=%d/10",
                         n_done, len(work_units),
                         res["head"], res["aggregator"], res["window_size_bp"],
                         res["mean_abs_recall"], res["mean_ratio_tcw"],
                         res["mean_ratio_cpg"], res["n_bh_signif"])
            except Exception as ex:
                log.error("FAIL: %s", ex)

    # 6. Build the all_heads_sweep.csv table
    rows = []
    for r in results:
        rows.append({
            "head": r["head"],
            "aggregator": r["aggregator"],
            "window_size_bp": r["window_size_bp"],
            "n_cancers": r["n_cancers"],
            "n_units": r["n_units"],
            "k_top1pct": r["k"],
            "mean_abs_recall": r["mean_abs_recall"],
            "ci_lo": r["ci_abs_lo"],
            "ci_hi": r["ci_abs_hi"],
            "mean_ratio_vs_TCW": r["mean_ratio_tcw"],
            "ratio_ci_lo": r["ratio_tcw_ci_lo"],
            "ratio_ci_hi": r["ratio_tcw_ci_hi"],
            "mean_ratio_vs_CpG": r["mean_ratio_cpg"],
            "ratio_cpg_ci_lo": r["ratio_cpg_ci_lo"],
            "ratio_cpg_ci_hi": r["ratio_cpg_ci_hi"],
            "n_cancers_above_TCW": r["n_cancers_above_TCW"],
            "n_cancers_above_CpG": r["n_cancers_above_CpG"],
            "n_bh_signif": r["n_bh_signif"],
        })
    df = pd.DataFrame(rows)
    # Sort: by construction, then head order
    df["agg_ws"] = df["aggregator"] + "_" + df["window_size_bp"].astype(str)
    construction_order = [f"{ag}_{ws}" for ag, ws in CONSTRUCTIONS]
    df["agg_ws_order"] = df["agg_ws"].map({k: i for i, k in
                                            enumerate(construction_order)})
    df["head_order"] = df["head"].map({h: i for i, h in enumerate(HEADS)})
    df = df.sort_values(["agg_ws_order", "head_order"]).drop(
        columns=["agg_ws", "agg_ws_order", "head_order"]).reset_index(drop=True)
    out_csv = OUT_DIR / "all_heads_sweep.csv"
    df.to_csv(out_csv, index=False)
    log.info("Wrote %s (%d rows)", out_csv, len(df))

    # 7. Top-1% Jaccard between binary and apobec1 panels (sum × 1000 bp)
    log.info("Computing top-1%% Jaccard (binary vs apobec1, sum × 1000) ...")
    ws = 1000
    win_path = OUT_DIR / f"_allheads_windows_w{ws}.parquet"
    w = pd.read_parquet(win_path)
    n = len(w)
    k = max(1, int(round(n * TOP_PCT)))
    s_bin = w["score_binary__sum"].to_numpy(dtype=np.float64)
    s_apo = w["score_apobec1__sum"].to_numpy(dtype=np.float64)
    bin_top = set(np.argpartition(-s_bin, k - 1)[:k].tolist())
    apo_top = set(np.argpartition(-s_apo, k - 1)[:k].tolist())
    jaccard = len(bin_top & apo_top) / len(bin_top | apo_top)
    log.info("  binary top-1%%: %d windows; apobec1 top-1%%: %d; "
             "intersection=%d; union=%d; Jaccard=%.4f",
             len(bin_top), len(apo_top), len(bin_top & apo_top),
             len(bin_top | apo_top), jaccard)

    # 8. Combined "binary OR apobec1" union panel (top-1% by max-score)
    log.info("Evaluating union panel (binary OR apobec1, sum × 1000 bp) ...")
    # Build sum-aggregation of the synthetic max-head
    p_or = panel.copy()
    p_or["pos"] = p_or["pos"].astype(int)
    p_or["win_start"] = (p_or["pos"] // ws) * ws
    or_grp = p_or.groupby(["chrom", "win_start"])["score_binary_OR_apobec1"].sum()
    or_w = or_grp.reset_index().rename(
        columns={"score_binary_OR_apobec1": "or_sum"})
    # Align to existing window order
    w_aligned = w[["chrom", "win_start"]].merge(or_w,
                                                 on=["chrom", "win_start"],
                                                 how="left")
    or_scores = w_aligned["or_sum"].to_numpy(dtype=np.float64)
    cpg = w["cpg_density"].to_numpy(dtype=np.float64)
    tcw = w["tcw_density"].to_numpy(dtype=np.float64)
    # Reuse the mut_per_cancer for ws=1000
    mut_path = OUT_DIR / f"_allheads_mut_w{ws}.json"
    mut_pc = json.loads(Path(mut_path).read_text())
    mut_per_cancer = {}
    for cancer, vals in mut_pc.items():
        arr = np.zeros(len(w), dtype=np.int32)
        for idx, count in vals:
            arr[int(idx)] = int(count)
        mut_per_cancer[cancer] = arr
    or_res = evaluate_construction(or_scores, cpg, tcw, mut_per_cancer,
                                    TOP_PCT, perm_reps, n_boot,
                                    SEED_BASE + 9999)
    log.info("  UNION abs_rec=%.4f vs_TCW=%.2f vs_CpG=%.2f bh=%d/10",
             or_res["mean_abs_recall"], or_res["mean_ratio_tcw"],
             or_res["mean_ratio_cpg"], or_res["n_bh_signif"])

    # Append union-panel row to the CSV for completeness
    extra_row = {
        "head": "UNION_binary_OR_apobec1",
        "aggregator": "sum",
        "window_size_bp": 1000,
        "n_cancers": or_res["n_cancers"],
        "n_units": or_res["n_units"],
        "k_top1pct": or_res["k"],
        "mean_abs_recall": or_res["mean_abs_recall"],
        "ci_lo": or_res["ci_abs_lo"],
        "ci_hi": or_res["ci_abs_hi"],
        "mean_ratio_vs_TCW": or_res["mean_ratio_tcw"],
        "ratio_ci_lo": or_res["ratio_tcw_ci_lo"],
        "ratio_ci_hi": or_res["ratio_tcw_ci_hi"],
        "mean_ratio_vs_CpG": or_res["mean_ratio_cpg"],
        "ratio_cpg_ci_lo": or_res["ratio_cpg_ci_lo"],
        "ratio_cpg_ci_hi": or_res["ratio_cpg_ci_hi"],
        "n_cancers_above_TCW": or_res["n_cancers_above_TCW"],
        "n_cancers_above_CpG": or_res["n_cancers_above_CpG"],
        "n_bh_signif": or_res["n_bh_signif"],
    }
    df = pd.concat([df, pd.DataFrame([extra_row])], ignore_index=True)
    df.to_csv(out_csv, index=False)

    # 9. Plot: grouped bars by head × construction, y=mean_ratio_vs_TCW with CI
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plot_df = df[df["head"] != "UNION_binary_OR_apobec1"].copy()
        plot_df["construction"] = (plot_df["aggregator"] + " × " +
                                    plot_df["window_size_bp"].astype(str) + "bp")
        constr_list = [f"{ag} × {ws}bp" for ag, ws in CONSTRUCTIONS]
        head_order = HEADS

        fig, ax = plt.subplots(figsize=(13, 6))
        n_constr = len(constr_list)
        n_heads = len(head_order)
        width = 0.8 / n_constr
        x = np.arange(n_heads)
        colors = ["#3b7dd8", "#e98a45", "#7b54b3"]
        for i, constr in enumerate(constr_list):
            sub = plot_df[plot_df["construction"] == constr].set_index("head")
            sub = sub.reindex(head_order)
            means = sub["mean_ratio_vs_TCW"].values
            lo = sub["ratio_ci_lo"].values
            hi = sub["ratio_ci_hi"].values
            yerr_lo = means - lo
            yerr_hi = hi - means
            ax.bar(x + i * width - 0.4 + width / 2, means, width,
                   yerr=[yerr_lo, yerr_hi], capsize=3, label=constr,
                   color=colors[i % len(colors)], edgecolor="black",
                   linewidth=0.5)
        ax.axhline(1.0, color="red", linestyle="--", alpha=0.5,
                   label="ratio=1 (no enrichment)")
        ax.set_xticks(x)
        ax.set_xticklabels([h.replace("score_", "") for h in head_order],
                            rotation=20, ha="right")
        ax.set_ylabel("Mean ratio NN / TCW-motif baseline (10 cancers, 95% CI)")
        ax.set_title("All-heads sweep: ratio vs TCW-motif at top three "
                      "constructions")
        ax.legend(loc="best", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        out_png = OUT_DIR / "all_heads_sweep.png"
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close()
        log.info("Wrote %s", out_png)
    except Exception as ex:
        log.error("Figure generation failed: %s", ex)

    # 10. Markdown report
    log.info("Writing ALL_HEADS_RESULTS.md ...")
    md_lines = []
    md_lines.append("# All-Heads Panel-Construction Sweep")
    md_lines.append("")
    md_lines.append("Compares all 7 score heads across the top three panel "
                    "constructions identified in the binary-head sweep.")
    md_lines.append("")
    md_lines.append("**Stratum:** TCW-non-CpG C>T mutations only (strict "
                    "APOBEC stratum). 10 cancer types: blca, brca, cesc, "
                    "coadread, esca, hnsc, lihc, lusc, skcm, stad. CIs are "
                    f"bootstrap 95% over 10 cancers ({n_boot:,} resamples). "
                    f"BH-FDR threshold q<0.025.")
    md_lines.append("")
    md_lines.append("## 1. Per-construction summary table")
    md_lines.append("")

    for ag, ws in CONSTRUCTIONS:
        md_lines.append(f"### Construction: **{ag} × {ws} bp**")
        md_lines.append("")
        md_lines.append("| head | abs_recall | 95% CI | ratio_vs_TCW | "
                        "TCW 95% CI | ratio_vs_CpG | n_above_TCW | "
                        "n_BH_sig |")
        md_lines.append("|------|------------|--------|--------------|"
                        "------------|---------------|--------------|"
                        "----------|")
        sub = df[(df["aggregator"] == ag) & (df["window_size_bp"] == ws)
                 & (df["head"].isin(HEADS))]
        sub = sub.set_index("head").reindex(HEADS).reset_index()
        for _, row in sub.iterrows():
            md_lines.append(
                f"| {row['head']} | {row['mean_abs_recall']*100:.2f}% | "
                f"[{row['ci_lo']*100:.2f}, {row['ci_hi']*100:.2f}]% | "
                f"{row['mean_ratio_vs_TCW']:.3f} | "
                f"[{row['ratio_ci_lo']:.3f}, {row['ratio_ci_hi']:.3f}] | "
                f"{row['mean_ratio_vs_CpG']:.2f} | "
                f"{int(row['n_cancers_above_TCW'])}/10 | "
                f"{int(row['n_bh_signif'])}/10 |"
            )
        md_lines.append("")

    md_lines.append("## 2. Key questions")
    md_lines.append("")

    # Q1: Best head on ratio_vs_TCW for the winning sum × 1000 construction
    sub_winning = df[(df["aggregator"] == "sum") &
                      (df["window_size_bp"] == 1000) &
                      (df["head"].isin(HEADS))].copy()
    sub_winning = sub_winning.sort_values("mean_ratio_vs_TCW",
                                          ascending=False).reset_index(drop=True)
    md_lines.append("### Q1: Does any head beat **score_binary** on "
                    "ratio_vs_TCW?")
    md_lines.append("")
    md_lines.append("Ranking at the winning construction (sum × 1000 bp):")
    md_lines.append("")
    md_lines.append("| rank | head | ratio_vs_TCW | 95% CI | abs_recall |")
    md_lines.append("|------|------|--------------|--------|------------|")
    for i, row in sub_winning.iterrows():
        md_lines.append(
            f"| {i+1} | {row['head']} | "
            f"{row['mean_ratio_vs_TCW']:.3f} | "
            f"[{row['ratio_ci_lo']:.3f}, {row['ratio_ci_hi']:.3f}] | "
            f"{row['mean_abs_recall']*100:.2f}% |"
        )
    md_lines.append("")
    binary_ratio = sub_winning[sub_winning["head"] == "score_binary"][
        "mean_ratio_vs_TCW"].iloc[0]
    binary_ci_hi = sub_winning[sub_winning["head"] == "score_binary"][
        "ratio_ci_hi"].iloc[0]
    binary_ci_lo = sub_winning[sub_winning["head"] == "score_binary"][
        "ratio_ci_lo"].iloc[0]
    best = sub_winning.iloc[0]
    if best["head"] == "score_binary":
        md_lines.append(f"**Answer: NO — score_binary remains the top head "
                        f"({binary_ratio:.3f}). The next-best head, "
                        f"{sub_winning.iloc[1]['head']}, has ratio="
                        f"{sub_winning.iloc[1]['mean_ratio_vs_TCW']:.3f}.**")
    else:
        # Check whether CIs overlap (i.e. is the difference real)
        best_lo = best["ratio_ci_lo"]
        if best_lo > binary_ci_hi:
            md_lines.append(
                f"**Answer: YES — `{best['head']}` beats score_binary "
                f"({best['mean_ratio_vs_TCW']:.3f} vs {binary_ratio:.3f}); "
                f"95% CIs do NOT overlap (best CI lower={best_lo:.3f} > "
                f"binary CI upper={binary_ci_hi:.3f}).**")
        else:
            md_lines.append(
                f"**Answer: PARTIALLY — `{best['head']}` has a numerically "
                f"higher ratio ({best['mean_ratio_vs_TCW']:.3f} vs "
                f"score_binary={binary_ratio:.3f}) but 95% CIs overlap, "
                f"so the difference is not statistically distinguishable.**")
    md_lines.append("")

    # Q2: Apobec1 disjointness via Jaccard
    md_lines.append("### Q2: Does the **apobec1** head capture different "
                    "windows than **binary**?")
    md_lines.append("")
    md_lines.append(f"Top-1% panels at sum × 1000 bp ({k} windows each):")
    md_lines.append("")
    md_lines.append(f"- binary ∩ apobec1: **{len(bin_top & apo_top):,}** "
                    f"windows")
    md_lines.append(f"- binary ∪ apobec1: **{len(bin_top | apo_top):,}** "
                    f"windows")
    md_lines.append(f"- **Jaccard = {jaccard:.3f}**")
    md_lines.append("")
    if jaccard > 0.7:
        md_lines.append(f"**Answer: NO — the panels are largely the same "
                        f"(Jaccard={jaccard:.3f}). The apobec1 head is not "
                        f"capturing materially different mutations.**")
    elif jaccard > 0.4:
        md_lines.append(f"**Answer: PARTIALLY — moderate overlap "
                        f"(Jaccard={jaccard:.3f}). Roughly half of the "
                        f"selected windows are unique to each head, so "
                        f"there IS room for orthogonal coverage.**")
    else:
        md_lines.append(f"**Answer: YES — substantial disjointness "
                        f"(Jaccard={jaccard:.3f}). The two heads pick "
                        f"largely different windows.**")
    md_lines.append("")

    # Q3: Combining heads
    md_lines.append("### Q3: Could **combining heads** (max → sum × 1000) "
                    "push recall higher?")
    md_lines.append("")
    md_lines.append("Union panel construction: at each position take "
                    "`max(score_binary, score_apobec1)`, then aggregate "
                    "(sum × 1000 bp) and select the top-1% windows.")
    md_lines.append("")
    md_lines.append("| panel | abs_recall | 95% CI | ratio_vs_TCW | "
                    "TCW 95% CI | n_BH_sig |")
    md_lines.append("|-------|------------|--------|--------------|"
                    "------------|----------|")
    binary_row = sub_winning[sub_winning["head"] == "score_binary"].iloc[0]
    apo_row = sub_winning[sub_winning["head"] == "score_apobec1"].iloc[0]
    md_lines.append(
        f"| score_binary | {binary_row['mean_abs_recall']*100:.2f}% | "
        f"[{binary_row['ci_lo']*100:.2f}, "
        f"{binary_row['ci_hi']*100:.2f}]% | "
        f"{binary_row['mean_ratio_vs_TCW']:.3f} | "
        f"[{binary_row['ratio_ci_lo']:.3f}, "
        f"{binary_row['ratio_ci_hi']:.3f}] | "
        f"{int(binary_row['n_bh_signif'])}/10 |"
    )
    md_lines.append(
        f"| score_apobec1 | {apo_row['mean_abs_recall']*100:.2f}% | "
        f"[{apo_row['ci_lo']*100:.2f}, "
        f"{apo_row['ci_hi']*100:.2f}]% | "
        f"{apo_row['mean_ratio_vs_TCW']:.3f} | "
        f"[{apo_row['ratio_ci_lo']:.3f}, "
        f"{apo_row['ratio_ci_hi']:.3f}] | "
        f"{int(apo_row['n_bh_signif'])}/10 |"
    )
    md_lines.append(
        f"| **UNION (max)** | {or_res['mean_abs_recall']*100:.2f}% | "
        f"[{or_res['ci_abs_lo']*100:.2f}, "
        f"{or_res['ci_abs_hi']*100:.2f}]% | "
        f"{or_res['mean_ratio_tcw']:.3f} | "
        f"[{or_res['ratio_tcw_ci_lo']:.3f}, "
        f"{or_res['ratio_tcw_ci_hi']:.3f}] | "
        f"{or_res['n_bh_signif']}/10 |"
    )
    md_lines.append("")
    if or_res["mean_abs_recall"] > binary_row["mean_abs_recall"] + 0.001:
        if or_res["ci_abs_lo"] > binary_row["ci_hi"]:
            md_lines.append(f"**Answer: YES — the union panel beats "
                            f"score_binary alone (recall "
                            f"{or_res['mean_abs_recall']*100:.2f}% vs "
                            f"{binary_row['mean_abs_recall']*100:.2f}%, "
                            f"CIs disjoint).**")
        else:
            md_lines.append(f"**Answer: NUMERICALLY YES, NOT RIGOROUSLY — "
                            f"union recall "
                            f"{or_res['mean_abs_recall']*100:.2f}% > "
                            f"binary {binary_row['mean_abs_recall']*100:.2f}%, "
                            f"but 95% CIs overlap.**")
    else:
        md_lines.append(f"**Answer: NO — the union panel does not improve "
                        f"recall ({or_res['mean_abs_recall']*100:.2f}% vs "
                        f"binary alone "
                        f"{binary_row['mean_abs_recall']*100:.2f}%).**")
    md_lines.append("")

    # Anomalies
    md_lines.append("## 3. Anomalies & sanity checks")
    md_lines.append("")
    anomalies = []
    # Heads with mean_ratio < 0.5 vs TCW = clearly worse than motif baseline
    sub_w = df[(df["aggregator"] == "sum") & (df["window_size_bp"] == 1000) &
               (df["head"].isin(HEADS))]
    weak_heads = sub_w[sub_w["mean_ratio_vs_TCW"] < 0.5][
        "head"].tolist()
    if weak_heads:
        anomalies.append(f"Heads with ratio_vs_TCW < 0.5 at sum×1000: "
                          f"{weak_heads}. These heads underperform the "
                          f"simple TCW-motif baseline.")
    # Heads with n_bh_signif < 5 (i.e. fewer than half the cancers reach FDR)
    weak_bh = sub_w[sub_w["n_bh_signif"] < 5]["head"].tolist()
    if weak_bh:
        anomalies.append(f"Heads with <5/10 BH-significant cancers at "
                          f"sum×1000: {weak_bh}.")
    if not anomalies:
        anomalies.append("None — all heads behaved within expected ranges "
                          "(non-zero recall, sensible ratio CIs).")
    for a in anomalies:
        md_lines.append(f"- {a}")
    md_lines.append("")

    md_lines.append("## 4. Files")
    md_lines.append("")
    md_lines.append("- `all_heads_sweep.csv` — full results table "
                     "(7 heads × 3 constructions = 21 rows + 1 union row)")
    md_lines.append("- `all_heads_sweep.png` — grouped bar chart")
    md_lines.append("- `ALL_HEADS_RESULTS.md` — this report")

    out_md = OUT_DIR / "ALL_HEADS_RESULTS.md"
    out_md.write_text("\n".join(md_lines))
    log.info("Wrote %s", out_md)

    # 11. Cleanup tmp files
    if not args.quick:
        for ws in window_sizes_needed:
            for tmp in [OUT_DIR / f"_allheads_windows_w{ws}.parquet",
                        OUT_DIR / f"_allheads_mut_w{ws}.json"]:
                if tmp.exists():
                    tmp.unlink()

    # Headline log
    log.info("=== HEADLINE: per-head ratio_vs_TCW @ sum × 1000 bp ===")
    for _, row in sub_winning.iterrows():
        log.info("  %-22s  vs_TCW=%.3f [%.3f, %.3f]  rec=%.4f  bh=%d/10",
                 row["head"], row["mean_ratio_vs_TCW"],
                 row["ratio_ci_lo"], row["ratio_ci_hi"],
                 row["mean_abs_recall"], int(row["n_bh_signif"]))
    log.info("UNION (binary OR apobec1)  vs_TCW=%.3f  rec=%.4f  bh=%d/10",
             or_res["mean_ratio_tcw"], or_res["mean_abs_recall"],
             or_res["n_bh_signif"])
    log.info("Top-1%% Jaccard (binary, apobec1) = %.4f", jaccard)

    return 0


if __name__ == "__main__":
    sys.exit(main())
