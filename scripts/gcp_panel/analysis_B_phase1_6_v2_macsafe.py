#!/usr/bin/env python3
"""Analysis B Phase 1.6 v2 — 100K-perm definitive primary endpoint with macOS-safe MP.

The original Phase 1.6 run (analysis_B_phase1_6_definitive_run.log) hung at
the gather/reduce step using `mp.get_context("fork").Pool`. This v2 script
uses `concurrent.futures.ProcessPoolExecutor` (different IPC mechanism, no
known macOS deadlock at this scale).

Reuses the already-built `windows_phase1_5.parquet` (250 bp, max-pool, all
score+count columns). Just runs the per-cancer permutation null and BH-FDR
at 100K perms for the primary endpoint.

Output: enrichment_primary_phase1_6_definitive_v2.json + REPORT_phase1_6_v2.md
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
from statsmodels.stats.multitest import multipletests

ROOT = Path("/Users/shaharharel/Documents/github/edit-rna-apobec")
WIN_PATH = ROOT / "experiments/multi_enzyme/outputs/pcawg_tcw_panel/analysis_B_coding_panel/windows_phase1_5.parquet"
OUT_DIR = ROOT / "experiments/multi_enzyme/outputs/pcawg_tcw_panel/analysis_B_coding_panel"

CANCERS_TCGA = ["blca", "brca", "cesc", "coadread", "esca", "hnsc", "lihc", "lusc", "skcm", "stad"]
PRIMARY_HEAD = "binary"
PRIMARY_PCT = 0.01
PRIMARY_FILTER_B = "tcw_not_cpg"
PRIMARY_ALPHA = 0.025
PRIMARY_MIN_LIFT_A = 1.5
PRIMARY_MIN_SIGNIF = 6
PRIMARY_MIN_LIFT_DRIVER = 1.3
PRIMARY_MIN_LIFT_MASKED = 1.3

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def recall_at_k(scores: np.ndarray, mut_counts: np.ndarray, pct: float):
    if len(scores) == 0:
        return 0, 0, 0
    k = max(1, int(len(scores) * pct))
    idx = np.argpartition(-scores, k - 1)[:k]
    return int(mut_counts[idx].sum()), int(mut_counts.sum()), k


def recall_ratio_with_perm(scores, base_scores, muts, pct, perm_reps, seed):
    """Window-level recall ratio + permutation null on score-rank labels."""
    rng = np.random.default_rng(seed)
    total = int(muts.sum())
    if total == 0 or len(scores) == 0:
        return {"recall_model": 0.0, "recall_baseline": 0.0, "ratio": float("nan"),
                "mut_in_top_model": 0, "total_mut": 0, "n_windows": int(len(scores)),
                "k": 0, "p_perm": 1.0, "n_perm": 0,
                "perm_mean": float("nan"), "perm_std": float("nan")}
    mut_top_obs, _, k = recall_at_k(scores, muts, pct)
    mut_top_base, _, _ = recall_at_k(base_scores, muts, pct)
    recall_model = mut_top_obs / total
    recall_base = mut_top_base / total if total > 0 else 0.0
    ratio = recall_model / recall_base if recall_base > 0 else float("nan")

    n_geq = 0
    perm_top_recall = np.zeros(perm_reps, dtype=np.int64)
    scores_perm = scores.copy()
    for i in range(perm_reps):
        rng.shuffle(scores_perm)
        mt, _, _ = recall_at_k(scores_perm, muts, pct)
        perm_top_recall[i] = mt
        if mt >= mut_top_obs:
            n_geq += 1
    p_perm = (n_geq + 1) / (perm_reps + 1)
    return {
        "recall_model": float(recall_model),
        "recall_baseline": float(recall_base),
        "ratio": float(ratio),
        "mut_in_top_model": int(mut_top_obs),
        "mut_in_top_baseline": int(mut_top_base),
        "total_mut": int(total),
        "n_windows": int(len(scores)),
        "k": int(k),
        "p_perm": float(p_perm),
        "n_perm": int(perm_reps),
        "perm_mean": float(perm_top_recall.mean()),
        "perm_std": float(perm_top_recall.std()),
    }


def per_cancer_worker(args):
    """Per-cancer worker: 4 scenarios (raw, masked, driver_ablated, primary).
    Receives plain numpy arrays — no DF pickle, no fork."""
    (cancer_idx, cancer, scores, mut_counts, cpg_density, is_driver,
     training_contam, perm_reps, seed) = args

    # Raw: all windows
    raw = recall_ratio_with_perm(scores, cpg_density, mut_counts, PRIMARY_PCT,
                                 perm_reps, seed)
    # Masked: drop training-contaminated
    keep = ~training_contam
    masked = recall_ratio_with_perm(scores[keep], cpg_density[keep],
                                    mut_counts[keep], PRIMARY_PCT,
                                    perm_reps, seed + 1)
    # Driver-ablated: drop driver windows
    keep = ~is_driver
    driver = recall_ratio_with_perm(scores[keep], cpg_density[keep],
                                    mut_counts[keep], PRIMARY_PCT,
                                    perm_reps, seed + 2)
    # Primary: drop both
    keep = ~is_driver & ~training_contam
    primary = recall_ratio_with_perm(scores[keep], cpg_density[keep],
                                     mut_counts[keep], PRIMARY_PCT,
                                     perm_reps, seed + 3)

    return cancer, {"raw": raw, "masked": masked, "driver_ablated": driver, "primary": primary}


def bootstrap_mean_ci(per_cancer_ratios, threshold, n_boot=10000, seed=20260425):
    arr = np.array([v for v in per_cancer_ratios if v == v], dtype=np.float64)
    n = len(arr)
    if n == 0:
        return {"n_cancers": 0, "mean_observed": float("nan"),
                "median_boot": float("nan"), "ci95_low": float("nan"),
                "ci95_high": float("nan"), "p_boot_le_thresh": float("nan"),
                "threshold": float(threshold), "n_boot": int(n_boot)}
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means[i] = sample.mean()
    obs_mean = float(arr.mean())
    p_boot = float((boot_means <= threshold).sum() + 1) / (n_boot + 1)
    return {
        "n_cancers": int(n), "mean_observed": obs_mean,
        "median_boot": float(np.median(boot_means)),
        "ci95_low": float(np.percentile(boot_means, 2.5)),
        "ci95_high": float(np.percentile(boot_means, 97.5)),
        "p_boot_le_thresh": p_boot, "threshold": float(threshold),
        "n_boot": int(n_boot),
    }


def joint_exceedance_test(per_cancer_ratios, threshold=1.0):
    arr = np.array([v for v in per_cancer_ratios if v == v], dtype=np.float64)
    n = len(arr)
    n_above = int((arr > threshold).sum())
    if n == 0:
        return {"n_cancers": 0, "n_above_1.0": 0, "binomial_p_one_sided": 1.0,
                "threshold": float(threshold)}
    from scipy.stats import binomtest
    res = binomtest(n_above, n, p=0.5, alternative="greater")
    return {
        "n_cancers": int(n), "n_above_1.0": n_above,
        "binomial_p_one_sided": float(res.pvalue),
        "threshold": float(threshold),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perm-reps", type=int, default=100000)
    ap.add_argument("--n-workers", type=int, default=8)
    ap.add_argument("--win-path", type=Path, default=WIN_PATH)
    ap.add_argument("--out-suffix", type=str, default="_phase1_6_definitive_v2")
    args = ap.parse_args()

    log.info("=== Analysis B Phase 1.6 v2 (macOS-safe) ===")
    log.info("perm_reps=%d, n_workers=%d, win_path=%s",
             args.perm_reps, args.n_workers, args.win_path)

    # Load Phase 1.5 windows
    log.info("Loading windows ...")
    w = pd.read_parquet(args.win_path)
    log.info("  rows: %d, cols: %d", len(w), len(w.columns))

    # The score column for max-aggregator is `score_binary` (raw)
    score_col = f"score_{PRIMARY_HEAD}"
    cpg_arr = w["cpg_density"].to_numpy(dtype=np.float64)
    is_driver = w["is_driver"].to_numpy(dtype=bool)
    training_contam = w["training_contaminated"].to_numpy(dtype=bool)
    scores = w[score_col].to_numpy(dtype=np.float64)

    # Pre-extract per-cancer mutation counts as 1-D arrays
    cancers_avail = []
    mut_arrs = {}
    for c in CANCERS_TCGA:
        col = f"n_tcw_not_cpg_{c}"
        if col in w.columns:
            cancers_avail.append(c)
            mut_arrs[c] = w[col].to_numpy(dtype=np.int64)
    log.info("  cancers: %s", cancers_avail)

    # Build worker args (plain numpy — pickle is safe)
    work_args = []
    for i, c in enumerate(cancers_avail):
        seed = 20260428 + i * 1000
        work_args.append((i, c, scores, mut_arrs[c], cpg_arr, is_driver,
                          training_contam, args.perm_reps, seed))

    log.info("Launching %d workers via ProcessPoolExecutor (macOS-safe spawn) ...",
             args.n_workers)
    t0 = time.time()
    outputs = []
    with ProcessPoolExecutor(max_workers=args.n_workers) as ex:
        futs = {ex.submit(per_cancer_worker, wa): wa[1] for wa in work_args}
        for f in as_completed(futs):
            cancer = futs[f]
            try:
                outputs.append(f.result())
                elapsed = time.time() - t0
                log.info("  [%d/%d] %s done in %.1fs",
                         len(outputs), len(work_args), cancer, elapsed)
            except Exception as ex:
                log.error("  %s FAILED: %s", cancer, ex)

    # Restore deterministic cancer order
    by_cancer = {c: d for c, d in outputs}
    outputs = [(c, by_cancer[c]) for c in cancers_avail if c in by_cancer]

    # Aggregate
    p_values = []; ratios_p = []; ratios_m = []; ratios_d = []; ratios_r = []
    results = {"per_cancer": {}, "pooled": {}, "pass_criteria": {}}
    for cancer, det in outputs:
        results["per_cancer"][cancer] = det
        raw = det["raw"]; masked = det["masked"]
        driver = det["driver_ablated"]; primary = det["primary"]
        log.info("  %s raw=%.3f masked=%.3f driver=%.3f primary=%.3f p_perm=%.3e total_mut=%d",
                 cancer, raw["ratio"], masked["ratio"], driver["ratio"],
                 primary["ratio"], primary["p_perm"], primary["total_mut"])
        p_values.append(primary["p_perm"])
        ratios_r.append(raw["ratio"]); ratios_m.append(masked["ratio"])
        ratios_d.append(driver["ratio"]); ratios_p.append(primary["ratio"])

    if p_values:
        rej, q, _, _ = multipletests(p_values, alpha=PRIMARY_ALPHA, method="fdr_bh")
    else:
        rej = []; q = []
    cancers_in_order = [c for c, _ in outputs]
    for i, c in enumerate(cancers_in_order):
        if i < len(q):
            results["per_cancer"][c]["primary"]["q_bh"] = float(q[i])
            results["per_cancer"][c]["primary"]["reject_bh"] = bool(rej[i])

    mean_p = float(np.nanmean(ratios_p)) if ratios_p else 0.0
    mean_m = float(np.nanmean(ratios_m)) if ratios_m else 0.0
    mean_d = float(np.nanmean(ratios_d)) if ratios_d else 0.0
    n_signif = int(sum(rej)) if len(rej) else 0
    pass_a = bool(mean_p >= PRIMARY_MIN_LIFT_A)
    pass_b = bool(n_signif >= PRIMARY_MIN_SIGNIF)
    pass_c = bool(mean_d >= PRIMARY_MIN_LIFT_DRIVER)
    pass_d = bool(mean_m >= PRIMARY_MIN_LIFT_MASKED)
    passed = pass_a and pass_b and pass_c and pass_d

    boot_a = bootstrap_mean_ci(ratios_p, PRIMARY_MIN_LIFT_A, n_boot=10000, seed=20260611)
    boot_c = bootstrap_mean_ci(ratios_d, PRIMARY_MIN_LIFT_DRIVER, n_boot=10000, seed=20260612)
    boot_d_ = bootstrap_mean_ci(ratios_m, PRIMARY_MIN_LIFT_MASKED, n_boot=10000, seed=20260613)
    joint_exc = joint_exceedance_test(ratios_p, threshold=1.0)

    results["pooled"] = {
        "mean_ratio_raw": float(np.nanmean(ratios_r)) if ratios_r else 0.0,
        "mean_ratio_masked": mean_m,
        "mean_ratio_driver": mean_d,
        "mean_ratio_primary": mean_p,
        "n_cancers_signif_q025": n_signif,
    }
    results["pass_criteria"] = {
        "(a)_mean_primary_>=_1.5": {"val": mean_p, "thresh": PRIMARY_MIN_LIFT_A,
                                    "pass": pass_a, "bootstrap": boot_a},
        "(b)_signif_q<0.025_>=_6": {"val": n_signif, "thresh": PRIMARY_MIN_SIGNIF,
                                    "pass": pass_b, "alpha": PRIMARY_ALPHA},
        "(c)_driver_>=_1.3": {"val": mean_d, "thresh": PRIMARY_MIN_LIFT_DRIVER,
                              "pass": pass_c, "bootstrap": boot_c},
        "(d)_masked_>=_1.3": {"val": mean_m, "thresh": PRIMARY_MIN_LIFT_MASKED,
                              "pass": pass_d, "bootstrap": boot_d_},
        "joint_exceedance_n_above_1.0": joint_exc,
        "PASS": passed,
    }
    results["config"] = {
        "perm_reps": args.perm_reps,
        "n_workers": args.n_workers,
        "win_path": str(args.win_path),
        "mp_mode": "concurrent.futures.ProcessPoolExecutor",
        "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    log.info("PRIMARY: %s", "PASS" if passed else "FAIL")
    log.info("  (a) mean_primary=%.3f  thresh=%.3f  pass=%s  CI95=[%.3f, %.3f]",
             mean_p, PRIMARY_MIN_LIFT_A, pass_a,
             boot_a.get("ci95_low", float("nan")), boot_a.get("ci95_high", float("nan")))
    log.info("  (b) n_signif=%d/%d  thresh=%d  pass=%s  alpha=%.3f",
             n_signif, len(cancers_in_order), PRIMARY_MIN_SIGNIF, pass_b, PRIMARY_ALPHA)
    log.info("  (c) mean_driver=%.3f  thresh=%.3f  pass=%s",
             mean_d, PRIMARY_MIN_LIFT_DRIVER, pass_c)
    log.info("  (d) mean_masked=%.3f  thresh=%.3f  pass=%s",
             mean_m, PRIMARY_MIN_LIFT_MASKED, pass_d)
    log.info("  joint_exceedance: n_above_1.0=%d/%d binom_p=%.3e",
             joint_exc.get("n_above_1.0", 0), joint_exc.get("n_cancers", 0),
             joint_exc.get("binomial_p_one_sided", 1.0))

    out_json = OUT_DIR / f"enrichment_primary{args.out_suffix}.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info("Wrote %s", out_json)

    # Brief report
    report = OUT_DIR / f"REPORT{args.out_suffix}.md"
    with open(report, "w") as f:
        f.write(f"# Analysis B Phase 1.6 v2 — {args.perm_reps} perms (macOS-safe)\n\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(f"## Pre-registered primary endpoint result: **{'PASS' if passed else 'FAIL'}**\n\n")
        f.write(f"Tested at {args.perm_reps} permutations per cancer.\n\n")
        f.write(f"- (a) mean_primary >= 1.5: **{mean_p:.3f}** (CI95 [{boot_a['ci95_low']:.3f}, {boot_a['ci95_high']:.3f}]) — {'PASS' if pass_a else 'FAIL'}\n")
        f.write(f"- (b) n_signif (BH q<{PRIMARY_ALPHA}) >= 6: **{n_signif}**/10 — {'PASS' if pass_b else 'FAIL'}\n")
        f.write(f"- (c) mean_driver >= 1.3: **{mean_d:.3f}** — {'PASS' if pass_c else 'FAIL'}\n")
        f.write(f"- (d) mean_masked >= 1.3: **{mean_m:.3f}** — {'PASS' if pass_d else 'FAIL'}\n\n")
        f.write("## Per-cancer (primary scenario, post driver+training-mask ablation)\n\n")
        f.write("| cancer | ratio | recall_model | recall_cpg | total_mut | p_perm | q_bh | reject |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---|\n")
        for c, det in results["per_cancer"].items():
            pr = det["primary"]
            f.write(f"| {c} | {pr['ratio']:.3f} | {pr['recall_model']:.4f} | "
                    f"{pr['recall_baseline']:.4f} | {pr['total_mut']} | "
                    f"{pr['p_perm']:.3e} | {pr.get('q_bh', float('nan')):.3g} | "
                    f"{'Y' if pr.get('reject_bh', False) else 'N'} |\n")
    log.info("Wrote %s", report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
