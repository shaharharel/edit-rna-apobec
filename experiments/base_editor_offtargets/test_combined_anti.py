#!/usr/bin/env python3
"""Test combined anti-predictor: A1 + A3G-DNA + phase3_binary all-high → safe regions.

Hypothesis: combining the three anti-predictors gives a stronger / more
practically useful safety filter than any single anti-predictor.

Metrics:
  Enrichment view: top-T% of combined anti-prior → OR (should be < 1 = depleted of off-targets)
  Deployment view: if we mask top-T% combined → what fraction of off-targets removed?
                   compared to T% (random expectation) and to single-head masking
  Combined predictor view: top-T% A3G AND bottom-T% combined-anti → likely-off-target
"""
from __future__ import annotations
import json, logging
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

ROOT = Path("/Users/shaharharel/Documents/github/edit-rna-apobec")
IN_CSV = ROOT / "experiments/base_editor_offtargets/outputs/lei_a3g_dna_test/lei_bperm_with_a3g_dna.csv"
OUT = ROOT / "experiments/base_editor_offtargets/outputs/lei_combined_anti"
OUT.mkdir(parents=True, exist_ok=True)

ANTI_HEADS = ["a1_new_sp1", "a3g_dna_v1", "score_phase3_binary"]  # latter has score_ prefix from v5
PERCENTILES = [60, 70, 75, 80, 85, 90, 95, 99]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("combined")


def fisher_or(scores, is_pos, thr):
    above = scores >= thr
    pa = int((is_pos & above).sum()); pb = int((is_pos & ~above).sum())
    ca = int((~is_pos & above).sum()); cb = int((~is_pos & ~above).sum())
    if pb == 0 or ca == 0:
        or_v = float("inf") if pb == 0 else 0.0
    else:
        or_v = (pa * cb) / (pb * ca)
    _, p = fisher_exact([[pa, pb], [ca, cb]])
    return float(or_v), float(p), pa, ca


def main():
    log.info("Loading scored Bperm with A3G-DNA + originals...")
    df = pd.read_csv(IN_CSV, low_memory=False)
    log.info("  rows: %d", len(df))

    # Find valid valid_recomp
    if "valid_recomp" in df.columns:
        df = df[df["valid_recomp"]].reset_index(drop=True)

    # Build combined scores
    # Anti-priors (high score = predicted-NOT-edited): a1_new_sp1, a3g_dna_v1, score_phase3_binary
    s_a1 = df["score_a1_new_sp1"].values
    s_a3g_dna = df["score_a3g_dna_v1"].values
    s_binary = df["score_phase3_binary"].values
    s_a3g_rna = df["score_phase3_A3G"].values  # positive predictor

    # Min-max normalize each to [0,1] before combining (so equal weight)
    def minmax(x):
        lo, hi = np.percentile(x, [1, 99])
        return np.clip((x - lo) / max(hi - lo, 1e-9), 0, 1)

    s_a1_n = minmax(s_a1)
    s_a3g_dna_n = minmax(s_a3g_dna)
    s_binary_n = minmax(s_binary)
    s_a3g_rna_n = minmax(s_a3g_rna)

    # Combined anti-prior: mean of normalized anti-heads (high = SAFE)
    combined_anti = (s_a1_n + s_a3g_dna_n + s_binary_n) / 3
    df["score_combined_anti"] = combined_anti

    # Combined positive predictor: A3G adapter (positive) MINUS combined anti
    df["score_combined_positive"] = s_a3g_rna_n - combined_anti
    # Also: simple max-vote anti (high if ANY head says high)
    df["score_max_anti"] = np.maximum.reduce([s_a1_n, s_a3g_dna_n, s_binary_n])
    # Strict AND: min of three (high only if ALL heads say high)
    df["score_min_anti"] = np.minimum.reduce([s_a1_n, s_a3g_dna_n, s_binary_n])

    df.to_csv(OUT / "scored_with_combined.csv", index=False)
    log.info("Wrote %s", OUT / "scored_with_combined.csv")

    # Enrichment for each combined predictor
    rows = []
    for ctrl_lbl, ctrl_name in [("control_local", "local"), ("control_global", "global")]:
        is_pos = (df["label"] == "positive").values
        is_ctrl = (df["label"] == ctrl_lbl).values
        keep = is_pos | is_ctrl
        ip = is_pos[keep]
        n_pos = int(ip.sum()); n_ctrl = int((~ip).sum())
        log.info("Control set %s: pos=%d ctrl=%d", ctrl_name, n_pos, n_ctrl)
        for head in ["score_combined_anti", "score_combined_positive", "score_max_anti", "score_min_anti",
                      "score_a1_new_sp1", "score_a3g_dna_v1", "score_phase3_binary",
                      "score_phase3_A3G"]:
            sc = df.loc[keep, head].values
            for pct in PERCENTILES:
                thr = float(np.percentile(sc, pct))
                or_v, p, pa, ca = fisher_or(sc, ip, thr)
                rows.append({
                    "ctrl_set": ctrl_name, "head": head.replace("score_", ""),
                    "percentile": pct,
                    "n_pos": n_pos, "n_ctrl": n_ctrl,
                    "pos_above": pa, "ctrl_above": ca,
                    "or": or_v, "p_value": p,
                })
    enrich = pd.DataFrame(rows)
    enrich["q_value"] = multipletests(enrich["p_value"].fillna(1.0), method="fdr_bh")[1]
    enrich.to_csv(OUT / "enrichment_combined.csv", index=False)
    log.info("Wrote %s", OUT / "enrichment_combined.csv")

    # Deployment view: how many off-targets removed at each cumulative-mask threshold?
    deploy_rows = []
    is_pos = (df["label"] == "positive").values
    is_ctrl_g = (df["label"] == "control_global").values
    n_pos_total = is_pos.sum()
    for head in ["score_combined_anti", "score_max_anti", "score_min_anti",
                  "score_a1_new_sp1", "score_a3g_dna_v1", "score_phase3_binary"]:
        sc_all = df[head].values  # use full set for percentile computation
        for pct in [50, 75, 90, 95, 99]:
            thr_top = float(np.percentile(sc_all, pct))
            # If we MASK (exclude) top-T% of combined anti score regions:
            #   genome retained = T/100 of positions
            #   off-targets retained = positives below threshold / total positives
            below_thr = sc_all < thr_top
            pos_retained = (is_pos & below_thr).sum()
            frac_pos_retained = pos_retained / n_pos_total
            frac_genome_retained = pct / 100.0
            density_reduction = frac_pos_retained / frac_genome_retained if frac_genome_retained > 0 else float("inf")
            deploy_rows.append({
                "head": head.replace("score_", ""), "mask_top_pct": 100 - pct,
                "thr": thr_top,
                "n_pos_total": int(n_pos_total),
                "n_pos_retained_below_thr": int(pos_retained),
                "frac_pos_retained": frac_pos_retained,
                "frac_genome_retained": frac_genome_retained,
                "density_reduction_factor": density_reduction,
            })
    deploy = pd.DataFrame(deploy_rows)
    deploy.to_csv(OUT / "deployment_combined.csv", index=False)

    # Print summary
    print("\n" + "=" * 100)
    print("ENRICHMENT (Bperm/global) — combined anti-predictor vs single heads")
    print("=" * 100)
    sub = enrich[enrich["ctrl_set"] == "global"]
    piv = sub.pivot_table(index="head", columns="percentile", values="or", aggfunc="first")
    piv_q = sub.pivot_table(index="head", columns="percentile", values="q_value", aggfunc="first")
    out_str = piv.copy().astype(object)
    for h in piv.index:
        for pct in piv.columns:
            or_v = piv.loc[h, pct]; q = piv_q.loc[h, pct]
            if pd.isna(or_v): out_str.loc[h, pct] = "—"
            else: out_str.loc[h, pct] = f"{or_v:.2f}(q={q:.0e})"
    print(out_str.to_string())

    print("\n" + "=" * 100)
    print("DEPLOYMENT — if we mask top-X% of each anti-predictor, how much off-target burden remains?")
    print("=" * 100)
    print("density_reduction_factor < 1.0 = retained genome has FEWER off-targets per bp (good safety filter)")
    print()
    pivd = deploy.pivot_table(index="head", columns="mask_top_pct",
                                values="density_reduction_factor", aggfunc="first")
    print(pivd.to_string(float_format="%.3f"))

    print("\n" + "=" * 100)
    print("COMBINED POSITIVE PREDICTOR (A3G adapter score MINUS combined anti)")
    print("=" * 100)
    sub2 = enrich[(enrich["ctrl_set"] == "global") & (enrich["head"] == "combined_positive")]
    print(sub2[["percentile", "n_pos", "n_ctrl", "pos_above", "ctrl_above",
                "or", "p_value", "q_value"]].to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
