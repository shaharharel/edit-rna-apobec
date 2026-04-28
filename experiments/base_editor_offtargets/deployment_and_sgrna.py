#!/usr/bin:env python3
"""Deployment metric + per-sgRNA sensitivity from v5 all_scored.csv.

Deployment metric:
  For each percentile T (10, 25, 50, 75, 90, 95, 99) on each head's score:
    - frac_positives_above = fraction of Lei positives in top-T% scores
    - frac_genome_above = fraction of all sites (pool) in top-T% scores (= 1-T/100 by def)
    - enrichment_factor = frac_positives_above / frac_genome_above
  For ANTI-PREDICTORS (A1):
    - frac_positives_below = fraction of positives in bottom-T% (= NOT in top-T%)
    - "safe region" = bottom-T% of score
    - genome_fraction_safe = T/100
    - off_target_density = positives in safe / total positives

Per-sgRNA sensitivity:
  Split positives + their motif-matched controls by sgRNA (4 sgRNAs).
  Recompute A3G p95 enrichment per sgRNA. If consistent, robust; if outlier-driven, fragile.
"""
from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

ROOT = Path("/Users/shaharharel/Documents/github/edit-rna-apobec")
V5 = ROOT / "experiments/base_editor_offtargets/outputs/lei_v5_sensitivity"
OUT = V5 / "deployment_sgrna"
OUT.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("deploy")

# Use Bperm + filtered pipeline (motif-controlled, sgRNA-filtered = most defensible)
HEADS_FOR_DEPLOYMENT = ["a1_new_sp1", "a1_old_sp1", "phase3_A3G", "phase3_A3A"]
DEPLOY_PERCENTILES = [99.9, 99, 95, 90, 75, 50, 25]


def load_pipeline(df, opt="Bperm", filter_mode="filtered"):
    is_cand = df["src"] == "candidate"
    if opt == "A":
        cand_mask = is_cand & (df["in_A"] == True)
    elif opt == "Bperm":
        cand_mask = is_cand & (df["in_Bperm"] == True)
    elif opt == "Bstrict":
        cand_mask = is_cand & (df["in_Bstrict"] == True)
    pos = df[cand_mask & df["valid"]].copy()
    pos_set = set(zip(pos["chrom"], pos["pos"]))
    ctrl_local = df[(df["src"] == "ctrl_local") & df["valid"] &
                    df.apply(lambda r: (r["chrom"], r["matched_to"]) in pos_set, axis=1)].copy()
    ctrl_global = df[(df["src"] == "ctrl_global") & df["valid"] &
                     df.apply(lambda r: (r["chrom"], r["matched_to"]) in pos_set, axis=1)].copy()
    if filter_mode == "filtered":
        pos = pos[~pos["grna_dep"]]
        ctrl_local = ctrl_local[~ctrl_local["grna_dep"]]
        ctrl_global = ctrl_global[~ctrl_global["grna_dep"]]
    return pos, ctrl_local, ctrl_global


def deployment_metric(pos, ctrl, head):
    """For each percentile T, compute:
    - mask top-T% as 'predicted off-target zone'
    - frac_pos_in_zone, frac_zone_of_genome, enrichment_factor
    Plus 'anti-predictor' view:
    - mask bottom-T% as 'safe zone'
    - frac_pos_in_safe, etc.
    """
    score_col = f"score_{head}"
    pos_scores = pos[score_col].values
    ctrl_scores = ctrl[score_col].values
    combined = np.concatenate([pos_scores, ctrl_scores])
    n_pos = len(pos_scores); n_ctrl = len(ctrl_scores)
    rows = []
    for pct in DEPLOY_PERCENTILES:
        thr = float(np.percentile(combined, pct))
        # POSITIVE PREDICTOR view: top-T% = predicted off-target zone
        pos_in_top = (pos_scores >= thr).sum()
        ctrl_in_top = (ctrl_scores >= thr).sum()
        frac_pos_top = pos_in_top / n_pos if n_pos > 0 else 0
        frac_genome_top = (100 - pct) / 100  # nominal
        enrichment = frac_pos_top / frac_genome_top if frac_genome_top > 0 else float("inf")

        # ANTI-PREDICTOR view: top-T% = "AVOID this region"
        # If we EXCLUDE top-T% from sgRNA candidate selection:
        #   - fraction of genome retained = pct/100
        #   - fraction of off-targets retained = 1 - frac_pos_top
        #   - density reduction factor = (frac_pos_retained) / (frac_genome_retained)
        frac_pos_retained = 1 - frac_pos_top
        frac_genome_retained = pct / 100
        density_reduction = frac_pos_retained / frac_genome_retained if frac_genome_retained > 0 else float("inf")

        rows.append({
            "head": head, "percentile": pct, "threshold": thr,
            "n_pos": n_pos, "n_ctrl": n_ctrl,
            "pos_in_top": int(pos_in_top), "ctrl_in_top": int(ctrl_in_top),
            "frac_pos_top": frac_pos_top,
            "frac_genome_top": frac_genome_top,
            "enrichment_factor": enrichment,
            "frac_pos_retained_below_thr": frac_pos_retained,
            "frac_genome_retained_below_thr": frac_genome_retained,
            "density_reduction_factor": density_reduction,
        })
    return pd.DataFrame(rows)


def per_sgrna_enrichment(pos, ctrl_global, head, percentile=95):
    """Per-sgRNA breakdown of enrichment."""
    score_col = f"score_{head}"
    rows = []
    for sgrna in pos["sgRNA"].unique():
        p = pos[pos["sgRNA"] == sgrna]
        c = ctrl_global[ctrl_global["sgRNA"] == sgrna]
        if len(p) < 30 or len(c) < 30:
            log.info("  Skipping sgRNA %s: pos=%d ctrl=%d", sgrna, len(p), len(c))
            continue
        sc = np.concatenate([p[score_col].values, c[score_col].values])
        is_pos = np.concatenate([np.ones(len(p), bool), np.zeros(len(c), bool)])
        thr = float(np.percentile(sc, percentile))
        above = sc >= thr
        pa = int((is_pos & above).sum())
        pb = int((is_pos & ~above).sum())
        ca = int((~is_pos & above).sum())
        cb = int((~is_pos & ~above).sum())
        if pb > 0 and ca > 0:
            or_v = (pa * cb) / (pb * ca)
        else:
            or_v = float("inf") if pb == 0 else 0.0
        _, p_v = fisher_exact([[pa, pb], [ca, cb]])
        rows.append({
            "sgRNA": sgrna, "head": head, "percentile": percentile,
            "n_pos": int(is_pos.sum()), "n_ctrl": int((~is_pos).sum()),
            "pos_above": pa, "ctrl_above": ca,
            "or": or_v, "p_value": float(p_v),
        })
    return pd.DataFrame(rows)


def main():
    log.info("Loading v5 all_scored.csv...")
    df = pd.read_csv(V5 / "all_scored.csv", low_memory=False)
    log.info("  %d rows", len(df))

    # Use Bperm + filtered + global as primary pipeline
    log.info("Loading Bperm/filtered pipeline...")
    pos, ctrl_local, ctrl_global = load_pipeline(df, "Bperm", "filtered")
    log.info("Pos=%d, ctrl_local=%d, ctrl_global=%d", len(pos), len(ctrl_local), len(ctrl_global))

    # ----- Deployment metric -----
    log.info("Computing deployment metrics for each head...")
    all_deploy = []
    for head in HEADS_FOR_DEPLOYMENT:
        # vs global
        d = deployment_metric(pos, ctrl_global, head)
        d["control_set"] = "global"
        all_deploy.append(d)
        # vs local
        d2 = deployment_metric(pos, ctrl_local, head)
        d2["control_set"] = "local"
        all_deploy.append(d2)
    deploy_df = pd.concat(all_deploy, ignore_index=True)
    deploy_df.to_csv(OUT / "deployment_metrics.csv", index=False)
    log.info("Wrote %s", OUT / "deployment_metrics.csv")

    # Print deployment summary
    print("\n" + "=" * 100)
    print("DEPLOYMENT METRIC: Bperm/filtered/global")
    print("=" * 100)
    print("Positive-predictor view: top-T% scores capture what fraction of off-targets?")
    print("Anti-predictor view: if you AVOID top-T% scores, what fraction of off-targets remain?")
    print()
    for head in HEADS_FOR_DEPLOYMENT:
        sub = deploy_df[(deploy_df["head"] == head) & (deploy_df["control_set"] == "global")]
        print(f"\n--- {head} ---")
        print(sub[["percentile", "frac_pos_top", "frac_genome_top",
                   "enrichment_factor", "frac_pos_retained_below_thr",
                   "density_reduction_factor"]].to_string(index=False, float_format="%.3f"))

    # ----- Per-sgRNA sensitivity -----
    log.info("\nPer-sgRNA sensitivity (Bperm/filtered/global, p95)...")
    all_sgrna = []
    for head in HEADS_FOR_DEPLOYMENT:
        d = per_sgrna_enrichment(pos, ctrl_global, head, percentile=95)
        all_sgrna.append(d)
    sgrna_df = pd.concat(all_sgrna, ignore_index=True)
    sgrna_df["q_value"] = multipletests(sgrna_df["p_value"].fillna(1.0), method="fdr_bh")[1]
    sgrna_df.to_csv(OUT / "per_sgrna.csv", index=False)
    log.info("Wrote %s", OUT / "per_sgrna.csv")

    print("\n" + "=" * 100)
    print("PER-sgRNA SENSITIVITY at p95 (Bperm/filtered/global)")
    print("=" * 100)
    for head in HEADS_FOR_DEPLOYMENT:
        sub = sgrna_df[sgrna_df["head"] == head].sort_values("sgRNA")
        if sub.empty: continue
        print(f"\n--- {head} ---")
        print(sub[["sgRNA", "n_pos", "n_ctrl", "pos_above", "ctrl_above",
                   "or", "p_value", "q_value"]].to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
