#!/usr/bin/env python3
"""Extract p95/p99/p999 enrichment from v5 all_scored.csv across all 8 pipelines.

p999 not in original PERCENTILES list — recompute from scored data.
For pipelines with N too small at p999 (e.g., Bstrict has 391 positives →
top 0.1% = 0.4 positives), report explicitly.
"""
from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

ROOT = Path("/Users/shaharharel/Documents/github/edit-rna-apobec")
V5_DIR = ROOT / "experiments/base_editor_offtargets/outputs/lei_v5_sensitivity"
OUT_PATH = V5_DIR / "high_percentiles.csv"

PERCENTILES = [90, 95, 99, 99.9]
N_BOOT = 500
SEED = 42

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("highpct")


def fisher_or(scores, is_pos, thr):
    above = scores >= thr
    pa = int((is_pos & above).sum()); pb = int((is_pos & ~above).sum())
    ca = int((~is_pos & above).sum()); cb = int((~is_pos & ~above).sum())
    if pb == 0 or ca == 0:
        or_v = float("inf") if pb == 0 else 0.0
    else:
        or_v = (pa * cb) / (pb * ca)
    _, p = fisher_exact([[pa, pb], [ca, cb]])
    return float(or_v), float(p), pa, pb, ca, cb


def bootstrap_ci(scores, is_pos, thr, n_boot=N_BOOT):
    rng = np.random.default_rng(SEED)
    n_pos = is_pos.sum(); n_ctrl = (~is_pos).sum()
    if n_pos == 0 or n_ctrl == 0: return np.nan, np.nan
    pos_idx = np.where(is_pos)[0]; ctrl_idx = np.where(~is_pos)[0]
    ors = []
    for _ in range(n_boot):
        pi = rng.choice(pos_idx, n_pos, replace=True)
        ci = rng.choice(ctrl_idx, n_ctrl, replace=True)
        pa = (scores[pi] >= thr).sum(); pb = n_pos - pa
        ca = (scores[ci] >= thr).sum(); cb = n_ctrl - ca
        if pb > 0 and ca > 0: ors.append((pa * cb) / (pb * ca))
    if not ors: return np.nan, np.nan
    return tuple(np.percentile(ors, [2.5, 97.5]))


def get_scores_and_pos(df, opt, filter_mode, ctrl_mode, score_col):
    is_cand = df["src"] == "candidate"
    if opt == "A":
        cand_mask = is_cand & (df["in_A"] == True)
    elif opt == "Bperm":
        cand_mask = is_cand & (df["in_Bperm"] == True)
    elif opt == "Bstrict":
        cand_mask = is_cand & (df["in_Bstrict"] == True)
    else:
        return None, None

    pos_set = set(zip(df.loc[cand_mask, "chrom"], df.loc[cand_mask, "pos"]))
    ctrl_mask = (df["src"] == f"ctrl_{ctrl_mode}") & (
        df.apply(lambda r: (r["chrom"], r["matched_to"]) in pos_set, axis=1))

    if filter_mode == "filtered":
        cand_mask = cand_mask & (~df["grna_dep"])
        ctrl_mask = ctrl_mask & (~df["grna_dep"])

    cand_mask = cand_mask & df["valid"]
    ctrl_mask = ctrl_mask & df["valid"]

    pos_scores = df.loc[cand_mask, score_col].values
    ctrl_scores = df.loc[ctrl_mask, score_col].values
    if len(pos_scores) < 50 or len(ctrl_scores) < 50:
        return None, None
    combined = np.concatenate([pos_scores, ctrl_scores])
    is_pos = np.concatenate([np.ones(len(pos_scores), bool),
                              np.zeros(len(ctrl_scores), bool)])
    return combined, is_pos


def main():
    log.info("Loading all_scored.csv (this is large)...")
    df = pd.read_csv(V5_DIR / "all_scored.csv", low_memory=False)
    log.info("  %d rows", len(df))

    score_cols = [c for c in df.columns if c.startswith("score_")]
    heads = [c.replace("score_", "") for c in score_cols]

    rows = []
    pipelines = []
    for opt in ["A", "Bperm", "Bstrict"]:
        for fm in ["unfiltered", "filtered"]:
            for cm in ["local", "global"]:
                pipelines.append((opt, fm, cm))

    log.info("Computing %d pipelines × %d heads × %d percentiles...",
             len(pipelines), len(heads), len(PERCENTILES))

    for opt, fm, cm in pipelines:
        for head in heads:
            score_col = f"score_{head}"
            combined, is_pos = get_scores_and_pos(df, opt, fm, cm, score_col)
            if combined is None:
                continue
            for pct in PERCENTILES:
                thr = float(np.percentile(combined, pct))
                or_v, p, pa, pb, ca, cb = fisher_or(combined, is_pos, thr)
                ci_lo, ci_hi = bootstrap_ci(combined, is_pos, thr)
                rows.append({
                    "pipeline": f"{opt}_{fm}_{cm}", "option": opt,
                    "filter": fm, "ctrl_set": cm, "head": head,
                    "percentile": pct, "threshold": thr,
                    "n_pos": int(is_pos.sum()), "n_ctrl": int((~is_pos).sum()),
                    "n_pos_above": pa, "n_ctrl_above": ca,
                    "or": or_v, "p_value": p,
                    "ci_lo": float(ci_lo) if ci_lo == ci_lo else np.nan,
                    "ci_hi": float(ci_hi) if ci_hi == ci_hi else np.nan,
                })
    out = pd.DataFrame(rows)
    out["q_value"] = multipletests(out["p_value"].fillna(1.0), method="fdr_bh")[1]
    out.to_csv(OUT_PATH, index=False)
    log.info("Wrote %s", OUT_PATH)

    # Print key results
    KEY_HEADS = ["phase3_A3G", "phase3_A3A", "a1_new_sp1", "a1_old_sp1"]
    for head in KEY_HEADS:
        print("\n" + "=" * 90)
        print(f"{head} — OR by pipeline × percentile")
        print("=" * 90)
        sub = out[out["head"] == head].copy()
        if sub.empty: continue
        # Pivot: pipeline × percentile
        piv = sub.pivot_table(index="pipeline", columns="percentile",
                               values="or", aggfunc="first")
        piv_p = sub.pivot_table(index="pipeline", columns="percentile",
                                 values="q_value", aggfunc="first")
        piv_n = sub.pivot_table(index="pipeline", columns="percentile",
                                 values="n_pos_above", aggfunc="first")
        # Format combined: "OR (q, n_above)"
        out_str = piv.copy().astype(object)
        for pipe in piv.index:
            for pct in piv.columns:
                or_v = piv.loc[pipe, pct]
                q = piv_p.loc[pipe, pct]
                n_a = piv_n.loc[pipe, pct]
                if pd.isna(or_v):
                    out_str.loc[pipe, pct] = "—"
                elif or_v == float("inf"):
                    out_str.loc[pipe, pct] = "inf"
                else:
                    out_str.loc[pipe, pct] = f"{or_v:.2f} (q={q:.1e}, n+={int(n_a)})"
        print(out_str.to_string())


if __name__ == "__main__":
    main()
