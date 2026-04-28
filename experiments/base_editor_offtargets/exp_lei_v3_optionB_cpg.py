#!/usr/bin/env python3
"""Lei v3 — Option B (TC-only) + CpG-stratified analysis.

Reuses v2's scored_sites.csv (no recomputation). Filters to:
- TC-context positives + their TC-matched controls (Option B)
- Non-CpG TC: TCA / TCT / TCC (excludes TCG = potential m5C deamination confound)

Outputs enrichment tables stratified by these subsets.
"""
from __future__ import annotations
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

ROOT = Path("/Users/shaharharel/Documents/github/edit-rna-apobec")
V2_OUT = ROOT / "experiments/base_editor_offtargets/outputs/lei_preview_v2"
OUT = ROOT / "experiments/base_editor_offtargets/outputs/lei_v3_optionB_cpg"
OUT.mkdir(parents=True, exist_ok=True)

PERCENTILES = [90, 95, 99]
N_BOOTSTRAP = 1000
SEED = 42

HEADS = ["phase3_binary", "phase3_A3A", "phase3_A3B", "phase3_A3G",
         "phase3_A3A_A3G", "phase3_Neither", "apobec1_sp1", "apobec1_sp0",
         "motif_only"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("v3")


def bootstrap_or(scores, is_pos, thr, n_boot=N_BOOTSTRAP, rng=None):
    if rng is None: rng = np.random.default_rng(SEED)
    n_pos = is_pos.sum(); n_ctrl = (~is_pos).sum()
    pos_idx = np.where(is_pos)[0]; ctrl_idx = np.where(~is_pos)[0]
    if len(pos_idx) == 0 or len(ctrl_idx) == 0:
        return np.nan, np.nan
    ors = []
    for _ in range(n_boot):
        pi = rng.choice(pos_idx, n_pos, replace=True)
        ci = rng.choice(ctrl_idx, n_ctrl, replace=True)
        pa = (scores[pi] >= thr).sum(); pb = n_pos - pa
        ca = (scores[ci] >= thr).sum(); cb = n_ctrl - ca
        if pb > 0 and ca > 0:
            ors.append((pa * cb) / (pb * ca))
    if not ors: return np.nan, np.nan
    return tuple(np.percentile(ors, [2.5, 97.5]))


def enrich(df, control_label, name):
    is_pos = (df["label"] == "positive").values
    is_ctrl = (df["label"] == control_label).values
    keep = is_pos | is_ctrl
    sub = df[keep].reset_index(drop=True)
    is_pos_k = (sub["label"] == "positive").values
    rows = []
    for head in HEADS:
        col = f"score_{head}"
        if col not in sub.columns:
            continue
        sc = sub[col].values
        for pct in PERCENTILES:
            thr = float(np.percentile(sc, pct))
            above = sc >= thr
            pa = int((is_pos_k & above).sum())
            pb = int((is_pos_k & ~above).sum())
            ca = int((~is_pos_k & above).sum())
            cb = int((~is_pos_k & ~above).sum())
            try:
                or_val = (pa * cb) / (pb * ca) if pb > 0 and ca > 0 else (
                    float("inf") if pb == 0 else 0.0)
            except ZeroDivisionError:
                or_val = float("nan")
            _, p = fisher_exact([[pa, pb], [ca, cb]])
            ci_lo, ci_hi = bootstrap_or(sc, is_pos_k, thr)
            rows.append({
                "control_set": name, "head": head, "percentile": pct,
                "threshold": thr,
                "n_pos": int(is_pos_k.sum()), "n_ctrl": int((~is_pos_k).sum()),
                "n_pos_above": pa, "n_pos_below": pb,
                "n_ctrl_above": ca, "n_ctrl_below": cb,
                "or": or_val, "p_value": float(p),
                "or_ci_lo": float(ci_lo), "or_ci_hi": float(ci_hi),
            })
    out = pd.DataFrame(rows)
    out["q_value"] = multipletests(out["p_value"].fillna(1.0), method="fdr_bh")[1]
    return out


def main():
    log.info("Loading v2 scored sites...")
    df = pd.read_csv(V2_OUT / "scored_sites.csv")
    log.info("v2 total: %d (valid=%d)", len(df), df["valid"].sum())
    df = df[df["valid"]].reset_index(drop=True)

    # trinuc column should already be there
    if "trinuc" not in df.columns:
        log.error("trinuc column missing — recompute from seq required")
        return

    # ----- Stratum 1: ALL TC (Option B) -----
    tc_mask = df["trinuc"].str[0].isin(["T", "U"])
    df_tc = df[tc_mask].reset_index(drop=True)
    log.info("TC-only subset: %d (%d pos, %d local, %d global)",
             len(df_tc),
             (df_tc["label"] == "positive").sum(),
             (df_tc["label"] == "control_local").sum(),
             (df_tc["label"] == "control_global").sum())

    # ----- Stratum 2: TC + non-CpG -----
    non_cpg_mask = tc_mask & (df["trinuc"].str[2] != "G")
    df_tc_noncpg = df[non_cpg_mask].reset_index(drop=True)
    log.info("TC + non-CpG subset: %d", len(df_tc_noncpg))

    # ----- Stratum 3: TCG only (CpG, for comparison) -----
    tcg_mask = tc_mask & (df["trinuc"].str[2] == "G")
    df_tcg = df[tcg_mask].reset_index(drop=True)
    log.info("TCG (CpG) subset: %d", len(df_tcg))

    # Trinuc breakdown of positives
    trinuc_breakdown = df[df["label"] == "positive"]["trinuc"].value_counts().head(15)
    log.info("Top-15 trinuc in positives:\n%s", trinuc_breakdown.to_string())

    # Compute enrichment for each stratum
    results = []
    for name, subset in [
        ("TC_all_local", (df_tc, "control_local")),
        ("TC_all_global", (df_tc, "control_global")),
        ("TC_nonCpG_local", (df_tc_noncpg, "control_local")),
        ("TC_nonCpG_global", (df_tc_noncpg, "control_global")),
        ("TCG_only_local", (df_tcg, "control_local")),
        ("TCG_only_global", (df_tcg, "control_global")),
    ]:
        d, ctrl = subset
        if len(d) < 50:
            log.warning("Skipping %s — too few sites (%d)", name, len(d))
            continue
        log.info("Computing enrichment for %s...", name)
        e = enrich(d, ctrl, name)
        results.append(e)

    if results:
        all_e = pd.concat(results, ignore_index=True)
        all_e["q_value"] = multipletests(all_e["p_value"].fillna(1.0), method="fdr_bh")[1]
        all_e.to_csv(OUT / "enrichment_stratified.csv", index=False)
        log.info("Wrote %s", OUT / "enrichment_stratified.csv")

        # Print top results
        for s in all_e["control_set"].unique():
            print("\n" + "=" * 70)
            print(f"Stratum: {s}")
            print("=" * 70)
            tab = all_e[(all_e["control_set"] == s) & (all_e["percentile"] == 90)]
            tab = tab.sort_values("or", ascending=False)
            print(tab[["head", "n_pos", "n_ctrl", "or", "or_ci_lo", "or_ci_hi",
                       "p_value", "q_value"]].to_string(index=False))

    summary = {
        "n_v2_total_valid": int(len(df)),
        "n_TC_total": int(len(df_tc)),
        "n_TC_pos": int((df_tc["label"] == "positive").sum()),
        "n_TC_nonCpG_total": int(len(df_tc_noncpg)),
        "n_TC_nonCpG_pos": int((df_tc_noncpg["label"] == "positive").sum()),
        "n_TCG_total": int(len(df_tcg)),
        "n_TCG_pos": int((df_tcg["label"] == "positive").sum()),
        "top_trinucs_in_positives": trinuc_breakdown.to_dict(),
    }
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)


if __name__ == "__main__":
    main()
