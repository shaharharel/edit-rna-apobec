#!/usr/bin/env python3
"""Test simple baselines vs our combined predictor on Lei BE4max off-targets.

If a simple sequence-composition feature (GC content, G-skew, etc.) gives
OR >= 3 at p70-p90, our complicated combined_positive is just a noisy
proxy for accessibility/composition.

Baselines tested:
  - gc_201:        fraction G+C in full 201-nt window
  - gc_local:      fraction G+C in ±10 nt around target C
  - gc_protospacer: fraction G+C in ±20 nt (typical Cas9 footprint)
  - g_skew:        (G - C) / (G + C) over 201 nt — R-loop driver
  - g_skew_local:  same in ±20 nt
  - max_g_run:     longest consecutive G-run in 201 nt
  - at_content:    fraction A+T (inverse of GC)
  - random:        random scores (sanity check, should be OR ≈ 1)
"""
from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

ROOT = Path("/Users/shaharharel/Documents/github/edit-rna-apobec")
IN_CSV = ROOT / "experiments/base_editor_offtargets/outputs/lei_combined_anti/scored_with_combined.csv"
OUT = ROOT / "experiments/base_editor_offtargets/outputs/lei_simple_baselines"
OUT.mkdir(parents=True, exist_ok=True)

CENTER = 100
PERCENTILES = [70, 75, 80, 85, 90, 95]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("baselines")


def gc_fraction(seq):
    """Fraction G+C in the entire sequence."""
    seq = seq.upper()
    return (seq.count("G") + seq.count("C")) / len(seq)


def gc_window(seq, half_width):
    """Fraction G+C in ±half_width around CENTER."""
    s = seq[max(0, CENTER - half_width):min(len(seq), CENTER + half_width + 1)]
    if not s: return 0.0
    return (s.count("G") + s.count("C")) / len(s)


def g_skew(seq, half_width=None):
    """(G - C) / (G + C); high G-skew → R-loop forming on opposite strand."""
    if half_width:
        s = seq[max(0, CENTER - half_width):min(len(seq), CENTER + half_width + 1)]
    else:
        s = seq
    g = s.upper().count("G"); c = s.upper().count("C")
    if g + c == 0: return 0.0
    return (g - c) / (g + c)


def max_run(seq, base):
    """Longest consecutive run of base in the entire sequence."""
    seq = seq.upper()
    cur = best = 0
    for ch in seq:
        if ch == base:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


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
    log.info("Loading scored sites with sequences...")
    df = pd.read_csv(IN_CSV, low_memory=False)
    if "valid_recomp" in df.columns:
        df = df[df["valid_recomp"]].reset_index(drop=True)
    df = df[df["seq"].notna()].reset_index(drop=True)
    log.info("Rows with seq: %d", len(df))

    log.info("Computing simple baselines...")
    df["base_gc_201"] = df["seq"].apply(gc_fraction)
    df["base_gc_local"] = df["seq"].apply(lambda s: gc_window(s, 10))
    df["base_gc_proto"] = df["seq"].apply(lambda s: gc_window(s, 20))
    df["base_gc_50"] = df["seq"].apply(lambda s: gc_window(s, 50))
    df["base_g_skew"] = df["seq"].apply(g_skew)
    df["base_g_skew_proto"] = df["seq"].apply(lambda s: g_skew(s, 20))
    df["base_max_g_run"] = df["seq"].apply(lambda s: max_run(s, "G"))
    df["base_max_c_run"] = df["seq"].apply(lambda s: max_run(s, "C"))
    df["base_at_content"] = 1 - df["base_gc_201"]
    rng = np.random.default_rng(42)
    df["base_random"] = rng.random(len(df))

    df.to_csv(OUT / "scored_with_baselines.csv", index=False)
    log.info("Wrote %s", OUT / "scored_with_baselines.csv")

    # Compute enrichment for each baseline + our complex predictors, both control sets
    rows = []
    HEADS = [
        # Simple baselines:
        "base_gc_201", "base_gc_local", "base_gc_proto", "base_gc_50",
        "base_g_skew", "base_g_skew_proto",
        "base_max_g_run", "base_max_c_run", "base_at_content", "base_random",
        # Our model predictors for comparison:
        "score_phase3_A3G", "score_combined_positive", "score_combined_anti",
    ]
    for ctrl_lbl, ctrl_name in [("control_local", "local"), ("control_global", "global")]:
        is_pos = (df["label"] == "positive").values
        is_ctrl = (df["label"] == ctrl_lbl).values
        keep = is_pos | is_ctrl
        ip = is_pos[keep]
        n_pos = int(ip.sum()); n_ctrl = int((~ip).sum())
        log.info("Control set %s: pos=%d ctrl=%d", ctrl_name, n_pos, n_ctrl)
        for head in HEADS:
            sc = df.loc[keep, head].values
            for pct in PERCENTILES:
                thr = float(np.percentile(sc, pct))
                or_v, p, pa, ca = fisher_or(sc, ip, thr)
                rows.append({
                    "ctrl_set": ctrl_name, "head": head,
                    "percentile": pct, "thr": thr,
                    "n_pos": n_pos, "n_ctrl": n_ctrl,
                    "pos_above": pa, "ctrl_above": ca,
                    "or": or_v, "p_value": p,
                })
    enrich = pd.DataFrame(rows)
    enrich["q_value"] = multipletests(enrich["p_value"].fillna(1.0), method="fdr_bh")[1]
    enrich.to_csv(OUT / "enrichment_baselines.csv", index=False)

    # Print headline tables
    print("\n" + "=" * 100)
    print("Simple-baseline OR at p90 — does any sequence-composition feature beat our model?")
    print("=" * 100)
    for cs in ["local", "global"]:
        print(f"\n--- vs {cs} controls ---")
        sub = enrich[(enrich["ctrl_set"] == cs) & (enrich["percentile"] == 90)].sort_values("or", ascending=False)
        print(sub[["head", "or", "p_value", "q_value", "pos_above", "ctrl_above"]].to_string(index=False, float_format="%.3f"))

    print("\n" + "=" * 100)
    print("Full percentile sweep — TOP candidate baselines + our combined_positive (vs global)")
    print("=" * 100)
    KEY_HEADS = ["score_combined_positive", "score_phase3_A3G",
                  "base_gc_201", "base_gc_proto", "base_gc_local",
                  "base_g_skew_proto", "base_g_skew", "base_max_g_run",
                  "base_at_content", "base_random"]
    sub = enrich[(enrich["ctrl_set"] == "global") & (enrich["head"].isin(KEY_HEADS))]
    piv = sub.pivot_table(index="head", columns="percentile", values="or", aggfunc="first")
    piv_q = sub.pivot_table(index="head", columns="percentile", values="q_value", aggfunc="first")
    out_str = piv.copy().astype(object)
    for h in piv.index:
        for pct in piv.columns:
            or_v = piv.loc[h, pct]; q = piv_q.loc[h, pct]
            if pd.isna(or_v): out_str.loc[h, pct] = "—"
            else: out_str.loc[h, pct] = f"{or_v:.2f}(q={q:.0e})"
    # Order: combined_positive first, baselines after
    out_str = out_str.reindex([h for h in KEY_HEADS if h in out_str.index])
    print(out_str.to_string())

    # Also: does GC content correlate with combined_positive score?
    print("\n" + "=" * 100)
    print("Correlations: combined_positive vs simple baselines (Spearman)")
    print("=" * 100)
    from scipy.stats import spearmanr
    for h in ["base_gc_201", "base_gc_proto", "base_gc_local", "base_g_skew_proto",
               "base_max_g_run", "base_at_content"]:
        r, p = spearmanr(df["score_combined_positive"], df[h])
        print(f"  combined_positive vs {h}: r={r:.3f}  (p={p:.0e})")
    print()
    for h in ["base_gc_201", "base_gc_proto", "base_gc_local", "base_g_skew_proto"]:
        r, p = spearmanr(df["score_phase3_A3G"], df[h])
        print(f"  phase3_A3G vs {h}: r={r:.3f}  (p={p:.0e})")


if __name__ == "__main__":
    main()
