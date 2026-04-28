#!/usr/bin/env python3
"""Position-level trinucleotide diagnostic for v3 vs v4_cancer vs v4_cds.

For each model, takes the top-1% positions by score_binary and tabulates the
strand-corrected trinucleotide context. Compares to the panel's overall
distribution. Confirms visually whether the anti-TCW polarity (top positions
skewed toward CpG / away from TCW) seen in v3 is gone in v4.

Outputs:
  experiments/multi_enzyme/outputs/pcawg_tcw_panel/v4_outputs/topx_trinuc_breakdown.png
  experiments/multi_enzyme/outputs/pcawg_tcw_panel/v4_outputs/topx_trinuc_breakdown.csv
"""
from __future__ import annotations
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/shaharharel/Documents/github/edit-rna-apobec")
HG19 = ROOT / "data/raw/genomes/hg19.fa"
OUT_DIR = ROOT / "experiments/multi_enzyme/outputs/pcawg_tcw_panel/v4_outputs"

V3_PANEL = ROOT / "experiments/multi_enzyme/outputs/pcawg_tcw_panel/panel_scores_cds.parquet"
V4_CANCER = OUT_DIR / "panel_scores_v4_cancer.parquet"
V4_CDS = OUT_DIR / "panel_scores_v4_cds.parquet"

TOP_PCT = 0.01

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    stream=sys.stdout)
log = logging.getLogger(__name__)


def trinuc_strand_corrected(panel: pd.DataFrame) -> np.ndarray:
    """For each panel row, return the strand-corrected trinucleotide as a
    3-character string. + strand: seq[pos-1:pos+2] uppercase. - strand:
    reverse-complement of that."""
    from pyfaidx import Fasta
    log.info("Computing strand-corrected trinucleotides for %d positions ...",
             len(panel))
    genome = Fasta(str(HG19), as_raw=False, sequence_always_upper=True)
    n = len(panel)
    trinuc = np.full(n, "NNN", dtype="<U3")

    chroms = panel["chrom"].to_numpy()
    poses = panel["pos"].astype(int).to_numpy()
    strands = panel["strand"].to_numpy()
    idx_all = np.arange(n)

    rc_map = str.maketrans("ACGTN", "TGCAN")

    for ch in pd.Series(chroms).unique():
        mask = chroms == ch
        idx = idx_all[mask]
        if len(idx) == 0:
            continue
        try:
            seq = str(genome[ch][:]).upper()
        except Exception:
            continue
        L = len(seq)
        ps = poses[mask]
        ss = strands[mask]
        for i, (p, s) in zip(idx, zip(ps, ss)):
            if p < 1 or p + 1 >= L:
                continue
            tn = seq[p - 1:p + 2]
            if s == "-":
                tn = tn.translate(rc_map)[::-1]
            trinuc[i] = tn
    return trinuc


def categorize_trinuc(tn: str) -> str:
    """Bucket a trinucleotide into one of: TCW (TCA, TCT), TCN_other (TCC, TCG),
    NCG (CpG, non-TC), other_C, or non-C."""
    if len(tn) != 3 or tn[1] != "C":
        return "non-C"
    left, right = tn[0], tn[2]
    if left == "T" and right in ("A", "T"):
        return "TCW"
    if left == "T" and right == "G":
        return "TCG (CpG)"
    if left == "T" and right == "C":
        return "TCC"
    if right == "G":
        return "NCG (non-TC CpG)"
    return "other_C"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Loading panels ...")
    v3 = pd.read_parquet(V3_PANEL, columns=["chrom", "pos", "strand", "score_binary"])
    v4c = pd.read_parquet(V4_CANCER, columns=["chrom", "pos", "strand", "score_binary"])
    v4d = pd.read_parquet(V4_CDS, columns=["chrom", "pos", "strand", "score_binary"])
    assert (v3["chrom"].values == v4c["chrom"].values).all()
    assert (v3["pos"].values == v4c["pos"].values).all()
    assert (v3["chrom"].values == v4d["chrom"].values).all()
    assert (v3["pos"].values == v4d["pos"].values).all()
    log.info("All three panels share row order. n=%d", len(v3))

    # Compute trinuc once on shared layout.
    trinuc = trinuc_strand_corrected(v3[["chrom", "pos", "strand"]])
    cats = np.array([categorize_trinuc(t) for t in trinuc], dtype="<U16")
    log.info("Trinuc categories overall: %s",
             dict(zip(*np.unique(cats, return_counts=True))))

    n = len(v3)
    k = max(1, int(round(n * TOP_PCT)))
    log.info("Top-1%% k = %d", k)

    panels = {
        "v3": v3["score_binary"].to_numpy(),
        "v4_cancer": v4c["score_binary"].to_numpy(),
        "v4_cds": v4d["score_binary"].to_numpy(),
    }

    cat_order = ["TCW", "TCG (CpG)", "TCC", "NCG (non-TC CpG)", "other_C", "non-C"]
    rows = []
    # Overall.
    overall = pd.Series(cats).value_counts(normalize=True).to_dict()
    for c in cat_order:
        rows.append({"model": "overall_panel", "category": c,
                     "frac": overall.get(c, 0.0),
                     "count": int(np.sum(cats == c))})

    top_cats_per_model = {}
    for model, scores in panels.items():
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_c = cats[top_idx]
        top_cats_per_model[model] = top_c
        cnt = pd.Series(top_c).value_counts(normalize=True).to_dict()
        cnt_abs = pd.Series(top_c).value_counts().to_dict()
        for c in cat_order:
            rows.append({"model": f"{model}_top1pct", "category": c,
                         "frac": cnt.get(c, 0.0),
                         "count": int(cnt_abs.get(c, 0))})

    df = pd.DataFrame(rows)
    out_csv = OUT_DIR / "topx_trinuc_breakdown.csv"
    df.to_csv(out_csv, index=False)
    log.info("Wrote %s", out_csv)

    # Plot.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))
    width = 0.18
    x = np.arange(len(cat_order))
    series_order = ["overall_panel", "v3_top1pct", "v4_cancer_top1pct", "v4_cds_top1pct"]
    colors = {"overall_panel": "#888", "v3_top1pct": "#d62728",
              "v4_cancer_top1pct": "#2ca02c", "v4_cds_top1pct": "#1f77b4"}
    for i, s in enumerate(series_order):
        sub = df[df["model"] == s].set_index("category").reindex(cat_order)["frac"].values
        ax.bar(x + (i - 1.5) * width, sub * 100, width=width,
               label=s, color=colors[s])
    ax.set_xticks(x)
    ax.set_xticklabels(cat_order, rotation=20)
    ax.set_ylabel("Fraction of panel positions (%)")
    ax.set_title("Top-1% trinucleotide breakdown by model (v3 vs v4_cancer vs v4_cds)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out_png = OUT_DIR / "topx_trinuc_breakdown.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Wrote %s", out_png)

    # Sanity: print TCW vs CpG fraction comparison
    log.info("TCW fraction: overall=%.3f, v3_top1=%.3f, v4_cancer_top1=%.3f, v4_cds_top1=%.3f",
             overall.get("TCW", 0),
             (top_cats_per_model["v3"] == "TCW").mean(),
             (top_cats_per_model["v4_cancer"] == "TCW").mean(),
             (top_cats_per_model["v4_cds"] == "TCW").mean())
    log.info("CpG (NCG+TCG) fraction: overall=%.3f, v3_top1=%.3f, v4_cancer_top1=%.3f, v4_cds_top1=%.3f",
             overall.get("TCG (CpG)", 0) + overall.get("NCG (non-TC CpG)", 0),
             ((top_cats_per_model["v3"] == "TCG (CpG)") |
              (top_cats_per_model["v3"] == "NCG (non-TC CpG)")).mean(),
             ((top_cats_per_model["v4_cancer"] == "TCG (CpG)") |
              (top_cats_per_model["v4_cancer"] == "NCG (non-TC CpG)")).mean(),
             ((top_cats_per_model["v4_cds"] == "TCG (CpG)") |
              (top_cats_per_model["v4_cds"] == "NCG (non-TC CpG)")).mean())
    return 0


if __name__ == "__main__":
    sys.exit(main())
