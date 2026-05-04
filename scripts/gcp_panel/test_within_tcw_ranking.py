#!/usr/bin/env python3
"""Within-TCW ranking test for v4.

Hypothesis: when ranking is restricted to TCW positions only (so the panel and
the TCW-density baseline pick from the same pool), does v4's per-position score
beat random-within-TCW selection?

If yes: v4 has structural / context preference within TCW context, beyond what
        TCW-motif identity alone provides.
If no:  v4 adds nothing once you condition on TCW.

Setup:
- Panel: panel_scores_v4_cds_apobec1retrained.parquet (8.45 M positions)
- TCW = trinucleotide TCA or TCT (excludes CpG by definition; W = A or T)
- Restrict panel to TCW positions; compute the top-1%/5%/10% slice within that pool
- Mutation set: TCGA-MC3 + PCAWG-coding pan-cancer C>T mutations, filtered to
  TCW_nonCpG and to in-panel positions
- Baseline: random selection of k positions from the TCW pool, 1000 draws
- Heads tested: score_binary, score_A3A, score_A3B, score_A3A_A3G, score_apobec1_v4_cds

Output:
- experiments/multi_enzyme/outputs/pcawg_tcw_panel/v4_outputs/within_tcw_test.csv
- experiments/multi_enzyme/outputs/pcawg_tcw_panel/v4_outputs/WITHIN_TCW_RESULTS.md
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/shaharharel/Documents/github/edit-rna-apobec")
PANEL = ROOT / "experiments/multi_enzyme/outputs/pcawg_tcw_panel/v4_outputs/panel_scores_v4_cds_apobec1retrained.parquet"
HG19 = ROOT / "data/raw/genomes/hg19.fa"
TCGA_DIR = ROOT / "data/raw/tcga"
PCAWG_DIR = ROOT / "data/raw/pcawg/by_cancer"
OUT_DIR = ROOT / "experiments/multi_enzyme/outputs/pcawg_tcw_panel/v4_outputs"
OUT_CSV = OUT_DIR / "within_tcw_test.csv"
OUT_MD = OUT_DIR / "WITHIN_TCW_RESULTS.md"

CANCERS = ["blca", "brca", "cesc", "coadread", "esca", "hnsc", "lihc",
           "lusc", "skcm", "stad"]
HEADS = ["score_binary", "score_A3A", "score_A3B", "score_A3A_A3G",
         "score_apobec1_v4_cds"]
TOP_PCTS = [0.01, 0.05, 0.10]
N_RANDOM_DRAWS = 1000
N_BOOT = 10000
SEED = 20260427

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    stream=sys.stdout)
log = logging.getLogger(__name__)


def annotate_tcw(panel: pd.DataFrame) -> pd.DataFrame:
    from pyfaidx import Fasta
    log.info("Annotating panel with is_TCW_C ...")
    genome = Fasta(str(HG19), as_raw=False, sequence_always_upper=True)
    n = len(panel)
    is_tcw_c = np.zeros(n, dtype=bool)
    chroms = panel["chrom"].to_numpy()
    poses = panel["pos"].astype(int).to_numpy()
    strands = panel["strand"].to_numpy()
    idx = np.arange(n)
    for ch in pd.Series(chroms).unique():
        mask = chroms == ch
        i_ch = idx[mask]
        if len(i_ch) == 0:
            continue
        try:
            seq = np.frombuffer(str(genome[ch][:]).upper().encode("ascii"),
                                dtype=np.uint8)
        except Exception:
            continue
        L = len(seq)
        ps = poses[mask]
        ss = strands[mask]
        ok = (ps >= 1) & (ps + 1 < L)
        valid = i_ch[ok]
        ps_ok = ps[ok]
        ss_ok = ss[ok]
        left = seq[ps_ok - 1]
        right = seq[ps_ok + 1]
        is_plus = ss_ok == "+"
        is_minus = ~is_plus
        right_AT = (right == ord("A")) | (right == ord("T"))
        left_AT = (left == ord("A")) | (left == ord("T"))
        is_tcw_c[valid[is_plus]] = (left[is_plus] == ord("T")) & right_AT[is_plus]
        is_tcw_c[valid[is_minus]] = (right[is_minus] == ord("A")) & left_AT[is_minus]
    panel = panel.copy()
    panel["is_TCW_C"] = is_tcw_c
    log.info("  TCW positions: %d (%.2f%%)", is_tcw_c.sum(), 100 * is_tcw_c.mean())
    return panel


def _load_one_maf(path: Path, cancer: str, source: str):
    if not path.exists():
        return None
    df = pd.read_csv(path, sep="\t", low_memory=False)
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
        "+", "-")
    df["pos"] = df["Start_Position"].astype(int) - 1
    df["chrom"] = df["Chromosome"].astype(str)
    df.loc[~df["chrom"].str.startswith("chr"), "chrom"] = "chr" + df["chrom"]
    df["cancer"] = cancer
    return df[["chrom", "pos", "strand", "cancer"]]


def load_mutations() -> pd.DataFrame:
    log.info("Loading TCGA + PCAWG-coding MAFs ...")
    rows = []
    for cancer in CANCERS:
        d = _load_one_maf(PCAWG_DIR / f"{cancer}_pcawg_mutations.txt", cancer, "pcawg")
        if d is not None:
            rows.append(d)
        d = _load_one_maf(TCGA_DIR / f"{cancer}_tcga_pan_can_atlas_2018_mutations.txt",
                          cancer, "tcga")
        if d is not None:
            rows.append(d)
    df = pd.concat(rows, ignore_index=True)
    valid = set([f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"])
    df = df[df["chrom"].isin(valid)]
    df = df.drop_duplicates(["chrom", "pos", "strand", "cancer"])
    log.info("  %d unique cancer C>T sites", len(df))
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    t0 = time.time()
    log.info("Loading panel %s", PANEL)
    panel = pd.read_parquet(PANEL)
    log.info("  panel n=%d", len(panel))

    panel = annotate_tcw(panel)
    panel_idx = np.arange(len(panel))
    panel["panel_idx"] = panel_idx
    tcw_mask = panel["is_TCW_C"].to_numpy()
    tcw_idx = panel_idx[tcw_mask]
    n_tcw = len(tcw_idx)
    log.info("  TCW pool size: %d", n_tcw)

    # Mutations
    muts = load_mutations()
    panel_set = set(zip(panel["chrom"].astype(str), panel["pos"].astype(int)))
    in_panel = np.array([(c, int(p)) in panel_set
                         for c, p in zip(muts["chrom"], muts["pos"])])
    muts = muts.iloc[np.where(in_panel)[0]].reset_index(drop=True)
    log.info("  mutations in panel: %d", len(muts))

    # Map mutations to panel index
    pos_to_idx = {(c, int(p)): i for i, (c, p) in enumerate(
        zip(panel["chrom"].astype(str), panel["pos"].astype(int)))}
    muts["panel_idx"] = [pos_to_idx.get((c, int(p)), -1)
                         for c, p in zip(muts["chrom"], muts["pos"])]
    muts = muts[muts["panel_idx"] >= 0].copy()
    muts["is_tcw"] = panel["is_TCW_C"].to_numpy()[muts["panel_idx"].to_numpy()]
    log.info("  mutations on TCW positions: %d (%.2f%% of in-panel)",
             muts["is_tcw"].sum(), 100 * muts["is_tcw"].mean())

    # Filter mutations to TCW (= filter_TCW_nonCpG since W excludes G)
    muts_tcw = muts[muts["is_tcw"]].copy()
    log.info("  TCW_nonCpG mutations (= TCW since W=A/T): %d", len(muts_tcw))

    # ---------- per-cancer top-X within TCW ----------
    rows = []
    for top_pct in TOP_PCTS:
        # Relative to FULL panel size for comparability with main sweep
        k_full = int(round(len(panel) * top_pct))
        # Restricted to TCW: how many positions = same fraction of TCW pool
        k_tcw_relative = int(round(n_tcw * top_pct))
        log.info("=== top-%.0f%%: k_full=%d, k_tcw=%d ===",
                 top_pct * 100, k_full, k_tcw_relative)

        for k_label, k in [("k_of_full_panel", k_full),
                           ("k_of_tcw_pool", k_tcw_relative)]:
            if k > n_tcw:
                log.info("  [%s] k=%d exceeds TCW pool n=%d, skip", k_label, k, n_tcw)
                continue
            log.info("  [%s] k=%d (= %.2f%% of TCW pool)",
                     k_label, k, 100 * k / n_tcw)

            # For each head, rank within TCW
            head_top_sets = {}
            for head in HEADS:
                if head not in panel.columns:
                    continue
                tcw_scores = panel[head].to_numpy()[tcw_idx]
                top_local = np.argpartition(-tcw_scores, k - 1)[:k]
                top_panel_idx = tcw_idx[top_local]
                head_top_sets[head] = set(top_panel_idx.tolist())

            # Random baselines: 1000 draws of k from TCW pool — vectorised
            # Pre-extract per-cancer mutation panel_idx arrays
            cancer_mut_idx = {}
            for c in CANCERS:
                sub = muts_tcw[muts_tcw["cancer"] == c]
                if len(sub) > 0:
                    cancer_mut_idx[c] = sub["panel_idx"].to_numpy()
            all_mut_idx = muts_tcw["panel_idx"].to_numpy()
            n_panel = len(panel)
            random_recalls = {c: np.zeros(N_RANDOM_DRAWS) for c in cancer_mut_idx}
            random_recalls["all"] = np.zeros(N_RANDOM_DRAWS)
            for d in range(N_RANDOM_DRAWS):
                draw = rng.choice(tcw_idx, size=k, replace=False)
                # Boolean indicator over panel — vectorised hit-test
                hit = np.zeros(n_panel, dtype=bool)
                hit[draw] = True
                for c, mids in cancer_mut_idx.items():
                    random_recalls[c][d] = hit[mids].sum() / len(mids)
                random_recalls["all"][d] = hit[all_mut_idx].sum() / max(len(all_mut_idx), 1)
                if d % 100 == 0:
                    log.info("    random-draw %d/%d", d, N_RANDOM_DRAWS)

            # Per-head per-cancer recall — vectorised against indicator
            for head, top_set in head_top_sets.items():
                top_arr = np.fromiter(top_set, dtype=np.int64, count=len(top_set))
                head_hit = np.zeros(n_panel, dtype=bool)
                head_hit[top_arr] = True
                for c, mids in cancer_mut_idx.items():
                    rb_arr = random_recalls[c]
                    rb_mean = rb_arr.mean()
                    h_recall = head_hit[mids].sum() / len(mids)
                    rows.append({
                        "top_pct": top_pct, "k_kind": k_label, "k": k,
                        "head": head, "cancer": c, "n_mutations": int(len(mids)),
                        "head_recall": h_recall,
                        "random_recall_mean": rb_mean,
                        "random_recall_ci_lo": np.percentile(rb_arr, 2.5),
                        "random_recall_ci_hi": np.percentile(rb_arr, 97.5),
                        "ratio_vs_random_within_tcw": h_recall / max(rb_mean, 1e-9),
                    })
                rb_arr = random_recalls["all"]
                rb_mean = rb_arr.mean()
                h_recall = head_hit[all_mut_idx].sum() / max(len(all_mut_idx), 1)
                rows.append({
                    "top_pct": top_pct, "k_kind": k_label, "k": k,
                    "head": head, "cancer": "all",
                    "n_mutations": int(len(all_mut_idx)),
                    "head_recall": h_recall,
                    "random_recall_mean": rb_mean,
                    "random_recall_ci_lo": np.percentile(rb_arr, 2.5),
                    "random_recall_ci_hi": np.percentile(rb_arr, 97.5),
                    "ratio_vs_random_within_tcw": h_recall / max(rb_mean, 1e-9),
                })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    log.info("Wrote %s (%d rows)", OUT_CSV, len(df))

    # ---------- bootstrap CI across cancers (per head, per cut) ----------
    log.info("Bootstrap CI across 10 cancers ...")
    boot_rows = []
    for top_pct in TOP_PCTS:
        for k_kind in ("k_of_full_panel", "k_of_tcw_pool"):
            for head in HEADS:
                sub = df[(df["top_pct"] == top_pct)
                         & (df["k_kind"] == k_kind)
                         & (df["head"] == head)
                         & (df["cancer"].isin(CANCERS))]
                if len(sub) == 0:
                    continue
                ratios = sub["ratio_vs_random_within_tcw"].to_numpy()
                if len(ratios) < 2:
                    continue
                boot = []
                for _ in range(N_BOOT):
                    s = rng.choice(ratios, size=len(ratios), replace=True)
                    boot.append(s.mean())
                boot = np.asarray(boot)
                boot_rows.append({
                    "top_pct": top_pct,
                    "k_kind": k_kind,
                    "head": head,
                    "n_cancers": len(ratios),
                    "mean_ratio": ratios.mean(),
                    "median_ratio": np.median(ratios),
                    "ci_lo": np.percentile(boot, 2.5),
                    "ci_hi": np.percentile(boot, 97.5),
                    "n_above_1": int((ratios > 1).sum()),
                })
    boot_df = pd.DataFrame(boot_rows)
    boot_df.to_csv(OUT_DIR / "within_tcw_test_summary.csv", index=False)
    log.info("Wrote %s", OUT_DIR / "within_tcw_test_summary.csv")

    # ---------- markdown summary ----------
    lines = []
    lines.append("# Within-TCW ranking test — v4 (PCAWG/TCGA, 10 cancers)\n")
    lines.append(
        "Question: when we restrict ranking to TCW positions only, does v4's "
        "per-position score beat random-within-TCW selection?\n\n"
        "If `ratio_vs_random_within_tcw > 1` with CI lower bound > 1, the model "
        "has within-TCW structural/context preference that outperforms a uniform "
        "TCW-density baseline (= the QA limitation we're testing).\n\n"
    )
    lines.append(f"- TCW pool size: **{n_tcw:,}** ({100*n_tcw/len(panel):.2f}% of panel)\n")
    lines.append(f"- TCW_nonCpG mutations in panel: **{len(muts_tcw):,}**\n")
    lines.append(f"- Random draws per cell: {N_RANDOM_DRAWS}\n")
    lines.append(f"- Bootstrap resamples across cancers: {N_BOOT}\n\n")

    lines.append("## Mean ratio vs random-within-TCW (10 cancers, k = top-X% of full panel)\n")
    pivot1 = boot_df[boot_df["k_kind"] == "k_of_full_panel"].pivot_table(
        index="head", columns="top_pct",
        values="mean_ratio", aggfunc="first")
    cilo = boot_df[boot_df["k_kind"] == "k_of_full_panel"].pivot_table(
        index="head", columns="top_pct",
        values="ci_lo", aggfunc="first")
    cihi = boot_df[boot_df["k_kind"] == "k_of_full_panel"].pivot_table(
        index="head", columns="top_pct",
        values="ci_hi", aggfunc="first")
    lines.append("| Head | top-1% | top-5% | top-10% |\n")
    lines.append("|---|---|---|---|\n")
    for h in HEADS:
        if h not in pivot1.index:
            continue
        line = f"| `{h}` |"
        for tp in TOP_PCTS:
            m = pivot1.loc[h, tp] if tp in pivot1.columns else None
            lo = cilo.loc[h, tp] if tp in cilo.columns else None
            hi = cihi.loc[h, tp] if tp in cihi.columns else None
            if m is None or pd.isna(m):
                line += " — |"
            else:
                line += f" {m:.2f} [{lo:.2f}, {hi:.2f}] |"
        lines.append(line + "\n")
    lines.append("\n")

    lines.append("## Mean ratio vs random-within-TCW (10 cancers, k = top-X% of TCW pool)\n")
    pivot2 = boot_df[boot_df["k_kind"] == "k_of_tcw_pool"].pivot_table(
        index="head", columns="top_pct",
        values="mean_ratio", aggfunc="first")
    cilo2 = boot_df[boot_df["k_kind"] == "k_of_tcw_pool"].pivot_table(
        index="head", columns="top_pct",
        values="ci_lo", aggfunc="first")
    cihi2 = boot_df[boot_df["k_kind"] == "k_of_tcw_pool"].pivot_table(
        index="head", columns="top_pct",
        values="ci_hi", aggfunc="first")
    lines.append("| Head | top-1% | top-5% | top-10% |\n")
    lines.append("|---|---|---|---|\n")
    for h in HEADS:
        if h not in pivot2.index:
            continue
        line = f"| `{h}` |"
        for tp in TOP_PCTS:
            m = pivot2.loc[h, tp] if tp in pivot2.columns else None
            lo = cilo2.loc[h, tp] if tp in cilo2.columns else None
            hi = cihi2.loc[h, tp] if tp in cihi2.columns else None
            if m is None or pd.isna(m):
                line += " — |"
            else:
                line += f" {m:.2f} [{lo:.2f}, {hi:.2f}] |"
        lines.append(line + "\n")
    lines.append("\n")

    lines.append("## Interpretation\n\n")
    # Find biggest ratio_above_1
    pn = boot_df[boot_df["mean_ratio"] > 1].sort_values("mean_ratio", ascending=False).head(5)
    if len(pn) > 0:
        lines.append("Top cells where v4 BEATS random-within-TCW:\n")
        for _, r in pn.iterrows():
            lines.append(f"- `{r['head']}` at top-{int(r['top_pct']*100)}% ({r['k_kind']}): "
                         f"ratio = **{r['mean_ratio']:.2f}** [{r['ci_lo']:.2f}, {r['ci_hi']:.2f}], "
                         f"{r['n_above_1']}/{r['n_cancers']} cancers above 1\n")
    pf = boot_df[boot_df["mean_ratio"] < 0.95].sort_values("mean_ratio").head(5)
    if len(pf) > 0:
        lines.append("\nTop cells where v4 LOSES to random-within-TCW:\n")
        for _, r in pf.iterrows():
            lines.append(f"- `{r['head']}` at top-{int(r['top_pct']*100)}% ({r['k_kind']}): "
                         f"ratio = **{r['mean_ratio']:.2f}** [{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]\n")
    lines.append("\n")

    lines.append("## Verdict\n\n")
    any_pass = (boot_df["ci_lo"] > 1).any()
    if any_pass:
        n = (boot_df["ci_lo"] > 1).sum()
        lines.append(f"**YES — at {n} cell(s), v4 beats random-within-TCW with CI lower bound > 1.** "
                     "The model has within-TCW structural/context preference; the "
                     "previous limitation 'cannot beat TCW-density on TCW-restricted filter' "
                     "should be revised.\n")
    else:
        lines.append("**NO — no cell beats random-within-TCW with CI lower bound > 1.** "
                     "The model adds nothing once trinucleotide context is conditioned on. "
                     "The original limitation stands: v4 cannot beat TCW-density on a "
                     "TCW-restricted filter.\n")

    OUT_MD.write_text("".join(lines))
    log.info("Wrote %s", OUT_MD)
    log.info("Done in %.1f min", (time.time() - t0) / 60.0)


if __name__ == "__main__":
    main()
