#!/usr/bin/env python3
"""Analysis B — TCGA + PCAWG-coding combined enrichment with QA fixes.

Same pre-registered shape as Analysis A but mutation source = coding-only TCGA-MC3
+ cBioPortal-PCAWG (10 cancers). Primary endpoint:

    Of all TCW-non-CpG C>T SNVs in the combined TCGA+PCAWG-coding cohort that
    fall within the 8.45 M scored CDS positions, what fraction are recalled in
    the top-1% of 1 kb CDS windows ranked by binary head MEAN score? Compare to
    CpG-density-ranked baseline. Pass if (a) mean_ratio >= 1.5x across 10 cancers,
    (b) >=6 cancers BH-FDR q<0.05 under permutation null, (c) survives driver
    ablation (>=1.3x), (d) survives training-site mask (>=1.3x).

(B1 is N/A here — Analysis B does not use SBS attribution.)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(line_buffering=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from analysis_A_pcawg_wgs import (  # noqa: E402
    PRIMARY_HEAD, PRIMARY_PCT, PRIMARY_BASELINE,
    PRIMARY_MIN_LIFT_A, PRIMARY_MIN_SIGNIF, PRIMARY_MIN_LIFT_DRIVER,
    PRIMARY_MIN_LIFT_MASKED, PRIMARY_ALPHA, WINDOW_BP, PERM_REPS,
    HEADS, PERCENTILES, count_tcw_in_window,
    load_panel_scores, load_v3_positions_hg19, load_bailey_drivers,
    recall_ratio_with_perm, COMP,
)
from statsmodels.stats.multitest import multipletests   # noqa: E402

# Analysis B primary filter: TCW-non-CpG (no SBS attribution available coding-only)
PRIMARY_FILTER_B = "tcw_not_cpg"
CANCERS_TCGA = ["blca", "brca", "cesc", "coadread", "esca", "hnsc", "lihc", "lusc", "skcm", "stad"]


def _load_one_maf(path: Path, cancer: str, source: str) -> pd.DataFrame | None:
    if not path.exists():
        logger.warning("Missing %s — skipping cancer %s/%s", path, cancer, source)
        return None
    try:
        df = pd.read_csv(path, sep="\t", low_memory=False)
    except Exception as ex:
        logger.warning("  %s/%s: read failed: %s", cancer, source, ex)
        return None
    cols = df.columns.tolist()
    if "Chromosome" not in cols or "Start_Position" not in cols:
        logger.warning("  %s/%s: missing Chromosome/Start_Position", cancer, source)
        return None
    df = df[df.get("Variant_Type", "SNP") == "SNP"]
    ref = df.get("Reference_Allele")
    alt = df.get("Tumor_Seq_Allele2", df.get("Tumor_Seq_Allele"))
    if ref is None or alt is None:
        return None
    is_CT = (ref == "C") & (alt == "T")
    is_GA = (ref == "G") & (alt == "A")
    df = df[is_CT | is_GA].copy()
    df["strand"] = np.where((df["Reference_Allele"] == "C") & (df["Tumor_Seq_Allele2"] == "T"), "+", "-")
    df["pos"] = df["Start_Position"].astype(int) - 1
    df["chrom"] = df["Chromosome"].astype(str)
    df.loc[~df["chrom"].str.startswith("chr"), "chrom"] = "chr" + df["chrom"]
    df["cancer"] = cancer
    df["source"] = source
    out = df[["chrom", "pos", "strand", "cancer", "source"]]
    logger.info("  %s/%s: %d C>T mutations", cancer, source, len(out))
    return out


def load_combined_coding_maf(pcawg_dir: Path, tcga_dir: Path) -> pd.DataFrame:
    rows = []
    for cancer in CANCERS_TCGA:
        d = _load_one_maf(pcawg_dir / f"{cancer}_pcawg_mutations.txt", cancer, "pcawg_coding")
        if d is not None:
            rows.append(d)
        d = _load_one_maf(tcga_dir / f"{cancer}_tcga_pan_can_atlas_2018_mutations.txt", cancer, "tcga_mc3")
        if d is not None:
            rows.append(d)
    if not rows:
        raise RuntimeError(f"No coding MAFs found in {pcawg_dir} or {tcga_dir}")
    combined = pd.concat(rows, ignore_index=True)
    combined = combined[combined["chrom"].isin(set([f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]))]
    logger.info("Combined coding C>T: %d, %d cancers, sources=%s",
                len(combined), combined["cancer"].nunique(),
                combined["source"].value_counts().to_dict())
    return combined


def build_windows_b(panel: pd.DataFrame, maf: pd.DataFrame, drivers: set[str],
                    v3_hg19: pd.DataFrame, hg19_fa: Path, out_dir: Path) -> pd.DataFrame:
    from pyfaidx import Fasta
    genome = Fasta(str(hg19_fa))

    panel = panel.copy()
    panel["win_id"] = panel["chrom"].astype(str) + "_" + (panel["pos"] // WINDOW_BP).astype(str)
    score_cols = [c for c in panel.columns if c.startswith("score_")]
    agg_max = {c: "max" for c in score_cols}
    panel_for_mean = panel.copy()
    for c in score_cols:
        panel_for_mean[c + "_mean"] = panel_for_mean[c]
    agg_mean = {c + "_mean": "mean" for c in score_cols}
    agg = {**agg_max, **agg_mean, "pos": "min",
           "gene": lambda s: s.mode().iloc[0] if len(s) > 0 else ""}
    if "valid" in panel.columns:
        agg["valid"] = "any"
    w = panel_for_mean.groupby(["chrom", "win_id"], sort=False, as_index=False).agg({
        **agg, "strand": "first",
    })
    counts = panel.groupby(["chrom", "win_id"]).size().reset_index(name="n_cands")
    w = w.merge(counts, on=["chrom", "win_id"], how="left")
    logger.info("  %d windows", len(w))

    cpg = np.zeros(len(w), dtype=np.int32); tcw = np.zeros(len(w), dtype=np.int32)
    for i, r in enumerate(w.itertuples(index=False)):
        try:
            win_n = int(r.win_id.split("_")[-1])
            seq = str(genome[r.chrom][win_n * WINDOW_BP:(win_n + 1) * WINDOW_BP]).upper()
            cpg[i] = seq.count("CG")
            tcw[i] = count_tcw_in_window(seq)
        except Exception:
            continue
    w["cpg_density"] = cpg
    w["tcw_density"] = tcw

    v3_win = set()
    for r in v3_hg19.itertuples(index=False):
        base = int(r.pos) // WINDOW_BP
        for dw in range(-1, 2):
            v3_win.add((r.chrom, base + dw))
    w["training_contaminated"] = [((r.chrom, int(r.win_id.split("_")[-1])) in v3_win)
                                  for r in w.itertuples(index=False)]
    w["is_driver"] = w["gene"].astype(str).str.upper().isin({g.upper() for g in drivers})

    # Panel restriction
    n_total_maf = len(maf)
    panel_pos = set(zip(panel["chrom"].astype(str).values, panel["pos"].astype(int).values))
    in_panel_arr = np.array([(c, int(p)) in panel_pos for c, p in zip(maf["chrom"], maf["pos"])])
    coverage_pct = 100 * in_panel_arr.sum() / max(n_total_maf, 1)
    logger.info("MAF in panel: %d/%d (%.2f%%)", in_panel_arr.sum(), n_total_maf, coverage_pct)
    coverage_per_cancer_source = (
        pd.Series(in_panel_arr).groupby([maf["cancer"].values, maf["source"].values]).mean().to_dict()
    )
    maf = maf.iloc[np.where(in_panel_arr)[0]].copy()
    coverage_stats = {
        "n_total_maf_C2T": int(n_total_maf),
        "n_in_panel": int(len(maf)),
        "coverage_pct": float(coverage_pct),
        "per_cancer_source_coverage": {f"{k[0]}_{k[1]}": float(v)
                                       for k, v in coverage_per_cancer_source.items()},
    }
    with open(out_dir / "panel_coverage_stats.json", "w") as f:
        json.dump(coverage_stats, f, indent=2)

    # Trinuc + win_id
    maf["win_id"] = maf["chrom"].astype(str) + "_" + (maf["pos"].astype(int) // WINDOW_BP).astype(str)
    tris = []
    for r in maf.itertuples(index=False):
        try:
            s = str(genome[r.chrom][r.pos - 1: r.pos + 2]).upper()
        except Exception:
            s = "NNN"
        if r.strand == "-":
            s = s.translate(COMP)[::-1]
        tris.append(s if len(s) == 3 else "NNN")
    maf["subtype"] = tris

    # Counts (per cancer; sources combined for coverage; report split too)
    ct_counts = maf.groupby(["chrom", "win_id", "cancer"]).size().unstack(fill_value=0)
    ct_counts.columns = [f"n_CT_{c}" for c in ct_counts.columns]
    tcw_mask = (maf["subtype"].str[0] == "T") & (maf["subtype"].str[2].isin(["A", "T"]))
    tcw_counts = maf[tcw_mask].groupby(["chrom", "win_id", "cancer"]).size().unstack(fill_value=0)
    tcw_counts.columns = [f"n_tcw_not_cpg_{c}" for c in tcw_counts.columns]
    cpg_mask = maf["subtype"].str[2] == "G"
    cpg_counts = maf[cpg_mask].groupby(["chrom", "win_id", "cancer"]).size().unstack(fill_value=0)
    cpg_counts.columns = [f"n_cpg_{c}" for c in cpg_counts.columns]

    w2 = w.set_index(["chrom", "win_id"])
    for tbl in [ct_counts, tcw_counts, cpg_counts]:
        w2 = w2.join(tbl, how="left")
    w2 = w2.fillna(0)
    out = out_dir / "windows.parquet"
    w2.reset_index().to_parquet(out, index=False)
    logger.info("Wrote %s (%.1f MB, %d windows)", out, out.stat().st_size / 1e6, len(w2))
    return w2.reset_index()


def filter_to_count_col_b(filt: str, cancer: str) -> str:
    return {
        "all_C2T": f"n_CT_{cancer}",
        "tcw_not_cpg": f"n_tcw_not_cpg_{cancer}",
        "cpg": f"n_cpg_{cancer}",
    }.get(filt, f"n_tcw_not_cpg_{cancer}")


def run_primary_b(w: pd.DataFrame, out_dir: Path) -> dict:
    logger.info("\n%s\nPRIMARY (Analysis B)\n%s", "=" * 70, "=" * 70)
    score_col = f"score_{PRIMARY_HEAD}_mean"
    if score_col not in w.columns:
        score_col = f"score_{PRIMARY_HEAD}"
    cancers = [c for c in CANCERS_TCGA if filter_to_count_col_b(PRIMARY_FILTER_B, c) in w.columns]
    logger.info("  cancers: %s", cancers)
    p_values = []; ratios_p = []; ratios_m = []; ratios_d = []; ratios_r = []
    rng = np.random.default_rng(20260428)
    results = {"per_cancer": {}, "pooled": {}, "pass_criteria": {}}
    for cancer in cancers:
        mut_col = filter_to_count_col_b(PRIMARY_FILTER_B, cancer)
        raw = recall_ratio_with_perm(w, score_col, mut_col, "cpg_density", PRIMARY_PCT, rng=rng)
        masked = recall_ratio_with_perm(w, score_col, mut_col, "cpg_density", PRIMARY_PCT,
                                        mask_col="training_contaminated", rng=rng)
        wd = w[~w["is_driver"]]
        driver = recall_ratio_with_perm(wd, score_col, mut_col, "cpg_density", PRIMARY_PCT, rng=rng)
        wp = w[(~w["is_driver"]) & (~w["training_contaminated"])]
        primary = recall_ratio_with_perm(wp, score_col, mut_col, "cpg_density", PRIMARY_PCT, rng=rng)
        results["per_cancer"][cancer] = {"raw": raw, "masked": masked,
                                         "driver_ablated": driver, "primary": primary}
        logger.info("  %s  raw=%.3f  masked=%.3f  driver=%.3f  primary=%.3f  p_perm=%.2e  total_mut=%d",
                    cancer, raw["ratio"], masked["ratio"], driver["ratio"],
                    primary["ratio"], primary["p_perm"], primary["total_mut"])
        p_values.append(primary["p_perm"])
        ratios_r.append(raw["ratio"]); ratios_m.append(masked["ratio"])
        ratios_d.append(driver["ratio"]); ratios_p.append(primary["ratio"])

    if p_values:
        rej, q, _, _ = multipletests(p_values, alpha=PRIMARY_ALPHA, method="fdr_bh")
    else:
        rej = []; q = []
    for i, c in enumerate(cancers):
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
    results["pooled"] = {"mean_ratio_primary": mean_p, "mean_ratio_masked": mean_m,
                        "mean_ratio_driver": mean_d, "n_cancers_signif_q05": n_signif}
    results["pass_criteria"] = {
        "(a)_mean_primary_>=_1.5": {"val": mean_p, "thresh": PRIMARY_MIN_LIFT_A, "pass": pass_a},
        "(b)_signif_>=_6": {"val": n_signif, "thresh": PRIMARY_MIN_SIGNIF, "pass": pass_b},
        "(c)_driver_>=_1.3": {"val": mean_d, "thresh": PRIMARY_MIN_LIFT_DRIVER, "pass": pass_c},
        "(d)_masked_>=_1.3": {"val": mean_m, "thresh": PRIMARY_MIN_LIFT_MASKED, "pass": pass_d},
        "PASS": passed,
    }
    logger.info("Analysis B PRIMARY: %s", "PASS" if passed else "FAIL")
    out = out_dir / "enrichment_primary.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results


def run_secondary_b(w: pd.DataFrame, out_dir: Path) -> dict:
    logger.info("\n%s\nSECONDARY (B)\n%s", "=" * 70, "=" * 70)
    w_pool = w[(~w["is_driver"]) & (~w["training_contaminated"])].copy()
    cancers = [c for c in CANCERS_TCGA if filter_to_count_col_b(PRIMARY_FILTER_B, c) in w_pool.columns]
    if "cpg_density" in w_pool.columns and (w_pool["cpg_density"].nunique() >= 10):
        w_pool["cpg_decile"] = pd.qcut(w_pool["cpg_density"], q=10, labels=False, duplicates="drop")
    else:
        w_pool["cpg_decile"] = 0
    decile = {}
    rng = np.random.default_rng(20260429)
    for d in sorted(w_pool["cpg_decile"].dropna().unique()):
        wd = w_pool[w_pool["cpg_decile"] == d]
        per_c = {}
        score_col = "score_binary_mean" if "score_binary_mean" in wd.columns else "score_binary"
        for c in cancers:
            mut_col = filter_to_count_col_b(PRIMARY_FILTER_B, c)
            r = recall_ratio_with_perm(wd, score_col, mut_col, "cpg_density", 0.01,
                                       perm_reps=2000, rng=rng)
            per_c[c] = r
        decile[int(d)] = {
            "n_windows": len(wd),
            "mean_ratio": float(np.nanmean([v["ratio"] for v in per_c.values() if v["ratio"] == v["ratio"]])),
            "per_cancer": per_c,
        }
    rows = []
    rng2 = np.random.default_rng(20260430)
    for head in HEADS:
        for kind in ("_mean", ""):
            score_col = f"score_{head}{kind}"
            if score_col in w_pool.columns:
                break
        else:
            continue
        for filt in ("all_C2T", "tcw_not_cpg", "cpg"):
            for pct in PERCENTILES:
                for cancer in cancers:
                    mut_col = filter_to_count_col_b(filt, cancer)
                    if mut_col not in w_pool.columns:
                        continue
                    r = recall_ratio_with_perm(w_pool, score_col, mut_col, "cpg_density",
                                               pct, perm_reps=1000, rng=rng2)
                    rows.append({"head": head, "filter": filt, "pct": pct, "cancer": cancer, **r})
    df = pd.DataFrame(rows)
    if len(df):
        p = df["p_perm"].clip(0, 1).values
        rej, q, _, _ = multipletests(p, alpha=PRIMARY_ALPHA, method="fdr_bh")
        df["q_bh"] = q; df["reject_bh"] = rej
    df.to_csv(out_dir / "enrichment_secondary.csv", index=False)
    summary = {
        "n_rows": int(len(df)),
        "n_signif_q05": int((df["q_bh"] < 0.05).sum()) if len(df) else 0,
        "per_cpg_decile": decile,
        "per_decile_min_ratio": float(min([v["mean_ratio"] for v in decile.values()
                                           if v["mean_ratio"] == v["mean_ratio"]]) if decile else float("nan")),
    }
    with open(out_dir / "enrichment_secondary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    return summary


def write_report_b(primary, secondary, out_dir: Path):
    lines = ["# Analysis B — TCGA + PCAWG-coding enrichment\n",
             f"\nGenerated: {time.strftime('%Y-%m-%d %H:%M')}\n\n",
             "## Pre-registered primary endpoint\n",
             f"- Head: `{PRIMARY_HEAD}` | Filter: `{PRIMARY_FILTER_B}` | Top pct: {PRIMARY_PCT}\n",
             f"- Test: permutation null + BH-FDR across cancers\n\n"]
    pc = primary["pass_criteria"]
    lines.append(f"### RESULT: **{'PASS' if pc['PASS'] else 'FAIL'}**\n\n")
    for k, v in pc.items():
        if k == "PASS":
            continue
        marker = "PASS" if v["pass"] else "FAIL"
        lines.append(f"- [{marker}] {k}: value={v['val']:.3f} threshold={v['thresh']}\n")
    lines.append("\n## Per-cancer primary\n")
    lines.append("| cancer | mut_ratio | recall_model | recall_cpg | total_mut | p_perm | q_bh | reject |\n")
    lines.append("|--------|-----------|--------------|-----------|-----------|--------|------|--------|\n")
    for cancer, det in primary["per_cancer"].items():
        pr = det["primary"]
        q = pr.get("q_bh", float("nan"))
        rej = pr.get("reject_bh", False)
        lines.append(f"| {cancer} | {pr['ratio']:.3f} | {pr['recall_model']:.4f} | "
                     f"{pr['recall_baseline']:.4f} | {pr['total_mut']} | "
                     f"{pr['p_perm']:.2e} | {q:.3g} | {'Y' if rej else 'N'} |\n")
    lines.append(f"\n## Secondary\n- Rows: {secondary['n_rows']}\n- Significant q<0.05: {secondary['n_signif_q05']}\n")
    lines.append(f"- Per-decile min mean_ratio: {secondary.get('per_decile_min_ratio', float('nan')):.3f}\n")
    with open(out_dir / "REPORT.md", "w") as f:
        f.writelines(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", type=Path, required=True)
    ap.add_argument("--pcawg-dir", type=Path, default=PROJECT_ROOT / "data/raw/pcawg/by_cancer")
    ap.add_argument("--tcga-dir", type=Path, default=PROJECT_ROOT / "data/raw/tcga")
    ap.add_argument("--bailey-drivers", type=Path, default=None)
    ap.add_argument("--v3-splits", type=Path, default=PROJECT_ROOT / "data/processed/multi_enzyme/splits_multi_enzyme_v3_with_negatives.csv")
    ap.add_argument("--hg19", type=Path, default=PROJECT_ROOT / "data/raw/genomes/hg19.fa")
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "experiments/multi_enzyme/outputs/pcawg_tcw_panel/analysis_B_coding_panel")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    panel = load_panel_scores(args.panel)
    maf = load_combined_coding_maf(args.pcawg_dir, args.tcga_dir)
    drivers = load_bailey_drivers(args.bailey_drivers)
    v3 = load_v3_positions_hg19(args.v3_splits)
    w = build_windows_b(panel, maf, drivers, v3, args.hg19, args.out_dir)
    primary = run_primary_b(w, args.out_dir)
    secondary = run_secondary_b(w, args.out_dir)
    write_report_b(primary, secondary, args.out_dir)
    logger.info("DONE Analysis B.")


if __name__ == "__main__":
    main()
