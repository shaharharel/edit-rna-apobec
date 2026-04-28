#!/usr/bin/env python3
"""Analysis D — POG570 independent WGS validation.

Independent test cohort: 570 metastatic pan-cancer WGS from BC Cancer
(Pleasance et al. Nat Cancer 2020). Open-tier click-wrap, hg19, no DAR.

Same primary endpoint structure as Phase 1.5/1.6 Analysis B:
- 250 bp non-overlapping CDS windows aggregated by max(NN binary score)
- Top-1% windows
- Mutation filter: TCW-non-CpG C>T SNVs (cleanest APOBEC stratum)
- Compare to CpG-density-ranked baseline
- Per-cancer recall ratio + permutation null + BH-FDR

Output: experiments/multi_enzyme/outputs/pcawg_tcw_panel/analysis_D_pog570/
  enrichment_primary_pog570.json
  REPORT.md
"""
from __future__ import annotations
import json
import logging
import multiprocessing as mp
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control

ROOT = Path("/Users/shaharharel/Documents/github/edit-rna-apobec")
PANEL_PATH = ROOT / "experiments/multi_enzyme/outputs/pcawg_tcw_panel/panel_scores_cds.parquet"
POG570_PATH = ROOT / "data/raw/pog570/POG570_small_mutations.txt.gz"
OUT_DIR = ROOT / "experiments/multi_enzyme/outputs/pcawg_tcw_panel/analysis_D_pog570"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# POG570 analysis_cohort -> our target-cancer mapping
COHORT_MAP = {
    "COLO": "coadread",
    "SKCM": "skcm",
    "BRCA": "brca",
    "LUNG": "lusc",   # POG570 doesn't separate LUSC vs LUAD; treat as LUSC proxy
    "ESCA": "esca",
    "HNSC": "hnsc",
    "STAD": "stad",
    "HCC": "lihc",
    "BLCA": "blca",
    "CERV": "cesc",
}
TARGET_CANCERS = list(COHORT_MAP.values())
WINDOW_SIZE = 250
TOP_PCT = 0.01
PERM_REPS = 10000
SEED_BASE = 20260427

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def is_tcw_not_cpg_ct(chrom: str, pos: int, ref: str, alt: str, fasta_lookup):
    """Detect a TCW-non-CpG C>T mutation given the genomic chrom/pos and ref/alt.
    fasta_lookup(chrom, pos) returns the trinucleotide centered at pos (1-based).
    Forward strand: ref=C, alt=T, 5'-T-C-{A,T}, 3'-not-G
    Reverse strand: ref=G, alt=A, 5'-{A,T}-G-A, 3'-not-C
    """
    if ref == "C" and alt == "T":
        try:
            tri = fasta_lookup(chrom, pos)  # returns string at [pos-1, pos, pos+1]
            if len(tri) != 3:
                return False
            return tri[0] == "T" and tri[2] in "AT"
        except Exception:
            return False
    if ref == "G" and alt == "A":
        try:
            tri = fasta_lookup(chrom, pos)
            if len(tri) != 3:
                return False
            return tri[2] == "A" and tri[0] in "AT"
        except Exception:
            return False
    return False


def load_pog570(path: Path) -> pd.DataFrame:
    """Load and filter POG570 small mutations to C>T SNVs in our 10 cohorts."""
    log.info("Loading POG570 small mutations from %s", path)
    df = pd.read_csv(path, sep="\t", compression="gzip", low_memory=False)
    log.info("  raw rows: %d", len(df))
    # Filter to SNVs only (length 1)
    df = df[(df["ref"].str.len() == 1) & (df["alt"].str.len() == 1)].copy()
    # Filter to C>T or G>A
    df = df[((df["ref"] == "C") & (df["alt"] == "T")) | ((df["ref"] == "G") & (df["alt"] == "A"))]
    log.info("  C>T/G>A SNVs: %d", len(df))
    # Map cohort
    df["cancer"] = df["analysis_cohort"].map(COHORT_MAP)
    df = df.dropna(subset=["cancer"])
    log.info("  in target cohorts: %d", len(df))
    # Normalize chrom (POG570 has chrom without "chr" prefix; panel has "chr" prefix)
    df["chrom"] = "chr" + df["chrom"].astype(str)
    # Coerce pos to int. POG570 is 1-based VCF; our panel is 0-based (we did
    # Start_Position - 1 when building the TCGA pipelines). Convert here.
    df["pos"] = pd.to_numeric(df["pos"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["pos"])
    df["pos"] = df["pos"].astype(int) - 1   # 1-based VCF -> 0-based panel
    return df[["chrom", "pos", "ref", "alt", "cancer", "patient_id"]]


def load_panel(path: Path) -> pd.DataFrame:
    log.info("Loading panel scores from %s", path)
    df = pd.read_parquet(path)
    log.info("  rows: %d  cols: %s", len(df), list(df.columns)[:12])
    # Need chrom, pos, plus is_cpg, is_tc and the binary score
    keep = ["chrom", "pos", "score_binary"]
    if "is_cpg" in df.columns:
        keep.append("is_cpg")
    if "is_tc" in df.columns or "is_tcw" in df.columns:
        keep.append("is_tcw" if "is_tcw" in df.columns else "is_tc")
    return df[[c for c in keep if c in df.columns]].copy()


def filter_tcw_non_cpg(mut_df: pd.DataFrame, panel_df: pd.DataFrame) -> pd.DataFrame:
    """For each POG570 mutation, restrict to (chrom, pos) in the panel and apply
    TCW-non-CpG filter using hg19.fa trinucleotide context.

    For C>T at p (forward strand mutation): genome[p-1]=='T' AND genome[p+1] in {'A','T'}.
    For G>A at p (minus-strand mutation, equivalent to C>T on -):
        genome[p+1]=='A' AND genome[p-1] in {'A','T'}.
    The non-CpG part falls out automatically because we exclude G as 3' neighbor.
    """
    from pyfaidx import Fasta
    HG19 = ROOT / "data/raw/genomes/hg19.fa"
    log.info("Loading hg19.fa for trinucleotide context check ...")
    genome = Fasta(str(HG19), as_raw=False, sequence_always_upper=True)

    # Restrict to mutations whose (chrom, pos) is in the panel positions
    log.info("Filtering POG570 mutations to those in-panel ...")
    panel_set = panel_df[["chrom", "pos"]].copy()
    panel_set["pos"] = panel_set["pos"].astype(int)
    merged = mut_df.merge(panel_set, on=["chrom", "pos"], how="inner")
    log.info("  in-panel mutations: %d / %d (%.1f%%)",
             len(merged), len(mut_df), 100.0 * len(merged) / max(1, len(mut_df)))

    # Apply TCW-non-CpG filter using genome.
    # NOTE: panel.pos and merged.pos are 0-based (we did -1 shift in load_pog570).
    # pyfaidx genome[chrom][i] returns base at 0-indexed position i.
    keep = []
    sanity_ref_mismatches = 0
    for chrom, pos, ref, alt in zip(merged["chrom"], merged["pos"], merged["ref"], merged["alt"]):
        try:
            chrom_str = chrom if str(chrom).startswith("chr") else f"chr{chrom}"
            p = int(pos)  # 0-based panel pos
            base_at = str(genome[chrom_str][p]).upper()
            base_left = str(genome[chrom_str][p - 1]).upper()    # 5' neighbor
            base_right = str(genome[chrom_str][p + 1]).upper()   # 3' neighbor
        except Exception:
            keep.append(False)
            continue
        # Sanity: the base at the panel pos should match ref
        if base_at != ref:
            sanity_ref_mismatches += 1
            keep.append(False)
            continue
        if ref == "C" and alt == "T":
            keep.append(base_left == "T" and base_right in ("A", "T"))
        elif ref == "G" and alt == "A":
            keep.append(base_right == "A" and base_left in ("A", "T"))
        else:
            keep.append(False)
    if sanity_ref_mismatches > 0:
        log.warning("  %d/%d positions had ref-base mismatch with hg19 (off-by-one or wrong build?)",
                    sanity_ref_mismatches, len(merged))
    merged = merged[pd.Series(keep, index=merged.index)]
    log.info("  TCW-non-CpG mutations: %d", len(merged))
    return merged


def build_windows(panel_df: pd.DataFrame, window_size: int, genome=None) -> pd.DataFrame:
    """Build 250 bp non-overlapping CDS windows from the panel positions.
    Each window aggregates max(score_binary). CpG counts are computed from hg19
    genome by scanning each panel position's neighborhood. Falls back to "C with
    G immediately following on plus strand" using genome[pos+1]=='G' check.
    """
    log.info("Building %d bp non-overlapping windows over panel positions ...", window_size)
    p = panel_df[["chrom", "pos", "score_binary"]].copy()
    p["pos"] = p["pos"].astype(int)
    p["win_start"] = (p["pos"] // window_size) * window_size
    grp = p.groupby(["chrom", "win_start"])
    out = grp.agg(
        max_score_binary=("score_binary", "max"),
        n_pos=("pos", "size"),
    ).reset_index()
    out["win_end"] = out["win_start"] + window_size

    # Compute CpG count per panel position from hg19 (each panel C-position is CpG
    # iff genome[pos+1]=='G'; each G-position is CpG iff genome[pos-1]=='C').
    if "is_cpg" in panel_df.columns:
        is_cpg = panel_df.copy()
    elif genome is not None:
        log.info("  computing is_cpg per panel position from hg19 ...")
        chrom_arr = panel_df["chrom"].to_numpy()
        pos_arr = panel_df["pos"].astype(int).to_numpy()
        # Use 'strand' column when present to know if pos is C (+) or G (-)
        if "strand" in panel_df.columns:
            strand_arr = panel_df["strand"].to_numpy()
        else:
            strand_arr = np.array(["+"] * len(panel_df))
        is_cpg_arr = np.zeros(len(panel_df), dtype=bool)
        # Fast loop over chromosomes (load each chr's seq once)
        unique_chroms = pd.Series(chrom_arr).unique()
        for ch in unique_chroms:
            mask = chrom_arr == ch
            try:
                seq = str(genome[ch][:]).upper()
            except Exception:
                log.warning("  couldn't load %s; skipping CpG annotation there", ch)
                continue
            for i in np.where(mask)[0]:
                p_i = int(pos_arr[i])
                s = strand_arr[i]
                try:
                    if s == "+":
                        is_cpg_arr[i] = (p_i + 1 < len(seq)) and (seq[p_i + 1] == "G")
                    else:
                        is_cpg_arr[i] = (p_i - 1 >= 0) and (seq[p_i - 1] == "C")
                except Exception:
                    pass
        is_cpg = panel_df.copy()
        is_cpg["is_cpg"] = is_cpg_arr
    else:
        log.warning("  no genome provided; setting cpg_count to 0 (BASELINE WILL BE BROKEN)")
        out["cpg_count"] = 0
        log.info("  built %d windows", len(out))
        return out

    is_cpg["pos"] = is_cpg["pos"].astype(int)
    is_cpg["win_start"] = (is_cpg["pos"] // window_size) * window_size
    cpg_grp = is_cpg.groupby(["chrom", "win_start"])["is_cpg"].sum().reset_index()
    cpg_grp = cpg_grp.rename(columns={"is_cpg": "cpg_count"})
    out = out.merge(cpg_grp, on=["chrom", "win_start"], how="left")
    out["cpg_count"] = out["cpg_count"].fillna(0).astype(int)
    log.info("  built %d windows", len(out))
    return out


def assign_mutations_to_windows(muts: pd.DataFrame, window_size: int) -> pd.Series:
    return (muts["pos"].astype(int) // window_size) * window_size


def compute_per_cancer_primary(args):
    """Worker: per-cancer permutation null + recall ratio."""
    cancer, windows_df_arr, mut_pos_arr, n_top, perm_reps, seed = args
    rng = np.random.default_rng(seed)
    # windows_df_arr: numpy array of shape (n_windows, 3) — [win_idx, max_score, cpg_count]
    n_windows = windows_df_arr.shape[0]
    # Top-k by NN
    nn_top_idx = np.argpartition(-windows_df_arr[:, 1], n_top)[:n_top]
    cpg_top_idx = np.argpartition(-windows_df_arr[:, 2], n_top)[:n_top]
    # mut_pos_arr is indices into windows_df_arr
    mut_per_window = np.bincount(mut_pos_arr, minlength=n_windows)
    nn_recall = mut_per_window[nn_top_idx].sum() / max(1, mut_per_window.sum())
    cpg_recall = mut_per_window[cpg_top_idx].sum() / max(1, mut_per_window.sum())
    ratio = nn_recall / cpg_recall if cpg_recall > 0 else float("nan")
    # Permutation null: shuffle scores
    null_ratios = []
    scores = windows_df_arr[:, 1].copy()
    for _ in range(perm_reps):
        perm = rng.permutation(scores)
        perm_top_idx = np.argpartition(-perm, n_top)[:n_top]
        perm_recall = mut_per_window[perm_top_idx].sum() / max(1, mut_per_window.sum())
        if cpg_recall > 0:
            null_ratios.append(perm_recall / cpg_recall)
    null_ratios = np.array(null_ratios)
    p_perm = ((null_ratios >= ratio).sum() + 1) / (len(null_ratios) + 1)
    return {
        "cancer": cancer,
        "n_mutations_in_panel_tcw_noncpg": int(mut_per_window.sum()),
        "n_windows": n_windows,
        "n_top": n_top,
        "nn_recall": float(nn_recall),
        "cpg_recall": float(cpg_recall),
        "ratio": float(ratio),
        "p_perm": float(p_perm),
        "null_mean": float(null_ratios.mean()),
        "null_std": float(null_ratios.std()),
    }


def main():
    log.info("=== Analysis D: POG570 validation ===")
    panel = load_panel(PANEL_PATH)
    muts = load_pog570(POG570_PATH)
    tcw_muts = filter_tcw_non_cpg(muts, panel)

    if len(tcw_muts) == 0:
        log.error("No TCW-non-CpG mutations found in panel positions. Aborting.")
        return 1

    # Reuse the genome handle from filter_tcw_non_cpg
    from pyfaidx import Fasta
    HG19 = ROOT / "data/raw/genomes/hg19.fa"
    genome = Fasta(str(HG19), as_raw=False, sequence_always_upper=True)
    windows_df = build_windows(panel, WINDOW_SIZE, genome=genome)
    n_top = max(1, int(round(len(windows_df) * TOP_PCT)))
    log.info("Top-1%% of windows = %d", n_top)

    # Build per-cancer args
    windows_df["win_id"] = np.arange(len(windows_df))
    windows_df["chrom_str"] = windows_df["chrom"].astype(str)
    windows_arr = windows_df[["win_id", "max_score_binary", "cpg_count"]].to_numpy()

    # For each mutation, find which window it belongs to
    log.info("Mapping mutations to windows ...")
    tcw_muts["win_start"] = (tcw_muts["pos"].astype(int) // WINDOW_SIZE) * WINDOW_SIZE
    win_id_lookup = windows_df.set_index(["chrom", "win_start"])["win_id"]
    tcw_muts["win_id"] = tcw_muts.set_index(["chrom", "win_start"]).index.map(win_id_lookup)
    tcw_muts = tcw_muts.dropna(subset=["win_id"])
    tcw_muts["win_id"] = tcw_muts["win_id"].astype(int)
    log.info("  %d mutations mapped to windows", len(tcw_muts))

    cancers = sorted(tcw_muts["cancer"].unique())
    log.info("Cancers found: %s", cancers)

    args_list = []
    for i, cancer in enumerate(cancers):
        sub = tcw_muts[tcw_muts["cancer"] == cancer]
        if len(sub) < 100:
            log.warning("  %s has only %d mutations — including but flagging low power", cancer, len(sub))
        mut_arr = sub["win_id"].astype(int).to_numpy()
        seed = SEED_BASE + i * 1000
        args_list.append((cancer, windows_arr, mut_arr, n_top, PERM_REPS, seed))

    log.info("Running per-cancer in pool ...")
    with mp.Pool(min(8, len(args_list))) as pool:
        results = pool.map(compute_per_cancer_primary, args_list)

    # BH-FDR across the 10 cancers
    p_vals = [r["p_perm"] for r in results]
    q_vals = false_discovery_control(p_vals, method="bh")
    for r, q in zip(results, q_vals):
        r["q_bh"] = float(q)
        r["reject_bh"] = bool(q < 0.025)

    # Aggregate
    ratios = [r["ratio"] for r in results if not np.isnan(r["ratio"])]
    summary = {
        "experiment": "Analysis D — POG570 validation",
        "panel_path": str(PANEL_PATH),
        "pog570_path": str(POG570_PATH),
        "window_size": WINDOW_SIZE,
        "aggregator": "max",
        "top_pct": TOP_PCT,
        "perm_reps": PERM_REPS,
        "filter": "tcw_not_cpg",
        "head": "binary",
        "mean_ratio": float(np.mean(ratios)),
        "median_ratio": float(np.median(ratios)),
        "n_cancers_above_1": int(sum(r > 1 for r in ratios)),
        "n_cancers_above_1_5": int(sum(r > 1.5 for r in ratios)),
        "n_bh_significant_p025": int(sum(r["reject_bh"] for r in results)),
        "primary_pass_a_mean_15": float(np.mean(ratios)) >= 1.5,
        "primary_pass_b_bh_6_of_n": int(sum(r["reject_bh"] for r in results)) >= 6,
        "per_cancer": results,
    }

    out_json = OUT_DIR / "enrichment_primary_pog570.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Wrote %s", out_json)

    # Quick report
    log.info("\n=== Summary ===")
    log.info("Mean ratio across %d cancers: %.2f×", len(ratios), summary["mean_ratio"])
    log.info("Cancers above 1.0×: %d / %d", summary["n_cancers_above_1"], len(results))
    log.info("BH-significant (q<0.025): %d / %d", summary["n_bh_significant_p025"], len(results))
    log.info("Primary (a) (mean ratio >=1.5): %s", "PASS" if summary["primary_pass_a_mean_15"] else "FAIL")
    log.info("Primary (b) (>=6/N BH<0.025): %s", "PASS" if summary["primary_pass_b_bh_6_of_n"] else "FAIL")
    log.info("\nPer-cancer:")
    for r in sorted(results, key=lambda x: -x["ratio"] if not np.isnan(x["ratio"]) else -999):
        log.info("  %-10s ratio=%6.2f  q=%.4f  rej=%s  n_mut=%d",
                 r["cancer"], r["ratio"], r["q_bh"], r["reject_bh"],
                 r["n_mutations_in_panel_tcw_noncpg"])

    return 0


if __name__ == "__main__":
    sys.exit(main())
