#!/usr/bin/env python3
"""Multi-cell-line replication-timing covariate ablation.

Refactor of `repliseq_quintile_ablation.py` that takes a cell line label and
bigWig path as arguments. Used to replicate the GM12878 ablation on HepG2 and
MCF-7 to confirm the all_CT lift is not cell-line-specific.

Usage:
  conda run -n quris python scripts/multi_enzyme/repliseq_quintile_ablation_multicell.py \
      --bigwig data/raw/repliseq/HepG2_repliseq_wavelet.bw \
      --label HepG2

Outputs (per cell line, suffixed with --label):
  - repliseq_lift_by_quintile_<label>.csv
  - repliseq_top1pct_concentration_<label>.csv
  - repliseq_quintile_distribution_<label>.csv
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
PANEL = ROOT / (
    "experiments/multi_enzyme/outputs/pcawg_tcw_panel/v4_outputs/"
    "panel_scores_v4_cds_apobec1retrained.parquet"
)
TCGA_DIR = ROOT / "data/raw/tcga"
PCAWG_DIR = ROOT / "data/raw/pcawg/by_cancer"
HG19 = ROOT / "data/raw/genomes/hg19.fa"
OUT = ROOT / "experiments/multi_enzyme/outputs/pcawg_tcw_panel/v4_outputs"

CANCERS = [
    "blca", "brca", "cesc", "coadread", "esca", "hnsc", "lihc", "lusc",
    "skcm", "stad",
]
HEADS = ["score_binary", "score_A3A"]
FILTERS = ["TCW_nonCpG", "all_CT"]

TOP_PCT = 0.01
N_BOOT = 5000
SEED = 20260427

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Load mutations (identical to original script)
# ------------------------------------------------------------------ #
def _load_one_maf(path: Path, cancer: str, source: str) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, sep="\t", low_memory=False)
    except Exception as ex:
        log.warning("  %s/%s: read failed: %s", cancer, source, ex)
        return None
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
        "+", "-",
    )
    df["pos"] = df["Start_Position"].astype(int) - 1
    df["chrom"] = df["Chromosome"].astype(str)
    df.loc[~df["chrom"].str.startswith("chr"), "chrom"] = "chr" + df["chrom"]
    df["cancer"] = cancer
    df["source"] = source
    return df[["chrom", "pos", "strand", "cancer", "source"]]


def load_combined_coding_maf() -> pd.DataFrame:
    log.info("Loading TCGA-MC3 + cBioPortal-PCAWG coding MAFs ...")
    rows = []
    for cancer in CANCERS:
        d = _load_one_maf(
            PCAWG_DIR / f"{cancer}_pcawg_mutations.txt", cancer, "pcawg_coding"
        )
        if d is not None:
            rows.append(d)
        d = _load_one_maf(
            TCGA_DIR / f"{cancer}_tcga_pan_can_atlas_2018_mutations.txt",
            cancer,
            "tcga_mc3",
        )
        if d is not None:
            rows.append(d)
    combined = pd.concat(rows, ignore_index=True)
    valid_chroms = set([f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"])
    combined = combined[combined["chrom"].isin(valid_chroms)]
    return combined


def annotate_panel_context(panel: pd.DataFrame) -> pd.DataFrame:
    from pyfaidx import Fasta
    log.info("Annotating panel positions with is_cpg + is_TCW_C ...")
    genome = Fasta(str(HG19), as_raw=False, sequence_always_upper=True)
    n = len(panel)
    is_cpg = np.zeros(n, dtype=bool)
    is_tcw_c = np.zeros(n, dtype=bool)
    chroms = panel["chrom"].to_numpy()
    poses = panel["pos"].astype(int).to_numpy()
    strands = panel["strand"].to_numpy()
    panel_idx = np.arange(n)
    for ch in pd.Series(chroms).unique():
        mask = chroms == ch
        idx = panel_idx[mask]
        if len(idx) == 0:
            continue
        try:
            seq = np.frombuffer(
                str(genome[ch][:]).upper().encode("ascii"), dtype=np.uint8
            )
        except Exception:
            continue
        L = len(seq)
        ps = poses[mask]
        ss = strands[mask]
        ok = (ps >= 1) & (ps + 1 < L)
        valid_idx = idx[ok]
        ps_ok = ps[ok]
        ss_ok = ss[ok]
        left = seq[ps_ok - 1]
        right = seq[ps_ok + 1]
        is_plus = ss_ok == "+"
        is_minus = ~is_plus
        is_cpg[valid_idx[is_plus]] = right[is_plus] == ord("G")
        is_cpg[valid_idx[is_minus]] = left[is_minus] == ord("C")
        right_AT = (right == ord("A")) | (right == ord("T"))
        left_AT = (left == ord("A")) | (left == ord("T"))
        is_tcw_c[valid_idx[is_plus]] = (
            (left[is_plus] == ord("T")) & right_AT[is_plus]
        )
        is_tcw_c[valid_idx[is_minus]] = (
            (right[is_minus] == ord("A")) & left_AT[is_minus]
        )
    panel = panel.copy()
    panel["is_cpg"] = is_cpg
    panel["is_TCW_C"] = is_tcw_c
    panel["is_TCW_nonCpG"] = is_tcw_c & ~is_cpg
    return panel


def annotate_repliseq(panel: pd.DataFrame, bw_path: Path) -> pd.DataFrame:
    import pyBigWig
    log.info("Annotating panel positions with Repli-seq value (%s) ...", bw_path)
    bw = pyBigWig.open(str(bw_path))
    bw_chroms = set(bw.chroms().keys())
    n = len(panel)
    repliseq = np.full(n, np.nan, dtype=np.float32)
    chroms_arr = panel["chrom"].to_numpy()
    poses = panel["pos"].astype(int).to_numpy()
    panel_idx = np.arange(n)
    t0 = time.time()
    for ch in pd.Series(chroms_arr).unique():
        if ch not in bw_chroms:
            log.warning("  %s not in bigWig (skipping)", ch)
            continue
        mask = chroms_arr == ch
        idx = panel_idx[mask]
        ps = poses[mask]
        ch_size = bw.chroms()[ch]
        try:
            vec = bw.values(ch, 0, ch_size, numpy=True)
        except Exception as ex:
            log.warning("  read fail %s: %s", ch, ex)
            continue
        ok = (ps >= 0) & (ps < ch_size)
        valid_idx = idx[ok]
        ps_ok = ps[ok]
        repliseq[valid_idx] = vec[ps_ok]
    bw.close()
    log.info("  Repli-seq annotation took %.1fs", time.time() - t0)
    panel = panel.copy()
    panel["repliseq"] = repliseq
    n_nan = int(np.isnan(repliseq).sum())
    log.info(
        "  Total: n=%d, NaN=%d (%.2f%%), min=%.2f, max=%.2f, median=%.2f",
        n, n_nan, 100 * n_nan / n,
        np.nanmin(repliseq), np.nanmax(repliseq), np.nanmedian(repliseq),
    )
    return panel


def assign_quintiles(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.dropna(subset=["repliseq"]).copy()
    qcut = pd.qcut(
        panel["repliseq"], 5, labels=["Q5", "Q4", "Q3", "Q2", "Q1"], duplicates="drop"
    )
    panel["repliseq_quintile"] = qcut.astype(str)
    return panel


def build_mutation_lookup(maf: pd.DataFrame) -> pd.DataFrame:
    g = maf.groupby(["chrom", "pos", "cancer"]).size().reset_index(name="n_mut")
    return g


def annotate_panel_with_mutations(panel: pd.DataFrame, mut_grouped: pd.DataFrame) -> pd.DataFrame:
    pivot = mut_grouped.pivot_table(
        index=["chrom", "pos"], columns="cancer", values="n_mut",
        fill_value=0, aggfunc="sum",
    ).reset_index()
    pivot.columns = [c if c in ("chrom", "pos") else f"mut_{c}" for c in pivot.columns]
    out = panel.merge(pivot, how="left", on=["chrom", "pos"])
    for c in CANCERS:
        col = f"mut_{c}"
        if col not in out.columns:
            out[col] = 0
        out[col] = out[col].fillna(0).astype(np.int32)
    return out


def panel_lift_in_subset(
    sub: pd.DataFrame, head: str, filter_col: str | None,
    top_pct: float, seed: int,
) -> dict:
    if filter_col is not None:
        valid = sub[sub[filter_col]].copy()
    else:
        valid = sub.copy()
    n_valid = len(valid)
    if n_valid < 20:
        return {
            "n_panel_positions": n_valid, "n_top1pct_in_quintile": 0,
            "mean_recall": float("nan"), "lift_vs_random": float("nan"),
            "ci_lo": float("nan"), "ci_hi": float("nan"),
            "per_cancer_recalls": {c: float("nan") for c in CANCERS},
        }
    k = max(1, int(round(top_pct * n_valid)))
    scores = valid[head].to_numpy()
    top_idx = np.argpartition(-scores, k - 1)[:k]
    top_mask = np.zeros(n_valid, dtype=bool)
    top_mask[top_idx] = True
    rng = np.random.default_rng(seed)
    per_cancer_recalls = {}
    per_cancer_lifts = []
    for cancer in CANCERS:
        mut_arr = valid[f"mut_{cancer}"].to_numpy()
        total_mut = int(mut_arr.sum())
        if total_mut == 0:
            per_cancer_recalls[cancer] = float("nan")
            continue
        head_recall = float(mut_arr[top_mask].sum()) / total_mut
        random_recall = k / n_valid
        lift = head_recall / random_recall if random_recall > 0 else float("nan")
        per_cancer_recalls[cancer] = head_recall
        per_cancer_lifts.append(lift)
    if len(per_cancer_lifts) == 0:
        return {
            "n_panel_positions": n_valid, "n_top1pct_in_quintile": k,
            "mean_recall": float("nan"), "lift_vs_random": float("nan"),
            "ci_lo": float("nan"), "ci_hi": float("nan"),
            "per_cancer_recalls": per_cancer_recalls,
        }
    arr = np.asarray(per_cancer_lifts, dtype=float)
    mean_lift = float(np.mean(arr))
    n_arr = len(arr)
    bs_idx = rng.integers(0, n_arr, size=(N_BOOT, n_arr))
    bs = arr[bs_idx].mean(axis=1)
    lo = float(np.percentile(bs, 2.5))
    hi = float(np.percentile(bs, 97.5))
    valid_recalls = [r for r in per_cancer_recalls.values() if not np.isnan(r)]
    mean_recall = float(np.mean(valid_recalls)) if valid_recalls else float("nan")
    return {
        "n_panel_positions": n_valid, "n_top1pct_in_quintile": k,
        "mean_recall": mean_recall, "lift_vs_random": mean_lift,
        "ci_lo": lo, "ci_hi": hi,
        "per_cancer_recalls": per_cancer_recalls,
    }


def concentration_top1pct(panel: pd.DataFrame, head: str) -> pd.DataFrame:
    n = len(panel)
    k = max(1, int(round(TOP_PCT * n)))
    scores = panel[head].to_numpy()
    top_idx = np.argpartition(-scores, k - 1)[:k]
    top_q = panel.iloc[top_idx]["repliseq_quintile"].value_counts().to_dict()
    rows = []
    for q in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
        n_panel_q = int((panel["repliseq_quintile"] == q).sum())
        n_top_q = int(top_q.get(q, 0))
        rows.append({
            "head": head, "quintile": q,
            "n_panel_positions": n_panel_q,
            "n_top1pct_positions": n_top_q,
            "fraction_of_top1pct": n_top_q / k if k > 0 else float("nan"),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ #
# Cached annotated panel (mutations + trinuc context don't depend on cell line)
# ------------------------------------------------------------------ #
def get_base_panel() -> pd.DataFrame:
    """Loads panel + trinuc context + mutation hits.
    Cached as parquet to avoid recomputation when running multiple cell lines.
    """
    cache = OUT / "_panel_base_with_mutations.parquet"
    if cache.exists():
        log.info("Loading cached base panel: %s", cache)
        return pd.read_parquet(cache)

    log.info("Building base panel (no Repli-seq) ...")
    panel = pd.read_parquet(PANEL)
    panel = panel[panel["valid"]].copy()
    panel["pos"] = panel["pos"].astype(int)
    panel = annotate_panel_context(panel)
    maf = load_combined_coding_maf()
    mut_grouped = build_mutation_lookup(maf)
    panel = annotate_panel_with_mutations(panel, mut_grouped)
    log.info("Caching base panel: %s", cache)
    panel.to_parquet(cache, index=False)
    return panel


def run_one_cellline(label: str, bw_path: Path, base_panel: pd.DataFrame) -> dict:
    log.info("=" * 70)
    log.info("CELL LINE: %s   (bigWig: %s)", label, bw_path.name)
    log.info("=" * 70)

    panel = annotate_repliseq(base_panel, bw_path)
    panel = assign_quintiles(panel)
    counts = panel["repliseq_quintile"].value_counts().sort_index()
    log.info("  quintile counts: %s", counts.to_dict())

    # Distribution
    dist_rows = []
    for q in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
        sub = panel[panel["repliseq_quintile"] == q]
        dist_rows.append({
            "cellline": label, "quintile": q, "n_positions": len(sub),
            "repliseq_min": float(sub["repliseq"].min()) if len(sub) else float("nan"),
            "repliseq_max": float(sub["repliseq"].max()) if len(sub) else float("nan"),
            "repliseq_median": float(sub["repliseq"].median()) if len(sub) else float("nan"),
        })
    dist_df = pd.DataFrame(dist_rows)
    dist_df.to_csv(OUT / f"repliseq_quintile_distribution_{label}.csv", index=False)

    # Concentration
    conc_rows = [concentration_top1pct(panel, h) for h in HEADS]
    conc_df = pd.concat(conc_rows, ignore_index=True)
    conc_df["cellline"] = label
    conc_df.to_csv(OUT / f"repliseq_top1pct_concentration_{label}.csv", index=False)

    # Stratified lift
    lift_rows = []
    rng = np.random.default_rng(SEED)
    for head in HEADS:
        for filt in FILTERS:
            filter_col = "is_TCW_nonCpG" if filt == "TCW_nonCpG" else None
            for q in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
                qsub = panel[panel["repliseq_quintile"] == q]
                seed_local = int(rng.integers(0, 2**31 - 1))
                if filt == "all_CT":
                    res = panel_lift_in_subset(qsub, head, None, TOP_PCT, seed_local)
                else:
                    res = panel_lift_in_subset(qsub, head, filter_col, TOP_PCT, seed_local)
                row = {
                    "cellline": label, "head": head, "filter": filt, "quintile": q,
                    "n_panel_positions": res["n_panel_positions"],
                    "n_top1pct_in_quintile": res["n_top1pct_in_quintile"],
                    "mean_recall": res["mean_recall"],
                    "lift_vs_random": res["lift_vs_random"],
                    "ci_lo": res["ci_lo"], "ci_hi": res["ci_hi"],
                }
                for c in CANCERS:
                    row[f"recall_{c}"] = res["per_cancer_recalls"].get(c, float("nan"))
                lift_rows.append(row)
    lift_df = pd.DataFrame(lift_rows)
    lift_df.to_csv(OUT / f"repliseq_lift_by_quintile_{label}.csv", index=False)

    log.info(
        "[%s] lift summary:\n%s",
        label,
        lift_df[["head", "filter", "quintile", "lift_vs_random", "ci_lo", "ci_hi"]]
            .to_string(index=False),
    )

    return {"label": label, "lift_df": lift_df, "conc_df": conc_df, "dist_df": dist_df}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bigwig", type=Path, required=True, help="Path to bigWig file")
    p.add_argument("--label", type=str, required=True, help="Cell line label (e.g. HepG2)")
    args = p.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    base_panel = get_base_panel()
    run_one_cellline(args.label, args.bigwig, base_panel)
    log.info("DONE: %s", args.label)


if __name__ == "__main__":
    main()
