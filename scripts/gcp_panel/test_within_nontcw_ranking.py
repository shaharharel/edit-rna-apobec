#!/usr/bin/env python3
"""Within-non-TCW ranking test (companion to test_within_tcw_ranking.py).

Tests whether the model's ranking of NON-TCW positions captures more cancer
mutations than random selection within the non-TCW pool.

Hypothesis: TCW positions are roughly homogeneous (all editable by APOBEC), so
within-TCW ranking has limited room to add signal. Non-TCW positions are
heterogeneous: most never mutate, but a subset (likely those with distinctive
structural / contextual features) accumulate mutations.  If the model
identifies these exceptional non-TCW hotspots, the within-non-TCW lift will be
much higher than the within-TCW lift.

Inputs:
  experiments/multi_enzyme/outputs/pcawg_tcw_panel/v4_outputs/panel_scores_v4_cds_apobec1retrained.parquet
  data/raw/genomes/hg19.fa
  data/raw/tcga/, data/raw/pcawg/by_cancer/   (10-cancer pan-cancer C>T)

Outputs:
  experiments/multi_enzyme/outputs/pcawg_tcw_panel/v4_outputs/within_nontcw_test.csv
  experiments/multi_enzyme/outputs/pcawg_tcw_panel/v4_outputs/within_nontcw_test_summary.csv
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/shaharharel/Documents/github/edit-rna-apobec")
PANEL = ROOT / "experiments/multi_enzyme/outputs/pcawg_tcw_panel/v4_outputs/panel_scores_v4_cds_apobec1retrained.parquet"
HG19 = ROOT / "data/raw/genomes/hg19.fa"
TCGA = ROOT / "data/raw/tcga"
PCAWG = ROOT / "data/raw/pcawg/by_cancer"
OUT_DIR = ROOT / "experiments/multi_enzyme/outputs/pcawg_tcw_panel/v4_outputs"
OUT_CSV = OUT_DIR / "within_nontcw_test.csv"
OUT_SUMMARY = OUT_DIR / "within_nontcw_test_summary.csv"

CANCERS = ["blca", "brca", "cesc", "coadread", "esca", "hnsc", "lihc", "lusc", "skcm", "stad"]
HEADS = ["score_binary", "score_A3A", "score_A3B", "score_A3G", "score_A3A_A3G", "score_apobec1_v4_cds"]
TOP_PCTS = [0.01, 0.05, 0.10]
N_RANDOM_DRAWS = 1000
N_BOOT = 10000
SEED = 20260427

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    stream=sys.stdout)
log = logging.getLogger(__name__)


def annotate_tcw(panel: pd.DataFrame) -> np.ndarray:
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
        seq = np.frombuffer(str(genome[ch][:]).upper().encode("ascii"), dtype=np.uint8)
        L = len(seq)
        ps = poses[mask]; ss = strands[mask]
        ok = (ps >= 1) & (ps + 1 < L)
        valid = i_ch[ok]; ps_ok = ps[ok]; ss_ok = ss[ok]
        left = seq[ps_ok - 1]; right = seq[ps_ok + 1]
        is_plus = ss_ok == "+"; is_minus = ~is_plus
        right_AT = (right == ord("A")) | (right == ord("T"))
        left_AT = (left == ord("A")) | (left == ord("T"))
        is_tcw_c[valid[is_plus]] = (left[is_plus] == ord("T")) & right_AT[is_plus]
        is_tcw_c[valid[is_minus]] = (right[is_minus] == ord("A")) & left_AT[is_minus]
    log.info("  TCW positions: %d (%.2f%%); non-TCW: %d (%.2f%%)",
             is_tcw_c.sum(), 100 * is_tcw_c.mean(),
             (~is_tcw_c).sum(), 100 * (~is_tcw_c).mean())
    return is_tcw_c


def load_one_maf(path: Path, cancer: str):
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
    df["pos"] = df["Start_Position"].astype(int) - 1
    df["chrom"] = df["Chromosome"].astype(str)
    df.loc[~df["chrom"].str.startswith("chr"), "chrom"] = "chr" + df["chrom"]
    df["cancer"] = cancer
    return df[["chrom", "pos", "cancer"]]


def load_mutations() -> pd.DataFrame:
    log.info("Loading 10-cancer C>T MAFs ...")
    rows = []
    for c in CANCERS:
        for pp in [PCAWG / f"{c}_pcawg_mutations.txt",
                   TCGA / f"{c}_tcga_pan_can_atlas_2018_mutations.txt"]:
            d = load_one_maf(pp, c)
            if d is not None:
                rows.append(d)
    df = pd.concat(rows, ignore_index=True)
    valid = set([f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"])
    df = df[df["chrom"].isin(valid)]
    df = df.drop_duplicates(["chrom", "pos", "cancer"])
    log.info("  %d unique cancer C>T sites", len(df))
    return df


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading panel %s", PANEL)
    panel = pd.read_parquet(PANEL)
    n = len(panel)
    log.info("  panel n=%d", n)

    is_tcw = annotate_tcw(panel)
    nontcw_idx = np.arange(n)[~is_tcw]
    n_nontcw = len(nontcw_idx)

    muts = load_mutations()
    panel_set = set(zip(panel["chrom"].astype(str), panel["pos"].astype(int)))
    in_panel = np.array([(c, int(p)) in panel_set for c, p in zip(muts["chrom"], muts["pos"])])
    muts = muts.iloc[np.where(in_panel)[0]].reset_index(drop=True)

    pos_to_idx = {(c, int(p)): i for i, (c, p) in enumerate(
        zip(panel["chrom"].astype(str), panel["pos"].astype(int)))}
    muts["panel_idx"] = [pos_to_idx.get((c, int(p)), -1) for c, p in zip(muts["chrom"], muts["pos"])]
    muts = muts[muts["panel_idx"] >= 0].copy()
    muts["is_tcw"] = is_tcw[muts["panel_idx"].to_numpy()]
    muts_nontcw = muts[~muts["is_tcw"]].copy()
    log.info("  in-panel mutations total=%d  TCW=%d (%.1f%%)  non-TCW=%d (%.1f%%)",
             len(muts), muts["is_tcw"].sum(), 100 * muts["is_tcw"].mean(),
             (~muts["is_tcw"]).sum(), 100 * (~muts["is_tcw"]).mean())

    cancer_mut_idx = {c: muts_nontcw[muts_nontcw["cancer"] == c]["panel_idx"].to_numpy()
                      for c in CANCERS}

    rows = []
    for tp in TOP_PCTS:
        k = int(round(n_nontcw * tp))
        log.info("=== top-%.0f%% within non-TCW: k=%d ===", tp * 100, k)

        head_top_sets = {}
        for h in HEADS:
            scores = panel[h].to_numpy()[nontcw_idx]
            top_local = np.argpartition(-scores, k - 1)[:k]
            head_top_sets[h] = nontcw_idx[top_local]

        random_recalls = {c: np.zeros(N_RANDOM_DRAWS) for c in cancer_mut_idx}
        for d in range(N_RANDOM_DRAWS):
            draw = rng.choice(nontcw_idx, size=k, replace=False)
            hit = np.zeros(n, dtype=bool); hit[draw] = True
            for c, mids in cancer_mut_idx.items():
                random_recalls[c][d] = hit[mids].sum() / max(len(mids), 1)
            if d % 200 == 0:
                log.info("    random-draw %d/%d", d, N_RANDOM_DRAWS)

        for h, top_arr in head_top_sets.items():
            head_hit = np.zeros(n, dtype=bool); head_hit[top_arr] = True
            for c, mids in cancer_mut_idx.items():
                rb = random_recalls[c]
                rb_mean = rb.mean()
                hr = head_hit[mids].sum() / max(len(mids), 1)
                rows.append({
                    "top_pct": tp, "k": k, "head": h, "cancer": c,
                    "n_mutations_nontcw": int(len(mids)),
                    "head_recall": hr,
                    "random_recall_mean": rb_mean,
                    "random_recall_ci_lo": np.percentile(rb, 2.5),
                    "random_recall_ci_hi": np.percentile(rb, 97.5),
                    "lift_vs_random_within_nontcw": hr / max(rb_mean, 1e-9),
                })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_CSV, index=False)
    log.info("Wrote %s (%d rows)", OUT_CSV, len(df_out))

    log.info("Bootstrap CI across cancers ...")
    boot_rows = []
    for tp in TOP_PCTS:
        for h in HEADS:
            sub = df_out[(df_out["top_pct"] == tp) & (df_out["head"] == h)]
            ratios = sub["lift_vs_random_within_nontcw"].to_numpy()
            boot = []
            for _ in range(N_BOOT):
                s = rng.choice(ratios, size=len(ratios), replace=True)
                boot.append(s.mean())
            boot = np.asarray(boot)
            boot_rows.append({
                "top_pct": tp, "head": h, "n_cancers": len(ratios),
                "mean_lift": ratios.mean(),
                "median_lift": np.median(ratios),
                "ci_lo": np.percentile(boot, 2.5),
                "ci_hi": np.percentile(boot, 97.5),
                "n_above_1": int((ratios > 1).sum()),
            })
    boot_df = pd.DataFrame(boot_rows)
    boot_df.to_csv(OUT_SUMMARY, index=False)
    log.info("Wrote %s", OUT_SUMMARY)
    log.info("Done in %.1f s", time.time() - t0)


if __name__ == "__main__":
    main()
