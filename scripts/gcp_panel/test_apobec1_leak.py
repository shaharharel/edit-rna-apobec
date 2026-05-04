"""Leave-leak-out test for the APOBEC1 v4_cds head.

Goal
----
The retrained APOBEC1 head (`score_apobec1_v4_cds`) wins COADREAD and ESCA in
both PCAWG and POG570 cohorts. Is the win biological signal, or partial
memorization of training positions that happen to recur in cancer mutations?

For the A3A head we ran an analogous test (5.93% training overlap, leave-leak
delta 0.00 pp). We need the equivalent number for APOBEC1.

Key fact about APOBEC1 v4_cds training data
--------------------------------------------
The training CSV (`data/raw/apobec1/v4/apobec1_v4_cds_with_negatives.csv`)
contains 484 positives + 484 negatives, all with:

    build              == "mm9"
    coordinate_system  == "mm9"
    species            == "mouse"

i.e. the model was trained entirely on mouse genome coordinates. Cancer
mutations in PCAWG / TCGA / POG570 are all on the human genome (GRCh37/hg19).
Direct (chrom, pos) overlap is therefore 0 by construction, because chr8:64516163
in mm9 is a different genomic position from chr8:64516163 in hg19.

The leave-leak-out test on raw coordinates is consequently a *lower bound*: it
rules out direct coordinate-memorization but not sequence-level memorization
at orthologous loci. We document both.

Procedure
---------
1. Load APOBEC1 v4_cds positives.
2. Survey: n, coordinate system, trinuc distribution, source-type breakdown.
3. Build training-position keys two ways:
     (a) raw mm9 keys (chrom, pos)        — measures direct memorization
     (b) mm9 → hg19 lift via ortholog
         (we *attempt* a naive same-chrom + nearest-position match if any
         human gene/coord could match; otherwise we skip — but this match
         is impossible without a real liftover chain, so we report it as
         "not attempted" and document the limitation).
4. Compute coordinate overlap with PCAWG/TCGA + POG570 mutations for the 4
   headline cells (PCAWG COADREAD, PCAWG ESCA, POG570 COADREAD, POG570 ESCA).
5. Identify panel positions whose (chrom, pos) overlap the training set
   ("leak positions"). Recompute top-1% panel using `score_apobec1_v4_cds`
   with leak positions excluded; recompute OR.
6. Output per-cell delta and verdict.

Reproducibility: seed = 20260427 (no random selection used; recorded for log).
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path("/Users/shaharharel/Documents/github/edit-rna-apobec")
TRAIN_CSV = ROOT / "data/raw/apobec1/v4/apobec1_v4_cds_with_negatives.csv"
PANEL_PARQUET = ROOT / (
    "experiments/multi_enzyme/outputs/pcawg_tcw_panel/v4_outputs/"
    "panel_scores_v4_cds_apobec1retrained.parquet"
)
PCAWG_DIR = ROOT / "data/raw/pcawg/by_cancer"
TCGA_DIR = ROOT / "data/raw/tcga"
POG570_PATH = ROOT / "data/raw/pog570/POG570_small_mutations.txt.gz"
HG19 = ROOT / "data/raw/genomes/hg19.fa"
EXISTING_PCAWG_CSV = ROOT / (
    "experiments/multi_enzyme/outputs/pcawg_tcw_panel/v4_outputs/"
    "per_cancer_enrichment_v4_pcawg.csv"
)
EXISTING_POG570_CSV = ROOT / (
    "experiments/multi_enzyme/outputs/pcawg_tcw_panel/v4_outputs/"
    "per_cancer_enrichment_v4_pog570.csv"
)

OUT_DIR = ROOT / "experiments/multi_enzyme/outputs/pcawg_tcw_panel/v4_outputs"
OUT_CSV = OUT_DIR / "apobec1_leave_leak_out.csv"
OUT_MD = OUT_DIR / "APOBEC1_LEAVE_LEAK_OUT.md"

HEAD = "score_apobec1_v4_cds"
PANEL_PCT = 0.01
SEED = 20260427

POG_COHORT_MAP = {
    "COLO": "coadread",
    "ESCA": "esca",
}

CELLS = [
    ("PCAWG", "coadread"),
    ("PCAWG", "esca"),
    ("POG570", "coadread"),
    ("POG570", "esca"),
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("apobec1_leak")


def or_fisher(a: int, b: int, c: int, d: int) -> dict:
    """OR with Haldane-Anscombe correction + Fisher one-sided p + log-CI."""
    try:
        _, p = stats.fisher_exact([[a, b], [c, d]], alternative="greater")
    except Exception:
        p = float("nan")
    if a == 0 or b == 0 or c == 0 or d == 0:
        af, bf, cf, df_ = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    else:
        af, bf, cf, df_ = float(a), float(b), float(c), float(d)
    or_val = (af * df_) / (bf * cf)
    se = np.sqrt(1.0 / af + 1.0 / bf + 1.0 / cf + 1.0 / df_)
    log_or = np.log(or_val)
    return {
        "a": int(a), "b": int(b), "c": int(c), "d": int(d),
        "OR": float(or_val),
        "OR_ci_lo": float(np.exp(log_or - 1.96 * se)),
        "OR_ci_hi": float(np.exp(log_or + 1.96 * se)),
        "p_fisher": float(p),
    }


def load_training_positives() -> pd.DataFrame:
    df = pd.read_csv(TRAIN_CSV)
    pos = df[df["is_edited"] == 1].copy()
    pos["chr"] = pos["chr"].astype(str)
    pos.loc[~pos["chr"].str.startswith("chr"), "chr"] = "chr" + pos["chr"]
    pos["pos_0based"] = pos["start"].astype(int) - 1
    return pos


def load_pcawg_combined() -> pd.DataFrame:
    """Load PCAWG + TCGA C>T/G>A SNVs for the two headline cancers only."""
    cancers = ["coadread", "esca"]
    rows = []
    for cancer in cancers:
        for path, src in [
            (PCAWG_DIR / f"{cancer}_pcawg_mutations.txt", "pcawg"),
            (TCGA_DIR / f"{cancer}_tcga_pan_can_atlas_2018_mutations.txt", "tcga"),
        ]:
            if not path.exists():
                log.warning("missing: %s", path)
                continue
            d = pd.read_csv(path, sep="\t", low_memory=False)
            d = d[d.get("Variant_Type", "SNP") == "SNP"]
            ref = d.get("Reference_Allele")
            alt = d.get("Tumor_Seq_Allele2", d.get("Tumor_Seq_Allele"))
            if ref is None or alt is None:
                continue
            mask = ((ref == "C") & (alt == "T")) | ((ref == "G") & (alt == "A"))
            d = d[mask].copy()
            d["strand"] = np.where(
                (d["Reference_Allele"] == "C") & (d["Tumor_Seq_Allele2"] == "T"),
                "+", "-",
            )
            d["pos"] = d["Start_Position"].astype(int) - 1
            d["chrom"] = d["Chromosome"].astype(str)
            d.loc[~d["chrom"].str.startswith("chr"), "chrom"] = "chr" + d["chrom"]
            d["cancer"] = cancer
            d["source"] = src
            rows.append(d[["chrom", "pos", "strand", "cancer", "source"]])
    combined = pd.concat(rows, ignore_index=True)
    valid_chroms = {f"chr{i}" for i in range(1, 23)} | {"chrX", "chrY"}
    combined = combined[combined["chrom"].isin(valid_chroms)].reset_index(drop=True)
    log.info("PCAWG+TCGA combined C>T/G>A: %d, cancers=%s",
             len(combined), combined["cancer"].unique().tolist())
    return combined


def load_pog570() -> pd.DataFrame:
    df = pd.read_csv(POG570_PATH, sep="\t", compression="gzip", low_memory=False)
    df = df[(df["ref"].str.len() == 1) & (df["alt"].str.len() == 1)]
    mask = ((df["ref"] == "C") & (df["alt"] == "T")) | (
        (df["ref"] == "G") & (df["alt"] == "A")
    )
    df = df[mask].copy()
    df["chrom"] = "chr" + df["chrom"].astype(str)
    df["pos"] = pd.to_numeric(df["pos"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["pos"])
    df["pos"] = df["pos"].astype(int) - 1
    df["strand"] = np.where(df["ref"] == "C", "+", "-")
    df["cancer"] = df["analysis_cohort"].map(POG_COHORT_MAP)
    df = df.dropna(subset=["cancer"]).reset_index(drop=True)
    log.info("POG570 C>T/G>A SNVs in COADREAD+ESCA: %d, cancers=%s",
             len(df), df["cancer"].unique().tolist())
    return df[["chrom", "pos", "strand", "cancer"]]


def annotate_panel_tcw(panel: pd.DataFrame) -> pd.DataFrame:
    """Annotate panel positions with is_TCW_C / is_CpG via hg19 reference.

    Same logic as scripts/gcp_panel/per_cancer_enrichment_v4.py.
    """
    from pyfaidx import Fasta
    log.info("Annotating panel with hg19 trinuc context (n=%d)", len(panel))
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

    out = panel.copy()
    out["is_cpg"] = is_cpg
    out["is_TCW_C"] = is_tcw_c
    out["is_TCW_nonCpG"] = is_tcw_c & ~is_cpg
    log.info("  TCW_nonCpG panel positions: %d (%.2f%%)",
             out["is_TCW_nonCpG"].sum(), 100 * out["is_TCW_nonCpG"].mean())
    return out


def annotate_mut_tcw(maf: pd.DataFrame) -> pd.DataFrame:
    from pyfaidx import Fasta
    log.info("Annotating muts with hg19 trinuc context (n=%d)", len(maf))
    genome = Fasta(str(HG19), as_raw=False, sequence_always_upper=True)
    n = len(maf)
    is_tcw = np.zeros(n, dtype=bool)
    is_cpg = np.zeros(n, dtype=bool)

    chroms = maf["chrom"].to_numpy()
    poses = maf["pos"].astype(int).to_numpy()
    strands = maf["strand"].to_numpy()
    idx_all = np.arange(n)

    for ch in pd.Series(chroms).unique():
        mask = chroms == ch
        idx = idx_all[mask]
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
        center = seq[ps_ok]
        right = seq[ps_ok + 1]
        is_plus = ss_ok == "+"
        is_minus = ~is_plus
        plus_tcw = (
            is_plus & (left == ord("T")) & (center == ord("C"))
            & ((right == ord("A")) | (right == ord("T")))
        )
        minus_tcw = (
            is_minus & (right == ord("A")) & (center == ord("G"))
            & ((left == ord("A")) | (left == ord("T")))
        )
        plus_cpg = is_plus & (center == ord("C")) & (right == ord("G"))
        minus_cpg = is_minus & (center == ord("G")) & (left == ord("C"))
        is_tcw[valid_idx] = plus_tcw | minus_tcw
        is_cpg[valid_idx] = plus_cpg | minus_cpg

    out = maf.copy()
    out["is_TCW"] = is_tcw
    out["is_CpG"] = is_cpg
    out["is_TCW_nonCpG"] = is_tcw & ~is_cpg
    return out


def main() -> None:
    log.info("Seed: %d (no random selection used)", SEED)
    np.random.seed(SEED)

    # ---------- 1. Survey training data ---------- #
    train_pos = load_training_positives()
    log.info("APOBEC1 v4_cds training positives: n=%d", len(train_pos))
    log.info("  build:   %s", train_pos["build"].unique().tolist())
    log.info("  coord:   %s", train_pos["coordinate_system"].unique().tolist())
    log.info("  species: %s", train_pos["species"].unique().tolist())

    train_full = pd.read_csv(TRAIN_CSV)
    n_neg = (train_full["is_edited"] == 0).sum()
    log.info("  n_neg in training:    %d", n_neg)
    trinuc_dist = train_pos["trinuc"].value_counts().to_dict()
    log.info("  trinuc dist (top 5): %s", list(trinuc_dist.items())[:5])

    # source_type / dataset_source breakdown for annotation context
    ds_counts = train_pos["dataset_source"].value_counts()
    log.info("  top dataset_sources: %s", ds_counts.head(5).to_dict())

    # ---------- 2. Coordinate-overlap test ---------- #
    log.info("=" * 70)
    log.info("Coordinate-overlap test (raw, no liftover)")
    log.info("=" * 70)
    log.info(
        "Training is mouse mm9; cancer mutations are human GRCh37. "
        "Direct (chrom, pos) overlap is bounded above by 0 by construction."
    )

    train_keys = set(zip(
        train_pos["chr"].astype(str).values,
        train_pos["pos_0based"].astype(int).values,
    ))
    log.info("  unique training positions: %d", len(train_keys))

    pcawg = load_pcawg_combined()
    pog = load_pog570()

    overlap_per_source = {}
    for cohort_label, df_, sub_filter in [
        ("PCAWG_COADREAD",
         pcawg[(pcawg["cancer"] == "coadread") & (pcawg["source"] == "pcawg")],
         "pcawg"),
        ("PCAWG_ESCA",
         pcawg[(pcawg["cancer"] == "esca") & (pcawg["source"] == "pcawg")],
         "pcawg"),
        ("TCGA_COADREAD",
         pcawg[(pcawg["cancer"] == "coadread") & (pcawg["source"] == "tcga")],
         "tcga"),
        ("TCGA_ESCA",
         pcawg[(pcawg["cancer"] == "esca") & (pcawg["source"] == "tcga")],
         "tcga"),
        ("POG570_COADREAD", pog[pog["cancer"] == "coadread"], "pog"),
        ("POG570_ESCA", pog[pog["cancer"] == "esca"], "pog"),
    ]:
        keys_ = set(zip(
            df_["chrom"].astype(str).values,
            df_["pos"].astype(int).values,
        ))
        ov = train_keys & keys_
        overlap_per_source[cohort_label] = {
            "n_mut_unique_pos": len(keys_),
            "n_overlap": len(ov),
            "pct_train_in_source": (
                100.0 * len(ov) / len(train_keys) if train_keys else 0.0
            ),
        }
        log.info(
            "  %s: muts_unique=%d, train_∩=%d (%.4f%% of training pos)",
            cohort_label, len(keys_), len(ov),
            overlap_per_source[cohort_label]["pct_train_in_source"],
        )

    union_keys = set()
    for label, df_, _ in [
        ("PCAWG_COADREAD",
         pcawg[(pcawg["cancer"] == "coadread") & (pcawg["source"] == "pcawg")], 0),
        ("PCAWG_ESCA",
         pcawg[(pcawg["cancer"] == "esca") & (pcawg["source"] == "pcawg")], 0),
        ("TCGA_COADREAD",
         pcawg[(pcawg["cancer"] == "coadread") & (pcawg["source"] == "tcga")], 0),
        ("TCGA_ESCA",
         pcawg[(pcawg["cancer"] == "esca") & (pcawg["source"] == "tcga")], 0),
        ("POG570_COADREAD", pog[pog["cancer"] == "coadread"], 0),
        ("POG570_ESCA", pog[pog["cancer"] == "esca"], 0),
    ]:
        union_keys |= set(zip(
            df_["chrom"].astype(str).values,
            df_["pos"].astype(int).values,
        ))
    union_overlap = train_keys & union_keys
    union_pct = 100.0 * len(union_overlap) / len(train_keys) if train_keys else 0.0
    log.info(
        "Total training positions in *any* of the 6 cancer sources: %d / %d (%.4f%%)",
        len(union_overlap), len(train_keys), union_pct,
    )

    # ---------- 3. Panel-level leak (training ∩ panel) ---------- #
    log.info("Loading panel scores ...")
    panel = pd.read_parquet(PANEL_PARQUET)
    log.info("  panel shape: %s, columns: %s", panel.shape, list(panel.columns))
    panel_keys = pd.DataFrame({
        "chrom": panel["chrom"].astype(str).values,
        "pos": panel["pos"].astype(int).values,
        "_uidx": np.arange(len(panel)),
    })
    train_df = pd.DataFrame({
        "chrom": [c for c, _ in train_keys],
        "pos": [p for _, p in train_keys],
    })
    train_df["_train"] = True
    merged = panel_keys.merge(train_df, on=["chrom", "pos"], how="left")
    is_leak = merged["_train"].fillna(False).to_numpy().astype(bool)
    n_leak = int(is_leak.sum())
    log.info(
        "Panel positions matching APOBEC1 training positions: %d / %d (%.6f%%)",
        n_leak, len(panel), 100.0 * n_leak / len(panel),
    )

    # ---------- 4. Annotate panel + muts with TCW context ---------- #
    panel = annotate_panel_tcw(panel)
    pcawg = annotate_mut_tcw(pcawg)
    pog = annotate_mut_tcw(pog)

    # ---------- 5. Per-cell leave-leak-out OR ---------- #
    n_units = len(panel)
    score = panel[HEAD].to_numpy(dtype=np.float64)
    panel_lookup = pd.DataFrame({
        "chrom": panel["chrom"].astype(str).values,
        "pos": panel["pos"].astype(int).values,
        "_uidx": np.arange(n_units),
    })

    # Top-k full
    k_full = max(1, int(round(n_units * PANEL_PCT)))
    top_full_idx = np.argpartition(-score, k_full - 1)[:k_full]
    top_full_mask = np.zeros(n_units, dtype=bool)
    top_full_mask[top_full_idx] = True

    # Top-k excluding leak positions: zero-out leak scores so they cannot rank in
    score_excl = score.copy()
    score_excl[is_leak] = -np.inf
    # Also reduce the eligible panel size by n_leak: the leave-leak-out panel
    # is the original panel minus leak positions. Top-1% should be 1% of
    # that reduced panel size.
    n_units_excl = n_units - n_leak
    k_excl = max(1, int(round(n_units_excl * PANEL_PCT)))
    if k_excl >= n_units_excl:
        k_excl = n_units_excl - 1
    top_excl_idx = np.argpartition(-score_excl, k_excl - 1)[:k_excl]
    top_excl_mask = np.zeros(n_units, dtype=bool)
    top_excl_mask[top_excl_idx] = True

    log.info(
        "Top-1%% selection sizes: full=%d, leave-leak-out=%d, leak in full top1=%d",
        k_full, k_excl, int(is_leak[top_full_idx].sum()),
    )

    # Mutation maps per (cohort, cancer) restricted to TCW_nonCpG, in panel
    rows = []
    for cohort, cancer in CELLS:
        if cohort == "PCAWG":
            mdf = pcawg[(pcawg["cancer"] == cancer) & pcawg["is_TCW_nonCpG"]]
        else:
            mdf = pog[(pog["cancer"] == cancer) & pog["is_TCW_nonCpG"]]

        m = mdf[["chrom", "pos"]].copy()
        m["pos"] = m["pos"].astype(int)
        m = m.merge(panel_lookup, on=["chrom", "pos"], how="inner")
        mut_uidx = np.unique(m["_uidx"].astype(int).to_numpy())
        n_mut_total_full = len(mut_uidx)

        # ---- FULL panel OR (matches per_cancer_enrichment_v4 logic) ----
        mut_mask = np.zeros(n_units, dtype=bool)
        mut_mask[mut_uidx] = True
        a_full = int((top_full_mask & mut_mask).sum())
        b_full = k_full - a_full
        c_full = n_mut_total_full - a_full
        d_full = (n_units - k_full) - c_full
        st_full = or_fisher(a_full, b_full, c_full, d_full)

        # ---- LEAVE-LEAK-OUT OR ----
        # Drop leak panel positions from BOTH the panel and the mutations.
        keep_mask = ~is_leak  # eligible panel positions
        # Mutations: drop those whose panel position is a leak position
        mut_uidx_excl = mut_uidx[~is_leak[mut_uidx]]
        n_mut_total_excl = len(mut_uidx_excl)

        mut_mask_excl = np.zeros(n_units, dtype=bool)
        mut_mask_excl[mut_uidx_excl] = True

        # Top-k_excl already restricted to non-leak positions
        a_excl = int((top_excl_mask & mut_mask_excl).sum())
        b_excl = k_excl - a_excl
        c_excl = n_mut_total_excl - a_excl
        d_excl = (n_units_excl - k_excl) - c_excl
        st_excl = or_fisher(a_excl, b_excl, c_excl, d_excl)

        n_mut_dropped = n_mut_total_full - n_mut_total_excl
        rel_change = (
            (st_excl["OR"] - st_full["OR"]) / st_full["OR"]
            if st_full["OR"] > 0 else float("nan")
        )
        delta = st_excl["OR"] - st_full["OR"]

        # Verdict
        if abs(delta) < 0.5:
            verdict = "PASS"
        elif abs(delta) < 1.0:
            verdict = "WARNING"
        else:
            verdict = "FAIL"

        rows.append({
            "cohort": cohort,
            "cancer": cancer,
            "head": HEAD,
            "panel_pct": PANEL_PCT,
            "filter": "filter_TCW_nonCpG",
            "n_units_full": n_units,
            "n_units_excl": n_units_excl,
            "k_full": k_full,
            "k_excl": k_excl,
            "n_leak_panel_pos": n_leak,
            "pct_panel_excluded": 100.0 * n_leak / n_units,
            "n_mut_total_full": n_mut_total_full,
            "n_mut_total_excl": n_mut_total_excl,
            "n_mut_excluded": n_mut_dropped,
            "pct_mut_excluded": (
                100.0 * n_mut_dropped / n_mut_total_full
                if n_mut_total_full else 0.0
            ),
            "n_mut_in_panel_full": st_full["a"],
            "n_mut_in_panel_excl": st_excl["a"],
            "OR_full": st_full["OR"],
            "OR_full_ci_lo": st_full["OR_ci_lo"],
            "OR_full_ci_hi": st_full["OR_ci_hi"],
            "p_fisher_full": st_full["p_fisher"],
            "OR_excl": st_excl["OR"],
            "OR_excl_ci_lo": st_excl["OR_ci_lo"],
            "OR_excl_ci_hi": st_excl["OR_ci_hi"],
            "p_fisher_excl": st_excl["p_fisher"],
            "delta_OR": delta,
            "rel_change_OR_pct": 100.0 * rel_change,
            "verdict": verdict,
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_CSV, index=False)
    log.info("Wrote %s", OUT_CSV)

    # ---------- 6. Markdown report ---------- #
    write_markdown(
        out_df,
        train_pos,
        n_neg,
        trinuc_dist,
        ds_counts,
        overlap_per_source,
        union_overlap,
        union_pct,
        n_leak,
        n_units,
    )
    log.info("Wrote %s", OUT_MD)

    # ---------- 7. Console summary ---------- #
    print("\n" + "=" * 72)
    print("APOBEC1 v4_cds leave-leak-out summary")
    print("=" * 72)
    print(
        f"Training positives: {len(train_pos)}, build=mm9 (mouse). "
        f"Direct coordinate overlap with human cancer mutations: "
        f"{len(union_overlap)} / {len(train_pos)} ({union_pct:.4f}%)."
    )
    for r in rows:
        print(
            f"  [{r['cohort']:6s} {r['cancer']:8s}] "
            f"OR_full={r['OR_full']:.3f} "
            f"OR_excl={r['OR_excl']:.3f} "
            f"delta={r['delta_OR']:+.3f} ({r['rel_change_OR_pct']:+.2f}%) "
            f"-> {r['verdict']}"
        )


def write_markdown(
    out_df: pd.DataFrame,
    train_pos: pd.DataFrame,
    n_neg: int,
    trinuc_dist: dict,
    ds_counts: pd.Series,
    overlap_per_source: dict,
    union_overlap: set,
    union_pct: float,
    n_leak: int,
    n_units: int,
) -> None:
    lines: list[str] = []
    lines.append("# APOBEC1 v4_cds leave-leak-out test")
    lines.append("")
    lines.append(
        "**Goal**: assess whether the APOBEC1 head's COADREAD/ESCA wins in "
        "PCAWG and POG570 are biological signal or partial memorization of "
        "training positions that recur in cancer mutations."
    )
    lines.append("")
    lines.append(f"Seed: `{SEED}` (recorded; no random selection used).")
    lines.append("")
    lines.append("## 1. Training-data survey (apobec1_v4_cds)")
    lines.append("")
    lines.append(f"- Positives: **{len(train_pos)}**")
    lines.append(f"- Negatives: **{n_neg}**")
    lines.append(
        "- Coordinate system: **mm9 (mouse)** — `build`, `coordinate_system`, "
        "`species` columns are uniformly mm9/mm9/mouse for every row."
    )
    lines.append(
        "- Source datasets: Rosenberg2011 (T3/T4 3'UTR + CDS) and Blanc2014 "
        "(intestinal/liver/hyperediting), all mouse Apobec1 transcriptomes."
    )
    lines.append("")
    lines.append("Top dataset sources (positives only):")
    lines.append("")
    lines.append("| dataset_source | n |")
    lines.append("|---|---|")
    for k, v in ds_counts.head(8).items():
        lines.append(f"| `{k}` | {v} |")
    lines.append("")
    lines.append("Trinucleotide distribution (positives, top 8):")
    lines.append("")
    lines.append("| trinuc | count | % |")
    lines.append("|---|---|---|")
    n_pos = len(train_pos)
    for tri, cnt in list(trinuc_dist.items())[:8]:
        lines.append(f"| {tri} | {cnt} | {100*cnt/n_pos:.1f}% |")
    lines.append("")
    lines.append(
        "> **Critical**: the APOBEC1 v4_cds head was trained entirely on "
        "**mouse mm9** coordinates. Cancer mutations from PCAWG, TCGA-MC3, "
        "and POG570 are all human GRCh37/hg19 coordinates. A direct "
        "(chrom, pos) match between training positives and human mutations "
        "is **mathematically impossible** — chr1:12345 in mm9 is not the "
        "same locus as chr1:12345 in hg19. The coordinate-overlap test below "
        "therefore measures direct memorization only and is a *lower bound* "
        "for sequence-level memorization at orthologous loci (which would "
        "require an mm9→hg19 liftover; see Limitations)."
    )
    lines.append("")
    lines.append("## 2. Coordinate overlap with cancer-mutation sources")
    lines.append("")
    lines.append("Per-source unique-position overlap with training positives:")
    lines.append("")
    lines.append("| Source | unique mut positions | training ∩ source | % of training |")
    lines.append("|---|---:|---:|---:|")
    for k, v in overlap_per_source.items():
        lines.append(
            f"| {k} | {v['n_mut_unique_pos']} | {v['n_overlap']} "
            f"| {v['pct_train_in_source']:.4f}% |"
        )
    lines.append(
        f"| **Union (any of 6)** | — | **{len(union_overlap)}** "
        f"| **{union_pct:.4f}%** |"
    )
    lines.append("")
    lines.append(
        f"Panel positions whose (chrom, pos) match training positives: "
        f"**{n_leak} / {n_units}** ({100.0 * n_leak / n_units:.6f}% of panel)."
    )
    lines.append("")
    lines.append("## 3. Leave-leak-out per cell")
    lines.append("")
    lines.append(
        "All cells use top-1% of `score_apobec1_v4_cds`, filter "
        "`filter_TCW_nonCpG`, Haldane-Anscombe-corrected OR with "
        "log-CI and one-sided Fisher exact (greater)."
    )
    lines.append("")
    lines.append(
        "| Cohort | Cancer | n_panel_excluded | n_muts_excluded | "
        "OR_full (95% CI) | OR_leave_leak (95% CI) | ΔOR | rel ΔOR | Verdict |"
    )
    lines.append(
        "|---|---|---:|---:|---|---|---:|---:|---|"
    )
    for _, r in out_df.iterrows():
        lines.append(
            f"| {r['cohort']} | {r['cancer']} | "
            f"{int(r['n_leak_panel_pos'])} ({r['pct_panel_excluded']:.4f}%) | "
            f"{int(r['n_mut_excluded'])} ({r['pct_mut_excluded']:.4f}%) | "
            f"{r['OR_full']:.3f} ({r['OR_full_ci_lo']:.2f}–{r['OR_full_ci_hi']:.2f}) | "
            f"{r['OR_excl']:.3f} ({r['OR_excl_ci_lo']:.2f}–{r['OR_excl_ci_hi']:.2f}) | "
            f"{r['delta_OR']:+.3f} | {r['rel_change_OR_pct']:+.2f}% | "
            f"{r['verdict']} |"
        )
    lines.append("")
    lines.append("Verdict thresholds (per spec):")
    lines.append("")
    lines.append("- **PASS**: |ΔOR| < 0.5 (≈ <12% relative change) — biological signal real")
    lines.append("- **WARNING**: 0.5 ≤ |ΔOR| < 1.0 — partial memorization; footnote required")
    lines.append("- **FAIL**: |ΔOR| ≥ 1.0 — claim is largely memorization")
    lines.append("")
    lines.append("## 4. Discussion: does the APOBEC1 → intestinal/oesophageal claim survive?")
    lines.append("")
    n_pass = (out_df["verdict"] == "PASS").sum()
    if n_pass == len(out_df):
        lines.append(
            f"**All {len(out_df)} cells PASS the coordinate-leak test.** "
            "Across PCAWG COADREAD, PCAWG ESCA, POG570 COADREAD and POG570 "
            "ESCA, removing panel positions whose hg19 coordinates match "
            "the (mouse) training positives changes the OR by at most "
            f"|ΔOR| = {out_df['delta_OR'].abs().max():.3f}. "
            "The APOBEC1 head's enrichment in COADREAD and ESCA is therefore "
            "**not driven by direct coordinate memorization**, which is "
            "consistent with the fact that mouse and human coordinates are "
            "non-overlapping by construction."
        )
    else:
        lines.append(
            f"**{n_pass} of {len(out_df)} cells PASS.** Cells with WARNING "
            "or FAIL verdicts indicate possible memorization and are flagged "
            "in the table above."
        )
    lines.append("")
    lines.append("### Why was the coordinate overlap zero?")
    lines.append("")
    lines.append(
        "The v4_cds APOBEC1 head was trained on 484 mouse-validated APOBEC1 "
        "edited sites from Rosenberg 2011 and Blanc 2014, all annotated in "
        "mm9 coordinates. The panel and the cancer-mutation MAFs are GRCh37. "
        "A position like `chr8:64516163` in mm9 (Apob mRNA region) refers to "
        "a different physical locus from `chr8:64516163` in hg19. The two "
        "coordinate systems share no genomic positions; the model has never "
        "seen any human GRCh37 coordinate during training."
    )
    lines.append("")
    lines.append("### Limitations: what this test does *not* rule out")
    lines.append("")
    lines.append(
        "1. **Sequence-level memorization at orthologs.** If the model learned "
        "    flanking-sequence + structure features around mouse Apobec1 "
        "    targets and the human ortholog of that gene exists at hg19 "
        "    coordinates with similar local sequence, the model could score "
        "    that human position highly because the *features* match, not "
        "    because the coordinate matches. Detecting that would require "
        "    mm9→hg19 liftover via UCSC chains and re-running the test on "
        "    lifted positions. This is not implemented here."
        ""
    )
    lines.append(
        "2. **Trinucleotide / k-mer leakage.** APOBEC1 positives are "
        "    enriched for ACA / ACT / TCA / TCT contexts (mooring-rich "
        "    pattern). If panel-recall correlates with these contexts and "
        "    the contexts themselves correlate with cancer mutability, that "
        "    is biological signal mediated by sequence preference, not "
        "    coordinate memorization."
    )
    lines.append("")
    lines.append("### Bottom line")
    lines.append("")
    if n_pass == len(out_df):
        lines.append(
            "The APOBEC1 head's COADREAD/ESCA enrichment in PCAWG and POG570 "
            "is **defensible** against the coordinate-memorization "
            "hypothesis: there is literally no shared coordinate between the "
            "(mouse) training set and the (human) evaluation set. The "
            "remaining hypothesis to test in future work is sequence/feature "
            "leakage at orthologous loci, which would require an mm9→hg19 "
            "liftover."
        )
    else:
        lines.append(
            "The APOBEC1 enrichment is partly compromised by panel positions "
            "that overlap training (see WARNING/FAIL cells)."
        )
    lines.append("")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main() or 0)
