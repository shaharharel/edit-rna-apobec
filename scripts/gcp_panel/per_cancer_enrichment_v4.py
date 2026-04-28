#!/usr/bin/env python3
"""Per-cancer × per-head enrichment of v4_cds panel (advisor v2 style).

For each (cancer, head, panel_size) cell compute:
    - panel_units (k)
    - n_mutations_in_panel
    - n_mutations_total
    - abs_recall = n_in_panel / n_total
    - 2x2 contingency table over panel positions (in/out_panel × mutated/not):
        a = in_panel & mutated     (mutated unique positions inside top-k)
        b = in_panel & not_mut     (k - a)
        c = out_panel & mutated    (n_mut_unique_total - a)
        d = out_panel & not_mut    ((n_units - k) - c)
      Mutations are counted *per position* (unique positions hit), so a+c is the
      number of distinct mutated panel positions for the cancer/filter.
    - Odds ratio with Haldane–Anscombe correction (add 0.5 if any cell == 0)
    - Fisher's exact test p-value
    - 95% CI on OR via log(OR) ± 1.96 * sqrt(1/a + 1/b + 1/c + 1/d)
        with HA correction applied for the SE.

Two filters: filter_TCW_nonCpG, filter_all_CT
Three panel sizes: top-1%, top-5%, top-10%
Six heads: score_binary, score_A3A, score_A3B, score_A3G, score_A3A_A3G,
           score_apobec1_v4_cds

Outputs (under v4_outputs/):
    per_cancer_enrichment_v4_pcawg.csv
    per_cancer_enrichment_v4_pog570.csv
    per_cancer_OR_pcawg_top1pct.png
    per_cancer_OR_pcawg_top5pct.png
    per_cancer_OR_pog570_top1pct.png
    per_cancer_OR_concordance_top1pct.png   (PCAWG vs POG570 scatter)
    PER_CANCER_ENRICHMENT_V4.md
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path("/Users/shaharharel/Documents/github/edit-rna-apobec")
TCGA_DIR = ROOT / "data/raw/tcga"
PCAWG_DIR = ROOT / "data/raw/pcawg/by_cancer"
HG19 = ROOT / "data/raw/genomes/hg19.fa"
POG570_PATH = ROOT / "data/raw/pog570/POG570_small_mutations.txt.gz"

PANEL_PATH = (
    ROOT
    / "experiments/multi_enzyme/outputs/pcawg_tcw_panel/v4_outputs"
    / "panel_scores_v4_cds_apobec1retrained.parquet"
)
OUT_DIR = (
    ROOT / "experiments/multi_enzyme/outputs/pcawg_tcw_panel/v4_outputs"
)

PCAWG_CANCERS = ["blca", "brca", "cesc", "coadread", "esca", "hnsc", "lihc",
                 "lusc", "skcm", "stad"]

# POG570 analysis_cohort -> reference cancer label (matches v4 sweep mapping)
POG_COHORT_MAP = {
    "COLO": "coadread",
    "SKCM": "skcm",
    "BRCA": "brca",
    "LUNG": "lusc",
    "ESCA": "esca",
    "HNSC": "hnsc",
    "STAD": "stad",
    "HCC": "lihc",
    "BLCA": "blca",
    "CERV": "cesc",
}

HEADS = [
    "score_binary",
    "score_A3A",
    "score_A3B",
    "score_A3G",
    "score_A3A_A3G",
    "score_apobec1_v4_cds",
]
TOP_PCTS = [0.01, 0.05, 0.10]
FILTERS = ["filter_TCW_nonCpG", "filter_all_CT"]

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


# =========================================================================== #
# Mutation loaders (PCAWG/TCGA + POG570) — copied from v4 sweep & POG570 v4
# =========================================================================== #


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
    df["pos"] = df["Start_Position"].astype(int) - 1  # 1-based MAF -> 0-based panel
    df["chrom"] = df["Chromosome"].astype(str)
    df.loc[~df["chrom"].str.startswith("chr"), "chrom"] = "chr" + df["chrom"]
    df["cancer"] = cancer
    df["source"] = source
    return df[["chrom", "pos", "strand", "cancer", "source"]]


def load_pcawg_combined() -> pd.DataFrame:
    log.info("Loading TCGA-MC3 + cBioPortal-PCAWG MAFs (10 cancers) ...")
    rows = []
    for cancer in PCAWG_CANCERS:
        d = _load_one_maf(
            PCAWG_DIR / f"{cancer}_pcawg_mutations.txt", cancer, "pcawg_coding"
        )
        if d is not None:
            rows.append(d)
        d = _load_one_maf(
            TCGA_DIR / f"{cancer}_tcga_pan_can_atlas_2018_mutations.txt",
            cancer, "tcga_mc3",
        )
        if d is not None:
            rows.append(d)
    if not rows:
        raise RuntimeError("No MAFs loaded")
    combined = pd.concat(rows, ignore_index=True)
    valid_chroms = set([f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"])
    combined = combined[combined["chrom"].isin(valid_chroms)]
    log.info("  combined C>T/G>A: %d, %d cancers",
             len(combined), combined["cancer"].nunique())
    return combined


def load_pog570() -> pd.DataFrame:
    log.info("Loading POG570 small mutations from %s", POG570_PATH)
    df = pd.read_csv(POG570_PATH, sep="\t", compression="gzip", low_memory=False)
    log.info("  raw rows: %d", len(df))
    df = df[(df["ref"].str.len() == 1) & (df["alt"].str.len() == 1)].copy()
    df = df[
        ((df["ref"] == "C") & (df["alt"] == "T"))
        | ((df["ref"] == "G") & (df["alt"] == "A"))
    ]
    log.info("  C>T/G>A SNVs: %d", len(df))
    df["chrom"] = "chr" + df["chrom"].astype(str)
    df["pos"] = pd.to_numeric(df["pos"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["pos"])
    df["pos"] = df["pos"].astype(int) - 1  # 1-based VCF -> 0-based panel
    df["strand"] = np.where(df["ref"] == "C", "+", "-")
    # Map cohort to reference cancer label (use mapping; if no map, keep raw)
    df["cancer"] = df["analysis_cohort"].map(POG_COHORT_MAP)
    log.info("  unique analysis_cohort labels: %s",
             df["analysis_cohort"].value_counts().head(20).to_dict())
    return df[["chrom", "pos", "ref", "alt", "strand", "cancer",
               "analysis_cohort", "patient_id"]].copy()


# =========================================================================== #
# Trinuc context annotation (panel positions and mutations)
# =========================================================================== #


def annotate_panel_positions(panel: pd.DataFrame) -> pd.DataFrame:
    from pyfaidx import Fasta
    log.info("Annotating panel positions with is_TCW_C + is_CpG ...")
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
        is_tcw_c[valid_idx[is_plus]] = (left[is_plus] == ord("T")) & right_AT[is_plus]
        is_tcw_c[valid_idx[is_minus]] = (right[is_minus] == ord("A")) & left_AT[is_minus]

    panel = panel.copy()
    panel["is_cpg"] = is_cpg
    panel["is_TCW_C"] = is_tcw_c
    log.info("  panel n=%d, is_TCW_C=%d (%.2f%%), is_CpG=%d (%.2f%%)",
             n, is_tcw_c.sum(), 100 * is_tcw_c.mean(),
             is_cpg.sum(), 100 * is_cpg.mean())
    return panel


def annotate_mut_context(maf: pd.DataFrame) -> pd.DataFrame:
    from pyfaidx import Fasta
    log.info("Annotating mutations with hg19 trinuc context (n=%d) ...", len(maf))
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
    log.info("  total=%d  TCW_nonCpG=%d (%.1f%%)  all_CT=%d",
             n, out["is_TCW_nonCpG"].sum(),
             100 * out["is_TCW_nonCpG"].mean(), n)
    return out


# =========================================================================== #
# 2x2 OR with Haldane-Anscombe correction + log-CI + Fisher's exact
# =========================================================================== #


def or_fisher(a: int, b: int, c: int, d: int) -> dict:
    """Compute Odds Ratio with Haldane-Anscombe correction, Fisher exact p,
    and 95% log-CI.

    a = in_panel  & mutated
    b = in_panel  & not mutated
    c = out_panel & mutated
    d = out_panel & not mutated
    """
    # Fisher exact on raw counts (always ints >= 0)
    try:
        _, p_fisher = stats.fisher_exact([[a, b], [c, d]], alternative="greater")
    except Exception:
        p_fisher = float("nan")
    # Haldane-Anscombe correction: if any cell == 0, add 0.5 to all
    if a == 0 or b == 0 or c == 0 or d == 0:
        af, bf, cf, df_ = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    else:
        af, bf, cf, df_ = float(a), float(b), float(c), float(d)
    or_val = (af * df_) / (bf * cf)
    se = np.sqrt(1.0 / af + 1.0 / bf + 1.0 / cf + 1.0 / df_)
    log_or = np.log(or_val)
    lo = float(np.exp(log_or - 1.96 * se))
    hi = float(np.exp(log_or + 1.96 * se))
    return {
        "a": int(a), "b": int(b), "c": int(c), "d": int(d),
        "OR": float(or_val),
        "OR_ci_lo": lo,
        "OR_ci_hi": hi,
        "p_fisher": float(p_fisher),
    }


# =========================================================================== #
# Per-cell enrichment driver
# =========================================================================== #


def compute_per_cancer_enrichment(
    panel: pd.DataFrame,
    muts_in_panel: pd.DataFrame,
    cancers: list[str],
    cohort_label: str,
) -> pd.DataFrame:
    """Compute (cancer, head, panel_size, filter) enrichment.

    Mutations are reduced to *unique panel positions hit per (cancer, filter)*.
    The 2x2 table is over panel positions (n=panel rows).
    """
    n_units = len(panel)
    log.info("[%s] n_units = %d", cohort_label, n_units)

    # panel position -> integer index (0..n-1)
    panel_lookup = pd.DataFrame({
        "chrom": panel["chrom"].astype(str).values,
        "pos": panel["pos"].astype(int).values,
        "_uidx": np.arange(n_units),
    })

    # Filter sets
    fset = {
        "filter_TCW_nonCpG": muts_in_panel[muts_in_panel["is_TCW_nonCpG"]].copy(),
        "filter_all_CT": muts_in_panel.copy(),
    }
    for fn, fdf in fset.items():
        log.info("  [%s] filter %s: %d in-panel mutations",
                 cohort_label, fn, len(fdf))

    # Build (cancer, filter) -> set of unique panel uidx hit
    # We iterate per filter once and group by cancer using merge.
    mut_uidx_per_cancer_filter: dict[tuple[str, str], set] = {}
    for filter_name, fdf in fset.items():
        m = fdf[["chrom", "pos", "cancer"]].copy()
        m["pos"] = m["pos"].astype(int)
        m = m.merge(panel_lookup, on=["chrom", "pos"], how="inner")
        for cancer in cancers:
            sub = m[m["cancer"] == cancer]
            mut_uidx_per_cancer_filter[(cancer, filter_name)] = set(
                sub["_uidx"].astype(int).tolist()
            )
            log.info(
                "    [%s/%s/%s] %d total muts -> %d unique panel positions",
                cohort_label, cancer, filter_name,
                len(sub), len(mut_uidx_per_cancer_filter[(cancer, filter_name)]),
            )

    # Pre-compute top-k indices for each head x panel_pct
    log.info("  [%s] computing top-k panel sets ...", cohort_label)
    top_idx_cache: dict[tuple[str, float], np.ndarray] = {}
    for head in HEADS:
        scores = panel[head].to_numpy(dtype=np.float64)
        for tp in TOP_PCTS:
            k = max(1, int(round(n_units * tp)))
            if k >= n_units:
                k = n_units - 1
            top_idx = np.argpartition(-scores, k - 1)[:k]
            # convert to sorted np array for set membership later
            top_set = np.zeros(n_units, dtype=bool)
            top_set[top_idx] = True
            top_idx_cache[(head, tp)] = top_set

    # Build rows
    rows = []
    for cancer in cancers:
        for filter_name in FILTERS:
            mut_uidx = mut_uidx_per_cancer_filter.get((cancer, filter_name), set())
            n_mut_total = len(mut_uidx)
            mut_mask = np.zeros(n_units, dtype=bool)
            if n_mut_total > 0:
                mut_arr = np.fromiter(mut_uidx, dtype=np.int64, count=n_mut_total)
                mut_mask[mut_arr] = True
            for head in HEADS:
                for tp in TOP_PCTS:
                    in_panel_mask = top_idx_cache[(head, tp)]
                    k = int(in_panel_mask.sum())
                    a = int((in_panel_mask & mut_mask).sum())
                    b = k - a
                    c = n_mut_total - a
                    d = (n_units - k) - c
                    abs_recall = (a / n_mut_total) if n_mut_total > 0 else float("nan")
                    if n_mut_total == 0:
                        # no mutations -> degenerate; record N/A
                        row = {
                            "cohort": cohort_label,
                            "cancer": cancer,
                            "head": head,
                            "panel_pct": tp,
                            "filter": filter_name,
                            "panel_units": k,
                            "n_units": n_units,
                            "n_mut_total": 0,
                            "n_mut_in_panel": 0,
                            "abs_recall": float("nan"),
                            "a": a, "b": b, "c": c, "d": d,
                            "OR": float("nan"),
                            "OR_ci_lo": float("nan"),
                            "OR_ci_hi": float("nan"),
                            "p_fisher": float("nan"),
                        }
                    else:
                        st = or_fisher(a, b, c, d)
                        row = {
                            "cohort": cohort_label,
                            "cancer": cancer,
                            "head": head,
                            "panel_pct": tp,
                            "filter": filter_name,
                            "panel_units": k,
                            "n_units": n_units,
                            "n_mut_total": n_mut_total,
                            "n_mut_in_panel": a,
                            "abs_recall": abs_recall,
                            "a": st["a"], "b": st["b"], "c": st["c"], "d": st["d"],
                            "OR": st["OR"],
                            "OR_ci_lo": st["OR_ci_lo"],
                            "OR_ci_hi": st["OR_ci_hi"],
                            "p_fisher": st["p_fisher"],
                        }
                    rows.append(row)
    return pd.DataFrame(rows)


# =========================================================================== #
# Heatmap plotting
# =========================================================================== #


def plot_or_heatmap(df: pd.DataFrame, panel_pct: float, filter_name: str,
                    cohort_label: str, out_path: Path):
    """Heatmap: cancers (rows) x heads (cols), value = log10(OR).

    Annotated with raw OR and stars for p-value significance.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = df[
        (df["panel_pct"].round(3) == round(panel_pct, 3))
        & (df["filter"] == filter_name)
    ].copy()
    if len(sub) == 0:
        log.warning("No data for heatmap %s/%s/%s", cohort_label,
                    panel_pct, filter_name)
        return

    # Order cancers by mean OR descending
    cancer_mean = (
        sub.groupby("cancer")["OR"].mean().sort_values(ascending=False)
    )
    cancer_order = cancer_mean.index.tolist()
    head_order = HEADS  # fixed order

    M = np.full((len(cancer_order), len(head_order)), np.nan)
    P = np.full_like(M, np.nan)
    A = np.empty_like(M, dtype=object)
    for i, cancer in enumerate(cancer_order):
        for j, head in enumerate(head_order):
            r = sub[(sub["cancer"] == cancer) & (sub["head"] == head)]
            if len(r) == 0:
                A[i, j] = ""
                continue
            M[i, j] = r.iloc[0]["OR"]
            P[i, j] = r.iloc[0]["p_fisher"]
            A[i, j] = f"{r.iloc[0]['OR']:.2f}"

    fig, ax = plt.subplots(
        figsize=(1.3 * len(head_order) + 2.5, 0.5 * len(cancer_order) + 2),
        constrained_layout=True,
    )
    log_M = np.log10(np.clip(M, 1e-3, None))
    vmax = np.nanmax(np.abs(log_M))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0
    im = ax.imshow(
        log_M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto",
    )
    ax.set_xticks(range(len(head_order)))
    ax.set_xticklabels([h.replace("score_", "") for h in head_order],
                       rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(cancer_order)))
    ax.set_yticklabels(cancer_order, fontsize=9)
    ax.set_title(
        f"Per-cancer OR — {cohort_label} v4_cds — top-{int(panel_pct*100)}% — "
        f"{filter_name}\ncolor=log10(OR), annot=OR (stars: p<1e-10***, "
        "p<1e-5**, p<0.05*)"
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("log10(OR)")

    for i in range(len(cancer_order)):
        for j in range(len(head_order)):
            if not np.isfinite(M[i, j]):
                continue
            star = ""
            if np.isfinite(P[i, j]):
                if P[i, j] < 1e-10:
                    star = "***"
                elif P[i, j] < 1e-5:
                    star = "**"
                elif P[i, j] < 0.05:
                    star = "*"
            txt_color = "white" if abs(log_M[i, j]) > 0.5 * vmax else "black"
            ax.text(j, i, f"{A[i, j]}{star}", ha="center", va="center",
                    fontsize=8, color=txt_color)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Wrote %s", out_path)


def plot_pcawg_pog570_concordance(df_p: pd.DataFrame, df_g: pd.DataFrame,
                                  panel_pct: float, filter_name: str,
                                  out_path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub_p = df_p[
        (df_p["panel_pct"].round(3) == round(panel_pct, 3))
        & (df_p["filter"] == filter_name)
    ].copy()
    sub_g = df_g[
        (df_g["panel_pct"].round(3) == round(panel_pct, 3))
        & (df_g["filter"] == filter_name)
    ].copy()
    if len(sub_p) == 0 or len(sub_g) == 0:
        log.warning("Concordance plot skipped: no rows.")
        return

    merged = sub_p.merge(
        sub_g, on=["cancer", "head", "panel_pct", "filter"],
        suffixes=("_pcawg", "_pog570"), how="inner",
    )
    merged = merged[
        np.isfinite(merged["OR_pcawg"]) & np.isfinite(merged["OR_pog570"])
        & (merged["n_mut_total_pog570"] > 0)
    ]
    if len(merged) == 0:
        log.warning("Concordance plot skipped: no overlap.")
        return

    rho, p_rho = stats.spearmanr(merged["OR_pcawg"], merged["OR_pog570"])

    fig, ax = plt.subplots(figsize=(7.5, 6.5), constrained_layout=True)
    head_colors = {
        "score_binary": "#000000",
        "score_A3A": "#d62728",
        "score_A3B": "#ff7f0e",
        "score_A3G": "#2ca02c",
        "score_A3A_A3G": "#1f77b4",
        "score_apobec1_v4_cds": "#9467bd",
    }
    for head in HEADS:
        sub_h = merged[merged["head"] == head]
        if len(sub_h) == 0:
            continue
        ax.scatter(
            sub_h["OR_pcawg"], sub_h["OR_pog570"],
            c=head_colors.get(head, "grey"), label=head.replace("score_", ""),
            s=60, alpha=0.8, edgecolor="black",
        )
    lim = max(1.5, float(merged[["OR_pcawg", "OR_pog570"]].max().max()) * 1.05)
    ax.plot([0, lim], [0, lim], "k--", alpha=0.4, lw=1)
    ax.axhline(1.0, color="grey", lw=0.6, ls=":")
    ax.axvline(1.0, color="grey", lw=0.6, ls=":")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("OR — PCAWG/TCGA")
    ax.set_ylabel("OR — POG570")
    ax.set_title(
        f"PCAWG vs POG570 per-cancer OR — top-{int(panel_pct*100)}% — "
        f"{filter_name}\nSpearman ρ={rho:.3f} (p={p_rho:.2e}, n={len(merged)})"
    )
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Wrote %s (rho=%.3f, p=%.2e, n=%d)", out_path, rho, p_rho,
             len(merged))


# =========================================================================== #
# Markdown report
# =========================================================================== #


def fmt_or_cell(or_val, ci_lo, ci_hi, p):
    if not np.isfinite(or_val):
        return "—"
    star = ""
    if np.isfinite(p):
        if p < 1e-10:
            star = "***"
        elif p < 1e-5:
            star = "**"
        elif p < 0.05:
            star = "*"
    return f"{or_val:.2f}{star} ({ci_lo:.2f}–{ci_hi:.2f})"


def make_or_matrix(df: pd.DataFrame, panel_pct: float, filter_name: str
                   ) -> pd.DataFrame:
    sub = df[
        (df["panel_pct"].round(3) == round(panel_pct, 3))
        & (df["filter"] == filter_name)
    ].copy()
    sub = sub[sub["n_mut_total"] > 0]
    if len(sub) == 0:
        return pd.DataFrame()
    mat = sub.pivot_table(
        index="cancer", columns="head", values="OR", aggfunc="first",
    )
    return mat[HEADS]


def write_md_table(f, df: pd.DataFrame, panel_pct: float, filter_name: str,
                   title: str):
    sub = df[
        (df["panel_pct"].round(3) == round(panel_pct, 3))
        & (df["filter"] == filter_name)
    ].copy()
    sub = sub[sub["n_mut_total"] > 0]
    if len(sub) == 0:
        f.write(f"### {title}\n\n_No rows_\n\n")
        return
    # Order cancers by max OR
    cancer_max = sub.groupby("cancer")["OR"].max().sort_values(ascending=False)
    cancer_order = cancer_max.index.tolist()

    f.write(f"### {title}\n\n")
    header = "| cancer | n_mut |"
    sep = "|---|---:|"
    for h in HEADS:
        header += f" {h.replace('score_', '')} |"
        sep += "---|"
    f.write(header + "\n" + sep + "\n")
    for cancer in cancer_order:
        rrow = sub[sub["cancer"] == cancer]
        if len(rrow) == 0:
            continue
        n_mut = int(rrow.iloc[0]["n_mut_total"])
        line = f"| {cancer} | {n_mut} |"
        for h in HEADS:
            r = rrow[rrow["head"] == h]
            if len(r) == 0:
                line += " — |"
            else:
                line += " " + fmt_or_cell(
                    r.iloc[0]["OR"], r.iloc[0]["OR_ci_lo"],
                    r.iloc[0]["OR_ci_hi"], r.iloc[0]["p_fisher"],
                ) + " |"
        f.write(line + "\n")
    f.write("\n")


def write_markdown_report(out_path: Path, df_pcawg: pd.DataFrame,
                          df_pog570: pd.DataFrame, panel_path: Path,
                          n_units: int):
    log.info("Writing markdown report: %s", out_path)
    with open(out_path, "w") as f:
        f.write("# Per-cancer enrichment — v4_cds panel, all 6 heads\n\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(f"- Panel: `{panel_path}`\n")
        f.write(f"- Panel positions (n_units): {n_units:,}\n")
        f.write(f"- Heads: {', '.join(HEADS)}\n")
        f.write(f"- Panel sizes: {[f'{int(p*100)}%' for p in TOP_PCTS]}\n")
        f.write(f"- Filters: {FILTERS}\n")
        f.write("- OR computed per-cancer over panel positions; "
                "Haldane–Anscombe correction; 95% CI via log-OR ± 1.96·SE; "
                "p from Fisher's exact (one-sided, greater).\n")
        f.write("- Stars: `*` p<0.05, `**` p<1e-5, `***` p<1e-10\n\n")

        # ==== PCAWG ====
        f.write("## PCAWG/TCGA — 10 reference cancers\n\n")
        for tp in TOP_PCTS:
            for filt in FILTERS:
                write_md_table(
                    f, df_pcawg, tp, filt,
                    f"PCAWG — top-{int(tp*100)}% — {filt} (cancer × head OR (CI))",
                )

        # ==== POG570 ====
        cancers_pog = sorted(df_pog570["cancer"].dropna().unique().tolist())
        f.write(f"## POG570 — {len(cancers_pog)} cohorts mapped to PCAWG cancers\n\n")
        for tp in TOP_PCTS:
            for filt in FILTERS:
                write_md_table(
                    f, df_pog570, tp, filt,
                    f"POG570 — top-{int(tp*100)}% — {filt} (cancer × head OR (CI))",
                )

        # ==== Highlights ====
        f.write("## Highlights\n\n")
        for cohort_name, df_ in [("PCAWG", df_pcawg), ("POG570", df_pog570)]:
            f.write(f"### {cohort_name}\n\n")
            sub = df_[
                (df_["panel_pct"].round(3) == 0.01)
                & (df_["filter"] == "filter_TCW_nonCpG")
                & (df_["n_mut_total"] > 0)
            ].copy()

            f.write("**Cells with OR > 10 (top-1%, TCW_nonCpG):**\n\n")
            big = sub[sub["OR"] > 10].sort_values("OR", ascending=False)
            if len(big):
                for _, r in big.iterrows():
                    f.write(
                        f"- {r['cancer']} × {r['head'].replace('score_', '')}: "
                        f"OR={r['OR']:.2f} (CI {r['OR_ci_lo']:.2f}–"
                        f"{r['OR_ci_hi']:.2f}, p={r['p_fisher']:.2e}, "
                        f"n_mut={int(r['n_mut_total'])})\n"
                    )
            else:
                f.write("- _none_\n")
            f.write("\n**Cells with OR > 5 (top-1%, TCW_nonCpG):**\n\n")
            mid = sub[(sub["OR"] > 5) & (sub["OR"] <= 10)].sort_values(
                "OR", ascending=False
            )
            if len(mid):
                for _, r in mid.iterrows():
                    f.write(
                        f"- {r['cancer']} × {r['head'].replace('score_', '')}: "
                        f"OR={r['OR']:.2f} (CI {r['OR_ci_lo']:.2f}–"
                        f"{r['OR_ci_hi']:.2f}, p={r['p_fisher']:.2e}, "
                        f"n_mut={int(r['n_mut_total'])})\n"
                    )
            else:
                f.write("- _none_\n")
            f.write("\n**Cells with p < 1e-10 (top-1%, TCW_nonCpG):**\n\n")
            sig = sub[sub["p_fisher"] < 1e-10].sort_values("p_fisher")
            if len(sig):
                for _, r in sig.iterrows():
                    f.write(
                        f"- {r['cancer']} × {r['head'].replace('score_', '')}: "
                        f"p={r['p_fisher']:.2e}, OR={r['OR']:.2f} "
                        f"(n_mut={int(r['n_mut_total'])})\n"
                    )
            else:
                f.write("- _none_\n")
            f.write("\n")

            # Per-cancer best head
            f.write("**Best head per cancer (top-1%, TCW_nonCpG):**\n\n")
            f.write("| cancer | best head | OR | p |\n|---|---|---:|---:|\n")
            for cancer, sub_c in sub.groupby("cancer"):
                if not len(sub_c):
                    continue
                rr = sub_c.loc[sub_c["OR"].idxmax()]
                f.write(
                    f"| {cancer} | {rr['head'].replace('score_', '')} | "
                    f"{rr['OR']:.2f} | {rr['p_fisher']:.2e} |\n"
                )
            f.write("\n**Best cancer per head (top-1%, TCW_nonCpG):**\n\n")
            f.write("| head | best cancer | OR | p |\n|---|---|---:|---:|\n")
            for head, sub_h in sub.groupby("head"):
                if not len(sub_h):
                    continue
                rr = sub_h.loc[sub_h["OR"].idxmax()]
                f.write(
                    f"| {head.replace('score_', '')} | {rr['cancer']} | "
                    f"{rr['OR']:.2f} | {rr['p_fisher']:.2e} |\n"
                )
            f.write("\n")

            # A3A profile
            a3a_rows = sub[sub["head"] == "score_A3A"].sort_values(
                "OR", ascending=False
            )
            f.write("**A3A head per-cancer profile (top-1%, TCW_nonCpG):**\n\n")
            f.write("| cancer | OR | CI | p | n_mut |\n|---|---:|---:|---:|---:|\n")
            for _, r in a3a_rows.iterrows():
                f.write(
                    f"| {r['cancer']} | {r['OR']:.2f} | "
                    f"{r['OR_ci_lo']:.2f}–{r['OR_ci_hi']:.2f} | "
                    f"{r['p_fisher']:.2e} | {int(r['n_mut_total'])} |\n"
                )

            # Weak APOBEC cancers
            f.write("\n**Weak APOBEC cancers (max OR ≤ 1.5 across all heads, "
                    "top-1%, TCW_nonCpG):**\n\n")
            cancer_max_or = sub.groupby("cancer")["OR"].max()
            weak = cancer_max_or[cancer_max_or <= 1.5].sort_values()
            if len(weak):
                for cancer, mx in weak.items():
                    f.write(f"- {cancer}: max OR = {mx:.2f}\n")
            else:
                f.write("- _none — every cancer has at least one head with OR > 1.5_\n")
            f.write("\n")

            # A3A dominant cancers (A3A is the head with max OR for that cancer)
            f.write("**Cancers where v4_cds A3A head dominates** "
                    "(A3A is best head, top-1%, TCW_nonCpG):\n\n")
            dom = []
            for cancer, sub_c in sub.groupby("cancer"):
                if not len(sub_c):
                    continue
                rr = sub_c.loc[sub_c["OR"].idxmax()]
                if rr["head"] == "score_A3A":
                    dom.append((cancer, rr["OR"]))
            if dom:
                for cancer, or_val in sorted(dom, key=lambda x: -x[1]):
                    f.write(f"- {cancer}: OR(A3A) = {or_val:.2f}\n")
            else:
                f.write("- _none_\n")
            f.write("\n")

        # ==== Concordance ====
        f.write("## PCAWG vs POG570 concordance (per-cancer × per-head OR)\n\n")
        for tp in [0.01, 0.05]:
            for filt in ["filter_TCW_nonCpG", "filter_all_CT"]:
                sub_p = df_pcawg[
                    (df_pcawg["panel_pct"].round(3) == round(tp, 3))
                    & (df_pcawg["filter"] == filt)
                    & (df_pcawg["n_mut_total"] > 0)
                ]
                sub_g = df_pog570[
                    (df_pog570["panel_pct"].round(3) == round(tp, 3))
                    & (df_pog570["filter"] == filt)
                    & (df_pog570["n_mut_total"] > 0)
                ]
                merged = sub_p.merge(
                    sub_g, on=["cancer", "head", "panel_pct", "filter"],
                    suffixes=("_pcawg", "_pog570"), how="inner",
                )
                if len(merged) == 0:
                    continue
                rho, p_rho = stats.spearmanr(
                    merged["OR_pcawg"], merged["OR_pog570"]
                )
                f.write(
                    f"- **top-{int(tp*100)}% / {filt}**: "
                    f"Spearman ρ = {rho:.3f} (p={p_rho:.2e}, "
                    f"n={len(merged)} cells)\n"
                )

                # Outlier cells (large discordance)
                merged["disc"] = np.log2(
                    (merged["OR_pcawg"] + 0.1) / (merged["OR_pog570"] + 0.1)
                ).abs()
                top_outliers = merged.sort_values("disc", ascending=False).head(5)
                f.write(f"  - Top discordant cells:\n")
                for _, r in top_outliers.iterrows():
                    f.write(
                        f"    - {r['cancer']} × {r['head'].replace('score_', '')}: "
                        f"PCAWG OR={r['OR_pcawg']:.2f}, "
                        f"POG570 OR={r['OR_pog570']:.2f}\n"
                    )
        f.write("\n")

        # ==== A3A breadth question ====
        f.write("## A3A signal: broad or driven by hot cancers?\n\n")
        for cohort_name, df_ in [("PCAWG", df_pcawg), ("POG570", df_pog570)]:
            sub = df_[
                (df_["panel_pct"].round(3) == 0.01)
                & (df_["filter"] == "filter_TCW_nonCpG")
                & (df_["head"] == "score_A3A")
                & (df_["n_mut_total"] > 0)
            ].copy()
            n_above_2 = int((sub["OR"] > 2.0).sum())
            n_above_3 = int((sub["OR"] > 3.0).sum())
            n_above_5 = int((sub["OR"] > 5.0).sum())
            n_total = len(sub)
            mean_or = sub["OR"].mean()
            median_or = sub["OR"].median()
            f.write(
                f"- **{cohort_name}** (top-1%, TCW_nonCpG, score_A3A): "
                f"n_cancers = {n_total}; mean OR = {mean_or:.2f}, "
                f"median OR = {median_or:.2f}; "
                f"OR>2: {n_above_2}/{n_total}; "
                f"OR>3: {n_above_3}/{n_total}; "
                f"OR>5: {n_above_5}/{n_total}\n"
            )

        f.write(
            "\nIf the A3A signal is broad, OR>2 should hold across most cancers; "
            "if it is driven by 2–3 hot cancers (BLCA, CESC), the median "
            "should be near 1 with one or two extreme outliers.\n"
        )

    log.info("Wrote %s", out_path)


# =========================================================================== #
# Main
# =========================================================================== #


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load + annotate panel.
    log.info("Loading panel: %s", PANEL_PATH)
    keep = ["chrom", "pos", "strand"] + HEADS
    panel = pd.read_parquet(PANEL_PATH)[keep]
    log.info("  panel rows: %d  cols: %s", len(panel), list(panel.columns))
    panel = annotate_panel_positions(panel)
    n_units = len(panel)

    panel_set = set(zip(
        panel["chrom"].astype(str).values,
        panel["pos"].astype(int).values,
    ))

    # 2. PCAWG/TCGA load
    pcawg_maf = load_pcawg_combined()
    log.info("Restricting PCAWG MAF to in-panel positions ...")
    in_panel_p = np.array(
        [(c, int(p)) in panel_set for c, p in zip(pcawg_maf["chrom"], pcawg_maf["pos"])]
    )
    pcawg_inp = pcawg_maf.iloc[np.where(in_panel_p)[0]].reset_index(drop=True)
    log.info("  in-panel PCAWG: %d / %d (%.2f%%)",
             len(pcawg_inp), len(pcawg_maf),
             100 * len(pcawg_inp) / max(1, len(pcawg_maf)))
    pcawg_inp = annotate_mut_context(pcawg_inp)

    df_pcawg = compute_per_cancer_enrichment(
        panel, pcawg_inp, PCAWG_CANCERS, "PCAWG"
    )
    pcawg_csv = OUT_DIR / "per_cancer_enrichment_v4_pcawg.csv"
    df_pcawg.to_csv(pcawg_csv, index=False)
    log.info("Wrote %s (%d rows)", pcawg_csv, len(df_pcawg))

    # 3. POG570 load
    pog570 = load_pog570()
    log.info("Restricting POG570 to in-panel positions ...")
    in_panel_g = np.array(
        [(c, int(p)) in panel_set for c, p in zip(pog570["chrom"], pog570["pos"])]
    )
    pog570_inp = pog570.iloc[np.where(in_panel_g)[0]].reset_index(drop=True)
    log.info("  in-panel POG570: %d / %d (%.2f%%)",
             len(pog570_inp), len(pog570),
             100 * len(pog570_inp) / max(1, len(pog570)))

    # Choose cancer label set: top 10 cohorts by total in-panel mutations,
    # using the COHORT_MAP'd label where present, else use the raw cohort tag.
    pog570_inp = pog570_inp.copy()
    pog570_inp["cancer"] = pog570_inp["cancer"].fillna(
        pog570_inp["analysis_cohort"].astype(str).str.lower().add("_pog")
    )
    pog570_inp = annotate_mut_context(pog570_inp)
    cancer_counts = pog570_inp["cancer"].value_counts()
    log.info("POG570 in-panel mutations per cancer (top 15):")
    log.info("\n%s", cancer_counts.head(15).to_string())
    pog_cancers = cancer_counts.head(10).index.tolist()
    log.info("Selected POG570 cancers (top 10): %s", pog_cancers)

    df_pog570 = compute_per_cancer_enrichment(
        panel, pog570_inp[pog570_inp["cancer"].isin(pog_cancers)],
        pog_cancers, "POG570",
    )
    pog_csv = OUT_DIR / "per_cancer_enrichment_v4_pog570.csv"
    df_pog570.to_csv(pog_csv, index=False)
    log.info("Wrote %s (%d rows)", pog_csv, len(df_pog570))

    # 4. Heatmaps
    plot_or_heatmap(
        df_pcawg, 0.01, "filter_TCW_nonCpG", "PCAWG",
        OUT_DIR / "per_cancer_OR_pcawg_top1pct.png",
    )
    plot_or_heatmap(
        df_pcawg, 0.05, "filter_TCW_nonCpG", "PCAWG",
        OUT_DIR / "per_cancer_OR_pcawg_top5pct.png",
    )
    plot_or_heatmap(
        df_pcawg, 0.01, "filter_all_CT", "PCAWG",
        OUT_DIR / "per_cancer_OR_pcawg_top1pct_allCT.png",
    )
    plot_or_heatmap(
        df_pog570, 0.01, "filter_TCW_nonCpG", "POG570",
        OUT_DIR / "per_cancer_OR_pog570_top1pct.png",
    )
    plot_or_heatmap(
        df_pog570, 0.05, "filter_TCW_nonCpG", "POG570",
        OUT_DIR / "per_cancer_OR_pog570_top5pct.png",
    )

    # 5. Concordance plot
    plot_pcawg_pog570_concordance(
        df_pcawg, df_pog570, 0.01, "filter_TCW_nonCpG",
        OUT_DIR / "per_cancer_OR_concordance_top1pct.png",
    )

    # 6. Markdown
    write_markdown_report(
        OUT_DIR / "PER_CANCER_ENRICHMENT_V4.md",
        df_pcawg, df_pog570, PANEL_PATH, n_units,
    )

    log.info("DONE in %.1fs.", time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
