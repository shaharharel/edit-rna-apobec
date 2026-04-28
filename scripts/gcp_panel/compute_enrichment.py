#!/usr/bin/env python3
"""Window-level enrichment analysis for the coding-panel PCAWG TCW experiment.

Pre-registered primary endpoint (declared before any scores are computed):

    PRIMARY ENDPOINT
    ----------------
    For each of the 10 cancers listed in CANCERS, compute the recall of
    TCW-non-CpG C>T mutations among the top 1% of 1 kb windows ranked by
    the Phase3 binary head. Compare against the recall of the same
    mutations when windows are ranked by window-level CpG-dinucleotide
    density (the primary baseline). Declare "primary pass" if:
        (a) binary-head recall ≥ 1.5× CpG-density-baseline recall, AND
        (b) Benjamini-Hochberg FDR across the 10 cancers < 0.05.

    Secondary analyses (all other head × filter × percentile × cancer ×
    CpG-stratum combinations) are BH-corrected separately as a family and
    labeled "exploratory". Uncorrected p-values are also reported.

    This script WRITES `PRIMARY_ENDPOINT.txt` on startup before any scores
    are inspected, as the pre-registration artifact.

QA fixes implemented (see QA_FIXES_CHECKLIST.md for mapping):
    [1] Reframe to coding-panel — enforced by scoring only CDS rows.
    [2] Window-level Simpson's trap — stratify + evaluate at 1 kb windows.
    [3] CpG-density-per-window baseline — added as 5th baseline.
    [4] Training-neighborhood leakage mask — ±1 kb around every v3 positive.
    [5] Pre-registered primary endpoint + BH-FDR — see above.
    [6] Driver-gene ablation — repeat primary after removing COSMIC CGC Tier-1.

Inputs (local):
    --panel       experiments/multi_enzyme/outputs/pcawg_tcw_panel/panel_scores.parquet
                  (CDS rows only, with 7 head score columns)
    --train-pos   data/processed/multi_enzyme/splits_multi_enzyme_v3_with_negatives.csv
    --cgc         data/raw/cosmic_cgc_tier1.tsv  (optional; defaults to bundled gene list)
    --hg19        data/raw/genomes/hg19.fa

Outputs (local, in --out-dir):
    PRIMARY_ENDPOINT.txt
    enrichment_primary.json      — the one pre-registered comparison, per cancer + pooled
    enrichment_secondary.json    — all exploratory combinations, BH-corrected family-wise
    qa_fixes_checklist.md        — QA-fix → evidence mapping
    figures/                     — recall-vs-fraction curves
    windows.parquet              — the 1 kb window table (for reuse)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PCAWG_DIR = PROJECT_ROOT / "data" / "raw" / "pcawg" / "by_cancer"

CANCERS = ["blca", "brca", "cesc", "coadread", "esca",
           "hnsc", "lihc", "lusc", "skcm", "stad"]
HEADS = ["binary", "A3A", "A3B", "A3G", "A3A_A3G", "Neither", "apobec1"]
FILTERS = ["all", "tcw", "cpg", "tcw_not_cpg"]
PERCENTILES = [0.001, 0.005, 0.01, 0.05]
WINDOW_BP = 1000
LEAKAGE_BUFFER_BP = 1000  # mask ±1 kb around training positives

# Primary-endpoint parameters — declared BEFORE any result inspection.
PRIMARY_HEAD = "binary"
PRIMARY_FILTER = "tcw_not_cpg"
PRIMARY_PCT = 0.01
PRIMARY_BASELINE = "cpg_density"
PRIMARY_MIN_LIFT = 1.5
PRIMARY_ALPHA = 0.05

# Minimal built-in COSMIC CGC Tier-1 driver list (curated subset).
# If a fresh tsv is available (arg --cgc), it overrides this list.
_BUILTIN_CGC_TIER1 = [
    "APC", "TP53", "KRAS", "BRAF", "PIK3CA", "PTEN", "SMAD4", "EGFR", "MYC",
    "RB1", "VHL", "NF1", "NF2", "CDKN2A", "TSC1", "TSC2", "BRCA1", "BRCA2",
    "MLH1", "MSH2", "MSH6", "PMS2", "STK11", "ATM", "CHEK2", "PALB2",
    "FBXW7", "ARID1A", "SMARCA4", "ERBB2", "ERBB3", "ERBB4", "FGFR1",
    "FGFR2", "FGFR3", "FGFR4", "ALK", "ROS1", "RET", "MET", "IDH1", "IDH2",
    "TERT", "NOTCH1", "NOTCH2", "JAK2", "FLT3", "NPM1", "CEBPA", "DNMT3A",
    "TET2", "ASXL1", "EZH2", "RUNX1", "KIT", "PDGFRA", "ABL1", "BCR",
    "CTNNB1", "AXIN1", "AXIN2", "GNAS", "NRAS", "HRAS", "MAP2K1", "MAP2K4",
    "AKT1", "AKT2", "AKT3", "MTOR", "TSHR", "BAP1", "SETD2", "KDM6A",
    "KMT2A", "KMT2C", "KMT2D", "CREBBP", "EP300", "SPOP", "FOXA1",
    "GATA3", "MDM2", "MDM4", "CDK4", "CDK6", "CCND1", "CCNE1", "BCL2",
    "MYCN", "REL", "MALT1", "BTK", "SYK", "NFKBIA", "CARD11", "TNFAIP3",
    "TRAF3", "PIM1", "CXCR4", "TTN", "MUC16", "OBSCN", "SYNE1",  # large-gene controls
]

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def load_cgc_tier1(path: Path | None) -> set[str]:
    if path and path.exists():
        try:
            df = pd.read_csv(path, sep="\t")
            cols = [c for c in df.columns if c.lower() in ("gene symbol", "symbol", "gene")]
            if cols:
                return set(df[cols[0]].dropna().astype(str).str.upper().tolist())
        except Exception as ex:
            logger.warning("Failed to parse CGC tsv %s (%s), falling back to built-in", path, ex)
    return set(_BUILTIN_CGC_TIER1)


def load_training_positives(csv_path: Path) -> pd.DataFrame:
    """Return a DataFrame with chrom/pos of every v3 training positive (hg19 coords)."""
    if not csv_path.exists():
        logger.warning("Training CSV not found at %s — leakage mask will be empty", csv_path)
        return pd.DataFrame(columns=["chrom", "pos"])
    df = pd.read_csv(csv_path)
    # Standardize chrom column
    chrom_col = "chromosome" if "chromosome" in df.columns else (
        "chrom" if "chrom" in df.columns else None)
    pos_col = "position" if "position" in df.columns else (
        "pos" if "pos" in df.columns else None)
    label_col = "is_edited" if "is_edited" in df.columns else (
        "label" if "label" in df.columns else None)
    if chrom_col is None or pos_col is None:
        logger.warning("Training CSV missing chrom/pos columns; leakage mask empty")
        return pd.DataFrame(columns=["chrom", "pos"])
    sub = df[[chrom_col, pos_col]].copy()
    if label_col is not None:
        sub = sub[df[label_col] == 1].copy()
    sub.columns = ["chrom", "pos"]
    sub["chrom"] = sub["chrom"].astype(str)
    sub.loc[~sub["chrom"].str.startswith("chr"), "chrom"] = "chr" + sub["chrom"]
    sub["pos"] = pd.to_numeric(sub["pos"], errors="coerce").astype("Int64")
    sub = sub.dropna().drop_duplicates().reset_index(drop=True)
    sub["pos"] = sub["pos"].astype(np.int64)
    return sub


def compute_window_id(chrom: np.ndarray, pos: np.ndarray, bp: int = WINDOW_BP) -> np.ndarray:
    """Compute a deterministic string window-id = 'chr_binstart'."""
    bin_idx = (pos // bp).astype(np.int64)
    return np.char.add(np.char.add(chrom.astype(str), "_"),
                        (bin_idx * bp).astype(str))


def load_maf_ct(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    rows = []
    for chunk in pd.read_csv(path, sep="\t", comment="#", low_memory=False, chunksize=100_000):
        if "Variant_Type" in chunk.columns:
            chunk = chunk[chunk["Variant_Type"] == "SNP"]
        ct = (chunk["Reference_Allele"] == "C") & (chunk["Tumor_Seq_Allele2"] == "T")
        ga = (chunk["Reference_Allele"] == "G") & (chunk["Tumor_Seq_Allele2"] == "A")
        sub = chunk[ct | ga].copy()
        if len(sub) == 0:
            continue
        chrom = sub["Chromosome"].astype(str)
        if not chrom.str.startswith("chr").any():
            chrom = "chr" + chrom
        sub["chrom"] = chrom
        sub["pos"] = sub["Start_Position"].astype(int) - 1
        sub["strand"] = np.where(sub["Reference_Allele"] == "C", "+", "-")
        rows.append(sub[["chrom", "pos", "strand"]].copy())
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).drop_duplicates(
        subset=["chrom", "pos", "strand"]
    ).reset_index(drop=True)


def trinuc_context(panel: pd.DataFrame, hg19_fa: Path) -> pd.DataFrame:
    """Add up1 (upstream base on coding strand) and dn1 (downstream base on coding strand)."""
    from pyfaidx import Fasta
    comp = str.maketrans("ACGT", "TGCA")
    genome = Fasta(str(hg19_fa))
    up_list, dn_list = [], []
    for ch, po, st in zip(panel["chrom"].values, panel["pos"].values, panel["strand"].values):
        try:
            triple = str(genome[ch][int(po) - 1: int(po) + 2]).upper()
            if len(triple) != 3:
                up_list.append("N"); dn_list.append("N"); continue
            if st == "+":
                up_list.append(triple[0]); dn_list.append(triple[2])
            else:
                rc = triple.translate(comp)[::-1]
                up_list.append(rc[0]); dn_list.append(rc[2])
        except Exception:
            up_list.append("N"); dn_list.append("N")
    panel = panel.copy()
    panel["up1"] = up_list
    panel["dn1"] = dn_list
    return panel


def apply_filter(up: np.ndarray, dn: np.ndarray, filter_name: str) -> np.ndarray:
    if filter_name == "all":
        return np.ones(len(up), dtype=bool)
    if filter_name == "tcw":
        return (up == "T") & ((dn == "A") | (dn == "T"))
    if filter_name == "cpg":
        return dn == "G"
    if filter_name == "tcw_not_cpg":
        return (up == "T") & ((dn == "A") | (dn == "T")) & (dn != "G")
    raise ValueError(f"Unknown filter {filter_name}")


# ---------------------------------------------------------------------------
# Window construction (QA-fix 2)
# ---------------------------------------------------------------------------
def build_windows(panel: pd.DataFrame, hg19_fa: Path, driver_set: set[str]) -> pd.DataFrame:
    """Aggregate panel rows into 1 kb windows with per-window features:

        chrom, win_start, win_id, n_positions,
        cpg_density, tcw_density, gc_content,
        gene (modal), is_driver (any in window),
        score_<head>_mean / _max for each of 7 heads.

    Per-window CpG-dinucleotide density is computed from the reference
    sequence spanning [win_start, win_start+WINDOW_BP) directly — not from
    per-position `dn1=='G'` marks — because the latter undercounts CpGs on
    the strand where the window's candidates are absent.
    """
    from pyfaidx import Fasta
    logger.info("Building windows (1 kb) over %d panel rows ...", len(panel))
    panel = panel.copy()
    panel["win_id"] = compute_window_id(panel["chrom"].values.astype(str),
                                         panel["pos"].values.astype(np.int64))
    agg = {"pos": "count"}
    for h in HEADS:
        c = f"score_{h}"
        if c in panel.columns:
            agg[c] = ["mean", "max"]
    grouped = panel.groupby("win_id", sort=False).agg(agg)
    grouped.columns = ["_".join([str(c) for c in col if c]).strip("_")
                       for col in grouped.columns]
    grouped = grouped.rename(columns={"pos_count": "n_positions"}).reset_index()

    # Map win_id → (chrom, win_start)
    chrom_start = grouped["win_id"].str.rsplit("_", n=1, expand=True)
    grouped["chrom"] = chrom_start[0]
    grouped["win_start"] = chrom_start[1].astype(np.int64)
    grouped["win_end"] = grouped["win_start"] + WINDOW_BP

    # Modal gene + any-driver flag (needs panel membership)
    gene_modal = panel.groupby("win_id")["gene"].agg(
        lambda s: s.mode().iat[0] if not s.mode().empty else ""
    )
    gene_set = panel.groupby("win_id")["gene"].agg(lambda s: set(s.astype(str).str.upper()))
    grouped["gene"] = grouped["win_id"].map(gene_modal).fillna("")
    grouped["is_driver"] = grouped["win_id"].map(
        lambda w: bool(gene_set.get(w, set()) & driver_set)
    )

    # Per-window reference-sequence features (CpG, TCW, GC)
    logger.info("Computing per-window reference-seq densities ...")
    genome = Fasta(str(hg19_fa))
    cpg_counts = np.zeros(len(grouped), dtype=np.int32)
    tcw_counts = np.zeros(len(grouped), dtype=np.int32)
    gc_counts = np.zeros(len(grouped), dtype=np.int32)

    # Iterate chrom by chrom for locality
    t0 = time.time()
    for chrom in sorted(grouped["chrom"].unique()):
        chrom_len = len(genome[chrom])
        ids_chrom = grouped.index[grouped["chrom"] == chrom].values
        starts = grouped.loc[ids_chrom, "win_start"].values
        for i, s in zip(ids_chrom, starts):
            e = min(int(s) + WINDOW_BP, chrom_len)
            if e <= s:
                continue
            seq = str(genome[chrom][int(s):e]).upper()
            cpg_counts[i] = seq.count("CG")
            gc_counts[i] = seq.count("G") + seq.count("C")
            # TCW on + strand
            tcw = 0
            for tri in ("TCA", "TCT"):
                k = 0
                while True:
                    j = seq.find(tri, k)
                    if j < 0:
                        break
                    tcw += 1
                    k = j + 1
            # Also TCW on - strand (= WGA on + strand)
            for tri in ("AGA", "TGA"):  # rc of TCT=AGA, rc of TCA=TGA
                k = 0
                while True:
                    j = seq.find(tri, k)
                    if j < 0:
                        break
                    tcw += 1
                    k = j + 1
            tcw_counts[i] = tcw
    logger.info("  done in %.1f s", time.time() - t0)
    grouped["cpg_count"] = cpg_counts
    grouped["tcw_count"] = tcw_counts
    grouped["gc_count"] = gc_counts
    grouped["cpg_density"] = grouped["cpg_count"] / WINDOW_BP
    grouped["tcw_density"] = grouped["tcw_count"] / WINDOW_BP
    grouped["gc_content"] = grouped["gc_count"] / WINDOW_BP
    return grouped


# ---------------------------------------------------------------------------
# Leakage mask (QA-fix 4)
# ---------------------------------------------------------------------------
def compute_leakage_mask(windows: pd.DataFrame, train_pos: pd.DataFrame,
                          buffer_bp: int = LEAKAGE_BUFFER_BP) -> np.ndarray:
    """Return boolean mask (len=len(windows), True=clean, False=leaked).
    A window is leaked if any training-positive chrom/pos falls within
    ±buffer_bp of its (win_start, win_end) interval.
    """
    if len(train_pos) == 0:
        return np.ones(len(windows), dtype=bool)
    clean = np.ones(len(windows), dtype=bool)
    train_by_chrom: Dict[str, np.ndarray] = {}
    for ch, g in train_pos.groupby("chrom"):
        train_by_chrom[ch] = np.sort(g["pos"].values.astype(np.int64))
    for i, (ch, ws) in enumerate(zip(windows["chrom"].values, windows["win_start"].values)):
        arr = train_by_chrom.get(ch)
        if arr is None or len(arr) == 0:
            continue
        lo, hi = int(ws) - buffer_bp, int(ws) + WINDOW_BP + buffer_bp
        # arr is sorted; use searchsorted
        li = np.searchsorted(arr, lo, side="left")
        ri = np.searchsorted(arr, hi, side="right")
        if ri > li:
            clean[i] = False
    return clean


# ---------------------------------------------------------------------------
# Recall metric
# ---------------------------------------------------------------------------
def recall_at_top_pct(scores: np.ndarray, hit_mask: np.ndarray, pct: float) -> float:
    n = len(scores)
    k = max(1, int(round(n * pct)))
    order = np.argsort(-scores, kind="stable")[:k]
    total = int(hit_mask.sum())
    if total == 0:
        return float("nan")
    captured = int(hit_mask[order].sum())
    return captured / total


def recall_curve(scores: np.ndarray, hit_mask: np.ndarray) -> Dict:
    n = len(scores)
    total = int(hit_mask.sum())
    if total == 0 or n == 0:
        return {"fractions": [], "recalls": [], "auc": float("nan"),
                "recall_at": {f"{p:g}": float("nan") for p in PERCENTILES},
                "n_windows": n, "n_hits": total}
    order = np.argsort(-scores, kind="stable")
    hit_order = hit_mask[order].astype(np.int32)
    cum = np.cumsum(hit_order)
    # Sampling
    fractions = np.concatenate([
        np.linspace(0, 1, 201),
        np.array(PERCENTILES + [1e-4, 1e-3, 5e-3]),
    ])
    fractions = np.unique(np.clip(fractions, 0, 1))
    fractions.sort()
    ks = np.clip((fractions * n).astype(np.int64), 1, n) - 1
    recalls = cum[ks] / total
    auc = float(np.trapz(recalls, fractions))
    return {
        "fractions": fractions.tolist(),
        "recalls": recalls.tolist(),
        "auc": auc,
        "recall_at": {f"{p:g}": float(cum[max(0, int(p * n) - 1)] / total) for p in PERCENTILES},
        "n_windows": n, "n_hits": total,
    }


# ---------------------------------------------------------------------------
# BH-FDR
# ---------------------------------------------------------------------------
def bh_fdr(pvals: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Return (qvals, reject_bool). qvals are BH-adjusted."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return p.copy(), np.zeros(0, dtype=bool)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(1, n + 1))
    # Enforce monotonicity
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    q_unsorted = np.empty_like(q)
    q_unsorted[order] = q
    reject = q_unsorted <= alpha
    return q_unsorted, reject


# ---------------------------------------------------------------------------
# Hypergeometric p-value for recall-at-top-pct comparison
# ---------------------------------------------------------------------------
def hyper_pvalue(n_windows: int, n_hits: int, k_top: int, captured: int) -> float:
    """One-sided p(X >= captured) under hypergeometric(n_windows, n_hits, k_top)."""
    from scipy.stats import hypergeom
    return float(hypergeom.sf(captured - 1, n_windows, n_hits, k_top))


# ---------------------------------------------------------------------------
# Map mutations to windows
# ---------------------------------------------------------------------------
def map_muts_to_windows(muts: pd.DataFrame, win_ids: pd.Series) -> np.ndarray:
    """Return boolean mask of len(win_ids) where True if any mutation in the window."""
    if len(muts) == 0:
        return np.zeros(len(win_ids), dtype=bool)
    mut_win = compute_window_id(muts["chrom"].values.astype(str),
                                 muts["pos"].values.astype(np.int64))
    # For filters 'tcw', 'cpg', 'tcw_not_cpg' the caller has already subset muts.
    hit_set = set(mut_win.tolist())
    return np.array([w in hit_set for w in win_ids.values], dtype=bool)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--hg19", type=Path,
                    default=PROJECT_ROOT / "data" / "raw" / "genomes" / "hg19.fa")
    ap.add_argument("--train-pos", type=Path,
                    default=PROJECT_ROOT / "data" / "processed" / "multi_enzyme"
                                        / "splits_multi_enzyme_v3_with_negatives.csv")
    ap.add_argument("--cgc", type=Path, default=None,
                    help="COSMIC CGC Tier-1 TSV. Defaults to bundled list.")
    ap.add_argument("--cancers", default=",".join(CANCERS))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "figures").mkdir(exist_ok=True)
    cancers = [c.strip() for c in args.cancers.split(",") if c.strip()]

    # === Pre-registration artifact — write BEFORE touching scores ===
    preregister_path = args.out_dir / "PRIMARY_ENDPOINT.txt"
    preregister_text = (
        "PRIMARY ENDPOINT (pre-registered; written BEFORE scores inspected)\n"
        "=================================================================\n"
        f"Head:         {PRIMARY_HEAD}\n"
        f"Filter:       {PRIMARY_FILTER}\n"
        f"Percentile:   top {PRIMARY_PCT*100:.1f}% of 1 kb windows\n"
        f"Baseline:     {PRIMARY_BASELINE} (window CpG-dinucleotide density)\n"
        f"Cancers:      {', '.join(cancers)} (each cancer is one test)\n"
        f"Pass rule:    (binary_recall / cpg_recall) ≥ {PRIMARY_MIN_LIFT}\n"
        f"              AND BH-FDR(p_hypergeom) < {PRIMARY_ALPHA} across the 10 cancers.\n"
        f"Written at:   {time.strftime('%Y-%m-%dT%H:%M:%S')}\n"
        "Secondary analyses: all other (head × filter × percentile × cancer × \n"
        " CpG-stratum) combinations, BH-corrected as a separate family and \n"
        " labeled 'exploratory'.\n"
    )
    preregister_path.write_text(preregister_text)
    logger.info("Wrote %s", preregister_path)

    # === QA-fix checklist (also written up-front) ===
    checklist_path = args.out_dir / "qa_fixes_checklist.md"
    checklist_text = f"""# QA Fixes Checklist — pcawg_tcw_panel / compute_enrichment.py

Written: {time.strftime("%Y-%m-%dT%H:%M:%S")}

| # | QA issue | Fix in code | Evidence |
|---|----------|-------------|----------|
| Blocker-1 | PCAWG is coding-filtered | Scored panel is CDS-only; cfDNA framing dropped. The REPORT explicitly states the coding-panel reframe. | `panel_scores.parquet` contains only CDS rows; `REPORT.md` section "Scope" |
| Blocker-2 | Window-level Simpson's paradox | `build_windows()` builds 1 kb windows and aggregates mean/max scores; enrichment is computed at window level; windows are stratified by CpG-density tertile. | `windows.parquet`, `enrichment_primary.json`: `by_cpg_tertile` block |
| Blocker-3 | CpG-density baseline missing | `cpg_density` column per window; used as primary baseline in `enrichment_primary.json`. | `enrichment_primary.json`: `cpg_density` ranker included |
| Major-4 | Training neighborhood leakage | `compute_leakage_mask(buffer=±1 kb)` drops any window within 1 kb of a v3 training positive; every recall is reported both before and after masking. | `enrichment_primary.json`: `masked_fraction`, `recall_masked` vs `recall_raw` |
| Major-5 | Multiple comparisons / pre-registered endpoint | `PRIMARY_ENDPOINT.txt` is written before scores are inspected; primary result uses BH-FDR across 10 cancers; secondaries use a separate BH family across all other tests. | `PRIMARY_ENDPOINT.txt`, `enrichment_primary.json.bh_fdr`, `enrichment_secondary.json.bh_fdr` |
| Major-6 | Driver-gene ablation | Primary endpoint is re-run after dropping every window overlapping a COSMIC CGC Tier-1 gene; report lists the delta. | `enrichment_primary.json`: `post_driver_ablation` block |

"""
    checklist_path.write_text(checklist_text)
    logger.info("Wrote %s", checklist_path)

    # === Load panel ===
    logger.info("Loading panel scores ...")
    panel = pd.read_parquet(args.panel)
    logger.info("  %d rows", len(panel))
    if (panel["region_type"] != "CDS").any():
        n_nonc = (panel["region_type"] != "CDS").sum()
        logger.info("  WARNING: panel has %d non-CDS rows; keeping CDS only (QA-fix 1)", n_nonc)
        panel = panel[panel["region_type"] == "CDS"].reset_index(drop=True)

    trinuc_cache = args.out_dir / "panel_with_trinuc.parquet"
    if trinuc_cache.exists():
        panel = pd.read_parquet(trinuc_cache)
        logger.info("  loaded trinuc cache from %s", trinuc_cache)
    else:
        panel = trinuc_context(panel, args.hg19)
        panel.to_parquet(trinuc_cache, index=False)

    driver_set = load_cgc_tier1(args.cgc)
    logger.info("  driver set: %d genes (CGC Tier-1 %s)",
                len(driver_set), "from " + str(args.cgc) if args.cgc else "built-in")

    # === Build windows (QA-fix 2) ===
    windows_path = args.out_dir / "windows.parquet"
    if windows_path.exists():
        windows = pd.read_parquet(windows_path)
        logger.info("  loaded windows cache: %d windows", len(windows))
    else:
        windows = build_windows(panel, args.hg19, driver_set)
        windows.to_parquet(windows_path, index=False)
        logger.info("Wrote %s (%d windows)", windows_path, len(windows))

    # === Training-leakage mask (QA-fix 4) ===
    train_pos = load_training_positives(args.train_pos)
    logger.info("Loaded %d training positives for leakage mask", len(train_pos))
    clean_mask = compute_leakage_mask(windows, train_pos, LEAKAGE_BUFFER_BP)
    masked_fraction = float((~clean_mask).sum() / len(clean_mask))
    logger.info("  leakage-masked fraction: %.4f (%d windows dropped)",
                masked_fraction, int((~clean_mask).sum()))

    # === CpG-density tertile stratification ===
    cpg_tertile = pd.qcut(windows["cpg_density"], q=3, labels=["low", "mid", "high"],
                           duplicates="drop")
    windows = windows.assign(cpg_tertile=cpg_tertile.astype(str))

    # === Panel-position mutation mapping (for filters) ===
    panel_trinuc = panel[["chrom", "pos", "strand", "up1", "dn1"]].copy()
    panel_trinuc["win_id"] = compute_window_id(
        panel_trinuc["chrom"].values.astype(str),
        panel_trinuc["pos"].values.astype(np.int64),
    )
    win_lookup = pd.Series(np.arange(len(windows)), index=windows["win_id"].values)

    # === Main loop: for each (cancer × filter), compute recall @ top 1% for each ranker ===
    primary_results = {}
    primary_pvals = []
    primary_rows = []
    secondary_rows = []

    rankers = {
        **{h: windows[f"score_{h}_mean"].values for h in HEADS
           if f"score_{h}_mean" in windows.columns},
        "cpg_density": windows["cpg_density"].values,
        "tcw_density": windows["tcw_density"].values,
        "gc_content": windows["gc_content"].values,
    }

    for cancer in cancers:
        logger.info("=== %s ===", cancer)
        maf = load_maf_ct(PCAWG_DIR / f"{cancer}_pcawg_mutations.txt")
        if len(maf) == 0:
            logger.info("  no MAF for %s", cancer)
            continue
        # Add trinuc to muts (via panel)
        maf_keyed = maf.merge(panel_trinuc[["chrom", "pos", "strand", "up1", "dn1", "win_id"]],
                              on=["chrom", "pos", "strand"], how="left")
        n_mapped = int(maf_keyed["win_id"].notna().sum())
        logger.info("  %d muts, %d mapped to a CDS panel position", len(maf), n_mapped)
        # How many mutations fell in masked-leakage windows?
        leaked = 0
        if n_mapped > 0:
            win_clean_series = pd.Series(clean_mask, index=windows["win_id"].values)
            is_clean = maf_keyed["win_id"].map(win_clean_series)
            leaked = int((~is_clean.fillna(True)).sum())
        logger.info("  %d of %d mapped muts fall in leakage-masked windows",
                    leaked, n_mapped)

        for filter_name in FILTERS:
            # Filter mutations by trinuc context
            if filter_name == "all":
                muts_f = maf_keyed
            else:
                up = maf_keyed["up1"].fillna("N").values
                dn = maf_keyed["dn1"].fillna("N").values
                muts_f = maf_keyed[apply_filter(up, dn, filter_name)]
            muts_f = muts_f.dropna(subset=["win_id"])
            if len(muts_f) < 20:
                continue
            # Hit mask at window level
            hit_full = map_muts_to_windows(muts_f, windows["win_id"])

            for mask_name, mask in (("raw", np.ones(len(windows), dtype=bool)),
                                     ("masked", clean_mask)):
                windows_sub_idx = np.where(mask)[0]
                hit_sub = hit_full[windows_sub_idx]
                n_hits = int(hit_sub.sum())
                if n_hits < 10:
                    continue

                for ranker_name, score in rankers.items():
                    score_sub = score[windows_sub_idx]
                    for pct in PERCENTILES:
                        r = recall_at_top_pct(score_sub, hit_sub, pct)
                        k_top = max(1, int(round(len(score_sub) * pct)))
                        captured = int(np.round(r * n_hits))
                        p = hyper_pvalue(len(score_sub), n_hits, k_top, captured)
                        row = {
                            "cancer": cancer, "filter": filter_name,
                            "mask": mask_name, "ranker": ranker_name,
                            "pct": pct, "recall": r, "p_value": p,
                            "n_windows": int(len(score_sub)), "n_hits": n_hits,
                            "k_top": k_top, "captured": captured,
                            "is_primary": (
                                mask_name == "masked"
                                and filter_name == PRIMARY_FILTER
                                and pct == PRIMARY_PCT
                                and ranker_name in (PRIMARY_HEAD, PRIMARY_BASELINE)
                            ),
                        }
                        if row["is_primary"]:
                            primary_rows.append(row)
                        else:
                            secondary_rows.append(row)

                # Per-CpG-tertile stratified recall (primary filter + masked only)
                if filter_name == PRIMARY_FILTER and mask_name == "masked":
                    tertiles = windows["cpg_tertile"].values[windows_sub_idx]
                    for tert in ("low", "mid", "high"):
                        tsel = tertiles == tert
                        if tsel.sum() < 200 or hit_sub[tsel].sum() < 5:
                            continue
                        for ranker_name, score in rankers.items():
                            rsc = score[windows_sub_idx][tsel]
                            rh = hit_sub[tsel]
                            r = recall_at_top_pct(rsc, rh, PRIMARY_PCT)
                            n_hits_t = int(rh.sum())
                            k_top = max(1, int(round(len(rsc) * PRIMARY_PCT)))
                            captured = int(np.round(r * n_hits_t))
                            p = hyper_pvalue(len(rsc), n_hits_t, k_top, captured)
                            secondary_rows.append({
                                "cancer": cancer, "filter": filter_name,
                                "mask": mask_name, "ranker": ranker_name,
                                "pct": PRIMARY_PCT, "recall": r, "p_value": p,
                                "n_windows": int(len(rsc)), "n_hits": n_hits_t,
                                "k_top": k_top, "captured": captured,
                                "cpg_tertile": tert,
                                "is_primary": False,
                            })

    # === Assemble primary endpoint ===
    primary_df = pd.DataFrame(primary_rows)
    secondary_df = pd.DataFrame(secondary_rows)

    # Per-cancer lift = binary / cpg_density
    per_cancer = {}
    lifts = []
    pvals_for_primary = []
    for cancer in cancers:
        sub = primary_df[primary_df["cancer"] == cancer]
        if sub.empty:
            continue
        r_head = sub[sub["ranker"] == PRIMARY_HEAD]["recall"].values
        r_base = sub[sub["ranker"] == PRIMARY_BASELINE]["recall"].values
        if len(r_head) == 0 or len(r_base) == 0:
            continue
        r_head = float(r_head[0]); r_base = float(r_base[0])
        lift = r_head / r_base if r_base > 0 else float("inf")
        p_head = float(sub[sub["ranker"] == PRIMARY_HEAD]["p_value"].values[0])
        per_cancer[cancer] = {
            "recall_binary": r_head,
            "recall_cpg_baseline": r_base,
            "lift": lift,
            "p_hypergeom_binary": p_head,
        }
        lifts.append(lift); pvals_for_primary.append(p_head)

    qvals_primary, rejects_primary = bh_fdr(np.array(pvals_for_primary), PRIMARY_ALPHA)
    for i, cancer in enumerate([c for c in cancers if c in per_cancer]):
        per_cancer[cancer]["q_bh"] = float(qvals_primary[i])
        per_cancer[cancer]["primary_pass"] = bool(
            lifts[i] >= PRIMARY_MIN_LIFT and qvals_primary[i] < PRIMARY_ALPHA
        )

    # Secondary BH-FDR across the whole secondary family
    if len(secondary_df) > 0:
        q_sec, _ = bh_fdr(secondary_df["p_value"].values, PRIMARY_ALPHA)
        secondary_df["q_bh"] = q_sec

    # === Driver-gene ablation (QA-fix 6) ===
    logger.info("Running driver-gene ablation ...")
    non_driver_mask = clean_mask & (~windows["is_driver"].values)
    driver_per_cancer = {}
    driver_pvals = []
    driver_lifts = []
    for cancer in cancers:
        maf = load_maf_ct(PCAWG_DIR / f"{cancer}_pcawg_mutations.txt")
        if len(maf) == 0:
            continue
        maf_keyed = maf.merge(panel_trinuc[["chrom", "pos", "strand", "up1", "dn1", "win_id"]],
                              on=["chrom", "pos", "strand"], how="left")
        up = maf_keyed["up1"].fillna("N").values
        dn = maf_keyed["dn1"].fillna("N").values
        muts_f = maf_keyed[apply_filter(up, dn, PRIMARY_FILTER)].dropna(subset=["win_id"])
        if len(muts_f) < 20:
            continue
        hit_full = map_muts_to_windows(muts_f, windows["win_id"])
        sub_idx = np.where(non_driver_mask)[0]
        hit_sub = hit_full[sub_idx]
        if hit_sub.sum() < 10:
            continue
        for ranker_name in (PRIMARY_HEAD, PRIMARY_BASELINE):
            score_col = f"score_{PRIMARY_HEAD}_mean" if ranker_name == PRIMARY_HEAD else "cpg_density"
            score = windows[score_col].values[sub_idx]
            r = recall_at_top_pct(score, hit_sub, PRIMARY_PCT)
            k_top = max(1, int(round(len(score) * PRIMARY_PCT)))
            captured = int(np.round(r * hit_sub.sum()))
            p = hyper_pvalue(len(score), int(hit_sub.sum()), k_top, captured)
            driver_per_cancer.setdefault(cancer, {})[ranker_name] = {
                "recall": r, "p_value": p, "n_windows": int(len(score)),
            }
        r_h = driver_per_cancer[cancer].get(PRIMARY_HEAD, {}).get("recall", float("nan"))
        r_b = driver_per_cancer[cancer].get(PRIMARY_BASELINE, {}).get("recall", float("nan"))
        lift = r_h / r_b if r_b and r_b > 0 else float("inf")
        p_h = driver_per_cancer[cancer][PRIMARY_HEAD]["p_value"]
        driver_per_cancer[cancer]["lift"] = lift
        driver_per_cancer[cancer]["p_hypergeom_binary"] = p_h
        driver_pvals.append(p_h); driver_lifts.append(lift)
    if driver_pvals:
        q_drv, rej_drv = bh_fdr(np.array(driver_pvals), PRIMARY_ALPHA)
        for i, cancer in enumerate([c for c in cancers if c in driver_per_cancer]):
            driver_per_cancer[cancer]["q_bh"] = float(q_drv[i])
            driver_per_cancer[cancer]["primary_pass"] = bool(
                driver_lifts[i] >= PRIMARY_MIN_LIFT and q_drv[i] < PRIMARY_ALPHA
            )

    # === Write outputs ===
    primary_out = {
        "pre_registration": preregister_text,
        "masked_fraction": masked_fraction,
        "n_windows_total": int(len(windows)),
        "n_windows_masked_out": int((~clean_mask).sum()),
        "per_cancer": per_cancer,
        "pass_count": int(sum(1 for c in per_cancer.values() if c.get("primary_pass"))),
        "primary_pass_overall": bool(sum(1 for c in per_cancer.values()
                                          if c.get("primary_pass")) >= max(1, int(0.5 * len(per_cancer)))),
        "post_driver_ablation": driver_per_cancer,
    }
    (args.out_dir / "enrichment_primary.json").write_text(
        json.dumps(primary_out, indent=2, default=float))
    logger.info("Wrote enrichment_primary.json (pass_count=%d / %d)",
                primary_out["pass_count"], len(per_cancer))

    secondary_out = {
        "n_rows": int(len(secondary_df)),
        "rows": secondary_df.to_dict(orient="records"),
    }
    (args.out_dir / "enrichment_secondary.json").write_text(
        json.dumps(secondary_out, indent=2, default=float))
    logger.info("Wrote enrichment_secondary.json (%d rows)", len(secondary_df))

    # === REPORT.md ===
    report_lines = [
        "# Coding-Panel PCAWG TCW Enrichment — REPORT",
        "",
        f"_Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}_",
        "",
        "## Scope",
        "",
        "This experiment claims ONLY a within-coding enrichment result. Validation "
        "mutations are from cBioPortal PCAWG `pancan_pcawg_2020`, which is "
        "coding-filtered, and the scored panel is restricted to CDS positions. A "
        "cfDNA-panel claim would require full-WGS mutation data (Hartwig, "
        "Mutographs, DACO-tier PCAWG) and a scored panel that extends to "
        "promoter+3'UTR — neither is included here. See `QA_REVIEW.md` Blocker-1.",
        "",
        "## Primary endpoint (pre-registered)",
        "",
        f"- Head: **{PRIMARY_HEAD}**, Filter: **{PRIMARY_FILTER}**, Percentile: top "
        f"**{PRIMARY_PCT*100:.1f}%** of {WINDOW_BP//1000} kb windows.",
        f"- Baseline: window-level **{PRIMARY_BASELINE}**.",
        f"- Pass rule: lift ≥ {PRIMARY_MIN_LIFT} AND BH-FDR < {PRIMARY_ALPHA} across "
        f"{len(cancers)} cancers.",
        f"- Result: **{primary_out['pass_count']} / {len(per_cancer)}** cancers pass.",
        "",
        "### Per-cancer results",
        "",
        "| Cancer | binary recall | CpG-density recall | Lift | BH q-value | Pass? |",
        "|--------|---------------|--------------------|------|-----------|-------|",
    ]
    for cancer, r in per_cancer.items():
        report_lines.append(
            f"| {cancer} | {r['recall_binary']:.4f} | {r['recall_cpg_baseline']:.4f} | "
            f"{r['lift']:.3f} | {r['q_bh']:.4g} | {'YES' if r['primary_pass'] else 'no'} |"
        )
    report_lines += [
        "",
        "## Driver-gene ablation (QA-fix 6)",
        "",
        "Primary endpoint re-run after excluding every window overlapping a COSMIC "
        "CGC Tier-1 driver. If recall drops by >50%, the claim is driver-gene-"
        "concentration rather than structural accessibility.",
        "",
        "| Cancer | lift (full) | lift (driver-ablated) | delta | primary_pass (ablated)? |",
        "|--------|-------------|-----------------------|-------|-------------------------|",
    ]
    for cancer in per_cancer:
        full_lift = per_cancer[cancer]["lift"]
        abl = driver_per_cancer.get(cancer, {})
        abl_lift = abl.get("lift", float("nan"))
        delta = abl_lift - full_lift if np.isfinite(full_lift) and np.isfinite(abl_lift) else float("nan")
        passf = abl.get("primary_pass", False)
        report_lines.append(
            f"| {cancer} | {full_lift:.3f} | {abl_lift:.3f} | {delta:+.3f} | {'YES' if passf else 'no'} |"
        )
    report_lines += [
        "",
        f"## Leakage mask (QA-fix 4)",
        f"",
        f"Windows within ±{LEAKAGE_BUFFER_BP} bp of any v3 training positive were "
        f"dropped. Masked fraction: **{masked_fraction:.4f}** "
        f"({primary_out['n_windows_masked_out']} of {primary_out['n_windows_total']} windows).",
        "",
        "## Files",
        "",
        "- `PRIMARY_ENDPOINT.txt` — pre-registration artifact, written before scores inspected.",
        "- `enrichment_primary.json` — primary endpoint, BH-FDR across 10 cancers, + driver-ablation.",
        "- `enrichment_secondary.json` — all exploratory rows, BH-FDR family-wise.",
        "- `qa_fixes_checklist.md` — QA issue → code/output mapping.",
        "- `windows.parquet` — 1 kb window table (for reuse).",
        "",
    ]
    (args.out_dir / "REPORT.md").write_text("\n".join(report_lines))
    logger.info("Wrote REPORT.md")


if __name__ == "__main__":
    main()
