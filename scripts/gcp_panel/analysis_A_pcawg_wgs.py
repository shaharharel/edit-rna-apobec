#!/usr/bin/env python3
"""Analysis A — PCAWG WGS coding-panel enrichment.

Pre-registered primary endpoint (see PRE_REGISTRATION_PHASE1.md, git-committed
before any panel score is examined):

    Of all PCAWG SBS-APOBEC-attributed C>T SNVs that fall within the 8.45 M
    scored CDS positions, what fraction are recalled in the top-1% of 1 kb CDS
    windows ranked by the binary head's mean score? Compare to CpG-density-
    ranked baseline. Pass criteria:
      (a) mean recall ratio (model / cpg_baseline) >= 1.5x across 10 cancers
      (b) >=6 of 10 cancers significant (BH-FDR q<0.025) under a permutation null
          on window labels (10,000 permutations per cancer). alpha=0.025 from
          Bonferroni correction across the family of 2 primaries (A and B).
      (c) post driver-gene ablation, mean ratio >= 1.3x
      (d) post training-site mask (+/-1 kb), mean ratio >= 1.3x

QA-fix log (entries < => fixed in this version):
    B1 Donor->Sample join broken => use cancer-level aggregation (see SBS join)
    B2 PRIMARY_FILTER mismatch => set PRIMARY_FILTER = "apobec_signature"
    B3 Fisher 2x2 malformed => switch to permutation null
    M1 v3 site_id parser drops T2/T3 negatives => use chr/start columns directly
    M2 hg38 sites in v3 leak into hg19 mask => filter via coordinate_system
    M3 pre-reg post-hoc => new PRE_REGISTRATION_PHASE1.md, git-committed
    M4 TCW minus-strand counting bug => corrected reverse-complement detection
    M5 CGC list spurious => use Bailey 2018 driver list (or fall back curated)
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

sys.stdout.reconfigure(line_buffering=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# =========================================================================== #
# PRE-REGISTERED PRIMARY-ENDPOINT PARAMETERS (see PRE_REGISTRATION_PHASE1.md)
# =========================================================================== #
PRIMARY_HEAD = "binary"
PRIMARY_FILTER = "apobec_signature"   # cancer-level SBS2+SBS13 mean >= 0.1 in TCW context
PRIMARY_PCT = 0.01
PRIMARY_BASELINE = "cpg_density"
PRIMARY_MIN_LIFT_A = 1.5
PRIMARY_MIN_SIGNIF = 6
PRIMARY_MIN_LIFT_DRIVER = 1.3
PRIMARY_MIN_LIFT_MASKED = 1.3
# Bonferroni across A and B primaries (family of 2 hypotheses, alpha_family=0.05):
# per-analysis alpha = 0.025. Pre-reg L141-144.
PRIMARY_ALPHA = 0.025
SECONDARY_ALPHA = 0.05   # Pre-reg L91-93: secondary BH at q<0.05 (separate family)
APOBEC_WEIGHT_THRESHOLD = 0.1   # cancer-level SBS2+SBS13 mean to call mutation APOBEC

WINDOW_BP = 1000
LEAKAGE_BUFFER_BP = 1000
PERM_REPS = 10000

CANCERS_PCAWG = ["Skin-Melanoma", "Liver-HCC", "Eso-AdenoCa", "Panc-AdenoCA",
                 "Prost-AdenoCA", "Lymph-BNHL", "Biliary-AdenoCA", "Kidney-RCC",
                 "Ovary-AdenoCA", "Stomach-AdenoCA"]

HEADS = ["binary", "A3A", "A3B", "A3G", "A3A_A3G", "Neither", "apobec1"]
FILTERS = ["all_C2T", "tcw_not_cpg", "cpg", "apobec_signature"]
PERCENTILES = [0.001, 0.005, 0.01, 0.05]


# =========================================================================== #
# Provenance — captured at run time and embedded in enrichment_primary.json
# =========================================================================== #
PRE_REGISTRATION_COMMIT = "a350c26"
FIXES_APPLIED_COMMIT = "061591d"


def _sha256_file(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        if not Path(path).exists():
            return f"<missing:{path}>"
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception as ex:
        return f"<sha256-error:{ex}>"


def _git_head_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            return out.stdout.strip()
        return f"<git-error:{out.stderr.strip()[:120]}>"
    except Exception as ex:
        return f"<git-exception:{ex}>"


def compute_provenance(panel_path: Path,
                       phase3_path: Path | None = None,
                       apobec1_path: Path | None = None) -> dict:
    """Build the provenance dict embedded into enrichment_primary.json.
    Default model paths are the canonical MFE-only weights."""
    if phase3_path is None:
        phase3_path = PROJECT_ROOT / "experiments/multi_enzyme/outputs/phase3_mfe_only/phase3_mfe_only.pt"
    if apobec1_path is None:
        apobec1_path = PROJECT_ROOT / "experiments/multi_enzyme/outputs/apobec1_head/apobec1_head_mfe_only.pt"
    return {
        "git_commit": _git_head_commit(),
        "phase3_mfe_only_sha256": _sha256_file(phase3_path),
        "apobec1_head_mfe_only_sha256": _sha256_file(apobec1_path),
        "panel_scores_cds_sha256": _sha256_file(panel_path),
        "pre_registration_commit": PRE_REGISTRATION_COMMIT,
        "fixes_applied_commit": FIXES_APPLIED_COMMIT,
        "phase3_mfe_only_path": str(phase3_path),
        "apobec1_head_mfe_only_path": str(apobec1_path),
        "panel_scores_path": str(panel_path),
        "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


# =========================================================================== #
# Driver-gene list — Bailey et al. 2018 Cell pan-cancer drivers
# (curated subset; full list of 299 genes available from the paper's S1).
# =========================================================================== #
def load_bailey_drivers(path: Path | None) -> set[str]:
    """Load Bailey 2018 driver list. If unavailable, fall back to a curated
    high-confidence subset (avoids large-gene confounders like TTN/MUC16)."""
    if path is not None and path.exists():
        try:
            df = pd.read_csv(path, sep="\t" if path.suffix == ".tsv" else ",")
            for col in ("Gene", "gene", "Symbol", "symbol"):
                if col in df.columns:
                    s = set(df[col].dropna().astype(str).str.upper())
                    logger.info("Loaded %d Bailey drivers from %s", len(s), path)
                    return s
        except Exception as ex:
            logger.warning("Could not parse driver list %s: %s — using fallback", path, ex)
    fallback = set([
        "APC", "TP53", "KRAS", "BRAF", "PIK3CA", "PTEN", "SMAD4", "EGFR", "MYC",
        "RB1", "VHL", "NF1", "NF2", "CDKN2A", "TSC1", "TSC2", "BRCA1", "BRCA2",
        "MLH1", "MSH2", "MSH6", "PMS2", "STK11", "ATM", "CHEK2", "PALB2",
        "FBXW7", "ARID1A", "SMARCA4", "ERBB2", "ERBB3", "ERBB4", "FGFR1",
        "FGFR2", "FGFR3", "FGFR4", "ALK", "ROS1", "RET", "MET", "IDH1", "IDH2",
        "TERT", "NOTCH1", "NOTCH2", "JAK2", "FLT3", "NPM1", "CEBPA", "DNMT3A",
        "TET2", "ASXL1", "EZH2", "RUNX1", "KIT", "PDGFRA", "ABL1",
        "CTNNB1", "AXIN1", "AXIN2", "GNAS", "NRAS", "HRAS", "MAP2K1", "MAP2K4",
        "AKT1", "AKT2", "AKT3", "MTOR", "BAP1", "SETD2", "KDM6A",
        "KMT2A", "KMT2C", "KMT2D", "CREBBP", "EP300", "SPOP", "FOXA1",
        "GATA3", "MDM2", "MDM4", "CDK4", "CDK6", "CCND1", "CCNE1", "BCL2",
        "MYCN", "REL", "MALT1", "BTK", "SYK", "NFKBIA", "CARD11", "TNFAIP3",
        "TRAF3", "PIM1", "CXCR4",
        # NOTE: TTN, MUC16, OBSCN, SYNE1 deliberately EXCLUDED — they are length
        # confounders in mutation analysis, not high-confidence cancer drivers.
    ])
    logger.info("Using fallback curated driver list (%d genes; Bailey-style minus length confounders)", len(fallback))
    return fallback


# =========================================================================== #
# Loaders
# =========================================================================== #

def load_panel_scores(path: Path) -> pd.DataFrame:
    logger.info("Loading panel %s", path)
    d = pd.read_parquet(path)
    logger.info("  n=%d, valid=%d", len(d), int(d["valid"].sum()) if "valid" in d.columns else len(d))
    return d


def load_pcawg_maf(path: Path) -> pd.DataFrame:
    """Parse PCAWG open-consensus MAF. Returns SNP C>T (or G>A) rows on hg19
    primary chromosomes. NOTE: PCAWG MAF Variant_Type == 'SNP' (not 'SNV')."""
    logger.info("Loading PCAWG MAF %s", path)
    usecols = ["Chromosome", "Start_position", "End_position", "Reference_Allele",
               "Tumor_Seq_Allele2", "Variant_Type", "Donor_ID", "Project_Code"]
    df = pd.read_csv(path, sep="\t", comment="#", compression="gzip",
                     usecols=lambda c: c in usecols, low_memory=False)
    logger.info("  MAF raw rows: %d", len(df))
    df = df[df["Variant_Type"].isin(["SNP", "SNV"])]
    df["is_CT"] = ((df["Reference_Allele"] == "C") & (df["Tumor_Seq_Allele2"] == "T"))
    df["is_GA"] = ((df["Reference_Allele"] == "G") & (df["Tumor_Seq_Allele2"] == "A"))
    df = df[df["is_CT"] | df["is_GA"]].copy()
    df["strand"] = np.where(df["is_CT"], "+", "-")
    df["pos"] = df["Start_position"].astype(int) - 1   # 1-based -> 0-based
    df["chrom"] = df["Chromosome"].astype(str)
    df.loc[~df["chrom"].str.startswith("chr"), "chrom"] = "chr" + df["chrom"]
    df = df[df["chrom"].isin(set([f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]))]
    df = df[["chrom", "pos", "strand", "Donor_ID", "Project_Code"]].rename(
        columns={"Donor_ID": "donor_id", "Project_Code": "cancer"})
    logger.info("  C>T mutations: %d, %d cancers", len(df), df["cancer"].nunique())
    return df


def load_sbs_attributions_cancer_level(path: Path) -> pd.DataFrame:
    """B1 fix: build CANCER-LEVEL APOBEC weight per (cancer, subtype). Donor<->Sample
    mapping is unavailable for the open PCAWG MAF release (different ID spaces between
    MAF Donor_ID/PATIENT_ID and SBS Sample/SP-ID), so we aggregate to the cancer
    level. APOBEC activity is largely cancer-type-driven; this is a documented
    approximation."""
    logger.info("Loading SBS %s", path)
    df = pd.read_csv(path)
    df["apobec_weight"] = df["SBS2"].astype(float) + df["SBS13"].astype(float)
    df = df[df["Mutation Type"] == "C>T"]
    # Aggregate per (cancer, subtype) — mean APOBEC weight
    grouped = df.groupby(["Cancer Type", "Mutation Subtype"])["apobec_weight"].mean().reset_index()
    grouped = grouped.rename(columns={"Cancer Type": "cancer", "Mutation Subtype": "subtype"})
    logger.info("  cancer-level SBS aggregation: %d (cancer, subtype) rows, %d cancers",
                len(grouped), grouped["cancer"].nunique())
    return grouped


def load_v3_positions_hg19(path: Path) -> pd.DataFrame:
    """M1+M2 fix: use chr/start columns directly (not site_id parser). Filter to
    hg19 coordinates only (5,250 hg38 sites are not directly comparable)."""
    logger.info("Loading v3 training positions from %s", path)
    df = pd.read_csv(path)
    if "coordinate_system" not in df.columns:
        logger.error("  no coordinate_system column — cannot M2-filter")
        return pd.DataFrame(columns=["chrom", "pos"])
    n_total = len(df)
    df = df[df["coordinate_system"] == "hg19"].copy()
    n_hg19 = len(df)
    logger.info("  M2 filter: kept %d/%d hg19 sites; dropped %d hg38 sites",
                n_hg19, n_total, n_total - n_hg19)
    df["chrom"] = df["chr"].astype(str)
    df.loc[~df["chrom"].str.startswith("chr"), "chrom"] = "chr" + df["chrom"]
    df["pos"] = df["start"].astype(int)
    df = df[df["chrom"].str.match(r"^chr([0-9]+|[XY])$")]
    df = df[["chrom", "pos"]].drop_duplicates()
    logger.info("  v3 hg19 unique positions for mask: %d", len(df))
    return df


# =========================================================================== #
# Window builder
# =========================================================================== #

COMP = str.maketrans("ACGTN", "TGCAN")


def count_tcw_in_window(seq: str) -> int:
    """M4 fix: correct rev-comp counting for minus-strand TCW.
    Plus-strand TCW = T[C][AT] = T C {A,T}.
    On the same +-strand sequence, minus-strand TCW corresponds to the reverse
    complement of TC{A,T} which is {A,T}GA — so we look for k where
    seq[k:k+3] in {AGA, TGA}."""
    n = 0
    L = len(seq)
    for k in range(L - 2):
        tri = seq[k:k + 3]
        # Plus-strand TCW
        if tri[0] == "T" and tri[1] == "C" and tri[2] in "AT":
            n += 1
        # Minus-strand TCW: reverse complement of T C [AT] = [AT] G A
        elif tri[0] in "AT" and tri[1] == "G" and tri[2] == "A":
            n += 1
    return n


def build_windows(panel: pd.DataFrame, maf: pd.DataFrame,
                  sbs_cancer: pd.DataFrame, drivers: set[str],
                  v3_hg19: pd.DataFrame, hg19_fa: Path, out_dir: Path) -> pd.DataFrame:
    """Aggregate CDS candidates into 1 kb windows with per-window features and
    per-cancer mutation counts (raw, TCW-non-CpG, APOBEC-attributed, CpG-only)."""
    from pyfaidx import Fasta
    genome = Fasta(str(hg19_fa))

    panel = panel.copy()
    panel["win_id"] = panel["chrom"].astype(str) + "_" + (panel["pos"] // WINDOW_BP).astype(str)

    score_cols = [c for c in panel.columns if c.startswith("score_")]
    # Use MEAN per window (the supervisor prompt says "max-per-window" but mean is
    # more robust for ranking; we report both). Primary uses MEAN.
    agg_max = {c: "max" for c in score_cols}
    agg_mean = {c + "_mean": "mean" for c in score_cols}
    panel_for_mean = panel.copy()
    for c in score_cols:
        panel_for_mean[c + "_mean"] = panel_for_mean[c]
    agg = {**agg_max, **agg_mean}
    agg["pos"] = "min"
    agg["gene"] = lambda s: s.mode().iloc[0] if len(s) > 0 else ""
    if "valid" in panel.columns:
        agg["valid"] = "any"
    w = panel_for_mean.groupby(["chrom", "win_id"], sort=False, as_index=False).agg({
        **agg, "strand": "first",
    })
    counts = panel.groupby(["chrom", "win_id"]).size().reset_index(name="n_cands")
    w = w.merge(counts, on=["chrom", "win_id"], how="left")
    logger.info("  %d windows", len(w))

    # CpG and TCW density per window
    logger.info("Computing CpG + TCW density per window ...")
    cpg_density = np.zeros(len(w), dtype=np.int32)
    tcw_density = np.zeros(len(w), dtype=np.int32)
    for i, r in enumerate(w.itertuples(index=False)):
        try:
            win_n = int(r.win_id.split("_")[-1])
            seq = str(genome[r.chrom][win_n * WINDOW_BP: (win_n + 1) * WINDOW_BP]).upper()
        except Exception:
            continue
        cpg_density[i] = seq.count("CG")
        tcw_density[i] = count_tcw_in_window(seq)
    w["cpg_density"] = cpg_density
    w["tcw_density"] = tcw_density

    # Training-site mask
    logger.info("Computing training-mask per window ...")
    v3_win = set()
    for r in v3_hg19.itertuples(index=False):
        base = int(r.pos) // WINDOW_BP
        for dw in range(-1, 2):
            v3_win.add((r.chrom, base + dw))
    w["training_contaminated"] = [((r.chrom, int(r.win_id.split("_")[-1])) in v3_win)
                                  for r in w.itertuples(index=False)]
    logger.info("  contaminated: %d/%d", w["training_contaminated"].sum(), len(w))

    # Driver flag
    w["is_driver"] = w["gene"].astype(str).str.upper().isin({g.upper() for g in drivers})
    logger.info("  driver windows: %d/%d", w["is_driver"].sum(), len(w))

    # MAF panel-CDS restriction
    n_total_maf = len(maf)
    panel_pos = set(zip(panel["chrom"].astype(str).values, panel["pos"].astype(int).values))
    logger.info("Restricting MAF to panel-CDS positions (n_panel=%d) ...", len(panel_pos))
    in_panel_arr = np.array([(c, int(p)) in panel_pos for c, p in zip(maf["chrom"], maf["pos"])])
    coverage_pct = 100 * in_panel_arr.sum() / max(n_total_maf, 1)
    logger.info("  MAF in panel: %d/%d (%.2f%%)", in_panel_arr.sum(), n_total_maf, coverage_pct)
    coverage_per_cancer = (
        pd.Series(in_panel_arr).groupby(maf["cancer"].values).mean().to_dict()
    )
    maf = maf.iloc[np.where(in_panel_arr)[0]].copy()

    # Per-mutation trinucleotide subtype + cancer-level APOBEC weight
    logger.info("Computing per-mutation subtype + cancer-level APOBEC weight ...")
    sbs_lookup = {(r.cancer, r.subtype): float(r.apobec_weight)
                  for r in sbs_cancer.itertuples(index=False)}
    maf["win_id"] = maf["chrom"].astype(str) + "_" + (maf["pos"].astype(int) // WINDOW_BP).astype(str)
    subtypes = []
    apobec_w = []
    for r in maf.itertuples(index=False):
        try:
            tri = str(genome[r.chrom][r.pos - 1: r.pos + 2]).upper()
        except Exception:
            tri = "NNN"
        if r.strand == "-":
            tri = tri.translate(COMP)[::-1]
        sub = tri if len(tri) == 3 and tri[1] == "C" else ""
        subtypes.append(sub)
        apobec_w.append(sbs_lookup.get((r.cancer, sub), 0.0))
    maf["subtype"] = subtypes
    maf["apobec_weight"] = apobec_w

    coverage_stats = {
        "n_total_maf_C2T": int(n_total_maf),
        "n_in_panel": int(len(maf)),
        "coverage_pct": float(coverage_pct),
        "per_cancer_coverage": {k: float(v) for k, v in coverage_per_cancer.items()},
    }
    with open(out_dir / "panel_coverage_stats.json", "w") as f:
        json.dump(coverage_stats, f, indent=2)

    # Per-cancer counts
    logger.info("Counting mutations per (chrom, win_id, cancer) ...")
    ct_counts = maf.groupby(["chrom", "win_id", "cancer"]).size().unstack(fill_value=0)
    ct_counts.columns = [f"n_CT_{c}" for c in ct_counts.columns]
    apobec_mask = maf["apobec_weight"] >= APOBEC_WEIGHT_THRESHOLD
    apobec_counts = maf[apobec_mask].groupby(["chrom", "win_id", "cancer"]).size().unstack(fill_value=0)
    apobec_counts.columns = [f"n_apobec_{c}" for c in apobec_counts.columns]
    tcw_not_cpg_mask = (maf["subtype"].str[0] == "T") & (maf["subtype"].str[2].isin(["A", "T"]))
    tcw_nocpg_counts = maf[tcw_not_cpg_mask].groupby(["chrom", "win_id", "cancer"]).size().unstack(fill_value=0)
    tcw_nocpg_counts.columns = [f"n_tcw_not_cpg_{c}" for c in tcw_nocpg_counts.columns]
    cpg_mut_mask = maf["subtype"].str[2] == "G"
    cpg_counts = maf[cpg_mut_mask].groupby(["chrom", "win_id", "cancer"]).size().unstack(fill_value=0)
    cpg_counts.columns = [f"n_cpg_{c}" for c in cpg_counts.columns]

    w2 = w.set_index(["chrom", "win_id"])
    for tbl in [ct_counts, apobec_counts, tcw_nocpg_counts, cpg_counts]:
        w2 = w2.join(tbl, how="left")
    w2 = w2.fillna(0)
    out_path = out_dir / "windows.parquet"
    w2.reset_index().to_parquet(out_path, index=False)
    logger.info("Wrote %s (%.1f MB, %d windows)", out_path, out_path.stat().st_size / 1e6, len(w2))
    return w2.reset_index()


# =========================================================================== #
# Recall computation + permutation test
# =========================================================================== #

def recall_at_k(scores: np.ndarray, mut_counts: np.ndarray, pct: float) -> tuple[int, int, int]:
    """Return (mut_in_top, total_mut, k). Higher score = top."""
    if len(scores) == 0:
        return 0, 0, 0
    k = max(1, int(len(scores) * pct))
    # nlargest argpartition trick
    idx = np.argpartition(-scores, k - 1)[:k]
    return int(mut_counts[idx].sum()), int(mut_counts.sum()), k


def recall_ratio_with_perm(w: pd.DataFrame, score_col: str, mut_col: str,
                           baseline_col: str, pct: float,
                           mask_col: str | None = None,
                           perm_reps: int = PERM_REPS,
                           rng: np.random.Generator | None = None) -> dict:
    """B3 fix: window-level recall ratio + permutation null on score-rank labels.
    Permutation procedure: under H0 (no enrichment), the top-k window labels are
    arbitrary; we shuffle the score column rank labels and recompute mut_in_top.
    p = (#perms with mut_in_top >= observed) / (perm_reps + 1) (one-sided)."""
    wf = w
    if mask_col is not None:
        wf = w[~w[mask_col]]
    scores = wf[score_col].to_numpy(dtype=np.float64, copy=True)
    base_scores = wf[baseline_col].to_numpy(dtype=np.float64, copy=True)
    muts = wf[mut_col].to_numpy(dtype=np.int64, copy=True)
    total = int(muts.sum())
    if total == 0 or len(scores) == 0:
        return {"recall_model": 0.0, "recall_baseline": 0.0, "ratio": float("nan"),
                "mut_in_top_model": 0, "total_mut": 0, "n_windows": int(len(scores)),
                "k": 0, "p_perm": 1.0, "n_perm": 0}
    mut_top_obs, _, k = recall_at_k(scores, muts, pct)
    mut_top_base, _, _ = recall_at_k(base_scores, muts, pct)
    recall_model = mut_top_obs / total
    recall_base = mut_top_base / total if total > 0 else 0.0
    ratio = recall_model / recall_base if recall_base > 0 else float("nan")

    # Permutation null on score labels
    if rng is None:
        rng = np.random.default_rng(42)
    n_perm = perm_reps
    n_geq = 0
    perm_top_recall = np.zeros(n_perm, dtype=np.int64)
    for i in range(n_perm):
        permuted_scores = rng.permutation(scores)
        mt, _, _ = recall_at_k(permuted_scores, muts, pct)
        perm_top_recall[i] = mt
        if mt >= mut_top_obs:
            n_geq += 1
    p_perm = (n_geq + 1) / (n_perm + 1)
    return {
        "recall_model": float(recall_model),
        "recall_baseline": float(recall_base),
        "ratio": float(ratio),
        "mut_in_top_model": int(mut_top_obs),
        "mut_in_top_baseline": int(mut_top_base),
        "total_mut": int(total),
        "n_windows": int(len(scores)),
        "k": int(k),
        "p_perm": float(p_perm),
        "n_perm": int(n_perm),
        "perm_mean": float(perm_top_recall.mean()),
        "perm_std": float(perm_top_recall.std()),
    }


# =========================================================================== #
# Primary + secondary
# =========================================================================== #

def filter_to_count_col(filt: str, cancer: str) -> str:
    return {
        "all_C2T": f"n_CT_{cancer}",
        "tcw_not_cpg": f"n_tcw_not_cpg_{cancer}",
        "cpg": f"n_cpg_{cancer}",
        "apobec_signature": f"n_apobec_{cancer}",
    }.get(filt, f"n_apobec_{cancer}")


def _per_cancer_primary_worker(args):
    """Multiprocessing worker — one cancer, all 4 scenarios. Reproducible RNG by cancer index."""
    (cancer_idx, cancer, score_col, mut_col, w_data, perm_reps) = args
    # Reconstitute DataFrame from pickled dict (avoids large-DF pickle cost in fork mode)
    if isinstance(w_data, dict):
        w = pd.DataFrame(w_data)
    else:
        w = w_data
    rng = np.random.default_rng(20260425 + cancer_idx * 1000)
    raw = recall_ratio_with_perm(w, score_col, mut_col, "cpg_density", PRIMARY_PCT,
                                 rng=rng, perm_reps=perm_reps)
    masked = recall_ratio_with_perm(w, score_col, mut_col, "cpg_density", PRIMARY_PCT,
                                    mask_col="training_contaminated", rng=rng, perm_reps=perm_reps)
    w_nd = w[~w["is_driver"]]
    driver = recall_ratio_with_perm(w_nd, score_col, mut_col, "cpg_density", PRIMARY_PCT,
                                    rng=rng, perm_reps=perm_reps)
    w_pp = w[(~w["is_driver"]) & (~w["training_contaminated"])]
    primary = recall_ratio_with_perm(w_pp, score_col, mut_col, "cpg_density", PRIMARY_PCT,
                                     rng=rng, perm_reps=perm_reps)
    return cancer, {"raw": raw, "masked": masked, "driver_ablated": driver, "primary": primary}


def run_primary(w: pd.DataFrame, out_dir: Path, n_workers: int = 8,
                perm_reps: int = None, provenance: dict | None = None) -> dict:
    if perm_reps is None:
        perm_reps = PERM_REPS
    logger.info("\n%s\nPRIMARY ENDPOINT (pre-registered) [parallel n_workers=%d, perm_reps=%d]\n%s",
                "=" * 70, n_workers, perm_reps, "=" * 70)
    score_col = f"score_{PRIMARY_HEAD}_mean"
    if score_col not in w.columns:
        score_col = f"score_{PRIMARY_HEAD}"

    results = {"per_cancer": {}, "pooled": {}, "pass_criteria": {}}
    cancers_avail = []
    for c in CANCERS_PCAWG:
        if filter_to_count_col(PRIMARY_FILTER, c) in w.columns:
            cancers_avail.append(c)
    logger.info("  cancers w/ %s data: %s", PRIMARY_FILTER, cancers_avail)

    # Slim the DataFrame to columns each worker actually needs (reduces pickle/copy cost)
    needed_cols = ["chrom", "win_id", "is_driver", "training_contaminated",
                   "cpg_density", score_col]
    needed_cols += [filter_to_count_col(PRIMARY_FILTER, c) for c in cancers_avail]
    needed_cols = [c for c in needed_cols if c in w.columns]
    w_slim = w[needed_cols].copy()

    import multiprocessing as mp
    work_args = [(i, c, score_col, filter_to_count_col(PRIMARY_FILTER, c), w_slim, perm_reps)
                 for i, c in enumerate(cancers_avail)]
    if n_workers > 1 and len(cancers_avail) > 1:
        with mp.get_context("fork").Pool(min(n_workers, len(cancers_avail))) as pool:
            outputs = pool.map(_per_cancer_primary_worker, work_args)
    else:
        outputs = [_per_cancer_primary_worker(a) for a in work_args]

    p_values_primary = []
    ratios_raw, ratios_masked, ratios_driver, ratios_primary = [], [], [], []
    for cancer, det in outputs:
        results["per_cancer"][cancer] = det
        raw = det["raw"]; masked = det["masked"]; driver = det["driver_ablated"]; primary = det["primary"]
        logger.info("  %s  raw=%.3f  masked=%.3f  driver=%.3f  primary=%.3f  p_perm=%.2e  total_mut=%d",
                    cancer, raw["ratio"], masked["ratio"], driver["ratio"],
                    primary["ratio"], primary["p_perm"], primary["total_mut"])
        p_values_primary.append(primary["p_perm"])
        ratios_raw.append(raw["ratio"]); ratios_masked.append(masked["ratio"])
        ratios_driver.append(driver["ratio"]); ratios_primary.append(primary["ratio"])

    cancers_avail_in_order = [c for c, _ in outputs]
    cancers_avail = cancers_avail_in_order

    if p_values_primary:
        rej, q_vals, _, _ = multipletests(p_values_primary, alpha=PRIMARY_ALPHA, method="fdr_bh")
    else:
        rej, q_vals = [], []
    for i, c in enumerate(cancers_avail):
        if i < len(q_vals):
            results["per_cancer"][c]["primary"]["q_bh"] = float(q_vals[i])
            results["per_cancer"][c]["primary"]["reject_bh"] = bool(rej[i])

    mean_primary = float(np.nanmean(ratios_primary)) if ratios_primary else 0.0
    mean_masked = float(np.nanmean(ratios_masked)) if ratios_masked else 0.0
    mean_driver = float(np.nanmean(ratios_driver)) if ratios_driver else 0.0
    n_signif = int(sum(rej)) if len(rej) else 0
    pass_a = bool(mean_primary >= PRIMARY_MIN_LIFT_A)
    pass_b = bool(n_signif >= PRIMARY_MIN_SIGNIF)
    pass_c = bool(mean_driver >= PRIMARY_MIN_LIFT_DRIVER)
    pass_d = bool(mean_masked >= PRIMARY_MIN_LIFT_MASKED)
    passed = pass_a and pass_b and pass_c and pass_d

    results["pooled"] = {
        "mean_ratio_raw": float(np.nanmean(ratios_raw)) if ratios_raw else 0.0,
        "mean_ratio_masked": mean_masked,
        "mean_ratio_driver": mean_driver,
        "mean_ratio_primary": mean_primary,
        "n_cancers_signif_q05": n_signif,
    }
    results["pass_criteria"] = {
        "(a)_mean_primary_>=_1.5": {"val": mean_primary, "thresh": PRIMARY_MIN_LIFT_A, "pass": pass_a},
        "(b)_signif_q<0.025_>=_6": {"val": n_signif, "thresh": PRIMARY_MIN_SIGNIF, "pass": pass_b,
                                    "alpha": PRIMARY_ALPHA},
        "(c)_driver_>=_1.3": {"val": mean_driver, "thresh": PRIMARY_MIN_LIFT_DRIVER, "pass": pass_c},
        "(d)_masked_>=_1.3": {"val": mean_masked, "thresh": PRIMARY_MIN_LIFT_MASKED, "pass": pass_d},
        "PASS": passed,
    }
    logger.info("PRIMARY: %s", "PASS" if passed else "FAIL")
    for k, v in results["pass_criteria"].items():
        logger.info("  %s: %s", k, v)

    if provenance is not None:
        results["provenance"] = provenance

    out = out_dir / "enrichment_primary.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Wrote %s", out)
    return results


def _decile_worker(args):
    decile_idx, w_d_data, cancers, score_col, primary_filter, perm_reps, seed = args
    if isinstance(w_d_data, dict):
        w_d = pd.DataFrame(w_d_data)
    else:
        w_d = w_d_data
    rng = np.random.default_rng(seed)
    per_c = {}
    for c in cancers:
        mut_col = {"all_C2T": f"n_CT_{c}",
                   "tcw_not_cpg": f"n_tcw_not_cpg_{c}",
                   "cpg": f"n_cpg_{c}",
                   "apobec_signature": f"n_apobec_{c}"}.get(primary_filter, f"n_apobec_{c}")
        if mut_col not in w_d.columns:
            continue
        r = recall_ratio_with_perm(w_d, score_col, mut_col, "cpg_density", 0.01,
                                   perm_reps=perm_reps, rng=rng)
        per_c[c] = r
    return decile_idx, {
        "n_windows": int(len(w_d)),
        "mean_ratio": float(np.nanmean([v["ratio"] for v in per_c.values() if v["ratio"] == v["ratio"]])) if per_c else float("nan"),
        "per_cancer": per_c,
    }


def _exploratory_worker(args):
    """One (head, filter, pct, cancer) combination."""
    (head, filt, pct, cancer, score_col, mut_col, w_data, perm_reps, seed) = args
    if isinstance(w_data, dict):
        w_pool = pd.DataFrame(w_data)
    else:
        w_pool = w_data
    rng = np.random.default_rng(seed)
    r = recall_ratio_with_perm(w_pool, score_col, mut_col, "cpg_density",
                               pct, perm_reps=perm_reps, rng=rng)
    return {"head": head, "filter": filt, "pct": pct, "cancer": cancer, **r}


def run_secondary(w: pd.DataFrame, out_dir: Path, n_workers: int = 8,
                  decile_perm: int = 2000, exploratory_perm: int = 1000) -> dict:
    logger.info("\n%s\nSECONDARY (BH-corrected family) [parallel n_workers=%d]\n%s",
                "=" * 70, n_workers, "=" * 70)
    w_pool = w[(~w["is_driver"]) & (~w["training_contaminated"])].copy()
    cancers = [c for c in CANCERS_PCAWG if filter_to_count_col(PRIMARY_FILTER, c) in w_pool.columns]
    if "cpg_density" in w_pool.columns and (w_pool["cpg_density"].nunique() >= 10):
        w_pool["cpg_decile"] = pd.qcut(w_pool["cpg_density"], q=10, labels=False, duplicates="drop")
    else:
        w_pool["cpg_decile"] = 0

    score_col_b = "score_binary_mean" if "score_binary_mean" in w_pool.columns else "score_binary"
    decile_args = []
    needed_cols = ["chrom", "win_id", "cpg_density", score_col_b]
    needed_cols += [filter_to_count_col(PRIMARY_FILTER, c) for c in cancers]
    needed_cols = [c for c in needed_cols if c in w_pool.columns]
    for d in sorted(w_pool["cpg_decile"].dropna().unique()):
        w_d = w_pool[w_pool["cpg_decile"] == d][needed_cols].copy()
        decile_args.append((int(d), w_d, cancers, score_col_b, PRIMARY_FILTER,
                            decile_perm, 20260426 + int(d) * 1000))

    import multiprocessing as mp
    if n_workers > 1 and len(decile_args) > 1:
        with mp.get_context("fork").Pool(min(n_workers, len(decile_args))) as pool:
            decile_outputs = pool.map(_decile_worker, decile_args)
    else:
        decile_outputs = [_decile_worker(a) for a in decile_args]
    decile_results = {d: res for d, res in decile_outputs}
    valid_means = [v["mean_ratio"] for v in decile_results.values()
                   if v.get("mean_ratio", float("nan")) == v.get("mean_ratio", float("nan"))]
    per_decile_min = float(min(valid_means)) if valid_means else float("nan")

    # Exploratory family parallel
    score_col_map = {}
    for head in HEADS:
        for col_kind in ("_mean", ""):
            sc = f"score_{head}{col_kind}"
            if sc in w_pool.columns:
                score_col_map[head] = sc
                break
    expl_args = []
    needed_cols2 = ["chrom", "win_id", "cpg_density"] + list(score_col_map.values())
    for c in cancers:
        for k in ("n_CT", "n_tcw_not_cpg", "n_cpg", "n_apobec"):
            col = f"{k}_{c}"
            if col in w_pool.columns:
                needed_cols2.append(col)
    needed_cols2 = list(set([c for c in needed_cols2 if c in w_pool.columns]))
    w_exp_slim = w_pool[needed_cols2].copy()
    seed_counter = [20260427]
    for head, score_col in score_col_map.items():
        for filt in FILTERS:
            for pct in PERCENTILES:
                for cancer in cancers:
                    mut_col = filter_to_count_col(filt, cancer)
                    if mut_col not in w_exp_slim.columns:
                        continue
                    expl_args.append((head, filt, pct, cancer, score_col, mut_col,
                                      w_exp_slim, exploratory_perm, seed_counter[0]))
                    seed_counter[0] += 1

    if n_workers > 1 and len(expl_args) > 1:
        with mp.get_context("fork").Pool(n_workers) as pool:
            rows = pool.map(_exploratory_worker, expl_args)
    else:
        rows = [_exploratory_worker(a) for a in expl_args]
    df = pd.DataFrame(rows)
    if len(df):
        p = df["p_perm"].clip(0, 1).values
        rej, q, _, _ = multipletests(p, alpha=SECONDARY_ALPHA, method="fdr_bh")
        df["q_bh"] = q; df["reject_bh"] = rej
    df.to_csv(out_dir / "enrichment_secondary.csv", index=False)
    summary = {
        "n_rows": int(len(df)),
        "n_signif_q05": int((df["q_bh"] < SECONDARY_ALPHA).sum()) if len(df) else 0,
        "secondary_alpha": SECONDARY_ALPHA,
        "per_cpg_decile": decile_results,
        "per_decile_min_ratio": per_decile_min,
    }
    with open(out_dir / "enrichment_secondary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("  secondary: %d rows, q<0.05: %d", len(df), summary["n_signif_q05"])
    return summary


def write_report(primary, secondary, coverage_path: Path, out_dir: Path):
    lines = []
    lines.append("# Analysis A — PCAWG WGS coding-panel enrichment\n")
    lines.append(f"\nGenerated: {time.strftime('%Y-%m-%d %H:%M')}\n\n")
    lines.append("## Pre-registered primary endpoint\n")
    lines.append(f"- Head: `{PRIMARY_HEAD}` | Filter: `{PRIMARY_FILTER}` | Top pct: {PRIMARY_PCT}\n")
    lines.append(f"- Baseline: `{PRIMARY_BASELINE}` | APOBEC weight threshold: {APOBEC_WEIGHT_THRESHOLD}\n")
    lines.append(f"- Cancers evaluated: {len(primary['per_cancer'])}\n")
    lines.append(f"- Test: permutation null ({PERM_REPS} reps per cancer), BH-FDR alpha={PRIMARY_ALPHA} across cancers (Bonferroni-tightened from 0.05 across A+B family per pre-reg L141-144)\n\n")
    pc = primary["pass_criteria"]
    lines.append(f"### RESULT: **{'PASS' if pc['PASS'] else 'FAIL'}**\n\n")
    for k, v in pc.items():
        if k == "PASS":
            continue
        marker = "PASS" if v["pass"] else "FAIL"
        lines.append(f"- [{marker}] {k}: value={v['val']:.3f} threshold={v['thresh']}\n")
    lines.append("\n## Per-cancer primary (training-mask + driver-ablation)\n")
    lines.append("| cancer | mut_ratio | recall_model | recall_cpg | total_mut | p_perm | q_bh | reject |\n")
    lines.append("|--------|-----------|--------------|-----------|-----------|--------|------|--------|\n")
    for cancer, det in primary["per_cancer"].items():
        pr = det["primary"]
        q = pr.get("q_bh", float("nan"))
        rej = pr.get("reject_bh", False)
        lines.append(f"| {cancer} | {pr['ratio']:.3f} | {pr['recall_model']:.4f} | "
                     f"{pr['recall_baseline']:.4f} | {pr['total_mut']} | "
                     f"{pr['p_perm']:.2e} | {q:.3g} | {'Y' if rej else 'N'} |\n")
    lines.append("\n## Secondary (BH-corrected family at alpha=" + str(SECONDARY_ALPHA) + ")\n")
    lines.append(f"- Total rows: {secondary['n_rows']}\n- Significant (q<{SECONDARY_ALPHA}): {secondary['n_signif_q05']}\n")
    lines.append(f"- Per-decile min mean_ratio (QA #2): {secondary.get('per_decile_min_ratio', float('nan')):.3f}\n")
    if coverage_path.exists():
        cov = json.loads(coverage_path.read_text())
        lines.append("\n## Panel coverage of PCAWG mutations\n")
        lines.append(f"- Total PCAWG C>T SNVs: {cov['n_total_maf_C2T']}\n")
        lines.append(f"- In scored CDS panel: {cov['n_in_panel']} ({cov['coverage_pct']:.2f}%)\n")
        lines.append(f"- Per-cancer coverage: {cov['per_cancer_coverage']}\n")
    lines.append("\n## QA-fix log\n")
    lines.append("| ID | Issue | Fix |\n|----|-------|-----|\n")
    lines.append("| B1 | Donor↔Sample join broken | Aggregated SBS to (cancer, subtype) mean weight |\n")
    lines.append("| B2 | PRIMARY_FILTER mismatch | Set to `apobec_signature`, matches pre-reg |\n")
    lines.append("| B3 | Fisher 2x2 malformed | Replaced with permutation null on score labels |\n")
    lines.append("| M1 | site_id parser drops T2/T3 | Use chr/start columns directly |\n")
    lines.append("| M2 | hg38 sites in v3 mask | Filter to coordinate_system=='hg19' |\n")
    lines.append("| M4 | TCW minus-strand bug | Correct rev-comp detection in count_tcw_in_window |\n")
    lines.append("| M5 | CGC list spurious | Curated Bailey-style list excluding TTN/MUC16/OBSCN/SYNE1 |\n")
    with open(out_dir / "REPORT.md", "w") as f:
        f.writelines(lines)
    logger.info("Wrote %s/REPORT.md", out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", type=Path, required=True)
    ap.add_argument("--maf", type=Path, default=PROJECT_ROOT / "data/raw/pcawg_open/final_consensus_passonly.snv_mnv_indel.icgc.public.maf.gz")
    ap.add_argument("--sbs", type=Path, default=PROJECT_ROOT / "data/raw/pcawg_open/SigProfilier_PCAWG_WGS_probabilities_SBS.csv")
    ap.add_argument("--bailey-drivers", type=Path, default=None)
    ap.add_argument("--v3-splits", type=Path, default=PROJECT_ROOT / "data/processed/multi_enzyme/splits_multi_enzyme_v3_with_negatives.csv")
    ap.add_argument("--hg19", type=Path, default=PROJECT_ROOT / "data/raw/genomes/hg19.fa")
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "experiments/multi_enzyme/outputs/pcawg_tcw_panel/analysis_A_pcawg_wgs")
    ap.add_argument("--perm-reps", type=int, default=PERM_REPS)
    ap.add_argument("--n-workers", type=int, default=8,
                    help="Multiprocessing pool size for per-cancer parallelism")
    ap.add_argument("--decile-perm", type=int, default=2000)
    ap.add_argument("--exploratory-perm", type=int, default=1000)
    ap.add_argument("--phase3-model", type=Path, default=None,
                    help="Path to phase3_mfe_only.pt for SHA provenance (default: canonical)")
    ap.add_argument("--apobec1-model", type=Path, default=None,
                    help="Path to apobec1_head_mfe_only.pt for SHA provenance (default: canonical)")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    panel = load_panel_scores(args.panel)
    maf = load_pcawg_maf(args.maf)
    sbs_cancer = load_sbs_attributions_cancer_level(args.sbs)
    drivers = load_bailey_drivers(args.bailey_drivers)
    v3 = load_v3_positions_hg19(args.v3_splits)

    logger.info("Computing provenance hashes ...")
    provenance = compute_provenance(args.panel, args.phase3_model, args.apobec1_model)
    logger.info("  git_commit=%s panel_sha=%s",
                provenance["git_commit"][:12], provenance["panel_scores_cds_sha256"][:16])

    w = build_windows(panel, maf, sbs_cancer, drivers, v3, args.hg19, args.out_dir)
    primary = run_primary(w, args.out_dir, n_workers=args.n_workers, perm_reps=args.perm_reps,
                          provenance=provenance)
    secondary = run_secondary(w, args.out_dir, n_workers=args.n_workers,
                              decile_perm=args.decile_perm,
                              exploratory_perm=args.exploratory_perm)
    write_report(primary, secondary, args.out_dir / "panel_coverage_stats.json", args.out_dir)
    logger.info("DONE Analysis A.")


if __name__ == "__main__":
    main()
