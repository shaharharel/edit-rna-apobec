#!/usr/bin/env python
"""Cross-species comparison: Human vs Chimpanzee APOBEC RNA editing sites.

For each editing site in human (hg38), lifts over to chimp (panTro6) and asks:
1. Is the edited C conserved in chimp?
2. Is the upstream motif (TC/CC) preserved?
3. Does the chimp ortholog score as "editable" by our GB model?
4. Are editing sites more/less conserved than matched controls?

Usage:
    conda run -n quris python scripts/multi_enzyme/cross_species_comparison.py
"""

import gc
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyliftover import LiftOver
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.apobec_feature_extraction import (
    extract_motif_features,
    extract_loop_features,
    extract_structure_delta_features,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Paths
SPLITS_CSV = PROJECT_ROOT / "data/processed/multi_enzyme/splits_multi_enzyme_v3_with_negatives.csv"
SEQ_JSON = PROJECT_ROOT / "data/processed/multi_enzyme/multi_enzyme_sequences_v3_with_negatives.json"
LOOP_CSV = PROJECT_ROOT / "data/processed/multi_enzyme/loop_position_per_site_v3.csv"
STRUCT_CACHE = PROJECT_ROOT / "data/processed/multi_enzyme/structure_cache_multi_enzyme_v3.npz"

HG38_FA = PROJECT_ROOT / "data/raw/genomes/hg38.fa"
HG19_FA = PROJECT_ROOT / "data/raw/genomes/hg19.fa"
PANTRO6_FA = PROJECT_ROOT / "data/raw/genomes/panTro6.fa"

OUTPUT_DIR = PROJECT_ROOT / "experiments/multi_enzyme/outputs/cross_species"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
WINDOW = 100  # ±100 nt around edit site


def load_genome(fa_path):
    """Load genome with pyfaidx."""
    from pyfaidx import Fasta
    return Fasta(str(fa_path))


def reverse_complement(seq):
    comp = str.maketrans("ACGTNacgtn", "TGCANtgcan")
    return seq.translate(comp)[::-1]


def extract_window(genome, chrom, pos, strand, window=100):
    """Extract 2*window+1 nt centered on pos (0-based).
    Returns RNA-sense sequence with edit site at center."""
    try:
        start = max(0, pos - window)
        end = pos + window + 1
        seq = str(genome[chrom][start:end]).upper()
        if len(seq) < 2 * window + 1:
            return None
        if strand == "-":
            seq = reverse_complement(seq)
        return seq
    except (KeyError, ValueError):
        return None


def get_trinucleotide(seq, center=100):
    """Get trinucleotide context at center position of 201-nt sequence."""
    if seq is None or len(seq) < center + 2:
        return None
    return seq[center - 1:center + 2]


def get_dinucleotide_5p(seq, center=100):
    """Get 5' dinucleotide (position -1 and 0) from 201-nt sequence."""
    if seq is None or len(seq) < center + 1:
        return None
    return seq[center - 1:center + 1]


def compute_motif_density(seq, motifs=("TC", "CC")):
    """Count APOBEC motif density in a sequence (per kb)."""
    if seq is None:
        return {m: 0.0 for m in motifs}
    seq = seq.upper().replace("U", "T")
    results = {}
    for motif in motifs:
        count = sum(1 for i in range(len(seq) - len(motif) + 1)
                    if seq[i:i + len(motif)] == motif)
        results[motif] = count / (len(seq) / 1000.0) if len(seq) > 0 else 0.0
    return results


def compute_substitution_rate(human_seq, chimp_seq):
    """Compute substitution rate between two aligned sequences."""
    if human_seq is None or chimp_seq is None:
        return None
    if len(human_seq) != len(chimp_seq):
        return None
    diffs = sum(1 for a, b in zip(human_seq, chimp_seq)
                if a != "N" and b != "N" and a != b)
    valid = sum(1 for a, b in zip(human_seq, chimp_seq)
                if a != "N" and b != "N")
    return diffs / valid if valid > 0 else None


def run_cross_species():
    t_start = time.time()

    # ---------------------------------------------------------------
    # 1. Load editing sites
    # ---------------------------------------------------------------
    logger.info("Loading editing sites...")
    df = pd.read_csv(SPLITS_CSV)
    # Only positives with hg38 coordinates
    positives = df[(df["is_edited"] == 1) & (df["chr"].notna())].copy()
    logger.info("  Total positives with coordinates: %d", len(positives))

    # Also get negatives for matched controls
    negatives = df[(df["is_edited"] == 0) & (df["chr"].notna())].copy()
    logger.info("  Total negatives with coordinates: %d", len(negatives))

    # ---------------------------------------------------------------
    # 2. LiftOver → panTro6 (handles both hg19 and hg38 sources)
    # ---------------------------------------------------------------
    logger.info("LiftOver → panTro6...")
    lo_hg38 = LiftOver("hg38", "panTro6")
    lo_hg19 = LiftOver("hg19", "panTro6")

    def liftover_site(row):
        try:
            chrom = str(row["chr"])
            pos = int(row["start"])
            coord_sys = str(row.get("coordinate_system", "hg38"))
            lo = lo_hg19 if coord_sys == "hg19" else lo_hg38
            result = lo.convert_coordinate(chrom, pos)
            if result and len(result) > 0:
                return result[0]  # (chrom, pos, strand, score)
        except Exception:
            pass
        return None

    # Lift positives
    pos_lifted = []
    pos_failed = 0
    for _, row in positives.iterrows():
        r = liftover_site(row)
        if r is not None:
            pos_lifted.append({
                "site_id": row["site_id"],
                "enzyme": row.get("enzyme", "unknown"),
                "human_chr": row["chr"],
                "human_pos": int(row["start"]),
                "human_strand": row.get("strand", "+"),
                "coord_system_orig": row.get("coordinate_system", "hg38"),
                "chimp_chr": r[0],
                "chimp_pos": r[1],
                "chimp_strand": r[2],
                "liftover_score": r[3],
            })
        else:
            pos_failed += 1

    logger.info("  Positives: %d lifted, %d failed (%.1f%%)",
                len(pos_lifted), pos_failed,
                100 * pos_failed / max(len(positives), 1))

    # Lift negatives (sample same size for controls)
    neg_sample = negatives.sample(n=min(len(negatives), len(positives) * 2),
                                   random_state=SEED)
    neg_lifted = []
    neg_failed = 0
    for _, row in neg_sample.iterrows():
        r = liftover_site(row)
        if r is not None:
            neg_lifted.append({
                "site_id": row["site_id"],
                "enzyme": row.get("enzyme", "unknown"),
                "human_chr": row["chr"],
                "human_pos": int(row["start"]),
                "human_strand": row.get("strand", "+"),
                "coord_system_orig": row.get("coordinate_system", "hg38"),
                "chimp_chr": r[0],
                "chimp_pos": r[1],
                "chimp_strand": r[2],
                "liftover_score": r[3],
            })
        else:
            neg_failed += 1

    logger.info("  Negatives: %d lifted, %d failed (%.1f%%)",
                len(neg_lifted), neg_failed,
                100 * neg_failed / max(len(neg_sample), 1))

    pos_df = pd.DataFrame(pos_lifted)
    neg_df = pd.DataFrame(neg_lifted)

    # ---------------------------------------------------------------
    # 3. Extract sequences from both genomes
    # ---------------------------------------------------------------
    logger.info("Loading genomes...")
    hg38 = load_genome(HG38_FA)
    hg19 = load_genome(HG19_FA) if HG19_FA.exists() else None

    if not PANTRO6_FA.exists():
        logger.error("panTro6.fa not found at %s. Download and gunzip first.", PANTRO6_FA)
        logger.error("  curl -L https://hgdownload.soe.ucsc.edu/goldenPath/panTro6/bigZips/panTro6.fa.gz | gunzip > %s", PANTRO6_FA)
        sys.exit(1)

    pantro6 = load_genome(PANTRO6_FA)
    logger.info("  Genomes loaded (hg38, hg19=%s, panTro6).", "yes" if hg19 else "no")

    def extract_seqs(site_df, label):
        """Extract human and chimp sequences for a set of sites."""
        human_seqs = []
        chimp_seqs = []
        for _, row in site_df.iterrows():
            # Use correct human genome based on coordinate system
            coord_sys = str(row.get("coord_system_orig", "hg38"))
            human_genome = hg19 if (coord_sys == "hg19" and hg19 is not None) else hg38
            h_seq = extract_window(human_genome, row["human_chr"],
                                    row["human_pos"], row["human_strand"],
                                    WINDOW)
            c_seq = extract_window(pantro6, row["chimp_chr"],
                                    row["chimp_pos"], row["chimp_strand"],
                                    WINDOW)
            human_seqs.append(h_seq)
            chimp_seqs.append(c_seq)
        site_df = site_df.copy()
        site_df["human_seq"] = human_seqs
        site_df["chimp_seq"] = chimp_seqs
        n_both = sum(1 for h, c in zip(human_seqs, chimp_seqs) if h and c)
        logger.info("  %s: %d/%d have both human and chimp sequences",
                     label, n_both, len(site_df))
        return site_df

    pos_df = extract_seqs(pos_df, "Positives")
    neg_df = extract_seqs(neg_df, "Negatives")

    # ---------------------------------------------------------------
    # 3b. Filter to true orthologs (sub_rate < 5%)
    # Many sites (esp. Kockler cancer cell line) have unreliable coordinates
    # that liftOver to non-orthologous chimp regions.
    # ---------------------------------------------------------------

    def compute_sub_rate_quick(row):
        return compute_substitution_rate(row["human_seq"], row["chimp_seq"])

    pos_df["sub_rate_raw"] = pos_df.apply(compute_sub_rate_quick, axis=1)
    neg_df["sub_rate_raw"] = neg_df.apply(compute_sub_rate_quick, axis=1)

    ORTHOLOG_THRESHOLD = 0.05  # max 5% divergence for true orthologs
    n_pos_before = len(pos_df)
    n_neg_before = len(neg_df)
    pos_df = pos_df[pos_df["sub_rate_raw"] < ORTHOLOG_THRESHOLD].copy()
    neg_df = neg_df[neg_df["sub_rate_raw"] < ORTHOLOG_THRESHOLD].copy()

    logger.info("  Ortholog filter (<%.0f%% divergence): pos %d→%d, neg %d→%d",
                100 * ORTHOLOG_THRESHOLD,
                n_pos_before, len(pos_df), n_neg_before, len(neg_df))

    # ---------------------------------------------------------------
    # 4. Core analyses (on true orthologs only)
    # ---------------------------------------------------------------

    results = {}
    results["ortholog_filter"] = {
        "threshold": ORTHOLOG_THRESHOLD,
        "positives_before": n_pos_before,
        "positives_after": len(pos_df),
        "negatives_before": n_neg_before,
        "negatives_after": len(neg_df),
    }

    # 4a. Conservation at edit position
    logger.info("Analyzing conservation at edit position...")

    def check_center_conservation(row):
        """Check if center base is conserved (C in human, C in chimp)."""
        h = row["human_seq"]
        c = row["chimp_seq"]
        if h is None or c is None:
            return {"human_base": None, "chimp_base": None, "conserved": None}
        center = WINDOW
        hb = h[center] if len(h) > center else None
        cb = c[center] if len(c) > center else None
        return {
            "human_base": hb,
            "chimp_base": cb,
            "conserved": (hb == cb) if (hb and cb) else None,
        }

    pos_conservation = pos_df.apply(check_center_conservation, axis=1,
                                     result_type="expand")
    neg_conservation = neg_df.apply(check_center_conservation, axis=1,
                                     result_type="expand")

    pos_df = pd.concat([pos_df, pos_conservation], axis=1)
    neg_df = pd.concat([neg_df, neg_conservation], axis=1)

    pos_valid = pos_df[pos_df["conserved"].notna()]
    neg_valid = neg_df[neg_df["conserved"].notna()]

    pos_c_conserved = pos_valid["conserved"].sum()
    neg_c_conserved = neg_valid["conserved"].sum()

    results["center_conservation"] = {
        "positives": {
            "total": int(len(pos_valid)),
            "conserved": int(pos_c_conserved),
            "rate": float(pos_c_conserved / max(len(pos_valid), 1)),
        },
        "negatives": {
            "total": int(len(neg_valid)),
            "conserved": int(neg_c_conserved),
            "rate": float(neg_c_conserved / max(len(neg_valid), 1)),
        },
    }
    # Fisher exact test: is the conservation rate different?
    a = int(pos_c_conserved)
    b = int(len(pos_valid) - pos_c_conserved)
    c = int(neg_c_conserved)
    d = int(len(neg_valid) - neg_c_conserved)
    if a + b > 0 and c + d > 0:
        fe_or, fe_p = stats.fisher_exact([[a, b], [c, d]])
        results["center_conservation"]["fisher_or"] = float(fe_or)
        results["center_conservation"]["fisher_p"] = float(fe_p)

    logger.info("  Positives: %d/%d (%.1f%%) C conserved in chimp",
                pos_c_conserved, len(pos_valid),
                100 * pos_c_conserved / max(len(pos_valid), 1))
    logger.info("  Negatives: %d/%d (%.1f%%) C conserved in chimp",
                neg_c_conserved, len(neg_valid),
                100 * neg_c_conserved / max(len(neg_valid), 1))

    # 4b. Motif preservation
    logger.info("Analyzing motif preservation...")

    def check_motif_preservation(row):
        h = row["human_seq"]
        c = row["chimp_seq"]
        if h is None or c is None:
            return {"human_dinuc": None, "chimp_dinuc": None,
                    "human_trinuc": None, "chimp_trinuc": None,
                    "motif_preserved": None}
        hd = get_dinucleotide_5p(h, WINDOW)
        cd = get_dinucleotide_5p(c, WINDOW)
        ht = get_trinucleotide(h, WINDOW)
        ct = get_trinucleotide(c, WINDOW)
        return {
            "human_dinuc": hd, "chimp_dinuc": cd,
            "human_trinuc": ht, "chimp_trinuc": ct,
            "motif_preserved": (hd == cd) if (hd and cd) else None,
        }

    pos_motif = pos_df.apply(check_motif_preservation, axis=1, result_type="expand")
    neg_motif = neg_df.apply(check_motif_preservation, axis=1, result_type="expand")
    pos_df = pd.concat([pos_df, pos_motif], axis=1)
    neg_df = pd.concat([neg_df, neg_motif], axis=1)

    pos_motif_valid = pos_df[pos_df["motif_preserved"].notna()]
    neg_motif_valid = neg_df[neg_df["motif_preserved"].notna()]

    results["motif_preservation"] = {
        "positives": {
            "total": int(len(pos_motif_valid)),
            "preserved": int(pos_motif_valid["motif_preserved"].sum()),
            "rate": float(pos_motif_valid["motif_preserved"].mean()),
        },
        "negatives": {
            "total": int(len(neg_motif_valid)),
            "preserved": int(neg_motif_valid["motif_preserved"].sum()),
            "rate": float(neg_motif_valid["motif_preserved"].mean()),
        },
    }

    # Per-enzyme motif preservation
    enzyme_motif = {}
    for enz in pos_df["enzyme"].unique():
        enz_df = pos_df[(pos_df["enzyme"] == enz) & (pos_df["motif_preserved"].notna())]
        if len(enz_df) > 0:
            enzyme_motif[enz] = {
                "n": int(len(enz_df)),
                "preserved": int(enz_df["motif_preserved"].sum()),
                "rate": float(enz_df["motif_preserved"].mean()),
            }
    results["motif_preservation"]["per_enzyme"] = enzyme_motif

    logger.info("  Positives: %d/%d (%.1f%%) dinucleotide motif preserved",
                int(pos_motif_valid["motif_preserved"].sum()),
                len(pos_motif_valid),
                100 * pos_motif_valid["motif_preserved"].mean())

    # 4c. Substitution rates
    logger.info("Analyzing substitution rates...")

    def compute_sub_rate(row):
        return compute_substitution_rate(row["human_seq"], row["chimp_seq"])

    pos_df["sub_rate"] = pos_df.apply(compute_sub_rate, axis=1)
    neg_df["sub_rate"] = neg_df.apply(compute_sub_rate, axis=1)

    pos_rates = pos_df["sub_rate"].dropna()
    neg_rates = neg_df["sub_rate"].dropna()

    if len(pos_rates) > 10 and len(neg_rates) > 10:
        mw_stat, mw_p = stats.mannwhitneyu(pos_rates, neg_rates, alternative="two-sided")
        results["substitution_rates"] = {
            "positives": {"mean": float(pos_rates.mean()), "median": float(pos_rates.median()),
                          "std": float(pos_rates.std()), "n": int(len(pos_rates))},
            "negatives": {"mean": float(neg_rates.mean()), "median": float(neg_rates.median()),
                          "std": float(neg_rates.std()), "n": int(len(neg_rates))},
            "mannwhitney_U": float(mw_stat),
            "mannwhitney_p": float(mw_p),
        }
        logger.info("  Pos sub rate: %.4f ± %.4f (n=%d)",
                     pos_rates.mean(), pos_rates.std(), len(pos_rates))
        logger.info("  Neg sub rate: %.4f ± %.4f (n=%d)",
                     neg_rates.mean(), neg_rates.std(), len(neg_rates))
        logger.info("  Mann-Whitney p=%.2e", mw_p)

    # Per-enzyme substitution rates
    enzyme_sub = {}
    for enz in pos_df["enzyme"].unique():
        enz_rates = pos_df[pos_df["enzyme"] == enz]["sub_rate"].dropna()
        if len(enz_rates) > 5:
            enzyme_sub[enz] = {
                "mean": float(enz_rates.mean()),
                "median": float(enz_rates.median()),
                "n": int(len(enz_rates)),
            }
    results["substitution_rates_per_enzyme"] = enzyme_sub

    # 4d. Motif density comparison
    logger.info("Analyzing APOBEC motif density in human vs chimp...")

    def compute_densities(row):
        h_dens = compute_motif_density(row["human_seq"])
        c_dens = compute_motif_density(row["chimp_seq"])
        return {
            "human_TC_density": h_dens["TC"],
            "chimp_TC_density": c_dens["TC"],
            "human_CC_density": h_dens["CC"],
            "chimp_CC_density": c_dens["CC"],
        }

    pos_dens = pos_df.apply(compute_densities, axis=1, result_type="expand")
    pos_df = pd.concat([pos_df, pos_dens], axis=1)

    neg_dens = neg_df.apply(compute_densities, axis=1, result_type="expand")
    neg_df = pd.concat([neg_df, neg_dens], axis=1)

    # Compare TC density change: human - chimp
    pos_tc_delta = (pos_df["human_TC_density"] - pos_df["chimp_TC_density"]).dropna()
    neg_tc_delta = (neg_df["human_TC_density"] - neg_df["chimp_TC_density"]).dropna()

    results["motif_density"] = {}
    for motif in ["TC", "CC"]:
        h_col = f"human_{motif}_density"
        c_col = f"chimp_{motif}_density"
        pos_h = pos_df[h_col].dropna()
        pos_c = pos_df[c_col].dropna()
        neg_h = neg_df[h_col].dropna()
        neg_c = neg_df[c_col].dropna()

        pos_delta = (pos_df[h_col] - pos_df[c_col]).dropna()
        neg_delta = (neg_df[h_col] - neg_df[c_col]).dropna()

        if len(pos_delta) > 10 and len(neg_delta) > 10:
            stat, p = stats.mannwhitneyu(pos_delta, neg_delta, alternative="two-sided")
            results["motif_density"][motif] = {
                "pos_human_mean": float(pos_h.mean()),
                "pos_chimp_mean": float(pos_c.mean()),
                "pos_delta_mean": float(pos_delta.mean()),
                "neg_human_mean": float(neg_h.mean()),
                "neg_chimp_mean": float(neg_c.mean()),
                "neg_delta_mean": float(neg_delta.mean()),
                "mannwhitney_p": float(p),
            }

    # 4e. Chimp editability scoring
    logger.info("Scoring chimp orthologs with GB model features...")

    # Build chimp sequences dict for feature extraction
    chimp_seq_dict = {}
    for _, row in pos_df.iterrows():
        if row["chimp_seq"] is not None and len(str(row["chimp_seq"])) == 2 * WINDOW + 1:
            chimp_seq_dict[str(row["site_id"]) + "_chimp"] = str(row["chimp_seq"])

    human_seq_dict = {}
    for _, row in pos_df.iterrows():
        if row["human_seq"] is not None and len(str(row["human_seq"])) == 2 * WINDOW + 1:
            human_seq_dict[str(row["site_id"]) + "_human"] = str(row["human_seq"])

    # Extract motif features for both
    if chimp_seq_dict and human_seq_dict:
        chimp_ids = list(chimp_seq_dict.keys())
        human_ids = list(human_seq_dict.keys())

        chimp_motif = extract_motif_features(chimp_seq_dict, chimp_ids)
        human_motif = extract_motif_features(human_seq_dict, human_ids)

        # Compare motif feature vectors
        # Hamming-like distance between motif feature vectors
        paired = []
        for sid in pos_df["site_id"].values:
            h_key = str(sid) + "_human"
            c_key = str(sid) + "_chimp"
            if h_key in human_seq_dict and c_key in chimp_seq_dict:
                h_idx = human_ids.index(h_key)
                c_idx = chimp_ids.index(c_key)
                diff = np.abs(human_motif[h_idx] - chimp_motif[c_idx]).sum()
                paired.append({"site_id": sid, "motif_feature_diff": float(diff)})

        paired_df = pd.DataFrame(paired)
        results["editability_scoring"] = {
            "n_paired": int(len(paired_df)),
            "mean_motif_feature_diff": float(paired_df["motif_feature_diff"].mean()),
            "median_motif_feature_diff": float(paired_df["motif_feature_diff"].median()),
            "pct_identical_motif": float((paired_df["motif_feature_diff"] == 0).mean()),
        }
        logger.info("  %d paired sites; %.1f%% have identical motif features",
                     len(paired_df),
                     100 * (paired_df["motif_feature_diff"] == 0).mean())

    # 4f. Per-enzyme breakdown
    logger.info("Per-enzyme conservation summary...")
    enzyme_conservation = {}
    for enz in pos_df["enzyme"].unique():
        enz_df = pos_df[(pos_df["enzyme"] == enz) & (pos_df["conserved"].notna())]
        if len(enz_df) > 0:
            enzyme_conservation[enz] = {
                "n": int(len(enz_df)),
                "c_conserved": int(enz_df["conserved"].sum()),
                "c_conservation_rate": float(enz_df["conserved"].mean()),
                "motif_preserved": float(enz_df[enz_df["motif_preserved"].notna()]["motif_preserved"].mean())
                    if enz_df["motif_preserved"].notna().any() else None,
            }
    results["per_enzyme_conservation"] = enzyme_conservation

    # ---------------------------------------------------------------
    # 5. Generate figures
    # ---------------------------------------------------------------
    logger.info("Generating figures...")

    # Fig 1: Conservation rates by category
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1a: Center base conservation
    ax = axes[0]
    cats = ["Editing sites", "Controls"]
    cons_rates = [
        results["center_conservation"]["positives"]["rate"],
        results["center_conservation"]["negatives"]["rate"],
    ]
    bars = ax.bar(cats, cons_rates, color=["#2563eb", "#94a3b8"])
    ax.set_ylabel("Conservation rate")
    ax.set_title("Center C conservation\n(human → chimp)")
    ax.set_ylim(0, 1.05)
    for bar, rate in zip(bars, cons_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{rate:.1%}", ha="center", fontsize=10)
    p_val = results["center_conservation"].get("fisher_p", 1.0)
    ax.set_xlabel(f"Fisher p={p_val:.2e}" if p_val < 0.05 else f"Fisher p={p_val:.3f} (ns)")

    # 1b: Motif preservation
    ax = axes[1]
    motif_rates = [
        results["motif_preservation"]["positives"]["rate"],
        results["motif_preservation"]["negatives"]["rate"],
    ]
    bars = ax.bar(cats, motif_rates, color=["#2563eb", "#94a3b8"])
    ax.set_ylabel("Preservation rate")
    ax.set_title("5' dinucleotide motif preservation\n(human → chimp)")
    ax.set_ylim(0, 1.05)
    for bar, rate in zip(bars, motif_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{rate:.1%}", ha="center", fontsize=10)

    # 1c: Substitution rates
    ax = axes[2]
    if "substitution_rates" in results:
        data_to_plot = [pos_rates.values, neg_rates.values]
        bp = ax.boxplot(data_to_plot, labels=cats, patch_artist=True)
        bp["boxes"][0].set_facecolor("#2563eb")
        bp["boxes"][0].set_alpha(0.5)
        bp["boxes"][1].set_facecolor("#94a3b8")
        bp["boxes"][1].set_alpha(0.5)
        ax.set_ylabel("Substitution rate")
        ax.set_title("Human-chimp divergence\n(±100bp window)")
        p_val = results["substitution_rates"].get("mannwhitney_p", 1.0)
        ax.set_xlabel(f"Mann-Whitney p={p_val:.2e}" if p_val < 0.05 else f"p={p_val:.3f} (ns)")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "conservation_overview.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Fig 2: Per-enzyme conservation
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    enzymes_sorted = sorted(enzyme_conservation.keys(),
                             key=lambda e: enzyme_conservation[e]["n"], reverse=True)

    # 2a: Center C conservation per enzyme
    ax = axes[0]
    enz_names = [e for e in enzymes_sorted if enzyme_conservation[e]["n"] >= 10]
    enz_c_rates = [enzyme_conservation[e]["c_conservation_rate"] for e in enz_names]
    enz_n = [enzyme_conservation[e]["n"] for e in enz_names]
    colors = ["#2563eb", "#16a34a", "#dc2626", "#eab308", "#8b5cf6", "#f97316"]
    bars = ax.barh(enz_names, enz_c_rates,
                    color=[colors[i % len(colors)] for i in range(len(enz_names))])
    ax.set_xlabel("C conservation rate (human → chimp)")
    ax.set_title("Center C conservation by enzyme")
    ax.set_xlim(0, 1.05)
    for bar, rate, n in zip(bars, enz_c_rates, enz_n):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{rate:.1%} (n={n})", va="center", fontsize=9)

    # 2b: Motif preservation per enzyme
    ax = axes[1]
    enz_m_rates = []
    for e in enz_names:
        mp = enzyme_conservation[e].get("motif_preserved")
        enz_m_rates.append(mp if mp is not None else 0)
    bars = ax.barh(enz_names, enz_m_rates,
                    color=[colors[i % len(colors)] for i in range(len(enz_names))])
    ax.set_xlabel("Dinucleotide motif preservation rate")
    ax.set_title("Motif preservation by enzyme")
    ax.set_xlim(0, 1.05)
    for bar, rate in zip(bars, enz_m_rates):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{rate:.1%}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "per_enzyme_conservation.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Fig 3: Motif density comparison (human vs chimp, editing vs control)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for idx, motif in enumerate(["TC", "CC"]):
        ax = axes[idx]
        h_col = f"human_{motif}_density"
        c_col = f"chimp_{motif}_density"

        pos_h = pos_df[h_col].dropna()
        pos_c = pos_df[c_col].dropna()
        neg_h = neg_df[h_col].dropna()
        neg_c = neg_df[c_col].dropna()

        x = np.arange(2)
        width = 0.35
        ax.bar(x - width / 2,
               [pos_h.mean(), pos_c.mean()],
               width, label="Editing sites", color="#2563eb", alpha=0.7)
        ax.bar(x + width / 2,
               [neg_h.mean(), neg_c.mean()],
               width, label="Controls", color="#94a3b8", alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(["Human", "Chimp"])
        ax.set_ylabel(f"{motif} density (per kb)")
        ax.set_title(f"{motif} motif density")
        ax.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "motif_density_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Fig 4: Trinucleotide change heatmap (human → chimp at edit site)
    pos_with_tri = pos_df[pos_df["human_trinuc"].notna() & pos_df["chimp_trinuc"].notna()]
    if len(pos_with_tri) > 50:
        fig, ax = plt.subplots(figsize=(8, 6))
        # Count transitions
        transitions = Counter()
        for _, row in pos_with_tri.iterrows():
            ht = str(row["human_trinuc"]).replace("U", "T")
            ct = str(row["chimp_trinuc"]).replace("U", "T")
            transitions[(ht, ct)] += 1

        # Most common trinucleotides
        top_human = [t for t, _ in Counter(
            str(r["human_trinuc"]).replace("U", "T")
            for _, r in pos_with_tri.iterrows()
        ).most_common(8)]

        conserved_pct = []
        changed_to = []
        for ht in top_human:
            total = sum(v for (h, c), v in transitions.items() if h == ht)
            same = transitions.get((ht, ht), 0)
            conserved_pct.append(same / max(total, 1))
            # Most common non-self transition
            non_self = [(c, v) for (h, c), v in transitions.items()
                        if h == ht and c != ht]
            if non_self:
                top_change = max(non_self, key=lambda x: x[1])
                changed_to.append(f"→{top_change[0]} ({top_change[1]})")
            else:
                changed_to.append("")

        bars = ax.barh(top_human, conserved_pct, color="#2563eb", alpha=0.7)
        ax.set_xlabel("Trinucleotide conservation rate (human → chimp)")
        ax.set_title("Trinucleotide conservation at editing sites")
        ax.set_xlim(0, 1.05)
        for bar, pct, change in zip(bars, conserved_pct, changed_to):
            label = f"{pct:.1%}"
            if change:
                label += f"  {change}"
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    label, va="center", fontsize=8)

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "trinucleotide_conservation.png", dpi=150,
                    bbox_inches="tight")
        plt.close()

    # ---------------------------------------------------------------
    # 6. Save results
    # ---------------------------------------------------------------
    logger.info("Saving results...")

    # Summary
    results["summary"] = {
        "n_positives_input": int(len(positives)),
        "n_positives_lifted": int(len(pos_df)),
        "n_negatives_lifted": int(len(neg_df)),
        "liftover_success_rate": float(len(pos_df) / max(len(positives), 1)),
    }

    with open(OUTPUT_DIR / "cross_species_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Save detailed CSV
    cols_to_save = ["site_id", "enzyme", "human_chr", "human_pos", "human_strand",
                     "chimp_chr", "chimp_pos", "chimp_strand",
                     "conserved", "motif_preserved", "sub_rate",
                     "human_dinuc", "chimp_dinuc", "human_trinuc", "chimp_trinuc"]
    pos_df[[c for c in cols_to_save if c in pos_df.columns]].to_csv(
        OUTPUT_DIR / "editing_sites_cross_species.csv", index=False)

    elapsed = time.time() - t_start
    logger.info("\n=== Cross-Species Comparison Complete (%.0fs) ===", elapsed)
    logger.info("Center C conservation: pos=%.1f%% neg=%.1f%%",
                results["center_conservation"]["positives"]["rate"] * 100,
                results["center_conservation"]["negatives"]["rate"] * 100)
    logger.info("Motif preservation: pos=%.1f%% neg=%.1f%%",
                results["motif_preservation"]["positives"]["rate"] * 100,
                results["motif_preservation"]["negatives"]["rate"] * 100)
    if "substitution_rates" in results:
        logger.info("Substitution rate: pos=%.4f neg=%.4f (p=%.2e)",
                     results["substitution_rates"]["positives"]["mean"],
                     results["substitution_rates"]["negatives"]["mean"],
                     results["substitution_rates"]["mannwhitney_p"])
    logger.info("Output: %s", OUTPUT_DIR)


if __name__ == "__main__":
    run_cross_species()
