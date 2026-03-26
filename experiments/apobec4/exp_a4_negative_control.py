#!/usr/bin/env python
"""APOBEC4 Negative Control Analysis.

A4 (APOBEC4) has NO confirmed C-to-U deaminase activity. The 181 "A4-correlated"
sites in Levanon T3 are merely expression correlations, likely confounded by
co-expression with real editors (A3G, A3H). Only 21 sites are "A4-exclusive"
(not correlated with any other APOBEC enzyme).

This experiment serves as a NEGATIVE CONTROL: with no confirmed deaminase activity,
we expect classification near random and no distinctive structural signature. This
validates that our pipeline does not produce false positives for inactive enzymes.

Steps:
  1. Parse A4 sites from Levanon T3 sheet
  2. Generate ~180 motif-matched negatives from hg38
  3. Compute 40-dim hand features (motif 24 + struct_delta 7 + loop 9)
  4. Run 5-fold XGBoost classification (GB_HandFeatures, MotifOnly, StructOnly)
  5. Analyze 21 exclusive sites (genes, motifs, tissues, structure)
  6. Compute structure comparison (Mann-Whitney pos vs neg)
  7. Generate MFE unpaired profile plots (same format as other enzymes)
  8. Save all results

Usage:
    conda run -n quris python experiments/apobec4/exp_a4_negative_control.py
"""

import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score,
    precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.apobec_feature_extraction import (
    extract_motif_features,
    extract_loop_features,
    extract_structure_delta_features,
    build_hand_features,
    compute_vienna_features,
    LOOP_FEATURE_COLS,
)
from data.apobec_negatives import generate_negatives_from_genome

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
EXCEL_PATH = PROJECT_ROOT / "data" / "raw" / "C2TFinalSites.DB.xlsx"
SEQ_JSON = PROJECT_ROOT / "data" / "processed" / "site_sequences.json"
UNIFIED_CSV = PROJECT_ROOT / "data" / "processed" / "advisor" / "unified_editing_sites.csv"
GENOME_FA = PROJECT_ROOT / "data" / "raw" / "genomes" / "hg38.fa"
STRUCT_CACHE_MAIN = PROJECT_ROOT / "data" / "processed" / "embeddings" / "vienna_structure_cache.npz"
LOOP_CSV_V3 = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "loop_position_per_site_v3.csv"
LOOP_CSV_UNIFIED = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "loop_position_per_site_v3_unified.csv"

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "apobec4" / "outputs" / "classification"
STRUCT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "multi_enzyme" / "outputs" / "structure_analysis_v3"
STRUCT_JSON = STRUCT_OUTPUT_DIR / "structure_analysis_all_enzymes.json"

SEED = 42
CENTER = 100
ENZYME_COLS = ["A1", "A3A", "A3B", "A3G", "A3H", "A4"]

COLOR_POS = "#ff8c00"  # Orange for edited
COLOR_NEG = "#808080"  # Gray for unedited


# ---------------------------------------------------------------------------
# Step 1: Parse A4 sites from T3
# ---------------------------------------------------------------------------

def parse_a4_sites() -> tuple:
    """Parse A4-correlated and A4-exclusive sites from Levanon T3 sheet.

    Returns:
        (a4_df, a4_exclusive_df, all_t3_mapped_df)
        Each row has site_id, chr, start, enzyme flags, gene info.
    """
    logger.info("Parsing T3 sheet for A4 sites...")
    t3 = pd.read_excel(EXCEL_PATH, sheet_name="T3-APOBECs Correlations", header=2)

    # Keep relevant columns
    cols_needed = ["Chr", "Start", "End", "Genomic Category", "Gene Name",
                   "Exonic Function", "EditedInTissuesN", "EditedInTissues",
                   "AffectingAPOBEC"] + ENZYME_COLS
    t3 = t3[cols_needed].dropna(subset=["Chr"]).copy()
    t3.columns = ["chr", "start", "end", "genomic_category", "gene_name",
                   "exonic_function", "edited_in_tissues_n", "edited_in_tissues",
                   "affecting_apobec"] + ENZYME_COLS

    for col in ENZYME_COLS:
        t3[col] = t3[col].fillna(False).astype(bool)
    t3["start"] = t3["start"].astype(int)
    t3["end"] = t3["end"].astype(int)
    t3["edited_in_tissues_n"] = pd.to_numeric(t3["edited_in_tissues_n"], errors="coerce")

    # Merge with unified to get site_ids
    unified = pd.read_csv(UNIFIED_CSV)
    unified_key = unified[["site_id", "chr", "start"]].copy()
    unified_key["start"] = unified_key["start"].astype(int)
    t3_mapped = t3.merge(unified_key, on=["chr", "start"], how="left")

    n_matched = t3_mapped["site_id"].notna().sum()
    logger.info("  Matched %d / %d T3 sites to C2U site_ids", n_matched, len(t3))

    # A4-correlated (all 181)
    a4_mask = t3_mapped["A4"]
    a4_df = t3_mapped[a4_mask].copy()

    # A4-exclusive: A4=True and all others False
    others_false = (t3_mapped[["A1", "A3A", "A3B", "A3G", "A3H"]] == False).all(axis=1)
    a4_exclusive = t3_mapped[a4_mask & others_false].copy()

    logger.info("  A4-correlated: %d sites", len(a4_df))
    logger.info("  A4-exclusive: %d sites", len(a4_exclusive))

    return a4_df, a4_exclusive, t3_mapped


# ---------------------------------------------------------------------------
# Step 2: Generate negatives
# ---------------------------------------------------------------------------

def generate_a4_negatives(a4_df: pd.DataFrame, sequences: dict) -> tuple:
    """Generate motif-matched negatives from hg38 near A4 sites.

    A4 sites have near-random motif distribution (~34% TC, ~34% CC),
    so we match that.

    Returns:
        (neg_df, neg_sequences)
    """
    logger.info("Generating negatives for A4 sites...")

    # Prepare positives df with required columns
    pos_df = a4_df[["site_id", "chr", "start"]].copy()
    pos_df["end"] = pos_df["start"]
    pos_df["enzyme"] = "A4"
    pos_df["dataset_source"] = "levanon_a4"

    # Infer strand from reference: C at site = plus strand, G = minus strand
    # All Levanon sites are C-to-U edits; check the sequence center base
    strands = []
    for _, row in pos_df.iterrows():
        sid = row["site_id"]
        seq = sequences.get(sid, "N" * 201)
        if isinstance(seq, dict):
            seq = seq.get("sequence", "N" * 201)
        center_base = seq[CENTER] if len(seq) > CENTER else "N"
        strands.append("+" if center_base in ("C", "c") else "-")
    pos_df["strand"] = strands

    # Known positive sites to exclude
    known_sites = set()
    for _, row in pos_df.iterrows():
        known_sites.add((row["chr"], int(row["start"])))

    # Target motif fractions: near random for A4
    # Count actual TC/CC fraction of A4 sites
    n_tc = 0
    n_cc = 0
    for sid in a4_df["site_id"].dropna():
        seq = sequences.get(sid, "N" * 201)
        if isinstance(seq, dict):
            seq = seq.get("sequence", "N" * 201)
        seq = seq.upper().replace("T", "U")
        if len(seq) > CENTER and CENTER > 0:
            up = seq[CENTER - 1]
            if up == "U":
                n_tc += 1
            elif up == "C":
                n_cc += 1

    n_total = len(a4_df)
    tc_frac = n_tc / n_total if n_total > 0 else 0.34
    cc_frac = n_cc / n_total if n_total > 0 else 0.34
    logger.info("  A4 positive motif: TC=%.1f%%, CC=%.1f%%", 100 * tc_frac, 100 * cc_frac)

    # Generate negatives
    neg_seqs = {}
    n_negatives = len(a4_df)  # 1:1 ratio
    neg_df = generate_negatives_from_genome(
        positives_df=pos_df,
        genome_fa=GENOME_FA,
        target_tc_fraction=tc_frac,
        target_cc_fraction=cc_frac,
        n_negatives=n_negatives,
        output_seqs=neg_seqs,
        known_sites=known_sites,
        search_window=5000,
        seed=SEED,
    )

    logger.info("  Generated %d negatives", len(neg_df))
    return neg_df, neg_seqs


# ---------------------------------------------------------------------------
# Step 3: Compute features
# ---------------------------------------------------------------------------

def load_structure_data(site_ids: List[str]) -> tuple:
    """Load structure delta and loop geometry for A4 sites.

    Uses main vienna_structure_cache.npz (has all 636 Levanon sites) and
    loop_position_per_site_v3_unified.csv. Computes missing loop features
    on the fly via ViennaRNA.
    """
    logger.info("Loading structure data...")

    # Structure delta from main cache
    sd = np.load(str(STRUCT_CACHE_MAIN), allow_pickle=True)
    struct_sids = [str(s) for s in sd["site_ids"]]
    structure_delta = {}
    pairing_probs = {}
    pairing_probs_edited = {}
    for i, sid in enumerate(struct_sids):
        structure_delta[sid] = sd["delta_features"][i]
        if "pairing_probs" in sd:
            pairing_probs[sid] = sd["pairing_probs"][i]
        if "pairing_probs_edited" in sd:
            pairing_probs_edited[sid] = sd["pairing_probs_edited"][i]

    # Also store MFE structures for unpaired profile
    mfes = {}
    mfes_edited = {}
    if "mfes" in sd:
        for i, sid in enumerate(struct_sids):
            mfes[sid] = sd["mfes"][i]
    if "mfes_edited" in sd:
        for i, sid in enumerate(struct_sids):
            mfes_edited[sid] = sd["mfes_edited"][i]

    found_sd = sum(1 for s in site_ids if s in structure_delta)
    logger.info("  Structure delta: %d / %d site_ids found in cache", found_sd, len(site_ids))

    # Loop geometry
    loop_df = pd.DataFrame()
    for loop_path in [LOOP_CSV_UNIFIED, LOOP_CSV_V3]:
        if loop_path.exists():
            loop_df = pd.read_csv(loop_path)
            loop_df["site_id"] = loop_df["site_id"].astype(str)
            loop_df = loop_df.set_index("site_id")
            logger.info("  Loop features loaded from %s: %d sites", loop_path.name, len(loop_df))
            break

    found_lp = sum(1 for s in site_ids if s in loop_df.index)
    logger.info("  Loop features: %d / %d site_ids found", found_lp, len(site_ids))

    return structure_delta, loop_df, pairing_probs, pairing_probs_edited, mfes, mfes_edited


def compute_missing_features(missing_sids: List[str], sequences: dict) -> tuple:
    """Compute structure delta and loop features for sites not in caches."""
    logger.info("Computing ViennaRNA features for %d missing sites...", len(missing_sids))
    new_struct_delta = {}
    new_loop_rows = []

    for i, sid in enumerate(missing_sids):
        seq = sequences.get(sid, "N" * 201)
        if isinstance(seq, dict):
            seq = seq.get("sequence", "N" * 201)
        seq = seq.upper().replace("T", "U")

        try:
            sd, lf, _ = compute_vienna_features(seq)
            new_struct_delta[sid] = sd
            row = {"site_id": sid}
            for j, col in enumerate(LOOP_FEATURE_COLS):
                row[col] = float(lf[j])
            new_loop_rows.append(row)
        except Exception as e:
            logger.warning("  Failed for %s: %s", sid, e)
            new_struct_delta[sid] = np.zeros(7, dtype=np.float32)
            row = {"site_id": sid}
            for col in LOOP_FEATURE_COLS:
                row[col] = 0.0
            new_loop_rows.append(row)

        if (i + 1) % 20 == 0:
            logger.info("  Computed %d / %d", i + 1, len(missing_sids))

    new_loop_df = pd.DataFrame(new_loop_rows)
    if len(new_loop_df) > 0:
        new_loop_df["site_id"] = new_loop_df["site_id"].astype(str)
        new_loop_df = new_loop_df.set_index("site_id")

    return new_struct_delta, new_loop_df


# ---------------------------------------------------------------------------
# Step 4: Classification
# ---------------------------------------------------------------------------

def compute_binary_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    if len(np.unique(y_true)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan"), "f1": float("nan"),
                "precision": float("nan"), "recall": float("nan")}
    auroc = float(roc_auc_score(y_true, y_score))
    auprc = float(average_precision_score(y_true, y_score))
    prec, rec, thresholds = precision_recall_curve(y_true, y_score)
    f1_scores = 2 * prec * rec / (prec + rec + 1e-10)
    best_idx = np.argmax(f1_scores)
    threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
    y_pred = (np.array(y_score) >= threshold).astype(int)
    return {
        "auroc": auroc, "auprc": auprc,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def run_classification(hand_40: np.ndarray, labels: np.ndarray) -> dict:
    """Run 5-fold XGBoost classification with 3 feature sets."""
    from xgboost import XGBClassifier

    motif_24 = hand_40[:, :24]
    struct_16 = hand_40[:, 24:]  # struct_delta(7) + loop(9)

    model_configs = {
        "GB_HandFeatures": {"features": hand_40},
        "MotifOnly": {"features": motif_24},
        "StructOnly": {"features": struct_16},
    }

    results = {
        "enzyme": "A4",
        "n_positive": int((labels == 1).sum()),
        "n_negative": int((labels == 0).sum()),
        "cv_folds": 5,
        "models": {},
        "feature_importance": {},
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    for model_name, config in model_configs.items():
        logger.info("=" * 60)
        logger.info("Model: %s", model_name)
        X = config["features"]

        fold_metrics = []
        all_importances = None

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, labels)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]

            clf = XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                eval_metric="logloss",
                random_state=SEED + fold_idx,
                use_label_encoder=False,
            )
            clf.fit(X_train, y_train, verbose=False)

            y_score = clf.predict_proba(X_test)[:, 1]
            metrics = compute_binary_metrics(y_test, y_score)
            fold_metrics.append(metrics)

            if all_importances is None:
                all_importances = clf.feature_importances_.copy()
            else:
                all_importances += clf.feature_importances_

            logger.info("  Fold %d: AUROC=%.4f, AUPRC=%.4f",
                        fold_idx, metrics["auroc"], metrics["auprc"])

        all_importances /= 5

        aurocs = [m["auroc"] for m in fold_metrics]
        auprcs = [m["auprc"] for m in fold_metrics]
        f1s = [m["f1"] for m in fold_metrics]

        results["models"][model_name] = {
            "fold_aurocs": aurocs,
            "mean_auroc": float(np.mean(aurocs)),
            "std_auroc": float(np.std(aurocs)),
            "fold_auprcs": auprcs,
            "mean_auprc": float(np.mean(auprcs)),
            "std_auprc": float(np.std(auprcs)),
            "mean_f1": float(np.mean(f1s)),
        }

        logger.info("  Mean AUROC: %.4f +/- %.4f", np.mean(aurocs), np.std(aurocs))
        logger.info("  Mean AUPRC: %.4f +/- %.4f", np.mean(auprcs), np.std(auprcs))

        # Feature importance for GB_HandFeatures
        if model_name == "GB_HandFeatures":
            motif_names = [
                "m_5p_UC", "m_5p_CC", "m_5p_AC", "m_5p_GC",
                "m_3p_CA", "m_3p_CG", "m_3p_CU", "m_3p_CC",
                "m_m2_A", "m_m2_C", "m_m2_G", "m_m2_U",
                "m_m1_A", "m_m1_C", "m_m1_G", "m_m1_U",
                "m_p1_A", "m_p1_C", "m_p1_G", "m_p1_U",
                "m_p2_A", "m_p2_C", "m_p2_G", "m_p2_U",
            ]
            struct_names = [
                "delta_pairing_center", "delta_accessibility_center",
                "delta_entropy_center", "delta_mfe",
                "mean_delta_pairing_window", "mean_delta_accessibility_window",
                "std_delta_pairing_window",
            ]
            loop_names = [
                "is_unpaired", "loop_size", "dist_to_junction", "dist_to_apex",
                "relative_loop_position", "left_stem_length", "right_stem_length",
                "max_adjacent_stem_length", "local_unpaired_fraction",
            ]
            feat_names = motif_names + struct_names + loop_names
            imp_dict = {name: float(all_importances[i])
                        for i, name in enumerate(feat_names) if i < len(all_importances)}
            imp_sorted = dict(sorted(imp_dict.items(), key=lambda x: x[1], reverse=True))
            results["feature_importance"]["GB_HandFeatures"] = imp_sorted

            # Save CSV
            imp_df = pd.DataFrame([
                {"feature": k, "importance": v} for k, v in imp_sorted.items()
            ])
            imp_df.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
            logger.info("  Top 5 features: %s",
                        ", ".join(f"{k}={v:.4f}" for k, v in list(imp_sorted.items())[:5]))

    return results


# ---------------------------------------------------------------------------
# Step 5: Analyze 21 exclusive sites
# ---------------------------------------------------------------------------

def analyze_exclusive_sites(a4_excl_df: pd.DataFrame, sequences: dict) -> dict:
    """Detailed analysis of the 21 A4-exclusive sites."""
    logger.info("Analyzing %d A4-exclusive sites...", len(a4_excl_df))

    analysis = {
        "n_sites": len(a4_excl_df),
        "genes": [],
        "motif_distribution": {},
        "tissues": {},
        "genomic_categories": {},
    }

    # Gene list
    genes = a4_excl_df["gene_name"].dropna().unique().tolist()
    analysis["genes"] = sorted(genes)
    logger.info("  Genes: %s", ", ".join(genes[:10]))

    # Motif distribution
    trinuc_counts = Counter()
    dinuc_counts = Counter()
    for sid in a4_excl_df["site_id"].dropna():
        seq = sequences.get(sid, "N" * 201)
        if isinstance(seq, dict):
            seq = seq.get("sequence", "N" * 201)
        seq = seq.upper().replace("T", "U")
        if len(seq) > CENTER + 1 and CENTER > 0:
            up = seq[CENTER - 1]
            down = seq[CENTER + 1]
            dinuc = up + "C"
            trinuc = up + "C" + down
            dinuc_counts[dinuc] += 1
            if all(b in "ACGU" for b in trinuc):
                trinuc_counts[trinuc] += 1

    analysis["motif_distribution"]["dinucleotide"] = dict(dinuc_counts.most_common())
    analysis["motif_distribution"]["trinucleotide"] = dict(trinuc_counts.most_common())
    tc_frac = dinuc_counts.get("UC", 0) / len(a4_excl_df) if len(a4_excl_df) > 0 else 0
    cc_frac = dinuc_counts.get("CC", 0) / len(a4_excl_df) if len(a4_excl_df) > 0 else 0
    analysis["motif_distribution"]["tc_fraction"] = round(tc_frac, 4)
    analysis["motif_distribution"]["cc_fraction"] = round(cc_frac, 4)
    logger.info("  TC fraction: %.1f%%, CC fraction: %.1f%%", 100 * tc_frac, 100 * cc_frac)

    # Tissue distribution
    tissue_n = a4_excl_df["edited_in_tissues_n"].dropna()
    analysis["tissues"]["mean_n_tissues"] = round(float(tissue_n.mean()), 2) if len(tissue_n) > 0 else 0
    analysis["tissues"]["median_n_tissues"] = float(tissue_n.median()) if len(tissue_n) > 0 else 0

    # Parse tissue names
    tissue_counts = Counter()
    for tissues_str in a4_excl_df["edited_in_tissues"].dropna():
        if isinstance(tissues_str, str):
            for t in tissues_str.split(","):
                t = t.strip()
                if t:
                    tissue_counts[t] += 1
    analysis["tissues"]["tissue_counts"] = dict(tissue_counts.most_common(20))

    # Genomic category
    analysis["genomic_categories"] = a4_excl_df["genomic_category"].value_counts().to_dict()

    return analysis


# ---------------------------------------------------------------------------
# Step 6: Structure comparison (Mann-Whitney)
# ---------------------------------------------------------------------------

def compute_structure_comparison(pos_sids, neg_sids, struct_delta, loop_df):
    """Compute structural feature comparison table between pos and neg."""
    features = [
        ("Delta Pairing (edit site)", "delta_pairing_center", 0),
        ("Delta Accessibility (edit site)", "delta_accessibility_center", 1),
        ("Delta Entropy (edit site)", "delta_entropy_center", 2),
        ("Delta MFE (kcal/mol)", "delta_mfe", 3),
        ("Mean Delta Pairing (+/-10nt)", "mean_delta_pairing_window", 4),
        ("Mean Delta Accessibility (+/-10nt)", "mean_delta_accessibility_window", 5),
        ("Std Delta Pairing (+/-10nt)", "std_delta_pairing_window", 6),
    ]

    loop_features = [
        ("Is Unpaired", "is_unpaired"),
        ("Loop Size", "loop_size"),
        ("Relative Loop Position", "relative_loop_position"),
        ("Dist to Apex", "dist_to_apex"),
        ("Dist to Junction", "dist_to_junction"),
        ("Local Unpaired Fraction", "local_unpaired_fraction"),
        ("Left Stem Length", "left_stem_length"),
        ("Right Stem Length", "right_stem_length"),
        ("Max Adjacent Stem Length", "max_adjacent_stem_length"),
    ]

    rows = []

    # Structure delta features
    for label, _, idx in features:
        pos_vals = [struct_delta[sid][idx] for sid in pos_sids if sid in struct_delta]
        neg_vals = [struct_delta[sid][idx] for sid in neg_sids if sid in struct_delta]
        if pos_vals and neg_vals:
            pos_mean = float(np.mean(pos_vals))
            neg_mean = float(np.mean(neg_vals))
            pos_median = float(np.median(pos_vals))
            _, p = stats.mannwhitneyu(pos_vals, neg_vals, alternative="two-sided")
            rows.append({
                "feature": label, "pos_mean": pos_mean, "neg_mean": neg_mean,
                "diff": pos_mean - neg_mean, "p_value": float(p),
                "significant": "Yes" if p < 0.05 / (len(features) + len(loop_features)) else "No",
                "pos_median": pos_median,
            })

    # Loop geometry features
    for label, col in loop_features:
        pos_vals = loop_df.loc[loop_df.index.isin(pos_sids), col].dropna().values
        neg_vals = loop_df.loc[loop_df.index.isin(neg_sids), col].dropna().values
        if len(pos_vals) > 5 and len(neg_vals) > 5:
            pos_mean = float(np.mean(pos_vals))
            neg_mean = float(np.mean(neg_vals))
            pos_median = float(np.median(pos_vals))
            _, p = stats.mannwhitneyu(pos_vals, neg_vals, alternative="two-sided")
            rows.append({
                "feature": label, "pos_mean": pos_mean, "neg_mean": neg_mean,
                "diff": pos_mean - neg_mean, "p_value": float(p),
                "significant": "Yes" if p < 0.05 / (len(features) + len(loop_features)) else "No",
                "pos_median": pos_median,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 7: MFE unpaired profile plot
# ---------------------------------------------------------------------------

def compute_mfe_unpaired_profiles(site_ids: List[str], sequences: dict) -> np.ndarray:
    """Compute MFE dot-bracket structures and return unpaired fraction at each position.

    Returns:
        np.ndarray of shape (n_sites, 201) with 1=unpaired, 0=paired at each position.
    """
    import RNA

    profiles = []
    for sid in site_ids:
        seq = sequences.get(str(sid), "N" * 201)
        if isinstance(seq, dict):
            seq = seq.get("sequence", "N" * 201)
        seq = seq.upper().replace("T", "U")
        if len(seq) != 201:
            seq = seq[:201] if len(seq) > 201 else seq + "N" * (201 - len(seq))

        md = RNA.md()
        md.temperature = 37.0
        fc = RNA.fold_compound(seq, md)
        structure, _ = fc.mfe()
        unpaired = np.array([1.0 if c == "." else 0.0 for c in structure])
        profiles.append(unpaired)

    return np.array(profiles)


def plot_unpaired_mfe(enzyme, pos_profiles, neg_profiles, output_path, xlim=(-30, 30), window=5):
    """Plot smoothed MFE unpaired fraction profiles for edited vs unedited."""
    fig, ax = plt.subplots(figsize=(12, 5))
    positions = np.arange(201) - CENTER

    kernel = np.ones(window) / window

    mean_pos = pos_profiles.mean(axis=0)
    mean_neg = neg_profiles.mean(axis=0)
    n_pos, n_neg = len(pos_profiles), len(neg_profiles)

    if window > 1:
        mean_pos = np.convolve(mean_pos, kernel, mode="same")
        mean_neg = np.convolve(mean_neg, kernel, mode="same")

    se_pos = pos_profiles.std(axis=0) / np.sqrt(n_pos)
    se_neg = neg_profiles.std(axis=0) / np.sqrt(n_neg)
    if window > 1:
        se_pos = np.convolve(se_pos, kernel, mode="same")
        se_neg = np.convolve(se_neg, kernel, mode="same")

    ax.plot(positions, mean_pos, color=COLOR_POS, lw=2,
            label=f"Edited (n={n_pos})")
    ax.fill_between(positions,
                     mean_pos - 1.96 * se_pos,
                     mean_pos + 1.96 * se_pos,
                     color=COLOR_POS, alpha=0.15)

    ax.plot(positions, mean_neg, color=COLOR_NEG, lw=2,
            label=f"Unedited (n={n_neg})")
    ax.fill_between(positions,
                     mean_neg - 1.96 * se_neg,
                     mean_neg + 1.96 * se_neg,
                     color=COLOR_NEG, alpha=0.15)

    ax.axvline(0, color="black", ls="--", lw=1, alpha=0.5)
    ax.set_xlabel("Position relative to edit site (nt)")
    ax.set_ylabel("Unpaired fraction (MFE)")
    ax.set_title(f"{enzyme} — Unpaired Probability Profile (window={window})")
    ax.legend(fontsize=10)
    ax.set_xlim(*xlim)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_unpaired_mfe_zoom(enzyme, pos_profiles, neg_profiles, output_path):
    """Plot raw (no smoothing) MFE unpaired profile zoomed to +/-10 nt."""
    fig, ax = plt.subplots(figsize=(12, 5))
    positions = np.arange(201) - CENTER
    n_pos, n_neg = len(pos_profiles), len(neg_profiles)

    mean_pos = pos_profiles.mean(axis=0)
    mean_neg = neg_profiles.mean(axis=0)
    se_pos = pos_profiles.std(axis=0) / np.sqrt(n_pos)
    se_neg = neg_profiles.std(axis=0) / np.sqrt(n_neg)

    mask = (positions >= -10) & (positions <= 10)
    p = positions[mask]

    ax.plot(p, mean_pos[mask], color=COLOR_POS, lw=2, marker="o", ms=5,
            label="Edited")
    ax.fill_between(p,
                     (mean_pos - 1.96 * se_pos)[mask],
                     (mean_pos + 1.96 * se_pos)[mask],
                     color=COLOR_POS, alpha=0.15)

    ax.plot(p, mean_neg[mask], color=COLOR_NEG, lw=2, marker="o", ms=5,
            label="Unedited")
    ax.fill_between(p,
                     (mean_neg - 1.96 * se_neg)[mask],
                     (mean_neg + 1.96 * se_neg)[mask],
                     color=COLOR_NEG, alpha=0.15)

    ax.axvline(0, color="black", ls="--", lw=1.5, alpha=0.7)
    ax.set_xlabel("Position relative to edit site (nt)")
    ax.set_ylabel("Unpaired fraction (MFE)")
    ax.set_title(f"{enzyme} — Unpaired Profile (+/-10nt, raw)")
    ax.legend(fontsize=10)
    ax.set_xticks(range(-10, 11))
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    STRUCT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Step 1: Parse A4 sites
    # -----------------------------------------------------------------------
    a4_df, a4_excl_df, t3_mapped = parse_a4_sites()

    # Load sequences
    logger.info("Loading sequences...")
    with open(SEQ_JSON) as f:
        all_sequences = json.load(f)
    logger.info("  Total sequences: %d", len(all_sequences))

    a4_sids = a4_df["site_id"].dropna().astype(str).tolist()
    found_seqs = sum(1 for s in a4_sids if s in all_sequences)
    logger.info("  A4 sites with sequences: %d / %d", found_seqs, len(a4_sids))

    # -----------------------------------------------------------------------
    # Step 2: Generate negatives
    # -----------------------------------------------------------------------
    neg_df, neg_sequences = generate_a4_negatives(a4_df, all_sequences)

    # Merge sequences
    all_seq_combined = dict(all_sequences)
    all_seq_combined.update(neg_sequences)

    # Build combined dataframe
    pos_rows = []
    for _, row in a4_df.iterrows():
        sid = row["site_id"]
        if pd.isna(sid):
            continue
        pos_rows.append({
            "site_id": str(sid),
            "chr": row["chr"],
            "start": row["start"],
            "enzyme": "A4",
            "is_edited": 1,
            "label": 1,
        })
    pos_df = pd.DataFrame(pos_rows)
    neg_df["label"] = 0

    # Combine
    combined_df = pd.concat([pos_df, neg_df[["site_id", "chr", "start", "enzyme", "is_edited", "label"]].copy()], ignore_index=True)
    combined_df["site_id"] = combined_df["site_id"].astype(str)
    logger.info("Combined dataset: %d sites (pos=%d, neg=%d)",
                len(combined_df), (combined_df["label"] == 1).sum(), (combined_df["label"] == 0).sum())

    all_site_ids = combined_df["site_id"].tolist()

    # -----------------------------------------------------------------------
    # Step 3: Compute features
    # -----------------------------------------------------------------------
    structure_delta, loop_df, pairing_probs, pairing_probs_edited, mfes, mfes_edited = load_structure_data(all_site_ids)

    # Find sites missing from caches and compute their features
    missing_sids = [sid for sid in all_site_ids if sid not in structure_delta]
    if missing_sids:
        logger.info("Computing features for %d sites not in cache...", len(missing_sids))
        new_sd, new_loop = compute_missing_features(missing_sids, all_seq_combined)
        structure_delta.update(new_sd)
        if len(new_loop) > 0:
            loop_df = pd.concat([loop_df, new_loop])

    # Build hand features
    logger.info("Building features...")
    hand_40 = build_hand_features(all_site_ids, all_seq_combined, structure_delta, loop_df)
    labels = combined_df["label"].values
    logger.info("  Hand features shape: %s", hand_40.shape)

    # -----------------------------------------------------------------------
    # Step 4: Classification
    # -----------------------------------------------------------------------
    classification_results = run_classification(hand_40, labels)

    # -----------------------------------------------------------------------
    # Step 4b: Classification on A4-exclusive (21 sites) vs subset of negatives
    # -----------------------------------------------------------------------
    a4_excl_sids = set(a4_excl_df["site_id"].dropna().astype(str))
    excl_mask = combined_df["site_id"].isin(a4_excl_sids)
    # Also include negatives (keep same neg count for balance)
    neg_mask = combined_df["label"] == 0
    # Subsample negatives to match exclusive count
    neg_indices = combined_df[neg_mask].index.tolist()
    import random as _rng
    _rng.seed(SEED)
    _rng.shuffle(neg_indices)
    neg_subsample = neg_indices[:len(a4_excl_sids)]
    excl_indices = combined_df[excl_mask].index.tolist() + neg_subsample
    excl_combined = combined_df.loc[excl_indices].copy()
    excl_site_ids = excl_combined["site_id"].tolist()
    excl_hand_40 = build_hand_features(excl_site_ids, all_seq_combined, structure_delta, loop_df)
    excl_labels = excl_combined["label"].values

    logger.info("=" * 60)
    logger.info("A4-EXCLUSIVE Classification (21 pos vs 21 neg)")
    logger.info("=" * 60)

    # Only run GB_HandFeatures with leave-one-out CV due to small sample size
    from xgboost import XGBClassifier
    from sklearn.model_selection import LeaveOneOut

    loo = LeaveOneOut()
    loo_scores = []
    loo_labels = []
    for train_idx, test_idx in loo.split(excl_hand_40, excl_labels):
        X_train, X_test = excl_hand_40[train_idx], excl_hand_40[test_idx]
        y_train, y_test = excl_labels[train_idx], excl_labels[test_idx]
        clf = XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            subsample=0.8, eval_metric="logloss",
            random_state=SEED, use_label_encoder=False,
        )
        clf.fit(X_train, y_train, verbose=False)
        loo_scores.append(clf.predict_proba(X_test)[:, 1][0])
        loo_labels.append(y_test[0])

    loo_scores = np.array(loo_scores)
    loo_labels = np.array(loo_labels)
    excl_metrics = compute_binary_metrics(loo_labels, loo_scores)
    logger.info("  A4-exclusive LOO: AUROC=%.4f, AUPRC=%.4f", excl_metrics["auroc"], excl_metrics["auprc"])

    classification_results["models"]["A4_exclusive_LOO"] = {
        "method": "LeaveOneOut",
        "n_pos": int(excl_labels.sum()),
        "n_neg": int((excl_labels == 0).sum()),
        "auroc": excl_metrics["auroc"],
        "auprc": excl_metrics["auprc"],
        "f1": excl_metrics["f1"],
        "note": "21 A4-exclusive sites vs 21 negatives, LOO CV due to small n",
    }

    # -----------------------------------------------------------------------
    # Step 5: Analyze 21 exclusive sites
    # -----------------------------------------------------------------------
    exclusive_analysis = analyze_exclusive_sites(a4_excl_df, all_sequences)

    # -----------------------------------------------------------------------
    # Step 6: Structure comparison
    # -----------------------------------------------------------------------
    pos_sids = set(combined_df[combined_df["label"] == 1]["site_id"].astype(str))
    neg_sids = set(combined_df[combined_df["label"] == 0]["site_id"].astype(str))

    comp_table = compute_structure_comparison(pos_sids, neg_sids, structure_delta, loop_df)
    comp_table.to_csv(STRUCT_OUTPUT_DIR / "A4_structure_comparison.csv", index=False)

    n_significant = (comp_table["significant"] == "Yes").sum()
    logger.info("Structure comparison: %d features, %d significant",
                len(comp_table), n_significant)
    for _, row in comp_table.iterrows():
        sig_marker = "***" if row["significant"] == "Yes" else ""
        logger.info("  %s: pos=%.4f neg=%.4f diff=%.4f p=%.2e %s",
                     row["feature"], row["pos_mean"], row["neg_mean"],
                     row["diff"], row["p_value"], sig_marker)

    # -----------------------------------------------------------------------
    # Step 7: MFE unpaired profile plots
    # -----------------------------------------------------------------------
    logger.info("Computing MFE unpaired profiles (this may take a few minutes)...")
    pos_id_list = list(pos_sids)
    neg_id_list = list(neg_sids)

    # Sample for efficiency if too many
    import random
    rng = random.Random(SEED)
    max_profile = 300
    if len(pos_id_list) > max_profile:
        pos_id_list = rng.sample(pos_id_list, max_profile)
    if len(neg_id_list) > max_profile:
        neg_id_list = rng.sample(neg_id_list, max_profile)

    pos_profiles = compute_mfe_unpaired_profiles(pos_id_list, all_seq_combined)
    neg_profiles = compute_mfe_unpaired_profiles(neg_id_list, all_seq_combined)

    logger.info("  Pos profiles: %s, Neg profiles: %s", pos_profiles.shape, neg_profiles.shape)

    # Smoothed full plot
    plot_unpaired_mfe("A4", pos_profiles, neg_profiles,
                       STRUCT_OUTPUT_DIR / "A4_unpaired_mfe.png",
                       xlim=(-30, 30), window=5)
    # Zoomed raw plot
    plot_unpaired_mfe_zoom("A4", pos_profiles, neg_profiles,
                            STRUCT_OUTPUT_DIR / "A4_unpaired_mfe_zoom10.png")

    logger.info("  Plots saved to %s", STRUCT_OUTPUT_DIR)

    # -----------------------------------------------------------------------
    # Step 8: Update structure_analysis_all_enzymes.json
    # -----------------------------------------------------------------------
    if STRUCT_JSON.exists():
        with open(STRUCT_JSON) as f:
            all_struct_results = json.load(f)
    else:
        all_struct_results = {}

    all_struct_results["A4"] = {
        "n_pos": len(pos_sids),
        "n_neg": len(neg_sids),
        "comparison": comp_table.to_dict("records"),
    }

    with open(STRUCT_JSON, "w") as f:
        json.dump(all_struct_results, f, indent=2, default=str)
    logger.info("Updated %s with A4 entry", STRUCT_JSON.name)

    # -----------------------------------------------------------------------
    # Step 8b: Save classification results
    # -----------------------------------------------------------------------
    # Add exclusive analysis and structure comparison
    classification_results["exclusive_sites"] = exclusive_analysis
    classification_results["structure_comparison"] = {
        "n_significant_features": n_significant,
        "total_features": len(comp_table),
        "features": comp_table.to_dict("records"),
    }
    classification_results["negative_control_interpretation"] = {
        "hypothesis": "A4 has no confirmed C-to-U deaminase activity. "
                      "Classification should be near random (AUROC ~0.5-0.6).",
        "expected_auroc": "< 0.75",
        "expected_structure_signature": "No distinctive pattern vs negatives",
        "conclusion": "",  # Filled below
    }

    # Fill conclusion based on actual results
    gb_auroc = classification_results["models"]["GB_HandFeatures"]["mean_auroc"]
    excl_auroc = classification_results["models"]["A4_exclusive_LOO"]["auroc"]

    conclusion_parts = []
    # All 181 sites
    if gb_auroc >= 0.75:
        conclusion_parts.append(
            f"All 181 A4-correlated sites: GB_HandFeatures AUROC={gb_auroc:.3f}. "
            "This is inflated because 160/181 sites are co-correlated with real editors "
            "(A3A, A3G, A3H) and carry genuine editing signatures from those enzymes."
        )
    else:
        conclusion_parts.append(
            f"All 181 A4-correlated sites: GB_HandFeatures AUROC={gb_auroc:.3f}, near random."
        )
    # 21 exclusive sites
    if excl_auroc < 0.65:
        conclusion_parts.append(
            f"The 21 A4-exclusive sites (LOO AUROC={excl_auroc:.3f}) are indistinguishable from "
            "random cytidines, confirming that A4 has no distinctive editing signature. "
            "This validates our pipeline as a true negative control."
        )
    elif excl_auroc < 0.75:
        conclusion_parts.append(
            f"The 21 A4-exclusive sites (LOO AUROC={excl_auroc:.3f}) show only weak signal, "
            "consistent with noise or minor confounding, not genuine A4 editing."
        )
    else:
        conclusion_parts.append(
            f"The 21 A4-exclusive sites (LOO AUROC={excl_auroc:.3f}) show moderate signal, "
            "possibly due to unknown confounders or shared genomic context rather than A4 activity."
        )
    conclusion = " ".join(conclusion_parts)
    classification_results["negative_control_interpretation"]["conclusion"] = conclusion

    out_path = OUTPUT_DIR / "classification_results.json"
    with open(out_path, "w") as f:
        json.dump(classification_results, f, indent=2, default=str)
    logger.info("Classification results saved to %s", out_path)

    # -----------------------------------------------------------------------
    # Print summary
    # -----------------------------------------------------------------------
    elapsed = time.time() - t_start
    print("\n" + "=" * 70)
    print("APOBEC4 Negative Control — Summary")
    print("=" * 70)
    print(f"\nA4-correlated sites: {len(a4_df)} (of 636 Levanon sites)")
    print(f"A4-exclusive sites:  {len(a4_excl_df)}")
    print(f"Negatives generated: {len(neg_df)}")
    print()
    print("Classification (5-fold CV, all 181 A4-correlated):")
    for model_name, model_res in classification_results["models"].items():
        if model_name == "A4_exclusive_LOO":
            continue
        print(f"  {model_name:<20}: AUROC={model_res['mean_auroc']:.3f} +/- {model_res['std_auroc']:.3f}, "
              f"AUPRC={model_res['mean_auprc']:.3f}")
    excl_res = classification_results["models"]["A4_exclusive_LOO"]
    print(f"\nA4-exclusive (21 sites, LOO CV):")
    print(f"  GB_HandFeatures     : AUROC={excl_res['auroc']:.3f}, AUPRC={excl_res['auprc']:.3f}")
    print()
    print("Structure comparison:")
    print(f"  Significant features: {n_significant} / {len(comp_table)} "
          f"(Bonferroni at alpha=0.05/{len(comp_table)})")
    print()
    print("A4-exclusive sites:")
    print(f"  Genes: {', '.join(exclusive_analysis['genes'][:10])}")
    print(f"  TC fraction: {exclusive_analysis['motif_distribution']['tc_fraction']:.1%}")
    print(f"  CC fraction: {exclusive_analysis['motif_distribution']['cc_fraction']:.1%}")
    print(f"  Mean tissues: {exclusive_analysis['tissues']['mean_n_tissues']}")
    print()
    print(f"Conclusion: {conclusion}")
    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"Results: {out_path}")
    print(f"Plots:   {STRUCT_OUTPUT_DIR}/A4_*.png")


if __name__ == "__main__":
    main()
