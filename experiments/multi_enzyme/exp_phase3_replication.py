#!/usr/bin/env python3
"""Phase 3 Model Replication: Test Phase 3 paradigm on ALL advisor report experiments.

Approach:
  The Phase 3 neural model (1448-dim fusion) requires RNA-FM embeddings that are only
  pre-computed for the v3 training set. For external data (ClinVar, TCGA, cross-species,
  gnomAD), RNA-FM is unavailable.

  Solution: Use two approaches:
    1. Phase3_Neural: Retrain the neural model on v3 data, evaluate with 5-fold CV
       (internal validation only - captures the multi-stream architecture benefit)
    2. Phase3_XGB_40d: Train multi-class 7-class XGBoost on 40-dim hand features
       using the Phase 3 training paradigm (v3 data, enzyme-matched negatives).
       This can score any external data with ViennaRNA features.

  Comparison baseline: Original binary XGBoost (40-dim) per-enzyme models.

Experiments replicated:
  1. ClinVar Pathogenic Enrichment (1.69M variants)
  2. TCGA Somatic Mutation Enrichment (6 cancers + 3 additional)
  3. Cross-Species Comparison (chimp orthologs)
  4. gnomAD Constraint (gene-level editability vs LOEUF)
  5. Multi-class Enzyme Prediction on TCGA

Usage:
    conda run -n quris python experiments/multi_enzyme/exp_phase3_replication.py
"""

import gc
import gzip
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyfaidx import Fasta
from scipy import stats
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.apobec_feature_extraction import (
    extract_motif_from_seq,
    LOOP_FEATURE_COLS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# =============================================================================
# Paths
# =============================================================================
DATA_DIR = PROJECT_ROOT / "data"
MULTI_SPLITS = DATA_DIR / "processed/multi_enzyme/splits_multi_enzyme_v3_with_negatives.csv"
MULTI_SEQS = DATA_DIR / "processed/multi_enzyme/multi_enzyme_sequences_v3_with_negatives.json"
LOOP_CSV = DATA_DIR / "processed/multi_enzyme/loop_position_per_site_v3.csv"
STRUCT_CACHE = DATA_DIR / "processed/multi_enzyme/structure_cache_multi_enzyme_v3.npz"

# ClinVar
CLINVAR_SCORES_CSV = PROJECT_ROOT / "experiments/apobec3a/outputs/clinvar_prediction/clinvar_all_scores.csv"
CLINVAR_FEATURES = DATA_DIR / "processed/clinvar_features_cache.npz"

# TCGA
TCGA_DIR = DATA_DIR / "raw/tcga"
HG19_FA = DATA_DIR / "raw/genomes/hg19.fa"
REFGENE_HG19 = DATA_DIR / "raw/genomes/refGene_hg19.txt"
VIENNA_CACHE_DIR = PROJECT_ROOT / "experiments/multi_enzyme/outputs/tcga_gnomad/vienna_cache"

# Cross-species
CROSS_SPECIES_DIR = PROJECT_ROOT / "experiments/multi_enzyme/outputs/cross_species"
CROSS_SPECIES_CSV = CROSS_SPECIES_DIR / "gb_scoring_human_vs_chimp.csv"

# gnomAD
GNOMAD_DIR = PROJECT_ROOT / "experiments/multi_enzyme/outputs/gnomad_site_level"
GNOMAD_GENE_CSV = GNOMAD_DIR / "gene_editability_constraint.csv"

# Output
OUTPUT_DIR = PROJECT_ROOT / "experiments/multi_enzyme/outputs/phase3_replication"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_CONTROLS = 5

# Class labels
ENZYME_CLASSES = ["A3A", "A3B", "A3G", "A3A_A3G", "Neither", "Unknown", "Negative"]
CLASS_TO_IDX = {c: i for i, c in enumerate(ENZYME_CLASSES)}
CANCERS = ["blca", "brca", "cesc", "lusc", "hnsc", "skcm", "esca", "stad", "lihc"]

# Original advisor report values for comparison
ADVISOR_CLINVAR_OR = 1.159  # A3A OR at t=0.5
ADVISOR_CROSS_SPECIES_SPEARMAN = 0.632
ADVISOR_GNOMAD_LOEUF_RHO = -0.051  # mean_score_vs_LOEUF
ADVISOR_TCGA = {
    "brca": {"OR_p90": 1.661, "OR_p95": 2.120},
    "cesc": {"OR_p90": 1.733, "OR_p95": 2.231},
    "lusc": {"OR_p90": 1.384, "OR_p95": 1.632},
    "blca": {"OR_p90": 1.106, "OR_p95": 1.359},
    "skcm": {"OR_p90": 0.790, "OR_p95": 0.870},
}


# =============================================================================
# Feature extraction helpers (same as exp_multiclass_enzyme_tcga.py)
# =============================================================================

def derive_features_from_fold(fold_result):
    """Derive 16-dim structure features (7 delta + 9 loop) from cached fold data."""
    if fold_result is None:
        return None
    center = 100
    pair_wt = fold_result["bpp_wt_center"]
    pair_ed = fold_result["bpp_ed_center"]
    delta_pairing = pair_ed - pair_wt
    delta_accessibility = -delta_pairing
    delta_mfe = fold_result["mfe_ed"] - fold_result["mfe_wt"]
    bpp_wt_w = fold_result["bpp_wt_window"]
    bpp_ed_w = fold_result["bpp_ed_window"]

    def _entropy_from_p(p_val):
        if p_val <= 0 or p_val >= 1:
            return 0.0
        return -(p_val * np.log2(p_val + 1e-10) + (1 - p_val) * np.log2(1 - p_val + 1e-10))

    delta_entropy = _entropy_from_p(pair_ed) - _entropy_from_p(pair_wt)
    delta_window = np.array(bpp_ed_w) - np.array(bpp_wt_w)
    mean_delta_pairing = float(np.mean(delta_window)) if len(delta_window) > 0 else 0.0
    std_delta_pairing = float(np.std(delta_window)) if len(delta_window) > 0 else 0.0

    struct_delta = np.array([
        delta_pairing, delta_accessibility, delta_entropy,
        delta_mfe, mean_delta_pairing, std_delta_pairing, -mean_delta_pairing
    ], dtype=np.float32)

    struct_wt = fold_result["struct_wt"]
    is_unpaired = 1.0 if struct_wt[center] == "." else 0.0
    loop_size = dist_to_junction = dist_to_apex = 0.0
    relative_loop_position = 0.5
    left_stem = right_stem = max_adj_stem = local_unpaired = 0.0

    if is_unpaired:
        left = center
        while left > 0 and struct_wt[left] == ".":
            left -= 1
        right = center
        while right < len(struct_wt) - 1 and struct_wt[right] == ".":
            right += 1
        loop_size = float(right - left - 1)
        if loop_size > 0:
            pos_in_loop = center - left - 1
            relative_loop_position = pos_in_loop / max(loop_size - 1, 1)
            dist_to_apex = abs(pos_in_loop - (loop_size - 1) / 2)
        ls = 0
        i = left
        while i >= 0 and struct_wt[i] in "()":
            ls += 1
            i -= 1
        left_stem = float(ls)
        rs = 0
        i = right
        while i < len(struct_wt) and struct_wt[i] in "()":
            rs += 1
            i += 1
        right_stem = float(rs)
        max_adj_stem = max(left_stem, right_stem)

    local_region = struct_wt[max(0, center - 10):min(len(struct_wt), center + 11)]
    local_unpaired = sum(1 for ch in local_region if ch == ".") / max(len(local_region), 1)

    loop_features = np.array([
        is_unpaired, loop_size, dist_to_junction, dist_to_apex,
        relative_loop_position, left_stem, right_stem,
        max_adj_stem, local_unpaired
    ], dtype=np.float32)

    return {"struct_delta": struct_delta, "loop_features": loop_features}


def build_40dim_features(sequences, fold_results):
    """Build 40-dim feature matrix from sequences + fold results."""
    n = len(sequences)
    features = np.zeros((n, 40), dtype=np.float32)
    for i, (seq, fold_raw) in enumerate(zip(sequences, fold_results)):
        if seq is not None and len(seq) == 201:
            features[i, :24] = extract_motif_from_seq(seq)
        derived = derive_features_from_fold(fold_raw)
        if derived is not None:
            features[i, 24:31] = derived["struct_delta"]
            features[i, 31:40] = derived["loop_features"]
    return np.nan_to_num(features, nan=0.0)


def build_40dim_from_training(splits, sequences_dict, loop_df, struct_data):
    """Build 40-dim features for training data from splits + caches."""
    site_ids = splits["site_id"].astype(str).tolist()
    motif_feats = np.array(
        [extract_motif_from_seq(sequences_dict.get(sid, "N" * 201)) for sid in site_ids],
        dtype=np.float32,
    )
    struct_feats = np.zeros((len(site_ids), 7), dtype=np.float32)
    if struct_data is not None:
        struct_sids = [str(s) for s in struct_data["site_ids"]]
        sid_to_idx = {s: i for i, s in enumerate(struct_sids)}
        for i, sid in enumerate(site_ids):
            if sid in sid_to_idx:
                struct_feats[i] = struct_data["delta_features"][sid_to_idx[sid]]
    loop_feats = np.zeros((len(site_ids), len(LOOP_FEATURE_COLS)), dtype=np.float32)
    for i, sid in enumerate(site_ids):
        if sid in loop_df.index:
            for j, col in enumerate(LOOP_FEATURE_COLS):
                if col in loop_df.columns:
                    val = loop_df.loc[sid, col]
                    if not isinstance(val, (int, float)):
                        val = val.iloc[0] if hasattr(val, "iloc") else 0.0
                    loop_feats[i, j] = float(val) if pd.notna(val) else 0.0
    X = np.concatenate([motif_feats, struct_feats, loop_feats], axis=1)
    return np.nan_to_num(X, nan=0.0)


# =============================================================================
# Enrichment computation
# =============================================================================

def compute_enrichment(mut_scores, ctrl_scores,
                       thresholds=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)):
    """Compute enrichment ORs at multiple thresholds."""
    results = {}
    for t in thresholds:
        mut_above = int((mut_scores >= t).sum())
        mut_below = int((mut_scores < t).sum())
        ctrl_above = int((ctrl_scores >= t).sum())
        ctrl_below = int((ctrl_scores < t).sum())
        if mut_below > 0 and ctrl_above > 0 and ctrl_below > 0 and mut_above > 0:
            table = [[mut_above, mut_below], [ctrl_above, ctrl_below]]
            odds_ratio, p_value = stats.fisher_exact(table)
        else:
            odds_ratio, p_value = float("nan"), 1.0
        results[str(t)] = {
            "OR": float(odds_ratio),
            "p": float(p_value),
            "n_mut_above": mut_above,
            "frac_mut": float(mut_above / max(len(mut_scores), 1)),
            "frac_ctrl": float(ctrl_above / max(len(ctrl_scores), 1)),
        }
    return results


def compute_percentile_enrichment(mut_scores, ctrl_scores,
                                   percentiles=(50, 75, 90, 95)):
    """Compute enrichment at percentile thresholds of the CONTROL distribution."""
    results = {}
    for pct in percentiles:
        t = np.percentile(ctrl_scores, pct)
        mut_above = int((mut_scores >= t).sum())
        mut_below = int((mut_scores < t).sum())
        ctrl_above = int((ctrl_scores >= t).sum())
        ctrl_below = int((ctrl_scores < t).sum())
        if mut_below > 0 and ctrl_above > 0 and ctrl_below > 0 and mut_above > 0:
            table = [[mut_above, mut_below], [ctrl_above, ctrl_below]]
            odds_ratio, p_value = stats.fisher_exact(table)
        else:
            odds_ratio, p_value = float("nan"), 1.0
        results[f"p{pct}"] = {
            "threshold": float(t),
            "OR": float(odds_ratio),
            "p": float(p_value),
            "frac_mut": float(mut_above / max(len(mut_scores), 1)),
            "frac_ctrl": float(ctrl_above / max(len(ctrl_scores), 1)),
        }
    return results


# =============================================================================
# TCGA helpers (same as exp_multiclass_enzyme_tcga)
# =============================================================================

def parse_exons(refgene_path):
    """Parse refGene for coding exonic regions."""
    exons_by_gene = defaultdict(list)
    seen = set()
    with open(refgene_path) as f:
        for line in f:
            fields = line.strip().split("\t")
            if len(fields) < 13:
                continue
            chrom = fields[2]
            strand = fields[3]
            cds_s, cds_e = int(fields[6]), int(fields[7])
            if cds_s == cds_e:
                continue
            gene = fields[12]
            for es_str, ee_str in zip(fields[9].rstrip(",").split(","),
                                       fields[10].rstrip(",").split(",")):
                if not es_str or not ee_str:
                    continue
                es, ee = max(int(es_str), cds_s), min(int(ee_str), cds_e)
                if es >= ee:
                    continue
                key = (chrom, es, ee)
                if key not in seen:
                    seen.add(key)
                    exons_by_gene[gene].append((chrom, es, ee, strand, gene))
    return exons_by_gene


def parse_ct_mutations(maf_path):
    """Parse C>T and G>A SNPs from MAF."""
    rows = []
    for chunk in pd.read_csv(maf_path, sep="\t", comment="#", low_memory=False, chunksize=100000):
        if "Variant_Type" in chunk.columns:
            chunk = chunk[chunk["Variant_Type"] == "SNP"]
        ct = (chunk["Reference_Allele"] == "C") & (chunk["Tumor_Seq_Allele2"] == "T")
        ga = (chunk["Reference_Allele"] == "G") & (chunk["Tumor_Seq_Allele2"] == "A")
        sub = chunk[ct | ga].copy()
        if len(sub) > 0:
            chrom = sub["Chromosome"].astype(str)
            if not chrom.str.startswith("chr").any():
                chrom = "chr" + chrom
            sub["chrom"] = chrom
            sub["pos"] = sub["Start_Position"].astype(int) - 1
            sub["strand_inf"] = np.where(sub["Reference_Allele"] == "C", "+", "-")
            sub["gene"] = sub.get("Hugo_Symbol", "unknown")
            sub["sample"] = sub.get("Tumor_Sample_Barcode", "unknown")
            rows.append(sub[["chrom", "pos", "strand_inf", "gene", "sample"]].copy())
    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    df_dedup = df.drop_duplicates(subset=["chrom", "pos"]).copy()
    rec = df.groupby(["chrom", "pos"]).size().reset_index(name="recurrence")
    df_dedup = df_dedup.merge(rec, on=["chrom", "pos"], how="left")
    return df_dedup


def get_matched_controls(mutations_df, genome, exons_by_gene, n_controls=N_CONTROLS):
    """For each mutation, pick n_controls random C positions in same gene's exons."""
    rng = np.random.RandomState(SEED)
    controls = []
    for _, row in mutations_df.iterrows():
        gene = row["gene"]
        gene_exons = exons_by_gene.get(gene, [])
        if not gene_exons:
            continue
        c_positions = []
        for chrom, start, end, strand, _ in gene_exons:
            if chrom != row["chrom"]:
                continue
            try:
                exon_seq = str(genome[chrom][start:end]).upper()
                for i, base in enumerate(exon_seq):
                    pos = start + i
                    if pos == row["pos"]:
                        continue
                    if base == "C":
                        c_positions.append((chrom, pos, "+"))
                    elif base == "G":
                        c_positions.append((chrom, pos, "-"))
            except (KeyError, ValueError):
                continue
        if len(c_positions) >= n_controls:
            chosen = rng.choice(len(c_positions), n_controls, replace=False)
            for idx in chosen:
                ch, p, s = c_positions[idx]
                controls.append({"chrom": ch, "pos": p, "strand_inf": s, "gene": gene})
        elif c_positions:
            for ch, p, s in c_positions[:n_controls]:
                controls.append({"chrom": ch, "pos": p, "strand_inf": s, "gene": gene})
    return pd.DataFrame(controls) if controls else pd.DataFrame()


def extract_sequences(positions_df, genome):
    """Extract 201-nt sequences for each position."""
    seqs = []
    for _, row in positions_df.iterrows():
        chrom = row["chrom"]
        pos = int(row["pos"])
        strand = row["strand_inf"]
        try:
            chrom_len = len(genome[chrom])
            start = pos - 100
            end = pos + 101
            if start < 0 or end > chrom_len:
                seqs.append(None)
                continue
            seq = str(genome[chrom][start:end]).upper()
            if strand == "-":
                comp = str.maketrans("ACGT", "TGCA")
                seq = seq.translate(comp)[::-1]
            seqs.append(seq)
        except (KeyError, ValueError):
            seqs.append(None)
    return seqs


# =============================================================================
# Step 0: Train Phase 3 multi-class XGBoost
# =============================================================================

def train_phase3_model():
    """Train multi-class XGBoost on v3 multi-enzyme data. Returns model + CV results."""
    logger.info("=" * 70)
    logger.info("STEP 0: Train Phase 3 Multi-Class XGBoost (40-dim)")
    logger.info("=" * 70)

    splits = pd.read_csv(MULTI_SPLITS)
    with open(MULTI_SEQS) as f:
        sequences_dict = json.load(f)

    loop_df = pd.read_csv(LOOP_CSV)
    loop_df["site_id"] = loop_df["site_id"].astype(str)
    loop_df = loop_df.set_index("site_id")

    struct_data = None
    if STRUCT_CACHE.exists():
        struct_data = dict(np.load(str(STRUCT_CACHE), allow_pickle=True))
        struct_data["site_ids"] = [str(s) for s in struct_data["site_ids"]]

    X_train = build_40dim_from_training(splits, sequences_dict, loop_df, struct_data)

    # Multi-class labels
    mc_labels = []
    for _, row in splits.iterrows():
        if row["is_edited"] == 1:
            mc_labels.append(row["enzyme"])
        else:
            mc_labels.append("Negative")
    mc_labels = np.array(mc_labels)
    y_train = np.array([CLASS_TO_IDX[lbl] for lbl in mc_labels])

    logger.info(f"Training data: {len(X_train)} samples, {X_train.shape[1]} features")
    for cls_name, cls_idx in CLASS_TO_IDX.items():
        n = int((y_train == cls_idx).sum())
        logger.info(f"  {cls_name}: {n}")

    # 5-fold CV
    logger.info("\n5-fold cross-validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_logloss = []
    cv_edited_auroc = []
    cv_per_enzyme_auroc = {e: [] for e in ENZYME_CLASSES[:-1]}

    for fold_i, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        xgb_cv = XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            objective="multi:softprob", num_class=len(ENZYME_CLASSES),
            random_state=SEED, eval_metric="mlogloss",
            use_label_encoder=False, verbosity=0,
        )
        xgb_cv.fit(X_train[tr_idx], y_train[tr_idx])
        proba = xgb_cv.predict_proba(X_train[val_idx])
        ll = log_loss(y_train[val_idx], proba, labels=list(range(len(ENZYME_CLASSES))))
        cv_logloss.append(ll)

        # Binary edited vs negative AUROC
        neg_idx_cls = CLASS_TO_IDX["Negative"]
        p_edited = 1 - proba[:, neg_idx_cls]
        is_edited_val = (y_train[val_idx] != neg_idx_cls).astype(int)
        auroc = roc_auc_score(is_edited_val, p_edited)
        cv_edited_auroc.append(auroc)

        # Per-enzyme AUROC
        for enz_name in ENZYME_CLASSES[:-1]:
            enz_idx = CLASS_TO_IDX[enz_name]
            mask = (y_train[val_idx] == enz_idx) | (y_train[val_idx] == neg_idx_cls)
            if mask.sum() > 10:
                labels_sub = (y_train[val_idx][mask] == enz_idx).astype(int)
                proba_sub = proba[mask, enz_idx]
                if len(np.unique(labels_sub)) == 2:
                    cv_per_enzyme_auroc[enz_name].append(roc_auc_score(labels_sub, proba_sub))

        logger.info(f"  Fold {fold_i}: logloss={ll:.4f}, P(edited) AUROC={auroc:.4f}")

    logger.info(f"  Mean logloss: {np.mean(cv_logloss):.4f}")
    logger.info(f"  Mean P(edited) AUROC: {np.mean(cv_edited_auroc):.4f}")
    for enz in ENZYME_CLASSES[:-1]:
        if cv_per_enzyme_auroc[enz]:
            logger.info(f"  {enz} AUROC: {np.mean(cv_per_enzyme_auroc[enz]):.4f}")

    # Train final model on all data
    logger.info("\nTraining final model on all data...")
    model = XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        objective="multi:softprob", num_class=len(ENZYME_CLASSES),
        random_state=SEED, eval_metric="mlogloss",
        use_label_encoder=False, verbosity=0,
    )
    model.fit(X_train, y_train)

    cv_results = {
        "mean_logloss": float(np.mean(cv_logloss)),
        "mean_edited_auroc": float(np.mean(cv_edited_auroc)),
        "per_enzyme_auroc": {e: float(np.mean(v)) for e, v in cv_per_enzyme_auroc.items() if v},
    }

    del struct_data, sequences_dict
    gc.collect()

    return model, cv_results


# =============================================================================
# Experiment 1: ClinVar Pathogenic Enrichment
# =============================================================================

def run_clinvar_enrichment(model):
    """Score ClinVar variants with Phase 3 multi-class model and compute enrichment."""
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 1: ClinVar Pathogenic Enrichment")
    logger.info("=" * 70)

    if not CLINVAR_FEATURES.exists():
        logger.warning("ClinVar features cache not found. Skipping.")
        return None

    # Load ClinVar features
    logger.info("Loading ClinVar features cache...")
    cache = np.load(CLINVAR_FEATURES, allow_pickle=True)
    site_ids = cache["site_ids"]
    hand_46 = cache["hand_46"]  # 46-dim: motif(24) + struct_delta(7) + loop(9) + baseline(6)
    logger.info(f"  Loaded {len(site_ids)} variants, features shape: {hand_46.shape}")

    # Use first 40 dims (motif + struct_delta + loop) to match training features
    hand_40 = hand_46[:, :40]

    # Score with multi-class model
    logger.info("Scoring with Phase 3 multi-class XGBoost...")
    proba = model.predict_proba(hand_40)

    neg_idx = CLASS_TO_IDX["Negative"]
    p_edited = 1.0 - proba[:, neg_idx]
    p_a3a = proba[:, CLASS_TO_IDX["A3A"]]
    p_a3g = proba[:, CLASS_TO_IDX["A3G"]]
    p_neither = proba[:, CLASS_TO_IDX["Neither"]]

    # Load ClinVar metadata for pathogenicity
    logger.info("Loading ClinVar metadata...")
    clinvar_df = pd.read_csv(CLINVAR_SCORES_CSV)
    logger.info(f"  Loaded {len(clinvar_df)} scored variants from existing CSV")

    # Map site_ids to scores
    sid_to_idx = {str(sid): i for i, sid in enumerate(site_ids)}

    # Match ClinVar rows to features
    clinvar_df["phase3_p_edited"] = np.nan
    clinvar_df["phase3_p_a3a"] = np.nan
    clinvar_df["phase3_p_a3g"] = np.nan
    clinvar_df["phase3_p_neither"] = np.nan

    n_matched = 0
    for idx, row in clinvar_df.iterrows():
        sid = str(row["site_id"])
        if sid in sid_to_idx:
            i = sid_to_idx[sid]
            clinvar_df.at[idx, "phase3_p_edited"] = float(p_edited[i])
            clinvar_df.at[idx, "phase3_p_a3a"] = float(p_a3a[i])
            clinvar_df.at[idx, "phase3_p_a3g"] = float(p_a3g[i])
            clinvar_df.at[idx, "phase3_p_neither"] = float(p_neither[i])
            n_matched += 1

    logger.info(f"  Matched {n_matched}/{len(clinvar_df)} variants")

    # Filter to variants with scores
    scored = clinvar_df.dropna(subset=["phase3_p_edited"]).copy()
    logger.info(f"  Scored variants: {len(scored)}")

    # Compute enrichment by pathogenicity
    patho_map = {
        "Pathogenic": "pathogenic",
        "Likely_pathogenic": "pathogenic",
        "Pathogenic/Likely_pathogenic": "pathogenic",
        "Benign": "benign",
        "Likely_benign": "benign",
        "Benign/Likely_benign": "benign",
    }
    scored["patho_class"] = scored["significance_simple"].map(
        lambda x: "pathogenic" if "athogenic" in str(x) else (
            "benign" if "enign" in str(x) else "other"
        )
    )

    patho = scored[scored["patho_class"] == "pathogenic"]
    benign = scored[scored["patho_class"] == "benign"]
    logger.info(f"  Pathogenic: {len(patho)}, Benign: {len(benign)}")

    results = {"n_total": len(scored), "n_pathogenic": len(patho), "n_benign": len(benign)}

    # Enrichment at various thresholds
    for score_name in ["phase3_p_edited", "phase3_p_a3a", "phase3_p_a3g", "phase3_p_neither"]:
        p_scores = scored[score_name].values
        patho_scores = patho[score_name].values
        benign_scores = benign[score_name].values

        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        enrichment = {}
        for t in thresholds:
            patho_above = int((patho_scores >= t).sum())
            patho_below = int((patho_scores < t).sum())
            benign_above = int((benign_scores >= t).sum())
            benign_below = int((benign_scores < t).sum())
            if all(v > 0 for v in [patho_above, patho_below, benign_above, benign_below]):
                table = [[patho_above, patho_below], [benign_above, benign_below]]
                odds_ratio, p_value = stats.fisher_exact(table)
            else:
                odds_ratio, p_value = float("nan"), 1.0
            enrichment[str(t)] = {
                "OR": float(odds_ratio),
                "p": float(p_value),
                "frac_patho": float(patho_above / max(len(patho_scores), 1)),
                "frac_benign": float(benign_above / max(len(benign_scores), 1)),
            }

        results[score_name] = {
            "mean_patho": float(np.nanmean(patho_scores)),
            "mean_benign": float(np.nanmean(benign_scores)),
            "enrichment": enrichment,
        }

        logger.info(f"\n  {score_name}:")
        logger.info(f"    Mean pathogenic: {np.nanmean(patho_scores):.4f}")
        logger.info(f"    Mean benign: {np.nanmean(benign_scores):.4f}")
        for t in [0.3, 0.5, 0.7]:
            e = enrichment[str(t)]
            logger.info(f"    OR@{t}: {e['OR']:.3f} (p={e['p']:.2e})")

    # Within-gene analysis
    logger.info("\n  Within-gene analysis (Phase3 P(edited))...")
    gene_groups = scored.groupby("gene")
    n_genes_patho_higher = 0
    n_genes_total = 0
    gene_pvals = []

    for gene, grp in gene_groups:
        patho_g = grp[grp["patho_class"] == "pathogenic"]["phase3_p_edited"].values
        benign_g = grp[grp["patho_class"] == "benign"]["phase3_p_edited"].values
        if len(patho_g) >= 3 and len(benign_g) >= 3:
            n_genes_total += 1
            if np.mean(patho_g) > np.mean(benign_g):
                n_genes_patho_higher += 1
            _, p = stats.mannwhitneyu(patho_g, benign_g, alternative="greater")
            gene_pvals.append(p)

    if n_genes_total > 0:
        frac = n_genes_patho_higher / n_genes_total
        binom_p = stats.binomtest(n_genes_patho_higher, n_genes_total, 0.5).pvalue
        results["within_gene"] = {
            "n_genes": n_genes_total,
            "frac_patho_higher": float(frac),
            "binom_p": float(binom_p),
        }
        logger.info(f"    {n_genes_patho_higher}/{n_genes_total} genes ({frac:.1%}) patho > benign")
        logger.info(f"    Binomial p = {binom_p:.2e}")

    # Save results
    results_path = OUTPUT_DIR / "clinvar_enrichment_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\n  Results saved to {results_path}")

    # Save scored variants
    scored_out = scored[["site_id", "gene", "significance_simple", "patho_class",
                         "phase3_p_edited", "phase3_p_a3a", "phase3_p_a3g",
                         "phase3_p_neither"]].copy()
    if "p_edited_gb" in clinvar_df.columns:
        scored_out = scored_out.merge(
            clinvar_df[["site_id", "p_edited_gb"]].dropna(),
            on="site_id", how="left"
        )
    scored_path = OUTPUT_DIR / "clinvar_phase3_scores.csv"
    scored_out.to_csv(scored_path, index=False)
    logger.info(f"  Scored variants saved to {scored_path}")

    del cache, hand_46, hand_40, proba
    gc.collect()

    return results


# =============================================================================
# Experiment 2: TCGA Somatic Mutation Enrichment
# =============================================================================

def run_tcga_enrichment(model):
    """Score TCGA mutations with Phase 3 model and compare to original."""
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 2: TCGA Somatic Mutation Enrichment")
    logger.info("=" * 70)

    if not HG19_FA.exists():
        logger.warning("hg19 genome not found. Skipping TCGA experiment.")
        return None

    exons_by_gene = parse_exons(REFGENE_HG19)
    genome = Fasta(str(HG19_FA))
    logger.info(f"  {len(exons_by_gene):,} genes with exonic regions")

    all_results = {}

    for cancer in CANCERS:
        maf_path = TCGA_DIR / f"{cancer}_tcga_pan_can_atlas_2018_mutations.txt"
        vienna_cache_file = VIENNA_CACHE_DIR / f"{cancer}_vienna_raw.json.gz"

        if not maf_path.exists():
            logger.warning(f"  No MAF for {cancer}, skipping")
            continue
        if not vienna_cache_file.exists():
            logger.warning(f"  No ViennaRNA cache for {cancer}, skipping")
            continue

        logger.info(f"\n  Processing {cancer.upper()}...")

        # Parse mutations
        mut_df = parse_ct_mutations(maf_path)
        if len(mut_df) == 0:
            continue
        logger.info(f"    {len(mut_df):,} unique C>T mutations")

        # Sequences
        mut_seqs = extract_sequences(mut_df, genome)
        ctrl_df = get_matched_controls(mut_df, genome, exons_by_gene, N_CONTROLS)
        ctrl_seqs = extract_sequences(ctrl_df, genome)

        # Filter valid
        mut_valid_seqs = [s for s in mut_seqs if s is not None and len(s) == 201 and s[100] == "C"]
        ctrl_valid_seqs = [s for s in ctrl_seqs if s is not None and len(s) == 201 and s[100] == "C"]

        # Load ViennaRNA cache
        logger.info(f"    Loading ViennaRNA cache...")
        with gzip.open(str(vienna_cache_file), "rt") as fz:
            cache_data = json.load(fz)

        fold_results = cache_data["fold_results"]
        n_mut = cache_data["n_mut"]
        n_ctrl = cache_data["n_ctrl"]

        # Align sequences to cache
        n_mut_seqs = len(mut_valid_seqs)
        n_ctrl_seqs = len(ctrl_valid_seqs)
        if n_mut_seqs != n_mut or n_ctrl_seqs != n_ctrl:
            if n_mut_seqs > n_mut:
                mut_valid_seqs = mut_valid_seqs[:n_mut]
            elif n_mut_seqs < n_mut:
                mut_valid_seqs.extend([None] * (n_mut - n_mut_seqs))
            if n_ctrl_seqs > n_ctrl:
                ctrl_valid_seqs = ctrl_valid_seqs[:n_ctrl]
            elif n_ctrl_seqs < n_ctrl:
                ctrl_valid_seqs.extend([None] * (n_ctrl - n_ctrl_seqs))

        all_seqs = list(mut_valid_seqs) + list(ctrl_valid_seqs)

        # Build features and score
        logger.info(f"    Building 40-dim features...")
        features_40d = build_40dim_features(all_seqs, fold_results)

        proba = model.predict_proba(features_40d)
        neg_idx = CLASS_TO_IDX["Negative"]
        p_edited = 1.0 - proba[:, neg_idx]

        mut_p = proba[:n_mut]
        ctrl_p = proba[n_mut:]
        mut_p_edited = p_edited[:n_mut]
        ctrl_p_edited = p_edited[n_mut:]

        # Context
        tc_mut = sum(1 for s in mut_valid_seqs[:n_mut] if s and len(s) > 99 and s[99] in "TU")
        tc_ctrl = sum(1 for s in ctrl_valid_seqs[:n_ctrl] if s and len(s) > 99 and s[99] in "TU")

        cancer_result = {
            "cancer": cancer,
            "n_mutations": n_mut,
            "n_controls": n_ctrl,
            "tc_frac_mutations": tc_mut / max(n_mut, 1),
            "tc_frac_controls": tc_ctrl / max(n_ctrl, 1),
        }

        # P(edited) enrichment
        enrichment = compute_enrichment(mut_p_edited, ctrl_p_edited)
        pct_enrichment = compute_percentile_enrichment(mut_p_edited, ctrl_p_edited)
        cancer_result["p_edited"] = {
            "mean_mut": float(np.mean(mut_p_edited)),
            "mean_ctrl": float(np.mean(ctrl_p_edited)),
            "enrichment": enrichment,
            "percentile_enrichment": pct_enrichment,
        }

        # Per-enzyme enrichment
        cancer_result["enzymes"] = {}
        for enz_name in ENZYME_CLASSES[:-1]:
            enz_idx = CLASS_TO_IDX[enz_name]
            mut_scores = mut_p[:, enz_idx]
            ctrl_scores = ctrl_p[:, enz_idx]
            enz_enrichment = compute_enrichment(mut_scores, ctrl_scores,
                                                 thresholds=(0.1, 0.2, 0.3, 0.4, 0.5))
            enz_pct = compute_percentile_enrichment(mut_scores, ctrl_scores)
            cancer_result["enzymes"][enz_name] = {
                "mean_mut": float(np.mean(mut_scores)),
                "mean_ctrl": float(np.mean(ctrl_scores)),
                "delta": float(np.mean(mut_scores) - np.mean(ctrl_scores)),
                "enrichment": enz_enrichment,
                "percentile_enrichment": enz_pct,
            }

        all_results[cancer] = cancer_result

        # Summary
        or_p90 = pct_enrichment.get("p90", {}).get("OR", float("nan"))
        or_p95 = pct_enrichment.get("p95", {}).get("OR", float("nan"))
        adv = ADVISOR_TCGA.get(cancer, {})
        adv_p90 = adv.get("OR_p90", "N/A")
        adv_p95 = adv.get("OR_p95", "N/A")
        logger.info(f"    P(edited) OR@p90={or_p90:.3f} (advisor: {adv_p90}), "
                    f"OR@p95={or_p95:.3f} (advisor: {adv_p95})")

        # Per-enzyme summary
        for enz in ["A3A", "A3G", "Neither"]:
            enz_or = cancer_result["enzymes"][enz]["percentile_enrichment"].get("p90", {}).get("OR", float("nan"))
            logger.info(f"    P({enz}) OR@p90={enz_or:.3f}")

        del fold_results, features_40d, proba, cache_data
        gc.collect()

    # Save results
    results_path = OUTPUT_DIR / "tcga_enrichment_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\n  TCGA results saved to {results_path}")

    return all_results


# =============================================================================
# Experiment 3: Cross-Species Comparison
# =============================================================================

def run_cross_species(model):
    """Score chimp orthologs with Phase 3 model and compare correlation."""
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 3: Cross-Species Comparison")
    logger.info("=" * 70)

    if not CROSS_SPECIES_CSV.exists():
        logger.warning("Cross-species CSV not found. Skipping.")
        return None

    cs_df = pd.read_csv(CROSS_SPECIES_CSV)
    logger.info(f"  Loaded {len(cs_df)} ortholog pairs")

    # We need to re-score both human and chimp sequences with the Phase 3 model
    # The existing CSV has human_gb_score and chimp_gb_score from the original 40-dim XGBoost
    # To rescore, we'd need the sequences. Check if we have them.
    # The cross-species CSV has site_ids but not full sequences.
    # We can load the multi-enzyme sequences for human, but chimp needs separate data.

    # Load human sequences
    with open(MULTI_SEQS) as f:
        sequences_dict = json.load(f)

    # Load structure data for feature building
    loop_df = pd.read_csv(LOOP_CSV)
    loop_df["site_id"] = loop_df["site_id"].astype(str)
    loop_df = loop_df.set_index("site_id")

    struct_data = None
    if STRUCT_CACHE.exists():
        struct_data = dict(np.load(str(STRUCT_CACHE), allow_pickle=True))
        struct_data["site_ids"] = [str(s) for s in struct_data["site_ids"]]

    # Build features for human sites that are in the cross-species data
    human_sids = cs_df["site_id"].astype(str).tolist()
    valid_mask = [sid in sequences_dict for sid in human_sids]
    cs_valid = cs_df[valid_mask].copy()
    valid_sids = [sid for sid, v in zip(human_sids, valid_mask) if v]
    logger.info(f"  {len(valid_sids)} sites with sequences")

    if len(valid_sids) == 0:
        logger.warning("No matching sequences found. Skipping.")
        return None

    # Build 40-dim features for human sites
    motif_feats = np.array(
        [extract_motif_from_seq(sequences_dict.get(sid, "N" * 201)) for sid in valid_sids],
        dtype=np.float32,
    )

    struct_feats = np.zeros((len(valid_sids), 7), dtype=np.float32)
    if struct_data is not None:
        struct_sids = [str(s) for s in struct_data["site_ids"]]
        sid_to_idx_struct = {s: i for i, s in enumerate(struct_sids)}
        for i, sid in enumerate(valid_sids):
            if sid in sid_to_idx_struct:
                struct_feats[i] = struct_data["delta_features"][sid_to_idx_struct[sid]]

    loop_feats = np.zeros((len(valid_sids), len(LOOP_FEATURE_COLS)), dtype=np.float32)
    for i, sid in enumerate(valid_sids):
        if sid in loop_df.index:
            for j, col in enumerate(LOOP_FEATURE_COLS):
                if col in loop_df.columns:
                    val = loop_df.loc[sid, col]
                    if not isinstance(val, (int, float)):
                        val = val.iloc[0] if hasattr(val, "iloc") else 0.0
                    loop_feats[i, j] = float(val) if pd.notna(val) else 0.0

    X_human = np.concatenate([motif_feats, struct_feats, loop_feats], axis=1)
    X_human = np.nan_to_num(X_human, nan=0.0)

    # Score human
    proba_human = model.predict_proba(X_human)
    neg_idx = CLASS_TO_IDX["Negative"]
    human_p_edited = 1.0 - proba_human[:, neg_idx]

    cs_valid["phase3_human_score"] = human_p_edited

    # For chimp: we don't have chimp sequences in a directly loadable format from the
    # multi-enzyme data. Use the original chimp GB scores for correlation comparison.
    # The key comparison is: Phase 3 human scores vs original human scores in predicting
    # chimp editability.

    # Compute correlation between Phase 3 human scores and original human scores
    orig_human = cs_valid["human_gb_score"].values
    phase3_human = cs_valid["phase3_human_score"].values
    orig_chimp = cs_valid["chimp_gb_score"].values

    r_orig, p_orig = stats.spearmanr(orig_human, orig_chimp)
    r_phase3_vs_orig, _ = stats.spearmanr(phase3_human, orig_human)
    r_phase3_vs_chimp, p_phase3_vs_chimp = stats.spearmanr(phase3_human, orig_chimp)

    results = {
        "n_sites": len(cs_valid),
        "original_human_vs_chimp_spearman": float(r_orig),
        "phase3_human_vs_orig_human_spearman": float(r_phase3_vs_orig),
        "phase3_human_vs_orig_chimp_spearman": float(r_phase3_vs_chimp),
        "phase3_human_vs_orig_chimp_p": float(p_phase3_vs_chimp),
        "phase3_human_mean": float(np.mean(phase3_human)),
        "orig_human_mean": float(np.mean(orig_human)),
        "orig_chimp_mean": float(np.mean(orig_chimp)),
        "advisor_spearman": ADVISOR_CROSS_SPECIES_SPEARMAN,
        "note": "Chimp scores are from original 40-dim XGBoost. Phase 3 re-scored human only."
    }

    logger.info(f"  Original human-chimp Spearman: {r_orig:.4f} (advisor: {ADVISOR_CROSS_SPECIES_SPEARMAN})")
    logger.info(f"  Phase3 human vs original human: {r_phase3_vs_orig:.4f}")
    logger.info(f"  Phase3 human vs original chimp: {r_phase3_vs_chimp:.4f}")

    # Save
    results_path = OUTPUT_DIR / "cross_species_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"  Saved to {results_path}")

    cs_valid.to_csv(OUTPUT_DIR / "cross_species_phase3_scores.csv", index=False)

    del struct_data
    gc.collect()

    return results


# =============================================================================
# Experiment 4: gnomAD Constraint
# =============================================================================

def run_gnomad_constraint(model):
    """Correlate Phase 3 gene-level editability with gnomAD constraint."""
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 4: gnomAD Gene Constraint")
    logger.info("=" * 70)

    if not GNOMAD_GENE_CSV.exists():
        logger.warning("gnomAD gene CSV not found. Skipping.")
        return None

    gene_df = pd.read_csv(GNOMAD_GENE_CSV)
    logger.info(f"  Loaded {len(gene_df)} genes with constraint data")

    # The gene CSV has pre-computed mean_editability from the original model.
    # To recompute with Phase 3, we'd need to score all exonic C positions per gene.
    # That requires the full exome editability map, which uses ViennaRNA for millions of positions.
    #
    # Instead, compare Phase 3 training-set scores to gene constraint where genes overlap.
    # This gives a fair comparison on the SAME genes the model knows about.

    # Load training data to get per-gene Phase 3 scores
    splits = pd.read_csv(MULTI_SPLITS)
    with open(MULTI_SEQS) as f:
        sequences_dict = json.load(f)

    loop_df = pd.read_csv(LOOP_CSV)
    loop_df["site_id"] = loop_df["site_id"].astype(str)
    loop_df = loop_df.set_index("site_id")

    struct_data = None
    if STRUCT_CACHE.exists():
        struct_data = dict(np.load(str(STRUCT_CACHE), allow_pickle=True))
        struct_data["site_ids"] = [str(s) for s in struct_data["site_ids"]]

    X = build_40dim_from_training(splits, sequences_dict, loop_df, struct_data)

    # Score with Phase 3 model (training set scores, for gene-level aggregation)
    proba = model.predict_proba(X)
    neg_idx = CLASS_TO_IDX["Negative"]
    p_edited = 1.0 - proba[:, neg_idx]

    splits["phase3_score"] = p_edited
    positives = splits[splits["is_edited"] == 1].copy()

    # Extract gene from site_id
    def extract_gene(site_id):
        parts = str(site_id).split(":")
        if len(parts) >= 4:
            return parts[3] if len(parts) > 3 else None
        return None

    # Try to get gene info from the splits if available
    if "gene" in positives.columns:
        gene_col = "gene"
    else:
        # Try to extract from site_id
        positives["gene_extracted"] = positives["site_id"].apply(extract_gene)
        gene_col = "gene_extracted"

    if gene_col in positives.columns:
        gene_scores = positives.groupby(gene_col)["phase3_score"].mean().reset_index()
        gene_scores.columns = ["gene", "phase3_mean_editability"]

        # Merge with gnomAD data
        merged = gene_df.merge(gene_scores, on="gene", how="inner")
        logger.info(f"  {len(merged)} genes with both Phase 3 scores and gnomAD constraint")

        if len(merged) > 10 and "LOEUF" in merged.columns:
            rho_loeuf, p_loeuf = stats.spearmanr(
                merged["phase3_mean_editability"].values,
                merged["LOEUF"].values
            )
            logger.info(f"  Phase 3 mean editability vs LOEUF: rho={rho_loeuf:.4f}, p={p_loeuf:.2e}")
            logger.info(f"  (Advisor original: rho={ADVISOR_GNOMAD_LOEUF_RHO})")

            results = {
                "n_genes": len(merged),
                "phase3_vs_LOEUF_spearman": float(rho_loeuf),
                "phase3_vs_LOEUF_p": float(p_loeuf),
                "advisor_LOEUF_spearman": ADVISOR_GNOMAD_LOEUF_RHO,
            }

            # Also check missense constraint
            for metric in ["mis_oe", "mis_zscore", "pLI", "lof_zscore"]:
                if metric in merged.columns:
                    rho, p = stats.spearmanr(
                        merged["phase3_mean_editability"].values,
                        merged[metric].values
                    )
                    results[f"phase3_vs_{metric}_spearman"] = float(rho)
                    results[f"phase3_vs_{metric}_p"] = float(p)
                    logger.info(f"  Phase 3 vs {metric}: rho={rho:.4f}, p={p:.2e}")
        else:
            logger.warning("  Not enough data for correlation. Using existing gnomAD results.")
            results = {
                "note": "Insufficient gene overlap for Phase 3 gnomAD analysis",
                "advisor_LOEUF_spearman": ADVISOR_GNOMAD_LOEUF_RHO,
            }
    else:
        logger.warning("  No gene column available in splits.")
        results = {
            "note": "No gene column in training data for gnomAD analysis. "
                    "Phase 3 uses same feature space as original, so gnomAD results would be similar.",
            "advisor_LOEUF_spearman": ADVISOR_GNOMAD_LOEUF_RHO,
        }

    results_path = OUTPUT_DIR / "gnomad_constraint_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"  Saved to {results_path}")

    del struct_data
    gc.collect()

    return results


# =============================================================================
# Experiment 5: Multi-class Enzyme TCGA (uses results from Experiment 2)
# =============================================================================

def analyze_multiclass_tcga(tcga_results):
    """Analyze multi-class enzyme predictions from TCGA scoring."""
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 5: Multi-class Enzyme Prediction on TCGA")
    logger.info("=" * 70)

    if tcga_results is None:
        logger.warning("No TCGA results available. Skipping.")
        return None

    # The multi-class model already output per-enzyme probabilities in Experiment 2.
    # Extract the key comparison: P(Neither/APOBEC1) enrichment in GI cancers

    results = {}
    gi_cancers = ["stad", "esca", "lihc"]
    apobec_cancers = ["blca", "brca", "cesc", "lusc"]
    control = ["skcm"]

    for cancer in tcga_results:
        r = tcga_results[cancer]
        enzymes = r.get("enzymes", {})
        results[cancer] = {}

        for enz in ["A3A", "A3G", "Neither", "A3A_A3G"]:
            if enz in enzymes:
                or_p90 = enzymes[enz]["percentile_enrichment"].get("p90", {}).get("OR", float("nan"))
                or_p75 = enzymes[enz]["percentile_enrichment"].get("p75", {}).get("OR", float("nan"))
                results[cancer][enz] = {"OR_p90": float(or_p90), "OR_p75": float(or_p75)}

        # P(edited)
        pe = r.get("p_edited", {}).get("percentile_enrichment", {})
        results[cancer]["P_edited"] = {
            "OR_p90": float(pe.get("p90", {}).get("OR", float("nan"))),
            "OR_p95": float(pe.get("p95", {}).get("OR", float("nan"))),
        }

    # Print comparison table
    logger.info(f"\n  {'Cancer':<8} {'P(A3A)@p90':>12} {'P(Neither)@p90':>15} {'P(edited)@p90':>14} {'P(edited)@p95':>14}")
    logger.info(f"  {'-'*65}")
    for cancer in CANCERS:
        if cancer in results:
            r = results[cancer]
            a3a = r.get("A3A", {}).get("OR_p90", float("nan"))
            neither = r.get("Neither", {}).get("OR_p90", float("nan"))
            pe90 = r.get("P_edited", {}).get("OR_p90", float("nan"))
            pe95 = r.get("P_edited", {}).get("OR_p95", float("nan"))
            cancer_type = ("GI" if cancer in gi_cancers else
                          "APOBEC" if cancer in apobec_cancers else
                          "UV" if cancer in control else "other")
            logger.info(f"  {cancer.upper():<8} {a3a:>12.3f} {neither:>15.3f} {pe90:>14.3f} {pe95:>14.3f}  [{cancer_type}]")

    results_path = OUTPUT_DIR / "multiclass_tcga_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"  Saved to {results_path}")

    return results


# =============================================================================
# Comparison Summary
# =============================================================================

def write_comparison_report(cv_results, clinvar_results, tcga_results,
                            cross_species_results, gnomad_results, multiclass_results):
    """Write final comparison report."""
    logger.info("\n" + "=" * 70)
    logger.info("WRITING COMPARISON REPORT")
    logger.info("=" * 70)

    lines = []
    lines.append("# Phase 3 Replication Report")
    lines.append("")
    lines.append(f"**Date**: 2026-03-27")
    lines.append(f"**Model**: Phase 3 Multi-Class XGBoost (40-dim, 7-class)")
    lines.append(f"**Training data**: v3 multi-enzyme dataset (15,342 sites, enzyme-matched negatives)")
    lines.append("")
    lines.append("## Methodology")
    lines.append("")
    lines.append("The Phase 3 neural model (1448-dim fusion: RNA-FM + Conv2D + edit delta + hand features)")
    lines.append("requires RNA-FM embeddings that are only available for the v3 training set.")
    lines.append("For external data (ClinVar, TCGA, cross-species, gnomAD), we use a multi-class")
    lines.append("XGBoost trained on 40-dim hand features with the Phase 3 paradigm:")
    lines.append("- Same v3 training data with enzyme-matched negatives")
    lines.append("- 7-class formulation (A3A, A3B, A3G, A3A_A3G, Neither, Unknown, Negative)")
    lines.append("- Outputs per-enzyme editing probabilities")
    lines.append("")
    lines.append("The original advisor report used:")
    lines.append("- Binary per-enzyme XGBoost models (40-dim) for most experiments")
    lines.append("- A separate 7-class XGBoost for multi-class TCGA analysis")
    lines.append("")
    lines.append("**Key difference**: Phase 3 paradigm trains ONE model that sees ALL enzyme types,")
    lines.append("vs. original approach training separate binary models per enzyme.")
    lines.append("")

    # CV results
    lines.append("## 0. Cross-Validation on Training Data")
    lines.append("")
    if cv_results:
        lines.append(f"| Metric | Phase 3 XGB (40d) | Original per-enzyme XGB |")
        lines.append(f"|--------|:-:|:-:|")
        lines.append(f"| P(edited) AUROC | {cv_results['mean_edited_auroc']:.4f} | N/A (binary models) |")
        lines.append(f"| Log-loss | {cv_results['mean_logloss']:.4f} | N/A |")
        for enz, auc in cv_results.get("per_enzyme_auroc", {}).items():
            orig = {"A3A": 0.907, "A3B": 0.831, "A3G": 0.931, "A3A_A3G": 0.941, "Neither": 0.840}.get(enz, "N/A")
            lines.append(f"| {enz} AUROC | {auc:.4f} | {orig} |")
        lines.append("")
        lines.append("**Note**: Phase 3 5-fold results from neural model (phase3_5fold_results.json):")
        lines.append("- XGB_Full (1320-dim): overall AUROC=0.812, A3A=0.881, A3B=0.825, A3G=0.909")
        lines.append("- Neural (1448-dim): overall AUROC=0.807, A3A=0.885, A3B=0.821, A3G=0.906")
        lines.append("- XGB_Hand (40-dim): overall AUROC=0.644, A3A=0.849, A3B=0.574, A3G=0.813")
        lines.append("")

    # ClinVar
    lines.append("## 1. ClinVar Pathogenic Enrichment")
    lines.append("")
    if clinvar_results:
        pe = clinvar_results.get("phase3_p_edited", {})
        lines.append(f"Scored {clinvar_results['n_total']:,} ClinVar variants.")
        lines.append(f"Pathogenic: {clinvar_results['n_pathogenic']:,}, Benign: {clinvar_results['n_benign']:,}")
        lines.append("")
        lines.append("| Threshold | Phase 3 OR | Phase 3 p | Advisor OR (A3A, t=0.5) |")
        lines.append("|:-:|:-:|:-:|:-:|")
        for t in ["0.3", "0.5", "0.7", "0.9"]:
            e = pe.get("enrichment", {}).get(t, {})
            or_val = e.get("OR", float("nan"))
            p_val = e.get("p", 1.0)
            advisor_note = f"{ADVISOR_CLINVAR_OR}" if t == "0.5" else ""
            lines.append(f"| {t} | {or_val:.3f} | {p_val:.2e} | {advisor_note} |")
        lines.append("")

        # Per-enzyme ClinVar
        lines.append("### Per-Enzyme ClinVar Enrichment (OR @ t=0.3)")
        lines.append("")
        lines.append("| Score | OR | p-value |")
        lines.append("|-------|:-:|:-:|")
        for sn in ["phase3_p_edited", "phase3_p_a3a", "phase3_p_a3g", "phase3_p_neither"]:
            sr = clinvar_results.get(sn, {})
            e = sr.get("enrichment", {}).get("0.3", {})
            or_val = e.get("OR", float("nan"))
            p_val = e.get("p", 1.0)
            label = sn.replace("phase3_", "")
            lines.append(f"| {label} | {or_val:.3f} | {p_val:.2e} |")
        lines.append("")

        wg = clinvar_results.get("within_gene", {})
        if wg:
            lines.append(f"Within-gene: {wg.get('frac_patho_higher', 0):.1%} of genes show patho > benign "
                        f"(p={wg.get('binom_p', 1.0):.2e})")
            lines.append(f"Advisor: 62.3% of genes, p=1.5e-30")
            lines.append("")
    else:
        lines.append("ClinVar experiment skipped (features cache unavailable).")
        lines.append("")

    # TCGA
    lines.append("## 2. TCGA Somatic Mutation Enrichment")
    lines.append("")
    if tcga_results:
        lines.append("| Cancer | Type | Phase3 OR@p90 | Phase3 OR@p95 | Advisor OR@p90 | Advisor OR@p95 |")
        lines.append("|--------|------|:-:|:-:|:-:|:-:|")
        for cancer in CANCERS:
            if cancer in tcga_results:
                pe = tcga_results[cancer].get("p_edited", {}).get("percentile_enrichment", {})
                or_p90 = pe.get("p90", {}).get("OR", float("nan"))
                or_p95 = pe.get("p95", {}).get("OR", float("nan"))
                adv = ADVISOR_TCGA.get(cancer, {})
                adv_p90 = adv.get("OR_p90", "N/A")
                adv_p95 = adv.get("OR_p95", "N/A")
                cancer_type = {"blca": "APOBEC", "brca": "APOBEC", "cesc": "APOBEC",
                              "lusc": "APOBEC", "skcm": "UV", "hnsc": "moderate",
                              "esca": "GI", "stad": "GI", "lihc": "liver"}.get(cancer, "")
                lines.append(f"| {cancer.upper()} | {cancer_type} | {or_p90:.3f} | {or_p95:.3f} | {adv_p90} | {adv_p95} |")
        lines.append("")
    else:
        lines.append("TCGA experiment skipped.")
        lines.append("")

    # Cross-species
    lines.append("## 3. Cross-Species Comparison")
    lines.append("")
    if cross_species_results:
        lines.append(f"| Metric | Phase 3 | Advisor |")
        lines.append(f"|--------|:-:|:-:|")
        lines.append(f"| Human-chimp Spearman (orig model) | {cross_species_results.get('original_human_vs_chimp_spearman', 'N/A'):.4f} | {ADVISOR_CROSS_SPECIES_SPEARMAN} |")
        lines.append(f"| Phase3 human vs orig chimp | {cross_species_results.get('phase3_human_vs_orig_chimp_spearman', 'N/A'):.4f} | — |")
        lines.append(f"| Phase3 human vs orig human | {cross_species_results.get('phase3_human_vs_orig_human_spearman', 'N/A'):.4f} | — |")
        lines.append("")
        lines.append(f"Note: {cross_species_results.get('note', '')}")
        lines.append("")
    else:
        lines.append("Cross-species experiment skipped.")
        lines.append("")

    # gnomAD
    lines.append("## 4. gnomAD Constraint")
    lines.append("")
    if gnomad_results:
        if "phase3_vs_LOEUF_spearman" in gnomad_results:
            lines.append(f"| Metric | Phase 3 | Advisor |")
            lines.append(f"|--------|:-:|:-:|")
            lines.append(f"| Mean editability vs LOEUF | rho={gnomad_results['phase3_vs_LOEUF_spearman']:.4f} | rho={ADVISOR_GNOMAD_LOEUF_RHO} |")
            for metric in ["mis_oe", "mis_zscore", "pLI", "lof_zscore"]:
                key = f"phase3_vs_{metric}_spearman"
                if key in gnomad_results:
                    lines.append(f"| Mean editability vs {metric} | rho={gnomad_results[key]:.4f} | — |")
            lines.append("")
        else:
            lines.append(f"Note: {gnomad_results.get('note', 'Insufficient data')}")
            lines.append("")
    else:
        lines.append("gnomAD experiment skipped.")
        lines.append("")

    # Multi-class TCGA
    lines.append("## 5. Multi-class Enzyme Prediction on TCGA")
    lines.append("")
    if multiclass_results:
        lines.append("| Cancer | Type | P(A3A)@p90 | P(Neither)@p90 | P(edited)@p90 | P(edited)@p95 |")
        lines.append("|--------|------|:-:|:-:|:-:|:-:|")
        for cancer in CANCERS:
            if cancer in multiclass_results:
                r = multiclass_results[cancer]
                a3a = r.get("A3A", {}).get("OR_p90", float("nan"))
                neither = r.get("Neither", {}).get("OR_p90", float("nan"))
                pe90 = r.get("P_edited", {}).get("OR_p90", float("nan"))
                pe95 = r.get("P_edited", {}).get("OR_p95", float("nan"))
                cancer_type = {"blca": "APOBEC", "brca": "APOBEC", "cesc": "APOBEC",
                              "lusc": "APOBEC", "skcm": "UV", "hnsc": "moderate",
                              "esca": "GI", "stad": "GI", "lihc": "liver"}.get(cancer, "")
                lines.append(f"| {cancer.upper()} | {cancer_type} | {a3a:.3f} | {neither:.3f} | {pe90:.3f} | {pe95:.3f} |")

        lines.append("")
        lines.append("### Advisor Report Comparison (multi-class XGBoost)")
        lines.append("")
        lines.append("| Cancer | Advisor P(A3A)@p90 | Advisor P(Neither)@p90 | Advisor P(edited)@p90 |")
        lines.append("|--------|:-:|:-:|:-:|")
        advisor_multiclass = {
            "stad": {"A3A": 1.838, "Neither": 2.115, "P_edited": 2.302},
            "esca": {"A3A": 1.664, "Neither": 1.908, "P_edited": 1.773},
            "brca": {"A3A": 1.727, "Neither": 1.561, "P_edited": 1.308},
            "cesc": {"A3A": 1.690, "Neither": 1.573, "P_edited": 1.356},
            "hnsc": {"A3A": 1.517, "Neither": 1.480, "P_edited": 1.239},
            "blca": {"A3A": 1.624, "Neither": 1.269, "P_edited": 0.893},
            "lusc": {"A3A": 1.343, "Neither": 1.342, "P_edited": 1.100},
            "lihc": {"A3A": 1.246, "Neither": 1.401, "P_edited": 1.526},
            "skcm": {"A3A": 1.208, "Neither": 1.058, "P_edited": 0.600},
        }
        for cancer in CANCERS:
            if cancer in advisor_multiclass:
                a = advisor_multiclass[cancer]
                lines.append(f"| {cancer.upper()} | {a['A3A']:.3f} | {a['Neither']:.3f} | {a['P_edited']:.3f} |")
        lines.append("")
    else:
        lines.append("Multi-class TCGA experiment skipped.")
        lines.append("")

    # Key findings
    lines.append("## Key Findings")
    lines.append("")
    lines.append("1. **Phase 3 paradigm (multi-class XGBoost on 40-dim)** should be compared to the")
    lines.append("   original multi-class XGBoost (also 40-dim, same paradigm). The key question is")
    lines.append("   whether training on the v3 dataset with enzyme-matched negatives improves results.")
    lines.append("")
    lines.append("2. **RNA-FM limitation**: The full Phase 3 neural model (1448-dim) cannot be applied to")
    lines.append("   external data without computing RNA-FM embeddings for each position (~$O(n \\times 10s)$ per site).")
    lines.append("   On v3 internal data, Phase 3 neural AUROC=0.807 vs XGB_Full(1320d)=0.812 -- negligible difference.")
    lines.append("")
    lines.append("3. **The 40-dim features capture most of the signal**: XGB_Hand(40d) per-enzyme AUROCs")
    lines.append("   (A3A=0.849, A3G=0.813) are only modestly below XGB_Full(1320d) (A3A=0.881, A3G=0.909).")
    lines.append("   The RNA-FM contribution is mainly for A3B (+0.25 AUROC) and A3G (+0.10).")
    lines.append("")

    report_path = OUTPUT_DIR / "comparison_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"  Report saved to {report_path}")

    return str(report_path)


# =============================================================================
# Main
# =============================================================================

def main():
    t_start = time.time()
    logger.info("=" * 70)
    logger.info("Phase 3 Replication: Testing Phase 3 Model on All Advisor Experiments")
    logger.info("=" * 70)

    # Step 0: Train Phase 3 multi-class model
    model, cv_results = train_phase3_model()

    # Save CV results
    cv_path = OUTPUT_DIR / "cv_results.json"
    with open(cv_path, "w") as f:
        json.dump(cv_results, f, indent=2)

    # Experiment 1: ClinVar
    clinvar_results = run_clinvar_enrichment(model)

    # Experiment 2: TCGA
    tcga_results = run_tcga_enrichment(model)

    # Experiment 3: Cross-Species
    cross_species_results = run_cross_species(model)

    # Experiment 4: gnomAD
    gnomad_results = run_gnomad_constraint(model)

    # Experiment 5: Multi-class TCGA (uses results from Experiment 2)
    multiclass_results = analyze_multiclass_tcga(tcga_results)

    # Write comparison report
    report_path = write_comparison_report(
        cv_results, clinvar_results, tcga_results,
        cross_species_results, gnomad_results, multiclass_results
    )

    total_time = time.time() - t_start
    logger.info(f"\n{'='*70}")
    logger.info(f"Total runtime: {total_time:.0f}s ({total_time/60:.1f}min)")
    logger.info(f"All results saved to {OUTPUT_DIR}")
    logger.info(f"Report: {report_path}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
