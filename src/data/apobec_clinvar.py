"""ClinVar scoring utilities for APOBEC editing prediction.

Provides functions for loading ClinVar C>U variants, computing features
in parallel using ViennaRNA, scoring with trained models, and analyzing
pathogenic enrichment with Bayesian prior recalibration.
"""

import logging
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from src.data.apobec_feature_extraction import (
    compute_vienna_features,
    encode_rnasee_from_seq,
    extract_motif_from_seq,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ClinVar data loading
# ---------------------------------------------------------------------------

def load_clinvar_c2u(csv_path: Path) -> pd.DataFrame:
    """Load ClinVar C>U variants with simplified clinical significance.

    Expects a CSV with at least: site_id, sequence, clinical_significance.
    Adds a 'significance_simple' column with standardized categories:
    Pathogenic, Likely_pathogenic, VUS, Conflicting, Likely_benign, Benign, Other.

    Args:
        csv_path: Path to ClinVar C>U variants CSV.

    Returns:
        DataFrame with significance_simple column added.
    """
    df = pd.read_csv(csv_path, low_memory=False)
    logger.info("Loaded %d ClinVar C>U variants from %s", len(df), csv_path.name)

    def simplify_significance(sig):
        sig_lower = str(sig).lower()
        if "pathogenic" in sig_lower and "likely" not in sig_lower and "conflicting" not in sig_lower:
            if "benign" not in sig_lower:
                return "Pathogenic"
        if "likely_pathogenic" in sig_lower:
            return "Likely_pathogenic"
        if "benign" in sig_lower and "likely" not in sig_lower and "pathogenic" not in sig_lower:
            if "conflicting" not in sig_lower:
                return "Benign"
        if "likely_benign" in sig_lower:
            return "Likely_benign"
        if "uncertain" in sig_lower:
            return "VUS"
        if "conflicting" in sig_lower:
            return "Conflicting"
        return "Other"

    df["significance_simple"] = df["clinical_significance"].apply(simplify_significance)

    logger.info("Clinical significance breakdown:")
    for sig, count in df["significance_simple"].value_counts().items():
        logger.info("  %s: %d", sig, count)

    return df


# ---------------------------------------------------------------------------
# Parallel feature computation
# ---------------------------------------------------------------------------

def _compute_single_variant_features(args):
    """Worker function: compute all features for one ClinVar variant.

    Returns (site_id, hand_46, rnasee_50) or (site_id, None, None) on failure.
    """
    site_id, sequence = args
    try:
        seq = sequence.upper().replace("T", "U")

        # Motif features (24-dim)
        motif = extract_motif_from_seq(seq)

        # RNAsee 50-bit encoding
        rnasee = encode_rnasee_from_seq(seq)

        # ViennaRNA structure features (slow)
        struct_delta, loop_feats, baseline_struct = compute_vienna_features(seq)

        # Concatenate: motif(24) + struct_delta(7) + loop(9) + baseline(6) = 46
        hand_46 = np.concatenate([motif, struct_delta, loop_feats, baseline_struct])

        return site_id, hand_46, rnasee
    except Exception:
        return site_id, None, None


def compute_clinvar_features_batch(
    sequences: List[Tuple[str, str]],
    n_workers: int = 12,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Compute features for ClinVar variants using multiprocessing.

    Args:
        sequences: List of (site_id, sequence) tuples.
        n_workers: Number of parallel workers (default 12).

    Returns:
        Tuple of:
            site_ids: List of successfully processed site IDs.
            hand_46_features: [N, 46] float32 array.
            rnasee_features: [N, 50] float32 array.
    """
    logger.info("Computing features for %d variants with %d workers...",
                len(sequences), n_workers)

    site_ids = []
    hand_features = []
    rnasee_features = []
    n_failed = 0

    with Pool(n_workers) as pool:
        for result in pool.imap_unordered(_compute_single_variant_features, sequences):
            sid, hand_46, rnasee = result
            if hand_46 is not None:
                site_ids.append(sid)
                hand_features.append(hand_46)
                rnasee_features.append(rnasee)
            else:
                n_failed += 1

    if n_failed > 0:
        logger.warning("%d variants failed feature computation", n_failed)

    logger.info("Successfully computed features for %d / %d variants",
                len(site_ids), len(sequences))

    hand_arr = np.array(hand_features, dtype=np.float32) if hand_features else np.zeros((0, 46), dtype=np.float32)
    rnasee_arr = np.array(rnasee_features, dtype=np.float32) if rnasee_features else np.zeros((0, 50), dtype=np.float32)

    return site_ids, hand_arr, rnasee_arr


# ---------------------------------------------------------------------------
# Model scoring
# ---------------------------------------------------------------------------

def score_clinvar_with_model(
    features: np.ndarray,
    model,
    site_ids: List[str],
) -> pd.Series:
    """Apply a trained sklearn model to features and return prediction scores.

    Args:
        features: [N, D] feature matrix.
        model: Trained sklearn model with predict_proba method.
        site_ids: List of site IDs corresponding to feature rows.

    Returns:
        Series of predicted probabilities indexed by site_id.
    """
    probs = model.predict_proba(features)[:, 1]
    return pd.Series(probs, index=site_ids, name="p_edited")


# ---------------------------------------------------------------------------
# Pathogenic enrichment analysis
# ---------------------------------------------------------------------------

def compute_pathogenic_enrichment(
    scores: pd.Series,
    labels: pd.Series,
    thresholds: List[float],
) -> Dict:
    """Compute odds ratio and p-value for pathogenic enrichment at each threshold.

    Uses Fisher's exact test on Pathogenic+Likely_pathogenic vs Benign+Likely_benign.

    Args:
        scores: Series of model scores indexed by site_id.
        labels: Series of clinical significance labels (same index as scores).
        thresholds: List of score thresholds to evaluate.

    Returns:
        Dict mapping threshold -> {or, p, ci_low, ci_high, n_predicted, n_path,
        n_benign, path_rate, bg_rate}.
    """
    # Align indices
    common = scores.index.intersection(labels.index)
    scores = scores.loc[common]
    labels = labels.loc[common]

    # Filter to Path/LP/Ben/LB only
    path_mask = labels.isin(["Pathogenic", "Likely_pathogenic"])
    ben_mask = labels.isin(["Benign", "Likely_benign"])
    evaluable = path_mask | ben_mask
    scores_eval = scores[evaluable]
    is_path = path_mask[evaluable]

    results = {}
    for t in thresholds:
        predicted = scores_eval >= t
        n_pred = predicted.sum()
        if n_pred < 10:
            continue

        p_pred = is_path[predicted].sum()
        b_pred = (~is_path)[predicted].sum()
        p_rest = is_path[~predicted].sum()
        b_rest = (~is_path)[~predicted].sum()

        if (p_pred + b_pred) == 0 or (p_rest + b_rest) == 0:
            continue

        table = [[p_pred, b_pred], [p_rest, b_rest]]
        odds, pval = fisher_exact(table)

        # 95% CI via Woolf's method
        log_or = np.log(odds) if odds > 0 else 0
        se = np.sqrt(
            1 / max(p_pred, 1) + 1 / max(b_pred, 1)
            + 1 / max(p_rest, 1) + 1 / max(b_rest, 1)
        )
        ci_low = np.exp(log_or - 1.96 * se)
        ci_high = np.exp(log_or + 1.96 * se)

        results[t] = {
            "or": round(odds, 4),
            "p": float(pval),
            "ci_low": round(ci_low, 4),
            "ci_high": round(ci_high, 4),
            "n_predicted": int(p_pred + b_pred),
            "n_path": int(p_pred),
            "n_benign": int(b_pred),
            "path_rate": round(p_pred / (p_pred + b_pred) * 100, 2),
            "bg_rate": round(p_rest / (p_rest + b_rest) * 100, 2) if (p_rest + b_rest) > 0 else 0,
        }

    return results


# ---------------------------------------------------------------------------
# Bayesian prior recalibration
# ---------------------------------------------------------------------------

def bayesian_recalibrate(
    p_model: np.ndarray,
    pi_model: float,
    pi_real: float,
) -> np.ndarray:
    """Adjust model probabilities from training prevalence to real-world prevalence.

    Uses Bayes' theorem to recalibrate:
        P_cal = P_model * (pi_real/pi_model) /
                [P_model * (pi_real/pi_model) + (1-P_model) * ((1-pi_real)/(1-pi_model))]

    Args:
        p_model: Array of model predicted probabilities.
        pi_model: Training prevalence (effective prior used during training).
        pi_real: Real-world prevalence of positive class.

    Returns:
        Array of calibrated probabilities.
    """
    prior_ratio = pi_real / pi_model
    inv_prior_ratio = (1 - pi_real) / (1 - pi_model)
    p_cal = (p_model * prior_ratio) / (
        p_model * prior_ratio + (1 - p_model) * inv_prior_ratio
    )
    return p_cal


def calibrated_threshold(
    pi_model: float,
    pi_real: float,
    p_cal_target: float = 0.5,
) -> float:
    """Find model threshold where calibrated probability equals p_cal_target.

    Solves: p_cal_target = p * a / (p * a + (1-p) * b)
    where a = pi_real/pi_model, b = (1-pi_real)/(1-pi_model).

    Args:
        pi_model: Training prevalence.
        pi_real: Real-world prevalence.
        p_cal_target: Desired calibrated probability (default 0.5).

    Returns:
        Model probability threshold.
    """
    a = pi_real / pi_model
    b = (1 - pi_real) / (1 - pi_model)
    t = p_cal_target * b / (a * (1 - p_cal_target) + p_cal_target * b)
    return t
