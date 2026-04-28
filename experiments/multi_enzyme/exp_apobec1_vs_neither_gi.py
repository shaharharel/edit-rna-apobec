#!/usr/bin/env python3
"""APOBEC1 head vs Neither adapter — does the APOBEC1 head also predict GI
cancer enrichment the way the Neither adapter does?

Motivating observation (from tcga_apobec1_head_OR.csv):
  Neither adapter        GI mean OR@p90 = 3.63  non-GI = 2.76  ratio = 1.31
  APOBEC1 head (mouse)   GI mean OR@p90 = 1.13  non-GI = 1.51  ratio = 0.75

If Neither were truly learning APOBEC1 biology, a head trained directly on
484 mouse KO-validated APOBEC1 sites should replicate the GI-high-non-GI-low
pattern. It does the opposite. Quantify this with:

  1. Pooled GI (COADREAD,STAD,ESCA,LIHC) vs pooled non-GI ORs at p50/p75/p90/p95,
     with Fisher's exact p-values, for each head.
  2. Per-cancer Spearman(APOBEC1, Neither) across mutations only.
  3. Logistic regression on pooled GI: is_mutation ~ APOBEC1 + Neither,
     to see if APOBEC1 adds any signal beyond Neither (and vice versa).
  4. Permutation test on the GI/non-GI OR ratio — is APOBEC1's reversal
     (ratio < 1) statistically distinguishable from Neither's (ratio > 1)?

Uses cached TCGA embeddings + hand features, and the saved APOBEC1 head
(experiments/multi_enzyme/outputs/apobec1_head/apobec1_head.pt) +
Phase3Model checkpoint. Does not retrain.

Output: experiments/multi_enzyme/outputs/apobec1_head/gi_vs_nongi.json
        experiments/multi_enzyme/outputs/apobec1_head/gi_vs_nongi.log
"""

from __future__ import annotations

import gc
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from scipy.stats import fisher_exact

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "multi_enzyme" / "outputs" / "apobec1_head"
LOG_FILE = OUTPUT_DIR / "gi_vs_nongi.log"
RESULTS_JSON = OUTPUT_DIR / "gi_vs_nongi.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(LOG_FILE, mode="w")],
)
logger = logging.getLogger(__name__)

HAND_DIR = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "tcga_hand_features"
EMB_DIR = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "embeddings"
RAW_SCORES_DIR = PROJECT_ROOT / "experiments" / "multi_enzyme" / "outputs" / "tcga_gnomad" / "raw_scores"
PHASE3_CKPT = (
    PROJECT_ROOT / "experiments" / "multi_enzyme" / "outputs"
    / "phase3_neural_true_validation" / "phase3_neural_full.pt"
)
APOBEC1_HEAD_CKPT = OUTPUT_DIR / "apobec1_head.pt"

D_INPUT = 1320
D_SHARED = 128
ENZYMES = ["A3A", "A3B", "A3G", "A3A_A3G", "Neither"]
N_ENZYMES_CLS = 6
GI_CANCERS = ["coadread", "stad", "esca", "lihc"]
NON_GI_CANCERS = ["blca", "brca", "cesc", "lusc", "hnsc", "skcm"]
ALL_CANCERS = GI_CANCERS + NON_GI_CANCERS
PERCENTILES = (50, 75, 90, 95)
SEED = 42

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else torch.device("cpu")
)
logger.info("Device: %s", DEVICE)


class Phase3Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_encoder = nn.Sequential(
            nn.Linear(D_INPUT, 256), nn.GELU(), nn.Dropout(0.3), nn.LayerNorm(256),
            nn.Linear(256, D_SHARED), nn.GELU(), nn.Dropout(0.2),
        )
        self.binary_head = nn.Linear(D_SHARED, 1)
        self.enzyme_adapters = nn.ModuleDict({
            enz: nn.Sequential(nn.Linear(D_SHARED, 32), nn.GELU(), nn.Linear(32, 1))
            for enz in ENZYMES
        })
        self.enzyme_classifier = nn.Sequential(
            nn.Linear(D_SHARED, 64), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(64, N_ENZYMES_CLS),
        )


class APOBEC1Head(nn.Module):
    def __init__(self, d_shared: int = 128):
        super().__init__()
        self.species_proj = nn.Sequential(
            nn.Linear(1, 16), nn.GELU(), nn.Linear(16, d_shared),
        )
        self.head = nn.Sequential(
            nn.Linear(d_shared, 32), nn.GELU(), nn.Linear(32, 1),
        )

    def forward(self, shared: torch.Tensor, species: torch.Tensor) -> torch.Tensor:
        bias = self.species_proj(species)
        x = shared + bias
        return self.head(x).squeeze(-1)


def load_models():
    logger.info("Loading Phase3Model checkpoint from %s", PHASE3_CKPT)
    ckpt = torch.load(PHASE3_CKPT, weights_only=False, map_location="cpu")
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model = Phase3Model()
    model.load_state_dict(state, strict=False)
    model.to(DEVICE).eval()

    logger.info("Loading APOBEC1 head from %s", APOBEC1_HEAD_CKPT)
    head = APOBEC1Head(D_SHARED)
    head.load_state_dict(torch.load(APOBEC1_HEAD_CKPT, weights_only=False, map_location="cpu"))
    head.to(DEVICE).eval()
    return model, head


def score_cancer(model: Phase3Model, head: APOBEC1Head, cancer: str):
    """Returns dict with mut_mask, ctrl_mask, apobec1_probs, neither_probs.
    Returns None if caches are missing."""
    raw_path = RAW_SCORES_DIR / f"{cancer}_scores.csv"
    hand_path = HAND_DIR / f"{cancer}_hand40.npy"
    emb_path = EMB_DIR / f"rnafm_tcga_{cancer}.pt"
    if not all(p.exists() for p in (raw_path, hand_path, emb_path)):
        logger.warning("  %s: missing caches (%s/%s/%s)", cancer,
                       raw_path.exists(), hand_path.exists(), emb_path.exists())
        return None

    raw = pd.read_csv(raw_path)
    hand = np.load(hand_path)
    emb = torch.load(emb_path, weights_only=False, map_location="cpu")
    n = len(raw)
    if hand.shape[0] != n or emb["pooled_orig"].shape[0] != n:
        logger.warning("  %s: shape mismatch", cancer)
        return None

    mut_mask = (raw["type"].values == "mutation")
    ctrl_mask = (raw["type"].values == "control")

    pooled_orig = emb["pooled_orig"].numpy().astype(np.float32)
    pooled_edited = emb["pooled_edited"].numpy().astype(np.float32)
    edit_delta = pooled_edited - pooled_orig
    del emb
    X = np.concatenate([pooled_orig, edit_delta, hand], axis=1).astype(np.float32)
    del pooled_orig, pooled_edited, edit_delta
    gc.collect()

    # Score both heads in one pass over the shared encoder
    apobec1_probs = np.zeros(n, dtype=np.float32)
    neither_probs = np.zeros(n, dtype=np.float32)
    BATCH = 256
    with torch.no_grad():
        for b in range(0, n, BATCH):
            e = min(b + BATCH, n)
            xb = torch.from_numpy(X[b:e]).float().to(DEVICE)
            shared = model.shared_encoder(xb)
            # APOBEC1 head (species = 0 for human TCGA)
            sb = torch.zeros((e - b, 1), dtype=torch.float32, device=DEVICE)
            apobec1_probs[b:e] = torch.sigmoid(head(shared, sb)).cpu().numpy()
            # Neither adapter on shared encoder
            neither_logit = model.enzyme_adapters["Neither"](shared).squeeze(-1)
            neither_probs[b:e] = torch.sigmoid(neither_logit).cpu().numpy()
    del X
    gc.collect()
    return {
        "cancer": cancer,
        "n_mut": int(mut_mask.sum()),
        "n_ctrl": int(ctrl_mask.sum()),
        "mut_mask": mut_mask,
        "ctrl_mask": ctrl_mask,
        "apobec1": apobec1_probs,
        "neither": neither_probs,
    }


def or_at_percentile(scores, mut_mask, ctrl_mask, pct):
    pool = scores[mut_mask | ctrl_mask]
    if len(pool) < 50:
        return {"OR": float("nan"), "p": 1.0, "threshold": float("nan")}
    thresh = float(np.percentile(pool, pct))
    ma = int((scores[mut_mask] >= thresh).sum())
    mb = int((scores[mut_mask] < thresh).sum())
    ca = int((scores[ctrl_mask] >= thresh).sum())
    cb = int((scores[ctrl_mask] < thresh).sum())
    if all(x > 0 for x in (ma, mb, ca, cb)):
        OR, p = fisher_exact([[ma, mb], [ca, cb]])
    else:
        OR, p = float("nan"), 1.0
    return {
        "OR": float(OR), "p": float(p), "threshold": thresh,
        "mut_above": ma, "mut_below": mb,
        "ctrl_above": ca, "ctrl_below": cb,
    }


def _global_threshold(per_cancer, head_key, cancers, pct):
    """Percentile threshold over pooled mutations+controls across given cancers."""
    pool = []
    for c in cancers:
        rec = per_cancer.get(c)
        if rec is None:
            continue
        mask = rec["mut_mask"] | rec["ctrl_mask"]
        pool.append(rec[head_key][mask])
    if not pool:
        return float("nan")
    arr = np.concatenate(pool)
    return float(np.percentile(arr, pct))


def _per_cancer_counts(per_cancer, head_key, cancers, thresh):
    """For a given (head, threshold), return per-cancer (mut_above, mut_below,
    ctrl_above, ctrl_below) dict keyed by cancer."""
    out = {}
    for c in cancers:
        rec = per_cancer.get(c)
        if rec is None:
            continue
        s = rec[head_key]
        above = (s >= thresh)
        ma = int((above & rec["mut_mask"]).sum())
        mb = int(rec["mut_mask"].sum() - ma)
        ca = int((above & rec["ctrl_mask"]).sum())
        cb = int(rec["ctrl_mask"].sum() - ca)
        out[c] = (ma, mb, ca, cb)
    return out


def _OR_from_counts_sum(counts_dict, selected):
    ma = sum(counts_dict[c][0] for c in selected if c in counts_dict)
    mb = sum(counts_dict[c][1] for c in selected if c in counts_dict)
    ca = sum(counts_dict[c][2] for c in selected if c in counts_dict)
    cb = sum(counts_dict[c][3] for c in selected if c in counts_dict)
    if all(x > 0 for x in (ma, mb, ca, cb)):
        OR, p = fisher_exact([[ma, mb], [ca, cb]])
        return float(OR), float(p), ma, mb, ca, cb
    return float("nan"), 1.0, ma, mb, ca, cb


def pooled_or(per_cancer, head_key, cancers, pct, global_thresh=None):
    """Pooled OR across the given cancers at `pct`. If global_thresh is given,
    use it (no per-call sort). Otherwise compute threshold from this pool."""
    if global_thresh is None:
        thresh = _global_threshold(per_cancer, head_key, cancers, pct)
    else:
        thresh = global_thresh
    if not np.isfinite(thresh):
        return None
    counts = _per_cancer_counts(per_cancer, head_key, cancers, thresh)
    OR, p, ma, mb, ca, cb = _OR_from_counts_sum(counts, cancers)
    return {"OR": OR, "p": p, "threshold": thresh,
            "mut_above": ma, "mut_below": mb, "ctrl_above": ca, "ctrl_below": cb}


def permutation_reversal_test(per_cancer, head_a, head_b, pct=90,
                              exhaustive_if_possible=True, n_perm=2000, seed=SEED):
    """Null: head_a and head_b share the same GI/non-GI OR pattern.
    Statistic: (OR_GI - OR_nonGI)[head_a] - (OR_GI - OR_nonGI)[head_b].

    Optimization: the threshold per head is fixed to the GLOBAL (all cancers)
    p-th percentile of the mut+ctrl pool. Per-cancer (ma, mb, ca, cb) counts
    at that threshold are precomputed once. Each permutation then just sums
    counts for the selected GI subset and computes a Fisher OR — O(10) per perm.

    If C(n,k) <= 2000 we enumerate all splits exactly; otherwise sample n_perm.
    """
    from itertools import combinations

    available = [c for c in ALL_CANCERS if c in per_cancer]
    n_gi = min(len(GI_CANCERS), len(available) - 1)

    # Global thresholds — one per head, over ALL available cancers
    thresh_a = _global_threshold(per_cancer, head_a, available, pct)
    thresh_b = _global_threshold(per_cancer, head_b, available, pct)

    # Per-cancer counts at those thresholds
    counts_a = _per_cancer_counts(per_cancer, head_a, available, thresh_a)
    counts_b = _per_cancer_counts(per_cancer, head_b, available, thresh_b)

    def _stat(gi_subset):
        ng_subset = [c for c in available if c not in set(gi_subset)]
        a_gi, *_ = _OR_from_counts_sum(counts_a, gi_subset)
        a_ng, *_ = _OR_from_counts_sum(counts_a, ng_subset)
        b_gi, *_ = _OR_from_counts_sum(counts_b, gi_subset)
        b_ng, *_ = _OR_from_counts_sum(counts_b, ng_subset)
        da = a_gi - a_ng
        db = b_gi - b_ng
        return (da - db), da, db

    obs_diff, obs_a_delta, obs_b_delta = _stat(list(GI_CANCERS))

    # Enumerate or sample
    total_splits = 1
    for i in range(n_gi):
        total_splits = total_splits * (len(available) - i) // (i + 1)

    perm_stats = []
    if exhaustive_if_possible and total_splits <= 5000:
        splits = list(combinations(available, n_gi))
        mode = "exhaustive"
    else:
        rng = np.random.default_rng(seed)
        splits = []
        for _ in range(n_perm):
            perm = list(available)
            rng.shuffle(perm)
            splits.append(tuple(perm[:n_gi]))
        mode = "sampled"

    for sub in splits:
        d, _, _ = _stat(list(sub))
        if np.isfinite(d):
            perm_stats.append(d)
    perm_stats = np.array(perm_stats)

    if len(perm_stats) < 10:
        return {
            "percentile": pct, "mode": mode,
            "obs_diff_APOBEC1_minus_Neither": float(obs_diff),
            "obs_apobec1_GI_minus_nonGI": float(obs_a_delta),
            "obs_neither_GI_minus_nonGI": float(obs_b_delta),
            "thresh_apobec1": float(thresh_a),
            "thresh_neither": float(thresh_b),
            "p_two_sided": float("nan"),
            "n_valid_perms": int(len(perm_stats)),
        }

    p_two = float((np.abs(perm_stats) >= abs(obs_diff)).mean())
    return {
        "percentile": pct, "mode": mode,
        "thresh_apobec1": float(thresh_a),
        "thresh_neither": float(thresh_b),
        "obs_diff_APOBEC1_minus_Neither": float(obs_diff),
        "obs_apobec1_GI_minus_nonGI": float(obs_a_delta),
        "obs_neither_GI_minus_nonGI": float(obs_b_delta),
        "p_two_sided": p_two,
        "n_valid_perms": int(len(perm_stats)),
        "perm_stat_mean": float(perm_stats.mean()),
        "perm_stat_std": float(perm_stats.std()),
    }


def joint_logreg_on_gi(per_cancer):
    """Per GI cancer: logistic regression is_mutation ~ apobec1 + neither
    on mut+ctrl pooled sample. Tests whether APOBEC1 adds incremental signal
    beyond Neither.
    """
    import statsmodels.api as sm
    out = {}
    for c in GI_CANCERS:
        rec = per_cancer.get(c)
        if rec is None:
            continue
        mask = rec["mut_mask"] | rec["ctrl_mask"]
        y = rec["mut_mask"][mask].astype(int)
        a = rec["apobec1"][mask]
        n = rec["neither"][mask]
        X = sm.add_constant(np.column_stack([a, n]))
        try:
            m = sm.Logit(y, X).fit(disp=0, maxiter=100)
            params = dict(zip(["const", "apobec1", "neither"], m.params))
            ps = dict(zip(["const", "apobec1", "neither"], m.pvalues))
            out[c] = {
                "n": int(len(y)), "n_mut": int(y.sum()),
                "coef_apobec1": float(params["apobec1"]),
                "p_apobec1": float(ps["apobec1"]),
                "coef_neither": float(params["neither"]),
                "p_neither": float(ps["neither"]),
                "llf": float(m.llf),
                "pseudo_r2": float(m.prsquared),
            }
        except Exception as e:
            out[c] = {"error": str(e)}
    return out


def main():
    logger.info("=" * 70)
    logger.info("APOBEC1 head vs Neither adapter — GI vs non-GI analysis")
    logger.info("=" * 70)
    t0 = time.time()

    model, head = load_models()

    per_cancer = {}
    for c in ALL_CANCERS:
        t_c = time.time()
        rec = score_cancer(model, head, c)
        if rec is None:
            continue
        per_cancer[c] = rec
        # Free the masks/probs? No — we need them for pooled OR.
        logger.info("  %s: n=%d  mut=%d  (%.1fs)", c,
                    rec["n_mut"] + rec["n_ctrl"], rec["n_mut"], time.time() - t_c)

    if len(per_cancer) < len(ALL_CANCERS):
        missing = set(ALL_CANCERS) - set(per_cancer.keys())
        logger.warning("Missing cancers: %s", missing)

    # Per-cancer OR at each percentile
    per_cancer_ORs = []
    for c, rec in per_cancer.items():
        row = {"cancer": c, "group": "GI" if c in GI_CANCERS else "non-GI",
               "n_mut": rec["n_mut"], "n_ctrl": rec["n_ctrl"]}
        for head_key in ("apobec1", "neither"):
            for pct in PERCENTILES:
                stat = or_at_percentile(rec[head_key], rec["mut_mask"], rec["ctrl_mask"], pct)
                row[f"{head_key}_OR_p{pct}"] = stat["OR"]
                row[f"{head_key}_p_p{pct}"] = stat["p"]
        # per-cancer Spearman on mutations only
        rho, rho_p = stats.spearmanr(
            rec["apobec1"][rec["mut_mask"]], rec["neither"][rec["mut_mask"]]
        )
        row["spearman_mut_only"] = float(rho)
        row["spearman_p_mut_only"] = float(rho_p)
        per_cancer_ORs.append(row)

    df_ors = pd.DataFrame(per_cancer_ORs)
    df_ors.to_csv(OUTPUT_DIR / "gi_vs_nongi_per_cancer.csv", index=False)
    logger.info("Wrote %s", OUTPUT_DIR / "gi_vs_nongi_per_cancer.csv")

    # Pooled GI vs pooled non-GI
    pooled = {"apobec1": {}, "neither": {}}
    for head_key in ("apobec1", "neither"):
        for pct in PERCENTILES:
            pooled[head_key][f"GI_p{pct}"] = pooled_or(per_cancer, head_key, GI_CANCERS, pct)
            pooled[head_key][f"nonGI_p{pct}"] = pooled_or(per_cancer, head_key, NON_GI_CANCERS, pct)

    # Permutation test at p90 and p95
    perm_tests = {}
    for pct in (90, 95):
        logger.info("Running permutation test (p%d, n_perm=2000)...", pct)
        perm_tests[f"p{pct}"] = permutation_reversal_test(
            per_cancer, "apobec1", "neither", pct=pct, n_perm=2000
        )

    # Joint logreg on GI cancers
    logger.info("Running joint logistic regression on GI cancers...")
    joint = joint_logreg_on_gi(per_cancer)

    # Per-cancer Spearman summary
    spearman_summary = {
        "GI_mean": float(df_ors[df_ors["group"] == "GI"]["spearman_mut_only"].mean()),
        "nonGI_mean": float(df_ors[df_ors["group"] == "non-GI"]["spearman_mut_only"].mean()),
        "all_mean": float(df_ors["spearman_mut_only"].mean()),
    }

    # Summary log
    logger.info("\n" + "=" * 70)
    logger.info("POOLED OR SUMMARY")
    logger.info("=" * 70)
    for head_key in ("apobec1", "neither"):
        for pct in PERCENTILES:
            gi = pooled[head_key][f"GI_p{pct}"] or {"OR": float("nan"), "p": float("nan")}
            ng = pooled[head_key][f"nonGI_p{pct}"] or {"OR": float("nan"), "p": float("nan")}
            logger.info(
                "  %-8s p%d:  GI OR=%.3f (p=%.2e)   nonGI OR=%.3f (p=%.2e)   GI/nonGI=%.3f",
                head_key, pct, gi["OR"], gi["p"], ng["OR"], ng["p"],
                gi["OR"] / ng["OR"] if ng["OR"] and ng["OR"] > 0 else float("nan"),
            )

    logger.info("\nPermutation reversal tests:")
    for pct_key, r in perm_tests.items():
        logger.info(
            "  %s:  obs_diff=%.3f  APOBEC1 GI-nonGI=%.3f  Neither GI-nonGI=%.3f  p=%.4f",
            pct_key,
            r.get("obs_diff_APOBEC1_minus_Neither", float("nan")),
            r.get("obs_apobec1_GI_minus_nonGI", float("nan")),
            r.get("obs_neither_GI_minus_nonGI", float("nan")),
            r.get("p_two_sided", float("nan")),
        )

    logger.info("\nJoint logistic regression (is_mut ~ apobec1 + neither) on GI cancers:")
    for c, r in joint.items():
        if "error" in r:
            logger.info("  %s: ERROR %s", c, r["error"])
            continue
        logger.info(
            "  %s: coef_APOBEC1=%.3f (p=%.2e)  coef_Neither=%.3f (p=%.2e)",
            c, r["coef_apobec1"], r["p_apobec1"], r["coef_neither"], r["p_neither"],
        )

    results = {
        "per_cancer_ORs": per_cancer_ORs,
        "pooled": pooled,
        "permutation_tests": perm_tests,
        "joint_logreg_gi": joint,
        "spearman_summary": spearman_summary,
        "gi_cancers": GI_CANCERS,
        "non_gi_cancers": NON_GI_CANCERS,
        "percentiles": list(PERCENTILES),
    }
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("\nSaved results: %s", RESULTS_JSON)
    logger.info("Total: %.1f min", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
