#!/usr/bin/env python3
"""Tests 6 (random-label null) and 7 (feature ablation) on the Neither head.

Test 6 — Random-label null:
  Train 5 fresh adapters on the frozen Phase3 shared encoder, each with 206
  positives sampled at random from non-Neither Levanon categories (A3A / A3G /
  A3A_A3G / Unknown) plus 206 motif-matched negatives. Score TCGA (via cached
  PCAWG COADREAD because that file has chrom/pos) and compute per-stratum
  OR@p90. If random-label adapters produce Neither-like OR, the Neither signal
  is regime-driven.

Test 7 — Feature ablation:
  Load the existing Phase3 model + Neither adapter. At PCAWG COADREAD scoring
  time, zero out (a) the 24 motif features, (b) the 16 struct/loop features,
  (c) both. Rescore, recompute per-stratum OR@p90.

Output:
  experiments/multi_enzyme/outputs/neither_adversarial/test_6_random_labels.json
  experiments/multi_enzyme/outputs/neither_adversarial/test_7_feature_ablation.json
  experiments/multi_enzyme/outputs/neither_adversarial/run.log
"""
from __future__ import annotations

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
from pyfaidx import Fasta
from scipy.stats import fisher_exact

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.data.apobec_feature_extraction import build_hand_features  # noqa

OUT_DIR = PROJECT_ROOT / "experiments" / "multi_enzyme" / "outputs" / "neither_adversarial"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = OUT_DIR / "run.log"

sys.stdout.reconfigure(line_buffering=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(LOG_FILE, mode="w")],
)
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = PROJECT_ROOT / "data"
HG19 = DATA_DIR / "raw" / "genomes" / "hg19.fa"
ME_DIR = DATA_DIR / "processed" / "multi_enzyme"
V3_SPLITS = ME_DIR / "splits_multi_enzyme_v3_with_negatives.csv"
V3_SEQS = ME_DIR / "multi_enzyme_sequences_v3_with_negatives.json"
V3_LOOP = ME_DIR / "loop_position_per_site_v3.csv"
V3_STRUCT = ME_DIR / "structure_cache_multi_enzyme_v3.npz"
V3_RNAFM_O = ME_DIR / "embeddings" / "rnafm_pooled_v3.pt"
V3_RNAFM_E = ME_DIR / "embeddings" / "rnafm_pooled_edited_v3.pt"
PHASE3_CKPT = (
    PROJECT_ROOT / "experiments" / "multi_enzyme" / "outputs"
    / "phase3_neural_true_validation" / "phase3_neural_full.pt"
)
PCAWG_CACHE = PROJECT_ROOT / "experiments" / "multi_enzyme" / "outputs" / "pcawg" / "coadread_cache.npz"

D_INPUT = 1320
D_SHARED = 128
ENZYMES = ["A3A", "A3B", "A3G", "A3A_A3G", "Neither"]
N_ENZYMES_CLS = 6
DEVICE = torch.device("cpu")  # run on CPU to not compete with MPS users


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


class SimpleAdapter(nn.Module):
    """Same shape as Phase3Model's enzyme_adapters entries: 128 -> 32 -> 1."""
    def __init__(self, d_shared=D_SHARED):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_shared, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, shared):
        return self.net(shared).squeeze(-1)


def load_v3_features_for_site_ids(site_ids, seqs_all, loop_df, struct_map, rnafm_o_d, rnafm_e_d):
    hand = build_hand_features(site_ids, seqs_all, struct_map, loop_df)
    n = len(site_ids)
    emb_o = np.zeros((n, 640), dtype=np.float32)
    emb_d = np.zeros((n, 640), dtype=np.float32)
    for i, sid in enumerate(site_ids):
        o = rnafm_o_d.get(sid); e = rnafm_e_d.get(sid)
        if o is None or e is None:
            continue
        if hasattr(o, "numpy"): o = o.numpy()
        if hasattr(e, "numpy"): e = e.numpy()
        emb_o[i] = o; emb_d[i] = e - o
    return np.concatenate([emb_o, emb_d, hand], axis=1).astype(np.float32)


def load_v3_all():
    logger.info("Loading v3 data...")
    import json as _json
    splits = pd.read_csv(V3_SPLITS)
    with open(V3_SEQS) as f:
        seqs_all = _json.load(f)
    loop_df = pd.read_csv(V3_LOOP).drop_duplicates(subset=["site_id"]).set_index("site_id")
    sc = np.load(V3_STRUCT, allow_pickle=True)
    struct_map = {sid: sc["delta_features"][i] for i, sid in enumerate(sc["site_ids"])}
    rnafm_o = torch.load(V3_RNAFM_O, weights_only=False, map_location="cpu")
    rnafm_e = torch.load(V3_RNAFM_E, weights_only=False, map_location="cpu")
    return splits, seqs_all, loop_df, struct_map, rnafm_o, rnafm_e


def train_adapter_on_frozen_encoder(model, X, y, n_epochs=20, lr=1e-3, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    head = SimpleAdapter().to(DEVICE)
    head.train()
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    X_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y).float()
    n = len(X); BATCH = 64
    model.eval()
    with torch.no_grad():
        shared_all = model.shared_encoder(X_t.to(DEVICE)).cpu()
    for ep in range(n_epochs):
        idx = np.random.RandomState(seed + 100 + ep).permutation(n)
        for b in range(0, n, BATCH):
            bi = idx[b:b+BATCH]
            s = shared_all[bi].to(DEVICE)
            yb = y_t[bi].to(DEVICE)
            logit = head(s)
            loss = F.binary_cross_entropy_with_logits(logit, yb)
            opt.zero_grad(); loss.backward(); opt.step()
    return head


def score_adapter_on_cache(head, model, cache, batch=512):
    hand = cache["hand"]; emb_o = cache["emb_o"]; emb_d = cache["emb_d"]
    X = np.concatenate([emb_o, emb_d, hand], axis=1).astype(np.float32)
    n = len(X)
    probs = np.zeros(n, dtype=np.float32)
    model.eval(); head.eval()
    with torch.no_grad():
        for b in range(0, n, batch):
            e = min(b+batch, n)
            xb = torch.from_numpy(X[b:e]).float().to(DEVICE)
            sh = model.shared_encoder(xb)
            probs[b:e] = torch.sigmoid(head(sh)).cpu().numpy()
    return probs


def score_existing_neither(model, cache, zero_motif=False, zero_struct=False, batch=512):
    hand = cache["hand"].copy()
    if zero_motif:
        hand[:, :24] = 0.0
    if zero_struct:
        hand[:, 24:40] = 0.0
    X = np.concatenate([cache["emb_o"], cache["emb_d"], hand], axis=1).astype(np.float32)
    n = len(X); probs = np.zeros(n, dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for b in range(0, n, batch):
            e = min(b+batch, n)
            xb = torch.from_numpy(X[b:e]).float().to(DEVICE)
            sh = model.shared_encoder(xb)
            probs[b:e] = torch.sigmoid(model.enzyme_adapters["Neither"](sh).squeeze(-1)).cpu().numpy()
    return probs


def trinuc_cpg_tc(genome, chrom, pos):
    try:
        tri = str(genome[chrom][pos-1:pos+2]).upper()
        if len(tri) != 3 or tri[1] not in "CG":
            return None, None
        if tri[1] == "C":
            return tri[0] == "T", tri[2] == "G"
        else:
            return tri[2] == "A", tri[0] == "C"
    except Exception:
        return None, None


def or_at_p90(scores, m, c):
    pool = scores[m | c]
    if len(pool) < 50:
        return float("nan"), float("nan"), float("nan")
    t = float(np.percentile(pool, 90))
    ma = int((scores[m] >= t).sum()); mb = int((scores[m] < t).sum())
    ca = int((scores[c] >= t).sum()); cb = int((scores[c] < t).sum())
    if all(x > 0 for x in (ma, mb, ca, cb)):
        OR, p = fisher_exact([[ma, mb], [ca, cb]])
    else:
        OR, p = float("nan"), 1.0
    return float(OR), float(p), t


def stratified_ors(scores, chrom_arr, pos_arr, type_arr, genome):
    tc = np.zeros(len(scores), dtype=object); cpg = np.zeros(len(scores), dtype=object)
    for i in range(len(scores)):
        tc[i], cpg[i] = trinuc_cpg_tc(genome, str(chrom_arr[i]), int(pos_arr[i]))
    keep = np.array([t is not None and c is not None for t, c in zip(tc, cpg)])
    m_all = (type_arr == "mutation") & keep
    c_all = (type_arr == "control") & keep
    tc_b = np.array([bool(t) if t is not None else False for t in tc])
    cpg_b = np.array([bool(c) if c is not None else False for c in cpg])

    strata = {
        "overall": keep,
        "TC_CpG":      keep & tc_b & cpg_b,
        "TC_nonCpG":   keep & tc_b & ~cpg_b,
        "nonTC_CpG":   keep & ~tc_b & cpg_b,
        "nonTC_nonCpG":keep & ~tc_b & ~cpg_b,
    }
    out = {}
    for name, mask in strata.items():
        m = m_all & mask; c = c_all & mask
        OR, p, t = or_at_p90(scores, m, c)
        out[name] = {"n_mut": int(m.sum()), "n_ctrl": int(c.sum()), "OR_p90": OR, "p_p90": p, "thresh": t}
    return out


# ============================================================================
# Test 6 — Random-label null
# ============================================================================
def test_6(model, n_runs=5):
    logger.info("\n=== TEST 6 — Random-label null ===")
    splits, seqs_all, loop_df, struct_map, rnafm_o, rnafm_e = load_v3_all()
    # Pool of non-Neither positives from v3
    non_nei_pos = splits[(splits["enzyme"] != "Neither") & (splits["is_edited"] == 1)].copy()
    non_nei_neg = splits[(splits["enzyme"] != "Neither") & (splits["is_edited"] == 0)].copy()
    # Restrict to sites with sequence
    have_seq = set(seqs_all.keys())
    non_nei_pos = non_nei_pos[non_nei_pos["site_id"].isin(have_seq)]
    non_nei_neg = non_nei_neg[non_nei_neg["site_id"].isin(have_seq)]
    logger.info("  Non-Neither pool: %d positives, %d negatives", len(non_nei_pos), len(non_nei_neg))

    genome = Fasta(str(HG19))
    cache = np.load(PCAWG_CACHE, allow_pickle=True)

    results = {"runs": [], "Neither_reference": None}

    # Reference: the existing Neither adapter (use Phase3Model directly)
    neither_probs = score_existing_neither(model, cache)
    neither_strat = stratified_ors(neither_probs, cache["chrom"], cache["pos"], cache["type"], genome)
    results["Neither_reference"] = neither_strat
    logger.info("  Neither reference per-stratum OR@p90:")
    for k, v in neither_strat.items():
        logger.info("    %-13s n_mut=%6d n_ctrl=%6d OR=%.3f", k, v["n_mut"], v["n_ctrl"], v["OR_p90"])

    for run_i in range(n_runs):
        rng = np.random.RandomState(1000 + run_i)
        pos_idx = rng.choice(len(non_nei_pos), 206, replace=False)
        neg_idx = rng.choice(len(non_nei_neg), 206, replace=False)
        pos_sids = non_nei_pos.iloc[pos_idx]["site_id"].tolist()
        neg_sids = non_nei_neg.iloc[neg_idx]["site_id"].tolist()
        all_sids = pos_sids + neg_sids
        y = np.array([1]*len(pos_sids) + [0]*len(neg_sids), dtype=np.float32)

        X = load_v3_features_for_site_ids(all_sids, seqs_all, loop_df, struct_map, rnafm_o, rnafm_e)
        t0 = time.time()
        head = train_adapter_on_frozen_encoder(model, X, y, n_epochs=20, lr=1e-3, seed=run_i)
        probs = score_adapter_on_cache(head, model, cache)
        strat = stratified_ors(probs, cache["chrom"], cache["pos"], cache["type"], genome)
        run_secs = time.time() - t0
        run_summary = {"run": run_i, "seed": 1000+run_i, "n_pos": 206, "n_neg": 206,
                        "stratified": strat, "time_s": run_secs,
                        "pos_enzymes": non_nei_pos.iloc[pos_idx]["enzyme"].value_counts().to_dict()}
        results["runs"].append(run_summary)
        logger.info("  Run %d (seed=%d) in %.1fs  enzymes=%s",
                    run_i, 1000+run_i, run_secs, run_summary["pos_enzymes"])
        for k in ("overall","TC_CpG","TC_nonCpG","nonTC_CpG","nonTC_nonCpG"):
            logger.info("    %-13s OR=%.3f  (vs Neither %.3f)", k,
                        strat[k]["OR_p90"], neither_strat[k]["OR_p90"])

    # Summary: distribution of random-label OR per stratum
    summary = {}
    for k in ("overall","TC_CpG","TC_nonCpG","nonTC_CpG","nonTC_nonCpG"):
        vals = [r["stratified"][k]["OR_p90"] for r in results["runs"]]
        summary[k] = {
            "random_label_ORs": vals,
            "random_label_median": float(np.median(vals)),
            "random_label_mean": float(np.mean(vals)),
            "random_label_std": float(np.std(vals)),
            "Neither_OR": neither_strat[k]["OR_p90"],
        }
    results["summary"] = summary
    logger.info("\n  Summary (PCAWG COADREAD, n_runs=%d):", n_runs)
    logger.info("    %-14s Neither  |  random median  random range", " ")
    for k, v in summary.items():
        logger.info("    %-14s  %.3f     |  %.3f            [%.3f, %.3f]",
                    k, v["Neither_OR"], v["random_label_median"], min(v["random_label_ORs"]), max(v["random_label_ORs"]))
    return results


# ============================================================================
# Test 7 — Feature ablation
# ============================================================================
def test_7(model):
    logger.info("\n=== TEST 7 — Feature ablation on Neither head ===")
    genome = Fasta(str(HG19))
    cache = np.load(PCAWG_CACHE, allow_pickle=True)

    runs = {}
    for label, zm, zs in [
        ("full",          False, False),
        ("zero_motif",    True,  False),
        ("zero_struct",   False, True),
        ("zero_all_hand", True,  True),
    ]:
        logger.info("  %s ...", label)
        probs = score_existing_neither(model, cache, zero_motif=zm, zero_struct=zs)
        strat = stratified_ors(probs, cache["chrom"], cache["pos"], cache["type"], genome)
        runs[label] = strat
        for k, v in strat.items():
            logger.info("    %-13s OR=%.3f", k, v["OR_p90"])

    # Deltas
    logger.info("\n  OR deltas vs full:")
    delta = {}
    for k in ("overall","TC_CpG","TC_nonCpG","nonTC_CpG","nonTC_nonCpG"):
        full = runs["full"][k]["OR_p90"]
        delta[k] = {
            "full": full,
            "zero_motif_delta": runs["zero_motif"][k]["OR_p90"] - full,
            "zero_struct_delta": runs["zero_struct"][k]["OR_p90"] - full,
            "zero_all_hand_delta": runs["zero_all_hand"][k]["OR_p90"] - full,
        }
        logger.info("    %-13s full=%.3f  Δ(zero_motif)=%+.3f  Δ(zero_struct)=%+.3f  Δ(zero_all)=%+.3f",
                    k, full, delta[k]["zero_motif_delta"],
                    delta[k]["zero_struct_delta"], delta[k]["zero_all_hand_delta"])
    return {"runs": runs, "delta_vs_full": delta}


# ============================================================================
# Main
# ============================================================================
def main():
    logger.info("Loading Phase3Model...")
    ckpt = torch.load(PHASE3_CKPT, weights_only=False, map_location="cpu")
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model = Phase3Model()
    model.load_state_dict(state, strict=False)
    model.to(DEVICE)

    t_start = time.time()
    t6 = test_6(model, n_runs=5)
    with (OUT_DIR / "test_6_random_labels.json").open("w") as f:
        json.dump(t6, f, indent=2, default=str)
    logger.info("Saved Test 6 results")

    t7 = test_7(model)
    with (OUT_DIR / "test_7_feature_ablation.json").open("w") as f:
        json.dump(t7, f, indent=2, default=str)
    logger.info("Saved Test 7 results")

    logger.info("Total: %.1f min", (time.time()-t_start)/60)


if __name__ == "__main__":
    main()
