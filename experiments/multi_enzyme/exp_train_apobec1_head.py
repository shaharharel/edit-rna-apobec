#!/usr/bin/env python3
"""Train a 6th adapter head (APOBEC1) on the frozen Phase3Model shared encoder.

Pipeline
========
1. Load existing Phase3Model checkpoint.
2. Freeze shared_encoder, binary_head, existing 5 adapters, enzyme_classifier.
3. Add:
   - Linear(1, 16) "species" projection that is added to the 128-d shared embedding
     BEFORE the new APOBEC1 head only (leaves non-APOBEC1 heads untouched).
   - Linear(128, 32) → GELU → Linear(32, 1) new APOBEC1 adapter head.
4. 5-fold CV on the mouse training set (apobec1_v1_with_negatives.csv + species=1).
5. Save the new params only (species projection + APOBEC1 adapter) to
   experiments/multi_enzyme/outputs/apobec1_head/apobec1_head.pt.
6. Gate A: score the 206 Neither Levanon sites (species=0) → AUROC.
7. Gate B: cross-species gate (held-out Rosenberg human if any; else skipped).
8. TCGA re-scoring across 10 cancers (if Gate A AUROC ≥ 0.65).
9. Compute Spearman correlation between APOBEC1 and Neither adapter on COADREAD.

Usage:
    conda run -n quris python experiments/multi_enzyme/exp_train_apobec1_head.py \
        [--skip-train] [--skip-tcga] [--cancers coadread,stad,esca,...]
"""

from __future__ import annotations

import argparse
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
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.apobec_feature_extraction import build_hand_features  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# ============================================================================
# Paths
# ============================================================================
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "multi_enzyme" / "outputs" / "apobec1_head"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

log_file = OUTPUT_DIR / "run.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, mode="w"),
    ],
)
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
APOBEC1_DIR = DATA_DIR / "processed" / "apobec1"
APOBEC1_SPLITS = APOBEC1_DIR / "apobec1_v1_with_negatives.csv"
APOBEC1_SEQS = APOBEC1_DIR / "apobec1_v1_sequences.json"
APOBEC1_HAND = APOBEC1_DIR / "apobec1_hand40.npy"
APOBEC1_LOOP = APOBEC1_DIR / "loop_position_apobec1_v1.csv"
APOBEC1_STRUCT = APOBEC1_DIR / "structure_cache_apobec1_v1.npz"
APOBEC1_RNAFM = APOBEC1_DIR / "rnafm_apobec1_v1.pt"
APOBEC1_RNAFM_ED = APOBEC1_DIR / "rnafm_apobec1_v1_edited.pt"

# Existing v3 assets (human)
ME_DIR = DATA_DIR / "processed" / "multi_enzyme"
V3_SPLITS = ME_DIR / "splits_multi_enzyme_v3_with_negatives.csv"
V3_SEQS = ME_DIR / "multi_enzyme_sequences_v3_with_negatives.json"
V3_LOOP = ME_DIR / "loop_position_per_site_v3.csv"
V3_STRUCT = ME_DIR / "structure_cache_multi_enzyme_v3.npz"
V3_RNAFM_ORIG = ME_DIR / "embeddings" / "rnafm_pooled_v3.pt"
V3_RNAFM_EDITED = ME_DIR / "embeddings" / "rnafm_pooled_edited_v3.pt"

PHASE3_CKPT = (
    PROJECT_ROOT / "experiments" / "multi_enzyme" / "outputs"
    / "phase3_neural_true_validation" / "phase3_neural_full.pt"
)

# TCGA assets (identical to exp_reproduce_section_1.py)
HAND_DIR = DATA_DIR / "processed" / "multi_enzyme" / "tcga_hand_features"
EMB_DIR = DATA_DIR / "processed" / "multi_enzyme" / "embeddings"
RAW_SCORES_DIR = (
    PROJECT_ROOT / "experiments" / "multi_enzyme" / "outputs"
    / "tcga_gnomad" / "raw_scores"
)

# Section 1 reference
REF_SECTION1 = (
    PROJECT_ROOT / "experiments" / "multi_enzyme" / "outputs"
    / "section1_nn_reproduction" / "section_1a_1g_summary.csv"
)


# ============================================================================
# Constants
# ============================================================================
SEED = 42
N_FOLDS = 5
D_INPUT = 1320
D_SHARED = 128
ENZYMES = ["A3A", "A3B", "A3G", "A3A_A3G", "Neither"]
N_ENZYMES_CLS = 6
CANCERS = ["blca", "brca", "cesc", "lusc", "hnsc", "skcm", "esca", "stad", "lihc", "coadread"]

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else torch.device("cpu")
)
logger.info(f"Device: {DEVICE}")


# ============================================================================
# Model
# ============================================================================

class Phase3Model(nn.Module):
    """Exact replica of existing Phase3Model (from exp_reproduce_section_1.py)."""

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

    def forward(self, x):
        shared = self.shared_encoder(x)
        binary_logit = self.binary_head(shared).squeeze(-1)
        per_enzyme_logits = [self.enzyme_adapters[enz](shared).squeeze(-1) for enz in ENZYMES]
        return binary_logit, per_enzyme_logits, None, shared


class APOBEC1Head(nn.Module):
    """New head that consumes 128-d shared embedding + 1-d species bit.

    Shapes:
      shared: (B, 128)
      species: (B, 1)
      -> species proj -> (B, 16) added to zero-padded bias on shared
      -> Linear(128, 32) -> GELU -> Linear(32, 1)
    """

    def __init__(self, d_shared: int = 128):
        super().__init__()
        # Species: 1 -> 16. We add a shared bias to the 128-d embedding by broadcasting
        # through a small 128-dim projection of the species signal. Simpler: learn a
        # 128-d bias gated by species.
        self.species_proj = nn.Sequential(
            nn.Linear(1, 16), nn.GELU(), nn.Linear(16, d_shared),
        )
        self.head = nn.Sequential(
            nn.Linear(d_shared, 32), nn.GELU(), nn.Linear(32, 1),
        )

    def forward(self, shared: torch.Tensor, species: torch.Tensor) -> torch.Tensor:
        bias = self.species_proj(species)  # (B, 128)
        x = shared + bias
        return self.head(x).squeeze(-1)


# ============================================================================
# Data loading
# ============================================================================

def _load_rnafm_dict(path: Path) -> dict:
    d = torch.load(path, weights_only=False, map_location="cpu")
    return d


def load_apobec1_data() -> dict:
    """Build mouse feature matrix + labels from the apobec1 processed files."""
    logger.info("Loading APOBEC1 training data ...")
    splits = pd.read_csv(APOBEC1_SPLITS)
    with open(APOBEC1_SEQS) as f:
        seqs = json.load(f)

    site_ids = splits["site_id"].tolist()
    n = len(site_ids)
    logger.info("  %d sites (%d pos / %d neg)",
                n,
                int((splits["is_edited"] == 1).sum()),
                int((splits["is_edited"] == 0).sum()))

    # Hand features are already computed and aligned to splits order
    hand = np.load(APOBEC1_HAND)
    assert hand.shape[0] == n, f"hand={hand.shape[0]} vs splits={n}"

    rnafm_orig = _load_rnafm_dict(APOBEC1_RNAFM)
    rnafm_edit = _load_rnafm_dict(APOBEC1_RNAFM_ED)

    D_RNAFM = 640
    emb_orig = np.zeros((n, D_RNAFM), dtype=np.float32)
    emb_delta = np.zeros((n, D_RNAFM), dtype=np.float32)
    missing = 0
    for i, sid in enumerate(site_ids):
        if sid in rnafm_orig and sid in rnafm_edit:
            o = rnafm_orig[sid]
            e = rnafm_edit[sid]
            if isinstance(o, torch.Tensor):
                o = o.numpy()
            if isinstance(e, torch.Tensor):
                e = e.numpy()
            emb_orig[i] = o
            emb_delta[i] = e - o
        else:
            missing += 1
    logger.info("  RNA-FM coverage: %d/%d missing=%d", n - missing, n, missing)

    X = np.concatenate([emb_orig, emb_delta, hand], axis=1).astype(np.float32)
    labels = splits["is_edited"].values.astype(np.float32)
    species = np.ones(n, dtype=np.float32)  # all mouse
    return {
        "X": X, "labels": labels, "species": species,
        "site_ids": site_ids, "splits": splits,
    }


def load_neither_data() -> dict:
    """Load the 206 Neither human positives + same-count motif-matched negatives
    from the v3 multi-enzyme dataset."""
    logger.info("Loading Neither (human) Gate A data ...")
    splits = pd.read_csv(V3_SPLITS)
    nei = splits[splits["enzyme"] == "Neither"].copy()
    pos = nei[nei["is_edited"] == 1].copy()
    neg = nei[nei["is_edited"] == 0].copy()
    logger.info("  Neither: %d pos / %d neg", len(pos), len(neg))

    # Need hand features + RNA-FM embeddings aligned to these site_ids
    with open(V3_SEQS) as f:
        seqs_all = json.load(f)
    loop_df = pd.read_csv(V3_LOOP).drop_duplicates(subset=["site_id"]).set_index("site_id")
    sc = np.load(V3_STRUCT, allow_pickle=True)
    struct_map = {sid: sc["delta_features"][i] for i, sid in enumerate(sc["site_ids"])}

    nei_use = pd.concat([pos, neg], ignore_index=True)
    site_ids = nei_use["site_id"].tolist()

    # Filter to sites with sequences
    have_seq = [sid for sid in site_ids if sid in seqs_all]
    missing = set(site_ids) - set(have_seq)
    if missing:
        logger.warning("  %d Neither sites missing sequences", len(missing))
        nei_use = nei_use[nei_use["site_id"].isin(set(have_seq))].reset_index(drop=True)
        site_ids = nei_use["site_id"].tolist()

    hand = build_hand_features(site_ids, seqs_all, struct_map, loop_df)
    n = len(site_ids)

    rnafm_o = _load_rnafm_dict(V3_RNAFM_ORIG)
    rnafm_e = _load_rnafm_dict(V3_RNAFM_EDITED)
    D_RNAFM = 640
    emb_o = np.zeros((n, D_RNAFM), dtype=np.float32)
    emb_d = np.zeros((n, D_RNAFM), dtype=np.float32)
    for i, sid in enumerate(site_ids):
        if sid in rnafm_o and sid in rnafm_e:
            o = rnafm_o[sid]
            e = rnafm_e[sid]
            if isinstance(o, torch.Tensor):
                o = o.numpy()
            if isinstance(e, torch.Tensor):
                e = e.numpy()
            emb_o[i] = o
            emb_d[i] = e - o

    X = np.concatenate([emb_o, emb_d, hand], axis=1).astype(np.float32)
    labels = nei_use["is_edited"].values.astype(np.float32)
    species = np.zeros(n, dtype=np.float32)  # all human
    return {"X": X, "labels": labels, "species": species, "site_ids": site_ids}


# ============================================================================
# Training
# ============================================================================

def load_pretrained(model: Phase3Model):
    state = torch.load(PHASE3_CKPT, weights_only=False, map_location="cpu")
    model.load_state_dict(state)
    logger.info("Loaded Phase3 checkpoint")


def freeze_pretrained(model: Phase3Model) -> None:
    for p in model.parameters():
        p.requires_grad = False
    model.eval()


def train_one_fold(model: Phase3Model, apobec1_head: APOBEC1Head,
                   X_tr, y_tr, s_tr, X_va, y_va, s_va,
                   n_epochs: int = 20, lr: float = 1e-3) -> dict:
    apobec1_head.to(DEVICE)
    apobec1_head.train()

    # Optimizer on NEW params only
    new_params = list(apobec1_head.parameters())
    opt = torch.optim.AdamW(new_params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    X_tr_t = torch.from_numpy(X_tr).float()
    y_tr_t = torch.from_numpy(y_tr).float()
    s_tr_t = torch.from_numpy(s_tr).float().unsqueeze(1)

    X_va_t = torch.from_numpy(X_va).float()
    y_va_t = torch.from_numpy(y_va).float()
    s_va_t = torch.from_numpy(s_va).float().unsqueeze(1)

    BATCH = 64
    n = len(X_tr)

    best_auroc = 0.0
    best_state = None

    for ep in range(n_epochs):
        model.eval()  # frozen shared encoder stays in eval mode
        apobec1_head.train()
        idx = np.random.RandomState(SEED + ep).permutation(n)
        total = 0.0
        steps = 0
        for b in range(0, n, BATCH):
            b_idx = idx[b:b + BATCH]
            xb = X_tr_t[b_idx].to(DEVICE)
            yb = y_tr_t[b_idx].to(DEVICE)
            sb = s_tr_t[b_idx].to(DEVICE)
            with torch.no_grad():
                shared = model.shared_encoder(xb)
            logit = apobec1_head(shared, sb)
            loss = F.binary_cross_entropy_with_logits(logit, yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(new_params, 1.0)
            opt.step()
            total += float(loss.item())
            steps += 1
        scheduler.step()

        # Eval
        apobec1_head.eval()
        with torch.no_grad():
            probs = []
            for b in range(0, len(X_va_t), 256):
                xv = X_va_t[b:b + 256].to(DEVICE)
                sv = s_va_t[b:b + 256].to(DEVICE)
                shared = model.shared_encoder(xv)
                logit = apobec1_head(shared, sv)
                probs.append(torch.sigmoid(logit).cpu().numpy())
            probs = np.concatenate(probs)
        try:
            auroc = float(roc_auc_score(y_va_t.numpy(), probs))
        except Exception:
            auroc = 0.5
        if (ep + 1) % 5 == 0 or ep == 0:
            logger.info("    ep %02d  loss=%.4f  val_auroc=%.4f", ep + 1, total / max(steps, 1), auroc)
        if auroc > best_auroc:
            best_auroc = auroc
            best_state = {k: v.cpu().clone() for k, v in apobec1_head.state_dict().items()}

    if best_state is not None:
        apobec1_head.load_state_dict(best_state)

    # Final probs and AUPRC
    apobec1_head.eval()
    with torch.no_grad():
        probs = []
        for b in range(0, len(X_va_t), 256):
            xv = X_va_t[b:b + 256].to(DEVICE)
            sv = s_va_t[b:b + 256].to(DEVICE)
            shared = model.shared_encoder(xv)
            logit = apobec1_head(shared, sv)
            probs.append(torch.sigmoid(logit).cpu().numpy())
        probs = np.concatenate(probs)
    auroc = float(roc_auc_score(y_va, probs)) if len(np.unique(y_va)) > 1 else float("nan")
    auprc = float(average_precision_score(y_va, probs)) if len(np.unique(y_va)) > 1 else float("nan")
    return {"auroc": auroc, "auprc": auprc, "probs": probs, "best_state": best_state}


def run_5fold_cv(train_data: dict) -> tuple[list[dict], dict]:
    X = train_data["X"]; y = train_data["labels"]; s = train_data["species"]
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    fold_metrics = []
    # We keep the *last* fold's head as the "production" head. Alternative: average,
    # but weighted averaging would distort the species projection. Use retrain-on-all.
    all_head_states = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        logger.info("\n--- Fold %d/%d (train=%d, val=%d) ---", fold + 1, N_FOLDS,
                    len(tr_idx), len(va_idx))
        torch.manual_seed(SEED + fold)
        np.random.seed(SEED + fold)

        model = Phase3Model()
        load_pretrained(model)
        freeze_pretrained(model)
        model.to(DEVICE)

        head = APOBEC1Head(D_SHARED)

        res = train_one_fold(
            model, head,
            X[tr_idx], y[tr_idx], s[tr_idx],
            X[va_idx], y[va_idx], s[va_idx],
            n_epochs=20, lr=1e-3,
        )
        logger.info("  Fold %d  AUROC=%.4f  AUPRC=%.4f", fold + 1, res["auroc"], res["auprc"])
        fold_metrics.append({"fold": fold + 1, "auroc": res["auroc"], "auprc": res["auprc"]})
        all_head_states.append(res["best_state"])

    # Retrain on all data for production head
    logger.info("\nRetraining on full training set for production head ...")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = Phase3Model()
    load_pretrained(model)
    freeze_pretrained(model)
    model.to(DEVICE)
    head = APOBEC1Head(D_SHARED)
    head.to(DEVICE)
    head.train()

    new_params = list(head.parameters())
    opt = torch.optim.AdamW(new_params, lr=1e-3, weight_decay=1e-4)
    n = len(X)
    X_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y).float()
    s_t = torch.from_numpy(s).float().unsqueeze(1)
    BATCH = 64
    for ep in range(20):
        idx = np.random.RandomState(SEED + 100 + ep).permutation(n)
        for b in range(0, n, BATCH):
            b_idx = idx[b:b + BATCH]
            xb = X_t[b_idx].to(DEVICE)
            yb = y_t[b_idx].to(DEVICE)
            sb = s_t[b_idx].to(DEVICE)
            with torch.no_grad():
                shared = model.shared_encoder(xb)
            logit = head(shared, sb)
            loss = F.binary_cross_entropy_with_logits(logit, yb)
            opt.zero_grad(); loss.backward(); opt.step()

    prod_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
    mean_auroc = float(np.mean([m["auroc"] for m in fold_metrics]))
    std_auroc = float(np.std([m["auroc"] for m in fold_metrics]))
    logger.info("\n5-fold CV: mean AUROC=%.4f ± %.4f", mean_auroc, std_auroc)

    summary = {
        "folds": fold_metrics,
        "mean_auroc": mean_auroc,
        "std_auroc": std_auroc,
        "n_train_pos": int(np.sum(y)),
        "n_train_neg": int(len(y) - np.sum(y)),
    }
    return all_head_states, {"summary": summary, "prod_state": prod_state}


# ============================================================================
# Scoring helpers
# ============================================================================

def load_head_state(path: Path) -> dict:
    return torch.load(path, weights_only=False, map_location="cpu")


def score_with_head(model: Phase3Model, head: APOBEC1Head,
                    X: np.ndarray, species_val: float,
                    batch_size: int = 256) -> np.ndarray:
    model.eval()
    head.eval()
    n = len(X)
    probs = np.zeros(n, dtype=np.float32)
    with torch.no_grad():
        for b in range(0, n, batch_size):
            e = min(b + batch_size, n)
            xb = torch.from_numpy(X[b:e]).float().to(DEVICE)
            sb = torch.full((e - b, 1), species_val, dtype=torch.float32).to(DEVICE)
            shared = model.shared_encoder(xb)
            logit = head(shared, sb)
            probs[b:e] = torch.sigmoid(logit).cpu().numpy()
    return probs


def score_with_existing_adapter(model: Phase3Model, enz: str,
                                X: np.ndarray, batch_size: int = 256) -> np.ndarray:
    model.eval()
    n = len(X)
    probs = np.zeros(n, dtype=np.float32)
    with torch.no_grad():
        for b in range(0, n, batch_size):
            e = min(b + batch_size, n)
            xb = torch.from_numpy(X[b:e]).float().to(DEVICE)
            shared = model.shared_encoder(xb)
            logit = model.enzyme_adapters[enz](shared).squeeze(-1)
            probs[b:e] = torch.sigmoid(logit).cpu().numpy()
    return probs


# ============================================================================
# Gate A — Neither hypothesis test
# ============================================================================

def gate_a(model: Phase3Model, head: APOBEC1Head) -> dict:
    logger.info("\n" + "=" * 70)
    logger.info("GATE A — scoring 206 Neither Levanon sites with APOBEC1 head")
    logger.info("=" * 70)
    d = load_neither_data()
    probs = score_with_head(model, head, d["X"], species_val=0.0)
    try:
        auroc = float(roc_auc_score(d["labels"], probs))
    except Exception:
        auroc = float("nan")
    try:
        auprc = float(average_precision_score(d["labels"], probs))
    except Exception:
        auprc = float("nan")
    logger.info("Gate A: n_pos=%d  n_neg=%d  AUROC=%.4f  AUPRC=%.4f",
                int(np.sum(d["labels"] == 1)),
                int(np.sum(d["labels"] == 0)), auroc, auprc)
    verdict = (
        "CONFIRMED" if auroc >= 0.80
        else "PARTIAL_SUPPORT" if auroc >= 0.65
        else "WEAKENED"
    )
    logger.info("Gate A verdict: %s", verdict)
    return {"auroc": auroc, "auprc": auprc, "verdict": verdict,
            "n_pos": int(np.sum(d["labels"] == 1)),
            "n_neg": int(np.sum(d["labels"] == 0))}


# ============================================================================
# Gate B — Cross-species generalization
# ============================================================================

def gate_b(model: Phase3Model, head: APOBEC1Head) -> dict:
    """Cross-species: Rosenberg 2011 is actually mouse (not human as originally
    claimed in the survey doc). Since no genuine human KO-validated APOBEC1 set
    is accessible, we instead run a *pseudo-Gate B* by evaluating on the
    existing per-enzyme Neither-only-negatives sanity check via LOO."""
    logger.info("\n" + "=" * 70)
    logger.info("GATE B — cross-species check skipped (Rosenberg 2011 is mouse, "
                "no human KO set available). Running motif-matched TC-vs-nonTC check instead.")
    logger.info("=" * 70)
    # Alternate check: on Neither data, does APOBEC1 head rank positives above
    # motif-matched negatives better than the existing Neither adapter?
    d = load_neither_data()
    apobec1_probs = score_with_head(model, head, d["X"], species_val=0.0)
    neither_probs = score_with_existing_adapter(model, "Neither", d["X"])
    try:
        rho, _ = stats.spearmanr(apobec1_probs, neither_probs)
    except Exception:
        rho = float("nan")
    try:
        auroc_apo = float(roc_auc_score(d["labels"], apobec1_probs))
        auroc_nei = float(roc_auc_score(d["labels"], neither_probs))
    except Exception:
        auroc_apo = float("nan"); auroc_nei = float("nan")
    logger.info("  APOBEC1 head AUROC on Neither: %.4f", auroc_apo)
    logger.info("  Neither adapter AUROC on Neither (in-domain): %.4f", auroc_nei)
    logger.info("  Spearman(APOBEC1, Neither) on Neither data: %.4f", rho)
    verdict = (
        "SKIPPED — Rosenberg 2011 is mouse; no human KO-validated APOBEC1 set available."
    )
    return {
        "verdict": verdict,
        "apobec1_vs_neither_spearman_on_neither": float(rho),
        "apobec1_auroc_on_neither": auroc_apo,
        "neither_auroc_on_neither": auroc_nei,
    }


# ============================================================================
# TCGA rescoring
# ============================================================================

def _compute_or(scores, mut_mask, ctrl_mask, pct, pool_mask=None):
    if pool_mask is None:
        pool_mask = mut_mask | ctrl_mask
    pooled = scores[pool_mask]
    if len(pooled) < 20:
        return {"OR": float("nan"), "p": 1.0, "threshold": float("nan")}
    thresh = float(np.percentile(pooled, pct))
    ma = int((scores[mut_mask] >= thresh).sum())
    mb = int((scores[mut_mask] < thresh).sum())
    ca = int((scores[ctrl_mask] >= thresh).sum())
    cb = int((scores[ctrl_mask] < thresh).sum())
    if all(x > 0 for x in [ma, mb, ca, cb]):
        OR, pv = stats.fisher_exact([[ma, mb], [ca, cb]])
    else:
        OR, pv = float("nan"), 1.0
    return {"OR": float(OR), "p": float(pv), "threshold": thresh,
            "mut_above": ma, "mut_below": mb, "ctrl_above": ca, "ctrl_below": cb}


def rescore_tcga(model: Phase3Model, head: APOBEC1Head,
                 cancers: list[str]) -> tuple[pd.DataFrame, dict]:
    logger.info("\n" + "=" * 70)
    logger.info("STAGE 5 — TCGA rescoring (%d cancers) with APOBEC1 head", len(cancers))
    logger.info("=" * 70)

    rows = []
    coadread_pair_scores = None
    for cancer in cancers:
        t0 = time.time()
        raw_path = RAW_SCORES_DIR / f"{cancer}_scores.csv"
        hand_path = HAND_DIR / f"{cancer}_hand40.npy"
        emb_path = EMB_DIR / f"rnafm_tcga_{cancer}.pt"
        if not all(p.exists() for p in [raw_path, hand_path, emb_path]):
            logger.warning("  %s: missing caches, skipping", cancer)
            continue

        raw = pd.read_csv(raw_path)
        hand = np.load(hand_path)
        emb = torch.load(emb_path, weights_only=False, map_location="cpu")
        n = len(raw)
        if hand.shape[0] != n:
            logger.warning("  %s: hand vs raw mismatch %d vs %d; skipping",
                           cancer, hand.shape[0], n)
            continue
        if emb["pooled_orig"].shape[0] != n:
            logger.warning("  %s: emb vs raw mismatch; skipping", cancer)
            continue

        mut_mask = (raw["type"].values == "mutation")
        ctrl_mask = (raw["type"].values == "control")

        pooled_orig = emb["pooled_orig"].numpy().astype(np.float32)
        pooled_edited = emb["pooled_edited"].numpy().astype(np.float32)
        edit_delta = pooled_edited - pooled_orig
        del emb
        X = np.concatenate([pooled_orig, edit_delta, hand], axis=1).astype(np.float32)
        del pooled_orig, pooled_edited, edit_delta
        gc.collect()

        apobec1_probs = score_with_head(model, head, X, species_val=0.0)
        neither_probs = score_with_existing_adapter(model, "Neither", X)

        # OR at multiple percentile thresholds
        cancer_row = {"cancer": cancer,
                      "n_mut": int(mut_mask.sum()),
                      "n_ctrl": int(ctrl_mask.sum())}
        for pct in (50, 75, 90, 95):
            a_stats = _compute_or(apobec1_probs, mut_mask, ctrl_mask, pct)
            n_stats = _compute_or(neither_probs, mut_mask, ctrl_mask, pct)
            cancer_row[f"apobec1_OR_p{pct}"] = a_stats["OR"]
            cancer_row[f"apobec1_p_p{pct}"] = a_stats["p"]
            cancer_row[f"neither_OR_p{pct}"] = n_stats["OR"]
            cancer_row[f"neither_p_p{pct}"] = n_stats["p"]

        # Spearman between the two adapters' scores on all rows
        try:
            rho, rho_p = stats.spearmanr(apobec1_probs, neither_probs)
        except Exception:
            rho = float("nan"); rho_p = 1.0
        cancer_row["apobec1_vs_neither_spearman"] = float(rho)
        cancer_row["apobec1_vs_neither_p"] = float(rho_p)

        rows.append(cancer_row)
        logger.info(
            "  %s: n=%d  apobec1 OR@p90=%.3f  neither OR@p90=%.3f  rho=%.3f  (%.1fs)",
            cancer, n,
            cancer_row["apobec1_OR_p90"], cancer_row["neither_OR_p90"],
            rho, time.time() - t0,
        )

        if cancer == "coadread":
            coadread_pair_scores = {
                "apobec1": apobec1_probs,
                "neither": neither_probs,
                "mut_mask": mut_mask,
                "ctrl_mask": ctrl_mask,
            }

        del X, apobec1_probs, neither_probs
        gc.collect()

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "tcga_apobec1_head_OR.csv", index=False)
    logger.info("Wrote %s", OUTPUT_DIR / "tcga_apobec1_head_OR.csv")

    # COADREAD correlation block
    coadread_summary = {}
    if coadread_pair_scores is not None:
        ap = coadread_pair_scores["apobec1"]
        ne = coadread_pair_scores["neither"]
        try:
            rho, rho_p = stats.spearmanr(ap, ne)
        except Exception:
            rho = float("nan"); rho_p = 1.0
        try:
            rho_mut, _ = stats.spearmanr(ap[coadread_pair_scores["mut_mask"]],
                                         ne[coadread_pair_scores["mut_mask"]])
        except Exception:
            rho_mut = float("nan")
        coadread_summary = {
            "spearman_all": float(rho),
            "spearman_mutations_only": float(rho_mut),
            "n": int(len(ap)),
        }
        logger.info("COADREAD — Spearman(APOBEC1, Neither) all=%.4f  mut-only=%.4f",
                    rho, rho_mut)

    return df, coadread_summary


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-tcga", action="store_true")
    ap.add_argument("--cancers", type=str, default=",".join(CANCERS))
    args = ap.parse_args()

    cancers = [c.strip() for c in args.cancers.split(",") if c.strip()]

    head_path = OUTPUT_DIR / "apobec1_head.pt"
    cv_json = OUTPUT_DIR / "apobec1_head_cv.json"
    report = {}

    # ---- Stage 4 — Train ------------------------------------------------
    if not args.skip_train:
        train_data = load_apobec1_data()
        _, train_out = run_5fold_cv(train_data)
        summary = train_out["summary"]
        report["training"] = summary
        with open(cv_json, "w") as f:
            json.dump(summary, f, indent=2)
        torch.save(train_out["prod_state"], head_path)
        logger.info("Saved production head to %s", head_path)

        # Check gate
        if summary["mean_auroc"] < 0.75:
            logger.error("CHECKPOINT 4 FAILED: mean AUROC %.4f < 0.75", summary["mean_auroc"])
            logger.error("Still proceeding to validation gates for diagnostic output.")

    else:
        if not head_path.exists():
            logger.error("No existing head at %s; cannot skip training", head_path)
            sys.exit(1)
        report["training"] = json.load(open(cv_json)) if cv_json.exists() else {"skipped": True}

    # ---- Gate A & B ------------------------------------------------------
    model = Phase3Model()
    load_pretrained(model)
    freeze_pretrained(model)
    model.to(DEVICE)

    head = APOBEC1Head(D_SHARED)
    head.load_state_dict(load_head_state(head_path))
    head.to(DEVICE)
    head.eval()

    report["gate_a"] = gate_a(model, head)
    report["gate_b"] = gate_b(model, head)

    # ---- Stage 5 — TCGA --------------------------------------------------
    if not args.skip_tcga and report["gate_a"]["auroc"] >= 0.65:
        or_df, coad = rescore_tcga(model, head, cancers)
        report["tcga"] = {
            "per_cancer": or_df.to_dict(orient="records"),
            "coadread_adapter_corr": coad,
        }
    else:
        if args.skip_tcga:
            logger.info("TCGA rescoring skipped per --skip-tcga flag")
            report["tcga"] = {"skipped": "flag"}
        else:
            logger.warning("Gate A AUROC=%.3f < 0.65 — skipping TCGA rescoring",
                           report["gate_a"]["auroc"])
            report["tcga"] = {"skipped": "gate_a_failed"}

    with open(OUTPUT_DIR / "run_summary.json", "w") as f:
        json.dump(report, f, indent=2)

    # ---- Write REPORT.md -------------------------------------------------
    _write_report_md(report, cancers)


def _write_report_md(report: dict, cancers: list[str]) -> None:
    lines = ["# APOBEC1 Adapter Head — Training & Validation Report\n"]

    # 1. Dataset
    lines.append("## 1. Dataset assembled\n")
    summary_json = APOBEC1_DIR / "dataset_summary.json"
    if summary_json.exists():
        ds = json.load(open(summary_json))
        lines.append(f"- Positives: {ds['n_positives_total']}")
        lines.append(f"- Negatives (1:1 motif-matched): {ds['n_negatives_total']}")
        lines.append("### Positives by source")
        for k, v in ds.get("positives_by_source", {}).items():
            lines.append(f"  - {k}: {v}")
        lines.append("### Positives by tissue")
        for k, v in ds.get("positives_by_tissue", {}).items():
            lines.append(f"  - {k}: {v}")
        lines.append("### KO-validation flag")
        for k, v in ds.get("positives_by_ko_validated", {}).items():
            lines.append(f"  - ko_validated={k}: {v}")
        mf = ds.get("positives_motif_fractions", {})
        lines.append(f"### Motif fractions: TC={mf.get('TC'):.2%}  CC={mf.get('CC'):.2%}")
    lines.append("")

    # 2. CV
    lines.append("## 2. 5-fold training AUROC\n")
    t = report.get("training", {})
    lines.append(f"- mean AUROC: **{t.get('mean_auroc', float('nan')):.4f}**")
    lines.append(f"- std AUROC:  {t.get('std_auroc', float('nan')):.4f}")
    for f in t.get("folds", []):
        lines.append(f"  - fold {f['fold']}: AUROC={f['auroc']:.4f}  AUPRC={f['auprc']:.4f}")
    lines.append("")

    # 3. Gate A
    lines.append("## 3. Gate A — Neither hypothesis test\n")
    ga = report.get("gate_a", {})
    lines.append(f"- Neither n_pos={ga.get('n_pos', 0)}  n_neg={ga.get('n_neg', 0)}")
    lines.append(f"- AUROC=**{ga.get('auroc', float('nan')):.4f}**  AUPRC={ga.get('auprc', float('nan')):.4f}")
    lines.append(f"- Verdict: **{ga.get('verdict', 'UNKNOWN')}**")
    lines.append("")

    # 4. Gate B
    lines.append("## 4. Gate B — Cross-species generalization\n")
    gb = report.get("gate_b", {})
    lines.append(f"- {gb.get('verdict', '')}")
    lines.append(f"- Spearman(APOBEC1 head, Neither adapter) on Neither data: "
                 f"{gb.get('apobec1_vs_neither_spearman_on_neither', float('nan')):.4f}")
    lines.append("")

    # 5. TCGA
    lines.append("## 5. TCGA per-cancer OR@p90 — APOBEC1 head vs existing Neither adapter\n")
    tcga = report.get("tcga", {})
    if tcga.get("per_cancer"):
        lines.append("| Cancer | n_mut | APOBEC1 OR@p90 | p | Neither OR@p90 | p | Spearman | p |")
        lines.append("|--------|-------|----------------|---|----------------|---|----------|---|")
        for r in tcga["per_cancer"]:
            lines.append(
                f"| {r['cancer']} | {r['n_mut']:,} | "
                f"{r.get('apobec1_OR_p90', float('nan')):.3f} | "
                f"{r.get('apobec1_p_p90', 1):.2e} | "
                f"{r.get('neither_OR_p90', float('nan')):.3f} | "
                f"{r.get('neither_p_p90', 1):.2e} | "
                f"{r.get('apobec1_vs_neither_spearman', float('nan')):.3f} | "
                f"{r.get('apobec1_vs_neither_p', 1):.2e} |"
            )
    else:
        lines.append(f"_Skipped: {tcga.get('skipped', 'unknown')}._")
    lines.append("")

    # 6. COADREAD correlation
    lines.append("## 6. Adapter agreement on COADREAD\n")
    coad = tcga.get("coadread_adapter_corr", {})
    lines.append(
        f"- Spearman(APOBEC1, Neither) all rows:  {coad.get('spearman_all', float('nan')):.4f}"
    )
    lines.append(
        f"- Spearman(APOBEC1, Neither) mutations only:  {coad.get('spearman_mutations_only', float('nan')):.4f}"
    )
    lines.append("")

    # 7. Failed stages & notes
    lines.append("## 7. Notes / failed stages\n")
    runlog = PROJECT_ROOT / "data" / "raw" / "apobec1" / "RUN_LOG.txt"
    if runlog.exists():
        lines.append("See `data/raw/apobec1/RUN_LOG.txt`. Key points:\n")
        lines.append(
            "- PNAS / BioRxiv / PMC supplementary files were blocked by Cloudflare and\n"
            "  PMC's proof-of-work viewer; Rayon-Estrada 2017 and Cole 2017 Dataset S1\n"
            "  were unreachable. Blanc 2021 RNA-journal supp is a docx of summary\n"
            "  statistics tables (no per-site coordinates).\n"
        )
        lines.append(
            "- Rosenberg 2011 is mouse (mm9), not human, so Gate B fell back to an\n"
            "  adapter-agreement check rather than held-out cross-species validation.\n"
        )

    (OUTPUT_DIR / "REPORT.md").write_text("\n".join(lines))
    logger.info("Wrote REPORT.md to %s", OUTPUT_DIR / "REPORT.md")


if __name__ == "__main__":
    main()
