#!/usr/bin/env python3
"""Train APOBEC1 head v4 (cancer or cds variant) on top of the FROZEN
phase3_v4_cds shared encoder, with v4-trinuc-matched negatives.

Architecture mirrors `exp_train_apobec1_head_mfe_only.py`:
  - 1320-d input  (RNA-FM 640 orig + 640 delta + 40 hand)
  - struct_delta (abs slice [1304:1311]) zeroed during MFE-only regime
  - APOBEC1Head: species_proj + small linear head (BCE on apobec1 labels)

Differences vs the v3 trainer:
  - Loads v4-cancer or v4-cds dataset (CLI flag --variant)
  - Uses phase3_v4_cds shared encoder (FROZEN), per team's choice for cross-variant consistency
  - Saves to experiments/multi_enzyme/outputs/apobec1_head_v4_{variant}/

Usage:
    python experiments/multi_enzyme/exp_train_apobec1_head_v4.py --variant cancer
    python experiments/multi_enzyme/exp_train_apobec1_head_v4.py --variant cds
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
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

SEED = 20260427
N_FOLDS = 5
D_INPUT = 1320
D_SHARED = 128
ENZYMES = ["A3A", "A3B", "A3G", "A3A_A3G", "Neither"]
N_ENZYMES_CLS = 5  # phase3 v4 uses 5 classes (no "Unknown")
STRUCT_DELTA_START = 640 + 640 + 24  # 1304
STRUCT_DELTA_END = STRUCT_DELTA_START + 7  # 1311

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else torch.device("cpu")
)


class Phase3Model(nn.Module):
    """Mirrors the v4 CDS Phase3 architecture (frozen here)."""
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

    def forward(self, shared, species):
        bias = self.species_proj(species)
        return self.head(shared + bias).squeeze(-1)


def zero_struct_delta(X: np.ndarray) -> np.ndarray:
    X = X.copy()
    X[:, STRUCT_DELTA_START:STRUCT_DELTA_END] = 0.0
    return X


def _load_rnafm_dict(path):
    return torch.load(path, weights_only=False, map_location="cpu")


def load_v4_data(variant: str, logger):
    """Load the v4 apobec1 training data (cancer or cds)."""
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_V4 = DATA_DIR / "raw" / "apobec1" / "v4"
    PROC_V4 = DATA_DIR / "processed" / "apobec1" / "v4"

    splits_path = RAW_V4 / f"apobec1_v4_{variant}_with_negatives.csv"
    seqs_path = RAW_V4 / f"apobec1_v4_{variant}_sequences.json"
    hand_path = PROC_V4 / f"apobec1_hand40_v4_{variant}.npy"
    rnafm_orig = PROC_V4 / f"rnafm_apobec1_v4_{variant}.pt"
    rnafm_edit = PROC_V4 / f"rnafm_apobec1_v4_{variant}_edited.pt"

    logger.info("Loading APOBEC1 v4_%s data ...", variant)
    splits = pd.read_csv(splits_path)
    site_ids = splits["site_id"].tolist()
    n = len(site_ids)
    n_pos = int((splits["is_edited"] == 1).sum())
    n_neg = int((splits["is_edited"] == 0).sum())
    logger.info("  %d sites (%d pos / %d neg)", n, n_pos, n_neg)

    hand = np.load(hand_path)
    assert hand.shape[0] == n, f"hand shape mismatch: {hand.shape[0]} vs {n}"

    o_dict = _load_rnafm_dict(rnafm_orig)
    e_dict = _load_rnafm_dict(rnafm_edit)
    D_RNAFM = 640
    emb_o = np.zeros((n, D_RNAFM), dtype=np.float32)
    emb_d = np.zeros((n, D_RNAFM), dtype=np.float32)
    miss = 0
    for i, sid in enumerate(site_ids):
        if sid in o_dict and sid in e_dict:
            o = o_dict[sid]
            e = e_dict[sid]
            if isinstance(o, torch.Tensor):
                o = o.numpy()
            if isinstance(e, torch.Tensor):
                e = e.numpy()
            emb_o[i] = o
            emb_d[i] = e - o
        else:
            miss += 1
    logger.info("  RNA-FM coverage: %d/%d (missing=%d)", n - miss, n, miss)
    X = np.concatenate([emb_o, emb_d, hand], axis=1).astype(np.float32)
    X = zero_struct_delta(X)
    labels = splits["is_edited"].values.astype(np.float32)
    species = np.ones(n, dtype=np.float32)  # mouse-trained → species=1
    return {"X": X, "labels": labels, "species": species, "site_ids": site_ids}


def load_pretrained(model: Phase3Model, ckpt_path: Path, logger):
    state = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    # Allow extra keys (e.g. enzyme adapters not present); strict=False
    miss, unexp = model.load_state_dict(state, strict=False)
    logger.info("Loaded Phase3 v4 CDS checkpoint: %s (missing=%d, unexpected=%d)",
                ckpt_path, len(miss), len(unexp))


def freeze_pretrained(model):
    for p in model.parameters():
        p.requires_grad = False
    model.eval()


def train_one_fold(model, head, X_tr, y_tr, s_tr, X_va, y_va, s_va, logger,
                   n_epochs=20, lr=1e-3, batch=64):
    head.to(DEVICE)
    new_params = list(head.parameters())
    opt = torch.optim.AdamW(new_params, lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    X_tr_t = torch.from_numpy(X_tr).float()
    y_tr_t = torch.from_numpy(y_tr).float()
    s_tr_t = torch.from_numpy(s_tr).float().unsqueeze(1)
    X_va_t = torch.from_numpy(X_va).float()
    y_va_t = torch.from_numpy(y_va).float()
    s_va_t = torch.from_numpy(s_va).float().unsqueeze(1)
    n = len(X_tr)
    best_auroc = 0.0
    best_state = None
    for ep in range(n_epochs):
        model.eval()
        head.train()
        idx = np.random.RandomState(SEED + ep).permutation(n)
        total = 0.0
        steps = 0
        for b in range(0, n, batch):
            bi = idx[b:b + batch]
            xb = X_tr_t[bi].to(DEVICE)
            yb = y_tr_t[bi].to(DEVICE)
            sb = s_tr_t[bi].to(DEVICE)
            with torch.no_grad():
                shared = model.shared_encoder(xb)
            logit = head(shared, sb)
            loss = F.binary_cross_entropy_with_logits(logit, yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(new_params, 1.0)
            opt.step()
            total += float(loss.item())
            steps += 1
        sch.step()
        head.eval()
        with torch.no_grad():
            probs = []
            for b in range(0, len(X_va_t), 256):
                xv = X_va_t[b:b + 256].to(DEVICE)
                sv = s_va_t[b:b + 256].to(DEVICE)
                shared = model.shared_encoder(xv)
                logit = head(shared, sv)
                probs.append(torch.sigmoid(logit).cpu().numpy())
            probs = np.concatenate(probs)
        try:
            auroc = float(roc_auc_score(y_va_t.numpy(), probs))
        except Exception:
            auroc = 0.5
        if (ep + 1) % 5 == 0 or ep == 0:
            logger.info("    ep %02d  loss=%.4f  val_auroc=%.4f",
                        ep + 1, total / max(steps, 1), auroc)
        if auroc > best_auroc:
            best_auroc = auroc
            best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
    if best_state is not None:
        head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        probs = []
        for b in range(0, len(X_va_t), 256):
            xv = X_va_t[b:b + 256].to(DEVICE)
            sv = s_va_t[b:b + 256].to(DEVICE)
            shared = model.shared_encoder(xv)
            logit = head(shared, sv)
            probs.append(torch.sigmoid(logit).cpu().numpy())
        probs = np.concatenate(probs)
    auroc = float(roc_auc_score(y_va, probs)) if len(np.unique(y_va)) > 1 else float("nan")
    auprc = float(average_precision_score(y_va, probs)) if len(np.unique(y_va)) > 1 else float("nan")
    return {"auroc": auroc, "auprc": auprc, "probs": probs, "best_state": best_state}


def run_5fold_cv(d, ckpt_path, logger):
    X = d["X"]
    y = d["labels"]
    s = d["species"]
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = []
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        logger.info("\n--- Fold %d/%d (train=%d, val=%d) ---",
                    fold + 1, N_FOLDS, len(tr), len(va))
        torch.manual_seed(SEED + fold)
        np.random.seed(SEED + fold)
        model = Phase3Model()
        load_pretrained(model, ckpt_path, logger)
        freeze_pretrained(model)
        model.to(DEVICE)
        head = APOBEC1Head(D_SHARED)
        res = train_one_fold(model, head, X[tr], y[tr], s[tr],
                             X[va], y[va], s[va], logger)
        logger.info("  Fold %d  AUROC=%.4f  AUPRC=%.4f",
                    fold + 1, res["auroc"], res["auprc"])
        folds.append({"fold": fold + 1, "auroc": res["auroc"], "auprc": res["auprc"]})
    # Retrain on all
    logger.info("\nRetraining on full training set ...")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = Phase3Model()
    load_pretrained(model, ckpt_path, logger)
    freeze_pretrained(model)
    model.to(DEVICE)
    head = APOBEC1Head(D_SHARED)
    head.to(DEVICE)
    head.train()
    X_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y).float()
    s_t = torch.from_numpy(s).float().unsqueeze(1)
    BATCH = 64
    opt = torch.optim.AdamW(list(head.parameters()), lr=1e-3, weight_decay=1e-4)
    n = len(X)
    for ep in range(20):
        idx = np.random.RandomState(SEED + 100 + ep).permutation(n)
        for b in range(0, n, BATCH):
            bi = idx[b:b + BATCH]
            xb = X_t[bi].to(DEVICE)
            yb = y_t[bi].to(DEVICE)
            sb = s_t[bi].to(DEVICE)
            with torch.no_grad():
                shared = model.shared_encoder(xb)
            logit = head(shared, sb)
            loss = F.binary_cross_entropy_with_logits(logit, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    prod_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
    mean_a = float(np.mean([m["auroc"] for m in folds]))
    std_a = float(np.std([m["auroc"] for m in folds]))
    logger.info("5-fold CV: mean AUROC=%.4f +/- %.4f", mean_a, std_a)
    return prod_state, {
        "folds": folds,
        "mean_auroc": mean_a,
        "std_auroc": std_a,
        "n_train_pos": int(np.sum(y)),
        "n_train_neg": int(len(y) - np.sum(y)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["cancer", "cds"], required=True)
    args = ap.parse_args()

    output_dir = PROJECT_ROOT / "experiments" / "multi_enzyme" / "outputs" / f"apobec1_head_v4_{args.variant}"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, mode="w")],
        force=True,
    )
    logger = logging.getLogger(__name__)
    logger.info("Device: %s", DEVICE)
    logger.info("Variant: %s   seed=%d", args.variant, SEED)

    ckpt = PROJECT_ROOT / "experiments" / "multi_enzyme" / "outputs" / "v4_cds_unbiased" / "phase3_v4_cds.pt"

    t0 = time.time()
    train = load_v4_data(args.variant, logger)
    prod_state, summary = run_5fold_cv(train, ckpt, logger)

    save_path = output_dir / f"apobec1_head_v4_{args.variant}.pt"
    torch.save(prod_state, save_path)
    logger.info("Saved %s", save_path)

    summary["device"] = str(DEVICE)
    summary["seed"] = SEED
    summary["variant"] = args.variant
    summary["phase3_ckpt"] = str(ckpt)
    summary["runtime_min"] = (time.time() - t0) / 60.0
    with open(output_dir / "cv_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Runtime: %.1f min", summary["runtime_min"])


if __name__ == "__main__":
    main()
