#!/usr/bin/env python3
"""Train `apobec1_head_mfe_only.pt` on top of the frozen `phase3_mfe_only.pt`
shared encoder, with struct_delta slots held at zero throughout.

This is a trimmed re-implementation of `exp_train_apobec1_head.py` that:
 - loads the MFE-only Phase3 checkpoint instead of the canonical one
 - zeros struct_delta (abs slice [1304:1311]) in every training / eval batch
 - saves to `experiments/multi_enzyme/outputs/apobec1_head/apobec1_head_mfe_only.pt`
 - ALSO runs 5-fold CV and gate-A (Neither sites) so we can compare to the
   canonical head's AUROC (0.??) reported in `apobec1_head/run_summary.json`.

All architecture + hyperparameters match the canonical trainer exactly; the
only differences are the pretrained checkpoint path and the feature-zeroing.
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
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.apobec_feature_extraction import build_hand_features  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "multi_enzyme" / "outputs" / "apobec1_head"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

log_file = OUTPUT_DIR / "run_mfe_only.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, mode="w")],
)
logger = logging.getLogger(__name__)

SEED = 42
N_FOLDS = 5
D_INPUT = 1320
D_SHARED = 128
ENZYMES = ["A3A", "A3B", "A3G", "A3A_A3G", "Neither"]
N_ENZYMES_CLS = 6
STRUCT_DELTA_START = 640 + 640 + 24  # 1304
STRUCT_DELTA_END = STRUCT_DELTA_START + 7  # 1311

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else torch.device("cpu")
)
logger.info("Device: %s", DEVICE)

DATA_DIR = PROJECT_ROOT / "data"
APOBEC1_DIR = DATA_DIR / "processed" / "apobec1"
APOBEC1_SPLITS = APOBEC1_DIR / "apobec1_v1_with_negatives.csv"
APOBEC1_SEQS = APOBEC1_DIR / "apobec1_v1_sequences.json"
APOBEC1_HAND = APOBEC1_DIR / "apobec1_hand40.npy"
APOBEC1_RNAFM = APOBEC1_DIR / "rnafm_apobec1_v1.pt"
APOBEC1_RNAFM_ED = APOBEC1_DIR / "rnafm_apobec1_v1_edited.pt"

ME_DIR = DATA_DIR / "processed" / "multi_enzyme"
V3_SPLITS = ME_DIR / "splits_multi_enzyme_v3_with_negatives.csv"
V3_SEQS = ME_DIR / "multi_enzyme_sequences_v3_with_negatives.json"
V3_LOOP = ME_DIR / "loop_position_per_site_v3.csv"
V3_STRUCT = ME_DIR / "structure_cache_multi_enzyme_v3.npz"
V3_RNAFM_ORIG = ME_DIR / "embeddings" / "rnafm_pooled_v3.pt"
V3_RNAFM_EDITED = ME_DIR / "embeddings" / "rnafm_pooled_edited_v3.pt"

PHASE3_MFE_CKPT = (
    PROJECT_ROOT / "experiments" / "multi_enzyme" / "outputs"
    / "phase3_mfe_only" / "phase3_mfe_only.pt"
)


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

    def forward(self, shared, species):
        bias = self.species_proj(species)
        return self.head(shared + bias).squeeze(-1)


def zero_struct_delta(X: np.ndarray) -> np.ndarray:
    X = X.copy()
    X[:, STRUCT_DELTA_START:STRUCT_DELTA_END] = 0.0
    return X


def _load_rnafm_dict(path):
    return torch.load(path, weights_only=False, map_location="cpu")


def load_apobec1_data():
    logger.info("Loading APOBEC1 training data ...")
    splits = pd.read_csv(APOBEC1_SPLITS)
    with open(APOBEC1_SEQS) as f:
        seqs = json.load(f)
    site_ids = splits["site_id"].tolist()
    n = len(site_ids)
    logger.info("  %d sites (%d pos / %d neg)", n, int((splits["is_edited"] == 1).sum()),
                int((splits["is_edited"] == 0).sum()))
    hand = np.load(APOBEC1_HAND)
    assert hand.shape[0] == n
    rnafm_orig = _load_rnafm_dict(APOBEC1_RNAFM)
    rnafm_edit = _load_rnafm_dict(APOBEC1_RNAFM_ED)
    D_RNAFM = 640
    emb_orig = np.zeros((n, D_RNAFM), dtype=np.float32)
    emb_delta = np.zeros((n, D_RNAFM), dtype=np.float32)
    missing = 0
    for i, sid in enumerate(site_ids):
        if sid in rnafm_orig and sid in rnafm_edit:
            o = rnafm_orig[sid]; e = rnafm_edit[sid]
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
    X = zero_struct_delta(X)  # MFE-only regime
    labels = splits["is_edited"].values.astype(np.float32)
    species = np.ones(n, dtype=np.float32)
    return {"X": X, "labels": labels, "species": species, "site_ids": site_ids, "splits": splits}


def load_neither_data():
    logger.info("Loading Neither (human) Gate A data ...")
    splits = pd.read_csv(V3_SPLITS)
    nei = splits[splits["enzyme"] == "Neither"].copy()
    with open(V3_SEQS) as f:
        seqs_all = json.load(f)
    loop_df = pd.read_csv(V3_LOOP).drop_duplicates(subset=["site_id"]).set_index("site_id")
    sc = np.load(V3_STRUCT, allow_pickle=True)
    struct_map = {sid: sc["delta_features"][i] for i, sid in enumerate(sc["site_ids"])}
    nei_use = nei.reset_index(drop=True)
    site_ids = nei_use["site_id"].tolist()
    have_seq = [sid for sid in site_ids if sid in seqs_all]
    if len(have_seq) < len(site_ids):
        logger.warning("  %d missing sequences", len(site_ids) - len(have_seq))
        nei_use = nei_use[nei_use["site_id"].isin(set(have_seq))].reset_index(drop=True)
        site_ids = nei_use["site_id"].tolist()
    hand = build_hand_features(site_ids, seqs_all, struct_map, loop_df)
    n = len(site_ids)
    D_RNAFM = 640
    rnafm_o = _load_rnafm_dict(V3_RNAFM_ORIG)
    rnafm_e = _load_rnafm_dict(V3_RNAFM_EDITED)
    emb_o = np.zeros((n, D_RNAFM), dtype=np.float32)
    emb_d = np.zeros((n, D_RNAFM), dtype=np.float32)
    for i, sid in enumerate(site_ids):
        if sid in rnafm_o and sid in rnafm_e:
            o = rnafm_o[sid]; e = rnafm_e[sid]
            if isinstance(o, torch.Tensor):
                o = o.numpy()
            if isinstance(e, torch.Tensor):
                e = e.numpy()
            emb_o[i] = o
            emb_d[i] = e - o
    X = np.concatenate([emb_o, emb_d, hand], axis=1).astype(np.float32)
    X = zero_struct_delta(X)  # MFE-only regime
    labels = nei_use["is_edited"].values.astype(np.float32)
    species = np.zeros(n, dtype=np.float32)
    logger.info("  Neither data: %d pos / %d neg", int((labels == 1).sum()), int((labels == 0).sum()))
    return {"X": X, "labels": labels, "species": species, "site_ids": site_ids}


def load_pretrained(model: Phase3Model):
    state = torch.load(PHASE3_MFE_CKPT, weights_only=False, map_location="cpu")
    model.load_state_dict(state)
    logger.info("Loaded MFE-only Phase3 checkpoint: %s", PHASE3_MFE_CKPT)


def freeze_pretrained(model):
    for p in model.parameters():
        p.requires_grad = False
    model.eval()


def train_one_fold(model, head, X_tr, y_tr, s_tr, X_va, y_va, s_va, n_epochs=20, lr=1e-3):
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
    BATCH = 64
    n = len(X_tr)
    best_auroc = 0.0
    best_state = None
    for ep in range(n_epochs):
        model.eval()
        head.train()
        idx = np.random.RandomState(SEED + ep).permutation(n)
        total = 0.0
        steps = 0
        for b in range(0, n, BATCH):
            bi = idx[b:b + BATCH]
            xb = X_tr_t[bi].to(DEVICE); yb = y_tr_t[bi].to(DEVICE); sb = s_tr_t[bi].to(DEVICE)
            with torch.no_grad():
                shared = model.shared_encoder(xb)
            logit = head(shared, sb)
            loss = F.binary_cross_entropy_with_logits(logit, yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(new_params, 1.0)
            opt.step()
            total += float(loss.item()); steps += 1
        sch.step()
        head.eval()
        with torch.no_grad():
            probs = []
            for b in range(0, len(X_va_t), 256):
                xv = X_va_t[b:b + 256].to(DEVICE); sv = s_va_t[b:b + 256].to(DEVICE)
                shared = model.shared_encoder(xv)
                logit = head(shared, sv)
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
            best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
    if best_state is not None:
        head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        probs = []
        for b in range(0, len(X_va_t), 256):
            xv = X_va_t[b:b + 256].to(DEVICE); sv = s_va_t[b:b + 256].to(DEVICE)
            shared = model.shared_encoder(xv)
            logit = head(shared, sv)
            probs.append(torch.sigmoid(logit).cpu().numpy())
        probs = np.concatenate(probs)
    auroc = float(roc_auc_score(y_va, probs)) if len(np.unique(y_va)) > 1 else float("nan")
    auprc = float(average_precision_score(y_va, probs)) if len(np.unique(y_va)) > 1 else float("nan")
    return {"auroc": auroc, "auprc": auprc, "probs": probs, "best_state": best_state}


def run_5fold_cv(d):
    X = d["X"]; y = d["labels"]; s = d["species"]
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = []
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        logger.info("\n--- Fold %d/%d (train=%d, val=%d) ---", fold + 1, N_FOLDS, len(tr), len(va))
        torch.manual_seed(SEED + fold); np.random.seed(SEED + fold)
        model = Phase3Model(); load_pretrained(model); freeze_pretrained(model); model.to(DEVICE)
        head = APOBEC1Head(D_SHARED)
        res = train_one_fold(model, head, X[tr], y[tr], s[tr], X[va], y[va], s[va])
        logger.info("  Fold %d  AUROC=%.4f  AUPRC=%.4f", fold + 1, res["auroc"], res["auprc"])
        folds.append({"fold": fold + 1, "auroc": res["auroc"], "auprc": res["auprc"]})
    # Retrain on all
    logger.info("\nRetraining on full training set ...")
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = Phase3Model(); load_pretrained(model); freeze_pretrained(model); model.to(DEVICE)
    head = APOBEC1Head(D_SHARED); head.to(DEVICE); head.train()
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
            xb = X_t[bi].to(DEVICE); yb = y_t[bi].to(DEVICE); sb = s_t[bi].to(DEVICE)
            with torch.no_grad():
                shared = model.shared_encoder(xb)
            logit = head(shared, sb)
            loss = F.binary_cross_entropy_with_logits(logit, yb)
            opt.zero_grad(); loss.backward(); opt.step()
    prod_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
    mean_a = float(np.mean([m["auroc"] for m in folds]))
    std_a = float(np.std([m["auroc"] for m in folds]))
    logger.info("5-fold CV: mean AUROC=%.4f ± %.4f", mean_a, std_a)
    return prod_state, {"folds": folds, "mean_auroc": mean_a, "std_auroc": std_a,
                        "n_train_pos": int(np.sum(y)), "n_train_neg": int(len(y) - np.sum(y))}


def gate_a(model, head):
    logger.info("\n" + "=" * 60)
    logger.info("GATE A — Neither sites (species=0)")
    logger.info("=" * 60)
    d = load_neither_data()
    n = len(d["X"])
    probs = np.zeros(n, dtype=np.float32)
    head.eval(); model.eval()
    with torch.no_grad():
        for b in range(0, n, 256):
            e = min(b + 256, n)
            xb = torch.from_numpy(d["X"][b:e]).float().to(DEVICE)
            sb = torch.full((e - b, 1), 0.0, dtype=torch.float32).to(DEVICE)
            shared = model.shared_encoder(xb)
            logit = head(shared, sb)
            probs[b:e] = torch.sigmoid(logit).cpu().numpy()
    try:
        auroc = float(roc_auc_score(d["labels"], probs))
        auprc = float(average_precision_score(d["labels"], probs))
    except Exception:
        auroc = float("nan"); auprc = float("nan")
    logger.info("Gate A: n_pos=%d  n_neg=%d  AUROC=%.4f  AUPRC=%.4f",
                int((d["labels"] == 1).sum()), int((d["labels"] == 0).sum()), auroc, auprc)
    return {"auroc": auroc, "auprc": auprc, "n_pos": int((d["labels"] == 1).sum()),
            "n_neg": int((d["labels"] == 0).sum())}


def main():
    t0 = time.time()
    logger.info("apobec1_head_mfe_only training — struct_delta zeroed, frozen phase3_mfe_only shared encoder")
    train = load_apobec1_data()
    prod_state, summary = run_5fold_cv(train)
    save_path = OUTPUT_DIR / "apobec1_head_mfe_only.pt"
    torch.save(prod_state, save_path)
    logger.info("Saved %s", save_path)

    # Load prod head for gate A
    model = Phase3Model(); load_pretrained(model); freeze_pretrained(model); model.to(DEVICE)
    head = APOBEC1Head(D_SHARED); head.load_state_dict(prod_state); head.to(DEVICE)
    ga = gate_a(model, head)

    summary["gate_a"] = ga
    summary["device"] = str(DEVICE)
    summary["runtime_min"] = (time.time() - t0) / 60.0
    with open(OUTPUT_DIR / "apobec1_head_mfe_only_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Runtime: %.1f min", summary["runtime_min"])


if __name__ == "__main__":
    main()
