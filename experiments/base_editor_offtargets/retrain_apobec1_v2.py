#!/usr/bin/env python3
"""Retrain APOBEC1 head on KO-validated subset of v1 + sanity check.

Plan:
1. Load existing v1 features (rnafm orig+delta, hand40, splits CSV).
2. Sanity-check OLD apobec1_head on v1 train distribution (positives should score higher than negatives).
3. Filter to ko_validated == 1 (120 KO-validated positives, plus their motif-matched negatives).
4. Train new APOBEC1 head MLP on Phase3 shared encoder embeddings.
5. Save as apobec1_head_mfe_only_v2.pt.
6. Score Lei v2 sequences (already computed RNA-FM + hand40 for v2) with new head.
7. Recompute Lei enrichment with new head alongside old.

Outputs:
- experiments/multi_enzyme/outputs/apobec1_head/apobec1_head_mfe_only_v2.pt
- experiments/base_editor_offtargets/outputs/apobec1_v2_retrain/sanity_check.json
- experiments/base_editor_offtargets/outputs/apobec1_v2_retrain/training_curve.json
- experiments/base_editor_offtargets/outputs/lei_preview_v2/scored_sites_with_v2_a1.csv
"""
from __future__ import annotations
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

ROOT = Path("/Users/shaharharel/Documents/github/edit-rna-apobec")
A1_DIR = ROOT / "data/processed/apobec1"
PHASE3_CKPT = ROOT / "experiments/multi_enzyme/outputs/phase3_mfe_only/phase3_mfe_only.pt"
A1_CKPT = ROOT / "experiments/multi_enzyme/outputs/apobec1_head/apobec1_head_mfe_only.pt"
OUT_MODEL = ROOT / "experiments/multi_enzyme/outputs/apobec1_head/apobec1_head_mfe_only_v2.pt"
OUT = ROOT / "experiments/base_editor_offtargets/outputs/apobec1_v2_retrain"
OUT.mkdir(parents=True, exist_ok=True)

LEI_V2_DIR = ROOT / "experiments/base_editor_offtargets/outputs/lei_preview_v2"

D_INPUT = 1320
D_SHARED = 128
EMB_DIM = 640
ENZYMES = ["A3A", "A3B", "A3G", "A3A_A3G", "Neither"]
N_ENZYMES_CLS = 6
SEED = 42

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("retrain")


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
        self.species_proj = nn.Sequential(nn.Linear(1, 16), nn.GELU(), nn.Linear(16, d_shared))
        self.head = nn.Sequential(nn.Linear(d_shared, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, shared, species):
        bias = self.species_proj(species)
        return self.head(shared + bias).squeeze(-1)


def load_v1_data():
    """Load v1 features + splits."""
    splits = pd.read_csv(A1_DIR / "apobec1_v1_with_negatives.csv")
    rnafm_orig = torch.load(A1_DIR / "rnafm_apobec1_v1.pt", weights_only=False)
    rnafm_edit = torch.load(A1_DIR / "rnafm_apobec1_v1_edited.pt", weights_only=False)
    hand40 = np.load(A1_DIR / "apobec1_hand40.npy")
    log.info("v1 splits: %d rows (%d positives, %d negatives)",
             len(splits),
             (splits["is_edited"] == 1).sum(),
             (splits["is_edited"] == 0).sum())
    log.info("rnafm_orig: %s, rnafm_edit: %s, hand40: %s",
             type(rnafm_orig).__name__, type(rnafm_edit).__name__, hand40.shape)
    return splits, rnafm_orig, rnafm_edit, hand40


def to_array(rnafm_obj):
    """Convert torch tensor or dict to (n, 640) numpy array."""
    if isinstance(rnafm_obj, dict):
        # Assume keys ordered by site_id; try common ordering
        keys = list(rnafm_obj.keys())
        if all(isinstance(v, torch.Tensor) for v in rnafm_obj.values()):
            arr = torch.stack([rnafm_obj[k] for k in keys]).cpu().numpy()
        else:
            arr = np.stack([rnafm_obj[k] for k in keys])
        log.info("Converted rnafm dict (%d entries) → array %s", len(keys), arr.shape)
        return arr, keys
    elif isinstance(rnafm_obj, torch.Tensor):
        return rnafm_obj.cpu().numpy(), None
    else:
        return np.asarray(rnafm_obj), None


def get_shared_embeddings(orig, delta, hand40, valid, p3_model, device, batch=512):
    """Run Phase3 shared encoder on (orig, delta, hand40) → 128-d shared embeddings."""
    n = len(orig)
    shared_out = np.zeros((n, D_SHARED), dtype=np.float32)
    p3_model.eval()
    with torch.no_grad():
        for i in range(0, n, batch):
            j = min(i + batch, n)
            x = np.empty((j - i, D_INPUT), dtype=np.float32)
            x[:, :EMB_DIM] = orig[i:j].astype(np.float32)
            x[:, EMB_DIM:2 * EMB_DIM] = delta[i:j].astype(np.float32)
            x[:, 2 * EMB_DIM:] = hand40[i:j].astype(np.float32)
            x[~valid[i:j]] = 0.0
            xt = torch.from_numpy(x).to(device)
            shared = p3_model.shared_encoder(xt)
            shared_out[i:j] = shared.cpu().numpy()
    return shared_out


def score_with_a1(shared, species_val, a1_head, device, batch=512):
    """Score with apobec1 head."""
    n = len(shared)
    out = np.zeros(n, dtype=np.float32)
    a1_head.eval()
    with torch.no_grad():
        for i in range(0, n, batch):
            j = min(i + batch, n)
            sh = torch.from_numpy(shared[i:j]).to(device)
            sp = torch.full((j - i, 1), species_val, dtype=torch.float32, device=device)
            out[i:j] = torch.sigmoid(a1_head(sh, sp)).cpu().numpy()
    return out


def train_a1_head(shared, labels, species, n_folds=5, epochs=30, lr=1e-3, device="cpu"):
    """Train new A1 head with k-fold CV."""
    rng = np.random.default_rng(SEED)
    n = len(shared)
    idx = np.arange(n)
    rng.shuffle(idx)
    kf = KFold(n_splits=n_folds, shuffle=False)

    all_aucs = []
    fold_predictions = np.zeros(n, dtype=np.float32)

    for fold, (tr, va) in enumerate(kf.split(idx)):
        tr_idx = idx[tr]; va_idx = idx[va]
        head = APOBEC1Head().to(device)
        opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
        bce = nn.BCEWithLogitsLoss()

        sh_tr = torch.from_numpy(shared[tr_idx]).to(device)
        sp_tr = torch.from_numpy(species[tr_idx].reshape(-1, 1).astype(np.float32)).to(device)
        y_tr = torch.from_numpy(labels[tr_idx].astype(np.float32)).to(device)
        sh_va = torch.from_numpy(shared[va_idx]).to(device)
        sp_va = torch.from_numpy(species[va_idx].reshape(-1, 1).astype(np.float32)).to(device)
        y_va = labels[va_idx]

        head.train()
        for ep in range(epochs):
            # mini-batch
            perm = torch.randperm(len(tr_idx), device=device)
            for k in range(0, len(perm), 64):
                b = perm[k:k + 64]
                logit = head(sh_tr[b], sp_tr[b])
                loss = bce(logit, y_tr[b])
                opt.zero_grad()
                loss.backward()
                opt.step()

        head.eval()
        with torch.no_grad():
            logit_va = head(sh_va, sp_va)
            prob_va = torch.sigmoid(logit_va).cpu().numpy()
        try:
            auc = roc_auc_score(y_va, prob_va)
        except ValueError:
            auc = float("nan")
        log.info("  fold %d AUC=%.4f", fold, auc)
        all_aucs.append(auc)
        fold_predictions[va_idx] = prob_va

    # Final model: train on all data
    log.info("Training final head on all data...")
    head = APOBEC1Head().to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    sh_all = torch.from_numpy(shared).to(device)
    sp_all = torch.from_numpy(species.reshape(-1, 1).astype(np.float32)).to(device)
    y_all = torch.from_numpy(labels.astype(np.float32)).to(device)
    head.train()
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        for k in range(0, len(perm), 64):
            b = perm[k:k + 64]
            logit = head(sh_all[b], sp_all[b])
            loss = bce(logit, y_all[b])
            opt.zero_grad(); loss.backward(); opt.step()
    head.eval()

    return head, all_aucs, fold_predictions


def main():
    t0 = time.time()
    log.info("=" * 60)
    log.info("APOBEC1 head retrain v2 (KO-validated subset)")
    log.info("=" * 60)

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    log.info("Device: %s", device)

    # Load v1 data
    splits, rnafm_orig_obj, rnafm_edit_obj, hand40 = load_v1_data()

    # Convert RNA-FM features to arrays in splits order
    site_ids = splits["site_id"].tolist()

    if isinstance(rnafm_orig_obj, dict):
        orig = np.stack([rnafm_orig_obj[k].cpu().numpy() if isinstance(rnafm_orig_obj[k], torch.Tensor)
                          else np.asarray(rnafm_orig_obj[k]) for k in site_ids]).astype(np.float32)
        edit = np.stack([rnafm_edit_obj[k].cpu().numpy() if isinstance(rnafm_edit_obj[k], torch.Tensor)
                          else np.asarray(rnafm_edit_obj[k]) for k in site_ids]).astype(np.float32)
    else:
        orig = rnafm_orig_obj.cpu().numpy().astype(np.float32) if isinstance(rnafm_orig_obj, torch.Tensor) else rnafm_orig_obj.astype(np.float32)
        edit = rnafm_edit_obj.cpu().numpy().astype(np.float32) if isinstance(rnafm_edit_obj, torch.Tensor) else rnafm_edit_obj.astype(np.float32)

    delta = edit - orig

    log.info("orig %s, delta %s, hand40 %s", orig.shape, delta.shape, hand40.shape)
    n = len(splits)
    valid = np.ones(n, dtype=bool)

    # Load Phase3 model
    log.info("Loading Phase3 model...")
    p3 = Phase3Model()
    state = torch.load(PHASE3_CKPT, weights_only=False, map_location="cpu")
    state = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state
    p3.load_state_dict(state, strict=False)
    p3 = p3.eval().to(device)

    # Compute shared embeddings for ALL v1 sites
    log.info("Computing Phase3 shared embeddings for v1...")
    shared = get_shared_embeddings(orig, delta, hand40, valid, p3, device)
    log.info("  shared: %s", shared.shape)

    # ----- Sanity check: OLD A1 head on v1 training distribution -----
    log.info("Loading OLD A1 head (apobec1_head_mfe_only.pt)...")
    old_a1 = APOBEC1Head().to(device)
    old_a1.load_state_dict(torch.load(A1_CKPT, weights_only=False, map_location="cpu"))
    old_a1.eval()

    # Determine species: positives are mouse, negatives matched to positives → mouse too
    # In v1 training, species was derived from rows. Check if there's a species column.
    if "species" in splits.columns:
        species_v1 = (splits["species"].astype(str).str.lower() == "mouse").astype(np.float32).values
    else:
        species_v1 = np.ones(n, dtype=np.float32)  # all mouse

    is_pos_v1 = (splits["is_edited"] == 1).values

    log.info("v1 species channel breakdown (1=mouse, 0=human): mean=%.3f", species_v1.mean())

    # Score with OLD head, both species channels
    score_old_sp1 = score_with_a1(shared, 1.0, old_a1, device)
    score_old_sp0 = score_with_a1(shared, 0.0, old_a1, device)

    log.info("SANITY CHECK — OLD A1 head on v1 training data:")
    log.info("  pos mean (sp=1): %.4f, neg mean (sp=1): %.4f, AUC: %.4f",
             score_old_sp1[is_pos_v1].mean(), score_old_sp1[~is_pos_v1].mean(),
             roc_auc_score(is_pos_v1, score_old_sp1))
    log.info("  pos mean (sp=0): %.4f, neg mean (sp=0): %.4f, AUC: %.4f",
             score_old_sp0[is_pos_v1].mean(), score_old_sp0[~is_pos_v1].mean(),
             roc_auc_score(is_pos_v1, score_old_sp0))

    sanity = {
        "old_head_v1_auc_sp1": float(roc_auc_score(is_pos_v1, score_old_sp1)),
        "old_head_v1_auc_sp0": float(roc_auc_score(is_pos_v1, score_old_sp0)),
        "old_head_v1_pos_mean_sp1": float(score_old_sp1[is_pos_v1].mean()),
        "old_head_v1_neg_mean_sp1": float(score_old_sp1[~is_pos_v1].mean()),
        "old_head_v1_pos_mean_sp0": float(score_old_sp0[is_pos_v1].mean()),
        "old_head_v1_neg_mean_sp0": float(score_old_sp0[~is_pos_v1].mean()),
        "n_pos": int(is_pos_v1.sum()),
        "n_neg": int((~is_pos_v1).sum()),
    }

    # ----- Filter to KO-validated -----
    if "ko_validated" not in splits.columns:
        log.warning("No ko_validated column; using all positives")
        ko_mask = np.ones(n, dtype=bool)
    else:
        # Keep all KO-validated positives + their negatives.
        # Strategy: take all rows where is_edited=1 AND ko_validated=1, then sample matched negatives 1:1.
        ko_pos_mask = (splits["is_edited"] == 1) & (splits["ko_validated"] == 1)
        log.info("KO-validated positives: %d", ko_pos_mask.sum())

        # For negatives, use all v1 negatives (they're motif-matched globally)
        # Sample 1:1
        rng = np.random.default_rng(SEED)
        neg_idx = np.where(splits["is_edited"] == 0)[0]
        n_pos = ko_pos_mask.sum()
        if n_pos < len(neg_idx):
            sampled_neg = rng.choice(neg_idx, n_pos, replace=False)
        else:
            sampled_neg = neg_idx
        ko_mask = np.zeros(n, dtype=bool)
        ko_mask[ko_pos_mask] = True
        ko_mask[sampled_neg] = True

    sub_shared = shared[ko_mask]
    sub_labels = (splits["is_edited"].values[ko_mask]).astype(np.int64)
    sub_species = species_v1[ko_mask]
    log.info("Training subset: %d (%d pos, %d neg)",
             len(sub_shared), int((sub_labels == 1).sum()), int((sub_labels == 0).sum()))

    # ----- Train new head -----
    log.info("Training new A1 head with 5-fold CV...")
    new_head, fold_aucs, fold_preds = train_a1_head(
        sub_shared, sub_labels, sub_species,
        n_folds=5, epochs=30, lr=1e-3, device=device)

    log.info("Mean CV AUC: %.4f (std %.4f)", np.mean(fold_aucs), np.std(fold_aucs))
    log.info("Saving new head to %s", OUT_MODEL)
    torch.save(new_head.state_dict(), OUT_MODEL)

    summary = {
        "sanity_check": sanity,
        "training_subset": {
            "n_total": int(len(sub_shared)),
            "n_pos": int((sub_labels == 1).sum()),
            "n_neg": int((sub_labels == 0).sum()),
        },
        "cv_fold_aucs": [float(a) for a in fold_aucs],
        "cv_mean_auc": float(np.mean(fold_aucs)),
        "cv_std_auc": float(np.std(fold_aucs)),
        "saved_to": str(OUT_MODEL),
    }
    with open(OUT / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # ----- Score Lei v2 sequences with new head -----
    log.info("Loading Lei v2 scored sites for re-scoring with new head...")
    lei_scored = pd.read_csv(LEI_V2_DIR / "scored_sites.csv")
    n_lei = len(lei_scored)
    log.info("Lei v2 sites: %d", n_lei)

    # We need RNA-FM + hand40 for Lei sites. Recompute? Or save from v2 run?
    # The v2 run didn't save those arrays — only scored_sites.csv. Recompute is needed.
    # Quick alternative: redirect to a dedicated scoring step in a follow-up script.
    log.warning("Lei re-scoring with new A1 head requires RNA-FM + hand40 features which "
                "weren't saved by v2 run. Will need a separate re-scoring step.")
    log.warning("To re-score: extract sequences from Lei v2 scored_sites.csv (chrom,pos,strand), "
                "recompute features, score with apobec1_head_mfe_only_v2.pt.")

    log.info("Total time: %.1f min", (time.time() - t0) / 60)
    log.info("Saved new head: %s", OUT_MODEL)
    log.info("Mean CV AUC: %.4f", np.mean(fold_aucs))


if __name__ == "__main__":
    main()
