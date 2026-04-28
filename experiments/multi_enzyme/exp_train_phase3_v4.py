#!/usr/bin/env python3
"""Train Phase3 multi-enzyme APOBEC predictor on v4 trinucleotide-matched negatives.

v4 fixes the v3 negative-control bias (positives 38.5% TC, negatives 57.2% TC) by
matching negatives' trinucleotide context to one of two reference distributions:

  - cancer_matched : negatives match TCGA pan-cancer C>T trinucleotide distribution
                     (RNA-to-DNA transfer claim)
  - cds_unbiased   : negatives match genome CDS-C trinucleotide distribution
                     (intrinsic motif-bias-free editability classifier)

This script is a parameterized version of `exp_train_phase3_mfe_only.py` adapted to:
  * Load v4 splits / sequences / loop / structure / RNA-FM (npz) for either variant.
  * Drop the Neither/APOBEC1 enzyme heads (positives now exclude Neither).
  * Use seed = 20260427 (project-mandated).
  * Save checkpoint, fold results, run log, OOF predictions for later bias diagnostics.

Feature layout (1320-d input):
    [0:640]        RNA-FM pooled original
    [640:1280]     RNA-FM edit delta (= edited - original)
    [1280:1304]    motif (24-d)
    [1304:1311]    struct_delta (7-d)   -- ZEROED in MFE-only regime
    [1311:1320]    loop geometry (9-d)

Heads (v4):
    - binary       : is_edited (1) vs not (0)
    - per-enzyme   : A3A, A3B, A3G, A3A_A3G  (4 heads; Neither dropped)
    - enzyme_cls   : 5-way softmax over {A3A, A3B, A3G, A3A_A3G, Unknown}

Usage:
    conda run -n quris python experiments/multi_enzyme/exp_train_phase3_v4.py \
        --variant cancer_matched \
        --out-dir experiments/multi_enzyme/outputs/v4_cancer_matched

    conda run -n quris python experiments/multi_enzyme/exp_train_phase3_v4.py \
        --variant cds_unbiased \
        --out-dir experiments/multi_enzyme/outputs/v4_cds_unbiased
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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.apobec_feature_extraction import build_hand_features  # noqa: E402

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else torch.device("cpu")
)

# v4 enzymes: dropped Neither / apobec1 (those positives are excluded from v4).
ENZYMES = ["A3A", "A3B", "A3G", "A3A_A3G"]                       # per-enzyme binary heads
ENZYME_CLASSES = ["A3A", "A3B", "A3G", "A3A_A3G", "Unknown"]     # softmax classes
ENZYME_TO_IDX = {e: i for i, e in enumerate(ENZYME_CLASSES)}
N_ENZYMES = len(ENZYME_CLASSES)
N_PER_ENZYME = len(ENZYMES)

D_RNAFM = 640
D_EDIT_DELTA = 640
D_HAND = 40
D_INPUT = D_RNAFM + D_EDIT_DELTA + D_HAND  # 1320
D_SHARED = 128

STRUCT_DELTA_START = D_RNAFM + D_EDIT_DELTA + 24      # 1304
STRUCT_DELTA_END = STRUCT_DELTA_START + 7             # 1311

# Defaults (matching v3 phase3_mfe_only)
DEFAULT_STAGE1_EPOCHS = 10
DEFAULT_STAGE1_LR = 1e-3
DEFAULT_STAGE2_EPOCHS = 20
DEFAULT_STAGE2_LR = 5e-4
DEFAULT_BATCH_SIZE = 64
DEFAULT_WEIGHT_DECAY = 1e-4
ENZYME_HEAD_WEIGHT = 0.3
ENZYME_CLS_WEIGHT = 0.1

DEFAULT_SEED = 20260427
N_FOLDS = 5

DATA_DIR = PROJECT_ROOT / "data"


def get_paths(variant: str) -> dict:
    if variant not in ("cancer_matched", "cds_unbiased"):
        raise ValueError(f"unknown variant: {variant!r}")
    return {
        "splits": DATA_DIR / "processed" / "multi_enzyme" / f"splits_multi_enzyme_v4_{variant}.csv",
        "seqs": DATA_DIR / "processed" / "multi_enzyme" / f"multi_enzyme_sequences_v4_{variant}.json",
        "loop": DATA_DIR / "processed" / "multi_enzyme" / f"loop_position_per_site_v4_{variant}.csv",
        "struct": DATA_DIR / "processed" / "embeddings" / f"structure_cache_multi_enzyme_v4_{variant}.npz",
        "rnafm": DATA_DIR / "processed" / "embeddings" / f"rnafm_v4_{variant}.npz",
    }


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class Phase3Model(nn.Module):
    """Same architecture family as v3 phase3, but with v4 head set (4 per-enzyme heads,
    5-way enzyme classifier).

    The state_dict keys differ from v3 by the enzyme_adapters/classifier shape; do not
    attempt to load v3 checkpoints into this class.
    """

    def __init__(self):
        super().__init__()
        self.shared_encoder = nn.Sequential(
            nn.Linear(D_INPUT, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.LayerNorm(256),
            nn.Linear(256, D_SHARED),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.binary_head = nn.Linear(D_SHARED, 1)
        self.enzyme_adapters = nn.ModuleDict({
            enz: nn.Sequential(nn.Linear(D_SHARED, 32), nn.GELU(), nn.Linear(32, 1))
            for enz in ENZYMES
        })
        self.enzyme_classifier = nn.Sequential(
            nn.Linear(D_SHARED, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, N_ENZYMES),
        )

    def forward(self, x):
        shared = self.shared_encoder(x)
        binary_logit = self.binary_head(shared).squeeze(-1)
        per_enzyme_logits = [self.enzyme_adapters[e](shared).squeeze(-1) for e in ENZYMES]
        enzyme_cls_logits = self.enzyme_classifier(shared)
        return binary_logit, per_enzyme_logits, enzyme_cls_logits, shared


class SimpleDataset(Dataset):
    def __init__(self, X, lb, le, pel):
        self.X = torch.from_numpy(X).float()
        self.lb = torch.from_numpy(lb).float()
        self.le = torch.from_numpy(le).long()
        self.pel = torch.from_numpy(pel).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            "x": self.X[idx],
            "label_binary": self.lb[idx],
            "label_enzyme": self.le[idx],
            "per_enzyme_labels": self.pel[idx],
        }


def compute_loss(binary_logit, per_enzyme_logits, enzyme_cls_logits, batch):
    device = binary_logit.device
    loss_binary = F.binary_cross_entropy_with_logits(binary_logit, batch["label_binary"])

    pel = batch["per_enzyme_labels"]
    loss_pe = torch.tensor(0.0, device=device)
    n_active = 0
    for h in range(N_PER_ENZYME):
        mask = pel[:, h] >= 0
        if mask.sum() < 2:
            continue
        hl = per_enzyme_logits[h][mask]
        lbl = pel[:, h][mask]
        loss_pe = loss_pe + F.binary_cross_entropy_with_logits(hl, lbl)
        n_active += 1
    if n_active > 0:
        loss_pe = loss_pe / n_active

    pos_mask = batch["label_binary"] == 1
    loss_cls = torch.tensor(0.0, device=device)
    if pos_mask.sum() > 0:
        loss_cls = F.cross_entropy(enzyme_cls_logits[pos_mask], batch["label_enzyme"][pos_mask])

    return loss_binary + ENZYME_HEAD_WEIGHT * loss_pe + ENZYME_CLS_WEIGHT * loss_cls


def train_epoch(model, loader, optim, device):
    model.train()
    total = 0.0
    n = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optim.zero_grad()
        bl, pel, ecl, _ = model(batch["x"])
        loss = compute_loss(bl, pel, ecl, batch)
        if not torch.isfinite(loss):
            return float("nan")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        total += loss.item()
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_bp = []
    all_pep = [[] for _ in range(N_PER_ENZYME)]
    all_lb = []
    all_le = []
    all_pel = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        bl, pel, _, _ = model(batch["x"])
        all_bp.append(torch.sigmoid(bl).cpu().numpy())
        for h in range(N_PER_ENZYME):
            all_pep[h].append(torch.sigmoid(pel[h]).cpu().numpy())
        all_lb.append(batch["label_binary"].cpu().numpy())
        all_le.append(batch["label_enzyme"].cpu().numpy())
        all_pel.append(batch["per_enzyme_labels"].cpu().numpy())
    bp = np.concatenate(all_bp)
    pep = [np.concatenate(p) for p in all_pep]
    lb = np.concatenate(all_lb)
    pel_arr = np.concatenate(all_pel)
    try:
        overall = float(roc_auc_score(lb, bp))
    except ValueError:
        overall = 0.5
    per_enz = {}
    for h, e in enumerate(ENZYMES):
        mask = pel_arr[:, h] >= 0
        labels = pel_arr[:, h][mask]
        probs = pep[h][mask]
        if len(np.unique(labels)) < 2:
            continue
        try:
            per_enz[e] = float(roc_auc_score(labels, probs))
        except ValueError:
            pass
    return {
        "overall_auroc": overall,
        "per_enzyme_auroc": per_enz,
        "binary_probs": bp,
        "per_enzyme_probs": pep,
        "labels_binary": lb,
        "per_enzyme_labels": pel_arr,
    }


def train_two_stage(model, train_ds, val_ds, hp, device, logger):
    train_loader = DataLoader(train_ds, batch_size=hp["bs"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=hp["bs"], shuffle=False, num_workers=0) if val_ds else None

    best_auroc = 0.0
    best_state = None

    optim1 = torch.optim.AdamW(model.parameters(), lr=hp["s1_lr"], weight_decay=hp["wd"])
    sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(optim1, T_max=hp["s1_epochs"])
    for epoch in range(hp["s1_epochs"]):
        loss = train_epoch(model, train_loader, optim1, device)
        sch1.step()
        if val_loader:
            m = evaluate(model, val_loader, device)
            if m["overall_auroc"] > best_auroc:
                best_auroc = m["overall_auroc"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if (epoch + 1) % 5 == 0:
                enz_str = ", ".join(f"{e}={m['per_enzyme_auroc'].get(e,0):.3f}" for e in ENZYMES)
                logger.info("  [S1] Ep %d: loss=%.4f, auroc=%.4f, %s", epoch + 1, loss, m["overall_auroc"], enz_str)
        else:
            if (epoch + 1) % 5 == 0:
                logger.info("  [S1] Ep %d: loss=%.4f", epoch + 1, loss)

    for p in model.shared_encoder.parameters():
        p.requires_grad = False
    adapter_params = list(model.binary_head.parameters())
    for enz in ENZYMES:
        adapter_params.extend(model.enzyme_adapters[enz].parameters())
    adapter_params.extend(model.enzyme_classifier.parameters())
    optim2 = torch.optim.AdamW(adapter_params, lr=hp["s2_lr"], weight_decay=hp["wd"])
    sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(optim2, T_max=hp["s2_epochs"])
    for epoch in range(hp["s2_epochs"]):
        loss = train_epoch(model, train_loader, optim2, device)
        sch2.step()
        if val_loader:
            m = evaluate(model, val_loader, device)
            if m["overall_auroc"] > best_auroc:
                best_auroc = m["overall_auroc"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if (epoch + 1) % 10 == 0:
                enz_str = ", ".join(f"{e}={m['per_enzyme_auroc'].get(e,0):.3f}" for e in ENZYMES)
                logger.info("  [S2] Ep %d: loss=%.4f, auroc=%.4f, %s", epoch + 1, loss, m["overall_auroc"], enz_str)
        else:
            if (epoch + 1) % 10 == 0:
                logger.info("  [S2] Ep %d: loss=%.4f", epoch + 1, loss)

    for p in model.parameters():
        p.requires_grad = True
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_state, best_auroc


# --------------------------------------------------------------------------- #
# Data loading (v4)
# --------------------------------------------------------------------------- #
def load_v4_data(variant: str, logger):
    paths = get_paths(variant)
    for k, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"v4 ({variant}) missing {k}: {p}")

    logger.info("Loading v4 (%s) data...", variant)
    df = pd.read_csv(paths["splits"])
    with open(paths["seqs"]) as f:
        seqs = json.load(f)

    # Loop CSV — set index on site_id (drop dup just in case)
    loop_df = pd.read_csv(paths["loop"]).drop_duplicates(subset=["site_id"]).set_index("site_id")

    # Structure cache (npz with delta_features)
    sc = np.load(paths["struct"], allow_pickle=True)
    struct_map = {sid: sc["delta_features"][i] for i, sid in enumerate(sc["site_ids"])}

    # RNA-FM npz: site_ids, pooled[N,640], pooled_edited[N,640]
    rnafm = np.load(paths["rnafm"], allow_pickle=True)
    rnafm_ids = rnafm["site_ids"]
    rnafm_pooled = rnafm["pooled"]
    rnafm_pooled_edited = rnafm["pooled_edited"]
    rnafm_idx = {str(sid): i for i, sid in enumerate(rnafm_ids)}

    site_ids_all = df["site_id"].astype(str).tolist()
    keep_mask = np.array([sid in seqs and sid in rnafm_idx for sid in site_ids_all])
    n_drop = int((~keep_mask).sum())
    if n_drop:
        logger.warning("  dropping %d rows missing seq or RNA-FM", n_drop)
    df = df[keep_mask].reset_index(drop=True)
    site_ids = df["site_id"].astype(str).tolist()
    n = len(site_ids)
    logger.info("  %d sites usable", n)

    # 40-d hand features
    hand = build_hand_features(site_ids, seqs, struct_map, loop_df)
    assert hand.shape == (n, D_HAND), hand.shape

    # RNA-FM original + delta
    rnafm_orig = np.zeros((n, D_RNAFM), dtype=np.float32)
    rnafm_delta = np.zeros((n, D_EDIT_DELTA), dtype=np.float32)
    for i, sid in enumerate(site_ids):
        ridx = rnafm_idx[sid]
        o = rnafm_pooled[ridx]
        e = rnafm_pooled_edited[ridx]
        rnafm_orig[i] = o
        rnafm_delta[i] = e - o
    logger.info("  RNA-FM coverage: %d/%d", n, n)

    X = np.concatenate([rnafm_orig, rnafm_delta, hand], axis=1).astype(np.float32)

    # MFE-only regime: zero struct_delta block [1304:1311]
    pre = X[:, STRUCT_DELTA_START:STRUCT_DELTA_END].copy()
    logger.info("  Pre-zero struct_delta |abs|.mean=%.6f", float(np.mean(np.abs(pre))))
    X[:, STRUCT_DELTA_START:STRUCT_DELTA_END] = 0.0
    logger.info("  Feature matrix: %s", X.shape)

    # Labels
    labels_binary = df["is_edited"].values.astype(np.float32)
    enz_strs = df["enzyme"].astype(str).fillna("Unknown").values
    labels_enzyme = np.array(
        [ENZYME_TO_IDX.get(e, ENZYME_TO_IDX["Unknown"]) for e in enz_strs],
        dtype=np.int64,
    )

    # Per-enzyme labels: 1.0 if (pos AND enzyme==e), 0.0 if (neg AND enzyme==e), else -1
    per_enzyme_labels = np.full((n, N_PER_ENZYME), -1, dtype=np.float32)
    for h, enz in enumerate(ENZYMES):
        ei = ENZYME_TO_IDX[enz]
        pos_mask = (labels_binary == 1) & (labels_enzyme == ei)
        neg_mask = (labels_binary == 0) & (labels_enzyme == ei)
        per_enzyme_labels[pos_mask, h] = 1.0
        per_enzyme_labels[neg_mask, h] = 0.0
        logger.info("  %s: pos=%d, neg=%d", enz, int(pos_mask.sum()), int(neg_mask.sum()))

    # trinuc per row (DNA letters from splits column)
    trinuc = df["trinuc"].astype(str).str.upper().str.replace("U", "T").values

    del rnafm, rnafm_pooled, rnafm_pooled_edited, sc
    gc.collect()
    return {
        "df": df,
        "X": X,
        "site_ids": site_ids,
        "labels_binary": labels_binary,
        "labels_enzyme": labels_enzyme,
        "per_enzyme_labels": per_enzyme_labels,
        "trinuc": trinuc,
    }


# --------------------------------------------------------------------------- #
# Cross-validation + final-model training
# --------------------------------------------------------------------------- #
def run_cv(data, hp, out_dir: Path, seed: int, device, logger):
    logger.info("\n" + "=" * 60)
    logger.info("5-fold CV (v4 MFE-only)")
    logger.info("=" * 60)
    X = data["X"]; lb = data["labels_binary"]; le = data["labels_enzyme"]; pel = data["per_enzyme_labels"]
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    fold_results = []

    # Out-of-fold predictions on the full set for bias diagnostics.
    oof_binary = np.full(len(X), np.nan, dtype=np.float32)
    oof_per_enz = np.full((len(X), N_PER_ENZYME), np.nan, dtype=np.float32)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, lb)):
        torch.manual_seed(seed + fold)
        np.random.seed(seed + fold)
        t0 = time.time()
        logger.info("\n--- Fold %d/%d (train=%d, val=%d) ---",
                    fold + 1, N_FOLDS, len(train_idx), len(val_idx))
        train_ds = SimpleDataset(X[train_idx], lb[train_idx], le[train_idx], pel[train_idx])
        val_ds = SimpleDataset(X[val_idx], lb[val_idx], le[val_idx], pel[val_idx])
        model = Phase3Model().to(device)
        if fold == 0:
            n_params = sum(p.numel() for p in model.parameters())
            logger.info("  Model params: %d", n_params)
        train_two_stage(model, train_ds, val_ds, hp, device, logger)
        val_loader = DataLoader(val_ds, batch_size=hp["bs"], shuffle=False, num_workers=0)
        m = evaluate(model, val_loader, device)
        # Store OOF preds
        oof_binary[val_idx] = m["binary_probs"]
        for h in range(N_PER_ENZYME):
            oof_per_enz[val_idx, h] = m["per_enzyme_probs"][h]
        elapsed = time.time() - t0
        logger.info("  Fold %d done in %.0fs, overall AUROC=%.4f",
                    fold + 1, elapsed, m["overall_auroc"])
        for enz in ENZYMES:
            logger.info("    %s: %.3f", enz, m["per_enzyme_auroc"].get(enz, 0))
        fold_results.append(m)
        torch.save(model.state_dict(), out_dir / f"neural_fold{fold}.pt")
        del model, train_ds, val_ds
        if device.type == "mps":
            torch.mps.empty_cache()
        gc.collect()

    summary = {
        "overall_auroc": {
            "mean": float(np.mean([r["overall_auroc"] for r in fold_results])),
            "std": float(np.std([r["overall_auroc"] for r in fold_results])),
            "folds": [float(r["overall_auroc"]) for r in fold_results],
        },
        "per_enzyme_auroc": {},
    }
    for enz in ENZYMES:
        vals = [r["per_enzyme_auroc"].get(enz, 0.0) for r in fold_results]
        summary["per_enzyme_auroc"][enz] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "folds": [float(v) for v in vals],
        }
    logger.info("\n--- CV Summary ---")
    logger.info("Overall AUROC: %.4f +/- %.4f",
                summary["overall_auroc"]["mean"], summary["overall_auroc"]["std"])
    for enz in ENZYMES:
        s = summary["per_enzyme_auroc"][enz]
        logger.info("  %s: %.3f +/- %.3f", enz, s["mean"], s["std"])
    with open(out_dir / "cv_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save OOF predictions for diagnostics
    np.savez_compressed(
        out_dir / "oof_predictions.npz",
        site_ids=np.asarray(data["site_ids"], dtype=object),
        trinuc=data["trinuc"],
        labels_binary=lb,
        labels_enzyme=le,
        per_enzyme_labels=pel,
        oof_binary=oof_binary,
        oof_per_enzyme=oof_per_enz,
    )
    logger.info("Saved OOF predictions: %s", out_dir / "oof_predictions.npz")

    return summary, oof_binary, oof_per_enz


def train_full_model(data, hp, out_dir: Path, ckpt_name: str, seed: int, device, logger):
    logger.info("\n" + "=" * 60)
    logger.info("Training full model on all v4 data")
    logger.info("=" * 60)
    torch.manual_seed(seed)
    np.random.seed(seed)
    full_ds = SimpleDataset(data["X"], data["labels_binary"], data["labels_enzyme"], data["per_enzyme_labels"])
    model = Phase3Model().to(device)
    train_two_stage(model, full_ds, val_ds=None, hp=hp, device=device, logger=logger)
    save_path = out_dir / ckpt_name
    torch.save(model.state_dict(), save_path)
    logger.info("Saved full model to %s", save_path)
    return save_path


# --------------------------------------------------------------------------- #
# Bias diagnostic: trinuc-stratified score summary on OOF predictions
# --------------------------------------------------------------------------- #
def bias_diagnostic(data, oof_binary, oof_per_enz, out_dir: Path, variant: str, logger):
    """Compute mean OOF predicted probability per NCN trinucleotide bin for each head.

    Uses out-of-fold predictions on the held-out validation rows; this is the gold-
    standard estimate of how the model's score is distributed across motifs in the
    population represented by the (positives + matched-negatives) training set.

    Output CSV columns:
        trinuc, n_positions, n_pos, n_neg,
        mean_score_binary, mean_score_A3A, mean_score_A3B, mean_score_A3G, mean_score_A3A_A3G,
        mean_score_binary_neg, mean_score_binary_pos
    """
    trinuc = data["trinuc"]
    lb = data["labels_binary"]
    df = pd.DataFrame({
        "trinuc": trinuc,
        "is_edited": lb.astype(int),
        "binary": oof_binary,
    })
    for h, enz in enumerate(ENZYMES):
        df[f"score_{enz}"] = oof_per_enz[:, h]

    # 16 NCN bins (DNA letters); we may have small bins for non-NCN edge cases — keep all.
    rows = []
    for tn, sub in df.groupby("trinuc"):
        n = len(sub)
        n_pos = int((sub["is_edited"] == 1).sum())
        n_neg = int((sub["is_edited"] == 0).sum())
        row = {
            "trinuc": tn,
            "n_positions": n,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "mean_score_binary": float(np.nanmean(sub["binary"])),
            "mean_score_binary_pos": float(np.nanmean(sub.loc[sub["is_edited"] == 1, "binary"])) if n_pos else float("nan"),
            "mean_score_binary_neg": float(np.nanmean(sub.loc[sub["is_edited"] == 0, "binary"])) if n_neg else float("nan"),
        }
        for enz in ENZYMES:
            row[f"mean_score_{enz}"] = float(np.nanmean(sub[f"score_{enz}"]))
        rows.append(row)
    bias_df = pd.DataFrame(rows).sort_values("trinuc").reset_index(drop=True)
    out_csv = out_dir / f"bias_diagnostic_{variant}.csv"
    bias_df.to_csv(out_csv, index=False)
    logger.info("Saved bias diagnostic: %s", out_csv)

    # ---- TCW vs non-TCW false-positive rate at p>=0.5 (binary head) ----
    is_tcw = pd.Series(trinuc).isin(["TCA", "TCT"]).values
    neg_mask = (lb == 0)
    fp_thresh = 0.5
    diag_extra = {}
    for grp_name, grp_mask in [("TCW", is_tcw), ("non_TCW", ~is_tcw)]:
        m = neg_mask & grp_mask & ~np.isnan(oof_binary)
        n = int(m.sum())
        fp = int((oof_binary[m] >= fp_thresh).sum()) if n else 0
        rate = fp / n if n else float("nan")
        diag_extra[f"fp_rate_{grp_name}_negatives_p>=0.5"] = {
            "n_negatives": n,
            "n_false_positives": fp,
            "rate": rate,
        }
        logger.info("  FP rate (p>=0.5) on %s negatives: %d/%d = %.3f", grp_name, fp, n, rate)

    # Mean score on negatives stratified TCW vs non-TCW
    for grp_name, grp_mask in [("TCW", is_tcw), ("non_TCW", ~is_tcw)]:
        m_neg = neg_mask & grp_mask & ~np.isnan(oof_binary)
        m_pos = (lb == 1) & grp_mask & ~np.isnan(oof_binary)
        diag_extra[f"mean_score_negatives_{grp_name}"] = float(oof_binary[m_neg].mean()) if m_neg.any() else float("nan")
        diag_extra[f"mean_score_positives_{grp_name}"] = float(oof_binary[m_pos].mean()) if m_pos.any() else float("nan")

    # AUROC per top-5 most populated trinuc (binary head)
    top5 = bias_df.sort_values("n_positions", ascending=False).head(5)["trinuc"].tolist()
    per_trinuc_auroc = {}
    for tn in top5:
        m = (trinuc == tn) & ~np.isnan(oof_binary)
        if m.sum() < 5:
            continue
        ll = lb[m]
        pp = oof_binary[m]
        if len(np.unique(ll)) < 2:
            continue
        try:
            per_trinuc_auroc[tn] = {"n": int(m.sum()), "auroc": float(roc_auc_score(ll, pp))}
        except ValueError:
            pass
    diag_extra["per_trinuc_auroc_top5"] = per_trinuc_auroc

    # Variance of mean score across NCN bins (excluding bins with <10 rows)
    big = bias_df[bias_df["n_positions"] >= 10]
    diag_extra["bin_score_variance_binary"] = float(big["mean_score_binary"].var()) if len(big) else float("nan")
    diag_extra["bin_score_range_binary"] = [
        float(big["mean_score_binary"].min()) if len(big) else float("nan"),
        float(big["mean_score_binary"].max()) if len(big) else float("nan"),
    ]
    diag_extra["n_bins_used"] = int(len(big))

    # Anti-TCW polarity check: median score of TCW negs vs non-TCW negs
    # Anti-TCW polarity = TCW score < non-TCW score on negatives (the v3 bug).
    tcw_neg_score = diag_extra.get("mean_score_negatives_TCW", float("nan"))
    nontcw_neg_score = diag_extra.get("mean_score_negatives_non_TCW", float("nan"))
    diag_extra["anti_TCW_polarity_present"] = bool(
        not (np.isnan(tcw_neg_score) or np.isnan(nontcw_neg_score))
        and tcw_neg_score < nontcw_neg_score
    )

    with open(out_dir / f"bias_diagnostic_{variant}_summary.json", "w") as f:
        json.dump(diag_extra, f, indent=2)
    logger.info("Saved bias diagnostic summary: %s", out_dir / f"bias_diagnostic_{variant}_summary.json")
    logger.info("  bin_score_variance_binary=%.4f, anti_TCW_polarity_present=%s",
                diag_extra["bin_score_variance_binary"],
                diag_extra["anti_TCW_polarity_present"])

    return bias_df, diag_extra


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", required=True, choices=["cancer_matched", "cds_unbiased"])
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--epochs", type=int, default=None,
                   help="Override total epoch budget (split 1/3 stage1, 2/3 stage2). "
                        "Default: stage1=10, stage2=20.")
    p.add_argument("--lr", type=float, default=DEFAULT_STAGE1_LR,
                   help="Stage-1 LR (stage-2 LR = lr/2).")
    p.add_argument("--bs", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--struct-delta-zero", action="store_true",
                   help="Always zero struct_delta (default: ON for training to match v3 phase3_mfe_only).")
    p.add_argument("--no-cv", action="store_true", help="Skip CV (debug only).")
    p.add_argument("--no-final", action="store_true", help="Skip final-model training (debug only).")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    log_file = out_dir / "run.log"
    # Reset logging to write into this run's log file
    for h in list(logging.getLogger().handlers):
        logging.getLogger().removeHandler(h)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, mode="w")],
    )
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("v4 phase3 training: variant=%s", args.variant)
    logger.info("out_dir=%s", out_dir)
    logger.info("device=%s", DEVICE)
    logger.info("seed=%d", args.seed)
    logger.info("=" * 60)

    if args.epochs is not None:
        s1 = max(1, args.epochs // 3)
        s2 = args.epochs - s1
    else:
        s1, s2 = DEFAULT_STAGE1_EPOCHS, DEFAULT_STAGE2_EPOCHS
    hp = {
        "s1_lr": args.lr,
        "s2_lr": args.lr / 2.0,
        "s1_epochs": s1,
        "s2_epochs": s2,
        "wd": DEFAULT_WEIGHT_DECAY,
        "bs": args.bs,
    }
    logger.info("Hyperparams: %s", hp)

    t0 = time.time()
    data = load_v4_data(args.variant, logger)

    # Run CV
    if not args.no_cv:
        cv_summary, oof_binary, oof_per_enz = run_cv(data, hp, out_dir, args.seed, DEVICE, logger)
        # Bias diagnostic on OOF preds
        bias_diagnostic(data, oof_binary, oof_per_enz, out_dir, args.variant, logger)
    else:
        logger.warning("--no-cv: skipping CV and bias diagnostic")

    # Train full model
    if not args.no_final:
        ckpt_name = "phase3_v4_cancer.pt" if args.variant == "cancer_matched" else "phase3_v4_cds.pt"
        train_full_model(data, hp, out_dir, ckpt_name, args.seed, DEVICE, logger)

    logger.info("Total runtime: %.1f min", (time.time() - t0) / 60.0)


if __name__ == "__main__":
    main()
