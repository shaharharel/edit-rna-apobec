#!/usr/bin/env python
"""Unified multi-enzyme network v1: shared backbone + enzyme-specific heads.

Architecture:
  Shared Encoder: RNA-FM pooled embeddings (640-dim) + edit diff (640-dim) + hand features (40-dim)
                  → shared MLP backbone (1320 → 512 → 256)

  Binary Head: shared_repr → MLP → is_edited (all sites)
  Enzyme Head: shared_repr → MLP → 6-class enzyme (positives only, mask negatives)

Training:
  Phase 1 (20 epochs): Binary only — learn what makes a site edited
  Phase 2 (30 epochs): Binary + Enzyme jointly — learn enzyme specificity

  Loss: BCE(binary) + α * CE(enzyme, masked for negatives)

Evaluation:
  - Per-enzyme binary AUROC (train unified, evaluate per-enzyme)
  - 6-class enzyme classification (macro F1, confusion matrix)
  - Compare to per-enzyme models (does shared learning help?)

Usage:
    conda run -n quris python experiments/multi_enzyme/exp_unified_network_v1.py
"""

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
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    confusion_matrix, classification_report,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.apobec_feature_extraction import (
    extract_motif_features, extract_loop_features,
    extract_structure_delta_features,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_ME_DIR = PROJECT_ROOT / "data/processed/multi_enzyme"
SPLITS_CSV = _ME_DIR / "splits_multi_enzyme_v3_with_negatives.csv"
SEQS_JSON = _ME_DIR / "multi_enzyme_sequences_v3_with_negatives.json"
EMB_DIR = _ME_DIR / "embeddings"
EMB_POOLED = EMB_DIR / "rnafm_pooled_v3.pt"
EMB_POOLED_ED = EMB_DIR / "rnafm_pooled_edited_v3.pt"
STRUCT_CACHE = _ME_DIR / "structure_cache_multi_enzyme_v3.npz"
LOOP_CSV = _ME_DIR / "loop_position_per_site_v3.csv"
OUTPUT_DIR = Path(__file__).parent / "outputs" / "unified_network_v1"

SEED = 42
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
ENZYME_CLASSES = ["A3A", "A3A_A3G", "A3B", "A3G", "Neither", "Unknown"]
N_ENZYMES = len(ENZYME_CLASSES)


class UnifiedDataset(Dataset):
    """Dataset for unified multi-enzyme training."""

    def __init__(self, emb_orig, emb_edited, hand_features, binary_labels, enzyme_labels):
        self.emb_orig = torch.FloatTensor(emb_orig)
        self.emb_edited = torch.FloatTensor(emb_edited)
        self.hand = torch.FloatTensor(hand_features)
        self.binary = torch.FloatTensor(binary_labels)
        self.enzyme = torch.LongTensor(enzyme_labels)  # -1 for negatives

    def __len__(self):
        return len(self.binary)

    def __getitem__(self, idx):
        return (self.emb_orig[idx], self.emb_edited[idx], self.hand[idx],
                self.binary[idx], self.enzyme[idx])


class UnifiedNetwork(nn.Module):
    """Shared backbone + binary head + enzyme head."""

    def __init__(self, d_emb=640, d_hand=40, d_shared=256, n_enzymes=N_ENZYMES, dropout=0.3):
        super().__init__()

        # Shared backbone: processes edit diff + context + hand features
        d_input = d_emb + d_emb + d_hand  # orig + diff + hand = 1320
        self.shared = nn.Sequential(
            nn.Linear(d_input, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(dropout),
            nn.Linear(512, d_shared),
            nn.GELU(),
            nn.LayerNorm(d_shared),
            nn.Dropout(dropout),
        )

        # Binary head: is this site edited?
        self.binary_head = nn.Sequential(
            nn.Linear(d_shared, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

        # Enzyme head: which enzyme? (only for positives)
        self.enzyme_head = nn.Sequential(
            nn.Linear(d_shared, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_enzymes),
        )

    def forward(self, emb_orig, emb_edited, hand):
        diff = emb_edited - emb_orig
        x = torch.cat([emb_orig, diff, hand], dim=-1)
        shared = self.shared(x)
        binary_logit = self.binary_head(shared).squeeze(-1)
        enzyme_logits = self.enzyme_head(shared)
        return binary_logit, enzyme_logits, shared


class UnifiedNetworkV2(nn.Module):
    """V2: Separate edit embedding + context pathway with gated fusion."""

    def __init__(self, d_emb=640, d_hand=40, d_shared=256, n_enzymes=N_ENZYMES, dropout=0.3):
        super().__init__()

        # Edit pathway: captures what changed
        self.edit_encoder = nn.Sequential(
            nn.Linear(d_emb, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.GELU(),
        )

        # Context pathway: captures where it happened
        self.context_encoder = nn.Sequential(
            nn.Linear(d_emb, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.GELU(),
        )

        # Hand feature pathway: interpretable features
        self.hand_encoder = nn.Sequential(
            nn.Linear(d_hand, 64), nn.GELU(), nn.Dropout(dropout),
        )

        # Gated fusion
        d_fused = 128 + 128 + 64  # 320
        self.gate = nn.Sequential(
            nn.Linear(d_fused, d_fused),
            nn.Sigmoid(),
        )

        # Shared representation
        self.shared = nn.Sequential(
            nn.Linear(d_fused, d_shared),
            nn.GELU(),
            nn.LayerNorm(d_shared),
            nn.Dropout(dropout),
        )

        # Heads
        self.binary_head = nn.Sequential(
            nn.Linear(d_shared, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 1),
        )
        self.enzyme_head = nn.Sequential(
            nn.Linear(d_shared, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, n_enzymes),
        )

    def forward(self, emb_orig, emb_edited, hand):
        edit_feat = self.edit_encoder(emb_edited - emb_orig)
        ctx_feat = self.context_encoder(emb_orig)
        hand_feat = self.hand_encoder(hand)

        concat = torch.cat([edit_feat, ctx_feat, hand_feat], dim=-1)
        gate = self.gate(concat)
        fused = concat * gate

        shared = self.shared(fused)
        binary_logit = self.binary_head(shared).squeeze(-1)
        enzyme_logits = self.enzyme_head(shared)
        return binary_logit, enzyme_logits, shared


def train_unified(model, train_loader, optimizer, phase, enzyme_weight=1.0):
    """Train one epoch."""
    model.train()
    total_loss = 0
    n_samples = 0

    for emb_o, emb_e, hand, binary, enzyme in train_loader:
        emb_o, emb_e, hand = emb_o.to(DEVICE), emb_e.to(DEVICE), hand.to(DEVICE)
        binary, enzyme = binary.to(DEVICE), enzyme.to(DEVICE)

        bin_logit, enz_logits, _ = model(emb_o, emb_e, hand)

        # Binary loss (all sites)
        loss_bin = F.binary_cross_entropy_with_logits(bin_logit, binary)

        # Enzyme loss (positives only, enzyme >= 0)
        loss_enz = torch.tensor(0.0, device=DEVICE)
        if phase >= 2:
            pos_mask = enzyme >= 0
            if pos_mask.sum() > 0:
                loss_enz = F.cross_entropy(enz_logits[pos_mask], enzyme[pos_mask])

        loss = loss_bin + enzyme_weight * loss_enz

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(binary)
        n_samples += len(binary)

    return total_loss / n_samples


def evaluate_unified(model, loader, le):
    """Evaluate: per-enzyme binary AUROC + enzyme classification."""
    model.eval()
    all_binary_true, all_binary_score = [], []
    all_enzyme_true, all_enzyme_pred = [], []
    all_enzymes_for_binary = []  # to compute per-enzyme AUROC

    with torch.no_grad():
        for emb_o, emb_e, hand, binary, enzyme in loader:
            emb_o, emb_e, hand = emb_o.to(DEVICE), emb_e.to(DEVICE), hand.to(DEVICE)
            bin_logit, enz_logits, _ = model(emb_o, emb_e, hand)

            bin_score = torch.sigmoid(bin_logit).cpu().numpy()
            all_binary_true.extend(binary.numpy().tolist())
            all_binary_score.extend(bin_score.tolist())
            all_enzymes_for_binary.extend(enzyme.numpy().tolist())

            # Enzyme predictions for positives
            pos_mask = enzyme >= 0
            if pos_mask.sum() > 0:
                enz_pred = enz_logits[pos_mask].argmax(dim=-1).cpu().numpy()
                all_enzyme_true.extend(enzyme[pos_mask].numpy().tolist())
                all_enzyme_pred.extend(enz_pred.tolist())

    # Overall binary AUROC
    overall_auroc = roc_auc_score(all_binary_true, all_binary_score)

    # Per-enzyme binary AUROC
    per_enzyme_auroc = {}
    enzymes_arr = np.array(all_enzymes_for_binary)
    binary_arr = np.array(all_binary_true)
    score_arr = np.array(all_binary_score)

    for enz_idx, enz_name in enumerate(le.classes_):
        # Sites from this enzyme + all negatives
        mask = (enzymes_arr == enz_idx) | (enzymes_arr == -1)
        if mask.sum() > 0 and len(np.unique(binary_arr[mask])) == 2:
            per_enzyme_auroc[enz_name] = float(roc_auc_score(binary_arr[mask], score_arr[mask]))

    # Enzyme classification
    enzyme_acc = float(np.mean(np.array(all_enzyme_true) == np.array(all_enzyme_pred))) if all_enzyme_true else 0

    return {
        "overall_auroc": float(overall_auroc),
        "per_enzyme_auroc": per_enzyme_auroc,
        "enzyme_accuracy": enzyme_acc,
        "enzyme_true": all_enzyme_true,
        "enzyme_pred": all_enzyme_pred,
    }


def main():
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Device: %s", DEVICE)

    # Load data
    df = pd.read_csv(SPLITS_CSV)
    with open(SEQS_JSON) as f:
        seqs = json.load(f)

    emb_orig = torch.load(EMB_POOLED, map_location="cpu", weights_only=False)
    emb_edited = torch.load(EMB_POOLED_ED, map_location="cpu", weights_only=False)
    logger.info("Embeddings: %d orig, %d edited", len(emb_orig), len(emb_edited))

    struct_delta = {}
    if STRUCT_CACHE.exists():
        data = np.load(str(STRUCT_CACHE), allow_pickle=True)
        for i, sid in enumerate(data["site_ids"].astype(str)):
            struct_delta[sid] = data["delta_features"][i]
        del data; gc.collect()

    loop_df = pd.read_csv(LOOP_CSV)
    loop_df["site_id"] = loop_df["site_id"].astype(str)
    loop_df = loop_df.set_index("site_id")

    # Filter to sites with embeddings
    site_ids = df["site_id"].values
    has_emb = np.array([str(sid) in emb_orig and str(sid) in emb_edited for sid in site_ids])
    df = df[has_emb].copy().reset_index(drop=True)
    site_ids = df["site_id"].values
    logger.info("Sites with embeddings: %d/%d", len(df), has_emb.sum())

    # Build features
    orig_embs = np.array([emb_orig[str(sid)].numpy() for sid in site_ids])
    edited_embs = np.array([emb_edited[str(sid)].numpy() for sid in site_ids])

    motif = extract_motif_features(seqs, list(site_ids))
    struct = extract_structure_delta_features(struct_delta, list(site_ids))
    loop = extract_loop_features(loop_df, list(site_ids))
    hand_40 = np.nan_to_num(np.concatenate([motif, struct, loop], axis=1), nan=0.0)

    # Labels
    binary_labels = df["is_edited"].values.astype(float)
    le = LabelEncoder()
    le.fit(ENZYME_CLASSES)
    enzyme_labels = np.full(len(df), -1, dtype=int)  # -1 for negatives
    pos_mask = df["is_edited"] == 1
    enzyme_labels[pos_mask] = le.transform(df.loc[pos_mask, "enzyme"].values)

    logger.info("Binary: %d pos, %d neg", pos_mask.sum(), (~pos_mask).sum())
    logger.info("Enzymes: %s", dict(zip(le.classes_, np.bincount(enzyme_labels[enzyme_labels >= 0]))))

    # 5-fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    # Stratify by enzyme (using enzyme_labels, with -1 as its own class)
    strat_labels = enzyme_labels.copy()
    strat_labels[strat_labels == -1] = N_ENZYMES  # negatives as separate class

    all_results = {"v1": [], "v2": []}

    for model_name, model_cls in [("v1", UnifiedNetwork), ("v2", UnifiedNetworkV2)]:
        logger.info("\n" + "=" * 60)
        logger.info("MODEL: %s", model_name.upper())
        logger.info("=" * 60)

        fold_results = []

        for fold_i, (train_idx, test_idx) in enumerate(skf.split(hand_40, strat_labels)):
            torch.manual_seed(SEED + fold_i)
            logger.info("\n--- Fold %d ---", fold_i + 1)

            train_ds = UnifiedDataset(
                orig_embs[train_idx], edited_embs[train_idx], hand_40[train_idx],
                binary_labels[train_idx], enzyme_labels[train_idx],
            )
            test_ds = UnifiedDataset(
                orig_embs[test_idx], edited_embs[test_idx], hand_40[test_idx],
                binary_labels[test_idx], enzyme_labels[test_idx],
            )
            train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
            test_loader = DataLoader(test_ds, batch_size=128)

            model = model_cls().to(DEVICE)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

            # Phase 1: Binary only (20 epochs)
            sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
            for epoch in range(20):
                train_unified(model, train_loader, optimizer, phase=1)
                sched1.step()

            # Eval after phase 1
            r1 = evaluate_unified(model, test_loader, le)
            logger.info("  Phase 1: overall AUROC=%.3f", r1["overall_auroc"])

            # Phase 2: Binary + Enzyme (30 epochs)
            optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
            sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
            best_auroc = r1["overall_auroc"]
            patience = 0
            for epoch in range(30):
                train_unified(model, train_loader, optimizer, phase=2, enzyme_weight=0.5)
                sched2.step()
                r = evaluate_unified(model, test_loader, le)
                if r["overall_auroc"] > best_auroc:
                    best_auroc = r["overall_auroc"]
                    patience = 0
                else:
                    patience += 1
                if patience >= 10:
                    break

            # Final eval
            final = evaluate_unified(model, test_loader, le)
            logger.info("  Phase 2: overall AUROC=%.3f, enzyme acc=%.3f",
                         final["overall_auroc"], final["enzyme_accuracy"])
            for enz, auroc in final["per_enzyme_auroc"].items():
                logger.info("    %s binary AUROC=%.3f", enz, auroc)

            fold_results.append({
                "overall_auroc": final["overall_auroc"],
                "per_enzyme_auroc": final["per_enzyme_auroc"],
                "enzyme_accuracy": final["enzyme_accuracy"],
            })

            del model; gc.collect()
            if DEVICE == "mps":
                torch.mps.empty_cache()

        all_results[model_name] = fold_results

        # Summary for this model
        mean_auroc = np.mean([r["overall_auroc"] for r in fold_results])
        mean_enz_acc = np.mean([r["enzyme_accuracy"] for r in fold_results])
        logger.info("\n%s Summary: AUROC=%.3f±%.3f, Enzyme Acc=%.3f",
                     model_name.upper(), mean_auroc,
                     np.std([r["overall_auroc"] for r in fold_results]),
                     mean_enz_acc)

        # Per-enzyme AUROC means
        for enz in le.classes_:
            vals = [r["per_enzyme_auroc"].get(enz, 0) for r in fold_results]
            logger.info("  %s: AUROC=%.3f±%.3f", enz, np.mean(vals), np.std(vals))

    # Save
    with open(OUTPUT_DIR / "unified_network_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print comparison
    print("\n" + "=" * 80)
    print("UNIFIED NETWORK vs PER-ENZYME COMPARISON")
    print("=" * 80)

    per_enzyme_ref = {"A3A": 0.880, "A3B": 0.971, "A3G": 0.893, "A3A_A3G": 0.935, "Neither": 0.829}

    for model_name in ["v1", "v2"]:
        print(f"\n--- {model_name.upper()} ---")
        print(f"{'Enzyme':>10} {'Unified':>10} {'PerEnzyme':>10} {'Delta':>10}")
        for enz in le.classes_:
            vals = [r["per_enzyme_auroc"].get(enz, 0) for r in all_results[model_name]]
            unified = np.mean(vals)
            ref = per_enzyme_ref.get(enz, 0)
            delta = unified - ref if ref > 0 else 0
            print(f"{enz:>10} {unified:10.3f} {ref:10.3f} {delta:+10.3f}")

    logger.info("\nTotal time: %.0fs", time.time() - t0)


if __name__ == "__main__":
    main()
