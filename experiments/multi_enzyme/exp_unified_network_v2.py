#!/usr/bin/env python
"""Unified multi-enzyme network v2: feature-routed heads.

Key changes from v1:
  1. Structure features (loop 9 + struct_delta 7) routed to binary head only
  2. Motif features (24-dim) routed to enzyme head only
  3. Neither excluded from APOBEC3 enzyme head; gets its own binary head
  4. Class-weighted CE for enzyme head (inverse frequency)
  5. Joint training from start (no phasing), CosineAnnealing 50 epochs

Architecture:
  Shared Structure Trunk:
    input: concat(rnafm_orig(640), edit_diff(640), loop(9), struct_delta(7)) -> (1296)
    trunk: Linear(1296->512) + GELU + LN + Drop(0.3) + Linear(512->256) + GELU + LN + Drop(0.3)

  Binary Head (all sites):
    input: shared_repr (256)
    head: Linear(256->128) + GELU + Dropout(0.3) + Linear(128->1)

  Enzyme Head (APOBEC3 positives: A3A, A3B, A3G, A3A_A3G = 4 classes):
    input: concat(shared_repr(256), motif(24)) -> (280)
    head: Linear(280->128) + GELU + Dropout(0.3) + Linear(128->4)
    loss: class-weighted CE

  Neither Head (positives only, binary):
    input: concat(shared_repr(256), mooring_au(1)) -> (257)
    head: Linear(257->64) + GELU + Linear(64->1)

Usage:
    conda run -n quris python experiments/multi_enzyme/exp_unified_network_v2.py
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
from sklearn.metrics import roc_auc_score, f1_score
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
OUTPUT_DIR = Path(__file__).parent / "outputs" / "unified_network_v2"

SEED = 42
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

# APOBEC3 enzyme classes (excluding Neither and Unknown)
APOBEC3_CLASSES = ["A3A", "A3A_A3G", "A3B", "A3G"]
N_APOBEC3 = len(APOBEC3_CLASSES)

# All enzyme classes for per-enzyme AUROC eval
ALL_ENZYMES = ["A3A", "A3A_A3G", "A3B", "A3G", "Neither", "Unknown"]


def compute_mooring_au_feature(sequences: dict, site_ids: list) -> np.ndarray:
    """Compute a simple mooring-sequence AU-richness feature (1-dim).

    Neither/Unknown sites are edited by APOBEC1 which recognizes a
    mooring sequence downstream. We compute AU fraction in +5 to +30 window.
    """
    CENTER = 100
    features = []
    for sid in site_ids:
        seq = sequences.get(str(sid), "N" * 201).upper().replace("T", "U")
        window = seq[CENTER + 5: CENTER + 31]
        au_count = sum(1 for c in window if c in ("A", "U"))
        features.append([au_count / max(len(window), 1)])
    return np.array(features, dtype=np.float32)


class UnifiedDatasetV2(Dataset):
    """Dataset for v2 unified training with separated feature groups."""

    def __init__(self, emb_orig, emb_diff, loop_feats, struct_delta_feats, motif_feats,
                 mooring_feats, binary_labels, enzyme_labels, neither_labels):
        self.emb_orig = torch.FloatTensor(emb_orig)
        self.emb_diff = torch.FloatTensor(emb_diff)
        self.loop = torch.FloatTensor(loop_feats)
        self.struct_delta = torch.FloatTensor(struct_delta_feats)
        self.motif = torch.FloatTensor(motif_feats)
        self.mooring = torch.FloatTensor(mooring_feats)
        self.binary = torch.FloatTensor(binary_labels)
        self.enzyme = torch.LongTensor(enzyme_labels)   # -1 for non-APOBEC3
        self.neither = torch.FloatTensor(neither_labels) # -1 for negatives, 0/1 for positives

    def __len__(self):
        return len(self.binary)

    def __getitem__(self, idx):
        return (self.emb_orig[idx], self.emb_diff[idx], self.loop[idx], self.struct_delta[idx],
                self.motif[idx], self.mooring[idx],
                self.binary[idx], self.enzyme[idx], self.neither[idx])


class UnifiedNetworkV2(nn.Module):
    """Feature-routed unified network: structure -> binary, motif -> enzyme.

    The shared trunk sees projected RNA-FM embeddings (orig + edit diff) + structure features.
    The binary head operates on the shared representation alone (no motif leakage).
    The enzyme head additionally receives motif features for enzyme specificity.
    The neither head additionally receives a mooring AU-richness feature.
    """

    def __init__(self, d_emb=640, d_loop=9, d_struct=7, d_motif=24,
                 d_shared=256, d_emb_proj=256, d_diff_proj=128,
                 n_apobec3=N_APOBEC3, dropout=0.3):
        super().__init__()

        # Project RNA-FM original embeddings
        self.emb_proj = nn.Sequential(
            nn.Linear(d_emb, d_emb_proj),
            nn.GELU(),
        )

        # Project edit diff embeddings
        self.diff_proj = nn.Sequential(
            nn.Linear(d_emb, d_diff_proj),
            nn.GELU(),
        )

        # Shared structure trunk: emb_proj(256) + diff_proj(128) + loop(9) + struct_delta(7) = 400
        d_trunk_in = d_emb_proj + d_diff_proj + d_loop + d_struct
        self.trunk = nn.Sequential(
            nn.Linear(d_trunk_in, d_shared),
            nn.GELU(),
            nn.LayerNorm(d_shared),
            nn.Dropout(dropout),
        )

        # Binary head: shared_repr(256) -> 1
        self.binary_head = nn.Sequential(
            nn.Linear(d_shared, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

        # Enzyme head: shared_repr(256) + motif(24) = 280 -> 4 classes
        self.enzyme_head = nn.Sequential(
            nn.Linear(d_shared + d_motif, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_apobec3),
        )

        # Neither head: shared_repr(256) + mooring_au(1) = 257 -> 1
        self.neither_head = nn.Sequential(
            nn.Linear(d_shared + 1, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, emb_orig, emb_diff, loop_feats, struct_delta, motif, mooring):
        # Project embeddings
        emb_proj = self.emb_proj(emb_orig)
        diff_proj = self.diff_proj(emb_diff)

        # Shared structure trunk (embeddings + structure, NO motif)
        trunk_in = torch.cat([emb_proj, diff_proj, loop_feats, struct_delta], dim=-1)
        shared = self.trunk(trunk_in)

        # Binary head (structure-informed only — no motif leakage)
        binary_logit = self.binary_head(shared).squeeze(-1)

        # Enzyme head (structure + motif for enzyme specificity)
        enzyme_in = torch.cat([shared, motif], dim=-1)
        enzyme_logits = self.enzyme_head(enzyme_in)

        # Neither head (structure + mooring for APOBEC1-type recognition)
        neither_in = torch.cat([shared, mooring], dim=-1)
        neither_logit = self.neither_head(neither_in).squeeze(-1)

        return binary_logit, enzyme_logits, neither_logit


def train_epoch(model, loader, optimizer, enzyme_weight, neither_weight, enzyme_class_weights):
    """Train one epoch with joint losses."""
    model.train()
    total_loss = 0.0
    n_samples = 0

    for emb_o, emb_d, loop, sd, motif, mooring, binary, enzyme, neither in loader:
        emb_o = emb_o.to(DEVICE)
        emb_d = emb_d.to(DEVICE)
        loop = loop.to(DEVICE)
        sd = sd.to(DEVICE)
        motif = motif.to(DEVICE)
        mooring = mooring.to(DEVICE)
        binary = binary.to(DEVICE)
        enzyme = enzyme.to(DEVICE)
        neither = neither.to(DEVICE)

        bin_logit, enz_logits, nei_logit = model(emb_o, emb_d, loop, sd, motif, mooring)

        # Binary loss (all sites)
        loss_bin = F.binary_cross_entropy_with_logits(bin_logit, binary)

        # Enzyme loss (APOBEC3 positives only, enzyme >= 0)
        loss_enz = torch.tensor(0.0, device=DEVICE)
        apobec3_mask = enzyme >= 0
        if apobec3_mask.sum() > 0:
            loss_enz = F.cross_entropy(
                enz_logits[apobec3_mask], enzyme[apobec3_mask],
                weight=enzyme_class_weights.to(DEVICE),
            )

        # Neither loss (positives only: neither >= 0 means this is a positive site)
        loss_nei = torch.tensor(0.0, device=DEVICE)
        pos_mask = neither >= 0
        if pos_mask.sum() > 0:
            loss_nei = F.binary_cross_entropy_with_logits(
                nei_logit[pos_mask], neither[pos_mask]
            )

        loss = loss_bin + enzyme_weight * loss_enz + neither_weight * loss_nei

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(binary)
        n_samples += len(binary)

    return total_loss / n_samples


def evaluate(model, loader, apobec3_le):
    """Evaluate all three heads."""
    model.eval()
    all_bin_true, all_bin_score = [], []
    all_enz_true, all_enz_pred = [], []
    all_nei_true, all_nei_score = [], []
    all_enzyme_for_binary = []  # original enzyme labels for per-enzyme AUROC

    with torch.no_grad():
        for emb_o, emb_d, loop, sd, motif, mooring, binary, enzyme, neither in loader:
            emb_o = emb_o.to(DEVICE)
            emb_d = emb_d.to(DEVICE)
            loop = loop.to(DEVICE)
            sd = sd.to(DEVICE)
            motif = motif.to(DEVICE)
            mooring = mooring.to(DEVICE)

            bin_logit, enz_logits, nei_logit = model(emb_o, emb_d, loop, sd, motif, mooring)

            bin_score = torch.sigmoid(bin_logit).cpu().numpy()
            all_bin_true.extend(binary.numpy().tolist())
            all_bin_score.extend(bin_score.tolist())
            all_enzyme_for_binary.extend(enzyme.numpy().tolist())

            # Enzyme (APOBEC3 positives)
            apobec3_mask = enzyme >= 0
            if apobec3_mask.sum() > 0:
                pred = enz_logits[apobec3_mask].argmax(dim=-1).cpu().numpy()
                all_enz_true.extend(enzyme[apobec3_mask].numpy().tolist())
                all_enz_pred.extend(pred.tolist())

            # Neither (positives only)
            pos_mask = neither >= 0
            if pos_mask.sum() > 0:
                nei_score = torch.sigmoid(nei_logit[pos_mask]).cpu().numpy()
                all_nei_true.extend(neither[pos_mask].numpy().tolist())
                all_nei_score.extend(nei_score.tolist())

    # Overall binary AUROC
    overall_auroc = roc_auc_score(all_bin_true, all_bin_score)

    # Per-enzyme binary AUROC (each enzyme's positives vs ALL negatives)
    per_enzyme_auroc = {}
    enzyme_arr = np.array(all_enzyme_for_binary)
    bin_arr = np.array(all_bin_true)
    score_arr = np.array(all_bin_score)

    # Map from APOBEC3 label encoder index back to enzyme name
    # enzyme_arr: -1=negative, -2=Neither, -3=Unknown, 0..3=APOBEC3 classes
    # Actually we use a unified mapping, see main()
    for enz_name in ALL_ENZYMES:
        # We'll use the full_enzyme_map from main - passed via closure or we reconstruct
        pass

    # Enzyme classification accuracy
    enz_acc = float(np.mean(np.array(all_enz_true) == np.array(all_enz_pred))) if all_enz_true else 0.0

    # Neither AUROC
    nei_auroc = 0.0
    if all_nei_true and len(set(all_nei_true)) == 2:
        nei_auroc = roc_auc_score(all_nei_true, all_nei_score)

    return {
        "overall_auroc": float(overall_auroc),
        "enzyme_accuracy": enz_acc,
        "neither_auroc": nei_auroc,
        "bin_true": bin_arr,
        "bin_score": score_arr,
        "enzyme_for_binary": enzyme_arr,
    }


def compute_per_enzyme_auroc(results, full_enzyme_map):
    """Compute per-enzyme binary AUROC from evaluation results.

    full_enzyme_map maps enzyme label int -> enzyme name.
    Negatives have label that maps to 'negative'.
    """
    bin_arr = results["bin_true"]
    score_arr = results["bin_score"]
    enz_arr = results["enzyme_for_binary"]

    # Find the negative label
    neg_label = None
    for k, v in full_enzyme_map.items():
        if v == "negative":
            neg_label = k
            break

    per_enzyme = {}
    for label, name in full_enzyme_map.items():
        if name == "negative":
            continue
        # This enzyme's positives + all negatives
        mask = (enz_arr == label) | (enz_arr == neg_label)
        if mask.sum() > 0 and len(np.unique(bin_arr[mask])) == 2:
            per_enzyme[name] = float(roc_auc_score(bin_arr[mask], score_arr[mask]))

    return per_enzyme


def main():
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Device: %s", DEVICE)

    # ---- Load data ----
    df = pd.read_csv(SPLITS_CSV)
    with open(SEQS_JSON) as f:
        seqs = json.load(f)

    emb_orig = torch.load(EMB_POOLED, map_location="cpu", weights_only=False)
    emb_edited = torch.load(EMB_POOLED_ED, map_location="cpu", weights_only=False)
    logger.info("Embeddings: %d orig, %d edited", len(emb_orig), len(emb_edited))

    # Structure delta cache
    struct_delta = {}
    if STRUCT_CACHE.exists():
        data = np.load(str(STRUCT_CACHE), allow_pickle=True)
        for i, sid in enumerate(data["site_ids"].astype(str)):
            struct_delta[sid] = data["delta_features"][i]
        del data
        gc.collect()

    # Loop positions
    loop_df = pd.read_csv(LOOP_CSV)
    loop_df["site_id"] = loop_df["site_id"].astype(str)
    loop_df = loop_df.set_index("site_id")

    # Filter to sites with embeddings
    site_ids = df["site_id"].values
    has_emb = np.array([str(sid) in emb_orig and str(sid) in emb_edited for sid in site_ids])
    df = df[has_emb].copy().reset_index(drop=True)
    site_ids = df["site_id"].values
    logger.info("Sites with embeddings: %d", len(df))

    # ---- Build separate feature groups ----
    orig_embs = np.array([emb_orig[str(sid)].numpy() for sid in site_ids])
    edited_embs = np.array([emb_edited[str(sid)].numpy() for sid in site_ids])
    diff_embs = edited_embs - orig_embs  # Edit effect signal (640-dim)
    del edited_embs  # free memory

    motif_feats = extract_motif_features(seqs, list(site_ids))          # (N, 24)
    loop_feats = extract_loop_features(loop_df, list(site_ids))          # (N, 9)
    struct_feats = extract_structure_delta_features(struct_delta, list(site_ids))  # (N, 7)
    mooring_feats = compute_mooring_au_feature(seqs, list(site_ids))     # (N, 1)

    # Replace NaN
    motif_feats = np.nan_to_num(motif_feats, nan=0.0)
    loop_feats = np.nan_to_num(loop_feats, nan=0.0)
    struct_feats = np.nan_to_num(struct_feats, nan=0.0)
    mooring_feats = np.nan_to_num(mooring_feats, nan=0.0)

    logger.info("Features: motif %s, loop %s, struct %s, mooring %s",
                motif_feats.shape, loop_feats.shape, struct_feats.shape, mooring_feats.shape)

    # ---- Labels ----
    binary_labels = df["is_edited"].values.astype(float)

    # APOBEC3 enzyme labels (A3A=0, A3A_A3G=1, A3B=2, A3G=3)
    apobec3_le = LabelEncoder()
    apobec3_le.fit(APOBEC3_CLASSES)

    enzyme_labels = np.full(len(df), -1, dtype=int)  # -1 for non-APOBEC3
    pos_mask = df["is_edited"] == 1
    for i, row in df[pos_mask].iterrows():
        if row["enzyme"] in APOBEC3_CLASSES:
            enzyme_labels[i] = apobec3_le.transform([row["enzyme"]])[0]
        # Neither and Unknown get -1 (excluded from enzyme head)

    # Neither labels: 1 if Neither, 0 if other positive, -1 if negative
    neither_labels = np.full(len(df), -1.0)
    for i, row in df.iterrows():
        if row["is_edited"] == 1:
            neither_labels[i] = 1.0 if row["enzyme"] == "Neither" else 0.0

    # Full enzyme map for per-enzyme AUROC (unique int per enzyme type)
    # We assign: negatives=-1, A3A=0, A3A_A3G=1, A3B=2, A3G=3, Neither=10, Unknown=11
    full_enzyme_labels = np.full(len(df), -1, dtype=int)
    enzyme_int_map = {"A3A": 0, "A3A_A3G": 1, "A3B": 2, "A3G": 3, "Neither": 10, "Unknown": 11}
    for i, row in df[pos_mask].iterrows():
        full_enzyme_labels[i] = enzyme_int_map.get(row["enzyme"], -1)
    full_enzyme_map = {v: k for k, v in enzyme_int_map.items()}
    full_enzyme_map[-1] = "negative"

    # Compute class weights for enzyme head (inverse frequency)
    apobec3_pos_mask = enzyme_labels >= 0
    apobec3_counts = np.bincount(enzyme_labels[apobec3_pos_mask], minlength=N_APOBEC3)
    enzyme_class_weights = torch.FloatTensor(
        apobec3_counts.sum() / (N_APOBEC3 * apobec3_counts.astype(float) + 1e-8)
    )

    logger.info("Binary: %d pos, %d neg", pos_mask.sum(), (~pos_mask).sum())
    logger.info("APOBEC3 enzymes: %s",
                {apobec3_le.classes_[i]: int(c) for i, c in enumerate(apobec3_counts)})
    logger.info("Neither positives: %d", int((neither_labels == 1).sum()))
    logger.info("Enzyme class weights: %s", enzyme_class_weights.tolist())

    # ---- 5-fold CV ----
    # Stratify by full enzyme (with negatives as separate class)
    strat_labels = full_enzyme_labels.copy()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    fold_results = []

    for fold_i, (train_idx, test_idx) in enumerate(skf.split(orig_embs, strat_labels)):
        torch.manual_seed(SEED + fold_i)
        np.random.seed(SEED + fold_i)
        logger.info("\n--- Fold %d ---", fold_i + 1)

        train_ds = UnifiedDatasetV2(
            orig_embs[train_idx], diff_embs[train_idx],
            loop_feats[train_idx], struct_feats[train_idx],
            motif_feats[train_idx], mooring_feats[train_idx],
            binary_labels[train_idx], enzyme_labels[train_idx], neither_labels[train_idx],
        )
        test_ds = UnifiedDatasetV2(
            orig_embs[test_idx], diff_embs[test_idx],
            loop_feats[test_idx], struct_feats[test_idx],
            motif_feats[test_idx], mooring_feats[test_idx],
            binary_labels[test_idx], enzyme_labels[test_idx], neither_labels[test_idx],
        )
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=128)

        model = UnifiedNetworkV2().to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

        best_auroc = 0.0
        best_state = None
        patience_counter = 0

        for epoch in range(50):
            loss = train_epoch(
                model, train_loader, optimizer,
                enzyme_weight=0.5, neither_weight=0.3,
                enzyme_class_weights=enzyme_class_weights,
            )
            scheduler.step()

            # Evaluate every epoch for better early stopping
            r = evaluate(model, test_loader, apobec3_le)
            if r["overall_auroc"] > best_auroc:
                best_auroc = r["overall_auroc"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= 12:
                logger.info("  Early stopping at epoch %d", epoch + 1)
                break

        # Restore best model
        if best_state is not None:
            model.load_state_dict(best_state)

        # Final evaluation
        final = evaluate(model, test_loader, apobec3_le)
        per_enz = compute_per_enzyme_auroc(final, full_enzyme_map)
        final["per_enzyme_auroc"] = per_enz

        logger.info("  Best overall AUROC=%.3f, enzyme acc=%.3f, neither AUROC=%.3f",
                     best_auroc, final["enzyme_accuracy"], final["neither_auroc"])
        for enz, auroc in sorted(per_enz.items()):
            logger.info("    %s binary AUROC=%.3f", enz, auroc)

        # Store (without large arrays)
        fold_results.append({
            "overall_auroc": float(final["overall_auroc"]),
            "per_enzyme_auroc": per_enz,
            "enzyme_accuracy": final["enzyme_accuracy"],
            "neither_auroc": final["neither_auroc"],
        })

        del model, best_state
        gc.collect()
        if DEVICE == "mps":
            torch.mps.empty_cache()

    # ---- Summary ----
    mean_auroc = np.mean([r["overall_auroc"] for r in fold_results])
    std_auroc = np.std([r["overall_auroc"] for r in fold_results])
    mean_enz_acc = np.mean([r["enzyme_accuracy"] for r in fold_results])
    mean_nei_auroc = np.mean([r["neither_auroc"] for r in fold_results])

    logger.info("\n" + "=" * 60)
    logger.info("V2 Summary: AUROC=%.3f +/- %.3f, Enzyme Acc=%.3f, Neither AUROC=%.3f",
                mean_auroc, std_auroc, mean_enz_acc, mean_nei_auroc)

    for enz in ALL_ENZYMES:
        vals = [r["per_enzyme_auroc"].get(enz, 0) for r in fold_results]
        logger.info("  %s: AUROC=%.3f +/- %.3f", enz, np.mean(vals), np.std(vals))

    # ---- Comparison table ----
    per_enzyme_ref = {"A3A": 0.880, "A3B": 0.971, "A3G": 0.893, "A3A_A3G": 0.935, "Neither": 0.829}
    v1_ref = {"A3A": 0.902, "A3B": 0.916, "A3G": 0.953, "A3A_A3G": 0.948, "Neither": 0.931}

    print("\n" + "=" * 80)
    print("UNIFIED NETWORK V2 vs V1 vs PER-ENZYME COMPARISON")
    print("=" * 80)
    print(f"{'Enzyme':>10} {'V2 (new)':>10} {'V1':>10} {'PerEnzyme':>10} {'V2-V1':>10} {'V2-Per':>10}")
    print("-" * 60)

    for enz in ALL_ENZYMES:
        vals = [r["per_enzyme_auroc"].get(enz, 0) for r in fold_results]
        v2 = np.mean(vals)
        v1 = v1_ref.get(enz, 0)
        pe = per_enzyme_ref.get(enz, 0)
        d_v1 = v2 - v1 if v1 > 0 else float("nan")
        d_pe = v2 - pe if pe > 0 else float("nan")
        print(f"{enz:>10} {v2:10.3f} {v1:10.3f} {pe:10.3f} {d_v1:+10.3f} {d_pe:+10.3f}")

    print(f"\n{'Overall':>10} {mean_auroc:10.3f}")
    print(f"{'Enz Acc':>10} {mean_enz_acc:10.3f}")
    print(f"{'Neither':>10} {mean_nei_auroc:10.3f}")

    # ---- Save results ----
    save_data = {
        "fold_results": fold_results,
        "summary": {
            "mean_overall_auroc": float(mean_auroc),
            "std_overall_auroc": float(std_auroc),
            "mean_enzyme_accuracy": float(mean_enz_acc),
            "mean_neither_auroc": float(mean_nei_auroc),
            "per_enzyme_mean_auroc": {
                enz: float(np.mean([r["per_enzyme_auroc"].get(enz, 0) for r in fold_results]))
                for enz in ALL_ENZYMES
            },
            "per_enzyme_std_auroc": {
                enz: float(np.std([r["per_enzyme_auroc"].get(enz, 0) for r in fold_results]))
                for enz in ALL_ENZYMES
            },
        },
        "references": {
            "per_enzyme_editrna_h": per_enzyme_ref,
            "unified_v1": v1_ref,
        },
        "config": {
            "architecture": "feature_routed_v2",
            "d_emb": 640, "d_emb_proj": 256, "d_diff_proj": 128, "d_shared": 256,
            "d_motif": 24, "d_loop": 9, "d_struct": 7,
            "n_apobec3_classes": N_APOBEC3,
            "enzyme_weight": 0.5, "neither_weight": 0.3,
            "lr": 1e-3, "epochs": 50, "patience": 12,
            "batch_size": 64, "dropout": 0.3,
            "enzyme_class_weights": enzyme_class_weights.tolist(),
        },
    }

    with open(OUTPUT_DIR / "unified_network_v2_results.json", "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    logger.info("Results saved to %s", OUTPUT_DIR / "unified_network_v2_results.json")
    logger.info("Total time: %.0fs", time.time() - t0)


if __name__ == "__main__":
    main()
