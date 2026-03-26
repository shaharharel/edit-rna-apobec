#!/usr/bin/env python
"""A3B neural model analysis: What do EditRNA+Hand and CNN+Hand learn that GB misses?

GB_HandFeatures AUROC=0.580 vs EditRNA+Hand AUROC=0.810 on A3B classification.
This script investigates what the neural models capture beyond the 40-dim hand features.

Analyses:
  1. Retrain EditRNA+Hand and CNN+Hand with corrected data (NaN→0), report AUROC
  2. Gradient-based feature attribution on hand features in neural context
  3. PCA of RNA-FM embeddings — what dimensions separate A3B pos from neg?
  4. CNN saliency analysis — which sequence positions matter?
  5. Misclassification analysis — what patterns does neural get right that GB misses?

Usage:
    conda run -n quris python experiments/multi_enzyme/exp_a3b_neural_analysis.py
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
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
STRUCT_CACHE = _ME_DIR / "structure_cache_multi_enzyme_v3.npz"
LOOP_CSV = _ME_DIR / "loop_position_per_site_v3.csv"
EMB_ORIG = _ME_DIR / "embeddings/rnafm_pooled_v3.pt"
EMB_EDIT = _ME_DIR / "embeddings/rnafm_pooled_edited_v3.pt"
OUTPUT_DIR = Path(__file__).parent / "outputs" / "a3b_neural_analysis"

SEED = 42
N_FOLDS = 5
CENTER = 100
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

FEATURE_NAMES = (
    ['5p_UC', '5p_CC', '5p_AC', '5p_GC',
     '3p_CA', '3p_CG', '3p_CU', '3p_CC',
     'm2_A', 'm2_C', 'm2_G', 'm2_U',
     'm1_A', 'm1_C', 'm1_G', 'm1_U',
     'p1_A', 'p1_C', 'p1_G', 'p1_U',
     'p2_A', 'p2_C', 'p2_G', 'p2_U'] +
    ['delta_pairing_center', 'delta_accessibility_center', 'delta_entropy_center',
     'delta_mfe', 'mean_delta_pairing_window', 'mean_delta_accessibility_window',
     'std_delta_pairing_window'] +
    ['is_unpaired', 'loop_size', 'dist_to_junction', 'dist_to_apex',
     'relative_loop_position', 'left_stem_length', 'right_stem_length',
     'max_adjacent_stem_length', 'local_unpaired_fraction']
)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class EditMLPHand(nn.Module):
    """EditRNA+Hand: RNA-FM edit embedding + context + 40-dim hand features."""
    def __init__(self):
        super().__init__()
        self.edit_proj = nn.Sequential(nn.Linear(640, 256), nn.GELU(), nn.Dropout(0.3))
        self.ctx_proj = nn.Sequential(nn.Linear(640, 256), nn.GELU(), nn.Dropout(0.3))
        self.hand_proj = nn.Sequential(nn.Linear(40, 64), nn.GELU(), nn.Dropout(0.3))
        self.head = nn.Sequential(nn.Linear(576, 256), nn.GELU(), nn.Dropout(0.3), nn.Linear(256, 1))

    def forward(self, o, e, h):
        return self.head(torch.cat([self.edit_proj(e - o), self.ctx_proj(o), self.hand_proj(h)], -1)).squeeze(-1)


class SeqCNNHand(nn.Module):
    """CNN+Hand: 1D-CNN on one-hot sequence + 40-dim hand features."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(4, 64, 7, padding=3), nn.GELU(),
            nn.Conv1d(64, 128, 5, padding=2), nn.GELU(),
            nn.AdaptiveMaxPool1d(20),
            nn.Conv1d(128, 128, 3, padding=1), nn.GELU(),
            nn.AdaptiveMaxPool1d(1),
        )
        self.hand_proj = nn.Sequential(nn.Linear(40, 128), nn.GELU(), nn.Dropout(0.3))
        self.head = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.3), nn.Linear(128, 1))

    def forward(self, seq_oh, hand):
        return self.head(torch.cat([self.conv(seq_oh).squeeze(-1), self.hand_proj(hand)], -1)).squeeze(-1)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data():
    """Load all A3B data."""
    logger.info("Loading data...")
    df = pd.read_csv(SPLITS_CSV)
    a3b = df[df["enzyme"] == "A3B"].copy().reset_index(drop=True)
    logger.info("  A3B: %d sites (pos=%d, neg=%d)",
                len(a3b), (a3b["is_edited"] == 1).sum(), (a3b["is_edited"] == 0).sum())

    with open(SEQS_JSON) as f:
        seqs = json.load(f)

    # Structure delta
    sc = np.load(str(STRUCT_CACHE), allow_pickle=True)
    struct_sids = [str(s) for s in sc["site_ids"]]
    struct_feats = sc["delta_features"]
    structure_delta = dict(zip(struct_sids, struct_feats))

    # Loop
    loop_df = pd.read_csv(LOOP_CSV)
    loop_df["site_id"] = loop_df["site_id"].astype(str)
    loop_df = loop_df.set_index("site_id")

    # RNA-FM embeddings
    emb_orig = torch.load(str(EMB_ORIG), map_location="cpu")
    emb_edit = torch.load(str(EMB_EDIT), map_location="cpu")

    site_ids = a3b["site_id"].astype(str).tolist()
    labels = a3b["is_edited"].values.astype(int)

    # Build hand features (40-dim)
    motif = extract_motif_features(seqs, site_ids)
    loop = extract_loop_features(loop_df, site_ids)
    struct = extract_structure_delta_features(structure_delta, site_ids)
    hand = np.concatenate([motif, struct, loop], axis=1)
    hand = np.nan_to_num(hand, nan=0.0).astype(np.float32)
    logger.info("  Hand features: %s, NaN count: %d", hand.shape, np.isnan(hand).sum())

    # RNA-FM embeddings
    emb_o = np.array([emb_orig[sid].numpy() for sid in site_ids], dtype=np.float32)
    emb_e = np.array([emb_edit[sid].numpy() for sid in site_ids], dtype=np.float32)
    logger.info("  RNA-FM orig: %s, edited: %s", emb_o.shape, emb_e.shape)

    # One-hot sequences
    base_map = {"A": 0, "C": 1, "G": 2, "T": 3, "U": 3, "N": -1}
    seq_oh = np.zeros((len(site_ids), 4, 201), dtype=np.float32)
    for i, sid in enumerate(site_ids):
        seq = seqs[sid].upper()
        for j, base in enumerate(seq):
            idx = base_map.get(base, -1)
            if idx >= 0:
                seq_oh[i, idx, j] = 1.0

    return a3b, site_ids, labels, hand, emb_o, emb_e, seq_oh, seqs


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_editmlp(emb_o, emb_e, hand, labels, train_idx, test_idx, epochs=50, patience=12):
    """Train EditMLPHand and return test predictions + trained model."""
    model = EditMLPHand().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    o_train = torch.tensor(emb_o[train_idx]).to(DEVICE)
    e_train = torch.tensor(emb_e[train_idx]).to(DEVICE)
    h_train = torch.tensor(hand[train_idx]).to(DEVICE)
    y_train = torch.tensor(labels[train_idx], dtype=torch.float32).to(DEVICE)

    o_test = torch.tensor(emb_o[test_idx]).to(DEVICE)
    e_test = torch.tensor(emb_e[test_idx]).to(DEVICE)
    h_test = torch.tensor(hand[test_idx]).to(DEVICE)

    best_loss = float("inf")
    best_state = None
    no_improve = 0

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(o_train, e_train, h_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss - 1e-4:
            best_loss = loss.item()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds = torch.sigmoid(model(o_test, e_test, h_test)).cpu().numpy()
    return preds, model


def train_cnn(seq_oh, hand, labels, train_idx, test_idx, epochs=50, patience=12):
    """Train SeqCNNHand and return test predictions + trained model."""
    model = SeqCNNHand().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    s_train = torch.tensor(seq_oh[train_idx]).to(DEVICE)
    h_train = torch.tensor(hand[train_idx]).to(DEVICE)
    y_train = torch.tensor(labels[train_idx], dtype=torch.float32).to(DEVICE)

    s_test = torch.tensor(seq_oh[test_idx]).to(DEVICE)
    h_test = torch.tensor(hand[test_idx]).to(DEVICE)

    best_loss = float("inf")
    best_state = None
    no_improve = 0

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(s_train, h_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss - 1e-4:
            best_loss = loss.item()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds = torch.sigmoid(model(s_test, h_test)).cpu().numpy()
    return preds, model


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def gradient_attribution_editmlp(model, emb_o, emb_e, hand, indices, device):
    """Compute gradient-based attribution for hand features in EditMLPHand."""
    model.eval()
    o = torch.tensor(emb_o[indices], dtype=torch.float32).to(device)
    e = torch.tensor(emb_e[indices], dtype=torch.float32).to(device)
    h = torch.tensor(hand[indices], dtype=torch.float32).to(device).requires_grad_(True)

    logits = model(o, e, h)
    logits.sum().backward()

    grad = h.grad.detach().cpu().numpy()  # (N, 40)
    # Attribution = input * gradient (integrated gradient approximation)
    attr = np.abs(grad * hand[indices])
    return attr  # (N, 40)


def gradient_attribution_cnn(model, seq_oh, hand, indices, device):
    """Compute gradient-based attribution for hand features and sequence positions in CNN."""
    model.eval()
    s = torch.tensor(seq_oh[indices], dtype=torch.float32).to(device).requires_grad_(True)
    h = torch.tensor(hand[indices], dtype=torch.float32).to(device).requires_grad_(True)

    logits = model(s, h)
    logits.sum().backward()

    hand_grad = h.grad.detach().cpu().numpy()
    seq_grad = s.grad.detach().cpu().numpy()  # (N, 4, 201)

    hand_attr = np.abs(hand_grad * hand[indices])
    # Sequence saliency: max over channels for each position
    seq_saliency = np.abs(seq_grad).max(axis=1)  # (N, 201)

    return hand_attr, seq_saliency


def pca_analysis(emb_o, emb_e, labels, output_dir):
    """PCA of RNA-FM embeddings to understand what separates pos from neg."""
    logger.info("Running PCA analysis on RNA-FM embeddings...")

    # Analyze three representations: original, edited, and edit-effect (delta)
    emb_delta = emb_e - emb_o

    results = {}
    for name, emb in [("original", emb_o), ("edited", emb_e), ("edit_effect", emb_delta)]:
        pca = PCA(n_components=50, random_state=SEED)
        proj = pca.fit_transform(emb)

        # For each PC, compute AUROC as separability metric
        pc_aurocs = []
        for pc_idx in range(min(50, proj.shape[1])):
            try:
                auc = roc_auc_score(labels, proj[:, pc_idx])
                auc = max(auc, 1 - auc)  # direction-invariant
            except ValueError:
                auc = 0.5
            pc_aurocs.append(auc)

        results[name] = {
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "cumulative_variance": np.cumsum(pca.explained_variance_ratio_).tolist(),
            "pc_aurocs": pc_aurocs,
            "top_pcs": sorted(range(len(pc_aurocs)), key=lambda i: -pc_aurocs[i])[:10],
        }

        logger.info("  %s: top 5 PC separability: %s",
                     name, [(i, f"{pc_aurocs[i]:.3f}") for i in results[name]["top_pcs"][:5]])

        # Plot PCA
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # PC1 vs PC2 colored by label
        ax = axes[0]
        for lab, color, lbl in [(0, "#1f77b4", "Negative"), (1, "#d62728", "Positive")]:
            mask = labels == lab
            ax.scatter(proj[mask, 0], proj[mask, 1], c=color, alpha=0.3, s=5, label=lbl)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
        ax.set_title(f"{name} — PC1 vs PC2")
        ax.legend(markerscale=3)

        # Best separating PCs
        top2 = results[name]["top_pcs"][:2]
        ax = axes[1]
        for lab, color, lbl in [(0, "#1f77b4", "Negative"), (1, "#d62728", "Positive")]:
            mask = labels == lab
            ax.scatter(proj[mask, top2[0]], proj[mask, top2[1]], c=color, alpha=0.3, s=5, label=lbl)
        ax.set_xlabel(f"PC{top2[0]+1} (AUROC={pc_aurocs[top2[0]]:.3f})")
        ax.set_ylabel(f"PC{top2[1]+1} (AUROC={pc_aurocs[top2[1]]:.3f})")
        ax.set_title(f"{name} — Best separating PCs")
        ax.legend(markerscale=3)

        # PC AUROC bar chart
        ax = axes[2]
        ax.bar(range(20), pc_aurocs[:20], color="#2ca02c")
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Principal Component")
        ax.set_ylabel("AUROC (direction-invariant)")
        ax.set_title(f"{name} — PC separability")

        plt.tight_layout()
        plt.savefig(output_dir / f"pca_{name}.png", dpi=150, bbox_inches="tight")
        plt.close()

    return results


def cnn_saliency_plot(saliency_pos, saliency_neg, output_dir):
    """Plot average CNN saliency across sequence positions for pos vs neg."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 8))

    # Mean saliency per position
    mean_pos = saliency_pos.mean(axis=0)
    mean_neg = saliency_neg.mean(axis=0)

    ax = axes[0]
    ax.plot(range(201), mean_pos, color="#d62728", alpha=0.8, label="Positive", linewidth=0.8)
    ax.plot(range(201), mean_neg, color="#1f77b4", alpha=0.8, label="Negative", linewidth=0.8)
    ax.axvline(100, color="black", linestyle="--", alpha=0.5, label="Edit site")
    ax.set_xlabel("Sequence position")
    ax.set_ylabel("Mean saliency")
    ax.set_title("CNN saliency: which positions matter?")
    ax.legend()

    # Differential saliency (pos - neg)
    diff = mean_pos - mean_neg
    ax = axes[1]
    ax.bar(range(201), diff, color=["#d62728" if d > 0 else "#1f77b4" for d in diff], width=1.0, alpha=0.7)
    ax.axvline(100, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Sequence position")
    ax.set_ylabel("Differential saliency (pos - neg)")
    ax.set_title("CNN: positions more important for positives (red) vs negatives (blue)")

    plt.tight_layout()
    plt.savefig(output_dir / "cnn_saliency_positions.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Zoom into center region
    fig, ax = plt.subplots(figsize=(12, 4))
    window = slice(80, 121)
    positions = range(80, 121)
    ax.bar(positions, diff[window], color=["#d62728" if d > 0 else "#1f77b4" for d in diff[window]],
           width=0.8, alpha=0.8)
    ax.axvline(100, color="black", linestyle="--", alpha=0.5, label="Edit site (C)")
    ax.set_xlabel("Sequence position (relative to edit site at 100)")
    ax.set_ylabel("Differential saliency")
    ax.set_title("CNN saliency: center region (+/- 20nt around edit)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "cnn_saliency_center_zoom.png", dpi=150, bbox_inches="tight")
    plt.close()

    return {
        "peak_pos_position": int(np.argmax(mean_pos)),
        "peak_neg_position": int(np.argmax(mean_neg)),
        "peak_diff_position": int(np.argmax(np.abs(diff))),
        "center_20nt_saliency_fraction": float(mean_pos[80:121].sum() / mean_pos.sum()),
    }


def misclassification_analysis(gb_scores, editmlp_scores, cnn_scores, labels, hand, site_ids, seqs, output_dir):
    """Analyze sites where neural models are right but GB is wrong."""
    logger.info("Running misclassification analysis...")

    threshold = 0.5
    gb_pred = (gb_scores >= threshold).astype(int)
    editmlp_pred = (editmlp_scores >= threshold).astype(int)
    cnn_pred = (cnn_scores >= threshold).astype(int)

    gb_correct = (gb_pred == labels)
    editmlp_correct = (editmlp_pred == labels)
    cnn_correct = (cnn_pred == labels)

    # Sites where EditMLP gets right but GB gets wrong
    editmlp_wins = ~gb_correct & editmlp_correct
    # Sites where CNN gets right but GB gets wrong
    cnn_wins = ~gb_correct & cnn_correct
    # Sites where both neural get right but GB wrong
    neural_wins = ~gb_correct & editmlp_correct & cnn_correct
    # Sites where GB gets right but both neural wrong
    gb_wins = gb_correct & ~editmlp_correct & ~cnn_correct

    logger.info("  EditMLP wins (right where GB wrong): %d", editmlp_wins.sum())
    logger.info("  CNN wins: %d", cnn_wins.sum())
    logger.info("  Both neural win: %d", neural_wins.sum())
    logger.info("  GB wins (right where both neural wrong): %d", gb_wins.sum())

    results = {
        "total": len(labels),
        "gb_accuracy": float(gb_correct.mean()),
        "editmlp_accuracy": float(editmlp_correct.mean()),
        "cnn_accuracy": float(cnn_correct.mean()),
        "editmlp_wins_count": int(editmlp_wins.sum()),
        "cnn_wins_count": int(cnn_wins.sum()),
        "neural_wins_count": int(neural_wins.sum()),
        "gb_wins_count": int(gb_wins.sum()),
    }

    # Analyze hand feature distributions for neural-wins vs gb-wins
    if neural_wins.sum() > 10 and gb_wins.sum() > 10:
        neural_win_feats = hand[neural_wins]
        gb_win_feats = hand[gb_wins]

        # Feature means comparison
        feat_comparison = []
        for fi, fname in enumerate(FEATURE_NAMES):
            nw_mean = neural_win_feats[:, fi].mean()
            gw_mean = gb_win_feats[:, fi].mean()
            all_mean = hand[:, fi].mean()
            feat_comparison.append({
                "feature": fname,
                "neural_wins_mean": float(nw_mean),
                "gb_wins_mean": float(gw_mean),
                "all_mean": float(all_mean),
                "neural_vs_gb_diff": float(nw_mean - gw_mean),
            })

        feat_df = pd.DataFrame(feat_comparison)
        feat_df = feat_df.sort_values("neural_vs_gb_diff", key=abs, ascending=False)
        feat_df.to_csv(output_dir / "misclass_feature_comparison.csv", index=False)
        results["top_differentiating_features"] = feat_df.head(10).to_dict("records")

    # Sequence context analysis for neural-wins
    if neural_wins.sum() > 10:
        nw_sids = [site_ids[i] for i in range(len(site_ids)) if neural_wins[i]]
        nw_labels = labels[neural_wins]

        # Analyze motif composition
        motif_counts = {}
        for sid, lab in zip(nw_sids, nw_labels):
            seq = seqs.get(sid, "N" * 201).upper().replace("T", "U")
            dinuc_5p = seq[99:101] if len(seq) > 100 else "NN"
            dinuc_3p = seq[100:102] if len(seq) > 101 else "NN"
            key = f"{dinuc_5p}|{dinuc_3p}"
            motif_counts[key] = motif_counts.get(key, {"pos": 0, "neg": 0})
            motif_counts[key]["pos" if lab == 1 else "neg"] += 1

        results["neural_wins_motif_breakdown"] = motif_counts

    # Sequence position base composition for neural-wins (around edit site)
    if neural_wins.sum() > 10:
        nw_indices = np.where(neural_wins)[0]
        gw_indices = np.where(gb_wins)[0] if gb_wins.sum() > 0 else np.array([])

        # Base frequency at each position in window around edit for neural-wins
        base_freq_nw = np.zeros((41, 4))  # positions -20 to +20, ACGU
        base_freq_gw = np.zeros((41, 4))
        base_freq_all = np.zeros((41, 4))

        base_idx = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3}

        for idx in range(len(site_ids)):
            seq = seqs.get(site_ids[idx], "N" * 201).upper().replace("T", "U")
            for p in range(80, 121):
                b = seq[p] if p < len(seq) else "N"
                bi = base_idx.get(b, -1)
                if bi >= 0:
                    base_freq_all[p - 80, bi] += 1
                    if idx in set(nw_indices):
                        base_freq_nw[p - 80, bi] += 1
                    if len(gw_indices) > 0 and idx in set(gw_indices):
                        base_freq_gw[p - 80, bi] += 1

        # Normalize
        for arr in [base_freq_nw, base_freq_gw, base_freq_all]:
            row_sums = arr.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            arr[:] = arr / row_sums

        results["base_composition_saved"] = True

        # Plot
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        colors = ["#2ca02c", "#1f77b4", "#d62728", "#ff7f0e"]
        base_labels = ["A", "C", "G", "U"]
        positions = np.arange(-20, 21)

        ax = axes[0]
        for bi, (bl, col) in enumerate(zip(base_labels, colors)):
            ax.plot(positions, base_freq_nw[:, bi], color=col, label=bl, alpha=0.8)
        ax.axvline(0, color="black", linestyle="--", alpha=0.3)
        ax.set_title(f"Base composition: sites neural gets right, GB wrong (n={neural_wins.sum()})")
        ax.set_xlabel("Position relative to edit site")
        ax.set_ylabel("Base frequency")
        ax.legend()

        ax = axes[1]
        if gb_wins.sum() > 5:
            for bi, (bl, col) in enumerate(zip(base_labels, colors)):
                ax.plot(positions, base_freq_gw[:, bi], color=col, label=bl, alpha=0.8)
            ax.axvline(0, color="black", linestyle="--", alpha=0.3)
            ax.set_title(f"Base composition: sites GB gets right, neural wrong (n={gb_wins.sum()})")
        else:
            ax.set_title("Insufficient GB-wins for comparison")
        ax.set_xlabel("Position relative to edit site")
        ax.set_ylabel("Base frequency")
        ax.legend()

        plt.tight_layout()
        plt.savefig(output_dir / "misclass_base_composition.png", dpi=150, bbox_inches="tight")
        plt.close()

    # Plot confidence distributions
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, scores, name in [
        (axes[0], gb_scores, "GB_HandFeatures"),
        (axes[1], editmlp_scores, "EditRNA+Hand"),
        (axes[2], cnn_scores, "CNN+Hand"),
    ]:
        ax.hist(scores[labels == 1], bins=50, alpha=0.6, color="#d62728", label="Positive", density=True)
        ax.hist(scores[labels == 0], bins=50, alpha=0.6, color="#1f77b4", label="Negative", density=True)
        ax.set_title(name)
        ax.set_xlabel("Predicted score")
        ax.set_ylabel("Density")
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "score_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Device: %s", DEVICE)

    a3b_df, site_ids, labels, hand, emb_o, emb_e, seq_oh, seqs = load_data()

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    # Storage for cross-fold predictions
    all_gb_scores = np.zeros(len(labels))
    all_editmlp_scores = np.zeros(len(labels))
    all_cnn_scores = np.zeros(len(labels))
    all_editmlp_attr = np.zeros((len(labels), 40))
    all_cnn_hand_attr = np.zeros((len(labels), 40))
    all_cnn_saliency = np.zeros((len(labels), 201))

    fold_results = {"editmlp": [], "cnn": [], "gb": []}

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(hand, labels)):
        logger.info("=" * 60)
        logger.info("Fold %d/%d (train=%d, test=%d)", fold_idx + 1, N_FOLDS,
                     len(train_idx), len(test_idx))

        # --- GB ---
        scale_pos = (labels[train_idx] == 0).sum() / max((labels[train_idx] == 1).sum(), 1)
        gb = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
            colsample_bytree=0.8, scale_pos_weight=scale_pos,
            eval_metric="logloss", random_state=SEED + fold_idx,
        )
        gb.fit(hand[train_idx], labels[train_idx], verbose=False)
        gb_preds = gb.predict_proba(hand[test_idx])[:, 1]
        gb_auroc = roc_auc_score(labels[test_idx], gb_preds)
        fold_results["gb"].append(gb_auroc)
        all_gb_scores[test_idx] = gb_preds
        logger.info("  GB AUROC: %.4f", gb_auroc)

        # --- EditMLP+Hand ---
        editmlp_preds, editmlp_model = train_editmlp(
            emb_o, emb_e, hand, labels, train_idx, test_idx
        )
        editmlp_auroc = roc_auc_score(labels[test_idx], editmlp_preds)
        fold_results["editmlp"].append(editmlp_auroc)
        all_editmlp_scores[test_idx] = editmlp_preds
        logger.info("  EditMLP+Hand AUROC: %.4f", editmlp_auroc)

        # Gradient attribution for EditMLP
        attr = gradient_attribution_editmlp(editmlp_model, emb_o, emb_e, hand, test_idx, DEVICE)
        all_editmlp_attr[test_idx] = attr

        # --- CNN+Hand ---
        cnn_preds, cnn_model = train_cnn(seq_oh, hand, labels, train_idx, test_idx)
        cnn_auroc = roc_auc_score(labels[test_idx], cnn_preds)
        fold_results["cnn"].append(cnn_auroc)
        all_cnn_scores[test_idx] = cnn_preds
        logger.info("  CNN+Hand AUROC: %.4f", cnn_auroc)

        # Gradient attribution for CNN
        hand_attr, seq_sal = gradient_attribution_cnn(cnn_model, seq_oh, hand, test_idx, DEVICE)
        all_cnn_hand_attr[test_idx] = hand_attr
        all_cnn_saliency[test_idx] = seq_sal

        # Cleanup
        del editmlp_model, cnn_model
        gc.collect()
        if DEVICE == "mps":
            torch.mps.empty_cache()

    # -----------------------------------------------------------------------
    # Aggregate results
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    summary = {
        "n_sites": len(labels),
        "n_pos": int((labels == 1).sum()),
        "n_neg": int((labels == 0).sum()),
        "device": DEVICE,
    }

    for model_name, key in [("GB_HandFeatures", "gb"), ("EditRNA+Hand", "editmlp"), ("CNN+Hand", "cnn")]:
        aurocs = fold_results[key]
        summary[model_name] = {
            "mean_auroc": float(np.mean(aurocs)),
            "std_auroc": float(np.std(aurocs)),
            "fold_aurocs": [float(a) for a in aurocs],
        }
        logger.info("  %s: AUROC=%.4f +/- %.4f", model_name, np.mean(aurocs), np.std(aurocs))

    # -----------------------------------------------------------------------
    # Analysis 2: Gradient attribution on hand features
    # -----------------------------------------------------------------------
    logger.info("Analyzing gradient attributions...")

    # EditMLP attribution
    editmlp_mean_attr = all_editmlp_attr.mean(axis=0)
    editmlp_attr_rank = sorted(range(40), key=lambda i: -editmlp_mean_attr[i])
    summary["editmlp_hand_attribution"] = {
        FEATURE_NAMES[i]: float(editmlp_mean_attr[i])
        for i in editmlp_attr_rank
    }

    # CNN hand attribution
    cnn_mean_attr = all_cnn_hand_attr.mean(axis=0)
    cnn_attr_rank = sorted(range(40), key=lambda i: -cnn_mean_attr[i])
    summary["cnn_hand_attribution"] = {
        FEATURE_NAMES[i]: float(cnn_mean_attr[i])
        for i in cnn_attr_rank
    }

    # Plot attributions
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    for ax, mean_attr, rank, title in [
        (axes[0], editmlp_mean_attr, editmlp_attr_rank, "EditRNA+Hand"),
        (axes[1], cnn_mean_attr, cnn_attr_rank, "CNN+Hand"),
    ]:
        top20 = rank[:20]
        names = [FEATURE_NAMES[i] for i in top20]
        vals = [mean_attr[i] for i in top20]
        colors = []
        for i in top20:
            if i < 24:
                colors.append("#2ca02c")  # motif
            elif i < 31:
                colors.append("#1f77b4")  # struct delta
            else:
                colors.append("#d62728")  # loop
        ax.barh(range(len(top20)), vals, color=colors)
        ax.set_yticks(range(len(top20)))
        ax.set_yticklabels(names, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(f"{title}: Hand feature attribution (input x gradient)")
        ax.set_xlabel("Mean attribution")
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#2ca02c", label="Motif (24)"),
            Patch(facecolor="#1f77b4", label="Struct delta (7)"),
            Patch(facecolor="#d62728", label="Loop (9)"),
        ]
        ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "hand_feature_attribution.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Also compare GB feature importance vs neural attribution
    gb_full = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
        colsample_bytree=0.8, eval_metric="logloss", random_state=SEED,
    )
    gb_full.fit(hand, labels, verbose=False)
    gb_importance = gb_full.feature_importances_

    fig, ax = plt.subplots(figsize=(10, 8))
    # Normalize each to [0, 1]
    gb_norm = gb_importance / gb_importance.max()
    editmlp_norm = editmlp_mean_attr / editmlp_mean_attr.max()
    cnn_norm = cnn_mean_attr / cnn_mean_attr.max()

    x = np.arange(40)
    width = 0.25
    ax.barh(x - width, gb_norm, width, label="GB importance", color="#1f77b4", alpha=0.8)
    ax.barh(x, editmlp_norm, width, label="EditMLP attribution", color="#d62728", alpha=0.8)
    ax.barh(x + width, cnn_norm, width, label="CNN attribution", color="#2ca02c", alpha=0.8)
    ax.set_yticks(x)
    ax.set_yticklabels(FEATURE_NAMES, fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("Normalized importance/attribution")
    ax.set_title("Feature importance comparison: GB vs EditMLP vs CNN")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    summary["gb_feature_importance"] = {
        FEATURE_NAMES[i]: float(gb_importance[i])
        for i in sorted(range(40), key=lambda i: -gb_importance[i])
    }

    # -----------------------------------------------------------------------
    # Analysis 3: PCA of RNA-FM embeddings
    # -----------------------------------------------------------------------
    pca_results = pca_analysis(emb_o, emb_e, labels, OUTPUT_DIR)
    summary["pca_analysis"] = pca_results

    # -----------------------------------------------------------------------
    # Analysis 4: CNN saliency
    # -----------------------------------------------------------------------
    logger.info("Analyzing CNN saliency...")
    saliency_pos = all_cnn_saliency[labels == 1]
    saliency_neg = all_cnn_saliency[labels == 0]
    saliency_results = cnn_saliency_plot(saliency_pos, saliency_neg, OUTPUT_DIR)
    summary["cnn_saliency"] = saliency_results

    # -----------------------------------------------------------------------
    # Analysis 5: Misclassification analysis
    # -----------------------------------------------------------------------
    misclass_results = misclassification_analysis(
        all_gb_scores, all_editmlp_scores, all_cnn_scores,
        labels, hand, site_ids, seqs, OUTPUT_DIR,
    )
    summary["misclassification"] = misclass_results

    # -----------------------------------------------------------------------
    # Additional: What does the embedding capture that hand features don't?
    # -----------------------------------------------------------------------
    logger.info("Analyzing embedding information beyond hand features...")

    # Train GB on RNA-FM features to see how much the embedding alone captures
    gb_emb = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
        eval_metric="logloss", random_state=SEED,
    )
    emb_delta = emb_e - emb_o
    emb_combined = np.concatenate([emb_o, emb_delta], axis=1)  # 1280-dim

    emb_aurocs = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(hand, labels)):
        gb_emb = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
            eval_metric="logloss", random_state=SEED + fold_idx,
        )
        gb_emb.fit(emb_combined[train_idx], labels[train_idx], verbose=False)
        preds = gb_emb.predict_proba(emb_combined[test_idx])[:, 1]
        emb_aurocs.append(roc_auc_score(labels[test_idx], preds))

    summary["GB_on_RNAFM_embeddings"] = {
        "mean_auroc": float(np.mean(emb_aurocs)),
        "std_auroc": float(np.std(emb_aurocs)),
        "fold_aurocs": [float(a) for a in emb_aurocs],
    }
    logger.info("  GB on RNA-FM (1280-dim): AUROC=%.4f +/- %.4f", np.mean(emb_aurocs), np.std(emb_aurocs))

    # GB on hand + PCA of embeddings (40 + 50 = 90-dim)
    pca = PCA(n_components=50, random_state=SEED)
    emb_pca = pca.fit_transform(emb_combined)
    hand_plus_pca = np.concatenate([hand, emb_pca], axis=1)

    hand_pca_aurocs = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(hand, labels)):
        gb_hp = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
            eval_metric="logloss", random_state=SEED + fold_idx,
        )
        gb_hp.fit(hand_plus_pca[train_idx], labels[train_idx], verbose=False)
        preds = gb_hp.predict_proba(hand_plus_pca[test_idx])[:, 1]
        hand_pca_aurocs.append(roc_auc_score(labels[test_idx], preds))

    summary["GB_Hand+EmbPCA50"] = {
        "mean_auroc": float(np.mean(hand_pca_aurocs)),
        "std_auroc": float(np.std(hand_pca_aurocs)),
        "fold_aurocs": [float(a) for a in hand_pca_aurocs],
    }
    logger.info("  GB on Hand+PCA50: AUROC=%.4f +/- %.4f", np.mean(hand_pca_aurocs), np.std(hand_pca_aurocs))

    # GB on just one-hot k-mer features (wider sequence context)
    logger.info("  Computing extended k-mer features...")
    # 3-mer frequencies in 5 windows around edit site
    kmer_features = []
    for i, sid in enumerate(site_ids):
        seq = seqs.get(sid, "N" * 201).upper().replace("T", "U")
        feats = []
        # Windows: [-50:-20], [-20:-5], [-5:+5], [+5:+20], [+20:+50]
        windows = [(50, 80), (80, 95), (95, 106), (106, 120), (120, 150)]
        for ws, we in windows:
            subseq = seq[ws:we]
            # Dinucleotide frequencies (16-dim per window)
            dinuc_counts = np.zeros(16)
            bases = "ACGU"
            for j in range(len(subseq) - 1):
                b1 = bases.find(subseq[j])
                b2 = bases.find(subseq[j + 1])
                if b1 >= 0 and b2 >= 0:
                    dinuc_counts[b1 * 4 + b2] += 1
            total = max(dinuc_counts.sum(), 1)
            feats.extend((dinuc_counts / total).tolist())
        kmer_features.append(feats)
    kmer_features = np.array(kmer_features, dtype=np.float32)
    kmer_features = np.nan_to_num(kmer_features, nan=0.0)

    hand_kmer = np.concatenate([hand, kmer_features], axis=1)
    kmer_aurocs = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(hand, labels)):
        gb_k = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
            eval_metric="logloss", random_state=SEED + fold_idx,
        )
        gb_k.fit(hand_kmer[train_idx], labels[train_idx], verbose=False)
        preds = gb_k.predict_proba(hand_kmer[test_idx])[:, 1]
        kmer_aurocs.append(roc_auc_score(labels[test_idx], preds))

    summary["GB_Hand+KmerWindows"] = {
        "mean_auroc": float(np.mean(kmer_aurocs)),
        "std_auroc": float(np.std(kmer_aurocs)),
        "fold_aurocs": [float(a) for a in kmer_aurocs],
    }
    logger.info("  GB on Hand+Kmer: AUROC=%.4f +/- %.4f", np.mean(kmer_aurocs), np.std(kmer_aurocs))

    # -----------------------------------------------------------------------
    # Final comparison table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("A3B NEURAL vs GB ANALYSIS — FINAL RESULTS")
    print("=" * 70)
    models_ordered = [
        ("GB_HandFeatures", summary["GB_HandFeatures"]),
        ("GB_on_RNAFM", summary["GB_on_RNAFM_embeddings"]),
        ("GB_Hand+EmbPCA50", summary["GB_Hand+EmbPCA50"]),
        ("GB_Hand+KmerWindows", summary["GB_Hand+KmerWindows"]),
        ("CNN+Hand", summary["CNN+Hand"]),
        ("EditRNA+Hand", summary["EditRNA+Hand"]),
    ]
    print(f"{'Model':<25s} {'AUROC':>12s}")
    print("-" * 40)
    for name, data in models_ordered:
        print(f"{name:<25s} {data['mean_auroc']:.4f} +/- {data['std_auroc']:.4f}")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    with open(OUTPUT_DIR / "a3b_neural_analysis_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else str(x))

    elapsed = time.time() - t0
    logger.info("Total time: %.1f seconds", elapsed)
    logger.info("Results saved to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
