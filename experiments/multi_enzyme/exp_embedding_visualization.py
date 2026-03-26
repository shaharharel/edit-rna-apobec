#!/usr/bin/env python
"""Embedding space visualization for the unified multi-enzyme APOBEC network.

Trains UnifiedNetwork (v1) on all data, extracts 256-dim shared representations,
then generates UMAP, t-SNE, and PCA 2D projections colored by:
  (a) enzyme, (b) edited/not, (c) motif (TC/CC/AC/GC), (d) structure (unpaired/paired)

Output: 3 row-level PNGs + 1 combined 3x4 grid PNG in
        experiments/multi_enzyme/outputs/embedding_viz/

Usage:
    conda run -n quris python experiments/multi_enzyme/exp_embedding_visualization.py
"""

import gc
import json
import logging
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.apobec_feature_extraction import (
    extract_motif_features,
    extract_loop_features,
    extract_structure_delta_features,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --- Paths ---
_ME_DIR = PROJECT_ROOT / "data/processed/multi_enzyme"
SPLITS_CSV = _ME_DIR / "splits_multi_enzyme_v3_with_negatives.csv"
SEQS_JSON = _ME_DIR / "multi_enzyme_sequences_v3_with_negatives.json"
EMB_DIR = _ME_DIR / "embeddings"
EMB_POOLED = EMB_DIR / "rnafm_pooled_v3.pt"
EMB_POOLED_ED = EMB_DIR / "rnafm_pooled_edited_v3.pt"
STRUCT_CACHE = _ME_DIR / "structure_cache_multi_enzyme_v3.npz"
LOOP_CSV = _ME_DIR / "loop_position_per_site_v3.csv"
OUTPUT_DIR = Path(__file__).parent / "outputs" / "embedding_viz"

SEED = 42
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
ENZYME_CLASSES = ["A3A", "A3A_A3G", "A3B", "A3G", "Neither", "Unknown"]
N_ENZYMES = len(ENZYME_CLASSES)

# --- Colors ---
ENZYME_COLORS = {
    "A3A": "#1a73e8",
    "A3B": "#f59e0b",
    "A3G": "#16a34a",
    "A3A_A3G": "#7b1fa2",
    "Neither": "#d93025",
    "Unknown": "#6b7280",
    "Negative": "#cccccc",
}
EDITED_COLORS = {1: "#1a73e8", 0: "#cccccc"}
MOTIF_COLORS = {"TC": "#1a73e8", "CC": "#16a34a", "AC": "#f59e0b", "GC": "#d93025", "Other": "#6b7280"}
STRUCT_COLORS = {"Unpaired": "#1a73e8", "Paired": "#cccccc", "Unknown": "#6b7280"}


# --- Model (same as exp_unified_network_v1.py) ---
class UnifiedNetwork(nn.Module):
    def __init__(self, d_emb=640, d_hand=40, d_shared=256, n_enzymes=N_ENZYMES, dropout=0.3):
        super().__init__()
        d_input = d_emb + d_emb + d_hand
        self.shared = nn.Sequential(
            nn.Linear(d_input, 512), nn.GELU(), nn.LayerNorm(512), nn.Dropout(dropout),
            nn.Linear(512, d_shared), nn.GELU(), nn.LayerNorm(d_shared), nn.Dropout(dropout),
        )
        self.binary_head = nn.Sequential(
            nn.Linear(d_shared, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 1),
        )
        self.enzyme_head = nn.Sequential(
            nn.Linear(d_shared, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, n_enzymes),
        )

    def forward(self, emb_orig, emb_edited, hand):
        diff = emb_edited - emb_orig
        x = torch.cat([emb_orig, diff, hand], dim=-1)
        shared = self.shared(x)
        binary_logit = self.binary_head(shared).squeeze(-1)
        enzyme_logits = self.enzyme_head(shared)
        return binary_logit, enzyme_logits, shared


class AllDataset(Dataset):
    def __init__(self, emb_orig, emb_edited, hand_features, binary_labels, enzyme_labels):
        self.emb_orig = torch.FloatTensor(emb_orig)
        self.emb_edited = torch.FloatTensor(emb_edited)
        self.hand = torch.FloatTensor(hand_features)
        self.binary = torch.FloatTensor(binary_labels)
        self.enzyme = torch.LongTensor(enzyme_labels)

    def __len__(self):
        return len(self.binary)

    def __getitem__(self, idx):
        return (self.emb_orig[idx], self.emb_edited[idx], self.hand[idx],
                self.binary[idx], self.enzyme[idx])


def train_one_epoch(model, loader, optimizer, phase, enzyme_weight=0.5):
    model.train()
    total_loss = 0
    n = 0
    for emb_o, emb_e, hand, binary, enzyme in loader:
        emb_o, emb_e, hand = emb_o.to(DEVICE), emb_e.to(DEVICE), hand.to(DEVICE)
        binary, enzyme = binary.to(DEVICE), enzyme.to(DEVICE)

        bin_logit, enz_logits, _ = model(emb_o, emb_e, hand)
        loss_bin = F.binary_cross_entropy_with_logits(bin_logit, binary)

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
        n += len(binary)
    return total_loss / n


def extract_shared_representations(model, loader):
    model.eval()
    all_shared = []
    with torch.no_grad():
        for emb_o, emb_e, hand, _, _ in loader:
            emb_o, emb_e, hand = emb_o.to(DEVICE), emb_e.to(DEVICE), hand.to(DEVICE)
            _, _, shared = model(emb_o, emb_e, hand)
            all_shared.append(shared.cpu().numpy())
    return np.concatenate(all_shared, axis=0)


def make_row_figure(coords_2d, labels_list, colors_map_list, titles, legend_labels_list, suptitle, filename):
    """Create a 1x4 row of scatter plots."""
    fig, axes = plt.subplots(1, 4, figsize=(24, 5.5))
    fig.suptitle(suptitle, fontsize=16, fontweight="bold", y=1.02)

    for ax, labels, colors_map, title, legend_labels in zip(
        axes, labels_list, colors_map_list, titles, legend_labels_list
    ):
        unique_labels = sorted(set(labels), key=lambda x: (x == "Negative", x))
        # Draw "background" categories last so foreground is on top
        draw_order = []
        for lbl in unique_labels:
            if lbl in ("Negative", "Paired", "Unknown", 0):
                draw_order.insert(0, lbl)  # background first
            else:
                draw_order.append(lbl)

        for lbl in draw_order:
            mask = np.array([l == lbl for l in labels])
            color = colors_map.get(lbl, "#999999")
            ax.scatter(
                coords_2d[mask, 0], coords_2d[mask, 1],
                c=color, s=3, alpha=0.3, rasterized=True, linewidths=0,
            )

        # Legend
        handles = []
        for lbl in legend_labels:
            handles.append(mpatches.Patch(color=colors_map.get(lbl, "#999999"), label=str(lbl)))
        ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.7,
                  markerscale=2, handlelength=1, handletextpad=0.5)

        ax.set_title(title, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")

    plt.tight_layout()
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", filename)


def main():
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Device: %s", DEVICE)

    # ---- Load data ----
    logger.info("Loading data...")
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
        del data
        gc.collect()

    loop_df = pd.read_csv(LOOP_CSV)
    loop_df["site_id"] = loop_df["site_id"].astype(str)
    loop_df = loop_df.set_index("site_id")

    # Filter to sites with embeddings
    site_ids = df["site_id"].values
    has_emb = np.array([str(sid) in emb_orig and str(sid) in emb_edited for sid in site_ids])
    df = df[has_emb].copy().reset_index(drop=True)
    site_ids = df["site_id"].values
    logger.info("Sites with embeddings: %d", len(df))

    # Build features
    orig_embs = np.array([emb_orig[str(sid)].numpy() for sid in site_ids])
    edited_embs = np.array([emb_edited[str(sid)].numpy() for sid in site_ids])

    motif = extract_motif_features(seqs, list(site_ids))
    struct = extract_structure_delta_features(struct_delta, list(site_ids))
    loop = extract_loop_features(loop_df, list(site_ids))
    hand_40 = np.nan_to_num(np.concatenate([motif, struct, loop], axis=1), nan=0.0)
    logger.info("Hand features shape: %s", hand_40.shape)

    # Labels
    binary_labels = df["is_edited"].values.astype(float)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(ENZYME_CLASSES)
    enzyme_labels = np.full(len(df), -1, dtype=int)
    pos_mask = df["is_edited"] == 1
    enzyme_labels[pos_mask] = le.transform(df.loc[pos_mask, "enzyme"].values)

    logger.info("Binary: %d pos, %d neg", pos_mask.sum(), (~pos_mask).sum())

    # ---- Train unified network ----
    logger.info("Training unified network (30 epochs total)...")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    dataset = AllDataset(orig_embs, edited_embs, hand_40, binary_labels, enzyme_labels)
    loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0)

    model = UnifiedNetwork().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Phase 1: Binary only (15 epochs)
    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
    for epoch in range(15):
        loss = train_one_epoch(model, loader, optimizer, phase=1)
        scheduler1.step()
        if (epoch + 1) % 5 == 0:
            logger.info("  Phase 1, epoch %d: loss=%.4f", epoch + 1, loss)

    # Phase 2: Binary + Enzyme (15 epochs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
    for epoch in range(15):
        loss = train_one_epoch(model, loader, optimizer, phase=2, enzyme_weight=0.5)
        scheduler2.step()
        if (epoch + 1) % 5 == 0:
            logger.info("  Phase 2, epoch %d: loss=%.4f", epoch + 1, loss)

    # ---- Extract 256-dim shared representations ----
    logger.info("Extracting shared representations...")
    eval_loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
    shared_repr = extract_shared_representations(model, eval_loader)
    logger.info("Shared repr shape: %s", shared_repr.shape)

    del model
    gc.collect()
    if DEVICE == "mps":
        torch.mps.empty_cache()

    # ---- Build coloring arrays ----
    # (a) Enzyme labels
    enzyme_color_labels = []
    for i in range(len(df)):
        if df.iloc[i]["is_edited"] == 0:
            enzyme_color_labels.append("Negative")
        else:
            enzyme_color_labels.append(df.iloc[i]["enzyme"])
    enzyme_legend = ["A3A", "A3B", "A3G", "A3A_A3G", "Neither", "Unknown", "Negative"]

    # (b) Edited/not
    edited_labels = df["is_edited"].values.tolist()
    edited_legend = [1, 0]

    # (c) Motif: check base at position 99 (0-indexed) in the 201-nt sequence
    motif_labels = []
    for sid in site_ids:
        seq = seqs.get(str(sid), "")
        if len(seq) > 99:
            base_before = seq[99].upper()  # base at edit position (C in original)
            # The motif context is the base BEFORE the edit site (position 98)
            base_5p = seq[98].upper() if len(seq) > 98 else "?"
            if base_5p == "T" or base_5p == "U":
                motif_labels.append("TC")
            elif base_5p == "C":
                motif_labels.append("CC")
            elif base_5p == "A":
                motif_labels.append("AC")
            elif base_5p == "G":
                motif_labels.append("GC")
            else:
                motif_labels.append("Other")
        else:
            motif_labels.append("Other")
    motif_legend = ["TC", "CC", "AC", "GC", "Other"]

    # (d) Structure: unpaired/paired from loop_position_per_site
    struct_labels = []
    for sid in site_ids:
        sid_str = str(sid)
        if sid_str in loop_df.index and "is_unpaired" in loop_df.columns:
            val = loop_df.loc[sid_str, "is_unpaired"]
            if isinstance(val, pd.Series):
                val = val.iloc[0]
            if val == 1:
                struct_labels.append("Unpaired")
            elif val == 0:
                struct_labels.append("Paired")
            else:
                struct_labels.append("Unknown")
        else:
            struct_labels.append("Unknown")
    struct_legend = ["Unpaired", "Paired", "Unknown"]

    # ---- Dimensionality reduction ----
    logger.info("Running PCA...")
    pca = PCA(n_components=2, random_state=SEED)
    pca_2d = pca.fit_transform(shared_repr)
    logger.info("PCA explained variance: %.1f%%, %.1f%%",
                pca.explained_variance_ratio_[0] * 100, pca.explained_variance_ratio_[1] * 100)

    logger.info("Running t-SNE (perplexity=30)...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=SEED, n_jobs=-1)
    tsne_2d = tsne.fit_transform(shared_repr)

    logger.info("Running UMAP (n_neighbors=30, min_dist=0.3)...")
    import umap
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=SEED, n_jobs=-1)
    umap_2d = reducer.fit_transform(shared_repr)

    # ---- Generate plots ----
    coloring_titles = ["Enzyme", "Edited vs Not", "Motif Context", "Structure"]
    labels_list = [enzyme_color_labels, edited_labels, motif_labels, struct_labels]
    colors_list = [ENZYME_COLORS, EDITED_COLORS, MOTIF_COLORS, STRUCT_COLORS]
    legends_list = [enzyme_legend, edited_legend, motif_legend, struct_legend]

    # Row-level PNGs
    for name, coords in [("UMAP", umap_2d), ("t-SNE", tsne_2d), ("PCA", pca_2d)]:
        fname = OUTPUT_DIR / f"{name.lower().replace('-', '')}_grid.png"
        make_row_figure(
            coords, labels_list, colors_list, coloring_titles, legends_list,
            suptitle=f"{name} of 256-dim Shared Representations (Unified Network v1)",
            filename=str(fname),
        )

    # Combined 3x4 grid
    logger.info("Creating combined 3x4 grid...")
    fig, axes = plt.subplots(3, 4, figsize=(24, 16))
    fig.suptitle("Unified Network Embedding Space: Multi-Enzyme APOBEC Editing",
                 fontsize=18, fontweight="bold", y=0.98)

    for row_idx, (method_name, coords) in enumerate([("UMAP", umap_2d), ("t-SNE", tsne_2d), ("PCA", pca_2d)]):
        for col_idx, (labels, colors_map, title, legend_labels) in enumerate(
            zip(labels_list, colors_list, coloring_titles, legends_list)
        ):
            ax = axes[row_idx, col_idx]

            unique_labels = sorted(set(labels), key=lambda x: (x == "Negative", x == "Paired", x == "Unknown", x == 0, str(x)))
            draw_order = []
            for lbl in unique_labels:
                if lbl in ("Negative", "Paired", "Unknown", 0):
                    draw_order.insert(0, lbl)
                else:
                    draw_order.append(lbl)

            for lbl in draw_order:
                mask = np.array([l == lbl for l in labels])
                color = colors_map.get(lbl, "#999999")
                ax.scatter(
                    coords[mask, 0], coords[mask, 1],
                    c=color, s=3, alpha=0.3, rasterized=True, linewidths=0,
                )

            handles = [mpatches.Patch(color=colors_map.get(lbl, "#999999"), label=str(lbl))
                       for lbl in legend_labels]
            ax.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.7,
                      markerscale=2, handlelength=1, handletextpad=0.5)

            if row_idx == 0:
                ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])
            if col_idx == 0:
                ax.set_ylabel(method_name, fontsize=13, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    combined_path = OUTPUT_DIR / "embedding_viz_combined.png"
    fig.savefig(str(combined_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", combined_path)

    # Save metadata
    meta = {
        "n_sites": int(len(df)),
        "n_positives": int(pos_mask.sum()),
        "n_negatives": int((~pos_mask).sum()),
        "pca_explained_var": pca.explained_variance_ratio_.tolist(),
        "training_epochs": {"phase1_binary": 15, "phase2_joint": 15},
        "motif_counts": {k: motif_labels.count(k) for k in set(motif_labels)},
        "struct_counts": {k: struct_labels.count(k) for k in set(struct_labels)},
        "enzyme_counts": {k: enzyme_color_labels.count(k) for k in set(enzyme_color_labels)},
    }
    with open(OUTPUT_DIR / "embedding_viz_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Total time: %.0fs", time.time() - t0)
    logger.info("Output directory: %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
