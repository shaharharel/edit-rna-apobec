#!/usr/bin/env python
"""Unified interpretability iteration 2: deeper biological analysis.

Building on iteration 1 findings:
  1. Biological analysis of ALL rescued sites (motif, structure, dataset, genes)
  2. A3B "lost" site analysis (why does A3B not benefit from unified?)
  3. Showcase figure (2x2 grid)
  4. Output detailed CSVs for narrative writing

Usage:
    conda run -n quris python experiments/multi_enzyme/exp_unified_interpretability_v2.py
"""

import gc
import json
import logging
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.apobec_feature_extraction import (
    extract_motif_features, extract_loop_features,
    extract_structure_delta_features,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────
_ME_DIR = PROJECT_ROOT / "data/processed/multi_enzyme"
SPLITS_CSV = _ME_DIR / "splits_multi_enzyme_v3_with_negatives.csv"
SEQS_JSON = _ME_DIR / "multi_enzyme_sequences_v3_with_negatives.json"
EMB_DIR = _ME_DIR / "embeddings"
EMB_POOLED = EMB_DIR / "rnafm_pooled_v3.pt"
EMB_POOLED_ED = EMB_DIR / "rnafm_pooled_edited_v3.pt"
STRUCT_CACHE = _ME_DIR / "structure_cache_multi_enzyme_v3.npz"
LOOP_CSV = _ME_DIR / "loop_position_per_site_v3.csv"
REFGENE = PROJECT_ROOT / "data/raw/genomes/refGene.txt"
OUTPUT_DIR = Path(__file__).parent / "outputs" / "unified_interpretability"

SEED = 42
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
ENZYME_CLASSES = ["A3A", "A3A_A3G", "A3B", "A3G", "Neither", "Unknown"]
N_ENZYMES = len(ENZYME_CLASSES)

HAND_FEATURE_NAMES = (
    ["motif_UC", "motif_CC", "motif_AC", "motif_GC",
     "motif_CA", "motif_CG", "motif_CU", "motif_CC_3p",
     "trinuc_m2_A", "trinuc_m2_C", "trinuc_m2_G", "trinuc_m2_U",
     "trinuc_m1_A", "trinuc_m1_C", "trinuc_m1_G", "trinuc_m1_U",
     "trinuc_p1_A", "trinuc_p1_C", "trinuc_p1_G", "trinuc_p1_U",
     "trinuc_p2_A", "trinuc_p2_C", "trinuc_p2_G", "trinuc_p2_U"] +
    ["delta_pairing_center", "delta_accessibility_center", "delta_entropy_center",
     "delta_mfe", "mean_delta_pairing_window", "std_delta_pairing_window",
     "mean_delta_accessibility_window"] +
    ["is_unpaired", "loop_size", "dist_to_junction", "dist_to_apex",
     "relative_loop_position", "left_stem_length", "right_stem_length",
     "max_adjacent_stem_length", "local_unpaired_fraction"]
)

LOOP_FEAT_INDICES = {
    "is_unpaired": 31, "loop_size": 32, "dist_to_junction": 33,
    "dist_to_apex": 34, "relative_loop_position": 35,
    "left_stem_length": 36, "right_stem_length": 37,
    "max_adjacent_stem_length": 38, "local_unpaired_fraction": 39,
}

STRUCT_DELTA_INDICES = {
    "delta_pairing_center": 24, "delta_accessibility_center": 25,
    "delta_entropy_center": 26, "delta_mfe": 27,
    "mean_delta_pairing_window": 28, "std_delta_pairing_window": 29,
    "mean_delta_accessibility_window": 30,
}


# ── Models (same as v1) ──────────────────────────────────────────────────

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


class PerEnzymeNetwork(nn.Module):
    def __init__(self, d_emb=640, d_hand=40, d_shared=256, dropout=0.3):
        super().__init__()
        d_input = d_emb + d_emb + d_hand
        self.shared = nn.Sequential(
            nn.Linear(d_input, 512), nn.GELU(), nn.LayerNorm(512), nn.Dropout(dropout),
            nn.Linear(512, d_shared), nn.GELU(), nn.LayerNorm(d_shared), nn.Dropout(dropout),
        )
        self.binary_head = nn.Sequential(
            nn.Linear(d_shared, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 1),
        )

    def forward(self, emb_orig, emb_edited, hand):
        diff = emb_edited - emb_orig
        x = torch.cat([emb_orig, diff, hand], dim=-1)
        shared = self.shared(x)
        binary_logit = self.binary_head(shared).squeeze(-1)
        return binary_logit, shared


# ── Training helpers ──────────────────────────────────────────────────────

def train_unified_epoch(model, data, indices, batch_size=64, optimizer=None, phase=1, enz_weight=0.5):
    model.train()
    np.random.shuffle(indices)
    total_loss, n = 0, 0
    orig, edited, hand, binary, enzyme = data
    for start in range(0, len(indices), batch_size):
        idx = indices[start:start + batch_size]
        o, e, h = orig[idx].to(DEVICE), edited[idx].to(DEVICE), hand[idx].to(DEVICE)
        b, ez = binary[idx].to(DEVICE), enzyme[idx].to(DEVICE)
        bin_logit, enz_logits, _ = model(o, e, h)
        loss = F.binary_cross_entropy_with_logits(bin_logit, b)
        if phase >= 2:
            pos_mask = ez >= 0
            if pos_mask.sum() > 0:
                loss = loss + enz_weight * F.cross_entropy(enz_logits[pos_mask], ez[pos_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(idx)
        n += len(idx)
    return total_loss / n


def train_perenz_epoch(model, data, indices, batch_size=64, optimizer=None):
    model.train()
    np.random.shuffle(indices)
    total_loss, n = 0, 0
    orig, edited, hand, binary = data
    for start in range(0, len(indices), batch_size):
        idx = indices[start:start + batch_size]
        o, e, h = orig[idx].to(DEVICE), edited[idx].to(DEVICE), hand[idx].to(DEVICE)
        b = binary[idx].to(DEVICE)
        bin_logit, _ = model(o, e, h)
        loss = F.binary_cross_entropy_with_logits(bin_logit, b)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(idx)
        n += len(idx)
    return total_loss / n


@torch.no_grad()
def predict_scores(model, orig, edited, hand, indices, batch_size=256, unified=True):
    model.eval()
    scores, reprs = [], []
    for start in range(0, len(indices), batch_size):
        idx = indices[start:start + batch_size]
        o, e, h = orig[idx].to(DEVICE), edited[idx].to(DEVICE), hand[idx].to(DEVICE)
        if unified:
            bin_logit, _, shared = model(o, e, h)
        else:
            bin_logit, shared = model(o, e, h)
        scores.append(torch.sigmoid(bin_logit).cpu())
        reprs.append(shared.cpu())
    return torch.cat(scores).numpy(), torch.cat(reprs).numpy()


# ── Gene mapping ─────────────────────────────────────────────────────────

def load_refgene(path):
    """Load refGene as interval lookup: chr -> list of (start, end, gene_name)."""
    gene_map = defaultdict(list)
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 13:
                continue
            chrom = parts[2]
            tx_start = int(parts[4])
            tx_end = int(parts[5])
            gene_name = parts[12]
            gene_map[chrom].append((tx_start, tx_end, gene_name))
    # Sort by start
    for chrom in gene_map:
        gene_map[chrom].sort()
    return gene_map


def lookup_gene(gene_map, chrom, pos):
    """Find gene containing position using linear scan (sufficient for small N)."""
    for start, end, name in gene_map.get(chrom, []):
        if start <= pos <= end:
            return name
    return None


# ── Analysis functions ───────────────────────────────────────────────────

def get_motif_5p(seq):
    """Get 5' dinucleotide motif from 201-nt sequence."""
    if len(seq) < 101:
        return "??"
    return seq[99:101].upper().replace("T", "U")


def get_context_5nt(seq):
    if len(seq) < 103:
        return "?????"
    return seq[98:103].upper().replace("T", "U")


def analyze_rescued_vs_nonrescued(df_sites, hand_40, seqs, site_ids, gene_map,
                                  rescued_mask, enzyme_name, label="rescued"):
    """Compute stats for rescued (or lost) sites vs non-rescued positives."""
    pos_mask = df_sites["is_edited"] == 1
    all_pos_idx = np.where(pos_mask)[0]
    target_idx = np.where(pos_mask & rescued_mask)[0]
    non_target_idx = np.where(pos_mask & ~rescued_mask)[0]

    if len(target_idx) == 0:
        return {}

    stats = {"n": len(target_idx), "n_non_target": len(non_target_idx)}

    # Motif distribution
    target_motifs = [get_motif_5p(seqs.get(str(site_ids[i]), "N" * 201)) for i in target_idx]
    non_target_motifs = [get_motif_5p(seqs.get(str(site_ids[i]), "N" * 201)) for i in non_target_idx]

    motif_counts = Counter(target_motifs)
    total = len(target_motifs) or 1
    stats["tc_pct"] = motif_counts.get("UC", 0) / total * 100
    stats["cc_pct"] = motif_counts.get("CC", 0) / total * 100
    stats["ac_pct"] = motif_counts.get("AC", 0) / total * 100
    stats["gc_pct"] = motif_counts.get("GC", 0) / total * 100
    stats["motif_counts"] = dict(motif_counts)

    non_target_mc = Counter(non_target_motifs)
    nt_total = len(non_target_motifs) or 1
    stats["non_target_tc_pct"] = non_target_mc.get("UC", 0) / nt_total * 100

    # Structure features
    for feat_name, feat_idx in LOOP_FEAT_INDICES.items():
        target_vals = hand_40[target_idx, feat_idx].numpy()
        non_target_vals = hand_40[non_target_idx, feat_idx].numpy() if len(non_target_idx) > 0 else np.array([0])
        stats[f"{feat_name}_mean"] = float(np.mean(target_vals))
        stats[f"{feat_name}_non_target_mean"] = float(np.mean(non_target_vals))

    # Dataset distribution
    target_datasets = df_sites.iloc[target_idx]["dataset_source"].value_counts().to_dict()
    stats["datasets"] = target_datasets

    # Gene mapping
    genes = []
    for i in target_idx[:100]:  # top 100 for gene mapping
        chrom = df_sites.iloc[i].get("chr", "")
        pos = df_sites.iloc[i].get("start", 0)
        if pd.notna(chrom) and pd.notna(pos) and chrom != "":
            gene = lookup_gene(gene_map, str(chrom), int(pos))
            if gene:
                genes.append(gene)
    stats["top_genes"] = dict(Counter(genes).most_common(20))

    return stats


def create_showcase_figure(df, unified_scores, per_enzyme_scores, hand_40,
                           binary_labels, enzyme_labels, le, seqs, site_ids,
                           enzyme_site_mask, rescued_masks, lost_masks,
                           output_path, gradient_attrs=None):
    """Create 2x2 showcase figure."""
    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # ── Panel A: Score scatter for A3A with rescued highlighted ────────
    ax_a = fig.add_subplot(gs[0, 0])
    enz_name = "A3A"
    valid = ~np.isnan(per_enzyme_scores[enz_name])
    pos_mask = (df["is_edited"] == 1).values
    neg_mask = (df["is_edited"] == 0).values

    # Plot negatives
    neg_valid = valid & neg_mask
    ax_a.scatter(per_enzyme_scores[enz_name][neg_valid],
                 unified_scores[neg_valid],
                 c="#cccccc", s=4, alpha=0.3, label="Negatives", rasterized=True)

    # Plot non-rescued positives
    enz_pos = valid & pos_mask & (enzyme_labels == le.transform([enz_name])[0])
    rescued_full = np.zeros(len(df), dtype=bool)
    rescued_full[rescued_masks.get(enz_name, np.array([], dtype=int))] = True
    non_rescued_pos = enz_pos & ~rescued_full
    ax_a.scatter(per_enzyme_scores[enz_name][non_rescued_pos],
                 unified_scores[non_rescued_pos],
                 c="#1f77b4", s=10, alpha=0.5, label="A3A positives")

    # Plot rescued positives
    rescued_pos = enz_pos & rescued_full
    ax_a.scatter(per_enzyme_scores[enz_name][rescued_pos],
                 unified_scores[rescued_pos],
                 c="#d62728", s=25, alpha=0.8, label=f"Rescued (n={rescued_pos.sum()})",
                 edgecolors="black", linewidths=0.5, zorder=5)

    ax_a.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=0.8)
    ax_a.axhline(0.5, color="gray", linestyle=":", alpha=0.3)
    ax_a.axvline(0.5, color="gray", linestyle=":", alpha=0.3)
    ax_a.set_xlabel("Per-enzyme score", fontsize=11)
    ax_a.set_ylabel("Unified score", fontsize=11)
    ax_a.set_title("A: A3A score comparison", fontsize=13, fontweight="bold")
    ax_a.legend(fontsize=9, loc="lower right")
    ax_a.set_xlim(-0.02, 1.02)
    ax_a.set_ylim(-0.02, 1.02)

    # ── Panel B: Feature comparison of rescued vs non-rescued ─────────
    ax_b = fig.add_subplot(gs[0, 1])

    # Collect feature stats for rescued vs non-rescued across enzymes
    features_to_show = [
        ("is_unpaired", "Unpaired"),
        ("relative_loop_position", "Rel. loop pos."),
        ("loop_size", "Loop size"),
        ("local_unpaired_fraction", "Local unp. frac."),
        ("left_stem_length", "Left stem len."),
    ]

    # Use A3A as representative
    enz_name = "A3A"
    enz_idx_val = le.transform([enz_name])[0]
    enz_pos_mask = (enzyme_labels == enz_idx_val) & pos_mask
    rescued_enz = enz_pos_mask & rescued_full

    rescued_means = []
    nonrescued_means = []
    feat_labels = []

    for feat_name, display_name in features_to_show:
        fidx = LOOP_FEAT_INDICES[feat_name]
        r_vals = hand_40[rescued_enz, fidx].numpy()
        nr_vals = hand_40[enz_pos_mask & ~rescued_full, fidx].numpy()
        if len(r_vals) > 0 and len(nr_vals) > 0:
            rescued_means.append(np.mean(r_vals))
            nonrescued_means.append(np.mean(nr_vals))
            feat_labels.append(display_name)

    x_pos = np.arange(len(feat_labels))
    width = 0.35
    bars1 = ax_b.bar(x_pos - width / 2, rescued_means, width, label="Rescued",
                     color="#d62728", alpha=0.8)
    bars2 = ax_b.bar(x_pos + width / 2, nonrescued_means, width, label="Non-rescued",
                     color="#1f77b4", alpha=0.8)
    ax_b.set_xticks(x_pos)
    ax_b.set_xticklabels(feat_labels, rotation=30, ha="right", fontsize=9)
    ax_b.set_ylabel("Mean value", fontsize=11)
    ax_b.set_title("B: A3A rescued vs non-rescued features", fontsize=13, fontweight="bold")
    ax_b.legend(fontsize=9)

    # ── Panel C: Motif distribution rescued vs all positives ──────────
    ax_c = fig.add_subplot(gs[1, 0])

    enzymes_for_motif = ["A3A", "A3B", "A3G"]
    motif_data = {}

    for enz_name in enzymes_for_motif:
        enz_idx_val = le.transform([enz_name])[0]
        enz_pos = (enzyme_labels == enz_idx_val) & pos_mask

        # Rescued positives
        rescued_enz_mask = np.zeros(len(df), dtype=bool)
        rescued_enz_mask[rescued_masks.get(enz_name, np.array([], dtype=int))] = True
        rescued_pos_idx = np.where(enz_pos & rescued_enz_mask)[0]
        all_pos_idx = np.where(enz_pos)[0]

        rescued_motifs = [get_motif_5p(seqs.get(str(site_ids[i]), "N" * 201)) for i in rescued_pos_idx]
        all_motifs = [get_motif_5p(seqs.get(str(site_ids[i]), "N" * 201)) for i in all_pos_idx]

        r_tc = sum(1 for m in rescued_motifs if m == "UC") / max(len(rescued_motifs), 1) * 100
        a_tc = sum(1 for m in all_motifs if m == "UC") / max(len(all_motifs), 1) * 100
        r_cc = sum(1 for m in rescued_motifs if m == "CC") / max(len(rescued_motifs), 1) * 100
        a_cc = sum(1 for m in all_motifs if m == "CC") / max(len(all_motifs), 1) * 100

        motif_data[enz_name] = {
            "rescued_TC": r_tc, "all_TC": a_tc,
            "rescued_CC": r_cc, "all_CC": a_cc,
            "n_rescued": len(rescued_pos_idx), "n_all": len(all_pos_idx),
        }

    # Grouped bar chart
    x_pos = np.arange(len(enzymes_for_motif))
    width = 0.2

    tc_rescued = [motif_data[e]["rescued_TC"] for e in enzymes_for_motif]
    tc_all = [motif_data[e]["all_TC"] for e in enzymes_for_motif]
    cc_rescued = [motif_data[e]["rescued_CC"] for e in enzymes_for_motif]
    cc_all = [motif_data[e]["all_CC"] for e in enzymes_for_motif]

    ax_c.bar(x_pos - 1.5 * width, tc_rescued, width, label="TC rescued", color="#d62728", alpha=0.8)
    ax_c.bar(x_pos - 0.5 * width, tc_all, width, label="TC all pos.", color="#ff9999", alpha=0.8)
    ax_c.bar(x_pos + 0.5 * width, cc_rescued, width, label="CC rescued", color="#2ca02c", alpha=0.8)
    ax_c.bar(x_pos + 1.5 * width, cc_all, width, label="CC all pos.", color="#99cc99", alpha=0.8)

    ax_c.set_xticks(x_pos)
    ax_c.set_xticklabels(enzymes_for_motif, fontsize=11)
    ax_c.set_ylabel("% of sites", fontsize=11)
    ax_c.set_title("C: Motif distribution (rescued vs all positives)", fontsize=13, fontweight="bold")
    ax_c.legend(fontsize=8, ncol=2)

    # ── Panel D: Gradient attribution heatmap ─────────────────────────
    ax_d = fig.add_subplot(gs[1, 1])

    if gradient_attrs is not None:
        # Show top features by mean absolute gradient
        mean_abs = np.mean(np.abs(gradient_attrs), axis=0)
        top_k = 12
        top_idx = np.argsort(mean_abs)[-top_k:][::-1]
        top_names = [HAND_FEATURE_NAMES[i] for i in top_idx]

        # Per-enzyme mean gradients for top features
        heatmap_data = np.zeros((len(ENZYME_CLASSES), top_k))
        for enz_i, enz_name in enumerate(ENZYME_CLASSES):
            enz_idx_val = le.transform([enz_name])[0]
            enz_mask = enzyme_labels == enz_idx_val
            if enz_mask.sum() > 0:
                enz_grads = gradient_attrs[enz_mask]
                for j, fidx in enumerate(top_idx):
                    heatmap_data[enz_i, j] = np.mean(np.abs(enz_grads[:, fidx]))

        # Normalize per feature
        for j in range(top_k):
            maxv = heatmap_data[:, j].max()
            if maxv > 0:
                heatmap_data[:, j] /= maxv

        im = ax_d.imshow(heatmap_data, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
        ax_d.set_xticks(range(top_k))
        ax_d.set_xticklabels(top_names, rotation=45, ha="right", fontsize=8)
        ax_d.set_yticks(range(len(ENZYME_CLASSES)))
        ax_d.set_yticklabels(ENZYME_CLASSES, fontsize=10)
        ax_d.set_title("D: Gradient attribution (normalized per feature)", fontsize=13, fontweight="bold")
        plt.colorbar(im, ax=ax_d, shrink=0.8, label="Relative importance")
    else:
        ax_d.text(0.5, 0.5, "Gradient attribution\n(computed below)", transform=ax_d.transAxes,
                  ha="center", va="center", fontsize=14)

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved showcase figure to %s", output_path)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Device: %s", DEVICE)

    # ── Load data ──────────────────────────────────────────────────────────
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

    # Load gene map
    gene_map = load_refgene(REFGENE)
    logger.info("Loaded refGene: %d chromosomes", len(gene_map))

    # Filter to sites with embeddings
    site_ids = df["site_id"].values
    has_emb = np.array([str(sid) in emb_orig and str(sid) in emb_edited for sid in site_ids])
    df = df[has_emb].copy().reset_index(drop=True)
    site_ids = df["site_id"].values
    logger.info("Sites with embeddings: %d", len(df))

    # Build tensors
    orig_embs = torch.FloatTensor(np.array([emb_orig[str(sid)].numpy() for sid in site_ids]))
    edited_embs = torch.FloatTensor(np.array([emb_edited[str(sid)].numpy() for sid in site_ids]))

    motif = extract_motif_features(seqs, list(site_ids))
    struct = extract_structure_delta_features(struct_delta, list(site_ids))
    loop = extract_loop_features(loop_df, list(site_ids))
    hand_40 = torch.FloatTensor(np.nan_to_num(np.concatenate([motif, struct, loop], axis=1), nan=0.0))

    binary_labels = torch.FloatTensor(df["is_edited"].values.astype(float))
    le = LabelEncoder()
    le.fit(ENZYME_CLASSES)
    enzyme_labels = np.full(len(df), -1, dtype=int)
    pos_mask_arr = df["is_edited"] == 1
    enzyme_labels[pos_mask_arr] = le.transform(df.loc[pos_mask_arr, "enzyme"].values)
    enzyme_labels_t = torch.LongTensor(enzyme_labels)

    logger.info("Binary: %d pos, %d neg", pos_mask_arr.sum(), (~pos_mask_arr).sum())

    # Per-enzyme site masks
    enzyme_site_mask = {}
    for enz_idx, enz_name in enumerate(le.classes_):
        mask = (enzyme_labels == enz_idx) | (enzyme_labels == -1)
        enzyme_site_mask[enz_name] = np.where(mask)[0]

    # ── Phase 1: 5-fold CV to get ALL per-site predictions ────────────
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 1: 5-fold CV (storing ALL per-site predictions)")
    logger.info("=" * 70)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    strat_labels = enzyme_labels.copy()
    strat_labels[strat_labels == -1] = N_ENZYMES

    unified_scores = np.zeros(len(df))
    unified_reprs = np.zeros((len(df), 256))
    per_enzyme_scores = {enz: np.full(len(df), np.nan) for enz in le.classes_}

    for fold_i, (train_idx, test_idx) in enumerate(skf.split(hand_40.numpy(), strat_labels)):
        logger.info("\n--- Fold %d/5 ---", fold_i + 1)
        torch.manual_seed(SEED + fold_i)
        np.random.seed(SEED + fold_i)
        train_idx, test_idx = np.array(train_idx), np.array(test_idx)

        # Train unified
        unified_data = (orig_embs, edited_embs, hand_40, binary_labels, enzyme_labels_t)
        model_u = UnifiedNetwork().to(DEVICE)
        opt = torch.optim.AdamW(model_u.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)
        for ep in range(20):
            train_unified_epoch(model_u, unified_data, train_idx.copy(), optimizer=opt, phase=1)
            sched.step()
        opt = torch.optim.AdamW(model_u.parameters(), lr=5e-4, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)
        for ep in range(30):
            train_unified_epoch(model_u, unified_data, train_idx.copy(), optimizer=opt, phase=2)
            sched.step()

        scores_u, reprs_u = predict_scores(model_u, orig_embs, edited_embs, hand_40, test_idx, unified=True)
        unified_scores[test_idx] = scores_u
        unified_reprs[test_idx] = reprs_u

        logger.info("  Unified AUROC: %.3f",
                     roc_auc_score(binary_labels[test_idx].numpy(), scores_u))

        # Train per-enzyme
        for enz_idx, enz_name in enumerate(le.classes_):
            enz_indices = enzyme_site_mask[enz_name]
            enz_train = np.intersect1d(train_idx, enz_indices)
            enz_test = np.intersect1d(test_idx, enz_indices)
            if len(enz_train) < 50 or len(enz_test) < 10:
                continue
            y_tr = binary_labels[enz_train].numpy()
            y_te = binary_labels[enz_test].numpy()
            if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
                continue

            model_pe = PerEnzymeNetwork().to(DEVICE)
            opt_pe = torch.optim.AdamW(model_pe.parameters(), lr=1e-3, weight_decay=1e-4)
            sched_pe = torch.optim.lr_scheduler.CosineAnnealingLR(opt_pe, T_max=50)
            for ep in range(50):
                train_perenz_epoch(model_pe, (orig_embs, edited_embs, hand_40, binary_labels),
                                   enz_train.copy(), optimizer=opt_pe)
                sched_pe.step()
            scores_pe, _ = predict_scores(model_pe, orig_embs, edited_embs, hand_40, enz_test, unified=False)
            per_enzyme_scores[enz_name][enz_test] = scores_pe
            logger.info("  Per-enzyme %s AUROC: %.3f", enz_name,
                         roc_auc_score(y_te, scores_pe))
            del model_pe; gc.collect()

        del model_u; gc.collect()
        if DEVICE == "mps":
            torch.mps.empty_cache()

    # ── Save all per-site scores ──────────────────────────────────────────
    scores_df = df[["site_id", "chr", "start", "strand", "enzyme", "dataset_source", "is_edited"]].copy()
    scores_df["unified_score"] = unified_scores
    for enz_name in le.classes_:
        scores_df[f"perenz_score_{enz_name}"] = per_enzyme_scores[enz_name]
    scores_df.to_csv(OUTPUT_DIR / "all_per_site_scores_v2.csv", index=False)
    logger.info("Saved all per-site scores (%d sites)", len(scores_df))

    # ── Phase 2: Comprehensive rescued/lost analysis ──────────────────
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 2: Biological analysis of rescued and lost sites")
    logger.info("=" * 70)

    threshold = 0.5
    all_results = {"v2_analysis": {}}
    rescued_masks = {}
    lost_masks = {}

    for enz_name in le.classes_:
        valid = ~np.isnan(per_enzyme_scores[enz_name])
        if valid.sum() < 10:
            continue

        y_true = binary_labels.numpy()
        u_pred = (unified_scores >= threshold).astype(int)
        p_pred = (per_enzyme_scores[enz_name] >= threshold).astype(int)

        # Create masks for valid sites only
        unified_correct = (u_pred == y_true) & valid
        perenz_wrong = (p_pred != y_true) & valid
        rescued = unified_correct & perenz_wrong

        perenz_correct = (p_pred == y_true) & valid
        unified_wrong = (u_pred != y_true) & valid
        lost = perenz_correct & unified_wrong

        rescued_idx = np.where(rescued)[0]
        lost_idx = np.where(lost)[0]
        rescued_masks[enz_name] = rescued_idx
        lost_masks[enz_name] = lost_idx

        logger.info("\n  %s: %d rescued, %d lost (net %+d)",
                     enz_name, len(rescued_idx), len(lost_idx),
                     len(rescued_idx) - len(lost_idx))

        # Analyze rescued positives (is_edited==1 that went from wrong to right)
        rescued_pos_mask = rescued & (df["is_edited"] == 1).values
        lost_pos_mask = lost & (df["is_edited"] == 1).values
        rescued_neg_mask = rescued & (df["is_edited"] == 0).values
        lost_neg_mask = lost & (df["is_edited"] == 0).values

        n_rescued_pos = rescued_pos_mask.sum()
        n_rescued_neg = rescued_neg_mask.sum()
        n_lost_pos = lost_pos_mask.sum()
        n_lost_neg = lost_neg_mask.sum()

        logger.info("    Rescued: %d positives, %d negatives", n_rescued_pos, n_rescued_neg)
        logger.info("    Lost:    %d positives, %d negatives", n_lost_pos, n_lost_neg)

        # Detailed analysis for rescued positives
        rescued_stats = analyze_rescued_vs_nonrescued(
            df, hand_40, seqs, site_ids, gene_map,
            pd.Series(rescued_pos_mask), enz_name, "rescued"
        )

        # Detailed analysis for lost positives
        lost_stats = analyze_rescued_vs_nonrescued(
            df, hand_40, seqs, site_ids, gene_map,
            pd.Series(lost_pos_mask), enz_name, "lost"
        )

        # Analysis for ALL positives (baseline)
        enz_idx_val = le.transform([enz_name])[0]
        all_pos_this_enz = (enzyme_labels == enz_idx_val) & (df["is_edited"] == 1).values
        all_pos_stats = analyze_rescued_vs_nonrescued(
            df, hand_40, seqs, site_ids, gene_map,
            pd.Series(all_pos_this_enz), enz_name, "all"
        )

        enz_analysis = {
            "n_rescued_pos": int(n_rescued_pos),
            "n_rescued_neg": int(n_rescued_neg),
            "n_lost_pos": int(n_lost_pos),
            "n_lost_neg": int(n_lost_neg),
            "rescued_pos_stats": rescued_stats,
            "lost_pos_stats": lost_stats,
            "all_pos_stats": all_pos_stats,
        }

        # Log key comparisons
        if rescued_stats and all_pos_stats:
            logger.info("    Rescued TC%%: %.1f%% vs All TC%%: %.1f%%",
                         rescued_stats.get("tc_pct", 0), all_pos_stats.get("tc_pct", 0))
            logger.info("    Rescued CC%%: %.1f%% vs All CC%%: %.1f%%",
                         rescued_stats.get("cc_pct", 0), all_pos_stats.get("cc_pct", 0))
            logger.info("    Rescued is_unpaired: %.2f vs All: %.2f",
                         rescued_stats.get("is_unpaired_mean", 0),
                         all_pos_stats.get("is_unpaired_mean", 0))
            logger.info("    Rescued RLP: %.3f vs All: %.3f",
                         rescued_stats.get("relative_loop_position_mean", 0),
                         all_pos_stats.get("relative_loop_position_mean", 0))
            logger.info("    Rescued loop_size: %.1f vs All: %.1f",
                         rescued_stats.get("loop_size_mean", 0),
                         all_pos_stats.get("loop_size_mean", 0))
            logger.info("    Rescued datasets: %s", rescued_stats.get("datasets", {}))
            logger.info("    Rescued top genes: %s",
                         list(rescued_stats.get("top_genes", {}).keys())[:10])

        all_results["v2_analysis"][enz_name] = enz_analysis

    # ── Phase 3: A3B specific "why does it not benefit" analysis ──────
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 3: A3B deep-dive -- why no benefit from unified training?")
    logger.info("=" * 70)

    enz_name = "A3B"
    valid = ~np.isnan(per_enzyme_scores[enz_name])
    enz_idx_val = le.transform([enz_name])[0]
    a3b_pos = (enzyme_labels == enz_idx_val) & valid

    # Sites where per-enzyme is RIGHT but unified is WRONG (lost A3B positives)
    a3b_lost_pos = lost_masks.get(enz_name, np.array([]))
    a3b_lost_pos_edited = np.array([i for i in a3b_lost_pos if binary_labels[i] == 1])

    logger.info("A3B lost positives: %d", len(a3b_lost_pos_edited))

    if len(a3b_lost_pos_edited) > 0:
        # Motif analysis
        lost_motifs = [get_motif_5p(seqs.get(str(site_ids[i]), "N" * 201)) for i in a3b_lost_pos_edited]
        lost_mc = Counter(lost_motifs)

        all_a3b_pos_idx = np.where(a3b_pos & (df["is_edited"] == 1).values)[0]
        all_motifs = [get_motif_5p(seqs.get(str(site_ids[i]), "N" * 201)) for i in all_a3b_pos_idx]
        all_mc = Counter(all_motifs)

        logger.info("  Lost A3B motif distribution: %s", dict(lost_mc))
        logger.info("  All A3B motif distribution: %s", dict(all_mc))

        # Structure comparison
        for feat_name in ["is_unpaired", "loop_size", "relative_loop_position",
                          "local_unpaired_fraction", "left_stem_length"]:
            fidx = LOOP_FEAT_INDICES[feat_name]
            lost_vals = hand_40[a3b_lost_pos_edited, fidx].numpy()
            all_vals = hand_40[all_a3b_pos_idx, fidx].numpy()
            logger.info("  %s: lost=%.3f, all=%.3f", feat_name,
                         np.mean(lost_vals), np.mean(all_vals))

        # Dataset source
        lost_datasets = df.iloc[a3b_lost_pos_edited]["dataset_source"].value_counts()
        logger.info("  Lost A3B by dataset: %s", lost_datasets.to_dict())

        # Score distributions
        lost_u = unified_scores[a3b_lost_pos_edited]
        lost_p = per_enzyme_scores[enz_name][a3b_lost_pos_edited]
        logger.info("  Lost A3B scores: unified=%.3f+/-%.3f, perenz=%.3f+/-%.3f",
                     np.mean(lost_u), np.std(lost_u), np.mean(lost_p), np.std(lost_p))

        all_results["a3b_deep_dive"] = {
            "n_lost_pos": len(a3b_lost_pos_edited),
            "lost_motifs": dict(lost_mc),
            "all_motifs": dict(all_mc),
            "lost_tc_pct": lost_mc.get("UC", 0) / max(len(lost_motifs), 1) * 100,
            "all_tc_pct": all_mc.get("UC", 0) / max(len(all_motifs), 1) * 100,
            "structure_comparison": {},
            "lost_datasets": df.iloc[a3b_lost_pos_edited]["dataset_source"].value_counts().to_dict(),
        }
        for feat_name in ["is_unpaired", "loop_size", "relative_loop_position",
                          "local_unpaired_fraction", "left_stem_length"]:
            fidx = LOOP_FEAT_INDICES[feat_name]
            all_results["a3b_deep_dive"]["structure_comparison"][feat_name] = {
                "lost_mean": float(np.mean(hand_40[a3b_lost_pos_edited, fidx].numpy())),
                "all_mean": float(np.mean(hand_40[all_a3b_pos_idx, fidx].numpy())),
            }

        # Gene analysis for lost A3B
        lost_genes = []
        for i in a3b_lost_pos_edited[:200]:
            chrom = df.iloc[i].get("chr", "")
            pos = df.iloc[i].get("start", 0)
            if pd.notna(chrom) and pd.notna(pos) and chrom != "":
                gene = lookup_gene(gene_map, str(chrom), int(pos))
                if gene:
                    lost_genes.append(gene)
        all_results["a3b_deep_dive"]["lost_genes"] = dict(Counter(lost_genes).most_common(15))

    # ── Phase 4: Gradient attribution for showcase figure ─────────────
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 4: Gradient attribution (final model)")
    logger.info("=" * 70)

    torch.manual_seed(SEED)
    model_final = UnifiedNetwork().to(DEVICE)
    unified_data = (orig_embs, edited_embs, hand_40, binary_labels, enzyme_labels_t)
    all_indices = np.arange(len(df))

    opt = torch.optim.AdamW(model_final.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)
    for ep in range(20):
        train_unified_epoch(model_final, unified_data, all_indices.copy(), optimizer=opt, phase=1)
        sched.step()
    opt = torch.optim.AdamW(model_final.parameters(), lr=5e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)
    for ep in range(30):
        train_unified_epoch(model_final, unified_data, all_indices.copy(), optimizer=opt, phase=2)
        sched.step()

    # Gradient attribution
    model_final.eval()
    pos_indices = np.where(pos_mask_arr.values)[0]
    all_grads = []
    for start in range(0, len(pos_indices), 256):
        idx = pos_indices[start:start + 256]
        o = orig_embs[idx].to(DEVICE)
        e = edited_embs[idx].to(DEVICE)
        h = hand_40[idx].to(DEVICE).requires_grad_(True)
        bin_logit, _, _ = model_final(o, e, h)
        bin_logit.sum().backward()
        all_grads.append(h.grad.cpu().numpy())
    gradient_attrs_pos = np.concatenate(all_grads, axis=0)

    # Map back to full array
    gradient_attrs = np.zeros((len(df), 40))
    gradient_attrs[pos_indices] = gradient_attrs_pos

    logger.info("Gradient attribution computed for %d positive sites", len(pos_indices))

    del model_final; gc.collect()

    # ── Phase 5: Create showcase figure ──────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 5: Create showcase figure")
    logger.info("=" * 70)

    create_showcase_figure(
        df, unified_scores, per_enzyme_scores, hand_40,
        binary_labels.numpy(), enzyme_labels, le, seqs, site_ids,
        enzyme_site_mask, rescued_masks, lost_masks,
        OUTPUT_DIR / "showcase_figure.png",
        gradient_attrs=gradient_attrs,
    )

    # ── Phase 6: Gene enrichment for specific rescued sites ──────────
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 6: Notable rescued sites with gene context")
    logger.info("=" * 70)

    notable_sites = []
    for enz_name in ["A3A", "A3B", "A3G"]:
        rescued_idx = rescued_masks.get(enz_name, np.array([]))
        rescued_pos_idx = [i for i in rescued_idx if binary_labels[i] == 1]

        # Sort by score difference
        if len(rescued_pos_idx) == 0:
            continue

        score_diffs = [(i, unified_scores[i] - per_enzyme_scores[enz_name][i])
                       for i in rescued_pos_idx]
        score_diffs.sort(key=lambda x: x[1], reverse=True)

        for i, diff in score_diffs[:15]:
            chrom = str(df.iloc[i].get("chr", ""))
            pos = df.iloc[i].get("start", 0)
            gene = lookup_gene(gene_map, chrom, int(pos)) if pd.notna(pos) and chrom else None
            ctx = get_context_5nt(seqs.get(str(site_ids[i]), "N" * 201))
            motif = get_motif_5p(seqs.get(str(site_ids[i]), "N" * 201))
            is_unp = hand_40[i, LOOP_FEAT_INDICES["is_unpaired"]].item()
            rlp = hand_40[i, LOOP_FEAT_INDICES["relative_loop_position"]].item()
            ls = hand_40[i, LOOP_FEAT_INDICES["loop_size"]].item()
            ds = df.iloc[i].get("dataset_source", "")

            notable_sites.append({
                "enzyme": enz_name,
                "site_id": str(site_ids[i]),
                "chr": chrom,
                "pos": int(pos) if pd.notna(pos) else 0,
                "gene": gene or "intergenic",
                "motif_5p": motif,
                "context_5nt": ctx,
                "unified_score": float(unified_scores[i]),
                "perenz_score": float(per_enzyme_scores[enz_name][i]),
                "score_diff": float(diff),
                "is_unpaired": float(is_unp),
                "rlp": float(rlp),
                "loop_size": float(ls),
                "dataset_source": ds,
            })

    # Log notable sites
    for site in notable_sites[:30]:
        logger.info("  %s %s %s:%d gene=%s motif=%s unified=%.3f perenz=%.3f unp=%s rlp=%.2f ls=%.0f",
                     site["enzyme"], site["site_id"], site["chr"], site["pos"],
                     site["gene"], site["motif_5p"],
                     site["unified_score"], site["perenz_score"],
                     "Y" if site["is_unpaired"] > 0.5 else "N",
                     site["rlp"], site["loop_size"])

    all_results["notable_rescued_sites"] = notable_sites

    # ── Save results ──────────────────────────────────────────────────────
    with open(OUTPUT_DIR / "unified_interpretability_v2_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("\nSaved v2 results to %s", OUTPUT_DIR / "unified_interpretability_v2_results.json")

    elapsed = time.time() - t0
    logger.info("\nTotal time: %.1f minutes", elapsed / 60)


if __name__ == "__main__":
    main()
