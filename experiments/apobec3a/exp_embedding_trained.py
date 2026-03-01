#!/usr/bin/env python
"""Contextual Edit Embedding from trained EditRNA-A3A model.

Loads the trained EditRNA-A3A model, runs inference on ALL sites in the
dataset to extract the `fused` representation (d_fused=512) from
GatedModalityFusion -- the "contextual edit embedding" -- and produces
t-SNE and UMAP dimensionality reduction visualizations colored by label,
dataset source, and editing rate.

Usage:
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    KMP_DUPLICATE_LIB_OK=TRUE \
    /Users/shaharharel/miniconda3/envs/quris/bin/python \
    experiments/apobec3a/exp_embedding_trained.py
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

EMB_DIR = PROJECT_ROOT / "data" / "processed" / "embeddings"
SPLITS_CSV = PROJECT_ROOT / "data" / "processed" / "splits_expanded_a3a.csv"
STRUCT_CACHE = EMB_DIR / "vienna_structure_cache.npz"
MODEL_PATH = PROJECT_ROOT / "experiments" / "apobec" / "outputs" / "baselines" / "editrna" / "best_model.pt"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "apobec" / "outputs" / "embedding_trained"

DATASET_LABELS = {
    "advisor_c2t": "Advisor",
    "asaoka_2019": "Asaoka",
    "sharma_2015": "Sharma",
    "alqassim_2021": "Alqassim",
    "tier2_negative": "Tier2 Neg",
    "tier3_negative": "Tier3 Neg",
}

DATASET_COLORS = {
    "Advisor": "#2563eb",
    "Asaoka": "#16a34a",
    "Sharma": "#dc2626",
    "Alqassim": "#d97706",
    "Tier2 Neg": "#6b7280",
    "Tier3 Neg": "#374151",
}

BATCH_SIZE = 64
MAX_TSNE = 3000

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
})


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model():
    """Load the trained EditRNA-A3A model with CachedRNAEncoder."""
    from models.editrna_a3a import EditRNA_A3A, EditRNAConfig
    from models.encoders import CachedRNAEncoder

    logger.info("Loading embedding caches...")
    pooled_cache = torch.load(EMB_DIR / "rnafm_pooled.pt", weights_only=False)
    pooled_edited = torch.load(EMB_DIR / "rnafm_pooled_edited.pt", weights_only=False)

    tokens_path = EMB_DIR / "rnafm_tokens.pt"
    tokens_edited_path = EMB_DIR / "rnafm_tokens_edited.pt"
    if tokens_path.exists():
        tokens_cache = torch.load(tokens_path, weights_only=False)
        tokens_edited = torch.load(tokens_edited_path, weights_only=False)
        pooled_only = False
        logger.info("Loaded token-level embeddings (%d sites)", len(tokens_cache))
    else:
        tokens_cache = None
        tokens_edited = None
        pooled_only = True
        logger.warning("Token embeddings not found -- using pooled-only mode")

    logger.info("Loaded %d pooled embeddings", len(pooled_cache))

    cached_encoder = CachedRNAEncoder(
        tokens_cache=tokens_cache,
        pooled_cache=pooled_cache,
        tokens_edited_cache=tokens_edited,
        pooled_edited_cache=pooled_edited,
        d_model=640,
    )

    model_config = EditRNAConfig(
        primary_encoder="cached",
        d_model=640,
        pooled_only=pooled_only,
    )

    model = EditRNA_A3A(config=model_config, primary_encoder=cached_encoder)
    logger.info("Loading model weights from %s", MODEL_PATH)
    state_dict = torch.load(MODEL_PATH, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model loaded: %s parameters", f"{n_params:,}")

    return model, pooled_cache


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(pooled_cache):
    """Load splits, structure deltas, and determine available site_ids."""
    splits_df = pd.read_csv(SPLITS_CSV)
    logger.info("Loaded splits: %d sites", len(splits_df))

    # Structure delta features
    structure_delta = {}
    if STRUCT_CACHE.exists():
        data = np.load(STRUCT_CACHE, allow_pickle=True)
        feat_key = "delta_features" if "delta_features" in data else "features"
        if "site_ids" in data and feat_key in data:
            sids = data["site_ids"]
            feats = data[feat_key]
            structure_delta = {str(sid): feats[i] for i, sid in enumerate(sids)}
            logger.info("Loaded %d structure delta features", len(structure_delta))

    # Filter to available sites
    available_ids = set(pooled_cache.keys())
    splits_df = splits_df[splits_df["site_id"].isin(available_ids)].copy()
    site_ids = splits_df["site_id"].tolist()
    logger.info("Sites with embeddings: %d", len(site_ids))

    return splits_df, site_ids, structure_delta


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_fused_embeddings(model, site_ids, structure_delta):
    """Run inference in batches and extract fused embeddings (B, 512)."""
    all_fused = []
    n_batches = (len(site_ids) + BATCH_SIZE - 1) // BATCH_SIZE

    logger.info("Extracting fused embeddings from %d sites in %d batches...",
                len(site_ids), n_batches)

    for i in range(0, len(site_ids), BATCH_SIZE):
        batch_sids = site_ids[i:i + BATCH_SIZE]
        B = len(batch_sids)

        # Build structure delta tensor
        sd_list = []
        for sid in batch_sids:
            if sid in structure_delta:
                sd_list.append(torch.tensor(structure_delta[sid], dtype=torch.float32))
            else:
                sd_list.append(torch.zeros(7, dtype=torch.float32))
        structure_delta_tensor = torch.stack(sd_list)

        editrna_batch = {
            "sequences": ["N" * 201] * B,
            "site_ids": batch_sids,
            "edit_pos": torch.full((B,), 100, dtype=torch.long),
            "flanking_context": torch.zeros(B, dtype=torch.long),
            "concordance_features": torch.zeros(B, 5),
            "structure_delta": structure_delta_tensor,
        }

        output = model(editrna_batch)
        all_fused.append(output["fused"].cpu())

        if (i // BATCH_SIZE + 1) % 50 == 0 or (i + BATCH_SIZE) >= len(site_ids):
            logger.info("  Batch %d/%d done", i // BATCH_SIZE + 1, n_batches)

    fused_matrix = torch.cat(all_fused, dim=0)
    logger.info("Fused embeddings shape: %s", tuple(fused_matrix.shape))
    return fused_matrix


# ---------------------------------------------------------------------------
# Dimensionality reduction
# ---------------------------------------------------------------------------

def run_dimensionality_reduction(fused_np):
    """Run t-SNE and UMAP on the fused embeddings."""
    from sklearn.decomposition import PCA

    results = {}

    # PCA first for dimensionality reduction (speed up t-SNE/UMAP)
    logger.info("Running PCA (50 components) for preprocessing...")
    pca = PCA(n_components=min(50, fused_np.shape[1], fused_np.shape[0]))
    pca_50 = pca.fit_transform(fused_np)
    logger.info("PCA variance explained (top 10): %s",
                [f"{v:.3f}" for v in pca.explained_variance_ratio_[:10]])

    # Subsample for t-SNE / UMAP
    n = len(fused_np)
    if n > MAX_TSNE:
        idx = np.random.RandomState(42).choice(n, MAX_TSNE, replace=False)
        idx = np.sort(idx)
        logger.info("Subsampling %d/%d sites for t-SNE/UMAP", MAX_TSNE, n)
    else:
        idx = np.arange(n)
    results["subsample_idx"] = idx

    pca_sub = pca_50[idx, :30]

    # t-SNE
    logger.info("Running t-SNE on %d samples...", len(idx))
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42, n_jobs=1)
    results["tsne_2d"] = tsne.fit_transform(pca_sub)

    # UMAP
    try:
        import umap
        logger.info("Running UMAP on %d samples...", len(idx))
        reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.3, random_state=42)
        results["umap_2d"] = reducer.fit_transform(pca_sub)
    except ImportError:
        logger.warning("umap-learn not installed. Skipping UMAP.")
        results["umap_2d"] = None

    return results


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_tsne_label(coords, labels, output_path):
    """t-SNE colored by label (positive=blue, negative=red)."""
    fig = plt.figure(figsize=(10, 8), dpi=150)
    ax = fig.add_subplot(111)

    neg_mask = labels == 0
    pos_mask = labels == 1

    ax.scatter(coords[neg_mask, 0], coords[neg_mask, 1],
               s=8, alpha=0.4, c="#dc2626", label="Negative", edgecolors="none")
    ax.scatter(coords[pos_mask, 0], coords[pos_mask, 1],
               s=8, alpha=0.3, c="#2563eb", label="Positive", edgecolors="none")

    ax.legend(fontsize=10, markerscale=2, loc="best", framealpha=0.8)
    ax.set_title("Contextual Edit Embedding (t-SNE) -- Label", fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Saved %s", output_path.name)


def plot_tsne_dataset(coords, dataset_labels, output_path):
    """t-SNE colored by dataset source."""
    fig = plt.figure(figsize=(10, 8), dpi=150)
    ax = fig.add_subplot(111)

    for label in sorted(set(dataset_labels)):
        mask = np.array(dataset_labels) == label
        color = DATASET_COLORS.get(label, None)
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   s=8, alpha=0.4, label=label, c=color, edgecolors="none")

    ax.legend(fontsize=8, markerscale=2, loc="best", framealpha=0.8)
    ax.set_title("Contextual Edit Embedding (t-SNE) -- Dataset", fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Saved %s", output_path.name)


def plot_umap_label(coords, labels, output_path):
    """UMAP colored by label (positive=blue, negative=red)."""
    fig = plt.figure(figsize=(10, 8), dpi=150)
    ax = fig.add_subplot(111)

    neg_mask = labels == 0
    pos_mask = labels == 1

    ax.scatter(coords[neg_mask, 0], coords[neg_mask, 1],
               s=8, alpha=0.4, c="#dc2626", label="Negative", edgecolors="none")
    ax.scatter(coords[pos_mask, 0], coords[pos_mask, 1],
               s=8, alpha=0.3, c="#2563eb", label="Positive", edgecolors="none")

    ax.legend(fontsize=10, markerscale=2, loc="best", framealpha=0.8)
    ax.set_title("Contextual Edit Embedding (UMAP) -- Label", fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Saved %s", output_path.name)


def plot_umap_dataset(coords, dataset_labels, output_path):
    """UMAP colored by dataset source."""
    fig = plt.figure(figsize=(10, 8), dpi=150)
    ax = fig.add_subplot(111)

    for label in sorted(set(dataset_labels)):
        mask = np.array(dataset_labels) == label
        color = DATASET_COLORS.get(label, None)
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   s=8, alpha=0.4, label=label, c=color, edgecolors="none")

    ax.legend(fontsize=8, markerscale=2, loc="best", framealpha=0.8)
    ax.set_title("Contextual Edit Embedding (UMAP) -- Dataset", fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Saved %s", output_path.name)


def plot_umap_rate(coords, rates, output_path):
    """UMAP colored by editing rate (log scale) for sites with rates."""
    fig = plt.figure(figsize=(10, 8), dpi=150)
    ax = fig.add_subplot(111)

    valid_rate = ~np.isnan(rates) & (rates > 0)
    log_rates = np.full_like(rates, np.nan)
    log_rates[valid_rate] = np.log10(np.clip(rates[valid_rate], 0.01, None))

    if valid_rate.sum() > 0:
        # Plot sites without rates as gray background
        ax.scatter(coords[~valid_rate, 0], coords[~valid_rate, 1],
                   s=4, alpha=0.15, c="gray", edgecolors="none", label="No rate")

        sc = ax.scatter(coords[valid_rate, 0], coords[valid_rate, 1],
                        s=8, alpha=0.5, c=log_rates[valid_rate], cmap="viridis",
                        edgecolors="none")
        plt.colorbar(sc, ax=ax, shrink=0.8, label="log10(editing rate %)")

    ax.set_title("Contextual Edit Embedding (UMAP) -- Editing Rate", fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Saved %s", output_path.name)


# ---------------------------------------------------------------------------
# Cluster metrics
# ---------------------------------------------------------------------------

def compute_cluster_metrics(fused_np, labels):
    """Compute silhouette score and semantic margin between pos/neg."""
    pos_mask = labels == 1
    neg_mask = labels == 0

    pos_centroid = fused_np[pos_mask].mean(axis=0)
    neg_centroid = fused_np[neg_mask].mean(axis=0)

    semantic_margin = float(np.linalg.norm(pos_centroid - neg_centroid))

    # Silhouette score (subsample for speed)
    n = len(labels)
    max_sil = 5000
    if n > max_sil:
        rng = np.random.RandomState(42)
        idx = rng.choice(n, max_sil, replace=False)
        sil_labels = labels[idx]
        sil_data = fused_np[idx]
    else:
        sil_labels = labels
        sil_data = fused_np

    sil = silhouette_score(sil_data, sil_labels)

    return {
        "silhouette_score": float(sil),
        "semantic_margin": semantic_margin,
        "pos_centroid_norm": float(np.linalg.norm(pos_centroid)),
        "neg_centroid_norm": float(np.linalg.norm(neg_centroid)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    torch.manual_seed(42)

    # --- Load model ---
    model, pooled_cache = load_model()

    # --- Load data ---
    splits_df, site_ids, structure_delta = load_data(pooled_cache)

    # --- Extract fused embeddings ---
    fused_tensor = extract_fused_embeddings(model, site_ids, structure_delta)
    fused_np = fused_tensor.numpy()

    # Save raw embeddings for reuse
    emb_save_path = OUTPUT_DIR / "contextual_edit_embeddings.pt"
    torch.save({"site_ids": site_ids, "fused": fused_tensor}, emb_save_path)
    logger.info("Saved raw embeddings to %s", emb_save_path)

    # --- Build metadata arrays ---
    sid_to_ds = dict(zip(splits_df["site_id"], splits_df["dataset_source"]))
    sid_to_label = dict(zip(splits_df["site_id"], splits_df["label"]))
    sid_to_rate = dict(zip(splits_df["site_id"],
                           pd.to_numeric(splits_df["editing_rate"], errors="coerce")))

    labels = np.array([sid_to_label.get(sid, 0) for sid in site_ids], dtype=int)
    dataset_labels = [DATASET_LABELS.get(sid_to_ds.get(sid, ""), "Unknown") for sid in site_ids]
    rates = np.array([sid_to_rate.get(sid, np.nan) for sid in site_ids], dtype=float)

    n_positive = int((labels == 1).sum())
    n_negative = int((labels == 0).sum())
    logger.info("Sites: %d total (%d positive, %d negative)", len(site_ids), n_positive, n_negative)

    # --- Cluster metrics ---
    logger.info("Computing cluster metrics...")
    cluster_metrics = compute_cluster_metrics(fused_np, labels)
    logger.info("Silhouette score (pos vs neg): %.4f", cluster_metrics["silhouette_score"])
    logger.info("Semantic margin (centroid distance): %.4f", cluster_metrics["semantic_margin"])

    # --- Dimensionality reduction ---
    dr_results = run_dimensionality_reduction(fused_np)
    idx = dr_results["subsample_idx"]

    sub_labels = labels[idx]
    sub_dataset_labels = [dataset_labels[i] for i in idx]
    sub_rates = rates[idx]

    # --- Visualizations ---
    logger.info("Generating visualizations...")

    # t-SNE plots
    plot_tsne_label(
        dr_results["tsne_2d"], sub_labels,
        OUTPUT_DIR / "contextual_edit_tsne_label.png",
    )
    plot_tsne_dataset(
        dr_results["tsne_2d"], sub_dataset_labels,
        OUTPUT_DIR / "contextual_edit_tsne_dataset.png",
    )

    # UMAP plots
    if dr_results["umap_2d"] is not None:
        plot_umap_label(
            dr_results["umap_2d"], sub_labels,
            OUTPUT_DIR / "contextual_edit_umap_label.png",
        )
        plot_umap_dataset(
            dr_results["umap_2d"], sub_dataset_labels,
            OUTPUT_DIR / "contextual_edit_umap_dataset.png",
        )
        plot_umap_rate(
            dr_results["umap_2d"], sub_rates,
            OUTPUT_DIR / "contextual_edit_umap_rate.png",
        )

    # --- Save results JSON ---
    results = {
        "n_sites": len(site_ids),
        "n_positive": n_positive,
        "n_negative": n_negative,
        "embedding_type": "Contextual Edit Embedding (GatedModalityFusion, 512d)",
        "model_source": "EditRNA-A3A (best_model.pt, AUROC=0.956)",
        "cluster_metrics": cluster_metrics,
        "semantic_margin": cluster_metrics["semantic_margin"],
        "dimensionality_reduction": {
            "n_subsampled": len(idx),
            "tsne_perplexity": 30,
            "umap_available": dr_results["umap_2d"] is not None,
        },
    }

    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved results to %s", results_path)

    # --- Print summary ---
    print("\n" + "=" * 80)
    print("CONTEXTUAL EDIT EMBEDDING ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"\n{'Metric':<45} {'Value':>15}")
    print("-" * 62)
    print(f"{'Total sites':<45} {len(site_ids):>15,}")
    print(f"{'Positive sites':<45} {n_positive:>15,}")
    print(f"{'Negative sites':<45} {n_negative:>15,}")
    print(f"{'Embedding dimension':<45} {'512':>15}")
    print(f"{'Model source':<45} {'EditRNA-A3A':>15}")
    print(f"{'Silhouette score (pos vs neg)':<45} {cluster_metrics['silhouette_score']:>15.4f}")
    print(f"{'Semantic margin (centroid dist)':<45} {cluster_metrics['semantic_margin']:>15.4f}")
    print(f"{'Positive centroid norm':<45} {cluster_metrics['pos_centroid_norm']:>15.4f}")
    print(f"{'Negative centroid norm':<45} {cluster_metrics['neg_centroid_norm']:>15.4f}")

    print("\n--- Output Files ---")
    for p in sorted(OUTPUT_DIR.iterdir()):
        size_kb = p.stat().st_size / 1024
        print(f"  {p.name:<45} {size_kb:>8.1f} KB")

    print("=" * 80)
    logger.info("All outputs saved to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
