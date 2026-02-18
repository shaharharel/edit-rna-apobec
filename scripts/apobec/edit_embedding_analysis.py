#!/usr/bin/env python
"""Edit embedding interpretability analysis for the trained EditRNA-A3A model.

Loads the trained Exp 2b model (Levanon + tiered TC-motif negatives) and
performs comprehensive analysis of the edit embedding space:

1. Extract edit embeddings and fused representations for all sites
2. Dimensionality reduction (UMAP, t-SNE) with multiple colorings
3. Clustering analysis (K-means, HDBSCAN) with composition breakdown
4. Linear probing to measure information content vs raw RNA-FM features
5. Cross-attention weight analysis (position importance)
6. Fusion gate weight analysis

Outputs all figures and numerical results to:
    experiments/apobec/outputs/exp4_interpretability/

Usage:
    python scripts/apobec/edit_embedding_analysis.py
"""

import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.apobec_dataset import (
    APOBECDataConfig,
    APOBECDataset,
    APOBECSiteSample,
    N_TISSUES,
    apobec_collate_fn,
    compute_concordance_features,
    encode_apobec_class,
    encode_exonic_function,
    encode_structure_type,
    encode_tissue_spec,
    get_flanking_context,
)
from models.editrna_a3a import EditRNA_A3A, EditRNAConfig
from models.encoders import CachedRNAEncoder

logger = logging.getLogger(__name__)

# Paths
CHECKPOINT_PATH = PROJECT_ROOT / "experiments" / "apobec" / "outputs" / "exp2_levanon_tiered_negatives" / "best_model.pt"
CONFIG_PATH = PROJECT_ROOT / "experiments" / "apobec" / "outputs" / "exp2_levanon_tiered_negatives" / "config.json"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "processed" / "embeddings"
SPLITS_LEVANON = PROJECT_ROOT / "data" / "processed" / "splits_levanon_expanded.csv"
SPLITS_EXPANDED = PROJECT_ROOT / "data" / "processed" / "splits_expanded.csv"
LABELS_CSV = PROJECT_ROOT / "data" / "processed" / "editing_sites_labels.csv"
SEQUENCES_JSON = PROJECT_ROOT / "data" / "processed" / "site_sequences.json"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "apobec" / "outputs" / "exp4_interpretability"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_config():
    """Load experiment config."""
    with open(CONFIG_PATH) as f:
        return json.load(f)


def load_model(config: dict, device: torch.device) -> EditRNA_A3A:
    """Load the trained model from checkpoint."""
    # Load cached embeddings
    logger.info("Loading cached embeddings from %s", EMBEDDINGS_DIR)
    tokens_cache = torch.load(EMBEDDINGS_DIR / "rnafm_tokens.pt", weights_only=False, map_location="cpu")
    pooled_cache = torch.load(EMBEDDINGS_DIR / "rnafm_pooled.pt", weights_only=False, map_location="cpu")
    tokens_edited = torch.load(EMBEDDINGS_DIR / "rnafm_tokens_edited.pt", weights_only=False, map_location="cpu")
    pooled_edited = torch.load(EMBEDDINGS_DIR / "rnafm_pooled_edited.pt", weights_only=False, map_location="cpu")
    logger.info("  Loaded %d original + %d edited embedding caches", len(tokens_cache), len(tokens_edited))

    cached_encoder = CachedRNAEncoder(
        tokens_cache=tokens_cache,
        pooled_cache=pooled_cache,
        tokens_edited_cache=tokens_edited,
        pooled_edited_cache=pooled_edited,
        d_model=config["d_model"],
    )

    model_config = EditRNAConfig(
        primary_encoder="cached",
        d_model=config["d_model"],
        d_edit=config["d_edit"],
        d_fused=config["d_fused"],
        edit_n_heads=config["edit_n_heads"],
        use_structure_delta=config["use_structure_delta"],
        use_dual_encoder=config.get("use_dual_encoder", False),
        use_gnn=config.get("use_gnn", False),
        finetune_last_n=0,
        head_dropout=config["head_dropout"],
        fusion_dropout=config["fusion_dropout"],
        focal_gamma=config["focal_gamma"],
        focal_alpha_binary=config["focal_alpha_binary"],
        focal_alpha_conservation=config.get("focal_alpha_conservation", 0.85),
    )
    model = EditRNA_A3A(config=model_config, primary_encoder=cached_encoder)

    # Load checkpoint
    logger.info("Loading checkpoint from %s", CHECKPOINT_PATH)
    ckpt = torch.load(CHECKPOINT_PATH, weights_only=False, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device)
    model.eval()
    logger.info("  Model loaded (best epoch: %d)", ckpt.get("epoch", -1))

    return model


def build_dataset(config: dict) -> tuple:
    """Build the full dataset from the splits CSV used for training.

    Returns (dataset, metadata_df) where metadata_df has per-sample info
    for coloring plots.
    """
    splits_path = Path(config["splits_csv"])
    if not splits_path.is_absolute():
        splits_path = PROJECT_ROOT / splits_path
    splits_df = pd.read_csv(splits_path)
    logger.info("Loaded splits: %d sites from %s", len(splits_df), splits_path)

    # Load sequences
    sequences = None
    if SEQUENCES_JSON.exists():
        with open(SEQUENCES_JSON) as f:
            sequences = json.load(f)
        logger.info("Loaded %d sequences", len(sequences))

    # Filter to sites with cached embeddings
    site_ids_path = EMBEDDINGS_DIR / "rnafm_site_ids.json"
    if site_ids_path.exists():
        with open(site_ids_path) as f:
            cached_ids = set(json.load(f))
        n_before = len(splits_df)
        splits_df = splits_df[splits_df["site_id"].isin(cached_ids)].reset_index(drop=True)
        logger.info("Filtered to cached sites: %d -> %d", n_before, len(splits_df))

    # Load rich labels for Levanon sites
    levanon_labels = {}
    if LABELS_CSV.exists():
        labels_df = pd.read_csv(LABELS_CSV)
        levanon_labels = {row["site_id"]: row for _, row in labels_df.iterrows()}
        logger.info("Loaded %d Levanon labels for enrichment", len(levanon_labels))

    window_size = config.get("window_size", 100)
    samples = []
    metadata_rows = []

    for _, row in splits_df.iterrows():
        site_id = str(row["site_id"])
        label = int(row["label"])
        split = row.get("split", "train")
        dataset_source = str(row.get("dataset_source", "unknown"))

        # Get sequence
        seq = None
        if sequences and site_id in sequences:
            seq = sequences[site_id]
        else:
            # Generate synthetic for sites without sequence
            rng = np.random.RandomState(hash(site_id) % (2**31))
            nucs = ["A", "C", "G", "U"]
            seq = "".join(rng.choice(nucs, size=window_size * 2 + 1))

        edit_pos = min(window_size, len(seq) // 2)
        seq_list = list(seq)
        seq_list[edit_pos] = "C"
        seq = "".join(seq_list)

        flanking = get_flanking_context(seq, edit_pos)

        # Determine TC motif presence
        has_tc = flanking == 0  # TC context = index 0

        if label == 1 and site_id in levanon_labels:
            lrow = levanon_labels[site_id]
            apobec_class = encode_apobec_class(lrow.get("apobec_class", ""))
            structure_type = encode_structure_type(lrow.get("structure_type", ""))
            tissue_spec_class = encode_tissue_spec(lrow.get("tissue_class", ""))
            concordance = compute_concordance_features(
                lrow.get("structure_type_mRNA", ""),
                lrow.get("structure_type_premRNA", ""),
            )
        else:
            apobec_class = -1
            structure_type = -1
            tissue_spec_class = -1
            concordance = np.zeros(5, dtype=np.float32)
            if label == 0:
                concordance[0] = 1.0

        sample = APOBECSiteSample(
            sequence=seq,
            edit_pos=edit_pos,
            is_edited=float(label),
            editing_rate_log2=float("nan"),
            apobec_class=apobec_class,
            structure_type=structure_type,
            tissue_spec_class=tissue_spec_class,
            n_tissues_log2=float("nan"),
            exonic_function=-1,
            conservation=float("nan"),
            cancer_survival=float("nan"),
            tissue_rates=np.full(N_TISSUES, np.nan, dtype=np.float32),
            hek293_rate=float("nan"),
            flanking_context=flanking,
            concordance_features=concordance,
            site_id=site_id,
        )
        samples.append(sample)

        metadata_rows.append({
            "site_id": site_id,
            "label": label,
            "split": split,
            "dataset_source": dataset_source,
            "has_tc_motif": bool(has_tc),
            "flanking_context": flanking,
        })

    data_config = APOBECDataConfig(window_size=window_size)
    dataset = APOBECDataset(samples, data_config)
    metadata_df = pd.DataFrame(metadata_rows)

    logger.info("Built dataset: %d samples (%d pos, %d neg)",
                len(dataset), dataset.n_positive, dataset.n_negative)

    return dataset, metadata_df


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def extract_embeddings(
    model: EditRNA_A3A,
    dataset: APOBECDataset,
    device: torch.device,
    batch_size: int = 64,
) -> dict:
    """Run forward pass and extract intermediate representations.

    Returns dict with numpy arrays:
        edit_embeddings: (N, d_edit)
        fused_representations: (N, d_fused)
        primary_pooled: (N, d_model)
        attention_weights: list of (seq_len,) arrays -- cross-attention weights per sample
        gate_weights: (N, n_modalities)
        binary_logits: (N,)
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=apobec_collate_fn,
        num_workers=0,
    )

    all_edit_emb = []
    all_fused = []
    all_primary_pooled = []
    all_attn_weights = []
    all_gate_weights = []
    all_binary_logits = []

    model.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            # Move to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
                elif isinstance(v, dict):
                    batch[k] = {kk: vv.to(device) if isinstance(vv, torch.Tensor) else vv
                                for kk, vv in v.items()}

            sequences = batch["sequences"]
            edit_pos = batch["edit_pos"]
            flanking_context = batch["flanking_context"]
            concordance_features = batch["concordance_features"]
            structure_delta = batch["structure_delta"]
            site_ids = batch.get("site_ids")

            # Step 1: Encode
            primary_out = model._encode_primary(sequences, site_ids=site_ids, edited=False)
            f_background = primary_out["tokens"]
            primary_pooled = primary_out["pooled"]

            edited_sequences = model._make_edited_sequences(sequences, edit_pos)
            edited_out = model._encode_primary(edited_sequences, site_ids=site_ids, edited=True)
            f_edited = edited_out["tokens"]

            min_len = min(f_background.shape[1], f_edited.shape[1])
            f_background = f_background[:, :min_len, :]
            f_edited = f_edited[:, :min_len, :]
            seq_mask = torch.ones(f_background.shape[0], min_len, dtype=torch.bool, device=device)

            # Step 2: Edit embedding (with attention weight extraction)
            edit_emb_module = model.edit_embedding
            B = f_background.shape[0]
            d_model = edit_emb_module.d_model

            pos_idx = edit_pos.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, d_model)
            f_bg_at_pos = f_background.gather(1, pos_idx).squeeze(1)

            query = f_bg_at_pos.unsqueeze(1)
            key_padding_mask = ~seq_mask

            # Get attention weights from cross-attention
            context_out, attn_w = edit_emb_module.context_attention(
                query, f_background, f_background,
                key_padding_mask=key_padding_mask,
                need_weights=True,
                average_attn_weights=True,
            )
            # attn_w shape: (B, 1, seq_len) -> squeeze to (B, seq_len)
            attn_w_np = attn_w.squeeze(1).cpu().numpy()
            for i in range(B):
                all_attn_weights.append(attn_w_np[i])

            # Full forward for edit embedding
            edit_emb = edit_emb_module(
                f_background=f_background,
                f_edited=f_edited,
                edit_pos=edit_pos,
                flanking_context=flanking_context,
                structure_delta=structure_delta,
                concordance_features=concordance_features,
                seq_mask=seq_mask,
            )

            # Step 3: Fusion
            fused = model.fusion(
                primary_pooled=primary_pooled,
                edit_emb=edit_emb,
            )

            # Step 4: Gate weights
            gate_w = model.fusion.get_gate_weights(
                primary_pooled=primary_pooled,
                edit_emb=edit_emb,
            )

            # Step 5: Predictions
            predictions = model.heads(fused, edit_emb)
            binary_logits = predictions["binary_logit"].squeeze(-1)

            all_edit_emb.append(edit_emb.cpu().numpy())
            all_fused.append(fused.cpu().numpy())
            all_primary_pooled.append(primary_pooled.cpu().numpy())
            all_gate_weights.append(gate_w.cpu().numpy())
            all_binary_logits.append(binary_logits.cpu().numpy())

            if (batch_idx + 1) % 20 == 0:
                logger.info("  Processed %d / %d batches", batch_idx + 1, len(loader))

    result = {
        "edit_embeddings": np.concatenate(all_edit_emb, axis=0),
        "fused_representations": np.concatenate(all_fused, axis=0),
        "primary_pooled": np.concatenate(all_primary_pooled, axis=0),
        "attention_weights": all_attn_weights,
        "gate_weights": np.concatenate(all_gate_weights, axis=0),
        "binary_logits": np.concatenate(all_binary_logits, axis=0),
    }

    logger.info("Extracted embeddings: edit=%s, fused=%s, pooled=%s",
                result["edit_embeddings"].shape,
                result["fused_representations"].shape,
                result["primary_pooled"].shape)

    return result


# ---------------------------------------------------------------------------
# Analysis 1: Dimensionality reduction
# ---------------------------------------------------------------------------

def run_dimensionality_reduction(
    embeddings: dict,
    metadata: pd.DataFrame,
    output_dir: Path,
) -> dict:
    """Run UMAP and t-SNE on edit embeddings with multiple colorings."""
    from sklearn.manifold import TSNE

    results = {}
    edit_emb = embeddings["edit_embeddings"]

    # -- t-SNE --
    logger.info("Running t-SNE on edit embeddings (n=%d, d=%d)...", *edit_emb.shape)
    tsne = TSNE(n_components=2, perplexity=min(30, len(edit_emb) - 1),
                random_state=42, max_iter=1000)
    tsne_coords = tsne.fit_transform(edit_emb)
    results["tsne_coords"] = tsne_coords.tolist()

    # -- UMAP --
    try:
        import umap
        logger.info("Running UMAP on edit embeddings...")
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        umap_coords = reducer.fit_transform(edit_emb)
        results["umap_coords"] = umap_coords.tolist()
        has_umap = True
    except ImportError:
        logger.warning("umap-learn not installed; skipping UMAP")
        umap_coords = None
        has_umap = False

    # Colorings
    labels = metadata["label"].values
    sources = metadata["dataset_source"].values
    tc_motif = metadata["has_tc_motif"].values.astype(int)

    # --- Plot function ---
    def _plot_scatter(coords, color_vals, color_name, cmap_or_colors, title, filename,
                      legend_labels=None, is_categorical=True):
        fig, ax = plt.subplots(figsize=(10, 8))
        if is_categorical:
            unique_vals = sorted(set(color_vals))
            # Auto-generate colors if dict is empty or not a dict
            auto_cmap = plt.cm.get_cmap("tab10", max(len(unique_vals), 1))
            auto_colors = {val: auto_cmap(i) for i, val in enumerate(unique_vals)}
            color_map = cmap_or_colors if isinstance(cmap_or_colors, dict) and cmap_or_colors else auto_colors

            for val in unique_vals:
                mask = color_vals == val
                lbl = legend_labels.get(val, str(val)) if legend_labels else str(val)
                ax.scatter(coords[mask, 0], coords[mask, 1],
                          c=[color_map.get(val, auto_colors[val])], label=lbl,
                          alpha=0.6, s=12, edgecolors="none")
            ax.legend(fontsize=9, markerscale=2, framealpha=0.7)
        else:
            sc = ax.scatter(coords[:, 0], coords[:, 1], c=color_vals,
                           cmap=cmap_or_colors, alpha=0.6, s=12, edgecolors="none")
            plt.colorbar(sc, ax=ax, label=color_name)

        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Dimension 1", fontsize=11)
        ax.set_ylabel("Dimension 2", fontsize=11)
        ax.tick_params(labelsize=9)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info("  Saved %s", filename)

    label_colors = {0: "#4477AA", 1: "#EE6677"}
    label_labels = {0: "Negative (unedited)", 1: "Positive (edited)"}
    tc_colors = {0: "#BBBBBB", 1: "#228833"}
    tc_labels = {0: "Non-TC context", 1: "TC context"}

    # t-SNE plots
    _plot_scatter(tsne_coords, labels, "Label", label_colors,
                  "t-SNE of Edit Embeddings (by label)", "tsne_by_label.png",
                  legend_labels=label_labels)

    _plot_scatter(tsne_coords, sources, "Dataset", {},
                  "t-SNE of Edit Embeddings (by dataset)", "tsne_by_dataset.png")

    _plot_scatter(tsne_coords, tc_motif, "TC motif", tc_colors,
                  "t-SNE of Edit Embeddings (by TC motif)", "tsne_by_tc_motif.png",
                  legend_labels=tc_labels)

    # Binary logit coloring (continuous)
    binary_probs = 1.0 / (1.0 + np.exp(-embeddings["binary_logits"]))
    _plot_scatter(tsne_coords, binary_probs, "P(edited)",
                  "RdYlGn", "t-SNE colored by P(edited)", "tsne_by_prob.png",
                  is_categorical=False)

    # UMAP plots
    if has_umap:
        _plot_scatter(umap_coords, labels, "Label", label_colors,
                      "UMAP of Edit Embeddings (by label)", "umap_by_label.png",
                      legend_labels=label_labels)

        _plot_scatter(umap_coords, sources, "Dataset", {},
                      "UMAP of Edit Embeddings (by dataset)", "umap_by_dataset.png")

        _plot_scatter(umap_coords, tc_motif, "TC motif", tc_colors,
                      "UMAP of Edit Embeddings (by TC motif)", "umap_by_tc_motif.png",
                      legend_labels=tc_labels)

        _plot_scatter(umap_coords, binary_probs, "P(edited)",
                      "RdYlGn", "UMAP colored by P(edited)", "umap_by_prob.png",
                      is_categorical=False)

    return results


# ---------------------------------------------------------------------------
# Analysis 2: Clustering
# ---------------------------------------------------------------------------

def run_clustering_analysis(
    embeddings: dict,
    metadata: pd.DataFrame,
    output_dir: Path,
) -> dict:
    """K-means and optional HDBSCAN clustering of edit embeddings."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    edit_emb = embeddings["edit_embeddings"]
    results = {}

    # K-means with various k
    silhouettes = []
    best_k = 2
    best_score = -1
    for k in range(2, 11):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = km.fit_predict(edit_emb)
        score = silhouette_score(edit_emb, cluster_labels, sample_size=min(5000, len(edit_emb)))
        silhouettes.append({"k": k, "silhouette": float(score)})
        if score > best_score:
            best_score = score
            best_k = k

    results["silhouette_scores"] = silhouettes
    results["best_k"] = best_k
    logger.info("Best K-means: k=%d (silhouette=%.3f)", best_k, best_score)

    # Use best k for analysis
    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cluster_labels = km.fit_predict(edit_emb)
    metadata_with_clusters = metadata.copy()
    metadata_with_clusters["cluster"] = cluster_labels

    # Cluster composition
    cluster_composition = []
    for c in range(best_k):
        mask = cluster_labels == c
        n_total = mask.sum()
        n_pos = (metadata["label"].values[mask] == 1).sum()
        n_neg = (metadata["label"].values[mask] == 0).sum()
        pos_frac = n_pos / n_total if n_total > 0 else 0

        source_counts = pd.Series(metadata["dataset_source"].values[mask]).value_counts().to_dict()
        tc_frac = metadata["has_tc_motif"].values[mask].mean()

        cluster_composition.append({
            "cluster": int(c),
            "n_total": int(n_total),
            "n_positive": int(n_pos),
            "n_negative": int(n_neg),
            "positive_fraction": float(pos_frac),
            "tc_motif_fraction": float(tc_frac),
            "dataset_source_counts": {str(k): int(v) for k, v in source_counts.items()},
        })

    results["cluster_composition"] = cluster_composition

    # Plot silhouette scores
    fig, ax = plt.subplots(figsize=(8, 5))
    ks = [s["k"] for s in silhouettes]
    sils = [s["silhouette"] for s in silhouettes]
    ax.plot(ks, sils, "o-", color="#4477AA", linewidth=2)
    ax.axvline(best_k, color="#EE6677", linestyle="--", alpha=0.7, label=f"Best k={best_k}")
    ax.set_xlabel("Number of clusters (k)", fontsize=11)
    ax.set_ylabel("Silhouette score", fontsize=11)
    ax.set_title("K-means Silhouette Analysis", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "kmeans_silhouette.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Plot cluster composition
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Positive fraction per cluster
    clusters = [c["cluster"] for c in cluster_composition]
    pos_fracs = [c["positive_fraction"] for c in cluster_composition]
    sizes = [c["n_total"] for c in cluster_composition]

    ax = axes[0]
    bars = ax.bar(clusters, pos_fracs, color="#228833", alpha=0.8, edgecolor="black", linewidth=0.5)
    for bar, s in zip(bars, sizes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"n={s}", ha="center", fontsize=8)
    ax.set_xlabel("Cluster", fontsize=11)
    ax.set_ylabel("Positive fraction", fontsize=11)
    ax.set_title("Positive fraction per cluster", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.axhline(metadata["label"].mean(), color="#EE6677", linestyle="--", alpha=0.7, label="Overall rate")
    ax.legend(fontsize=9)

    # TC motif fraction per cluster
    tc_fracs = [c["tc_motif_fraction"] for c in cluster_composition]
    ax = axes[1]
    bars = ax.bar(clusters, tc_fracs, color="#CCBB44", alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Cluster", fontsize=11)
    ax.set_ylabel("TC motif fraction", fontsize=11)
    ax.set_title("TC motif fraction per cluster", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.axhline(metadata["has_tc_motif"].mean(), color="#EE6677", linestyle="--", alpha=0.7, label="Overall rate")
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(output_dir / "cluster_composition.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # HDBSCAN (optional)
    try:
        import hdbscan
        logger.info("Running HDBSCAN...")
        clusterer = hdbscan.HDBSCAN(min_cluster_size=max(15, len(edit_emb) // 100),
                                      min_samples=5, metric="euclidean")
        hdb_labels = clusterer.fit_predict(edit_emb)
        n_clusters_hdb = len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0)
        n_noise = (hdb_labels == -1).sum()
        results["hdbscan"] = {
            "n_clusters": int(n_clusters_hdb),
            "n_noise": int(n_noise),
        }
        logger.info("HDBSCAN: %d clusters, %d noise points", n_clusters_hdb, n_noise)
    except ImportError:
        logger.warning("hdbscan not installed; skipping")

    return results


# ---------------------------------------------------------------------------
# Analysis 3: Linear probing
# ---------------------------------------------------------------------------

def run_linear_probing(
    embeddings: dict,
    metadata: pd.DataFrame,
    output_dir: Path,
) -> dict:
    """Train linear probes on edit embedding vs raw RNA-FM pooled to predict labels."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    results = {}
    labels = metadata["label"].values

    # Representations to probe
    representations = {
        "edit_embedding": embeddings["edit_embeddings"],
        "fused_representation": embeddings["fused_representations"],
        "raw_rnafm_pooled": embeddings["primary_pooled"],
    }

    # Tasks to probe
    probe_tasks = {
        "is_edited": labels,
    }

    # Dataset source as a task (encode as integers)
    source_to_int = {s: i for i, s in enumerate(sorted(metadata["dataset_source"].unique()))}
    probe_tasks["dataset_source"] = metadata["dataset_source"].map(source_to_int).values

    for task_name, task_labels in probe_tasks.items():
        results[task_name] = {}

        # Filter valid samples
        valid = ~pd.isna(task_labels)
        y = task_labels[valid].astype(int)

        if len(np.unique(y)) < 2:
            logger.warning("Skipping probe for %s: only one class", task_name)
            continue

        is_binary = len(np.unique(y)) == 2

        for repr_name, X_full in representations.items():
            X = X_full[valid]

            # 5-fold cross-validation
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            fold_metrics = []

            for train_idx, test_idx in skf.split(X, y):
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X[train_idx])
                X_test = scaler.transform(X[test_idx])
                y_train, y_test = y[train_idx], y[test_idx]

                if is_binary:
                    clf = LogisticRegression(max_iter=500, C=1.0, random_state=42)
                    clf.fit(X_train, y_train)
                    y_prob = clf.predict_proba(X_test)[:, 1]
                    y_pred = clf.predict(X_test)
                    auroc = roc_auc_score(y_test, y_prob)
                    acc = accuracy_score(y_test, y_pred)
                    fold_metrics.append({"auroc": auroc, "accuracy": acc})
                else:
                    clf = LogisticRegression(max_iter=500, C=1.0, random_state=42)
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)
                    fold_metrics.append({"accuracy": acc})

            # Average metrics
            avg_metrics = {}
            for key in fold_metrics[0]:
                vals = [f[key] for f in fold_metrics]
                avg_metrics[key] = float(np.mean(vals))
                avg_metrics[f"{key}_std"] = float(np.std(vals))

            results[task_name][repr_name] = avg_metrics
            logger.info("  Probe %s -> %s: %s", repr_name, task_name,
                       ", ".join(f"{k}={v:.3f}" for k, v in avg_metrics.items() if not k.endswith("_std")))

    # Plot comparison
    if "is_edited" in results and len(results["is_edited"]) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        repr_names = list(results["is_edited"].keys())
        aurocs = [results["is_edited"][r].get("auroc", 0) for r in repr_names]
        auroc_stds = [results["is_edited"][r].get("auroc_std", 0) for r in repr_names]

        # Clean names for display
        display_names = [n.replace("_", " ").title() for n in repr_names]

        bars = ax.bar(range(len(repr_names)), aurocs, yerr=auroc_stds,
                     color=["#4477AA", "#228833", "#EE6677"][:len(repr_names)],
                     alpha=0.8, edgecolor="black", linewidth=0.5, capsize=5)
        ax.set_xticks(range(len(repr_names)))
        ax.set_xticklabels(display_names, fontsize=10)
        ax.set_ylabel("AUROC (5-fold CV)", fontsize=11)
        ax.set_title("Linear Probe: is_edited Classification", fontsize=13)
        ax.set_ylim(0.5, 1.0)
        ax.grid(axis="y", alpha=0.3)

        for bar, val in zip(bars, aurocs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", fontsize=10, fontweight="bold")

        fig.tight_layout()
        fig.savefig(output_dir / "linear_probe_auroc.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    return results


# ---------------------------------------------------------------------------
# Analysis 4: Attention pattern analysis
# ---------------------------------------------------------------------------

def run_attention_analysis(
    embeddings: dict,
    metadata: pd.DataFrame,
    dataset: APOBECDataset,
    output_dir: Path,
) -> dict:
    """Analyze cross-attention weight patterns in the edit embedding module."""
    attn_weights = embeddings["attention_weights"]
    labels = metadata["label"].values
    results = {}

    # Compute relative position attention profiles
    # For each sample, we have attention weights over sequence positions.
    # We want to know how attention relates to distance from the edit site.
    window_size = dataset.config.window_size  # typically 100

    # Define a window of positions to aggregate over
    max_relative_pos = 50  # look +/- 50 positions around edit
    pos_range = np.arange(-max_relative_pos, max_relative_pos + 1)

    pos_attn = np.zeros(len(pos_range), dtype=np.float64)
    neg_attn = np.zeros(len(pos_range), dtype=np.float64)
    pos_count = 0
    neg_count = 0

    for i, (aw, lbl) in enumerate(zip(attn_weights, labels)):
        edit_pos = dataset.samples[i].edit_pos
        seq_len = len(aw)

        profile = np.zeros(len(pos_range))
        for j, rel_pos in enumerate(pos_range):
            abs_pos = edit_pos + rel_pos
            if 0 <= abs_pos < seq_len:
                profile[j] = aw[abs_pos]

        if lbl == 1:
            pos_attn += profile
            pos_count += 1
        else:
            neg_attn += profile
            neg_count += 1

    if pos_count > 0:
        pos_attn /= pos_count
    if neg_count > 0:
        neg_attn /= neg_count

    results["positive_attention_profile"] = pos_attn.tolist()
    results["negative_attention_profile"] = neg_attn.tolist()
    results["relative_positions"] = pos_range.tolist()

    # Plot attention profiles
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Overlay
    ax = axes[0]
    ax.plot(pos_range, pos_attn, color="#EE6677", linewidth=1.5, label="Positive (edited)", alpha=0.8)
    ax.plot(pos_range, neg_attn, color="#4477AA", linewidth=1.5, label="Negative (unedited)", alpha=0.8)
    ax.axvline(0, color="black", linestyle="--", alpha=0.5, label="Edit site")
    ax.set_xlabel("Position relative to edit site", fontsize=11)
    ax.set_ylabel("Mean attention weight", fontsize=11)
    ax.set_title("Cross-Attention Profile by Label", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Difference
    ax = axes[1]
    diff = pos_attn - neg_attn
    colors = np.where(diff > 0, "#EE6677", "#4477AA")
    ax.bar(pos_range, diff, color=colors, alpha=0.7, width=1.0)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Position relative to edit site", fontsize=11)
    ax.set_ylabel("Attention difference (pos - neg)", fontsize=11)
    ax.set_title("Attention Difference: Positive minus Negative", fontsize=13)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "attention_profiles.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Top attention positions (aggregated)
    all_attn_mean = (pos_attn * pos_count + neg_attn * neg_count) / (pos_count + neg_count)
    top_positions_idx = np.argsort(all_attn_mean)[::-1][:10]
    results["top_attention_positions"] = [
        {"relative_pos": int(pos_range[i]), "mean_attention": float(all_attn_mean[i])}
        for i in top_positions_idx
    ]

    # Attention entropy analysis
    pos_entropies = []
    neg_entropies = []
    for aw, lbl in zip(attn_weights, labels):
        aw_safe = np.clip(aw, 1e-10, None)
        aw_norm = aw_safe / aw_safe.sum()
        entropy = -np.sum(aw_norm * np.log(aw_norm))
        if lbl == 1:
            pos_entropies.append(entropy)
        else:
            neg_entropies.append(entropy)

    results["attention_entropy"] = {
        "positive_mean": float(np.mean(pos_entropies)) if pos_entropies else 0,
        "positive_std": float(np.std(pos_entropies)) if pos_entropies else 0,
        "negative_mean": float(np.mean(neg_entropies)) if neg_entropies else 0,
        "negative_std": float(np.std(neg_entropies)) if neg_entropies else 0,
    }

    # Plot entropy histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(pos_entropies, bins=40, alpha=0.6, color="#EE6677", label="Positive", density=True)
    ax.hist(neg_entropies, bins=40, alpha=0.6, color="#4477AA", label="Negative", density=True)
    ax.set_xlabel("Attention entropy", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Cross-Attention Entropy Distribution", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "attention_entropy.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    logger.info("Attention entropy: pos=%.3f +/- %.3f, neg=%.3f +/- %.3f",
                results["attention_entropy"]["positive_mean"],
                results["attention_entropy"]["positive_std"],
                results["attention_entropy"]["negative_mean"],
                results["attention_entropy"]["negative_std"])

    return results


# ---------------------------------------------------------------------------
# Analysis 5: Gate weight analysis
# ---------------------------------------------------------------------------

def run_gate_analysis(
    embeddings: dict,
    metadata: pd.DataFrame,
    output_dir: Path,
) -> dict:
    """Analyze fusion gate weights across samples."""
    gate_weights = embeddings["gate_weights"]  # (N, n_modalities)
    labels = metadata["label"].values
    results = {}

    n_modalities = gate_weights.shape[1]
    modality_names = ["Primary (RNA-FM)", "Edit Embedding"]
    if n_modalities > 2:
        modality_names.append("Secondary")
    if n_modalities > 3:
        modality_names.append("GNN")

    # Overall distribution
    mean_weights = gate_weights.mean(axis=0)
    std_weights = gate_weights.std(axis=0)
    results["overall_gate_weights"] = {
        name: {"mean": float(mean_weights[i]), "std": float(std_weights[i])}
        for i, name in enumerate(modality_names)
    }

    # By label
    pos_mask = labels == 1
    neg_mask = labels == 0

    results["gate_by_label"] = {
        "positive": {
            name: {"mean": float(gate_weights[pos_mask, i].mean()),
                   "std": float(gate_weights[pos_mask, i].std())}
            for i, name in enumerate(modality_names)
        },
        "negative": {
            name: {"mean": float(gate_weights[neg_mask, i].mean()),
                   "std": float(gate_weights[neg_mask, i].std())}
            for i, name in enumerate(modality_names)
        },
    }

    # Plot gate weights
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart by label
    ax = axes[0]
    x = np.arange(n_modalities)
    width = 0.35

    pos_means = [gate_weights[pos_mask, i].mean() for i in range(n_modalities)]
    pos_stds = [gate_weights[pos_mask, i].std() for i in range(n_modalities)]
    neg_means = [gate_weights[neg_mask, i].mean() for i in range(n_modalities)]
    neg_stds = [gate_weights[neg_mask, i].std() for i in range(n_modalities)]

    ax.bar(x - width/2, pos_means, width, yerr=pos_stds, label="Positive",
           color="#EE6677", alpha=0.8, capsize=5, edgecolor="black", linewidth=0.5)
    ax.bar(x + width/2, neg_means, width, yerr=neg_stds, label="Negative",
           color="#4477AA", alpha=0.8, capsize=5, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(modality_names, fontsize=10)
    ax.set_ylabel("Gate weight", fontsize=11)
    ax.set_title("Fusion Gate Weights by Label", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # Distribution plots
    ax = axes[1]
    for i, name in enumerate(modality_names):
        ax.hist(gate_weights[:, i], bins=40, alpha=0.5, label=name, density=True)
    ax.set_xlabel("Gate weight", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Gate Weight Distributions", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "gate_weights.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    return results


# ---------------------------------------------------------------------------
# Analysis 6: Summary figure
# ---------------------------------------------------------------------------

def create_summary_figure(
    embeddings: dict,
    metadata: pd.DataFrame,
    dim_red_results: dict,
    probe_results: dict,
    attention_results: dict,
    gate_results: dict,
    output_dir: Path,
):
    """Create a multi-panel summary figure."""
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    labels = metadata["label"].values
    label_colors = {0: "#4477AA", 1: "#EE6677"}

    # Panel A: t-SNE by label
    ax = fig.add_subplot(gs[0, 0])
    tsne_coords = np.array(dim_red_results["tsne_coords"])
    for lbl, color in label_colors.items():
        mask = labels == lbl
        name = "Positive" if lbl == 1 else "Negative"
        ax.scatter(tsne_coords[mask, 0], tsne_coords[mask, 1],
                  c=color, label=name, alpha=0.5, s=8, edgecolors="none")
    ax.set_title("A) t-SNE of Edit Embeddings", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, markerscale=2)
    ax.set_xticks([])
    ax.set_yticks([])

    # Panel B: Attention profiles
    ax = fig.add_subplot(gs[0, 1])
    pos_range = np.array(attention_results["relative_positions"])
    pos_attn = np.array(attention_results["positive_attention_profile"])
    neg_attn = np.array(attention_results["negative_attention_profile"])
    ax.plot(pos_range, pos_attn, color="#EE6677", linewidth=1.2, label="Positive", alpha=0.8)
    ax.plot(pos_range, neg_attn, color="#4477AA", linewidth=1.2, label="Negative", alpha=0.8)
    ax.axvline(0, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Position relative to edit", fontsize=10)
    ax.set_ylabel("Attention weight", fontsize=10)
    ax.set_title("B) Cross-Attention Profile", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel C: Gate weights
    ax = fig.add_subplot(gs[0, 2])
    gate_w = embeddings["gate_weights"]
    n_mod = gate_w.shape[1]
    mod_names = ["RNA-FM", "Edit Emb."][:n_mod]
    pos_mask = labels == 1
    neg_mask = labels == 0
    x = np.arange(n_mod)
    width = 0.35
    ax.bar(x - width/2, [gate_w[pos_mask, i].mean() for i in range(n_mod)],
           width, color="#EE6677", alpha=0.8, label="Positive",
           edgecolor="black", linewidth=0.5)
    ax.bar(x + width/2, [gate_w[neg_mask, i].mean() for i in range(n_mod)],
           width, color="#4477AA", alpha=0.8, label="Negative",
           edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(mod_names, fontsize=10)
    ax.set_ylabel("Gate weight", fontsize=10)
    ax.set_title("C) Fusion Gate Weights", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Panel D: Linear probe comparison
    ax = fig.add_subplot(gs[1, 0])
    if "is_edited" in probe_results:
        repr_names = list(probe_results["is_edited"].keys())
        aurocs = [probe_results["is_edited"][r].get("auroc", 0) for r in repr_names]
        auroc_stds = [probe_results["is_edited"][r].get("auroc_std", 0) for r in repr_names]
        display_names = ["Edit Emb.", "Fused", "RNA-FM"][:len(repr_names)]
        colors = ["#4477AA", "#228833", "#EE6677"][:len(repr_names)]
        bars = ax.bar(range(len(repr_names)), aurocs, yerr=auroc_stds,
                     color=colors, alpha=0.8, capsize=5, edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(repr_names)))
        ax.set_xticklabels(display_names, fontsize=10)
        ax.set_ylim(0.5, 1.05)
        for bar, val in zip(bars, aurocs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylabel("AUROC", fontsize=10)
    ax.set_title("D) Linear Probe: is_edited", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Panel E: Attention entropy
    ax = fig.add_subplot(gs[1, 1])
    entropy_data = attention_results.get("attention_entropy", {})
    cats = ["Positive", "Negative"]
    means = [entropy_data.get("positive_mean", 0), entropy_data.get("negative_mean", 0)]
    stds = [entropy_data.get("positive_std", 0), entropy_data.get("negative_std", 0)]
    bars = ax.bar(cats, means, yerr=stds, color=["#EE6677", "#4477AA"],
                 alpha=0.8, capsize=5, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Attention entropy", fontsize=10)
    ax.set_title("E) Attention Focus", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Panel F: t-SNE by prediction probability
    ax = fig.add_subplot(gs[1, 2])
    binary_probs = 1.0 / (1.0 + np.exp(-embeddings["binary_logits"]))
    sc = ax.scatter(tsne_coords[:, 0], tsne_coords[:, 1], c=binary_probs,
                   cmap="RdYlGn", alpha=0.5, s=8, edgecolors="none")
    plt.colorbar(sc, ax=ax, label="P(edited)")
    ax.set_title("F) t-SNE by P(edited)", fontsize=12, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])

    fig.suptitle("EditRNA-A3A: Edit Embedding Interpretability Analysis",
                fontsize=15, fontweight="bold", y=0.98)
    fig.savefig(output_dir / "summary_figure.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved summary figure")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Run the full interpretability analysis."""
    # Setup
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    # 1. Load config and model
    logger.info("=" * 60)
    logger.info("Edit Embedding Interpretability Analysis")
    logger.info("=" * 60)

    config = load_config()
    model = load_model(config, device)

    # 2. Build dataset
    logger.info("\n--- Building dataset ---")
    dataset, metadata = build_dataset(config)

    # 3. Extract embeddings
    logger.info("\n--- Extracting embeddings ---")
    embeddings = extract_embeddings(model, dataset, device, batch_size=64)

    # Save raw embeddings
    np.save(OUTPUT_DIR / "edit_embeddings.npy", embeddings["edit_embeddings"])
    np.save(OUTPUT_DIR / "fused_representations.npy", embeddings["fused_representations"])
    np.save(OUTPUT_DIR / "primary_pooled.npy", embeddings["primary_pooled"])
    np.save(OUTPUT_DIR / "gate_weights.npy", embeddings["gate_weights"])
    metadata.to_csv(OUTPUT_DIR / "metadata.csv", index=False)
    logger.info("Saved raw embeddings and metadata")

    # 4. Dimensionality reduction
    logger.info("\n--- Dimensionality reduction ---")
    dim_red_results = run_dimensionality_reduction(embeddings, metadata, OUTPUT_DIR)

    # 5. Clustering
    logger.info("\n--- Clustering analysis ---")
    cluster_results = run_clustering_analysis(embeddings, metadata, OUTPUT_DIR)

    # 6. Linear probing
    logger.info("\n--- Linear probing ---")
    probe_results = run_linear_probing(embeddings, metadata, OUTPUT_DIR)

    # 7. Attention analysis
    logger.info("\n--- Attention analysis ---")
    attention_results = run_attention_analysis(embeddings, metadata, dataset, OUTPUT_DIR)

    # 8. Gate analysis
    logger.info("\n--- Gate weight analysis ---")
    gate_results = run_gate_analysis(embeddings, metadata, OUTPUT_DIR)

    # 9. Summary figure
    logger.info("\n--- Creating summary figure ---")
    create_summary_figure(embeddings, metadata, dim_red_results, probe_results,
                         attention_results, gate_results, OUTPUT_DIR)

    # 10. Save all numerical results
    all_results = {
        "dimensionality_reduction": {
            k: v for k, v in dim_red_results.items()
            if k not in ("tsne_coords", "umap_coords")  # skip large arrays
        },
        "clustering": cluster_results,
        "linear_probing": probe_results,
        "attention_analysis": attention_results,
        "gate_analysis": gate_results,
        "dataset_summary": {
            "n_samples": len(dataset),
            "n_positive": dataset.n_positive,
            "n_negative": dataset.n_negative,
            "edit_embedding_dim": int(embeddings["edit_embeddings"].shape[1]),
            "fused_dim": int(embeddings["fused_representations"].shape[1]),
            "n_modalities": int(embeddings["gate_weights"].shape[1]),
        },
    }

    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x))
    logger.info("Saved numerical results to %s", results_path)

    logger.info("\n" + "=" * 60)
    logger.info("Analysis complete. All outputs saved to: %s", OUTPUT_DIR)
    logger.info("=" * 60)

    # Print key findings
    logger.info("\n--- Key Findings ---")
    if "is_edited" in probe_results:
        for repr_name, metrics in probe_results["is_edited"].items():
            logger.info("  %s probe AUROC: %.3f +/- %.3f",
                       repr_name, metrics.get("auroc", 0), metrics.get("auroc_std", 0))

    logger.info("  Best K-means k: %d", cluster_results.get("best_k", -1))

    for name, vals in gate_results.get("overall_gate_weights", {}).items():
        logger.info("  Gate weight %s: %.3f +/- %.3f", name, vals["mean"], vals["std"])


if __name__ == "__main__":
    main()
