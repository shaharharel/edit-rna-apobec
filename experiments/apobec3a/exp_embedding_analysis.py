#!/usr/bin/env python
"""Embedding space analysis and Sharma failure investigation.

Analyzes the RNA-FM edit embedding space to understand:
1. How editing sites from different datasets cluster
2. Why Sharma sites fail to generalize
3. What structural/motif features distinguish dataset clusters

Usage:
    python experiments/apobec3a/exp_embedding_analysis.py
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logger = logging.getLogger(__name__)

EMB_DIR = PROJECT_ROOT / "data" / "processed" / "embeddings"
SPLITS_CSV = PROJECT_ROOT / "data" / "processed" / "splits_expanded.csv"
STRUCT_CACHE = EMB_DIR / "vienna_structure_cache.npz"
SEQ_JSON = PROJECT_ROOT / "data" / "processed" / "site_sequences.json"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "apobec" / "outputs" / "embedding_analysis"

DATASET_LABELS = {
    "advisor_c2t": "Levanon",
    "asaoka_2019": "Asaoka",
    "sharma_2015": "Sharma",
    "alqassim_2021": "Alqassim",
    "baysal_2016": "Baysal",
    "tier2_negative": "Tier2 Neg",
    "tier3_negative": "Tier3 Neg",
}


def analyze_motifs(sequences, site_ids, splits_df):
    """Analyze sequence motifs around edit site for each dataset."""
    results = {}

    for ds in ["advisor_c2t", "asaoka_2019", "sharma_2015", "alqassim_2021"]:
        ds_ids = set(splits_df[splits_df["dataset_source"] == ds]["site_id"])
        ds_seqs = [sequences[sid] for sid in site_ids if sid in ds_ids and sid in sequences]

        if not ds_seqs:
            continue

        # Analyze -2 to +2 context around center (position 100)
        center = 100
        contexts = {"trinuc": {}, "pentanuc": {}, "upstream_2": {}, "downstream_2": {}}

        for seq in ds_seqs:
            if len(seq) < center + 3:
                continue
            # Trinucleotide (pos -1, 0, +1)
            tri = seq[center-1:center+2]
            contexts["trinuc"][tri] = contexts["trinuc"].get(tri, 0) + 1

            # Pentanucleotide (pos -2 to +2)
            if center >= 2 and center + 3 <= len(seq):
                penta = seq[center-2:center+3]
                contexts["pentanuc"][penta] = contexts["pentanuc"].get(penta, 0) + 1

            # Individual positions
            if center >= 2:
                contexts["upstream_2"][seq[center-2]] = contexts["upstream_2"].get(seq[center-2], 0) + 1
            contexts["downstream_2"][seq[center+2] if center+2 < len(seq) else "N"] = \
                contexts["downstream_2"].get(seq[center+2] if center+2 < len(seq) else "N", 0) + 1

        # Sort by frequency
        for key in contexts:
            contexts[key] = dict(sorted(contexts[key].items(), key=lambda x: -x[1]))

        # TC motif fraction
        n_tc = sum(1 for seq in ds_seqs if len(seq) > center and seq[center-1:center+1] in ("UC", "TC"))
        tc_frac = n_tc / len(ds_seqs) if ds_seqs else 0

        results[DATASET_LABELS[ds]] = {
            "n_sites": len(ds_seqs),
            "tc_motif_fraction": tc_frac,
            "top_trinuc": list(contexts["trinuc"].items())[:10],
            "top_pentanuc": list(contexts["pentanuc"].items())[:10],
        }

    return results


def analyze_structure_differences(struct_data, site_ids, splits_df):
    """Analyze structural feature differences between datasets."""
    sids_arr = struct_data["site_ids"]
    delta = struct_data["delta_features"]
    mfes = struct_data["mfes"]
    mfes_ed = struct_data["mfes_edited"]

    sid_to_idx = {str(sid): i for i, sid in enumerate(sids_arr)}

    results = {}
    for ds in ["advisor_c2t", "asaoka_2019", "sharma_2015", "alqassim_2021",
               "tier2_negative", "tier3_negative"]:
        ds_ids = set(splits_df[splits_df["dataset_source"] == ds]["site_id"])
        indices = [sid_to_idx[sid] for sid in ds_ids if sid in sid_to_idx]

        if not indices:
            continue

        ds_delta = delta[indices]
        ds_mfe = mfes[indices]
        ds_mfe_ed = mfes_ed[indices]

        # Filter out zeros (failed predictions)
        valid = np.abs(ds_delta).sum(axis=1) > 0
        if valid.sum() == 0:
            continue

        ds_delta_valid = ds_delta[valid]
        ds_mfe_valid = ds_mfe[valid & (ds_mfe != 0)]
        ds_mfe_ed_valid = ds_mfe_ed[valid & (ds_mfe_ed != 0)]

        results[DATASET_LABELS[ds]] = {
            "n_sites": len(indices),
            "n_with_structure": int(valid.sum()),
            "mean_delta_pairing": float(ds_delta_valid[:, 0].mean()),
            "mean_delta_accessibility": float(ds_delta_valid[:, 1].mean()),
            "mean_delta_entropy": float(ds_delta_valid[:, 2].mean()),
            "mean_delta_mfe": float(ds_delta_valid[:, 3].mean()),
            "mean_local_delta_pairing": float(ds_delta_valid[:, 4].mean()),
            "std_delta_pairing": float(ds_delta_valid[:, 0].std()),
            "mean_mfe_original": float(ds_mfe_valid.mean()) if len(ds_mfe_valid) > 0 else None,
            "mean_mfe_edited": float(ds_mfe_ed_valid.mean()) if len(ds_mfe_ed_valid) > 0 else None,
        }

    return results


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    splits_df = pd.read_csv(SPLITS_CSV)
    pooled_orig = torch.load(EMB_DIR / "rnafm_pooled.pt", weights_only=False)
    pooled_edited = torch.load(EMB_DIR / "rnafm_pooled_edited.pt", weights_only=False)
    logger.info("Loaded %d pooled embeddings", len(pooled_orig))

    struct_data = np.load(STRUCT_CACHE, allow_pickle=True)

    with open(SEQ_JSON) as f:
        sequences = json.load(f)
    logger.info("Loaded %d sequences", len(sequences))

    # Get site IDs in order
    available_ids = set(pooled_orig.keys()) & set(pooled_edited.keys())
    site_ids = [sid for sid in splits_df["site_id"] if sid in available_ids]
    logger.info("Sites with embeddings: %d", len(site_ids))

    # Build embedding matrices
    orig_matrix = torch.stack([pooled_orig[sid] for sid in site_ids]).numpy()
    edited_matrix = torch.stack([pooled_edited[sid] for sid in site_ids]).numpy()
    diff_matrix = edited_matrix - orig_matrix  # Edit effect embedding (subtraction)

    # Dataset labels for each site
    sid_to_ds = dict(zip(splits_df["site_id"], splits_df["dataset_source"]))
    ds_labels = [DATASET_LABELS.get(sid_to_ds.get(sid, ""), "Unknown") for sid in site_ids]

    # Label (positive/negative) for each site
    sid_to_label = dict(zip(splits_df["site_id"], splits_df["label"]))
    labels = [sid_to_label.get(sid, -1) for sid in site_ids]

    # ===================================================================
    # 1. PCA Analysis
    # ===================================================================
    logger.info("Running PCA on diff embeddings...")
    pca = PCA(n_components=50)
    diff_pca = pca.fit_transform(diff_matrix)

    logger.info("PCA variance explained (top 10): %s",
                [f"{v:.3f}" for v in pca.explained_variance_ratio_[:10]])
    logger.info("Cumulative variance (10 PCs): %.3f",
                sum(pca.explained_variance_ratio_[:10]))
    logger.info("Cumulative variance (50 PCs): %.3f",
                sum(pca.explained_variance_ratio_[:50]))

    # ===================================================================
    # 2. t-SNE on diff embeddings (subsampled for speed)
    # ===================================================================
    MAX_TSNE = 2000
    if len(diff_pca) > MAX_TSNE:
        rng = np.random.RandomState(42)
        tsne_indices = rng.choice(len(diff_pca), MAX_TSNE, replace=False)
        logger.info("Subsampling %d/%d sites for t-SNE...", MAX_TSNE, len(diff_pca))
    else:
        tsne_indices = np.arange(len(diff_pca))

    logger.info("Running t-SNE on %d samples (PCA-50)...", len(tsne_indices))
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    diff_tsne_sub = tsne.fit_transform(diff_pca[tsne_indices])

    # Map back to full array (NaN for non-sampled)
    diff_tsne = np.full((len(diff_pca), 2), np.nan)
    diff_tsne[tsne_indices] = diff_tsne_sub

    # ===================================================================
    # 3. Dataset clustering analysis
    # ===================================================================
    logger.info("Analyzing dataset clustering in embedding space...")

    # Per-dataset centroids and distances
    unique_ds = sorted(set(ds_labels))
    centroids = {}
    for ds in unique_ds:
        mask = [dl == ds for dl in ds_labels]
        centroids[ds] = diff_matrix[mask].mean(axis=0)

    # Inter-dataset distances
    centroid_distances = {}
    for ds1 in unique_ds:
        for ds2 in unique_ds:
            if ds1 <= ds2:
                dist = np.linalg.norm(centroids[ds1] - centroids[ds2])
                centroid_distances[f"{ds1} <-> {ds2}"] = float(dist)

    # Silhouette score (pos datasets only)
    pos_mask = [l == 1 for l in labels]
    pos_ds = [ds_labels[i] for i, m in enumerate(pos_mask) if m]
    pos_diff = diff_matrix[pos_mask]

    if len(set(pos_ds)) > 1:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        pos_ds_encoded = le.fit_transform(pos_ds)
        sil = silhouette_score(pos_diff[:, :50], pos_ds_encoded, sample_size=min(2000, len(pos_ds)))
        logger.info("Silhouette score (positive sites by dataset): %.4f", sil)

    # ===================================================================
    # 4. Sharma vs Others analysis
    # ===================================================================
    logger.info("Analyzing Sharma vs other datasets...")

    sharma_mask = [dl == "Sharma" for dl in ds_labels]
    levanon_mask = [dl == "Levanon" for dl in ds_labels]
    asaoka_mask = [dl == "Asaoka" for dl in ds_labels]
    neg_mask = [dl in ("Tier2 Neg", "Tier3 Neg") for dl in ds_labels]

    # Distances from Sharma centroid to others
    sharma_centroid = diff_matrix[sharma_mask].mean(axis=0)
    levanon_centroid = diff_matrix[levanon_mask].mean(axis=0)
    neg_centroid = diff_matrix[neg_mask].mean(axis=0)

    sharma_to_levanon = np.linalg.norm(sharma_centroid - levanon_centroid)
    sharma_to_neg = np.linalg.norm(sharma_centroid - neg_centroid)
    levanon_to_neg = np.linalg.norm(levanon_centroid - neg_centroid)

    logger.info("Distance Sharma <-> Levanon: %.4f", sharma_to_levanon)
    logger.info("Distance Sharma <-> Negatives: %.4f", sharma_to_neg)
    logger.info("Distance Levanon <-> Negatives: %.4f", levanon_to_neg)

    # Sharma overlap with negatives in PCA space
    sharma_pca = diff_pca[sharma_mask]
    neg_pca = diff_pca[neg_mask]
    levanon_pca = diff_pca[levanon_mask]

    # Distribution overlap (PC1 and PC2)
    for pc_idx, pc_name in [(0, "PC1"), (1, "PC2")]:
        sharma_vals = sharma_pca[:, pc_idx]
        neg_vals = neg_pca[:, pc_idx]
        levanon_vals = levanon_pca[:, pc_idx]

        logger.info("%s distributions - Sharma: [%.2f, %.2f], Neg: [%.2f, %.2f], Levanon: [%.2f, %.2f]",
                    pc_name,
                    sharma_vals.mean(), sharma_vals.std(),
                    neg_vals.mean(), neg_vals.std(),
                    levanon_vals.mean(), levanon_vals.std())

    # ===================================================================
    # 5. Motif analysis
    # ===================================================================
    logger.info("Analyzing sequence motifs...")
    motif_results = analyze_motifs(sequences, site_ids, splits_df)

    for ds_name, info in motif_results.items():
        logger.info("  %s: %d sites, TC-motif fraction=%.3f, top tri: %s",
                    ds_name, info["n_sites"], info["tc_motif_fraction"],
                    info["top_trinuc"][:5])

    # ===================================================================
    # 6. Structure differences
    # ===================================================================
    logger.info("Analyzing structural differences...")
    struct_results = analyze_structure_differences(struct_data, site_ids, splits_df)

    for ds_name, info in struct_results.items():
        logger.info("  %s: delta_mfe=%.4f, delta_pair=%.4f, n=%d",
                    ds_name, info["mean_delta_mfe"], info["mean_delta_pairing"],
                    info["n_with_structure"])

    # ===================================================================
    # Save results
    # ===================================================================

    def serialize(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return str(obj)

    results = {
        "pca": {
            "variance_explained_top10": pca.explained_variance_ratio_[:10].tolist(),
            "cumulative_50pc": float(sum(pca.explained_variance_ratio_[:50])),
        },
        "centroid_distances": centroid_distances,
        "sharma_analysis": {
            "sharma_to_levanon_dist": float(sharma_to_levanon),
            "sharma_to_neg_dist": float(sharma_to_neg),
            "levanon_to_neg_dist": float(levanon_to_neg),
            "sharma_closer_to_neg": bool(sharma_to_neg < levanon_to_neg),
        },
        "motif_analysis": motif_results,
        "structure_analysis": struct_results,
        "dataset_sizes": {DATASET_LABELS.get(ds, ds): int(n) for ds, n
                         in splits_df["dataset_source"].value_counts().items()},
    }

    # Save t-SNE coordinates for plotting
    tsne_df = pd.DataFrame({
        "site_id": site_ids,
        "tsne_x": diff_tsne[:, 0],
        "tsne_y": diff_tsne[:, 1],
        "dataset": ds_labels,
        "label": labels,
    })
    tsne_df.to_csv(OUTPUT_DIR / "tsne_coordinates.csv", index=False)

    with open(OUTPUT_DIR / "embedding_analysis.json", "w") as f:
        json.dump(results, f, indent=2, default=serialize)

    # ===================================================================
    # Print summary
    # ===================================================================
    print("\n" + "=" * 80)
    print("EMBEDDING SPACE ANALYSIS SUMMARY")
    print("=" * 80)

    print("\n--- PCA ---")
    print(f"Top 10 PCs explain {sum(pca.explained_variance_ratio_[:10]):.1%} of variance")
    print(f"Top 50 PCs explain {sum(pca.explained_variance_ratio_[:50]):.1%} of variance")

    print("\n--- Centroid Distances (edit effect embedding) ---")
    for pair, dist in sorted(centroid_distances.items(), key=lambda x: x[1]):
        print(f"  {pair}: {dist:.4f}")

    print("\n--- Sharma Failure Analysis ---")
    print(f"  Sharma <-> Levanon: {sharma_to_levanon:.4f}")
    print(f"  Sharma <-> Negatives: {sharma_to_neg:.4f}")
    print(f"  Levanon <-> Negatives: {levanon_to_neg:.4f}")
    if sharma_to_neg < levanon_to_neg:
        print("  *** Sharma is CLOSER to negatives than Levanon is! ***")
        print("  This explains why Levanon-trained models classify Sharma as negative.")

    print("\n--- TC-Motif Distribution ---")
    for ds, info in motif_results.items():
        print(f"  {ds}: {info['tc_motif_fraction']:.1%} TC-motif ({info['n_sites']} sites)")

    print("\n--- Structure Delta by Dataset ---")
    print(f"  {'Dataset':<15} {'Delta MFE':>10} {'Delta Pair':>12} {'N sites':>8}")
    for ds, info in struct_results.items():
        print(f"  {ds:<15} {info['mean_delta_mfe']:>10.4f} {info['mean_delta_pairing']:>12.4f} {info['n_with_structure']:>8}")

    print("=" * 80)

    logger.info("Results saved to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
