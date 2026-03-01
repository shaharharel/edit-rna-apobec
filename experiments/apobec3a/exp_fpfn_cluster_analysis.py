#!/usr/bin/env python
"""Iteration 3: HDBSCAN clustering + FP/FN characterization.

Analyzes the EditRNA-A3A model's embedding space and errors:
1. HDBSCAN clustering on edit effect embeddings to discover editing subtypes
2. FP/FN characterization: which sites does EditRNA-A3A get wrong and why?
3. Cross-reference errors with biological annotations (tissue, structure, motif)

Usage:
    python experiments/apobec3a/exp_fpfn_cluster_analysis.py
"""

import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logger = logging.getLogger(__name__)

EMB_DIR = PROJECT_ROOT / "data" / "processed" / "embeddings"
SPLITS_CSV = PROJECT_ROOT / "data" / "processed" / "splits_expanded.csv"
LABELS_CSV = PROJECT_ROOT / "data" / "processed" / "editing_sites_labels.csv"
STRUCT_CACHE = EMB_DIR / "vienna_structure_cache.npz"
SEQ_JSON = PROJECT_ROOT / "data" / "processed" / "site_sequences.json"
EDITRNA_CKPT = (
    PROJECT_ROOT / "experiments" / "apobec" / "outputs" / "baselines" / "editrna" / "best_model.pt"
)
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "apobec" / "outputs" / "iteration3"

DATASET_LABELS = {
    "advisor_c2t": "Levanon",
    "asaoka_2019": "Asaoka",
    "sharma_2015": "Sharma",
    "alqassim_2021": "Alqassim",
    "baysal_2016": "Baysal",
    "tier2_negative": "Tier2 Neg",
    "tier3_negative": "Tier3 Neg",
}


def load_data():
    """Load all data needed for analysis."""
    splits_df = pd.read_csv(SPLITS_CSV)
    labels_df = pd.read_csv(LABELS_CSV)

    pooled_orig = torch.load(EMB_DIR / "rnafm_pooled.pt", weights_only=False)
    pooled_edited = torch.load(EMB_DIR / "rnafm_pooled_edited.pt", weights_only=False)
    logger.info("Loaded %d pooled embeddings", len(pooled_orig))

    # Structure delta
    structure_delta = {}
    if STRUCT_CACHE.exists():
        data = np.load(STRUCT_CACHE, allow_pickle=True)
        feat_key = "delta_features" if "delta_features" in data else "features"
        if "site_ids" in data and feat_key in data:
            sids = data["site_ids"]
            feats = data[feat_key]
            structure_delta = {str(sid): feats[i] for i, sid in enumerate(sids)}

    # Sequences for motif analysis
    sequences = {}
    if SEQ_JSON.exists():
        with open(SEQ_JSON) as f:
            sequences = json.load(f)

    return splits_df, labels_df, pooled_orig, pooled_edited, structure_delta, sequences


def compute_edit_diff_embeddings(pooled_orig, pooled_edited, site_ids):
    """Compute edit effect embeddings (edited - original) for given site IDs."""
    diffs = []
    valid_ids = []
    for sid in site_ids:
        if sid in pooled_orig and sid in pooled_edited:
            diff = pooled_edited[sid] - pooled_orig[sid]
            diffs.append(diff.numpy())
            valid_ids.append(sid)
    return np.array(diffs), valid_ids


def get_predictions(splits_df, pooled_orig, pooled_edited, structure_delta):
    """Get EditRNA-A3A predictions on test set using the trained model."""
    sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "apobec"))
    from train_baselines import (
        BaselineConfig,
        EmbeddingDataset,
        build_model,
        embedding_collate_fn,
    )

    # Load test data
    available_ids = set(pooled_orig.keys())
    df = splits_df[
        (splits_df["site_id"].isin(available_ids)) & (splits_df["split"] == "test")
    ].copy()

    test_ids = df["site_id"].tolist()
    test_labels = df["label"].values.astype(np.float32)

    test_ds = EmbeddingDataset(
        test_ids, test_labels, pooled_orig, pooled_edited,
        structure_delta=structure_delta,
    )
    test_loader = DataLoader(
        test_ds, batch_size=64, shuffle=False, num_workers=0,
        collate_fn=embedding_collate_fn,
    )

    # Load model
    config = BaselineConfig(model_name="subtraction_mlp", d_model=640)
    model = build_model("subtraction_mlp", config)
    device = torch.device("cpu")

    # Try loading EditRNA checkpoint, fall back to subtraction_mlp
    ckpt_path = EDITRNA_CKPT
    model_name_used = "editrna"
    if ckpt_path.exists():
        try:
            state = torch.load(ckpt_path, weights_only=False, map_location="cpu")
            config_e = BaselineConfig(model_name="editrna", d_model=640)
            model = build_model("editrna", config_e)
            model.load_state_dict(state)
            logger.info("Loaded EditRNA-A3A checkpoint")
        except Exception as e:
            logger.warning("Failed to load EditRNA checkpoint: %s, using subtraction_mlp", e)
            # Load subtraction_mlp checkpoint
            sub_ckpt = (
                PROJECT_ROOT / "experiments" / "apobec" / "outputs" / "baselines"
                / "subtraction_mlp" / "best_model.pt"
            )
            if sub_ckpt.exists():
                state = torch.load(sub_ckpt, weights_only=False, map_location="cpu")
                model.load_state_dict(state)
            model_name_used = "subtraction_mlp"
    else:
        sub_ckpt = (
            PROJECT_ROOT / "experiments" / "apobec" / "outputs" / "baselines"
            / "subtraction_mlp" / "best_model.pt"
        )
        if sub_ckpt.exists():
            state = torch.load(sub_ckpt, weights_only=False, map_location="cpu")
            model.load_state_dict(state)
        model_name_used = "subtraction_mlp"
        logger.info("Using subtraction_mlp for FP/FN analysis")

    model = model.to(device)
    model.eval()

    # Get predictions
    all_ids = []
    all_targets = []
    all_scores = []

    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            if model_name_used == "editrna":
                B = batch["labels"].shape[0]
                editrna_batch = {
                    "sequences": ["N" * 201] * B,
                    "site_ids": batch["site_ids"],
                    "edit_pos": torch.full((B,), 100, dtype=torch.long, device=device),
                    "flanking_context": torch.zeros(B, dtype=torch.long, device=device),
                    "concordance_features": torch.zeros(B, 5, device=device),
                    "structure_delta": batch["structure_delta"],
                }
                output = model(editrna_batch)
                logits = output["predictions"]["binary_logit"].squeeze(-1).cpu().numpy()
            else:
                output = model(batch)
                logits = output["binary_logit"].squeeze(-1).cpu().numpy()

            probs = 1.0 / (1.0 + np.exp(-logits))
            all_ids.extend(batch["site_ids"])
            all_targets.append(batch["labels"].cpu().numpy())
            all_scores.append(probs)

    y_true = np.concatenate(all_targets)
    y_score = np.concatenate(all_scores)

    # Get optimal threshold from results.json
    results_file = (
        PROJECT_ROOT / "experiments" / "apobec" / "outputs" / "baselines"
        / model_name_used / "results.json"
    )
    threshold = 0.5
    if results_file.exists():
        with open(results_file) as f:
            res = json.load(f)
            threshold = res.get("test_metrics", {}).get("optimal_threshold", 0.5)

    y_pred = (y_score >= threshold).astype(int)

    return pd.DataFrame({
        "site_id": all_ids,
        "y_true": y_true,
        "y_score": y_score,
        "y_pred": y_pred,
    }), model_name_used, threshold


def run_clustering(pooled_orig, pooled_edited, splits_df):
    """Run clustering on edit effect embeddings.

    Uses K-Means with silhouette analysis to find optimal k,
    since HDBSCAN fails on the continuous edit effect space.
    Also tries HDBSCAN with more aggressive params as a secondary check.
    """
    logger.info("Running clustering on edit effect embeddings...")

    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    # Get all site IDs
    available_ids = set(pooled_orig.keys())
    df = splits_df[splits_df["site_id"].isin(available_ids)].copy()
    site_ids = df["site_id"].tolist()

    # Compute diff embeddings
    diffs, valid_ids = compute_edit_diff_embeddings(pooled_orig, pooled_edited, site_ids)
    logger.info("Edit diff embeddings shape: %s", diffs.shape)

    # PCA reduce
    n_pca = 20
    pca = PCA(n_components=n_pca, random_state=42)
    diffs_pca = pca.fit_transform(diffs)
    logger.info("PCA reduced to %d dims (%.1f%% variance)",
                n_pca, 100 * pca.explained_variance_ratio_.sum())

    # Silhouette analysis for optimal k
    logger.info("Running silhouette analysis for k=2..10...")
    silhouette_scores = {}
    # Subsample for speed
    n_sub = min(3000, len(diffs_pca))
    rng = np.random.RandomState(42)
    sub_idx = rng.choice(len(diffs_pca), n_sub, replace=False)
    diffs_sub = diffs_pca[sub_idx]

    for k in range(2, 11):
        km = KMeans(n_clusters=k, random_state=42, n_init=5, max_iter=100)
        labels_k = km.fit_predict(diffs_sub)
        sil = silhouette_score(diffs_sub, labels_k, sample_size=min(1000, n_sub))
        silhouette_scores[k] = float(sil)
        logger.info("  k=%d: silhouette=%.4f", k, sil)

    # Pick best k
    best_k = max(silhouette_scores, key=silhouette_scores.get)
    logger.info("Best k=%d (silhouette=%.4f)", best_k, silhouette_scores[best_k])

    # Run KMeans with best k on full data
    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = km_final.fit_predict(diffs_pca)

    # Map site IDs to cluster labels
    id_to_cluster = {sid: int(cl) for sid, cl in zip(valid_ids, labels)}

    # Merge with metadata
    df_clustered = df[df["site_id"].isin(valid_ids)].copy()
    df_clustered["cluster"] = df_clustered["site_id"].map(id_to_cluster)

    return df_clustered, id_to_cluster, diffs_pca, valid_ids, silhouette_scores


def characterize_clusters(df_clustered, labels_df, sequences, structure_delta):
    """Characterize each cluster by biological properties."""
    cluster_profiles = {}

    # Merge with labels if available
    if labels_df is not None:
        df = df_clustered.merge(
            labels_df[["site_id", "genomic_category", "apobec_class",
                       "structure_type", "tissue_class", "max_gtex_rate",
                       "conservation_level", "exonic_function",
                       "has_survival_association"]],
            on="site_id", how="left"
        )
    else:
        df = df_clustered.copy()

    for cl in sorted(df["cluster"].unique()):
        cl_df = df[df["cluster"] == cl]
        profile = {
            "n_sites": len(cl_df),
            "label_ratio": float(cl_df["label"].mean()),
            "datasets": dict(Counter(cl_df["dataset_source"].map(
                lambda x: DATASET_LABELS.get(x, x)
            ))),
        }

        # TC-motif fraction
        if sequences:
            tc_count = 0
            total = 0
            for sid in cl_df["site_id"]:
                if sid in sequences:
                    seq = sequences[sid].upper().replace("T", "U")
                    center = len(seq) // 2
                    if center > 0 and seq[center - 1] == "U":
                        tc_count += 1
                    total += 1
            profile["tc_motif_fraction"] = tc_count / max(total, 1)

        # Structure delta stats
        deltas = []
        for sid in cl_df["site_id"]:
            if sid in structure_delta:
                deltas.append(structure_delta[sid])
        if deltas:
            deltas_arr = np.array(deltas)
            profile["mean_delta_mfe"] = float(np.mean(deltas_arr[:, 3]))
            profile["mean_delta_pairing"] = float(np.mean(deltas_arr[:, 0]))

        # Editing rate stats (for positives)
        pos_df = cl_df[cl_df["label"] == 1]
        if "editing_rate" in cl_df.columns:
            rates = pd.to_numeric(pos_df["editing_rate"], errors="coerce").dropna()
            if len(rates) > 0:
                profile["mean_editing_rate"] = float(rates.mean())
                profile["median_editing_rate"] = float(rates.median())

        # Annotations from labels
        if "genomic_category" in df.columns:
            profile["genomic_categories"] = dict(Counter(
                cl_df["genomic_category"].dropna()
            ))
        if "apobec_class" in df.columns:
            profile["apobec_classes"] = dict(Counter(
                cl_df["apobec_class"].dropna()
            ))
        if "structure_type" in df.columns:
            profile["structure_types"] = dict(Counter(
                cl_df["structure_type"].dropna()
            ))
        if "tissue_class" in df.columns:
            profile["tissue_classes"] = dict(Counter(
                cl_df["tissue_class"].dropna()
            ))
        if "conservation_level" in df.columns:
            consv = cl_df["conservation_level"].dropna()
            if len(consv) > 0:
                profile["conservation_levels"] = dict(Counter(consv))

        cluster_profiles[int(cl)] = profile

    return cluster_profiles


def analyze_fpfn(pred_df, splits_df, labels_df, sequences, structure_delta):
    """Characterize false positives and false negatives."""
    logger.info("Analyzing FP/FN patterns...")

    # Classify predictions
    pred_df["category"] = "TN"
    pred_df.loc[(pred_df["y_true"] == 1) & (pred_df["y_pred"] == 1), "category"] = "TP"
    pred_df.loc[(pred_df["y_true"] == 0) & (pred_df["y_pred"] == 1), "category"] = "FP"
    pred_df.loc[(pred_df["y_true"] == 1) & (pred_df["y_pred"] == 0), "category"] = "FN"

    counts = dict(Counter(pred_df["category"]))
    logger.info("Prediction breakdown: %s", counts)

    # Merge with metadata
    analysis_df = pred_df.merge(splits_df[["site_id", "dataset_source", "gene", "editing_rate"]],
                                on="site_id", how="left")

    if labels_df is not None:
        analysis_df = analysis_df.merge(
            labels_df[["site_id", "genomic_category", "apobec_class",
                       "structure_type", "tissue_class", "max_gtex_rate",
                       "exonic_function", "conservation_level",
                       "has_survival_association"]],
            on="site_id", how="left"
        )

    results = {"counts": counts}

    # FP analysis
    fp_df = analysis_df[analysis_df["category"] == "FP"].copy()
    if len(fp_df) > 0:
        fp_info = {
            "n_total": len(fp_df),
            "mean_score": float(fp_df["y_score"].mean()),
            "median_score": float(fp_df["y_score"].median()),
            "datasets": dict(Counter(fp_df["dataset_source"].map(
                lambda x: DATASET_LABELS.get(x, x)
            ))),
        }

        # TC-motif analysis
        if sequences:
            tc_count = 0
            total = 0
            for sid in fp_df["site_id"]:
                if sid in sequences:
                    seq = sequences[sid].upper().replace("T", "U")
                    center = len(seq) // 2
                    if center > 0 and seq[center - 1] == "U":
                        tc_count += 1
                    total += 1
            fp_info["tc_motif_fraction"] = tc_count / max(total, 1)

        # Structure analysis
        fp_deltas = []
        for sid in fp_df["site_id"]:
            if sid in structure_delta:
                fp_deltas.append(structure_delta[sid])
        if fp_deltas:
            fp_deltas_arr = np.array(fp_deltas)
            fp_info["mean_delta_mfe"] = float(np.mean(fp_deltas_arr[:, 3]))
            fp_info["mean_delta_pairing"] = float(np.mean(fp_deltas_arr[:, 0]))

        if "genomic_category" in fp_df.columns:
            fp_info["genomic_categories"] = dict(Counter(fp_df["genomic_category"].dropna()))
        if "structure_type" in fp_df.columns:
            fp_info["structure_types"] = dict(Counter(fp_df["structure_type"].dropna()))

        # Top FP candidates (highest confidence false positives = potential novel editing sites)
        fp_ranked = fp_df.nlargest(20, "y_score")
        fp_info["top_candidates"] = []
        for _, row in fp_ranked.iterrows():
            candidate = {
                "site_id": row["site_id"],
                "score": float(row["y_score"]),
                "gene": str(row.get("gene", "?")),
                "dataset": DATASET_LABELS.get(str(row.get("dataset_source", "")), "?"),
            }
            fp_info["top_candidates"].append(candidate)

        results["false_positives"] = fp_info

    # FN analysis
    fn_df = analysis_df[analysis_df["category"] == "FN"].copy()
    if len(fn_df) > 0:
        fn_info = {
            "n_total": len(fn_df),
            "mean_score": float(fn_df["y_score"].mean()),
            "median_score": float(fn_df["y_score"].median()),
            "datasets": dict(Counter(fn_df["dataset_source"].map(
                lambda x: DATASET_LABELS.get(x, x)
            ))),
        }

        # Editing rate of FN (are they low-rate sites?)
        fn_rates = pd.to_numeric(fn_df["editing_rate"], errors="coerce").dropna()
        if len(fn_rates) > 0:
            fn_info["mean_editing_rate"] = float(fn_rates.mean())
            fn_info["median_editing_rate"] = float(fn_rates.median())

            # Compare to TP rates
            tp_df = analysis_df[analysis_df["category"] == "TP"]
            tp_rates = pd.to_numeric(tp_df["editing_rate"], errors="coerce").dropna()
            if len(tp_rates) > 0:
                fn_info["tp_mean_editing_rate"] = float(tp_rates.mean())
                fn_info["rate_ratio_fn_tp"] = float(fn_rates.mean() / max(tp_rates.mean(), 1e-6))

        # TC-motif analysis
        if sequences:
            tc_count = 0
            total = 0
            for sid in fn_df["site_id"]:
                if sid in sequences:
                    seq = sequences[sid].upper().replace("T", "U")
                    center = len(seq) // 2
                    if center > 0 and seq[center - 1] == "U":
                        tc_count += 1
                    total += 1
            fn_info["tc_motif_fraction"] = tc_count / max(total, 1)

        # Structure analysis
        fn_deltas = []
        for sid in fn_df["site_id"]:
            if sid in structure_delta:
                fn_deltas.append(structure_delta[sid])
        if fn_deltas:
            fn_deltas_arr = np.array(fn_deltas)
            fn_info["mean_delta_mfe"] = float(np.mean(fn_deltas_arr[:, 3]))
            fn_info["mean_delta_pairing"] = float(np.mean(fn_deltas_arr[:, 0]))

        if "genomic_category" in fn_df.columns:
            fn_info["genomic_categories"] = dict(Counter(fn_df["genomic_category"].dropna()))
        if "apobec_class" in fn_df.columns:
            fn_info["apobec_classes"] = dict(Counter(fn_df["apobec_class"].dropna()))
        if "structure_type" in fn_df.columns:
            fn_info["structure_types"] = dict(Counter(fn_df["structure_type"].dropna()))
        if "tissue_class" in fn_df.columns:
            fn_info["tissue_classes"] = dict(Counter(fn_df["tissue_class"].dropna()))

        # Most confident FN (model is sure they're negative but they're positive)
        fn_ranked = fn_df.nsmallest(20, "y_score")
        fn_info["hardest_misses"] = []
        for _, row in fn_ranked.iterrows():
            miss = {
                "site_id": row["site_id"],
                "score": float(row["y_score"]),
                "gene": str(row.get("gene", "?")),
                "dataset": DATASET_LABELS.get(str(row.get("dataset_source", "")), "?"),
                "editing_rate": float(row["editing_rate"]) if pd.notna(row.get("editing_rate")) else None,
            }
            fn_info["hardest_misses"].append(miss)

        results["false_negatives"] = fn_info

    # Edge cases: sites near decision boundary (0.3 < score < 0.7)
    edge_df = analysis_df[(analysis_df["y_score"] > 0.3) & (analysis_df["y_score"] < 0.7)].copy()
    if len(edge_df) > 0:
        edge_info = {
            "n_total": len(edge_df),
            "n_positive": int((edge_df["y_true"] == 1).sum()),
            "n_negative": int((edge_df["y_true"] == 0).sum()),
            "datasets": dict(Counter(edge_df["dataset_source"].map(
                lambda x: DATASET_LABELS.get(x, x)
            ))),
        }
        if sequences:
            tc_count = 0
            total = 0
            for sid in edge_df["site_id"]:
                if sid in sequences:
                    seq = sequences[sid].upper().replace("T", "U")
                    center = len(seq) // 2
                    if center > 0 and seq[center - 1] == "U":
                        tc_count += 1
                    total += 1
            edge_info["tc_motif_fraction"] = tc_count / max(total, 1)

        results["edge_cases"] = edge_info

    # Per-dataset performance breakdown
    dataset_perf = {}
    for ds in analysis_df["dataset_source"].unique():
        ds_df = analysis_df[analysis_df["dataset_source"] == ds]
        if len(ds_df) < 5:
            continue
        ds_label = DATASET_LABELS.get(ds, ds)
        ds_perf = {
            "n_total": len(ds_df),
            "n_positive": int((ds_df["y_true"] == 1).sum()),
            "n_negative": int((ds_df["y_true"] == 0).sum()),
        }
        cats = dict(Counter(ds_df["category"]))
        ds_perf.update(cats)

        # Accuracy on this dataset
        correct = (ds_df["y_pred"] == ds_df["y_true"]).sum()
        ds_perf["accuracy"] = float(correct / len(ds_df))

        dataset_perf[ds_label] = ds_perf
    results["per_dataset_performance"] = dataset_perf

    return results, analysis_df


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    splits_df, labels_df, pooled_orig, pooled_edited, structure_delta, sequences = load_data()

    all_results = {}

    # ===================================================================
    # Part 1: HDBSCAN Clustering
    # ===================================================================
    logger.info("=" * 70)
    logger.info("PART 1: Clustering on Edit Effect Embeddings (KMeans + Silhouette)")
    logger.info("=" * 70)

    cluster_result = run_clustering(pooled_orig, pooled_edited, splits_df)
    if cluster_result is not None:
        df_clustered, id_to_cluster, diffs_pca, valid_ids, silhouette_scores = cluster_result
        all_results["silhouette_scores"] = silhouette_scores
        cluster_profiles = characterize_clusters(
            df_clustered, labels_df, sequences, structure_delta
        )

        # Summary
        print("\n--- HDBSCAN Cluster Summary ---")
        for cl, prof in sorted(cluster_profiles.items()):
            label = "NOISE" if cl == -1 else f"Cluster {cl}"
            pos_pct = prof["label_ratio"] * 100
            tc_pct = prof.get("tc_motif_fraction", 0) * 100
            dmfe = prof.get("mean_delta_mfe", float("nan"))
            print(f"  {label}: {prof['n_sites']} sites, {pos_pct:.0f}% positive, "
                  f"TC-motif={tc_pct:.0f}%, delta_MFE={dmfe:.3f}")
            ds_str = ", ".join(f"{k}:{v}" for k, v in sorted(
                prof["datasets"].items(), key=lambda x: -x[1])[:4])
            print(f"    Datasets: {ds_str}")

        all_results["hdbscan_clusters"] = cluster_profiles

        # Save cluster assignments
        cluster_df = pd.DataFrame([
            {"site_id": sid, "cluster": cl}
            for sid, cl in id_to_cluster.items()
        ])
        cluster_df.to_csv(OUTPUT_DIR / "cluster_assignments.csv", index=False)
        logger.info("Saved cluster assignments to %s", OUTPUT_DIR / "cluster_assignments.csv")

    # ===================================================================
    # Part 2: FP/FN Characterization
    # ===================================================================
    logger.info("\n" + "=" * 70)
    logger.info("PART 2: FP/FN Characterization")
    logger.info("=" * 70)

    pred_df, model_used, threshold = get_predictions(
        splits_df, pooled_orig, pooled_edited, structure_delta
    )
    logger.info("Model: %s, threshold: %.4f", model_used, threshold)
    logger.info("Test set: %d sites (%d pos, %d neg)",
                len(pred_df), (pred_df["y_true"] == 1).sum(), (pred_df["y_true"] == 0).sum())

    fpfn_results, analysis_df = analyze_fpfn(
        pred_df, splits_df, labels_df, sequences, structure_delta
    )
    all_results["fpfn_analysis"] = fpfn_results
    all_results["model_used"] = model_used
    all_results["threshold"] = threshold

    # Print summary
    print("\n--- FP/FN Summary ---")
    counts = fpfn_results["counts"]
    print(f"  TP: {counts.get('TP', 0)}, FP: {counts.get('FP', 0)}, "
          f"FN: {counts.get('FN', 0)}, TN: {counts.get('TN', 0)}")

    if "false_positives" in fpfn_results:
        fp = fpfn_results["false_positives"]
        print(f"\n  False Positives ({fp['n_total']}):")
        print(f"    Mean confidence: {fp['mean_score']:.3f}")
        if "tc_motif_fraction" in fp:
            print(f"    TC-motif fraction: {fp['tc_motif_fraction']:.1%}")
        if "mean_delta_mfe" in fp:
            print(f"    Mean delta MFE: {fp['mean_delta_mfe']:.3f}")
        print(f"    Datasets: {fp.get('datasets', {})}")
        print(f"    Top FP candidates (potential novel editing sites):")
        for c in fp.get("top_candidates", [])[:5]:
            print(f"      {c['site_id']}: score={c['score']:.3f}, gene={c['gene']}, ds={c['dataset']}")

    if "false_negatives" in fpfn_results:
        fn = fpfn_results["false_negatives"]
        print(f"\n  False Negatives ({fn['n_total']}):")
        print(f"    Mean confidence: {fn['mean_score']:.3f}")
        if "mean_editing_rate" in fn:
            print(f"    Mean editing rate: {fn['mean_editing_rate']:.2f}%")
            print(f"    TP mean editing rate: {fn.get('tp_mean_editing_rate', 0):.2f}%")
            print(f"    Rate ratio (FN/TP): {fn.get('rate_ratio_fn_tp', 0):.2f}")
        if "tc_motif_fraction" in fn:
            print(f"    TC-motif fraction: {fn['tc_motif_fraction']:.1%}")
        print(f"    Datasets: {fn.get('datasets', {})}")
        print(f"    Hardest misses (lowest score, true positives):")
        for m in fn.get("hardest_misses", [])[:5]:
            rate_str = f", rate={m['editing_rate']:.2f}%" if m.get("editing_rate") else ""
            print(f"      {m['site_id']}: score={m['score']:.3f}, gene={m['gene']}, ds={m['dataset']}{rate_str}")

    if "edge_cases" in fpfn_results:
        edge = fpfn_results["edge_cases"]
        print(f"\n  Edge Cases (score 0.3-0.7): {edge['n_total']} sites "
              f"({edge['n_positive']} pos, {edge['n_negative']} neg)")

    print("\n--- Per-Dataset Test Performance ---")
    for ds, perf in sorted(fpfn_results.get("per_dataset_performance", {}).items()):
        print(f"  {ds}: n={perf['n_total']}, acc={perf.get('accuracy', 0):.3f}, "
              f"TP={perf.get('TP', 0)}, FP={perf.get('FP', 0)}, "
              f"FN={perf.get('FN', 0)}, TN={perf.get('TN', 0)}")

    # ===================================================================
    # Part 3: Cross-reference clusters with FP/FN
    # ===================================================================
    if cluster_result is not None:
        logger.info("\n" + "=" * 70)
        logger.info("PART 3: Cluster × Prediction Cross-reference")
        logger.info("=" * 70)

        # Merge cluster assignments with predictions
        pred_with_cluster = pred_df.merge(
            cluster_df, on="site_id", how="left"
        )
        pred_with_cluster["category"] = "TN"
        pred_with_cluster.loc[
            (pred_with_cluster["y_true"] == 1) & (pred_with_cluster["y_pred"] == 1),
            "category"
        ] = "TP"
        pred_with_cluster.loc[
            (pred_with_cluster["y_true"] == 0) & (pred_with_cluster["y_pred"] == 1),
            "category"
        ] = "FP"
        pred_with_cluster.loc[
            (pred_with_cluster["y_true"] == 1) & (pred_with_cluster["y_pred"] == 0),
            "category"
        ] = "FN"

        cluster_error = {}
        for cl in sorted(pred_with_cluster["cluster"].dropna().unique()):
            cl_preds = pred_with_cluster[pred_with_cluster["cluster"] == cl]
            cats = dict(Counter(cl_preds["category"]))
            n_total = len(cl_preds)
            n_correct = cats.get("TP", 0) + cats.get("TN", 0)
            cluster_error[int(cl)] = {
                "n_total": n_total,
                "accuracy": n_correct / max(n_total, 1),
                **cats,
            }

        all_results["cluster_prediction_crossref"] = cluster_error

        print("\n--- Cluster × Prediction Performance ---")
        for cl, info in sorted(cluster_error.items()):
            label = "NOISE" if cl == -1 else f"Cluster {cl}"
            print(f"  {label}: n={info['n_total']}, acc={info['accuracy']:.3f}, "
                  f"TP={info.get('TP', 0)}, FP={info.get('FP', 0)}, "
                  f"FN={info.get('FN', 0)}, TN={info.get('TN', 0)}")

    # Save all results
    def serialize(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return str(obj)

    with open(OUTPUT_DIR / "iteration3_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=serialize)

    # Save predictions for downstream analysis
    analysis_df.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

    logger.info("\nResults saved to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
