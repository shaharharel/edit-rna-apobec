#!/usr/bin/env python
"""Binary prediction score → editing rate correlation analysis.

Tests whether the EditRNA-A3A binary classifier's P(edited) score
correlates with the actual editing rate for positive sites.
This demonstrates that the model implicitly learns rate information
even when trained only on binary labels.

Usage:
    python experiments/apobec3a/exp_binary_rate_correlation.py
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "apobec3a"))

from train_baselines import (
    BaselineConfig,
    EmbeddingDataset,
    embedding_collate_fn,
    load_data,
)

logger = logging.getLogger(__name__)

EMB_DIR = PROJECT_ROOT / "data" / "processed" / "embeddings"
SPLITS_CSV = PROJECT_ROOT / "data" / "processed" / "splits_expanded.csv"
EDITRNA_CKPT = (
    PROJECT_ROOT / "experiments" / "apobec3a" / "outputs" / "baselines" / "editrna" / "best_model.pt"
)
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "apobec3a" / "outputs" / "binary_rate_correlation"

DATASET_LABELS = {
    "advisor_c2t": "Levanon",
    "asaoka_2019": "Asaoka",
    "sharma_2015": "Sharma",
    "alqassim_2021": "Alqassim",
    "baysal_2016": "Baysal",
}


def load_editrna_model(config):
    """Load trained EditRNA-A3A model with cached encoder.

    Falls back to SubtractionMLP if token embeddings are not available.
    """
    emb_dir = Path(config.embeddings_dir)
    tokens_path = emb_dir / "rnafm_tokens.pt"

    if tokens_path.exists() and EDITRNA_CKPT.exists():
        from models.editrna_a3a import EditRNA_A3A, EditRNAConfig
        from models.encoders import CachedRNAEncoder

        tokens_cache = torch.load(tokens_path, weights_only=False)
        pooled_cache = torch.load(emb_dir / "rnafm_pooled.pt", weights_only=False)
        tokens_edited = torch.load(emb_dir / "rnafm_tokens_edited.pt", weights_only=False)
        pooled_edited = torch.load(emb_dir / "rnafm_pooled_edited.pt", weights_only=False)

        cached_encoder = CachedRNAEncoder(
            tokens_cache=tokens_cache,
            pooled_cache=pooled_cache,
            tokens_edited_cache=tokens_edited,
            pooled_edited_cache=pooled_edited,
            d_model=config.d_model,
        )
        model_config = EditRNAConfig(
            primary_encoder="cached",
            d_model=config.d_model,
        )
        model = EditRNA_A3A(config=model_config, primary_encoder=cached_encoder)

        ckpt = torch.load(EDITRNA_CKPT, weights_only=False, map_location="cpu")
        model.load_state_dict(ckpt, strict=False)
        model.eval()
        logger.info("Loaded EditRNA-A3A checkpoint from %s", EDITRNA_CKPT)
        return model, "editrna"
    else:
        # Fall back to SubtractionMLP (best pooled-only model)
        logger.warning("Token embeddings not found — using SubtractionMLP instead of EditRNA-A3A")
        from models.baselines.subtraction_mlp import SubtractionMLPBaseline
        sub_ckpt = PROJECT_ROOT / "experiments" / "apobec3a" / "outputs" / "baselines" / "subtraction_mlp" / "best_model.pt"
        model = SubtractionMLPBaseline()
        if sub_ckpt.exists():
            ckpt = torch.load(sub_ckpt, weights_only=False, map_location="cpu")
            model.load_state_dict(ckpt)
            logger.info("Loaded SubtractionMLP checkpoint from %s", sub_ckpt)
        else:
            logger.warning("No SubtractionMLP checkpoint found, using random weights")
        model.eval()
        return model, "subtraction_mlp"


def create_editrna_batch(batch):
    """Convert embedding batch to EditRNA-A3A forward pass format."""
    B = batch["pooled_orig"].shape[0]
    seq_len = 201
    return {
        "sequences": ["N" * seq_len] * B,
        "edit_pos": torch.full((B,), 100, dtype=torch.long),
        "flanking_context": torch.zeros(B, dtype=torch.long),
        "concordance_features": torch.zeros(B, 5),
        "structure_delta": batch["structure_delta"],
        "site_ids": batch["site_ids"],
    }


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data using same pipeline as other experiments
    config = BaselineConfig()
    pooled_orig, pooled_edited, _, _, structure_delta, splits_df = load_data(config)

    # Load model
    model, model_type = load_editrna_model(config)

    # Get test split
    available_ids = set(pooled_orig.keys()) & set(pooled_edited.keys())
    test_df = splits_df[
        (splits_df["site_id"].isin(available_ids)) &
        (splits_df["split"] == "test")
    ].copy()
    test_ids = test_df["site_id"].tolist()
    test_labels = test_df["label"].values

    logger.info("Test set: %d sites (model: %s)", len(test_ids), model_type)

    # Create test dataloader
    test_dataset = EmbeddingDataset(
        test_ids, test_labels,
        pooled_orig, pooled_edited,
        structure_delta=structure_delta,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=64, shuffle=False,
        num_workers=0, collate_fn=embedding_collate_fn,
    )

    # Run inference on all test samples
    all_scores = []
    all_ids = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            if model_type == "editrna":
                editrna_batch = create_editrna_batch(batch)
                output = model(editrna_batch)
                logits = output["predictions"]["binary_logit"].squeeze(-1).cpu().numpy()
            else:
                output = model(batch)
                logits = output["binary_logit"].squeeze(-1).cpu().numpy()
            probs = 1.0 / (1.0 + np.exp(-logits))
            all_scores.append(probs)
            all_ids.extend(batch["site_ids"])
            all_labels.append(batch["labels"].cpu().numpy())

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)
    site_ids = all_ids

    logger.info("Inference complete: %d sites (%d positive, %d negative)",
                len(scores), (labels == 1).sum(), (labels == 0).sum())

    # Match with rates for positive sites
    rate_map = dict(zip(test_df["site_id"], pd.to_numeric(test_df["editing_rate"], errors="coerce")))
    ds_map = dict(zip(test_df["site_id"], test_df["dataset_source"]))

    pos_mask = labels == 1
    pos_scores = []
    pos_rates = []
    pos_ds = []
    pos_ids = []

    for i, sid in enumerate(site_ids):
        if not pos_mask[i]:
            continue
        rate = rate_map.get(sid)
        if rate is None or np.isnan(rate) or rate <= 0:
            continue
        pos_scores.append(scores[i])
        pos_rates.append(rate)
        pos_ds.append(ds_map.get(sid, "unknown"))
        pos_ids.append(sid)

    pos_scores = np.array(pos_scores)
    pos_rates = np.array(pos_rates)
    pos_ds = np.array(pos_ds)

    logger.info("Positive sites with rates: %d", len(pos_scores))

    # Compute correlations
    from scipy.stats import pearsonr, spearmanr

    pr, pp = pearsonr(pos_scores, pos_rates)
    sr, sp = spearmanr(pos_scores, pos_rates)
    logger.info("Overall: Pearson=%.3f (p=%.2e), Spearman=%.3f (p=%.2e)",
                pr, pp, sr, sp)

    # Log-rate correlation
    log_rates = np.log2(np.clip(pos_rates, 0.01, None))
    pr_log, pp_log = pearsonr(pos_scores, log_rates)
    sr_log, sp_log = spearmanr(pos_scores, log_rates)
    logger.info("Log2-rate: Pearson=%.3f (p=%.2e), Spearman=%.3f (p=%.2e)",
                pr_log, pp_log, sr_log, sp_log)

    results = {
        "n_sites": int(len(pos_scores)),
        "overall": {
            "pearson_r": float(pr), "pearson_p": float(pp),
            "spearman_r": float(sr), "spearman_p": float(sp),
        },
        "log2_rate": {
            "pearson_r": float(pr_log), "pearson_p": float(pp_log),
            "spearman_r": float(sr_log), "spearman_p": float(sp_log),
        },
        "per_dataset": {},
    }

    # Per-dataset correlation
    logger.info("\nPer-dataset binary score → rate correlation:")
    for ds_name, ds_label in DATASET_LABELS.items():
        mask = pos_ds == ds_name
        if mask.sum() < 5:
            continue

        ds_scores = pos_scores[mask]
        ds_rates = pos_rates[mask]
        ds_log_rates = log_rates[mask]

        ds_pr, ds_pp = pearsonr(ds_scores, ds_rates)
        ds_sr, ds_sp = spearmanr(ds_scores, ds_rates)
        ds_pr_log, _ = pearsonr(ds_scores, ds_log_rates)
        ds_sr_log, _ = spearmanr(ds_scores, ds_log_rates)

        results["per_dataset"][ds_label] = {
            "n": int(mask.sum()),
            "pearson_r": float(ds_pr), "pearson_p": float(ds_pp),
            "spearman_r": float(ds_sr), "spearman_p": float(ds_sp),
            "log2_pearson_r": float(ds_pr_log),
            "log2_spearman_r": float(ds_sr_log),
            "mean_score": float(ds_scores.mean()),
            "mean_rate": float(ds_rates.mean()),
        }
        logger.info("  %s (n=%d): Pearson=%.3f, Spearman=%.3f, mean_score=%.3f, mean_rate=%.2f%%",
                    ds_label, mask.sum(), ds_pr, ds_sr,
                    ds_scores.mean(), ds_rates.mean())

    # Score bins analysis
    logger.info("\nScore bins analysis:")
    bins = [(0.0, 0.5, "Low (0-0.5)"),
            (0.5, 0.7, "Medium (0.5-0.7)"),
            (0.7, 0.9, "High (0.7-0.9)"),
            (0.9, 1.01, "Very High (0.9-1.0)")]

    bin_results = {}
    for lo, hi, label in bins:
        mask = (pos_scores >= lo) & (pos_scores < hi)
        if mask.sum() < 3:
            bin_results[label] = {"n": int(mask.sum()), "mean_rate": None}
            continue
        bin_rates = pos_rates[mask]
        bin_results[label] = {
            "n": int(mask.sum()),
            "mean_rate": float(bin_rates.mean()),
            "median_rate": float(np.median(bin_rates)),
            "std_rate": float(bin_rates.std()),
        }
        logger.info("  %s: n=%d, mean_rate=%.3f%%, median_rate=%.3f%%",
                    label, mask.sum(), bin_rates.mean(), np.median(bin_rates))

    results["score_bins"] = bin_results

    # Save results
    with open(OUTPUT_DIR / "binary_rate_correlation.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save per-site predictions
    pred_df = pd.DataFrame({
        "site_id": pos_ids,
        "binary_score": pos_scores,
        "editing_rate": pos_rates,
        "log2_rate": log_rates,
        "dataset": pos_ds,
    })
    pred_df.to_csv(OUTPUT_DIR / "binary_score_vs_rate.csv", index=False)

    logger.info("\nResults saved to %s", OUTPUT_DIR)

    # Print summary
    print("\n" + "=" * 70)
    print("BINARY SCORE → EDITING RATE CORRELATION")
    print("=" * 70)
    print(f"\nTotal positive test sites with rates: {len(pos_scores)}")
    print(f"\nOverall correlation (binary P(edited) vs rate):")
    print(f"  Pearson  r = {pr:.3f} (p = {pp:.2e})")
    print(f"  Spearman ρ = {sr:.3f} (p = {sp:.2e})")
    print(f"\nLog2-rate correlation:")
    print(f"  Pearson  r = {pr_log:.3f} (p = {pp_log:.2e})")
    print(f"  Spearman ρ = {sr_log:.3f} (p = {sp_log:.2e})")
    print(f"\nPer-dataset:")
    for ds_label, info in results["per_dataset"].items():
        print(f"  {ds_label} (n={info['n']}): Pearson={info['pearson_r']:.3f}, "
              f"Spearman={info['spearman_r']:.3f}, mean_score={info['mean_score']:.3f}")
    print(f"\nScore bin → mean rate:")
    for label, info in bin_results.items():
        if info["mean_rate"] is not None:
            print(f"  {label}: n={info['n']}, mean_rate={info['mean_rate']:.3f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
