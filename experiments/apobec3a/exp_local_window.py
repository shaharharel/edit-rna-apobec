#!/usr/bin/env python
"""Window size + architecture ablation for EditRNA-A3A.

Tests multiple token window sizes and the pooled-only variant:
  1. Pooled-only (no token-level attention)
  2. Window=10 (21 tokens)
  3. Window=25 (51 tokens)
  4. Window=50 (101 tokens)
  5. Full (all ~201 tokens)

Evaluates on BOTH classification (binary) and rate prediction tasks.
Key question: does token-level attention help, and if so, how local?

Usage:
    python experiments/apobec3a/exp_local_window.py
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score, average_precision_score, mean_squared_error, r2_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.apobec_dataset import (
    APOBECDataConfig,
    APOBECDataset,
    APOBECSiteSample,
    N_TISSUES,
    apobec_collate_fn,
    get_flanking_context,
)
from models.editrna_a3a import EditRNA_A3A, EditRNAConfig
from models.encoders import CachedRNAEncoder

logger = logging.getLogger(__name__)

EMB_DIR = PROJECT_ROOT / "data" / "processed" / "embeddings"
SPLITS_CSV = PROJECT_ROOT / "data" / "processed" / "splits_expanded.csv"
SEQ_JSON = PROJECT_ROOT / "data" / "processed" / "site_sequences.json"
STRUCT_CACHE = EMB_DIR / "vienna_structure_cache.npz"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "apobec3a" / "outputs" / "local_window"

WINDOW_CONFIGS = [
    {"name": "Pooled-only", "local_window": 0, "pooled_only": True},
    {"name": "Window=10", "local_window": 10, "pooled_only": False},
    {"name": "Window=25", "local_window": 25, "pooled_only": False},
    {"name": "Window=50", "local_window": 50, "pooled_only": False},
    {"name": "Full (201)", "local_window": 0, "pooled_only": False},
]


def compute_log2_rate(editing_rate):
    if pd.isna(editing_rate) or editing_rate < 0:
        return float("nan")
    rate = float(editing_rate)
    if rate > 1.0:
        rate = rate / 100.0
    return np.log2(rate + 0.01)


def build_samples(df, sequences, structure_delta, include_negatives=False, window_size=100):
    """Build APOBECSiteSample list."""
    samples = []
    for _, row in df.iterrows():
        site_id = str(row["site_id"])
        label = float(row["label"])
        seq = sequences.get(site_id, "A" * (window_size * 2 + 1))
        edit_pos = min(window_size, len(seq) // 2)
        seq_list = list(seq)
        seq_list[edit_pos] = "C"
        seq = "".join(seq_list)
        flanking = get_flanking_context(seq, edit_pos)
        struct_d = structure_delta.get(site_id)
        if struct_d is not None:
            struct_d = np.array(struct_d, dtype=np.float32)

        rate = row.get("editing_rate", float("nan"))
        log2_rate = compute_log2_rate(rate) if label == 1 else float("nan")
        concordance = np.zeros(5, dtype=np.float32)

        sample = APOBECSiteSample(
            sequence=seq, edit_pos=edit_pos,
            is_edited=label,
            editing_rate_log2=log2_rate,
            apobec_class=-1, structure_type=-1, tissue_spec_class=-1,
            n_tissues_log2=float("nan"), exonic_function=-1,
            conservation=float("nan"), cancer_survival=float("nan"),
            tissue_rates=np.full(N_TISSUES, np.nan, dtype=np.float32),
            hek293_rate=float("nan"),
            flanking_context=flanking, concordance_features=concordance,
            structure_delta=struct_d, site_id=site_id,
            chrom=str(row.get("chr", "")),
            position=int(row.get("start", 0)),
            gene=str(row.get("gene", "")),
        )
        samples.append(sample)
    return samples


def batch_to_device(batch, device):
    result = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            result[k] = v.to(device)
        elif isinstance(v, dict):
            result[k] = {kk: vv.to(device) if isinstance(vv, torch.Tensor) else vv
                         for kk, vv in v.items()}
        else:
            result[k] = v
    return result


def train_eval_variant(config_dict, train_df, val_df, test_df,
                       sequences, structure_delta,
                       tokens_orig, pooled_orig, tokens_edited, pooled_edited,
                       epochs=60, lr=1e-4, patience=15, seed=42):
    """Train and evaluate an EditRNA-A3A variant with specific window config."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")

    data_config = APOBECDataConfig(window_size=100)
    train_ds = APOBECDataset(build_samples(train_df, sequences, structure_delta, include_negatives=True), data_config)
    val_ds = APOBECDataset(build_samples(val_df, sequences, structure_delta, include_negatives=True), data_config)
    test_ds = APOBECDataset(build_samples(test_df, sequences, structure_delta, include_negatives=True), data_config)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                              collate_fn=apobec_collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False,
                            collate_fn=apobec_collate_fn, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False,
                             collate_fn=apobec_collate_fn, num_workers=0)

    cached_encoder = CachedRNAEncoder(
        tokens_cache=tokens_orig, pooled_cache=pooled_orig,
        tokens_edited_cache=tokens_edited, pooled_edited_cache=pooled_edited,
        d_model=640,
    )
    config = EditRNAConfig(
        primary_encoder="cached", d_model=640, d_edit=256, d_fused=512,
        edit_n_heads=8, use_structure_delta=True,
        head_dropout=0.2, fusion_dropout=0.2,
        focal_gamma=2.0, focal_alpha_binary=0.75,
        local_window=config_dict["local_window"],
        pooled_only=config_dict["pooled_only"],
    )
    model = EditRNA_A3A(config=config, primary_encoder=cached_encoder).to(device)
    optimizer = AdamW(model.get_parameter_groups(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            batch = batch_to_device(batch, device)
            optimizer.zero_grad()
            output = model(batch)
            loss, _ = model.compute_loss(output, batch["targets"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        val_loss = 0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch_to_device(batch, device)
                output = model(batch)
                loss, _ = model.compute_loss(output, batch["targets"])
                val_loss += loss.item() * len(batch["sequences"])
                n_val += len(batch["sequences"])
        val_loss /= max(n_val, 1)

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break

    if best_state:
        model.load_state_dict(best_state)

    # Evaluate
    model.eval()
    results = {}
    for split_name, loader in [("test", test_loader)]:
        all_binary_logits = []
        all_rate_preds = []
        all_labels = []
        all_rate_targets = []

        with torch.no_grad():
            for batch in loader:
                batch = batch_to_device(batch, device)
                output = model(batch)
                binary_logit = output["predictions"]["binary_logit"].squeeze(-1).cpu().numpy()
                rate_pred = output["predictions"]["editing_rate"].squeeze(-1).cpu().numpy()
                labels = batch["targets"]["binary"].cpu().numpy()
                rate_target = batch["targets"]["rate"].cpu().numpy()

                all_binary_logits.append(binary_logit)
                all_rate_preds.append(rate_pred)
                all_labels.append(labels)
                all_rate_targets.append(rate_target)

        binary_logits = np.concatenate(all_binary_logits)
        rate_preds = np.concatenate(all_rate_preds)
        labels = np.concatenate(all_labels)
        rate_targets = np.concatenate(all_rate_targets)

        # Binary metrics
        binary_probs = 1.0 / (1.0 + np.exp(-binary_logits))
        auroc = float(roc_auc_score(labels, binary_probs)) if len(np.unique(labels)) > 1 else float("nan")
        auprc = float(average_precision_score(labels, binary_probs)) if len(np.unique(labels)) > 1 else float("nan")

        # Rate metrics (positive sites only)
        rate_mask = np.isfinite(rate_targets) & (labels == 1)
        if rate_mask.sum() >= 3:
            sp_rho, _ = spearmanr(rate_targets[rate_mask], rate_preds[rate_mask])
            pe_r, _ = pearsonr(rate_targets[rate_mask], rate_preds[rate_mask])
            rate_mse = float(mean_squared_error(rate_targets[rate_mask], rate_preds[rate_mask]))
        else:
            sp_rho = pe_r = rate_mse = float("nan")

        results[split_name] = {
            "auroc": auroc,
            "auprc": auprc,
            "spearman": float(sp_rho) if np.isfinite(sp_rho) else float("nan"),
            "pearson": float(pe_r) if np.isfinite(pe_r) else float("nan"),
            "rate_mse": rate_mse,
            "n_total": len(labels),
            "n_positive": int(labels.sum()),
            "n_rate": int(rate_mask.sum()),
        }

    return results


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)
    torch.manual_seed(42)

    t_global = time.time()

    # Load data
    logger.info("Loading data...")
    pooled_orig = torch.load(EMB_DIR / "rnafm_pooled.pt", weights_only=False)
    pooled_edited = torch.load(EMB_DIR / "rnafm_pooled_edited.pt", weights_only=False)
    tokens_path = EMB_DIR / "rnafm_tokens.pt"
    tokens_edited_path = EMB_DIR / "rnafm_tokens_edited.pt"
    has_tokens = tokens_path.exists() and tokens_edited_path.exists()
    if has_tokens:
        tokens_orig = torch.load(tokens_path, weights_only=False)
        tokens_edited = torch.load(tokens_edited_path, weights_only=False)
    else:
        tokens_orig = tokens_edited = None
        logger.warning("Token embeddings not found — EditRNA-A3A will be skipped")

    splits_df = pd.read_csv(SPLITS_CSV)
    logger.info("Loaded %d sites", len(splits_df))

    sequences = {}
    if SEQ_JSON.exists():
        sequences = json.loads(SEQ_JSON.read_text())

    structure_delta = {}
    if STRUCT_CACHE.exists():
        data = np.load(STRUCT_CACHE, allow_pickle=True)
        feat_key = "delta_features" if "delta_features" in data else "features"
        if "site_ids" in data and feat_key in data:
            sids = data["site_ids"]
            feats = data[feat_key]
            structure_delta = {str(sid): feats[i] for i, sid in enumerate(sids)}

    # Filter to sites with embeddings
    available = set(pooled_orig.keys()) & set(pooled_edited.keys())
    if has_tokens:
        token_available = set(tokens_orig.keys()) & set(tokens_edited.keys())
        all_available = available & token_available
    else:
        all_available = available

    df = splits_df[splits_df["site_id"].isin(all_available)].copy()
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]
    logger.info("Train: %d, Val: %d, Test: %d", len(train_df), len(val_df), len(test_df))

    # Run each variant
    all_results = {}
    for cfg_dict in WINDOW_CONFIGS:
        name = cfg_dict["name"]
        needs_tokens = not cfg_dict.get("pooled_only", False)
        if needs_tokens and not has_tokens:
            logger.info("\n  Skipping %s (requires token embeddings)", name)
            continue
        logger.info("\n" + "=" * 70)
        logger.info("Variant: %s (local_window=%d, pooled_only=%s)",
                     name, cfg_dict["local_window"], cfg_dict["pooled_only"])
        logger.info("=" * 70)
        t0 = time.time()

        results = train_eval_variant(
            cfg_dict, train_df, val_df, test_df,
            sequences, structure_delta,
            tokens_orig, pooled_orig, tokens_edited, pooled_edited,
        )
        elapsed = time.time() - t0
        results["time_seconds"] = elapsed
        all_results[name] = results

        m = results["test"]
        logger.info("  Time: %.1fs  AUROC=%.4f  Spearman=%.4f  n=%d",
                     elapsed, m["auroc"], m["spearman"], m["n_total"])

    total_time = time.time() - t_global

    # Summary table
    print("\n" + "=" * 100)
    print("LOCAL WINDOW ABLATION: EditRNA-A3A VARIANTS")
    print("=" * 100)
    header = f"{'Variant':<20}{'AUROC':>10}{'AUPRC':>10}{'Spearman':>10}{'Pearson':>10}{'Time(s)':>10}"
    print(header)
    print("-" * 100)
    for name, res in all_results.items():
        m = res["test"]
        t = res.get("time_seconds", 0)
        print(f"{name:<20}{m['auroc']:>10.4f}{m['auprc']:>10.4f}"
              f"{m['spearman']:>10.4f}{m['pearson']:>10.4f}{t:>10.1f}")
    print("=" * 100)

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    names = list(all_results.keys())
    aurocs = [all_results[n]["test"]["auroc"] for n in names]
    spearmans = [all_results[n]["test"]["spearman"] for n in names]

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))

    axes[0].bar(range(len(names)), aurocs, color=colors, edgecolor="black")
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels(names, fontsize=9, rotation=20, ha="right")
    axes[0].set_ylabel("AUROC", fontsize=11)
    axes[0].set_title("Binary Classification", fontsize=12, fontweight="bold")
    axes[0].set_ylim(0.9, 1.005)
    for i, v in enumerate(aurocs):
        axes[0].text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=8, fontweight="bold")

    axes[1].bar(range(len(names)), spearmans, color=colors, edgecolor="black")
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels(names, fontsize=9, rotation=20, ha="right")
    axes[1].set_ylabel("Spearman rho", fontsize=11)
    axes[1].set_title("Rate Prediction", fontsize=12, fontweight="bold")
    for i, v in enumerate(spearmans):
        if np.isfinite(v):
            axes[1].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8, fontweight="bold")

    plt.suptitle("EditRNA-A3A: Token Window Size Ablation", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "local_window_ablation.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Save results
    def serialize(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return str(obj)

    json_results = {
        "description": "Window size ablation for EditRNA-A3A architecture variants",
        "variants": all_results,
        "total_time_seconds": total_time,
    }
    with open(OUTPUT_DIR / "local_window_results.json", "w") as f:
        json.dump(json_results, f, indent=2, default=serialize)
    logger.info("Saved results to %s", OUTPUT_DIR / "local_window_results.json")
    logger.info("Total time: %.1fs", total_time)


if __name__ == "__main__":
    main()
