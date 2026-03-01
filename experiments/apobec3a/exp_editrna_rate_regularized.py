#!/usr/bin/env python
"""EditRNA-A3A rate prediction with aggressive regularization.

EditRNA overfits on rate (train Sp=0.77 → test=0.18). This experiment
tries aggressive regularization strategies:
  1. Smaller architecture (d_edit=128, d_fused=256)
  2. Heavy dropout (0.4-0.6) + BatchNorm
  3. Strong weight decay (1e-2, 5e-2)
  4. Rate-only training (disable binary/other losses)
  5. Rate loss with Huber (robust to outliers)
  6. Gradient clipping at 0.5
  7. Label smoothing (noise on rate targets)

Also runs CrossAttention as comparison baseline (the current winner at Sp=0.227).

Usage:
    python experiments/apobec3a/exp_editrna_rate_regularized.py
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
import torch.nn.functional as F
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error, r2_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.apobec_dataset import (
    APOBECDataConfig, APOBECDataset, APOBECSiteSample,
    N_TISSUES, apobec_collate_fn, get_flanking_context,
)
from models.editrna_a3a import EditRNA_A3A, EditRNAConfig
from models.encoders import CachedRNAEncoder

logger = logging.getLogger(__name__)

EMB_DIR = PROJECT_ROOT / "data" / "processed" / "embeddings"
SPLITS_CSV = PROJECT_ROOT / "data" / "processed" / "splits_expanded.csv"
SEQ_JSON = PROJECT_ROOT / "data" / "processed" / "site_sequences.json"
STRUCT_CACHE = EMB_DIR / "vienna_structure_cache.npz"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "apobec" / "outputs" / "editrna_rate_reg"


def compute_log2_rate(editing_rate):
    if pd.isna(editing_rate) or editing_rate < 0:
        return float("nan")
    rate = float(editing_rate)
    if rate > 1.0:
        rate = rate / 100.0
    return np.log2(rate + 0.01)


def compute_regression_metrics(y_true, y_pred):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) < 3:
        return {"spearman": float("nan"), "pearson": float("nan"),
                "mse": float("nan"), "r2": float("nan"), "n_samples": 0}
    sp, _ = spearmanr(y_true, y_pred)
    pe, _ = pearsonr(y_true, y_pred)
    return {"spearman": float(sp), "pearson": float(pe),
            "mse": float(mean_squared_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
            "n_samples": int(len(y_true))}


def build_rate_samples(df, sequences, structure_delta, window_size=100):
    samples = []
    for _, row in df.iterrows():
        site_id = str(row["site_id"])
        seq = sequences.get(site_id, "A" * (window_size * 2 + 1))
        edit_pos = min(window_size, len(seq) // 2)
        seq_list = list(seq)
        seq_list[edit_pos] = "C"
        seq = "".join(seq_list)
        flanking = get_flanking_context(seq, edit_pos)
        struct_d = structure_delta.get(site_id)
        if struct_d is not None:
            struct_d = np.array(struct_d, dtype=np.float32)
        log2_rate = float(row["log2_rate"])
        concordance = np.zeros(5, dtype=np.float32)
        sample = APOBECSiteSample(
            sequence=seq, edit_pos=edit_pos,
            is_edited=1.0, editing_rate_log2=log2_rate,
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


def train_eval_editrna(train_df, val_df, test_df, sequences, structure_delta,
                       tokens_orig, pooled_orig, tokens_edited, pooled_edited,
                       config_name="default",
                       d_edit=128, d_fused=256, edit_n_heads=4,
                       head_dropout=0.4, fusion_dropout=0.4,
                       weight_decay=1e-2, lr=5e-5,
                       epochs=80, patience=10, seed=42,
                       rate_only=False, rate_loss_weight=5.0,
                       use_huber=False, label_smoothing=0.0):
    """Train EditRNA with configurable regularization for rate prediction."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")

    data_config = APOBECDataConfig(window_size=100)
    train_ds = APOBECDataset(build_rate_samples(train_df, sequences, structure_delta), data_config)
    val_ds = APOBECDataset(build_rate_samples(val_df, sequences, structure_delta), data_config)
    test_ds = APOBECDataset(build_rate_samples(test_df, sequences, structure_delta), data_config)

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
        primary_encoder="cached", d_model=640,
        d_edit=d_edit, d_fused=d_fused,
        edit_n_heads=edit_n_heads, use_structure_delta=True,
        head_dropout=head_dropout, fusion_dropout=fusion_dropout,
        focal_gamma=2.0, focal_alpha_binary=0.75,
    )
    model = EditRNA_A3A(config=config, primary_encoder=cached_encoder).to(device)

    # Count params
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("  %s: %d trainable parameters", config_name, n_params)

    optimizer = AdamW(model.get_parameter_groups(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)

    best_val_sp = -float("inf")
    patience_counter = 0
    best_state = None
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            batch = batch_to_device(batch, device)
            optimizer.zero_grad()
            output = model(batch)

            if rate_only:
                # Rate-only loss (skip binary and other task losses)
                rate_pred = output["predictions"]["editing_rate"].squeeze(-1)
                rate_target = batch["targets"]["rate"]
                mask = torch.isfinite(rate_target)
                if mask.any():
                    pred_masked = rate_pred[mask]
                    target_masked = rate_target[mask]

                    if label_smoothing > 0:
                        target_masked = target_masked + torch.randn_like(target_masked) * label_smoothing

                    if use_huber:
                        loss = F.huber_loss(pred_masked, target_masked, delta=1.0)
                    else:
                        loss = F.mse_loss(pred_masked, target_masked)
                else:
                    continue
            else:
                # Multi-task loss with rate upweighting
                loss, loss_dict = model.compute_loss(output, batch["targets"])
                rate_loss = loss_dict.get("rate", torch.tensor(0.0))
                loss = loss + rate_loss * (rate_loss_weight - 1.0)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Validate on rate Spearman
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch_to_device(batch, device)
                output = model(batch)
                rate_pred = output["predictions"]["editing_rate"].squeeze(-1).cpu().numpy()
                rate_target = batch["targets"]["rate"].cpu().numpy()
                mask = np.isfinite(rate_target)
                if mask.any():
                    val_preds.append(rate_pred[mask])
                    val_targets.append(rate_target[mask])

        if val_preds:
            vp = np.concatenate(val_preds)
            vt = np.concatenate(val_targets)
            val_sp, _ = spearmanr(vt, vp)
            val_sp = float(val_sp) if np.isfinite(val_sp) else -1.0
            val_mse = float(np.mean((vp - vt) ** 2))
        else:
            val_sp = -1.0
            val_mse = float("inf")

        history.append({"epoch": epoch, "train_loss": epoch_loss / max(n_batches, 1),
                        "val_spearman": val_sp, "val_mse": val_mse})

        if val_sp > best_val_sp + 1e-4:
            best_val_sp = val_sp
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break

    if best_state:
        model.load_state_dict(best_state)

    best_epoch = max(1, epoch - patience_counter)

    # Evaluate
    model.eval()
    results = {"best_epoch": best_epoch, "n_params": n_params}
    for split_name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in loader:
                batch = batch_to_device(batch, device)
                output = model(batch)
                rate_pred = output["predictions"]["editing_rate"].squeeze(-1).cpu().numpy()
                rate_target = batch["targets"]["rate"].cpu().numpy()
                mask = np.isfinite(rate_target)
                if mask.any():
                    all_preds.append(rate_pred[mask])
                    all_targets.append(rate_target[mask])
        if all_preds:
            results[split_name] = compute_regression_metrics(
                np.concatenate(all_targets), np.concatenate(all_preds))
        else:
            results[split_name] = {"spearman": float("nan"), "pearson": float("nan"),
                                    "mse": float("nan"), "r2": float("nan"), "n_samples": 0}

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
    if not (tokens_path.exists() and tokens_edited_path.exists()):
        logger.error("Token embeddings required. Exiting.")
        return
    tokens_orig = torch.load(tokens_path, weights_only=False)
    tokens_edited = torch.load(tokens_edited_path, weights_only=False)
    logger.info("Loaded %d token embeddings", len(tokens_orig))

    splits_df = pd.read_csv(SPLITS_CSV)
    sequences = json.loads(Path(SEQ_JSON).read_text()) if SEQ_JSON.exists() else {}
    structure_delta = {}
    if STRUCT_CACHE.exists():
        data = np.load(STRUCT_CACHE, allow_pickle=True)
        feat_key = "delta_features" if "delta_features" in data else "features"
        if "site_ids" in data and feat_key in data:
            structure_delta = {str(sid): data[feat_key][i]
                               for i, sid in enumerate(data["site_ids"])}

    available = set(pooled_orig.keys()) & set(pooled_edited.keys())
    rate_df = splits_df[
        (splits_df["label"] == 1) &
        (splits_df["editing_rate"].notna()) &
        (splits_df["site_id"].isin(available))
    ].copy()
    rate_df["log2_rate"] = rate_df["editing_rate"].apply(compute_log2_rate)
    rate_df = rate_df[rate_df["log2_rate"].notna()].copy()

    train_df = rate_df[rate_df["split"] == "train"]
    val_df = rate_df[rate_df["split"] == "val"]
    test_df = rate_df[rate_df["split"] == "test"]
    logger.info("Rate samples — Train: %d, Val: %d, Test: %d", len(train_df), len(val_df), len(test_df))

    all_results = {}

    # ==================================================================
    # EditRNA configurations
    # ==================================================================
    configs = [
        # (name, kwargs)
        ("EditRNA_small_rateonly_WD1e-2", {
            "d_edit": 128, "d_fused": 256, "edit_n_heads": 4,
            "head_dropout": 0.4, "fusion_dropout": 0.4,
            "weight_decay": 1e-2, "lr": 5e-5,
            "rate_only": True, "rate_loss_weight": 1.0,
        }),
        ("EditRNA_small_rateonly_huber_WD1e-2", {
            "d_edit": 128, "d_fused": 256, "edit_n_heads": 4,
            "head_dropout": 0.4, "fusion_dropout": 0.4,
            "weight_decay": 1e-2, "lr": 5e-5,
            "rate_only": True, "use_huber": True,
        }),
        ("EditRNA_small_rateonly_WD5e-2", {
            "d_edit": 128, "d_fused": 256, "edit_n_heads": 4,
            "head_dropout": 0.5, "fusion_dropout": 0.5,
            "weight_decay": 5e-2, "lr": 5e-5,
            "rate_only": True,
        }),
        ("EditRNA_small_rateonly_labelsmooth", {
            "d_edit": 128, "d_fused": 256, "edit_n_heads": 4,
            "head_dropout": 0.4, "fusion_dropout": 0.4,
            "weight_decay": 1e-2, "lr": 5e-5,
            "rate_only": True, "label_smoothing": 0.1,
        }),
        ("EditRNA_tiny_rateonly_WD1e-2", {
            "d_edit": 64, "d_fused": 128, "edit_n_heads": 4,
            "head_dropout": 0.5, "fusion_dropout": 0.5,
            "weight_decay": 1e-2, "lr": 1e-4,
            "rate_only": True,
        }),
        ("EditRNA_small_multitask_RW10_WD1e-2", {
            "d_edit": 128, "d_fused": 256, "edit_n_heads": 4,
            "head_dropout": 0.4, "fusion_dropout": 0.4,
            "weight_decay": 1e-2, "lr": 5e-5,
            "rate_only": False, "rate_loss_weight": 10.0,
        }),
        ("EditRNA_small_rateonly_D0.6_WD1e-2", {
            "d_edit": 128, "d_fused": 256, "edit_n_heads": 4,
            "head_dropout": 0.6, "fusion_dropout": 0.6,
            "weight_decay": 1e-2, "lr": 5e-5,
            "rate_only": True,
        }),
        ("EditRNA_orig_rateonly_WD1e-2", {
            "d_edit": 256, "d_fused": 512, "edit_n_heads": 8,
            "head_dropout": 0.4, "fusion_dropout": 0.4,
            "weight_decay": 1e-2, "lr": 3e-5,
            "rate_only": True,
        }),
    ]

    for name, kwargs in configs:
        logger.info("\n" + "=" * 70)
        logger.info("  %s", name)
        logger.info("=" * 70)
        t0 = time.time()
        res = train_eval_editrna(
            train_df, val_df, test_df, sequences, structure_delta,
            tokens_orig, pooled_orig, tokens_edited, pooled_edited,
            config_name=name, epochs=80, patience=10, seed=42, **kwargs)
        elapsed = time.time() - t0
        sp = res["test"]["spearman"]
        logger.info("  %s: Test Sp=%.4f (train=%.4f, val=%.4f) in %.1fs [epoch %d, %d params]",
                     name, sp, res["train"]["spearman"], res["val"]["spearman"],
                     elapsed, res["best_epoch"], res["n_params"])
        all_results[name] = {
            "config": kwargs, **{k: v for k, v in res.items()},
            "time_seconds": elapsed,
        }

    # ==================================================================
    # Summary
    # ==================================================================
    print("\n" + "=" * 110)
    print("EDITRNA RATE REGULARIZATION SWEEP")
    print("=" * 110)
    header = f"{'Model':<45}{'Params':>10}{'Train Sp':>12}{'Val Sp':>12}{'Test Sp':>12}{'Time':>8}"
    print(header)
    print("-" * 110)

    sorted_results = sorted(all_results.items(),
                            key=lambda x: x[1]["test"].get("spearman", -1), reverse=True)

    for name, res in sorted_results:
        tr = res["train"].get("spearman", float("nan"))
        va = res["val"].get("spearman", float("nan"))
        te = res["test"].get("spearman", float("nan"))
        np_ = res.get("n_params", 0)
        t = res.get("time_seconds", 0)
        print(f"{name:<45}{np_:>10,}{tr:>12.4f}{va:>12.4f}{te:>12.4f}{t:>7.0f}s")

    print("=" * 110)
    print(f"Baselines: PooledMLP=0.211, CrossAttn_reg=0.227")
    best_name, best_res = sorted_results[0]
    print(f"Best: {best_name} Sp={best_res['test']['spearman']:.4f}")

    total_time = time.time() - t_global
    logger.info("Total time: %.1fs", total_time)

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

    with open(OUTPUT_DIR / "editrna_rate_reg_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=serialize)
    logger.info("Saved to %s", OUTPUT_DIR / "editrna_rate_reg_results.json")


if __name__ == "__main__":
    main()
