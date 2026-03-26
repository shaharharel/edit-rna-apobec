#!/usr/bin/env python
"""Multi-seed evaluation for confidence intervals on key metrics.

Runs each model with 5 random seeds and reports mean +/- std for all metrics.
Tests statistical significance of EditRNA-A3A vs SubtractionMLP.

Models evaluated:
  - SubtractionMLP (baseline)
  - ConcatMLP
  - EditRNA-A3A (ours)

Metrics with confidence intervals:
  - Binary classification: AUROC, AUPRC, F1
  - Rate prediction: Spearman, Pearson, MSE

Usage:
    python experiments/apobec3a/exp_multiseed.py
    python experiments/apobec3a/exp_multiseed.py --seeds 10  # more seeds
"""

import argparse
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
from scipy.stats import spearmanr, pearsonr, wilcoxon, ttest_rel
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader

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
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "apobec3a" / "outputs" / "multiseed"

N_SEEDS = 5
SEED_LIST = [42, 123, 456, 789, 1024]


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def compute_log2_rate(editing_rate):
    if pd.isna(editing_rate) or editing_rate < 0:
        return float("nan")
    rate = float(editing_rate)
    if rate > 1.0:
        rate = rate / 100.0
    return np.log2(rate + 0.01)


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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data():
    """Load all shared data."""
    logger.info("Loading embeddings...")
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
    logger.info("  %d embeddings loaded", len(pooled_orig))

    splits_df = pd.read_csv(SPLITS_CSV)
    available = set(pooled_orig.keys()) & set(pooled_edited.keys())
    splits_df = splits_df[splits_df["site_id"].isin(available)].copy()

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

    return (tokens_orig, pooled_orig, tokens_edited, pooled_edited,
            splits_df, sequences, structure_delta)


# ---------------------------------------------------------------------------
# SubtractionMLP
# ---------------------------------------------------------------------------

class SubtractionDataset(Dataset):
    def __init__(self, site_ids, labels, pooled_orig, pooled_edited):
        self.site_ids = site_ids
        self.labels = labels
        self.pooled_orig = pooled_orig
        self.pooled_edited = pooled_edited

    def __len__(self):
        return len(self.site_ids)

    def __getitem__(self, idx):
        sid = self.site_ids[idx]
        x = self.pooled_edited[sid] - self.pooled_orig[sid]
        return x, torch.tensor(self.labels[idx], dtype=torch.float32)


class SubtractionMLP(nn.Module):
    def __init__(self, d_input=640, hidden=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_input, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_subtraction_binary(splits_df, pooled_orig, pooled_edited, seed=42):
    """Train SubtractionMLP for binary classification."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_df = splits_df[splits_df["split"] == "train"]
    val_df = splits_df[splits_df["split"] == "val"]
    test_df = splits_df[splits_df["split"] == "test"]

    def make_loader(df, shuffle=False, bs=64):
        ds = SubtractionDataset(df["site_id"].tolist(),
                                 df["label"].values.astype(np.float32),
                                 pooled_orig, pooled_edited)
        return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=0)

    train_loader = make_loader(train_df, shuffle=True)
    val_loader = make_loader(val_df)
    test_loader = make_loader(test_df)

    model = SubtractionMLP()
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-7)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(1, 51):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = F.binary_cross_entropy_with_logits(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        vp, vl = [], []
        with torch.no_grad():
            for x, y in val_loader:
                vp.append(torch.sigmoid(model(x)).numpy())
                vl.append(y.numpy())
        vp, vl = np.concatenate(vp), np.concatenate(vl)
        val_loss = float(F.binary_cross_entropy(torch.tensor(vp), torch.tensor(vl)))

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
        if patience_counter >= 10:
            break

    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    all_p, all_l = [], []
    with torch.no_grad():
        for x, y in test_loader:
            all_p.append(torch.sigmoid(model(x)).numpy())
            all_l.append(y.numpy())
    y_pred = np.concatenate(all_p)
    y_true = np.concatenate(all_l)

    auroc = roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) >= 2 else float("nan")
    auprc = average_precision_score(y_true, y_pred) if len(np.unique(y_true)) >= 2 else float("nan")
    y_binary = (y_pred >= 0.5).astype(int)
    f1 = f1_score(y_true, y_binary, zero_division=0)

    return {"auroc": auroc, "auprc": auprc, "f1": f1}


def train_subtraction_rate(splits_df, pooled_orig, pooled_edited, seed=42):
    """Train SubtractionMLP for rate regression."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    pos_df = splits_df[splits_df["label"] == 1].copy()
    pos_df["log2_rate"] = pos_df["editing_rate"].apply(compute_log2_rate)
    pos_df = pos_df[pos_df["log2_rate"].notna()]

    if len(pos_df) < 50:
        return {"spearman": float("nan"), "pearson": float("nan"), "mse": float("nan")}

    train_df = pos_df[pos_df["split"] == "train"]
    val_df = pos_df[pos_df["split"] == "val"]
    test_df = pos_df[pos_df["split"] == "test"]

    class RateDS(Dataset):
        def __init__(self, sids, targets, po, pe):
            self.sids, self.targets, self.po, self.pe = sids, targets, po, pe

        def __len__(self):
            return len(self.sids)

        def __getitem__(self, idx):
            sid = self.sids[idx]
            x = self.pe[sid] - self.po[sid]
            return x, torch.tensor(self.targets[idx], dtype=torch.float32)

    def make_loader(df, shuffle=False):
        ds = RateDS(df["site_id"].tolist(), df["log2_rate"].values.astype(np.float32),
                    pooled_orig, pooled_edited)
        return DataLoader(ds, batch_size=64, shuffle=shuffle, num_workers=0)

    train_loader = make_loader(train_df, shuffle=True)
    val_loader = make_loader(val_df)
    test_loader = make_loader(test_df)

    model = SubtractionMLP()
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=80, eta_min=1e-7)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(1, 81):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = F.mse_loss(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        vp, vt = [], []
        with torch.no_grad():
            for x, y in val_loader:
                vp.append(model(x).numpy())
                vt.append(y.numpy())
        val_loss = float(np.mean((np.concatenate(vp) - np.concatenate(vt)) ** 2))

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
        if patience_counter >= 15:
            break

    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    tp, tt = [], []
    with torch.no_grad():
        for x, y in test_loader:
            tp.append(model(x).numpy())
            tt.append(y.numpy())
    y_pred = np.concatenate(tp)
    y_true = np.concatenate(tt)

    sp, _ = spearmanr(y_true, y_pred)
    pe, _ = pearsonr(y_true, y_pred)
    mse = float(np.mean((y_pred - y_true) ** 2))

    return {"spearman": float(sp), "pearson": float(pe), "mse": mse}


# ---------------------------------------------------------------------------
# EditRNA-A3A
# ---------------------------------------------------------------------------

def build_editrna_samples(df, sequences, structure_delta, window_size=100):
    """Build APOBECSiteSample list."""
    samples = []
    for _, row in df.iterrows():
        site_id = str(row["site_id"])
        label = int(row["label"])

        seq = sequences.get(site_id, "A" * (window_size * 2 + 1))
        edit_pos = min(window_size, len(seq) // 2)
        seq_list = list(seq)
        seq_list[edit_pos] = "C"
        seq = "".join(seq_list)

        flanking = get_flanking_context(seq, edit_pos)

        struct_d = structure_delta.get(site_id)
        if struct_d is not None:
            struct_d = np.array(struct_d, dtype=np.float32)

        editing_rate = row.get("editing_rate", np.nan)
        if pd.notna(editing_rate) and editing_rate > 0:
            rate = float(editing_rate)
            if rate > 1.0:
                rate = rate / 100.0
            log2_rate = np.log2(rate + 0.01)
        else:
            log2_rate = float("nan")

        concordance = np.zeros(5, dtype=np.float32)
        if label == 0:
            concordance[0] = 1.0

        sample = APOBECSiteSample(
            sequence=seq, edit_pos=edit_pos,
            is_edited=float(label),
            editing_rate_log2=log2_rate if label == 1 else float("nan"),
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


def train_editrna_binary(splits_df, sequences, structure_delta,
                          tokens_orig, pooled_orig, tokens_edited, pooled_edited,
                          seed=42):
    """Train EditRNA-A3A and evaluate binary classification."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")

    train_df = splits_df[splits_df["split"] == "train"]
    val_df = splits_df[splits_df["split"] == "val"]
    test_df = splits_df[splits_df["split"] == "test"]

    data_config = APOBECDataConfig(window_size=100)

    train_ds = APOBECDataset(build_editrna_samples(train_df, sequences, structure_delta), data_config)
    val_ds = APOBECDataset(build_editrna_samples(val_df, sequences, structure_delta), data_config)
    test_ds = APOBECDataset(build_editrna_samples(test_df, sequences, structure_delta), data_config)

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
    )
    model = EditRNA_A3A(config=config, primary_encoder=cached_encoder).to(device)
    optimizer = AdamW(model.get_parameter_groups(), lr=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-7)

    best_val_auroc = -1.0
    patience_counter = 0
    best_state = None

    for epoch in range(1, 41):
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
        all_y, all_s = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch_to_device(batch, device)
                output = model(batch)
                logits = output["predictions"]["binary_logit"].squeeze(-1)
                all_s.append(torch.sigmoid(logits).cpu().numpy())
                all_y.append(batch["targets"]["binary"].cpu().numpy())
        y_true = np.concatenate(all_y)
        y_score = np.concatenate(all_s)
        val_auroc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) >= 2 else 0.0

        if val_auroc > best_val_auroc + 1e-4:
            best_val_auroc = val_auroc
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
        if patience_counter >= 10:
            break

    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    all_y, all_s = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch_to_device(batch, device)
            output = model(batch)
            logits = output["predictions"]["binary_logit"].squeeze(-1)
            all_s.append(torch.sigmoid(logits).cpu().numpy())
            all_y.append(batch["targets"]["binary"].cpu().numpy())
    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_s)

    auroc = roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) >= 2 else float("nan")
    auprc = average_precision_score(y_true, y_pred) if len(np.unique(y_true)) >= 2 else float("nan")
    y_binary = (y_pred >= 0.5).astype(int)
    f1 = f1_score(y_true, y_binary, zero_division=0)

    # Also get rate predictions
    rate_metrics = {"spearman": float("nan"), "pearson": float("nan"), "mse": float("nan")}
    all_rp, all_rt = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch_to_device(batch, device)
            output = model(batch)
            rate_pred = output["predictions"]["editing_rate"].squeeze(-1).cpu().numpy()
            rate_true = batch["targets"]["rate"].cpu().numpy()
            mask = np.isfinite(rate_true)
            if mask.any():
                all_rp.append(rate_pred[mask])
                all_rt.append(rate_true[mask])

    if all_rp:
        rp = np.concatenate(all_rp)
        rt = np.concatenate(all_rt)
        if len(rp) >= 3:
            sp, _ = spearmanr(rt, rp)
            pe, _ = pearsonr(rt, rp)
            rate_metrics = {
                "spearman": float(sp), "pearson": float(pe),
                "mse": float(np.mean((rp - rt) ** 2)),
            }

    return {"auroc": auroc, "auprc": auprc, "f1": f1, **{f"rate_{k}": v for k, v in rate_metrics.items()}}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=N_SEEDS, help="Number of seeds")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    seeds = SEED_LIST[:args.seeds]
    logger.info("Running %d seeds: %s", len(seeds), seeds)

    t_total = time.time()

    # Load data
    (tokens_orig, pooled_orig, tokens_edited, pooled_edited,
     splits_df, sequences, structure_delta) = load_data()

    # ---------------------------------------------------------------------------
    # Run all models across all seeds
    # ---------------------------------------------------------------------------

    has_tokens = tokens_orig is not None
    models_to_run = ["SubtractionMLP"]
    if has_tokens:
        models_to_run.append("EditRNA-A3A")
    else:
        logger.warning("Skipping EditRNA-A3A (no token embeddings)")
    all_results = {m: [] for m in models_to_run}

    for seed_idx, seed in enumerate(seeds):
        logger.info("\n" + "=" * 60)
        logger.info("SEED %d/%d (seed=%d)", seed_idx + 1, len(seeds), seed)
        logger.info("=" * 60)

        # SubtractionMLP
        logger.info("\nSubtractionMLP (binary)...")
        t0 = time.time()
        sub_binary = train_subtraction_binary(splits_df, pooled_orig, pooled_edited, seed=seed)
        logger.info("  Binary: AUROC=%.4f, AUPRC=%.4f (%.1fs)",
                    sub_binary["auroc"], sub_binary["auprc"], time.time() - t0)

        logger.info("SubtractionMLP (rate)...")
        t0 = time.time()
        sub_rate = train_subtraction_rate(splits_df, pooled_orig, pooled_edited, seed=seed)
        logger.info("  Rate: Spearman=%.4f, Pearson=%.4f (%.1fs)",
                    sub_rate["spearman"], sub_rate["pearson"], time.time() - t0)

        all_results["SubtractionMLP"].append({
            **sub_binary,
            **{f"rate_{k}": v for k, v in sub_rate.items()},
        })

        # EditRNA-A3A
        if has_tokens:
            logger.info("\nEditRNA-A3A (binary + rate)...")
            t0 = time.time()
            editrna_metrics = train_editrna_binary(
                splits_df, sequences, structure_delta,
                tokens_orig, pooled_orig, tokens_edited, pooled_edited,
                seed=seed,
            )
            logger.info("  Binary: AUROC=%.4f, AUPRC=%.4f (%.1fs)",
                        editrna_metrics["auroc"], editrna_metrics["auprc"], time.time() - t0)
            logger.info("  Rate: Spearman=%.4f", editrna_metrics.get("rate_spearman", float("nan")))

            all_results["EditRNA-A3A"].append(editrna_metrics)

    # ---------------------------------------------------------------------------
    # Aggregate results
    # ---------------------------------------------------------------------------

    logger.info("\n" + "=" * 70)
    logger.info("MULTI-SEED RESULTS (%d seeds)", len(seeds))
    logger.info("=" * 70)

    summary = {}
    metric_names = ["auroc", "auprc", "f1", "rate_spearman", "rate_pearson", "rate_mse"]

    for model_name in models_to_run:
        results = all_results[model_name]
        model_summary = {}

        for metric in metric_names:
            vals = [r.get(metric, float("nan")) for r in results]
            vals = [v for v in vals if np.isfinite(v)]
            if vals:
                mean = np.mean(vals)
                std = np.std(vals)
                model_summary[metric] = {"mean": float(mean), "std": float(std), "values": vals}
                logger.info("  %-20s %-18s: %.4f +/- %.4f", model_name, metric, mean, std)
            else:
                model_summary[metric] = {"mean": float("nan"), "std": float("nan"), "values": []}

        summary[model_name] = model_summary

    # Statistical significance tests
    logger.info("\n--- Statistical Significance ---")
    sig_results = {}

    if "EditRNA-A3A" not in all_results:
        logger.info("  Skipped (EditRNA-A3A not available)")
    for metric in metric_names:
        if "EditRNA-A3A" not in all_results:
            break
        sub_vals = [r.get(metric, float("nan")) for r in all_results["SubtractionMLP"]]
        ern_vals = [r.get(metric, float("nan")) for r in all_results["EditRNA-A3A"]]

        # Filter to finite values (paired)
        pairs = [(s, e) for s, e in zip(sub_vals, ern_vals)
                 if np.isfinite(s) and np.isfinite(e)]

        if len(pairs) >= 3:
            sub_arr = np.array([p[0] for p in pairs])
            ern_arr = np.array([p[1] for p in pairs])

            t_stat, t_p = ttest_rel(ern_arr, sub_arr)
            try:
                w_stat, w_p = wilcoxon(ern_arr - sub_arr)
            except ValueError:
                w_stat, w_p = float("nan"), float("nan")

            diff = np.mean(ern_arr) - np.mean(sub_arr)
            logger.info("  %s: EditRNA - Sub = %.4f (t-test p=%.4f, Wilcoxon p=%.4f)",
                        metric, diff, t_p, w_p)
            sig_results[metric] = {
                "mean_diff": float(diff),
                "t_test_p": float(t_p),
                "wilcoxon_p": float(w_p) if np.isfinite(w_p) else None,
            }

    # ---------------------------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------------------------

    # Bar chart with error bars
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Binary metrics
    binary_metrics = ["auroc", "auprc", "f1"]
    x = np.arange(len(binary_metrics))
    width = 0.3

    for i, model_name in enumerate(models_to_run):
        means = [summary[model_name][m]["mean"] for m in binary_metrics]
        stds = [summary[model_name][m]["std"] for m in binary_metrics]
        axes[0].bar(x + i * width, means, width, yerr=stds,
                    label=model_name, alpha=0.8, capsize=4)
        for j, (m, s) in enumerate(zip(means, stds)):
            if np.isfinite(m):
                axes[0].text(x[j] + i * width, m + s + 0.005,
                             f"{m:.3f}\n±{s:.3f}", ha="center", fontsize=7)

    axes[0].set_xticks(x + width / 2)
    axes[0].set_xticklabels(["AUROC", "AUPRC", "F1"])
    axes[0].set_ylabel("Score")
    axes[0].set_title(f"Binary Classification ({len(seeds)} seeds)")
    axes[0].legend()
    axes[0].set_ylim(0, 1.1)
    axes[0].grid(True, alpha=0.3, axis="y")

    # Rate metrics
    rate_metrics_plot = ["rate_spearman", "rate_pearson"]
    x2 = np.arange(len(rate_metrics_plot))

    for i, model_name in enumerate(models_to_run):
        means = [summary[model_name][m]["mean"] for m in rate_metrics_plot]
        stds = [summary[model_name][m]["std"] for m in rate_metrics_plot]
        axes[1].bar(x2 + i * width, means, width, yerr=stds,
                    label=model_name, alpha=0.8, capsize=4)
        for j, (m, s) in enumerate(zip(means, stds)):
            if np.isfinite(m):
                axes[1].text(x2[j] + i * width, m + s + 0.01,
                             f"{m:.3f}\n±{s:.3f}", ha="center", fontsize=7)

    axes[1].set_xticks(x2 + width / 2)
    axes[1].set_xticklabels(["Spearman", "Pearson"])
    axes[1].set_ylabel("Correlation")
    axes[1].set_title(f"Rate Prediction ({len(seeds)} seeds)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.suptitle("Multi-Seed Evaluation: Mean ± Std", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "multiseed_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Save results
    final_results = {
        "n_seeds": len(seeds),
        "seeds": seeds,
        "summary": summary,
        "significance": sig_results,
        "raw_results": {k: v for k, v in all_results.items()},
    }

    with open(OUTPUT_DIR / "multiseed_results.json", "w") as f:
        json.dump(final_results, f, indent=2, default=str)

    elapsed = time.time() - t_total
    logger.info("\nTotal time: %.1fs", elapsed)
    logger.info("Results saved to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
