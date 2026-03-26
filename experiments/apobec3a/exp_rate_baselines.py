#!/usr/bin/env python
"""Rate prediction with classic ML baselines on two dataset settings.

Settings:
  1. Levanon-only: train/val/test from advisor_c2t only
  2. Combined: all datasets with editing rates (advisor_c2t + alqassim_2021 + sharma_2015)

Baselines:
  - Mean predictor (predict train mean for all)
  - Median predictor
  - Linear Regression on pooled subtraction embedding
  - Ridge Regression
  - Lasso Regression
  - Random Forest
  - Gradient Boosting (sklearn)
  - SubtractionMLP (DL baseline)
  - PooledMLP (DL baseline)
  - ConcatMLP (DL baseline)
  - Structure-only features (7-dim)

Usage:
    python experiments/apobec3a/exp_rate_baselines.py
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "apobec3a"))

logger = logging.getLogger(__name__)

EMB_DIR = PROJECT_ROOT / "data" / "processed" / "embeddings"
SPLITS_CSV = PROJECT_ROOT / "data" / "processed" / "splits_expanded.csv"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "apobec3a" / "outputs" / "rate_baselines"

# Datasets that have editing rates
RATE_DATASETS = {"advisor_c2t", "alqassim_2021", "sharma_2015"}


def load_embeddings():
    """Load pooled embeddings and structure features."""
    pooled_orig = torch.load(EMB_DIR / "rnafm_pooled.pt", weights_only=False)
    pooled_edited = torch.load(EMB_DIR / "rnafm_pooled_edited.pt", weights_only=False)

    structure_delta = {}
    struct_path = EMB_DIR / "vienna_structure_cache.npz"
    if struct_path.exists():
        data = np.load(struct_path, allow_pickle=True)
        feat_key = "delta_features" if "delta_features" in data else "features"
        if "site_ids" in data and feat_key in data:
            sids = data["site_ids"]
            feats = data[feat_key]
            structure_delta = {str(sid): feats[i] for i, sid in enumerate(sids)}

    return pooled_orig, pooled_edited, structure_delta


def prepare_features(site_ids, pooled_orig, pooled_edited, structure_delta):
    """Prepare feature matrices for ML models."""
    subtraction = []
    concat = []
    pooled_only = []
    struct_feats = []

    for sid in site_ids:
        po = pooled_orig[sid].numpy()
        pe = pooled_edited[sid].numpy()
        subtraction.append(pe - po)
        concat.append(np.concatenate([po, pe]))
        pooled_only.append(po)
        sf = structure_delta.get(sid, np.zeros(7))
        struct_feats.append(sf)

    return {
        "subtraction": np.array(subtraction),       # (N, 640)
        "concat": np.array(concat),                  # (N, 1280)
        "pooled": np.array(pooled_only),             # (N, 640)
        "structure": np.array(struct_feats),         # (N, 7)
        "sub+struct": np.hstack([np.array(subtraction), np.array(struct_feats)]),  # (N, 647)
    }


def compute_rate_metrics(y_true, y_pred):
    """Compute regression metrics."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]

    if len(y_true) < 3:
        return {"spearman": float("nan"), "pearson": float("nan"),
                "mse": float("nan"), "r2": float("nan"), "n": int(len(y_true))}

    sr, _ = spearmanr(y_true, y_pred)
    pr, _ = pearsonr(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {"spearman": float(sr), "pearson": float(pr),
            "mse": float(mse), "r2": float(r2), "n": int(len(y_true))}


def train_dl_rate_model(model_cls, X_train, y_train, X_val, y_val, X_test, y_test,
                        feature_type="subtraction", epochs=100, lr=1e-3):
    """Train a simple DL model for rate regression."""
    import torch.nn as nn

    d_in = X_train.shape[1]

    class RateMLP(nn.Module):
        def __init__(self, d_in):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d_in, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

    model = RateMLP(d_in)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.MSELoss()

    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.float32)
    X_v = torch.tensor(X_val, dtype=torch.float32)
    y_v = torch.tensor(y_val, dtype=torch.float32)
    X_te = torch.tensor(X_test, dtype=torch.float32)

    best_val_loss = float("inf")
    patience = 15
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_tr)
        loss = loss_fn(pred, y_tr)
        loss.backward()
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_v)
            val_loss = loss_fn(val_pred, y_v).item()

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

    model.eval()
    with torch.no_grad():
        test_pred = model(X_te).numpy()

    return test_pred


def run_setting(setting_name, splits_df, pooled_orig, pooled_edited, structure_delta):
    """Run all baselines on one dataset setting."""
    logger.info("\n%s", "=" * 70)
    logger.info("SETTING: %s", setting_name)
    logger.info("=" * 70)

    # Filter to sites with rates
    df = splits_df[
        (splits_df["editing_rate"].notna()) &
        (splits_df["editing_rate"] > 0) &
        (splits_df["site_id"].isin(pooled_orig.keys()))
    ].copy()

    logger.info("Sites with valid rates: %d", len(df))

    # Compute log2 rate target
    rates = df["editing_rate"].values.copy()
    # Normalize: if any rate > 1, assume percentage
    if (rates > 1).any():
        rates = np.where(rates > 1, rates / 100.0, rates)
    df["log2_rate"] = np.log2(rates + 0.01)

    # Split
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    logger.info("Train: %d, Val: %d, Test: %d", len(train_df), len(val_df), len(test_df))

    if len(test_df) < 3:
        logger.warning("Not enough test samples for %s. Skipping.", setting_name)
        return None

    # Prepare features
    train_features = prepare_features(train_df["site_id"].tolist(), pooled_orig, pooled_edited, structure_delta)
    val_features = prepare_features(val_df["site_id"].tolist(), pooled_orig, pooled_edited, structure_delta)
    test_features = prepare_features(test_df["site_id"].tolist(), pooled_orig, pooled_edited, structure_delta)

    y_train = train_df["log2_rate"].values
    y_val = val_df["log2_rate"].values
    y_test = test_df["log2_rate"].values

    # Standardize targets
    y_mean, y_std = y_train.mean(), y_train.std()
    logger.info("Target stats: mean=%.3f, std=%.3f", y_mean, y_std)

    results = {}

    # ---- Simple baselines ----
    # Mean predictor
    mean_pred = np.full_like(y_test, y_mean)
    results["Mean Predictor"] = compute_rate_metrics(y_test, mean_pred)
    results["Mean Predictor"]["feature_type"] = "none"

    # Median predictor
    median_pred = np.full_like(y_test, np.median(y_train))
    results["Median Predictor"] = compute_rate_metrics(y_test, median_pred)
    results["Median Predictor"]["feature_type"] = "none"

    # ---- ML baselines on different feature sets ----
    feature_configs = [
        ("subtraction", "Subtraction (640d)"),
        ("sub+struct", "Subtraction + Structure (647d)"),
        ("pooled", "Pooled-only (640d)"),
        ("structure", "Structure-only (7d)"),
    ]

    ml_models = [
        ("Linear Regression", LinearRegression),
        ("Ridge (α=1)", lambda: Ridge(alpha=1.0)),
        ("Lasso (α=0.01)", lambda: Lasso(alpha=0.01, max_iter=5000)),
        ("Random Forest", lambda: RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=1)),
        ("Gradient Boosting", lambda: GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)),
    ]

    for feat_name, feat_label in feature_configs:
        X_tr = train_features[feat_name]
        X_te = test_features[feat_name]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        for model_name, model_factory in ml_models:
            full_name = f"{model_name} [{feat_label}]"
            try:
                if callable(model_factory) and not isinstance(model_factory, type):
                    model = model_factory()
                else:
                    model = model_factory()
                model.fit(X_tr_s, y_train)
                pred = model.predict(X_te_s)
                results[full_name] = compute_rate_metrics(y_test, pred)
                results[full_name]["feature_type"] = feat_name
            except Exception as e:
                logger.warning("  %s failed: %s", full_name, e)
                results[full_name] = {"spearman": float("nan"), "pearson": float("nan"),
                                      "mse": float("nan"), "r2": float("nan"), "n": 0,
                                      "feature_type": feat_name, "error": str(e)}

    # ---- DL baselines ----
    dl_configs = [
        ("SubtractionMLP", "subtraction"),
        ("PooledMLP", "pooled"),
        ("ConcatMLP", "concat"),
        ("Sub+StructMLP", "sub+struct"),
    ]

    for model_name, feat_name in dl_configs:
        X_tr = train_features[feat_name]
        X_v = val_features[feat_name]
        X_te = test_features[feat_name]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_v_s = scaler.transform(X_v)
        X_te_s = scaler.transform(X_te)

        try:
            pred = train_dl_rate_model(
                None, X_tr_s, y_train, X_v_s, y_val, X_te_s, y_test,
                feature_type=feat_name, epochs=150, lr=1e-3,
            )
            results[model_name] = compute_rate_metrics(y_test, pred)
            results[model_name]["feature_type"] = feat_name
        except Exception as e:
            logger.warning("  %s failed: %s", model_name, e)
            results[model_name] = {"spearman": float("nan"), "pearson": float("nan"),
                                   "mse": float("nan"), "r2": float("nan"), "n": 0,
                                   "feature_type": feat_name, "error": str(e)}

    # ---- Per-dataset breakdown (test set) ----
    per_dataset = {}
    for ds_name in RATE_DATASETS:
        ds_test = test_df[test_df["dataset_source"] == ds_name]
        if len(ds_test) < 3:
            continue
        per_dataset[ds_name] = {"n": int(len(ds_test)),
                                "mean_rate": float(ds_test["editing_rate"].mean()),
                                "mean_log2_rate": float(ds_test["log2_rate"].mean())}

    # Print summary table
    print(f"\n{'=' * 100}")
    print(f"RATE PREDICTION BASELINES: {setting_name}")
    print(f"{'=' * 100}")
    print(f"{'Model':<45} {'Spearman':>10} {'Pearson':>10} {'MSE':>10} {'R²':>10} {'N':>6}")
    print(f"{'-' * 100}")
    for name, m in sorted(results.items(), key=lambda x: -x[1].get("spearman", float("-inf")) if not np.isnan(x[1].get("spearman", float("nan"))) else float("-inf")):
        sp = m.get("spearman", float("nan"))
        pr = m.get("pearson", float("nan"))
        mse = m.get("mse", float("nan"))
        r2 = m.get("r2", float("nan"))
        n = m.get("n", 0)
        print(f"{name:<45} {sp:>10.4f} {pr:>10.4f} {mse:>10.4f} {r2:>10.4f} {n:>6}")
    print(f"{'=' * 100}")

    return {
        "setting": setting_name,
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "target_mean": float(y_mean),
        "target_std": float(y_std),
        "models": results,
        "per_dataset": per_dataset,
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading embeddings...")
    pooled_orig, pooled_edited, structure_delta = load_embeddings()
    logger.info("Loaded %d pooled embeddings, %d structure features",
                len(pooled_orig), len(structure_delta))

    splits_df = pd.read_csv(SPLITS_CSV)
    logger.info("Loaded splits: %d sites", len(splits_df))

    all_results = {}

    # Setting 1: Levanon only
    levanon_df = splits_df[splits_df["dataset_source"] == "advisor_c2t"].copy()
    result = run_setting("Levanon Only (advisor_c2t)", levanon_df,
                         pooled_orig, pooled_edited, structure_delta)
    if result:
        all_results["levanon_only"] = result

    # Setting 2: Combined (all datasets with rates)
    combined_df = splits_df[splits_df["dataset_source"].isin(RATE_DATASETS)].copy()
    result = run_setting("Combined (Levanon + Alqassim + Sharma)", combined_df,
                         pooled_orig, pooled_edited, structure_delta)
    if result:
        all_results["combined"] = result

    # Save results
    with open(OUTPUT_DIR / "rate_baselines_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating,)) else int(x) if isinstance(x, (np.integer,)) else str(x))

    logger.info("\nResults saved to %s", OUTPUT_DIR / "rate_baselines_results.json")


if __name__ == "__main__":
    main()
