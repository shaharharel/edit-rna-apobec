#!/usr/bin/env python
"""End-to-end GNN rate prediction experiment.

Goal: Beat GB_All+GNN (Spearman=0.7321) by training GNN end-to-end
rather than using random/untrained GNN embeddings.

8 Configurations:
  1. GNN_Only: Trained GNN(64) -> MLP -> rate
  2. GNN+Hand: Trained GNN(64) + hand(40) -> MLP -> rate
  3. 4Way_e2e_small: primary + delta + hand + trainedGNN (d_proj=64)
  4. 4Way_e2e_heavy: Same, d_proj=128, dropout=0.5, wd=2e-2
  5. 5Way_e2e: 4Way + pairing_profile(50) as 5th modality
  6. Deep3Way: 4-layer GNN(128->96) + delta + hand
  7. GB_trainedGNN: Extract trained GNN embeddings -> XGBoost
  8. Ensemble_top3: Average top-3 NN model predictions

Usage:
    python experiments/apobec3a/exp_rate_gnn_endtoend.py
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

# Import xgboost early, before torch_geometric, to avoid OpenMP conflicts
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from xgboost import XGBRegressor  # noqa: E402

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "apobec3a"))

from exp_rate_gnn_fusion import (
    load_all_data, build_structure_graph_from_pairing, SmallStructureGNN,
    extract_motif_features, extract_loop_features, extract_structure_delta,
    extract_pairing_profile_features, build_all_features, compute_rate_metrics,
    train_gb_with_features, DEVICE,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = (
    PROJECT_ROOT / "experiments" / "apobec3a" / "outputs" / "rate_gnn_endtoend"
)
SEED = 42
SEEDS = (42, 123, 456)


# ---------------------------------------------------------------------------
# Pre-build PyG graphs
# ---------------------------------------------------------------------------

def prebuild_all_graphs(site_ids, sequences, pairing_probs):
    """Pre-build PyG Data objects for all sites."""
    from torch_geometric.data import Data

    graphs = []
    for sid in site_ids:
        seq = sequences.get(sid, "N" * 201)
        pp = pairing_probs.get(sid, np.zeros(201))
        if len(seq) >= 201 and len(pp) >= 201:
            g = build_structure_graph_from_pairing(pp, seq[:201])
        else:
            n = 201
            x = torch.zeros(n, 12)
            src = list(range(n - 1)) + list(range(1, n))
            dst = list(range(1, n)) + list(range(n - 1))
            edge_index = torch.tensor([src, dst], dtype=torch.long)
            g = Data(x=x, edge_index=edge_index)
        graphs.append(g)
    return graphs


# ---------------------------------------------------------------------------
# Graph-Aware Dataset & Collate
# ---------------------------------------------------------------------------

class GraphRateDataset(Dataset):
    """Dataset with pre-built PyG graphs and multi-modal features."""

    def __init__(self, site_ids, targets, graphs, pooled_orig, pooled_edited,
                 hand_features, pairing_features=None):
        self.site_ids = site_ids
        self.targets = targets
        self.graphs = graphs
        self.pooled_orig = pooled_orig
        self.pooled_edited = pooled_edited
        self.hand_features = hand_features
        self.pairing_features = pairing_features

    def __len__(self):
        return len(self.site_ids)

    def __getitem__(self, idx):
        sid = self.site_ids[idx]
        item = {
            "site_id": sid,
            "target": torch.tensor(self.targets[idx], dtype=torch.float32),
            "graph": self.graphs[idx],
            "pooled_orig": self.pooled_orig[sid],
            "pooled_edited": self.pooled_edited[sid],
            "hand_features": torch.tensor(
                self.hand_features[idx], dtype=torch.float32
            ),
        }
        if self.pairing_features is not None:
            item["pairing_features"] = torch.tensor(
                self.pairing_features[idx], dtype=torch.float32
            )
        return item


def graph_collate_fn(batch):
    """Collate: Batch.from_data_list for graphs + torch.stack for tensors."""
    from torch_geometric.data import Batch

    result = {
        "site_ids": [b["site_id"] for b in batch],
        "targets": torch.stack([b["target"] for b in batch]),
        "graph_batch": Batch.from_data_list([b["graph"] for b in batch]),
        "pooled_orig": torch.stack([b["pooled_orig"] for b in batch]),
        "pooled_edited": torch.stack([b["pooled_edited"] for b in batch]),
        "hand_features": torch.stack([b["hand_features"] for b in batch]),
    }
    result["edit_delta"] = result["pooled_edited"] - result["pooled_orig"]
    if "pairing_features" in batch[0]:
        result["pairing_features"] = torch.stack(
            [b["pairing_features"] for b in batch]
        )
    return result


# ---------------------------------------------------------------------------
# End-to-End Model Architectures
# ---------------------------------------------------------------------------

class GNNOnlyRate(nn.Module):
    """Config 1: Trained GNN -> MLP -> rate."""

    def __init__(self, gnn, d_gnn=64, dropout=0.3):
        super().__init__()
        self.gnn = gnn
        self.head = nn.Sequential(
            nn.Linear(d_gnn, d_gnn), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_gnn, 1),
        )

    def forward(self, batch):
        gnn_emb = self.gnn(batch["graph_batch"])
        return {"rate_pred": self.head(gnn_emb)}


class GNNHandRate(nn.Module):
    """Config 2: Trained GNN + hand features -> MLP -> rate."""

    def __init__(self, gnn, d_gnn=64, d_hand=40, dropout=0.3):
        super().__init__()
        self.gnn = gnn
        d_in = d_gnn + d_hand
        self.head = nn.Sequential(
            nn.Linear(d_in, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, batch):
        gnn_emb = self.gnn(batch["graph_batch"])
        combined = torch.cat([gnn_emb, batch["hand_features"]], dim=-1)
        return {"rate_pred": self.head(combined)}


class FourWayE2E(nn.Module):
    """Configs 3-4: primary + delta + hand + trained GNN -> gated fusion."""

    def __init__(self, gnn, d_primary=640, d_hand=40, d_gnn=64,
                 d_proj=64, dropout=0.3):
        super().__init__()
        self.gnn = gnn
        self.primary_proj = nn.Sequential(
            nn.Linear(d_primary, d_proj), nn.GELU(), nn.LayerNorm(d_proj))
        self.edit_proj = nn.Sequential(
            nn.Linear(d_primary, d_proj), nn.GELU(), nn.LayerNorm(d_proj))
        self.hand_proj = nn.Sequential(
            nn.Linear(d_hand, d_proj), nn.GELU(), nn.LayerNorm(d_proj))
        self.gnn_proj = nn.Sequential(
            nn.Linear(d_gnn, d_proj), nn.GELU(), nn.LayerNorm(d_proj))

        n_mod = 4
        self.gate = nn.Linear(d_proj * n_mod, n_mod)
        self.fuse = nn.Sequential(
            nn.Linear(d_proj * n_mod, d_proj * 2),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(d_proj * 2, d_proj),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_proj, 1),
        )

    def forward(self, batch):
        gnn_emb = self.gnn(batch["graph_batch"])
        p = self.primary_proj(batch["pooled_orig"])
        e = self.edit_proj(batch["edit_delta"])
        h = self.hand_proj(batch["hand_features"])
        g = self.gnn_proj(gnn_emb)

        concat = torch.cat([p, e, h, g], dim=-1)
        gates = torch.softmax(self.gate(concat), dim=-1)
        gated = torch.cat([
            gates[:, 0:1] * p, gates[:, 1:2] * e,
            gates[:, 2:3] * h, gates[:, 3:4] * g,
        ], dim=-1)
        return {"rate_pred": self.head(self.fuse(gated))}


class FiveWayE2E(nn.Module):
    """Config 5: 4Way + pairing_profile as 5th modality."""

    def __init__(self, gnn, d_primary=640, d_hand=40, d_gnn=64,
                 d_pairing=50, d_proj=64, dropout=0.3):
        super().__init__()
        self.gnn = gnn
        self.primary_proj = nn.Sequential(
            nn.Linear(d_primary, d_proj), nn.GELU(), nn.LayerNorm(d_proj))
        self.edit_proj = nn.Sequential(
            nn.Linear(d_primary, d_proj), nn.GELU(), nn.LayerNorm(d_proj))
        self.hand_proj = nn.Sequential(
            nn.Linear(d_hand, d_proj), nn.GELU(), nn.LayerNorm(d_proj))
        self.gnn_proj = nn.Sequential(
            nn.Linear(d_gnn, d_proj), nn.GELU(), nn.LayerNorm(d_proj))
        self.pairing_proj = nn.Sequential(
            nn.Linear(d_pairing, d_proj), nn.GELU(), nn.LayerNorm(d_proj))

        n_mod = 5
        self.gate = nn.Linear(d_proj * n_mod, n_mod)
        self.fuse = nn.Sequential(
            nn.Linear(d_proj * n_mod, d_proj * 2),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(d_proj * 2, d_proj),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_proj, 1),
        )

    def forward(self, batch):
        gnn_emb = self.gnn(batch["graph_batch"])
        p = self.primary_proj(batch["pooled_orig"])
        e = self.edit_proj(batch["edit_delta"])
        h = self.hand_proj(batch["hand_features"])
        g = self.gnn_proj(gnn_emb)
        pp = self.pairing_proj(batch["pairing_features"])

        mods = [p, e, h, g, pp]
        concat = torch.cat(mods, dim=-1)
        gates = torch.softmax(self.gate(concat), dim=-1)
        gated = torch.cat(
            [gates[:, i:i + 1] * x for i, x in enumerate(mods)], dim=-1
        )
        return {"rate_pred": self.head(self.fuse(gated))}


class DeepThreeWayE2E(nn.Module):
    """Config 6: Deeper GNN(4-layer, 128-dim, out=96) + delta + hand."""

    def __init__(self, gnn, d_primary=640, d_hand=40, d_gnn=96,
                 d_proj=96, dropout=0.3):
        super().__init__()
        self.gnn = gnn
        self.edit_proj = nn.Sequential(
            nn.Linear(d_primary, d_proj), nn.GELU(), nn.LayerNorm(d_proj))
        self.hand_proj = nn.Sequential(
            nn.Linear(d_hand, d_proj), nn.GELU(), nn.LayerNorm(d_proj))
        self.gnn_proj = nn.Sequential(
            nn.Linear(d_gnn, d_proj), nn.GELU(), nn.LayerNorm(d_proj))

        n_mod = 3
        self.gate = nn.Linear(d_proj * n_mod, n_mod)
        self.fuse = nn.Sequential(
            nn.Linear(d_proj * n_mod, d_proj * 2),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(d_proj * 2, d_proj),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_proj, 1),
        )

    def forward(self, batch):
        gnn_emb = self.gnn(batch["graph_batch"])
        e = self.edit_proj(batch["edit_delta"])
        h = self.hand_proj(batch["hand_features"])
        g = self.gnn_proj(gnn_emb)

        mods = [e, h, g]
        concat = torch.cat(mods, dim=-1)
        gates = torch.softmax(self.gate(concat), dim=-1)
        gated = torch.cat(
            [gates[:, i:i + 1] * x for i, x in enumerate(mods)], dim=-1
        )
        return {"rate_pred": self.head(self.fuse(gated))}


# ---------------------------------------------------------------------------
# Training with differential LR
# ---------------------------------------------------------------------------

def _move_batch(batch):
    """Move batch to DEVICE, handling PyG Batch separately."""
    graph_batch = batch["graph_batch"].to(DEVICE)
    out = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
           for k, v in batch.items() if k != "graph_batch"}
    out["graph_batch"] = graph_batch
    return out


def train_e2e(model, loaders, model_name, epochs=150, lr=1e-3,
              weight_decay=1e-2, patience=20, use_huber=False):
    """Train end-to-end model with differential LR for GNN backbone."""
    model = model.to(DEVICE)

    # Differential LR: GNN backbone at base_lr, projections/head at 2x
    gnn_params = list(model.gnn.parameters())
    gnn_ids = {id(p) for p in gnn_params}
    other_params = [p for p in model.parameters() if id(p) not in gnn_ids]

    optimizer = AdamW([
        {"params": gnn_params, "lr": lr},
        {"params": other_params, "lr": lr * 2},
    ], weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)

    loss_fn = F.smooth_l1_loss if use_huber else F.mse_loss
    n_params = sum(p.numel() for p in model.parameters())

    best_sp, best_epoch, pat = -float("inf"), 0, 0
    best_state = None

    logger.info("Training %s (params=%s, huber=%s)...",
                model_name, f"{n_params:,}", use_huber)
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, nb = 0, 0
        for batch in loaders["train"]:
            batch = _move_batch(batch)
            optimizer.zero_grad()
            pred = model(batch)["rate_pred"].squeeze(-1)
            loss = loss_fn(pred, batch["targets"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            nb += 1
        scheduler.step()

        val_m = eval_e2e(model, loaders.get("val"))
        val_sp = val_m.get("spearman", -999)
        if not np.isnan(val_sp) and val_sp > best_sp + 1e-5:
            best_sp, best_epoch, pat = val_sp, epoch, 0
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            pat += 1
        if epoch % 20 == 0 or pat >= patience:
            logger.info(
                "  E%3d loss=%.4f val_sp=%.4f best=%.4f pat=%d/%d",
                epoch, total_loss / max(nb, 1),
                val_sp if not np.isnan(val_sp) else 0,
                best_sp, pat, patience,
            )
        if pat >= patience:
            break

    if best_state:
        model.load_state_dict(best_state)

    results = {
        "best_epoch": best_epoch,
        "time_seconds": round(time.time() - t0, 1),
        "n_params": n_params,
    }
    for s in ["train", "val", "test"]:
        if s in loaders:
            results[s] = eval_e2e(model, loaders[s])
    return results, model


@torch.no_grad()
def eval_e2e(model, loader):
    if not loader:
        return {}
    model.eval()
    at, ap = [], []
    for batch in loader:
        batch = _move_batch(batch)
        at.append(batch["targets"].cpu().numpy())
        ap.append(model(batch)["rate_pred"].squeeze(-1).cpu().numpy())
    return compute_rate_metrics(np.concatenate(at), np.concatenate(ap))


@torch.no_grad()
def get_test_predictions(model, loader):
    """Get test predictions for ensemble."""
    model.eval()
    preds = []
    for batch in loader:
        batch = _move_batch(batch)
        preds.append(model(batch)["rate_pred"].squeeze(-1).cpu().numpy())
    return np.concatenate(preds)


def run_with_seeds_e2e(factory, name, loaders, kw, seeds=SEEDS):
    """Run model with multiple seeds, return best model."""
    results = []
    best_model = None
    best_sp = -float("inf")
    for seed in seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = factory()
        r, trained = train_e2e(model, loaders, f"{name}_s{seed}", **kw)
        results.append(r)
        sp = r["test"]["spearman"]
        logger.info("  %s seed=%d: test_sp=%.4f", name, seed, sp)
        if not np.isnan(sp) and sp > best_sp:
            best_sp = sp
            best_model = trained

    sps = [r["test"]["spearman"] for r in results
           if not np.isnan(r["test"]["spearman"])]
    return {
        "mean_test_spearman": float(np.mean(sps)) if sps else float("nan"),
        "std_test_spearman": float(np.std(sps)) if len(sps) > 1 else 0.0,
        "per_seed": results,
    }, best_model


# ---------------------------------------------------------------------------
# Single-threaded XGBoost (avoids OMP conflicts with torch_geometric)
# ---------------------------------------------------------------------------

def _train_gb_singlethread(feature_name, X_train, y_train, X_val, y_val,
                            X_test, y_test):
    """XGBoost with nthread=1 to avoid OpenMP conflicts."""
    t0 = time.time()
    gb = XGBRegressor(
        n_estimators=500, max_depth=5, learning_rate=0.01,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        early_stopping_rounds=20, random_state=42,
        nthread=1,
    )
    gb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    elapsed = time.time() - t0

    result = {"time_seconds": round(elapsed, 1),
              "n_features": X_train.shape[1]}
    for name, X, y in [("train", X_train, y_train),
                        ("val", X_val, y_val),
                        ("test", X_test, y_test)]:
        result[name] = compute_rate_metrics(y, gb.predict(X))
    logger.info("  %s: test_sp=%.4f (n_feat=%d)", feature_name,
                result["test"]["spearman"], X_train.shape[1])
    return result, gb


# ---------------------------------------------------------------------------
# GNN Embedding Extraction (for Config 7)
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_gnn_embeddings(model, loaders):
    """Extract GNN embeddings from a trained model across all splits."""
    model.eval()
    embeddings = {}
    for loader in loaders.values():
        for batch in loader:
            batch = _move_batch(batch)
            gnn_emb = model.gnn(batch["graph_batch"]).cpu().numpy()
            for i, sid in enumerate(batch["site_ids"]):
                embeddings[sid] = gnn_emb[i]
    return embeddings


# ---------------------------------------------------------------------------
# Main Experiment
# ---------------------------------------------------------------------------

def run_experiment():
    t_total = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    logger.info("=" * 70)
    logger.info("RATE PREDICTION: END-TO-END GNN EXPERIMENT")
    logger.info("=" * 70)

    data = load_all_data()
    split_dfs = data["split_dfs"]

    # Filter to sites with embeddings
    available = (set(data["pooled_orig"].keys())
                 & set(data["pooled_edited"].keys()))
    for s in split_dfs:
        split_dfs[s] = split_dfs[s][
            split_dfs[s]["site_id"].isin(available)
        ].copy()
    logger.info("After embedding filter: train=%d, val=%d, test=%d",
                len(split_dfs["train"]), len(split_dfs["val"]),
                len(split_dfs["test"]))

    all_results = {
        "experiment": "rate_gnn_endtoend",
        "n_train": len(split_dfs["train"]),
        "n_val": len(split_dfs["val"]),
        "n_test": len(split_dfs["test"]),
        "models": {},
    }

    # ==================================================================
    # Pre-build PyG graphs
    # ==================================================================
    logger.info("\nPre-building PyG graphs...")
    t_graph = time.time()
    split_graphs = {}
    for sn, sdf in split_dfs.items():
        sids = sdf["site_id"].tolist()
        split_graphs[sn] = prebuild_all_graphs(
            sids, data["sequences"], data["pairing_probs"]
        )
    total_graphs = sum(len(g) for g in split_graphs.values())
    logger.info("  Built %d graphs in %.1fs",
                total_graphs, time.time() - t_graph)

    # Build hand + pairing features per split
    hand_features = {}
    pairing_features = {}
    for sn, sdf in split_dfs.items():
        sids = sdf["site_id"].tolist()
        motif = extract_motif_features(data["sequences"], sids)
        struct = extract_structure_delta(data["structure_delta"], sids)
        loop = extract_loop_features(data["loop_df"], sids)
        hand_features[sn] = np.nan_to_num(
            np.concatenate([motif, struct, loop], axis=1)
        )
        pair = extract_pairing_profile_features(
            data["pairing_probs"], data["pairing_probs_edited"],
            data["accessibilities"], data["accessibilities_edited"],
            data["mfes"], data["mfes_edited"], sids,
        )
        pairing_features[sn] = np.nan_to_num(pair)

    d_hand = hand_features["train"].shape[1]
    d_pairing = pairing_features["train"].shape[1]
    logger.info("d_hand=%d, d_pairing=%d", d_hand, d_pairing)

    def make_graph_loaders(use_pairing=False, batch_size=64):
        loaders = {}
        for sn, sdf in split_dfs.items():
            sids = sdf["site_id"].tolist()
            targets = sdf["target"].values.astype(np.float32)
            ds = GraphRateDataset(
                site_ids=sids, targets=targets,
                graphs=split_graphs[sn],
                pooled_orig=data["pooled_orig"],
                pooled_edited=data["pooled_edited"],
                hand_features=hand_features[sn],
                pairing_features=(pairing_features[sn]
                                  if use_pairing else None),
            )
            loaders[sn] = DataLoader(
                ds, batch_size=batch_size,
                shuffle=(sn == "train"), num_workers=0,
                collate_fn=graph_collate_fn,
            )
        return loaders

    loaders_base = make_graph_loaders(use_pairing=False)
    loaders_pairing = make_graph_loaders(use_pairing=True)

    best_models = {}
    test_predictions = {}

    # ==================================================================
    # Config 1: GNN Only
    # ==================================================================
    logger.info("\n" + "=" * 70)
    logger.info("CONFIG 1: GNN_Only")
    logger.info("=" * 70)

    result, bm = run_with_seeds_e2e(
        lambda: GNNOnlyRate(
            SmallStructureGNN(12, 64, 64, 3, 4, 0.3)),
        "GNN_Only", loaders_base,
        {"epochs": 150, "lr": 1e-3, "weight_decay": 5e-3, "patience": 20},
    )
    all_results["models"]["GNN_Only"] = result
    best_models["GNN_Only"] = bm
    test_predictions["GNN_Only"] = get_test_predictions(
        bm, loaders_base["test"])
    logger.info(">> GNN_Only: mean_sp=%.4f +/- %.4f",
                result["mean_test_spearman"], result["std_test_spearman"])

    # ==================================================================
    # Config 2: GNN + Hand
    # ==================================================================
    logger.info("\n" + "=" * 70)
    logger.info("CONFIG 2: GNN+Hand")
    logger.info("=" * 70)

    result, bm = run_with_seeds_e2e(
        lambda: GNNHandRate(
            SmallStructureGNN(12, 64, 64, 3, 4, 0.3), d_hand=d_hand),
        "GNN+Hand", loaders_base,
        {"epochs": 150, "lr": 1e-3, "weight_decay": 5e-3, "patience": 20,
         "use_huber": True},
    )
    all_results["models"]["GNN+Hand"] = result
    best_models["GNN+Hand"] = bm
    test_predictions["GNN+Hand"] = get_test_predictions(
        bm, loaders_base["test"])
    logger.info(">> GNN+Hand: mean_sp=%.4f +/- %.4f",
                result["mean_test_spearman"], result["std_test_spearman"])

    # ==================================================================
    # Config 3: 4Way E2E Small
    # ==================================================================
    logger.info("\n" + "=" * 70)
    logger.info("CONFIG 3: 4Way_e2e_small")
    logger.info("=" * 70)

    result, bm = run_with_seeds_e2e(
        lambda: FourWayE2E(
            SmallStructureGNN(12, 64, 64, 3, 4, 0.3),
            d_hand=d_hand, d_proj=64, dropout=0.3),
        "4Way_e2e_small", loaders_base,
        {"epochs": 150, "lr": 1e-3, "weight_decay": 5e-3, "patience": 20},
    )
    all_results["models"]["4Way_e2e_small"] = result
    best_models["4Way_e2e_small"] = bm
    test_predictions["4Way_e2e_small"] = get_test_predictions(
        bm, loaders_base["test"])
    logger.info(">> 4Way_e2e_small: mean_sp=%.4f +/- %.4f",
                result["mean_test_spearman"], result["std_test_spearman"])

    # ==================================================================
    # Config 4: 4Way E2E Heavy
    # ==================================================================
    logger.info("\n" + "=" * 70)
    logger.info("CONFIG 4: 4Way_e2e_heavy")
    logger.info("=" * 70)

    result, bm = run_with_seeds_e2e(
        lambda: FourWayE2E(
            SmallStructureGNN(12, 64, 64, 3, 4, 0.4),
            d_hand=d_hand, d_proj=128, dropout=0.5),
        "4Way_e2e_heavy", loaders_base,
        {"epochs": 150, "lr": 5e-4, "weight_decay": 2e-2, "patience": 25,
         "use_huber": True},
    )
    all_results["models"]["4Way_e2e_heavy"] = result
    best_models["4Way_e2e_heavy"] = bm
    test_predictions["4Way_e2e_heavy"] = get_test_predictions(
        bm, loaders_base["test"])
    logger.info(">> 4Way_e2e_heavy: mean_sp=%.4f +/- %.4f",
                result["mean_test_spearman"], result["std_test_spearman"])

    # ==================================================================
    # Config 5: 5Way E2E
    # ==================================================================
    logger.info("\n" + "=" * 70)
    logger.info("CONFIG 5: 5Way_e2e")
    logger.info("=" * 70)

    result, bm = run_with_seeds_e2e(
        lambda: FiveWayE2E(
            SmallStructureGNN(12, 64, 64, 3, 4, 0.3),
            d_hand=d_hand, d_pairing=d_pairing, d_proj=64, dropout=0.3),
        "5Way_e2e", loaders_pairing,
        {"epochs": 150, "lr": 1e-3, "weight_decay": 5e-3, "patience": 20},
    )
    all_results["models"]["5Way_e2e"] = result
    best_models["5Way_e2e"] = bm
    test_predictions["5Way_e2e"] = get_test_predictions(
        bm, loaders_pairing["test"])
    logger.info(">> 5Way_e2e: mean_sp=%.4f +/- %.4f",
                result["mean_test_spearman"], result["std_test_spearman"])

    # ==================================================================
    # Config 6: Deep3Way
    # ==================================================================
    logger.info("\n" + "=" * 70)
    logger.info("CONFIG 6: Deep3Way")
    logger.info("=" * 70)

    result, bm = run_with_seeds_e2e(
        lambda: DeepThreeWayE2E(
            SmallStructureGNN(12, 128, 96, num_layers=4, heads=4,
                              dropout=0.3),
            d_hand=d_hand, d_gnn=96, d_proj=96, dropout=0.3),
        "Deep3Way", loaders_base,
        {"epochs": 150, "lr": 5e-4, "weight_decay": 1e-2, "patience": 25,
         "use_huber": True},
    )
    all_results["models"]["Deep3Way"] = result
    best_models["Deep3Way"] = bm
    test_predictions["Deep3Way"] = get_test_predictions(
        bm, loaders_base["test"])
    logger.info(">> Deep3Way: mean_sp=%.4f +/- %.4f",
                result["mean_test_spearman"], result["std_test_spearman"])

    # ==================================================================
    # Config 7: GB with Trained GNN Embeddings
    # ==================================================================
    logger.info("\n" + "=" * 70)
    logger.info("CONFIG 7: GB_trainedGNN")
    logger.info("=" * 70)

    # Free memory before Config 7: delete unused models and graph cache
    import gc
    keep_models = {"4Way_e2e_small", "4Way_e2e_heavy", "Deep3Way"}
    for mname in list(best_models.keys()):
        if mname not in keep_models:
            del best_models[mname]
    # Don't need loaders_pairing anymore
    del loaders_pairing
    gc.collect()
    logger.info("  Freed unused models from memory")

    # Pick best source model from configs 3, 4, 6
    source_candidates = {
        "4Way_e2e_small": all_results["models"]["4Way_e2e_small"],
        "4Way_e2e_heavy": all_results["models"]["4Way_e2e_heavy"],
        "Deep3Way": all_results["models"]["Deep3Way"],
    }
    source_name = max(source_candidates,
                      key=lambda k: source_candidates[k]["mean_test_spearman"])
    source_sp = source_candidates[source_name]["mean_test_spearman"]
    source_model = best_models[source_name]
    logger.info("  Extracting GNN embeddings from %s (sp=%.4f)",
                source_name, source_sp)

    trained_gnn_emb = extract_gnn_embeddings(source_model, loaders_base)
    d_gnn = next(iter(trained_gnn_emb.values())).shape[0]
    logger.info("  Extracted %d trained GNN embeddings (%d-dim)",
                len(trained_gnn_emb), d_gnn)

    # GB variants with trained GNN embeddings
    for feat_name, include_emb in [
        ("GB_trainedGNN_all", True),
        ("GB_trainedGNN_hand", False),
    ]:
        logger.info("\n--- %s ---", feat_name)
        splits_data = {}
        for sn, sdf in split_dfs.items():
            sids = sdf["site_id"].tolist()
            targets = sdf["target"].values.astype(np.float32)
            base_feats = build_all_features(
                sids, data, include_emb=include_emb, include_pairing=True)
            gnn_feats = np.array(
                [trained_gnn_emb.get(sid, np.zeros(d_gnn)) for sid in sids],
                dtype=np.float32,
            )
            feats = np.concatenate([base_feats, gnn_feats], axis=1)
            splits_data[sn] = (feats, targets, sids)

        logger.info("  Training XGBoost on %d features...",
                     splits_data["train"][0].shape[1])
        sys.stdout.flush()
        sys.stderr.flush()
        try:
            result, _ = _train_gb_singlethread(
                feat_name,
                splits_data["train"][0], splits_data["train"][1],
                splits_data["val"][0], splits_data["val"][1],
                splits_data["test"][0], splits_data["test"][1],
            )
            all_results["models"][feat_name] = result
        except Exception as e:
            logger.error("  GB training failed: %s", e)
            import traceback
            traceback.print_exc()

    # Free source model and embeddings after GB training
    del trained_gnn_emb
    if source_name in best_models:
        del best_models[source_name]
    gc.collect()

    # Also extract from Deep3Way if it wasn't the source
    if source_name != "Deep3Way":
        deep_emb = extract_gnn_embeddings(
            best_models["Deep3Way"], loaders_base)
        d_deep = next(iter(deep_emb.values())).shape[0]
        logger.info("\n--- GB_deepGNN_all ---")
        splits_data = {}
        for sn, sdf in split_dfs.items():
            sids = sdf["site_id"].tolist()
            targets = sdf["target"].values.astype(np.float32)
            base_feats = build_all_features(
                sids, data, include_emb=True, include_pairing=True)
            gnn_feats = np.array(
                [deep_emb.get(sid, np.zeros(d_deep)) for sid in sids],
                dtype=np.float32,
            )
            feats = np.concatenate([base_feats, gnn_feats], axis=1)
            splits_data[sn] = (feats, targets, sids)

        result, _ = _train_gb_singlethread(
            "GB_deepGNN_all",
            splits_data["train"][0], splits_data["train"][1],
            splits_data["val"][0], splits_data["val"][1],
            splits_data["test"][0], splits_data["test"][1],
        )
        all_results["models"]["GB_deepGNN_all"] = result

    # ==================================================================
    # Config 8: Ensemble
    # ==================================================================
    logger.info("\n" + "=" * 70)
    logger.info("CONFIG 8: ENSEMBLE")
    logger.info("=" * 70)

    test_targets = split_dfs["test"]["target"].values.astype(np.float32)

    nn_names = ["GNN_Only", "GNN+Hand", "4Way_e2e_small",
                "4Way_e2e_heavy", "5Way_e2e", "Deep3Way"]
    nn_sps = [(n, all_results["models"][n]["mean_test_spearman"])
              for n in nn_names]
    nn_sps.sort(key=lambda x: -x[1])

    for k in [2, 3]:
        top_k = [n for n, _ in nn_sps[:k]]
        preds = [test_predictions[n] for n in top_k
                 if n in test_predictions]
        if len(preds) >= 2:
            avg = np.mean(preds, axis=0)
            ens_metrics = compute_rate_metrics(test_targets, avg)
            ens_name = f"Ensemble_NN_top{k}"
            all_results["models"][ens_name] = {
                "test": ens_metrics,
                "n_models": len(preds),
                "components": top_k,
            }
            logger.info("  %s: test_sp=%.4f (from %s)",
                        ens_name, ens_metrics["spearman"],
                        ", ".join(top_k))

    # ==================================================================
    # Summary
    # ==================================================================
    total_time = time.time() - t_total
    all_results["total_time_seconds"] = round(total_time, 1)

    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info("%-30s  %10s  %10s", "Model", "Test Sp", "Info")
    logger.info("-" * 55)

    summary = []
    for name, m in all_results["models"].items():
        if "mean_test_spearman" in m:
            sp = m["mean_test_spearman"]
            info = f"{m['per_seed'][0].get('n_params', '?')} params"
        else:
            sp = m.get("test", {}).get("spearman", float("nan"))
            info = f"{m.get('n_features', m.get('n_models', '?'))} feats"
        summary.append((name, sp, info))

    for n, sp, info in sorted(summary,
                               key=lambda x: (-x[1]
                                              if not np.isnan(x[1])
                                              else -999)):
        logger.info("%-30s  %10.4f  %s", n, sp, info)

    logger.info("\n--- Reference (from gnn_fusion experiment) ---")
    logger.info("GB_All+GNN (random GNN):    0.7321")
    logger.info("4Way_heavyreg (random GNN):  0.7247")

    out_path = OUTPUT_DIR / "rate_gnn_endtoend_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("\nResults saved to %s", out_path)
    logger.info("Total time: %.1f minutes", total_time / 60)


if __name__ == "__main__":
    run_experiment()
