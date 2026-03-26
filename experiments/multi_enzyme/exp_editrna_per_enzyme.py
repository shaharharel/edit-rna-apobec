#!/usr/bin/env python
"""Train EditRNA and SubtractionMLP per enzyme using cached RNA-FM embeddings.

For each enzyme category, trains:
1. EditRNA-style: edit_embedding (token diff + pooled) → MLP → binary
2. SubtractionMLP: pooled_edited - pooled_original → MLP → binary
3. PooledMLP: pooled_original only → MLP → binary
4. EditRNA+Hand: edit embedding + 40-dim hand features → MLP → binary

All use pre-computed RNA-FM 640-dim pooled embeddings from v3.

Usage:
    conda run -n quris python experiments/multi_enzyme/exp_editrna_per_enzyme.py
"""

import gc
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.apobec_feature_extraction import (
    extract_motif_features, extract_loop_features,
    extract_structure_delta_features,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_ME_DIR = PROJECT_ROOT / "data/processed/multi_enzyme"
SPLITS_CSV = _ME_DIR / "splits_multi_enzyme_v3_with_negatives.csv"
EMB_DIR = _ME_DIR / "embeddings"
EMB_POOLED = EMB_DIR / "rnafm_pooled_v3.pt"
EMB_POOLED_ED = EMB_DIR / "rnafm_pooled_edited_v3.pt"
STRUCT_CACHE = _ME_DIR / "structure_cache_multi_enzyme_v3.npz"
LOOP_CSV = _ME_DIR / "loop_position_per_site_v3.csv"
SEQS_JSON = _ME_DIR / "multi_enzyme_sequences_v3_with_negatives.json"
OUTPUT_DIR = Path(__file__).parent / "outputs" / "editrna_per_enzyme"

SEED = 42
N_FOLDS = 5
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
D_EMB = 640


# ============================================================================
# Models
# ============================================================================

class SubtractionMLP(nn.Module):
    """pooled_edited - pooled_original → MLP."""
    def __init__(self, d_in=D_EMB, hidden=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, pooled_orig, pooled_edited, hand_feat=None):
        diff = pooled_edited - pooled_orig
        return self.net(diff).squeeze(-1)


class PooledMLP(nn.Module):
    """pooled_original only → MLP (no edit info)."""
    def __init__(self, d_in=D_EMB, hidden=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, pooled_orig, pooled_edited=None, hand_feat=None):
        return self.net(pooled_orig).squeeze(-1)


class EditEmbeddingMLP(nn.Module):
    """Edit embedding: concat(pooled_orig, pooled_edit-pooled_orig) → MLP.
    Simplified version of EditRNA without cross-attention (uses pooled, not token-level)."""
    def __init__(self, d_in=D_EMB, hidden=256, dropout=0.3):
        super().__init__()
        # Edit embedding: local diff + original context
        self.edit_proj = nn.Sequential(
            nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout),
        )
        self.ctx_proj = nn.Sequential(
            nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, pooled_orig, pooled_edited, hand_feat=None):
        edit_emb = self.edit_proj(pooled_edited - pooled_orig)
        ctx_emb = self.ctx_proj(pooled_orig)
        combined = torch.cat([edit_emb, ctx_emb], dim=-1)
        return self.head(combined).squeeze(-1)


class EditEmbeddingPlusHand(nn.Module):
    """Edit embedding + 40-dim hand features (motif+structure+loop)."""
    def __init__(self, d_emb=D_EMB, d_hand=40, hidden=256, dropout=0.3):
        super().__init__()
        self.edit_proj = nn.Sequential(
            nn.Linear(d_emb, hidden), nn.GELU(), nn.Dropout(dropout),
        )
        self.ctx_proj = nn.Sequential(
            nn.Linear(d_emb, hidden), nn.GELU(), nn.Dropout(dropout),
        )
        self.hand_proj = nn.Sequential(
            nn.Linear(d_hand, 64), nn.GELU(), nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 2 + 64, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, pooled_orig, pooled_edited, hand_feat):
        edit_emb = self.edit_proj(pooled_edited - pooled_orig)
        ctx_emb = self.ctx_proj(pooled_orig)
        hand_emb = self.hand_proj(hand_feat)
        combined = torch.cat([edit_emb, ctx_emb, hand_emb], dim=-1)
        return self.head(combined).squeeze(-1)


# ============================================================================
# Training
# ============================================================================

def run_cv(model_cls, data_dict, y, enzyme, model_name, epochs=60, lr=1e-3, bs=64):
    """5-fold CV for embedding-based model."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_aurocs, fold_auprcs = [], []

    for fold_i, (tr, te) in enumerate(skf.split(data_dict["hand"], y)):
        torch.manual_seed(SEED + fold_i)

        model = model_cls().to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        crit = nn.BCEWithLogitsLoss()

        # Build tensors
        keys = ["orig", "edited", "hand"]
        train_tensors = [torch.FloatTensor(data_dict[k][tr]) for k in keys] + [torch.FloatTensor(y[tr])]
        test_tensors = [torch.FloatTensor(data_dict[k][te]) for k in keys] + [torch.FloatTensor(y[te])]

        train_ds = TensorDataset(*train_tensors)
        test_ds = TensorDataset(*test_tensors)
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=bs * 2)

        best_auroc = 0
        patience = 0
        for epoch in range(epochs):
            model.train()
            for batch in train_loader:
                orig, edited, hand, yt = [b.to(DEVICE) for b in batch]
                logits = model(orig, edited, hand)
                loss = crit(logits, yt)
                opt.zero_grad(); loss.backward(); opt.step()
            sched.step()

            # Eval
            model.eval()
            all_ys = []
            with torch.no_grad():
                for batch in test_loader:
                    orig, edited, hand, _ = [b.to(DEVICE) for b in batch]
                    all_ys.extend(torch.sigmoid(model(orig, edited, hand)).cpu().tolist())
            auroc = roc_auc_score(y[te], all_ys)
            if auroc > best_auroc:
                best_auroc = auroc
                patience = 0
            else:
                patience += 1
            if patience >= 12:
                break

        # Final eval
        model.eval()
        all_ys = []
        with torch.no_grad():
            for batch in test_loader:
                orig, edited, hand, _ = [b.to(DEVICE) for b in batch]
                all_ys.extend(torch.sigmoid(model(orig, edited, hand)).cpu().tolist())

        auroc = roc_auc_score(y[te], all_ys)
        auprc = average_precision_score(y[te], all_ys)
        fold_aurocs.append(auroc)
        fold_auprcs.append(auprc)
        logger.info("  %s fold %d: AUROC=%.3f", model_name, fold_i + 1, auroc)

        del model; gc.collect()
        if DEVICE == "mps":
            torch.mps.empty_cache()

    mean_auroc = np.mean(fold_aurocs)
    logger.info("  %s mean: AUROC=%.3f±%.3f", model_name, mean_auroc, np.std(fold_aurocs))
    return {
        "auroc": float(mean_auroc), "auroc_std": float(np.std(fold_aurocs)),
        "auprc": float(np.mean(fold_auprcs)),
        "fold_aurocs": [float(x) for x in fold_aurocs],
    }


def main():
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Device: %s", DEVICE)

    # Check embeddings exist
    if not EMB_POOLED.exists():
        logger.error("RNA-FM embeddings not found at %s. Run generate_rnafm_embeddings_v3.py first.", EMB_POOLED)
        return

    # Load embeddings (dict format)
    logger.info("Loading RNA-FM embeddings...")
    emb_orig = torch.load(EMB_POOLED, map_location="cpu", weights_only=False)
    emb_edited = torch.load(EMB_POOLED_ED, map_location="cpu", weights_only=False)
    logger.info("Embeddings: %d orig, %d edited", len(emb_orig), len(emb_edited))

    # Load splits and hand features
    df = pd.read_csv(SPLITS_CSV)
    with open(SEQS_JSON) as f:
        seqs = json.load(f)

    struct_delta = {}
    if STRUCT_CACHE.exists():
        data = np.load(str(STRUCT_CACHE), allow_pickle=True)
        for i, sid in enumerate(data["site_ids"].astype(str)):
            struct_delta[sid] = data["delta_features"][i]
        del data; gc.collect()

    loop_df = pd.read_csv(LOOP_CSV)
    loop_df["site_id"] = loop_df["site_id"].astype(str)
    loop_df = loop_df.set_index("site_id")

    results = {}

    for enzyme in ["A3A", "A3B", "A3G", "A3A_A3G", "Neither"]:
        logger.info("\n=== %s ===", enzyme)
        mask = df["enzyme"] == enzyme
        sub = df[mask].copy()
        if sub["is_edited"].nunique() < 2:
            continue

        site_ids = sub["site_id"].values
        y = sub["is_edited"].values.astype(int)

        # Check embedding coverage
        covered = [str(sid) in emb_orig and str(sid) in emb_edited for sid in site_ids]
        n_covered = sum(covered)
        if n_covered < len(site_ids) * 0.9:
            logger.warning("  Only %d/%d sites have embeddings, skipping", n_covered, len(site_ids))
            continue

        # Filter to sites with embeddings
        covered_mask = np.array(covered)
        site_ids = site_ids[covered_mask]
        y = y[covered_mask]
        logger.info("  %d sites with embeddings (pos=%d, neg=%d)",
                     len(y), (y == 1).sum(), (y == 0).sum())

        # Build feature arrays
        orig_embs = np.array([emb_orig[str(sid)].numpy() for sid in site_ids])
        edited_embs = np.array([emb_edited[str(sid)].numpy() for sid in site_ids])

        motif = extract_motif_features(seqs, list(site_ids))
        struct = extract_structure_delta_features(struct_delta, list(site_ids))
        loop = extract_loop_features(loop_df, list(site_ids))
        hand_40 = np.nan_to_num(np.concatenate([motif, struct, loop], axis=1), nan=0.0)

        data_dict = {"orig": orig_embs, "edited": edited_embs, "hand": hand_40}

        enzyme_results = {}

        # 1. SubtractionMLP
        enzyme_results["SubtractionMLP"] = run_cv(
            lambda: SubtractionMLP(D_EMB, 256, 0.3),
            data_dict, y, enzyme, "SubtractionMLP", epochs=60, lr=1e-3,
        )

        # 2. PooledMLP
        enzyme_results["PooledMLP"] = run_cv(
            lambda: PooledMLP(D_EMB, 256, 0.3),
            data_dict, y, enzyme, "PooledMLP", epochs=60, lr=1e-3,
        )

        # 3. EditEmbeddingMLP (simplified EditRNA)
        enzyme_results["EditRNA"] = run_cv(
            lambda: EditEmbeddingMLP(D_EMB, 256, 0.3),
            data_dict, y, enzyme, "EditRNA", epochs=60, lr=1e-3,
        )

        # 4. EditRNA + Hand Features
        enzyme_results["EditRNA+Hand"] = run_cv(
            lambda: EditEmbeddingPlusHand(D_EMB, 40, 256, 0.3),
            data_dict, y, enzyme, "EditRNA+Hand", epochs=60, lr=5e-4,
        )

        results[enzyme] = enzyme_results

    # Save
    with open(OUTPUT_DIR / "editrna_per_enzyme_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Summary
    print("\n" + "=" * 80)
    print("EDITRNA PER-ENZYME RESULTS")
    print("=" * 80)
    print(f"\n{'Enzyme':>10} {'SubMLP':>10} {'PooledMLP':>10} {'EditRNA':>10} {'EditRNA+H':>10}")
    for enz, r in results.items():
        row = f"{enz:>10}"
        for m in ["SubtractionMLP", "PooledMLP", "EditRNA", "EditRNA+Hand"]:
            auroc = r.get(m, {}).get("auroc", 0)
            row += f"{auroc:10.3f}"
        print(row)

    logger.info("\nTotal time: %.0fs", time.time() - t0)


if __name__ == "__main__":
    main()
