#!/usr/bin/env python3
"""Train Phase3 model in MFE-only feature regime (all 7 struct_delta slots zeroed).

This model is consistent with an inference pipeline that has access to only the
MFE structure string (the 9-d canonical loop features) plus the 24-d motif, but
NOT to partition-function-derived struct_delta features. Training with those
7 slots held at zero prevents a train-inference mismatch.

Why MFE-only?
-------------
Phase 2 extends panel scoring to promoter + 3'UTR (~17 M more positions). Full
canonical Vienna (MFE + pf + bpp) takes ~10 days on ai-chem; MFE-only is ~6x
faster (overnight). Keeping the feature regime consistent now avoids re-folding.

Feature layout (1320-d input):
    [0:640]        RNA-FM pooled original
    [640:1280]     RNA-FM edit delta
    [1280:1304]    motif (24-d)           <- fresh
    [1304:1311]    struct_delta (7-d)     <- ZEROED (MFE-only regime)
    [1311:1320]    loop geometry (9-d)    <- canonical from MFE dot-bracket

Output:
    experiments/multi_enzyme/outputs/phase3_mfe_only/
        phase3_mfe_only.pt
        cv_results.json
        comparison_vs_phase3.md
        neural_fold{0..4}.pt
        run.log

Usage:
    conda run -n quris python experiments/multi_enzyme/exp_train_phase3_mfe_only.py
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
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.apobec_feature_extraction import build_hand_features  # noqa: E402

# --------------------------------------------------------------------------- #
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "multi_enzyme" / "outputs" / "phase3_mfe_only"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

log_file = OUTPUT_DIR / "run.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, mode="w")],
)
logger = logging.getLogger(__name__)

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else torch.device("cpu")
)
logger.info("Device: %s", DEVICE)

SEED = 42
N_FOLDS = 5
BATCH_SIZE = 64
ENZYMES = ["A3A", "A3B", "A3G", "A3A_A3G", "Neither"]
ENZYME_CLASSES = ["A3A", "A3B", "A3G", "A3A_A3G", "Neither", "Unknown"]
ENZYME_TO_IDX = {e: i for i, e in enumerate(ENZYME_CLASSES)}
N_ENZYMES = len(ENZYME_CLASSES)
N_PER_ENZYME = len(ENZYMES)

D_RNAFM = 640
D_EDIT_DELTA = 640
D_HAND = 40
D_INPUT = D_RNAFM + D_EDIT_DELTA + D_HAND  # 1320
D_SHARED = 128

# Absolute index slice of struct_delta in the 1320-d input.
# Hand block layout = [motif(24), struct_delta(7), loop(9)]
# Hand block starts at dim 1280 -> struct_delta occupies [1304, 1311).
STRUCT_DELTA_START = D_RNAFM + D_EDIT_DELTA + 24        # 1304
STRUCT_DELTA_END = STRUCT_DELTA_START + 7               # 1311

STAGE1_EPOCHS = 10
STAGE1_LR = 1e-3
STAGE2_EPOCHS = 20
STAGE2_LR = 5e-4
WEIGHT_DECAY = 1e-4
ENZYME_HEAD_WEIGHT = 0.3
ENZYME_CLS_WEIGHT = 0.1

DATA_DIR = PROJECT_ROOT / "data"
EMB_DIR = DATA_DIR / "processed" / "multi_enzyme" / "embeddings"
MULTI_SPLITS = DATA_DIR / "processed" / "multi_enzyme" / "splits_multi_enzyme_v3_with_negatives.csv"
MULTI_SEQS = DATA_DIR / "processed" / "multi_enzyme" / "multi_enzyme_sequences_v3_with_negatives.json"
LOOP_CSV = DATA_DIR / "processed" / "multi_enzyme" / "loop_position_per_site_v3.csv"
STRUCT_CACHE = DATA_DIR / "processed" / "multi_enzyme" / "structure_cache_multi_enzyme_v3.npz"


class Phase3Model(nn.Module):
    """Identical architecture to exp_phase3_neural_true_validation — same state_dict shape."""

    def __init__(self):
        super().__init__()
        self.shared_encoder = nn.Sequential(
            nn.Linear(D_INPUT, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.LayerNorm(256),
            nn.Linear(256, D_SHARED),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.binary_head = nn.Linear(D_SHARED, 1)
        self.enzyme_adapters = nn.ModuleDict({
            enz: nn.Sequential(nn.Linear(D_SHARED, 32), nn.GELU(), nn.Linear(32, 1))
            for enz in ENZYMES
        })
        self.enzyme_classifier = nn.Sequential(
            nn.Linear(D_SHARED, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, N_ENZYMES),
        )

    def forward(self, x):
        shared = self.shared_encoder(x)
        binary_logit = self.binary_head(shared).squeeze(-1)
        per_enzyme_logits = [self.enzyme_adapters[e](shared).squeeze(-1) for e in ENZYMES]
        enzyme_cls_logits = self.enzyme_classifier(shared)
        return binary_logit, per_enzyme_logits, enzyme_cls_logits, shared


class SimpleDataset(Dataset):
    def __init__(self, X, lb, le, pel):
        self.X = torch.from_numpy(X).float()
        self.lb = torch.from_numpy(lb).float()
        self.le = torch.from_numpy(le).long()
        self.pel = torch.from_numpy(pel).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            "x": self.X[idx],
            "label_binary": self.lb[idx],
            "label_enzyme": self.le[idx],
            "per_enzyme_labels": self.pel[idx],
        }


def compute_loss(binary_logit, per_enzyme_logits, enzyme_cls_logits, batch):
    device = binary_logit.device
    loss_binary = F.binary_cross_entropy_with_logits(binary_logit, batch["label_binary"])

    pel = batch["per_enzyme_labels"]
    loss_pe = torch.tensor(0.0, device=device)
    n_active = 0
    for h in range(N_PER_ENZYME):
        mask = pel[:, h] >= 0
        if mask.sum() < 2:
            continue
        hl = per_enzyme_logits[h][mask]
        lbl = pel[:, h][mask]
        loss_pe = loss_pe + F.binary_cross_entropy_with_logits(hl, lbl)
        n_active += 1
    if n_active > 0:
        loss_pe = loss_pe / n_active

    pos_mask = batch["label_binary"] == 1
    loss_cls = torch.tensor(0.0, device=device)
    if pos_mask.sum() > 0:
        loss_cls = F.cross_entropy(enzyme_cls_logits[pos_mask], batch["label_enzyme"][pos_mask])

    return loss_binary + ENZYME_HEAD_WEIGHT * loss_pe + ENZYME_CLS_WEIGHT * loss_cls


def train_epoch(model, loader, optim):
    model.train()
    total = 0.0
    n = 0
    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        optim.zero_grad()
        bl, pel, ecl, _ = model(batch["x"])
        loss = compute_loss(bl, pel, ecl, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        total += loss.item()
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_bp = []
    all_pep = [[] for _ in range(N_PER_ENZYME)]
    all_lb = []
    all_le = []
    all_pel = []
    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        bl, pel, _, _ = model(batch["x"])
        all_bp.append(torch.sigmoid(bl).cpu().numpy())
        for h in range(N_PER_ENZYME):
            all_pep[h].append(torch.sigmoid(pel[h]).cpu().numpy())
        all_lb.append(batch["label_binary"].cpu().numpy())
        all_le.append(batch["label_enzyme"].cpu().numpy())
        all_pel.append(batch["per_enzyme_labels"].cpu().numpy())
    bp = np.concatenate(all_bp)
    pep = [np.concatenate(p) for p in all_pep]
    lb = np.concatenate(all_lb)
    pel_arr = np.concatenate(all_pel)
    try:
        overall = float(roc_auc_score(lb, bp))
    except ValueError:
        overall = 0.5
    per_enz = {}
    for h, e in enumerate(ENZYMES):
        mask = pel_arr[:, h] >= 0
        labels = pel_arr[:, h][mask]
        probs = pep[h][mask]
        if len(np.unique(labels)) < 2:
            continue
        try:
            per_enz[e] = float(roc_auc_score(labels, probs))
        except ValueError:
            pass
    return {"overall_auroc": overall, "per_enzyme_auroc": per_enz}


def train_two_stage(model, train_ds, val_ds=None):
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0) if val_ds else None

    best_auroc = 0.0
    best_state = None

    optim1 = torch.optim.AdamW(model.parameters(), lr=STAGE1_LR, weight_decay=WEIGHT_DECAY)
    sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(optim1, T_max=STAGE1_EPOCHS)
    for epoch in range(STAGE1_EPOCHS):
        loss = train_epoch(model, train_loader, optim1)
        sch1.step()
        if val_loader:
            m = evaluate(model, val_loader)
            if m["overall_auroc"] > best_auroc:
                best_auroc = m["overall_auroc"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if (epoch + 1) % 5 == 0:
                enz_str = ", ".join(f"{e}={m['per_enzyme_auroc'].get(e,0):.3f}" for e in ENZYMES)
                logger.info("  [S1] Ep %d: loss=%.4f, auroc=%.4f, %s", epoch + 1, loss, m["overall_auroc"], enz_str)
        else:
            if (epoch + 1) % 5 == 0:
                logger.info("  [S1] Ep %d: loss=%.4f", epoch + 1, loss)

    for p in model.shared_encoder.parameters():
        p.requires_grad = False
    adapter_params = list(model.binary_head.parameters())
    for enz in ENZYMES:
        adapter_params.extend(model.enzyme_adapters[enz].parameters())
    adapter_params.extend(model.enzyme_classifier.parameters())
    optim2 = torch.optim.AdamW(adapter_params, lr=STAGE2_LR, weight_decay=WEIGHT_DECAY)
    sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(optim2, T_max=STAGE2_EPOCHS)
    for epoch in range(STAGE2_EPOCHS):
        loss = train_epoch(model, train_loader, optim2)
        sch2.step()
        if val_loader:
            m = evaluate(model, val_loader)
            if m["overall_auroc"] > best_auroc:
                best_auroc = m["overall_auroc"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if (epoch + 1) % 10 == 0:
                enz_str = ", ".join(f"{e}={m['per_enzyme_auroc'].get(e,0):.3f}" for e in ENZYMES)
                logger.info("  [S2] Ep %d: loss=%.4f, auroc=%.4f, %s", epoch + 1, loss, m["overall_auroc"], enz_str)
        else:
            if (epoch + 1) % 10 == 0:
                logger.info("  [S2] Ep %d: loss=%.4f", epoch + 1, loss)

    for p in model.parameters():
        p.requires_grad = True
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_state, best_auroc


def load_v3_data():
    logger.info("Loading v3 data...")
    df = pd.read_csv(MULTI_SPLITS)
    with open(MULTI_SEQS) as f:
        seqs = json.load(f)
    orig_emb = torch.load(EMB_DIR / "rnafm_pooled_v3.pt", weights_only=False)
    edited_emb = torch.load(EMB_DIR / "rnafm_pooled_edited_v3.pt", weights_only=False)
    loop_df = pd.read_csv(LOOP_CSV).drop_duplicates(subset=["site_id"]).set_index("site_id")
    sc = np.load(STRUCT_CACHE, allow_pickle=True)
    struct_map = {sid: sc["delta_features"][i] for i, sid in enumerate(sc["site_ids"])}

    site_ids = df["site_id"].tolist()
    valid = [sid in seqs for sid in site_ids]
    df = df[valid].reset_index(drop=True)
    site_ids = df["site_id"].tolist()
    n = len(site_ids)
    logger.info("  %d sites with sequences", n)

    hand = build_hand_features(site_ids, seqs, struct_map, loop_df)
    rnafm_orig = np.zeros((n, D_RNAFM), dtype=np.float32)
    rnafm_delta = np.zeros((n, D_EDIT_DELTA), dtype=np.float32)
    n_found = 0
    for i, sid in enumerate(site_ids):
        if sid in orig_emb and sid in edited_emb:
            o = orig_emb[sid].numpy() if isinstance(orig_emb[sid], torch.Tensor) else orig_emb[sid]
            e = edited_emb[sid].numpy() if isinstance(edited_emb[sid], torch.Tensor) else edited_emb[sid]
            rnafm_orig[i] = o
            rnafm_delta[i] = e - o
            n_found += 1
    logger.info("  RNA-FM coverage: %d/%d", n_found, n)
    X = np.concatenate([rnafm_orig, rnafm_delta, hand], axis=1).astype(np.float32)

    # === MFE-only regime: zero struct_delta slots [1304:1311] ===
    assert STRUCT_DELTA_START == 1304 and STRUCT_DELTA_END == 1311, (
        f"expected struct_delta slice [1304:1311], got [{STRUCT_DELTA_START}:{STRUCT_DELTA_END}]")
    pre = X[:, STRUCT_DELTA_START:STRUCT_DELTA_END].copy()
    logger.info("  Pre-zero struct_delta |abs|.mean=%.6f  (all 7 slots, all %d rows)",
                float(np.mean(np.abs(pre))), n)
    X[:, STRUCT_DELTA_START:STRUCT_DELTA_END] = 0.0
    post = X[:, STRUCT_DELTA_START:STRUCT_DELTA_END]
    assert np.abs(post).max() == 0.0, "struct_delta slot zeroing failed"
    logger.info("  Post-zero struct_delta |abs|.max=%.6f  (confirms zero)", float(np.abs(post).max()))
    logger.info("  Feature matrix: %s", X.shape)

    labels_binary = df["is_edited"].values.astype(np.float32)
    labels_enzyme = np.array([ENZYME_TO_IDX.get(e, 5) for e in df["enzyme"].values], dtype=np.int64)
    per_enzyme_labels = np.full((n, N_PER_ENZYME), -1, dtype=np.float32)
    for h, enz in enumerate(ENZYMES):
        ei = ENZYME_TO_IDX[enz]
        pos_mask = (labels_binary == 1) & (labels_enzyme == ei)
        neg_mask = (labels_binary == 0) & (labels_enzyme == ei)
        per_enzyme_labels[pos_mask, h] = 1.0
        per_enzyme_labels[neg_mask, h] = 0.0
    for h, enz in enumerate(ENZYMES):
        n_pos = int((per_enzyme_labels[:, h] == 1).sum())
        n_neg = int((per_enzyme_labels[:, h] == 0).sum())
        logger.info("  %s: %d pos, %d neg", enz, n_pos, n_neg)

    del orig_emb, edited_emb
    gc.collect()
    return {
        "df": df, "X": X, "site_ids": site_ids,
        "labels_binary": labels_binary, "labels_enzyme": labels_enzyme,
        "per_enzyme_labels": per_enzyme_labels,
    }


def run_cv(data):
    logger.info("\n" + "=" * 60)
    logger.info("5-fold CV (MFE-only regime)")
    logger.info("=" * 60)
    X = data["X"]; lb = data["labels_binary"]; le = data["labels_enzyme"]; pel = data["per_enzyme_labels"]
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_results = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, lb)):
        torch.manual_seed(SEED + fold)
        np.random.seed(SEED + fold)
        t0 = time.time()
        logger.info("\n--- Fold %d/%d (train=%d, val=%d) ---", fold + 1, N_FOLDS, len(train_idx), len(val_idx))
        train_ds = SimpleDataset(X[train_idx], lb[train_idx], le[train_idx], pel[train_idx])
        val_ds = SimpleDataset(X[val_idx], lb[val_idx], le[val_idx], pel[val_idx])
        model = Phase3Model().to(DEVICE)
        if fold == 0:
            n_params = sum(p.numel() for p in model.parameters())
            logger.info("  Model params: %d", n_params)
        train_two_stage(model, train_ds, val_ds)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        m = evaluate(model, val_loader)
        elapsed = time.time() - t0
        logger.info("  Fold %d done in %.0fs, overall AUROC=%.4f", fold + 1, elapsed, m["overall_auroc"])
        for enz in ENZYMES:
            logger.info("    %s: %.3f", enz, m["per_enzyme_auroc"].get(enz, 0))
        fold_results.append(m)
        torch.save(model.state_dict(), OUTPUT_DIR / f"neural_fold{fold}.pt")
        del model, train_ds, val_ds
        if DEVICE.type == "mps":
            torch.mps.empty_cache()
        gc.collect()
    summary = {
        "overall_auroc": {
            "mean": float(np.mean([r["overall_auroc"] for r in fold_results])),
            "std": float(np.std([r["overall_auroc"] for r in fold_results])),
        },
        "per_enzyme_auroc": {},
    }
    for enz in ENZYMES:
        vals = [r["per_enzyme_auroc"].get(enz, 0) for r in fold_results]
        summary["per_enzyme_auroc"][enz] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "folds": [float(v) for v in vals],
        }
    logger.info("\n--- CV Summary (MFE-only) ---")
    logger.info("Overall AUROC: %.4f +/- %.4f", summary["overall_auroc"]["mean"], summary["overall_auroc"]["std"])
    for enz in ENZYMES:
        s = summary["per_enzyme_auroc"][enz]
        logger.info("  %s: %.3f +/- %.3f", enz, s["mean"], s["std"])
    with open(OUTPUT_DIR / "cv_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def train_full_model(data):
    logger.info("\n" + "=" * 60)
    logger.info("Training full MFE-only model on all v3 data")
    logger.info("=" * 60)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    full_ds = SimpleDataset(data["X"], data["labels_binary"], data["labels_enzyme"], data["per_enzyme_labels"])
    model = Phase3Model().to(DEVICE)
    train_two_stage(model, full_ds, val_ds=None)
    save_path = OUTPUT_DIR / "phase3_mfe_only.pt"
    torch.save(model.state_dict(), save_path)
    logger.info("Saved full model to %s", save_path)


def generate_comparison_report(cv_summary):
    # Reference per-head AUROCs from the existing canonical Phase3 cv_results.json
    ref_path = PROJECT_ROOT / "experiments" / "multi_enzyme" / "outputs" / "phase3_neural_true_validation" / "cv_results.json"
    ref = {}
    if ref_path.exists():
        ref = json.load(open(ref_path))
    lines = []
    lines.append("# MFE-only Phase3 — 5-fold CV comparison vs canonical")
    lines.append("")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"Training device: {DEVICE}")
    lines.append("")
    lines.append("## Regime difference")
    lines.append("")
    lines.append("- **Canonical Phase3** (reference): full 40-d hand features including struct_delta")
    lines.append("  (center-bpp delta, entropy delta, mfe delta, window bpp statistics — 7 slots).")
    lines.append("- **MFE-only Phase3** (this run): identical architecture and RNA-FM inputs,")
    lines.append("  but **struct_delta slots [1304:1311] held at 0.0** throughout training.")
    lines.append("  Effective hand dim = 33/40 (24 motif + 9 loop + 0 struct_delta).")
    lines.append("")
    lines.append("## Per-head 5-fold AUROC")
    lines.append("")
    lines.append("| Head | Canonical Phase3 | MFE-only Phase3 | Δ (MFE-only − canonical) |")
    lines.append("|------|-----------------|-----------------|--------------------------|")
    for enz in ENZYMES:
        mfe_m = cv_summary["per_enzyme_auroc"][enz]["mean"]
        mfe_s = cv_summary["per_enzyme_auroc"][enz]["std"]
        ref_m = ref.get("per_enzyme_auroc", {}).get(enz, {}).get("mean", float("nan"))
        delta = mfe_m - ref_m if ref_m == ref_m else float("nan")
        lines.append(f"| {enz} | {ref_m:.3f} | {mfe_m:.3f} ± {mfe_s:.3f} | {delta:+.3f} |")
    ref_ov = ref.get("overall_auroc", {}).get("mean", float("nan"))
    mfe_ov = cv_summary["overall_auroc"]["mean"]
    mfe_ov_s = cv_summary["overall_auroc"]["std"]
    lines.append(f"| **Overall** | {ref_ov:.3f} | {mfe_ov:.3f} ± {mfe_ov_s:.3f} | {(mfe_ov - ref_ov):+.3f} |")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("Per Test-7 feature-ablation evidence, struct_delta slots contribute little to the")
    lines.append("overall TCGA-mutation-enrichment OR (Δ ≈ −0.84 from 7.33 → 6.49 when zeroed at")
    lines.append("inference). Training drop per head is therefore expected to be small (<0.01 AUROC).")
    lines.append("If any head drops >0.02 AUROC it suggests that slot is actually informative and")
    lines.append("the MFE-only simplification pays a measurable cost — to be weighed against the")
    lines.append("6× Phase-2 folding speedup.")
    with open(OUTPUT_DIR / "comparison_vs_phase3.md", "w") as f:
        f.write("\n".join(lines))
    logger.info("Wrote %s", OUTPUT_DIR / "comparison_vs_phase3.md")


def main():
    t0 = time.time()
    logger.info("phase3_mfe_only training — all 7 struct_delta slots zeroed")
    data = load_v3_data()
    cv_summary = run_cv(data)
    train_full_model(data)
    generate_comparison_report(cv_summary)
    logger.info("Total runtime: %.1f min", (time.time() - t0) / 60.0)


if __name__ == "__main__":
    main()
