#!/usr/bin/env python
"""Cross-dataset generalization evaluation for EditRNA-A3A.

Evaluates how well a model trained on the Levanon dataset generalizes to
editing sites identified by independent studies (Asaoka 2019, Sharma 2015,
Alqassim 2021).

Experiment design:
1. Load trained Exp 2b model (Levanon + tiered negatives)
2. For each external dataset, create an evaluation set:
   - Positive sites from the external dataset (never seen in training)
   - Tiered negative sites from the test split (same negative distribution)
3. Compute EditRNA-A3A predictions + simple baselines (pooled, diff, concat)
4. Report per-dataset AUROC/AUPRC with confidence intervals

Usage:
    python scripts/apobec/cross_dataset_eval.py
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.apobec_dataset import (
    APOBECDataConfig,
    APOBECDataset,
    APOBECSiteSample,
    apobec_collate_fn,
    get_flanking_context,
    compute_concordance_features,
)
from models.editrna_a3a import EditRNA_A3A, EditRNAConfig
from models.encoders import CachedRNAEncoder

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = PROJECT_ROOT / "data" / "processed"
EMB_DIR = DATA_DIR / "embeddings"
EXP2B_DIR = PROJECT_ROOT / "experiments" / "apobec" / "outputs" / "exp2_levanon_tiered_negatives"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "apobec" / "outputs" / "exp3_cross_dataset"

N_TISSUES = 49  # from apobec_dataset


def load_embedding_caches():
    """Load pre-computed RNA-FM embedding caches."""
    logger.info("Loading embedding caches...")
    tokens = torch.load(EMB_DIR / "rnafm_tokens.pt", weights_only=False)
    pooled = torch.load(EMB_DIR / "rnafm_pooled.pt", weights_only=False)
    tokens_ed = torch.load(EMB_DIR / "rnafm_tokens_edited.pt", weights_only=False)
    pooled_ed = torch.load(EMB_DIR / "rnafm_pooled_edited.pt", weights_only=False)
    logger.info("  Loaded %d embeddings", len(tokens))
    return tokens, pooled, tokens_ed, pooled_ed


def load_trained_model(tokens, pooled, tokens_ed, pooled_ed, device):
    """Load the trained Exp 2b EditRNA-A3A model."""
    logger.info("Loading trained model from %s", EXP2B_DIR)

    with open(EXP2B_DIR / "config.json") as f:
        exp_config = json.load(f)

    cached_encoder = CachedRNAEncoder(
        tokens_cache=tokens,
        pooled_cache=pooled,
        tokens_edited_cache=tokens_ed,
        pooled_edited_cache=pooled_ed,
        d_model=exp_config["d_model"],
    )

    model_config = EditRNAConfig(
        primary_encoder="cached",
        d_model=exp_config["d_model"],
        d_edit=exp_config["d_edit"],
        d_fused=exp_config["d_fused"],
        edit_n_heads=exp_config["edit_n_heads"],
        use_structure_delta=exp_config["use_structure_delta"],
        use_dual_encoder=exp_config["use_dual_encoder"],
        use_gnn=exp_config["use_gnn"],
        finetune_last_n=0,
        head_dropout=exp_config["head_dropout"],
        fusion_dropout=exp_config["fusion_dropout"],
    )

    model = EditRNA_A3A(config=model_config, primary_encoder=cached_encoder)

    checkpoint = torch.load(EXP2B_DIR / "best_model.pt", weights_only=False, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    logger.info("  Model loaded (best epoch=%d)", checkpoint["epoch"])
    return model


def build_site_sample(site_id, label, seq, window_size=100):
    """Build an APOBECSiteSample with minimal labels."""
    edit_pos = min(window_size, len(seq) // 2)
    seq_list = list(seq)
    seq_list[edit_pos] = "C"
    seq = "".join(seq_list)

    flanking = get_flanking_context(seq, edit_pos)
    concordance = np.zeros(5, dtype=np.float32)
    if label == 0:
        concordance[0] = 1.0

    return APOBECSiteSample(
        sequence=seq,
        edit_pos=edit_pos,
        is_edited=float(label),
        editing_rate_log2=float("nan"),
        apobec_class=-1 if label == 1 else 3,
        structure_type=-1,
        tissue_spec_class=-1,
        n_tissues_log2=float("nan"),
        exonic_function=-1,
        conservation=float("nan"),
        cancer_survival=float("nan"),
        tissue_rates=np.full(N_TISSUES, np.nan, dtype=np.float32),
        hek293_rate=float("nan"),
        flanking_context=flanking,
        concordance_features=concordance,
        site_id=site_id,
        chrom="",
        position=0,
        gene="",
    )


def create_dataset_eval_sets(splits_expanded, sequences, emb_ids):
    """Create per-dataset evaluation sets.

    For each external dataset, create:
    - Positive sites from that dataset (all splits, since none were in Exp 2b training)
    - Negative sites from the tiered negative test split

    Returns dict: {dataset_name: (positive_ids, negative_ids)}
    """
    # Get test negatives (tiered negatives from test split)
    test_negatives = splits_expanded[
        (splits_expanded["label"] == 0) &
        (splits_expanded["split"] == "test") &
        (splits_expanded["dataset_source"].isin(["tier2_negative", "tier3_negative"]))
    ]
    test_neg_ids = set(test_negatives["site_id"]) & emb_ids
    logger.info("Test negatives available: %d", len(test_neg_ids))

    # Also get all negatives for baselines (train split for fitting, test for eval)
    train_negatives = splits_expanded[
        (splits_expanded["label"] == 0) &
        (splits_expanded["split"] == "train") &
        (splits_expanded["dataset_source"].isin(["tier2_negative", "tier3_negative"]))
    ]
    train_neg_ids = set(train_negatives["site_id"]) & emb_ids

    # Get Levanon training positives (for baseline fitting)
    levanon_train = splits_expanded[
        (splits_expanded["dataset_source"] == "advisor_c2t") &
        (splits_expanded["split"] == "train") &
        (splits_expanded["label"] == 1)
    ]
    levanon_train_ids = set(levanon_train["site_id"]) & emb_ids

    # Levanon test positives (for reference - same as Exp 2b test)
    levanon_test = splits_expanded[
        (splits_expanded["dataset_source"] == "advisor_c2t") &
        (splits_expanded["split"] == "test") &
        (splits_expanded["label"] == 1)
    ]
    levanon_test_ids = set(levanon_test["site_id"]) & emb_ids

    eval_sets = {}

    # Reference: Levanon test (same as Exp 2b)
    eval_sets["levanon_test"] = {
        "pos_ids": sorted(levanon_test_ids),
        "neg_ids": sorted(test_neg_ids),
        "desc": f"Levanon test ({len(levanon_test_ids)} pos + {len(test_neg_ids)} neg)",
    }

    # External datasets - use ALL their sites (never in Levanon training)
    for dataset_name in ["asaoka_2019", "sharma_2015", "alqassim_2021"]:
        ext_sites = splits_expanded[
            (splits_expanded["dataset_source"] == dataset_name) &
            (splits_expanded["label"] == 1)
        ]
        ext_ids = set(ext_sites["site_id"]) & emb_ids
        if len(ext_ids) == 0:
            logger.warning("No sites with embeddings for %s", dataset_name)
            continue

        eval_sets[dataset_name] = {
            "pos_ids": sorted(ext_ids),
            "neg_ids": sorted(test_neg_ids),
            "desc": f"{dataset_name} ({len(ext_ids)} pos + {len(test_neg_ids)} neg)",
        }

    # Training data for baselines
    train_info = {
        "pos_ids": sorted(levanon_train_ids),
        "neg_ids": sorted(train_neg_ids),
    }

    return eval_sets, train_info


@torch.no_grad()
def evaluate_editrna(model, eval_set, sequences, device, batch_size=32):
    """Evaluate EditRNA-A3A model on a dataset."""
    all_ids = eval_set["pos_ids"] + eval_set["neg_ids"]
    all_labels = [1] * len(eval_set["pos_ids"]) + [0] * len(eval_set["neg_ids"])

    # Build samples
    data_config = APOBECDataConfig(window_size=100)
    samples = []
    for sid, label in zip(all_ids, all_labels):
        seq = sequences.get(sid, "A" * 201)
        samples.append(build_site_sample(sid, label, seq))

    dataset = APOBECDataset(samples, data_config)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=apobec_collate_fn, num_workers=0,
    )

    all_scores = []
    for batch in loader:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else
            ({kk: vv.to(device) if isinstance(vv, torch.Tensor) else vv for kk, vv in v.items()} if isinstance(v, dict) else v)
            for k, v in batch.items()
        }
        output = model(batch)
        logits = output["predictions"]["binary_logit"].squeeze(-1).cpu().numpy()
        probs = 1.0 / (1.0 + np.exp(-logits))
        all_scores.append(probs)

    scores = np.concatenate(all_scores)
    labels = np.array(all_labels)

    auroc = roc_auc_score(labels, scores)
    auprc = average_precision_score(labels, scores)

    return {"auroc": auroc, "auprc": auprc, "n_pos": len(eval_set["pos_ids"]), "n_neg": len(eval_set["neg_ids"])}


def evaluate_baselines(eval_set, train_info, pooled_cache, pooled_edited_cache):
    """Evaluate simple baselines (pooled, diff, concat) with LogReg and GB."""
    results = {}

    # Get training embeddings
    train_ids = train_info["pos_ids"] + train_info["neg_ids"]
    train_labels = np.array([1] * len(train_info["pos_ids"]) + [0] * len(train_info["neg_ids"]))

    X_train_orig = np.stack([pooled_cache[sid].numpy() for sid in train_ids])
    X_train_edit = np.stack([pooled_edited_cache[sid].numpy() for sid in train_ids])

    # Get eval embeddings
    eval_ids = eval_set["pos_ids"] + eval_set["neg_ids"]
    eval_labels = np.array([1] * len(eval_set["pos_ids"]) + [0] * len(eval_set["neg_ids"]))

    X_eval_orig = np.stack([pooled_cache[sid].numpy() for sid in eval_ids])
    X_eval_edit = np.stack([pooled_edited_cache[sid].numpy() for sid in eval_ids])

    # Feature representations
    representations = {
        "pooled": (X_train_orig, X_eval_orig),
        "diff": (X_train_edit - X_train_orig, X_eval_edit - X_eval_orig),
        "concat": (
            np.concatenate([X_train_orig, X_train_edit], axis=1),
            np.concatenate([X_eval_orig, X_eval_edit], axis=1),
        ),
    }

    for repr_name, (X_tr, X_ev) in representations.items():
        # Logistic Regression
        lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        lr.fit(X_tr, train_labels)
        lr_scores = lr.predict_proba(X_ev)[:, 1]
        results[f"logreg_{repr_name}"] = {
            "auroc": roc_auc_score(eval_labels, lr_scores),
            "auprc": average_precision_score(eval_labels, lr_scores),
        }

        # Gradient Boosting
        gb = GradientBoostingClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42,
        )
        gb.fit(X_tr, train_labels)
        gb_scores = gb.predict_proba(X_ev)[:, 1]
        results[f"gb_{repr_name}"] = {
            "auroc": roc_auc_score(eval_labels, gb_scores),
            "auprc": average_precision_score(eval_labels, gb_scores),
        }

    return results


def bootstrap_ci(labels, scores, metric_fn, n_bootstrap=1000, ci=0.95, seed=42):
    """Compute bootstrap confidence interval for a metric."""
    rng = np.random.RandomState(seed)
    n = len(labels)
    stats = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        if len(np.unique(labels[idx])) < 2:
            continue
        stats.append(metric_fn(labels[idx], scores[idx]))

    if not stats:
        return float("nan"), float("nan"), float("nan")

    stats = sorted(stats)
    alpha = (1 - ci) / 2
    lo = stats[int(alpha * len(stats))]
    hi = stats[int((1 - alpha) * len(stats))]
    return np.mean(stats), lo, hi


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    splits_expanded = pd.read_csv(DATA_DIR / "splits_expanded.csv")
    with open(DATA_DIR / "site_sequences.json") as f:
        sequences = json.load(f)

    with open(EMB_DIR / "rnafm_site_ids.json") as f:
        emb_ids = set(json.load(f))

    # Load embeddings
    tokens, pooled, tokens_ed, pooled_ed = load_embedding_caches()

    # Device
    device = torch.device("cpu")  # cached mode is fast on CPU
    logger.info("Device: %s", device)

    # Load model
    model = load_trained_model(tokens, pooled, tokens_ed, pooled_ed, device)

    # Create evaluation sets
    eval_sets, train_info = create_dataset_eval_sets(splits_expanded, sequences, emb_ids)

    for name, es in eval_sets.items():
        logger.info("  %s: %s", name, es["desc"])

    # Evaluate
    all_results = {}

    for dataset_name, eval_set in eval_sets.items():
        logger.info("\n=== Evaluating: %s ===", dataset_name)
        logger.info("  %s", eval_set["desc"])

        # EditRNA-A3A
        editrna_results = evaluate_editrna(model, eval_set, sequences, device)
        logger.info("  EditRNA-A3A: AUROC=%.4f  AUPRC=%.4f",
                     editrna_results["auroc"], editrna_results["auprc"])

        # Simple baselines
        baseline_results = evaluate_baselines(eval_set, train_info, pooled, pooled_ed)
        for bname, bres in baseline_results.items():
            logger.info("  %s: AUROC=%.4f  AUPRC=%.4f", bname, bres["auroc"], bres["auprc"])

        all_results[dataset_name] = {
            "editrna_a3a": editrna_results,
            "baselines": baseline_results,
        }

    # Save results
    results_path = OUTPUT_DIR / "cross_dataset_results.json"

    def _convert(obj):
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        raise TypeError(f"Cannot serialize {type(obj)}")

    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=_convert)
    logger.info("\nResults saved to %s", results_path)

    # Print summary table
    print("\n" + "=" * 90)
    print("CROSS-DATASET GENERALIZATION RESULTS")
    print("Model trained on: Levanon (629 sites) + tiered negatives")
    print("=" * 90)
    print(f"{'Dataset':<20} {'N_pos':>6} {'N_neg':>6} | {'EditRNA':>8} | {'LR-pool':>8} {'LR-diff':>8} {'GB-diff':>8} {'GB-cat':>8}")
    print(f"{'':20} {'':>6} {'':>6} | {'AUROC':>8} | {'AUROC':>8} {'AUROC':>8} {'AUROC':>8} {'AUROC':>8}")
    print("-" * 90)

    for dataset_name in ["levanon_test", "asaoka_2019", "sharma_2015", "alqassim_2021"]:
        if dataset_name not in all_results:
            continue
        r = all_results[dataset_name]
        er = r["editrna_a3a"]
        bl = r["baselines"]
        print(f"{dataset_name:<20} {er['n_pos']:>6} {er['n_neg']:>6} | "
              f"{er['auroc']:>8.4f} | "
              f"{bl['logreg_pooled']['auroc']:>8.4f} "
              f"{bl['logreg_diff']['auroc']:>8.4f} "
              f"{bl['gb_diff']['auroc']:>8.4f} "
              f"{bl['gb_concat']['auroc']:>8.4f}")

    print("-" * 90)
    print("\n")

    # AUPRC table
    print(f"{'Dataset':<20} {'N_pos':>6} {'N_neg':>6} | {'EditRNA':>8} | {'LR-pool':>8} {'LR-diff':>8} {'GB-diff':>8} {'GB-cat':>8}")
    print(f"{'':20} {'':>6} {'':>6} | {'AUPRC':>8} | {'AUPRC':>8} {'AUPRC':>8} {'AUPRC':>8} {'AUPRC':>8}")
    print("-" * 90)

    for dataset_name in ["levanon_test", "asaoka_2019", "sharma_2015", "alqassim_2021"]:
        if dataset_name not in all_results:
            continue
        r = all_results[dataset_name]
        er = r["editrna_a3a"]
        bl = r["baselines"]
        print(f"{dataset_name:<20} {er['n_pos']:>6} {er['n_neg']:>6} | "
              f"{er['auprc']:>8.4f} | "
              f"{bl['logreg_pooled']['auprc']:>8.4f} "
              f"{bl['logreg_diff']['auprc']:>8.4f} "
              f"{bl['gb_diff']['auprc']:>8.4f} "
              f"{bl['gb_concat']['auprc']:>8.4f}")

    print("-" * 90)

    # Compute advantage of EditRNA over best baseline per dataset
    print("\n=== EditRNA-A3A advantage over best baseline ===")
    for dataset_name in ["levanon_test", "asaoka_2019", "sharma_2015", "alqassim_2021"]:
        if dataset_name not in all_results:
            continue
        r = all_results[dataset_name]
        editrna_auroc = r["editrna_a3a"]["auroc"]
        best_baseline_auroc = max(bl["auroc"] for bl in r["baselines"].values())
        best_baseline_name = max(r["baselines"].items(), key=lambda x: x[1]["auroc"])[0]
        delta = editrna_auroc - best_baseline_auroc
        print(f"  {dataset_name:<20}: EditRNA {editrna_auroc:.4f} vs {best_baseline_name} {best_baseline_auroc:.4f}  (Δ={delta:+.4f})")


if __name__ == "__main__":
    main()
