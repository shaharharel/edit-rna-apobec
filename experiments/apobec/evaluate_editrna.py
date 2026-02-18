#!/usr/bin/env python
"""Comprehensive multi-task evaluation for EditRNA-A3A experiments.

Evaluates a trained EditRNA-A3A checkpoint on all 11 tasks:

PRIMARY:
  1. Binary editing classification (AUROC, AUPRC, F1)
  2. Editing rate regression (Spearman rho, MSE)
  3. Enzyme specificity (accuracy, macro F1)

SECONDARY:
  4. Structure type (accuracy, macro F1)
  5. Tissue specificity (accuracy, macro F1)
  6. N tissues edited (Spearman rho)

TERTIARY:
  7. Functional impact (accuracy, macro F1)
  8. Conservation (AUROC)
  9. Cancer survival (AUROC)

AUXILIARY:
  10. HEK293 rate (Spearman rho)

Also generates:
  - Edit embedding UMAP/t-SNE visualizations
  - Task weight analysis from uncertainty weighting
  - Comparison with Experiment 0 tabular baseline

Usage:
    python experiments/apobec/evaluate_editrna.py \\
        --checkpoint experiments/apobec/outputs/exp1_rnafm_v1/best_model.pt \\
        --output_dir experiments/apobec/outputs/exp1_rnafm_v1/eval
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.apobec_dataset import (
    APOBECDataConfig,
    APOBECDataset,
    apobec_collate_fn,
    create_apobec_dataloaders,
)
from models.editrna_a3a import EditRNA_A3A, EditRNAConfig, create_editrna_mock
from models.prediction_heads import (
    ENZYME_CLASSES,
    STRUCTURE_CLASSES,
    TISSUE_SPEC_CLASSES,
    FUNCTION_CLASSES,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def compute_binary_metrics(y_true, y_score):
    """Compute binary classification metrics."""
    from sklearn.metrics import (
        accuracy_score, average_precision_score, f1_score,
        precision_recall_curve, precision_score, recall_score, roc_auc_score,
    )
    mask = ~(np.isnan(y_true) | np.isnan(y_score))
    y_true, y_score = y_true[mask], y_score[mask]
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan"), "f1": float("nan"),
                "precision": float("nan"), "recall": float("nan"), "n": 0}

    auroc = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)
    prec_arr, rec_arr, thresholds = precision_recall_curve(y_true, y_score)
    f1_arr = 2 * prec_arr[:-1] * rec_arr[:-1] / (prec_arr[:-1] + rec_arr[:-1] + 1e-8)
    best_idx = np.argmax(f1_arr)
    thresh = thresholds[best_idx] if len(thresholds) > 0 else 0.5
    y_pred = (y_score >= thresh).astype(int)

    return {
        "auroc": auroc, "auprc": auprc,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "threshold": float(thresh),
        "n_positive": int(y_true.sum()),
        "n_negative": int(len(y_true) - y_true.sum()),
    }


def compute_multiclass_metrics(y_true, y_logits, class_names):
    """Compute multi-class metrics."""
    from sklearn.metrics import accuracy_score, f1_score
    mask = y_true >= 0
    y_true, y_logits = y_true[mask], y_logits[mask]
    if len(y_true) == 0:
        return {"accuracy": float("nan"), "macro_f1": float("nan"), "n": 0}

    y_pred = np.argmax(y_logits, axis=1)
    result = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "n": int(len(y_true)),
    }
    for i, name in enumerate(class_names):
        cls_mask = y_true == i
        if cls_mask.sum() > 0:
            result[f"{name}_acc"] = float((y_pred[cls_mask] == i).mean())
            result[f"{name}_n"] = int(cls_mask.sum())
    return result


def compute_regression_metrics(y_true, y_pred):
    """Compute regression metrics (filters NaN)."""
    from scipy.stats import spearmanr
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) < 5:
        return {"spearman_rho": float("nan"), "mse": float("nan"), "n": 0}
    rho, p = spearmanr(y_true, y_pred)
    mse = float(np.mean((y_true - y_pred) ** 2))
    return {"spearman_rho": rho, "p_value": p, "mse": mse, "n": int(len(y_true))}


# ---------------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_predictions(model, dataloader, device):
    """Run model on dataloader and collect all predictions + targets + embeddings."""
    model.eval()
    all_preds = {}
    all_targets = {}
    all_edit_embs = []
    all_fused = []
    all_site_ids = []

    for batch in dataloader:
        # Move to device
        batch_dev = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch_dev[k] = v.to(device)
            elif isinstance(v, dict):
                batch_dev[k] = {kk: vv.to(device) if isinstance(vv, torch.Tensor) else vv
                                for kk, vv in v.items()}
            else:
                batch_dev[k] = v

        output = model(batch_dev)
        preds = output["predictions"]

        # Collect predictions
        for key, val in preds.items():
            if key not in all_preds:
                all_preds[key] = []
            all_preds[key].append(val.cpu().numpy())

        # Collect targets
        for key, val in batch_dev["targets"].items():
            if key not in all_targets:
                all_targets[key] = []
            all_targets[key].append(val.cpu().numpy())

        # Collect embeddings
        all_edit_embs.append(output["edit_embedding"].cpu().numpy())
        all_fused.append(output["fused"].cpu().numpy())

        # Site IDs
        if "site_ids" in batch:
            all_site_ids.extend(batch["site_ids"])

    # Concatenate
    for k in all_preds:
        all_preds[k] = np.concatenate(all_preds[k])
    for k in all_targets:
        all_targets[k] = np.concatenate(all_targets[k])
    edit_embs = np.concatenate(all_edit_embs)
    fused_embs = np.concatenate(all_fused)

    return all_preds, all_targets, edit_embs, fused_embs, all_site_ids


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------

def evaluate_all_tasks(preds, targets):
    """Evaluate all 11 tasks and return structured results.

    Tasks that are only meaningful for positive editing sites (rate, enzyme,
    structure, tissue_spec, n_tissues, function, conservation, cancer, hek293)
    are evaluated on positive samples only (binary target > 0.5).
    Binary classification is evaluated on all samples.
    """
    results = {}
    pos_mask = targets["binary"] > 0.5  # positive editing sites

    # --- PRIMARY ---
    # 1. Binary editing (all samples)
    binary_logits = preds["binary_logit"].squeeze(-1)
    binary_probs = 1.0 / (1.0 + np.exp(-binary_logits))
    results["binary"] = compute_binary_metrics(targets["binary"], binary_probs)

    # 2. Editing rate (positive sites only)
    rate_preds = preds["editing_rate"].squeeze(-1)
    rate_mask = ~np.isnan(targets["rate"]) & pos_mask
    rate_true = np.where(rate_mask, targets["rate"], np.nan)
    rate_pred = np.where(rate_mask, rate_preds, np.nan)
    results["rate"] = compute_regression_metrics(rate_true, rate_pred)

    # 3. Enzyme specificity (positive sites only, exclude Unknown=-1)
    enzyme_names = [k for k, v in sorted(ENZYME_CLASSES.items(), key=lambda x: x[1])]
    enzyme_targets_pos = np.where(pos_mask, targets["enzyme"], -1)
    enzyme_logits_pos = preds["enzyme_logits"]
    results["enzyme"] = compute_multiclass_metrics(
        enzyme_targets_pos, enzyme_logits_pos, enzyme_names)

    # --- SECONDARY ---
    # 4. Structure type (positive sites only)
    structure_names = [k for k, v in sorted(STRUCTURE_CLASSES.items(), key=lambda x: x[1])]
    struct_targets_pos = np.where(pos_mask, targets["structure"], -1)
    results["structure"] = compute_multiclass_metrics(
        struct_targets_pos, preds["structure_logits"], structure_names)

    # 5. Tissue specificity (positive sites only)
    tissue_names = [k for k, v in sorted(TISSUE_SPEC_CLASSES.items(), key=lambda x: x[1])]
    tissue_targets_pos = np.where(pos_mask, targets["tissue_spec"], -1)
    results["tissue_spec"] = compute_multiclass_metrics(
        tissue_targets_pos, preds["tissue_spec_logits"], tissue_names)

    # 6. N tissues edited (positive sites only)
    n_tissue_preds = preds["n_tissues"].squeeze(-1)
    n_tissue_mask = ~np.isnan(targets["n_tissues"]) & pos_mask
    nt_true = np.where(n_tissue_mask, targets["n_tissues"], np.nan)
    nt_pred = np.where(n_tissue_mask, n_tissue_preds, np.nan)
    results["n_tissues"] = compute_regression_metrics(nt_true, nt_pred)

    # --- TERTIARY ---
    # 7. Functional impact (CDS-only positive sites)
    function_names = [k for k, v in sorted(FUNCTION_CLASSES.items(), key=lambda x: x[1])]
    func_targets_pos = np.where(pos_mask, targets["function"], -1)
    results["function"] = compute_multiclass_metrics(
        func_targets_pos, preds["function_logits"], function_names)

    # 8. Conservation (positive sites only)
    cons_logits = preds["conservation_logit"].squeeze(-1)
    cons_probs = 1.0 / (1.0 + np.exp(-cons_logits))
    cons_mask = ~np.isnan(targets["conservation"]) & pos_mask
    cons_true = np.where(cons_mask, targets["conservation"], np.nan)
    cons_pred = np.where(cons_mask, cons_probs, np.nan)
    results["conservation"] = compute_binary_metrics(cons_true, cons_pred)

    # 9. Cancer survival (positive sites only)
    cancer_logits = preds["cancer_logit"].squeeze(-1)
    cancer_probs = 1.0 / (1.0 + np.exp(-cancer_logits))
    cancer_mask = ~np.isnan(targets["cancer"]) & pos_mask
    cancer_true = np.where(cancer_mask, targets["cancer"], np.nan)
    cancer_pred = np.where(cancer_mask, cancer_probs, np.nan)
    results["cancer"] = compute_binary_metrics(cancer_true, cancer_pred)

    # --- AUXILIARY ---
    # 10. HEK293 rate (positive sites only)
    hek_preds = preds["hek293_rate"].squeeze(-1)
    hek_mask = ~np.isnan(targets["hek293"]) & pos_mask
    hek_true = np.where(hek_mask, targets["hek293"], np.nan)
    hek_pred = np.where(hek_mask, hek_preds, np.nan)
    results["hek293"] = compute_regression_metrics(hek_true, hek_pred)

    return results


# ---------------------------------------------------------------------------
# Embedding visualization
# ---------------------------------------------------------------------------

def visualize_embeddings(edit_embs, targets, site_ids, output_dir):
    """Generate UMAP and t-SNE plots of edit embeddings colored by various labels."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
    except ImportError:
        logger.warning("matplotlib or sklearn not available, skipping embedding visualization")
        return

    try:
        import umap
        has_umap = True
    except ImportError:
        has_umap = False
        logger.info("umap-learn not installed, using t-SNE only")

    output_dir = Path(output_dir)

    # Only use positive sites for embedding visualization (they have meaningful labels)
    pos_mask = targets["binary"] > 0.5
    pos_embs = edit_embs[pos_mask]
    pos_ids = [sid for sid, m in zip(site_ids, pos_mask) if m] if site_ids else []

    if len(pos_embs) < 10:
        logger.warning("Too few positive sites for embedding visualization")
        return

    # t-SNE
    logger.info("Computing t-SNE on %d positive edit embeddings...", len(pos_embs))
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(pos_embs) - 1))
    emb_2d_tsne = tsne.fit_transform(pos_embs)

    # UMAP
    emb_2d_umap = None
    if has_umap:
        logger.info("Computing UMAP on %d positive edit embeddings...", len(pos_embs))
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(pos_embs) - 1))
        emb_2d_umap = reducer.fit_transform(pos_embs)

    # Color maps for different label types
    color_configs = []

    # Enzyme class
    enzyme_labels = targets["enzyme"][pos_mask]
    enzyme_mask = enzyme_labels >= 0
    if enzyme_mask.sum() > 0:
        enzyme_names = [k for k, v in sorted(ENZYME_CLASSES.items(), key=lambda x: x[1])]
        color_configs.append({
            "name": "enzyme_class",
            "labels": enzyme_labels,
            "mask": enzyme_mask,
            "class_names": enzyme_names,
            "title": "Edit Embeddings colored by APOBEC Enzyme",
            "cmap": "tab10",
        })

    # Structure type
    struct_labels = targets["structure"][pos_mask]
    struct_mask = struct_labels >= 0
    if struct_mask.sum() > 0:
        struct_names = [k for k, v in sorted(STRUCTURE_CLASSES.items(), key=lambda x: x[1])]
        color_configs.append({
            "name": "structure_type",
            "labels": struct_labels,
            "mask": struct_mask,
            "class_names": struct_names,
            "title": "Edit Embeddings colored by Structure Type",
            "cmap": "Set2",
        })

    # Tissue specificity
    tissue_labels = targets["tissue_spec"][pos_mask]
    tissue_mask = tissue_labels >= 0
    if tissue_mask.sum() > 0:
        tissue_names = [k for k, v in sorted(TISSUE_SPEC_CLASSES.items(), key=lambda x: x[1])]
        color_configs.append({
            "name": "tissue_specificity",
            "labels": tissue_labels,
            "mask": tissue_mask,
            "class_names": tissue_names,
            "title": "Edit Embeddings colored by Tissue Specificity",
            "cmap": "Set1",
        })

    # Conservation (binary)
    cons_labels = targets["conservation"][pos_mask]
    cons_mask = ~np.isnan(cons_labels)
    if cons_mask.sum() > 0:
        color_configs.append({
            "name": "conservation",
            "labels": cons_labels.astype(int),
            "mask": cons_mask,
            "class_names": ["Not Conserved", "Conserved"],
            "title": "Edit Embeddings colored by Mammalian Conservation",
            "cmap": "RdYlGn",
        })

    # Editing rate (continuous)
    rate_labels = targets["rate"][pos_mask]
    rate_mask = ~np.isnan(rate_labels)

    # Generate plots
    for cfg in color_configs:
        for method_name, emb_2d in [("tsne", emb_2d_tsne), ("umap", emb_2d_umap)]:
            if emb_2d is None:
                continue

            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            mask = cfg["mask"]
            labels = cfg["labels"][mask]
            points = emb_2d[mask]

            n_classes = len(cfg["class_names"])
            cmap = plt.cm.get_cmap(cfg["cmap"], n_classes)

            for i, name in enumerate(cfg["class_names"]):
                cls_mask = labels == i
                if cls_mask.sum() > 0:
                    ax.scatter(points[cls_mask, 0], points[cls_mask, 1],
                              c=[cmap(i)], label=f"{name} (n={cls_mask.sum()})",
                              alpha=0.7, s=30, edgecolors="k", linewidths=0.3)

            ax.set_title(f"{cfg['title']} ({method_name.upper()})")
            ax.legend(loc="best", fontsize=8)
            ax.set_xlabel(f"{method_name.upper()} 1")
            ax.set_ylabel(f"{method_name.upper()} 2")
            plt.tight_layout()
            plt.savefig(output_dir / f"emb_{cfg['name']}_{method_name}.png", dpi=150)
            plt.close()

    # Editing rate (continuous colormap)
    if rate_mask.sum() > 0:
        for method_name, emb_2d in [("tsne", emb_2d_tsne), ("umap", emb_2d_umap)]:
            if emb_2d is None:
                continue
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            points = emb_2d[rate_mask]
            rates = rate_labels[rate_mask]
            sc = ax.scatter(points[:, 0], points[:, 1], c=rates,
                           cmap="viridis", alpha=0.7, s=30, edgecolors="k", linewidths=0.3)
            plt.colorbar(sc, ax=ax, label="log2(max_rate + 0.01)")
            ax.set_title(f"Edit Embeddings colored by Editing Rate ({method_name.upper()})")
            ax.set_xlabel(f"{method_name.upper()} 1")
            ax.set_ylabel(f"{method_name.upper()} 2")
            plt.tight_layout()
            plt.savefig(output_dir / f"emb_editing_rate_{method_name}.png", dpi=150)
            plt.close()

    # Save embeddings for later analysis
    np.save(output_dir / "edit_embeddings_positive.npy", pos_embs)
    np.save(output_dir / "tsne_2d.npy", emb_2d_tsne)
    if emb_2d_umap is not None:
        np.save(output_dir / "umap_2d.npy", emb_2d_umap)
    if pos_ids:
        with open(output_dir / "embedding_site_ids.json", "w") as f:
            json.dump(pos_ids, f)

    logger.info("Saved embedding visualizations to %s", output_dir)


# ---------------------------------------------------------------------------
# Task weight analysis
# ---------------------------------------------------------------------------

def analyze_task_weights(model, output_dir):
    """Analyze learned task weights from uncertainty weighting."""
    output_dir = Path(output_dir)
    weights = model.loss_fn.get_task_weights()
    log_vars = {name: model.loss_fn.log_vars[name].item()
                for name in model.loss_fn.task_names}

    result = {
        "effective_weights": weights,
        "log_variances": log_vars,
    }

    with open(output_dir / "task_weights.json", "w") as f:
        json.dump(result, f, indent=2)

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = list(weights.keys())
        vals = [weights[n] for n in names]

        fig, ax = plt.subplots(1, 1, figsize=(12, 5))
        bars = ax.bar(range(len(names)), vals, color="steelblue", edgecolor="k")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_ylabel("Effective Weight (precision)")
        ax.set_title("Learned Multi-Task Weights (Uncertainty Weighting)")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        plt.tight_layout()
        plt.savefig(output_dir / "task_weights.png", dpi=150)
        plt.close()
    except ImportError:
        pass

    return result


# ---------------------------------------------------------------------------
# Comparison with tabular baseline
# ---------------------------------------------------------------------------

def compare_with_tabular(results, tabular_path=None, output_dir=None):
    """Generate comparison table between neural model and tabular baseline."""
    if tabular_path is None:
        tabular_path = PROJECT_ROOT / "experiments" / "apobec" / "outputs" / "exp0_tabular" / "exp0_results.json"

    if not Path(tabular_path).exists():
        logger.info("Tabular baseline results not found at %s, skipping comparison", tabular_path)
        return None

    with open(tabular_path) as f:
        tabular = json.load(f)

    comparison = {
        "binary": {
            "neural_auroc": results["binary"].get("auroc", float("nan")),
            "neural_auprc": results["binary"].get("auprc", float("nan")),
            "neural_f1": results["binary"].get("f1", float("nan")),
            "tabular_auroc_common": tabular["binary_classification"]["common_only"]["gradient_boosting"]["test"]["auroc"],
            "tabular_auprc_common": tabular["binary_classification"]["common_only"]["gradient_boosting"]["test"]["auprc"],
            "tabular_f1_common": tabular["binary_classification"]["common_only"]["gradient_boosting"]["test"]["f1"],
            "tabular_auroc_all": tabular["binary_classification"]["all_features"]["gradient_boosting"]["test"]["auroc"],
            "tabular_auprc_all": tabular["binary_classification"]["all_features"]["gradient_boosting"]["test"]["auprc"],
            "tabular_f1_all": tabular["binary_classification"]["all_features"]["gradient_boosting"]["test"]["f1"],
        },
        "rate": {
            "neural_spearman": results["rate"].get("spearman_rho", float("nan")),
            "tabular_spearman_rf": tabular["rate_regression"]["random_forest"]["test"]["spearman_rho"],
            "tabular_spearman_gb": tabular["rate_regression"]["gradient_boosting"]["test"]["spearman_rho"],
        },
        "enzyme": {
            "neural_accuracy": results["enzyme"].get("accuracy", float("nan")),
            "neural_macro_f1": results["enzyme"].get("macro_f1", float("nan")),
            "tabular_accuracy_gb": tabular["multiclass_tasks"]["enzyme"]["gradient_boosting"]["test"]["accuracy"],
            "tabular_macro_f1_gb": tabular["multiclass_tasks"]["enzyme"]["gradient_boosting"]["test"]["macro_f1"],
        },
        "tissue_spec": {
            "neural_accuracy": results["tissue_spec"].get("accuracy", float("nan")),
            "neural_macro_f1": results["tissue_spec"].get("macro_f1", float("nan")),
            "tabular_accuracy_gb": tabular["multiclass_tasks"]["tissue_spec"]["gradient_boosting"]["test"]["accuracy"],
            "tabular_macro_f1_gb": tabular["multiclass_tasks"]["tissue_spec"]["gradient_boosting"]["test"]["macro_f1"],
        },
        "structure": {
            "neural_accuracy": results["structure"].get("accuracy", float("nan")),
            "tabular_accuracy_gb": tabular["multiclass_tasks"]["structure"]["gradient_boosting"]["test"]["accuracy"],
        },
        "function": {
            "neural_accuracy": results["function"].get("accuracy", float("nan")),
            "neural_macro_f1": results["function"].get("macro_f1", float("nan")),
            "tabular_accuracy_gb": tabular["multiclass_tasks"]["function"]["gradient_boosting"]["test"]["accuracy"],
            "tabular_macro_f1_gb": tabular["multiclass_tasks"]["function"]["gradient_boosting"]["test"]["macro_f1"],
        },
    }

    if output_dir:
        with open(Path(output_dir) / "comparison_vs_tabular.json", "w") as f:
            json.dump(comparison, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, float)) else x)

    # Print comparison table
    print("\n" + "=" * 80)
    print("COMPARISON: Neural (EditRNA-A3A) vs. Tabular Baseline (Exp 0)")
    print("=" * 80)

    print(f"\n{'Task':<25} {'Metric':<15} {'Neural':<10} {'Tabular':<10} {'Delta':<10}")
    print("-" * 70)

    rows = [
        ("Binary", "AUROC", comparison["binary"]["neural_auroc"],
         comparison["binary"]["tabular_auroc_common"], "vs common"),
        ("Binary", "AUROC", comparison["binary"]["neural_auroc"],
         comparison["binary"]["tabular_auroc_all"], "vs all-feat"),
        ("Rate", "Spearman", comparison["rate"]["neural_spearman"],
         comparison["rate"]["tabular_spearman_rf"], "vs RF"),
        ("Enzyme", "Accuracy", comparison["enzyme"]["neural_accuracy"],
         comparison["enzyme"]["tabular_accuracy_gb"], "vs GB"),
        ("Enzyme", "Macro-F1", comparison["enzyme"]["neural_macro_f1"],
         comparison["enzyme"]["tabular_macro_f1_gb"], "vs GB"),
        ("Tissue Spec", "Accuracy", comparison["tissue_spec"]["neural_accuracy"],
         comparison["tissue_spec"]["tabular_accuracy_gb"], "vs GB"),
        ("Structure", "Accuracy", comparison["structure"]["neural_accuracy"],
         comparison["structure"]["tabular_accuracy_gb"], "vs GB"),
        ("Function", "Accuracy", comparison["function"]["neural_accuracy"],
         comparison["function"]["tabular_accuracy_gb"], "vs GB"),
    ]

    for task, metric, neural, tabular_val, note in rows:
        delta = neural - tabular_val if not (np.isnan(neural) or np.isnan(tabular_val)) else float("nan")
        sign = "+" if delta > 0 else ""
        print(f"  {task:<23} {metric:<15} {neural:<10.4f} {tabular_val:<10.4f} {sign}{delta:.4f} {note}")

    return comparison


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate EditRNA-A3A checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (.pt)")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--skip_embeddings", action="store_true")
    parser.add_argument("--tabular_results", type=str, default=None)
    args = parser.parse_args()

    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = checkpoint_path.parent / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "eval.log", mode="w"),
        ],
    )

    logger.info("Loading checkpoint from %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config_dict = checkpoint["config"]

    # Reconstruct experiment config
    from train_editrna import ExperimentConfig, build_datasets_from_csvs
    config = ExperimentConfig(**{k: v for k, v in config_dict.items()
                                  if hasattr(ExperimentConfig, k)})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Build datasets
    logger.info("Building datasets...")
    train_dataset, val_dataset, test_dataset = build_datasets_from_csvs(config)
    eval_dataset = test_dataset if args.split == "test" else val_dataset
    logger.info("Evaluating on %s set: %d samples (%d pos, %d neg)",
                args.split, len(eval_dataset), eval_dataset.n_positive, eval_dataset.n_negative)

    # Create dataloader
    from torch.utils.data import DataLoader
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=apobec_collate_fn,
        num_workers=0,
    )

    # Build and load model
    logger.info("Building model (encoder=%s)...", config.encoder)
    from train_editrna import build_model
    model = build_model(config, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    logger.info("Model loaded from epoch %d (best %s=%.4f)",
                checkpoint.get("epoch", -1),
                config.primary_metric,
                checkpoint.get("best_metric", float("nan")))

    # Collect predictions
    logger.info("Running inference...")
    preds, targets, edit_embs, fused_embs, site_ids = collect_predictions(model, eval_loader, device)

    # Evaluate all tasks
    logger.info("Evaluating all 11 tasks...")
    results = evaluate_all_tasks(preds, targets)

    # Print results
    print("\n" + "=" * 80)
    print(f"EXPERIMENT 1 EVALUATION ({args.split.upper()} SET)")
    print("=" * 80)

    print("\n--- PRIMARY TASKS ---")
    print(f"  Binary:    AUROC={results['binary'].get('auroc', 'N/A'):.4f}  "
          f"AUPRC={results['binary'].get('auprc', 'N/A'):.4f}  "
          f"F1={results['binary'].get('f1', 'N/A'):.4f}")
    print(f"  Rate:      Spearman={results['rate'].get('spearman_rho', 'N/A'):.4f}  "
          f"MSE={results['rate'].get('mse', 'N/A'):.4f}  "
          f"(n={results['rate'].get('n', 0)})")
    print(f"  Enzyme:    Accuracy={results['enzyme'].get('accuracy', 'N/A'):.4f}  "
          f"Macro-F1={results['enzyme'].get('macro_f1', 'N/A'):.4f}  "
          f"(n={results['enzyme'].get('n', 0)})")

    print("\n--- SECONDARY TASKS ---")
    print(f"  Structure: Accuracy={results['structure'].get('accuracy', 'N/A'):.4f}  "
          f"Macro-F1={results['structure'].get('macro_f1', 'N/A'):.4f}  "
          f"(n={results['structure'].get('n', 0)})")
    print(f"  Tissue:    Accuracy={results['tissue_spec'].get('accuracy', 'N/A'):.4f}  "
          f"Macro-F1={results['tissue_spec'].get('macro_f1', 'N/A'):.4f}  "
          f"(n={results['tissue_spec'].get('n', 0)})")
    print(f"  N Tissues: Spearman={results['n_tissues'].get('spearman_rho', 'N/A'):.4f}  "
          f"(n={results['n_tissues'].get('n', 0)})")

    print("\n--- TERTIARY TASKS ---")
    print(f"  Function:  Accuracy={results['function'].get('accuracy', 'N/A'):.4f}  "
          f"Macro-F1={results['function'].get('macro_f1', 'N/A'):.4f}  "
          f"(n={results['function'].get('n', 0)})")
    print(f"  Conserv.:  AUROC={results['conservation'].get('auroc', 'N/A'):.4f}  "
          f"(n_pos={results['conservation'].get('n_positive', 0)}, "
          f"n_neg={results['conservation'].get('n_negative', 0)})")
    print(f"  Cancer:    AUROC={results['cancer'].get('auroc', 'N/A'):.4f}  "
          f"(n_pos={results['cancer'].get('n_positive', 0)}, "
          f"n_neg={results['cancer'].get('n_negative', 0)})")

    print("\n--- AUXILIARY TASKS ---")
    print(f"  HEK293:   Spearman={results['hek293'].get('spearman_rho', 'N/A'):.4f}  "
          f"(n={results['hek293'].get('n', 0)})")

    # Save results
    results_path = output_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, float)) else x)
    logger.info("Results saved to %s", results_path)

    # Task weights
    logger.info("Analyzing task weights...")
    weight_results = analyze_task_weights(model, output_dir)

    # Embedding visualization
    if not args.skip_embeddings:
        logger.info("Generating embedding visualizations...")
        visualize_embeddings(edit_embs, targets, site_ids, output_dir)

    # Comparison with tabular baseline
    tabular_path = args.tabular_results or (
        PROJECT_ROOT / "experiments" / "apobec" / "outputs" / "exp0_tabular" / "exp0_results.json"
    )
    comparison = compare_with_tabular(results, tabular_path, output_dir)

    logger.info("Evaluation complete. All results saved to %s", output_dir)
    return results


if __name__ == "__main__":
    main()
