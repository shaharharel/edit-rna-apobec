#!/usr/bin/env python
"""Generate publication-quality figures for the EditRNA-A3A paper.

Figures:
  1. Main results bar chart (AUROC for all models + RNAsee)
  2. Feature ablation waterfall chart
  3. Cross-dataset generalization heatmap (4x4 NxN matrix)
  4. Gate weight analysis (by dataset + by confidence)
  5. Embedding space UMAP/t-SNE colored by dataset and label
  6. Cross-tissue correlation heatmap

Usage:
    python experiments/apobec3a/generate_paper_figures.py
"""

import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "paper" / "figures"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "apobec3a" / "outputs"

logger = logging.getLogger(__name__)

# Consistent color palette
COLORS = {
    "editrna": "#2563eb",
    "diff_attention": "#7c3aed",
    "subtraction_mlp": "#059669",
    "structure_only": "#d97706",
    "concat_mlp": "#dc2626",
    "cross_attention": "#ea580c",
    "pooled_mlp": "#6b7280",
    "rnasee": "#9333ea",
}

DATASET_COLORS = {
    "Levanon": "#2563eb",
    "Asaoka": "#059669",
    "Sharma": "#dc2626",
    "Alqassim": "#d97706",
    "Baysal": "#7c3aed",
    "Tier2 Neg": "#9ca3af",
    "Tier3 Neg": "#6b7280",
}

MODEL_DISPLAY_NAMES = {
    "pooled_mlp": "Pooled MLP",
    "cross_attention": "Cross-Attn",
    "concat_mlp": "Concat MLP",
    "structure_only": "Structure Only",
    "subtraction_mlp": "Subtraction MLP",
    "diff_attention": "Diff-Attention",
    "editrna": "EditRNA-A3A\n(ours)",
}


def load_baseline_results():
    """Load all baseline results."""
    results = {}
    for model_dir in (RESULTS_DIR / "baselines").iterdir():
        if model_dir.is_dir():
            results_file = model_dir / "results.json"
            if results_file.exists():
                with open(results_file) as f:
                    data = json.load(f)
                results[model_dir.name] = data
    return results


def fig1_main_results(results):
    """Figure 1: Main results bar chart comparing all models."""
    fig, ax = plt.subplots(figsize=(10, 5))

    # Order models by performance
    model_order = [
        "pooled_mlp", "cross_attention", "concat_mlp",
        "structure_only", "subtraction_mlp", "diff_attention", "editrna"
    ]

    aurocs = []
    auprcs = []
    f1s = []
    names = []
    colors = []

    for model in model_order:
        if model in results:
            m = results[model].get("test_metrics", {})
            aurocs.append(m.get("auroc", 0))
            auprcs.append(m.get("auprc", 0))
            f1s.append(m.get("f1", 0))
            names.append(MODEL_DISPLAY_NAMES.get(model, model))
            colors.append(COLORS.get(model, "#6b7280"))

    x = np.arange(len(names))
    width = 0.25

    bars1 = ax.bar(x - width, aurocs, width, label="AUROC", color=colors, alpha=0.9)
    bars2 = ax.bar(x, auprcs, width, label="AUPRC", color=colors, alpha=0.6)
    bars3 = ax.bar(x + width, f1s, width, label="F1", color=colors, alpha=0.35)

    # Add RNAsee reference line
    ax.axhline(y=0.962, color="#9333ea", linestyle="--", linewidth=1.5,
               label="RNAsee (0.962)")

    # Value labels on AUROC bars
    for bar, val in zip(bars1, aurocs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax.set_ylim(0.65, 1.02)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Binary Editing Site Prediction: All Models", fontsize=13, fontweight="bold")
    ax.legend(loc="lower left", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig1_main_results.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / "fig1_main_results.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved Figure 1: Main results")


def fig2_ablation(gate_ablation_results):
    """Figure 2: Feature ablation waterfall chart."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ablation = gate_ablation_results["ablation"]
    configs = [
        ("Full Model", "full_model"),
        ("$-$ Edit\nEmbedding", "zero_edit_embedding"),
        ("$-$ Primary\nEncoder", "zero_primary_encoder"),
        ("$-$ Structure\n$\\Delta$", "zero_structure_delta"),
        ("$-$ Concordance", "zero_concordance"),
    ]

    names = [c[0] for c in configs]
    aurocs = [ablation[c[1]]["metrics"]["auroc"] for c in configs]
    deltas = [ablation[c[1]]["delta_auroc"] for c in configs]

    bar_colors = []
    for d in deltas:
        if d == 0:
            bar_colors.append("#2563eb")  # blue for full/no change
        elif d < -0.1:
            bar_colors.append("#dc2626")  # red for large drop
        elif d < 0:
            bar_colors.append("#d97706")  # yellow for small drop
        else:
            bar_colors.append("#059669")  # green for improvement

    bars = ax.bar(names, aurocs, color=bar_colors, edgecolor="white", linewidth=1.5)

    # Value labels
    for bar, val, delta in zip(bars, aurocs, deltas):
        label = f"{val:.3f}"
        if delta != 0:
            label += f"\n({delta:+.3f})"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                label, ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylim(0.55, 1.05)
    ax.set_ylabel("AUROC", fontsize=12)
    ax.set_title("Feature Ablation: Edit Embedding Is Critical", fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig2_ablation.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / "fig2_ablation.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved Figure 2: Ablation")


def fig3_cross_dataset(cross_dataset_results):
    """Figure 3: Cross-dataset generalization heatmap."""
    nxn = cross_dataset_results["nxn_matrix"]

    datasets = ["Levanon", "Asaoka", "Sharma", "Alqassim"]
    matrix = np.zeros((len(datasets), len(datasets)))

    for i, train_ds in enumerate(datasets):
        for j, test_ds in enumerate(datasets):
            key = f"{train_ds}\u2192{test_ds}"
            if key in nxn:
                matrix[i, j] = nxn[key].get("auroc", 0)

    fig, ax = plt.subplots(figsize=(7, 6))
    mask = np.zeros_like(matrix, dtype=bool)

    im = sns.heatmap(
        matrix,
        annot=True,
        fmt=".3f",
        xticklabels=datasets,
        yticklabels=datasets,
        cmap="RdYlGn",
        vmin=0.45,
        vmax=1.0,
        center=0.75,
        linewidths=1,
        linecolor="white",
        ax=ax,
        annot_kws={"fontsize": 11, "fontweight": "bold"},
    )

    ax.set_xlabel("Test Dataset", fontsize=12)
    ax.set_ylabel("Train Dataset", fontsize=12)
    ax.set_title("Cross-Dataset Generalization (AUROC)", fontsize=13, fontweight="bold")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig3_cross_dataset.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / "fig3_cross_dataset.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved Figure 3: Cross-dataset heatmap")


def fig4_gate_weights(gate_ablation_results):
    """Figure 4: Gate weight analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Gate weights by dataset
    ax = axes[0]
    gw = gate_ablation_results["gate_weights"]["by_dataset"]
    ds_order = sorted(gw.keys(), key=lambda k: gw[k]["Edit Embedding"]["mean"])

    ds_names = []
    edit_weights = []
    for ds in ds_order:
        ds_names.append(ds)
        edit_weights.append(gw[ds]["Edit Embedding"]["mean"] * 100)

    bars = ax.barh(ds_names, edit_weights,
                   color=[DATASET_COLORS.get(ds, "#6b7280") for ds in ds_names],
                   edgecolor="white", linewidth=1)

    for bar, val in zip(bars, edit_weights):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=9)

    ax.set_xlabel("Edit Embedding Gate Weight (%)", fontsize=11)
    ax.set_title("A) Gate Weight by Dataset", fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, max(edit_weights) * 1.3)

    # Panel B: Gate weights by confidence
    ax = axes[1]
    conf_data = gate_ablation_results["gate_weights"]["by_confidence"]
    conf_names = ["High Confidence\n(score > 0.9)", "Edge Cases\n(0.3-0.7)"]
    conf_edit = [
        conf_data["high_confidence"]["Edit Embedding"]["mean"] * 100,
        conf_data["edge_cases"]["Edit Embedding"]["mean"] * 100,
    ]
    conf_ns = [
        conf_data["high_confidence"]["n"],
        conf_data["edge_cases"]["n"],
    ]

    bars = ax.bar(conf_names, conf_edit, color=["#059669", "#dc2626"],
                  edgecolor="white", linewidth=1.5, width=0.5)
    for bar, val, n in zip(bars, conf_edit, conf_ns):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{val:.2f}%\n(n={n})", ha="center", va="bottom", fontsize=10,
                fontweight="bold")

    ax.set_ylabel("Edit Embedding Gate Weight (%)", fontsize=11)
    ax.set_title("B) Adaptive Gating by Confidence", fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add annotation for fold change
    ax.annotate(
        "133× increase\nfor edge cases",
        xy=(1, conf_edit[1]),
        xytext=(1.3, conf_edit[1] * 0.7),
        fontsize=10,
        color="#dc2626",
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.5),
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig4_gate_weights.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / "fig4_gate_weights.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved Figure 4: Gate weights")


def fig5_embedding_space():
    """Figure 5: Embedding space visualization."""
    # Load gate weights CSV for coloring
    gate_csv = RESULTS_DIR / "gate_ablation" / "gate_weights.csv"
    if not gate_csv.exists():
        logger.warning("Gate weights CSV not found, skipping Figure 5")
        return

    gate_df = pd.read_csv(gate_csv)

    # Load splits for dataset info
    splits_df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "splits_expanded.csv")
    splits_df = splits_df[["site_id", "dataset_source"]].drop_duplicates()

    DATASET_LABELS = {
        "advisor_c2t": "Levanon",
        "asaoka_2019": "Asaoka",
        "sharma_2015": "Sharma",
        "alqassim_2021": "Alqassim",
    "baysal_2016": "Baysal",
        "tier2_negative": "Tier2 Neg",
        "tier3_negative": "Tier3 Neg",
    }
    splits_df["dataset"] = splits_df["dataset_source"].map(DATASET_LABELS)

    # Load diff embeddings for t-SNE
    import torch
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    emb_dir = PROJECT_ROOT / "data" / "processed" / "embeddings"
    pooled_orig = torch.load(emb_dir / "rnafm_pooled.pt", weights_only=False)
    pooled_edited = torch.load(emb_dir / "rnafm_pooled_edited.pt", weights_only=False)

    # Get test site IDs from gate_df
    test_ids = gate_df["site_id"].tolist()

    # Compute diffs for test set
    diffs = []
    valid_ids = []
    for sid in test_ids:
        if sid in pooled_orig and sid in pooled_edited:
            diff = pooled_edited[sid] - pooled_orig[sid]
            diffs.append(diff.numpy())
            valid_ids.append(sid)

    if len(diffs) == 0:
        logger.warning("No valid embeddings for Figure 5")
        return

    X = np.stack(diffs)

    # PCA -> t-SNE (subsample for speed)
    n_sub = min(1000, len(X))
    rng = np.random.RandomState(42)
    idx = rng.choice(len(X), n_sub, replace=False)
    X_sub = X[idx]
    ids_sub = [valid_ids[i] for i in idx]

    pca = PCA(n_components=20, random_state=42)
    X_pca = pca.fit_transform(X_sub)

    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    X_tsne = tsne.fit_transform(X_pca)

    # Build dataframe for plotting
    plot_df = pd.DataFrame({
        "site_id": ids_sub,
        "tsne_1": X_tsne[:, 0],
        "tsne_2": X_tsne[:, 1],
    })

    # Merge with metadata
    label_map = dict(zip(gate_df["site_id"], gate_df["label"]))
    plot_df["label"] = plot_df["site_id"].map(label_map)
    plot_df["label_name"] = plot_df["label"].map({1: "Edited (positive)", 0: "Not edited (negative)"})

    ds_map = dict(zip(splits_df["site_id"], splits_df["dataset"]))
    plot_df["dataset"] = plot_df["site_id"].map(ds_map)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Colored by label
    ax = axes[0]
    for label_val, label_name, color, marker in [
        (1, "Edited", "#2563eb", "o"),
        (0, "Not edited", "#dc2626", "x"),
    ]:
        mask = plot_df["label"] == label_val
        ax.scatter(plot_df.loc[mask, "tsne_1"], plot_df.loc[mask, "tsne_2"],
                   c=color, label=label_name, s=15, alpha=0.6, marker=marker)
    ax.set_xlabel("t-SNE 1", fontsize=11)
    ax.set_ylabel("t-SNE 2", fontsize=11)
    ax.set_title("A) Edit Effect Embedding (by label)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel B: Colored by dataset
    ax = axes[1]
    for ds in ["Levanon", "Asaoka", "Sharma", "Alqassim", "Tier2 Neg", "Tier3 Neg"]:
        mask = plot_df["dataset"] == ds
        if mask.sum() == 0:
            continue
        ax.scatter(plot_df.loc[mask, "tsne_1"], plot_df.loc[mask, "tsne_2"],
                   c=DATASET_COLORS.get(ds, "#6b7280"), label=ds, s=15, alpha=0.6)
    ax.set_xlabel("t-SNE 1", fontsize=11)
    ax.set_ylabel("t-SNE 2", fontsize=11)
    ax.set_title("B) Edit Effect Embedding (by dataset)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig5_embedding_space.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / "fig5_embedding_space.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved Figure 5: Embedding space")


def fig6_cross_tissue():
    """Figure 6: Cross-tissue module correlation heatmap."""
    ct_file = RESULTS_DIR / "cross_tissue_disease" / "cross_tissue_disease_results.json"
    if not ct_file.exists():
        logger.warning("Cross-tissue results not found, skipping Figure 6")
        return

    with open(ct_file) as f:
        ct_data = json.load(f)

    # Get module correlation matrix (9 tissue modules)
    module_corr = ct_data.get("tissue_correlations", {}).get("module_correlation_matrix", {})
    if not module_corr:
        logger.warning("No module correlation matrix, skipping Figure 6")
        return

    # Build matrix from nested dict
    modules = list(module_corr.keys())
    n = len(modules)
    corr_matrix = np.zeros((n, n))
    for i, m1 in enumerate(modules):
        for j, m2 in enumerate(modules):
            corr_matrix[i, j] = module_corr[m1].get(m2, 0.0)

    # Also get tissue module membership for annotation
    tissue_modules = ct_data.get("tissue_correlations", {}).get("tissue_modules", {})
    module_sizes = [len(tissue_modules.get(m, [])) for m in modules]
    labels = [f"{m}\n(n={s})" for m, s in zip(modules, module_sizes)]

    # Also get mean editing rates per module from tissue_mean_rates
    tissue_rates = ct_data.get("tissue_mean_rates", {})
    module_rates = []
    for m in modules:
        member_tissues = tissue_modules.get(m, [])
        rates = [tissue_rates.get(t, {}).get("mean_rate", 0) for t in member_tissues]
        module_rates.append(np.mean(rates) if rates else 0)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7),
                             gridspec_kw={"width_ratios": [3, 1]})

    # Panel A: Module correlation heatmap
    ax = axes[0]
    sns.heatmap(
        corr_matrix,
        xticklabels=labels,
        yticklabels=labels,
        cmap="RdBu_r",
        center=0.5,
        vmin=0.0,
        vmax=1.0,
        ax=ax,
        linewidths=1,
        linecolor="white",
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 8},
    )
    ax.set_title("A) Tissue Module Editing Correlation", fontsize=12, fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)

    # Panel B: Mean editing rate per module
    ax = axes[1]
    sorted_idx = np.argsort(module_rates)[::-1]
    sorted_modules = [modules[i] for i in sorted_idx]
    sorted_rates = [module_rates[i] for i in sorted_idx]
    colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(sorted_modules)))
    bars = ax.barh(range(len(sorted_modules)), sorted_rates, color=colors)
    ax.set_yticks(range(len(sorted_modules)))
    ax.set_yticklabels(sorted_modules, fontsize=9)
    ax.set_xlabel("Mean Editing Rate", fontsize=10)
    ax.set_title("B) Module Editing Rate", fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, rate in zip(bars, sorted_rates):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f"{rate:.2f}", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig6_cross_tissue.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / "fig6_cross_tissue.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved Figure 6: Cross-tissue module heatmap")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load results
    logger.info("Loading results...")
    baseline_results = load_baseline_results()

    gate_ablation_file = RESULTS_DIR / "gate_ablation" / "gate_ablation_results.json"
    with open(gate_ablation_file) as f:
        gate_ablation = json.load(f)

    cross_dataset_file = RESULTS_DIR / "cross_dataset" / "cross_dataset_results.json"
    with open(cross_dataset_file) as f:
        cross_dataset = json.load(f)

    # Generate figures
    logger.info("Generating figures...")

    fig1_main_results(baseline_results)
    fig2_ablation(gate_ablation)
    fig3_cross_dataset(cross_dataset)
    fig4_gate_weights(gate_ablation)
    fig5_embedding_space()
    fig6_cross_tissue()

    logger.info("All figures saved to %s", OUTPUT_DIR)


if __name__ == "__main__":
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    main()
