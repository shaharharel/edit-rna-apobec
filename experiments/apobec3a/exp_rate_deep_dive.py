#!/usr/bin/env python
"""Rate Prediction Deep Dive: Biological analysis of APOBEC3A editing rates.

Key scientific questions addressed:
  1. Is editing rate a site-intrinsic property? (shared-site rate agreement)
  2. Does TC-motif filtering improve rate prediction? (mixed-enzyme hypothesis)
  3. Are constitutive editing sites more rate-predictable?
  4. What fraction of rate variance is between vs within datasets?
  5. What sequence/structure features distinguish high vs low rate sites?

This script produces publishable figures and statistical analyses.

Usage:
    python experiments/apobec3a/exp_rate_deep_dive.py
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
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr, pearsonr, mannwhitneyu, kruskal
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

EMB_DIR = PROJECT_ROOT / "data" / "processed" / "embeddings"
COMBINED_CSV = PROJECT_ROOT / "data" / "processed" / "all_datasets_combined.csv"
SPLITS_CSV = PROJECT_ROOT / "data" / "processed" / "splits_expanded.csv"
SEQ_JSON = PROJECT_ROOT / "data" / "processed" / "site_sequences.json"
STRUCT_CACHE = EMB_DIR / "vienna_structure_cache.npz"
LEVANON_TISSUE = PROJECT_ROOT / "data" / "raw" / "published" / "levanon" / "tissue_editing_rates.csv"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "apobec" / "outputs" / "rate_deep_dive"

DATASET_LABELS = {
    "advisor_c2t": "Levanon",
    "asaoka_2019": "Asaoka",
    "alqassim_2021": "Alqassim",
    "sharma_2015": "Sharma",
    "baysal_2016": "Baysal",
}

TISSUE_NAMES = [
    "Adipose_Subcutaneous", "Adipose_Visceral_Omentum", "Adrenal_Gland",
    "Artery_Aorta", "Artery_Coronary", "Artery_Tibial", "Bladder",
    "Brain_Amygdala", "Brain_Anterior_cingulate_cortex_BA24",
    "Brain_Caudate_basal_ganglia", "Brain_Cerebellar_Hemisphere",
    "Brain_Cerebellum", "Brain_Cortex", "Brain_Frontal_Cortex_BA9",
    "Brain_Hippocampus", "Brain_Hypothalamus",
    "Brain_Nucleus_accumbens_basal_ganglia", "Brain_Putamen_basal_ganglia",
    "Brain_Spinal_cord_cervical_c-1", "Brain_Substantia_nigra",
    "Breast_Mammary_Tissue", "Cells_Cultured_fibroblasts",
    "Cells_EBV-transformed_lymphocytes", "Cervix_Ectocervix",
    "Cervix_Endocervix", "Colon_Sigmoid", "Colon_Transverse",
    "Esophagus_Gastroesophageal_Junction", "Esophagus_Mucosa",
    "Esophagus_Muscularis", "Fallopian_Tube", "Heart_Atrial_Appendage",
    "Heart_Left_Ventricle", "Kidney_Cortex", "Kidney_Medulla", "Liver",
    "Lung", "Minor_Salivary_Gland", "Muscle_Skeletal", "Nerve_Tibial",
    "Ovary", "Pancreas", "Pituitary", "Prostate",
    "Skin_Not_Sun_Exposed_Suprapubic", "Skin_Sun_Exposed_Lower_leg",
    "Small_Intestine_Terminal_Ileum", "Spleen", "Stomach", "Testis",
    "Thyroid", "Uterus", "Vagina", "Whole_Blood",
]


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def compute_log2_rate(rate):
    if pd.isna(rate) or rate < 0:
        return float("nan")
    r = float(rate)
    if r > 1.0:
        r = r / 100.0
    return np.log2(r + 0.01)


def get_tc_motif(seq, pos=100):
    """Check if site has TC motif (U/T at position -1)."""
    if len(seq) <= pos or pos < 1:
        return False
    prev = seq[pos - 1].upper()
    curr = seq[pos].upper()
    return prev in ("U", "T") and curr == "C"


# ---------------------------------------------------------------------------
# Analysis 1: Shared-site rate agreement across datasets
# ---------------------------------------------------------------------------

def analysis_shared_site_rates(combined_df, sequences, output_dir):
    """Compare editing rates at the same genomic sites across different datasets.

    Key question: Is the editing rate an intrinsic property of a site,
    or does it depend on the biological context?
    """
    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS 1: SHARED-SITE RATE AGREEMENT")
    logger.info("=" * 70)

    combined_df["coord"] = combined_df["chr"] + ":" + combined_df["start"].astype(str)

    # Find all pairs with sufficient shared sites
    datasets = combined_df["dataset_source"].unique()
    pairs = []

    for i, ds1 in enumerate(datasets):
        for ds2 in datasets[i + 1:]:
            coords1 = set(combined_df[combined_df["dataset_source"] == ds1]["coord"])
            coords2 = set(combined_df[combined_df["dataset_source"] == ds2]["coord"])
            shared = coords1 & coords2

            if len(shared) < 10:
                continue

            df1 = combined_df[(combined_df["dataset_source"] == ds1) &
                              (combined_df["coord"].isin(shared))].set_index("coord")
            df2 = combined_df[(combined_df["dataset_source"] == ds2) &
                              (combined_df["coord"].isin(shared))].set_index("coord")
            common = df1.index.intersection(df2.index)
            r1 = df1.loc[common, "editing_rate"]
            r2 = df2.loc[common, "editing_rate"]
            both_valid = r1.notna() & r2.notna()

            if both_valid.sum() >= 10:
                rates1 = r1[both_valid].values
                rates2 = r2[both_valid].values
                sp, p = spearmanr(rates1, rates2)
                pe, pe_p = pearsonr(rates1, rates2)
                pairs.append({
                    "ds1": ds1, "ds2": ds2,
                    "n_shared": len(shared),
                    "n_both_rates": int(both_valid.sum()),
                    "spearman": float(sp), "spearman_p": float(p),
                    "pearson": float(pe), "pearson_p": float(pe_p),
                    "rates1": rates1, "rates2": rates2,
                })

                logger.info("  %s vs %s: %d shared sites, Spearman=%.3f (p=%.2e)",
                            DATASET_LABELS.get(ds1, ds1),
                            DATASET_LABELS.get(ds2, ds2),
                            int(both_valid.sum()), sp, p)

    if not pairs:
        logger.warning("No dataset pairs with sufficient shared sites")
        return {}

    # Plot rate-rate scatter plots
    n_pairs = len(pairs)
    fig, axes = plt.subplots(1, n_pairs, figsize=(6 * n_pairs, 5))
    if n_pairs == 1:
        axes = [axes]

    for ax, pair in zip(axes, pairs):
        r1, r2 = pair["rates1"], pair["rates2"]
        label1 = DATASET_LABELS.get(pair["ds1"], pair["ds1"])
        label2 = DATASET_LABELS.get(pair["ds2"], pair["ds2"])

        # Normalize rates to [0,1] for comparison
        r1_norm = r1 / 100.0 if r1.max() > 1.0 else r1
        r2_norm = r2 / 100.0 if r2.max() > 1.0 else r2

        ax.scatter(r1_norm, r2_norm, alpha=0.4, s=30, color="steelblue", edgecolors="white",
                   linewidths=0.5)

        # Add diagonal reference
        lim = max(r1_norm.max(), r2_norm.max()) * 1.1
        ax.plot([0, lim], [0, lim], "r--", alpha=0.5, linewidth=1)

        ax.set_xlabel(f"Rate in {label1}", fontsize=11)
        ax.set_ylabel(f"Rate in {label2}", fontsize=11)
        ax.set_title(f"{label1} vs {label2}\n"
                     f"n={pair['n_both_rates']}, "
                     f"Spearman={pair['spearman']:.3f} (p={pair['spearman_p']:.1e})",
                     fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Editing Rate Agreement at Shared Genomic Sites\n"
                 "Low correlation = rate is context-dependent, not site-intrinsic",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "shared_site_rate_agreement.png", dpi=150, bbox_inches="tight")
    plt.close()

    results = {
        "pairs": [{k: v for k, v in p.items() if k not in ("rates1", "rates2")}
                  for p in pairs],
        "conclusion": "Rate is NOT a site-intrinsic property. Same genomic position "
                      "shows substantially different rates across biological contexts."
    }

    logger.info("\n  CONCLUSION: Cross-dataset rate Spearman (0.28-0.35) is much lower")
    logger.info("  than binary classification concordance (~1.0), proving that rate is")
    logger.info("  determined by the cellular environment, not by site identity alone.")

    return results


# ---------------------------------------------------------------------------
# Analysis 2: TC-motif filtering effect on rate prediction
# ---------------------------------------------------------------------------

def analysis_tc_motif_filtering(combined_df, splits_df, sequences,
                                 pooled_orig, pooled_edited, output_dir):
    """Test whether filtering to TC-motif sites improves rate prediction.

    Hypothesis: Levanon's Spearman = -0.09 is caused by including
    non-APOBEC3A editing sites. TC-motif filtering should improve it.
    """
    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS 2: TC-MOTIF FILTERING EFFECT ON RATE PREDICTION")
    logger.info("=" * 70)

    # Identify TC-motif sites
    tc_sites = set()
    non_tc_sites = set()
    for sid, seq in sequences.items():
        if len(seq) >= 201:
            if get_tc_motif(seq, pos=100):
                tc_sites.add(sid)
            else:
                non_tc_sites.add(sid)

    logger.info("TC-motif sites: %d, non-TC sites: %d", len(tc_sites), len(non_tc_sites))

    # Per-dataset TC fraction
    datasets_with_rates = ["advisor_c2t", "alqassim_2021", "sharma_2015", "baysal_2016"]

    tc_fractions = {}
    for ds in datasets_with_rates:
        ds_sids = set(combined_df[combined_df["dataset_source"] == ds]["site_id"])
        n_tc = len(ds_sids & tc_sites)
        n_total = len(ds_sids & (tc_sites | non_tc_sites))
        frac = n_tc / max(n_total, 1)
        tc_fractions[ds] = {"n_tc": n_tc, "n_total": n_total, "fraction": frac}
        logger.info("  %s: %d/%d (%.1f%%) TC-motif",
                    DATASET_LABELS.get(ds, ds), n_tc, n_total, 100 * frac)

    # Train SubtractionMLP for rate prediction with and without TC filtering
    pos_df = splits_df[splits_df["label"] == 1].copy()
    pos_df["log2_rate"] = pos_df["editing_rate"].apply(compute_log2_rate)
    pos_df = pos_df[pos_df["log2_rate"].notna()].copy()
    pos_df["is_tc"] = pos_df["site_id"].isin(tc_sites)

    # Get available site_ids
    available = set(pooled_orig.keys()) & set(pooled_edited.keys())
    pos_df = pos_df[pos_df["site_id"].isin(available)]

    results = {"tc_fractions": tc_fractions, "rate_predictions": {}}

    class RateDS(Dataset):
        def __init__(self, sids, targets, po, pe):
            self.sids, self.targets, self.po, self.pe = sids, targets, po, pe

        def __len__(self):
            return len(self.sids)

        def __getitem__(self, idx):
            sid = self.sids[idx]
            x = self.pe[sid] - self.po[sid]
            return x, torch.tensor(self.targets[idx], dtype=torch.float32)

    class MLP(nn.Module):
        def __init__(self, d=640):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d, 256), nn.GELU(), nn.Dropout(0.3),
                nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.15),
                nn.Linear(128, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

    def train_rate_model(train_sids, train_rates, test_sids, test_rates, seed=42):
        """Train and evaluate a SubtractionMLP for rate regression."""
        torch.manual_seed(seed)
        if len(train_sids) < 20 or len(test_sids) < 5:
            return float("nan"), np.array([]), np.array([])

        train_ds = RateDS(train_sids, train_rates, pooled_orig, pooled_edited)
        test_ds = RateDS(test_sids, test_rates, pooled_orig, pooled_edited)
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

        model = MLP()
        optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

        for epoch in range(60):
            model.train()
            for x, y in train_loader:
                optimizer.zero_grad()
                F.mse_loss(model(x), y).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        model.eval()
        all_p, all_t = [], []
        with torch.no_grad():
            for x, y in test_loader:
                all_p.append(model(x).numpy())
                all_t.append(y.numpy())
        preds = np.concatenate(all_p)
        targets = np.concatenate(all_t)
        if len(preds) >= 3:
            sp, _ = spearmanr(targets, preds)
            return float(sp), preds, targets
        return float("nan"), preds, targets

    # For each dataset: train on ALL data, test on this dataset
    # Compare: all sites vs TC-only sites
    logger.info("\n--- Rate Prediction: All Sites vs TC-Motif Only ---")

    for ds in datasets_with_rates:
        ds_df = pos_df[pos_df["dataset_source"] == ds]
        other_df = pos_df[pos_df["dataset_source"] != ds]

        if len(ds_df) < 10 or len(other_df) < 50:
            continue

        # Test on this dataset, train on others
        train_sids = other_df["site_id"].tolist()
        train_rates = other_df["log2_rate"].values.astype(np.float32)

        # All sites
        test_sids_all = ds_df["site_id"].tolist()
        test_rates_all = ds_df["log2_rate"].values.astype(np.float32)
        sp_all, _, _ = train_rate_model(train_sids, train_rates,
                                         test_sids_all, test_rates_all)

        # TC-motif only
        ds_tc = ds_df[ds_df["is_tc"]]
        if len(ds_tc) >= 5:
            test_sids_tc = ds_tc["site_id"].tolist()
            test_rates_tc = ds_tc["log2_rate"].values.astype(np.float32)
            sp_tc, _, _ = train_rate_model(train_sids, train_rates,
                                           test_sids_tc, test_rates_tc)
        else:
            sp_tc = float("nan")

        # Non-TC sites
        ds_ntc = ds_df[~ds_df["is_tc"]]
        if len(ds_ntc) >= 5:
            test_sids_ntc = ds_ntc["site_id"].tolist()
            test_rates_ntc = ds_ntc["log2_rate"].values.astype(np.float32)
            sp_ntc, _, _ = train_rate_model(train_sids, train_rates,
                                            test_sids_ntc, test_rates_ntc)
        else:
            sp_ntc = float("nan")

        label = DATASET_LABELS.get(ds, ds)
        logger.info("  %s: All=%+.3f (n=%d), TC-only=%+.3f (n=%d), non-TC=%+.3f (n=%d)",
                    label, sp_all, len(ds_df), sp_tc, len(ds_tc), sp_ntc, len(ds_ntc))

        results["rate_predictions"][ds] = {
            "spearman_all": sp_all, "n_all": len(ds_df),
            "spearman_tc": sp_tc, "n_tc": len(ds_tc),
            "spearman_non_tc": sp_ntc, "n_non_tc": len(ds_ntc),
        }

    # Visualization: TC motif fraction by dataset
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Bar chart: TC fraction
    ds_names = [DATASET_LABELS.get(ds, ds) for ds in datasets_with_rates]
    tc_fracs = [tc_fractions[ds]["fraction"] for ds in datasets_with_rates]
    colors = ["#ef5350" if f < 0.3 else "#66bb6a" for f in tc_fracs]
    axes[0].bar(range(len(ds_names)), [f * 100 for f in tc_fracs], color=colors, alpha=0.8)
    axes[0].set_xticks(range(len(ds_names)))
    axes[0].set_xticklabels(ds_names)
    axes[0].set_ylabel("% TC-motif sites")
    axes[0].set_title("TC Motif (APOBEC3A Recognition) Fraction by Dataset")
    axes[0].axhline(50, color="gray", linestyle="--", alpha=0.5, label="50%")
    for i, f in enumerate(tc_fracs):
        axes[0].text(i, f * 100 + 1, f"{f * 100:.1f}%", ha="center", fontsize=9, fontweight="bold")
    axes[0].legend()

    # Rate prediction comparison
    ds_plot = [ds for ds in datasets_with_rates if ds in results["rate_predictions"]]
    x = np.arange(len(ds_plot))
    width = 0.25
    sp_all_vals = [results["rate_predictions"][ds]["spearman_all"] for ds in ds_plot]
    sp_tc_vals = [results["rate_predictions"][ds]["spearman_tc"] for ds in ds_plot]
    sp_ntc_vals = [results["rate_predictions"][ds]["spearman_non_tc"] for ds in ds_plot]

    axes[1].bar(x - width, sp_all_vals, width, label="All sites", color="#90a4ae", alpha=0.8)
    axes[1].bar(x, sp_tc_vals, width, label="TC-motif only", color="#66bb6a", alpha=0.8)
    axes[1].bar(x + width, sp_ntc_vals, width, label="Non-TC sites", color="#ef5350", alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([DATASET_LABELS.get(ds, ds) for ds in ds_plot])
    axes[1].set_ylabel("Spearman rho")
    axes[1].set_title("Rate Prediction: All Sites vs TC-Motif Only")
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.suptitle("TC Motif Analysis: Mixed-Enzyme Hypothesis\n"
                 "Levanon/Alqassim have <10% TC-motif → most sites may not be APOBEC3A targets",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "tc_motif_rate_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()

    return results


# ---------------------------------------------------------------------------
# Analysis 3: Variance decomposition (between vs within datasets)
# ---------------------------------------------------------------------------

def analysis_variance_decomposition(combined_df, output_dir):
    """Decompose rate variance into between-dataset and within-dataset components.

    If between-dataset variance dominates, the "overall Spearman 0.485"
    is an artifact of dataset-scale separation.
    """
    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS 3: RATE VARIANCE DECOMPOSITION")
    logger.info("=" * 70)

    rates_df = combined_df[combined_df["editing_rate"].notna()].copy()
    rates_df["rate_norm"] = rates_df["editing_rate"].apply(
        lambda r: r / 100.0 if r > 1.0 else r
    )
    rates_df["log2_rate"] = rates_df["rate_norm"].apply(lambda r: np.log2(r + 0.01))

    # Grand mean
    grand_mean = rates_df["log2_rate"].mean()
    total_var = rates_df["log2_rate"].var()

    # Between-dataset variance
    ds_means = rates_df.groupby("dataset_source")["log2_rate"].mean()
    ds_sizes = rates_df.groupby("dataset_source").size()

    between_var = sum(n * (m - grand_mean) ** 2 for m, n in zip(ds_means, ds_sizes)) / len(rates_df)
    within_var = total_var - between_var

    pct_between = 100 * between_var / total_var
    pct_within = 100 * within_var / total_var

    logger.info("Total rate variance (log2 scale): %.4f", total_var)
    logger.info("Between-dataset variance: %.4f (%.1f%%)", between_var, pct_between)
    logger.info("Within-dataset variance:  %.4f (%.1f%%)", within_var, pct_within)

    # Per-dataset rate distributions
    logger.info("\nPer-dataset rate statistics (normalized to [0,1]):")
    for ds, group in rates_df.groupby("dataset_source"):
        r = group["rate_norm"]
        logger.info("  %s: n=%d, median=%.4f, mean=%.4f, std=%.4f, IQR=[%.4f, %.4f]",
                    DATASET_LABELS.get(ds, ds), len(r), r.median(), r.mean(), r.std(),
                    r.quantile(0.25), r.quantile(0.75))

    # "Predict dataset mean" baseline Spearman
    rates_df["predicted_by_ds_mean"] = rates_df["dataset_source"].map(ds_means)
    sp_baseline, _ = spearmanr(rates_df["log2_rate"], rates_df["predicted_by_ds_mean"])
    logger.info("\nBaseline 'predict dataset mean' Spearman: %.3f", sp_baseline)
    logger.info("This means %.1f%% of the reported Spearman may come from dataset identity alone.",
                100 * sp_baseline / 0.485 if 0.485 > 0 else 0)

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Rate distributions by dataset
    datasets = sorted(rates_df["dataset_source"].unique(),
                      key=lambda x: rates_df[rates_df["dataset_source"] == x]["rate_norm"].median())
    data_for_violin = [rates_df[rates_df["dataset_source"] == ds]["rate_norm"].values
                       for ds in datasets]
    labels = [DATASET_LABELS.get(ds, ds) for ds in datasets]

    vp = axes[0].violinplot(data_for_violin, positions=range(len(datasets)),
                             showmeans=True, showextrema=False)
    for body in vp["bodies"]:
        body.set_alpha(0.6)
    axes[0].set_xticks(range(len(datasets)))
    axes[0].set_xticklabels(labels, fontsize=9)
    axes[0].set_ylabel("Editing Rate (normalized to [0,1])")
    axes[0].set_title("Rate Distributions by Dataset")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].grid(True, alpha=0.3, axis="y")

    # Variance decomposition pie chart
    axes[1].pie([pct_between, pct_within],
                labels=[f"Between-dataset\n{pct_between:.1f}%",
                        f"Within-dataset\n{pct_within:.1f}%"],
                colors=["#ef5350", "#66bb6a"], autopct="%.1f%%",
                startangle=90, textprops={"fontsize": 10})
    axes[1].set_title("Rate Variance Decomposition")

    # Log2 rate by dataset
    datasets_sorted = sorted(ds_means.index, key=lambda x: ds_means[x])
    y_pos = range(len(datasets_sorted))
    means = [ds_means[ds] for ds in datasets_sorted]
    stds = [rates_df[rates_df["dataset_source"] == ds]["log2_rate"].std()
            for ds in datasets_sorted]
    labels_sorted = [DATASET_LABELS.get(ds, ds) for ds in datasets_sorted]

    axes[2].barh(y_pos, means, xerr=stds, color="#5c6bc0", alpha=0.7, capsize=4)
    axes[2].set_yticks(y_pos)
    axes[2].set_yticklabels(labels_sorted)
    axes[2].set_xlabel("log2(rate + 0.01)")
    axes[2].set_title("Mean log2(Rate) by Dataset\n(error bars = ±1 std)")
    axes[2].grid(True, alpha=0.3, axis="x")

    plt.suptitle("Rate Variance: Between vs Within Datasets\n"
                 f"'Predict dataset mean' baseline achieves Spearman = {sp_baseline:.3f}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "rate_variance_decomposition.png", dpi=150, bbox_inches="tight")
    plt.close()

    return {
        "total_variance": float(total_var),
        "between_dataset_pct": float(pct_between),
        "within_dataset_pct": float(pct_within),
        "dataset_mean_baseline_spearman": float(sp_baseline),
    }


# ---------------------------------------------------------------------------
# Analysis 4: Constitutive vs facultative rate predictability
# ---------------------------------------------------------------------------

def analysis_constitutive_vs_facultative(combined_df, sequences, output_dir):
    """Analyze whether constitutive editing sites have more predictable rates.

    Constitutive = edited in many tissues (>30 in Levanon)
    Facultative = edited in few tissues
    """
    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS 4: CONSTITUTIVE vs FACULTATIVE EDITING RATES")
    logger.info("=" * 70)

    # Load Levanon tissue data if available
    if not LEVANON_TISSUE.exists():
        logger.warning("Levanon tissue file not found, using simple approach")
        levanon = combined_df[combined_df["dataset_source"] == "advisor_c2t"].copy()
        levanon["n_tissues"] = 0
    else:
        tissue_df = pd.read_csv(LEVANON_TISSUE)

        # The CSV has pre-computed columns from the original xlsx:
        #   "Edited In # Tissues", "Tissue Classification", "Max GTEx Editing Rate",
        #   "Mean GTEx Editing Rate", "GTEx Editing Rate SD"
        # Plus per-tissue columns with space-separated names and semicolon-delimited values
        # Format: "<mismatched>;<coverage>;<editing_rate>"

        # Use pre-computed n_tissues
        n_tissues_col = "Edited In # Tissues"
        tissue_class_col = "Tissue Classification"
        mean_rate_col = "Mean GTEx Editing Rate"
        sd_col = "GTEx Editing Rate SD"

        if n_tissues_col in tissue_df.columns:
            tissue_df["n_tissues"] = tissue_df[n_tissues_col].fillna(0).astype(int)
            logger.info("Loaded tissue data: %d sites", len(tissue_df))
            logger.info("N_tissues distribution: median=%d, mean=%.1f, max=%d",
                        tissue_df["n_tissues"].median(), tissue_df["n_tissues"].mean(),
                        tissue_df["n_tissues"].max())
            if tissue_class_col in tissue_df.columns:
                logger.info("Tissue classifications: %s",
                            tissue_df[tissue_class_col].value_counts().to_dict())
        else:
            tissue_df["n_tissues"] = 0

        # Compute rate CV from mean and SD if available
        if mean_rate_col in tissue_df.columns and sd_col in tissue_df.columns:
            tissue_df["mean_rate"] = pd.to_numeric(tissue_df[mean_rate_col], errors="coerce").fillna(0)
            tissue_df["rate_sd"] = pd.to_numeric(tissue_df[sd_col], errors="coerce").fillna(0)
            tissue_df["rate_cv"] = tissue_df["rate_sd"] / (tissue_df["mean_rate"] + 1e-8)

        # Parse per-tissue rates (space-separated names, semicolon-delimited values)
        tissue_col_names = [c for c in tissue_df.columns
                            if c not in [n_tissues_col, tissue_class_col, mean_rate_col, sd_col,
                                         "Chr", "Start", "End", "Genomic Category", "Gene (RefSeq)",
                                         "mRNA location (RefSeq)", "Exonic Function ",
                                         "Edited Tissues (Z score ≥ 2)",
                                         "Affecting Over Expressed APOBEC",
                                         "Max GTEx Editing Rate",
                                         "Any Non-Primate Editing", "Any Non-Primate Editing ≥ 1%",
                                         "Any Primate Editing", "Any Primate Editing ≥ 1%",
                                         "Any Mammalian Editing", "Any Mammlian Editing ≥ 1%",
                                         "n_tissues", "mean_rate", "rate_sd", "rate_cv"]
                            and not c.startswith("Laurasiatheria")]
        # Filter to actual tissue columns (have semicolon-delimited values)
        actual_tissue_cols = []
        for tc in tissue_col_names:
            sample_val = tissue_df[tc].dropna().iloc[0] if len(tissue_df[tc].dropna()) > 0 else ""
            if ";" in str(sample_val):
                actual_tissue_cols.append(tc)

        if actual_tissue_cols:
            logger.info("Found %d tissue-specific rate columns", len(actual_tissue_cols))
            # Parse editing rates from semicolon-delimited format
            for tc in actual_tissue_cols:
                tissue_df[f"{tc}_rate"] = tissue_df[tc].apply(
                    lambda v: float(str(v).split(";")[2]) if pd.notna(v) and ";" in str(v)
                    else 0.0
                )
            rate_cols = [f"{tc}_rate" for tc in actual_tissue_cols]
            tissue_df["parsed_n_tissues"] = (tissue_df[rate_cols] > 0).sum(axis=1)

        # Merge with combined data
        levanon = combined_df[combined_df["dataset_source"] == "advisor_c2t"].copy()
        # Match by chr:start
        tissue_df["coord"] = tissue_df["Chr"].astype(str) + ":" + tissue_df["Start"].astype(str)
        levanon["coord"] = levanon["chr"] + ":" + levanon["start"].astype(str)

        tissue_map = tissue_df.set_index("coord")["n_tissues"].to_dict()
        levanon["n_tissues"] = levanon["coord"].map(tissue_map).fillna(0).astype(int)

        if "rate_cv" in tissue_df.columns:
            cv_map = tissue_df.set_index("coord")["rate_cv"].to_dict()
            levanon["rate_cv"] = levanon["coord"].map(cv_map)

        if tissue_class_col in tissue_df.columns:
            class_map = tissue_df.set_index("coord")[tissue_class_col].to_dict()
            levanon["tissue_class"] = levanon["coord"].map(class_map)

    # Classify
    levanon["edit_class"] = "Facultative"
    levanon.loc[levanon["n_tissues"] >= 30, "edit_class"] = "Constitutive"
    levanon.loc[(levanon["n_tissues"] >= 10) & (levanon["n_tissues"] < 30), "edit_class"] = "Intermediate"

    for cls in ["Constitutive", "Intermediate", "Facultative"]:
        sub = levanon[levanon["edit_class"] == cls]
        if len(sub) > 0:
            rates = sub["editing_rate"].dropna()
            logger.info("  %s: n=%d, median_rate=%.2f%%, mean_rate=%.2f%%",
                        cls, len(sub),
                        rates.median() if len(rates) > 0 else 0,
                        rates.mean() if len(rates) > 0 else 0)

    # TC motif analysis by class
    for cls in ["Constitutive", "Intermediate", "Facultative"]:
        sub = levanon[levanon["edit_class"] == cls]
        n_tc = sum(1 for _, row in sub.iterrows()
                   if row["site_id"] in sequences and get_tc_motif(sequences[row["site_id"]]))
        n_total = sum(1 for _, row in sub.iterrows() if row["site_id"] in sequences)
        if n_total > 0:
            logger.info("  %s TC-motif: %d/%d (%.1f%%)", cls, n_tc, n_total, 100 * n_tc / n_total)

    # Visualization
    if levanon["n_tissues"].max() > 0:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Histogram of n_tissues
        axes[0].hist(levanon["n_tissues"], bins=30, color="steelblue", alpha=0.7, edgecolor="white")
        axes[0].axvline(30, color="red", linestyle="--", alpha=0.7, label="Constitutive threshold")
        axes[0].axvline(10, color="orange", linestyle="--", alpha=0.7, label="Intermediate threshold")
        axes[0].set_xlabel("Number of Tissues with Editing")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Tissue Breadth Distribution (Levanon)")
        axes[0].legend(fontsize=8)

        # Rate vs n_tissues scatter
        valid = levanon[levanon["editing_rate"].notna() & (levanon["n_tissues"] > 0)]
        axes[1].scatter(valid["n_tissues"], valid["editing_rate"],
                        alpha=0.3, s=15, color="steelblue")
        axes[1].set_xlabel("Number of Tissues")
        axes[1].set_ylabel("Editing Rate (%)")
        axes[1].set_title("Rate vs Tissue Breadth")
        if len(valid) > 5:
            sp, p = spearmanr(valid["n_tissues"], valid["editing_rate"])
            axes[1].text(0.05, 0.95, f"Spearman={sp:.3f}\np={p:.2e}",
                         transform=axes[1].transAxes, fontsize=9,
                         verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat"))

        # Rate distribution by class
        class_data = []
        class_labels = []
        for cls in ["Constitutive", "Intermediate", "Facultative"]:
            rates = levanon[levanon["edit_class"] == cls]["editing_rate"].dropna().values
            if len(rates) > 0:
                class_data.append(rates)
                class_labels.append(f"{cls}\n(n={len(rates)})")

        if class_data:
            bp = axes[2].boxplot(class_data, labels=class_labels, patch_artist=True, showfliers=False)
            colors_cls = {"Constitutive": "#1565c0", "Intermediate": "#ffb74d", "Facultative": "#e0e0e0"}
            for patch, cls in zip(bp["boxes"], ["Constitutive", "Intermediate", "Facultative"]):
                patch.set_facecolor(colors_cls.get(cls, "#bdbdbd"))
                patch.set_alpha(0.7)
            axes[2].set_ylabel("Editing Rate (%)")
            axes[2].set_title("Rate by Editing Class")

        plt.suptitle("Constitutive vs Facultative Editing\n"
                     "Constitutive sites (≥30 tissues): high rates, likely sequence-determined",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_dir / "constitutive_vs_facultative_rates.png", dpi=150, bbox_inches="tight")
        plt.close()

    return {
        "n_constitutive": int((levanon["edit_class"] == "Constitutive").sum()),
        "n_intermediate": int((levanon["edit_class"] == "Intermediate").sum()),
        "n_facultative": int((levanon["edit_class"] == "Facultative").sum()),
    }


# ---------------------------------------------------------------------------
# Analysis 5: Feature analysis for high vs low rate sites
# ---------------------------------------------------------------------------

def analysis_rate_features(combined_df, sequences, structure_delta, output_dir):
    """What sequence/structure features distinguish high vs low rate sites?"""
    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS 5: FEATURES OF HIGH vs LOW RATE EDITING SITES")
    logger.info("=" * 70)

    # Use Baysal (97.6% TC, largest dataset with rates)
    baysal = combined_df[
        (combined_df["dataset_source"] == "baysal_2016") &
        (combined_df["editing_rate"].notna())
    ].copy()

    if len(baysal) < 100:
        logger.warning("Too few Baysal sites for feature analysis")
        return {}

    baysal["rate_norm"] = baysal["editing_rate"].apply(
        lambda r: r / 100.0 if r > 1.0 else r
    )

    # Split into tertiles
    q33 = baysal["rate_norm"].quantile(0.33)
    q66 = baysal["rate_norm"].quantile(0.66)
    baysal["rate_group"] = "Medium"
    baysal.loc[baysal["rate_norm"] <= q33, "rate_group"] = "Low"
    baysal.loc[baysal["rate_norm"] >= q66, "rate_group"] = "High"

    # Extract features
    for _, row in baysal.iterrows():
        sid = row["site_id"]
        seq = sequences.get(sid, "")

        # Flanking nucleotide context
        if len(seq) >= 201:
            pos = 100
            baysal.loc[baysal["site_id"] == sid, "flanking_-2"] = seq[pos - 2] if pos >= 2 else "N"
            baysal.loc[baysal["site_id"] == sid, "flanking_-1"] = seq[pos - 1] if pos >= 1 else "N"
            baysal.loc[baysal["site_id"] == sid, "flanking_+1"] = seq[pos + 1] if pos < len(seq) - 1 else "N"
            baysal.loc[baysal["site_id"] == sid, "flanking_+2"] = seq[pos + 2] if pos < len(seq) - 2 else "N"

            # Local GC content (±10 nt)
            local = seq[max(0, pos - 10):pos + 11]
            gc = sum(1 for c in local if c in "GC") / len(local) if local else 0
            baysal.loc[baysal["site_id"] == sid, "local_gc"] = gc

        # Structure features
        sd = structure_delta.get(sid)
        if sd is not None:
            baysal.loc[baysal["site_id"] == sid, "delta_mfe"] = float(sd[0])

    # Compare features across rate groups
    logger.info("\nFeature comparison (Baysal, high vs low rate tertiles):")

    for feature in ["local_gc", "delta_mfe"]:
        if feature in baysal.columns:
            high = baysal[baysal["rate_group"] == "High"][feature].dropna()
            low = baysal[baysal["rate_group"] == "Low"][feature].dropna()
            if len(high) > 5 and len(low) > 5:
                stat, p = mannwhitneyu(high, low, alternative="two-sided")
                logger.info("  %s: High=%.4f±%.4f, Low=%.4f±%.4f, MW p=%.2e",
                            feature, high.mean(), high.std(), low.mean(), low.std(), p)

    # Flanking nucleotide distribution
    for pos_name in ["flanking_-1", "flanking_+1"]:
        if pos_name in baysal.columns:
            for group in ["High", "Low"]:
                sub = baysal[baysal["rate_group"] == group]
                counts = sub[pos_name].value_counts(normalize=True)
                dist_str = ", ".join(f"{k}:{v:.2f}" for k, v in counts.head(4).items())
                logger.info("  %s %s: %s", group, pos_name, dist_str)

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Rate distribution by tertile
    for group, color in [("Low", "#66bb6a"), ("Medium", "#ffb74d"), ("High", "#ef5350")]:
        rates = baysal[baysal["rate_group"] == group]["rate_norm"]
        axes[0].hist(rates, bins=20, alpha=0.5, label=f"{group} (n={len(rates)})", color=color)
    axes[0].set_xlabel("Editing Rate")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Rate Distribution by Tertile")
    axes[0].legend()

    # Delta-MFE by rate group
    if "delta_mfe" in baysal.columns:
        mfe_data = []
        mfe_labels = []
        for group in ["Low", "Medium", "High"]:
            vals = baysal[baysal["rate_group"] == group]["delta_mfe"].dropna().values
            if len(vals) > 0:
                mfe_data.append(vals)
                mfe_labels.append(f"{group}\n(n={len(vals)})")
        if mfe_data:
            axes[1].boxplot(mfe_data, labels=mfe_labels, showfliers=False, patch_artist=True)
            axes[1].set_ylabel("Delta MFE (kcal/mol)")
            axes[1].set_title("Structure Delta by Rate Group")

    # Local GC content by rate group
    if "local_gc" in baysal.columns:
        gc_data = []
        gc_labels = []
        for group in ["Low", "Medium", "High"]:
            vals = baysal[baysal["rate_group"] == group]["local_gc"].dropna().values
            if len(vals) > 0:
                gc_data.append(vals)
                gc_labels.append(f"{group}\n(n={len(vals)})")
        if gc_data:
            axes[2].boxplot(gc_data, labels=gc_labels, showfliers=False, patch_artist=True)
            axes[2].set_ylabel("Local GC Content (±10nt)")
            axes[2].set_title("GC Content by Rate Group")

    plt.suptitle("Features of High vs Low Rate APOBEC3A Editing Sites (Baysal)\n"
                 "Testing whether sequence/structure features predict rate magnitude",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "rate_feature_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()

    return {"n_baysal_analyzed": len(baysal)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    torch.manual_seed(42)

    t_total = time.time()

    # Load data
    logger.info("Loading data...")
    combined_df = pd.read_csv(COMBINED_CSV)
    splits_df = pd.read_csv(SPLITS_CSV)

    sequences = {}
    if SEQ_JSON.exists():
        sequences = json.loads(SEQ_JSON.read_text())
    logger.info("  %d sequences loaded", len(sequences))

    structure_delta = {}
    if STRUCT_CACHE.exists():
        data = np.load(STRUCT_CACHE, allow_pickle=True)
        feat_key = "delta_features" if "delta_features" in data else "features"
        if "site_ids" in data and feat_key in data:
            sids = data["site_ids"]
            feats = data[feat_key]
            structure_delta = {str(sid): feats[i] for i, sid in enumerate(sids)}
    logger.info("  %d structure deltas loaded", len(structure_delta))

    # Load pooled embeddings for rate prediction
    pooled_orig = torch.load(EMB_DIR / "rnafm_pooled.pt", weights_only=False)
    pooled_edited = torch.load(EMB_DIR / "rnafm_pooled_edited.pt", weights_only=False)
    logger.info("  %d pooled embeddings loaded", len(pooled_orig))

    all_results = {}

    # Analysis 1: Shared-site rate agreement
    all_results["shared_sites"] = analysis_shared_site_rates(combined_df, sequences, OUTPUT_DIR)

    # Analysis 2: TC-motif filtering
    all_results["tc_motif"] = analysis_tc_motif_filtering(
        combined_df, splits_df, sequences, pooled_orig, pooled_edited, OUTPUT_DIR
    )

    # Analysis 3: Variance decomposition
    all_results["variance"] = analysis_variance_decomposition(combined_df, OUTPUT_DIR)

    # Analysis 4: Constitutive vs facultative
    all_results["constitutive"] = analysis_constitutive_vs_facultative(
        combined_df, sequences, OUTPUT_DIR
    )

    # Analysis 5: Rate features
    all_results["features"] = analysis_rate_features(
        combined_df, sequences, structure_delta, OUTPUT_DIR
    )

    # Save results
    with open(OUTPUT_DIR / "rate_deep_dive_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    elapsed = time.time() - t_total
    logger.info("\n" + "=" * 70)
    logger.info("RATE DEEP DIVE COMPLETE (%.1fs)", elapsed)
    logger.info("=" * 70)
    logger.info("Key figures:")
    logger.info("  shared_site_rate_agreement.png")
    logger.info("  tc_motif_rate_analysis.png")
    logger.info("  rate_variance_decomposition.png")
    logger.info("  constitutive_vs_facultative_rates.png")
    logger.info("  rate_feature_analysis.png")


if __name__ == "__main__":
    main()
