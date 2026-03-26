#!/usr/bin/env python
"""Generate structure analysis for all enzyme categories.

Produces per-enzyme:
1. Structural Feature Comparison table (pos vs neg) with MW p-values
2. Pairing probability profile plot (smoothed, pos vs neg)
3. Loop position analysis

All using corrected full 201nt sequences.

Usage:
    conda run -n quris python experiments/multi_enzyme/generate_structure_analysis_all_enzymes.py
"""

import gc
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_ME_DIR = PROJECT_ROOT / "data/processed/multi_enzyme"
SPLITS_CSV = _ME_DIR / "splits_multi_enzyme_v3_with_negatives.csv"
SEQS_JSON = _ME_DIR / "multi_enzyme_sequences_v3_with_negatives.json"
STRUCT_CACHE = _ME_DIR / "structure_cache_multi_enzyme_v3.npz"
LOOP_CSV = _ME_DIR / "loop_position_per_site_v3.csv"
OUTPUT_DIR = PROJECT_ROOT / "experiments/multi_enzyme/outputs/structure_analysis_v3"

CENTER = 100
COLOR_POS = "#1a73e8"
COLOR_NEG = "#d93025"


def compute_structure_comparison(pos_sids, neg_sids, struct_delta, loop_df):
    """Compute structural feature comparison table between pos and neg."""
    features = [
        ("Delta Pairing (edit site)", "delta_pairing_center", 0),
        ("Delta Accessibility (edit site)", "delta_accessibility_center", 1),
        ("Delta Entropy (edit site)", "delta_entropy_center", 2),
        ("Delta MFE (kcal/mol)", "delta_mfe", 3),
        ("Mean Delta Pairing (±10nt)", "mean_delta_pairing_window", 4),
        ("Mean Delta Accessibility (±10nt)", "mean_delta_accessibility_window", 5),
        ("Std Delta Pairing (±10nt)", "std_delta_pairing_window", 6),
    ]

    loop_features = [
        ("Is Unpaired", "is_unpaired"),
        ("Loop Size", "loop_size"),
        ("Relative Loop Position", "relative_loop_position"),
        ("Dist to Apex", "dist_to_apex"),
        ("Dist to Junction", "dist_to_junction"),
        ("Local Unpaired Fraction", "local_unpaired_fraction"),
        ("Left Stem Length", "left_stem_length"),
        ("Right Stem Length", "right_stem_length"),
        ("Max Adjacent Stem Length", "max_adjacent_stem_length"),
    ]

    rows = []

    # Structure delta features
    for label, _, idx in features:
        pos_vals = [struct_delta[sid][idx] for sid in pos_sids if sid in struct_delta]
        neg_vals = [struct_delta[sid][idx] for sid in neg_sids if sid in struct_delta]
        if pos_vals and neg_vals:
            pos_mean = np.mean(pos_vals)
            neg_mean = np.mean(neg_vals)
            _, p = stats.mannwhitneyu(pos_vals, neg_vals, alternative="two-sided")
            rows.append({
                "feature": label, "pos_mean": pos_mean, "neg_mean": neg_mean,
                "diff": pos_mean - neg_mean, "p_value": p,
                "significant": "Yes" if p < 0.05 / len(features) else "No",
            })

    # Loop geometry features
    for label, col in loop_features:
        pos_vals = loop_df.loc[loop_df.index.isin(pos_sids), col].dropna().values
        neg_vals = loop_df.loc[loop_df.index.isin(neg_sids), col].dropna().values
        if len(pos_vals) > 5 and len(neg_vals) > 5:
            pos_mean = np.mean(pos_vals)
            neg_mean = np.mean(neg_vals)
            _, p = stats.mannwhitneyu(pos_vals, neg_vals, alternative="two-sided")
            rows.append({
                "feature": label, "pos_mean": pos_mean, "neg_mean": neg_mean,
                "diff": pos_mean - neg_mean, "p_value": p,
                "significant": "Yes" if p < 0.05 / len(loop_features) else "No",
            })

    return pd.DataFrame(rows)


def plot_pairing_profile(enzyme, pos_pairing, neg_pairing, output_path):
    """Plot smoothed pairing probability profiles for pos vs neg."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    positions = np.arange(201) - CENTER

    for ax, window, title in zip(axes, [1, 5, 11],
                                  ["Raw", "Window=5 (smoothed)", "Window=11 (heavy smooth)"]):
        kernel = np.ones(window) / window

        mean_pos = pos_pairing.mean(axis=0)
        mean_neg = neg_pairing.mean(axis=0)
        n_pos, n_neg = len(pos_pairing), len(neg_pairing)

        if window > 1:
            mean_pos = np.convolve(mean_pos, kernel, mode="same")
            mean_neg = np.convolve(mean_neg, kernel, mode="same")

        se_pos = pos_pairing.std(axis=0) / np.sqrt(n_pos)
        se_neg = neg_pairing.std(axis=0) / np.sqrt(n_neg)
        if window > 1:
            se_pos = np.convolve(se_pos, kernel, mode="same")
            se_neg = np.convolve(se_neg, kernel, mode="same")

        ax.plot(positions, mean_pos, color=COLOR_POS, lw=1.5,
                label=f"Edited (n={n_pos})")
        ax.fill_between(positions,
                         mean_pos - 1.96 * se_pos,
                         mean_pos + 1.96 * se_pos,
                         color=COLOR_POS, alpha=0.15)

        ax.plot(positions, mean_neg, color=COLOR_NEG, lw=1.5,
                label=f"Unedited (n={n_neg})")
        ax.fill_between(positions,
                         mean_neg - 1.96 * se_neg,
                         mean_neg + 1.96 * se_neg,
                         color=COLOR_NEG, alpha=0.15)

        ax.axvline(0, color="black", ls="--", lw=1, alpha=0.5)
        ax.axvspan(-10, 10, color="gray", alpha=0.05)
        ax.set_xlabel("Position relative to edit site (nt)")
        ax.set_ylabel("Mean pairing probability")
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8)
        ax.set_xlim(-50, 50)

    fig.suptitle(f"{enzyme} — Pairing Probability Profile", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv(SPLITS_CSV)
    logger.info("Loaded %d sites", len(df))

    # Structure cache
    struct_data = np.load(str(STRUCT_CACHE), allow_pickle=True)
    struct_sids = [str(s) for s in struct_data["site_ids"]]
    struct_delta = {}
    pairing_probs = {}
    for i, sid in enumerate(struct_sids):
        struct_delta[sid] = struct_data["delta_features"][i]
        if "pairing_probs" in struct_data:
            pairing_probs[sid] = struct_data["pairing_probs"][i]

    has_pairing = len(pairing_probs) > 0
    logger.info("Structure delta: %d sites, pairing profiles: %s",
                len(struct_delta), len(pairing_probs) if has_pairing else "NOT AVAILABLE")

    # Loop positions
    loop_df = pd.read_csv(LOOP_CSV)
    loop_df["site_id"] = loop_df["site_id"].astype(str)
    loop_df = loop_df.set_index("site_id")

    # If no pairing probs in struct cache, compute them from ViennaRNA
    if not has_pairing:
        logger.info("Computing pairing probabilities from sequences (ViennaRNA)...")
        import RNA
        seqs = json.load(open(SEQS_JSON))
        for enzyme in ["A3A", "A3B", "A3G", "A3A_A3G", "Neither"]:
            enzyme_df = df[df["enzyme"] == enzyme]
            # Sample up to 500 per class for profile plots (full ViennaRNA is slow)
            for is_ed in [1, 0]:
                sub = enzyme_df[enzyme_df["is_edited"] == is_ed]
                if len(sub) > 500:
                    sub = sub.sample(500, random_state=42)
                for sid in sub["site_id"].astype(str):
                    if sid in pairing_probs:
                        continue
                    seq = seqs.get(sid, "N" * 201).upper().replace("T", "U")
                    md = RNA.md()
                    md.temperature = 37.0
                    fc = RNA.fold_compound(seq, md)
                    fc.mfe()
                    fc.pf()
                    bpp = np.array(fc.bpp())
                    n = len(seq)
                    bpp = bpp[1:n+1, 1:n+1]
                    pp = np.clip(np.sum(bpp, axis=0) + np.sum(bpp, axis=1), 0, 1)
                    pairing_probs[sid] = pp
            logger.info("  %s: %d pairing profiles computed", enzyme, len(pairing_probs))

    all_results = {}

    for enzyme in ["A3A", "A3B", "A3G", "A3A_A3G", "Neither", "Unknown"]:
        logger.info("\n=== %s ===", enzyme)
        enzyme_df = df[df["enzyme"] == enzyme]
        pos_sids = set(enzyme_df[enzyme_df["is_edited"] == 1]["site_id"].astype(str))
        neg_sids = set(enzyme_df[enzyme_df["is_edited"] == 0]["site_id"].astype(str))

        if not pos_sids or not neg_sids:
            logger.warning("  Skipping %s (no pos or neg)", enzyme)
            continue

        # Structure comparison table
        comp_table = compute_structure_comparison(pos_sids, neg_sids, struct_delta, loop_df)
        comp_table.to_csv(OUTPUT_DIR / f"{enzyme}_structure_comparison.csv", index=False)
        logger.info("  Structure comparison: %d features", len(comp_table))

        # Log key differences
        for _, row in comp_table.iterrows():
            if row["significant"] == "Yes":
                logger.info("  %s: pos=%.4f neg=%.4f diff=%.4f p=%.2e ***",
                             row["feature"], row["pos_mean"], row["neg_mean"],
                             row["diff"], row["p_value"])

        # Pairing profile plot
        pos_pp = np.array([pairing_probs[sid] for sid in pos_sids
                           if sid in pairing_probs and len(pairing_probs[sid]) == 201])
        neg_pp = np.array([pairing_probs[sid] for sid in neg_sids
                           if sid in pairing_probs and len(pairing_probs[sid]) == 201])

        if len(pos_pp) > 10 and len(neg_pp) > 10:
            plot_pairing_profile(enzyme, pos_pp, neg_pp,
                                  OUTPUT_DIR / f"{enzyme}_pairing_profile.png")
            logger.info("  Pairing profile: %d pos, %d neg", len(pos_pp), len(neg_pp))
        else:
            logger.warning("  Not enough pairing data for profile plot")

        all_results[enzyme] = {
            "n_pos": len(pos_sids),
            "n_neg": len(neg_sids),
            "comparison": comp_table.to_dict("records"),
        }

    # Save combined results
    with open(OUTPUT_DIR / "structure_analysis_all_enzymes.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info("\nSaved to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
