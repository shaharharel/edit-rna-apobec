#!/usr/bin/env python
"""Per-enzyme structural context analysis for multi-APOBEC editing sites.

For each enzyme (A1, A3A, A3B, A3G, A3H, A4), analyzes:
1. Loop vs stem preference (in-loop fraction)
2. Loop size distribution
3. Loop type distribution (hairpin, internal, bulge, external)
4. Distance to loop apex and junction
5. Stem length distribution
6. Delta MFE (stability change from C->U edit)
7. Mann-Whitney U tests for positive vs negative per enzyme

Input:  data/processed/multi_enzyme/splits_multi_enzyme_v2.csv
        data/processed/multi_enzyme/multi_enzyme_sequences_v2.json
        data/processed/multi_enzyme/loop_position_per_site_v2.csv
        data/processed/multi_enzyme/structure_cache_multi_enzyme_v2.npz
Output: experiments/multi_enzyme/outputs/structure_analysis/

Usage:
    conda run -n quris python experiments/multi_enzyme/exp_per_enzyme_structure.py
"""

import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, chi2_contingency, fisher_exact
try:
    from scipy.stats import binomtest as binom_test_scipy
    def binom_test(k, n, p, alternative="two-sided"):
        return binom_test_scipy(k, n, p, alternative=alternative).pvalue
except ImportError:
    from scipy.stats import binom_test  # scipy < 1.9

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SPLITS_CSV = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "splits_multi_enzyme_v2.csv"
LOOP_POS_CSV = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "loop_position_per_site_v2.csv"
STRUCT_CACHE = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "structure_cache_multi_enzyme_v2.npz"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "multi_enzyme" / "outputs" / "structure_analysis"

ENZYME_COLORS = {
    "A3A": "#2563eb",
    "A3B": "#dc2626",
    "A3G": "#16a34a",
    "A1":  "#d97706",
    "A3H": "#7c3aed",
    "A4":  "#6b7280",
}

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
})


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data():
    """Load splits, loop position data, and optional structure cache."""
    if not SPLITS_CSV.exists():
        logger.error("Splits CSV not found: %s", SPLITS_CSV)
        sys.exit(1)

    df = pd.read_csv(SPLITS_CSV)
    logger.info("Loaded %d sites from splits CSV", len(df))

    # Load loop position data — HARD FAIL if missing.
    # Without this file, all in-loop fractions and loop geometry analysis would be
    # silently wrong (all zeros), producing misleading structural comparison results.
    if not LOOP_POS_CSV.exists():
        logger.error(
            "FATAL: loop_position_per_site_v2.csv not found: %s\n"
            "Run loop_position step FIRST. Without it, all loop geometry features "
            "(is_unpaired, relative_loop_position, etc.) would be missing — "
            "structure analysis results would be meaningless.",
            LOOP_POS_CSV,
        )
        sys.exit(1)
    loop_df = pd.read_csv(LOOP_POS_CSV)
    loop_df["site_id"] = loop_df["site_id"].astype(str)
    logger.info("Loaded %d loop position records", len(loop_df))

    # Load structure cache for delta features — HARD FAIL if missing.
    # Without this, delta MFE and structure delta analyses would silently produce
    # empty results with no error, making the structure analysis incomplete.
    if not STRUCT_CACHE.exists():
        logger.error(
            "FATAL: Structure cache not found: %s\n"
            "Run structure cache generation first! Without it, delta MFE and "
            "structure delta analyses will be empty.",
            STRUCT_CACHE,
        )
        sys.exit(1)
    struct_dict = {}
    struct_data = np.load(str(STRUCT_CACHE), allow_pickle=True)
    struct_sids = [str(s) for s in struct_data["site_ids"]]
    delta_features = struct_data["delta_features"]
    mfes = struct_data["mfes"]
    mfes_edited = struct_data["mfes_edited"]
    for i, sid in enumerate(struct_sids):
        struct_dict[sid] = {
            "delta_features": delta_features[i],
            "mfe_orig": float(mfes[i]),
            "mfe_edited": float(mfes_edited[i]),
            "delta_mfe": float(mfes_edited[i]) - float(mfes[i]),
        }
    logger.info("Loaded %d structure cache entries", len(struct_dict))

    return df, loop_df, struct_dict


def safe_mannwhitneyu(x, y, alternative="two-sided"):
    """Mann-Whitney U test with safety checks."""
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if len(x) < 2 or len(y) < 2:
        return float("nan"), float("nan")
    try:
        u, p = mannwhitneyu(x, y, alternative=alternative)
        return float(u), float(p)
    except Exception:
        return float("nan"), float("nan")


def compute_group_stats(values):
    """Compute summary statistics for a numeric array."""
    arr = np.array(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return {"n": 0, "mean": None, "median": None, "std": None}
    return {
        "n": int(len(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_inloop_fraction_comparison(enzyme_results, output_dir):
    """Bar chart: in-loop fraction per enzyme, positives vs negatives.

    Key hypothesis: A3A and A3G prefer loop context (>0.5), while
    A3B is predicted to LACK loop preference (Zhang 2023).
    A3A reference in-loop fraction ~0.63 shown as dashed line.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    enzymes = sorted(enzyme_results.keys())
    x = np.arange(len(enzymes))
    width = 0.35

    pos_fracs = []
    neg_fracs = []
    for e in enzymes:
        r = enzyme_results[e]
        pos_fracs.append(r.get("positive_inloop_fraction", 0))
        neg_fracs.append(r.get("negative_inloop_fraction", 0))

    ax.bar(x - width / 2, pos_fracs, width, label="Positive (edited)",
           color=[ENZYME_COLORS.get(e, "#999") for e in enzymes], alpha=0.8,
           edgecolor="black", linewidth=0.5)
    ax.bar(x + width / 2, neg_fracs, width, label="Negative (unedited)",
           color=[ENZYME_COLORS.get(e, "#999") for e in enzymes], alpha=0.3,
           edgecolor="black", linewidth=0.5)

    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Random (50%)")
    ax.axhline(0.63, color="#2563eb", linestyle=":", alpha=0.5, label="A3A reference (63%)")
    ax.set_xticks(x)
    ax.set_xticklabels(enzymes)
    ax.set_ylabel("In-Loop Fraction")
    ax.set_title("Stem-Loop Preference by Enzyme\n"
                 "Key test: Does A3B lack loop preference? (Zhang 2023 prediction)")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(0, 1.0)

    # Annotate each bar with the binomial test result (vs random 0.5)
    for i, e in enumerate(enzymes):
        r = enzyme_results[e]
        frac = pos_fracs[i]

        # Binomial test significance (vs 0.5 random)
        binom_p = r.get("inloop_vs_random_binom_p", 1.0)
        if isinstance(binom_p, float) and not np.isnan(binom_p):
            sig = "***" if binom_p < 0.001 else "**" if binom_p < 0.01 else "*" if binom_p < 0.05 else "ns"
        else:
            sig = ""
        ax.text(i - width / 2, frac + 0.02, f"{frac:.0%}\n{sig}",
                ha="center", fontsize=7, fontweight="bold")

        # Special annotation for A3B
        if e == "A3B":
            has_loop = r.get("has_loop_preference", None)
            if has_loop is False:
                ax.annotate("No loop\npreference",
                            xy=(i - width / 2, frac),
                            xytext=(i + 0.8, frac + 0.15),
                            arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.5),
                            fontsize=8, color="#dc2626", fontweight="bold",
                            ha="center")

    fig.tight_layout()
    fig.savefig(output_dir / "inloop_fraction_comparison.png")
    plt.close(fig)
    logger.info("Saved inloop_fraction_comparison.png")


def plot_loop_size_by_enzyme(enzyme_loop_data, output_dir):
    """Box plot: loop size distribution per enzyme (positives only)."""
    enzymes = sorted(enzyme_loop_data.keys())
    data = []
    labels = []
    colors = []
    for e in enzymes:
        vals = enzyme_loop_data[e].get("pos_loop_sizes", [])
        if len(vals) > 0:
            data.append(vals)
            labels.append(e)
            colors.append(ENZYME_COLORS.get(e, "#999"))

    if not data:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    ax.set_ylabel("Loop Size (nt)")
    ax.set_title("Loop Size Distribution by Enzyme (Positive Sites)")
    fig.tight_layout()
    fig.savefig(output_dir / "loop_size_by_enzyme.png")
    plt.close(fig)
    logger.info("Saved loop_size_by_enzyme.png")


def plot_delta_mfe_by_enzyme(enzyme_struct_data, output_dir):
    """Box plot: delta MFE distribution per enzyme."""
    enzymes = sorted(enzyme_struct_data.keys())
    data_pos = []
    data_neg = []
    labels = []
    for e in enzymes:
        pos_mfe = enzyme_struct_data[e].get("pos_delta_mfe", [])
        neg_mfe = enzyme_struct_data[e].get("neg_delta_mfe", [])
        if len(pos_mfe) > 5:
            data_pos.append(pos_mfe)
            data_neg.append(neg_mfe if len(neg_mfe) > 5 else [0])
            labels.append(e)

    if not data_pos:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    width = 0.35

    bp1 = ax.boxplot(data_pos, positions=x - width / 2, widths=width * 0.8,
                     patch_artist=True, showfliers=False)
    for patch in bp1["boxes"]:
        patch.set_facecolor("#2563eb")
        patch.set_alpha(0.5)

    if any(len(d) > 1 for d in data_neg):
        bp2 = ax.boxplot(data_neg, positions=x + width / 2, widths=width * 0.8,
                         patch_artist=True, showfliers=False)
        for patch in bp2["boxes"]:
            patch.set_facecolor("#dc2626")
            patch.set_alpha(0.5)

    ax.axhline(0, color="black", linestyle="--", alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Delta MFE (kcal/mol)")
    ax.set_title("Stability Change (C->U) by Enzyme: Positive (blue) vs Negative (red)")
    fig.tight_layout()
    fig.savefig(output_dir / "delta_mfe_by_enzyme.png")
    plt.close(fig)
    logger.info("Saved delta_mfe_by_enzyme.png")


def plot_loop_type_by_enzyme(enzyme_loop_data, output_dir):
    """Stacked bar chart: loop type fractions per enzyme."""
    enzymes = sorted(enzyme_loop_data.keys())
    all_types = set()
    for e in enzymes:
        lt = enzyme_loop_data[e].get("pos_loop_type_dist", {})
        all_types.update(lt.keys())

    loop_types = sorted(all_types)
    if not loop_types:
        return

    type_colors = {
        "hairpin": "#2563eb",
        "internal": "#16a34a",
        "bulge_left": "#d97706",
        "bulge_right": "#f59e0b",
        "multiloop": "#7c3aed",
        "external": "#6b7280",
        "other": "#94a3b8",
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(enzymes))
    bottoms = np.zeros(len(enzymes))

    for lt in loop_types:
        fracs = []
        for e in enzymes:
            dist = enzyme_loop_data[e].get("pos_loop_type_dist", {})
            total = sum(dist.values()) or 1
            fracs.append(dist.get(lt, 0) / total)
        color = type_colors.get(lt, "#999999")
        ax.bar(x, fracs, bottom=bottoms, label=lt, color=color, alpha=0.8)
        bottoms += np.array(fracs)

    ax.set_xticks(x)
    ax.set_xticklabels(enzymes)
    ax.set_ylabel("Fraction")
    ax.set_title("Loop Type Distribution by Enzyme (Positive Sites)")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    fig.savefig(output_dir / "loop_type_by_enzyme.png")
    plt.close(fig)
    logger.info("Saved loop_type_by_enzyme.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df, loop_df, struct_dict = load_data()

    if "enzyme" not in df.columns:
        logger.error("No 'enzyme' column in splits CSV")
        sys.exit(1)

    enzymes = sorted(df["enzyme"].unique())
    logger.info("Enzymes: %s", enzymes)

    # Determine label column
    if "is_edited" in df.columns:
        label_col = "is_edited"
    elif "label" in df.columns:
        label_col = "label"
    else:
        label_col = None
        df = df.copy()
        df["_all_positive"] = 1
        logger.info("No is_edited/label column found — treating all sites as positive")

    all_results = {}
    enzyme_loop_data = {}
    enzyme_struct_data = {}

    for enzyme in enzymes:
        logger.info("=" * 60)
        logger.info("Analyzing structure for: %s", enzyme)
        logger.info("=" * 60)

        enz_df = df[df["enzyme"] == enzyme]
        _lc = label_col if label_col else "_all_positive"
        pos = enz_df[enz_df[_lc] == 1]
        neg = enz_df[enz_df[_lc] == 0]
        n_pos = len(pos)
        n_neg = len(neg)
        logger.info("  Positive: %d, Negative: %d", n_pos, n_neg)

        result = {
            "enzyme": enzyme,
            "n_positive": n_pos,
            "n_negative": n_neg,
        }

        # --- Loop position analysis ---
        enzyme_loop_data[enzyme] = {}
        if loop_df is not None and len(loop_df) > 0:
            pos_ids = set(pos["site_id"].astype(str))
            neg_ids = set(neg["site_id"].astype(str))

            pos_loop = loop_df[loop_df["site_id"].isin(pos_ids)]
            neg_loop = loop_df[loop_df["site_id"].isin(neg_ids)]

            # Sanity check: all in-loop fractions = 0.0 => structure not loaded
            if len(pos_loop) > 0 and "is_unpaired" in pos_loop.columns:
                pos_inloop = pos_loop["is_unpaired"].sum()
                neg_inloop = neg_loop["is_unpaired"].sum() if len(neg_loop) > 0 else 0

                result["positive_inloop_fraction"] = float(pos_inloop / max(len(pos_loop), 1))
                result["negative_inloop_fraction"] = float(neg_inloop / max(len(neg_loop), 1))

                # Binomial test: is positive in-loop fraction different from 0.5 (random)?
                # This tests whether this enzyme has a loop PREFERENCE (>0.5) or
                # AVOIDANCE (<0.5) relative to random expectation.
                if len(pos_loop) >= 10:
                    binom_p = binom_test(int(pos_inloop), len(pos_loop), 0.5,
                                        alternative="two-sided")
                    result["inloop_vs_random_binom_p"] = float(binom_p)
                    result["inloop_vs_random_significant"] = binom_p < 0.05
                    result["has_loop_preference"] = (
                        result["positive_inloop_fraction"] > 0.5 and binom_p < 0.05
                    )
                    result["has_stem_preference"] = (
                        result["positive_inloop_fraction"] < 0.5 and binom_p < 0.05
                    )

                # Sanity check
                if result["positive_inloop_fraction"] == 0.0 and n_pos > 50:
                    logger.warning(
                        "  SANITY FAIL: %s all in-loop fractions = 0.0 for %d positive sites. "
                        "Structure cache may not be loaded correctly!",
                        enzyme, n_pos
                    )

                # Chi-squared test: is in-loop fraction different between pos and neg?
                if len(pos_loop) > 10 and len(neg_loop) > 10:
                    table = [
                        [int(pos_inloop), len(pos_loop) - int(pos_inloop)],
                        [int(neg_inloop), len(neg_loop) - int(neg_inloop)],
                    ]
                    try:
                        chi2, p, _, _ = chi2_contingency(table)
                        result["inloop_chi2"] = float(chi2)
                        result["inloop_chi2_p"] = float(p)
                    except Exception:
                        result["inloop_chi2_p"] = float("nan")

                # Loop size (positives only, unpaired)
                pos_unpaired = pos_loop[pos_loop["is_unpaired"] == True]
                neg_unpaired = neg_loop[neg_loop["is_unpaired"] == True] if len(neg_loop) > 0 else pd.DataFrame()

                if len(pos_unpaired) > 0 and "loop_size" in pos_unpaired.columns:
                    pos_loop_sizes = pos_unpaired["loop_size"].dropna().values.tolist()
                    enzyme_loop_data[enzyme]["pos_loop_sizes"] = pos_loop_sizes
                    result["positive_loop_size"] = compute_group_stats(pos_loop_sizes)

                    # Loop type distribution (column may not exist in all
                    # loop_position CSVs — the A3A version doesn't produce it)
                    if "loop_type" in pos_unpaired.columns:
                        lt_counts = pos_unpaired["loop_type"].value_counts().to_dict()
                        enzyme_loop_data[enzyme]["pos_loop_type_dist"] = {
                            k: int(v) for k, v in lt_counts.items()
                        }
                        result["positive_loop_type_distribution"] = {
                            k: int(v) for k, v in lt_counts.items()
                        }

                        # External loops from N-padded sequences have inflated loop_size.
                        # Report corrected loop_size excluding external loops.
                        n_external = lt_counts.get("external", 0)
                        n_unpaired_total = len(pos_unpaired)
                        result["external_loop_fraction"] = n_external / max(n_unpaired_total, 1)
                        pos_no_ext = pos_unpaired[pos_unpaired["loop_type"] != "external"]
                        if len(pos_no_ext) > 0:
                            no_ext_sizes = pos_no_ext["loop_size"].dropna().values.tolist()
                            result["positive_loop_size_no_external"] = compute_group_stats(no_ext_sizes)
                            enzyme_loop_data[enzyme]["pos_loop_sizes_no_ext"] = no_ext_sizes
                        if n_external > 0:
                            logger.info(
                                "  %s: %d/%d unpaired sites (%.1f%%) classified as external loops "
                                "(N-padding artifact); loop_size excluding external: mean=%.1f",
                                enzyme, n_external, n_unpaired_total,
                                100 * n_external / n_unpaired_total,
                                result.get("positive_loop_size_no_external", {}).get("mean", float("nan")),
                            )
                    else:
                        logger.info("  %s: loop_type column not in CSV, skipping type distribution", enzyme)

                    # Relative loop position
                    if "relative_loop_position" in pos_unpaired.columns:
                        pos_rlp = pos_unpaired["relative_loop_position"].dropna().values
                        result["positive_relative_loop_position"] = compute_group_stats(pos_rlp)

                    # Dist to apex
                    if "dist_to_apex" in pos_unpaired.columns:
                        pos_apex = pos_unpaired["dist_to_apex"].dropna().values
                        result["positive_dist_to_apex"] = compute_group_stats(pos_apex)

                # Feature comparisons (Mann-Whitney): pos vs neg
                compare_features = [
                    "loop_size", "dist_to_apex", "relative_loop_position",
                    "max_adjacent_stem_length", "dist_to_junction", "local_unpaired_fraction",
                ]
                feature_tests = {}
                for feat in compare_features:
                    if feat not in pos_loop.columns:
                        continue
                    # For loop-specific features, compare unpaired only
                    if feat in ("loop_size", "dist_to_apex", "relative_loop_position"):
                        pv = pos_unpaired[feat].dropna().values if len(pos_unpaired) > 0 else np.array([])
                        nv = neg_unpaired[feat].dropna().values if len(neg_unpaired) > 0 else np.array([])
                    else:
                        pv = pos_loop[feat].dropna().values
                        nv = neg_loop[feat].dropna().values if len(neg_loop) > 0 else np.array([])

                    u_stat, p_val = safe_mannwhitneyu(pv, nv)
                    feature_tests[feat] = {
                        "positive": compute_group_stats(pv),
                        "negative": compute_group_stats(nv),
                        "mannwhitney_U": u_stat,
                        "mannwhitney_p": p_val,
                        "significant": p_val < 0.05 if not np.isnan(p_val) else None,
                    }
                result["feature_tests"] = feature_tests
            else:
                logger.warning("  No loop position data for %s", enzyme)
                result["positive_inloop_fraction"] = 0.0
                result["negative_inloop_fraction"] = 0.0
        else:
            result["positive_inloop_fraction"] = 0.0
            result["negative_inloop_fraction"] = 0.0

        # --- Structure delta features ---
        enzyme_struct_data[enzyme] = {}
        if struct_dict:
            pos_ids = pos["site_id"].astype(str).tolist()
            neg_ids = neg["site_id"].astype(str).tolist()

            pos_delta_mfe = [struct_dict[sid]["delta_mfe"] for sid in pos_ids if sid in struct_dict]
            neg_delta_mfe = [struct_dict[sid]["delta_mfe"] for sid in neg_ids if sid in struct_dict]

            enzyme_struct_data[enzyme]["pos_delta_mfe"] = pos_delta_mfe
            enzyme_struct_data[enzyme]["neg_delta_mfe"] = neg_delta_mfe

            result["positive_delta_mfe"] = compute_group_stats(pos_delta_mfe)
            result["negative_delta_mfe"] = compute_group_stats(neg_delta_mfe)

            if len(pos_delta_mfe) > 5 and len(neg_delta_mfe) > 5:
                u, p = safe_mannwhitneyu(pos_delta_mfe, neg_delta_mfe)
                result["delta_mfe_mannwhitney_p"] = p

        all_results[enzyme] = result

        # Print summary
        print(f"\n--- {enzyme} Structure Summary ---")
        print(f"  Positive: {n_pos}, Negative: {n_neg}")
        print(f"  In-loop fraction: pos={result.get('positive_inloop_fraction', 0):.3f}, "
              f"neg={result.get('negative_inloop_fraction', 0):.3f}")
        if "inloop_chi2_p" in result:
            sig = "***" if result["inloop_chi2_p"] < 0.001 else "ns"
            print(f"  Chi2 (in-loop pos vs neg): p={result['inloop_chi2_p']:.2e} {sig}")
        if "positive_loop_size" in result and result["positive_loop_size"]["mean"] is not None:
            print(f"  Mean loop size (pos): {result['positive_loop_size']['mean']:.1f}")
        if "positive_delta_mfe" in result and result["positive_delta_mfe"]["mean"] is not None:
            print(f"  Mean delta MFE (pos): {result['positive_delta_mfe']['mean']:.4f}")

    # ---------------------------------------------------------------------------
    # Generate plots
    # ---------------------------------------------------------------------------
    logger.info("Generating plots...")
    plot_inloop_fraction_comparison(all_results, OUTPUT_DIR)
    plot_loop_size_by_enzyme(enzyme_loop_data, OUTPUT_DIR)
    plot_delta_mfe_by_enzyme(enzyme_struct_data, OUTPUT_DIR)
    plot_loop_type_by_enzyme(enzyme_loop_data, OUTPUT_DIR)

    # ---------------------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------------------
    def serialize(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return str(obj)

    with open(OUTPUT_DIR / "per_enzyme_structure_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=serialize)
    logger.info("Results saved to %s", OUTPUT_DIR / "per_enzyme_structure_results.json")

    # Print final summary table
    print("\n" + "=" * 80)
    print("CROSS-ENZYME STRUCTURE SUMMARY")
    print("=" * 80)
    print(f"{'Enzyme':>6s} {'n_pos':>6s} {'n_neg':>6s} {'InLoop%':>8s} "
          f"{'LoopSz':>7s} {'dMFE':>8s} {'sig':>5s}")
    print("-" * 55)
    for enzyme in sorted(all_results.keys()):
        r = all_results[enzyme]
        inloop = r.get("positive_inloop_fraction", 0)
        loop_sz = r.get("positive_loop_size", {}).get("mean", None)
        dmfe = r.get("positive_delta_mfe", {}).get("mean", None)
        p = r.get("inloop_chi2_p", float("nan"))
        sig = "***" if isinstance(p, float) and not np.isnan(p) and p < 0.001 else "ns"
        print(f"{enzyme:>6s} {r['n_positive']:>6d} {r['n_negative']:>6d} "
              f"{inloop:>7.1%} "
              f"{loop_sz if loop_sz else 'N/A':>7} "
              f"{f'{dmfe:.4f}' if dmfe else 'N/A':>8s} "
              f"{sig:>5s}")
    print("=" * 80)


if __name__ == "__main__":
    main()
