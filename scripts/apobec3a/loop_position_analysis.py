#!/usr/bin/env python
"""Loop position analysis of RNA secondary structures at APOBEC3A editing sites.

For each editing site, determines:
1. Whether the edited C is in a loop or stem (paired vs unpaired)
2. If in a loop: position within the loop (apex vs base), loop size
3. Distance from the edited C to the nearest stem base pair
4. Stem length (number of base pairs in the adjacent stem)
5. Distance from the edited C to the loop-stem junction

Compares positive vs negative sites for each feature and performs
Mann-Whitney U tests. Produces per-dataset breakdowns.

The script uses ViennaRNA (RNA.fold) for structure prediction on 201nt
windows centered on the edit site (position 100, 0-indexed).

Usage:
    conda activate quris
    python scripts/apobec3a/loop_position_analysis.py
"""

import json
import logging
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLITS_CSV = PROJECT_ROOT / "data" / "processed" / "splits_expanded_a3a.csv"
SEQ_JSON = PROJECT_ROOT / "data" / "processed" / "site_sequences.json"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "apobec3a" / "outputs" / "loop_position"

EDIT_POS = 100  # 0-indexed center of 201nt window

DATASET_LABELS = {
    "advisor_c2t": "Advisor",
    "asaoka_2019": "Asaoka",
    "sharma_2015": "Sharma",
    "alqassim_2021": "Alqassim",
    "tier2_negative": "Tier2 Neg",
    "tier3_negative": "Tier3 Neg",
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
# Structure analysis functions
# ---------------------------------------------------------------------------

def fold_sequence(seq: str) -> tuple:
    """Fold an RNA sequence using ViennaRNA and return (dot_bracket, mfe)."""
    import RNA
    md = RNA.md()
    md.temperature = 37.0
    fc = RNA.fold_compound(seq, md)
    structure, mfe = fc.mfe()
    return structure, mfe


def find_loop_boundaries(dot_bracket: str, pos: int) -> tuple:
    """Find the boundaries of the loop containing position `pos`.

    Walks left and right from `pos` to find the nearest paired characters.

    Returns:
        (left_boundary, right_boundary) -- indices of the first paired
        characters on each side (exclusive of the loop), or (-1, -1) if
        the position is not in a loop.
    """
    if pos < 0 or pos >= len(dot_bracket):
        return -1, -1
    if dot_bracket[pos] in "()":
        return -1, -1  # position is paired, not in a loop

    n = len(dot_bracket)

    # Walk left to find nearest paired character
    left = pos - 1
    while left >= 0 and dot_bracket[left] == ".":
        left -= 1

    # Walk right to find nearest paired character
    right = pos + 1
    while right < n and dot_bracket[right] == ".":
        right += 1

    return left, right


def compute_loop_size(dot_bracket: str, left: int, right: int) -> int:
    """Number of unpaired nucleotides in the loop between boundaries."""
    if left < 0 and right >= len(dot_bracket):
        # External loop spanning the entire sequence
        return sum(1 for c in dot_bracket if c == ".")
    start = left + 1 if left >= 0 else 0
    end = right if right < len(dot_bracket) else len(dot_bracket)
    return sum(1 for c in dot_bracket[start:end] if c == ".")


def compute_loop_position(pos: int, left: int, right: int, n: int) -> dict:
    """Compute position-within-loop metrics.

    Returns dict with:
        loop_start: first unpaired position in the loop
        loop_end: last unpaired position in the loop (inclusive)
        loop_size: number of unpaired nt
        dist_to_left_boundary: distance from pos to left stem boundary
        dist_to_right_boundary: distance from pos to right stem boundary
        relative_position: 0.0 = at left boundary, 1.0 = at right boundary
        dist_to_apex: distance from the loop center (apex)
    """
    loop_start = (left + 1) if left >= 0 else 0
    loop_end = (right - 1) if right < n else n - 1
    loop_size = loop_end - loop_start + 1

    dist_left = pos - loop_start
    dist_right = loop_end - pos

    if loop_size > 1:
        relative_pos = dist_left / (loop_size - 1)
    else:
        relative_pos = 0.5

    apex = (loop_start + loop_end) / 2.0
    dist_to_apex = abs(pos - apex)

    return {
        "loop_start": loop_start,
        "loop_end": loop_end,
        "loop_size": loop_size,
        "dist_to_left_boundary": dist_left,
        "dist_to_right_boundary": dist_right,
        "relative_position": relative_pos,
        "dist_to_apex": dist_to_apex,
    }


def compute_stem_length(dot_bracket: str, boundary_pos: int, direction: str) -> int:
    """Count consecutive paired characters starting from boundary_pos.

    direction: 'left' walks leftward, 'right' walks rightward.
    """
    n = len(dot_bracket)
    if boundary_pos < 0 or boundary_pos >= n:
        return 0
    if dot_bracket[boundary_pos] not in "()":
        return 0

    count = 0
    if direction == "left":
        i = boundary_pos
        while i >= 0 and dot_bracket[i] in "()":
            count += 1
            i -= 1
    elif direction == "right":
        i = boundary_pos
        while i < n and dot_bracket[i] in "()":
            count += 1
            i += 1
    return count


def classify_loop_type(dot_bracket: str, left: int, right: int) -> str:
    """Classify the type of loop based on boundary characters.

    - hairpin: left='(' and right=')'
    - internal/bulge: both are paired but same direction (e.g., both '(' or
      one '(' and one ')')
    - external: one or both boundaries are at sequence edges
    - multiloop: more complex cases
    """
    n = len(dot_bracket)
    if left < 0 or right >= n:
        return "external"
    left_char = dot_bracket[left]
    right_char = dot_bracket[right]
    if left_char == "(" and right_char == ")":
        return "hairpin"
    elif left_char == ")" and right_char == "(":
        return "multiloop"
    elif left_char == "(" and right_char == "(":
        return "bulge_left"
    elif left_char == ")" and right_char == ")":
        return "bulge_right"
    return "other"


def analyze_site_structure(dot_bracket: str, pos: int = EDIT_POS) -> dict:
    """Full structural analysis for a single site.

    Returns a dict with all loop/stem features for the edit position.
    """
    n = len(dot_bracket)
    result = {
        "is_unpaired": None,
        "loop_type": None,
        "loop_size": None,
        "dist_to_left_boundary": None,
        "dist_to_right_boundary": None,
        "dist_to_nearest_stem": None,
        "relative_loop_position": None,
        "dist_to_apex": None,
        "left_stem_length": None,
        "right_stem_length": None,
        "max_adjacent_stem_length": None,
        "dist_to_junction": None,
        "local_unpaired_fraction": None,
        "mfe": None,
    }

    if not dot_bracket or pos >= n or pos < 0:
        return result

    is_unpaired = dot_bracket[pos] == "."
    result["is_unpaired"] = is_unpaired

    # Local unpaired fraction (within +/- 10 nt)
    window = 10
    w_start = max(0, pos - window)
    w_end = min(n, pos + window + 1)
    local_region = dot_bracket[w_start:w_end]
    result["local_unpaired_fraction"] = sum(
        1 for c in local_region if c == "."
    ) / len(local_region)

    if is_unpaired:
        # In a loop -- find boundaries and compute loop features
        left, right = find_loop_boundaries(dot_bracket, pos)
        loop_info = compute_loop_position(pos, left, right, n)

        result["loop_size"] = loop_info["loop_size"]
        result["dist_to_left_boundary"] = loop_info["dist_to_left_boundary"]
        result["dist_to_right_boundary"] = loop_info["dist_to_right_boundary"]
        result["dist_to_nearest_stem"] = min(
            loop_info["dist_to_left_boundary"],
            loop_info["dist_to_right_boundary"],
        )
        result["relative_loop_position"] = loop_info["relative_position"]
        result["dist_to_apex"] = loop_info["dist_to_apex"]
        result["loop_type"] = classify_loop_type(dot_bracket, left, right)

        # Stem lengths on each side of the loop
        result["left_stem_length"] = compute_stem_length(
            dot_bracket, left, "left"
        )
        result["right_stem_length"] = compute_stem_length(
            dot_bracket, right, "right"
        )
        result["max_adjacent_stem_length"] = max(
            result["left_stem_length"], result["right_stem_length"]
        )
        # Distance to junction = distance to nearest stem boundary
        result["dist_to_junction"] = result["dist_to_nearest_stem"]
    else:
        # In a stem -- find distance to nearest unpaired region
        # Walk left to find nearest '.'
        dist_left = 0
        i = pos - 1
        while i >= 0 and dot_bracket[i] in "()":
            dist_left += 1
            i -= 1
        # Walk right to find nearest '.'
        dist_right = 0
        j = pos + 1
        while j < n and dot_bracket[j] in "()":
            dist_right += 1
            j += 1
        result["dist_to_junction"] = min(dist_left, dist_right)

        # Measure the stem the edit site sits in
        # Walk both directions while paired
        stem_left = compute_stem_length(dot_bracket, pos, "left")
        stem_right = compute_stem_length(dot_bracket, pos, "right")
        result["left_stem_length"] = stem_left
        result["right_stem_length"] = stem_right
        result["max_adjacent_stem_length"] = max(stem_left, stem_right)

    return result


# ---------------------------------------------------------------------------
# Statistical comparison helpers
# ---------------------------------------------------------------------------

def safe_mannwhitneyu(x, y, alternative="two-sided"):
    """Run Mann-Whitney U test, returning (U, p) or (nan, nan) on failure."""
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
        return {"n": 0, "mean": None, "median": None, "std": None,
                "q25": None, "q75": None}
    return {
        "n": int(len(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "q25": float(np.percentile(arr, 25)),
        "q75": float(np.percentile(arr, 75)),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_paired_vs_unpaired(df, output_dir):
    """Bar chart: fraction of sites in loop vs stem, positive vs negative."""
    fig, ax = plt.subplots(figsize=(6, 4))

    groups = []
    for label, label_name in [(1, "Positive"), (0, "Negative")]:
        sub = df[df["label"] == label]
        n_total = len(sub)
        if n_total == 0:
            continue
        n_unpaired = sub["is_unpaired"].sum()
        n_paired = n_total - n_unpaired
        groups.append({
            "group": label_name,
            "unpaired_frac": n_unpaired / n_total,
            "paired_frac": n_paired / n_total,
            "n": n_total,
        })

    if not groups:
        plt.close()
        return

    x = np.arange(len(groups))
    width = 0.35
    ax.bar(x - width / 2, [g["unpaired_frac"] for g in groups],
           width, label="In loop (unpaired)", color="#3b82f6")
    ax.bar(x + width / 2, [g["paired_frac"] for g in groups],
           width, label="In stem (paired)", color="#ef4444")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{g['group']}\n(n={g['n']})" for g in groups])
    ax.set_ylabel("Fraction of sites")
    ax.set_title("Edit site structural context: loop vs stem")
    ax.legend()
    ax.set_ylim(0, 1.0)

    fig.tight_layout()
    fig.savefig(output_dir / "paired_vs_unpaired.png")
    plt.close(fig)
    logger.info("Saved paired_vs_unpaired.png")


def plot_loop_size_distribution(df, output_dir):
    """Histogram: loop size distribution for positive vs negative (unpaired only)."""
    fig, ax = plt.subplots(figsize=(8, 4))

    for label, label_name, color in [(1, "Positive", "#3b82f6"),
                                      (0, "Negative", "#ef4444")]:
        sub = df[(df["label"] == label) & (df["is_unpaired"] == True)]
        vals = sub["loop_size"].dropna().values
        if len(vals) == 0:
            continue
        bins = np.arange(0.5, min(vals.max() + 2, 61), 1)
        ax.hist(vals, bins=bins, alpha=0.5, label=f"{label_name} (n={len(vals)})",
                color=color, density=True, edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Loop size (nt)")
    ax.set_ylabel("Density")
    ax.set_title("Loop size distribution (unpaired sites only)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "loop_size_distribution.png")
    plt.close(fig)
    logger.info("Saved loop_size_distribution.png")


def plot_loop_type_distribution(df, output_dir):
    """Bar chart: loop type distribution for positive vs negative."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, (label, label_name) in zip(axes, [(1, "Positive"), (0, "Negative")]):
        sub = df[(df["label"] == label) & (df["is_unpaired"] == True)]
        if len(sub) == 0:
            ax.set_title(f"{label_name} (n=0)")
            continue
        counts = sub["loop_type"].value_counts()
        ax.barh(counts.index, counts.values, color="#6366f1")
        ax.set_xlabel("Count")
        ax.set_title(f"{label_name} loop types (n={len(sub)})")
        for i, (lt, cnt) in enumerate(counts.items()):
            ax.text(cnt + 0.5, i, f"{cnt/len(sub)*100:.1f}%", va="center")

    fig.tight_layout()
    fig.savefig(output_dir / "loop_type_distribution.png")
    plt.close(fig)
    logger.info("Saved loop_type_distribution.png")


def plot_dist_to_junction(df, output_dir):
    """Box plot: distance to loop-stem junction, positive vs negative."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # All sites
    ax = axes[0]
    pos_vals = df[df["label"] == 1]["dist_to_junction"].dropna().values
    neg_vals = df[df["label"] == 0]["dist_to_junction"].dropna().values
    data = []
    labels_bp = []
    if len(pos_vals) > 0:
        data.append(pos_vals)
        labels_bp.append(f"Positive\n(n={len(pos_vals)})")
    if len(neg_vals) > 0:
        data.append(neg_vals)
        labels_bp.append(f"Negative\n(n={len(neg_vals)})")
    if data:
        bp = ax.boxplot(data, labels=labels_bp, patch_artist=True)
        colors = ["#3b82f6", "#ef4444"]
        for patch, c in zip(bp["boxes"], colors[:len(data)]):
            patch.set_facecolor(c)
            patch.set_alpha(0.4)
    ax.set_ylabel("Distance to junction (nt)")
    ax.set_title("Distance to loop-stem junction (all sites)")

    # Unpaired only
    ax = axes[1]
    pos_vals = df[(df["label"] == 1) & (df["is_unpaired"] == True)]["dist_to_nearest_stem"].dropna().values
    neg_vals = df[(df["label"] == 0) & (df["is_unpaired"] == True)]["dist_to_nearest_stem"].dropna().values
    data = []
    labels_bp = []
    if len(pos_vals) > 0:
        data.append(pos_vals)
        labels_bp.append(f"Positive\n(n={len(pos_vals)})")
    if len(neg_vals) > 0:
        data.append(neg_vals)
        labels_bp.append(f"Negative\n(n={len(neg_vals)})")
    if data:
        bp = ax.boxplot(data, labels=labels_bp, patch_artist=True)
        colors = ["#3b82f6", "#ef4444"]
        for patch, c in zip(bp["boxes"], colors[:len(data)]):
            patch.set_facecolor(c)
            patch.set_alpha(0.4)
    ax.set_ylabel("Distance to nearest stem (nt)")
    ax.set_title("Distance to nearest stem (unpaired sites only)")

    fig.tight_layout()
    fig.savefig(output_dir / "dist_to_junction.png")
    plt.close(fig)
    logger.info("Saved dist_to_junction.png")


def plot_stem_length(df, output_dir):
    """Histogram: adjacent stem length distribution."""
    fig, ax = plt.subplots(figsize=(8, 4))

    for label, label_name, color in [(1, "Positive", "#3b82f6"),
                                      (0, "Negative", "#ef4444")]:
        sub = df[df["label"] == label]
        vals = sub["max_adjacent_stem_length"].dropna().values
        if len(vals) == 0:
            continue
        bins = np.arange(0.5, min(vals.max() + 2, 51), 1)
        ax.hist(vals, bins=bins, alpha=0.5,
                label=f"{label_name} (n={len(vals)})",
                color=color, density=True, edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Max adjacent stem length (bp)")
    ax.set_ylabel("Density")
    ax.set_title("Adjacent stem length distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "stem_length_distribution.png")
    plt.close(fig)
    logger.info("Saved stem_length_distribution.png")


def plot_per_dataset_paired_fraction(df, output_dir):
    """Bar chart: paired fraction per dataset."""
    fig, ax = plt.subplots(figsize=(10, 5))

    datasets = df["dataset_source"].unique()
    dataset_stats = []
    for ds in sorted(datasets):
        sub = df[df["dataset_source"] == ds]
        n_total = len(sub)
        if n_total == 0:
            continue
        n_unpaired = sub["is_unpaired"].sum()
        ds_label = DATASET_LABELS.get(ds, ds)
        dataset_stats.append({
            "dataset": ds_label,
            "unpaired_frac": n_unpaired / n_total,
            "n": n_total,
        })

    if not dataset_stats:
        plt.close()
        return

    x = np.arange(len(dataset_stats))
    ax.bar(x, [d["unpaired_frac"] for d in dataset_stats], color="#6366f1")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{d['dataset']}\n(n={d['n']})" for d in dataset_stats],
        rotation=30, ha="right",
    )
    ax.set_ylabel("Fraction unpaired (in loop)")
    ax.set_title("Fraction of sites in loop per dataset")
    ax.set_ylim(0, 1.0)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(output_dir / "per_dataset_paired_fraction.png")
    plt.close(fig)
    logger.info("Saved per_dataset_paired_fraction.png")


def plot_apex_position(df, output_dir):
    """Histogram: distance from apex for unpaired sites."""
    fig, ax = plt.subplots(figsize=(8, 4))

    for label, label_name, color in [(1, "Positive", "#3b82f6"),
                                      (0, "Negative", "#ef4444")]:
        sub = df[(df["label"] == label) & (df["is_unpaired"] == True)]
        vals = sub["dist_to_apex"].dropna().values
        if len(vals) == 0:
            continue
        bins = np.arange(-0.5, min(vals.max() + 2, 31), 1)
        ax.hist(vals, bins=bins, alpha=0.5,
                label=f"{label_name} (n={len(vals)})",
                color=color, density=True, edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Distance from loop apex (nt)")
    ax.set_ylabel("Density")
    ax.set_title("Position within loop: distance from apex (unpaired sites)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "dist_to_apex.png")
    plt.close(fig)
    logger.info("Saved dist_to_apex.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logger.info("=== Loop Position Analysis ===")

    # 1. Load data
    if not SEQ_JSON.exists():
        logger.error("Sequences JSON not found: %s", SEQ_JSON)
        sys.exit(1)
    if not SPLITS_CSV.exists():
        logger.error("Splits CSV not found: %s", SPLITS_CSV)
        sys.exit(1)

    with open(SEQ_JSON) as f:
        sequences = json.load(f)
    logger.info("Loaded %d sequences from %s", len(sequences), SEQ_JSON)

    splits = pd.read_csv(SPLITS_CSV)
    logger.info("Loaded %d sites from %s", len(splits), SPLITS_CSV)

    # 2. Fold sequences and analyze structures
    import RNA
    logger.info("ViennaRNA version: %s", RNA.__version__)

    records = []
    n_total = len(splits)
    n_folded = 0
    n_skipped = 0

    for i, (_, row) in enumerate(splits.iterrows()):
        site_id = row["site_id"]
        seq = sequences.get(site_id)

        if not seq or len(seq) != 201:
            n_skipped += 1
            continue

        # Fold with ViennaRNA
        try:
            dot_bracket, mfe = fold_sequence(seq)
        except Exception as e:
            logger.warning("Folding failed for %s: %s", site_id, e)
            n_skipped += 1
            continue

        # Analyze structure
        features = analyze_site_structure(dot_bracket, EDIT_POS)
        features["mfe"] = float(mfe)
        features["site_id"] = site_id
        features["label"] = row["label"]
        features["dataset_source"] = row["dataset_source"]
        features["editing_rate"] = row.get("editing_rate", None)
        features["dot_bracket"] = dot_bracket
        records.append(features)
        n_folded += 1

        if (i + 1) % 500 == 0:
            logger.info("Processed %d/%d sites (%d folded)", i + 1, n_total, n_folded)

    logger.info("Folding complete: %d folded, %d skipped", n_folded, n_skipped)

    df = pd.DataFrame(records)
    logger.info("Results DataFrame: %d rows, %d columns", len(df), len(df.columns))

    # 3. Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 4. Compute comparison statistics
    results = {}

    # -- Paired vs unpaired fraction --
    for label, name in [(1, "positive"), (0, "negative")]:
        sub = df[df["label"] == label]
        n = len(sub)
        if n == 0:
            continue
        n_unpaired = int(sub["is_unpaired"].sum())
        results[f"{name}_total"] = n
        results[f"{name}_unpaired"] = n_unpaired
        results[f"{name}_paired"] = n - n_unpaired
        results[f"{name}_unpaired_fraction"] = n_unpaired / n
        results[f"{name}_paired_fraction"] = (n - n_unpaired) / n

    # Chi-squared test for paired vs unpaired
    from scipy.stats import chi2_contingency
    pos_sub = df[df["label"] == 1]
    neg_sub = df[df["label"] == 0]
    if len(pos_sub) > 0 and len(neg_sub) > 0:
        table = pd.crosstab(df["label"], df["is_unpaired"])
        try:
            chi2, chi2_p, dof, expected = chi2_contingency(table)
            results["chi2_paired_vs_unpaired"] = float(chi2)
            results["chi2_p_paired_vs_unpaired"] = float(chi2_p)
        except Exception:
            pass

    # -- Feature comparisons (Mann-Whitney) --
    features_to_compare = [
        ("loop_size", "Loop size (unpaired only)", True),
        ("dist_to_nearest_stem", "Distance to nearest stem (unpaired only)", True),
        ("dist_to_apex", "Distance from apex (unpaired only)", True),
        ("relative_loop_position", "Relative loop position (unpaired only)", True),
        ("max_adjacent_stem_length", "Max adjacent stem length", False),
        ("dist_to_junction", "Distance to junction", False),
        ("local_unpaired_fraction", "Local unpaired fraction (+/-10nt)", False),
        ("left_stem_length", "Left stem length", False),
        ("right_stem_length", "Right stem length", False),
        ("mfe", "Minimum free energy (kcal/mol)", False),
    ]

    feature_comparisons = {}
    for feat_col, feat_name, unpaired_only in features_to_compare:
        if unpaired_only:
            pos_vals = df[(df["label"] == 1) & (df["is_unpaired"] == True)][feat_col].dropna().values
            neg_vals = df[(df["label"] == 0) & (df["is_unpaired"] == True)][feat_col].dropna().values
        else:
            pos_vals = df[df["label"] == 1][feat_col].dropna().values
            neg_vals = df[df["label"] == 0][feat_col].dropna().values

        u_stat, p_val = safe_mannwhitneyu(pos_vals, neg_vals)

        feature_comparisons[feat_col] = {
            "name": feat_name,
            "unpaired_only": unpaired_only,
            "positive": compute_group_stats(pos_vals),
            "negative": compute_group_stats(neg_vals),
            "mannwhitney_U": u_stat,
            "mannwhitney_p": p_val,
            "significant_0.05": p_val < 0.05 if not np.isnan(p_val) else None,
            "significant_0.01": p_val < 0.01 if not np.isnan(p_val) else None,
        }

    results["feature_comparisons"] = feature_comparisons

    # -- Per-dataset breakdown --
    per_dataset = {}
    for ds in sorted(df["dataset_source"].unique()):
        sub = df[df["dataset_source"] == ds]
        n = len(sub)
        if n == 0:
            continue
        n_unpaired = int(sub["is_unpaired"].sum())
        ds_info = {
            "n": n,
            "n_unpaired": n_unpaired,
            "n_paired": n - n_unpaired,
            "unpaired_fraction": n_unpaired / n,
            "loop_size": compute_group_stats(
                sub[sub["is_unpaired"] == True]["loop_size"].dropna().values
            ),
            "dist_to_junction": compute_group_stats(
                sub["dist_to_junction"].dropna().values
            ),
            "max_adjacent_stem_length": compute_group_stats(
                sub["max_adjacent_stem_length"].dropna().values
            ),
            "mfe": compute_group_stats(sub["mfe"].dropna().values),
        }
        # Loop type distribution
        unpaired_sub = sub[sub["is_unpaired"] == True]
        if len(unpaired_sub) > 0:
            lt_counts = unpaired_sub["loop_type"].value_counts().to_dict()
            ds_info["loop_type_distribution"] = {
                k: int(v) for k, v in lt_counts.items()
            }
        per_dataset[DATASET_LABELS.get(ds, ds)] = ds_info

    results["per_dataset"] = per_dataset

    # -- Loop type distribution overall --
    loop_type_dist = {}
    for label, name in [(1, "positive"), (0, "negative")]:
        sub = df[(df["label"] == label) & (df["is_unpaired"] == True)]
        if len(sub) > 0:
            lt = sub["loop_type"].value_counts().to_dict()
            loop_type_dist[name] = {k: int(v) for k, v in lt.items()}
    results["loop_type_distribution"] = loop_type_dist

    # -- Loop size distribution (binned) --
    loop_size_bins = {}
    for label, name in [(1, "positive"), (0, "negative")]:
        sub = df[(df["label"] == label) & (df["is_unpaired"] == True)]
        vals = sub["loop_size"].dropna().values
        if len(vals) == 0:
            continue
        bins = [1, 3, 5, 8, 12, 20, 50, 200]
        bin_labels = ["1-2", "3-4", "5-7", "8-11", "12-19", "20-49", "50+"]
        hist, _ = np.histogram(vals, bins=bins)
        loop_size_bins[name] = {
            bl: int(h) for bl, h in zip(bin_labels, hist)
        }
    results["loop_size_bins"] = loop_size_bins

    # 5. Save results JSON
    results_path = OUTPUT_DIR / "loop_position_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Saved results to %s", results_path)

    # 6. Save per-site CSV for downstream use
    csv_cols = [
        "site_id", "label", "dataset_source", "editing_rate",
        "is_unpaired", "loop_type", "loop_size",
        "dist_to_left_boundary", "dist_to_right_boundary",
        "dist_to_nearest_stem", "relative_loop_position", "dist_to_apex",
        "left_stem_length", "right_stem_length", "max_adjacent_stem_length",
        "dist_to_junction", "local_unpaired_fraction", "mfe",
    ]
    csv_cols = [c for c in csv_cols if c in df.columns]
    df[csv_cols].to_csv(OUTPUT_DIR / "loop_position_per_site.csv", index=False)
    logger.info("Saved per-site CSV (%d rows)", len(df))

    # 7. Generate plots
    logger.info("Generating plots...")
    plot_paired_vs_unpaired(df, OUTPUT_DIR)
    plot_loop_size_distribution(df, OUTPUT_DIR)
    plot_loop_type_distribution(df, OUTPUT_DIR)
    plot_dist_to_junction(df, OUTPUT_DIR)
    plot_stem_length(df, OUTPUT_DIR)
    plot_per_dataset_paired_fraction(df, OUTPUT_DIR)
    plot_apex_position(df, OUTPUT_DIR)

    # 8. Print summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info("Total sites analyzed: %d", len(df))
    logger.info("  Positive: %d, Negative: %d",
                results.get("positive_total", 0),
                results.get("negative_total", 0))
    logger.info("  Positive unpaired: %.1f%%",
                results.get("positive_unpaired_fraction", 0) * 100)
    logger.info("  Negative unpaired: %.1f%%",
                results.get("negative_unpaired_fraction", 0) * 100)

    if "chi2_p_paired_vs_unpaired" in results:
        logger.info("  Chi2 test (paired vs unpaired): p=%.2e",
                    results["chi2_p_paired_vs_unpaired"])

    logger.info("\nFeature comparisons (positive vs negative):")
    for feat_col, info in feature_comparisons.items():
        pos_stats = info["positive"]
        neg_stats = info["negative"]
        sig = "*" if info.get("significant_0.05") else ""
        sig += "*" if info.get("significant_0.01") else ""
        logger.info(
            "  %s: pos=%.2f+/-%.2f (n=%d), neg=%.2f+/-%.2f (n=%d), p=%.2e %s",
            info["name"],
            pos_stats["mean"] if pos_stats["mean"] is not None else 0,
            pos_stats["std"] if pos_stats["std"] is not None else 0,
            pos_stats["n"],
            neg_stats["mean"] if neg_stats["mean"] is not None else 0,
            neg_stats["std"] if neg_stats["std"] is not None else 0,
            neg_stats["n"],
            info["mannwhitney_p"] if not np.isnan(info["mannwhitney_p"]) else 0,
            sig,
        )

    logger.info("\nPer-dataset unpaired fraction:")
    for ds, info in per_dataset.items():
        logger.info("  %s: %.1f%% unpaired (n=%d)",
                    ds, info["unpaired_fraction"] * 100, info["n"])

    logger.info("\nOutputs saved to: %s", OUTPUT_DIR)
    logger.info("Done.")


if __name__ == "__main__":
    main()
