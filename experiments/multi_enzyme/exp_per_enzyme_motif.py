#!/usr/bin/env python
"""Per-enzyme motif analysis for multi-APOBEC editing sites.

For each enzyme (A1, A3A, A3B, A3G, A3H, A4), analyzes:
1. TC motif fraction and comparison to random expectation (25%)
2. Position-specific nucleotide frequencies (-10 to +10)
3. Information content at each position
4. Trinucleotide and dinucleotide context distributions
5. Fisher exact test: TC fraction vs random baseline
6. Per-enzyme comparison of preferred motifs

Input:  data/processed/multi_enzyme/splits_multi_enzyme_v2.csv
        data/processed/multi_enzyme/multi_enzyme_sequences_v2.json
Output: experiments/multi_enzyme/outputs/motif_analysis/

Usage:
    conda run -n quris python experiments/multi_enzyme/exp_per_enzyme_motif.py
"""

import json
import logging
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, chi2_contingency

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
SEQ_JSON = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "multi_enzyme_sequences_v2.json"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "multi_enzyme" / "outputs" / "motif_analysis"

BASES = ["A", "C", "G", "U"]

# Expected enzymes and their known motif preferences (for validation)
ENZYME_INFO = {
    # primary_motif: which dinucleotide motif to validate (TC or CC)
    # expected_tc: fraction with T/U at -1; expected_cc: fraction with C at -1
    # Sanity checks gate on expected > 0.4, so low values (e.g. A3G expected_tc=0.05) won't trigger false warnings
    "A3A": {"primary_motif": "TC", "expected_tc": 0.79, "expected_cc": 0.10, "motif_desc": "5'-TC-3' (UCG/UCA preferred)"},
    "A3B": {"primary_motif": "CC", "expected_tc": 0.60, "expected_cc": 0.25, "motif_desc": "5'-UCC-3' (C at -1, C at -2)"},
    "A3G": {"primary_motif": "CC", "expected_tc": 0.05, "expected_cc": 0.60, "motif_desc": "5'-CC-3' (NOT TC — C at -1 position)"},
    "A1":  {"primary_motif": None, "expected_tc": 0.25, "expected_cc": 0.25, "motif_desc": "AU-rich 3' UTR, no strong dinucleotide preference"},
    "A3H": {"primary_motif": None, "expected_tc": None, "expected_cc": None, "motif_desc": "Unknown (expression-correlated sites only)"},
    "A4":  {"primary_motif": None, "expected_tc": None, "expected_cc": None, "motif_desc": "No confirmed deaminase activity — negative control"},
}

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
# Analysis functions
# ---------------------------------------------------------------------------

def get_context(seq, center=100, window=10):
    """Extract flanking context around the edit site."""
    seq = seq.upper().replace("T", "U")
    start = max(0, center - window)
    end = min(len(seq), center + window + 1)
    return seq[start:end], center - start


def compute_position_frequencies(seqs, center=100, window=10):
    """Compute nucleotide frequencies at each position relative to edit site."""
    context_len = 2 * window + 1
    counts = np.zeros((context_len, 4))

    for seq in seqs:
        ctx, c_pos = get_context(seq, center, window)
        for i, nt in enumerate(ctx):
            pos = i - c_pos + window
            if 0 <= pos < context_len and nt in BASES:
                counts[pos, BASES.index(nt)] += 1

    total = counts.sum(axis=1, keepdims=True)
    freqs = counts / np.maximum(total, 1)
    return freqs, counts


def compute_information_content(freqs):
    """Compute information content (bits) at each position."""
    ic = np.zeros(freqs.shape[0])
    for i in range(freqs.shape[0]):
        entropy = 0
        for j in range(4):
            if freqs[i, j] > 0:
                entropy -= freqs[i, j] * np.log2(freqs[i, j])
        ic[i] = 2.0 - entropy
    return ic


def compute_tc_fraction(seqs, center=100):
    """Compute fraction of sites with T/U at position -1 (5'-TC-3' motif)."""
    n_tc = 0
    n_total = 0
    for seq in seqs:
        seq = seq.upper().replace("T", "U")
        if center > 0 and center < len(seq):
            n_total += 1
            if seq[center - 1] == "U":
                n_tc += 1
    return n_tc, n_total


def compute_minus1_fractions(seqs, center=100):
    """Compute nucleotide fractions at position -1 (immediately upstream of edit site).

    Returns dict with counts for each base at -1, plus:
      - cc_m1: fraction with C at -1 (CC dinucleotide context, A3G/A3B signature)
      - strict_cc_m2m1: fraction with C at both -2 AND -1 (strict CC, A3B UCC)
    """
    counts = {"A": 0, "C": 0, "G": 0, "U": 0}
    n_strict_cc = 0  # C at both -2 and -1
    n_total = 0

    for seq in seqs:
        seq = seq.upper().replace("T", "U")
        if center > 0 and center < len(seq):
            n_total += 1
            nt_m1 = seq[center - 1]
            if nt_m1 in counts:
                counts[nt_m1] += 1

            # Strict CC: C at both -2 and -1
            if center >= 2 and seq[center - 1] == "C" and seq[center - 2] == "C":
                n_strict_cc += 1

    return {
        "n_total": n_total,
        "minus1_counts": counts,
        "cc_m1_count": counts["C"],       # C at -1 (CC motif)
        "cc_m1_fraction": counts["C"] / max(n_total, 1),
        "strict_cc_m2m1_count": n_strict_cc,  # C at -2 AND -1
        "strict_cc_m2m1_fraction": n_strict_cc / max(n_total, 1),
    }


def compute_dinucleotide_context(seqs, center=100):
    """Compute dinucleotide at position -1,0."""
    counter = Counter()
    for seq in seqs:
        seq = seq.upper().replace("T", "U")
        if center >= 1 and center < len(seq):
            di = seq[center - 1:center + 1]
            if len(di) == 2:
                counter[di] += 1
    return counter


def compute_trinucleotide_context(seqs, center=100):
    """Compute trinucleotide at position -1,0,+1."""
    counter = Counter()
    for seq in seqs:
        seq = seq.upper().replace("T", "U")
        if center >= 1 and center + 1 < len(seq):
            tri = seq[center - 1:center + 2]
            if len(tri) == 3:
                counter[tri] += 1
    return counter


def tc_fisher_test(n_tc, n_total, expected_frac=0.25):
    """Fisher exact test: is TC fraction significantly different from expected?

    Uses a 2x2 table:
        [[n_tc, n_total - n_tc],
         [expected_tc_count, expected_non_tc_count]]
    where expected counts are from the null hypothesis.
    """
    if n_total == 0:
        return float("nan"), float("nan")
    n_non_tc = n_total - n_tc
    expected_tc = int(n_total * expected_frac)
    expected_non_tc = n_total - expected_tc
    table = [[n_tc, n_non_tc], [expected_tc, expected_non_tc]]
    try:
        odds_ratio, p_value = fisher_exact(table, alternative="two-sided")
        return float(odds_ratio), float(p_value)
    except Exception:
        return float("nan"), float("nan")


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------

def plot_tc_fraction_comparison(enzyme_results, output_dir):
    """Bar chart comparing TC fraction across enzymes."""
    fig, ax = plt.subplots(figsize=(10, 5))

    enzymes = sorted(e for e in enzyme_results.keys() if isinstance(enzyme_results[e], dict) and "tc_fraction" in enzyme_results[e])
    tc_fracs = [enzyme_results[e]["tc_fraction"] for e in enzymes]
    n_sites = [enzyme_results[e]["n_sites"] for e in enzymes]
    colors = [ENZYME_COLORS.get(e, "#999999") for e in enzymes]
    sig_markers = []
    for e in enzymes:
        p = enzyme_results[e].get("fisher_tc_p", enzyme_results[e].get("fisher_p", 1.0))
        if p < 0.001:
            sig_markers.append("***")
        elif p < 0.01:
            sig_markers.append("**")
        elif p < 0.05:
            sig_markers.append("*")
        else:
            sig_markers.append("ns")

    x = np.arange(len(enzymes))
    bars = ax.bar(x, tc_fracs, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)

    ax.axhline(0.25, color="gray", linestyle="--", alpha=0.7, label="Random (25%)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{e}\n(n={n})" for e, n in zip(enzymes, n_sites)])
    ax.set_ylabel("TC Motif Fraction")
    ax.set_title("5'-TC-3' Motif Fraction by APOBEC Enzyme")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right")

    for bar, sig, frac in zip(bars, sig_markers, tc_fracs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{frac:.1%}\n{sig}", ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "tc_fraction_comparison.png")
    plt.close(fig)
    logger.info("Saved tc_fraction_comparison.png")


def plot_tc_vs_cc_comparison(enzyme_results, output_dir):
    """Side-by-side grouped bar chart: TC vs CC fraction per enzyme.

    This is the key discriminator plot:
    - A3A: high TC, low CC
    - A3G: low TC, high CC
    - A3B: mixed (TC or CC depending on context)
    """
    enzymes = sorted(e for e in enzyme_results.keys() if isinstance(enzyme_results[e], dict) and "tc_fraction" in enzyme_results[e])
    if not enzymes:
        return

    fig, ax = plt.subplots(figsize=(12, 5))

    x = np.arange(len(enzymes))
    width = 0.25

    tc_fracs = [enzyme_results[e]["tc_fraction"] for e in enzymes]
    cc_m1_fracs = [enzyme_results[e].get("cc_m1_fraction", 0) for e in enzymes]
    strict_cc_fracs = [enzyme_results[e].get("strict_cc_m2m1_fraction", 0) for e in enzymes]
    n_sites = [enzyme_results[e]["n_sites"] for e in enzymes]

    bars1 = ax.bar(x - width, tc_fracs, width, label="TC (U at -1)",
                   color="#2563eb", alpha=0.8, edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x, cc_m1_fracs, width, label="CC (C at -1)",
                   color="#dc2626", alpha=0.8, edgecolor="black", linewidth=0.5)
    bars3 = ax.bar(x + width, strict_cc_fracs, width, label="Strict CC (C at -2,-1)",
                   color="#16a34a", alpha=0.8, edgecolor="black", linewidth=0.5)

    ax.axhline(0.25, color="gray", linestyle="--", alpha=0.5, label="Random (25%)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{e}\n(n={n})" for e, n in zip(enzymes, n_sites)])
    ax.set_ylabel("Fraction")
    ax.set_title("TC vs CC Motif Fraction by Enzyme\n"
                 "(A3A = TC enzyme, A3G = CC enzyme, A3B = UCC)")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right", fontsize=8)

    # Annotate TC bars
    for bar, frac in zip(bars1, tc_fracs):
        if frac > 0.03:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{frac:.0%}", ha="center", fontsize=7, color="#2563eb")
    # Annotate CC bars
    for bar, frac in zip(bars2, cc_m1_fracs):
        if frac > 0.03:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{frac:.0%}", ha="center", fontsize=7, color="#dc2626")

    fig.tight_layout()
    fig.savefig(output_dir / "tc_vs_cc_comparison.png")
    plt.close(fig)
    logger.info("Saved tc_vs_cc_comparison.png")


def plot_position_frequencies_grid(enzyme_freq_data, output_dir, window=10):
    """Grid of position-specific frequency heatmaps, one per enzyme."""
    enzymes = sorted(enzyme_freq_data.keys())
    n_enzymes = len(enzymes)
    if n_enzymes == 0:
        return

    n_cols = min(3, n_enzymes)
    n_rows = (n_enzymes + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    if n_enzymes == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    positions = list(range(-window, window + 1))

    for idx, enzyme in enumerate(enzymes):
        ax = axes[idx]
        freqs = np.array(enzyme_freq_data[enzyme]["frequencies"])
        n_sites = enzyme_freq_data[enzyme]["n_sites"]

        im = ax.imshow(freqs.T, aspect="auto", cmap="YlOrRd", vmin=0, vmax=0.6)
        ax.set_yticks(range(4))
        ax.set_yticklabels(BASES)
        ax.set_xticks(range(0, len(positions), 2))
        ax.set_xticklabels([positions[i] for i in range(0, len(positions), 2)], fontsize=7)
        ax.set_xlabel("Position relative to edit site")
        ax.set_title(f"{enzyme} (n={n_sites})")
        plt.colorbar(im, ax=ax, shrink=0.7, label="Frequency")

    for idx in range(n_enzymes, len(axes)):
        axes[idx].axis("off")

    plt.suptitle("Position-Specific Nucleotide Frequencies by Enzyme", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_dir / "position_frequencies_grid.png")
    plt.close(fig)
    logger.info("Saved position_frequencies_grid.png")


def plot_information_content_overlay(enzyme_freq_data, output_dir, window=10):
    """Overlay IC plots for all enzymes."""
    fig, ax = plt.subplots(figsize=(12, 5))
    positions = list(range(-window, window + 1))

    for enzyme in sorted(enzyme_freq_data.keys()):
        ic = enzyme_freq_data[enzyme]["information_content"]
        color = ENZYME_COLORS.get(enzyme, "#999999")
        n = enzyme_freq_data[enzyme]["n_sites"]
        ax.plot(positions, ic, color=color, linewidth=1.5,
                label=f"{enzyme} (n={n})", alpha=0.8)

    ax.axhline(0, color="gray", linestyle="--", alpha=0.3)
    ax.axvline(0, color="black", linestyle=":", alpha=0.3, label="Edit site")
    ax.set_xlabel("Position relative to edit site")
    ax.set_ylabel("Information Content (bits)")
    ax.set_title("Sequence Information Content by Enzyme")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(-0.1, 1.5)

    fig.tight_layout()
    fig.savefig(output_dir / "information_content_overlay.png")
    plt.close(fig)
    logger.info("Saved information_content_overlay.png")


def plot_dinucleotide_heatmap(enzyme_results, output_dir):
    """Heatmap of dinucleotide (-1,0) frequencies across enzymes."""
    enzymes = sorted(enzyme_results.keys())
    all_dinucs = sorted(set(
        di for e in enzymes for di in enzyme_results[e].get("dinucleotides", {})
    ))

    if not all_dinucs or not enzymes:
        return

    matrix = np.zeros((len(enzymes), len(all_dinucs)))
    for i, e in enumerate(enzymes):
        di_counts = enzyme_results[e].get("dinucleotides", {})
        total = sum(di_counts.values()) or 1
        for j, di in enumerate(all_dinucs):
            matrix[i, j] = di_counts.get(di, 0) / total

    fig, ax = plt.subplots(figsize=(12, max(4, len(enzymes) * 0.8)))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0)
    ax.set_yticks(range(len(enzymes)))
    ax.set_yticklabels(enzymes)
    ax.set_xticks(range(len(all_dinucs)))
    ax.set_xticklabels(all_dinucs, rotation=45, ha="right", fontsize=8)
    ax.set_title("Dinucleotide (-1,0) Frequency by Enzyme")
    plt.colorbar(im, ax=ax, shrink=0.7, label="Fraction")

    # Annotate cells
    for i in range(len(enzymes)):
        for j in range(len(all_dinucs)):
            val = matrix[i, j]
            if val > 0.01:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if val > 0.3 else "black")

    fig.tight_layout()
    fig.savefig(output_dir / "dinucleotide_heatmap.png")
    plt.close(fig)
    logger.info("Saved dinucleotide_heatmap.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    if not SPLITS_CSV.exists():
        logger.error("Multi-enzyme splits CSV not found: %s", SPLITS_CSV)
        logger.error("Run data-engineer pipeline first (tasks #1 and #2).")
        sys.exit(1)
    if not SEQ_JSON.exists():
        logger.error("Multi-enzyme sequences JSON not found: %s", SEQ_JSON)
        sys.exit(1)

    df = pd.read_csv(SPLITS_CSV)
    logger.info("Loaded %d sites from %s", len(df), SPLITS_CSV)

    with open(SEQ_JSON) as f:
        sequences = json.load(f)
    logger.info("Loaded %d sequences", len(sequences))

    # Require 'enzyme' column
    if "enzyme" not in df.columns:
        logger.error("No 'enzyme' column in splits CSV. Expected columns: %s", list(df.columns))
        sys.exit(1)

    enzymes = sorted(df["enzyme"].unique())
    logger.info("Enzymes found: %s", enzymes)

    # Filter to positive sites only for motif analysis
    # Multi-enzyme datasets have all-positive sites; single-enzyme datasets may have is_edited/label
    if "is_edited" in df.columns:
        pos_df = df[df["is_edited"] == 1]
    elif "label" in df.columns:
        pos_df = df[df["label"] == 1]
    else:
        # All sites are editing sites (no negatives in this dataset)
        pos_df = df.copy()
        logger.info("No is_edited/label column found — treating all sites as positive")
    logger.info("Positive sites: %d", len(pos_df))

    all_results = {}
    enzyme_freq_data = {}

    window = 10

    for enzyme in enzymes:
        logger.info("=" * 60)
        logger.info("Analyzing enzyme: %s", enzyme)
        logger.info("=" * 60)

        enz_df = pos_df[pos_df["enzyme"] == enzyme]
        enz_seqs = []
        for sid in enz_df["site_id"]:
            seq = sequences.get(str(sid))
            if seq and len(seq) >= 201:
                enz_seqs.append(seq)

        n_sites = len(enz_seqs)
        if n_sites == 0:
            logger.warning("No sequences for enzyme %s, skipping", enzyme)
            continue

        logger.info("  %d sequences available", n_sites)

        # TC fraction (U at position -1)
        n_tc, n_total = compute_tc_fraction(enz_seqs)
        tc_frac = n_tc / max(n_total, 1)
        tc_odds_ratio, tc_fisher_p = tc_fisher_test(n_tc, n_total)

        # CC / strict CC fractions (C at -1, C at -2 and -1)
        m1_data = compute_minus1_fractions(enz_seqs)
        cc_m1_frac = m1_data["cc_m1_fraction"]
        cc_m1_count = m1_data["cc_m1_count"]
        strict_cc_frac = m1_data["strict_cc_m2m1_fraction"]
        strict_cc_count = m1_data["strict_cc_m2m1_count"]

        # Fisher test for CC at -1 vs random (25%)
        cc_odds_ratio, cc_fisher_p = tc_fisher_test(cc_m1_count, n_total)

        # Sanity checks: validate each enzyme's PRIMARY motif fraction.
        # Only fires when expected > 0.4 (i.e. a known motif enzyme) but observed < 0.05.
        # A3G (primary=CC, expected_tc=0.05) will NOT trigger the TC check.
        enz_info = ENZYME_INFO.get(enzyme, {})
        primary_motif = enz_info.get("primary_motif")
        expected_tc = enz_info.get("expected_tc")
        expected_cc = enz_info.get("expected_cc")

        if expected_tc is not None and expected_tc > 0.4 and tc_frac < 0.05:
            logger.warning(
                "  SANITY FAIL: %s TC fraction = %.3f (expected ~%.0f%%, primary_motif=%s). "
                "Coordinates may be wrong or sequences misextracted!",
                enzyme, tc_frac, expected_tc * 100, primary_motif
            )
        if expected_cc is not None and expected_cc > 0.4 and cc_m1_frac < 0.05:
            logger.warning(
                "  SANITY FAIL: %s CC fraction = %.3f (expected ~%.0f%%, primary_motif=%s). "
                "Coordinates may be wrong or sequences misextracted!",
                enzyme, cc_m1_frac, expected_cc * 100, primary_motif
            )

        # Position-specific frequencies
        freqs, counts = compute_position_frequencies(enz_seqs, window=window)
        ic = compute_information_content(freqs)

        enzyme_freq_data[enzyme] = {
            "frequencies": freqs.tolist(),
            "information_content": ic.tolist(),
            "n_sites": n_sites,
        }

        # Dinucleotide and trinucleotide
        di_counts = compute_dinucleotide_context(enz_seqs)
        tri_counts = compute_trinucleotide_context(enz_seqs)

        result = {
            "enzyme": enzyme,
            "n_sites": n_sites,
            # TC motif (U at -1)
            "tc_fraction": tc_frac,
            "tc_count": n_tc,
            "fisher_tc_odds_ratio": tc_odds_ratio,
            "fisher_tc_p": tc_fisher_p,
            "tc_significant_vs_random": tc_fisher_p < 0.05 if not np.isnan(tc_fisher_p) else None,
            # CC motif (C at -1)
            "cc_m1_fraction": cc_m1_frac,
            "cc_m1_count": cc_m1_count,
            "fisher_cc_odds_ratio": cc_odds_ratio,
            "fisher_cc_p": cc_fisher_p,
            "cc_significant_vs_random": cc_fisher_p < 0.05 if not np.isnan(cc_fisher_p) else None,
            # Strict CC (C at -2 AND -1, relevant for A3B UCC motif)
            "strict_cc_m2m1_fraction": strict_cc_frac,
            "strict_cc_m2m1_count": strict_cc_count,
            # Per-base fractions at position -1
            "minus1_base_fractions": {
                b: m1_data["minus1_counts"][b] / max(n_total, 1)
                for b in BASES
            },
            # Legacy cc_fraction (same as cc_m1_fraction for backwards compat)
            "cc_fraction": cc_m1_frac,
            "cc_count": cc_m1_count,
            # Keep old fisher_p key pointing to TC test for backwards compat
            "fisher_odds_ratio": tc_odds_ratio,
            "fisher_p": tc_fisher_p,
            # Context distributions
            "dinucleotides": dict(di_counts.most_common(16)),
            "trinucleotides": dict(tri_counts.most_common(20)),
            "information_content_mean": float(np.mean(ic)),
            "information_content_max": float(np.max(ic)),
            "information_content_at_minus1": float(ic[window - 1]),
        }
        all_results[enzyme] = result

        # Print summary for this enzyme
        print(f"\n--- {enzyme} Motif Summary (n={n_sites}) ---")
        print(f"  Position -1 base fractions: "
              f"A={result['minus1_base_fractions']['A']:.3f}, "
              f"C={result['minus1_base_fractions']['C']:.3f}, "
              f"G={result['minus1_base_fractions']['G']:.3f}, "
              f"U={result['minus1_base_fractions']['U']:.3f}")
        print(f"  TC fraction (U at -1): {tc_frac:.3f} ({n_tc}/{n_total})")
        print(f"  CC fraction (C at -1): {cc_m1_frac:.3f} ({cc_m1_count}/{n_total})")
        print(f"  Strict CC (C at -2,-1): {strict_cc_frac:.3f} ({strict_cc_count}/{n_total})")
        tc_sig = "***" if tc_fisher_p < 0.001 else "**" if tc_fisher_p < 0.01 else "*" if tc_fisher_p < 0.05 else "ns"
        cc_sig = "***" if cc_fisher_p < 0.001 else "**" if cc_fisher_p < 0.01 else "*" if cc_fisher_p < 0.05 else "ns"
        print(f"  Fisher TC vs random: OR={tc_odds_ratio:.2f}, p={tc_fisher_p:.2e} {tc_sig}")
        print(f"  Fisher CC vs random: OR={cc_odds_ratio:.2f}, p={cc_fisher_p:.2e} {cc_sig}")
        print(f"  Top 5 dinucleotides: {dict(di_counts.most_common(5))}")
        print(f"  Top 5 trinucleotides: {dict(tri_counts.most_common(5))}")

    if not all_results:
        logger.error("No enzymes had sequences. Check data pipeline.")
        sys.exit(1)

    # ---------------------------------------------------------------------------
    # Cross-enzyme comparisons
    # ---------------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("Cross-Enzyme Motif Comparison")
    logger.info("=" * 60)

    # Pairwise chi-squared tests for TC fraction
    pairwise_tc = {}
    enzyme_list = sorted(all_results.keys())
    for i, e1 in enumerate(enzyme_list):
        for e2 in enzyme_list[i + 1:]:
            r1 = all_results[e1]
            r2 = all_results[e2]
            table = [
                [r1["tc_count"], r1["n_sites"] - r1["tc_count"]],
                [r2["tc_count"], r2["n_sites"] - r2["tc_count"]],
            ]
            try:
                chi2, p, _, _ = chi2_contingency(table)
                pairwise_tc[f"{e1}_vs_{e2}"] = {
                    "chi2": float(chi2),
                    "p_value": float(p),
                    "significant": p < 0.05,
                    f"{e1}_tc": r1["tc_fraction"],
                    f"{e2}_tc": r2["tc_fraction"],
                }
            except Exception:
                pass

    all_results["pairwise_tc_tests"] = pairwise_tc

    # Pairwise chi-squared tests for CC fraction (C at -1)
    pairwise_cc = {}
    for i, e1 in enumerate(enzyme_list):
        for e2 in enzyme_list[i + 1:]:
            r1 = all_results[e1]
            r2 = all_results[e2]
            table = [
                [r1["cc_m1_count"], r1["n_sites"] - r1["cc_m1_count"]],
                [r2["cc_m1_count"], r2["n_sites"] - r2["cc_m1_count"]],
            ]
            try:
                chi2, p, _, _ = chi2_contingency(table)
                pairwise_cc[f"{e1}_vs_{e2}"] = {
                    "chi2": float(chi2),
                    "p_value": float(p),
                    "significant": p < 0.05,
                    f"{e1}_cc": r1["cc_m1_fraction"],
                    f"{e2}_cc": r2["cc_m1_fraction"],
                }
            except Exception:
                pass

    all_results["pairwise_cc_tests"] = pairwise_cc

    # Print pairwise comparisons
    print("\n--- Pairwise TC Fraction Comparisons (Chi-squared) ---")
    for pair, info in sorted(pairwise_tc.items()):
        sig = "***" if info["p_value"] < 0.001 else "**" if info["p_value"] < 0.01 else "*" if info["p_value"] < 0.05 else "ns"
        print(f"  {pair}: chi2={info['chi2']:.1f}, p={info['p_value']:.2e} {sig}")

    print("\n--- Pairwise CC Fraction Comparisons (Chi-squared) ---")
    for pair, info in sorted(pairwise_cc.items()):
        sig = "***" if info["p_value"] < 0.001 else "**" if info["p_value"] < 0.01 else "*" if info["p_value"] < 0.05 else "ns"
        print(f"  {pair}: chi2={info['chi2']:.1f}, p={info['p_value']:.2e} {sig}")

    # Trinucleotide diversity (Shannon entropy of trinucleotide distribution)
    for enzyme in enzyme_list:
        tri = all_results[enzyme].get("trinucleotides", {})
        total = sum(tri.values())
        if total > 0:
            probs = np.array(list(tri.values())) / total
            shannon = -np.sum(probs * np.log2(probs + 1e-10))
            all_results[enzyme]["trinucleotide_diversity"] = float(shannon)

    # ---------------------------------------------------------------------------
    # Generate plots
    # ---------------------------------------------------------------------------
    logger.info("Generating plots...")
    plot_tc_fraction_comparison(all_results, OUTPUT_DIR)
    plot_tc_vs_cc_comparison(all_results, OUTPUT_DIR)
    plot_position_frequencies_grid(enzyme_freq_data, OUTPUT_DIR, window=window)
    plot_information_content_overlay(enzyme_freq_data, OUTPUT_DIR, window=window)
    plot_dinucleotide_heatmap(all_results, OUTPUT_DIR)

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
        return str(obj)

    with open(OUTPUT_DIR / "per_enzyme_motif_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=serialize)
    logger.info("Results saved to %s", OUTPUT_DIR / "per_enzyme_motif_results.json")

    # Print final summary table
    print("\n" + "=" * 80)
    print("CROSS-ENZYME MOTIF SUMMARY")
    print("=" * 80)
    print(f"{'Enzyme':>6s} {'n':>6s} {'TC%':>7s} {'CC%':>7s} {'CC-2%':>7s} "
          f"{'TC sig':>8s} {'CC sig':>8s} {'IC@-1':>7s} {'Tri-div':>8s}")
    print("-" * 75)
    for enzyme in enzyme_list:
        r = all_results[enzyme]
        tc_sig = "***" if r["fisher_tc_p"] < 0.001 else "ns"
        cc_sig = "***" if r["fisher_cc_p"] < 0.001 else "ns"
        tri_div = r.get("trinucleotide_diversity", 0)
        print(f"{enzyme:>6s} {r['n_sites']:>6d} "
              f"{r['tc_fraction']:>6.1%} {r['cc_m1_fraction']:>6.1%} {r['strict_cc_m2m1_fraction']:>6.1%} "
              f"{tc_sig:>8s} {cc_sig:>8s} "
              f"{r['information_content_at_minus1']:>7.3f} {tri_div:>8.2f}")
    print("=" * 80)

    # Print dominant motif per enzyme
    print("\nDominant motif at position -1:")
    for enzyme in enzyme_list:
        r = all_results[enzyme]
        fracs = r["minus1_base_fractions"]
        dominant = max(fracs, key=fracs.get)
        dominant_frac = fracs[dominant]
        motif_type = {"U": "TC (A3A-like)", "C": "CC (A3G-like)"}.get(dominant, f"{dominant}C")
        print(f"  {enzyme}: {dominant} at -1 ({dominant_frac:.0%}) => {motif_type}")
    print("=" * 80)


if __name__ == "__main__":
    main()
