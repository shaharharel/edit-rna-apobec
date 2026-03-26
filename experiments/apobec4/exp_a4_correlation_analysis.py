#!/usr/bin/env python
"""APOBEC4 Correlation Analysis.

APOBEC4 has no confirmed C-to-U deaminase activity. However, the Levanon dataset
(C2TFinalSites.DB.xlsx, T3 sheet) reports 181 editing sites whose GTEx tissue
editing rates are significantly correlated with APOBEC4 expression
(Pearson R > 0.1, FDR q ≤ 0.05).

This experiment asks: do A4-correlated editing sites have a distinct
sequence/structure profile compared to other Levanon editing sites?

If yes, this suggests either:
  - A4 expression is a proxy for some tissue-specific state that enables
    a particular class of editors
  - A4 regulates or competes with the actual editing enzyme at specific sites

Analyses:
  1. TC motif fraction: A4-correlated vs non-A4-correlated sites
  2. Trinucleotide context: -1,0,+1 around the edited C
  3. Position-specific frequency (±10 nt)
  4. Structure profile: loop vs stem, loop length
  5. Tissue breadth: edited in how many tissues
  6. Multi-enzyme co-occurrence (A4 vs A3A, A3G, A3H)
  7. Comparison: A4-exclusive sites vs A4-co-occurring sites

Output: experiments/apobec4/outputs/a4_analysis_results.json + figures

Usage:
    python experiments/apobec4/exp_a4_correlation_analysis.py
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
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

EXCEL_PATH = PROJECT_ROOT / "data" / "raw" / "C2TFinalSites.DB.xlsx"
SEQ_JSON = PROJECT_ROOT / "data" / "processed" / "site_sequences.json"
UNIFIED_CSV = PROJECT_ROOT / "data" / "processed" / "advisor" / "unified_editing_sites.csv"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "apobec4" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ENZYME_COLS = ["A1", "A3A", "A3B", "A3G", "A3H", "A4"]
SEQ_CENTER = 100  # edit site is at position 100 in 201-nt window


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_t3_flags() -> pd.DataFrame:
    """Load T3 sheet with per-site APOBEC enzyme correlation flags."""
    t3 = pd.read_excel(EXCEL_PATH, sheet_name="T3-APOBECs Correlations", header=2)
    cols_needed = ["Chr", "Start", "End", "Genomic Category", "Gene Name",
                   "Exonic Function", "EditedInTissuesN", "AffectingAPOBEC"] + ENZYME_COLS
    t3 = t3[cols_needed].dropna(subset=["Chr"]).copy()
    t3.columns = ["chr", "start", "end", "genomic_category", "gene_name",
                  "exonic_function", "edited_in_tissues_n", "affecting_apobec"] + ENZYME_COLS
    # Boolean columns
    for col in ENZYME_COLS:
        t3[col] = t3[col].fillna(False).astype(bool)
    t3["start"] = t3["start"].astype(int)
    t3["end"] = t3["end"].astype(int)
    t3["edited_in_tissues_n"] = pd.to_numeric(t3["edited_in_tissues_n"], errors="coerce")
    return t3.reset_index(drop=True)


def load_sequences() -> dict:
    with open(SEQ_JSON) as f:
        return json.load(f)


def load_unified() -> pd.DataFrame:
    return pd.read_csv(UNIFIED_CSV)


def merge_with_site_ids(t3: pd.DataFrame, unified: pd.DataFrame) -> pd.DataFrame:
    """Join T3 flags with C2U site_ids and structure data."""
    unified_key = unified[["site_id", "chr", "start", "structure_structure_type",
                            "structure_loop_length", "max_gtex_editing_rate",
                            "mean_gtex_editing_rate"]].copy()
    unified_key["start"] = unified_key["start"].astype(int)
    merged = t3.merge(unified_key, on=["chr", "start"], how="left")
    n_matched = merged["site_id"].notna().sum()
    logger.info("Matched %d / %d T3 sites to C2U site_ids", n_matched, len(t3))
    return merged


# ---------------------------------------------------------------------------
# Motif analysis
# ---------------------------------------------------------------------------

def get_seq(seqs: dict, site_id: str) -> str | None:
    s = seqs.get(site_id)
    if s is None:
        return None
    if isinstance(s, dict):
        return s.get("sequence") or s.get("seq") or list(s.values())[0]
    return str(s)


def get_context(seq: str, center: int = SEQ_CENTER, window: int = 10):
    seq = seq.upper().replace("T", "U")
    lo = max(0, center - window)
    hi = min(len(seq), center + window + 1)
    ctx = seq[lo:hi]
    rel_center = center - lo
    return ctx, rel_center


def tc_fraction(seqs_list: list[str]) -> float:
    """Fraction of sequences with T (U) at -1 position (TC motif)."""
    tc = 0
    for seq in seqs_list:
        seq_u = seq.upper().replace("T", "U")
        if len(seq_u) > SEQ_CENTER and seq_u[SEQ_CENTER - 1] in ("U", "T"):
            tc += 1
    return tc / len(seqs_list) if seqs_list else 0.0


def trinucleotide_counts(seqs_list: list[str]) -> Counter:
    """Count trinucleotides at -1, 0, +1 (0 = edited C)."""
    counts = Counter()
    for seq in seqs_list:
        seq_u = seq.upper().replace("T", "U")
        if SEQ_CENTER - 1 >= 0 and SEQ_CENTER + 1 < len(seq_u):
            tri = seq_u[SEQ_CENTER - 1 : SEQ_CENTER + 2]
            if all(b in "ACGU" for b in tri):
                counts[tri] += 1
    return counts


def position_frequencies(seqs_list: list[str], window: int = 10) -> np.ndarray:
    """Nucleotide frequencies at each position relative to edit site."""
    bases = list("ACGU")
    n = 2 * window + 1
    counts = np.zeros((n, 4))
    for seq in seqs_list:
        ctx, rel = get_context(seq, SEQ_CENTER, window)
        for i, nt in enumerate(ctx):
            pos = i - rel + window
            if 0 <= pos < n and nt in bases:
                counts[pos, bases.index(nt)] += 1
    freq = counts / (counts.sum(axis=1, keepdims=True) + 1e-9)
    return freq


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_tc_comparison(groups: dict, output_path: Path):
    """Bar chart of TC-motif fraction per group."""
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = list(groups.keys())
    vals = [groups[k]["tc_fraction"] for k in labels]
    ns = [groups[k]["n"] for k in labels]
    colors = ["#e74c3c" if "A4" in k else "#3498db" for k in labels]
    bars = ax.bar(labels, vals, color=colors, alpha=0.8)
    ax.set_ylabel("TC motif fraction")
    ax.set_title("TC Motif Fraction by APOBEC Correlation Group")
    ax.set_ylim(0, 1)
    for bar, n in zip(bars, ns):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"n={n}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_position_freq(freq: np.ndarray, title: str, output_path: Path, window: int = 10):
    """4-row sequence logo-style frequency heatmap."""
    fig, ax = plt.subplots(figsize=(12, 3))
    positions = list(range(-window, window + 1))
    bases = list("ACGU")
    colors = {"A": "#2ecc71", "C": "#3498db", "G": "#f39c12", "U": "#e74c3c"}
    bottom = np.zeros(len(positions))
    for i, base in enumerate(bases):
        vals = freq[:, i]
        ax.bar(positions, vals, bottom=bottom, label=base, color=colors[base], alpha=0.85)
        bottom += vals
    ax.axvline(0, color="black", linewidth=1.5, linestyle="--", alpha=0.6)
    ax.set_xlabel("Position relative to edit site")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.legend(title="Base", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_structure_comparison(df_a4: pd.DataFrame, df_non: pd.DataFrame, output_path: Path):
    """Structure type distribution for A4-correlated vs non-A4-correlated."""
    types = ["In Loop", "dsRNA", "In Stem", "ssRNA"]
    col = "structure_structure_type"
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (label, df) in zip(axes, [("A4-correlated", df_a4), ("Non-A4-correlated", df_non)]):
        counts = df[col].value_counts()
        vals = [counts.get(t, 0) for t in types]
        ax.bar(types, vals, color=["#e74c3c", "#3498db", "#2ecc71", "#f39c12"], alpha=0.8)
        ax.set_title(f"{label} (n={len(df)})")
        ax.set_ylabel("Site count")
        ax.set_xticklabels(types, rotation=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_trinucleotides(groups: dict, top_n: int = 15, output_path: Path = None):
    """Trinucleotide comparison between A4-correlated and non-A4 groups."""
    all_tris = set()
    for g in groups.values():
        all_tris.update(g["trinucleotide_counts"].keys())

    # Rank by A4-correlated fraction
    a4_counts = groups.get("A4-correlated", {}).get("trinucleotide_counts", {})
    a4_total = sum(a4_counts.values()) or 1
    sorted_tris = sorted(all_tris, key=lambda t: a4_counts.get(t, 0) / a4_total, reverse=True)[:top_n]

    x = np.arange(len(sorted_tris))
    width = 0.35
    fig, ax = plt.subplots(figsize=(14, 5))
    for i, (grp_name, data) in enumerate(groups.items()):
        total = sum(data["trinucleotide_counts"].values()) or 1
        vals = [data["trinucleotide_counts"].get(t, 0) / total for t in sorted_tris]
        ax.bar(x + i * width, vals, width, label=grp_name, alpha=0.8)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(sorted_tris, rotation=45, ha="right")
    ax.set_ylabel("Fraction of sites")
    ax.set_title("Top Trinucleotide Contexts (ranked by A4-correlated frequency)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def compare_tc_fisher(n_tc_a: int, n_a: int, n_tc_b: int, n_b: int):
    """Fisher's exact test comparing TC fraction between two groups."""
    table = [[n_tc_a, n_a - n_tc_a], [n_tc_b, n_b - n_tc_b]]
    odds, pval = stats.fisher_exact(table)
    return float(odds), float(pval)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logger.info("Loading data...")
    t3 = load_t3_flags()
    seqs = load_sequences()
    unified = load_unified()

    logger.info("T3 sites: %d, A4-correlated: %d", len(t3), t3["A4"].sum())

    # Merge with site_ids
    df = merge_with_site_ids(t3, unified)

    # Define analysis groups
    a4_mask = df["A4"]
    non_a4_mask = ~df["A4"]
    a4_only_mask = a4_mask & (df[["A1", "A3A", "A3B", "A3G", "A3H"]] == False).all(axis=1)
    a3a_mask = df["A3A"] & ~df["A4"]
    a3g_mask = df["A3G"] & ~df["A4"]

    groups_def = {
        "A4-correlated": a4_mask,
        "A4-exclusive": a4_only_mask,
        "A3A-correlated (no A4)": a3a_mask,
        "A3G-correlated (no A4)": a3g_mask,
        "Non-A4-correlated": non_a4_mask,
    }

    results = {}
    group_data = {}

    for group_name, mask in groups_def.items():
        sub = df[mask]
        site_ids = sub["site_id"].dropna().tolist()
        group_seqs = [get_seq(seqs, sid) for sid in site_ids if get_seq(seqs, sid)]
        n = len(group_seqs)
        logger.info("Group '%s': %d sites, %d with sequences", group_name, len(sub), n)

        if n == 0:
            continue

        tc_frac = tc_fraction(group_seqs)
        tri_counts = dict(trinucleotide_counts(group_seqs).most_common(20))
        pos_freq = position_frequencies(group_seqs, window=10).tolist()

        # Structure distribution
        struct_counts = sub["structure_structure_type"].value_counts().to_dict()
        in_loop_n = struct_counts.get("In Loop", 0)
        in_loop_frac = in_loop_n / len(sub) if len(sub) > 0 else 0.0

        # Tissue breadth
        tissue_n = sub["edited_in_tissues_n"].dropna()
        tissue_mean = float(tissue_n.mean()) if len(tissue_n) > 0 else 0.0
        tissue_median = float(tissue_n.median()) if len(tissue_n) > 0 else 0.0

        results[group_name] = {
            "n_sites": len(sub),
            "n_with_seq": n,
            "tc_fraction": round(tc_frac, 4),
            "trinucleotide_counts": tri_counts,
            "structure_type_counts": {k: int(v) for k, v in struct_counts.items()},
            "in_loop_fraction": round(in_loop_frac, 4),
            "tissue_n_mean": round(tissue_mean, 2),
            "tissue_n_median": tissue_median,
            "position_frequencies": pos_freq,
        }
        group_data[group_name] = {"seqs": group_seqs, "df": sub,
                                   "tc_fraction": tc_frac, "n": n,
                                   "trinucleotide_counts": dict(trinucleotide_counts(group_seqs))}

    # Statistical tests: A4-correlated vs Non-A4-correlated
    if "A4-correlated" in results and "Non-A4-correlated" in results:
        a4_r = results["A4-correlated"]
        non_r = results["Non-A4-correlated"]
        n_tc_a4 = round(a4_r["tc_fraction"] * a4_r["n_with_seq"])
        n_tc_non = round(non_r["tc_fraction"] * non_r["n_with_seq"])
        odds, pval = compare_tc_fisher(n_tc_a4, a4_r["n_with_seq"],
                                       n_tc_non, non_r["n_with_seq"])
        results["stats_tc_fisher"] = {
            "A4_vs_non_A4_odds_ratio": round(odds, 4),
            "A4_vs_non_A4_pvalue": float(f"{pval:.4e}"),
            "A4_tc_n": n_tc_a4,
            "non_A4_tc_n": n_tc_non,
        }
        logger.info("TC Fisher test: OR=%.3f, p=%.4e", odds, pval)

        # Mann-Whitney on tissue breadth
        a4_tissues = df[a4_mask]["edited_in_tissues_n"].dropna()
        non_tissues = df[non_a4_mask]["edited_in_tissues_n"].dropna()
        if len(a4_tissues) > 0 and len(non_tissues) > 0:
            stat, pval_tissue = stats.mannwhitneyu(a4_tissues, non_tissues, alternative="two-sided")
            results["stats_tissue_breadth"] = {
                "A4_mean_tissues": round(float(a4_tissues.mean()), 2),
                "non_A4_mean_tissues": round(float(non_tissues.mean()), 2),
                "mannwhitney_pvalue": float(f"{pval_tissue:.4e}"),
            }
            logger.info("Tissue breadth: A4 mean=%.1f vs non-A4 mean=%.1f, p=%.4e",
                        a4_tissues.mean(), non_tissues.mean(), pval_tissue)

    # Co-occurrence summary
    a4_df = df[a4_mask]
    co_occurrence = {}
    for col in ["A1", "A3A", "A3B", "A3G", "A3H"]:
        n_co = int((a4_df[col] == True).sum())
        co_occurrence[f"A4_and_{col}"] = n_co
    co_occurrence["A4_only"] = int(a4_only_mask.sum())
    results["a4_co_occurrence"] = co_occurrence

    # AffectingAPOBEC distribution for A4 sites
    results["a4_affecting_apobec"] = a4_df["affecting_apobec"].value_counts(dropna=False).to_dict()

    # Genomic category
    results["a4_genomic_category"] = a4_df["genomic_category"].value_counts().to_dict()

    # Save results JSON
    out_json = OUTPUT_DIR / "a4_analysis_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results written to %s", out_json)

    # -----------------------------------------------------------------------
    # Figures
    # -----------------------------------------------------------------------
    logger.info("Generating figures...")

    # 1. TC fraction bar chart
    tc_groups = {k: {"tc_fraction": v["tc_fraction"], "n": v["n_with_seq"]}
                 for k, v in results.items() if isinstance(v, dict) and "tc_fraction" in v}
    plot_tc_comparison(tc_groups, OUTPUT_DIR / "tc_fraction_by_group.png")

    # 2. Position frequency plots for key groups
    for gname in ["A4-correlated", "A4-exclusive", "Non-A4-correlated"]:
        if gname in group_data:
            freq = np.array(results[gname]["position_frequencies"])
            safe = gname.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "")
            plot_position_freq(freq, f"Position Frequency: {gname}",
                               OUTPUT_DIR / f"pos_freq_{safe}.png")

    # 3. Trinucleotide comparison (A4-correlated vs Non-A4 vs A3A)
    tri_compare = {k: group_data[k] for k in ["A4-correlated", "Non-A4-correlated",
                                                "A3A-correlated (no A4)"]
                   if k in group_data}
    if tri_compare:
        plot_trinucleotides(tri_compare, top_n=15,
                           output_path=OUTPUT_DIR / "trinucleotide_comparison.png")

    # 4. Structure type comparison
    if "A4-correlated" in groups_def and "Non-A4-correlated" in groups_def:
        a4_df_plot = df[a4_mask]
        non_df_plot = df[non_a4_mask]
        plot_structure_comparison(a4_df_plot, non_df_plot,
                                  OUTPUT_DIR / "structure_type_comparison.png")

    # 5. Tissue breadth histogram
    fig, ax = plt.subplots(figsize=(8, 4))
    a4_tissues_vals = df[a4_mask]["edited_in_tissues_n"].dropna()
    non_tissues_vals = df[non_a4_mask]["edited_in_tissues_n"].dropna()
    bins = range(0, 60, 3)
    ax.hist(a4_tissues_vals, bins=bins, alpha=0.6, label=f"A4-correlated (n={len(a4_tissues_vals)})",
            density=True, color="#e74c3c")
    ax.hist(non_tissues_vals, bins=bins, alpha=0.6, label=f"Non-A4 (n={len(non_tissues_vals)})",
            density=True, color="#3498db")
    ax.set_xlabel("Number of GTEx tissues with editing")
    ax.set_ylabel("Density")
    ax.set_title("Tissue Breadth: A4-correlated vs Non-A4 Sites")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "tissue_breadth_histogram.png", dpi=150)
    plt.close()

    logger.info("All figures saved to %s", OUTPUT_DIR)

    # -----------------------------------------------------------------------
    # Print summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("APOBEC4 Correlation Analysis — Summary")
    print("=" * 60)
    print(f"\nTotal Levanon sites: {len(df)}")
    print(f"A4-correlated sites: {a4_mask.sum()} ({100*a4_mask.mean():.1f}%)")
    print(f"A4-exclusive sites:  {a4_only_mask.sum()}")
    print()
    print("TC motif fraction:")
    for gname in ["A4-correlated", "A4-exclusive", "Non-A4-correlated",
                  "A3A-correlated (no A4)", "A3G-correlated (no A4)"]:
        if gname in results:
            r = results[gname]
            print(f"  {gname:<30}: {r['tc_fraction']:.3f}  (n={r['n_with_seq']})")
    if "stats_tc_fisher" in results:
        s = results["stats_tc_fisher"]
        print(f"\n  Fisher test A4 vs non-A4: OR={s['A4_vs_non_A4_odds_ratio']:.3f}, "
              f"p={s['A4_vs_non_A4_pvalue']}")
    print()
    print("In-loop fraction:")
    for gname in ["A4-correlated", "Non-A4-correlated", "A3A-correlated (no A4)"]:
        if gname in results:
            r = results[gname]
            print(f"  {gname:<30}: {r['in_loop_fraction']:.3f}")
    print()
    print("Mean tissues edited:")
    for gname in ["A4-correlated", "Non-A4-correlated"]:
        if gname in results:
            r = results[gname]
            print(f"  {gname:<30}: {r['tissue_n_mean']:.1f}")
    if "stats_tissue_breadth" in results:
        s = results["stats_tissue_breadth"]
        print(f"  Mann-Whitney p = {s['mannwhitney_pvalue']}")
    print()
    print("A4 co-occurrence with other enzymes:")
    if "a4_co_occurrence" in results:
        for k, v in results["a4_co_occurrence"].items():
            print(f"  {k}: {v}")
    print()
    print(f"\nResults: {out_json}")
    print(f"Figures: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
