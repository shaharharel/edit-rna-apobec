#!/usr/bin/env python
"""Extended motif analysis for APOBEC3A editing sites.

Analyzes sequence context preferences beyond the known ±2 nt motif:
1. Position-specific nucleotide frequencies (-10 to +10 around edit site)
2. Information content / sequence logo analysis
3. Motif enrichment in high-rate vs low-rate sites
4. Motif comparison: positive vs negative, TP vs FN
5. Dinucleotide and tetranucleotide context analysis

Usage:
    python experiments/apobec/exp_motif_analysis.py
"""

import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logger = logging.getLogger(__name__)

SPLITS_CSV = PROJECT_ROOT / "data" / "processed" / "splits_expanded.csv"
SEQ_JSON = PROJECT_ROOT / "data" / "processed" / "site_sequences.json"
LABELS_CSV = PROJECT_ROOT / "data" / "processed" / "editing_sites_labels.csv"
PRED_CSV = PROJECT_ROOT / "experiments" / "apobec" / "outputs" / "iteration3" / "test_predictions.csv"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "apobec" / "outputs" / "motif_analysis"

DATASET_LABELS = {
    "advisor_c2t": "Levanon",
    "asaoka_2019": "Asaoka",
    "sharma_2015": "Sharma",
    "alqassim_2021": "Alqassim",
    "baysal_2016": "Baysal",
    "tier2_negative": "Tier2 Neg",
    "tier3_negative": "Tier3 Neg",
}

BASES = ["A", "C", "G", "U"]


def load_data():
    """Load sequences and metadata."""
    splits_df = pd.read_csv(SPLITS_CSV)

    with open(SEQ_JSON) as f:
        sequences = json.load(f)
    logger.info("Loaded %d sequences", len(sequences))

    labels_df = None
    if LABELS_CSV.exists():
        labels_df = pd.read_csv(LABELS_CSV)

    return splits_df, sequences, labels_df


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
            pos = i - c_pos + window  # position in matrix
            if 0 <= pos < context_len and nt in BASES:
                counts[pos, BASES.index(nt)] += 1

    # Normalize
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
        ic[i] = 2.0 - entropy  # max entropy for 4 bases = 2 bits
    return ic


def analyze_motif_patterns(seqs, label, window=5):
    """Analyze k-mer patterns around edit site."""
    center = 100  # 201nt sequences, 0-indexed center

    # Position-specific
    di_counter = Counter()  # dinucleotide at -1,0
    tri_counter = Counter()  # trinucleotide -1,0,+1
    tetra_counter = Counter()  # -2,-1,0,+1

    for seq in seqs:
        seq = seq.upper().replace("T", "U")
        if len(seq) <= center + 1:
            continue

        # Dinucleotide: position -1 and 0
        if center >= 1:
            di = seq[center - 1:center + 1]
            if len(di) == 2:
                di_counter[di] += 1

        # Trinucleotide: -1, 0, +1
        if center >= 1 and center + 1 < len(seq):
            tri = seq[center - 1:center + 2]
            if len(tri) == 3:
                tri_counter[tri] += 1

        # Tetranucleotide: -2, -1, 0, +1
        if center >= 2 and center + 1 < len(seq):
            tetra = seq[center - 2:center + 2]
            if len(tetra) == 4:
                tetra_counter[tetra] += 1

    return {
        "label": label,
        "n_seqs": len(seqs),
        "dinucleotides": dict(di_counter.most_common(16)),
        "trinucleotides": dict(tri_counter.most_common(20)),
        "tetranucleotides": dict(tetra_counter.most_common(20)),
    }


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    splits_df, sequences, labels_df = load_data()
    all_results = {}

    # ===================================================================
    # Part 1: Position-specific nucleotide frequencies
    # ===================================================================
    logger.info("=" * 70)
    logger.info("PART 1: Position-Specific Nucleotide Frequencies")
    logger.info("=" * 70)

    window = 10

    # Group sequences by dataset and label
    groups = {
        "positive_all": [],
        "negative_all": [],
    }
    per_dataset = {}

    for _, row in splits_df.iterrows():
        sid = row["site_id"]
        if sid not in sequences:
            continue
        ds = row["dataset_source"]
        label = row["label"]

        if label == 1:
            groups["positive_all"].append(sequences[sid])
        else:
            groups["negative_all"].append(sequences[sid])

        ds_label = DATASET_LABELS.get(ds, ds)
        if ds_label not in per_dataset:
            per_dataset[ds_label] = []
        per_dataset[ds_label].append(sequences[sid])

    # Compute frequencies
    freq_results = {}
    for group_name, seqs in groups.items():
        if not seqs:
            continue
        freqs, counts = compute_position_frequencies(seqs, window=window)
        ic = compute_information_content(freqs)
        freq_results[group_name] = {
            "n_seqs": len(seqs),
            "frequencies": freqs.tolist(),
            "information_content": ic.tolist(),
        }
        logger.info("%s: %d sequences", group_name, len(seqs))

    all_results["position_frequencies"] = freq_results

    # Print position-specific summary
    print("\n--- Position-Specific Nucleotide Frequencies (Positives) ---")
    pos_freqs = np.array(freq_results["positive_all"]["frequencies"])
    pos_ic = freq_results["positive_all"]["information_content"]
    print(f"{'Pos':>4s} {'A':>6s} {'C':>6s} {'G':>6s} {'U':>6s} {'IC(bits)':>8s} {'Dominant':>8s}")
    print("-" * 50)
    for i in range(len(pos_freqs)):
        pos = i - window
        dominant = BASES[np.argmax(pos_freqs[i])]
        print(f"{pos:>+4d} {pos_freqs[i,0]:>6.3f} {pos_freqs[i,1]:>6.3f} "
              f"{pos_freqs[i,2]:>6.3f} {pos_freqs[i,3]:>6.3f} "
              f"{pos_ic[i]:>8.3f} {dominant:>8s}")

    # Print negative comparison for key positions
    print("\n--- Position-Specific: Positive vs Negative (key positions) ---")
    neg_freqs = np.array(freq_results["negative_all"]["frequencies"])
    neg_ic = freq_results["negative_all"]["information_content"]
    print(f"{'Pos':>4s} {'Pos_A':>6s} {'Pos_C':>6s} {'Pos_G':>6s} {'Pos_U':>6s} | "
          f"{'Neg_A':>6s} {'Neg_C':>6s} {'Neg_G':>6s} {'Neg_U':>6s} | {'Diff_IC':>8s}")
    print("-" * 80)
    for i in [window - 3, window - 2, window - 1, window, window + 1, window + 2, window + 3]:
        pos = i - window
        diff_ic = pos_ic[i] - neg_ic[i]
        print(f"{pos:>+4d} {pos_freqs[i,0]:>6.3f} {pos_freqs[i,1]:>6.3f} "
              f"{pos_freqs[i,2]:>6.3f} {pos_freqs[i,3]:>6.3f} | "
              f"{neg_freqs[i,0]:>6.3f} {neg_freqs[i,1]:>6.3f} "
              f"{neg_freqs[i,2]:>6.3f} {neg_freqs[i,3]:>6.3f} | "
              f"{diff_ic:>+8.3f}")

    # ===================================================================
    # Part 2: Per-dataset motif patterns
    # ===================================================================
    logger.info("\n" + "=" * 70)
    logger.info("PART 2: Per-Dataset Motif Patterns")
    logger.info("=" * 70)

    dataset_motifs = {}
    for ds_label, seqs in per_dataset.items():
        result = analyze_motif_patterns(seqs, ds_label)
        dataset_motifs[ds_label] = result

    all_results["dataset_motifs"] = dataset_motifs

    # Print per-dataset dinucleotide at position -1,0
    print("\n--- Dinucleotide at Position -1,0 (per dataset) ---")
    for ds_label in ["Levanon", "Asaoka", "Sharma", "Alqassim", "Tier2 Neg", "Tier3 Neg"]:
        if ds_label not in dataset_motifs:
            continue
        info = dataset_motifs[ds_label]
        top5 = list(info["dinucleotides"].items())[:5]
        total = sum(info["dinucleotides"].values())
        top_str = ", ".join(f"{k}:{v} ({100*v/total:.0f}%)" for k, v in top5)
        print(f"  {ds_label:12s} (n={info['n_seqs']:>5d}): {top_str}")

    # Print trinucleotide patterns
    print("\n--- Top 5 Trinucleotides (-1,0,+1) per Dataset ---")
    for ds_label in ["Levanon", "Asaoka", "Sharma", "Alqassim"]:
        if ds_label not in dataset_motifs:
            continue
        info = dataset_motifs[ds_label]
        top5 = list(info["trinucleotides"].items())[:5]
        total = sum(info["trinucleotides"].values())
        top_str = ", ".join(f"{k}:{100*v/total:.1f}%" for k, v in top5)
        print(f"  {ds_label:12s}: {top_str}")

    # ===================================================================
    # Part 3: High-rate vs low-rate motif comparison (Levanon only)
    # ===================================================================
    logger.info("\n" + "=" * 70)
    logger.info("PART 3: High-Rate vs Low-Rate Motif Comparison (Levanon)")
    logger.info("=" * 70)

    levanon_df = splits_df[splits_df["dataset_source"] == "advisor_c2t"].copy()
    levanon_df["editing_rate"] = pd.to_numeric(levanon_df["editing_rate"], errors="coerce")
    levanon_with_rate = levanon_df.dropna(subset=["editing_rate"])

    if len(levanon_with_rate) > 0:
        median_rate = levanon_with_rate["editing_rate"].median()
        high_rate = levanon_with_rate[levanon_with_rate["editing_rate"] > median_rate]
        low_rate = levanon_with_rate[levanon_with_rate["editing_rate"] <= median_rate]

        high_seqs = [sequences[sid] for sid in high_rate["site_id"] if sid in sequences]
        low_seqs = [sequences[sid] for sid in low_rate["site_id"] if sid in sequences]

        high_motifs = analyze_motif_patterns(high_seqs, "High-rate")
        low_motifs = analyze_motif_patterns(low_seqs, "Low-rate")

        all_results["rate_motifs"] = {
            "median_rate": float(median_rate),
            "high_rate": high_motifs,
            "low_rate": low_motifs,
        }

        # Position frequencies
        high_freqs, _ = compute_position_frequencies(high_seqs, window=window)
        low_freqs, _ = compute_position_frequencies(low_seqs, window=window)

        print(f"\n--- High-Rate (>{median_rate:.1f}%) vs Low-Rate Levanon Sites ---")
        print(f"High-rate: {len(high_seqs)} sites, Low-rate: {len(low_seqs)} sites")

        print(f"\n{'Pos':>4s} {'Hi_A':>6s} {'Hi_C':>6s} {'Hi_G':>6s} {'Hi_U':>6s} | "
              f"{'Lo_A':>6s} {'Lo_C':>6s} {'Lo_G':>6s} {'Lo_U':>6s}")
        print("-" * 60)
        for i in [window - 3, window - 2, window - 1, window, window + 1, window + 2, window + 3]:
            pos = i - window
            print(f"{pos:>+4d} {high_freqs[i,0]:>6.3f} {high_freqs[i,1]:>6.3f} "
                  f"{high_freqs[i,2]:>6.3f} {high_freqs[i,3]:>6.3f} | "
                  f"{low_freqs[i,0]:>6.3f} {low_freqs[i,1]:>6.3f} "
                  f"{low_freqs[i,2]:>6.3f} {low_freqs[i,3]:>6.3f}")

        # Trinucleotide comparison
        print("\nHigh-rate top trinucleotides:")
        for k, v in list(high_motifs["trinucleotides"].items())[:8]:
            pct = 100 * v / max(high_motifs["n_seqs"], 1)
            print(f"  {k}: {v} ({pct:.1f}%)")
        print("Low-rate top trinucleotides:")
        for k, v in list(low_motifs["trinucleotides"].items())[:8]:
            pct = 100 * v / max(low_motifs["n_seqs"], 1)
            print(f"  {k}: {v} ({pct:.1f}%)")

    # ===================================================================
    # Part 4: TP vs FN motif comparison
    # ===================================================================
    logger.info("\n" + "=" * 70)
    logger.info("PART 4: TP vs FN Motif Comparison")
    logger.info("=" * 70)

    if PRED_CSV.exists():
        pred_df = pd.read_csv(PRED_CSV)

        # Classify predictions
        if "y_true" in pred_df.columns and "y_pred" in pred_df.columns:
            tp_ids = pred_df[(pred_df["y_true"] == 1) & (pred_df["y_pred"] == 1)]["site_id"].tolist()
            fn_ids = pred_df[(pred_df["y_true"] == 1) & (pred_df["y_pred"] == 0)]["site_id"].tolist()
            fp_ids = pred_df[(pred_df["y_true"] == 0) & (pred_df["y_pred"] == 1)]["site_id"].tolist()

            tp_seqs = [sequences[sid] for sid in tp_ids if sid in sequences]
            fn_seqs = [sequences[sid] for sid in fn_ids if sid in sequences]
            fp_seqs = [sequences[sid] for sid in fp_ids if sid in sequences]

            tp_motifs = analyze_motif_patterns(tp_seqs, "TP")
            fn_motifs = analyze_motif_patterns(fn_seqs, "FN")

            all_results["prediction_motifs"] = {
                "tp": tp_motifs,
                "fn": fn_motifs,
            }
            if fp_seqs:
                fp_motifs = analyze_motif_patterns(fp_seqs, "FP")
                all_results["prediction_motifs"]["fp"] = fp_motifs

            print(f"\n--- TP ({len(tp_seqs)}) vs FN ({len(fn_seqs)}) Motif Comparison ---")

            # Dinucleotide at -1,0
            tp_di = tp_motifs["dinucleotides"]
            fn_di = fn_motifs["dinucleotides"]
            tp_total = sum(tp_di.values())
            fn_total = sum(fn_di.values())

            print("\nDinucleotide at -1,0:")
            all_di = set(list(tp_di.keys())[:8] + list(fn_di.keys())[:8])
            print(f"  {'Di':>4s} {'TP%':>8s} {'FN%':>8s} {'Diff':>8s}")
            for di in sorted(all_di, key=lambda x: -(tp_di.get(x, 0) / max(tp_total, 1))):
                tp_pct = 100 * tp_di.get(di, 0) / max(tp_total, 1)
                fn_pct = 100 * fn_di.get(di, 0) / max(fn_total, 1)
                print(f"  {di:>4s} {tp_pct:>7.1f}% {fn_pct:>7.1f}% {fn_pct-tp_pct:>+7.1f}%")

            # Position frequencies
            tp_freq, _ = compute_position_frequencies(tp_seqs, window=5)
            fn_freq, _ = compute_position_frequencies(fn_seqs, window=5)

            print(f"\n{'Pos':>4s} {'TP_A':>6s} {'TP_C':>6s} {'TP_G':>6s} {'TP_U':>6s} | "
                  f"{'FN_A':>6s} {'FN_C':>6s} {'FN_G':>6s} {'FN_U':>6s}")
            print("-" * 60)
            for i in range(11):
                pos = i - 5
                print(f"{pos:>+4d} {tp_freq[i,0]:>6.3f} {tp_freq[i,1]:>6.3f} "
                      f"{tp_freq[i,2]:>6.3f} {tp_freq[i,3]:>6.3f} | "
                      f"{fn_freq[i,0]:>6.3f} {fn_freq[i,1]:>6.3f} "
                      f"{fn_freq[i,2]:>6.3f} {fn_freq[i,3]:>6.3f}")

    else:
        logger.info("Prediction CSV not found, skipping TP/FN comparison")

    # ===================================================================
    # Part 5: Jalili 2023 motif validation
    # ===================================================================
    logger.info("\n" + "=" * 70)
    logger.info("PART 5: Jalili 2023 Motif Validation")
    logger.info("=" * 70)

    # Check for CAUC, CUUC, UAUC, UUUC loop motifs (Jalili optimal substrates)
    jalili_motifs = ["CAUC", "CUUC", "UAUC", "UUUC"]
    jalili_results = {}

    for ds_label in ["Levanon", "Asaoka", "Sharma", "Alqassim"]:
        ds_key = {v: k for k, v in DATASET_LABELS.items()}.get(ds_label)
        if not ds_key:
            continue
        ds_df = splits_df[(splits_df["dataset_source"] == ds_key) & (splits_df["label"] == 1)]
        ds_seqs = [sequences[sid] for sid in ds_df["site_id"] if sid in sequences]

        motif_counts = {m: 0 for m in jalili_motifs}
        for seq in ds_seqs:
            seq = seq.upper().replace("T", "U")
            center = len(seq) // 2
            # Check if the tetranucleotide at -2,-1,0,+1 matches
            if center >= 2 and center + 1 < len(seq):
                tetra = seq[center - 2:center + 2]
                for m in jalili_motifs:
                    if tetra == m:
                        motif_counts[m] += 1

        total = len(ds_seqs)
        any_jalili = sum(motif_counts.values())
        jalili_results[ds_label] = {
            "n_sites": total,
            "any_jalili_motif": any_jalili,
            "fraction": any_jalili / max(total, 1),
            "per_motif": {m: c for m, c in motif_counts.items()},
        }

    all_results["jalili_validation"] = jalili_results

    print("\n--- Jalili 2023 Optimal Loop Motif Prevalence ---")
    print(f"{'Dataset':>12s} {'n':>6s} {'CAUC':>6s} {'CUUC':>6s} {'UAUC':>6s} {'UUUC':>6s} {'Total':>6s} {'%':>6s}")
    print("-" * 60)
    for ds, info in jalili_results.items():
        pm = info["per_motif"]
        print(f"{ds:>12s} {info['n_sites']:>6d} {pm['CAUC']:>6d} {pm['CUUC']:>6d} "
              f"{pm['UAUC']:>6d} {pm['UUUC']:>6d} {info['any_jalili_motif']:>6d} "
              f"{100*info['fraction']:>5.1f}%")

    print("=" * 80)

    # Save all results
    def serialize(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    with open(OUTPUT_DIR / "motif_analysis_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=serialize)

    logger.info("\nResults saved to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
