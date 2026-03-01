"""
RNAsee_RF: Random Forest with 50-bit binary nucleotide encoding on A3A tiered negatives.

Encoding: 15nt upstream + 10nt downstream of center (position 100, 0-indexed).
2 bits per nucleotide: (is_purine, pairs_GC)
  A = (1, 0)
  G = (1, 1)
  C = (0, 1)
  U/T = (0, 0)
Total: 25 nucleotides * 2 bits = 50 features.

5-fold KFold CV with RandomForestClassifier.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score

# --- Paths ---
DATA_DIR = Path("/Users/shaharharel/Documents/github/edit-rna-apobec/data/processed")
OUTPUT_DIR = Path("/Users/shaharharel/Documents/github/edit-rna-apobec/experiments/apobec3a/outputs/classification_a3a_5fold")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS_CSV = DATA_DIR / "splits_expanded_a3a.csv"
SEQUENCES_JSON = DATA_DIR / "site_sequences.json"

# --- RNAsee 50-bit encoding ---
ENCODING = {
    'A': (1, 0),
    'G': (1, 1),
    'C': (0, 1),
    'U': (0, 0),
    'T': (0, 0),
}

CENTER = 100  # 0-indexed center of 201nt window
UPSTREAM = 15
DOWNSTREAM = 10
# Window: positions [100-15, 100+10] = [85, 110], 25nt total + center excluded?
# RNAsee uses 15nt upstream + 10nt downstream of center = 25 positions (not including center itself?
# Actually the standard RNAsee encoding is 25nt around the edit site. Let's use:
# positions 85..99 (15nt upstream) + 101..110 (10nt downstream) = 25nt, 50 bits
# The center (position 100) is always C (the edited cytidine), so it's excluded.

def encode_sequence(seq):
    """Encode 25 nucleotides around center into 50-bit binary vector."""
    upstream_start = CENTER - UPSTREAM  # 85
    upstream_end = CENTER               # 100 (exclusive)
    downstream_start = CENTER + 1       # 101
    downstream_end = CENTER + 1 + DOWNSTREAM  # 111 (exclusive)

    window = seq[upstream_start:upstream_end] + seq[downstream_start:downstream_end]
    assert len(window) == UPSTREAM + DOWNSTREAM, f"Expected {UPSTREAM + DOWNSTREAM}nt, got {len(window)}"

    bits = []
    for nt in window:
        nt_upper = nt.upper()
        if nt_upper in ENCODING:
            bits.extend(ENCODING[nt_upper])
        else:
            # Unknown nucleotide (N, etc.) -> (0, 0)
            bits.extend((0, 0))
    return bits


def main():
    # Load data
    print("Loading data...")
    df = pd.read_csv(SPLITS_CSV)
    with open(SEQUENCES_JSON, "r") as f:
        sequences = json.load(f)

    print(f"Total sites in CSV: {len(df)}")
    print(f"Total sequences: {len(sequences)}")
    print(f"Label distribution:\n{df['is_edited'].value_counts().to_string()}")

    # Filter to sites with sequences
    df = df[df["site_id"].isin(sequences)].copy()
    print(f"Sites with sequences: {len(df)}")

    # Encode features
    print("Encoding features (50-bit binary)...")
    X_list = []
    valid_idx = []
    for i, row in df.iterrows():
        seq = sequences[row["site_id"]]
        if len(seq) < CENTER + 1 + DOWNSTREAM:
            print(f"  Skipping {row['site_id']}: sequence too short ({len(seq)})")
            continue
        bits = encode_sequence(seq)
        X_list.append(bits)
        valid_idx.append(i)

    df = df.loc[valid_idx].reset_index(drop=True)
    X = np.array(X_list, dtype=np.float32)
    y = df["is_edited"].values.astype(int)

    print(f"Final dataset: {X.shape[0]} sites, {X.shape[1]} features")
    print(f"  Positives: {(y == 1).sum()}, Negatives: {(y == 0).sum()}")

    # 5-fold CV
    print("\nRunning 5-fold CV...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    fold_results = []
    for fold_i, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = RandomForestClassifier(
            n_estimators=500,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        )
        clf.fit(X_train, y_train)

        y_prob = clf.predict_proba(X_test)[:, 1]
        y_pred = clf.predict(X_test)

        auroc = roc_auc_score(y_test, y_prob)
        auprc = average_precision_score(y_test, y_prob)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        fold_results.append({
            "fold": fold_i + 1,
            "auroc": auroc,
            "auprc": auprc,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "n_test_pos": int(y_test.sum()),
            "n_test_neg": int((y_test == 0).sum()),
        })

        print(f"  Fold {fold_i+1}: AUROC={auroc:.4f}  AUPRC={auprc:.4f}  "
              f"F1={f1:.4f}  Prec={precision:.4f}  Rec={recall:.4f}")

    # Aggregate
    metrics = ["auroc", "auprc", "f1", "precision", "recall"]
    summary = {}
    print("\n" + "=" * 60)
    print("RNAsee_RF Results (5-fold CV, A3A tiered negatives)")
    print("=" * 60)
    for m in metrics:
        vals = [r[m] for r in fold_results]
        mean_val = np.mean(vals)
        std_val = np.std(vals)
        summary[m] = {"mean": float(mean_val), "std": float(std_val)}
        print(f"  {m.upper():>12s}: {mean_val:.4f} +/- {std_val:.4f}")
    print("=" * 60)

    # Save results
    results = {
        "model": "RNAsee_RF",
        "description": "Random Forest with 50-bit binary nucleotide encoding (RNAsee style)",
        "encoding": "15nt upstream + 10nt downstream, 2 bits/nt (is_purine, pairs_GC), center excluded",
        "n_features": 50,
        "n_sites": int(len(df)),
        "n_positives": int((y == 1).sum()),
        "n_negatives": int((y == 0).sum()),
        "cv_folds": 5,
        "classifier": "RandomForestClassifier(n_estimators=500, max_depth=15, random_state=42)",
        "summary": summary,
        "fold_results": fold_results,
    }

    out_path = OUTPUT_DIR / "rnasee_rf_tiered_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
