"""A3B Feature Challenge v3: Close the final gap.

v2 showed: best hand = 0.781 (ALL seq), target = 0.810. Gap = 0.029.
Key insight: k-mer spectrum dominates. Try:
1. Wider k-mer windows (±50, ±80, full 201nt)
2. 5-mer spectrum
3. Multiple k-mer scales combined
4. Better XGBoost tuning
5. Position-weighted k-mers (closer to center = more weight)
"""

import functools
import json
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

print = functools.partial(print, flush=True)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.apobec_feature_extraction import extract_motif_from_seq, CENTER

OUT_DIR = ROOT / "experiments" / "multi_enzyme" / "outputs" / "a3b_feature_challenge"


def load_data():
    df = pd.read_csv(ROOT / "data/processed/multi_enzyme/splits_multi_enzyme_v3_with_negatives.csv")
    a3b = df[df["enzyme"] == "A3B"].copy().reset_index(drop=True)
    with open(ROOT / "data/processed/multi_enzyme/multi_enzyme_sequences_v3_with_negatives.json") as f:
        all_seqs = json.load(f)
    emb = torch.load(ROOT / "data/processed/multi_enzyme/embeddings/rnafm_pooled_v3.pt", weights_only=False)
    site_ids = a3b["site_id"].astype(str).tolist()
    labels = a3b["is_edited"].values
    seqs = {str(sid): all_seqs.get(str(sid), "N" * 201) for sid in site_ids}
    return site_ids, labels, seqs, emb


def kmer_spectrum(site_ids, seqs, k, window, weight_by_distance=False):
    bases = "ACGU"
    kmers = [''.join(p) for p in product(bases, repeat=k)]
    kmer_idx = {km: i for i, km in enumerate(kmers)}
    dim = len(kmers)
    feats = np.zeros((len(site_ids), dim), dtype=np.float32)
    for i, sid in enumerate(site_ids):
        seq = seqs[sid].upper().replace("T", "U")
        s, e = max(0, CENTER - window), min(len(seq), CENTER + window + 1)
        region = seq[s:e]
        center_in_region = CENTER - s
        total_weight = 0.0
        for j in range(len(region) - k + 1):
            km = region[j:j+k]
            if km in kmer_idx:
                if weight_by_distance:
                    dist = abs(j + k//2 - center_in_region)
                    w = 1.0 / (1.0 + dist / 10.0)
                else:
                    w = 1.0
                feats[i, kmer_idx[km]] += w
                total_weight += w
        if total_weight > 0:
            feats[i] /= total_weight
    return feats


def extended_motif(site_ids, seqs, window):
    NUC_MAP = {"A": 0, "C": 1, "G": 2, "U": 3}
    dim = 4 * (2 * window)
    feats = np.zeros((len(site_ids), dim), dtype=np.float32)
    for i, sid in enumerate(site_ids):
        seq = seqs[sid].upper().replace("T", "U")
        idx = 0
        for offset in range(-window, window + 1):
            if offset == 0:
                continue
            pos = CENTER + offset
            if 0 <= pos < len(seq) and seq[pos] in NUC_MAP:
                feats[i, idx + NUC_MAP[seq[pos]]] = 1.0
            idx += 4
    return feats


def dinuc_freq(site_ids, seqs, window):
    dinucs = [a+b for a in "ACGU" for b in "ACGU"]
    feats = np.zeros((len(site_ids), 16), dtype=np.float32)
    for i, sid in enumerate(site_ids):
        seq = seqs[sid].upper().replace("T", "U")
        s, e = max(0, CENTER-window), min(len(seq), CENTER+window+1)
        region = seq[s:e]
        total = max(len(region)-1, 1)
        for k in range(len(region)-1):
            dn = region[k:k+2]
            if dn in dinucs:
                feats[i, dinucs.index(dn)] += 1.0/total
    return feats


def trinuc_freq(site_ids, seqs, window):
    trinucs = [a+b+c for a in "ACGU" for b in "ACGU" for c in "ACGU"]
    feats = np.zeros((len(site_ids), 64), dtype=np.float32)
    for i, sid in enumerate(site_ids):
        seq = seqs[sid].upper().replace("T", "U")
        s, e = max(0, CENTER-window), min(len(seq), CENTER+window+1)
        region = seq[s:e]
        total = max(len(region)-2, 1)
        for k in range(len(region)-2):
            tn = region[k:k+3]
            if tn in trinucs:
                feats[i, trinucs.index(tn)] += 1.0/total
    return feats


def gc_content_multi(site_ids, seqs):
    windows = [3, 5, 10, 15, 20, 30, 50, 80, 100]
    feats = np.zeros((len(site_ids), len(windows)), dtype=np.float32)
    for i, sid in enumerate(site_ids):
        seq = seqs[sid].upper().replace("T", "U")
        for j, w in enumerate(windows):
            s, e = max(0, CENTER-w), min(len(seq), CENTER+w+1)
            region = seq[s:e]
            feats[i, j] = sum(1 for c in region if c in "GC") / max(len(region), 1)
    return feats


def nuc_content_multi_window(site_ids, seqs):
    """Nucleotide fractions at multiple windows. 4 nucs * 6 windows = 24 dim."""
    windows = [5, 10, 20, 30, 50, 100]
    feats = np.zeros((len(site_ids), 4 * len(windows)), dtype=np.float32)
    for i, sid in enumerate(site_ids):
        seq = seqs[sid].upper().replace("T", "U")
        for j, w in enumerate(windows):
            s, e = max(0, CENTER-w), min(len(seq), CENTER+w+1)
            region = seq[s:e]
            rlen = max(len(region), 1)
            for bi, b in enumerate("ACGU"):
                feats[i, j*4+bi] = sum(1 for c in region if c == b) / rlen
    return feats


def rnafm_pooled(site_ids, emb):
    feats = np.zeros((len(site_ids), 640), dtype=np.float32)
    for i, sid in enumerate(site_ids):
        if sid in emb:
            feats[i] = emb[sid].numpy()
    return feats


def evaluate(X, labels, name, n_folds=5, seed=42, n_est=500, max_d=6, lr=0.1):
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    aurocs = []
    for fold, (tr, va) in enumerate(skf.split(X, labels)):
        model = XGBClassifier(
            n_estimators=n_est, max_depth=max_d, learning_rate=lr,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            eval_metric="logloss", random_state=seed+fold, n_jobs=-1, verbosity=0,
        )
        model.fit(X[tr], labels[tr], eval_set=[(X[va], labels[va])], verbose=False)
        aurocs.append(roc_auc_score(labels[va], model.predict_proba(X[va])[:, 1]))
    m, s = np.mean(aurocs), np.std(aurocs)
    print(f"  {name:65s} dim={X.shape[1]:4d}  AUROC={m:.4f} +/- {s:.4f}")
    return m, s, aurocs


def main():
    print("Loading data...")
    site_ids, labels, seqs, emb = load_data()
    print(f"A3B: {len(site_ids)} sites ({sum(labels)} pos, {len(labels)-sum(labels)} neg)")

    results = []

    def run(name, X, **eval_kw):
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        m, s, folds = evaluate(X, labels, name, **eval_kw)
        results.append({"name": name, "dim": X.shape[1], "auroc_mean": m, "auroc_std": s})
        return X

    def combine(name, arrays, **eval_kw):
        return run(name, np.concatenate(arrays, axis=1), **eval_kw)

    # ── k-mer spectrum at different scales ──
    print("\n=== K-MER SPECTRUM EXPLORATION ===")
    X_4m_30 = run("4-mer ±30", kmer_spectrum(site_ids, seqs, 4, 30))
    X_4m_50 = run("4-mer ±50", kmer_spectrum(site_ids, seqs, 4, 50))
    X_4m_80 = run("4-mer ±80", kmer_spectrum(site_ids, seqs, 4, 80))
    X_4m_100 = run("4-mer ±100 (full)", kmer_spectrum(site_ids, seqs, 4, 100))

    X_3m_30 = run("3-mer ±30", kmer_spectrum(site_ids, seqs, 3, 30))
    X_3m_50 = run("3-mer ±50", kmer_spectrum(site_ids, seqs, 3, 50))
    X_3m_100 = run("3-mer ±100 (full)", kmer_spectrum(site_ids, seqs, 3, 100))

    print("\n=== 5-MER (1024-dim) ===")
    X_5m_30 = run("5-mer ±30", kmer_spectrum(site_ids, seqs, 5, 30))
    X_5m_50 = run("5-mer ±50", kmer_spectrum(site_ids, seqs, 5, 50))
    X_5m_100 = run("5-mer ±100 (full)", kmer_spectrum(site_ids, seqs, 5, 100))

    print("\n=== DISTANCE-WEIGHTED K-MERS ===")
    X_4mw_30 = run("4-mer ±30 weighted", kmer_spectrum(site_ids, seqs, 4, 30, weight_by_distance=True))
    X_4mw_50 = run("4-mer ±50 weighted", kmer_spectrum(site_ids, seqs, 4, 50, weight_by_distance=True))
    X_4mw_100 = run("4-mer ±100 weighted", kmer_spectrum(site_ids, seqs, 4, 100, weight_by_distance=True))

    # ── Multi-scale k-mer combinations ──
    print("\n=== MULTI-SCALE COMBINATIONS ===")
    X_dinuc30 = dinuc_freq(site_ids, seqs, 30)
    X_trinuc30 = trinuc_freq(site_ids, seqs, 30)
    X_ext20 = extended_motif(site_ids, seqs, 20)
    X_gc = gc_content_multi(site_ids, seqs)
    X_nuc = nuc_content_multi_window(site_ids, seqs)

    combine("3-mer + 4-mer + 5-mer ±50", [X_3m_50, X_4m_50, X_5m_50])
    combine("3-mer + 4-mer ±100", [X_3m_100, X_4m_100])
    combine("4-mer ±30 + 4-mer ±100 (multi-scale)", [X_4m_30, X_4m_100])
    X_mega_seq = combine("ExtMotif±20 + dinuc + trinuc + 4mer±50 + 5mer±50 + GC + nuc",
        [X_ext20, X_dinuc30, X_trinuc30, X_4m_50, X_5m_50, X_gc, X_nuc])

    # ── Best sequence features with better XGBoost hyperparams ──
    print("\n=== HYPERPARAMETER TUNING ===")
    run("4-mer ±50 (depth=8)", X_4m_50, max_d=8)
    run("4-mer ±50 (n_est=1000)", X_4m_50, n_est=1000)
    run("4-mer ±50 (depth=8, n_est=1000)", X_4m_50, max_d=8, n_est=1000)
    run("4-mer ±50 (depth=10, n_est=1000, lr=0.05)", X_4m_50, max_d=10, n_est=1000, lr=0.05)
    run("5-mer ±50 (depth=8, n_est=1000)", X_5m_50, max_d=8, n_est=1000)

    X_best_seq = combine("MEGA seq (depth=8, n_est=1000)",
        [X_ext20, X_dinuc30, X_trinuc30, X_4m_50, X_5m_50, X_gc, X_nuc],
        max_d=8, n_est=1000)

    # ── RNA-FM comparisons ──
    print("\n=== RNA-FM BASELINES ===")
    X_rnafm = run("RNA-FM 640", rnafm_pooled(site_ids, emb))
    combine("RNA-FM + 4-mer ±50", [X_rnafm, X_4m_50])
    combine("RNA-FM + best_seq", [X_rnafm, X_mega_seq])
    combine("RNA-FM + best_seq (depth=8, n=1000)", [X_rnafm, X_mega_seq], max_d=8, n_est=1000)

    # ── Try upstream/downstream separate k-mers ──
    print("\n=== UPSTREAM/DOWNSTREAM SEPARATE K-MERS ===")
    # Compute 4-mers separately for upstream and downstream
    def kmer_up_down(site_ids, seqs, k, window):
        bases = "ACGU"
        kmers = [''.join(p) for p in product(bases, repeat=k)]
        kmer_idx = {km: i for i, km in enumerate(kmers)}
        dim = 2 * len(kmers)  # up + down
        feats = np.zeros((len(site_ids), dim), dtype=np.float32)
        for i, sid in enumerate(site_ids):
            seq = seqs[sid].upper().replace("T", "U")
            # Upstream
            s_up, e_up = max(0, CENTER-window), CENTER
            region_up = seq[s_up:e_up]
            total_up = max(len(region_up)-k+1, 1)
            for j in range(len(region_up)-k+1):
                km = region_up[j:j+k]
                if km in kmer_idx:
                    feats[i, kmer_idx[km]] += 1.0/total_up
            # Downstream
            s_dn, e_dn = CENTER+1, min(len(seq), CENTER+window+1)
            region_dn = seq[s_dn:e_dn]
            total_dn = max(len(region_dn)-k+1, 1)
            for j in range(len(region_dn)-k+1):
                km = region_dn[j:j+k]
                if km in kmer_idx:
                    feats[i, len(kmers)+kmer_idx[km]] += 1.0/total_dn
        return feats

    X_4m_updn_50 = run("4-mer up/down ±50 (512-dim)", kmer_up_down(site_ids, seqs, 4, 50))
    X_3m_updn_50 = run("3-mer up/down ±50 (128-dim)", kmer_up_down(site_ids, seqs, 3, 50))
    combine("3-mer + 4-mer up/down ±50", [X_3m_updn_50, X_4m_updn_50])
    combine("3-mer + 4-mer up/down ±50 (depth=8, n=1000)",
            [X_3m_updn_50, X_4m_updn_50], max_d=8, n_est=1000)

    # ── Multi-window position-specific features ──
    print("\n=== POSITIONAL BINNED K-MERS ===")
    def kmer_binned(site_ids, seqs, k, bins):
        """K-mer spectrum in multiple overlapping bins around center."""
        bases = "ACGU"
        kmers = [''.join(p) for p in product(bases, repeat=k)]
        kmer_idx = {km: i for i, km in enumerate(kmers)}
        nk = len(kmers)
        dim = nk * len(bins)
        feats = np.zeros((len(site_ids), dim), dtype=np.float32)
        for i, sid in enumerate(site_ids):
            seq = seqs[sid].upper().replace("T", "U")
            for b_idx, (b_start, b_end) in enumerate(bins):
                s, e = max(0, CENTER+b_start), min(len(seq), CENTER+b_end+1)
                region = seq[s:e]
                total = max(len(region)-k+1, 1)
                for j in range(len(region)-k+1):
                    km = region[j:j+k]
                    if km in kmer_idx:
                        feats[i, b_idx*nk + kmer_idx[km]] += 1.0/total
        return feats

    # 3-mer in 5 bins: [-50,-20], [-20,-5], [-5,5], [5,20], [20,50]
    bins5 = [(-50,-20), (-20,-5), (-5,5), (5,20), (20,50)]
    X_3m_binned = run("3-mer 5-bin (320-dim)", kmer_binned(site_ids, seqs, 3, bins5))

    # 4-mer in 3 bins: [-50,-10], [-10,10], [10,50]
    bins3 = [(-50,-10), (-10,10), (10,50)]
    X_4m_binned = run("4-mer 3-bin (768-dim)", kmer_binned(site_ids, seqs, 4, bins3))

    combine("3-mer 5-bin + 4-mer 3-bin", [X_3m_binned, X_4m_binned])
    combine("3-mer 5-bin + 4-mer 3-bin (depth=8, n=1000)",
            [X_3m_binned, X_4m_binned], max_d=8, n_est=1000)

    # ── Summary ──
    print("\n" + "="*110)
    print("SUMMARY (sorted by AUROC)")
    print("="*110)
    results_df = pd.DataFrame(results).sort_values("auroc_mean", ascending=False)
    for _, r in results_df.iterrows():
        tag = ""
        if "RNA-FM" not in r["name"]:
            tag = " <-- HAND" if r["auroc_mean"] >= 0.800 else ""
        print(f"  {r['name']:65s} dim={r['dim']:4d}  AUROC={r['auroc_mean']:.4f} +/- {r['auroc_std']:.4f}{tag}")

    results_df.to_csv(OUT_DIR / "feature_challenge_v3_results.csv", index=False)

    non_rnafm = results_df[~results_df["name"].str.contains("RNA-FM")]
    best = non_rnafm.iloc[0]
    print(f"\nBest hand: {best['name']} -> AUROC={best['auroc_mean']:.4f}")
    print(f"Target: 0.810")
    print(f"Gap: {0.810 - best['auroc_mean']:.4f}")


if __name__ == "__main__":
    main()
