"""A3B Feature Engineering Challenge: Close the gap between GB (0.596) and EditRNA+Hand (0.810).

Iterates through many feature combinations to find what RNA-FM captures that hand features miss.
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

# Project root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.apobec_feature_extraction import (
    extract_motif_from_seq, CENTER,
    LOOP_FEATURE_COLS,
)

OUT_DIR = ROOT / "experiments" / "multi_enzyme" / "outputs" / "a3b_feature_challenge"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Data Loading ──────────────────────────────────────────────────────────

def load_data():
    """Load A3B data, sequences, structure cache, loop positions, embeddings."""
    df = pd.read_csv(ROOT / "data/processed/multi_enzyme/splits_multi_enzyme_v3_with_negatives.csv")
    a3b = df[df["enzyme"] == "A3B"].copy().reset_index(drop=True)

    with open(ROOT / "data/processed/multi_enzyme/multi_enzyme_sequences_v3_with_negatives.json") as f:
        all_seqs = json.load(f)

    sc = np.load(ROOT / "data/processed/multi_enzyme/structure_cache_multi_enzyme_v3.npz", allow_pickle=True)
    sc_ids = list(sc["site_ids"])
    sc_delta = sc["delta_features"]
    sc_mfes = sc["mfes"]
    sc_mfes_ed = sc["mfes_edited"]
    struct_delta = {str(sid): sc_delta[i] for i, sid in enumerate(sc_ids)}
    mfe_dict = {str(sid): float(sc_mfes[i]) for i, sid in enumerate(sc_ids)}
    mfe_ed_dict = {str(sid): float(sc_mfes_ed[i]) for i, sid in enumerate(sc_ids)}

    loop_df = pd.read_csv(ROOT / "data/processed/multi_enzyme/loop_position_per_site_v3.csv")
    loop_df["site_id"] = loop_df["site_id"].astype(str)
    loop_df = loop_df.set_index("site_id")

    emb = torch.load(ROOT / "data/processed/multi_enzyme/embeddings/rnafm_pooled_v3.pt", weights_only=False)

    site_ids = a3b["site_id"].astype(str).tolist()
    labels = a3b["is_edited"].values

    seqs = {str(sid): all_seqs.get(str(sid), "N" * 201) for sid in site_ids}

    return site_ids, labels, seqs, struct_delta, mfe_dict, mfe_ed_dict, loop_df, emb


# ── Feature Extraction Functions ──────────────────────────────────────────

NUC_MAP = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3, "N": -1}


def feat_motif_24(site_ids, seqs, **kw):
    """Original 24-dim motif features."""
    return np.array([extract_motif_from_seq(seqs[sid]) for sid in site_ids], dtype=np.float32)


def feat_struct_delta_7(site_ids, struct_delta, **kw):
    """Original 7-dim structure delta features."""
    return np.array([struct_delta.get(sid, np.zeros(7)) for sid in site_ids], dtype=np.float32)


def feat_loop_9(site_ids, loop_df, **kw):
    """Original 9-dim loop geometry features."""
    cols = LOOP_FEATURE_COLS
    feats = []
    for sid in site_ids:
        if sid in loop_df.index:
            row = loop_df.loc[sid]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            vals = []
            for c in cols:
                if c in row.index:
                    vals.append(float(row[c]) if pd.notna(row[c]) else 0.0)
                else:
                    vals.append(0.0)
            feats.append(vals)
        else:
            feats.append([0.0] * len(cols))
    return np.array(feats, dtype=np.float32)


def feat_hand_40(site_ids, seqs, struct_delta, loop_df, **kw):
    """Original 40-dim hand features (motif24 + struct7 + loop9)."""
    m = feat_motif_24(site_ids, seqs=seqs)
    s = feat_struct_delta_7(site_ids, struct_delta=struct_delta)
    l = feat_loop_9(site_ids, loop_df=loop_df)
    return np.concatenate([m, s, l], axis=1)


def feat_extended_motif(site_ids, seqs, window=10, **kw):
    """Position-specific one-hot encoding at each position from -window to +window (excluding center).
    Dimension: 4 * (2*window).
    """
    dim = 4 * (2 * window)
    feats = np.zeros((len(site_ids), dim), dtype=np.float32)
    for i, sid in enumerate(site_ids):
        seq = seqs[sid].upper().replace("T", "U")
        idx = 0
        for offset in range(-window, window + 1):
            if offset == 0:
                continue
            pos = CENTER + offset
            if 0 <= pos < len(seq):
                nuc = NUC_MAP.get(seq[pos], -1)
                if nuc >= 0:
                    feats[i, idx + nuc] = 1.0
            idx += 4
    return feats


def feat_gc_content(site_ids, seqs, windows=(5, 10, 20, 50), **kw):
    """GC content in multiple windows around edit site."""
    feats = np.zeros((len(site_ids), len(windows)), dtype=np.float32)
    for i, sid in enumerate(site_ids):
        seq = seqs[sid].upper().replace("T", "U")
        for j, w in enumerate(windows):
            s = max(0, CENTER - w)
            e = min(len(seq), CENTER + w + 1)
            region = seq[s:e]
            gc = sum(1 for c in region if c in "GC") / max(len(region), 1)
            feats[i, j] = gc
    return feats


def feat_dinuc_freq(site_ids, seqs, window=20, **kw):
    """Dinucleotide frequencies in ±window around edit site. 16-dim."""
    dinucs = [a + b for a in "ACGU" for b in "ACGU"]
    feats = np.zeros((len(site_ids), 16), dtype=np.float32)
    for i, sid in enumerate(site_ids):
        seq = seqs[sid].upper().replace("T", "U")
        s = max(0, CENTER - window)
        e = min(len(seq), CENTER + window + 1)
        region = seq[s:e]
        total = max(len(region) - 1, 1)
        for k in range(len(region) - 1):
            dn = region[k:k+2]
            if dn in dinucs:
                feats[i, dinucs.index(dn)] += 1.0 / total
    return feats


def feat_trinuc_freq(site_ids, seqs, window=20, **kw):
    """Trinucleotide frequencies in ±window. 64-dim."""
    trinucs = [a + b + c for a in "ACGU" for b in "ACGU" for c in "ACGU"]
    feats = np.zeros((len(site_ids), 64), dtype=np.float32)
    for i, sid in enumerate(site_ids):
        seq = seqs[sid].upper().replace("T", "U")
        s = max(0, CENTER - window)
        e = min(len(seq), CENTER + window + 1)
        region = seq[s:e]
        total = max(len(region) - 2, 1)
        for k in range(len(region) - 2):
            tn = region[k:k+2+1]
            if tn in trinucs:
                feats[i, trinucs.index(tn)] += 1.0 / total
    return feats


def feat_kmer_spectrum(site_ids, seqs, k=4, window=30, **kw):
    """K-mer frequency spectrum in ±window. 4^k dim."""
    bases = "ACGU"
    from itertools import product
    kmers = [''.join(p) for p in product(bases, repeat=k)]
    kmer_idx = {km: i for i, km in enumerate(kmers)}
    dim = len(kmers)
    feats = np.zeros((len(site_ids), dim), dtype=np.float32)
    for i, sid in enumerate(site_ids):
        seq = seqs[sid].upper().replace("T", "U")
        s = max(0, CENTER - window)
        e = min(len(seq), CENTER + window + 1)
        region = seq[s:e]
        total = max(len(region) - k + 1, 1)
        for j in range(len(region) - k + 1):
            km = region[j:j+k]
            if km in kmer_idx:
                feats[i, kmer_idx[km]] += 1.0 / total
    return feats


def feat_positional_composition(site_ids, seqs, window=20, **kw):
    """Nucleotide composition at each position, but using sliding averages.
    5-nt sliding windows from -window to +window. Each window: 4 nuc fractions.
    Dimension: (2*window//5) * 4.
    """
    step = 5
    positions = list(range(-window, window + 1, step))
    feats = np.zeros((len(site_ids), len(positions) * 4), dtype=np.float32)
    for i, sid in enumerate(site_ids):
        seq = seqs[sid].upper().replace("T", "U")
        for j, p in enumerate(positions):
            s = max(0, CENTER + p - 2)
            e = min(len(seq), CENTER + p + 3)
            region = seq[s:e]
            for bi, b in enumerate("ACGU"):
                feats[i, j * 4 + bi] = sum(1 for c in region if c == b) / max(len(region), 1)
    return feats


def feat_nuc_asymmetry(site_ids, seqs, window=20, **kw):
    """Upstream vs downstream nucleotide composition asymmetry. 4-dim."""
    feats = np.zeros((len(site_ids), 4), dtype=np.float32)
    for i, sid in enumerate(site_ids):
        seq = seqs[sid].upper().replace("T", "U")
        up = seq[max(0, CENTER - window):CENTER]
        dn = seq[CENTER + 1:min(len(seq), CENTER + window + 1)]
        up_len = max(len(up), 1)
        dn_len = max(len(dn), 1)
        for bi, b in enumerate("ACGU"):
            up_frac = sum(1 for c in up if c == b) / up_len
            dn_frac = sum(1 for c in dn if c == b) / dn_len
            feats[i, bi] = up_frac - dn_frac
    return feats


def feat_purine_pyrimidine(site_ids, seqs, windows=(5, 10, 20), **kw):
    """Purine fraction, transition rate, and purine/pyrimidine runs in multiple windows."""
    feats = np.zeros((len(site_ids), len(windows) * 3), dtype=np.float32)
    for i, sid in enumerate(site_ids):
        seq = seqs[sid].upper().replace("T", "U")
        for j, w in enumerate(windows):
            s = max(0, CENTER - w)
            e = min(len(seq), CENTER + w + 1)
            region = seq[s:e]
            rlen = max(len(region), 1)
            # Purine fraction
            pur = sum(1 for c in region if c in "AG") / rlen
            feats[i, j * 3] = pur
            # Transition count (purine<->pyrimidine changes)
            trans = sum(1 for k in range(len(region) - 1)
                       if (region[k] in "AG") != (region[k+1] in "AG")) / max(rlen - 1, 1)
            feats[i, j * 3 + 1] = trans
            # Max purine run
            max_run = 0
            cur_run = 0
            for c in region:
                if c in "AG":
                    cur_run += 1
                    max_run = max(max_run, cur_run)
                else:
                    cur_run = 0
            feats[i, j * 3 + 2] = max_run / rlen
    return feats


def feat_extended_loop(site_ids, loop_df, **kw):
    """Extended loop features: loop_type encoding, dot_bracket patterns."""
    extra_cols = ["loop_type", "dist_to_left_boundary", "dist_to_right_boundary",
                  "dist_to_nearest_stem"]
    loop_types = ["hairpin", "internal", "bulge", "multibranch", "external"]
    dim = len(loop_types) + len(extra_cols) - 1  # one-hot for loop_type + numeric cols
    feats = np.zeros((len(site_ids), dim), dtype=np.float32)
    for i, sid in enumerate(site_ids):
        if sid in loop_df.index:
            row = loop_df.loc[sid]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            # Loop type one-hot
            lt = str(row.get("loop_type", "")).lower() if pd.notna(row.get("loop_type", np.nan)) else ""
            for j, ltype in enumerate(loop_types):
                if lt == ltype:
                    feats[i, j] = 1.0
            # Numeric features
            idx = len(loop_types)
            for c in extra_cols[1:]:
                if c in row.index and pd.notna(row[c]):
                    feats[i, idx] = float(row[c])
                idx += 1
    return feats


def feat_dot_bracket_encoding(site_ids, loop_df, seqs, window=15, **kw):
    """Encode the MFE dot-bracket structure around edit site.
    At each position: 3-dim one-hot for '.', '(', ')'.
    Also count structural elements in windows.
    Dimension: 3 * (2*window) + 12.
    """
    dim = 3 * (2 * window) + 12
    feats = np.zeros((len(site_ids), dim), dtype=np.float32)
    for i, sid in enumerate(site_ids):
        db = ""
        if sid in loop_df.index:
            row = loop_df.loc[sid]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            db = str(row.get("dot_bracket", "")) if pd.notna(row.get("dot_bracket", np.nan)) else ""

        if not db or len(db) < 2 * window + 1:
            continue

        center = len(db) // 2
        idx = 0
        for offset in range(-window, window + 1):
            if offset == 0:
                continue
            pos = center + offset
            if 0 <= pos < len(db):
                ch = db[pos]
                if ch == ".":
                    feats[i, idx] = 1.0
                elif ch == "(":
                    feats[i, idx + 1] = 1.0
                elif ch == ")":
                    feats[i, idx + 2] = 1.0
            idx += 3

        # Count structural elements in windows
        base = 3 * (2 * window)
        for j, w in enumerate([5, 10, 20, 30]):
            s = max(0, center - w)
            e = min(len(db), center + w + 1)
            region = db[s:e]
            rlen = max(len(region), 1)
            feats[i, base + j * 3] = sum(1 for c in region if c == ".") / rlen
            feats[i, base + j * 3 + 1] = sum(1 for c in region if c == "(") / rlen
            feats[i, base + j * 3 + 2] = sum(1 for c in region if c == ")") / rlen

    return feats


def feat_rnafm_pooled(site_ids, emb, **kw):
    """RNA-FM 640-dim pooled embeddings."""
    feats = np.zeros((len(site_ids), 640), dtype=np.float32)
    for i, sid in enumerate(site_ids):
        if sid in emb:
            feats[i] = emb[sid].numpy()
    return feats


def feat_rnafm_pca(site_ids, emb, n_components=20, **kw):
    """RNA-FM embeddings reduced to top PCA components."""
    from sklearn.decomposition import PCA
    raw = feat_rnafm_pooled(site_ids, emb=emb)
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(raw).astype(np.float32)


def feat_rnafm_pca_50(site_ids, emb, **kw):
    return feat_rnafm_pca(site_ids, emb=emb, n_components=50)


def feat_rnafm_pca_100(site_ids, emb, **kw):
    return feat_rnafm_pca(site_ids, emb=emb, n_components=100)


def feat_sequence_complexity(site_ids, seqs, **kw):
    """Sequence complexity measures: entropy, repeat content, homopolymer runs."""
    feats = np.zeros((len(site_ids), 8), dtype=np.float32)
    for i, sid in enumerate(site_ids):
        seq = seqs[sid].upper().replace("T", "U")
        for w_idx, w in enumerate([10, 30]):
            s = max(0, CENTER - w)
            e = min(len(seq), CENTER + w + 1)
            region = seq[s:e]
            rlen = max(len(region), 1)
            # Shannon entropy
            from collections import Counter
            counts = Counter(region)
            ent = -sum((c / rlen) * np.log2(c / rlen + 1e-10) for c in counts.values())
            feats[i, w_idx * 4] = ent
            # Max homopolymer run
            max_run = 0
            cur_run = 1
            for k in range(1, len(region)):
                if region[k] == region[k-1]:
                    cur_run += 1
                    max_run = max(max_run, cur_run)
                else:
                    cur_run = 1
            feats[i, w_idx * 4 + 1] = max_run / rlen
            # Unique 3-mers / possible 3-mers
            unique_3mers = len(set(region[k:k+3] for k in range(len(region) - 2)))
            feats[i, w_idx * 4 + 2] = unique_3mers / max(rlen - 2, 1)
            # CG skew
            cg = sum(1 for c in region if c == "C") - sum(1 for c in region if c == "G")
            feats[i, w_idx * 4 + 3] = cg / rlen
    return feats


def feat_interaction_motif_struct(site_ids, seqs, struct_delta, loop_df, **kw):
    """Interaction features: motif × structure cross-products."""
    motif = feat_motif_24(site_ids, seqs=seqs)
    struct = feat_struct_delta_7(site_ids, struct_delta=struct_delta)
    loop = feat_loop_9(site_ids, loop_df=loop_df)

    # Key interactions: each motif bit × each struct feature
    # Too many if all (24 × 7 = 168). Select top structural features.
    # Use: is_unpaired (loop[0]), relative_loop_position (loop[4]),
    #      local_unpaired_fraction (loop[8]), delta_mfe (struct[3])
    key_struct = np.column_stack([loop[:, 0], loop[:, 4], loop[:, 8], struct[:, 3]])  # 4 features
    # Motif: use 5' dinuc (4) + 3' dinuc (4) = 8 features
    key_motif = motif[:, :8]  # 8 features

    # Cross products: 8 × 4 = 32
    interactions = np.zeros((len(site_ids), 32), dtype=np.float32)
    idx = 0
    for j in range(8):
        for k in range(4):
            interactions[:, idx] = key_motif[:, j] * key_struct[:, k]
            idx += 1
    return interactions


def feat_polynomial_struct(site_ids, struct_delta, loop_df, **kw):
    """Polynomial features on structure: squares and pairwise products of top features."""
    struct = feat_struct_delta_7(site_ids, struct_delta=struct_delta)
    loop = feat_loop_9(site_ids, loop_df=loop_df)
    combined = np.concatenate([struct, loop], axis=1)  # 16 features

    # Squares (16) + select pairwise (top 10 pairs)
    squares = combined ** 2
    # Top pairwise products
    pairs = []
    pair_indices = [(0, 9), (0, 13), (3, 9), (3, 13), (3, 15),
                    (9, 13), (9, 15), (13, 15), (4, 9), (4, 13)]
    for a, b in pair_indices:
        if a < combined.shape[1] and b < combined.shape[1]:
            pairs.append(combined[:, a] * combined[:, b])
    pairs = np.column_stack(pairs) if pairs else np.zeros((len(site_ids), 0), dtype=np.float32)
    return np.concatenate([squares, pairs], axis=1)


def feat_mfe_features(site_ids, mfe_dict, mfe_ed_dict, **kw):
    """MFE-related features: raw MFE, edited MFE, delta, ratio."""
    feats = np.zeros((len(site_ids), 4), dtype=np.float32)
    for i, sid in enumerate(site_ids):
        mfe = mfe_dict.get(sid, 0.0)
        mfe_ed = mfe_ed_dict.get(sid, 0.0)
        feats[i, 0] = mfe
        feats[i, 1] = mfe_ed
        feats[i, 2] = mfe_ed - mfe
        feats[i, 3] = mfe_ed / (mfe + 1e-10) if abs(mfe) > 0.01 else 0.0
    return feats


# ── Evaluation ──────────────────────────────────────────────────────────

def evaluate_features(feature_matrix, labels, name, n_folds=5, seed=42):
    """Run 5-fold stratified CV with XGBoost. Returns mean AUROC."""
    X = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    y = labels

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    aurocs = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            eval_metric="logloss",
            random_state=seed + fold,
            n_jobs=-1,
            verbosity=0,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        y_prob = model.predict_proba(X_val)[:, 1]
        auroc = roc_auc_score(y_val, y_prob)
        aurocs.append(auroc)

    mean_auroc = np.mean(aurocs)
    std_auroc = np.std(aurocs)
    print(f"  {name:55s} dim={X.shape[1]:4d}  AUROC={mean_auroc:.4f} ± {std_auroc:.4f}  folds={[f'{a:.4f}' for a in aurocs]}")
    return mean_auroc, std_auroc, aurocs


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    site_ids, labels, seqs, struct_delta, mfe_dict, mfe_ed_dict, loop_df, emb = load_data()
    print(f"A3B sites: {len(site_ids)} ({sum(labels)} pos, {len(labels) - sum(labels)} neg)")

    kw = dict(
        site_ids=site_ids, seqs=seqs, struct_delta=struct_delta,
        loop_df=loop_df, emb=emb, mfe_dict=mfe_dict, mfe_ed_dict=mfe_ed_dict,
    )

    results = []

    def run(name, feat_fn, **extra_kw):
        merged = {**kw, **extra_kw}
        t0 = time.time()
        X = feat_fn(**merged)
        dt = time.time() - t0
        mean_auroc, std_auroc, folds = evaluate_features(X, labels, name)
        results.append({
            "name": name,
            "dim": X.shape[1],
            "auroc_mean": mean_auroc,
            "auroc_std": std_auroc,
            "folds": folds,
            "time_feat": dt,
        })
        return X

    def run_combined(name, arrays):
        X = np.concatenate(arrays, axis=1)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        mean_auroc, std_auroc, folds = evaluate_features(X, labels, name)
        results.append({
            "name": name,
            "dim": X.shape[1],
            "auroc_mean": mean_auroc,
            "auroc_std": std_auroc,
            "folds": folds,
            "time_feat": 0,
        })
        return X

    # ── Round 0: Baselines ──
    print("\n=== ROUND 0: BASELINES ===")
    X_motif24 = run("Motif24 (original)", feat_motif_24)
    X_struct7 = run("StructDelta7 (original)", feat_struct_delta_7)
    X_loop9 = run("Loop9 (original)", feat_loop_9)
    X_hand40 = run("Hand40 (original motif+struct+loop)", feat_hand_40)

    # ── Round 1: Extended Sequence Features ──
    print("\n=== ROUND 1: EXTENDED SEQUENCE ===")
    X_ext5 = run("ExtMotif ±5 (40-dim)", feat_extended_motif, window=5)
    X_ext10 = run("ExtMotif ±10 (80-dim)", feat_extended_motif, window=10)
    X_ext20 = run("ExtMotif ±20 (160-dim)", feat_extended_motif, window=20)
    X_gc = run("GC content (4 windows)", feat_gc_content)
    X_dinuc = run("Dinuc freq ±20 (16-dim)", feat_dinuc_freq)
    X_trinuc = run("Trinuc freq ±20 (64-dim)", feat_trinuc_freq)
    X_4mer = run("4-mer spectrum ±30 (256-dim)", feat_kmer_spectrum)
    X_poscomp = run("Positional composition ±20 (36-dim)", feat_positional_composition)
    X_asym = run("Nuc asymmetry ±20 (4-dim)", feat_nuc_asymmetry)
    X_purpyr = run("Purine/pyrimidine patterns (9-dim)", feat_purine_pyrimidine)
    X_complex = run("Sequence complexity (8-dim)", feat_sequence_complexity)

    # ── Round 1.5: Best sequence combinations ──
    print("\n=== ROUND 1.5: SEQUENCE COMBINATIONS ===")
    run_combined("ExtMotif±10 + GC + dinuc + trinuc", [X_ext10, X_gc, X_dinuc, X_trinuc])
    run_combined("ExtMotif±20 + all seq features", [X_ext20, X_gc, X_dinuc, X_trinuc, X_asym, X_purpyr, X_complex])
    X_all_seq = run_combined("ALL sequence features (mega)", [X_ext20, X_gc, X_dinuc, X_trinuc, X_4mer, X_poscomp, X_asym, X_purpyr, X_complex])

    # ── Round 2: Extended Structure Features ──
    print("\n=== ROUND 2: EXTENDED STRUCTURE ===")
    X_ext_loop = run("Extended loop features", feat_extended_loop)
    X_db = run("DotBracket encoding ±15", feat_dot_bracket_encoding)
    X_mfe = run("MFE features (4-dim)", feat_mfe_features)
    X_poly_struct = run("Polynomial struct (26-dim)", feat_polynomial_struct)

    # All structure
    X_all_struct = run_combined("ALL structure features",
        [X_struct7, X_loop9, X_ext_loop, X_db, X_mfe, X_poly_struct])

    # ── Round 2.5: Best seq + struct ──
    print("\n=== ROUND 2.5: SEQUENCE + STRUCTURE ===")
    run_combined("Hand40 + ExtMotif±10", [X_hand40, X_ext10])
    run_combined("Hand40 + all seq features", [X_hand40, X_all_seq])
    run_combined("All struct + all seq", [X_all_struct, X_all_seq])
    X_best_hand = run_combined("ExtMotif±20 + all struct + dinuc + trinuc + GC + complex",
        [X_ext20, X_all_struct, X_dinuc, X_trinuc, X_gc, X_complex, X_asym, X_purpyr])

    # ── Round 3: RNA-FM Embeddings ──
    print("\n=== ROUND 3: RNA-FM EMBEDDINGS ===")
    X_rnafm = run("RNA-FM 640-dim pooled", feat_rnafm_pooled)
    X_pca20 = run("RNA-FM PCA-20", feat_rnafm_pca)
    X_pca50 = run("RNA-FM PCA-50", feat_rnafm_pca_50)
    X_pca100 = run("RNA-FM PCA-100", feat_rnafm_pca_100)

    # RNA-FM + hand
    run_combined("RNA-FM 640 + Hand40", [X_rnafm, X_hand40])
    run_combined("RNA-FM PCA-50 + Hand40", [X_pca50, X_hand40])
    run_combined("RNA-FM PCA-20 + Hand40", [X_pca20, X_hand40])
    run_combined("RNA-FM 640 + best_hand_engineered", [X_rnafm, X_best_hand])

    # ── Round 4: Interaction Features ──
    print("\n=== ROUND 4: INTERACTION FEATURES ===")
    X_interact = run("Motif×Struct interactions (32-dim)", feat_interaction_motif_struct)
    run_combined("Hand40 + interactions", [X_hand40, X_interact])
    run_combined("Best hand + interactions", [X_best_hand, X_interact])

    # ── Round 5: Kitchen sink ──
    print("\n=== ROUND 5: KITCHEN SINK ===")
    X_everything = run_combined("EVERYTHING (all hand features)",
        [X_all_seq, X_all_struct, X_interact, X_poly_struct])
    run_combined("EVERYTHING + RNA-FM", [X_everything, X_rnafm])
    run_combined("EVERYTHING + RNA-FM PCA-50", [X_everything, X_pca50])

    # ── Save Results ──
    print("\n\n" + "=" * 100)
    print("SUMMARY (sorted by AUROC)")
    print("=" * 100)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("auroc_mean", ascending=False)

    for _, row in results_df.iterrows():
        print(f"  {row['name']:55s} dim={row['dim']:4d}  AUROC={row['auroc_mean']:.4f} ± {row['auroc_std']:.4f}")

    results_df.to_csv(OUT_DIR / "feature_challenge_results.csv", index=False)

    # Best non-RNAFM result
    non_rnafm = results_df[~results_df["name"].str.contains("RNA-FM")]
    best_hand = non_rnafm.iloc[0]
    print(f"\nBest hand-engineered: {best_hand['name']} → AUROC={best_hand['auroc_mean']:.4f}")

    best_rnafm = results_df[results_df["name"].str.contains("RNA-FM")].iloc[0]
    print(f"Best with RNA-FM:     {best_rnafm['name']} → AUROC={best_rnafm['auroc_mean']:.4f}")

    print(f"\nTarget (EditRNA+Hand): 0.810")
    print(f"Gap remaining (hand): {0.810 - best_hand['auroc_mean']:.4f}")
    print(f"Gap remaining (RNAFM): {0.810 - best_rnafm['auroc_mean']:.4f}")

    return results_df


if __name__ == "__main__":
    results_df = main()
