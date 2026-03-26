"""A3B Feature Engineering Challenge v2: Leak-free features.

Key discovery from v1: dot_bracket and loop_type columns in the v3 loop position file
have wildly different NaN rates between positives and negatives, creating trivial leakage.

This version:
1. Computes MFE structures from sequences using ViennaRNA (uniform coverage)
2. Extracts loop geometry from those structures
3. Tests all feature combinations leak-free
"""

import json
import os
import sys
import time
import functools
from pathlib import Path

# Force unbuffered output
print = functools.partial(print, flush=True)

import numpy as np
import pandas as pd
import torch
import RNA
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.apobec_feature_extraction import (
    extract_motif_from_seq, CENTER, _extract_loop_geometry, _entropy_at_pos,
)

OUT_DIR = ROOT / "experiments" / "multi_enzyme" / "outputs" / "a3b_feature_challenge"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NUC_MAP = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3, "N": -1}


# ── Compute ViennaRNA features for all sites ──

def compute_vienna_for_all(site_ids, seqs, cache_path=None):
    """Compute MFE structure, pairing probs, accessibilities for all sites.
    Returns dict with loop_geom (N,9), struct_delta (N,7), pairing_profile (N, profile_dim).
    """
    if cache_path and os.path.exists(cache_path):
        print(f"Loading ViennaRNA cache from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return {k: data[k] for k in data.keys()}

    n_sites = len(site_ids)
    loop_geom = np.zeros((n_sites, 9), dtype=np.float32)
    struct_delta = np.zeros((n_sites, 7), dtype=np.float32)
    pairing_orig = np.zeros((n_sites, 201), dtype=np.float32)
    pairing_edit = np.zeros((n_sites, 201), dtype=np.float32)
    access_orig = np.zeros((n_sites, 201), dtype=np.float32)
    access_edit = np.zeros((n_sites, 201), dtype=np.float32)
    mfes = np.zeros(n_sites, dtype=np.float32)
    mfes_edit = np.zeros(n_sites, dtype=np.float32)
    dot_brackets = []

    md = RNA.md()
    md.temperature = 37.0

    t0 = time.time()
    for i, sid in enumerate(site_ids):
        seq = seqs[sid].upper().replace("T", "U")
        n = len(seq)
        center = n // 2

        # Original
        fc = RNA.fold_compound(seq, md)
        mfe_ss, mfe_val = fc.mfe()
        fc.pf()
        bpp_raw = np.array(fc.bpp())
        bpp = bpp_raw[1:n+1, 1:n+1]
        pp = np.clip(np.sum(bpp, axis=0) + np.sum(bpp, axis=1), 0, 1)
        acc = 1.0 - pp
        ent_center = _entropy_at_pos(bpp, center)

        # Edited
        seq_list = list(seq)
        if center < len(seq_list) and seq_list[center].upper() == "C":
            seq_list[center] = "U"
        edited_seq = "".join(seq_list)
        fc_ed = RNA.fold_compound(edited_seq, md)
        mfe_ss_ed, mfe_val_ed = fc_ed.mfe()
        fc_ed.pf()
        bpp_raw_ed = np.array(fc_ed.bpp())
        bpp_ed = bpp_raw_ed[1:n+1, 1:n+1]
        pp_ed = np.clip(np.sum(bpp_ed, axis=0) + np.sum(bpp_ed, axis=1), 0, 1)
        acc_ed = 1.0 - pp_ed
        ent_center_ed = _entropy_at_pos(bpp_ed, center)

        # Store
        pairing_orig[i] = pp
        pairing_edit[i] = pp_ed
        access_orig[i] = acc
        access_edit[i] = acc_ed
        mfes[i] = mfe_val
        mfes_edit[i] = mfe_val_ed
        dot_brackets.append(mfe_ss)

        # Loop geometry from MFE structure
        loop_geom[i] = _extract_loop_geometry(mfe_ss, center)

        # Structure delta (7-dim)
        w = 10
        s, e = max(0, center - w), min(n, center + w + 1)
        dp = pp_ed - pp
        da = acc_ed - acc
        struct_delta[i, 0] = dp[center]
        struct_delta[i, 1] = da[center]
        struct_delta[i, 2] = ent_center_ed - ent_center
        struct_delta[i, 3] = mfe_val_ed - mfe_val
        struct_delta[i, 4] = np.mean(dp[s:e])
        struct_delta[i, 5] = np.mean(da[s:e])
        struct_delta[i, 6] = np.std(dp[s:e])

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (n_sites - i - 1) / rate
            print(f"  ViennaRNA: {i+1}/{n_sites} ({rate:.1f}/s, ~{remaining:.0f}s remaining)")

    print(f"  ViennaRNA done: {n_sites} sites in {time.time()-t0:.1f}s")

    if cache_path:
        np.savez_compressed(cache_path,
            loop_geom=loop_geom, struct_delta=struct_delta,
            pairing_orig=pairing_orig, pairing_edit=pairing_edit,
            access_orig=access_orig, access_edit=access_edit,
            mfes=mfes, mfes_edit=mfes_edit,
            dot_brackets=np.array(dot_brackets, dtype=object))

    return {
        "loop_geom": loop_geom, "struct_delta": struct_delta,
        "pairing_orig": pairing_orig, "pairing_edit": pairing_edit,
        "access_orig": access_orig, "access_edit": access_edit,
        "mfes": mfes, "mfes_edit": mfes_edit,
        "dot_brackets": np.array(dot_brackets, dtype=object),
    }


# ── Data Loading ──

def load_data():
    df = pd.read_csv(ROOT / "data/processed/multi_enzyme/splits_multi_enzyme_v3_with_negatives.csv")
    a3b = df[df["enzyme"] == "A3B"].copy().reset_index(drop=True)

    with open(ROOT / "data/processed/multi_enzyme/multi_enzyme_sequences_v3_with_negatives.json") as f:
        all_seqs = json.load(f)

    emb = torch.load(ROOT / "data/processed/multi_enzyme/embeddings/rnafm_pooled_v3.pt", weights_only=False)

    site_ids = a3b["site_id"].astype(str).tolist()
    labels = a3b["is_edited"].values
    seqs = {str(sid): all_seqs.get(str(sid), "N" * 201) for sid in site_ids}

    return site_ids, labels, seqs, emb, a3b


# ── Feature Extraction (all leak-free) ──

def feat_motif_24(site_ids, seqs, **kw):
    return np.array([extract_motif_from_seq(seqs[sid]) for sid in site_ids], dtype=np.float32)

def feat_struct_delta_7(vienna, **kw):
    return vienna["struct_delta"]

def feat_loop_9(vienna, **kw):
    return vienna["loop_geom"]

def feat_hand_40(site_ids, seqs, vienna, **kw):
    m = feat_motif_24(site_ids, seqs=seqs)
    s = vienna["struct_delta"]
    l = vienna["loop_geom"]
    return np.concatenate([m, s, l], axis=1)

def feat_extended_motif(site_ids, seqs, window=10, **kw):
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

def feat_gc_content(site_ids, seqs, **kw):
    windows = (5, 10, 20, 50)
    feats = np.zeros((len(site_ids), len(windows)), dtype=np.float32)
    for i, sid in enumerate(site_ids):
        seq = seqs[sid].upper().replace("T", "U")
        for j, w in enumerate(windows):
            s = max(0, CENTER - w)
            e = min(len(seq), CENTER + w + 1)
            region = seq[s:e]
            feats[i, j] = sum(1 for c in region if c in "GC") / max(len(region), 1)
    return feats

def feat_dinuc_freq(site_ids, seqs, window=20, **kw):
    dinucs = [a + b for a in "ACGU" for b in "ACGU"]
    feats = np.zeros((len(site_ids), 16), dtype=np.float32)
    for i, sid in enumerate(site_ids):
        seq = seqs[sid].upper().replace("T", "U")
        s, e = max(0, CENTER - window), min(len(seq), CENTER + window + 1)
        region = seq[s:e]
        total = max(len(region) - 1, 1)
        for k in range(len(region) - 1):
            dn = region[k:k+2]
            if dn in dinucs:
                feats[i, dinucs.index(dn)] += 1.0 / total
    return feats

def feat_trinuc_freq(site_ids, seqs, window=20, **kw):
    trinucs = [a + b + c for a in "ACGU" for b in "ACGU" for c in "ACGU"]
    feats = np.zeros((len(site_ids), 64), dtype=np.float32)
    for i, sid in enumerate(site_ids):
        seq = seqs[sid].upper().replace("T", "U")
        s, e = max(0, CENTER - window), min(len(seq), CENTER + window + 1)
        region = seq[s:e]
        total = max(len(region) - 2, 1)
        for k in range(len(region) - 2):
            tn = region[k:k+3]
            if tn in trinucs:
                feats[i, trinucs.index(tn)] += 1.0 / total
    return feats

def feat_kmer_spectrum(site_ids, seqs, k=4, window=30, **kw):
    from itertools import product
    kmers = [''.join(p) for p in product("ACGU", repeat=k)]
    kmer_idx = {km: i for i, km in enumerate(kmers)}
    dim = len(kmers)
    feats = np.zeros((len(site_ids), dim), dtype=np.float32)
    for i, sid in enumerate(site_ids):
        seq = seqs[sid].upper().replace("T", "U")
        s, e = max(0, CENTER - window), min(len(seq), CENTER + window + 1)
        region = seq[s:e]
        total = max(len(region) - k + 1, 1)
        for j in range(len(region) - k + 1):
            km = region[j:j+k]
            if km in kmer_idx:
                feats[i, kmer_idx[km]] += 1.0 / total
    return feats

def feat_sequence_complexity(site_ids, seqs, **kw):
    from collections import Counter
    feats = np.zeros((len(site_ids), 8), dtype=np.float32)
    for i, sid in enumerate(site_ids):
        seq = seqs[sid].upper().replace("T", "U")
        for w_idx, w in enumerate([10, 30]):
            s, e = max(0, CENTER - w), min(len(seq), CENTER + w + 1)
            region = seq[s:e]
            rlen = max(len(region), 1)
            counts = Counter(region)
            ent = -sum((c / rlen) * np.log2(c / rlen + 1e-10) for c in counts.values())
            feats[i, w_idx * 4] = ent
            max_run, cur_run = 0, 1
            for k in range(1, len(region)):
                if region[k] == region[k-1]: cur_run += 1; max_run = max(max_run, cur_run)
                else: cur_run = 1
            feats[i, w_idx * 4 + 1] = max_run / rlen
            feats[i, w_idx * 4 + 2] = len(set(region[k:k+3] for k in range(len(region)-2))) / max(rlen-2, 1)
            cg = sum(1 for c in region if c == "C") - sum(1 for c in region if c == "G")
            feats[i, w_idx * 4 + 3] = cg / rlen
    return feats

def feat_purine_pyrimidine(site_ids, seqs, **kw):
    windows = (5, 10, 20)
    feats = np.zeros((len(site_ids), len(windows) * 3), dtype=np.float32)
    for i, sid in enumerate(site_ids):
        seq = seqs[sid].upper().replace("T", "U")
        for j, w in enumerate(windows):
            s, e = max(0, CENTER - w), min(len(seq), CENTER + w + 1)
            region = seq[s:e]
            rlen = max(len(region), 1)
            feats[i, j*3] = sum(1 for c in region if c in "AG") / rlen
            feats[i, j*3+1] = sum(1 for k in range(len(region)-1) if (region[k] in "AG") != (region[k+1] in "AG")) / max(rlen-1, 1)
            max_run, cur_run = 0, 0
            for c in region:
                if c in "AG": cur_run += 1; max_run = max(max_run, cur_run)
                else: cur_run = 0
            feats[i, j*3+2] = max_run / rlen
    return feats

def feat_nuc_asymmetry(site_ids, seqs, **kw):
    feats = np.zeros((len(site_ids), 4), dtype=np.float32)
    for i, sid in enumerate(site_ids):
        seq = seqs[sid].upper().replace("T", "U")
        up = seq[max(0, CENTER-20):CENTER]
        dn = seq[CENTER+1:min(len(seq), CENTER+21)]
        for bi, b in enumerate("ACGU"):
            feats[i, bi] = sum(1 for c in up if c == b)/max(len(up),1) - sum(1 for c in dn if c == b)/max(len(dn),1)
    return feats

def feat_pairing_profile(vienna, **kw):
    """Pairing probability profile at positions ±10 around edit site (both orig and edited). 42-dim."""
    pp = vienna["pairing_orig"]
    pp_ed = vienna["pairing_edit"]
    n = pp.shape[0]
    w = 10
    # pp at each position, pp_ed at each position = 2*(2w+1) = 42
    feats = np.zeros((n, 42), dtype=np.float32)
    for j, offset in enumerate(range(-w, w+1)):
        pos = CENTER + offset
        feats[:, j] = pp[:, pos]
        feats[:, 21 + j] = pp_ed[:, pos]
    return feats

def feat_accessibility_profile(vienna, **kw):
    """Accessibility profile at positions ±10 (orig and edited). 42-dim."""
    acc = vienna["access_orig"]
    acc_ed = vienna["access_edit"]
    n = acc.shape[0]
    w = 10
    feats = np.zeros((n, 42), dtype=np.float32)
    for j, offset in enumerate(range(-w, w+1)):
        pos = CENTER + offset
        feats[:, j] = acc[:, pos]
        feats[:, 21 + j] = acc_ed[:, pos]
    return feats

def feat_delta_pairing_profile(vienna, **kw):
    """Delta pairing probability profile ±20 positions. 41-dim."""
    dp = vienna["pairing_edit"] - vienna["pairing_orig"]
    n = dp.shape[0]
    w = 20
    feats = np.zeros((n, 2*w+1), dtype=np.float32)
    for j, offset in enumerate(range(-w, w+1)):
        feats[:, j] = dp[:, CENTER + offset]
    return feats

def feat_windowed_struct_stats(vienna, **kw):
    """Windowed statistics of pairing/accessibility at multiple scales. ~50-dim."""
    pp = vienna["pairing_orig"]
    pp_ed = vienna["pairing_edit"]
    acc = vienna["access_orig"]
    acc_ed = vienna["access_edit"]
    mfes = vienna["mfes"]
    mfes_ed = vienna["mfes_edit"]
    n = pp.shape[0]

    feats = []
    for i in range(n):
        feat = []
        for w in [5, 11, 21]:
            s, e = max(0, CENTER - w//2), min(201, CENTER + w//2 + 1)
            win_pp = pp[i, s:e]
            win_pp_ed = pp_ed[i, s:e]
            feat.extend([win_pp.mean(), win_pp.std(), win_pp_ed.mean(), win_pp_ed.std(), (win_pp_ed - win_pp).mean()])
        for w in [5, 11, 21]:
            s, e = max(0, CENTER - w//2), min(201, CENTER + w//2 + 1)
            win_acc = acc[i, s:e]
            win_acc_ed = acc_ed[i, s:e]
            feat.extend([win_acc.mean(), win_acc.std(), win_acc_ed.mean(), win_acc_ed.std(), (win_acc_ed - win_acc).mean()])
        feat.extend([pp[i, CENTER], pp_ed[i, CENTER], pp_ed[i, CENTER] - pp[i, CENTER],
                     acc[i, CENTER], acc_ed[i, CENTER], acc_ed[i, CENTER] - acc[i, CENTER]])
        feat.extend([mfes[i], mfes_ed[i], mfes_ed[i] - mfes[i]])
        dp = pp_ed[i] - pp[i]
        for b in range(10):
            s, e = b*20, min(201, (b+1)*20)
            feat.append(dp[s:e].mean())
        feat.append(dp[90:111].mean())
        feats.append(feat)
    return np.array(feats, dtype=np.float32)

def feat_dot_bracket_clean(vienna, window=15, **kw):
    """Dot-bracket one-hot encoding from freshly computed MFE structures (no leakage)."""
    dbs = vienna["dot_brackets"]
    n = len(dbs)
    dim = 3 * (2 * window) + 12
    feats = np.zeros((n, dim), dtype=np.float32)
    for i, db in enumerate(dbs):
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
                if ch == ".": feats[i, idx] = 1.0
                elif ch == "(": feats[i, idx + 1] = 1.0
                elif ch == ")": feats[i, idx + 2] = 1.0
            idx += 3
        base = 3 * (2 * window)
        for j, w in enumerate([5, 10, 20, 30]):
            s, e = max(0, center - w), min(len(db), center + w + 1)
            region = db[s:e]
            rlen = max(len(region), 1)
            feats[i, base + j*3] = sum(1 for c in region if c == ".") / rlen
            feats[i, base + j*3 + 1] = sum(1 for c in region if c == "(") / rlen
            feats[i, base + j*3 + 2] = sum(1 for c in region if c == ")") / rlen
    return feats

def feat_struct_elements(vienna, **kw):
    """Count structural elements: stems, loops, bulges from dot-bracket. 12-dim."""
    dbs = vienna["dot_brackets"]
    n = len(dbs)
    feats = np.zeros((n, 12), dtype=np.float32)
    for i, db in enumerate(dbs):
        if not db:
            continue
        center = len(db) // 2
        for j, w in enumerate([10, 25, 50]):
            s, e = max(0, center - w), min(len(db), center + w + 1)
            region = db[s:e]
            rlen = max(len(region), 1)
            # Count transitions (structural boundaries)
            transitions = sum(1 for k in range(len(region)-1) if region[k] != region[k+1])
            feats[i, j*4] = transitions / rlen
            # Number of loop regions (runs of dots)
            in_loop = False
            n_loops = 0
            for c in region:
                if c == ".":
                    if not in_loop: n_loops += 1; in_loop = True
                else:
                    in_loop = False
            feats[i, j*4+1] = n_loops / rlen
            # Number of stem regions
            in_stem = False
            n_stems = 0
            for c in region:
                if c in "()":
                    if not in_stem: n_stems += 1; in_stem = True
                else:
                    in_stem = False
            feats[i, j*4+2] = n_stems / rlen
            # Fraction unpaired
            feats[i, j*4+3] = sum(1 for c in region if c == ".") / rlen
    return feats

def feat_rnafm_pooled(site_ids, emb, **kw):
    feats = np.zeros((len(site_ids), 640), dtype=np.float32)
    for i, sid in enumerate(site_ids):
        if sid in emb:
            feats[i] = emb[sid].numpy()
    return feats

def feat_rnafm_pca(site_ids, emb, n_components=20, **kw):
    from sklearn.decomposition import PCA
    raw = feat_rnafm_pooled(site_ids, emb=emb)
    return PCA(n_components=n_components, random_state=42).fit_transform(raw).astype(np.float32)


# ── Evaluation ──

def evaluate_features(X, labels, name, n_folds=5, seed=42):
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    aurocs = []
    for fold, (tr, va) in enumerate(skf.split(X, labels)):
        model = XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            eval_metric="logloss", random_state=seed+fold, n_jobs=-1, verbosity=0,
        )
        model.fit(X[tr], labels[tr], eval_set=[(X[va], labels[va])], verbose=False)
        aurocs.append(roc_auc_score(labels[va], model.predict_proba(X[va])[:, 1]))
    m, s = np.mean(aurocs), np.std(aurocs)
    print(f"  {name:60s} dim={X.shape[1]:4d}  AUROC={m:.4f} +/- {s:.4f}  [{', '.join(f'{a:.4f}' for a in aurocs)}]")
    return m, s, aurocs


def main():
    print("Loading data...")
    site_ids, labels, seqs, emb, a3b_df = load_data()
    print(f"A3B: {len(site_ids)} sites ({sum(labels)} pos, {len(labels)-sum(labels)} neg)")

    print("\nComputing ViennaRNA features for all A3B sites...")
    cache_path = OUT_DIR / "vienna_a3b_cache.npz"
    vienna = compute_vienna_for_all(site_ids, seqs, cache_path=str(cache_path))

    results = []

    def run(name, X):
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        m, s, folds = evaluate_features(X, labels, name)
        results.append({"name": name, "dim": X.shape[1], "auroc_mean": m, "auroc_std": s, "folds": folds})
        return X

    def combine(name, arrays):
        return run(name, np.concatenate(arrays, axis=1))

    kw = dict(site_ids=site_ids, seqs=seqs, vienna=vienna, emb=emb)

    # ── Round 0: Baselines (leak-free) ──
    print("\n=== ROUND 0: BASELINES (leak-free) ===")
    X_motif = run("Motif24", feat_motif_24(**kw))
    X_struct = run("StructDelta7", feat_struct_delta_7(**kw))
    X_loop = run("Loop9 (fresh ViennaRNA)", feat_loop_9(**kw))
    X_hand = combine("Hand40 (motif+struct+loop)", [X_motif, X_struct, X_loop])

    # ── Round 1: Sequence features ──
    print("\n=== ROUND 1: SEQUENCE FEATURES ===")
    X_ext5 = run("ExtMotif ±5", feat_extended_motif(**kw, window=5))
    X_ext10 = run("ExtMotif ±10", feat_extended_motif(**kw, window=10))
    X_ext20 = run("ExtMotif ±20", feat_extended_motif(**kw, window=20))
    X_gc = run("GC content (4 windows)", feat_gc_content(**kw))
    X_dinuc = run("Dinuc freq ±20", feat_dinuc_freq(**kw))
    X_trinuc = run("Trinuc freq ±20", feat_trinuc_freq(**kw))
    X_4mer = run("4-mer spectrum ±30", feat_kmer_spectrum(**kw))
    X_complex = run("Sequence complexity", feat_sequence_complexity(**kw))
    X_purpyr = run("Purine/pyrimidine", feat_purine_pyrimidine(**kw))
    X_asym = run("Nuc asymmetry", feat_nuc_asymmetry(**kw))

    # Best seq combos
    print("\n=== ROUND 1.5: SEQUENCE COMBINATIONS ===")
    X_seq_all = combine("ALL seq features", [X_ext20, X_gc, X_dinuc, X_trinuc, X_4mer, X_complex, X_purpyr, X_asym])

    # ── Round 2: Structure features (all leak-free from ViennaRNA) ──
    print("\n=== ROUND 2: STRUCTURE FEATURES (leak-free) ===")
    X_pair_prof = run("Pairing profile ±10", feat_pairing_profile(**kw))
    X_acc_prof = run("Accessibility profile ±10", feat_accessibility_profile(**kw))
    X_dp_prof = run("Delta pairing profile ±20", feat_delta_pairing_profile(**kw))
    X_wstats = run("Windowed struct stats (~50-dim)", feat_windowed_struct_stats(**kw))
    X_db = run("DotBracket ±15 (clean)", feat_dot_bracket_clean(**kw))
    X_selements = run("Structural elements (12-dim)", feat_struct_elements(**kw))

    X_struct_all = combine("ALL structure features", [X_struct, X_loop, X_pair_prof, X_acc_prof, X_dp_prof, X_wstats, X_db, X_selements])

    # ── Round 2.5: Seq + Struct ──
    print("\n=== ROUND 2.5: SEQ + STRUCT ===")
    combine("Hand40 + ExtMotif±10", [X_hand, X_ext10])
    combine("Hand40 + trinuc + 4mer", [X_hand, X_trinuc, X_4mer])
    combine("All struct + all seq", [X_struct_all, X_seq_all])
    X_best_hand = combine("Best hand combo: ext20+allstruct+dinuc+trinuc+gc",
        [X_ext20, X_struct_all, X_dinuc, X_trinuc, X_gc, X_complex])

    # ── Round 3: RNA-FM ──
    print("\n=== ROUND 3: RNA-FM EMBEDDINGS ===")
    X_rnafm = run("RNA-FM 640-dim", feat_rnafm_pooled(**kw))
    X_pca20 = run("RNA-FM PCA-20", feat_rnafm_pca(**kw, n_components=20))
    X_pca50 = run("RNA-FM PCA-50", feat_rnafm_pca(**kw, n_components=50))
    X_pca100 = run("RNA-FM PCA-100", feat_rnafm_pca(**kw, n_components=100))

    combine("RNA-FM 640 + Hand40", [X_rnafm, X_hand])
    combine("RNA-FM PCA-50 + Hand40", [X_pca50, X_hand])
    combine("RNA-FM 640 + best_hand", [X_rnafm, X_best_hand])

    # ── Round 4: Kitchen sink ──
    print("\n=== ROUND 4: KITCHEN SINK ===")
    X_everything = combine("EVERYTHING hand", [X_seq_all, X_struct_all])
    combine("EVERYTHING + RNA-FM", [X_everything, X_rnafm])
    combine("EVERYTHING + RNA-FM PCA-50", [X_everything, X_pca50])

    # ── Summary ──
    print("\n" + "="*110)
    print("SUMMARY (sorted by AUROC)")
    print("="*110)
    results_df = pd.DataFrame(results).sort_values("auroc_mean", ascending=False)
    for _, r in results_df.iterrows():
        print(f"  {r['name']:60s} dim={r['dim']:4d}  AUROC={r['auroc_mean']:.4f} +/- {r['auroc_std']:.4f}")

    results_df.to_csv(OUT_DIR / "feature_challenge_v2_results.csv", index=False)

    non_rnafm = results_df[~results_df["name"].str.contains("RNA-FM")]
    best_hand = non_rnafm.iloc[0]
    best_rnafm = results_df[results_df["name"].str.contains("RNA-FM")].iloc[0]
    print(f"\nBest hand-engineered: {best_hand['name']} -> AUROC={best_hand['auroc_mean']:.4f}")
    print(f"Best with RNA-FM:     {best_rnafm['name']} -> AUROC={best_rnafm['auroc_mean']:.4f}")
    print(f"Target (EditRNA+Hand): 0.810")
    print(f"Gap (hand):  {0.810 - best_hand['auroc_mean']:.4f}")
    print(f"Gap (RNAFM): {0.810 - best_rnafm['auroc_mean']:.4f}")

    return results_df


if __name__ == "__main__":
    main()
