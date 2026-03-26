#!/usr/bin/env python
"""Structure-negative tier evaluation: GB vs RNAsee_RF.

Adds ~2,000 structure-negative sites (TC-motif, all in stems/paired regions)
from the tier3 pool to the existing A3A dataset, then runs 5-fold CV comparing
GB_HandFeatures, GB_AllFeatures, and RNAsee_RF.

Uses transcript-based sequence extraction (same as generate_tiered_negatives.py).

Usage:
    conda run -n quris python experiments/apobec3a/exp_struct_neg_comparison.py
"""

import json
import logging
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from pyfaidx import Fasta
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score,
)
from sklearn.model_selection import KFold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Paths
SPLITS_CSV = PROJECT_ROOT / "data" / "processed" / "splits_expanded_a3a.csv"
TIER3_CSV = PROJECT_ROOT / "data" / "processed" / "negatives_tier3.csv"
GENOME_FA = PROJECT_ROOT / "data" / "raw" / "genomes" / "hg38.fa"
REFGENE = PROJECT_ROOT / "data" / "raw" / "genomes" / "refGene.txt"
SEQ_JSON = PROJECT_ROOT / "data" / "processed" / "site_sequences.json"
STRUCT_CACHE = PROJECT_ROOT / "data" / "processed" / "embeddings" / "vienna_structure_cache.npz"
LOOP_CSV = (PROJECT_ROOT / "experiments" / "apobec3a" / "outputs"
            / "loop_position" / "loop_position_per_site.csv")
OUTPUT_DIR = (PROJECT_ROOT / "experiments" / "apobec3a" / "outputs"
              / "struct_neg_comparison")

CENTER = 100
WINDOW = 201
FLANK = WINDOW // 2  # 100
N_STRUCT_NEG = 2000
SEED = 42


# ---------------------------------------------------------------------------
# Transcript sequence extraction (from generate_tiered_negatives.py)
# ---------------------------------------------------------------------------
def load_refgene(path):
    """Load refGene.txt into DataFrame."""
    cols = ["bin", "name", "chrom", "strand", "txStart", "txEnd",
            "cdsStart", "cdsEnd", "exonCount", "exonStarts", "exonEnds",
            "score", "name2", "cdsStartStat", "cdsEndStat", "exonFrames"]
    return pd.read_csv(path, sep="\t", names=cols, comment="#")


def get_exon_intervals(row):
    starts = [int(s) for s in row["exonStarts"].rstrip(",").split(",") if s]
    ends = [int(e) for e in row["exonEnds"].rstrip(",").split(",") if e]
    return list(zip(starts, ends))


def extract_transcript_exonic_sequence(genome, chrom, exons, strand):
    """Extract concatenated exonic sequence and coordinate map."""
    if chrom not in genome:
        return "", []
    chrom_len = len(genome[chrom])
    seq_parts = []
    coord_map = []
    for start, end in sorted(exons):
        s = max(0, start)
        e = min(chrom_len, end)
        if s >= e:
            continue
        seg = str(genome[chrom][s:e]).upper()
        seq_parts.append(seg)
        coord_map.extend(range(s, e))
    exonic_seq = "".join(seq_parts)
    if strand == "-":
        comp = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
        exonic_seq = "".join(comp.get(b, "N") for b in reversed(exonic_seq))
        coord_map = list(reversed(coord_map))
    return exonic_seq, coord_map


def extract_window(exonic_seq, pos_in_transcript, flank=FLANK):
    """Extract 201nt window centered on position in transcript."""
    start = pos_in_transcript - flank
    end = pos_in_transcript + flank + 1
    if start < 0 or end > len(exonic_seq):
        return None
    window = exonic_seq[start:end]
    if len(window) != WINDOW:
        return None
    return window


# ---------------------------------------------------------------------------
# RNAsee 50-bit encoding
# ---------------------------------------------------------------------------
def encode_rnasee(seq):
    BWD, FWD = 15, 10
    features = np.zeros((BWD + FWD) * 2, dtype=np.float32)
    if len(seq) < CENTER + FWD + 1:
        return features
    idx = 0
    for offset in range(-BWD, 0):
        pos = CENTER + offset
        if 0 <= pos < len(seq):
            nuc = seq[pos].upper()
            features[idx] = 1.0 if nuc in ("A", "G") else 0.0
            features[idx + 1] = 1.0 if nuc in ("G", "C") else 0.0
        idx += 2
    for offset in range(1, FWD + 1):
        pos = CENTER + offset
        if 0 <= pos < len(seq):
            nuc = seq[pos].upper()
            features[idx] = 1.0 if nuc in ("A", "G") else 0.0
            features[idx + 1] = 1.0 if nuc in ("G", "C") else 0.0
        idx += 2
    return features


# ---------------------------------------------------------------------------
# Motif features (24-dim)
# ---------------------------------------------------------------------------
NUC_MAP = {"A": 0, "C": 1, "G": 2, "T": 3, "U": 3}

def encode_motif(seq):
    feat = np.zeros(24, dtype=np.float32)
    c = CENTER
    for i, offset in enumerate([-2, -1, 1, 2]):
        pos = c + offset
        if 0 <= pos < len(seq):
            nuc = seq[pos].upper()
            if nuc in NUC_MAP:
                feat[i * 4 + NUC_MAP[nuc]] = 1.0
    if c >= 1:
        n5 = seq[c - 1].upper()
        if n5 in NUC_MAP:
            feat[16 + NUC_MAP[n5]] = 1.0
    if c + 1 < len(seq):
        n3 = seq[c + 1].upper()
        if n3 in NUC_MAP:
            feat[20 + NUC_MAP[n3]] = 1.0
    return feat


# ---------------------------------------------------------------------------
# ViennaRNA structure features
# ---------------------------------------------------------------------------
def compute_vienna_features(seq):
    """Compute struct_delta(7), loop_feats(9), baseline_struct(6) for a 201nt seq."""
    import RNA
    seq_str = seq.upper().replace("U", "T")
    seq_rna = seq_str.replace("T", "U")
    seq_ed = list(seq_rna)
    seq_ed[CENTER] = "U"
    seq_ed = "".join(seq_ed)

    fc_orig = RNA.fold_compound(seq_rna)
    ss_orig, mfe_orig = fc_orig.mfe()
    fc_orig.exp_params_rescale(mfe_orig)
    fc_orig.pf()
    bpp_orig = np.array(fc_orig.bpp())
    pairing_orig = np.clip(np.sum(bpp_orig, axis=0) + np.sum(bpp_orig, axis=1), 0, 1)

    fc_ed = RNA.fold_compound(seq_ed)
    ss_ed, mfe_ed = fc_ed.mfe()
    fc_ed.exp_params_rescale(mfe_ed)
    fc_ed.pf()
    bpp_ed = np.array(fc_ed.bpp())
    pairing_ed = np.clip(np.sum(bpp_ed, axis=0) + np.sum(bpp_ed, axis=1), 0, 1)

    acc_orig = 1.0 - pairing_orig
    acc_ed = 1.0 - pairing_ed
    delta_pairing = pairing_ed - pairing_orig
    delta_acc = acc_ed - acc_orig

    LOCAL = 10
    sl = slice(CENTER - LOCAL, CENTER + LOCAL + 1)

    # Entropy at center
    def _center_entropy(bpp_mat, pos):
        n = bpp_mat.shape[0]
        probs = []
        for j in range(n):
            p = bpp_mat[pos][j] if j > pos else bpp_mat[j][pos]
            if p > 1e-10:
                probs.append(p)
        unpaired = max(0, 1.0 - sum(probs))
        if unpaired > 1e-10:
            probs.append(unpaired)
        return -sum(p * np.log2(p) for p in probs if p > 1e-10)

    entropy_center = _center_entropy(bpp_orig, CENTER)

    struct_delta = np.array([
        delta_pairing[CENTER],
        delta_acc[CENTER],
        0.0,
        mfe_ed - mfe_orig,
        np.mean(delta_pairing[sl]),
        np.std(delta_pairing[sl]),
        np.mean(delta_acc[sl]),
    ], dtype=np.float32)

    loop_feats = _compute_loop_features(ss_orig, CENTER)

    baseline_struct = np.array([
        pairing_orig[CENTER],
        acc_orig[CENTER],
        entropy_center,
        np.mean(pairing_orig[sl]),
        np.mean(acc_orig[sl]),
        0.0,  # mean entropy placeholder
    ], dtype=np.float32)

    return struct_delta, loop_feats, baseline_struct


def _compute_loop_features(dot_bracket, pos):
    feat = np.zeros(9, dtype=np.float32)
    n = len(dot_bracket)
    is_unpaired = dot_bracket[pos] == "."
    feat[3] = 1.0 if is_unpaired else 0.0

    local_start = max(0, pos - 10)
    local_end = min(n, pos + 11)
    local_region = dot_bracket[local_start:local_end]
    feat[4] = local_region.count(".") / len(local_region)

    left_stem = 0
    for i in range(pos - 1, -1, -1):
        if dot_bracket[i] in ("(", ")"):
            left_stem += 1
        else:
            break
    right_stem = 0
    for i in range(pos + 1, n):
        if dot_bracket[i] in ("(", ")"):
            right_stem += 1
        else:
            break
    feat[5] = left_stem
    feat[6] = right_stem
    feat[7] = max(left_stem, right_stem)

    if is_unpaired:
        left_bound = pos
        for i in range(pos - 1, -1, -1):
            if dot_bracket[i] != ".":
                left_bound = i + 1
                break
        right_bound = pos
        for i in range(pos + 1, n):
            if dot_bracket[i] != ".":
                right_bound = i - 1
                break
        loop_size = right_bound - left_bound + 1
        feat[0] = loop_size
        feat[1] = min(pos - left_bound, right_bound - pos)
        feat[2] = (pos - left_bound) / max(loop_size, 1)

    dist_junction = 0
    for d in range(1, n):
        if pos - d >= 0 and dot_bracket[pos - d] in ("(", ")"):
            dist_junction = d
            break
        if pos + d < n and dot_bracket[pos + d] in ("(", ")"):
            dist_junction = d
            break
    feat[8] = dist_junction
    return feat


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # 1. Load existing dataset
    logger.info("Loading existing dataset...")
    splits = pd.read_csv(SPLITS_CSV)
    existing_coords = set(zip(splits["chr"], splits["start"]))
    n_orig_pos = int((splits["is_edited"] == 1).sum())
    n_orig_neg = int((splits["is_edited"] == 0).sum())
    logger.info(f"  Existing: {len(splits)} sites ({n_orig_pos} pos, {n_orig_neg} neg)")

    # 2. Load tier3 pool and select available sites
    logger.info("Loading tier3 pool...")
    tier3 = pd.read_csv(TIER3_CSV)
    available = tier3[~tier3.apply(
        lambda r: (r["chr"], r["start"]) in existing_coords, axis=1
    )].copy()
    logger.info(f"  Tier3 pool: {len(tier3)}, available: {len(available)}")

    # 3. Extract sequences using transcript-based approach
    logger.info("Loading genome and refGene...")
    genome = Fasta(str(GENOME_FA))
    refgene = load_refgene(REFGENE)

    # Build transcript lookup
    tx_lookup = {}
    for _, row in refgene.iterrows():
        tx_lookup[row["name"]] = row

    # Extract sequences per transcript
    logger.info("Extracting sequences from transcripts...")
    tx_cache = {}  # transcript -> (exonic_seq, coord_map)
    new_sequences = {}
    n_ok = 0
    n_fail = 0

    for _, site in available.iterrows():
        tx_name = site["transcript"]
        if tx_name not in tx_lookup:
            n_fail += 1
            continue

        if tx_name not in tx_cache:
            tx_row = tx_lookup[tx_name]
            exons = get_exon_intervals(tx_row)
            exonic_seq, coord_map = extract_transcript_exonic_sequence(
                genome, tx_row["chrom"], exons, tx_row["strand"]
            )
            tx_cache[tx_name] = (exonic_seq, coord_map)

        exonic_seq, coord_map = tx_cache[tx_name]
        pos = site["pos_in_transcript"]

        window = extract_window(exonic_seq, pos)
        if window is None:
            n_fail += 1
            continue
        if window[CENTER] != "C":
            n_fail += 1
            continue

        new_sequences[site["site_id"]] = window
        n_ok += 1

    logger.info(f"  Extracted: {n_ok}, failed: {n_fail}")

    # 4. Sample from successfully extracted sites
    available_with_seq = available[available["site_id"].isin(new_sequences)].copy()
    n_sample = min(N_STRUCT_NEG, len(available_with_seq))
    struct_neg = available_with_seq.sample(n=n_sample, random_state=SEED)
    logger.info(f"  Sampled {n_sample} structure-negative sites")

    # Keep only sampled sequences
    sampled_seqs = {sid: new_sequences[sid] for sid in struct_neg["site_id"]}

    # 5. Load existing sequences
    logger.info("Loading existing sequences...")
    with open(SEQ_JSON) as f:
        sequences = json.load(f)

    # 6. Build combined dataset
    all_site_ids = list(splits["site_id"].values) + list(struct_neg["site_id"].values)
    all_labels = np.array(
        list(splits["is_edited"].values) + [0] * len(struct_neg), dtype=int
    )
    all_sequences = {}
    all_sequences.update(sequences)
    all_sequences.update(sampled_seqs)

    n_total = len(all_site_ids)
    n_pos = int((all_labels == 1).sum())
    n_neg = int((all_labels == 0).sum())
    logger.info(f"\nCombined dataset: {n_total} sites ({n_pos} pos, {n_neg} neg)")
    logger.info(f"  Original negatives: {n_orig_neg}")
    logger.info(f"  Structure-negatives added: {len(struct_neg)}")

    # 7. Build RNAsee features
    logger.info("Building RNAsee 50-bit features...")
    rnasee_feats = np.zeros((n_total, 50), dtype=np.float32)
    for i, sid in enumerate(all_site_ids):
        seq = all_sequences.get(str(sid), "N" * 201)
        rnasee_feats[i] = encode_rnasee(seq)

    # 8. Build hand features for existing sites from caches
    logger.info("Building hand features from caches...")
    struct_cache = dict(np.load(STRUCT_CACHE, allow_pickle=True))
    cache_sids = np.array([str(s) for s in struct_cache["site_ids"]])
    cache_sid_to_idx = {sid: i for i, sid in enumerate(cache_sids)}

    loop_df = pd.read_csv(LOOP_CSV)
    loop_sid_to_row = {str(r["site_id"]): r for _, r in loop_df.iterrows()}

    hand_40 = np.zeros((n_total, 40), dtype=np.float32)
    hand_46 = np.zeros((n_total, 46), dtype=np.float32)

    n_existing = len(splits)
    n_cache_hit = 0
    for i in range(n_existing):
        sid = str(all_site_ids[i])
        seq = all_sequences.get(sid, "N" * 201)
        motif = encode_motif(seq)

        struct_delta = np.zeros(7, dtype=np.float32)
        baseline = np.zeros(6, dtype=np.float32)
        if sid in cache_sid_to_idx:
            ci = cache_sid_to_idx[sid]
            struct_delta = struct_cache["delta_features"][ci].astype(np.float32)
            LOCAL = 10
            sl = slice(CENTER - LOCAL, CENTER + LOCAL + 1)
            pp = struct_cache["pairing_probs"][ci]
            acc = struct_cache["accessibilities"][ci]
            ent = struct_cache["entropies"][ci]
            baseline = np.array([
                pp[CENTER], acc[CENTER], ent[CENTER],
                np.mean(pp[sl]), np.mean(acc[sl]), np.mean(ent[sl]),
            ], dtype=np.float32)
            n_cache_hit += 1

        loop = np.zeros(9, dtype=np.float32)
        if sid in loop_sid_to_row:
            lr = loop_sid_to_row[sid]
            loop = np.array([
                lr.get("loop_size", 0) or 0,
                lr.get("dist_to_apex", 0) or 0,
                lr.get("relative_loop_position", 0) or 0,
                1.0 if lr.get("is_unpaired", False) else 0.0,
                lr.get("local_unpaired_fraction", 0) or 0,
                lr.get("left_stem_length", 0) or 0,
                lr.get("right_stem_length", 0) or 0,
                lr.get("max_adjacent_stem_length", 0) or 0,
                lr.get("dist_to_junction", 0) or 0,
            ], dtype=np.float32)

        hand_40[i] = np.concatenate([motif, struct_delta, loop])
        hand_46[i] = np.concatenate([motif, struct_delta, loop, baseline])

    logger.info(f"  Existing: {n_cache_hit}/{n_existing} cache hits")

    # 9. Compute ViennaRNA for new structure-negatives
    n_new = len(struct_neg)
    logger.info(f"Computing ViennaRNA features for {n_new} new sites...")
    t_vienna = time.time()
    for j, (_, row) in enumerate(struct_neg.iterrows()):
        i = n_existing + j
        sid = str(row["site_id"])
        seq = sampled_seqs[sid]
        motif = encode_motif(seq)

        try:
            struct_delta, loop_feats, baseline = compute_vienna_features(seq)
        except Exception as e:
            logger.warning(f"  ViennaRNA failed for {sid}: {e}")
            struct_delta = np.zeros(7, dtype=np.float32)
            loop_feats = np.zeros(9, dtype=np.float32)
            baseline = np.zeros(6, dtype=np.float32)

        hand_40[i] = np.concatenate([motif, struct_delta, loop_feats])
        hand_46[i] = np.concatenate([motif, struct_delta, loop_feats, baseline])

        if (j + 1) % 500 == 0:
            elapsed = time.time() - t_vienna
            logger.info(f"  {j+1}/{n_new} ({(j+1)/elapsed:.1f} sites/sec)")

    logger.info(f"  ViennaRNA done in {time.time() - t_vienna:.0f}s")

    hand_40 = np.nan_to_num(hand_40, nan=0.0)
    hand_46 = np.nan_to_num(hand_46, nan=0.0)
    rnasee_feats = np.nan_to_num(rnasee_feats, nan=0.0)

    # 10. 5-fold CV
    logger.info("\n" + "=" * 60)
    logger.info("5-Fold Cross-Validation")
    logger.info("=" * 60)

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    results = {}
    models_config = {
        "GB_HandFeatures": ("gb", hand_40),
        "GB_AllFeatures": ("gb", hand_46),
        "RNAsee_RF": ("rf", rnasee_feats),
    }

    for model_name, (model_type, X) in models_config.items():
        logger.info(f"\n--- {model_name} ---")
        fold_results = []
        for fold_i, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, y_train = X[train_idx], all_labels[train_idx]
            X_test, y_test = X[test_idx], all_labels[test_idx]

            if model_type == "gb":
                clf = GradientBoostingClassifier(
                    n_estimators=300, max_depth=5, learning_rate=0.1,
                    subsample=0.8, random_state=SEED,
                )
            else:
                clf = RandomForestClassifier(
                    n_estimators=500, max_depth=15, random_state=SEED, n_jobs=-1,
                )
            clf.fit(X_train, y_train)
            y_prob = clf.predict_proba(X_test)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)

            fold_results.append({
                "auroc": roc_auc_score(y_test, y_prob),
                "auprc": average_precision_score(y_test, y_prob),
                "f1": f1_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
            })
            logger.info(f"  Fold {fold_i+1}: AUROC={fold_results[-1]['auroc']:.4f}  "
                         f"AUPRC={fold_results[-1]['auprc']:.4f}  "
                         f"F1={fold_results[-1]['f1']:.4f}")

        mean_r = {k: np.mean([r[k] for r in fold_results]) for k in fold_results[0]}
        std_r = {k: np.std([r[k] for r in fold_results]) for k in fold_results[0]}
        results[model_name] = {
            "fold_results": fold_results,
            "mean_auroc": mean_r["auroc"], "std_auroc": std_r["auroc"],
            "mean_auprc": mean_r["auprc"],
            "mean_f1": mean_r["f1"], "std_f1": std_r["f1"],
            "mean_precision": mean_r["precision"],
            "mean_recall": mean_r["recall"],
        }
        logger.info(f"  Mean: AUROC={mean_r['auroc']:.4f}+/-{std_r['auroc']:.4f}  "
                     f"AUPRC={mean_r['auprc']:.4f}  F1={mean_r['f1']:.4f}")

    # 11. Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Dataset: {n_total} ({n_pos} pos, {n_neg} neg)")
    logger.info(f"  Original neg: {n_orig_neg}  |  Struct-neg added: {len(struct_neg)}")
    logger.info("")
    logger.info(f"{'Model':<20} {'AUROC':>15} {'AUPRC':>8} {'F1':>15}")
    logger.info("-" * 62)
    for name, r in results.items():
        logger.info(f"{name:<20} {r['mean_auroc']:.4f}+/-{r['std_auroc']:.4f}"
                     f"  {r['mean_auprc']:.4f}"
                     f"  {r['mean_f1']:.4f}+/-{r['std_f1']:.4f}")

    out = {
        "description": "Structure-negative tier: GB vs RNAsee_RF with added stem/paired negatives",
        "n_total": n_total, "n_positive": n_pos, "n_negative": n_neg,
        "n_orig_negative": n_orig_neg, "n_struct_negative": len(struct_neg),
        "models": results,
    }
    out_path = OUTPUT_DIR / "struct_neg_comparison_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    logger.info(f"\nSaved to {out_path}")
    logger.info(f"Total time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
