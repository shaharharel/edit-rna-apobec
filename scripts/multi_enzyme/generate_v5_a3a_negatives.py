#!/usr/bin/env python
"""Generate motif-matched A3A negatives for V5, compute features, rebuild V5, rerun screen.

The problem: V5's A3A negatives (tier2/tier3) are not motif/loop matched.
A3A positives are 68% in loops, negatives only 28%. This makes classification
trivially easy (AUROC=0.996).

Pipeline:
  Step 1: Generate motif-matched A3A negatives using generate_negatives_from_genome()
  Step 2: Compute ViennaRNA structures (dot-bracket, MFE, delta features, loop geometry)
  Step 3: Compute RNA-FM embeddings (original + edited) for new negatives
  Step 4: Rebuild V5 CSV/JSON/loop/structure files with new negatives
  Step 5: Rerun A6+A8 screen (4 combos: A6/A8 x baseline/pretext, 2-fold CV)

Run: /opt/miniconda3/envs/quris/bin/python -u scripts/multi_enzyme/generate_v5_a3a_negatives.py

Output:
  data/processed/multi_enzyme/splits_multi_enzyme_v5_matched.csv
  data/processed/multi_enzyme/multi_enzyme_sequences_v5_matched.json
  data/processed/multi_enzyme/loop_position_per_site_v5_matched.csv
  data/processed/multi_enzyme/structure_cache_v5_matched.npz
  data/processed/multi_enzyme/embeddings/rnafm_pooled_v5_matched.pt
  data/processed/multi_enzyme/embeddings/rnafm_pooled_edited_v5_matched.pt
  experiments/multi_enzyme/outputs/architecture_screen/v5_matched_screen_results.json
"""

import gc
import json
import logging
import math
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.apobec_negatives import generate_negatives_from_genome
from data.apobec_feature_extraction import build_hand_features


# ---------------------------------------------------------------------------
# ViennaRNA workers (must be at module level for pickling)
# ---------------------------------------------------------------------------

def _fold_and_bpp_worker_toplevel(args):
    """Worker for parallel folding: returns (site_id, 41x41 submatrix)."""
    import RNA
    site_id, seq = args
    try:
        seq = seq.upper().replace("T", "U")
        nn = len(seq)
        md = RNA.md()
        md.temperature = 37.0
        fc = RNA.fold_compound(seq, md)
        fc.mfe()
        fc.pf()
        bpp_raw = np.array(fc.bpp())
        bpp = bpp_raw[1:nn + 1, 1:nn + 1].astype(np.float32)
        sub = bpp[80:121, 80:121].copy()
        for r in range(41):
            for c in range(41):
                if abs(r - c) < 3:
                    sub[r, c] = 0.0
        return site_id, sub
    except Exception:
        return site_id, np.zeros((41, 41), dtype=np.float32)


def _fold_db_worker_toplevel(args):
    """Worker: returns (site_id, dot_bracket_string)."""
    import RNA
    site_id, seq = args
    try:
        seq = seq.upper().replace("T", "U")
        fc = RNA.fold_compound(seq)
        struct, _ = fc.mfe()
        return site_id, struct
    except Exception:
        return site_id, "." * len(seq)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "multi_enzyme"
EMB_DIR = DATA_DIR / "embeddings"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "multi_enzyme" / "outputs" / "architecture_screen"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GENOME_FA = PROJECT_ROOT / "data" / "raw" / "genomes" / "hg38.fa"
V5_SPLITS = DATA_DIR / "splits_multi_enzyme_v5.csv"
V5_SEQS = DATA_DIR / "multi_enzyme_sequences_v5.json"

# Output paths for rebuilt V5
V5M_SPLITS = DATA_DIR / "splits_multi_enzyme_v5_matched.csv"
V5M_SEQS = DATA_DIR / "multi_enzyme_sequences_v5_matched.json"
V5M_LOOP = DATA_DIR / "loop_position_per_site_v5_matched.csv"
V5M_STRUCT = DATA_DIR / "structure_cache_v5_matched.npz"
V5M_RNAFM = EMB_DIR / "rnafm_pooled_v5_matched.pt"
V5M_RNAFM_ED = EMB_DIR / "rnafm_pooled_edited_v5_matched.pt"

CENTER = 100
SEED = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(OUTPUT_DIR / "v5_matched_pipeline.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)


# ===========================================================================
# Step 1: Generate motif-matched A3A negatives
# ===========================================================================

def step1_generate_negatives() -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Generate motif-matched negatives for A3A positives in V5."""
    logger.info("=" * 70)
    logger.info("STEP 1: Generate motif-matched A3A negatives")
    logger.info("=" * 70)

    # Load V5
    v5 = pd.read_csv(V5_SPLITS)
    with open(V5_SEQS) as f:
        v5_seqs = json.load(f)

    # Extract A3A positives
    a3a_pos = v5[(v5["enzyme"] == "A3A") & (v5["is_edited"] == 1)].copy()
    a3a_neg = v5[(v5["enzyme"] == "A3A") & (v5["is_edited"] == 0)].copy()
    logger.info(f"A3A positives: {len(a3a_pos)}, current negatives: {len(a3a_neg)}")

    # Compute TC and CC fractions from positives
    tc_count = 0
    cc_count = 0
    for _, row in a3a_pos.iterrows():
        sid = row["site_id"]
        if sid in v5_seqs:
            seq = v5_seqs[sid]
            upstream = seq[CENTER - 1] if CENTER > 0 else "N"
            if upstream == "U":
                tc_count += 1
            elif upstream == "C":
                cc_count += 1

    tc_frac = tc_count / len(a3a_pos)
    cc_frac = cc_count / len(a3a_pos)
    logger.info(f"A3A positive motif fractions: TC={tc_frac:.3f}, CC={cc_frac:.3f}")

    # Build known sites set (all positions in V5 to exclude)
    known_sites = set()
    for _, row in v5.iterrows():
        if pd.notna(row["chr"]) and pd.notna(row["start"]):
            known_sites.add((str(row["chr"]), int(row["start"])))
    logger.info(f"Known sites to exclude: {len(known_sites)}")

    # Generate negatives
    new_seqs: Dict[str, str] = {}
    n_target = len(a3a_pos)  # 1:1 ratio
    logger.info(f"Generating {n_target} motif-matched negatives (TC={tc_frac:.2f}, CC={cc_frac:.2f})...")

    neg_df = generate_negatives_from_genome(
        positives_df=a3a_pos,
        genome_fa=GENOME_FA,
        target_tc_fraction=tc_frac,
        target_cc_fraction=cc_frac,
        n_negatives=n_target,
        output_seqs=new_seqs,
        known_sites=known_sites,
        search_window=5000,
        seed=SEED,
    )

    logger.info(f"Generated {len(neg_df)} motif-matched negatives with {len(new_seqs)} sequences")

    # Verify motif fractions
    if len(neg_df) > 0:
        tc_neg = sum(1 for sid in neg_df["site_id"] if sid in new_seqs and new_seqs[sid][CENTER - 1] == "U")
        cc_neg = sum(1 for sid in neg_df["site_id"] if sid in new_seqs and new_seqs[sid][CENTER - 1] == "C")
        logger.info(f"New negatives: TC={tc_neg}/{len(neg_df)} ({100*tc_neg/len(neg_df):.1f}%), "
                     f"CC={cc_neg}/{len(neg_df)} ({100*cc_neg/len(neg_df):.1f}%)")

    # Save intermediate
    neg_df.to_csv(DATA_DIR / "a3a_matched_negatives_intermediate.csv", index=False)
    with open(DATA_DIR / "a3a_matched_negatives_seqs_intermediate.json", "w") as f:
        json.dump(new_seqs, f)
    logger.info("Saved intermediate negatives")

    return neg_df, new_seqs


# ===========================================================================
# Step 2: Compute ViennaRNA structures for new negatives
# ===========================================================================

def _vienna_worker(args):
    """Worker: compute ViennaRNA structure features for one sequence.

    Returns: (site_id, dot_bracket, mfe_wt, mfe_ed, delta_features_7dim, loop_features_dict)
    """
    import RNA
    site_id, seq = args
    seq = seq.upper().replace("T", "U")
    n = len(seq)
    center = 100

    try:
        # --- Wildtype fold ---
        fc = RNA.fold_compound(seq)
        struct_wt, mfe_wt = fc.mfe()

        # Partition function for base-pair probabilities
        md = RNA.md()
        md.temperature = 37.0
        fc_pf = RNA.fold_compound(seq, md)
        fc_pf.mfe()
        fc_pf.pf()
        bpp_raw = np.array(fc_pf.bpp())
        bpp = bpp_raw[1:n+1, 1:n+1].astype(np.float32)

        # Pairing probability at center
        pairing_wt = bpp[center, :].sum()
        # Accessibility: probability of being unpaired
        accessibility_wt = 1.0 - pairing_wt

        # Entropy at center position
        probs = bpp[center, :]
        probs = probs[probs > 0]
        entropy_wt = -np.sum(probs * np.log2(probs + 1e-10))

        # Window features (positions 90-110)
        win_start, win_end = max(0, center - 10), min(n, center + 11)
        pairing_window_wt = np.array([bpp[j, :].sum() for j in range(win_start, win_end)])

        # --- Edited fold (C->U at center) ---
        seq_ed = list(seq)
        seq_ed[center] = "U"
        seq_ed = "".join(seq_ed)

        fc_ed = RNA.fold_compound(seq_ed)
        _, mfe_ed = fc_ed.mfe()

        md_ed = RNA.md()
        md_ed.temperature = 37.0
        fc_pf_ed = RNA.fold_compound(seq_ed, md_ed)
        fc_pf_ed.mfe()
        fc_pf_ed.pf()
        bpp_raw_ed = np.array(fc_pf_ed.bpp())
        bpp_ed = bpp_raw_ed[1:n+1, 1:n+1].astype(np.float32)

        pairing_ed = bpp_ed[center, :].sum()
        accessibility_ed = 1.0 - pairing_ed
        probs_ed = bpp_ed[center, :]
        probs_ed = probs_ed[probs_ed > 0]
        entropy_ed = -np.sum(probs_ed * np.log2(probs_ed + 1e-10))
        pairing_window_ed = np.array([bpp_ed[j, :].sum() for j in range(win_start, win_end)])

        # --- Delta features (7-dim) ---
        delta_pairing_center = pairing_ed - pairing_wt
        delta_accessibility_center = accessibility_ed - accessibility_wt
        delta_entropy_center = entropy_ed - entropy_wt
        delta_mfe = mfe_ed - mfe_wt
        delta_pairing_window = pairing_window_ed - pairing_window_wt
        mean_delta_pairing_window = float(np.mean(delta_pairing_window))
        std_delta_pairing_window = float(np.std(delta_pairing_window))
        mean_delta_accessibility_window = float(np.mean(1.0 - pairing_window_ed) - np.mean(1.0 - pairing_window_wt))

        delta_features = np.array([
            delta_pairing_center,
            delta_accessibility_center,
            delta_entropy_center,
            delta_mfe,
            mean_delta_pairing_window,
            std_delta_pairing_window,
            mean_delta_accessibility_window,
        ], dtype=np.float32)

        # --- Loop geometry features ---
        loop_feats = _extract_loop_geometry(struct_wt, center)
        loop_feats["mfe"] = mfe_wt
        loop_feats["dot_bracket"] = struct_wt

        # --- BP submatrix (41x41) ---
        sub = bpp[80:121, 80:121].copy()
        for r in range(41):
            for c in range(41):
                if abs(r - c) < 3:
                    sub[r, c] = 0.0

        return site_id, struct_wt, mfe_wt, mfe_ed, delta_features, loop_feats, sub

    except Exception as e:
        return site_id, "." * n, 0.0, 0.0, np.zeros(7, dtype=np.float32), _empty_loop_features(), np.zeros((41, 41), dtype=np.float32)


def _extract_loop_geometry(struct: str, center: int) -> dict:
    """Extract loop geometry features from dot-bracket structure."""
    n = len(struct)

    # Check if center is unpaired
    is_unpaired = 1.0 if struct[center] == "." else 0.0

    if is_unpaired == 0.0:
        return {
            "is_unpaired": 0.0,
            "loop_type": None,
            "loop_size": 0.0,
            "dist_to_left_boundary": 0.0,
            "dist_to_right_boundary": 0.0,
            "dist_to_nearest_stem": 0.0,
            "relative_loop_position": 0.0,
            "dist_to_apex": 0.0,
            "left_stem_length": 0.0,
            "right_stem_length": 0.0,
            "max_adjacent_stem_length": 0.0,
            "dist_to_junction": 0.0,
            "local_unpaired_fraction": 0.0,
        }

    # Find loop boundaries (nearest paired bases left and right)
    left_boundary = center
    while left_boundary > 0 and struct[left_boundary - 1] == ".":
        left_boundary -= 1

    right_boundary = center
    while right_boundary < n - 1 and struct[right_boundary + 1] == ".":
        right_boundary += 1

    loop_size = right_boundary - left_boundary + 1
    dist_to_left = center - left_boundary
    dist_to_right = right_boundary - center

    # Relative position in loop (0 = left edge, 1 = right edge)
    relative_pos = dist_to_left / max(loop_size - 1, 1)

    # Distance to apex (middle of loop)
    apex = (left_boundary + right_boundary) / 2.0
    dist_to_apex = abs(center - apex) / max(loop_size / 2, 1)

    # Determine loop type
    if left_boundary == 0 or right_boundary == n - 1:
        loop_type = "external"
    elif left_boundary > 0 and struct[left_boundary - 1] == "(" and right_boundary < n - 1 and struct[right_boundary + 1] == ")":
        loop_type = "hairpin"
    elif left_boundary > 0 and struct[left_boundary - 1] == ")" and right_boundary < n - 1 and struct[right_boundary + 1] == "(":
        loop_type = "interior"
    else:
        loop_type = "other"

    # Stem lengths
    left_stem = 0
    pos = left_boundary - 1
    while pos >= 0 and struct[pos] in ("(", ")"):
        left_stem += 1
        pos -= 1

    right_stem = 0
    pos = right_boundary + 1
    while pos < n and struct[pos] in ("(", ")"):
        right_stem += 1
        pos += 1

    max_adj_stem = max(left_stem, right_stem)
    dist_to_nearest_stem = min(dist_to_left, dist_to_right)

    # Local unpaired fraction (window of 21 around center)
    local_start = max(0, center - 10)
    local_end = min(n, center + 11)
    local_struct = struct[local_start:local_end]
    local_unpaired = sum(1 for c in local_struct if c == ".") / len(local_struct)

    # Distance to junction (nearest multi-loop junction)
    dist_to_junction = min(left_boundary, n - 1 - right_boundary)

    return {
        "is_unpaired": is_unpaired,
        "loop_type": loop_type,
        "loop_size": float(loop_size),
        "dist_to_left_boundary": float(dist_to_left),
        "dist_to_right_boundary": float(dist_to_right),
        "dist_to_nearest_stem": float(dist_to_nearest_stem),
        "relative_loop_position": float(relative_pos),
        "dist_to_apex": float(dist_to_apex),
        "left_stem_length": float(left_stem),
        "right_stem_length": float(right_stem),
        "max_adjacent_stem_length": float(max_adj_stem),
        "dist_to_junction": float(dist_to_junction),
        "local_unpaired_fraction": float(local_unpaired),
    }


def _empty_loop_features() -> dict:
    return {
        "is_unpaired": 0.0,
        "loop_type": None,
        "loop_size": 0.0,
        "dist_to_left_boundary": 0.0,
        "dist_to_right_boundary": 0.0,
        "dist_to_nearest_stem": 0.0,
        "relative_loop_position": 0.0,
        "dist_to_apex": 0.0,
        "left_stem_length": 0.0,
        "right_stem_length": 0.0,
        "max_adjacent_stem_length": 0.0,
        "dist_to_junction": 0.0,
        "local_unpaired_fraction": 0.0,
        "mfe": 0.0,
        "dot_bracket": None,
    }


def step2_compute_structures(neg_df: pd.DataFrame, new_seqs: Dict[str, str]) -> Tuple[dict, dict, dict]:
    """Compute ViennaRNA structures and features for new negatives.

    Returns:
        structure_delta: {site_id: 7-dim array}
        loop_records: list of dicts for loop CSV
        bp_submatrices: {site_id: 41x41 array}
    """
    logger.info("=" * 70)
    logger.info("STEP 2: Compute ViennaRNA structures for new negatives")
    logger.info("=" * 70)

    site_ids = neg_df["site_id"].tolist()
    work_items = [(sid, new_seqs[sid]) for sid in site_ids if sid in new_seqs]
    logger.info(f"Computing structures for {len(work_items)} sequences...")

    n_workers = min(14, os.cpu_count() or 4)
    structure_delta = {}
    loop_records = []
    bp_subs = {}
    mfes_wt = {}
    mfes_ed = {}
    dot_brackets = {}

    t0 = time.time()
    done = 0

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_vienna_worker, item): item[0] for item in work_items}
        for future in as_completed(futures):
            sid, db, mfe_wt, mfe_ed, delta, loop_feats, bp_sub = future.result()
            structure_delta[sid] = delta
            mfes_wt[sid] = mfe_wt
            mfes_ed[sid] = mfe_ed
            dot_brackets[sid] = db
            bp_subs[sid] = bp_sub

            # Build loop record
            rec = dict(loop_feats)
            rec["site_id"] = sid
            # Find enzyme and dataset_source from neg_df
            row = neg_df[neg_df["site_id"] == sid]
            if len(row) > 0:
                rec["enzyme"] = row.iloc[0]["enzyme"]
                rec["dataset_source"] = row.iloc[0]["dataset_source"]
                rec["label"] = 0.0
            loop_records.append(rec)

            done += 1
            if done % 1000 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed
                remaining = (len(work_items) - done) / rate if rate > 0 else 0
                logger.info(f"  Folded {done}/{len(work_items)} ({rate:.1f} seq/s, ~{remaining/60:.1f}m remaining)")

    elapsed = time.time() - t0
    logger.info(f"Structure computation complete: {done} sequences in {elapsed:.0f}s ({done/elapsed:.1f} seq/s)")

    # Check loop stats
    n_unpaired = sum(1 for r in loop_records if r.get("is_unpaired", 0) > 0.5)
    logger.info(f"New negatives in loops: {n_unpaired}/{len(loop_records)} ({100*n_unpaired/max(len(loop_records),1):.1f}%)")

    # Save intermediate
    np.savez_compressed(
        DATA_DIR / "a3a_matched_structure_intermediate.npz",
        site_ids=np.array(list(structure_delta.keys()), dtype=object),
        delta_features=np.array([structure_delta[sid] for sid in structure_delta], dtype=np.float32),
        mfes=np.array([mfes_wt[sid] for sid in structure_delta], dtype=np.float32),
        mfes_edited=np.array([mfes_ed[sid] for sid in structure_delta], dtype=np.float32),
    )
    logger.info("Saved intermediate structure cache")

    return structure_delta, loop_records, bp_subs, dot_brackets


# ===========================================================================
# Step 3: Compute RNA-FM embeddings for new negatives
# ===========================================================================

def step3_compute_rnafm(neg_df: pd.DataFrame, new_seqs: Dict[str, str]) -> Tuple[dict, dict]:
    """Compute RNA-FM pooled embeddings (original + edited) for new negatives."""
    import torch
    import fm

    logger.info("=" * 70)
    logger.info("STEP 3: Compute RNA-FM embeddings for new negatives")
    logger.info("=" * 70)

    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Device: {DEVICE}")

    # Load model
    model, alphabet = fm.pretrained.rna_fm_t12()
    model = model.eval().to(DEVICE)
    batch_converter = alphabet.get_batch_converter()
    logger.info("RNA-FM loaded")

    site_ids = [sid for sid in neg_df["site_id"].tolist() if sid in new_seqs]
    sequences = [new_seqs[sid].upper().replace("T", "U") for sid in site_ids]
    logger.info(f"Computing embeddings for {len(site_ids)} sequences...")

    pooled_dict = {}
    pooled_ed_dict = {}
    batch_size = 16

    t0 = time.time()
    for i in range(0, len(site_ids), batch_size):
        batch_sids = site_ids[i:i + batch_size]
        batch_seqs = sequences[i:i + batch_size]

        # Original
        data = [(f"seq_{j}", seq) for j, seq in enumerate(batch_seqs)]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(DEVICE)
        with torch.no_grad():
            results = model(tokens, repr_layers=[12])
        emb = results["representations"][12][:, 1:-1, :].mean(dim=1).cpu()

        # Edited (C->U at center)
        ed_seqs = []
        for seq in batch_seqs:
            seq_list = list(seq)
            if seq_list[CENTER] == "C":
                seq_list[CENTER] = "U"
            ed_seqs.append("".join(seq_list))

        data_ed = [(f"seq_{j}", seq) for j, seq in enumerate(ed_seqs)]
        _, _, tokens_ed = batch_converter(data_ed)
        tokens_ed = tokens_ed.to(DEVICE)
        with torch.no_grad():
            results_ed = model(tokens_ed, repr_layers=[12])
        emb_ed = results_ed["representations"][12][:, 1:-1, :].mean(dim=1).cpu()

        for k, sid in enumerate(batch_sids):
            pooled_dict[sid] = emb[k]
            pooled_ed_dict[sid] = emb_ed[k]

        done = min(i + batch_size, len(site_ids))
        if done % (batch_size * 50) < batch_size or done == len(site_ids):
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            remaining = (len(site_ids) - done) / rate if rate > 0 else 0
            logger.info(f"  {done}/{len(site_ids)} ({rate:.1f} seq/s, ~{remaining/60:.1f}m remaining)")

    elapsed = time.time() - t0
    logger.info(f"RNA-FM complete: {len(pooled_dict)} embeddings in {elapsed:.0f}s")

    # Save intermediate
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(pooled_dict, EMB_DIR / "rnafm_a3a_matched_neg_intermediate.pt")
    torch.save(pooled_ed_dict, EMB_DIR / "rnafm_a3a_matched_neg_edited_intermediate.pt")
    logger.info("Saved intermediate RNA-FM embeddings")

    # Clean up GPU memory
    del model
    gc.collect()
    if DEVICE == "mps":
        torch.mps.empty_cache()

    return pooled_dict, pooled_ed_dict


# ===========================================================================
# Step 4: Rebuild V5 with matched negatives
# ===========================================================================

def step4_rebuild_v5(neg_df: pd.DataFrame, new_seqs: Dict[str, str],
                     structure_delta: dict, loop_records: list,
                     bp_subs: dict, dot_brackets: dict,
                     rnafm_pooled: dict, rnafm_pooled_ed: dict):
    """Replace old A3A tier2/tier3 negatives with new motif-matched ones."""
    import torch

    logger.info("=" * 70)
    logger.info("STEP 4: Rebuild V5 with motif-matched A3A negatives")
    logger.info("=" * 70)

    # Load original V5
    v5 = pd.read_csv(V5_SPLITS)
    with open(V5_SEQS) as f:
        v5_seqs = json.load(f)

    # Remove old A3A negatives (tier2/tier3)
    old_a3a_neg_mask = (v5["enzyme"] == "A3A") & (v5["is_edited"] == 0)
    old_neg_ids = set(v5[old_a3a_neg_mask]["site_id"])
    logger.info(f"Removing {old_a3a_neg_mask.sum()} old A3A tier2/tier3 negatives")

    v5_clean = v5[~old_a3a_neg_mask].copy()
    logger.info(f"V5 after removing old negatives: {len(v5_clean)}")

    # Remove old negative sequences
    for sid in old_neg_ids:
        v5_seqs.pop(sid, None)

    # Add new negatives
    v5_matched = pd.concat([v5_clean, neg_df], ignore_index=True)
    logger.info(f"V5 matched total: {len(v5_matched)}")

    # Add new sequences
    v5_seqs.update(new_seqs)
    logger.info(f"Total sequences: {len(v5_seqs)}")

    # Verify counts
    for enz in v5_matched["enzyme"].unique():
        pos = (v5_matched["enzyme"] == enz) & (v5_matched["is_edited"] == 1)
        neg = (v5_matched["enzyme"] == enz) & (v5_matched["is_edited"] == 0)
        logger.info(f"  {enz}: {pos.sum()} pos, {neg.sum()} neg")

    # Save splits and sequences
    v5_matched.to_csv(V5M_SPLITS, index=False)
    with open(V5M_SEQS, "w") as f:
        json.dump(v5_seqs, f)
    logger.info(f"Saved {V5M_SPLITS.name} and {V5M_SEQS.name}")

    # --- Rebuild loop position CSV ---
    # Load existing loop data (from V3 and V5)
    loop_v3_path = DATA_DIR / "loop_position_per_site_v3.csv"
    loop_v5_path = DATA_DIR / "loop_position_per_site_v5.csv"

    existing_loop = pd.DataFrame()
    for lp in [loop_v3_path, loop_v5_path]:
        if lp.exists():
            tmp = pd.read_csv(lp)
            existing_loop = pd.concat([existing_loop, tmp], ignore_index=True)
    existing_loop = existing_loop.drop_duplicates(subset=["site_id"], keep="last")

    # Filter to non-A3A-neg site_ids + add new loop records
    keep_sids = set(v5_matched["site_id"]) - set(neg_df["site_id"])
    existing_keep = existing_loop[existing_loop["site_id"].isin(keep_sids)]

    new_loop_df = pd.DataFrame(loop_records)
    combined_loop = pd.concat([existing_keep, new_loop_df], ignore_index=True)
    combined_loop = combined_loop.drop_duplicates(subset=["site_id"], keep="last")
    combined_loop.to_csv(V5M_LOOP, index=False)
    logger.info(f"Saved loop positions: {len(combined_loop)} rows to {V5M_LOOP.name}")

    # --- Rebuild structure cache ---
    # Load existing caches
    struct_v3_path = DATA_DIR / "structure_cache_multi_enzyme_v3.npz"
    existing_struct = {}
    existing_mfes = {}
    existing_mfes_ed = {}
    if struct_v3_path.exists():
        sd = np.load(struct_v3_path, allow_pickle=True)
        for i, sid in enumerate(sd["site_ids"]):
            existing_struct[str(sid)] = sd["delta_features"][i]
            existing_mfes[str(sid)] = sd["mfes"][i] if "mfes" in sd else 0.0
            existing_mfes_ed[str(sid)] = sd["mfes_edited"][i] if "mfes_edited" in sd else 0.0

    # A3A structure cache
    a3a_struct_path = PROJECT_ROOT / "data" / "processed" / "embeddings" / "vienna_structure_cache.npz"
    if a3a_struct_path.exists():
        sd = np.load(a3a_struct_path, allow_pickle=True)
        for i, sid in enumerate(sd["site_ids"]):
            sid = str(sid)
            if sid not in existing_struct:
                existing_struct[sid] = sd["delta_features"][i]
                existing_mfes[sid] = sd["mfes"][i] if "mfes" in sd else 0.0
                existing_mfes_ed[sid] = sd["mfes_edited"][i] if "mfes_edited" in sd else 0.0

    # Add new structure data
    for sid, delta in structure_delta.items():
        existing_struct[sid] = delta

    # Filter to V5 matched site_ids only
    v5m_sids = set(v5_matched["site_id"])
    all_sids = [sid for sid in v5m_sids if sid in existing_struct]
    all_deltas = np.array([existing_struct[sid] for sid in all_sids], dtype=np.float32)
    all_mfes_arr = np.array([existing_mfes.get(sid, 0.0) for sid in all_sids], dtype=np.float32)
    all_mfes_ed_arr = np.array([existing_mfes_ed.get(sid, 0.0) for sid in all_sids], dtype=np.float32)

    np.savez_compressed(V5M_STRUCT,
                        site_ids=np.array(all_sids, dtype=object),
                        delta_features=all_deltas,
                        mfes=all_mfes_arr,
                        mfes_edited=all_mfes_ed_arr)
    logger.info(f"Saved structure cache: {len(all_sids)} sites to {V5M_STRUCT.name}")

    # --- Rebuild RNA-FM embeddings ---
    # Load existing V3 embeddings
    v3_pooled_path = EMB_DIR / "rnafm_pooled_v3.pt"
    v3_edited_path = EMB_DIR / "rnafm_pooled_edited_v3.pt"

    all_pooled = {}
    all_pooled_ed = {}
    if v3_pooled_path.exists():
        all_pooled = torch.load(v3_pooled_path, map_location="cpu", weights_only=False)
    if v3_edited_path.exists():
        all_pooled_ed = torch.load(v3_edited_path, map_location="cpu", weights_only=False)
    logger.info(f"Loaded V3 embeddings: {len(all_pooled)} original, {len(all_pooled_ed)} edited")

    # Add new negatives
    for sid, emb in rnafm_pooled.items():
        all_pooled[sid] = emb
    for sid, emb in rnafm_pooled_ed.items():
        all_pooled_ed[sid] = emb
    logger.info(f"Total pooled embeddings: {len(all_pooled)}")

    torch.save(all_pooled, V5M_RNAFM)
    torch.save(all_pooled_ed, V5M_RNAFM_ED)
    logger.info(f"Saved RNA-FM embeddings to {V5M_RNAFM.name} and {V5M_RNAFM_ED.name}")

    return v5_matched, v5_seqs, bp_subs, dot_brackets


# ===========================================================================
# Step 5: Rerun A6+A8 screen on fixed V5
# ===========================================================================

def step5_rerun_screen(v5_matched: pd.DataFrame, v5_seqs: Dict[str, str],
                       new_bp_subs: dict, new_dot_brackets: dict):
    """Rerun A6 and A8 architectures with baseline and pretext on fixed V5."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from torch.utils.data import DataLoader, Dataset

    logger.info("=" * 70)
    logger.info("STEP 5: Rerun A6+A8 screen on fixed V5 (motif-matched negatives)")
    logger.info("=" * 70)

    DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    D_RNAFM = 640
    D_EDIT_DELTA = 640
    D_HAND = 40
    D_SHARED = 128
    ENZYME_CLASSES = ["A3A", "A3B", "A3G", "A3A_A3G", "Neither", "Unknown"]
    ENZYME_TO_IDX = {e: i for i, e in enumerate(ENZYME_CLASSES)}
    N_ENZYMES = len(ENZYME_CLASSES)
    PER_ENZYME_HEADS = ["A3A", "A3B", "A3G", "A3A_A3G", "Neither"]
    N_PER_ENZYME = len(PER_ENZYME_HEADS)

    # --- Load all features ---
    df = v5_matched.copy()
    site_ids = df["site_id"].tolist()
    n = len(site_ids)
    logger.info(f"Dataset: {n} sites")

    # --- Loop geometry ---
    loop_df = pd.read_csv(V5M_LOOP)
    loop_df = loop_df.drop_duplicates(subset=["site_id"]).set_index("site_id")
    n_loop = sum(1 for sid in site_ids if sid in loop_df.index)
    logger.info(f"  Loop coverage: {n_loop}/{n}")

    # --- Structure delta ---
    struct_data = np.load(V5M_STRUCT, allow_pickle=True)
    structure_delta = {str(sid): struct_data["delta_features"][i]
                       for i, sid in enumerate(struct_data["site_ids"])}
    n_struct = sum(1 for sid in site_ids if sid in structure_delta)
    logger.info(f"  Structure delta coverage: {n_struct}/{n}")

    # --- Hand features (40-dim) ---
    hand_features = build_hand_features(site_ids, v5_seqs, structure_delta, loop_df)
    logger.info(f"  Hand features shape: {hand_features.shape}")

    # --- RNA-FM embeddings ---
    rnafm_emb = torch.load(V5M_RNAFM, map_location="cpu", weights_only=False)
    rnafm_edited_emb = torch.load(V5M_RNAFM_ED, map_location="cpu", weights_only=False)

    rnafm_matrix = np.zeros((n, D_RNAFM), dtype=np.float32)
    rnafm_edited_matrix = np.zeros((n, D_RNAFM), dtype=np.float32)
    n_rnafm = 0
    for i, sid in enumerate(site_ids):
        if sid in rnafm_emb:
            rnafm_matrix[i] = rnafm_emb[sid].numpy()
            n_rnafm += 1
        if sid in rnafm_edited_emb:
            rnafm_edited_matrix[i] = rnafm_edited_emb[sid].numpy()
    edit_delta_matrix = rnafm_edited_matrix - rnafm_matrix
    logger.info(f"  RNA-FM coverage: {n_rnafm}/{n} ({100*n_rnafm/n:.1f}%)")

    del rnafm_emb, rnafm_edited_emb
    gc.collect()

    # --- BP submatrices ---
    bp_cache_path = OUTPUT_DIR / "bp_submatrices_v5_matched.npz"
    if bp_cache_path.exists():
        logger.info(f"  Loading cached V5M BP submatrices")
        bp_data = np.load(bp_cache_path, allow_pickle=True)
        bp_submatrices = bp_data["bp_submatrices"]
        bp_site_ids = list(bp_data["site_ids"])
        if bp_site_ids == site_ids:
            pass
        else:
            bp_sid_map = {sid: i for i, sid in enumerate(bp_site_ids)}
            aligned = np.zeros((n, 41, 41), dtype=np.float32)
            for i, sid in enumerate(site_ids):
                if sid in bp_sid_map:
                    aligned[i] = bp_submatrices[bp_sid_map[sid]]
            bp_submatrices = aligned
    else:
        # Load from V3 cache + V5 cache + new
        bp_submatrices = np.zeros((n, 41, 41), dtype=np.float32)
        sid_to_idx = {sid: i for i, sid in enumerate(site_ids)}

        # Load V3 BP cache
        bp_v3_path = OUTPUT_DIR / "bp_submatrices_v3.npz"
        n_cached = 0
        if bp_v3_path.exists():
            bp_v3 = np.load(bp_v3_path, allow_pickle=True)
            bp_v3_ids = list(bp_v3["site_ids"])
            bp_v3_data = bp_v3["bp_submatrices"]
            bp_v3_map = {sid: i for i, sid in enumerate(bp_v3_ids)}
            for i, sid in enumerate(site_ids):
                if sid in bp_v3_map:
                    bp_submatrices[i] = bp_v3_data[bp_v3_map[sid]]
                    n_cached += 1
            del bp_v3, bp_v3_data
            gc.collect()

        # Load V5 BP cache
        bp_v5_path = OUTPUT_DIR / "bp_submatrices_v5.npz"
        if bp_v5_path.exists():
            bp_v5 = np.load(bp_v5_path, allow_pickle=True)
            bp_v5_ids = list(bp_v5["site_ids"])
            bp_v5_data = bp_v5["bp_submatrices"]
            bp_v5_map = {sid: i for i, sid in enumerate(bp_v5_ids)}
            for i, sid in enumerate(site_ids):
                if sid in bp_v5_map and bp_submatrices[i].sum() == 0:
                    bp_submatrices[i] = bp_v5_data[bp_v5_map[sid]]
                    n_cached += 1
            del bp_v5, bp_v5_data
            gc.collect()

        logger.info(f"  BP from cache: {n_cached}/{n}")

        # Add new BP submatrices
        for sid, sub in new_bp_subs.items():
            if sid in sid_to_idx:
                bp_submatrices[sid_to_idx[sid]] = sub

        # Check for missing and compute
        missing_bp = [(sid, v5_seqs[sid]) for i, sid in enumerate(site_ids)
                      if bp_submatrices[i].sum() == 0 and sid in v5_seqs]
        if missing_bp:
            logger.info(f"  Computing BP for {len(missing_bp)} missing sites...")
            n_workers = min(14, os.cpu_count() or 4)
            t_bp = time.time()
            done_bp = 0
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {executor.submit(_fold_and_bpp_worker_toplevel, item): item[0] for item in missing_bp}
                for future in as_completed(futures):
                    sid, sub = future.result()
                    if sid in sid_to_idx:
                        bp_submatrices[sid_to_idx[sid]] = sub
                    done_bp += 1
                    if done_bp % 1000 == 0:
                        logger.info(f"    BP computed: {done_bp}/{len(missing_bp)} ({done_bp/(time.time()-t_bp):.1f} seq/s)")
            logger.info(f"    BP done: {len(missing_bp)} in {time.time()-t_bp:.0f}s")

        # Save
        np.savez_compressed(bp_cache_path,
                            bp_submatrices=bp_submatrices,
                            site_ids=np.array(site_ids, dtype=object))
        logger.info(f"  Saved BP cache: {bp_cache_path.name}")

    # --- Dot brackets ---
    db_cache_path = OUTPUT_DIR / "dot_brackets_v5_matched.json"
    if db_cache_path.exists():
        with open(db_cache_path) as f:
            dot_brackets_all = json.load(f)
    else:
        # Load from V3 and V5 caches
        dot_brackets_all = {}
        for db_path in [OUTPUT_DIR / "dot_brackets_v3.json", OUTPUT_DIR / "dot_brackets_v5.json"]:
            if db_path.exists():
                with open(db_path) as f:
                    dot_brackets_all.update(json.load(f))
        dot_brackets_all.update(new_dot_brackets)

        # Check for loop_df dot_bracket column
        if "dot_bracket" in loop_df.columns:
            for sid in site_ids:
                if sid not in dot_brackets_all and sid in loop_df.index:
                    db = loop_df.loc[sid, "dot_bracket"]
                    if isinstance(db, str) and len(db) == 201:
                        dot_brackets_all[sid] = db

        # Compute missing
        missing_db = [sid for sid in site_ids if sid not in dot_brackets_all and sid in v5_seqs]
        if missing_db:
            logger.info(f"  Computing dot-bracket for {len(missing_db)} missing sites...")
            work_items = [(sid, v5_seqs[sid]) for sid in missing_db]
            n_workers = min(14, os.cpu_count() or 4)
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {executor.submit(_fold_db_worker_toplevel, item): item[0] for item in work_items}
                for future in as_completed(futures):
                    sid, db = future.result()
                    dot_brackets_all[sid] = db

        with open(db_cache_path, "w") as f:
            json.dump(dot_brackets_all, f)
        logger.info(f"  Saved dot-bracket cache: {db_cache_path.name}")

    # --- Labels ---
    labels_binary = df["is_edited"].values.astype(np.float32)
    labels_enzyme = np.array([ENZYME_TO_IDX.get(e, 5) for e in df["enzyme"].values], dtype=np.int64)

    per_enzyme_labels = np.full((n, N_PER_ENZYME), -1, dtype=np.float32)
    for head_idx, enz_name in enumerate(PER_ENZYME_HEADS):
        enz_idx = ENZYME_TO_IDX[enz_name]
        pos_mask = (labels_binary == 1) & (labels_enzyme == enz_idx)
        per_enzyme_labels[pos_mask, head_idx] = 1.0
        neg_mask = (labels_binary == 0) & (labels_enzyme == enz_idx)
        per_enzyme_labels[neg_mask, head_idx] = 0.0

    for head_idx, enz_name in enumerate(PER_ENZYME_HEADS):
        n_pos = int((per_enzyme_labels[:, head_idx] == 1).sum())
        n_neg = int((per_enzyme_labels[:, head_idx] == 0).sum())
        logger.info(f"  {enz_name}: {n_pos} pos, {n_neg} neg")

    data = {
        "site_ids": site_ids,
        "df": df,
        "sequences": v5_seqs,
        "hand_features": hand_features,
        "rnafm_features": rnafm_matrix,
        "edit_delta_features": edit_delta_matrix,
        "bp_submatrices": bp_submatrices,
        "dot_brackets": dot_brackets_all,
        "labels_binary": labels_binary,
        "labels_enzyme": labels_enzyme,
        "per_enzyme_labels": per_enzyme_labels,
    }

    # -----------------------------------------------------------------------
    # Define architectures and training (reuse from exp_architecture_screen)
    # -----------------------------------------------------------------------

    class HeadsMixin:
        def _init_heads(self):
            self.binary_head = nn.Linear(D_SHARED, 1)
            self.enzyme_adapters = nn.ModuleDict({
                enz: nn.Sequential(
                    nn.Linear(D_SHARED, 32),
                    nn.GELU(),
                    nn.Linear(32, 1),
                ) for enz in PER_ENZYME_HEADS
            })
            self.enzyme_classifier = nn.Sequential(
                nn.Linear(D_SHARED, 64),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(64, N_ENZYMES),
            )

        def _apply_heads(self, shared):
            binary_logit = self.binary_head(shared).squeeze(-1)
            per_enzyme_logits = [
                self.enzyme_adapters[enz](shared).squeeze(-1) for enz in PER_ENZYME_HEADS
            ]
            enzyme_cls_logits = self.enzyme_classifier(shared)
            return binary_logit, per_enzyme_logits, enzyme_cls_logits

    class Conv2DBP(nn.Module, HeadsMixin):
        name = "A6_Conv2DBP"
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(4),
            )
            self.conv_fc = nn.Linear(32 * 4 * 4, 128)
            d_fused = 128 + D_RNAFM + D_HAND
            self.encoder = nn.Sequential(
                nn.Linear(d_fused, 256), nn.GELU(), nn.Dropout(0.3), nn.LayerNorm(256),
                nn.Linear(256, D_SHARED), nn.GELU(), nn.Dropout(0.2),
            )
            self._init_heads()
        def get_shared_params(self):
            return list(self.conv.parameters()) + list(self.conv_fc.parameters()) + list(self.encoder.parameters())
        def get_adapter_params(self):
            params = list(self.binary_head.parameters())
            for enz in PER_ENZYME_HEADS:
                params.extend(self.enzyme_adapters[enz].parameters())
            params.extend(self.enzyme_classifier.parameters())
            return params
        def forward(self, batch):
            bp = batch["bp_submatrix"]
            conv_out = self.conv(bp).flatten(1)
            conv_out = self.conv_fc(conv_out)
            fused = torch.cat([conv_out, batch["rnafm"], batch["hand_feat"]], dim=-1)
            shared = self.encoder(fused)
            return self._apply_heads(shared)

    class HierarchicalAttention(nn.Module, HeadsMixin):
        name = "A8_HierarchicalAttention"
        def __init__(self):
            super().__init__()
            d_local = 41
            self.local_proj = nn.Linear(d_local, 64)
            self.local_pos_enc = nn.Parameter(torch.randn(41, 64) * 0.02)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=64, nhead=4, dim_feedforward=128,
                dropout=0.1, activation="gelu", batch_first=True,
            )
            self.local_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.local_pool_attn = nn.Linear(64, 1)
            self.cross_q = nn.Linear(64, 64)
            self.cross_k = nn.Linear(D_RNAFM, 64)
            self.cross_v = nn.Linear(D_RNAFM, 64)
            d_fused = 64 + 64 + D_RNAFM + D_HAND
            self.encoder = nn.Sequential(
                nn.Linear(d_fused, 256), nn.GELU(), nn.Dropout(0.3), nn.LayerNorm(256),
                nn.Linear(256, D_SHARED), nn.GELU(), nn.Dropout(0.2),
            )
            self._init_heads()
        def get_shared_params(self):
            return (
                list(self.local_proj.parameters()) + [self.local_pos_enc] +
                list(self.local_transformer.parameters()) +
                list(self.local_pool_attn.parameters()) +
                list(self.cross_q.parameters()) + list(self.cross_k.parameters()) +
                list(self.cross_v.parameters()) + list(self.encoder.parameters())
            )
        def get_adapter_params(self):
            params = list(self.binary_head.parameters())
            for enz in PER_ENZYME_HEADS:
                params.extend(self.enzyme_adapters[enz].parameters())
            params.extend(self.enzyme_classifier.parameters())
            return params
        def forward(self, batch):
            bp = batch["bp_submatrix"].squeeze(1)
            rnafm = batch["rnafm"]
            hand = batch["hand_feat"]
            local_in = self.local_proj(bp) + self.local_pos_enc.unsqueeze(0)
            local_out = self.local_transformer(local_in)
            attn_w = torch.softmax(self.local_pool_attn(local_out), dim=1)
            local_repr = (local_out * attn_w).sum(dim=1)
            q = self.cross_q(local_repr).unsqueeze(1)
            k = self.cross_k(rnafm).unsqueeze(1)
            v = self.cross_v(rnafm).unsqueeze(1)
            attn_scores = (q * k).sum(-1) / math.sqrt(64)
            attn_weights = torch.softmax(attn_scores, dim=-1)
            cross_repr = (attn_weights.unsqueeze(-1) * v).squeeze(1)
            fused = torch.cat([local_repr, cross_repr, rnafm, hand], dim=-1)
            shared = self.encoder(fused)
            return self._apply_heads(shared)

    class ScreenDataset(Dataset):
        def __init__(self, indices, data):
            self.indices = indices
            self.data = data
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, idx):
            i = self.indices[idx]
            return {
                "rnafm": torch.from_numpy(self.data["rnafm_features"][i]),
                "edit_delta": torch.from_numpy(self.data["edit_delta_features"][i]),
                "bp_submatrix": torch.from_numpy(self.data["bp_submatrices"][i]).unsqueeze(0),
                "hand_feat": torch.from_numpy(self.data["hand_features"][i]),
                "label_binary": torch.tensor(self.data["labels_binary"][i]),
                "label_enzyme": torch.tensor(self.data["labels_enzyme"][i]),
                "per_enzyme_labels": torch.from_numpy(self.data["per_enzyme_labels"][i]),
                "index": i,
            }

    def standard_collate(batch_list):
        result = {}
        for key in batch_list[0]:
            vals = [b[key] for b in batch_list]
            if isinstance(vals[0], torch.Tensor):
                result[key] = torch.stack(vals)
            else:
                result[key] = vals
        return result

    STAGE1_EPOCHS = 10
    STAGE1_LR = 1e-3
    STAGE1_WD = 1e-4
    STAGE1_ENZ_W = 0.3
    STAGE1_CLS_W = 0.1
    STAGE2_EPOCHS = 20
    STAGE2_LR = 5e-4
    STAGE2_WD = 1e-4
    BATCH_SIZE = 64
    N_FOLDS = 2

    def compute_loss(binary_logit, per_enzyme_logits, enzyme_cls_logits, batch):
        label_binary = batch["label_binary"]
        label_enzyme = batch["label_enzyme"]
        per_enzyme_labels_b = batch["per_enzyme_labels"]

        loss_binary = F.binary_cross_entropy_with_logits(binary_logit, label_binary)

        loss_enzyme_heads = torch.tensor(0.0, device=binary_logit.device)
        n_valid = 0
        for h_idx in range(N_PER_ENZYME):
            enz_labels = per_enzyme_labels_b[:, h_idx]
            mask = enz_labels >= 0
            if mask.sum() > 0:
                loss_enzyme_heads += F.binary_cross_entropy_with_logits(
                    per_enzyme_logits[h_idx][mask], enz_labels[mask])
                n_valid += 1
        if n_valid > 0:
            loss_enzyme_heads /= n_valid

        pos_mask = label_binary > 0.5
        loss_cls = torch.tensor(0.0, device=binary_logit.device)
        if pos_mask.sum() > 0:
            loss_cls = F.cross_entropy(enzyme_cls_logits[pos_mask], label_enzyme[pos_mask])

        total = loss_binary + STAGE1_ENZ_W * loss_enzyme_heads + STAGE1_CLS_W * loss_cls
        return total, loss_binary.item()

    def train_one_epoch(model, loader, optimizer, device):
        model.train()
        total_loss, total_bce, nb = 0.0, 0.0, 0
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            optimizer.zero_grad()
            bl, pel, ecl = model(batch)
            loss, bce = compute_loss(bl, pel, ecl, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            total_bce += bce
            nb += 1
        return total_loss / max(nb, 1), total_bce / max(nb, 1)

    @torch.no_grad()
    def evaluate_model(model, loader, device):
        model.eval()
        all_probs, all_labels, all_enzymes = [], [], []
        all_pe_probs = [[] for _ in range(N_PER_ENZYME)]
        all_pe_labels = [[] for _ in range(N_PER_ENZYME)]
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            bl, pel, _ = model(batch)
            probs = torch.sigmoid(bl).cpu().numpy()
            labels = batch["label_binary"].cpu().numpy()
            enzymes = batch["label_enzyme"].cpu().numpy()
            pe_labels = batch["per_enzyme_labels"].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels)
            all_enzymes.extend(enzymes)
            for h in range(N_PER_ENZYME):
                ep = torch.sigmoid(pel[h]).cpu().numpy()
                mask = pe_labels[:, h] >= 0
                if mask.any():
                    all_pe_probs[h].extend(ep[mask])
                    all_pe_labels[h].extend(pe_labels[mask, h])

        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)

        try:
            overall_auroc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            overall_auroc = 0.5

        pe_aurocs = {}
        for h, enz in enumerate(PER_ENZYME_HEADS):
            if len(all_pe_probs[h]) > 10:
                y_true = np.array(all_pe_labels[h])
                y_pred = np.array(all_pe_probs[h])
                if len(np.unique(y_true)) > 1:
                    try:
                        pe_aurocs[enz] = roc_auc_score(y_true, y_pred)
                    except ValueError:
                        pe_aurocs[enz] = 0.5
                else:
                    pe_aurocs[enz] = float("nan")
            else:
                pe_aurocs[enz] = float("nan")

        return overall_auroc, pe_aurocs

    # --- Pretext task: predict center unpaired ---
    class PretextHead(nn.Module):
        def __init__(self, d_shared):
            super().__init__()
            self.head = nn.Linear(d_shared, 1)
        def forward(self, shared):
            return self.head(shared).squeeze(-1)

    def make_pretext_labels(site_ids_list, dot_brackets_dict):
        labels = np.zeros(len(site_ids_list), dtype=np.float32)
        for i, sid in enumerate(site_ids_list):
            db = dot_brackets_dict.get(sid, "")
            if len(db) > CENTER:
                labels[i] = 1.0 if db[CENTER] == "." else 0.0
        return labels

    def train_pretext(model, data_dict, device, n_epochs=5):
        """Pre-train encoder with structure pretext task."""
        pretext_labels = make_pretext_labels(data_dict["site_ids"], data_dict["dot_brackets"])
        pretext_head = PretextHead(D_SHARED).to(device)
        all_idx = np.arange(len(pretext_labels))
        ds = ScreenDataset(all_idx, data_dict)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=standard_collate)

        optimizer = torch.optim.Adam(
            list(model.get_shared_params()) + list(pretext_head.parameters()),
            lr=1e-3, weight_decay=1e-4)

        model.train()
        for epoch in range(n_epochs):
            total_loss, nb = 0.0, 0
            for batch in loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                indices = batch["index"]
                targets = torch.tensor([pretext_labels[i] for i in indices], dtype=torch.float32).to(device)

                optimizer.zero_grad()
                # Get shared representation
                if hasattr(model, 'conv'):  # Conv2DBP
                    bp = batch["bp_submatrix"]
                    conv_out = model.conv(bp).flatten(1)
                    conv_out = model.conv_fc(conv_out)
                    fused = torch.cat([conv_out, batch["rnafm"], batch["hand_feat"]], dim=-1)
                    shared = model.encoder(fused)
                else:  # HierarchicalAttention
                    bp = batch["bp_submatrix"].squeeze(1)
                    local_in = model.local_proj(bp) + model.local_pos_enc.unsqueeze(0)
                    local_out = model.local_transformer(local_in)
                    attn_w = torch.softmax(model.local_pool_attn(local_out), dim=1)
                    local_repr = (local_out * attn_w).sum(dim=1)
                    rnafm = batch["rnafm"]
                    q = model.cross_q(local_repr).unsqueeze(1)
                    k = model.cross_k(rnafm).unsqueeze(1)
                    v = model.cross_v(rnafm).unsqueeze(1)
                    attn_scores = (q * k).sum(-1) / math.sqrt(64)
                    attn_weights = torch.softmax(attn_scores, dim=-1)
                    cross_repr = (attn_weights.unsqueeze(-1) * v).squeeze(1)
                    fused = torch.cat([local_repr, cross_repr, rnafm, batch["hand_feat"]], dim=-1)
                    shared = model.encoder(fused)

                pred = pretext_head(shared)
                loss = F.binary_cross_entropy_with_logits(pred, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                nb += 1

            logger.info(f"    Pretext epoch {epoch+1}: loss={total_loss/max(nb,1):.4f}")

        del pretext_head
        return model

    # -----------------------------------------------------------------------
    # Run 4 configurations: A6/A8 x baseline/pretext
    # -----------------------------------------------------------------------

    architectures = [
        ("A6_Conv2DBP_T1_baseline", Conv2DBP, False),
        ("A6_Conv2DBP_T6_pretext", Conv2DBP, True),
        ("A8_HierarchicalAttention_T1_baseline", HierarchicalAttention, False),
        ("A8_HierarchicalAttention_T6_pretext", HierarchicalAttention, True),
    ]

    all_results = {}

    for arch_name, model_cls, use_pretext in architectures:
        logger.info(f"\n{'='*70}")
        logger.info(f"Training: {arch_name}")
        logger.info(f"{'='*70}")

        strat_key = labels_enzyme * 2 + labels_binary.astype(np.int64)
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        fold_results = []
        total_t0 = time.time()

        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(n), strat_key)):
            logger.info(f"\n--- Fold {fold+1}/{N_FOLDS} ({len(train_idx)} train, {len(val_idx)} val) ---")
            fold_t0 = time.time()

            fold_model = model_cls().to(DEVICE)

            # Pretext pretraining if requested
            if use_pretext:
                logger.info("  Pretext pretraining (5 epochs)...")
                fold_model = train_pretext(fold_model, data, DEVICE, n_epochs=5)

            # Stage 1: Joint training
            train_ds = ScreenDataset(train_idx, data)
            val_ds = ScreenDataset(val_idx, data)
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=standard_collate)
            val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=standard_collate)

            optimizer = torch.optim.Adam([
                {"params": fold_model.get_shared_params(), "lr": STAGE1_LR},
                {"params": fold_model.get_adapter_params(), "lr": STAGE1_LR},
            ], weight_decay=STAGE1_WD)

            logger.info(f"  Stage 1: {STAGE1_EPOCHS} epochs")
            for epoch in range(STAGE1_EPOCHS):
                avg_loss, avg_bce = train_one_epoch(fold_model, train_loader, optimizer, DEVICE)
                val_auroc, val_pe = evaluate_model(fold_model, val_loader, DEVICE)
                enz_str = " ".join(f"{e}={v:.3f}" for e, v in val_pe.items() if not np.isnan(v))
                logger.info(f"    Epoch {epoch+1:2d} | loss={avg_loss:.4f} bce={avg_bce:.4f} | val={val_auroc:.4f} | {enz_str}")

            # Stage 2: Adapter-only
            for p in fold_model.get_shared_params():
                p.requires_grad = False
            optimizer2 = torch.optim.Adam(fold_model.get_adapter_params(), lr=STAGE2_LR, weight_decay=STAGE2_WD)

            logger.info(f"  Stage 2: {STAGE2_EPOCHS} epochs")
            for epoch in range(STAGE2_EPOCHS):
                avg_loss, avg_bce = train_one_epoch(fold_model, train_loader, optimizer2, DEVICE)
                val_auroc, val_pe = evaluate_model(fold_model, val_loader, DEVICE)
                if (epoch + 1) % 5 == 0 or epoch == STAGE2_EPOCHS - 1:
                    enz_str = " ".join(f"{e}={v:.3f}" for e, v in val_pe.items() if not np.isnan(v))
                    logger.info(f"    Epoch {STAGE1_EPOCHS+epoch+1:2d} | loss={avg_loss:.4f} | val={val_auroc:.4f} | {enz_str}")

            # Final eval
            final_auroc, final_pe = evaluate_model(fold_model, val_loader, DEVICE)
            fold_time = time.time() - fold_t0
            logger.info(f"  Fold {fold+1} FINAL: overall={final_auroc:.4f} | time={fold_time:.0f}s")
            for enz, auroc in final_pe.items():
                logger.info(f"    {enz}: {auroc:.4f}")

            fold_results.append({
                "fold": fold + 1,
                "overall_auroc": final_auroc,
                "per_enzyme_aurocs": final_pe,
                "time_s": fold_time,
            })

            del fold_model, optimizer, train_loader, val_loader, train_ds, val_ds
            gc.collect()
            if DEVICE.type == "mps":
                torch.mps.empty_cache()

        total_time = time.time() - total_t0

        # Aggregate
        overall_aurocs = [r["overall_auroc"] for r in fold_results]
        mean_overall = np.mean(overall_aurocs)
        std_overall = np.std(overall_aurocs)

        pe_means, pe_stds = {}, {}
        for enz in PER_ENZYME_HEADS:
            vals = [r["per_enzyme_aurocs"].get(enz, float("nan")) for r in fold_results]
            vals = [v for v in vals if not np.isnan(v)]
            if vals:
                pe_means[enz] = round(np.mean(vals), 4)
                pe_stds[enz] = round(np.std(vals), 4)

        logger.info(f"\n  {arch_name} SUMMARY:")
        logger.info(f"    Overall: {mean_overall:.4f} +/- {std_overall:.4f}")
        for enz in PER_ENZYME_HEADS:
            if enz in pe_means:
                logger.info(f"    {enz}: {pe_means[enz]:.4f} +/- {pe_stds[enz]:.4f}")
        logger.info(f"    Time: {total_time:.0f}s")

        all_results[arch_name] = {
            "overall_auroc_mean": round(mean_overall, 4),
            "overall_auroc_std": round(std_overall, 4),
            "per_enzyme_auroc_mean": pe_means,
            "per_enzyme_auroc_std": pe_stds,
            "total_time_s": round(total_time, 1),
            "fold_results": fold_results,
        }

        # Save incrementally
        with open(OUTPUT_DIR / "v5_matched_screen_results.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # --- Also run XGB baseline ---
    logger.info("\n" + "=" * 70)
    logger.info("XGB Per-Enzyme Baseline on V5 matched")
    logger.info("=" * 70)

    try:
        from xgboost import XGBClassifier
        xgb_results = {}
        for enz in PER_ENZYME_HEADS:
            enz_idx = ENZYME_TO_IDX[enz]
            mask = labels_enzyme == enz_idx
            if mask.sum() < 20:
                continue
            X_enz = hand_features[mask]
            y_enz = labels_binary[mask]
            n_pos = int(y_enz.sum())
            n_neg = int((y_enz == 0).sum())
            logger.info(f"  {enz}: {n_pos} pos, {n_neg} neg")

            skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
            fold_aurocs = []
            for fold, (tr, va) in enumerate(skf.split(X_enz, y_enz)):
                clf = XGBClassifier(
                    n_estimators=200, max_depth=4, learning_rate=0.1,
                    subsample=0.8, colsample_bytree=0.8,
                    use_label_encoder=False, eval_metric="logloss",
                    random_state=SEED, verbosity=0,
                )
                clf.fit(X_enz[tr], y_enz[tr])
                probs = clf.predict_proba(X_enz[va])[:, 1]
                try:
                    auroc = roc_auc_score(y_enz[va], probs)
                except ValueError:
                    auroc = 0.5
                fold_aurocs.append(auroc)
                logger.info(f"    Fold {fold+1}: {auroc:.4f}")

            mean_auroc = np.mean(fold_aurocs)
            xgb_results[enz] = round(mean_auroc, 4)
            logger.info(f"  {enz} XGB mean AUROC: {mean_auroc:.4f}")

        all_results["XGB_40d"] = {
            "overall_auroc_mean": round(np.mean(list(xgb_results.values())), 4) if xgb_results else None,
            "per_enzyme_auroc_mean": xgb_results,
        }
    except ImportError:
        logger.warning("XGBoost not available, skipping baseline")

    # Save final results
    with open(OUTPUT_DIR / "v5_matched_screen_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # --- Print comparison table ---
    logger.info("\n" + "=" * 90)
    logger.info("FINAL COMPARISON: V5 matched (motif-matched A3A negatives)")
    logger.info("=" * 90)

    header = f"{'Architecture':<40} {'Overall':>10} {'A3A':>8} {'A3B':>8} {'A3G':>8} {'A3A_A3G':>8} {'Neither':>8}"
    logger.info(header)
    logger.info("-" * 90)

    for name, res in all_results.items():
        overall = res.get("overall_auroc_mean", "N/A")
        pe = res.get("per_enzyme_auroc_mean", {})
        overall_str = f"{overall:.4f}" if isinstance(overall, float) else str(overall)
        parts = [f"{name:<40}", f"{overall_str:>10}"]
        for enz in PER_ENZYME_HEADS:
            val = pe.get(enz, "N/A")
            parts.append(f"{val:.4f}" if isinstance(val, float) else f"{'N/A':>8}")
        logger.info(" ".join(parts))

    # Compare with old V5 results
    old_results_path = OUTPUT_DIR / "v5_screen_results.json"
    if old_results_path.exists():
        with open(old_results_path) as f:
            old_results = json.load(f)
        logger.info("\n--- Comparison with OLD V5 (unmatched negatives) ---")
        for name in all_results:
            if name in old_results:
                old_a3a = old_results[name].get("per_enzyme_auroc_mean", {}).get("A3A", "N/A")
                new_a3a = all_results[name].get("per_enzyme_auroc_mean", {}).get("A3A", "N/A")
                logger.info(f"  {name}: A3A old={old_a3a}, new={new_a3a}")

    logger.info("\nPipeline complete!")
    return all_results


# ===========================================================================
# Main
# ===========================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=0,
                        help="Run from this step (0=all, 5=screen only)")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("V5 MOTIF-MATCHED A3A NEGATIVES PIPELINE")
    logger.info("=" * 70)

    t0 = time.time()

    if args.step <= 1:
        # Step 1: Generate negatives
        neg_df, new_seqs = step1_generate_negatives()
    else:
        neg_df = pd.read_csv(DATA_DIR / "a3a_matched_negatives_intermediate.csv")
        with open(DATA_DIR / "a3a_matched_negatives_seqs_intermediate.json") as f:
            new_seqs = json.load(f)
        logger.info(f"Loaded intermediate: {len(neg_df)} negatives, {len(new_seqs)} seqs")

    if args.step <= 2:
        # Step 2: Compute structures
        structure_delta, loop_records, bp_subs, dot_brackets = step2_compute_structures(neg_df, new_seqs)
    else:
        sd = np.load(DATA_DIR / "a3a_matched_structure_intermediate.npz", allow_pickle=True)
        structure_delta = {str(sid): sd["delta_features"][i] for i, sid in enumerate(sd["site_ids"])}
        bp_subs = {}
        dot_brackets = {}
        loop_records = []
        logger.info(f"Loaded intermediate: {len(structure_delta)} structure deltas")

    if args.step <= 3:
        import torch
        # Step 3: Compute RNA-FM embeddings
        rnafm_pooled, rnafm_pooled_ed = step3_compute_rnafm(neg_df, new_seqs)
    else:
        import torch
        rnafm_pooled = torch.load(EMB_DIR / "rnafm_a3a_matched_neg_intermediate.pt",
                                   map_location="cpu", weights_only=False)
        rnafm_pooled_ed = torch.load(EMB_DIR / "rnafm_a3a_matched_neg_edited_intermediate.pt",
                                      map_location="cpu", weights_only=False)
        logger.info(f"Loaded intermediate: {len(rnafm_pooled)} embeddings")

    if args.step <= 4:
        # Step 4: Rebuild V5
        v5_matched, v5_seqs, bp_subs, dot_brackets = step4_rebuild_v5(
            neg_df, new_seqs, structure_delta, loop_records,
            bp_subs, dot_brackets, rnafm_pooled, rnafm_pooled_ed)
    else:
        v5_matched = pd.read_csv(V5M_SPLITS)
        with open(V5M_SEQS) as f:
            v5_seqs = json.load(f)
        bp_subs = {}
        dot_brackets = {}
        logger.info(f"Loaded V5 matched: {len(v5_matched)} sites")

    # Step 5: Rerun screen
    results = step5_rerun_screen(v5_matched, v5_seqs, bp_subs, dot_brackets)

    total_time = time.time() - t0
    logger.info(f"\nTotal pipeline time: {total_time/60:.1f} minutes")


if __name__ == "__main__":
    main()
