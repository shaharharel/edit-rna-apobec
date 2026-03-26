"""Centralized feature extraction for APOBEC C-to-U RNA editing prediction.

Provides functions for extracting motif, loop geometry, structure delta,
pairing profile, and RNAsee features from RNA sequences and pre-computed caches.

All sequences are 201-nt with the edit site (C) at center position 100 (0-indexed).
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CENTER = 100  # Edit site position in 201-nt window (0-indexed)


# ---------------------------------------------------------------------------
# Motif features (24-dim)
# ---------------------------------------------------------------------------

def extract_motif_from_seq(seq: str) -> np.ndarray:
    """Extract 24-dim motif features from a 201-nt sequence.

    Features: 5' dinucleotide (4), 3' dinucleotide (4),
              trinucleotide upstream m2+m1 (8), trinucleotide downstream p1+p2 (8).

    Args:
        seq: 201-nt RNA sequence with edit site C at position 100.

    Returns:
        24-dim float32 array.
    """
    seq = seq.upper().replace("T", "U")
    ep = CENTER
    bases = ["A", "C", "G", "U"]

    up = seq[ep - 1] if ep > 0 else "N"
    down = seq[ep + 1] if ep < len(seq) - 1 else "N"

    # 5' dinucleotide (4-dim)
    feat_5p = [1.0 if up + "C" == m else 0.0 for m in ["UC", "CC", "AC", "GC"]]
    # 3' dinucleotide (4-dim)
    feat_3p = [1.0 if "C" + down == m else 0.0 for m in ["CA", "CG", "CU", "CC"]]

    # Trinucleotide upstream context (8-dim: m2 + m1)
    trinuc_up = [0.0] * 8
    for offset, bo in [(-2, 0), (-1, 4)]:
        pos = ep + offset
        if 0 <= pos < len(seq):
            for bi, b in enumerate(bases):
                if seq[pos] == b:
                    trinuc_up[bo + bi] = 1.0

    # Trinucleotide downstream context (8-dim: p1 + p2)
    trinuc_down = [0.0] * 8
    for offset, bo in [(1, 0), (2, 4)]:
        pos = ep + offset
        if 0 <= pos < len(seq):
            for bi, b in enumerate(bases):
                if seq[pos] == b:
                    trinuc_down[bo + bi] = 1.0

    return np.array(feat_5p + feat_3p + trinuc_up + trinuc_down, dtype=np.float32)


def extract_motif_features(sequences: Dict[str, str], site_ids: List[str]) -> np.ndarray:
    """Extract 24-dim motif features for a batch of sites.

    Args:
        sequences: Mapping of site_id -> 201-nt sequence.
        site_ids: List of site IDs to extract features for.

    Returns:
        [N, 24] float32 array.
    """
    features = []
    for sid in site_ids:
        seq = sequences.get(str(sid), "N" * 201)
        features.append(extract_motif_from_seq(seq))
    return np.array(features, dtype=np.float32)


# ---------------------------------------------------------------------------
# Loop geometry features (9-dim)
# ---------------------------------------------------------------------------

LOOP_FEATURE_COLS = [
    "is_unpaired", "loop_size", "dist_to_junction", "dist_to_apex",
    "relative_loop_position", "left_stem_length", "right_stem_length",
    "max_adjacent_stem_length", "local_unpaired_fraction",
]


def extract_loop_features(loop_df: pd.DataFrame, site_ids: List[str]) -> np.ndarray:
    """Extract 9-dim loop geometry features from pre-computed loop position data.

    Args:
        loop_df: DataFrame indexed by site_id with loop geometry columns.
        site_ids: List of site IDs.

    Returns:
        [N, 9] float32 array.
    """
    features = []
    for sid in site_ids:
        if str(sid) in loop_df.index:
            features.append(loop_df.loc[str(sid), LOOP_FEATURE_COLS].values.astype(np.float32))
        else:
            features.append(np.zeros(len(LOOP_FEATURE_COLS), dtype=np.float32))
    return np.array(features, dtype=np.float32)


# ---------------------------------------------------------------------------
# Structure delta features (7-dim)
# ---------------------------------------------------------------------------

def extract_structure_delta_features(structure_delta: Dict[str, np.ndarray],
                                     site_ids: List[str]) -> np.ndarray:
    """Extract 7-dim structure delta features from pre-computed cache.

    Features: delta_pairing_center, delta_accessibility_center, delta_entropy_center,
              delta_mfe, mean_delta_pairing_window, mean_delta_accessibility_window,
              std_delta_pairing_window.

    Args:
        structure_delta: Mapping of site_id -> 7-dim delta feature array.
        site_ids: List of site IDs.

    Returns:
        [N, 7] float32 array.
    """
    return np.array(
        [structure_delta.get(str(sid), np.zeros(7)) for sid in site_ids],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Pairing profile features (~50-dim)
# ---------------------------------------------------------------------------

def extract_pairing_profile_features(data: Dict, site_ids: List[str]) -> np.ndarray:
    """Extract ~50-dim pairing profile features from structure cache.

    Uses windowed statistics of pairing probabilities, accessibilities,
    and MFE values from ViennaRNA computations.

    Args:
        data: Dict with keys 'pairing_probs', 'pairing_probs_edited',
              'accessibilities', 'accessibilities_edited', 'mfes', 'mfes_edited'.
        site_ids: List of site IDs.

    Returns:
        [N, ~50] float32 array.
    """
    features = []
    for sid in site_ids:
        sid = str(sid)
        pp = data["pairing_probs"].get(sid, np.zeros(201))
        pp_ed = data["pairing_probs_edited"].get(sid, np.zeros(201))
        acc = data["accessibilities"].get(sid, np.zeros(201))
        acc_ed = data["accessibilities_edited"].get(sid, np.zeros(201))
        mfe = data["mfes"].get(sid, 0.0)
        mfe_ed = data["mfes_edited"].get(sid, 0.0)
        ep = CENTER

        feat = []
        # Windowed pairing probability statistics
        for w in [5, 11, 21]:
            s, e = max(0, ep - w // 2), min(201, ep + w // 2 + 1)
            win_pp = pp[s:e]
            win_pp_ed = pp_ed[s:e]
            feat.extend([win_pp.mean(), win_pp.std(),
                         win_pp_ed.mean(), win_pp_ed.std(),
                         (win_pp_ed - win_pp).mean()])

        # Windowed accessibility statistics
        for w in [5, 11, 21]:
            s, e = max(0, ep - w // 2), min(201, ep + w // 2 + 1)
            win_acc = acc[s:e]
            win_acc_ed = acc_ed[s:e]
            feat.extend([win_acc.mean(), win_acc.std(),
                         win_acc_ed.mean(), win_acc_ed.std(),
                         (win_acc_ed - win_acc).mean()])

        # Center position features
        feat.extend([pp[ep], pp_ed[ep], pp_ed[ep] - pp[ep],
                     acc[ep], acc_ed[ep], acc_ed[ep] - acc[ep]])

        # MFE features
        feat.extend([mfe, mfe_ed, mfe_ed - mfe])

        # Regional delta pairing (10 bins + center region)
        delta_pp = pp_ed - pp
        for i in range(10):
            s = i * 20
            e = min(201, (i + 1) * 20)
            feat.append(delta_pp[s:e].mean())
        feat.append(delta_pp[90:111].mean())

        features.append(feat)
    return np.array(features, dtype=np.float32)


# ---------------------------------------------------------------------------
# Combined hand feature builders
# ---------------------------------------------------------------------------

def build_hand_features(site_ids: List[str], sequences: Dict[str, str],
                        structure_delta: Dict[str, np.ndarray],
                        loop_df: pd.DataFrame) -> np.ndarray:
    """Build 40-dim hand feature matrix.

    40 = motif(24) + struct_delta(7) + loop(9).

    Args:
        site_ids: List of site IDs.
        sequences: Mapping of site_id -> 201-nt sequence.
        structure_delta: Mapping of site_id -> 7-dim delta features.
        loop_df: DataFrame indexed by site_id with loop geometry columns.

    Returns:
        [N, 40] float32 array with NaN replaced by 0.
    """
    motif = extract_motif_features(sequences, site_ids)
    struct = extract_structure_delta_features(structure_delta, site_ids)
    loop = extract_loop_features(loop_df, site_ids)
    hand = np.concatenate([motif, struct, loop], axis=1)
    return np.nan_to_num(hand, nan=0.0)


def build_hand_46_features(site_ids: List[str], sequences: Dict[str, str],
                           structure_delta: Dict[str, np.ndarray],
                           loop_df: pd.DataFrame,
                           baseline_struct: Dict[str, np.ndarray]) -> np.ndarray:
    """Build 46-dim hand feature matrix.

    46 = motif(24) + struct_delta(7) + loop(9) + baseline_struct(6).

    Args:
        site_ids: List of site IDs.
        sequences: Mapping of site_id -> 201-nt sequence.
        structure_delta: Mapping of site_id -> 7-dim delta features.
        loop_df: DataFrame indexed by site_id with loop geometry columns.
        baseline_struct: Mapping of site_id -> 6-dim baseline structure features.

    Returns:
        [N, 46] float32 array with NaN replaced by 0.
    """
    features = []
    for sid in site_ids:
        sid_str = str(sid)

        # Motif (24-dim)
        seq = sequences.get(sid_str, "N" * 201)
        motif = extract_motif_from_seq(seq)

        # Structure delta (7-dim)
        sd = structure_delta.get(sid_str, np.zeros(7, dtype=np.float32))
        if not isinstance(sd, np.ndarray):
            sd = np.array(sd, dtype=np.float32)

        # Loop geometry (9-dim)
        if sid_str in loop_df.index:
            loop = loop_df.loc[sid_str, LOOP_FEATURE_COLS].values.astype(np.float32)
        else:
            loop = np.zeros(len(LOOP_FEATURE_COLS), dtype=np.float32)

        # Baseline structure (6-dim)
        bl = baseline_struct.get(sid_str, np.zeros(6, dtype=np.float32))

        feat = np.concatenate([motif, sd, loop, bl])
        features.append(feat)

    result = np.array(features, dtype=np.float32)
    return np.nan_to_num(result, nan=0.0)


# ---------------------------------------------------------------------------
# RNAsee 50-bit binary encoding
# ---------------------------------------------------------------------------

def encode_rnasee_from_seq(seq: str) -> np.ndarray:
    """Encode a 201-nt sequence using RNAsee's exact 50-bit binary encoding.

    15nt upstream + 10nt downstream of center C (excluded).
    Each nt: 2 bits (is_purine, pairs_GC). A=(1,0), G=(1,1), C=(0,1), U=(0,0).

    Args:
        seq: 201-nt RNA sequence with edit site C at position 100.

    Returns:
        50-dim float32 array.
    """
    BWD, FWD = 15, 10
    features = np.zeros((BWD + FWD) * 2, dtype=np.float32)
    if len(seq) < CENTER + FWD + 1:
        return features

    feat_idx = 0
    for offset in range(-BWD, 0):
        pos = CENTER + offset
        if pos < 0:
            feat_idx += 2
            continue
        nuc = seq[pos].upper()
        features[feat_idx] = 1.0 if nuc in ("A", "G") else 0.0
        features[feat_idx + 1] = 1.0 if nuc in ("G", "C") else 0.0
        feat_idx += 2

    for offset in range(1, FWD + 1):
        pos = CENTER + offset
        if pos >= len(seq):
            feat_idx += 2
            continue
        nuc = seq[pos].upper()
        features[feat_idx] = 1.0 if nuc in ("A", "G") else 0.0
        features[feat_idx + 1] = 1.0 if nuc in ("G", "C") else 0.0
        feat_idx += 2

    return features


# ---------------------------------------------------------------------------
# ViennaRNA structure computation
# ---------------------------------------------------------------------------

def _entropy_at_pos(bpp: np.ndarray, pos: int) -> float:
    """Compute structure entropy at a single position from base pair probability matrix."""
    probs = bpp[pos, :]
    probs = probs[probs > 1e-10]
    if len(probs) == 0:
        return 0.0
    unpaired = max(0, 1.0 - np.sum(probs))
    if unpaired > 1e-10:
        probs = np.append(probs, unpaired)
    return float(-np.sum(probs * np.log2(probs + 1e-10)))


def _stem_length(dot_bracket: str, boundary_pos: int, direction: str) -> int:
    """Count consecutive paired characters from boundary_pos.

    Args:
        dot_bracket: Secondary structure in dot-bracket notation.
        boundary_pos: Position to start counting from.
        direction: 'left' (scan leftward) or 'right' (scan rightward).

    Returns:
        Number of consecutive paired bases.
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
    else:
        i = boundary_pos
        while i < n and dot_bracket[i] in "()":
            count += 1
            i += 1
    return count


def _extract_loop_geometry(dot_bracket: str, pos: int) -> np.ndarray:
    """Extract 9-dim loop geometry features from dot-bracket structure.

    Features: is_unpaired, loop_size, dist_to_junction, dist_to_apex,
              relative_loop_position, left_stem_length, right_stem_length,
              max_adjacent_stem_length, local_unpaired_fraction.

    Args:
        dot_bracket: MFE secondary structure in dot-bracket notation.
        pos: Position of the edit site.

    Returns:
        9-dim float32 array.
    """
    n = len(dot_bracket)
    feats = np.zeros(9, dtype=np.float32)

    if not dot_bracket or pos >= n or pos < 0:
        return feats

    is_unpaired = dot_bracket[pos] == "."
    feats[0] = float(is_unpaired)

    # Local unpaired fraction (within +/- 10 nt)
    w = 10
    w_start = max(0, pos - w)
    w_end = min(n, pos + w + 1)
    local_region = dot_bracket[w_start:w_end]
    feats[8] = sum(1 for c in local_region if c == ".") / len(local_region)

    if is_unpaired:
        # Find loop boundaries
        left = pos - 1
        while left >= 0 and dot_bracket[left] == ".":
            left -= 1
        right = pos + 1
        while right < n and dot_bracket[right] == ".":
            right += 1

        loop_start = (left + 1) if left >= 0 else 0
        loop_end = (right - 1) if right < n else n - 1
        loop_size = loop_end - loop_start + 1

        dist_left = pos - loop_start
        dist_right = loop_end - pos
        dist_to_nearest = min(dist_left, dist_right)
        relative_pos = dist_left / (loop_size - 1) if loop_size > 1 else 0.5
        apex = (loop_start + loop_end) / 2.0
        dist_to_apex = abs(pos - apex)

        # Stem lengths
        left_stem = _stem_length(dot_bracket, left, "left")
        right_stem = _stem_length(dot_bracket, right, "right")

        feats[1] = loop_size
        feats[2] = dist_to_nearest  # dist_to_junction
        feats[3] = dist_to_apex
        feats[4] = relative_pos
        feats[5] = left_stem
        feats[6] = right_stem
        feats[7] = max(left_stem, right_stem)
    else:
        # In a stem
        dist_left = 0
        i = pos - 1
        while i >= 0 and dot_bracket[i] in "()":
            dist_left += 1
            i -= 1
        dist_right = 0
        j = pos + 1
        while j < n and dot_bracket[j] in "()":
            dist_right += 1
            j += 1
        feats[2] = min(dist_left, dist_right)  # dist_to_junction

        stem_left = _stem_length(dot_bracket, pos, "left")
        stem_right = _stem_length(dot_bracket, pos, "right")
        feats[5] = stem_left
        feats[6] = stem_right
        feats[7] = max(stem_left, stem_right)

    return feats


def compute_vienna_features(seq: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ViennaRNA structure features for original and C->U edited sequences.

    Computes MFE structure, base pair probabilities, and accessibilities for
    both the original and edited sequence. Returns delta features, loop geometry,
    and baseline structure features.

    Args:
        seq: 201-nt RNA sequence with C at center position.

    Returns:
        Tuple of:
            struct_delta: 7-dim delta features (pairing/accessibility/entropy changes).
            loop_feats: 9-dim loop geometry features from MFE structure.
            baseline_struct: 6-dim baseline structure features of original sequence.
    """
    import RNA

    seq = seq.upper().replace("T", "U")
    n = len(seq)
    center = n // 2

    # -- Original sequence --
    md = RNA.md()
    md.temperature = 37.0
    fc = RNA.fold_compound(seq, md)
    mfe_structure, mfe = fc.mfe()
    _, _ = fc.pf()
    bpp_raw = np.array(fc.bpp())
    bpp = bpp_raw[1:n + 1, 1:n + 1]
    pairing_prob = np.clip(np.sum(bpp, axis=0) + np.sum(bpp, axis=1), 0, 1)
    accessibility = 1.0 - pairing_prob
    entropy_center = _entropy_at_pos(bpp, center)

    # -- Edited sequence (C->U at center) --
    seq_list = list(seq)
    if center < len(seq_list) and seq_list[center].upper() == "C":
        seq_list[center] = "U"
    edited_seq = "".join(seq_list)

    fc_ed = RNA.fold_compound(edited_seq, md)
    _, mfe_ed = fc_ed.mfe()
    _, _ = fc_ed.pf()
    bpp_raw_ed = np.array(fc_ed.bpp())
    bpp_ed = bpp_raw_ed[1:n + 1, 1:n + 1]
    pairing_prob_ed = np.clip(np.sum(bpp_ed, axis=0) + np.sum(bpp_ed, axis=1), 0, 1)
    accessibility_ed = 1.0 - pairing_prob_ed
    entropy_center_ed = _entropy_at_pos(bpp_ed, center)

    # -- 7-dim structure delta features --
    window = 10
    start = max(0, center - window)
    end = min(n, center + window + 1)
    dp = pairing_prob_ed - pairing_prob
    da = accessibility_ed - accessibility

    struct_delta = np.zeros(7, dtype=np.float32)
    struct_delta[0] = dp[center]
    struct_delta[1] = da[center]
    struct_delta[2] = entropy_center_ed - entropy_center
    struct_delta[3] = mfe_ed - mfe
    struct_delta[4] = np.mean(dp[start:end])
    struct_delta[5] = np.mean(da[start:end])
    struct_delta[6] = np.std(dp[start:end])

    # -- 9-dim loop geometry features from MFE structure --
    loop_feats = _extract_loop_geometry(mfe_structure, center)

    # -- 6-dim baseline structure features (original sequence) --
    w_start = max(0, center - window)
    w_end = min(n, center + window + 1)
    baseline_struct = np.array([
        pairing_prob[center],
        accessibility[center],
        entropy_center,
        mfe,
        np.mean(pairing_prob[w_start:w_end]),
        np.mean(accessibility[w_start:w_end]),
    ], dtype=np.float32)

    return struct_delta, loop_feats, baseline_struct
