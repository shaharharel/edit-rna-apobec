#!/usr/bin/env python
"""Build v4 multi-enzyme dataset with cancer-trinucleotide-matched negatives.

Pipeline:
  1) Load v3 positives (unchanged).
  2) Load cancer C>T trinucleotide distribution
     (data/processed/multi_enzyme/cancer_ct_trinuc_distribution.csv).
  3) Generate v4 negatives via
     src.data.apobec_negatives.generate_negatives_cancer_matched
     using the appropriate genome (hg19 or hg38) for each positive.
  4) Write splits CSV and sequences JSON for v4.
  5) Compute loop_position_per_site for v4 negatives (extending v3 file).
  6) Compute / extend the structure cache for v4 negatives.

Outputs:
  - data/processed/multi_enzyme/splits_multi_enzyme_v4_cancer_matched.csv
  - data/processed/multi_enzyme/multi_enzyme_sequences_v4_cancer_matched.json
  - data/processed/multi_enzyme/loop_position_per_site_v4_cancer_matched.csv
  - data/processed/embeddings/structure_cache_multi_enzyme_v4_cancer_matched.npz

Usage:
    conda run -n quris python scripts/multi_enzyme/build_v4_cancer_matched_dataset.py
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.apobec_negatives import generate_negatives_cancer_matched

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
ME_DIR = DATA_DIR / "processed/multi_enzyme"

# Inputs
V3_SPLITS = ME_DIR / "splits_multi_enzyme_v3_with_negatives.csv"
V3_SEQS = ME_DIR / "multi_enzyme_sequences_v3_with_negatives.json"
V3_LOOP = ME_DIR / "loop_position_per_site_v3.csv"
V3_STRUCT = ME_DIR / "structure_cache_multi_enzyme_v3.npz"

CANCER_TRINUC_CSV = ME_DIR / "cancer_ct_trinuc_distribution.csv"

GENOME_HG19 = DATA_DIR / "raw/genomes/hg19.fa"
GENOME_HG38 = DATA_DIR / "raw/genomes/hg38.fa"

# Outputs
OUT_SPLITS = ME_DIR / "splits_multi_enzyme_v4_cancer_matched.csv"
OUT_SEQS = ME_DIR / "multi_enzyme_sequences_v4_cancer_matched.json"
OUT_LOOP = ME_DIR / "loop_position_per_site_v4_cancer_matched.csv"
OUT_STRUCT = DATA_DIR / "processed/embeddings/structure_cache_multi_enzyme_v4_cancer_matched.npz"

SEED = 20260427
N_NEGATIVES_TARGET = 7788  # match v3 count


def _trinuc_of_seq(seq: str) -> str:
    """Return the trinucleotide centered at position 100 (the C)."""
    if len(seq) < 102:
        return ""
    s = seq.upper().replace("U", "T")
    triplet = s[99:102]
    if len(triplet) != 3:
        return ""
    return triplet


def main():
    for path in [V3_SPLITS, V3_SEQS, CANCER_TRINUC_CSV, GENOME_HG19, GENOME_HG38]:
        if not path.exists():
            logger.error("Missing input: %s", path)
            sys.exit(1)

    # 1) Load v3 positives + sequences
    logger.info("Loading v3 splits / sequences ...")
    v3 = pd.read_csv(V3_SPLITS)
    pos = v3[v3["is_edited"] == 1].copy().reset_index(drop=True)
    logger.info("v3 positives: %d", len(pos))

    with open(V3_SEQS) as f:
        v3_seqs = json.load(f)
    logger.info("v3 sequences: %d", len(v3_seqs))

    # Sequences for v4: start with positives only (so we know we have them all),
    # then fall back to v3 if any positive site is missing.
    v4_seqs = {}
    for sid in pos["site_id"]:
        sid = str(sid)
        if sid in v3_seqs:
            v4_seqs[sid] = v3_seqs[sid]
        else:
            logger.warning("Positive %s missing from v3 sequences", sid)

    # 2) Load cancer trinuc distribution
    cancer = pd.read_csv(CANCER_TRINUC_CSV)
    cancer_fracs = dict(zip(cancer["trinuc"], cancer["fraction"]))
    logger.info("Cancer trinuc fractions: %s",
                {k: round(v, 3) for k, v in cancer_fracs.items()})

    # 3) Generate v4 negatives via cancer-matched sampling
    # Exclude all known v3 sites (positives + negatives) so v4 negatives are fresh.
    known_sites = set()
    for _, row in v3.iterrows():
        if pd.notna(row.get("chr")) and pd.notna(row.get("start")):
            known_sites.add((str(row["chr"]), int(row["start"])))
    logger.info("Known sites to exclude (v3 pos+neg): %d", len(known_sites))

    logger.info("Generating v4 negatives (target=%d, seed=%d) ...",
                N_NEGATIVES_TARGET, SEED)
    t0 = time.time()
    new_seqs = {}
    neg_df = generate_negatives_cancer_matched(
        positives_df=pos,
        genomes={"hg19": GENOME_HG19, "hg38": GENOME_HG38},
        cancer_trinuc_fractions=cancer_fracs,
        n_negatives=N_NEGATIVES_TARGET,
        output_seqs=new_seqs,
        known_sites=known_sites,
        search_window=5000,
        seed=SEED,
    )
    logger.info("v4 negative generation: %d rows in %.1fs",
                len(neg_df), time.time() - t0)

    # 4) Build combined splits CSV
    pos_for_concat = pos.copy()
    if "trinuc" not in pos_for_concat.columns:
        # Compute pos trinuc from sequences for sanity / record-keeping
        pos_for_concat["trinuc"] = pos_for_concat["site_id"].astype(str).map(
            lambda sid: _trinuc_of_seq(v3_seqs.get(sid, ""))
        )

    combined = pd.concat([pos_for_concat, neg_df], ignore_index=True)

    # Standard column order
    cols = [
        "site_id", "chr", "start", "end", "strand", "enzyme", "dataset_source",
        "coordinate_system", "editing_rate", "is_edited",
        "flanking_seq", "seq_center", "source_type", "trinuc",
    ]
    final_cols = [c for c in cols if c in combined.columns]
    extra = [c for c in combined.columns if c not in final_cols]
    combined = combined[final_cols + extra]

    OUT_SPLITS.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT_SPLITS, index=False)
    logger.info("Saved: %s (%d rows)", OUT_SPLITS, len(combined))

    # Merge new neg sequences into v4_seqs
    v4_seqs.update(new_seqs)
    logger.info("v4 sequences: %d", len(v4_seqs))
    with open(OUT_SEQS, "w") as f:
        json.dump(v4_seqs, f)
    logger.info("Saved: %s", OUT_SEQS)

    # 5) Loop position features for v4 negatives (positives reuse v3).
    logger.info("Building loop-position file for v4 ...")
    build_loop_positions(combined, v4_seqs, neg_df)

    # 6) Structure cache for v4 (extends v3 cache with new negatives).
    logger.info("Building structure cache for v4 ...")
    build_structure_cache(v4_seqs)

    # Final sanity
    logger.info("--- v4 build complete ---")
    n_pos = (combined["is_edited"] == 1).sum()
    n_neg = (combined["is_edited"] == 0).sum()
    print(f"v4 splits: {len(combined)} sites ({n_pos} pos, {n_neg} neg)")


# ---------------------------------------------------------------------------
# Helpers (loop position + structure cache for v4 negatives)
# ---------------------------------------------------------------------------

EDIT_POS = 100


def _fold_features(seq: str):
    """Compute loop-position features at position EDIT_POS."""
    import RNA

    seq_rna = seq.upper().replace("T", "U")
    md = RNA.md()
    md.temperature = 37.0
    fc = RNA.fold_compound(seq_rna, md)
    dot_bracket, mfe = fc.mfe()
    return dot_bracket, float(mfe)


def _analyze_site_structure(dot_bracket: str, mfe: float, pos: int = EDIT_POS) -> dict:
    """Compute the same loop features as scripts/multi_enzyme/generate_loop_positions.py."""
    n = len(dot_bracket)
    result = {
        "is_unpaired": None,
        "loop_type": None,
        "loop_size": None,
        "dist_to_left_boundary": None,
        "dist_to_right_boundary": None,
        "dist_to_nearest_stem": None,
        "relative_loop_position": None,
        "dist_to_apex": None,
        "left_stem_length": None,
        "right_stem_length": None,
        "max_adjacent_stem_length": None,
        "dist_to_junction": None,
        "local_unpaired_fraction": None,
        "mfe": float(mfe),
    }
    if not dot_bracket or pos >= n or pos < 0:
        return result

    is_unpaired = dot_bracket[pos] == "."
    result["is_unpaired"] = bool(is_unpaired)

    window = 10
    w_start = max(0, pos - window)
    w_end = min(n, pos + window + 1)
    local_region = dot_bracket[w_start:w_end]
    result["local_unpaired_fraction"] = sum(1 for c in local_region if c == ".") / max(len(local_region), 1)

    def _stem_length(boundary_pos, direction):
        if boundary_pos < 0 or boundary_pos >= n or dot_bracket[boundary_pos] not in "()":
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
        rel_pos = dist_left / (loop_size - 1) if loop_size > 1 else 0.5
        apex = (loop_start + loop_end) / 2.0
        dist_apex = abs(pos - apex)

        # Loop type
        if left < 0 or right >= n:
            loop_type = "external"
        else:
            lc, rc = dot_bracket[left], dot_bracket[right]
            if lc == "(" and rc == ")":
                loop_type = "hairpin"
            elif lc == ")" and rc == "(":
                loop_type = "multiloop"
            elif lc == "(" and rc == "(":
                loop_type = "bulge_left"
            elif lc == ")" and rc == ")":
                loop_type = "bulge_right"
            else:
                loop_type = "other"

        result["loop_size"] = loop_size
        result["dist_to_left_boundary"] = dist_left
        result["dist_to_right_boundary"] = dist_right
        result["dist_to_nearest_stem"] = min(dist_left, dist_right)
        result["relative_loop_position"] = rel_pos
        result["dist_to_apex"] = dist_apex
        result["loop_type"] = loop_type
        result["left_stem_length"] = _stem_length(left, "left")
        result["right_stem_length"] = _stem_length(right, "right")
        result["max_adjacent_stem_length"] = max(result["left_stem_length"],
                                                 result["right_stem_length"])
        result["dist_to_junction"] = result["dist_to_nearest_stem"]
    else:
        dist_left_stem = 0
        i = pos - 1
        while i >= 0 and dot_bracket[i] in "()":
            dist_left_stem += 1
            i -= 1
        dist_right_stem = 0
        j = pos + 1
        while j < n and dot_bracket[j] in "()":
            dist_right_stem += 1
            j += 1
        result["dist_to_junction"] = min(dist_left_stem, dist_right_stem)
        stem_l = _stem_length(pos, "left")
        stem_r = _stem_length(pos, "right")
        result["left_stem_length"] = stem_l
        result["right_stem_length"] = stem_r
        result["max_adjacent_stem_length"] = max(stem_l, stem_r)

    return result


def build_loop_positions(splits_df: pd.DataFrame, seqs: dict, neg_df: pd.DataFrame) -> None:
    """Build loop_position file for v4 by reusing v3 rows for positives and
    computing features for v4 negatives.
    """
    # Reuse v3 rows for sites that appear in both
    if V3_LOOP.exists():
        v3_loop = pd.read_csv(V3_LOOP)
    else:
        v3_loop = pd.DataFrame()
    v3_sites = set(v3_loop["site_id"].astype(str)) if len(v3_loop) else set()

    v4_sites = set(splits_df["site_id"].astype(str))
    site_to_meta = {
        str(r["site_id"]): r for _, r in splits_df.iterrows()
    }

    # Reuse v3 rows where possible
    reuse_mask = v3_loop["site_id"].astype(str).isin(v4_sites) if len(v3_loop) else pd.Series([], dtype=bool)
    reused = v3_loop[reuse_mask].copy() if len(v3_loop) else pd.DataFrame()

    needed_sites = v4_sites - set(reused["site_id"].astype(str)) if len(reused) else v4_sites
    logger.info("Loop positions: reusing %d v3 rows, computing %d new",
                len(reused), len(needed_sites))

    # Compute new loop features
    new_rows = []
    t0 = time.time()
    for k, sid in enumerate(needed_sites, 1):
        seq = seqs.get(sid, "")
        if len(seq) != 201:
            continue
        try:
            db, mfe = _fold_features(seq)
        except Exception as e:
            logger.warning("fold failed for %s: %s", sid, e)
            continue
        feat = _analyze_site_structure(db, mfe, EDIT_POS)
        feat["site_id"] = sid
        meta = site_to_meta.get(sid)
        feat["enzyme"] = meta["enzyme"] if meta is not None else None
        feat["dataset_source"] = meta["dataset_source"] if meta is not None else None
        feat["label"] = int(meta["is_edited"]) if meta is not None else None
        feat["dot_bracket"] = db
        new_rows.append(feat)
        if k % 500 == 0:
            elapsed = time.time() - t0
            rate = k / elapsed
            remaining = (len(needed_sites) - k) / max(rate, 1e-6)
            logger.info("  loop %d/%d (%.0f/min, ~%.0fs left)",
                        k, len(needed_sites), rate * 60, remaining)

    new_df = pd.DataFrame(new_rows)
    out = pd.concat([reused, new_df], ignore_index=True) if len(reused) else new_df

    # Order rows to match splits ordering
    order = {s: i for i, s in enumerate(splits_df["site_id"].astype(str))}
    out["_order"] = out["site_id"].astype(str).map(order)
    out = out.sort_values("_order").drop(columns="_order").reset_index(drop=True)

    out.to_csv(OUT_LOOP, index=False)
    logger.info("Saved: %s (%d rows)", OUT_LOOP, len(out))


def build_structure_cache(seqs: dict) -> None:
    """Build per-site structure cache (delta features + mfes) for v4.

    Reuses entries from the v3 multi-enzyme cache where possible and computes
    new entries for v4 negatives.
    """
    import RNA  # noqa: F401  (early import)

    # Load v3 cache
    existing = {}
    if V3_STRUCT.exists():
        data = np.load(V3_STRUCT, allow_pickle=True)
        if "site_ids" in data:
            sids = list(data["site_ids"])
            delta = data["delta_features"]
            mfes = data["mfes"]
            mfes_ed = data["mfes_edited"]
            for i, sid in enumerate(sids):
                existing[str(sid)] = {
                    "delta_features": delta[i],
                    "mfe": float(mfes[i]),
                    "mfe_edited": float(mfes_ed[i]),
                }
        logger.info("Loaded v3 structure cache: %d entries", len(existing))

    needed = [(sid, seq) for sid, seq in seqs.items() if sid not in existing and len(seq) == 201]
    logger.info("Structure cache: reusing %d entries, computing %d new",
                len(existing), len(needed))

    results = dict(existing)
    if needed:
        t0 = time.time()
        for k, (sid, seq) in enumerate(needed, 1):
            try:
                results[sid] = _structure_features_for_site(seq)
            except Exception as e:
                logger.warning("struct failed for %s: %s", sid, e)
            if k % 200 == 0:
                elapsed = time.time() - t0
                rate = k / elapsed
                remaining = (len(needed) - k) / max(rate, 1e-6)
                logger.info("  struct %d/%d (%.1f/sec, ~%.0fs left)",
                            k, len(needed), rate, remaining)

    # Save in array format
    sids_arr = np.array(list(results.keys()), dtype=object)
    delta_arr = np.array([results[s]["delta_features"] for s in sids_arr], dtype=np.float32)
    mfes_arr = np.array([results[s]["mfe"] for s in sids_arr], dtype=np.float32)
    mfes_ed_arr = np.array([results[s]["mfe_edited"] for s in sids_arr], dtype=np.float32)
    OUT_STRUCT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_STRUCT,
        site_ids=sids_arr,
        delta_features=delta_arr,
        mfes=mfes_arr,
        mfes_edited=mfes_ed_arr,
    )
    logger.info("Saved: %s (%d entries)", OUT_STRUCT, len(results))


def _structure_features_for_site(seq: str) -> dict:
    """Compute delta features + mfes for a single 201-nt site (matches v3 cache)."""
    import RNA

    SEQ_LEN = 201
    seq_rna = seq.upper().replace("T", "U")
    n = len(seq_rna)
    center = n // 2

    md = RNA.md()
    md.temperature = 37.0

    fc = RNA.fold_compound(seq_rna, md)
    _, mfe = fc.mfe()
    fc.pf()
    bpp = np.array(fc.bpp())[1:n + 1, 1:n + 1]
    pp = np.clip(np.sum(bpp, axis=0) + np.sum(bpp, axis=1), 0, 1)
    acc = 1.0 - pp
    ent = np.zeros(n)
    for i in range(n):
        probs = bpp[i, :]
        probs = probs[probs > 1e-10]
        if len(probs) > 0:
            unpaired = max(0, 1.0 - np.sum(probs))
            if unpaired > 1e-10:
                probs = np.append(probs, unpaired)
            ent[i] = -np.sum(probs * np.log2(probs + 1e-10))

    # Edited (C->U at center)
    seq_list = list(seq_rna)
    if seq_list[center] == "C":
        seq_list[center] = "U"
    seq_ed = "".join(seq_list)

    fc_ed = RNA.fold_compound(seq_ed, md)
    _, mfe_ed = fc_ed.mfe()
    fc_ed.pf()
    bpp_ed = np.array(fc_ed.bpp())[1:n + 1, 1:n + 1]
    pp_ed = np.clip(np.sum(bpp_ed, axis=0) + np.sum(bpp_ed, axis=1), 0, 1)
    acc_ed = 1.0 - pp_ed

    # 7-dim delta features
    window = 10
    start = max(0, center - window)
    end = min(n, center + window + 1)
    dp = pp_ed - pp
    da = acc_ed - acc

    delta = np.zeros(7, dtype=np.float32)
    delta[0] = dp[center]
    delta[1] = da[center]
    # entropy delta at center
    ent_ed = np.zeros(n)
    for i in range(n):
        probs = bpp_ed[i, :]
        probs = probs[probs > 1e-10]
        if len(probs) > 0:
            unpaired = max(0, 1.0 - np.sum(probs))
            if unpaired > 1e-10:
                probs = np.append(probs, unpaired)
            ent_ed[i] = -np.sum(probs * np.log2(probs + 1e-10))
    de = ent_ed - ent
    delta[2] = de[center]
    delta[3] = float(mfe_ed) - float(mfe)
    delta[4] = float(np.mean(dp[start:end]))
    delta[5] = float(np.mean(da[start:end]))
    delta[6] = float(np.std(dp[start:end]))

    return {
        "delta_features": delta,
        "mfe": float(mfe),
        "mfe_edited": float(mfe_ed),
    }


if __name__ == "__main__":
    main()
