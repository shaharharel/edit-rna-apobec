#!/usr/bin/env python
"""Build v4 multi-enzyme datasets with two trinucleotide-matched negative sets.

This script produces TWO parallel v4 datasets sharing the same set of positives:

  v4_cancer_matched: negatives sampled to match the TCGA + PCAWG-coding pan-cancer
                     C>T mutation trinucleotide distribution (transfer claim).
  v4_cds_unbiased:   negatives sampled to match the genome CDS-C trinucleotide
                     distribution (predictor claim).

Key design points:
  - APOBEC1 sites are excluded (v3 enzyme=='Neither', n=206) because they have
    no DNA-editing analog in cancer.
  - n_neg target = n_pos = 7,358 (1:1 ratio).
  - Random seed = 20260427.

Outputs (per version):
  - data/processed/multi_enzyme/splits_multi_enzyme_v4_<ver>.csv
  - data/processed/multi_enzyme/multi_enzyme_sequences_v4_<ver>.json
  - data/processed/multi_enzyme/loop_position_per_site_v4_<ver>.csv
  - data/processed/embeddings/structure_cache_multi_enzyme_v4_<ver>.npz

Usage:
    conda run -n quris python scripts/multi_enzyme/build_v4_datasets.py
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.apobec_negatives import generate_negatives_cancer_matched

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
ME_DIR = DATA_DIR / "processed/multi_enzyme"
EMB_DIR = DATA_DIR / "processed/embeddings"

# Inputs
V3_SPLITS = ME_DIR / "splits_multi_enzyme_v3_with_negatives.csv"
V3_SEQS = ME_DIR / "multi_enzyme_sequences_v3_with_negatives.json"
V3_LOOP = ME_DIR / "loop_position_per_site_v3.csv"
# v3 structure cache historically lives under multi_enzyme/, not embeddings/
V3_STRUCT = ME_DIR / "structure_cache_multi_enzyme_v3.npz"

CANCER_TRINUC_CSV = ME_DIR / "cancer_ct_trinuc_distribution.csv"
CDS_TRINUC_CSV = ME_DIR / "cds_c_trinuc_distribution.csv"

GENOME_HG19 = DATA_DIR / "raw/genomes/hg19.fa"
GENOME_HG38 = DATA_DIR / "raw/genomes/hg38.fa"

SEED = 20260427
EXCLUDED_ENZYMES = {"Neither"}  # APOBEC1: no DNA-editing analog
EDIT_POS = 100


# ---------------------------------------------------------------------------
# Filter v3 -> v4 positives
# ---------------------------------------------------------------------------

def load_v4_positives(v3_df: pd.DataFrame) -> pd.DataFrame:
    pos = v3_df[v3_df["is_edited"] == 1].copy()
    pos = pos[~pos["enzyme"].isin(EXCLUDED_ENZYMES)].reset_index(drop=True)
    return pos


def trinuc_of_seq(seq: str) -> str:
    if not seq or len(seq) < 102:
        return ""
    s = seq.upper().replace("U", "T")
    return s[99:102]


# ---------------------------------------------------------------------------
# Loop position
# ---------------------------------------------------------------------------

def _fold_for_loop(seq: str):
    import RNA
    seq_rna = seq.upper().replace("T", "U")
    md = RNA.md(); md.temperature = 37.0
    fc = RNA.fold_compound(seq_rna, md)
    db, mfe = fc.mfe()
    return db, float(mfe)


def _analyze_site(dot_bracket: str, mfe: float, pos: int = EDIT_POS) -> dict:
    """Replicates scripts/multi_enzyme/generate_loop_positions.py / build_v4_cancer_matched_dataset.py."""
    n = len(dot_bracket)
    result = {
        "is_unpaired": None, "loop_type": None, "loop_size": None,
        "dist_to_left_boundary": None, "dist_to_right_boundary": None,
        "dist_to_nearest_stem": None, "relative_loop_position": None,
        "dist_to_apex": None, "left_stem_length": None, "right_stem_length": None,
        "max_adjacent_stem_length": None, "dist_to_junction": None,
        "local_unpaired_fraction": None, "mfe": float(mfe),
    }
    if not dot_bracket or pos >= n or pos < 0:
        return result
    is_unpaired = dot_bracket[pos] == "."
    result["is_unpaired"] = bool(is_unpaired)
    window = 10
    w_start = max(0, pos - window); w_end = min(n, pos + window + 1)
    local = dot_bracket[w_start:w_end]
    result["local_unpaired_fraction"] = sum(1 for c in local if c == ".") / max(len(local), 1)

    def _stem_length(boundary_pos, direction):
        if boundary_pos < 0 or boundary_pos >= n or dot_bracket[boundary_pos] not in "()":
            return 0
        cnt = 0
        if direction == "left":
            i = boundary_pos
            while i >= 0 and dot_bracket[i] in "()":
                cnt += 1; i -= 1
        else:
            i = boundary_pos
            while i < n and dot_bracket[i] in "()":
                cnt += 1; i += 1
        return cnt

    if is_unpaired:
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
        if left < 0 or right >= n:
            loop_type = "external"
        else:
            lc, rc = dot_bracket[left], dot_bracket[right]
            if lc == "(" and rc == ")": loop_type = "hairpin"
            elif lc == ")" and rc == "(": loop_type = "multiloop"
            elif lc == "(" and rc == "(": loop_type = "bulge_left"
            elif lc == ")" and rc == ")": loop_type = "bulge_right"
            else: loop_type = "other"
        result["loop_size"] = loop_size
        result["dist_to_left_boundary"] = dist_left
        result["dist_to_right_boundary"] = dist_right
        result["dist_to_nearest_stem"] = min(dist_left, dist_right)
        result["relative_loop_position"] = rel_pos
        result["dist_to_apex"] = dist_apex
        result["loop_type"] = loop_type
        result["left_stem_length"] = _stem_length(left, "left")
        result["right_stem_length"] = _stem_length(right, "right")
        result["max_adjacent_stem_length"] = max(result["left_stem_length"], result["right_stem_length"])
        result["dist_to_junction"] = result["dist_to_nearest_stem"]
    else:
        dl = 0; i = pos - 1
        while i >= 0 and dot_bracket[i] in "()":
            dl += 1; i -= 1
        dr = 0; j = pos + 1
        while j < n and dot_bracket[j] in "()":
            dr += 1; j += 1
        result["dist_to_junction"] = min(dl, dr)
        sl = _stem_length(pos, "left"); sr = _stem_length(pos, "right")
        result["left_stem_length"] = sl; result["right_stem_length"] = sr
        result["max_adjacent_stem_length"] = max(sl, sr)
    return result


def build_loop_positions(splits_df: pd.DataFrame, seqs: dict, out_path: Path):
    """Reuse v3 loop rows for shared site_ids; compute new ones for v4 negatives."""
    if V3_LOOP.exists():
        v3_loop = pd.read_csv(V3_LOOP)
    else:
        v3_loop = pd.DataFrame()
    v4_sites = set(splits_df["site_id"].astype(str))
    site_to_meta = {str(r["site_id"]): r for _, r in splits_df.iterrows()}

    if len(v3_loop):
        reuse_mask = v3_loop["site_id"].astype(str).isin(v4_sites)
        reused = v3_loop[reuse_mask].copy()
    else:
        reused = pd.DataFrame()

    needed = v4_sites - set(reused["site_id"].astype(str)) if len(reused) else v4_sites
    logger.info("  loop: reusing %d v3 rows, computing %d new", len(reused), len(needed))

    new_rows = []
    t0 = time.time()
    for k, sid in enumerate(needed, 1):
        seq = seqs.get(sid, "")
        if len(seq) != 201:
            continue
        try:
            db, mfe = _fold_for_loop(seq)
        except Exception as e:
            logger.warning("fold failed for %s: %s", sid, e)
            continue
        feat = _analyze_site(db, mfe, EDIT_POS)
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
            remaining = (len(needed) - k) / max(rate, 1e-6)
            logger.info("    loop %d/%d (%.0f/min, ~%.0fs left)", k, len(needed), rate * 60, remaining)
    new_df = pd.DataFrame(new_rows)
    out = pd.concat([reused, new_df], ignore_index=True) if len(reused) else new_df
    order = {s: i for i, s in enumerate(splits_df["site_id"].astype(str))}
    out["_order"] = out["site_id"].astype(str).map(order)
    out = out.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    out.to_csv(out_path, index=False)
    logger.info("  saved: %s (%d rows)", out_path, len(out))


# ---------------------------------------------------------------------------
# Structure cache
# ---------------------------------------------------------------------------

def _structure_features_for_site(seq: str) -> dict:
    import RNA
    seq_rna = seq.upper().replace("T", "U")
    n = len(seq_rna)
    center = n // 2
    md = RNA.md(); md.temperature = 37.0
    fc = RNA.fold_compound(seq_rna, md)
    _, mfe = fc.mfe()
    fc.pf()
    bpp = np.array(fc.bpp())[1:n + 1, 1:n + 1]
    pp = np.clip(np.sum(bpp, axis=0) + np.sum(bpp, axis=1), 0, 1)
    acc = 1.0 - pp
    ent = np.zeros(n)
    for i in range(n):
        probs = bpp[i, :]; probs = probs[probs > 1e-10]
        if len(probs) > 0:
            unpaired = max(0, 1.0 - np.sum(probs))
            if unpaired > 1e-10:
                probs = np.append(probs, unpaired)
            ent[i] = -np.sum(probs * np.log2(probs + 1e-10))
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
    window = 10
    start = max(0, center - window); end = min(n, center + window + 1)
    dp = pp_ed - pp; da = acc_ed - acc
    delta = np.zeros(7, dtype=np.float32)
    delta[0] = dp[center]
    delta[1] = da[center]
    ent_ed = np.zeros(n)
    for i in range(n):
        probs = bpp_ed[i, :]; probs = probs[probs > 1e-10]
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
    return {"delta_features": delta, "mfe": float(mfe), "mfe_edited": float(mfe_ed)}


def build_structure_cache(seqs: dict, out_path: Path, shared_cache: Dict[str, dict]):
    """Build per-site structure cache. Reuses v3 cache + previously-computed entries
    in shared_cache (dict keyed by site_id), and updates shared_cache with new entries.
    """
    import RNA  # noqa
    existing = dict(shared_cache)
    if V3_STRUCT.exists():
        data = np.load(V3_STRUCT, allow_pickle=True)
        if "site_ids" in data:
            sids = list(data["site_ids"])
            delta = data["delta_features"]; mfes = data["mfes"]; mfes_ed = data["mfes_edited"]
            for i, sid in enumerate(sids):
                key = str(sid)
                if key not in existing:
                    existing[key] = {
                        "delta_features": delta[i],
                        "mfe": float(mfes[i]),
                        "mfe_edited": float(mfes_ed[i]),
                    }
    needed = [(sid, seq) for sid, seq in seqs.items() if sid not in existing and len(seq) == 201]
    logger.info("  struct: %d in cache, %d to compute", len(existing), len(needed))
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
                logger.info("    struct %d/%d (%.1f/sec, ~%.0fs left)", k, len(needed), rate, remaining)
    # Write only entries present in seqs (so the cache matches its split)
    keep_ids = [s for s in seqs if s in results]
    sids_arr = np.array(keep_ids, dtype=object)
    delta_arr = np.array([results[s]["delta_features"] for s in keep_ids], dtype=np.float32)
    mfes_arr = np.array([results[s]["mfe"] for s in keep_ids], dtype=np.float32)
    mfes_ed_arr = np.array([results[s]["mfe_edited"] for s in keep_ids], dtype=np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, site_ids=sids_arr, delta_features=delta_arr,
                        mfes=mfes_arr, mfes_edited=mfes_ed_arr)
    logger.info("  saved: %s (%d entries)", out_path, len(keep_ids))
    # Update shared_cache for next version
    shared_cache.update(results)


# ---------------------------------------------------------------------------
# Main pipeline (per version)
# ---------------------------------------------------------------------------

def run_version(version: str, target_csv: Path, n_negatives: int,
                v3_df: pd.DataFrame, pos: pd.DataFrame, v3_seqs: dict,
                shared_struct_cache: Dict[str, dict]):
    """Build splits + sequences + loop + structure cache for one v4 version."""
    logger.info("=== Building v4_%s ===", version)

    out_splits = ME_DIR / f"splits_multi_enzyme_v4_{version}.csv"
    out_seqs = ME_DIR / f"multi_enzyme_sequences_v4_{version}.json"
    out_loop = ME_DIR / f"loop_position_per_site_v4_{version}.csv"
    out_struct = EMB_DIR / f"structure_cache_multi_enzyme_v4_{version}.npz"

    # Build target distribution
    target_df = pd.read_csv(target_csv)
    target_fracs = dict(zip(target_df["trinuc"], target_df["fraction"]))
    logger.info("Target distribution loaded from %s", target_csv.name)

    # v4 sequences: positives only first
    v4_seqs = {}
    for sid in pos["site_id"]:
        sid = str(sid)
        if sid in v3_seqs:
            v4_seqs[sid] = v3_seqs[sid]
        else:
            logger.warning("Positive %s missing from v3 sequences", sid)

    # Excluded sites: v3 positives + v3 negatives (so v4 negs are fresh)
    known_sites = set()
    for _, row in v3_df.iterrows():
        if pd.notna(row.get("chr")) and pd.notna(row.get("start")):
            known_sites.add((str(row["chr"]), int(row["start"])))

    # Generate negatives
    new_seqs: Dict[str, str] = {}
    t0 = time.time()
    neg_df = generate_negatives_cancer_matched(
        positives_df=pos,
        genomes={"hg19": GENOME_HG19, "hg38": GENOME_HG38},
        cancer_trinuc_fractions=target_fracs,
        n_negatives=n_negatives,
        output_seqs=new_seqs,
        known_sites=known_sites,
        search_window=5000,
        seed=SEED,
    )
    logger.info("  generated %d negatives in %.1fs", len(neg_df), time.time() - t0)

    # Tag source_type with the version so downstream code can tell apart
    if "source_type" in neg_df.columns:
        neg_df["source_type"] = f"negative_control_v4_{version}"
    if "dataset_source" in neg_df.columns:
        # Replace generic "_neg_v4" suffix with version-specific suffix
        neg_df["dataset_source"] = neg_df["dataset_source"].astype(str).str.replace(
            "_neg_v4", f"_neg_v4_{version}", regex=False
        )

    # Compute trinuc on positives (for record-keeping)
    pos_for_concat = pos.copy()
    if "trinuc" not in pos_for_concat.columns:
        pos_for_concat["trinuc"] = pos_for_concat["site_id"].astype(str).map(
            lambda s: trinuc_of_seq(v3_seqs.get(s, ""))
        )
    if "trinuc" not in neg_df.columns:
        neg_df["trinuc"] = neg_df["site_id"].astype(str).map(
            lambda s: trinuc_of_seq(new_seqs.get(s, ""))
        )

    combined = pd.concat([pos_for_concat, neg_df], ignore_index=True)
    cols = ["site_id", "chr", "start", "end", "strand", "enzyme", "dataset_source",
            "coordinate_system", "editing_rate", "is_edited", "flanking_seq",
            "seq_center", "source_type", "trinuc"]
    final_cols = [c for c in cols if c in combined.columns]
    extra = [c for c in combined.columns if c not in final_cols]
    combined = combined[final_cols + extra]

    out_splits.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_splits, index=False)
    logger.info("  saved: %s (%d rows: %d pos, %d neg)", out_splits, len(combined),
                int((combined["is_edited"] == 1).sum()),
                int((combined["is_edited"] == 0).sum()))

    # Merge sequences and save
    v4_seqs.update(new_seqs)
    with open(out_seqs, "w") as f:
        json.dump(v4_seqs, f)
    logger.info("  saved: %s (%d seqs)", out_seqs, len(v4_seqs))

    # Loop positions
    build_loop_positions(combined, v4_seqs, out_loop)

    # Structure cache
    build_structure_cache(v4_seqs, out_struct, shared_struct_cache)

    # Quick distribution report
    neg_trinucs = neg_df["trinuc"].value_counts().to_dict()
    total = sum(neg_trinucs.values())
    logger.info("  neg trinuc distribution (target vs actual):")
    for tri in sorted(target_fracs.keys()):
        tgt = target_fracs[tri] * 100
        act = 100 * neg_trinucs.get(tri, 0) / total if total else 0
        logger.info("    %s: target %5.2f%%  actual %5.2f%%  (delta %+.2fpp)",
                    tri, tgt, act, act - tgt)


def main():
    for path in [V3_SPLITS, V3_SEQS, CANCER_TRINUC_CSV, CDS_TRINUC_CSV,
                 GENOME_HG19, GENOME_HG38]:
        if not path.exists():
            logger.error("Missing input: %s", path)
            sys.exit(1)

    logger.info("Loading v3 ...")
    v3 = pd.read_csv(V3_SPLITS)
    pos = load_v4_positives(v3)
    n_pos = len(pos)
    n_neg_target = n_pos
    logger.info("v4 positives (excl Neither/APOBEC1): %d  (target n_neg = %d)", n_pos, n_neg_target)

    with open(V3_SEQS) as f:
        v3_seqs = json.load(f)
    logger.info("v3 sequences: %d", len(v3_seqs))

    # Shared structure cache to avoid recomputing positive entries between versions
    shared_struct_cache: Dict[str, dict] = {}

    run_version("cancer_matched", CANCER_TRINUC_CSV, n_neg_target,
                v3, pos, v3_seqs, shared_struct_cache)
    run_version("cds_unbiased", CDS_TRINUC_CSV, n_neg_target,
                v3, pos, v3_seqs, shared_struct_cache)

    logger.info("=== v4 build complete ===")


if __name__ == "__main__":
    main()
