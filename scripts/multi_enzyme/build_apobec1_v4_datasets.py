#!/usr/bin/env python3
"""Build v4 APOBEC1 datasets with trinucleotide-matched negatives (cancer / CDS).

The original APOBEC1 v3 dataset (data/processed/apobec1/apobec1_v1_with_negatives.csv)
matched negatives only at the 5'-T dinucleotide (TC=46%, CC=3.5%, other=50.5%) to
positives. That's the same negative-control bias the v4 multi-enzyme work replaced
with trinucleotide matching.

This script produces TWO parallel APOBEC1 v4 datasets sharing the same 484
mouse positives:

  apobec1_v4_cancer: negatives sampled from mm9 to match the human pan-cancer
                     C>T mutation trinucleotide distribution.
  apobec1_v4_cds:    negatives sampled from mm9 to match the human CDS-C
                     trinucleotide distribution.

Rationale: when the head is applied to the human exome panel, we want it
unbiased w.r.t. the cancer/CDS trinuc distribution. Even though positives are
mouse, the *negatives*' trinuc target is the same human distribution used for
the multi-enzyme v4 (predictor / transfer claims).

Outputs (per version, stored under data/raw/apobec1/v4/):
  - apobec1_v4_<ver>_with_negatives.csv  (positives + new v4 negatives)
  - apobec1_v4_<ver>_sequences.json
  - apobec1_v4_<ver>_trinuc_summary.csv  (pos/neg/target distributions)

Random seed: 20260427.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.apobec_negatives import (  # noqa: E402
    FLANK_SIZE,
    CENTER,
    extract_from_genome,
    _ALL_TRINUCS,
    _revcomp,
    _trinuc_distance,
    _fallback_order,
)

DATA_DIR = PROJECT_ROOT / "data"
APOBEC1_PROC = DATA_DIR / "processed" / "apobec1"
V4_DIR = DATA_DIR / "raw" / "apobec1" / "v4"
V4_DIR.mkdir(parents=True, exist_ok=True)

GENOME_MM9 = DATA_DIR / "raw" / "genomes" / "mm9.fa"
ME_DIR = DATA_DIR / "processed" / "multi_enzyme"
CANCER_TRINUC_CSV = ME_DIR / "cancer_ct_trinuc_distribution.csv"
CDS_TRINUC_CSV = ME_DIR / "cds_c_trinuc_distribution.csv"

V3_SPLITS = APOBEC1_PROC / "apobec1_v1_with_negatives.csv"
V3_SEQS = APOBEC1_PROC / "apobec1_v1_sequences.json"

SEED = 20260427
SEARCH_WINDOW = 5000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def trinuc_of_seq(seq: str) -> str:
    if not seq or len(seq) < 102:
        return ""
    s = seq.upper().replace("U", "T")
    return s[99:102]


def generate_negatives_mm9_trinuc_matched(
    positives_df: pd.DataFrame,
    genome_path: Path,
    target_trinuc_fractions: Dict[str, float],
    n_negatives: int,
    output_seqs: Dict[str, str],
    known_sites_xy: set,
    search_window: int = SEARCH_WINDOW,
    seed: int = SEED,
    label: str = "v4",
) -> pd.DataFrame:
    """mm9 trinucleotide-matched negative generation.

    Mirrors src.data.apobec_negatives.generate_negatives_cancer_matched but:
      - Uses a single mm9 genome (no hg19/hg38 dispatch).
      - Coordinate system is "mm9".
      - Excludes the same `known_sites_xy` set ({(chrom, pos)}) of positives.

    Args:
      positives_df: rows with chr, start, strand columns (mouse mm9).
      genome_path: path to mm9.fa
      target_trinuc_fractions: dict {trinuc: fraction} for the target distribution
      n_negatives: total negatives to draw
      output_seqs: dict to append {site_id: rna_seq}
      known_sites_xy: existing (chrom, pos) tuples to exclude
      label: tag for source_type column
    """
    from pyfaidx import Fasta

    genome = Fasta(str(genome_path))
    rng = random.Random(seed)
    used_sites = set(known_sites_xy)

    # 1) Collect candidate Cs binned by trinucleotide
    candidates_by_trinuc: Dict[str, list] = {t: [] for t in _ALL_TRINUCS}

    valid_positives = positives_df[
        positives_df["chr"].notna() & positives_df["start"].notna()
    ].copy()
    logger.info("[%s] scanning near %d mouse positives (window=%d)",
                label, len(valid_positives), search_window)

    n_processed = 0
    for _, row in valid_positives.iterrows():
        chrom = str(row["chr"])
        pos = int(row["start"])
        strand = str(row["strand"])
        ds = str(row.get("dataset_source", "apobec1"))

        if chrom not in genome:
            continue
        chrom_len = len(genome[chrom])

        search_start = max(1, pos - search_window)
        search_end = min(chrom_len, pos + search_window)
        if search_end - search_start < 3:
            continue

        slice_start = max(0, search_start - 1)
        slice_end = min(chrom_len, search_end + 1)
        region_seq = str(genome[chrom][slice_start:slice_end]).upper()

        target_base = "C" if strand == "+" else "G"
        for i in range(1, len(region_seq) - 1):
            base = region_seq[i]
            if base != target_base:
                continue
            genome_pos = slice_start + i + 1  # 1-based
            if (chrom, genome_pos) in used_sites or genome_pos == pos:
                continue
            pos_0 = genome_pos - 1
            if pos_0 < FLANK_SIZE or pos_0 + FLANK_SIZE >= chrom_len:
                continue
            if strand == "+":
                triplet = region_seq[i - 1:i + 2]
            else:
                triplet = _revcomp(region_seq[i - 1:i + 2])
            if len(triplet) != 3 or "N" in triplet or triplet[1] != "C":
                continue

            candidates_by_trinuc[triplet].append({
                "chrom": chrom,
                "genome_pos": genome_pos,
                "strand": strand,
                "dataset_source": ds,
                "trinuc": triplet,
            })
        n_processed += 1
        if n_processed % 100 == 0:
            tot = sum(len(v) for v in candidates_by_trinuc.values())
            logger.info("  [%s] scanned %d/%d, candidates=%d",
                        label, n_processed, len(valid_positives), tot)

    total_cands = sum(len(v) for v in candidates_by_trinuc.values())
    logger.info("[%s] total candidates: %d across 16 bins", label, total_cands)
    for t in _ALL_TRINUCS:
        logger.info("  [%s] %s: %d", label, t, len(candidates_by_trinuc[t]))

    # 2) Compute per-bin quotas
    fractions = {t: max(0.0, float(target_trinuc_fractions.get(t, 0.0)))
                 for t in _ALL_TRINUCS}
    fr_sum = sum(fractions.values())
    if fr_sum <= 0:
        raise ValueError("target_trinuc_fractions sum to 0; cannot build quotas")
    fractions = {t: v / fr_sum for t, v in fractions.items()}

    raw = {t: fractions[t] * n_negatives for t in _ALL_TRINUCS}
    floors = {t: int(np.floor(raw[t])) for t in _ALL_TRINUCS}
    remainder = n_negatives - sum(floors.values())
    fr_parts = sorted(_ALL_TRINUCS,
                      key=lambda t: (raw[t] - floors[t]), reverse=True)
    targets = dict(floors)
    for t in fr_parts[:max(0, remainder)]:
        targets[t] += 1

    logger.info("[%s] per-bin quotas (target -> available):", label)
    for t in _ALL_TRINUCS:
        logger.info("  [%s] %s: target=%d available=%d",
                    label, t, targets[t], len(candidates_by_trinuc[t]))

    # 3) Sample
    for t in _ALL_TRINUCS:
        rng.shuffle(candidates_by_trinuc[t])
    selected = []
    take_idx = {t: 0 for t in _ALL_TRINUCS}
    bin_overflow = {t: 0 for t in _ALL_TRINUCS}
    leftover_target = 0
    for t in _ALL_TRINUCS:
        avail = len(candidates_by_trinuc[t])
        want = targets[t]
        take = min(want, avail)
        if take > 0:
            selected.extend(candidates_by_trinuc[t][:take])
            take_idx[t] = take
        if want > avail:
            bin_overflow[t] = want - avail
            leftover_target += want - avail

    if leftover_target > 0:
        logger.info("[%s] %d slots to redistribute from depleted bins",
                    label, leftover_target)
        for t in _ALL_TRINUCS:
            need = bin_overflow[t]
            if need <= 0:
                continue
            for fb in _fallback_order(t):
                if need <= 0:
                    break
                avail = len(candidates_by_trinuc[fb]) - take_idx[fb]
                if avail <= 0:
                    continue
                grab = min(need, avail)
                selected.extend(candidates_by_trinuc[fb][take_idx[fb]:take_idx[fb] + grab])
                take_idx[fb] += grab
                need -= grab
            if need > 0:
                logger.warning("  [%s] could not fully fill %s: short %d",
                               label, t, need)

    rng.shuffle(selected)
    logger.info("[%s] %d candidates selected (target=%d)",
                label, len(selected), n_negatives)

    # 4) Extract sequences and build rows
    negatives = []
    for cand in selected:
        chrom = cand["chrom"]
        genome_pos = cand["genome_pos"]
        strand = cand["strand"]
        if (chrom, genome_pos) in used_sites:
            continue
        rna_seq = extract_from_genome(genome, chrom, genome_pos, strand)
        if rna_seq is None:
            continue
        site_id = f"{chrom}:{genome_pos}:{strand}"
        if site_id in output_seqs:
            site_id = f"{chrom}:{genome_pos}:{strand}:mm9"
            if site_id in output_seqs:
                continue
        used_sites.add((chrom, genome_pos))
        output_seqs[site_id] = rna_seq

        negatives.append({
            "site_id": site_id,
            "chr": chrom,
            "start": genome_pos,
            "end": genome_pos,
            "strand": strand,
            "enzyme": "APOBEC1",
            "dataset_source": cand["dataset_source"] + f"_neg_{label}",
            "tissue": "intestine",
            "species": "mouse",
            "build": "mm9",
            "is_edited": 0,
            "ko_validated": 0,
            "editing_rate": 0.0,
            "coordinate_system": "mm9",
            "source_type": f"negative_control_{label}",
            "flanking_seq": rna_seq,
            "seq_center": CENTER,
            "trinuc": cand["trinuc"],
        })

    neg_df = pd.DataFrame(negatives)
    logger.info("[%s] produced %d negative rows", label, len(neg_df))
    return neg_df


def build_one_version(version_label: str, target_csv: Path) -> None:
    """Build a single v4 dataset version.

    version_label: "v4_cancer" or "v4_cds"
    target_csv: trinuc distribution CSV (cancer_ct or cds_c)
    """
    logger.info("=" * 70)
    logger.info("Building APOBEC1 %s", version_label)
    logger.info("=" * 70)

    # 1) Load positives from v3 dataset (positives are unchanged)
    splits_v3 = pd.read_csv(V3_SPLITS)
    pos = splits_v3[splits_v3["is_edited"] == 1].copy().reset_index(drop=True)
    logger.info("Loaded %d positives from %s", len(pos), V3_SPLITS.name)

    with open(V3_SEQS) as f:
        seqs_v3 = json.load(f)

    # Pull positive sequences only
    output_seqs: Dict[str, str] = {}
    for sid in pos["site_id"]:
        if sid in seqs_v3:
            output_seqs[sid] = seqs_v3[sid]
    logger.info("Loaded %d positive sequences", len(output_seqs))

    # 2) Compute pos trinuc summary
    pos_trinuc = Counter()
    for sid in pos["site_id"]:
        s = output_seqs.get(sid, "")
        t = trinuc_of_seq(s)
        if t:
            pos_trinuc[t] += 1
    pos_total = sum(pos_trinuc.values())
    logger.info("Positives trinuc top 5: %s",
                sorted(pos_trinuc.items(), key=lambda kv: -kv[1])[:5])

    # 3) Load target distribution
    tdf = pd.read_csv(target_csv)
    target_fracs = dict(zip(tdf["trinuc"], tdf["fraction"]))

    # 4) Generate negatives
    n_neg = len(pos)  # 1:1 ratio
    known_xy = {(r["chr"], int(r["start"])) for _, r in pos.iterrows()}
    neg_df = generate_negatives_mm9_trinuc_matched(
        positives_df=pos,
        genome_path=GENOME_MM9,
        target_trinuc_fractions=target_fracs,
        n_negatives=n_neg,
        output_seqs=output_seqs,
        known_sites_xy=known_xy,
        seed=SEED,
        label=version_label,
    )

    # 5) Verify negative trinuc distribution
    neg_trinuc = Counter()
    for sid in neg_df["site_id"]:
        s = output_seqs.get(sid, "")
        t = trinuc_of_seq(s)
        if t:
            neg_trinuc[t] += 1
    neg_total = sum(neg_trinuc.values())
    logger.info("Negatives trinuc top 5: %s",
                sorted(neg_trinuc.items(), key=lambda kv: -kv[1])[:5])

    # 6) Combined splits
    # Re-add columns to positives that match negative df schema
    pos_for_concat = pos.copy()
    if "coordinate_system" not in pos_for_concat.columns:
        pos_for_concat["coordinate_system"] = "mm9"
    if "trinuc" not in pos_for_concat.columns:
        pos_for_concat["trinuc"] = pos_for_concat["site_id"].apply(
            lambda s: trinuc_of_seq(output_seqs.get(s, ""))
        )
    if "source_type" not in pos_for_concat.columns:
        pos_for_concat["source_type"] = "positive"
    if "species" not in pos_for_concat.columns:
        pos_for_concat["species"] = "mouse"
    if "build" not in pos_for_concat.columns:
        pos_for_concat["build"] = "mm9"

    common = sorted(set(pos_for_concat.columns) & set(neg_df.columns))
    if "site_id" not in common:
        raise ValueError("site_id missing from union schema")
    pos_for_concat = pos_for_concat[common].copy()
    neg_for_concat = neg_df[common].copy()
    combined = pd.concat([pos_for_concat, neg_for_concat], ignore_index=True)

    out_splits = V4_DIR / f"apobec1_{version_label}_with_negatives.csv"
    out_seqs = V4_DIR / f"apobec1_{version_label}_sequences.json"
    out_summary = V4_DIR / f"apobec1_{version_label}_trinuc_summary.csv"

    combined.to_csv(out_splits, index=False)
    with open(out_seqs, "w") as f:
        json.dump(output_seqs, f)

    # Trinuc summary CSV
    rows = []
    for t in _ALL_TRINUCS:
        pos_f = pos_trinuc.get(t, 0) / max(1, pos_total)
        neg_f = neg_trinuc.get(t, 0) / max(1, neg_total)
        tgt_f = target_fracs.get(t, 0.0)
        rows.append({"trinuc": t,
                     "pos_count": pos_trinuc.get(t, 0),
                     "pos_fraction": pos_f,
                     "neg_count": neg_trinuc.get(t, 0),
                     "neg_fraction": neg_f,
                     "target_fraction": tgt_f})
    pd.DataFrame(rows).to_csv(out_summary, index=False)

    logger.info("Wrote %s (%d rows: %d pos / %d neg)",
                out_splits, len(combined),
                int((combined["is_edited"] == 1).sum()),
                int((combined["is_edited"] == 0).sum()))
    logger.info("Wrote %s (%d sequences)", out_seqs, len(output_seqs))
    logger.info("Wrote %s", out_summary)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=["v4_cancer", "v4_cds", "all"],
                    default="all")
    args = ap.parse_args()

    if not GENOME_MM9.exists():
        logger.error("mm9 genome not found at %s", GENOME_MM9)
        sys.exit(1)
    if not CANCER_TRINUC_CSV.exists():
        logger.error("Cancer trinuc CSV not found at %s", CANCER_TRINUC_CSV)
        sys.exit(1)
    if not CDS_TRINUC_CSV.exists():
        logger.error("CDS trinuc CSV not found at %s", CDS_TRINUC_CSV)
        sys.exit(1)

    if args.version in ("v4_cancer", "all"):
        build_one_version("v4_cancer", CANCER_TRINUC_CSV)
    if args.version in ("v4_cds", "all"):
        build_one_version("v4_cds", CDS_TRINUC_CSV)


if __name__ == "__main__":
    main()
