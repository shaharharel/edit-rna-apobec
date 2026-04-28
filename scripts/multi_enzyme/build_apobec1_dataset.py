#!/usr/bin/env python3
"""Build APOBEC1 training dataset from published KO-validated mouse sites.

Sources (see data/raw/apobec1/RUN_LOG.txt for what was accessible):
  - Blanc et al. 2014 Genome Biology Additional File 1 (PDF)    — mm9 intestine/liver
  - Rosenberg et al. 2011 NSMB Supp Table S3 (PDF page 8)       — mm9 3'UTR sites
  - Rosenberg et al. 2011 NSMB Supp Table S4 (XLS)              — mm9 predicted 3'UTR sites

Outputs:
  - data/processed/apobec1/apobec1_positives.csv           — unified coord table
  - data/processed/apobec1/apobec1_v1_with_negatives.csv   — pos + motif-matched neg
  - data/processed/apobec1/apobec1_v1_sequences.json       — 201-nt mm9 sequences
  - data/processed/apobec1/dataset_summary.json            — counts by source / context

Usage:
    conda run -n quris python scripts/multi_enzyme/build_apobec1_dataset.py
"""

from __future__ import annotations

import json
import logging
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.apobec_negatives import extract_from_genome  # noqa: E402

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "apobec1"
GENOME_MM9 = PROJECT_ROOT / "data" / "raw" / "genomes" / "mm9.fa"
PROC_DIR = PROJECT_ROOT / "data" / "processed" / "apobec1"
PROC_DIR.mkdir(parents=True, exist_ok=True)

FLANK = 100  # ±100 nt → 201-nt window centered on edited C
CENTER = 100
NEG_RATIO = 1  # 1:1 negatives
NEG_SEARCH_WINDOW = 5000
SEED = 42

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Source parsers
# ---------------------------------------------------------------------------

def parse_blanc_2014_pdf() -> pd.DataFrame:
    """Extract (chr, pos, strand) tuples from Blanc 2014 Genome Biology supp PDF.

    The PDF has several tables (1A-1E, 3A-3C, 7, 8, 14) listing Apobec-1
    KO-validated editing sites in mouse (mm9). We scrape any line with
    format '<chr> <pos> (+|-)' or 'Chr<chr>:<pos>'.
    """
    import pymupdf

    pdf_path = RAW_DIR / "blanc_2014_additional1.pdf"
    if not pdf_path.exists():
        logger.warning("Blanc 2014 PDF not found at %s", pdf_path)
        return pd.DataFrame()

    pdf = pymupdf.open(str(pdf_path))

    # Standalone strand-aware pattern: "<chr> <pos> (+|-)"
    pat_standalone = re.compile(
        r"(?:^|\s)(\d{1,2}|X|Y)\s+(\d{5,})\s*\((\+|\-)\)"
    )
    # Inline pattern: "Chr<chr>: <pos>" (no strand)
    pat_inline = re.compile(
        r"[Cc]hr\s?(\d{1,2}|X|Y)\s*:\s*(\d{5,})"
    )

    strand_known = set()
    strand_unknown = []
    page_sources = {}  # sid -> (src_table, tissue)

    # Map page ranges to tissue for source labeling
    page_to_src = {
        0: ("Blanc2014_T1A_intestine_exonic", "intestine"),
        1: ("Blanc2014_T1B_intestine", "intestine"),
        2: ("Blanc2014_T1C_cohort", "intestine"),
        3: ("Blanc2014_T1D_enterocyte", "intestine"),
        4: ("Blanc2014_T1E_liver_exonic", "liver"),
        5: ("Blanc2014_T2_intestine_mooring", "intestine"),
        6: ("Blanc2014_T2_intestine_mooring", "intestine"),
        7: ("Blanc2014_T3A_intestine", "intestine"),
        8: ("Blanc2014_T3B_liver", "liver"),
        9: ("Blanc2014_T3C_hyperediting", "liver"),
        10: ("Blanc2014_T4_liver_mooring", "liver"),
        14: ("Blanc2014_T7_intestine_miRNA", "intestine"),
        16: ("Blanc2014_T8_liver_miRNA", "liver"),
        42: ("Blanc2014_T12_intestine_primers", "intestine"),
        43: ("Blanc2014_T12_intestine_primers", "intestine"),
        44: ("Blanc2014_T13_liver_primers", "liver"),
        45: ("Blanc2014_T13_liver_primers", "liver"),
    }

    for i in range(len(pdf)):
        text = pdf[i].get_text()
        if i not in page_to_src:
            # still try standalone pattern on any page
            src, tissue = ("Blanc2014_other", "unknown")
        else:
            src, tissue = page_to_src[i]

        # Standalone
        for m in pat_standalone.finditer(text):
            raw_chr = m.group(1)
            pos = int(m.group(2))
            strand = m.group(3)
            key = (raw_chr, pos, strand)
            strand_known.add(key)
            sid = f"chr{raw_chr}:{pos}:{strand}"
            page_sources.setdefault(sid, (src, tissue))

        # Inline (add only if a stranded entry does not already exist)
        for m in pat_inline.finditer(text):
            raw_chr = m.group(1)
            pos = int(m.group(2))
            strand_unknown.append((raw_chr, pos, src, tissue))

    pdf.close()

    rows = []
    for raw_chr, pos, strand in strand_known:
        sid = f"chr{raw_chr}:{pos}:{strand}"
        src, tissue = page_sources[sid]
        rows.append(
            {
                "site_id": sid,
                "chr": f"chr{raw_chr}",
                "start": pos,
                "strand": strand,
                "enzyme": "APOBEC1",
                "dataset_source": src,
                "tissue": tissue,
                "species": "mouse",
                "build": "mm9",
                "is_edited": 1,
                "ko_validated": 1,
                "editing_rate": np.nan,
            }
        )
    # strand-unknown entries — keep only those without a stranded counterpart
    already = {(c, p) for (c, p, _s) in strand_known}
    for raw_chr, pos, src, tissue in strand_unknown:
        if (raw_chr, pos) in already:
            continue
        # we cannot resolve strand without the genome; queue as + and mark unclear
        sid = f"chr{raw_chr}:{pos}:?"
        rows.append(
            {
                "site_id": sid,
                "chr": f"chr{raw_chr}",
                "start": pos,
                "strand": "?",
                "enzyme": "APOBEC1",
                "dataset_source": src,
                "tissue": tissue,
                "species": "mouse",
                "build": "mm9",
                "is_edited": 1,
                "ko_validated": 1,
                "editing_rate": np.nan,
            }
        )

    df = pd.DataFrame(rows)
    logger.info(
        "Blanc 2014: %d unique coords (%d stranded + %d strand-ambiguous)",
        len(df),
        sum(df["strand"].isin(["+", "-"])),
        sum(df["strand"] == "?"),
    )
    return df


def parse_rosenberg_2011_table3_pdf() -> pd.DataFrame:
    """Extract the 33 validated mouse APOBEC1 sites from Rosenberg 2011 Supp Table S3.

    The PDF page 8 has lines like:
        chr12:8014860(+) Apob CDS C T 255 ...
        chr8:46391931(-) Cyp4v3 3'UTR G R 228 ...
    """
    import pymupdf

    pdf_path = RAW_DIR / "rosenberg_2011_supp.pdf"
    if not pdf_path.exists():
        logger.warning("Rosenberg 2011 PDF not found at %s", pdf_path)
        return pd.DataFrame()

    pdf = pymupdf.open(str(pdf_path))
    text = pdf[8].get_text()
    pdf.close()

    # chrX:pos(+|-) followed by Gene name and Type
    # Allow both - and – (figure dash)
    pat = re.compile(r"chr(\d{1,2}|X|Y):(\d+)\s*\((\+|-|–|−)\)\s+(\S+)\s+(\S+)")

    rows = []
    for m in pat.finditer(text):
        raw_chr = m.group(1)
        pos = int(m.group(2))
        strand_raw = m.group(3)
        strand = "+" if strand_raw == "+" else "-"
        gene = m.group(4)
        region_type = m.group(5)
        sid = f"chr{raw_chr}:{pos}:{strand}"
        # Region type normalization: "3'UTR" / "CDS" / "5'UTR"
        reg = region_type.upper().replace("′", "'").replace("’", "'")
        rows.append(
            {
                "site_id": sid,
                "chr": f"chr{raw_chr}",
                "start": pos,
                "strand": strand,
                "enzyme": "APOBEC1",
                "dataset_source": f"Rosenberg2011_T3_{reg}",
                "tissue": "intestine",  # Rosenberg 2011 is mouse small intestine
                "species": "mouse",
                "build": "mm9",
                "is_edited": 1,
                "ko_validated": 1,
                "editing_rate": np.nan,
                "gene": gene,
            }
        )

    df = pd.DataFrame(rows)
    logger.info("Rosenberg 2011 Table S3: %d sites", len(df))
    return df


def parse_rosenberg_2011_table4_xls() -> pd.DataFrame:
    """Rosenberg 2011 Table S4: ~376 predicted APOBEC1 pattern-match sites.

    Columns in the XLS: Chr, Start, End, Strand, Transcript Accession,
    Gene, Sequence Type, Pattern Match.

    These are predicted (motif-matched) not KO-validated, so we mark
    ko_validated=0. The paper validated a subset of these as Table S3.
    """
    xls_path = RAW_DIR / "rosenberg_2011_table4.xls"
    if not xls_path.exists():
        logger.warning("Rosenberg 2011 T4 xls not found")
        return pd.DataFrame()

    t4 = pd.read_excel(xls_path)
    t4.columns = [str(c).strip() for c in t4.columns]

    # The "Start" column is the leftmost genomic coord of the motif pattern,
    # which starts ~10 nt upstream of the edited C depending on pattern.
    # Paper defines edited C as the "C" in "WC" part of the pattern, but
    # without ground truth we use Start+offset heuristic: treat 'Start' as
    # the edited C (conservative; will mostly hit an A/U in practice).
    # Since these are *predicted* sites we will filter later by checking
    # that the genome base at the position is actually "C" on the correct
    # strand; if not we skip.
    rows = []
    for _, r in t4.iterrows():
        chrom = str(r.get("Chr", "")).strip()
        if not chrom:
            continue
        start = int(r.get("Start", 0))
        strand_val = str(r.get("Strand", "+")).strip()
        strand = "+" if strand_val in ("+", "1", "sense") else "-"
        # The edited C is typically at position where the pattern "WC" appears.
        # Pattern Match column e.g. "UCAAACAAAAUUAUUAU" — first C in W_C context.
        # Without strand-aware parsing it's safer to search for C within ±20 of Start.
        gene = str(r.get("Gene", "")).strip()
        region = str(r.get("Sequence Type", "")).strip()
        pattern = str(r.get("Pattern Match", "")).strip()

        rows.append(
            {
                "chr": chrom,
                "start": start,  # refined below
                "strand": strand,
                "enzyme": "APOBEC1",
                "dataset_source": f"Rosenberg2011_T4_{region}",
                "tissue": "intestine",
                "species": "mouse",
                "build": "mm9",
                "is_edited": 1,
                "ko_validated": 0,  # predicted only, not functionally validated
                "editing_rate": np.nan,
                "gene": gene,
                "pattern": pattern,
            }
        )
    df = pd.DataFrame(rows)
    logger.info("Rosenberg 2011 Table S4: %d candidate entries", len(df))
    return df


# ---------------------------------------------------------------------------
# Coordinate resolution helpers
# ---------------------------------------------------------------------------

def resolve_c_position(genome, chrom: str, approx_pos: int, strand: str,
                       search_radius: int = 25) -> int | None:
    """Given an approximate position from a supp table, snap to the nearest C on
    the given strand. Returns 1-based C position, or None if no C within radius.

    Blanc 2014 positions are typically the edited C directly. Rosenberg T4 lists
    the leftmost position of a 16-nt pattern, so snapping to the nearest C is
    needed to actually find the edited base.
    """
    if chrom not in genome:
        return None
    chrom_len = len(genome[chrom])
    if approx_pos < 1 or approx_pos > chrom_len:
        return None

    pos_0 = approx_pos - 1
    start = max(0, pos_0 - search_radius)
    end = min(chrom_len, pos_0 + search_radius + 1)
    region = str(genome[chrom][start:end]).upper()

    # Target base on reference is C (+ strand) or G (- strand)
    target = "C" if strand == "+" else "G"

    # Prefer the position at or closest to approx_pos
    best = None
    best_dist = None
    for i, base in enumerate(region):
        if base != target:
            continue
        genome_pos = start + i + 1  # 1-based
        dist = abs(genome_pos - approx_pos)
        if best_dist is None or dist < best_dist:
            best = genome_pos
            best_dist = dist
    return best


def build_sequences(positives: pd.DataFrame, genome) -> tuple[pd.DataFrame, dict]:
    """Extract 201-nt ±100 sequences for each positive coordinate. Drop rows
    where the center base after strand-normalization is not C.

    Unknown-strand rows: try both strands and take whichever yields a C center.
    """
    kept = []
    dropped = 0
    seqs: dict[str, str] = {}

    for _, row in positives.iterrows():
        chrom = row["chr"]
        pos = int(row["start"])
        strand = row["strand"]
        ko_flag = int(row.get("ko_validated", 1))

        if strand == "?":
            # try both strands
            candidates = ["+", "-"]
        else:
            candidates = [strand]

        for s in candidates:
            # For predicted (T4), snap to nearest C; for KO-validated, accept only exact C
            radius = 25 if ko_flag == 0 else 3
            snapped = resolve_c_position(genome, chrom, pos, s, search_radius=radius)
            if snapped is None:
                continue
            seq = extract_from_genome(genome, chrom, snapped, s)
            if seq is None:
                continue
            sid = f"{chrom}:{snapped}:{s}"
            new_row = row.copy()
            new_row["site_id"] = sid
            new_row["start"] = snapped
            new_row["end"] = snapped
            new_row["strand"] = s
            new_row["coordinate_system"] = "mm9"
            new_row["source_type"] = "positive"
            new_row["flanking_seq"] = seq
            new_row["seq_center"] = CENTER
            kept.append(new_row)
            seqs[sid] = seq
            break
        else:
            dropped += 1

    out = pd.DataFrame(kept)
    # drop duplicates keeping KO-validated first
    out = out.sort_values("ko_validated", ascending=False).drop_duplicates(
        subset=["site_id"], keep="first"
    )
    logger.info("Extracted %d sequences (dropped %d unresolvable)", len(out), dropped)
    return out.reset_index(drop=True), seqs


def generate_motif_matched_negatives(positives: pd.DataFrame, genome,
                                     sequences: dict,
                                     tc_fraction: float, cc_fraction: float,
                                     n_negatives: int) -> tuple[pd.DataFrame, dict]:
    """Generate motif-matched negatives from mm9 near positive sites.

    Similar logic to src/data/apobec_negatives.generate_negatives_from_genome
    but adapted for (chr, start, strand) df from mouse positives (there is no
    enzyme column variety here — all are APOBEC1).
    """
    rng = random.Random(SEED)
    known = {(r["chr"], int(r["start"])) for _, r in positives.iterrows()}

    all_cands = []
    for _, row in positives.iterrows():
        chrom = row["chr"]
        pos = int(row["start"])
        strand = row["strand"]
        tissue = row.get("tissue", "unknown")
        if chrom not in genome:
            continue
        chrom_len = len(genome[chrom])
        s_start = max(0, pos - NEG_SEARCH_WINDOW)
        s_end = min(chrom_len, pos + NEG_SEARCH_WINDOW)
        region = str(genome[chrom][s_start:s_end]).upper()

        target = "C" if strand == "+" else "G"
        for i, base in enumerate(region):
            if base != target:
                continue
            gp = s_start + i + 1  # 1-based
            if (chrom, gp) in known or gp == pos:
                continue
            pos_0 = gp - 1
            if pos_0 < FLANK or pos_0 + FLANK >= chrom_len:
                continue
            # upstream-base for oriented TC/CC
            if strand == "+":
                up = region[i - 1] if i > 0 else "N"
                dinuc = up + "C"
            else:
                down = region[i + 1] if i + 1 < len(region) else "N"
                comp = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
                dinuc = comp.get(down, "N") + "C"
            is_tc = dinuc == "TC"
            is_cc = dinuc == "CC"
            all_cands.append({
                "chrom": chrom, "pos": gp, "strand": strand,
                "tissue": tissue, "is_tc": is_tc, "is_cc": is_cc,
            })

    logger.info("Found %d candidate negatives", len(all_cands))
    if not all_cands:
        return pd.DataFrame(), sequences

    tc = [c for c in all_cands if c["is_tc"]]
    cc = [c for c in all_cands if c["is_cc"]]
    other = [c for c in all_cands if not c["is_tc"] and not c["is_cc"]]

    n_tc = min(int(round(n_negatives * tc_fraction)), len(tc))
    n_cc = min(int(round(n_negatives * cc_fraction)), len(cc))
    n_other = min(n_negatives - n_tc - n_cc, len(other))
    # Top up if under-filled
    remaining = n_negatives - n_tc - n_cc - n_other
    for pool_name, pool_list, current in [("tc", tc, n_tc), ("cc", cc, n_cc), ("other", other, n_other)]:
        if remaining <= 0:
            break
        avail = len(pool_list) - current
        if avail > 0:
            add = min(remaining, avail)
            if pool_name == "tc":
                n_tc += add
            elif pool_name == "cc":
                n_cc += add
            else:
                n_other += add
            remaining -= add

    rng.shuffle(tc); rng.shuffle(cc); rng.shuffle(other)
    picked = tc[:n_tc] + cc[:n_cc] + other[:n_other]
    rng.shuffle(picked)
    logger.info("Selected negatives: TC=%d  CC=%d  other=%d  total=%d",
                n_tc, n_cc, n_other, len(picked))

    rows = []
    used = set(known)
    for c in picked:
        chrom, gp, strand = c["chrom"], c["pos"], c["strand"]
        if (chrom, gp) in used:
            continue
        seq = extract_from_genome(genome, chrom, gp, strand)
        if seq is None:
            continue
        sid = f"{chrom}:{gp}:{strand}"
        used.add((chrom, gp))
        sequences[sid] = seq
        rows.append({
            "site_id": sid,
            "chr": chrom, "start": gp, "end": gp, "strand": strand,
            "enzyme": "APOBEC1",
            "dataset_source": f"apobec1_neg_{c['tissue']}",
            "tissue": c["tissue"],
            "species": "mouse",
            "build": "mm9",
            "coordinate_system": "mm9",
            "is_edited": 0,
            "ko_validated": 0,
            "editing_rate": 0.0,
            "source_type": "negative_control",
            "flanking_seq": seq,
            "seq_center": CENTER,
        })
    df = pd.DataFrame(rows)
    logger.info("Generated %d negative sequences", len(df))
    return df, sequences


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if not GENOME_MM9.exists():
        logger.error("mm9 genome not found at %s", GENOME_MM9)
        sys.exit(1)

    from pyfaidx import Fasta
    genome = Fasta(str(GENOME_MM9))

    logger.info("Parsing Blanc 2014 Additional File 1 PDF ...")
    b2014 = parse_blanc_2014_pdf()
    logger.info("Parsing Rosenberg 2011 Supp Table S3 PDF ...")
    r2011_t3 = parse_rosenberg_2011_table3_pdf()
    logger.info("Parsing Rosenberg 2011 Supp Table S4 XLS ...")
    r2011_t4 = parse_rosenberg_2011_table4_xls()

    all_pos = pd.concat([b2014, r2011_t3, r2011_t4], ignore_index=True)
    logger.info("Combined raw positives: %d rows", len(all_pos))

    # Deduplicate on (chr, start, strand) keeping KO-validated first
    all_pos["_dedup_key"] = all_pos["chr"] + ":" + all_pos["start"].astype(str) + ":" + all_pos["strand"].astype(str)
    all_pos = all_pos.sort_values("ko_validated", ascending=False)
    all_pos = all_pos.drop_duplicates(subset=["_dedup_key"], keep="first").drop(columns=["_dedup_key"])
    logger.info("After dedup: %d positives", len(all_pos))

    # Save raw positives table (pre-sequence extraction)
    all_pos.to_csv(PROC_DIR / "apobec1_positives_raw.csv", index=False)

    logger.info("Extracting 201-nt sequences from mm9 ...")
    positives_with_seq, sequences = build_sequences(all_pos, genome)
    logger.info("Positives with sequences: %d", len(positives_with_seq))

    # Motif fractions of positives -> targets for neg-matching
    def _dinuc(seq: str) -> str:
        return seq[CENTER - 1] + seq[CENTER]
    dinucs = [_dinuc(sequences[sid]) for sid in positives_with_seq["site_id"]]
    tc_frac = sum(1 for d in dinucs if d == "UC") / max(1, len(dinucs))
    cc_frac = sum(1 for d in dinucs if d == "CC") / max(1, len(dinucs))
    logger.info("Positive motif fractions: TC=%.2f CC=%.2f", tc_frac, cc_frac)

    n_pos = len(positives_with_seq)
    n_neg = n_pos * NEG_RATIO
    logger.info("Generating %d motif-matched negatives ...", n_neg)
    negatives, sequences = generate_motif_matched_negatives(
        positives_with_seq, genome, sequences,
        tc_fraction=tc_frac, cc_fraction=cc_frac, n_negatives=n_neg,
    )

    combined = pd.concat([positives_with_seq, negatives], ignore_index=True)
    combined.to_csv(PROC_DIR / "apobec1_v1_with_negatives.csv", index=False)
    logger.info("Wrote %s (%d rows)", PROC_DIR / "apobec1_v1_with_negatives.csv", len(combined))

    # Sequences JSON
    with open(PROC_DIR / "apobec1_v1_sequences.json", "w") as f:
        json.dump(sequences, f)
    logger.info("Wrote %s (%d sequences)", PROC_DIR / "apobec1_v1_sequences.json", len(sequences))

    # Summary
    summary = {
        "n_positives_total": int(len(positives_with_seq)),
        "n_negatives_total": int(len(negatives)),
        "positives_by_source": positives_with_seq["dataset_source"].value_counts().to_dict(),
        "positives_by_tissue": positives_with_seq["tissue"].value_counts().to_dict(),
        "positives_by_ko_validated": positives_with_seq["ko_validated"].value_counts().to_dict(),
        "positives_motif_fractions": {
            "TC": float(tc_frac), "CC": float(cc_frac),
            "n_TC": int(sum(1 for d in dinucs if d == "UC")),
            "n_CC": int(sum(1 for d in dinucs if d == "CC")),
        },
    }
    with open(PROC_DIR / "dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("\nSummary:\n%s", json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
