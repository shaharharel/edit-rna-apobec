"""
Build unified multi-enzyme dataset for APOBEC motif/structure analysis.

Inputs:
  data/processed/multi_enzyme/kockler_2026_sites.csv   (A3A, A3B, hg38 transcriptomic coords)
  data/processed/multi_enzyme/dang_2019_sites.csv      (A3G, hg19)
  data/processed/multi_enzyme/zhang_2024_sites.csv     (A3B, hg38 genomic coords)
  data/raw/genomes/hg38.fa                             (for Zhang)
  data/raw/genomes/hg19.fa                             (for Dang fallback)

Sequence strategy:
  Kockler: uses CONTEXT(+/-20) column (41 nt, center=20)
    - Kockler uses transcriptomic/non-standard coords — genome extraction fails
    - +strand: context already in RNA-sense direction (uppercase = C, edit target)
    - -strand: reverse complement to get RNA-sense direction, then uppercase is C
    - Padded with Ns to 201 nt (center → 100) so downstream tools work uniformly
  Dang:    uses up/down flanking seqs (31 nt total, center=15)
    - Same padding strategy
  Zhang:   uses hg38.fa extraction (coordinates are standard genomic 1-based)
    - 201-nt window, center=100, all plus-strand sites

Outputs:
  data/processed/multi_enzyme/splits_multi_enzyme.csv
  data/processed/multi_enzyme/multi_enzyme_sequences.json
"""
import json
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
KOCKLER_CSV = PROJECT_ROOT / "data/processed/multi_enzyme/kockler_2026_sites.csv"
DANG_CSV    = PROJECT_ROOT / "data/processed/multi_enzyme/dang_2019_sites.csv"
ZHANG_CSV   = PROJECT_ROOT / "data/processed/multi_enzyme/zhang_2024_sites.csv"

HG38_FA = PROJECT_ROOT / "data/raw/genomes/hg38.fa"
HG19_FA = PROJECT_ROOT / "data/raw/genomes/hg19.fa"

OUT_DIR  = PROJECT_ROOT / "data/processed/multi_enzyme"
OUT_CSV  = OUT_DIR / "splits_multi_enzyme_v2.csv"
OUT_JSON = OUT_DIR / "multi_enzyme_sequences_v2.json"

TARGET_LEN = 201
CENTER = 100

RC_MAP = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")


def rev_comp(seq: str) -> str:
    return seq[::-1].translate(RC_MAP)


def pad_to_length(seq: str, orig_center: int, target_len: int = TARGET_LEN, target_center: int = CENTER) -> str:
    """Pad a short sequence with Ns so edit site moves from orig_center to target_center."""
    left_pad  = target_center - orig_center
    right_pad = target_len - len(seq) - left_pad
    if left_pad < 0 or right_pad < 0:
        raise ValueError(f"Cannot pad: seq len={len(seq)}, orig_center={orig_center}, "
                         f"target_len={target_len}, target_center={target_center}")
    return "N" * left_pad + seq + "N" * right_pad


def context_to_rna(context: str, strand: str, orig_center: int = 20) -> str | None:
    """Convert a MAF-style DNA context to RNA sequence with edit site at position CENTER.

    MAF context convention: sequence in 5'->3' direction on + strand, edit base uppercase.
    For + strand: context is in RNA-sense direction; uppercase = C (edit target in DNA).
    For - strand: context shows + strand reference (uppercase = G); rev_comp to get RNA view.
    Returns 201-nt RNA sequence (T→U) with edit site at index CENTER, or None on failure.
    """
    if not context or not isinstance(context, str):
        return None

    ctx = context.strip()
    if len(ctx) < orig_center + 1:
        return None

    if strand == "-":
        ctx = rev_comp(ctx)
        # After rev_comp, edit position moves to len-1-orig_center
        orig_center = len(ctx) - 1 - orig_center

    ctx_rna = ctx.upper().replace("T", "U")

    try:
        padded = pad_to_length(ctx_rna, orig_center, TARGET_LEN, CENTER)
    except ValueError:
        return None

    return padded


def load_genome(fa_path: Path):
    try:
        from pyfaidx import Fasta
        logger.info("Loading genome %s ...", fa_path.name)
        return Fasta(str(fa_path))
    except ImportError:
        logger.error("pyfaidx not installed; run: pip install pyfaidx")
        sys.exit(1)
    except Exception as exc:
        logger.error("Failed to load genome %s: %s", fa_path, exc)
        return None


def extract_from_genome(genome, chrom: str, pos: int, strand: str) -> str | None:
    """Extract 201-nt RNA sequence centered on pos (1-based, + strand coords).

    All Zhang sites are + strand; after extraction the edit site at pos is C.
    Returns None if region is out of bounds or chrom not found.
    """
    window = 100
    start = pos - window - 1   # 0-based
    end   = pos + window        # 0-based exclusive

    if chrom not in genome.keys():
        alt = chrom.replace("chr", "") if chrom.startswith("chr") else f"chr{chrom}"
        if alt in genome.keys():
            chrom = alt
        else:
            return None

    chrom_len = len(genome[chrom])
    if start < 0 or end > chrom_len:
        return None

    seq = str(genome[chrom][start:end]).upper()

    if strand == "-":
        seq = rev_comp(seq)

    return seq.replace("T", "U")


def build_kockler_sequences(df: pd.DataFrame) -> dict:
    """Extract sequences from the CONTEXT(+/-20) column.

    Kockler uses non-standard/transcriptomic coordinates; genomic extraction fails.
    We use the 41-nt context directly, convert to RNA sense strand, and pad to 201 nt.
    """
    seqs = {}
    bad = 0
    for _, row in df.iterrows():
        sid = row["site_id"]
        strand = row.get("strand", "+")
        context = row.get("flanking_seq", None)

        seq = context_to_rna(context, strand, orig_center=20)
        if seq is None:
            bad += 1
            continue

        # Verify center = C (the edit target)
        if seq[CENTER] != "C":
            bad += 1
            continue

        seqs[sid] = seq

    logger.info("Kockler sequences: %d extracted, %d failed", len(seqs), bad)
    return seqs


def build_dang_sequences(df: pd.DataFrame) -> dict:
    """Extract sequences from up/down flanking columns (31 nt total: 15+C+15).

    Dang uses hg19 coordinates; context sequences are already in RNA-sense direction.
    The flanking_seq in our CSV = up_plusmRNASeq + ref_base + down_plusmRNASeq (31 nt, center=15).
    """
    seqs = {}
    bad = 0
    for _, row in df.iterrows():
        sid = row["site_id"]
        strand = row.get("strand", "+")
        context = row.get("flanking_seq", None)

        # Dang flanking_seq was built as: up(15) + ref_base + down(15) = 31 nt, center=15
        # For + strand sites: context is already in RNA sense, ref_base = C
        # For - strand sites: context is in + strand direction, ref_base = G → rev_comp needed
        seq = context_to_rna(context, strand, orig_center=15)
        if seq is None:
            bad += 1
            continue

        if seq[CENTER] != "C":
            bad += 1
            continue

        seqs[sid] = seq

    logger.info("Dang sequences: %d extracted, %d failed", len(seqs), bad)
    return seqs


def build_zhang_sequences(df: pd.DataFrame, genome) -> dict:
    """Extract sequences from hg38 genome (Zhang uses standard 1-based genomic coords)."""
    seqs = {}
    bad = 0
    for _, row in df.iterrows():
        sid = row["site_id"]
        chrom = row["chr"]
        pos = int(row["start"])
        strand = row.get("strand", "+")

        seq = extract_from_genome(genome, chrom, pos, strand)
        if seq is None or len(seq) != TARGET_LEN:
            bad += 1
            continue

        if seq[CENTER] != "C":
            bad += 1
            continue

        seqs[sid] = seq

    logger.info("Zhang sequences: %d extracted, %d failed", len(seqs), bad)
    return seqs


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    COMMON_COLS = [
        "site_id", "chr", "start", "strand",
        "enzyme", "dataset_source", "coordinate_system",
        "editing_rate", "flanking_seq",
    ]

    # Load datasets
    logger.info("Loading Kockler 2026 (A3A, A3B) ...")
    kockler = pd.read_csv(KOCKLER_CSV)
    for c in COMMON_COLS:
        if c not in kockler.columns:
            kockler[c] = None
    kockler = kockler[COMMON_COLS].copy()

    logger.info("Loading Dang 2019 (A3G NK_Hyp) ...")
    dang_full = pd.read_csv(DANG_CSV)
    if "condition" in dang_full.columns:
        dang = dang_full[dang_full["condition"] == "NK_Hyp"].copy()
    else:
        dang = dang_full.copy()
    for c in COMMON_COLS:
        if c not in dang.columns:
            dang[c] = None
    dang = dang[COMMON_COLS].copy()
    logger.info("  Dang NK_Hyp sites: %d", len(dang))

    logger.info("Loading Zhang 2024 (A3B) ...")
    zhang = pd.read_csv(ZHANG_CSV)
    for c in COMMON_COLS:
        if c not in zhang.columns:
            zhang[c] = None
    zhang = zhang[COMMON_COLS].copy()

    # Load hg38 for Zhang
    hg38 = load_genome(HG38_FA)

    # Build sequences per dataset
    kockler_seqs = build_kockler_sequences(kockler)
    dang_seqs    = build_dang_sequences(dang)
    zhang_seqs   = build_zhang_sequences(zhang, hg38)

    all_seqs = {**kockler_seqs, **dang_seqs, **zhang_seqs}

    # Combine dataframes, keeping only sites with valid sequences
    df = pd.concat([kockler, dang, zhang], ignore_index=True)
    df = df.drop_duplicates(subset=["site_id"])
    df = df[df["site_id"].isin(all_seqs)].copy()
    df["seq_center"] = CENTER

    logger.info("Final unified dataset: %d sites", len(df))
    for enzyme, grp in df.groupby("enzyme"):
        logger.info("  %-4s : %d sites | datasets: %s | with_rate: %d",
                    enzyme, len(grp),
                    sorted(grp.dataset_source.unique()),
                    grp.editing_rate.notna().sum())

    df.to_csv(OUT_CSV, index=False)
    logger.info("Saved splits CSV → %s", OUT_CSV)

    with open(OUT_JSON, "w") as f:
        json.dump(all_seqs, f)
    logger.info("Saved sequences JSON → %s  (%d entries)", OUT_JSON, len(all_seqs))

    print("\n=== Dataset Summary ===")
    for enzyme, grp in df.groupby("enzyme"):
        print(f"  {enzyme}: {len(grp)} sites | {sorted(grp.dataset_source.unique())} | "
              f"rate: {grp.editing_rate.notna().sum()}")
    print(f"\nTotal: {len(df)} sites, {len(all_seqs)} sequences")


if __name__ == "__main__":
    main()
