"""Build clean APOBEC1 training set v2.

Aggregates published A1 RNA editing sites into a unified CSV with provenance.

Sources:
  - Davidson 2014 (GSE57910 BEDs, mm9): 78 sites total (intestine 56, liver 22)
  - Rosenberg 2011 (Suppl Table 3, mm9): 33 Sanger-validated sites
  - Blanc 2014 (additional file 1 PDF, mm9): exonic + 3'UTR validated edit sites
  - Rayon-Estrada 2017 (PNAS, mm9/mm10): TODO - SI PDF blocked behind PNAS/PMC anti-scraping
  - Cole 2017 (PNAS, mm9/mm10): TODO - SI PDF blocked behind PNAS/PMC anti-scraping

Output: data/processed/apobec1_clean/apobec1_training_v2.csv
"""
from __future__ import annotations

import gzip
import re
from pathlib import Path

import pandas as pd
import pdfplumber

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "apobec1"
OUT_DIR = ROOT / "data" / "processed" / "apobec1_clean"
OUT_CSV = OUT_DIR / "apobec1_training_v2.csv"

# ----------------------------------------------------------------------------
# Source 1: Davidson 2014 BED files (mm9)
# ----------------------------------------------------------------------------

def parse_davidson(tissue: str, path: Path) -> list[dict]:
    """Parse a Davidson 2014 BED file (chr, start, end - 0-based half-open)."""
    rows: list[dict] = []
    with gzip.open(path, "rt") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("track"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            chrom = parts[0]
            start = int(parts[1])
            end = int(parts[2])
            # BED: 0-based start, end-exclusive. Convert to 1-based pos = end (start+1).
            pos = start + 1
            strand = parts[5] if len(parts) >= 6 else "."
            rows.append(
                {
                    "site_id": f"davidson_{tissue}_{chrom}_{pos}_{strand}",
                    "source_study": "davidson_2014",
                    "species": "mouse",
                    "tissue": tissue,
                    "confidence": "RNA_seq_only",
                    "genome_build": "mm9",
                    "chrom": chrom,
                    "pos": pos,
                    "strand": strand,
                    "gene": None,
                    "seq_context": None,
                }
            )
    return rows


# ----------------------------------------------------------------------------
# Source 2: Rosenberg 2011 (Suppl Table 3 in PDF, page 9)
# ----------------------------------------------------------------------------

# Hard-coded from PDF text extraction (page 9). 33 Sanger-validated sites.
# Format from PDF: "chr12:8014860(+) Apob CDS C T 255 255 204 0.93 ..."
ROSENBERG_VALIDATED_TEXT = """\
chr12:8014860(+) Apob CDS C T 255 255 204 0.93
chr2:121978638(+) B2m 3UTR C Y 228 228 2860 0.18
chrX:109671648(+) 2010106E10Rik 3UTR C Y 228 228 688 0.46
chr8:46391931(-) Cyp4v3 3UTR G R 228 228 112 0.38
chr3:129616676(+) Casp6 3UTR C Y 228 228 107 0.50
chr17:44416335(+) Clic5 3UTR C Y 175 175 186 0.31
chr10:57235791(-) Serinc1 3UTR G R 77 170 29 0.75
chr5:87984364(-) Sult1d1 3UTR G R 60 154 28 0.79
chr2:143811725(-) Rrbp1 3UTR G R 149 149 40 0.38
chr10:7487994(-) BC013529 3UTR G R 141 141 20 0.45
chr9:79617629(-) Tmem30a 3UTR G R 129 135 22 0.55
chr1:152208563(-) BC003331 3UTR G R 54 132 23 0.74
chr4:57203753(-) Ptpn3 3UTR G R 67 124 15 0.67
chr16:77116537(+) Usp25 3UTR C Y 116 116 16 0.50
chr3:119135667(+) Dpyd 3UTR C Y 115 115 26 0.32
chr16:84955113(-) App 3UTR G R 108 108 563 0.21
chr13:96397289(-) Iqgap2 3UTR G R 103 103 514 0.23
chr3:144259976(+) Sep15 3UTR C Y 93 103 13 0.54
chrX:136207009(+) Rnf128 3UTR C Y 91 91 669 0.20
chrX:106355759(+) Sh3bgrl 3UTR C Y 89 89 23 0.30
chrX:50374459(+) Hprt1 3UTR C Y 85 85 55 0.22
chr4:94304303(-) Lrrc19 3UTR G R 85 85 38 0.26
chr3:119135669(+) Dpyd 3UTR C Y 84 84 25 0.28
chr14:73595382(-) Rb1 3UTR G R 83 83 21 0.33
chr12:85772761(-) Aldh6a1 3UTR G R 64 80 9 0.56
chr2:73654730(-) Atf2 3UTR G R 73 73 21 0.29
chr16:43981376(-) Gramd1c 3UTR G R 64 64 17 0.29
chr16:84954758(-) App 3UTR G R 60 60 293 0.21
chr10:69486962(+) Ank3 3UTR C Y 56 56 11 0.36
chr13:96397211(-) Iqgap2 3UTR G R 55 55 124 0.38
chr3:73442586(-) Bche 3UTR G R 54 54 14 0.36
chr1:192830761(-) Mfsd7b 3UTR G A 2 48 9 0.78
chr15:99239051(+) Tmbim6 3UTR C Y 45 45 389 0.20
"""


def parse_rosenberg() -> list[dict]:
    """Parse 33 validated A1 editing sites from Rosenberg 2011 Suppl Table 3."""
    rows: list[dict] = []
    pattern = re.compile(r"^(chr[\w]+):(\d+)\(([+-])\)\s+(\S+)\s+(\S+)")
    for line in ROSENBERG_VALIDATED_TEXT.strip().split("\n"):
        m = pattern.match(line.strip())
        if not m:
            continue
        chrom, pos, strand, gene, site_type = m.group(1), int(m.group(2)), m.group(3), m.group(4), m.group(5)
        # Tissue inference: Rosenberg studied small intestine enterocytes (mouse)
        rows.append(
            {
                "site_id": f"rosenberg_{chrom}_{pos}_{strand}",
                "source_study": "rosenberg_2011",
                "species": "mouse",
                "tissue": "intestine",
                "confidence": "Sanger_confirmed",
                "genome_build": "mm9",
                "chrom": chrom,
                "pos": pos,
                "strand": strand,
                "gene": gene,
                "seq_context": None,
            }
        )
    return rows


# ----------------------------------------------------------------------------
# Source 3: Blanc 2014 additional file 1 PDF
# ----------------------------------------------------------------------------

# Pages 1-5 = Tables 1A-1E (exonic targets, KO-validated by RNA-seq + Sanger)
# Pages 8-10 = Tables 3A-3C (3'UTR targets, validated)
# We extract from text. Each row: "1. ApoB 12 8014860 (+) C 98% ..." -- gene + chr + pos + (strand).

BLANC_PDF = RAW / "blanc_2014_additional1.pdf"

# Match a row like:
#   "1. ApoB 12 8014860 (+) C 98% 100% (20/20) Glu-Stop"
#   "ApoB 12 8015181 (+) C 14% 30% (6/20) His-Tyr"     (continued)
# or for table 3A:
#   "9 114658301(+) 12% 194 ..."  -- continuation lines have only chr+pos+strand
# We need flexible regex.

_BLANC_RX = re.compile(
    r"""
    ^\s*
    (?:\d+\s*\.\s*)?         # optional row number "1. " or "4." (no space)
    ([A-Za-z][A-Za-z0-9._;:-]*?)   # gene
    \s+
    ([0-9XY]{1,2}|MT)        # chromosome (numeric/X/Y/MT, no "chr" prefix)
    \s+
    (\d{3,})                 # position (>=3 digits)
    \s*
    \(([+-])\)               # strand
    """,
    re.VERBOSE,
)
# Fallback for continuation rows: chr + pos + strand (no gene)
_BLANC_RX_NOGENE = re.compile(
    r"^\s*([0-9XY]{1,2}|MT)\s+(\d{3,})\s*\(([+-])\)"
)
# Fallback for continuation rows: pos + strand only (reuses last gene+chrom)
_BLANC_RX_POSONLY = re.compile(
    r"^\s*(\d{6,})\s*\(([+-])\)"
)


def parse_blanc_table_page(text: str, table_id: str, tissue: str, confidence: str,
                           default_gene: str | None = None) -> list[dict]:
    """Parse one Blanc 2014 table page text into site rows."""
    rows: list[dict] = []
    last_gene: str | None = default_gene
    last_chrom_raw: str | None = None
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        # skip header / footer lines
        if any(
            tag in line
            for tag in (
                "Supplemental Table",
                "RNA Chr Position",
                "Genotype:",
                "BT Below Threshold",
                "Bold numbers",
                "(+) Sense strand",
                "(-) Antisense strand",
                "WRAUYANUAU",
                "ApoB RNA sequences",
                "Consensus mooring",
                "Edited cytidines",
                "Nucleotides matching",
                "Light orange",
                "Matched nucleotides",
            )
        ):
            continue
        # also skip header continuation rows
        if line.startswith("efficiency") or line.startswith("WT Apobec") or line.startswith("-seq"):
            continue
        if line.startswith("rescue") or line.startswith("Apobec-1"):
            continue
        m = _BLANC_RX.match(line)
        gene_used = None
        chrom_raw = pos = strand = None
        if m:
            gene_used = m.group(1)
            chrom_raw = m.group(2)
            pos = int(m.group(3))
            strand = m.group(4)
            last_gene = gene_used
            last_chrom_raw = chrom_raw
        else:
            m2 = _BLANC_RX_NOGENE.match(line)
            if m2 and last_gene is not None:
                gene_used = last_gene
                chrom_raw = m2.group(1)
                pos = int(m2.group(2))
                strand = m2.group(3)
                last_chrom_raw = chrom_raw
            else:
                m3 = _BLANC_RX_POSONLY.match(line)
                if m3 and last_gene is not None and last_chrom_raw is not None:
                    gene_used = last_gene
                    chrom_raw = last_chrom_raw
                    pos = int(m3.group(1))
                    strand = m3.group(2)
        if gene_used is None or chrom_raw is None or pos is None:
            continue
        # Sanity: skip absurdly short positions (likely mis-parsed)
        if pos < 1000:
            continue
        chrom = f"chr{chrom_raw}"
        rows.append(
            {
                "site_id": f"blanc_{table_id}_{chrom}_{pos}_{strand}",
                "source_study": "blanc_2014",
                "species": "mouse",
                "tissue": tissue,
                "confidence": confidence,
                "genome_build": "mm9",
                "chrom": chrom,
                "pos": pos,
                "strand": strand,
                "gene": gene_used,
                "seq_context": None,
            }
        )
    return rows


def parse_blanc() -> list[dict]:
    """Parse Blanc 2014 PDF into site rows.

    Tables we keep:
      1A intestine exonic (KO_validated, Sanger_confirmed where Sanger>0)
      1E liver exonic (KO_validated)
      3A intestine 3'UTR multi-genotype (KO_validated)
      3B hepatic 3'UTR ad-A1 rescue (KO_validated)
      3C hepatic 3'UTR additional Sanger sites (Sanger_confirmed; uses Aldh6a1, Tmem30a, etc.)

    Tables we skip:
      1B/1C/1D — Sanger-discordant / cohort (negative or low confidence)
      Table 2/4 — same sites as 1A/1E with mooring annotation (redundant)
      Tables 5-13 — expression / miRNA / primers (no new sites)
    """
    rows: list[dict] = []
    with pdfplumber.open(str(BLANC_PDF)) as pdf:
        # Page indices (0-based) per table from earlier scan
        page_specs = [
            (0, "1A_intestine_exonic", "intestine", "KO_validated", None),
            (4, "1E_liver_exonic", "liver", "KO_validated", None),
            (7, "3A_intestine_3UTR", "intestine", "KO_validated", None),
            (8, "3B_hepatic_3UTR", "liver", "KO_validated", None),
            (9, "3C_hepatic_3UTR_extra", "liver", "Sanger_confirmed", None),
        ]
        for page_idx, table_id, tissue, conf, default_gene in page_specs:
            text = pdf.pages[page_idx].extract_text() or ""
            rows.extend(parse_blanc_table_page(text, table_id, tissue, conf, default_gene))
    return rows


# ----------------------------------------------------------------------------
# Aggregate, dedupe, write
# ----------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sources: list[tuple[str, list[dict]]] = []

    # Davidson
    dav_int = parse_davidson("intestine", RAW / "davidson_2014_intestine.bed.gz")
    dav_liv = parse_davidson("liver", RAW / "davidson_2014_liver.bed.gz")
    sources.append(("davidson_2014", dav_int + dav_liv))

    # Rosenberg
    sources.append(("rosenberg_2011", parse_rosenberg()))

    # Blanc
    try:
        sources.append(("blanc_2014", parse_blanc()))
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] Blanc 2014 parsing failed: {e}")
        sources.append(("blanc_2014", []))

    # Rayon-Estrada / Cole — TODO (SI PDFs blocked)
    sources.append(("rayon_estrada_2017", []))
    sources.append(("cole_2017", []))

    # Per-source counts
    print("\n=== Sites per source (pre-dedup) ===")
    all_rows: list[dict] = []
    for name, rows in sources:
        print(f"  {name:25s} {len(rows):5d}")
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    if df.empty:
        print("[ERROR] No rows extracted from any source.")
        return

    # Normalise strand
    df["strand"] = df["strand"].fillna(".").replace({"": "."})

    # Dedupe within same genome build on (chrom, pos, strand)
    # Confidence priority for "keep first": KO_validated > Sanger_confirmed > RNA_seq_only
    conf_rank = {"KO_validated": 0, "Sanger_confirmed": 1, "RNA_seq_only": 2}
    df["_conf_rank"] = df["confidence"].map(conf_rank).fillna(9).astype(int)

    # When tissues overlap for the same site, concatenate them
    before = len(df)
    agg = (
        df.groupby(["genome_build", "chrom", "pos", "strand"], dropna=False)
        .agg(
            tissue=("tissue", lambda s: ";".join(sorted(set(s.dropna())))),
            source_study=("source_study", lambda s: ";".join(sorted(set(s.dropna())))),
            species=("species", "first"),
            confidence=("confidence", lambda s: sorted(s, key=lambda x: conf_rank.get(x, 9))[0]),
            gene=("gene", lambda s: next((x for x in s if pd.notna(x)), None)),
            seq_context=("seq_context", lambda s: next((x for x in s if pd.notna(x)), None)),
            site_id=("site_id", "first"),
        )
        .reset_index()
    )
    df = agg
    after = len(df)
    print(f"\nDedup: {before} -> {after} unique (chrom, pos, strand) within genome_build")
    print("  (tissue and source_study fields concatenated with ';' on overlap)")

    # Write CSV
    cols = [
        "site_id",
        "source_study",
        "species",
        "tissue",
        "confidence",
        "genome_build",
        "chrom",
        "pos",
        "strand",
        "gene",
        "seq_context",
    ]
    df = df[cols]
    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}  ({len(df)} rows)")

    # Summary breakdowns
    print("\n=== Source x tissue x species breakdown ===")
    print(df.groupby(["source_study", "species", "tissue"]).size().to_string())
    print("\n=== Confidence distribution ===")
    print(df["confidence"].value_counts().to_string())
    print("\n=== Genome build distribution ===")
    print(df["genome_build"].value_counts().to_string())


if __name__ == "__main__":
    main()
