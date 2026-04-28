#!/usr/bin/env python3
"""Parse PCAWG consensus SNV MAF (cBioPortal pancan_pcawg_2020 distribution)
and split into per-cancer-bucket TCGA-compatible MAF TSVs.

Downloaded from:
  https://media.githubusercontent.com/media/cBioPortal/datahub/master/public/pancan_pcawg_2020/data_mutations.txt

Build: GRCh37 (same as our TCGA pipeline's hg19, NO LIFTOVER NEEDED).

PCAWG HISTOLOGY_ABBREVIATION -> our 10 TCGA-style buckets:
  Liver-HCC          -> lihc    (314 samples,  9,449 C>T)
  ColoRect-AdenoCA   -> coadread (52 samples, 24,725 C>T)
  Stomach-AdenoCA    -> stad     (68 samples,  8,421 C>T)
  Eso-AdenoCA        -> esca     (97 samples,  8,667 C>T)
  Breast-* (AdenoCA, LobularCA, DCIS) -> brca (211, 8,035)
  Cervix-SCC/AdenoCA -> cesc     (20,   892)
  Bladder-TCC        -> blca     (23, 2,987)
  Lung-SCC           -> lusc     (47, 4,783)
  Head-SCC           -> hnsc     (56, 4,373)
  Skin-Melanoma      -> skcm    (107,77,856)

Output:
  data/raw/pcawg/by_cancer/<bucket>_mutations.txt  (TCGA-schema MAF per cancer)
  data/raw/pcawg/by_cancer/parse_stats.json        (per-bucket counts + timings)

Usage:
  conda run -n quris python scripts/multi_enzyme/parse_pcawg.py
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "pcawg"
DOWNLOAD_DIR = RAW_DIR / "_download"
OUT_DIR = RAW_DIR / "by_cancer"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAF_PATH = DOWNLOAD_DIR / "data_mutations.txt"
CLIN_SAMPLE = DOWNLOAD_DIR / "data_clinical_sample.txt"

sys.stdout.reconfigure(line_buffering=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# PCAWG HISTOLOGY_ABBREVIATION -> our TCGA-style bucket
BUCKET_MAP = {
    "Liver-HCC": "lihc",
    "ColoRect-AdenoCA": "coadread",
    "Stomach-AdenoCA": "stad",
    "Eso-AdenoCA": "esca",
    "Breast-AdenoCA": "brca",
    "Breast-LobularCA": "brca",
    "Breast-DCIS": "brca",
    "Cervix-SCC": "cesc",
    "Cervix-AdenoCA": "cesc",
    "Bladder-TCC": "blca",
    "Lung-SCC": "lusc",
    "Head-SCC": "hnsc",
    "Skin-Melanoma": "skcm",
}

# The TCGA pipeline's parse_ct_mutations reads these columns. Keep them all.
REQUIRED_COLS = [
    "Hugo_Symbol", "Entrez_Gene_Id", "Chromosome", "Start_Position", "End_Position",
    "Strand", "Variant_Classification", "Variant_Type",
    "Reference_Allele", "Tumor_Seq_Allele1", "Tumor_Seq_Allele2",
    "Tumor_Sample_Barcode", "HGVSp_Short", "Consequence", "NCBI_Build",
]


def main():
    t0 = time.time()

    logger.info("Loading clinical sample metadata from %s", CLIN_SAMPLE)
    clin = pd.read_csv(CLIN_SAMPLE, sep="\t", comment="#", low_memory=False)
    clin["bucket"] = clin["HISTOLOGY_ABBREVIATION"].map(BUCKET_MAP)
    n_kept = int(clin["bucket"].notna().sum())
    logger.info("Mapped %d / %d PCAWG samples to %d buckets",
                n_kept, len(clin), clin["bucket"].dropna().nunique())

    sample_to_bucket = dict(zip(clin["SAMPLE_ID"], clin["bucket"]))

    logger.info("Loading MAF from %s", MAF_PATH)
    df = pd.read_csv(MAF_PATH, sep="\t", comment="#", low_memory=False)
    logger.info("  total rows: %d", len(df))

    # Keep only the cols we need (if present)
    present_cols = [c for c in REQUIRED_COLS if c in df.columns]
    missing_req = set(REQUIRED_COLS) - set(present_cols)
    if missing_req:
        logger.warning("  missing columns in PCAWG MAF (will be filled with NA): %s", sorted(missing_req))

    # Filter to C>T SNPs on either genomic strand
    is_snp = df["Variant_Type"] == "SNP"
    is_ct = (df["Reference_Allele"] == "C") & (df["Tumor_Seq_Allele2"] == "T")
    is_ga = (df["Reference_Allele"] == "G") & (df["Tumor_Seq_Allele2"] == "A")
    ct = df[is_snp & (is_ct | is_ga)].copy()
    logger.info("  C>T SNPs: %d", len(ct))

    # Add bucket
    ct["bucket"] = ct["Tumor_Sample_Barcode"].map(sample_to_bucket)
    ct = ct[ct["bucket"].notna()].copy()
    logger.info("  C>T SNPs in mapped buckets: %d (dropped %d unmapped)",
                len(ct), int(df[is_snp & (is_ct | is_ga)].shape[0] - len(ct)))

    # Ensure required columns exist for downstream (TCGA-schema) parsers
    for col in REQUIRED_COLS:
        if col not in ct.columns:
            ct[col] = ""

    # Normalize chromosome — downstream code expects "chr1" etc.
    ct["Chromosome"] = ct["Chromosome"].astype(str)
    if not ct["Chromosome"].str.startswith("chr").any():
        ct["Chromosome"] = "chr" + ct["Chromosome"]

    # NCBI_Build consistency (should be GRCh37 throughout)
    build_counts = ct["NCBI_Build"].value_counts().to_dict()
    logger.info("  NCBI_Build distribution: %s", build_counts)

    stats = {"parse_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "total_ct_snps": int(len(ct)),
             "per_bucket": {},
             "bucket_map": BUCKET_MAP}

    for bucket, sub in ct.groupby("bucket"):
        # Order columns to match TCGA MAF layout
        out_cols = REQUIRED_COLS + [c for c in ct.columns if c not in REQUIRED_COLS + ["bucket"]]
        out_cols = [c for c in out_cols if c in sub.columns]
        out = sub[out_cols].copy()
        outp = OUT_DIR / f"{bucket}_pcawg_mutations.txt"
        out.to_csv(outp, sep="\t", index=False)
        stats["per_bucket"][bucket] = {
            "n_ct_snps": int(len(sub)),
            "n_unique_samples": int(sub["Tumor_Sample_Barcode"].nunique()),
            "n_unique_positions": int(sub[["Chromosome", "Start_Position"]].drop_duplicates().shape[0]),
            "output_file": str(outp.relative_to(PROJECT_ROOT)),
        }
        logger.info("  wrote %s (%d rows, %d unique positions, %d samples)",
                    outp.name, len(sub),
                    stats["per_bucket"][bucket]["n_unique_positions"],
                    stats["per_bucket"][bucket]["n_unique_samples"])

    with (OUT_DIR / "parse_stats.json").open("w") as f:
        json.dump(stats, f, indent=2)

    logger.info("Done in %.1fs. Stats -> %s", time.time() - t0, OUT_DIR / "parse_stats.json")


if __name__ == "__main__":
    main()
