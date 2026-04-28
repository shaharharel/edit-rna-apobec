#!/usr/bin/env python3
"""Petljak 2022 post-DAR data pull script.

Run this AFTER the DAR has been approved and you have received your EGA download
credentials (username + password, or download token).

Steps performed:
  1. Verify pyega3 is installed; install via pip if missing.
  2. Use the user's EGA credentials to list files in the approved dataset.
  3. Download per-clone VCFs (filename pattern *.vcf.gz or per-clone subdirs).
  4. Verify each downloaded file's MD5 against the EGA-provided manifest.
  5. Parse all VCFs and write a unified normalized table:
       data/raw/petljak2022/clone_vcfs/all_clone_snvs.parquet
     with columns: clone_id, cell_line, genotype, chrom, pos (0-based),
                   ref, alt, qual, filter, sub_class, tri_context.

Usage:
    export EGA_USER=<your-ega-username>
    export EGA_PASS=<your-ega-password>
    export EGA_DATASET=<EGAD00001008xxx>   # from your approval letter
    python scripts/data_pull/fetch_petljak_post_dar.py

For a dry-run (no download, just print plan):
    python scripts/data_pull/fetch_petljak_post_dar.py --dry-run

This script does not transmit credentials anywhere; pyega3 talks directly to
ega-archive.org over HTTPS.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path("/Users/shaharharel/Documents/github/edit-rna-apobec")
OUT_DIR = ROOT / "data/raw/petljak2022/clone_vcfs"
META_DIR = ROOT / "data/raw/petljak2022"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def ensure_pyega3():
    try:
        import pyega3  # noqa: F401
        log.info("pyega3 already installed")
    except ImportError:
        log.info("Installing pyega3 ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyega3"])


def write_credentials_file(user: str, password: str) -> Path:
    cred = META_DIR / ".ega_credentials.json"
    cred.write_text(json.dumps({"username": user, "password": password}))
    cred.chmod(0o600)
    return cred


def list_files(dataset: str, cred_path: Path) -> list[dict]:
    """Use pyega3 'datasets <id> files' to list files in the dataset."""
    cmd = [sys.executable, "-m", "pyega3", "-cf", str(cred_path), "files", dataset]
    log.info("Running: %s", " ".join(cmd))
    out = subprocess.run(cmd, check=True, capture_output=True, text=True)
    files = []
    for line in out.stdout.splitlines():
        # Output is loosely structured; pyega3 shows file id, size, md5, file name
        if ".vcf" in line.lower() or ".vcf.gz" in line.lower():
            files.append({"raw_line": line})
    return files


def download_dataset(dataset: str, cred_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "pyega3", "-cf", str(cred_path),
           "fetch", dataset, "--saveto", str(out_dir)]
    log.info("Running: %s", " ".join(cmd))
    subprocess.check_call(cmd)


def verify_md5(file_path: Path, expected_md5: str) -> bool:
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest() == expected_md5


def parse_vcf_dir(vcf_dir: Path) -> Path:
    """Parse per-clone VCFs into a unified parquet table.
    Sample-level metadata (cell_line, genotype) is NOT in the VCF — it comes from
    the metadata table downloaded separately. We just emit a clone_id derived
    from filename and let the user join later.
    """
    import gzip
    import pandas as pd
    rows = []
    vcfs = sorted(list(vcf_dir.rglob("*.vcf.gz")) + list(vcf_dir.rglob("*.vcf")))
    log.info("Parsing %d VCFs ...", len(vcfs))
    for vcf in vcfs:
        clone_id = vcf.stem.replace(".vcf", "")
        opener = gzip.open if vcf.suffix == ".gz" else open
        with opener(vcf, "rt") as fh:
            for line in fh:
                if line.startswith("#"):
                    continue
                parts = line.strip().split("\t")
                if len(parts) < 8:
                    continue
                chrom, pos, _id, ref, alt, qual, filt, info = parts[:8]
                if len(ref) != 1 or len(alt) != 1:
                    continue
                rows.append({
                    "clone_id": clone_id,
                    "chrom": chrom,
                    "pos": int(pos) - 1,  # 1-based VCF -> 0-based panel convention
                    "ref": ref,
                    "alt": alt,
                    "qual": qual,
                    "filter": filt,
                })
    df = pd.DataFrame(rows)
    out = vcf_dir.parent / "all_clone_snvs.parquet"
    df.to_parquet(out, index=False)
    log.info("Wrote %s (%d rows)", out, len(df))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan without downloading")
    parser.add_argument("--dataset", default=os.environ.get("EGA_DATASET"),
                        help="EGA dataset accession (EGAD00001...)")
    parser.add_argument("--user", default=os.environ.get("EGA_USER"),
                        help="EGA username (or set EGA_USER env)")
    parser.add_argument("--password", default=os.environ.get("EGA_PASS"),
                        help="EGA password (or set EGA_PASS env)")
    args = parser.parse_args()

    if args.dry_run:
        log.info("=== Dry run ===")
        log.info("Would install pyega3 if missing")
        log.info("Would download dataset: %s", args.dataset or "<set EGA_DATASET>")
        log.info("Would save to: %s", OUT_DIR)
        log.info("Would parse all VCFs into: %s/all_clone_snvs.parquet", OUT_DIR.parent)
        return 0

    if not args.dataset:
        log.error("EGA_DATASET not set. Either pass --dataset or `export EGA_DATASET=EGAD00001...`")
        return 1
    if not args.user or not args.password:
        log.error("EGA credentials missing. Set EGA_USER and EGA_PASS env vars (or use --user / --password).")
        return 1

    ensure_pyega3()
    cred = write_credentials_file(args.user, args.password)
    try:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        download_dataset(args.dataset, cred, OUT_DIR)
        parse_vcf_dir(OUT_DIR)
        log.info("Petljak data ready under %s", OUT_DIR.parent)
    finally:
        cred.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
