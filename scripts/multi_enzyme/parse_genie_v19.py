#!/usr/bin/env python
"""
Stage AACR Project GENIE v19.0-public for Phase-3 NN scoring.

Pipeline stages (controlled by --stage flag):
  1. download   - Fetch release files from Synapse (requires auth)
  2. parse      - Filter C>T/G>A SNVs, map samples → cancer buckets, split per-cancer
  3. liftover   - GRCh37 → GRCh38 with pyliftover; drop unmappable
  4. panels     - Copy capture BEDs into panels/; report per-panel coverage
  5. report     - Write STAGING_REPORT.md with counts and liftover loss rates
  all (default) - Run all five stages in order

Outputs:
  data/raw/genie_v19/_download/        Raw downloaded files
  data/raw/genie_v19/by_cancer/*.tsv   Per-cancer MAF-schema tables (hg19)
  data/raw/genie_v19/by_cancer/*_hg38.tsv  Lifted to hg38
  data/raw/genie_v19/panels/           Per-center capture BEDs
  data/raw/genie_v19/RUN_LOG.txt       Append-only command log
  experiments/multi_enzyme/outputs/genie_v19/STAGING_REPORT.md

Authentication: requires ~/.synapseConfig or SYNAPSE_AUTH_TOKEN.
Get a token at https://www.synapse.org/Profile:v/settings (Personal Access Token).
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GENIE_ROOT = PROJECT_ROOT / "data/raw/genie_v19"
DOWNLOAD_DIR = GENIE_ROOT / "_download"
BY_CANCER_DIR = GENIE_ROOT / "by_cancer"
PANELS_DIR = GENIE_ROOT / "panels"
REPORT_DIR = PROJECT_ROOT / "experiments/multi_enzyme/outputs/genie_v19"
RUN_LOG = GENIE_ROOT / "RUN_LOG.txt"

# GENIE v19.0-public release parent (confirmed from Synapse)
# syn7222066 is the GENIE Public-Access project; v19 lives in syn60548225 (Sep-2024 release).
# Agents SHOULD re-resolve at runtime via synapseclient.get_children(syn7222066).
GENIE_PROJECT_ID = "syn7222066"
GENIE_V19_RELEASE_ID = "syn60548225"   # v19.0-public (Sep 2024). Override with --release-id if a newer release exists.

REQUIRED_FILES = [
    "data_mutations_extended.txt",
    "data_clinical_sample.txt",
    "data_clinical_patient.txt",
    "genomic_information.txt",       # per-panel capture regions (all centers combined)
    "assay_information.txt",
]

# ONCOTREE code → our cancer bucket. See http://oncotree.mskcc.org (2021_11_02 codes).
# Order matters: we use startswith on hierarchy codes where appropriate.
# These buckets match the existing TCGA pipeline (COADREAD, STAD, ESCA, BLCA, BRCA,
# CESC, LUSC, HNSC, SKCM, LIHC) plus PAAD (new for GENIE).
ONCOTREE_BUCKETS = {
    # Colorectal (TCGA COADREAD = COAD + READ)
    "COADREAD": {"COADREAD", "COAD", "READ", "CRC", "MACR", "RCC", "CAIS",
                 "SRCCR", "MUP", "COADREADNOS"},
    # Gastric
    "STAD":     {"STAD", "STAS", "ESTT", "GIST", "DSTAD", "ISTAD", "MSTAD",
                 "PSTAD", "TSTAD", "HGSOS", "STADNOS"},
    # Esophageal
    "ESCA":     {"ESCA", "EGC", "ESCC", "EAC", "ESMM", "GEJ"},
    # Bladder / Urothelial
    "BLCA":     {"BLCA", "BLADDER", "UCU", "UTUC", "UCC", "SCBC", "BLSC",
                 "UMEC", "UPA", "USCC", "UTC", "UCA"},
    # Breast
    "BRCA":     {"BRCA", "IDC", "ILC", "MDLC", "BRCANOS", "BRCNOS", "MBC",
                 "PBS", "IMMC", "MPT", "BREAST"},
    # Cervical
    "CESC":     {"CESC", "CEAD", "CESC", "CEAS", "CEAD", "CEEN", "CEGCC",
                 "CEMN", "CESC", "CESU", "CEMU", "SCCE"},
    # Lung squamous (exclude lung adeno)
    "LUSC":     {"LUSC", "LUAS"},
    # Head & neck squamous
    "HNSC":     {"HNSC", "HNSCUP", "OCSC", "OPHSC", "LXSC", "HPHSC", "HEAD_NECK",
                 "OHNCA", "HNMUCM", "HNNE"},
    # Skin cutaneous melanoma
    "SKCM":     {"SKCM", "MEL", "CSCC", "ACRM", "DESM", "UM", "MUP"},
    # Liver hepatocellular
    "LIHC":     {"LIHC", "HCC", "HB", "FL-HCC", "CHOL", "IHCH", "EHCH",
                 "GBC", "CHL"},
    # Pancreatic adenocarcinoma (NEW — not in TCGA pipeline yet)
    "PAAD":     {"PAAD", "PDAC", "PANET", "PAAC", "PAACC", "UCP", "PB"},
}


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("genie_v19")


log = setup_logging()


def run_log(msg: str):
    GENIE_ROOT.mkdir(parents=True, exist_ok=True)
    with open(RUN_LOG, "a") as f:
        f.write(f"[{datetime.now().isoformat(timespec='seconds')}] {msg}\n")
    log.info(msg)


# ---------------------------------------------------------------------------
# STAGE 1: download
# ---------------------------------------------------------------------------
def stage_download(release_id: str):
    try:
        import synapseclient
    except ImportError:
        log.error("synapseclient not installed. Run: pip install synapseclient")
        sys.exit(2)

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    syn = synapseclient.Synapse()
    try:
        syn.login(silent=True)
    except Exception as e:
        log.error("Synapse auth failed: %s", e)
        log.error("Set SYNAPSE_AUTH_TOKEN or create ~/.synapseConfig with a PAT.")
        log.error("Get a token at https://www.synapse.org/Profile:v/settings")
        sys.exit(3)

    run_log(f"Synapse login OK; resolving release {release_id}")

    # List children of the release folder
    children = list(syn.getChildren(release_id))
    name_to_entity = {c["name"]: c for c in children}
    run_log(f"Release {release_id} has {len(children)} files")

    missing = [f for f in REQUIRED_FILES if f not in name_to_entity]
    # Some releases bundle panel BEDs inside a sub-folder called 'genie_panels' or
    # expose them as individual data_gene_panel_*.txt files. Log both patterns.
    panel_entities = [c for c in children
                      if c["name"].startswith("data_gene_panel_")
                      or c["name"] == "genie_panels"
                      or c["name"] == "genomic_information.txt"]
    run_log(f"Panel-related entities: {len(panel_entities)}")

    if missing:
        run_log(f"WARNING: missing from release listing: {missing}")

    # Download required files + all panel entities
    to_fetch = [name_to_entity[n] for n in REQUIRED_FILES if n in name_to_entity]
    to_fetch.extend(c for c in panel_entities if c not in to_fetch)

    for ent in to_fetch:
        run_log(f"Downloading {ent['name']} ({ent['id']})")
        syn.get(ent["id"], downloadLocation=str(DOWNLOAD_DIR),
                ifcollision="keep.local")

    run_log(f"Download complete → {DOWNLOAD_DIR}")


# ---------------------------------------------------------------------------
# STAGE 2: parse
# ---------------------------------------------------------------------------
def _open_maybe_gz(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "rt")


def _build_oncotree_lookup():
    lookup = {}
    for bucket, codes in ONCOTREE_BUCKETS.items():
        for c in codes:
            lookup[c.upper()] = bucket
    return lookup


def stage_parse():
    maf_path = DOWNLOAD_DIR / "data_mutations_extended.txt"
    sample_path = DOWNLOAD_DIR / "data_clinical_sample.txt"
    if not maf_path.exists() or not sample_path.exists():
        log.error("Required files missing. Run --stage download first.")
        sys.exit(4)

    BY_CANCER_DIR.mkdir(parents=True, exist_ok=True)
    lookup = _build_oncotree_lookup()

    # --- Load clinical sample table (skip # header lines)
    run_log(f"Loading {sample_path}")
    sample_df = pd.read_csv(sample_path, sep="\t", comment="#", low_memory=False)
    # GENIE's canonical column names
    sample_id_col = "SAMPLE_ID" if "SAMPLE_ID" in sample_df.columns else "Tumor_Sample_Barcode"
    onco_col = "ONCOTREE_CODE" if "ONCOTREE_CODE" in sample_df.columns else "CANCER_TYPE"
    panel_col = "SEQ_ASSAY_ID" if "SEQ_ASSAY_ID" in sample_df.columns else None

    sample_df["BUCKET"] = sample_df[onco_col].astype(str).str.upper().map(lookup)
    bucket_counts = sample_df["BUCKET"].value_counts(dropna=False)
    run_log(f"Sample bucket distribution:\n{bucket_counts.to_string()}")

    sample_to_bucket = dict(zip(sample_df[sample_id_col], sample_df["BUCKET"]))
    sample_to_panel = (dict(zip(sample_df[sample_id_col], sample_df[panel_col]))
                       if panel_col else {})

    # --- Stream MAF
    # GENIE's MAF uses the standard schema (same cols as TCGA). Chunked read.
    keep_cols_out = [
        "Hugo_Symbol", "Chromosome", "Start_Position", "End_Position",
        "Strand", "Variant_Classification", "Variant_Type",
        "Reference_Allele", "Tumor_Seq_Allele1", "Tumor_Seq_Allele2",
        "Tumor_Sample_Barcode", "HGVSp_Short",
    ]

    # Per-bucket in-memory accumulators; each bucket typically <5M rows before filtering
    # so we stream chunks and append to per-bucket CSVs via line-writer.
    writers = {}        # bucket -> open file handle
    header_written = {}
    panel_by_bucket = {}  # bucket -> Counter of panels
    row_counts = {b: 0 for b in ONCOTREE_BUCKETS}
    total_rows_scanned = 0
    total_ctga = 0

    run_log(f"Streaming {maf_path} (this may take several minutes)")
    chunk_iter = pd.read_csv(
        maf_path, sep="\t", comment="#", low_memory=False,
        chunksize=200_000, dtype=str,
    )
    from collections import Counter

    try:
        for ci, chunk in enumerate(chunk_iter):
            total_rows_scanned += len(chunk)
            # Filter SNVs C>T or G>A
            snv_mask = chunk.get("Variant_Type", pd.Series(dtype=str)) == "SNP"
            ref = chunk["Reference_Allele"]
            alt = chunk["Tumor_Seq_Allele2"]
            ct = (ref == "C") & (alt == "T")
            ga = (ref == "G") & (alt == "A")
            sub = chunk[snv_mask & (ct | ga)].copy()
            if len(sub) == 0:
                continue
            total_ctga += len(sub)

            sub["BUCKET"] = sub["Tumor_Sample_Barcode"].map(sample_to_bucket)
            sub = sub[sub["BUCKET"].notna()]
            if len(sub) == 0:
                continue

            # Keep only the columns our downstream parser expects
            out_cols = [c for c in keep_cols_out if c in sub.columns]
            sub_out = sub[out_cols + ["BUCKET"]].copy()

            for bucket, g in sub_out.groupby("BUCKET"):
                path = BY_CANCER_DIR / f"{bucket}_mutations.tsv"
                if bucket not in writers:
                    writers[bucket] = open(path, "w")
                    g.drop(columns=["BUCKET"]).to_csv(writers[bucket], sep="\t",
                                                      index=False, header=True)
                    header_written[bucket] = True
                else:
                    g.drop(columns=["BUCKET"]).to_csv(writers[bucket], sep="\t",
                                                      index=False, header=False)
                row_counts[bucket] += len(g)
                if sample_to_panel:
                    cnt = panel_by_bucket.setdefault(bucket, Counter())
                    for sid in g["Tumor_Sample_Barcode"]:
                        p = sample_to_panel.get(sid)
                        if p:
                            cnt[p] += 1

            if (ci + 1) % 10 == 0:
                run_log(f"  processed {(ci+1)*200_000:,} lines, {total_ctga:,} C>T/G>A rows")
    finally:
        for h in writers.values():
            h.close()

    run_log(f"Scanned {total_rows_scanned:,} MAF rows; {total_ctga:,} were C>T/G>A SNVs")
    run_log(f"Per-cancer row counts: {row_counts}")

    # Save per-cancer panel counts
    panel_summary = {b: dict(c) for b, c in panel_by_bucket.items()}
    with open(GENIE_ROOT / "per_cancer_panel_counts.json", "w") as f:
        json.dump(panel_summary, f, indent=2)

    with open(GENIE_ROOT / "parse_stats.json", "w") as f:
        json.dump({
            "total_rows_scanned": total_rows_scanned,
            "total_ctga_snvs": total_ctga,
            "per_cancer_rows": row_counts,
            "bucket_sample_counts": bucket_counts.dropna().to_dict(),
        }, f, indent=2)


# ---------------------------------------------------------------------------
# STAGE 3: liftover GRCh37 → GRCh38
# ---------------------------------------------------------------------------
def stage_liftover():
    try:
        from pyliftover import LiftOver
    except ImportError:
        log.error("pyliftover not installed. Run: pip install pyliftover")
        sys.exit(5)

    lo = LiftOver("hg19", "hg38")
    stats = {}
    for tsv in sorted(BY_CANCER_DIR.glob("*_mutations.tsv")):
        if tsv.name.endswith("_hg38.tsv"):
            continue
        out_path = tsv.with_name(tsv.stem + "_hg38.tsv")
        run_log(f"Liftover {tsv.name} → {out_path.name}")
        df = pd.read_csv(tsv, sep="\t", low_memory=False, dtype=str)
        if len(df) == 0:
            df.to_csv(out_path, sep="\t", index=False)
            stats[tsv.stem] = {"input": 0, "lifted": 0, "lost": 0, "loss_pct": 0.0}
            continue

        chroms = df["Chromosome"].astype(str)
        chroms = chroms.where(chroms.str.startswith("chr"), "chr" + chroms)
        starts = df["Start_Position"].astype(int)
        # pyliftover wants 0-based coords; GENIE Start_Position is 1-based.
        new_chrom, new_pos = [], []
        for ch, pos in zip(chroms.values, starts.values):
            r = lo.convert_coordinate(ch, pos - 1)
            if r:
                new_chrom.append(r[0][0])
                new_pos.append(r[0][1] + 1)  # back to 1-based
            else:
                new_chrom.append(None)
                new_pos.append(None)
        df["Chromosome"] = new_chrom
        df["Start_Position"] = new_pos
        # Adjust End_Position (all SNVs so end = start)
        if "End_Position" in df.columns:
            df["End_Position"] = df["Start_Position"]

        n_in = len(df)
        df = df[df["Chromosome"].notna()].copy()
        n_out = len(df)
        # Strip "chr" prefix to match TCGA parser's input expectation
        # (parser re-adds it via startswith check). Actually TCGA parser auto-adds chr,
        # so either convention works — we keep the 'chr' prefix for safety.
        df.to_csv(out_path, sep="\t", index=False)
        stats[tsv.stem] = {
            "input": n_in, "lifted": n_out, "lost": n_in - n_out,
            "loss_pct": 100.0 * (n_in - n_out) / max(n_in, 1),
        }
        run_log(f"  {tsv.stem}: {n_in} → {n_out} ({stats[tsv.stem]['loss_pct']:.2f}% lost)")

    with open(GENIE_ROOT / "liftover_stats.json", "w") as f:
        json.dump(stats, f, indent=2)


# ---------------------------------------------------------------------------
# STAGE 4: panels
# ---------------------------------------------------------------------------
def stage_panels():
    PANELS_DIR.mkdir(parents=True, exist_ok=True)
    # GENIE v19 ships capture regions in a single file `genomic_information.txt`
    # with columns: Chromosome, Start_Position, End_Position, Hugo_Symbol,
    # SEQ_ASSAY_ID, Feature_Type, includeInPanel.
    gi = DOWNLOAD_DIR / "genomic_information.txt"
    if not gi.exists():
        run_log(f"WARNING: {gi} missing; panels stage skipped")
        return

    df = pd.read_csv(gi, sep="\t", low_memory=False)
    panel_col = "SEQ_ASSAY_ID"
    if panel_col not in df.columns:
        run_log(f"WARNING: {panel_col} missing from genomic_information.txt")
        return

    summary = {}
    for panel_id, g in df.groupby(panel_col):
        if "includeInPanel" in g.columns:
            g = g[g["includeInPanel"].astype(str).str.lower().isin(["true", "yes", "1"])]
        g_bed = g[["Chromosome", "Start_Position", "End_Position"]].copy()
        g_bed["Chromosome"] = g_bed["Chromosome"].astype(str)
        g_bed["Chromosome"] = g_bed["Chromosome"].where(
            g_bed["Chromosome"].str.startswith("chr"),
            "chr" + g_bed["Chromosome"])
        # BED is 0-based half-open
        g_bed["Start_Position"] = g_bed["Start_Position"].astype(int) - 1
        g_bed["End_Position"] = g_bed["End_Position"].astype(int)
        out = PANELS_DIR / f"{panel_id}.bed"
        g_bed.to_csv(out, sep="\t", index=False, header=False)
        covered_bp = int((g_bed["End_Position"] - g_bed["Start_Position"]).sum())
        summary[panel_id] = {"intervals": len(g_bed), "covered_bp": covered_bp}
        run_log(f"  panel {panel_id}: {len(g_bed):,} intervals, {covered_bp:,} bp")

    with open(GENIE_ROOT / "panel_coverage.json", "w") as f:
        json.dump(summary, f, indent=2)


# ---------------------------------------------------------------------------
# STAGE 5: report
# ---------------------------------------------------------------------------
def stage_report(release_id: str):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "STAGING_REPORT.md"

    def _load_json(p: Path):
        return json.loads(p.read_text()) if p.exists() else {}

    parse_stats = _load_json(GENIE_ROOT / "parse_stats.json")
    liftover_stats = _load_json(GENIE_ROOT / "liftover_stats.json")
    panel_coverage = _load_json(GENIE_ROOT / "panel_coverage.json")
    panel_counts = _load_json(GENIE_ROOT / "per_cancer_panel_counts.json")

    lines = []
    lines.append(f"# GENIE v19.0-public staging report\n")
    lines.append(f"- Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- Release: {release_id}")
    lines.append(f"- Download dir: `{DOWNLOAD_DIR.relative_to(PROJECT_ROOT)}`")
    lines.append(f"- Per-cancer dir: `{BY_CANCER_DIR.relative_to(PROJECT_ROOT)}`")
    lines.append(f"- Panel BEDs: `{PANELS_DIR.relative_to(PROJECT_ROOT)}`\n")

    lines.append("## 1. Downloaded files\n")
    if DOWNLOAD_DIR.exists():
        for p in sorted(DOWNLOAD_DIR.iterdir()):
            size_mb = p.stat().st_size / 1e6
            lines.append(f"- `{p.name}` — {size_mb:.1f} MB")
    lines.append("")

    lines.append("## 2. Parse stats\n")
    if parse_stats:
        lines.append(f"- Total MAF rows scanned: {parse_stats.get('total_rows_scanned', 0):,}")
        lines.append(f"- Total C>T / G>A SNVs: {parse_stats.get('total_ctga_snvs', 0):,}\n")
        lines.append("### Per-cancer mutation counts (hg19)\n")
        lines.append("| Cancer | Mutations |")
        lines.append("|---|---:|")
        for c, n in sorted(parse_stats.get("per_cancer_rows", {}).items(),
                           key=lambda x: -x[1]):
            lines.append(f"| {c} | {n:,} |")
        lines.append("")
        lines.append("### Sample counts per cancer bucket (from data_clinical_sample.txt)\n")
        lines.append("| Cancer | Samples |")
        lines.append("|---|---:|")
        for c, n in sorted(parse_stats.get("bucket_sample_counts", {}).items(),
                           key=lambda x: -x[1]):
            lines.append(f"| {c} | {int(n):,} |")
        lines.append("")
    else:
        lines.append("*No parse_stats.json — parse stage has not run.*\n")

    lines.append("## 3. Liftover GRCh37 → GRCh38\n")
    if liftover_stats:
        lines.append("| File | Input | Lifted | Lost | Loss % |")
        lines.append("|---|---:|---:|---:|---:|")
        for name, s in sorted(liftover_stats.items()):
            lines.append(f"| {name} | {s['input']:,} | {s['lifted']:,} | "
                         f"{s['lost']:,} | {s['loss_pct']:.2f}% |")
        lines.append("")
    else:
        lines.append("*No liftover_stats.json — liftover stage has not run.*\n")

    lines.append("## 4. Per-panel capture coverage\n")
    if panel_coverage:
        lines.append("| SEQ_ASSAY_ID | Intervals | Covered bp |")
        lines.append("|---|---:|---:|")
        for p, s in sorted(panel_coverage.items()):
            lines.append(f"| {p} | {s['intervals']:,} | {s['covered_bp']:,} |")
        lines.append("")
    else:
        lines.append("*No panel_coverage.json — panels stage has not run.*\n")

    lines.append("## 5. Per-cancer × per-panel sample counts\n")
    if panel_counts:
        for cancer, panels in sorted(panel_counts.items()):
            top = sorted(panels.items(), key=lambda x: -x[1])[:15]
            lines.append(f"### {cancer}")
            lines.append("| SEQ_ASSAY_ID | Mutations |")
            lines.append("|---|---:|")
            for pid, n in top:
                lines.append(f"| {pid} | {n:,} |")
            lines.append("")

    lines.append("## 6. Next step — scoring driver\n")
    lines.append("```bash")
    lines.append("# PLACEHOLDER: Phase-3 NN scoring across GENIE-per-cancer MAFs")
    lines.append("# To be implemented next session. Expected invocation:")
    lines.append("#   conda activate quris")
    lines.append("#   python experiments/multi_enzyme/exp_enzyme_tissue_disease.py \\")
    lines.append("#       --cohort genie_v19 \\")
    lines.append("#       --maf-dir data/raw/genie_v19/by_cancer \\")
    lines.append("#       --maf-suffix _hg38.tsv \\")
    lines.append("#       --panels-dir data/raw/genie_v19/panels \\")
    lines.append("#       --cancers COADREAD STAD ESCA BLCA BRCA CESC LUSC HNSC SKCM LIHC PAAD")
    lines.append("```\n")

    out.write_text("\n".join(lines))
    run_log(f"Wrote staging report → {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Stage GENIE v19.0-public.")
    ap.add_argument("--stage", choices=["download", "parse", "liftover",
                                        "panels", "report", "all"],
                    default="all")
    ap.add_argument("--release-id", default=GENIE_V19_RELEASE_ID,
                    help="Synapse ID of the v19.0-public release folder")
    args = ap.parse_args()

    GENIE_ROOT.mkdir(parents=True, exist_ok=True)
    run_log(f"=== parse_genie_v19.py --stage {args.stage} --release-id {args.release_id} ===")

    if args.stage in ("download", "all"):
        stage_download(args.release_id)
    if args.stage in ("parse", "all"):
        stage_parse()
    if args.stage in ("liftover", "all"):
        stage_liftover()
    if args.stage in ("panels", "all"):
        stage_panels()
    if args.stage in ("report", "all"):
        stage_report(args.release_id)

    run_log(f"=== stage {args.stage} done ===")


if __name__ == "__main__":
    main()
