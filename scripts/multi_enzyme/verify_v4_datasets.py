#!/usr/bin/env python
"""Verify v4 multi-enzyme datasets and emit a markdown summary.

Reports:
  - Cancer C>T trinuc distribution (16 bins)
  - CDS-C trinuc distribution (16 bins)
  - Side-by-side neg trinucs: v3, v4_cancer, v4_cds
  - File paths and counts
  - Jaccard overlap between v4_cancer and v4_cds negatives
  - Whether RNA-FM is computed or deferred

Output: data/processed/multi_enzyme/V4_DATA_PREP.md

Usage:
    conda run -n quris python scripts/multi_enzyme/verify_v4_datasets.py
"""

import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ME_DIR = PROJECT_ROOT / "data/processed/multi_enzyme"
EMB_DIR = PROJECT_ROOT / "data/processed/embeddings"

V3_SPLITS = ME_DIR / "splits_multi_enzyme_v3_with_negatives.csv"
V3_SEQS = ME_DIR / "multi_enzyme_sequences_v3_with_negatives.json"
V3_LOOP = ME_DIR / "loop_position_per_site_v3.csv"
V3_STRUCT = EMB_DIR / "structure_cache_multi_enzyme_v3.npz"

CANCER_TRINUC_CSV = ME_DIR / "cancer_ct_trinuc_distribution.csv"
CDS_TRINUC_CSV = ME_DIR / "cds_c_trinuc_distribution.csv"

OUT_MD = ME_DIR / "V4_DATA_PREP.md"

ALL_TRINUCS = [f"{a}C{b}" for a in "ACGT" for b in "ACGT"]


def trinuc_of_seq(seq: str) -> str:
    if not seq or len(seq) < 102:
        return ""
    s = seq.upper().replace("U", "T")
    return s[99:102]


def trinuc_dist_from_seqs(rows: pd.DataFrame, seqs: dict) -> Counter:
    cnt = Counter()
    for sid in rows["site_id"].astype(str):
        s = seqs.get(sid)
        if not s:
            continue
        t = trinuc_of_seq(s)
        if len(t) == 3 and t[1] == "C" and "N" not in t:
            cnt[t] += 1
    return cnt


def trinuc_table(cnt: Counter) -> dict:
    total = sum(cnt.values())
    return {t: 100.0 * cnt.get(t, 0) / total if total else 0.0 for t in ALL_TRINUCS}


def fmt_pct(v: float) -> str:
    return f"{v:5.2f}"


def main():
    lines = []
    lines.append("# v4 Data Prep — Two Trinuc-Matched Negative Sets\n")
    lines.append(f"_Generated {pd.Timestamp.now().isoformat(timespec='seconds')}_\n")
    lines.append("\n## Overview\n")
    lines.append(
        "v3 negatives were biased: positives 38.9% TC vs negatives 57.2% TC; "
        "positives 19.8% CpG vs negatives 9.1% CpG. The model learned anti-TCW "
        "polarity. v4 corrects this with two parallel negative sets:\n"
    )
    lines.append("- **v4_cancer_matched** — negatives match TCGA + PCAWG-coding pan-cancer C>T trinuc distribution (transfer claim).\n")
    lines.append("- **v4_cds_unbiased** — negatives match the genome CDS-C trinuc distribution (predictor claim).\n")
    lines.append("- APOBEC1 sites (v3 enzyme=='Neither', n=206) are excluded — no DNA-editing analog.\n")
    lines.append(f"- Random seed: 20260427.\n")

    # ----------------------------------------------------------------
    # 1. Cancer + CDS trinuc distributions
    # ----------------------------------------------------------------
    cancer = pd.read_csv(CANCER_TRINUC_CSV)
    cds = pd.read_csv(CDS_TRINUC_CSV)
    cancer_frac = dict(zip(cancer["trinuc"], cancer["fraction"] * 100))
    cds_frac = dict(zip(cds["trinuc"], cds["fraction"] * 100))

    lines.append("\n## Trinucleotide distributions (16 bins, strand-collapsed N-C-N)\n")
    lines.append("| Trinuc | Cancer C>T % | CDS-C % |\n")
    lines.append("|--------|------------:|--------:|\n")
    for t in ALL_TRINUCS:
        lines.append(f"| {t} | {cancer_frac.get(t, 0):5.2f} | {cds_frac.get(t, 0):5.2f} |\n")
    cancer_total = int(cancer["count"].sum())
    cds_total = int(cds["count"].sum())
    lines.append(f"\n_Cancer total: {cancer_total:,} C>T mutations across 10 cancers (TCGA+PCAWG)._\n")
    lines.append(f"_CDS-C total: {cds_total:,} C positions in pan-cancer CDS panel (hg19, 0-indexed)._\n")

    # ----------------------------------------------------------------
    # 2. v3 vs v4_cancer vs v4_cds negative trinuc distributions
    # ----------------------------------------------------------------
    logger.info("Loading v3 + sequences ...")
    v3 = pd.read_csv(V3_SPLITS)
    with open(V3_SEQS) as f:
        v3_seqs = json.load(f)

    pos_full = v3[v3["is_edited"] == 1]
    pos_filtered = pos_full[pos_full["enzyme"] != "Neither"]
    n_pos_v3 = len(pos_full)
    n_pos_v4 = len(pos_filtered)
    excluded = n_pos_v3 - n_pos_v4

    neg_v3 = v3[v3["is_edited"] == 0]

    versions = {}
    for ver in ["cancer_matched", "cds_unbiased"]:
        sp = ME_DIR / f"splits_multi_enzyme_v4_{ver}.csv"
        sq = ME_DIR / f"multi_enzyme_sequences_v4_{ver}.json"
        lp = ME_DIR / f"loop_position_per_site_v4_{ver}.csv"
        st = EMB_DIR / f"structure_cache_multi_enzyme_v4_{ver}.npz"
        if not (sp.exists() and sq.exists()):
            logger.warning("Missing v4 %s files; skipping", ver)
            continue
        df = pd.read_csv(sp)
        with open(sq) as f:
            seqs = json.load(f)
        versions[ver] = {"df": df, "seqs": seqs, "splits": sp, "seqs_path": sq,
                          "loop": lp, "struct": st}
        logger.info("Loaded v4_%s: %d total (%d pos, %d neg)", ver, len(df),
                    int((df["is_edited"] == 1).sum()),
                    int((df["is_edited"] == 0).sum()))

    # Compute neg trinuc tables
    cnt_v3_neg = trinuc_dist_from_seqs(neg_v3, v3_seqs)
    table_v3_neg = trinuc_table(cnt_v3_neg)

    pos_table = trinuc_table(trinuc_dist_from_seqs(pos_filtered, v3_seqs))

    neg_tables = {}
    for ver, data in versions.items():
        df = data["df"]; seqs = data["seqs"]
        neg = df[df["is_edited"] == 0]
        cnt = trinuc_dist_from_seqs(neg, seqs)
        neg_tables[ver] = trinuc_table(cnt)

    lines.append("\n## v3 vs v4 negative trinucleotide distributions\n")
    lines.append("All percentages computed from the actual sequences in the JSON files.\n\n")
    lines.append("| Trinuc | v4 pos % | v3 neg % | v4_cancer neg % | Cancer target % | Δ_cancer | v4_cds neg % | CDS target % | Δ_cds |\n")
    lines.append("|--------|---------:|--------:|----------------:|----------------:|---------:|-------------:|-------------:|------:|\n")
    for t in ALL_TRINUCS:
        p = pos_table.get(t, 0)
        v3p = table_v3_neg.get(t, 0)
        c = neg_tables.get("cancer_matched", {}).get(t, 0)
        ct = cancer_frac.get(t, 0)
        d = neg_tables.get("cds_unbiased", {}).get(t, 0)
        dt = cds_frac.get(t, 0)
        lines.append(f"| {t} | {p:5.2f} | {v3p:5.2f} | {c:5.2f} | {ct:5.2f} | {c-ct:+5.2f} | {d:5.2f} | {dt:5.2f} | {d-dt:+5.2f} |\n")

    # TC/CpG aggregated
    def agg_tc(tab):
        return sum(tab.get(f"TC{x}", 0) for x in "ACGT")
    def agg_cpg(tab):
        return sum(tab.get(f"{x}CG", 0) for x in "ACGT")

    lines.append("\n### Aggregated metrics\n")
    lines.append("| Subset | TC% | CpG% |\n|---|---:|---:|\n")
    lines.append(f"| v4 positives | {agg_tc(pos_table):5.2f} | {agg_cpg(pos_table):5.2f} |\n")
    lines.append(f"| v3 negatives | {agg_tc(table_v3_neg):5.2f} | {agg_cpg(table_v3_neg):5.2f} |\n")
    if "cancer_matched" in neg_tables:
        lines.append(f"| v4_cancer negatives | {agg_tc(neg_tables['cancer_matched']):5.2f} | {agg_cpg(neg_tables['cancer_matched']):5.2f} |\n")
        lines.append(f"| Cancer target | {agg_tc(cancer_frac):5.2f} | {agg_cpg(cancer_frac):5.2f} |\n")
    if "cds_unbiased" in neg_tables:
        lines.append(f"| v4_cds negatives | {agg_tc(neg_tables['cds_unbiased']):5.2f} | {agg_cpg(neg_tables['cds_unbiased']):5.2f} |\n")
        lines.append(f"| CDS-C target | {agg_tc(cds_frac):5.2f} | {agg_cpg(cds_frac):5.2f} |\n")

    # ----------------------------------------------------------------
    # 3. Counts and file paths
    # ----------------------------------------------------------------
    lines.append("\n## Counts\n")
    lines.append(f"- v3 positives total: {n_pos_v3}\n")
    lines.append(f"- v3 'Neither' positives (APOBEC1) excluded: {excluded}\n")
    lines.append(f"- v4 positives (post-exclusion, shared by both versions): **{n_pos_v4}**\n")
    for ver, data in versions.items():
        df = data["df"]
        lines.append(f"- v4_{ver}: {len(df)} sites = {int((df['is_edited']==1).sum())} pos + {int((df['is_edited']==0).sum())} neg\n")

    # ----------------------------------------------------------------
    # 4. Jaccard overlap
    # ----------------------------------------------------------------
    if "cancer_matched" in versions and "cds_unbiased" in versions:
        a = versions["cancer_matched"]["df"]
        b = versions["cds_unbiased"]["df"]
        a_neg = set(a[a["is_edited"] == 0]["site_id"].astype(str))
        b_neg = set(b[b["is_edited"] == 0]["site_id"].astype(str))
        inter = len(a_neg & b_neg)
        union = len(a_neg | b_neg)
        jac = inter / union if union else 0.0
        lines.append("\n## Jaccard overlap of v4 negative sets\n")
        lines.append(f"- |cancer ∩ cds| = {inter}\n")
        lines.append(f"- |cancer ∪ cds| = {union}\n")
        lines.append(f"- Jaccard = {jac:.4f}\n")

    # ----------------------------------------------------------------
    # 5. Loop / structure / RNA-FM coverage check
    # ----------------------------------------------------------------
    lines.append("\n## Coverage checks\n")
    rnafm_status = {}
    for ver, data in versions.items():
        df = data["df"]; seqs = data["seqs"]
        all_ids = set(df["site_id"].astype(str))
        with_seq = sum(1 for sid in all_ids if sid in seqs and len(seqs[sid]) == 201)

        lp_ok = "n/a"
        if data["loop"].exists():
            lp = pd.read_csv(data["loop"])
            lp_ids = set(lp["site_id"].astype(str))
            lp_ok = f"{len(all_ids & lp_ids)}/{len(all_ids)}"

        st_ok = "n/a"
        if data["struct"].exists():
            st = np.load(data["struct"], allow_pickle=True)
            st_ids = set(str(s) for s in st["site_ids"])
            st_ok = f"{len(all_ids & st_ids)}/{len(all_ids)}"

        rnafm_path = EMB_DIR / f"rnafm_v4_{ver}.npz"
        rnafm_status[ver] = rnafm_path.exists()

        lines.append(f"- **v4_{ver}**: sequences {with_seq}/{len(all_ids)}, loop {lp_ok}, structure {st_ok}, RNA-FM={'yes' if rnafm_status[ver] else 'DEFERRED'}\n")

    # ----------------------------------------------------------------
    # 6. File paths + READY block
    # ----------------------------------------------------------------
    lines.append("\n## File paths\n")
    for ver, data in versions.items():
        lines.append(f"\n**v4_{ver}**\n")
        lines.append(f"- splits CSV: `{data['splits']}`\n")
        lines.append(f"- sequences JSON: `{data['seqs_path']}`\n")
        lines.append(f"- loop position CSV: `{data['loop']}`{'' if data['loop'].exists() else '  *(MISSING)*'}\n")
        lines.append(f"- structure cache NPZ: `{data['struct']}`{'' if data['struct'].exists() else '  *(MISSING)*'}\n")
        rnafm_path = EMB_DIR / f"rnafm_v4_{ver}.npz"
        lines.append(f"- RNA-FM embeddings: `{rnafm_path}`{'' if rnafm_path.exists() else '  *(DEFERRED — see below)*'}\n")

    lines.append("\n## Auxiliary files\n")
    lines.append(f"- Cancer trinuc CSV: `{CANCER_TRINUC_CSV}`\n")
    lines.append(f"- CDS-C trinuc CSV: `{CDS_TRINUC_CSV}`\n")

    # RNA-FM deferral note
    if not all(rnafm_status.values()):
        lines.append("\n## RNA-FM embeddings — deferred\n")
        lines.append("Run after this script using:\n\n")
        lines.append("```bash\nconda run -n quris python scripts/multi_enzyme/compute_rnafm_embeddings_v4.py\n```\n")

    OUT_MD.write_text("".join(lines))
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
