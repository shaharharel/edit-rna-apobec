#!/usr/bin/env python3
"""Bias diagnostic for retrained APOBEC1 v4 heads.

For each variant (cancer, cds), sample 100K random positions from the
CDS-C panel (the merged retrained parquet), look up trinucleotide context
(strand-aware) from hg19, and report mean predicted P per trinuc bin.

Verifies the v4 heads are NOT anti-TCW (i.e. they should NOT score TCW
trinucs lower than non-TCW). For an APOBEC1 head we expect higher
scores at AC* / TC* (mooring-rich) trinucs and lower at GC* / CCG.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyfaidx

ROOT = Path(__file__).resolve().parents[2]
HG19 = ROOT / "data" / "raw" / "genomes" / "hg19.fa"
V4_DIR = ROOT / "experiments" / "multi_enzyme" / "outputs" / "pcawg_tcw_panel" / "v4_outputs"

CASES = [
    ("cancer", V4_DIR / "panel_scores_v4_cancer_apobec1retrained.parquet",
     V4_DIR / "bias_diagnostic_apobec1_v4_cancer.json"),
    ("cds", V4_DIR / "panel_scores_v4_cds_apobec1retrained.parquet",
     V4_DIR / "bias_diagnostic_apobec1_v4_cds.json"),
]

SEED = 20260427
N_SAMPLE = 100_000

COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}


def revcomp(seq: str) -> str:
    return "".join(COMPLEMENT[b] for b in reversed(seq))


def main():
    fa = pyfaidx.Fasta(str(HG19))
    rng = np.random.RandomState(SEED)

    for variant, parquet_path, out_path in CASES:
        print(f"\n=== variant: {variant} ===")
        df = pd.read_parquet(parquet_path, columns=["chrom", "pos", "strand", "valid",
                                                     f"score_apobec1_v4_{variant}"])
        valid_idx = np.where(df["valid"].values)[0]
        sample_idx = rng.choice(valid_idx, size=min(N_SAMPLE, len(valid_idx)), replace=False)
        sub = df.iloc[sample_idx].reset_index(drop=True)
        print(f"  sampled {len(sub)} valid positions")

        chroms = sub["chrom"].values
        pos = sub["pos"].values  # hg19, 1-based per panel convention
        strand = sub["strand"].values
        scores = sub[f"score_apobec1_v4_{variant}"].values

        trinucs = []
        miss = 0
        for c, p, s in zip(chroms, pos, strand):
            try:
                # Panel pos is hg19 0-based (verified by spot-check). Trinuc is [p-1, p, p+1].
                tri = str(fa[c][p - 1:p + 2]).upper()
            except Exception:
                tri = "NNN"
                miss += 1
            if s == "-":
                tri = revcomp(tri)
            trinucs.append(tri)
        trinucs = np.array(trinucs)
        print(f"  trinuc lookup miss={miss}")

        # Restrict to ACGT trinucs centered on C (panel is C-centric)
        center_C = np.array([(t[1] == "C") for t in trinucs])
        tri_ok = np.array([all(b in "ACGT" for b in t) for t in trinucs])
        keep = center_C & tri_ok
        print(f"  trinucs centered on C: {keep.sum()}/{len(trinucs)}")

        records = []
        for tri in sorted(set(trinucs[keep])):
            mask = trinucs == tri
            records.append({
                "trinuc": tri,
                "n": int(mask.sum()),
                "mean_p": float(np.nanmean(scores[mask])),
                "median_p": float(np.nanmedian(scores[mask])),
            })
        bias_df = pd.DataFrame(records).sort_values("mean_p", ascending=False)
        print(bias_df.to_string(index=False))

        # Anti-TCW polarity check:
        #  TCW = TCA or TCT  (canonical APOBEC3A motif)
        #  non-TCW canonical-C trinucs: anything else with center C
        tcw = bias_df[bias_df["trinuc"].isin(["TCA", "TCT"])]["mean_p"].mean()
        nontcw = bias_df[~bias_df["trinuc"].isin(["TCA", "TCT"])]["mean_p"].mean()
        # Apobec1 mooring-rich vs non-rich (expect ACA/ACT high, ACG/CCG/GCG low):
        rich = bias_df[bias_df["trinuc"].isin(["ACA", "ACT", "TCA", "TCT"])]["mean_p"].mean()
        depleted = bias_df[bias_df["trinuc"].isin(["ACG", "CCG", "GCG", "TCG"])]["mean_p"].mean()
        anti_tcw = bool(tcw < nontcw)

        verdict = {
            "variant": variant,
            "n_sampled": int(len(sub)),
            "n_with_C_center": int(keep.sum()),
            "trinuc_breakdown": records,
            "TCW_mean": float(tcw),
            "nonTCW_mean": float(nontcw),
            "anti_TCW_polarity_present": anti_tcw,
            "mooring_rich_mean (ACA/ACT/TCA/TCT)": float(rich),
            "CpG_depleted_mean (ACG/CCG/GCG/TCG)": float(depleted),
            "rich_vs_CpG_ratio": float(rich / depleted) if depleted > 0 else None,
        }
        with open(out_path, "w") as f:
            json.dump(verdict, f, indent=2)
        print(f"  TCW mean={tcw:.4f}  nonTCW mean={nontcw:.4f}  anti_TCW_polarity={anti_tcw}")
        print(f"  mooring-rich={rich:.4f}  CpG-depleted={depleted:.4f}  ratio={rich/depleted:.3f}" if depleted > 0 else "")
        print(f"  -> {out_path.name}")


if __name__ == "__main__":
    main()
