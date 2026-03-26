"""Analyze overlap between AID/APOBEC DNA mutation hotspots (COSMIC signatures)
and APOBEC RNA editing predictions.

COSMIC single-base substitution signatures relevant to C>T (=C>U on RNA):
  SBS2:  TCA>TTA  — APOBEC DNA mutagenesis
  SBS13: TCT>TTT  — APOBEC DNA mutagenesis
  SBS84: WRC>WYC  (W=A/T, R=A/G) — AID somatic hypermutation
  SBS85: WRC>WYC  — AID somatic hypermutation

For RNA editing, the relevant motifs are (reading the RNA strand):
  APOBEC3A: TC (upstream C + edit C) — strong preference
  APOBEC3G: CC (upstream C + edit C) — strong preference
  AID: WRC on DNA = overlaps with AC/GC dinucleotide upstream on RNA

This script:
1. Extracts trinucleotide context for each ClinVar C>U variant from cached features
2. Classifies contexts as AID-like, APOBEC-like (SBS2/13), A3G-like, or other
3. Analyzes how RNA editing scores distribute across mutation signature contexts
4. Identifies genes with both high RNA editing scores and known cancer mutation signatures
"""

import os
import sys
import json
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / "experiments" / "multi_enzyme" / "outputs" / "cosmic_overlap"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Load ClinVar scored data ──────────────────────────────────────────────

print("Loading ClinVar scored data...")
clinvar_path = PROJECT_ROOT / "experiments" / "apobec3a" / "outputs" / "clinvar_prediction" / "clinvar_all_scores.csv"
df = pd.read_csv(clinvar_path)
print(f"  Loaded {len(df):,} ClinVar C>T/C>U variants")

# ── 2. Load feature cache to decode trinucleotide context ─────────────────────

print("Loading feature cache...")
cache = np.load(PROJECT_ROOT / "data" / "processed" / "clinvar_features_cache.npz", allow_pickle=True)
cache_ids = cache["site_ids"]
hand_46 = cache["hand_46"]

# The 24-dim motif features are:
# [0:4]  5' dinucleotide: UC, CC, AC, GC  (what's upstream of the C)
# [4:8]  3' dinucleotide: CA, CG, CU, CC  (what's downstream of the C)
# [8:12] m2 position: A, C, G, U
# [12:16] m1 position: A, C, G, U  (same as 5' but explicit)
# [16:20] p1 position: A, C, G, U  (same as 3' but explicit)
# [20:24] p2 position: A, C, G, U

# Decode 5' (upstream) base from one-hot [0:4] → UC=U, CC=C, AC=A, GC=G
# Decode 3' (downstream) base from one-hot [4:8] → CA=A, CG=G, CU=U, CC=C

up_bases_map = {0: "U", 1: "C", 2: "A", 3: "G"}  # index in [0:4]
down_bases_map = {0: "A", 1: "G", 2: "U", 3: "C"}  # index in [4:8]

# Also decode m2 (position -2) from [8:12]
m2_bases_map = {0: "A", 1: "C", 2: "G", 3: "U"}

print("Decoding trinucleotide contexts...")
upstream = hand_46[:, 0:4]
downstream = hand_46[:, 4:8]
m2_onehot = hand_46[:, 8:12]

up_idx = np.argmax(upstream, axis=1)
down_idx = np.argmax(downstream, axis=1)
m2_idx = np.argmax(m2_onehot, axis=1)

# Check for ambiguous (no base set) — sum == 0
up_valid = upstream.sum(axis=1) > 0.5
down_valid = downstream.sum(axis=1) > 0.5
m2_valid = m2_onehot.sum(axis=1) > 0.5

up_base = np.array([up_bases_map.get(i, "N") for i in up_idx])
down_base = np.array([down_bases_map.get(i, "N") for i in down_idx])
m2_base = np.array([m2_bases_map.get(i, "N") for i in m2_idx])

up_base[~up_valid] = "N"
down_base[~down_valid] = "N"
m2_base[~m2_valid] = "N"

# Build trinuc context: upstream + C + downstream (RNA strand, T→U)
trinuc = np.array([f"{u}C{d}" for u, d, in zip(up_base, down_base)])
# Build pentanuc: m2 + upstream + C + downstream + p2
# (we only need trinuc and upstream dinuc for signature classification)

# Map cache IDs to DataFrame
cache_id_to_idx = {sid: i for i, sid in enumerate(cache_ids)}

# Add context columns to DataFrame
df_trinuc = []
df_upstream = []
df_downstream = []
df_m2 = []

for sid in df["site_id"]:
    idx = cache_id_to_idx.get(sid)
    if idx is not None:
        df_trinuc.append(trinuc[idx])
        df_upstream.append(up_base[idx])
        df_downstream.append(down_base[idx])
        df_m2.append(m2_base[idx])
    else:
        df_trinuc.append("N")
        df_upstream.append("N")
        df_downstream.append("N")
        df_m2.append("N")

df["trinuc"] = df_trinuc
df["upstream"] = df_upstream
df["downstream"] = df_downstream
df["m2"] = df_m2

print(f"  Context decoded for {(df['trinuc'] != 'N').sum():,} / {len(df):,} variants")

# ── 3. Classify mutation signature context ────────────────────────────────────

# DNA strand mapping: RNA C>U = DNA C>T on sense strand
# SBS2: TCA>TTA → RNA: UCA context (upstream=U, center=C, downstream=A)
# SBS13: TCT>TTT → RNA: UCU context (upstream=U, center=C, downstream=U)
# AID (SBS84/85): WRC>WYC where W=A/T(U), R=A/G → RNA: upstream=A or G, m2=A or U
# A3G-like: CC context → upstream=C

def classify_signature(row):
    """Classify the trinucleotide context into mutation signature categories."""
    up = row["upstream"]
    down = row["downstream"]
    m2 = row["m2"]

    labels = []

    # APOBEC SBS2/SBS13: TC context (upstream U on RNA = T on DNA)
    # DNA: TCA→TTA (SBS2), TCT→TTT (SBS13)
    if up == "U":  # TC on DNA
        if down == "A":
            labels.append("SBS2_TCA")
        elif down == "U":
            labels.append("SBS13_TCU")
        else:
            labels.append("APOBEC_TC_other")

    # A3G-like: CC context
    if up == "C":
        labels.append("A3G_CC")

    # AID (SBS84/85): WRC on DNA where W=A/T, R=A/G
    # On RNA: m2=A or U (=W), upstream=A or G (=R), center=C
    if up in ("A", "G") and m2 in ("A", "U"):
        labels.append("AID_WRC")

    # A3A RNA editing: TC context (same as APOBEC DNA)
    if up == "U":
        labels.append("A3A_TC")

    if not labels:
        labels.append("Other")

    return "|".join(labels)

print("Classifying mutation signatures...")
df["signature_class"] = df.apply(classify_signature, axis=1)

# Create simplified primary classification
def primary_class(sig):
    if "SBS2_TCA" in sig or "SBS13_TCU" in sig:
        return "APOBEC_DNA_hotspot"
    elif "APOBEC_TC_other" in sig or "A3A_TC" in sig:
        return "APOBEC_TC_other"
    elif "AID_WRC" in sig:
        return "AID_hotspot"
    elif "A3G_CC" in sig:
        return "A3G_CC"
    else:
        return "Other"

df["primary_class"] = df["signature_class"].apply(primary_class)

# ── 4. Summary statistics ────────────────────────────────────────────────────

print("\n" + "="*80)
print("SECTION 1: Trinucleotide Context Distribution")
print("="*80)

trinuc_counts = df["trinuc"].value_counts()
print(f"\nAll {len(df):,} ClinVar C>U variants by trinucleotide context:")
for ctx, n in trinuc_counts.head(16).items():
    pct = 100 * n / len(df)
    print(f"  {ctx}: {n:>8,} ({pct:5.1f}%)")

print(f"\nPrimary signature classification:")
for cls, n in df["primary_class"].value_counts().items():
    pct = 100 * n / len(df)
    print(f"  {cls:25s}: {n:>8,} ({pct:5.1f}%)")

# ── 5. Editing scores by signature class ──────────────────────────────────────

print("\n" + "="*80)
print("SECTION 2: A3A RNA Editing Scores by DNA Mutation Signature Context")
print("="*80)

gb_col = "p_edited_gb"
rnasee_col = "p_edited_rnasee"

print(f"\nMean editing probability by signature class:")
print(f"{'Class':25s} {'N':>10s} {'GB_mean':>8s} {'GB_med':>8s} {'GB>=0.5':>8s} {'RNAsee_mean':>10s}")
print("-" * 80)

results_by_class = {}
for cls in ["APOBEC_DNA_hotspot", "APOBEC_TC_other", "AID_hotspot", "A3G_CC", "Other"]:
    mask = df["primary_class"] == cls
    sub = df[mask]
    n = len(sub)
    if n == 0:
        continue
    gb_mean = sub[gb_col].mean()
    gb_med = sub[gb_col].median()
    high = (sub[gb_col] >= 0.5).sum()
    rnasee_mean = sub[rnasee_col].mean()
    print(f"  {cls:25s} {n:>8,} {gb_mean:>8.4f} {gb_med:>8.4f} {high:>8,} {rnasee_mean:>10.4f}")
    results_by_class[cls] = {
        "n": int(n), "gb_mean": float(gb_mean), "gb_median": float(gb_med),
        "n_high_gb": int(high), "rnasee_mean": float(rnasee_mean)
    }

# ── 6. Among high-score predictions, what fraction are in each context? ───────

print("\n" + "="*80)
print("SECTION 3: Among High-Score Editing Predictions (GB >= 0.5)")
print("="*80)

high_mask = df[gb_col] >= 0.5
high_df = df[high_mask]
print(f"\n{len(high_df):,} variants with GB editing probability >= 0.5")

print(f"\nSignature context breakdown among high-score predictions:")
high_class_counts = high_df["primary_class"].value_counts()
for cls, n in high_class_counts.items():
    pct = 100 * n / len(high_df)
    pct_of_class = 100 * n / (df["primary_class"] == cls).sum()
    print(f"  {cls:25s}: {n:>7,} ({pct:5.1f}% of high-score, {pct_of_class:5.1f}% of class predicted high)")

print(f"\nTrinucleotide context among high-score predictions:")
high_trinuc = high_df["trinuc"].value_counts()
for ctx, n in high_trinuc.head(16).items():
    pct = 100 * n / len(high_df)
    print(f"  {ctx}: {n:>7,} ({pct:5.1f}%)")

# ── 7. Overlap: AID vs APOBEC predictions ────────────────────────────────────

print("\n" + "="*80)
print("SECTION 4: AID vs APOBEC Overlap Analysis")
print("="*80)

# Check if any sites are classified as both AID and APOBEC
both_mask = df["signature_class"].str.contains("AID") & df["signature_class"].str.contains("A3A_TC|SBS2|SBS13|APOBEC_TC")
n_both = both_mask.sum()
print(f"\nSites with BOTH AID (WRC) and APOBEC (TC) contexts: {n_both:,}")
print("  (This should be 0 — AID requires upstream A/G, APOBEC requires upstream T/U)")
print(f"  → AID and APOBEC target DIFFERENT motif contexts by definition")

# But check: among AID-context sites, how many are predicted as RNA-edited?
aid_mask = df["primary_class"] == "AID_hotspot"
aid_high = (df[aid_mask][gb_col] >= 0.5).sum()
apobec_mask = df["primary_class"].isin(["APOBEC_DNA_hotspot", "APOBEC_TC_other"])
apobec_high = (df[apobec_mask][gb_col] >= 0.5).sum()

print(f"\nAID-context sites predicted as RNA-edited (GB>=0.5): {aid_high:,} / {aid_mask.sum():,} ({100*aid_high/aid_mask.sum():.2f}%)")
print(f"APOBEC-context sites predicted as RNA-edited (GB>=0.5): {apobec_high:,} / {apobec_mask.sum():,} ({100*apobec_high/apobec_mask.sum():.2f}%)")

# Enrichment ratio
aid_rate = aid_high / aid_mask.sum() if aid_mask.sum() > 0 else 0
apobec_rate = apobec_high / apobec_mask.sum() if apobec_mask.sum() > 0 else 0
other_mask = df["primary_class"] == "Other"
other_high = (df[other_mask][gb_col] >= 0.5).sum()
other_rate = other_high / other_mask.sum() if other_mask.sum() > 0 else 0
baseline_rate = (df[gb_col] >= 0.5).sum() / len(df)

print(f"\nEditing prediction rate (GB>=0.5) by context:")
print(f"  APOBEC (TC) context: {100*apobec_rate:.2f}%")
print(f"  AID (WRC) context:   {100*aid_rate:.2f}%")
print(f"  A3G (CC) context:    {100*(df[df['primary_class']=='A3G_CC'][gb_col]>=0.5).sum()/df[df['primary_class']=='A3G_CC'].shape[0]:.2f}%")
print(f"  Other context:       {100*other_rate:.2f}%")
print(f"  Overall baseline:    {100*baseline_rate:.2f}%")

# ── 8. Gene-level analysis ───────────────────────────────────────────────────

print("\n" + "="*80)
print("SECTION 5: Gene-Level Overlap — Genes with Both High Editing Scores")
print("         in APOBEC-context AND AID-context")
print("="*80)

# Genes with high-score APOBEC-context variants
apobec_high_genes = set(df[apobec_mask & (df[gb_col] >= 0.5)]["gene"].dropna().unique())
aid_high_genes = set(df[aid_mask & (df[gb_col] >= 0.5)]["gene"].dropna().unique())
a3g_high_genes = set(df[(df["primary_class"] == "A3G_CC") & (df[gb_col] >= 0.5)]["gene"].dropna().unique())

overlap_apobec_aid = apobec_high_genes & aid_high_genes
overlap_apobec_a3g = apobec_high_genes & a3g_high_genes
overlap_all = apobec_high_genes & aid_high_genes & a3g_high_genes

print(f"\nGenes with high editing scores (GB>=0.5):")
print(f"  In APOBEC (TC) context: {len(apobec_high_genes):,} genes")
print(f"  In AID (WRC) context:   {len(aid_high_genes):,} genes")
print(f"  In A3G (CC) context:    {len(a3g_high_genes):,} genes")
print(f"  Overlap APOBEC∩AID:     {len(overlap_apobec_aid):,} genes")
print(f"  Overlap APOBEC∩A3G:     {len(overlap_apobec_a3g):,} genes")
print(f"  Overlap all three:      {len(overlap_all):,} genes")

# ── 9. Clinical significance by signature + editing score ─────────────────────

print("\n" + "="*80)
print("SECTION 6: Clinical Significance × Signature Context × Editing Score")
print("="*80)

# Focus on pathogenic/likely pathogenic
path_mask = df["significance_simple"].isin(["Pathogenic", "Likely_pathogenic"])
path_df = df[path_mask]
print(f"\nPathogenic/Likely pathogenic variants: {len(path_df):,}")

print(f"\nPathogenic variants by signature class:")
for cls in ["APOBEC_DNA_hotspot", "APOBEC_TC_other", "AID_hotspot", "A3G_CC", "Other"]:
    mask_cls = path_df["primary_class"] == cls
    n = mask_cls.sum()
    if n == 0:
        continue
    n_high = (path_df[mask_cls][gb_col] >= 0.5).sum()
    pct_high = 100 * n_high / n if n > 0 else 0
    print(f"  {cls:25s}: {n:>6,} total, {n_high:>5,} high-score ({pct_high:.1f}%)")

# Enrichment: pathogenic fraction among high-score vs low-score, per class
print(f"\nPathogenic enrichment (high-score vs low-score editing predictions):")
print(f"{'Class':25s} {'Path_high%':>10s} {'Path_low%':>10s} {'OR':>8s}")
print("-" * 60)

enrichment_results = {}
for cls in ["APOBEC_DNA_hotspot", "APOBEC_TC_other", "AID_hotspot", "A3G_CC", "Other"]:
    cls_mask = df["primary_class"] == cls
    cls_df = df[cls_mask]
    if len(cls_df) < 100:
        continue

    high = cls_df[cls_df[gb_col] >= 0.5]
    low = cls_df[cls_df[gb_col] < 0.5]

    path_high = high["significance_simple"].isin(["Pathogenic", "Likely_pathogenic"]).sum()
    path_low = low["significance_simple"].isin(["Pathogenic", "Likely_pathogenic"]).sum()

    rate_high = path_high / len(high) if len(high) > 0 else 0
    rate_low = path_low / len(low) if len(low) > 0 else 0

    odds_ratio = (rate_high / (1 - rate_high)) / (rate_low / (1 - rate_low)) if rate_low > 0 and rate_high > 0 and rate_high < 1 and rate_low < 1 else float("nan")

    print(f"  {cls:25s} {100*rate_high:>9.2f}% {100*rate_low:>9.2f}% {odds_ratio:>8.2f}")
    enrichment_results[cls] = {
        "n_high": int(len(high)), "n_low": int(len(low)),
        "path_rate_high": float(rate_high), "path_rate_low": float(rate_low),
        "odds_ratio": float(odds_ratio)
    }

# ── 10. APOBEC3A vs APOBEC3B vs APOBEC3G RNA vs DNA preferences ──────────────

print("\n" + "="*80)
print("SECTION 7: RNA Editing vs DNA Mutation — Motif Comparison")
print("="*80)

print("""
Known DNA mutation signatures (COSMIC):
  APOBEC3A/3B DNA: TC→TT (SBS2: TCA→TTA, SBS13: TCT→TTT)
  AID DNA:         WRC→WYC (SBS84/85)
  A3G DNA:         CC→CT (some cancers)

Known RNA editing preferences (this project):
  APOBEC3A RNA:    TC context (strong), UC dinucleotide
  APOBEC3B RNA:    mixed context, no strong bias
  APOBEC3G RNA:    CC context (strong), extreme 3' tetraloop preference

Key observations:
1. A3A RNA editing and A3A DNA mutagenesis share the SAME TC motif.
   → Sites predicted as A3A RNA editing targets are in the SAME trinucleotide
     context as APOBEC DNA mutation hotspots (SBS2/SBS13).
   → This is NOT coincidence: both activities derive from the same catalytic
     domain that recognizes TC dinucleotide.

2. A3G RNA editing and A3G-like DNA mutations share the CC motif.
   → A3G edits CC contexts on RNA AND mutates CC contexts on DNA.

3. AID targets WRC on DNA, which does NOT overlap with TC or CC.
   → AID-driven mutations and APOBEC RNA editing are at DIFFERENT sites.

4. A3B is unique: strong TC preference on DNA but mixed context on RNA.
   → A3B's RNA editing substrate preference differs from its DNA substrate.
   → This may reflect different binding modes for RNA vs ssDNA.
""")

# ── 11. Save detailed results ────────────────────────────────────────────────

print("\n" + "="*80)
print("SECTION 8: Saving Results")
print("="*80)

# Save high-score predictions with signature annotation
high_annotated = high_df[["site_id", "chr", "start", "gene", "clinical_significance",
                           "significance_simple", gb_col, rnasee_col,
                           "trinuc", "primary_class", "signature_class"]].copy()
high_annotated.to_csv(OUT_DIR / "high_score_predictions_with_signature.csv", index=False)
print(f"  Saved {len(high_annotated):,} high-score predictions to high_score_predictions_with_signature.csv")

# Save per-gene summary
gene_summary = []
for gene in df["gene"].dropna().unique():
    g = df[df["gene"] == gene]
    row = {
        "gene": gene,
        "n_variants": len(g),
        "n_high_gb": (g[gb_col] >= 0.5).sum(),
        "n_pathogenic": g["significance_simple"].isin(["Pathogenic", "Likely_pathogenic"]).sum(),
    }
    for cls in ["APOBEC_DNA_hotspot", "AID_hotspot", "A3G_CC"]:
        cls_mask = g["primary_class"] == cls
        row[f"n_{cls}"] = int(cls_mask.sum())
        row[f"n_{cls}_high"] = int((g[cls_mask][gb_col] >= 0.5).sum())
    gene_summary.append(row)

gene_df = pd.DataFrame(gene_summary)
gene_df = gene_df.sort_values("n_high_gb", ascending=False)
gene_df.to_csv(OUT_DIR / "gene_level_signature_summary.csv", index=False)
print(f"  Saved gene-level summary ({len(gene_df):,} genes) to gene_level_signature_summary.csv")

# Save summary statistics as JSON
summary = {
    "total_variants": int(len(df)),
    "total_high_score": int(len(high_df)),
    "signature_distribution": {k: int(v) for k, v in df["primary_class"].value_counts().items()},
    "high_score_by_signature": {k: int(v) for k, v in high_df["primary_class"].value_counts().items()},
    "editing_rate_by_class": results_by_class,
    "pathogenic_enrichment_by_class": enrichment_results,
    "gene_overlap": {
        "apobec_tc_genes": len(apobec_high_genes),
        "aid_wrc_genes": len(aid_high_genes),
        "a3g_cc_genes": len(a3g_high_genes),
        "apobec_aid_overlap": len(overlap_apobec_aid),
        "apobec_a3g_overlap": len(overlap_apobec_a3g),
        "all_three_overlap": len(overlap_all),
    }
}
with open(OUT_DIR / "analysis_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"  Saved analysis_summary.json")

# ── 12. Top genes with dual RNA editing + DNA mutation signature ──────────────

print("\n" + "="*80)
print("SECTION 9: Top Genes with Both APOBEC RNA Editing AND DNA Mutation Context")
print("="*80)

# Genes with the most high-score predictions in APOBEC DNA hotspot contexts
top_genes = gene_df[gene_df["n_APOBEC_DNA_hotspot_high"] > 0].nlargest(20, "n_APOBEC_DNA_hotspot_high")
print(f"\nTop 20 genes by number of high-score APOBEC-context variants:")
print(f"{'Gene':15s} {'Total':>6s} {'High':>6s} {'APOBEC_hi':>10s} {'AID_hi':>7s} {'Path':>5s}")
print("-" * 55)
for _, row in top_genes.iterrows():
    print(f"  {row['gene']:15s} {row['n_variants']:>5} {row['n_high_gb']:>5} "
          f"{row['n_APOBEC_DNA_hotspot_high']:>9} {row['n_AID_hotspot_high']:>6} {row['n_pathogenic']:>5}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
