"""
PCPG / Cancer Deep-Dive Analysis for EditRNA-A3A
=================================================
159 of 636 Levanon editing sites have survival associations in PCPG
(Pheochromocytoma/Paraganglioma - adrenal tumors). Adrenal glands are part of
the Kidney/Other endocrine module with an independent editing program.
This script characterizes WHY PCPG dominates.

Analyses:
  1. PCPG site characterization & gene enrichment
  2. Editing rate comparison across 54 tissues
  3. Tissue specificity / breadth of PCPG sites
  4. Sequence motif analysis (extended context)
  5. Embedding space analysis (UMAP/tSNE)
  6. Model prediction analysis (EditRNA-A3A scores)
  7. Cancer type breakdown across all cancer types
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import json
import warnings

import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import seaborn as sns
from scipy import stats
from collections import Counter, OrderedDict
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = "/Users/shaharharel/Documents/github/edit-rna-apobec"
OUT = os.path.join(BASE, "experiments/apobec/outputs/pcpg_analysis")
os.makedirs(OUT, exist_ok=True)

T5_PATH = os.path.join(BASE, "data/processed/advisor/t5_tcga_survival.csv")
T1_PATH = os.path.join(BASE, "data/processed/advisor/t1_gtex_editing_&_conservation.csv")
SPLITS_PATH = os.path.join(BASE, "data/processed/splits_expanded.csv")
SEQ_PATH = os.path.join(BASE, "data/processed/site_sequences.json")
EMB_PATH = os.path.join(BASE, "data/processed/embeddings/rnafm_pooled.pt")
EMB_ED_PATH = os.path.join(BASE, "data/processed/embeddings/rnafm_pooled_edited.pt")
EMB_IDS_PATH = os.path.join(BASE, "data/processed/embeddings/rnafm_site_ids.json")
PRED_PATH = os.path.join(BASE, "experiments/apobec/outputs/iteration3/test_predictions.csv")
RATE_PATH = os.path.join(BASE, "experiments/apobec/outputs/rate_prediction/rate_predictions.csv")

# ---------------------------------------------------------------------------
# Publication-quality style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

# Colour palette
PCPG_COLOR = "#D62728"  # red
OTHER_CANCER_COLOR = "#FF7F0E"  # orange
NO_CANCER_COLOR = "#1F77B4"  # blue
PALETTE = {"PCPG": PCPG_COLOR, "Other cancer": OTHER_CANCER_COLOR, "No association": NO_CANCER_COLOR}

# ---------------------------------------------------------------------------
# Helper: parse tissue rate from "reads;total;rate" format
# ---------------------------------------------------------------------------
def parse_tissue_rate(val):
    if pd.isna(val) or val == "" or val == "nan":
        return np.nan
    parts = str(val).split(";")
    if len(parts) == 3:
        try:
            return float(parts[2])
        except ValueError:
            return np.nan
    return np.nan


def parse_tissue_coverage(val):
    """Return total reads (denominator) from 'edited;total;rate'."""
    if pd.isna(val) or val == "" or val == "nan":
        return np.nan
    parts = str(val).split(";")
    if len(parts) == 3:
        try:
            return float(parts[1])
        except ValueError:
            return np.nan
    return np.nan


# ============================================================================
# Load data
# ============================================================================
print("=" * 80)
print("LOADING DATA")
print("=" * 80)

t5 = pd.read_csv(T5_PATH).dropna(subset=["Chr", "Start"])
t5["Start"] = t5["Start"].astype(int)
t5["End"] = t5["End"].astype(int)
print(f"T5 cancer survival: {len(t5)} rows, {t5[['Chr','Start']].drop_duplicates().shape[0]} unique sites")

t1 = pd.read_csv(T1_PATH)
print(f"T1 tissue data: {len(t1)} sites x {t1.shape[1]} columns")

splits = pd.read_csv(SPLITS_PATH)
advisor = splits[splits["dataset_source"] == "advisor_c2t"].copy()
print(f"Splits expanded: {len(splits)} total, {len(advisor)} advisor sites")

with open(SEQ_PATH) as f:
    sequences = json.load(f)
print(f"Sequences: {len(sequences)} entries")

# ---------------------------------------------------------------------------
# Build site-level cancer annotation from T5
# ---------------------------------------------------------------------------
# T5 has 252 unique sites with survival associations; remaining 636-252=384
# do NOT appear in t5 (no cancer associations)
# The "Cancers_with..." column is semicolon-separated

cancer_col = "Cancers_with_Editing_Significantly_Associated_with_Survival"

# Build per-site cancer set
site_cancers = {}
for _, row in t5.iterrows():
    key = (row["Chr"], row["Start"])
    cancers_str = str(row[cancer_col])
    if cancers_str == "nan":
        continue
    cancer_list = [c.strip() for c in cancers_str.split(";") if c.strip()]
    if key not in site_cancers:
        site_cancers[key] = set()
    site_cancers[key].update(cancer_list)

print(f"\nSites with any cancer association: {len(site_cancers)}")

# Determine PCPG status for each site in t1 (all 636 Levanon sites)
t1["site_key"] = list(zip(t1["Chr"], t1["Start"]))
t1["cancers"] = t1["site_key"].map(lambda k: site_cancers.get(k, set()))
t1["has_pcpg"] = t1["cancers"].apply(lambda s: "PCPG" in s)
t1["has_any_cancer"] = t1["cancers"].apply(lambda s: len(s) > 0)
t1["cancer_group"] = t1.apply(
    lambda r: "PCPG" if r["has_pcpg"] else ("Other cancer" if r["has_any_cancer"] else "No association"),
    axis=1,
)

# Also tag t5 unique sites
t5["site_key"] = list(zip(t5["Chr"], t5["Start"]))
t5["cancers_set"] = t5["site_key"].map(lambda k: site_cancers.get(k, set()))
t5["has_pcpg"] = t5["cancers_set"].apply(lambda s: "PCPG" in s)

# Stats
n_pcpg = t1["has_pcpg"].sum()
n_other = ((~t1["has_pcpg"]) & t1["has_any_cancer"]).sum()
n_none = (~t1["has_any_cancer"]).sum()
print(f"PCPG-associated sites: {n_pcpg}")
print(f"Other-cancer-only sites: {n_other}")
print(f"No cancer association: {n_none}")

# Link to splits for site_id
advisor_lookup = {}
for _, row in advisor.iterrows():
    k = (row["chr"], row["start"])
    advisor_lookup[k] = row["site_id"]

t1["site_id"] = t1["site_key"].map(lambda k: advisor_lookup.get(k, None))

# Tissue columns
TISSUE_COLS = [
    c
    for c in t1.columns
    if c
    not in [
        "Chr", "Start", "End", "Genomic_Category", "Gene_(RefSeq)",
        "mRNA_location_(RefSeq)", "Exonic_Function", "Edited_In_#_Tissues",
        "Edited_Tissues_(Z_score_>=_2)", "Edited_Tissues_(Z_score_\u22652)",
        "Tissue_Classification", "Affecting_Over_Expressed_APOBEC",
        "Max_GTEx_Editing_Rate", "Mean_GTEx_Editing_Rate", "GTEx_Editing_Rate_SD",
        "Any_Non-Primate_Editing", "Any_Non-Primate_Editing_>=_1%",
        "Any_Non-Primate_Editing_\u22651%",
        "Any_Primate_Editing", "Any_Primate_Editing_>=_1%",
        "Any_Primate_Editing_\u22651%",
        "Any_Mammalian_Editing", "Any_Mammlian_Editing_>=_1%",
        "Any_Mammlian_Editing_\u22651%",
        "non-Boreoeutheria_(Primitve_mammals)",
        "Laurasiatheria_(non_rodent_or_primate_placental_mammals)",
        "Glires_(rodents_&_rabbits)",
        "non-Catarrhini_Primates_(new_world_monekys_and_lemurs)",
        "Cercopithecinae_(most_old_world_monkeys)",
        "Laurasiatherianon-Human_Homininae_(Apes)",
        "site_key", "cancers", "has_pcpg", "has_any_cancer",
        "cancer_group", "site_id",
    ]
]
# Filter to only real tissue names (54 columns)
TISSUE_COLS = [c for c in TISSUE_COLS if not c.startswith("Any_") and not c.startswith("non-")]
print(f"\nTissue columns identified: {len(TISSUE_COLS)}")

# Parse editing rates for all tissues
rate_matrix = pd.DataFrame(index=t1.index)
for tc in TISSUE_COLS:
    rate_matrix[tc] = t1[tc].apply(parse_tissue_rate)
rate_matrix.index = t1.index

# Define tissue groups relevant to PCPG
ADRENAL_KIDNEY = ["Adrenal_Gland", "Kidney_Cortex", "Kidney_Medulla"]
BRAIN_TISSUES = [c for c in TISSUE_COLS if c.startswith("Brain_")]
BLOOD_IMMUNE = ["Whole_Blood", "Spleen", "Cells_EBV-transformed_lymphocytes"]
ENDOCRINE = ["Adrenal_Gland", "Pituitary", "Thyroid", "Pancreas"]

# ============================================================================
# Results collector
# ============================================================================
results = {}

# ============================================================================
# ANALYSIS 1: PCPG Site Characterization & Gene Enrichment
# ============================================================================
print("\n" + "=" * 80)
print("ANALYSIS 1: PCPG SITE CHARACTERIZATION & GENE ENRICHMENT")
print("=" * 80)

pcpg_sites = t1[t1["has_pcpg"]].copy()
non_pcpg = t1[~t1["has_pcpg"]].copy()

# Gene list
pcpg_genes_raw = pcpg_sites["Gene_(RefSeq)"].dropna().tolist()
pcpg_genes = []
for g in pcpg_genes_raw:
    pcpg_genes.extend([x.strip() for x in str(g).split(";")])
pcpg_gene_counts = Counter(pcpg_genes)

all_genes_raw = t1["Gene_(RefSeq)"].dropna().tolist()
all_genes = []
for g in all_genes_raw:
    all_genes.extend([x.strip() for x in str(g).split(";")])
all_gene_counts = Counter(all_genes)

print(f"\nPCPG sites: {len(pcpg_sites)}")
print(f"Unique genes in PCPG sites: {len(pcpg_gene_counts)}")
print(f"Top 20 PCPG genes:")
for gene, cnt in pcpg_gene_counts.most_common(20):
    total = all_gene_counts.get(gene, 0)
    print(f"  {gene}: {cnt}/{total} sites")

# Genomic category breakdown
print("\nGenomic category (PCPG vs all):")
for cat in t1["Genomic_Category"].dropna().unique():
    n_pcpg_cat = (pcpg_sites["Genomic_Category"] == cat).sum()
    n_all_cat = (t1["Genomic_Category"] == cat).sum()
    print(f"  {cat}: PCPG={n_pcpg_cat} ({100*n_pcpg_cat/len(pcpg_sites):.1f}%), All={n_all_cat} ({100*n_all_cat/len(t1):.1f}%)")

# Exonic function breakdown
print("\nExonic function (PCPG vs all):")
for func in t1["Exonic_Function"].dropna().unique():
    n_pcpg_func = (pcpg_sites["Exonic_Function"] == func).sum()
    n_all_func = (t1["Exonic_Function"] == func).sum()
    pct_pcpg = 100 * n_pcpg_func / len(pcpg_sites) if len(pcpg_sites) > 0 else 0
    pct_all = 100 * n_all_func / len(t1) if len(t1) > 0 else 0
    print(f"  {func}: PCPG={n_pcpg_func} ({pct_pcpg:.1f}%), All={n_all_func} ({pct_all:.1f}%)")

# APOBEC enzyme association
print("\nAPOBEC enzyme (PCPG vs all):")
for enz in t1["Affecting_Over_Expressed_APOBEC"].dropna().unique():
    n_p = (pcpg_sites["Affecting_Over_Expressed_APOBEC"] == enz).sum()
    n_a = (t1["Affecting_Over_Expressed_APOBEC"] == enz).sum()
    print(f"  {enz}: PCPG={n_p}/{len(pcpg_sites)}, All={n_a}/{len(t1)}")

# Known cancer gene list (curated COSMIC Census + common)
KNOWN_CANCER_GENES = {
    "TP53", "BRCA1", "BRCA2", "KRAS", "EGFR", "MYC", "APC", "RB1", "PTEN",
    "PIK3CA", "VHL", "NF1", "NF2", "RET", "SDHB", "SDHD", "SDHA", "SDHC",
    "MAX", "TMEM127", "HRAS", "EPAS1", "FH",  # PCPG-specific
    "IDH1", "IDH2", "ATRX", "BRAF", "CDH1", "CDKN2A", "CTNNB1", "ERBB2",
    "FGFR3", "KIT", "MET", "MLH1", "MSH2", "NRAS", "PDGFRA", "RAF1",
    "SMAD4", "STK11", "TSC1", "TSC2", "WT1", "ALK", "JAK2", "ABL1",
    "NUP98", "STIM1", "SCD", "GOT1",
}

pcpg_cancer_genes = [g for g in pcpg_gene_counts if g in KNOWN_CANCER_GENES]
all_cancer_genes = [g for g in all_gene_counts if g in KNOWN_CANCER_GENES]
print(f"\nKnown cancer/PCPG genes in PCPG sites: {pcpg_cancer_genes}")
print(f"Known cancer genes in all sites: {all_cancer_genes}")

# Conservation comparison
print("\nConservation (non-primate editing):")
for grp_name, grp_df in [("PCPG", pcpg_sites), ("Non-PCPG", non_pcpg)]:
    yes = (grp_df["Any_Non-Primate_Editing"] == 1.0).sum()
    print(f"  {grp_name}: {yes}/{len(grp_df)} ({100*yes/len(grp_df):.1f}%) have non-primate editing")

results["analysis1"] = {
    "n_pcpg_sites": int(n_pcpg),
    "n_other_cancer": int(n_other),
    "n_no_cancer": int(n_none),
    "n_unique_pcpg_genes": len(pcpg_gene_counts),
    "top20_pcpg_genes": pcpg_gene_counts.most_common(20),
    "pcpg_cancer_genes": pcpg_cancer_genes,
    "genomic_categories": {
        cat: {
            "pcpg": int((pcpg_sites["Genomic_Category"] == cat).sum()),
            "all": int((t1["Genomic_Category"] == cat).sum()),
        }
        for cat in t1["Genomic_Category"].dropna().unique()
    },
}

# ============================================================================
# ANALYSIS 2: Editing Rate Comparison Across 54 Tissues
# ============================================================================
print("\n" + "=" * 80)
print("ANALYSIS 2: EDITING RATE COMPARISON ACROSS 54 TISSUES")
print("=" * 80)

# Mean editing rate per group across tissues
pcpg_idx = t1[t1["has_pcpg"]].index
non_pcpg_idx = t1[~t1["has_pcpg"]].index
other_cancer_idx = t1[(~t1["has_pcpg"]) & t1["has_any_cancer"]].index
no_cancer_idx = t1[~t1["has_any_cancer"]].index

tissue_stats = {}
for tc in TISSUE_COLS:
    rates = rate_matrix[tc]
    pcpg_rates = rates.loc[pcpg_idx].dropna()
    non_pcpg_rates = rates.loc[non_pcpg_idx].dropna()

    stat_result = stats.mannwhitneyu(pcpg_rates, non_pcpg_rates, alternative="two-sided") if len(pcpg_rates) > 5 and len(non_pcpg_rates) > 5 else None

    tissue_stats[tc] = {
        "pcpg_mean": float(pcpg_rates.mean()) if len(pcpg_rates) > 0 else np.nan,
        "pcpg_median": float(pcpg_rates.median()) if len(pcpg_rates) > 0 else np.nan,
        "non_pcpg_mean": float(non_pcpg_rates.mean()) if len(non_pcpg_rates) > 0 else np.nan,
        "non_pcpg_median": float(non_pcpg_rates.median()) if len(non_pcpg_rates) > 0 else np.nan,
        "n_pcpg": int(len(pcpg_rates)),
        "n_non_pcpg": int(len(non_pcpg_rates)),
        "mann_whitney_p": float(stat_result.pvalue) if stat_result else np.nan,
        "mann_whitney_U": float(stat_result.statistic) if stat_result else np.nan,
    }

# Sort tissues by fold difference
tissue_df = pd.DataFrame(tissue_stats).T
tissue_df["fold_diff"] = tissue_df["pcpg_mean"] / tissue_df["non_pcpg_mean"].replace(0, np.nan)
tissue_df = tissue_df.sort_values("fold_diff", ascending=False)

print("\nTop 10 tissues where PCPG sites have HIGHER editing:")
for i, (tc, row) in enumerate(tissue_df.head(10).iterrows()):
    sig = "*" if row["mann_whitney_p"] < 0.05 else ""
    print(f"  {i+1}. {tc}: PCPG={row['pcpg_mean']:.2f}%, non-PCPG={row['non_pcpg_mean']:.2f}% (fold={row['fold_diff']:.2f}, p={row['mann_whitney_p']:.4f}{sig})")

print("\nAdrenal/Kidney tissues specifically:")
for tc in ADRENAL_KIDNEY:
    if tc in tissue_df.index:
        row = tissue_df.loc[tc]
        print(f"  {tc}: PCPG={row['pcpg_mean']:.2f}%, non-PCPG={row['non_pcpg_mean']:.2f}% (fold={row['fold_diff']:.2f}, p={row['mann_whitney_p']:.4f})")

# Max editing rate comparison
pcpg_max = t1.loc[pcpg_idx, "Max_GTEx_Editing_Rate"].dropna()
non_pcpg_max = t1.loc[non_pcpg_idx, "Max_GTEx_Editing_Rate"].dropna()
mw = stats.mannwhitneyu(pcpg_max, non_pcpg_max, alternative="two-sided")
print(f"\nMax GTEx editing rate: PCPG median={pcpg_max.median():.2f}%, non-PCPG median={non_pcpg_max.median():.2f}% (MWU p={mw.pvalue:.2e})")

results["analysis2"] = {
    "tissue_stats": {
        k: {kk: vv for kk, vv in v.items() if not (isinstance(vv, float) and np.isnan(vv))}
        for k, v in tissue_stats.items()
    },
    "adrenal_kidney_enrichment": {
        tc: {"fold": float(tissue_df.loc[tc, "fold_diff"]), "p": float(tissue_df.loc[tc, "mann_whitney_p"])}
        for tc in ADRENAL_KIDNEY if tc in tissue_df.index
    },
}

# ============================================================================
# ANALYSIS 3: Tissue Specificity / Breadth
# ============================================================================
print("\n" + "=" * 80)
print("ANALYSIS 3: TISSUE SPECIFICITY OF PCPG SITES")
print("=" * 80)

# Number of tissues each site is edited in
pcpg_breadth = t1.loc[pcpg_idx, "Edited_In_#_Tissues"].dropna()
non_pcpg_breadth = t1.loc[non_pcpg_idx, "Edited_In_#_Tissues"].dropna()
other_cancer_breadth = t1.loc[other_cancer_idx, "Edited_In_#_Tissues"].dropna()
no_cancer_breadth = t1.loc[no_cancer_idx, "Edited_In_#_Tissues"].dropna()

mw_breadth = stats.mannwhitneyu(pcpg_breadth, non_pcpg_breadth, alternative="two-sided")
print(f"Tissue breadth (# tissues edited):")
print(f"  PCPG: median={pcpg_breadth.median():.0f}, mean={pcpg_breadth.mean():.1f}")
print(f"  Non-PCPG: median={non_pcpg_breadth.median():.0f}, mean={non_pcpg_breadth.mean():.1f}")
print(f"  Other cancer: median={other_cancer_breadth.median():.0f}, mean={other_cancer_breadth.mean():.1f}")
print(f"  No cancer: median={no_cancer_breadth.median():.0f}, mean={no_cancer_breadth.mean():.1f}")
print(f"  MWU PCPG vs non-PCPG: p={mw_breadth.pvalue:.2e}")

# Tissue classification
print("\nTissue classification:")
for cls in t1["Tissue_Classification"].dropna().unique():
    n_p = (pcpg_sites["Tissue_Classification"] == cls).sum()
    n_a = (t1["Tissue_Classification"] == cls).sum()
    pct_p = 100 * n_p / len(pcpg_sites) if len(pcpg_sites) > 0 else 0
    pct_a = 100 * n_a / len(t1)
    print(f"  {cls}: PCPG={n_p} ({pct_p:.1f}%), All={n_a} ({pct_a:.1f}%)")

# Adrenal-specific editing: how many PCPG sites are edited in Adrenal_Gland?
adrenal_rates = rate_matrix["Adrenal_Gland"]
pcpg_adrenal_edited = (adrenal_rates.loc[pcpg_idx] > 0).sum()
non_pcpg_adrenal_edited = (adrenal_rates.loc[non_pcpg_idx] > 0).sum()
print(f"\nEdited in Adrenal_Gland (rate > 0%):")
print(f"  PCPG: {pcpg_adrenal_edited}/{len(pcpg_idx)} ({100*pcpg_adrenal_edited/len(pcpg_idx):.1f}%)")
print(f"  Non-PCPG: {non_pcpg_adrenal_edited}/{len(non_pcpg_idx)} ({100*non_pcpg_adrenal_edited/len(non_pcpg_idx):.1f}%)")

# Tissue specificity index: coefficient of variation of editing rates
cv_pcpg = []
cv_non = []
for idx_val in pcpg_idx:
    rates_row = rate_matrix.loc[idx_val].dropna()
    if len(rates_row) >= 5 and rates_row.mean() > 0:
        cv_pcpg.append(rates_row.std() / rates_row.mean())
for idx_val in non_pcpg_idx:
    rates_row = rate_matrix.loc[idx_val].dropna()
    if len(rates_row) >= 5 and rates_row.mean() > 0:
        cv_non.append(rates_row.std() / rates_row.mean())

if cv_pcpg and cv_non:
    mw_cv = stats.mannwhitneyu(cv_pcpg, cv_non, alternative="two-sided")
    print(f"\nEditing rate CV (tissue specificity index):")
    print(f"  PCPG: median={np.median(cv_pcpg):.2f}, mean={np.mean(cv_pcpg):.2f}")
    print(f"  Non-PCPG: median={np.median(cv_non):.2f}, mean={np.mean(cv_non):.2f}")
    print(f"  MWU p={mw_cv.pvalue:.2e}")

results["analysis3"] = {
    "pcpg_breadth_median": float(pcpg_breadth.median()),
    "pcpg_breadth_mean": float(pcpg_breadth.mean()),
    "non_pcpg_breadth_median": float(non_pcpg_breadth.median()),
    "non_pcpg_breadth_mean": float(non_pcpg_breadth.mean()),
    "breadth_mwu_p": float(mw_breadth.pvalue),
    "pcpg_adrenal_edited_pct": float(100 * pcpg_adrenal_edited / len(pcpg_idx)),
    "non_pcpg_adrenal_edited_pct": float(100 * non_pcpg_adrenal_edited / len(non_pcpg_idx)),
}

# ============================================================================
# ANALYSIS 4: Sequence Motif Analysis
# ============================================================================
print("\n" + "=" * 80)
print("ANALYSIS 4: SEQUENCE MOTIF ANALYSIS")
print("=" * 80)

# For each site, the sequence is 200nt centered on the edit site (position 100)
# Extract +/-10nt context
CONTEXT_RADIUS = 10
CENTER = 100  # 0-indexed position of edit site in 200nt window

pcpg_contexts = []
non_pcpg_contexts = []
other_cancer_contexts = []
no_cancer_contexts = []

for idx_val in t1.index:
    sid = t1.loc[idx_val, "site_id"]
    if sid is None or sid not in sequences:
        continue
    seq = sequences[sid]
    if len(seq) < CENTER + CONTEXT_RADIUS + 1:
        continue
    context = seq[CENTER - CONTEXT_RADIUS : CENTER + CONTEXT_RADIUS + 1]
    if len(context) != 2 * CONTEXT_RADIUS + 1:
        continue

    group = t1.loc[idx_val, "cancer_group"]
    if group == "PCPG":
        pcpg_contexts.append(context)
    elif group == "Other cancer":
        other_cancer_contexts.append(context)
    else:
        no_cancer_contexts.append(context)

non_pcpg_all = other_cancer_contexts + no_cancer_contexts

print(f"PCPG contexts: {len(pcpg_contexts)}")
print(f"Non-PCPG contexts: {len(non_pcpg_all)}")

# Build position frequency matrices
BASES = ["A", "C", "G", "U"]


def build_pfm(contexts, length=21):
    """Build position frequency matrix from context strings."""
    pfm = np.zeros((length, 4))
    for ctx in contexts:
        for i, base in enumerate(ctx.upper()):
            if base in BASES:
                pfm[i, BASES.index(base)] += 1
    # Normalize to frequencies
    row_sums = pfm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    pfm = pfm / row_sums
    return pfm


pfm_pcpg = build_pfm(pcpg_contexts)
pfm_non_pcpg = build_pfm(non_pcpg_all)
pfm_diff = pfm_pcpg - pfm_non_pcpg

# Print motif differences
positions = list(range(-CONTEXT_RADIUS, CONTEXT_RADIUS + 1))
print("\nPosition frequency difference (PCPG - non-PCPG) at key positions:")
print(f"{'Pos':>4s}  {'A':>6s}  {'C':>6s}  {'G':>6s}  {'U':>6s}")
for i, pos in enumerate(positions):
    if abs(pos) <= 3 or abs(pfm_diff[i]).max() > 0.05:
        print(f"{pos:>4d}  {pfm_diff[i,0]:>+6.3f}  {pfm_diff[i,1]:>+6.3f}  {pfm_diff[i,2]:>+6.3f}  {pfm_diff[i,3]:>+6.3f}")

# Trinucleotide context (pos -1, 0, +1)
tri_pcpg = Counter()
tri_non = Counter()
for ctx in pcpg_contexts:
    tri = ctx[CONTEXT_RADIUS - 1: CONTEXT_RADIUS + 2].upper()
    tri_pcpg[tri] += 1
for ctx in non_pcpg_all:
    tri = ctx[CONTEXT_RADIUS - 1: CONTEXT_RADIUS + 2].upper()
    tri_non[tri] += 1

print("\nTrinucleotide context (NCN):")
all_tris = sorted(set(list(tri_pcpg.keys()) + list(tri_non.keys())))
print(f"{'Tri':>5s}  {'PCPG':>6s}  {'PCPG%':>6s}  {'non':>6s}  {'non%':>6s}")
for tri in all_tris:
    p_cnt = tri_pcpg.get(tri, 0)
    n_cnt = tri_non.get(tri, 0)
    p_pct = 100 * p_cnt / len(pcpg_contexts) if pcpg_contexts else 0
    n_pct = 100 * n_cnt / len(non_pcpg_all) if non_pcpg_all else 0
    if p_cnt + n_cnt >= 3:
        print(f"{tri:>5s}  {p_cnt:>6d}  {p_pct:>5.1f}%  {n_cnt:>6d}  {n_pct:>5.1f}%")

results["analysis4"] = {
    "n_pcpg_contexts": len(pcpg_contexts),
    "n_non_pcpg_contexts": len(non_pcpg_all),
    "trinucleotide_pcpg": dict(tri_pcpg.most_common(10)),
    "trinucleotide_non_pcpg": dict(tri_non.most_common(10)),
}

# ============================================================================
# ANALYSIS 5: Embedding Space Analysis (UMAP/tSNE)
# ============================================================================
print("\n" + "=" * 80)
print("ANALYSIS 5: EMBEDDING SPACE ANALYSIS")
print("=" * 80)

# Load embeddings
emb_orig = torch.load(EMB_PATH, map_location="cpu")
emb_edit = torch.load(EMB_ED_PATH, map_location="cpu")
with open(EMB_IDS_PATH) as f:
    emb_ids = json.load(f)

print(f"Embeddings: {len(emb_orig)} original, {len(emb_edit)} edited")
print(f"Embedding IDs: {len(emb_ids)}")

# Compute edit effect embeddings (edited - original) for Levanon sites
levanon_sids = set(t1["site_id"].dropna().tolist())
edit_effect_vecs = []
edit_effect_sids = []
edit_effect_groups = []

sid_to_group = dict(zip(t1["site_id"], t1["cancer_group"]))

for sid in emb_ids:
    if sid not in levanon_sids:
        continue
    if sid not in emb_orig or sid not in emb_edit:
        continue
    diff = emb_edit[sid] - emb_orig[sid]
    edit_effect_vecs.append(diff.numpy())
    edit_effect_sids.append(sid)
    edit_effect_groups.append(sid_to_group.get(sid, "No association"))

edit_effect_mat = np.stack(edit_effect_vecs)
print(f"Edit effect vectors: {edit_effect_mat.shape}")

# Also do original embeddings for comparison
orig_vecs = np.stack([emb_orig[sid].numpy() for sid in edit_effect_sids])

# t-SNE on edit effect embeddings
print("Running t-SNE on edit effect embeddings...", flush=True)
try:
    tsne_ee = TSNE(n_components=2, perplexity=min(30, len(edit_effect_mat) - 1),
                   random_state=42, max_iter=1000, n_jobs=1)
    tsne_coords_ee = tsne_ee.fit_transform(edit_effect_mat)
    print("  t-SNE on edit effects done.", flush=True)
except Exception as e:
    print(f"  t-SNE on edit effects failed: {e}", flush=True)
    tsne_coords_ee = np.random.randn(len(edit_effect_mat), 2)

# t-SNE on original embeddings
print("Running t-SNE on original embeddings...", flush=True)
try:
    tsne_orig = TSNE(n_components=2, perplexity=min(30, len(orig_vecs) - 1),
                     random_state=42, max_iter=1000, n_jobs=1)
    tsne_coords_orig = tsne_orig.fit_transform(orig_vecs)
    print("  t-SNE on original done.", flush=True)
except Exception as e:
    print(f"  t-SNE on original failed: {e}", flush=True)
    tsne_coords_orig = np.random.randn(len(orig_vecs), 2)

# UMAP on edit effect
try:
    import umap
    print("Running UMAP on edit effect embeddings...", flush=True)
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1,
                        n_jobs=1)  # single-threaded to avoid segfault
    umap_coords_ee = reducer.fit_transform(edit_effect_mat)
    has_umap = True
    print("  UMAP done.", flush=True)
except Exception as e:
    print(f"UMAP failed ({e}), skipping...")
    has_umap = False
    umap_coords_ee = None

# Compute centroid distances
groups_arr = np.array(edit_effect_groups)
pcpg_mask = groups_arr == "PCPG"
other_mask = groups_arr == "Other cancer"
none_mask = groups_arr == "No association"

if pcpg_mask.sum() > 0:
    centroid_pcpg = edit_effect_mat[pcpg_mask].mean(axis=0)
    centroid_other = edit_effect_mat[~pcpg_mask].mean(axis=0)
    dist = np.linalg.norm(centroid_pcpg - centroid_other)
    print(f"\nEdit effect centroid distance (PCPG vs rest): {dist:.4f}")

    # Within-group vs between-group distance
    from scipy.spatial.distance import cdist
    within_pcpg = cdist(edit_effect_mat[pcpg_mask], edit_effect_mat[pcpg_mask]).mean()
    within_other = cdist(edit_effect_mat[~pcpg_mask], edit_effect_mat[~pcpg_mask]).mean()
    between = cdist(edit_effect_mat[pcpg_mask], edit_effect_mat[~pcpg_mask]).mean()
    print(f"Within-PCPG mean distance: {within_pcpg:.4f}")
    print(f"Within-non-PCPG mean distance: {within_other:.4f}")
    print(f"Between-group mean distance: {between:.4f}")
    print(f"Ratio (between/within-PCPG): {between/within_pcpg:.3f}")

results["analysis5"] = {
    "n_edit_effect_vectors": int(edit_effect_mat.shape[0]),
    "n_pcpg_embedded": int(pcpg_mask.sum()),
    "centroid_distance": float(dist) if pcpg_mask.sum() > 0 else None,
    "within_pcpg_distance": float(within_pcpg) if pcpg_mask.sum() > 0 else None,
    "between_group_distance": float(between) if pcpg_mask.sum() > 0 else None,
}

# ============================================================================
# ANALYSIS 6: Model Prediction Analysis
# ============================================================================
print("\n" + "=" * 80)
print("ANALYSIS 6: MODEL PREDICTION ANALYSIS")
print("=" * 80)

# Load iteration3 predictions
pred_df = pd.read_csv(PRED_PATH)
print(f"Test predictions: {len(pred_df)} samples")

# Link to PCPG status
pred_df["site_key"] = pred_df["site_id"].map(
    lambda sid: (advisor.set_index("site_id").loc[sid, "chr"], advisor.set_index("site_id").loc[sid, "start"])
    if sid in advisor["site_id"].values else None
)

# Safer approach: build lookup
advisor_indexed = advisor.set_index("site_id")
pred_pcpg_status = []
for _, row in pred_df.iterrows():
    sid = row["site_id"]
    if sid in advisor_indexed.index:
        key = (advisor_indexed.loc[sid, "chr"], advisor_indexed.loc[sid, "start"])
        cancers = site_cancers.get(key, set())
        if "PCPG" in cancers:
            pred_pcpg_status.append("PCPG")
        elif len(cancers) > 0:
            pred_pcpg_status.append("Other cancer")
        else:
            pred_pcpg_status.append("No association")
    else:
        pred_pcpg_status.append("Non-Levanon")

pred_df["pcpg_group"] = pred_pcpg_status

# Filter to Levanon positive sites
levanon_pred = pred_df[pred_df["pcpg_group"] != "Non-Levanon"].copy()
print(f"Levanon sites in test set: {len(levanon_pred)}")
print(f"  PCPG: {(levanon_pred['pcpg_group']=='PCPG').sum()}")
print(f"  Other cancer: {(levanon_pred['pcpg_group']=='Other cancer').sum()}")
print(f"  No association: {(levanon_pred['pcpg_group']=='No association').sum()}")

# Compare P(edited) scores
for grp in ["PCPG", "Other cancer", "No association"]:
    subset = levanon_pred[levanon_pred["pcpg_group"] == grp]
    if len(subset) > 0:
        print(f"  {grp}: mean y_score={subset['y_score'].mean():.3f}, median={subset['y_score'].median():.3f}")

# Also check the "has_survival_association" column
if "has_survival_association" in pred_df.columns:
    surv_yes = pred_df[pred_df["has_survival_association"] == True]
    surv_no = pred_df[pred_df["has_survival_association"] == False]
    if len(surv_yes) > 0 and len(surv_no) > 0:
        mw_surv = stats.mannwhitneyu(surv_yes["y_score"].dropna(), surv_no["y_score"].dropna(), alternative="two-sided")
        print(f"\nSurvival-associated vs not (all datasets):")
        print(f"  Survival: mean={surv_yes['y_score'].mean():.3f}, n={len(surv_yes)}")
        print(f"  No survival: mean={surv_no['y_score'].mean():.3f}, n={len(surv_no)}")
        print(f"  MWU p={mw_surv.pvalue:.2e}")

# Rate predictions
rate_df = pd.read_csv(RATE_PATH)
rate_df_levanon = rate_df[rate_df["site_id"].isin(levanon_sids)].copy()
rate_pcpg_status = []
for _, row in rate_df_levanon.iterrows():
    sid = row["site_id"]
    if sid in advisor_indexed.index:
        key = (advisor_indexed.loc[sid, "chr"], advisor_indexed.loc[sid, "start"])
        cancers = site_cancers.get(key, set())
        rate_pcpg_status.append("PCPG" if "PCPG" in cancers else ("Other cancer" if cancers else "No association"))
    else:
        rate_pcpg_status.append("Unknown")
rate_df_levanon["pcpg_group"] = rate_pcpg_status

print(f"\nRate predictions for Levanon sites: {len(rate_df_levanon)}")
for grp in ["PCPG", "Other cancer", "No association"]:
    subset = rate_df_levanon[rate_df_levanon["pcpg_group"] == grp]
    if len(subset) > 0:
        corr = np.corrcoef(subset["true_rate"], subset["pred_rate"])[0, 1] if len(subset) > 2 else np.nan
        print(f"  {grp}: n={len(subset)}, true_rate_mean={subset['true_rate'].mean():.2f}, pred_rate_mean={subset['pred_rate'].mean():.2f}, corr={corr:.3f}")

results["analysis6"] = {
    "n_levanon_in_test": int(len(levanon_pred)),
    "n_pcpg_in_test": int((levanon_pred["pcpg_group"] == "PCPG").sum()),
    "pcpg_mean_score": float(levanon_pred[levanon_pred["pcpg_group"] == "PCPG"]["y_score"].mean())
    if (levanon_pred["pcpg_group"] == "PCPG").sum() > 0 else None,
}

# ============================================================================
# ANALYSIS 7: Cancer Type Breakdown
# ============================================================================
print("\n" + "=" * 80)
print("ANALYSIS 7: CANCER TYPE BREAKDOWN")
print("=" * 80)

# Flatten all cancer-site associations
cancer_type_sites = {}
for key, cancers in site_cancers.items():
    for c in cancers:
        if c not in cancer_type_sites:
            cancer_type_sites[c] = []
        cancer_type_sites[c].append(key)

# Cancer type descriptions
CANCER_NAMES = {
    "PCPG": "Pheochromocytoma/Paraganglioma",
    "KIRC": "Kidney Renal Clear Cell Carcinoma",
    "LGG": "Brain Lower Grade Glioma",
    "TGCT": "Testicular Germ Cell Tumors",
    "LUAD": "Lung Adenocarcinoma",
    "PAAD": "Pancreatic Adenocarcinoma",
    "SARC": "Sarcoma",
    "KIRP": "Kidney Renal Papillary Cell Carcinoma",
    "UCEC": "Uterine Corpus Endometrial Carcinoma",
    "CESC": "Cervical Squamous Cell Carcinoma",
    "ACC": "Adrenocortical Carcinoma",
    "LIHC": "Liver Hepatocellular Carcinoma",
    "COAD": "Colon Adenocarcinoma",
    "GBM": "Glioblastoma Multiforme",
    "ESCA": "Esophageal Carcinoma",
    "PRAD": "Prostate Adenocarcinoma",
    "BRCA": "Breast Invasive Carcinoma",
    "OV": "Ovarian Serous Cystadenocarcinoma",
    "LUSC": "Lung Squamous Cell Carcinoma",
    "HNSC": "Head and Neck Squamous Cell Carcinoma",
}

print(f"\n{'Cancer':>6s}  {'Sites':>6s}  {'Full Name'}")
print("-" * 60)
cancer_summary = {}
for cancer, sites in sorted(cancer_type_sites.items(), key=lambda x: -len(x[1])):
    full_name = CANCER_NAMES.get(cancer, cancer)
    print(f"{cancer:>6s}  {len(sites):>6d}  {full_name}")
    cancer_summary[cancer] = len(sites)

# Overlap analysis: how many sites are shared between PCPG and other cancers?
pcpg_site_set = set(cancer_type_sites.get("PCPG", []))
print(f"\nPCPG overlap with other cancer types:")
for cancer, sites in sorted(cancer_type_sites.items(), key=lambda x: -len(x[1])):
    if cancer == "PCPG":
        continue
    overlap = pcpg_site_set & set(sites)
    if overlap:
        print(f"  PCPG & {cancer}: {len(overlap)} shared sites ({100*len(overlap)/len(pcpg_site_set):.1f}% of PCPG)")

# Characterize each major cancer type by tissue breadth and editing rate
print("\nMean tissue breadth and editing rate by cancer type (top 10):")
for cancer in list(cancer_summary.keys())[:10]:
    site_keys = set(cancer_type_sites[cancer])
    mask = t1["site_key"].isin(site_keys)
    if mask.sum() == 0:
        continue
    breadth = t1.loc[mask, "Edited_In_#_Tissues"].mean()
    max_rate = t1.loc[mask, "Max_GTEx_Editing_Rate"].mean()
    print(f"  {cancer}: n={mask.sum()}, mean_breadth={breadth:.1f}, mean_max_rate={max_rate:.1f}%")

# PCPG-exclusive sites (only associated with PCPG, no other cancer)
pcpg_exclusive = []
for key in pcpg_site_set:
    if site_cancers[key] == {"PCPG"}:
        pcpg_exclusive.append(key)
print(f"\nPCPG-exclusive sites (only PCPG, no other cancer): {len(pcpg_exclusive)} / {len(pcpg_site_set)}")

# Characterize PCPG-exclusive vs PCPG-shared
pcpg_excl_mask = t1["site_key"].isin(set(pcpg_exclusive))
pcpg_shared_mask = t1["has_pcpg"] & ~pcpg_excl_mask
if pcpg_excl_mask.sum() > 0 and pcpg_shared_mask.sum() > 0:
    excl_breadth = t1.loc[pcpg_excl_mask, "Edited_In_#_Tissues"].mean()
    shared_breadth = t1.loc[pcpg_shared_mask, "Edited_In_#_Tissues"].mean()
    excl_rate = t1.loc[pcpg_excl_mask, "Max_GTEx_Editing_Rate"].mean()
    shared_rate = t1.loc[pcpg_shared_mask, "Max_GTEx_Editing_Rate"].mean()
    print(f"  Exclusive: breadth={excl_breadth:.1f}, max_rate={excl_rate:.1f}%")
    print(f"  Shared: breadth={shared_breadth:.1f}, max_rate={shared_rate:.1f}%")

results["analysis7"] = {
    "cancer_type_counts": cancer_summary,
    "pcpg_exclusive_count": len(pcpg_exclusive),
    "pcpg_total": len(pcpg_site_set),
}


# ============================================================================
# FIGURE 1: Multi-panel overview (6 panels)
# ============================================================================
print("\n" + "=" * 80)
print("GENERATING FIGURES")
print("=" * 80)

fig = plt.figure(figsize=(18, 24))
gs = gridspec.GridSpec(4, 3, hspace=0.35, wspace=0.3)

# ---------- Panel A: Cancer type barplot ----------
ax_a = fig.add_subplot(gs[0, 0])
top_cancers = sorted(cancer_summary.items(), key=lambda x: -x[1])[:15]
cancer_names_short = [c[0] for c in top_cancers]
cancer_counts = [c[1] for c in top_cancers]
colors_bar = [PCPG_COLOR if c == "PCPG" else "#7F7F7F" for c in cancer_names_short]
bars = ax_a.barh(range(len(cancer_names_short)), cancer_counts, color=colors_bar, edgecolor="white")
ax_a.set_yticks(range(len(cancer_names_short)))
ax_a.set_yticklabels(cancer_names_short)
ax_a.invert_yaxis()
ax_a.set_xlabel("Number of sites")
ax_a.set_title("A. Cancer Types with Survival-Associated\n    Editing Sites", fontweight="bold")
for i, (name, cnt) in enumerate(top_cancers):
    ax_a.text(cnt + 1, i, str(cnt), va="center", fontsize=7)

# ---------- Panel B: Genomic category comparison ----------
ax_b = fig.add_subplot(gs[0, 1])
cats = t1["Genomic_Category"].dropna().unique()
x_pos = np.arange(len(cats))
w = 0.35
pcpg_pcts = [100 * (pcpg_sites["Genomic_Category"] == c).sum() / len(pcpg_sites) for c in cats]
non_pcpg_pcts = [100 * (non_pcpg["Genomic_Category"] == c).sum() / len(non_pcpg) for c in cats]
ax_b.bar(x_pos - w / 2, pcpg_pcts, w, label="PCPG", color=PCPG_COLOR, alpha=0.8)
ax_b.bar(x_pos + w / 2, non_pcpg_pcts, w, label="Non-PCPG", color=NO_CANCER_COLOR, alpha=0.8)
ax_b.set_xticks(x_pos)
ax_b.set_xticklabels([c.replace(" ", "\n") for c in cats], fontsize=7)
ax_b.set_ylabel("Percentage of sites")
ax_b.set_title("B. Genomic Category Distribution", fontweight="bold")
ax_b.legend(fontsize=7)

# ---------- Panel C: Tissue breadth distribution ----------
ax_c = fig.add_subplot(gs[0, 2])
bins_breadth = np.arange(0, 55, 2)
ax_c.hist(pcpg_breadth, bins=bins_breadth, alpha=0.6, density=True, label="PCPG", color=PCPG_COLOR)
ax_c.hist(non_pcpg_breadth, bins=bins_breadth, alpha=0.6, density=True, label="Non-PCPG", color=NO_CANCER_COLOR)
ax_c.axvline(pcpg_breadth.median(), color=PCPG_COLOR, ls="--", lw=1.5)
ax_c.axvline(non_pcpg_breadth.median(), color=NO_CANCER_COLOR, ls="--", lw=1.5)
ax_c.set_xlabel("Number of tissues edited")
ax_c.set_ylabel("Density")
ax_c.set_title(f"C. Tissue Breadth Distribution\n    (MWU p={mw_breadth.pvalue:.2e})", fontweight="bold")
ax_c.legend(fontsize=7)

# ---------- Panel D: Editing rate heatmap across tissue groups ----------
ax_d = fig.add_subplot(gs[1, :])

# Focus on key tissue groups
key_tissues = ADRENAL_KIDNEY + ENDOCRINE + BRAIN_TISSUES[:6] + BLOOD_IMMUNE + ["Testis", "Liver", "Lung", "Colon_Sigmoid"]
key_tissues = list(OrderedDict.fromkeys(key_tissues))  # deduplicate preserving order
key_tissues = [t for t in key_tissues if t in TISSUE_COLS]

heatmap_data = []
for grp, grp_idx, color in [
    ("PCPG", pcpg_idx, PCPG_COLOR),
    ("Other cancer", other_cancer_idx, OTHER_CANCER_COLOR),
    ("No association", no_cancer_idx, NO_CANCER_COLOR),
]:
    means = []
    for tc in key_tissues:
        vals = rate_matrix[tc].loc[grp_idx].dropna()
        means.append(vals.mean() if len(vals) > 0 else 0)
    heatmap_data.append(means)

heatmap_arr = np.array(heatmap_data)
tissue_labels = [t.replace("_", " ") for t in key_tissues]
im = ax_d.imshow(heatmap_arr, aspect="auto", cmap="YlOrRd", interpolation="nearest")
ax_d.set_xticks(range(len(tissue_labels)))
ax_d.set_xticklabels(tissue_labels, rotation=55, ha="right", fontsize=7)
ax_d.set_yticks(range(3))
ax_d.set_yticklabels(["PCPG", "Other cancer", "No association"], fontsize=8)
ax_d.set_title("D. Mean Editing Rate (%) Across Key Tissues by Cancer Group", fontweight="bold")
cbar = plt.colorbar(im, ax=ax_d, shrink=0.6, label="Mean editing rate (%)")

# Add values to cells
for i in range(heatmap_arr.shape[0]):
    for j in range(heatmap_arr.shape[1]):
        val = heatmap_arr[i, j]
        text_color = "white" if val > heatmap_arr.max() * 0.6 else "black"
        ax_d.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=6, color=text_color)

# ---------- Panel E: t-SNE of edit effect embeddings ----------
ax_e = fig.add_subplot(gs[2, 0])
group_arr = np.array(edit_effect_groups)
# Plot in reverse order so PCPG is on top
for grp, color, marker, size, zorder in [
    ("No association", NO_CANCER_COLOR, "o", 15, 1),
    ("Other cancer", OTHER_CANCER_COLOR, "s", 20, 2),
    ("PCPG", PCPG_COLOR, "o", 20, 3),
]:
    mask = group_arr == grp
    if mask.sum() > 0:
        ax_e.scatter(tsne_coords_ee[mask, 0], tsne_coords_ee[mask, 1],
                     c=color, s=size, alpha=0.6, label=f"{grp} (n={mask.sum()})",
                     edgecolors="white", linewidths=0.3, marker=marker, zorder=zorder)
ax_e.set_xlabel("t-SNE 1")
ax_e.set_ylabel("t-SNE 2")
ax_e.set_title("E. Edit Effect Embedding Space\n    (t-SNE)", fontweight="bold")
ax_e.legend(fontsize=6, markerscale=1.5)

# ---------- Panel F: UMAP of edit effect embeddings ----------
ax_f = fig.add_subplot(gs[2, 1])
if has_umap and umap_coords_ee is not None:
    for grp, color, marker, size, zorder in [
        ("No association", NO_CANCER_COLOR, "o", 15, 1),
        ("Other cancer", OTHER_CANCER_COLOR, "s", 20, 2),
        ("PCPG", PCPG_COLOR, "o", 20, 3),
    ]:
        mask = group_arr == grp
        if mask.sum() > 0:
            ax_f.scatter(umap_coords_ee[mask, 0], umap_coords_ee[mask, 1],
                         c=color, s=size, alpha=0.6, label=f"{grp} (n={mask.sum()})",
                         edgecolors="white", linewidths=0.3, marker=marker, zorder=zorder)
    ax_f.set_xlabel("UMAP 1")
    ax_f.set_ylabel("UMAP 2")
    ax_f.set_title("F. Edit Effect Embedding Space\n    (UMAP)", fontweight="bold")
    ax_f.legend(fontsize=6, markerscale=1.5)
else:
    ax_f.text(0.5, 0.5, "UMAP not available", ha="center", va="center", transform=ax_f.transAxes)
    ax_f.set_title("F. Edit Effect Embedding Space (UMAP)", fontweight="bold")

# ---------- Panel G: Model predictions by PCPG status ----------
ax_g = fig.add_subplot(gs[2, 2])
plot_groups = ["PCPG", "Other cancer", "No association"]
plot_data = [levanon_pred[levanon_pred["pcpg_group"] == g]["y_score"].dropna().values for g in plot_groups]
plot_data = [d for d in plot_data if len(d) > 0]
plot_labels = [g for g, d in zip(plot_groups, [levanon_pred[levanon_pred["pcpg_group"] == g]["y_score"].dropna() for g in plot_groups]) if len(d) > 0]
bp = ax_g.boxplot(plot_data, labels=plot_labels, patch_artist=True, widths=0.5)
for patch, grp in zip(bp["boxes"], plot_labels):
    patch.set_facecolor(PALETTE.get(grp, "#7F7F7F"))
    patch.set_alpha(0.7)
ax_g.set_ylabel("P(edited) score")
ax_g.set_title("G. EditRNA-A3A Prediction Scores\n    by Cancer Group", fontweight="bold")
ax_g.tick_params(axis="x", rotation=15)

# ---------- Panel H: Motif heatmap ----------
ax_h = fig.add_subplot(gs[3, 0:2])
# Plot PFM difference
pos_labels = [str(p) for p in positions]
im2 = ax_h.imshow(pfm_diff.T, aspect="auto", cmap="RdBu_r", vmin=-0.15, vmax=0.15, interpolation="nearest")
ax_h.set_xticks(range(len(pos_labels)))
ax_h.set_xticklabels(pos_labels, fontsize=7)
ax_h.set_yticks(range(4))
ax_h.set_yticklabels(BASES)
ax_h.set_xlabel("Position relative to edit site")
ax_h.set_title("H. Sequence Context Difference (PCPG - Non-PCPG)\n    Base Frequency", fontweight="bold")
plt.colorbar(im2, ax=ax_h, shrink=0.6, label="Frequency difference")
# Mark the edit site
ax_h.axvline(CONTEXT_RADIUS, color="black", ls=":", lw=1)
ax_h.text(CONTEXT_RADIUS, -0.7, "C>U", ha="center", fontsize=7, fontweight="bold")

# ---------- Panel I: Adrenal editing rate scatter ----------
ax_i = fig.add_subplot(gs[3, 2])
for grp, color, zorder in [
    ("No association", NO_CANCER_COLOR, 1),
    ("Other cancer", OTHER_CANCER_COLOR, 2),
    ("PCPG", PCPG_COLOR, 3),
]:
    mask = t1["cancer_group"] == grp
    adrenal_vals = rate_matrix["Adrenal_Gland"].loc[t1[mask].index]
    max_vals = t1.loc[mask, "Max_GTEx_Editing_Rate"]
    valid = adrenal_vals.notna() & max_vals.notna()
    if valid.sum() > 0:
        ax_i.scatter(max_vals[valid], adrenal_vals[valid],
                     c=color, s=15, alpha=0.5, label=f"{grp} (n={valid.sum()})",
                     edgecolors="white", linewidths=0.2, zorder=zorder)
ax_i.set_xlabel("Max GTEx Editing Rate (%)")
ax_i.set_ylabel("Adrenal Gland Editing Rate (%)")
ax_i.set_title("I. Adrenal vs Max Editing Rate", fontweight="bold")
ax_i.legend(fontsize=6, markerscale=1.5)
ax_i.plot([0, 100], [0, 100], "k--", alpha=0.3, lw=0.5)

plt.savefig(os.path.join(OUT, "pcpg_overview_multipanel.png"), dpi=300)
plt.close()
print(f"Saved: pcpg_overview_multipanel.png")

# ============================================================================
# FIGURE 2: Detailed tissue editing rate comparison
# ============================================================================
fig2, axes2 = plt.subplots(2, 1, figsize=(16, 12))

# Panel A: Mean editing rate per tissue, grouped by PCPG status
ax = axes2[0]
tissue_order = tissue_df.sort_values("fold_diff", ascending=False).index.tolist()
x = np.arange(len(tissue_order))
w = 0.35
pcpg_means = [tissue_df.loc[t, "pcpg_mean"] for t in tissue_order]
non_pcpg_means = [tissue_df.loc[t, "non_pcpg_mean"] for t in tissue_order]
ax.bar(x - w / 2, pcpg_means, w, color=PCPG_COLOR, alpha=0.8, label="PCPG sites")
ax.bar(x + w / 2, non_pcpg_means, w, color=NO_CANCER_COLOR, alpha=0.8, label="Non-PCPG sites")
# Mark significant
for i, tc in enumerate(tissue_order):
    if tissue_df.loc[tc, "mann_whitney_p"] < 0.05:
        ax.text(i, max(pcpg_means[i], non_pcpg_means[i]) + 0.1, "*", ha="center", fontsize=8, color="red")
ax.set_xticks(x)
ax.set_xticklabels([t.replace("_", "\n") for t in tissue_order], rotation=90, fontsize=5)
ax.set_ylabel("Mean editing rate (%)")
ax.set_title("A. Mean Editing Rate Across All 54 Tissues (PCPG vs Non-PCPG sites)\n    * = p < 0.05 (Mann-Whitney U)", fontweight="bold")
ax.legend(fontsize=8)

# Panel B: Fold difference
ax2 = axes2[1]
folds = tissue_df.loc[tissue_order, "fold_diff"].values
colors_fold = [PCPG_COLOR if f > 1 else NO_CANCER_COLOR for f in folds]
ax2.bar(x, folds, color=colors_fold, alpha=0.8)
ax2.axhline(1.0, color="black", ls="--", lw=0.5)
ax2.set_xticks(x)
ax2.set_xticklabels([t.replace("_", "\n") for t in tissue_order], rotation=90, fontsize=5)
ax2.set_ylabel("Fold difference (PCPG / Non-PCPG)")
ax2.set_title("B. Fold Difference in Mean Editing Rate (PCPG / Non-PCPG)", fontweight="bold")

# Highlight adrenal/kidney
for i, tc in enumerate(tissue_order):
    if tc in ADRENAL_KIDNEY:
        ax2.get_children()[i].set_edgecolor("black")
        ax2.get_children()[i].set_linewidth(2)

plt.tight_layout()
plt.savefig(os.path.join(OUT, "pcpg_tissue_comparison.png"), dpi=300)
plt.close()
print(f"Saved: pcpg_tissue_comparison.png")

# ============================================================================
# FIGURE 3: Gene-level analysis
# ============================================================================
fig3, axes3 = plt.subplots(1, 2, figsize=(14, 8))

# Panel A: Top genes in PCPG
ax = axes3[0]
top_genes = pcpg_gene_counts.most_common(25)
gene_names = [g[0] for g in top_genes]
gene_cnts = [g[1] for g in top_genes]
colors_gene = [PCPG_COLOR if g in KNOWN_CANCER_GENES else "#7F7F7F" for g in gene_names]
ax.barh(range(len(gene_names)), gene_cnts, color=colors_gene, edgecolor="white")
ax.set_yticks(range(len(gene_names)))
ax.set_yticklabels(gene_names, fontsize=7)
ax.invert_yaxis()
ax.set_xlabel("Number of PCPG-associated sites")
ax.set_title("Top 25 Genes with PCPG-Associated Sites\n(Red = known cancer gene)", fontweight="bold")

# Panel B: PCPG exclusive vs shared
ax2 = axes3[1]
# Count number of cancer types per PCPG site
pcpg_cancer_counts = []
for key in pcpg_site_set:
    pcpg_cancer_counts.append(len(site_cancers[key]))
cnt_hist = Counter(pcpg_cancer_counts)
x_vals = sorted(cnt_hist.keys())
y_vals = [cnt_hist[x] for x in x_vals]
ax2.bar(x_vals, y_vals, color=PCPG_COLOR, alpha=0.8, edgecolor="white")
ax2.set_xlabel("Number of cancer types per site")
ax2.set_ylabel("Number of sites")
ax2.set_title(f"PCPG Sites: Number of Associated Cancer Types\n({len(pcpg_exclusive)}/{len(pcpg_site_set)} are PCPG-exclusive)", fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(OUT, "pcpg_gene_analysis.png"), dpi=300)
plt.close()
print(f"Saved: pcpg_gene_analysis.png")

# ============================================================================
# FIGURE 4: Sequence motif visualization
# ============================================================================
fig4, axes4 = plt.subplots(3, 1, figsize=(12, 10))

# Panel A: PCPG PFM
ax = axes4[0]
bottom = np.zeros(len(positions))
base_colors = {"A": "#2CA02C", "C": "#1F77B4", "G": "#FF7F0E", "U": "#D62728"}
for bi, base in enumerate(BASES):
    ax.bar(positions, pfm_pcpg[:, bi], bottom=bottom, color=base_colors[base], label=base, width=0.8)
    bottom += pfm_pcpg[:, bi]
ax.set_ylabel("Base frequency")
ax.set_title("A. PCPG Site Sequence Context", fontweight="bold")
ax.legend(fontsize=7, ncol=4)
ax.axvline(0, color="black", ls=":", lw=1)

# Panel B: Non-PCPG PFM
ax = axes4[1]
bottom = np.zeros(len(positions))
for bi, base in enumerate(BASES):
    ax.bar(positions, pfm_non_pcpg[:, bi], bottom=bottom, color=base_colors[base], label=base, width=0.8)
    bottom += pfm_non_pcpg[:, bi]
ax.set_ylabel("Base frequency")
ax.set_title("B. Non-PCPG Site Sequence Context", fontweight="bold")
ax.legend(fontsize=7, ncol=4)
ax.axvline(0, color="black", ls=":", lw=1)

# Panel C: Difference
ax = axes4[2]
for bi, base in enumerate(BASES):
    ax.plot(positions, pfm_diff[:, bi], color=base_colors[base], label=base, marker="o", markersize=3, lw=1.5)
ax.axhline(0, color="gray", ls="--", lw=0.5)
ax.axvline(0, color="black", ls=":", lw=1)
ax.set_xlabel("Position relative to edit site (C>U)")
ax.set_ylabel("Frequency difference\n(PCPG - non-PCPG)")
ax.set_title("C. Motif Difference (PCPG - Non-PCPG)", fontweight="bold")
ax.legend(fontsize=7, ncol=4)

plt.tight_layout()
plt.savefig(os.path.join(OUT, "pcpg_motif_analysis.png"), dpi=300)
plt.close()
print(f"Saved: pcpg_motif_analysis.png")

# ============================================================================
# FIGURE 5: Detailed embedding analysis
# ============================================================================
fig5, axes5 = plt.subplots(1, 3, figsize=(18, 6))

# Panel A: t-SNE colored by number of tissues
n_tissues_arr = []
for sid in edit_effect_sids:
    row = t1[t1["site_id"] == sid]
    if len(row) > 0:
        n_tissues_arr.append(row["Edited_In_#_Tissues"].values[0])
    else:
        n_tissues_arr.append(0)
n_tissues_arr = np.array(n_tissues_arr)

ax = axes5[0]
sc = ax.scatter(tsne_coords_ee[:, 0], tsne_coords_ee[:, 1],
                c=n_tissues_arr, cmap="viridis", s=15, alpha=0.6, edgecolors="white", linewidths=0.2)
plt.colorbar(sc, ax=ax, label="# Tissues edited")
ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")
ax.set_title("A. Edit Effect t-SNE\n    Colored by Tissue Breadth", fontweight="bold")

# Panel B: t-SNE colored by max editing rate
max_rate_arr = []
for sid in edit_effect_sids:
    row = t1[t1["site_id"] == sid]
    if len(row) > 0:
        max_rate_arr.append(row["Max_GTEx_Editing_Rate"].values[0])
    else:
        max_rate_arr.append(0)
max_rate_arr = np.array(max_rate_arr)

ax = axes5[1]
sc = ax.scatter(tsne_coords_ee[:, 0], tsne_coords_ee[:, 1],
                c=np.log1p(max_rate_arr), cmap="magma", s=15, alpha=0.6, edgecolors="white", linewidths=0.2)
plt.colorbar(sc, ax=ax, label="log(1 + Max editing rate %)")
ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")
ax.set_title("B. Edit Effect t-SNE\n    Colored by Max Editing Rate", fontweight="bold")

# Panel C: Edit effect vector magnitude
ee_magnitude = np.linalg.norm(edit_effect_mat, axis=1)
ax = axes5[2]
for grp, color in [("No association", NO_CANCER_COLOR), ("Other cancer", OTHER_CANCER_COLOR), ("PCPG", PCPG_COLOR)]:
    mask = group_arr == grp
    if mask.sum() > 0:
        ax.hist(ee_magnitude[mask], bins=30, alpha=0.5, density=True, label=f"{grp} (n={mask.sum()})", color=color)
ax.set_xlabel("Edit effect embedding magnitude (L2 norm)")
ax.set_ylabel("Density")
ax.set_title("C. Edit Effect Vector Magnitude\n    Distribution", fontweight="bold")
ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig(os.path.join(OUT, "pcpg_embedding_analysis.png"), dpi=300)
plt.close()
print(f"Saved: pcpg_embedding_analysis.png")

# ============================================================================
# FIGURE 6: Cancer comparison radar / profile
# ============================================================================
fig6, axes6 = plt.subplots(2, 2, figsize=(14, 12))

# Panel A: Overlap matrix (top cancers)
ax = axes6[0, 0]
top_cancer_types = [c for c, _ in sorted(cancer_summary.items(), key=lambda x: -x[1])[:10]]
overlap_matrix = np.zeros((len(top_cancer_types), len(top_cancer_types)))
for i, c1 in enumerate(top_cancer_types):
    for j, c2 in enumerate(top_cancer_types):
        s1 = set(cancer_type_sites.get(c1, []))
        s2 = set(cancer_type_sites.get(c2, []))
        if len(s1) > 0:
            overlap_matrix[i, j] = len(s1 & s2) / len(s1) * 100
im = ax.imshow(overlap_matrix, cmap="Blues", interpolation="nearest")
ax.set_xticks(range(len(top_cancer_types)))
ax.set_xticklabels(top_cancer_types, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(len(top_cancer_types)))
ax.set_yticklabels(top_cancer_types, fontsize=8)
plt.colorbar(im, ax=ax, shrink=0.8, label="% overlap (row cancer)")
ax.set_title("A. Site Overlap Between Cancer Types\n    (% of row cancer's sites)", fontweight="bold")
for i in range(len(top_cancer_types)):
    for j in range(len(top_cancer_types)):
        val = overlap_matrix[i, j]
        if val > 0:
            text_color = "white" if val > 50 else "black"
            ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=6, color=text_color)

# Panel B: Mean tissue breadth by cancer type
ax = axes6[0, 1]
breadths = []
for cancer in top_cancer_types:
    keys = set(cancer_type_sites[cancer])
    mask = t1["site_key"].isin(keys)
    breadths.append(t1.loc[mask, "Edited_In_#_Tissues"].mean())
colors_ct = [PCPG_COLOR if c == "PCPG" else "#7F7F7F" for c in top_cancer_types]
ax.barh(range(len(top_cancer_types)), breadths, color=colors_ct, edgecolor="white")
ax.set_yticks(range(len(top_cancer_types)))
ax.set_yticklabels(top_cancer_types)
ax.invert_yaxis()
ax.set_xlabel("Mean number of tissues edited")
ax.set_title("B. Tissue Breadth by Cancer Type", fontweight="bold")

# Panel C: Mean max editing rate by cancer type
ax = axes6[1, 0]
max_rates = []
for cancer in top_cancer_types:
    keys = set(cancer_type_sites[cancer])
    mask = t1["site_key"].isin(keys)
    max_rates.append(t1.loc[mask, "Max_GTEx_Editing_Rate"].mean())
ax.barh(range(len(top_cancer_types)), max_rates, color=colors_ct, edgecolor="white")
ax.set_yticks(range(len(top_cancer_types)))
ax.set_yticklabels(top_cancer_types)
ax.invert_yaxis()
ax.set_xlabel("Mean max GTEx editing rate (%)")
ax.set_title("C. Max Editing Rate by Cancer Type", fontweight="bold")

# Panel D: APOBEC enzyme preference by cancer type
ax = axes6[1, 1]
apobec_cats = ["APOBEC3A Only", "APOBEC3G Only", "Both", "Neither"]
apobec_data = {}
for cancer in top_cancer_types:
    keys = set(cancer_type_sites[cancer])
    mask = t1["site_key"].isin(keys)
    subset = t1.loc[mask]
    total = len(subset)
    if total == 0:
        apobec_data[cancer] = [0] * 4
        continue
    apobec_data[cancer] = [
        100 * (subset["Affecting_Over_Expressed_APOBEC"] == cat).sum() / total
        for cat in apobec_cats
    ]

apobec_df = pd.DataFrame(apobec_data, index=apobec_cats).T
apobec_colors = ["#E41A1C", "#377EB8", "#984EA3", "#999999"]
bottom_vals = np.zeros(len(top_cancer_types))
for i, cat in enumerate(apobec_cats):
    ax.barh(range(len(top_cancer_types)), apobec_df[cat].values,
            left=bottom_vals, color=apobec_colors[i], label=cat, alpha=0.8)
    bottom_vals += apobec_df[cat].values
ax.set_yticks(range(len(top_cancer_types)))
ax.set_yticklabels(top_cancer_types)
ax.invert_yaxis()
ax.set_xlabel("Percentage of sites")
ax.set_title("D. APOBEC Enzyme Preference by Cancer Type", fontweight="bold")
ax.legend(fontsize=6, loc="lower right")

plt.tight_layout()
plt.savefig(os.path.join(OUT, "pcpg_cancer_comparison.png"), dpi=300)
plt.close()
print(f"Saved: pcpg_cancer_comparison.png")

# ============================================================================
# FIGURE 7: PCPG-specific adrenal / endocrine focus
# ============================================================================
fig7, axes7 = plt.subplots(2, 2, figsize=(14, 12))

# Panel A: Adrenal editing rates violin plot
ax = axes7[0, 0]
adrenal_data = {}
for grp in ["PCPG", "Other cancer", "No association"]:
    mask = t1["cancer_group"] == grp
    vals = rate_matrix["Adrenal_Gland"].loc[t1[mask].index].dropna()
    adrenal_data[grp] = vals.values

parts = ax.violinplot([adrenal_data[g] for g in ["PCPG", "Other cancer", "No association"] if len(adrenal_data.get(g, [])) > 0],
                       positions=range(sum(1 for g in ["PCPG", "Other cancer", "No association"] if len(adrenal_data.get(g, [])) > 0)),
                       showmeans=True, showmedians=True)
valid_groups = [g for g in ["PCPG", "Other cancer", "No association"] if len(adrenal_data.get(g, [])) > 0]
ax.set_xticks(range(len(valid_groups)))
ax.set_xticklabels(valid_groups)
ax.set_ylabel("Adrenal Gland Editing Rate (%)")
ax.set_title("A. Adrenal Gland Editing by Cancer Group", fontweight="bold")

# Color violins
colors_v = [PALETTE[g] for g in valid_groups]
for pc, color in zip(parts["bodies"], colors_v):
    pc.set_facecolor(color)
    pc.set_alpha(0.6)

# Panel B: Endocrine tissue editing (PCPG vs non-PCPG)
ax = axes7[0, 1]
endocrine_order = [t for t in ENDOCRINE if t in TISSUE_COLS]
x = np.arange(len(endocrine_order))
w = 0.35
pcpg_endo = [rate_matrix[t].loc[pcpg_idx].dropna().mean() for t in endocrine_order]
non_pcpg_endo = [rate_matrix[t].loc[non_pcpg_idx].dropna().mean() for t in endocrine_order]
ax.bar(x - w / 2, pcpg_endo, w, color=PCPG_COLOR, alpha=0.8, label="PCPG")
ax.bar(x + w / 2, non_pcpg_endo, w, color=NO_CANCER_COLOR, alpha=0.8, label="Non-PCPG")
ax.set_xticks(x)
ax.set_xticklabels([t.replace("_", "\n") for t in endocrine_order], fontsize=8)
ax.set_ylabel("Mean editing rate (%)")
ax.set_title("B. Endocrine Tissue Editing Rates", fontweight="bold")
ax.legend(fontsize=8)

# Panel C: Kidney editing comparison
ax = axes7[1, 0]
kidney_tissues = [t for t in ["Kidney_Cortex", "Kidney_Medulla"] if t in TISSUE_COLS]
for i, tc in enumerate(kidney_tissues):
    pcpg_vals = rate_matrix[tc].loc[pcpg_idx].dropna()
    non_pcpg_vals = rate_matrix[tc].loc[non_pcpg_idx].dropna()
    positions_bp = [i * 3, i * 3 + 1]
    bp1 = ax.boxplot([pcpg_vals.values], positions=[positions_bp[0]], patch_artist=True, widths=0.6)
    bp2 = ax.boxplot([non_pcpg_vals.values], positions=[positions_bp[1]], patch_artist=True, widths=0.6)
    bp1["boxes"][0].set_facecolor(PCPG_COLOR)
    bp1["boxes"][0].set_alpha(0.7)
    bp2["boxes"][0].set_facecolor(NO_CANCER_COLOR)
    bp2["boxes"][0].set_alpha(0.7)
    # Stat test
    if len(pcpg_vals) > 5 and len(non_pcpg_vals) > 5:
        mw_k = stats.mannwhitneyu(pcpg_vals, non_pcpg_vals, alternative="two-sided")
        ax.text(i * 3 + 0.5, ax.get_ylim()[1] * 0.9, f"p={mw_k.pvalue:.3f}", ha="center", fontsize=7)

ax.set_xticks([0.5, 3.5])
ax.set_xticklabels(kidney_tissues)
ax.set_ylabel("Editing Rate (%)")
ax.set_title("C. Kidney Tissue Editing Rates", fontweight="bold")
legend_elements = [Patch(facecolor=PCPG_COLOR, alpha=0.7, label="PCPG"),
                   Patch(facecolor=NO_CANCER_COLOR, alpha=0.7, label="Non-PCPG")]
ax.legend(handles=legend_elements, fontsize=8)

# Panel D: Number of cancers distribution
ax = axes7[1, 1]
n_cancers_all = t1["cancers"].apply(len)
n_cancers_counts = Counter(n_cancers_all)
x_vals = sorted(n_cancers_counts.keys())
y_vals = [n_cancers_counts[x] for x in x_vals]
ax.bar(x_vals, y_vals, color="#7F7F7F", edgecolor="white", alpha=0.8)
ax.set_xlabel("Number of cancer types per site")
ax.set_ylabel("Number of sites")
ax.set_title("D. Distribution of Cancer Associations\n    per Editing Site", fontweight="bold")
# Annotate
for x_val, y_val in zip(x_vals, y_vals):
    if y_val > 5:
        ax.text(x_val, y_val + 3, str(y_val), ha="center", fontsize=7)

plt.tight_layout()
plt.savefig(os.path.join(OUT, "pcpg_endocrine_focus.png"), dpi=300)
plt.close()
print(f"Saved: pcpg_endocrine_focus.png")

# ============================================================================
# Save all results to JSON
# ============================================================================
# Make results JSON-serializable
def make_serializable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, dict):
        return {str(k): make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(item) for item in obj]
    elif isinstance(obj, pd.Timestamp):
        return str(obj)
    elif pd.isna(obj) if isinstance(obj, float) else False:
        return None
    return obj

results_serializable = make_serializable(results)

with open(os.path.join(OUT, "pcpg_analysis_results.json"), "w") as f:
    json.dump(results_serializable, f, indent=2, default=str)
print(f"\nSaved: pcpg_analysis_results.json")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"""
PCPG Cancer Deep-Dive Analysis Complete
========================================

Key Findings:
1. PCPG dominance: {n_pcpg}/{len(t1)} sites ({100*n_pcpg/len(t1):.1f}%) have PCPG survival associations
   - This is {n_pcpg/max(n_other,1):.1f}x more than the next cancer type (KIRC: {cancer_summary.get('KIRC',0)})
   - {len(pcpg_exclusive)}/{len(pcpg_site_set)} PCPG sites are PCPG-exclusive

2. Tissue breadth: PCPG sites are edited in median {pcpg_breadth.median():.0f} tissues
   vs {non_pcpg_breadth.median():.0f} for non-PCPG (MWU p={mw_breadth.pvalue:.2e})

3. Adrenal editing: {100*pcpg_adrenal_edited/len(pcpg_idx):.1f}% of PCPG sites show adrenal editing
   vs {100*non_pcpg_adrenal_edited/len(non_pcpg_idx):.1f}% of non-PCPG sites

4. Known cancer/PCPG genes in PCPG sites: {pcpg_cancer_genes}

5. Embedding analysis: {edit_effect_mat.shape[0]} sites with edit effect vectors

Output directory: {OUT}
Files generated:
  - pcpg_overview_multipanel.png   (9-panel overview)
  - pcpg_tissue_comparison.png     (54-tissue rate comparison)
  - pcpg_gene_analysis.png         (gene enrichment)
  - pcpg_motif_analysis.png        (sequence motif)
  - pcpg_embedding_analysis.png    (embedding space)
  - pcpg_cancer_comparison.png     (cross-cancer comparison)
  - pcpg_endocrine_focus.png       (adrenal/kidney focus)
  - pcpg_analysis_results.json     (all numerical results)
""")
