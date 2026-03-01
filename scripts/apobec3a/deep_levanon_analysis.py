"""
Comprehensive analysis of the Levanon (advisor) dataset and cross-dataset comparisons.
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

BASE = '/Users/shaharharel/Documents/github/edit-rna-apobec'

# ============================================================================
# PART 1: Levanon 636-site tissue profiling
# ============================================================================
print("=" * 80)
print("PART 1: LEVANON 636-SITE TISSUE PROFILING")
print("=" * 80)

t1 = pd.read_csv(f'{BASE}/data/processed/advisor/t1_gtex_editing_&_conservation.csv')
print(f"\nTotal sites: {len(t1)}")

# Identify the 54 tissue columns (they start after conservation columns)
# Tissue columns are from index 26 onward based on the column list
tissue_cols = [c for c in t1.columns if c not in [
    'Chr', 'Start', 'End', 'Genomic_Category', 'Gene_(RefSeq)',
    'mRNA_location_(RefSeq)', 'Exonic_Function', 'Edited_In_#_Tissues',
    'Edited_Tissues_(Z_score_≥_2)', 'Tissue_Classification',
    'Affecting_Over_Expressed_APOBEC', 'Max_GTEx_Editing_Rate',
    'Mean_GTEx_Editing_Rate', 'GTEx_Editing_Rate_SD',
    'Any_Non-Primate_Editing', 'Any_Non-Primate_Editing_≥_1%',
    'Any_Primate_Editing', 'Any_Primate_Editing_≥_1%',
    'Any_Mammalian_Editing', 'Any_Mammlian_Editing_≥_1%',
    'non-Boreoeutheria_(Primitve_mammals)',
    'Laurasiatheria_(non_rodent_or_primate_placental_mammals)',
    'Glires_(rodents_&_rabbits)',
    'non-Catarrhini_Primates_(new_world_monekys_and_lemurs)',
    'Cercopithecinae_(most_old_world_monkeys)',
    'Laurasiatherianon-Human_Homininae_(Apes)'
]]

print(f"Number of tissue columns: {len(tissue_cols)}")

# Parse tissue data: format is "reads;total_reads;rate"
def parse_tissue_rate(val):
    """Extract editing rate from 'reads;total_reads;rate' format."""
    if pd.isna(val) or val == '' or val == 'nan':
        return np.nan
    parts = str(val).split(';')
    if len(parts) == 3:
        try:
            return float(parts[2])
        except ValueError:
            return np.nan
    return np.nan

def parse_tissue_reads(val):
    """Extract edited reads from 'reads;total_reads;rate' format."""
    if pd.isna(val) or val == '' or val == 'nan':
        return np.nan
    parts = str(val).split(';')
    if len(parts) == 3:
        try:
            return float(parts[0])
        except ValueError:
            return np.nan
    return np.nan

def parse_tissue_total(val):
    """Extract total reads from 'reads;total_reads;rate' format."""
    if pd.isna(val) or val == '' or val == 'nan':
        return np.nan
    parts = str(val).split(';')
    if len(parts) == 3:
        try:
            return float(parts[1])
        except ValueError:
            return np.nan
    return np.nan

# Build rate matrix
rate_matrix = pd.DataFrame()
for tc in tissue_cols:
    rate_matrix[tc] = t1[tc].apply(parse_tissue_rate)

print("\n--- 1a. Tissues with highest mean editing rate (top 20) ---")
mean_rates = rate_matrix.mean().sort_values(ascending=False)
for tissue, rate in mean_rates.head(20).items():
    n_sites = rate_matrix[tissue].notna().sum()
    n_edited = (rate_matrix[tissue] > 0).sum()
    print(f"  {tissue:50s}  mean_rate={rate:7.3f}%  n_sites_with_data={n_sites}  n_edited={n_edited}")

print("\n--- 1b. Tissues with the most sites showing editing (rate > 0) ---")
n_edited_per_tissue = (rate_matrix > 0).sum().sort_values(ascending=False)
for tissue, count in n_edited_per_tissue.head(20).items():
    total = rate_matrix[tissue].notna().sum()
    pct = 100 * count / total if total > 0 else 0
    print(f"  {tissue:50s}  n_edited={count}  of_total={total}  pct={pct:.1f}%")

print("\n--- 1c. Tissues with most sites edited at >= 1% rate ---")
n_edited_gt1 = (rate_matrix >= 1.0).sum().sort_values(ascending=False)
for tissue, count in n_edited_gt1.head(20).items():
    total = rate_matrix[tissue].notna().sum()
    pct = 100 * count / total if total > 0 else 0
    print(f"  {tissue:50s}  n_edited_ge1pct={count}  of_total={total}  pct={pct:.1f}%")

# 1d. Tissue modules
print("\n--- 1d. Tissue module definitions and statistics ---")
tissue_modules = {
    'Brain': [c for c in tissue_cols if 'Brain' in c],
    'Blood/Immune': [c for c in tissue_cols if any(x in c for x in ['Blood', 'Spleen', 'lymphocyte'])],
    'Cardiovascular': [c for c in tissue_cols if any(x in c for x in ['Heart', 'Artery'])],
    'GI/Digestive': [c for c in tissue_cols if any(x in c for x in ['Colon', 'Esophagus', 'Stomach', 'Small_Intestine', 'Liver', 'Pancreas'])],
    'Reproductive': [c for c in tissue_cols if any(x in c for x in ['Ovary', 'Testis', 'Uterus', 'Vagina', 'Fallopian', 'Cervix', 'Prostate', 'Breast'])],
    'Skin': [c for c in tissue_cols if 'Skin' in c],
    'Adipose': [c for c in tissue_cols if 'Adipose' in c],
    'Kidney': [c for c in tissue_cols if 'Kidney' in c],
    'Lung': [c for c in tissue_cols if c == 'Lung'],
    'Other': [c for c in tissue_cols if any(x in c for x in ['Nerve', 'Muscle', 'Pituitary', 'Thyroid', 'Adrenal', 'Bladder', 'Minor_Salivary', 'Cells_Cultured'])]
}

for module, tissues in tissue_modules.items():
    if not tissues:
        continue
    module_rates = rate_matrix[tissues].values.flatten()
    module_rates = module_rates[~np.isnan(module_rates)]
    n_edited = np.sum(module_rates > 0)
    print(f"\n  {module} ({len(tissues)} tissues):")
    print(f"    Tissues: {', '.join(tissues)}")
    print(f"    Mean rate across all sites x tissues: {np.mean(module_rates):.3f}%")
    print(f"    Median rate: {np.median(module_rates):.3f}%")
    print(f"    Max rate: {np.max(module_rates):.3f}%")
    print(f"    Sites with editing > 0: {n_edited} / {len(module_rates)}")

# 1e. Tissue correlation (mean rate per module)
print("\n--- 1e. Cross-tissue-module correlations (using per-site mean rate within module) ---")
module_site_means = pd.DataFrame()
for module, tissues in tissue_modules.items():
    if tissues:
        module_site_means[module] = rate_matrix[tissues].mean(axis=1)

corr = module_site_means.corr()
print("\nPairwise correlations between tissue module mean rates:")
print(corr.to_string(float_format=lambda x: f"{x:.3f}"))

# Top 10 most correlated tissue pairs
print("\n--- 1f. Top 20 most correlated tissue PAIRS (individual tissues) ---")
tissue_corr = rate_matrix.corr()
pairs = []
for i, t1_name in enumerate(tissue_cols):
    for j, t2_name in enumerate(tissue_cols):
        if i < j:
            r = tissue_corr.loc[t1_name, t2_name]
            if not np.isnan(r):
                pairs.append((t1_name, t2_name, r))
pairs.sort(key=lambda x: -x[2])
for t1_name, t2_name, r in pairs[:20]:
    print(f"  {t1_name:45s} <-> {t2_name:45s}  r={r:.4f}")

print("\n--- 1g. Bottom 10 least correlated tissue pairs ---")
for t1_name, t2_name, r in pairs[-10:]:
    print(f"  {t1_name:45s} <-> {t2_name:45s}  r={r:.4f}")

# 1h. Tissue classification distribution
print("\n--- 1h. Tissue classification distribution ---")
print(t1['Tissue_Classification'].value_counts().to_string())

print("\n--- 1i. Distribution of # tissues edited per site ---")
tissue_counts = t1['Edited_In_#_Tissues']
print(f"  Mean: {tissue_counts.mean():.1f}")
print(f"  Median: {tissue_counts.median():.1f}")
print(f"  Min: {tissue_counts.min()}, Max: {tissue_counts.max()}")
print(f"  Quartiles: {tissue_counts.quantile([0.25, 0.5, 0.75]).to_dict()}")
bins = [0, 1, 5, 10, 20, 30, 40, 54]
labels = ['1', '2-5', '6-10', '11-20', '21-30', '31-40', '41-54']
binned = pd.cut(tissue_counts, bins=bins, labels=labels, right=True)
print("  Binned distribution:")
print(binned.value_counts().sort_index().to_string())


# ============================================================================
# PART 2: Structure analysis
# ============================================================================
print("\n" + "=" * 80)
print("PART 2: STRUCTURE ANALYSIS")
print("=" * 80)

t3 = pd.read_csv(f'{BASE}/data/processed/advisor/supp_t3_structures.csv')
print(f"\nTotal sites with structure info: {len(t3)}")

print("\n--- 2a. Structure type distribution ---")
print(t3['Structure_Type'].value_counts().to_string())

print("\n--- 2b. Structure type vs APOBEC class ---")
ct = pd.crosstab(t3['Structure_Type'], t3['Affecting_Over_Expressed_APOBEC'], margins=True)
print(ct.to_string())

# Merge with t1 for rate info
t3_merged = t3.merge(t1[['Chr', 'Start', 'End', 'Max_GTEx_Editing_Rate', 'Mean_GTEx_Editing_Rate', 'Edited_In_#_Tissues', 'Tissue_Classification']], on=['Chr', 'Start', 'End'], how='left')

print("\n--- 2c. Editing rate distribution per structure type ---")
for stype in t3_merged['Structure_Type'].dropna().unique():
    subset = t3_merged[t3_merged['Structure_Type'] == stype]
    rates = subset['Mean_GTEx_Editing_Rate'].dropna()
    max_rates = subset['Max_GTEx_Editing_Rate'].dropna()
    n_tissues = subset['Edited_In_#_Tissues'].dropna()
    print(f"\n  {stype} (n={len(subset)}):")
    print(f"    Mean rate: mean={rates.mean():.3f}%, median={rates.median():.3f}%, std={rates.std():.3f}%")
    print(f"    Max rate:  mean={max_rates.mean():.3f}%, median={max_rates.median():.3f}%")
    print(f"    Tissues edited: mean={n_tissues.mean():.1f}, median={n_tissues.median():.1f}")

print("\n--- 2d. Structure type vs Tissue classification ---")
ct2 = pd.crosstab(t3_merged['Structure_Type'], t3_merged['Tissue_Classification'], margins=True)
print(ct2.to_string())

print("\n--- 2e. Structure concordance (mRNA vs pre-mRNA) ---")
print(f"Structure_Type_mRNA distribution:")
print(t3['Structure_Type_mRNA'].value_counts().to_string())
print(f"\nStructure_Type_pre-mRNA distribution:")
print(t3['Structure_TypePre_mRNA'].value_counts().to_string())

ct3 = pd.crosstab(t3['Structure_Type_mRNA'], t3['Structure_TypePre_mRNA'], margins=True)
print(f"\nmRNA vs pre-mRNA structure concordance:")
print(ct3.to_string())


# ============================================================================
# PART 3: Cancer / TCGA survival analysis
# ============================================================================
print("\n" + "=" * 80)
print("PART 3: CANCER / TCGA SURVIVAL ANALYSIS")
print("=" * 80)

t5 = pd.read_csv(f'{BASE}/data/processed/advisor/t5_tcga_survival.csv')
print(f"\nTotal sites: {len(t5)}")
print(f"Sites with any survival association: {t5['Cancers_with_Editing_Significantly_Associated_with_Survival'].notna().sum()}")

# Parse cancer types
t5_with_cancer = t5[t5['Cancers_with_Editing_Significantly_Associated_with_Survival'].notna()].copy()
cancer_list = []
for _, row in t5_with_cancer.iterrows():
    cancers = str(row['Cancers_with_Editing_Significantly_Associated_with_Survival']).split(';')
    for c in cancers:
        c = c.strip()
        if c and c != 'nan':
            cancer_list.append({
                'cancer_type': c,
                'gene': row['Gene_(RefSeq)'],
                'chr': row['Chr'],
                'start': row['Start'],
                'n_tissues': row['Edited_In_#_Tissues'],
                'apobec': row['Affecting_Over_Expressed_APOBEC']
            })

cancer_df = pd.DataFrame(cancer_list)
print(f"\nTotal site-cancer associations: {len(cancer_df)}")

print("\n--- 3a. Cancer types most commonly showing editing-survival associations ---")
cancer_counts = cancer_df['cancer_type'].value_counts()
print(cancer_counts.to_string())

print("\n--- 3b. Number of cancers per site ---")
n_cancers = t5['#_Cancers_with_Editing_Significantly_Associated_with_Survival'].dropna()
print(f"  Mean: {n_cancers.mean():.2f}")
print(f"  Max: {n_cancers.max():.0f}")
print(f"  Distribution:")
print(n_cancers.value_counts().sort_index().to_string())

print("\n--- 3c. Cancer associations vs APOBEC class ---")
if len(cancer_df) > 0:
    ct4 = cancer_df.groupby('cancer_type')['apobec'].value_counts().unstack(fill_value=0)
    print(ct4.to_string())

print("\n--- 3d. Cancer types cross-referenced with tissue specificity ---")
# Merge with tissue info
t5_tissue = t5.merge(t3[['Chr', 'Start', 'End', 'Tissue_Specificity']], on=['Chr', 'Start', 'End'], how='left')
t5_tissue_cancer = t5_tissue[t5_tissue['Cancers_with_Editing_Significantly_Associated_with_Survival'].notna()]
print(f"\nTissue specificity of sites with survival associations:")
print(t5_tissue_cancer['Tissue_Specificity'].value_counts().to_string())

# Map cancer types to approximate tissue
cancer_tissue_map = {
    'LIHC': 'Liver', 'KIRC': 'Kidney', 'KIRP': 'Kidney', 'KICH': 'Kidney',
    'LGG': 'Brain', 'GBM': 'Brain', 'BLCA': 'Bladder', 'BRCA': 'Breast',
    'COAD': 'Colon', 'READ': 'Colon', 'LUAD': 'Lung', 'LUSC': 'Lung',
    'PRAD': 'Prostate', 'STAD': 'Stomach', 'UCEC': 'Uterus',
    'OV': 'Ovary', 'THCA': 'Thyroid', 'SKCM': 'Skin',
    'HNSC': 'Head_Neck', 'SARC': 'Soft_Tissue', 'LAML': 'Blood',
    'ACC': 'Adrenal', 'PAAD': 'Pancreas', 'MESO': 'Mesothelium',
    'UVM': 'Eye', 'PCPG': 'Adrenal', 'UCS': 'Uterus',
    'CESC': 'Cervix', 'TGCT': 'Testis', 'THYM': 'Thymus',
    'CHOL': 'Bile_Duct', 'DLBC': 'Lymph'
}
cancer_df['tissue_origin'] = cancer_df['cancer_type'].map(cancer_tissue_map)
print("\n  Cancer type to tissue origin mapping:")
print(cancer_df['tissue_origin'].value_counts().to_string())


# ============================================================================
# PART 4: Cross-dataset coordinate and gene overlap
# ============================================================================
print("\n" + "=" * 80)
print("PART 4: CROSS-DATASET COORDINATE AND GENE OVERLAP")
print("=" * 80)

combined = pd.read_csv(f'{BASE}/data/processed/all_datasets_combined.csv')
print(f"\nTotal entries: {len(combined)}")
print(f"\nDataset source counts:")
print(combined['dataset_source'].value_counts().to_string())

# Create coordinate key
combined['coord'] = combined['chr'] + ':' + combined['start'].astype(str) + '-' + combined['end'].astype(str)

levanon = combined[combined['dataset_source'] == 'advisor_c2t']
asaoka = combined[combined['dataset_source'] == 'asaoka_2019']
alqassim = combined[combined['dataset_source'] == 'alqassim_2021']
sharma = combined[combined['dataset_source'] == 'sharma_2015']

print(f"\nLevanon sites: {len(levanon)}")
print(f"Asaoka sites: {len(asaoka)}")
print(f"Alqassim sites: {len(alqassim)}")
print(f"Sharma sites: {len(sharma)}")

# Coordinate overlaps
lev_coords = set(levanon['coord'])
asa_coords = set(asaoka['coord'])
alq_coords = set(alqassim['coord'])
sha_coords = set(sharma['coord'])

print(f"\n--- 4a. Coordinate overlaps ---")
print(f"  Levanon & Asaoka:   {len(lev_coords & asa_coords)} shared coordinates")
print(f"  Levanon & Alqassim: {len(lev_coords & alq_coords)} shared coordinates")
print(f"  Levanon & Sharma:   {len(lev_coords & sha_coords)} shared coordinates")
print(f"  Asaoka & Alqassim:  {len(asa_coords & alq_coords)} shared coordinates")
print(f"  Asaoka & Sharma:    {len(asa_coords & sha_coords)} shared coordinates")
print(f"  Alqassim & Sharma:  {len(alq_coords & sha_coords)} shared coordinates")

# Gene overlaps
lev_genes = set(levanon['gene'].dropna())
asa_genes = set(asaoka['gene'].dropna())
alq_genes = set(alqassim['gene'].dropna())
sha_genes = set(sharma['gene'].dropna())

print(f"\n--- 4b. Gene overlaps ---")
print(f"  Levanon unique genes: {len(lev_genes)}")
print(f"  Asaoka unique genes:  {len(asa_genes)}")
print(f"  Alqassim unique genes: {len(alq_genes)}")
print(f"  Sharma unique genes:  {len(sha_genes)}")
print(f"\n  Levanon & Asaoka genes:   {len(lev_genes & asa_genes)}")
print(f"  Levanon & Alqassim genes: {len(lev_genes & alq_genes)}")
print(f"  Levanon & Sharma genes:   {len(lev_genes & sha_genes)}")
print(f"  Asaoka & Alqassim genes:  {len(asa_genes & alq_genes)}")
print(f"  Asaoka & Sharma genes:    {len(asa_genes & sha_genes)}")
print(f"  Alqassim & Sharma genes:  {len(alq_genes & sha_genes)}")

# 4c. Shared coordinates - rate distributions
print(f"\n--- 4c. Rate distributions for shared vs unique sites ---")
shared_lev_asa = lev_coords & asa_coords
shared_lev_alq = lev_coords & alq_coords

# Levanon sites shared with Asaoka vs not
lev_shared_asa = levanon[levanon['coord'].isin(shared_lev_asa)]
lev_unique_asa = levanon[~levanon['coord'].isin(shared_lev_asa)]
print(f"\n  Levanon sites shared with Asaoka (n={len(lev_shared_asa)}):")
print(f"    Mean editing rate: {lev_shared_asa['editing_rate'].mean():.3f}")
print(f"    Median: {lev_shared_asa['editing_rate'].median():.3f}")
print(f"    Std: {lev_shared_asa['editing_rate'].std():.3f}")
print(f"  Levanon sites NOT in Asaoka (n={len(lev_unique_asa)}):")
print(f"    Mean editing rate: {lev_unique_asa['editing_rate'].mean():.3f}")
print(f"    Median: {lev_unique_asa['editing_rate'].median():.3f}")
print(f"    Std: {lev_unique_asa['editing_rate'].std():.3f}")

# Asaoka sites shared with Levanon vs not
asa_shared = asaoka[asaoka['coord'].isin(shared_lev_asa)]
asa_unique = asaoka[~asaoka['coord'].isin(shared_lev_asa)]
print(f"\n  Asaoka sites shared with Levanon (n={len(asa_shared)}):")
print(f"    Mean editing rate: {asa_shared['editing_rate'].mean():.3f}")
print(f"    Median: {asa_shared['editing_rate'].median():.3f}")
print(f"  Asaoka sites NOT in Levanon (n={len(asa_unique)}):")
print(f"    Mean editing rate: {asa_unique['editing_rate'].mean():.3f}")
print(f"    Median: {asa_unique['editing_rate'].median():.3f}")

# Levanon sites shared with Alqassim vs not
lev_shared_alq = levanon[levanon['coord'].isin(shared_lev_alq)]
lev_unique_alq = levanon[~levanon['coord'].isin(shared_lev_alq)]
print(f"\n  Levanon sites shared with Alqassim (n={len(lev_shared_alq)}):")
print(f"    Mean editing rate: {lev_shared_alq['editing_rate'].mean():.3f}")
print(f"    Median: {lev_shared_alq['editing_rate'].median():.3f}")
print(f"  Levanon sites NOT in Alqassim (n={len(lev_unique_alq)}):")
print(f"    Mean editing rate: {lev_unique_alq['editing_rate'].mean():.3f}")
print(f"    Median: {lev_unique_alq['editing_rate'].median():.3f}")

# 4d. Gene profiles of shared sites
print(f"\n--- 4d. Gene profiles of shared sites ---")
shared_genes_lev_asa = lev_genes & asa_genes
shared_genes_lev_alq = lev_genes & alq_genes
print(f"\n  Top genes in Levanon & Asaoka shared coordinates ({len(shared_lev_asa)} sites):")
if len(lev_shared_asa) > 0:
    print(lev_shared_asa['gene'].value_counts().head(15).to_string())

print(f"\n  Top genes in Levanon & Alqassim shared coordinates ({len(shared_lev_alq)} sites):")
if len(lev_shared_alq) > 0:
    print(lev_shared_alq['gene'].value_counts().head(15).to_string())

# 4e. Feature distribution across datasets
print(f"\n--- 4e. Feature/exonic function distribution by dataset ---")
for src in ['advisor_c2t', 'asaoka_2019', 'alqassim_2021', 'sharma_2015']:
    subset = combined[combined['dataset_source'] == src]
    print(f"\n  {src} (n={len(subset)}):")
    print(f"    Feature distribution:")
    print(subset['feature'].value_counts().to_string())
    print(f"    is_edited distribution: {subset['is_edited'].value_counts().to_dict()}")
    rates = subset['editing_rate'].dropna()
    if len(rates) > 0:
        print(f"    Editing rate: mean={rates.mean():.3f}, median={rates.median():.3f}, std={rates.std():.3f}")

# 4f. Edit type distribution
print(f"\n--- 4f. Edit type distribution by dataset ---")
for src in ['advisor_c2t', 'asaoka_2019', 'alqassim_2021', 'sharma_2015']:
    subset = combined[combined['dataset_source'] == src]
    print(f"  {src}: {subset['edit_type'].value_counts().to_dict()}")


# ============================================================================
# PART 5: Sharma anomaly analysis
# ============================================================================
print("\n" + "=" * 80)
print("PART 5: SHARMA ANOMALY - 1 COORDINATE BUT 75 GENE OVERLAP WITH LEVANON")
print("=" * 80)

print(f"\nSharma dataset: {len(sharma)} sites, {len(sha_genes)} unique genes")
print(f"Levanon dataset: {len(levanon)} sites, {len(lev_genes)} unique genes")
print(f"Shared coordinates: {len(lev_coords & sha_coords)}")
print(f"Shared genes: {len(lev_genes & sha_genes)}")

# The shared coordinate
shared_sha_lev_coords = lev_coords & sha_coords
print(f"\nShared coordinate(s): {shared_sha_lev_coords}")
if shared_sha_lev_coords:
    for coord in shared_sha_lev_coords:
        lev_site = levanon[levanon['coord'] == coord]
        sha_site = sharma[sharma['coord'] == coord]
        print(f"\n  Levanon entry at {coord}:")
        print(f"    Gene: {lev_site['gene'].values[0]}, Rate: {lev_site['editing_rate'].values[0]}, Feature: {lev_site['feature'].values[0]}")
        print(f"  Sharma entry at {coord}:")
        print(f"    Gene: {sha_site['gene'].values[0]}, Rate: {sha_site['editing_rate'].values[0]}, Feature: {sha_site['feature'].values[0]}")

# 5a. Characterize Sharma genes that overlap with Levanon
shared_genes_sha_lev = sha_genes & lev_genes
print(f"\n--- 5a. Shared genes between Sharma and Levanon ({len(shared_genes_sha_lev)}) ---")
print(f"  Genes: {sorted(shared_genes_sha_lev)}")

# For shared genes, how many sites does each dataset have in those genes?
print(f"\n--- 5b. Sites per shared gene ---")
sha_in_shared_genes = sharma[sharma['gene'].isin(shared_genes_sha_lev)]
lev_in_shared_genes = levanon[levanon['gene'].isin(shared_genes_sha_lev)]
print(f"  Sharma sites in shared genes: {len(sha_in_shared_genes)}")
print(f"  Levanon sites in shared genes: {len(lev_in_shared_genes)}")

# Show per-gene counts
gene_counts = pd.DataFrame({
    'sharma_sites': sharma[sharma['gene'].isin(shared_genes_sha_lev)].groupby('gene').size(),
    'levanon_sites': levanon[levanon['gene'].isin(shared_genes_sha_lev)].groupby('gene').size()
}).fillna(0).astype(int).sort_values('sharma_sites', ascending=False)
print(f"\n  Per-gene site counts (top 20):")
print(gene_counts.head(20).to_string())

# 5c. Why different coordinates? Different positions within same genes
print(f"\n--- 5c. Sharma characterization ---")
print(f"  Sharma chromosome distribution:")
print(sharma['chr'].value_counts().head(10).to_string())
print(f"\n  Levanon chromosome distribution:")
print(levanon['chr'].value_counts().head(10).to_string())

print(f"\n  Sharma feature distribution:")
print(sharma['feature'].value_counts().to_string())
print(f"\n  Levanon feature distribution:")
print(levanon['feature'].value_counts().to_string())

# Sharma editing rates vs Levanon
sha_rates = sharma['editing_rate'].dropna()
lev_rates = levanon['editing_rate'].dropna()
print(f"\n  Sharma editing rates: mean={sha_rates.mean():.3f}, median={sha_rates.median():.3f}, std={sha_rates.std():.3f}, n={len(sha_rates)}")
print(f"  Levanon editing rates: mean={lev_rates.mean():.3f}, median={lev_rates.median():.3f}, std={lev_rates.std():.3f}, n={len(lev_rates)}")

# 5d. Are Sharma sites in different positions within the same genes?
print(f"\n--- 5d. Position analysis for shared genes ---")
for gene in sorted(shared_genes_sha_lev)[:10]:
    sha_gene = sharma[sharma['gene'] == gene]
    lev_gene = levanon[levanon['gene'] == gene]
    print(f"\n  Gene {gene}:")
    print(f"    Sharma positions:  {sorted(sha_gene['start'].values)}")
    print(f"    Levanon positions: {sorted(lev_gene['start'].values)}")


# ============================================================================
# PART 6: Conservation patterns from editing_sites_labels.csv
# ============================================================================
print("\n" + "=" * 80)
print("PART 6: CONSERVATION PATTERNS")
print("=" * 80)

labels = pd.read_csv(f'{BASE}/data/processed/editing_sites_labels.csv')
print(f"\nTotal sites with labels: {len(labels)}")

print("\n--- 6a. Conservation level distribution ---")
print(labels['conservation_level'].value_counts().sort_index().to_string())

print("\n--- 6b. Conservation vs editing rate ---")
for level in sorted(labels['conservation_level'].unique()):
    subset = labels[labels['conservation_level'] == level]
    rates = subset['mean_gtex_rate'].dropna()
    max_rates = subset['max_gtex_rate'].dropna()
    print(f"\n  Conservation level {level} (n={len(subset)}):")
    print(f"    Mean GTEx rate: mean={rates.mean():.3f}, median={rates.median():.3f}")
    print(f"    Max GTEx rate:  mean={max_rates.mean():.3f}, median={max_rates.median():.3f}")

print("\n--- 6c. Conservation vs tissue breadth ---")
for level in sorted(labels['conservation_level'].unique()):
    subset = labels[labels['conservation_level'] == level]
    n_tissues = subset['n_tissues_edited'].dropna()
    print(f"  Level {level}: mean_n_tissues={n_tissues.mean():.1f}, median={n_tissues.median():.1f}")

print("\n--- 6d. Conservation vs structure type ---")
ct5 = pd.crosstab(labels['conservation_level'], labels['structure_type'], margins=True)
print(ct5.to_string())

print("\n--- 6e. Conservation vs APOBEC class ---")
ct6 = pd.crosstab(labels['conservation_level'], labels['apobec_class'], margins=True)
print(ct6.to_string())

print("\n--- 6f. Conservation vs tissue class ---")
ct7 = pd.crosstab(labels['conservation_level'], labels['tissue_class'], margins=True)
print(ct7.to_string())

print("\n--- 6g. Conservation vs genomic category ---")
ct8 = pd.crosstab(labels['conservation_level'], labels['genomic_category'], margins=True)
print(ct8.to_string())

print("\n--- 6h. Conservation vs has survival association ---")
ct9 = pd.crosstab(labels['conservation_level'], labels['has_survival_association'], margins=True)
print(ct9.to_string())

# Summary: most conserved sites
print("\n--- 6i. Most conserved sites (conservation_level >= 1) characterization ---")
conserved = labels[labels['conservation_level'] >= 1]
not_conserved = labels[labels['conservation_level'] == 0]
print(f"\n  Conserved sites (n={len(conserved)}):")
print(f"    Mean GTEx rate: {conserved['mean_gtex_rate'].mean():.3f}")
print(f"    Mean max rate: {conserved['max_gtex_rate'].mean():.3f}")
print(f"    Mean tissues: {conserved['n_tissues_edited'].mean():.1f}")
print(f"    Tissue class distribution: {conserved['tissue_class'].value_counts().to_dict()}")
print(f"    Structure type: {conserved['structure_type'].value_counts().to_dict()}")
print(f"    APOBEC class: {conserved['apobec_class'].value_counts().to_dict()}")

print(f"\n  Non-conserved sites (n={len(not_conserved)}):")
print(f"    Mean GTEx rate: {not_conserved['mean_gtex_rate'].mean():.3f}")
print(f"    Mean max rate: {not_conserved['max_gtex_rate'].mean():.3f}")
print(f"    Mean tissues: {not_conserved['n_tissues_edited'].mean():.1f}")
print(f"    Tissue class distribution: {not_conserved['tissue_class'].value_counts().to_dict()}")
print(f"    Structure type: {not_conserved['structure_type'].value_counts().to_dict()}")
print(f"    APOBEC class: {not_conserved['apobec_class'].value_counts().to_dict()}")

print("\n--- 6j. HEK293 rates vs conservation ---")
for level in sorted(labels['conservation_level'].unique()):
    subset = labels[labels['conservation_level'] == level]
    hek = subset['hek293_rate'].dropna()
    print(f"  Level {level}: n_with_hek293={len(hek)}, mean={hek.mean():.3f}" if len(hek) > 0 else f"  Level {level}: no HEK293 data")


# ============================================================================
# PART 7: Additional cross-dataset generalization analysis
# ============================================================================
print("\n" + "=" * 80)
print("PART 7: CROSS-DATASET GENERALIZATION POTENTIAL")
print("=" * 80)

print("\n--- 7a. Rate correlation for shared sites ---")
# For Levanon & Asaoka shared coordinates, compare rates
if len(shared_lev_asa) > 0:
    lev_rates_shared = levanon[levanon['coord'].isin(shared_lev_asa)][['coord', 'editing_rate']].set_index('coord')
    asa_rates_shared = asaoka[asaoka['coord'].isin(shared_lev_asa)][['coord', 'editing_rate']].set_index('coord')
    merged_rates = lev_rates_shared.join(asa_rates_shared, lsuffix='_levanon', rsuffix='_asaoka')
    if len(merged_rates.dropna()) > 1:
        corr_val = merged_rates['editing_rate_levanon'].corr(merged_rates['editing_rate_asaoka'])
        print(f"  Levanon vs Asaoka rate correlation (n={len(merged_rates.dropna())}): r={corr_val:.4f}")
        print(f"  Levanon mean: {merged_rates['editing_rate_levanon'].mean():.3f}")
        print(f"  Asaoka mean: {merged_rates['editing_rate_asaoka'].mean():.3f}")

print("\n--- 7b. Gene-level rate comparison ---")
lev_gene_rates = levanon.groupby('gene')['editing_rate'].mean()
asa_gene_rates = asaoka.groupby('gene')['editing_rate'].mean()
alq_gene_rates = alqassim.groupby('gene')['editing_rate'].mean()
sha_gene_rates = sharma.groupby('gene')['editing_rate'].mean()

# Common genes
for name, other_rates, other_name in [
    ('Asaoka', asa_gene_rates, 'asaoka'),
    ('Alqassim', alq_gene_rates, 'alqassim'),
    ('Sharma', sha_gene_rates, 'sharma')
]:
    common = lev_gene_rates.index.intersection(other_rates.index)
    if len(common) > 1:
        r = lev_gene_rates[common].corr(other_rates[common])
        print(f"  Levanon vs {name} gene-level rate correlation (n_genes={len(common)}): r={r:.4f}")

print("\n--- 7c. Dataset feature composition comparison ---")
for src in ['advisor_c2t', 'asaoka_2019', 'alqassim_2021', 'sharma_2015']:
    subset = combined[combined['dataset_source'] == src]
    total = len(subset)
    edited = subset['is_edited'].sum()
    print(f"\n  {src}:")
    print(f"    Total sites: {total}, Edited: {edited} ({100*edited/total:.1f}%), Not edited: {total-edited} ({100*(total-edited)/total:.1f}%)")
    print(f"    Rate stats (edited only): mean={subset[subset['is_edited']==1]['editing_rate'].mean():.3f}, median={subset[subset['is_edited']==1]['editing_rate'].median():.3f}")

print("\n--- 7d. Unique sites per dataset (not in any other) ---")
all_coords = {'advisor_c2t': lev_coords, 'asaoka_2019': asa_coords, 'alqassim_2021': alq_coords, 'sharma_2015': sha_coords}
for src, coords in all_coords.items():
    other_coords = set()
    for other_src, other in all_coords.items():
        if other_src != src:
            other_coords |= other
    unique = coords - other_coords
    print(f"  {src}: {len(unique)} sites unique (not in any other dataset), {len(coords - unique)} shared")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
