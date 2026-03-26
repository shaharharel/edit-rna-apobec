# Germline Mutation Coupling at APOBEC RNA Editing Sites

## Key Question
Do APOBEC RNA editing sites show elevated germline (heritable) mutation rates?
If APOBEC enzymes are active in germline tissues (testis, ovary), their enzymatic
activity on DNA could cause permanent, heritable C>T mutations.

## Method
- Counted ClinVar C>T variants within +/-250bp of each Levanon/Advisor editing site (636 sites)
- Compared to 10 trinucleotide-context-matched control sites per editing site (systematically scanned +/-50kb)
- ClinVar is predominantly germline: pathogenic/likely-pathogenic variants reflect Mendelian disease; benign/likely-benign reflect population polymorphisms
- Stratified by enzyme category (A3A, A3G, A3A_A3G, Neither, Unknown) and tissue type
- Statistical tests: Wilcoxon signed-rank (paired editing vs control), Mann-Whitney U (group comparisons), Fisher exact (context fractions), permutation test (1000 shuffles)

## Results

### A. ClinVar Variant Density: Editing Sites vs Matched Controls

All enzyme categories show highly significant enrichment of ClinVar C>T variants near editing sites compared to trinucleotide-matched controls in the same genomic neighborhood.

| Enzyme | N sites | Edit mean | Ctrl mean | Fold change | p-value |
|--------|---------|-----------|-----------|-------------|---------|
| A3A | 120 | 10.45 | 1.59 | 6.6x | 5.16e-12 |
| A3G | 60 | 4.33 | 1.46 | 3.0x | 2.75e-02 |
| A3A_A3G | 178 | 9.63 | 2.48 | 3.9x | 1.88e-11 |
| Neither | 206 | 5.76 | 1.60 | 3.6x | 6.18e-10 |
| Unknown | 72 | 8.40 | 1.74 | 4.8x | 3.27e-07 |

### B. Pathogenic Variant Density (Germline Disease Mutations)

Editing sites also show elevated pathogenic (disease-causing germline) ClinVar variants, though the effect is smaller and noisier due to low counts.

| Enzyme | N sites | Edit mean | Ctrl mean | Fold change | p-value |
|--------|---------|-----------|-----------|-------------|---------|
| A3A | 120 | 0.492 | 0.057 | 8.7x | 2.23e-03 |
| A3G | 60 | 0.167 | 0.077 | 2.2x | 7.50e-01 (ns) |
| A3A_A3G | 178 | 0.292 | 0.121 | 2.4x | 4.50e-02 |
| Neither | 206 | 0.252 | 0.065 | 3.9x | 2.43e-02 |
| Unknown | 72 | 0.167 | 0.025 | 6.7x | 8.52e-02 |

### C. By Tissue Classification

Blood-specific editing sites show the highest ClinVar density (12.08 variants/site), while intestine-specific show the lowest (1.10). This pattern reflects ClinVar ascertainment bias: blood diseases (hemoglobinopathies, coagulopathies) are heavily represented in ClinVar, while intestinal-specific genes are underrepresented.

| Tissue | N sites | Edit mean | Ctrl mean | Path mean |
|--------|---------|-----------|-----------|-----------|
| Testis Specific | 141 | 4.75 | 1.70 | 0.121 |
| Blood Specific | 159 | 12.08 | 2.06 | 0.522 |
| Ubiquitous | 153 | 8.14 | 2.34 | 0.261 |
| Intestine Specific | 73 | 1.10 | 0.49 | 0.000 |
| Non-Specific | 110 | 10.04 | 1.93 | 0.409 |

### D. Testis-Edited vs Non-Testis Sites

Testis-edited sites (rate > 1%) do NOT show elevated germline variant density. In fact, they show LOWER ClinVar density (6.56 vs 8.84, p=0.003). This is the opposite of the A3G-testis hypothesis prediction.

### E. Germline Tissue Editing Rate Correlation

No significant correlation between germline tissue (testis+ovary) editing rate and nearby ClinVar variant density:
- Spearman r = -0.065, p = 0.198 (total variants)
- Spearman r = -0.056, p = 0.269 (pathogenic variants)

### F. Context-Specific Analysis

Neither CC>CT (A3G signature) nor TC>TT (A3A signature) mutations are enriched at editing sites relative to controls:
- CC>CT at A3G sites: OR=0.655, p=0.13 (trend toward DEPLETION)
- TC>TT at A3A sites: OR=0.805, p=0.054 (trend toward DEPLETION)

### G. A3G-Testis Deep Dive

A3G testis-specific sites (31 sites) vs A3G other-tissue sites (29 sites):
- Testis-specific: 3.06 nearby ClinVar variants (mean)
- Other tissues: 5.69 nearby ClinVar variants (mean)
- Mann-Whitney U p = 0.33 (not significant)
- No evidence for elevated germline mutations at testis-edited A3G sites

### H. Permutation Test

Shuffling tissue labels 1000 times confirms the testis-specific signal is not significant:
- A3G: observed diff = -2.63, permutation p = 0.218
- A3A: observed diff = -7.68, permutation p = 0.120
- A3A_A3G: observed diff = -3.01, permutation p = 0.406

## Interpretation

### Primary Finding: Editing Sites Are in ClinVar-Dense Regions

All editing sites show 3-7x elevated ClinVar variant density compared to local trinucleotide-matched controls. This is a real and robust signal (p < 0.05 for all 5 enzyme categories), but its interpretation is nuanced:

1. **ClinVar ascertainment bias**: Editing sites are predominantly in coding exons of well-characterized disease genes (APOB, FLNA, PKD1, CDH23, etc.). These genes are systematically enriched in ClinVar because they cause Mendelian diseases that prompt genetic testing. The +/-50kb matched controls partially overlap these genes but also extend into intergenic/intronic regions with lower ClinVar coverage.

2. **Functional constraint**: Editing sites are at evolutionarily conserved cytidines in coding sequences. These positions are inherently more likely to harbor pathogenic variants because nonsynonymous changes at conserved residues are more likely to be clinically ascertained.

3. **Not specific to germline tissues**: The enrichment is seen for ALL enzyme categories regardless of tissue specificity. Blood-specific sites show the highest density, not testis-specific sites. This argues against a germline-APOBEC-mutagenesis mechanism.

### The A3G-Testis Hypothesis: Not Supported

The hypothesis that A3G's testis-specific editing activity causes heritable germline mutations finds no support:
- A3G testis-specific sites show LOWER, not higher, ClinVar density than other A3G sites
- No CC>CT enrichment at A3G sites (trend toward depletion: OR=0.655)
- No correlation between testis editing rate and nearby germline variant density
- Permutation test confirms: p=0.218

### Why the Hypothesis May Fail

Several biological reasons could explain the negative result:
1. **A3G edits RNA, not DNA**: APOBEC3G is primarily an RNA editor in this context; its DNA-editing activity (known in HIV restriction) may not operate on host genomic DNA in germline cells
2. **Editing is post-transcriptional**: RNA editing occurs on mature mRNA, not DNA. Even if A3G is present in testis, it may not access genomic DNA
3. **Selection against germline damage**: If APOBEC enzymes DID cause germline DNA mutations, strong purifying selection would eliminate such sites from the genome. The sites we observe today are precisely those where APOBEC acts on RNA without damaging DNA
4. **ClinVar is biased toward somatic-like diseases**: ClinVar's ascertainment favors diseases diagnosed through clinical genetics panels, which may not capture subtle fertility/germline effects

### What the ClinVar Enrichment DOES Mean

The 3-7x ClinVar enrichment at editing sites reflects that:
- APOBEC editing targets are in functionally important, well-characterized coding genes
- These genes are under strong selection, so variants at nearby positions are clinically significant
- This is consistent with APOBEC editing having biological function at these specific cytidines

## Data Files
- Per-site data: `experiments/multi_enzyme/outputs/germline_mutation_coupling/per_site_germline_variants.csv`
- Figure: `experiments/multi_enzyme/outputs/germline_mutation_coupling/germline_mutation_coupling.png`
- Full script: `experiments/multi_enzyme/exp_germline_mutation_coupling.py`
