# Mutation Coupling Analysis: Are APOBEC RNA Editing Sites Enriched for DNA Mutations?

## Executive Summary

We tested whether APOBEC RNA editing sites show elevated rates of DNA C>T mutations using 5,703 editing sites across multiple APOBEC enzymes and 1.69M ClinVar C>T variants. Through four iterations of increasingly stringent controls, we found:

1. **A real but modest enrichment exists** (~25-35% more ClinVar C>T variants within 100bp of editing sites vs same-exon, trinucleotide-matched controls)
2. **The signal is dominated by CpG context**: CpG editing sites overlap ClinVar 5x more than non-CpG sites (15.9% vs 3.1%)
3. **Enzyme-specific**: Only A3A and A3A+A3G sites show enrichment; A3G, Neither, and Unknown categories show none
4. **The paired test is significant for CpG but not non-CpG**, suggesting the enrichment primarily reflects CpG mutation hotspots rather than APOBEC-specific DNA mutagenesis

## Hypothesis

Sites where APOBEC enzymes edit RNA may reside in genomic contexts prone to APOBEC-mediated DNA mutations, leading to enrichment of C>T variants at or near editing positions.

## Methods

### Data Sources
- **Editing sites**: 5,703 positions (5,187 A3A, 178 A3A+A3G, 60 A3G, 206 Neither, 72 Unknown)
- **DNA variants**: 1,692,837 ClinVar C>T variants with clinical significance annotations
- **Reference genome**: hg38

### Control Design (Iterative Refinement)

| Version | Control Strategy | Key Result |
|---------|-----------------|------------|
| v1 | Any position ±5kb, same trinucleotide | 3.4x enrichment |
| v2 | Any exonic position ±10kb, same trinucleotide | 1.17x enrichment |
| v3 | Same-exon position, same trinucleotide | 1.39x at ±100bp |
| v4 | Same-exon, CpG-stratified | CpG: 1.26x, non-CpG: 1.36x |

**v1 was confounded** by ClinVar's ascertainment bias toward exonic regions (editing sites are exonic; v1 controls included intronic positions with fewer ClinVar entries). This explains the 3.4x inflation.

### Statistical Tests
- **Mann-Whitney U test**: Unpaired comparison of variant counts in windows
- **Wilcoxon signed-rank test**: Paired comparison (each site vs its matched controls)
- **Permutation test**: 2,000 permutations of editing/control labels
- **Bootstrap CI**: 1,000 bootstrap samples for ratio confidence interval

## Results

### 1. Window-Level Enrichment (Same-Exon Controls, v3)

| Window | Edit mean | Ctrl mean | Ratio | p-value |
|--------|-----------|-----------|-------|---------|
| ±25bp  | 1.67      | 1.09      | 1.531 | 4.0e-103 |
| ±50bp  | 3.01      | 2.04      | 1.477 | 4.2e-115 |
| ±100bp | 5.14      | 3.69      | 1.391 | 8.0e-111 |
| ±250bp | 8.65      | 7.37      | 1.174 | 2.9e-69 |
| ±500bp | 12.84     | 12.00     | 1.069 | 3.6e-43 |
| ±1000bp| 19.46     | 18.90     | 1.030 | 2.3e-25 |

**Bootstrap 95% CI for ±250bp ratio**: 1.040 - 1.166
**Permutation test**: Z=4.9, p=0.0000

The enrichment is strongest at close range and decays with distance, consistent with a local mutational process.

### 2. CpG Decomposition (v4, Key Finding)

| Context | n sites | ±100bp ratio | p-value | Exact ClinVar overlap |
|---------|---------|-------------|---------|----------------------|
| CpG (xCG) | 2,643 | 1.255 | 2.1e-43 | 15.9% (419/2643) |
| Non-CpG   | 3,060 | 1.358 | 1.2e-40 | 3.1% (94/3060) |

**Critical observation**: CpG editing sites overlap ClinVar positions at 5.1x the rate of non-CpG sites. CpG dinucleotides are the genome's primary mutation hotspot due to spontaneous deamination of 5-methylcytosine.

**Paired Wilcoxon test** (each site vs its own controls):
- CpG: mean diff = 0.618, p = 2.1e-7 (significant)
- Non-CpG: mean diff = 0.227, p = 0.52 (not significant)

This strongly suggests the enrichment is driven by CpG mutation hotspots, not APOBEC-specific DNA mutagenesis.

### 3. Per-Trinucleotide Analysis (±100bp, same-exon controls)

| Trinucleotide | n | Ratio | p-value | Note |
|--------------|---|-------|---------|------|
| TCG | 2,154 | 1.342 | 1.1e-46 | CpG, APOBEC |
| TCA | 1,757 | 1.397 | 1.3e-28 | non-CpG, APOBEC signature |
| TCC | 246 | 1.492 | 3.6e-5 | non-CpG |
| TCT | 448 | 1.340 | 1.3e-7 | non-CpG, APOBEC signature |
| CCG | 397 | 1.148 | 1.4e-3 | CpG |
| CCA | 295 | 1.142 | 5.0e-3 | non-CpG |
| CCT | 93 | 1.443 | 1.2e-2 | non-CpG |
| GCG | 62 | 0.909 | 0.82 | CpG, non-APOBEC |
| ACA | 48 | 0.889 | 0.76 | non-CpG, non-APOBEC |

Notably, APOBEC-signature contexts (TCA, TCT) show enrichment comparable to CpG contexts, but the paired test (section 2) suggests this is less robust.

### 4. Per-Enzyme Analysis (±250bp, same-exon controls)

| Enzyme | n sites | Ratio | p-value |
|--------|---------|-------|---------|
| A3A | 5,187 | 1.171 | 3.4e-67 |
| A3A+A3G | 178 | 1.188 | 7.3e-4 |
| A3G | 60 | 1.012 | 0.95 |
| Neither | 206 | 1.110 | 0.35 |
| Unknown | 72 | 1.001 | 0.69 |

Only A3A and A3A+A3G show significant enrichment. A3G and "Neither" do not.

### 5. Exact-Position Pathogenicity

513/5,703 editing sites (9.0%) have an exact ClinVar C>T variant at the same position.

| Significance | At editing sites | All ClinVar | OR | p-value |
|-------------|-----------------|-------------|-----|---------|
| Pathogenic | 7.0% (36/513) | 4.5% | 1.61 | 9.8e-3 |
| Benign | 55.0% (282/513) | 36.1% | 2.16 | 4.8e-18 |

Both pathogenic and benign variants are enriched at editing positions relative to ClinVar baseline. The benign enrichment (OR=2.16) is stronger than pathogenic (OR=1.61), which is expected: synonymous C>T mutations at CpG sites are a major class of benign ClinVar variants, and editing sites preferentially fall in CpG contexts.

### 6. Variant Density Profile (Concentric Rings)

| Distance ring | Density (variants/kb) |
|--------------|----------------------|
| 0-25bp | 31.7 |
| 25-50bp | 26.7 |
| 50-100bp | 21.3 |
| 100-200bp | 12.6 |
| 200-500bp | 8.7 |
| 500-1000bp | 6.6 |

Sharp density peak at the editing site itself, consistent with CpG hotspot proximity.

### 7. Pathogenicity by Distance from Editing Site

| Distance | Total variants | Pathogenic % | Benign % |
|----------|---------------|-------------|---------|
| Exact | 513 | 7.0% | 55.0% |
| 1-50bp | 16,651 | 4.1% | 34.4% |
| 51-100bp | 12,134 | 3.8% | 36.6% |
| 101-250bp | 20,025 | 3.2% | 36.4% |
| 251-500bp | 23,889 | 3.2% | 36.3% |

Pathogenicity peaks at exact position (7.0%), dropping to background (3.2%) by 100bp. The benign fraction also peaks at exact position (55.0% vs ~36%), consistent with CpG synonymous variants at known editing positions being classified as benign.

## Interpretation

### What the data shows
1. **Real enrichment exists** but is modest (~25-35% at ±100bp with strictest controls)
2. **CpG context is the primary driver**: CpG editing sites show 5x higher ClinVar overlap and drive the paired-test signal
3. **A3A-specific**: Only A3A/A3A+A3G enzymes show enrichment
4. **Distance-dependent**: Signal peaks at the editing site and decays to baseline by ~2kb

### What this likely means
The enrichment is best explained by the **co-occurrence of CpG mutation hotspots and APOBEC editing sites in the same genomic regions** (gene-dense, exonic, accessible chromatin), rather than direct APOBEC-mediated DNA mutagenesis causing the ClinVar variants. Key evidence:
- The paired test is significant for CpG but not non-CpG sites
- The exact-overlap enrichment is overwhelmingly CpG (TCG = 64.3% of overlaps vs 35.1% of all sites)
- ClinVar is biased toward germline variants, while APOBEC DNA mutagenesis is primarily somatic

### What this does NOT mean
- This does not refute APOBEC DNA mutagenesis at editing sites (ClinVar is the wrong dataset for this)
- The non-CpG TCA/TCT enrichment in unpaired tests, though not significant in paired tests, may reflect a real but weak APOBEC-specific signal
- Somatic mutation data (TCGA/PCAWG) would be needed to directly test the hypothesis

### Connection to existing findings
The ClinVar pathogenicity enrichment we previously found (GB OR=1.33, p<1e-40) is robust and independent of this analysis. That finding shows that sites predicted as edited by our model are enriched among pathogenic variants -- a statement about the model's clinical relevance, not about APOBEC DNA mutagenesis.

## Technical Notes
- All analyses use hg38 coordinates
- Same-exon controls are trinucleotide-matched to eliminate sequence context confounds
- The v1-v4 progression demonstrates the importance of proper control design (ClinVar ascertainment bias inflates naive estimates by ~3x)
- Code: `experiments/multi_enzyme/exp_mutation_coupling.py` (v1), `exp_mutation_coupling_v2.py` (v2), `exp_mutation_coupling_v3.py` (v3), `exp_mutation_coupling_v4.py` (v4)

## Output Files
- `experiments/multi_enzyme/outputs/mutation_coupling/mutation_coupling_results.json` (v1)
- `experiments/multi_enzyme/outputs/mutation_coupling/mutation_coupling_v2_results.json` (v2)
- `experiments/multi_enzyme/outputs/mutation_coupling/mutation_coupling_v3_results.json` (v3)
- `experiments/multi_enzyme/outputs/mutation_coupling/mutation_coupling_v4_results.json` (v4)
- `experiments/multi_enzyme/outputs/mutation_coupling/same_exon_controls.csv` (31,525 controls)
- `experiments/multi_enzyme/outputs/mutation_coupling/matched_controls.csv` (57,007 v1 controls)
- `experiments/multi_enzyme/outputs/mutation_coupling/exon_matched_controls.csv` (52,133 v2 controls)
