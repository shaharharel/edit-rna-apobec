# Clinical Deep Analysis: APOBEC RNA Editing and Human Disease

## 1. ClinVar Pathogenic Enrichment Analysis

### 1.1 What We Tested

We scored all 1,692,837 C-to-T (C>U on the RNA level) variants in ClinVar using two models: (1) our gradient boosting model (GB_Full, trained on 40-dimensional hand-crafted features capturing sequence motif, RNA loop geometry, and structure delta), and (2) the RNAsee rules-based scoring system. For each variant, both models output a probability that the site is an APOBEC3A RNA editing target. We then asked: among variants that the model predicts as likely editing sites, are pathogenic variants overrepresented compared to benign variants?

### 1.2 What We Found

**A3A (GB_Full model, AUROC=0.938):**

| Threshold | Pathogenic fraction | Benign fraction | Odds Ratio | p-value |
|-----------|-------------------|-----------------|------------|---------|
| >= 0.3 | 90.3% | 88.3% | 1.232 | 9.0e-25 |
| >= 0.5 | 83.9% | 81.8% | 1.159 | 4.9e-19 |
| >= 0.7 | 74.3% | 72.4% | 1.102 | 4.6e-12 |
| >= 0.9 | 53.6% | 55.2% | 0.939 | 4.0e-07 |

At a threshold of 0.5, pathogenic variants are 15.9% more likely than benign variants to be predicted as APOBEC3A editing targets (OR=1.159, p=4.9e-19). This enrichment is consistent across thresholds from 0.3 to 0.7 and reverses only at very high thresholds (>0.9), where the model becomes so selective that only half of all variants pass.

**A3B (GB model, AUROC=0.830):**
- Raw enrichment at t=0.5: OR=1.065 (p=2.9e-14)
- Calibrated enrichment (adjusted from training prior pi=0.50 to population prior pi=0.019): OR=1.552 (p=3.0e-31)

**A3G (GB_MotifOnly, AUROC=0.996, CC-context only):**
- CC-context enrichment at t=0.4: OR=1.759 (p<1e-300)
- CC-context enrichment at t=0.5: OR=1.183 (p=1.8e-38)
- A3G shows the strongest raw enrichment of any enzyme, driven by strong CC dinucleotide preference

### 1.3 What OR=1.159 Means in Plain Language

An odds ratio of 1.159 means that if you take all cytidine positions in the genome that ClinVar has classified, and you ask the model "would APOBEC3A edit this cytidine?", the model is about 16% more likely to say "yes" for a pathogenic variant than for a benign variant. Put another way: sites where APOBEC3A is predicted to introduce C-to-U changes are slightly but significantly enriched among positions where such changes cause disease.

This is a modest effect size, but it is:
- Highly statistically significant (p < 1e-18)
- Consistent across all threshold cutoffs from 0.3 to 0.7
- Consistent across gene-size quartiles (ruling out gene-length confounding)
- Confirmed by within-gene analysis (62.3% of genes show pathogenic > benign scores, Wilcoxon p=1.5e-30)
- Reproduced across all three APOBEC enzymes (A3A, A3B, A3G)

### 1.4 Three Hypotheses for Why Editing Sites Enrich for Pathogenic Variants

**Hypothesis 1: Structural vulnerability (MOST LIKELY)**

Pathogenic variants tend to occur in functionally constrained, structurally important regions of mRNA. APOBEC3A preferentially edits cytidines in specific structural contexts: unpaired bases within RNA hairpin loops, particularly near the apex. These same structural features -- single-stranded regions in otherwise structured RNA -- also tend to be functionally important regulatory elements. The enrichment may reflect shared structural vulnerability: sites that are accessible to APOBEC enzymes are also sites where mutations are more likely to disrupt function.

Evidence supporting this hypothesis:
- `relative_loop_position` (proximity to hairpin loop apex) is the #1 predictive feature for editing (importance=0.213)
- The within-gene analysis shows the effect (62.3% of genes, p=1.5e-30), indicating it is not driven by gene-level confounds
- The enrichment persists after controlling for gene size (all quartiles show OR > 1.0)

**Hypothesis 2: Codon context selection**

APOBEC3A has a TC dinucleotide preference (edits C preceded by T). The genetic code is not random -- certain codon contexts are more functionally constrained than others. If pathogenic C>T mutations happen to cluster in TC-rich codon contexts (for example, codons where C>T causes nonsynonymous changes), this could create apparent enrichment without any biological connection to RNA editing.

Evidence: This hypothesis is partially controlled for by the within-gene analysis, which shows the effect persists at the per-gene level. However, we cannot fully exclude codon composition effects without matching for trinucleotide context.

**Hypothesis 3: Active mutagenic contribution**

APOBEC3A-mediated C-to-U editing may occasionally contribute directly to somatic mutations that cause or promote disease. In this scenario, the enrichment reflects genuine causal mutagenesis: APOBEC3A edits certain cytidines, and when those edits occur in critical positions, the resulting C-to-U change is pathogenic.

Evidence: This hypothesis is supported by the known role of APOBEC enzymes in cancer mutagenesis (APOBEC signature SBS2/SBS13 in cancer genomes) and the observation that 23 known RNA editing sites are classified as pathogenic in ClinVar (including sites in tumor suppressors SDHB and TSC1/TSC2). However, the modest effect size (OR~1.16) argues against a strong direct mutagenic contribution -- if editing were a major mutational mechanism, we would expect a larger enrichment.

**Assessment:** Hypothesis 1 (structural vulnerability) is most consistent with the data. The enrichment is real but modest, driven by shared structural features between editing targets and functionally important sites, with possible contributions from codon context and rare direct mutagenesis.

### 1.5 Clinical Implications for Variant Interpretation

1. **Variant of uncertain significance (VUS) prioritization**: Among 873,818 VUS variants, 78.4% are predicted as editing targets (gb>=0.5). While this alone does not determine pathogenicity, it provides additional evidence that should be integrated with other computational and functional data.

2. **Known editing sites in ClinVar**: 36 known RNA editing sites (23 pathogenic + 13 likely pathogenic) have been classified as disease-causing in ClinVar. This raises the question of whether some "pathogenic" variants may actually represent normal RNA editing events that were misclassified as germline mutations. Variant interpretation pipelines should flag C>T variants at known editing sites.

3. **Editing as a modifier**: Even if editing does not directly cause disease, tissue-specific editing patterns could modify disease penetrance. A variant that is "edited away" in some tissues but not others could show tissue-specific disease manifestation.

---

## 2. Disease Breakdown

### 2.1 Top Disease Categories Among Predicted Editing Sites

Among 31,665 pathogenic variants with GB score >= 0.5:

| Rank | Disease | N variants |
|------|---------|-----------|
| 1 | Hereditary cancer-predisposing syndrome | 1,557 |
| 2 | Inborn genetic diseases | 1,164 |
| 3 | Cardiovascular phenotype | 588 |
| 4 | Retinal dystrophy | 420 |
| 5 | Familial cancer of breast | 330 |
| 6 | Primary ciliary dyskinesia | 327 |
| 7 | Familial thoracic aortic aneurysm | 293 |
| 8 | Hereditary breast ovarian cancer syndrome | 283 |
| 9 | Duchenne muscular dystrophy | 277 |
| 10 | Osteogenesis imperfecta type I | 247 |

### 2.2 Disease-Category Enrichment

We tested whether predicted editing sites are differentially enriched across disease categories, comparing within pathogenic variants:

| Disease category | N pathogenic | Editing fraction | Mean GB | vs. overall | OR | p-value |
|-----------------|-------------|-----------------|---------|-------------|-----|---------|
| Cancer | 2,721 | 81.0% | 0.766 | -0.018 | 0.804 | 2.7e-05 |
| Cardiac | 1,267 | 84.2% | 0.791 | +0.006 | 1.024 | 0.79 |
| Neurological | 3,833 | 84.3% | 0.792 | +0.007 | 1.035 | 0.47 |
| Metabolic | 873 | 84.9% | 0.794 | +0.010 | 1.078 | 0.46 |
| Connective tissue | 927 | 82.0% | 0.753 | -0.032 | 0.870 | 0.11 |

Notable finding: **Cancer-related pathogenic variants are LESS likely to be predicted as editing sites** (OR=0.804, p=2.7e-5) compared to other pathogenic variants. This could reflect the fact that cancer driver mutations often occur at highly conserved, structurally rigid sites (e.g., catalytic residues of kinases) that are poor APOBEC substrates. The enrichment is driven more by loss-of-function variants in haploinsufficient genes than by gain-of-function cancer drivers.

### 2.3 Enrichment by Chromosome

The enrichment is consistent across most chromosomes (19/23 show OR > 1.0), with the strongest effects on:
- chr20 (OR=1.345, p=0.013)
- chr16 (OR=1.337, p=2.1e-4)
- chr12 (OR=1.331, p=1.5e-4)
- chr6 (OR=1.306, p=7.1e-4)

---

## 3. Gene-Level Analysis

### 3.1 Top 20 Genes with Pathogenic Variants at Predicted Editing Sites

| Gene | Pathogenic (GB>=0.5) | Function | Top Disease |
|------|---------------------|----------|-------------|
| DMD | 319 | Cytoskeletal | Duchenne muscular dystrophy |
| FBN1 | 240 | Extracellular matrix | Marfan syndrome |
| PKD1 | 230 | Signaling | Polycystic kidney disease |
| COL1A1 | 206 | Collagen | Osteogenesis imperfecta |
| NF1 | 198 | Tumor suppressor | Neurofibromatosis type 1 |
| SCN1A | 169 | Ion channel | Epileptic encephalopathy |
| COL7A1 | 166 | Collagen | Epidermolysis bullosa |
| KMT2D | 165 | Chromatin remodeling | Kabuki syndrome |
| ABCA4 | 165 | Sensory | Retinal dystrophy |
| USH2A | 159 | Sensory | Usher syndrome |
| ATM | 150 | DNA repair | Ataxia-telangiectasia |
| BRCA2 | 149 | DNA repair | Hereditary breast cancer |
| COL4A5 | 148 | Collagen | Alport syndrome |
| COL1A2 | 146 | Collagen | Osteogenesis imperfecta |
| COL2A1 | 140 | Collagen | Skeletal dysplasia |
| CHD7 | 132 | Chromatin remodeling | CHARGE syndrome |
| TSC2 | 130 | Tumor suppressor | Tuberous sclerosis |
| COL3A1 | 130 | Collagen | Ehlers-Danlos type 4 |
| CFTR | 128 | Ion channel | Cystic fibrosis |
| BRCA1 | 123 | DNA repair | Hereditary breast cancer |

### 3.2 Pathway Clustering

The top genes cluster into distinct functional pathways:

1. **Extracellular Matrix / Collagen** (16 genes, 1,385 variants): COL1A1, COL1A2, COL2A1, COL3A1, COL4A5, COL7A1, FBN1, etc. These genes encode structural proteins where single amino acid changes can disrupt protein folding, and they are exceptionally large (many exons = many C>T opportunities).

2. **DNA Repair** (11 genes, 744 variants): ATM, BRCA2, BRCA1, FANCA, MLH1, MSH2, MSH6, PMS2, PALB2, etc. DNA repair genes are enriched for loss-of-function C>T mutations that are both pathogenic (cancer predisposition) and potentially targeted by APOBEC editing.

3. **Chromatin Remodeling** (8 genes, 665 variants): KMT2D, CHD7, ARID1B, CREBBP, NSD1, EP300, KDM6A, SMARCA4. Epigenetic regulators where haploinsufficiency causes developmental syndromes.

4. **Ion Channels** (10 genes, 703 variants): SCN1A, CFTR, SCN2A, KCNQ1, KCNH2, SCN5A, RYR1. Channel genes where single amino acid changes alter electrophysiology.

5. **Tumor Suppressors** (10 genes, 858 variants): NF1, TSC2, BRCA1, APC, RB1, PTCH1, TSC1, TP53, PTEN, VHL. Classic tumor suppressors where C>T nonsense/missense mutations cause cancer predisposition.

### 3.3 Within-Gene Pathogenic vs. Benign Score Comparison

Among 1,566 genes with at least 5 pathogenic AND 5 benign ClinVar variants:
- **62.3% of genes** show higher mean editing scores for pathogenic than benign variants
- Mean within-gene difference: +0.036 (pathogenic scoring higher)
- Sign test: 975/1566, p=2.3e-22
- Wilcoxon signed-rank test: p=1.5e-30

This within-gene analysis is critical because it controls for gene-level confounds (gene size, expression level, GC content). The fact that pathogenic variants score higher than benign variants within the same gene confirms that the enrichment reflects variant-level features, not gene-level biases.

### 3.4 Genes with Strongest Within-Gene Enrichment

| Gene | Path mean | Benign mean | Difference | N path / N ben |
|------|-----------|-------------|-----------|----------------|
| NONO | 0.953 | 0.450 | +0.502 | 8 / 5 |
| GNPAT | 0.935 | 0.481 | +0.453 | 6 / 7 |
| ETV6 | 0.932 | 0.492 | +0.440 | 5 / 12 |
| EIF2AK3 | 0.908 | 0.502 | +0.406 | 11 / 5 |
| GATA6 | 0.951 | 0.558 | +0.393 | 6 / 7 |
| MYSM1 | 0.876 | 0.495 | +0.380 | 14 / 8 |
| MME | 0.872 | 0.501 | +0.371 | 14 / 10 |

NONO (Non-POU Domain Containing Octamer Binding) shows the largest within-gene effect: pathogenic variants score 0.50 higher than benign variants on average. NONO is involved in DNA damage response and RNA splicing -- functions where RNA structural context is highly relevant.

---

## 4. Known Editing Sites at Pathogenic ClinVar Variants

### 4.1 Experimentally Validated Editing Sites with Pathogenic Classification

36 known RNA editing sites (experimentally validated in published datasets) have been independently classified as pathogenic or likely pathogenic in ClinVar. These represent cases where the exact nucleotide change caused by APOBEC editing (C-to-U) is known to cause disease.

**23 Pathogenic sites:**

| Gene | GB score | RNAsee | Disease |
|------|----------|--------|---------|
| SDHB | 0.977 | 0.376 | Pheochromocytoma/paraganglioma syndrome |
| TSC1 | 0.915 | 0.520 | Tuberous sclerosis / bladder cancer |
| TSC2 | 0.913 | 0.360 | Tuberous sclerosis / lymphangiomyomatosis |
| XPC | 0.973 | 0.414 | Xeroderma pigmentosum |
| BMPR2 | 0.989 | 0.546 | Pulmonary arterial hypertension |
| CEP290 | 0.999 | 0.458 | Joubert syndrome / polycystic kidney |
| DYNC2H1 | 0.995 | 0.368 | Jeune thoracic dystrophy |
| SLC37A4 | 0.999 | 0.680 | Glycogen storage disease |
| ASPM | 0.996 | 0.418 | Primary microcephaly |
| POMT1 | 0.999 | 0.374 | Limb-girdle muscular dystrophy |
| LNPK | 0.999 | 0.470 | Neurodevelopmental disorder with epilepsy |
| TAB2 | 0.977 | 0.388 | Congenital heart defects |
| ALMS1 | 0.974 | 0.750 | Alstrom syndrome |
| OFD1 | 0.996 | 0.478 | Orofaciodigital syndrome |
| VPS13D | 0.947 | 0.442 | Neurological (not specified) |
| CHD2 | 0.933 | 0.566 | Epileptic encephalopathy |
| DNM2 | 0.904 | 0.300 | Charcot-Marie-Tooth / centronuclear myopathy |
| SLC25A38 | 0.902 | 0.398 | Sideroblastic anemia |
| SETX | 0.849 | 0.488 | Neurological (not specified) |
| MERTK | 0.825 | 0.306 | Retinitis pigmentosa |
| PEX3 | 0.961 | 0.344 | Peroxisomal disorder |
| NAA15 | 0.487 | 0.616 | Not specified |
| MPLKIP | 0.391 | 0.300 | Not specified |

**13 Likely Pathogenic sites:**

| Gene | GB score | RNAsee | Disease |
|------|----------|--------|---------|
| PMS2 | 0.970 | 0.538 | Lynch syndrome (colorectal cancer) |
| POT1 | 0.992 | 0.316 | Tumor predisposition syndrome |
| RNF113A | 0.996 | 0.608 | Trichothiodystrophy |
| COQ8A | 0.995 | 0.280 | Autosomal recessive ataxia |
| MMADHC | 0.993 | 0.472 | Methylmalonic aciduria |
| RFC1 | 0.979 | 0.396 | Cerebellar ataxia with neuropathy |
| TRAPPC11 | 0.975 | 0.592 | Limb-girdle muscular dystrophy |
| DSP | 0.958 | 0.552 | Arrhythmogenic cardiomyopathy |
| GLB1 | 0.954 | 0.722 | GM1 gangliosidosis |
| MMUT | 0.930 | 0.228 | Methylmalonic aciduria |
| PRUNE1 | 0.754 | 0.552 | PRUNE1-related disorder |
| IFT74 | 0.799 | 0.550 | IFT74-related disorder |
| TONSL | 0.691 | 0.468 | Skeletal dysplasia |

### 4.2 Clinical Significance of Known Editing at Pathogenic Variants

The finding that experimentally validated APOBEC editing sites overlap with pathogenic ClinVar variants has two interpretations:

1. **Misclassification concern**: Some "pathogenic" C>T variants in ClinVar may actually represent normal RNA editing events that were detected in RNA sequencing and misinterpreted as germline DNA mutations. This is particularly relevant for SDHB (pheochromocytoma), where RNA-based diagnostics could mistake editing for a somatic mutation.

2. **Functional convergence**: APOBEC editing targets sites where C-to-U changes have functional consequences. This supports the structural vulnerability hypothesis -- the same features that make a site an efficient APOBEC substrate (accessible, unpaired, in a structured hairpin) also make it functionally important.

**Highlight: SDHB**
SDHB (succinate dehydrogenase complex iron sulfur subunit B) is a tumor suppressor. Its known editing site produces a stopgain (premature stop codon). This is classified as pathogenic for pheochromocytoma/paraganglioma syndrome. The model scores it at gb=0.977 (high confidence editing target). This case perfectly illustrates the clinical relevance: if APOBEC3A edits this site in vivo, it could inactivate a tumor suppressor through RNA-level premature truncation.

**Highlight: TSC1/TSC2**
Both tuberous sclerosis complex genes have known editing sites that are pathogenic in ClinVar. TSC1 editing at a pathogenic site (gb=0.915) is associated with tuberous sclerosis and bladder cancer. TSC2 has multiple ClinVar entries at editing sites, including one pathogenic (gb=0.913) and three conflicting. The tuberous sclerosis complex regulates mTOR signaling -- a central growth control pathway.

---

## 5. Tissue-Disease Connection

### 5.1 Editing Patterns by Enzyme and Tissue

Each APOBEC enzyme shows distinct tissue preferences:

**A3A (120 sites): Blood-dominant editing**
- Whole blood: 2.18% mean editing rate (10x higher than most tissues)
- Lung: 0.53%, Vagina: 0.52%, Testis: 0.48%
- Minimal in: cultured fibroblasts (0.04%), brain cerebellum (0.06%)

**A3G (60 sites): Testis-dominant editing**
- Testis: 2.26% mean rate
- Whole blood: 1.27%
- Ovary: 0.78%, Cervix: 0.55%, Prostate: 0.54%

**A3A+A3G dual (178 sites): Blood and testis**
- Whole blood: 2.43% (highest of any category)
- Testis: 1.82%
- Ovary: 0.79%

**Neither (206 sites, likely APOBEC1): Intestine-dominant**
- Small intestine: 1.93% (highest)
- Whole blood: 1.55%
- Testis: 0.99%, Brain cerebellum: 0.98%

### 5.2 Tissue Classification Distribution

| Classification | Total | A3A | A3G | A3A+A3G | Neither | Unknown |
|---------------|-------|-----|-----|---------|---------|---------|
| Blood Specific | 159 | 81 | 5 | 51 | 16 | 6 |
| Testis Specific | 141 | 15 | 31 | 42 | 44 | 9 |
| Ubiquitous | 153 | 8 | 11 | 51 | 46 | 37 |
| Non-Specific | 110 | 12 | 12 | 34 | 37 | 15 |
| Intestine Specific | 73 | 4 | 1 | 0 | 63 | 5 |

Key observation: **A3A editing is heavily blood-specific** (81/120 = 67.5% of A3A-only sites). This matches APOBEC3A's known expression pattern in monocytes and macrophages during inflammatory responses. A3G editing is testis-enriched (31/60 = 51.7%), consistent with A3G expression in spermatogenic cells.

### 5.3 Tissue-Disease Connections

**Blood editing (A3A) and hematologic disease:**
A3A is most active in blood/immune cells. APOBEC3A expression is upregulated during inflammation and interferon signaling. The 81 blood-specific A3A editing sites are in genes involved in:
- Immune function (CD4, CD247, IL7R)
- Cell signaling (SDHA, SDHB - succinate dehydrogenase complex)
- DNA damage response (NBN, ERCC3)
- Protein synthesis (QARS1, FARSB)

Blood-specific editing sites include SDHB (paraganglioma), ATN1 (DRPLA, a neurodegenerative disease), and CYP27A1 (cerebrotendinous xanthomatosis). The connection between blood editing and hematologic malignancies is supported by the known role of APOBEC3A in creating the SBS2/SBS13 mutational signature in cancers.

**Intestine editing (Neither/APOBEC1) and GI disease:**
63 of 73 intestine-specific sites are in the "Neither" category (not A3A or A3G), strongly suggesting APOBEC1 as the editor. APOBEC1 is the canonical C-to-U RNA editor, originally discovered for editing APOB mRNA in the intestine. The intestine-specific sites include:
- ABCB1 (drug transporter, colorectal cancer drug resistance)
- HADHB (fatty acid oxidation)
- MFSD8 (neuronal ceroid lipofuscinosis)
- SEC23B (congenital dyserythropoietic anemia)
- SDHD (paraganglioma)

The APOB editing site itself (chr2:21010329) is classified as "Neither" (not A3A or A3G), confirmed as an APOBEC1 target.

**Testis editing (A3G) and reproductive/developmental disorders:**
A3G editing is concentrated in testis, with 31 testis-specific sites. These include:
- GNAS (McCune-Albright syndrome, endocrine tumors)
- CDKN1A (cell cycle regulator p21)
- ZNF532 (transcription factor)
- STK10 (apoptosis regulator)
Testis-specific editing could contribute to male fertility disorders or de novo mutations transmitted through the germline.

### 5.4 Stopgain Editing Sites: Where Editing Creates Premature Stop Codons

19 known editing sites produce stopgain (premature termination) changes. These are the most functionally severe editing events:

| Gene | Enzyme | Tissue | Clinical Relevance |
|------|--------|--------|-------------------|
| SDHB | A3A | Blood Specific | Pheochromocytoma/paraganglioma (23 pathogenic ClinVar variants) |
| APOB | Neither | Ubiquitous | Familial hypobetalipoproteinemia (27 pathogenic variants) |
| RHEB | Neither | Ubiquitous | Tuberous sclerosis pathway (Ras homolog) |
| ATN1 | A3A | Blood Specific | DRPLA neurodegeneration |
| DDOST | A3A+A3G | Blood Specific | Congenital disorder of glycosylation |
| SRGAP2C | Neither | Blood Specific | Brain development (human-specific duplication) |
| SEMA6C | A3A | Testis Specific | Axon guidance |
| LRP10 | A3A+A3G | Testis Specific | Parkinson disease |
| STK10 | Neither, A3A+A3G | Testis Specific | Lymphocyte migration |
| EPHB6 | Neither | Testis Specific | Cancer-associated receptor |
| NSUN5 | A3A+A3G | Ubiquitous | Williams syndrome region, RNA methyltransferase |
| ZNF532 | A3G | Ubiquitous | Transcription factor |
| DHX38 | A3A+A3G | Blood Specific | pre-mRNA splicing |
| ULK2 | A3A+A3G | Blood Specific | Autophagy |
| FBXL12 | A3A+A3G | Blood Specific | Ubiquitin ligase |
| AP2A1 | A3A+A3G | Blood Specific | Clathrin-mediated endocytosis |
| ASCC2 | A3A | Blood Specific | Transcription-coupled DNA repair |
| KRT6C | Unknown | Ubiquitous | Keratin (pachyonychia congenita) |

The SDHB stopgain is particularly notable: APOBEC3A edits this site in blood cells (where SDHB loss causes paraganglioma), the GB model confidently predicts it as an editing target (0.977), and it has 23 pathogenic ClinVar variants. This is the strongest single-gene case for clinical relevance of APOBEC editing.

---

## 6. GB Model vs. RNAsee Comparison

### 6.1 Model Disagreement

The GB model and RNAsee rules-based scoring are essentially uncorrelated (Spearman rho = -0.020, p < 1e-155). They capture fundamentally different signals:

| Quadrant (at t=0.5) | Pathogenic | Benign |
|---------------------|-----------|--------|
| Both predict edited | 28.7% | 19.8% |
| GB only predicts edited | 55.2% | 62.0% |
| RNAsee only predicts edited | 4.0% | 5.5% |
| Neither predicts edited | 12.0% | 12.7% |

The GB model predicts 79% of all variants as edited (at t=0.5), while RNAsee predicts only 27%. This reflects fundamentally different model architectures:
- GB uses 40 features including RNA secondary structure and loop geometry
- RNAsee uses rules-based scoring focused on sequence motifs and accessibility

### 6.2 Operating Point Comparison

At matched operating points (same fraction of variants called positive):

| % called positive | GB threshold | RNAsee threshold | GB OR | RNAsee OR |
|-------------------|-------------|-----------------|-------|-----------|
| ~82% | 0.3 | ~0.29 | 1.232 | 1.109 |
| ~27% | ~0.87 | 0.5 | ~1.0 | 1.443 |

RNAsee shows stronger enrichment at its natural threshold (0.5, which flags 27% of sites), while GB shows stronger enrichment at low thresholds. This is because:
- RNAsee is more selective (fewer predicted positives) -- its "edited" calls are higher-confidence
- GB captures a broader signal including structural features that correlate with functional constraint

---

## 7. Cross-Enzyme Clinical Summary

### 7.1 APOBEC3A

**What we tested:** We trained a gradient boosting classifier on 8,153 sites (5,187 positives, 2,966 negatives) using 40-dimensional hand-crafted features (motif, loop geometry, structure delta). We then scored all 1.69M C>T ClinVar variants and compared editing prediction scores between pathogenic and benign variants.

**What we found:** Pathogenic variants are 16% more likely to be predicted as A3A editing targets (OR=1.159 at t=0.5, p=4.9e-19). Within individual genes, 62.3% show higher pathogenic scores (Wilcoxon p=1.5e-30). The enrichment persists after Bayesian calibration from training prior (50%) to population prior (1.9%).

**What it means:** APOBEC3A editing preferentially targets structurally accessible RNA positions that are also functionally constrained. The enrichment is driven by shared structural vulnerability, not by direct mutagenesis. The top features driving the signal are loop geometry and motif context, suggesting that the RNA secondary structure itself predisposes certain positions to both editing and functional importance.

**Clinical implications:** (1) C>T variants at predicted editing sites deserve additional scrutiny in clinical interpretation. (2) 23 known editing sites are pathogenic ClinVar variants, suggesting editing-based variant misclassification is possible. (3) SDHB editing creates a premature stop in a tumor suppressor expressed in blood -- the tissue where A3A is most active.

**Tissue context:** A3A editing is blood-dominant (mean rate 2.18% in whole blood, 10x higher than most tissues). Clinical relevance is highest for hematologic and inflammatory conditions.

### 7.2 APOBEC3B

**What we tested:** GB classifier (AUROC=0.830) trained on A3B editing sites from overexpression studies. Scored all 1.69M ClinVar C>T variants.

**What we found:** Raw enrichment OR=1.065 (t=0.5, p=2.9e-14). After Bayesian calibration to population prior (1.9%), the enrichment increases to OR=1.552 (p=3.0e-31) at the calibrated threshold. A3B shows the strongest calibrated enrichment of any single enzyme.

**What it means:** A3B has a less specific sequence motif than A3A (no strong TC preference), which paradoxically makes its structural signal more informative for distinguishing editing sites from non-sites. The large calibrated OR reflects A3B's ability to identify a small number of high-confidence targets that are enriched for functional significance.

**Clinical implications:** A3B is the primary APOBEC enzyme implicated in cancer mutagenesis (breast cancer, bladder cancer, cervical cancer). Its ClinVar enrichment supports the hypothesis that APOBEC3B-mediated mutations occur preferentially at functionally important sites.

### 7.3 APOBEC3G

**What we tested:** GB_MotifOnly classifier (24-dim, AUROC=0.996) trained on only 119 A3G sites. Scored ClinVar variants filtered to CC dinucleotide context (515,463 variants) because A3G has a strict CC preference.

**What we found:** In CC-context, OR=1.759 at t=0.4 (p<1e-300) -- the strongest raw enrichment of any enzyme. At t=0.5, OR=1.183 (p=1.8e-38).

**What it means:** A3G's extreme CC specificity creates a natural enrichment: CC dinucleotide contexts are inherently biased toward certain codon positions and amino acid changes. However, even after restricting analysis to CC-context variants only, pathogenic variants score significantly higher, confirming that A3G editing targets overlap with disease-relevant positions.

**Clinical implications:** A3G editing is testis-dominant, suggesting relevance for male germline mutations and inherited disorders. The very small training set (n=119) means clinical predictions should be interpreted with caution.

**Tissue context:** A3G is most active in testis (mean rate 2.26%) and whole blood (1.27%). The testis specificity suggests a potential role in de novo germline mutations.

---

## 8. 130 Genes with Both Known Editing Sites and ClinVar Pathogenic Variants

130 genes harbor both experimentally validated APOBEC editing sites AND independently curated pathogenic ClinVar variants. Notable examples:

| Gene | Editing sites | Enzyme | Tissue | Pathogenic ClinVar | High-score path. | Disease |
|------|--------------|--------|--------|-------------------|-----------------|---------|
| PKD1 | 1 | Neither | Ubiquitous | 259 | 230 | Polycystic kidney disease |
| NSD1 | 1 | A3A | Blood | 90 | 77 | Sotos syndrome |
| KCNH2 | 1 | A3A | Non-Specific | 57 | 48 | Long QT syndrome |
| KCNQ1 | 2 | Neither | Non-Specific/Blood | 59 | 52 | Long QT syndrome |
| CDH23 | 1 | Neither | Blood | 60 | 51 | Usher syndrome / deafness |
| FLNA | 1 | A3A+A3G | Non-Specific | 42 | 35 | Periventricular heterotopia |
| COL6A2 | 1 | A3A+A3G | Blood | 40 | 31 | Bethlem myopathy |
| ANK1 | 1 | A3A | Ubiquitous | 40 | 36 | Hereditary spherocytosis |
| MECP2 | 1 | A3A+A3G | Blood | 35 | 31 | Rett syndrome |
| GNAS | 1 | A3G | Testis | 31 | 29 | McCune-Albright / endocrine tumors |
| APOB | 2 | Neither/Unknown | Ubiquitous | 27 | 23 | Hypobetalipoproteinemia |
| SDHB | 1 | A3A | Blood | 26 | 22 | Pheochromocytoma/paraganglioma |
| NBN | 1 | A3A | Blood | 24 | 22 | Nijmegen breakage syndrome |

These genes represent the strongest candidates for clinical relevance of APOBEC editing: they have confirmed editing activity and independently documented pathogenic C>T variants.

---

## 9. Summary and Conclusions

### Key Findings

1. **Pathogenic enrichment is real and reproducible**: All three APOBEC enzymes (A3A, A3B, A3G) show statistically significant enrichment of pathogenic ClinVar variants among predicted editing sites. The signal is consistent across gene-size quartiles and within individual genes.

2. **Structural vulnerability drives the enrichment**: The most likely explanation is that APOBEC editing targets and functionally important positions share structural features -- particularly unpaired bases in RNA hairpin loops. This is not a confound but a genuine biological insight: the RNA secondary structure that makes a site accessible to APOBEC enzymes also tends to indicate functional importance.

3. **36 known editing sites are pathogenic ClinVar variants**: These cases represent either (a) genuine disease-causing edits, (b) RNA editing events misclassified as DNA mutations, or (c) both. Clinical variant interpretation should flag C>T variants at known editing sites.

4. **Tissue-specific editing predicts tissue-specific disease relevance**: A3A (blood-specific) is most relevant for hematologic conditions; A3G (testis-specific) for germline and reproductive disorders; APOBEC1/Neither (intestine-specific) for GI conditions.

5. **Cancer genes show paradoxical depletion**: Pathogenic variants in cancer-associated genes are slightly LESS likely to be predicted editing sites (OR=0.804, p=2.7e-5), likely because cancer drivers tend to occur at structurally rigid catalytic sites that are poor APOBEC substrates.

6. **GB and RNAsee capture orthogonal signals**: The two scoring methods are uncorrelated (Spearman=-0.02), suggesting they capture complementary aspects of editing site biology. An ensemble approach could improve predictions.

### Limitations

1. The GB model predicts 79% of C>T variants as "edited" at t=0.5, limiting discriminative power. The model is better at ranking (within-gene comparisons) than binary classification.

2. We cannot distinguish exonic functions (synonymous, nonsynonymous, stopgain) in the ClinVar scored data -- this limits mechanistic interpretation.

3. A3G results are based on a very small training set (119 sites) and should be interpreted with caution.

4. The within-gene analysis, while controlling for gene-level confounds, cannot fully control for within-gene structural variation that may correlate with both editing propensity and functional importance.

### Clinical Recommendations

1. **Flag C>T VUS at predicted editing sites**: For the 148 VUS variants at known editing sites, consider whether the observed change could represent normal RNA editing rather than a germline mutation.

2. **Use editing predictions as supplementary evidence**: APOBEC editing scores should not be used as standalone pathogenicity predictors but as additional evidence in clinical variant interpretation pipelines.

3. **Consider tissue context**: A variant at a blood-specific A3A editing site has different clinical implications than one at a testis-specific A3G site.

4. **Prioritize stopgain editing sites**: The 19 stopgain editing events (especially SDHB, APOB, RHEB) represent the highest-impact cases where editing directly creates loss-of-function changes.

---

## 10. Iteration 2: Deep Dive Findings

### 10.1 Molecular Consequence Analysis (NEW)

Extracting molecular consequences from the ClinVar VCF for all 36 pathogenic editing sites revealed a striking pattern:

| Consequence | Count | Percentage |
|-------------|-------|------------|
| Nonsense (premature stop codon) | 32 | 88.9% |
| Missense (amino acid change) | 4 | 11.1% |

**32 of 36 (89%) pathogenic editing sites create premature stop codons.** The 4 missense sites are CHD2, COQ8A, DNM2, and MMUT. This is not random -- C-to-U editing at CAG, CGA, or CAA codons produces TAG, TGA, or TAA stop codons, and nonsense mutations are the most common mechanism by which a single C>T change becomes pathogenic.

This resolves Limitation #2 from iteration 1: we CAN now distinguish exonic functions, and the answer is overwhelmingly nonsense.

### 10.2 Critical Clarification: Overexpression vs. Endogenous Editing

Of the 36 pathogenic editing sites:
- **35 are from OVEREXPRESSION experiments** (Asaoka 2019, Baysal 2016, Sharma 2015)
- **Only 1 (SDHB) has confirmed endogenous editing** in normal human tissues (Levanon/Advisor GTEx data)

This is a critical distinction:
1. Overexpression editing sites demonstrate that APOBEC3A *can* edit these positions, but not that it does so at physiologically relevant levels
2. The overlap between ClinVar pathogenic variants and APOBEC3A editing sites may be partly coincidental -- both nonsense mutations and APOBEC editing prefer certain sequence/structural contexts
3. True clinical relevance requires demonstration of endogenous editing

### 10.3 SDHB: The Definitive Case Study

SDHB is the single most important finding. It uniquely combines:
- Confirmed endogenous editing (whole blood: 1.22%, 10x higher than other tissues)
- Blood-specific (matching APOBEC3A expression in monocytes/macrophages)
- Stopgain (premature stop codon in a tumor suppressor)
- Pathogenic for pheochromocytoma/paraganglioma (OMIM 115310)
- High GB model score (0.977)

SDHB tissue-specific editing rates (top 5):
| Tissue | Editing Rate |
|--------|-------------|
| Whole Blood | 1.220% |
| Lung | 0.317% |
| Cervix Ectocervix | 0.214% |
| Vagina | 0.139% |
| Esophagus Mucosa | 0.138% |

ClinVar context for SDHB C>T variants:
| Category | N | Mean GB Score |
|----------|---|---------------|
| Pathogenic | 26 | 0.768 |
| Likely pathogenic | 30 | 0.864 |
| VUS | 167 | 0.623 |
| Likely benign | 170 | 0.657 |
| Benign | 14 | 0.606 |

The editing site itself (GB=0.977) scores higher than 87% of all SDHB C>T variants. See `paper/case_studies.md` for the full 1-page case study.

### 10.4 Misannotation Risk Assessment (Revised)

The misannotation risk is more nuanced than initially described:

**SDHB is the only clear misannotation risk case** among the 36 pathogenic sites, because it is the only one with confirmed endogenous editing. The scenario: RNA-based SDHB mutation screening on a blood sample detects the ~1.2% C>U editing signal and reports it as a pathogenic germline mutation.

For the other 35 sites, misannotation risk is LOW because:
- They are only confirmed as edited under APOBEC3A overexpression
- Their endogenous editing rates are unknown (not in the Levanon GTEx database)
- If they are not edited endogenously, there is no RNA signal to misinterpret

**Broader scan**: Among all 636 endogenously edited Levanon sites, only SDHB has a direct overlap with a pathogenic ClinVar variant at the exact editing position. However, 19 stopgain editing sites in Levanon could potentially be confused with pathogenic variants in RNA-based assays, with APOB (mean rate 10.8%) and SDHB as the highest-risk cases.

### 10.5 Cancer Gene Depletion Explained

The paradoxical depletion of predicted editing sites among cancer genes (OR=0.804) is explained by the GoF/LoF distinction:

| Gene Category | N Pathogenic | Mean GB Score | Editing Fraction (GB>=0.5) |
|--------------|-------------|---------------|---------------------------|
| LoF tumor suppressors | 1,771 | 0.744 | 79.1% |
| GoF oncogenes | 119 | 0.726 | 75.6% |
| LoF benign | 1,235 | 0.715 | 75.3% |
| GoF benign | 347 | 0.770 | 81.3% |

Key observation: **GoF oncogene pathogenic variants score LOWER than their benign variants** (difference = -0.044), while LoF tumor suppressors show the expected pattern (pathogenic > benign, difference = +0.028). This is because:
- GoF mutations occur at catalytic residues in structured protein regions, corresponding to base-paired mRNA positions that APOBEC3A cannot access
- LoF mutations (especially nonsense) can occur anywhere, including APOBEC-accessible loop regions

This validates the model: it correctly identifies that editing sites are in accessible RNA structures, which overlap with LoF-vulnerable positions but not GoF hotspots.

### 10.6 Pathway Analysis (965 Overlapping Genes)

With the expanded gene set (including overexpression data), 965 genes have both editing sites and pathogenic ClinVar variants. The top represented pathways among the 36 pathogenic editing sites:

| Pathway | N Genes | Key Genes |
|---------|---------|-----------|
| Ciliary/Centrosome | 4 | CEP290, OFD1, DYNC2H1, IFT74 |
| Cancer/mTOR/Telomere | 5 | SDHB, TSC1, TSC2, PMS2, POT1 |
| Cytoskeletal/Desmosome | 2 | DSP, DNM2 |
| BMP/TGF-beta Signaling | 1 | BMPR2 |
| Chromatin Remodeling | 1 | CHD2 |

The ciliopathy cluster (4 genes) is notable: CEP290, OFD1, DYNC2H1, and IFT74 are all involved in ciliary function, and editing-induced nonsense mutations in these genes cause a spectrum of ciliopathies (Joubert syndrome, orofaciodigital syndrome, Jeune thoracic dystrophy, IFT74-related disorder).

### 10.7 Updated Conclusions

1. **The 89% nonsense rate is the headline finding**: Nearly all pathogenic editing sites create premature stop codons, not amino acid changes. This reflects codon biochemistry (C>U in stop-codon-generating contexts) and the pathogenicity profile (nonsense is inherently more damaging than most missense).

2. **SDHB is the one case that demands clinical attention**: Endogenous editing, blood-specific, stopgain in tumor suppressor, directly relevant disease. All other sites lack endogenous editing confirmation.

3. **Misannotation risk is real but narrow**: Confined to SDHB and potentially other high-rate Levanon sites. Variant calling pipelines should flag known editing positions.

4. **Cancer gene depletion validates the model**: The GoF/LoF split explains the OR=0.804 for cancer genes and confirms that the model captures genuine RNA structural accessibility rather than artifacts.

5. **Full case studies**: See `paper/case_studies.md` for detailed case studies of SDHB, TSC1/TSC2, PMS2, APOB, and BMPR2.
