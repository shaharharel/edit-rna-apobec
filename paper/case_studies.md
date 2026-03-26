# Case Studies: APOBEC RNA Editing at Pathogenic ClinVar Variants

## Overview

Among 36 experimentally validated APOBEC3A editing sites that overlap with pathogenic ClinVar variants, we performed deep molecular characterization using the ClinVar VCF annotations. The key finding: **32 of 36 (89%) create premature stop codons (nonsense mutations)**. Only 4 are missense. This is not random -- C-to-U editing at CAG, CGA, or CAA codons produces TAG, TGA, or TAA stop codons, and nonsense mutations are the most common mechanism by which a single C>T change becomes pathogenic.

Critical clarification: 35 of these 36 sites were identified in APOBEC3A **overexpression** experiments (Asaoka 2019, Baysal 2016). Only one (SDHB) is confirmed as endogenously edited in normal human tissues. The other 35 demonstrate that APOBEC3A *can* edit these positions, but whether it does so at physiologically relevant levels in vivo remains unknown.

---

## Case Study 1: SDHB -- Tumor Suppressor Stopgain in Blood

**The single strongest case for clinical relevance of APOBEC RNA editing.**

### Gene and Disease

SDHB (Succinate Dehydrogenase Complex Iron Sulfur Subunit B) encodes a subunit of mitochondrial complex II. It is a classic two-hit tumor suppressor: germline loss-of-function mutations cause hereditary paraganglioma/pheochromocytoma syndrome type 4 (OMIM 115310). Paragangliomas arise from chromaffin cells in the adrenal medulla and sympathetic ganglia.

### The Editing Event

- **Position**: chr1:17,044,824 (GRCh38, minus strand)
- **Molecular consequence**: Nonsense (premature stop codon)
- **Enzyme**: APOBEC3A Only
- **Tissue**: Blood Specific (whole blood: 1.22%, all other tissues < 0.32%)
- **GB model score**: 0.977 (high confidence editing target)
- **RNAsee score**: 0.376

### Tissue-Disease Connection

SDHB editing is overwhelmingly blood-specific, with whole blood showing 10x higher editing rates than any other tissue (1.22% vs 0.32% in lung, the next highest). This matches APOBEC3A's known expression pattern in monocytes and macrophages during inflammatory responses.

The disease tissue -- chromaffin cells of the adrenal medulla -- is not directly the same as blood. However, pheochromocytomas arise in the highly vascularized adrenal gland, where blood-borne monocytes/macrophages are abundant in the tumor microenvironment. Whether editing of SDHB mRNA in these cells contributes to tumor susceptibility is an open question.

### ClinVar Context

SDHB has 434 C>T variants in ClinVar:
- 26 Pathogenic (mean GB=0.768)
- 30 Likely pathogenic (mean GB=0.864)
- 167 VUS (mean GB=0.623)
- 170 Likely benign (mean GB=0.657)
- 14 Benign (mean GB=0.606)

The known editing site scores in the top 13th percentile of all SDHB variants (GB=0.977), higher than the average pathogenic variant (0.768). Within SDHB, pathogenic variants score significantly higher than benign variants (+0.162 mean difference), consistent with the genome-wide pattern.

### Misannotation Risk

SDHB presents the clearest case for clinical misannotation:

1. A patient with suspected paraganglioma undergoes genetic testing
2. If RNA-seq or cDNA-based SDHB sequencing is used on a blood sample, the ~1.2% editing rate at this position would produce C>T reads
3. This could be reported as a pathogenic germline SDHB mutation (it IS classified as pathogenic in ClinVar)
4. The patient would receive genetic counseling for hereditary paraganglioma syndrome
5. But the "mutation" is actually normal RNA editing

**Mitigation**: Always confirm RNA-detected SDHB variants with DNA-based sequencing. Maintain a database of known RNA editing sites to flag in variant calling pipelines.

### Why This Matters

SDHB editing demonstrates that APOBEC3A can create functionally devastating changes (stopgain in a tumor suppressor) at an endogenous editing site in the exact tissue (blood/immune cells) where the enzyme is most active. Even at 1.2% editing rate, in a cell population of millions, a fraction of cells would have SDHB mRNA truncated. Whether this contributes to tumorigenesis -- perhaps as a "second hit" in cells already carrying a germline SDHB mutation -- is a testable hypothesis.

---

## Case Study 2: TSC1 and TSC2 -- mTOR Pathway Dual Targeting

### Gene and Disease

TSC1 (hamartin) and TSC2 (tuberin) form a complex that inhibits mTOR signaling. Loss of either gene causes Tuberous Sclerosis Complex (TSC), characterized by benign tumors (hamartomas) in multiple organs including brain, kidney, skin, heart, and lung. TSC affects ~1 in 6,000 births.

### The Editing Events

- **TSC1**: chr9:132,927,247 (minus strand), nonsense, GB=0.915
  - ClinVar: Pathogenic for tuberous sclerosis, bladder cancer
  - A3A overexpression rate: 31.4%

- **TSC2**: chr16:2,080,178 (plus strand), nonsense, GB=0.913
  - ClinVar: Pathogenic for tuberous sclerosis, hereditary cancer-predisposing syndrome
  - A3A overexpression rate: 17.2%

### Significance

Both components of the TSC1/TSC2 complex have APOBEC3A-editable sites that, when edited, create premature stop codons. This is particularly notable because:

1. **Both hits of a pathway**: APOBEC3A targets both genes in the mTOR growth control pathway
2. **Haploinsufficiency**: TSC1/TSC2 are haploinsufficient -- loss of one copy is sufficient for tumor formation
3. **High editing rates**: Both show substantial editing in overexpression (31.4% and 17.2%)

Neither site is confirmed as endogenously edited (not in the Levanon database), so the clinical relevance depends on whether these sites are edited at detectable levels in vivo, particularly in tissues affected by TSC (brain, kidney).

### ClinVar Context

TSC2 has 154 pathogenic and 102 benign C>T variants. TSC1 has 73 pathogenic and 51 benign. Both genes show the genome-wide pattern of higher editing scores for pathogenic variants.

---

## Case Study 3: PMS2 -- DNA Mismatch Repair and Lynch Syndrome

### Gene and Disease

PMS2 is a DNA mismatch repair gene. Germline loss-of-function mutations cause Lynch syndrome type 4 (hereditary nonpolyposis colorectal cancer, HNPCC). Lynch syndrome accounts for ~3% of all colorectal cancers.

### The Editing Event

- **Position**: chr7:5,986,954 (minus strand)
- **Molecular consequence**: Nonsense (premature stop codon)
- **GB score**: 0.970
- **A3A overexpression rate**: 16.2%

### Significance

PMS2 editing creating a stopgain is clinically significant because:

1. **DNA repair deficiency**: Loss of PMS2 function leads to microsatellite instability (MSI), a hallmark of mismatch repair deficiency
2. **Cancer driver pathway**: MSI-high tumors have characteristic mutational signatures and respond to immune checkpoint inhibitors
3. **APOBEC connection**: APOBEC enzymes are already known to contribute to cancer mutagenesis (SBS2/SBS13 signatures). If APOBEC3A editing inactivates a DNA repair gene at the RNA level, this creates a potential positive feedback loop -- APOBEC activity reduces DNA repair capacity, potentially allowing more DNA mutations to accumulate

This site is from overexpression data only (not endogenously confirmed), but the mechanistic connection between APOBEC editing and DNA repair is noteworthy.

---

## Case Study 4: APOB -- The Canonical Physiological Edit

### Gene and Disease

APOB (apolipoprotein B) is the canonical example of C-to-U RNA editing. APOBEC1 (not APOBEC3A) edits APOB mRNA at position 6666 (CAA to UAA stop codon) in the intestine, converting the full-length APOB-100 protein to the truncated APOB-48 isoform. APOB-48 is essential for chylomicron assembly and dietary lipid absorption.

### The Editing Event

- **Position**: chr2:21,010,329 (classified as "Neither" enzyme -- consistent with APOBEC1)
- **Tissue**: Ubiquitous editing, but highest in small intestine (86.9%) and brain cerebellum (100% in pooled data)
- **Mean editing rate**: 10.8% (extremely high for an editing site)

### Significance

APOB editing is the positive control -- the one case where RNA editing creating a premature stop codon is known to be PHYSIOLOGICAL rather than pathogenic. This editing:

1. **Is essential**: APOB-48 is required for intestinal lipid absorption
2. **Is tissue-regulated**: High in intestine, lower elsewhere
3. **Is evolutionarily conserved**: Present in all mammals
4. **Is NOT disease-causing**: The stopgain is the intended biological function

This case illustrates that not all stopgain editing events are pathogenic. Context matters: the same molecular event (premature truncation) can be physiological (APOB) or potentially pathogenic (SDHB) depending on gene function and tissue context.

### ClinVar Context

APOB has 27 pathogenic C>T variants (familial hypobetalipoproteinemia), but the known editing site itself is NOT classified as pathogenic -- correctly reflecting its physiological role.

---

## Case Study 5: BMPR2 -- Pulmonary Arterial Hypertension

### Gene and Disease

BMPR2 (Bone Morphogenetic Protein Receptor Type 2) is a TGF-beta superfamily receptor. Germline loss-of-function mutations cause hereditary pulmonary arterial hypertension (PAH), a progressive disease of the pulmonary vasculature with high mortality.

### The Editing Event

- **Position**: chr2:202,530,819 (plus strand)
- **Molecular consequence**: Nonsense (premature stop codon)
- **GB score**: 0.989 (very high confidence editing target)
- **A3A overexpression rate**: 26.2%

### Significance

BMPR2 editing is interesting because:

1. **Haploinsufficiency with incomplete penetrance**: Only ~20% of BMPR2 mutation carriers develop PAH. Environmental factors and modifier genes determine penetrance.
2. **Could editing be a modifier?**: If APOBEC3A edits BMPR2 mRNA at this position in pulmonary endothelial cells during inflammatory episodes, the transient reduction in BMPR2 protein could lower the threshold for PAH development in genetically predisposed individuals.
3. **A3A is inflammation-induced**: APOBEC3A expression spikes during viral infection and interferon signaling. Repeated inflammatory episodes could transiently edit BMPR2 in lung tissue.

This site is from overexpression data only. APOBEC3A editing in lung tissue is relatively low (0.53% mean rate across A3A sites), but the incomplete penetrance of BMPR2 mutations makes even modest RNA-level modulation potentially significant.

ClinVar context: BMPR2 has 68 pathogenic C>T variants, reflecting the many different positions where nonsense mutations can inactivate this receptor.

---

## Summary Table

| Gene | Consequence | GB Score | Endogenous? | Disease | Key Insight |
|------|------------|----------|-------------|---------|-------------|
| SDHB | Stopgain | 0.977 | Yes (blood 1.2%) | Paraganglioma | Tumor suppressor LoF, misannotation risk |
| TSC1/TSC2 | Stopgain | 0.915/0.913 | No | Tuberous sclerosis | Dual mTOR pathway targeting |
| PMS2 | Stopgain | 0.970 | No | Lynch syndrome | DNA repair + APOBEC feedback loop |
| APOB | Stopgain | N/A | Yes (intestine 87%) | Physiological | Positive control: editing is the function |
| BMPR2 | Stopgain | 0.989 | No | Pulmonary hypertension | Incomplete penetrance modifier? |

---

## Conclusions

1. **The 89% nonsense rate is the key finding**: APOBEC3A editing at pathogenic ClinVar positions almost exclusively creates premature stop codons, not missense changes. This reflects the biochemistry of C-to-U editing in specific codon contexts.

2. **SDHB is the single strongest clinical case**: It uniquely combines endogenous editing, blood specificity, stopgain in a tumor suppressor, and direct disease relevance.

3. **Overexpression vs. endogenous**: 35 of 36 sites are overexpression-only. While this demonstrates that APOBEC3A *can* edit these positions, the clinical relevance depends on whether endogenous editing occurs at detectable levels. High-depth RNA-seq studies in relevant tissues could resolve this.

4. **APOB is the important counterexample**: Not all stopgain editing is pathogenic. Clinical interpretation must consider the biological function of the truncated protein.

5. **Misannotation risk is real but narrow**: Only SDHB has clear misannotation potential (endogenous editing at a pathogenic site). Variant calling pipelines should flag known editing sites, especially for RNA-based diagnostic assays.
