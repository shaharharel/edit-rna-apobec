# Biologist Review

I now have a comprehensive understanding of the project. Here is my deep biological review.

---

# Deep Biological Review: edit-rna-apobec

## 1. "Both" (A3A+A3G) Sites: Biological Mechanisms for Dual Recognition

### The motif paradox is less paradoxical than it appears

The framing of this question assumes A3A and A3G have non-overlapping motif preferences. In reality, the motif selectivity is statistical, not absolute. Consider the empirical data:

- **A3A**: ~51% TC (from Kockler BT-474), but also edits non-TC contexts at lower rates
- **A3G**: ~91% CC, but 9.2% of its sites are TC
- **Levanon 636 overall**: ~38% TC after the hg38 coordinate fix

The 178 "Both" sites are classified by the Levanon/Advisor database based on HEK293 overexpression experiments -- sites where editing increased upon EITHER A3A or A3G overexpression. This creates several biologically plausible explanations:

**Hypothesis 1: Overlapping trinucleotide contexts.** The most likely explanation is that many "Both" sites sit in sequence contexts tolerated by both enzymes. The key trinucleotide to check is **TCC** (5'-UCC-3'): this is simultaneously TC context for A3A (reading the C at position 0) and CC context for A3G (reading the same C). I predict the "Both" sites will be enriched for TCC/UCC trinucleotides compared to A3A-only (enriched for TCA, TCG, TCT) or A3G-only (enriched for ACC, GCC).

**Testable with existing data**: Extract the -1 and +1 positions for all 178 "Both" sites from the 201-nt sequences. Compute trinucleotide frequencies at positions -1,0,+1. Compare to A3A-only and A3G-only. If UCC/TCC is enriched in "Both" relative to both single-enzyme categories, this confirms the shared motif hypothesis.

**Hypothesis 2: Structure-driven permissivity.** Some hairpin structures may be so favorable (ideal loop size, position, stability) that they overcome suboptimal motif preferences for either enzyme. Your data already shows that structure is more important than motif for all three enzymes. If "Both" sites have notably stronger structural features (smaller loops, more central loop position, higher delta-MFE), this would argue that structure can compensate for motif.

**Testable with existing data**: Compare loop geometry distributions (loop_size, relative_loop_position, dist_to_apex, local_unpaired_fraction) across A3A-only, A3G-only, and Both. If "Both" sites have the most extreme structural features, structure is driving dual recognition.

**Hypothesis 3: Expression overlap is a confounder.** HEK293T cells express low levels of endogenous APOBEC enzymes. In the overexpression experiments that defined these categories, both A3A and A3G were overexpressed separately. A site classified as "Both" simply had detectable editing in both conditions. Some of these may reflect low-specificity editing at supraphysiological enzyme concentrations rather than genuine dual targeting in vivo. The 54-tissue GTEx rate data could help distinguish: if "Both" sites show editing patterns more similar to A3A-dominant tissues (immune, blood) than A3G-dominant tissues (specific cell types under stress), the "Both" classification may be an artifact of overexpression sensitivity.

**Testable with GTEx data**: Compute correlation between each "Both" site's tissue rate profile and the mean profiles of A3A-only and A3G-only sites. Cluster the "Both" sites -- do they partition into A3A-like and A3G-like subgroups, or do they form a genuinely intermediate cluster?

### Critical caveat about the "Both" category

The Levanon database defines enzyme attribution using overexpression in a single cell line. This is a biochemical competence test, not an endogenous activity measurement. A site being editable by both enzymes in vitro does not mean both enzymes edit it in vivo. Tissue expression patterns of A3A vs A3G are quite different: A3A is broadly expressed (immune cells, epithelial tissues), while A3G is more restricted (lymphocytes, induced by interferon, activated by mitochondrial hypoxia in NK cells per the Dang 2019 data). In most tissues, only one enzyme would be the active editor.

## 2. "Neither" Sites: Candidate Editors

The 206 "Neither" sites are the most scientifically interesting category because they represent unexplained biology. Here is a ranked list of candidate explanations:

### Candidate 1: APOBEC1 (highest probability)

APOBEC1 is the original C-to-U RNA editor, responsible for the canonical apoB mRNA editing that gives the APOBEC family its name. Key features:

- **Motif**: Requires a downstream "mooring sequence" (typically 4-8 nt 3' of the edit site) rather than an upstream dinucleotide context. The canonical APOBEC1 recognition involves a spacer element and an AU-rich mooring.
- **Tissue specificity**: Primarily expressed in small intestine and liver in humans. Recently shown to have broader low-level expression.
- **Cofactor dependence**: Requires ACF/A1CF (APOBEC1 complementation factor) or RBM47 for editing activity.
- **Structure**: Less dependent on hairpin structures than A3A/A3G; the mooring sequence functions as a linear sequence element.

**Testable predictions**:
1. **Tissue enrichment**: If "Neither" sites are preferentially edited in intestine, liver, or colon in the 54-tissue GTEx data, this strongly implicates APOBEC1. This is the single most informative test.
2. **Mooring sequence**: Check the +4 to +12 region downstream of the edit site for enrichment of AU-rich motifs. APOBEC1's mooring is typically within 2-8 nt of the edited C.
3. **No upstream dinucleotide preference**: "Neither" sites should show no enrichment for TC or CC at position -1. If the -1 position is essentially random (uniform A/C/G/U), this rules out APOBEC3 family editing.
4. **Structure independence**: If "Neither" sites show weaker association with loop structures (lower is_unpaired fraction, no relative_loop_position preference), this is consistent with APOBEC1 which does not require hairpin context.

### Candidate 2: APOBEC3B

A3B was not tested in the original Levanon classification (which used A3A and A3G overexpression only). If some "Neither" sites are A3B targets, they should:
- Show editing in breast/cancer tissues (where A3B is highly expressed)
- Have UCC motif preference (per your Kockler/Zhang data showing A3B prefers UCC)
- Show hairpin structure but no 3'-end positional bias (per your RLP=0.515 finding)

**Testable**: Score all 206 "Neither" sites with your trained A3B GB_HandFeatures classifier. If a substantial subset (>20%) scores above the classification threshold, these may be A3B targets that were invisible to the A3A/A3G-only overexpression screen.

### Candidate 3: APOBEC3C, 3F, or 3H

These APOBEC family members have demonstrated or suspected RNA editing activity but are poorly characterized. APOBEC3H in particular has been shown to have C-to-U RNA editing activity in monocytes and macrophages.

### Candidate 4: ADAR-mediated C-to-U (unlikely but worth ruling out)

ADAR enzymes perform A-to-I editing, not C-to-U. However, misannotation is a real concern in this field (as your own 7.3% vs 38% TC-motif correction demonstrates). Some "Neither" sites could be:
- A-to-I sites at positions adjacent to C positions, creating apparent C-to-U changes in sequencing
- Somatic C>T mutations misidentified as editing

**Testable**: Check if "Neither" sites are enriched near known ADAR editing hotspots (Alu elements, dsRNA structures). If they cluster in Alu repeats, they are almost certainly ADAR misannotations.

## 3. Clinical Relevance: Disease Mechanisms

Your ClinVar enrichment findings (A3A OR=1.33, A3B OR=1.55 calibrated, A3G OR=1.76 CC-context) require careful mechanistic interpretation. There are two fundamentally different causal directions:

### Direction 1: APOBEC editing CREATES pathogenic variants (editing-causes-disease)

C-to-U RNA editing at coding positions creates missense, nonsense, or splice-affecting changes at the RNA level. If a C-to-U change at a particular codon position is pathogenic when it occurs in DNA (as annotated in ClinVar), then the same change occurring transiently at the RNA level would produce a fraction of mutant protein proportional to the editing rate.

**Key mechanistic considerations**:
- **Editing rates matter enormously.** A site edited at 2% produces 98% wild-type protein. The pathogenic impact would be negligible unless the mutant protein is dominant-negative or acts as a gain-of-function at very low concentrations (e.g., constitutively active kinases, toxic misfolded proteins).
- **Tissue-specificity creates tissue-specific risk.** If A3A editing is highest in monocytes (as the GTEx data suggests), then the pathogenic impact of A3A editing would be most relevant to hematopoietic malignancies or immune disorders. A3G's restriction to hypoxic NK cells would limit its pathogenic impact to very specific contexts.
- **Nonsynonymous vs synonymous enrichment.** The OR should be stratified by exonic function. If the pathogenic enrichment is specifically among nonsynonymous/stopgain variants (not synonymous), this supports the editing-creates-disease direction.

**Testable with existing data**: In your 1.69M ClinVar scored variants, compare the pathogenic enrichment OR separately for: (a) nonsynonymous C>T, (b) synonymous C>T, (c) intronic/non-coding C>T. If enrichment is concentrated in nonsynonymous variants, this supports functional editing creating disease-relevant protein changes.

### Direction 2: Pathogenic DNA variants DISRUPT normal editing (disease-disrupts-editing)

Some ClinVar pathogenic variants may be at positions that happen to be normally edited. The variant itself (a germline C>T mutation) would eliminate the editing site, potentially disrupting a regulatory mechanism that depends on the C-to-U transition being controllable. This is the loss-of-editing-regulation hypothesis.

**This direction is less likely to explain the observed enrichment** because:
- There are only ~5,000-10,000 reproducible editing sites genome-wide, so the prior probability of a random ClinVar variant hitting an editing site is very low
- Your enrichment is detected at the model score level (predicted editing probability), not direct coordinate overlap

### The most nuanced interpretation

The enrichment likely reflects a third explanation: **APOBEC editing sites are in structurally and functionally constrained regions** (hairpin loops in coding mRNAs). These same regions are under purifying selection, meaning mutations at these positions are more likely to be pathogenic simply because the positions are important for RNA function or protein coding. The hairpin structure that attracts APOBEC is itself a marker of functional constraint.

**Critical control**: Compare your enrichment OR to what you would get from a model that predicts "is this position in a hairpin loop in a coding region?" with no APOBEC-specific features. If the hairpin-loop-in-CDS model shows similar pathogenic enrichment, then the APOBEC signal may be an indirect reflection of structural constraint, not editing per se.

## 4. The Rate Prediction Ceiling

Spearman=0.122 is indeed low, and I believe this reflects genuine biological complexity rather than model failure. Here is my assessment of what determines editing rate, ordered by likely impact:

### Factor 1: Enzyme expression level (dominant, inaccessible to your model)

The single largest determinant of editing rate at any site is how much enzyme is present in that cell type and condition. APOBEC3A expression varies over 3 orders of magnitude across GTEx tissues (highest in whole blood and immune cells, lowest in brain). Your model has no access to enzyme expression levels -- it sees only the RNA sequence and structure context. This immediately caps the achievable correlation when pooling rates across tissues/datasets.

**This is why cross-dataset rate prediction fails.** Baysal (HEK293T overexpression), Levanon (endogenous GTEx), and Asaoka (HEK293T overexpression, different conditions) have fundamentally different enzyme concentrations. Per-dataset Z-scoring partially mitigates this, but only within each dataset's dynamic range.

### Factor 2: Local RNA structure accessibility (partially captured)

Your model captures this through ViennaRNA delta features, but equilibrium thermodynamic predictions are imperfect proxies for the in-vivo structure. The actual structure depends on:
- Co-transcriptional folding kinetics
- RNA-binding protein occupancy
- RNA modifications (m6A, pseudouridine) that alter local structure
- Ribosome transit (editing in CDS regions may compete with translation)

### Factor 3: Kinetic competition (inaccessible)

The editing rate at a given site reflects a kinetic competition between: (a) APOBEC binding and catalysis, (b) RNA degradation, (c) translation, (d) other RNA-binding proteins occupying the same region. None of these are captured by sequence/structure features alone.

### Factor 4: Chromatin state and transcription rate (inaccessible)

If APOBEC3A editing occurs co-transcriptionally or shortly after transcription, then genes with higher transcription rates expose more RNA substrate per unit time, potentially reducing the editing fraction per molecule even at the same number of editing events.

### Practical implication

Rate prediction within a single tissue/condition (controlling for enzyme expression) should perform substantially better than across tissues. Your within-Baysal rate analysis (single cell line, single condition) is the fairest test. If Spearman within Baysal is also low (~0.1), then the rate variance is driven by factors beyond sequence/structure context (factors 2-4 above). If within-Baysal Spearman is noticeably higher (>0.2), this confirms that most rate variance is inter-dataset (factor 1).

**Recommendation**: Report rate prediction performance stratified by dataset. The cross-dataset failure is not a bug -- it is a genuine biological finding about what determines rate.

## 5. GTEx Tissue Rate Analyses for the Levanon Expansion

The 54-tissue rate profiles for all 636 sites are the most valuable unexploited data in this project. Here are the analyses ranked by scientific impact:

### Analysis 1: Tissue clustering reveals editing programs (HIGH IMPACT)

Cluster the 54 tissues by their editing rate profiles across sites. This will reveal which tissues share editing machinery. Prediction: you will find 3-4 major clusters:
- **Immune cluster**: Whole blood, spleen, lymph nodes (A3A-driven)
- **Epithelial cluster**: Skin, esophagus, colon (A3A-driven, different expression levels)
- **Restricted cluster**: Specific tissues where A3G is induced (possibly testis, transformed cells)
- **Low-editing cluster**: Brain, muscle, adipose (low APOBEC expression)

Use hierarchical clustering with Spearman correlation distance. The dendrogram structure is the figure.

### Analysis 2: Enzyme category predicts tissue distribution (HIGH IMPACT)

For each of the 4 enzyme categories, compute the mean editing rate profile across 54 tissues. The key predictions:
- **A3A-only sites**: Highest rates in immune and epithelial tissues
- **A3G-only sites**: May show peak rates in specific tissues (testis? transformed cells?)
- **Both sites**: Intermediate -- edited in tissues expressing either enzyme
- **Neither sites**: If APOBEC1, should peak in small intestine/liver

This is the single most informative analysis for identifying the "Neither" editor. A heatmap of mean rates per enzyme category per tissue, with hierarchical clustering on both axes, would be a central figure.

### Analysis 3: Site-level tissue profiles predict enzyme category (MEDIUM IMPACT)

Train a simple classifier (multinomial logistic regression or random forest) on the 54-tissue rate vector to predict enzyme category (A3A/A3G/Both/Neither). If the 54-tissue profile can accurately classify enzyme attribution, this confirms that the enzyme categories reflect genuinely different tissue-expression-driven programs. Feature importances would reveal which tissues are diagnostic for each enzyme.

### Analysis 4: Rate variance decomposition (MEDIUM IMPACT)

For each site, decompose rate variance into: (a) mean rate (overall editability), (b) tissue breadth (number of tissues with detectable editing), (c) tissue specificity (entropy of the rate distribution). Plot these three metrics by enzyme category. Prediction: "A3A-only" sites should have the broadest tissue distribution, "A3G-only" the narrowest, "Both" intermediate, and "Neither" will be informative for identifying the editor.

## 6. Publication Strategy

### The core story

This project has evolved beyond the original EditRNA edit-effect framework into something potentially more impactful: **the first systematic computational comparison of three APOBEC editing programs, revealing that RNA structure is the dominant determinant of site selection across all enzymes, with each enzyme showing distinct structural preferences.**

The publication-ready claims, ranked by strength:

**Claim 1 (strongest): Structure, not motif, is the primary determinant of APOBEC editing site selection.** Evidence: For all three enzymes, structural features (relative_loop_position, dist_to_apex, local_unpaired_fraction) outperform motif-only models. A3A: structure AUROC=0.908, motif-only substantially lower. A3G: StructOnly AUROC=0.935 vs MotifOnly=0.689. This is the central finding.

**Claim 2 (strong): Three enzymes define three distinct editing programs.** A3A requires TC + moderate 3' loop position (RLP~0.68), A3B uses hairpins but with no positional bias (RLP~0.51, resolving the Butt/Alonso de la Vega contradiction), A3G requires CC + extreme 3' tetraloop (RLP~0.92). This is novel comparative biology.

**Claim 3 (strong, with caveats): All three enzymes show ClinVar pathogenic enrichment.** GB models trained on structure+motif features show significant enrichment. The enrichment persists after Bayesian prior calibration. The caveat: you must address whether this reflects editing biology or structural constraint at conserved positions (see control above).

**Claim 4 (moderate): The Levanon expansion reveals a fourth editing program.** "Neither" sites (pending the tissue analysis) likely identify APOBEC1 or another editor, distinguishable by tissue distribution and lack of dinucleotide motif.

### What the paper should NOT claim

- Do NOT claim that the edit-effect framework outperforms subtraction baselines for this task until that comparison is properly done. The current results show GB on hand-crafted features outperforming neural approaches.
- Do NOT claim rate prediction works. Spearman=0.122 is not clinically or biologically useful. Frame it as "rate is determined by factors beyond sequence context."
- Do NOT claim individual ClinVar variants are pathogenic because of APOBEC editing. The enrichment is a statistical population-level finding.

### Key figures for the paper

1. **Figure 1**: Three-panel loop position distributions for A3A/A3B/A3G, showing the distinct structural preferences. This is the visual proof of three editing programs.
2. **Figure 2**: Classification performance comparison across enzymes (AUROC bar chart with structure vs motif ablation for each enzyme).
3. **Figure 3**: ClinVar enrichment by enzyme, with calibrated ORs and the structural-constraint control.
4. **Figure 4**: GTEx tissue heatmap for the 4 enzyme categories (once Levanon expansion is done).
5. **Supplementary**: Feature importance rankings for each enzyme, cross-dataset generalization matrix, motif logos.

### Critical controls needed before publication

1. **Structural constraint control for ClinVar**: Build a "is-this-position-in-a-hairpin-loop" model and check its ClinVar enrichment. If similar to your APOBEC model, the enrichment may not be APOBEC-specific.
2. **Negative set validation**: Your negatives are generated from the genome, matched by motif. Verify they are not depleted for known editing sites in other databases (RADAR, REDIportal). If they are, your positives and negatives may differ in ways beyond editing.
3. **Asaoka data quality resolution**: The 97.6% Baysal TC-motif vs the broader Asaoka set (which Baysal is a subset of) needs clear explanation. Are you using Asaoka sites that passed Baysal's filters, or the full Asaoka set? The pipeline deduplication needs to be transparent.
4. **Independence of Levanon categories**: Verify that the "Affecting Over Expressed APOBEC" column is based on a statistical test (not just presence/absence of editing signal), and report the criteria used by the original authors.

## 7. Asaoka Data Quality

The question asks about "97.6% non-TC" in Asaoka but this appears to be a confusion. From the code:
- **Baysal** (subset of Asaoka) has **97.6% TC** (4094/4196 are TC)
- **Levanon** has only **~38% TC** (after hg38 correction, previously reported as 7.3% with wrong coordinates)
- **Alqassim** has only **~8.7% TC**

The Levanon and Alqassim low TC fractions are biologically expected because these datasets contain sites edited by ALL APOBEC enzymes (A3A, A3G, Both, Neither), not just A3A. A3G sites are CC-context, and "Neither" sites may lack any dinucleotide preference. The ~38% TC in Levanon is consistent with: 120 A3A-only sites (most TC) + 60 A3G (mostly CC) + 178 Both (mixed) + 206 Neither (unknown motif) = approximately 38% overall TC.

For Asaoka specifically: the Asaoka 2019 study was A3A overexpression in HEK293T, so the non-TC sites in that dataset could represent:
1. **Low-specificity editing at high enzyme concentrations**: At supraphysiological expression, A3A can edit non-TC contexts at low rates. This is well-documented in vitro.
2. **Endogenous ADAR editing in HEK293T**: These cells have endogenous ADAR activity. Without proper controls (e.g., catalytic-dead A3A), some detected C-to-U changes could be from endogenous editing.
3. **Technical noise**: False positive editing calls at the 1-2% editing rate threshold.

The Baysal filtering likely applied a stricter threshold or TC-context filter, retaining only high-confidence A3A targets (hence 97.6% TC). This makes Baysal the cleanest A3A rate dataset but introduces ascertainment bias: by filtering to TC-only, you lose information about whether structure can compensate for suboptimal motif.

### Recommendation

For classification, use the full dataset (including non-TC positives) with TC as a feature, not a filter. For rate prediction, the Baysal TC-only subset is appropriate since it removes the confound of mixed-enzyme sites. Make this distinction explicit in the paper.

---

## Summary of Priority Actions

1. **(Highest)** Run the GTEx 54-tissue analysis for all 4 enzyme categories. This is the most likely source of a novel biological finding (especially for "Neither" sites).
2. **(High)** Check trinucleotide context of "Both" sites for TCC/UCC enrichment.
3. **(High)** Build the structural-constraint ClinVar control to rule out confounding of the pathogenic enrichment.
4. **(High)** Score "Neither" sites with the A3B classifier.
5. **(Medium)** Stratify ClinVar enrichment by exonic function (nonsynonymous vs synonymous).
6. **(Medium)** Report within-dataset rate prediction separately to demonstrate that cross-dataset failure is biological, not methodological.
7. **(Lower)** Check "Neither" sites for mooring sequence enrichment and Alu element proximity.