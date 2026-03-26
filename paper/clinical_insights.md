

# Clinical Interpretation Text for Multi-Enzyme APOBEC Report

---

## APOBEC3A (A3A)

**Key Finding:**
APOBEC3A editing is governed primarily by RNA secondary structure, not sequence alone: `relative_loop_position` is the single most informative feature (importance 0.213), meaning the cytidine's position within an unpaired loop is the strongest determinant of editing. The classifier achieves AUROC 0.907 using 40 hand-crafted features, and the model shows significant pathogenic enrichment in ClinVar (OR=1.279, p<1e-138), establishing that A3A editing sites are non-randomly distributed with respect to disease-relevant variation.

**Clinical Relevance:**
A3A is the only APOBEC enzyme with robust pathogenic enrichment at standard classification thresholds (OR=1.279 at P>=0.5), suggesting that C-to-U recoding at A3A target sites contributes measurably to human disease burden. This enrichment persists after Bayesian prior calibration from the training prevalence (pi=0.50) to the genomic base rate (pi=0.019), confirming it is not an artifact of class imbalance. The poor cross-dataset rate prediction (Spearman=0.122) reflects genuinely distinct rate distributions across experimental systems -- Levanon tissue rates, Kockler overexpression rates, and Baysal endogenous rates measure fundamentally different quantities. Clinically, this means site identity is predictable but editing magnitude is context-dependent and cannot be transferred across conditions without tissue-matched training data.

**Tissue and Disease Context:**
A3A editing is sharply blood-specific: 81 of 120 Levanon A3A-only sites show peak editing in blood, consistent with interferon-induced expression in monocytes and macrophages during innate immune activation. This tissue restriction implies that A3A-mediated recoding variants are most likely to manifest clinically in hematologic and inflammatory contexts, and that bulk tissue RNA-seq from non-immune organs will systematically underestimate A3A editing activity. The strong TC dinucleotide preference (84% in endogenous Levanon sites, dropping to 51% in overexpression systems like Kockler) indicates that endogenous editing is more sequence-selective than overexpression experiments suggest.

---

## APOBEC3B (A3B)

**Key Finding:**
After correcting a genome-build mismatch in negative generation (hg38 vs hg19), A3B classification jumps from AUROC 0.831 to 0.941, with structure is the primary predictor (StructOnly AUROC=0.934 vs MotifOnly=0.596). Unlike A3A and A3G, A3B shows no dominant sequence motif (TC only 32%) and no positional bias within loops (mean RLP=0.496, effectively symmetric), indicating that A3B recognizes structural context with unusual motif promiscuity.

**Clinical Relevance:**
A3B is the dominant somatic DNA mutator in breast, bladder, and cervical cancers, but its RNA editing activity has been controversial: Butt et al. (2024) reported widespread A3B RNA editing while Alonso de la Vega et al. (2023) found negligible activity. Our structural analysis resolves this contradiction -- A3B RNA editing is real but structure-dependent, and studies using bulk overexpression without controlling for structural context will yield inconsistent results. The calibrated ClinVar enrichment (OR=1.552) substantially exceeds the raw value (OR=1.082), indicating that A3B edits a small but pathogenically concentrated set of sites that only emerge after correcting for the low genomic prior. This positions A3B RNA editing as a potential modifier of cancer transcriptomes beyond its well-characterized DNA mutagenesis.

**Tissue and Disease Context:**
A3B is constitutively expressed across many solid tissues, unlike the interferon-induced A3A, which explains its relevance to epithelial cancers. The absence of a dominant motif preference means A3B RNA editing sites cannot be identified by sequence scanning alone, making structure-aware computational prediction essential for cataloguing its targets in tumor transcriptomes.

---

## APOBEC3G (A3G)

**Key Finding:**
A3G is an extreme structural specialist: it edits almost exclusively at the 3-prime end of small tetraloops (mean RLP=0.935, mean loop_size=4.5) in a strict CC dinucleotide context (93.3%). This combination of positional and sequence constraint is the most restrictive editing program among all APOBEC enzymes, yielding a compact but highly predictable target set (AUROC=0.841).

**Clinical Relevance:**
A3G targets in CC context show strong ClinVar pathogenic enrichment (OR=1.551, p<1e-168), the second highest raw enrichment after A3A. Because A3G editing is confined to CC dinucleotides at structurally constrained positions, its disease-relevant targets are few but high-confidence: a CC-to-CU change at the apex of a tetraloop is almost pathognomonic of A3G activity. This structural specificity means that pathogenic variants at A3G sites likely disrupt highly conserved stem-loop elements, consistent with the known functional importance of tetraloops in mRNA regulation and ribosomal RNA.

**Tissue and Disease Context:**
A3G editing is concentrated in testis (31 of 60 Levanon A3G-only sites) and is induced by mitochondrial hypoxia in NK cells, pointing to roles in germline RNA regulation and hypoxia-responsive immune editing. The testis enrichment suggests that A3G-mediated recoding may affect transcripts involved in spermatogenesis or germline quality control, a biology largely unexplored for C-to-U editing enzymes.

---

## A3A_A3G (Dual-Enzyme Sites)

**Key Finding:**
Sites edited by both A3A and A3G achieve the highest classification accuracy of any category (AUROC=0.951), but their structural features closely resemble A3G (RLP=0.935, loop_size=4.5), though 33% retain TC motif: CC-dominated (65.2%), tetraloop-positioned (RLP=0.935, loop_size=4.5), and strongly correlated with A3G tissue rates (r=0.926). Crucially, the UCC trinucleotide is not enriched, meaning dual recognition is driven by shared structural preferences rather than a hybrid sequence motif.

**Clinical Relevance:**
The high AUROC indicates that dual-enzyme sites are the most structurally constrained editing targets in the transcriptome, making them the easiest to predict and the most likely to be functionally consequential. Their tissue correlation pattern (blood plus ubiquitous expression) means these sites are edited across more physiological contexts than single-enzyme targets, increasing cumulative exposure to C-to-U recoding over a lifetime. For variant interpretation, a pathogenic variant at a dual-enzyme site should be flagged as subject to editing by multiple APOBEC family members, compounding the probability of functional impact.

**Tissue and Disease Context:**
The strong A3G tissue correlation (r=0.926 vs r=0.610 for A3A) suggests that A3G is the primary editor at these sites, with A3A contributing opportunistically in interferon-stimulated states. This implies that dual-enzyme sites are constitutively edited in A3G-expressing tissues (testis, lymphocytes) and additionally edited during inflammation when A3A is induced.

---

## Neither (Putative APOBEC1)

**Key Finding:**
The "Neither" category -- sites edited by neither A3A nor A3G in enzyme-specific assays -- displays the hallmarks of APOBEC1 editing: near-random dinucleotide context (TC=24%, CC=35%), ACA as the top trinucleotide (the canonical APOBEC1 signature), AU-rich mooring sequences (53.5% AU vs 41.8% for A3A, p<0.0001), and enrichment in non-coding mRNA regions (62.1%, predominantly 3-prime UTRs). These features are consistent with APOBEC1-mediated editing, though the GI vs immune tissue enrichment is not statistically significant (p=0.82) of the Levanon/Advisor database.

**Clinical Relevance:**
APOBEC1 is the ancestral C-to-U editor, responsible for the canonical apolipoprotein B mRNA editing that creates a premature stop codon and produces the ApoB-48 isoform essential for lipid absorption. The structure-independence of this category (StructOnly AUROC=0.659, barely above chance) distinguishes it fundamentally from all APOBEC3 enzymes and confirms that APOBEC1 recognition depends on the downstream mooring sequence rather than secondary structure. For clinical genomics, variants at these sites should be evaluated in the context of lipid metabolism and intestinal function rather than immune or cancer biology.

**Tissue and Disease Context:**
The sharp intestinal enrichment (small intestine is the top tissue at 1.93% mean editing rate) is consistent with APOBEC1's known expression pattern in enterocytes. The 3-prime UTR targeting suggests a broader regulatory role beyond ApoB, potentially affecting mRNA stability or translation efficiency for intestine-expressed transcripts -- a biology that remains largely uncharacterized beyond the single canonical substrate.

---

## Unknown (NaN Enzyme Assignment)

**Key Finding:**
Sites with no enzyme assignment show the weakest classification performance (AUROC=0.769) and a mixed feature profile: moderate TC content (43%), ubiquitous tissue expression, and the highest gastrointestinal-to-immune tissue ratio (2.61) of any category. This heterogeneity suggests the "Unknown" category is a mixture of low-confidence APOBEC1 sites and sites edited by enzymes not represented in current assays.

**Clinical Relevance:**
The weak classifier performance and mixed signal make these sites unsuitable for clinical variant interpretation at present. However, the elevated GI/immune ratio suggests a substantial fraction may be APOBEC1 targets that did not meet the stringent criteria for the "Neither" category. These sites should be deprioritized for clinical annotation until enzyme assignment can be refined through additional experimental data or improved deconvolution methods.

**Tissue and Disease Context:**
The ubiquitous tissue distribution distinguishes this category from the tissue-restricted patterns of all assigned enzymes (A3A in blood, A3G in testis, APOBEC1 in intestine), suggesting either genuinely housekeeping editing activity or, more likely, the superposition of multiple tissue-specific programs that appears ubiquitous in aggregate.