# Evolutionary Analysis Plan: APOBEC Editability Across Species and Populations

**Status**: Planning — experiments to be prioritized and run

---

## Current Results (Baseline)

What we've shown so far:
- Editing sites are 24% more conserved than controls (human-chimp, p=5.94e-37)
- 99.3% of edited Cs conserved in chimp; 98.4% motif preserved
- Editability scores drop 6.1% in chimp (Spearman r=0.632)
- Genes with more editable positions are under stronger constraint (gnomAD LOEUF r=-0.211)
- High-editability positions have fewer germline variants (chr22 site-level, p=8e-156)

**What's missing**: These are descriptive associations. We haven't shown:
- Whether editability differences between species correlate with phenotypic divergence
- Whether editability predicts WHICH positions mutate in evolution
- Whether natural selection acts on editability itself (not just the sequence)
- Deep multi-species comparison beyond human-chimp

---

## Proposed Experiments (Priority Order)

### E1. Editability as Predictor of Human-Chimp Divergence (HIGH)

**Hypothesis**: Positions with higher editability scores show LESS divergence between human and chimp (purifying selection on editable positions).

**Method**:
1. Score ALL exonic C positions with full model (exome map — running now)
2. For each scored position, check if it's diverged in chimp (using liftOver + panTro6)
3. Logistic regression: P(diverged) ~ editability_score + trinucleotide + CpG + gene_expression
4. Does editability independently predict divergence after controlling for confounds?

**Expected result**: Higher editability → lower divergence probability. This would show editability captures functional constraint.

**Data**: Full exome editability map + panTro6 genome + existing liftOver pipeline.

### E2. Editability Predicts Position-Specific gnomAD Variant Frequency (HIGH)

**Hypothesis**: At the single-nucleotide level, higher editability correlates with lower allele frequency in gnomAD (rarer = more deleterious = under selection).

**Method**:
1. Download gnomAD exome VCF for all chromosomes (large download, ~50GB)
   - OR use gnomAD API for specific positions
   - OR download just the C>T variants (filter during download)
2. For each exonic C position in the editability map, look up gnomAD C>T variant (if any) and its allele frequency
3. Compare: at high-editability positions (top decile), are C>T variants rarer (lower AF)?
4. Control for trinucleotide, CpG, gene constraint (LOEUF), gene expression

**Expected result**: Negative correlation between editability and AF. Strong purifying selection at editable positions removes damaging variants.

**Data**: Full exome editability map + gnomAD VCF. Chr22 pilot already done (rho=-0.058, p=3e-69). Full genome would be definitive.

### E3. Multi-Species Editability Comparison (MEDIUM)

**Hypothesis**: The editable transcriptome has diverged more between distantly related species (human-mouse) than closely related ones (human-chimp), and the divergence rate differs by enzyme.

**Method**:
1. Download mouse genome (mm39) and perform liftOver from hg38
2. Score mouse orthologous sequences with our model (same pipeline as chimp)
3. Compare editability conservation: human-chimp (r=0.632) vs human-mouse (expected lower)
4. Per-enzyme: which enzyme's targets are most/least conserved across species?

**Expected result**: A3G targets (extreme structural constraints, tetraloop) most conserved; A3B targets (broad, context-dependent) least conserved. APOBEC1/"Neither" targets may show species-specific patterns (APOBEC1 function diverges between species).

**Data**: mm39 genome + liftOver chain file + existing pipeline.

### E4. Lineage-Specific Editability Gain/Loss (MEDIUM)

**Hypothesis**: Some positions gained editability in the human lineage (editable in human but not in chimp) — these may be under positive selection or contribute to human-specific gene regulation.

**Method**:
1. From the 3,610 ortholog pairs, identify positions where:
   - Human score HIGH, chimp score LOW → gained editability in human
   - Human score LOW, chimp score HIGH → lost editability in human
2. Characterize: what genes are these in? What functions?
3. Test: are "gained" sites in genes under positive selection (dN/dS > 1)?
4. Test: are "gained" sites in human-specific regulatory regions?

**Expected result**: Gained editability sites may be enriched in immune genes (APOBEC is part of innate immunity) or brain-expressed genes (human brain evolution).

**Data**: Existing cross-species scored data (3,610 orthologs) + dN/dS from Ensembl Compara.

### E5. Population-Level Editability Variation (LOW — future work)

**Hypothesis**: Common human polymorphisms (from gnomAD/1000 Genomes) alter editability at nearby cytidines, creating population-level variation in the editable transcriptome.

**Method**:
1. For common SNPs (AF > 5%) near known editing sites (±10bp), compute editability with and without the SNP
2. Identify "editability-altering variants" (eQTL analog for editing)
3. Test: are editability-altering variants under selection? (lower AF than neutral SNPs?)

**Expected result**: Some common variants significantly alter editing potential, creating inter-individual variation in APOBEC susceptibility.

**Data**: 1000 Genomes Phase 3 VCF + our model. Computationally expensive (millions of variants × ViennaRNA folding).

### E6. Tissue-Specific Evolutionary Constraint (LOW — future work)

**Hypothesis**: Editing sites active in many tissues (ubiquitous) are under stronger evolutionary constraint than tissue-specific sites.

**Method**:
1. Use Levanon tissue breadth data (54 GTEx tissues per site)
2. Correlate tissue breadth with: divergence rate (cross-species), gnomAD constraint, ClinVar pathogenicity
3. Test: ubiquitous editing sites are more conserved than tissue-specific ones

**Expected result**: Ubiquitous sites are more constrained (they affect more tissues, so selection is stronger). Blood-specific A3A sites may be less constrained (only functional in blood).

**Data**: Levanon tissue rates (already available) + cross-species data + gnomAD.

---

## Priority and Dependencies

| Experiment | Priority | Depends on | Effort | Expected impact |
|-----------|---------|-----------|--------|----------------|
| E1 (divergence prediction) | HIGH | Full exome map | 1 day | Strong — model-driven evolutionary test |
| E2 (gnomAD site-level full) | HIGH | Full exome map + gnomAD VCF | 2 days | Strong — genome-wide selection test |
| E3 (multi-species) | MEDIUM | mm39 download + liftOver | 1 day | Moderate — extends chimp result |
| E4 (lineage-specific) | MEDIUM | Existing data | 1 day | Potentially high if finds immune/brain genes |
| E5 (population variation) | LOW | 1000 Genomes + massive compute | 1 week | Future paper |
| E6 (tissue-specific constraint) | LOW | Existing data | Half day | Supporting |

**Recommended for current paper**: E1 + E2 (both require the full exome map, which is running).
**Recommended for follow-up**: E3 + E4.
**Future work**: E5 + E6.

---

## Key Question to Answer

Can we claim: "The editable transcriptome is under evolutionary constraint, and editability itself (not just sequence) is a target of natural selection"?

Current evidence supports the first part (constraint) but not the second (editability as selection target). E1 and E2 would strengthen the constraint argument. E4 and E5 would address whether editability itself is selected (by showing that editability-changing variants are under selection).
