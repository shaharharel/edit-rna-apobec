# Publication Plan: Multi-Enzyme APOBEC C-to-U RNA Editing

**Synthesized from**: AI scientist, expert biologist, PI advisor, and reviewer assessments
**Date**: 2026-03-19
**Status**: Active

---

## Target Journal

**Primary: Genome Biology** (IF ~13)
- Multi-enzyme computational analysis + clinical relevance + resource release
- The A3B contradiction resolution + cross-enzyme ClinVar is a natural fit
- Reviewers will value the systematic dataset curation and reproducibility

**Fallback: Nucleic Acids Research** (IF ~14) — Methods/resource track
**Stretch: Nature Communications** (IF ~17) — if the "Both"/"Neither" analysis yields a genuine surprise

---

## Paper Title (Working)

"Distinct structural programs govern APOBEC-mediated C-to-U RNA editing across enzyme families, with implications for pathogenic variant interpretation"

---

## Core Narrative

**Three APOBEC enzymes execute distinct RNA editing programs** defined by coupled motif-structure rules. Machine learning classifiers trained on these rules reveal that:
1. Structure (loop geometry) is the primary determinant across all enzymes
2. Each enzyme has a unique structural signature (A3A=moderate 3' loop, A3B=hairpin but symmetric, A3G=extreme 3' tetraloop)
3. ML-based editing predictions show significant ClinVar pathogenic enrichment — a signal invisible to existing rules-based predictors
4. "Both" (A3A+A3G) sites are CC-dominated with blood-specific tissue patterns; "Neither" sites are intestine-enriched with random motif, suggesting APOBEC1

**The contribution is biological discovery enabled by computation**, not computation validated by biology.

---

## Key Results Summary (Updated with Levanon Expansion)

### Classification (5-fold CV, GB_HandFeatures)
| Enzyme | n_pos | n_neg | AUROC | Bootstrap CI | Top Feature |
|--------|-------|-------|-------|-------------|-------------|
| A3A | 5,187 | 2,966 | 0.923 | — | relative_loop_position (0.213) |
| A3B | 4,180 | 4,177 | 0.831 | — | local_unpaired_fraction |
| A3G | 179 | 179 | 0.929 | 0.898–0.956 | dist_to_apex (0.319) |
| A3A_A3G (Both) | 178 | 178 | 0.941 | 0.917–0.962 | TBD |
| Neither | 206 | 206 | 0.840 | 0.793–0.874 | TBD (motif-dominated) |
| Unknown (NaN) | 72 | 72 | 0.782 | 0.703–0.865 | TBD |

### ClinVar Enrichment (GB_Full, pathogenic vs benign OR)
| Enzyme | Raw OR | Calibrated |
|--------|--------|------------|
| A3A | 1.33 (p<1e-40) | Persists |
| A3B | 1.08 raw / 1.55 calibrated | — |
| A3G | 1.76 CC-context | — |

### Key Biological Findings (New from Levanon Expansion)
1. **"Both" (A3A_A3G) sites are biologically A3G-like**: CC=65.2%, tissue rates correlate with A3G (r=0.926) not A3A (r=0.539). Classified better (AUROC=0.941) than either single enzyme.
2. **"Neither" sites are likely APOBEC1**: Random motif (TC=24%, CC=35%), intestine-specific tissue pattern (63/206), structure is uninformative (StructOnly AUROC=0.639). This matches APOBEC1 biology.
3. **"Unknown" (NaN enzyme) are mixed/weakly-edited**: TC=43%, tissue=ubiquitous, lowest AUROC (0.782). Likely a mix of low-confidence assignments.
4. **A3G expansion (119→179) maintains performance**: AUROC=0.929 (was 0.931 with 119). StructOnly still dominates (0.916). Tetraloop specialist confirmed.

---

## Paper Structure (7 Figures)

### Figure 1: Dataset Overview and Multi-Enzyme Landscape
- Panel A: Dataset composition (7 sources, 5 categories, sample sizes)
- Panel B: Motif logos per enzyme category
- Panel C: Loop geometry distributions (relative_loop_position, loop_size)
- Panel D: Tissue editing rate heatmap (54 GTEx tissues × 5 categories)

### Figure 2: Three Distinct Editing Programs
- Panel A: 3D scatter: TC%, CC%, relative_loop_position per enzyme
- Panel B: A3B contradiction resolution (Butt 2024 vs Alonso de la Vega 2023)
  - A3B IS in hairpins (64.2%) but NO 3'-of-loop bias (rlp=0.505)
- Panel C: A3G tetraloop specialist (rlp=0.913, loop_size=4 peak)
- Panel D: "Both" sites — CC-dominated, structurally intermediate

### Figure 3: Classification Performance
- Panel A: AUROC comparison across enzymes (bar chart with bootstrap CI)
- Panel B: Feature importance (GB_HandFeatures) per enzyme — stacked bar
- Panel C: Motif vs Structure vs Combined contribution (ablation)
- Panel D: Cross-enzyme classifier transfer matrix

### Figure 4: "Neither" Sites and APOBEC1 Identification
- Panel A: Motif distribution — near-random (neither TC nor CC enriched)
- Panel B: Tissue enrichment — intestine-specific (APOBEC1 territory)
- Panel C: Rate analysis — no correlation with A3A/A3G tissue patterns
- Panel D: Comparison with known APOBEC1 characteristics

### Figure 5: ClinVar Pathogenic Enrichment
- Panel A: OR vs threshold curves for all 3 enzymes
- Panel B: GB vs RNAsee comparison (GB enrichment, RNAsee depletion)
- Panel C: Bayesian prior calibration (π_model=0.50 → π_real=0.019)
- Panel D: Cross-enzyme enrichment comparison

### Figure 6: "Both" Sites and Dual-Enzyme Recognition
- Panel A: Trinucleotide analysis — UCC/TCC enrichment hypothesis
- Panel B: Tissue rate correlation with A3G (r=0.926) vs A3A (r=0.539)
- Panel C: Structural features compared to single-enzyme sites
- Panel D: Blood-specific editing pattern

### Figure 7: Rate Prediction and Limitations
- Panel A: Rate prediction performance (Spearman per model)
- Panel B: Cross-dataset generalization failure
- Panel C: Feature importance for rate (relative_loop_position still #1)
- Panel D: Tissue breadth vs mean rate per category

### Figure 8: Cross-Species Conservation (Human vs Chimpanzee) — NEW
- Panel A: Substitution rate at editing sites vs controls (boxplot, p=5.94e-37)
- Panel B: Center C conservation and motif preservation rates (bar chart)
- Panel C: Per-enzyme conservation breakdown
- Panel D: TC/CC motif density unchanged between species

### Supplementary
- Full model comparison tables (13+ models for A3A)
- All fold-level CV results
- ClinVar detailed scoring methodology
- Genome build discovery documentation
- Dataset QA checklist
- APOBEC4 analysis (negative control validation)
- Cross-species detailed trinucleotide transition data

---

## Critical Reviewer Concerns to Address Pre-Submission

### From Reviewer Assessment:

1. **De-emphasize "Edit Effect Framework"**: The gap EditRNA (0.928) vs GB_HandFeatures (0.923) is too small to claim the framework is a major contribution. Frame as one approach among several, not the central claim. The real story is biological, not architectural.

2. **Fix the Sigmoid bug**: EditRNA_rate results with R²=-0.049 should not appear in the paper with a known unfixed bug. Either fix it or omit neural rate prediction entirely and report only GB results.

3. **Asaoka data quality**: 97.6% non-TC in A3A overexpression is a serious concern. Show classification results both WITH and WITHOUT Asaoka. If performance holds, it validates the model. If it drops, acknowledge the contamination.

4. **ClinVar OR=1.33 context**: Need to compare against other disease-variant enrichment methods. OR=1.33 is modest. Frame as "statistically significant" and "complementary to existing annotation" rather than "clinically actionable."

5. **Missing controls**:
   - Add logistic regression baseline (not just GB and neural)
   - Show model is not just learning motif → ablate motif features and show residual signal
   - Independent validation: Hold out one dataset entirely (e.g., Alqassim) as unseen test

6. **A3G sample size**: 179 sites is small. Bootstrap CI covers this, but explicitly discuss power limitations.

### From AI Scientist:

7. **Architecture improvement**: Consider LoRA fine-tuning of last 2 RNA-FM layers (rank 4-8) instead of frozen embeddings. This is the single highest-ROI change for neural performance.

8. **Rate prediction**: Replace MSE with pairwise ranking loss (BPR/Bradley-Terry). Framing shift: predict rank order, not absolute values. Spearman IS the right metric for rankings.

9. **Multi-scale attention**: Inject structure priors (ViennaRNA pairing probabilities) as attention biases into the cross-attention mechanism.

### From Biologist:

10. **"Both" sites — test UCC trinucleotide hypothesis**: Extract -1,0,+1 trinucleotide. If UCC enriched in "Both" vs A3A-only and A3G-only, this confirms the shared motif.

11. **"Neither" sites — APOBEC1 validation**: Check for mooring sequence (WCWN2-4WRAUYANUAU) downstream of edit site. Check liver enrichment in GTEx. Check 3' UTR enrichment (APOBEC1 targets 3' UTR AU-rich regions).

12. **Tissue rate clustering**: Use the 54-tissue × 636 site matrix to cluster both sites and tissues. Expect: A3A sites cluster with immune tissues, A3G with hypoxic/blood, "Neither" with intestine/liver.

---

## Pre-Submission Experiments (Priority Order)

1. [HIGH] **UCC trinucleotide test for "Both" sites** — 1 hour
2. [HIGH] **APOBEC1 feature check for "Neither" sites** — 2 hours (mooring sequence, 3'UTR, liver)
3. [HIGH] **Fix Sigmoid bug in rate_head** — 30 min code change, 2h rerun
4. [HIGH] **Logistic regression baseline** — 1 hour
5. [HIGH] **Motif ablation experiment** — 2 hours (show performance without motif features)
6. [MEDIUM] **Held-out dataset validation** — 3 hours (train on all-but-Alqassim, test on Alqassim)
7. [MEDIUM] **Tissue rate clustering heatmap** — 3 hours
8. [MEDIUM] **LoRA fine-tuning experiment** — 8 hours
9. [LOW] **Pairwise ranking loss for rate** — 4 hours
10. [DONE] **Cross-species comparison (human vs chimp)** — COMPLETED 2026-03-24
    - 3,640 true orthologs analyzed; editing sites 24% more conserved than controls (p=5.94e-37)
    - 99.3% center C conserved; 98.4% motif preserved; 97.4% identical features in chimp
    - See paper/cross_species_comparison.md
11. [DONE] **APOBEC4 full analysis + report tab** — COMPLETED 2026-03-24
    - A4 added to multi-enzyme report with classification, feature importance, structure comparison
    - 181 A4-correlated sites (AUROC=0.876); 21 A4-exclusive sites (LOO AUROC=0.637)
12. [PENDING] **gnomAD gene constraint analysis** — half day
    - Use pLI/LOEUF scores to test if editing-site genes are more constrained

---

## Timeline

| Week | Task |
|------|------|
| 1 | Pre-submission experiments #1-5 |
| 2 | Pre-submission experiments #6-7, fix known bugs |
| 3 | Write manuscript draft (Methods + Results) |
| 4 | Generate all figures, write Introduction + Discussion |
| 5 | Internal review, revisions |
| 6 | Submit to Genome Biology |

---

## Individual Expert Reviews

Full reviews saved to:
- `paper/ai-scientist_review.md`
- `paper/biologist_review.md`
- `paper/pi-advisor_review.md`
- `paper/reviewer_review.md`

---

## Updated Results (March 22, 2026)

### Data Integrity Overhaul
Seven critical data issues discovered and fixed during this session:
1. N-padding artifacts (Kockler 41nt → full 201nt)
2. Genome build mislabeling (Kockler=hg19, Dang=hg38)
3. A3B negatives from wrong genome (hg38→hg19)
4. Missing loop positions for 3,655 A3B negatives
5. NaN leakage in XGBoost StructOnly (NaN treated as missing value signal)
6. A3A using wrong dataset (Kockler v3 → v1 pipeline)
7. Stale Asaoka TC claim (2.4% was genome-build artifact, actual 86.2%)

### Corrected Classification Results
| Enzyme | GB | MotifOnly | StructOnly | EditRNA+H |
|--------|-----|-----------|------------|-----------|
| A3A | 0.907 | 0.869 | 0.747 | 0.880 |
| A3B | 0.574 | 0.596 | 0.526 | 0.810 |
| A3G | 0.841 | 0.706 | 0.814 | 0.893 |
| Both | 0.951 | 0.866 | 0.820 | 0.935 |
| Neither | 0.840 | 0.805 | 0.654 | 0.829 |

### A3B Feature Challenge — Key Finding
The 0.574→0.810 gap is explained by CONTEXT WIDTH, not model architecture:
- Trinucleotide frequencies at ±20nt: 0.736
- All hand features (826-dim): 0.760
- GB on RNA-FM embeddings: 0.818 (matches EditRNA)
A3B is the only enzyme requiring extended sequence context. All others are captured by ±2nt features.

### New Sections Added
- A3B dual role (DNA mutator vs RNA editor comparison table)
- A4 full enzyme analysis in multi-enzyme report (newly discovered by advisor lab)
- Deaminase family comparison (AID/APOBEC)
- COSMIC/AID overlap analysis (complete)
- Clinical deep analysis with SDHB case study
- Unified network interpretability with rescued sites
- Cross-species comparison: human vs chimp (3,640 orthologs, 24% lower divergence at editing sites)
- Mutation coupling analysis (25-35% more C>T variants near editing sites vs controls)
