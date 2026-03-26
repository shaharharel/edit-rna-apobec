# Reviewer Review

I now have enough information for a thorough review. Let me compile the findings.

---

# Peer Review: "Structural Determinants of APOBEC-Mediated C-to-U RNA Editing"

## Overall Assessment

This manuscript presents a computational framework for predicting APOBEC-mediated C-to-U RNA editing sites across three enzymes (A3A, A3B, A3G), identifies loop geometry as a dominant structural feature, and reports ClinVar pathogenic enrichment among predicted editing sites. The work is ambitious in scope and commendably transparent about its known bugs and limitations. However, several major concerns regarding data quality, statistical interpretation, unsupported claims, and missing controls would need to be addressed before publication.

**Verdict: Major Revision**

---

## 1. Major Concerns

### 1.1 The "Edit Effect Framework" Claim is Unsupported by the Data

The manuscript title and framing invoke the "causal edit effect framework" as a core contribution, yet the actual evidence contradicts this claim. Examining the classification results:

| Model | Mean AUROC |
|-------|-----------|
| GB_HandFeatures (40-dim hand features) | 0.907 |
| GB_AllFeatures (hand + embeddings) | 0.922 |
| EditRNA-A3A (edit embeddings) | 0.928 |
| EditRNA+Features (edit embeddings + hand) | 0.935 |
| SubtractionMLP (subtraction baseline) | 0.841 |
| SubtractionMLP+Features | 0.886 |

While EditRNA+Features does outperform SubtractionMLP+Features (0.935 vs 0.886), the critical comparison should be EditRNA vs. GB_HandFeatures -- and the gap is only 0.021 AUROC (0.928 vs 0.907). More importantly, EditRNA+Features (0.935) barely exceeds GB_AllFeatures (0.922), suggesting that the 640-dim RNA-FM embeddings provide modest additional signal over 40 hand-crafted features. The fact that GB with hand features nearly matches a pretrained language model is more interesting scientifically than the edit effect claim.

For rate prediction, the edit effect framework performs even worse: GB_HandFeatures achieves Spearman=0.121 with R-squared=0.014, while EditRNA_rate achieves Spearman=0.137 but R-squared=-0.049 (worse than predicting the mean). This is acknowledged as a Sigmoid bug, but shipping results with a known bug that invalidates the primary comparison is not acceptable for a methods paper. The edit effect claim cannot rest on classification alone when the rate task is broken.

**The SubtractionMLP results (AUROC=0.841) are actually not a fair baseline for the "edit effect" comparison.** SubtractionMLP uses RNA-FM embeddings of the original and edited sequences with subtraction -- it tests whether RNA-FM captures the edit effect through general-purpose representations. But the actual scientific question is whether structured edit embeddings capture something that independent prediction and differencing cannot. This requires comparing against: (a) independent classifiers on original and edited features separately, and (b) simple concatenation of original and edited features without any "edit-aware" architecture. ConcatMLP and PooledMLP have NaN results -- these essential baselines did not run.

### 1.2 GB_HandFeatures AUROC Discrepancy

The manuscript summary states GB_HandFeatures AUROC=0.923 for A3A, but the actual JSON results show **mean_auroc=0.907** with a note "Rerun with correct loop features." The individual fold AUROCs range from 0.914 to 0.934. The discrepancy between the claimed 0.923 and actual 0.907 needs clarification. The 0.923 figure may come from a different run configuration or the ClinVar-specific GB_Full model (which has 46 features including baseline structure features, not 40). This inconsistency undermines confidence in the reported numbers.

### 1.3 ClinVar Enrichment: Effect Size is Marginal and Potentially Confounded

The headline ClinVar result (OR=1.33 at P>=0.5) sounds impressive, but closer examination of the calibrated results tells a different story.

After Bayesian calibration with the Tier 1 prior (pi_real=0.019), the enrichment at P_cal>=0.5 is **OR=1.0008, p=0.93** -- entirely non-significant. At P_cal>=0.25 it is OR=1.053, barely above 1. The enrichment only becomes notable at very low calibrated thresholds (P_cal>=0.001: OR=1.66), which correspond to essentially all variants scored above baseline.

This means the ClinVar enrichment is driven by scoring a very large number of variants slightly above background, not by identifying a meaningful subset of high-confidence editing candidates with elevated pathogenicity. An OR of 1.05-1.33 across 600K+ variants is statistically significant purely due to sample size, but it is not clinically actionable or biologically specific. For comparison, known cancer driver mutation enrichment analyses typically show ORs of 3-10+ in well-defined categories.

The claim that "rules-based (RNAsee) shows depletion" also needs scrutiny. The RNAsee RF shows OR=0.37 at P_cal>=0.001 but with a massive confidence interval (0.04-3.57, p=0.37) -- this is not a meaningful depletion. The comparison is not apples-to-apples because RNAsee was designed as a rules-based scoring approach, not a machine learning classifier trained on the same data.

### 1.4 Asaoka 2019 Data Quality is a Fundamental Problem

Asaoka 2019 contributes 4,933 sites to the A3A training set, but 97.6% of these are non-TC motif. APOBEC3A has strong TC-context preference. This means either: (a) the vast majority of Asaoka sites are not genuine A3A targets (ADAR contamination or other artifacts), or (b) A3A editing has much broader motif specificity than believed.

Either way, training on Asaoka data conflates A3A signal with noise. The motif analysis confirms this: Asaoka positives have 2.4% TC, while Baysal has 94.4% TC. A model trained on both will learn a mixed signal. The TC-motif reanalysis (restricting to TC sites only) shows that Baysal alone drives rate prediction (Spearman=0.134 on TC-only Baysal test, n=641). Without Asaoka-specific ablation experiments showing what the model learns from 4,800 non-TC sites, the dataset composition is a serious confound.

### 1.5 Half the Models Failed to Run

The classification results show NaN for PooledMLP, ConcatMLP, CrossAttention, DiffAttention, and DiffAttention+Features. These are 5 of 13 models that did not produce results. Presenting incomplete results as a comprehensive comparison is misleading. ConcatMLP in particular is essential as the naive alternative to edit-aware architectures.

---

## 2. Data Quality Assessment

### 2.1 A3G: n=119 is Insufficient

A3G classification achieves AUROC=0.931 with bootstrap CI [0.889, 0.961]. While AUROC appears high, with only 119 positive and 119 negative examples, 5-fold CV produces folds of ~48 samples. The fold-level variance confirms this: AUROCs range from 0.889 to 0.980. The wide CI makes meaningful comparison with A3A or A3B impossible. Additionally, GB_HandFeatures and GB_AllFeatures produce *identical* results for A3G (same fold AUROCs to every decimal place), suggesting that the additional embedding features added no signal at all -- consistent with the model overfitting on the small dataset.

The claim of "three distinct editing programs" based on 119 A3G sites from a single experiment (Dang 2019, NK cells under hypoxia) is overreach. This is one experimental condition in one cell type. The A3G "program" could be a Dang 2019 artifact rather than a general A3G property.

### 2.2 Negative Controls

Negatives are genome-sampled cytidines from genes containing positive sites. This is a reasonable choice, but the tiered structure (Tier 1: all C, Tier 2: TC-motif, Tier 3: TC in stem-loops) creates an implicit confound: the final training set uses a subsample of 2,000 Tier 2 and 1,000 Tier 3, meaning 67% of negatives have TC context. Yet only ~40% of positives have TC context (after including the problematic Asaoka sites). The negative set's motif distribution does not match the positive set's, and this mismatch -- documented as a known bug for ClinVar -- likely affects the primary classifier too.

### 2.3 Kockler/Dang Data Issues

Kockler data uses BT-474 transcriptomic coordinates and short (41/31nt) context windows padded with N to 201nt. ViennaRNA treats N as fixed-unpaired, making loop geometry features from these sites non-comparable to sites with full 201nt genomic context. The manuscript acknowledges this but does not perform sensitivity analyses excluding Kockler sites.

---

## 3. Statistical Rigor

### 3.1 5-Fold CV Without Nested CV

The primary evaluation uses 5-fold KFold CV. For GB models (XGBoost), hyperparameters appear to be set globally (not tuned per fold with an inner loop). If default or pre-specified hyperparameters are used, this is acceptable but should be explicitly stated. However, for neural models (EditRNA), early stopping on a validation split within each fold (80/20 inner split) introduces a dependency between model selection and evaluation that nested CV would address.

### 3.2 ClinVar Multiple Testing

The ClinVar analysis reports p-values from Fisher exact tests across multiple thresholds (0.3, 0.4, 0.5, 0.6, 0.7 for A3B; 8 calibrated thresholds for A3A). No correction for multiple testing is applied. Given 1.68M scored variants, even the Fisher test denominators are enormous, making virtually any directional difference "significant." The p-values are uninterpretable without effect size context, which the manuscript partially provides via OR and CI, but the text emphasizes p-values (e.g., "p<1e-40") in a way that overstates the finding.

### 3.3 Rate Prediction Signal is Near-Zero

The best rate predictor (GB_HandFeatures) achieves Spearman=0.121, R-squared=0.014. This means the model explains 1.4% of variance in editing rates. The cross-dataset rate prediction explicitly "fails" (acknowledged). These results indicate that editing rate is either not well predicted by local sequence/structure features, or that the rate measurements across datasets are too noisy/incompatible for meaningful regression. Presenting Spearman=0.121 as a positive result is misleading; this is essentially unpredictive.

---

## 4. Missing Controls

### 4.1 Essential Missing Baselines

1. **Logistic regression on the same 40 features.** Without this, we cannot assess whether XGBoost's nonlinearity contributes anything. If logistic regression achieves AUROC=0.89, the story changes substantially.

2. **Motif-only classifier.** The A3A classification does not include a motif-only baseline (though A3B and A3G do). For A3B, MotifOnly AUROC=0.606; for A3G, MotifOnly AUROC=0.689. These numbers are important for quantifying how much structure adds beyond motif. For A3A, we can only infer this indirectly.

3. **Sequence length and GC content controls.** No analysis checks whether predicted editing probability correlates with trivial sequence properties.

4. **Independent hold-out validation.** All reported metrics use internal CV. No external dataset is held out completely. The cross-dataset generalization experiment shows moderate off-diagonal performance for classification (e.g., train Asaoka, test Levanon: AUROC=0.904 for GB_HandFeatures), which partially addresses this. But for rate prediction and ClinVar, no independent validation exists.

5. **ConcatMLP results.** This is the most important missing baseline -- simple concatenation of original and edited embeddings without edit-aware architecture. It failed to run.

### 4.2 Confound Controls

1. **Gene identity leakage.** Multiple sites from the same gene appear in both training and test folds. Gene-stratified CV splits would address whether the model generalizes across genes or memorizes gene-level features.

2. **Dataset identity leakage.** The cross-dataset AUROC matrix partially addresses this, but the primary 5-fold CV mixes all datasets. A dataset-stratified analysis is needed.

3. **Editing rate vs. detection sensitivity.** Different datasets have different detection sensitivities (ADAR overexpression vs. endogenous). The "editing rate" may reflect experimental conditions more than biology.

---

## 5. Novelty Assessment

### 5.1 RNAsee Comparison

The comparison with RNAsee (Van Norden et al. 2024) is informative but limited. RNAsee achieves AUROC=0.871 on this data vs. GB_Full AUROC=0.938 on ClinVar. However: (a) the comparison uses different training data (RNAsee was trained on its own curated set), (b) the RNAsee replication uses a default sklearn RandomForest which may not match the original implementation, and (c) RNAsee's rules-based component is not compared separately from its RF component.

### 5.2 Edit Effect Framework

As discussed in Major Concern 1.1, the edit effect framework is not convincingly demonstrated. What IS demonstrated is that ViennaRNA structural features (particularly loop geometry) are strong predictors, which is a genuine contribution. But this is feature engineering, not a novel framework. The framing should be adjusted accordingly.

### 5.3 The A3B "Contradiction Resolution"

This claim (resolving Butt 2024 vs. Alonso de la Vega 2023) is interesting but requires careful framing. Computational analysis can *suggest* a resolution but cannot resolve a biological contradiction without experimental validation. The specific resolution should be clearly stated with supporting evidence from the data.

### 5.4 Comparison with ESM/Protein Language Models

No comparison with modern nucleotide foundation models beyond RNA-FM is provided. RNA-FM is from 2022; newer models (e.g., RNA-BERT variants, Nucleotide Transformer) may provide better representations. The paper should at minimum discuss why RNA-FM was chosen and acknowledge this limitation.

---

## 6. Clinical Claims Assessment

### 6.1 OR=1.33 is Not Clinically Actionable

An odds ratio of 1.33 for pathogenic enrichment means that among sites scored P>=0.5, the pathogenic rate rises from ~10% to ~11.3%. This is not useful for clinical variant interpretation. For context, CADD scores achieve ORs of 5-20+ for pathogenic enrichment at stringent thresholds. ClinGen variant classification requires much stronger evidence.

### 6.2 Bayesian Calibration

The calibration itself is correctly implemented. However, the choice of pi_model=0.50 (assuming scale_pos_weight=3 on 1:3 data creates an effective 50% prior) is an approximation. The actual effective prior depends on the loss landscape and model capacity, not just the weighting factor. Sensitivity analysis with pi_model=0.25 and pi_model=0.50 is provided, which is good practice.

The key finding -- that enrichment vanishes at calibrated P_cal>=0.5 (OR=1.0008) -- should be the headline result, not buried. This means the model cannot identify a high-confidence subset of editing candidates enriched for pathogenic variants at realistic prevalence.

### 6.3 What Would Be Needed for Clinical Utility

Minimum requirements: (a) OR>3 at calibrated thresholds, (b) validation on an independent ClinVar snapshot or clinical cohort, (c) comparison with established pathogenicity predictors (CADD, REVEL, AlphaMissense), (d) functional validation of top candidates. The current work is far from clinical utility and should not be framed as having clinical implications.

---

## 7. Presentation Recommendations

### 7.1 Manuscript Structure

Four separate HTML reports are not a manuscript. The work should be organized as:

**Main Figures (suggested 5-6):**
1. Overview schematic: data sources, pipeline, multi-enzyme scope
2. Classification performance: A3A/A3B/A3G side-by-side, with motif-only and structure-only ablation bars. This is the strongest result.
3. Feature importance: relative_loop_position as #1 feature across enzymes, with structural schematic showing where editing occurs in stem-loops
4. Multi-enzyme motif/structure comparison: the three editing programs (TC vs CC, loop position bias, tetraloop preference). This is scientifically interesting.
5. ClinVar analysis: calibrated enrichment curves with appropriate caveats
6. Cross-dataset generalization matrix

**Supplementary:**
- Full model comparison tables (all 13 models)
- Rate prediction results (clearly labeled as "weakly predictive")
- Embedding visualizations (currently overanalyzed for modest insight)
- All detailed statistics

### 7.2 Key Narrative

The manuscript should refocus from "edit effect framework with clinical implications" to "structural determinants of APOBEC editing site selection." The strongest contributions are:
1. Loop geometry, particularly relative_loop_position, is the dominant predictor across enzymes
2. Three enzymes have distinct but interpretable structural programs
3. Structure-augmented ML outperforms rules-based approaches
4. Modest but significant ClinVar pathogenic enrichment exists at liberal thresholds

---

## 8. Verdict and Recommendations

### Decision: Major Revision

### Required Revisions (Before Resubmission)

1. **Fix the EditRNA_rate Sigmoid bug** and re-evaluate all rate prediction claims. This is non-negotiable for a methods paper.

2. **Run the missing baselines** (PooledMLP, ConcatMLP, logistic regression on 40 features, motif-only for A3A). Without ConcatMLP, the edit effect claim is unsubstantiated.

3. **Perform gene-stratified CV** to verify no gene-level leakage.

4. **Address the Asaoka data quality issue** explicitly: either remove non-TC Asaoka sites from the primary analysis and present as sensitivity analysis, or provide a biological argument for why non-TC sites are genuine A3A targets.

5. **Reframe the ClinVar results honestly.** Lead with the calibrated results (OR~1.0 at P_cal>=0.5). The enrichment exists but is marginal and not clinically actionable.

6. **Drop or substantially qualify the "edit effect framework" claim** unless ConcatMLP and other baselines are run and clearly show the edit-aware architecture provides unique value.

7. **Reconcile the AUROC numbers** reported in the summary (0.923) vs. JSON results (0.907).

### Recommended Revisions

8. Add a simple logistic regression baseline to establish the contribution of XGBoost nonlinearity.
9. Perform dataset-identity-aware CV splits.
10. Compare with at least one additional RNA language model.
11. Add gene-stratified rate prediction to assess whether cross-dataset failure reflects biological differences or technical artifacts.
12. Present A3G results with appropriate uncertainty bounds and the caveat that they derive from a single experiment.

### Journal Recommendation

In current form: not suitable for Nature Methods or Genome Biology. The edit effect framework is undersupported, the clinical claim is overstated, and fundamental baselines are missing.

After revision (if edit effect is substantiated and framing is corrected): **Nucleic Acids Research** or **Bioinformatics** (computational biology category). If the A3B contradiction resolution is developed into a compelling biological story with independent support: **Genome Biology** is possible.

If the edit effect framework is dropped and the paper is reframed as a feature engineering and structural biology study: **RNA** or **NAR Genomics and Bioinformatics** would be appropriate.

---

## Key Files Examined

- `/Users/shaharharel/Documents/github/edit-rna-apobec/experiments/apobec3a/outputs/classification_a3a_5fold/classification_a3a_5fold_results.json` -- primary classification metrics
- `/Users/shaharharel/Documents/github/edit-rna-apobec/experiments/apobec3a/outputs/rate_5fold_zscore/rate_5fold_results.json` -- rate prediction metrics
- `/Users/shaharharel/Documents/github/edit-rna-apobec/experiments/apobec3a/outputs/clinvar_calibrated/calibrated_enrichment_results.json` -- ClinVar calibrated enrichment
- `/Users/shaharharel/Documents/github/edit-rna-apobec/experiments/apobec3a/outputs/clinvar_prediction/clinvar_prediction_results.json` -- ClinVar raw results
- `/Users/shaharharel/Documents/github/edit-rna-apobec/experiments/apobec3a/outputs/classification_a3a_5fold/feature_importance_cls_gb_hand.csv` -- feature importance rankings
- `/Users/shaharharel/Documents/github/edit-rna-apobec/experiments/apobec3a/exp_clinvar_calibrated.py` -- Bayesian calibration implementation
- `/Users/shaharharel/Documents/github/edit-rna-apobec/experiments/apobec3a/exp_classification_a3a_5fold.py` -- classification experiment code
- `/Users/shaharharel/Documents/github/edit-rna-apobec/experiments/apobec3g/outputs/classification/classification_a3g_results.json` -- A3G results
- `/Users/shaharharel/Documents/github/edit-rna-apobec/experiments/apobec3b/outputs/clinvar/a3b_clinvar_results.json` -- A3B ClinVar results
- `/Users/shaharharel/Documents/github/edit-rna-apobec/experiments/apobec3a/outputs/tc_motif_reanalysis/tc_motif_reanalysis_results.json` -- TC motif reanalysis