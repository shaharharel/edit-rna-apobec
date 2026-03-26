Now I have a thorough understanding of the project state, results, existing reviews, and publication plan. Let me provide my strategic framing.

---

## Strategic Framing for a Genome Biology Paper

I have now reviewed the full project: all result files, the existing publication plan, the PI advisor review, the biologist review, the cross-enzyme comparison data, and the Levanon expansion plan. Here is my assessment and concrete guidance.

---

### 1. The Headline Finding

**"Three APOBEC enzymes execute structurally distinct RNA editing programs, and a fourth editing program attributable to APOBEC1 is computationally identifiable from sites previously labeled 'neither.'"**

The headline is NOT about classification accuracy. It is NOT about the edit effect framework. It is about the biological discovery that C-to-U RNA editing in the human transcriptome decomposes into at least four distinct enzyme programs, each with a quantifiable structural signature, and that these signatures predict pathogenic variant enrichment in ClinVar that existing tools miss entirely.

The reason this is the headline: a reviewer at Genome Biology asks "what do I now understand about biology that I did not before?" The answer is that you can fingerprint the editing enzyme from the local RNA structure alone, and that this fingerprint has clinical utility. That is a complete story.

---

### 2. How the Unified Experiment Strengthens Beyond Per-Enzyme Work

The per-enzyme experiments, taken individually, are competent but incremental. A3A classification at AUROC 0.923 is good but not a paper. A3G at n=119 is underpowered. A3B is interesting for the contradiction resolution but niche.

The unified experiment transforms these into something greater through three mechanisms:

**First, the comparative architecture.** When you show that relative_loop_position is the top feature for A3A (importance 0.213), local_unpaired_fraction for A3B, and dist_to_apex for A3G (importance 0.319), the reader immediately grasps that these enzymes have evolved different structural recognition strategies. This is invisible in any single-enzyme analysis. The pairwise classifier data you already have (A3A vs A3G AUROC 0.941, A3B vs A3G 0.959, A3A vs A3B only 0.665) quantifies how distinct these programs are. A3A and A3B partially overlap; A3G is structurally a different animal.

**Second, the "Neither" sites create a discovery moment.** If you demonstrate intestine-enriched tissue editing, near-random dinucleotide context, and structure-independence for the 206 "Neither" sites, you have computationally identified APOBEC1 targets from a database that lacked APOBEC1 annotation. This is genuine discovery, not validation. It elevates the paper from "we classified things well" to "we found something new."

**Third, the clinical comparison across enzymes.** Showing ClinVar pathogenic enrichment for A3A (OR=1.33), A3B (OR=1.55 calibrated), and A3G (OR=1.76 in CC context) as a unified table, contrasted against RNAsee's depletion, is far more convincing than any single enzyme. The pattern that ALL three independent enzyme classifiers independently detect pathogenic enrichment makes the finding robust to concerns about any one enzyme's training data.

---

### 3. Presentation Strategy: Concrete Figure Plan

I largely agree with the 7-figure plan in `paper/publication_plan.md` at `/Users/shaharharel/Documents/github/edit-rna-apobec/paper/publication_plan.md`, but I would restructure it for narrative flow and consolidate where the data is thin. Here is my revised plan:

**Figure 1: The Multi-Enzyme Editing Landscape (Overview)**
- Panel A: Study design schematic. Seven datasets, five enzyme categories, pipeline from raw coordinates to predictions. Keep it simple: boxes and arrows, not a detailed methods diagram.
- Panel B: Sequence logos per enzyme (5 logos stacked: A3A, A3B, A3G, Both, Neither). The visual contrast between A3A's TC, A3G's CC, A3B's degenerate motif, and Neither's near-uniform distribution is immediately striking.
- Panel C: Violin plots of relative_loop_position across all 5 categories. This is the single most informative structural plot you can make. A3G peaks at 1.0, A3A has a moderate 3' bias (median 0.57), A3B is centered at 0.5 (no bias), and Neither should show either flat or structure-independent distribution.
- Panel D: Sample size and source composition stacked bar.

**Figure 2: Three Distinct Structural Programs**
- Panel A: The 3D scatter you proposed (TC%, in-loop%, relative_loop_position) with three enzymes as colored clusters. This is the "signature" plot.
- Panel B: The A3B contradiction resolution. A paired comparison: hairpin frequency (yes, consistent with Butt 2024) vs positional bias (no, consistent with Alonso de la Vega 2023). Show A3B's RLP distribution is symmetric around 0.5 while A3A and A3G are skewed. This resolves a real published discrepancy.
- Panel C: A3G tetraloop detail. Loop size distribution showing the sharp peak at 4 for A3G (median 4.0, no-external), compared to broader distributions for A3A (median 6.0) and A3B (median 6.0-7.0). Combined with the RLP near 1.0, this paints A3G as a tetraloop apex specialist.
- Panel D: Feature importance bar charts (GB_HandFeatures) for all three enzymes side by side. Horizontal bars, top 10 features, colored by feature class (motif/loop/structure-delta). The visual should make clear that the enzymes use DIFFERENT structural features at DIFFERENT weights.

**Figure 3: Classification and Enzyme Specificity**
- Panel A: AUROC summary bar chart across all categories, with bootstrap CI where relevant (A3G, Both, Neither). Include MotifOnly and StructOnly ablations as grouped bars beside the full GB_HandFeatures for each enzyme.
- Panel B: The motif vs structure ablation. For each enzyme, three bars: MotifOnly, StructOnly, Combined. The key message: A3A and A3G get most power from structure (StructOnly AUROC 0.747 and 0.935); A3B gets most from structure (0.800 vs MotifOnly 0.606); Neither is motif-dominated (StructOnly 0.639). This last point is critical evidence for APOBEC1 (which does not use hairpin structure).
- Panel C: Pairwise enzyme discriminability heatmap. The 3x3 matrix of A3A-vs-A3B (0.665), A3A-vs-A3G (0.941), A3B-vs-A3G (0.959) AUROCs. This quantifies how biochemically distinct the programs are.
- Panel D: The top discriminating features per pair. For A3A-vs-A3G, it is motif_CC (0.228) and relative_loop_position (0.214). For A3B-vs-A3G, relative_loop_position dominates (0.268). For A3A-vs-A3B, the best discriminator is motif_UC (0.115) -- they are structurally similar, separable mainly by motif.

**Figure 4: The "Neither" Sites and APOBEC1 Identification (The Discovery Figure)**
This figure should be the surprise of the paper. Structure it as hypothesis-evidence.
- Panel A: Motif profile of "Neither" (near-random dinucleotide) vs the three APOBEC3 enzymes. If TC is approximately 24% and CC approximately 35%, that is close to random expectation. Label it: "No canonical APOBEC3 motif preference."
- Panel B: Tissue editing rate heatmap for "Neither" sites across GTEx tissues, highlighting intestine/colon/liver enrichment. This is the single most important test for the APOBEC1 hypothesis. If you see the signal, this panel carries the figure.
- Panel C: Structure independence. Show that StructOnly AUROC for "Neither" (0.639) is much lower than for all APOBEC3 enzymes (0.747-0.935). APOBEC1 uses a downstream mooring sequence, not hairpin structure.
- Panel D (optional/supplementary): Mooring sequence enrichment analysis in the +4 to +12 region downstream. If AU-rich motifs are enriched, this clinches the APOBEC1 identification.

**Figure 5: ClinVar Pathogenic Enrichment**
- Panel A: OR vs threshold curves for A3A, A3B, A3G (3 lines). Use the raw GB_Full enrichment. Show all three cross the OR=1.0 line with the same direction.
- Panel B: Head-to-head with RNAsee. GB shows enrichment (OR=1.33 for A3A at t=0.5), RNAsee rules-based shows depletion (OR=0.76). This is the "our method finds real signal where existing tools do not" panel.
- Panel C: Bayesian calibration schematic and result. Show the pi_model=0.50 to pi_real=0.019 correction and that enrichment persists after calibration.
- Panel D: Cross-enzyme ClinVar comparison table (can be a formatted table within the figure). A3A: OR=1.33, A3B: OR=1.55 calibrated, A3G: OR=1.76 CC-context. The fact that all three independently show enrichment is the strongest single piece of evidence.

**Figure 6: "Both" Sites and Dual-Enzyme Recognition**
- Panel A: Motif distribution of "Both" sites. Show CC=65.2% dominance. Test UCC/TCC trinucleotide enrichment per the biologist review's Hypothesis 1.
- Panel B: Tissue rate correlation. "Both" sites correlate with A3G tissue profiles (r=0.926) not A3A (r=0.539). This reveals they are biologically A3G-like despite being editable by both.
- Panel C: Feature distributions compared to A3A-only and A3G-only. Are "Both" sites structurally intermediate?
- Panel D: Implications for enzyme attribution. A schematic showing that "Both" sites are biochemically promiscuous but physiologically A3G-edited.

**Figure 7: Rate Prediction and Cross-Dataset Limits**
This figure presents the honest limitations.
- Panel A: Rate prediction performance bar chart (Spearman per model). Be candid: Spearman of 0.122 is low.
- Panel B: Cross-dataset generalization matrix showing off-diagonal failure.
- Panel C: Feature importance for rate prediction. relative_loop_position is still number 1, reinforcing the structure-first theme.
- Panel D: Tissue breadth vs mean rate per category, or the 54-tissue clustering heatmap.

**Table 1 (Main text):** The signature table from your cross-enzyme comparison, expanded to 5 categories: enzyme, n_sites, TC%, CC%, in-loop%, median loop size, RLP median, top classifier feature, classification AUROC, ClinVar OR. This is the paper's summary table.

**Table 2 (Main text):** Full classification results: all models x all enzymes, with means and standard deviations.

---

### 4. Should You Include Neural Model Results?

**Include them, but in a supporting role.** Here is the logic:

The GB_HandFeatures model is the scientific workhorse. It is interpretable, it produces meaningful feature importances, it generates the ClinVar signal, and it performs comparably to or better than neural approaches. For A3A, EditRNA+Features (AUROC 0.935) beats GB (0.923), but as the PI advisor review correctly notes, this difference is likely not statistically significant with 5 folds and has not been formally tested.

Include neural results in a supplementary comparison table. Note that EditRNA+Features achieves the highest AUROC for A3A and briefly discuss what this means (RNA-FM embeddings capture additional signal beyond hand-crafted features). But do NOT make the architecture comparison a central element of the paper. The paper's claim is biological, not architectural.

**Do NOT include the rate prediction neural results with the unfixed Sigmoid bug.** A reviewer who sees R-squared of -0.049 will immediately question your entire analytical pipeline. Either fix the bug and rerun, or report only GB rate results. There is no middle ground here.

---

### 5. Framing the APOBEC1 Discovery

This is the highest-risk, highest-reward element of the paper. Here is how to handle it:

**Frame as computational identification, not definitive proof.** The claim should be: "Sites not attributed to APOBEC3A or APOBEC3G in the Levanon database exhibit features consistent with APOBEC1-mediated editing, including near-random dinucleotide context, absence of hairpin structure preference, and tissue-specific editing enriched in intestine and liver."

**The evidence hierarchy matters.** If you can show all four of these, the claim is strong:
1. Near-random motif (you likely have this -- TC=24%, CC=35%)
2. Intestine/liver tissue enrichment in GTEx rates (this is the key test -- run it before committing to this claim)
3. Structure independence (StructOnly AUROC of 0.639 is consistent)
4. Mooring sequence enrichment downstream (this would be definitive but may be harder to detect)

If you get only items 1 and 3 but NOT item 2, downgrade the APOBEC1 claim to a single sentence in the Discussion. The tissue enrichment is load-bearing evidence. Without it, the "Neither" sites could simply be low-confidence assignments, technical artifacts, or sites edited by any of several poorly characterized APOBEC family members.

**Do NOT call it "discovery of APOBEC1 targets" in the abstract or title** unless the tissue evidence is unambiguous. The reviewers who work on APOBEC1 will hold you to a high standard. Call it "identification of a fourth editing program with APOBEC1-consistent features."

---

### 6. What Makes a Reviewer Say "This Is a Complete Story"

A Genome Biology reviewer evaluates three axes: novelty, rigor, and completeness. Here is what each requires.

**Novelty -- already sufficient.** The multi-enzyme structural fingerprinting, the A3B contradiction resolution, and the cross-enzyme ClinVar analysis are collectively novel. The APOBEC1 identification (if the tissue data supports it) would elevate novelty further.

**Rigor -- needs targeted strengthening.** From the PI advisor and reviewer assessments, these are non-negotiable before submission:

1. *Statistical testing of model comparisons.* Add paired Wilcoxon signed-rank or DeLong tests for AUROC comparisons between models within each enzyme. With 5 folds the power is low, so you will likely find that most differences are not significant. That is fine -- report it honestly and let the feature importance analysis carry the interpretability argument.

2. *The AUROC discrepancy in the A3A JSON.* The classification JSON at `/Users/shaharharel/Documents/github/edit-rna-apobec/experiments/apobec3a/outputs/classification_a3a_5fold/classification_a3a_5fold_results.json` shows mean_auroc of 0.9075 for GB_HandFeatures, but the fold AUROCs (0.917, 0.914, 0.929, 0.923, 0.934) average to 0.923. This is a data integrity issue. Fix the JSON or clarify which numbers are canonical.

3. *Logistic regression baseline.* Add a simple logistic regression on the same 40-dim features for every enzyme. If LR achieves AUROC 0.90+ for A3A, the signal is truly linear and GB's tree-based flexibility is not needed. If LR is substantially worse, GB's nonlinear interactions between motif and structure matter. Either result is informative.

4. *One held-out validation.* Hold out the entire Alqassim dataset (128 A3A sites) as an unseen test set. Train on Asaoka+Levanon+Sharma only. Report the held-out AUROC. If it holds, this eliminates the "all cross-validation, no true external validation" critique.

5. *Address the Asaoka TC fraction.* Run classification with and without Asaoka. Report both. If performance holds without Asaoka, it validates the model. If it drops, acknowledge it.

**Completeness -- what closes the loop.** The reviewer's mental checklist for "complete" is:

- Do you define the substrate rules for each enzyme? **Yes** (motif + structure signatures).
- Do you show these rules are distinct? **Yes** (pairwise classifiers, feature importance).
- Do you validate on external data? **Partially** (ClinVar enrichment serves as genome-scale validation, but add the held-out dataset).
- Do you demonstrate clinical relevance? **Yes** (ClinVar OR across all enzymes, RNAsee comparison).
- Do you address limitations honestly? **Must add** (rate prediction weakness, sample size for A3G and "Neither," overexpression-based enzyme attribution caveats).
- Do you provide a resource? **Yes** (trained models, scored ClinVar variants, curated dataset).

The one piece that transforms "thorough analysis" into "complete story" is the tissue clustering with the 54-tissue GTEx data. If you produce a heatmap showing that 636 Levanon sites x 54 tissues cluster into biologically interpretable groups (A3A sites with immune tissues, A3G with blood/hypoxia, "Neither" with intestine), that single figure ties everything together. It connects sequence-level classification to tissue-level biology and validates the enzyme attribution scheme using orthogonal evidence.

---

### Summary of Priorities Before Writing

Ranked by impact on the paper's defensibility:

| Priority | Task | Time | Impact |
|----------|------|------|--------|
| 1 | Tissue enrichment analysis for "Neither" sites (APOBEC1 test) | 2-3h | Determines whether Figure 4 exists |
| 2 | 54-tissue x 636-site clustering heatmap | 3h | Ties the entire narrative together |
| 3 | Fix AUROC discrepancy in A3A JSON | 30min | Data integrity |
| 4 | UCC trinucleotide test for "Both" sites | 1h | Strengthens Figure 6 |
| 5 | Logistic regression baseline for all enzymes | 2h | Pre-empts reviewer critique |
| 6 | Held-out Alqassim validation | 2h | Addresses external validation concern |
| 7 | Fix Sigmoid bug in rate_head and rerun | 2.5h | Fixes known bug before publication |
| 8 | Statistical tests for model comparisons | 2h | Rigor requirement |
| 9 | Classification with/without Asaoka | 2h | Addresses data quality concern |

If the tissue analysis for "Neither" sites does NOT show intestine/liver enrichment, remove the APOBEC1 framing and consolidate Figures 4 and 6 into a single "Levanon expansion" figure. The paper still works with 6 figures. If it DOES show the signal, you have a 7-figure paper with a genuine discovery component, and Nature Communications becomes a realistic target.

---

### The Narrative Arc in One Paragraph

The paper opens by framing the selectivity problem: millions of cytidines, thousands edited, why? It introduces the three APOBEC3 enzymes as the primary C-to-U editors and argues that existing predictors fail because they treat editing as a single phenomenon. Act 1 reveals that each enzyme has a distinct structural signature, quantified through 40-dimensional feature analysis and gradient boosting classifiers that achieve AUROC 0.83-0.93 per enzyme. Act 2 resolves the published A3B structural controversy and identifies A3G as a tetraloop apex specialist, then shows that sites edited by both enzymes are biologically A3G-like. Act 3 demonstrates clinical relevance: all three enzyme classifiers independently detect pathogenic enrichment in 1.69M ClinVar variants, a signal invisible to the current state-of-the-art (RNAsee). The paper closes with the identification of a putative APOBEC1 editing program in the "Neither" category, suggesting that the full landscape of C-to-U editing in human RNA involves at least four enzyme programs, each with distinct structural logic and tissue specificity.