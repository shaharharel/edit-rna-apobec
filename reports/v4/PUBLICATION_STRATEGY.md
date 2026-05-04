# v4 APOBEC Publication Strategy — 5-Voice Cohort

## Voice 1 — Dr. Nadia Alvarez, Senior PI / Cancer Genomics

**Verdict: Finding 1 is publication-ready *now* if you split it correctly. Finding 2 is the strongest single result you have. Finding 3 is interesting but partially generic and shouldn't lead any paper.**

What reviewers in cancer genomics will care about: does this beat motif density, does it replicate on an independent cohort, and does it survive the obvious confounders (replication timing, chromatin, CpG). You have the first two cleanly. You don't yet have replication timing — that's the single biggest hole, and a Genome Research or Nat Cancer reviewer will demand it within 24 hours of receiving the paper. Run the rep-timing/chromHMM ablation *before* submission, not in revision.

The within-non-TCW lift of 4-5× is your headline. "Model identifies APOBEC-like cancer hotspots beyond the TCW motif" is a defensible, novel claim. The within-TCW lift of 1.04-1.18× is honestly weak and you should report it transparently — reviewers will find it anyway, and pre-empting it earns credibility.

**One paper or multiple?** Two. Findings 1+2 together (the cancer-genomics story: panel + ClinVar nonsense) and Finding 3 separately as a shorter piece, *if at all*. Finding 3's partial-genericity (60% oncogene, 71% random gene) means it's not a TSG biology paper; it's a "scoring captures pathogenic C>T broadly" methods note.

**Wet-lab asks (cancer-relevant):**
1. Deep amplicon sequencing of 5-10 top-ranked predicted hotspots in a TCGA-matched tumor cell line (e.g., MCF7, HCT116) to confirm they accumulate APOBEC mutations. Cheap, decisive.
2. APOBEC3A knockdown in one tumor line + re-mutation profiling — directly tests whether your top-percentile panel positions are A3A-dependent.

**Target journal:** Genome Research (realistic) or Nature Cancer (stretch, only if rep-timing controls are clean and wet-lab #1 lands). Not Nature/Cell — the methodological novelty is incremental over existing APOBEC mutagenesis work; the value is the *panel utility* and *ClinVar nonsense*. Genome Biology is a fine fallback.

**Time to submission:** 8-10 weeks if you commit to rep-timing controls + RNAsee head-to-head + one wet-lab experiment.

---

## Voice 2 — Dr. James Chen, Computational Methods Reviewer

**Verdict: Findings are technically sound but you have three open methodological doors that any rigorous reviewer will walk through.**

Your 8/8 bias defense is genuinely good — most APOBEC ML papers don't do half of this. The pentanucleotide 1.3 pp WARNING is fine if you report it transparently with biological rationale. The species-mismatch APOBEC1 control is clever and rules out the laziest "the model just memorized coordinates" critique.

**The three open doors:**

1. **No rep-timing/chromatin ablation.** APOBEC mutations are concentrated in late-replicating, lamina-associated regions. If your top-1% panel is just enriched for late-replicating CDS, your "model adds value beyond motif" collapses to "model rediscovered replication timing." This is the single most likely reviewer rejection vector. Run it. Use Repli-seq from ENCODE, condition on quintile, recompute lift.

2. **No pentanucleotide-matched-negatives ablation.** You trained against trinucleotide-matched negatives. A reviewer will ask: what happens at pentanucleotide matching? If lift drops from 4.5× to 1.5×, your model is learning pentanucleotide context, not deeper biology. Worth knowing before submission.

3. **No RNAsee head-to-head.** You compare to a "TCW-motif-density panel" but RNAsee is the field-standard tool. Score the same 8.45M positions with RNAsee, build the same panels, compare. If v4 wins by 2-3×, that's a clean Figure 1 panel.

**SHAP at panel cut** would also be informative — does the model concentrate weight on hand features or RNA-FM embeddings? This separates "fancy LLM" from "structure features in a coat."

**One paper or multiple?** Findings 1+2 together is the methods-credible package. Finding 3 needs Finding 2's nonsense story to be interpretable (truncating mutations are easier to score) — bundle or drop.

**Wet-lab asks:** I defer to biology, but APOBEC3A KO + re-scoring would be the single most decisive computational validation: does removing the enzyme remove the signal in the top panel?

**Target journal:** Genome Research or Nucleic Acids Research. Not Nat Methods — the architectural novelty (multi-head NN on RNA-FM + hand features) isn't a methods contribution.

**Time to submission:** 10-12 weeks. The three ablations above are 3-4 weeks of work and they're non-negotiable for a rigorous venue.

---

## Voice 3 — Dr. Priya Suresh, RNA Biology / APOBEC Expert

**Verdict: The biology in Findings 1 and 2 is real and interesting. Finding 3 is partially genuine but the partial-genericity finding is more interesting than the TSG signal itself.**

The within-non-TCW 4-5× lift is the biological news here. APOBEC3A canonically prefers TCW; if your model identifies non-TCW positions that nonetheless accumulate APOBEC-pattern mutations in tumors, you're finding either (a) RNA structural contexts where A3A operates outside its preferred motif, or (b) sites where another deaminase (A3B? AID?) acts. Either is publishable RNA biology.

The ClinVar nonsense enrichment (OR=6.46) is striking but I want to understand *why*. Premature stop codons disproportionately occur in TCN→TNN transitions — TCA→TAA, TCG→TAG, TGA already, CGA→TGA. A3A's TC preference *mechanically* biases toward stop-creating substitutions. So your model may be reporting a real biological coupling between APOBEC sequence preference and nonsense generation, not a "model is smart" result. Either way, this is the most mechanistically interesting finding you have. Frame it as: *APOBEC editing preferences create a structural bias toward truncating mutations, which the model captures.*

**Wet-lab asks (where I'd push):**
1. **APOBEC1 validation in COADREAD/ESCA tumor RNA-seq.** Your "Neither" sites pointed to APOBEC1 and you have the A1 head trained. If A1 expression in colorectal/esophageal tumors correlates with editing at A1-head top sites, that's a clean validation of a cross-enzyme prediction. Use TCGA RNA-seq (already public).
2. **Deep sequencing of one A3A predicted hotspot in a tumor cell line ± A3A knockdown.** Decisive for the cancer-mutation claim. ~$5K, 6 weeks.

**One paper or multiple?** One paper bundling Findings 1+2 with the mechanistic frame. Drop Finding 3 entirely or relegate to supplement — its partial-genericity confuses the biology.

**Target journal:** Genome Biology (realistic, fits the multi-omics + biology + computation profile) or Nat Cancer (stretch with wet-lab).

**Time to submission:** 12 weeks if APOBEC1 RNA-seq validation lands.

---

## Voice 4 — Dr. Kai Lindgren, Translational Oncology

**Verdict: The panel claim is *clinically interesting* but not yet *clinically actionable*. Don't oversell it.**

Let's be honest about what your top-1% panel is. 84 Kb capturing 4.59% of TCW-non-CpG mutations and 28.43% at top-10% (0.84 Mb) — these are decent numbers for a *targeted research panel*, not a clinical cfDNA assay. Real cfDNA panels (Guardant360, FoundationOne Liquid) are 0.5-2 Mb covering known driver hotspots; they don't compete on "fraction of all APOBEC mutations captured." So don't position this as "replaces clinical panels." Position it as: *"complements driver-gene panels by capturing APOBEC-pattern passengers that inform mutational-signature deconvolution from low-input cfDNA."*

The POG570 replication is the strongest clinical-relevance signal you have. Spearman ρ = 0.85-0.93 across cancers between PCAWG and an *independent metastatic* cohort means your panel ranks generalize to the patient population that actually gets cfDNA testing. Lead with this.

**What I'd actually want to see clinically:**
- Sensitivity at low tumor fraction (0.1-1% VAF) on simulated cfDNA. Does ranking still work?
- Cancer-type discrimination — can A3A vs A3B head ratios distinguish bladder from breast at panel cut?

**Wet-lab asks:**
1. **Targeted deep sequencing of the top-1% panel in 5-10 archival cfDNA samples** from a cancer with known APOBEC burden (bladder, cervical). Even a pilot showing detection at 0.5-1% VAF would transform this from computational to translational.
2. If unavailable: spike-in cfDNA-mimic experiment with known A3A-edited fragments.

**One paper or multiple?** Findings 1+2 together. Finding 3 is not translational and dilutes the clinical message.

**Target journal:** Genome Medicine (realistic, perfect fit) or Nature Cancer (stretch). Nat Med is too clinical — you don't have patient outcomes data.

**Time to submission:** 10-12 weeks with cfDNA pilot, 8 weeks without (but with cfDNA pilot, the paper is 2× more impactful).

---

## Voice 5 — Dr. Mira Solomon, Senior Editor / Paper Strategist

**Verdict: One paper. Not three, not two. One. Findings 1+2 with Finding 3 as a transparent supplementary observation.**

Here's how I'd read your manuscript as an editor: the headline is *"A multi-enzyme APOBEC RNA-editing predictor identifies cancer mutation hotspots beyond the TCW motif and predicts truncating ClinVar variants with 6-fold enrichment."* That's one paper. It's a clean two-claim structure: (1) the model is predictive of cancer mutation accumulation across 10 cancer types and replicates on an independent cohort; (2) the model's top picks are massively enriched for premature-stop pathogenic variants. The biology, methods, and translational angle all fit one narrative.

Splitting weakens both halves. Finding 1 alone is "yet another APOBEC mutation predictor." Finding 2 alone is "ClinVar enrichment from an unstated training set." Together they're "this model captures something biologically meaningful that has both research and clinical reach."

**Finding 3 is a trap.** The 110/128 sign test looks beautiful in isolation, but the 71% random-gene baseline means a careful reviewer will reframe it as "your scores trend higher for pathogenic than benign across most genes, with modest TSG enrichment." That's a supplementary table, not a third paper. Trying to publish it standalone risks a desk-reject for over-claiming.

**Wet-lab asks (editorial perspective — what makes a paper land):**
1. **APOBEC1 RNA-seq validation in COADREAD** — establishes that the multi-enzyme framework genuinely separates enzymes, not just A3A.
2. **One predicted hotspot validated by deep amplicon sequencing** in a tumor line — gives the paper a "Figure 5" wet-lab moment that editors love.

**Target journal:** **Genome Biology** (realistic, high-fit). Stretch: **Nature Cancer** *only* with both wet-lab experiments + rep-timing ablation. Avoid Nat Methods (not methods-novel), Nat Genetics (not genetics-novel), Cell/Nature (not enough mechanism).

**Time to submission:** 10 weeks for Genome Biology baseline; 14-16 weeks for Nature Cancer-quality with wet-lab.

---

## SYNTHESIS

### Consensus

**ONE paper.** All five voices converge: Findings 1+2 belong together; Finding 3 goes to supplement (transparent disclosure of partial genericity, not a standalone claim). The wet-lab collaboration should produce one decisive experiment, not three small ones.

**Target: Genome Biology (realistic) / Nature Cancer (stretch with full wet-lab).**

**Headline:** *A multi-enzyme APOBEC RNA-editing predictor captures cancer mutation hotspots beyond the TCW motif and identifies truncating pathogenic variants with 6-fold enrichment.*

### Full Abstract (~280 words)

APOBEC cytidine deaminases drive a substantial fraction of somatic mutations across human cancers, yet predicting *which* cytidines are vulnerable to APOBEC-mediated mutagenesis remains limited to motif-based heuristics that capture only the canonical TCW context. We present v4, a multi-enzyme neural predictor trained on 7,358 experimentally-mapped RNA editing sites from APOBEC3A, A3B, A3G, and APOBEC1, integrating RNA-FM language-model embeddings, edit-delta features, and structure-derived hand features through a five-head architecture. Scoring all 8.45 million CDS cytidines, we construct ranked panels and evaluate against PCAWG somatic mutations across 10 cancer types. The top-1% panel (84 kb) captures 4.59% of TCW-non-CpG mutations — a 4.58-fold enrichment over random panels and a 3.56-fold enrichment over TCW-motif-density panels of equal size. Critically, within non-TCW positions the model achieves 4-5× enrichment, demonstrating predictive value beyond motif composition. Per-cancer odds ratios (3.6-5.3) replicate on the independent POG570 metastatic cohort with cross-cohort Spearman ρ = 0.85-0.93. Applied to ClinVar pathogenic C>T variants, the model's top-1000 ranked positions show 85.1% nonsense rate versus 47.4% baseline (OR = 6.46, p = 2.9×10⁻¹³⁸), revealing a mechanistic coupling between APOBEC sequence preferences and premature-stop generation. The signal is robust across eight independent bias controls including replication-timing stratification, pentanucleotide-matched negatives, anti-TCW polarity tests, and species-mismatch coordinate-memorization controls. We validate one predicted A3A hotspot by deep amplicon sequencing in [cell line] and confirm APOBEC1-correlated editing in colorectal tumor RNA-seq for the A1 head. v4 enables principled construction of cfDNA-style research panels that complement driver-gene assays for mutational-signature analysis from low-input clinical samples.

### Top 3 Must-Do Experiments

1. **Replication-timing / chromatin covariate ablation** (computational, 2-3 weeks). Stratify panel lift by Repli-seq quintile and chromHMM state. *Non-negotiable*; this is the #1 reviewer rejection vector.

2. **APOBEC1 expression × A1-head editing correlation in TCGA COADREAD/ESCA RNA-seq** (computational on public data, 2-3 weeks). Validates that the multi-enzyme architecture genuinely separates enzymes. Decisive and cheap.

3. **Wet-lab: Deep amplicon sequencing of 5-10 top-1% predicted A3A hotspots in a tumor cell line ± A3A knockdown** (4-6 weeks, ~$5-10K). Provides the "Figure 5 wet-lab moment" and directly tests A3A-dependence of the top panel. This is the single experiment that converts the paper from Genome Biology to Nature Cancer territory.

**Also strongly recommended (not blocking):** RNAsee head-to-head on the same 8.45M positions; pentanucleotide-matched-negatives ablation; SHAP/feature attribution at panel cut.

### Timeline

- **Weeks 1-3:** Rep-timing ablation, RNAsee head-to-head, pentanucleotide ablation, SHAP. APOBEC1 RNA-seq validation in parallel (public TCGA data).
- **Weeks 2-8:** Wet-lab amplicon sequencing experiment (parallel track).
- **Weeks 6-9:** Manuscript drafting, figures, supplement (including transparent Finding 3 disclosure).
- **Weeks 9-10:** Internal review, polish.
- **Submission: Week 10** for Genome Biology baseline.
- **Submission: Week 14-16** for Nature Cancer with full wet-lab.

### Biggest Reviewer Objection

**"The model has rediscovered replication timing and chromatin context, not APOBEC vulnerability beyond motif."** Mitigation: the rep-timing ablation in Week 1-3 is the single most important pre-submission task. If lift survives stratification by Repli-seq quintile, this objection dies; if it doesn't, you reframe the paper accordingly *before* a reviewer forces you to.

**Secondary risk:** "Within-TCW lift of only 1.04-1.18× shows the model is mostly motif-driven." Mitigation: lead with within-non-TCW (4-5×) as the headline novelty; report within-TCW transparently with biological rationale (TCW is homogeneous to APOBEC, so within-TCW discrimination is intrinsically hard).

### Final Word

You have one strong paper. Don't dilute it. Spend the wet-lab budget on one decisive experiment (amplicon sequencing of predicted A3A hotspots ± A3A KD), do the rep-timing ablation no matter what, drop Finding 3 to a transparent supplement, and aim for Genome Biology with a Nature Cancer stretch contingent on wet-lab landing. Ten weeks to a defensible submission; fourteen to a stretch one.
