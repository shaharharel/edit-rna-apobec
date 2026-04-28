# SCIENTIST TEAM REVIEW — v4 Multi-Enzyme APOBEC Panel

## 1. Dr. Sara Kapoor — Senior PI, Cancer Genomics

**Journal target: Genome Research (realistic) / Nature Communications (stretch).** Not Nature or Cell — the headline isn't a clinical actionable or a mechanism, it's a transferability claim. But the cross-modality story (RNA editing → DNA mutation prediction) plus per-cancer enzyme assignment (APOBEC1 winning COADREAD/ESCA in two independent cohorts) is genuinely interesting to a cancer-genomics audience. Genome Research likes biological-discovery-with-method papers; this fits.

**Strongest claim.** Per-cancer head dominance replicating across PCAWG and POG570 with biologically coherent assignments (APOBEC1 → intestinal). That's not a metric, that's a finding. The 4.55-4.63× A3A enrichment being universal is the second-strongest plank.

**Weakest part.** The leap from "RNA-editable C site" to "DNA somatic mutation hotspot" is mechanistically vague. Reviewers will ask: is this transfer learning capturing APOBEC sequence preference + accessible-RNA-secondary-structure proxying for ssDNA accessibility, or is it just trinucleotide-corrected motif density with extra steps? You need to *show* the model uses non-motif features at panel-selection time.

**Three experiments to push tier:**
1. **Feature-attribution at the panel cut.** SHAP/IG decomposition of top-1% panel positions: how much of the OR comes from motif (24-d), loop geometry (9-d), structure delta (7-d), RNA-FM context (640-d), edit-delta (640-d)? If structure + context contribute >30% you've earned the "beyond motif" claim.
2. **Drivers vs passengers stratification.** Re-run top-1% enrichment after excluding known driver hotspots (TP53, PIK3CA hotspot codons). If the OR survives, you're predicting passenger landscape, which is the more defensible claim.
3. **Replication-timing + chromatin covariate ablation.** Add RepliSeq + ATAC as additive predictors; show your panel adds OR beyond them. Without this, reviewer #2 will reject.

**Work estimate: 4-6 weeks.**

---

## 2. Dr. Tom Reedy — Computational Methods Reviewer

**Journal target: NAR Genomics & Bioinformatics (realistic) / Nature Methods (stretch, only if you reframe).** The trinuc-matched-negatives fix is a real methods contribution — and you correctly note prior tools (RNAsee) inherited the same shortcut. That's the kind of "the field has been wrong" angle Nature Methods likes, but only if you make RNAsee-vs-v3-vs-v4 the front-line comparison and quantify the bias across the prior literature.

**Strongest claim.** AUROC drop 0.93 → 0.84 with bias diagnostic confirming polarity removal. This is honest and exactly what a methods reviewer wants to see. The QA suite (shuffle test ratio 0.97, leave-leak-out delta 0.00pp, tie-pool=1) is professional.

**Weakest part.** Permutation floor at 2K. You cannot claim Bonferroni significance across a 5-head × 20-cancer × multiple-cuts test family with 2K perms. P-value floor is ~5e-4; with ~100 cells you need 30-100K. Also: per-fold AUROC 0.842 ± 0.009 across 5 folds is a very tight CI — verify the folds are gene-level or chromosome-level split, not site-level. Site-level CV with neighboring positions in train+test is the classic invisible leak.

**Three experiments:**
1. **30K-100K permutations on headline cells** (panel top-1%, top-10%, per-cancer A3A and APOBEC1). Non-negotiable for any methods venue.
2. **Gene-level or chromosome-level held-out split** verification — re-report AUROC. If it drops below 0.80 you have a leakage problem; if it holds you have a stronger paper.
3. **Direct head-to-head vs RNAsee on v4 negatives.** Re-score RNAsee, plot OR-vs-cut on the same axes, report calibration curves. This is the single highest-leverage missing experiment for methods framing.

**Work estimate: 3-4 weeks** (permutations are compute-bound but parallelizable).

---

## 3. Dr. Ana Mendez — RNA Biology / APOBEC Expert

**Journal target: RNA (realistic) / Genome Biology (stretch).** The biology is interesting but currently underdeveloped. The "Both"/"Neither"/A3G distinct-programs finding (CC=65%, intestine-specific Neither → APOBEC1) is more novel than the panel transfer claim, and you've buried it.

**Strongest claim.** Three distinct editing programs with the APOBEC1 head winning gut cancers in two independent cohorts. That's a mechanistic prediction (APOBEC1 contributes to colorectal/esophageal mutagenesis) and it's testable. Cross-species 24% conservation is the second-strongest — structural-vulnerability is a real hypothesis.

**Weakest part.** No mechanistic anchor between editing site and mutation site. APOBEC3A deaminates ssDNA in stem-loop apices in vivo (Buisson 2019, Hoopes 2016). Your loop-geometry features capture this for RNA, but you haven't shown the *DNA* mutations you predict are also in stem-loops. That's the bridge. Also: APOBEC1 in human gut is barely expressed compared to mouse small intestine — reviewers will demand expression evidence (GTEx/HPA) before accepting "APOBEC1 drives ESCA mutations."

**Three experiments:**
1. **DNA stem-loop check on top-1% panel hits in tumor mutations.** Use mFold/RNAfold on ±50nt around mutated C; if your panel hits enrich for apex-of-hairpin DNA structures more than non-panel TCW mutations do, you've closed the mechanistic loop.
2. **APOBEC1 expression validation in COADREAD/ESCA tumors.** TCGA RNA-seq APOBEC1 levels vs APOBEC1-head OR per tumor. If high APOBEC1 expression tracks with high APOBEC1-head enrichment, that's a beautiful figure.
3. **UCC trinucleotide test + mooring-sequence search for "Neither"/APOBEC1 candidates.** Already on your priority list — do it. It's the cleanest enzyme-assignment validation possible.

**Work estimate: 5-7 weeks** (the structure-on-DNA analysis is the heavy lift).

---

## 4. Dr. Kai Patel — Paper & Grants Strategist

**Journal target: Genome Biology (realistic) / Nature Communications (stretch).** GB is the right home: methods + biology + clinical-adjacent fits their scope, and the editorial board responds well to "we fixed a community-wide bug + here's the biology." Do NOT aim for Nature/Cell — you don't have a single hero finding; you have a constellation, and constellation papers get desk-rejected at top-tier.

**Strongest claim for narrative.** Frame as: "A trinucleotide-corrected APOBEC predictor reveals enzyme-specific contributions to cancer mutagenesis, including an under-recognized APOBEC1 signature in gastrointestinal tumors." That's a one-sentence abstract hook. The methods fix is the *setup*, the per-cancer enzyme story is the *payoff*.

**Weakest part — strategic.** You're trying to sell three papers in one: (a) trinuc-matched negatives methods, (b) RNA→DNA transfer, (c) multi-enzyme cancer biology. Pick ONE primary, demote the others to supporting. Right now the panel claim is primary and the multi-enzyme biology is buried — wrong choice. The biology is the discovery; the panel is the validation.

**Three experiments / additions:**
1. **Re-run ClinVar with v4** (you flagged this as not-done). If the v3 OR=1.33 holds with trinuc-corrected negatives, the clinical-relevance plank is restored. If it collapses, kill the claim cleanly — better than reviewers killing it.
2. **Reframe figure 1 around the multi-enzyme programs** (A3A/A3B/A3G/Both/Neither/APOBEC1). Currently figure 1 is probably the panel transfer; move that to figure 3-4. Lead with biology, close with translation.
3. **Add a wet-lab collaborator letter or pilot experiment** — even a single APOBEC1-knockdown qPCR on a colon line, or a published-data reanalysis of APOBEC1-KO mice. Reviewers will be 2x more generous with one biological validation, however small.

**Work estimate: 6-8 weeks** including reframing + ClinVar rerun + light biology.

---

## Synthesis

**Consensus journal target: Genome Biology.** Three of four voices independently land here or adjacent (GR, NAR-GB, RNA). It's the right scope: methods + biology + cancer-relevance, audience that will appreciate the trinuc-correction fix and the per-cancer enzyme story.

**Stretch target: Nature Communications** — achievable only with (a) v4 ClinVar replication holding, (b) DNA stem-loop mechanistic bridge, (c) one wet-lab validation or a clean APOBEC1-expression correlation in TCGA.

**Top 3 must-do experiments before submission (ranked):**
1. **30K-100K permutations + chromosome-level CV verification** (Reedy). Without this, statistical claims are not defensible — methods reviewer will reject. ~1 week compute, 2-3 days analysis.
2. **v4 ClinVar replication + RNAsee v4 head-to-head** (Patel + Reedy). The v3 ClinVar OR=1.33 claim is currently orphaned; either resurrect or retract cleanly. RNAsee comparison defines the field-level contribution. ~1.5 weeks.
3. **Replication-timing + chromatin ablation on the panel transfer claim** (Kapoor). This is the experiment reviewer #2 will demand; pre-empting it is worth a tier. ~1.5-2 weeks.

**Top 2 nice-to-have (revision-response acceptable):**
1. **DNA stem-loop check on panel-predicted DNA mutations** (Mendez) — closes the mechanistic loop, can be added in revision.
2. **APOBEC1 expression vs head-OR correlation across TCGA tumors** (Mendez + Patel) — beautiful single figure, doable in revision if needed but ideal to include up-front.

**Realistic timeline to submission: 8-10 weeks.** Permutations + CV verification (week 1-2), v4 ClinVar + RNAsee (week 3-4), replication-timing ablation (week 5-6), figure assembly + reframe around multi-enzyme biology (week 7-8), internal review + polish (week 9-10).

**Biggest reviewer concern: the cross-modality leap.** "Why do RNA editability scores predict DNA somatic mutations beyond motif density?" If you cannot answer with (a) feature-attribution showing structure/context drive a meaningful fraction of the panel signal, AND (b) replication-timing + chromatin ablation showing the panel adds OR over those covariates, this paper gets a major-revision-bordering-reject from any methods-aware reviewer. The trinuc fix protects you from the easy critique; the structure-driven feature attribution protects you from the hard one. Plan for both.
