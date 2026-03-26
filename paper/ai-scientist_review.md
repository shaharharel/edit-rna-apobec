# Ai Scientist Review

Now I have a thorough understanding of the actual architecture. Let me provide the review.

---

## Architecture Summary

The core architecture is a dual-forward-pass system: both the original and C-to-U edited sequences are independently encoded through a frozen RNA-FM (640-dim), and the resulting token tensors are used to construct an edit embedding via local difference at the edit position, a cross-attention query from the edit site to the full context, and concatenation of hand-crafted signal channels (motif, structure delta, concordance). The edit embedding then fuses with a pooled background representation through a gated modality fusion module into a multi-task prediction head. The primary validation claim is that this edit effect formulation (AUROC=0.928-0.935) outperforms SubtractionMLP (AUROC=0.841), with GB on 40-dim hand features (AUROC=0.923) as the dominant non-neural baseline.

---

## 1. Architecture Improvements: What to Actually Do

### The Double-Forward-Pass Is the Core Architectural Problem

The architecture encodes both `seq_before` and `seq_after` through frozen RNA-FM, then computes `f_edited - f_bg` at the edit position. This is topologically equivalent to the SubtractionMLP baseline, but at the token level rather than the pooled level. The gap between EditRNA (0.928) and SubtractionMLP (0.841) likely comes from two specific components: the cross-attention query (edit site attends to full context) and the hand features injected into the GNN slot. This is not the same as causal modeling — it is feature-level subtraction plus attention pooling.

The fundamental issue is that a frozen RNA-FM, pretrained on MLM objectives over general RNA sequences, has no notion of APOBEC substrate recognition. Both sequences pass through the same frozen weights, so the difference signal at position 100 carries only what RNA-FM encodes about C vs U in context — which is not specifically APOBEC-relevant. The model is learning to use hand features (motif + loop geometry injected as `hand_features` into the GNN slot) as its primary signal, with the RNA-FM delta as a noisy auxiliary.

**Evidence for this interpretation**: GB_HandFeatures at 0.923 AUROC uses only 40 dimensions. EditRNA achieves 0.928-0.935, a modest improvement over 40 features with 640-dim frozen embeddings on 8,153 samples. The delta is almost entirely explainable by the 33-dim features injected via the `hand_features` slot in `EditRNA+Features`.

**Concrete architectural alternatives in priority order:**

Priority 1 — LoRA/adapter fine-tuning of the last 2-3 RNA-FM layers specifically for C→U context discrimination. The current `finetune_last_n=0` default means no gradient flows into RNA-FM. With 8,153 samples, fine-tuning all 640-dim×12-layer weights would overfit, but a 4-8 rank LoRA applied to the last 2 attention layers (roughly 320K additional parameters vs RNA-FM's ~100M) would let the model learn APOBEC-specific representations. The key insight: RNA-FM was not pre-trained on any editing task signal, so its frozen representations encode sequence/structure co-variation, not editing propensity.

Priority 2 — Replace the cross-attention query (edit position queries full sequence) with a structure-aware graph attention where nodes are RNA structural units (stems, loops, bulges) and the edit site node has directed edges to nodes within the same structural element. The current approach attends uniformly over linear sequence positions; the #1 feature (`relative_loop_position` with importance 0.213) encodes structural position, but the attention module ignores this entirely. The GNN branch exists in the config but is off by default and has not been evaluated against the full pipeline.

Priority 3 — The `local_window` ablation in the config (slices tokens to ±w positions) is worth running at w=10 and w=20. If performance matches full-sequence attention, it confirms that RNA-FM's global context adds nothing beyond local sequence, which would simplify the architecture substantially and reduce inference cost.

**What to avoid**: The dual-encoder (RNA-FM + UTR-LM) adds 128-dim at the edit position and has no published ablation. Given the signal is structure-dominated at low dimensionality, the extra encoder is unlikely to help and adds training instability.

---

## 2. Rate Prediction: This Is Likely a Ceiling Problem

Spearman ~0.12 is not primarily an architecture problem — it reflects the data's actual information content given the experimental setup.

The fundamental issue is **distributional confounding across datasets**, which you've correctly identified. Per-dataset Z-scoring prevents dataset-identity leakage into targets, but it also destroys any absolute-rate signal across the full 4,462 sites. After Z-scoring, each dataset's rate becomes a within-dataset relative ranking. Cross-dataset training then amounts to asking: "can features predict relative editing efficiency within any experimental system?" The cross-dataset off-diagonal Spearman collapse answers this: no, because the determinants of relative rate differ between in vitro overexpression (Baysal, Asaoka) and endogenous expression in tissues (Levanon GTEx). The 54-tissue Levanon rates are the only endogenous signal, and they represent 120 sites (A3A-only).

**What would actually move the needle:**

The signal ceiling at Spearman~0.12 is consistent with the features being collectively necessary but not sufficient. Missing features that matter at the rate level: (a) local protein occupancy — RBPs protecting or exposing the site; (b) APOBEC3A protein expression levels (available from GTEx RNA-seq); (c) co-transcriptional folding kinetics (transcription speed affects which structures form, which directly affects loop accessibility); (d) RNA modifications at neighboring positions (m6A or pseudouridine can block APOBEC access). None of these are captured in the current 40-dim feature set, and they require external databases to include.

**On contrastive/ranking losses**: A pairwise ranking loss (e.g., RankNet or ListMLE) operating within-dataset would be architecturally appropriate here, since the Z-scored target is an ordinal signal. The current Huber loss treats Z-score differences as metric, which they are not when the distributions are independent across datasets. Applying a within-batch pairwise ranking loss restricted to same-dataset samples would better exploit the actual signal structure.

**The Sigmoid bug is worth fixing before any further architecture iterations**: The `rate_head` with `nn.Sigmoid()` bounding outputs to (0,1) while targets range [-3, +3] (Z-scored) guarantees that Huber loss will push predictions toward 0.5 as a compromise. The model achieves Spearman=0.137 > GB's 0.122 only because rank order is partially preserved by the nonlinearity's monotonicity. R²=-0.049 reveals the scale mismatch. Fix the Sigmoid and re-run before interpreting any rate architecture results.

---

## 3. Edit Effect Framework Validation: The Gap Is Not Yet Convincing

The 0.087 AUROC gap between EditRNA (0.928) and SubtractionMLP (0.841) is presented as the primary evidence for the edit effect framework. Several confounds make this gap difficult to interpret cleanly:

**Confound 1 — Architecture asymmetry.** SubtractionMLP receives pooled (mean-pooled) embeddings of 640 dimensions, then runs a 2-layer MLP (640→256→128→1). EditRNA receives full token sequences, computes cross-attention, injects hand features, runs multi-modal gated fusion, and has a multi-task head. This is not a controlled comparison of "subtraction vs causal formulation" — it is a comparison of a simple model against a complex model with more inductive biases and more signal channels.

**Confound 2 — Hand feature injection.** The `EditRNA+Features` model injects 33-dim hand features via the `hand_features` slot in the GNN position. The SubtractionMLP does not receive these features. A `SubtractionMLP+Features` model exists in the experiment list but the gap in the reported results (not shown in the summary above) is what matters for the framework claim. If `SubtractionMLP+Features` matches `EditRNA+Features` closely, the "edit effect framework" advantage reduces to the cross-attention component, which is a much weaker claim.

**Confound 3 — The subtraction operation itself.** SubtractionMLP subtracts pooled embeddings, discarding all positional information. For a C→U SNV at position 100 of a 201-nt window, the change in pooled 640-dim embedding will be negligible (single-nucleotide change diluted across 201 positions). The subtraction signal is near-zero by construction for this task. A token-level subtraction at position 100 only (rather than pooled subtraction) would be a fairer baseline and would likely perform much closer to EditRNA.

**To make the claim convincing**, run these three ablations in order:

1. `TokenSubtractionMLP`: Use only `f_edited[100] - f_bg[100]` (single-position token subtraction, no cross-attention, no hand features). This isolates what RNA-FM token-level difference contributes at the edit site.

2. `SubtractionMLP+Features` vs `EditRNA+Features` (if not already done with full results reported): If these are within 0.5% AUROC, the "edit effect framework" claim reduces to "hand features + any embedding = works."

3. `HandFeaturesOnly+CrossAttention`: Use only the 33-dim hand features injected via the GNN slot, add a cross-attention module over those features. Measures whether the cross-attention provides value beyond the hand features independent of RNA-FM.

---

## 4. Foundation Model Usage

**Frozen RNA-FM with 201-nt windows is appropriate given data scale.** 8,153 samples is far below the fine-tuning threshold for 100M parameter models without aggressive regularization. LoRA at rank 4-8 on the last 2 layers is the right intervention, with held-out validation across seeds to confirm it doesn't overfit.

**Newer RNA foundation models to consider:**

EVO (Arc Institute, 2024) was trained on 2.7 billion nucleotides of diverse DNA/RNA with multi-scale context. Its 7B parameter version would be overkill, but the 7M parameter variant might extract better single-nucleotide perturbation responses than RNA-FM because its pretraining objective included next-token prediction at variable context lengths.

RNA-BERT (Akiyama & Sakakibara) and RNABERT (already in the codebase) were specifically designed for structural alignment, which aligns better with this task than RNA-FM's general sequence MLM. The codebase already has `rnabert.py` — this is worth running in the main classification experiment before investing in newer models.

**The single most impactful change**: Per-token position-specific representations rather than pooled representations for the background encoder. Currently `primary_pooled` (mean over 201 tokens) is used as the background signal in the fusion module. This discards the structural context of the edit site entirely. Using the token embedding at position 100 (`f_background[:, 100, :]`) instead of the mean pool would give the fusion module position-specific information. This is a one-line change with potentially significant impact.

---

## 5. Publication Potential

The project has strong narrative components but needs sharper architectural contribution framing and one additional empirical element.

**Current publishable strengths:**
- Three-enzyme comparative analysis with distinct structural programs is genuinely novel and interpretable. The A3G tetraloop preference and A3B's tissue-nonspecific mixed editing are findings that stand independent of ML methodology.
- ClinVar enrichment with Bayesian recalibration at OR=1.33 for A3A is a clinically relevant result with proper statistical treatment.
- The "structure over sequence" finding is consistently reproduced across all three enzymes and matches the current understanding of APOBEC substrate recognition.
- Resolving the Butt 2024 vs Alonso de la Vega 2023 contradiction with A3B data is publication-worthy.

**What's needed for Nature Methods / Genome Biology:**

The edit effect framework claim needs a cleaner controlled comparison (addressed in section 3). As currently presented, a skeptical reviewer will note that SubtractionMLP uses pooled embeddings while EditRNA uses token-level attention plus hand features. That asymmetry is the gap, not the causal formulation.

The rate prediction at Spearman=0.12 is honest but weak. For Methods-tier work, a concrete biological interpretation of what drives rate variation within a single experimental system (Levanon 54-tissue analysis, or Baysal in-vitro series) would strengthen this. The tissue-conditioning experiment exists in the codebase (`exp_tissue_conditioned_rate.py`) — if Levanon tissue rates show meaningful variation across the 54 tissues for the same 120 sites, a tissue-conditioned model that predicts tissue-specific rates would be a stronger contribution than a single-rate predictor.

**Strongest narrative arc**: "APOBEC enzymes have evolved distinct structural selectivity programs captured by accessible loop geometry; this structural logic predicts ClinVar pathogenic sites enriched for C→U editing, with implications for somatic mutation interpretation." The edit effect framework is a supporting methodology point, not the headline. The three-enzyme comparison and ClinVar enrichment are the headline.

For Genome Biology the current results are publishable with addition of: (a) wet lab validation of 3-5 novel ClinVar-predicted sites (contacts, not DIY), (b) the multi-enzyme comparison in a single figure with confidence intervals, (c) code and data release with reproducible pipeline.

---

## 6. The "Both" Category Architecture

**The 178 A3A+A3G shared substrates are the most informative class for understanding shared vs specific recognition.**

**What the overlap tells you**: Both A3A and A3G require the site to be accessible (unpaired). A3A prefers TC context, A3G prefers CC context. The "Both" category likely has intermediate or promiscuous context (sites that tolerate either dinucleotide) combined with strong structural accessibility. This hypothesis is directly testable with your current feature set before building new architecture.

**Architecture recommendations for multi-enzyme modeling:**

Multi-task learning with shared backbone is the right starting point, not mixture of experts. The biological reasoning: structure accessibility features are shared across enzymes (shared backbone), while motif features are enzyme-specific (task-specific heads or embeddings). The current `enzyme_head` in `APOBECMultiTaskHead` predicts enzyme specificity as a 4-class problem from the fused representation — this is a discriminative formulation when what you want is generative: given this site, what is the probability of A3A editing, A3G editing, and their joint probability?

The specific architectural addition that is worth implementing: replace the single enzyme specificity head with two independent binary classification heads (one for A3A, one for A3G), trained simultaneously on their respective positive/negative sets. The "Both" sites are positive for both heads. A site with high P(A3A) and high P(A3G) is predicted as "Both" at inference. This factorization allows the model to learn independent recognition mechanisms rather than treating "Both" as an atomic class (which conflates two distinct recognition events).

The product of the two independent heads' probabilities should be regularized against the empirical joint rate — sites that are edited by both enzymes tend to have lower per-enzyme rates individually (competition), which is a testable biological prior. This kind of probabilistic factorization would directly support the "shared structural program, distinct sequence programs" narrative and is straightforwardly implementable given the existing multi-task head structure at `/Users/shaharharel/Documents/github/edit-rna-apobec/src/models/prediction_heads.py`.

---

## Key Ablations in Priority Order

1. Fix the Sigmoid bug in `rate_head` (one-line change in `/Users/shaharharel/Documents/github/edit-rna-apobec/src/models/apobec_edit_embedding.py` line 303 of the original `apobec_edit_embedding.py` file) and re-run rate experiments. All current rate comparisons are confounded.

2. `TokenSubtractionMLP`: token-level subtraction at edit position only, no pooling, no hand features. This is the honest baseline for the edit effect framework claim.

3. `SubtractionMLP+Features` with the same 33-dim feature injection as `EditRNA+Features`. Determines if the framework gap is real or artifactual from feature asymmetry.

4. `local_window=10` ablation of EditRNA: does restricting attention to ±10 positions match full-sequence attention? If yes, RNA-FM's global context is unused and the architecture can be simplified.

5. Position-100 token pooling vs mean pooling for `primary_pooled` in the fusion module. Replace `primary_out["pooled"]` (mean) with `f_background[:, 100, :]` (edit-site token) for the background representation.

6. Per-enzyme binary head factorization vs 4-class enzyme head: train on A3A-positives + A3G-positives simultaneously with independent binary heads and report per-enzyme AUROC on "Both" sites vs single-class head.

---

Relevant source files for the issues identified:

- `/Users/shaharharel/Documents/github/edit-rna-apobec/src/models/apobec_edit_embedding.py` — Sigmoid bug at line 303 (rate_head), context attention implementation
- `/Users/shaharharel/Documents/github/edit-rna-apobec/src/models/baselines/subtraction_mlp.py` — pooled subtraction (creates the confounded comparison)
- `/Users/shaharharel/Documents/github/edit-rna-apobec/src/models/editrna_a3a.py` — double forward pass, `hand_features` slot injection at line 438, `pooled_only` flag
- `/Users/shaharharel/Documents/github/edit-rna-apobec/src/models/prediction_heads.py` — multi-task head structure, enzyme 4-class formulation to refactor
- `/Users/shaharharel/Documents/github/edit-rna-apobec/src/models/apobec_model.py` — `MultiModalFusion` gating and the mean-pool vs position-100-token decision at line 368-371