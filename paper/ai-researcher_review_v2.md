Now I have a thorough understanding of the codebase. Let me provide a rigorous architectural review and experiment specification.

---

## Architecture Summary

The existing EditRNA_A3A uses a dual-forward RNA-FM pass to produce a structured edit embedding (local token diff + cross-attention context + motif + structure delta), which is then fused with optional GNN/ViT modalities via gated softmax fusion and routed to multi-task heads. The proposed experiment extends this single-enzyme architecture to a unified 6-class multi-enzyme classifier, adding complexity at every level: more classes, more training signal sources, and a harder learning objective.

---

## What Works Well

**The multi-task loss infrastructure is already positioned for this.** `APOBECMultiTaskHead` in `src/models/prediction_heads.py` already has an `enzyme_head` outputting 4 logits. The `APOBECMultiTaskLoss` handles the masking convention (`enzyme >= 0`, with -1 for Unknown) correctly. Extending to 6 classes is a minimal change to head dimensions, not architecture.

**The 40-dim hand feature set captures the genuinely decisive signals.** Results show structure dominates across all enzymes. The fact that GB_HandFeatures is competitive or best-in-class on A3A (0.923), A3G (0.931), and A3A_A3G (0.941) is not a weakness of the feature set — it is strong evidence that the problem is low-dimensional and locally governed. This is the right inductive bias for a dataset of ~15k sites.

**The Kendall uncertainty weighting is appropriate.** When binary classification, enzyme attribution, and rate regression have incompatible loss magnitudes, learnable log-variance weighting is the principled choice. The existing implementation is correct.

**NaN masking throughout the loss.** Partially labeled data (sites known to be edited but enzyme unknown, or rate available only for Levanon/Baysal) is handled correctly through the mask-before-loss pattern. This is essential and already done well.

---

## Architectural Risks / Failure Modes

### 1. Unified vs Per-Enzyme: The Central Architectural Question

Training one model for all 6 classes is seductive but likely wrong for this dataset. Here is why:

The enzyme classes do not form a flat taxonomy of equal difficulty. A3A and A3G have distinct biochemistry (TC motif vs CC motif, distinct structural preferences). A3A_A3G is a genuinely ambiguous label — it means the site is edited by both, not that there is a halfway-enzyme. "Neither" is a negatives-derived class that exists by data construction, not biology. "Unknown" is a missing-label class. Training a single softmax head over these 6 classes treats them as if they are mutually exclusive and exhaustively defined by distinct generative processes, which they are not.

The deeper problem: in the current dataset of 15,342 sites, negatives (7,778 sites) were generated per-enzyme via motif-matched negative sampling. A unified model trained to predict "which enzyme" for negatives has no correct answer — negatives are not "edited by Unknown enzyme," they are "not edited." The binary head and enzyme head are therefore solving inconsistent objectives for the same samples unless the binary head always fires first and gates the enzyme head, which the current architecture does not enforce.

**Recommended structure:** A two-stage hierarchical classifier, not a flat 6-class softmax. Stage 1 predicts binary (is_edited / not). Stage 2, conditioned on Stage 1 = positive, predicts enzyme attribution (A3A / A3G / A3A_A3G / Neither / Unknown). This has clean semantics. The enzyme head should never receive negatives as training examples.

### 2. The Edit Embedding Is Enzyme-Agnostic By Construction

`APOBECEditEmbedding` encodes the dinucleotide flanking context as a 4-way embedding (TC=0, CC=1, AC=2, GC=3). TC is the A3A motif, CC is the A3G motif. This single feature already captures most of the enzyme discrimination signal, but it is compressed to a 32-dim embedding shared across all enzymes. There is no mechanism that allows the edit embedding to attend differently to the surrounding context depending on which enzyme is being predicted.

This matters because A3G and A3A have structurally different preferences beyond dinucleotide: A3G favors extreme 3' tetraloop geometries (per CLAUDE.md, "extreme 3' tetraloop"), while A3A favors moderate stem-loop positions. If the `flanking_embed` learns a generic enzyme-discriminating embedding, it will be a confound inside the edit embedding rather than an input to enzyme-head routing.

**Risk:** The edit embedding entangles enzyme identity with the causal edit representation, which undermines the causal interpretation of `EditEffectHead`. An edit effect prediction from the edit embedding should be enzyme-agnostic (the C-to-U change has a physical effect regardless of which enzyme catalyzes it). But if the edit embedding has learned to predict enzyme class, the `EditEffectHead` is actually predicting enzyme-specific effects, which is a category error.

### 3. Negative Construction Leakage Into Enzyme Classification

The multi-enzyme dataset (`splits_multi_enzyme_v2_with_negatives.csv`) contains negatives that were generated per-enzyme via motif-matched sampling. This means A3G negatives are sampled to have CC-motif sequences, and A3A negatives have TC-motif sequences. If a unified model sees these negatives labeled as "not edited" alongside enzyme-labeled positives, it will learn: "CC-motif + not edited = A3G-category negative, CC-motif + edited = A3G positive." The enzyme head implicitly leaks into the binary head via shared backbone. Motif becomes a shortcut for both tasks simultaneously, making the enzyme attribution trivially learned from the one feature that is already most predictive of binary editing.

This is not a fatal flaw, but it means that enzyme attribution AUROC in the unified model will be inflated relative to what the model actually learns about enzyme-specific structural determinism beyond the motif.

### 4. The Tissue Rate Head Has a Sigmoid Bug (Still Unfixed)

Per CLAUDE.md, the `rate_head` was previously bounded to (0,1) via `nn.Sigmoid()` while Z-scored targets span [-3, +3]. The current `prediction_heads.py` shows this was fixed in the Levanon-oriented `APOBECMultiTaskHead` (no Sigmoid on `EditingRateHead` or `HEK293RateHead`), but the `apobec_edit_embedding.py` version still has the old comment "No Sigmoid: Z-scored targets range [-3, +3]" as documentation of the fix. Verify the version of the head being used in any new experiment explicitly outputs unbounded values.

### 5. The GatedModalityFusion Softmax Gate Is Miscalibrated for Structure Modalities

The gate computes a softmax over all N modalities, meaning adding a new modality (e.g., GNN) necessarily reduces the weight of existing modalities. With 4 modalities at `proj_dim = d_fused // 2 = 256` each, the gate input is a 1024-dim vector projecting to 4 scalars. This projection has no expressivity for learning which *site-specific* features should upweight structure vs sequence. A sigmoid gate per modality (independent, not competing via softmax) would allow the model to learn "this site has strong structure signal and weak sequence signal" without a zero-sum tradeoff.

More critically: the fusion operates on pooled representations. The GNN and ViT both pool their entire graph/contact-map to a single vector before fusion. For a site-level prediction where the local structural context (5-15nt around the edit) matters far more than global structure, global pooling discards the decisive information. GAT/GCN global-mean-pool over a 201-node graph gives equal weight to nucleotides 100 positions away from the edit as to the ones immediately flanking it. A center-weighted pool (e.g., exponentially decaying from the edit position) would be more principled.

### 6. Multi-Task Learning With Highly Imbalanced Auxiliary Tasks Will Destabilize Training

The tertiary tasks have severe class imbalance: conservation is 95 vs 541 (15% positive), cancer survival is 252 vs 384. Focal loss is applied to conservation but cancer survival uses plain BCE. More importantly, the tertiary/auxiliary tasks have coverage only over the 636 Levanon sites, not the full 15,342 site dataset. When training on the multi-enzyme dataset, these tasks will have NaN masks for ~97% of the batch at every step. The uncertainty weighting will accordingly set their log-variance high (low precision), but the gradient through the mask-then-loss computation on 3% of the batch is still a gradient signal. With small effective batch sizes for tertiary tasks (~0.6 samples per batch at batch_size=20), the Kendall weighting will be unstable.

**Recommendation:** For the multi-enzyme experiment, run only PRIMARY tasks (binary, enzyme) and optionally rate. Disable secondary, tertiary, and auxiliary tasks or train them as a separate fine-tuning head after the primary tasks converge.

### 7. The Contact Map ViT Has Architectural Overcomplexity Given Dataset Scale

The contact map ViT operates on 201x201 base-pair probability matrices. For a dataset of ~15k sites, training a 4-layer ViT on this input will overfit severely. The contact map encodes long-range pairing probabilities, but the decisive structural features (is the C in a loop, how large is the loop, how long are the flanking stems) are already captured in the 7-dim structure delta and 9-dim loop geometry features. The ViT adds ~8M parameters processing information that GB captures in 7 numbers. Unless you have evidence that global contact topology beyond the hand features adds predictive signal (which the GB results do not suggest), the ViT is a liability at this data scale.

---

## Key Ablations or Experiments Needed

Listed in priority order:

**1. Flat unified vs hierarchical (two-stage) enzyme classification.** Train (a) one model with a 6-class softmax over all sites, (b) one binary model then a 4-class enzyme model on positives only, (c) per-enzyme binary models as currently exist. Compare on: binary AUROC, per-enzyme AUROC for enzyme attribution task, and confusion matrix. This directly tests whether joint training helps or hurts. Without this, you cannot justify the unified architecture for a paper.

**2. Edit embedding vs no edit embedding (subtraction baseline) for enzyme attribution.** This is the core claim of the edit-chem framework. Train two variants: one where the enzyme head receives `fused` from the full EditRNA pipeline, and one where it receives `F(seq_after) - F(seq_before)` from RNA-FM without the structured edit embedding. If the structured edit embedding does not improve enzyme attribution over raw subtraction, the framework's value for enzyme classification is not established.

**3. GB_HandFeatures (40-dim) vs EditRNA_unified on enzyme attribution specifically.** GB achieves 0.923-0.941 on binary per-enzyme tasks. The unified neural model must exceed this on enzyme attribution (measured by multi-class macro-AUROC, not just binary AUROC) to justify its complexity. This is the paper's primary numerical claim.

**4. With vs without the flanking motif embedding in the edit embedding.** Given that TC vs CC is the strongest single feature for A3A vs A3G discrimination, test whether removing `flanking_embed` from `APOBECEditEmbedding` significantly degrades enzyme classification but not binary classification. If it does not significantly degrade binary classification, it proves that the edit embedding is learning enzyme-identity rather than causal edit effect — an important architectural finding.

**5. Shared backbone vs separate encoder per enzyme class.** Add a small enzyme-conditioned adapter (e.g., FiLM conditioning on enzyme-class embedding) to the backbone and compare to the baseline shared backbone. This tests whether enzyme-specific structural priors can be injected without requiring separate per-enzyme models.

**6. Negative construction sensitivity.** Subsample the training data to use only Levanon-quality negatives (motif-matched to a different chromosome region, not enzyme-matched) and retrain. If unified model AUROC drops more than per-enzyme models drop, it confirms the enzyme-motif leakage described above.

---

## Suggested Improvements or Alternatives

### Unified Architecture: The Design I Would Recommend

Rather than the full EditRNA pipeline for the enzyme attribution task, use a two-stage architecture:

**Stage 1 (Binary, per-enzyme or shared):** Train the existing GB_HandFeatures or an EditRNA-lite (RNA-FM frozen, edit embedding only, no GNN/ViT, no tertiary tasks) for binary classification. This is already done and works well.

**Stage 2 (Enzyme Attribution, positives only):** On confirmed-positive sites only, train a lightweight enzyme attribution model. Input: the 40-dim hand features + the 256-dim edit embedding from Stage 1 (frozen). The enzyme attribution head is a 3-layer MLP with 128 hidden units, predicting 4 classes (A3A / A3G / A3A_A3G / Neither). This has only ~50k parameters and is trained on ~3k labeled positive sites with enzyme annotations.

This design has four concrete advantages over a unified flat model. First, it has clean semantics: enzyme attribution is defined only over editing-positive sites. Second, the edit embedding from Stage 1 is already trained to capture what makes a site edited — Stage 2 just reads off the enzyme-relevant dimensions. Third, it avoids the gradient interference between binary loss and enzyme loss when training jointly on a mixed batch. Fourth, it is directly comparable to the per-enzyme GB baseline: you are asking "given a confirmed edit site, which enzyme?" rather than the harder and semantically muddier "given a random C in the transcriptome, is it edited and if so by which enzyme?"

### Enzyme Conditioning in the Edit Embedding

If you want a unified model, add a 4-class enzyme conditioning embedding (32-dim) as an additional input to `APOBECEditEmbedding`. During training, this is the known enzyme label (with NaN dropout at 30% to prevent over-reliance). During inference, you either marginalize over enzyme classes or use the Stage 1 enzyme prediction as input. This is a FiLM-style conditioning and adds interpretability: you can ask "what does this site's edit embedding look like when we condition on A3A vs A3G?"

### Replacing the Softmax Gate with Sigmoid Gates

In `GatedModalityFusion`, replace:
```python
gate_weights = self.gate(concat)  # Softmax, sums to 1
```
with per-modality sigmoid gates. Each modality gets an independent probability of being useful rather than competing for a fixed budget. This is mechanistically sounder because structure and sequence information are not substitutes — a high-information sequence representation does not make a high-information structure representation less valuable.

### Metric Reporting for the Paper

The fair comparison between GB and neural models requires reporting them on the same evaluation axes:

For binary classification, report AUROC and AUPRC (both, since positive class is ~50% in the balanced dataset but the real-world prior is ~0.019). Report these separately per enzyme category, not just aggregated. A model that achieves 0.93 binary AUROC by learning the TC/CC motif perfectly is not scientifically interesting — report motif-stratified AUROC (within TC sites vs non-TC sites) to test whether the model learns anything beyond the sequence motif.

For enzyme attribution, report macro-averaged one-vs-rest AUROC across the 4 known enzyme classes (excluding "Unknown" and negatives). Report the confusion matrix. The most important cell is A3A vs A3G (TC vs CC discrimination) — a model that perfectly separates TC-edited and CC-edited sites is not learning enzyme specificity, it is learning sequence composition.

For the paper's central claim, report: delta-AUROC (unified model - per-enzyme model) on a held-out generalization set (e.g., train on A3A+A3G+A3A_A3G, test on sites from a different tissue or chromosomal region). If the unified model transfers better than per-enzyme models on out-of-distribution sites, that is a genuine scientific contribution.

### Training Protocol for the Unified Model

If you proceed with the full EditRNA_A3A as backbone, use a three-phase protocol:

Phase 1 (10 epochs): Freeze RNA-FM entirely. Train only edit embedding, fusion, and binary head on the full 15k dataset. Learning rate 3e-4.

Phase 2 (10 epochs): Unfreeze the last 2 layers of RNA-FM (`finetune_last_n=2`). Add enzyme head, trained on positives only (mask enzyme loss for negatives via the existing mask convention). Reduce LR by factor of 10 for RNA-FM layers. Run primary tasks only (binary + enzyme).

Phase 3 (5 epochs): Full model, all tasks enabled, uncertainty weighting active. This prevents the tertiary tasks (with tiny coverage) from corrupting Phase 1 and Phase 2 training.

The key invariant: **the enzyme loss must be masked to zero for all negative sites throughout all phases.** The current `APOBECMultiTaskLoss` already supports `enzyme >= 0` masking; you must set enzyme label to -1 for all negative sites in the data pipeline, not just for "Unknown" enzyme sites.

---

Relevant files for implementation:

- `/Users/shaharharel/Documents/github/edit-rna-apobec/src/models/editrna_a3a.py` — main model, `EditRNAConfig`, phase-gated training via `get_parameter_groups`
- `/Users/shaharharel/Documents/github/edit-rna-apobec/src/models/apobec_edit_embedding.py` — `APOBECEditEmbedding`, enzyme conditioning should be added here
- `/Users/shaharharel/Documents/github/edit-rna-apobec/src/models/prediction_heads.py` — `APOBECMultiTaskLoss`, enzyme mask convention already correct; extend `n_classes=6` in `EnzymeSpecificityHead`
- `/Users/shaharharel/Documents/github/edit-rna-apobec/src/models/fusion.py` — `GatedModalityFusion`, softmax gate is the target for sigmoid replacement
- `/Users/shaharharel/Documents/github/edit-rna-apobec/experiments/multi_enzyme/exp_per_enzyme_classification.py` — existing per-enzyme GB baseline; the new unified experiment should output results in the same format for direct comparison