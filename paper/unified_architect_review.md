Now I have a thorough understanding of the existing architecture and data landscape. Let me provide the architectural review.

---

### Architecture Summary

The request is to design a unified multi-enzyme APOBEC editing prediction network performing joint binary classification (edited vs. not) and enzyme identity classification (6-class) with a shared backbone, enzyme-specific components, and proper handling of the asymmetry that negatives carry no enzyme label. This builds directly on the existing `EditRNA_A3A` / `APOBECEditEmbedding` stack.

---

### What Works Well

The framing of the two-head hierarchy is biologically sound. Binary editability and enzyme identity are genuinely different levels of the prediction problem: the shared structure signal (loop position, stem geometry) is predictive of editability across all APOBEC3 enzymes, while the motif and loop-position preference distinguish enzymes. Separating these into a shared trunk and enzyme-specific heads is the right inductive bias.

The existing 40-dim hand feature decomposition (motif 24 + loop 9 + struct_delta 7) already encodes exactly the features known to differ across enzymes. This is the right place to apply enzyme-specific adaptation rather than building it into the heavy RNA-FM encoder.

Your existing `APOBECEditEmbedding` + `GatedModalityFusion` pipeline is reusable without modification as the shared backbone. The gated fusion already allows the network to learn that different modalities matter more for different inputs, which will naturally specialize across enzyme types once you add the enzyme head.

The NaN/mask-based loss convention in `APOBECMultiTaskLoss` gives you the correct mechanism to suppress the enzyme loss on negatives, since you can assign enzyme label = -1 for all negatives and the existing cross-entropy masking with `targets["enzyme"] >= 0` handles it cleanly.

---

### Architectural Risks / Failure Modes

**1. Task conflict between binary head and enzyme head on negatives.**
This is the central structural risk. During training, negatives contribute to the binary loss (label=0) but are masked from the enzyme loss. The shared trunk therefore receives conflicting gradient signals: the binary loss pushes the trunk to represent negatives distinctly from all positives, while the enzyme loss only pulls on the positive subspace. At scale this creates a degenerate solution where the shared backbone learns a binary-discriminative representation that has nothing to do with the enzyme-specific structure that the enzyme head needs. The trunk will converge on whatever resolves the binary loss first (motif + loop), and the enzyme head will then try to linearly separate A3A/A3B/A3G in a space optimized for the binary problem. Since your per-enzyme per-feature results already show that motif alone (XGB_MotifOnly) achieves reasonable AUROC, the trunk is likely to anchor on motif early and the enzyme head becomes a near-linear motif decoder — which is not a shared learning win.

**2. A3G class collapse due to extreme imbalance.**
A3G has 179 positives vs the combined pool of thousands of negatives and thousands of other-enzyme positives. With Kendall uncertainty weighting, the enzyme loss on A3G will be dominated by A3A and A3B samples (4180 A3B positives alone). The network will learn to classify A3G as a residual: samples that are not A3A and not A3B. This is fragile. The tetraloop-specific signal for A3G is real and strong (StructOnly AUROC=0.935 > GB_HandFeatures AUROC=0.841), but if the trunk is gradient-dominated by the A3B binary signal, the tetraloop geometry will be washed out of the shared representation.

**3. "Neither" is not an APOBEC3 enzyme and poisons the shared inductive bias.**
The Neither class (206 sites, mooring-sequence-based, no structure preference) has fundamentally different recognition logic. Training it jointly in the enzyme head will push the trunk to also represent a mooring-sequence feature, which is orthogonal to the stem-loop signal that all APOBEC3 enzymes share. This is not a shared learning benefit — it is interference. The Neither class should be treated as a separate binary problem or excluded from the joint enzyme head entirely.

**4. The existing APOBECEditEmbedding's flanking_embed is A3A/A3G-specific.**
The `FLANKING_CONTEXT` embedding in `apobec_edit_embedding.py` encodes TC=0, CC=1, AC=2, GC=3. This is the right motif encoding for A3A and A3G, but A3B has no meaningful motif preference. Using this as a shared component means A3B's flanking signal will be noisy supervision on an embedding that is dimensionally incorrect for it. The flanking embedding should be shared but its gradient contribution should be enzyme-conditioned — or it should be moved into a feature that gets weighted per enzyme.

**5. The binary loss is unsupervised with respect to enzyme identity for positives.**
If a site is positive but the enzyme head is uncertain (e.g., an A3A_A3G co-edited site), the binary loss will push its representation toward the positive cluster without anchoring it to the right enzyme subspace. The uncertainty weighting will not resolve this because the enzyme loss weight applies uniformly to all labeled samples, regardless of class confidence.

**6. Class imbalance in the enzyme head is not just about count — it is about feature overlap.**
A3A (2749) and A3B (4180) have substantial feature overlap at the trunk level (both have roughly similar loop access to sites). The enzyme head will need to disentangle them using the motif signal, which is subtle (TC vs mixed). With the current concatenation architecture `[fused, edit_emb]` as input to the enzyme head, the network has no incentive to route the motif signal specifically through the enzyme head rather than using it in the binary head, which will distort both.

---

### Key Ablations or Experiments Needed

These are ordered by architectural criticality, not implementation difficulty.

**A1 — Binary head trained alone vs jointly with enzyme head.**
Train the network with only the binary loss (no enzyme head). Evaluate binary AUROC per enzyme. Then train jointly. If joint training hurts binary AUROC on any enzyme (especially A3G), it confirms task conflict. This is the most important ablation before committing to the joint design.

**A2 — Shared trunk with frozen trunk + fine-tuned enzyme heads.**
After training the binary classifier to convergence, freeze the trunk and train only the enzyme head on positives only. Compare to end-to-end joint training. If the frozen-trunk enzyme head matches or beats joint training, the tasks are orthogonal enough to warrant staged training and the shared learning hypothesis is weakened.

**A3 — Neither class excluded from enzyme head.**
Train the enzyme head on {A3A, A3B, A3G, A3A_A3G} only (4 classes), with Neither handled by a separate lightweight binary head. Compare macro-F1 on the Neither class in both settings. This tests whether Neither hurts the APOBEC3 subfamily classification.

**A4 — Per-enzyme feature importance under the shared trunk.**
After training, extract the gate weights from `GatedModalityFusion` and the attention weights from `APOBECEditEmbedding.context_attention` for each enzyme subpopulation. If A3G does not show elevated attention to tetraloop-proximal positions compared to A3A, the shared trunk is not learning enzyme-specific structural features — it is learning a generic editability feature that the enzyme head cannot refine.

**A5 — GB_HandFeatures enzyme-conditioned baseline.**
Train a single GradientBoosting model on the 40-dim features with a 6-class enzyme label (positives only). This is the strongest legitimate baseline for the enzyme classification task. The neural network must beat this cleanly on per-enzyme AUROC, otherwise the shared backbone adds no value over conditionally stratified GB models.

---

### Suggested Improvements or Alternatives

**Recommended first-iteration architecture: Hierarchical two-stage with enzyme-conditioned feature routing.**

The core idea is to separate the trunk into two components: a structure-aware shared component and a motif-conditioned enzyme-routing component. The enzyme head then receives a motif-routed representation rather than the full fused trunk.

```
Input: [RNA-FM pooled (640), hand_features (40), edit_emb (256)]

Stage 1 — Shared structure trunk:
    structure_repr = MLP([loop_feats_9, struct_delta_7])  # (B, 64)
    rnafm_proj = Linear(640 -> 256)  # frozen RNA-FM, project only
    shared = concat([rnafm_proj, structure_repr])  # (B, 320)
    binary_logit = BinaryHead(shared)  # (B, 1)

Stage 2 — Enzyme routing (positives only in training):
    motif_repr = MotifMLP(motif_24 -> 64)  # (B, 64)
    enzyme_input = concat([shared, motif_repr])  # (B, 384)
    enzyme_logits = EnzymeHead(enzyme_input)  # (B, n_enzyme_classes)
```

The separation of structure_repr from motif_repr enforces the inductive bias: structure features go to both binary and enzyme heads, but motif features go only to the enzyme head. This prevents the binary head from using motif (which would let it overfit to the motif distribution of the training set's negatives) while giving the enzyme head direct access to the only features that actually distinguish enzymes.

The Neither class should receive its own binary head taking as input a mooring-sequence embedding extracted from a 15-nt upstream window rather than the structure features. This is a fundamentally different problem.

For the A3G class collapse problem: apply class-weighted cross-entropy in the enzyme head with inverse-frequency weights. This is not enough on its own — you should also oversample A3G sites (or upsample A3A_A3G which contains A3G-edited sites) during the enzyme-head training phase. Given A3G has only 179 sites, 5-fold CV will give only ~143 training samples per fold for A3G, which is insufficient for a neural head.

**Alternative: GB ensemble with a learned neural arbitrator.**

Given the data scale (15K sites, of which only 7.5K are positive, and A3G has 179), the case for a full neural multi-task network is weak. A more defensible architecture is:

1. Per-enzyme GB classifiers on 40-dim hand features (existing, validated).
2. A meta-classifier that takes the 5 per-enzyme GB probability scores as input and predicts the enzyme label on positives.
3. The binary decision is max(score_A3A, score_A3B, score_A3G, score_A3A_A3G, score_Neither) > threshold.

This achieves shared learning through the feature space rather than through a shared neural trunk, requires no special handling of negatives in the enzyme classification, and is interpretable. The tradeoff is that it does not learn cross-enzyme feature interactions at the representation level — but given the low data regime, this is a favorable bias.

**If pursuing the neural architecture: concrete first iteration.**

Operationally, the simplest version that avoids the failure modes above without adding much complexity is:

- Shared trunk: frozen RNA-FM pooled (640) projected to 256, concatenated with loop_feats_9 and struct_delta_7 and projected to 256. Total: Linear(256+9+7 -> 256) + GELU + LayerNorm.
- Binary head: Linear(256 -> 1) with focal loss alpha=0.75.
- Enzyme head: concatenate shared_repr (256) with motif_onehot (24) and hand_feats_40, then Linear(320 -> 128) + GELU + Dropout(0.3) + Linear(128 -> n_enzyme_classes). Train with inverse-frequency weighted cross-entropy on positives only (enzyme != -1).
- Loss masking: enzyme loss masked to `enzyme_label >= 0` (existing convention in `APOBECMultiTaskLoss`).
- Training: Joint from the start but with gradient accumulation strategy that ensures each enzyme class sees roughly equal updates per epoch (i.e., oversample A3G batches).

This is preferable to a phased training scheme because the binary and enzyme heads share the structure_repr, and alternating optimization of the two losses on a low-data problem tends to lead to representation oscillation.

The one component you should not add in the first iteration is FiLM conditioning or a Mixture of Experts layer. Both require enough per-enzyme data to learn the conditioning function reliably, which you do not have for A3G and A3A_A3G. Add these in a second iteration if and only if A3G enzyme head accuracy exceeds 0.85 AUROC in the simple first-iteration design, which would indicate the data is informative enough to support conditioning.

---

### Relevant files for reference

- `/Users/shaharharel/Documents/github/edit-rna-apobec/src/models/apobec_edit_embedding.py` — existing `APOBECEditEmbedding`, reusable as shared trunk component; `FLANKING_CONTEXT` encoding is A3A/A3G-biased and needs treatment
- `/Users/shaharharel/Documents/github/edit-rna-apobec/src/models/prediction_heads.py` — `APOBECMultiTaskHead` and `APOBECMultiTaskLoss`; enzyme masking convention (`enzyme >= 0`) is correct and reusable
- `/Users/shaharharel/Documents/github/edit-rna-apobec/src/models/fusion.py` — `GatedModalityFusion`; the gate compute-then-apply pattern can be reused to route motif vs structure features
- `/Users/shaharharel/Documents/github/edit-rna-apobec/src/data/apobec_feature_extraction.py` — feature decomposition into motif_24, loop_9, struct_delta_7; the proposed architecture routes these to different heads rather than concatenating all 40 for both
- `/Users/shaharharel/Documents/github/edit-rna-apobec/experiments/multi_enzyme/exp_unified_v2.py` — the existing `XGB_MotifOnly` and `XGB_StructOnly` ablations here constitute the minimum baseline that any neural enzyme head must beat