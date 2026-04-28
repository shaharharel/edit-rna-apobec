# Architecture & Training Plan V2 — 2026-03-30

**Goal**: Find the best RNA editing prediction model within our data constraints.
**Phase**: Architecture + training method search. No external validation (TCGA/ClinVar/cross-species) until convergence.
**Evaluation**: Standard ML metrics (AUROC per enzyme, 2-fold CV for screening, 5-fold for top models).
**Baseline**: XGB 40d always included for comparison.

---

## 1. Architecture Candidates

### A1. Current Phase 3 NN (baseline)
- RNA-FM pooled (640) + edit_delta (640) + hand features (40) = 1320 → shared encoder → heads
- Already trained. Reference for all comparisons.
- CV: A3A=0.885, A3B=0.826, A3G=0.915, A3A_A3G=0.960, Neither=0.882

### A2. Phase 3 NN without edit delta
- RNA-FM pooled (640) + hand features (40) = 680 → same architecture
- Tests whether the C→U perturbation signal adds anything
- If equal to A1, we drop delta (halves RNA-FM compute — no need to embed edited sequences)

### A3. AdarEdit-adapted GAT + RNA-FM
- Graph Attention Network on RNA secondary structure (from AdarEdit paper)
- Nodes = nucleotides, typed edges = backbone + base pairs (stem, loop, bulge)
- **Addition**: inject RNA-FM pooled embedding (640) + hand features (40) into the graph node features or as a global conditioning vector
- ViennaRNA dot-bracket → graph construction (already have structure cached)
- Does NOT need token-level RNA-FM — uses pooled embedding as global context

### A4a. RNA-FM LoRA Fine-Tuning (standalone)
- Take the pretrained RNA-FM model (12 layers, 99.5M params)
- Add LoRA adapters (rank 4-8) to attention layers — only ~100K trainable params
- Fine-tune on our editing classification task directly
- Output: CLS token → classification heads
- **Key advantage**: adapts the foundation model's representations to editing-specific patterns without catastrophic forgetting
- Needs token-level forward pass but NOT cached token embeddings — runs end-to-end
- MPS compatible (model fits in memory, ~400MB)

### A4b. RNA-FM LoRA + Hand Features
- Same LoRA fine-tuning as A4a
- After CLS token extraction: concatenate with hand features (40d) before classification heads
- CLS(640) + hand(40) = 680 → heads
- Tests whether LoRA-adapted RNA-FM already captures what hand features encode, or if they're complementary

### A5. Gated Multimodal Fusion
- Instead of concatenation, dynamic gating between modalities:
  - g_rnafm = sigmoid(W₁ · hand) — hand features gate RNA-FM contribution
  - g_struct = sigmoid(W₂ · rnafm) — RNA-FM gates structure contribution
  - fusion = g_rnafm ⊙ rnafm + g_struct ⊙ structure_features + hand
- Tests whether modalities should be weighted differently per sample
- Simple, few extra params, drop-in replacement for concatenation in A1

### A6. Conv2D on BP Matrix + RNA-FM + Features (Enhanced Phase 1 winner)
- Conv2D encoder on 41×41 ViennaRNA base-pair probability submatrix → 128-dim
- Concatenate with RNA-FM pooled (640) + hand features (40) = 808-dim
- Phase 1 Conv2D_BP scored 0.793 on multi-enzyme (best deep model), but without RNA-FM
- Adding RNA-FM may push it further
- BP submatrix already cached in structure cache

### A7. [DROPPED — VAE generative framing not well-motivated for discriminative task]

### A8. Hierarchical Attention (Local Structure + Global Context + Features)
- Two-level attention:
  - Level 1: Local attention over the 41-nt window around edit site (structure-focused, uses BP probabilities as attention bias)
  - Level 2: Global RNA-FM pooled embedding (640) provides context
  - Cross-attention: local structure attends to global context
- Fuse with hand features (40d) after cross-attention: [local_out | global_out | hand] → heads
- Motivated by: editing depends on LOCAL loop structure but also GLOBAL RNA context (e.g., where in the transcript)
- Uses pooled RNA-FM (no token-level needed) + BP submatrix (cached) + hand features

---

## 2. Head Architectures

Test these head configurations on the best encoder:

### H1. Binary + Per-enzyme adapters (current)
- Binary head on all data + 5 enzyme-specific adapter heads

### H2. Per-enzyme only (no binary head)
- Remove binary head. Use max(enzyme_heads) or sum for binary decision.
- Tests whether the binary head hurts enzyme-specific discrimination

### H3. Hierarchical heads
- First: binary head (edited or not?)
- Then: enzyme classifier (which enzyme?) — only trained on positives
- At inference: P(edited) × P(enzyme|edited) = P(enzyme)
- Proper probabilistic decomposition

### H4. Mixture-of-experts heads
- Shared encoder → routing network decides which enzyme expert to activate
- Each expert is a small MLP specialized for one enzyme
- Routing is soft (weighted combination) during training, hard at inference
- Could help with the enzyme imbalance problem (A3A has 5K sites, A3G has 179)

---

## 3. Training Methods

Each tested independently on A1 architecture, 2-fold CV, then best methods combined with best architecture.

### T1. Baseline (current two-stage training)

### T2. v4-random negatives (1:10)
- XGB baseline on v4-random included for comparison

### T3. v4-hard negatives (TC+loop, 1:5)
- XGB baseline on v4-hard included

### T4. v4-large mixed negatives (1:50)
- XGB baseline on v4-large included

### T5. Hard negative curriculum
- Epochs 1-5: v4-random only → Epochs 5-10: mix → Epochs 10+: v4-hard only

### T6. m6A transfer pretraining
- Pretrain shared encoder on 797K m6A sites (binary: m6A vs random)
- Then fine-tune on APOBEC editing data
- m6A shares structural preferences with APOBEC editing (both prefer unpaired regions)

### T7. Contrastive pretraining
- Contrastive loss: pull edited-site embeddings close, push non-editing apart
- Pretrain encoder, then fine-tune with classification heads

### T8. Meta-learning (Prototypical Networks)
- Episode training: sample enzymes as tasks, K-shot support + query
- Learn embedding space where enzyme prototypes are well-separated
- Specifically useful for: learning new enzymes from few examples (A4: 21 sites)
- Evaluate with enzyme-holdout: train on 4 enzymes, predict 5th

---

## 4. Evaluation Splits

### Standard (all comparisons)
- 2-fold CV for screening, 5-fold for top models
- Metrics: AUROC per enzyme, overall AUROC, AUPRC

### Challenge splits (top models only)
- **Gene-holdout**: Hold out entire genes. Tests generalization to unseen genes.
- **Chromosome-holdout**: Train chr1-20, test chr21-22+X.
- **Enzyme-holdout (LOO)**: Train on 4 enzymes, predict 5th. Key for T8 evaluation.

---

## 5. Execution Order

### Phase A: Architecture Screen (MPS)
Run A1, A2, A3, A4a, A4b, A5, A6, A8 on v3, 2-fold CV. ~30 min each = ~4h total.
XGB 40d baseline always included. (A7 dropped.)

### Phase B: Training Screen (MPS)
Run T1-T8 on best architecture from Phase A, 2-fold CV. ~20 min each = ~3h total.
XGB on v4 datasets for comparison.

### Phase C: Head Screen (MPS)
Run H1-H4 on best architecture + best training. 2-fold CV. ~20 min each = ~1.5h.

### Phase D: Top Combinations (MPS)
Best 2-3 architectures × best 2-3 training methods. 5-fold CV. ~1h each = ~6h.
Challenge splits on winner.

### Phase E: External Validation (next iteration)
Winner → full TCGA + ClinVar + cross-species evaluation.

**Total MPS time: ~15h for Phases A-D**

### CPU jobs (parallel with MPS):
- XGB baselines on v4 datasets
- Token-level RNA-FM embedding cache (if needed for A4 LoRA)
- Challenge split data preparation

---

## 6. Resource Notes

- **MPS**: Single Apple Silicon GPU. One training job at a time. ~10 min/epoch on v3.
- **CPU**: Unlimited. XGB trains in seconds. ViennaRNA cached.
- **Memory**: 128GB RAM.
- **Cached data**:
  - RNA-FM pooled (v3, v4, TCGA, ClinVar, cross-species): all done
  - RNA-FM token-level: NOT cached. Need ~15 min to compute for v3 (7.4GB). Only needed for A4.
  - ViennaRNA: cached for v3, TCGA, full exome
  - BP probability submatrix: in structure cache for v3
