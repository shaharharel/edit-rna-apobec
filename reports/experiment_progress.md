# Experiment Progress Tracker

---

## 2026-03-30 00:30 — Plan Created

### Architecture Screen (Phase A)
| ID | Architecture | Status | 2-fold AUROC | Notes |
|----|-------------|--------|:---:|-------|
| A1 | Phase 3 NN (1320d baseline) | **DONE** (previous) | 0.814 overall | A3A=0.885, Neither=0.882 |
| A2 | A1 without edit delta (680d) | PENDING | — | Tests delta contribution |
| A3 | AdarEdit GAT + RNA-FM | PENDING | — | Graph on structure + RNA-FM global |
| A4a | RNA-FM LoRA fine-tuning | PENDING | — | End-to-end, ~100K trainable params |
| A4b | RNA-FM LoRA + hand features | PENDING | — | LoRA CLS(640) + hand(40) → heads |
| A5 | Gated Multimodal Fusion | PENDING | — | Dynamic modality weighting |
| A6 | Conv2D BP + RNA-FM + features | PENDING | — | Enhanced Phase 1 winner |
| ~~A7~~ | ~~VAE generative~~ | DROPPED | — | Not well-motivated for discriminative task |
| A8 | Hierarchical Attention + features | PENDING | — | Local structure + global context + hand |
| — | XGB 40d baseline | **DONE** | 0.923 A3A | Reference |

### Training Screen (Phase B)
| ID | Method | Status | Architecture | Notes |
|----|--------|--------|:-:|-------|
| T1 | Baseline two-stage | **DONE** (A1) | A1 | Reference |
| T2 | v4-random (1:10) | PENDING | TBD | |
| T3 | v4-hard (TC+loop, 1:5) | PENDING | TBD | |
| T4 | v4-large (1:50) | PENDING | TBD | |
| T5 | Hard negative curriculum | PENDING | TBD | |
| T6 | m6A transfer pretraining | PENDING | TBD | |
| T7 | Contrastive pretraining | PENDING | TBD | |
| T8 | Meta-learning (Prototypical) | PENDING | TBD | |

### Head Screen (Phase C)
| ID | Configuration | Status | Notes |
|----|--------------|--------|-------|
| H1 | Binary + per-enzyme adapters | **DONE** (current) | Reference |
| H2 | Per-enzyme only | PENDING | |
| H3 | Hierarchical (binary → enzyme) | PENDING | |
| H4 | Mixture-of-experts | PENDING | |

### Dataset Comparison
| Dataset | XGB 40d | Best NN | Status |
|---------|:-:|:-:|--------|
| v3 (1:1) | 0.923 A3A | 0.885 A3A | DONE |
| v4-random (1:10) | PENDING | PENDING | |
| v4-hard (1:5) | PENDING | PENDING | |
| v4-large (1:50) | PENDING | PENDING | |

---

## Completed Experiments (prior to this plan)

### 2026-03-27 — Phase 1 Architectures
- Conv2D_BP: 0.793, DualPath: 0.790, FiLM: 0.748, EditRNA: 0.744, Transformer: 0.684
- XGB on same v3 multi-enzyme: 0.644
- Deep models beat XGB by +0.15 on multi-enzyme task

### 2026-03-28 — Phase 3 Neural + TCGA Fix
- Phase 3 NN trained (RNA-FM + delta + hand features)
- TCGA hand feature alignment bug found and fixed
- NN TCGA results: binary weaker than XGB, but per-enzyme adapters strong
- Neither adapter: COADREAD OR=4.72, STAD OR=4.17

### 2026-03-29 — NN Validation Complete
- SKCM negative control: binary OR=0.536 (depleted) ✓
- ClinVar: NN binary OR=1.30@p90, A3B OR=1.43@p90 ✓
- ClinVar GI-gene stratification: Neither OR=1.43@p90 in GI genes (Simpson's paradox resolved)
- TC-stratified: NN passes all cancers (OR=1.5-4.5 in TC context) ✓
- Exome-wide ClinVar with XGB full model: stable OR=1.22-1.28 ✓
- E2 gnomAD full genome: completed (null after trinucleotide control)
- E1 divergence: completed (null after CpG control)
- E4 lineage-specific: completed (structural features differ, no constraint difference)

---

## 2026-03-30 00:35 — Architecture Screen Started

### XGB v4 Baselines (CPU) — DONE (1.6 min)
| Dataset | Overall | A3A | A3B | A3G | Neither |
|---------|:-:|:-:|:-:|:-:|:-:|
| v3 (1:1) | 0.635 | 0.721 | 0.554 | 0.779 | 0.698 |
| v4-random (1:10) | 1.000 | — | — | — | — |
| v4-hard (1:5) | 1.000 | — | — | — | — |
| v4-large (1:50) | 0.993 | 0.731 | 0.559 | 0.751 | 0.729 |

Note: v4-random/hard overall=1.0 because negatives trivially separable. Per-enzyme NaN because v4 negatives lack enzyme labels.

## 2026-03-30 01:30 — Architecture Screen Results (non-LoRA)

| Architecture | Overall | A3A | A3B | A3G | A3A_A3G | Neither | Time |
|-------------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **A8 Hierarchical** | **0.812** | **0.886** | **0.817** | **0.924** | **0.980** | **0.891** | 192s |
| A6 Conv2DBP | 0.805 | 0.882 | 0.812 | 0.908 | 0.969 | 0.885 | 160s |
| A1 Baseline | 0.803 | 0.880 | 0.815 | 0.900 | 0.945 | 0.854 | 136s |
| A2 NoDelta | 0.803 | 0.880 | 0.813 | 0.890 | 0.949 | 0.843 | 137s |
| A5 GatedFusion | 0.802 | 0.877 | 0.815 | 0.902 | 0.945 | 0.875 | 160s |
| A3 GAT | 0.801 | 0.878 | 0.812 | 0.895 | 0.944 | 0.848 | 811s |

**Key takeaways:**
- A8 (Hierarchical Attention) wins across all enzymes
- Edit delta adds nothing (A2 = A1) — can be dropped
- GAT is slowest and doesn't beat simpler models
- A6 (Conv2D BP) is solid second place

## 2026-03-30 07:10 — Architecture Screen COMPLETE

| Rank | Architecture | Overall | A3A | A3B | A3G | A3A_A3G | Neither | Time |
|:---:|-------------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **1** | **A4b LoRA+Feat** | **0.824** | 0.881 | **0.845** | 0.880 | 0.942 | 0.859 | 10,327s |
| **2** | **A8 Hierarchical** | 0.812 | **0.886** | 0.817 | **0.924** | **0.980** | **0.891** | 192s |
| 3 | A6 Conv2DBP | 0.805 | 0.882 | 0.812 | 0.908 | 0.969 | 0.885 | 160s |
| 4 | A1 Baseline | 0.803 | 0.880 | 0.815 | 0.900 | 0.945 | 0.854 | 136s |
| 5 | A2 NoDelta | 0.803 | 0.880 | 0.813 | 0.890 | 0.949 | 0.843 | 137s |
| 6 | A5 GatedFusion | 0.802 | 0.877 | 0.815 | 0.902 | 0.945 | 0.875 | 160s |
| 7 | A3 GAT | 0.801 | 0.878 | 0.812 | 0.895 | 0.944 | 0.848 | 811s |
| 8 | A4a LoRA | 0.789 | 0.741 | 0.832 | 0.800 | 0.837 | 0.733 | 10,663s |

**Conclusions:** A4b best overall, A8 best minority classes + 54x faster. Edit delta useless. GAT slow and unhelpful.
**Next:** Phase B — training methods on A8 (fast) and A4b (best).

## 2026-03-30 12:30 — Training Screen (Phase B) COMPLETE

**Best per-enzyme results (T5 curriculum):**

| Architecture | A3A | A3B | A3G | A3A_A3G | Neither |
|-------------|:-:|:-:|:-:|:-:|:-:|
| **A6+T5** | 0.885 | 0.819 | **0.929** | **0.986** | **0.912** |
| A8+T5 | 0.881 | 0.820 | 0.898 | 0.964 | 0.895 |
| A1+T5 | 0.883 | 0.820 | 0.904 | 0.957 | 0.900 |

**Conclusions:**
- T5 curriculum is best training method for per-enzyme
- A6 Conv2DBP + T5 curriculum = best combination overall
- v4-large (T4) hurts per-enzyme — too many easy negatives
- Architecture matters less than training method

**Top candidates for Phase C/D:**
1. A6+T5 (Conv2D + curriculum) — best minority classes
2. A8+T5 (Hierarchical + curriculum) — Phase A winner, moderate with curriculum
3. A1+T5 (Baseline + curriculum) — simple but effective

## 2026-03-30 20:00 — V5 Dataset Created

V5 = v3 with A3A positives replaced by full A3A pipeline (asaoka+advisor+alqassim+sharma):
- 17,790 sites (10,002 pos + 7,788 neg)
- A3A: 5,187 pos (was 2,749 in v3) — +2,438 A3A sites
- All sequences, structure, RNA-FM available (0 missing)

### Datasets Summary
| Dataset | Positives | Negatives | Ratio | Key change |
|---------|:-:|:-:|:-:|---|
| v3 | 7,564 | 7,788 | 1:1 | kockler A3A only (2,749) |
| **v5** | **10,002** | 7,788 | 1.3:1 | **Full A3A (5,187)** from asaoka+advisor+alqassim |
| v4-random | 7,564 | 75,640 | 1:10 | Random exonic C negatives |
| v4-hard | 7,564 | 37,820 | 1:5 | TC+loop negatives |
| v4-large | 7,564 | 378,200 | 1:50 | Mixed negatives |

## 2026-03-31 01:50 — Phase C+D COMPLETE

### Phase C: Head Configurations (A8, 2-fold)
| Head | Overall | A3A | A3B | A3G | A3A_A3G | Neither |
|------|:-:|:-:|:-:|:-:|:-:|:-:|
| **H4 Shared+Private** | **0.811** | **0.885** | 0.816 | **0.932** | **0.979** | **0.898** |
| H1 Binary+Adapters | 0.804 | 0.878 | 0.817 | 0.893 | 0.936 | 0.842 |

### Phase D.1: 5-Fold CV
| Model | A3A | A3B | A3G | A3A_A3G | Neither |
|-------|:-:|:-:|:-:|:-:|:-:|
| A8+T1 | 0.887±0.011 | 0.824±0.007 | 0.926±0.041 | 0.982±0.017 | 0.907±0.031 |
| XGB | 0.857±0.009 | 0.592±0.007 | 0.835±0.024 | 0.932±0.017 | 0.842±0.020 |

### Phase D.2: Challenge Splits
| Challenge | XGB | A8+T1 |
|-----------|:-:|:-:|
| Gene holdout | 0.642 | **0.787** |
| Enzyme LOO A3B | 0.519 | **0.697** |
| Enzyme LOO A3G | 0.752 | **0.879** |

### V5 Bug Investigation
- Loop feature bug found: `dist_to_junction` was measuring distance to sequence edge, not to stem
- Fixed with correct `src/_extract_loop_geometry` implementation
- BUT: A3A still 1.000 even after fix — generated negatives fundamentally too easy
- Need tier2/tier3-style negatives (same transcripts) for V5 A3A
- Original A3A report replicated: AUROC=0.916 (report: 0.923)

## Running Now

| Task | PID | Status | ETA |
|------|-----|--------|-----|
| Winner screen: 12 combos 3-fold → top 5 challenge splits | writing | Agent writing + will launch | ~3h total |
