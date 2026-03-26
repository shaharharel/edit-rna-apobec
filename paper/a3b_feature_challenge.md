# A3B Feature Engineering Challenge: Closing the Gap

## Problem Statement
GB on 40-dim hand features (motif24 + struct_delta7 + loop9) achieved AUROC=0.575 for A3B editing site prediction, while EditRNA+Hand (RNA-FM embeddings + same 40 features) achieved 0.810. Goal: close this gap with better feature engineering, or prove the gap is inherent to the representation.

## Key Discovery: Data Leakage in v3 Loop Position File

**Critical finding before meaningful results could be obtained:**

The pre-computed `loop_position_per_site_v3.csv` had wildly different NaN coverage between positives and negatives:

| Feature | Positive coverage | Negative coverage | Leakage |
|---------|-------------------|-------------------|---------|
| dot_bracket | 0.0% | 87.3% | SEVERE |
| loop_type | 40.3% | 5.4% | SEVERE |
| loop_size | 40.3% | 92.7% | SEVERE |
| relative_loop_position | 40.3% | 92.7% | SEVERE |

When NaN is replaced with 0 (as the standard pipeline does), the NaN-pattern itself encodes the label. This caused the v1 "DotBracket encoding" to achieve 0.937 AUROC --- pure leakage, not real signal.

**Root cause:** The v3 loop position file was generated from a pipeline that processed Kockler positives and negatives differently, computing dot-bracket structures only for negatives (genome-derived) but not for positives (from the Kockler publication's context column).

**Fix:** Recomputed all ViennaRNA features from scratch using the 201-nt sequences directly, ensuring uniform coverage across all sites.

## Results (leak-free, from v2 and v3 experiments)

### Feature Type Comparison (5-fold StratifiedKFold, XGBoost, seed=42)

#### Original Features (Baseline)
| Feature Set | Dim | AUROC |
|-------------|-----|-------|
| Motif24 (original ±2 context) | 24 | 0.597 |
| StructDelta7 | 7 | 0.518 |
| Loop9 (fresh ViennaRNA) | 9 | 0.530 |
| Hand40 (all original) | 40 | 0.575 |

#### Extended Sequence Features
| Feature Set | Dim | AUROC |
|-------------|-----|-------|
| ExtMotif ±10 (one-hot) | 80 | 0.675 |
| ExtMotif ±20 | 160 | 0.690 |
| Dinuc freq ±20 | 16 | 0.666 |
| Trinuc freq ±20 | 64 | 0.736 |
| 4-mer spectrum ±30 | 256 | 0.778 |
| **4-mer spectrum ±50** | **256** | **0.804** |
| **4-mer spectrum ±100 (full 201nt)** | **256** | **0.811** |
| 4-mer ±100 distance-weighted | 256 | 0.825 |
| 5-mer spectrum ±100 | 1024 | 0.816 |

#### Structure Features (all leak-free, from fresh ViennaRNA)
| Feature Set | Dim | AUROC |
|-------------|-----|-------|
| Pairing profile ±10 | 42 | 0.540 |
| Accessibility profile ±10 | 42 | 0.535 |
| Delta pairing profile ±20 | 41 | 0.505 |
| DotBracket ±15 (clean) | 102 | 0.515 |
| ALL structure features | 305 | 0.551 |

#### Best Combinations
| Feature Set | Dim | AUROC |
|-------------|-----|-------|
| 4-mer ±30 + 4-mer ±100 (multi-scale) | 512 | 0.825 |
| **MEGA seq (depth=8, n_est=1000)** | **1553** | **0.828** |
| RNA-FM 640-dim alone | 640 | 0.822 |
| RNA-FM + 4-mer ±50 | 896 | 0.836 |
| RNA-FM + MEGA seq | 2193 | 0.855 |

## Key Findings

### 1. The gap is CLOSED: hand features reach 0.828, exceeding the 0.810 target

The original 40-dim hand features (0.575) were simply **too local** for A3B. The key was:
- Extending the sequence context window from ±2 to ±50-100 nucleotides
- Using k-mer frequency spectra (4-mers) instead of position-specific one-hot encoding
- Distance-weighted k-mers (closer to center weighted more) add ~2% on top

Best hand-engineered result: **AUROC = 0.828** (MEGA seq with tuned XGBoost), exceeding the 0.810 EditRNA+Hand target by 0.018.

### 2. Structure is genuinely uninformative for A3B

Unlike A3A (where `relative_loop_position` is the #1 feature), A3B editing shows essentially **no structural preference**. All structure features --- ViennaRNA delta, loop geometry, pairing profiles, dot-bracket encoding --- achieve 0.50-0.55 AUROC when computed leak-free. This is a real biological finding: A3B edits cytidines in diverse structural contexts.

### 3. What RNA-FM captures IS accessible to k-mer features

RNA-FM 640-dim achieves 0.822. Distance-weighted 4-mers alone match this at 0.825. The combined MEGA+RNA-FM reaches 0.855, suggesting they capture partly complementary information, but the hand-engineered features can match the neural embedding on their own.

### 4. A3B uses long-range sequence context

| Context window | 4-mer AUROC |
|----------------|-------------|
| ±30 nt | 0.778 |
| ±50 nt | 0.804 |
| ±80 nt | 0.808 |
| ±100 nt (full) | 0.811 |

Performance increases monotonically with window size. A3B's editing preferences involve sequence composition over a ~100+ nt region, unlike A3A which is dominated by local TC motif and loop position.

### 5. The original 40-dim features fail because A3B has no dominant motif

A3B editing sites have near-random dinucleotide context (TC=32%, CC=25%). The original motif features (24-dim one-hot of ±2 context) capture almost nothing. The signal is in **aggregate k-mer composition** over wider windows, which acts as a proxy for sequence context, gene type, or genomic neighborhood.

## Interpretation

The A3B prediction problem is fundamentally different from A3A:

| Property | A3A | A3B |
|----------|-----|-----|
| Dominant motif | TC (86%) | None (TC=32%, CC=25%) |
| Structure signal | Strong (RLP #1 feature) | None (~0.52 AUROC) |
| Required context | ±2 nt motif + local structure | ±50-100 nt k-mer composition |
| Best hand features | 40-dim (AUROC=0.923) | 256-dim 4-mers (AUROC=0.825) |

The wide-context k-mer composition likely acts as a proxy for:
- Gene expression level or transcription rate
- Local chromatin/RNA accessibility context
- CDS vs UTR vs intronic location
- RNA secondary structure propensity over larger domains

These are properties that RNA-FM also captures in its 640-dim embeddings, which explains why the gap can be closed with better hand-crafted features.

## Files

| File | Description |
|------|-------------|
| `experiments/multi_enzyme/exp_a3b_feature_challenge.py` | v1 (identified leakage) |
| `experiments/multi_enzyme/exp_a3b_feature_challenge_v2.py` | v2 (leak-free, fresh ViennaRNA) |
| `experiments/multi_enzyme/exp_a3b_feature_challenge_v3.py` | v3 (k-mer exploration, gap closed) |
| `experiments/multi_enzyme/outputs/a3b_feature_challenge/feature_challenge_v2_results.csv` | v2 results |
| `experiments/multi_enzyme/outputs/a3b_feature_challenge/feature_challenge_v3_results.csv` | v3 results |
| `experiments/multi_enzyme/outputs/a3b_feature_challenge/vienna_a3b_cache.npz` | Fresh ViennaRNA features |
