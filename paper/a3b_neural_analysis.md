# A3B Neural Model Analysis: What Do Neural Models Learn That Hand Features Miss?

## Summary

The gap between gradient boosting on 40-dim hand features (AUROC=0.578) and neural models
(EditRNA+Hand AUROC=0.811) for A3B editing site classification is the largest in the project.
This analysis identifies what information the neural models capture.

## Key Finding: RNA-FM Embeddings Encode Sequence Context Beyond Local Motifs

The 40-dim hand features capture only the immediate editing neighborhood (trinucleotide
motif, loop geometry, structure delta). Neural models access either RNA-FM embeddings
(EditRNA+Hand) or raw 201-nt sequence (CNN+Hand), which encode longer-range context.

**Critical experiment**: GB trained on RNA-FM embeddings alone (1280-dim) achieves
AUROC=0.818 — matching EditRNA+Hand (0.811). This proves the signal is IN the embeddings,
not in the neural architecture's nonlinear feature interactions.

## Model Comparison

| Model | AUROC | What it accesses |
|-------|-------|-----------------|
| GB_HandFeatures | 0.578 +/- 0.013 | 40-dim local features |
| GB_Hand+KmerWindows | 0.743 +/- 0.007 | 40-dim + dinucleotide freq in 5 windows |
| CNN+Hand | 0.761 +/- 0.013 | Raw 201-nt one-hot + 40-dim |
| EditRNA+Hand | 0.811 +/- 0.007 | RNA-FM embeddings + 40-dim |
| GB_on_RNAFM | 0.818 +/- 0.007 | RNA-FM embeddings (1280-dim) |
| GB_Hand+EmbPCA50 | 0.819 +/- 0.010 | 40-dim + top 50 PCs of RNA-FM |

## What the Embeddings Capture

### PCA Separability
- **PC1 of original RNA-FM embeddings** achieves AUROC=0.727 by itself (single dimension!)
- This PC explains 35% of embedding variance and likely encodes transcript-level properties
  (gene expression context, UTR vs CDS positioning, or GC content patterns)
- The edit-effect embedding (edited - original) has weaker but more distributed separability:
  best single PC is only 0.575, but many PCs contribute (PCs 20-30 most informative)

### Wider Sequence Context Matters
- Adding dinucleotide frequencies in 5 windows spanning +/- 50nt boosts GB from 0.578 to
  0.743 — a +0.165 jump from wider sequence composition alone
- CNN saliency peaks at the edit site (position 100) but 24.5% of total saliency comes from
  the central +/- 20nt window, meaning 75.5% comes from MORE DISTAL positions
- This confirms A3B editing depends on sequence context well beyond the local motif

## Gradient Attribution: What Hand Features Matter in Neural Context

Both neural models agree on feature importance ranking (input x gradient):

1. **max_adjacent_stem_length** (0.119 EditMLP, 0.157 CNN) — strongest by far
2. **right_stem_length** (0.084, 0.123)
3. **left_stem_length** (0.058, 0.092)
4. **delta_mfe** (0.048, 0.051)
5. **loop_size** (0.044, 0.053)

Stem length features dominate in the neural context but have near-zero importance in GB
(because GB cannot combine them with embedding context). The neural models use stem lengths
as interaction features with the wider sequence representation.

## Misclassification Analysis

- Neural models correctly classify 1,939 sites that GB gets wrong
- GB correctly classifies only 636 sites that both neural models miss
- Sites where neural wins have **smaller loop sizes** (2.98 vs 4.25 for GB-wins) and
  **shorter dist_to_junction** (1.33 vs 1.91)
- This suggests neural models better handle sites in tight structural contexts where
  the wider sequence context disambiguates editing potential

## CNN Saliency: Which Sequence Positions Matter

- The CNN saliency peaks sharply at position 100 (the edit site C), confirming it learns
  the identity of the central base
- Differential saliency (pos - neg) shows the immediate +/- 5nt context matters most for
  distinguishing positive from negative sites
- But substantial signal extends to +/- 50nt, consistent with the k-mer window experiment

## Interpretation

The A3B hand features fail not because the features are wrong, but because they are
**too local**. A3B editing specificity depends on:

1. **Global sequence composition** (captured by RNA-FM PC1, AUROC=0.727 alone)
2. **Extended motif context** (dinucleotide frequencies in +/- 50nt windows)
3. **Interaction between local structure and wider context** (stem lengths matter
   only when combined with embedding information)

This contrasts with A3A, where local motif + loop geometry suffices (GB AUROC=0.923).
The difference may reflect A3A's stronger intrinsic sequence preference (TC motif) vs
A3B's weaker motif specificity, requiring more context for discrimination.

## Files

- Results: `experiments/multi_enzyme/outputs/a3b_neural_analysis/a3b_neural_analysis_results.json`
- Figures: `experiments/multi_enzyme/outputs/a3b_neural_analysis/*.png`
- Feature comparison: `experiments/multi_enzyme/outputs/a3b_neural_analysis/misclass_feature_comparison.csv`
- Experiment script: `experiments/multi_enzyme/exp_a3b_neural_analysis.py`
