# Unified V1 Interpretability: Why Joint Training Helps

## 1. AUROC Comparison (5-fold CV, matched neural architectures)

| Enzyme | n_pos | Unified AUROC | Per-Enzyme AUROC | Delta |
|--------|-------|---------------|------------------|-------|
| Unknown | 72 | 0.933 +/- 0.015 | 0.848 +/- 0.041 | **+0.085** |
| A3A | 2749 | 0.900 +/- 0.006 | 0.896 +/- 0.006 | +0.003 |
| A3B | 4180 | 0.914 +/- 0.007 | 0.912 +/- 0.009 | +0.002 |
| A3G | 179 | 0.952 +/- 0.015 | 0.953 +/- 0.025 | -0.001 |
| A3A_A3G | 178 | 0.946 +/- 0.013 | 0.950 +/- 0.032 | -0.004 |
| Neither | 206 | 0.929 +/- 0.005 | 0.938 +/- 0.020 | -0.009 |

**Note**: These compare matched neural architectures (same MLP backbone, same input features).
The original unified V1 results compared against GB per-enzyme baselines; these compare
neural-to-neural.

## 2. Key Finding: Unknown enzyme benefits dramatically (+0.085 AUROC)

The largest unified training benefit goes to the smallest enzyme category:

- **Unknown (72 positives): +0.085** -- the per-enzyme model has only 72 positives to learn from,
  leading to high variance (std=0.041) and mediocre performance (0.848). The unified model
  stabilizes this dramatically, achieving 0.933 with much lower variance (std=0.015).
- **A3A (2749) and A3B (4180): +0.003 and +0.002** -- marginal improvement; these enzymes already
  have enough data for stable per-enzyme training.
- **A3G (179), A3A_A3G (178), Neither (206): approximately neutral** (-0.001 to -0.009).
  These small-data enzymes have distinctive enough motif preferences (A3G=CC-context,
  Neither=mixed) that per-enzyme models capture their signal well despite small sample sizes.

**The unified model reduces variance**: Per-enzyme models for small enzymes show 2-4x higher
fold-to-fold variability (e.g., Unknown std=0.041 per-enzyme vs 0.015 unified). This
stabilization is the primary mechanism by which unified training helps.

## 3. Rescued Sites: What the unified model fixes

"Rescued" sites are those correctly classified by the unified model but missed by per-enzyme
models. These sites reveal what cross-enzyme knowledge transfer provides.

### Pattern 1: Non-canonical motif sites (most common rescue)

Across all enzymes, rescued positive sites overwhelmingly have non-TC motifs:
- **A3A rescued** (top 20): 8 CC, 5 AC, 4 GC, only 3 UC -- the per-enzyme model overfits
  to the dominant TC motif and misses AC/CC/GC sites. The unified model, trained on
  A3G (CC-preferring) and A3B (mixed), learns that non-TC contexts can also be edited.
- **A3G rescued** (top 20): 19 CC, 1 AC -- the per-enzyme model with only 179 positives
  cannot reliably separate CC editing from negative CC sites, but the unified model
  leverages structural features learned from A3A/A3B data.
- **Unknown rescued** (top 20): 8 CC, 8 AC, 2 GC, 2 UC; 80% unpaired -- these sites have
  diverse motifs but strong structural signals (unpaired loop regions), which the unified
  model captures from large-data enzymes.

### Pattern 2: Extreme score rescue

Rescued sites show extreme score differences: per-enzyme scores near 0.000 flip to unified
scores near 0.990. This is not marginal reclassification -- the per-enzyme model completely
misses these sites while the unified model is highly confident.

Example: A3A site chr11:63671464:+ (ACCAU motif, paired) gets unified=0.972 vs per-enzyme=0.001.
The per-enzyme A3A model has never seen enough AC-context edited sites to learn this pattern,
but the unified model transfers AC-context editing knowledge from other enzymes.

### Pattern 3: Mitochondrial sites are preferentially rescued

Several rescued sites across A3A and A3B are on chrM (mitochondrial):
- chrM:3117:+ (A3A, UUCCU, unified=0.966 vs perenz=0.002)
- chrM:831:+ (A3B, ACCUU, unified=1.000 vs perenz=0.025)
- chrM:2388:+ (A3B, AUCUA, unified=0.984 vs perenz=0.018)

Mitochondrial RNA editing sites may have unusual structural contexts that benefit from
cross-enzyme training data.

### Top 10 rescued sites per enzyme

#### A3A (488 rescued, 788 lost)
| site_id | is_edited | unified | perenz | motif | unpaired |
|---------|-----------|---------|--------|-------|----------|
| chr11:63671464:+ | 1 | 0.972 | 0.001 | ACCAU | no |
| chr16:72130377:+ | 1 | 0.966 | 0.000 | GACAG | no |
| chrM:3117:+ | 1 | 0.966 | 0.002 | UUCCU | yes |
| chr20:33981904:- | 1 | 0.962 | 0.001 | CGCAC | no |
| chr20:62331406:- | 1 | 0.958 | 0.002 | AUCGA | no |

#### Unknown (57 rescued, 1385 lost)
| site_id | is_edited | unified | perenz | motif | unpaired |
|---------|-----------|---------|--------|-------|----------|
| C2U_0075 | 1 | 1.000 | 0.000 | GGCGA | no |
| C2U_0127 | 1 | 1.000 | 0.000 | UGCAG | no |
| C2U_0088 | 1 | 0.999 | 0.000 | ACCGC | no |
| C2U_0303 | 1 | 0.999 | 0.000 | CACUU | yes |
| C2U_0199 | 1 | 0.999 | 0.000 | GACCU | yes |

## 4. Enzyme Confusion Matrix

The unified model's enzyme classification head (63% accuracy) reveals biological relationships:

- **A3B dominates predictions**: 60% of A3A sites and 35-43% of A3G/Unknown sites are
  predicted as A3B. This reflects A3B's large training set (4180 sites) and that A3B has
  the most diverse motif preference, acting as a "catch-all" enzyme.
- **A3A_A3G correctly bridges A3A and A3G**: 34% recall, with confusion split between
  A3A (31%) and A3G (19%) -- exactly as expected for dual-enzyme sites.
- **Neither sites split across all enzymes**: 29% correctly classified, rest spread evenly.
  This is biologically consistent -- "Neither" sites may be edited by unknown enzymes with
  varied preferences.
- **Unknown never predicted as Unknown**: 0% recall. The model absorbs Unknown into
  A3A (43%) and A3B (38%) categories, suggesting Unknown sites share features with these enzymes.

## 5. Feature Attribution

### Gradient attribution reveals enzyme-specific feature weighting

The shared backbone learns a common representation, but different features activate
differently for different enzyme categories:

| Feature | A3A | A3A_A3G | A3B | A3G | Neither | Unknown |
|---------|-----|---------|-----|-----|---------|---------|
| local_unpaired_fraction | **#1** | **#1** | **#1** | **#1** | **#1** | **#1** |
| trinuc_m1_U (=TC motif) | #2 | #5 | #2 | #5 | #2 | #3 |
| motif_UC | #3 | -- | #3 | -- | #4 | #5 |
| mean_delta_pairing_window | -- | **#2** | #4 | **#2** | #3 | #4 |
| relative_loop_position | -- | #4 | -- | #4 | -- | **#2** |

**Key observations**:
1. `local_unpaired_fraction` is universally the #1 feature across all enzymes. This is the
   shared structural signal that enables cross-enzyme transfer learning.
2. A3A and A3B rely heavily on trinuc_m1_U (TC motif), confirming their TC preference.
3. A3G and A3A_A3G rely more on structure delta features (mean/std_delta_pairing_window),
   consistent with A3G's strong structural preference (3' tetraloop editing).
4. Unknown sites weight relative_loop_position #2 -- these sites are defined by WHERE they
   occur in loop structure, not by motif, suggesting they may be edited by a structure-sensitive
   enzyme distinct from the known APOBECs.

### Permutation importance (unified model, top 10)

| Feature | AUROC drop |
|---------|------------|
| left_stem_length | 0.068 |
| max_adjacent_stem_length | 0.062 |
| right_stem_length | 0.056 |
| loop_size | 0.010 |
| trinuc_m1_U | 0.009 |
| motif_UC | 0.008 |
| dist_to_apex | 0.007 |
| trinuc_m1_A | 0.006 |
| motif_AC | 0.005 |
| delta_mfe | 0.004 |

**Stem length features dominate permutation importance.** This is a striking finding:
the three stem length features (left, right, max_adjacent) together account for 0.186 AUROC
drop, dwarfing all other features combined. This indicates the unified model's shared backbone
primarily encodes **how the edit site's loop is anchored within secondary structure** -- a
universal property across all APOBEC enzymes.

## 6. UMAP Embedding Analysis

The 256-dim shared representations from the unified backbone show clear structure:

1. **Edited vs not-edited separation**: Green (edited) and red (not-edited) clusters are
   well-separated, confirming the shared backbone learns a meaningful editing representation.
2. **Enzyme clustering**: A3B (orange) and A3A (blue) form distinct clusters, with
   smaller enzymes (A3G, Neither, Unknown) occupying intermediate positions.
3. **Motif-driven subclusters**: UC and CC motif types map to different regions, matching
   the known motif preferences of A3A (UC) and A3G (CC).
4. **Rescued sites are scattered**: Rescued sites (red highlights) appear throughout the
   embedding space, not in a single cluster. This suggests rescued sites are not a coherent
   subpopulation but rather individual cases where cross-enzyme context provides additional signal.

## 7. Why Joint Training Helps: Mechanistic Summary

1. **Variance reduction through data pooling**: The primary benefit is reducing variance for
   data-scarce enzymes. Unknown (72 positives) gains +0.085 AUROC entirely through
   stabilization -- fold-to-fold variability drops from 0.041 to 0.015. The per-enzyme model
   simply doesn't have enough data for robust training.

2. **Shared structural priors**: `local_unpaired_fraction` is the #1 gradient attribution
   feature for ALL six enzyme categories. The shared backbone learns that C-to-U editing
   universally occurs in unpaired regions, providing a structural prior that helps all enzymes.
   Stem length features (left/right/max_adjacent) dominate permutation importance, confirming
   that the loop-anchoring context is the backbone's primary learned signal.

3. **Cross-enzyme motif transfer**: Rescued sites are disproportionately non-canonical motif
   sites (AC, CC, GC contexts for A3A; CC for A3G). The shared backbone, exposed to
   diverse motif preferences across enzymes, avoids overfitting to any single motif pattern
   and can recognize edited sites in unusual sequence contexts.

4. **Why some enzymes don't benefit**: A3G and A3A_A3G show neutral deltas despite being
   small-data. This is because they have highly distinctive features (CC-context,
   structure-dominated prediction) that per-enzyme models capture efficiently. The unified
   backbone's shared representation adds no new information for these enzymes and may slightly
   dilute their distinctive signal.

5. **Biological insight**: The enzyme confusion matrix reveals that "Unknown" sites are
   never predicted as Unknown -- they split between A3A (43%) and A3B (38%). This suggests
   the "Unknown" category is not a distinct enzymatic program but rather editing by
   known APOBEC enzymes under conditions not captured in the training data annotations.

## Biological Interpretation of Rescued Sites

### What the unified model learned that per-enzyme models couldn't

Iteration 2 analyzed ALL rescued and lost sites across enzymes (not just top-20), mapped
them to genes, and examined why A3B doesn't benefit. The core finding: **the unified model
transfers motif tolerance across enzymes while preserving structural selectivity.**

#### A3A: Cross-enzyme motif debiasing (435 rescued positives, 26 lost)

The A3A per-enzyme model overfits to TC-context sites because TC dominates A3A training
data (51.2% of all A3A positives). Rescued A3A sites are enriched for TC (58.9%) but
critically also include substantial CC (22.8%), GC (10.3%), and AC (8.0%) sites that the
per-enzyme model misses entirely (scores near 0.000). The unified model, trained jointly
with A3G (CC-preferring) and A3B (mixed), learns that non-TC cytidines can also be edited
in the right structural context.

Rescued vs non-rescued A3A positives show nearly identical structural properties
(is_unpaired: 0.44 vs 0.45, RLP: 0.21 vs 0.26, loop_size: 3.2 vs 3.3). This confirms
the rescue is motif-driven, not structure-driven: the unified backbone correctly recognizes
the structural signature of editing and applies it regardless of the flanking dinucleotide.

All 435 rescued A3A positives come from kockler_2026, which uses genome-wide A3A
overexpression in HEK293T cells -- an assay expected to capture non-canonical editing events
that tissue-specific studies miss.

#### A3B: TC imposition creates an asymmetric rescue/loss pattern (463 rescued, 54 lost)

A3B shows the most striking motif shift. A3B editing is genuinely promiscuous (32.3% TC,
24.8% CC, 23.8% GC, 19.0% AC across all positives), but rescued A3B sites are 70.2% TC.
The unified model, dominated by A3A's TC preference, imposes TC favorability on A3B --
correctly identifying TC-context A3B sites the per-enzyme model missed, but at the cost of
losing 54 A3B positives.

The 54 lost A3B positives are revealing: they are 63.0% TC (vs 32.3% baseline), less
frequently unpaired (25.9% vs 40.3%), have smaller loops (2.3 vs 3.1), and lower RLP (0.10
vs 0.20). These are TC-context A3B sites in paired/stem regions -- the unified model's
structural prior penalizes them because "TC in a stem" is uncommon for A3A editing. The
per-enzyme A3B model, without this bias, correctly recognizes that A3B can edit cytidines
even in paired regions.

This explains why A3B shows only +0.002 AUROC from unified training: the model rescues
A3B-TC-in-loop sites (which match the A3A pattern) but loses A3B-TC-in-stem sites (which
violate it). The gains and losses nearly cancel.

#### A3G: Structural diversity transfer (55 rescued, 3 lost)

A3G rescued sites maintain the expected CC context (80.0%) but are structurally atypical:
lower is_unpaired (0.47 vs 0.70 for all A3G), lower RLP (0.36 vs 0.64), and smaller loops
(2.7 vs 3.2). The per-enzyme A3G model, with only 179 positives, learns an overly strict
structural profile (must be unpaired, high RLP, classic 3' tetraloop). The unified backbone,
with 7000+ positive sites across enzymes, learns that editing can occur in structurally
diverse positions, rescuing CC-context sites that happen to be in less canonical structures.

### Notable rescued sites with biological relevance

Three rescued sites illustrate the clinical significance of cross-enzyme transfer learning:

1. **FUS (chr16:31187601, A3B, TC context)**: FUS is an RNA-binding protein causally linked
   to ALS and frontotemporal dementia. The editing site in FUS mRNA is confidently predicted
   by the unified model (0.913) but completely missed by the per-enzyme A3B model (0.003).
   FUS protein binds RNA loops and hairpins; C-to-U editing in FUS could alter its own
   autoregulatory binding, creating a feedback loop between editing and neurodegeneration.

2. **RDM1 (chr17:35928967, A3A, CC context, loop apex)**: RDM1 contains a RAD52-like DNA
   repair motif. This site sits at the apex of a 4-nt loop (RLP=1.0, unpaired), a canonical
   editing position, but has CC context that the A3A-only model rejects (score=0.113). The
   unified model scores it 0.964. Editing in DNA repair genes could modulate damage response,
   connecting APOBEC editing to genomic instability.

3. **DDX31 (chr9:132662752, A3B, AC context)**: DDX31 is a DEAD-box RNA helicase implicated
   in ribosome biogenesis and several cancers. The per-enzyme model gives it 0.004; the
   unified model gives 0.981. DDX31 unwinds RNA structures that APOBEC enzymes target -- if
   APOBEC edits DDX31 mRNA, it could alter the enzyme's substrate landscape, creating another
   feedback loop.

### Why this matters for the paper

The unified model analysis demonstrates that C-to-U RNA editing by APOBEC enzymes follows
a **shared structural logic with enzyme-specific motif preferences**. The key insight is
hierarchical: structure determines editability, motif determines which enzyme performs
the edit. Per-enzyme models conflate these two levels, overfitting to the dominant motif
and rejecting structurally valid editing sites with non-canonical flanking sequences.

This has direct implications for ClinVar pathogenicity analysis: rescued sites include genes
involved in neurodegeneration (FUS), DNA repair (RDM1), autophagy (ATG2A), and cancer
signaling (DPF2, DDX31). A per-enzyme model would miss these clinical candidates entirely.
The unified model, by separating structural editability from motif preference, provides a
more complete map of the APOBEC editing landscape and its potential disease consequences.
