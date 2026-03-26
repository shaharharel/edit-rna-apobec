# AID/APOBEC DNA Mutation Hotspots vs RNA Editing Predictions

## Overview

This analysis compares COSMIC DNA mutation signatures (SBS2, SBS13, SBS84/85) with
APOBEC RNA editing predictions from our GB model, using 1.69M ClinVar C>U variants
scored for A3A editing probability.

## Mutation Signature Definitions

| Signature | Enzyme | DNA Motif | RNA Equivalent | N ClinVar |
|-----------|--------|-----------|----------------|-----------|
| SBS2      | APOBEC3A/3B | TCA>TTA | UCA context | 65,847 |
| SBS13     | APOBEC3A/3B | TCT>TTT | UCU context | 71,515 |
| SBS84/85  | AID | WRC>WYC | AC/GC context, m2=A/U | 388,605 |
| A3G-like  | APOBEC3G | CC>CT | CC context | 515,463 |

## Key Findings

### 1. AID and APOBEC target mutually exclusive motifs

Zero sites have both AID (WRC) and APOBEC (TC) contexts. This is a strict motif
exclusion: AID requires upstream purine (A/G), APOBEC requires upstream pyrimidine (T/U).
At the DNA mutation level, these are completely non-overlapping mutation programs.

### 2. The A3A RNA editing model scores structure, not motif

The GB model was trained with TC-matched negatives (both positives and negatives
are ~86% TC context). As a result, TC motif is not discriminative in the model.
When applied to ClinVar:

| Upstream Context | N variants | Mean GB | Median GB | % >= 0.5 |
|-----------------|-----------|---------|-----------|----------|
| TC (APOBEC)     | 358,349   | 0.377   | 0.274     | 36.7%    |
| CC (A3G)        | 515,463   | 0.877   | 0.958     | 94.8%    |
| AC              | 392,737   | 0.839   | 0.944     | 89.8%    |
| GC              | 426,288   | 0.803   | 0.921     | 85.4%    |

The model assigns LOWER scores to TC-context sites than to any other context.
This is because TC-context ClinVar sites are enriched in CpG dinucleotides
(TCG is the most common TC trinucleotide), and CpG sites tend to be in
base-paired, structured regions that are unfavorable for editing.

**Implication**: The GB model's ClinVar predictions are driven primarily by RNA
secondary structure features (loop position, unpaired status), not by sequence
motif. High-scoring non-TC sites are structurally favorable for editing but would
not be recognized by A3A's catalytic domain.

### 3. Pathogenic enrichment is strongest at APOBEC DNA hotspots

Despite the overall score inversion, the pathogenic enrichment analysis reveals
that the APOBEC DNA hotspot context (SBS2/SBS13) shows the STRONGEST pathogenic
enrichment among high-scoring predictions:

| Context | Path% (high-score) | Path% (low-score) | Odds Ratio |
|---------|--------------------|--------------------|------------|
| **APOBEC DNA hotspot** | **5.74%** | **3.65%** | **1.61** |
| A3G (CC) | 5.31% | 4.14% | 1.30 |
| APOBEC TC other | 3.81% | 3.15% | 1.22 |
| Other | 3.34% | 3.46% | 0.96 |
| AID (WRC) | 5.29% | 6.63% | 0.79 |

**The OR=1.61 for APOBEC DNA hotspots means**: among the ~18% of APOBEC-context
sites that the model predicts as structurally favorable for editing, pathogenic
variants are 61% more likely than among the 82% predicted as unfavorable.

Conversely, AID-context sites show pathogenic DEPLETION (OR=0.79) among
high-scoring predictions. This makes biological sense: AID mutations cause
disease through a different mechanism (somatic hypermutation in B cells),
and the structural features that predict A3A editing are not relevant to AID.

### 4. Gene-level overlap is extensive but uninformative

At the gene level, overlap is nearly complete: 14,603 genes have high-scoring
predictions in ALL three contexts (APOBEC, AID, A3G). This reflects the fact
that most human genes are large enough to contain multiple C sites in every
trinucleotide context, and the model's structure-based scoring assigns high
scores broadly.

Top genes by APOBEC-context high-scoring variants (TTN, ATM, BRCA2, APC, NEB)
are simply the largest genes in the genome — this is a gene-length artifact,
not a biological signal.

### 5. RNA vs DNA substrate preferences differ by enzyme

| Enzyme | DNA Motif | RNA Motif | Same? |
|--------|-----------|-----------|-------|
| APOBEC3A | TC (SBS2/13) | TC (strong) | YES |
| APOBEC3B | TC (SBS2/13) | mixed, no strong bias | NO |
| APOBEC3G | CC | CC (strong) | YES |
| AID | WRC | N/A (no known RNA editing) | N/A |

A3A and A3G maintain the same motif preference for both DNA mutagenesis and
RNA editing. A3B diverges: strong TC preference on DNA but mixed context on RNA.
This may reflect different binding modes or substrate presentation for the two
nucleic acid types.

## Clinical Implications

1. **Sites at APOBEC DNA mutation hotspots that are also predicted RNA editing
   targets** represent a special category: they may be subject to BOTH somatic
   DNA mutation in cancer AND physiological RNA editing. The OR=1.61 pathogenic
   enrichment suggests these dual-target sites are clinically relevant.

2. **AID mutation hotspots and APOBEC RNA editing sites do not overlap** at the
   motif level. AID-driven B-cell lymphoma mutations and APOBEC-driven RNA editing
   are biologically independent processes affecting different genomic positions.

3. **The model's structure-driven scoring** means that ClinVar enrichment results
   should be interpreted as: "pathogenic variants in structurally accessible
   positions within TC-context regions are enriched" — the signal combines
   both motif specificity (from the TC context filter) and structural accessibility
   (from the GB model's learned features).

## Methods

- 1,692,837 ClinVar C>U variants scored by A3A GB model (p_edited_gb) and RNAsee
- Trinucleotide context decoded from 46-dim hand-crafted feature cache
- Signature classification: SBS2 (UCA), SBS13 (UCU), AID/WRC (upstream A/G + m2 A/U), A3G (CC)
- Pathogenic enrichment: odds ratio of pathogenic fraction in high-score (>=0.5) vs low-score (<0.5)

## Output Files

- `experiments/multi_enzyme/outputs/cosmic_overlap/high_score_predictions_with_signature.csv` — 1.34M high-score variants annotated with signature class
- `experiments/multi_enzyme/outputs/cosmic_overlap/gene_level_signature_summary.csv` — 18,626 genes with per-signature counts
- `experiments/multi_enzyme/outputs/cosmic_overlap/analysis_summary.json` — summary statistics
