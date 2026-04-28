# v4 Data Prep — Two Trinuc-Matched Negative Sets
_Generated 2026-04-27T22:18:16_

## Overview
v3 negatives were biased: positives 38.9% TC vs negatives 57.2% TC; positives 19.8% CpG vs negatives 9.1% CpG. The model learned anti-TCW polarity. v4 corrects this with two parallel negative sets:
- **v4_cancer_matched** — negatives match TCGA + PCAWG-coding pan-cancer C>T trinuc distribution (transfer claim).
- **v4_cds_unbiased** — negatives match the genome CDS-C trinuc distribution (predictor claim).
- APOBEC1 sites (v3 enzyme=='Neither', n=206) are excluded — no DNA-editing analog.
- Random seed: 20260427.

## Trinucleotide distributions (16 bins, strand-collapsed N-C-N)
| Trinuc | Cancer C>T % | CDS-C % |
|--------|------------:|--------:|
| ACA |  1.35 |  6.92 |
| ACC |  2.32 |  6.45 |
| ACG |  6.78 |  2.64 |
| ACT |  1.23 |  5.17 |
| CCA |  4.65 |  9.25 |
| CCC |  6.18 |  8.05 |
| CCG |  8.04 |  4.10 |
| CCT |  5.19 |  8.37 |
| GCA |  2.07 |  7.43 |
| GCC |  4.11 |  8.49 |
| GCG |  9.38 |  3.50 |
| GCT |  2.89 |  7.54 |
| TCA | 10.78 |  6.92 |
| TCC | 13.93 |  6.78 |
| TCG | 13.51 |  2.30 |
| TCT |  7.60 |  6.07 |

_Cancer total: 1,158,068 C>T mutations across 10 cancers (TCGA+PCAWG)._
_CDS-C total: 8,446,858 C positions in pan-cancer CDS panel (hg19, 0-indexed)._

## v3 vs v4 negative trinucleotide distributions
All percentages computed from the actual sequences in the JSON files.

| Trinuc | v4 pos % | v3 neg % | v4_cancer neg % | Cancer target % | Δ_cancer | v4_cds neg % | CDS target % | Δ_cds |
|--------|---------:|--------:|----------------:|----------------:|---------:|-------------:|-------------:|------:|
| ACA |  5.44 |  3.48 |  1.37 |  1.35 | +0.01 |  6.93 |  6.92 | +0.01 |
| ACC |  4.72 |  3.25 |  2.34 |  2.32 | +0.01 |  6.47 |  6.45 | +0.02 |
| ACG |  2.65 |  0.91 |  6.75 |  6.78 | -0.03 |  2.64 |  2.64 | -0.00 |
| ACT |  3.21 |  3.65 |  1.23 |  1.23 | +0.00 |  5.16 |  5.17 | -0.01 |
| CCA |  8.32 |  5.28 |  4.67 |  4.65 | +0.03 |  9.27 |  9.25 | +0.02 |
| CCC |  5.42 |  5.64 |  6.16 |  6.18 | -0.02 |  8.02 |  8.05 | -0.03 |
| CCG |  5.34 |  2.05 |  8.03 |  8.04 | -0.00 |  4.10 |  4.10 | +0.00 |
| CCT |  6.21 |  5.34 |  5.20 |  5.19 | +0.01 |  8.36 |  8.37 | -0.01 |
| GCA |  5.23 |  3.57 |  2.05 |  2.07 | -0.02 |  7.42 |  7.43 | -0.01 |
| GCC |  5.69 |  4.55 |  4.14 |  4.11 | +0.02 |  8.48 |  8.49 | -0.00 |
| GCG |  3.51 |  1.19 |  9.39 |  9.38 | +0.01 |  3.50 |  3.50 | +0.00 |
| GCT |  5.40 |  3.90 |  2.91 |  2.89 | +0.02 |  7.53 |  7.54 | -0.01 |
| TCA | 13.94 | 16.50 | 10.83 | 10.78 | +0.05 |  6.93 |  6.92 | +0.01 |
| TCC |  9.98 | 17.40 | 13.96 | 13.93 | +0.03 |  6.80 |  6.78 | +0.01 |
| TCG |  8.32 |  4.89 | 13.35 | 13.51 | -0.16 |  2.30 |  2.30 | -0.00 |
| TCT |  6.63 | 18.40 |  7.62 |  7.60 | +0.03 |  6.07 |  6.07 | -0.00 |

### Aggregated metrics
| Subset | TC% | CpG% |
|---|---:|---:|
| v4 positives | 38.87 | 19.82 |
| v3 negatives | 57.19 |  9.05 |
| v4_cancer negatives | 45.77 | 37.51 |
| Cancer target | 45.82 | 37.70 |
| v4_cds negatives | 22.10 | 12.54 |
| CDS-C target | 22.08 | 12.54 |

## Counts
- v3 positives total: 7564
- v3 'Neither' positives (APOBEC1) excluded: 206
- v4 positives (post-exclusion, shared by both versions): **7358**
- v4_cancer_matched: 14678 sites = 7358 pos + 7320 neg
- v4_cds_unbiased: 14701 sites = 7358 pos + 7343 neg

## Jaccard overlap of v4 negative sets
- |cancer ∩ cds| = 4576
- |cancer ∪ cds| = 10087
- Jaccard = 0.4537

## Coverage checks
- **v4_cancer_matched**: sequences 14678/14678, loop 14678/14678, structure 14678/14678, RNA-FM=yes
- **v4_cds_unbiased**: sequences 14701/14701, loop 14701/14701, structure 14701/14701, RNA-FM=yes

## File paths

**v4_cancer_matched**
- splits CSV: `/Users/shaharharel/Documents/github/edit-rna-apobec/data/processed/multi_enzyme/splits_multi_enzyme_v4_cancer_matched.csv`
- sequences JSON: `/Users/shaharharel/Documents/github/edit-rna-apobec/data/processed/multi_enzyme/multi_enzyme_sequences_v4_cancer_matched.json`
- loop position CSV: `/Users/shaharharel/Documents/github/edit-rna-apobec/data/processed/multi_enzyme/loop_position_per_site_v4_cancer_matched.csv`
- structure cache NPZ: `/Users/shaharharel/Documents/github/edit-rna-apobec/data/processed/embeddings/structure_cache_multi_enzyme_v4_cancer_matched.npz`
- RNA-FM embeddings: `/Users/shaharharel/Documents/github/edit-rna-apobec/data/processed/embeddings/rnafm_v4_cancer_matched.npz`

**v4_cds_unbiased**
- splits CSV: `/Users/shaharharel/Documents/github/edit-rna-apobec/data/processed/multi_enzyme/splits_multi_enzyme_v4_cds_unbiased.csv`
- sequences JSON: `/Users/shaharharel/Documents/github/edit-rna-apobec/data/processed/multi_enzyme/multi_enzyme_sequences_v4_cds_unbiased.json`
- loop position CSV: `/Users/shaharharel/Documents/github/edit-rna-apobec/data/processed/multi_enzyme/loop_position_per_site_v4_cds_unbiased.csv`
- structure cache NPZ: `/Users/shaharharel/Documents/github/edit-rna-apobec/data/processed/embeddings/structure_cache_multi_enzyme_v4_cds_unbiased.npz`
- RNA-FM embeddings: `/Users/shaharharel/Documents/github/edit-rna-apobec/data/processed/embeddings/rnafm_v4_cds_unbiased.npz`

## Auxiliary files
- Cancer trinuc CSV: `/Users/shaharharel/Documents/github/edit-rna-apobec/data/processed/multi_enzyme/cancer_ct_trinuc_distribution.csv`
- CDS-C trinuc CSV: `/Users/shaharharel/Documents/github/edit-rna-apobec/data/processed/multi_enzyme/cds_c_trinuc_distribution.csv`
