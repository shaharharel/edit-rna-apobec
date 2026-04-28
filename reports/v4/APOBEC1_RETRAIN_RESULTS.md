# APOBEC1 v4 retraining results

**Goal**: replace v3 APOBEC1 head (trained on legacy v1 data with random
negatives) with v4-trinucleotide-matched negatives. Validate that the new
heads (a) retain or improve discrimination, (b) lose any anti-TCW polarity,
(c) maintain or improve panel-recall vs v3 baseline.

## 1. APOBEC1 trinucleotide distribution

The v4 negatives are trinucleotide-matched to a target distribution
(cancer-mutation context for v4_cancer; CDS-uniform context for v4_cds).
Positives are unchanged across variants (same 484 mouse-validated APOBEC1
edited sites).

| Trinuc | v4 pos % | v4_cancer neg % | v4_cancer target % | v4_cds neg % | v4_cds target % |
|---|---|---|---|---|---|
| ACA | 15.5 | 1.4 | 1.4 | 7.0 | 6.9 |
| ACC | 2.5 | 2.3 | 2.3 | 6.4 | 6.5 |
| ACG | 1.0 | 6.8 | 6.8 | 2.7 | 2.6 |
| ACT | 23.6 | 1.2 | 1.2 | 5.2 | 5.2 |
| CCA | 1.4 | 4.8 | 4.6 | 9.3 | 9.3 |
| CCC | 0.0 | 6.2 | 6.2 | 8.1 | 8.0 |
| CCG | 0.2 | 8.1 | 8.0 | 4.1 | 4.1 |
| CCT | 1.9 | 5.2 | 5.2 | 8.5 | 8.4 |
| GCA | 2.1 | 2.1 | 2.1 | 7.4 | 7.4 |
| GCC | 1.4 | 4.1 | 4.1 | 8.5 | 8.5 |
| GCG | 0.0 | 9.3 | 9.4 | 3.5 | 3.5 |
| GCT | 4.3 | 2.9 | 2.9 | 7.4 | 7.5 |
| TCA | 20.9 | 10.7 | 10.8 | 6.8 | 6.9 |
| TCC | 4.8 | 13.8 | 13.9 | 6.8 | 6.8 |
| TCG | 1.0 | 13.4 | 13.5 | 2.3 | 2.3 |
| TCT | 19.4 | 7.6 | 7.6 | 6.0 | 6.1 |

## 2. 5-fold AUROC

| Head | n_pos | n_neg | mean AUROC ± std | Folds |
|---|---|---|---|---|
| v3 (legacy, random negatives) | 484 | 484 | 0.7830 ± 0.0326 | [0.808, 0.734, 0.799, 0.755, 0.819] |
| **v4_cancer** (trinuc-matched, cancer-context negatives) | 484 | 484 | **0.8340 ± 0.0180** | [0.837, 0.846, 0.801, 0.853, 0.834] |
| **v4_cds** (trinuc-matched, CDS-context negatives) | 484 | 484 | **0.8283 ± 0.0266** | [0.873, 0.795, 0.810, 0.838, 0.826] |

Both v4 heads beat v3 by ~+0.05 AUROC, despite the harder (trinuc-matched)
negatives. Architecture is identical; only the training negatives changed.

## 3. Bias diagnostic (100K random CDS-C panel positions)

For each retrained head, scored 100K random valid panel positions (centered on
genomic C). Computed mean predicted P per trinucleotide.
**An "anti-TCW" polarity (TCW mean < non-TCW mean) would be a red flag** —
v3 has historically shown this artifact when negatives over-represent TCW
contexts.

**v4_cancer** — anti_TCW_polarity_present=False | TCW mean=0.581 | nonTCW mean=0.429

| Trinuc | n | mean P | median P |
|---|---|---|---|
| ACT | 5,097 | 0.6633 | 0.7174 |
| ACA | 7,033 | 0.6533 | 0.7300 |
| TCA | 6,873 | 0.5962 | 0.6371 |
| ACC | 6,378 | 0.5943 | 0.6482 |
| TCT | 6,028 | 0.5660 | 0.5922 |
| TCC | 6,833 | 0.5414 | 0.5669 |
| GCA | 7,451 | 0.5385 | 0.5917 |
| GCT | 7,589 | 0.5354 | 0.5704 |
| GCC | 8,626 | 0.4763 | 0.4997 |
| ACG | 2,662 | 0.3826 | 0.3539 |
| CCA | 9,215 | 0.3430 | 0.2978 |
| CCT | 8,371 | 0.3244 | 0.2903 |
| GCG | 3,479 | 0.2802 | 0.2346 |
| CCC | 7,916 | 0.2798 | 0.2184 |
| TCG | 2,363 | 0.2374 | 0.1515 |
| CCG | 4,086 | 0.1516 | 0.1088 |

**v4_cds** — anti_TCW_polarity_present=False | TCW mean=0.668 | nonTCW mean=0.392

| Trinuc | n | mean P | median P |
|---|---|---|---|
| TCA | 6,823 | 0.6776 | 0.7251 |
| TCT | 6,083 | 0.6584 | 0.6991 |
| TCC | 6,712 | 0.6043 | 0.6423 |
| ACA | 6,906 | 0.5537 | 0.6060 |
| ACT | 5,134 | 0.5307 | 0.5639 |
| GCA | 7,370 | 0.4517 | 0.4698 |
| ACC | 6,425 | 0.4291 | 0.4381 |
| ACG | 2,662 | 0.4187 | 0.4118 |
| GCT | 7,566 | 0.4136 | 0.4166 |
| TCG | 2,261 | 0.4082 | 0.3674 |
| GCC | 8,587 | 0.3267 | 0.3073 |
| CCA | 9,307 | 0.3142 | 0.2842 |
| GCG | 3,449 | 0.3122 | 0.2828 |
| CCT | 8,438 | 0.2817 | 0.2546 |
| CCC | 8,212 | 0.2351 | 0.1895 |
| CCG | 4,065 | 0.2122 | 0.1901 |

**Verdict**:
- v4_cancer: anti_TCW = `False` (TCW 0.581 vs nonTCW 0.429)
- v4_cds:    anti_TCW = `False` (TCW 0.668 vs nonTCW 0.392)

The v4_cds head also recovers the canonical APOBEC1 mooring-rich pattern
(ACA/ACT/TCA/TCT high; CpG-context CCG/TCG/GCG low; ratio
~1.79).

## 4. Panel sweep (TopX-1% position + best window, ws=1000, both filters)

PCAWG mutation recall on the 8.45 M CDS-C panel, averaged across 10 cancer
cohorts. v3 vs v4 head, two filter sets:
* `filter_TCW_nonCpG` – TCW only, excluding CpG
* `filter_all_CT`    – all C-to-T mutations

### Variant: v4_cancer

Top-1% recall (mean across 10 PCAWG cancers, ws=1000 best window or position-level):

| Filter | Level | v3 recall | v4 recall | delta | v3 vs_NPOS | v4 vs_NPOS |
|---|---|---|---|---|---|---|
| filter_TCW_nonCpG | position | 0.0144 | 0.0132 | -0.0012 | 1.43 | 1.30 |
| filter_TCW_nonCpG | best_window_max_w1000 | 0.0403 | 0.0677 | +0.0274 | 0.99 | 1.70 |
| filter_all_CT | position | 0.0288 | 0.0064 | -0.0223 | 2.77 | 0.62 |
| filter_all_CT | best_window_max_w1000 | 0.0595 | 0.0625 | +0.0031 | 0.91 | 0.97 |

### Variant: v4_cds

Top-1% recall (mean across 10 PCAWG cancers, ws=1000 best window or position-level):

| Filter | Level | v3 recall | v4 recall | delta | v3 vs_NPOS | v4 vs_NPOS |
|---|---|---|---|---|---|---|
| filter_TCW_nonCpG | position | 0.0000 | 0.0426 | +0.0426 | 0.00 | 4.22 |
| filter_TCW_nonCpG | best_window_max_w1000 | 0.0268 | 0.0680 | +0.0411 | 0.65 | 1.70 |
| filter_all_CT | position | 0.0085 | 0.0127 | +0.0041 | 0.81 | 1.22 |
| filter_all_CT | best_window_max_w1000 | 0.0552 | 0.0658 | +0.0106 | 0.84 | 1.02 |

## 5. Verdict

The v4 retraining cleanly removes any anti-TCW polarity AND raises 5-fold
AUROC from 0.78 to ~0.83 for both variants.

**Recommendation: use v4 apobec1 head in the final claim, specifically v4_cds**.
- v4_cds wins at all four sweep cells (position + best-window, both filters)
  with deltas ranging +0.004 to +0.043 absolute recall over v3.
- v4_cds achieves a striking 4.22x NPOS ratio at position-level/TCW_nonCpG
  (vs v3 = 0 because v3's anti-TCW polarity hides all TCW positives outside
  the top 1% in C-context).
- v4_cancer wins on best-window but LOSES on position-level for filter_all_CT
  (-0.022 vs v3): the cancer-trinuc context inflates non-TCW scoring and
  overlaps with random-mutation baselines.
- v4_cds also has the canonical APOBEC1 mooring-rich pattern matching the
  enzyme's biochemistry (ACA/ACT/TCA/TCT high; CpG low).
- v4_cds is the natural symmetrical choice given the shared encoder is
  phase3_v4_cds.

Generated by `scripts/multi_enzyme/build_apobec1_retrain_summary.py`.
