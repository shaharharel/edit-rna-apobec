# PRE_REGISTRATION_PHASE1 — pcawg_tcw_panel Phase 1

**This document supersedes the prior `PRE_REGISTRATION.md`** which had a fake
timestamp and several errors. This file is committed to git as the primary
record of pre-registration; the **commit hash** (recorded at the top after
commit) is the verifiable timestamp.

> Authored: 2026-04-25 14:35 IDT, BEFORE any panel scoring or analysis run.
> Pipeline state at registration: RNA-FM CDS in progress on ai-gpu2 (~3% of
> chr1 done at chr1=4.5%); no scored chroms yet; no analyses started.
> Git commit: *<inserted at commit time>*

## Why this document supersedes PRE_REGISTRATION.md

The prior file had:
1. PRIMARY_FILTER mismatch — file declared `apobec-attributed (SBS2+SBS13≥0.5)`
   but the analysis script set `tcw_not_cpg`. Fixed in this version.
2. Fake timestamp — file claimed 23:35 local but birth time was 00:04, after
   the analysis scripts. Fixed by git-commit hash as proof.
3. Used Fisher 2×2 with malformed contingency table. Replaced with permutation
   null (described below).
4. Used SBS attribution at sample level when no Donor↔Sample map exists in the
   public data. Replaced with cancer-level aggregation (described below).

## Feature regime

**MFE-only**: all 7 struct_delta slots zeroed at training AND inference.
- Phase3 model: `experiments/multi_enzyme/outputs/phase3_mfe_only/phase3_mfe_only.pt`
- APOBEC1 head: `experiments/multi_enzyme/outputs/apobec1_head/apobec1_head_mfe_only.pt`

## Panel target

**CDS only**, 8,446,859 cache-aligned candidates from
`data/processed/gcp_panel/candidates_cache_aligned.parquet`. All have a 'C' at
the central index of a 201-nt hg19 window. Reference: hg19 (GRCh37).

---

## Analysis A — PCAWG WGS coding-panel enrichment (PRIMARY)

**Source data**:
- `data/raw/pcawg_open/final_consensus_passonly.snv_mnv_indel.icgc.public.maf.gz`
  (PCAWG WGS open MAF, passonly, GRCh37). Filter to **Variant_Type == 'SNP'**
  (NOT 'SNV' — PCAWG uses 'SNP'), C>T (or G>A on - strand) on coding strand.
- `data/raw/pcawg_open/SigProfilier_PCAWG_WGS_probabilities_SBS.csv`
  (per-sample × subtype SBS attributions, ~50 MB).

**Cancer set** (10 cancers, sufficient mutation count): Skin-Melanoma,
Liver-HCC, Eso-AdenoCa, Panc-AdenoCA, Prost-AdenoCA, Lymph-BNHL, Biliary-AdenoCA,
Kidney-RCC, Ovary-AdenoCA, Stomach-AdenoCA. PCAWG `Project_Code` matches SBS
`Cancer Type` directly.

**SBS attribution — cancer-level** (B1 fix):
For each (cancer, trinucleotide subtype) pair, compute the **mean** SBS2+SBS13
weight across all PCAWG samples in that cancer type for that subtype. A
mutation with subtype X in cancer Y is "APOBEC-attributed" if the cancer-level
mean SBS2+SBS13 weight at (Y, X) ≥ 0.1. This is a documented approximation due
to the absence of a working Donor↔Sample mapping for the public PCAWG MAF.

**Panel restriction** (per supervisor prompt):
Restrict mutations to those falling within the 8.45 M scored CDS positions.
Report the per-cancer fraction (`panel_coverage_stats.json`); this is "the
fraction of the cancer's mutations the panel even has scoring coverage for".

**Window construction**:
1 kb non-overlapping windows over CDS positions only (window contains ≥1 CDS
candidate). Per-window features: mean and max of each of 7 head scores
(binary, A3A, A3B, A3G, A3A_A3G, Neither, apobec1), CpG dinucleotide density,
TCW trinucleotide density (counting both strands; M4-fixed counter), modal
gene, training-mask flag, driver flag.

**Primary endpoint** (this is the ONE pre-registered comparison):

> Across the 10 PCAWG cancers, compute the **recall** of APOBEC-attributed C>T
> SNVs (cancer-level SBS2+SBS13 mean ≥ 0.1) contained within the **top 1%** of
> 1 kb CDS windows ranked by `score_binary_mean` (mean phase3_mfe_only binary
> head score per window). Compare against the recall obtained by ranking
> windows by **CpG dinucleotide density** (primary baseline).

**Statistical test (B3 fix — permutation null, NOT Fisher):**
For each cancer, generate 10,000 random permutations of the per-window
`score_binary_mean` labels (preserving the score distribution but breaking the
score↔window assignment); for each permutation, recompute mut_in_top under
the same top-1% selection. The one-sided p-value is the fraction of
permutations with mut_in_top ≥ observed (Laplace-smoothed: `(n+1)/(N+1)`).

**Pass criteria** (all four must hold for PRIMARY = PASS):
- (a) mean recall ratio (model / cpg_baseline) ≥ 1.5× across 10 cancers
      (after training-mask + driver-ablation)
- (b) BH-FDR-adjusted q < 0.05 in at least 6/10 cancers (BH correction across
      the 10 per-cancer permutation p-values; primary endpoint's BH family is
      these 10 tests)
- (c) signal survives driver-gene ablation: with all driver-gene windows
      removed (driver list = Bailey 2018 curated; M5 fix excludes TTN/MUC16/
      OBSCN/SYNE1 length confounders), recall ratio remains ≥ 1.3× on average
- (d) signal survives training-site ±1 kb mask: with all 1 kb windows ±1 kb
      from any v3 training site removed, recall ratio remains ≥ 1.3× on average

**Training-site mask**:
v3 split positions are loaded from `splits_multi_enzyme_v3_with_negatives.csv`
using `chr` + `start` columns directly (M1 fix; not `site_id` parser). Only
hg19-coordinate sites are used (M2 fix; 5,250 hg38 entries are documented and
dropped from the mask).

**Driver-gene ablation**:
Bailey-2018-style curated driver list (~95 genes, available in
`scripts/gcp_panel/analysis_A_pcawg_wgs.py:load_bailey_drivers`). A window is
"driver" if its modal gene is in the list. M5 fix: TTN, MUC16, OBSCN, SYNE1
are explicitly excluded (these are large-gene length confounders, not high-
confidence drivers).

**Secondary endpoints** (BH-corrected as a single family):
- All other heads × filters × percentiles × cancers
- Within-CpG-density-decile primary (QA #2): per-decile mean ratio reported;
  if any decile shows ratio < 1.2 while overall ≥ 1.5, this is a CpG-confound
  red flag noted in REPORT.md.

---

## Analysis B — TCGA-MC3 + PCAWG-coding combined enrichment (PRIMARY)

**Source data**:
- TCGA MC3: `data/raw/tcga/<cancer>_tcga_pan_can_atlas_2018_mutations.txt`
- cBioPortal PCAWG-coding: `data/raw/pcawg/by_cancer/<cancer>_pcawg_mutations.txt`
- 10 cancers: blca, brca, cesc, coadread, esca, hnsc, lihc, lusc, skcm, stad.

**Mutation filter**: TCW-non-CpG (clean APOBEC trinucleotide context). Both
sources combined; mutations tagged with `source` column.

**Primary endpoint**:
Same shape as Analysis A but mutation source = combined TCGA+PCAWG-coding,
filter = `tcw_not_cpg` (no SBS attribution available coding-only).

**Pass criteria**: identical (a)/(b)/(c)/(d) thresholds; permutation null;
BH across cancers.

---

## Multiple-testing correction across A and B primaries

Two pre-registered primary endpoints (A and B). Family across both: Bonferroni
α=0.05 → per-analysis α=0.025. We declare primary PASS at q < 0.025 for the
binding (b) criterion; (a)/(c)/(d) are pure point-estimates without
multiple-testing.

## Changes allowed post-registration

- Bug fixes in data loading / code (logged in REPORT.md "Changes after pre-reg")
- Additional exploratory analyses clearly labeled "post-hoc"
- NOT allowed: relaxing pass thresholds, switching the primary head, swapping
  the baseline, changing the cancer list.

## SHA / provenance (recorded at scoring/analysis time)

- phase3_mfe_only.pt SHA: *to be recorded by analysis script*
- apobec1_head_mfe_only.pt SHA: *to be recorded by analysis script*
- candidates_cache_aligned.parquet SHA: *to be recorded by analysis script*
- This file's git commit SHA: *recorded after first commit by tagging*

## Caveats carried forward

1. **MAF Variant_Type bug fix**: PCAWG uses 'SNP' not 'SNV' — confirmed and
   fixed in `analysis_A_pcawg_wgs.py:load_pcawg_maf`.
2. **SBS attribution is cancer-level, not sample-level** (B1 fallback). Real
   SBS2+SBS13 varies within-cancer between tumors; we approximate with the
   cancer-mean. Per-tumor noise is averaged out, which slightly weakens the
   signal but does not bias it.
3. **Vienna-version drift**: 0.6% of cached dot-brackets diverge from fresh-
   fold; cache values used canonically.
4. **Layout-swap fix already in phase3_mfe_only.pt**: prior layout swap
   verified absent in current model.
