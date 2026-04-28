# Sweep v3 — Fair Re-evaluation of the Panel-Construction Family

Re-runs the 21 panel constructions x 7 score heads x 4 mutation
filters under three QA-mandated remediations:

1. **Same-bases baselines.** CpG- and TCW-density baselines
   counted ONLY over the panel CDS-C positions in each window
   (using per-position trinucleotide context recomputed from
   hg19), NOT over the full window sequence. Aligns the NN's
   sum-over-CDS-C with what motif baselines see.
2. **n_pos-only baseline.** Each window also evaluated against
   ranking by `n_panel_positions_in_window` — pure gene-body
   density, no model, no motif.
3. **Real shuffle permutation null.** 10K random k-subset
   draws per (window-size, cancer, filter); p_perm = fraction
   of perms with mut_in_top >= observed.
4. **Bonferroni at alpha=0.05** across 1470
   tests (147 cells x 10 cancers): q < 3.40e-05.

**Stratum:** Default = filter_TCW_nonCpG. Also computed for
filter_all_TCW, filter_all_CT, filter_random_C.
**Cancers (10):** ['blca', 'brca', 'cesc', 'coadread', 'esca', 'hnsc', 'lihc', 'lusc', 'skcm', 'stad'].
**Bootstrap:** N_BOOT=10000 resamples across 10 cancers.

## Table 1 — Top 10 cells by ratio_vs_TCW (same-bases)

Ranked across all 4 filters; the strict TCW-nonCpG stratum is
the headline.

| rank | head | agg | win | filter | abs_recall (CI) | ratio_vs_TCW (CI) | ratio_vs_NPOS | bonf/10 |
|------|------|-----|-----|--------|-----------------|-------------------|---------------|---------|
| 1 | score_Neither | top3_mean | 0 | filter_all_CT | 3.81% [3.30, 4.32] | 5.192 [2.151, 9.171] | 3.680 | 0/10 |
| 2 | score_Neither | p95 | 0 | filter_all_CT | 3.81% [3.30, 4.32] | 5.192 [2.151, 9.171] | 3.680 | 0/10 |
| 3 | score_Neither | sum | 0 | filter_all_CT | 3.81% [3.30, 4.32] | 5.192 [2.151, 9.171] | 3.680 | 0/10 |
| 4 | score_Neither | mean | 0 | filter_all_CT | 3.81% [3.30, 4.32] | 5.192 [2.151, 9.171] | 3.680 | 0/10 |
| 5 | score_Neither | max | 0 | filter_all_CT | 3.81% [3.30, 4.32] | 5.192 [2.151, 9.171] | 3.680 | 0/10 |
| 6 | score_A3A_A3G | max | 0 | filter_all_CT | 3.17% [2.72, 3.65] | 4.544 [1.790, 8.066] | 3.051 | 0/10 |
| 7 | score_A3A_A3G | p95 | 0 | filter_all_CT | 3.17% [2.72, 3.65] | 4.544 [1.790, 8.066] | 3.051 | 0/10 |
| 8 | score_A3A_A3G | sum | 0 | filter_all_CT | 3.17% [2.72, 3.65] | 4.544 [1.790, 8.066] | 3.051 | 0/10 |
| 9 | score_A3A_A3G | mean | 0 | filter_all_CT | 3.17% [2.72, 3.65] | 4.544 [1.790, 8.066] | 3.051 | 0/10 |
| 10 | score_A3A_A3G | top3_mean | 0 | filter_all_CT | 3.17% [2.72, 3.65] | 4.544 [1.790, 8.066] | 3.051 | 0/10 |

## Table 2 — n_pos-only baseline: ratio_vs_TCW per (agg, win)

Diagnostic: how does ranking by **panel-position count alone**
(no model, no motif) compare to the TCW-density baseline? If
this ratio > 1, then much of the headline 'NN beats TCW' could
be explained by gene-body density alone.

Per-construction n_pos-only ratio_vs_TCW under filter_TCW_nonCpG:

| level | aggregator | window | n_pos_only_recall | TCW_baseline_recall | ratio_NPOS_vs_TCW | 95% CI |
|-------|------------|--------|-------------------|---------------------|-------------------|--------|
| position | (none) | 0 | 1.025% | 7.905% | 0.130 | [0.120, 0.140] |
| window | max | 1000 | 4.079% | 8.963% | 0.468 | [0.407, 0.528] |
| window | mean | 1000 | 4.079% | 8.963% | 0.468 | [0.407, 0.528] |
| window | sum | 1000 | 4.079% | 8.963% | 0.468 | [0.407, 0.528] |
| window | top3_mean | 1000 | 4.079% | 8.963% | 0.468 | [0.407, 0.528] |
| window | p95 | 1000 | 4.079% | 8.963% | 0.468 | [0.407, 0.528] |

(NPOS top-k and TCW top-k baselines do not depend on the score
head or aggregator, so values for `score_binary/sum` are
representative of all 7 heads x 5 aggregators at that ws.)

## Table 3 — Bonferroni-surviving cells

Per-cancer p_perm tested at q < 3.40e-05. A cell is
'surviving' if at least one cancer's p_perm clears Bonferroni;
'majority surviving' if >=6 of 10 cancers do.

- Cells with >=1 cancer surviving Bonferroni: **0 / 280**
- Cells with majority (>=6/10) surviving: **0 / 280**

### Top 15 cells by `n_cancers_bonf_signif`, tied by ratio_vs_TCW

| head | agg | win | filter | abs_recall | ratio_vs_TCW | ratio_vs_NPOS | bonf/10 |
|------|-----|-----|--------|------------|--------------|---------------|---------|
| score_Neither | max | 0 | filter_all_CT | 3.81% | 5.192 | 3.680 | 0/10 |
| score_Neither | mean | 0 | filter_all_CT | 3.81% | 5.192 | 3.680 | 0/10 |
| score_Neither | p95 | 0 | filter_all_CT | 3.81% | 5.192 | 3.680 | 0/10 |
| score_Neither | sum | 0 | filter_all_CT | 3.81% | 5.192 | 3.680 | 0/10 |
| score_Neither | top3_mean | 0 | filter_all_CT | 3.81% | 5.192 | 3.680 | 0/10 |
| score_A3A_A3G | max | 0 | filter_all_CT | 3.17% | 4.544 | 3.051 | 0/10 |
| score_A3A_A3G | mean | 0 | filter_all_CT | 3.17% | 4.544 | 3.051 | 0/10 |
| score_A3A_A3G | p95 | 0 | filter_all_CT | 3.17% | 4.544 | 3.051 | 0/10 |
| score_A3A_A3G | sum | 0 | filter_all_CT | 3.17% | 4.544 | 3.051 | 0/10 |
| score_A3A_A3G | top3_mean | 0 | filter_all_CT | 3.17% | 4.544 | 3.051 | 0/10 |
| score_A3G | max | 0 | filter_all_CT | 2.73% | 3.366 | 2.647 | 0/10 |
| score_A3G | mean | 0 | filter_all_CT | 2.73% | 3.366 | 2.647 | 0/10 |
| score_A3G | p95 | 0 | filter_all_CT | 2.73% | 3.366 | 2.647 | 0/10 |
| score_A3G | sum | 0 | filter_all_CT | 2.73% | 3.366 | 2.647 | 0/10 |
| score_A3G | top3_mean | 0 | filter_all_CT | 2.73% | 3.366 | 2.647 | 0/10 |

## Headline rewrite

**Original headline:** score_binary, sum, 1000 bp, 
filter_TCW_nonCpG = 1.31x ratio_vs_TCW (window-seq baseline).

**Fair re-eval, same construction:**
- ratio_vs_TCW (same-bases): **0.467** [0.412, 0.522]
- ratio_vs_NPOS (n_pos-only): **1.006** [0.953, 1.059]
- abs_recall: 4.09% [3.72, 4.47]
- Bonferroni-surviving cancers: 0/10

**Best per-head at same construction:** score_apobec1 = ratio_vs_TCW 0.508 [0.461, 0.558]
   ratio_vs_NPOS = 1.116, bonf 0/10

**Strongest defensible claim:** NO cell at filter_TCW_nonCpG
simultaneously (a) majority Bonferroni-surviving, 
(b) ratio_vs_TCW > 1, AND (c) ratio_vs_NPOS > 1. The 1.31x
headline does not survive the fair comparison.

## Verdict

**Q1: Does the 1.31x headline survive same-bases TCW?**  **NO**  (ratio_vs_TCW = 0.467 [0.412, 0.522])
**Q2: Does ANY construction beat n_pos alone (CI lo > 1)?**  
**YES** — best: score_apobec1 max w=0 ratio_vs_NPOS=1.295 [1.084, 1.528]

## Files

- `sweep_v3_fair.csv` — flat table, 1 row per (head, agg, ws, filter)
- `sweep_v3_fair_per_cancer.csv` — per-cancer drill-down
- `sweep_v3_fair.png` — ratio_vs_TCW per head, faceted by filter
- `SWEEP_V3_FAIR_RESULTS.md` — this report