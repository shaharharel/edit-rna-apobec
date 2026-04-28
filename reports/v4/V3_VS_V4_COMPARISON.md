# V3 vs V4 Comparison: Panel-Recall Sweep

**Question:** Did v4 (trinucleotide-matched negatives, anti-TCW
polarity removed in bias diagnostic) unlock a real RNA-to-DNA
transfer signal that v3 lacked?

- v3 sweep file: `sweep_v3_fair.csv` (280 cells)
- v4_cancer sweep file: `sweep_v4_cancer_fair.csv` (600 cells)
- v4_cds sweep file: `sweep_v4_cds_fair.csv` (600 cells)
- v3 levels available: ['position', 'win_1000'] window sizes: [0, 1000]
- v4 levels available: ['position', 'win_100', 'win_1000', 'win_250', 'win_500'] window sizes: [0, 100, 250, 500, 1000]
- v3 Bonferroni: q < 3.40e-05 (n_tests = 1470, 7 heads)
- v4 Bonferroni: q < 3.97e-05 (n_tests = 1260, 6 heads)

## 1. Headline: Did v4 unlock a real signal?

**Best score_binary cell at filter_TCW_nonCpG for each model:**

| model | construction | ratio_vs_TCW (CI) | ratio_vs_NPOS (CI) | abs_recall (CI) | bonf/10 |
|-------|--------------|-------------------|--------------------|-----------------|---------|
| v3 | sum, ws=1000, level=win_1000 | 0.467 [0.412, 0.522] | 1.006 [0.953, 1.059] | 4.095% [3.720, 4.472] | 0/10 |
| v4_cancer | sum, ws=1000, level=win_1000 | 0.544 [0.492, 0.595] | 1.181 [1.117, 1.243] | 4.797% [4.404, 5.194] | 10/10 |
| v4_cds | max, ws=0, level=position | 0.585 [0.528, 0.633] | 4.577 [4.022, 5.115] | 4.591% [4.229, 4.876] | 10/10 |

**Position-level score_binary at filter_TCW_nonCpG:**

| model | ratio_vs_TCW (CI) | ratio_vs_NPOS (CI) | abs_recall (CI) | bonf/10 |
|-------|-------------------|--------------------|-----------------|---------|
| v3 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000% [0.000, 0.000] | 0/10 |
| v4_cancer | 0.005 [0.002, 0.009] | 0.041 [0.013, 0.080] | 0.041% [0.012, 0.079] | 0/10 |
| v4_cds | 0.585 [0.528, 0.633] | 4.577 [4.022, 5.115] | 4.591% [4.229, 4.876] | 10/10 |

## 2. Side-by-side: three reference cells

Filter = filter_TCW_nonCpG throughout this table.

### 2a. score_binary, sum, win=1000

| model | ratio_vs_TCW (CI) | ratio_vs_NPOS (CI) | ratio_vs_CpG (CI) | abs_recall (CI) | bonf/10 |
|-------|-------------------|--------------------|-------------------|-----------------|---------|
| v3 | 0.467 [0.412, 0.522] | 1.006 [0.953, 1.059] | 2.457 [2.161, 2.813] | 4.095% [3.720, 4.472] | 0/10 |
| v4_cancer | 0.544 [0.492, 0.595] | 1.181 [1.117, 1.243] | 2.901 [2.504, 3.379] | 4.797% [4.404, 5.194] | 10/10 |
| v4_cds | 0.530 [0.479, 0.581] | 1.151 [1.081, 1.216] | 2.824 [2.437, 3.274] | 4.671% [4.290, 5.059] | 10/10 |

### 2b. score_binary, max, position-level

| model | ratio_vs_TCW (CI) | ratio_vs_NPOS (CI) | ratio_vs_CpG (CI) | abs_recall (CI) | bonf/10 |
|-------|-------------------|--------------------|-------------------|-----------------|---------|
| v3 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | n/a | 0.000% [0.000, 0.000] | 0/10 |
| v4_cancer | 0.005 [0.002, 0.009] | 0.041 [0.013, 0.080] | n/a | 0.041% [0.012, 0.079] | 0/10 |
| v4_cds | 0.585 [0.528, 0.633] | 4.577 [4.022, 5.115] | n/a | 4.591% [4.229, 4.876] | 10/10 |

### 2c. Top apobec1 cell (any construction at filter_TCW_nonCpG)

v3 head = `score_apobec1`; v4 head = `score_apobec1_v3`.

| model | construction | ratio_vs_TCW (CI) | ratio_vs_NPOS (CI) | ratio_vs_CpG (CI) | abs_recall (CI) | bonf/10 |
|-------|--------------|-------------------|--------------------|-------------------|-----------------|---------|
| v3 | sum, ws=1000, level=win_1000 | 0.508 [0.461, 0.558] | 1.116 [1.007, 1.277] | 2.731 [2.297, 3.248] | 4.483% [4.076, 4.870] | 0/10 |
| v4_cancer | sum, ws=1000, level=win_1000 | 0.463 [0.402, 0.525] | 0.992 [0.951, 1.042] | 2.415 [2.153, 2.708] | 4.035% [3.702, 4.352] | 10/10 |
| v4_cds | sum, ws=1000, level=win_1000 | 0.310 [0.253, 0.368] | 0.651 [0.595, 0.706] | 1.567 [1.415, 1.711] | 2.684% [2.298, 3.054] | 8/10 |

## 3. Per-head v4_cancer vs v4_cds at the v4_cancer winning construction

Winning construction (chosen by best `score_binary` `mean_ratio_vs_TCW` in v4_cancer at filter_TCW_nonCpG): `agg=sum, ws=1000, level=win_1000`.

| head | model | ratio_vs_TCW (CI) | ratio_vs_NPOS (CI) | abs_recall (CI) | bonf/10 |
|------|-------|-------------------|--------------------|-----------------|---------|
| score_binary | v4_cancer | 0.544 [0.492, 0.595] | 1.181 [1.117, 1.243] | 4.797% [4.404, 5.194] | 10/10 |
| score_binary | v4_cds | 0.530 [0.479, 0.581] | 1.151 [1.081, 1.216] | 4.671% [4.290, 5.059] | 10/10 |
| score_A3A | v4_cancer | 0.578 [0.526, 0.627] | 1.258 [1.184, 1.330] | 5.101% [4.713, 5.521] | 10/10 |
| score_A3A | v4_cds | 0.548 [0.495, 0.599] | 1.190 [1.118, 1.265] | 4.830% [4.430, 5.244] | 10/10 |
| score_A3B | v4_cancer | 0.530 [0.476, 0.584] | 1.148 [1.084, 1.209] | 4.666% [4.259, 5.069] | 10/10 |
| score_A3B | v4_cds | 0.518 [0.466, 0.571] | 1.125 [1.050, 1.195] | 4.569% [4.158, 4.977] | 9/10 |
| score_A3G | v4_cancer | 0.348 [0.296, 0.401] | 0.741 [0.705, 0.780] | 3.028% [2.708, 3.324] | 8/10 |
| score_A3G | v4_cds | 0.330 [0.277, 0.383] | 0.699 [0.653, 0.747] | 2.866% [2.525, 3.184] | 8/10 |
| score_A3A_A3G | v4_cancer | 0.393 [0.340, 0.453] | 0.844 [0.786, 0.897] | 3.449% [3.058, 3.819] | 8/10 |
| score_A3A_A3G | v4_cds | 0.378 [0.330, 0.433] | 0.813 [0.759, 0.865] | 3.317% [2.967, 3.640] | 8/10 |
| score_apobec1_v3 | v4_cancer | 0.463 [0.402, 0.525] | 0.992 [0.951, 1.042] | 4.035% [3.702, 4.352] | 10/10 |
| score_apobec1_v3 | v4_cds | 0.310 [0.253, 0.368] | 0.651 [0.595, 0.706] | 2.684% [2.298, 3.054] | 8/10 |

## 4. Bonferroni-surviving cells

- **v3** (q < 3.40e-05): 0/280 cells with >=1 cancer surviving, 0/280 cells with >=6 cancers (majority).
- **v4_cancer** (q < 3.97e-05): 310/600 cells with >=1 cancer surviving, 180/600 cells with >=6 cancers (majority).
- **v4_cds** (q < 3.97e-05): 359/600 cells with >=1 cancer surviving, 224/600 cells with >=6 cancers (majority).

### v3: no Bonferroni-surviving cells

### v4_cancer: top 8 Bonferroni-surviving cells

| head | agg | ws | level | filter | ratio_vs_TCW | ratio_vs_NPOS | abs_recall | bonf/10 |
|------|-----|----|-------|--------|--------------|---------------|------------|---------|
| score_apobec1_v3 | max | 0 | position | filter_all_CT | 3.913 [1.602, 6.922] | 2.773 [2.342, 3.248] | 2.876% [2.442, 3.352] | 10/10 |
| score_apobec1_v3 | mean | 0 | position | filter_all_CT | 3.913 [1.602, 6.922] | 2.773 [2.342, 3.248] | 2.876% [2.442, 3.352] | 10/10 |
| score_apobec1_v3 | p95 | 0 | position | filter_all_CT | 3.913 [1.602, 6.922] | 2.773 [2.342, 3.248] | 2.876% [2.442, 3.352] | 10/10 |
| score_apobec1_v3 | sum | 0 | position | filter_all_CT | 3.913 [1.602, 6.922] | 2.773 [2.342, 3.248] | 2.876% [2.442, 3.352] | 10/10 |
| score_apobec1_v3 | top3_mean | 0 | position | filter_all_CT | 3.913 [1.602, 6.922] | 2.773 [2.342, 3.248] | 2.876% [2.442, 3.352] | 10/10 |
| score_A3G | sum | 100 | win_100 | filter_random_C | 1.628 [1.450, 1.797] | 0.920 [0.876, 0.954] | 2.973% [2.813, 3.117] | 10/10 |
| score_apobec1_v3 | sum | 100 | win_100 | filter_random_C | 1.609 [1.462, 1.759] | 0.914 [0.864, 0.960] | 2.952% [2.776, 3.116] | 10/10 |
| score_binary | sum | 100 | win_100 | filter_random_C | 1.562 [1.363, 1.748] | 0.880 [0.823, 0.932] | 2.842% [2.657, 3.020] | 10/10 |

### v4_cds: top 8 Bonferroni-surviving cells

| head | agg | ws | level | filter | ratio_vs_TCW | ratio_vs_NPOS | abs_recall | bonf/10 |
|------|-----|----|-------|--------|--------------|---------------|------------|---------|
| score_A3A_A3G | max | 0 | position | filter_all_CT | 4.307 [1.945, 7.371] | 3.248 [2.781, 3.684] | 3.364% [2.923, 3.785] | 10/10 |
| score_A3A_A3G | mean | 0 | position | filter_all_CT | 4.307 [1.945, 7.371] | 3.248 [2.781, 3.684] | 3.364% [2.923, 3.785] | 10/10 |
| score_A3A_A3G | p95 | 0 | position | filter_all_CT | 4.307 [1.945, 7.371] | 3.248 [2.781, 3.684] | 3.364% [2.923, 3.785] | 10/10 |
| score_A3A_A3G | sum | 0 | position | filter_all_CT | 4.307 [1.945, 7.371] | 3.248 [2.781, 3.684] | 3.364% [2.923, 3.785] | 10/10 |
| score_A3A_A3G | top3_mean | 0 | position | filter_all_CT | 4.307 [1.945, 7.371] | 3.248 [2.781, 3.684] | 3.364% [2.923, 3.785] | 10/10 |
| score_A3A | max | 0 | position | filter_all_CT | 4.217 [1.976, 7.545] | 3.389 [2.628, 4.158] | 3.498% [2.719, 4.273] | 10/10 |
| score_A3A | mean | 0 | position | filter_all_CT | 4.217 [1.976, 7.545] | 3.389 [2.628, 4.158] | 3.498% [2.719, 4.273] | 10/10 |
| score_A3A | p95 | 0 | position | filter_all_CT | 4.217 [1.976, 7.545] | 3.389 [2.628, 4.158] | 3.498% [2.719, 4.273] | 10/10 |

## 5. Position-level diagnostic: top-1% trinucleotide breakdown

Strand-corrected trinucleotide context of the top-1% panel
positions ranked by `score_binary` (per model). Compare to
the panel's overall distribution. **Anti-TCW polarity**
(v3) means top-1% should be CpG-skewed and TCW-depleted
relative to the overall distribution; v4 should not be.

| trinuc bucket | overall panel | v3 top-1% | v4_cancer top-1% | v4_cds top-1% |
|---------------|---------------|-----------|------------------|---------------|
| TCW | 13.00% | 0.00% | 0.21% | 54.43% |
| TCG (CpG) | 2.30% | 0.01% | 0.00% | 29.44% |
| TCC | 6.78% | 0.00% | 0.00% | 6.56% |
| NCG (non-TC CpG) | 10.24% | 53.86% | 0.00% | 4.68% |
| other_C | 67.68% | 46.13% | 99.79% | 4.89% |
| non-C | 0.00% | 0.00% | 0.00% | 0.00% |

**TCW vs CpG ratios (top-1% / overall):**

| model | TCW enrichment | CpG enrichment | TCW polarity |
|-------|----------------|----------------|--------------|
| v3 | 0.00x | 4.30x | ANTI-TCW |
| v4_cancer | 0.02x | 0.00x | neutral |
| v4_cds | 4.19x | 2.72x | TCW-positive |

See `topx_trinuc_breakdown.png` for visualization.

## 6. Verdict

Two defensibility tiers (within filter_TCW_nonCpG):

- **Tier S (strong)**: ratio_vs_TCW CI lo > 1 AND ratio_vs_NPOS CI lo > 1
  AND >=6/10 cancers Bonferroni-surviving. Beats both same-bases TCW
  density and gene-body density.
- **Tier A (defensible)**: ratio_vs_NPOS CI lo > 1 AND >=6/10 cancers
  Bonferroni-surviving (no constraint on ratio_vs_TCW). Beats gene-body
  density alone. The TCW density baseline is structurally privileged
  in this filter because all surviving mutations are TCW; rather, the
  question is whether the model's positional ranking is more informative
  than just `n_panel_positions_in_window`.

### Tier S (strong: beats both TCW AND n_pos)

- **v3**: no Tier-S cell found.
- **v4_cancer**: no Tier-S cell found.
- **v4_cds**: no Tier-S cell found.

### Tier A (defensible: beats n_pos density, may lose to TCW)

- **v3**: no Tier-A cell found.
- **v4_cancer**: `score_binary, sum, ws=250, level=win_250` -- ratio_vs_NPOS = 1.841 [1.549, 2.183]; ratio_vs_TCW = 0.414 [0.358, 0.469]; abs_recall = 2.064% [1.886, 2.214]; 6/10 Bonf.
- **v4_cds**: `score_binary, max, ws=0, level=position` -- ratio_vs_NPOS = 4.577 [4.022, 5.115]; ratio_vs_TCW = 0.585 [0.528, 0.633]; abs_recall = 4.591% [4.229, 4.876]; 10/10 Bonf.

### Winning variant

**v4_cds** has the strongest defensible claim (Tier A).

- Construction: `score_binary, max, ws=0, level=position`
- Effect size (vs n_pos density): 4.577 [4.022, 5.115]
- ratio_vs_TCW: 0.585 [0.528, 0.633] (loses to TCW-density same-bases baseline -- see note above)
- abs_recall: 4.591% [4.229, 4.876]
- Bonferroni-surviving cancers: 10/10

**Claim:** the v4_cds model's
positional ranking is informative about cancer C>T mutation locations beyond
what gene-body density alone explains, with effect size 4.58x 
(95% CI [4.02, 5.12]).

### v3 -> v4 deltas (binary head, sum/win_1000/TCW_nonCpG)

| metric | v3 | v4_cancer | v4_cds | delta v4_cancer-v3 | delta v4_cds-v3 |
|--------|----|-----------|--------|--------------------|------------------|
| ratio_vs_TCW | 0.467 | 0.544 | 0.530 | +0.078 | +0.064 |
| ratio_vs_NPOS | 1.006 | 1.181 | 1.151 | +0.175 | +0.145 |
| abs_recall | 4.095% | 4.797% | 4.671% | +0.702pp | +0.576pp |
| bonf/10 | 0 | 10 | 10 | +10 | +10 |

### Position-level claim is now non-zero?

- v3 position-level binary abs_recall = 0.000% (0% literally -- anti-TCW polarity zeroed out the recall on TCW_nonCpG mutations)
- v4_cancer position-level binary abs_recall = 0.041% (non-zero but tiny; v4_cancer top-1% is overwhelmingly other_C, not TCW)
- v4_cds position-level binary abs_recall = 4.591% (non-zero AND substantial; ratio_vs_NPOS = 4.577 [4.022, 5.115], 10/10 Bonf)

## Files

- `sweep_v3_fair.csv` (v3 sweep, position + win_1000 only)
- `sweep_v4_cancer_fair.csv` (v4_cancer sweep, all 21 constructions)
- `sweep_v4_cds_fair.csv` (v4_cds sweep, all 21 constructions)
- `topx_trinuc_breakdown.png` and `.csv` (position-level trinuc diagnostic)
