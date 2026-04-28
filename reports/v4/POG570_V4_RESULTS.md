# POG570 v4 validation — v4_cds binary head

Generated: 2026-04-28 18:55

- Panel: `experiments/multi_enzyme/outputs/pcawg_tcw_panel/v4_outputs/panel_scores_v4_cds_apobec1retrained.parquet`
- Panel positions (n_units): 8,446,859
- Head: `score_binary`
- POG570 source: `/Users/shaharharel/Documents/github/edit-rna-apobec/data/raw/pog570/POG570_small_mutations.txt.gz`
- Random seed: 20260427

## POG570 mutation counts (after filtering)

- Raw POG570 C>T/G>A SNVs (10 cohorts mapping to PCAWG cancers): 2,631,946
- In-panel (CDS-C, v4_cds): 15,620 (0.59%)
- In-panel TCW_nonCpG: 2,854
- In-panel all_CT: 15,620

## Headline numbers (position-level binary head)

| filter | top% | NN recall (CI) | ratio_vs_TCW (CI) | ratio_vs_NPOS (CI) | ratio_vs_random (CI) | bonf_sig/n |
|---|---:|---:|---:|---:|---:|---:|
| filter_TCW_nonCpG | 0.01 | 0.039 (0.027–0.049) | 0.53× (0.45–0.62) | 4.22× (3.27–5.08) | 3.93× (2.74–4.91) | 4/9 |
| filter_TCW_nonCpG | 0.05 | 0.145 (0.102–0.178) | 0.40× (0.33–0.46) | 4.18× (2.62–5.63) | 2.90× (2.04–3.54) | 6/9 |
| filter_TCW_nonCpG | 0.10 | 0.230 (0.162–0.282) | 0.30× (0.21–0.36) | 3.23× (2.19–4.02) | 2.30× (1.61–2.82) | 6/9 |
| filter_all_CT | 0.01 | 0.031 (0.019–0.049) | 2.20× (1.36–3.40) | 2.72× (1.63–4.38) | 3.12× (1.85–4.83) | 4/10 |
| filter_all_CT | 0.05 | 0.112 (0.098–0.128) | 1.77× (1.19–2.58) | 2.30× (1.76–2.87) | 2.24× (1.97–2.56) | 8/10 |
| filter_all_CT | 0.10 | 0.221 (0.170–0.292) | 2.04× (1.13–3.23) | 2.11× (1.70–2.61) | 2.21× (1.70–2.90) | 7/10 |

## PCAWG vs POG570 (10-cancer aggregate, top-1%, position-level)

| metric | PCAWG (v4_cds) | POG570 (v4_cds) | replicates? |
|---|---:|---:|:---:|
| ratio_vs_TCW (all_CT, top-1%) | 3.56× | 2.20× (1.36–3.40) | yes |
| ratio_vs_NPOS (TCW_nonCpG, top-1%) | 4.58× | 4.22× (3.27–5.08) | yes |
| abs recall (TCW_nonCpG, top-1%) | 0.0459 | 0.0393 (0.0273–0.0492) | yes |

## POG570 per-cohort breakdown (top 10 cohorts by mutation count)

| analysis_cohort | mapped_cancer | n_total_in_panel | n_TCW_nonCpG_in_panel |
|---|---|---:|---:|
| SKCM | skcm | 4,772 | 869 |
| COLO | coadread | 3,637 | 472 |
| BRCA | brca | 3,018 | 849 |
| LUNG | lusc | 1,927 | 334 |
| HNSC | hnsc | 970 | 153 |
| ESCA | esca | 846 | 30 |
| CERV | cesc | 216 | 109 |
| STAD | stad | 205 | 37 |
| BLCA | blca | 19 | 1 |
| HCC | lihc | 10 | 0 |

## Per-cancer ratios at top-1%, TCW_nonCpG (binary head)

| cancer | n_mut | NN_recall | TCW_recall | NPOS_recall | ratio_TCW | ratio_NPOS | p_perm |
|---|---:|---:|---:|---:|---:|---:|---:|
| skcm | 873 | 0.0596 | 0.0733 | 0.0115 | 0.81 | 5.20 | 0.0001 |
| cesc | 109 | 0.0459 | 0.0826 | 0.0092 | 0.56 | 5.00 | 0.0054 |
| hnsc | 153 | 0.0327 | 0.0719 | 0.0065 | 0.45 | 5.00 | 0.0204 |
| coadread | 476 | 0.0399 | 0.0756 | 0.0126 | 0.53 | 3.17 | 0.0001 |
| brca | 854 | 0.0351 | 0.0679 | 0.0129 | 0.52 | 2.73 | 0.0001 |
| blca | 1 | 0.0000 | 0.0000 | 0.0000 | nan | nan | 1.0000 |
| esca | 30 | 0.0333 | 0.1000 | 0.0000 | 0.33 | nan | 0.2614 |
| lusc | 337 | 0.0534 | 0.1039 | 0.0000 | 0.51 | nan | 0.0001 |
| stad | 37 | 0.0541 | 0.1081 | 0.0000 | 0.50 | nan | 0.0515 |

## Verdict

v4_cds binary head on POG570 (independent cohort):

- ratio_vs_NPOS at top-1% (TCW_nonCpG) = 4.22× (95% CI 3.27–5.08); CI lower bound > 1.0
- ratio_vs_TCW at top-1% (all_CT) = 2.20× (95% CI 1.36–3.40); CI lower bound > 1.0

Verdict: **REPLICATES**

## Notes on baseline construction (v4 vs v3)

- v3 POG570 was confounded by a baseline mismatch: PCAWG analysis
  used `seq.count('CG')` over the literal hg19 window sequence,
  whereas POG570 v1 used `sum(is_cpg)` over panel positions.
  Different windows ranked highest by these two definitions.
- v4 uses **same-bases baselines exclusively**: TCW-density and
  n_pos counted *only over CDS-C panel positions*, matching the
  v4 PCAWG fair sweep so PCAWG and POG570 are on the same footing.
- Position-level (not 250 bp window) is the operational unit, 
  removing the gene-density artefact (windows that span large
  CDS regions had inflated n_pos under any window-aggregator).
