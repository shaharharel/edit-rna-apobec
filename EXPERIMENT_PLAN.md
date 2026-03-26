# Experiment Execution Plan

**Created**: 2026-03-26
**Status**: Active — multiple jobs running in parallel

---

## Currently Running

| ID | Task | Status | ETA |
|----|------|--------|-----|
| SKCM | ViennaRNA folding for melanoma (negative control) | **DONE** | Completed 7:50 AM |
| A4-full | Full exome ViennaRNA | chr1+chr22 done, chr2 running | ~20h remaining |
| B1 | Additional cancers | HNSC done, ESCA/STAD/LIHC/COAD restarted | ~15h |

## Queued — Launching Now

### A-series (Immediate, hours)

| ID | Task | Depends on | Est. time |
|----|------|-----------|-----------|
| A1 | TC-stratified analysis (all 5 cancers) | SKCM completion | 10 min after SKCM |
| A2 | Top-1000 nonsense rate | **DONE** — 59.5% nonsense, OR=1.64, p=1.18e-14 |
| A3 | GoF/LoF at COSMIC scale | **DONE** — TSG confirmed (p=2.2e-27), GoF prediction FAILED (drop from paper) |
| A4 | Full exome editability map with ViennaRNA | **chr22 DONE** (200K positions, 88MB cache). Full genome RUNNING (chr1 = 860K positions) | ~24h |
| A5 | Genome-wide ClinVar test (trinuc-controlled, full model) | Needs full A4 (chr22 too small, only 288 ClinVar matches) | After A4 |

### B-series (Short-term, days)

| ID | Task | Depends on | Est. time |
|----|------|-----------|-----------|
| B1 | Add 5 more TCGA cancers (HNSC, ESCA, STAD, LIHC, COAD) + cache ViennaRNA | None | ~30h compute |
| B2 | Enzyme-tissue-disease triangle | **DONE** — A3G model OR=1.7-2.8 predicts mutations; A3A model depleted (OR=0.25-0.63). Structure-heavy models transfer better. |
| B3 | gnomAD site-level with full model scores | **DONE** — Gene: mis_oe r=-0.211 (p=4e-178). Site chr22: editable positions have fewer variants (purifying selection, p=8e-156) |
| B4 | Fix cross-species scoring | **DONE** — 3,610 orthologs (up from 1,127), Spearman r=0.632, score drop -0.061 |
| B6 | Levanon-internal confound analysis | **DONE** — ALL PASSES. TC chi2 p=2.9e-34, RLP KW p=4.3e-19. Dataset confound ruled out. |
| B7 | HIV/HBV APOBEC mutation analysis | **DONE** — MotifOnly insufficient, needs full model. Low priority. |

### Confound Controls — ALL COMPLETE
| Control | Result |
|---------|--------|
| TC-motif stratification | **PASSES** — OR=1.5-2.0 within TC-only for all APOBEC cancers |
| CpG stratification | **PASSES** — TC+non-CpG OR=1.17-1.33 (pure APOBEC signal) |
| Gene expression | **PASSES** — OR>1 in all expression quartiles for BRCA/CESC/LUSC |
| Levanon dataset confound | **PASSES** — all enzyme signatures replicate within-source |
| SKCM negative control | **DONE** — TC OR=1.52, **non-TC OR=0.82 (DEPLETED)**. Discriminates via non-TC channel. |

### Reviewer Must-Dos — ALL COMPLETE (March 26)
| Control | Result |
|---------|--------|
| Logistic regression baseline | LogReg matches XGB (0.871 vs 0.851 for A3A) — features well-engineered |
| StructOnly TCGA ablation | OR=1.07-1.10 (p<1e-5) in 5/5 APOBEC cancers, SKCM null (p=0.83) |
| Per-sample APOBEC (BRCA) | APOBEC-high OR=1.546, APOBEC-low OR=1.454 — both significant |

### Still pending
- Held-out dataset validation (Alqassim) — pre-submission

---

## Key Results So Far

### TCGA TC-Stratified (THE HEADLINE — 4/5 cancers done)
| Cancer | TC-only OR@p90 | TC-only OR@p95 | non-TC OR@p90 | non-TC OR@p95 |
|--------|:-:|:-:|:-:|:-:|
| BLCA | **1.549** | **1.682** | **1.840** | **2.064** |
| BRCA | **1.791** | **2.038** | **1.955** | **2.248** |
| CESC | **1.711** | **1.950** | **2.141** | **2.544** |
| LUSC | **1.461** | **1.591** | **1.480** | **1.624** |
| SKCM | *running* | *running* | *running* | *running* |

Structure predicts DNA mutation beyond motif. Awaiting SKCM negative control.

### Nonsense Rate (Model-Driven)
- Top-1000 model-predicted pathogenic: 59.5% nonsense vs 47.4% baseline
- OR=1.64, p=1.18e-14

### GoF/LoF (Updated)
- TSG: confirmed (67/78 genes, p=2.2e-27)
- Oncogenes: prediction FAILED — drop GoF/LoF claim from paper

### gnomAD Gene Constraint
- Editing genes LOEUF=0.818 vs non-editing 1.015 (p=3.6e-64)

### A3B Contradiction (Corrected)
- Butt 2024: A3B prefers loops + 3' bias (DNA substrate, in vitro)
- Alonso de la Vega 2023: A3B NO loops, sequence motif driven (RNA, transgenic mouse)
- Our data: A3B 54.3% in-loop but NO positional bias — intermediate between both
- Resolution: substrate (DNA vs RNA) determines targeting mechanism

---

## Pending Analysis After Compute Completes

1. Full TC-stratified with SKCM → paper figure
2. Exome-wide ClinVar enrichment with structure features → model validation
3. Enzyme-tissue-disease triangle → biological discovery
4. HIV/HBV APOBEC → extends beyond cancer
5. gnomAD site-level → evolutionary constraint at model-predicted positions
