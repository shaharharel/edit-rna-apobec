# CLAUDE.md - Project Guidelines for AI Assistants

## Environment Setup

**Use the `quris` conda environment for all Python operations.**

---

## Project Overview

edit-rna-apobec is a framework for predicting APOBEC3A-mediated C-to-U RNA editing effects using the causal edit effect framework. It applies structured edit embeddings to model how cytidine-to-uridine changes affect RNA properties.

### Scientific Motivation

C-to-U RNA editing by APOBEC enzymes is widespread in human tissues but poorly understood. Out of millions of cytidines in the transcriptome, only ~5,000–10,000 are reproducibly edited. Why these sites and not others? Current predictors (e.g., RNAsee 2024) use simple rules-based scoring or random forests on hand-crafted features, achieving limited accuracy and showing no meaningful enrichment for disease-relevant variants.

**Key questions this project addresses:**
1. **Site prediction**: What determines whether a cytidine is edited by APOBEC3A? Can we build a classifier that outperforms rules-based methods?
2. **Rate prediction**: What determines *how much* a site is edited? Can we predict editing rates across tissues and conditions?
3. **Clinical relevance**: Are predicted editing sites enriched among pathogenic variants? Does C-to-U editing contribute to disease?

### Biological Context

APOBEC3A (apolipoprotein B mRNA editing enzyme, catalytic polypeptide-like 3A) performs C-to-U RNA editing in specific sequence contexts. This project:
- Predicts which cytidine sites will be edited
- Models editing rate differences across tissues and conditions
- Uses the edit effect framework to capture how local edits propagate through RNA structure

---

## Causal Edit Effect Framework

The central idea is to explicitly model how biological systems respond to **interventions**, rather than only learning correlations over static inputs.

We formulate prediction as an **edit effect problem**: given a baseline system and a defined edit, predict the resulting change in a property of interest.

### Edit Embeddings

Edit embeddings encode edits as **structured, context-aware interventions**, capturing how local changes propagate through sequence and structure. This decouples:
- The **representation of the underlying RNA** (the background system)
- The **representation of the intervention** applied to it

### Primary Validation: Edit Effect vs. Subtraction Baseline

| Approach | Method | Formula |
|----------|--------|---------|
| **Subtraction baseline** | Predict property independently for each sequence, then subtract | `F(seq_after) - F(seq_before) = Δproperty` |
| **Edit effect framework** | Learn directly from supervised delta signal with edit embeddings | `F(seq_before, edit) → Δproperty` |

---

## Code Organization Rules

### Strict Source Code Policy

**All code must be written within the project's source structure:**

```
src/           # Core library code (data loaders, embedders, models, utils)
experiments/   # Experiment runners
scripts/       # Data preprocessing and analysis scripts
tests/         # Unit tests
```

**Do NOT write code outside these directories.**

---


## Key Patterns

- **Embedders**: Inherit from base classes in `src/embedding/`
- **Experiments**: Config-driven, enzyme-namespaced: `experiments/apobec3a/`, `experiments/apobec3b/`, `experiments/apobec3g/`
- **Preprocessing**: Scripts in `scripts/apobec3a/` (A3A pipeline) and `scripts/multi_enzyme/` (cross-enzyme dataset)
- **Feature extraction**: `src/data/apobec_feature_extraction.py` provides enzyme-agnostic feature computation (motif, loop geometry, structure delta) used by all per-enzyme experiments
- **Multi-enzyme report**: `experiments/multi_enzyme/generate_html_report.py` aggregates results from all per-enzyme experiments into a unified comparison report

---

## Setting Up on a New Machine

After cloning the git repo, you need to regenerate all processed data from the raw source files.

### Step 1: Unpack raw data

Copy `editrna_raw_data.tar.gz` (8.7MB) to the project root and unpack:

```bash
tar -xzf editrna_raw_data.tar.gz
```

This extracts all raw datasets into `data/raw/`:
- `C2TFinalSites.DB.xlsx` (Advisor/Levanon 636 sites)
- `asaoka_2019_table_s1.xls`, `sharma_2015_supp_data.xls`, `alqassim_2021/` (published datasets)
- `baysal_2016/` (supplementary tables, ~4,200 sites)
- `levanon/tissue_editing_rates.csv` (54 GTEx tissue rates, pre-extracted from advisor T1 sheet)

### Step 2: Download reference genome (hg38)

```bash
mkdir -p data/raw/genomes && cd data/raw/genomes
wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz && gunzip hg38.fa.gz
wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/refGene.txt.gz && gunzip refGene.txt.gz
python -c "from pyfaidx import Fasta; Fasta('hg38.fa')"
cd ../../..
```

### Step 3: Set up conda environment, generate data, and run the APOBEC3A analysis

```bash
conda activate quris
pip install -r requirements.txt
```

**Phase 1 — Data pipeline** (stages 1-7, see `data_generation.md`):
```bash
# Run all data preprocessing steps (parsing, sequences, embeddings, structure cache)
# Full instructions with expected outputs in data_generation.md
```

**Phase 2 — APOBEC3A analysis** (all experiments → `v3_report.html`):
```bash
# Run all experiments in the correct dependency order
bash scripts/apobec3a/run_experiments.sh

# Or skip slow steps (ClinVar ~4.5h, neural models ~2h) for a quick first pass:
bash scripts/apobec3a/run_experiments.sh --skip-slow
```

The script enforces the correct execution order (see below) and outputs
`experiments/apobec3a/outputs/v3_report.html` as the final deliverable.

> **⚠️ Do not run experiments manually in arbitrary order.** Several experiments
> produce intermediate files that later experiments silently depend on. See
> "Experiment Execution Order" below and `data_generation.md` for full details.

---

## Canonical Datasets & Evaluation

### Rate Prediction
- **Data**: baysal from `splits_expanded.csv` + advisor/alqassim from `splits_expanded_a3a.csv`
- **Sites**: ~4,462 (120 advisor A3A-only + 128 alqassim + 6 sharma + 4,208 baysal)
- **IMPORTANT**: advisor_c2t has 636 sites total but only 120 are APOBEC3A-only.
  Always load advisor from `splits_expanded_a3a.csv`, never from `splits_expanded.csv`.
- **Target**: Per-dataset Z-score of log2(editing_rate_normalized + 0.01)
- **Evaluation**: 5-fold KFold CV, metrics: Spearman, Pearson, R-squared
- **Models (6)**: Mean Baseline, StructureOnly, GB_HandFeatures, GB_AllFeatures, EditRNA_rate, 4Way_heavyreg
- **NEVER include negatives (is_edited==0) in rate regression**
- **Rate scale issue**: Levanon/Advisor reports rates as percentages (0-100), all others use fractions (0-1). The `editing_rate_normalized` column divides Levanon by 100. Without per-dataset Z-scoring, dataset identity leaks into the target distribution.

### Binary Classification
- **Data**: `splits_expanded_a3a.csv` (APOBEC3A-only sites)
- **Sites**: ~8,153 (~5,187 positives, ~2,966 negatives)
- **No Baysal in classification**: Baysal sites are a subset of Asaoka 2019 (same A3A overexpression in HEK293T). They are deduplicated against Asaoka during the pipeline, so their sites already appear under `asaoka_2019` in `splits_expanded_a3a.csv`. Baysal's unique contribution is editing rates, used only for rate prediction.
- **Evaluation**: 5-fold KFold CV, metrics: AUROC, AUPRC, F1, Precision, Recall

### Structure Features
- **Cache**: `data/processed/embeddings/vienna_structure_cache.npz`
- Must cover ALL positive sites (no zero-vector defaults for real data)
- Run `generate_structure_cache.py --incremental` if sites are missing

---

## Experiment Execution Order

All experiments under `experiments/apobec3a/` are part of the APOBEC3A RNA editing analysis.
They feed into a single HTML report (`v3_report.html`). **Order matters.**

```
exp_loop_position_analysis          ← MUST run first; produces loop_position_per_site.csv
    ↓
exp_classification_a3a_5fold        ← uses loop_position_per_site.csv (GB hand features)
exp_rate_5fold_zscore               ← uses loop_position_per_site.csv
exp_rate_feature_importance         ← uses loop_position_per_site.csv
    ↓
exp_structure_analysis              ← independent (any order after loop_position)
exp_motif_analysis
exp_cross_dataset_full
exp_rnasee_comparison
exp_tc_motif_reanalysis
exp_dataset_deep_analysis
exp_embedding_viz_v2
exp_a3a_filtered
    ↓
exp_clinvar_prediction              ← ~4.5 hours; produces clinvar_all_scores.csv
    ↓
exp_clinvar_calibrated              ← depends on clinvar_all_scores.csv
replicate_rnasee_cds.py             ← depends on clinvar_all_scores.csv
    ↓
generate_html_report.py             ← final output: v3_report.html
```

Use `bash scripts/apobec3a/run_experiments.sh` to run in the correct order automatically.

### Why loop_position must run first

`exp_classification_a3a_5fold.py` calls `load_all_data()` which checks
`if LOOP_POS_CSV.exists()` and silently uses an empty DataFrame if the file is missing.
This causes all 9 loop geometry features (is_unpaired, relative_loop_position, etc.) to be
zero-vectors. **No error is raised.** The model trains on motif+struct only:
- GB_HandFeatures AUROC drops from 0.923 → 0.908
- `relative_loop_position` (the #1 classifier feature) shows 0.000 importance
- The bug is invisible until you check the feature importance CSV

---

## Current Status (Mar 2026)

### What's Done

| Area | Status | Key Result |
|------|--------|------------|
| **Data pipeline** | Complete | 7 raw datasets → unified splits, embeddings, structure cache. Reproducible from raw data via `data_generation.md` + `run_experiments.sh` |
| **Binary classification** | Complete | 13 models, 5-fold CV on 8,153 A3A sites. EditRNA+Features AUROC=0.935, GB_HandFeatures AUROC=0.923 |
| **Rate prediction** | Complete | 6 models, 5-fold CV on 4,462 positives. GB_HandFeatures best (Spearman=0.122). Loop geometry features dominate |
| **Cross-dataset generalization** | Complete | Poor off-diagonal Spearman — rate distributions differ fundamentally across datasets |
| **ClinVar pathogenicity** | Complete | GB_Full shows significant pathogenic enrichment (OR=1.33, p<1e-40). RNAsee's RF shows only marginal (OR=1.08). Rules-based is depleted |
| **Prior calibration** | Complete | Bayesian recalibration confirms enrichment is real, not a training-prior artifact |
| **Embedding visualization** | Complete | 12-scheme UMAP analysis. TC-motif and loop structure drive embedding separation |
| **HTML report** | Complete | Self-contained `v3_report.html` with all figures and results |

### Key Findings

1. **Structure > sequence for editing prediction**: `relative_loop_position` is the #1 classifier feature (0.213 importance), followed by motif features. Sites in unpaired loop regions are strongly favored for editing.
2. **GB outperforms neural models on rate**: Hand-crafted structure features in gradient boosting (Spearman=0.122) beat end-to-end neural approaches (EditRNA Spearman=0.137 but R²=-0.049 due to Sigmoid bug). The signal is low-dimensional and structure-driven.
3. **GB is the only model with real ClinVar signal**: At P≥0.5, GB_Full shows OR=1.279 (p<1e-138) for pathogenic enrichment — RNAsee's rules-based approach shows pathogenic *depletion* (OR=0.76).
4. **Cross-dataset rate prediction fails**: Models trained on one dataset do not generalize to others. This reflects genuinely different rate distributions (tissue-specific, enzyme-expression-dependent), not model failure.
5. **Prior calibration validates the signal**: After Bayesian recalibration from π_model=0.50 to π_real=0.019 (Tier 1), pathogenic enrichment persists at calibrated thresholds.

### What's Next

- **Per-enzyme experiments (A3B, A3G)**: Code created in `experiments/apobec3b/` and `experiments/apobec3g/`. Requires running: generate negatives, classification, ClinVar scoring. See `plan.md` for execution order.
- **Updated multi-enzyme report**: `experiments/multi_enzyme/generate_html_report.py` updated with classification, feature importance, ClinVar, and clinical interpretation sections. Regenerate after per-enzyme experiments complete.
- Edit effect framework validation (edit embeddings vs subtraction baseline)
- Deeper tissue-specific modeling using Levanon 54-tissue rates
- Paper writing

---

## Technical Notes & Pitfalls

### Experiment Execution Pitfalls (Silent Bugs)

1. **loop_position must run before classifiers** — see "Experiment Execution Order" above.
   If `loop_position_per_site.csv` is missing, all loop geometry features are silently zero.
   Symptom: `relative_loop_position` shows 0.000 importance, GB_HandFeatures AUROC ~0.908.

2. **ClinVar training: TC motif fraction must match between pos and neg**.
   Tier2/tier3 negatives are 99.9% TC; positives are 86.1% TC. Training with this imbalance
   inverts the TC signal (model learns TC → negative). Use hybrid negatives via
   `scripts/apobec3a/retrain_clinvar_and_rescore.py` which matches TC% between pos/neg.

3. **EditRNA_rate Sigmoid bug**: `APOBECMultiTaskHead.rate_head` ends with `nn.Sigmoid()`,
   bounding output to (0,1). Z-scored targets range ~[-3, +3], so MSE is miscalibrated.
   The model gets positive Spearman by preserving rank order but R²=-0.049 (worse than mean).
   Fix: remove `nn.Sigmoid()` from rate_head. Not applied yet — reference report uses the bug.

### Critical Data Rules
- **NEVER include negatives in rate regression**: `is_edited == 1` only. Previous bug with `editing_rate_normalized.notna()` let negatives (rate=0) in, inflating Spearman from 0.21 → 0.82.
- **Always use `splits_expanded_a3a.csv`** for classification and advisor sites. `splits_expanded.csv` includes non-A3A enzyme sites.
- **advisor_c2t has 636 sites but only 120 are A3A-only**. Always load advisor from `splits_expanded_a3a.csv`.
- **Rate scale issue**: Levanon/Advisor rates are percentages (0-100), others are fractions (0-1). Use per-dataset Z-scoring to prevent dataset identity leaking into targets.

### Hand Features (40-dim)
- **Motif** (24-dim): trinucleotide context (m2, m1, p1, p2 one-hot) + dinucleotide (5p, 3p)
- **Loop Geometry** (9-dim): loop_size, dist_to_apex, relative_loop_position, is_unpaired, local_unpaired_fraction, left/right_stem_length, max_adjacent_stem_length, dist_to_junction
- **Structure Delta** (7-dim): delta_pairing_center, delta_accessibility_center, delta_entropy_center, delta_mfe, mean/std_delta_pairing_window, mean_delta_accessibility_window
- Feature-augmented models use 33-dim (motif 24 + loop 9, no struct delta) injected via d_gnn slot

### ClinVar Key Files
- `experiments/apobec3a/outputs/clinvar_prediction/clinvar_all_scores.csv` — 1.68M scored variants
- `experiments/apobec3a/exp_clinvar_calibrated.py` — Bayesian prior calibration (π_model=0.50 → π_real=0.019)
- `experiments/apobec3a/replicate_rnasee_cds.py` — RNAsee replication + GB comparison

### Deprecated Models
- **DiffAttention / DiffAttention+Features**: Skip in all experiments. 25 min/fold vs 1.5 min for EditRNA-A3A, worst performer. RNA-FM embeddings already capture sequence context.
