# CLAUDE.md - Project Guidelines for AI Assistants

## Environment Setup

**Use the `quris` conda environment for all Python operations.**

---

## Project Overview

edit-rna-apobec is a framework for predicting APOBEC3A-mediated C-to-U RNA editing effects using the causal edit effect framework. It applies structured edit embeddings to model how cytidine-to-uridine changes affect RNA properties.

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

## Project Structure

```
src/
├── data/                  # Data loading and extraction
│   ├── editing_sites.py   # APOBEC C-to-U editing site data
│   ├── dataset.py         # RNAPairDataset, RNASequenceDataset
│   ├── sequence_utils.py  # RNA sequence utilities
│   ├── base_extractor.py  # Abstract base for data extractors
│   └── graph_cache.py     # GNN graph caching
├── embedding/             # RNA embedders
│   ├── base.py            # RNAEmbedder abstract base
│   ├── nucleotide.py      # Simple baseline embedder
│   ├── rnafm.py           # RNA-FM (640-dim)
│   ├── rnabert.py         # RNABERT (768-dim)
│   ├── utrlm.py           # UTR-LM (128-dim)
│   ├── codonbert.py       # CodonBERT
│   ├── rna_fm_encoder.py  # RNA-FM nn.Module for end-to-end training
│   ├── edit_embedder.py   # Difference-based edit embedder
│   ├── position_aware_edit_embedder.py
│   ├── structured_edit_embedder.py
│   ├── rnaplfold.py       # ViennaRNA structure
│   ├── eternafold.py      # EternaFold structure
│   └── structure_graph.py # PyG graph construction
├── models/                # Neural network architectures
│   ├── predictors/        # Edit effect predictors
│   ├── architectures/     # Model components
│   ├── dataset.py         # EditDataset, create_dataloaders
│   ├── trainer.py         # Trainer class
│   └── structure_gnn.py   # GNN for RNA structure
└── utils/                 # Splits, metrics, caching, logging

experiments/
└── apobec3a/              # APOBEC3A experiments

scripts/
└── apobec3a/              # APOBEC3A preprocessing scripts

data/                      # Raw and processed datasets
tests/                     # Unit tests
```

---

## Key Patterns

- **Embedders**: Inherit from base classes in `src/embedding/`
- **Experiments**: Config-driven, placed in `experiments/apobec3a/`
- **Preprocessing**: Scripts go in `scripts/apobec3a/`

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

### Step 2: Download reference genome

```bash
mkdir -p data/raw/genomes && cd data/raw/genomes
wget https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz && gunzip hg19.fa.gz
wget https://hgdownload.soe.ucsc.edu/goldenPath/hg19/database/refGene.txt.gz && gunzip refGene.txt.gz
python -c "from pyfaidx import Fasta; Fasta('hg19.fa')"
cd ../../..
```

### Step 3: Set up conda environment and regenerate data

```bash
conda activate quris
pip install -r requirements.txt
```

Then run the full pipeline. See **`data_generation.md`** for the complete step-by-step
guide with all commands, dependencies, and expected outputs.

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

## Recent Work Summary (Feb 2026)

### Data Integrity Fixes
Two critical bugs were found and fixed:
1. **Rate prediction included negatives**: `editing_rate_normalized.notna()` let 2,966 negative sites (rate=0) into regression, inflating Spearman from ~0.21 to 0.82. Fixed: filter to `is_edited == 1` only.
2. **Classification used wrong splits file**: Baselines ran on `splits_expanded.csv` (includes Baysal/non-A3A sites) instead of `splits_expanded_a3a.csv`. Fixed: new unified experiment uses A3A-filtered data.

Old broken results archived to `outputs/_archive_pre_fix/`.

### A3A Filtering Fix for Rate Experiments (Feb 2026)
All rate experiments previously used 636 unfiltered advisor sites.
Only 120 are APOBEC3A-only. Fixed by loading advisor from `splits_expanded_a3a.csv`.
Total rate sites: 4,462 (was 4,978). Cut 3 underperforming models
(DiffAttention_reg, GatedFusion_small, EditRNA+Feat_heavyreg) → 6 models remaining.

### Structure Cache Update
Computed ViennaRNA structure features for ~4,208 Baysal sites using `--incremental` mode (vienna conda env). Cache now covers all positive sites.

### Completed Experiments

**Rate Prediction (5-fold CV, positives only)** — `exp_rate_5fold.py`
- Output: `outputs/rate_5fold_positives/`
- 4,978 positive sites, 9 models
- Best: GB_HandFeatures (Spearman=0.407, R²=0.621)
- Loop geometry features dominate (~70% importance), `local_unpaired_fraction` alone ~40%
- Feature importance CSVs saved for GB models

**Classification (5-fold CV, A3A-filtered)** — `exp_classification_a3a_5fold.py`
- Output: `outputs/classification_a3a_5fold/`
- 8,153 A3A sites (5,187 pos, 2,966 neg), 13 models
- All NN models use FocalLoss(gamma=2.0, alpha=0.75)

**Cross-Dataset Rate Prediction** — `exp_cross_dataset_rate.py`
- Output: `outputs/cross_dataset_rate/`
- 3 models (GB_HandFeatures, EditRNA_rate, 4Way_heavyreg) × 3 train × 4 test datasets
- Key finding: cross-dataset generalization is poor (near-zero Spearman off-diagonal, negative R²)
- Reflects fundamentally different rate distributions across datasets
- Sharma (n=6) is test-only

**Rate Prediction with Z-scored Labels** — `exp_rate_5fold_zscore.py`
- Output: `outputs/rate_5fold_zscore/`
- Same as rate_5fold but target = per-dataset Z-score of log2(rate_normalized + 0.01)
- Each dataset's log2 rates are independently standardized (mean=0, std=1) before aggregation
- 6 models (cut DiffAttention_reg, GatedFusion_small, EditRNA+Feat_heavyreg)
- 4,462 A3A-confirmed sites (advisor loaded from `splits_expanded_a3a.csv`)
- Motivation: global log2 target lets models exploit dataset identity (Levanon rates ≪ Baysal/Alqassim)

### Key Bug Fixes in Code
- **`src/models/editrna_a3a.py`**: Added `batch["hand_features"]` support in forward() — allows hand feature injection via the GNN slot in GatedModalityFusion
- **`exp_classification_a3a_5fold.py`**: Fixed NaN in hand-augmented features with `np.nan_to_num`; added `_inject_hand_features` helper for EditRNA+Features model

### HTML Report
`experiments/apobec3a/generate_html_report.py` → `outputs/v3_report.html`
- Rate results with feature importance table (importance bars + Spearman ρ + descriptions)
- Classification results (pending experiment completion)
- Feature importance comparison (rate vs classification)

### Hand Features (40-dim)
- **Motif** (24-dim): trinucleotide context (m2, m1, p1, p2 one-hot) + dinucleotide (5p, 3p)
- **Loop Geometry** (9-dim): loop_size, dist_to_apex, relative_loop_position, is_unpaired, local_unpaired_fraction, left/right_stem_length, max_adjacent_stem_length, dist_to_junction
- **Structure Delta** (7-dim): delta_pairing_center, delta_accessibility_center, delta_entropy_center, delta_mfe, mean/std_delta_pairing_window, mean_delta_accessibility_window
- Feature-augmented models use 33-dim (motif 24 + loop 9, no struct delta) injected via d_gnn slot

### ClinVar Pathogenicity Replication (Feb 2026)

**Goal**: Replicate RNAsee 2024's reported pathogenic enrichment (22.7% vs 19.0%) and evaluate our GB model.

**Key findings**:
- RNAsee's enrichment cannot be exactly replicated due to ClinVar version difference (May 2022 ~101K C>U SNPs vs our 1.68M with 80% VUS) and keyword-match binning that inflates pathogenic counts by including "Conflicting interpretations of pathogenicity"
- GB_Full is the only predictor showing statistically significant pathogenic enrichment (OR=1.33, p<1e-40)
- Rules-based methods show pathogenic DEPLETION (OR=0.76), RF shows marginal enrichment (OR=1.08, p=0.026)
- GB-only predictions (175K variants RF misses at P>=0.5) also show strong pathogenic enrichment

| Method | OR (Path+LP/Ben+LB) | p-value | Direction |
|--------|---------------------|---------|-----------|
| CDS Rules (score>9) | 0.762 | 9.6e-14 | Depleted |
| RF (RNAsee, p>=0.5) | 1.080 | 0.026 | Marginal |
| **GB_Full (p>=0.5)** | **1.333** | **6.5e-41** | **Enriched** |

**Key files**:
- `experiments/apobec3a/replicate_rnasee_cds.py` — CDS-based replication with keyword binning validation + GB-only analysis
- `experiments/apobec3a/replot_clinvar.py` — OR comparison charts, enrichment-by-threshold, pathogenic ratio plots
- `experiments/apobec3a/outputs/clinvar_prediction/rnasee_cds_replication.json` — Full results
- `experiments/apobec3a/outputs/clinvar_prediction/clinvar_all_scores.csv` — 1.68M scored variants

### Prior-Calibrated ClinVar Analysis (Mar 2026)

**Problem**: Models trained on 1:3 positive:negative data with `scale_pos_weight=3.0` have effective π_model=0.50. Real editing site prevalence is much lower:
- **Tier 1 universe**: 5,187 positives / 277,495 total → π_real = 1.87% (1:53)
- **ClinVar-specific**: 413 known / 1,679,864 variants → π_clinvar = 0.025% (1:4,068)

**Bayesian recalibration**: P_cal = P_model × (π_real/π_model) / [P_model × (π_real/π_model) + (1-P_model) × ((1-π_real)/(1-π_model))]

Calibrated threshold (P_model where P_cal=0.5, Tier1 prior): **t_cal ≈ 0.946**

**Key finding**: Pathogenic enrichment persists (or strengthens) at calibrated thresholds, confirming the signal is not an artifact of the inflated training prior.

**Key files**:
- `experiments/apobec3a/exp_clinvar_calibrated.py` — Main calibration analysis
- `experiments/apobec3a/outputs/clinvar_calibrated/` — Results JSON + calibrated scores CSV + figures
- `experiments/apobec3a/replot_clinvar.py` — Now includes calibrated OR comparison plot
- `experiments/apobec3a/exp_disease_enrichment.py` — Now includes model-predicted gene enrichment (section 6)

### Embedding Visualization Overhaul (Feb 2026)

**Fixed**: `exp_embedding_trained.py` now uses `splits_expanded_a3a.csv` (was `splits_expanded.csv`) and renames advisor_c2t → "Advisor" (was "Levanon"), removes baysal_2016.

**New**: `exp_embedding_viz_comprehensive.py` — comprehensive embedding visualization with 12 coloring schemes on UMAP:
1. Label (pos/neg), 2. Dataset source, 3. Editing rate, 4. Rate Z-score,
5. Trinucleotide motif (TC/CC/AC/GC), 6. Genomic feature, 7. Delta MFE,
8. Pairing probability at center, 9. Loop size, 10. Upstream 5nt context,
11. Chromosome group, 12. Negative tier

Output: `outputs/embedding_viz_comprehensive/` — individual UMAP plots, summary grid, separation metrics JSON.

### Deprecated Models
- **DiffAttention / DiffAttention+Features**: Skip in all experiments. Runs a redundant 2-layer TransformerEncoder on top of RNA-FM tokens — 25 min/fold vs 1.5 min for EditRNA-A3A, while being the worst performer (Spearman=0.201 rate, lowest classification). The RNA-FM embeddings already capture sequence context.
