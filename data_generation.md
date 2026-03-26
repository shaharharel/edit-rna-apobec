# Data Generation Pipeline

Complete guide to regenerating all processed data from raw source files.

## Overview

The pipeline has two phases: **data generation** (stages 1-7) and **experiments**.
Total processing time: ~2-4 hours for data, ~3-5 hours for experiments (excluding ClinVar).

**Genome build: GRCh38/hg38** — the entire pipeline uses hg38 coordinates throughout.

```
Phase 1: Data Generation
  Raw source files (Excel, XLS)
      ↓ Stage 1: Parse → per-dataset CSVs
      ↓ Stage 2: Extract labels + negatives
      ↓ Stage 3: Build unified dataset
      ↓ Stage 4: Extract sequences (requires hg38 genome)
      ↓ Stage 5: Generate tiered negatives (requires hg38 + RNAfold)
      ↓ Stage 6: Expand dataset + generate embeddings (requires GPU)
      ↓ Stage 6b: Filter to APOBEC3A-Only Sites
      ↓ Stage 6c: Download ClinVar
      ↓ Stage 7: Generate structure cache (requires ViennaRNA)

Phase 2: Experiments (ORDER MATTERS — see below)
  exp_loop_position_analysis        ← MUST run first (produces loop_position_per_site.csv)
      ↓
  exp_classification_a3a_5fold      ← depends on loop_position_per_site.csv
  exp_rate_5fold_zscore             ← depends on loop_position_per_site.csv
  exp_rate_feature_importance       ← depends on loop_position_per_site.csv
      ↓
  [other analysis experiments]      ← independent, any order
      ↓
  exp_clinvar_prediction            ← ~4.5 hours, depends on clinvar_c2u_variants.csv
      ↓
  exp_clinvar_calibrated + rnasee   ← depend on clinvar_prediction outputs
      ↓
  generate_html_report              ← run last
```

> **⚠️ Critical**: Always run `exp_loop_position_analysis` before `exp_classification_a3a_5fold`.
> If `loop_position_per_site.csv` is missing when the classifier runs, all loop geometry features
> (is_unpaired, relative_loop_position, etc.) will be silent zero-vectors. The model trains without
> these features but does not warn you. This caused GB_HandFeatures AUROC to drop from 0.923 to 0.908
> and `relative_loop_position` (the #1 feature) to show 0.0 importance.

---

## Prerequisites

### Software

| Tool | Environment | Purpose |
|------|-------------|---------|
| Python 3.11+ | `quris` conda env | Main processing |
| ViennaRNA RNAfold | `/opt/miniconda3/envs/vienna/bin/RNAfold` | RNA structure prediction |
| ViennaRNA Python (`RNA`) | `vienna` conda env | Structure feature cache |
| RNA-FM | Installed in `quris` env | Sequence embeddings (GPU recommended) |
| pyfaidx | Installed in `quris` env | Fast genome FASTA access |

### External Downloads

**hg38 Reference Genome** (~900MB compressed, ~3GB uncompressed):
```bash
mkdir -p data/raw/genomes
cd data/raw/genomes
wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz
gunzip hg38.fa.gz
# Index for pyfaidx:
python -c "from pyfaidx import Fasta; Fasta('hg38.fa')"
```

**RefSeq Gene Annotations** (~23MB):
```bash
cd data/raw/genomes
wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/refGene.txt.gz
gunzip refGene.txt.gz
```

---

## Raw Source Files (provided in `editrna_raw_data.tar.gz`)

These files cannot be generated — they must be obtained from the original sources.

| File | Size | Source |
|------|------|--------|
| `data/raw/C2TFinalSites.DB.xlsx` | 6.1MB | Advisor's curated database (636 C-to-U editing sites) |
| `data/raw/published/asaoka_2019_table_s1.xls` | 1.0MB | Asaoka et al. 2019, DOI: 10.3390/ijms20225621 |
| `data/raw/published/sharma_2015_supp_data.xls` | 3.2MB | Sharma et al. 2015, DOI: 10.1038/ncomms7881 |
| `data/raw/published/alqassim_2021/alqassim_2021_supp_data1.xlsx` | 96KB | Alqassim et al. 2021, DOI: 10.1038/s42003-020-01620-x |
| `data/raw/published/alqassim_2021/alqassim_2021_supp_data2.xlsx` | 80KB | Alqassim et al. 2021 (gene expression) |
| `data/raw/published/baysal_2016/krnb-14-05-1184387-s001/` | 2.7MB | Baysal et al. 2016, DOI: 10.1080/15476286.2016.1184387 |
| `data/raw/published/levanon/tissue_editing_rates.csv` | 640KB | Levanon/Advisor tissue editing rates (pre-extracted T1 sheet, 54 GTEx tissues) |

Unpack with:
```bash
tar -xzf editrna_raw_data.tar.gz
```

---

## Pipeline Steps

All commands assume you are in the project root and using the `quris` conda environment
unless noted otherwise.

```bash
conda activate quris
```

### Stage 1: Parse Raw Datasets

Parse each source file into standardized CSVs. All coordinates are stored in **hg38**.

```bash
# 1a. Parse advisor Excel (636 positive sites + supplementary tables)
python scripts/apobec3a/parse_advisor_excel.py --input data/raw/C2TFinalSites.DB.xlsx
# Output: data/processed/advisor/*.csv (multiple files)
#         data/processed/advisor/unified_editing_sites.csv

# 1b. Parse published datasets
python scripts/apobec3a/parse_asaoka_2019.py
# Output: data/processed/published/asaoka_2019_editing_sites.csv (5,208 sites)

python scripts/apobec3a/parse_sharma_2015.py
# Output: data/processed/published/sharma_2015_editing_sites.csv (278 sites)

python scripts/apobec3a/parse_alqassim_2021.py
# Output: data/processed/published/alqassim_2021_editing_sites.csv (209 sites)

# 1e. Parse Baysal 2016 (native hg38 coordinates — no liftover needed)
python scripts/apobec3a/parse_baysal_2016.py \
    --input "data/raw/published/baysal_2016/krnb-14-05-1184387-s001/Supplemental Table 5.xls"
# Output: data/processed/published/baysal_2016_editing_sites.csv (~4,373 sites)
```

### Stage 2: Extract Labels and Negative Controls

```bash
# 2a. Extract multi-task labels from advisor data
python scripts/apobec3a/extract_labels.py --input data/raw/C2TFinalSites.DB.xlsx
# Output: data/processed/editing_sites_labels.csv (636 sites, 30 label columns)
#         data/processed/splits.csv (train/val/test by gene stratification)

# 2b. Extract negative controls from non-AG mismatch sites
python scripts/apobec3a/extract_negative_controls.py
# Output: data/processed/advisor/negative_controls_ct.csv
#         data/processed/advisor/negative_controls_filtered.csv (exonic only, ~325 sites)
```

### Stage 3: Build Unified Dataset

Combine all datasets into a single table with standardized hg38 columns.

```bash
python scripts/apobec3a/build_unified_dataset.py
# Output: data/processed/all_datasets_combined.csv (all positive sites, ~10,704 entries)
```

### Stage 4: Extract Sequences and Structures

Extract 201nt RNA sequences (±100 flanking around edit site) from **hg38** and
predict secondary structures with RNAfold.

```bash
python scripts/apobec3a/extract_sequences_and_structures.py \
    --genome data/raw/genomes/hg38.fa \
    --negatives
# Output: data/processed/sequences_and_structures.csv
# Time: ~10-20 minutes (depends on RNAfold speed)
```

### Stage 5: Generate Tiered Negatives

Generate hard negative controls at three difficulty tiers from RefSeq exonic regions
of genes that contain positive editing sites.

```bash
python scripts/apobec3a/generate_tiered_negatives.py
# Output: data/processed/negatives_tier1.csv (~272K all exonic C sites)
#         data/processed/negatives_tier2.csv (~132K TC-motif C sites)
#         data/processed/negatives_tier3.csv (~75K TC in stem-loops)
# Time: ~30-60 minutes (Tier 3 requires RNAfold on each site)
```

Requires: `hg38.fa`, `refGene.txt`, ViennaRNA RNAfold.

### Stage 6: Expand Dataset and Generate Embeddings

Subsample tiered negatives, merge with positives, extract sequences for new sites,
generate RNA-FM embeddings, and create train/val/test splits.

```bash
python scripts/apobec3a/expand_dataset.py \
    --max-tier2-neg 2000 \
    --max-tier3-neg 1000 \
    --batch-size 16
# Output: data/processed/site_sequences.json (all 201nt sequences)
#         data/processed/embeddings/rnafm_pooled.pt (640-dim pooled embeddings)
#         data/processed/embeddings/rnafm_tokens.pt (per-token embeddings)
#         data/processed/embeddings/rnafm_pooled_edited.pt (C->U edited)
#         data/processed/embeddings/rnafm_tokens_edited.pt (edited tokens)
#         data/processed/embeddings/rnafm_site_ids.json
#         data/processed/splits_expanded.csv (full dataset splits)
#         data/processed/splits_levanon_expanded.csv (Levanon-only splits)
# Time: ~30 minutes with GPU, ~2+ hours CPU-only
```

### Stage 6b: Filter to APOBEC3A-Only Sites

Create the A3A-filtered dataset used by all modern experiments. Removes non-A3A
advisor sites (keeps only 120 of 636) and baysal_2016 (subset of asaoka_2019).

```bash
python scripts/apobec3a/filter_a3a_splits.py
# Output: data/processed/splits_expanded_a3a.csv (~8,153 sites: 5,187 pos + 2,966 neg)
```

### Stage 6c: Download and Process ClinVar

Download ClinVar variants and extract C>T (potential C-to-U editing) SNPs.

```bash
python scripts/apobec3a/download_clinvar.py
# Output: data/raw/clinvar/clinvar_grch38.vcf.gz (~189MB download)
#         data/processed/clinvar_c2u_variants.csv (~1.68M variants)
```

### Stage 7: Generate Structure Cache

Compute ViennaRNA 2D structure features (pairing probability, accessibility, entropy)
for all sites. **Must be run in the `vienna` conda environment.**

```bash
/opt/miniconda3/envs/vienna/bin/python scripts/apobec3a/generate_structure_cache.py
# Output: data/processed/embeddings/vienna_structure_cache.npz
# Time: ~10-15 minutes
```

---

## Quick Reference: Full Pipeline

### Phase 1: Data Generation

```bash
conda activate quris

# Unpack raw data
tar -xzf editrna_raw_data.tar.gz

# Stage 1: Parse
python scripts/apobec3a/parse_advisor_excel.py --input data/raw/C2TFinalSites.DB.xlsx
python scripts/apobec3a/parse_asaoka_2019.py
python scripts/apobec3a/parse_sharma_2015.py
python scripts/apobec3a/parse_alqassim_2021.py
python scripts/apobec3a/parse_baysal_2016.py --input "data/raw/published/baysal_2016/krnb-14-05-1184387-s001/Supplemental Table 5.xls"

# Stage 2: Labels + negatives
python scripts/apobec3a/extract_labels.py --input data/raw/C2TFinalSites.DB.xlsx
python scripts/apobec3a/extract_negative_controls.py

# Stage 3: Unify
python scripts/apobec3a/build_unified_dataset.py

# Stage 4: Sequences (requires hg38.fa)
python scripts/apobec3a/extract_sequences_and_structures.py --genome data/raw/genomes/hg38.fa --negatives

# Stage 5: Tiered negatives (requires hg38.fa + RNAfold)
python scripts/apobec3a/generate_tiered_negatives.py

# Stage 6: Expand + embed (requires GPU for speed)
python scripts/apobec3a/expand_dataset.py --max-tier2-neg 2000 --max-tier3-neg 1000

# Stage 6b: Filter to A3A-only sites
python scripts/apobec3a/filter_a3a_splits.py

# Stage 6c: Download ClinVar
python scripts/apobec3a/download_clinvar.py

# Stage 7: Structure cache (vienna env)
/opt/miniconda3/envs/vienna/bin/python scripts/apobec3a/generate_structure_cache.py
```

### Phase 2: Experiments

Use the provided script — it enforces the correct order and skips already-completed steps:

```bash
# Run all experiments (including ClinVar ~4.5h)
bash scripts/apobec3a/run_experiments.sh

# Skip slow steps (ClinVar + neural models) for a quick GB-only run
bash scripts/apobec3a/run_experiments.sh --skip-slow

# Only GB/structure models (<10 min)
bash scripts/apobec3a/run_experiments.sh --only-gb
```

Or run manually in this exact order:

```bash
# 1. FIRST: loop position (produces loop_position_per_site.csv needed by classifiers)
python experiments/apobec3a/exp_loop_position_analysis.py

# 2. Core models (require loop_position_per_site.csv)
python experiments/apobec3a/exp_classification_a3a_5fold.py
python experiments/apobec3a/exp_rate_5fold_zscore.py
python experiments/apobec3a/exp_rate_feature_importance.py

# 3. Analysis experiments (any order)
python experiments/apobec3a/exp_structure_analysis.py
python experiments/apobec3a/exp_motif_analysis.py
python experiments/apobec3a/exp_cross_dataset_full.py
python experiments/apobec3a/exp_rnasee_comparison.py
python experiments/apobec3a/exp_tc_motif_reanalysis.py
python experiments/apobec3a/exp_dataset_deep_analysis.py
python experiments/apobec3a/exp_embedding_viz_v2.py
python experiments/apobec3a/exp_a3a_filtered.py

# 4. ClinVar (~4.5h, 12 workers)
python experiments/apobec3a/exp_clinvar_prediction.py
python experiments/apobec3a/exp_clinvar_calibrated.py
python experiments/apobec3a/replicate_rnasee_cds.py

# 5. Final report
python experiments/apobec3a/generate_html_report.py
```

---

## Output File Summary

After running the full pipeline, the `data/processed/` directory should contain:

```
data/processed/
├── advisor/                          # Parsed from C2TFinalSites.DB.xlsx
│   ├── unified_editing_sites.csv
│   ├── negative_controls_ct.csv
│   └── negative_controls_filtered.csv
├── published/                        # Parsed from published papers
│   ├── asaoka_2019_editing_sites.csv
│   ├── sharma_2015_editing_sites.csv
│   ├── alqassim_2021_editing_sites.csv
│   └── baysal_2016_editing_sites.csv
├── editing_sites_labels.csv          # 636 sites with multi-task labels
├── splits.csv                        # Base train/val/test splits
├── all_datasets_combined.csv         # All positives unified (hg38, ~10,704 entries)
├── sequences_and_structures.csv      # 201nt sequences + RNAfold structures
├── negatives_tier1.csv               # ~272K all exonic C
├── negatives_tier2.csv               # ~132K TC-motif
├── negatives_tier3.csv               # ~75K TC in stem-loops
├── site_sequences.json               # All sequences by site_id
├── splits_expanded.csv               # Full dataset splits (all datasets)
├── splits_expanded_a3a.csv           # A3A-filtered (8,153 sites, canonical)
├── splits_levanon_expanded.csv       # Levanon + tiered negatives splits
├── clinvar_c2u_variants.csv          # ClinVar C>T variants (1.68M)
└── embeddings/
    ├── rnafm_pooled.pt               # 640-dim pooled RNA-FM embeddings
    ├── rnafm_tokens.pt               # Per-token RNA-FM embeddings (~5GB)
    ├── rnafm_pooled_edited.pt        # Edited sequence pooled embeddings
    ├── rnafm_tokens_edited.pt        # Edited sequence token embeddings (~5GB)
    ├── rnafm_site_ids.json           # Site IDs in embedding cache
    └── vienna_structure_cache.npz    # ViennaRNA 2D structure features
```

---

## Troubleshooting

**`pyfaidx` index error**: If hg38.fa is not indexed, run:
```bash
python -c "from pyfaidx import Fasta; Fasta('data/raw/genomes/hg38.fa')"
```

**RNAfold not found**: Ensure ViennaRNA is installed and the path
`/opt/miniconda3/envs/vienna/bin/RNAfold` exists. If installed elsewhere, edit the
`RNAFOLD` constant in the relevant scripts.

**CUDA out of memory**: Reduce batch size for embedding generation:
```bash
python scripts/apobec3a/expand_dataset.py --batch-size 4
```

**Check environment**: Run the diagnostic script to verify all dependencies:
```bash
python scripts/apobec3a/check_environment.py
```

---

## Known Reproduction Pitfalls

These are bugs that have caused silent incorrect results during reproduction. Read before running.

### 1. Loop features are silent zeros if run out of order (CRITICAL)

`exp_loop_position_analysis.py` must run before `exp_classification_a3a_5fold.py`.

If `loop_position_per_site.csv` is missing, the code in `load_all_data()` initializes
`loop_df = pd.DataFrame()` and silently returns zero-vectors for all loop features. XGBoost
trains on motif+struct only, loop geometry features show 0.000 importance, and GB_HandFeatures
AUROC drops from 0.923 to 0.908. **No warning is printed.**

Fix if this happened: re-run GB models with the patched script:
```bash
python scripts/apobec3a/retrain_clinvar_and_rescore.py  # re-scores ClinVar too
```
Or simply rerun `exp_classification_a3a_5fold.py` after generating loop_position_per_site.csv.

### 2. ClinVar training: TC motif fraction must match between positives and negatives

The APOBEC3A positive set is 86.1% TC-context. Tier2/tier3 negatives are 99.9% TC-context.
Training a classifier with this imbalance inverts the TC signal: the model learns that TC predicts
**negative** and scores non-TC ClinVar variants as high-risk. AUROC looks fine (~0.93) but
enrichment direction is wrong.

Fix: use hybrid negatives (see `scripts/apobec3a/retrain_clinvar_and_rescore.py`) that match
the TC% of positives by adding Asaoka non-TC sites to the negative set.

### 3. EditRNA_rate: Sigmoid output bounded to (0,1) but targets are Z-scored (range ~[-3, +3])

`APOBECMultiTaskHead.rate_head` in `src/models/apobec_edit_embedding.py` ends with `nn.Sigmoid()`.
MSE loss against Z-scored targets causes systematic underfit. The model achieves positive Spearman
by preserving rank order but R²=-0.049 (worse than mean predictor). GB_HandFeatures (R²=+0.014)
is actually the best rate model despite lower Spearman.

Fix: remove `nn.Sigmoid()` from `rate_head`. Not applied yet to preserve reference reproducibility.

### 4. Rate regression: never include negatives

Only include `is_edited == 1` sites in rate regression. Including negatives (rate=0) inflates
Spearman from ~0.12 to ~0.82 by exploiting the binary label signal, not actual rate prediction.

### 5. Baysal sites in classification

Baysal 2016 sites are a subset of Asaoka 2019 (same A3A HEK293T system). They are deduplicated
into `asaoka_2019` during `filter_a3a_splits.py`. Never add Baysal back to the classification
dataset — it causes duplicate-site data leakage.
