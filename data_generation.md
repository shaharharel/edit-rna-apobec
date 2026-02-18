# Data Generation Pipeline

Complete guide to regenerating all processed data from raw source files.

## Overview

The data pipeline transforms raw datasets (Excel files, published supplementary tables) into
training-ready splits with pre-computed embeddings. Total processing time: ~1-2 hours (with GPU).

```
Raw source files (Excel, XLS)
    ↓
Stage 1: Parse → per-dataset CSVs
    ↓
Stage 2: Extract labels + negatives
    ↓
Stage 3: Build unified dataset
    ↓
Stage 4: Extract sequences (requires hg19 genome)
    ↓
Stage 5: Generate tiered negatives (requires hg19 + RNAfold)
    ↓
Stage 6: Expand dataset + generate embeddings (requires GPU)
    ↓
Stage 7: Generate structure cache (requires ViennaRNA)
```

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

**hg19 Reference Genome** (~3GB):
```bash
mkdir -p data/raw/genomes
cd data/raw/genomes
wget https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz
gunzip hg19.fa.gz
# Index for pyfaidx:
python -c "from pyfaidx import Fasta; Fasta('hg19.fa')"
```

**RefSeq Gene Annotations** (~23MB):
```bash
cd data/raw/genomes
wget https://hgdownload.soe.ucsc.edu/goldenPath/hg19/database/refGene.txt.gz
gunzip refGene.txt.gz
```

---

## Raw Source Files (provided in `editrna_raw_data.tar.gz`)

These files cannot be generated — they must be obtained from the original sources.

| File | Size | Source |
|------|------|--------|
| `C2TFinalSites.DB.xlsx` | 6.1MB | Advisor's curated database (636 C-to-U editing sites) |
| `data/raw/published/asaoka_2019_table_s1.xls` | 1.0MB | Asaoka et al. 2019, DOI: 10.3390/ijms20225621 |
| `data/raw/published/sharma_2015_supp_data.xls` | 3.2MB | Sharma et al. 2015, DOI: 10.1038/ncomms7881 |
| `data/raw/published/alqassim_2021/alqassim_2021_supp_data1.xlsx` | 96KB | Alqassim et al. 2021, DOI: 10.1038/s42003-020-01620-x |
| `data/raw/published/alqassim_2021/alqassim_2021_supp_data2.xlsx` | 80KB | Alqassim et al. 2021 (gene expression) |

---

## Pipeline Steps

All commands assume you are in the project root and using the `quris` conda environment
unless noted otherwise.

```bash
conda activate quris
```

### Stage 1: Parse Raw Datasets

Parse each source file into standardized CSVs.

```bash
# 1a. Parse advisor Excel (636 positive sites + supplementary tables)
python scripts/apobec/parse_advisor_excel.py --excel C2TFinalSites.DB.xlsx
# Output: data/processed/advisor/*.csv (11 files)
#         data/processed/advisor/unified_editing_sites.csv

# 1b. Parse published datasets
python scripts/apobec/parse_asaoka_2019.py
# Output: data/processed/published/asaoka_2019_editing_sites.csv (5,208 sites)

python scripts/apobec/parse_sharma_2015.py
# Output: data/processed/published/sharma_2015_editing_sites.csv (333 sites)

python scripts/apobec/parse_alqassim_2021.py
# Output: data/processed/published/alqassim_2021_editing_sites.csv (209 sites)
```

### Stage 2: Extract Labels and Negative Controls

```bash
# 2a. Extract multi-task labels from advisor data
python scripts/apobec/extract_labels.py
# Output: data/processed/editing_sites_labels.csv (636 sites, 30 label columns)
#         data/processed/splits.csv (train/val/test by gene stratification)

# 2b. Extract negative controls from non-AG mismatch sites
python scripts/apobec/extract_negative_controls.py
# Output: data/processed/advisor/negative_controls_ct.csv (CT mismatches, positives removed)
#         data/processed/advisor/negative_controls_filtered.csv (exonic only, ~325 sites)
```

### Stage 3: Build Unified Dataset

Combine all datasets into a single table with standardized columns.

```bash
python scripts/apobec/build_unified_dataset.py
# Output: data/processed/all_datasets_combined.csv (all positive sites, deduplicated)
```

### Stage 4: Extract Sequences and Structures

Extract 201nt RNA sequences (±100 flanking around edit site) from hg19 and
predict secondary structures with RNAfold.

```bash
python scripts/apobec/extract_sequences_and_structures.py \
    --genome data/raw/genomes/hg19.fa \
    --negatives
# Output: data/processed/sequences_and_structures.csv
# Time: ~10-20 minutes (depends on RNAfold speed)
```

### Stage 5: Generate Tiered Negatives

Generate hard negative controls at three difficulty tiers from RefSeq exonic regions
of genes that contain positive editing sites.

```bash
python scripts/apobec/generate_tiered_negatives.py
# Output: data/processed/negatives_tier1.csv (~272K all exonic C sites)
#         data/processed/negatives_tier2.csv (~132K TC-motif C sites)
#         data/processed/negatives_tier3.csv (~75K TC in stem-loops)
# Time: ~30-60 minutes (Tier 3 requires RNAfold on each site)
```

Requires: `hg19.fa`, `refGene.txt`, ViennaRNA RNAfold.

### Stage 6: Expand Dataset and Generate Embeddings

Subsample tiered negatives, merge with positives, extract sequences for new sites,
generate RNA-FM embeddings, and create train/val/test splits.

```bash
python scripts/apobec/expand_dataset.py \
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
# Time: ~30 minutes with GPU, ~2 hours CPU-only
```

**Note:** This step generates RNA-FM embeddings internally. If you want to generate
embeddings separately (e.g., with a different encoder), use:

```bash
python scripts/apobec/generate_embeddings.py \
    --sequences_json data/processed/site_sequences.json \
    --include_edited \
    --encoder rnafm \
    --batch_size 32
```

### Stage 7: Generate Structure Cache

Compute ViennaRNA 2D structure features (pairing probability, accessibility, entropy)
for all sites. **Must be run in the `vienna` conda environment.**

```bash
/opt/miniconda3/envs/vienna/bin/python scripts/apobec/generate_structure_cache.py
# Output: data/processed/embeddings/vienna_structure_cache.npz
# Time: ~10-15 minutes
```

---

## Quick Reference: Full Pipeline

Copy-paste to run the entire pipeline:

```bash
conda activate quris

# Stage 1: Parse
python scripts/apobec/parse_advisor_excel.py --excel C2TFinalSites.DB.xlsx
python scripts/apobec/parse_asaoka_2019.py
python scripts/apobec/parse_sharma_2015.py
python scripts/apobec/parse_alqassim_2021.py

# Stage 2: Labels + negatives
python scripts/apobec/extract_labels.py
python scripts/apobec/extract_negative_controls.py

# Stage 3: Unify
python scripts/apobec/build_unified_dataset.py

# Stage 4: Sequences (requires hg19.fa)
python scripts/apobec/extract_sequences_and_structures.py --genome data/raw/genomes/hg19.fa --negatives

# Stage 5: Tiered negatives (requires hg19.fa + RNAfold)
python scripts/apobec/generate_tiered_negatives.py

# Stage 6: Expand + embed (requires GPU for speed)
python scripts/apobec/expand_dataset.py --max-tier2-neg 2000 --max-tier3-neg 1000

# Stage 7: Structure cache (vienna env)
/opt/miniconda3/envs/vienna/bin/python scripts/apobec/generate_structure_cache.py
```

---

## Output File Summary

After running the full pipeline, the `data/processed/` directory should contain:

```
data/processed/
├── advisor/                          # Parsed from C2TFinalSites.DB.xlsx
│   ├── t1_gtex_editing_&_conservation.csv
│   ├── t2_additional_editing.csv
│   ├── t3_enzyme_correlations.csv
│   ├── t4_immune_cell_editing.csv
│   ├── t5_tcga_survival.csv
│   ├── supp_t3_structures.csv
│   ├── supp_tx_all_non_ag_mm_sites.csv
│   ├── unified_editing_sites.csv
│   ├── negative_controls_ct.csv
│   └── negative_controls_filtered.csv
├── published/                        # Parsed from published papers
│   ├── asaoka_2019_editing_sites.csv
│   ├── sharma_2015_editing_sites.csv
│   └── alqassim_2021_editing_sites.csv
├── editing_sites_labels.csv          # 636 sites with multi-task labels
├── splits.csv                        # Base train/val/test splits
├── all_datasets_combined.csv         # All positives unified
├── sequences_and_structures.csv      # 201nt sequences + RNAfold structures
├── negatives_tier1.csv               # ~272K all exonic C
├── negatives_tier2.csv               # ~132K TC-motif
├── negatives_tier3.csv               # ~75K TC in stem-loops
├── site_sequences.json               # All sequences by site_id
├── splits_expanded.csv               # Full dataset splits (all datasets)
├── splits_levanon_expanded.csv       # Levanon + tiered negatives splits
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

**`pyfaidx` index error**: If hg19.fa is not indexed, run:
```bash
python -c "from pyfaidx import Fasta; Fasta('data/raw/genomes/hg19.fa')"
```

**RNAfold not found**: Ensure ViennaRNA is installed and the path
`/opt/miniconda3/envs/vienna/bin/RNAfold` exists. If installed elsewhere, edit the
`RNAFOLD` constant in the relevant scripts.

**CUDA out of memory**: Reduce batch size for embedding generation:
```bash
python scripts/apobec/expand_dataset.py --batch-size 4
```

**Check environment**: Run the diagnostic script to verify all dependencies:
```bash
python scripts/apobec/check_environment.py
```
