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
└── apobec/                # APOBEC experiments

scripts/
└── apobec/                # APOBEC preprocessing scripts

data/                      # Raw and processed datasets
tests/                     # Unit tests
```

---

## Key Patterns

- **Embedders**: Inherit from base classes in `src/embedding/`
- **Experiments**: Config-driven, placed in `experiments/apobec/`
- **Preprocessing**: Scripts go in `scripts/apobec/`

---

## Setting Up on a New Machine

After cloning the git repo, you need to restore the large data files that are excluded from git (embeddings, genomes, model checkpoints, processed CSVs, etc.).

### Step 1: Unpack data files

Two tar.gz archives contain all large files. Copy them to the project root directory and run:

```bash
# Unpack data directory (processed data, embeddings, genomes, published datasets, Excel)
tar -xzf editrna_data.tar.gz

# Unpack model checkpoints and experiment results (best_model.pt, result JSONs, numpy arrays)
tar -xzf editrna_model_outputs.tar.gz
```

Both archives preserve the original directory structure and will extract into the correct locations (`data/`, `experiments/apobec/outputs/`, and `C2TFinalSites.DB.xlsx`).

### Step 2: Verify the data

After unpacking, confirm the key files exist:

```bash
# Genome reference (needed for sequence extraction)
ls data/raw/genomes/hg19.fa

# Pre-computed RNA-FM embeddings (needed for training/evaluation)
ls data/processed/embeddings/rnafm_*.pt

# Processed splits (needed for all experiments)
ls data/processed/splits_levanon_expanded.csv
ls data/processed/splits_expanded.csv

# Best model checkpoints
ls experiments/apobec/outputs/exp2_levanon_tiered_negatives/best_model.pt
```

### Step 3: Set up the conda environment

```bash
conda activate quris
pip install -r requirements.txt
```

### What each archive contains

| Archive | Contents | Size (compressed) |
|---------|----------|-------------------|
| `editrna_data.tar.gz` | `data/` directory + `C2TFinalSites.DB.xlsx` — genomes, embeddings, processed CSVs, published datasets, tiered negatives, sequences | ~14GB uncompressed |
| `editrna_model_outputs.tar.gz` | Model checkpoints (`.pt`), result JSONs, numpy arrays from `experiments/apobec/outputs/` | ~9GB uncompressed |

### Regenerating data (if archives unavailable)

If you don't have the archives, most data can be regenerated from scripts:

```bash
# 1. Download hg19 genome
# Download from UCSC: https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz
# Place at: data/raw/genomes/hg19.fa

# 2. Parse the source Excel file
python scripts/apobec/parse_advisor_excel.py

# 3. Extract sequences (requires hg19.fa)
python scripts/apobec/extract_sequences_and_structures.py

# 4. Generate tiered negatives
python scripts/apobec/generate_tiered_negatives.py

# 5. Build expanded dataset splits
python scripts/apobec/expand_dataset.py

# 6. Generate RNA-FM embeddings (requires GPU, ~30min)
python scripts/apobec/generate_embeddings.py

# 7. Generate structure cache
python scripts/apobec/generate_structure_cache.py
```
