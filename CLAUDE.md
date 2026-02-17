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
