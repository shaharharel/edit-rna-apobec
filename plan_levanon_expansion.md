# Levanon Enzyme Expansion Plan

**Created**: 2026-03-18

---

## Project Summary (What's Done)

### Data Pipeline (Complete)
- 7 raw datasets parsed into unified format
- hg38 genome sequences extracted (201-nt centered on edit site)
- RNA-FM embeddings (pooled + token-level)
- ViennaRNA structure cache (pairing probabilities, MFE, delta features)
- Loop position geometry for all sites

### A3A Pipeline (Complete)
- **Data**: 5,187 positives (Asaoka 4933, Levanon/Advisor 120 A3A-only, Alqassim 128, Sharma 6) + 2,966 negatives
- **Baysal 2016**: 4,211 sites used for rate prediction only (subset of Asaoka, deduplicated)
- **30+ experiments** reproduced: classification, rate prediction, ClinVar, cross-dataset, embedding viz, motif/structure analysis
- **Classification**: EditRNA+Features AUROC=0.935, GB_HandFeatures AUROC=0.923
- **Rate**: GB_HandFeatures Spearman=0.122 (best). Rate prediction is hard — cross-dataset generalization fails
- **ClinVar**: GB_Full OR=1.33 (p<1e-40) for pathogenic enrichment. RNAsee rules-based shows depletion (OR=0.76)
- **Report**: `experiments/apobec3a/outputs/v3_report.html` (1.1 MB, self-contained)

### Multi-Enzyme Pipeline (Complete)
- **Refactored `src/` modules**: `apobec_feature_extraction.py`, `apobec_negatives.py`, `apobec_clinvar.py`
- **A3B**: 4,180 positives (Kockler 3679 + Zhang 501) + 4,177 negatives (TC-matched ~32%)
  - Classification AUROC=0.831, ClinVar OR=1.08 raw / 1.55 calibrated
  - Top feature: `local_unpaired_fraction`. No 3'-end loop preference (RLP=0.515)
  - Report: `experiments/apobec3b/outputs/apobec3b_report.html` (232 KB)
- **A3G**: 119 positives (Dang 2019) + 119 negatives (CC-matched ~91%)
  - Classification AUROC=0.931 (bootstrap CI 0.89–0.96), ClinVar CC-context OR=1.76
  - Top features: `dist_to_apex`=0.319, `relative_loop_position`=0.170. Tetraloop specialist (RLP=0.920)
  - Report: `experiments/apobec3g/outputs/apobec3g_report.html` (316 KB)
- **Cross-enzyme report**: `experiments/multi_enzyme/outputs/multi_enzyme_report.html` (982 KB)
  - 9 sections: dataset overview, motif, structure, cross-enzyme comparison, classification, feature importance, ClinVar, clinical interpretation, methods

### Key Scientific Findings
1. **Structure > sequence**: For all enzymes, structural features (loop geometry) outperform motif-only models
2. **Three distinct editing programs**: A3A=TC+moderate 3'-bias, A3B=mixed motif+no positional bias, A3G=CC+extreme 3'-end tetraloop
3. **A3B resolves published contradiction**: Butt 2024 (uses loops ✓) + Alonso de la Vega 2023 (no positional preference ✓) — both correct, measuring different things
4. **All enzymes show ClinVar pathogenic enrichment**: A3A OR=1.33, A3B OR=1.55 calibrated, A3G OR=1.76 CC-context
5. **GB is the only model with real ClinVar signal**: Neural models and RNAsee rules-based show no enrichment

### Report Locations
| Report | Path | Size |
|--------|------|------|
| A3A | `experiments/apobec3a/outputs/v3_report.html` | 1.1 MB |
| A3B | `experiments/apobec3b/outputs/apobec3b_report.html` | 232 KB |
| A3G | `experiments/apobec3g/outputs/apobec3g_report.html` | 316 KB |
| Multi-enzyme | `experiments/multi_enzyme/outputs/multi_enzyme_report.html` | 982 KB |

### Key Data Files
| File | Contents |
|------|----------|
| `data/processed/splits_expanded_a3a.csv` | 8,153 A3A-only sites (classification + rate) |
| `data/processed/splits_expanded.csv` | 8,669 sites (includes non-A3A advisor sites) |
| `data/processed/multi_enzyme/splits_multi_enzyme_v2_with_negatives.csv` | 14,310 sites (A3A/A3B/A3G + negatives) |
| `data/processed/multi_enzyme/multi_enzyme_sequences_v2_with_negatives.json` | 14,310 × 201-nt sequences |
| `data/processed/multi_enzyme/loop_position_per_site_v2.csv` | Loop geometry for all 14,310 sites |
| `data/processed/multi_enzyme/structure_cache_multi_enzyme_v2.npz` | ViennaRNA delta features for all 14,310 |
| `data/processed/clinvar_features_cache.npz` | 1.69M ClinVar variants × 46 features |
| `data/raw/C2TFinalSites.DB.xlsx` | Levanon/Advisor 636 C-to-U sites |
| `data/raw/genomes/hg38.fa` | Reference genome |

---

## What's NOT Done — The Levanon Expansion

### The Problem

The Levanon/Advisor database (`data/raw/C2TFinalSites.DB.xlsx`) has 636 C-to-U editing sites categorized by which APOBEC enzyme affects them:

| Category | n | Status |
|----------|---|--------|
| APOBEC3A Only | 120 | **Already modeled** (in A3A pipeline as `advisor_c2t`) |
| APOBEC3G Only | 60 | **NOT modeled** |
| Both (A3A+A3G) | 178 | **NOT modeled** |
| Neither | 206 | **NOT modeled** |

All 636 sites have:
- Genomic coordinates (Chr, Start, End) in hg38
- No explicit strand column (need to infer from gene/CDS annotation or check C vs G at position)
- GTEx editing rates across 54 tissues
- Conservation data across species
- Gene name, exonic function (synonymous/nonsynonymous/stopgain)

**Note**: The existing A3A pipeline already has 120 "APOBEC3A Only" sites in `splits_expanded_a3a.csv` as `advisor_c2t`. The original parsing script is at `scripts/apobec3a/parse_advisor_excel.py`.

### The existing 120 A3A-only sites are already in `data/processed/site_sequences.json` with 201-nt sequences. Check if ALL 636 sites were extracted or just the 120.

---

## Detailed Plan

### Phase 1: Parse All Levanon Sites

**Script**: `scripts/multi_enzyme/parse_levanon_all_categories.py`

1. Read `data/raw/C2TFinalSites.DB.xlsx` (header at row 1)
2. For each of the 636 sites:
   - Extract Chr, Start, End
   - Determine strand: check reference genome — if base at position is C → plus strand, if G → minus strand (C-to-U editing must be on a C)
   - Assign enzyme category from "Affecting Over Expressed APOBEC" column
   - Extract mean GTEx editing rate
3. Extract 201-nt sequences from hg38.fa (center on edit position, ensure C at position 100)
4. Assign site_ids: `levanon_a3g_{i}`, `levanon_both_{i}`, `levanon_neither_{i}`
   (The existing 120 A3A-only sites keep their `C2U_NNNN` IDs)

**Output**:
- `data/processed/multi_enzyme/levanon_all_categories.csv` — 636 rows with columns: site_id, chr, start, strand, enzyme_category, dataset_source, editing_rate, gene, exonic_function, tissue_classification
- Updated sequences JSON

### Phase 2: Build Expanded Multi-Enzyme Dataset (v3)

**Script**: `scripts/multi_enzyme/build_multi_enzyme_dataset_v3.py`

Merge existing v2 dataset with new Levanon categories:

| Enzyme | Existing positives | New Levanon | Total positives |
|--------|-------------------|-------------|-----------------|
| A3A | 2,749 (Kockler) | 0 (120 already in A3A-only pipeline, not in multi-enzyme) | 2,749 |
| A3B | 4,180 (Kockler+Zhang) | 0 | 4,180 |
| A3G | 119 (Dang) | +60 (Levanon A3G-only) | 179 |
| A3A_A3G | 0 | +178 (Levanon Both) | 178 |
| Neither | 0 | +206 (Levanon Neither) | 206 |

**Decision point for "Both" sites**: These 178 sites respond to both A3A and A3G overexpression. They should be modeled as their own category `A3A_A3G` since they represent a shared editing substrate. Alternatively, they could be added to BOTH A3A and A3G — but that risks data leakage if A3A and A3G models are compared.

**Recommendation**: Create `A3A_A3G` as a new enzyme category. This is scientifically interesting — what makes a site editable by both enzymes?

### Phase 3: Generate Negatives for New Categories

Use `src/data/apobec_negatives.py` → `generate_negatives_from_genome()`:

| Category | n_pos | Motif to match | Notes |
|----------|-------|----------------|-------|
| A3G (expanded) | 179 | CC ~91% | Match Dang A3G motif distribution |
| A3A_A3G | 178 | Check actual motif distribution first | May be intermediate between A3A (TC~51%) and A3G (CC~91%) |
| Neither | 206 | Check actual motif distribution first | Unknown editor — match whatever motif these sites have |

**Important**: Check the actual motif distribution of "Both" and "Neither" sites BEFORE generating negatives. Don't assume they match A3A or A3G.

### Phase 4: Update Caches

1. Run `scripts/multi_enzyme/generate_loop_positions.py --incremental` for new sites
2. Run `scripts/multi_enzyme/generate_structure_cache.py --incremental` for new sites
3. Both scripts already support incremental mode

### Phase 5: Run Experiments

#### 5a: Update A3G experiments (119 → 179 positives)
- Rerun `experiments/apobec3g/exp_classification_a3g.py` with expanded data
- Rerun `experiments/apobec3g/exp_clinvar_a3g.py`
- Rerun `experiments/apobec3g/exp_rate_analysis_a3g.py` (Levanon sites have GTEx rates!)
- Regenerate `experiments/apobec3g/generate_html_report.py`

#### 5b: New A3A_A3G experiments
- Create `experiments/apobec_both/` directory
- `exp_classification_both.py` — what features distinguish dual-target sites?
- `exp_motif_analysis_both.py` — motif distribution (TC? CC? both?)
- `exp_rate_analysis_both.py` — GTEx tissue rates (do they match A3A or A3G patterns?)
- `generate_html_report.py` — per-category report

#### 5c: New "Neither" experiments
- Create `experiments/apobec_neither/` directory
- `exp_classification_neither.py` — can we classify these? What features matter?
- `exp_motif_analysis_neither.py` — what motif do "Neither" sites prefer?
- `exp_rate_analysis_neither.py` — GTEx rates and tissue distribution
- `generate_html_report.py`

#### 5d: Update multi-enzyme report
- Add A3A_A3G and Neither to cross-enzyme comparison
- Update motif/structure/classification comparison tables
- Regenerate `experiments/multi_enzyme/generate_html_report.py`

### Phase 6: Scientific Analysis

Key questions to answer:
1. **A3A_A3G (Both)**: What makes a site editable by both? Is it intermediate motif (some TC, some CC)? Is structure different?
2. **Neither**: What enzyme edits these? APOBEC1? Check if they're enriched for specific tissues (liver = APOBEC1 territory for apoB mRNA editing). Check conservation patterns.
3. **A3G expanded**: Does adding 60 Levanon sites improve A3G classification? Do GTEx rates correlate with Dang NK-cell rates?

---

## Execution Order

```bash
# Phase 1: Parse Levanon (create script + run, ~10 min)
conda run -n quris python scripts/multi_enzyme/parse_levanon_all_categories.py

# Phase 2: Build v3 dataset (~5 min)
conda run -n quris python scripts/multi_enzyme/build_multi_enzyme_dataset_v3.py

# Phase 3: Generate negatives (~30 min, needs hg38.fa)
conda run -n quris python scripts/multi_enzyme/generate_negatives_v3.py

# Phase 4: Update caches (~30 min)
conda run -n quris python scripts/multi_enzyme/generate_loop_positions.py --incremental
conda run -n quris python scripts/multi_enzyme/generate_structure_cache.py --incremental

# Phase 5a: Rerun A3G experiments (~5 min)
conda run -n quris python experiments/apobec3g/exp_classification_a3g.py
conda run -n quris python experiments/apobec3g/exp_rate_analysis_a3g.py
conda run -n quris python experiments/apobec3g/exp_clinvar_a3g.py
conda run -n quris python experiments/apobec3g/generate_html_report.py

# Phase 5b: A3A_A3G experiments (~10 min)
conda run -n quris python experiments/apobec_both/exp_classification_both.py
conda run -n quris python experiments/apobec_both/exp_rate_analysis_both.py
conda run -n quris python experiments/apobec_both/generate_html_report.py

# Phase 5c: Neither experiments (~10 min)
conda run -n quris python experiments/apobec_neither/exp_classification_neither.py
conda run -n quris python experiments/apobec_neither/exp_rate_analysis_neither.py
conda run -n quris python experiments/apobec_neither/generate_html_report.py

# Phase 5d: Regenerate multi-enzyme report (~2 min)
conda run -n quris python experiments/multi_enzyme/generate_html_report.py
```

---

## Verification Checklist

- [ ] All 636 Levanon sites parsed (120 A3A + 60 A3G + 178 Both + 206 Neither)
- [ ] All sites have 201-nt sequences with C at position 100
- [ ] A3G: 179 positives (119 Dang + 60 Levanon)
- [ ] A3A_A3G: 178 positives with motif-matched negatives
- [ ] Neither: 206 positives with motif-matched negatives
- [ ] Loop positions and structure cache cover all new sites
- [ ] Check "Neither" tissue distribution — liver enrichment would suggest APOBEC1
- [ ] Check "Both" motif — intermediate TC/CC would confirm dual recognition
- [ ] All HTML reports regenerated with new data
- [ ] Multi-enzyme report updated with 5 enzyme categories
