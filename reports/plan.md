# Per-Enzyme Report Expansion with Core `src/` Refactoring

**Last updated**: 2026-03-08

## Context

The current multi-enzyme report (5 sections) is missing classification, feature importance, ClinVar clinical analysis, rate analysis, and more. Goals:
1. **Separate per-enzyme experiment directories** (`experiments/apobec3b/`, `experiments/apobec3g/`)
2. **Core capabilities in `src/`** (feature extraction, data loading, negative generation)
3. **Maximum clinical analysis** — per-enzyme ClinVar with clinical consultant agents
4. **Per-enzyme HTML reports** (like v3 APOBEC3A report) + updated cross-enzyme comparison

Feature extraction logic was duplicated in 15+ experiment files across `experiments/apobec3a/`. Refactoring centralizes this in `src/` so `apobec3b/` and `apobec3g/` simply import.

---

## Architecture

```
src/
├── data/
│   └── apobec_feature_extraction.py   ← NEW: motif/loop/structure feature extraction
│   └── apobec_negatives.py            ← NEW: motif-matched negative generation
│   └── apobec_clinvar.py              ← NEW: ClinVar scoring utilities

experiments/
├── apobec3a/          ← EXISTING: update to import from src/ (no functional change)
├── apobec3b/          ← NEW: A3B-specific experiments + report
│   ├── exp_classification_a3b.py
│   ├── exp_clinvar_a3b.py
│   ├── exp_rate_analysis_a3b.py
│   ├── generate_html_report.py
│   └── outputs/
├── apobec3g/          ← NEW: A3G-specific experiments + report
│   ├── exp_classification_a3g.py
│   ├── exp_clinvar_a3g.py
│   ├── exp_rate_analysis_a3g.py
│   ├── generate_html_report.py
│   └── outputs/
└── multi_enzyme/      ← UPDATE: add classification, feature importance, ClinVar cross-enzyme sections
```

---

## Step-by-Step Execution Status

### Part 1: `src/` Core Modules

| Step | Action | File | Status |
|------|--------|------|--------|
| 1 | CREATE | `src/data/apobec_feature_extraction.py` | ✅ Done |
| 2 | CREATE | `src/data/apobec_negatives.py` | ✅ Done |
| 3 | CREATE | `src/data/apobec_clinvar.py` | ✅ Done |

### Part 2: Negative Generation Script

| Step | Action | File | Status |
|------|--------|------|--------|
| 4 | CREATE | `scripts/multi_enzyme/generate_negatives_v2.py` | ✅ Done |
| 5 | **RUN** | `generate_negatives_v2.py` (~30min, requires hg38.fa) | ⏳ Needs run |
| 6 | **RUN** | `generate_loop_positions.py --incremental` (~5min) | ⏳ Needs run |
| 7 | **RUN** | `generate_structure_cache.py --incremental` (~5min) | ⏳ Needs run |

### Part 3: `experiments/apobec3b/`

| Step | Action | File | Status |
|------|--------|------|--------|
| 8 | CREATE | `experiments/apobec3b/exp_classification_a3b.py` | ✅ Done |
| 9 | CREATE | `experiments/apobec3g/exp_classification_a3g.py` | ✅ Done |
| 10 | **RUN** | Both classification experiments (~30min) | ⏳ After step 7 |
| 11 | CREATE | `experiments/apobec3b/exp_rate_analysis_a3b.py` | ✅ Done |
| 12 | CREATE | `experiments/apobec3g/exp_rate_analysis_a3g.py` | ✅ Done |
| 13 | **RUN** | Both rate experiments (~5min) | ⏳ After step 7 |
| 14 | CREATE | `experiments/apobec3b/exp_clinvar_a3b.py` | ✅ Done |
| 15 | CREATE | `experiments/apobec3g/exp_clinvar_a3g.py` | ✅ Done |
| 16 | **RUN** | Both ClinVar experiments (4-6h or <30min if cache) | ⏳ After step 7 |
| 17 | **LAUNCH** | Clinical consultant agents (~20min parallel) | ⏳ After step 16 |

### Part 4: HTML Reports

| Step | Action | File | Status |
|------|--------|------|--------|
| 18 | CREATE | `experiments/apobec3b/generate_html_report.py` | 🔄 Agent writing |
| 19 | CREATE | `experiments/apobec3g/generate_html_report.py` | 🔄 Agent writing |
| 20 | UPDATE | `experiments/multi_enzyme/generate_html_report.py` (+4 new sections) | ✅ Done |
| 21 | **RUN** | All 3 report generators (~5min) | ⏳ After step 16 |

---

## Key Files Created/Modified

| File | Action | Lines | Source |
|------|--------|-------|--------|
| `src/data/apobec_feature_extraction.py` | CREATE | ~300 | Extract from `exp_classification_a3a_5fold.py:131-210` + `exp_clinvar_prediction.py:97-408` |
| `src/data/apobec_negatives.py` | CREATE | ~200 | Extract from `build_multi_enzyme_dataset.py`, `generate_negatives.py` |
| `src/data/apobec_clinvar.py` | CREATE | ~200 | Extract from `exp_clinvar_prediction.py:384-600` + `exp_clinvar_calibrated.py:51-90` |
| `scripts/multi_enzyme/generate_negatives_v2.py` | CREATE | ~150 | Calls src modules |
| `experiments/apobec3b/exp_classification_a3b.py` | CREATE | ~250 | Mirrors `exp_classification_a3a_5fold.py`, imports src/ |
| `experiments/apobec3b/exp_clinvar_a3b.py` | CREATE | ~300 | Mirrors `exp_clinvar_prediction.py`, imports src/ |
| `experiments/apobec3b/exp_rate_analysis_a3b.py` | CREATE | ~200 | New |
| `experiments/apobec3b/generate_html_report.py` | CREATE | ~400 | Mirrors A3A v3_report style |
| `experiments/apobec3g/exp_classification_a3g.py` | CREATE | ~220 | Mirrors A3B version + bootstrap CI |
| `experiments/apobec3g/exp_clinvar_a3g.py` | CREATE | ~280 | CC-context focus, small-n caveats |
| `experiments/apobec3g/exp_rate_analysis_a3g.py` | CREATE | ~170 | Dang NK_Hyp vs NK_Norm |
| `experiments/apobec3g/generate_html_report.py` | CREATE | ~400 | Small-dataset caveats throughout |
| `experiments/multi_enzyme/generate_html_report.py` | UPDATE | +200 | 4 new sections: Classification, Feature Importance, ClinVar, Clinical Interpretation |

---

## ClinVar Feature Cache Strategy

The ClinVar feature computation is the bottleneck (4-6h for 1.68M variants).

1. **Check** `data/processed/clinvar_features_cache.npz` — if it exists, all 3 enzyme models can use it
2. **If missing**: `exp_clinvar_a3b.py` runs feature computation first (saves to cache), then A3G reuses
3. **Cache format**: `{ 'site_ids': [...], 'hand_features': np.ndarray[1.68M × 46] }` — enzyme-agnostic
4. Once cached, all 3 models score in <5 min each

The existing `clinvar_all_scores.csv` from A3A pipeline has SCORES but not the raw feature vectors. Need to check if the A3A feature computation left a cache.

---

## ClinVar Clinical Consultant Agents (Step 17)

After per-enzyme ClinVar scoring completes, launch 3 agents in parallel:

**Agent 1: `disease-annotator`**
- Aggregate predictions by ICD-10 disease category per enzyme
- Top 20 diseases by predicted editing enrichment
- Diseases exclusively enriched by A3B or A3G (not A3A)

**Agent 2: `variant-interpreter`**
- Top 100 predicted pathogenic targets per enzyme: gene names, molecular consequence
- Do A3B and A3G edit different gene categories than A3A?
- Oncogenes vs tumor suppressors differentially targeted?

**Agent 3: `enzyme-comparator`**
- Cross-enzyme ClinVar enrichment comparison
- Do A3B/A3G predictions complement A3A?
- Cross-enzyme comparison table and clinical relevance summary

Output: `experiments/multi_enzyme/outputs/clinvar/clinical_interpretation.md`

---

## Execution Commands

All commands use the `quris` conda environment. Phases 2+3 can run in parallel after Phase 1.

```bash
# Phase 1: Generate negatives and update caches (~40min)
conda run -n quris python scripts/multi_enzyme/generate_negatives_v2.py
conda run -n quris python scripts/multi_enzyme/generate_loop_positions.py --incremental
conda run -n quris python scripts/multi_enzyme/generate_structure_cache.py --incremental

# Phase 2: A3B experiments (~45min, or ~5h if no ClinVar cache)
conda run -n quris python experiments/apobec3b/exp_classification_a3b.py
conda run -n quris python experiments/apobec3b/exp_rate_analysis_a3b.py
conda run -n quris python experiments/apobec3b/exp_clinvar_a3b.py
conda run -n quris python experiments/apobec3b/generate_html_report.py

# Phase 3: A3G experiments (~10min, parallel with Phase 2)
conda run -n quris python experiments/apobec3g/exp_classification_a3g.py
conda run -n quris python experiments/apobec3g/exp_rate_analysis_a3g.py
conda run -n quris python experiments/apobec3g/exp_clinvar_a3g.py
conda run -n quris python experiments/apobec3g/generate_html_report.py

# Phase 4: Regenerate multi-enzyme report (~1min)
conda run -n quris python experiments/multi_enzyme/generate_html_report.py

# Phase 5: Clinical consultant agents (after ClinVar completes)
# → Launched automatically by team-lead
```

---

## Verification Checklist

- [ ] `python -c "from src.data.apobec_feature_extraction import build_hand_features; print('OK')"` — src import test
- [ ] Check negatives: `df.groupby(['enzyme','is_edited']).size()` — balanced classes
- [ ] Check motif matching: TC% for A3A negatives ~51%, A3B ~32%, A3G CC% ~91%
- [ ] Classification sanity: All enzyme AUROCs > 0.6 (above random)
- [ ] ClinVar: A3A enrichment matches v3 report (OR ~ 1.3 at threshold 0.5)
- [ ] HTML reports: All images load, no missing JSON sections
