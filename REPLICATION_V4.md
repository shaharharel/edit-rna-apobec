# V4 Pipeline Replication Guide

End-to-end reproduction of the v4 (cancer_matched / cds_unbiased) pipeline,
from raw data through panel scoring, fair sweeps, POG570 validation, and the
final HTML report.

The single entry point is:

```bash
bash scripts/replicate_v4_pipeline.sh
```

## Quick Reference

| Mode | Command | Notes |
|------|---------|-------|
| Full pipeline (GCP V100) | `bash scripts/replicate_v4_pipeline.sh` | Default. Stage 4 uses `ai-gpu2`. |
| Full pipeline (local Mac MPS) | `bash scripts/replicate_v4_pipeline.sh --local-mps` | ~24h Stage 4. |
| Resume after data prep | `bash scripts/replicate_v4_pipeline.sh --skip-prep` | Re-uses existing v4 datasets. |
| Sweeps only | `bash scripts/replicate_v4_pipeline.sh --skip-prep --skip-train --skip-apobec1 --skip-score` | Requires panel parquets. |
| Custom GCP VM | `bash scripts/replicate_v4_pipeline.sh --gcp-instance my-vm --gcp-zone us-east1-b` | |

## Stage Skip Flags

```
--skip-prep      Stage 1: build_v4_datasets
--skip-train     Stage 2: Phase3 cancer + cds training
--skip-apobec1   Stage 3: APOBEC1 v4 head retraining
--skip-score     Stage 4: Panel scoring (GPU)
--skip-sweep     Stage 5: Fair sweep / topX / per-cancer / POG570
--skip-verify    Stage 6: QA verification scripts
--skip-report    Stage 7: HTML report
--local-mps      Use local PyTorch-MPS path for Stage 4 (instead of GCP)
```

## Prerequisites

Before running, the following must already be in place. Stage 0 of the script
verifies all of these and aborts with a clear message on failure.

### 1. Conda environment `quris`

```bash
conda env list | grep quris
# If missing:
conda create -n quris python=3.11
conda activate quris
pip install -r requirements.txt
```

### 2. Raw data unpacked

```bash
tar -xzf editrna_raw_data.tar.gz
# verifies: data/raw/C2TFinalSites.DB.xlsx and friends
```

Required files:

```
data/raw/C2TFinalSites.DB.xlsx
data/raw/asaoka_2019_table_s1.xls
data/raw/sharma_2015_supp_data.xls
data/raw/alqassim_2021/
data/raw/baysal_2016/
data/raw/levanon/tissue_editing_rates.csv
```

### 3. Reference genomes

Both hg19 and hg38 are required (Stage 1 needs hg38 for original v3 sites,
hg19 for cancer-matched coordinates):

```bash
mkdir -p data/raw/genomes && cd data/raw/genomes
# hg38
wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz && gunzip hg38.fa.gz
# hg19
wget https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz && gunzip hg19.fa.gz
python -c "from pyfaidx import Fasta; Fasta('hg19.fa'); Fasta('hg38.fa')"
cd ../../..
```

### 4. Phase3 v3 training data

These should already be present from the v3 pipeline. They are **inputs** to v4:

```
data/processed/multi_enzyme/splits_multi_enzyme_v3_with_negatives.csv
data/processed/multi_enzyme/multi_enzyme_sequences_v3_with_negatives.json
data/processed/multi_enzyme/loop_position_per_site_v3.csv
data/processed/multi_enzyme/structure_cache_multi_enzyme_v3.npz
```

If missing: run the v3 pipeline first (`scripts/multi_enzyme/build_multi_enzyme_dataset_v3.py`).

### 5. Cancer mutation data (for Stage 5)

```
data/raw/tcga/*_mutations.txt              (10 cancer types: BLCA, BRCA, CESC, COADREAD, ESCA, HNSC, LIHC, LUSC, SKCM, STAD)
data/raw/pcawg/by_cancer/<cancer>/*.maf    (per-cancer MAFs)
data/raw/pog570/POG570_small_mutations.txt.gz
```

### 6. Panel candidate caches (for Stage 4)

The 95M-row CDS-C panel was precomputed and resides on the GCP VM
(`ai-gpu2`) at `~/data/panel/`:

```
candidates_cache_aligned.parquet
hand40_cache_aligned.npy
valid_mask.npy
rnafm_cds_kept/orig.npy
rnafm_cds_kept/delta.npy
```

For **local-MPS** scoring, mirror these to `~/data/panel/` on the local machine.
Total size ≈ 35 GB.

### 7. GCP setup (default Stage 4 path)

```bash
gcloud auth login
gcloud config set project <project-id>
gcloud compute instances list  # should show ai-gpu2 in us-central1-a
```

V100 quota required (`gcloud compute regions describe us-central1`). If quota
is unavailable, use `--local-mps`.

## Stage-by-stage Outputs

### Stage 1 — v4 data prep (~1–2 h on M1 Pro)

Driver: `scripts/multi_enzyme/build_v4_datasets.py`

Produces:

```
data/processed/multi_enzyme/
    cancer_ct_trinuc_distribution.csv
    cds_c_trinuc_distribution.csv
    splits_multi_enzyme_v4_cancer_matched.csv
    splits_multi_enzyme_v4_cds_unbiased.csv
    multi_enzyme_sequences_v4_cancer_matched.json
    multi_enzyme_sequences_v4_cds_unbiased.json
    loop_position_per_site_v4_cancer_matched.csv
    loop_position_per_site_v4_cds_unbiased.csv
data/processed/embeddings/
    structure_cache_multi_enzyme_v4_cancer_matched.npz
    structure_cache_multi_enzyme_v4_cds_unbiased.npz
    rnafm_v4_cancer_matched.npz
    rnafm_v4_cds_unbiased.npz
```

Most of the wall time is ViennaRNA structure folding (~7,358 positives + 7,358
negatives × 2 variants).

### Stage 2 — Phase3 training (~15 min total, sequential)

Drivers:
- `experiments/multi_enzyme/exp_train_phase3_v4.py --variant cancer_matched`
- `experiments/multi_enzyme/exp_train_phase3_v4.py --variant cds_unbiased`

Produces:

```
experiments/multi_enzyme/outputs/v4_cancer_matched/
    phase3_v4_cancer.pt
    cv_results.json
    bias_diagnostic_cancer_matched.csv
    bias_diagnostic_cancer_matched_summary.json
    neural_fold{0..4}.pt
    oof_predictions.npz
    run.log
experiments/multi_enzyme/outputs/v4_cds_unbiased/
    phase3_v4_cds.pt
    cv_results.json
    bias_diagnostic_cds_unbiased.csv
    bias_diagnostic_cds_unbiased_summary.json
    neural_fold{0..4}.pt
    oof_predictions.npz
    run.log
```

### Stage 3 — APOBEC1 head retraining (~30 min)

Drivers:
- `scripts/multi_enzyme/build_apobec1_v4_datasets.py --version all`
- `scripts/multi_enzyme/compute_apobec1_v4_features.py --version all --stage all`
- `experiments/multi_enzyme/exp_train_apobec1_head_v4.py --variant {cancer,cds}`

Produces:

```
experiments/multi_enzyme/outputs/apobec1_head_v4_cancer/apobec1_head_v4_cancer.pt
experiments/multi_enzyme/outputs/apobec1_head_v4_cds/apobec1_head_v4_cds.pt
```

### Stage 4 — Panel scoring (GPU required)

**GCP path (default, ~15 min on V100):**

The script:
1. Starts `ai-gpu2` (`gcloud compute instances start ai-gpu2 --zone=us-central1-a`).
2. Uploads the v4 checkpoints and `score_panel.py`.
3. Runs `score_panel.py` four times remotely:
   - Phase3 v4 cds + APOBEC1 v3
   - Phase3 v4 cancer + APOBEC1 v3
   - Phase3 v4 cds + APOBEC1 v4 cds (retrained)
   - Phase3 v4 cds + APOBEC1 v4 cancer (retrained)
4. Downloads the four parquets back to `experiments/multi_enzyme/outputs/pcawg_tcw_panel/v4_outputs/`.
5. Calls `scripts/multi_enzyme/merge_apobec1_v4_into_panel.py` to produce the final
   `*_apobec1retrained.parquet` files.
6. Stops the VM.

**Local-MPS path (`--local-mps`, ~24 h on M2 Pro):**

Same logic, but `score_panel.py` runs locally. Requires the panel caches mirrored
to `~/data/panel/` on the workstation. Note: PyTorch MPS does not currently
support all attention ops; expect occasional fallbacks.

Produces:

```
experiments/multi_enzyme/outputs/pcawg_tcw_panel/v4_outputs/
    panel_scores_v4_cds.parquet
    panel_scores_v4_cancer.parquet
    panel_scores_v4_cds_apobec1retrained.parquet
    panel_scores_v4_cancer_apobec1retrained.parquet
```

### Stage 5 — Fair sweep / topX / per-cancer / POG570 (~30 min)

Drivers:
- `scripts/gcp_panel/compute_panel_recall_sweep_fair_v4.py` — 21-construction sweep × 6 heads × 10 cancers.
- `scripts/gcp_panel/compute_panel_recall_topx_v4.py` — top-1/5/10% × P-threshold × heads.
- `scripts/gcp_panel/per_cancer_enrichment_v4.py` — Fisher OR, 2×2, advisor v2 style.
- `scripts/gcp_panel/analysis_D_pog570_validation_v4.py` — POG570 cohort replication.

Produces (under `v4_outputs/`):

```
fair_v4_cds.csv, fair_v4_cds_per_cancer.csv, fair_v4_cds.png, fair_v4_cds_RESULTS.md
fair_v4_cancer.csv, ...
topx_v4_cds.csv, ...
per_cancer_enrichment_v4_pcawg.csv
per_cancer_enrichment_v4_pog570.csv
per_cancer_OR_pcawg_top1pct.png
per_cancer_OR_pog570_top1pct.png
per_cancer_OR_concordance_top1pct.png
PER_CANCER_ENRICHMENT_V4.md
analysis_D_pog570_validation_v4_results.* (CSVs/JSON)
```

### Stage 6 — QA verification (~30 min)

Runs `qa_verification/check{1..4}_*.py` if present:
- `check1_make_shuffle.py` — generate shuffled control panel.
- `check1_quick_shuffle.py` — verify scores → null without permutations.
- `check2_check4.py` — independent re-derivation of top-1% recall.
- `check3_overlap.py` — verify v3 / v4 panel position overlap.
- `check4_recompute.py` — re-run main metrics with full nulls.

Each is run with `|| log "WARN ..."` (non-fatal) so a single check failure does
not abort the pipeline. Outputs (logs, JSONs) land alongside the scripts.

### Stage 7 — HTML report

Driver: `scripts/multi_enzyme/generate_v4_html_report.py` (TODO; pending task #19).

Until that script exists, Stage 7 prints a warning and exits 0. Once created,
re-run with `--skip-*` flags for stages 1–6 to regenerate just the report.

## Expected Wall-Clock

| Stage | Apple M2 Pro | GCP n2-standard + V100 | Notes |
|-------|--------------|------------------------|-------|
| 0 prereq | <5 s | <5 s | |
| 1 prep | 60–120 min | 60–120 min | dominated by ViennaRNA folding (CPU only) |
| 2 train | ~15 min (CPU) | ~5 min (V100) | 5-fold CV per variant × 2 variants |
| 3 apobec1 | ~30 min | ~10 min | feature compute + 5-fold CV |
| 4 score | ~24 h (MPS) | ~15 min | 95M candidates × 7 heads |
| 5 sweep | ~30 min | ~30 min | macOS-safe multiprocessing, 8 workers |
| 6 verify | ~30 min | ~30 min | mostly CPU pandas |
| 7 report | <2 min | <2 min | once script exists |
| **Total** | **~26 h** | **~3 h** | |

## Troubleshooting

### `conda env 'quris' not found`
Run `conda create -n quris python=3.11 && pip install -r requirements.txt`.

### `RNA-FM cache miss` (Stage 1 or 3)
`compute_rnafm_embeddings_v4.py` (called inside `build_v4_datasets.py`) needs
`fm.pretrained_models.rnafm`. If you see HuggingFace errors, run:
```python
import fm
fm.pretrained.rna_fm_t12()  # downloads ~700 MB to ~/.cache/torch/hub
```

### V100 quota exceeded
Request quota in GCP console for `NVIDIA_V100_GPUS` in `us-central1`. Or use
`--local-mps`. Typical quota approval: <1 h.

### Mac MPS OOM during Stage 4
PyTorch MPS allocator can OOM on 95M-row scoring. Reduce `--batch` from 4096 to 1024:
edit Stage 4 in `replicate_v4_pipeline.sh`. The local-MPS path also needs ~24 GB
free RAM (caches are mmap'd).

### `merge_apobec1_v4_into_panel.py` fails with "row count mismatch"
The v4_cds and v4_cancer panels and the `_retrain_raw/scored_v4cds_apo1*.parquet`
files must come from the **same candidates parquet**. If you re-enumerated
candidates between runs, redo Stage 4 from step 4d onward.

### Stage 5 `compute_panel_recall_sweep_fair_v4.py` hangs
MacOS multiprocessing requires fork to be safe. The script uses
`concurrent.futures.ProcessPoolExecutor` (spawn). If hangs persist, reduce
`--n-workers` to 4 or 2.

### POG570 file missing
Download from: https://www.bcgsc.ca/downloads/POG570/POG570_small_mutations.txt.gz
Place at `data/raw/pog570/POG570_small_mutations.txt.gz`.

### Stage 6 QA scripts fail with "/tmp/v4_shuffle_test.parquet not found"
Some QA scripts depend on intermediate files that `check1_make_shuffle.py`
creates. Run them in numeric order (the orchestrator does this).

## Master Log

The pipeline appends to:

```
experiments/multi_enzyme/outputs/v4_pipeline_replication.log
```

Per-stage timings, the exact commands invoked, and stdout/stderr from every
script land here. On failure, this log is the first place to check.

## Re-running Single Stages

To rebuild only the per-cancer enrichment plots after fixing a bug:

```bash
bash scripts/replicate_v4_pipeline.sh \
    --skip-prep --skip-train --skip-apobec1 --skip-score --skip-verify --skip-report
```

To regenerate only the HTML report:

```bash
bash scripts/replicate_v4_pipeline.sh \
    --skip-prep --skip-train --skip-apobec1 --skip-score --skip-sweep --skip-verify
```

## Known TODOs

- **`generate_v4_html_report.py`** does not yet exist (task #19 pending). Stage 7
  is a no-op warning until it is added.
- The local-MPS Stage 4 path leaves the retrained-APOBEC1 merge in a
  best-effort state — it expects `_retrain_raw/scored_v4cds_apo1{cancer,cds}.parquet`
  which the local path does not currently produce automatically. For a fully
  reproducible local-MPS run, manually invoke `score_panel.py` twice with
  the v4 APOBEC1 heads (see Stage 4 GCP commands as templates) before the
  merge step.
- ViennaRNA structure folding (~80% of Stage 1 wall time) is single-threaded.
  A parallel wrapper would shave the local pipeline by 30–60 minutes.
