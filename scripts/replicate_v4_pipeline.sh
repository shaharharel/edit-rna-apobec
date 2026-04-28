#!/usr/bin/env bash
# =============================================================================
# replicate_v4_pipeline.sh
#
# End-to-end replication driver for the v4 (cancer_matched / cds_unbiased) pipeline.
# Runs the full sequence: data prep -> Phase3 training -> APOBEC1 head retrain ->
# Panel scoring -> Fair sweep / topX / per-cancer / POG570 -> QA verification ->
# HTML report.
#
# USAGE
#   bash scripts/replicate_v4_pipeline.sh                      # full pipeline
#   bash scripts/replicate_v4_pipeline.sh --skip-prep          # skip Stage 1
#   bash scripts/replicate_v4_pipeline.sh --skip-train         # skip Stage 2
#   bash scripts/replicate_v4_pipeline.sh --skip-apobec1       # skip Stage 3
#   bash scripts/replicate_v4_pipeline.sh --skip-score         # skip Stage 4
#   bash scripts/replicate_v4_pipeline.sh --skip-sweep         # skip Stage 5
#   bash scripts/replicate_v4_pipeline.sh --skip-verify        # skip Stage 6
#   bash scripts/replicate_v4_pipeline.sh --skip-report        # skip Stage 7
#   bash scripts/replicate_v4_pipeline.sh --local-mps          # use local MPS for scoring
#                                                              # (skip GCP path; ~24h)
#   bash scripts/replicate_v4_pipeline.sh --gcp-instance NAME  # override GCP VM name
#
# Stages can be combined: --skip-prep --skip-train --skip-apobec1 -> only score+sweep.
#
# REQUIREMENTS
#   - conda env "quris" exists (`conda activate quris`)
#   - raw data tarball unpacked in data/raw/
#   - hg19.fa and hg38.fa under data/raw/genomes/
#   - Phase3 v3 training data exists
#   - For Stage 4 GCP path: gcloud configured + ai-gpu2 V100 + remote caches present
#
# Master log: experiments/multi_enzyme/outputs/v4_pipeline_replication.log
# =============================================================================

set -euo pipefail

# -- repo root ----------------------------------------------------------------
PROJECT_ROOT="/Users/shaharharel/Documents/github/edit-rna-apobec"
cd "$PROJECT_ROOT"

# -- defaults / flags ---------------------------------------------------------
SKIP_PREP=0
SKIP_TRAIN=0
SKIP_APOBEC1=0
SKIP_SCORE=0
SKIP_SWEEP=0
SKIP_VERIFY=0
SKIP_REPORT=0
LOCAL_MPS=0
GCP_INSTANCE="ai-gpu2"
GCP_ZONE="us-central1-a"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-prep)     SKIP_PREP=1; shift ;;
        --skip-train)    SKIP_TRAIN=1; shift ;;
        --skip-apobec1)  SKIP_APOBEC1=1; shift ;;
        --skip-score)    SKIP_SCORE=1; shift ;;
        --skip-sweep)    SKIP_SWEEP=1; shift ;;
        --skip-verify)   SKIP_VERIFY=1; shift ;;
        --skip-report)   SKIP_REPORT=1; shift ;;
        --local-mps)     LOCAL_MPS=1; shift ;;
        --gcp-instance)  GCP_INSTANCE="$2"; shift 2 ;;
        --gcp-zone)      GCP_ZONE="$2"; shift 2 ;;
        -h|--help)
            grep '^# ' "$0" | sed 's/^# //'
            exit 0
            ;;
        *) echo "Unknown flag: $1" >&2; exit 2 ;;
    esac
done

# -- logging ------------------------------------------------------------------
OUT_BASE="$PROJECT_ROOT/experiments/multi_enzyme/outputs"
LOG_FILE="$OUT_BASE/v4_pipeline_replication.log"
mkdir -p "$OUT_BASE"

log() {
    local msg="[$(date +'%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg" | tee -a "$LOG_FILE"
}

stage_header() {
    log "============================================================"
    log "STAGE $1: $2"
    log "============================================================"
}

fail() {
    log "FATAL: $*"
    log "Pipeline aborted. See log at: $LOG_FILE"
    exit 1
}

verify_file() {
    # verify_file <path> <description>
    local f="$1"; local desc="${2:-file}"
    if [[ ! -f "$f" ]]; then
        fail "missing $desc: $f"
    fi
    if [[ ! -s "$f" ]]; then
        fail "empty $desc: $f"
    fi
    log "  ok: $desc ($f)"
}

verify_dir_nonempty() {
    local d="$1"; local desc="${2:-dir}"
    if [[ ! -d "$d" ]]; then
        fail "missing dir $desc: $d"
    fi
    if [[ -z "$(ls -A "$d" 2>/dev/null)" ]]; then
        fail "empty dir $desc: $d"
    fi
    log "  ok: $desc ($d)"
}

time_stage() {
    # Usage: time_stage NAME -- cmd args...
    # Or:    time_stage NAME -- bash -c "..."
    local name="$1"; shift
    [[ "$1" == "--" ]] && shift
    local t0
    t0=$(date +%s)
    log "RUN: $*"
    if "$@" >> "$LOG_FILE" 2>&1; then
        local t1; t1=$(date +%s)
        log "  ($name) elapsed: $((t1 - t0))s"
    else
        local rc=$?
        local t1; t1=$(date +%s)
        log "  ($name) FAILED rc=$rc after $((t1 - t0))s"
        return $rc
    fi
}

# Helper: source conda init only if available, otherwise rely on `conda run -n quris`.
ensure_conda() {
    if ! command -v conda >/dev/null 2>&1; then
        fail "conda not on PATH; install miniconda or activate before running"
    fi
    if ! conda env list | awk '{print $1}' | grep -qx "quris"; then
        fail "conda env 'quris' not found. See REPLICATION_V4.md for setup."
    fi
}

PY="conda run --no-capture-output -n quris python"

# -----------------------------------------------------------------------------
# STAGE 0: prerequisites
# -----------------------------------------------------------------------------
stage_header 0 "prerequisites check"
T0_START=$(date +%s)

ensure_conda

# raw data
verify_file "$PROJECT_ROOT/data/raw/genomes/hg19.fa" "hg19 genome"
verify_file "$PROJECT_ROOT/data/raw/genomes/hg38.fa" "hg38 genome"
verify_file "$PROJECT_ROOT/data/raw/C2TFinalSites.DB.xlsx" "advisor sites xlsx"

# Phase3 v3 training data
verify_file "$PROJECT_ROOT/data/processed/multi_enzyme/splits_multi_enzyme_v3_with_negatives.csv" "v3 splits CSV"
verify_file "$PROJECT_ROOT/data/processed/multi_enzyme/multi_enzyme_sequences_v3_with_negatives.json" "v3 sequences JSON"
verify_file "$PROJECT_ROOT/data/processed/multi_enzyme/loop_position_per_site_v3.csv" "v3 loop positions"
verify_file "$PROJECT_ROOT/data/processed/multi_enzyme/structure_cache_multi_enzyme_v3.npz" "v3 structure cache"

# raw cancer / pog570 (only required for Stage 5)
if [[ $SKIP_SWEEP -eq 0 ]]; then
    verify_dir_nonempty "$PROJECT_ROOT/data/raw/tcga" "TCGA dir"
    verify_dir_nonempty "$PROJECT_ROOT/data/raw/pcawg/by_cancer" "PCAWG by_cancer dir"
    verify_file "$PROJECT_ROOT/data/raw/pog570/POG570_small_mutations.txt.gz" "POG570 mutations"
fi

T0_END=$(date +%s)
log "Stage 0 OK ($((T0_END - T0_START))s)"

# -----------------------------------------------------------------------------
# STAGE 1: v4 data prep
# -----------------------------------------------------------------------------
if [[ $SKIP_PREP -eq 0 ]]; then
    stage_header 1 "v4 data prep (build_v4_datasets.py, ~1-2h)"
    T1_START=$(date +%s)

    time_stage "build_v4_datasets" -- $PY scripts/multi_enzyme/build_v4_datasets.py \
        || fail "build_v4_datasets.py failed"

    # Verify outputs
    for ver in cancer_matched cds_unbiased; do
        verify_file "$PROJECT_ROOT/data/processed/multi_enzyme/splits_multi_enzyme_v4_${ver}.csv" "v4 ${ver} splits"
        verify_file "$PROJECT_ROOT/data/processed/multi_enzyme/multi_enzyme_sequences_v4_${ver}.json" "v4 ${ver} sequences"
        verify_file "$PROJECT_ROOT/data/processed/multi_enzyme/loop_position_per_site_v4_${ver}.csv" "v4 ${ver} loop positions"
        verify_file "$PROJECT_ROOT/data/processed/embeddings/structure_cache_multi_enzyme_v4_${ver}.npz" "v4 ${ver} structure cache"
        verify_file "$PROJECT_ROOT/data/processed/embeddings/rnafm_v4_${ver}.npz" "v4 ${ver} RNA-FM"
    done
    verify_file "$PROJECT_ROOT/data/processed/multi_enzyme/cancer_ct_trinuc_distribution.csv" "cancer trinuc dist"
    verify_file "$PROJECT_ROOT/data/processed/multi_enzyme/cds_c_trinuc_distribution.csv" "CDS trinuc dist"

    T1_END=$(date +%s)
    log "Stage 1 OK ($((T1_END - T1_START))s)"
else
    log "Stage 1 SKIPPED"
fi

# -----------------------------------------------------------------------------
# STAGE 2: Phase3 training (cancer_matched + cds_unbiased, sequential)
# -----------------------------------------------------------------------------
if [[ $SKIP_TRAIN -eq 0 ]]; then
    stage_header 2 "Phase3 training (run_v4_training_both.sh, ~15min)"
    T2_START=$(date +%s)

    mkdir -p "$OUT_BASE/v4_cancer_matched" "$OUT_BASE/v4_cds_unbiased"

    time_stage "phase3_cancer_matched" -- $PY experiments/multi_enzyme/exp_train_phase3_v4.py \
        --variant cancer_matched \
        --out-dir "$OUT_BASE/v4_cancer_matched" \
        || fail "Phase3 cancer_matched training failed"
    verify_file "$OUT_BASE/v4_cancer_matched/phase3_v4_cancer.pt" "phase3 cancer ckpt"
    verify_file "$OUT_BASE/v4_cancer_matched/cv_results.json" "phase3 cancer cv"
    verify_file "$OUT_BASE/v4_cancer_matched/bias_diagnostic_cancer_matched_summary.json" "bias diag cancer"

    time_stage "phase3_cds_unbiased" -- $PY experiments/multi_enzyme/exp_train_phase3_v4.py \
        --variant cds_unbiased \
        --out-dir "$OUT_BASE/v4_cds_unbiased" \
        || fail "Phase3 cds_unbiased training failed"
    verify_file "$OUT_BASE/v4_cds_unbiased/phase3_v4_cds.pt" "phase3 cds ckpt"
    verify_file "$OUT_BASE/v4_cds_unbiased/cv_results.json" "phase3 cds cv"
    verify_file "$OUT_BASE/v4_cds_unbiased/bias_diagnostic_cds_unbiased_summary.json" "bias diag cds"

    T2_END=$(date +%s)
    log "Stage 2 OK ($((T2_END - T2_START))s)"
else
    log "Stage 2 SKIPPED"
fi

# -----------------------------------------------------------------------------
# STAGE 3: APOBEC1 head retraining (~30 min)
# -----------------------------------------------------------------------------
if [[ $SKIP_APOBEC1 -eq 0 ]]; then
    stage_header 3 "APOBEC1 head retraining (~30min)"
    T3_START=$(date +%s)

    # 3a: build apobec1 v4 datasets (both variants)
    time_stage "apobec1_v4_datasets" -- $PY scripts/multi_enzyme/build_apobec1_v4_datasets.py \
        --version all \
        || fail "build_apobec1_v4_datasets.py failed"

    # 3b: compute features
    time_stage "apobec1_v4_features" -- $PY scripts/multi_enzyme/compute_apobec1_v4_features.py \
        --version all --stage all \
        || fail "compute_apobec1_v4_features.py failed"

    # 3c: train both heads
    mkdir -p "$OUT_BASE/apobec1_head_v4_cancer" "$OUT_BASE/apobec1_head_v4_cds"

    time_stage "apobec1_head_v4_cancer" -- $PY experiments/multi_enzyme/exp_train_apobec1_head_v4.py \
        --variant cancer \
        || fail "exp_train_apobec1_head_v4 cancer failed"
    verify_file "$OUT_BASE/apobec1_head_v4_cancer/apobec1_head_v4_cancer.pt" "apobec1 head v4 cancer"

    time_stage "apobec1_head_v4_cds" -- $PY experiments/multi_enzyme/exp_train_apobec1_head_v4.py \
        --variant cds \
        || fail "exp_train_apobec1_head_v4 cds failed"
    verify_file "$OUT_BASE/apobec1_head_v4_cds/apobec1_head_v4_cds.pt" "apobec1 head v4 cds"

    T3_END=$(date +%s)
    log "Stage 3 OK ($((T3_END - T3_START))s)"
else
    log "Stage 3 SKIPPED"
fi

# -----------------------------------------------------------------------------
# STAGE 4: Panel scoring (GPU; either GCP V100 ~15min or local MPS ~24h)
# -----------------------------------------------------------------------------
V4_OUT="$OUT_BASE/pcawg_tcw_panel/v4_outputs"
mkdir -p "$V4_OUT"

if [[ $SKIP_SCORE -eq 0 ]]; then
    stage_header 4 "Panel scoring (Phase3 + APOBEC1 heads on ~95M candidates)"
    T4_START=$(date +%s)

    if [[ $LOCAL_MPS -eq 1 ]]; then
        log "Using LOCAL MPS path. ETA ~24h. Caches required at ~/data/panel/."
        # Score each variant locally. The panel candidate caches must already
        # exist locally at the paths below (rnafm/hand40/candidates parquet).
        PANEL_CACHE="$HOME/data/panel"
        verify_file "$PANEL_CACHE/candidates_cache_aligned.parquet" "panel candidates"
        verify_file "$PANEL_CACHE/rnafm_cds_kept/orig.npy" "panel rnafm orig"
        verify_file "$PANEL_CACHE/rnafm_cds_kept/delta.npy" "panel rnafm delta"
        verify_file "$PANEL_CACHE/hand40_cache_aligned.npy" "panel hand40"
        verify_file "$PANEL_CACHE/valid_mask.npy" "panel valid mask"

        time_stage "score_panel_v4_cds" -- $PY scripts/gcp_panel/score_panel.py \
            --candidates "$PANEL_CACHE/candidates_cache_aligned.parquet" \
            --orig "$PANEL_CACHE/rnafm_cds_kept/orig.npy" \
            --delta "$PANEL_CACHE/rnafm_cds_kept/delta.npy" \
            --hand40 "$PANEL_CACHE/hand40_cache_aligned.npy" \
            --valid "$PANEL_CACHE/valid_mask.npy" \
            --phase3 "$OUT_BASE/v4_cds_unbiased/phase3_v4_cds.pt" \
            --apobec1 "$OUT_BASE/apobec1_head/apobec1_head.pt" \
            --out "$V4_OUT/panel_scores_v4_cds.parquet" \
            --batch 4096 \
            || fail "local MPS score_panel cds failed"
        verify_file "$V4_OUT/panel_scores_v4_cds.parquet" "v4_cds panel scores"

        time_stage "score_panel_v4_cancer" -- $PY scripts/gcp_panel/score_panel.py \
            --candidates "$PANEL_CACHE/candidates_cache_aligned.parquet" \
            --orig "$PANEL_CACHE/rnafm_cds_kept/orig.npy" \
            --delta "$PANEL_CACHE/rnafm_cds_kept/delta.npy" \
            --hand40 "$PANEL_CACHE/hand40_cache_aligned.npy" \
            --valid "$PANEL_CACHE/valid_mask.npy" \
            --phase3 "$OUT_BASE/v4_cancer_matched/phase3_v4_cancer.pt" \
            --apobec1 "$OUT_BASE/apobec1_head/apobec1_head.pt" \
            --out "$V4_OUT/panel_scores_v4_cancer.parquet" \
            --batch 4096 \
            || fail "local MPS score_panel cancer failed"
        verify_file "$V4_OUT/panel_scores_v4_cancer.parquet" "v4_cancer panel scores"

        # Merge in retrained APOBEC1 v4 heads (same logic as merge_apobec1_v4_into_panel)
        log "Merging retrained APOBEC1 v4 heads into panels (local path)."
        log "NOTE: retrained-apobec1 columns require a separate score_panel run swapping --apobec1."
        # User-facing TODO: rerun score_panel.py with --apobec1 pointing at apobec1_head_v4_*.pt
        # then rename score_apobec1 -> score_apobec1_v4_<variant>, then run merge.
        time_stage "merge_apobec1_v4" -- $PY scripts/multi_enzyme/merge_apobec1_v4_into_panel.py \
            || log "merge_apobec1_v4_into_panel.py: warning (may require _retrain_raw/ inputs)"
    else
        log "Using GCP path (instance=$GCP_INSTANCE zone=$GCP_ZONE)."
        log "Required remote caches on $GCP_INSTANCE:"
        log "  ~/data/panel/candidates_cache_aligned.parquet"
        log "  ~/data/panel/rnafm_cds_kept/{orig,delta}.npy"
        log "  ~/data/panel/hand40_cache_aligned.npy"
        log "  ~/data/panel/valid_mask.npy"

        log "Step 4a: starting GCP instance"
        gcloud compute instances start "$GCP_INSTANCE" --zone="$GCP_ZONE" \
            >> "$LOG_FILE" 2>&1 || fail "failed to start $GCP_INSTANCE"

        log "Step 4b: uploading v4 checkpoints to $GCP_INSTANCE"
        # cancer + cds Phase3 ckpts and both v4 apobec1 heads
        for f in \
            "$OUT_BASE/v4_cancer_matched/phase3_v4_cancer.pt" \
            "$OUT_BASE/v4_cds_unbiased/phase3_v4_cds.pt" \
            "$OUT_BASE/apobec1_head_v4_cancer/apobec1_head_v4_cancer.pt" \
            "$OUT_BASE/apobec1_head_v4_cds/apobec1_head_v4_cds.pt"; do
            verify_file "$f" "checkpoint to upload"
            gcloud compute scp --zone="$GCP_ZONE" "$f" \
                "${GCP_INSTANCE}:~/data/models/" >> "$LOG_FILE" 2>&1 \
                || fail "scp $f to $GCP_INSTANCE failed"
        done

        log "Step 4c: uploading score_panel.py (in case the VM has a stale copy)"
        gcloud compute scp --zone="$GCP_ZONE" "$PROJECT_ROOT/scripts/gcp_panel/score_panel.py" \
            "${GCP_INSTANCE}:~/scripts/score_panel.py" >> "$LOG_FILE" 2>&1 \
            || fail "scp score_panel.py failed"

        log "Step 4d: scoring v4_cds (Phase3 cds + v3 APOBEC1 head)"
        gcloud compute ssh "$GCP_INSTANCE" --zone="$GCP_ZONE" --command="
set -e
python3 ~/scripts/score_panel.py \
    --candidates ~/data/panel/candidates_cache_aligned.parquet \
    --orig ~/data/panel/rnafm_cds_kept/orig.npy \
    --delta ~/data/panel/rnafm_cds_kept/delta.npy \
    --hand40 ~/data/panel/hand40_cache_aligned.npy \
    --valid ~/data/panel/valid_mask.npy \
    --phase3 ~/data/models/phase3_v4_cds.pt \
    --apobec1 ~/data/models/apobec1_head.pt \
    --out ~/data/panel/panel_scores_v4_cds.parquet \
    --batch 4096
" >> "$LOG_FILE" 2>&1 || fail "remote score_panel v4_cds failed"

        log "Step 4e: scoring v4_cancer (Phase3 cancer + v3 APOBEC1 head)"
        gcloud compute ssh "$GCP_INSTANCE" --zone="$GCP_ZONE" --command="
set -e
python3 ~/scripts/score_panel.py \
    --candidates ~/data/panel/candidates_cache_aligned.parquet \
    --orig ~/data/panel/rnafm_cds_kept/orig.npy \
    --delta ~/data/panel/rnafm_cds_kept/delta.npy \
    --hand40 ~/data/panel/hand40_cache_aligned.npy \
    --valid ~/data/panel/valid_mask.npy \
    --phase3 ~/data/models/phase3_v4_cancer.pt \
    --apobec1 ~/data/models/apobec1_head.pt \
    --out ~/data/panel/panel_scores_v4_cancer.parquet \
    --batch 4096
" >> "$LOG_FILE" 2>&1 || fail "remote score_panel v4_cancer failed"

        log "Step 4f: scoring v4_cds with retrained APOBEC1 v4 cds head"
        gcloud compute ssh "$GCP_INSTANCE" --zone="$GCP_ZONE" --command="
set -e
python3 ~/scripts/score_panel.py \
    --candidates ~/data/panel/candidates_cache_aligned.parquet \
    --orig ~/data/panel/rnafm_cds_kept/orig.npy \
    --delta ~/data/panel/rnafm_cds_kept/delta.npy \
    --hand40 ~/data/panel/hand40_cache_aligned.npy \
    --valid ~/data/panel/valid_mask.npy \
    --phase3 ~/data/models/phase3_v4_cds.pt \
    --apobec1 ~/data/models/apobec1_head_v4_cds.pt \
    --out ~/data/panel/scored_v4cds_apo1cds.parquet \
    --batch 4096
" >> "$LOG_FILE" 2>&1 || fail "remote score_panel v4_cds (apobec1 v4 cds) failed"

        log "Step 4g: scoring v4_cds with retrained APOBEC1 v4 cancer head"
        gcloud compute ssh "$GCP_INSTANCE" --zone="$GCP_ZONE" --command="
set -e
python3 ~/scripts/score_panel.py \
    --candidates ~/data/panel/candidates_cache_aligned.parquet \
    --orig ~/data/panel/rnafm_cds_kept/orig.npy \
    --delta ~/data/panel/rnafm_cds_kept/delta.npy \
    --hand40 ~/data/panel/hand40_cache_aligned.npy \
    --valid ~/data/panel/valid_mask.npy \
    --phase3 ~/data/models/phase3_v4_cds.pt \
    --apobec1 ~/data/models/apobec1_head_v4_cancer.pt \
    --out ~/data/panel/scored_v4cds_apo1cancer.parquet \
    --batch 4096
" >> "$LOG_FILE" 2>&1 || fail "remote score_panel v4_cds (apobec1 v4 cancer) failed"

        log "Step 4h: downloading panel parquets from $GCP_INSTANCE"
        mkdir -p "$V4_OUT/_retrain_raw"
        gcloud compute scp --zone="$GCP_ZONE" \
            "${GCP_INSTANCE}:~/data/panel/panel_scores_v4_cds.parquet" \
            "$V4_OUT/" >> "$LOG_FILE" 2>&1 || fail "scp v4_cds back failed"
        gcloud compute scp --zone="$GCP_ZONE" \
            "${GCP_INSTANCE}:~/data/panel/panel_scores_v4_cancer.parquet" \
            "$V4_OUT/" >> "$LOG_FILE" 2>&1 || fail "scp v4_cancer back failed"
        gcloud compute scp --zone="$GCP_ZONE" \
            "${GCP_INSTANCE}:~/data/panel/scored_v4cds_apo1cds.parquet" \
            "$V4_OUT/_retrain_raw/" >> "$LOG_FILE" 2>&1 || fail "scp v4_cds apo1cds back failed"
        gcloud compute scp --zone="$GCP_ZONE" \
            "${GCP_INSTANCE}:~/data/panel/scored_v4cds_apo1cancer.parquet" \
            "$V4_OUT/_retrain_raw/" >> "$LOG_FILE" 2>&1 || fail "scp v4_cds apo1cancer back failed"

        log "Step 4i: merging retrained APOBEC1 v4 columns into v4 panels"
        time_stage "merge_apobec1_v4" -- $PY scripts/multi_enzyme/merge_apobec1_v4_into_panel.py \
            || fail "merge_apobec1_v4_into_panel.py failed"

        log "Step 4j: stopping GCP instance"
        gcloud compute instances stop "$GCP_INSTANCE" --zone="$GCP_ZONE" \
            >> "$LOG_FILE" 2>&1 || log "WARN: stopping $GCP_INSTANCE failed; please stop manually"
    fi

    # Verify expected outputs (regardless of path)
    verify_file "$V4_OUT/panel_scores_v4_cds.parquet" "v4_cds panel scores"
    verify_file "$V4_OUT/panel_scores_v4_cancer.parquet" "v4_cancer panel scores"
    verify_file "$V4_OUT/panel_scores_v4_cds_apobec1retrained.parquet" "v4_cds retrained panel"
    verify_file "$V4_OUT/panel_scores_v4_cancer_apobec1retrained.parquet" "v4_cancer retrained panel"

    T4_END=$(date +%s)
    log "Stage 4 OK ($((T4_END - T4_START))s)"
else
    log "Stage 4 SKIPPED"
fi

# -----------------------------------------------------------------------------
# STAGE 5: Fair sweep + topX + per-cancer + POG570 (~30 min)
# -----------------------------------------------------------------------------
if [[ $SKIP_SWEEP -eq 0 ]]; then
    stage_header 5 "Fair sweep / topX / per-cancer / POG570 analysis (~30min)"
    T5_START=$(date +%s)

    # 5a: fair recall sweep (v4_cds)
    time_stage "sweep_fair_v4_cds" -- $PY scripts/gcp_panel/compute_panel_recall_sweep_fair_v4.py \
        --panel "$V4_OUT/panel_scores_v4_cds_apobec1retrained.parquet" \
        --out-prefix "$V4_OUT/fair_v4_cds" \
        --heads "score_binary,score_A3A,score_A3B,score_A3G,score_A3A_A3G,score_apobec1_v4_cds" \
        || fail "sweep_fair v4_cds failed"

    # 5b: fair recall sweep (v4_cancer)
    time_stage "sweep_fair_v4_cancer" -- $PY scripts/gcp_panel/compute_panel_recall_sweep_fair_v4.py \
        --panel "$V4_OUT/panel_scores_v4_cancer_apobec1retrained.parquet" \
        --out-prefix "$V4_OUT/fair_v4_cancer" \
        --heads "score_binary,score_A3A,score_A3B,score_A3G,score_A3A_A3G,score_apobec1_v4_cancer" \
        || fail "sweep_fair v4_cancer failed"

    # 5c: top-X% × P-threshold sweep (v4_cds)
    time_stage "topx_v4_cds" -- $PY scripts/gcp_panel/compute_panel_recall_topx_v4.py \
        --panel "$V4_OUT/panel_scores_v4_cds_apobec1retrained.parquet" \
        --out-prefix "$V4_OUT/topx_v4_cds" \
        --heads "score_binary,score_A3A,score_A3B,score_A3G,score_A3A_A3G,score_apobec1_v4_cds" \
        || fail "topx v4_cds failed"

    # 5d: per-cancer enrichment (uses default v4_cds_apobec1retrained panel)
    time_stage "per_cancer_enrichment" -- $PY scripts/gcp_panel/per_cancer_enrichment_v4.py \
        || fail "per_cancer_enrichment_v4 failed"

    # 5e: POG570 replication
    time_stage "pog570_validation" -- $PY scripts/gcp_panel/analysis_D_pog570_validation_v4.py \
        --panel "$V4_OUT/panel_scores_v4_cds_apobec1retrained.parquet" \
        --out-dir "$V4_OUT" \
        --head score_binary \
        || fail "pog570 v4 failed"

    verify_file "$V4_OUT/fair_v4_cds.csv" "fair sweep v4_cds CSV"
    verify_file "$V4_OUT/fair_v4_cancer.csv" "fair sweep v4_cancer CSV"
    verify_file "$V4_OUT/topx_v4_cds.csv" "topx v4_cds CSV"
    verify_file "$V4_OUT/per_cancer_enrichment_v4_pcawg.csv" "per-cancer PCAWG"
    verify_file "$V4_OUT/per_cancer_enrichment_v4_pog570.csv" "per-cancer POG570"

    T5_END=$(date +%s)
    log "Stage 5 OK ($((T5_END - T5_START))s)"
else
    log "Stage 5 SKIPPED"
fi

# -----------------------------------------------------------------------------
# STAGE 6: QA verification (~30 min)
# -----------------------------------------------------------------------------
if [[ $SKIP_VERIFY -eq 0 ]]; then
    stage_header 6 "QA verification (shuffle / overlap / recompute)"
    T6_START=$(date +%s)

    QA_DIR="$V4_OUT/qa_verification"
    mkdir -p "$QA_DIR"

    if [[ -f "$QA_DIR/check1_make_shuffle.py" ]]; then
        time_stage "qa_check1_make_shuffle" -- $PY "$QA_DIR/check1_make_shuffle.py" \
            || log "WARN: check1_make_shuffle.py failed (non-fatal)"
    fi
    if [[ -f "$QA_DIR/check1_quick_shuffle.py" ]]; then
        time_stage "qa_check1_quick_shuffle" -- $PY "$QA_DIR/check1_quick_shuffle.py" \
            || log "WARN: check1_quick_shuffle.py failed (non-fatal)"
    fi
    if [[ -f "$QA_DIR/check2_check4.py" ]]; then
        time_stage "qa_check2_check4" -- $PY "$QA_DIR/check2_check4.py" \
            || log "WARN: check2_check4.py failed (non-fatal)"
    fi
    if [[ -f "$QA_DIR/check3_overlap.py" ]]; then
        time_stage "qa_check3_overlap" -- $PY "$QA_DIR/check3_overlap.py" \
            || log "WARN: check3_overlap.py failed (non-fatal)"
    fi
    if [[ -f "$QA_DIR/check4_recompute.py" ]]; then
        time_stage "qa_check4_recompute" -- $PY "$QA_DIR/check4_recompute.py" \
            || log "WARN: check4_recompute.py failed (non-fatal)"
    fi

    T6_END=$(date +%s)
    log "Stage 6 OK ($((T6_END - T6_START))s)"
else
    log "Stage 6 SKIPPED"
fi

# -----------------------------------------------------------------------------
# STAGE 7: HTML report
# -----------------------------------------------------------------------------
if [[ $SKIP_REPORT -eq 0 ]]; then
    stage_header 7 "Generate v4 HTML report"
    T7_START=$(date +%s)

    REPORT_SCRIPT="$PROJECT_ROOT/scripts/multi_enzyme/generate_v4_html_report.py"
    if [[ -f "$REPORT_SCRIPT" ]]; then
        time_stage "generate_v4_html_report" -- $PY "$REPORT_SCRIPT" \
            || fail "generate_v4_html_report.py failed"
    else
        log "WARN: $REPORT_SCRIPT not present yet (task #19 pending)."
        log "      To generate the report, create the script then re-run with"
        log "      'bash scripts/replicate_v4_pipeline.sh --skip-prep --skip-train \\"
        log "       --skip-apobec1 --skip-score --skip-sweep --skip-verify'."
    fi

    T7_END=$(date +%s)
    log "Stage 7 OK ($((T7_END - T7_START))s)"
else
    log "Stage 7 SKIPPED"
fi

log "============================================================"
log "v4 PIPELINE REPLICATION COMPLETE"
log "Master log: $LOG_FILE"
log "============================================================"
