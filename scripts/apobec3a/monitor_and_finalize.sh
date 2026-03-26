#!/usr/bin/env bash
# Monitor running experiments and run finalization steps when complete.
# Usage: bash scripts/apobec3a/monitor_and_finalize.sh

set -euo pipefail
cd "$(dirname "$0")/../.."

A3A_FILTERED_OUTPUT="experiments/apobec3a/outputs/a3a_filtered/a3a_filtered_results.json"
CLINVAR_ALL_SCORES="experiments/apobec3a/outputs/clinvar_prediction/clinvar_all_scores.csv"
CLINVAR_RESULTS="experiments/apobec3a/outputs/clinvar_prediction/clinvar_prediction_results.json"
CALIBRATED_RESULTS="experiments/apobec3a/outputs/clinvar_calibrated/calibrated_enrichment_results.json"
RNASEE_RESULTS="experiments/apobec3a/outputs/rnasee_cds/rnasee_cds_results.json"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=== monitor_and_finalize.sh started ==="
log "Watching for:"
log "  a3a_filtered: $A3A_FILTERED_OUTPUT"
log "  clinvar_all_scores: $CLINVAR_ALL_SCORES"

# ------------------------------------------------------------------
# Wait for a3a_filtered if not done
# ------------------------------------------------------------------
if [ ! -f "$A3A_FILTERED_OUTPUT" ]; then
    log "Waiting for a3a_filtered experiment to complete..."
    while [ ! -f "$A3A_FILTERED_OUTPUT" ]; do
        sleep 60
        CHUNKS=$(ls experiments/apobec3a/outputs/clinvar_prediction/intermediate/ 2>/dev/null | wc -l | tr -d ' ')
        A3A_RUNNING=$(ps aux | grep "exp_a3a_filtered" | grep -v grep | wc -l | tr -d ' ')
        log "  ClinVar chunks done: $CHUNKS/34  |  a3a_filtered running: $A3A_RUNNING"
    done
    log "a3a_filtered COMPLETE. Output: $A3A_FILTERED_OUTPUT"
else
    log "a3a_filtered already complete."
fi

# Regenerate HTML report with a3a_filtered section
log "Regenerating HTML report (with a3a_filtered)..."
conda run -n quris python experiments/apobec3a/generate_html_report.py
log "HTML report updated."

# ------------------------------------------------------------------
# Wait for ClinVar to complete
# ------------------------------------------------------------------
if [ ! -f "$CLINVAR_ALL_SCORES" ]; then
    log "Waiting for ClinVar prediction to complete..."
    CLINVAR_PID=$(ps aux | grep "exp_clinvar_prediction" | grep -v grep | awk '{print $2}' | head -1)
    if [ -n "$CLINVAR_PID" ]; then
        log "ClinVar main PID: $CLINVAR_PID"
        while kill -0 "$CLINVAR_PID" 2>/dev/null; do
            CHUNKS=$(ls experiments/apobec3a/outputs/clinvar_prediction/intermediate/ 2>/dev/null | wc -l | tr -d ' ')
            log "  ClinVar progress: $CHUNKS/34 chunks done"
            sleep 120
        done
        log "ClinVar process $CLINVAR_PID has exited."
    else
        log "ClinVar process not found - waiting for output file..."
        while [ ! -f "$CLINVAR_ALL_SCORES" ]; do
            CHUNKS=$(ls experiments/apobec3a/outputs/clinvar_prediction/intermediate/ 2>/dev/null | wc -l | tr -d ' ')
            log "  ClinVar progress: $CHUNKS/34 chunks done"
            sleep 120
        done
    fi
    # Small wait to ensure file write is complete
    sleep 5
fi

if [ ! -f "$CLINVAR_ALL_SCORES" ]; then
    log "ERROR: ClinVar all_scores.csv not found after waiting. Check exp_clinvar_prediction.py output."
    exit 1
fi
log "ClinVar prediction COMPLETE."

# ------------------------------------------------------------------
# Run calibration experiment
# ------------------------------------------------------------------
if [ ! -f "$CALIBRATED_RESULTS" ]; then
    log "Running exp_clinvar_calibrated.py..."
    conda run -n quris python experiments/apobec3a/exp_clinvar_calibrated.py
    log "Calibration complete."
else
    log "Calibration results already exist."
fi

# ------------------------------------------------------------------
# Run RNAsee CDS comparison
# ------------------------------------------------------------------
if [ ! -f "$RNASEE_RESULTS" ]; then
    log "Running replicate_rnasee_cds.py..."
    conda run -n quris python experiments/apobec3a/replicate_rnasee_cds.py
    log "RNAsee CDS comparison complete."
else
    log "RNAsee CDS results already exist."
fi

# ------------------------------------------------------------------
# Final HTML report generation
# ------------------------------------------------------------------
log "Generating final HTML report (with all sections)..."
conda run -n quris python experiments/apobec3a/generate_html_report.py
REPORT_SIZE=$(wc -c < experiments/apobec3a/outputs/v3_report.html | tr -d ' ')
log "Final HTML report generated: $REPORT_SIZE bytes"

log "=== ALL EXPERIMENTS COMPLETE ==="
log "Report: experiments/apobec3a/outputs/v3_report.html"
