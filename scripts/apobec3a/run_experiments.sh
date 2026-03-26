#!/usr/bin/env bash
# APOBEC3A RNA Editing Analysis — full experiment pipeline.
#
# Runs all experiments in `experiments/apobec3a/` in the correct dependency order
# and produces the final HTML report at:
#   experiments/apobec3a/outputs/v3_report.html
#
# PREREQUISITE: Phase 1 data pipeline must be complete (data_generation.md stages 1-7).
# Specifically, these files must exist before this script runs:
#   data/processed/splits_expanded_a3a.csv
#   data/processed/site_sequences.json
#   data/processed/embeddings/vienna_structure_cache.npz
#   data/processed/embeddings/rnafm_pooled.pt
#   data/processed/embeddings/rnafm_pooled_edited.pt
#
# CRITICAL dependency: exp_loop_position_analysis MUST run before exp_classification_a3a_5fold.
# If loop_position_per_site.csv is missing when the classifier runs, all loop geometry
# features (is_unpaired, relative_loop_position, etc.) are silent zero-vectors with no warning.
# This script enforces the correct order and skips already-completed steps.
#
# Usage:
#   bash scripts/apobec3a/run_experiments.sh             # all experiments (~6-8h total)
#   bash scripts/apobec3a/run_experiments.sh --skip-slow # skip ClinVar + neural (~20 min)
#   bash scripts/apobec3a/run_experiments.sh --only-gb   # GB/structure models only (<10 min)

set -euo pipefail
cd "$(dirname "$0")/../.."

PYTHON="conda run -n quris python"
OUT="experiments/apobec3a/outputs"

SKIP_SLOW=false
ONLY_GB=false
for arg in "$@"; do
    case $arg in
        --skip-slow) SKIP_SLOW=true ;;
        --only-gb)   ONLY_GB=true; SKIP_SLOW=true ;;
    esac
done

log() { echo "[$(date '+%H:%M:%S')] $*"; }
check_file() {
    if [ ! -f "$1" ]; then
        echo "ERROR: required file missing: $1"
        echo "Run the data pipeline (data_generation.md) first."
        exit 1
    fi
}

# -----------------------------------------------------------------------
# Validate data pipeline outputs exist before starting
# -----------------------------------------------------------------------
log "Checking data pipeline outputs..."
check_file "data/processed/splits_expanded_a3a.csv"
check_file "data/processed/site_sequences.json"
check_file "data/processed/embeddings/vienna_structure_cache.npz"
check_file "data/processed/embeddings/rnafm_pooled.pt"
check_file "data/processed/embeddings/rnafm_pooled_edited.pt"
log "Data pipeline outputs OK."

# -----------------------------------------------------------------------
# Phase 1: Loop position (MUST run before classification and rate models)
# -----------------------------------------------------------------------
# Produces: experiments/apobec3a/outputs/loop_position/loop_position_per_site.csv
# Used by:  classification_a3a_5fold (GB hand features), rate_5fold_zscore, rate_feature_importance
LOOP_CSV="$OUT/loop_position/loop_position_per_site.csv"
if [ ! -f "$LOOP_CSV" ]; then
    log "=== Phase 1: Loop position analysis ==="
    $PYTHON experiments/apobec3a/exp_loop_position_analysis.py
    log "loop_position done → $LOOP_CSV"
else
    log "loop_position: already done ($LOOP_CSV)"
fi

# -----------------------------------------------------------------------
# Phase 2: Core classification and rate experiments
#          (both require loop_position_per_site.csv)
# -----------------------------------------------------------------------
log "=== Phase 2: Core classification and rate models ==="

CLS_JSON="$OUT/classification_a3a_5fold/classification_a3a_5fold_results.json"
if [ ! -f "$CLS_JSON" ]; then
    log "  classification_a3a_5fold..."
    $PYTHON experiments/apobec3a/exp_classification_a3a_5fold.py
    log "  classification done → $CLS_JSON"
else
    log "  classification_a3a_5fold: already done"
fi

if [ "$ONLY_GB" = false ]; then
    RATE_JSON="$OUT/rate_5fold_zscore/rate_5fold_results.json"
    if [ ! -f "$RATE_JSON" ]; then
        log "  rate_5fold_zscore..."
        $PYTHON experiments/apobec3a/exp_rate_5fold_zscore.py
        log "  rate done → $RATE_JSON"
    else
        log "  rate_5fold_zscore: already done"
    fi
fi

# -----------------------------------------------------------------------
# Phase 3: Analysis experiments (independent, safe to run in any order)
# -----------------------------------------------------------------------
log "=== Phase 3: Analysis experiments ==="

run_if_missing() {
    local script="$1"
    local output="$2"
    local label="$3"
    if [ ! -f "$output" ]; then
        log "  $label..."
        $PYTHON "$script"
        log "  $label done"
    else
        log "  $label: already done"
    fi
}

run_if_missing \
    "experiments/apobec3a/exp_structure_analysis.py" \
    "$OUT/structure_analysis/structure_analysis.json" \
    "structure_analysis"

run_if_missing \
    "experiments/apobec3a/exp_motif_analysis.py" \
    "$OUT/motif_analysis/motif_analysis_results.json" \
    "motif_analysis"

run_if_missing \
    "experiments/apobec3a/exp_cross_dataset_full.py" \
    "$OUT/cross_dataset_full/cross_dataset_full_results.json" \
    "cross_dataset_full"

run_if_missing \
    "experiments/apobec3a/exp_rnasee_comparison.py" \
    "$OUT/rnasee_comparison/rnasee_comparison_results.json" \
    "rnasee_comparison"

run_if_missing \
    "experiments/apobec3a/exp_embedding_viz_v2.py" \
    "$OUT/embedding_viz_v2/embedding_viz_results.json" \
    "embedding_viz_v2"

run_if_missing \
    "experiments/apobec3a/exp_rate_feature_importance.py" \
    "$OUT/rate_feature_importance/results.json" \
    "rate_feature_importance"

run_if_missing \
    "experiments/apobec3a/exp_rate_per_dataset.py" \
    "$OUT/rate_per_dataset/rate_per_dataset_results.json" \
    "rate_per_dataset"

run_if_missing \
    "experiments/apobec3a/exp_tc_motif_reanalysis.py" \
    "$OUT/tc_motif_reanalysis/tc_motif_reanalysis_results.json" \
    "tc_motif_reanalysis"

run_if_missing \
    "experiments/apobec3a/exp_dataset_deep_analysis.py" \
    "$OUT/dataset_deep_analysis/dataset_statistics.json" \
    "dataset_deep_analysis"

if [ "$ONLY_GB" = false ]; then
    run_if_missing \
        "experiments/apobec3a/exp_a3a_filtered.py" \
        "$OUT/a3a_filtered/a3a_filtered_results.json" \
        "a3a_filtered"
fi

# -----------------------------------------------------------------------
# Phase 4: ClinVar analysis (slow: ~4.5 hours)
# -----------------------------------------------------------------------
if [ "$SKIP_SLOW" = false ]; then
    log "=== Phase 4: ClinVar prediction (~4.5 hours) ==="
    CLINVAR_CSV="$OUT/clinvar_prediction/clinvar_all_scores.csv"
    if [ ! -f "$CLINVAR_CSV" ]; then
        check_file "data/processed/clinvar_c2u_variants.csv"
        log "  Starting clinvar_prediction (N_WORKERS=12, ~4.5h)..."
        $PYTHON experiments/apobec3a/exp_clinvar_prediction.py
        log "  clinvar_prediction done"
    else
        log "  clinvar_prediction: already done ($CLINVAR_CSV)"
    fi

    CALIB_JSON="$OUT/clinvar_calibrated/calibrated_enrichment_results.json"
    if [ ! -f "$CALIB_JSON" ]; then
        log "  clinvar_calibrated..."
        $PYTHON experiments/apobec3a/exp_clinvar_calibrated.py
        log "  clinvar_calibrated done"
    else
        log "  clinvar_calibrated: already done"
    fi

    RNASEE_JSON="$OUT/rnasee_cds/rnasee_cds_results.json"
    if [ ! -f "$RNASEE_JSON" ]; then
        log "  replicate_rnasee_cds..."
        $PYTHON experiments/apobec3a/replicate_rnasee_cds.py
        log "  rnasee_cds done"
    else
        log "  rnasee_cds: already done"
    fi
fi

# -----------------------------------------------------------------------
# Phase 5: Final HTML report
# -----------------------------------------------------------------------
log "=== Phase 5: Generating HTML report ==="
$PYTHON experiments/apobec3a/generate_html_report.py
REPORT_SIZE=$(wc -c < "$OUT/v3_report.html" | tr -d ' ')
log "HTML report: $OUT/v3_report.html ($REPORT_SIZE bytes)"

log ""
log "=== ALL EXPERIMENTS COMPLETE ==="
log "Report: $OUT/v3_report.html"
