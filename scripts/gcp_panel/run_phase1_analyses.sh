#!/bin/bash
# Run Phase 1 analyses (A and B) in parallel + comparison + final report.
# Acceptance criteria: complete in <= 1 hour wall-clock.
#
# Usage:
#   bash scripts/gcp_panel/run_phase1_analyses.sh [panel.parquet]
#
# Default panel: experiments/multi_enzyme/outputs/pcawg_tcw_panel/panel_scores_cds.parquet

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PANEL="${1:-$PROJECT_ROOT/experiments/multi_enzyme/outputs/pcawg_tcw_panel/panel_scores_cds.parquet}"
OUT_DIR="$PROJECT_ROOT/experiments/multi_enzyme/outputs/pcawg_tcw_panel"
N_WORKERS="${N_WORKERS:-8}"
PERM_REPS="${PERM_REPS:-10000}"
DECILE_PERM="${DECILE_PERM:-2000}"
EXPLORATORY_PERM="${EXPLORATORY_PERM:-1000}"

if [ ! -f "$PANEL" ]; then
  echo "ERROR: panel scores not found at $PANEL"
  exit 1
fi

echo "[$(date +%H:%M:%S)] Starting Phase 1 analyses on $PANEL"
echo "[$(date +%H:%M:%S)] N_WORKERS=$N_WORKERS, PERM_REPS=$PERM_REPS"

START_TS=$(date +%s)

# Launch A and B in parallel as background subprocesses
(conda run -n quris python3 "$PROJECT_ROOT/scripts/gcp_panel/analysis_A_pcawg_wgs.py" \
    --panel "$PANEL" \
    --out-dir "$OUT_DIR/analysis_A_pcawg_wgs" \
    --n-workers "$N_WORKERS" --perm-reps "$PERM_REPS" \
    --decile-perm "$DECILE_PERM" --exploratory-perm "$EXPLORATORY_PERM" \
    > "$OUT_DIR/analysis_A_run.log" 2>&1) &
A_PID=$!
echo "[$(date +%H:%M:%S)] Analysis A pid=$A_PID launched"

(conda run -n quris python3 "$PROJECT_ROOT/scripts/gcp_panel/analysis_B_tcga_pcawg_coding.py" \
    --panel "$PANEL" \
    --out-dir "$OUT_DIR/analysis_B_coding_panel" \
    --n-workers "$N_WORKERS" --perm-reps "$PERM_REPS" \
    --decile-perm "$DECILE_PERM" --exploratory-perm "$EXPLORATORY_PERM" \
    > "$OUT_DIR/analysis_B_run.log" 2>&1) &
B_PID=$!
echo "[$(date +%H:%M:%S)] Analysis B pid=$B_PID launched"

wait "$A_PID"
A_RC=$?
echo "[$(date +%H:%M:%S)] Analysis A exit code: $A_RC"

wait "$B_PID"
B_RC=$?
echo "[$(date +%H:%M:%S)] Analysis B exit code: $B_RC"

if [ "$A_RC" -ne 0 ] || [ "$B_RC" -ne 0 ]; then
  echo "ERROR: Analysis A=$A_RC or B=$B_RC failed; see logs in $OUT_DIR"
  exit 1
fi

echo "[$(date +%H:%M:%S)] Running compare_A_B.py ..."
conda run -n quris python3 "$PROJECT_ROOT/scripts/gcp_panel/compare_A_B.py" \
    > "$OUT_DIR/compare_run.log" 2>&1
CMP_RC=$?
echo "[$(date +%H:%M:%S)] compare_A_B exit: $CMP_RC"

if [ "$CMP_RC" -eq 0 ] && [ -f "$OUT_DIR/PHASE_1_DONE.flag" ]; then
  END_TS=$(date +%s)
  ELAPSED=$((END_TS - START_TS))
  echo "[$(date +%H:%M:%S)] Phase 1 COMPLETE. wall-clock=${ELAPSED}s"
  echo "Outputs: $OUT_DIR/{COMPARISON_PHASE1.md,FINAL_REPORT_PHASE1.md,PHASE_1_DONE.flag}"
  exit 0
else
  echo "ERROR: compare_A_B failed or PHASE_1_DONE.flag missing"
  exit 1
fi
