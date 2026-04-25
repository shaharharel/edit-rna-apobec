#!/bin/bash
# Run Phase 1.5: same panel, finer window (250 bp) + max aggregator + v2 §1c reproduction.
# Acceptance criteria: complete in <= 1 hour wall-clock (estimate ~10 min, since panel
# scoring is reused).
#
# Usage:
#   bash scripts/gcp_panel/run_phase1_5_analyses.sh [panel.parquet]

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PANEL="${1:-$PROJECT_ROOT/experiments/multi_enzyme/outputs/pcawg_tcw_panel/panel_scores_cds.parquet}"
OUT_DIR="$PROJECT_ROOT/experiments/multi_enzyme/outputs/pcawg_tcw_panel"
N_WORKERS="${N_WORKERS:-8}"
PERM_REPS="${PERM_REPS:-10000}"
DECILE_PERM="${DECILE_PERM:-2000}"
EXPLORATORY_PERM="${EXPLORATORY_PERM:-1000}"
WINDOW_SIZE="${WINDOW_SIZE:-250}"
AGGREGATOR="${AGGREGATOR:-max}"
SUFFIX="${SUFFIX:-_phase1_5}"

if [ ! -f "$PANEL" ]; then
  echo "ERROR: panel scores not found at $PANEL"
  exit 1
fi

if [ ! -f "$OUT_DIR/PHASE_1_DONE.flag" ]; then
  echo "WARNING: PHASE_1_DONE.flag missing. Phase 1.5 should run AFTER Phase 1."
fi

echo "[$(date +%H:%M:%S)] Phase 1.5 starting on $PANEL"
echo "[$(date +%H:%M:%S)] window=$WINDOW_SIZE bp, aggregator=$AGGREGATOR, suffix=$SUFFIX"
echo "[$(date +%H:%M:%S)] N_WORKERS=$N_WORKERS, PERM_REPS=$PERM_REPS"

START_TS=$(date +%s)

(conda run -n quris python3 "$PROJECT_ROOT/scripts/gcp_panel/analysis_A_pcawg_wgs.py" \
    --panel "$PANEL" \
    --out-dir "$OUT_DIR/analysis_A_pcawg_wgs" \
    --n-workers "$N_WORKERS" --perm-reps "$PERM_REPS" \
    --decile-perm "$DECILE_PERM" --exploratory-perm "$EXPLORATORY_PERM" \
    --window-size "$WINDOW_SIZE" --aggregator "$AGGREGATOR" \
    --output-suffix "$SUFFIX" \
    > "$OUT_DIR/analysis_A_phase1_5_run.log" 2>&1) &
A_PID=$!
echo "[$(date +%H:%M:%S)] Analysis A (1.5) pid=$A_PID launched"

(conda run -n quris python3 "$PROJECT_ROOT/scripts/gcp_panel/analysis_B_tcga_pcawg_coding.py" \
    --panel "$PANEL" \
    --out-dir "$OUT_DIR/analysis_B_coding_panel" \
    --n-workers "$N_WORKERS" --perm-reps "$PERM_REPS" \
    --decile-perm "$DECILE_PERM" --exploratory-perm "$EXPLORATORY_PERM" \
    --window-size "$WINDOW_SIZE" --aggregator "$AGGREGATOR" \
    --output-suffix "$SUFFIX" \
    > "$OUT_DIR/analysis_B_phase1_5_run.log" 2>&1) &
B_PID=$!
echo "[$(date +%H:%M:%S)] Analysis B (1.5) pid=$B_PID launched"

(conda run -n quris python3 "$PROJECT_ROOT/scripts/gcp_panel/analysis_C_v2_section1c_repro.py" \
    --out-dir "$OUT_DIR/analysis_C_v2_section1c_repro" \
    > "$OUT_DIR/analysis_C_run.log" 2>&1) &
C_PID=$!
echo "[$(date +%H:%M:%S)] Analysis C (v2 §1c repro) pid=$C_PID launched"

wait "$A_PID"
A_RC=$?
echo "[$(date +%H:%M:%S)] Analysis A (1.5) exit code: $A_RC"

wait "$B_PID"
B_RC=$?
echo "[$(date +%H:%M:%S)] Analysis B (1.5) exit code: $B_RC"

wait "$C_PID"
C_RC=$?
echo "[$(date +%H:%M:%S)] Analysis C exit code: $C_RC"

if [ "$A_RC" -ne 0 ] || [ "$B_RC" -ne 0 ] || [ "$C_RC" -ne 0 ]; then
  echo "ERROR: A=$A_RC B=$B_RC C=$C_RC; see logs in $OUT_DIR"
  exit 1
fi

echo "[$(date +%H:%M:%S)] Running compare_phase1_vs_phase1_5.py ..."
conda run -n quris python3 "$PROJECT_ROOT/scripts/gcp_panel/compare_phase1_vs_phase1_5.py" \
    > "$OUT_DIR/compare_phase1_vs_1_5_run.log" 2>&1
CMP_RC=$?
echo "[$(date +%H:%M:%S)] compare exit: $CMP_RC"

if [ "$CMP_RC" -eq 0 ] && [ -f "$OUT_DIR/PHASE_1_5_DONE.flag" ]; then
  END_TS=$(date +%s)
  ELAPSED=$((END_TS - START_TS))
  echo "[$(date +%H:%M:%S)] Phase 1.5 COMPLETE. wall-clock=${ELAPSED}s"
  echo "Outputs: $OUT_DIR/{COMPARISON_PHASE1_VS_1_5.md,PHASE_1_5_DONE.flag}"
  exit 0
else
  echo "ERROR: compare failed or PHASE_1_5_DONE.flag missing"
  exit 1
fi
