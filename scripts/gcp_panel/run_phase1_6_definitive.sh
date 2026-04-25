#!/bin/bash
# Phase 1.6 definitive: 250 bp + max-pool + 100K perms + bootstrap CIs
# pre-registered primary endpoint at commit e790cd5.
#
# Single-window, A and B in parallel. ~5-7 hours wall-clock.

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PANEL="${1:-$PROJECT_ROOT/experiments/multi_enzyme/outputs/pcawg_tcw_panel/panel_scores_cds.parquet}"
OUT_DIR="$PROJECT_ROOT/experiments/multi_enzyme/outputs/pcawg_tcw_panel"
N_WORKERS="${N_WORKERS:-8}"
PERM_REPS="${PERM_REPS:-100000}"
WINDOW_SIZE="${WINDOW_SIZE:-250}"
SUFFIX="${SUFFIX:-_phase1_6_definitive}"

if [ ! -f "$PANEL" ]; then
  echo "ERROR: panel scores not found at $PANEL"
  exit 1
fi

echo "[$(date +%H:%M:%S)] Phase 1.6 definitive starting"
echo "[$(date +%H:%M:%S)] window=$WINDOW_SIZE bp, agg=max, perm=$PERM_REPS, suffix=$SUFFIX"

START_TS=$(date +%s)

(conda run -n quris python3 "$PROJECT_ROOT/scripts/gcp_panel/analysis_A_pcawg_wgs.py" \
    --panel "$PANEL" \
    --out-dir "$OUT_DIR/analysis_A_pcawg_wgs" \
    --n-workers "$N_WORKERS" --perm-reps "$PERM_REPS" \
    --decile-perm 2000 --exploratory-perm 1000 \
    --window-size "$WINDOW_SIZE" --aggregator max \
    --output-suffix "$SUFFIX" \
    > "$OUT_DIR/analysis_A_phase1_6_definitive_run.log" 2>&1) &
A_PID=$!
echo "[$(date +%H:%M:%S)] Analysis A pid=$A_PID"

(conda run -n quris python3 "$PROJECT_ROOT/scripts/gcp_panel/analysis_B_tcga_pcawg_coding.py" \
    --panel "$PANEL" \
    --out-dir "$OUT_DIR/analysis_B_coding_panel" \
    --n-workers "$N_WORKERS" --perm-reps "$PERM_REPS" \
    --decile-perm 2000 --exploratory-perm 1000 \
    --window-size "$WINDOW_SIZE" --aggregator max \
    --output-suffix "$SUFFIX" \
    > "$OUT_DIR/analysis_B_phase1_6_definitive_run.log" 2>&1) &
B_PID=$!
echo "[$(date +%H:%M:%S)] Analysis B pid=$B_PID"

wait "$A_PID"
A_RC=$?
echo "[$(date +%H:%M:%S)] A rc=$A_RC"
wait "$B_PID"
B_RC=$?
echo "[$(date +%H:%M:%S)] B rc=$B_RC"

END_TS=$(date +%s)
ELAPSED=$((END_TS - START_TS))
echo "[$(date +%H:%M:%S)] Definitive 100K-perm complete. wall-clock=${ELAPSED}s"

if [ "$A_RC" -eq 0 ] && [ "$B_RC" -eq 0 ]; then
  exit 0
else
  echo "ERROR: A=$A_RC B=$B_RC"
  exit 1
fi
