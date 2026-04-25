#!/bin/bash
# Phase 1.6 sweep: window size in {100,250,500,1000,2000} bp, max-pool, 10K perms.
# 5 windows x 2 analyses (A, B) = 10 jobs; throttle to 4 concurrent to fit Mac cores.
#
# Usage:
#   bash scripts/gcp_panel/run_phase1_6_sweep.sh [panel.parquet]

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PANEL="${1:-$PROJECT_ROOT/experiments/multi_enzyme/outputs/pcawg_tcw_panel/panel_scores_cds.parquet}"
OUT_DIR="$PROJECT_ROOT/experiments/multi_enzyme/outputs/pcawg_tcw_panel"
N_WORKERS="${N_WORKERS:-4}"   # per analysis (so 4 windows x 4 workers = 16 procs max)
PERM_REPS="${PERM_REPS:-10000}"
DECILE_PERM="${DECILE_PERM:-2000}"
EXPL_PERM="${EXPL_PERM:-1000}"
WINDOWS=("${WINDOWS_SET[@]:-100 250 500 1000 2000}")
WINDOWS_SET="${WINDOWS_SET:-100 250 500 1000 2000}"
MAX_CONCURRENT="${MAX_CONCURRENT:-4}"

if [ ! -f "$PANEL" ]; then
  echo "ERROR: panel scores not found at $PANEL"
  exit 1
fi

echo "[$(date +%H:%M:%S)] Phase 1.6 sweep starting on $PANEL"
echo "[$(date +%H:%M:%S)] windows=$WINDOWS_SET, agg=max, perm=$PERM_REPS, max_conc=$MAX_CONCURRENT"

START_TS=$(date +%s)
declare -a PIDS=()

run_one() {
    local script="$1"; local subdir="$2"; local win="$3"; local kind="$4"
    local suffix="_phase1_6_${kind}_w${win}"
    local logf="$OUT_DIR/run_phase1_6_${kind}_w${win}.log"
    conda run -n quris python3 "$PROJECT_ROOT/scripts/gcp_panel/${script}" \
        --panel "$PANEL" \
        --out-dir "$OUT_DIR/$subdir" \
        --n-workers "$N_WORKERS" --perm-reps "$PERM_REPS" \
        --decile-perm "$DECILE_PERM" --exploratory-perm "$EXPL_PERM" \
        --window-size "$win" --aggregator max \
        --output-suffix "$suffix" \
        > "$logf" 2>&1
    local rc=$?
    echo "[$(date +%H:%M:%S)] ${kind} w=${win} rc=${rc}"
    return $rc
}

# Build the (analysis, window) job list and run with concurrency cap
JOBS=()
for w in $WINDOWS_SET; do
  JOBS+=("A:$w")
  JOBS+=("B:$w")
done
echo "[$(date +%H:%M:%S)] Total jobs: ${#JOBS[@]}"

active_pids=()
for job in "${JOBS[@]}"; do
    kind="${job%%:*}"; win="${job##*:}"
    if [ "$kind" = "A" ]; then
        script="analysis_A_pcawg_wgs.py"; subdir="analysis_A_pcawg_wgs"
    else
        script="analysis_B_tcga_pcawg_coding.py"; subdir="analysis_B_coding_panel"
    fi
    while [ "${#active_pids[@]}" -ge "$MAX_CONCURRENT" ]; do
        # Wait for any to finish
        new_active=()
        for pid in "${active_pids[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                new_active+=("$pid")
            fi
        done
        active_pids=("${new_active[@]}")
        if [ "${#active_pids[@]}" -ge "$MAX_CONCURRENT" ]; then
            sleep 5
        fi
    done
    echo "[$(date +%H:%M:%S)] Launching ${kind} window=${win}"
    run_one "$script" "$subdir" "$win" "$kind" &
    active_pids+=($!)
    sleep 2
done

# Wait for all remaining
for pid in "${active_pids[@]}"; do
    wait "$pid" || echo "[$(date +%H:%M:%S)] pid=$pid exited non-zero"
done

END_TS=$(date +%s)
ELAPSED=$((END_TS - START_TS))
echo "[$(date +%H:%M:%S)] Sweep complete. wall-clock=${ELAPSED}s"
ls -la "$OUT_DIR/analysis_A_pcawg_wgs"/enrichment_primary_phase1_6_*.json 2>&1 | head -10
ls -la "$OUT_DIR/analysis_B_coding_panel"/enrichment_primary_phase1_6_*.json 2>&1 | head -10
