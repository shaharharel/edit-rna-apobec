#!/bin/bash
# Run topx threshold sweep for both v4 panel variants.
# Sequential: cds first (more important), then cancer.

set -e
cd /Users/shaharharel/Documents/github/edit-rna-apobec

OUT_DIR=experiments/multi_enzyme/outputs/pcawg_tcw_panel/v4_outputs
MASTER=$OUT_DIR/topx_master.log

run_one() {
    local variant="$1"
    local panel="$OUT_DIR/panel_scores_v4_${variant}.parquet"
    local prefix="$OUT_DIR/topx_threshold_sweep_v4_${variant}"
    local logf="$OUT_DIR/topx_v4_${variant}_run.log"
    echo "[$(date +%H:%M:%S)] starting topx sweep: ${variant}" | tee -a $MASTER
    conda run -n quris --no-capture-output python -u \
        scripts/gcp_panel/compute_panel_recall_topx_v4.py \
        --panel "$panel" --out-prefix "$prefix" --n-workers 8 --perm-reps 10000 \
        > "$logf" 2>&1
    local rc=$?
    echo "[$(date +%H:%M:%S)] ${variant} rc=${rc}" | tee -a $MASTER
    return $rc
}

run_one cds; A_RC=$?
if [ $A_RC -eq 0 ]; then
    run_one cancer; B_RC=$?
else
    echo "ABORT: cds failed; not running cancer" | tee -a $MASTER
    exit $A_RC
fi

if [ $A_RC -eq 0 ] && [ $B_RC -eq 0 ]; then
    echo "[$(date +%H:%M:%S)] BOTH DONE" | tee -a $MASTER
fi
