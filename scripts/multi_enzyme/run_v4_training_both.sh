#!/bin/bash
# Trains v4_cancer_matched then v4_cds_unbiased sequentially.
# Designed to run via Bash run_in_background so it survives agent lifecycle.

set -e
cd /Users/shaharharel/Documents/github/edit-rna-apobec

OUT_BASE=experiments/multi_enzyme/outputs
MASTER=$OUT_BASE/v4_training_master.log

echo "[$(date +%H:%M:%S)] v4 training: starting cancer_matched" | tee -a $MASTER
conda run -n quris python experiments/multi_enzyme/exp_train_phase3_v4.py \
    --variant cancer_matched \
    --out-dir $OUT_BASE/v4_cancer_matched \
    >> $OUT_BASE/v4_cancer_matched/run.log 2>&1
A_RC=$?
echo "[$(date +%H:%M:%S)] cancer_matched rc=$A_RC" | tee -a $MASTER

if [ $A_RC -ne 0 ]; then
  echo "ABORT: cancer_matched failed; not running cds_unbiased" | tee -a $MASTER
  exit $A_RC
fi

echo "[$(date +%H:%M:%S)] v4 training: starting cds_unbiased" | tee -a $MASTER
conda run -n quris python experiments/multi_enzyme/exp_train_phase3_v4.py \
    --variant cds_unbiased \
    --out-dir $OUT_BASE/v4_cds_unbiased \
    >> $OUT_BASE/v4_cds_unbiased/run.log 2>&1
B_RC=$?
echo "[$(date +%H:%M:%S)] cds_unbiased rc=$B_RC" | tee -a $MASTER

if [ $A_RC -eq 0 ] && [ $B_RC -eq 0 ]; then
  echo "[$(date +%H:%M:%S)] BOTH DONE" | tee -a $MASTER
  exit 0
else
  echo "[$(date +%H:%M:%S)] FAILURES: cancer=$A_RC cds=$B_RC" | tee -a $MASTER
  exit 1
fi
