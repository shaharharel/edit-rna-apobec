#!/bin/bash
# Single-shot OVERNIGHT_STATUS.md refresh. Runs from Mac. Called in a loop
# externally (e.g. `watch -n 600 bash scripts/gcp_panel/update_overnight_status.sh`).

set -u
STATUS=/Users/shaharharel/Documents/github/edit-rna-apobec/experiments/multi_enzyme/outputs/pcawg_tcw_panel/OVERNIGHT_STATUS.md
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S %Z')

# Collect ai-gpu2 status
GPU2_PROGRESS=$(gcloud compute ssh ai-gpu2 --zone=us-central1-a --command="tail -3 ~/logs/rnafm_progress.log 2>/dev/null; tail -3 ~/logs/score_watcher.log 2>/dev/null; df -h ~/ | tail -1; ls ~/data/panel/rnafm/*.npz 2>/dev/null | wc -l; ls ~/data/panel/scored_chroms/*.parquet 2>/dev/null | wc -l" 2>/dev/null)

# MAF local progress
MAF_SIZE=$(stat -f %z /Users/shaharharel/Documents/github/edit-rna-apobec/data/raw/pcawg_open/final_consensus_passonly.snv_mnv_indel.icgc.public.maf.gz 2>/dev/null | head -1)
MAF_MB=$((MAF_SIZE / 1024 / 1024))

# Append snapshot at the bottom of the status file
{
    echo ""
    echo "---"
    echo ""
    echo "## Snapshot $TIMESTAMP"
    echo ""
    echo '```'
    echo "MAF download: ${MAF_MB}MB / 882MB"
    echo ""
    echo "=== ai-gpu2 ==="
    echo "$GPU2_PROGRESS"
    echo '```'
} >> "$STATUS"
