#!/bin/bash
# Monitor all running experiments
# Usage: bash monitor_all.sh

echo "=========================================="
echo "  EXPERIMENT MONITOR - $(date)"
echo "=========================================="

echo ""
echo "=== COADREAD ==="
tail -3 experiments/multi_enzyme/outputs/tcga_gnomad/coadread_run.log 2>/dev/null || echo "No log"

echo ""
echo "=== EXOME MAP ==="
tail -3 experiments/multi_enzyme/outputs/exome_map_restart_run.log 2>/dev/null || echo "No log"
echo "Caches:" && ls experiments/multi_enzyme/outputs/exome_map/vienna_cache/ 2>/dev/null | wc -l | tr -d ' ' && echo " chromosomes"

echo ""
echo "=== ARCHITECTURE TRAINING ==="
tail -5 experiments/multi_enzyme/outputs/deep_architectures/logs/training.log 2>/dev/null || echo "No training log yet"
ls experiments/multi_enzyme/outputs/deep_architectures/*.json 2>/dev/null || echo "No results yet"

echo ""
echo "=== V4 DATASETS ==="
ls -lh data/processed/multi_enzyme/splits_v4_*.csv 2>/dev/null || echo "Generating..."

echo ""
echo "=== PROCESSES ==="
ps aux | grep -E "python.*(exp_|scripts/multi|tcga_coadread)" | grep -v grep | awk '{printf "  PID:%-6s CPU:%-6s MEM:%-8s %s %s\n", $2, $3"%", $6/1024"MB", $12, $13}'

echo ""
echo "=== WORKERS ==="
ps aux | grep "multiprocessing-fork" | grep -v grep | wc -l | tr -d ' '
echo " ViennaRNA workers active"
