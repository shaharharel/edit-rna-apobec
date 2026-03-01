#!/bin/bash
# Full pipeline rebuild after switching to hg38
# Run: bash scripts/apobec/rebuild_pipeline_hg38.sh
set -e

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export KMP_DUPLICATE_LIB_OK=TRUE

CONDA_ENV="quris"
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON="conda run -n $CONDA_ENV python"

echo "============================================"
echo "EditRNA-A3A: Full Pipeline Rebuild (hg38)"
echo "============================================"
echo "Project root: $PROJECT_ROOT"
echo "Start time: $(date)"
echo ""

# Step 0: Check hg38 genome exists
GENOME="$PROJECT_ROOT/data/raw/genomes/hg38.fa"
if [ ! -f "$GENOME" ]; then
    echo "ERROR: hg38.fa not found at $GENOME"
    echo "Download with: curl -o data/raw/genomes/hg38.fa.gz https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz && gunzip data/raw/genomes/hg38.fa.gz"
    exit 1
fi

# Step 0b: Index genome if needed
if [ ! -f "${GENOME}.fai" ]; then
    echo "==> Indexing hg38 genome with pyfaidx..."
    $PYTHON -c "from pyfaidx import Fasta; Fasta('$GENOME')"
fi

# Step 1: Rebuild unified dataset (hg38 coordinates)
echo ""
echo "=== Step 1: Rebuild unified dataset (hg38) ==="
$PYTHON scripts/apobec/build_unified_dataset.py

# Step 2: Re-extract sequences from hg38
echo ""
echo "=== Step 2: Extract 201-nt sequences from hg38 ==="
$PYTHON scripts/apobec/extract_sequences_and_structures.py --genome "$GENOME"

# Step 3: Expand dataset (tiered negatives, embeddings)
echo ""
echo "=== Step 3: Expand dataset with negatives + embeddings ==="
$PYTHON scripts/apobec/expand_dataset.py

# Step 4: Generate hard negatives
echo ""
echo "=== Step 4: Generate structure-matched hard negatives ==="
$PYTHON scripts/apobec/generate_hardneg_pipeline.py --neg-ratio 5 --candidates-ratio 20 --workers 4

# Step 5: Download ClinVar (GRCh38 version)
echo ""
echo "=== Step 5: Download and process ClinVar (GRCh38) ==="
$PYTHON scripts/apobec/download_clinvar.py

# Step 6: Run experiments
echo ""
echo "=== Step 6: Run experiments ==="

echo "  6a: Tabular baselines..."
$PYTHON experiments/apobec/exp0_tabular_baseline.py 2>&1 | tail -5

echo "  6b: Dataset matrix..."
$PYTHON experiments/apobec/exp_dataset_matrix.py 2>&1 | tail -5

echo "  6c: Rate prediction..."
$PYTHON experiments/apobec/exp_rate_prediction.py 2>&1 | tail -5

echo "  6d: Rate deep dive..."
$PYTHON experiments/apobec/exp_rate_deep_dive.py 2>&1 | tail -5

echo "  6e: TC-motif reanalysis..."
$PYTHON experiments/apobec/exp_tc_motif_reanalysis.py 2>&1 | tail -5

echo "  6f: ClinVar prediction..."
$PYTHON experiments/apobec/exp_clinvar_prediction.py 2>&1 | tail -5

echo "  6g: Hard negative baselines..."
$PYTHON experiments/apobec/exp_hardneg_baselines.py 2>&1 | tail -5

echo ""
echo "============================================"
echo "Pipeline complete at $(date)"
echo "============================================"
