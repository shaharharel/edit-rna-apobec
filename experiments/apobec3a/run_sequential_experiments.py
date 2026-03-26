#!/usr/bin/env python
"""Sequential experiment runner — RAM-safe overnight execution.

Runs ALL experiments one at a time in order of increasing RAM usage.
Each experiment runs as a subprocess so memory is fully released between runs.

Usage:
    conda activate quris
    KMP_DUPLICATE_LIB_OK=TRUE python experiments/apobec3a/run_sequential_experiments.py

    # Resume from a specific step (e.g., after a crash):
    KMP_DUPLICATE_LIB_OK=TRUE python experiments/apobec3a/run_sequential_experiments.py --start-from 5

    # Run only specific experiments:
    KMP_DUPLICATE_LIB_OK=TRUE python experiments/apobec3a/run_sequential_experiments.py --only dataset_deep_analysis multitask

    # Dry-run (show what would run):
    KMP_DUPLICATE_LIB_OK=TRUE python experiments/apobec3a/run_sequential_experiments.py --dry-run
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "apobec3a" / "outputs"

logger = logging.getLogger("runner")

# ============================================================================
# Experiment definitions — ordered by RAM usage (lightest first)
# ============================================================================

EXPERIMENTS = [
    # --- Phase 1: Analysis-only (CSV data, no embeddings) ~50MB each ---
    {
        "name": "dataset_deep_analysis",
        "script": "experiments/apobec3a/exp_dataset_deep_analysis.py",
        "description": "Deep dataset statistics and visualizations",
        "ram_category": "light",
        "estimated_ram_mb": 100,
    },
    {
        "name": "gate_ablation",
        "script": "experiments/apobec3a/exp_gate_ablation.py",
        "description": "Gate weight analysis and feature ablation visualization",
        "ram_category": "light",
        "estimated_ram_mb": 200,
    },
    {
        "name": "motif_analysis",
        "script": "experiments/apobec3a/exp_motif_analysis.py",
        "description": "Extended motif analysis + Jalili 2023 validation",
        "ram_category": "light",
        "estimated_ram_mb": 100,
    },
    {
        "name": "structure_analysis",
        "script": "experiments/apobec3a/exp_structure_analysis.py",
        "description": "RNA structure delta analysis",
        "ram_category": "light",
        "estimated_ram_mb": 200,
    },

    # --- Phase 2: Pooled embeddings only (~200MB each) ---
    {
        "name": "embedding_visualizations",
        "script": "experiments/apobec3a/exp_embedding_visualizations.py",
        "description": "UMAP/TSNE/PCA with semantic coloring",
        "ram_category": "medium",
        "estimated_ram_mb": 500,
    },
    {
        "name": "multitask",
        "script": "experiments/apobec3a/exp_multitask_comparison.py",
        "description": "Binary vs regression vs multitask comparison",
        "ram_category": "medium",
        "estimated_ram_mb": 500,
    },
    {
        "name": "incremental_spearman",
        "script": "experiments/apobec3a/exp_incremental_spearman.py",
        "description": "Incremental dataset addition — Spearman correlation",
        "ram_category": "medium",
        "estimated_ram_mb": 500,
    },
    {
        "name": "incremental_datasets",
        "script": "experiments/apobec3a/exp_incremental_datasets.py",
        "description": "Incremental dataset addition — AUROC (SubtractionMLP)",
        "ram_category": "medium",
        "estimated_ram_mb": 500,
    },
    {
        "name": "dataset_matrix",
        "script": "experiments/apobec3a/exp_dataset_matrix.py",
        "description": "Per-dataset cross-training matrix (SubtractionMLP)",
        "ram_category": "medium",
        "estimated_ram_mb": 500,
    },
    {
        "name": "rate_prediction",
        "script": "experiments/apobec3a/exp_rate_prediction.py",
        "description": "Editing rate regression comparison",
        "ram_category": "medium",
        "estimated_ram_mb": 800,
    },

    # --- Phase 3: Tissue/biology analyses (pooled embeddings + CSVs) ---
    {
        "name": "cross_tissue_disease",
        "script": "experiments/apobec3a/exp_cross_tissue_disease.py",
        "description": "54-tissue GTEx + TCGA cancer analysis",
        "ram_category": "medium",
        "estimated_ram_mb": 500,
    },
    {
        "name": "constitutive_facultative",
        "script": "scripts/apobec3a/constitutive_facultative_analysis.py",
        "description": "Constitutive vs facultative editing analysis",
        "ram_category": "medium",
        "estimated_ram_mb": 600,
    },
    {
        "name": "tissue_conditioned_rate",
        "script": "experiments/apobec3a/exp_tissue_conditioned_rate.py",
        "description": "Tissue-conditioned rate prediction",
        "ram_category": "medium",
        "estimated_ram_mb": 800,
    },
    {
        "name": "tissue_motifs",
        "script": "scripts/apobec3a/tissue_motif_analysis.py",
        "description": "Tissue-specific motif discovery",
        "ram_category": "medium",
        "estimated_ram_mb": 400,
    },
    {
        "name": "pcpg_analysis",
        "script": "scripts/apobec3a/pcpg_cancer_analysis.py",
        "description": "PCPG adrenal tumor deep dive",
        "ram_category": "medium",
        "estimated_ram_mb": 500,
    },
    {
        "name": "binary_rate_correlation",
        "script": "experiments/apobec3a/exp_binary_rate_correlation.py",
        "description": "Binary classifier score vs editing rate correlation",
        "ram_category": "medium",
        "estimated_ram_mb": 300,
    },
    {
        "name": "levanon_tissue_viz",
        "script": "experiments/apobec3a/exp_levanon_tissue_viz.py",
        "description": "Levanon tissue embedding visualization",
        "ram_category": "medium",
        "estimated_ram_mb": 500,
    },

    # --- Phase 4: Heavy experiments (token-level embeddings ~13GB) ---
    {
        "name": "dataset_matrix_editrna",
        "script": "experiments/apobec3a/exp_dataset_matrix_editrna.py",
        "description": "Per-dataset cross-training matrix (EditRNA-A3A)",
        "ram_category": "heavy",
        "estimated_ram_mb": 15000,
    },
    {
        "name": "incremental_editrna",
        "script": "experiments/apobec3a/exp_incremental_editrna.py",
        "description": "Incremental dataset addition (EditRNA-A3A)",
        "ram_category": "heavy",
        "estimated_ram_mb": 15000,
    },
]


def check_ram():
    """Return available RAM in MB."""
    try:
        import psutil
        return psutil.virtual_memory().available / 1e6
    except ImportError:
        return -1


def run_experiment(exp, python_bin, timeout_minutes=120):
    """Run a single experiment as a subprocess."""
    script_path = PROJECT_ROOT / exp["script"]
    if not script_path.exists():
        return {"status": "skipped", "reason": f"Script not found: {exp['script']}"}

    available_ram = check_ram()
    if available_ram > 0 and exp["estimated_ram_mb"] > available_ram * 0.8:
        return {
            "status": "skipped",
            "reason": f"Insufficient RAM: need ~{exp['estimated_ram_mb']}MB, have {available_ram:.0f}MB",
        }

    env = os.environ.copy()
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["PYTHONUNBUFFERED"] = "1"

    cmd = [python_bin, str(script_path)]

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_minutes * 60,
            cwd=str(PROJECT_ROOT),
            env=env,
        )
        elapsed = time.time() - t0

        if result.returncode != 0:
            # Extract last 30 lines of stderr for debugging
            stderr_tail = "\n".join(result.stderr.strip().split("\n")[-30:])
            return {
                "status": "failed",
                "returncode": result.returncode,
                "elapsed_seconds": elapsed,
                "stderr_tail": stderr_tail,
            }

        return {
            "status": "success",
            "elapsed_seconds": elapsed,
            "stdout_tail": "\n".join(result.stdout.strip().split("\n")[-10:]),
        }

    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "timeout_minutes": timeout_minutes,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Sequential experiment runner")
    parser.add_argument("--start-from", type=int, default=1,
                        help="Start from this step number (1-indexed)")
    parser.add_argument("--only", nargs="+", default=None,
                        help="Run only these experiments (by name)")
    parser.add_argument("--skip-heavy", action="store_true",
                        help="Skip experiments that need token embeddings")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print execution plan without running")
    parser.add_argument("--python", default=sys.executable,
                        help="Python interpreter to use")
    parser.add_argument("--timeout", type=int, default=120,
                        help="Timeout per experiment in minutes")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Filter experiments
    experiments = EXPERIMENTS
    if args.only:
        experiments = [e for e in experiments if e["name"] in args.only]
    if args.skip_heavy:
        experiments = [e for e in experiments if e["ram_category"] != "heavy"]

    total = len(experiments)
    logger.info("=" * 80)
    logger.info("SEQUENTIAL EXPERIMENT RUNNER")
    logger.info("=" * 80)
    logger.info("Total experiments: %d", total)
    logger.info("Starting from step: %d", args.start_from)
    logger.info("Python: %s", args.python)
    logger.info("Available RAM: %.0f MB", check_ram())
    logger.info("=" * 80)

    if args.dry_run:
        for i, exp in enumerate(experiments, 1):
            skip = " [SKIP]" if i < args.start_from else ""
            print(f"  Step {i:2d}: [{exp['ram_category']:>6s}] {exp['name']:<30s} "
                  f"~{exp['estimated_ram_mb']:>6d}MB  {exp['description']}{skip}")
        print(f"\nTotal estimated heavy RAM: "
              f"{sum(e['estimated_ram_mb'] for e in experiments if e['ram_category']=='heavy')}MB")
        return

    # Run experiments
    results = []
    t_total_start = time.time()

    for i, exp in enumerate(experiments, 1):
        if i < args.start_from:
            logger.info("[%d/%d] SKIPPING (start-from=%d): %s", i, total, args.start_from, exp["name"])
            continue

        ram_mb = check_ram()
        logger.info("")
        logger.info("=" * 70)
        logger.info("[%d/%d] %s — %s", i, total, exp["name"], exp["description"])
        logger.info("  RAM: %.0f MB available, ~%d MB needed, category=%s",
                     ram_mb, exp["estimated_ram_mb"], exp["ram_category"])
        logger.info("=" * 70)

        result = run_experiment(exp, args.python, args.timeout)
        result["step"] = i
        result["name"] = exp["name"]
        result["ram_category"] = exp["ram_category"]
        results.append(result)

        status = result["status"]
        if status == "success":
            elapsed = result["elapsed_seconds"]
            logger.info("  SUCCESS in %.1fs (%.1f min)", elapsed, elapsed / 60)
            if "stdout_tail" in result:
                for line in result["stdout_tail"].split("\n")[-3:]:
                    logger.info("  > %s", line)
        elif status == "failed":
            logger.error("  FAILED (exit code %d) in %.1fs", result["returncode"], result["elapsed_seconds"])
            if "stderr_tail" in result:
                for line in result["stderr_tail"].split("\n")[-5:]:
                    logger.error("  ! %s", line)
        elif status == "skipped":
            logger.warning("  SKIPPED: %s", result["reason"])
        elif status == "timeout":
            logger.error("  TIMEOUT after %d minutes", args.timeout)
        else:
            logger.error("  ERROR: %s", result.get("message", "unknown"))

    # Summary
    total_time = time.time() - t_total_start
    n_success = sum(1 for r in results if r["status"] == "success")
    n_failed = sum(1 for r in results if r["status"] == "failed")
    n_skipped = sum(1 for r in results if r["status"] == "skipped")
    n_timeout = sum(1 for r in results if r["status"] == "timeout")

    logger.info("")
    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info("Total time: %.1f min", total_time / 60)
    logger.info("Success: %d | Failed: %d | Skipped: %d | Timeout: %d",
                n_success, n_failed, n_skipped, n_timeout)

    for r in results:
        icon = {"success": "OK", "failed": "FAIL", "skipped": "SKIP", "timeout": "TIME"}.get(r["status"], "??")
        elapsed = f"{r.get('elapsed_seconds', 0):.0f}s" if "elapsed_seconds" in r else "-"
        logger.info("  [%4s] Step %2d: %-30s %s", icon, r["step"], r["name"], elapsed)

    # Save run log
    log_path = OUTPUT_DIR / "sequential_run_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def serialize(obj):
        if hasattr(obj, "item"):
            return obj.item()
        return str(obj)

    with open(log_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_time_seconds": total_time,
            "summary": {"success": n_success, "failed": n_failed, "skipped": n_skipped, "timeout": n_timeout},
            "results": results,
        }, f, indent=2, default=serialize)
    logger.info("\nRun log saved to %s", log_path)

    if n_failed > 0:
        logger.warning("\n%d experiments FAILED. Check stderr output above.", n_failed)
        sys.exit(1)


if __name__ == "__main__":
    main()
