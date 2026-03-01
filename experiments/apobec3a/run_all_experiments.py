#!/usr/bin/env python
"""Orchestrate all baseline experiments across models, neg-ratios, and seeds.

Runs every combination of:
  - Model: all baselines + editrna
  - Negative ratio: 1, 3, 5, 10
  - Seed: 42, 123, 456 (for variance estimation)

Produces a comprehensive results table and saves to CSV/JSON.

Usage:
    # Run all experiments (full grid)
    python experiments/apobec3a/run_all_experiments.py

    # Run a subset
    python experiments/apobec3a/run_all_experiments.py --models concat_mlp subtraction_mlp
    python experiments/apobec3a/run_all_experiments.py --neg-ratios 5 10
    python experiments/apobec3a/run_all_experiments.py --seeds 42

    # Just re-aggregate existing results (no training)
    python experiments/apobec3a/run_all_experiments.py --aggregate-only
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = PROJECT_ROOT / "experiments" / "apobec" / "train_baselines.py"

logger = logging.getLogger(__name__)

ALL_MODELS = [
    "pooled_mlp",
    "subtraction_mlp",
    "concat_mlp",
    "cross_attention",
    "diff_attention",
    "structure_only",
]

DEFAULT_NEG_RATIOS = [1, 3, 5, 10]
DEFAULT_SEEDS = [42, 123, 456]


def run_experiment(
    model: str,
    neg_ratio: int,
    seed: int,
    epochs: int,
    output_base: Path,
    embeddings_dir: str,
    splits_csv: str,
    python_bin: str,
) -> Optional[Dict]:
    """Run a single experiment via subprocess."""
    run_name = f"{model}_neg{neg_ratio}_seed{seed}"
    run_dir = output_base / run_name

    # Check if results already exist
    results_path = run_dir / model / "results.json"
    if results_path.exists():
        logger.info("  [SKIP] %s (results exist)", run_name)
        try:
            with open(results_path) as f:
                return json.load(f)
        except Exception:
            pass

    logger.info("  [RUN]  %s", run_name)

    cmd = [
        python_bin,
        str(TRAIN_SCRIPT),
        "--model", model,
        "--neg-ratio", str(neg_ratio),
        "--seed", str(seed),
        "--epochs", str(epochs),
        "--embeddings-dir", embeddings_dir,
        "--splits-csv", splits_csv,
        "--output-dir", str(run_dir),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,
            cwd=str(PROJECT_ROOT),
        )

        if result.returncode != 0:
            logger.error("  FAILED: %s\n  stderr: %s", run_name, result.stderr[-500:])
            return {"model": model, "neg_ratio": neg_ratio, "seed": seed, "error": result.stderr[-200:]}

        # Load results
        if results_path.exists():
            with open(results_path) as f:
                data = json.load(f)
            data["neg_ratio"] = neg_ratio
            data["seed"] = seed
            return data

    except subprocess.TimeoutExpired:
        logger.error("  TIMEOUT: %s", run_name)
        return {"model": model, "neg_ratio": neg_ratio, "seed": seed, "error": "timeout"}
    except Exception as e:
        logger.error("  ERROR: %s - %s", run_name, e)
        return {"model": model, "neg_ratio": neg_ratio, "seed": seed, "error": str(e)}

    return None


def aggregate_results(output_base: Path) -> pd.DataFrame:
    """Scan output directory and aggregate all results into a DataFrame."""
    rows = []

    for run_dir in sorted(output_base.iterdir()):
        if not run_dir.is_dir():
            continue

        # Parse run name: model_negN_seedN
        parts = run_dir.name.rsplit("_seed", 1)
        if len(parts) != 2:
            continue
        prefix, seed_str = parts
        neg_parts = prefix.rsplit("_neg", 1)
        if len(neg_parts) != 2:
            continue
        model_name, neg_str = neg_parts

        try:
            seed = int(seed_str)
            neg_ratio = int(neg_str)
        except ValueError:
            continue

        # Look for results file
        results_path = run_dir / model_name / "results.json"
        if not results_path.exists():
            continue

        try:
            with open(results_path) as f:
                data = json.load(f)
        except Exception:
            continue

        test_m = data.get("test_metrics", {})
        val_m = data.get("val_metrics", {})

        rows.append({
            "model": model_name,
            "neg_ratio": neg_ratio,
            "seed": seed,
            "best_epoch": data.get("best_epoch", 0),
            "train_time_s": data.get("train_time_seconds", 0),
            "val_auroc": val_m.get("auroc", np.nan),
            "val_auprc": val_m.get("auprc", np.nan),
            "val_f1": val_m.get("f1", np.nan),
            "test_auroc": test_m.get("auroc", np.nan),
            "test_auprc": test_m.get("auprc", np.nan),
            "test_f1": test_m.get("f1", np.nan),
            "test_precision": test_m.get("precision", np.nan),
            "test_recall": test_m.get("recall", np.nan),
            "test_p90r": test_m.get("precision_at_90recall", np.nan),
            "test_ece": test_m.get("ece", np.nan),
        })

    return pd.DataFrame(rows)


def print_summary_table(df: pd.DataFrame):
    """Print aggregated results with mean +/- std across seeds."""
    if df.empty:
        print("No results to display.")
        return

    print("\n" + "=" * 120)
    print("ALL EXPERIMENTS -- SUMMARY (mean +/- std across seeds)")
    print("=" * 120)

    group_cols = ["model", "neg_ratio"]
    metric_cols = ["test_auroc", "test_auprc", "test_f1", "test_p90r", "test_ece"]

    agg = df.groupby(group_cols)[metric_cols].agg(["mean", "std", "count"])

    print(f"{'Model':<22} {'Neg Ratio':>9} {'N':>3} "
          f"{'AUROC':>14} {'AUPRC':>14} {'F1':>14} "
          f"{'P@90R':>14} {'ECE':>14}")
    print("-" * 120)

    for (model, neg_ratio), row in agg.iterrows():
        n = int(row[("test_auroc", "count")])

        def fmt(col):
            m = row[(col, "mean")]
            s = row[(col, "std")]
            if np.isnan(m):
                return f"{'N/A':>14}"
            if n == 1 or np.isnan(s):
                return f"{m:>14.4f}"
            return f"{m:.4f}+/-{s:.4f}"

        print(f"{model:<22} {neg_ratio:>9} {n:>3} "
              f"{fmt('test_auroc'):>14} {fmt('test_auprc'):>14} {fmt('test_f1'):>14} "
              f"{fmt('test_p90r'):>14} {fmt('test_ece'):>14}")

    print("=" * 120)

    # Best model per neg_ratio
    print("\nBest model per negative ratio (by test AUROC):")
    for neg_ratio in sorted(df["neg_ratio"].unique()):
        subset = df[df["neg_ratio"] == neg_ratio]
        best_idx = subset.groupby("model")["test_auroc"].mean().idxmax()
        best_auroc = subset.groupby("model")["test_auroc"].mean().max()
        print(f"  neg_ratio={neg_ratio}: {best_idx} (AUROC={best_auroc:.4f})")


def main():
    parser = argparse.ArgumentParser(
        description="Run all baseline experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--models", nargs="+", default=None,
                        help="Models to train (default: all baselines)")
    parser.add_argument("--neg-ratios", nargs="+", type=int, default=None,
                        help="Negative ratios to test")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Random seeds for variance estimation")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--embeddings-dir", type=str,
                        default=str(PROJECT_ROOT / "data" / "processed" / "embeddings"))
    parser.add_argument("--splits-csv", type=str,
                        default=str(PROJECT_ROOT / "data" / "processed" / "splits_expanded.csv"))
    parser.add_argument("--output-dir", type=str,
                        default=str(PROJECT_ROOT / "experiments" / "apobec" / "outputs" / "baselines"))
    parser.add_argument("--python", type=str, default=sys.executable,
                        help="Python interpreter to use")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="Only aggregate existing results (no training)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    models = args.models or ALL_MODELS
    neg_ratios = args.neg_ratios or DEFAULT_NEG_RATIOS
    seeds = args.seeds or DEFAULT_SEEDS
    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    total = len(models) * len(neg_ratios) * len(seeds)
    logger.info("Experiment grid: %d models x %d neg_ratios x %d seeds = %d runs",
                len(models), len(neg_ratios), len(seeds), total)

    if not args.aggregate_only:
        completed = 0
        for neg_ratio in neg_ratios:
            for model in models:
                for seed in seeds:
                    completed += 1
                    logger.info("[%d/%d] model=%s neg_ratio=%d seed=%d",
                                completed, total, model, neg_ratio, seed)
                    run_experiment(
                        model=model,
                        neg_ratio=neg_ratio,
                        seed=seed,
                        epochs=args.epochs,
                        output_base=output_base,
                        embeddings_dir=args.embeddings_dir,
                        splits_csv=args.splits_csv,
                        python_bin=args.python,
                    )

    # Aggregate results
    logger.info("Aggregating results from %s ...", output_base)
    df = aggregate_results(output_base)

    if df.empty:
        logger.warning("No results found. Run experiments first.")
        return

    # Save full results
    csv_path = output_base / "all_experiments.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Full results saved to %s (%d rows)", csv_path, len(df))

    # Save summary
    summary = df.groupby(["model", "neg_ratio"]).agg(
        test_auroc_mean=("test_auroc", "mean"),
        test_auroc_std=("test_auroc", "std"),
        test_auprc_mean=("test_auprc", "mean"),
        test_auprc_std=("test_auprc", "std"),
        test_f1_mean=("test_f1", "mean"),
        test_f1_std=("test_f1", "std"),
        test_p90r_mean=("test_p90r", "mean"),
        test_ece_mean=("test_ece", "mean"),
        n_runs=("test_auroc", "count"),
    ).reset_index()
    summary.to_csv(output_base / "experiment_summary.csv", index=False)

    # Print table
    print_summary_table(df)

    logger.info("Done. Results in %s", output_base)


if __name__ == "__main__":
    main()
