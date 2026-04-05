#!/usr/bin/env python3
"""Neural network confound control experiments 1b-1f for TCGA cancers.

Replicates the XGB-based confound controls using the Phase3 neural model.
Uses pre-computed neural scores where possible, re-scores only for ablations.

Experiments:
  1b. TC-Stratified Analysis: OR@p90 within TC-only and non-TC subsets
  1c. CpG Stratification: OR@p90 for TC+nonCpG, TC+CpG, nonTC+nonCpG, nonTC+CpG
  1d. Gene Expression Stratification: OR by expression quartile (if RSEM available)
  1e. StructOnly Ablation: Score with motif features zeroed (cols 0:24)
  1f. Per-Sample APOBEC Signature (BRCA only): OR for APOBEC-high vs low samples

Usage:
    /opt/miniconda3/envs/quris/bin/python -u experiments/multi_enzyme/exp_nn_tcga_controls.py
"""

import gc
import gzip
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ============================================================================
# Paths
# ============================================================================
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "multi_enzyme" / "outputs" / "neural_tcga_fixed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(OUTPUT_DIR / "nn_controls_run.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
EMB_DIR = DATA_DIR / "processed" / "multi_enzyme" / "embeddings"
HAND_DIR = DATA_DIR / "processed" / "multi_enzyme" / "tcga_hand_features"
RAW_SCORES_DIR = PROJECT_ROOT / "experiments" / "multi_enzyme" / "outputs" / "tcga_gnomad" / "raw_scores"
NN_SCORES_DIR = PROJECT_ROOT / "experiments" / "multi_enzyme" / "outputs" / "phase3_neural_true_validation"
TCGA_DIR = DATA_DIR / "raw" / "tcga"
NEURAL_MODEL_PATH = NN_SCORES_DIR / "phase3_neural_full.pt"

CANCERS = ["blca", "brca", "cesc", "lusc", "hnsc", "esca", "stad", "lihc"]
# Cancers with RSEM expression data
CANCERS_WITH_EXPR = ["blca", "brca", "cesc", "lusc"]

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else torch.device("cpu")
)
SEED = 42
BATCH_SIZE = 512

# Model constants
ENZYMES = ["A3A", "A3B", "A3G", "A3A_A3G", "Neither"]
N_PER_ENZYME = len(ENZYMES)
D_INPUT = 1320
D_SHARED = 128
N_ENZYMES = 6


# ============================================================================
# Model definition (must match training exactly)
# ============================================================================
class Phase3Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_encoder = nn.Sequential(
            nn.Linear(D_INPUT, 256), nn.GELU(), nn.Dropout(0.3), nn.LayerNorm(256),
            nn.Linear(256, D_SHARED), nn.GELU(), nn.Dropout(0.2),
        )
        self.binary_head = nn.Linear(D_SHARED, 1)
        self.enzyme_adapters = nn.ModuleDict({
            enz: nn.Sequential(nn.Linear(D_SHARED, 32), nn.GELU(), nn.Linear(32, 1))
            for enz in ENZYMES
        })
        self.enzyme_classifier = nn.Sequential(
            nn.Linear(D_SHARED, 64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, N_ENZYMES),
        )

    def forward(self, x):
        shared = self.shared_encoder(x)
        binary_logit = self.binary_head(shared).squeeze(-1)
        per_enzyme_logits = [self.enzyme_adapters[enz](shared).squeeze(-1) for enz in ENZYMES]
        return binary_logit, per_enzyme_logits, None, shared


# ============================================================================
# Helpers
# ============================================================================
def compute_or_fisher(scores, mut_mask, ctrl_mask, percentile, pooled_scores=None):
    """Compute OR using fisher_exact at a given percentile threshold.

    Threshold is computed on pooled mut+ctrl distribution (or on pooled_scores if provided).
    """
    mut_s = scores[mut_mask]
    ctrl_s = scores[ctrl_mask]
    if pooled_scores is not None:
        thresh = np.percentile(pooled_scores, percentile)
    else:
        all_s = np.concatenate([mut_s, ctrl_s])
        thresh = np.percentile(all_s, percentile)
    ma = int((mut_s >= thresh).sum())
    mb = int((mut_s < thresh).sum())
    ca = int((ctrl_s >= thresh).sum())
    cb = int((ctrl_s < thresh).sum())
    if all(x > 0 for x in [ma, mb, ca, cb]):
        OR, pv = stats.fisher_exact([[ma, mb], [ca, cb]])
    else:
        OR, pv = float("nan"), 1.0
    return {
        "OR": float(OR), "p": float(pv), "threshold": float(thresh),
        "mut_above": ma, "mut_below": mb, "ctrl_above": ca, "ctrl_below": cb,
    }


def compute_enrichment_full(scores, mut_mask, ctrl_mask):
    """Compute enrichment at p50, p75, p90, p95."""
    result = {}
    pooled = np.concatenate([scores[mut_mask], scores[ctrl_mask]])
    for pct in [50, 75, 90, 95]:
        result[f"p{pct}"] = compute_or_fisher(scores, mut_mask, ctrl_mask, pct, pooled_scores=pooled)
    result["n_mut"] = int(mut_mask.sum())
    result["n_ctrl"] = int(ctrl_mask.sum())
    result["mean_mut"] = float(np.mean(scores[mut_mask])) if mut_mask.sum() > 0 else 0.0
    result["mean_ctrl"] = float(np.mean(scores[ctrl_mask])) if ctrl_mask.sum() > 0 else 0.0
    return result


def load_cancer_data(cancer, model, device):
    """Load embeddings, hand features, and re-score with neural model.

    NOTE: The pre-computed neural scores in phase3_neural_true_validation/tcga_*_neural_scores.csv
    were scored with ZEROED hand features (a bug in the original script). We re-score here
    using the correct cached 40-dim hand features from tcga_hand_features/*.npy.
    """
    raw_path = RAW_SCORES_DIR / f"{cancer}_scores.csv"
    hand_path = HAND_DIR / f"{cancer}_hand40.npy"
    emb_path = EMB_DIR / f"rnafm_tcga_{cancer}.pt"

    raw_df = pd.read_csv(raw_path)
    hand = np.load(hand_path)
    emb_data = torch.load(emb_path, weights_only=False)

    n = len(raw_df)
    assert hand.shape[0] == n, f"Row mismatch: raw={n}, hand={hand.shape[0]}"

    pooled_orig = emb_data["pooled_orig"].numpy()
    pooled_edited = emb_data["pooled_edited"].numpy()
    edit_delta = pooled_edited - pooled_orig

    # Build 1320-dim features with CORRECT hand features
    X_full = np.concatenate([
        pooled_orig[:n], edit_delta[:n], hand[:n]
    ], axis=1).astype(np.float32)

    # Score with neural model
    nn_binary = score_neural(model, X_full, device, BATCH_SIZE)

    types = raw_df["type"].values
    mut_mask = types == "mutation"
    ctrl_mask = types == "control"
    tc_context = raw_df["tc_context"].values
    xgb_scores = raw_df["score"].values

    # Clean up large tensors
    del emb_data, pooled_orig, pooled_edited, edit_delta, X_full
    gc.collect()

    return {
        "nn_binary": nn_binary, "xgb_scores": xgb_scores,
        "mut_mask": mut_mask, "ctrl_mask": ctrl_mask,
        "tc_context": tc_context, "hand": hand, "n": n,
        "types": types,
    }


def score_neural(model, X, device, batch_size=512):
    """Score with neural model, return binary sigmoid scores."""
    all_b = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            end = min(start + batch_size, len(X))
            x = torch.from_numpy(X[start:end].astype(np.float32)).to(device)
            bl, _, _, _ = model(x)
            all_b.append(torch.sigmoid(bl).cpu().numpy())
    return np.concatenate(all_b)


# ============================================================================
# Experiment 1b: TC-Stratified Analysis
# ============================================================================
def exp_1b_tc_stratified(all_data):
    """For each cancer, compute OR@p90 separately for TC-only and non-TC."""
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 1b: TC-Stratified Analysis (Neural)")
    logger.info("=" * 70)

    results = {}
    for cancer in CANCERS:
        d = all_data[cancer]
        nn_scores = d["nn_binary"]
        xgb_scores = d["xgb_scores"]
        mut_mask = d["mut_mask"]
        ctrl_mask = d["ctrl_mask"]
        tc = d["tc_context"]

        cancer_result = {}
        for score_name, scores in [("neural", nn_scores), ("xgb", xgb_scores)]:
            for tc_label, tc_val in [("TC", 1), ("nonTC", 0)]:
                tc_mask = tc == tc_val
                tc_mut = mut_mask & tc_mask
                tc_ctrl = ctrl_mask & tc_mask
                if tc_mut.sum() < 10 or tc_ctrl.sum() < 10:
                    continue
                # Compute threshold on pooled TC-subset distribution
                pooled_tc = np.concatenate([scores[tc_mut], scores[tc_ctrl]])
                enr = {}
                for pct in [50, 75, 90, 95]:
                    enr[f"p{pct}"] = compute_or_fisher(scores, tc_mut, tc_ctrl, pct,
                                                        pooled_scores=pooled_tc)
                enr["n_mut"] = int(tc_mut.sum())
                enr["n_ctrl"] = int(tc_ctrl.sum())
                cancer_result[f"{score_name}_{tc_label}"] = enr

        results[cancer] = cancer_result

    # Print comparison table
    logger.info(f"\n{'Cancer':>8} | {'NN TC OR@p90':>14} {'NN nonTC OR@p90':>16} | "
                f"{'XGB TC OR@p90':>14} {'XGB nonTC OR@p90':>16}")
    logger.info("-" * 80)
    for cancer in CANCERS:
        r = results.get(cancer, {})
        nn_tc = r.get("neural_TC", {}).get("p90", {}).get("OR", float("nan"))
        nn_ntc = r.get("neural_nonTC", {}).get("p90", {}).get("OR", float("nan"))
        xgb_tc = r.get("xgb_TC", {}).get("p90", {}).get("OR", float("nan"))
        xgb_ntc = r.get("xgb_nonTC", {}).get("p90", {}).get("OR", float("nan"))
        logger.info(f"{cancer:>8} | {nn_tc:>14.3f} {nn_ntc:>16.3f} | {xgb_tc:>14.3f} {xgb_ntc:>16.3f}")

    return results


# ============================================================================
# Experiment 1c: CpG Stratification
# ============================================================================
def exp_1c_cpg_stratification(all_data):
    """Split by TC and CpG context, compute OR@p90 for each quadrant."""
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 1c: CpG Stratification (Neural)")
    logger.info("=" * 70)

    results = {}
    for cancer in CANCERS:
        d = all_data[cancer]
        nn_scores = d["nn_binary"]
        xgb_scores = d["xgb_scores"]
        mut_mask = d["mut_mask"]
        ctrl_mask = d["ctrl_mask"]
        tc = d["tc_context"]
        hand = d["hand"]

        # p1 one-hot is at cols 12:16 (A, C, G, T). Col 14 = p1=G means CpG.
        cpg = hand[:, 14] == 1.0

        cancer_result = {}
        quadrants = [
            ("TC_nonCpG", (tc == 1) & (~cpg)),
            ("TC_CpG", (tc == 1) & cpg),
            ("nonTC_nonCpG", (tc == 0) & (~cpg)),
            ("nonTC_CpG", (tc == 0) & cpg),
        ]

        for score_name, scores in [("neural", nn_scores), ("xgb", xgb_scores)]:
            for q_name, q_mask in quadrants:
                q_mut = mut_mask & q_mask
                q_ctrl = ctrl_mask & q_mask
                if q_mut.sum() < 10 or q_ctrl.sum() < 10:
                    continue
                pooled_q = np.concatenate([scores[q_mut], scores[q_ctrl]])
                enr = {}
                for pct in [50, 75, 90, 95]:
                    enr[f"p{pct}"] = compute_or_fisher(scores, q_mut, q_ctrl, pct,
                                                        pooled_scores=pooled_q)
                enr["n_mut"] = int(q_mut.sum())
                enr["n_ctrl"] = int(q_ctrl.sum())
                cancer_result[f"{score_name}_{q_name}"] = enr

        results[cancer] = cancer_result

    # Print comparison table (TC+nonCpG = "pure APOBEC")
    logger.info(f"\n{'Cancer':>8} | {'NN TC+noCpG':>12} {'NN TC+CpG':>10} {'NN nTC+noCpG':>13} {'NN nTC+CpG':>11} | "
                f"{'XGB TC+noCpG':>13} {'XGB TC+CpG':>11}")
    logger.info("-" * 100)
    for cancer in CANCERS:
        r = results.get(cancer, {})
        vals = []
        for prefix in ["neural", "xgb"]:
            for q in ["TC_nonCpG", "TC_CpG", "nonTC_nonCpG", "nonTC_CpG"]:
                v = r.get(f"{prefix}_{q}", {}).get("p90", {}).get("OR", float("nan"))
                vals.append(v)
        logger.info(f"{cancer:>8} | {vals[0]:>12.3f} {vals[1]:>10.3f} {vals[2]:>13.3f} {vals[3]:>11.3f} | "
                    f"{vals[4]:>13.3f} {vals[5]:>11.3f}")

    return results


# ============================================================================
# Experiment 1d: Gene Expression Stratification
# ============================================================================
def exp_1d_expression_stratification(all_data):
    """Split positions by gene expression quartile, compute OR per quartile."""
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 1d: Gene Expression Stratification (Neural)")
    logger.info("=" * 70)

    # Check which cancers have RSEM data
    available = []
    for cancer in CANCERS:
        rsem_path = TCGA_DIR / f"{cancer}_tcga_pan_can_atlas_2018_mrna_rsem.txt"
        if rsem_path.exists():
            available.append(cancer)
    logger.info(f"  RSEM data available for: {available}")

    if not available:
        logger.info("  No expression data available. Skipping 1d.")
        return {"status": "skipped", "reason": "no RSEM data"}

    results = {}
    for cancer in available:
        logger.info(f"\n  Processing {cancer.upper()}...")
        d = all_data[cancer]

        # Load RSEM: rows = genes, cols = samples. Compute median expression per gene.
        rsem_path = TCGA_DIR / f"{cancer}_tcga_pan_can_atlas_2018_mrna_rsem.txt"
        rsem = pd.read_csv(rsem_path, sep="\t", index_col=0)
        # Drop Entrez_Gene_Id column
        if "Entrez_Gene_Id" in rsem.columns:
            rsem = rsem.drop(columns=["Entrez_Gene_Id"])
        # Median expression across samples per gene
        gene_expr = rsem.median(axis=1)
        gene_expr = gene_expr[gene_expr > 0]  # drop unexpressed
        logger.info(f"    {len(gene_expr)} genes with expression > 0")

        # Parse MAF to get gene for each C>T mutation (in order)
        maf_path = TCGA_DIR / f"{cancer}_tcga_pan_can_atlas_2018_mutations.txt"
        mut_genes = []
        ctrl_genes = []
        seen_pos = set()

        for chunk in pd.read_csv(maf_path, sep="\t", comment="#", low_memory=False, chunksize=100000):
            ct = (chunk["Reference_Allele"] == "C") & (chunk["Tumor_Seq_Allele2"] == "T")
            ga = (chunk["Reference_Allele"] == "G") & (chunk["Tumor_Seq_Allele2"] == "A")
            sub = chunk[ct | ga].copy()
            if len(sub) == 0:
                continue
            chrom = sub["Chromosome"].astype(str)
            if not chrom.str.startswith("chr").any():
                chrom = "chr" + chrom
            sub["chrom"] = chrom
            sub["pos"] = sub["Start_Position"].astype(int) - 1
            for _, row in sub.iterrows():
                key = (row["chrom"], row["pos"])
                if key not in seen_pos:
                    seen_pos.add(key)
                    mut_genes.append(row.get("Hugo_Symbol", ""))

        n_mut = int(d["mut_mask"].sum())
        n_ctrl = int(d["ctrl_mask"].sum())

        # Mutations are first n_mut rows; controls are next n_ctrl
        # We only have gene info for mutations. For controls, gene is unknown (matched from same gene's exons).
        # The original scoring assigns controls from the same gene. So controls for a gene
        # roughly correspond to the same gene. However, we don't have a direct gene-to-control mapping.
        # Instead: assign expression quartile only to MUTATIONS based on their gene,
        # and use ALL controls as the reference for each quartile.

        if len(mut_genes) < n_mut:
            logger.warning(f"    Gene count mismatch: {len(mut_genes)} genes vs {n_mut} mutations")
            mut_genes = mut_genes + [""] * (n_mut - len(mut_genes))
        elif len(mut_genes) > n_mut:
            mut_genes = mut_genes[:n_mut]

        # Map mutation genes to expression
        gene_expr_dict = gene_expr.to_dict()
        mut_expr = np.array([gene_expr_dict.get(g, np.nan) for g in mut_genes])
        has_expr = ~np.isnan(mut_expr)
        logger.info(f"    {has_expr.sum()}/{n_mut} mutations mapped to expression")

        if has_expr.sum() < 100:
            logger.warning(f"    Too few mutations with expression. Skipping {cancer}.")
            results[cancer] = {"status": "skipped", "reason": "too few mutations with expression"}
            continue

        # Expression quartiles (on mutations with expression)
        expr_vals = mut_expr[has_expr]
        q25, q50, q75 = np.percentile(expr_vals, [25, 50, 75])

        cancer_result = {"quartile_thresholds": [float(q25), float(q50), float(q75)]}

        # Build full-length masks for quartile membership (only mutations, indexed into full array)
        mut_indices = np.where(d["mut_mask"])[0]
        ctrl_mask_full = d["ctrl_mask"]

        for score_name, scores in [("neural", d["nn_binary"]), ("xgb", d["xgb_scores"])]:
            quartile_results = {}
            for qi, (lo, hi, label) in enumerate([
                (0, q25, "Q1_low"),
                (q25, q50, "Q2"),
                (q50, q75, "Q3"),
                (q75, np.inf, "Q4_high"),
            ]):
                # Mutations in this quartile
                if qi == 0:
                    in_q = has_expr & (mut_expr <= hi)
                elif qi == 3:
                    in_q = has_expr & (mut_expr > lo)
                else:
                    in_q = has_expr & (mut_expr > lo) & (mut_expr <= hi)

                # Map to full array mask
                q_mut_full = np.zeros(len(scores), dtype=bool)
                q_mut_full[mut_indices[in_q]] = True

                n_q_mut = q_mut_full.sum()
                n_q_ctrl = ctrl_mask_full.sum()
                if n_q_mut < 10:
                    continue

                pooled = np.concatenate([scores[q_mut_full], scores[ctrl_mask_full]])
                enr = {}
                for pct in [50, 75, 90, 95]:
                    enr[f"p{pct}"] = compute_or_fisher(scores, q_mut_full, ctrl_mask_full, pct,
                                                        pooled_scores=pooled)
                enr["n_mut"] = int(n_q_mut)
                enr["n_ctrl"] = int(n_q_ctrl)
                quartile_results[label] = enr

            cancer_result[score_name] = quartile_results

        results[cancer] = cancer_result

        # Log
        logger.info(f"    Expression quartile OR@p90:")
        for score_name in ["neural", "xgb"]:
            qr = cancer_result.get(score_name, {})
            vals = [qr.get(q, {}).get("p90", {}).get("OR", float("nan"))
                    for q in ["Q1_low", "Q2", "Q3", "Q4_high"]]
            logger.info(f"      {score_name:>8}: Q1={vals[0]:.3f}  Q2={vals[1]:.3f}  Q3={vals[2]:.3f}  Q4={vals[3]:.3f}")

    return results


# ============================================================================
# Experiment 1e: StructOnly Ablation
# ============================================================================
def exp_1e_structonly_ablation(all_data, model):
    """Re-score TCGA with neural model but with motif features zeroed (cols 0:24).

    Keep structure delta (cols 24:31) and loop geometry (cols 31:40), zero motif (cols 0:24).
    """
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 1e: StructOnly Ablation (Neural)")
    logger.info("=" * 70)

    results = {}
    for cancer in CANCERS:
        t0 = time.time()
        logger.info(f"\n  Scoring {cancer.upper()} with motif-zeroed features...")

        d = all_data[cancer]
        hand = d["hand"].copy()
        # Zero out motif features (cols 0:24)
        hand_struct = hand.copy()
        hand_struct[:, :24] = 0.0

        # Load RNA-FM embeddings
        emb_path = EMB_DIR / f"rnafm_tcga_{cancer}.pt"
        emb_data = torch.load(emb_path, weights_only=False)
        pooled_orig = emb_data["pooled_orig"].numpy()
        pooled_edited = emb_data["pooled_edited"].numpy()
        edit_delta = pooled_edited - pooled_orig

        n = d["n"]
        # Build 1320-dim features with struct-only hand features
        X_struct = np.concatenate([
            pooled_orig[:n], edit_delta[:n], hand_struct[:n]
        ], axis=1).astype(np.float32)

        # Also build full features for comparison (should match pre-computed)
        X_full = np.concatenate([
            pooled_orig[:n], edit_delta[:n], hand[:n]
        ], axis=1).astype(np.float32)

        # Score struct-only
        struct_scores = score_neural(model, X_struct, DEVICE, BATCH_SIZE)
        full_scores = score_neural(model, X_full, DEVICE, BATCH_SIZE)

        # Compute enrichment
        mut_mask = d["mut_mask"]
        ctrl_mask = d["ctrl_mask"]

        struct_enr = compute_enrichment_full(struct_scores, mut_mask, ctrl_mask)
        full_enr = compute_enrichment_full(full_scores, mut_mask, ctrl_mask)
        xgb_enr = compute_enrichment_full(d["xgb_scores"], mut_mask, ctrl_mask)

        results[cancer] = {
            "neural_structonly": struct_enr,
            "neural_full": full_enr,
            "xgb": xgb_enr,
        }

        logger.info(f"    StructOnly OR@p90={struct_enr['p90']['OR']:.3f}  "
                    f"Full OR@p90={full_enr['p90']['OR']:.3f}  "
                    f"XGB OR@p90={xgb_enr['p90']['OR']:.3f}  ({time.time()-t0:.1f}s)")

        del emb_data, pooled_orig, pooled_edited, edit_delta, X_struct, X_full
        gc.collect()

    # Summary table
    logger.info(f"\n{'Cancer':>8} | {'NN StructOnly':>14} {'NN Full':>10} {'XGB':>8} | {'NN Struct p':>12} {'NN Full p':>12}")
    logger.info("-" * 75)
    for cancer in CANCERS:
        r = results.get(cancer, {})
        s_or = r.get("neural_structonly", {}).get("p90", {}).get("OR", float("nan"))
        s_p = r.get("neural_structonly", {}).get("p90", {}).get("p", float("nan"))
        f_or = r.get("neural_full", {}).get("p90", {}).get("OR", float("nan"))
        f_p = r.get("neural_full", {}).get("p90", {}).get("p", float("nan"))
        x_or = r.get("xgb", {}).get("p90", {}).get("OR", float("nan"))
        logger.info(f"{cancer:>8} | {s_or:>14.3f} {f_or:>10.3f} {x_or:>8.3f} | {s_p:>12.2e} {f_p:>12.2e}")

    return results


# ============================================================================
# Experiment 1f: Per-Sample APOBEC Signature (BRCA only)
# ============================================================================
def exp_1f_per_sample_brca(all_data):
    """BRCA: compute per-sample TC fraction, split into quartiles, score separately."""
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 1f: Per-Sample APOBEC Signature (BRCA)")
    logger.info("=" * 70)

    maf_path = TCGA_DIR / "brca_tcga_pan_can_atlas_2018_mutations.txt"
    if not maf_path.exists():
        logger.error(f"  BRCA MAF not found: {maf_path}")
        return {"status": "skipped", "reason": "MAF not found"}

    d = all_data["brca"]

    # Step 1: Parse MAF for per-sample TC mutation fraction
    logger.info("  Step 1: Computing per-sample TC fraction from MAF...")
    sample_ct_total = {}
    sample_tc_count = {}

    for chunk in pd.read_csv(maf_path, sep="\t", comment="#", low_memory=False, chunksize=100000):
        if "Variant_Type" in chunk.columns:
            chunk = chunk[chunk["Variant_Type"] == "SNP"]
        ct_plus = (chunk["Reference_Allele"] == "C") & (chunk["Tumor_Seq_Allele2"] == "T")
        ga_minus = (chunk["Reference_Allele"] == "G") & (chunk["Tumor_Seq_Allele2"] == "A")
        sub = chunk[ct_plus | ga_minus]

        for _, row in sub.iterrows():
            sample = row["Tumor_Sample_Barcode"]
            ref = row["Reference_Allele"]
            sample_ct_total[sample] = sample_ct_total.get(sample, 0) + 1

            context = str(row.get("CONTEXT", ""))
            if ref == "C" and len(context) >= 3:
                center_idx = len(context) // 2
                if center_idx > 0 and context[center_idx - 1].upper() == "T":
                    sample_tc_count[sample] = sample_tc_count.get(sample, 0) + 1
            elif ref == "G" and len(context) >= 3:
                center_idx = len(context) // 2
                if center_idx < len(context) - 1 and context[center_idx + 1].upper() == "A":
                    sample_tc_count[sample] = sample_tc_count.get(sample, 0) + 1

    # TC fraction per sample (minimum 5 mutations)
    sample_tc_frac = {}
    for sample in sample_ct_total:
        total = sample_ct_total[sample]
        tc = sample_tc_count.get(sample, 0)
        if total >= 5:
            sample_tc_frac[sample] = tc / total

    tc_fracs = np.array(list(sample_tc_frac.values()))
    q25_frac = np.percentile(tc_fracs, 25)
    q75_frac = np.percentile(tc_fracs, 75)

    apobec_high = {s for s, f in sample_tc_frac.items() if f >= q75_frac}
    apobec_low = {s for s, f in sample_tc_frac.items() if f <= q25_frac}

    logger.info(f"  {len(sample_tc_frac)} samples with >=5 C>T mutations")
    logger.info(f"  TC fraction: Q25={q25_frac:.3f}, Q75={q75_frac:.3f}")
    logger.info(f"  APOBEC-high (top quartile): {len(apobec_high)} samples")
    logger.info(f"  APOBEC-low (bottom quartile): {len(apobec_low)} samples")

    # Step 2: Re-parse MAF to get sample per unique mutation (same dedup order)
    logger.info("  Step 2: Matching mutations to samples...")
    mutation_rows = []
    for chunk in pd.read_csv(maf_path, sep="\t", comment="#", low_memory=False, chunksize=100000):
        if "Variant_Type" in chunk.columns:
            chunk = chunk[chunk["Variant_Type"] == "SNP"]
        ct_plus = (chunk["Reference_Allele"] == "C") & (chunk["Tumor_Seq_Allele2"] == "T")
        ga_minus = (chunk["Reference_Allele"] == "G") & (chunk["Tumor_Seq_Allele2"] == "A")
        sub = chunk[ct_plus | ga_minus].copy()
        if len(sub) > 0:
            chrom = sub["Chromosome"].astype(str)
            if not chrom.str.startswith("chr").any():
                chrom = "chr" + chrom
            sub = sub.copy()
            sub["chrom"] = chrom
            sub["pos"] = sub["Start_Position"].astype(int) - 1
            mutation_rows.append(sub[["chrom", "pos", "Tumor_Sample_Barcode"]].copy())

    all_muts = pd.concat(mutation_rows, ignore_index=True)
    # Keep first occurrence per position (matching original dedup)
    all_muts_dedup = all_muts.drop_duplicates(subset=["chrom", "pos"], keep="first")
    mutation_samples = all_muts_dedup["Tumor_Sample_Barcode"].values

    n_mut = int(d["mut_mask"].sum())
    logger.info(f"  Parsed {len(mutation_samples)} unique C>T mutations, expected {n_mut}")

    if len(mutation_samples) > n_mut:
        mutation_samples = mutation_samples[:n_mut]
    elif len(mutation_samples) < n_mut:
        # Pad with empty
        mutation_samples = np.concatenate([mutation_samples, np.array([""] * (n_mut - len(mutation_samples)))])

    # Step 3: Assign mutations to high/low
    mut_in_high = np.zeros(n_mut, dtype=bool)
    mut_in_low = np.zeros(n_mut, dtype=bool)
    for i in range(n_mut):
        sample = mutation_samples[i]
        if sample in apobec_high:
            mut_in_high[i] = True
        elif sample in apobec_low:
            mut_in_low[i] = True

    logger.info(f"  APOBEC-high mutations: {mut_in_high.sum()}")
    logger.info(f"  APOBEC-low mutations: {mut_in_low.sum()}")

    # Step 4: Compute enrichment for each group
    # Mutations are the first n_mut rows in the score arrays
    ctrl_mask_full = d["ctrl_mask"]
    mut_indices = np.where(d["mut_mask"])[0]

    results = {
        "n_samples": len(sample_tc_frac),
        "q25_tc_frac": float(q25_frac),
        "q75_tc_frac": float(q75_frac),
        "n_apobec_high_samples": len(apobec_high),
        "n_apobec_low_samples": len(apobec_low),
        "n_apobec_high_muts": int(mut_in_high.sum()),
        "n_apobec_low_muts": int(mut_in_low.sum()),
    }

    for score_name, scores in [("neural", d["nn_binary"]), ("xgb", d["xgb_scores"])]:
        for subset_name, mut_sub_mask in [("high", mut_in_high), ("low", mut_in_low)]:
            # Build full-length mask
            q_mut_full = np.zeros(len(scores), dtype=bool)
            q_mut_full[mut_indices[mut_sub_mask]] = True

            if q_mut_full.sum() < 10:
                continue

            pooled = np.concatenate([scores[q_mut_full], scores[ctrl_mask_full]])
            enr = {}
            for pct in [50, 75, 90, 95]:
                enr[f"p{pct}"] = compute_or_fisher(scores, q_mut_full, ctrl_mask_full, pct,
                                                    pooled_scores=pooled)
            enr["n_mut"] = int(q_mut_full.sum())
            enr["n_ctrl"] = int(ctrl_mask_full.sum())
            results[f"{score_name}_{subset_name}"] = enr

    # Log
    for pct in [90]:
        nn_h = results.get("neural_high", {}).get(f"p{pct}", {}).get("OR", float("nan"))
        nn_l = results.get("neural_low", {}).get(f"p{pct}", {}).get("OR", float("nan"))
        xgb_h = results.get("xgb_high", {}).get(f"p{pct}", {}).get("OR", float("nan"))
        xgb_l = results.get("xgb_low", {}).get(f"p{pct}", {}).get("OR", float("nan"))
        logger.info(f"  OR@p{pct}: NN high={nn_h:.3f}, NN low={nn_l:.3f} | "
                    f"XGB high={xgb_h:.3f}, XGB low={xgb_l:.3f}")

    return results


# ============================================================================
# Main
# ============================================================================
def main():
    t_start = time.time()
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Cancers: {CANCERS}")

    # Load neural model first (needed for re-scoring with correct hand features)
    logger.info("\nLoading Phase3 neural model...")
    model = Phase3Model()
    state = torch.load(NEURAL_MODEL_PATH, weights_only=True, map_location="cpu")
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    logger.info(f"  Model loaded on {DEVICE}")

    # Load all data and re-score with correct hand features
    logger.info("\nLoading data and re-scoring all cancers with correct hand features...")
    all_data = {}
    for cancer in CANCERS:
        logger.info(f"  Loading and scoring {cancer}...")
        try:
            all_data[cancer] = load_cancer_data(cancer, model, DEVICE)
            logger.info(f"    n={all_data[cancer]['n']:,}, "
                       f"mut={all_data[cancer]['mut_mask'].sum():,}, "
                       f"ctrl={all_data[cancer]['ctrl_mask'].sum():,}, "
                       f"nn_mean_mut={all_data[cancer]['nn_binary'][all_data[cancer]['mut_mask']].mean():.4f}, "
                       f"nn_mean_ctrl={all_data[cancer]['nn_binary'][all_data[cancer]['ctrl_mask']].mean():.4f}")
        except Exception as e:
            logger.error(f"    Failed to load {cancer}: {e}")

    # Run experiments
    all_results = {}

    # 1b: TC-Stratified
    all_results["exp_1b_tc_stratified"] = exp_1b_tc_stratified(all_data)

    # 1c: CpG Stratification
    all_results["exp_1c_cpg_stratification"] = exp_1c_cpg_stratification(all_data)

    # 1d: Gene Expression Stratification
    all_results["exp_1d_expression"] = exp_1d_expression_stratification(all_data)

    # 1e: StructOnly Ablation
    all_results["exp_1e_structonly"] = exp_1e_structonly_ablation(all_data, model)

    # 1f: Per-Sample APOBEC Signature (BRCA)
    all_results["exp_1f_per_sample_brca"] = exp_1f_per_sample_brca(all_data)

    # Save results
    output_path = OUTPUT_DIR / "nn_controls_1b_1f.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nResults saved to {output_path}")

    # ========================================================================
    # Final Summary Tables
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("FINAL SUMMARY: NN vs XGB for all confound controls")
    logger.info("=" * 80)

    # 1b Summary
    logger.info("\n--- 1b: TC-Stratified OR@p90 ---")
    logger.info(f"{'Cancer':>8} | {'NN_TC':>8} {'NN_nTC':>8} | {'XGB_TC':>8} {'XGB_nTC':>8} | {'NN TC>1?':>8} {'XGB TC>1?':>9}")
    logger.info("-" * 70)
    r1b = all_results.get("exp_1b_tc_stratified", {})
    for cancer in CANCERS:
        rc = r1b.get(cancer, {})
        nn_tc = rc.get("neural_TC", {}).get("p90", {}).get("OR", float("nan"))
        nn_ntc = rc.get("neural_nonTC", {}).get("p90", {}).get("OR", float("nan"))
        xgb_tc = rc.get("xgb_TC", {}).get("p90", {}).get("OR", float("nan"))
        xgb_ntc = rc.get("xgb_nonTC", {}).get("p90", {}).get("OR", float("nan"))
        nn_sig = "YES" if nn_tc > 1.0 else "no"
        xgb_sig = "YES" if xgb_tc > 1.0 else "no"
        logger.info(f"{cancer:>8} | {nn_tc:>8.3f} {nn_ntc:>8.3f} | {xgb_tc:>8.3f} {xgb_ntc:>8.3f} | {nn_sig:>8} {xgb_sig:>9}")

    # 1c Summary
    logger.info("\n--- 1c: CpG Stratification OR@p90 (TC+nonCpG = 'pure APOBEC') ---")
    logger.info(f"{'Cancer':>8} | {'NN TC+nCpG':>11} {'NN TC+CpG':>10} | {'XGB TC+nCpG':>12} {'XGB TC+CpG':>11}")
    logger.info("-" * 60)
    r1c = all_results.get("exp_1c_cpg_stratification", {})
    for cancer in CANCERS:
        rc = r1c.get(cancer, {})
        nn_tn = rc.get("neural_TC_nonCpG", {}).get("p90", {}).get("OR", float("nan"))
        nn_tc = rc.get("neural_TC_CpG", {}).get("p90", {}).get("OR", float("nan"))
        xgb_tn = rc.get("xgb_TC_nonCpG", {}).get("p90", {}).get("OR", float("nan"))
        xgb_tc = rc.get("xgb_TC_CpG", {}).get("p90", {}).get("OR", float("nan"))
        logger.info(f"{cancer:>8} | {nn_tn:>11.3f} {nn_tc:>10.3f} | {xgb_tn:>12.3f} {xgb_tc:>11.3f}")

    # 1d Summary
    logger.info("\n--- 1d: Expression Quartile OR@p90 ---")
    r1d = all_results.get("exp_1d_expression", {})
    if isinstance(r1d, dict) and r1d.get("status") != "skipped":
        for cancer in CANCERS:
            rc = r1d.get(cancer, {})
            if isinstance(rc, dict) and rc.get("status") != "skipped":
                for score_name in ["neural", "xgb"]:
                    qr = rc.get(score_name, {})
                    vals = [qr.get(q, {}).get("p90", {}).get("OR", float("nan"))
                            for q in ["Q1_low", "Q2", "Q3", "Q4_high"]]
                    logger.info(f"  {cancer:>8} {score_name:>8}: Q1={vals[0]:.3f}  Q2={vals[1]:.3f}  "
                               f"Q3={vals[2]:.3f}  Q4={vals[3]:.3f}")

    # 1e Summary
    logger.info("\n--- 1e: StructOnly Ablation OR@p90 ---")
    logger.info(f"{'Cancer':>8} | {'NN Struct':>10} {'NN Full':>8} {'XGB':>8} | {'Struct retains signal?':>22}")
    logger.info("-" * 65)
    r1e = all_results.get("exp_1e_structonly", {})
    for cancer in CANCERS:
        rc = r1e.get(cancer, {})
        s_or = rc.get("neural_structonly", {}).get("p90", {}).get("OR", float("nan"))
        f_or = rc.get("neural_full", {}).get("p90", {}).get("OR", float("nan"))
        x_or = rc.get("xgb", {}).get("p90", {}).get("OR", float("nan"))
        sig = "YES" if s_or > 1.0 else "no"
        logger.info(f"{cancer:>8} | {s_or:>10.3f} {f_or:>8.3f} {x_or:>8.3f} | {sig:>22}")

    # 1f Summary
    logger.info("\n--- 1f: Per-Sample APOBEC Signature (BRCA) OR@p90 ---")
    r1f = all_results.get("exp_1f_per_sample_brca", {})
    for pct in [50, 75, 90, 95]:
        nn_h = r1f.get("neural_high", {}).get(f"p{pct}", {}).get("OR", float("nan"))
        nn_l = r1f.get("neural_low", {}).get(f"p{pct}", {}).get("OR", float("nan"))
        xgb_h = r1f.get("xgb_high", {}).get(f"p{pct}", {}).get("OR", float("nan"))
        xgb_l = r1f.get("xgb_low", {}).get(f"p{pct}", {}).get("OR", float("nan"))
        logger.info(f"  @p{pct}: NN high={nn_h:.3f} low={nn_l:.3f} | XGB high={xgb_h:.3f} low={xgb_l:.3f}")

    elapsed = time.time() - t_start
    logger.info(f"\nTotal time: {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
