#!/usr/bin/env python3
"""Reproduce docx Section 1 (TCGA somatic mutation enrichment) with the neural network.

Covers:
  1a. OR@p50/p75/p90/p95 for all 10 TCGA cancers (binary head).
  1g. Per-enzyme adapter OR@p90/p95 for all 10 cancers (A3A, A3B, A3G, A3A_A3G, Neither).
  1b. TC-stratified OR@p90/p95 for binary + 5 adapters, all 10 cancers.
  1c. CpG stratification (TC+nonCpG, TC+CpG, nonTC+nonCpG, nonTC+CpG) for binary + 5 adapters.
  1d. Gene expression quartile OR for binary + 5 adapters (4 cancers with RSEM).

Reuses cached assets only — no re-extraction, no ViennaRNA re-fold:
  - experiments/multi_enzyme/outputs/tcga_gnomad/raw_scores/{cancer}_scores.csv
  - data/processed/multi_enzyme/tcga_hand_features/{cancer}_hand40.npy
  - data/processed/multi_enzyme/embeddings/rnafm_tcga_{cancer}.pt
  - experiments/multi_enzyme/outputs/phase3_neural_true_validation/phase3_neural_full.pt

Run:
    conda run -n quris python experiments/multi_enzyme/exp_reproduce_section_1.py
"""

import gc
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

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "multi_enzyme" / "outputs" / "section1_nn_reproduction"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(OUTPUT_DIR / "run.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
EMB_DIR = DATA_DIR / "processed" / "multi_enzyme" / "embeddings"
HAND_DIR = DATA_DIR / "processed" / "multi_enzyme" / "tcga_hand_features"
RAW_SCORES_DIR = PROJECT_ROOT / "experiments" / "multi_enzyme" / "outputs" / "tcga_gnomad" / "raw_scores"
TCGA_DIR = DATA_DIR / "raw" / "tcga"
NEURAL_MODEL_PATH = (
    PROJECT_ROOT / "experiments" / "multi_enzyme" / "outputs"
    / "phase3_neural_true_validation" / "phase3_neural_full.pt"
)

CANCERS = ["blca", "brca", "cesc", "lusc", "hnsc", "skcm", "esca", "stad", "lihc", "coadread"]
CANCERS_WITH_EXPR = ["blca", "brca", "cesc", "lusc"]

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else torch.device("cpu")
)
BATCH_SIZE = 1024
SEED = 42

ENZYMES = ["A3A", "A3B", "A3G", "A3A_A3G", "Neither"]
D_INPUT = 1320
D_SHARED = 128
N_ENZYMES_CLS = 6


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
            nn.Linear(D_SHARED, 64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, N_ENZYMES_CLS),
        )

    def forward(self, x):
        shared = self.shared_encoder(x)
        binary_logit = self.binary_head(shared).squeeze(-1)
        per_enzyme_logits = [self.enzyme_adapters[enz](shared).squeeze(-1) for enz in ENZYMES]
        return binary_logit, per_enzyme_logits, None, shared


def score_all_heads(model, X, device, batch_size):
    """Return dict of {head_name: np.ndarray of sigmoid scores}. 6 heads total."""
    out = {"binary": [], **{enz: [] for enz in ENZYMES}}
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            end = min(start + batch_size, len(X))
            x = torch.from_numpy(X[start:end].astype(np.float32)).to(device)
            bl, per_enz, _, _ = model(x)
            out["binary"].append(torch.sigmoid(bl).cpu().numpy())
            for enz, logit in zip(ENZYMES, per_enz):
                out[enz].append(torch.sigmoid(logit).cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def compute_or(scores, mut_mask, ctrl_mask, pct, pool_mask=None):
    """OR at percentile threshold; threshold computed on the pooled subset (pool_mask)."""
    if pool_mask is None:
        pool_mask = mut_mask | ctrl_mask
    pooled = scores[pool_mask]
    if len(pooled) < 20:
        return {"OR": float("nan"), "p": 1.0, "threshold": float("nan"),
                "mut_above": 0, "mut_below": 0, "ctrl_above": 0, "ctrl_below": 0}
    thresh = float(np.percentile(pooled, pct))
    mut_s = scores[mut_mask]
    ctrl_s = scores[ctrl_mask]
    ma = int((mut_s >= thresh).sum()); mb = int((mut_s < thresh).sum())
    ca = int((ctrl_s >= thresh).sum()); cb = int((ctrl_s < thresh).sum())
    if all(x > 0 for x in [ma, mb, ca, cb]):
        OR, pv = stats.fisher_exact([[ma, mb], [ca, cb]])
    else:
        OR, pv = float("nan"), 1.0
    return {"OR": float(OR), "p": float(pv), "threshold": thresh,
            "mut_above": ma, "mut_below": mb, "ctrl_above": ca, "ctrl_below": cb}


def enrichment_block(scores, mut_mask, ctrl_mask, pcts=(50, 75, 90, 95)):
    """Full OR block: percentiles + means + Mann-Whitney."""
    res = {"n_mut": int(mut_mask.sum()), "n_ctrl": int(ctrl_mask.sum())}
    if mut_mask.sum() == 0 or ctrl_mask.sum() == 0:
        return res
    res["mean_mut"] = float(scores[mut_mask].mean())
    res["mean_ctrl"] = float(scores[ctrl_mask].mean())
    try:
        _, p_mw = stats.mannwhitneyu(scores[mut_mask], scores[ctrl_mask], alternative="greater")
        res["mw_p"] = float(p_mw)
    except Exception:
        res["mw_p"] = 1.0
    for pct in pcts:
        res[f"p{pct}"] = compute_or(scores, mut_mask, ctrl_mask, pct)
    return res


def stratified_or(scores, mut_mask, ctrl_mask, strata_mask, pcts=(90, 95)):
    """OR@pct within a stratum; thresholds on the stratum pool."""
    mut_s_mask = mut_mask & strata_mask
    ctrl_s_mask = ctrl_mask & strata_mask
    res = {"n_mut": int(mut_s_mask.sum()), "n_ctrl": int(ctrl_s_mask.sum())}
    if mut_s_mask.sum() < 10 or ctrl_s_mask.sum() < 10:
        return res
    pool_mask = strata_mask
    for pct in pcts:
        res[f"p{pct}"] = compute_or(scores, mut_s_mask, ctrl_s_mask, pct, pool_mask=pool_mask)
    return res


def load_and_score_cancer(cancer, model):
    """Load cached assets for a cancer and score with NN (binary + 5 adapters)."""
    t0 = time.time()
    raw_path = RAW_SCORES_DIR / f"{cancer}_scores.csv"
    hand_path = HAND_DIR / f"{cancer}_hand40.npy"
    emb_path = EMB_DIR / f"rnafm_tcga_{cancer}.pt"

    raw = pd.read_csv(raw_path)
    hand = np.load(hand_path)
    emb = torch.load(emb_path, weights_only=False, map_location="cpu")

    n = len(raw)
    assert hand.shape[0] == n, f"{cancer}: hand={hand.shape[0]} vs raw={n}"
    assert emb["pooled_orig"].shape[0] == n, f"{cancer}: emb={emb['pooled_orig'].shape[0]} vs raw={n}"

    types = raw["type"].values
    mut_mask = types == "mutation"
    ctrl_mask = types == "control"
    xgb_scores = raw["score"].values.astype(np.float32)

    # Motif 24-dim layout from src/data/apobec_feature_extraction.py:
    #   [0:4]   feat_5p  one-hot ["UC","CC","AC","GC"]   (5' dinuc with oriented C)
    #   [4:8]   feat_3p  one-hot ["CA","CG","CU","CC"]   (3' dinuc with oriented C)
    #   [8:12]  m2       one-hot A,C,G,U
    #   [12:16] m1       one-hot A,C,G,U
    #   [16:20] p1       one-hot A,C,G,U
    #   [20:24] p2       one-hot A,C,G,U
    # Oriented TC  = UC at 5'  → col 0 (equivalently m1=U, col 15)
    # Oriented CpG = CG at 3'  → col 5 (equivalently p1=G, col 18)
    # The raw_scores tc_context column uses raw-strand orientation and disagrees with
    # the strand-aware definition on ~32% of rows (- strand mutations) — we use the
    # hand-feature version because the neural model was trained on oriented sequences.
    tc_mask = hand[:, 0] == 1.0
    cpg_mask = hand[:, 5] == 1.0
    tc_context = tc_mask.astype(np.int32)

    pooled_orig = emb["pooled_orig"].numpy().astype(np.float32)
    pooled_edited = emb["pooled_edited"].numpy().astype(np.float32)
    edit_delta = pooled_edited - pooled_orig
    del emb

    X = np.concatenate([pooled_orig, edit_delta, hand], axis=1).astype(np.float32)
    del pooled_orig, pooled_edited, edit_delta
    gc.collect()

    logger.info(f"  {cancer}: n={n:,}  mut={mut_mask.sum():,}  ctrl={ctrl_mask.sum():,}  "
                f"TC%={tc_mask.mean():.3f}  CpG%={cpg_mask.mean():.3f}")

    nn_scores = score_all_heads(model, X, DEVICE, BATCH_SIZE)
    del X
    gc.collect()

    logger.info(f"  {cancer}: scored all 6 heads in {time.time()-t0:.1f}s")

    return {
        "n": n,
        "mut_mask": mut_mask,
        "ctrl_mask": ctrl_mask,
        "tc": tc_context,
        "cpg": cpg_mask,
        "xgb": xgb_scores,
        "hand": hand,
        "nn": nn_scores,
    }


def compute_section_1a_1g(d, cancer):
    """Overall + per-head OR@p50/p75/p90/p95. Also XGB for comparison."""
    out = {}
    out["xgb"] = enrichment_block(d["xgb"], d["mut_mask"], d["ctrl_mask"])
    out["nn_binary"] = enrichment_block(d["nn"]["binary"], d["mut_mask"], d["ctrl_mask"])
    for enz in ENZYMES:
        out[f"nn_{enz}"] = enrichment_block(d["nn"][enz], d["mut_mask"], d["ctrl_mask"])
    return out


def compute_section_1b_1c(d, cancer):
    """TC-stratified and TC×CpG-stratified OR@p90 for all heads."""
    out = {}
    tc1 = d["tc"] == 1
    tc0 = d["tc"] == 0
    cpg = d["cpg"]

    strata = {
        "TC": tc1,
        "nonTC": tc0,
        "TC_nonCpG": tc1 & (~cpg),
        "TC_CpG": tc1 & cpg,
        "nonTC_nonCpG": tc0 & (~cpg),
        "nonTC_CpG": tc0 & cpg,
    }

    for head_name, scores in [("xgb", d["xgb"]),
                              ("nn_binary", d["nn"]["binary"])] + [(f"nn_{e}", d["nn"][e]) for e in ENZYMES]:
        out[head_name] = {}
        for s_name, s_mask in strata.items():
            out[head_name][s_name] = stratified_or(scores, d["mut_mask"], d["ctrl_mask"], s_mask,
                                                    pcts=(50, 75, 90, 95))
    return out


def compute_section_1d(d, cancer):
    """Expression quartile enrichment. Only for cancers with RSEM."""
    if cancer not in CANCERS_WITH_EXPR:
        return {"status": "skipped", "reason": "no RSEM"}

    # Parse MAF to recover gene per unique mutation position (same dedup as cache build).
    maf_path = TCGA_DIR / f"{cancer}_tcga_pan_can_atlas_2018_mutations.txt"
    mut_gene_in_order = []
    seen = set()
    for chunk in pd.read_csv(maf_path, sep="\t", comment="#", low_memory=False, chunksize=100000):
        ct = (chunk["Reference_Allele"] == "C") & (chunk["Tumor_Seq_Allele2"] == "T")
        ga = (chunk["Reference_Allele"] == "G") & (chunk["Tumor_Seq_Allele2"] == "A")
        sub = chunk[ct | ga]
        if len(sub) == 0:
            continue
        chrom = sub["Chromosome"].astype(str)
        if not chrom.str.startswith("chr").any():
            chrom = "chr" + chrom
        for c, p, g in zip(chrom, sub["Start_Position"].astype(int) - 1, sub["Hugo_Symbol"].fillna("")):
            key = (c, int(p))
            if key not in seen:
                seen.add(key)
                mut_gene_in_order.append(g)

    n_mut = int(d["mut_mask"].sum())
    if len(mut_gene_in_order) >= n_mut:
        mut_genes = mut_gene_in_order[:n_mut]
    else:
        mut_genes = mut_gene_in_order + [""] * (n_mut - len(mut_gene_in_order))

    # Gene medians
    rsem_path = TCGA_DIR / f"{cancer}_tcga_pan_can_atlas_2018_mrna_rsem.txt"
    rsem = pd.read_csv(rsem_path, sep="\t", index_col=0)
    if "Entrez_Gene_Id" in rsem.columns:
        rsem = rsem.drop(columns=["Entrez_Gene_Id"])
    gene_expr = rsem.median(axis=1)
    gene_expr = gene_expr[gene_expr > 0]
    expr_dict = gene_expr.to_dict()

    mut_expr = np.array([expr_dict.get(g, np.nan) for g in mut_genes])
    has_expr = ~np.isnan(mut_expr)
    if has_expr.sum() < 200:
        return {"status": "skipped", "reason": f"only {has_expr.sum()} mutations mapped"}

    vals = mut_expr[has_expr]
    q25, q50, q75 = [float(x) for x in np.percentile(vals, [25, 50, 75])]

    mut_indices = np.where(d["mut_mask"])[0]
    ctrl_mask_full = d["ctrl_mask"]

    quartiles = [
        ("Q1_low", (mut_expr > 0) & (mut_expr <= q25)),
        ("Q2", (mut_expr > q25) & (mut_expr <= q50)),
        ("Q3", (mut_expr > q50) & (mut_expr <= q75)),
        ("Q4_high", mut_expr > q75),
    ]

    out = {"thresholds": [q25, q50, q75], "n_mut_with_expr": int(has_expr.sum())}
    head_iter = [("xgb", d["xgb"]), ("nn_binary", d["nn"]["binary"])] + \
                [(f"nn_{e}", d["nn"][e]) for e in ENZYMES]
    for head_name, scores in head_iter:
        qres = {}
        for q_label, q_mut_flag in quartiles:
            q_mut_flag = q_mut_flag & has_expr
            q_mut_mask = np.zeros(len(scores), dtype=bool)
            q_mut_mask[mut_indices[q_mut_flag]] = True
            if q_mut_mask.sum() < 30:
                continue
            # Pool for threshold: quartile mutations + all controls
            pool_mask = q_mut_mask | ctrl_mask_full
            entry = {"n_mut": int(q_mut_mask.sum()), "n_ctrl": int(ctrl_mask_full.sum())}
            for pct in (50, 75, 90, 95):
                entry[f"p{pct}"] = compute_or(scores, q_mut_mask, ctrl_mask_full, pct, pool_mask=pool_mask)
            qres[q_label] = entry
        out[head_name] = qres
    return out


def print_section_1a_1g(all_results):
    """Print the two headline tables."""
    logger.info("\n" + "=" * 88)
    logger.info("SECTION 1a — NN Binary Head: OR@p90 / OR@p95 (10 cancers)")
    logger.info("=" * 88)
    header = f"{'Cancer':>9} | {'n_mut':>8}  {'OR@p50':>8} {'OR@p75':>8} {'OR@p90':>8} {'OR@p95':>8} | {'XGB p90':>8} {'XGB p95':>8}"
    logger.info(header)
    logger.info("-" * len(header))
    for c in CANCERS:
        r = all_results[c]["section_1a_1g"]
        nn = r["nn_binary"]
        xgb = r["xgb"]
        if "p90" not in nn:
            continue
        logger.info(
            f"{c:>9} | {nn['n_mut']:>8}  "
            f"{nn['p50']['OR']:>8.3f} {nn['p75']['OR']:>8.3f} {nn['p90']['OR']:>8.3f} {nn['p95']['OR']:>8.3f} | "
            f"{xgb['p90']['OR']:>8.3f} {xgb['p95']['OR']:>8.3f}"
        )

    logger.info("\n" + "=" * 110)
    logger.info("SECTION 1g — NN Per-Enzyme Adapters: OR@p90 (10 cancers)")
    logger.info("=" * 110)
    header = (f"{'Cancer':>9} | "
              f"{'Binary':>8} {'A3A':>8} {'A3B':>8} {'A3G':>8} {'A3A_A3G':>9} {'Neither':>9}")
    logger.info(header)
    logger.info("-" * len(header))
    for c in CANCERS:
        r = all_results[c]["section_1a_1g"]
        parts = [r["nn_binary"]["p90"]["OR"]] + [r[f"nn_{e}"]["p90"]["OR"] for e in ENZYMES]
        parts_str = " ".join(f"{v:>8.3f}" if i < 4 else f"{v:>9.3f}" for i, v in enumerate(parts))
        logger.info(f"{c:>9} | {parts_str}")

    logger.info("\n" + "=" * 110)
    logger.info("SECTION 1g — NN Per-Enzyme Adapters: OR@p95 (10 cancers)")
    logger.info("=" * 110)
    logger.info(header)
    logger.info("-" * len(header))
    for c in CANCERS:
        r = all_results[c]["section_1a_1g"]
        parts = [r["nn_binary"]["p95"]["OR"]] + [r[f"nn_{e}"]["p95"]["OR"] for e in ENZYMES]
        parts_str = " ".join(f"{v:>8.3f}" if i < 4 else f"{v:>9.3f}" for i, v in enumerate(parts))
        logger.info(f"{c:>9} | {parts_str}")


def print_section_1b_1c(all_results):
    """TC-stratified and CpG-stratified tables."""
    logger.info("\n" + "=" * 110)
    logger.info("SECTION 1b — TC-Stratified OR@p90: binary + Neither adapter (10 cancers)")
    logger.info("=" * 110)
    header = (f"{'Cancer':>9} | {'Bin TC':>8} {'Bin nTC':>8} | "
              f"{'Neith TC':>9} {'Neith nTC':>10} | {'A3A TC':>8} {'A3A nTC':>8} | {'A3G TC':>8} {'A3G nTC':>8}")
    logger.info(header)
    logger.info("-" * len(header))
    for c in CANCERS:
        s = all_results[c]["section_1b_1c"]
        def g(h, k):
            v = s.get(h, {}).get(k, {}).get("p90", {}).get("OR", float("nan"))
            return v
        logger.info(
            f"{c:>9} | {g('nn_binary','TC'):>8.3f} {g('nn_binary','nonTC'):>8.3f} | "
            f"{g('nn_Neither','TC'):>9.3f} {g('nn_Neither','nonTC'):>10.3f} | "
            f"{g('nn_A3A','TC'):>8.3f} {g('nn_A3A','nonTC'):>8.3f} | "
            f"{g('nn_A3G','TC'):>8.3f} {g('nn_A3G','nonTC'):>8.3f}"
        )

    logger.info("\n" + "=" * 110)
    logger.info("SECTION 1c — CpG Stratification OR@p90: binary head (10 cancers)")
    logger.info("=" * 110)
    header = (f"{'Cancer':>9} | {'TC+nCpG':>9} {'TC+CpG':>8} {'nTC+nCpG':>9} {'nTC+CpG':>9} | "
              f"{'n TC+nCpG':>10} {'n TC+CpG':>9}")
    logger.info(header)
    logger.info("-" * len(header))
    for c in CANCERS:
        s = all_results[c]["section_1b_1c"]["nn_binary"]
        def g(k):
            return s.get(k, {}).get("p90", {}).get("OR", float("nan"))
        n_tc_ncpg = s.get("TC_nonCpG", {}).get("n_mut", 0)
        n_tc_cpg = s.get("TC_CpG", {}).get("n_mut", 0)
        logger.info(
            f"{c:>9} | {g('TC_nonCpG'):>9.3f} {g('TC_CpG'):>8.3f} {g('nonTC_nonCpG'):>9.3f} {g('nonTC_CpG'):>9.3f} | "
            f"{n_tc_ncpg:>10} {n_tc_cpg:>9}"
        )


def print_section_1d(all_results):
    logger.info("\n" + "=" * 90)
    logger.info("SECTION 1d — Expression Quartile OR@p90: binary head (cancers w/ RSEM)")
    logger.info("=" * 90)
    header = f"{'Cancer':>9} | {'Q1_low':>8} {'Q2':>8} {'Q3':>8} {'Q4_high':>9} | {'n_mapped':>10}"
    logger.info(header)
    logger.info("-" * len(header))
    for c in CANCERS_WITH_EXPR:
        r = all_results[c]["section_1d"]
        if r.get("status") == "skipped":
            logger.info(f"{c:>9} | SKIPPED ({r.get('reason')})")
            continue
        nn = r.get("nn_binary", {})
        def g(q):
            return nn.get(q, {}).get("p90", {}).get("OR", float("nan"))
        logger.info(
            f"{c:>9} | {g('Q1_low'):>8.3f} {g('Q2'):>8.3f} {g('Q3'):>8.3f} {g('Q4_high'):>9.3f} | {r.get('n_mut_with_expr',0):>10}"
        )


def to_jsonable(o):
    if isinstance(o, dict):
        return {k: to_jsonable(v) for k, v in o.items()}
    if isinstance(o, list):
        return [to_jsonable(x) for x in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    return o


def main():
    t0 = time.time()
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Model: {NEURAL_MODEL_PATH}")
    logger.info(f"Cancers: {CANCERS}")

    logger.info("\nLoading phase3 neural model...")
    model = Phase3Model()
    state = torch.load(NEURAL_MODEL_PATH, weights_only=True, map_location="cpu")
    model.load_state_dict(state)
    model.to(DEVICE).eval()

    all_results = {}
    for c in CANCERS:
        logger.info(f"\n>>> {c.upper()}")
        try:
            d = load_and_score_cancer(c, model)
        except Exception as e:
            logger.error(f"  {c}: FAILED — {e}")
            continue

        res_c = {}
        res_c["section_1a_1g"] = compute_section_1a_1g(d, c)
        res_c["section_1b_1c"] = compute_section_1b_1c(d, c)
        res_c["section_1d"] = compute_section_1d(d, c)
        res_c["meta"] = {
            "n": d["n"],
            "n_mut": int(d["mut_mask"].sum()),
            "n_ctrl": int(d["ctrl_mask"].sum()),
            "tc_frac": float(d["tc"].mean()),
            "cpg_frac": float(d["cpg"].mean()),
        }
        all_results[c] = res_c

        # free memory before next cancer
        del d
        gc.collect()

    # Print summary tables
    print_section_1a_1g(all_results)
    print_section_1b_1c(all_results)
    print_section_1d(all_results)

    # Save JSON
    out_json = OUTPUT_DIR / "section1_nn_reproduction.json"
    with open(out_json, "w") as f:
        json.dump(to_jsonable(all_results), f, indent=2)
    logger.info(f"\nSaved: {out_json}")

    # Save flat CSVs for easy inspection
    rows = []
    for c, r in all_results.items():
        s = r["section_1a_1g"]
        for head in ["xgb", "nn_binary"] + [f"nn_{e}" for e in ENZYMES]:
            e = s.get(head, {})
            if "p90" not in e:
                continue
            rows.append({
                "cancer": c, "head": head,
                "n_mut": e.get("n_mut"), "n_ctrl": e.get("n_ctrl"),
                "mean_mut": e.get("mean_mut"), "mean_ctrl": e.get("mean_ctrl"),
                "mw_p": e.get("mw_p"),
                "OR_p50": e["p50"]["OR"], "OR_p75": e["p75"]["OR"],
                "OR_p90": e["p90"]["OR"], "OR_p95": e["p95"]["OR"],
                "p_p90": e["p90"]["p"], "p_p95": e["p95"]["p"],
            })
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "section_1a_1g_summary.csv", index=False)

    # Stratified CSV
    strat_rows = []
    for c, r in all_results.items():
        s = r["section_1b_1c"]
        for head in ["xgb", "nn_binary"] + [f"nn_{e}" for e in ENZYMES]:
            h = s.get(head, {})
            for stratum in ["TC", "nonTC", "TC_nonCpG", "TC_CpG", "nonTC_nonCpG", "nonTC_CpG"]:
                sr = h.get(stratum, {})
                if "p90" not in sr:
                    continue
                strat_rows.append({
                    "cancer": c, "head": head, "stratum": stratum,
                    "n_mut": sr.get("n_mut"), "n_ctrl": sr.get("n_ctrl"),
                    "OR_p50": sr.get("p50", {}).get("OR"),
                    "OR_p75": sr.get("p75", {}).get("OR"),
                    "OR_p90": sr["p90"]["OR"], "OR_p95": sr.get("p95", {}).get("OR"),
                    "p_p90": sr["p90"]["p"],
                })
    pd.DataFrame(strat_rows).to_csv(OUTPUT_DIR / "section_1b_1c_stratified.csv", index=False)

    # Expression quartile CSV
    exp_rows = []
    for c in CANCERS_WITH_EXPR:
        r = all_results.get(c, {}).get("section_1d", {})
        if r.get("status") == "skipped":
            continue
        for head in ["xgb", "nn_binary"] + [f"nn_{e}" for e in ENZYMES]:
            h = r.get(head, {})
            for q in ["Q1_low", "Q2", "Q3", "Q4_high"]:
                qr = h.get(q, {})
                if "p90" not in qr:
                    continue
                exp_rows.append({
                    "cancer": c, "head": head, "quartile": q,
                    "n_mut": qr.get("n_mut"),
                    "OR_p90": qr["p90"]["OR"], "OR_p95": qr.get("p95", {}).get("OR"),
                    "p_p90": qr["p90"]["p"],
                })
    pd.DataFrame(exp_rows).to_csv(OUTPUT_DIR / "section_1d_expression.csv", index=False)

    logger.info(f"\nTotal: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
