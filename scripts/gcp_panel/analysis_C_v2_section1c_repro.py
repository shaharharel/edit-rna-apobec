#!/usr/bin/env python3
"""Analysis C — v2 §1c apples-to-apples reproduction with MFE-only model.

Reuses the v2 §1c cached TCGA assets (RAW_SCORES with mutation/control labels,
hand40 features, RNA-FM embeddings) and rescores per-position with the new
phase3_mfe_only model (struct_delta slots zeroed). Computes the per-position
TC+nonCpG OR@p90 in the same way as v2 §1c and compares to the v2 reported
numbers (BLCA 1.33, BRCA 1.17, CESC 1.30, LUSC 1.21).

This is a TRUST-ANCHOR test: if today's MFE-only pipeline reproduces v2 §1c
within ±0.05, then the per-position model signal is intact and the Phase 1
1 kb mean-aggregation is the issue. If it does NOT reproduce, today's pipeline
has a deeper inconsistency to investigate.

Inputs:
  - experiments/multi_enzyme/outputs/tcga_gnomad/raw_scores/{cancer}_scores.csv
    (mutation/control labels, deterministic seed=42 from v2)
  - data/processed/multi_enzyme/tcga_hand_features/{cancer}_hand40.npy
  - data/processed/multi_enzyme/embeddings/rnafm_tcga_{cancer}.pt
    (pooled_orig + pooled_edited, 640-d each)
  - experiments/multi_enzyme/outputs/phase3_mfe_only/phase3_mfe_only.pt

Output:
  experiments/multi_enzyme/outputs/pcawg_tcw_panel/analysis_C_v2_section1c_repro/
    or_table.json
    REPORT.md
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats

sys.stdout.reconfigure(line_buffering=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
EMB_DIR = DATA_DIR / "processed" / "multi_enzyme" / "embeddings"
HAND_DIR = DATA_DIR / "processed" / "multi_enzyme" / "tcga_hand_features"
RAW_SCORES_DIR = PROJECT_ROOT / "experiments" / "multi_enzyme" / "outputs" / "tcga_gnomad" / "raw_scores"
DEFAULT_MFE_MODEL = PROJECT_ROOT / "experiments" / "multi_enzyme" / "outputs" / "phase3_mfe_only" / "phase3_mfe_only.pt"

# Same canonical phase3 architecture as the rest of the pipeline
ENZYMES = ["A3A", "A3B", "A3G", "A3A_A3G", "Neither"]
D_INPUT = 1320
D_SHARED = 128
N_ENZYMES_CLS = 6
STRUCT_DELTA_START_ABS = 1304
STRUCT_DELTA_END_ABS = 1311

# v2 §1c reported per-position OR@p90 for binary head, TC+non-CpG (oriented hand-feature
# definition: hand[:, 0] == 1 for TC, hand[:, 5] == 0 for non-CpG)
V2_SECTION_1C_OR_P90 = {
    "blca": 1.33,
    "brca": 1.17,
    "cesc": 1.30,
    "lusc": 1.21,
    # Other cancers were reported in v2 §1c but with smaller signal
    "hnsc": None,
    "skcm": None,
    "esca": None,
    "stad": None,
    "lihc": None,
    "coadread": None,
}
# Tolerance per supervisor: reproduction must land within +/- 0.05 of v2 numbers
REPRO_TOLERANCE = 0.05

CANCERS = ["blca", "brca", "cesc", "lusc", "hnsc", "skcm", "esca", "stad", "lihc", "coadread"]


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
        binary = self.binary_head(shared).squeeze(-1)
        per_enzyme = [self.enzyme_adapters[enz](shared).squeeze(-1) for enz in ENZYMES]
        return binary, per_enzyme, None, shared


def score_all_heads(model, X: np.ndarray, device, batch_size: int = 1024) -> dict:
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


def compute_or_at_p(scores: np.ndarray, mut_mask: np.ndarray, ctrl_mask: np.ndarray,
                    pct: int, pool_mask: np.ndarray | None = None) -> dict:
    """OR @ percentile threshold; threshold computed on the pooled subset (mut+ctrl in stratum)."""
    if pool_mask is None:
        pool_mask = mut_mask | ctrl_mask
    pooled = scores[pool_mask]
    if len(pooled) < 20:
        return {"OR": float("nan"), "p": 1.0, "threshold": float("nan"),
                "n_mut": int(mut_mask.sum()), "n_ctrl": int(ctrl_mask.sum())}
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
            "mut_above": ma, "mut_below": mb, "ctrl_above": ca, "ctrl_below": cb,
            "n_mut": int(mut_mask.sum()), "n_ctrl": int(ctrl_mask.sum())}


def stratified_or(scores, mut_mask, ctrl_mask, strata_mask, pct=90):
    """OR @ percentile within a stratum; threshold pooled within the stratum (matches v2 §1b/1c)."""
    mut_s_mask = mut_mask & strata_mask
    ctrl_s_mask = ctrl_mask & strata_mask
    return compute_or_at_p(scores, mut_s_mask, ctrl_s_mask, pct, pool_mask=strata_mask)


def load_and_score_cancer(cancer: str, model: Phase3Model, device, batch_size: int) -> dict | None:
    """Load v2-cached assets, build the 1320-d input with struct_delta zeroed, run all 6 heads."""
    raw_path = RAW_SCORES_DIR / f"{cancer}_scores.csv"
    hand_path = HAND_DIR / f"{cancer}_hand40.npy"
    emb_path = EMB_DIR / f"rnafm_tcga_{cancer}.pt"
    for p in (raw_path, hand_path, emb_path):
        if not p.exists():
            logger.warning("Missing %s — skipping %s", p, cancer)
            return None

    raw = pd.read_csv(raw_path)
    hand = np.load(hand_path)
    emb = torch.load(emb_path, weights_only=False, map_location="cpu")

    n = len(raw)
    if hand.shape[0] != n or emb["pooled_orig"].shape[0] != n:
        logger.warning("%s: shape mismatch (raw=%d, hand=%d, emb=%d) — skip", cancer,
                       n, hand.shape[0], emb["pooled_orig"].shape[0])
        return None

    types = raw["type"].values
    mut_mask = types == "mutation"
    ctrl_mask = types == "control"

    # Oriented hand-feature TC and CpG masks (same as v2 §1c)
    tc_mask = hand[:, 0] == 1.0      # 5'-UC: oriented TC
    cpg_mask = hand[:, 5] == 1.0     # 3'-CG: oriented CpG

    pooled_orig = emb["pooled_orig"].numpy().astype(np.float32)
    pooled_edited = emb["pooled_edited"].numpy().astype(np.float32)
    edit_delta = pooled_edited - pooled_orig
    del emb

    # Build 1320-d: [orig 640 | delta 640 | hand 40]
    X = np.concatenate([pooled_orig, edit_delta, hand], axis=1).astype(np.float32)
    del pooled_orig, pooled_edited, edit_delta
    gc.collect()

    # MFE-only regime: zero out struct_delta slots [24:31] (within hand40, absolute slots 1304:1311)
    struct_delta_max_before = float(np.abs(X[:, STRUCT_DELTA_START_ABS:STRUCT_DELTA_END_ABS]).max())
    X[:, STRUCT_DELTA_START_ABS:STRUCT_DELTA_END_ABS] = 0.0
    logger.info("  %s: n=%d mut=%d ctrl=%d  TC%%=%.3f  CpG%%=%.3f  struct_delta zeroed (was max=%.3f)",
                cancer, n, mut_mask.sum(), ctrl_mask.sum(), tc_mask.mean(), cpg_mask.mean(),
                struct_delta_max_before)

    nn_scores = score_all_heads(model, X, device, batch_size)
    del X
    gc.collect()
    return {
        "n": n,
        "mut_mask": mut_mask,
        "ctrl_mask": ctrl_mask,
        "tc": tc_mask,
        "cpg": cpg_mask,
        "nn": nn_scores,
    }


def _sha256_file(path: Path) -> str:
    if not Path(path).exists():
        return f"<missing:{path}>"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT),
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else f"<git-error:{out.stderr.strip()[:100]}>"
    except Exception as ex:
        return f"<git-exception:{ex}>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=DEFAULT_MFE_MODEL)
    ap.add_argument("--out-dir", type=Path,
                    default=PROJECT_ROOT / "experiments/multi_enzyme/outputs/pcawg_tcw_panel/analysis_C_v2_section1c_repro")
    ap.add_argument("--batch-size", type=int, default=1024)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("mps") if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
              else torch.device("cpu"))
    logger.info("Device: %s | Model: %s", device, args.model)

    if not args.model.exists():
        logger.error("Model not found: %s", args.model)
        return 1

    model = Phase3Model()
    model.load_state_dict(torch.load(args.model, weights_only=False, map_location="cpu"))
    model.to(device).eval()
    logger.info("Loaded phase3_mfe_only model. SHA256 will be recorded in provenance.")

    # Score all 10 cancers
    results = {}
    for cancer in CANCERS:
        t0 = time.time()
        d = load_and_score_cancer(cancer, model, device, args.batch_size)
        if d is None:
            results[cancer] = {"status": "missing_assets"}
            continue
        binary = d["nn"]["binary"]

        # Section 1c primary: TC+nonCpG, OR@p90 for binary head
        tc_noncpg = d["tc"] & (~d["cpg"])
        per_p = {}
        for pct in (50, 75, 90, 95):
            per_p[f"p{pct}"] = stratified_or(binary, d["mut_mask"], d["ctrl_mask"], tc_noncpg, pct=pct)

        # Same TC strata as v2 §1c (full 1c table) for completeness
        strata = {
            "TC": d["tc"],
            "nonTC": ~d["tc"],
            "TC_nonCpG": d["tc"] & (~d["cpg"]),
            "TC_CpG": d["tc"] & d["cpg"],
            "nonTC_nonCpG": (~d["tc"]) & (~d["cpg"]),
            "nonTC_CpG": (~d["tc"]) & d["cpg"],
        }
        head_results = {}
        for head_name, scores in [("binary", binary)] + [(enz, d["nn"][enz]) for enz in ENZYMES]:
            stratum_or = {}
            for s_name, s_mask in strata.items():
                stratum_or[s_name] = stratified_or(scores, d["mut_mask"], d["ctrl_mask"], s_mask, pct=90)
            head_results[head_name] = stratum_or

        v2_or = V2_SECTION_1C_OR_P90.get(cancer)
        today_or = per_p["p90"]["OR"]
        delta = (today_or - v2_or) if (v2_or is not None and not np.isnan(today_or)) else None
        within_tol = (delta is not None and abs(delta) <= REPRO_TOLERANCE)
        results[cancer] = {
            "status": "ok",
            "n_total": int(d["n"]),
            "n_mut": int(d["mut_mask"].sum()),
            "n_ctrl": int(d["ctrl_mask"].sum()),
            "n_tc_noncpg": int(tc_noncpg.sum()),
            "tc_noncpg_per_p": per_p,
            "v2_section_1c_or_p90": v2_or,
            "today_or_p90_tc_noncpg": today_or,
            "delta_vs_v2": delta,
            "within_tolerance_+-0.05": within_tol,
            "all_strata_or_p90_per_head": head_results,
        }
        logger.info("  %-9s  v2_OR=%s  today_OR=%.3f  delta=%s  n_mut=%d  n_ctrl=%d  n_tc_noncpg=%d  (%.1fs)",
                    cancer, str(v2_or) if v2_or else "n/a", today_or,
                    f"{delta:+.3f}" if delta is not None else "n/a",
                    d["mut_mask"].sum(), d["ctrl_mask"].sum(), tc_noncpg.sum(),
                    time.time() - t0)

    # Provenance
    provenance = {
        "git_commit": _git_head_commit(),
        "phase3_mfe_only_sha256": _sha256_file(args.model),
        "raw_scores_dir": str(RAW_SCORES_DIR),
        "hand_dir": str(HAND_DIR),
        "emb_dir": str(EMB_DIR),
        "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "device": str(device),
        "tolerance_+-": REPRO_TOLERANCE,
    }

    # Build summary table
    summary = []
    for cancer in CANCERS:
        r = results[cancer]
        if r.get("status") != "ok":
            continue
        v2 = r.get("v2_section_1c_or_p90")
        today = r.get("today_or_p90_tc_noncpg")
        delta = r.get("delta_vs_v2")
        within = r.get("within_tolerance_+-0.05")
        summary.append({
            "cancer": cancer.upper(),
            "n_mut": r.get("n_mut"),
            "n_ctrl": r.get("n_ctrl"),
            "n_tc_noncpg": r.get("n_tc_noncpg"),
            "v2_OR_p90": v2,
            "today_OR_p90": today,
            "delta": delta,
            "within_+-0.05": within,
            "p_value": r["tc_noncpg_per_p"]["p90"].get("p"),
        })

    out_json = {"per_cancer": results, "summary": summary, "provenance": provenance}
    with open(args.out_dir / "or_table.json", "w") as f:
        json.dump(out_json, f, indent=2, default=str)
    logger.info("Wrote %s/or_table.json", args.out_dir)

    # Build REPORT.md
    lines = []
    lines.append("# Analysis C — v2 §1c apples-to-apples reproduction\n\n")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M')}\n\n")
    lines.append("Trust-anchor test: scores TCGA mutation/control pairs (deterministic seed=42 from "
                 "v2 §1c) with phase3_mfe_only.pt (struct_delta zeroed) and computes per-position "
                 "TC+nonCpG OR@p90, comparing to v2 §1c reported numbers.\n\n")
    lines.append("## v2 §1c reported numbers vs today's MFE-only pipeline\n\n")
    lines.append("| Cancer | v2 §1c OR | Today's OR | Δ | within ±0.05 | n_mut | n_ctrl | n_tc_noncpg | p |\n")
    lines.append("|--------|----------:|-----------:|---:|:------------:|------:|-------:|------------:|---|\n")
    n_within = 0
    n_with_v2 = 0
    for r in summary:
        v2 = r["v2_OR_p90"]
        today = r["today_OR_p90"]
        delta = r["delta"]
        within = r["within_+-0.05"]
        v2_str = f"{v2:.2f}" if v2 is not None else "n/a"
        delta_str = f"{delta:+.3f}" if delta is not None else "n/a"
        within_str = "Y" if within else "N" if v2 is not None else "—"
        if v2 is not None:
            n_with_v2 += 1
            if within:
                n_within += 1
        try:
            today_str = f"{today:.3f}" if not np.isnan(today) else "nan"
        except Exception:
            today_str = "nan"
        p_str = f"{r['p_value']:.2e}" if r['p_value'] is not None and r['p_value'] == r['p_value'] else "nan"
        lines.append(f"| {r['cancer']} | {v2_str} | {today_str} | {delta_str} | {within_str} | "
                     f"{r['n_mut']} | {r['n_ctrl']} | {r['n_tc_noncpg']} | {p_str} |\n")
    lines.append(f"\n**Reproduction summary**: {n_within}/{n_with_v2} cancers within ±0.05 of v2 §1c.\n\n")
    if n_with_v2 > 0:
        if n_within == n_with_v2:
            lines.append("### CONCLUSION: today's MFE-only pipeline reproduces v2 §1c per-position OR@p90.\n")
            lines.append("The model + RNA-FM + hand-feature pipeline produces the same per-position "
                         "signal as v2. Therefore, Phase 1's failure to enrich at the 1 kb mean-window "
                         "level reflects window-aggregation dilution of a real but peaky per-position "
                         "signal — not a methodology bug or model-quality issue.\n\n")
        elif n_within == 0:
            lines.append("### CONCLUSION: today's MFE-only pipeline does NOT reproduce v2 §1c.\n")
            lines.append("This indicates a deeper inconsistency between the new MFE-only model + "
                         "pipeline vs. v2's setup. Investigate model architecture, feature layout, "
                         "control set construction before drawing conclusions from Phase 1.\n\n")
        else:
            lines.append(f"### CONCLUSION: partial reproduction ({n_within}/{n_with_v2} cancers within ±0.05).\n")
            lines.append("Mixed result. Review per-cancer deltas; cancers with large |Δ| may indicate "
                         "specific data drift (e.g. v2 §1c used canonical phase3 with struct_delta).\n\n")

    lines.append("## Provenance\n\n")
    for k, v in provenance.items():
        lines.append(f"- **{k}**: `{v}`\n")
    lines.append("\n## Method\n\n")
    lines.append("1. Loaded each cancer's cached assets from v2 §1c (raw_scores, hand40, RNA-FM).\n")
    lines.append("2. Built 1320-d input: orig (640) + delta (640) + hand40 (40).\n")
    lines.append("3. Zeroed struct_delta slots [1304:1311] to match MFE-only training regime.\n")
    lines.append("4. Scored all 6 heads with phase3_mfe_only.pt.\n")
    lines.append("5. Computed TC+nonCpG OR@p90 = Fisher OR(top 10% scores, mut vs ctrl) within "
                 "TC+nonCpG stratum (oriented hand-feature TC/CpG: hand[:, 0]==1 / hand[:, 5]==1).\n")
    lines.append("6. Threshold pooled within stratum (same as v2 §1c).\n")
    lines.append("7. Compared per-cancer OR to v2 §1c reported numbers; within ±0.05 = reproduces.\n\n")
    lines.append("## Per-cancer per-stratum (binary head, OR@p90)\n\n")
    lines.append("| Cancer | TC | nonTC | TC_nonCpG | TC_CpG | nonTC_nonCpG | nonTC_CpG |\n")
    lines.append("|--------|----:|-----:|----------:|-------:|-------------:|----------:|\n")
    for cancer in CANCERS:
        r = results.get(cancer)
        if not r or r.get("status") != "ok":
            continue
        sb = r["all_strata_or_p90_per_head"]["binary"]
        def _v(k):
            v = sb.get(k, {}).get("OR", float("nan"))
            return f"{v:.2f}" if v == v else "nan"
        lines.append(f"| {cancer.upper()} | {_v('TC')} | {_v('nonTC')} | {_v('TC_nonCpG')} | "
                     f"{_v('TC_CpG')} | {_v('nonTC_nonCpG')} | {_v('nonTC_CpG')} |\n")

    with open(args.out_dir / "REPORT.md", "w") as f:
        f.writelines(lines)
    logger.info("Wrote %s/REPORT.md", args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
