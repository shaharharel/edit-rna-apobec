#!/usr/bin/env python3
"""Generate COMPARISON_PHASE1_VS_1_5.md after Phase 1.5 finishes.

Reads:
  experiments/multi_enzyme/outputs/pcawg_tcw_panel/analysis_A_pcawg_wgs/enrichment_primary.json
  experiments/multi_enzyme/outputs/pcawg_tcw_panel/analysis_A_pcawg_wgs/enrichment_primary_phase1_5.json
  experiments/multi_enzyme/outputs/pcawg_tcw_panel/analysis_B_coding_panel/enrichment_primary.json
  experiments/multi_enzyme/outputs/pcawg_tcw_panel/analysis_B_coding_panel/enrichment_primary_phase1_5.json
  experiments/multi_enzyme/outputs/pcawg_tcw_panel/analysis_C_v2_section1c_repro/or_table.json

Writes:
  experiments/multi_enzyme/outputs/pcawg_tcw_panel/COMPARISON_PHASE1_VS_1_5.md
  experiments/multi_enzyme/outputs/pcawg_tcw_panel/PHASE_1_5_DONE.flag
"""
from __future__ import annotations
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PANEL_DIR = PROJECT_ROOT / "experiments/multi_enzyme/outputs/pcawg_tcw_panel"


def load_json(p: Path):
    if not p.exists():
        return None
    try:
        return json.load(open(p))
    except Exception:
        return None


def _val(crit: dict, key_substr: str):
    if not crit:
        return None
    for k, v in crit.items():
        if key_substr in k.lower():
            return v
    return None


def main():
    A1 = load_json(PANEL_DIR / "analysis_A_pcawg_wgs" / "enrichment_primary.json")
    A15 = load_json(PANEL_DIR / "analysis_A_pcawg_wgs" / "enrichment_primary_phase1_5.json")
    B1 = load_json(PANEL_DIR / "analysis_B_coding_panel" / "enrichment_primary.json")
    B15 = load_json(PANEL_DIR / "analysis_B_coding_panel" / "enrichment_primary_phase1_5.json")
    C = load_json(PANEL_DIR / "analysis_C_v2_section1c_repro" / "or_table.json")

    def row(name: str, agg: str, win: str, A, B):
        a_a = _val(A.get("pass_criteria") if A else None, "primary")
        a_b = _val(A.get("pass_criteria") if A else None, "signif")
        b_a = _val(B.get("pass_criteria") if B else None, "primary")
        b_b = _val(B.get("pass_criteria") if B else None, "signif")
        n_A = len(A.get("per_cancer", {})) if A else 0
        n_B = len(B.get("per_cancer", {})) if B else 0
        a_a_v = a_a["val"] if a_a else float("nan")
        b_a_v = b_a["val"] if b_a else float("nan")
        a_b_v = a_b["val"] if a_b else 0
        b_b_v = b_b["val"] if b_b else 0
        a_pass = (A.get("pass_criteria", {}).get("PASS") if A else False)
        b_pass = (B.get("pass_criteria", {}).get("PASS") if B else False)
        return (f"| {name} | {agg} | {win} | {a_a_v:.3f} | {b_a_v:.3f} | "
                f"{a_b_v}/{n_A} | {b_b_v}/{n_B} | "
                f"{'Y' if a_pass else 'N'} | {'Y' if b_pass else 'N'} |\n")

    lines = []
    lines.append("# Comparison Phase 1 vs Phase 1.5 — pcawg_tcw_panel\n\n")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M')}\n\n")
    lines.append("## Phase 1 vs Phase 1.5 primary endpoints\n\n")
    lines.append("| Phase | Aggregator | Window | Mean recall ratio (A) | Mean recall ratio (B) | "
                 "BH-pass cancers (A, q<0.025) | BH-pass cancers (B, q<0.025) | A PASS | B PASS |\n")
    lines.append("|-------|------------|--------|----------------------:|----------------------:|"
                 "----------------------------:|----------------------------:|:------:|:------:|\n")
    if A1 or B1:
        lines.append(row("Phase 1", "mean", "1000 bp", A1, B1))
    else:
        lines.append("| Phase 1 | mean | 1000 bp | (missing) | (missing) | — | — | — | — |\n")
    if A15 or B15:
        lines.append(row("Phase 1.5", "max", "250 bp", A15, B15))
    else:
        lines.append("| Phase 1.5 | max | 250 bp | (missing) | (missing) | — | — | — | — |\n")

    # Per-cancer deltas (Phase 1.5 minus Phase 1) for B (since A often only has 4 cancers)
    if B1 and B15:
        lines.append("\n## Per-cancer mut_ratio: Phase 1 vs Phase 1.5 (Analysis B)\n\n")
        lines.append("| cancer | n_mut (P1) | mut_ratio (P1, mean+1kb) | mut_ratio (P1.5, max+250bp) | Δ |\n")
        lines.append("|--------|----------:|--------------------------:|----------------------------:|---:|\n")
        b1_pc = B1.get("per_cancer", {})
        b15_pc = B15.get("per_cancer", {})
        for cancer in sorted(set(b1_pc) | set(b15_pc)):
            r1 = b1_pc.get(cancer, {}).get("primary", {})
            r15 = b15_pc.get(cancer, {}).get("primary", {})
            v1 = r1.get("ratio", float("nan"))
            v15 = r15.get("ratio", float("nan"))
            delta = v15 - v1 if (v1 == v1 and v15 == v15) else float("nan")
            n = r1.get("total_mut", r15.get("total_mut", 0))
            lines.append(f"| {cancer} | {n} | {v1:.3f} | {v15:.3f} | {delta:+.3f} |\n")

    if A1 and A15:
        lines.append("\n## Per-cancer mut_ratio: Phase 1 vs Phase 1.5 (Analysis A)\n\n")
        lines.append("| cancer | n_mut (P1) | mut_ratio (P1, mean+1kb) | mut_ratio (P1.5, max+250bp) | Δ |\n")
        lines.append("|--------|----------:|--------------------------:|----------------------------:|---:|\n")
        a1_pc = A1.get("per_cancer", {})
        a15_pc = A15.get("per_cancer", {})
        for cancer in sorted(set(a1_pc) | set(a15_pc)):
            r1 = a1_pc.get(cancer, {}).get("primary", {})
            r15 = a15_pc.get(cancer, {}).get("primary", {})
            v1 = r1.get("ratio", float("nan"))
            v15 = r15.get("ratio", float("nan"))
            delta = v15 - v1 if (v1 == v1 and v15 == v15) else float("nan")
            n = r1.get("total_mut", r15.get("total_mut", 0))
            lines.append(f"| {cancer} | {n} | {v1:.3f} | {v15:.3f} | {delta:+.3f} |\n")

    # Analysis C v2 §1c reproduction
    lines.append("\n## Analysis C — v2 §1c apples-to-apples reproduction\n\n")
    if C is None:
        lines.append("(Analysis C results missing.)\n\n")
    else:
        lines.append("Per-position TC+nonCpG OR@p90 with MFE-only model, vs v2 §1c reported.\n\n")
        lines.append("| Cancer | v2 §1c OR | Today's OR | Δ | within ±0.05 | n_mut | n_tc_noncpg | p |\n")
        lines.append("|--------|----------:|-----------:|---:|:------------:|------:|------------:|---|\n")
        n_with_v2 = 0; n_within = 0
        for r in C.get("summary", []):
            v2 = r.get("v2_OR_p90")
            today = r.get("today_OR_p90")
            delta = r.get("delta")
            within = r.get("within_+-0.05")
            v2_str = f"{v2:.2f}" if v2 is not None else "n/a"
            today_str = f"{today:.3f}" if today == today else "nan"
            delta_str = f"{delta:+.3f}" if delta is not None else "n/a"
            within_str = "Y" if within else ("N" if v2 is not None else "—")
            if v2 is not None:
                n_with_v2 += 1
                if within: n_within += 1
            p = r.get("p_value")
            p_str = f"{p:.2e}" if p is not None and p == p else "nan"
            lines.append(f"| {r['cancer']} | {v2_str} | {today_str} | {delta_str} | {within_str} | "
                         f"{r['n_mut']} | {r['n_tc_noncpg']} | {p_str} |\n")
        lines.append(f"\n**Reproduction**: {n_within}/{n_with_v2} cancers within ±0.05 of v2 §1c.\n\n")

    # Interpretation
    lines.append("\n## Interpretation guide (per supervisor)\n\n")
    lines.append("- **C reproduces v2 (within ±0.05) AND Phase 1.5 primary passes**: signal recovered "
                 "at finer resolution; model is real.\n")
    lines.append("- **C reproduces v2 BUT Phase 1.5 primary fails**: signal exists at position level "
                 "but doesn't survive a strict windowed comparison. Reframe paper accordingly.\n")
    lines.append("- **C does NOT reproduce v2**: today's pipeline has an inconsistency (model layout, "
                 "feature alignment, control set). Investigate before further conclusions.\n")
    lines.append("- **Both Phase 1 and Phase 1.5 fail AND C reproduces**: model is real at marginal "
                 "effect size but not panel-grade — honest negative for cfDNA-panel claim.\n\n")

    out = PANEL_DIR / "COMPARISON_PHASE1_VS_1_5.md"
    with open(out, "w") as f:
        f.writelines(lines)
    print(f"Wrote {out}")

    # Flag if both phase 1.5 + C present
    if (A15 or B15) and C:
        flag = PANEL_DIR / "PHASE_1_5_DONE.flag"
        flag.write_text(f"Phase 1.5 complete at {time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
        print(f"Wrote {flag}")
    return 0


if __name__ == "__main__":
    main()
