#!/usr/bin/env python3
"""Generate COMPARISON_PHASE1_VS_1_5_VS_1_6.md + FINAL_REPORT_PHASE1_6.md and
PHASE_1_6_DONE.flag.

Reads:
  experiments/multi_enzyme/outputs/pcawg_tcw_panel/analysis_{A,B}/enrichment_primary{,_phase1_5}.json
  ... /enrichment_primary_phase1_6_{A,B}_w{100,250,500,1000,2000}.json
  ... /enrichment_primary_phase1_6_definitive.json
"""
from __future__ import annotations
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PANEL_DIR = PROJECT_ROOT / "experiments/multi_enzyme/outputs/pcawg_tcw_panel"
WINDOWS = [100, 250, 500, 1000, 2000]


def load(p: Path):
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


def boot(crit: dict, key_substr: str):
    v = _val(crit, key_substr)
    if v is None or "bootstrap" not in v:
        return None
    return v["bootstrap"]


def fmt_ratio(j):
    if not j:
        return "—", "—", "—"
    pa = _val(j.get("pass_criteria"), "primary")
    pb = _val(j.get("pass_criteria"), "signif")
    n_pc = len(j.get("per_cancer", {}))
    pa_v = pa["val"] if pa else float("nan")
    pb_v = pb["val"] if pb else 0
    return f"{pa_v:.3f}", f"{pb_v}/{n_pc}", "Y" if j.get("pass_criteria", {}).get("PASS") else "N"


def main():
    A1 = load(PANEL_DIR / "analysis_A_pcawg_wgs" / "enrichment_primary.json")
    A15 = load(PANEL_DIR / "analysis_A_pcawg_wgs" / "enrichment_primary_phase1_5.json")
    B1 = load(PANEL_DIR / "analysis_B_coding_panel" / "enrichment_primary.json")
    B15 = load(PANEL_DIR / "analysis_B_coding_panel" / "enrichment_primary_phase1_5.json")
    AdefSweep = {w: load(PANEL_DIR / "analysis_A_pcawg_wgs" /
                         f"enrichment_primary_phase1_6_A_w{w}.json") for w in WINDOWS}
    BdefSweep = {w: load(PANEL_DIR / "analysis_B_coding_panel" /
                         f"enrichment_primary_phase1_6_B_w{w}.json") for w in WINDOWS}
    A_def = load(PANEL_DIR / "analysis_A_pcawg_wgs" / "enrichment_primary_phase1_6_definitive.json")
    B_def = load(PANEL_DIR / "analysis_B_coding_panel" / "enrichment_primary_phase1_6_definitive.json")

    lines = []
    lines.append("# Phase 1 vs Phase 1.5 vs Phase 1.6 - pcawg_tcw_panel\n\n")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M')}\n\n")
    lines.append("## Three-way primary endpoint table\n\n")
    lines.append("| Phase | Window | Aggregator | Perms | Mean ratio (A) | BH-sig (A) | A PASS | "
                 "Mean ratio (B) | BH-sig (B) | B PASS |\n")
    lines.append("|-------|--------|------------|------:|---------------:|-----------:|:------:|"
                 "---------------:|-----------:|:------:|\n")

    def emit_row(name, win, agg, perms, jA, jB):
        rA, sA, pA = fmt_ratio(jA)
        rB, sB, pB = fmt_ratio(jB)
        lines.append(f"| {name} | {win} | {agg} | {perms} | {rA} | {sA} | {pA} | {rB} | {sB} | {pB} |\n")

    emit_row("Phase 1", "1000 bp", "mean", "10K", A1, B1)
    emit_row("Phase 1.5", "250 bp", "max", "10K", A15, B15)
    for w in WINDOWS:
        emit_row(f"Phase 1.6 sweep w={w}", f"{w} bp", "max", "10K",
                 AdefSweep.get(w), BdefSweep.get(w))
    emit_row("**Phase 1.6 definitive**", "**250 bp**", "**max**", "**100K**", A_def, B_def)

    lines.append("\n## Phase 1.6 definitive: per-criterion breakdown with bootstrap CIs\n\n")
    lines.append("| Analysis | Criterion | Threshold | Observed | Pass | Bootstrap CI95 | "
                 "Bootstrap p (H0: <= thresh) |\n")
    lines.append("|----------|-----------|----------:|---------:|:----:|----------------|"
                 "----------------------------:|\n")
    for name, j in [("Analysis A", A_def), ("Analysis B", B_def)]:
        if not j:
            lines.append(f"| {name} | (data missing) | | | | | |\n")
            continue
        pc = j.get("pass_criteria", {})
        for key, label, thresh_label in [
            ("primary", "(a) mean ratio", "1.5"),
            ("driver", "(c) driver-ablated", "1.3"),
            ("masked", "(d) mask-survived", "1.3"),
        ]:
            entry = _val(pc, key)
            b = entry.get("bootstrap") if entry else None
            if not b:
                lines.append(f"| {name} | {label} | {thresh_label} | — | — | — | — |\n")
                continue
            obs = b.get("mean_observed", float("nan"))
            ci_lo = b.get("ci95_low", float("nan"))
            ci_hi = b.get("ci95_high", float("nan"))
            p = b.get("p_boot_le_thresh", float("nan"))
            passed = entry.get("pass", False)
            lines.append(f"| {name} | {label} | {thresh_label} | {obs:.3f} | "
                         f"{'Y' if passed else 'N'} | [{ci_lo:.3f}, {ci_hi:.3f}] | {p:.3e} |\n")
        # b
        entry_b = _val(pc, "signif")
        if entry_b:
            n = entry_b.get("val", 0)
            lines.append(f"| {name} | (b) BH-sig q<0.025 | {entry_b.get('thresh', 6)}/10 | "
                         f"{n}/10 | {'Y' if entry_b.get('pass') else 'N'} | (deterministic) | (deterministic) |\n")
        # joint
        je = pc.get("joint_exceedance_n_above_1.0")
        if je:
            lines.append(f"| {name} | joint exceedance >1.0 | informational | "
                         f"{je.get('n_above_1.0', 0)}/{je.get('n_cancers', 0)} | — | — | "
                         f"{je.get('binomial_p_one_sided', float('nan')):.3e} |\n")

    # Per-cancer table for definitive
    if B_def:
        lines.append("\n## Phase 1.6 definitive: per-cancer Analysis B (250 bp + max + 100K perms)\n\n")
        lines.append("| cancer | total_mut | recall_model | recall_cpg | mut_ratio | p_perm | q_bh | reject |\n")
        lines.append("|--------|----------:|-------------:|-----------:|----------:|-------:|------:|:------:|\n")
        for cancer, det in B_def.get("per_cancer", {}).items():
            pr = det.get("primary", {})
            q = pr.get("q_bh", float("nan"))
            rej = pr.get("reject_bh", False)
            lines.append(f"| {cancer} | {pr.get('total_mut', 0)} | "
                         f"{pr.get('recall_model', 0):.4f} | {pr.get('recall_baseline', 0):.4f} | "
                         f"{pr.get('ratio', float('nan')):.3f} | "
                         f"{pr.get('p_perm', 1):.2e} | {q:.3g} | {'Y' if rej else 'N'} |\n")

    if A_def:
        lines.append("\n## Phase 1.6 definitive: per-cancer Analysis A (250 bp + max + 100K perms)\n\n")
        lines.append("| cancer | total_mut | recall_model | recall_cpg | mut_ratio | p_perm | q_bh | reject |\n")
        lines.append("|--------|----------:|-------------:|-----------:|----------:|-------:|------:|:------:|\n")
        for cancer, det in A_def.get("per_cancer", {}).items():
            pr = det.get("primary", {})
            q = pr.get("q_bh", float("nan"))
            rej = pr.get("reject_bh", False)
            lines.append(f"| {cancer} | {pr.get('total_mut', 0)} | "
                         f"{pr.get('recall_model', 0):.4f} | {pr.get('recall_baseline', 0):.4f} | "
                         f"{pr.get('ratio', float('nan')):.3f} | "
                         f"{pr.get('p_perm', 1):.2e} | {q:.3g} | {'Y' if rej else 'N'} |\n")

    lines.append("\n## See also: phase1_6_sweep.png — recall ratio vs window size (visual)\n\n")
    lines.append("## Provenance (Phase 1.6 definitive)\n\n")
    for name, j in [("Analysis A", A_def), ("Analysis B", B_def)]:
        prov = j.get("provenance", {}) if j else {}
        if not prov:
            lines.append(f"### {name}: (provenance missing)\n")
            continue
        lines.append(f"### {name}\n")
        lines.append(f"- git_commit: `{prov.get('git_commit', '?')}`\n")
        lines.append(f"- pre_registration_commit: `{prov.get('pre_registration_commit', '?')}`\n")
        lines.append(f"- panel_scores_cds_sha256: `{prov.get('panel_scores_cds_sha256', '?')}`\n")
        lines.append(f"- phase3_mfe_only_sha256: `{prov.get('phase3_mfe_only_sha256', '?')}`\n")
        lines.append(f"- window_size_bp: {prov.get('window_size_bp', '?')}\n")
        lines.append(f"- aggregator: {prov.get('aggregator', '?')}\n")
        lines.append(f"- run_timestamp: {prov.get('run_timestamp', '?')}\n\n")

    out = PANEL_DIR / "COMPARISON_PHASE1_VS_1_5_VS_1_6.md"
    with open(out, "w") as f:
        f.writelines(lines)
    print(f"Wrote {out}")

    # FINAL_REPORT_PHASE1_6.md
    final = []
    final.append("# Final Report Phase 1.6 - pcawg_tcw_panel (MFE-only regime)\n\n")
    final.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M')}\n\n")
    final.append("Pre-reg: PRE_REGISTRATION_PHASE1.6.md (git commit `e790cd5`).\n\n")
    final.append("## Three-way table\n\n")
    final.append(open(out).read())
    final.append("\n\n## Outcome interpretation\n\n")
    a_pass = bool(A_def.get("pass_criteria", {}).get("PASS")) if A_def else False
    b_pass = bool(B_def.get("pass_criteria", {}).get("PASS")) if B_def else False
    if a_pass and b_pass:
        final.append("Phase 1.6 PRIMARY = **PASS** (A and B both pass all 4 criteria at 100K perms).\n")
    elif a_pass or b_pass:
        final.append(f"Phase 1.6 PRIMARY = MIXED (A PASS={a_pass}, B PASS={b_pass}).\n")
    else:
        final.append("Phase 1.6 PRIMARY = **FAIL** (strict per pre-reg, no relaxation).\n")
    final.append("\n## Reusable artifacts (unchanged across phases)\n\n")
    final.append("- `panel_scores_cds.parquet` (8.45M-position scored panel; sha256 in provenance)\n")
    final.append("- `phase3_mfe_only.pt`, `apobec1_head_mfe_only.pt`\n")
    final.append("- `~/data/panel/rnafm_cds_kept/*.npz` on ai-gpu2 (~22 GB; permanent embed cache)\n")
    final.append("- All scripts in `scripts/gcp_panel/`\n")
    final.append("- Pre-reg trail: `PRE_REGISTRATION_PHASE1.md` (a350c26), `PRE_REGISTRATION_PHASE1.5.md` (8f4462e), `PRE_REGISTRATION_PHASE1.6.md` (e790cd5)\n")
    out2 = PANEL_DIR / "FINAL_REPORT_PHASE1_6.md"
    with open(out2, "w") as f:
        f.writelines(final)
    print(f"Wrote {out2}")

    flag = PANEL_DIR / "PHASE_1_6_DONE.flag"
    flag.write_text(f"Phase 1.6 complete at {time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
    print(f"Wrote {flag}")
    return 0


if __name__ == "__main__":
    main()
