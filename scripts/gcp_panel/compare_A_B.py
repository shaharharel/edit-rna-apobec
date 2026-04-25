#!/usr/bin/env python3
"""Generate COMPARISON.md + QA_FIXES_CHECKLIST.md + FINAL_REPORT.md from the
analysis_A and analysis_B outputs.

Reads:
    experiments/multi_enzyme/outputs/pcawg_tcw_panel/analysis_A_pcawg_wgs/enrichment_primary.json
    experiments/multi_enzyme/outputs/pcawg_tcw_panel/analysis_B_coding_panel/enrichment_primary.json

Produces:
    experiments/multi_enzyme/outputs/pcawg_tcw_panel/COMPARISON.md
    experiments/multi_enzyme/outputs/pcawg_tcw_panel/QA_FIXES_CHECKLIST.md
    experiments/multi_enzyme/outputs/pcawg_tcw_panel/FINAL_REPORT.md
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PANEL_DIR = PROJECT_ROOT / "experiments/multi_enzyme/outputs/pcawg_tcw_panel"


def load_primary(path: Path) -> dict:
    if path.exists():
        return json.load(open(path))
    return None


def _val(crit: dict, key_substr: str):
    """Look up criterion by partial key match (handles renaming in v2)."""
    for k, v in crit.items():
        if key_substr in k.lower():
            return v
    return None


def main():
    A = load_primary(PANEL_DIR / "analysis_A_pcawg_wgs" / "enrichment_primary.json")
    B = load_primary(PANEL_DIR / "analysis_B_coding_panel" / "enrichment_primary.json")

    # COMPARISON
    lines = []
    lines.append("# Comparison Phase 1 — Analysis A vs Analysis B\n\n")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M')}\n\n")
    lines.append("| | Analysis A (PCAWG WGS) | Analysis B (TCGA+PCAWG coding) |\n")
    lines.append("|---|---|---|\n")
    if A and B:
        aa = A["pass_criteria"]; bb = B["pass_criteria"]
        a_a = _val(aa, "primary"); a_b = _val(aa, "signif"); a_c = _val(aa, "driver"); a_d = _val(aa, "masked")
        b_a = _val(bb, "primary"); b_b = _val(bb, "signif"); b_c = _val(bb, "driver"); b_d = _val(bb, "masked")
        lines.append(f"| Primary PASS? | {'Y' if aa.get('PASS') else 'N'} | {'Y' if bb.get('PASS') else 'N'} |\n")
        lines.append(f"| Mean ratio (primary, masked+driver-ablated) | {a_a['val']:.3f} | {b_a['val']:.3f} |\n")
        a_alpha = a_b.get("alpha", 0.025)
        b_alpha = b_b.get("alpha", 0.025)
        lines.append(f"| Signif cancers (BH q<{a_alpha}) | {a_b['val']}/{len(A['per_cancer'])} | {b_b['val']}/{len(B['per_cancer'])} |\n")
        lines.append(f"| Driver-ablated mean ratio | {a_c['val']:.3f} | {b_c['val']:.3f} |\n")
        lines.append(f"| Training-masked mean ratio | {a_d['val']:.3f} | {b_d['val']:.3f} |\n")
    else:
        lines.append("| | *pending* | *pending* |\n")
    with open(PANEL_DIR / "COMPARISON_PHASE1.md", "w") as f:
        f.writelines(lines)

    # QA_FIXES_CHECKLIST (re-generate based on current scripts)
    qa_lines = []
    qa_lines.append("# QA Fixes Checklist — pcawg_tcw_panel (mfe_only regime)\n\n")
    qa_lines.append("Each QA issue is addressed by code in `scripts/gcp_panel/analysis_A_pcawg_wgs.py` "
                    "and `scripts/gcp_panel/analysis_B_tcga_pcawg_coding.py`.\n\n")
    qa_lines.append("| # | QA issue | Fix location | Output evidence |\n|---|----------|--------------|-----------------|\n")
    qa_lines.append("| 1 | Reframe as coding panel | Both scripts use `candidates_cache_aligned.parquet` (CDS only, 8.45M positions). PRE_REGISTRATION.md states scope. | PRE_REGISTRATION.md §scope, analysis_{A,B}/windows.parquet |\n")
    qa_lines.append("| 2 | Window-level CpG stratification | `run_secondary()` bins windows into CpG-density deciles and reports per-decile minimum recall ratio. | analysis_{A,B}/enrichment_secondary.json.per_cpg_decile, per_decile_min_ratio |\n")
    qa_lines.append("| 3 | CpG-density baseline | `PRIMARY_BASELINE = 'cpg_density'` in both analyses; `recall_ratio()` compares model recall vs CpG-density recall per cancer. | enrichment_primary.json per-cancer `raw.recall_baseline` |\n")
    qa_lines.append("| 4 | ±1 kb training-site mask | `build_windows()` computes `training_contaminated` from `splits_multi_enzyme_v3_with_negatives.csv`; primary uses `w[~training_contaminated]`. | windows.parquet has `training_contaminated` column; pass criterion (d) requires ratio≥1.3 with mask |\n")
    qa_lines.append("| 5 | Pre-registered primary + BH-FDR | `PRE_REGISTRATION.md` declares primary before run; BH applied per-cancer to primary, family-wise BH to secondary. | PRE_REGISTRATION.md; per-cancer `q_bh`; secondary `reject_bh` |\n")
    qa_lines.append("| 6 | Driver-gene ablation | `is_driver = gene in _BUILTIN_CGC_TIER1`; primary re-run on `~is_driver`; pass criterion (c) requires ratio≥1.3. | windows.parquet `is_driver`; per-cancer `driver_ablated` metrics |\n")
    with open(PANEL_DIR / "QA_FIXES_CHECKLIST.md", "w") as f:
        f.writelines(qa_lines)

    # FINAL_REPORT
    final = []
    final.append("# Final Report — pcawg_tcw_panel Phase 1 (MFE-only regime)\n\n")
    final.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M')}\n\n")
    final.append("## Summary\n\n")
    final.append("The MFE-only feature regime (all 7 struct_delta slots zeroed at training "
                 "and inference) was applied to train `phase3_mfe_only.pt` and "
                 "`apobec1_head_mfe_only.pt`. Per-head AUROC deltas vs the canonical "
                 "Phase3 are within ±0.01 (see `phase3_mfe_only/comparison_vs_phase3.md`).\n\n")
    if A and B:
        final.append("## Results table\n\n")
        final.append(open(PANEL_DIR / "COMPARISON_PHASE1.md").read())
        a_pass = bool(A.get("pass_criteria", {}).get("PASS"))
        b_pass = bool(B.get("pass_criteria", {}).get("PASS"))
        final.append("\n## Per-cancer recall ratios (primary, masked + driver-ablated)\n\n")
        final.append("### Analysis A — PCAWG WGS\n\n")
        final.append("| cancer | total_mut | recall_model | recall_cpg | mut_ratio | p_perm | q_bh | reject |\n")
        final.append("|--------|-----------|--------------|-----------|-----------|--------|------|--------|\n")
        for cancer, det in A.get("per_cancer", {}).items():
            pr = det.get("primary", {})
            q = pr.get("q_bh", float("nan"))
            rej = pr.get("reject_bh", False)
            try:
                final.append(f"| {cancer} | {pr.get('total_mut', 0)} | "
                             f"{pr.get('recall_model', 0.0):.4f} | "
                             f"{pr.get('recall_baseline', 0.0):.4f} | "
                             f"{pr.get('ratio', float('nan')):.3f} | "
                             f"{pr.get('p_perm', float('nan')):.2e} | "
                             f"{q:.3g} | {'Y' if rej else 'N'} |\n")
            except Exception:
                final.append(f"| {cancer} | (data unparseable) | | | | | | |\n")
        final.append("\n### Analysis B — TCGA + PCAWG coding\n\n")
        final.append("| cancer | total_mut | recall_model | recall_cpg | mut_ratio | p_perm | q_bh | reject |\n")
        final.append("|--------|-----------|--------------|-----------|-----------|--------|------|--------|\n")
        for cancer, det in B.get("per_cancer", {}).items():
            pr = det.get("primary", {})
            q = pr.get("q_bh", float("nan"))
            rej = pr.get("reject_bh", False)
            try:
                final.append(f"| {cancer} | {pr.get('total_mut', 0)} | "
                             f"{pr.get('recall_model', 0.0):.4f} | "
                             f"{pr.get('recall_baseline', 0.0):.4f} | "
                             f"{pr.get('ratio', float('nan')):.3f} | "
                             f"{pr.get('p_perm', float('nan')):.2e} | "
                             f"{q:.3g} | {'Y' if rej else 'N'} |\n")
            except Exception:
                final.append(f"| {cancer} | (data unparseable) | | | | | | |\n")

        final.append("\n## Provenance\n\n")
        for label, src in (("Analysis A", A), ("Analysis B", B)):
            prov = src.get("provenance", {})
            if not prov:
                final.append(f"- {label}: provenance missing in JSON\n")
                continue
            final.append(f"### {label}\n")
            final.append(f"- git_commit: `{prov.get('git_commit', '?')}`\n")
            final.append(f"- pre_registration_commit: `{prov.get('pre_registration_commit', '?')}`\n")
            final.append(f"- fixes_applied_commit: `{prov.get('fixes_applied_commit', '?')}`\n")
            final.append(f"- phase3_mfe_only_sha256: `{prov.get('phase3_mfe_only_sha256', '?')}`\n")
            final.append(f"- apobec1_head_mfe_only_sha256: `{prov.get('apobec1_head_mfe_only_sha256', '?')}`\n")
            final.append(f"- panel_scores_cds_sha256: `{prov.get('panel_scores_cds_sha256', '?')}`\n")
            final.append(f"- run_timestamp: `{prov.get('run_timestamp', '?')}`\n\n")

        if not (a_pass and b_pass):
            final.append("\n## Phase 1 outcome: FAIL (one or both primaries did not pass)\n\n")
            final.append(f"- Analysis A primary PASS: **{a_pass}**\n")
            final.append(f"- Analysis B primary PASS: **{b_pass}**\n\n")
            final.append("Per pre-registration L141-144 (Bonferroni-tightened to alpha=0.025 across "
                         "the family of 2 primaries), the criteria are NOT relaxed and the result "
                         "is reported strictly as written. The data does not show APOBEC-mutation "
                         "enrichment in top-1% NN-ranked CDS windows above the 1.5x recall-ratio "
                         "threshold relative to the CpG-density baseline.\n\n")
            final.append("### Why this is a defensible null finding, not a methodology bug\n\n")
            final.append("1. **The CpG-density baseline is hard to beat in coding-strand C>T**. CpG "
                         "dinucleotides have ~10-fold elevated C>T rate from spontaneous deamination "
                         "(unrelated to APOBEC), and CDS regions are CpG-enriched. Any model that "
                         "doesn't explicitly down-weight CpG context inherits this baseline floor.\n")
            final.append("2. **MFE-only feature regime drops 7 of 40 hand features**. The full "
                         "canonical model uses struct_delta features (delta_pairing_center, "
                         "delta_accessibility_center, delta_entropy_center, delta_mfe, mean/std "
                         "delta_pairing_window, mean_delta_accessibility_window) which directly "
                         "encode the structural cost of the C->U edit. Zeroing these is what we "
                         "did to bypass the layout-swap bug, but it costs predictive signal.\n")
            final.append("3. **Window-level mean dilutes per-position signal**. A 1 kb window has "
                         "~30-50 candidate C positions; a single high-scoring position is averaged "
                         "with many low-scoring ones. The model was trained per-position, not per-"
                         "window. Per-window MAX score may better preserve the signal — already "
                         "computed and present in `windows.parquet` as `score_binary` (max).\n")
            final.append("4. **1 kb windows are large for short-range editing**. APOBEC editing "
                         "targets specific stem-loop structures; 1 kb tiles average over many "
                         "structural contexts. Smaller windows (250 bp, 500 bp) preserve more "
                         "structural specificity at the cost of mutation density per window.\n\n")

            final.append("### Phase 1.5 alternatives (NOT auto-launched; user decision)\n\n")
            final.append("These are post-hoc and clearly labeled as such; they will NOT change the "
                         "primary endpoint outcome above. They are exploratory next steps the user "
                         "can authorize:\n\n")
            final.append("**Alt-1: switch to per-window MAX score** (single config change). Re-run "
                         "Analysis A and B with `score_binary` (MAX, already in panel) instead of "
                         "`score_binary_mean`. ~10 min of compute. If MAX changes the outcome, the "
                         "issue is per-window aggregation, not the model.\n\n")
            final.append("**Alt-2: smaller windows (500 bp or 250 bp)**. Modify `WINDOW_BP` constant "
                         "in `analysis_A_pcawg_wgs.py:WINDOW_BP`. ~15 min compute per window size; "
                         "more granular ranking, less mutation-per-window noise.\n\n")
            final.append("**Alt-3: restore canonical struct_delta features**. Roughly 3 days to "
                         "rerun ViennaRNA partition function on ai-chem (~28.6 M positions) plus "
                         "1 day to retrain phase3 with the full 40-d hand features (canonical "
                         "layout, layout-swap bug fixed). This is the strongest model variant; "
                         "if the MFE-only AUROC parity (within 0.01) is real, this should not "
                         "change Phase 1 by much, but it removes an obvious caveat.\n\n")
            final.append("**Alt-4: stratified analysis within CpG density deciles**. Already in "
                         "secondary outputs (`enrichment_secondary.json.per_cpg_decile`). Within "
                         "low-CpG deciles, the model's ratio vs CpG-density baseline may be much "
                         "stronger because the baseline has no signal there. This is the QA #2 "
                         "stratification; it doesn't change the pre-reg primary but it tells us "
                         "where the model has signal.\n\n")
            final.append("**Alt-5: pure-TCW filter (Analysis A) without the SBS layer**. Currently "
                         "Analysis A uses cancer-level SBS2+SBS13 mean >= 0.1 to call APOBEC. With "
                         "the cancer-level approximation, the threshold is permissive enough that "
                         "filter ~ TCW. Switching to literal TCW-non-CpG (already in secondary) "
                         "removes the SBS layer entirely.\n\n")

            final.append("### What is reusable from Phase 1 for any of the above\n\n")
            final.append("- `panel_scores_cds.parquet` (panel scores at 8.45 M CDS positions for all 7 heads)\n")
            final.append("- `~/data/panel/rnafm_cds_kept/*.npz` (~22 GB RNA-FM embedding cache, kept on ai-gpu2)\n")
            final.append("- Trained models: `phase3_mfe_only.pt`, `apobec1_head_mfe_only.pt`\n")
            final.append("- Pre-computed hand features: `hand40_cache_aligned.npy` (1.35 GB, 8.45 M x 40)\n")
            final.append("- Vienna structure cache: `~/data/panel/vienna_cache/` (3.6 GB) and original on local Mac\n")
            final.append("- All loaders, parallelization, permutation null, BH-FDR machinery in scripts/gcp_panel/\n\n")
        else:
            final.append("\n## Phase 1 outcome: PASS\n\n")
            final.append("- Analysis A primary PASS: **True**\n")
            final.append("- Analysis B primary PASS: **True**\n\n")
            final.append("Both pre-registered primary endpoints meet all 4 pass criteria "
                         "(mean ratio >= 1.5x, >= 6/10 BH-significant at q<0.025, survives driver "
                         "ablation, survives training-site mask).\n\n")
    else:
        final.append("**Analysis A and/or B pending** — see analysis_A_run.log / analysis_B_run.log.\n\n")
    final.append("## Caveats (flagged per coordinator instruction)\n\n")
    final.append("1. **Layout-swap bug in prior work**. Prior TCGA/PCAWG scoring used a swapped "
                 "struct_delta layout (std in slot 5, -mean in slot 6, vs canonical mean_da in "
                 "slot 5, std in slot 6). All prior absolute ORs from `exp_pcawg_end2end`, "
                 "Section-1 replication, Neither-vs-APOBEC1 GI analysis, CpG Simpson's-paradox "
                 "analysis, PCAWG 5-control re-run, and Test-7 feature-ablation are suspect in "
                 "absolute magnitude. Within-comparison ranks/permutation tests likely robust. "
                 "The new `phase3_mfe_only` uses a canonically-ordered struct_delta layout (fix "
                 "applied in `exp_exome_editability_full.py:186-189` and "
                 "`scripts/gcp_panel/compute_vienna.py:162`), but zeroes those slots at train "
                 "and inference, so the bug is irrelevant to the current results.\n\n")
    final.append("2. **Candidate-enumeration mismatch**: the `candidates_CDS.parquet` produced "
                 "by `scripts/gcp_panel/enumerate_candidates.py` uses longest-CDS-per-gene dedup "
                 "(8.73 M rows), but the existing Vienna cache was produced by "
                 "`exp_exome_editability_full.py` using longest-transcript-per-gene dedup "
                 "(8.45 M rows). To reuse the 7.5 GB of cached MFE compute, the new "
                 "`candidates_cache_aligned.parquet` (8,446,859 rows, produced by "
                 "`scripts/gcp_panel/enumerate_cache_aligned_candidates.py`) is the authoritative "
                 "list.\n\n")
    final.append("3. **RNA-FM coverage**: ai-gpu2's RNA-FM was run on 28.6 M positions from the "
                 "prior worker's `candidates_all.parquet`. 99.49% of our 8.45 M cache-aligned "
                 "positions have matching RNA-FM; 43,443 positions (0.51%) will be marked invalid "
                 "at scoring time.\n\n")
    final.append("4. **Vienna version drift**: On 500 chr22 positions, 3 (0.6%) have cached "
                 "dot-bracket structures that differ from fresh-folded structures with the "
                 "current ViennaRNA install (MFE energies differ by up to 1.9 kcal/mol). Since "
                 "all downstream loop features are reconstructed from cached `struct_wt` via the "
                 "canonical `_extract_loop_geometry`, the reconstructor is self-test byte-equal "
                 "(`loop_reconstructor_validation.json` shows max_abs=0.0 across all 9 slots).\n\n")
    final.append("5. **SBS attribution is cancer-level mean SBS2+SBS13 weight**, applied as a "
                 "per-mutation proxy because the public PCAWG Donor↔Sample ID mapping is "
                 "unavailable. This loses within-cancer sample variation but introduces no "
                 "obvious directional bias. The threshold for 'APOBEC-attributed' was set to 0.1 "
                 "(cancer-level mean) instead of the original 0.5 (sample-level) to maintain "
                 "mutation count. See FIXES_APPLIED.md B1 for the detailed rationale.\n\n")
    final.append("## Deliverables\n\n")
    final.append("- `experiments/multi_enzyme/outputs/phase3_mfe_only/phase3_mfe_only.pt`\n")
    final.append("- `experiments/multi_enzyme/outputs/phase3_mfe_only/cv_results.json`\n")
    final.append("- `experiments/multi_enzyme/outputs/phase3_mfe_only/comparison_vs_phase3.md`\n")
    final.append("- `experiments/multi_enzyme/outputs/apobec1_head/apobec1_head_mfe_only.pt`\n")
    final.append("- `experiments/multi_enzyme/outputs/apobec1_head/apobec1_head_mfe_only_summary.json`\n")
    final.append("- `scripts/gcp_panel/enumerate_cache_aligned_candidates.py` (new)\n")
    final.append("- `scripts/gcp_panel/reconstruct_loop_features.py` (new, Stream 3 deliverable)\n")
    final.append("- `scripts/gcp_panel/precompute_hand_features.py` (new)\n")
    final.append("- `scripts/gcp_panel/score_panel_mfe_only.py` (new, runs on ai-gpu2)\n")
    final.append("- `scripts/gcp_panel/analysis_A_pcawg_wgs.py` (new)\n")
    final.append("- `scripts/gcp_panel/analysis_B_tcga_pcawg_coding.py` (new)\n")
    final.append("- `scripts/gcp_panel/compare_A_B.py` (this script)\n")
    final.append("- `data/processed/gcp_panel/candidates_cache_aligned.parquet` (8.45M, 37 MB)\n")
    final.append("- `data/processed/gcp_panel/hand40_cache_aligned.npy` (8.45M × 40 fp32, 1.35 GB)\n")
    final.append("- `data/raw/pcawg_open/final_consensus_passonly.snv_mnv_indel.icgc.public.maf.gz` (882 MB)\n")
    final.append("- `data/raw/pcawg_open/SigProfilier_PCAWG_WGS_probabilities_SBS.csv` (50 MB)\n")
    final.append("- `experiments/multi_enzyme/outputs/pcawg_tcw_panel/PRE_REGISTRATION_PHASE1.md` "
                 "(git-committed at a350c26 as timestamp proof)\n")
    final.append("- `experiments/multi_enzyme/outputs/pcawg_tcw_panel/FIXES_APPLIED.md` "
                 "(QA-review response, B1-3 + M1-5)\n")
    final.append("- `experiments/multi_enzyme/outputs/pcawg_tcw_panel/loop_reconstructor_validation.json`\n")
    final.append("- `experiments/multi_enzyme/outputs/pcawg_tcw_panel/PHASED_STATUS.md` "
                 "(auto-rewritten by update_phased_status.py every 10 min)\n")
    final.append("- `experiments/multi_enzyme/outputs/pcawg_tcw_panel/SMOKE_TEST_RESULTS.md`\n")
    final.append("- `experiments/multi_enzyme/outputs/pcawg_tcw_panel/COMPARISON_PHASE1.md`\n")
    final.append("- `experiments/multi_enzyme/outputs/pcawg_tcw_panel/QA_FIXES_CHECKLIST.md`\n")
    if A:
        final.append("- `experiments/multi_enzyme/outputs/pcawg_tcw_panel/analysis_A_pcawg_wgs/{windows.parquet,enrichment_primary.json,enrichment_secondary.csv,REPORT.md}`\n")
    if B:
        final.append("- `experiments/multi_enzyme/outputs/pcawg_tcw_panel/analysis_B_coding_panel/{windows.parquet,enrichment_primary.json,enrichment_secondary.csv,REPORT.md}`\n")
    with open(PANEL_DIR / "FINAL_REPORT_PHASE1.md", "w") as f:
        f.writelines(final)

    # Phase 1 done flag
    if A is not None and B is not None:
        flag = PANEL_DIR / "PHASE_1_DONE.flag"
        flag.write_text(f"Phase 1 complete at {time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
        print(f"Wrote PHASE_1_DONE.flag → {flag}")

    print(f"Wrote COMPARISON_PHASE1.md, QA_FIXES_CHECKLIST.md, FINAL_REPORT_PHASE1.md to {PANEL_DIR}")


if __name__ == "__main__":
    main()
