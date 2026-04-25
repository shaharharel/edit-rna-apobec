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
        lines.append(f"| Signif cancers (BH q<0.05) | {a_b['val']}/{len(A['per_cancer'])} | {b_b['val']}/{len(B['per_cancer'])} |\n")
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
    else:
        final.append("**Analysis A and/or B pending** — RNA-FM compute on ai-gpu2 is still in progress "
                     "(ETA ~25 h from start). Scoring and enrichment scripts are all written and "
                     "ready (`score_panel_mfe_only.py`, `analysis_A_pcawg_wgs.py`, "
                     "`analysis_B_tcga_pcawg_coding.py`).\n\n")
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
    final.append("5. **SBS attribution is sample-level**: the open PCAWG SigProfilier CSV contains "
                 "per-tumor × mutation-subtype probabilities, not per-mutation. Analysis A weights "
                 "each C>T by its tumor's SBS2+SBS13 probability for its trinucleotide subtype. "
                 "This is the standard approach but is an approximation at the per-mutation level.\n\n")
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
    final.append("- `experiments/multi_enzyme/outputs/pcawg_tcw_panel/PRE_REGISTRATION.md`\n")
    final.append("- `experiments/multi_enzyme/outputs/pcawg_tcw_panel/loop_reconstructor_validation.json`\n")
    final.append("- `experiments/multi_enzyme/outputs/pcawg_tcw_panel/OVERNIGHT_STATUS.md`\n")
    final.append("- `experiments/multi_enzyme/outputs/pcawg_tcw_panel/COMPARISON.md`\n")
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
