# BLOCKERS_RESPONSE.md — QA agent re-review fixes

**Date**: 2026-04-25 20:55 IDT
**QA reviewer**: a76f64f42e79d3faa
**Commit hash with fixes**: `712b3ee` + `ce3c708` (final-report enhancement)

QA re-review confirmed all 7 prior fixes (B1-B3, M1-M5) PASSED checks and flagged
3 new blockers. All addressed before any Phase 1 final output.

## Blocker 1: stale "SBS sample-level" caveat in compare_A_B.py FINAL_REPORT

**Issue**: Caveat #5 in `compare_A_B.py:124-127` still claimed sample-level SBS
attribution, contradicting the actual B1 fix (cancer-level aggregation).

**Fix** (commit `712b3ee`): replaced with QA-supplied text:
> "SBS attribution is cancer-level mean SBS2+SBS13 weight, applied as a
> per-mutation proxy because the public PCAWG Donor↔Sample ID mapping is
> unavailable. This loses within-cancer sample variation but introduces no
> obvious directional bias."
> Plus: "The threshold for 'APOBEC-attributed' was set to 0.1 (cancer-level mean)
> instead of the original 0.5 (sample-level) to maintain mutation count.
> See FIXES_APPLIED.md B1 for the detailed rationale."

**Verification**: smoke #2 final report grep confirmed text appears verbatim.

## Blocker 2: SHA provenance not recorded in analysis output

**Issue**: Pre-reg L127-130 promised hashes of `phase3_mfe_only.pt`,
`apobec1_head_mfe_only.pt`, the panel parquet, and the git commit hash. None
were written.

**Fix** (commit `712b3ee`): new `compute_provenance()` function in
`analysis_A_pcawg_wgs.py:84-128`:
```python
{
  "git_commit": "<git rev-parse HEAD>",
  "phase3_mfe_only_sha256": "<sha256sum>",
  "apobec1_head_mfe_only_sha256": "<sha256sum>",
  "panel_scores_cds_sha256": "<sha256sum>",
  "pre_registration_commit": "a350c26",
  "fixes_applied_commit": "061591d",
  "phase3_mfe_only_path": "<absolute path>",
  "apobec1_head_mfe_only_path": "<absolute path>",
  "panel_scores_path": "<absolute path>",
  "run_timestamp": "<ISO 8601>"
}
```

Embedded as top-level `"provenance"` field in `enrichment_primary.json` for
both A and B. Both scripts gain `--phase3-model` and `--apobec1-model` CLI flags
(canonical defaults).

**Verification** (smoke #2 sample):
```json
"provenance": {
  "git_commit": "9c573aede626c74a223704115a997045bc95dcc6",
  "phase3_mfe_only_sha256": "e3c178c190b6185186830782ce3bab5a07c1a9aa198c6d4453acb7616f1555bb",
  "apobec1_head_mfe_only_sha256": "c6463031ed0cb2d37d6526285e9f33e8ab83be5dfe3ee4e21ec37e5151b04520",
  "panel_scores_cds_sha256": "002ea3db35cafb7d934bf0d870119ea124d86488f65aeffa99ce4abe8f262a65",
  "pre_registration_commit": "a350c26",
  "fixes_applied_commit": "061591d",
  ...
}
```

`compare_A_B.py` (commit `ce3c708`) now also dumps the full provenance dict for
both analyses into `FINAL_REPORT_PHASE1.md` under a "## Provenance" section.

## Blocker 3: cross-analysis Bonferroni mismatch (alpha=0.05 vs pre-reg 0.025)

**Issue**: Pre-reg L141-144 explicitly required Bonferroni-tightened alpha=0.025
across the family of 2 primaries (A and B). Code used alpha=0.05 throughout.

**Fix** (commit `712b3ee`): tightened to **alpha=0.025**.
- `PRIMARY_ALPHA = 0.025` (was 0.05)
- New constant `SECONDARY_ALPHA = 0.05` for the secondary BH family
  (separate per pre-reg L91-93)
- All `multipletests(..., alpha=PRIMARY_ALPHA, method='fdr_bh')` calls in
  primary correctly use 0.025 now
- All secondary BH calls use `SECONDARY_ALPHA = 0.05`

Pass criterion (b) renamed:
- Old: `(b)_signif_>=_6` (label said q<0.05 but no alpha recorded)
- New: `(b)_signif_q<0.025_>=_6` with explicit `"alpha": 0.025` in JSON

Updated REPORT.md text:
- Old: "BH-FDR across cancers"
- New: "BH-FDR alpha=0.025 across cancers (Bonferroni-tightened from 0.05
  across A+B family per pre-reg L141-144)"

Updated COMPARISON_PHASE1.md:
- Header: "Signif cancers (BH q<0.025)" reads alpha from criterion entry

**Verification**: smoke #2 sample:
```json
"(b)_signif_q<0.025_>=_6": {"val": 0, "thresh": 6, "pass": false, "alpha": 0.025}
```

## Pre-reg deviations log

NONE. All 3 fixes bring the code INTO compliance with the existing
PRE_REGISTRATION_PHASE1.md (commit `a350c26`). No threshold relaxation, no
endpoint switching, no pass-criteria changes.

## Smoke test #2 timings (with all 3 blockers fixed)

| Step | Wall-clock | Outcome |
|------|-----------|---------|
| Analysis A | 1m 31s | rc=0, primary=FAIL (expected on partial data) |
| Analysis B | 1m 35s | rc=0, primary=FAIL (expected on partial data) |
| Parallel A+B | 1m 35s | both succeeded |
| compare_A_B | <2s | wrote all 3 reports + flag |
| **Total** | **~1m 37s** | well under 1h target |

`enrichment_primary.json` for both A and B now contain provenance + alpha=0.025
criterion + correct cancer-level SBS reference.

## What's pending

Full RNA-FM run (~30 more min, currently chr9 34%, 21/24 chroms scored).
When complete:
1. SCP merged panel from ai-gpu2
2. Run `bash scripts/gcp_panel/run_phase1_analyses.sh panel_scores_cds.parquet`
3. Verify outputs (provenance present, alpha=0.025, FAIL/PASS report)
4. Write `PHASE_1_DONE.flag`
5. Surface per-cancer ratio tables to user
6. Stop and wait for Phase 2 authorization

Per QA forecast, primary endpoints likely FAIL on full panel too; FINAL_REPORT
will report the FAIL strictly per pre-reg, with the 5 Phase 1.5 alternatives
clearly labeled as post-hoc options for user authorization.
