# PRE_REGISTRATION_PHASE1.5.md — pcawg_tcw_panel Phase 1.5

**Authored**: 2026-04-25 22:22 IDT, BEFORE any Phase 1.5 enrichment numbers are
computed. Phase 1 has just completed (PHASE_1_DONE.flag written; both primaries
FAIL with the strict pre-reg criteria, mean ratio A=0.000 / B=0.434 vs threshold
1.5×, q<0.025 cancers 0/4 (A) and 0/10 (B), reported honestly).

**Git commit (timestamp proof)**: *<inserted at first commit; this file is
git-committed BEFORE Phase 1.5 launches>*

## Justification

Phase 1's pre-registered primary endpoint used **MEAN aggregation over 1 kb
windows**. Across 8.45 M CDS positions and 10 cancers, this configuration:
- diluted any peaky per-position model signal across ~30-50 candidate C's per
  window
- could not beat the CpG-density baseline because CpG-elevated C>T from
  spontaneous deamination dominates 1 kb mean-mutation density in coding regions

v2 §1c reported per-position TC+nonCpG OR@p90 of **1.33 (BLCA) / 1.17 (BRCA) /
1.30 (CESC) / 1.21 (LUSC)** under a stratum-pooled threshold and mutation/control
matched controls. That signal is small (<1.5×) but real. Phase 1.5 tests whether
the same panel scores recover that signal at finer resolution.

**This pre-registration is written AFTER Phase 1 results land but BEFORE
looking at any Phase 1.5 enrichment numbers**. The acceptance threshold
structure is identical to Phase 1; we are testing an ALTERNATIVE
HYPOTHESIS (peakier local signal) — not relaxing thresholds.

## What Phase 1.5 changes from Phase 1

| Setting | Phase 1 (registered) | Phase 1.5 (this pre-reg) |
|---------|----------------------|--------------------------|
| Window size | 1000 bp | **250 bp** (4× finer) |
| Per-window score aggregator | MEAN (`score_<head>_mean`) | **MAX (`score_<head>`)** |
| Baseline | CpG-density | CpG-density (unchanged) |
| Cancer set (A) | 10 PCAWG cancers (unchanged) | 10 PCAWG cancers (unchanged) |
| Cancer set (B) | 10 TCGA+PCAWG-coding cancers | unchanged |
| Filter (A) | apobec_signature (cancer-mean SBS≥0.1) | unchanged |
| Filter (B) | tcw_not_cpg | unchanged |
| Test | permutation null + BH-FDR α=0.025 | unchanged |
| Pass thresholds | (a)≥1.5x (b)≥6/10 (c)≥1.3x (d)≥1.3x | **unchanged** |
| Training-mask buffer | ±1000 bp (=±1 window @ 1kb) | ±1000 bp (=±4 windows @ 250bp; recomputed in build_windows) |
| Driver list (M5) | Bailey-curated, no length confounders | unchanged |
| Provenance | git commit + 4 SHAs in JSON | unchanged |

## Primary endpoint — Analysis A (PCAWG WGS)

> Across 10 PCAWG cancers, compute the recall of APOBEC-attributed C>T SNVs
> (cancer-level SBS2+SBS13 mean ≥ 0.1) within the **top 1% of 250 bp CDS
> windows** ranked by `score_binary` (per-window MAX of phase3_mfe_only binary
> head). Compare against the recall obtained by ranking windows by CpG
> dinucleotide density (counted in the same 250 bp window).
>
> Pass criteria (all four):
> - (a) mean recall ratio (model / cpg_baseline) ≥ 1.5× across 10 cancers
> - (b) BH-FDR-adjusted q < 0.025 in at least 6/10 cancers (10K-permutation null)
> - (c) signal survives driver-gene ablation (mean ratio ≥ 1.3×)
> - (d) signal survives ±1 kb training-site mask (mean ratio ≥ 1.3×)

## Primary endpoint — Analysis B (TCGA + PCAWG-coding)

> Same shape as Analysis A but mutation source = combined TCGA-MC3 +
> cBioPortal-PCAWG-coding (10 cancers: blca, brca, cesc, coadread, esca, hnsc,
> lihc, lusc, skcm, stad). Filter = `tcw_not_cpg`. Aggregator = MAX. Window =
> 250 bp.

## Auxiliary endpoint — Analysis C (v2 §1c apples-to-apples reproduction)

This is a TRUST-ANCHOR test, not a primary. It runs alongside Phase 1.5 to
validate that the new MFE-only pipeline reproduces the per-position signal v2
§1c reported.

> Score the v2 §1c cached TCGA mutation/control pairs (deterministic seed=42)
> with phase3_mfe_only.pt (struct_delta zeroed). Compute per-position
> TC+nonCpG OR@p90 (Fisher exact, threshold pooled within stratum). Compare
> against v2 §1c reported numbers:
>   - BLCA: 1.33
>   - BRCA: 1.17
>   - CESC: 1.30
>   - LUSC: 1.21
>
> Pass criterion: at least 3 of 4 cancers within ±0.05 of v2.

## Multiple-testing correction

- Phase 1.5 primaries A and B form a NEW family (separate from Phase 1's
  primaries — same hypothesis structure but tested on a different aggregation).
- Bonferroni across the 1.5 family of 2: per-analysis α = 0.025 (matches
  Phase 1).
- Analysis C is a single trust-anchor comparison; no FDR (it's a reproduction
  with a strict tolerance, not a hypothesis test).

## Outcome interpretations (declared in advance)

- **C reproduces v2 (≥3/4 cancers within ±0.05) AND Phase 1.5 primary passes**:
  signal recovered at finer resolution; the per-position model is real and
  panel-grade at small window size.
- **C reproduces v2 BUT Phase 1.5 primary fails**: model produces a real
  per-position signal that doesn't survive the strict windowed comparison. Honest
  signal-too-small finding; model is real but cfDNA-panel claim must be
  reframed (e.g. "not panel-grade at 250 bp / 1% threshold either").
- **C does NOT reproduce v2 (<3/4 within ±0.05)**: today's pipeline has a
  deeper inconsistency (model layout, feature alignment, control set
  construction). Investigate before drawing further conclusions.
- **Both Phase 1 and Phase 1.5 fail AND C reproduces**: model is real at
  marginal effect size but not panel-grade — reported as honest negative for
  the panel claim.

## Allowed post-hoc analyses (clearly labeled)

- Within-CpG-density-decile per-cancer mut_ratio breakdown (already in
  secondary, with `_phase1_5` suffix)
- Per-cancer recall ratio at additional percentiles (0.1%, 0.5%, 5%) — already
  in secondary
- Per-CpG-decile apples-to-apples comparison Phase 1 vs Phase 1.5 (post-hoc,
  illustrative only)

## Not allowed post-registration

- Switching back to MEAN aggregation if Phase 1.5 fails (would be cherry-pick)
- Relaxing pass thresholds (a)/(b)/(c)/(d)
- Changing the cancer set or filter post-hoc
- Switching the baseline (always CpG-density)
- Selecting between Phase 1 and Phase 1.5 results based on which one passed

## SHA / provenance (recorded at run time)

Same as Phase 1: `provenance` block in each enrichment_primary_phase1_5.json
will contain git_commit, phase3_mfe_only_sha256, apobec1_head_mfe_only_sha256,
panel_scores_cds_sha256 (identical to Phase 1, since Phase 1.5 reuses the same
panel scores), pre_registration_commit (Phase 1: a350c26), plus a new field
`window_size_bp=250` and `aggregator=max`.

## Sequencing

1. Phase 1 done (PHASE_1_DONE.flag present): VERIFIED at 22:14 IDT
2. This pre-reg authored: 22:22 IDT
3. Git commit (this file): pending — will be recorded before any Phase 1.5
   subprocess launches
4. Phase 1.5 launched via `bash scripts/gcp_panel/run_phase1_5_analyses.sh`
5. PHASE_1_5_DONE.flag written by `compare_phase1_vs_phase1_5.py`
6. Surface results to user; STOP. Do not auto-start Phase 2.
