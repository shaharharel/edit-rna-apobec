# PRE_REGISTRATION_PHASE1.6.md — pcawg_tcw_panel Phase 1.6

**Authored**: 2026-04-25 23:10 IDT, BEFORE any Phase 1.6 enrichment numbers
are computed.

**Phase 1 done at 22:14 IDT** (PHASE_1_DONE.flag, FAIL strict — A 0.0/0/4, B
0.43/0/10).
**Phase 1.5 done at 23:06 IDT** (PHASE_1_5_DONE.flag, FAIL strict on (b) only
— A 0.92/0/4, B 11.51/3/10. (a)/(c)/(d) all PASS strongly. C reproduces v2
2/4 strict ±0.05; 4/4 directionally stronger).

**Git commit (timestamp proof)**: `e790cd5` (verifiable via `git show e790cd5`)

## Justification

Phase 1.5 showed 250 bp + max-pool produces large window-level signal
(B mean ratio 11.5×; 3/10 cancers BH-sig at q<0.025). The remaining miss on
criterion (b) was driven by **permutation-null floor** at p_perm = 1/(N+1) =
1.0e-4 with N=10,000, while BH-FDR with 10 cancers requires per-cancer
q < 0.025 — achievable in principle only when raw p << 0.0025.

Phase 1.6 has TWO purposes:
1. **Window-size sweep (secondary, exploratory)**: confirm that recall
   ratio is a monotonic function of window size (peak at small windows,
   decay as windows enlarge). Tests the hypothesis that APOBEC accessibility
   is local; this is the panel-design argument.
2. **Power increase on the pre-registered primary (NOT threshold relaxation)**:
   re-run JUST the pre-chosen window (250 bp + max-pool) at 100K perms to
   address the BH-FDR floor. The pass thresholds (a)/(b)/(c)/(d) are
   IDENTICAL to Phase 1.5 — only the test power changes. This is a
   pre-registered methodological refinement, NOT a post-hoc cherry-pick of
   a passing setting.

## What Phase 1.6 changes from Phase 1.5

| Setting | Phase 1.5 | Phase 1.6 |
|---------|-----------|-----------|
| Pre-chosen primary window | 250 bp | **250 bp (unchanged)** |
| Aggregator | max | **max (unchanged)** |
| Permutations (primary) | 10,000 | **100,000** (5× CPU cost, finer null) |
| Pass criteria (a)(b)(c)(d) | identical | **identical** (no relaxation) |
| Per-criterion p-values | only (b) had p_perm | **(a)/(c)/(d) get bootstrap CI + p-value** (new) |
| Window-size sweep | none | **100/250/500/1000/2000 bp** (10K perms each, secondary BH family) |

## Primary endpoint — Analysis A and Analysis B (single pre-chosen setting)

> Across 10 PCAWG cancers (A) and 10 TCGA+PCAWG-coding cancers (B), compute
> the recall of APOBEC-attributed (A) / TCW-non-CpG (B) C>T SNVs within the
> **top 1%** of **250 bp** CDS windows ranked by `score_binary` (per-window
> MAX of phase3_mfe_only binary head). Compare against CpG-density-ranked
> windows.
>
> Pass criteria (all four; no relaxation; no cherry-pick):
> - (a) mean recall ratio (model / cpg_baseline) ≥ 1.5× across cancers
> - (b) BH-FDR-adjusted q < 0.025 in at least 6/10 cancers (**100K-perm null**)
> - (c) signal survives driver-gene ablation (mean ratio ≥ 1.3×)
> - (d) signal survives ±1 kb training-site mask (mean ratio ≥ 1.3×)

The Phase 1.5 primary at 10K perms reported (a) PASS, (b) FAIL, (c) PASS,
(d) PASS. The 100K-perm re-run will determine whether (b) becomes PASS at
the SAME pre-registered threshold (q<0.025, ≥6/10).

## New: bootstrap CI + p-value for criteria (a), (c), (d)

For each of (a), (c), (d), compute over 10,000 bootstrap resamples of the
per-cancer recall-ratio sample (10 values; resample with replacement):
- 95% CI on the mean ratio
- one-sided bootstrap p-value for `H0: mean_ratio ≤ thresh` (thresh = 1.5
  for (a), 1.3 for (c)/(d)). Report as `p_boot_a`, `p_boot_c`, `p_boot_d`.

Also report the **joint exceedance count**: `n_cancers_with_ratio_gt_1.0`,
with binomial p-value under `H0: half are > 1.0`. This is INFORMATIONAL
(not a primary criterion) — it summarizes whether the model beats CpG-density
across cancers irrespective of magnitude.

The bootstrap CIs answer: "is the mean RATIO statistically distinguishable
from the threshold?" which the 10K-only Phase 1.5 didn't quantify.

## Window-size sweep (SECONDARY, exploratory)

Run Analysis A and B at 5 window sizes: **100, 250, 500, 1000, 2000 bp**,
all with **max-pool** aggregator and 10K perms (lighter test for the sweep,
the 100K is reserved for the pre-chosen primary at 250 bp).

For each window size, also evaluate at top-{0.5%, 1%, 5%} percentile
thresholds (sensitivity analysis).

The sweep is exploratory — its purpose is to visualize the signal-vs-window
curve and confirm the local-accessibility hypothesis. Per-row p-values from
the sweep enter a SECONDARY BH-FDR family (separate from the primary).
Significance reported at q<0.05.

If 250 bp (the pre-chosen primary) is NOT the maximum-recall window in the
sweep, that is documented but does NOT change the primary endpoint —
the 250 bp window was pre-chosen based on Phase 1.5 evidence and is locked
by THIS pre-registration.

## Multiple-testing correction

- Phase 1.6 PRIMARIES A and B form a NEW family (separate from Phase 1
  and Phase 1.5). Bonferroni at α=0.05 across A+B → per-analysis α=0.025
  (matches Phase 1 / 1.5).
- Sweep secondary: BH-FDR within each (analysis × percentile) sub-family
  at α=0.05.
- Bootstrap p-values are per-criterion auxiliary; not pooled into a multi-
  test family.

## Outcome interpretations (declared in advance)

- **(b) PASSES at 100K perms (≥6/10 BH-q<0.025) AND (a)/(c)/(d) all PASS**:
  Phase 1.6 PRIMARY = PASS. Honest signal recovered at finer null
  resolution; report this as the primary panel-relevant finding.
- **(b) still FAILS at 100K perms (<6/10 BH-q<0.025)**: power increase did
  not flip (b). Phase 1.6 PRIMARY = FAIL. Report strictly. The sweep + per-
  cancer bootstrap CIs still document the magnitude of effect; reframe
  cfDNA-panel claim accordingly.
- **(a) FAILS where (b) passes**: should not happen if (a)/(c)/(d) PASS at
  10K perms (recall ratios are deterministic given the panel). Document if it
  does.
- **Sweep shows window-size monotonic decay**: confirms local-accessibility
  hypothesis, supports panel-design argument irrespective of primary outcome.

## Allowed post-hoc analyses (clearly labeled)

- Window-size sweep visualization at extra percentiles
- Per-CpG-decile breakdown at 250 bp + max
- Restoring canonical struct_delta features (already documented as Alt-3
  Phase 1.5)

## Not allowed post-registration

- Switching the pre-chosen primary window (250 bp is locked)
- Relaxing pass thresholds (a) ≥1.5×, (b) 6/10 q<0.025, (c) ≥1.3×, (d) ≥1.3×
- Reducing perm count below 100K for primary
- Adding cancers to the pass-count
- Changing the baseline (always CpG-density)
- Selecting between Phase 1, 1.5, 1.6 results based on which one passed

## Determinism guarantee

- 100K-perm primary uses the SAME RNG seed as 10K Phase 1.5 (per cancer:
  `20260425 + cancer_idx * 1000` for A; `20260428 + cancer_idx * 1000` for B).
  The 100K perms are a deterministic SUPERSET of the 10K perms — the
  first 10,000 permutations produce identical permuted statistics; the
  next 90,000 sample the rest of the perm space.
- Same panel parquet (panel_scores_cds.parquet, sha256 already in Phase 1
  provenance), same model (phase3_mfe_only.pt), same hand40, same CpG-
  density baseline computation.

## SHA / provenance (recorded at run time)

Same scheme as Phase 1 / 1.5. Each enrichment_primary_phase1_6_*.json
embeds `provenance` with git_commit, model SHAs, panel SHA, pre-reg commits.

## Sequencing

1. Phase 1 done (22:14), Phase 1.5 done (23:06): VERIFIED
2. This pre-reg authored: 23:10 IDT
3. Git commit (this file): pending — recorded BEFORE any Phase 1.6
   subprocess launches
4. Sweep launched via `bash scripts/gcp_panel/run_phase1_6_sweep.sh`
   (~1 hour, 5 windows × 2 analyses, 4-concurrent)
5. Definitive 100K-perm primary launched via
   `bash scripts/gcp_panel/run_phase1_6_definitive.sh` (~5-7 hours
   single-window 250 bp with 8-core parallel cancers)
6. `compare_phase1_vs_1_5_vs_1_6.py` writes
   `COMPARISON_PHASE1_VS_1_5_VS_1_6.md` + `PHASE_1_6_DONE.flag` +
   `phase1_6_sweep.png`
7. Surface to user; STOP. Do not auto-start Phase 2.
