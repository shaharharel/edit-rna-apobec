# Pre-Registration — Lei BE4max Preview Analysis (Phase 1a-preview)

**Locked**: 2026-04-25, before opening Lei BED contents.
**Authored by**: this-session execution agent under coordinator instruction.
**Status**: PREVIEW run. NOT publication-grade. Same hypotheses + decision tree will apply to (a) post-A1-retrain rerun, and (b) Doman SRA reanalysis when pipeline is built.

---

## Scope

- N=565 Lei BE4max gRNA-independent off-target sites. Borderline statistical power.
- This run answers a single go/no-go question: does intrinsic APOBEC substrate preference, learned from RNA editing data, predict where BE4max deposits gRNA-independent off-targets?
- Result feeds two downstream decisions: (a) whether to invest in Doman SRA reanalysis (~1 week of pipeline build); (b) whether the existing APOBEC1 head needs immediate retraining or works as-is.

## Background

Cytidine base editors (CBEs) like BE3/BE4 are deployed in clinical trials (Verve VERVE-101, Beam BEAM-101). gRNA-independent off-targets — random C deamination at sites where ssDNA is transiently exposed (R-loops, replication forks) — are the primary FDA safety concern, particularly for A3A-fused CBEs. No predictor of these off-targets exists today beyond TCW motif scanning. Our hypothesis is that intrinsic APOBEC enzyme substrate preference, learned from RNA editing data, transfers to predict gRNA-independent BE off-target landing.

## Datasets used (locked)

**Positives**: union of 4 Lei 2021 (Nat Methods, GSE151265) Detect-seq BED files for BE4max-treated cells:
- `GSE151265_293T-VEGFA-Detect-seq_pRBS.bed.gz` (n≈349)
- `GSE151265_293T-HEK4-Detect-seq_pRBS.bed.gz` (n≈146)
- `GSE151265_MCF7-HEK4-Detect-seq_pRBS.bed.gz` (n≈37)
- `GSE151265_293T-EMX1-Detect-seq_pRBS.bed.gz` (n≈33)
- Total: ~565 sites

The 5th file (`GSE151265_MCF7-RNF2-Detect-seq_on-target.bed.gz`) is the on-target editing window and is **excluded** from positives.

Coordinate space: hg38 (Lei native). No liftover.

## Window definition (locked)

For every site (positive or control): 201-nt window from `data/raw/genomes/hg38.fa`, centered on the called off-target C at position 100 (0-indexed). If the BED reports the off-target on the reverse strand, the window is reverse-complemented so the deaminated C is at position 100 on the forward strand of the window.

## Background controls (locked)

For each positive site, sample **5** motif-matched random Cs:
- Same chromosome
- Within ±5 kb of the positive site
- Identical trinucleotide context (X-C-Y at the central position)
- Not within ±10 nt of any Lei positive
- Not within any sgRNA near-cognate window (defined below)

Sampling is with replacement at the population level but without replacement within a positive's neighborhood (no two controls share the exact same position).

If <5 matches available within ±5 kb for a positive, expand to ±10 kb. If still <5, accept fewer (logged).

## sgRNA-independent filter (locked)

Each Lei BED corresponds to one sgRNA (VEGFA, HEK4, EMX1, RNF2). Although the Detect-seq `_pRBS` files are already labeled gRNA-independent by the Lei pipeline, we apply an additional independent filter:

For each of the 5 sgRNAs (VEGFA, EMX1, HEK4 spacer in 293T, HEK4 spacer in MCF7, RNF2), compute near-cognate sites in hg38 using up to **4 mismatches + NGG PAM** (Cas-OFFinder-equivalent). Exclude any positive OR control within ±100 nt of any near-cognate.

Both filtered and unfiltered analyses are reported. Primary endpoint uses the filtered set.

## Models tested (locked, all scored in one pass)

1. **Phase3 binary head** (`phase3_mfe_only.pt`) — multi-enzyme generic APOBEC predictor
2. **Phase3 A3A adapter**
3. **Phase3 A3B adapter**
4. **Phase3 A3G adapter**
5. **Phase3 A3A_A3G adapter**
6. **Phase3 Neither adapter**
7. **APOBEC1 head** (`apobec1_head_mfe_only.pt`, current — pre-retrain)
8. **XGB hand40** — interpretable baseline trained on multi-enzyme v3
9. **Motif-only ablation** — XGB on motif 24-d only
10. **StructOnly ablation** — XGB on (loop 9-d + struct delta 7-d) only

All models receive identical inputs: hand40 features computed from the 201-nt window via `src/data/apobec_feature_extraction.py`. Phase3 also receives RNA-FM original + delta embeddings (640+640 dim) computed fresh from the 201-nt window.

## Hypotheses (locked)

**H1 (primary)**: At least one APOBEC-substrate-trained model (heads 1–8) achieves OR > 1.5 at p90 with Fisher's exact p < 0.05 on the sgRNA-filtered Lei positive vs control set.

**H2 (specificity)**: The strongest-performing model achieves OR ≥ 1.5× the motif-only ablation OR.

**H3 (matched-enzyme)**: For BE4max (rAPOBEC1-derived editor), APOBEC1 head OR ≥ A3A adapter OR. Open prediction — if H3 fails but H1 holds, conclusion is the current A1 head is too weak (retrain priority confirmed).

**H4 (cofactor caveat)**: If APOBEC1 head OR ≈ 1.0 while Phase3 binary or A3A adapter succeed, this is consistent with cofactor-dependent A1 substrate preference not transferring to cofactor-free BE context.

## Statistical analysis (locked)

For each (model, threshold) pair, build a 2×2 contingency table:
- (positive above threshold, positive below) × (control above, control below)

Threshold: p90 (top 10%), p95 (top 5%), p99 (top 1%) of the **combined** positive+control score distribution.

- Test: Fisher's exact, two-sided.
- Correction: Benjamini-Hochberg FDR across (10 models × 3 thresholds × 2 filter conditions = 60 tests). Pre-registered as a single multiple-testing family.
- 95% CI for OR: 1000 bootstrap resamples (resample positives + controls independently, recompute OR).

## Decision criteria (locked)

- **Escalate to Path B (Doman SRA reanalysis)**: ANY model achieves OR > 1.5 at p90 with BH-q < 0.05 on the sgRNA-filtered set AND H2 holds.
- **A1 retrain becomes top priority**: APOBEC1 head fails H3 (A1 head OR < A3A adapter OR) AND another APOBEC-trained head (heads 1–6) passes H1.
- **Stop / pivot the line entirely**: All models OR < 1.3 across all three thresholds with BH-q > 0.05 on both filter conditions. Report as a clean negative result. Reconsider whether RNA-substrate-trained APOBEC preferences transfer to BE off-target prediction at all.

## What this run is NOT testing

- gRNA-dependent off-targets (filtered out by design)
- Editor variants other than BE4max (no A3A-BE, no evoCDA-BE, no engineered low-OT)
- A>I editing or any non-cytidine deamination
- Site-level editing **rate** prediction (binary site / no-site only)
- Causal claims about why the model predicts what it predicts — this is enrichment only

## Pre-known caveats carried into the result write-up

1. **N=565 small**: ORs in 1.3–1.7 will have wide CIs. Anything subtle is undetectable here; we'll see it (or not) in Doman.
2. **Single experimental platform** (Detect-seq). Doman WGS post-this-run is the orthogonal-method confirmation.
3. **APOBEC1 head trained on partly-noisy data**: ~78 mouse Davidson sites + 206 "Neither" sites defined by exclusion. A retrain on cleaner A1 data is in flight.
4. **Cell-type effect not separable**: 293T vs MCF7 — only ~37 MCF7 sites, can't power a per-cell-type analysis.
5. **BE4max ≠ BE4 exactly**: BE4max has codon-optimized rAPOBEC1; substrate preference assumed identical.
6. **Cofactor caveat for APOBEC1 head**: trained on RNA editing data where A1+A1CF complex preferences dominate. BE has no cofactor. Retain this as primary interpretation if A1 head fails while others succeed.
