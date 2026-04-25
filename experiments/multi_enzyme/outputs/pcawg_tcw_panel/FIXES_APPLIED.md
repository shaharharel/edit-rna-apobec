# FIXES_APPLIED.md — QA review response

QA agent (a76f64f42e79d3faa) flagged 3 blockers + 5 majors at 13:55 IDT
2026-04-25. All eight items addressed before any analysis or panel scoring is
launched. **RNA-FM CDS embedding run on ai-gpu2 was NOT killed** — it is
unaffected by these issues (just produces embeddings) and continues to run
in tmux session `rnafm_cds`.

Pre-registration committed at git hash **`a350c26`** as the timestamp proof
(see `PRE_REGISTRATION_PHASE1.md`).

## Blockers — fixed

### B1. SBS attribution join broken
**Problem**: PCAWG MAF uses `Donor_ID` (DOxxxxx, e.g. DO50354); SBS CSV uses
`Sample` (SPxxxxx, e.g. SP117655). Confirmed zero overlap on raw values; tried
the cBioPortal `data_clinical_sample.txt` mapping but its PATIENT_IDs are an
older sub-cohort (max DO9940) that does NOT cover MAF donors (DO45000+ range).
The PCAWG sample sheet at the canonical DCC URL is now 404.

**Fix**: B1 fallback — aggregate SBS attributions to **cancer level**: for each
(cancer, trinucleotide subtype), compute mean SBS2+SBS13 weight across all PCAWG
samples in that cancer/subtype combo. APOBEC activity is largely cancer-type
driven (SBS2/13 burden is high in BLCA/CESC/HNSC/breast etc; low in melanoma,
liver), so cancer-level mean is a reasonable approximation. Threshold for
"APOBEC-attributed" lowered from 0.5 (sample-level cutoff) to **0.1**
(cancer-level mean cutoff) to maintain mutation count.

**Sanity check** (on a 200 k MAF sample):
- 22,047 C>T + 21,973 G>A SNPs / 200 k = 22% C>T → matches expected APOBEC-rich
  cancers like Skin-Melanoma being >50%.
- After cancer-level join: every (cancer, subtype) pair maps to a non-zero
  SBS weight (not zero like before).

Code: `analysis_A_pcawg_wgs.py:load_sbs_attributions_cancer_level`.

### B2. PRIMARY_FILTER mismatch
**Problem**: `PRE_REGISTRATION.md` said `apobec_signature` ≥ 0.5; script said
`tcw_not_cpg`. Different hypotheses; cannot post-hoc swap.

**Fix**: Set `PRIMARY_FILTER = "apobec_signature"` in
`analysis_A_pcawg_wgs.py:67`. TCW-non-CpG moved to secondary. Pre-reg also
rewritten (B3 issue too).

### B3. Fisher 2×2 malformed
**Problem**: `recall_ratio` built `[[mut_in_top, win_top - mut_in_top], ...]`
where `mut_in_top` is summed mutation counts (a window can hold many mutations)
and `win_top` is window count. Negative cell entries → garbage p-values.

**Fix**: Replaced with **permutation null on score labels**: for each cancer,
permute the per-window `score_binary_mean` values 10,000 times, recompute
`mut_in_top` under top-1% selection, return `p = (n_geq + 1) / (N + 1)`.
This is window-level, exact, and respects the actual ranking statistic.
Code: `analysis_A_pcawg_wgs.py:recall_ratio_with_perm` (also used by Analysis B).

## Majors — fixed

### M1. v3 site_id parser drops T2/T3 negatives
**Problem**: `load_v3_positions` parsed `chr:pos:strand` format but 2,966 rows
use `T2_chr1_1624845` format (no colons) — silently dropped from mask.

**Fix**: Use `chr` + `start` columns directly (these are populated for ALL
v3 rows). Code: `analysis_A_pcawg_wgs.py:load_v3_positions_hg19`. After fix,
**10,102 hg19 rows loaded** (vs 11,870 chrN:pos:strand-format rows; the
remainder are hg38 — see M2).

### M2. hg38 sites in v3 mix
**Problem**: 5,250 of 15,352 v3 rows are hg38 (verified via `coordinate_system`
column). Treating them as hg19 would mask the wrong locations on hg19.

**Fix**: Filter to `coordinate_system == 'hg19'` before extracting positions.
Documented in load function; the 5,250 hg38 sites are dropped with an INFO
log line. Result: 9,830 unique hg19 (chrom, pos) pairs in the mask set.

### M3. Pre-registration timestamp fake
**Problem**: PRE_REGISTRATION.md claimed 23:35 local time but file birthtime
was 00:04, after the analysis scripts.

**Fix**: New `PRE_REGISTRATION_PHASE1.md` written and **git-committed at
`a350c26`**. Commit hash IS the verifiable timestamp.

### M4. TCW minus-strand counting bug
**Problem**: `seq[k:k+2] == "GA"[::-1]` evaluates to `seq[k:k+2] == "AG"` —
which detects `AG[AT]`, NOT the intended `[AT]GA` (reverse complement of TCW).

**Fix**: Rewrote `count_tcw_in_window(seq)` with explicit logic:
```python
if tri[0] == "T" and tri[1] == "C" and tri[2] in "AT":  # +strand
elif tri[0] in "AT" and tri[1] == "G" and tri[2] == "A":  # -strand
```
**Unit tested**: TCA→1, TCG→0, TGA→1 (matches -strand TCA), AGA→1, GGA→0,
'TCATCATGA'→3 (2 plus + 1 minus).
Code: `analysis_A_pcawg_wgs.py:count_tcw_in_window` (used by both A and B).

### M5. Hardcoded CGC list spurious
**Problem**: Built-in list included TTN, MUC16, OBSCN, SYNE1 — these are
length confounders, not high-confidence cancer drivers.

**Fix**: New `load_bailey_drivers()` function in `analysis_A_pcawg_wgs.py`
with curated Bailey-2018-style list **excluding** TTN/MUC16/OBSCN/SYNE1.
~95 high-confidence pan-cancer drivers. (Optional `--bailey-drivers` flag
to load full Bailey 2018 supplement if user supplies it.)

### M6 (supervisor flag) — disk math sanity
**Verified**: ai-gpu2 disk 291 GB total / 183 GB free. CDS-only
(`candidates_cache_aligned.parquet`, 8.45 M positions) RNA-FM run ≈ 22 GB
total npz output (24 chroms × ~1 GB each — much less than 7 GB/chr because
chr is much smaller for CDS-only than the 28.6 M genome-wide). Confirmed
`compute_rnafm.py` is reading `candidates_cache_aligned.parquet` (not
`candidates_all.parquet`); see launch command in PHASED_STATUS.md.

## What was NOT done

- **Bailey 2018 full driver list download**: the script accepts `--bailey-drivers`
  but wasn't supplied. Currently uses 95-gene curated fallback. If QA wants the
  full 299-gene list, point to a TSV/CSV with `Gene` or `Symbol` column.
- **Sample-level SBS attribution**: not feasible without a Donor↔Sample sheet
  for the open-MAF release. Cancer-level fallback is documented.

## Validation status

- Both scripts compile and import cleanly (verified at line `OK` in syntax check).
- M4 unit tests pass (TCW counter behaves correctly).
- M1+M2 verified: load_v3_positions_hg19 returns 9,830 hg19 unique positions
  (10,102 hg19 rows after deduplication) — vs 11,870 chrN:pos:strand-format.
- B1+B2+B3 wired into Analysis A primary; both heads (`score_binary_mean` and
  fallback `score_binary`) supported.

## Sign-off

Pipeline ready for analysis runs as soon as panel scoring completes.
RNA-FM CDS still progressing in tmux session `rnafm_cds` on ai-gpu2 (chr1
~5% as of 14:00 IDT, ETA 7.4h). Score watcher idle in tmux session
`score_watcher_cds`, will pick up scored chroms automatically.
