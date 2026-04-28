#!/usr/bin/env python3
"""Stress-test the advisor-report finding 4b: 'Top-1000 model-predicted pathogenic
C>T have 59.5% nonsense vs 47.4% baseline (OR=1.64, p=1.18e-14)'.

Design note (2026-04-24 rewrite)
-------------------------------
The *primary* comparison is CODING-ONLY pathogenic C>T. The original 4b headline
used the full pathogenic set (including UTRs / introns / non-RefGene tx), so
~54% of rows re-derive as `non_coding`, dragging the baseline nonsense rate from
~75% (coding) down to ~34% (all). Top-K enrichment in that mixed pool is partly
a trivial "model up-ranks coding over non-coding" effect, not a real nonsense
preference. Restricting to coding removes that confound so the OR reflects the
substantive claim.

Suspected confound inside coding: C>T can create a stop codon only via
  CGA -> TGA (CpG site)
  CAA -> TAA (non-CpG)
  CAG -> TAG (non-CpG)
so any CpG bias trivially up-ranks CGA and inflates TGA nonsense as a
confound. CpG stratification isolates whether the signal survives in non-CpG
(where CGA-route cannot help).

Eight analyses (all on existing scored ClinVar file, no re-training):
  1. Primary reproduction — CODING-ONLY pathogenic C>T
     (plus an `all_pathogenic_reference` block for transparency)
  2. CpG vs non-CpG stratification (coding-only by construction)
  3. Codon-route decomposition (CGA/CAA/CAG)
  4. Achievability-matched baseline
  5. Per-gene clustering check
  6. Joint logistic regression controlling for CpG/motif/codon-pos/gene length
     (confound-control regression, NOT a competing predictor)
  7. Same stratified analysis on RNAsee (p_edited_rnasee), coding-only
  8. Head-to-head GB vs RNAsee on the coding subset — why our top picks are
     nonsense-enriched while RNAsee's are nonsense-depleted

Output:
  experiments/apobec3a/outputs/clinvar_nonsense_stress/results.json
  experiments/apobec3a/outputs/clinvar_nonsense_stress/REPORT.md

Idempotent: safe to rerun.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from pyfaidx import Fasta
from scipy import stats
from scipy.stats import fisher_exact

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
ROOT = Path("/Users/shaharharel/Documents/github/edit-rna-apobec")
CLINVAR_SCORES = ROOT / "experiments" / "apobec3a" / "outputs" / "clinvar_prediction" / "clinvar_all_scores.csv"
HG38_FA = ROOT / "data" / "raw" / "genomes" / "hg38.fa"
REFGENE_HG38 = ROOT / "data" / "raw" / "genomes" / "refGene.txt"

OUTDIR = ROOT / "experiments" / "apobec3a" / "outputs" / "clinvar_nonsense_stress"
OUTDIR.mkdir(parents=True, exist_ok=True)
RESULTS_JSON = OUTDIR / "results.json"
REPORT_MD = OUTDIR / "REPORT.md"
CACHE_DIR = OUTDIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

CODON_TABLE = {
    # Standard DNA codon -> amino acid
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

COMPLEMENT = str.maketrans("ACGTacgt", "TGCAtgca")


def revcomp(seq: str) -> str:
    return seq.translate(COMPLEMENT)[::-1]


# -----------------------------------------------------------------------------
# refGene parsing -> CDS transcripts
# -----------------------------------------------------------------------------
def load_refgene_transcripts(path: Path) -> pd.DataFrame:
    cols = [
        "bin", "name", "chrom", "strand", "txStart", "txEnd",
        "cdsStart", "cdsEnd", "exonCount", "exonStarts", "exonEnds",
        "score", "name2", "cdsStartStat", "cdsEndStat", "exonFrames",
    ]
    df = pd.read_csv(path, sep="\t", header=None, names=cols, low_memory=False)
    valid_chroms = {f"chr{i}" for i in range(1, 23)} | {"chrX", "chrY"}
    df = df[df["chrom"].isin(valid_chroms)]
    df = df[df["cdsStart"] != df["cdsEnd"]].copy()  # coding tx only
    # Keep longest tx per gene (same dedup strategy as run_a2_a3_a4.py)
    df["tx_len"] = df["txEnd"] - df["txStart"]
    df = df.sort_values("tx_len", ascending=False).drop_duplicates(subset="name2", keep="first")
    df = df.reset_index(drop=True)
    return df


def build_tx_cds_index(refgene_df: pd.DataFrame):
    """For each transcript, return dict with chrom, strand, gene, list of CDS
    exon segments (genomic start, end, half-open), and cumulative CDS offsets.

    Also returns an interval tree-ish dict: chrom -> list of (cds_start, cds_end, tx_idx)
    for fast lookup.
    """
    tx_info = []
    chrom_intervals = defaultdict(list)

    for idx, row in refgene_df.iterrows():
        chrom = row["chrom"]
        strand = row["strand"]
        gene = row["name2"]
        cds_start = int(row["cdsStart"])
        cds_end = int(row["cdsEnd"])

        starts = [int(x) for x in row["exonStarts"].rstrip(",").split(",")]
        ends = [int(x) for x in row["exonEnds"].rstrip(",").split(",")]

        # Build CDS-exon segments clipped to cdsStart/cdsEnd (half-open)
        segs = []
        for es, ee in zip(starts, ends):
            cs = max(es, cds_start)
            ce = min(ee, cds_end)
            if cs < ce:
                segs.append((cs, ce))
        if not segs:
            continue
        # segs are in genomic (+) order; sort
        segs.sort()

        tx_info.append({
            "tx_idx": len(tx_info),
            "chrom": chrom,
            "strand": strand,
            "gene": gene,
            "cds_start": cds_start,
            "cds_end": cds_end,
            "segs": segs,  # list of (start, end) genomic
        })
        for (s, e) in segs:
            chrom_intervals[chrom].append((s, e, tx_info[-1]["tx_idx"]))

    # Sort intervals per chrom for bisect-style lookup
    sorted_chrom = {}
    for chrom, ivs in chrom_intervals.items():
        ivs.sort()
        starts_arr = np.array([s for s, _, _ in ivs])
        ends_arr = np.array([e for _, e, _ in ivs])
        tx_arr = np.array([t for _, _, t in ivs])
        sorted_chrom[chrom] = (starts_arr, ends_arr, tx_arr)

    return tx_info, sorted_chrom


def find_transcripts_for_pos(chrom, pos, sorted_chrom):
    """Return list of (tx_idx, ...) for genomic pos (0-based) overlapping a CDS exon."""
    if chrom not in sorted_chrom:
        return []
    starts, ends, tx_arr = sorted_chrom[chrom]
    # All intervals with start <= pos < end
    # Use searchsorted: find where pos could be inserted in starts; those with start<=pos are candidates
    idx = np.searchsorted(starts, pos, side="right") - 1
    out = []
    # walk back (rare to overlap many; CDS segs often non-overlapping)
    i = idx
    while i >= 0:
        if ends[i] <= pos:
            # all earlier intervals with this same start also end before pos; but starts sorted so
            # can't break purely on this — need to keep walking back a bit. Walk back ~50 ivs max.
            if starts[i] < pos - 10_000_000:
                break
        else:
            if starts[i] <= pos < ends[i]:
                out.append(int(tx_arr[i]))
        i -= 1
        if i < idx - 1000:  # safety bound
            break
    # Deduplicate tx
    return list(set(out))


def cds_position_on_transcript(tx, pos):
    """Return (cds_pos_0based, strand) if pos falls inside CDS of this tx, else None.

    cds_pos_0based is 0-based index along the coding sequence in the direction
    of translation (5'->3' of mRNA).
    """
    segs = tx["segs"]
    strand = tx["strand"]
    if strand == "+":
        cum = 0
        for s, e in segs:
            if s <= pos < e:
                return cum + (pos - s), strand
            cum += (e - s)
        return None
    else:
        # For -strand, mRNA runs from genomic high->low; iterate segs in reverse
        cum = 0
        for s, e in reversed(segs):
            if s <= pos < e:
                # distance from e-1 down to pos
                return cum + (e - 1 - pos), strand
            cum += (e - s)
        return None


def get_codon_and_position(tx, pos, fasta):
    """For a tx that contains pos in CDS, return:
        (codon_str_on_mRNA, pos_in_codon, ref_base_on_mRNA, strand, cds_pos)
    codon_str_on_mRNA uses DNA alphabet (ACGT) in the direction of translation.
    pos_in_codon is 0, 1, or 2.
    Returns None if position is ambiguous (N) or out of bounds.
    """
    res = cds_position_on_transcript(tx, pos)
    if res is None:
        return None
    cds_pos, strand = res
    pos_in_codon = cds_pos % 3
    codon_start_cds = cds_pos - pos_in_codon  # 0-based start of codon in CDS coords

    # Need the 3 CDS nucleotides that form the codon
    # Map codon_start_cds .. codon_start_cds+2 back to genomic coords
    codon_bases = []
    segs = tx["segs"]
    strand = tx["strand"]
    if strand == "+":
        # accumulate until we find positions
        cds_positions_genomic = []
        cum = 0
        for s, e in segs:
            seg_len = e - s
            # does this seg cover [codon_start_cds..codon_start_cds+3)?
            seg_start_cds = cum
            seg_end_cds = cum + seg_len
            # iterate cds positions we need
            for k in range(3):
                target_cds = codon_start_cds + k
                if seg_start_cds <= target_cds < seg_end_cds:
                    genomic_pos = s + (target_cds - seg_start_cds)
                    while len(cds_positions_genomic) <= k:
                        cds_positions_genomic.append(None)
                    cds_positions_genomic[k] = genomic_pos
            cum = seg_end_cds
        if None in cds_positions_genomic or len(cds_positions_genomic) < 3:
            return None
        # Fetch bases from hg38 (+ strand)
        try:
            for gpos in cds_positions_genomic:
                b = str(fasta[tx["chrom"]][gpos:gpos+1]).upper()
                if b not in "ACGT":
                    return None
                codon_bases.append(b)
        except Exception:
            return None
        codon_str = "".join(codon_bases)
        ref_on_mrna = codon_str[pos_in_codon]
        return codon_str, pos_in_codon, ref_on_mrna, strand, cds_pos
    else:
        # strand == '-'
        # segs are sorted genomic ascending; mRNA runs high->low
        # Build cds positions going in mRNA direction
        cds_positions_genomic = [None, None, None]
        cum = 0
        for s, e in reversed(segs):
            seg_len = e - s
            seg_start_cds = cum
            seg_end_cds = cum + seg_len
            for k in range(3):
                target_cds = codon_start_cds + k
                if seg_start_cds <= target_cds < seg_end_cds:
                    # offset from e-1 going down
                    offset_from_top = target_cds - seg_start_cds
                    genomic_pos = (e - 1) - offset_from_top
                    cds_positions_genomic[k] = genomic_pos
            cum = seg_end_cds
        if None in cds_positions_genomic:
            return None
        try:
            for gpos in cds_positions_genomic:
                b_plus = str(fasta[tx["chrom"]][gpos:gpos+1]).upper()
                if b_plus not in "ACGT":
                    return None
                # mRNA base is complement of + strand base
                b_mrna = revcomp(b_plus)
                codon_bases.append(b_mrna)
        except Exception:
            return None
        codon_str = "".join(codon_bases)
        ref_on_mrna = codon_str[pos_in_codon]
        return codon_str, pos_in_codon, ref_on_mrna, strand, cds_pos


def annotate_consequence(codon_str, pos_in_codon, alt_base_on_mrna):
    """Classify the substitution as nonsense/missense/synonymous."""
    if codon_str not in CODON_TABLE:
        return None
    new_codon = codon_str[:pos_in_codon] + alt_base_on_mrna + codon_str[pos_in_codon+1:]
    if new_codon not in CODON_TABLE:
        return None
    old_aa = CODON_TABLE[codon_str]
    new_aa = CODON_TABLE[new_codon]
    if old_aa == "*":
        return "stop_lost"
    if new_aa == "*":
        return "nonsense"
    if old_aa == new_aa:
        return "synonymous"
    return "missense"


# -----------------------------------------------------------------------------
# CpG detection (on the mRNA / coding strand)
# -----------------------------------------------------------------------------
def is_cpg_on_coding_strand(chrom, pos, strand_on_tx, fasta):
    """Is the edited C part of a CpG dinucleotide on the coding strand
    (i.e., is the next base on the mRNA a G)?

    For + strand tx, the mRNA reads left->right on hg38, so we need base at pos+1 on + strand.
    For - strand tx, the mRNA reads right->left, so we need complement of base at pos-1 on +.
    """
    try:
        if strand_on_tx == "+":
            next_base = str(fasta[chrom][pos+1:pos+2]).upper()
            return next_base == "G"
        else:
            prev_plus = str(fasta[chrom][pos-1:pos]).upper()
            # on mRNA, "next" means going high->low, so prev_plus is the "next" on mRNA (after revcomp)
            next_base_mrna = revcomp(prev_plus) if prev_plus else ""
            return next_base_mrna == "G"
    except Exception:
        return False


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------
def build_annotated_dataframe():
    """Load ClinVar scores, filter to pathogenic C>T, re-annotate consequence
    from refGene CDS + hg38. Returns DataFrame with:
        site_id, chr, start, gene, score_gb, score_rnasee,
        codon, pos_in_codon, ref_mrna, alt_mrna, strand_on_tx,
        consequence, is_cpg, is_tc_motif, tx_length_cds
    """
    cache_path = CACHE_DIR / "annotated_pathogenic_ct.parquet"
    if cache_path.exists():
        print(f"Loading cached annotated DataFrame: {cache_path}")
        return pd.read_parquet(cache_path)

    print("Loading scores CSV...")
    scores = pd.read_csv(CLINVAR_SCORES, low_memory=False)
    print(f"  {len(scores):,} rows")

    # Restrict to pathogenic + likely_pathogenic
    mask_path = scores["significance_simple"].isin(["Pathogenic", "Likely_pathogenic"])
    path_df = scores[mask_path].copy().reset_index(drop=True)
    print(f"Pathogenic + Likely_pathogenic: {len(path_df):,}")

    # We'll annotate every row. Need editing_strand from the raw CSV.
    # The site_id encodes strand: "clinvar_chr1_69240_+" or "_-"
    path_df["editing_strand"] = path_df["site_id"].str.extract(r"_([+-])$")

    print("Loading refGene...")
    ref_df = load_refgene_transcripts(REFGENE_HG38)
    print(f"  {len(ref_df):,} transcripts")

    print("Building tx CDS index...")
    tx_info, sorted_chrom = build_tx_cds_index(ref_df)
    # Precompute tx CDS length in bases
    for tx in tx_info:
        tx["cds_len"] = sum(e - s for s, e in tx["segs"])

    # Gene -> preferred tx (longest CDS; when multiple tx map to same pos, we
    # prefer the one whose "gene" matches the ClinVar 'gene' column)
    gene_to_txidx = {tx["gene"]: i for i, tx in enumerate(tx_info)}

    print("Opening hg38 FASTA...")
    fasta = Fasta(str(HG38_FA), as_raw=False, sequence_always_upper=True)

    # Iterate rows
    n = len(path_df)
    out = []
    print(f"Annotating {n:,} variants...")
    for i, row in enumerate(path_df.itertuples(index=False)):
        if i % 5000 == 0:
            print(f"  {i:,}/{n:,}")

        chrom = row.chr
        pos = int(row.start)  # site_id "start" is 0-based (by convention of the pipeline)
        gene = row.gene if isinstance(row.gene, str) else None
        editing_strand = row.editing_strand

        # Pick a transcript whose strand matches editing_strand (so mRNA ref is C).
        # Prefer one whose gene matches; fall back to first overlapping strand-matched tx.
        tx = None
        if gene and gene in gene_to_txidx:
            candidate = tx_info[gene_to_txidx[gene]]
            if (candidate["chrom"] == chrom
                and candidate["strand"] == editing_strand
                and cds_position_on_transcript(candidate, pos) is not None):
                tx = candidate
        if tx is None:
            tx_candidates = find_transcripts_for_pos(chrom, pos, sorted_chrom)
            if tx_candidates:
                # Require strand-matched transcript (if editing_strand=-, need -strand tx)
                for ti in tx_candidates:
                    if tx_info[ti]["strand"] == editing_strand:
                        tx = tx_info[ti]
                        break

        if tx is None:
            out.append({
                "site_id": row.site_id, "chr": chrom, "start": pos, "gene": gene,
                "p_edited_gb": row.p_edited_gb, "p_edited_rnasee": row.p_edited_rnasee,
                "significance_simple": row.significance_simple,
                "editing_strand": editing_strand,
                "tx_strand": None, "tx_gene": None, "cds_len": np.nan,
                "codon": None, "pos_in_codon": None, "ref_mrna": None, "alt_mrna": None,
                "consequence": "non_coding", "is_cpg": None, "is_tc_motif": None,
                "codon_route": None,
            })
            continue

        # CDS annotation
        res = get_codon_and_position(tx, pos, fasta)
        if res is None:
            out.append({
                "site_id": row.site_id, "chr": chrom, "start": pos, "gene": gene,
                "p_edited_gb": row.p_edited_gb, "p_edited_rnasee": row.p_edited_rnasee,
                "significance_simple": row.significance_simple,
                "editing_strand": editing_strand,
                "tx_strand": tx["strand"], "tx_gene": tx["gene"], "cds_len": tx["cds_len"],
                "codon": None, "pos_in_codon": None, "ref_mrna": None, "alt_mrna": None,
                "consequence": "non_coding", "is_cpg": None, "is_tc_motif": None,
                "codon_route": None,
            })
            continue

        codon_str, pos_in_codon, ref_on_mrna, strand, cds_pos = res

        # On the mRNA, the edit is always C->U. So ref_on_mrna should be C for a
        # valid c-to-t on this tx's coding strand. If ref_on_mrna != 'C',
        # this tx's strand disagrees with where the C sits → skip (non-CDS for this direction).
        if ref_on_mrna != "C":
            # The genomic variant is still C>T but the coding strand is antisense → non-coding on this tx
            out.append({
                "site_id": row.site_id, "chr": chrom, "start": pos, "gene": gene,
                "p_edited_gb": row.p_edited_gb, "p_edited_rnasee": row.p_edited_rnasee,
                "significance_simple": row.significance_simple,
                "editing_strand": editing_strand,
                "tx_strand": strand, "tx_gene": tx["gene"], "cds_len": tx["cds_len"],
                "codon": codon_str, "pos_in_codon": int(pos_in_codon),
                "ref_mrna": ref_on_mrna, "alt_mrna": None,
                "consequence": "non_C_on_coding_strand", "is_cpg": None, "is_tc_motif": None,
                "codon_route": None,
            })
            continue

        alt_on_mrna = "T"  # DNA-alphabet; represents U
        consequence = annotate_consequence(codon_str, pos_in_codon, alt_on_mrna) or "unknown"

        # Codon route (only relevant for nonsense)
        codon_route = None
        if consequence == "nonsense":
            # Determine which route
            if codon_str == "CGA" and pos_in_codon == 1:  # CGA -> CTA (not stop); wait: C at pos 0, G pos1, A pos2
                # Actually CGA -> TGA requires change at pos 0 (C->T). Let's check.
                pass
            # General: any nonsense where ref is C->T with the C at pos_in_codon creating a stop codon
            # Routes:
            #   TGA from CGA (C at position 0)
            #   TAA from CAA (C at position 0)
            #   TAG from CAG (C at position 0)
            if codon_str == "CGA" and pos_in_codon == 0:
                codon_route = "CGA_TGA"
            elif codon_str == "CAA" and pos_in_codon == 0:
                codon_route = "CAA_TAA"
            elif codon_str == "CAG" and pos_in_codon == 0:
                codon_route = "CAG_TAG"
            else:
                codon_route = f"other_{codon_str}_pos{pos_in_codon}"

        # CpG on coding strand
        is_cpg = is_cpg_on_coding_strand(chrom, pos, strand, fasta)

        # TC motif: is the base immediately 5' on the mRNA a T (U)?
        # On + strand: base at pos-1 on hg38
        # On - strand: revcomp of base at pos+1 on hg38
        try:
            if strand == "+":
                b5 = str(fasta[chrom][pos-1:pos]).upper()
            else:
                b5_plus = str(fasta[chrom][pos+1:pos+2]).upper()
                b5 = revcomp(b5_plus) if b5_plus else ""
        except Exception:
            b5 = ""
        is_tc_motif = (b5 == "T")

        out.append({
            "site_id": row.site_id, "chr": chrom, "start": pos, "gene": gene,
            "p_edited_gb": row.p_edited_gb, "p_edited_rnasee": row.p_edited_rnasee,
            "significance_simple": row.significance_simple,
            "editing_strand": editing_strand,
            "tx_strand": strand, "tx_gene": tx["gene"], "cds_len": tx["cds_len"],
            "codon": codon_str, "pos_in_codon": int(pos_in_codon),
            "ref_mrna": ref_on_mrna, "alt_mrna": alt_on_mrna,
            "consequence": consequence, "is_cpg": bool(is_cpg), "is_tc_motif": bool(is_tc_motif),
            "codon_route": codon_route,
        })

    df = pd.DataFrame(out)
    df.to_parquet(cache_path, index=False)
    print(f"Cached: {cache_path}")
    return df


# -----------------------------------------------------------------------------
# Analyses
# -----------------------------------------------------------------------------

def fisher_topk(df, k, score_col, pos_class_col="is_nonsense"):
    """Top-K vs rest OR."""
    df_sorted = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    top = df_sorted.head(k)
    rest = df_sorted.iloc[k:]
    a = int(top[pos_class_col].sum())  # top & nonsense
    b = int(k - a)                     # top & not-nonsense
    c = int(rest[pos_class_col].sum())
    d = int(len(rest) - c)
    if b == 0 or c == 0:
        odds = float("inf") if a > 0 else 0.0
        p = fisher_exact([[a, b], [c, d]])[1]
    else:
        odds, p = fisher_exact([[a, b], [c, d]], alternative="two-sided")
    return {
        "k": int(k),
        "top_nonsense": a, "top_other": b,
        "rest_nonsense": c, "rest_other": d,
        "top_nonsense_frac": a / k if k > 0 else 0.0,
        "rest_nonsense_frac": c / (c + d) if (c + d) > 0 else 0.0,
        "odds_ratio": float(odds),
        "p_value": float(p),
        "min_score_in_top": float(top[score_col].min()) if len(top) else None,
    }


CODING_CONSEQS = {"nonsense", "missense", "synonymous"}


def analysis_1_reproduce(df_full):
    """Primary reproduction — CODING-ONLY.

    The original 4b headline mixed coding + non-coding pathogenic C>T; the OR
    then partly reflected "model up-ranks coding over non-coding" rather than a
    nonsense preference. We restrict to coding consequences (nonsense /
    missense / synonymous) so the baseline and top-K denominators are
    comparable.

    We also report the all-pathogenic view for transparency, flagged as a
    reference-only sanity check — NOT the primary result.
    """
    df_full = df_full.copy()
    df_full["is_nonsense"] = (df_full["consequence"] == "nonsense").astype(int)

    ks = [100, 500, 1000, 2000, 5000]

    # ---- Primary: coding-only ----
    df_cod = df_full[df_full["consequence"].isin(CODING_CONSEQS)].copy()
    baseline_frac_cod = df_cod["is_nonsense"].mean()
    baseline_counts_cod = df_cod["consequence"].value_counts().to_dict()
    top_k_cod = {str(k): fisher_topk(df_cod, k, "p_edited_gb") for k in ks}

    # ---- Reference: all-pathogenic (for transparency only) ----
    baseline_frac_all = df_full["is_nonsense"].mean()
    baseline_counts_all = df_full["consequence"].value_counts().to_dict()
    top_k_all = {str(k): fisher_topk(df_full, k, "p_edited_gb") for k in ks}

    return {
        "primary_coding_only": {
            "n_coding": int(len(df_cod)),
            "baseline_consequence_distribution": {k: int(v) for k, v in baseline_counts_cod.items()},
            "baseline_nonsense_fraction": float(baseline_frac_cod),
            "top_k_results": top_k_cod,
            "note": (
                "PRIMARY analysis. Coding-only pathogenic C>T "
                "(consequence in {nonsense, missense, synonymous}). "
                "Baseline and top-K denominators are like-for-like."
            ),
        },
        "all_pathogenic_reference": {
            "n_pathogenic": int(len(df_full)),
            "baseline_consequence_distribution": {k: int(v) for k, v in baseline_counts_all.items()},
            "baseline_nonsense_fraction": float(baseline_frac_all),
            "top_k_results": top_k_all,
            "note": (
                "REFERENCE ONLY (includes non_coding). The original 4b headline used "
                "this mixed pool; ~54% of rows are non-coding, so top-K OR inflates "
                "by ranking coding over non-coding. Do NOT report these numbers as "
                "the nonsense-enrichment claim."
            ),
        },
    }


def analysis_2_cpg_stratification(df_cds):
    """CpG vs non-CpG stratification."""
    df = df_cds.copy()
    df = df[df["is_cpg"].notna()].copy()  # drop rows we couldn't CpG-annotate
    df["is_cpg"] = df["is_cpg"].astype(bool)
    df["is_nonsense"] = (df["consequence"] == "nonsense").astype(int)

    result = {}
    for label, subdf in [("CpG", df[df["is_cpg"]]),
                         ("nonCpG", df[~df["is_cpg"].values])]:
        if len(subdf) == 0:
            continue
        baseline_frac = subdf["is_nonsense"].mean()
        result[label] = {
            "n": int(len(subdf)),
            "baseline_nonsense_frac": float(baseline_frac),
            "top_k": {str(k): fisher_topk(subdf, k, "p_edited_gb") for k in [100, 500, 1000, 2000]},
        }
    result["summary"] = {
        "n_cpg": int(df["is_cpg"].sum()),
        "n_noncpg": int((~df["is_cpg"]).sum()),
        "cpg_nonsense_frac": float(df[df["is_cpg"]]["is_nonsense"].mean()),
        "noncpg_nonsense_frac": float(df[~df["is_cpg"]]["is_nonsense"].mean()),
    }
    return result


def analysis_3_codon_routes(df_cds):
    """Decompose nonsense variants by codon route in top-K vs baseline."""
    df = df_cds.copy()
    df["is_nonsense"] = (df["consequence"] == "nonsense").astype(int)
    nonsense_all = df[df["is_nonsense"] == 1]
    baseline_routes = nonsense_all["codon_route"].value_counts(normalize=True).to_dict()
    baseline_counts = nonsense_all["codon_route"].value_counts().to_dict()

    out = {
        "baseline_nonsense_n": int(len(nonsense_all)),
        "baseline_route_counts": {k: int(v) for k, v in baseline_counts.items()},
        "baseline_route_fractions": {k: float(v) for k, v in baseline_routes.items()},
        "top_k": {},
    }

    for k in [100, 500, 1000, 2000]:
        top = df.sort_values("p_edited_gb", ascending=False).head(k)
        top_nons = top[top["is_nonsense"] == 1]
        out["top_k"][str(k)] = {
            "n_top": k,
            "n_nonsense_in_top": int(len(top_nons)),
            "route_counts": {kk: int(vv) for kk, vv in top_nons["codon_route"].value_counts().items()},
            "route_fractions_of_nonsense": {kk: float(vv) for kk, vv in
                                            top_nons["codon_route"].value_counts(normalize=True).items()},
        }
    return out


def analysis_4_achievability_matched(df_cds):
    """Restrict baseline to C>T at codon positions where nonsense IS achievable:
       codon must be CGA, CAA, or CAG with C at pos 0.
    """
    df = df_cds.copy()
    df["is_nonsense"] = (df["consequence"] == "nonsense").astype(int)

    achievable_mask = (
        (df["pos_in_codon"] == 0)
        & (df["codon"].isin(["CGA", "CAA", "CAG"]))
    )
    # Among these, every variant is nonsense-capable; the "nonsense rate" in top-K
    # then measures whether model up-ranks these codons vs other C>T in *other*
    # codons that could never be nonsense.
    #
    # Interpretation: the achievability-matched baseline restricts to variants that
    # COULD be nonsense. Within this set, everything IS nonsense by construction,
    # so that stratum's nonsense fraction = 1.0, uninformative.
    #
    # Instead: compare C>T variants by achievability. Define:
    #   achievable: codon IS CGA/CAA/CAG at pos 0  (becomes stop)
    #   non-achievable: other coding C>T  (cannot become stop)
    # Model-score distribution comparison:

    df["achievable"] = achievable_mask
    n_achievable = int(df["achievable"].sum())
    n_not = int((~df["achievable"]).sum())

    result = {
        "n_achievable": n_achievable,
        "n_non_achievable": n_not,
        "frac_achievable": n_achievable / len(df) if len(df) > 0 else 0.0,
        "top_k_achievability": {},
    }

    # Top-K: what fraction are achievable? (Compare to baseline frac_achievable)
    for k in [100, 500, 1000, 2000]:
        top = df.sort_values("p_edited_gb", ascending=False).head(k)
        n_ach_in_top = int(top["achievable"].sum())
        n_not_in_top = k - n_ach_in_top
        rest = df.sort_values("p_edited_gb", ascending=False).iloc[k:]
        n_ach_rest = int(rest["achievable"].sum())
        n_not_rest = len(rest) - n_ach_rest
        odds, p = fisher_exact(
            [[n_ach_in_top, n_not_in_top], [n_ach_rest, n_not_rest]],
            alternative="two-sided",
        )
        result["top_k_achievability"][str(k)] = {
            "n_top": k,
            "n_achievable_in_top": n_ach_in_top,
            "frac_achievable_in_top": n_ach_in_top / k if k else 0.0,
            "odds_ratio_vs_rest": float(odds),
            "p_value": float(p),
        }

    # Also compute "nonsense rate in top-K relative to achievability-restricted baseline"
    # by doing the Fisher test ONLY within achievable-capable codons (CGA/CAA/CAG
    # vs CGX/CAX/etc other C-starting codons). This asks: given that a variant
    # COULD be nonsense, does model score track nonsense status? For achievable-only
    # subset, top-K nonsense rate = 1.0 trivially; so instead compare the rate of
    # CGA (i.e., CpG) up-ranking vs CAA/CAG among achievable:
    ach = df[df["achievable"]].copy()
    if len(ach) >= 100:
        sub = {}
        for k in [100, 500, 1000]:
            if k > len(ach):
                continue
            top = ach.sort_values("p_edited_gb", ascending=False).head(k)
            cga = int((top["codon"] == "CGA").sum())
            cax = int((top["codon"].isin(["CAA", "CAG"])).sum())
            base_cga = int((ach["codon"] == "CGA").sum())
            base_cax = int(ach["codon"].isin(["CAA", "CAG"]).sum())
            base_frac_cga = base_cga / (base_cga + base_cax) if (base_cga + base_cax) else 0.0
            top_frac_cga = cga / (cga + cax) if (cga + cax) else 0.0
            sub[str(k)] = {
                "top_cga": cga, "top_cax": cax, "top_frac_cga": float(top_frac_cga),
                "baseline_frac_cga": float(base_frac_cga),
            }
        result["within_achievable_cga_vs_cax"] = sub

    return result


def analysis_5_per_gene(df_cds):
    """How many genes in top-K; is enrichment gene-broad or driven by few?"""
    df = df_cds.copy()
    df["is_nonsense"] = (df["consequence"] == "nonsense").astype(int)
    top1000 = df.sort_values("p_edited_gb", ascending=False).head(1000)
    gene_top_counts = top1000["gene"].value_counts()

    # Among genes with >=5 top-1000 hits: compare per-gene nonsense frac in top vs same-gene baseline
    big_genes = gene_top_counts[gene_top_counts >= 5].index.tolist()
    sign_test_data = []
    for g in big_genes:
        top_g = top1000[top1000["gene"] == g]
        all_g = df[df["gene"] == g]
        if len(all_g) < 10:
            continue
        top_frac = top_g["is_nonsense"].mean()
        base_frac = all_g["is_nonsense"].mean()
        sign_test_data.append({
            "gene": g,
            "n_top": int(len(top_g)),
            "n_total": int(len(all_g)),
            "top_nonsense_frac": float(top_frac),
            "baseline_nonsense_frac": float(base_frac),
            "diff": float(top_frac - base_frac),
        })

    # Sign test: how many genes have top_frac > base_frac?
    diffs = [g["diff"] for g in sign_test_data if g["diff"] != 0]
    n_pos = sum(1 for d in diffs if d > 0)
    n_neg = sum(1 for d in diffs if d < 0)
    if (n_pos + n_neg) > 0:
        from scipy.stats import binomtest
        sign_p = binomtest(n_pos, n_pos + n_neg, 0.5, alternative="greater").pvalue
    else:
        sign_p = None

    return {
        "n_top1000": int(len(top1000)),
        "n_distinct_genes_in_top1000": int(top1000["gene"].nunique()),
        "top_gene_counts": gene_top_counts.head(30).to_dict(),
        "top_gene_cumulative_frac": {
            "top5_genes": float(gene_top_counts.head(5).sum() / len(top1000)),
            "top10_genes": float(gene_top_counts.head(10).sum() / len(top1000)),
            "top20_genes": float(gene_top_counts.head(20).sum() / len(top1000)),
        },
        "per_gene_analysis": sign_test_data,
        "sign_test": {
            "n_genes_evaluated": len(sign_test_data),
            "n_genes_up": n_pos,
            "n_genes_down": n_neg,
            "p_value_one_sided_greater": float(sign_p) if sign_p is not None else None,
        },
    }


def analysis_6_logreg(df_cds):
    """Joint logistic regression: is_nonsense ~ score + is_CpG + is_TC + codon_pos + log_gene_length"""
    import statsmodels.api as sm

    df = df_cds.copy()
    df = df[df["consequence"].isin([
        "nonsense", "missense", "synonymous"
    ])].copy()  # only meaningful coding consequences
    df["is_nonsense"] = (df["consequence"] == "nonsense").astype(int)

    # Fill any NaNs
    df = df.dropna(subset=["p_edited_gb", "is_cpg", "is_tc_motif", "pos_in_codon", "cds_len"])
    df["log_cds_len"] = np.log1p(df["cds_len"].astype(float))
    df["is_cpg_num"] = df["is_cpg"].astype(int)
    df["is_tc_num"] = df["is_tc_motif"].astype(int)
    # one-hot codon position (pos_in_codon in {0,1,2}); use pos0/pos2 as dummies
    df["codon_pos0"] = (df["pos_in_codon"] == 0).astype(int)
    df["codon_pos2"] = (df["pos_in_codon"] == 2).astype(int)

    # Model 1: score-only
    X1 = sm.add_constant(df[["p_edited_gb"]])
    m1 = sm.Logit(df["is_nonsense"], X1).fit(disp=0)

    # Model 2: full controls
    cols = ["p_edited_gb", "is_cpg_num", "is_tc_num", "codon_pos0", "codon_pos2", "log_cds_len"]
    X2 = sm.add_constant(df[cols])
    m2 = sm.Logit(df["is_nonsense"], X2).fit(disp=0, maxiter=100)

    def summarize(model, cols_used):
        out = {"pseudo_r2": float(model.prsquared), "llf": float(model.llf), "n": int(model.nobs)}
        out["params"] = {
            c: {
                "coef": float(model.params[c]),
                "std_err": float(model.bse[c]),
                "p_value": float(model.pvalues[c]),
                "odds_ratio": float(np.exp(model.params[c])),
            } for c in cols_used
        }
        return out

    return {
        "n_used": int(len(df)),
        "score_only": summarize(m1, ["const", "p_edited_gb"]),
        "full_controls": summarize(m2, ["const"] + cols),
    }


def analysis_7_rnasee(df_full):
    """RNAsee stratified top-K, restricted to CODING pathogenic C>T.

    Matches the denominator of analysis_1's primary block so GB vs RNAsee are
    directly comparable.
    """
    df = df_full.copy()
    df = df[df["consequence"].isin(CODING_CONSEQS)].copy()
    df["is_nonsense"] = (df["consequence"] == "nonsense").astype(int)
    df = df.dropna(subset=["p_edited_rnasee"])

    overall = {str(k): fisher_topk(df, k, "p_edited_rnasee") for k in [100, 500, 1000, 2000]}

    df_cpg_valid = df[df["is_cpg"].notna()].copy()
    df_cpg_valid["is_cpg"] = df_cpg_valid["is_cpg"].astype(bool)
    strat = {}
    for label, sub in [("CpG", df_cpg_valid[df_cpg_valid["is_cpg"]]),
                       ("nonCpG", df_cpg_valid[~df_cpg_valid["is_cpg"].values])]:
        if len(sub) == 0:
            continue
        strat[label] = {
            "n": int(len(sub)),
            "baseline_nonsense_frac": float(sub["is_nonsense"].mean()),
            "top_k": {str(k): fisher_topk(sub, k, "p_edited_rnasee") for k in [100, 500, 1000] if k < len(sub)},
        }

    return {"overall_top_k": overall, "stratified": strat,
            "note": "Restricted to coding pathogenic C>T for comparability with analysis_1 primary."}


def analysis_8_head_to_head(df_full):
    """Direct GB vs RNAsee comparison on the coding pathogenic subset.

    Explains the "RNAsee top picks are depleted for nonsense, ours are enriched"
    finding by looking at:
      - Spearman(GB, RNAsee) rank correlation
      - The top-K sets' disagreement (overlap fraction)
      - Nonsense fraction and codon-route composition in each top-1000
      - What the RNAsee top-1000 picks instead (synonymous/missense, codon
        positions 1/2, non-CpG sites under evolutionary tolerance)
    """
    df = df_full.copy()
    df = df[df["consequence"].isin(CODING_CONSEQS)].copy()
    df = df.dropna(subset=["p_edited_gb", "p_edited_rnasee"])
    df["is_nonsense"] = (df["consequence"] == "nonsense").astype(int)
    n = len(df)

    # Rank correlation
    try:
        rho, rho_p = stats.spearmanr(df["p_edited_gb"], df["p_edited_rnasee"])
    except Exception:
        rho, rho_p = float("nan"), 1.0

    out = {
        "n_coding": int(n),
        "spearman_gb_vs_rnasee": float(rho),
        "spearman_p": float(rho_p),
        "top_1000_overlap": {},
        "composition": {},
    }

    # Top-1000 overlap between GB and RNAsee
    for k in (100, 500, 1000):
        gb_top = set(
            df.sort_values("p_edited_gb", ascending=False).head(k)["site_id"].tolist()
        )
        rnasee_top = set(
            df.sort_values("p_edited_rnasee", ascending=False).head(k)["site_id"].tolist()
        )
        inter = len(gb_top & rnasee_top)
        out["top_1000_overlap"][str(k)] = {
            "gb_top_k": k,
            "rnasee_top_k": k,
            "intersection": int(inter),
            "jaccard": float(inter / len(gb_top | rnasee_top)) if (gb_top | rnasee_top) else 0.0,
        }

    # Composition of top-1000 by consequence and codon position
    for label, col in (("gb", "p_edited_gb"), ("rnasee", "p_edited_rnasee")):
        top = df.sort_values(col, ascending=False).head(1000)
        out["composition"][label] = {
            "top_1000": 1000,
            "consequence_counts": {
                c: int((top["consequence"] == c).sum()) for c in CODING_CONSEQS
            },
            "nonsense_frac": float((top["consequence"] == "nonsense").mean()),
            "missense_frac": float((top["consequence"] == "missense").mean()),
            "synonymous_frac": float((top["consequence"] == "synonymous").mean()),
            "codon_pos_counts": {
                str(int(p)): int((top["pos_in_codon"] == p).sum())
                for p in sorted(top["pos_in_codon"].dropna().unique())
            },
            "cpg_frac": float(top["is_cpg"].astype(float).mean()),
            "tc_frac": float(top["is_tc_motif"].astype(float).mean()),
        }

    # Baseline composition for reference
    out["composition"]["coding_baseline"] = {
        "n": int(n),
        "consequence_counts": {
            c: int((df["consequence"] == c).sum()) for c in CODING_CONSEQS
        },
        "nonsense_frac": float((df["consequence"] == "nonsense").mean()),
        "missense_frac": float((df["consequence"] == "missense").mean()),
        "synonymous_frac": float((df["consequence"] == "synonymous").mean()),
        "cpg_frac": float(df["is_cpg"].astype(float).mean()),
        "tc_frac": float(df["is_tc_motif"].astype(float).mean()),
    }

    return out


# -----------------------------------------------------------------------------
# Reporter
# -----------------------------------------------------------------------------
def write_report(results, out_path):
    lines = []
    A = lines.append
    A("# ClinVar Nonsense Enrichment Stress Test")
    A("")
    A("Stress-testing advisor finding 4b: 'Top-1000 model-predicted pathogenic C>T "
      "have 59.5% nonsense vs 47.4% baseline (OR=1.64, p=1.18e-14).'")
    A("")
    A("Suspected confound: CpG sites are ClinVar pathogenic hotspots, and the "
      "only C>T stop route that is a CpG is CGA->TGA. A CpG-biased model "
      "would up-rank CGA sites and enrich TGA as a confound.")
    A("")
    A("---")
    A("")

    # ---- Analysis 1 ----
    a1 = results["analysis_1_reproduce"]
    a1p = a1["primary_coding_only"]
    a1r = a1["all_pathogenic_reference"]
    A("## 1. Reproduction from raw data (re-derived consequences)")
    A("")
    A("### 1a. PRIMARY — coding-only pathogenic C>T")
    A("")
    A(f"Restricted to coding consequences (nonsense / missense / synonymous).")
    A(f"This is the like-for-like comparison — the all-pathogenic pool in 1b is ")
    A(f"dominated by non-coding variants that can never be nonsense, so mixing them ")
    A(f"in inflates top-K enrichment by a trivial 'coding > non-coding' ranking effect.")
    A("")
    A(f"- Coding pathogenic C>T: **{a1p['n_coding']:,}**")
    A(f"- Baseline nonsense fraction (coding): **{a1p['baseline_nonsense_fraction']*100:.2f}%**")
    A(f"- Baseline consequence counts:")
    for k, v in sorted(a1p["baseline_consequence_distribution"].items(), key=lambda x: -x[1]):
        A(f"    - {k}: {v:,}")
    A("")
    A("| Top-K | n_nonsense | nonsense_frac | OR vs rest | p-value |")
    A("|-------|-----------:|--------------:|-----------:|--------:|")
    for k, r in a1p["top_k_results"].items():
        A(f"| {k} | {r['top_nonsense']} | {r['top_nonsense_frac']*100:.2f}% | {r['odds_ratio']:.3f} | {r['p_value']:.2e} |")
    A("")
    A(f"_Original paper reports: top-1000 nonsense 59.5%, baseline 47.38%, OR=1.64, p=1.18e-14._")
    A("")
    A("### 1b. REFERENCE ONLY — all pathogenic C>T (incl. non-coding)")
    A("")
    A("Do NOT report these numbers as the nonsense-enrichment claim.")
    A("")
    A(f"- All pathogenic C>T: **{a1r['n_pathogenic']:,}**")
    A(f"- Baseline nonsense fraction: {a1r['baseline_nonsense_fraction']*100:.2f}%")
    A(f"- Non-coding share: "
      f"{a1r['baseline_consequence_distribution'].get('non_coding', 0):,}"
      f" / {a1r['n_pathogenic']:,}")
    A("")
    A("| Top-K | nonsense_frac | OR vs rest |")
    A("|-------|--------------:|-----------:|")
    for k, r in a1r["top_k_results"].items():
        A(f"| {k} | {r['top_nonsense_frac']*100:.2f}% | {r['odds_ratio']:.3f} |")
    A("")

    # ---- Analysis 2 ----
    a2 = results["analysis_2_cpg_stratification"]
    A("## 2. CpG stratification")
    A("")
    s = a2["summary"]
    A(f"- CpG variants: {s['n_cpg']:,} (nonsense frac = {s['cpg_nonsense_frac']*100:.2f}%)")
    A(f"- non-CpG variants: {s['n_noncpg']:,} (nonsense frac = {s['noncpg_nonsense_frac']*100:.2f}%)")
    A("")
    for label in ["CpG", "nonCpG"]:
        if label not in a2:
            continue
        sub = a2[label]
        A(f"### {label} stratum (n={sub['n']:,}, baseline nonsense={sub['baseline_nonsense_frac']*100:.2f}%)")
        A("")
        A("| Top-K | n_nonsense | nonsense_frac | OR vs rest | p-value |")
        A("|-------|-----------:|--------------:|-----------:|--------:|")
        for k, r in sub["top_k"].items():
            A(f"| {k} | {r['top_nonsense']} | {r['top_nonsense_frac']*100:.2f}% | {r['odds_ratio']:.3f} | {r['p_value']:.2e} |")
        A("")

    # ---- Analysis 3 ----
    a3 = results["analysis_3_codon_routes"]
    A("## 3. Codon-route decomposition of nonsense variants")
    A("")
    A(f"Baseline nonsense n={a3['baseline_nonsense_n']:,}. Fraction by route:")
    A("")
    A("| Route | Baseline count | Baseline frac |")
    A("|-------|---------------:|--------------:|")
    for k in ["CGA_TGA", "CAA_TAA", "CAG_TAG"]:
        c = a3["baseline_route_counts"].get(k, 0)
        f = a3["baseline_route_fractions"].get(k, 0.0)
        A(f"| {k} | {c:,} | {f*100:.2f}% |")
    other_c = sum(v for k, v in a3["baseline_route_counts"].items() if k not in {"CGA_TGA", "CAA_TAA", "CAG_TAG"})
    A(f"| other | {other_c:,} | - |")
    A("")

    A("### Route breakdown within top-K nonsense calls")
    A("")
    A("| Top-K | n_nonsense | CGA→TGA | CAA→TAA | CAG→TAG |")
    A("|-------|-----------:|--------:|--------:|--------:|")
    for k, sub in a3["top_k"].items():
        cga = sub["route_fractions_of_nonsense"].get("CGA_TGA", 0.0) * 100
        caa = sub["route_fractions_of_nonsense"].get("CAA_TAA", 0.0) * 100
        cag = sub["route_fractions_of_nonsense"].get("CAG_TAG", 0.0) * 100
        A(f"| {k} | {sub['n_nonsense_in_top']} | {cga:.1f}% | {caa:.1f}% | {cag:.1f}% |")
    A("")

    # ---- Analysis 4 ----
    a4 = results["analysis_4_achievability_matched"]
    A("## 4. Achievability-matched baseline")
    A("")
    A(f"- Achievable-nonsense C>T (codon in CGA/CAA/CAG at pos 0): **{a4['n_achievable']:,}** "
      f"({a4['frac_achievable']*100:.2f}% of pathogenic)")
    A(f"- Non-achievable: {a4['n_non_achievable']:,}")
    A("")
    A("Does the model up-rank achievable-nonsense codons vs non-achievable?")
    A("")
    A("| Top-K | n_achievable_in_top | frac_achievable_in_top | OR vs rest | p-value |")
    A("|-------|--------------------:|-----------------------:|-----------:|--------:|")
    for k, r in a4["top_k_achievability"].items():
        A(f"| {k} | {r['n_achievable_in_top']} | {r['frac_achievable_in_top']*100:.2f}% | {r['odds_ratio']:.3f} | {r['p_value']:.2e} |")
    A("")
    if "within_achievable_cga_vs_cax" in a4:
        A("### Within achievable stratum: CGA (CpG) vs CAA+CAG (non-CpG)")
        A("")
        A(f"Baseline CGA/(CGA+CAX) frac: (see first row).")
        A("")
        A("| Top-K | CGA | CAA+CAG | top_frac_CGA | baseline_frac_CGA |")
        A("|-------|----:|--------:|-------------:|------------------:|")
        for k, r in a4["within_achievable_cga_vs_cax"].items():
            A(f"| {k} | {r['top_cga']} | {r['top_cax']} | {r['top_frac_cga']*100:.2f}% | {r['baseline_frac_cga']*100:.2f}% |")
        A("")

    # ---- Analysis 5 ----
    a5 = results["analysis_5_per_gene"]
    A("## 5. Per-gene clustering")
    A("")
    A(f"- Distinct genes in top-1000: **{a5['n_distinct_genes_in_top1000']}**")
    A(f"- Cumulative fraction of top-1000 covered by top-N genes:")
    A(f"    - top 5 genes: {a5['top_gene_cumulative_frac']['top5_genes']*100:.1f}%")
    A(f"    - top 10 genes: {a5['top_gene_cumulative_frac']['top10_genes']*100:.1f}%")
    A(f"    - top 20 genes: {a5['top_gene_cumulative_frac']['top20_genes']*100:.1f}%")
    A("")
    A("Top 15 genes in top-1000:")
    A("")
    A("| Gene | Hits |")
    A("|------|-----:|")
    for g, n in list(a5["top_gene_counts"].items())[:15]:
        A(f"| {g} | {n} |")
    A("")
    st = a5["sign_test"]
    A(f"Sign test across genes with >=5 top-1000 hits: "
      f"{st['n_genes_up']} up, {st['n_genes_down']} down (n={st['n_genes_evaluated']}, "
      f"p_one_sided={st['p_value_one_sided_greater']}).")
    A("")

    # ---- Analysis 6 ----
    a6 = results["analysis_6_logreg"]
    A("## 6. Joint logistic regression")
    A(f"Response: is_nonsense (coding C>T only, n={a6['n_used']:,}).")
    A("")
    A("### Score-only model")
    A("")
    A("| Term | coef | OR | std_err | p-value |")
    A("|------|-----:|---:|--------:|--------:|")
    for c, p in a6["score_only"]["params"].items():
        A(f"| {c} | {p['coef']:.4f} | {p['odds_ratio']:.3f} | {p['std_err']:.4f} | {p['p_value']:.2e} |")
    A(f"pseudo_R² = {a6['score_only']['pseudo_r2']:.4f}")
    A("")
    A("### Full-controls model")
    A("")
    A("| Term | coef | OR | std_err | p-value |")
    A("|------|-----:|---:|--------:|--------:|")
    for c, p in a6["full_controls"]["params"].items():
        A(f"| {c} | {p['coef']:.4f} | {p['odds_ratio']:.3f} | {p['std_err']:.4f} | {p['p_value']:.2e} |")
    A(f"pseudo_R² = {a6['full_controls']['pseudo_r2']:.4f}")
    A("")

    # ---- Analysis 7 ----
    a7 = results["analysis_7_rnasee"]
    A("## 7. Same analysis on RNAsee scores")
    A("")
    A("### Overall top-K")
    A("")
    A("| Top-K | n_nonsense | nonsense_frac | OR | p-value |")
    A("|-------|-----------:|--------------:|---:|--------:|")
    for k, r in a7["overall_top_k"].items():
        A(f"| {k} | {r['top_nonsense']} | {r['top_nonsense_frac']*100:.2f}% | {r['odds_ratio']:.3f} | {r['p_value']:.2e} |")
    A("")
    for label, sub in a7["stratified"].items():
        A(f"### {label} stratum (RNAsee, n={sub['n']:,}, base nonsense={sub['baseline_nonsense_frac']*100:.2f}%)")
        A("")
        A("| Top-K | nonsense_frac | OR | p-value |")
        A("|-------|--------------:|---:|--------:|")
        for k, r in sub["top_k"].items():
            A(f"| {k} | {r['top_nonsense_frac']*100:.2f}% | {r['odds_ratio']:.3f} | {r['p_value']:.2e} |")
        A("")

    # ---- Analysis 8 ----
    if "analysis_8_head_to_head" in results:
        a8 = results["analysis_8_head_to_head"]
        A("## 8. GB vs RNAsee head-to-head (coding-only)")
        A("")
        A(f"- Coding pathogenic C>T: **{a8['n_coding']:,}**")
        A(f"- Rank correlation Spearman(GB, RNAsee) = **{a8['spearman_gb_vs_rnasee']:.3f}** "
          f"(p = {a8['spearman_p']:.2e})")
        A("")
        A("### Top-K overlap between GB and RNAsee")
        A("")
        A("| Top-K | Intersection | Jaccard |")
        A("|-------|-------------:|--------:|")
        for k, r in a8["top_1000_overlap"].items():
            A(f"| {k} | {r['intersection']} | {r['jaccard']:.3f} |")
        A("")
        A("### Composition of each top-1000")
        A("")
        A("| Source | nonsense | missense | synonymous | CpG% | TC% |")
        A("|--------|---------:|---------:|-----------:|-----:|----:|")
        base = a8["composition"]["coding_baseline"]
        A(f"| baseline (coding) | {base['nonsense_frac']*100:.1f}% | "
          f"{base['missense_frac']*100:.1f}% | {base['synonymous_frac']*100:.1f}% | "
          f"{base['cpg_frac']*100:.1f}% | {base['tc_frac']*100:.1f}% |")
        for label in ("gb", "rnasee"):
            c = a8["composition"][label]
            A(f"| {label} top-1000 | {c['nonsense_frac']*100:.1f}% | "
              f"{c['missense_frac']*100:.1f}% | {c['synonymous_frac']*100:.1f}% | "
              f"{c['cpg_frac']*100:.1f}% | {c['tc_frac']*100:.1f}% |")
        A("")
        A("Interpretation: RNAsee is trained on *observed endogenous editing rates* ")
        A("in healthy GTEx tissues, so its top picks are positions evolution has ")
        A("*tolerated* being edited — disproportionately synonymous / tolerant codon ")
        A("positions, biased **away** from nonsense. Our GB is trained against ")
        A("motif-matched negatives, so it learns APOBEC *structural accessibility* ")
        A("regardless of whether editing was evolutionarily tolerated — hence its ")
        A("top picks include CGA/CAA/CAG positions where editing would create a ")
        A("stop. The two models disagree on direction for nonsense enrichment ")
        A("because they are solving different problems.")
        A("")

    # ---- Verdict ----
    A("---")
    A("")
    A("## Verdict")
    A("")
    verdict = results.get("verdict", "See analyses.")
    A(verdict)
    A("")

    out_path.write_text("\n".join(lines))
    print(f"Wrote report: {out_path}")


# -----------------------------------------------------------------------------
# Verdict synthesizer
# -----------------------------------------------------------------------------
def synthesize_verdict(results):
    a1 = results["analysis_1_reproduce"]
    a2 = results["analysis_2_cpg_stratification"]
    a6 = results["analysis_6_logreg"]
    a7 = results.get("analysis_7_rnasee", {})
    a8 = results.get("analysis_8_head_to_head", {})

    # PRIMARY = coding-only
    primary = a1["primary_coding_only"]
    top1000 = primary["top_k_results"].get("1000", {})
    repro_or = top1000.get("odds_ratio", float("nan"))
    repro_frac = top1000.get("top_nonsense_frac", float("nan"))
    base_frac = primary["baseline_nonsense_fraction"]

    # Reference (all pathogenic) for flagging the original-claim framing
    ref = a1["all_pathogenic_reference"]
    ref_top1000 = ref["top_k_results"].get("1000", {})
    ref_or = ref_top1000.get("odds_ratio", float("nan"))
    ref_frac = ref_top1000.get("top_nonsense_frac", float("nan"))
    ref_base = ref["baseline_nonsense_fraction"]

    noncpg_or = a2.get("nonCpG", {}).get("top_k", {}).get("1000", {}).get("odds_ratio", float("nan"))
    cpg_or = a2.get("CpG", {}).get("top_k", {}).get("1000", {}).get("odds_ratio", float("nan"))
    noncpg_p = a2.get("nonCpG", {}).get("top_k", {}).get("1000", {}).get("p_value", float("nan"))

    logreg_full = a6["full_controls"]["params"].get("p_edited_gb", {})
    logreg_score_only = a6["score_only"]["params"].get("p_edited_gb", {})
    coef_full = logreg_full.get("coef", float("nan"))
    p_full = logreg_full.get("p_value", float("nan"))
    coef_solo = logreg_score_only.get("coef", float("nan"))

    # RNAsee coding-only top-1000
    rnasee_top1000 = a7.get("overall_top_k", {}).get("1000", {})
    rnasee_frac = rnasee_top1000.get("top_nonsense_frac", float("nan"))
    rnasee_or = rnasee_top1000.get("odds_ratio", float("nan"))

    parts = []
    parts.append(
        f"**PRIMARY (coding-only)**: top-1000 nonsense = {repro_frac*100:.1f}% vs "
        f"baseline = {base_frac*100:.1f}%, OR = {repro_or:.2f}. The original 4b "
        f"headline (59.5% / 47.4% / OR=1.64) used the all-pathogenic pool; the "
        f"all-pathogenic re-derivation here is {ref_frac*100:.1f}% / "
        f"{ref_base*100:.1f}% / OR={ref_or:.2f}, and most of that OR is the "
        f"trivial 'coding > non-coding' ranking (non-coding is {ref['baseline_consequence_distribution'].get('non_coding', 0):,} "
        f"of {ref['n_pathogenic']:,} rows). The coding-only OR is the like-for-like claim."
    )

    if not np.isnan(noncpg_or):
        if noncpg_or > 1.3 and noncpg_p < 1e-3:
            parts.append(
                f"**CpG stratification**: enrichment SURVIVES in non-CpG stratum "
                f"(top-1000 OR={noncpg_or:.2f}, p={noncpg_p:.2e}). The claim is not a pure CpG artifact."
            )
        elif noncpg_or < 1.1:
            parts.append(
                f"**CpG stratification**: in the non-CpG stratum the top-1000 OR collapses to "
                f"{noncpg_or:.2f} (p={noncpg_p:.2e}). The enrichment is largely CpG-driven."
            )
        else:
            parts.append(
                f"**CpG stratification**: non-CpG top-1000 OR={noncpg_or:.2f} (p={noncpg_p:.2e}). Attenuated but not eliminated."
            )

    if not np.isnan(coef_full):
        if p_full < 1e-3 and coef_full > 0:
            parts.append(
                f"**Logistic regression with full controls**: model_score coefficient remains "
                f"significantly positive (coef={coef_full:.3f}, p={p_full:.2e}) after controlling for "
                f"CpG, TC motif, codon position, and gene length. The model captures nonsense signal "
                f"beyond pure CpG bias."
            )
        elif p_full > 0.01:
            parts.append(
                f"**Logistic regression**: after controlling for CpG/motif/codon-pos/gene-length, "
                f"model_score is NOT a significant predictor of nonsense (coef={coef_full:.3f}, "
                f"p={p_full:.2e}). The original claim is explained by CpG / codon-position confounds."
            )
        else:
            parts.append(
                f"**Logistic regression**: model_score is marginally significant after controls "
                f"(coef={coef_full:.3f}, p={p_full:.2e})."
            )

    # Overall label
    if not np.isnan(noncpg_or) and not np.isnan(p_full):
        if noncpg_or > 1.3 and p_full < 1e-3:
            parts.append("**Overall: CONFIRMED.** The nonsense enrichment is not a CpG artifact; "
                         "it persists in the non-CpG stratum and the model_score coefficient "
                         "is robust to full controls.")
        elif noncpg_or < 1.1 and p_full > 0.01:
            parts.append("**Overall: CONFIRMED BUT MECHANISTICALLY A CpG ARTIFACT.** The top-K "
                         "over-representation of nonsense variants is explained by the model's "
                         "CpG bias (CGA codons), which happen to overlap with stop-gain routes.")
        else:
            parts.append("**Overall: AMBIGUOUS / PARTIAL CpG ARTIFACT.** The nonsense signal is "
                         "attenuated but not eliminated after CpG control. A portion of the claim "
                         "survives, a portion is CpG-driven.")
    return " ".join(parts)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    print("="*70)
    print("ClinVar Nonsense Enrichment Stress Test")
    print("="*70)

    df = build_annotated_dataframe()
    print(f"\nAnnotated DF: {len(df):,} pathogenic C>T variants")
    print(f"Consequence class counts:")
    print(df["consequence"].value_counts().to_string())

    results = {}
    print("\n--- Analysis 1: Reproduction ---")
    results["analysis_1_reproduce"] = analysis_1_reproduce(df)

    print("\n--- Analysis 2: CpG stratification ---")
    results["analysis_2_cpg_stratification"] = analysis_2_cpg_stratification(df)

    print("\n--- Analysis 3: Codon-route decomposition ---")
    results["analysis_3_codon_routes"] = analysis_3_codon_routes(df)

    print("\n--- Analysis 4: Achievability-matched baseline ---")
    results["analysis_4_achievability_matched"] = analysis_4_achievability_matched(df)

    print("\n--- Analysis 5: Per-gene clustering ---")
    results["analysis_5_per_gene"] = analysis_5_per_gene(df)

    print("\n--- Analysis 6: Joint logistic regression ---")
    results["analysis_6_logreg"] = analysis_6_logreg(df)

    print("\n--- Analysis 7: Same analysis on RNAsee (coding-only) ---")
    results["analysis_7_rnasee"] = analysis_7_rnasee(df)

    print("\n--- Analysis 8: GB vs RNAsee head-to-head (coding-only) ---")
    results["analysis_8_head_to_head"] = analysis_8_head_to_head(df)

    # Synthesize verdict
    results["verdict"] = synthesize_verdict(results)

    # Save
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved results: {RESULTS_JSON}")

    write_report(results, REPORT_MD)

    # Stdout summary
    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)
    print(results["verdict"])


if __name__ == "__main__":
    main()
