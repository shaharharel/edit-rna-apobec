#!/usr/bin/env python3
"""TCGA APOBEC mutation enrichment, gnomAD constraint, and exome editability analysis.

Three analyses:
1. TCGA: Do C>T somatic mutations in APOBEC-high cancers fall at APOBEC-editable positions?
2. gnomAD: Are genes with more editing sites under stronger evolutionary constraint?
3. Exome editability: Score all exonic C positions on chr22 as a pilot.

Usage:
    conda activate quris
    python scripts/multi_enzyme/tcga_gnomad_editability.py [--pilot] [--full]
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from pyfaidx import Fasta

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.apobec_feature_extraction import extract_motif_from_seq

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ============================================================================
# Paths
# ============================================================================
DATA_DIR = PROJECT_ROOT / "data"
HG19_FA = DATA_DIR / "raw/genomes/hg19.fa"
REFGENE_HG19 = DATA_DIR / "raw/genomes/refGene_hg19.txt"
GNOMAD_TSV = DATA_DIR / "raw/gnomad/gnomad_v4.1_constraint.tsv"
MULTI_SPLITS = DATA_DIR / "processed/multi_enzyme/splits_multi_enzyme_v3_with_negatives.csv"
MULTI_SEQS = DATA_DIR / "processed/multi_enzyme/multi_enzyme_sequences_v3_with_negatives.json"

OUTPUT_DIR = PROJECT_ROOT / "experiments/multi_enzyme/outputs/tcga_gnomad"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# cBioPortal data hub URLs (pan_can_atlas_2018)
CBIO_BASE = "https://media.githubusercontent.com/media/cBioPortal/datahub/master/public"
CANCER_STUDIES = {
    "blca": "blca_tcga_pan_can_atlas_2018",
    "brca": "brca_tcga_pan_can_atlas_2018",
    "cesc": "cesc_tcga_pan_can_atlas_2018",
    "lusc": "lusc_tcga_pan_can_atlas_2018",
    "skcm": "skcm_tcga_pan_can_atlas_2018",  # negative control
}

APOBEC_HIGH = ["blca", "brca", "cesc", "lusc"]
APOBEC_LOW = ["skcm"]


# ============================================================================
# Helper: Train MotifOnly XGBoost on multi-enzyme v3 data
# ============================================================================
def train_motif_model():
    """Train a MotifOnly XGBoost classifier on multi-enzyme v3 data."""
    from xgboost import XGBClassifier

    logger.info("Loading multi-enzyme v3 data for MotifOnly model training...")
    splits = pd.read_csv(MULTI_SPLITS)
    with open(MULTI_SEQS) as f:
        sequences = json.load(f)

    # Use all data (pos and neg) for training
    site_ids = splits["site_id"].astype(str).tolist()
    labels = splits["is_edited"].values

    # Extract 24-dim motif features
    features = np.array([extract_motif_from_seq(sequences.get(sid, "N" * 201)) for sid in site_ids],
                        dtype=np.float32)

    logger.info(f"Training MotifOnly XGBoost on {len(labels)} sites "
                f"({labels.sum()} pos, {(1-labels).sum()} neg)...")

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
        use_label_encoder=False,
    )
    model.fit(features, labels)

    # Quick check: training AUC
    from sklearn.metrics import roc_auc_score
    train_probs = model.predict_proba(features)[:, 1]
    train_auc = roc_auc_score(labels, train_probs)
    logger.info(f"MotifOnly training AUC: {train_auc:.4f}")

    return model


def score_sequences_motif(model, seqs_201nt: list) -> np.ndarray:
    """Score a list of 201-nt sequences with the MotifOnly model.

    Args:
        model: Trained XGBClassifier.
        seqs_201nt: List of 201-nt strings (C at center position 100).

    Returns:
        Array of editability scores (probability of being edited).
    """
    if not seqs_201nt:
        return np.array([])
    features = np.array([extract_motif_from_seq(s) for s in seqs_201nt], dtype=np.float32)
    return model.predict_proba(features)[:, 1]


# ============================================================================
# Helper: Parse refGene for exonic regions
# ============================================================================
def parse_refgene_exons(chrom_filter=None):
    """Parse refGene to get coding exonic regions.

    refGene columns (no header):
    0:bin, 1:name, 2:chrom, 3:strand, 4:txStart, 5:txEnd,
    6:cdsStart, 7:cdsEnd, 8:exonCount, 9:exonStarts, 10:exonEnds,
    11:score, 12:name2(gene), 13:cdsStartStat, 14:cdsEndStat, 15:exonFrames

    Returns: list of (chrom, exon_start, exon_end, strand, gene_name)
    """
    logger.info(f"Parsing refGene from {REFGENE_HG19}...")
    exons = []
    seen = set()
    with open(REFGENE_HG19) as f:
        for line in f:
            fields = line.strip().split("\t")
            if len(fields) < 13:
                continue
            chrom = fields[2]
            if chrom_filter and chrom != chrom_filter:
                continue
            strand = fields[3]
            cds_start = int(fields[6])
            cds_end = int(fields[7])
            if cds_start == cds_end:  # non-coding
                continue
            gene = fields[12]
            exon_starts = [int(x) for x in fields[9].rstrip(",").split(",") if x]
            exon_ends = [int(x) for x in fields[10].rstrip(",").split(",") if x]
            for es, ee in zip(exon_starts, exon_ends):
                # Clip to CDS
                es = max(es, cds_start)
                ee = min(ee, cds_end)
                if es >= ee:
                    continue
                key = (chrom, es, ee)
                if key not in seen:
                    seen.add(key)
                    exons.append((chrom, es, ee, strand, gene))
    logger.info(f"Parsed {len(exons)} coding exonic regions" +
                (f" on {chrom_filter}" if chrom_filter else ""))
    return exons


# ============================================================================
# Helper: Extract 201-nt window around a position from genome
# ============================================================================
def extract_window(genome, chrom, pos, strand="+", window=100):
    """Extract 201-nt window centered on pos. Returns sequence with C at center for + strand.

    For + strand C>T: the C is at pos, extract [pos-100, pos+100].
    For - strand (G>A on + strand): the G is at pos on + strand, meaning C on - strand.
    We extract the window and reverse complement.

    Args:
        genome: pyfaidx Fasta object.
        chrom: Chromosome name.
        pos: 0-based position.
        strand: "+" or "-".
        window: Half-window size (default 100 for 201-nt).

    Returns:
        201-nt string (uppercase), or None if out of bounds.
    """
    chrom_len = len(genome[chrom])
    start = pos - window
    end = pos + window + 1  # inclusive
    if start < 0 or end > chrom_len:
        return None

    seq = str(genome[chrom][start:end]).upper()

    if strand == "-":
        comp = str.maketrans("ACGT", "TGCA")
        seq = seq.translate(comp)[::-1]

    # Convert to RNA-like (T->U not needed here, extract_motif_from_seq handles it)
    return seq


# ============================================================================
# Analysis 1: TCGA APOBEC Mutation Enrichment
# ============================================================================
def download_maf(cancer_type, cache_dir=None):
    """Download MAF file from cBioPortal. Returns path to local file."""
    if cache_dir is None:
        cache_dir = DATA_DIR / "raw/tcga"
    cache_dir.mkdir(parents=True, exist_ok=True)

    study = CANCER_STUDIES[cancer_type]
    local_path = cache_dir / f"{study}_mutations.txt"

    if local_path.exists():
        logger.info(f"Using cached MAF: {local_path}")
        return local_path

    url = f"{CBIO_BASE}/{study}/data_mutations.txt"
    logger.info(f"Downloading {cancer_type} MAF from {url}...")

    import urllib.request
    try:
        urllib.request.urlretrieve(url, local_path)
        logger.info(f"Downloaded {local_path} ({local_path.stat().st_size / 1e6:.1f} MB)")
    except Exception as e:
        logger.error(f"Failed to download {cancer_type} MAF: {e}")
        # Try alternative URL pattern
        alt_url = f"https://raw.githubusercontent.com/cBioPortal/datahub/master/public/{study}/data_mutations.txt"
        logger.info(f"Trying alternative URL: {alt_url}")
        try:
            urllib.request.urlretrieve(alt_url, local_path)
            logger.info(f"Downloaded {local_path} ({local_path.stat().st_size / 1e6:.1f} MB)")
        except Exception as e2:
            logger.error(f"Alternative URL also failed: {e2}")
            return None

    return local_path


def parse_maf_ct_mutations(maf_path, chrom_filter=None):
    """Parse MAF file to extract C>T and G>A somatic mutations.

    Returns DataFrame with columns: chrom, pos (0-based), ref, alt, strand_inferred, gene, sample_id
    """
    logger.info(f"Parsing MAF: {maf_path}...")

    # Read with flexible parsing - MAFs have comment lines starting with #
    chunks = []
    for chunk in pd.read_csv(maf_path, sep="\t", comment="#", low_memory=False, chunksize=100000):
        # Filter to SNPs
        if "Variant_Type" in chunk.columns:
            chunk = chunk[chunk["Variant_Type"] == "SNP"]

        # Get C>T and G>A mutations
        if "Reference_Allele" not in chunk.columns:
            continue

        ct_mask = (chunk["Reference_Allele"] == "C") & (chunk["Tumor_Seq_Allele2"] == "T")
        ga_mask = (chunk["Reference_Allele"] == "G") & (chunk["Tumor_Seq_Allele2"] == "A")
        chunk = chunk[ct_mask | ga_mask].copy()

        if chrom_filter:
            chrom_col = "Chromosome"
            # Ensure chr prefix
            chunk[chrom_col] = chunk[chrom_col].astype(str)
            if not chunk[chrom_col].str.startswith("chr").any():
                chunk[chrom_col] = "chr" + chunk[chrom_col]
            chunk = chunk[chunk[chrom_col] == chrom_filter]

        if len(chunk) > 0:
            chunks.append(chunk)

    if not chunks:
        logger.warning(f"No C>T/G>A mutations found in {maf_path}")
        return pd.DataFrame()

    df = pd.concat(chunks, ignore_index=True)

    # Standardize
    chrom_col = "Chromosome"
    df[chrom_col] = df[chrom_col].astype(str)
    if not df[chrom_col].str.startswith("chr").any():
        df[chrom_col] = "chr" + df[chrom_col]

    # Convert to 0-based
    pos_col = "Start_Position"
    result = pd.DataFrame({
        "chrom": df[chrom_col],
        "pos": df[pos_col].astype(int) - 1,  # MAF is 1-based -> 0-based
        "ref": df["Reference_Allele"],
        "alt": df["Tumor_Seq_Allele2"],
        "strand_inferred": np.where(df["Reference_Allele"] == "C", "+", "-"),
        "gene": df.get("Hugo_Symbol", "unknown"),
        "sample_id": df.get("Tumor_Sample_Barcode", "unknown"),
    })

    if chrom_filter:
        result = result[result["chrom"] == chrom_filter]

    logger.info(f"Found {len(result)} C>T/G>A mutations" +
                (f" on {chrom_filter}" if chrom_filter else "") +
                f" (C>T: {(result['ref']=='C').sum()}, G>A: {(result['ref']=='G').sum()})")

    return result


def generate_matched_controls(mutations_df, genome, exons_by_gene, n_controls=10, rng=None):
    """For each mutation, pick n_controls random C positions in the same gene's exons.

    Args:
        mutations_df: DataFrame with chrom, pos, strand_inferred, gene columns.
        genome: pyfaidx Fasta.
        exons_by_gene: dict of gene -> list of (chrom, start, end) tuples.
        n_controls: Number of control positions per mutation.
        rng: numpy random Generator.

    Returns:
        DataFrame with same columns as mutations_df + 'is_mutation' column.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    logger.info(f"Generating {n_controls} matched controls per mutation...")

    mutated_positions = set(zip(mutations_df["chrom"], mutations_df["pos"]))
    controls = []

    genes_processed = 0
    genes_with_controls = 0

    # Group by gene for efficiency
    for gene, group in mutations_df.groupby("gene"):
        gene_exons = exons_by_gene.get(gene, [])
        if not gene_exons:
            continue

        genes_processed += 1

        # Collect all C and G positions in this gene's exons
        c_positions = []  # (chrom, pos, strand)
        for chrom, es, ee, strand, _ in gene_exons:
            if chrom not in genome.keys():
                continue
            try:
                exon_seq = str(genome[chrom][es:ee]).upper()
            except Exception:
                continue
            for i, base in enumerate(exon_seq):
                abs_pos = es + i
                if (chrom, abs_pos) in mutated_positions:
                    continue
                if base == "C":
                    c_positions.append((chrom, abs_pos, "+"))
                elif base == "G":
                    c_positions.append((chrom, abs_pos, "-"))

        if not c_positions:
            continue

        genes_with_controls += 1

        # Sample controls for each mutation in this gene
        n_muts = len(group)
        n_needed = n_muts * n_controls
        if len(c_positions) >= n_needed:
            chosen = rng.choice(len(c_positions), size=n_needed, replace=False)
        else:
            chosen = rng.choice(len(c_positions), size=n_needed, replace=True)

        for i, (_, row) in enumerate(group.iterrows()):
            for j in range(n_controls):
                idx = chosen[i * n_controls + j]
                c_chrom, c_pos, c_strand = c_positions[idx]
                controls.append({
                    "chrom": c_chrom,
                    "pos": c_pos,
                    "ref": "C" if c_strand == "+" else "G",
                    "alt": "control",
                    "strand_inferred": c_strand,
                    "gene": gene,
                    "sample_id": "control",
                })

    logger.info(f"Generated {len(controls)} controls for {genes_processed} genes "
                f"({genes_with_controls} had C positions)")

    controls_df = pd.DataFrame(controls)
    return controls_df


def run_tcga_analysis(cancer_types, genome, model, chrom_filter=None, exons_by_gene=None):
    """Run TCGA mutation enrichment analysis for given cancer types.

    Returns dict of results per cancer type.
    """
    results = {}

    for ct in cancer_types:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {ct.upper()} cancer...")
        logger.info(f"{'='*60}")

        # Download/load MAF
        maf_path = download_maf(ct)
        if maf_path is None:
            logger.error(f"Skipping {ct}: MAF not available")
            continue

        # Parse C>T mutations
        mutations = parse_maf_ct_mutations(maf_path, chrom_filter=chrom_filter)
        if mutations.empty:
            logger.warning(f"No mutations for {ct}")
            continue

        # Deduplicate mutations by position (keep unique positions)
        mutations_dedup = mutations.drop_duplicates(subset=["chrom", "pos"]).copy()
        logger.info(f"Unique mutation positions: {len(mutations_dedup)} "
                    f"(from {len(mutations)} total including recurrences)")

        # Generate matched controls
        controls = generate_matched_controls(mutations_dedup, genome, exons_by_gene,
                                              n_controls=10)

        # Score mutations
        mut_seqs = []
        mut_valid_idx = []
        for i, (_, row) in enumerate(mutations_dedup.iterrows()):
            seq = extract_window(genome, row["chrom"], row["pos"], row["strand_inferred"])
            if seq is not None and len(seq) == 201:
                mut_seqs.append(seq)
                mut_valid_idx.append(i)

        logger.info(f"Scoring {len(mut_seqs)} mutation positions...")
        mut_scores = score_sequences_motif(model, mut_seqs)

        # Score controls
        ctrl_seqs = []
        ctrl_valid_idx = []
        for i, (_, row) in enumerate(controls.iterrows()):
            seq = extract_window(genome, row["chrom"], row["pos"], row["strand_inferred"])
            if seq is not None and len(seq) == 201:
                ctrl_seqs.append(seq)
                ctrl_valid_idx.append(i)

        logger.info(f"Scoring {len(ctrl_seqs)} control positions...")
        ctrl_scores = score_sequences_motif(model, ctrl_seqs)

        # Compute statistics
        mean_mut = np.mean(mut_scores) if len(mut_scores) > 0 else 0
        mean_ctrl = np.mean(ctrl_scores) if len(ctrl_scores) > 0 else 0

        # Mann-Whitney U test
        if len(mut_scores) > 0 and len(ctrl_scores) > 0:
            u_stat, u_pval = stats.mannwhitneyu(mut_scores, ctrl_scores, alternative="greater")
        else:
            u_stat, u_pval = 0, 1.0

        # Odds ratio at threshold 0.5
        threshold = 0.5
        mut_high = np.sum(mut_scores >= threshold)
        mut_low = np.sum(mut_scores < threshold)
        ctrl_high = np.sum(ctrl_scores >= threshold)
        ctrl_low = np.sum(ctrl_scores < threshold)

        if ctrl_high > 0 and mut_low > 0 and ctrl_low > 0:
            odds_ratio = (mut_high / mut_low) / (ctrl_high / ctrl_low)
        else:
            odds_ratio = float("inf") if mut_high > 0 else 0.0

        # Fisher's exact test
        from scipy.stats import fisher_exact
        table = [[mut_high, mut_low], [ctrl_high, ctrl_low]]
        fisher_or, fisher_p = fisher_exact(table, alternative="greater")

        # Recurrence analysis: count how many samples have mutations at each position
        recurrence = mutations.groupby(["chrom", "pos"]).size().reset_index(name="n_samples")
        # Merge scores with recurrence
        mut_pos_df = mutations_dedup.iloc[mut_valid_idx].copy()
        mut_pos_df["editability"] = mut_scores
        mut_pos_df = mut_pos_df.merge(recurrence, on=["chrom", "pos"], how="left")

        # Spearman correlation between editability and recurrence
        if len(mut_pos_df) > 10:
            spearman_r, spearman_p = stats.spearmanr(mut_pos_df["editability"],
                                                       mut_pos_df["n_samples"])
        else:
            spearman_r, spearman_p = 0, 1.0

        ct_result = {
            "cancer_type": ct,
            "n_mutations": len(mutations),
            "n_unique_positions": len(mutations_dedup),
            "n_scored_mutations": len(mut_scores),
            "n_controls": len(ctrl_scores),
            "mean_editability_mutations": float(mean_mut),
            "mean_editability_controls": float(mean_ctrl),
            "delta_editability": float(mean_mut - mean_ctrl),
            "mann_whitney_U": float(u_stat),
            "mann_whitney_p": float(u_pval),
            "threshold": threshold,
            "mutations_above_threshold": int(mut_high),
            "mutations_below_threshold": int(mut_low),
            "controls_above_threshold": int(ctrl_high),
            "controls_below_threshold": int(ctrl_low),
            "fisher_odds_ratio": float(fisher_or),
            "fisher_p": float(fisher_p),
            "recurrence_editability_spearman": float(spearman_r),
            "recurrence_editability_p": float(spearman_p),
        }
        results[ct] = ct_result

        # --- TC-context sub-analysis ---
        # Filter mutations to those at TC dinucleotide context (APOBEC signature)
        # For + strand C>T: check if upstream base is T
        # For - strand G>A: the C is on - strand, check if upstream on - strand is T
        tc_mut_mask = []
        tc_mut_seqs = []
        non_tc_mut_seqs = []
        for i, seq in enumerate(mut_seqs):
            # seq has C at position 100 (center). Check position 99 for T.
            upstream = seq[99].upper() if len(seq) > 99 else "N"
            is_tc = upstream in ("T", "U")
            tc_mut_mask.append(is_tc)
            if is_tc:
                tc_mut_seqs.append(seq)
            else:
                non_tc_mut_seqs.append(seq)

        tc_ctrl_mask = []
        tc_ctrl_seqs = []
        for i, seq in enumerate(ctrl_seqs):
            upstream = seq[99].upper() if len(seq) > 99 else "N"
            is_tc = upstream in ("T", "U")
            tc_ctrl_mask.append(is_tc)
            if is_tc:
                tc_ctrl_seqs.append(seq)

        tc_mut_scores = mut_scores[np.array(tc_mut_mask)] if len(tc_mut_mask) > 0 else np.array([])
        tc_ctrl_scores = ctrl_scores[np.array(tc_ctrl_mask)] if len(tc_ctrl_mask) > 0 else np.array([])
        non_tc_mut_scores = mut_scores[~np.array(tc_mut_mask)] if len(tc_mut_mask) > 0 else np.array([])

        tc_frac_mut = np.mean(tc_mut_mask) if tc_mut_mask else 0
        tc_frac_ctrl = np.mean(tc_ctrl_mask) if tc_ctrl_mask else 0

        # TC-context enrichment: is the TC fraction higher in mutations vs controls?
        # This directly tests APOBEC signature
        tc_fisher_table = [
            [int(np.sum(tc_mut_mask)), int(np.sum(~np.array(tc_mut_mask)))],
            [int(np.sum(tc_ctrl_mask)), int(np.sum(~np.array(tc_ctrl_mask)))]
        ]
        tc_context_or, tc_context_p = fisher_exact(tc_fisher_table, alternative="greater")

        # TC-context editability comparison
        if len(tc_mut_scores) > 0 and len(tc_ctrl_scores) > 0:
            tc_u, tc_p = stats.mannwhitneyu(tc_mut_scores, tc_ctrl_scores, alternative="greater")
        else:
            tc_u, tc_p = 0, 1.0

        tc_result = {
            "n_tc_mutations": int(np.sum(tc_mut_mask)),
            "n_tc_controls": int(np.sum(tc_ctrl_mask)),
            "tc_frac_mutations": float(tc_frac_mut),
            "tc_frac_controls": float(tc_frac_ctrl),
            "tc_context_enrichment_OR": float(tc_context_or),
            "tc_context_enrichment_p": float(tc_context_p),
            "tc_mean_editability_mutations": float(np.mean(tc_mut_scores)) if len(tc_mut_scores) > 0 else 0,
            "tc_mean_editability_controls": float(np.mean(tc_ctrl_scores)) if len(tc_ctrl_scores) > 0 else 0,
            "tc_mann_whitney_p": float(tc_p),
        }

        # Multi-threshold analysis
        multi_thresh = {}
        for thr in [0.3, 0.4, 0.5, 0.6, 0.7]:
            mh = int(np.sum(mut_scores >= thr))
            ml = int(np.sum(mut_scores < thr))
            ch = int(np.sum(ctrl_scores >= thr))
            cl = int(np.sum(ctrl_scores < thr))
            if ch > 0 and ml > 0 and cl > 0:
                ft_or, ft_p = fisher_exact([[mh, ml], [ch, cl]], alternative="greater")
            else:
                ft_or, ft_p = float("nan"), float("nan")
            multi_thresh[str(thr)] = {"OR": float(ft_or), "p": float(ft_p),
                                       "n_mut_above": mh, "n_ctrl_above": ch}

        ct_result["tc_context_analysis"] = tc_result
        ct_result["multi_threshold"] = multi_thresh

        logger.info(f"\n--- {ct.upper()} Results ---")
        logger.info(f"Mutations scored: {len(mut_scores)}, Controls: {len(ctrl_scores)}")
        logger.info(f"Mean editability - mutations: {mean_mut:.4f}, controls: {mean_ctrl:.4f}")
        logger.info(f"Mann-Whitney p={u_pval:.2e}")
        logger.info(f"Fisher OR={fisher_or:.3f}, p={fisher_p:.2e}")
        logger.info(f"Recurrence-editability Spearman r={spearman_r:.3f}, p={spearman_p:.2e}")
        logger.info(f"TC context: {tc_frac_mut:.1%} of mutations vs {tc_frac_ctrl:.1%} of controls, "
                    f"OR={tc_context_or:.3f}, p={tc_context_p:.2e}")
        logger.info(f"TC-only editability: mut={tc_result['tc_mean_editability_mutations']:.4f}, "
                    f"ctrl={tc_result['tc_mean_editability_controls']:.4f}, p={tc_p:.2e}")
        logger.info(f"Multi-threshold ORs: " +
                    ", ".join(f"t={t}: OR={v['OR']:.3f}" for t, v in multi_thresh.items()))

    return results


# ============================================================================
# Analysis 2: gnomAD Gene Constraint
# ============================================================================
def run_gnomad_analysis():
    """Test whether genes with more editing sites are more constrained."""
    logger.info("\n" + "=" * 60)
    logger.info("Analysis 2: gnomAD Gene Constraint")
    logger.info("=" * 60)

    # Load gnomAD constraint
    gnomad = pd.read_csv(GNOMAD_TSV, sep="\t", low_memory=False)
    logger.info(f"gnomAD: {len(gnomad)} rows")

    # Keep canonical transcripts only to avoid duplicates
    if "canonical" in gnomad.columns:
        gnomad_canon = gnomad[gnomad["canonical"] == True].copy()
        if len(gnomad_canon) > 1000:
            gnomad = gnomad_canon
    logger.info(f"gnomAD canonical: {len(gnomad)} genes")

    # Extract key constraint columns
    constraint_cols = {}
    for col in gnomad.columns:
        if "pLI" in col and "lof_hc_lc" not in col:
            constraint_cols["pLI"] = col
        if "oe_ci.upper" in col and "lof." in col and "rank" not in col and "bin" not in col:
            constraint_cols["LOEUF"] = col
        if "z_score" in col and "mis." in col:
            constraint_cols["mis_z"] = col

    logger.info(f"Constraint columns found: {constraint_cols}")

    # Build gene-level gnomAD table
    gene_constraint = gnomad[["gene"]].copy()
    for name, col in constraint_cols.items():
        gene_constraint[name] = pd.to_numeric(gnomad[col], errors="coerce")
    gene_constraint = gene_constraint.dropna(subset=list(constraint_cols.keys()), how="all")
    gene_constraint = gene_constraint.drop_duplicates(subset=["gene"], keep="first")
    logger.info(f"Genes with constraint data: {len(gene_constraint)}")

    # Load editing sites
    splits = pd.read_csv(MULTI_SPLITS)
    positives = splits[splits["is_edited"] == 1].copy()

    # Extract gene names from site IDs or use flanking info
    # Site IDs look like chr1:1603054:- — we need gene annotation
    # Use refGene to map positions to genes
    # For speed, just use the gene column if available
    # Let's map editing sites to genes using refGene hg38 (most sites are hg38)
    logger.info("Mapping editing sites to genes...")

    # Quick approach: load refGene hg38 and build position -> gene mapping
    refgene_path = DATA_DIR / "raw/genomes/refGene.txt"  # hg38
    gene_regions = defaultdict(list)  # gene -> [(chrom, start, end)]
    with open(refgene_path) as f:
        for line in f:
            fields = line.strip().split("\t")
            if len(fields) < 13:
                continue
            gene = fields[12]
            chrom = fields[2]
            tx_start = int(fields[4])
            tx_end = int(fields[5])
            gene_regions[gene].append((chrom, tx_start, tx_end))

    # Build interval lookup per chromosome
    chrom_intervals = defaultdict(list)  # chrom -> [(start, end, gene)]
    for gene, regions in gene_regions.items():
        for chrom, start, end in regions:
            chrom_intervals[chrom].append((start, end, gene))

    # Sort intervals
    for chrom in chrom_intervals:
        chrom_intervals[chrom].sort()

    # Map each editing site to a gene using binary search
    import bisect

    def find_gene(chrom, pos):
        intervals = chrom_intervals.get(chrom, [])
        # Binary search for intervals containing pos
        idx = bisect.bisect_right(intervals, (pos, float("inf"), "")) - 1
        # Check nearby intervals
        for i in range(max(0, idx - 5), min(len(intervals), idx + 5)):
            s, e, g = intervals[i]
            if s <= pos <= e:
                return g
        return None

    site_genes = []
    for _, row in positives.iterrows():
        gene = find_gene(row["chr"], row["start"])
        site_genes.append(gene)

    positives = positives.copy()
    positives["gene"] = site_genes
    positives_with_gene = positives.dropna(subset=["gene"])
    logger.info(f"Editing sites mapped to genes: {len(positives_with_gene)}/{len(positives)}")

    # Per-gene editing site count
    gene_edit_counts = positives_with_gene.groupby("gene").size().reset_index(name="n_editing_sites")

    # Merge with constraint
    merged = gene_edit_counts.merge(gene_constraint, on="gene", how="inner")
    logger.info(f"Genes with both editing sites and constraint data: {len(merged)}")

    # Also get genes WITHOUT editing sites for comparison
    all_genes_with_constraint = set(gene_constraint["gene"])
    genes_with_editing = set(gene_edit_counts["gene"])
    genes_without_editing = all_genes_with_constraint - genes_with_editing

    constraint_with_edit = gene_constraint[gene_constraint["gene"].isin(genes_with_editing)]
    constraint_without_edit = gene_constraint[gene_constraint["gene"].isin(genes_without_editing)]

    results = {"n_genes_with_editing": len(genes_with_editing),
               "n_genes_without_editing": len(genes_without_editing),
               "tests": {}}

    # Test 1: pLI comparison
    if "pLI" in constraint_with_edit.columns:
        pli_edit = constraint_with_edit["pLI"].dropna()
        pli_no_edit = constraint_without_edit["pLI"].dropna()
        u, p = stats.mannwhitneyu(pli_edit, pli_no_edit, alternative="greater")
        results["tests"]["pLI"] = {
            "mean_with_editing": float(pli_edit.mean()),
            "mean_without_editing": float(pli_no_edit.mean()),
            "median_with_editing": float(pli_edit.median()),
            "median_without_editing": float(pli_no_edit.median()),
            "mann_whitney_p": float(p),
        }
        logger.info(f"pLI: editing genes={pli_edit.mean():.3f}, non-editing={pli_no_edit.mean():.3f}, p={p:.2e}")

    # Test 2: LOEUF comparison (lower = more constrained)
    if "LOEUF" in constraint_with_edit.columns:
        loeuf_edit = constraint_with_edit["LOEUF"].dropna()
        loeuf_no_edit = constraint_without_edit["LOEUF"].dropna()
        u, p = stats.mannwhitneyu(loeuf_edit, loeuf_no_edit, alternative="less")
        results["tests"]["LOEUF"] = {
            "mean_with_editing": float(loeuf_edit.mean()),
            "mean_without_editing": float(loeuf_no_edit.mean()),
            "median_with_editing": float(loeuf_edit.median()),
            "median_without_editing": float(loeuf_no_edit.median()),
            "mann_whitney_p": float(p),
        }
        logger.info(f"LOEUF: editing genes={loeuf_edit.mean():.3f}, non-editing={loeuf_no_edit.mean():.3f}, p={p:.2e}")

    # Test 3: Missense Z-score (higher = more constrained)
    if "mis_z" in constraint_with_edit.columns:
        mz_edit = constraint_with_edit["mis_z"].dropna()
        mz_no_edit = constraint_without_edit["mis_z"].dropna()
        u, p = stats.mannwhitneyu(mz_edit, mz_no_edit, alternative="greater")
        results["tests"]["missense_z"] = {
            "mean_with_editing": float(mz_edit.mean()),
            "mean_without_editing": float(mz_no_edit.mean()),
            "median_with_editing": float(mz_edit.median()),
            "median_without_editing": float(mz_no_edit.median()),
            "mann_whitney_p": float(p),
        }
        logger.info(f"Missense Z: editing genes={mz_edit.mean():.3f}, non-editing={mz_no_edit.mean():.3f}, p={p:.2e}")

    # Test 4: Correlation between number of editing sites and constraint
    if "pLI" in merged.columns:
        valid = merged[["n_editing_sites", "pLI"]].dropna()
        if len(valid) > 10 and valid["n_editing_sites"].nunique() > 1:
            r, p = stats.spearmanr(valid["n_editing_sites"], valid["pLI"])
            results["tests"]["n_sites_vs_pLI_spearman"] = {"r": float(r), "p": float(p)}
            logger.info(f"n_editing_sites vs pLI: Spearman r={r:.3f}, p={p:.2e}")

    if "LOEUF" in merged.columns:
        valid = merged[["n_editing_sites", "LOEUF"]].dropna()
        if len(valid) > 10 and valid["n_editing_sites"].nunique() > 1:
            r, p = stats.spearmanr(valid["n_editing_sites"], valid["LOEUF"])
            results["tests"]["n_sites_vs_LOEUF_spearman"] = {"r": float(r), "p": float(p)}
            logger.info(f"n_editing_sites vs LOEUF: Spearman r={r:.3f}, p={p:.2e}")

    # Test 5: Fraction of editing-site genes with high pLI
    if "pLI" in constraint_with_edit.columns:
        frac_high_pli_edit = (constraint_with_edit["pLI"] > 0.9).mean()
        frac_high_pli_no_edit = (constraint_without_edit["pLI"] > 0.9).mean()
        # Fisher test
        a = int((constraint_with_edit["pLI"] > 0.9).sum())
        b = int((constraint_with_edit["pLI"] <= 0.9).sum())
        c = int((constraint_without_edit["pLI"] > 0.9).sum())
        d = int((constraint_without_edit["pLI"] <= 0.9).sum())
        fisher_or, fisher_p = stats.fisher_exact([[a, b], [c, d]], alternative="greater")
        results["tests"]["high_pLI_enrichment"] = {
            "frac_high_pli_editing_genes": float(frac_high_pli_edit),
            "frac_high_pli_nonediting_genes": float(frac_high_pli_no_edit),
            "fisher_OR": float(fisher_or),
            "fisher_p": float(fisher_p),
        }
        logger.info(f"High pLI (>0.9): editing genes={frac_high_pli_edit:.3f}, "
                    f"non-editing={frac_high_pli_no_edit:.3f}, OR={fisher_or:.2f}, p={fisher_p:.2e}")

    # --- Figures ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: pLI distribution
    if "pLI" in constraint_with_edit.columns:
        axes[0].hist(constraint_with_edit["pLI"].dropna(), bins=50, alpha=0.6,
                     label=f"Editing genes (n={len(constraint_with_edit)})", density=True)
        axes[0].hist(constraint_without_edit["pLI"].dropna(), bins=50, alpha=0.6,
                     label=f"Non-editing genes (n={len(constraint_without_edit)})", density=True)
        axes[0].set_xlabel("pLI")
        axes[0].set_ylabel("Density")
        axes[0].set_title("pLI Distribution")
        axes[0].legend(fontsize=8)

    # Panel 2: LOEUF distribution
    if "LOEUF" in constraint_with_edit.columns:
        axes[1].hist(constraint_with_edit["LOEUF"].dropna(), bins=50, alpha=0.6,
                     label="Editing genes", density=True)
        axes[1].hist(constraint_without_edit["LOEUF"].dropna(), bins=50, alpha=0.6,
                     label="Non-editing genes", density=True)
        axes[1].set_xlabel("LOEUF")
        axes[1].set_ylabel("Density")
        axes[1].set_title("LOEUF Distribution")
        axes[1].legend(fontsize=8)

    # Panel 3: n_editing_sites vs LOEUF
    if "LOEUF" in merged.columns:
        axes[2].scatter(merged["n_editing_sites"], merged["LOEUF"], alpha=0.3, s=10)
        axes[2].set_xlabel("Number of editing sites")
        axes[2].set_ylabel("LOEUF")
        axes[2].set_title("Editing sites vs Constraint")

    plt.tight_layout()
    fig_path = OUTPUT_DIR / "gnomad_constraint.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    logger.info(f"Saved gnomAD figure: {fig_path}")

    return results


# ============================================================================
# Analysis 3: Exome Editability Map (chr22 pilot)
# ============================================================================
def run_exome_editability(genome, model, chrom="chr22"):
    """Score all exonic C positions on a chromosome."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Analysis 3: Exome Editability Map ({chrom})")
    logger.info(f"{'='*60}")

    exons = parse_refgene_exons(chrom_filter=chrom)
    if not exons:
        logger.error(f"No exons found for {chrom}")
        return None

    # Enumerate all C positions
    records = []
    batch_seqs = []
    batch_meta = []

    for ex_chrom, es, ee, strand, gene in exons:
        try:
            exon_seq = str(genome[ex_chrom][es:ee]).upper()
        except Exception:
            continue

        for i, base in enumerate(exon_seq):
            abs_pos = es + i
            if base == "C":
                # Plus strand C
                trinuc = exon_seq[max(0, i-1):i+2]
                batch_meta.append((ex_chrom, abs_pos, "+", trinuc, gene))
            elif base == "G":
                # Minus strand C
                trinuc = exon_seq[max(0, i-1):i+2]
                batch_meta.append((ex_chrom, abs_pos, "-", trinuc, gene))

    logger.info(f"Found {len(batch_meta)} C/G positions in {chrom} exons")

    # Extract 201-nt windows and score in batches
    BATCH_SIZE = 10000
    all_scores = []

    for batch_start in range(0, len(batch_meta), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(batch_meta))
        batch = batch_meta[batch_start:batch_end]

        seqs = []
        valid_idx = []
        for i, (chrom_b, pos, strand, tri, gene) in enumerate(batch):
            seq = extract_window(genome, chrom_b, pos, strand)
            if seq is not None and len(seq) == 201:
                seqs.append(seq)
                valid_idx.append(batch_start + i)
            else:
                valid_idx.append(None)

        if seqs:
            scores = score_sequences_motif(model, seqs)
            score_iter = iter(scores)
            for i, (chrom_b, pos, strand, tri, gene) in enumerate(batch):
                if valid_idx[i] is not None:
                    all_scores.append((*batch[i], next(score_iter)))
                else:
                    pass  # skip positions too close to chromosome edge

        if batch_start % 50000 == 0:
            logger.info(f"  Scored {batch_start}/{len(batch_meta)} positions...")

    # Build output DataFrame
    df = pd.DataFrame(all_scores, columns=["chrom", "pos", "strand", "trinucleotide",
                                            "gene", "editability"])
    logger.info(f"Scored {len(df)} exonic C positions on {chrom}")
    logger.info(f"Score distribution: mean={df['editability'].mean():.4f}, "
                f"median={df['editability'].median():.4f}, "
                f">0.5: {(df['editability'] >= 0.5).sum()} ({(df['editability'] >= 0.5).mean()*100:.1f}%)")

    # Save
    out_path = OUTPUT_DIR / f"exome_editability_{chrom.replace('chr', 'chr')}.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Saved: {out_path}")

    # Figure: editability score distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(df["editability"], bins=100, edgecolor="none")
    axes[0].axvline(0.5, color="red", linestyle="--", label="threshold=0.5")
    axes[0].set_xlabel("Editability Score")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Editability Distribution ({chrom}, n={len(df):,})")
    axes[0].legend()

    # By trinucleotide context (top 10)
    tri_counts = df.groupby("trinucleotide")["editability"].agg(["mean", "count"])
    tri_counts = tri_counts[tri_counts["count"] >= 100].sort_values("mean", ascending=False).head(15)
    axes[1].barh(range(len(tri_counts)), tri_counts["mean"])
    axes[1].set_yticks(range(len(tri_counts)))
    axes[1].set_yticklabels(tri_counts.index)
    axes[1].set_xlabel("Mean Editability")
    axes[1].set_title("By Trinucleotide Context")

    plt.tight_layout()
    fig_path = OUTPUT_DIR / f"exome_editability_{chrom.replace('chr', 'chr')}.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    logger.info(f"Saved: {fig_path}")

    return {"chrom": chrom, "n_positions": len(df),
            "mean_editability": float(df["editability"].mean()),
            "frac_above_0.5": float((df["editability"] >= 0.5).mean())}


# ============================================================================
# Plotting: TCGA results comparison
# ============================================================================
def plot_tcga_results(results, suffix=""):
    """Create comparison plots across cancer types."""
    if not results:
        return

    cancer_types = list(results.keys())
    mean_mut = [results[ct]["mean_editability_mutations"] for ct in cancer_types]
    mean_ctrl = [results[ct]["mean_editability_controls"] for ct in cancer_types]
    fisher_ors = [results[ct]["fisher_odds_ratio"] for ct in cancer_types]
    fisher_ps = [results[ct]["fisher_p"] for ct in cancer_types]

    colors = ["#e74c3c" if ct in APOBEC_HIGH else "#3498db" for ct in cancer_types]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: Mean editability comparison
    x = np.arange(len(cancer_types))
    width = 0.35
    axes[0].bar(x - width/2, mean_mut, width, label="C>T mutations", color="#e74c3c", alpha=0.8)
    axes[0].bar(x + width/2, mean_ctrl, width, label="Matched controls", color="#3498db", alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([ct.upper() for ct in cancer_types])
    axes[0].set_ylabel("Mean Editability Score")
    axes[0].set_title("TCGA C>T Mutations vs Controls")
    axes[0].legend()

    # Panel 2: Odds ratios
    axes[1].bar(range(len(cancer_types)), fisher_ors, color=colors)
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_xticks(range(len(cancer_types)))
    axes[1].set_xticklabels([ct.upper() for ct in cancer_types])
    axes[1].set_ylabel("Fisher Odds Ratio (score >= 0.5)")
    axes[1].set_title("Enrichment of Editable Sites Among Mutations")

    # Add p-value annotations
    for i, (or_val, p_val) in enumerate(zip(fisher_ors, fisher_ps)):
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        axes[1].text(i, or_val + 0.02, sig, ha="center", fontsize=10)

    # Panel 3: Summary table as text
    axes[2].axis("off")
    table_data = []
    for ct in cancer_types:
        r = results[ct]
        p_str = f"{r['fisher_p']:.2e}" if r['fisher_p'] < 0.01 else f"{r['fisher_p']:.3f}"
        table_data.append([ct.upper(),
                           f"{r['n_scored_mutations']:,}",
                           f"{r['mean_editability_mutations']:.3f}",
                           f"{r['mean_editability_controls']:.3f}",
                           f"{r['fisher_odds_ratio']:.3f}",
                           p_str])

    table = axes[2].table(cellText=table_data,
                          colLabels=["Cancer", "N_mut", "Mean(mut)", "Mean(ctrl)", "OR", "p"],
                          cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    axes[2].set_title("Summary Statistics")

    plt.tight_layout()
    fig_path = OUTPUT_DIR / f"tcga_enrichment{suffix}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved TCGA figure: {fig_path}")


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="TCGA/gnomAD editability analysis")
    parser.add_argument("--pilot", action="store_true", default=False,
                        help="Run pilot only (chr22, bladder cancer)")
    parser.add_argument("--full", action="store_true", default=False,
                        help="Run full analysis (all cancers, all chromosomes)")
    parser.add_argument("--skip-tcga", action="store_true", default=False,
                        help="Skip TCGA analysis")
    parser.add_argument("--skip-gnomad", action="store_true", default=False,
                        help="Skip gnomAD analysis")
    parser.add_argument("--skip-exome", action="store_true", default=False,
                        help="Skip exome editability")
    args = parser.parse_args()

    if not args.pilot and not args.full:
        args.pilot = True  # default to pilot

    logger.info("=" * 70)
    logger.info("TCGA/gnomAD Editability Analysis")
    logger.info(f"Mode: {'PILOT (chr22, blca only)' if args.pilot else 'FULL (all cancers)'}")
    logger.info("=" * 70)

    # --- Train MotifOnly model ---
    model = train_motif_model()

    # --- Load genome ---
    logger.info(f"Loading hg19 genome from {HG19_FA}...")
    genome = Fasta(str(HG19_FA))

    # --- Build exon-by-gene lookup ---
    chrom_filter = "chr22" if args.pilot else None
    all_exons = parse_refgene_exons(chrom_filter=chrom_filter)
    exons_by_gene = defaultdict(list)
    for ex in all_exons:
        exons_by_gene[ex[4]].append(ex)  # gene name -> list of exon tuples

    all_results = {}

    # === Analysis 3: Exome Editability (run first - it's a foundation) ===
    if not args.skip_exome:
        exome_results = run_exome_editability(genome, model, chrom="chr22")
        all_results["exome_editability"] = exome_results

    # === Analysis 2: gnomAD Constraint ===
    if not args.skip_gnomad:
        gnomad_results = run_gnomad_analysis()
        all_results["gnomad_constraint"] = gnomad_results

        # Save gnomAD results
        with open(OUTPUT_DIR / "gnomad_constraint_results.json", "w") as f:
            json.dump(gnomad_results, f, indent=2)
        logger.info(f"Saved: {OUTPUT_DIR / 'gnomad_constraint_results.json'}")

    # === Analysis 1: TCGA ===
    if not args.skip_tcga:
        if args.pilot:
            # Pilot: bladder cancer, chr22 only
            logger.info("\n*** PILOT: bladder cancer, chr22 ***")
            # Need exons for all chromosomes for matched controls - but filter mutations to chr22
            if chrom_filter:
                # Re-parse for all exons in the genome for the gene matching
                all_exons_full = parse_refgene_exons(chrom_filter=None)
                exons_by_gene_full = defaultdict(list)
                for ex in all_exons_full:
                    exons_by_gene_full[ex[4]].append(ex)
            else:
                exons_by_gene_full = exons_by_gene

            tcga_results = run_tcga_analysis(
                ["blca"], genome, model, chrom_filter="chr22",
                exons_by_gene=exons_by_gene_full)

            all_results["tcga_pilot"] = tcga_results
            plot_tcga_results(tcga_results, suffix="_pilot")

            with open(OUTPUT_DIR / "tcga_pilot_results.json", "w") as f:
                json.dump(tcga_results, f, indent=2)
            logger.info(f"Saved: {OUTPUT_DIR / 'tcga_pilot_results.json'}")

            # If pilot looks good, run full
            if tcga_results:
                first_ct = list(tcga_results.values())[0]
                logger.info(f"\n*** PILOT SUMMARY ***")
                logger.info(f"Mutations scored: {first_ct['n_scored_mutations']}")
                logger.info(f"Mean editability (mut vs ctrl): "
                            f"{first_ct['mean_editability_mutations']:.4f} vs "
                            f"{first_ct['mean_editability_controls']:.4f}")
                logger.info(f"Fisher OR: {first_ct['fisher_odds_ratio']:.3f}, "
                            f"p={first_ct['fisher_p']:.2e}")

        if args.full or (args.pilot and input if False else True):
            # Full: all cancer types, all chromosomes
            logger.info("\n*** FULL ANALYSIS: all 5 cancer types ***")

            # Re-parse exons for full genome if we had a filter
            all_exons_full = parse_refgene_exons(chrom_filter=None)
            exons_by_gene_full = defaultdict(list)
            for ex in all_exons_full:
                exons_by_gene_full[ex[4]].append(ex)

            tcga_results_full = run_tcga_analysis(
                list(CANCER_STUDIES.keys()), genome, model, chrom_filter=None,
                exons_by_gene=exons_by_gene_full)

            all_results["tcga_full"] = tcga_results_full
            plot_tcga_results(tcga_results_full, suffix="_full")

            with open(OUTPUT_DIR / "tcga_full_results.json", "w") as f:
                json.dump(tcga_results_full, f, indent=2)
            logger.info(f"Saved: {OUTPUT_DIR / 'tcga_full_results.json'}")

    # === Final Summary ===
    logger.info("\n" + "=" * 70)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 70)

    if "exome_editability" in all_results and all_results["exome_editability"]:
        er = all_results["exome_editability"]
        logger.info(f"Exome editability (chr22): {er['n_positions']:,} positions, "
                    f"mean={er['mean_editability']:.4f}, "
                    f">{0.5}: {er['frac_above_0.5']*100:.1f}%")

    if "gnomad_constraint" in all_results:
        gr = all_results["gnomad_constraint"]
        logger.info(f"gnomAD: {gr['n_genes_with_editing']} editing genes, "
                    f"{gr['n_genes_without_editing']} non-editing genes")
        for test_name, test_res in gr.get("tests", {}).items():
            if "mann_whitney_p" in test_res:
                logger.info(f"  {test_name}: p={test_res['mann_whitney_p']:.2e}")
            elif "p" in test_res:
                logger.info(f"  {test_name}: r={test_res.get('r', 'N/A')}, p={test_res['p']:.2e}")

    for key in ["tcga_pilot", "tcga_full"]:
        if key in all_results:
            logger.info(f"\n{key}:")
            for ct, res in all_results[key].items():
                apobec_label = "APOBEC-high" if ct in APOBEC_HIGH else "APOBEC-low"
                logger.info(f"  {ct.upper()} ({apobec_label}): OR={res['fisher_odds_ratio']:.3f}, "
                            f"p={res['fisher_p']:.2e}, "
                            f"delta={res['delta_editability']:.4f}")

    logger.info(f"\nAll outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
