#!/usr/bin/env python3
"""Tissue-specific extended motif discovery for APOBEC3A C-to-U editing sites.

Discovers whether different tissue modules have distinct extended sequence motifs
beyond the known TC dinucleotide context. Analyses include:
  1. Dominant tissue assignment for each Levanon site
  2. Extended motif extraction and sequence logos per tissue module
  3. Position-specific nucleotide frequency comparison across tissue groups
  4. Trinucleotide/pentanucleotide context distributions
  5. Chi-squared tests for tissue-specific positional preferences
  6. Information content heatmap (position x tissue module)
  7. Pairwise motif similarity between tissue modules

Output: publication-quality PNGs in experiments/apobec3a/outputs/tissue_motifs/
"""

import json
import logging
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from scipy import stats

import logomaker

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
GTEX_CSV = DATA_DIR / "advisor" / "t1_gtex_editing_&_conservation.csv"
SPLITS_CSV = DATA_DIR / "splits_expanded.csv"
LABELS_CSV = DATA_DIR / "editing_sites_labels.csv"
GENOME_FA = PROJECT_ROOT / "data" / "raw" / "genomes" / "hg38.fa"
REFGENE_TXT = PROJECT_ROOT / "data" / "raw" / "genomes" / "refGene.txt"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "apobec" / "outputs" / "tissue_motifs"

FLANK = 10  # Nucleotides on each side of edit position for motif analysis

NUCLEOTIDES = ["A", "C", "G", "U"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tissue module definitions
# ---------------------------------------------------------------------------

TISSUE_MODULES = {
    "Brain": [
        "Brain_Amygdala", "Brain_Anterior_cingulate_cortex_BA24",
        "Brain_Caudate_basal_ganglia", "Brain_Cerebellar_Hemisphere",
        "Brain_Cerebellum", "Brain_Cortex", "Brain_Frontal_Cortex_BA9",
        "Brain_Hippocampus", "Brain_Hypothalamus",
        "Brain_Nucleus_accumbens_basal_ganglia",
        "Brain_Putamen_basal_ganglia", "Brain_Spinal_cord_cervical_c-1",
        "Brain_Substantia_nigra",
    ],
    "Blood_Immune": [
        "Whole_Blood", "Spleen", "Cells_EBV-transformed_lymphocytes",
    ],
    "Reproductive": [
        "Testis", "Ovary", "Prostate", "Uterus", "Vagina",
        "Cervix_Ectocervix", "Cervix_Endocervix", "Breast_Mammary_Tissue",
        "Fallopian_Tube",
    ],
    "GI": [
        "Colon_Sigmoid", "Colon_Transverse",
        "Esophagus_Gastroesophageal_Junction", "Esophagus_Mucosa",
        "Esophagus_Muscularis", "Stomach", "Liver", "Pancreas",
        "Small_Intestine_Terminal_Ileum",
    ],
    "Kidney": [
        "Kidney_Cortex", "Kidney_Medulla",
    ],
    "Cardiovascular": [
        "Artery_Aorta", "Artery_Coronary", "Artery_Tibial",
        "Heart_Atrial_Appendage", "Heart_Left_Ventricle",
    ],
    "Other": [
        "Adrenal_Gland", "Bladder", "Thyroid", "Nerve_Tibial",
        "Pituitary", "Minor_Salivary_Gland", "Lung", "Muscle_Skeletal",
        "Adipose_Subcutaneous", "Adipose_Visceral_Omentum",
        "Cells_Cultured_fibroblasts",
        "Skin_Not_Sun_Exposed_Suprapubic", "Skin_Sun_Exposed_Lower_leg",
    ],
}

MODULE_COLORS = {
    "Brain": "#E74C3C",
    "Blood_Immune": "#3498DB",
    "Reproductive": "#9B59B6",
    "GI": "#2ECC71",
    "Kidney": "#F39C12",
    "Cardiovascular": "#E67E22",
    "Other": "#95A5A6",
}

# Reverse map
TISSUE_TO_MODULE = {}
for mod, tissues in TISSUE_MODULES.items():
    for t in tissues:
        TISSUE_TO_MODULE[t] = mod


# ---------------------------------------------------------------------------
# Genome-based sequence extraction
# ---------------------------------------------------------------------------

def build_gene_strand_map() -> dict:
    """Build gene_name -> strand map from refGene.txt.

    Returns dict mapping gene name to strand ('+' or '-').
    If a gene appears multiple times, use the majority strand.
    """
    logger.info("Building gene strand map from %s", REFGENE_TXT)
    strand_votes = defaultdict(Counter)

    with open(REFGENE_TXT) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 13:
                continue
            strand = parts[3]
            gene = parts[12]
            strand_votes[gene][strand] += 1

    strand_map = {}
    for gene, votes in strand_votes.items():
        strand_map[gene] = votes.most_common(1)[0][0]

    logger.info("  %d genes with strand info", len(strand_map))
    return strand_map


def revcomp_dna(seq: str) -> str:
    """Reverse complement a DNA sequence."""
    comp = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
    return "".join(comp.get(b, "N") for b in reversed(seq.upper()))


def find_edit_c_position(genome, chrom: str, approx_pos: int,
                         search_radius: int = 5) -> tuple:
    """Find the actual edited C position near an approximate BED coordinate.

    For APOBEC3A C-to-U editing, the target C is typically in a TC context.
    This function searches within +/- search_radius of the approximate position
    for a C in TC context (+ strand) or G in GA context (- strand).

    Returns (actual_pos, strand) or (None, None) if not found.
    Prefers positions closest to approx_pos.
    """
    if chrom not in genome:
        return None, None

    chrom_len = len(genome[chrom])

    # Get a window around the approximate position
    w_start = max(0, approx_pos - search_radius - 1)
    w_end = min(chrom_len, approx_pos + search_radius + 2)
    window = str(genome[chrom][w_start:w_end]).upper()

    candidates = []

    for offset in range(-(search_radius), search_radius + 1):
        gpos = approx_pos + offset
        idx = gpos - w_start

        # Check TC on + strand (C at gpos, T at gpos-1)
        if 0 < idx < len(window) and window[idx] == "C" and window[idx - 1] == "T":
            candidates.append((abs(offset), gpos, "+"))

        # Check GA on + strand -> TC on - strand (G at gpos, A at gpos+1)
        if 0 <= idx < len(window) - 1 and window[idx] == "G" and window[idx + 1] == "A":
            candidates.append((abs(offset), gpos, "-"))

    if not candidates:
        # Fallback: look for any C (+ strand) or G (- strand) without TC/GA context
        for offset in range(-search_radius, search_radius + 1):
            gpos = approx_pos + offset
            idx = gpos - w_start
            if 0 <= idx < len(window):
                if window[idx] == "C":
                    candidates.append((abs(offset) + 10, gpos, "+"))  # Lower priority
                elif window[idx] == "G":
                    candidates.append((abs(offset) + 10, gpos, "-"))

    if not candidates:
        return None, None

    # Sort by distance from approx_pos (prefer TC/GA context = lower priority value)
    candidates.sort()
    return candidates[0][1], candidates[0][2]


def extract_flanking_from_genome(
    genome, chrom: str, pos: int, strand: str, flank: int = FLANK
) -> str:
    """Extract flanking RNA sequence centered on position.

    On + strand: extract directly, convert T->U
    On - strand: extract, reverse complement, convert T->U

    Returns uppercase RNA sequence of length 2*flank+1 with C at center.
    """
    if chrom not in genome:
        return ""

    chrom_len = len(genome[chrom])
    g_start = pos - flank
    g_end = pos + flank + 1

    if g_start < 0 or g_end > chrom_len:
        return ""

    dna_seq = str(genome[chrom][g_start:g_end]).upper()
    if len(dna_seq) != 2 * flank + 1:
        return ""

    if strand == "-":
        dna_seq = revcomp_dna(dna_seq)

    rna_seq = dna_seq.replace("T", "U")
    return rna_seq


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def parse_tissue_rate(cell_value) -> float:
    """Extract editing rate from 'count;total;rate' format."""
    if pd.isna(cell_value):
        return np.nan
    s = str(cell_value).strip()
    if not s:
        return np.nan
    parts = s.split(";")
    if len(parts) == 3:
        try:
            return float(parts[2])
        except ValueError:
            return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def load_data():
    """Load and parse all data sources."""
    from pyfaidx import Fasta

    logger.info("Loading GTEx editing rates from %s", GTEX_CSV)
    gtex = pd.read_csv(GTEX_CSV)
    logger.info("  %d sites, %d columns", *gtex.shape)

    # Identify all tissue columns in modules
    all_module_tissues = set()
    for tissues in TISSUE_MODULES.values():
        all_module_tissues.update(tissues)
    tissue_cols = [c for c in gtex.columns if c in all_module_tissues]
    logger.info("  Found %d tissue columns in modules", len(tissue_cols))

    # Map GTEx rows to site_ids and gene names
    labels = pd.read_csv(LABELS_CSV)
    merged = pd.merge(
        gtex, labels[["site_id", "chr", "start", "end", "gene_name"]],
        left_on=["Chr", "Start", "End"],
        right_on=["chr", "start", "end"],
        how="left",
    )
    n_mapped = merged["site_id"].notna().sum()
    logger.info("  Mapped %d/%d sites to site_ids", n_mapped, len(merged))

    # Parse tissue editing rates
    logger.info("Parsing tissue editing rates...")
    rate_cols = {}
    for tc in tissue_cols:
        col_name = f"rate_{tc}"
        merged[col_name] = merged[tc].apply(parse_tissue_rate)
        rate_cols[tc] = col_name

    # Load genome
    logger.info("Loading genome from %s", GENOME_FA)
    genome = Fasta(str(GENOME_FA))

    # Extract sequences by finding the actual edited C near each BED coordinate.
    # The BED coordinates may not point exactly to the C (off by a few bases),
    # so we search within +-5 bp for a C in TC context.
    logger.info("Finding edit C positions and extracting flanking sequences...")
    merged = merged[merged["site_id"].notna()].copy()
    sequences = {}
    n_tc_found = 0
    n_fallback = 0
    n_failed = 0

    for _, row in merged.iterrows():
        sid = row["site_id"]
        chrom = row["Chr"]
        approx_pos = int(row["Start"])

        # Find the actual edited C position
        actual_pos, strand = find_edit_c_position(
            genome, chrom, approx_pos, search_radius=5
        )

        if actual_pos is None:
            n_failed += 1
            continue

        seq = extract_flanking_from_genome(genome, chrom, actual_pos, strand, flank=FLANK)

        if not seq or len(seq) != 2 * FLANK + 1:
            n_failed += 1
            continue

        # Verify center is C
        if seq[FLANK] == "C":
            sequences[sid] = seq
            offset = actual_pos - approx_pos
            if abs(offset) <= 1:
                n_tc_found += 1
            else:
                n_fallback += 1
        else:
            n_failed += 1

    logger.info("  TC context found near coordinate: %d", n_tc_found)
    logger.info("  TC context found with offset: %d", n_fallback)
    logger.info("  No C found: %d", n_failed)
    logger.info("  Total valid sequences (C at center): %d", len(sequences))

    return merged, sequences, tissue_cols, rate_cols


# ---------------------------------------------------------------------------
# Analysis 1: Dominant tissue module assignment
# ---------------------------------------------------------------------------

def assign_dominant_module(df: pd.DataFrame, tissue_cols: list,
                           rate_cols: dict) -> pd.DataFrame:
    """Assign each site to its dominant (highest mean rate) tissue module."""
    logger.info("Assigning dominant tissue modules...")

    module_rates = {}
    for mod, tissues in TISSUE_MODULES.items():
        mod_tissue_cols = [rate_cols[t] for t in tissues if t in rate_cols]
        if mod_tissue_cols:
            module_rates[mod] = df[mod_tissue_cols].mean(axis=1)

    module_rate_df = pd.DataFrame(module_rates)
    df = df.copy()
    df["dominant_module"] = module_rate_df.idxmax(axis=1)
    df["dominant_module_rate"] = module_rate_df.max(axis=1)

    # Also store per-module mean rates for later use
    for mod in module_rates:
        df[f"module_rate_{mod}"] = module_rates[mod]

    counts = df["dominant_module"].value_counts()
    logger.info("  Dominant module assignment:")
    for mod, cnt in counts.items():
        logger.info("    %s: %d sites", mod, cnt)

    return df


# ---------------------------------------------------------------------------
# Analysis 2: Extract flanking sequences and build motif matrices
# ---------------------------------------------------------------------------

def extract_flanking_motifs(df: pd.DataFrame, sequences: dict) -> dict:
    """Extract flanking motif sequences for each tissue module.

    For each module, takes the top 50% of sites by rate in that module.
    Only includes sites where the center nucleotide is C (proper C-to-U context).

    Returns dict: module -> list of (2*FLANK+1)-length sequences.
    """
    logger.info("Extracting flanking motifs (flank=%d)...", FLANK)
    module_seqs = defaultdict(list)

    for mod in TISSUE_MODULES:
        # Get sites where this module is dominant
        mod_sites = df[df["dominant_module"] == mod].copy()
        if mod_sites.empty:
            continue

        # Top 50% by rate in that module
        rate_col = f"module_rate_{mod}"
        if rate_col in mod_sites.columns:
            threshold = mod_sites[rate_col].median()
            top_sites = mod_sites[mod_sites[rate_col] >= threshold]
        else:
            top_sites = mod_sites

        for _, row in top_sites.iterrows():
            sid = row["site_id"]
            if sid not in sequences:
                continue
            seq = sequences[sid]
            if len(seq) != 2 * FLANK + 1:
                continue
            if "N" in seq:
                continue
            # Only include if center is C (properly oriented C-to-U site)
            if seq[FLANK] != "C":
                continue
            module_seqs[mod].append(seq)

    for mod, seqlist in module_seqs.items():
        logger.info("  %s: %d valid motif sequences (C at center)", mod, len(seqlist))

    # Also build an "All Sites" combined set
    all_seqs = []
    for seqlist in module_seqs.values():
        all_seqs.extend(seqlist)
    logger.info("  Total valid sequences across all modules: %d", len(all_seqs))

    return dict(module_seqs)


def build_frequency_matrix(seqlist: list) -> pd.DataFrame:
    """Build position frequency matrix from list of equal-length sequences.

    Returns DataFrame with positions as rows and nucleotides as columns.
    """
    n = len(seqlist)
    if n == 0:
        return pd.DataFrame()
    seq_len = len(seqlist[0])
    half = seq_len // 2
    positions = list(range(-half, half + 1))

    counts = {nuc: np.zeros(seq_len) for nuc in NUCLEOTIDES}
    for seq in seqlist:
        for i, nt in enumerate(seq):
            if nt in counts:
                counts[nt][i] += 1

    freq = {nuc: counts[nuc] / n for nuc in NUCLEOTIDES}
    freq_df = pd.DataFrame(freq, index=positions)
    return freq_df


def build_info_matrix(freq_df: pd.DataFrame) -> pd.DataFrame:
    """Convert frequency matrix to information content matrix.

    IC(pos) = log2(4) - H(pos) where H = -sum(p * log2(p)).
    """
    if freq_df.empty:
        return pd.DataFrame()

    info_df = freq_df.copy()
    for pos in freq_df.index:
        row = freq_df.loc[pos]
        probs = row.values
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log2(probs))
        ic = np.log2(4) - entropy

        for nuc in NUCLEOTIDES:
            info_df.loc[pos, nuc] = ic * freq_df.loc[pos, nuc]

    return info_df


# ---------------------------------------------------------------------------
# Analysis 3 & 5: Position-specific frequency comparison
# ---------------------------------------------------------------------------

def positional_frequency_analysis(module_seqs: dict) -> pd.DataFrame:
    """Compare nucleotide frequencies at each position across tissue modules.

    Returns a DataFrame with chi-squared p-values for each position.
    """
    logger.info("Running positional frequency analysis...")
    modules = sorted(module_seqs.keys())
    if not modules:
        return pd.DataFrame()

    seq_len = len(list(module_seqs.values())[0][0])
    half = seq_len // 2
    positions = list(range(-half, half + 1))

    results = []
    for pos_idx, pos in enumerate(positions):
        contingency = np.zeros((len(modules), 4), dtype=int)
        for mod_idx, mod in enumerate(modules):
            for seq in module_seqs[mod]:
                nt = seq[pos_idx]
                if nt in NUCLEOTIDES:
                    contingency[mod_idx, NUCLEOTIDES.index(nt)] += 1

        # Remove columns with all zeros
        col_sums = contingency.sum(axis=0)
        valid_cols = col_sums > 0
        contingency_valid = contingency[:, valid_cols]

        # Remove rows (modules) with zero total
        row_sums = contingency_valid.sum(axis=1)
        valid_rows = row_sums > 0
        contingency_valid = contingency_valid[valid_rows, :]

        if contingency_valid.shape[1] < 2 or contingency_valid.shape[0] < 2:
            results.append({
                "position": pos, "chi2": np.nan,
                "p_value": 1.0, "cramers_v": 0.0,
            })
            continue

        try:
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency_valid)
            n_total = contingency_valid.sum()
            k = min(contingency_valid.shape) - 1
            cramers_v = np.sqrt(chi2 / (n_total * k)) if k > 0 and n_total > 0 else 0.0
        except ValueError:
            chi2, p_value, cramers_v = np.nan, 1.0, 0.0

        results.append({
            "position": pos, "chi2": chi2,
            "p_value": p_value, "cramers_v": cramers_v,
        })

    results_df = pd.DataFrame(results)
    n_tests = len(results_df)
    results_df["p_bonferroni"] = np.minimum(results_df["p_value"] * n_tests, 1.0)

    sig = results_df[results_df["p_bonferroni"] < 0.05]
    logger.info("  %d/%d positions significant (Bonferroni p<0.05)", len(sig), n_tests)
    if not sig.empty:
        for _, row in sig.iterrows():
            logger.info("    Position %+d: chi2=%.1f, p=%.2e, V=%.3f",
                        row["position"], row["chi2"], row["p_bonferroni"], row["cramers_v"])

    return results_df


# ---------------------------------------------------------------------------
# Analysis 4: Trinucleotide and pentanucleotide context
# ---------------------------------------------------------------------------

def kmer_analysis(module_seqs: dict, k: int = 3) -> dict:
    """Compare k-mer distributions centered on the edit site across modules."""
    logger.info("Running %d-mer analysis...", k)
    module_kmers = {}
    half = k // 2

    for mod, seqlist in module_seqs.items():
        kmer_counts = Counter()
        for seq in seqlist:
            center = len(seq) // 2
            kmer = seq[center - half: center + half + 1]
            if len(kmer) == k:
                kmer_counts[kmer] += 1
        module_kmers[mod] = kmer_counts
        top3 = kmer_counts.most_common(3)
        logger.info("  %s: %d unique %d-mers, top=%s", mod, len(kmer_counts), k,
                     ", ".join(f"{km}({c})" for km, c in top3))

    return module_kmers


def kmer_comparison_test(module_kmers: dict, k: int) -> pd.DataFrame:
    """Chi-squared test comparing k-mer distributions across module pairs."""
    modules = sorted(module_kmers.keys())
    all_kmers = sorted(set().union(*[set(v.keys()) for v in module_kmers.values()]))

    results = []
    for m1, m2 in combinations(modules, 2):
        counts1 = np.array([module_kmers[m1].get(km, 0) for km in all_kmers])
        counts2 = np.array([module_kmers[m2].get(km, 0) for km in all_kmers])

        mask = (counts1 + counts2) > 0
        if mask.sum() < 2:
            continue

        contingency = np.array([counts1[mask], counts2[mask]])
        try:
            chi2, p_value, dof, _ = stats.chi2_contingency(contingency)
            results.append({
                "module_1": m1, "module_2": m2,
                "chi2": chi2, "p_value": p_value, "dof": dof,
                f"{k}mer_count": int(mask.sum()),
            })
        except ValueError:
            pass

    return pd.DataFrame(results) if results else pd.DataFrame()


# ---------------------------------------------------------------------------
# Analysis 7: Pairwise motif similarity
# ---------------------------------------------------------------------------

def motif_similarity(module_seqs: dict) -> pd.DataFrame:
    """Compute pairwise Jensen-Shannon divergence between tissue module motifs."""
    logger.info("Computing pairwise motif similarity...")
    modules = sorted(module_seqs.keys())

    freq_matrices = {}
    for mod in modules:
        freq_matrices[mod] = build_frequency_matrix(module_seqs[mod])

    n_mod = len(modules)
    jsd_matrix = np.zeros((n_mod, n_mod))

    for i in range(n_mod):
        for j in range(i + 1, n_mod):
            m1, m2 = modules[i], modules[j]
            f1 = freq_matrices[m1].values.flatten()
            f2 = freq_matrices[m2].values.flatten()

            # Per-position JSD (average JSD across positions)
            seq_len = freq_matrices[m1].shape[0]
            pos_jsds = []
            for p in range(seq_len):
                p1 = freq_matrices[m1].iloc[p].values
                p2 = freq_matrices[m2].iloc[p].values
                m_dist = 0.5 * (p1 + p2)
                eps = 1e-10
                p1_safe = np.maximum(p1, eps)
                p2_safe = np.maximum(p2, eps)
                m_safe = np.maximum(m_dist, eps)
                kl1 = np.sum(np.where(p1 > eps, p1_safe * np.log2(p1_safe / m_safe), 0))
                kl2 = np.sum(np.where(p2 > eps, p2_safe * np.log2(p2_safe / m_safe), 0))
                pos_jsds.append(0.5 * (kl1 + kl2))

            jsd = np.mean(pos_jsds)
            jsd_matrix[i, j] = jsd
            jsd_matrix[j, i] = jsd

    jsd_df = pd.DataFrame(jsd_matrix, index=modules, columns=modules)

    # Report most/least similar pairs
    jsd_offdiag = jsd_matrix.copy()
    np.fill_diagonal(jsd_offdiag, np.inf)
    min_idx = np.unravel_index(jsd_offdiag.argmin(), jsd_offdiag.shape)
    logger.info("  Most similar: %s <-> %s (mean JSD=%.4f)",
                modules[min_idx[0]], modules[min_idx[1]], jsd_offdiag[min_idx])

    np.fill_diagonal(jsd_offdiag, -np.inf)
    max_idx = np.unravel_index(jsd_offdiag.argmax(), jsd_offdiag.shape)
    logger.info("  Most different: %s <-> %s (mean JSD=%.4f)",
                modules[max_idx[0]], modules[max_idx[1]], jsd_offdiag[max_idx])

    return jsd_df


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------

def plot_sequence_logo(info_df: pd.DataFrame, title: str, output_path: Path,
                       color_scheme: str = "classic"):
    """Create a publication-quality sequence logo."""
    if info_df.empty:
        return

    fig, ax = plt.subplots(1, 1, figsize=(12, 2.5))
    logomaker.Logo(info_df, ax=ax, color_scheme=color_scheme,
                   font_name="DejaVu Sans")

    ax.set_ylabel("Information\n(bits)", fontsize=10, fontweight="bold")
    ax.set_xlabel("Position relative to edit site", fontsize=10, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_ylim(0, max(info_df.sum(axis=1).max() * 1.1, 0.5))

    # Mark edit position
    ax.axvline(x=0, color="red", linestyle="--", alpha=0.5, linewidth=1)
    ax.text(0, ax.get_ylim()[1] * 0.95, "C", ha="center", va="top",
            fontsize=8, color="red", fontstyle="italic")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: %s", output_path.name)


def plot_all_logos(module_seqs: dict, output_dir: Path):
    """Generate sequence logos for all tissue modules and a combined comparison."""
    logger.info("Generating sequence logos...")

    # Individual logos
    for mod in sorted(module_seqs.keys()):
        seqlist = module_seqs[mod]
        if not seqlist:
            continue
        freq_df = build_frequency_matrix(seqlist)
        info_df = build_info_matrix(freq_df)
        n = len(seqlist)
        plot_sequence_logo(info_df, f"{mod} module (n={n})",
                           output_dir / f"logo_{mod}.png")

    # Combined comparison figure
    modules_with_data = [m for m in sorted(module_seqs.keys()) if module_seqs[m]]
    n_modules = len(modules_with_data)
    if n_modules == 0:
        return

    fig, axes = plt.subplots(n_modules, 1, figsize=(14, 2.8 * n_modules))
    if n_modules == 1:
        axes = [axes]

    for idx, mod in enumerate(modules_with_data):
        seqlist = module_seqs[mod]
        freq_df = build_frequency_matrix(seqlist)
        info_df = build_info_matrix(freq_df)

        logomaker.Logo(info_df, ax=axes[idx], color_scheme="classic",
                       font_name="DejaVu Sans")

        axes[idx].set_ylabel("Bits", fontsize=9)
        color = MODULE_COLORS.get(mod, "#333333")
        axes[idx].set_title(f"{mod} (n={len(seqlist)})", fontsize=11,
                            fontweight="bold", color=color, loc="left")
        axes[idx].set_ylim(0, max(info_df.sum(axis=1).max() * 1.1, 0.5))
        axes[idx].axvline(x=0, color="red", linestyle="--", alpha=0.4, linewidth=0.8)

        if idx < n_modules - 1:
            axes[idx].set_xlabel("")
        else:
            axes[idx].set_xlabel("Position relative to edit site", fontsize=10)

    plt.suptitle("Tissue-Module Sequence Logos: Extended Motif Context around C-to-U Edit",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(output_dir / "logos_all_modules_comparison.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: logos_all_modules_comparison.png")


def plot_frequency_heatmaps(module_seqs: dict, output_dir: Path):
    """Plot nucleotide frequency deviation from overall average per module."""
    logger.info("Generating frequency heatmaps...")

    modules = [m for m in sorted(module_seqs.keys()) if module_seqs[m]]
    if not modules:
        return

    # Overall frequency
    all_seqs = []
    for mod in modules:
        all_seqs.extend(module_seqs[mod])
    overall_freq = build_frequency_matrix(all_seqs)

    fig, axes = plt.subplots(len(modules), 1, figsize=(14, 2.2 * len(modules)),
                             sharex=True)
    if len(modules) == 1:
        axes = [axes]

    last_im = None
    for idx, mod in enumerate(modules):
        freq_df = build_frequency_matrix(module_seqs[mod])
        delta = freq_df - overall_freq

        positions = delta.index.values
        nucs = delta.columns.values
        data = delta.values.T

        im = axes[idx].imshow(data, aspect="auto", cmap="RdBu_r",
                              vmin=-0.15, vmax=0.15,
                              extent=[positions[0] - 0.5, positions[-1] + 0.5,
                                      -0.5, 3.5])
        axes[idx].set_yticks(range(4))
        axes[idx].set_yticklabels(nucs, fontsize=9)
        color = MODULE_COLORS.get(mod, "#333333")
        axes[idx].set_title(f"{mod} (n={len(module_seqs[mod])})", fontsize=10,
                            fontweight="bold", color=color, loc="left")
        axes[idx].axvline(x=0, color="black", linestyle="--", alpha=0.5, linewidth=0.8)
        last_im = im

    axes[-1].set_xlabel("Position relative to edit site", fontsize=10)
    plt.suptitle("Nucleotide Frequency Deviation from Overall Average",
                 fontsize=13, fontweight="bold")

    if last_im:
        cbar = fig.colorbar(last_im, ax=axes, shrink=0.5, pad=0.02)
        cbar.set_label("Frequency difference", fontsize=9)

    plt.tight_layout()
    fig.savefig(output_dir / "frequency_delta_heatmap.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: frequency_delta_heatmap.png")


def plot_information_content_heatmap(module_seqs: dict, output_dir: Path):
    """Heatmap of information content per position per tissue module."""
    logger.info("Generating information content heatmap...")

    modules = [m for m in sorted(module_seqs.keys()) if module_seqs[m]]
    if not modules:
        return

    seq_len = len(list(module_seqs[modules[0]])[0])
    half = seq_len // 2
    positions = list(range(-half, half + 1))

    ic_matrix = np.zeros((len(modules), len(positions)))

    for mod_idx, mod in enumerate(modules):
        freq_df = build_frequency_matrix(module_seqs[mod])
        for pos_idx, pos in enumerate(positions):
            probs = freq_df.loc[pos].values
            probs = probs[probs > 0]
            entropy = -np.sum(probs * np.log2(probs))
            ic_matrix[mod_idx, pos_idx] = np.log2(4) - entropy

    fig, ax = plt.subplots(1, 1, figsize=(14, 4))

    cmap = LinearSegmentedColormap.from_list("ic_cmap",
                                              ["#FFFFFF", "#FFF3CD", "#F39C12", "#C0392B"])
    im = ax.imshow(ic_matrix, aspect="auto", cmap=cmap, vmin=0,
                   vmax=max(ic_matrix.max(), 0.3),
                   extent=[positions[0] - 0.5, positions[-1] + 0.5,
                           -0.5, len(modules) - 0.5])

    ax.set_yticks(range(len(modules)))
    ax.set_yticklabels(modules, fontsize=10)
    ax.set_xlabel("Position relative to edit site", fontsize=11, fontweight="bold")
    ax.set_title("Information Content by Tissue Module and Position",
                 fontsize=13, fontweight="bold")
    ax.axvline(x=0, color="black", linestyle="--", alpha=0.6, linewidth=1)

    cbar = fig.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label("Information content (bits)", fontsize=10)

    plt.tight_layout()
    fig.savefig(output_dir / "information_content_heatmap.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: information_content_heatmap.png")


def plot_chi2_results(results_df: pd.DataFrame, output_dir: Path):
    """Plot chi-squared test results across positions."""
    if results_df.empty:
        return

    logger.info("Generating chi-squared results plot...")

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    positions = results_df["position"].values
    neg_log_p = -np.log10(results_df["p_bonferroni"].clip(lower=1e-50).values)
    cramers = results_df["cramers_v"].values

    # -log10(p) with significance threshold
    axes[0].bar(positions, neg_log_p, width=0.8, color="#3498DB", alpha=0.8,
                edgecolor="none")
    threshold = -np.log10(0.05)
    axes[0].axhline(y=threshold, color="red", linestyle="--", linewidth=1,
                     label="Bonferroni p=0.05")
    axes[0].set_ylabel("-log10(p-value)", fontsize=10, fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[0].axvline(x=0, color="gray", linestyle=":", alpha=0.5)

    # Cramer's V
    axes[1].bar(positions, cramers, width=0.8, color="#E74C3C", alpha=0.8,
                edgecolor="none")
    axes[1].set_ylabel("Cramer's V", fontsize=10, fontweight="bold")
    axes[1].set_xlabel("Position relative to edit site", fontsize=10, fontweight="bold")
    axes[1].axvline(x=0, color="gray", linestyle=":", alpha=0.5)

    plt.suptitle("Position-Specific Tissue Differences in Nucleotide Frequency",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_dir / "chi2_positional_test.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: chi2_positional_test.png")


def plot_kmer_comparison(module_kmers_3: dict, module_kmers_5: dict,
                         output_dir: Path):
    """Plot trinucleotide and pentanucleotide distributions per module."""
    logger.info("Generating k-mer comparison plots...")

    for k, module_kmers in [(3, module_kmers_3), (5, module_kmers_5)]:
        modules = [m for m in sorted(module_kmers.keys()) if module_kmers[m]]
        if not modules:
            continue

        total = Counter()
        for km in module_kmers.values():
            total.update(km)
        top_kmers = [kmer for kmer, _ in total.most_common(20)]

        data = np.zeros((len(modules), len(top_kmers)))
        for mod_idx, mod in enumerate(modules):
            total_count = sum(module_kmers[mod].values())
            for kmer_idx, kmer in enumerate(top_kmers):
                data[mod_idx, kmer_idx] = module_kmers[mod].get(kmer, 0) / max(total_count, 1)

        fig, ax = plt.subplots(1, 1, figsize=(max(10, len(top_kmers) * 0.7), 5))

        x = np.arange(len(top_kmers))
        width = 0.8 / len(modules)

        for mod_idx, mod in enumerate(modules):
            offset = (mod_idx - len(modules) / 2 + 0.5) * width
            color = MODULE_COLORS.get(mod, "#333333")
            ax.bar(x + offset, data[mod_idx], width=width, color=color,
                   alpha=0.85, label=mod, edgecolor="white", linewidth=0.3)

        ax.set_xticks(x)
        ax.set_xticklabels(top_kmers, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Frequency", fontsize=10, fontweight="bold")
        ax.set_title(f"Top {len(top_kmers)} {k}-mer Contexts at Edit Site",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, ncol=3, loc="upper right")

        plt.tight_layout()
        fig.savefig(output_dir / f"kmer_{k}mer_comparison.png", dpi=300,
                    bbox_inches="tight")
        plt.close(fig)
        logger.info("  Saved: kmer_%dmer_comparison.png", k)


def plot_motif_similarity(jsd_df: pd.DataFrame, output_dir: Path):
    """Plot pairwise JSD similarity matrix as a heatmap."""
    logger.info("Generating motif similarity heatmap...")

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))

    modules = list(jsd_df.index)
    data = jsd_df.values

    max_jsd = data[data > 0].max() if (data > 0).any() else 0.01

    im = ax.imshow(data, cmap="YlOrRd", vmin=0, vmax=max(max_jsd, 0.01))

    ax.set_xticks(range(len(modules)))
    ax.set_xticklabels(modules, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(len(modules)))
    ax.set_yticklabels(modules, fontsize=10)

    for i in range(len(modules)):
        for j in range(len(modules)):
            val = data[i, j]
            color = "white" if val > max_jsd * 0.6 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")

    ax.set_title("Pairwise Motif Divergence (Jensen-Shannon)",
                 fontsize=12, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Mean JSD per position (bits)", fontsize=10)

    plt.tight_layout()
    fig.savefig(output_dir / "motif_similarity_jsd.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: motif_similarity_jsd.png")


def plot_dominant_module_distribution(df: pd.DataFrame, output_dir: Path):
    """Plot the distribution of dominant tissue modules across sites."""
    logger.info("Generating dominant module distribution plot...")

    counts = df["dominant_module"].value_counts()
    modules = [m for m in TISSUE_MODULES if m in counts.index]
    values = [counts.get(m, 0) for m in modules]
    colors = [MODULE_COLORS.get(m, "#333333") for m in modules]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    bars = axes[0].bar(range(len(modules)), values, color=colors, edgecolor="white",
                       linewidth=0.5)
    axes[0].set_xticks(range(len(modules)))
    axes[0].set_xticklabels(modules, rotation=30, ha="right", fontsize=10)
    axes[0].set_ylabel("Number of sites", fontsize=11, fontweight="bold")
    axes[0].set_title("Sites per Dominant Tissue Module", fontsize=12,
                      fontweight="bold")
    for bar, val in zip(bars, values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                     str(val), ha="center", va="bottom", fontsize=9, fontweight="bold")

    axes[1].pie(values, labels=modules, colors=colors, autopct="%1.1f%%",
                startangle=90, textprops={"fontsize": 9})
    axes[1].set_title("Proportion", fontsize=12, fontweight="bold")

    plt.suptitle("Dominant Tissue Module Assignment (636 Levanon Sites)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_dir / "dominant_module_distribution.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: dominant_module_distribution.png")


def plot_summary_figure(module_seqs: dict, chi2_df: pd.DataFrame,
                        jsd_df: pd.DataFrame, output_dir: Path):
    """Create a multi-panel summary figure for publication."""
    logger.info("Generating summary figure...")

    modules = [m for m in sorted(module_seqs.keys()) if module_seqs[m]]
    n_mod = len(modules)
    if n_mod == 0:
        return

    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1.5, 1, 1],
                           hspace=0.35, wspace=0.25)

    # Panel A: Stacked sequence logos
    gs_logos = gridspec.GridSpecFromSubplotSpec(n_mod, 1, subplot_spec=gs[0, :],
                                                hspace=0.4)
    for idx, mod in enumerate(modules):
        ax = fig.add_subplot(gs_logos[idx])
        seqlist = module_seqs[mod]
        freq_df = build_frequency_matrix(seqlist)
        info_df = build_info_matrix(freq_df)

        logomaker.Logo(info_df, ax=ax, color_scheme="classic",
                       font_name="DejaVu Sans")
        ax.set_ylim(0, max(info_df.sum(axis=1).max() * 1.15, 0.4))
        color = MODULE_COLORS.get(mod, "#333333")
        ax.set_title(f"{mod} (n={len(seqlist)})", fontsize=9,
                     fontweight="bold", color=color, loc="left")
        ax.axvline(x=0, color="red", linestyle="--", alpha=0.4, linewidth=0.7)
        if idx < n_mod - 1:
            ax.set_xticklabels([])
            ax.set_xlabel("")
        else:
            ax.set_xlabel("Position relative to edit site", fontsize=9)
        ax.set_ylabel("Bits", fontsize=8)
        ax.tick_params(labelsize=7)

    # Panel B: Chi-squared test
    ax_chi2 = fig.add_subplot(gs[1, 0])
    if not chi2_df.empty:
        positions = chi2_df["position"].values
        neg_log_p = -np.log10(chi2_df["p_bonferroni"].clip(lower=1e-50).values)
        ax_chi2.bar(positions, neg_log_p, width=0.8, color="#3498DB", alpha=0.8)
        ax_chi2.axhline(y=-np.log10(0.05), color="red", linestyle="--", linewidth=0.8)
        ax_chi2.axvline(x=0, color="gray", linestyle=":", alpha=0.5)
    ax_chi2.set_title("B. Positional Chi-Squared Test", fontsize=11, fontweight="bold")
    ax_chi2.set_ylabel("-log10(p)", fontsize=9)
    ax_chi2.set_xlabel("Position", fontsize=9)

    # Panel C: Information content heatmap
    ax_ic = fig.add_subplot(gs[1, 1])
    seq_len = len(list(module_seqs[modules[0]])[0])
    half = seq_len // 2
    positions = list(range(-half, half + 1))

    ic_matrix = np.zeros((len(modules), len(positions)))
    for mod_idx, mod in enumerate(modules):
        freq_df = build_frequency_matrix(module_seqs[mod])
        for pos_idx, pos in enumerate(positions):
            probs = freq_df.loc[pos].values
            probs = probs[probs > 0]
            entropy = -np.sum(probs * np.log2(probs))
            ic_matrix[mod_idx, pos_idx] = np.log2(4) - entropy

    cmap = LinearSegmentedColormap.from_list("ic", ["#FFFFFF", "#FFF3CD", "#F39C12", "#C0392B"])
    im = ax_ic.imshow(ic_matrix, aspect="auto", cmap=cmap, vmin=0,
                      vmax=max(ic_matrix.max(), 0.3),
                      extent=[positions[0] - 0.5, positions[-1] + 0.5,
                              -0.5, len(modules) - 0.5])
    ax_ic.set_yticks(range(len(modules)))
    ax_ic.set_yticklabels(modules, fontsize=8)
    ax_ic.set_xlabel("Position", fontsize=9)
    ax_ic.set_title("C. Information Content", fontsize=11, fontweight="bold")
    ax_ic.axvline(x=0, color="black", linestyle="--", alpha=0.5, linewidth=0.7)
    fig.colorbar(im, ax=ax_ic, shrink=0.7, pad=0.02)

    # Panel D: JSD similarity matrix
    ax_jsd = fig.add_subplot(gs[2, 0])
    jsd_modules = [m for m in modules if m in jsd_df.index]
    jsd_data = jsd_df.loc[jsd_modules, jsd_modules].values
    max_jsd = jsd_data[jsd_data > 0].max() if (jsd_data > 0).any() else 0.01
    im2 = ax_jsd.imshow(jsd_data, cmap="YlOrRd", vmin=0, vmax=max_jsd)
    ax_jsd.set_xticks(range(len(jsd_modules)))
    ax_jsd.set_xticklabels(jsd_modules, rotation=40, ha="right", fontsize=8)
    ax_jsd.set_yticks(range(len(jsd_modules)))
    ax_jsd.set_yticklabels(jsd_modules, fontsize=8)
    for i in range(len(jsd_modules)):
        for j in range(len(jsd_modules)):
            color = "white" if jsd_data[i, j] > max_jsd * 0.5 else "black"
            ax_jsd.text(j, i, f"{jsd_data[i, j]:.3f}", ha="center", va="center",
                        fontsize=7, color=color)
    ax_jsd.set_title("D. Motif Divergence (JSD)", fontsize=11, fontweight="bold")
    fig.colorbar(im2, ax=ax_jsd, shrink=0.7, pad=0.02)

    # Panel E: Trinucleotide barplot
    ax_3mer = fig.add_subplot(gs[2, 1])
    all_3mer_counts = Counter()
    mod_3mers = {}
    for mod in modules:
        mod_3mers[mod] = Counter()
        for seq in module_seqs[mod]:
            center = len(seq) // 2
            kmer = seq[center - 1: center + 2]
            if len(kmer) == 3:
                mod_3mers[mod][kmer] += 1
                all_3mer_counts[kmer] += 1

    top_3mers = [km for km, _ in all_3mer_counts.most_common(12)]
    x = np.arange(len(top_3mers))
    width = 0.8 / max(len(modules), 1)
    for mod_idx, mod in enumerate(modules):
        total = sum(mod_3mers[mod].values())
        freqs = [mod_3mers[mod].get(km, 0) / max(total, 1) for km in top_3mers]
        offset = (mod_idx - len(modules) / 2 + 0.5) * width
        color = MODULE_COLORS.get(mod, "#333333")
        ax_3mer.bar(x + offset, freqs, width=width, color=color, alpha=0.85,
                    label=mod, edgecolor="white", linewidth=0.2)

    ax_3mer.set_xticks(x)
    ax_3mer.set_xticklabels(top_3mers, rotation=45, ha="right", fontsize=8)
    ax_3mer.set_ylabel("Frequency", fontsize=9)
    ax_3mer.set_title("E. Top 3-mer Contexts", fontsize=11, fontweight="bold")
    ax_3mer.legend(fontsize=6, ncol=3, loc="upper right")

    fig.suptitle("Tissue-Specific Extended Motif Discovery for APOBEC3A C-to-U Editing",
                 fontsize=15, fontweight="bold", y=1.01)

    fig.savefig(output_dir / "summary_figure.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: summary_figure.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Run the complete tissue-specific motif analysis pipeline."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("=" * 70)
    logger.info("Tissue-Specific Extended Motif Discovery")
    logger.info("Output directory: %s", OUTPUT_DIR)
    logger.info("=" * 70)

    # Load data (extracts sequences from genome with proper strand handling)
    df, sequences, tissue_cols, rate_cols = load_data()

    # Analysis 1: Dominant tissue module assignment
    df = assign_dominant_module(df, tissue_cols, rate_cols)
    plot_dominant_module_distribution(df, OUTPUT_DIR)

    # Analysis 2: Extract flanking motifs (only sites with C at center)
    module_seqs = extract_flanking_motifs(df, sequences)

    if not module_seqs or all(len(v) == 0 for v in module_seqs.values()):
        logger.error("No valid motif sequences extracted. Aborting.")
        sys.exit(1)

    # Remove empty modules
    module_seqs = {k: v for k, v in module_seqs.items() if v}

    # Generate sequence logos
    plot_all_logos(module_seqs, OUTPUT_DIR)

    # Analysis 3 & 5: Positional frequency comparison with chi-squared tests
    chi2_df = positional_frequency_analysis(module_seqs)
    chi2_df.to_csv(OUTPUT_DIR / "chi2_positional_results.csv", index=False)
    plot_chi2_results(chi2_df, OUTPUT_DIR)

    # Analysis 4: K-mer context
    module_kmers_3 = kmer_analysis(module_seqs, k=3)
    module_kmers_5 = kmer_analysis(module_seqs, k=5)
    kmer3_test = kmer_comparison_test(module_kmers_3, k=3)
    kmer5_test = kmer_comparison_test(module_kmers_5, k=5)
    if not kmer3_test.empty:
        kmer3_test.to_csv(OUTPUT_DIR / "kmer_3mer_pairwise_tests.csv", index=False)
    if not kmer5_test.empty:
        kmer5_test.to_csv(OUTPUT_DIR / "kmer_5mer_pairwise_tests.csv", index=False)
    plot_kmer_comparison(module_kmers_3, module_kmers_5, OUTPUT_DIR)

    # Analysis 6: Heatmaps
    plot_frequency_heatmaps(module_seqs, OUTPUT_DIR)
    plot_information_content_heatmap(module_seqs, OUTPUT_DIR)

    # Analysis 7: Motif similarity
    jsd_df = motif_similarity(module_seqs)
    jsd_df.to_csv(OUTPUT_DIR / "motif_jsd_matrix.csv")
    plot_motif_similarity(jsd_df, OUTPUT_DIR)

    # Summary figure
    plot_summary_figure(module_seqs, chi2_df, jsd_df, OUTPUT_DIR)

    # Save comprehensive results JSON
    results = {
        "n_sites_total": int(len(df)),
        "n_sites_with_c_at_center": sum(len(v) for v in module_seqs.values()),
        "module_motif_counts": {mod: len(seqs) for mod, seqs in module_seqs.items()},
        "dominant_module_distribution": {
            k: int(v) for k, v in df["dominant_module"].value_counts().items()
        },
        "chi2_significant_positions": chi2_df[chi2_df["p_bonferroni"] < 0.05][
            ["position", "chi2", "p_bonferroni", "cramers_v"]
        ].to_dict(orient="records") if not chi2_df.empty else [],
        "jsd_matrix": {str(k): {str(k2): float(v2) for k2, v2 in v.items()}
                       for k, v in jsd_df.to_dict().items()} if not jsd_df.empty else {},
        "top_3mers_per_module": {
            mod: dict(km.most_common(5)) for mod, km in module_kmers_3.items()
        },
        "top_5mers_per_module": {
            mod: dict(km.most_common(5)) for mod, km in module_kmers_5.items()
        },
    }

    with open(OUTPUT_DIR / "tissue_motif_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Saved: tissue_motif_results.json")

    logger.info("=" * 70)
    logger.info("Analysis complete. All outputs in: %s", OUTPUT_DIR)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
