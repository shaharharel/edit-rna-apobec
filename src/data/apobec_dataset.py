"""APOBEC C-to-U editing site dataset for training and evaluation.

Provides an end-to-end data pipeline that:
1. Loads positive editing sites from the unified dataset
2. Generates hard/medium/easy negative C-site samples
3. Extracts sequence windows centered on candidate C positions
4. Computes APOBEC-specific features (flanking context, structure, concordance)
5. Builds PyG graphs for the GNN branch
6. Packages everything into a PyTorch Dataset compatible with
   APOBECEditEmbedding and APOBECMultiTaskHead

The pipeline supports both pre-computed features (recommended for training)
and on-the-fly feature computation (for inference / prototyping).
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Flanking dinucleotide context (position -1 relative to target C)
FLANKING_CONTEXT_MAP = {"TC": 0, "CC": 1, "AC": 2, "GC": 3}

# APOBEC class encoding
APOBEC_CLASS_MAP = {
    "APOBEC3A Only": 0,
    "APOBEC3G Only": 1,
    "Both": 2,
    "Neither": 3,
}

# Exonic function encoding
EXONIC_FUNCTION_MAP = {
    "synonymous": 0,
    "nonsynonymous": 1,
    "stopgain": 2,
}

# Structure type encoding
STRUCTURE_TYPE_MAP = {
    "In Loop": 0,
    "dsRNA": 1,
    "ssRNA / Bulge": 2,
    "Open ssRNA": 3,
}

# Tissue specificity class encoding
TISSUE_SPEC_MAP = {
    "Blood Specific": 0,
    "Ubiquitous": 1,
    "Testis Specific": 2,
    "Non-Specific": 3,
    "Intestine Specific": 4,
}

# Structure concordance transition types
CONCORDANCE_TRANSITIONS = {
    "dsRNA_to_loop": 0,
    "loop_to_dsRNA": 1,
    "dsRNA_to_dsRNA": 2,
    "loop_to_loop": 3,
}

# Default GTEx tissue list (54 tissues)
GTEX_TISSUES = [
    "Adipose_Subcutaneous", "Adipose_Visceral_Omentum", "Adrenal_Gland",
    "Artery_Aorta", "Artery_Coronary", "Artery_Tibial", "Bladder",
    "Brain_Amygdala", "Brain_Anterior_cingulate_cortex_BA24",
    "Brain_Caudate_basal_ganglia", "Brain_Cerebellar_Hemisphere",
    "Brain_Cerebellum", "Brain_Cortex", "Brain_Frontal_Cortex_BA9",
    "Brain_Hippocampus", "Brain_Hypothalamus",
    "Brain_Nucleus_accumbens_basal_ganglia", "Brain_Putamen_basal_ganglia",
    "Brain_Spinal_cord_cervical_c-1", "Brain_Substantia_nigra",
    "Breast_Mammary_Tissue", "Cells_Cultured_fibroblasts",
    "Cells_EBV-transformed_lymphocytes", "Cervix_Ectocervix",
    "Cervix_Endocervix", "Colon_Sigmoid", "Colon_Transverse",
    "Esophagus_Gastroesophageal_Junction", "Esophagus_Mucosa",
    "Esophagus_Muscularis", "Fallopian_Tube", "Heart_Atrial_Appendage",
    "Heart_Left_Ventricle", "Kidney_Cortex", "Kidney_Medulla", "Liver",
    "Lung", "Minor_Salivary_Gland", "Muscle_Skeletal", "Nerve_Tibial",
    "Ovary", "Pancreas", "Pituitary", "Prostate",
    "Skin_Not_Sun_Exposed_Suprapubic", "Skin_Sun_Exposed_Lower_leg",
    "Small_Intestine_Terminal_Ileum", "Spleen", "Stomach", "Testis",
    "Thyroid", "Uterus", "Vagina", "Whole_Blood",
]

N_TISSUES = len(GTEX_TISSUES)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class APOBECDataConfig:
    """Configuration for the APOBEC dataset pipeline."""

    # Data paths
    labels_csv: Path = PROJECT_ROOT / "data" / "processed" / "editing_sites_labels.csv"
    unified_csv: Path = PROJECT_ROOT / "data" / "processed" / "advisor" / "unified_editing_sites.csv"
    structures_csv: Path = PROJECT_ROOT / "data" / "processed" / "advisor" / "supp_t3_structures.csv"
    genome_fasta: Optional[Path] = None  # hg19 indexed FASTA

    # Sequence extraction
    window_size: int = 100  # nucleotides on each side of edit -> 201nt total

    # Negative sampling
    neg_ratio: int = 5  # negatives per positive
    neg_strategy: str = "mixed"  # "easy", "hard", "mixed"
    hard_neg_fraction: float = 0.3  # fraction of negatives that are hard

    # Feature computation
    compute_structure: bool = True
    structure_window: int = 70  # for RNAplfold

    # Caching
    cache_dir: Path = PROJECT_ROOT / "data" / "cache" / "apobec"

    def __post_init__(self):
        if isinstance(self.labels_csv, str):
            self.labels_csv = Path(self.labels_csv)
        if isinstance(self.cache_dir, str):
            self.cache_dir = Path(self.cache_dir)


# ---------------------------------------------------------------------------
# Feature engineering functions
# ---------------------------------------------------------------------------

def get_flanking_context(sequence: str, edit_pos: int) -> int:
    """Get the flanking dinucleotide context class for a C site.

    Looks at position -1 relative to the target C to determine
    TC (A3A preferred), CC (A3G preferred), AC, or GC context.

    Args:
        sequence: RNA sequence (ACGU).
        edit_pos: 0-indexed position of the C in the sequence.

    Returns:
        Integer context class (0=TC, 1=CC, 2=AC, 3=GC).
    """
    if edit_pos <= 0 or edit_pos >= len(sequence):
        return FLANKING_CONTEXT_MAP.get("AC", 2)  # default

    preceding = sequence[edit_pos - 1].upper()
    dinuc = preceding + "C"
    return FLANKING_CONTEXT_MAP.get(dinuc, 2)


def compute_concordance_features(
    structure_type_mrna: str,
    structure_type_premrna: str,
) -> np.ndarray:
    """Compute structure concordance features between mRNA and pre-mRNA.

    Returns a 5-dim vector:
      [is_concordant, trans_dsRNA_loop, trans_loop_dsRNA,
       trans_dsRNA_dsRNA, trans_loop_loop]

    Args:
        structure_type_mrna: Structure type in mRNA.
        structure_type_premrna: Structure type in pre-mRNA.

    Returns:
        (5,) float32 array.
    """
    features = np.zeros(5, dtype=np.float32)

    if pd.isna(structure_type_mrna) or pd.isna(structure_type_premrna):
        return features

    mrna = str(structure_type_mrna).strip().lower()
    premrna = str(structure_type_premrna).strip().lower()

    # Is concordant?
    is_concordant = mrna == premrna
    features[0] = 1.0 if is_concordant else 0.0

    if not is_concordant:
        # Classify transition type
        mrna_is_ds = "dsrna" in mrna
        premrna_is_ds = "dsrna" in premrna
        mrna_is_loop = "loop" in mrna
        premrna_is_loop = "loop" in premrna

        if mrna_is_ds and premrna_is_loop:
            features[1] = 1.0  # dsRNA -> loop
        elif mrna_is_loop and premrna_is_ds:
            features[2] = 1.0  # loop -> dsRNA
        elif mrna_is_ds and premrna_is_ds:
            features[3] = 1.0  # dsRNA -> dsRNA (different sub-type)
        elif mrna_is_loop and premrna_is_loop:
            features[4] = 1.0  # loop -> loop (different sub-type)

    return features


def encode_apobec_class(class_str: str) -> int:
    """Encode APOBEC class string to integer label.

    Returns -1 for unknown/missing classes.
    """
    if pd.isna(class_str) or str(class_str).strip() == "":
        return -1
    return APOBEC_CLASS_MAP.get(str(class_str).strip(), -1)


def encode_exonic_function(func_str: str) -> int:
    """Encode exonic function string to integer label.

    Returns -1 for unknown/missing.
    """
    if pd.isna(func_str) or str(func_str).strip() == "":
        return -1
    return EXONIC_FUNCTION_MAP.get(str(func_str).strip().lower(), -1)


def encode_structure_type(struct_str: str) -> int:
    """Encode structure type string to integer label.

    Returns -1 for unknown/missing.
    """
    if pd.isna(struct_str) or str(struct_str).strip() == "":
        return -1
    return STRUCTURE_TYPE_MAP.get(str(struct_str).strip(), -1)


def encode_tissue_spec(tissue_str: str) -> int:
    """Encode tissue specificity class string to integer label.

    Returns -1 for unknown/missing.
    """
    if pd.isna(tissue_str) or str(tissue_str).strip() == "":
        return -1
    return TISSUE_SPEC_MAP.get(str(tissue_str).strip(), -1)


# ---------------------------------------------------------------------------
# APOBEC Site Sample: a single data point
# ---------------------------------------------------------------------------

@dataclass
class APOBECSiteSample:
    """A single candidate editing site ready for model input.

    All tensors are un-batched (no leading batch dimension).
    Carries all 11 multi-task labels plus the causal edit effect target.
    """

    # Sequence window (RNA, 201nt by default, centered on the C)
    sequence: str
    edit_pos: int  # position of C within the window

    # --- PRIMARY task labels ---
    is_edited: float  # 1.0 = positive, 0.0 = negative
    editing_rate_log2: float  # log2(max_rate + 0.01), NaN if unknown
    apobec_class: int  # 0=A3A, 1=A3G, 2=Both, 3=Neither, -1=Unknown

    # --- SECONDARY task labels ---
    structure_type: int  # 0=InLoop, 1=dsRNA, 2=ssRNA/Bulge, 3=OpenssRNA, -1=unknown
    tissue_spec_class: int  # 0=Blood, 1=Ubiq, 2=Testis, 3=NonSpec, 4=Intest, -1=unknown
    n_tissues_log2: float  # log2(n_tissues_edited), NaN if unknown

    # --- TERTIARY task labels ---
    exonic_function: int  # 0=syn, 1=nonsyn, 2=stopgain, -1=non-CDS/unknown
    conservation: float  # 1.0=conserved, 0.0=not, NaN=unknown
    cancer_survival: float  # 1.0=yes, 0.0=no, NaN=unknown

    # --- AUXILIARY task labels ---
    tissue_rates: np.ndarray  # (54,) per-tissue rates in [0,1], NaN where missing
    hek293_rate: float  # HEK293 rate in [0,1], NaN if unavailable

    # --- CAUSAL task label ---
    edit_effect: float = float("nan")  # delta property, NaN until defined

    # Features (pre-computed)
    flanking_context: int = 0  # 0=TC, 1=CC, 2=AC, 3=GC
    concordance_features: np.ndarray = field(default_factory=lambda: np.zeros(5, dtype=np.float32))
    structure_delta: Optional[np.ndarray] = None  # (7,) from RNAplfold

    # Metadata (not used for training, but useful for analysis)
    site_id: str = ""
    chrom: str = ""
    position: int = 0
    gene: str = ""


# ---------------------------------------------------------------------------
# Main Dataset class
# ---------------------------------------------------------------------------

class APOBECDataset(Dataset):
    """PyTorch Dataset for APOBEC C-to-U editing site prediction.

    Produces samples compatible with the APOBECEditEmbedding and
    APOBECMultiTaskHead modules. Each sample contains:
      - sequence window (str, for on-the-fly encoding)
      - edit position within the window
      - multi-task target labels
      - pre-computed features (flanking, concordance, structure delta)

    The dataset can be created from:
      1. Pre-built sample list (fast, for training)
      2. Directly from the labels CSV (builds sample list automatically)

    Args:
        samples: List of APOBECSiteSample objects.
        config: Dataset configuration.
    """

    def __init__(
        self,
        samples: List[APOBECSiteSample],
        config: Optional[APOBECDataConfig] = None,
    ):
        self.samples = samples
        self.config = config or APOBECDataConfig()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return a dictionary of tensors for model input.

        Keys returned:
            sequence: str (not a tensor; handled by collate_fn)
            edit_pos: (1,) long tensor
            flanking_context: (1,) long tensor
            concordance_features: (5,) float tensor
            structure_delta: (7,) float tensor (zeros if unavailable)
            targets: dict of 12 label tensors matching APOBECMultiTaskLoss
            site_id: str metadata
        """
        s = self.samples[idx]

        # All 12 target labels matching APOBECMultiTaskLoss expectations:
        #   binary (float), rate (float), enzyme (long),
        #   structure (long), tissue_spec (long), n_tissues (float),
        #   function (long), conservation (float), cancer (float),
        #   tissue_matrix (float tensor), hek293 (float), effect (float)
        targets = {
            # PRIMARY
            "binary": torch.tensor(s.is_edited, dtype=torch.float32),
            "rate": torch.tensor(s.editing_rate_log2, dtype=torch.float32),
            "enzyme": torch.tensor(s.apobec_class, dtype=torch.long),
            # SECONDARY
            "structure": torch.tensor(s.structure_type, dtype=torch.long),
            "tissue_spec": torch.tensor(s.tissue_spec_class, dtype=torch.long),
            "n_tissues": torch.tensor(s.n_tissues_log2, dtype=torch.float32),
            # TERTIARY
            "function": torch.tensor(s.exonic_function, dtype=torch.long),
            "conservation": torch.tensor(s.conservation, dtype=torch.float32),
            "cancer": torch.tensor(s.cancer_survival, dtype=torch.float32),
            # AUXILIARY
            "tissue_matrix": torch.tensor(s.tissue_rates, dtype=torch.float32),
            "hek293": torch.tensor(s.hek293_rate, dtype=torch.float32),
            # CAUSAL
            "effect": torch.tensor(s.edit_effect, dtype=torch.float32),
        }

        structure_delta = (
            torch.tensor(s.structure_delta, dtype=torch.float32)
            if s.structure_delta is not None
            else torch.zeros(7, dtype=torch.float32)
        )

        return {
            "sequence": s.sequence,
            "edit_pos": torch.tensor(s.edit_pos, dtype=torch.long),
            "flanking_context": torch.tensor(s.flanking_context, dtype=torch.long),
            "concordance_features": torch.tensor(
                s.concordance_features, dtype=torch.float32
            ),
            "structure_delta": structure_delta,
            "targets": targets,
            "site_id": s.site_id,
        }

    @property
    def n_positive(self) -> int:
        return sum(1 for s in self.samples if s.is_edited > 0.5)

    @property
    def n_negative(self) -> int:
        return sum(1 for s in self.samples if s.is_edited < 0.5)

    @property
    def class_ratio(self) -> float:
        """Ratio of negative to positive samples."""
        n_pos = self.n_positive
        return self.n_negative / n_pos if n_pos > 0 else float("inf")


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def apobec_collate_fn(batch: List[Dict]) -> Dict[str, Union[torch.Tensor, list]]:
    """Custom collate function for APOBECDataset.

    Handles variable-length sequences by keeping them as a list of strings.
    All tensor fields are stacked into batched tensors.
    """
    sequences = [item["sequence"] for item in batch]
    edit_pos = torch.stack([item["edit_pos"] for item in batch])
    flanking_context = torch.stack([item["flanking_context"] for item in batch])
    concordance = torch.stack([item["concordance_features"] for item in batch])
    structure_delta = torch.stack([item["structure_delta"] for item in batch])
    site_ids = [item["site_id"] for item in batch]

    # Collate targets
    targets = {}
    target_keys = batch[0]["targets"].keys()
    for key in target_keys:
        targets[key] = torch.stack([item["targets"][key] for item in batch])

    return {
        "sequences": sequences,
        "edit_pos": edit_pos,
        "flanking_context": flanking_context,
        "concordance_features": concordance,
        "structure_delta": structure_delta,
        "targets": targets,
        "site_ids": site_ids,
    }


# ---------------------------------------------------------------------------
# Dataset builder: constructs samples from labels CSV
# ---------------------------------------------------------------------------

class APOBECDatasetBuilder:
    """Builds APOBECDataset from the processed labels CSV.

    This class handles:
    - Loading and validating the labels CSV
    - Generating negative samples
    - Computing per-site features
    - Optionally computing structure features

    Usage:
        builder = APOBECDatasetBuilder(config)
        dataset = builder.build()
    """

    def __init__(self, config: Optional[APOBECDataConfig] = None):
        self.config = config or APOBECDataConfig()

    def build(
        self,
        labels_df: Optional[pd.DataFrame] = None,
        sequences: Optional[Dict[str, str]] = None,
    ) -> APOBECDataset:
        """Build the dataset from labels and sequences.

        Args:
            labels_df: Pre-loaded labels DataFrame. If None, loads from config path.
            sequences: Dict mapping site_id -> sequence window (201nt).
                If None, sequences must be fetched from genome (requires genome_fasta).

        Returns:
            APOBECDataset with all samples.
        """
        if labels_df is None:
            labels_df = self._load_labels()

        logger.info("Building APOBEC dataset from %d labeled sites", len(labels_df))

        # Build positive samples
        positive_samples = self._build_positive_samples(labels_df, sequences)
        logger.info("  Positive samples: %d", len(positive_samples))

        # Build negative samples
        negative_samples = self._build_negative_samples(
            labels_df, sequences, n_target=len(positive_samples) * self.config.neg_ratio
        )
        logger.info("  Negative samples: %d", len(negative_samples))

        all_samples = positive_samples + negative_samples
        logger.info("  Total samples: %d", len(all_samples))

        return APOBECDataset(all_samples, self.config)

    def _load_labels(self) -> pd.DataFrame:
        """Load the labels CSV with validation."""
        csv_path = self.config.labels_csv
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Labels CSV not found: {csv_path}\n"
                "Run: python scripts/apobec/parse_advisor_excel.py first."
            )
        df = pd.read_csv(csv_path)

        required_cols = ["site_id", "chr", "start", "end"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Labels CSV missing columns: {missing}")

        return df

    def _build_positive_samples(
        self,
        df: pd.DataFrame,
        sequences: Optional[Dict[str, str]],
    ) -> List[APOBECSiteSample]:
        """Build positive (edited) samples from the labels DataFrame."""
        samples = []
        w = self.config.window_size

        # Pre-resolve column names once
        rate_col = _find_column(df, ["log2_max_rate"])
        raw_rate_col = _find_column(df, ["max_gtex_rate", "max_gtex_editing_rate"])
        class_col = _find_column(df, ["apobec_class", "affecting_over_expressed_apobec"])
        func_col = _find_column(df, ["exonic_function"])
        struct_col = _find_column(df, ["structure_type"])
        tissue_cls_col = _find_column(df, ["tissue_class"])
        n_tissues_col = _find_column(df, ["n_tissues_edited"])
        conserv_col = _find_column(df, ["any_mammalian_conservation"])
        cancer_col = _find_column(df, ["has_survival_association"])
        hek_col = _find_column(df, ["hek293_rate"])
        mrna_col = _find_column(df, ["structure_type_mRNA", "structure_type_mrna"])
        premrna_col = _find_column(df, ["structure_type_premRNA", "structure_type_premrna"])

        for _, row in df.iterrows():
            site_id = str(row.get("site_id", ""))

            # Get sequence
            seq = self._get_sequence(site_id, row, sequences)
            if seq is None:
                continue

            # Edit position is at the center of the window
            edit_pos = min(w, len(seq) // 2)

            # --- PRIMARY ---
            # Editing rate: use pre-computed log2 column if available
            if rate_col:
                editing_rate_log2 = float(row.get(rate_col, np.nan))
            elif raw_rate_col:
                raw_rate = float(row.get(raw_rate_col, np.nan))
                if not np.isnan(raw_rate):
                    # Normalize percentage to fraction, then log2 transform
                    if raw_rate > 1.0:
                        raw_rate = raw_rate / 100.0
                    editing_rate_log2 = np.log2(raw_rate + 0.01)
                else:
                    editing_rate_log2 = np.nan
            else:
                editing_rate_log2 = np.nan

            # APOBEC class
            apobec_class = encode_apobec_class(row.get(class_col, "")) if class_col else -1

            # --- SECONDARY ---
            # Structure type
            structure_type = encode_structure_type(row.get(struct_col, "")) if struct_col else -1

            # Tissue specificity class
            tissue_spec_class = encode_tissue_spec(row.get(tissue_cls_col, "")) if tissue_cls_col else -1

            # N tissues edited (log2 transform)
            if n_tissues_col:
                n_raw = row.get(n_tissues_col, np.nan)
                n_tissues_log2 = np.log2(float(n_raw)) if not pd.isna(n_raw) and float(n_raw) > 0 else np.nan
            else:
                n_tissues_log2 = np.nan

            # --- TERTIARY ---
            # Exonic function
            exonic_func = encode_exonic_function(row.get(func_col, "")) if func_col else -1

            # Conservation (boolean -> float)
            if conserv_col:
                cv = row.get(conserv_col, np.nan)
                if pd.isna(cv) or str(cv).strip() == "":
                    conservation = float("nan")
                else:
                    conservation = 1.0 if str(cv).strip().lower() in ("true", "1", "yes") else 0.0
            else:
                conservation = float("nan")

            # Cancer survival association (boolean -> float)
            if cancer_col:
                cs = row.get(cancer_col, np.nan)
                if pd.isna(cs) or str(cs).strip() == "":
                    cancer_survival = float("nan")
                else:
                    cancer_survival = 1.0 if str(cs).strip().lower() in ("true", "1", "yes") else 0.0
            else:
                cancer_survival = float("nan")

            # --- AUXILIARY ---
            # Per-tissue rates
            tissue_rates = self._extract_tissue_rates(row, df.columns)

            # HEK293 rate (in [0,1], NaN if unavailable)
            if hek_col:
                hek_raw = row.get(hek_col, np.nan)
                if pd.isna(hek_raw):
                    hek293_rate = float("nan")
                else:
                    hek293_rate = float(hek_raw)
                    if hek293_rate > 1.0:
                        hek293_rate = hek293_rate / 100.0
            else:
                hek293_rate = float("nan")

            # --- Features ---
            flanking = get_flanking_context(seq, edit_pos)

            concordance = compute_concordance_features(
                row.get(mrna_col, "") if mrna_col else "",
                row.get(premrna_col, "") if premrna_col else "",
            )

            sample = APOBECSiteSample(
                sequence=seq,
                edit_pos=edit_pos,
                is_edited=1.0,
                editing_rate_log2=editing_rate_log2,
                apobec_class=apobec_class,
                structure_type=structure_type,
                tissue_spec_class=tissue_spec_class,
                n_tissues_log2=n_tissues_log2,
                exonic_function=exonic_func,
                conservation=conservation,
                cancer_survival=cancer_survival,
                tissue_rates=tissue_rates,
                hek293_rate=hek293_rate,
                flanking_context=flanking,
                concordance_features=concordance,
                site_id=site_id,
                chrom=str(row.get("chr", "")),
                position=int(row.get("start", 0)),
                gene=str(row.get("gene_name", row.get("gene_refseq", ""))),
            )
            samples.append(sample)

        return samples

    def _build_negative_samples(
        self,
        positive_df: pd.DataFrame,
        sequences: Optional[Dict[str, str]],
        n_target: int = 3000,
    ) -> List[APOBECSiteSample]:
        """Generate negative samples (non-edited C sites).

        Negative sampling strategy:
        - Easy: random C positions from different genes
        - Hard: C positions in same gene, similar structural context
        - Mixed: combination of both

        For now, uses the sequences from positive sites to generate negatives
        by picking non-center C positions within the same windows.
        When genome access is available, can also sample from unrelated regions.
        """
        samples = []
        w = self.config.window_size
        strategy = self.config.neg_strategy

        # Use positive site sequences for generating within-gene negatives
        # Collect all sequences we have
        pos_sequences = []
        for _, row in positive_df.iterrows():
            site_id = str(row.get("site_id", ""))
            seq = self._get_sequence(site_id, row, sequences)
            if seq is not None:
                pos_sequences.append((seq, row))

        if not pos_sequences:
            logger.warning("No sequences available for negative sampling")
            return samples

        rng = np.random.RandomState(42)

        # Hard negatives: other C positions within positive-site windows
        n_hard = int(n_target * self.config.hard_neg_fraction)
        n_easy = n_target - n_hard

        # --- Hard negatives: C positions in same sequence windows ---
        hard_candidates = []
        for seq, row in pos_sequences:
            center = min(w, len(seq) // 2)
            for i, nuc in enumerate(seq):
                if nuc.upper() == "C" and i != center:
                    # Must be at least 3nt from the true edit
                    if abs(i - center) >= 3:
                        hard_candidates.append((seq, i, row))

        if hard_candidates and n_hard > 0:
            chosen_indices = rng.choice(
                len(hard_candidates), size=min(n_hard, len(hard_candidates)), replace=False
            )
            for idx in chosen_indices:
                seq, neg_pos, row = hard_candidates[idx]
                sample = self._make_negative_sample(
                    seq, neg_pos, row, positive_df.columns
                )
                samples.append(sample)

        # --- Easy negatives: shifted windows centered on different C positions ---
        easy_generated = 0
        max_attempts = n_easy * 3
        attempts = 0
        while easy_generated < n_easy and attempts < max_attempts:
            attempts += 1
            seq, row = pos_sequences[rng.randint(len(pos_sequences))]
            # Pick a random C position far from center
            c_positions = [
                i for i, nuc in enumerate(seq)
                if nuc.upper() == "C" and abs(i - len(seq) // 2) > 20
            ]
            if not c_positions:
                continue
            neg_pos = c_positions[rng.randint(len(c_positions))]
            sample = self._make_negative_sample(
                seq, neg_pos, row, positive_df.columns
            )
            samples.append(sample)
            easy_generated += 1

        return samples

    def _make_negative_sample(
        self,
        sequence: str,
        neg_pos: int,
        source_row: pd.Series,
        columns: pd.Index,
    ) -> APOBECSiteSample:
        """Create a negative sample at a non-edited C position."""
        flanking = get_flanking_context(sequence, neg_pos)

        # Structure concordance defaults to concordant (unknown)
        concordance = np.zeros(5, dtype=np.float32)
        concordance[0] = 1.0  # assume concordant for negatives

        return APOBECSiteSample(
            sequence=sequence,
            edit_pos=neg_pos,
            is_edited=0.0,
            editing_rate_log2=float("nan"),  # no rate for negatives
            apobec_class=3,  # "Neither"
            structure_type=-1,  # unknown
            tissue_spec_class=-1,  # unknown
            n_tissues_log2=float("nan"),
            exonic_function=-1,  # unknown
            conservation=float("nan"),
            cancer_survival=float("nan"),
            tissue_rates=np.full(N_TISSUES, np.nan, dtype=np.float32),
            hek293_rate=float("nan"),
            flanking_context=flanking,
            concordance_features=concordance,
            site_id=f"NEG_{source_row.get('site_id', 'unk')}_{neg_pos}",
            chrom=str(source_row.get("chr", "")),
            position=int(source_row.get("start", 0)),
            gene=str(source_row.get("gene_name", source_row.get("gene_refseq", ""))),
        )

    def _get_sequence(
        self,
        site_id: str,
        row: pd.Series,
        sequences: Optional[Dict[str, str]],
    ) -> Optional[str]:
        """Get the sequence window for a site.

        Tries in order:
        1. Pre-provided sequences dict
        2. Genome FASTA extraction
        """
        # 1. Pre-provided sequences
        if sequences is not None and site_id in sequences:
            return sequences[site_id]

        # 2. Genome extraction
        if self.config.genome_fasta is not None:
            from .rna_structure import extract_sequence_context

            chrom = str(row.get("chr", ""))
            start = int(row.get("start", 0))
            seq = extract_sequence_context(
                chrom, start,
                flank_size=self.config.window_size,
                genome_fasta=self.config.genome_fasta,
            )
            return seq

        return None

    def _extract_tissue_rates(
        self,
        row: pd.Series,
        columns: pd.Index,
    ) -> np.ndarray:
        """Extract per-tissue editing rates from a row.

        Handles both the labels CSV format (direct rate columns) and
        the T1 format (mismatch;coverage;rate strings).
        """
        rates = np.full(N_TISSUES, np.nan, dtype=np.float32)

        for i, tissue in enumerate(GTEX_TISSUES):
            # Try direct column match
            matching = [c for c in columns if tissue in str(c).replace(" ", "_")]
            if not matching:
                continue

            val = row.get(matching[0], np.nan)
            if pd.isna(val) or val == "":
                continue

            val_str = str(val)
            # Handle mismatch;coverage;rate format
            if ";" in val_str:
                parts = val_str.split(";")
                if len(parts) == 3:
                    try:
                        rate = float(parts[2])
                        rates[i] = rate / 100.0 if rate > 1.0 else rate
                    except ValueError:
                        pass
            else:
                try:
                    rate = float(val_str)
                    rates[i] = rate / 100.0 if rate > 1.0 else rate
                except ValueError:
                    pass

        return rates


# ---------------------------------------------------------------------------
# Structure feature computation (optional, requires ViennaRNA)
# ---------------------------------------------------------------------------

def compute_structure_delta_features(
    sequence: str,
    edit_pos: int,
    structure_predictor=None,
) -> Optional[np.ndarray]:
    """Compute 7-dim structure delta features for a C-to-U edit.

    Features:
    1. delta_pairing_at_pos: change in pairing probability at edit position
    2. delta_accessibility_at_pos: change in accessibility at edit position
    3. delta_entropy_at_pos: change in structure entropy at edit position
    4. delta_mfe: change in minimum free energy
    5. delta_local_pairing: mean change in pairing probs in local window
    6. delta_local_accessibility: mean change in accessibility in local window
    7. local_pairing_std: std of pairing prob changes in local window

    Args:
        sequence: RNA sequence window.
        edit_pos: Position of C within the window.
        structure_predictor: RNAplfoldPredictor instance.

    Returns:
        (7,) float32 array, or None if structure prediction fails.
    """
    if structure_predictor is None:
        return None

    seq_before = sequence
    seq_list = list(sequence)
    if edit_pos < len(seq_list) and seq_list[edit_pos].upper() == "C":
        seq_list[edit_pos] = "U"
    seq_after = "".join(seq_list)

    try:
        delta = structure_predictor.compute_delta_structure(
            seq_before, seq_after, edit_pos
        )
    except Exception as e:
        logger.debug("Structure prediction failed: %s", e)
        return None

    window = 10
    n = len(sequence)
    start = max(0, edit_pos - window)
    end = min(n, edit_pos + window + 1)

    dp = delta["delta_pairing"]
    da = delta["delta_accessibility"]
    de = delta["delta_entropy"]

    features = np.array(
        [
            dp[edit_pos] if edit_pos < len(dp) else 0.0,
            da[edit_pos] if edit_pos < len(da) else 0.0,
            de[edit_pos] if edit_pos < len(de) else 0.0,
            delta["delta_mfe"],
            delta["delta_local_pairing"],
            np.mean(da[start:end]) if len(da) > 0 else 0.0,
            np.std(dp[start:end]) if len(dp) > 0 else 0.0,
        ],
        dtype=np.float32,
    )
    return features


def add_structure_features(
    dataset: APOBECDataset,
    structure_predictor=None,
    show_progress: bool = True,
) -> APOBECDataset:
    """Add structure delta features to all samples in a dataset.

    Modifies samples in-place and returns the same dataset.

    Args:
        dataset: APOBECDataset to augment.
        structure_predictor: RNAplfoldPredictor instance.
            If None, attempts to create one.
        show_progress: Show progress bar.

    Returns:
        The same dataset with structure_delta populated.
    """
    if structure_predictor is None:
        try:
            from ..embedding.rnaplfold import RNAplfoldPredictor

            structure_predictor = RNAplfoldPredictor()
            if not structure_predictor.is_available:
                logger.warning("ViennaRNA not available; skipping structure features")
                return dataset
        except ImportError:
            logger.warning("Cannot import RNAplfoldPredictor; skipping structure features")
            return dataset

    iterator = dataset.samples
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="Computing structure features")
        except ImportError:
            pass

    for sample in iterator:
        sample.structure_delta = compute_structure_delta_features(
            sample.sequence, sample.edit_pos, structure_predictor
        )

    return dataset


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def create_apobec_dataloaders(
    train_dataset: APOBECDataset,
    val_dataset: Optional[APOBECDataset] = None,
    test_dataset: Optional[APOBECDataset] = None,
    batch_size: int = 32,
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
    """Create DataLoaders for APOBEC datasets.

    Args:
        train_dataset: Training dataset.
        val_dataset: Optional validation dataset.
        test_dataset: Optional test dataset.
        batch_size: Batch size.
        num_workers: Number of data loading workers.

    Returns:
        Dict with 'train', and optionally 'val', 'test' DataLoaders.
    """
    loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=apobec_collate_fn,
            pin_memory=torch.cuda.is_available(),
        ),
    }

    if val_dataset is not None:
        loaders["val"] = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=apobec_collate_fn,
            pin_memory=torch.cuda.is_available(),
        )

    if test_dataset is not None:
        loaders["test"] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=apobec_collate_fn,
            pin_memory=torch.cuda.is_available(),
        )

    return loaders


# ---------------------------------------------------------------------------
# Train/Val/Test splitting with stratification
# ---------------------------------------------------------------------------

def split_dataset(
    dataset: APOBECDataset,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    stratify_by: str = "gene",
    seed: int = 42,
) -> Tuple[APOBECDataset, APOBECDataset, APOBECDataset]:
    """Split dataset into train/val/test with stratification.

    Stratification ensures that sites from the same gene do not leak
    across splits. This prevents the model from memorizing gene-specific
    patterns during training and evaluating on similar patterns in val/test.

    Args:
        dataset: Full APOBECDataset.
        val_fraction: Fraction for validation.
        test_fraction: Fraction for test.
        stratify_by: Stratification key ("gene" or "chrom").
        seed: Random seed.

    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    rng = np.random.RandomState(seed)
    samples = dataset.samples

    # Group samples by stratification key
    groups: Dict[str, List[int]] = {}
    for i, s in enumerate(samples):
        key = s.gene if stratify_by == "gene" else s.chrom
        if not key:
            key = "__unknown__"
        groups.setdefault(key, []).append(i)

    # Shuffle groups
    group_keys = list(groups.keys())
    rng.shuffle(group_keys)

    # Assign groups to splits, trying to hit target fractions
    n_total = len(samples)
    n_val_target = int(n_total * val_fraction)
    n_test_target = int(n_total * test_fraction)

    val_indices = []
    test_indices = []
    train_indices = []

    n_val = 0
    n_test = 0

    for key in group_keys:
        idxs = groups[key]
        if n_test < n_test_target:
            test_indices.extend(idxs)
            n_test += len(idxs)
        elif n_val < n_val_target:
            val_indices.extend(idxs)
            n_val += len(idxs)
        else:
            train_indices.extend(idxs)

    train_samples = [samples[i] for i in train_indices]
    val_samples = [samples[i] for i in val_indices]
    test_samples = [samples[i] for i in test_indices]

    logger.info(
        "Split: train=%d, val=%d, test=%d (total=%d)",
        len(train_samples), len(val_samples), len(test_samples), n_total,
    )

    return (
        APOBECDataset(train_samples, dataset.config),
        APOBECDataset(val_samples, dataset.config),
        APOBECDataset(test_samples, dataset.config),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Find the first matching column name from a list of candidates."""
    for c in candidates:
        if c in df.columns:
            return c
        # Case-insensitive fallback
        for col in df.columns:
            if col.lower() == c.lower():
                return col
    return None
