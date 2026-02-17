"""Data loading utilities for APOBEC C-to-U RNA editing sites.

Provides a unified interface for loading editing site data from the advisor's
processed dataset and published studies. Each site has genomic coordinates
(hg19), gene annotation, editing rates, and optional structure annotations.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "advisor"


@dataclass
class EditingSite:
    """A single C-to-U RNA editing site."""
    site_id: str
    chrom: str
    start: int
    end: int
    gene: str = ""
    genomic_category: str = ""
    tissue_classification: str = ""
    max_editing_rate: float = 0.0
    mean_editing_rate: float = 0.0
    structure_type: str = ""
    sequence_context: str = ""
    # Per-tissue editing rates (tissue_name -> rate)
    tissue_rates: dict[str, float] = field(default_factory=dict)


class EditingSiteDataset:
    """Load and access the unified editing site dataset."""

    def __init__(self, csv_path: Optional[Path] = None):
        if csv_path is None:
            csv_path = PROCESSED_DIR / "unified_editing_sites.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Unified site table not found: {csv_path}\n"
                "Run: python scripts/apobec/parse_advisor_excel.py first."
            )
        self.df = pd.read_csv(csv_path)
        self._validate()

    def _validate(self):
        """Basic validation of the loaded data."""
        required = ["site_id", "chr", "start", "end"]
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> pd.Series:
        return self.df.iloc[idx]

    @property
    def coordinates(self) -> pd.DataFrame:
        """Return BED-format coordinates (chr, start, end)."""
        return self.df[["chr", "start", "end"]].copy()

    def get_sites_by_gene(self, gene: str) -> pd.DataFrame:
        """Get all editing sites in a specific gene."""
        gene_col = [c for c in self.df.columns if "gene" in c.lower()][0]
        return self.df[self.df[gene_col].str.contains(gene, case=False, na=False)]

    def get_sites_by_tissue(self, tissue: str) -> pd.DataFrame:
        """Get sites edited in a specific tissue classification."""
        tissue_col = [c for c in self.df.columns if "tissue_classification" in c.lower()]
        if not tissue_col:
            raise ValueError("No tissue classification column found")
        return self.df[self.df[tissue_col[0]].str.contains(tissue, case=False, na=False)]

    def get_high_editing_sites(self, min_rate: float = 10.0) -> pd.DataFrame:
        """Get sites with max editing rate above threshold."""
        rate_col = [c for c in self.df.columns if "max" in c.lower() and "editing" in c.lower()]
        if not rate_col:
            raise ValueError("No max editing rate column found")
        return self.df[self.df[rate_col[0]] >= min_rate]

    def to_bed(self, output_path: Path):
        """Export coordinates as BED file."""
        bed = self.coordinates.copy()
        bed.insert(3, "name", self.df["site_id"])
        bed.insert(4, "score", 0)
        bed.insert(5, "strand", ".")
        bed.to_csv(output_path, sep="\t", header=False, index=False)

    def summary(self) -> dict:
        """Return summary statistics of the dataset."""
        info = {
            "n_sites": len(self.df),
            "n_chromosomes": self.df["chr"].nunique(),
            "chromosomes": sorted(self.df["chr"].unique()),
        }
        # Add tissue classification distribution if available
        tissue_col = [c for c in self.df.columns if "tissue_classification" in c.lower()]
        if tissue_col:
            info["tissue_distribution"] = self.df[tissue_col[0]].value_counts().to_dict()
        # Add genomic category distribution
        cat_col = [c for c in self.df.columns if "genomic_category" in c.lower()]
        if cat_col:
            info["genomic_categories"] = self.df[cat_col[0]].value_counts().to_dict()
        return info


def load_gtex_tissue_rates(csv_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load the per-tissue GTEx editing rates from the T1 parsed CSV.

    Returns a DataFrame with site_id as index and tissue columns as editing rates.
    The raw data encodes rates as 'mismatch;coverage;rate' strings.
    """
    if csv_dir is None:
        csv_dir = PROCESSED_DIR

    t1_path = csv_dir / "t1_gtex_editing_&_conservation.csv"
    if not t1_path.exists():
        raise FileNotFoundError(f"T1 CSV not found: {t1_path}")

    df = pd.read_csv(t1_path)

    # Tissue columns contain data like "346;2270;15.2" (mismatch;coverage;rate)
    # Find tissue columns (they start after conservation columns and are GTEx tissues)
    gtex_tissues = [
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

    def parse_rate(val):
        """Extract editing rate from 'mismatch;coverage;rate' string."""
        if pd.isna(val) or val == "" or val == "NA":
            return np.nan
        try:
            parts = str(val).split(";")
            if len(parts) == 3:
                return float(parts[2])
        except (ValueError, IndexError):
            pass
        return np.nan

    # Find matching tissue columns in the dataframe
    tissue_data = {}
    for tissue in gtex_tissues:
        matching = [c for c in df.columns if tissue in c.replace(" ", "_")]
        if matching:
            tissue_data[tissue] = df[matching[0]].apply(parse_rate)

    rates_df = pd.DataFrame(tissue_data)
    rates_df.insert(0, "chr", df.get("Chr", df.iloc[:, 0]))
    rates_df.insert(1, "start", df.get("Start", df.iloc[:, 1]))
    rates_df.insert(2, "end", df.get("End", df.iloc[:, 2]))

    return rates_df
