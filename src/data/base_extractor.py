"""
Base classes for data extraction.

This provides a common interface for extracting paired data from RNA datasets.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExtractionConfig(ABC):
    """
    Base configuration for any data extraction pipeline.

    Each modality-specific config should inherit from this and add
    modality-specific parameters.
    """

    data_dir: Path
    output_name: str  # Name for output files

    @property
    def output_dir(self) -> Path:
        """Directory for extracted data."""
        return self.data_dir / "extracted"

    @property
    def checkpoint_dir(self) -> Path:
        """Directory for checkpoints."""
        return self.data_dir / "checkpoints"

    def __post_init__(self):
        """Ensure directories exist."""
        if isinstance(self.data_dir, str):
            object.__setattr__(self, 'data_dir', Path(self.data_dir))

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)


class PairDataExtractor(ABC):
    """
    Base class for extracting paired data with properties.

    All extractors should:
    1. Download/access raw data
    2. Generate pairs (e.g., mutations for RNA sequences)
    3. Output in long format: (entity_a, entity_b, edit_info, property_name, value_a, value_b, delta)
    """

    def __init__(self, config: ExtractionConfig):
        """
        Initialize extractor with configuration.

        Args:
            config: Extraction configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def download_if_needed(self) -> bool:
        """
        Download raw data if not already cached.

        Returns:
            True if download was performed, False if already cached
        """
        pass

    @abstractmethod
    def extract_pairs(self, **kwargs) -> pd.DataFrame:
        """
        Extract pairs from downloaded data.

        Returns:
            DataFrame in long format with columns:
            - entity_a, entity_b: Identifiers (SMILES, sequence, etc.)
            - edit_info: Information about the transformation
            - property_name: Name of the property
            - value_a, value_b: Property values
            - delta: value_b - value_a
            - (modality-specific columns)
        """
        pass

    def run(self, skip_download: bool = False, **kwargs) -> pd.DataFrame:
        """
        Run full extraction pipeline.

        Args:
            skip_download: Skip download step (use existing data)
            **kwargs: Passed to extract_pairs()

        Returns:
            Extracted pairs DataFrame
        """
        self.logger.info(f"Starting extraction: {self.__class__.__name__}")

        # Step 1: Download if needed
        if not skip_download:
            downloaded = self.download_if_needed()
            if downloaded:
                self.logger.info("✓ Download complete")
            else:
                self.logger.info("✓ Using cached data")

        # Step 2: Extract pairs
        self.logger.info("Extracting pairs...")
        pairs_df = self.extract_pairs(**kwargs)

        self.logger.info(f"✓ Extraction complete")
        self.logger.info(f"  Total pairs: {len(pairs_df):,}")
        self.logger.info(f"  Unique edits: {pairs_df.get('edit_smiles', pairs_df.get('edit_info', pd.Series())).nunique():,}")
        self.logger.info(f"  Properties: {pairs_df['property_name'].nunique():,}")

        return pairs_df

    def get_status(self) -> Dict[str, Any]:
        """
        Get extraction status and statistics.

        Returns:
            Dictionary with status info
        """
        status = {
            'extractor': self.__class__.__name__,
            'data_dir': str(self.config.data_dir),
            'output_dir': str(self.config.output_dir),
        }

        # Check if data exists
        if self.config.output_dir.exists():
            files = list(self.config.output_dir.glob('*.csv'))
            status['existing_files'] = [f.name for f in files]
            status['file_count'] = len(files)
        else:
            status['existing_files'] = []
            status['file_count'] = 0

        return status
