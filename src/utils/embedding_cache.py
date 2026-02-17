"""
Embedding cache utilities for saving/loading pre-computed embeddings.
"""

import numpy as np
import pandas as pd
import pickle
import hashlib
from pathlib import Path
from typing import Optional, Tuple, Dict
import json


class EmbeddingCache:
    """
    Cache for molecular embeddings to avoid re-computation.

    Features:
    - Saves embeddings to disk with metadata
    - Loads cached embeddings if available
    - Validates cache against embedder configuration
    - Handles multiple datasets (train/val/test)

    """

    def __init__(self, cache_dir: str = '.embeddings_cache'):
        """
        Initialize embedding cache.

        Args:
            cache_dir: Directory to store cached embeddings
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(
        self,
        smiles: list,
        embedder_name: str,
        embedder_config: dict
    ) -> str:
        """
        Generate unique cache key based on SMILES and embedder configuration.

        Args:
            smiles: List of SMILES strings
            embedder_name: Name of embedder (e.g., 'chemberta', 'morgan')
            embedder_config: Embedder configuration dict

        Returns:
            Cache key (hex string)
        """
        # Create deterministic hash of SMILES + embedder config
        smiles_hash = hashlib.md5('|'.join(sorted(smiles)).encode()).hexdigest()
        config_str = json.dumps(embedder_config, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()

        return f"{embedder_name}_{smiles_hash[:16]}_{config_hash[:8]}"

    def _get_embedder_config(self, embedder) -> dict:
        """
        Extract embedder configuration for cache validation.

        Args:
            embedder: MoleculeEmbedder instance

        Returns:
            Configuration dict
        """
        config = {
            'name': embedder.name,
            'embedding_dim': embedder.embedding_dim
        }

        # Add embedder-specific config
        if hasattr(embedder, 'radius'):
            config['radius'] = embedder.radius
        if hasattr(embedder, 'n_bits'):
            config['n_bits'] = embedder.n_bits
        if hasattr(embedder, 'fp_type'):
            config['fp_type'] = embedder.fp_type
        if hasattr(embedder, 'model_name'):
            config['model_name'] = embedder.model_name

        return config

    def get_cache_path(
        self,
        dataset_name: str,
        embedder_name: str,
        smiles_hash: str
    ) -> Path:
        """
        Get path to cache file.

        Args:
            dataset_name: Name of dataset (e.g., 'train', 'val', 'test')
            embedder_name: Name of embedder
            smiles_hash: Hash of SMILES list

        Returns:
            Path to cache file
        """
        filename = f"{dataset_name}_{embedder_name}_{smiles_hash}.npz"
        return self.cache_dir / filename

    def load(
        self,
        dataset_name: str,
        embedder,
        smiles: list
    ) -> Optional[np.ndarray]:
        """
        Load cached embeddings if available and valid.

        Args:
            dataset_name: Name of dataset
            embedder: Embedder instance (for config validation)
            smiles: List of SMILES strings

        Returns:
            Cached embeddings if available, None otherwise
        """
        embedder_config = self._get_embedder_config(embedder)
        cache_key = self._get_cache_key(smiles, embedder_config['name'], embedder_config)
        cache_path = self.cache_dir / f"{dataset_name}_{cache_key}.npz"

        if not cache_path.exists():
            return None

        try:
            # Load cache file
            data = np.load(cache_path, allow_pickle=True)

            # Validate metadata
            metadata = data['metadata'].item()

            # Check embedder config matches
            if metadata['embedder_config'] != embedder_config:
                print(f"  ⚠️  Cache invalid: embedder config mismatch")
                return None

            # Check SMILES count matches
            if metadata['n_smiles'] != len(smiles):
                print(f"  ⚠️  Cache invalid: SMILES count mismatch")
                return None

            embeddings = data['embeddings']

            print(f"  ✓ Loaded cached embeddings from {cache_path.name}")
            print(f"    Shape: {embeddings.shape}, Embedder: {embedder_config['name']}")

            return embeddings

        except Exception as e:
            print(f"  ⚠️  Error loading cache: {e}")
            return None

    def save(
        self,
        embeddings: np.ndarray,
        dataset_name: str,
        embedder,
        smiles: list
    ):
        """
        Save embeddings to cache.

        Args:
            embeddings: Computed embeddings array
            dataset_name: Name of dataset
            embedder: Embedder instance (for metadata)
            smiles: List of SMILES strings
        """
        embedder_config = self._get_embedder_config(embedder)
        cache_key = self._get_cache_key(smiles, embedder_config['name'], embedder_config)
        cache_path = self.cache_dir / f"{dataset_name}_{cache_key}.npz"

        # Save with metadata
        metadata = {
            'embedder_config': embedder_config,
            'n_smiles': len(smiles),
            'dataset_name': dataset_name
        }

        np.savez_compressed(
            cache_path,
            embeddings=embeddings,
            metadata=np.array([metadata])  # Wrap in array for npz
        )

        print(f"  ✓ Saved embeddings to cache: {cache_path.name}")
        print(f"    Shape: {embeddings.shape}, Size: {cache_path.stat().st_size / 1024:.1f} KB")

    def get_or_compute(
        self,
        smiles: list,
        embedder,
        dataset_name: str,
        force_recompute: bool = False
    ) -> np.ndarray:
        """
        Get embeddings from cache or compute if not available.

        Args:
            smiles: List of SMILES strings
            embedder: Embedder instance
            dataset_name: Name of dataset (for cache naming)
            force_recompute: If True, ignore cache and recompute

        Returns:
            Embeddings array
        """
        # Try to load from cache
        if not force_recompute:
            cached = self.load(dataset_name, embedder, smiles)
            if cached is not None:
                return cached

        # Compute embeddings
        print(f"  Computing embeddings for {len(smiles)} molecules...")
        embeddings = embedder.encode(smiles)

        # Save to cache
        self.save(embeddings, dataset_name, embedder, smiles)

        return embeddings

    def clear(self):
        """Clear all cached embeddings."""
        for cache_file in self.cache_dir.glob("*.npz"):
            cache_file.unlink()
        print(f"✓ Cleared all cached embeddings from {self.cache_dir}")

    def list_cached(self) -> list:
        """
        List all cached embeddings.

        Returns:
            List of (dataset_name, embedder_name, file_size) tuples
        """
        cached = []
        for cache_file in self.cache_dir.glob("*.npz"):
            try:
                data = np.load(cache_file, allow_pickle=True)
                metadata = data['metadata'].item()
                embeddings = data['embeddings']

                cached.append({
                    'file': cache_file.name,
                    'dataset': metadata['dataset_name'],
                    'embedder': metadata['embedder_config']['name'],
                    'n_molecules': metadata['n_smiles'],
                    'shape': embeddings.shape,
                    'size_kb': cache_file.stat().st_size / 1024
                })
            except:
                pass

        return cached


def get_or_compute_embeddings_for_pairs(
    df: pd.DataFrame,
    embedder,
    cache: EmbeddingCache,
    dataset_name: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get or compute embeddings for molecular pairs (mol_a, mol_b).

    Args:
        df: DataFrame with 'mol_a' and 'mol_b' columns
        embedder: Embedder instance
        cache: EmbeddingCache instance
        dataset_name: Name of dataset (e.g., 'train', 'val', 'test')

    Returns:
        Tuple of (mol_emb_a, mol_emb_b)
    """
    # Get unique molecules from both columns
    unique_a = df['mol_a'].unique().tolist()
    unique_b = df['mol_b'].unique().tolist()
    all_unique = list(set(unique_a + unique_b))

    print(f"\n{dataset_name.upper()}: {len(df)} pairs")
    print(f"  Unique mol_a: {len(unique_a)}")
    print(f"  Unique mol_b: {len(unique_b)}")
    print(f"  Total unique: {len(all_unique)}")

    # Get or compute embeddings for all unique molecules
    all_embeddings = cache.get_or_compute(
        smiles=all_unique,
        embedder=embedder,
        dataset_name=f"{dataset_name}_unique_molecules"
    )

    # Create lookup dict
    emb_lookup = {smiles: emb for smiles, emb in zip(all_unique, all_embeddings)}

    # Map back to pairs
    mol_emb_a = np.array([emb_lookup[s] for s in df['mol_a']])
    mol_emb_b = np.array([emb_lookup[s] for s in df['mol_b']])

    return mol_emb_a, mol_emb_b


def get_or_compute_embeddings_for_molecules(
    df: pd.DataFrame,
    embedder,
    cache: EmbeddingCache,
    dataset_name: str,
    smiles_column: str = 'smiles'
) -> np.ndarray:
    """
    Get or compute embeddings for molecules.

    Args:
        df: DataFrame with smiles column
        embedder: Embedder instance
        cache: EmbeddingCache instance
        dataset_name: Name of dataset (e.g., 'train', 'val', 'test')
        smiles_column: Name of SMILES column

    Returns:
        Embeddings array
    """
    smiles_list = df[smiles_column].tolist()

    print(f"\n{dataset_name.upper()}: {len(smiles_list)} molecules")

    embeddings = cache.get_or_compute(
        smiles=smiles_list,
        embedder=embedder,
        dataset_name=dataset_name
    )

    return embeddings


def compute_all_embeddings_once(
    train_edit: pd.DataFrame,
    val_edit: pd.DataFrame,
    test_edit: pd.DataFrame,
    train_baseline: pd.DataFrame,
    val_baseline: pd.DataFrame,
    test_baseline: pd.DataFrame,
    embedder,
    cache: EmbeddingCache
) -> Dict[str, np.ndarray]:
    """
    Compute embeddings for all unique molecules across all datasets once.

    Args:
        train_edit: Training edit pairs DataFrame (mol_a, mol_b columns)
        val_edit: Validation edit pairs DataFrame
        test_edit: Test edit pairs DataFrame
        train_baseline: Training baseline DataFrame (smiles column)
        val_baseline: Validation baseline DataFrame
        test_baseline: Test baseline DataFrame
        embedder: Embedder instance
        cache: EmbeddingCache instance

    Returns:
        Dict mapping SMILES -> embedding
    """
    print("Collecting all unique molecules...")

    # Collect all unique molecules from all datasets
    all_smiles = set()

    # From edit pairs
    for df, name in [(train_edit, 'train_edit'), (val_edit, 'val_edit'), (test_edit, 'test_edit')]:
        all_smiles.update(df['mol_a'].unique())
        all_smiles.update(df['mol_b'].unique())

    # From baseline
    for df, name in [(train_baseline, 'train_baseline'), (val_baseline, 'val_baseline'), (test_baseline, 'test_baseline')]:
        all_smiles.update(df['smiles'].unique())

    all_smiles_list = sorted(list(all_smiles))

    print(f"Total unique molecules across all datasets: {len(all_smiles_list)}")

    # Compute or load embeddings once
    all_embeddings = cache.get_or_compute(
        smiles=all_smiles_list,
        embedder=embedder,
        dataset_name='all_unique_molecules'
    )

    # Create lookup dict
    emb_lookup = {smiles: emb for smiles, emb in zip(all_smiles_list, all_embeddings)}

    return emb_lookup


def map_embeddings_to_pairs(
    df: pd.DataFrame,
    emb_lookup: Dict[str, np.ndarray],
    dataset_name: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map pre-computed embeddings to edit pairs.

    Args:
        df: DataFrame with mol_a and mol_b columns
        emb_lookup: Dict mapping SMILES -> embedding
        dataset_name: Name for logging

    Returns:
        Tuple of (mol_emb_a, mol_emb_b)
    """
    print(f"{dataset_name}: Mapping {len(df)} pairs to embeddings")

    mol_emb_a = np.array([emb_lookup[s] for s in df['mol_a']])
    mol_emb_b = np.array([emb_lookup[s] for s in df['mol_b']])

    return mol_emb_a, mol_emb_b


def map_embeddings_to_molecules(
    df: pd.DataFrame,
    emb_lookup: Dict[str, np.ndarray],
    dataset_name: str,
    smiles_column: str = 'smiles'
) -> np.ndarray:
    """
    Map pre-computed embeddings to molecules.

    Args:
        df: DataFrame with smiles column
        emb_lookup: Dict mapping SMILES -> embedding
        dataset_name: Name for logging
        smiles_column: Name of SMILES column

    Returns:
        Embeddings array
    """
    print(f"{dataset_name}: Mapping {len(df)} molecules to embeddings")

    embeddings = np.array([emb_lookup[s] for s in df[smiles_column]])

    return embeddings


def compute_all_embeddings_with_fragments(
    train_edit: pd.DataFrame,
    val_edit: pd.DataFrame,
    test_edit: pd.DataFrame,
    train_baseline: pd.DataFrame,
    val_baseline: pd.DataFrame,
    test_baseline: pd.DataFrame,
    embedder,
    cache: EmbeddingCache,
    include_edit_fragments: bool = False
) -> Dict[str, np.ndarray]:
    """
    Compute embeddings for all unique molecules AND edit fragments across all datasets.

    Args:
        train_edit: Training edit pairs DataFrame (mol_a, mol_b, edit_smiles columns)
        val_edit: Validation edit pairs DataFrame
        test_edit: Test edit pairs DataFrame
        train_baseline: Training baseline DataFrame (smiles column)
        val_baseline: Validation baseline DataFrame
        test_baseline: Test baseline DataFrame
        embedder: Embedder instance
        cache: EmbeddingCache instance
        include_edit_fragments: If True, also embed edit fragments from edit_smiles column

    Returns:
        Dict mapping SMILES -> embedding (includes both full molecules and fragments)
    """
    try:
        from src.data.small_molecule.utils.chemistry import parse_edit_smiles
    except ImportError:
        raise NotImplementedError(
            "Fragment embedding functions require small molecule dependencies "
            "not available in this RNA-only repo"
        )

    print("Collecting all unique molecules...")

    # Collect all unique molecules from all datasets
    all_smiles = set()

    # From edit pairs
    for df, name in [(train_edit, 'train_edit'), (val_edit, 'val_edit'), (test_edit, 'test_edit')]:
        all_smiles.update(df['mol_a'].unique())
        all_smiles.update(df['mol_b'].unique())

        # Extract edit fragments if requested
        if include_edit_fragments and 'edit_smiles' in df.columns:
            for edit_smiles in df['edit_smiles'].unique():
                try:
                    frag_a, frag_b = parse_edit_smiles(edit_smiles)
                    all_smiles.add(frag_a)
                    all_smiles.add(frag_b)
                except ValueError as e:
                    print(f"  Warning: Skipping invalid edit_smiles '{edit_smiles}': {e}")

    # From baseline
    for df, name in [(train_baseline, 'train_baseline'), (val_baseline, 'val_baseline'), (test_baseline, 'test_baseline')]:
        all_smiles.update(df['smiles'].unique())

    all_smiles_list = sorted(list(all_smiles))

    print(f"Total unique molecules across all datasets: {len(all_smiles_list)}")
    if include_edit_fragments:
        print(f"  (includes edit fragments from edit_smiles column)")

    # Compute or load embeddings once
    all_embeddings = cache.get_or_compute(
        smiles=all_smiles_list,
        embedder=embedder,
        dataset_name='all_unique_molecules'
    )

    # Create lookup dict
    emb_lookup = {smiles: emb for smiles, emb in zip(all_smiles_list, all_embeddings)}

    return emb_lookup


def map_fragment_embeddings_to_pairs(
    df: pd.DataFrame,
    emb_lookup: Dict[str, np.ndarray],
    dataset_name: str,
    edit_smiles_column: str = 'edit_smiles'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map pre-computed embeddings to edit fragments from edit_smiles column.

    Args:
        df: DataFrame with edit_smiles column (format: "frag_a>>frag_b")
        emb_lookup: Dict mapping SMILES -> embedding
        dataset_name: Name for logging
        edit_smiles_column: Name of column containing edit SMILES (default: 'edit_smiles')

    Returns:
        Tuple of (frag_a_emb, frag_b_emb)

    """
    try:
        from src.data.small_molecule.utils.chemistry import parse_edit_smiles
    except ImportError:
        raise NotImplementedError(
            "Fragment embedding functions require small molecule dependencies "
            "not available in this RNA-only repo"
        )

    print(f"{dataset_name}: Mapping {len(df)} edit fragments to embeddings")

    frag_a_emb_list = []
    frag_b_emb_list = []

    for edit_smiles in df[edit_smiles_column]:
        try:
            frag_a, frag_b = parse_edit_smiles(edit_smiles)
            frag_a_emb_list.append(emb_lookup[frag_a])
            frag_b_emb_list.append(emb_lookup[frag_b])
        except (ValueError, KeyError) as e:
            raise ValueError(f"Failed to map edit_smiles '{edit_smiles}': {e}")

    frag_a_emb = np.array(frag_a_emb_list)
    frag_b_emb = np.array(frag_b_emb_list)

    return frag_a_emb, frag_b_emb
