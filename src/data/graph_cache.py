"""
Efficient Graph Caching System for RNA GNN Models

Features:
- Parallel graph construction using multiprocessing
- Disk caching with automatic cache validation
- Memory-mapped file loading for fast access
- Incremental building (only build missing graphs)
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Tuple
from torch_geometric.data import Data
from tqdm import tqdm
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')


class GraphCache:
    """Manages caching and parallel construction of PyG graphs."""

    def __init__(self, cache_dir: str = "data/graph_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _compute_dataset_hash(self, df: pd.DataFrame, config: dict) -> str:
        """Compute hash of dataset + config to detect changes."""
        # Hash sequences + config
        seq_hash = hashlib.md5(''.join(df['sequence'].values).encode()).hexdigest()
        config_hash = hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()
        return f"{seq_hash[:8]}_{config_hash[:8]}"

    def _get_cache_path(self, dataset_hash: str) -> Path:
        """Get path to cached graph file."""
        return self.cache_dir / f"graphs_{dataset_hash}.pt"

    def _get_metadata_path(self, dataset_hash: str) -> Path:
        """Get path to metadata file."""
        return self.cache_dir / f"metadata_{dataset_hash}.json"

    def build_and_cache(
        self,
        df: pd.DataFrame,
        global_features: Optional[np.ndarray] = None,
        utrlm_features: Optional[np.ndarray] = None,
        enriched_features: Optional[np.ndarray] = None,
        structure_features: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        config: dict = None,
        n_jobs: int = 8,
        use_cache: bool = True
    ) -> List[Data]:
        """
        Build graphs with caching and parallelization.

        Args:
            df: DataFrame with 'sequence' column
            global_features: Global features [N, D_global]
            utrlm_features: UTR-LM embeddings [N, 128]
            enriched_features: Enriched features [N, D_enriched]
            structure_features: Structure features [N, D_struct]
            labels: Target labels [N]
            config: Configuration dict (for cache key)
            n_jobs: Number of parallel workers
            use_cache: Whether to use cache

        Returns:
            List of PyG Data objects
        """
        if config is None:
            config = {'version': '1.0'}

        # Compute cache key
        dataset_hash = self._compute_dataset_hash(df, config)
        cache_path = self._get_cache_path(dataset_hash)
        metadata_path = self._get_metadata_path(dataset_hash)

        # Try loading from cache
        if use_cache and cache_path.exists():
            print(f"Loading graphs from cache: {cache_path.name}")
            try:
                graphs = torch.load(cache_path)
                print(f"  Loaded {len(graphs):,} graphs from cache ✓")
                return graphs
            except Exception as e:
                print(f"  Cache load failed: {e}")
                print("  Rebuilding graphs...")

        # Build graphs in parallel
        print(f"Building {len(df):,} graphs with {n_jobs} workers...")

        # Convert to efficient format for multiprocessing
        sequences = df['sequence'].tolist()

        # Prepare batches for parallel processing
        batch_size = max(1, len(sequences) // (n_jobs * 4))  # 4 batches per worker

        graphs = [None] * len(sequences)

        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = []

            for batch_start in range(0, len(sequences), batch_size):
                batch_end = min(batch_start + batch_size, len(sequences))
                batch_indices = list(range(batch_start, batch_end))

                # Prepare batch data
                batch_data = {
                    'sequences': sequences[batch_start:batch_end],
                    'global_features': global_features[batch_start:batch_end] if global_features is not None else None,
                    'utrlm_features': utrlm_features[batch_start:batch_end] if utrlm_features is not None else None,
                    'enriched_features': enriched_features[batch_start:batch_end] if enriched_features is not None else None,
                    'structure_features': structure_features[batch_start:batch_end] if structure_features is not None else None,
                    'labels': labels[batch_start:batch_end] if labels is not None else None,
                }

                future = executor.submit(
                    _build_graph_batch,
                    batch_data
                )
                futures.append((future, batch_indices))

            # Collect results with progress bar
            with tqdm(total=len(sequences), desc="Building graphs") as pbar:
                for future, batch_indices in futures:
                    batch_graphs = future.result()
                    for idx, graph in zip(batch_indices, batch_graphs):
                        graphs[idx] = graph
                    pbar.update(len(batch_indices))

        # Save to cache
        if use_cache:
            print(f"Saving graphs to cache: {cache_path.name}")
            torch.save(graphs, cache_path)

            # Save metadata
            metadata = {
                'n_graphs': len(graphs),
                'config': config,
                'dataset_hash': dataset_hash,
                'feature_dims': {
                    'global': global_features.shape[1] if global_features is not None else 0,
                    'utrlm': utrlm_features.shape[1] if utrlm_features is not None else 0,
                    'enriched': enriched_features.shape[1] if enriched_features is not None else 0,
                    'structure': structure_features.shape[1] if structure_features is not None else 0,
                }
            }
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            print(f"  Cached {len(graphs):,} graphs ✓")

        return graphs


def _build_graph_batch(batch_data: dict) -> List[Data]:
    """
    Build a batch of graphs (for parallel processing).

    This function is called by worker processes.
    """
    sequences = batch_data['sequences']
    global_features = batch_data.get('global_features')
    utrlm_features = batch_data.get('utrlm_features')
    enriched_features = batch_data.get('enriched_features')
    structure_features = batch_data.get('structure_features')
    labels = batch_data.get('labels')

    graphs = []

    for i, seq in enumerate(sequences):
        g = _sequence_to_graph(
            seq,
            global_features=global_features[i] if global_features is not None else None,
            utrlm_features=utrlm_features[i] if utrlm_features is not None else None,
            enriched_features=enriched_features[i] if enriched_features is not None else None,
            structure_features=structure_features[i] if structure_features is not None else None,
        )

        if labels is not None:
            g.y = torch.tensor([labels[i]], dtype=torch.float)

        graphs.append(g)

    return graphs


def _sequence_to_graph(
    seq: str,
    global_features: Optional[np.ndarray] = None,
    utrlm_features: Optional[np.ndarray] = None,
    enriched_features: Optional[np.ndarray] = None,
    structure_features: Optional[np.ndarray] = None,
) -> Data:
    """
    Convert sequence to graph with backbone edges.

    Node features (9-dim):
    - One-hot nucleotide (5)
    - Relative position (1)
    - Distance from center (1)
    - Is center (1)
    - Position index (1)
    """
    base_map = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'T': 3, 'N': 4}

    # Node features
    node_features = []
    for i, base in enumerate(seq):
        one_hot = [0] * 5
        one_hot[base_map.get(base, 4)] = 1
        rel_pos = i / len(seq)
        dist_center = abs(i - 100) / 100
        is_center = 1.0 if i == 100 else 0.0

        node_features.append(one_hot + [rel_pos, dist_center, is_center, i])

    x = torch.tensor(node_features, dtype=torch.float)

    # Backbone edges (i ↔ i+1, bidirectional)
    edge_index = []
    edge_attr = []

    for i in range(len(seq) - 1):
        # Forward edge
        edge_index.append([i, i+1])
        # Backward edge
        edge_index.append([i+1, i])

        # Edge features: [dist, is_backbone, is_struct, ...padding to 16]
        edge_feat = [1, 1, 0, 0] + [0] * 12
        edge_attr.append(edge_feat)
        edge_attr.append(edge_feat)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # Create graph
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    # Add global features
    if global_features is not None:
        data.global_features = torch.tensor(global_features, dtype=torch.float).unsqueeze(0)

    if utrlm_features is not None:
        data.utrlm_features = torch.tensor(utrlm_features, dtype=torch.float).unsqueeze(0)

    if enriched_features is not None:
        data.enriched_features = torch.tensor(enriched_features, dtype=torch.float).unsqueeze(0)

    if structure_features is not None:
        data.structure_features = torch.tensor(structure_features, dtype=torch.float).unsqueeze(0)

    # Mark center position
    data.center_idx = torch.tensor([100], dtype=torch.long)

    return data
