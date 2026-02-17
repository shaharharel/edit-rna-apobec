"""
RNA secondary structure graph construction utilities.

This module provides functions for building graph representations from
RNA secondary structures (dot-bracket notation) for use with GNNs.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict

try:
    import torch
    from torch_geometric.data import Data, Batch
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False


def parse_dot_bracket(structure: str) -> List[Tuple[int, int]]:
    """
    Parse dot-bracket notation to extract base pairs.

    Args:
        structure: Dot-bracket string (e.g., "(((...)))")

    Returns:
        List of (i, j) base pair tuples (0-indexed)
    """
    pairs = []
    stack = []

    for i, char in enumerate(structure):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if stack:
                j = stack.pop()
                pairs.append((j, i))

    return pairs


def compute_structure_features(
    structure: str,
    window: int = 10
) -> Dict[str, np.ndarray]:
    """
    Compute per-position features from dot-bracket structure.

    Args:
        structure: Dot-bracket notation
        window: Window size for local features

    Returns:
        Dictionary with:
        - pairing: Binary array (1 if paired, 0 if unpaired)
        - in_stem: Binary array (1 if in stem region)
        - in_loop: Binary array (1 if in loop/bulge)
        - local_pairing_density: Fraction of paired positions in local window
    """
    n = len(structure)
    pairing = np.zeros(n)
    in_stem = np.zeros(n)

    # Parse structure
    pairs = parse_dot_bracket(structure)
    pair_dict = {}
    for i, j in pairs:
        pair_dict[i] = j
        pair_dict[j] = i
        pairing[i] = 1.0
        pairing[j] = 1.0

    # Detect stem regions (consecutive base pairs)
    for i, j in pairs:
        # Check if neighbors also pair with each other
        if i > 0 and j < n - 1:
            if i - 1 in pair_dict and pair_dict[i - 1] == j + 1:
                in_stem[i] = 1.0
                in_stem[j] = 1.0
        if i < n - 1 and j > 0:
            if i + 1 in pair_dict and pair_dict[i + 1] == j - 1:
                in_stem[i] = 1.0
                in_stem[j] = 1.0

    # Loop regions = paired but not in stem
    in_loop = pairing * (1 - in_stem)

    # Local pairing density
    local_density = np.zeros(n)
    for i in range(n):
        start = max(0, i - window)
        end = min(n, i + window + 1)
        local_density[i] = pairing[start:end].mean()

    return {
        'pairing': pairing,
        'in_stem': in_stem,
        'in_loop': in_loop,
        'local_pairing_density': local_density
    }


def build_rna_graph(
    sequence: str,
    structure: str,
    include_sequence_edges: bool = True,
    center_position: Optional[int] = None,
    additional_node_features: Optional[np.ndarray] = None
) -> 'Data':
    """
    Build PyTorch Geometric graph from RNA sequence and structure.

    Args:
        sequence: RNA sequence (ACGU)
        structure: Dot-bracket structure notation
        include_sequence_edges: Add edges between sequential neighbors
        center_position: Position of interest (e.g., m6A site) for relative encoding
        additional_node_features: Optional (n, d) array of extra node features

    Returns:
        PyG Data object with:
        - x: Node features
        - edge_index: Edge connectivity
        - edge_type: Edge types (0=structure, 1=sequence)

    Raises:
        ImportError: If torch_geometric is not installed
    """
    if not HAS_TORCH_GEOMETRIC:
        raise ImportError(
            "torch_geometric required for graph construction. "
            "Install with: pip install torch-geometric"
        )

    n = len(sequence)

    # Node features: one-hot nucleotides (5-dim: A, C, G, U, N)
    nuc_to_idx = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'T': 3, 'N': 4}
    x = torch.zeros(n, 5)
    for i, nuc in enumerate(sequence.upper()):
        x[i, nuc_to_idx.get(nuc, 4)] = 1.0

    # Add relative position encoding if center provided
    if center_position is not None:
        rel_pos = torch.zeros(n, 1)
        for i in range(n):
            rel_pos[i] = (i - center_position) / (n / 2)  # Normalize to [-1, 1]
        x = torch.cat([x, rel_pos], dim=-1)

    # Add structure-derived features
    struct_feats = compute_structure_features(structure)
    struct_tensor = torch.tensor(np.stack([
        struct_feats['pairing'],
        struct_feats['in_stem'],
        struct_feats['local_pairing_density']
    ], axis=1), dtype=torch.float32)
    x = torch.cat([x, struct_tensor], dim=-1)

    # Add any additional features
    if additional_node_features is not None:
        add_feats = torch.tensor(additional_node_features, dtype=torch.float32)
        x = torch.cat([x, add_feats], dim=-1)

    # Build edges
    edge_list = []
    edge_types = []

    # Structure edges (base pairs) - bidirectional
    pairs = parse_dot_bracket(structure)
    for i, j in pairs:
        edge_list.append((i, j))
        edge_list.append((j, i))
        edge_types.extend([0, 0])

    # Sequence edges (backbone neighbors) - bidirectional
    if include_sequence_edges:
        for i in range(n - 1):
            edge_list.append((i, i + 1))
            edge_list.append((i + 1, i))
            edge_types.extend([1, 1])

    # Convert to tensors
    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_type = torch.tensor(edge_types, dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_type = torch.zeros(0, dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_type=edge_type)


def batch_build_graphs(
    sequences: List[str],
    structures: List[str],
    include_sequence_edges: bool = True,
    center_positions: Optional[List[int]] = None
) -> 'Batch':
    """
    Build a batch of graphs from multiple sequences/structures.

    Args:
        sequences: List of RNA sequences
        structures: List of dot-bracket structures
        include_sequence_edges: Add sequence neighbor edges
        center_positions: Optional list of center positions

    Returns:
        PyG Batch object
    """
    if not HAS_TORCH_GEOMETRIC:
        raise ImportError("torch_geometric required")

    graphs = []
    for i, (seq, struct) in enumerate(zip(sequences, structures)):
        center = center_positions[i] if center_positions else None
        graph = build_rna_graph(seq, struct, include_sequence_edges, center)
        graphs.append(graph)

    return Batch.from_data_list(graphs)


class StructureGraphBuilder:
    """
    Builder class for creating RNA structure graphs with caching.

    Useful when processing large datasets where the same sequence
    may appear multiple times.
    """

    def __init__(
        self,
        include_sequence_edges: bool = True,
        cache_enabled: bool = True
    ):
        self.include_sequence_edges = include_sequence_edges
        self.cache_enabled = cache_enabled
        self._cache: Dict[str, Data] = {}

    def build(
        self,
        sequence: str,
        structure: str,
        center_position: Optional[int] = None
    ) -> 'Data':
        """Build graph with optional caching."""
        cache_key = f"{sequence}_{structure}_{center_position}"

        if self.cache_enabled and cache_key in self._cache:
            return self._cache[cache_key]

        graph = build_rna_graph(
            sequence, structure,
            self.include_sequence_edges,
            center_position
        )

        if self.cache_enabled:
            self._cache[cache_key] = graph

        return graph

    def clear_cache(self):
        """Clear the graph cache."""
        self._cache.clear()

    def __call__(
        self,
        sequence: str,
        structure: str,
        center_position: Optional[int] = None
    ) -> 'Data':
        """Alias for build()."""
        return self.build(sequence, structure, center_position)
