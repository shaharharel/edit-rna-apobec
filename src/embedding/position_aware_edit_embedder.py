"""
Position-Aware RNA Edit Embedder.

This embedder creates edit representations by combining:
1. One-hot edit features (position, edit type, context type)
2. Per-token RNA-FM embeddings at the edit position (before and after)

The key insight is that RNA-FM provides rich per-token representations,
and the embedding at the edit position captures local sequence context
that's relevant for predicting edit effects.

Architecture:
    [one_hot | pos_emb_a | pos_emb_b | delta_emb] -> MLP -> edit_embedding

Where:
- one_hot: Position encoding + edit type one-hot + context type one-hot
- pos_emb_a: RNA-FM embedding at edit position in sequence A
- pos_emb_b: RNA-FM embedding at edit position in sequence B
- delta_emb: pos_emb_b - pos_emb_a (explicit difference)

This approach:
- Uses pretrained RNA-FM knowledge about local sequence context
- Captures how the edit changes the local representation
- Includes explicit edit type and position information
- Is trainable end-to-end with the predictor
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Union, List, Optional, Tuple, Dict


# Edit type mapping for RNA edits
EDIT_TYPES = {
    ('A', 'C'): 0, ('A', 'G'): 1, ('A', 'U'): 2, ('A', 'M'): 3,
    ('C', 'A'): 4, ('C', 'G'): 5, ('C', 'U'): 6,
    ('G', 'A'): 7, ('G', 'C'): 8, ('G', 'U'): 9,
    ('U', 'A'): 10, ('U', 'C'): 11, ('U', 'G'): 12,
    ('M', 'A'): 13,  # m6A to A
}


class PositionAwareEditEmbedder(nn.Module):
    """
    Position-aware edit embedder using RNA-FM per-token embeddings.

    Combines one-hot edit features with RNA-FM embeddings at the edit
    position to create rich edit representations.

    Args:
        onehot_dim: Dimension of one-hot features (position + edit type + context)
        seq_embed_dim: Dimension of sequence embeddings (640 for RNA-FM)
        hidden_dims: Hidden layer dimensions for fusion MLP
        output_dim: Output edit embedding dimension
        dropout: Dropout probability

    Input modes:
        1. Tuple (onehot, pos_emb_a, pos_emb_b) - precomputed position embeddings
        2. Dict with 'onehot', 'pos_emb_a', 'pos_emb_b' keys

    Example:
        >>> embedder = PositionAwareEditEmbedder(onehot_dim=24, seq_embed_dim=640)
        >>> edit_emb = embedder((onehot_features, pos_emb_a, pos_emb_b))
        >>> print(edit_emb.shape)  # [batch, output_dim]
    """

    def __init__(
        self,
        onehot_dim: int,
        seq_embed_dim: int = 640,
        hidden_dims: List[int] = None,
        output_dim: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128]

        self.onehot_dim = onehot_dim
        self.seq_embed_dim = seq_embed_dim
        self.output_dim = output_dim

        # Input: one-hot + pos_emb_a + pos_emb_b + delta_emb
        # delta_emb = pos_emb_b - pos_emb_a (explicit difference)
        total_input = onehot_dim + 3 * seq_embed_dim

        # Build MLP
        layers = []
        prev_dim = total_input

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

        # Store dimensions for inspection
        self._component_dims = {
            'onehot': onehot_dim,
            'pos_emb_a': seq_embed_dim,
            'pos_emb_b': seq_embed_dim,
            'delta_emb': seq_embed_dim,
            'total_input': total_input,
            'output': output_dim
        }

    def forward(
        self,
        inputs: Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """
        Compute position-aware edit embedding.

        Args:
            inputs: Either a tuple (onehot, pos_emb_a, pos_emb_b) or
                   a dict with 'onehot', 'pos_emb_a', 'pos_emb_b' keys

        Returns:
            Edit embeddings [batch, output_dim]
        """
        if isinstance(inputs, tuple):
            onehot, pos_emb_a, pos_emb_b = inputs
        elif isinstance(inputs, dict):
            onehot = inputs['onehot']
            pos_emb_a = inputs['pos_emb_a']
            pos_emb_b = inputs['pos_emb_b']
        else:
            raise ValueError(f"Expected tuple or dict, got {type(inputs)}")

        # Compute explicit difference
        delta_emb = pos_emb_b - pos_emb_a

        # Concatenate all features
        x = torch.cat([onehot, pos_emb_a, pos_emb_b, delta_emb], dim=-1)

        return self.mlp(x)

    @property
    def embedding_dim(self) -> int:
        """Return the output embedding dimension."""
        return self.output_dim

    @property
    def component_dims(self) -> Dict[str, int]:
        """Return dimensions of each component."""
        return self._component_dims


def create_edit_features(
    df,
    max_position: int = 7,
    include_edit_type: bool = True,
    include_context: bool = True
) -> np.ndarray:
    """
    Create one-hot edit features from a DataFrame.

    Args:
        df: DataFrame with columns:
            - edit_position: Position of edit (0-indexed)
            - nuc_a: Nucleotide in sequence A at edit position
            - nuc_b: Nucleotide in sequence B at edit position
            - pair_type: (optional) Context type (A_vs_A, M_vs_M, M_vs_A)
        max_position: Maximum position for one-hot encoding
        include_edit_type: Include edit type one-hot (14 dims)
        include_context: Include context type one-hot (3 dims)

    Returns:
        One-hot feature array [n_samples, feature_dim]
    """
    n_samples = len(df)

    # Position one-hot
    positions = df['edit_position'].values if 'edit_position' in df.columns else np.zeros(n_samples)
    pos_onehot = np.zeros((n_samples, max_position), dtype=np.float32)
    for i, pos in enumerate(positions):
        pos = int(min(pos, max_position - 1))
        pos_onehot[i, pos] = 1.0

    features = [pos_onehot]

    # Edit type one-hot
    if include_edit_type:
        edit_onehot = np.zeros((n_samples, len(EDIT_TYPES)), dtype=np.float32)

        if 'nuc_a' in df.columns and 'nuc_b' in df.columns:
            for i, (nuc_a, nuc_b) in enumerate(zip(df['nuc_a'], df['nuc_b'])):
                nuc_a = str(nuc_a).upper().replace('T', 'U')
                nuc_b = str(nuc_b).upper().replace('T', 'U')
                key = (nuc_a, nuc_b)
                if key in EDIT_TYPES:
                    edit_onehot[i, EDIT_TYPES[key]] = 1.0

        features.append(edit_onehot)

    # Context type one-hot
    if include_context and 'pair_type' in df.columns:
        context_types = ['A_vs_A', 'M_vs_M', 'M_vs_A']
        context_onehot = np.zeros((n_samples, len(context_types)), dtype=np.float32)

        for i, pt in enumerate(df['pair_type']):
            if pt in context_types:
                context_onehot[i, context_types.index(pt)] = 1.0

        features.append(context_onehot)

    return np.concatenate(features, axis=1)


def get_position_embeddings(
    df,
    embedder,
    position_col: str = 'edit_position',
    seq_a_col: str = 'seq_a',
    seq_b_col: str = 'seq_b'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract RNA-FM embeddings at edit positions.

    Args:
        df: DataFrame with sequence and position columns
        embedder: RNA-FM embedder with encode(seqs, return_per_token=True)
        position_col: Column name for edit positions
        seq_a_col: Column name for sequence A
        seq_b_col: Column name for sequence B

    Returns:
        (pos_emb_a, pos_emb_b): Position embeddings for each sequence
            Shape: [n_samples, embed_dim] (typically 640 for RNA-FM)
    """
    seqs_a = df[seq_a_col].tolist()
    seqs_b = df[seq_b_col].tolist()
    positions = df[position_col].values

    # Get per-token embeddings
    per_token_a = embedder.encode(seqs_a, return_per_token=True)
    per_token_b = embedder.encode(seqs_b, return_per_token=True)

    embed_dim = per_token_a.shape[-1]

    # Extract embedding at edit position
    pos_emb_a = np.zeros((len(df), embed_dim), dtype=np.float32)
    pos_emb_b = np.zeros((len(df), embed_dim), dtype=np.float32)

    for i, pos in enumerate(positions):
        pos = int(min(pos, per_token_a.shape[1] - 1))
        pos_emb_a[i] = per_token_a[i, pos]
        pos_emb_b[i] = per_token_b[i, pos]

    return pos_emb_a, pos_emb_b
