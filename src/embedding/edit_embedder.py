"""
RNA edit embedding via difference of sequence embeddings.

Computes edit embeddings for RNA sequences as:
    edit_embedding = embedding(seq_b) - embedding(seq_a)

This approach captures the direction and magnitude of change
in embedding space caused by a sequence edit.
"""

import numpy as np
from typing import Union, List, Optional, Dict
from .base import RNAEmbedder


class RNAEditEmbedder:
    """
    RNA edit embedder using difference of sequence embeddings.

    **Two Modes:**

    Mode 1 (use_local_context=False): Full Sequence Embeddings
        edit = Embed(full_seq_b) - Embed(full_seq_a)
        - Context-aware: captures global sequence effects
        - Works with any sequence pair

    Mode 2 (use_local_context=True): Local Context Embeddings
        edit = Embed(local_context_b) - Embed(local_context_a)
        - Position-specific: focuses on local effects
        - May generalize better across different backgrounds

    Additional features can be included:
    - Edit type encoding (SNV, insertion, deletion)
    - Position encoding (where in the sequence)
    - Motif features (Kozak score change, uAUG introduction, etc.)

    This approach:
    - Works with ANY RNA embedder (Nucleotide, RNA-FM, RNABERT)
    - Preserves the same dimensionality as sequence embeddings
    - Captures the direction of change in embedding space
    - Compatible with EditEffectPredictor

    Args:
        rna_embedder: Any RNAEmbedder instance
        use_local_context: If True, embed local context around edit (Mode 2)
        context_window: Size of local context window (nucleotides on each side)
        include_edit_features: If True, append edit-specific features

    Example:
        >>> from src.embedding import NucleotideEmbedder, RNAEditEmbedder
        >>> rna_emb = NucleotideEmbedder()
        >>> edit_emb = RNAEditEmbedder(rna_emb, use_local_context=False)
        >>> edit_vec = edit_emb.encode_from_sequences('AUGCAUGC', 'AUGGAUGC')
        >>> print(edit_vec.shape)  # Same as rna_emb.embedding_dim
    """

    def __init__(
        self,
        rna_embedder: RNAEmbedder,
        use_local_context: bool = False,
        context_window: int = 25,
        include_edit_features: bool = False
    ):
        self.rna_embedder = rna_embedder
        self.use_local_context = use_local_context
        self.context_window = context_window
        self.include_edit_features = include_edit_features

        # Calculate embedding dimension
        base_dim = rna_embedder.embedding_dim

        if include_edit_features:
            # Edit type (4: SNV, insertion, deletion, complex)
            # Position encoding (1: normalized position)
            # Edit size (1: log-scaled)
            # Motif features (3: kozak_change, uaug_introduced, gc_change)
            self._edit_feature_dim = 9
            self._embedding_dim = base_dim + self._edit_feature_dim
        else:
            self._edit_feature_dim = 0
            self._embedding_dim = base_dim

    def encode_from_sequences(
        self,
        seq_a: Union[str, List[str]],
        seq_b: Union[str, List[str]],
        edit_info: Optional[Union[Dict, List[Dict]]] = None
    ) -> np.ndarray:
        """
        Encode edit(s) from sequence A and sequence B.

        Args:
            seq_a: Reference sequence(s)
            seq_b: Variant sequence(s)
            edit_info: Optional pre-computed edit information (from extract_edit)

        Returns:
            Edit embedding(s) as numpy array
            - Single pair: shape (embedding_dim,)
            - Multiple pairs: shape (n_edits, embedding_dim)
        """
        # Handle single vs batch
        if isinstance(seq_a, str):
            assert isinstance(seq_b, str), "seq_a and seq_b must both be strings or lists"
            seq_a_list = [seq_a]
            seq_b_list = [seq_b]
            edit_info_list = [edit_info] if edit_info else None
            return_single = True
        else:
            assert len(seq_a) == len(seq_b), "seq_a and seq_b lists must have same length"
            seq_a_list = list(seq_a)
            seq_b_list = list(seq_b)
            edit_info_list = list(edit_info) if edit_info else None
            return_single = False

        # Get sequences to embed
        if self.use_local_context:
            # Mode 2: Extract and embed local context
            contexts_a = []
            contexts_b = []

            for i, (sa, sb) in enumerate(zip(seq_a_list, seq_b_list)):
                info = edit_info_list[i] if edit_info_list else None
                ctx_a, ctx_b = self._extract_local_context(sa, sb, info)
                contexts_a.append(ctx_a)
                contexts_b.append(ctx_b)

            # Encode local contexts
            emb_a = self.rna_embedder.encode(contexts_a)
            emb_b = self.rna_embedder.encode(contexts_b)
        else:
            # Mode 1: Encode full sequences
            emb_a = self.rna_embedder.encode(seq_a_list)
            emb_b = self.rna_embedder.encode(seq_b_list)

        # Ensure 2D
        if emb_a.ndim == 1:
            emb_a = emb_a.reshape(1, -1)
            emb_b = emb_b.reshape(1, -1)

        # Compute difference
        edit_emb = emb_b - emb_a

        # Add edit features if requested
        if self.include_edit_features:
            edit_features = []
            for i, (sa, sb) in enumerate(zip(seq_a_list, seq_b_list)):
                info = edit_info_list[i] if edit_info_list else None
                features = self._compute_edit_features(sa, sb, info)
                edit_features.append(features)

            edit_features = np.array(edit_features, dtype=np.float32)
            edit_emb = np.concatenate([edit_emb, edit_features], axis=1)

        if return_single:
            return edit_emb[0]
        return edit_emb

    def encode_from_pair_df(self, pairs_df) -> np.ndarray:
        """
        Encode edits from pairs DataFrame.

        Args:
            pairs_df: DataFrame with 'seq_a' and 'seq_b' columns

        Returns:
            Edit embeddings array of shape (n_pairs, embedding_dim)
        """
        return self.encode_from_sequences(
            pairs_df['seq_a'].tolist(),
            pairs_df['seq_b'].tolist()
        )

    def _extract_local_context(
        self,
        seq_a: str,
        seq_b: str,
        edit_info: Optional[Dict] = None
    ) -> tuple:
        """
        Extract local context around the edit position.

        Returns:
            (context_a, context_b) tuple of context sequences
        """
        from src.data.sequence_utils import extract_edit

        # Get edit position
        if edit_info is None:
            edit_info = extract_edit(seq_a, seq_b)

        pos = edit_info.get('position', 0)

        # Extract windows
        start_a = max(0, pos - self.context_window)
        end_a = min(len(seq_a), pos + self.context_window + 1)

        start_b = max(0, pos - self.context_window)
        end_b = min(len(seq_b), pos + self.context_window + 1)

        context_a = seq_a[start_a:end_a]
        context_b = seq_b[start_b:end_b]

        # Pad to consistent length if needed
        target_len = 2 * self.context_window + 1

        if len(context_a) < target_len:
            context_a = context_a + 'N' * (target_len - len(context_a))
        if len(context_b) < target_len:
            context_b = context_b + 'N' * (target_len - len(context_b))

        return context_a, context_b

    def _compute_edit_features(
        self,
        seq_a: str,
        seq_b: str,
        edit_info: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Compute additional edit-specific features.

        Features:
        - Edit type one-hot (4 dims: SNV, insertion, deletion, complex)
        - Normalized position (1 dim)
        - Log edit size (1 dim)
        - Kozak score change (1 dim)
        - uAUG introduced (1 dim: 0 or 1)
        - GC content change (1 dim)

        Returns:
            Feature array of shape (9,)
        """
        from src.data.sequence_utils import (
            extract_edit, compute_kozak_score, find_uaugs, gc_content
        )

        features = np.zeros(9, dtype=np.float32)

        # Get edit info
        if edit_info is None:
            edit_info = extract_edit(seq_a, seq_b)

        # Edit type one-hot
        edit_type = edit_info.get('edit_type', 'unknown')
        type_mapping = {'SNV': 0, 'insertion': 1, 'deletion': 2, 'complex': 3}
        type_idx = type_mapping.get(edit_type, 3)
        features[type_idx] = 1.0

        # Normalized position
        pos = edit_info.get('position', 0)
        max_len = max(len(seq_a), len(seq_b), 1)
        features[4] = pos / max_len

        # Log edit size
        edit_size = edit_info.get('edit_size', 1)
        features[5] = np.log1p(edit_size) / 5.0  # Normalize

        # Kozak score change (if there's an AUG in the sequence)
        uaugs_a = find_uaugs(seq_a)
        uaugs_b = find_uaugs(seq_b)

        if uaugs_a:
            kozak_a = max(compute_kozak_score(seq_a, p) for p in uaugs_a)
        else:
            kozak_a = 0

        if uaugs_b:
            kozak_b = max(compute_kozak_score(seq_b, p) for p in uaugs_b)
        else:
            kozak_b = 0

        features[6] = kozak_b - kozak_a

        # uAUG introduced
        new_uaugs = len(uaugs_b) - len(uaugs_a)
        features[7] = 1.0 if new_uaugs > 0 else (0.0 if new_uaugs == 0 else -1.0)

        # GC content change
        gc_a = gc_content(seq_a)
        gc_b = gc_content(seq_b)
        features[8] = gc_b - gc_a

        return features

    @property
    def embedding_dim(self) -> int:
        """Return the dimensionality of edit embeddings."""
        return self._embedding_dim

    @property
    def name(self) -> str:
        """Return the name of this edit embedding method."""
        mode = "local" if self.use_local_context else "full"
        feat = "_features" if self.include_edit_features else ""
        return f"rna_edit_diff_{self.rna_embedder.name}_{mode}{feat}"


class TrainableRNAEditEmbedder:
    """
    Trainable RNA edit embedder with learnable transformation.

    Learns to transform (seq_a_emb, seq_b_emb) → edit_emb through
    an MLP that can be trained end-to-end with the predictor.

    Architecture:
        [seq_a_emb | seq_b_emb] → MLP → edit_emb

    Or with additional gating:
        [seq_a_emb | seq_b_emb | seq_b_emb - seq_a_emb] → MLP → edit_emb

    Args:
        rna_dim: Dimension of RNA embeddings
        edit_dim: Output edit embedding dimension
        hidden_dims: Hidden layer dimensions for MLP
        dropout: Dropout probability
        use_difference: If True, include explicit difference in input
    """

    def __init__(
        self,
        rna_dim: int,
        edit_dim: Optional[int] = None,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        use_difference: bool = True
    ):
        import torch
        import torch.nn as nn

        self.rna_dim = rna_dim
        self.edit_dim = edit_dim if edit_dim is not None else rna_dim
        self.hidden_dims = hidden_dims if hidden_dims is not None else [512, 256]
        self.use_difference = use_difference

        # Input dimension
        if use_difference:
            input_dim = rna_dim * 3  # seq_a, seq_b, difference
        else:
            input_dim = rna_dim * 2  # seq_a, seq_b

        # Build MLP
        layers = []
        prev_dim = input_dim

        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, self.edit_dim))

        self.network = nn.Sequential(*layers)

    def forward(
        self,
        seq_a_emb: 'torch.Tensor',
        seq_b_emb: 'torch.Tensor'
    ) -> 'torch.Tensor':
        """
        Compute trainable edit embedding.

        Args:
            seq_a_emb: Reference sequence embeddings [batch, rna_dim]
            seq_b_emb: Variant sequence embeddings [batch, rna_dim]

        Returns:
            Edit embeddings [batch, edit_dim]
        """
        import torch

        if self.use_difference:
            diff = seq_b_emb - seq_a_emb
            x = torch.cat([seq_a_emb, seq_b_emb, diff], dim=-1)
        else:
            x = torch.cat([seq_a_emb, seq_b_emb], dim=-1)

        return self.network(x)

    @property
    def embedding_dim(self) -> int:
        """Return the edit embedding dimension."""
        return self.edit_dim
