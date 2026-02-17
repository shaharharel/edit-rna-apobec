"""
Structured RNA Edit Embedder.

A sophisticated edit embedding module that captures:
1. Mutation type (12 SNV types)
2. Mutation effect (Δ token embedding from pretrained LM)
3. Position encoding (sinusoidal + learned)
4. Local context (window ±10nt around edit)
5. Attention context (attention-weighted token sum)
6. (Optional) Δ-structure features (pairing prob, accessibility, entropy, MFE)

This embedder takes sequence A and edit information (position, from, to)
and produces a rich edit representation WITHOUT needing sequence B.

Can be instantiated with RNA-FM or UTR-LM as the base embedder.
"""

import math
import numpy as np
import torch
import torch.nn as nn
from typing import Union, List, Optional, Tuple, Dict


# Mutation type mapping: 12 possible SNV transitions
MUTATION_TYPES = {
    ('A', 'C'): 0, ('A', 'G'): 1, ('A', 'U'): 2,
    ('C', 'A'): 3, ('C', 'G'): 4, ('C', 'U'): 5,
    ('G', 'A'): 6, ('G', 'C'): 7, ('G', 'U'): 8,
    ('U', 'A'): 9, ('U', 'C'): 10, ('U', 'G'): 11,
}

# Nucleotide to index for token embeddings
NUC_TO_IDX = {'A': 0, 'C': 1, 'G': 2, 'U': 3}


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding as in Transformer.
    Encodes both absolute position and relative position (pos/seq_len).
    """

    def __init__(self, dim: int, max_len: int = 1024):
        super().__init__()
        self.dim = dim

        # Create sinusoidal encoding matrix
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))

        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            positions: [batch] tensor of integer positions

        Returns:
            [batch, dim] positional encodings
        """
        return self.pe[positions]


class StructuredRNAEditEmbedder(nn.Module):
    """
    Structured RNA Edit Embedder.

    Generates rich edit embeddings from:
    - Sequence A (the original sequence)
    - Edit position (0-indexed)
    - Edit from/to nucleotides

    Does NOT require sequence B, making it suitable for pure edit modeling.

    Architecture:
    1. Mutation type embedding: Learned embedding for 12 SNV types
    2. Mutation effect: Δ of pretrained nucleotide token embeddings
    3. Position encoding: Sinusoidal + learned
    4. Local context: Mean-pooled window around edit site
    5. Attention context: Attention-weighted token sum
    6. (Optional) Structure features: Δ-pairing, Δ-accessibility, Δ-entropy, Δ-MFE
    7. Fusion MLP: Combines all components

    Args:
        base_embedder: Base embedder (RNAFMEmbedder or UTRLMEmbedder)
        mutation_type_dim: Dimension of mutation type embedding (default: 64)
        mutation_effect_dim: Dimension of projected mutation effect (default: 256)
        position_dim: Total position encoding dimension (default: 64)
        local_context_dim: Dimension of local context (default: 256)
        attention_context_dim: Dimension of attention context (default: 128)
        structure_feature_dim: Dimension of structure features (default: 64)
        fusion_hidden_dims: Hidden dims for fusion MLP (default: [512, 384])
        output_dim: Final edit embedding dimension (default: 256)
        window_size: Number of nucleotides on each side of edit (default: 10)
        dropout: Dropout probability (default: 0.1)
        max_seq_len: Maximum sequence length (default: 512)
        use_structure: Whether to use structure features (default: False)
        structure_predictor: Optional RNAplfold predictor for structure features
    """

    def __init__(
        self,
        base_embedder: nn.Module,
        mutation_type_dim: int = 64,
        mutation_effect_dim: int = 256,
        position_dim: int = 64,
        local_context_dim: int = 256,
        attention_context_dim: int = 128,
        structure_feature_dim: int = 64,
        fusion_hidden_dims: List[int] = None,
        output_dim: int = 256,
        window_size: int = 10,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        use_structure: bool = False,
        structure_predictor: Optional[nn.Module] = None
    ):
        super().__init__()

        self.base_embedder = base_embedder
        self.base_dim = base_embedder.embedding_dim
        self.window_size = window_size
        self.output_dim = output_dim
        self.use_structure = use_structure
        self.structure_predictor = structure_predictor

        # Detect embedder type for token-level access
        self._embedder_type = self._detect_embedder_type()

        if fusion_hidden_dims is None:
            fusion_hidden_dims = [512, 384]

        # ========================================
        # 1. Mutation Type Embedding
        # ========================================
        self.mutation_type_embed = nn.Embedding(12, mutation_type_dim)

        # ========================================
        # 2. Mutation Effect (Δ token embedding)
        # ========================================
        self.nucleotide_embed = nn.Embedding(4, self.base_dim)
        self.mutation_effect_proj = nn.Sequential(
            nn.Linear(self.base_dim, mutation_effect_dim),
            nn.LayerNorm(mutation_effect_dim),
            nn.ReLU()
        )

        # ========================================
        # 3. Position Encoding
        # ========================================
        sin_dim = position_dim // 2
        self.sinusoidal_pos = SinusoidalPositionalEncoding(sin_dim, max_seq_len)
        learned_dim = position_dim - sin_dim
        self.learned_pos_embed = nn.Embedding(max_seq_len, learned_dim)
        self.relative_pos_proj = nn.Linear(1, position_dim // 4)
        self.position_dim = sin_dim + learned_dim + position_dim // 4

        # ========================================
        # 4. Local Context (window ±N around edit)
        # ========================================
        self.local_context_proj = nn.Sequential(
            nn.Linear(self.base_dim, local_context_dim),
            nn.LayerNorm(local_context_dim),
            nn.ReLU()
        )

        # ========================================
        # 5. Attention Context
        # ========================================
        self.attention_context_proj = nn.Sequential(
            nn.Linear(self.base_dim, attention_context_dim),
            nn.LayerNorm(attention_context_dim),
            nn.ReLU()
        )

        # ========================================
        # 6. Structure Features (Optional)
        # ========================================
        if use_structure:
            self.structure_feature_dim = structure_feature_dim
            # 7 raw features: Δ_pairing, Δ_accessibility, Δ_entropy, Δ_mfe,
            # Δ_local_pairing, Δ_local_accessibility, local_pairing_std
            self.structure_proj = nn.Sequential(
                nn.Linear(7, 32),
                nn.LayerNorm(32),
                nn.ReLU(),
                nn.Linear(32, structure_feature_dim),
                nn.LayerNorm(structure_feature_dim),
                nn.ReLU()
            )
        else:
            self.structure_feature_dim = 0

        # ========================================
        # 7. Fusion MLP
        # ========================================
        raw_dim = (
            mutation_type_dim +
            mutation_effect_dim +
            self.position_dim +
            local_context_dim +
            attention_context_dim +
            self.structure_feature_dim
        )

        fusion_layers = []
        prev_dim = raw_dim

        for hidden_dim in fusion_hidden_dims:
            fusion_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        fusion_layers.extend([
            nn.Linear(prev_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU()
        ])

        self.fusion_mlp = nn.Sequential(*fusion_layers)

        # Store dimensions
        self._component_dims = {
            'mutation_type': mutation_type_dim,
            'mutation_effect': mutation_effect_dim,
            'position': self.position_dim,
            'local_context': local_context_dim,
            'attention_context': attention_context_dim,
            'structure': self.structure_feature_dim,
            'raw_total': raw_dim,
            'output': output_dim
        }

        self._init_weights()

    def _detect_embedder_type(self) -> str:
        """Detect the type of base embedder."""
        class_name = self.base_embedder.__class__.__name__
        if 'RNAFM' in class_name:
            return 'rnafm'
        elif 'UTRLM' in class_name:
            return 'utrlm'
        elif 'RNABERT' in class_name:
            return 'rnabert'
        else:
            return 'unknown'

    def _init_weights(self):
        """Initialize weights with reasonable defaults."""
        with torch.no_grad():
            nn.init.xavier_uniform_(self.nucleotide_embed.weight)
            nn.init.xavier_uniform_(self.mutation_type_embed.weight)

    def _get_token_embeddings(
        self,
        sequences: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get token-level embeddings and attention weights from base embedder.

        Returns:
            token_embeddings: [batch, seq_len, embed_dim]
            attention_weights: [batch, seq_len, seq_len]
        """
        if self._embedder_type == 'rnafm':
            return self._get_rnafm_token_embeddings(sequences)
        elif self._embedder_type == 'utrlm':
            return self._get_utrlm_token_embeddings(sequences)
        elif self._embedder_type == 'rnabert':
            return self._get_rnabert_token_embeddings(sequences)
        else:
            # Fallback: use sequence-level embedding replicated
            return self._get_fallback_token_embeddings(sequences)

    def _get_rnafm_token_embeddings(
        self,
        sequences: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get RNA-FM token embeddings and attention weights."""
        data = [(f"seq_{i}", seq) for i, seq in enumerate(sequences)]
        _, _, batch_tokens = self.base_embedder.batch_converter(data)
        batch_tokens = batch_tokens.to(next(self.base_embedder.model.parameters()).device)

        with torch.no_grad():
            results = self.base_embedder.model(batch_tokens, repr_layers=[12], need_head_weights=True)
            token_embeddings = results["representations"][12]
            attentions = results["attentions"]

            if isinstance(attentions, torch.Tensor):
                attention_weights = attentions[:, -1, :, :, :].mean(dim=1)
            else:
                attention_weights = attentions[-1].mean(dim=1)

        return token_embeddings, attention_weights

    def _get_utrlm_token_embeddings(
        self,
        sequences: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get UTR-LM token embeddings and attention weights."""
        with torch.no_grad():
            # Get token-level embeddings
            token_embeddings = self.base_embedder(sequences, return_all_tokens=True)

            # Get attention weights if available
            if hasattr(self.base_embedder, 'get_attention_weights'):
                attention_weights = self.base_embedder.get_attention_weights(sequences)
            else:
                # Fallback: uniform attention
                batch_size, seq_len, _ = token_embeddings.shape
                device = token_embeddings.device
                attention_weights = torch.ones(batch_size, seq_len, seq_len, device=device) / seq_len

        return token_embeddings, attention_weights

    def _get_rnabert_token_embeddings(
        self,
        sequences: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get RNABERT token embeddings and attention weights."""
        from transformers import AutoTokenizer

        sequences = [seq.upper().replace('T', 'U') for seq in sequences]
        device = next(self.base_embedder.model.parameters()).device

        inputs = self.base_embedder.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.base_embedder._max_length
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.base_embedder.model(**inputs, output_attentions=True)
            token_embeddings = outputs.last_hidden_state

            # Average attention across heads and layers
            if hasattr(outputs, 'attentions') and outputs.attentions is not None:
                attention_weights = torch.stack(outputs.attentions).mean(dim=(0, 2))
            else:
                batch_size, seq_len, _ = token_embeddings.shape
                attention_weights = torch.ones(batch_size, seq_len, seq_len, device=device) / seq_len

        return token_embeddings, attention_weights

    def _get_fallback_token_embeddings(
        self,
        sequences: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fallback for embedders without token-level access."""
        device = next(self.parameters()).device
        batch_size = len(sequences)
        max_len = max(len(s) for s in sequences)

        # Get sequence-level embeddings and replicate
        seq_embs = self.base_embedder.encode(sequences)
        if isinstance(seq_embs, np.ndarray):
            seq_embs = torch.from_numpy(seq_embs).to(device)

        # Replicate to token level
        token_embeddings = seq_embs.unsqueeze(1).expand(-1, max_len + 2, -1)

        # Uniform attention
        attention_weights = torch.ones(batch_size, max_len + 2, max_len + 2, device=device) / (max_len + 2)

        return token_embeddings, attention_weights

    def _compute_structure_features(
        self,
        sequences: List[str],
        edit_positions: torch.Tensor,
        edit_from: List[str],
        edit_to: List[str]
    ) -> torch.Tensor:
        """
        Compute Δ-structure features using structure predictor.

        Returns:
            Structure features tensor of shape [batch, 7]
        """
        if not self.use_structure or self.structure_predictor is None:
            return torch.zeros(len(sequences), 7)

        device = edit_positions.device
        features = []

        for i, seq in enumerate(sequences):
            pos = edit_positions[i].item()
            nuc_from = edit_from[i].upper()
            nuc_to = edit_to[i].upper()

            # Create mutated sequence
            seq_list = list(seq)
            if pos < len(seq_list):
                seq_list[pos] = nuc_to
            seq_after = ''.join(seq_list)

            try:
                delta = self.structure_predictor.compute_delta_structure(
                    seq, seq_after, pos
                )
                feat = [
                    delta['delta_pairing'][pos] if pos < len(delta['delta_pairing']) else 0.0,
                    delta['delta_accessibility'][pos] if pos < len(delta['delta_accessibility']) else 0.0,
                    delta['delta_entropy'][pos] if pos < len(delta['delta_entropy']) else 0.0,
                    delta['delta_mfe'],
                    delta['delta_local_pairing'],
                    np.mean(delta['delta_accessibility'][max(0, pos-self.window_size):pos+self.window_size+1])
                    if len(delta['delta_accessibility']) > 0 else 0.0,
                    np.std(delta['delta_pairing'][max(0, pos-self.window_size):pos+self.window_size+1])
                    if len(delta['delta_pairing']) > 0 else 0.0,
                ]
            except Exception:
                feat = [0.0] * 7

            features.append(feat)

        return torch.tensor(features, dtype=torch.float32, device=device)

    def forward(
        self,
        sequences: Union[str, List[str]],
        edit_positions: Union[int, List[int], torch.Tensor],
        edit_from: Union[str, List[str]],
        edit_to: Union[str, List[str]]
    ) -> torch.Tensor:
        """
        Generate edit embeddings.

        Args:
            sequences: RNA sequence(s) - the original sequence A
            edit_positions: 0-indexed position(s) of the edit
            edit_from: Original nucleotide(s) at edit position
            edit_to: New nucleotide(s) after edit

        Returns:
            edit_embeddings: [batch, output_dim] tensor
        """
        # Handle single inputs
        if isinstance(sequences, str):
            sequences = [sequences]
            edit_positions = [edit_positions]
            edit_from = [edit_from]
            edit_to = [edit_to]

        batch_size = len(sequences)
        device = next(self.parameters()).device

        # Convert edit_positions to tensor
        if isinstance(edit_positions, list):
            edit_positions = torch.tensor(edit_positions, device=device)
        elif isinstance(edit_positions, int):
            edit_positions = torch.tensor([edit_positions], device=device)

        # Normalize sequences (T -> U)
        sequences = [seq.upper().replace('T', 'U') for seq in sequences]
        seq_lengths = torch.tensor([len(s) for s in sequences], device=device, dtype=torch.float)

        # ========================================
        # Get token embeddings
        # ========================================
        token_embeddings, attention_weights = self._get_token_embeddings(sequences)
        token_embeddings = token_embeddings.to(device)
        attention_weights = attention_weights.to(device)

        # ========================================
        # 1. Mutation Type Embedding
        # ========================================
        mutation_type_ids = torch.tensor([
            MUTATION_TYPES.get((f.upper(), t.upper()), 0)
            for f, t in zip(edit_from, edit_to)
        ], device=device)
        mutation_type_emb = self.mutation_type_embed(mutation_type_ids)

        # ========================================
        # 2. Mutation Effect (Δ token)
        # ========================================
        from_ids = torch.tensor([NUC_TO_IDX[f.upper()] for f in edit_from], device=device)
        to_ids = torch.tensor([NUC_TO_IDX[t.upper()] for t in edit_to], device=device)
        delta_token = self.nucleotide_embed(to_ids) - self.nucleotide_embed(from_ids)
        mutation_effect_emb = self.mutation_effect_proj(delta_token)

        # ========================================
        # 3. Position Encoding
        # ========================================
        sin_pos = self.sinusoidal_pos(edit_positions)
        learned_pos = self.learned_pos_embed(edit_positions)
        relative_pos = (edit_positions.float() / seq_lengths).unsqueeze(-1)
        relative_pos_emb = self.relative_pos_proj(relative_pos)
        position_emb = torch.cat([sin_pos, learned_pos, relative_pos_emb], dim=-1)

        # ========================================
        # 4. Local Context (window around edit)
        # ========================================
        local_contexts = []
        for i in range(batch_size):
            pos = edit_positions[i].item()
            seq_len = int(seq_lengths[i].item())
            # Account for special tokens (CLS at 0)
            start = max(1, pos + 1 - self.window_size)
            end = min(seq_len + 1, pos + 1 + self.window_size + 1)
            window = token_embeddings[i, start:end, :]
            local_contexts.append(window.mean(dim=0))
        local_context = torch.stack(local_contexts)
        local_context_emb = self.local_context_proj(local_context)

        # ========================================
        # 5. Attention Context
        # ========================================
        attention_contexts = []
        for i in range(batch_size):
            pos = edit_positions[i].item()
            seq_len = int(seq_lengths[i].item())
            attn_to_edit = attention_weights[i, :seq_len+2, pos+1]
            weighted_tokens = token_embeddings[i, :seq_len+2, :] * attn_to_edit.unsqueeze(-1)
            attention_contexts.append(weighted_tokens.sum(dim=0))
        attention_context = torch.stack(attention_contexts)
        attention_context_emb = self.attention_context_proj(attention_context)

        # ========================================
        # 6. Concatenate components
        # ========================================
        components = [
            mutation_type_emb,
            mutation_effect_emb,
            position_emb,
            local_context_emb,
            attention_context_emb
        ]

        # Add structure features if enabled
        if self.use_structure and self.structure_predictor is not None:
            structure_raw = self._compute_structure_features(
                sequences, edit_positions, edit_from, edit_to
            )
            structure_emb = self.structure_proj(structure_raw)
            components.append(structure_emb)

        raw_embedding = torch.cat(components, dim=-1)

        # ========================================
        # 7. Fusion MLP
        # ========================================
        edit_embedding = self.fusion_mlp(raw_embedding)

        return edit_embedding

    @property
    def embedding_dim(self) -> int:
        """Return the output embedding dimension."""
        return self.output_dim

    @property
    def component_dims(self) -> Dict[str, int]:
        """Return dimensions of each component."""
        return self._component_dims

    def get_component_embeddings(
        self,
        sequences: Union[str, List[str]],
        edit_positions: Union[int, List[int], torch.Tensor],
        edit_from: Union[str, List[str]],
        edit_to: Union[str, List[str]]
    ) -> Dict[str, torch.Tensor]:
        """
        Get individual component embeddings (for analysis/debugging).

        Returns dict with each component embedding.
        """
        if isinstance(sequences, str):
            sequences = [sequences]
            edit_positions = [edit_positions]
            edit_from = [edit_from]
            edit_to = [edit_to]

        batch_size = len(sequences)
        device = next(self.parameters()).device

        if isinstance(edit_positions, list):
            edit_positions = torch.tensor(edit_positions, device=device)

        sequences = [seq.upper().replace('T', 'U') for seq in sequences]
        seq_lengths = torch.tensor([len(s) for s in sequences], device=device, dtype=torch.float)

        token_embeddings, attention_weights = self._get_token_embeddings(sequences)
        token_embeddings = token_embeddings.to(device)
        attention_weights = attention_weights.to(device)

        # Compute each component
        mutation_type_ids = torch.tensor([
            MUTATION_TYPES.get((f.upper(), t.upper()), 0)
            for f, t in zip(edit_from, edit_to)
        ], device=device)
        mutation_type_emb = self.mutation_type_embed(mutation_type_ids)

        from_ids = torch.tensor([NUC_TO_IDX[f.upper()] for f in edit_from], device=device)
        to_ids = torch.tensor([NUC_TO_IDX[t.upper()] for t in edit_to], device=device)
        delta_token = self.nucleotide_embed(to_ids) - self.nucleotide_embed(from_ids)
        mutation_effect_emb = self.mutation_effect_proj(delta_token)

        sin_pos = self.sinusoidal_pos(edit_positions)
        learned_pos = self.learned_pos_embed(edit_positions)
        relative_pos = (edit_positions.float() / seq_lengths).unsqueeze(-1)
        relative_pos_emb = self.relative_pos_proj(relative_pos)
        position_emb = torch.cat([sin_pos, learned_pos, relative_pos_emb], dim=-1)

        local_contexts = []
        for i in range(batch_size):
            pos = edit_positions[i].item()
            seq_len = int(seq_lengths[i].item())
            start = max(1, pos + 1 - self.window_size)
            end = min(seq_len + 1, pos + 1 + self.window_size + 1)
            window = token_embeddings[i, start:end, :]
            local_contexts.append(window.mean(dim=0))
        local_context = torch.stack(local_contexts)
        local_context_emb = self.local_context_proj(local_context)

        attention_contexts = []
        for i in range(batch_size):
            pos = edit_positions[i].item()
            seq_len = int(seq_lengths[i].item())
            attn_to_edit = attention_weights[i, :seq_len+2, pos+1]
            weighted_tokens = token_embeddings[i, :seq_len+2, :] * attn_to_edit.unsqueeze(-1)
            attention_contexts.append(weighted_tokens.sum(dim=0))
        attention_context = torch.stack(attention_contexts)
        attention_context_emb = self.attention_context_proj(attention_context)

        components = [
            mutation_type_emb,
            mutation_effect_emb,
            position_emb,
            local_context_emb,
            attention_context_emb
        ]

        result = {
            'mutation_type': mutation_type_emb,
            'mutation_effect': mutation_effect_emb,
            'position': position_emb,
            'local_context': local_context_emb,
            'attention_context': attention_context_emb,
        }

        if self.use_structure and self.structure_predictor is not None:
            structure_raw = self._compute_structure_features(
                sequences, edit_positions, edit_from, edit_to
            )
            structure_emb = self.structure_proj(structure_raw)
            components.append(structure_emb)
            result['structure'] = structure_emb

        raw_embedding = torch.cat(components, dim=-1)
        fused_embedding = self.fusion_mlp(raw_embedding)

        result['raw'] = raw_embedding
        result['fused'] = fused_embedding

        return result


# =============================================================================
# Factory functions
# =============================================================================

def create_rnafm_structured_embedder(
    trainable: bool = False,
    use_structure: bool = False,
    **kwargs
) -> StructuredRNAEditEmbedder:
    """
    Create StructuredRNAEditEmbedder with RNA-FM as base.

    Args:
        trainable: Whether RNA-FM should be trainable
        use_structure: Whether to use structure features
        **kwargs: Additional arguments for StructuredRNAEditEmbedder

    Returns:
        Configured StructuredRNAEditEmbedder
    """
    from .rnafm import RNAFMEmbedder

    rnafm = RNAFMEmbedder(trainable=trainable)

    structure_predictor = None
    if use_structure:
        try:
            from .rnaplfold import RNAplfoldPredictor
            structure_predictor = RNAplfoldPredictor()
            if not structure_predictor.is_available:
                print("Warning: ViennaRNA not available, structure features disabled")
                structure_predictor = None
                use_structure = False
        except ImportError:
            print("Warning: Could not import RNAplfold, structure features disabled")
            use_structure = False

    return StructuredRNAEditEmbedder(
        base_embedder=rnafm,
        use_structure=use_structure,
        structure_predictor=structure_predictor,
        **kwargs
    )


def create_utrlm_structured_embedder(
    model_name: str = "multimolecule/utrlm-te_el",
    trainable: bool = False,
    use_structure: bool = False,
    **kwargs
) -> StructuredRNAEditEmbedder:
    """
    Create StructuredRNAEditEmbedder with UTR-LM as base.

    This provides structured edit embeddings using UTR-LM (5' UTR specific model).

    Args:
        model_name: HuggingFace model ID for UTR-LM
        trainable: Whether UTR-LM should be trainable
        use_structure: Whether to use structure features
        **kwargs: Additional arguments for StructuredRNAEditEmbedder

    Returns:
        Configured StructuredRNAEditEmbedder
    """
    from .utrlm import UTRLMEmbedder

    utrlm = UTRLMEmbedder(model_path=model_name, trainable=trainable)

    structure_predictor = None
    if use_structure:
        try:
            from .rnaplfold import RNAplfoldPredictor
            structure_predictor = RNAplfoldPredictor()
            if not structure_predictor.is_available:
                print("Warning: ViennaRNA not available, structure features disabled")
                structure_predictor = None
                use_structure = False
        except ImportError:
            print("Warning: Could not import RNAplfold, structure features disabled")
            use_structure = False

    return StructuredRNAEditEmbedder(
        base_embedder=utrlm,
        use_structure=use_structure,
        structure_predictor=structure_predictor,
        **kwargs
    )
