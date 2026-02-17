"""
Hierarchical Architectures for RNA Editing Prediction.

These architectures capture multi-scale patterns in RNA sequences:
- Local context (5nt window around editing site)
- Medium context (10-20nt)
- Broad context (50nt)
- Global context (full 101nt window)

Key insight: ADAR editing depends on:
1. Immediate sequence context (UAG motif)
2. Local dsRNA structure (stem-loops)
3. Broader structural features (MFE, paired_frac)
4. Proximity to other editing sites (captured implicitly through structure)

These architectures explicitly model this hierarchy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
import math


# =============================================================================
# Base Components
# =============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence positions."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, seq_len, d_model]"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiScaleConv1D(nn.Module):
    """Multi-scale 1D convolutions for capturing patterns at different resolutions."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: List[int] = [3, 5, 7, 11],
        dropout: float = 0.1
    ):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels // len(kernel_sizes),
                     kernel_size=k, padding=k // 2)
            for k in kernel_sizes
        ])
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, seq_len, channels] -> [batch, seq_len, out_channels]"""
        x = x.transpose(1, 2)  # [batch, channels, seq_len]
        outputs = [conv(x) for conv in self.convs]
        x = torch.cat(outputs, dim=1)  # [batch, out_channels, seq_len]
        x = x.transpose(1, 2)  # [batch, seq_len, out_channels]
        x = self.norm(x)
        x = self.activation(x)
        return self.dropout(x)


# =============================================================================
# 1. U-Net / Feature Pyramid Network for RNA
# =============================================================================

class RNAUNet(nn.Module):
    """
    U-Net / FPN-like architecture for RNA editing prediction.

    Processes RNA at multiple resolutions:
    - Level 0: Full 101nt (or configurable)
    - Level 1: 50nt (2x downsampled)
    - Level 2: 25nt (4x downsampled)
    - Level 3: 12nt (8x downsampled)
    - Level 4: 6nt (16x downsampled) - center region

    Then upsamples with skip connections, focusing on center position.

    Args:
        embedding_dim: Dimension of input embeddings (e.g., RNA-FM = 640)
        hidden_dim: Base hidden dimension
        n_levels: Number of hierarchical levels
        window_sizes: Window sizes at each level (from fine to coarse)
        dropout: Dropout probability
    """

    def __init__(
        self,
        embedding_dim: int = 640,
        hidden_dim: int = 256,
        n_levels: int = 4,
        window_sizes: List[int] = [5, 10, 20, 50],  # Fine to coarse
        dropout: float = 0.2,
        use_features: bool = True,
        n_features: int = 66
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_levels = n_levels
        self.window_sizes = window_sizes
        self.use_features = use_features

        # Input projection
        self.input_proj = nn.Linear(embedding_dim, hidden_dim)

        # Encoder (downsampling path)
        self.encoders = nn.ModuleList()
        self.downsample = nn.ModuleList()

        for i in range(n_levels):
            dim = hidden_dim * (2 ** i)
            next_dim = hidden_dim * (2 ** (i + 1))

            self.encoders.append(nn.Sequential(
                nn.Linear(dim, dim),
                nn.LayerNorm(dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim, dim),
                nn.LayerNorm(dim),
                nn.GELU()
            ))

            if i < n_levels - 1:
                self.downsample.append(nn.Linear(dim, next_dim))

        # Bottleneck
        bottleneck_dim = hidden_dim * (2 ** (n_levels - 1))
        self.bottleneck = nn.Sequential(
            nn.Linear(bottleneck_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Decoder (upsampling path with skip connections)
        self.decoders = nn.ModuleList()
        self.upsample = nn.ModuleList()

        for i in range(n_levels - 1, 0, -1):
            dim = hidden_dim * (2 ** i)
            prev_dim = hidden_dim * (2 ** (i - 1))

            self.upsample.append(nn.Linear(dim, prev_dim))
            # Skip connection doubles the dimension
            self.decoders.append(nn.Sequential(
                nn.Linear(prev_dim * 2, prev_dim),
                nn.LayerNorm(prev_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ))

        # Feature integration
        if use_features:
            self.feature_proj = nn.Sequential(
                nn.Linear(n_features, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU()
            )
            final_dim = hidden_dim * 2
        else:
            final_dim = hidden_dim

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(final_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def _extract_window(self, x: torch.Tensor, window_size: int) -> torch.Tensor:
        """Extract center window from sequence."""
        seq_len = x.size(1)
        center = seq_len // 2
        start = max(0, center - window_size // 2)
        end = min(seq_len, center + window_size // 2 + 1)
        return x[:, start:end, :].mean(dim=1)  # Pool to single vector

    def forward(
        self,
        embeddings: torch.Tensor,
        features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            embeddings: [batch, seq_len, embedding_dim] - per-position embeddings
                       OR [batch, embedding_dim] - pooled embeddings
            features: [batch, n_features] - handcrafted features (optional)

        Returns:
            logits: [batch] - editing prediction logits
        """
        # Handle pooled vs per-position embeddings
        if embeddings.dim() == 2:
            # Pooled embeddings - expand to sequence
            x = self.input_proj(embeddings)  # [batch, hidden]

            # Skip hierarchical processing, go directly to output
            if self.use_features and features is not None:
                feat_repr = self.feature_proj(features)
                x = torch.cat([x, feat_repr], dim=-1)

            return self.output_head(x).squeeze(-1)

        # Per-position embeddings
        x = self.input_proj(embeddings)  # [batch, seq_len, hidden]

        # Encoder path - extract at different scales
        encoder_outputs = []
        for i, (encoder, window_size) in enumerate(zip(self.encoders, self.window_sizes)):
            # Extract window at this scale
            window_repr = self._extract_window(x, window_size)
            window_repr = encoder(window_repr)
            encoder_outputs.append(window_repr)

            # Downsample for next level
            if i < self.n_levels - 1:
                x = self.downsample[i](self._extract_window(x, window_size).unsqueeze(1))
                x = x.expand(-1, embeddings.size(1), -1)  # Broadcast back

        # Bottleneck
        bottleneck_repr = self.bottleneck(encoder_outputs[-1])

        # Decoder path with skip connections
        x = bottleneck_repr
        for i, (upsample, decoder) in enumerate(zip(self.upsample, self.decoders)):
            x = upsample(x)
            skip_idx = self.n_levels - 2 - i
            skip = encoder_outputs[skip_idx]
            x = torch.cat([x, skip], dim=-1)
            x = decoder(x)

        # Combine with features
        if self.use_features and features is not None:
            feat_repr = self.feature_proj(features)
            x = torch.cat([x, feat_repr], dim=-1)

        return self.output_head(x).squeeze(-1)


# =============================================================================
# 2. Hierarchical Transformer for RNA
# =============================================================================

class HierarchicalTransformerLayer(nn.Module):
    """Single layer of hierarchical transformer processing."""

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1
    ):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = residual + self.dropout(x)

        # FFN
        residual = x
        x = self.norm2(x)
        x = residual + self.dropout(self.ffn(x))

        return x


class HierarchicalRNATransformer(nn.Module):
    """
    Hierarchical Transformer for RNA editing prediction.

    Multi-level attention:
    - Level 1: 5nt window (immediate context)
    - Level 2: 10nt window (local structure)
    - Level 3: 20nt window (stem region)
    - Level 4: 50nt window (broad structure)
    - Level 5: 101nt (global)

    Each level attends within its window, then outputs are combined.

    Args:
        embedding_dim: Input embedding dimension
        d_model: Transformer hidden dimension
        n_heads: Number of attention heads
        n_layers_per_level: Transformer layers per hierarchy level
        levels: List of window sizes (fine to coarse)
        dropout: Dropout probability
    """

    def __init__(
        self,
        embedding_dim: int = 640,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers_per_level: int = 2,
        levels: List[int] = [5, 10, 20, 50, 101],
        dropout: float = 0.1,
        use_features: bool = True,
        n_features: int = 66
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.d_model = d_model
        self.levels = levels
        self.n_levels = len(levels)
        self.use_features = use_features

        # Input projection
        self.input_proj = nn.Linear(embedding_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max(levels) + 10, dropout=dropout)

        # Level-specific transformers
        self.level_transformers = nn.ModuleList()
        self.level_norms = nn.ModuleList()

        for level_idx in range(self.n_levels):
            layers = nn.ModuleList([
                HierarchicalTransformerLayer(d_model, n_heads, d_model * 4, dropout)
                for _ in range(n_layers_per_level)
            ])
            self.level_transformers.append(layers)
            self.level_norms.append(nn.LayerNorm(d_model))

        # Cross-level attention (bottom-up aggregation)
        self.cross_level_attn = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            for _ in range(self.n_levels - 1)
        ])

        # Feature integration
        if use_features:
            self.feature_proj = nn.Sequential(
                nn.Linear(n_features, d_model),
                nn.LayerNorm(d_model),
                nn.GELU()
            )
            final_dim = d_model * (self.n_levels + 1)
        else:
            final_dim = d_model * self.n_levels

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(final_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )

    def _extract_window(
        self,
        x: torch.Tensor,
        window_size: int,
        center: int
    ) -> torch.Tensor:
        """Extract center window from sequence."""
        seq_len = x.size(1)
        start = max(0, center - window_size // 2)
        end = min(seq_len, start + window_size)
        # Adjust start if we hit the end
        if end - start < window_size:
            start = max(0, end - window_size)
        return x[:, start:end, :]

    def forward(
        self,
        embeddings: torch.Tensor,
        features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            embeddings: [batch, seq_len, embedding_dim] or [batch, embedding_dim]
            features: [batch, n_features] optional

        Returns:
            logits: [batch]
        """
        # Handle pooled embeddings
        if embeddings.dim() == 2:
            x = self.input_proj(embeddings)  # [batch, d_model]

            if self.use_features and features is not None:
                feat_repr = self.feature_proj(features)
                x = torch.cat([x] * self.n_levels + [feat_repr], dim=-1)
            else:
                x = x.repeat(1, self.n_levels)

            return self.output_head(x).squeeze(-1)

        # Per-position embeddings
        batch_size, seq_len, _ = embeddings.shape
        center = seq_len // 2

        x = self.input_proj(embeddings)
        x = self.pos_encoding(x)

        # Process each level
        level_outputs = []

        for level_idx, window_size in enumerate(self.levels):
            # Extract window
            window = self._extract_window(x, window_size, center)

            # Apply transformer layers
            for layer in self.level_transformers[level_idx]:
                window = layer(window)

            # Normalize and pool (center position or mean)
            window = self.level_norms[level_idx](window)

            # Pool to single vector (use center or mean)
            if window.size(1) > 1:
                center_pos = window.size(1) // 2
                level_repr = window[:, center_pos, :]
            else:
                level_repr = window.squeeze(1)

            level_outputs.append(level_repr)

        # Cross-level aggregation (optional - can be enabled for richer interaction)
        # For now, just concatenate level outputs

        # Combine all levels
        combined = torch.cat(level_outputs, dim=-1)  # [batch, d_model * n_levels]

        # Add features
        if self.use_features and features is not None:
            feat_repr = self.feature_proj(features)
            combined = torch.cat([combined, feat_repr], dim=-1)

        return self.output_head(combined).squeeze(-1)


# =============================================================================
# 3. Graph Neural Network with Hierarchical Pooling
# =============================================================================

class HierarchicalGraphPooling(nn.Module):
    """
    GNN with hierarchical pooling for RNA structure.

    Pools nodes at multiple scales:
    - Local pooling (nearby positions)
    - Medium pooling (structural motifs)
    - Global pooling (full graph)

    This captures both local and global structural information.
    """

    def __init__(
        self,
        node_dim: int = 640,
        hidden_dim: int = 256,
        n_gnn_layers: int = 3,
        pool_ratios: List[float] = [0.5, 0.5, 0.5],  # Fraction to keep at each level
        dropout: float = 0.2,
        use_features: bool = True,
        n_features: int = 66
    ):
        super().__init__()

        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.n_levels = len(pool_ratios)
        self.use_features = use_features

        # Try to import PyG
        try:
            from torch_geometric.nn import GCNConv, TopKPooling, global_mean_pool
            self.GCNConv = GCNConv
            self.TopKPooling = TopKPooling
            self.global_mean_pool = global_mean_pool
            self.has_pyg = True
        except ImportError:
            self.has_pyg = False
            print("Warning: torch_geometric not installed. HierarchicalGraphPooling will use fallback.")

        if self.has_pyg:
            # Input projection
            self.input_proj = nn.Linear(node_dim, hidden_dim)

            # GNN layers and pooling at each level
            self.gnn_layers = nn.ModuleList()
            self.pool_layers = nn.ModuleList()
            self.level_norms = nn.ModuleList()

            for i, ratio in enumerate(pool_ratios):
                # GNN layers at this level
                gnn = nn.ModuleList([
                    self.GCNConv(hidden_dim, hidden_dim)
                    for _ in range(n_gnn_layers)
                ])
                self.gnn_layers.append(gnn)

                # Pooling layer
                self.pool_layers.append(self.TopKPooling(hidden_dim, ratio=ratio))

                # Normalization
                self.level_norms.append(nn.LayerNorm(hidden_dim))

        # Feature integration
        if use_features:
            self.feature_proj = nn.Sequential(
                nn.Linear(n_features, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU()
            )
            final_dim = hidden_dim * (self.n_levels + 2)  # +1 for features, +1 for final
        else:
            final_dim = hidden_dim * (self.n_levels + 1)

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(final_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Node features [n_nodes, node_dim]
            edge_index: Edge indices [2, n_edges]
            batch: Batch assignment [n_nodes] (for batched graphs)
            features: [batch_size, n_features] optional

        Returns:
            logits: [batch_size]
        """
        if not self.has_pyg:
            raise ImportError("torch_geometric required for HierarchicalGraphPooling")

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Input projection
        x = self.input_proj(x)
        x = F.gelu(x)

        # Collect representations at each level
        level_reprs = []

        # Initial global pooling
        level_reprs.append(self.global_mean_pool(x, batch))

        # Hierarchical processing
        for i in range(self.n_levels):
            # GNN layers
            for gnn in self.gnn_layers[i]:
                x = gnn(x, edge_index)
                x = F.gelu(x)

            x = self.level_norms[i](x)

            # Pool
            x, edge_index, _, batch, _, _ = self.pool_layers[i](x, edge_index, batch=batch)

            # Collect global representation at this level
            level_reprs.append(self.global_mean_pool(x, batch))

        # Combine all levels
        combined = torch.cat(level_reprs, dim=-1)

        # Add features
        if self.use_features and features is not None:
            feat_repr = self.feature_proj(features)
            combined = torch.cat([combined, feat_repr], dim=-1)

        return self.output_head(combined).squeeze(-1)


# =============================================================================
# 4. Set Transformer with Induced Set Attention
# =============================================================================

class InducedSetAttentionBlock(nn.Module):
    """
    Induced Set Attention Block (ISAB) from Set Transformer.

    Uses inducing points to reduce O(n²) attention to O(nm) where m << n.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_inducing: int = 16,
        dropout: float = 0.1
    ):
        super().__init__()

        self.n_inducing = n_inducing

        # Learnable inducing points
        self.inducing_points = nn.Parameter(torch.randn(1, n_inducing, d_model))

        # Two attention layers: X -> I -> X
        self.attn_to_inducing = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.attn_from_inducing = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, n_elements, d_model]"""
        batch_size = x.size(0)

        # Expand inducing points for batch
        I = self.inducing_points.expand(batch_size, -1, -1)

        # Attention: I attends to X (compress)
        I_updated, _ = self.attn_to_inducing(I, x, x)
        I = I + self.dropout(I_updated)
        I = self.norm1(I)

        # Attention: X attends to I (decompress)
        x_updated, _ = self.attn_from_inducing(x, I, I)
        x = x + self.dropout(x_updated)
        x = self.norm2(x)

        # FFN
        x = x + self.dropout(self.ffn(x))

        return x


class SetTransformerRNA(nn.Module):
    """
    Set Transformer with Induced Set Attention for RNA editing.

    Treats nearby positions as sets and uses ISA for efficient attention.
    Multiple levels of induced attention capture different scales.

    Args:
        embedding_dim: Input embedding dimension
        d_model: Hidden dimension
        n_heads: Attention heads
        n_isab_layers: Number of ISAB layers
        n_inducing_per_level: Inducing points at each level
        levels: Window sizes for each level
        dropout: Dropout probability
    """

    def __init__(
        self,
        embedding_dim: int = 640,
        d_model: int = 256,
        n_heads: int = 8,
        n_isab_layers: int = 2,
        n_inducing_per_level: List[int] = [4, 8, 16, 32],
        levels: List[int] = [5, 10, 20, 50],
        dropout: float = 0.1,
        use_features: bool = True,
        n_features: int = 66
    ):
        super().__init__()

        self.d_model = d_model
        self.levels = levels
        self.n_levels = len(levels)
        self.use_features = use_features

        # Input projection
        self.input_proj = nn.Linear(embedding_dim, d_model)

        # ISAB layers at each level
        self.level_isabs = nn.ModuleList()

        for level_idx, (window_size, n_inducing) in enumerate(zip(levels, n_inducing_per_level)):
            isabs = nn.ModuleList([
                InducedSetAttentionBlock(d_model, n_heads, n_inducing, dropout)
                for _ in range(n_isab_layers)
            ])
            self.level_isabs.append(isabs)

        # Pooling attention (PMA)
        self.pool_seeds = nn.Parameter(torch.randn(1, 1, d_model))
        self.pool_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        # Feature integration
        if use_features:
            self.feature_proj = nn.Sequential(
                nn.Linear(n_features, d_model),
                nn.LayerNorm(d_model),
                nn.GELU()
            )
            final_dim = d_model * (self.n_levels + 1)
        else:
            final_dim = d_model * self.n_levels

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(final_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )

    def _extract_window(self, x: torch.Tensor, window_size: int, center: int) -> torch.Tensor:
        """Extract window as set of positions."""
        seq_len = x.size(1)
        start = max(0, center - window_size // 2)
        end = min(seq_len, start + window_size)
        return x[:, start:end, :]

    def _pool_set(self, x: torch.Tensor) -> torch.Tensor:
        """Pool set to single vector using attention."""
        batch_size = x.size(0)
        seed = self.pool_seeds.expand(batch_size, -1, -1)
        pooled, _ = self.pool_attn(seed, x, x)
        return pooled.squeeze(1)

    def forward(
        self,
        embeddings: torch.Tensor,
        features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            embeddings: [batch, seq_len, embedding_dim] or [batch, embedding_dim]
            features: [batch, n_features] optional

        Returns:
            logits: [batch]
        """
        # Handle pooled embeddings
        if embeddings.dim() == 2:
            x = self.input_proj(embeddings)

            if self.use_features and features is not None:
                feat_repr = self.feature_proj(features)
                x = torch.cat([x] * self.n_levels + [feat_repr], dim=-1)
            else:
                x = x.repeat(1, self.n_levels)

            return self.output_head(x).squeeze(-1)

        # Per-position embeddings
        batch_size, seq_len, _ = embeddings.shape
        center = seq_len // 2

        x = self.input_proj(embeddings)

        # Process each level
        level_outputs = []

        for level_idx, window_size in enumerate(self.levels):
            # Extract window as set
            window_set = self._extract_window(x, window_size, center)

            # Apply ISAB layers
            for isab in self.level_isabs[level_idx]:
                window_set = isab(window_set)

            # Pool to single vector
            level_repr = self._pool_set(window_set)
            level_outputs.append(level_repr)

        # Combine levels
        combined = torch.cat(level_outputs, dim=-1)

        # Add features
        if self.use_features and features is not None:
            feat_repr = self.feature_proj(features)
            combined = torch.cat([combined, feat_repr], dim=-1)

        return self.output_head(combined).squeeze(-1)


# =============================================================================
# 5. Multiscale Graph Transformer
# =============================================================================

class MultiscaleGraphTransformer(nn.Module):
    """
    Multiscale Graph Transformer for RNA editing.

    Combines graph structure with transformer attention at multiple scales.
    Uses structural edges (from RNA secondary structure) and sequential edges.

    Key idea: Use graph structure to guide attention patterns at different scales.

    Args:
        embedding_dim: Input embedding dimension
        d_model: Hidden dimension
        n_heads: Attention heads
        n_layers: Transformer layers
        scales: Distance scales for multiscale attention [5, 10, 20, 50]
        dropout: Dropout probability
    """

    def __init__(
        self,
        embedding_dim: int = 640,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        scales: List[int] = [5, 10, 20, 50],
        dropout: float = 0.1,
        use_features: bool = True,
        n_features: int = 66
    ):
        super().__init__()

        self.d_model = d_model
        self.scales = scales
        self.n_scales = len(scales)
        self.use_features = use_features

        # Input projection
        self.input_proj = nn.Linear(embedding_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)

        # Scale-specific projections (for scale embeddings)
        self.scale_embeddings = nn.Parameter(torch.randn(len(scales), d_model))

        # Multiscale transformer layers
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            layer = nn.ModuleDict({
                'scale_attns': nn.ModuleList([
                    nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
                    for _ in scales
                ]),
                'scale_fusion': nn.Linear(d_model * len(scales), d_model),
                'ffn': nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model * 4, d_model)
                ),
                'norm1': nn.LayerNorm(d_model),
                'norm2': nn.LayerNorm(d_model)
            })
            self.layers.append(layer)

        # Feature integration
        if use_features:
            self.feature_proj = nn.Sequential(
                nn.Linear(n_features, d_model),
                nn.LayerNorm(d_model),
                nn.GELU()
            )
            final_dim = d_model * 2
        else:
            final_dim = d_model

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(final_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )

    def _create_scale_mask(
        self,
        seq_len: int,
        scale: int,
        device: torch.device
    ) -> torch.Tensor:
        """Create attention mask for a specific scale (distance-based)."""
        # Only attend to positions within 'scale' distance
        positions = torch.arange(seq_len, device=device)
        dist_matrix = torch.abs(positions.unsqueeze(0) - positions.unsqueeze(1))

        # Mask: True where attention is NOT allowed
        mask = dist_matrix > scale
        return mask

    def forward(
        self,
        embeddings: torch.Tensor,
        features: Optional[torch.Tensor] = None,
        structure_edges: Optional[torch.Tensor] = None  # Optional RNA structure edges
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            embeddings: [batch, seq_len, embedding_dim] or [batch, embedding_dim]
            features: [batch, n_features] optional
            structure_edges: [2, n_edges] optional - RNA base pairing edges

        Returns:
            logits: [batch]
        """
        # Handle pooled embeddings
        if embeddings.dim() == 2:
            x = self.input_proj(embeddings)

            if self.use_features and features is not None:
                feat_repr = self.feature_proj(features)
                x = torch.cat([x, feat_repr], dim=-1)

            return self.output_head(x).squeeze(-1)

        # Per-position embeddings
        batch_size, seq_len, _ = embeddings.shape
        device = embeddings.device

        x = self.input_proj(embeddings)
        x = self.pos_encoding(x)

        # Create scale-based attention masks
        scale_masks = [
            self._create_scale_mask(seq_len, scale, device)
            for scale in self.scales
        ]

        # Apply multiscale transformer layers
        for layer in self.layers:
            # Multi-scale attention
            scale_outputs = []
            for scale_idx, (scale_attn, mask) in enumerate(zip(layer['scale_attns'], scale_masks)):
                # Add scale embedding
                scale_emb = self.scale_embeddings[scale_idx].unsqueeze(0).unsqueeze(0)
                x_scaled = x + scale_emb

                # Scale-specific attention
                attn_out, _ = scale_attn(x_scaled, x_scaled, x_scaled, attn_mask=mask)
                scale_outputs.append(attn_out)

            # Fuse scales
            fused = torch.cat(scale_outputs, dim=-1)
            fused = layer['scale_fusion'](fused)

            # Residual + norm
            x = layer['norm1'](x + fused)

            # FFN
            x = layer['norm2'](x + layer['ffn'](x))

        # Pool (center position)
        center = seq_len // 2
        x_center = x[:, center, :]

        # Add features
        if self.use_features and features is not None:
            feat_repr = self.feature_proj(features)
            x_center = torch.cat([x_center, feat_repr], dim=-1)

        return self.output_head(x_center).squeeze(-1)


# =============================================================================
# Factory function
# =============================================================================

def create_hierarchical_model(
    model_type: str,
    embedding_dim: int = 640,
    hidden_dim: int = 256,
    n_features: int = 66,
    use_features: bool = True,
    **kwargs
) -> nn.Module:
    """
    Factory function to create hierarchical RNA models.

    Args:
        model_type: One of 'unet', 'hierarchical_transformer', 'hierarchical_gnn',
                   'set_transformer', 'multiscale_graph_transformer'
        embedding_dim: Input embedding dimension
        hidden_dim: Hidden dimension
        n_features: Number of handcrafted features
        use_features: Whether to use handcrafted features
        **kwargs: Additional model-specific arguments

    Returns:
        Instantiated model
    """
    models = {
        'unet': RNAUNet,
        'rna_unet': RNAUNet,
        'hierarchical_transformer': HierarchicalRNATransformer,
        'hier_transformer': HierarchicalRNATransformer,
        'hierarchical_gnn': HierarchicalGraphPooling,
        'hier_gnn': HierarchicalGraphPooling,
        'set_transformer': SetTransformerRNA,
        'multiscale_graph_transformer': MultiscaleGraphTransformer,
        'multiscale_gt': MultiscaleGraphTransformer,
    }

    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(models.keys())}")

    model_class = models[model_type]

    # Common args
    common_kwargs = {
        'embedding_dim': embedding_dim,
        'use_features': use_features,
        'n_features': n_features,
    }

    # Model-specific defaults
    if model_type in ['unet', 'rna_unet']:
        common_kwargs['hidden_dim'] = hidden_dim
    elif model_type in ['hierarchical_transformer', 'hier_transformer']:
        common_kwargs['d_model'] = hidden_dim
    elif model_type in ['hierarchical_gnn', 'hier_gnn']:
        common_kwargs['hidden_dim'] = hidden_dim
        common_kwargs['node_dim'] = embedding_dim
        del common_kwargs['embedding_dim']
    elif model_type == 'set_transformer':
        common_kwargs['d_model'] = hidden_dim
    elif model_type in ['multiscale_graph_transformer', 'multiscale_gt']:
        common_kwargs['d_model'] = hidden_dim

    # Merge with user kwargs
    common_kwargs.update(kwargs)

    return model_class(**common_kwargs)
