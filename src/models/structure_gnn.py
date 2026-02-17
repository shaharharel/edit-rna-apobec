"""
GNN models for RNA secondary structure-based prediction.

This module provides Graph Neural Network models that operate on RNA secondary
structure graphs. Graph construction utilities are in src/embedding/structure_graph.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

try:
    from torch_geometric.nn import GCNConv, GATConv, TransformerConv, global_mean_pool
    from torch_geometric.data import Data
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False


class RNAStructureGNN(nn.Module):
    """
    GNN encoder for RNA secondary structure graphs.

    Produces fixed-size graph embeddings from variable-length RNA structures.

    Architecture:
    1. Node embedding layer
    2. Multiple GNN layers (GAT, GCN, or Transformer) with residual connections
    3. Global pooling
    4. Output projection
    """

    def __init__(
        self,
        node_features: int = 9,  # 5 one-hot + rel_pos + 3 structure features
        hidden_dim: int = 128,
        output_dim: int = 128,
        num_layers: int = 4,
        conv_type: str = 'gat',
        heads: int = 4,
        dropout: float = 0.2
    ):
        """
        Args:
            node_features: Input node feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output embedding dimension
            num_layers: Number of GNN layers
            conv_type: Convolution type ('gat', 'gcn', 'transformer')
            heads: Number of attention heads (for GAT/Transformer)
            dropout: Dropout rate
        """
        super().__init__()

        if not HAS_TORCH_GEOMETRIC:
            raise ImportError("torch_geometric required for GNN models")

        self.conv_type = conv_type

        # Node embedding
        self.node_embed = nn.Sequential(
            nn.Linear(node_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # GNN layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            if conv_type == 'gat':
                conv = GATConv(
                    hidden_dim, hidden_dim // heads,
                    heads=heads, concat=True, dropout=dropout
                )
            elif conv_type == 'gcn':
                conv = GCNConv(hidden_dim, hidden_dim)
            elif conv_type == 'transformer':
                conv = TransformerConv(
                    hidden_dim, hidden_dim // heads,
                    heads=heads, concat=True, dropout=dropout
                )
            else:
                raise ValueError(f"Unknown conv type: {conv_type}")

            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = nn.Dropout(dropout)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass.

        Args:
            data: PyG Data or Batch object with x, edge_index, batch

        Returns:
            Graph-level embeddings (batch_size, output_dim)
        """
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(
            x.size(0), dtype=torch.long, device=x.device
        )

        # Initial node embedding
        x = self.node_embed(x)

        # GNN layers with residual connections
        for conv, norm in zip(self.convs, self.norms):
            x_new = conv(x, edge_index)
            x_new = norm(x_new)
            x_new = F.relu(x_new)
            x_new = self.dropout(x_new)
            x = x + x_new  # Residual connection

        # Global pooling
        x = global_mean_pool(x, batch)

        # Output projection
        return self.output_proj(x)


class RNAStructureGNNPredictor(nn.Module):
    """
    Full predictor: GNN encoder + optional features + MLP head.

    Combines:
    - GNN embeddings from secondary structure graph
    - Optional hand-crafted features
    - MLP for final prediction
    """

    def __init__(
        self,
        node_features: int = 9,
        gnn_hidden_dim: int = 128,
        gnn_output_dim: int = 128,
        gnn_layers: int = 4,
        feature_dim: int = 0,
        mlp_hidden_dims: List[int] = [256, 128],
        dropout: float = 0.2,
        conv_type: str = 'gat'
    ):
        """
        Args:
            node_features: Input node feature dimension for GNN
            gnn_hidden_dim: GNN hidden dimension
            gnn_output_dim: GNN output dimension
            gnn_layers: Number of GNN layers
            feature_dim: Dimension of additional hand-crafted features (0 to disable)
            mlp_hidden_dims: Hidden dimensions for MLP head
            dropout: Dropout rate
            conv_type: GNN convolution type
        """
        super().__init__()

        self.gnn = RNAStructureGNN(
            node_features=node_features,
            hidden_dim=gnn_hidden_dim,
            output_dim=gnn_output_dim,
            num_layers=gnn_layers,
            conv_type=conv_type,
            dropout=dropout
        )

        self.feature_dim = feature_dim

        # MLP predictor
        mlp_input_dim = gnn_output_dim + feature_dim
        layers = []
        prev_dim = mlp_input_dim

        for hidden_dim in mlp_hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        data: Data,
        features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            data: PyG Data/Batch with structure graph
            features: Optional (batch_size, feature_dim) hand-crafted features

        Returns:
            Predictions (batch_size,)
        """
        # Get GNN embeddings
        gnn_emb = self.gnn(data)

        # Concatenate with features if provided
        if features is not None and self.feature_dim > 0:
            x = torch.cat([gnn_emb, features], dim=-1)
        else:
            x = gnn_emb

        # MLP prediction
        return self.mlp(x).squeeze(-1)


class HybridSequenceStructureModel(nn.Module):
    """
    Hybrid model combining sequence embeddings, structure GNN, and features.

    Uses cross-attention between:
    - Pre-trained sequence embeddings (e.g., UTR-LM)
    - Structure embeddings from GNN

    Plus optional hand-crafted features.
    """

    def __init__(
        self,
        seq_embed_dim: int,
        struct_node_features: int = 9,
        gnn_hidden_dim: int = 128,
        gnn_output_dim: int = 128,
        gnn_layers: int = 4,
        feature_dim: int = 0,
        hidden_dim: int = 256,
        num_attention_heads: int = 4,
        dropout: float = 0.2
    ):
        """
        Args:
            seq_embed_dim: Dimension of pre-trained sequence embeddings
            struct_node_features: Node features for structure GNN
            gnn_hidden_dim: GNN hidden dimension
            gnn_output_dim: GNN output dimension
            gnn_layers: Number of GNN layers
            feature_dim: Dimension of hand-crafted features (0 to disable)
            hidden_dim: Hidden dimension for projections and predictor
            num_attention_heads: Number of cross-attention heads
            dropout: Dropout rate
        """
        super().__init__()

        # Structure GNN
        self.gnn = RNAStructureGNN(
            node_features=struct_node_features,
            hidden_dim=gnn_hidden_dim,
            output_dim=gnn_output_dim,
            num_layers=gnn_layers,
            dropout=dropout
        )

        # Projections
        self.seq_proj = nn.Linear(seq_embed_dim, hidden_dim)
        self.struct_proj = nn.Linear(gnn_output_dim, hidden_dim)

        # Cross-attention: sequence <-> structure
        self.seq_to_struct_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )
        self.struct_to_seq_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )

        # Feature projection
        self.feature_dim = feature_dim
        if feature_dim > 0:
            self.feature_proj = nn.Linear(feature_dim, hidden_dim)

        # Final predictor
        predictor_input = hidden_dim * 4  # seq_attended + struct_attended + seq + struct
        if feature_dim > 0:
            predictor_input += hidden_dim

        self.predictor = nn.Sequential(
            nn.Linear(predictor_input, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(
        self,
        seq_embedding: torch.Tensor,
        graph_data: Data,
        features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            seq_embedding: (batch_size, seq_embed_dim) pre-trained embeddings
            graph_data: PyG Batch of structure graphs
            features: Optional (batch_size, feature_dim) hand-crafted features

        Returns:
            Predictions (batch_size,)
        """
        # Get GNN embedding
        struct_emb = self.gnn(graph_data)

        # Project embeddings
        seq_h = self.seq_proj(seq_embedding).unsqueeze(1)  # (B, 1, H)
        struct_h = self.struct_proj(struct_emb).unsqueeze(1)  # (B, 1, H)

        # Cross-attention
        seq_attended, _ = self.seq_to_struct_attention(seq_h, struct_h, struct_h)
        struct_attended, _ = self.struct_to_seq_attention(struct_h, seq_h, seq_h)

        # Remove sequence dimension
        seq_attended = seq_attended.squeeze(1)
        struct_attended = struct_attended.squeeze(1)
        seq_h = seq_h.squeeze(1)
        struct_h = struct_h.squeeze(1)

        # Combine representations
        combined = [seq_attended, struct_attended, seq_h, struct_h]

        if features is not None and self.feature_dim > 0:
            feat_h = self.feature_proj(features)
            combined.append(feat_h)

        combined = torch.cat(combined, dim=-1)

        return self.predictor(combined).squeeze(-1)
