"""Encoder wrappers for EditRNA-A3A model.

Provides unified interfaces for all encoder modalities:
1. RNAFMEncoderWrapper  - RNA-FM (640-dim, 12 layers) primary/secondary encoder
2. StructureGNNEncoder  - GAT/GCN/Transformer GNN on RNA secondary structure graphs
3. ContactMapViT        - Vision Transformer on RNA contact probability maps

Each encoder produces a dict with:
  - "embeddings": (batch, seq_len, dim) per-token representations
  - "pooled": (batch, dim) sequence-level representation
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GCNConv, GATConv, TransformerConv, global_mean_pool
    from torch_geometric.data import Data

    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False


# ---------------------------------------------------------------------------
# RNA-FM Encoder Wrapper
# ---------------------------------------------------------------------------


class RNAFMEncoderWrapper(nn.Module):
    """Wrapper around RNA-FM for extracting per-token RNA embeddings.

    Loads the RNA-FM pre-trained model (Chen et al., Nature Methods 2024)
    and provides frozen inference or partial fine-tuning of the last N
    transformer layers.

    Parameters
    ----------
    repr_layer : int
        Transformer layer to extract representations from (1-12).
    finetune_last_n : int
        Number of final transformer layers to keep trainable. 0 = frozen.
    projection_dim : int or None
        If set, project embeddings down to this dimension.
    """

    EMBEDDING_DIM = 640
    NUM_LAYERS = 12

    def __init__(
        self,
        repr_layer: int = 12,
        finetune_last_n: int = 0,
        projection_dim: Optional[int] = None,
    ):
        super().__init__()
        self.repr_layer = repr_layer
        self.finetune_last_n = finetune_last_n

        import fm

        self.model, self.alphabet = fm.pretrained.rna_fm_t12()
        self.batch_converter = self.alphabet.get_batch_converter()

        self._setup_freezing()

        self.projection = None
        out_dim = self.EMBEDDING_DIM
        if projection_dim is not None:
            self.projection = nn.Sequential(
                nn.Linear(self.EMBEDDING_DIM, projection_dim),
                nn.GELU(),
                nn.LayerNorm(projection_dim),
            )
            out_dim = projection_dim
        self.output_dim = out_dim

    def _setup_freezing(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = False

        if self.finetune_last_n > 0:
            layers = self.model.layers
            total = len(layers)
            for i in range(total - self.finetune_last_n, total):
                for param in layers[i].parameters():
                    param.requires_grad = True
            if hasattr(self.model, "emb_layer_norm_after"):
                for param in self.model.emb_layer_norm_after.parameters():
                    param.requires_grad = True

    def tokenize(
        self, sequences: list[str], names: Optional[list[str]] = None
    ) -> torch.Tensor:
        """Convert RNA sequences to token tensors.

        Returns (batch, max_seq_len + 2) including BOS/EOS.
        """
        if names is None:
            names = [f"seq_{i}" for i in range(len(sequences))]
        data = list(zip(names, sequences))
        _, _, batch_tokens = self.batch_converter(data)
        return batch_tokens

    def forward(
        self,
        tokens: Optional[torch.Tensor] = None,
        sequences: Optional[list[str]] = None,
    ) -> dict[str, torch.Tensor]:
        """Extract per-token embeddings from RNA-FM.

        Provide either ``tokens`` or ``sequences``.

        Returns
        -------
        dict with:
            "embeddings" : (batch, seq_len, dim) -- BOS/EOS stripped
            "pooled"     : (batch, dim) -- mean-pooled
            "tokens"     : (batch, seq_len) -- token ids used
        """
        if tokens is None and sequences is None:
            raise ValueError("Provide either tokens or sequences")

        if tokens is None:
            tokens = self.tokenize(sequences)
            tokens = tokens.to(next(self.model.parameters()).device)

        with torch.set_grad_enabled(self.finetune_last_n > 0 and self.training):
            results = self.model(tokens, repr_layers=[self.repr_layer])

        embeddings = results["representations"][self.repr_layer]
        embeddings = embeddings[:, 1:-1, :]  # strip BOS/EOS

        if self.projection is not None:
            embeddings = self.projection(embeddings)

        mask = (tokens[:, 1:-1] != self.alphabet.padding_idx).unsqueeze(-1).float()
        pooled = (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        return {"embeddings": embeddings, "pooled": pooled, "tokens": tokens}

    def get_embedding_dim(self) -> int:
        return self.output_dim

    @torch.no_grad()
    def embed_sequences(
        self, sequences: list[str], batch_size: int = 32
    ) -> torch.Tensor:
        """Embed sequences in batches, return (n_sequences, dim) pooled."""
        self.eval()
        device = next(self.model.parameters()).device
        all_pooled = []
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i : i + batch_size]
            tokens = self.tokenize(batch_seqs).to(device)
            result = self.forward(tokens=tokens)
            all_pooled.append(result["pooled"].cpu())
        return torch.cat(all_pooled, dim=0)


# ---------------------------------------------------------------------------
# Structure GNN Encoder
# ---------------------------------------------------------------------------


class StructureGNNEncoder(nn.Module):
    """GNN encoder for RNA secondary structure graphs.

    Produces fixed-size graph embeddings from variable-length RNA structures
    using GAT, GCN, or Transformer convolutions with residual connections.

    Parameters
    ----------
    node_features : int
        Input node feature dimension (5 one-hot + rel_pos + 3 structure).
    hidden_dim : int
        GNN hidden dimension.
    output_dim : int
        Output embedding dimension.
    num_layers : int
        Number of GNN layers.
    conv_type : str
        Convolution type: 'gat', 'gcn', or 'transformer'.
    heads : int
        Number of attention heads (GAT/Transformer only).
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        node_features: int = 9,
        hidden_dim: int = 128,
        output_dim: int = 128,
        num_layers: int = 4,
        conv_type: str = "gat",
        heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()

        if not HAS_TORCH_GEOMETRIC:
            raise ImportError("torch_geometric required for StructureGNNEncoder")

        self.conv_type = conv_type
        self.output_dim = output_dim

        self.node_embed = nn.Sequential(
            nn.Linear(node_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            if conv_type == "gat":
                conv = GATConv(
                    hidden_dim,
                    hidden_dim // heads,
                    heads=heads,
                    concat=True,
                    dropout=dropout,
                )
            elif conv_type == "gcn":
                conv = GCNConv(hidden_dim, hidden_dim)
            elif conv_type == "transformer":
                conv = TransformerConv(
                    hidden_dim,
                    hidden_dim // heads,
                    heads=heads,
                    concat=True,
                    dropout=dropout,
                )
            else:
                raise ValueError(f"Unknown conv type: {conv_type}")
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = nn.Dropout(dropout)

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, data: "Data") -> dict[str, torch.Tensor]:
        """Encode an RNA structure graph.

        Parameters
        ----------
        data : torch_geometric.data.Data or Batch
            Graph(s) with ``x``, ``edge_index``, and optional ``batch``.

        Returns
        -------
        dict with:
            "embeddings" : (total_nodes, hidden_dim) node embeddings
            "pooled"     : (batch_size, output_dim) graph-level embedding
        """
        x, edge_index = data.x, data.edge_index
        batch = (
            data.batch
            if hasattr(data, "batch")
            else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        )

        x = self.node_embed(x)
        node_embeddings = x

        for conv, norm in zip(self.convs, self.norms):
            x_new = conv(x, edge_index)
            x_new = norm(x_new)
            x_new = F.relu(x_new)
            x_new = self.dropout(x_new)
            x = x + x_new

        node_embeddings = x
        pooled = global_mean_pool(x, batch)
        pooled = self.output_proj(pooled)

        return {"embeddings": node_embeddings, "pooled": pooled}

    def get_embedding_dim(self) -> int:
        return self.output_dim


# ---------------------------------------------------------------------------
# Contact Map Vision Transformer
# ---------------------------------------------------------------------------


class ContactMapViT(nn.Module):
    """Vision Transformer encoder for RNA contact probability maps.

    Takes an (L, L) base-pair probability matrix (e.g. from RNAplfold or
    LinearPartition) and encodes it into a fixed-size representation using
    a lightweight ViT architecture.

    The contact map is treated as a single-channel 2D image: it is divided
    into non-overlapping patches, linearly embedded, and processed by
    standard Transformer encoder layers.

    Parameters
    ----------
    max_len : int
        Maximum sequence length the map can represent.
    patch_size : int
        Size of each square patch.
    d_model : int
        Transformer hidden dimension.
    n_heads : int
        Number of attention heads per layer.
    n_layers : int
        Number of Transformer encoder layers.
    output_dim : int
        Final output embedding dimension.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        max_len: int = 512,
        patch_size: int = 16,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        output_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.max_len = max_len
        self.patch_size = patch_size
        self.d_model = d_model
        self.output_dim = output_dim

        n_patches_per_side = max_len // patch_size
        self.n_patches = n_patches_per_side * n_patches_per_side
        patch_dim = patch_size * patch_size  # single channel

        # Patch embedding: flatten each patch and project
        self.patch_embed = nn.Linear(patch_dim, d_model)

        # Learnable positional embeddings + CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.n_patches + 1, d_model) * 0.02
        )
        self.pos_drop = nn.Dropout(dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        self.norm = nn.LayerNorm(d_model)

        self.output_proj = nn.Sequential(
            nn.Linear(d_model, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
        )

    def _pad_and_patchify(self, contact_map: torch.Tensor) -> torch.Tensor:
        """Pad contact map to max_len and extract patches.

        Parameters
        ----------
        contact_map : (batch, L, L) float
            Base-pair probability maps.

        Returns
        -------
        patches : (batch, n_patches, patch_dim)
        """
        B, L, _ = contact_map.shape
        if L < self.max_len:
            pad = self.max_len - L
            contact_map = F.pad(contact_map, (0, pad, 0, pad), value=0.0)
        elif L > self.max_len:
            contact_map = contact_map[:, : self.max_len, : self.max_len]

        ps = self.patch_size
        n = self.max_len // ps
        # Reshape into (B, n, ps, n, ps) then flatten patches
        patches = contact_map.reshape(B, n, ps, n, ps)
        patches = patches.permute(0, 1, 3, 2, 4).reshape(B, n * n, ps * ps)
        return patches

    def forward(self, contact_map: torch.Tensor) -> dict[str, torch.Tensor]:
        """Encode a batch of contact probability maps.

        Parameters
        ----------
        contact_map : (batch, L, L) float
            Symmetric base-pair probability matrices.

        Returns
        -------
        dict with:
            "pooled" : (batch, output_dim) CLS-pooled representation
        """
        patches = self._pad_and_patchify(contact_map)  # (B, n_patches, patch_dim)
        x = self.patch_embed(patches)  # (B, n_patches, d_model)

        # Prepend CLS token
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, 1 + n_patches, d_model)
        x = x + self.pos_embed[:, : x.shape[1], :]
        x = self.pos_drop(x)

        x = self.transformer(x)
        x = self.norm(x)

        cls_out = x[:, 0]  # CLS token
        pooled = self.output_proj(cls_out)

        return {"pooled": pooled}

    def get_embedding_dim(self) -> int:
        return self.output_dim


# ---------------------------------------------------------------------------
# Mock Encoder (for testing without pre-trained weights)
# ---------------------------------------------------------------------------


class MockRNAEncoder(nn.Module):
    """Lightweight mock encoder for pipeline testing without RNA-FM weights.

    Produces random but consistently shaped outputs matching the RNA-FM
    interface, allowing the full pipeline to be tested end-to-end.

    Parameters
    ----------
    d_model : int
        Output embedding dimension (640 for RNA-FM compatibility).
    """

    def __init__(self, d_model: int = 640):
        super().__init__()
        self.d_model = d_model
        self.output_dim = d_model
        self.embed = nn.Linear(4, d_model)

    def forward(
        self,
        tokens: Optional[torch.Tensor] = None,
        sequences: Optional[list[str]] = None,
    ) -> dict[str, torch.Tensor]:
        if sequences is None:
            raise ValueError("MockRNAEncoder requires sequences")

        device = self.embed.weight.device
        batch_size = len(sequences)
        max_len = max(len(s) for s in sequences)

        nuc_map = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3, "N": 0}
        indices = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        for i, seq in enumerate(sequences):
            for j, c in enumerate(seq):
                indices[i, j] = nuc_map.get(c.upper(), 0)

        one_hot = F.one_hot(indices, num_classes=4).float()
        embeddings = self.embed(one_hot)
        pooled = embeddings.mean(dim=1)

        return {"embeddings": embeddings, "pooled": pooled, "tokens": indices}

    def get_embedding_dim(self) -> int:
        return self.output_dim


# ---------------------------------------------------------------------------
# Cached RNA Encoder (for fast training with pre-computed embeddings)
# ---------------------------------------------------------------------------


class CachedRNAEncoder(nn.Module):
    """Encoder that returns pre-computed RNA-FM embeddings from a cache.

    Enables fast training without running the heavy RNA-FM forward pass
    on every batch. Requires pre-computed embedding dicts keyed by site_id.

    Parameters
    ----------
    tokens_cache : dict[str, Tensor]
        Mapping site_id -> (seq_len, d_model) per-token embeddings.
    pooled_cache : dict[str, Tensor]
        Mapping site_id -> (d_model,) pooled embeddings.
    tokens_edited_cache : dict[str, Tensor]
        Mapping site_id -> (seq_len, d_model) edited per-token embeddings.
    pooled_edited_cache : dict[str, Tensor]
        Mapping site_id -> (d_model,) edited pooled embeddings.
    d_model : int
        Embedding dimension (640 for RNA-FM).
    """

    def __init__(
        self,
        tokens_cache: dict,
        pooled_cache: dict,
        tokens_edited_cache: dict,
        pooled_edited_cache: dict,
        d_model: int = 640,
    ):
        super().__init__()
        self.tokens_cache = tokens_cache
        self.pooled_cache = pooled_cache
        self.tokens_edited_cache = tokens_edited_cache
        self.pooled_edited_cache = pooled_edited_cache
        self.d_model = d_model
        self.output_dim = d_model
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(
        self,
        tokens: Optional[torch.Tensor] = None,
        sequences: Optional[list[str]] = None,
        site_ids: Optional[list[str]] = None,
        edited: bool = False,
    ) -> dict[str, torch.Tensor]:
        if site_ids is None:
            raise ValueError("CachedRNAEncoder requires site_ids")

        tok_cache = self.tokens_edited_cache if edited else self.tokens_cache
        pool_cache = self.pooled_edited_cache if edited else self.pooled_cache

        device = self._dummy.device
        embeddings = torch.stack([tok_cache[sid].to(device) for sid in site_ids])
        pooled = torch.stack([pool_cache[sid].to(device) for sid in site_ids])

        return {"embeddings": embeddings, "pooled": pooled}

    def get_embedding_dim(self) -> int:
        return self.output_dim
