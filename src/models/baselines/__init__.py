"""Baseline models for APOBEC C-to-U editing prediction.

Each baseline isolates a specific aspect of the full EditRNA-A3A model
to measure the contribution of individual components. All baselines
accept a batch dict from ``apobec_collate_fn`` and return a dict with
at least a ``"binary_logit"`` key of shape ``(B, 1)``.

Models
------
ConcatMLP
    Concatenates pooled RNA-FM embeddings of original and edited sequences.
CrossAttentionBaseline
    Cross-attention between original and edited token-level embeddings.
DiffAttentionBaseline
    Transformer encoder on the token-level embedding difference.
StructureOnlyBaseline
    MLP on ViennaRNA structure delta features only.
PooledMLPBaseline
    MLP on pooled original embedding only (no edit information).
SubtractionMLPBaseline
    MLP on the subtraction of pooled embeddings (edited - original).
"""

from .concat_mlp import ConcatMLP, ConcatMLPConfig
from .cross_attention import CrossAttentionBaseline, CrossAttentionConfig
from .diff_attention import DiffAttentionBaseline, DiffAttentionConfig
from .structure_only import StructureOnlyBaseline, StructureOnlyConfig
from .pooled_mlp import PooledMLPBaseline, PooledMLPConfig
from .subtraction_mlp import SubtractionMLPBaseline, SubtractionMLPConfig

__all__ = [
    "ConcatMLP",
    "ConcatMLPConfig",
    "CrossAttentionBaseline",
    "CrossAttentionConfig",
    "DiffAttentionBaseline",
    "DiffAttentionConfig",
    "StructureOnlyBaseline",
    "StructureOnlyConfig",
    "PooledMLPBaseline",
    "PooledMLPConfig",
    "SubtractionMLPBaseline",
    "SubtractionMLPConfig",
]
