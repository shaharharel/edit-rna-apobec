"""
UTR-LM (5' UTR Language Model) embedder.

Based on https://github.com/a96123155/UTR-LM

UTR-LM is a semi-supervised language model for 5' UTR sequences, pretrained on
endogenous 5' UTRs from multiple species. It uses:
- 6 Transformer layers
- 16 multi-head attention heads
- 128-dimensional embeddings
- MLM + MFE + Secondary Structure training objectives

This wrapper supports both trainable and frozen modes.

Pretrained weights can be loaded from:
1. HuggingFace (recommended): pip install multimolecule
   - Model ID: "multimolecule/utrlm-te_el" or "multimolecule/utrlm-mrl"
2. Original checkpoint (.pkl): Download from GitHub releases
   - e.g., ESM2SISS_FS4.1_fiveSpeciesCao_6layers_16heads_128embedsize_...epoch93.pkl
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Dict, Union
from pathlib import Path
import warnings


# Check for HuggingFace multimolecule availability
_HF_AVAILABLE = False
try:
    import multimolecule  # noqa: F401
    from transformers import AutoModel, AutoTokenizer
    _HF_AVAILABLE = True
except ImportError:
    pass


class UTRLMEmbedder(nn.Module):
    """
    UTR-LM embedder for 5' UTR sequences.

    Provides contextualized embeddings specifically trained on 5' UTR sequences.
    Can be used in trainable or frozen mode.
    """

    # Vocabulary for UTR-LM (ACGU + special tokens)
    VOCAB = {
        '<pad>': 0,
        '<cls>': 1,
        '<eos>': 2,
        '<unk>': 3,
        '<mask>': 4,
        'A': 5,
        'C': 6,
        'G': 7,
        'U': 8,
    }

    def __init__(
        self,
        model_path: Optional[str] = None,
        embed_dim: int = 128,
        num_layers: int = 6,
        num_heads: int = 16,
        pooling: str = 'cls',
        trainable: bool = False,
        max_length: int = 256,
        dropout: float = 0.1,
        use_huggingface: bool = True
    ):
        """
        Initialize UTR-LM embedder.

        Args:
            model_path: Path to pretrained weights OR HuggingFace model ID.
                        If None, initializes randomly.
                        Examples:
                          - "multimolecule/utrlm-te_el" (HuggingFace, recommended)
                          - "multimolecule/utrlm-mrl" (HuggingFace)
                          - "/path/to/ESM2SISS_...epoch93.pkl" (local .pkl)
            embed_dim: Embedding dimension (128 for UTR-LM)
            num_layers: Number of transformer layers (6 for UTR-LM)
            num_heads: Number of attention heads (16 for UTR-LM)
            pooling: 'cls' for CLS token, 'mean' for mean pooling
            trainable: Whether to allow gradient updates
            max_length: Maximum sequence length
            dropout: Dropout rate
            use_huggingface: If True and model_path looks like a HF model ID,
                            use HuggingFace transformers to load the model.
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.pooling = pooling
        self._trainable = trainable
        self.max_length = max_length
        self.vocab_size = len(self.VOCAB)
        self._use_hf_model = False  # Will be set to True if HF model is loaded

        # Check if we should use HuggingFace model
        if model_path is not None and use_huggingface and self._is_huggingface_id(model_path):
            self._load_huggingface(model_path)
        else:
            # Build custom model architecture
            self._build_model(embed_dim, num_heads, num_layers, dropout)

            # Load pretrained weights if provided
            if model_path is not None:
                self._load_pretrained(model_path)

        # Set trainable mode
        if not trainable:
            self._freeze()

    def _is_huggingface_id(self, model_path: str) -> bool:
        """Check if model_path looks like a HuggingFace model ID."""
        # HuggingFace IDs contain '/' but are not file paths
        if '/' in model_path and not Path(model_path).exists():
            # Check if it starts with known HF orgs
            if model_path.startswith('multimolecule/'):
                return True
            # Could be any HF repo - if file doesn't exist, try HF
            if not model_path.startswith('/') and not model_path.startswith('.'):
                return True
        return False

    def _build_model(self, embed_dim: int, num_heads: int, num_layers: int, dropout: float):
        """Build the custom transformer model architecture."""
        # Token embedding
        self.token_embedding = nn.Embedding(
            self.vocab_size,
            embed_dim,
            padding_idx=self.VOCAB['<pad>']
        )

        # Positional encoding
        self.position_embedding = nn.Embedding(self.max_length, embed_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(embed_dim)
        )

        # Layer norm
        self.ln = nn.LayerNorm(embed_dim)

    def _load_huggingface(self, model_id: str):
        """Load pretrained UTR-LM from HuggingFace."""
        if not _HF_AVAILABLE:
            raise ImportError(
                "HuggingFace transformers and multimolecule packages required. "
                "Install with: pip install multimolecule transformers"
            )

        print(f"Loading UTR-LM from HuggingFace: {model_id}")

        # Load model and tokenizer from HuggingFace
        self._hf_model = AutoModel.from_pretrained(model_id)
        self._hf_tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._use_hf_model = True

        # Update embed_dim from loaded model
        self.embed_dim = self._hf_model.config.hidden_size

        print(f"Loaded UTR-LM with embed_dim={self.embed_dim}")

    def _load_pretrained(self, model_path: str):
        """Load pretrained UTR-LM weights from .pkl checkpoint file."""
        path = Path(model_path)
        if not path.exists():
            warnings.warn(f"Model file not found: {model_path}")
            return

        try:
            import pickle

            print(f"Loading UTR-LM checkpoint from: {model_path}")

            # Load the .pkl checkpoint
            if model_path.endswith('.pkl'):
                with open(model_path, 'rb') as f:
                    checkpoint = pickle.load(f)
            else:
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    state_dict = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    # Assume the dict itself is the state_dict
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            # Map UTR-LM official checkpoint keys to our model
            # The official UTR-LM uses ESM-style architecture
            mapped_dict = self._map_checkpoint_keys(state_dict)

            # Load with strict=False to allow partial loading
            missing, unexpected = self.load_state_dict(mapped_dict, strict=False)

            if missing:
                print(f"  Missing keys: {len(missing)} (may be expected for task heads)")
            if unexpected:
                print(f"  Unexpected keys: {len(unexpected)}")

            print(f"Successfully loaded UTR-LM weights from {model_path}")

        except Exception as e:
            warnings.warn(f"Could not load pretrained weights: {e}")
            import traceback
            traceback.print_exc()

    def _map_checkpoint_keys(self, state_dict: Dict) -> Dict:
        """Map official UTR-LM checkpoint keys to our model architecture."""
        mapped = {}

        # Key mapping from UTR-LM (ESM-style) to our model
        # UTR-LM uses fair-esm style naming
        key_mapping = {
            # Embeddings
            'embed_tokens.weight': 'token_embedding.weight',
            'embed_positions.weight': 'position_embedding.weight',
            # LayerNorm
            'emb_layer_norm_before.weight': 'ln.weight',
            'emb_layer_norm_before.bias': 'ln.bias',
            'emb_layer_norm_after.weight': 'ln.weight',
            'emb_layer_norm_after.bias': 'ln.bias',
        }

        for old_key, value in state_dict.items():
            new_key = old_key

            # Apply direct mappings
            for old_pattern, new_pattern in key_mapping.items():
                if old_pattern in old_key:
                    new_key = old_key.replace(old_pattern, new_pattern)
                    break

            # Map transformer layer keys
            # UTR-LM: layers.0.self_attn.k_proj.weight
            # Ours: transformer.layers.0.self_attn.in_proj_weight (combined QKV)
            if 'layers.' in old_key:
                # This is a transformer layer
                # Note: PyTorch TransformerEncoder uses different naming
                # We might need to skip some keys or use custom loading
                new_key = old_key.replace('layers.', 'transformer.layers.')

            mapped[new_key] = value

        return mapped

    def _freeze(self):
        """Freeze all parameters."""
        if self._use_hf_model:
            for param in self._hf_model.parameters():
                param.requires_grad = False
        else:
            for param in self.parameters():
                param.requires_grad = False

    def _unfreeze(self):
        """Unfreeze all parameters."""
        if self._use_hf_model:
            for param in self._hf_model.parameters():
                param.requires_grad = True
        else:
            for param in self.parameters():
                param.requires_grad = True

    @property
    def trainable(self) -> bool:
        return self._trainable

    @trainable.setter
    def trainable(self, value: bool):
        self._trainable = value
        if value:
            self._unfreeze()
        else:
            self._freeze()

    @property
    def output_dim(self) -> int:
        return self.embed_dim

    @property
    def embedding_dim(self) -> int:
        """Alias for output_dim to match RNAEmbedder interface."""
        return self.embed_dim

    def _tokenize(self, sequence: str) -> List[int]:
        """Convert sequence to token IDs."""
        sequence = sequence.upper().replace('T', 'U').replace(' ', '')
        tokens = [self.VOCAB['<cls>']]
        for c in sequence[:self.max_length - 2]:
            tokens.append(self.VOCAB.get(c, self.VOCAB['<unk>']))
        tokens.append(self.VOCAB['<eos>'])
        return tokens

    def _batch_tokenize(
        self,
        sequences: List[str],
        device: torch.device
    ) -> tuple:
        """Tokenize and pad a batch of sequences."""
        tokenized = [self._tokenize(seq) for seq in sequences]

        # Find max length
        max_len = max(len(t) for t in tokenized)

        # Pad
        padded = []
        masks = []
        for tokens in tokenized:
            padding = [self.VOCAB['<pad>']] * (max_len - len(tokens))
            padded.append(tokens + padding)
            masks.append([1] * len(tokens) + [0] * len(padding))

        input_ids = torch.tensor(padded, dtype=torch.long, device=device)
        attention_mask = torch.tensor(masks, dtype=torch.float, device=device)

        return input_ids, attention_mask

    def forward(
        self,
        sequences: List[str],
        return_all_tokens: bool = False
    ) -> torch.Tensor:
        """
        Get embeddings for sequences.

        Args:
            sequences: List of RNA sequences
            return_all_tokens: If True, return all token embeddings

        Returns:
            Embeddings tensor of shape (batch_size, embed_dim) or
            (batch_size, seq_len, embed_dim) if return_all_tokens=True
        """
        # Use HuggingFace model if available
        if self._use_hf_model:
            return self._forward_hf(sequences, return_all_tokens)

        # Otherwise use custom model
        return self._forward_custom(sequences, return_all_tokens)

    def _forward_hf(
        self,
        sequences: List[str],
        return_all_tokens: bool = False
    ) -> torch.Tensor:
        """Forward pass using HuggingFace model."""
        # Normalize sequences
        sequences = [seq.upper().replace('T', 'U') for seq in sequences]

        # Tokenize using HF tokenizer
        device = next(self._hf_model.parameters()).device
        inputs = self._hf_tokenizer(
            sequences,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.max_length
        ).to(device)

        # Forward through HF model
        outputs = self._hf_model(**inputs)

        # Get hidden states
        x = outputs.last_hidden_state

        if return_all_tokens:
            return x

        # Pool to sequence-level embedding
        if self.pooling == 'cls':
            return x[:, 0, :]
        elif self.pooling == 'mean':
            attention_mask = inputs['attention_mask'].unsqueeze(-1).float()
            return (x * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

    def _forward_custom(
        self,
        sequences: List[str],
        return_all_tokens: bool = False
    ) -> torch.Tensor:
        """Forward pass using custom model."""
        device = next(self.parameters()).device
        input_ids, attention_mask = self._batch_tokenize(sequences, device)

        batch_size, seq_len = input_ids.shape

        # Token embeddings
        x = self.token_embedding(input_ids)

        # Position embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        x = x + self.position_embedding(positions)

        # Create padding mask for transformer (True = ignore)
        padding_mask = (attention_mask == 0)

        # Transformer
        x = self.transformer(x, src_key_padding_mask=padding_mask)

        # Layer norm
        x = self.ln(x)

        if return_all_tokens:
            return x

        # Pool to sequence-level embedding
        if self.pooling == 'cls':
            # CLS token is at position 0
            return x[:, 0, :]
        elif self.pooling == 'mean':
            # Mean pool over non-padding tokens
            mask = attention_mask.unsqueeze(-1)
            return (x * mask).sum(dim=1) / mask.sum(dim=1)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

    def get_attention_weights(self, sequences: List[str]) -> torch.Tensor:
        """
        Get attention weights for sequences.

        Useful for interpretability - see which positions attend to each other.

        Returns:
            Attention weights tensor of shape (batch, heads, seq_len, seq_len)
        """
        device = next(self.parameters()).device
        input_ids, attention_mask = self._batch_tokenize(sequences, device)

        batch_size, seq_len = input_ids.shape

        # Get embeddings
        x = self.token_embedding(input_ids)
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        x = x + self.position_embedding(positions)

        # We need to access internal attention - this is a simplified version
        # Real implementation would need custom forward with attention output
        padding_mask = (attention_mask == 0)

        # For now, return dummy attention
        # TODO: Implement proper attention extraction
        return torch.ones(batch_size, self.num_heads, seq_len, seq_len, device=device) / seq_len

    def encode(self, sequences: Union[str, List[str]]) -> np.ndarray:
        """
        Encode sequences to numpy arrays.

        For compatibility with base embedder interface.
        """
        if isinstance(sequences, str):
            sequences = [sequences]
            squeeze = True
        else:
            squeeze = False

        with torch.no_grad():
            embeddings = self.forward(sequences)

        result = embeddings.cpu().numpy()
        if squeeze:
            result = result.squeeze(0)
        return result


# =============================================================================
# Factory functions for easy loading
# =============================================================================

# NOTE: UTRLMDeltaEmbedder and UTRLMLocalDeltaEmbedder were removed.
# For simple delta embeddings, use RNAEditEmbedder with UTRLMEmbedder:
#   from .edit_embedder import RNAEditEmbedder
#   edit_emb = RNAEditEmbedder(UTRLMEmbedder(), use_local_context=True)
#
# For structured edit embeddings with UTR-LM, use:
#   from .structured_edit_embedder import create_utrlm_structured_embedder
#   edit_emb = create_utrlm_structured_embedder(trainable=False)

def load_utrlm(
    model_name: str = "multimolecule/utrlm-te_el",
    trainable: bool = False,
    **kwargs
) -> UTRLMEmbedder:
    """
    Load a pretrained UTR-LM model.

    Args:
        model_name: HuggingFace model ID or path to checkpoint.
                    Available HuggingFace models:
                    - "multimolecule/utrlm-te_el" (Translation Efficiency)
                    - "multimolecule/utrlm-mrl" (Mean Ribosome Loading)
        trainable: Whether to allow fine-tuning the model
        **kwargs: Additional arguments passed to UTRLMEmbedder

    Returns:
        UTRLMEmbedder with pretrained weights

    Example:
        >>> from src.embedding.utrlm import load_utrlm
        >>> # Load pretrained (frozen)
        >>> embedder = load_utrlm("multimolecule/utrlm-te_el", trainable=False)
        >>> # Load for fine-tuning
        >>> embedder = load_utrlm("multimolecule/utrlm-te_el", trainable=True)
    """
    return UTRLMEmbedder(
        model_path=model_name,
        trainable=trainable,
        **kwargs
    )


