"""
CodonBERT embedder stub.

CodonBERT is a BERT-based model pretrained on codon sequences for
protein-related RNA analysis. This is useful for:
- CDS (coding sequence) analysis
- Codon usage bias detection
- Translation efficiency prediction

This is a STUB implementation for future development.
TODO: Implement actual CodonBERT loading and inference.

Reference: https://github.com/Shen-Lab/CodonBERT
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Dict, Union
import warnings


# Codon table (64 codons)
CODON_TO_AA = {
    'UUU': 'F', 'UUC': 'F', 'UUA': 'L', 'UUG': 'L',
    'UCU': 'S', 'UCC': 'S', 'UCA': 'S', 'UCG': 'S',
    'UAU': 'Y', 'UAC': 'Y', 'UAA': '*', 'UAG': '*',
    'UGU': 'C', 'UGC': 'C', 'UGA': '*', 'UGG': 'W',
    'CUU': 'L', 'CUC': 'L', 'CUA': 'L', 'CUG': 'L',
    'CCU': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'CAU': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'CGU': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'AUU': 'I', 'AUC': 'I', 'AUA': 'I', 'AUG': 'M',
    'ACU': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'AAU': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
    'AGU': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GUU': 'V', 'GUC': 'V', 'GUA': 'V', 'GUG': 'V',
    'GCU': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'GAU': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'GGU': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
}

# Create codon vocabulary
CODON_VOCAB = {codon: i for i, codon in enumerate(CODON_TO_AA.keys())}


class CodonBERTEmbedder(nn.Module):
    """
    CodonBERT embedder for codon-level RNA/mRNA analysis.

    STUB IMPLEMENTATION - uses simple codon embeddings.
    TODO: Replace with actual CodonBERT model.

    For now, this provides:
    - Codon-level tokenization
    - Simple learned embeddings (not pretrained)
    - Basic delta computation for synonymous/non-synonymous mutations
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        embed_dim: int = 256,
        trainable: bool = False,
        max_codons: int = 512
    ):
        """
        Initialize CodonBERT embedder.

        Args:
            model_path: Path to pretrained weights (not implemented yet)
            embed_dim: Embedding dimension
            trainable: Whether to allow gradient updates
            max_codons: Maximum number of codons in sequence
        """
        super().__init__()

        self.embed_dim = embed_dim
        self._trainable = trainable
        self.max_codons = max_codons
        self.n_codons = 64

        # Simple codon embeddings (placeholder for real CodonBERT)
        self.codon_embed = nn.Embedding(self.n_codons + 3, embed_dim)  # +3 for special tokens
        self.position_embed = nn.Embedding(max_codons, embed_dim)

        # Simple transformer layer (placeholder)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=8,
                dim_feedforward=embed_dim * 4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=2
        )

        self.ln = nn.LayerNorm(embed_dim)

        warnings.warn(
            "CodonBERT is a STUB implementation. "
            "For production use, implement actual CodonBERT model loading."
        )

        if model_path:
            warnings.warn(f"Model path provided but not implemented: {model_path}")

        if not trainable:
            self._freeze()

    def _freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def _unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    @property
    def trainable(self) -> bool:
        return self._trainable

    @property
    def output_dim(self) -> int:
        return self.embed_dim

    @staticmethod
    def sequence_to_codons(sequence: str) -> List[str]:
        """
        Split RNA sequence into codons.

        Args:
            sequence: RNA sequence (length should be multiple of 3 for CDS)

        Returns:
            List of codon strings
        """
        sequence = sequence.upper().replace('T', 'U').replace(' ', '')

        # Pad to multiple of 3 if needed
        remainder = len(sequence) % 3
        if remainder > 0:
            sequence += 'N' * (3 - remainder)

        codons = [sequence[i:i+3] for i in range(0, len(sequence), 3)]
        return codons

    def _tokenize(self, sequence: str) -> torch.Tensor:
        """Convert sequence to codon token IDs."""
        codons = self.sequence_to_codons(sequence)
        token_ids = []

        for codon in codons[:self.max_codons]:
            codon = codon.upper()
            if codon in CODON_VOCAB:
                token_ids.append(CODON_VOCAB[codon])
            else:
                # Unknown codon (e.g., contains N)
                token_ids.append(self.n_codons)  # UNK token

        return torch.tensor(token_ids, dtype=torch.long)

    def forward(
        self,
        sequences: List[str],
        return_all_codons: bool = False
    ) -> torch.Tensor:
        """
        Get embeddings for sequences.

        Args:
            sequences: List of RNA sequences
            return_all_codons: If True, return per-codon embeddings

        Returns:
            Embeddings tensor
        """
        device = next(self.parameters()).device

        # Tokenize
        tokenized = [self._tokenize(seq) for seq in sequences]

        # Pad to same length
        max_len = max(len(t) for t in tokenized)
        padded = torch.zeros(len(sequences), max_len, dtype=torch.long, device=device)
        mask = torch.zeros(len(sequences), max_len, dtype=torch.bool, device=device)

        for i, tokens in enumerate(tokenized):
            padded[i, :len(tokens)] = tokens.to(device)
            mask[i, len(tokens):] = True  # Mask padding

        # Embed
        x = self.codon_embed(padded)
        positions = torch.arange(max_len, device=device).unsqueeze(0).expand(len(sequences), -1)
        x = x + self.position_embed(positions)

        # Transform
        x = self.transformer(x, src_key_padding_mask=mask)
        x = self.ln(x)

        if return_all_codons:
            return x

        # Mean pool over non-padding tokens
        x = x.masked_fill(mask.unsqueeze(-1), 0)
        lengths = (~mask).sum(dim=1, keepdim=True).float()
        return x.sum(dim=1) / lengths

    def encode(self, sequences: Union[str, List[str]]) -> np.ndarray:
        """Encode to numpy (compatibility interface)."""
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


class CodonDeltaEmbedder(nn.Module):
    """
    Compute codon-level Δ-embeddings.

    Useful for analyzing mutations in coding regions:
    - Synonymous vs non-synonymous changes
    - Codon usage preference changes
    - Frame-shift detection
    """

    def __init__(
        self,
        codonbert: Optional[CodonBERTEmbedder] = None,
        trainable: bool = False,
        **kwargs
    ):
        super().__init__()

        if codonbert is None:
            codonbert = CodonBERTEmbedder(trainable=trainable, **kwargs)
        self.codonbert = codonbert

    @property
    def output_dim(self) -> int:
        return self.codonbert.output_dim

    def is_synonymous(self, codon_from: str, codon_to: str) -> bool:
        """Check if a codon change is synonymous."""
        aa_from = CODON_TO_AA.get(codon_from.upper().replace('T', 'U'), '?')
        aa_to = CODON_TO_AA.get(codon_to.upper().replace('T', 'U'), '?')
        return aa_from == aa_to

    def forward(
        self,
        seq_before: List[str],
        seq_after: List[str]
    ) -> torch.Tensor:
        """
        Compute Δ-embedding for sequence pairs.

        Args:
            seq_before: Original sequences
            seq_after: Mutated sequences

        Returns:
            Δ-embeddings of shape [batch, embed_dim]
        """
        emb_before = self.codonbert(seq_before)
        emb_after = self.codonbert(seq_after)
        return emb_after - emb_before

    def get_mutation_features(
        self,
        seq_before: str,
        seq_after: str,
        position: int
    ) -> Dict[str, float]:
        """
        Get features for a specific mutation.

        Args:
            seq_before: Original sequence
            seq_after: Mutated sequence
            position: Nucleotide position of mutation

        Returns:
            Dictionary with mutation features
        """
        # Find which codon was affected
        codon_idx = position // 3
        codon_pos = position % 3  # 0, 1, or 2 (position within codon)

        codons_before = self.codonbert.sequence_to_codons(seq_before)
        codons_after = self.codonbert.sequence_to_codons(seq_after)

        if codon_idx >= len(codons_before) or codon_idx >= len(codons_after):
            return {'is_synonymous': 0.0, 'codon_position': float(codon_pos)}

        codon_from = codons_before[codon_idx]
        codon_to = codons_after[codon_idx]

        is_syn = self.is_synonymous(codon_from, codon_to)

        return {
            'is_synonymous': 1.0 if is_syn else 0.0,
            'codon_position': float(codon_pos),
            'codon_from_idx': float(CODON_VOCAB.get(codon_from.replace('T', 'U'), -1)),
            'codon_to_idx': float(CODON_VOCAB.get(codon_to.replace('T', 'U'), -1)),
        }


# Factory function for future use
def load_codonbert(
    model_name: str = "codonbert-base",
    trainable: bool = False
) -> CodonBERTEmbedder:
    """
    Load a CodonBERT model.

    STUB: Currently returns placeholder model.
    TODO: Implement actual model loading from HuggingFace or local weights.

    Args:
        model_name: Model variant to load
        trainable: Whether to allow fine-tuning

    Returns:
        CodonBERT embedder
    """
    warnings.warn(
        f"load_codonbert is a stub. Requested model '{model_name}' "
        "will return a randomly initialized placeholder."
    )

    return CodonBERTEmbedder(trainable=trainable)
