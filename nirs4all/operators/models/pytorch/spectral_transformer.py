"""
Spectral Transformer - Modern Transformer Architecture for NIR Spectral Data.

This module provides a state-of-the-art transformer model specifically designed
for Near-Infrared (NIR) spectroscopy data classification and regression tasks.

Key Features:
- Learnable positional encoding for spectral data
- Pre-LayerNorm transformer blocks for stable training
- Patch embedding for efficient sequence handling
- Multi-scale attention with configurable heads
- Class token for aggregation (CLS token approach)
- Proper regularization (dropout, stochastic depth)
- Support for binary, multi-class classification, and regression

Architecture inspired by:
- Vision Transformer (ViT) for patch-based processing
- Pre-LN Transformer for training stability
- DeBERTa's disentangled attention concepts

Designed for ~4k samples with typical NIR spectral dimensions (100-2000 wavelengths).
"""

import math
from typing import Optional, List, Tuple, Dict, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from nirs4all.utils import framework


class PatchEmbedding1D(nn.Module):
    """
    Converts 1D spectral sequence into patches and projects to embedding dimension.

    For NIR spectra, this groups consecutive wavelengths into patches,
    reducing sequence length while increasing representational capacity.

    Args:
        in_channels: Number of input channels (usually 1 for raw spectra).
        embed_dim: Embedding dimension for each patch.
        patch_size: Number of wavelengths per patch.
        seq_len: Original sequence length (number of wavelengths).
        dropout: Dropout rate for the projection.
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch_size: int,
        seq_len: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size
        self.proj = nn.Conv1d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, channels, seq_len)

        Returns:
            Patch embeddings of shape (batch, num_patches, embed_dim)
        """
        # x: (B, C, L) -> (B, embed_dim, num_patches)
        x = self.proj(x)
        # (B, embed_dim, num_patches) -> (B, num_patches, embed_dim)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.dropout(x)
        return x


class LearnablePositionalEncoding(nn.Module):
    """
    Learnable positional encoding for sequence data.

    Unlike fixed sinusoidal encodings, learnable positions can adapt
    to the specific structure of NIR spectral data.

    Args:
        num_positions: Maximum number of positions (sequence length + 1 for CLS).
        embed_dim: Embedding dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        num_positions: int,
        embed_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, embed_dim))
        self.dropout = nn.Dropout(dropout)
        # Initialize with small random values
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        return self.dropout(x + self.pos_embed[:, :x.size(1)])


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention with optional attention dropout.

    Uses pre-normalization pattern for improved training stability.

    Args:
        embed_dim: Embedding dimension.
        num_heads: Number of attention heads.
        dropout: Dropout rate for attention weights.
        qkv_bias: Whether to use bias in QKV projections.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        qkv_bias: bool = True
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim)

        Returns:
            Output tensor of shape (batch, seq_len, embed_dim)
        """
        B, N, C = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Weighted sum
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class FeedForward(nn.Module):
    """
    Feed-forward network with GELU activation.

    Two-layer MLP with expansion and contraction.

    Args:
        embed_dim: Input/output embedding dimension.
        hidden_dim: Hidden layer dimension (typically 4x embed_dim).
        dropout: Dropout rate.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Pre-LayerNorm Transformer block.

    Pre-normalization improves training stability, especially for
    deeper models and smaller datasets.

    Args:
        embed_dim: Embedding dimension.
        num_heads: Number of attention heads.
        ff_mult: Multiplier for feed-forward hidden dimension.
        dropout: Dropout rate.
        drop_path: Stochastic depth rate.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_mult: float = 4.0,
        dropout: float = 0.1,
        drop_path: float = 0.0
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, int(embed_dim * ff_mult), dropout)

        # Stochastic depth for regularization
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm attention with residual
        x = x + self.drop_path(self.attn(self.norm1(x)))
        # Pre-norm feed-forward with residual
        x = x + self.drop_path(self.ff(self.norm2(x)))
        return x


class DropPath(nn.Module):
    """
    Stochastic Depth (drop path) regularization.

    Randomly drops entire residual branches during training.

    Args:
        drop_prob: Probability of dropping the path.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        # Work with batched tensors
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # Binarize
        output = x.div(keep_prob) * random_tensor
        return output


class SpectralTransformer(nn.Module):
    """
    Modern Transformer for NIR Spectral Classification.

    Designed specifically for spectroscopy data with:
    - Patch-based processing for efficiency
    - CLS token for classification
    - Pre-LayerNorm for stable training
    - Stochastic depth for regularization
    - Support for 2-class, multi-class, and regression

    Args:
        input_shape: Tuple of (channels, seq_len) for input spectra.
        num_classes: Number of output classes (1 for regression, 2+ for classification).
        embed_dim: Transformer embedding dimension.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        patch_size: Size of spectral patches.
        ff_mult: Feed-forward expansion multiplier.
        dropout: General dropout rate.
        drop_path: Maximum stochastic depth rate.
        pool: Pooling method - 'cls' for CLS token, 'mean' for mean pooling.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int],
        num_classes: int = 1,
        embed_dim: int = 64,
        depth: int = 4,
        num_heads: int = 4,
        patch_size: int = 16,
        ff_mult: float = 4.0,
        dropout: float = 0.1,
        drop_path: float = 0.1,
        pool: str = 'cls'
    ):
        super().__init__()

        in_channels, seq_len = input_shape
        self.num_classes = num_classes
        self.pool = pool

        # Calculate number of patches
        num_patches = seq_len // patch_size

        # Patch embedding
        self.patch_embed = PatchEmbedding1D(
            in_channels, embed_dim, patch_size, seq_len, dropout
        )

        # CLS token for classification
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Positional encoding (patches + CLS token)
        self.pos_embed = LearnablePositionalEncoding(
            num_patches + 1, embed_dim, dropout
        )

        # Stochastic depth decay
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_mult, dropout, dpr[i])
            for i in range(depth)
        ])

        # Final normalization
        self.norm = nn.LayerNorm(embed_dim)

        # Classification/Regression head
        self.head = self._build_head(embed_dim, num_classes, dropout)

        # Initialize weights
        self.apply(self._init_weights)

    def _build_head(
        self,
        embed_dim: int,
        num_classes: int,
        dropout: float
    ) -> nn.Module:
        """Build the output head based on task type.

        Uses same output conventions as existing nirs4all models:
        - Regression: linear output
        - Binary (num_classes=2): single sigmoid output
        - Multi-class (num_classes>2): softmax output
        """
        if num_classes == 1:
            # Regression - linear output
            return nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim // 2, 1)
            )
        elif num_classes == 2:
            # Binary classification - single sigmoid output (matches nicon.py convention)
            return nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim // 2, 1),
                nn.Sigmoid()
            )
        else:
            # Multi-class classification - softmax output (matches nicon.py convention)
            return nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim // 2, num_classes),
                nn.Softmax(dim=-1)
            )

    def _init_weights(self, m: nn.Module):
        """Initialize model weights."""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, channels, seq_len)

        Returns:
            Output predictions of shape (batch, num_classes) or (batch, 1)
        """
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches + 1, embed_dim)

        # Add positional encoding
        x = self.pos_embed(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final normalization
        x = self.norm(x)

        # Pooling
        if self.pool == 'cls':
            x = x[:, 0]  # Use CLS token
        else:
            x = x[:, 1:].mean(dim=1)  # Mean pooling (exclude CLS)

        # Classification/Regression head
        return self.head(x)


def _build_spectral_transformer(
    input_shape: Tuple[int, int],
    params: Dict[str, Any],
    num_classes: int = 1
) -> SpectralTransformer:
    """
    Build a SpectralTransformer with configurable parameters.

    Args:
        input_shape: Tuple of (channels, seq_len).
        params: Configuration dictionary.
        num_classes: Number of output classes.

    Returns:
        Configured SpectralTransformer instance.
    """
    # Default parameters tuned for ~4k samples with NIR spectra
    c, seq_len = input_shape

    # Auto-adjust patch size based on sequence length
    # Smaller patches preserve more spectral resolution
    default_patch_size = max(4, min(16, seq_len // 64))

    return SpectralTransformer(
        input_shape=input_shape,
        num_classes=num_classes,
        embed_dim=params.get('embed_dim', 64),
        depth=params.get('depth', 3),
        num_heads=params.get('num_heads', 4),
        patch_size=params.get('patch_size', 10),
        ff_mult=params.get('ff_mult', 2.0),  # Reduced from 4.0 for smaller datasets
        dropout=params.get('dropout', 0.1),  # Reduced from 0.15
        drop_path=params.get('drop_path', 0.0),  # Disabled by default for learning stability
        pool=params.get('pool', 'mean')  # Mean pooling often works better for spectral data
    )


# -----------------------------------------------------------------------------
#  Public API - Framework-decorated factory functions
# -----------------------------------------------------------------------------

@framework("pytorch")
def spectral_transformer(input_shape: Tuple[int, int], params: Dict[str, Any] = None) -> SpectralTransformer:
    """
    Create a SpectralTransformer for regression.

    Suitable for ~4k samples of NIR spectral data.

    Args:
        input_shape: Tuple of (channels, seq_len).
        params: Configuration dictionary with keys:
            - embed_dim (int): Embedding dimension (default: 64)
            - depth (int): Number of transformer blocks (default: 4)
            - num_heads (int): Number of attention heads (default: 4)
            - patch_size (int): Spectral patch size (default: auto)
            - ff_mult (float): Feed-forward multiplier (default: 4.0)
            - dropout (float): Dropout rate (default: 0.15)
            - drop_path (float): Stochastic depth rate (default: 0.1)
            - pool (str): Pooling method 'cls' or 'mean' (default: 'cls')

    Returns:
        SpectralTransformer model instance.
    """
    params = params or {}
    return _build_spectral_transformer(input_shape, params, num_classes=1)


@framework("pytorch")
def spectral_transformer_classification(
    input_shape: Tuple[int, int],
    num_classes: int = 2,
    params: Dict[str, Any] = None
) -> SpectralTransformer:
    """
    Create a SpectralTransformer for classification.

    Suitable for binary and multi-class classification on NIR spectral data.

    Args:
        input_shape: Tuple of (channels, seq_len).
        num_classes: Number of classes (2 for binary, 3+ for multi-class).
        params: Configuration dictionary (see spectral_transformer).

    Returns:
        SpectralTransformer model instance for classification.
    """
    params = params or {}
    return _build_spectral_transformer(input_shape, params, num_classes=num_classes)


@framework("pytorch")
def spectral_transformer_small(input_shape: Tuple[int, int], params: Dict[str, Any] = None) -> SpectralTransformer:
    """
    Small SpectralTransformer variant - faster training, less overfitting risk.

    Recommended for smaller datasets (<1k samples).

    Args:
        input_shape: Tuple of (channels, seq_len).
        params: Additional configuration parameters.

    Returns:
        Small SpectralTransformer model instance.
    """
    default_params = {
        'embed_dim': 32,
        'depth': 2,
        'num_heads': 2,
        'dropout': 0.2,
        'drop_path': 0.05
    }
    default_params.update(params or {})
    return _build_spectral_transformer(input_shape, default_params, num_classes=1)


@framework("pytorch")
def spectral_transformer_small_classification(
    input_shape: Tuple[int, int],
    num_classes: int = 2,
    params: Dict[str, Any] = None
) -> SpectralTransformer:
    """
    Small SpectralTransformer for classification.

    Args:
        input_shape: Tuple of (channels, seq_len).
        num_classes: Number of classes.
        params: Additional configuration parameters.

    Returns:
        Small SpectralTransformer model instance for classification.
    """
    default_params = {
        'embed_dim': 32,
        'depth': 2,
        'num_heads': 2,
        'dropout': 0.2,
        'drop_path': 0.05
    }
    default_params.update(params or {})
    return _build_spectral_transformer(input_shape, default_params, num_classes=num_classes)


@framework("pytorch")
def spectral_transformer_large(input_shape: Tuple[int, int], params: Dict[str, Any] = None) -> SpectralTransformer:
    """
    Large SpectralTransformer variant - higher capacity for larger datasets.

    Recommended for datasets with >5k samples.

    Args:
        input_shape: Tuple of (channels, seq_len).
        params: Additional configuration parameters.

    Returns:
        Large SpectralTransformer model instance.
    """
    default_params = {
        'embed_dim': 128,
        'depth': 6,
        'num_heads': 8,
        'ff_mult': 4.0,
        'dropout': 0.1,
        'drop_path': 0.15
    }
    default_params.update(params or {})
    return _build_spectral_transformer(input_shape, default_params, num_classes=1)


@framework("pytorch")
def spectral_transformer_large_classification(
    input_shape: Tuple[int, int],
    num_classes: int = 2,
    params: Dict[str, Any] = None
) -> SpectralTransformer:
    """
    Large SpectralTransformer for classification.

    Args:
        input_shape: Tuple of (channels, seq_len).
        num_classes: Number of classes.
        params: Additional configuration parameters.

    Returns:
        Large SpectralTransformer model instance for classification.
    """
    default_params = {
        'embed_dim': 128,
        'depth': 6,
        'num_heads': 8,
        'ff_mult': 4.0,
        'dropout': 0.1,
        'drop_path': 0.15
    }
    default_params.update(params or {})
    return _build_spectral_transformer(input_shape, default_params, num_classes=num_classes)


# Export all public functions
__all__ = [
    'SpectralTransformer',
    'spectral_transformer',
    'spectral_transformer_classification',
    'spectral_transformer_small',
    'spectral_transformer_small_classification',
    'spectral_transformer_large',
    'spectral_transformer_large_classification',
]
