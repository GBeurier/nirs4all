import math
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
#  Light‑weight Keras → PyTorch helpers
# -----------------------------------------------------------------------------

class DepthwiseConv1d(nn.Conv1d):
    """Depth‑wise 1‑D convolution (grouped conv where groups=in_channels)."""

    def __init__(self, in_channels: int, kernel_size: int, depth_multiplier: int = 1, **conv_kwargs):
        super().__init__(
            in_channels,
            in_channels * depth_multiplier,
            kernel_size,
            groups=in_channels,
            **conv_kwargs,
        )


class SeparableConv1d(nn.Module):
    """Separable 1‑D convolution = depth‑wise + point‑wise."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, depth_multiplier: int = 1, **conv_kwargs):
        super().__init__()
        self.depthwise = DepthwiseConv1d(in_channels, kernel_size, depth_multiplier, **conv_kwargs)
        self.pointwise = nn.Conv1d(in_channels * depth_multiplier, out_channels, kernel_size=1)

    def forward(self, x):  # (N,C,L)
        return self.pointwise(self.depthwise(x))


class SpatialDropout1d(nn.Dropout2d):
    """TensorFlow‑like spatial dropout along the channel axis."""

    def forward(self, x):  # (N,C,L)
        x = x.unsqueeze(3)  # (N,C,L,1)
        x = super().forward(x)
        return x.squeeze(3)


class GlobalAveragePooling1d(nn.Module):
    def forward(self, x):  # (N,C,L)
        return x.mean(dim=2)


# -----------------------------------------------------------------------------
#  Generic building blocks used by all architectures
# -----------------------------------------------------------------------------

def _activation(name: str):
    if name == "swish":
        return nn.SiLU()
    elif name == "selu":
        return nn.SELU()
    elif name == "elu":
        return nn.ELU()
    else:
        return nn.ReLU() if name == "relu" else getattr(nn, name.capitalize())()


def _norm(method: str, num_features: int):
    if method == "LayerNormalization":
        return nn.LayerNorm(num_features)
    return nn.BatchNorm1d(num_features)


# -----------------------------------------------------------------------------
#  Primitive model builders (DECON, NICON, Transformer)
# -----------------------------------------------------------------------------


def _build_decon(input_shape: Tuple[int, int], params: Dict, *, num_classes: int = 1):
    """Port of the TensorFlow `decon` family to PyTorch.

    This covers both regression (num_classes==1) and classification
    (num_classes>=2, including binary with sigmoid).
    """
    c, _ = input_shape  # channels, length
    layers: List[nn.Module] = [SpatialDropout1d(params.get("spatial_dropout", 0.2))]

    def _dconv(kernel, depth_multiplier):
        return DepthwiseConv1d(c, kernel, depth_multiplier=depth_multiplier, padding="same")

    # First block (7‑7)
    layers += [
        _dconv(7, 2), _activation("relu"),
        _dconv(7, 2), _activation("relu"),
        nn.MaxPool1d(2, 2), nn.BatchNorm1d(c * 2),
    ]

    # Second block (5‑5)
    layers += [
        _dconv(5, 2), _activation("relu"),
        _dconv(5, 2), _activation("relu"),
        nn.MaxPool1d(2, 2), nn.BatchNorm1d(c * 4),
    ]

    # Third block (9‑9)
    layers += [
        _dconv(9, 2), _activation("relu"),
        _dconv(9, 2), _activation("relu"),
        nn.MaxPool1d(2, 2), nn.BatchNorm1d(c * 8),
    ]

    # Separable + ordinary conv
    layers += [
        SeparableConv1d(c * 8, 64, 3, depth_multiplier=1, padding="same"), _activation("relu"),
        nn.Conv1d(64, 32, 3, padding="same"), _activation("relu"),
        nn.MaxPool1d(5, 3), SpatialDropout1d(0.1),
        nn.Flatten(),
        nn.Linear(32 * math.ceil(input_shape[1] / 8 / 3), 128), _activation("relu"),
        nn.Linear(128, 32), _activation("relu"),
        nn.Dropout(0.2),
    ]

    # Output
    if num_classes == 1:
        layers.append(nn.Linear(32, 1))
    elif num_classes == 2:
        layers.append(nn.Linear(32, 1))
        layers.append(nn.Sigmoid())
    else:
        layers.append(nn.Linear(32, num_classes))
        layers.append(nn.Softmax(dim=1))

    return nn.Sequential(*layers)


def _build_transformer(input_shape: Tuple[int, int], params: Dict, *, num_classes: int = 1):
    """1‑D Transformer with `nn.TransformerEncoder` blocks."""

    c, seq_len = input_shape
    head_size = params.get("head_size", 16)
    num_heads = params.get("num_heads", 2)
    ff_dim = params.get("ff_dim", 8)
    num_blocks = params.get("num_transformer_blocks", 1)
    dropout = params.get("dropout", 0.05)

    d_model = head_size * num_heads

    # Project the input channels to d_model, keeping length dimension
    proj_in = nn.Conv1d(c, d_model, 1)

    # Transformer encoder expects (L, N, E)
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=num_heads,
        dim_feedforward=ff_dim * d_model,
        batch_first=False,
        dropout=dropout,
        activation="relu",
    )
    encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_blocks)

    pooling = GlobalAveragePooling1d()

    mlp_units = params.get("mlp_units", [32, 8])
    mlp_dropout = params.get("mlp_dropout", 0.1)
    mlp: List[nn.Module] = []
    in_size = d_model
    for dim in mlp_units:
        mlp += [nn.Linear(in_size, dim), _activation("relu"), nn.Dropout(mlp_dropout)]
        in_size = dim

    # Output layer
    if num_classes == 1:
        out_layer = nn.Linear(in_size, 1)
    elif num_classes == 2:
        out_layer = nn.Sequential(nn.Linear(in_size, 1), nn.Sigmoid())
    else:
        out_layer = nn.Sequential(nn.Linear(in_size, num_classes), nn.Softmax(dim=1))

    class _Transformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj_in = proj_in
            self.encoder = encoder
            self.pool = pooling
            self.mlp = nn.Sequential(*mlp)
            self.out = out_layer

        def forward(self, x):  # x: (N, C, L)
            z = self.proj_in(x)  # (N, d_model, L)
            z = z.permute(2, 0, 1)  # (L, N, d_model)
            z = self.encoder(z)
            z = z.permute(1, 2, 0)  # (N, d_model, L)
            z = self.pool(z)  # (N, d_model)
            z = self.mlp(z)
            return self.out(z)

    return _Transformer()


# -----------------------------------------------------------------------------
#  Public API — function names must match the original tensorflow file
# -----------------------------------------------------------------------------

from nirs4all.core.utils import framework  # noqa: E402, pylint: disable=C0413

# --- DECON family ------------------------------------------------------------

@framework("pytorch")
def decon(input_shape, params={}):
    return _build_decon(input_shape, params, num_classes=1)


@framework("pytorch")
def decon_classification(input_shape, num_classes=2, params={}):
    return _build_decon(input_shape, params, num_classes=num_classes)


@framework("pytorch")
def decon_Sep(input_shape, params={}):
    # Alias to standard decon for now – separable path is handled internally
    return _build_decon(input_shape, params, num_classes=1)


@framework("pytorch")
def decon_Sep_classification(input_shape, num_classes=2, params={}):
    return _build_decon(input_shape, params, num_classes=num_classes)


# --- Transformer family ------------------------------------------------------

@framework("pytorch")
def transformer(input_shape, params={}):
    return _build_transformer(input_shape, params, num_classes=1)


@framework("pytorch")
def transformer_VG(input_shape, params={}):
    # Passthrough with different defaults handled in params already
    return _build_transformer(input_shape, params, num_classes=1)


@framework("pytorch")
def transformer_classification(input_shape, num_classes=2, params={}):
    return _build_transformer(input_shape, params, num_classes=num_classes)


@framework("pytorch")
def transformer_VG_classification(input_shape, num_classes=2, params={}):
    return _build_transformer(input_shape, params, num_classes=num_classes)


# --- NICON family (simplified) ----------------------------------------------

def _build_nicon(input_shape, params, *, num_classes: int = 1):
    c, _ = input_shape
    layers: List[nn.Module] = [SpatialDropout1d(params.get("spatial_dropout", 0.08))]
    layers += [
        nn.Conv1d(c, params.get("filters1", 8), kernel_size=15, stride=5), _activation("selu"),
        nn.Dropout(params.get("dropout_rate", 0.2)),
        nn.Conv1d(params.get("filters1", 8), params.get("filters2", 64), kernel_size=21, stride=3), _activation("relu"),
        nn.BatchNorm1d(params.get("filters2", 64)),
        nn.Conv1d(params.get("filters2", 64), params.get("filters3", 32), kernel_size=5, stride=3), _activation("elu"),
        nn.BatchNorm1d(params.get("filters3", 32)),
        nn.Flatten(),
        nn.Linear(params.get("filters3", 32), params.get("dense_units", 16)), _activation("sigmoid"),
    ]

    if num_classes == 1:
        layers.append(nn.Linear(params.get("dense_units", 16), 1))
    elif num_classes == 2:
        layers += [nn.Linear(params.get("dense_units", 16), 1), nn.Sigmoid()]
    else:
        layers += [nn.Linear(params.get("dense_units", 16), num_classes), nn.Softmax(dim=1)]

    return nn.Sequential(*layers)


@framework("pytorch")
def nicon(input_shape, params={}):
    return _build_nicon(input_shape, params, num_classes=1)


@framework("pytorch")
def nicon_classification(input_shape, num_classes=2, params={}):
    return _build_nicon(input_shape, params, num_classes=num_classes)


@framework("pytorch")
def nicon_VG(input_shape, params={}):
    return _build_nicon(input_shape, params, num_classes=1)


@framework("pytorch")
def nicon_VG_classification(input_shape, num_classes=2, params={}):
    return _build_nicon(input_shape, params, num_classes=num_classes)


# --- Customizable variants ---------------------------------------------------

@framework("pytorch")
def customizable_nicon(input_shape, params={}):
    return _build_nicon(input_shape, params, num_classes=1)


@framework("pytorch")
def customizable_nicon_classification(input_shape, num_classes=2, params={}):
    return _build_nicon(input_shape, params, num_classes=num_classes)


@framework("pytorch")
def customizable_decon(input_shape, params={}):
    return _build_decon(input_shape, params, num_classes=1)


@framework("pytorch")
def decon_layer_classification(input_shape, num_classes=2, params={}):
    return _build_decon(input_shape, params, num_classes=num_classes)
