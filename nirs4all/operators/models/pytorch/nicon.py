import math
from typing import List, Tuple, Dict, Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from nirs4all.utils import framework

# -----------------------------------------------------------------------------
#  Layers & Helpers
# -----------------------------------------------------------------------------

class SpatialDropout1D(nn.Module):
    """
    Spatial Dropout 1D (drops entire channels).
    Expects input shape: (Batch, Channels, Length)
    """
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0:
            return x
        # x is (N, C, L). We want to drop channels.
        # nn.Dropout2d accepts (N, C, H, W). We can treat L as H and W=1.
        x = x.unsqueeze(-1)  # (N, C, L, 1)
        x = F.dropout2d(x, self.p, training=True)
        return x.squeeze(-1)

class DepthwiseConv1D(nn.Module):
    def __init__(self, in_channels, kernel_size, stride=1, padding='same', depth_multiplier=1, bias=True, activation=None):
        super().__init__()
        self.padding_mode = padding
        self.stride = stride
        self.kernel_size = kernel_size

        out_channels = in_channels * depth_multiplier

        # Calculate padding
        self.pad_val = 0
        if padding == 'same':
            if kernel_size % 2 != 0:
                self.pad_val = kernel_size // 2
            else:
                self.pad_val = kernel_size // 2
        elif padding == 'valid':
            self.pad_val = 0

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.pad_val,
            groups=in_channels,
            bias=bias
        )
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        if self.activation:
            x = self.activation(x)
        return x

class SeparableConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='same', depth_multiplier=1, bias=True, activation=None):
        super().__init__()
        self.depthwise = DepthwiseConv1D(
            in_channels, kernel_size, stride, padding, depth_multiplier, bias=bias
        )
        self.pointwise = nn.Conv1d(
            in_channels * depth_multiplier, out_channels, kernel_size=1, bias=bias
        )
        self.activation = activation

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        if self.activation:
            x = self.activation(x)
        return x

class GlobalAveragePooling1D(nn.Module):
    def forward(self, x):
        return x.mean(dim=2)

class LayerNorm1D(nn.Module):
    """Layer Normalization over the channel dimension for (N, C, L) input."""
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        # We use GroupNorm with 1 group to simulate LayerNorm over channels
        self.norm = nn.GroupNorm(1, num_features, eps=eps)

    def forward(self, x):
        return self.norm(x)

def get_activation(name: str):
    if name is None:
        return None
    name = name.lower()
    if name == "swish":
        return nn.SiLU()
    elif name == "selu":
        return nn.SELU()
    elif name == "elu":
        return nn.ELU()
    elif name == "relu":
        return nn.ReLU()
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "softmax":
        return nn.Softmax(dim=1)
    elif name == "tanh":
        return nn.Tanh()
    else:
        return nn.ReLU()

def get_norm(method: str, num_features: int):
    if method == "LayerNormalization":
        return LayerNorm1D(num_features)
    return nn.BatchNorm1d(num_features)

# -----------------------------------------------------------------------------
#  Model Builders
# -----------------------------------------------------------------------------

def _build_decon(input_shape, params, num_classes=1):
    c, seq_len = input_shape
    layers = []
    layers.append(SpatialDropout1D(params.get('spatial_dropout', 0.2)))

    # Block 1
    layers.append(DepthwiseConv1D(c, 7, padding='same', depth_multiplier=2, activation=nn.ReLU()))
    layers.append(DepthwiseConv1D(c * 2, 7, padding='same', depth_multiplier=2, activation=nn.ReLU()))
    layers.append(nn.MaxPool1d(2, 2))
    layers.append(nn.BatchNorm1d(c * 4))
    seq_len = math.floor((seq_len - 2) / 2 + 1)

    # Block 2
    c_curr = c * 4
    layers.append(DepthwiseConv1D(c_curr, 5, padding='same', depth_multiplier=2, activation=nn.ReLU()))
    layers.append(DepthwiseConv1D(c_curr * 2, 5, padding='same', depth_multiplier=2, activation=nn.ReLU()))
    layers.append(nn.MaxPool1d(2, 2))
    layers.append(nn.BatchNorm1d(c_curr * 4))
    seq_len = math.floor((seq_len - 2) / 2 + 1)

    # Block 3
    c_curr = c_curr * 4
    layers.append(DepthwiseConv1D(c_curr, 9, padding='same', depth_multiplier=2, activation=nn.ReLU()))
    layers.append(DepthwiseConv1D(c_curr * 2, 9, padding='same', depth_multiplier=2, activation=nn.ReLU()))
    layers.append(nn.MaxPool1d(2, 2))
    layers.append(nn.BatchNorm1d(c_curr * 4))
    seq_len = math.floor((seq_len - 2) / 2 + 1)

    # Separable + Conv
    c_curr = c_curr * 4
    layers.append(SeparableConv1D(c_curr, 64, 3, padding='same', depth_multiplier=1, activation=nn.ReLU()))
    layers.append(nn.Conv1d(64, 32, 3, padding=1))
    layers.append(nn.ReLU())
    layers.append(nn.MaxPool1d(5, 3))
    seq_len = math.floor((seq_len - 5) / 3 + 1)

    layers.append(SpatialDropout1D(0.1))
    layers.append(nn.Flatten())

    layers.append(nn.Linear(32 * seq_len, 128))
    layers.append(nn.ReLU())
    layers.append(nn.Linear(128, 32))
    layers.append(nn.ReLU())
    layers.append(nn.Dropout(0.2))

    if num_classes == 1:
        layers.append(nn.Linear(32, 1))
        layers.append(nn.Sigmoid())
    elif num_classes == 2:
        layers.append(nn.Linear(32, 1))
        layers.append(nn.Sigmoid())
    else:
        layers.append(nn.Linear(32, num_classes))
        layers.append(nn.Softmax(dim=1))

    return nn.Sequential(*layers)

def _build_decon_sep(input_shape, params, num_classes=1):
    c, seq_len = input_shape
    layers = []
    layers.append(SpatialDropout1D(params.get('spatial_dropout', 0.2)))

    # Sep 1
    f1 = params.get('filters1', 64)
    k1 = params.get('kernel_size1', 3)
    s1 = params.get('strides1', 2)
    d1 = params.get('depth_multiplier1', 32)
    layers.append(SeparableConv1D(c, f1, k1, stride=s1, padding='same', depth_multiplier=d1, activation=nn.ReLU()))
    layers.append(nn.BatchNorm1d(f1))
    seq_len = math.ceil(seq_len / s1)

    # Sep 2
    f2 = params.get('filters2', 64)
    k2 = params.get('kernel_size2', 3)
    s2 = params.get('strides2', 2)
    d2 = params.get('depth_multiplier2', 32)
    layers.append(SeparableConv1D(f1, f2, k2, stride=s2, padding='same', depth_multiplier=d2, activation=nn.ReLU()))
    layers.append(nn.BatchNorm1d(f2))
    seq_len = math.ceil(seq_len / s2)

    # Sep 3
    f3 = params.get('filters3', 64)
    k3 = params.get('kernel_size3', 3)
    d3 = params.get('depth_multiplier3', 32)
    layers.append(SeparableConv1D(f2, f3, k3, stride=1, padding='same', depth_multiplier=d3, activation=nn.ReLU()))
    layers.append(nn.BatchNorm1d(f3))

    # Sep 4
    f4 = params.get('filters4', 64)
    k4 = params.get('kernel_size4', 3)
    d4 = params.get('depth_multiplier4', 32)
    layers.append(SeparableConv1D(f3, f4, k4, stride=1, padding='same', depth_multiplier=d4, activation=nn.ReLU()))
    layers.append(nn.BatchNorm1d(f4))

    # Conv 5
    f5 = params.get('filters5', 32)
    k5 = params.get('kernel_size5', 5)
    s5 = params.get('strides5', 2)
    pad5 = k5 // 2 if k5 % 2 != 0 else 0
    layers.append(nn.Conv1d(f4, f5, k5, stride=s5, padding=pad5))
    layers.append(nn.ReLU())
    seq_len = math.ceil(seq_len / s5)

    layers.append(nn.Flatten())
    layers.append(nn.BatchNorm1d(f5 * seq_len))

    du = params.get('dense_units', 32)
    layers.append(nn.Linear(f5 * seq_len, du))
    layers.append(nn.ReLU())
    layers.append(nn.Dropout(params.get('dropout_rate', 0.2)))

    if num_classes == 1:
        layers.append(nn.Linear(du, 1))
        layers.append(nn.Sigmoid())
    elif num_classes == 2:
        layers.append(nn.Linear(du, 1))
        layers.append(nn.Sigmoid())
    else:
        layers.append(nn.Linear(du, num_classes))
        layers.append(nn.Softmax(dim=1))

    return nn.Sequential(*layers)

def _build_nicon(input_shape, params, num_classes=1):
    c, seq_len = input_shape
    layers = []
    layers.append(SpatialDropout1D(params.get('spatial_dropout', 0.08)))

    f1 = params.get('filters1', 8)
    layers.append(nn.Conv1d(c, f1, kernel_size=15, stride=5))
    layers.append(nn.SELU())
    seq_len = math.floor((seq_len - 15) / 5 + 1)

    layers.append(nn.Dropout(params.get('dropout_rate', 0.2)))

    f2 = params.get('filters2', 64)
    layers.append(nn.Conv1d(f1, f2, kernel_size=21, stride=3))
    layers.append(nn.ReLU())
    layers.append(nn.BatchNorm1d(f2))
    seq_len = math.floor((seq_len - 21) / 3 + 1)

    f3 = params.get('filters3', 32)
    layers.append(nn.Conv1d(f2, f3, kernel_size=5, stride=3))
    layers.append(nn.ELU())
    layers.append(nn.BatchNorm1d(f3))
    seq_len = math.floor((seq_len - 5) / 3 + 1)

    layers.append(nn.Flatten())

    du = params.get('dense_units', 16)
    layers.append(nn.Linear(f3 * seq_len, du))
    layers.append(nn.Sigmoid())

    if num_classes == 1:
        layers.append(nn.Linear(du, 1))
        layers.append(nn.Sigmoid())
    elif num_classes == 2:
        layers.append(nn.Linear(du, 1))
        layers.append(nn.Sigmoid())
    else:
        layers.append(nn.Linear(du, num_classes))
        layers.append(nn.Softmax(dim=1))

    return nn.Sequential(*layers)

def _build_customizable_nicon(input_shape, params, num_classes=1):
    c, seq_len = input_shape
    layers = []
    layers.append(SpatialDropout1D(params.get('spatial_dropout', 0.08)))

    f1 = params.get('filters1', 8)
    k1 = params.get('kernel_size1', 15)
    s1 = params.get('strides1', 5)
    act1 = get_activation(params.get('activation1', "selu"))
    layers.append(nn.Conv1d(c, f1, kernel_size=k1, stride=s1))
    layers.append(act1)
    seq_len = math.floor((seq_len - k1) / s1 + 1)

    layers.append(nn.Dropout(params.get('dropout_rate', 0.2)))

    f2 = params.get('filters2', 64)
    k2 = params.get('kernel_size2', 21)
    s2 = params.get('strides2', 3)
    act2 = get_activation(params.get('activation2', "relu"))
    layers.append(nn.Conv1d(f1, f2, kernel_size=k2, stride=s2))
    layers.append(act2)
    norm1 = get_norm(params.get('normalization_method1', "BatchNormalization"), f2)
    layers.append(norm1)
    seq_len = math.floor((seq_len - k2) / s2 + 1)

    f3 = params.get('filters3', 32)
    k3 = params.get('kernel_size3', 5)
    s3 = params.get('strides3', 3)
    act3 = get_activation(params.get('activation3', "elu"))
    layers.append(nn.Conv1d(f2, f3, kernel_size=k3, stride=s3))
    layers.append(act3)
    norm2 = get_norm(params.get('normalization_method2', "BatchNormalization"), f3)
    layers.append(norm2)
    seq_len = math.floor((seq_len - k3) / s3 + 1)

    layers.append(nn.Flatten())

    du = params.get('dense_units', 16)
    act_d = get_activation(params.get('dense_activation', "sigmoid"))
    layers.append(nn.Linear(f3 * seq_len, du))
    layers.append(act_d)

    if num_classes == 1:
        layers.append(nn.Linear(du, 1))
        layers.append(nn.Sigmoid())
    elif num_classes == 2:
        layers.append(nn.Linear(du, 1))
        layers.append(nn.Sigmoid())
    else:
        layers.append(nn.Linear(du, num_classes))
        layers.append(nn.Softmax(dim=1))

    return nn.Sequential(*layers)

def _build_thin_nicon(input_shape, params, num_classes=1):
    c, seq_len = input_shape
    layers = []
    layers.append(SpatialDropout1D(params.get('spatial_dropout', 0.08)))

    f1 = params.get('filters1', 8)
    layers.append(nn.Conv1d(c, f1, kernel_size=7, stride=3))
    layers.append(nn.SELU())
    seq_len = math.floor((seq_len - 7) / 3 + 1)

    layers.append(nn.Dropout(params.get('dropout_rate', 0.2)))

    f2 = params.get('filters2', 64)
    layers.append(nn.Conv1d(f1, f2, kernel_size=11, stride=2))
    layers.append(nn.ReLU())
    layers.append(nn.BatchNorm1d(f2))
    seq_len = math.floor((seq_len - 11) / 2 + 1)

    f3 = params.get('filters3', 32)
    layers.append(nn.Conv1d(f2, f3, kernel_size=3, stride=2))
    layers.append(nn.ELU())
    layers.append(nn.BatchNorm1d(f3))
    seq_len = math.floor((seq_len - 3) / 2 + 1)

    layers.append(nn.Flatten())

    du = params.get('dense_units', 16)
    layers.append(nn.Linear(f3 * seq_len, du))
    layers.append(nn.Sigmoid())

    if num_classes == 1:
        layers.append(nn.Linear(du, 1))
        layers.append(nn.Sigmoid())
    elif num_classes == 2:
        layers.append(nn.Linear(du, 1))
        layers.append(nn.Sigmoid())
    else:
        layers.append(nn.Linear(du, num_classes))
        layers.append(nn.Softmax(dim=1))

    return nn.Sequential(*layers)

def _build_nicon_vg(input_shape, params, num_classes=1):
    c, seq_len = input_shape
    layers = []
    layers.append(SpatialDropout1D(params.get('spatial_dropout', 0.2)))

    f1 = params.get('filters1', 64)
    layers.append(nn.Conv1d(c, f1, kernel_size=3, padding=1))
    layers.append(nn.SiLU())

    f2 = params.get('filters2', 64)
    layers.append(nn.Conv1d(f1, f2, kernel_size=3, padding=1))
    layers.append(nn.SiLU())

    layers.append(nn.MaxPool1d(5, 3))
    seq_len = math.floor((seq_len - 5) / 3 + 1)

    layers.append(SpatialDropout1D(params.get('spatial_dropout', 0.2)))

    f3 = params.get('filters3', 128)
    layers.append(nn.Conv1d(f2, f3, kernel_size=3, padding=1))
    layers.append(nn.SiLU())

    f4 = params.get('filters4', 128)
    layers.append(nn.Conv1d(f3, f4, kernel_size=3, padding=1))
    layers.append(nn.SiLU())

    layers.append(nn.MaxPool1d(5, 3))
    seq_len = math.floor((seq_len - 5) / 3 + 1)

    layers.append(SpatialDropout1D(params.get('spatial_dropout', 0.2)))
    layers.append(nn.Flatten())

    du1 = params.get('dense_units1', 1024)
    layers.append(nn.Linear(f4 * seq_len, du1))
    layers.append(nn.ReLU())

    layers.append(nn.Dropout(params.get('dropout_rate', 0.2)))

    du2 = params.get('dense_units2', 1024)
    layers.append(nn.Linear(du1, du2))
    layers.append(nn.ReLU())

    if num_classes == 1:
        layers.append(nn.Linear(du2, 1))
        layers.append(nn.Sigmoid())
    elif num_classes == 2:
        layers.append(nn.Linear(du2, 1))
        layers.append(nn.Sigmoid())
    else:
        layers.append(nn.Linear(du2, num_classes))
        layers.append(nn.Softmax(dim=1))

    return nn.Sequential(*layers)

def _build_customizable_decon(input_shape, params, num_classes=1):
    c, seq_len = input_shape
    layers = []

    # Block 1
    layers.append(SpatialDropout1D(params.get('spatial_dropout1', 0.2)))

    k1 = params.get('kernel_size1', 7)
    d1 = params.get('depth_multiplier1', 2)
    act1 = get_activation(params.get('activationDCNN1', "relu"))
    layers.append(DepthwiseConv1D(c, k1, padding='same', depth_multiplier=d1, activation=act1))

    c_curr = c * d1
    k2 = params.get('kernel_size2', 7)
    d2 = params.get('depth_multiplier2', 2)
    act2 = get_activation(params.get('activationDCNN2', "relu"))
    layers.append(DepthwiseConv1D(c_curr, k2, padding='same', depth_multiplier=d2, activation=act2))

    c_curr = c_curr * d2
    p1 = params.get('pool_size1', 2)
    s1 = params.get('strides1', 2)
    layers.append(nn.MaxPool1d(p1, s1))

    layers.append(LayerNorm1D(c_curr))

    # Block 2
    k3 = params.get('kernel_size3', 5)
    d3 = params.get('depth_multiplier3', 2)
    act3 = get_activation(params.get('activationDCNN3', "relu"))
    layers.append(DepthwiseConv1D(c_curr, k3, padding='same', depth_multiplier=d3, activation=act3))

    c_curr = c_curr * d3
    k4 = params.get('kernel_size4', 5)
    d4 = params.get('depth_multiplier4', 2)
    act4 = get_activation(params.get('activationDCNN4', "relu"))
    layers.append(DepthwiseConv1D(c_curr, k4, padding='same', depth_multiplier=d4, activation=act4))

    c_curr = c_curr * d4
    p2 = params.get('pool_size2', 2)
    s2 = params.get('strides2', 2)
    layers.append(nn.MaxPool1d(p2, s2))

    layers.append(LayerNorm1D(c_curr))

    # Block 3
    k5 = params.get('kernel_size5', 9)
    d5 = params.get('depth_multiplier5', 2)
    act5 = get_activation(params.get('activationDCNN5', "relu"))
    layers.append(DepthwiseConv1D(c_curr, k5, padding='same', depth_multiplier=d5, activation=act5))

    c_curr = c_curr * d5
    k6 = params.get('kernel_size6', 9)
    d6 = params.get('depth_multiplier6', 2)
    act6 = get_activation(params.get('activationDCNN6', "relu"))
    layers.append(DepthwiseConv1D(c_curr, k6, padding='same', depth_multiplier=d6, activation=act6))

    c_curr = c_curr * d6
    p3 = params.get('pool_size3', 2)
    s3 = params.get('strides3', 2)
    layers.append(nn.MaxPool1d(p3, s3))

    layers.append(LayerNorm1D(c_curr))

    # Final Conv
    sep_f = params.get('separable_filters', 64)
    sep_k = params.get('separable_kernel_size', 3)
    sep_d = params.get('separable_depth_multiplier', 1)
    act_cnn1 = get_activation(params.get('activationCNN1', "relu"))
    layers.append(SeparableConv1D(c_curr, sep_f, sep_k, padding='same', depth_multiplier=sep_d, activation=act_cnn1))

    conv_f = params.get('conv_filters', 32)
    conv_k = params.get('conv_kernel_size', 3)
    pad_conv = conv_k // 2 if conv_k % 2 != 0 else 0
    layers.append(nn.Conv1d(sep_f, conv_f, conv_k, padding=pad_conv))

    fp = params.get('final_pool_size', 5)
    fs = params.get('final_pool_strides', 3)
    layers.append(nn.MaxPool1d(fp, fs))

    layers.append(SpatialDropout1D(params.get('spatial_dropout2', 0.1)))
    layers.append(nn.Flatten())

    # Use LazyLinear to infer input size at runtime
    du1 = params.get('dense_units1', 128)
    act_d1 = get_activation(params.get('activationDense1', "relu"))
    layers.append(nn.LazyLinear(du1))
    layers.append(act_d1)

    du2 = params.get('dense_units2', 32)
    act_d2 = get_activation(params.get('activationDense2', "relu"))
    layers.append(nn.Linear(du1, du2))
    layers.append(act_d2)

    layers.append(nn.Dropout(params.get('dropout_rate', 0.2)))

    out_units = params.get('output_units', 1)
    act_out = get_activation(params.get('activationDense3', "sigmoid"))

    if num_classes == 1:
        layers.append(nn.Linear(du2, out_units))
        layers.append(act_out)
    elif num_classes == 2:
        layers.append(nn.Linear(du2, 1))
        layers.append(nn.Sigmoid())
    else:
        layers.append(nn.Linear(du2, num_classes))
        layers.append(nn.Softmax(dim=1))

    return nn.Sequential(*layers)

def _build_transformer(input_shape, params, num_classes=1):
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
        dim_feedforward=ff_dim,
        batch_first=False,
        dropout=dropout,
        activation="relu",
    )
    encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_blocks)

    pooling = GlobalAveragePooling1D()

    mlp_units = params.get("mlp_units", [32, 8])
    mlp_dropout = params.get("mlp_dropout", 0.1)
    mlp = []
    in_size = d_model
    for dim in mlp_units:
        mlp.append(nn.Linear(in_size, dim))
        mlp.append(nn.ReLU())
        mlp.append(nn.Dropout(mlp_dropout))
        in_size = dim

    mlp_seq = nn.Sequential(*mlp)

    # Output layer
    if num_classes == 1:
        out_layer = nn.Sequential(nn.Linear(in_size, 1), nn.Sigmoid())
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
            self.mlp = mlp_seq
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
#  Public API
# -----------------------------------------------------------------------------

@framework("pytorch")
def decon(input_shape, params={}):
    return _build_decon(input_shape, params, num_classes=1)

@framework("pytorch")
def decon_classification(input_shape, num_classes=2, params={}):
    return _build_decon(input_shape, params, num_classes=num_classes)

@framework("pytorch")
def decon_Sep(input_shape, params={}):
    return _build_decon_sep(input_shape, params, num_classes=1)

@framework("pytorch")
def decon_Sep_classification(input_shape, num_classes=2, params={}):
    return _build_decon_sep(input_shape, params, num_classes=num_classes)

@framework("pytorch")
def nicon(input_shape, params={}):
    return _build_nicon(input_shape, params, num_classes=1)

@framework("pytorch")
def nicon_classification(input_shape, num_classes=2, params={}):
    return _build_nicon(input_shape, params, num_classes=num_classes)

@framework("pytorch")
def customizable_nicon(input_shape, params={}):
    return _build_customizable_nicon(input_shape, params, num_classes=1)

@framework("pytorch")
def customizable_nicon_classification(input_shape, num_classes=2, params={}):
    return _build_customizable_nicon(input_shape, params, num_classes=num_classes)

@framework("pytorch")
def thin_nicon(input_shape, params={}):
    return _build_thin_nicon(input_shape, params, num_classes=1)

@framework("pytorch")
def nicon_VG(input_shape, params={}):
    return _build_nicon_vg(input_shape, params, num_classes=1)

@framework("pytorch")
def nicon_VG_classification(input_shape, num_classes=2, params={}):
    return _build_nicon_vg(input_shape, params, num_classes=num_classes)

@framework("pytorch")
def customizable_decon(input_shape, params={}):
    return _build_customizable_decon(input_shape, params, num_classes=1)

@framework("pytorch")
def customizable_decon_classification(input_shape, num_classes=2, params={}):
    return _build_customizable_decon(input_shape, params, num_classes=num_classes)

@framework("pytorch")
def decon_layer_classification(input_shape, num_classes=2, params={}):
    return _build_customizable_decon(input_shape, params, num_classes=num_classes)

@framework("pytorch")
def transformer(input_shape, params={}):
    return _build_transformer(input_shape, params, num_classes=1)

@framework("pytorch")
def transformer_VG(input_shape, params={}):
    return _build_transformer(input_shape, params, num_classes=1)

@framework("pytorch")
def transformer_classification(input_shape, num_classes=2, params={}):
    return _build_transformer(input_shape, params, num_classes=num_classes)

@framework("pytorch")
def transformer_VG_classification(input_shape, num_classes=2, params={}):
    return _build_transformer(input_shape, params, num_classes=num_classes)

@framework("pytorch")
def transformer_model(input_shape, params={}):
    return _build_transformer(input_shape, params, num_classes=1)

@framework("pytorch")
def transformer_model_classification(input_shape, num_classes=2, params={}):
    return _build_transformer(input_shape, params, num_classes=num_classes)

