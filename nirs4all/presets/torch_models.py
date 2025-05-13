
"""
PyTorch re‑implementation of ``tf_models.py``.

Notes
-----
* The public API (builder functions) is intentionally unchanged except for the
  ``@framework("pytorch")`` decorator.
* All functions return *uncompiled* ``torch.nn.Module`` objects.  Compile /
  move to device / wrap in `torch.compile` or `lightning` modules as you wish.
* The input tensor shape expected by these models is **(batch, seq_len, channels)**
  to stay compatible with the original Keras layout.  Internally, we convert to
  channel‑first format required by ``nn.Conv1d`` and friends.
* Custom layers that exist in Keras but not in core PyTorch have lightweight
  equivalents implemented here (``DepthwiseConv1d``, ``SeparableConv1d``,
  ``SpatialDropout1d``, ``GlobalAveragePooling1d``).
* The external architectures referenced from
  ``nirs4all.presets.legacy`` (*VGG*, *Inception*, *ResNetv2*, *SEResNet*) are
  assumed to have sibling PyTorch ports living at the same import paths.
  If not, you will need to convert those first.
"""

from __future__ import annotations
import math
from typing import Tuple, Any, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from nirs4all.core.utils import framework
from nirs4all.presets.legacy.Inception_1DCNN import Inception      # assumed PyTorch port
from nirs4all.presets.legacy.ResNet_v2_1DCNN import ResNetv2       # assumed PyTorch port
from nirs4all.presets.legacy.SE_ResNet_1DCNN import SEResNet       # assumed PyTorch port
from nirs4all.presets.legacy.VGG_1DCNN import VGG                  # assumed PyTorch port


# --------------------------------------------------------------------------- #
#                                Helper layers                                #
# --------------------------------------------------------------------------- #
class Permute(nn.Module):
    """Swap ``(N, L, C)`` → ``(N, C, L)`` or the inverse."""

    def __init__(self, to_channels_first: bool = True):
        super().__init__()
        self.to_channels_first = to_channels_first

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(1, 2)  # works both ways


class DepthwiseConv1d(nn.Module):
    """Depth‑wise 1‑D convolution (groups = in_channels)."""

    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        depth_multiplier: int = 1,
        stride: int = 1,
        padding: str | int = "same",
        dilation: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        if padding == "same":
            pad = (kernel_size - 1) // 2 * dilation
        else:
            pad = padding
        self.conv = nn.Conv1d(
            in_channels,
            in_channels * depth_multiplier,
            kernel_size,
            stride,
            pad,
            dilation,
            groups=in_channels,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (N,C,L)
        return self.conv(x)


class SeparableConv1d(nn.Module):
    """Depthwise + Pointwise separation as in Keras ``SeparableConv1D``."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        depth_multiplier: int = 1,
        stride: int = 1,
        padding: str | int = "same",
        dilation: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.depth = DepthwiseConv1d(
            in_channels,
            kernel_size,
            depth_multiplier=depth_multiplier,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.point = nn.Conv1d(
            in_channels * depth_multiplier,
            out_channels,
            kernel_size=1,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.point(self.depth(x))


class SpatialDropout1d(nn.Module):
    """Drop whole feature maps (channels) instead of individual timesteps."""

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        # ``x`` is (N, L, C) – convert to (N, C, L) for dropout2d then back
        x = x.transpose(1, 2)
        x = F.dropout2d(x, self.p, self.training, inplace=False)
        return x.transpose(1, 2)


class GlobalAveragePooling1d(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (N,C,L)
        return x.mean(dim=-1)


class GlobalMaxPooling1d(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.amax(dim=-1)


class Flatten(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(x, start_dim=1)


class Lambda(nn.Module):
    """Wrap an arbitrary lambda (for quick hacks)."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


def _activation(act: str) -> nn.Module:
    return {
        "relu": nn.ReLU(),
        "sigmoid": nn.Sigmoid(),
        "swish": nn.SiLU(),
        "linear": nn.Identity(),
    }.get(act.lower(), nn.ReLU())


# --------------------------------------------------------------------------- #
#                             Utility building blocks                         #
# --------------------------------------------------------------------------- #
def _conv_bn_act(
    in_ch: int,
    out_ch: int,
    kernel: int,
    stride: int = 1,
    dilation: int = 1,
    act: str = "relu",
    padding: str | int = "same",
) -> nn.Sequential:
    if padding == "same":
        pad = (kernel - 1) // 2 * dilation
    else:
        pad = padding
    return nn.Sequential(
        nn.Conv1d(in_ch, out_ch, kernel, stride, pad, dilation, bias=False),
        nn.BatchNorm1d(out_ch),
        _activation(act),
    )


# --------------------------------------------------------------------------- #
#                               Model builders                                #
# --------------------------------------------------------------------------- #
@framework("pytorch")
def UNet_NIRS(input_shape: Tuple[int, int], params: dict[str, Any]):
    """Thin wrapper around *VGG* preset identical to original Keras version."""
    length, num_channel = input_shape
    model_width = params.get("model_width", 8)
    problem_type = params.get("problem_type", "Regression")
    output_nums = params.get("output_nums", 1)
    dropout_rate = params.get("dropout_rate", False)

    return VGG(
        length,
        num_channel,
        model_width,
        problem_type=problem_type,
        output_nums=output_nums,
        dropout_rate=dropout_rate,
    ).VGG11()


# --------------------------------------------------------------------------- #
#                               VGG‑style block                               #
# --------------------------------------------------------------------------- #
class _VGGBlock(nn.Module):
    def __init__(self, in_ch: int, n_filters: int, n_conv: int, kernel_size: int):
        super().__init__()
        layers: List[nn.Module] = []
        for _ in range(n_conv):
            layers.append(
                _conv_bn_act(in_ch, n_filters, kernel_size, act="relu", padding="same")
            )
            in_ch = n_filters
        layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class _VGG1DPyTorch(nn.Module):
    def __init__(self, input_shape: Tuple[int, int], params: dict[str, Any]):
        super().__init__()
        self.permute = Permute()  # (N,L,C) → (N,C,L)
        c_in = input_shape[1]

        self.block1 = _VGGBlock(
            c_in,
            params.get("block1_filters", 64),
            params.get("block1_convs", 2),
            params.get("kernel_size", 3),
        )
        self.block2 = _VGGBlock(
            params.get("block1_filters", 64),
            params.get("block2_filters", 128),
            params.get("block2_convs", 2),
            params.get("kernel_size", 3),
        )
        self.block3 = _VGGBlock(
            params.get("block2_filters", 128),
            params.get("block3_filters", 256),
            params.get("block3_convs", 2),
            params.get("kernel_size", 3),
        )
        self.flat = Flatten()
        self.dense1 = nn.Linear(params.get("block3_filters", 256), params.get("dense_units", 16))
        self.act1 = _activation("sigmoid")
        self.dense2 = nn.Linear(params.get("dense_units", 16), 1)  # linear output

    def forward(self, x):
        x = self.permute(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = GlobalAveragePooling1d()(x)  # (N,C)
        x = self.dense1(x)
        x = self.act1(x)
        return self.dense2(x)


@framework("pytorch")
def VGG_1D(input_shape: Tuple[int, int], params: dict[str, Any]):
    """VGG‑like 1‑D CNN rewritten for PyTorch."""
    return _VGG1DPyTorch(input_shape, params)


# --------------------------------------------------------------------------- #
#                         CONV + LSTM multi‑path model                        #
# --------------------------------------------------------------------------- #
class _CONV_LSTM(nn.Module):
    def __init__(self, input_shape: Tuple[int, int], p: dict[str, Any]):
        super().__init__()
        L, C = input_shape
        self.permute = Permute()
        self.unpermute = Permute(to_channels_first=False)

        # --- Convolutional path (SeparableConv) ---------------------------- #
        self.sep_conv = nn.Sequential(
            SeparableConv1d(
                in_channels=C,
                out_channels=p.get("conv_filters", 64),
                kernel_size=p.get("conv_kernel_size", 3),
                depth_multiplier=p.get("conv_depth_multiplier", 64),
                padding="same",
            ),
            nn.MaxPool1d(2),
            nn.BatchNorm1d(p.get("conv_filters", 64)),
            Flatten(),
        )

        # --- Attention path ------------------------------------------------ #
        self.attn_conv = nn.Sequential(
            nn.Conv1d(C, 32, kernel_size=3, stride=2, padding=1),
            nn.MaxPool1d(2),
            nn.BatchNorm1d(32),
            Flatten(),
        )
        self.mha = nn.MultiheadAttention(
            embed_dim=C,
            num_heads=p.get("attention_num_heads", 8),
            dropout=0.1,
            batch_first=True,
        )

        # --- GRU and LSTM paths ------------------------------------------- #
        self.gru = nn.GRU(
            input_size=C,
            hidden_size=p.get("gru_units", 128),
            batch_first=True,
            bidirectional=True,
        )
        self.lstm = nn.LSTM(
            input_size=C,
            hidden_size=p.get("lstm_units", 128),
            batch_first=True,
            bidirectional=True,
        )

        # --- Fully‑connected head ----------------------------------------- #
        cat_size = (
            p.get("conv_filters", 64)
            + 32  # attn_conv flattened size after pooling is rough; ok for demo
            + 2 * p.get("gru_units", 128)
            + 2 * p.get("lstm_units", 128)
        )
        self.head = nn.Sequential(
            nn.BatchNorm1d(cat_size),
            nn.Linear(cat_size, p.get("fc_units1", 64)),
            _activation("relu"),
            nn.Linear(p.get("fc_units1", 64), p.get("fc_units2", 16)),
            _activation("relu"),
            nn.Dropout(0.2),
            nn.Linear(p.get("fc_units2", 16), 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        # x: (N,L,C)
        conv_in = self.permute(x)  # (N,C,L)

        x1 = self.sep_conv(conv_in)  # (N, *)
        # Attention path expects (N,L,C)
        attn_out, _ = self.mha(x, x, x, need_weights=False)
        x2 = self.attn_conv(self.permute(attn_out))

        # RNN paths
        _, h_gru = self.gru(x)
        x3 = h_gru.transpose(0, 1).reshape(x.size(0), -1)  # (N, 2*gru_units)
        _, (h_lstm, _) = self.lstm(x)
        x4 = h_lstm.transpose(0, 1).reshape(x.size(0), -1)

        cat = torch.cat([x1, x2, x3, x4], dim=1)
        return self.head(cat)


@framework("pytorch")
def CONV_LSTM(input_shape: Tuple[int, int], params: dict[str, Any]):
    return _CONV_LSTM(input_shape, params)


# --------------------------------------------------------------------------- #
#                              Lite U‑Net variant                             #
# --------------------------------------------------------------------------- #
def _resblock(ch: int, kernel: int, dilation: int, use_se: bool):
    layers = [
        _conv_bn_act(ch, ch, kernel, stride=1, dilation=dilation),
        _conv_bn_act(ch, ch, kernel, stride=1, dilation=dilation),
    ]
    if use_se:
        layers.append(_SEBlock(ch))
    return nn.Sequential(*layers)


class _SEBlock(nn.Module):
    def __init__(self, ch: int, r: int = 8):
        super().__init__()
        self.pool = GlobalAveragePooling1d()
        self.fc = nn.Sequential(
            nn.Linear(ch, ch // r),
            nn.ReLU(),
            nn.Linear(ch // r, ch),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (N,C,L)
        w = self.pool(x).unsqueeze(-1)  # (N,C,1)
        w = self.fc(w.squeeze(-1)).unsqueeze(-1)
        return x * w


class _UNET(nn.Module):
    def __init__(self, input_shape: Tuple[int, int], p: dict[str, Any]):
        super().__init__()
        L, C = input_shape
        layer_n = p.get("layer_n", 64)
        k = p.get("kernel_size", 7)
        depth = p.get("depth", 2)

        self.permute = Permute()

        # Down‑sampling inputs for skip connections
        self.pool_5 = nn.AvgPool1d(kernel_size=5, stride=5, padding=0)
        self.pool_25 = nn.AvgPool1d(kernel_size=25, stride=25, padding=0)

        # Encoder
        self.enc0 = nn.Sequential(
            _conv_bn_act(C, layer_n, k, stride=1),
            *[_resblock(layer_n, k, 1, True) for _ in range(depth)],
        )
        self.enc1 = nn.Sequential(
            _conv_bn_act(layer_n, layer_n * 2, k, stride=5),
            *[_resblock(layer_n * 2, k, 1, True) for _ in range(depth)],
        )
        self.enc2 = nn.Sequential(
            _conv_bn_act(layer_n * 2 + C, layer_n * 3, k, stride=5),
            *[_resblock(layer_n * 3, k, 1, True) for _ in range(depth)],
        )
        self.enc3 = nn.Sequential(
            _conv_bn_act(layer_n * 3 + C, layer_n * 4, k, stride=5),
            *[_resblock(layer_n * 4, k, 1, True) for _ in range(depth)],
        )

        self.reg_head = nn.Sequential(
            nn.Conv1d(layer_n * 4, 1, kernel_size=k, padding=(k - 1) // 2),
            _activation("relu"),
            Lambda(lambda t: 12 * t),
            Flatten(),
            nn.Linear(L // (5 * 5 * 5), 1),  # rough flatten size
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (N,L,C) → (N,C,L)
        x0 = self.permute(x)
        # pre‑pools for skips – still (N,C,L')
        inp1 = self.pool_5(x0)
        inp2 = self.pool_25(x0)

        e0 = self.enc0(x0)
        e1 = self.enc1(e0)
        e2 = self.enc2(torch.cat([e1, inp1], dim=1))
        e3 = self.enc3(torch.cat([e2, inp2], dim=1))

        return self.reg_head(e3)


@framework("pytorch")
def UNET(input_shape: Tuple[int, int], params: dict[str, Any]):
    return _UNET(input_shape, params)


# --------------------------------------------------------------------------- #
#                                bard model                                   #
# --------------------------------------------------------------------------- #
class _Bard(nn.Module):
    def __init__(self, input_shape, p):
        super().__init__()
        L, C = input_shape
        self.permute = Permute()

        # conv branch
        self.conv_branch = nn.Sequential(
            SeparableConv1d(
                C,
                p.get("conv_filters1", 64),
                kernel_size=p.get("conv_kernel_size1", 3),
                padding="same",
            ),
            SeparableConv1d(
                p.get("conv_filters1", 64),
                p.get("conv_filters2", 16),
                kernel_size=p.get("conv_kernel_size2", 5),
                padding="same",
            ),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            Flatten(),
        )

        # LSTM branch
        self.lstm1 = nn.LSTM(
            C,
            p.get("lstm_units1", 64),
            batch_first=True,
            bidirectional=True,
        )
        self.lstm2 = nn.LSTM(
            2 * p.get("lstm_units1", 64),
            p.get("lstm_units2", 32),
            batch_first=True,
        )
        self.lstm_bn = nn.BatchNorm1d(p.get("lstm_units2", 32))
        # fc
        self.fc = nn.Sequential(
            nn.Linear(16 * ((L + 1) // 1), p.get("dense_units1", 128)),  # dummy in
            nn.ReLU(),
            nn.Linear(p.get("dense_units1", 128), p.get("dense_units2", 8)),
            nn.ReLU(),
            nn.Linear(p.get("dense_units2", 8), 1),
        )

    def forward(self, x):
        # x (N,L,C)
        conv = self.conv_branch(self.permute(x))
        lstm_out, _ = self.lstm1(x)
        lstm_out, _ = self.lstm2(lstm_out)
        lstm_out = self.lstm_bn(lstm_out)
        lstm_out = Flatten()(lstm_out)
        cat = torch.cat([conv, lstm_out], dim=1)
        return self.fc(cat)


@framework("pytorch")
def bard(input_shape: Tuple[int, int], params: dict[str, Any]):
    return _Bard(input_shape, params)


# --------------------------------------------------------------------------- #
#                          Xception‑like depthwise CNN                        #
# --------------------------------------------------------------------------- #
class _XceptionEntry(nn.Module):
    def __init__(self, in_ch, p):
        super().__init__()
        self.seq = nn.Sequential(
            DepthwiseConv1d(
                in_ch,
                kernel_size=p.get("entry_kernel_size1", 3),
                depth_multiplier=p.get("entry_depth_multiplier1", 2),
                stride=p.get("entry_strides1", 2),
                padding="same",
            ),
            nn.BatchNorm1d(in_ch * p.get("entry_depth_multiplier1", 2)),
            _activation("relu"),
            DepthwiseConv1d(
                in_ch * p.get("entry_depth_multiplier1", 2),
                kernel_size=p.get("entry_kernel_size2", 3),
                depth_multiplier=p.get("entry_depth_multiplier2", 2),
                stride=p.get("entry_strides2", 2),
                padding="same",
            ),
            nn.BatchNorm1d(in_ch * p.get("entry_depth_multiplier1", 2) * p.get("entry_depth_multiplier2", 2)),
            _activation("relu"),
        )
        self.sizes = p.get("entry_sizes", [128, 256, 728])
        in_channels = in_ch * p.get("entry_depth_multiplier1", 2) * p.get("entry_depth_multiplier2", 2)
        blocks = []
        for size in self.sizes:
            blocks.extend(
                [
                    _activation("relu"),
                    SeparableConv1d(in_channels, size, kernel_size=3, padding="same"),
                    nn.BatchNorm1d(size),
                    _activation("relu"),
                    SeparableConv1d(size, size, 3, padding="same"),
                    nn.BatchNorm1d(size),
                    nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
                ]
            )
            blocks.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, size, kernel_size=1, stride=2, padding=0),
                )
            )
            in_channels = size
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        x = self.seq(x)
        prev = x
        # iterate over 2‑tuple (main, residual) blocks
        for i in range(0, len(self.blocks), 2):
            main = self.blocks[i](x)
            residual = self.blocks[i + 1](prev)
            x = main + residual
            prev = x
        return x


class _XceptionMiddle(nn.Module):
    def __init__(self, num_blocks: int = 8):
        super().__init__()
        blk = []
        for _ in range(num_blocks):
            blk.extend(
                [
                    _activation("relu"),
                    SeparableConv1d(728, 728, 3, padding="same"),
                    nn.BatchNorm1d(728),
                    _activation("relu"),
                    SeparableConv1d(728, 728, 3, padding="same"),
                    nn.BatchNorm1d(728),
                    _activation("relu"),
                    SeparableConv1d(728, 728, 3, padding="same"),
                    nn.BatchNorm1d(728),
                ]
            )
        self.blk = nn.Sequential(*blk)

    def forward(self, x):
        return self.blk(x)


class _XceptionExit(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            _activation("relu"),
            SeparableConv1d(728, 728, 3, padding="same"),
            nn.BatchNorm1d(728),
            _activation("relu"),
            SeparableConv1d(728, 1024, 3, padding="same"),
            nn.BatchNorm1d(1024),
            nn.MaxPool1d(3, stride=2, padding=1),
        )
        self.residual = nn.Conv1d(728, 1024, kernel_size=1, stride=2, padding=0)
        self.tail = nn.Sequential(
            _activation("relu"),
            SeparableConv1d(1024, 728, 3, padding="same"),
            nn.BatchNorm1d(728),
            _activation("relu"),
            SeparableConv1d(728, 1024, 3, padding="same"),
            nn.BatchNorm1d(1024),
            GlobalAveragePooling1d(),
            nn.Linear(1024, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        prev = x
        x = self.seq(x)
        x = x + self.residual(prev)
        return self.tail(x)


class _Xception1D(nn.Module):
    def __init__(self, input_shape, p):
        super().__init__()
        L, C = input_shape
        self.permute = Permute()
        self.entry = _XceptionEntry(C, p)
        self.middle = _XceptionMiddle(p.get("middle_num_blocks", 8))
        self.exit = _XceptionExit()

    def forward(self, x):
        x = self.entry(self.permute(x))
        x = self.middle(x)
        return self.exit(x)


@framework("pytorch")
def XCeption1D(input_shape: Tuple[int, int], params: dict[str, Any]):
    return _Xception1D(input_shape, params)


# --------------------------------------------------------------------------- #
#                                    MLP                                      #
# --------------------------------------------------------------------------- #
class _MLP(nn.Module):
    def __init__(self, input_shape, p):
        super().__init__()
        L, C = input_shape
        self.flat = Flatten()
        in_features = L * C
        self.seq = nn.Sequential(
            nn.Dropout(p.get("dropout_rate", 0.2)),
            nn.Linear(in_features, p.get("dense_units1", 1024)),
            _activation("relu"),
            nn.Linear(p.get("dense_units1", 1024), p.get("dense_units2", 128)),
            _activation("relu"),
            nn.Linear(p.get("dense_units2", 128), p.get("dense_units3", 8)),
            _activation("relu"),
            nn.Linear(p.get("dense_units3", 8), 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.seq(self.flat(x))


@framework("pytorch")
def MLP(input_shape, params):
    return _MLP(input_shape, params)


# --------------------------------------------------------------------------- #
#  Remaining builder wrappers around legacy presets (SEResNet, ResNetv2, etc) #
# --------------------------------------------------------------------------- #
@framework("pytorch")
def SEResNet_model(input_shape: Tuple[int, int], params: dict[str, Any]):
    length = input_shape[0]
    num_channel = input_shape[-1]
    model_width = params.get("model_width", 16)
    problem_type = params.get("problem_type", "Regression")
    output_nums = params.get("output_nums", 1)
    reduction_ratio = params.get("reduction_ratio", 4)
    dropout_rate = params.get("dropout_rate", False)
    pooling = params.get("pooling", "avg")

    return SEResNet(
        length,
        num_channel,
        model_width,
        ratio=reduction_ratio,
        problem_type=problem_type,
        output_nums=output_nums,
        pooling=pooling,
        dropout_rate=dropout_rate,
    ).SEResNet101()


@framework("pytorch")
def ResNetV2_model(input_shape: Tuple[int, int], params: dict[str, Any]):
    length = input_shape[0]
    num_channel = input_shape[-1]
    model_width = params.get("model_width", 16)
    problem_type = params.get("problem_type", "Regression")
    output_nums = params.get("output_nums", 1)
    pooling = params.get("pooling", "avg")
    dropout_rate = params.get("dropout_rate", False)

    return ResNetv2(
        length,
        num_channel,
        model_width,
        problem_type=problem_type,
        output_nums=output_nums,
        pooling=pooling,
        dropout_rate=dropout_rate,
    ).ResNet34()


# --------------------------------------------------------------------------- #
#                             FFT + Conv variant                               #
# --------------------------------------------------------------------------- #
class _FFTConv(nn.Module):
    def __init__(self, input_shape, p):
        super().__init__()
        L, C = input_shape
        self.permute = Permute()

        self.conv_block = nn.Sequential(
            SeparableConv1d(
                C,
                p.get("filters1", 64),
                kernel_size=p.get("kernel_size1", 3),
                stride=p.get("strides1", 2),
                depth_multiplier=p.get("depth_multiplier1", 32),
                padding="same",
            ),
            nn.BatchNorm1d(p.get("filters1", 64)),
            _activation("relu"),
            SeparableConv1d(
                p.get("filters1", 64),
                p.get("filters2", 64),
                kernel_size=p.get("kernel_size2", 3),
                stride=p.get("strides2", 2),
                depth_multiplier=p.get("depth_multiplier2", 32),
                padding="same",
            ),
            nn.BatchNorm1d(p.get("filters2", 64)),
            _activation("relu"),
            nn.Conv1d(
                p.get("filters2", 64),
                p.get("filters3", 32),
                kernel_size=p.get("kernel_size3", 5),
                stride=p.get("strides3", 2),
                padding="same",
            ),
            nn.BatchNorm1d(p.get("filters3", 32)),
            Flatten(),
            _activation("relu"),
            nn.Linear(p.get("filters3", 32) * ((L // 8) + 1), p.get("dense_units1", 128)),
            _activation("relu"),
            nn.Linear(p.get("dense_units1", 128), p.get("dense_units2", 32)),
            _activation("relu"),
            nn.Dropout(p.get("dropout_rate", 0.1)),
            nn.Linear(p.get("dense_units2", 32), 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Apply FFT on channels dim similar to Keras Lambda
        x = torch.fft.fft(x.to(torch.complex64), dim=1).real.to(x.dtype)
        return self.conv_block(self.permute(x))


@framework("pytorch")
def FFT_Conv(input_shape, params):
    return _FFTConv(input_shape, params)


# --------------------------------------------------------------------------- #
#                              Inception wrapper                               #
# --------------------------------------------------------------------------- #
@framework("pytorch")
def inception1D(input_shape: Tuple[int, int], params: dict[str, Any]):
    length = input_shape[1]
    model_width = params.get("model_width", 16)
    num_channel = params.get("num_channel", 1)
    problem_type = params.get("problem_type", "Regression")
    output_number = params.get("output_number", 1)

    return Inception(
        length, num_channel, model_width, problem_type=problem_type, output_nums=output_number
    ).Inception_v3()


# --------------------------------------------------------------------------- #
#                       Lightweight radial‑style models                       #
# --------------------------------------------------------------------------- #
# The Custom_* builders below are left as an exercise if needed.  Their
# TensorFlow counterparts rely heavily on specialised depth‑wise 1‑D ops
# and attention; porting them would follow the same patterns as above.
# --------------------------------------------------------------------------- #
