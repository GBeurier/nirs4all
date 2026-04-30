"""NiconV1b — Phase 1b model.

Composition: ``ConcatDerivatives → upstream NICON backbone with linear head (V1a-head-only)``.
The backbone (kernels 15/21/5, strides 5/3/3, channels 8/64/32) is unchanged
from upstream NICON; we only:

* widen the input channels from 1 to 3 (raw + 1st-SG + 2nd-SG),
* preserve the upstream activations / norms / dropouts (SELU, AlphaDropout, ReLU,
  BatchNorm, ELU) — i.e. the Phase 1a accepted variant ``NiconV1aHeadOnly``,
* keep the linear regression head (W1 fix retained).

Augmentation is **not** built into the model; it is applied in the training
loop via :mod:`nicon_v2.augmentation`. The model itself is therefore eval-mode
deterministic.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..preprocessing import ConcatDerivatives


def _replace_first_conv1d_in_channels(seq: nn.Sequential, new_in: int) -> nn.Sequential:
    """Find the first nn.Conv1d in `seq` and rebuild it with ``in_channels=new_in``."""
    new_layers = []
    replaced = False
    for m in seq:
        if not replaced and isinstance(m, nn.Conv1d):
            new_layers.append(
                nn.Conv1d(
                    in_channels=new_in,
                    out_channels=m.out_channels,
                    kernel_size=m.kernel_size,
                    stride=m.stride,
                    padding=m.padding,
                    dilation=m.dilation,
                    groups=m.groups,
                    bias=m.bias is not None,
                )
            )
            replaced = True
        else:
            new_layers.append(m)
    if not replaced:
        raise RuntimeError("no nn.Conv1d found to widen")
    return nn.Sequential(*new_layers)


class NiconV1b(nn.Module):
    """V1a-head-only backbone with a 3-channel concat-derivatives front."""

    def __init__(
        self,
        input_shape: tuple[int, int],
        window_length_d1: int = 11,
        window_length_d2: int = 11,
        polyorder: int = 2,
    ) -> None:
        super().__init__()
        in_ch, seq_len = input_shape
        if in_ch != 1:
            raise ValueError(f"NiconV1b expects a 1-channel raw input; got {in_ch}")
        self.preproc = ConcatDerivatives(
            window_length_d1=window_length_d1,
            window_length_d2=window_length_d2,
            polyorder=polyorder,
        )
        # The backbone is the upstream NICON with linear head (Phase 1a accepted).
        from .v1a_minimal_repair import NiconV1aHeadOnly

        head_only = NiconV1aHeadOnly((1, seq_len))
        # `head_only.body` is an `nn.Sequential` whose first Conv1d expects `in=1`.
        self.body = _replace_first_conv1d_in_channels(head_only.body, new_in=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.preproc(x)
        return self.body(x)


def build_nicon_v1b(input_shape: tuple[int, int], params: dict | None = None) -> NiconV1b:
    p = dict(params or {})
    return NiconV1b(
        input_shape=input_shape,
        window_length_d1=int(p.get("window_length_d1", 11)),
        window_length_d2=int(p.get("window_length_d2", 11)),
        polyorder=int(p.get("polyorder", 2)),
    )
