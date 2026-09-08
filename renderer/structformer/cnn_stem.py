"""StructFormer CNN stem — KB §4 *StructFormer CNN Stem*.

Converts the structural buffers into a lower-resolution feature map before any
attention runs. Convolution is cheap for exactly what the structural input is
made of (depth discontinuities, normal boundaries, albedo edges), and reducing
resolution by 4x here is what keeps the window transformer bounded.

Input channels are M4's seven (KB Decision 35): ``depth_inv(1) + normal(3) +
albedo(3)``. §4's Explicit Boundaries hold: motion is not an appearance channel
(Decision 8 — it gets an explicit warp at M6) and raw entity IDs are not
concatenated (Decision 9 — IDs are addresses).

``nn.GroupNorm`` is reused from the M4 baseline: batch-size independent and, per
Decision 35's operator finding, exported as
``Shape -> Reshape -> InstanceNormalization -> Reshape -> Mul -> Add``, all of
which TensorRT already accepts on the proven M4 path.
"""

from __future__ import annotations

import torch
import torch.nn as nn

IN_CHANNELS = 7  # depth_inv(1) + normal(3) + albedo(3) — same contract as M4
STEM_STRIDE = 4  # two stride-2 convolutions


def _conv_block(c_in: int, c_out: int, stride: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1),
        nn.GroupNorm(min(8, c_out), c_out),
        nn.ReLU(inplace=True),
    )


class CNNStem(nn.Module):
    """``[B, in_channels, H, W]`` -> ``[B, dim, H // 4, W // 4]``."""

    def __init__(self, in_channels: int = IN_CHANNELS, dim: int = 32) -> None:
        super().__init__()
        self.body = nn.Sequential(
            _conv_block(in_channels, dim, 2),
            _conv_block(dim, dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)
