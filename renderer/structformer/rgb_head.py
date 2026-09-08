"""CNN RGB head — KB §4 *CNN RGB Head*.

Decoder features -> RGB at full resolution. Upsampling is nearest ``interpolate``
+ convolution, the same pair M4 exported successfully (ONNX ``Resize`` with
nearest mode); ``ConvTranspose2d`` is avoided because it buys nothing here and
adds checkerboard artefacts.

Output is sigmoid in ``[0, 1]``, matching the 8-bit sRGB teacher targets scaled
to ``[0, 1]`` — identical to the M4 baseline's output contract so the two are
comparable on the KB §11 ladder. Per §4's boundaries: no game-state output and
no segmentation output in the runtime path.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RGBHead(nn.Module):
    """``[B, dim, H // 4, W // 4]`` -> ``[B, 3, H, W]`` in ``[0, 1]``."""

    def __init__(self, dim: int, hidden: int, out_channels: int = 3) -> None:
        super().__init__()
        self.up1 = nn.Sequential(
            nn.Conv2d(dim, hidden, 3, padding=1),
            nn.GroupNorm(min(8, hidden), hidden),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Conv2d(hidden, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up1(F.interpolate(x, scale_factor=2.0, mode="nearest"))
        x = self.out(F.interpolate(x, scale_factor=2.0, mode="nearest"))
        return torch.sigmoid(x)
