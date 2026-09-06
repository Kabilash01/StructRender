"""Model A of the KB §11 ablation ladder — the plain CNN/U-Net baseline (M4).

A control, not the contribution: KB Decision 3 keeps U-Net as *a baseline* while
StructFormer is the renderer under development. Deliberately absent, because
their presence in Model A would collapse the ladder's controls:

- no attention (M5), no temporal conditioning (M6), no entity awareness (M7),
  no EPAM (M8), no style conditioning (single style at M4), no adversarial loss
  (Decision 4);
- ``motion`` is not an input channel — KB §4 CNN Stem boundary, "motion is not
  treated as an unexplained appearance channel" (Decision 8);
- ``entity_id`` is not an input channel — KB Decision 9, IDs are addresses.

Only ONNX-safe primitives are used (Conv2d, GroupNorm, ReLU, MaxPool2d, nearest
``interpolate``, Concat, Sigmoid) — KB §12 operator-compatibility mitigation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

IN_CHANNELS = 7  # depth_inv(1) + normal(3) + albedo(3)
OUT_CHANNELS = 3


def _double_conv(c_in: int, c_out: int) -> nn.Sequential:
    # GroupNorm, not BatchNorm: batch-size independent (the overfit run uses
    # batch 2) and it exports to ONNX without train/eval statistics.
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1),
        nn.GroupNorm(8, c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1),
        nn.GroupNorm(8, c_out),
        nn.ReLU(inplace=True),
    )


class UNetBaseline(nn.Module):
    """Structural buffers -> styled RGB, one frame at a time.

    Input  ``[B, 7, H, W]`` with H, W divisible by 16.
    Output ``[B, 3, H, W]`` in ``[0, 1]`` (sigmoid), matching the 8-bit sRGB
    teacher targets scaled to ``[0, 1]``.
    """

    def __init__(
        self,
        in_channels: int = IN_CHANNELS,
        out_channels: int = OUT_CHANNELS,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        c = base_channels
        chans = [c, c * 2, c * 4, c * 8]

        self.enc = nn.ModuleList()
        prev = in_channels
        for ch in chans:
            self.enc.append(_double_conv(prev, ch))
            prev = ch
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = _double_conv(chans[-1], chans[-1] * 2)

        self.dec = nn.ModuleList()
        prev = chans[-1] * 2
        for ch in reversed(chans):
            self.dec.append(_double_conv(prev + ch, ch))
            prev = ch
        self.head = nn.Conv2d(chans[0], out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for block in self.enc:
            x = block(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for block, skip in zip(self.dec, reversed(skips)):
            x = F.interpolate(x, scale_factor=2.0, mode="nearest")
            x = block(torch.cat([x, skip], dim=1))
        return torch.sigmoid(self.head(x))


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
