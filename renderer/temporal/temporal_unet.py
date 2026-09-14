"""Model B — the CNN temporal rung (KB §10 Milestone 9's ladder).

**Why this exists.** M6 compared Model D (StructFormer + temporal) against Model C
(StructFormer alone) and found no separable improvement. That cannot distinguish
"temporal reprojection does not help here" from "temporal reprojection does not
help *a transformer* here", because the CNN-temporal rung was never built. Model B
is the missing control: A -> B is the same single change as C -> D, on the other
backbone.

**The temporal mechanism is deliberately identical to Model D's** — the same
mask-gated residual on the bottleneck features, the same zero-initialised 1x1
fusion convolution, the same warp and the same confidence mask. If it differed,
A->B and C->D would not be the same delta and the comparison would be worthless.

Fusion point: the U-Net's bottleneck, which is its encoder/decoder junction —
the structural analogue of where Model D fuses. KB Decision 63.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from renderer.baselines.unet import IN_CHANNELS, OUT_CHANNELS, UNetBaseline
from renderer.temporal.warp import EngineGuidedWarp, resize_motion


class TemporalUNet(nn.Module):
    """U-Net with one step of engine-guided history at the bottleneck."""

    def __init__(
        self,
        in_channels: int = IN_CHANNELS,
        out_channels: int = OUT_CHANNELS,
        base_channels: int = 32,
        image_size: tuple[int, int] = (288, 512),
    ) -> None:
        super().__init__()
        self.net = UNetBaseline(in_channels, out_channels, base_channels)
        self.feature_dim = base_channels * 8 * 2          # bottleneck width
        self.image_size = tuple(image_size)
        # The U-Net pools 4x (one per encoder block), so the bottleneck is /16.
        self.feature_size = (image_size[0] // 16, image_size[1] // 16)
        # Constructed at FEATURE resolution, exactly as Model D's is.
        self.warp = EngineGuidedWarp(*self.feature_size)
        # Zero-initialised, exactly as Model D's: an untrained Model B is
        # numerically identical to Model A, so the rung adds a mechanism rather
        # than a different starting point.
        self.temporal_fuse = nn.Conv2d(2 * self.feature_dim, self.feature_dim, 1)
        nn.init.zeros_(self.temporal_fuse.weight)
        nn.init.zeros_(self.temporal_fuse.bias)

    # -- Model A's forward, split at the bottleneck --------------------------- #
    def encode(self, structure: torch.Tensor) -> tuple[torch.Tensor, list]:
        skips = []
        x = structure
        for block in self.net.enc:
            x = block(x)
            skips.append(x)
            x = self.net.pool(x)
        return self.net.bottleneck(x), skips

    def decode(self, features: torch.Tensor, skips: list) -> torch.Tensor:
        x = features
        for block, skip in zip(self.net.dec, reversed(skips)):
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = block(torch.cat((x, skip), dim=1))
        return torch.sigmoid(self.net.head(x))

    def fuse(self, features, prev_features, motion, confidence):
        """Mask-gated residual — the same rule Model D applies, step for step.

        Area pooling for the mask, not bilinear: one bottleneck cell covers a
        16x16 pixel block, so "most of my footprint has no usable history" must
        read as low confidence (KB Decision 55 item 4).
        """
        m = nn.functional.adaptive_avg_pool2d(confidence, self.feature_size)
        v = resize_motion(motion, self.feature_size)
        warped = self.warp(prev_features, v) * m
        delta = self.temporal_fuse(torch.cat((features, warped), dim=1))
        return features + m * delta

    def forward(
        self,
        structure: torch.Tensor,
        prev_features: torch.Tensor | None = None,
        motion: torch.Tensor | None = None,
        confidence: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features, skips = self.encode(structure)
        if prev_features is not None:
            if motion is None or confidence is None:
                raise ValueError("prev_features requires motion and confidence")
            features = self.fuse(features, prev_features, motion, confidence)
        return self.decode(features, skips)
