"""Two-style Model B — TemporalUNet with a per-style decoder modulation (KB Decision 94).

Model B (U-Net + engine-warped temporal fusion) is the steadiest and sharpest
renderer on the playable game (Decision 91), but it knows one style. This adds
live style switching with the smallest change that can carry it:

* the ENCODER stays style-free. It reads only the engine's structure, so the
  bottleneck features used as next frame's history are identical whichever style
  is on screen — switching style mid-game cannot disturb the temporal memory;
* every DECODER level gets FiLM (per-channel scale and shift) from a learned
  per-style vector, so style reaches colour and texture at every scale, not only
  through the bottleneck;
* the FiLM projections start at ZERO, so an untrained two-style B is exactly
  Model B, and the style difference is learned rather than initialised.

Styles are a fixed, trained set chosen by index (Decision 70: no zero-shot styles).
"""
from __future__ import annotations

import torch
from torch import nn

from renderer.temporal.temporal_unet import TemporalUNet

STYLE_DIM = 16


class StyleTemporalUNet(TemporalUNet):
    def __init__(self, style_ids: list[str], **kwargs) -> None:
        super().__init__(**kwargs)
        self.style_ids = list(style_ids)
        self.style_table = nn.Parameter(torch.empty(len(style_ids), STYLE_DIM))
        nn.init.trunc_normal_(self.style_table, std=0.02)   # not zeros: see StructFormerSingleStyle
        self.film = nn.ModuleList()
        for block in self.net.dec:
            ch = next(m.out_channels for m in reversed(list(block.modules())) if isinstance(m, nn.Conv2d))
            lin = nn.Linear(STYLE_DIM, 2 * ch)
            nn.init.zeros_(lin.weight); nn.init.zeros_(lin.bias)
            self.film.append(lin)

    def decode(self, features, skips, style_idx: torch.Tensor):
        s = self.style_table[style_idx]
        x = features
        for block, film, skip in zip(self.net.dec, self.film, reversed(skips)):
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = block(torch.cat((x, skip), dim=1))
            g, b = film(s).chunk(2, dim=-1)
            x = x * (1.0 + g[:, :, None, None]) + b[:, :, None, None]
        return torch.sigmoid(self.net.head(x))

    def forward(self, structure, prev_features=None, motion=None, confidence=None, style_idx=None):
        if style_idx is None:
            raise ValueError("StyleTemporalUNet needs style_idx (which trained style to render)")
        features, skips = self.encode(structure)
        if prev_features is not None:
            if motion is None or confidence is None:
                raise ValueError("prev_features requires motion and confidence")
            features = self.fuse(features, prev_features, motion, confidence)
        return self.decode(features, skips, style_idx)
