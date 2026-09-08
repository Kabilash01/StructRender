"""Window transformer decoder — KB §4 *Window Transformer Decoder*.

Mirrors :mod:`renderer.structformer.encoder`: nearest upsample + 1x1 convolution
to restore width, add the encoder skip, then ``depth`` window blocks. Reconstructs
high-resolution features after structure (and, from M6/M7, temporal and entity)
fusion.

§4 boundaries: does not predict future frames, does not own memory lifecycle,
does not perform entity identification.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from renderer.structformer.window_attention import WindowTransformerBlock


class DecoderStage(nn.Module):
    """Upsample ``[B, dim, H, W]`` -> ``[B, out_dim, 2H, 2W]``, fuse skip, attend."""

    def __init__(
        self,
        dim: int,
        out_dim: int,
        resolution: tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        attn_impl: str,
        shift_impl: str,
    ) -> None:
        super().__init__()
        self.reduce = nn.Conv2d(dim, out_dim, 1)
        self.blocks = nn.ModuleList(
            WindowTransformerBlock(
                dim=out_dim,
                input_resolution=resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if i % 2 == 0 else window_size // 2,
                attn_impl=attn_impl,
                shift_impl=shift_impl,
            )
            for i in range(depth)
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.reduce(F.interpolate(x, scale_factor=2.0, mode="nearest")) + skip
        for block in self.blocks:
            x = block(x)
        return x
