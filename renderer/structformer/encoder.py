"""Window transformer encoder — KB §4 *Window Transformer Encoder*.

A stack of stages; each stage is ``depth`` :class:`WindowTransformerBlock`s at a
fixed resolution, and stages are separated by a stride-2 convolution. The
convolutional downsample is a deliberate choice over Swin's patch-merging: patch
merging is a strided-slice + concat pattern, and KB §12's whole mitigation is to
"use ONNX-safe primitives" — a stride-2 ``Conv`` is already proven on the M4
TensorRT path and buys the same hierarchical receptive field.

Blocks alternate unshifted / shifted (``shift = window_size // 2`` on odd
blocks). With ``depth=1`` there is no shifted block at all, so window boundaries
become hard walls; that configuration exists for the M5 export probe's minimal
skeleton, not for a trained model.

§4 boundaries: no unbounded global attention at full resolution, no ViT-scale
model, no temporal history prediction.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from renderer.structformer.window_attention import WindowTransformerBlock


class EncoderStage(nn.Module):
    """``depth`` window blocks at one resolution, then an optional stride-2 conv."""

    def __init__(
        self,
        dim: int,
        out_dim: int | None,
        resolution: tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        attn_impl: str,
        shift_impl: str,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            WindowTransformerBlock(
                dim=dim,
                input_resolution=resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if i % 2 == 0 else window_size // 2,
                attn_impl=attn_impl,
                shift_impl=shift_impl,
            )
            for i in range(depth)
        )
        self.downsample = (
            nn.Conv2d(dim, out_dim, 3, stride=2, padding=1) if out_dim is not None else None
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(downsampled, skip)``; ``skip`` is the pre-downsample map."""
        for block in self.blocks:
            x = block(x)
        skip = x
        if self.downsample is not None:
            x = self.downsample(x)
        return x, skip
