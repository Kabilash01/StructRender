"""Export-time rewrites that make a trained model buildable by TensorRT.

`nn.GroupNorm` exports as ONNX `InstanceNormalization` (plus reshapes). On this
toolchain (TensorRT 11.2.1.2, CUDA 13.2, RTX 5060 Laptop) Myelin fuses that node
into a region its NVRTC compiler cannot compile, and the build fails with
"Error Code 10: Could not find any implementation for node {ForeignNode...}" at
every builder optimization level 0-5 (KB Decision 84). Written out as
reshape / mean / variance / affine it builds, and computes the same thing.

This is applied at EXPORT only. Training and checkpoints are untouched; the
rewritten module reuses the original's `weight` and `bias` tensors, so there is
nothing to convert and nothing to drift.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn


class ExplicitGroupNorm(nn.Module):
    """GroupNorm as elementary ops — no InstanceNormalization in the ONNX graph."""

    def __init__(self, gn: nn.GroupNorm):
        super().__init__()
        self.num_groups, self.eps = gn.num_groups, gn.eps
        self.weight, self.bias = gn.weight, gn.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        y = x.reshape(b, self.num_groups, (c // self.num_groups) * h * w)
        mu = y.mean(dim=2, keepdim=True)
        var = ((y - mu) ** 2).mean(dim=2, keepdim=True)   # biased, as nn.GroupNorm
        y = ((y - mu) / torch.sqrt(var + self.eps)).reshape(b, c, h, w)
        if self.weight is not None:
            y = y * self.weight[None, :, None, None] + self.bias[None, :, None, None]
        return y


def for_tensorrt(model: nn.Module) -> nn.Module:
    """A deep copy of `model` with every GroupNorm made TensorRT-buildable."""
    m = copy.deepcopy(model)

    def swap(mod):
        for name, child in mod.named_children():
            if isinstance(child, nn.GroupNorm):
                setattr(mod, name, ExplicitGroupNorm(child))
            else:
                swap(child)
    swap(m)
    return m.eval()
