"""U-Net baseline shape/range contract, and the M4 architecture boundaries."""

from __future__ import annotations

import inspect

import torch

from renderer.baselines import unet as unet_mod
from renderer.baselines.unet import UNetBaseline, parameter_count


def test_forward_shape_and_range():
    m = UNetBaseline(base_channels=8).eval()
    x = torch.randn(2, 7, 288, 512)
    with torch.no_grad():
        y = m(x)
    assert y.shape == (2, 3, 288, 512)
    assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0


def test_frame_independent_no_hidden_state():
    """Model A is frame-independent: the same input gives the same output, and a
    batch is just N independent frames (no temporal state, M6)."""
    torch.manual_seed(0)
    m = UNetBaseline(base_channels=8).eval()
    x = torch.randn(3, 7, 64, 64)
    with torch.no_grad():
        batched = m(x)
        singles = torch.cat([m(x[i : i + 1]) for i in range(3)])
        again = m(x)
    assert torch.allclose(batched, again, atol=0)
    assert torch.allclose(batched, singles, atol=1e-5)


def test_no_attention_no_adversarial_no_epam_in_the_baseline():
    src = inspect.getsource(unet_mod).lower()
    for banned in ("attention", "discriminator", "epam", "memory", "warp"):
        assert f"self.{banned}" not in src
    m = UNetBaseline(base_channels=8)
    kinds = {type(mod).__name__ for mod in m.modules()}
    assert "MultiheadAttention" not in kinds
    assert kinds <= {
        "UNetBaseline", "ModuleList", "Sequential",
        "Conv2d", "GroupNorm", "ReLU", "MaxPool2d",
    }


def test_parameter_count_is_reported_and_reasonable():
    n = parameter_count(UNetBaseline(base_channels=32))
    assert 1_000_000 < n < 50_000_000
