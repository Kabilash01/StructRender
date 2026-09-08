"""StructFormer assembly — shape flow and the runtime tensor contract.

KB §5's Runtime Tensor Contract and Decision 35's export shapes are the same
``(1, 7, 288, 512) -> (1, 3, 288, 512)`` M4 established, so Model A and Model B
stay directly comparable on the §11 ladder.
"""

from __future__ import annotations

import pytest
import torch

from renderer.structformer.model import StructFormer, parameter_count


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    return StructFormer(depths=(1, 1, 1)).eval()


def test_runtime_tensor_contract(tiny):
    with torch.no_grad():
        rgb = tiny(torch.randn(1, 7, 288, 512), torch.randn(1, 16))
    assert rgb.shape == (1, 3, 288, 512)
    assert rgb.min() >= 0.0 and rgb.max() <= 1.0  # sigmoid, same output range as M4


def test_the_style_token_actually_reaches_the_output(tiny):
    """§4's style token is cached, not decorative: two tokens must give two
    images, or style conditioning is silently a no-op."""
    x = torch.randn(1, 7, 288, 512)
    with torch.no_grad():
        a = tiny(x, torch.zeros(1, 16))
        b = tiny(x, torch.ones(1, 16) * 3.0)
    assert (a - b).abs().max() > 1e-4


@pytest.mark.parametrize("impl", ["manual", "sdpa"])
def test_attention_implementations_give_the_same_model(impl):
    torch.manual_seed(0)
    ref = StructFormer(depths=(1, 1, 1), attn_impl="manual").eval()
    other = StructFormer(depths=(1, 1, 1), attn_impl=impl).eval()
    other.load_state_dict(ref.state_dict())
    x, s = torch.randn(1, 7, 288, 512), torch.randn(1, 16)
    with torch.no_grad():
        assert torch.allclose(ref(x, s), other(x, s), atol=1e-5)


def test_depth_adds_shifted_blocks():
    """Depth 1 gives no shifted block at all, so window boundaries are hard walls.
    A trained model must not be configured that way — this pins the alternation
    rule so the M5 config choice is visible rather than accidental."""
    shallow = StructFormer(depths=(1, 1, 1))
    deep = StructFormer(depths=(2, 2, 2))
    shifts = lambda m: [b.shift_size for b in m.stage1.blocks]  # noqa: E731
    assert shifts(shallow) == [0]
    assert shifts(deep) == [0, 4]
    assert parameter_count(deep) > parameter_count(shallow)


def test_incompatible_image_size_is_rejected_at_construction():
    with pytest.raises(ValueError):
        StructFormer(image_size=(288, 500))
    with pytest.raises(ValueError):
        StructFormer(image_size=(288, 512), window_sizes=(8, 7))
