"""KB §4 *Entity-Masked Cross-Attention Module* — the masking rule, asserted.

M7's mechanism, tested at M5 because it is exported at M5 (see
``eval/reports/m5_export_probe.json``). The rule under test is §4's::

    "An NPC's spatial features must not retrieve a car's appearance memory."

which is stronger than "the mask tensor is applied somewhere". The test therefore
perturbs one entity's token and asserts that **nothing outside that entity's own
mask moves** — an off-by-one transpose of the mask, or a mask applied after the
softmax, would pass a shape check and fail this.
"""

from __future__ import annotations

import pytest
import torch

from renderer.structformer.masked_entity_attention import EntityMaskedCrossAttention

DIM, HEADS, K = 16, 2, 4
H, W = 8, 12


def _masks() -> torch.Tensor:
    """Disjoint rectangles; slot 3 is left empty to stand for a padded slot."""
    m = torch.zeros(1, K, H, W)
    m[0, 0, 0:3, 0:4] = 1.0
    m[0, 1, 3:6, 4:8] = 1.0
    m[0, 2, 6:8, 8:12] = 1.0
    return m


@pytest.mark.parametrize("impl", ["manual", "sdpa"])
def test_output_shape(impl):
    layer = EntityMaskedCrossAttention(DIM, DIM, HEADS, K, impl).eval()
    x = torch.randn(1, DIM, H, W)
    with torch.no_grad():
        assert layer(x, torch.randn(1, K, DIM), _masks()).shape == x.shape


@pytest.mark.parametrize("impl", ["manual", "sdpa"])
def test_perturbing_one_entity_token_changes_only_that_entity_mask(impl):
    torch.manual_seed(0)
    layer = EntityMaskedCrossAttention(DIM, DIM, HEADS, K, impl).eval()
    x = torch.randn(1, DIM, H, W)
    mask = _masks()
    tokens = torch.randn(1, K, DIM)

    with torch.no_grad():
        base = layer(x, tokens, mask)
        for i in range(K - 1):
            bumped = tokens.clone()
            bumped[0, i] += 5.0
            delta = (layer(x, bumped, mask) - base).abs().amax(dim=1)[0]  # [H, W]
            outside = delta * (1.0 - mask[0, i])
            assert outside.max() == 0.0, f"entity {i}'s token leaked outside its own mask"
            assert delta[mask[0, i] > 0].max() > 0.0, f"entity {i}'s token had no effect at all"


@pytest.mark.parametrize("impl", ["manual", "sdpa"])
def test_pixels_in_no_entity_mask_receive_exactly_zero(impl):
    """Softmax always sums to 1, so without the output gate a background pixel
    would still receive a blend of every entity's memory."""
    torch.manual_seed(0)
    layer = EntityMaskedCrossAttention(DIM, DIM, HEADS, K, impl).eval()
    mask = _masks()
    with torch.no_grad():
        out = layer(torch.randn(1, DIM, H, W), torch.randn(1, K, DIM), mask)
    background = mask.amax(dim=1)[0] == 0
    assert out[0, :, background].abs().max() == 0.0


@pytest.mark.parametrize("impl", ["manual", "sdpa"])
def test_an_all_zero_mask_produces_finite_zeros_not_nan(impl):
    """A frame with no visible entity is the common case, not an edge case: with a
    literal ``-inf`` mask every softmax row would be NaN and the whole frame
    would be destroyed."""
    layer = EntityMaskedCrossAttention(DIM, DIM, HEADS, K, impl).eval()
    with torch.no_grad():
        out = layer(torch.randn(1, DIM, H, W), torch.randn(1, K, DIM), torch.zeros(1, K, H, W))
    assert torch.isfinite(out).all()
    assert out.abs().max() == 0.0


def test_an_empty_padded_slot_cannot_contribute():
    """§5's contract pads unused entity slots. Padding is only safe because a
    padded slot's mask is all zero — assert that, so a future packing change that
    forgets to clear the mask fails here."""
    torch.manual_seed(0)
    layer = EntityMaskedCrossAttention(DIM, DIM, HEADS, K, "manual").eval()
    x, mask = torch.randn(1, DIM, H, W), _masks()
    tokens = torch.randn(1, K, DIM)
    junk = tokens.clone()
    junk[0, K - 1] = 1e3  # slot 3 is the padded one
    with torch.no_grad():
        assert torch.equal(layer(x, tokens, mask), layer(x, junk, mask))


def test_slot_count_is_fixed():
    layer = EntityMaskedCrossAttention(DIM, DIM, HEADS, K, "manual").eval()
    with pytest.raises(ValueError):
        layer(torch.randn(1, DIM, H, W), torch.randn(1, K + 1, DIM), _masks())
