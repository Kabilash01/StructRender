"""Correctness of the window-attention primitives (KB §4 Window Transformer Encoder).

The load-bearing test here is
:func:`test_window_attention_equals_full_attention_when_one_window_covers_the_map`:
with the window enlarged to the whole feature map, bounded window attention must
reduce **exactly** to ordinary full self-attention computed from the same
weights. That catches a wrong permutation, a mis-ordered head split or a
transposed q/k — none of which a shape assertion would see.
"""

from __future__ import annotations

import math

import pytest
import torch

from renderer.structformer.window_attention import (
    MASK_NEG,
    RelativePositionBias,
    WindowAttention,
    WindowTransformerBlock,
    build_shift_attention_mask,
    cyclic_shift,
    window_partition,
    window_reverse,
)


@pytest.mark.parametrize("shape,ws", [((2, 8, 16, 5), 4), ((1, 72, 128, 3), 8), ((1, 4, 4, 7), 4)])
def test_partition_reverse_round_trips(shape, ws):
    x = torch.randn(*shape)
    assert torch.equal(window_reverse(window_partition(x, ws), ws, shape[1], shape[2]), x)


def test_window_partition_extracts_the_expected_pixels():
    """Window (r, c) of batch b must land at index ``b * nW + r * (W // ws) + c``.

    The shifted-window mask and :func:`window_reverse` both assume that ordering,
    so it is asserted rather than inherited from the reshape.
    """
    b, h, w, c, ws = 2, 8, 16, 3, 4
    x = torch.randn(b, h, w, c)
    win = window_partition(x, ws)
    n_w = w // ws
    for bi in range(b):
        for r in range(h // ws):
            for col in range(n_w):
                got = win[bi * (h // ws) * n_w + r * n_w + col]
                want = x[bi, r * ws : (r + 1) * ws, col * ws : (col + 1) * ws, :].reshape(ws * ws, c)
                assert torch.equal(got, want)


def _reference_full_attention(attn: WindowAttention, x: torch.Tensor) -> torch.Tensor:
    """Textbook multi-head self-attention over ``x`` of shape ``[B, N, C]``.

    Written out from the module's own weights so the comparison is against the
    definition, not against another call into the code under test.
    """
    b, n, c = x.shape
    heads, hd = attn.num_heads, attn.head_dim
    qkv = torch.nn.functional.linear(x, attn.qkv.weight, attn.qkv.bias)
    q, k, v = (t.view(b, n, heads, hd).transpose(1, 2) for t in qkv.chunk(3, dim=-1))
    logits = q @ k.transpose(-2, -1) / math.sqrt(hd)
    out = (logits.softmax(-1) @ v).transpose(1, 2).reshape(b, n, c)
    return torch.nn.functional.linear(out, attn.proj.weight, attn.proj.bias)


@pytest.mark.parametrize("impl", ["manual", "sdpa"])
def test_window_attention_equals_full_attention_when_one_window_covers_the_map(impl):
    torch.manual_seed(0)
    h = w = 4  # one 4x4 window == the whole feature map
    dim, heads = 8, 2
    attn = WindowAttention(dim, h, heads, use_rel_pos_bias=False, attn_impl=impl).eval()
    x = torch.randn(1, h, w, dim)

    with torch.no_grad():
        got = attn(window_partition(x, h))
        want = _reference_full_attention(attn, x.reshape(1, h * w, dim))
    assert torch.allclose(got, want, atol=1e-6), (got - want).abs().max()


def test_manual_and_sdpa_are_numerically_equivalent():
    """The two attention spellings must be the same function, so the export
    probe's choice between them is purely an operator/latency decision."""
    torch.manual_seed(1)
    manual = WindowAttention(16, 4, 2, True, "manual").eval()
    sdpa = WindowAttention(16, 4, 2, True, "sdpa").eval()
    sdpa.load_state_dict(manual.state_dict())
    x = torch.randn(6, 16, 16)
    mask = build_shift_attention_mask(8, 12, 4, 2)
    with torch.no_grad():
        assert torch.allclose(manual(x, mask), sdpa(x, mask), atol=1e-6)


@pytest.mark.parametrize("shift", [-2, -1, 1, 3])
def test_roll_and_slice_shifts_agree(shift):
    """The slice+concat fallback must be bit-identical to ``torch.roll``, or it is
    not a fallback."""
    x = torch.randn(2, 8, 12, 3)
    assert torch.equal(cyclic_shift(x, shift, "roll"), cyclic_shift(x, shift, "slice"))


def test_relative_position_bias_reads_the_table_at_the_precomputed_index():
    ws, heads = 4, 3
    rpb = RelativePositionBias(ws, heads).eval()
    with torch.no_grad():
        bias = rpb()
    assert bias.shape == (heads, ws * ws, ws * ws)
    for i in (0, 5, 15):
        for j in (0, 7, 15):
            assert torch.equal(bias[:, i, j], rpb.table[rpb.index[i, j]])
    # Same relative offset => same bias. Pixels 0 and 5 are both (+1,+1) from
    # pixels 5 and 10 in a 4x4 window.
    assert rpb.index[0, 5] == rpb.index[5, 10]


def test_relative_position_index_is_not_a_checkpoint_tensor():
    """It is fully derived from ``window_size``; persisting it would let a stale
    checkpoint carry a wrong index. It must still be a buffer (a graph constant),
    not recomputed in ``forward``."""
    rpb = RelativePositionBias(4, 1)
    assert "index" not in rpb.state_dict()
    assert "index" in dict(rpb.named_buffers())


def test_shift_mask_forbids_wrapped_pairs_only():
    mask = build_shift_attention_mask(8, 12, 4, 2)
    n_w = (8 // 4) * (12 // 4)
    assert mask.shape == (n_w, 16, 16)
    assert set(mask.unique().tolist()) <= {0.0, MASK_NEG}
    assert torch.equal(mask.diagonal(dim1=-2, dim2=-1), torch.zeros(n_w, 16))  # self is always legal
    assert (mask == MASK_NEG).any(), "a shifted map must have at least one forbidden pair"
    assert torch.equal(mask[0], torch.zeros(16, 16)), "the unwrapped top-left window is unmasked"


@pytest.mark.parametrize("shift", [0, 4])
@pytest.mark.parametrize("impl", ["manual", "sdpa"])
def test_block_preserves_shape(shift, impl):
    block = WindowTransformerBlock(16, (24, 32), 2, 8, shift, attn_impl=impl).eval()
    x = torch.randn(1, 16, 24, 32)
    with torch.no_grad():
        assert block(x).shape == x.shape


def test_block_rejects_geometry_it_cannot_tile():
    """Fixed shapes are a KB §12 mitigation, so a mismatch is an error at
    construction rather than a silent runtime pad."""
    with pytest.raises(ValueError):
        WindowTransformerBlock(16, (24, 30), 2, 8)
    with pytest.raises(ValueError):
        WindowTransformerBlock(16, (24, 32), 2, 8, shift_size=8)
    with pytest.raises(ValueError):
        WindowAttention(16, 8, 3)  # dim not divisible by heads
