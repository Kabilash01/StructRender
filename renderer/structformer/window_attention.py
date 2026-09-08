"""Bounded window attention — KB §4 *Window Transformer Encoder*.

This module is the operator-risk surface flagged in KB §12
(*ONNX/TensorRT Operator Compatibility*: "custom attention/masking operation
exports poorly"). Everything here is therefore built from primitives that the
ONNX exporter lowers without a custom op, and every design choice that had two
plausible spellings is exposed as a constructor flag so the M5 export probe can
measure both rather than assume one:

- ``attn_impl="sdpa" | "manual"`` — ``F.scaled_dot_product_attention`` versus an
  explicit ``matmul -> (+bias) -> softmax -> matmul``. Identical mathematics,
  different exported graphs.
- ``shift_impl="roll" | "slice"`` — ``torch.roll`` versus a slice+concat
  equivalent, because ``roll`` is a known ONNX/TensorRT hazard.

KB §4 boundaries honoured here: bounded local attention only, no unbounded
global attention at full resolution, no temporal or entity state (those arrive
at M6/M7 and enter through separate modules).

Tensor conventions
------------------
Blocks take and return ``[B, C, H, W]`` so they compose directly with the CNN
stem/head; the channels-last ``[B, H, W, C]`` layout exists only inside the
block, where attention needs it.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

#: Additive-mask "forbidden" value. Finite on purpose: ``-inf`` makes a fully
#: masked softmax row produce NaN, and FP16 TensorRT engines have no headroom
#: for ``-1e9``. ``exp(-100)`` is ~1e-44 against logits of order 1, i.e. zero in
#: both FP32 and FP16, so this is a numerical spelling of ``-inf``, not a
#: relaxation of KB §4's masking rule.
MASK_NEG = -100.0

ATTN_IMPLS = ("manual", "sdpa")
SHIFT_IMPLS = ("roll", "slice")


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """``[B, H, W, C]`` -> ``[B * nW, window_size**2, C]``.

    ``H`` and ``W`` must be exact multiples of ``window_size``; the caller
    guarantees it by construction (KB §12 "prefer fixed/bounded shapes" — there
    is deliberately no runtime padding branch, which would put a ``Pad`` and a
    data-dependent ``If`` into the exported graph).

    Window order is row-major over the window grid, and batch-major overall:
    index ``b * nW + w``. :func:`window_reverse` and the shift mask both depend
    on that ordering.
    """
    b, h, w, c = x.shape
    ws = window_size
    x = x.view(b, h // ws, ws, w // ws, ws, c)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(-1, ws * ws, c)


def window_reverse(windows: torch.Tensor, window_size: int, h: int, w: int) -> torch.Tensor:
    """Inverse of :func:`window_partition`: ``[B * nW, ws**2, C]`` -> ``[B, H, W, C]``."""
    ws = window_size
    c = windows.shape[-1]
    b = windows.shape[0] // ((h // ws) * (w // ws))
    x = windows.view(b, h // ws, w // ws, ws, ws, c)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(b, h, w, c)


def cyclic_shift(x: torch.Tensor, shift: int, impl: str = "roll") -> torch.Tensor:
    """Shift ``[B, H, W, C]`` by ``shift`` along H and W (negative = Swin's shift).

    ``impl="slice"`` is the ONNX-safe fallback: ``torch.roll`` has historically
    exported badly, and the equivalent is two ``Slice``+``Concat`` pairs. Both
    are kept because the probe measures which one TensorRT accepts; they are
    asserted numerically identical in the unit tests.
    """
    if impl not in SHIFT_IMPLS:
        raise ValueError(f"unknown shift_impl {impl!r}, expected one of {SHIFT_IMPLS}")
    if shift == 0:
        return x
    if impl == "roll":
        return torch.roll(x, shifts=(shift, shift), dims=(1, 2))
    # Slice bounds must be PYTHON ints, never derived from ``x.shape``: a
    # ``shift % x.shape[1]`` reads the shape at runtime, which puts
    # Shape/Gather/Mod/Neg/Unsqueeze nodes in the graph and leaves the exporter
    # unable to infer the output shape (measured at M5 — the graph's output dims
    # came out symbolic, against KB §12's "prefer fixed/bounded shapes").
    # ``|shift| < window_size <= min(H, W)`` by construction, so the two signs
    # are all the cases there are.
    if shift > 0:
        x = torch.cat((x[:, -shift:, :, :], x[:, :-shift, :, :]), dim=1)
        return torch.cat((x[:, :, -shift:, :], x[:, :, :-shift, :]), dim=2)
    s = -shift
    x = torch.cat((x[:, s:, :, :], x[:, :s, :, :]), dim=1)
    return torch.cat((x[:, :, s:, :], x[:, :, :s, :]), dim=2)


def build_shift_attention_mask(
    h: int, w: int, window_size: int, shift_size: int
) -> torch.Tensor:
    """Swin's shifted-window attention mask, ``[nW, N, N]`` additive.

    After a cyclic shift a window can contain pixels that are not spatial
    neighbours (they wrapped around the image edge). Those pairs are forbidden
    with :data:`MASK_NEG`. Computed once at construction time and registered as
    a buffer so it lands in the exported graph as a **constant**, never as
    runtime arithmetic (KB §12 "use ONNX-safe primitives").
    """
    img = torch.zeros(1, h, w, 1)
    spans = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
    region = 0
    for hs in spans:
        for ws_ in spans:
            img[:, hs, ws_, :] = region
            region += 1
    win = window_partition(img, window_size).squeeze(-1)  # [nW, N]
    mask = win.unsqueeze(1) - win.unsqueeze(2)  # [nW, N, N]
    return torch.where(mask == 0, 0.0, MASK_NEG)


class RelativePositionBias(nn.Module):
    """Learned per-head bias on the relative offset between two window pixels.

    Hazard probed at M5: this is a ``Gather`` on a learned table driven by a
    precomputed index. The index is a **buffer**, not runtime arithmetic, so the
    exported graph carries it as an initializer/``Constant`` — a graph that
    recomputes ``meshgrid``/``arange`` at inference time is both slower and a
    larger operator-support surface.
    """

    def __init__(self, window_size: int, num_heads: int) -> None:
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        n = window_size * window_size
        self.table = nn.Parameter(torch.zeros((2 * window_size - 1) ** 2, num_heads))
        nn.init.trunc_normal_(self.table, std=0.02)

        coords = torch.stack(
            torch.meshgrid(torch.arange(window_size), torch.arange(window_size), indexing="ij")
        ).flatten(1)  # [2, N]
        rel = (coords[:, :, None] - coords[:, None, :]).permute(1, 2, 0).contiguous()  # [N, N, 2]
        rel[:, :, 0] += window_size - 1
        rel[:, :, 1] += window_size - 1
        rel[:, :, 0] *= 2 * window_size - 1
        # persistent=False: fully derived from window_size, so it does not belong
        # in a checkpoint, but it is still a graph constant at export time.
        self.register_buffer("index", rel.sum(-1).view(n, n), persistent=False)

    def forward(self) -> torch.Tensor:
        """-> ``[num_heads, N, N]``."""
        n = self.window_size * self.window_size
        bias = self.table[self.index.view(-1)].view(n, n, self.num_heads)
        return bias.permute(2, 0, 1).contiguous()


class WindowAttention(nn.Module):
    """Multi-head self-attention inside one window.

    Args:
        dim: channel width.
        window_size: side of the square window; ``N = window_size**2`` tokens.
        num_heads: attention heads.
        use_rel_pos_bias: attach a :class:`RelativePositionBias`.
        attn_impl: ``"manual"`` or ``"sdpa"`` (see module docstring).

    Forward takes ``[B_, N, C]`` where ``B_ = B * nW``, and an optional additive
    ``mask`` of shape ``[nW, N, N]``.
    """

    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        use_rel_pos_bias: bool = True,
        attn_impl: str = "manual",
    ) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim {dim} not divisible by num_heads {num_heads}")
        if attn_impl not in ATTN_IMPLS:
            raise ValueError(f"unknown attn_impl {attn_impl!r}, expected one of {ATTN_IMPLS}")
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.attn_impl = attn_impl
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.rel_pos_bias = (
            RelativePositionBias(window_size, num_heads) if use_rel_pos_bias else None
        )

    def _attn_bias(self, b_: int, mask: torch.Tensor | None) -> torch.Tensor | None:
        """Combine relative-position bias and window mask into one additive term."""
        bias = None
        if self.rel_pos_bias is not None:
            bias = self.rel_pos_bias().unsqueeze(0)  # [1, heads, N, N]
        if mask is not None:
            # Window order is batch-major (see window_partition), so tiling the
            # per-window mask B times along dim 0 lines up with B_ = B * nW.
            m = mask.unsqueeze(1).repeat(b_ // mask.shape[0], 1, 1, 1)  # [B_, 1, N, N]
            bias = m if bias is None else bias + m
        return bias

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        b_, n, c = x.shape
        qkv = self.qkv(x).view(b_, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each [B_, heads, N, head_dim]
        bias = self._attn_bias(b_, mask)

        if self.attn_impl == "sdpa":
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            if bias is not None:
                attn = attn + bias
            out = attn.softmax(dim=-1) @ v

        out = out.transpose(1, 2).reshape(b_, n, c)
        return self.proj(out)


class WindowTransformerBlock(nn.Module):
    """One (optionally shifted) window-attention block, ``[B, C, H, W]`` in and out.

    Pre-norm residual attention + MLP, the standard Swin arrangement. Shifted
    blocks are what give KB §4's "broader object-level interaction than a
    shallow CNN"; without a shift, window boundaries are hard walls and the
    receptive field never crosses them.

    ``input_resolution`` is fixed at construction because the shift mask is a
    constant buffer — consistent with KB §12's "prefer fixed/bounded shapes"
    and with the fixed ``(1, 7, 288, 512)`` runtime contract.
    """

    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        num_heads: int,
        window_size: int,
        shift_size: int = 0,
        mlp_ratio: float = 2.0,
        use_rel_pos_bias: bool = True,
        attn_impl: str = "manual",
        shift_impl: str = "roll",
    ) -> None:
        super().__init__()
        h, w = input_resolution
        if h % window_size or w % window_size:
            raise ValueError(f"resolution {input_resolution} not divisible by window {window_size}")
        if not 0 <= shift_size < window_size:
            raise ValueError(f"shift_size {shift_size} must be in [0, {window_size})")
        if shift_impl not in SHIFT_IMPLS:
            raise ValueError(f"unknown shift_impl {shift_impl!r}, expected one of {SHIFT_IMPLS}")
        self.input_resolution = (h, w)
        self.window_size = window_size
        self.shift_size = shift_size
        self.shift_impl = shift_impl

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads, use_rel_pos_bias, attn_impl)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

        if shift_size > 0:
            self.register_buffer(
                "attn_mask", build_shift_attention_mask(h, w, window_size, shift_size),
                persistent=False,
            )
        else:
            self.attn_mask = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        # Skipped while tracing: shapes are fixed by contract, and comparing
        # traced shape values would bake a Python bool into the ONNX graph.
        if not torch.jit.is_tracing() and (h, w) != self.input_resolution:
            raise ValueError(f"expected {self.input_resolution}, got {(h, w)}")
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]

        shortcut = x
        y = self.norm1(x)
        y = cyclic_shift(y, -self.shift_size, self.shift_impl)
        windows = window_partition(y, self.window_size)
        windows = self.attn(windows, self.attn_mask)
        y = window_reverse(windows, self.window_size, h, w)
        y = cyclic_shift(y, self.shift_size, self.shift_impl)
        x = shortcut + y
        x = x + self.mlp(self.norm2(x))

        return x.permute(0, 3, 1, 2).contiguous()
