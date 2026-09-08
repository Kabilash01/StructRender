"""Entity-masked cross-attention — KB §4 *Entity-Masked Cross-Attention Module*.

**Status: M7 mechanism, probed at M5, NOT part of the M5 Model C graph.** It is
written and exported now for one reason only: KB §12 flags "custom
attention/masking operation exports poorly" as the project's operator risk, and
this is the most custom masking operation StructRender will ever ship. Finding
out at M7 that it cannot reach TensorRT would change the architecture after the
architecture was built. Nothing here is trained, wired into
:mod:`renderer.structformer.model`, or on the M5 critical path.

The rule it implements is KB §4's, unchanged::

    A_{p,i} = Q_p K_i^T / sqrt(d)   if p in S_i
            = -inf                  otherwise

so an NPC's spatial features cannot retrieve a car's appearance memory. Memory
addressing is by engine mask only — never by visual similarity (KB §4 Explicit
Boundaries, KB Decision 9: entity IDs are addresses, not features).

Two things here are numerical spellings of that rule rather than departures from
it, and both are reported at M5 rather than discovered at M7:

1. Illegal keys are masked to ``-inf`` so their softmax weight is EXACTLY zero
   (KB Decision 52). An earlier version added a finite ``MASK_NEG = -100``, which
   only *discourages* an illegal key: once the legal logits fall below -100 the
   illegal ones dominate and one entity's memory leaks into another's pixels.
   Strict masking removes that failure mode for any finite logits. A literal
   ``-inf`` alone would make softmax return NaN on any pixel that
   belongs to no entity — which is most of a frame.
2. Because softmax always normalises to 1, a fully masked pixel would still emit
   a convex combination of entity values. The module therefore multiplies its
   output by a per-pixel gate ``max_i S_i(p)`` — 0 where no entity is visible.
   That is the exact intent of ``-inf`` for the "no legal key" case, and it is
   what keeps a background pixel from receiving any entity's memory.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from renderer.structformer.window_attention import ATTN_IMPLS


class EntityMaskedCrossAttention(nn.Module):
    """Spatial queries attend to at most ``k_max`` entity memory tokens.

    Args:
        dim: spatial feature width (queries).
        token_dim: entity token width (keys/values).
        num_heads: attention heads.
        k_max: fixed entity-slot count. Fixed, not dynamic, per KB §12's accepted
            debt ("fixed ``K_max`` entity slots") and §5's runtime contract, where
            unused slots are padded. Which entities occupy the slots is the
            packing policy's job (KB Decision 21: rank visible entities by
            ``pixel_count`` descending) and is deliberately outside this module.
        attn_impl: ``"manual"`` or ``"sdpa"``.

    Forward:
        ``x``: ``[B, C, H, W]`` spatial features.
        ``entity_tokens``: ``[B, K, Dt]`` — EPAM tokens, padded to ``k_max``.
        ``entity_mask``: ``[B, K, H, W]`` in ``{0, 1}`` — the engine's per-entity
        mask ``S_i``. Padded slots carry an all-zero mask, which is what makes
        padding harmless without a separate validity flag.

    Returns ``[B, C, H, W]``: the attention contribution, gated to zero outside
    every mask. The caller adds it residually.
    """

    def __init__(
        self,
        dim: int,
        token_dim: int,
        num_heads: int,
        k_max: int,
        attn_impl: str = "manual",
    ) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim {dim} not divisible by num_heads {num_heads}")
        if attn_impl not in ATTN_IMPLS:
            raise ValueError(f"unknown attn_impl {attn_impl!r}, expected one of {ATTN_IMPLS}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.k_max = k_max
        self.attn_impl = attn_impl
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(token_dim)
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(token_dim, dim)
        self.to_v = nn.Linear(token_dim, dim)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self, x: torch.Tensor, entity_tokens: torch.Tensor, entity_mask: torch.Tensor
    ) -> torch.Tensor:
        b, c, h, w = x.shape
        k = entity_tokens.shape[1]
        if k != self.k_max:
            raise ValueError(f"expected {self.k_max} entity slots, got {k}")
        if entity_mask.shape != (b, k, h, w):
            raise ValueError(f"entity_mask {tuple(entity_mask.shape)} != {(b, k, h, w)}")

        seq = x.flatten(2).transpose(1, 2)  # [B, HW, C]
        q = self.to_q(self.norm_q(seq)).view(b, h * w, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.norm_kv(entity_tokens)
        key = self.to_k(kv).view(b, k, self.num_heads, self.head_dim).transpose(1, 2)
        val = self.to_v(kv).view(b, k, self.num_heads, self.head_dim).transpose(1, 2)

        # [B, 1, HW, K] boolean: True where p in S_i. STRICT exclusion (KB
        # Decision 52) — an additive -100 only *discourages* an illegal key, and
        # once legal logits fall below it the illegal ones dominate and one
        # entity's memory leaks into another's pixels. Masking to -inf makes the
        # illegal softmax weight exactly 0 for any finite logits.
        legal = entity_mask.flatten(2).transpose(1, 2).unsqueeze(1).bool()  # [B,1,HW,K]
        # A row with no legal key would softmax to NaN. Give those rows a
        # uniform row (they are gated to exactly zero after projection anyway),
        # so nothing downstream sees a non-finite value.
        row_has_legal = legal.any(dim=-1, keepdim=True)
        safe = legal | ~row_has_legal

        if self.attn_impl == "sdpa":
            out = F.scaled_dot_product_attention(q, key, val, attn_mask=safe)
        else:
            attn = (q @ key.transpose(-2, -1)) * self.scale
            attn = attn.masked_fill(~safe, float("-inf"))
            out = attn.softmax(dim=-1) @ val

        out = out.transpose(1, 2).reshape(b, h * w, c)
        out = self.proj(out).transpose(1, 2).view(b, c, h, w)
        # Zero out pixels that belong to no entity (see module docstring, note 2).
        # This is what makes the uniform substitution above sound: a fully-masked
        # row contributes exactly 0.0, not a blend of every entity's memory.
        return out * entity_mask.amax(dim=1, keepdim=True)
