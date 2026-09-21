"""Masked entity-token injection — what entity conditioning actually is here.

KB §4 specifies entity-masked cross-attention::

    A_{p,i} = Q_p K_i^T / sqrt(d)   if p in S_i
            = -inf                  otherwise

Measured on ladder-v3 data (`scripts/probe_entity_attention_degeneracy.py`):
**1,171 bottleneck cells have a legal entity and 0 have more than one.** Entity
masks are disjoint by construction — §4 defines `S_i(x,y) = 1[ID(x,y) = i]`, so a
pixel belongs to exactly one entity — which means the softmax runs over a set of
size ONE and returns 1.0 whatever the scores are.

Two consequences, both measured rather than argued:

* the attention weights are **constant**, so `d(loss)/d(Q) = d(loss)/d(K) = 0`
  exactly. `norm_q`, `to_q`, `to_k` — **374,976 parameters, 5.0 % of Model E** —
  receive zero gradient at every step. Confirmed still dead at step 11 while
  every other entity parameter had come alive.
* what survives is not attention at all: pixel *p* receives entity *i*'s value
  vector where *i* is *p*'s entity. That is a **lookup**.

This module is that lookup, written as what it is. It is mathematically identical
to the masked-attention path on disjoint masks — the softmax it removes was the
constant 1.0 — so this is a simplification, **not** a change of behaviour and not
a departure from §4's rule. §4's `-inf` guarantee still holds exactly: an
entity's token reaches only pixels its engine mask covers.

The alternative — letting every pixel attend to every entity token so that Q/K
have something to choose between — would make attention meaningful but would
**violate** §4's masking rule, and is therefore not taken here.
"""

from __future__ import annotations

import torch
from torch import nn


class MaskedEntityInjection(nn.Module):
    """Add each pixel's own entity token, projected to feature space.

    ``x``: ``[B,C,H,W]`` features. ``tokens``: ``[B,K,token_dim]``.
    ``masks``: ``[B,K,H,W]``, disjoint, 1 where entity k owns the pixel.
    """

    def __init__(self, dim: int, token_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(token_dim)
        self.to_v = nn.Linear(token_dim, dim)
        self.proj = nn.Linear(dim, dim)
        # Zero output so Model E starts numerically identical to Model C, and
        # "did entity awareness help?" stays a question about training rather
        # than initialisation. The Model B lesson (Decision 66) is that zero-init
        # alone does NOT guarantee the branch ever engages, so engagement is
        # measured after training, not assumed here.
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, tokens: torch.Tensor,
                masks: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        v = self.to_v(self.norm(tokens))              # [B,K,C]
        # einsum over the mask IS the -inf rule on disjoint masks: a pixel sums
        # only the tokens whose mask covers it, and there is at most one.
        out = torch.einsum("bkhw,bkc->bchw", masks, v)
        out = self.proj(out.flatten(2).transpose(1, 2)).transpose(1, 2).view(b, c, h, w)
        # The gate is not decoration. `proj` has a bias, so a background pixel —
        # which sums no token — would still receive that bias and a fragment of
        # entity appearance would land where the engine says there is no entity.
        # The attention path gates for the same reason; dropping it here was the
        # one real difference between the two, and it showed up as a non-zero
        # background in the equivalence check.
        return out * masks.amax(dim=1, keepdim=True)
