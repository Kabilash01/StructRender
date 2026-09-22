"""Entity feature pooling and token projection — KB §4, Milestone 7.

Turns per-pixel spatial features into one token per entity slot, which
``EntityMaskedCrossAttention`` then lets each pixel read back from — but only
from the slots whose mask covers that pixel.

The pooled descriptor is built from the entity's OWN pixels and nothing else.
That is the whole point: an entity's token must describe that entity, so an NPC's
token cannot carry a car's appearance even before masking gets involved. Decision
9 again — the ID selects pixels, it is never itself a feature.
"""

from __future__ import annotations

import torch
from torch import nn

EPS = 1e-6


def masked_mean_pool(features: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """[B,C,H,W] x [B,K,H,W] -> [B,K,C], the mean feature over each entity's pixels.

    An empty slot pools to exactly zero rather than NaN. Dividing by a zero pixel
    count is the obvious way to write this and produces NaN, which then poisons
    every downstream token in the batch through the linear layers — the same
    class of failure the attention module's all-zero-mask test exists to catch.
    """
    b, c, h, w = features.shape
    k = masks.shape[1]
    if masks.shape != (b, k, h, w):
        raise ValueError(f"masks {tuple(masks.shape)} != {(b, k, h, w)}")
    f = features.flatten(2)                      # [B, C, HW]
    m = masks.flatten(2)                         # [B, K, HW]
    total = m.sum(dim=2, keepdim=True)           # [B, K, 1]
    pooled = torch.bmm(m, f.transpose(1, 2))     # [B, K, C]
    return pooled / total.clamp_min(EPS)


class EntityTokenProjection(nn.Module):
    """Pooled entity descriptors -> entity tokens for cross-attention.

    Zero-initialised output layer, so a freshly built entity-aware model starts
    numerically identical to the model without it. That makes "did entity
    awareness change anything?" a question about training rather than about
    initialisation, the same discipline Model B's temporal fusion used.
    """

    def __init__(self, feature_dim: int, token_dim: int, hidden: int | None = None) -> None:
        super().__init__()
        hidden = hidden or token_dim
        self.norm = nn.LayerNorm(feature_dim)
        self.fc1 = nn.Linear(feature_dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, token_dim)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(self.norm(pooled))))


def entity_consistency_loss(tokens: torch.Tensor, keys: list[list],
                            prev_tokens: torch.Tensor | None = None,
                            prev_keys: list[list] | None = None) -> torch.Tensor:
    """Penalise an entity's token for drifting between frames — KB §4, M7 deliverable.

    Entities are paired by **EntityKey**, never by slot index. Slot index is a
    per-frame ``pixel_count`` rank (see `selection.select_entities`), so two
    entities that swap relative on-screen size — the normal case as objects
    approach and recede — swap slots. A slot-keyed loss then pulls entity A's
    token toward **entity B's** previous token, which is identity resolved by
    rank position: exactly what Decision 9 says identity must never be. The
    first version of this function did that, and its tests did not catch it
    because none of them swapped slots.

    ``keys`` / ``prev_keys`` are per-batch-item lists of the EntityKey (or any
    hashable engine-owned identity) occupying each slot, ``None`` for empty
    slots. Only entities present in BOTH frames contribute; an entity that
    appeared, vanished, or was overflow-dropped has no pair to be consistent
    with, and including it would pull a real token toward a padded zero. With no
    valid pair the loss is exactly zero, so it stays safe to add unconditionally.
    """
    if prev_tokens is None or prev_keys is None:
        return tokens.new_zeros(())
    rows, prev_rows = [], []
    for b, (now, before) in enumerate(zip(keys, prev_keys)):
        where = {k: i for i, k in enumerate(before) if k is not None}
        for slot, k in enumerate(now):
            if k is not None and k in where:
                rows.append(tokens[b, slot])
                prev_rows.append(prev_tokens[b, where[k]])
    if not rows:
        return tokens.new_zeros(())
    return (torch.stack(rows) - torch.stack(prev_rows)).pow(2).mean()
