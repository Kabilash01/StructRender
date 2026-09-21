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


def entity_consistency_loss(tokens: torch.Tensor, masks: torch.Tensor,
                            prev_tokens: torch.Tensor | None = None,
                            prev_masks: torch.Tensor | None = None) -> torch.Tensor:
    """Penalise an entity's token for drifting between frames — KB §4, M7 deliverable.

    Only slots occupied in BOTH frames contribute; a slot that was empty in either
    has no pair to be consistent with, and including it would pull real tokens
    toward the zero of a padded slot. With no valid pair the loss is exactly zero,
    which keeps it safe to add to the objective unconditionally.
    """
    if prev_tokens is None or prev_masks is None:
        return tokens.new_zeros(())
    occupied = (masks.flatten(2).sum(2) > 0) & (prev_masks.flatten(2).sum(2) > 0)  # [B,K]
    if not bool(occupied.any()):
        return tokens.new_zeros(())
    d = (tokens - prev_tokens).pow(2).mean(dim=2)   # [B,K]
    return (d * occupied.float()).sum() / occupied.float().sum().clamp_min(EPS)
