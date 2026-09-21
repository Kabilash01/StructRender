"""Model E — StructFormer + structured entity awareness. KB §4, Milestone 7.

Rung E of the ablation ladder: Model C's architecture plus per-entity tokens that
each pixel may read back from, but **only** through the engine's own mask. The
rule is KB §4's, unchanged:

    A_{p,i} = Q_p K_i^T / sqrt(d)   if p in S_i
            = -inf                  otherwise

so an NPC's pixels cannot retrieve a car's token. Addressing is by engine mask
only, never by visual similarity (Decision 9: entity IDs are addresses).

**Placement.** Entity attention sits at the stage-2 bottleneck, on the deepest
and smallest feature map. Two reasons, and the second is the load-bearing one:
the bottleneck is where features are most semantic, and it is 1/16 resolution, so
the K_max x HW attention that KB §12 flags as the project's operator risk stays
small enough to export. The mechanism itself was already built, tested and
TensorRT-verified at M5 (`eval/reports/m7_mask_strictness_trt.json`) precisely so
that this wiring step could not discover an export failure after the fact.

**Zero-initialised output.** `EntityTokenProjection` ends in a zeroed linear, so
an untrained Model E is numerically identical to Model C. "Did entity awareness
change anything?" is then a question about training, not about initialisation —
the same discipline Model B's temporal fusion used, and the same discipline that
let Decision 66 prove B's fusion had never engaged.
"""

from __future__ import annotations

import torch
from torch import nn

from renderer.structformer.masked_entity_attention import EntityMaskedCrossAttention
from renderer.structformer.model import StructFormer

from .pooling import EntityTokenProjection, entity_consistency_loss, masked_mean_pool


class EntityAwareStructFormer(nn.Module):
    """Model E: Model C plus entity-masked cross-attention at the bottleneck."""

    def __init__(self, k_max: int = 8, entity_heads: int = 1,
                 token_dim: int | None = None, **structformer_kwargs) -> None:
        super().__init__()
        self.backbone = StructFormer(**structformer_kwargs)
        self.k_max = k_max
        dim = self.backbone.stage2.out_dim if hasattr(self.backbone.stage2, "out_dim") \
            else self.backbone.decoder.in_dim if hasattr(self.backbone.decoder, "in_dim") \
            else structformer_kwargs.get("embed_dim", 32) * 2
        self.feature_dim = dim
        self.token_dim = token_dim or dim
        self.pool_proj = EntityTokenProjection(dim, self.token_dim)
        self.entity_attn = EntityMaskedCrossAttention(
            dim=dim, token_dim=self.token_dim, num_heads=entity_heads, k_max=k_max,
            attn_impl=structformer_kwargs.get("attn_impl", "manual"))
        # Zeroing the TOKEN projection is not enough to start the entity path at
        # zero: `to_k`/`to_v` carry biases, so zero tokens still produce non-zero
        # keys and values, and `proj` adds a bias of its own. The attention's
        # OUTPUT projection is the only place that makes the whole branch vanish.
        # Without this, Model E differs from Model C at initialisation and "did
        # entity awareness help?" stops being answerable — the exact confound
        # Decision 66 found in Model B's fusion.
        nn.init.zeros_(self.entity_attn.proj.weight)
        nn.init.zeros_(self.entity_attn.proj.bias)

    def _tokens(self, feats: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        return self.pool_proj(masked_mean_pool(feats, masks))

    def forward(self, structure: torch.Tensor, style_token: torch.Tensor,
                entity_masks: torch.Tensor | None = None,
                return_tokens: bool = False):
        b = structure.shape[0]
        bb = self.backbone
        x = bb.stem(structure)
        gamma, beta = bb.style_film(style_token).chunk(2, dim=-1)
        x = x * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
        x, skip1 = bb.stage1(x)
        x, _ = bb.stage2(x)

        tokens = None
        if entity_masks is not None:
            # Masks arrive at IMAGE resolution; attention runs at the bottleneck.
            # area interpolation keeps a small entity from vanishing the way
            # nearest sampling can when it is thinner than the stride.
            m = torch.nn.functional.interpolate(
                entity_masks, size=x.shape[-2:], mode="area")
            m = (m > 0).to(x.dtype)
            tokens = self._tokens(x, m)
            x = x + self.entity_attn(x, tokens, m)

        x = bb.decoder(x, skip1)
        out = bb.head(x)
        return (out, tokens) if return_tokens else out


class EntityAwareSingleStyle(nn.Module):
    """Deployment form: one committed style folded into a learned parameter.

    Mirrors `StructFormerSingleStyle` (Decision 45's export contract) — a token
    that is constant at inference has no business being a runtime graph input.
    Entity masks DO stay an input: they change every frame.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.net = EntityAwareStructFormer(**kwargs)
        self.style_token = nn.Parameter(torch.zeros(1, self.net.backbone.style_dim))

    def forward(self, structure: torch.Tensor,
                entity_masks: torch.Tensor | None = None) -> torch.Tensor:
        tok = self.style_token.expand(structure.shape[0], -1)
        return self.net(structure, tok, entity_masks)


__all__ = ["EntityAwareStructFormer", "EntityAwareSingleStyle", "entity_consistency_loss"]
