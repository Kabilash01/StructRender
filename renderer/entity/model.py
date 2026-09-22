"""Model E — StructFormer + structured entity awareness. KB §4, Milestone 7.

Rung E of the ablation ladder: Model C's architecture plus per-entity tokens that
each pixel may read back from, but **only** through the engine's own mask. The
rule is KB §4's, unchanged:

    A_{p,i} = Q_p K_i^T / sqrt(d)   if p in S_i
            = -inf                  otherwise

so an NPC's pixels cannot retrieve a car's token. Addressing is by engine mask
only, never by visual similarity (Decision 9: entity IDs are addresses).

**Placement.** Entity conditioning sits at the stage-2 bottleneck, on the deepest
and smallest feature map. Two reasons, and the second is the load-bearing one:
the bottleneck is where features are most semantic, and it is **1/8** resolution
(STEM_STRIDE=4 plus one stage-2 halving = 36x64 at 288x512), so the K_max x HW
cost that KB §12 flags as the project's operator risk stays small. An earlier
version of this docstring said 1/16, which understated the attention positions
by 4x — the export-safety argument was being made against a resolution the model
does not have. The mechanism itself was already built, tested and
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

from renderer.structformer.model import StructFormer

from .injection import MaskedEntityInjection

from .pooling import EntityTokenProjection, entity_consistency_loss, masked_mean_pool


def downsample_masks_disjoint(masks: torch.Tensor, size) -> torch.Tensor:
    """Image-resolution entity masks -> bottleneck masks that are still DISJOINT.

    KB §4 defines `S_i(x,y) = 1[ID(x,y) = i]`, which is disjoint by construction:
    a pixel belongs to exactly one entity. Attention runs at the bottleneck, so
    the masks have to be resampled, and the obvious spelling — `interpolate(area)`
    then `> 0` — silently **dilates** them: any cell containing a single pixel of
    entity *i* becomes a full cell for *i*. Two entities that are disjoint at
    image resolution then OVERLAP at the bottleneck, and with that:

      * pooling stops being "this entity's own pixels" — a shared cell's features
        enter both descriptors, so an NPC standing in front of a car contributes
        to the car's token;
      * §4's `-inf` rule holds only against the *dilated* mask, so entity *i*'s
        token legally reaches pixels the engine says belong to *j*.

    Both are exactly what the entity mechanism exists to prevent, and neither is
    visible in a test whose entities sit far apart.

    So each cell is assigned to **at most one** entity: the one holding the most
    pixels in it (`argmax` over area-pooled coverage), and to none at all when the
    cell is background. Disjointness is preserved, which is the property §4
    actually specifies. A thin entity can still lose a contested cell — that is a
    real cost, but losing coverage is a weaker failure than leaking identity.
    """
    cov = torch.nn.functional.interpolate(masks, size=size, mode="area")  # [B,K,h,w]
    best = cov.argmax(dim=1, keepdim=True)                                # [B,1,h,w]
    onehot = torch.zeros_like(cov).scatter_(1, best, 1.0)
    return onehot * (cov.sum(dim=1, keepdim=True) > 0).to(cov.dtype)


class EntityAwareStructFormer(nn.Module):
    """Model E: Model C plus entity-masked cross-attention at the bottleneck."""

    def __init__(self, k_max: int = 8, entity_heads: int = 1,
                 token_dim: int | None = None, **structformer_kwargs) -> None:
        # `entity_heads` is accepted and ignored: it configured the attention
        # heads that no longer exist. Kept so ladder configs written against the
        # attention form still load rather than failing on an unexpected key.
        super().__init__()
        self.backbone = StructFormer(**structformer_kwargs)
        self.k_max = k_max
        # Both `hasattr` branches this used to have were dead — neither EncoderStage
        # nor DecoderStage defines out_dim/in_dim, so it always fell through to the
        # kwarg, and was correct only because the fallback default happened to match
        # StructFormer's. Stated directly instead: stage 2 doubles the stem width.
        dim = int(structformer_kwargs.get("embed_dim", 32)) * 2
        self.feature_dim = dim
        self.token_dim = token_dim or dim
        self.pool_proj = EntityTokenProjection(dim, self.token_dim)
        # A LOOKUP, not attention — and that is a measured fact, not a shortcut.
        # Entity masks are disjoint (§4: S_i = 1[ID = i]), so on this data every
        # bottleneck cell has at most one legal entity (1,171 cells, 0 with more
        # than one). A masked softmax over a single element returns 1.0 whatever
        # the scores are, so Q and K receive exactly zero gradient forever —
        # 374,976 params, 5.0% of Model E, still dead at step 11 while every
        # other entity parameter had come alive. `MaskedEntityInjection` is
        # bit-identical to the attention path on disjoint masks (max diff 0.00e+00)
        # with the dead half removed. KB Decision 75.
        self.entity_attn = MaskedEntityInjection(dim=dim, token_dim=self.token_dim)
        # Zeroing the TOKEN projection is not enough to start the entity path at
        # zero: `to_k`/`to_v` carry biases, so zero tokens still produce non-zero
        # keys and values, and `proj` adds a bias of its own. The attention's
        # OUTPUT projection is the only place that makes the whole branch vanish.
        # Without this, Model E differs from Model C at initialisation and "did
        # entity awareness help?" stops being answerable — the exact confound
        # Decision 66 found in Model B's fusion.
        # (MaskedEntityInjection already zeroes its own output projection.)

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
            m = downsample_masks_disjoint(entity_masks, x.shape[-2:])
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
        # NOT zeros. `StructFormerSingleStyle` uses trunc_normal_ deliberately:
        # with a zero token, d(loss)/d(style_film.weight) is exactly zero, so the
        # FiLM weight never learns. M5 recorded that and guards it with a
        # gradient test; initialising to zeros here would silently regress it.
        self.style_token = nn.Parameter(torch.empty(1, self.net.backbone.style_dim))
        nn.init.trunc_normal_(self.style_token, std=0.02)

    def forward(self, structure: torch.Tensor,
                entity_masks: torch.Tensor | None = None) -> torch.Tensor:
        tok = self.style_token.expand(structure.shape[0], -1)
        return self.net(structure, tok, entity_masks)


__all__ = ["EntityAwareStructFormer", "EntityAwareSingleStyle", "entity_consistency_loss"]
