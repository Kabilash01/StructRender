"""Model F — StructRender: StructFormer + entity conditioning + EPAM + multi-style.

Rung F of the ablation ladder (KB §10 M8). Built on Model E (Decision 77) and
trained, per the §4 curriculum, by resuming from E's checkpoint.

How memory enters the network. Per entity, the bottleneck descriptor
`F_i = Pool(F ⊙ S_i)` is folded into EPAM's EMA,

    M_i = gamma * M_i(prev).detach() + (1 - gamma) * F_i

and the entity token is `proj(M_i)`. With gamma = 0 this is exactly Model E
(token from the current frame alone), so **gamma = 0 is the no-memory control**
inside the same code, data and step budget — the ablation that isolates memory
and nothing else.

Memory is detached (§4: EPAM stores a compact, non-differentiable embedding),
which is also what makes clip training affordable: history frames need only a
no-grad forward.

Style is a learned token per style_id, selected by index. That is Decision 15's
"encode once and cache", with the cache being the learned table. Zero-shot styles
are out of scope (Decision 70); adding a style means generating its targets and
training its row.
"""

from __future__ import annotations

import torch
from torch import nn

from renderer.entity.injection import MaskedEntityInjection
from renderer.entity.model import downsample_masks_disjoint
from renderer.entity.pooling import EntityTokenProjection, masked_mean_pool
from renderer.structformer.model import StructFormer


class StructRenderF(nn.Module):
    def __init__(self, style_ids: list[str], k_max: int = 8, **structformer_kwargs) -> None:
        super().__init__()
        self.backbone = StructFormer(**structformer_kwargs)
        self.style_ids = list(style_ids)
        self.k_max = k_max
        self.style_table = nn.Parameter(torch.empty(len(style_ids), self.backbone.style_dim))
        nn.init.trunc_normal_(self.style_table, std=0.02)   # not zeros: see StructFormerSingleStyle
        dim = int(structformer_kwargs.get("embed_dim", 32)) * 2
        self.feature_dim = dim
        self.pool_proj = EntityTokenProjection(dim, dim)
        self.entity_attn = MaskedEntityInjection(dim=dim, token_dim=dim)

    # ---- stages, exposed so EPAM can sit between them at runtime ----------
    def encode(self, structure: torch.Tensor, style_idx: torch.Tensor):
        bb = self.backbone
        x = bb.stem(structure)
        g, b = bb.style_film(self.style_table[style_idx]).chunk(2, dim=-1)
        x = x * (1.0 + g[:, :, None, None]) + b[:, :, None, None]
        x, skip = bb.stage1(x)
        x, _ = bb.stage2(x)
        return x, skip

    def bottleneck_masks(self, entity_masks: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        return downsample_masks_disjoint(entity_masks, feats.shape[-2:])

    def describe(self, feats: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """Current-frame entity descriptors F_i, [B,K,D]."""
        return masked_mean_pool(feats, m)

    def decode(self, feats, skip, memory: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        tokens = self.pool_proj(memory)
        x = feats + self.entity_attn(feats, tokens, m)
        return self.backbone.head(self.backbone.decoder(x, skip))

    def forward(self, structure, style_idx, entity_masks, memory=None):
        """One frame. `memory` [B,K,D] is the EMA already including this frame;
        None means no history, i.e. memory = current descriptor (Model E)."""
        feats, skip = self.encode(structure, style_idx)
        m = self.bottleneck_masks(entity_masks, feats)
        cur = self.describe(feats, m)
        return self.decode(feats, skip, cur if memory is None else memory, m)

    @torch.no_grad()
    def load_from_model_e(self, e_state: dict) -> dict:
        """Curriculum resume (§4 stage 5 from stage 4). Maps E's backbone, pooling
        and injection weights 1:1, and seeds every style row from E's single
        folded style token so both styles start where E ended."""
        own = self.state_dict()
        mapped, skipped = {}, []
        for k, v in e_state.items():
            t = k[len("net."):] if k.startswith("net.") else k
            if t == "style_token":
                mapped["style_table"] = v.expand_as(own["style_table"]).clone()
            elif t in own and own[t].shape == v.shape:
                mapped[t] = v
            else:
                skipped.append(k)
        missing = [k for k in own if k not in mapped]
        self.load_state_dict(mapped, strict=False)
        return {"loaded": len(mapped), "skipped": skipped, "missing": missing}


def ema_memory(prev: torch.Tensor | None, prev_valid: torch.Tensor | None,
               cur: torch.Tensor, cur_valid: torch.Tensor, gamma: float) -> tuple:
    """Slot-aligned EMA step. `prev`/`cur` [B,K,D] must already be aligned by
    EntityKey (see `align_by_key`), never by raw slot index (Decision 76).

    An entity with no valid history starts from its current descriptor. An entity
    not visible now keeps its memory unchanged (§4: KEEP UNCHANGED while out of
    view) — handled by the caller carrying `prev` forward for absent keys."""
    if prev is None or gamma == 0.0:
        return cur, cur_valid
    has = (prev_valid & cur_valid).unsqueeze(-1)
    mem = torch.where(has, gamma * prev.detach() + (1.0 - gamma) * cur, cur)
    return mem, cur_valid


def align_by_key(prev: torch.Tensor, prev_keys: torch.Tensor,
                 cur_keys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Re-order `prev` [B,K,D] so slot k holds the memory of whatever entity is in
    slot k NOW. Keys are int64 [B,K] encoding (instance_id, generation), 0 = empty.
    Slots are pixel-count ranks and re-rank every frame, so matching by slot index
    would pair different entities — the Decision 76 defect."""
    b, k, d = prev.shape
    out = torch.zeros_like(prev)
    valid = torch.zeros(b, k, dtype=torch.bool, device=prev.device)
    for bi in range(b):
        where = {int(x): i for i, x in enumerate(prev_keys[bi].tolist()) if x != 0}
        for s, x in enumerate(cur_keys[bi].tolist()):
            if x != 0 and x in where:
                out[bi, s] = prev[bi, where[x]]
                valid[bi, s] = True
    return out, valid


def key_code(instance_id: int, generation: int) -> int:
    """Pack an EntityKey's per-session identity into one int64 for tensors.
    session_id is constant within a clip, so it is not needed here."""
    return (int(instance_id) << 16) | (int(generation) & 0xFFFF)
