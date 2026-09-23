"""The ONE memory step, shared by training, evaluation and the runtime.

Training a model under a slightly different memory rule than it runs under is the
kind of mismatch that looks fine in every test and shows up only in the game.
So there is exactly one implementation, and both the clip trainer and the live
loop call it.
"""

from __future__ import annotations

import torch

from renderer.epam.memory import visibility_confidence


def memory_step(stores: list[dict], keys: torch.Tensor, cur: torch.Tensor,
                px: torch.Tensor, gamma: float, tau: float) -> torch.Tensor:
    """Fold this frame into per-item memory stores and return slot-aligned memory.

    stores : one dict per batch item, {key_code: Tensor[D]} (detached)
    keys   : [B,K] int64 key codes, 0 = empty slot
    cur    : [B,K,D] current descriptors F_i (may carry grad)
    px     : [B,K]   visible pixel count per slot at image resolution

    Per slot, following §4's lifecycle:
      * reliable (C > tau) and remembered -> M = gamma*old + (1-gamma)*F  (UPDATE)
      * reliable and new                  -> M = F                        (CREATE)
      * unreliable and remembered         -> M = old, store untouched
      * unreliable and new                -> M = F, nothing stored (no entry is
        created from an unreliable first sighting, but the frame is still rendered
        from what is visible — Model E's behaviour)
    The store always receives a DETACHED value. The returned tensor keeps grad
    through the current descriptor, so the (1-gamma) path trains the pooling.
    Entities absent from this frame are never touched: KEEP UNCHANGED.
    """
    out = cur.clone()
    b, k, _ = cur.shape
    for bi in range(b):
        st = stores[bi]
        for s in range(k):
            code = int(keys[bi, s])
            if code == 0:
                continue
            reliable = visibility_confidence(int(px[bi, s])) > tau
            old = st.get(code)
            if old is not None:
                old = old.to(cur.device, cur.dtype)
                if reliable and gamma > 0.0:
                    out[bi, s] = gamma * old + (1.0 - gamma) * cur[bi, s]
                elif not reliable:
                    out[bi, s] = old
            if reliable:
                st[code] = out[bi, s].detach()
    return out
