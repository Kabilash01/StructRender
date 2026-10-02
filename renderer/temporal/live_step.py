"""One LIVE temporal step of two-style B, as a single exportable graph (M10).

The scored model (Decision 95) renders frame t from:
  * the 7-channel structure of t,
  * the encoder features of t-1,
  * the engine motion field and the confidence mask M_t of the pair (t-1, t).

Live, the runtime has the structure of t and t-1, the motion field, the
spawned-entity mask (KB scenario S7) and whether a previous frame exists. This
module builds M_t from those INSIDE the graph with the same `confidence_mask`
the training data and the scorer use, so there is no second implementation of
the mask to drift. The depth, finite-depth indicator and normals the mask needs
are recovered from the structure channels themselves:

    x[0] = 1 / (1 + depth)   (0 on background)  ->  depth = 1/x[0] - 1, finite = x[0] > 0
    x[1:4] = normal                              (0 on background, as load_geometry)

The float round trip costs ~1e-6 m of depth at 20 m, against a 0.175 m mask
scale (E_D_HALF_METRES).

Outputs the frame AND this frame's encoder features, which the runtime feeds
back as `feat_prev` next frame — exactly `encode(structure_prev)`, which is what
`eval_forward` recomputes, without paying for the encoder twice.

The style is a constant of the graph: one engine per style, as Decision 96.
"""
from __future__ import annotations

import torch
from torch import nn

from renderer.temporal.confidence import confidence_mask


def geometry_from_structure(x: torch.Tensor):
    """``(depth_m, finite, normal)`` of a ``(B, 7, H, W)`` structure tensor."""
    inv = x[:, 0:1]
    finite = (inv > 0).to(x.dtype)
    depth = finite * (1.0 / inv.clamp_min(1e-12) - 1.0)
    return depth, finite, x[:, 1:4]


class LiveStep(nn.Module):
    def __init__(self, model, style_idx: int):
        super().__init__()
        self.model, self.style_idx = model, int(style_idx)

    def confidence(self, x_t, x_prev, motion, no_prev, motion_valid):
        d_t, f_t, n_t = geometry_from_structure(x_t)
        d_p, f_p, n_p = geometry_from_structure(x_prev)
        return confidence_mask(depth_t=d_t, depth_prev=d_p, normal_t=n_t, normal_prev=n_p,
                               motion=motion, depth_finite_t=f_t, depth_finite_prev=f_p,
                               motion_valid=motion_valid.reshape(-1), no_prev_transform=no_prev)["mask"]

    def forward(self, x_t, x_prev, feat_prev, motion, no_prev, motion_valid):
        """motion (B,2,H,W) in UV units; no_prev (B,1,H,W) in {0,1}; motion_valid (B,1,1,1)."""
        mask = self.confidence(x_t, x_prev, motion, no_prev, motion_valid)
        feat, skips = self.model.encode(x_t)
        fused = self.model.fuse(feat, feat_prev, motion, mask)
        style = torch.full((x_t.shape[0],), self.style_idx, dtype=torch.long, device=x_t.device)
        return self.model.decode(fused, skips, style), feat
