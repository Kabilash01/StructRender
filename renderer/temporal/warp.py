"""Engine-guided backward warp — KB §4 *Engine-Guided Temporal Reprojection Module*.

    F~_{t-1} = W(F_{t-1}, V_t)

``V_t`` is the engine's motion field, whose convention is **pinned** by KB
Decision 25 and written out in ``data_pipeline/schemas/frame_structure.md`` §4:

* **backward** — ``motion[p]`` points from pixel ``p`` in frame ``t`` to where
  that same surface was in frame ``t-1``;
* **screen-UV normalised ``[0, 1]``**, **Y-down**, the same axis as every
  structural buffer;
* camera **and** rigid per-object motion, computed analytically from engine
  transforms, never estimated from pixels.

Two consequences follow directly and are the whole reason this module is ten
lines of arithmetic rather than a resampling library:

1. **UV is resolution-independent, so the field needs no rescaling to be applied
   at feature resolution.** A displacement of 0.01 UV is 0.01 of the width at
   512 px and at 128 px alike. Only the *base* grid changes with resolution.
   :func:`resize_motion` therefore resamples the field spatially and leaves the
   values alone — scaling them would be the classic optical-flow-in-pixels bug
   arriving through the back door.
2. **``grid_sample`` wants ``[-1, 1]``, so the grid is ``2 * (uv_curr + motion) - 1``**
   with **no** Y flip. Both axes point the same way: UV is Y-down and
   ``grid_sample``'s ``y`` also increases downward (``y = -1`` is row 0). This
   was verified against the recorded session rather than assumed — see
   ``renderer/temporal/tests/test_warp.py``, which measures the photometric
   error of the warp against the three wrong sign/axis variants on real frames
   with real camera motion, and against ``align_corners=True``.

``align_corners=False`` is not a preference either. With it, ``grid_sample``
maps a grid value ``g`` to pixel coordinate ``(g + 1) / 2 * N - 0.5``; feeding
``g = 2 * (x + 0.5) / N - 1`` gives exactly ``x``. That is the pixel-centre UV
the exporter emits (``uv_curr = (x + 0.5) / W``), so the two conventions compose
*exactly*, with zero motion mapping to the identity. ``align_corners=True``
carries an ``N / (N - 1)`` scale error — sub-pixel, but systematic, and it grows
with displacement.

Everything here is ONNX-safe and free of data-dependent control flow (§12):
fixed shapes, no ``if`` on tensor values, no dynamic padding. ``GridSample``
reaches a built and executed TensorRT engine in FP32 and FP16 — measured, see
``eval/reports/m6_gridsample_probe.json``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

#: Sampling options this module uses everywhere. ``zeros`` padding makes an
#: out-of-frame sample fetch 0 rather than smearing the border inward, so a
#: disocclusion at the frame edge reads as absent history instead of plausible
#: history. The confidence mask zeroes those pixels regardless; this just keeps
#: the intermediate honest.
MODE = "bilinear"
PADDING_MODE = "zeros"
ALIGN_CORNERS = False


def base_uv_grid(h: int, w: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Pixel-centre screen UV, ``(1, H, W, 2)`` ordered ``(u, v)`` — the exporter's
    ``uv_curr = ((x + 0.5) / W, (y + 0.5) / H)``."""
    ys = (torch.arange(h, device=device, dtype=dtype) + 0.5) / h
    xs = (torch.arange(w, device=device, dtype=dtype) + 0.5) / w
    vv, uu = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((uu, vv), dim=-1)[None]


def sampling_grid(motion: torch.Tensor, base_uv: torch.Tensor | None = None) -> torch.Tensor:
    """``(B, 2, H, W)`` motion -> ``(B, H, W, 2)`` grid in ``[-1, 1]``.

    ``2 * (uv_curr + motion) - 1``. No Y flip and no resolution rescaling; see
    the module docstring for why each is absent rather than forgotten.
    """
    if base_uv is None:
        base_uv = base_uv_grid(motion.shape[-2], motion.shape[-1], motion.device, motion.dtype)
    return 2.0 * (base_uv + motion.permute(0, 2, 3, 1)) - 1.0


def in_bounds(motion: torch.Tensor, base_uv: torch.Tensor | None = None) -> torch.Tensor:
    """``(B, 1, H, W)`` float 0/1 — did this pixel's sample land inside frame ``t-1``?

    A backward sample that leaves the previous frame has no history at all: the
    surface was off-screen. No exponential of a depth or normal error can express
    that, because there is nothing to compare against, so §4's ``M_t`` needs this
    as a hard zero rather than a term.

    The test is ``uv_curr + motion`` inside the unit square. That is the
    condition for the sample to be non-degenerate; a bilinear tap within half a
    pixel of the border still mixes in one clamped/zero-padded neighbour, which
    this deliberately tolerates — the alternative rejects a one-pixel frame
    border on every frame for a sub-pixel effect.
    """
    if base_uv is None:
        base_uv = base_uv_grid(motion.shape[-2], motion.shape[-1], motion.device, motion.dtype)
    uv = base_uv + motion.permute(0, 2, 3, 1)
    ok = (uv >= 0.0) & (uv <= 1.0)
    return (ok[..., 0] & ok[..., 1]).unsqueeze(1).to(motion.dtype)


def resize_motion(motion: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Resample the field to ``(h, w)``. **Values are not scaled** — UV is
    resolution-independent (Decision 25)."""
    if tuple(motion.shape[-2:]) == tuple(size):
        return motion
    return F.interpolate(motion, size=size, mode="bilinear", align_corners=ALIGN_CORNERS)


def backward_warp(
    prev: torch.Tensor,
    motion: torch.Tensor,
    base_uv: torch.Tensor | None = None,
    mode: str = MODE,
) -> torch.Tensor:
    """``W(prev, motion)`` — pull frame ``t-1`` into frame ``t`` coordinates.

    ``prev`` is ``(B, C, H, W)`` and may be an image or a feature map; ``motion``
    is ``(B, 2, H, W)`` at the SAME spatial size (use :func:`resize_motion`
    first, which is a no-op when they already match).
    """
    # Shape guard only, and skipped under tracing: comparing shapes there makes
    # the exporter warn about a Python bool on a traced value, and the shapes are
    # fixed by construction anyway (§12: fixed/bounded shapes).
    if not torch.jit.is_tracing() and motion.shape[-2:] != prev.shape[-2:]:
        raise ValueError(
            f"motion {tuple(motion.shape[-2:])} and prev {tuple(prev.shape[-2:])} differ; "
            "call resize_motion() explicitly — an implicit resize here would hide a "
            "resolution mismatch between the field and what it is applied to."
        )
    grid = sampling_grid(motion, base_uv)
    return F.grid_sample(prev, grid.to(prev.dtype), mode=mode,
                         padding_mode=PADDING_MODE, align_corners=ALIGN_CORNERS)


class EngineGuidedWarp(nn.Module):
    """The warp at one fixed resolution, with the base grid held as a buffer.

    Fixed shapes are §12's mitigation and §4's Target, so the resolution is a
    construction argument, not something read off the input at run time. The
    buffer is **non-persistent**: it is derived from that shape, not learned, and
    a persistent one would ride along in every checkpoint (the same call KB
    Decision 45 made for the relative-position index).
    """

    def __init__(self, height: int, width: int, mode: str = MODE):
        super().__init__()
        self.height, self.width, self.mode = height, width, mode
        self.register_buffer("base_uv", base_uv_grid(height, width), persistent=False)

    def forward(self, prev: torch.Tensor, motion: torch.Tensor) -> torch.Tensor:
        return backward_warp(prev, motion, self.base_uv.to(motion.dtype), self.mode)

    def in_bounds(self, motion: torch.Tensor) -> torch.Tensor:
        return in_bounds(motion, self.base_uv.to(motion.dtype))
