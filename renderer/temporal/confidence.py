"""Confidence / disocclusion mask — KB §4 *Engine-Guided Temporal Reprojection Module*.

§4 specifies three lines:

    e_D = |D_t - W(D_{t-1}, V_t)|
    e_N = 1 - N_t . W(N_{t-1}, V_t)
    M_t = exp(-alpha e_D - beta e_N)

and one boundary that the three lines cannot express on their own: *"newly
disoccluded pixels are not filled from stale history"*. This module is the
exponential **times** the hard zeros the boundary requires, because there are
cases where the exponential is not merely large but **meaningless** — it is
computed from operands that do not exist, or from a motion value that is
indistinguishable from a correct one.

**The five hard zeros, and why an exponential cannot stand in for any of them.**

1. ``motion_valid == False`` (frame 0). There is no previous frame at all, so
   ``W(.)`` samples an undefined tensor. Whole-frame zero.
2. **Out of bounds.** ``uv_curr + motion`` leaves the previous frame: the surface
   was off-screen and has no history. ``e_D`` would compare against zero-padding,
   which is a number, not evidence.
3. **S7 — an entity with no ``t-1`` transform.** The exporter writes ``(0, 0)``
   for it (``frame_structure.md`` §4), which is **bit-identical to the field it
   writes for a perfectly static surface**. ``e_D`` and ``e_N`` are then both
   ~0 and the exponential returns ~1.0: maximum confidence in history that does
   not exist. This is the case the formula gets exactly backwards, and it is why
   the mask reads the manifest and not only the tensors.
4. **Background / sky at ``t``.** ``depth == +inf`` there and motion is pinned to
   ``(0, 0)`` by convention, so ``e_D = |inf - inf|`` is NaN, not large. Same
   ``(0,0)``-means-two-things collision as S7.
5. **The warped source was background.** Not in the brief's list and added here:
   a pixel that is a surface now and was sky then is a genuine disocclusion, and
   ``e_D`` is again ``|D_t - inf|``. This is flagged for a Decision Log entry —
   §4's formula is silently undefined whenever *either* side of the subtraction
   is the background sentinel, and only zeroing the current-frame side leaves
   half of that undefined.

``alpha`` and ``beta`` are **calibrated from measured error distributions**, not
chosen. See :data:`ALPHA` / :data:`BETA` and
``eval/reports/m6_mask_calibration.json``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from renderer.temporal.warp import backward_warp, in_bounds

#: Depth error, in METRES, at which the depth term alone takes the mask to 0.5:
#: ``alpha = ln(2) / e_D*``.
#:
#: **Calibrated, not chosen.** The physical question is "has the warped sample
#: crossed onto a different surface", and the scene answers it directly: the
#: engine's ``entity_id`` map labels every pixel edge as *within a surface* or
#: *across an object silhouette* (an address, never a value — KB Decision 9), and
#: the depth step across each is measured. ``0.175 m`` is the **Youden-optimal
#: separation** between those two measured distributions: 91 % of silhouette
#: steps are above it and 3.2 % of within-surface steps are (J = 0.8797).
#:
#: The separation curve is **flat**: J = 0.879 at 0.175 m against 0.873 at 0.10 m
#: and 0.861 at 0.35 m, so the third digit carries no weight and the value should
#: not be re-tuned to move a result. Distributions, the sweep and the resulting
#: keep-fractions are in ``eval/reports/m6_mask_calibration.json``, written by
#: ``scripts/calibrate_m6_mask.py``.
E_D_HALF_METRES = 0.175

#: Normal error at which the normal term alone takes the mask to 0.5:
#: ``1 - cos(30 deg)``. A 30-degree change of surface orientation is a change of
#: facet, not a resampling artefact, and correctly-tracked normals sit far below
#: it — measured, within-surface neighbours differ by <= 2.5 degrees all the way
#: to the 99th percentile.
#:
#: **This term carries little information on THIS scene and the choice is not
#: load-bearing.** The same silhouette-vs-interior separation used for ``e_D``
#: is essentially binary for normals here — the geometry is axis-aligned boxes
#: and a ground plane, so a silhouette is usually a 90-degree face change — and
#: the Youden index is flat at 0.8995-0.8999 across every crossing from 10 to 60
#: degrees. 30 degrees sits inside that plateau. A scene with curved or
#: smoothly-varying geometry would make this term discriminative and would need
#: the calibration re-run; that is a property of the scene, not of the formula.
E_N_HALF = 1.0 - 0.8660254037844387  # 1 - cos(30 deg) = 0.133975

ALPHA = float(torch.log(torch.tensor(2.0)) / E_D_HALF_METRES)
BETA = float(torch.log(torch.tensor(2.0)) / E_N_HALF)

#: How much of a bilinear tap may come from previous-frame background before the
#: sample counts as disoccluded. 1e-3 means "essentially none": a tap that mixes
#: in any measurable amount of sky is not history.
PREV_COVERAGE_EPS = 1e-3


def depth_error(depth_t: torch.Tensor, depth_prev: torch.Tensor,
                motion: torch.Tensor) -> torch.Tensor:
    """``e_D = |D_t - W(D_{t-1}, V_t)|`` in metres.

    Both depths must already carry a **finite** background value (the ``+inf``
    sentinel replaced), with the affected pixels handled by the hard zeros —
    ``inf - inf`` is NaN, and a NaN multiplied by a hard zero is still NaN.
    """
    return (depth_t - backward_warp(depth_prev, motion)).abs()


def normal_error(normal_t: torch.Tensor, normal_prev: torch.Tensor,
                 motion: torch.Tensor) -> torch.Tensor:
    """``e_N = 1 - N_t . W(N_{t-1}, V_t)``, clamped to ``[0, 2]``.

    §4's form, taken literally: the warped normal is **not** renormalised.
    Bilinear interpolation across a silhouette shortens the vector, which raises
    ``e_N`` there — and a shortened normal at a silhouette *is* unreliable
    history, so the literal form is doing useful work rather than accumulating an
    artefact. Renormalising would hide it. The calibration artifact records both
    distributions so the choice is checkable rather than asserted.
    """
    dot = (normal_t * backward_warp(normal_prev, motion)).sum(dim=1, keepdim=True)
    return (1.0 - dot).clamp(0.0, 2.0)


def hard_valid(
    motion: torch.Tensor,
    depth_finite_t: torch.Tensor,
    depth_finite_prev: torch.Tensor,
    motion_valid: torch.Tensor | bool,
    no_prev_transform: torch.Tensor | None = None,
) -> torch.Tensor:
    """The product of the five hard zeros — ``(B, 1, H, W)`` float in ``{0, 1}``.

    Multiplicative, not additive: each term independently invalidates the pixel,
    and any one of them being zero must win regardless of the others.

    **The bounds term and the previous-coverage term overlap, deliberately.**
    :data:`renderer.temporal.warp.PADDING_MODE` is ``"zeros"``, so a sample that
    leaves the frame already warps the previous validity to 0 and the coverage
    term would catch it on its own — removing ``in_bounds`` here changes no
    output today. It is kept because the coverage term's coverage of this case is
    a *consequence of a sampling option*: switch to ``border`` padding and the
    clamped edge value makes the warped validity 1 off-frame, and the mask would
    silently start trusting off-frame history. ``in_bounds`` states the condition
    geometrically, independent of how the sampler pads. Asserted by
    ``test_each_term_alone_zeroes_the_off_frame_region``.
    """
    v = in_bounds(motion) * depth_finite_t
    v = v * (backward_warp(depth_finite_prev, motion) > 1.0 - PREV_COVERAGE_EPS).to(v.dtype)
    if no_prev_transform is not None:
        v = v * (1.0 - no_prev_transform)
    if isinstance(motion_valid, bool):
        motion_valid = torch.full((motion.shape[0], 1, 1, 1), float(motion_valid),
                                  dtype=v.dtype, device=v.device)
    return v * motion_valid.to(v.dtype).reshape(-1, 1, 1, 1)


def confidence_mask(
    depth_t: torch.Tensor,
    depth_prev: torch.Tensor,
    normal_t: torch.Tensor,
    normal_prev: torch.Tensor,
    motion: torch.Tensor,
    depth_finite_t: torch.Tensor,
    depth_finite_prev: torch.Tensor,
    motion_valid: torch.Tensor | bool = True,
    no_prev_transform: torch.Tensor | None = None,
    alpha: float = ALPHA,
    beta: float = BETA,
) -> dict:
    """``M_t`` and the two error fields it was built from.

    Returned as a dict because every caller that wants the mask also wants the
    errors — the calibration reads their distributions, and a temporal loss that
    reported only the product would make an unusable mask indistinguishable from
    a scene with no history.
    """
    e_d = depth_error(depth_t, depth_prev, motion)
    e_n = normal_error(normal_t, normal_prev, motion)
    valid = hard_valid(motion, depth_finite_t, depth_finite_prev,
                       motion_valid, no_prev_transform)
    # Errors are zeroed where they are undefined BEFORE the exponential, so an
    # inf/NaN operand can never reach it. A NaN times a hard zero is still NaN.
    soft = torch.exp(-alpha * (e_d * valid) - beta * (e_n * valid))
    return {"mask": valid * soft, "e_d": e_d, "e_n": e_n, "hard_valid": valid, "soft": soft}


class ConfidenceMask(nn.Module):
    """``M_t`` at one fixed resolution. Thin wrapper; the arithmetic is above."""

    def __init__(self, alpha: float = ALPHA, beta: float = BETA):
        super().__init__()
        self.alpha, self.beta = float(alpha), float(beta)

    def forward(self, **kw) -> dict:
        return confidence_mask(alpha=self.alpha, beta=self.beta, **kw)
