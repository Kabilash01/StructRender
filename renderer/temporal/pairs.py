"""Consecutive-frame pairs from a recorded session, for the M6 temporal path.

Shared by the mask calibration (``scripts/calibrate_m6_mask.py``), the temporal
metric (``eval/metrics/temporal.py``) and the warp tests, so that all three agree
on what a "pair" is, what a "tick delta" is and which pixels have no history.

**The tick delta is computed, never assumed.** The recorder drops to a reduced
rate during the long reappearance gaps: consecutive `frame_id`s are always one
apart, but the *simulation* time between them is 1 tick or 8 ticks in this
session, taken from each record's own ``tick`` field. That distinction is not
cosmetic — an 8-tick pair carries roughly 8x the displacement, and the training
split and the validation split do not contain the same mixture of the two.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class FramePair:
    """``curr`` warped back onto ``prev``. ``tick_delta`` is engine ticks, not frames."""

    prev_id: int
    curr_id: int
    tick_delta: int


def consecutive_pairs(session, frame_ids) -> list[FramePair]:
    """Every ``(f-1, f)`` with **both ends inside** ``frame_ids`` and ``motion_valid``.

    Both ends must be in the set. Reaching outside it for the previous frame
    would pull a guard-band or training frame into a validation measurement.
    """
    have = set(int(f) for f in frame_ids)
    out = []
    for f in sorted(have):
        if f - 1 not in have or not session.motion_valid(f):
            continue
        dt = int(session.record(f)["tick"]) - int(session.record(f - 1)["tick"])
        out.append(FramePair(prev_id=f - 1, curr_id=f, tick_delta=dt))
    return out


def _chw(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a)).float().permute(2, 0, 1)[None]


def load_geometry(session, frame_id: int) -> dict:
    """Depth / normal / entity-id / motion of one frame as ``(1, C, H, W)`` tensors.

    ``depth_m`` is linear eye-space **metres** with the ``+inf`` background
    sentinel replaced by ``0.0`` and a companion ``depth_finite`` indicator.
    §4's ``e_D = |D_t - W(D_{t-1})|`` is *undefined* on background — ``inf - inf``
    is NaN — so the sentinel cannot simply be carried into the arithmetic. It is
    replaced here and the affected pixels are hard-zeroed by the confidence mask
    instead, which is the only treatment that keeps the tensor finite without
    inventing a depth for the sky.
    """
    s = session.load_structure(frame_id)
    depth = np.asarray(s.depth, dtype=np.float32)
    finite = np.isfinite(depth)
    return {
        "depth_m": torch.from_numpy(np.where(finite, depth, 0.0))[None, None],
        "depth_finite": torch.from_numpy(finite.astype(np.float32))[None, None],
        "normal": _chw(np.asarray(s.normal, dtype=np.float32)),
        "entity_id": torch.from_numpy(np.ascontiguousarray(s.entity_id))[None, None],
        "motion": _chw(session.load_motion(frame_id)),
    }


def entities_without_previous_transform(session, prev_id: int, curr_id: int) -> list[int]:
    """Instance IDs visible at ``curr_id`` that had **no transform at** ``prev_id``.

    KB scenario **S7**. The exporter writes ``motion = (0, 0)`` for such an
    entity (``frame_structure.md`` §4: "entity spawned / became registered this
    frame: no valid previous correspondence -> 0"), which is bit-identical to
    the field it writes for a perfectly static surface. No exponential of a depth
    or normal error can tell those two apart, so the distinction has to come from
    the manifest, not from the field.

    Derived from the registration snapshots by set difference, and unioned with
    the record's own ``spawned`` flag — the derived form is authoritative
    (KB Decision 29: recompute derived fields, do not trust the asserted value)
    and the flag is a second, independent signal for the same condition.
    """
    prev = {int(e["instance_id"]) for e in session.record(prev_id)["entities"]}
    curr = session.record(curr_id)["entities"]
    fresh = {int(e["instance_id"]) for e in curr} - prev
    fresh |= {int(e["instance_id"]) for e in curr if e.get("spawned")}
    return sorted(fresh)


def spawned_pixel_mask(entity_id: torch.Tensor, fresh_ids) -> torch.Tensor:
    """``(1, 1, H, W)`` float 1.0 where the pixel belongs to a just-registered entity."""
    m = torch.zeros_like(entity_id, dtype=torch.float32)
    for i in fresh_ids:
        m = m + (entity_id == int(i)).float()
    return m.clamp(max=1.0)
