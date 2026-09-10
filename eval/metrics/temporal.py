"""Temporal error — KB §11 Core Metrics / Temporal: "motion-warp error".

    TE = mean |I_t - W(I_{t-1}, V_t)|   over the pixels the confidence mask keeps

across consecutive validation pairs. This is the quantity KB §10 Milestone 6's
exit criterion is stated in ("lower temporal error than frame-independent
StructFormer"), and it is measured here **before Model D exists** so the margin
can be pre-registered rather than chosen after seeing the result — the same
discipline KB Decision 46 established for M5's margin.

**Three things this module refuses to do.**

1. **It never reports TE alone.** A blurrier renderer has trivially lower
   temporal error: blur removes exactly the high-frequency content that a
   sub-pixel misalignment turns into an error. That is the same confound KB
   Decision 53 records for RCS, and Model C is *already* below M4's sharpness
   floor, so on this ladder it is not hypothetical. Every source therefore
   carries a sharpness figure computed on the same frames, and
   :func:`summarise` puts them in one record.
2. **The mask never depends on the source.** ``M_t`` is built from the engine's
   own depth, normals, motion and entity registration — never from the image
   being scored. A mask derived per-source would let a model be measured on a
   pixel set of its own choosing.
3. **It never mixes tick deltas.** This session's validation pairs are 27 %
   ``dt == 1`` and 73 % ``dt > 1`` (an 8-tick, ~8x-displacement pair recorded at
   a reduced rate during the long reappearance gaps), while training is 100 %
   ``dt == 1``. That is a real distribution shift, so the two are summarised
   separately and the headline gate is the ``dt == 1`` set; ``dt > 1`` is an
   out-of-distribution robustness figure and is labelled as one.

The **do-nothing anchor** for TE is the *unwarped* difference
``|I_t - I_{t-1}|`` over the identical pixel set (KB Decision 36's rule). Without
it, a TE of 0.004 says nothing: it could be a well-aligned warp or a scene that
barely moved.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch

from eval.metrics.image_quality import gradient_energy
from renderer.temporal.confidence import confidence_mask
from renderer.temporal.pairs import (
    entities_without_previous_transform,
    load_geometry,
    spawned_pixel_mask,
)
from renderer.temporal.warp import backward_warp

#: The mask is continuous in [0, 1]; "keeps" is read at its calibrated crossing.
#: 0.5 is not an extra free parameter — it is the point alpha and beta were
#: calibrated to land on (see eval/reports/m6_mask_calibration.json), which is
#: the whole reason the calibration was done before this metric was written.
KEEP_THRESHOLD = 0.5

#: A pair with almost nothing left after masking is not a measurement. Recorded
#: by omission with a count, never imputed.
MIN_KEPT_PIXELS = 5000

#: TE is stratified by displacement, because on this session's validation split
#: it is otherwise easy to read a number that measures nothing. The held-out
#: sequences are the two 8-second reappearance rungs, which PARK THE CAMERA for
#: most of their length — KB Decision 35 already recorded that property, for the
#: qualitative strips, and it lands on this metric too. Measured: 43 of the 44
#: dt==1 validation pairs displace by under half a pixel, where no warp can beat
#: the identity because bilinear resampling costs more than the alignment buys.
#: Without this breakdown, "TE" on that split is |I_t - I_{t-1}| wearing a
#: different name.
MOTION_BUCKETS = (
    ("lt_0.5px", 0.0, 0.5),
    ("0.5_to_2px", 0.5, 2.0),
    ("2_to_10px", 2.0, 10.0),
    ("ge_10px", 10.0, float("inf")),
)


@dataclass
class PairTE:
    prev_id: int
    curr_id: int
    tick_delta: int
    kept_fraction: float
    te: float                    # masked |I_t - W(I_{t-1})|
    te_weighted: float           # same, weighted by the continuous mask
    unwarped: float              # do-nothing anchor: masked |I_t - I_{t-1}|
    grad_energy: float           # sharpness of I_t, same frame
    grad_energy_target: float    # sharpness of the teacher target on that frame
    median_motion_px: float      # displacement over the kept pixels — see below


def engine_confidence(session, pair) -> dict:
    """``M_t`` for one pair, from engine buffers only.

    Deliberately takes the *session*, not any image: nothing a renderer produces
    may influence which pixels it is scored on.
    """
    c = load_geometry(session, pair.curr_id)
    p = load_geometry(session, pair.prev_id)
    fresh = entities_without_previous_transform(session, pair.prev_id, pair.curr_id)
    npt = spawned_pixel_mask(c["entity_id"], fresh) if fresh else None
    out = confidence_mask(
        depth_t=c["depth_m"], depth_prev=p["depth_m"],
        normal_t=c["normal"], normal_prev=p["normal"], motion=c["motion"],
        depth_finite_t=c["depth_finite"], depth_finite_prev=p["depth_finite"],
        motion_valid=session.motion_valid(pair.curr_id), no_prev_transform=npt)
    out["motion"] = c["motion"]
    return out


def _masked_mean(err: torch.Tensor, weight: torch.Tensor) -> float:
    return float((err * weight).sum() / weight.sum().clamp_min(1.0))


def evaluate_source(session, image_for_frame, pairs, target_for_frame=None,
                    masks: dict | None = None) -> list[PairTE]:
    """TE for every pair, for one source of images.

    ``image_for_frame(frame_id) -> (H, W, 3) float in [0, 1]`` is the only thing
    that varies between the native frames, the teacher targets and a model's
    predictions, so every source is scored through an identical path — the same
    construction ``eval/metrics/rcs.py`` uses.

    ``masks`` lets the caller compute each pair's confidence mask ONCE and share
    it across sources. Sharing is not an optimisation here, it is a correctness
    property: it makes it structurally impossible for two sources to be scored on
    different pixels.
    """
    def to_t(a: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).permute(2, 0, 1)[None]

    rows: list[PairTE] = []
    for pair in pairs:
        m = masks[pair.curr_id] if masks is not None else engine_confidence(session, pair)
        keep = (m["mask"] > KEEP_THRESHOLD).float()
        if float(keep.sum()) < MIN_KEPT_PIXELS:
            continue
        it, ip = to_t(image_for_frame(pair.curr_id)), to_t(image_for_frame(pair.prev_id))
        warped = backward_warp(ip, m["motion"])
        err = (it - warped).abs().mean(1, keepdim=True)
        raw = (it - ip).abs().mean(1, keepdim=True)
        tgt = to_t(target_for_frame(pair.curr_id)) if target_for_frame is not None else None
        mo = m["motion"]
        mag = torch.sqrt((mo[:, :1] * mo.shape[-1]) ** 2 + (mo[:, 1:] * mo.shape[-2]) ** 2)
        rows.append(PairTE(
            prev_id=pair.prev_id, curr_id=pair.curr_id, tick_delta=pair.tick_delta,
            kept_fraction=float(keep.mean()),
            te=_masked_mean(err, keep),
            te_weighted=_masked_mean(err, m["mask"]),
            unwarped=_masked_mean(raw, keep),
            grad_energy=float(gradient_energy(it)[0]),
            grad_energy_target=float(gradient_energy(tgt)[0]) if tgt is not None else float("nan"),
            median_motion_px=float(mag[keep > 0.5].median()),
        ))
    return rows


def _stats(vals) -> dict:
    a = np.asarray(vals, dtype=np.float64)
    return {"n": int(a.size),
            "mean": float(a.mean()) if a.size else float("nan"),
            "sd": float(a.std(ddof=1)) if a.size > 1 else None}


def summarise(rows: list[PairTE]) -> dict:
    """Per tick-delta group, always with the do-nothing anchor and sharpness.

    ``sharpness_ratio`` is ``grad_energy(source) / grad_energy(teacher target)``
    on the same frames — M4's quantity (KB Decision 41), so the number is
    comparable to the ladder's committed sharpness figures and to its 0.90 floor.
    """
    def group(rs: list[PairTE]) -> dict:
        if not rs:
            return {"n_pairs": 0}
        ratios = [r.grad_energy / r.grad_energy_target for r in rs
                  if np.isfinite(r.grad_energy_target) and r.grad_energy_target > 0]
        return {
            "n_pairs": len(rs),
            "temporal_error": _stats([r.te for r in rs]),
            "temporal_error_mask_weighted": _stats([r.te_weighted for r in rs]),
            "do_nothing_anchor_unwarped": _stats([r.unwarped for r in rs]),
            "warp_gain_vs_unwarped": _stats([r.unwarped / r.te for r in rs if r.te > 0]),
            "kept_fraction": _stats([r.kept_fraction for r in rs]),
            "sharpness_ratio_vs_target": _stats(ratios) if ratios else {"n": 0},
            "gradient_energy_abs": _stats([r.grad_energy for r in rs]),
            "median_motion_px": _stats([r.median_motion_px for r in rs]),
            "pairs_below_half_a_pixel": sum(1 for r in rs if r.median_motion_px < 0.5),
            "by_motion_bucket": {
                label: _stats([r.te for r in rs if lo <= r.median_motion_px < hi])
                for label, lo, hi in MOTION_BUCKETS
            },
        }

    return {
        "dt_eq_1": group([r for r in rows if r.tick_delta == 1]),
        "dt_gt_1": group([r for r in rows if r.tick_delta > 1]),
        "gating_note": ("dt_eq_1 is the HEADLINE gate: training is 100% dt==1, so it is the "
                        "in-distribution set. dt_gt_1 is an out-of-distribution robustness "
                        "figure (~8x the displacement) and must not be averaged into it."),
        "per_pair": [asdict(r) for r in rows],
    }
