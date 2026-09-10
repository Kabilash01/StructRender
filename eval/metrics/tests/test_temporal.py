"""Temporal-error metric behaviour — KB §11 Temporal. Pure CPU, no session needed.

The load-bearing test here is :func:`test_blurring_a_source_lowers_its_temporal_error`,
which demonstrates the confound the module's caveat asserts. Without it the
caveat is a comment; with it, the number is on record.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from eval.metrics.temporal import (
    KEEP_THRESHOLD,
    MIN_KEPT_PIXELS,
    PairTE,
    evaluate_source,
    summarise,
)
from renderer.temporal.pairs import FramePair

H, W = 96, 160


def _mask_bundle(motion: torch.Tensor, keep_all: bool = True) -> dict:
    mask = torch.ones(1, 1, H, W) if keep_all else torch.zeros(1, 1, H, W)
    return {"mask": mask, "motion": motion}


def _shift_field(dx_px: float) -> torch.Tensor:
    m = torch.zeros(1, 2, H, W)
    m[:, 0] = -dx_px / W
    return m


def _pair() -> FramePair:
    return FramePair(prev_id=0, curr_id=1, tick_delta=1)


def _frames(prev: np.ndarray, curr: np.ndarray):
    def load(fid: int) -> np.ndarray:
        return curr if fid == 1 else prev
    return load


def _noise(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).random((H, W, 3)).astype(np.float32)


def test_a_perfectly_warped_pair_has_almost_zero_temporal_error():
    prev = _noise(0)
    dx = 5
    curr = np.roll(prev, dx, axis=1)
    # np.roll wraps, so the leftmost dx columns of `curr` came from the right
    # edge and have no in-frame history. Masking them is what the real mask's
    # out-of-bounds hard zero does; leaving them in would put a 21x-instead-of-
    # infinite floor under this assertion and hide a genuine misalignment.
    mask = torch.ones(1, 1, H, W)
    mask[..., :dx] = 0.0
    rows = evaluate_source(None, _frames(prev, curr), [_pair()],
                           masks={1: {"mask": mask, "motion": _shift_field(dx)}})
    assert rows[0].te < 1e-5
    assert rows[0].unwarped > 100 * rows[0].te, "the do-nothing anchor must be far worse"


def test_a_wrong_motion_field_raises_the_temporal_error():
    prev = _noise(1)
    curr = np.roll(prev, 5, axis=1)
    p = _pair()
    right = evaluate_source(None, _frames(prev, curr), [p],
                            masks={1: _mask_bundle(_shift_field(5))})[0]
    wrong = evaluate_source(None, _frames(prev, curr), [p],
                            masks={1: _mask_bundle(_shift_field(-5))})[0]
    assert wrong.te > 10 * right.te


def test_only_the_masked_pixels_contribute():
    prev = _noise(2)
    curr = prev.copy()
    curr[:, :W // 2] = _noise(3)[:, :W // 2]      # corrupt the left half only
    mask = torch.ones(1, 1, H, W)
    mask[..., :W // 2] = 0.0                       # ... and mask it out
    rows = evaluate_source(None, _frames(prev, curr), [_pair()],
                           masks={1: {"mask": mask, "motion": torch.zeros(1, 2, H, W)}})
    assert rows[0].te < 1e-6


def test_a_pair_with_almost_no_kept_pixels_is_omitted_not_imputed():
    mask = torch.zeros(1, 1, H, W)
    mask[..., :4, :4] = 1.0
    assert int(mask.sum()) < MIN_KEPT_PIXELS
    rows = evaluate_source(None, _frames(_noise(4), _noise(5)), [_pair()],
                           masks={1: {"mask": mask, "motion": torch.zeros(1, 2, H, W)}})
    assert rows == []


def test_the_keep_set_is_read_at_the_calibrated_crossing():
    """0.5 is the mask's own half point, not a second free parameter."""
    mask = torch.full((1, 1, H, W), KEEP_THRESHOLD - 0.01)
    rows = evaluate_source(None, _frames(_noise(6), _noise(7)), [_pair()],
                           masks={1: {"mask": mask, "motion": torch.zeros(1, 2, H, W)}})
    assert rows == [], "a mask entirely below the crossing must keep nothing"


# --------------------------------------------------------------------------- #
# the confound, measured rather than asserted
# --------------------------------------------------------------------------- #
def test_blurring_a_source_lowers_its_temporal_error():
    """Why TE may never be reported without a sharpness figure beside it.

    Blur removes exactly the high-frequency content that a sub-pixel
    misalignment turns into error, so a strictly worse renderer posts a strictly
    better TE. This is KB Decision 53's RCS confound on a new metric, and Model C
    is already below M4's 0.90 sharpness floor — so on this ladder the rung most
    likely to post a low TE is the one already known to be too soft.
    """
    from eval.metrics.image_quality import gradient_energy

    prev = _noise(8)
    curr = np.roll(prev, 3, axis=1)                     # 3 px of real motion
    motion = _shift_field(2.5)                          # ... reported as 2.5: 0.5 px off

    def blurred(img: np.ndarray, sigma_px: int) -> np.ndarray:
        t = torch.from_numpy(img).permute(2, 0, 1)[None]
        k = torch.ones(3, 1, sigma_px, sigma_px) / (sigma_px * sigma_px)
        t = F.conv2d(F.pad(t, (sigma_px // 2,) * 4, mode="replicate"), k, groups=3)
        return t[0, :, :H, :W].permute(1, 2, 0).numpy()

    sharp = evaluate_source(None, _frames(prev, curr), [_pair()],
                            masks={1: _mask_bundle(motion)})[0]
    soft_prev, soft_curr = blurred(prev, 5), blurred(curr, 5)
    soft = evaluate_source(None, _frames(soft_prev, soft_curr), [_pair()],
                           masks={1: _mask_bundle(motion)})[0]

    assert soft.te < sharp.te, "precondition of the whole caveat"
    assert soft.te < 0.5 * sharp.te, (
        f"blur bought {sharp.te / soft.te:.1f}x on TE alone (sharp {sharp.te:.5f} "
        f"-> soft {soft.te:.5f})")
    e_sharp = float(gradient_energy(torch.from_numpy(curr).permute(2, 0, 1)[None])[0])
    e_soft = float(gradient_energy(torch.from_numpy(soft_curr).permute(2, 0, 1)[None])[0])
    assert e_soft < e_sharp, "and the sharpness column is what exposes it"


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def _row(dtick: int, te: float, motion_px: float = 1.0) -> PairTE:
    return PairTE(prev_id=0, curr_id=1, tick_delta=dtick, kept_fraction=1.0, te=te,
                  te_weighted=te, unwarped=2 * te, grad_energy=1.0,
                  grad_energy_target=2.0, median_motion_px=motion_px)


def test_tick_deltas_are_never_averaged_together():
    s = summarise([_row(1, 0.001), _row(1, 0.003), _row(8, 0.010)])
    assert s["dt_eq_1"]["temporal_error"]["mean"] == pytest.approx(0.002)
    assert s["dt_eq_1"]["n_pairs"] == 2
    assert s["dt_gt_1"]["temporal_error"]["mean"] == pytest.approx(0.010)
    assert s["dt_gt_1"]["n_pairs"] == 1


def test_every_group_carries_its_do_nothing_anchor_and_a_sharpness_ratio():
    g = summarise([_row(1, 0.001), _row(1, 0.002)])["dt_eq_1"]
    assert g["do_nothing_anchor_unwarped"]["mean"] == pytest.approx(0.003)
    assert g["sharpness_ratio_vs_target"]["mean"] == pytest.approx(0.5)
    assert g["warp_gain_vs_unwarped"]["mean"] == pytest.approx(2.0)


def test_sub_pixel_pairs_are_counted_and_bucketed():
    """The measurability warning has to be derivable from the numbers, not only
    from prose: a reader must be able to see how many pairs could not move."""
    g = summarise([_row(1, 0.001, 0.2), _row(1, 0.002, 0.3), _row(1, 0.004, 5.0)])["dt_eq_1"]
    assert g["pairs_below_half_a_pixel"] == 2
    assert g["by_motion_bucket"]["lt_0.5px"]["n"] == 2
    assert g["by_motion_bucket"]["2_to_10px"]["n"] == 1
    assert g["by_motion_bucket"]["ge_10px"]["n"] == 0


def test_an_empty_group_reports_zero_pairs_rather_than_a_number():
    s = summarise([_row(1, 0.001)])
    assert s["dt_gt_1"] == {"n_pairs": 0}
