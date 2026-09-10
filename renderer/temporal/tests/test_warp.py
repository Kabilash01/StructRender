"""KB §11 required unit tests "motion warp direction" and "camera motion convention".

§12 names the motion coordinate convention as the project's TOP fragile area —
previous/current direction, NDC vs UV, Y-axis inversion — and its mitigation is
"known translation tests" and a "checkerboard warp test". Both are here, plus the
one thing a synthetic test cannot give: the same comparison run on frames the
engine actually produced, where a sign error would show up as the warp making the
photometric error WORSE.

Nothing here is a smoke test. Every assertion has a do-nothing anchor (the
unwarped difference) or an explicitly-wrong variant on the other side of it.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from renderer.temporal.pairs import consecutive_pairs, load_geometry
from renderer.temporal.warp import (
    EngineGuidedWarp,
    backward_warp,
    base_uv_grid,
    in_bounds,
    resize_motion,
    sampling_grid,
)

H, W = 32, 48


def _checkerboard(h: int = H, w: int = W, c: int = 3, square: int = 4) -> torch.Tensor:
    """High-frequency content. A smooth image would let a wrong grid pass —
    §12's "checkerboard warp test"."""
    ys = (torch.arange(h) // square) % 2
    xs = (torch.arange(w) // square) % 2
    board = (ys[:, None] ^ xs[None, :]).float()
    ramp = torch.linspace(0.0, 1.0, w)[None, :].expand(h, w)
    return torch.stack([board, ramp, 1.0 - board], 0)[None][:, :c]


def _shift_field(dx_px: int, dy_px: int, h: int = H, w: int = W) -> torch.Tensor:
    """Backward motion for content that moved ``(+dx, +dy)`` pixels between t-1 and t.

    Backward means "where was this surface": a surface now at ``x`` was at
    ``x - dx``, so ``motion_u = -dx / W``.
    """
    m = torch.zeros(1, 2, h, w)
    m[:, 0] = -dx_px / w
    m[:, 1] = -dy_px / h
    return m


# --------------------------------------------------------------------------- #
# the convention, in isolation
# --------------------------------------------------------------------------- #
def test_zero_motion_is_exactly_the_identity():
    """The align_corners=False choice, stated as a test rather than a comment.

    ``grid_sample(align_corners=False)`` maps ``g`` to pixel ``(g+1)/2*N - 0.5``;
    the exporter's ``uv = (x+0.5)/N`` inverts it exactly. If either half of that
    pair were wrong the identity would be a resample, not a copy.
    """
    img = _checkerboard()
    out = backward_warp(img, torch.zeros(1, 2, H, W))
    # 1e-5 is float rounding in the interpolation weights, not a convention
    # error: the rejected align_corners=True option misses by 0.48 on this same
    # image (asserted below), five orders of magnitude away.
    assert (out - img).abs().max() < 1e-5


def test_align_corners_true_would_not_be_the_identity():
    """The anchor for the test above: the rejected option measurably differs."""
    img = _checkerboard()
    grid = sampling_grid(torch.zeros(1, 2, H, W))
    wrong = torch.nn.functional.grid_sample(
        img, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    assert (wrong - img).abs().max() > 0.1


@pytest.mark.parametrize("dx,dy", [(3, 0), (0, 2), (5, 4), (-6, -3)])
def test_known_integer_translation_is_reproduced_exactly(dx, dy):
    """§12's "known translation test". Interior only: the border has no history."""
    prev = _checkerboard()
    curr = torch.roll(prev, shifts=(dy, dx), dims=(2, 3))  # content moved by (dx, dy)
    got = backward_warp(curr * 0 + prev, _shift_field(dx, dy))
    pad = max(abs(dx), abs(dy))
    a = got[..., pad:H - pad, pad:W - pad]
    b = curr[..., pad:H - pad, pad:W - pad]
    assert torch.allclose(a, b, atol=1e-5), f"max diff {(a - b).abs().max():.3e}"


def test_y_axis_is_down_and_is_not_flipped():
    """The single most likely convention bug (§12: "Y-axis inversion").

    Both UV and ``grid_sample``'s ``y`` increase downward, so the correct field
    needs no flip. The flipped field is measured on the other side of the
    assertion so this cannot pass by both being equally wrong.
    """
    prev = _checkerboard()
    dy = 5
    curr = torch.roll(prev, shifts=dy, dims=2)
    right = backward_warp(prev, _shift_field(0, dy))
    flipped = backward_warp(prev, _shift_field(0, -dy))
    interior = (slice(None), slice(None), slice(dy, H - dy), slice(None))
    e_right = (right[interior] - curr[interior]).abs().mean()
    e_flip = (flipped[interior] - curr[interior]).abs().mean()
    assert e_right < 1e-6
    assert e_flip > 10 * max(float(e_right), 1e-9)


def test_backward_not_forward():
    """A forward field (negated) must be measurably worse, not merely different."""
    prev = _checkerboard()
    dx = 4
    curr = torch.roll(prev, shifts=dx, dims=3)
    back = backward_warp(prev, _shift_field(dx, 0))
    fwd = backward_warp(prev, -_shift_field(dx, 0))
    interior = (slice(None), slice(None), slice(None), slice(dx, W - dx))
    assert (back[interior] - curr[interior]).abs().mean() < \
           (fwd[interior] - curr[interior]).abs().mean() / 10


# --------------------------------------------------------------------------- #
# resolution independence — the property that lets one field drive every stage
# --------------------------------------------------------------------------- #
def test_uv_is_resolution_independent_so_resize_does_not_scale_values():
    m = _shift_field(4, 2)
    small = resize_motion(m, (H // 4, W // 4))
    assert small.shape[-2:] == (H // 4, W // 4)
    # a constant field must survive resampling with its VALUES unchanged
    assert torch.allclose(small, m[:, :, : H // 4, : W // 4], atol=1e-6)


def test_the_same_field_translates_by_the_same_fraction_at_two_resolutions():
    """Half a pixel at 48 wide and half a pixel at 12 wide are different distances;
    a quarter of the width is a quarter of the width at both."""
    dx_frac = 0.25
    for w, h in ((W, H), (W // 4, H // 4)):
        prev = _checkerboard(h, w, square=max(1, h // 8))
        m = torch.zeros(1, 2, h, w)
        m[:, 0] = -dx_frac
        got = backward_warp(prev, m)
        expect = torch.roll(prev, shifts=int(round(dx_frac * w)), dims=3)
        k = int(round(dx_frac * w))
        assert torch.allclose(got[..., k:], expect[..., k:], atol=1e-5)


def test_warp_runs_at_feature_resolution_and_channel_count():
    feat = torch.randn(1, 240, 72, 128)
    m = resize_motion(_shift_field(3, 1, 288, 512), (72, 128))
    assert backward_warp(feat, m).shape == feat.shape


def test_backward_warp_refuses_a_resolution_mismatch():
    with pytest.raises(ValueError, match="resize_motion"):
        backward_warp(torch.randn(1, 4, 16, 16), torch.zeros(1, 2, 8, 8))


# --------------------------------------------------------------------------- #
# out of bounds
# --------------------------------------------------------------------------- #
def test_in_bounds_marks_exactly_the_samples_that_leave_the_frame():
    m = _shift_field(10, 0)          # every surface was 10 px to the left at t-1
    ok = in_bounds(m)[0, 0]
    assert ok[:, 10:].all(), "interior samples must be in bounds"
    assert not ok[:, :9].any(), "the left 9 columns sampled off-frame"


def test_out_of_bounds_samples_fetch_zero_not_a_smeared_border():
    prev = torch.ones(1, 1, H, W)
    got = backward_warp(prev, _shift_field(10, 0))
    assert got[..., :9].abs().max() < 1e-6
    assert torch.allclose(got[..., 10:], torch.ones_like(got[..., 10:]), atol=1e-6)


def test_base_uv_grid_is_pixel_centred():
    uv = base_uv_grid(4, 8)
    assert torch.allclose(uv[0, 0, 0], torch.tensor([0.5 / 8, 0.5 / 4]))
    assert torch.allclose(uv[0, 3, 7], torch.tensor([7.5 / 8, 3.5 / 4]))


def test_module_buffer_is_not_persistent():
    """It is derived from the fixed shape, not learned; a persistent buffer would
    ride along in every checkpoint (the call KB Decision 45 made for rel-pos)."""
    assert EngineGuidedWarp(H, W).state_dict() == {}


def test_module_matches_the_functional_form():
    m = _shift_field(3, 2)
    img = _checkerboard()
    assert torch.equal(EngineGuidedWarp(H, W)(img, m), backward_warp(img, m))


# --------------------------------------------------------------------------- #
# ONNX-safety (§12) — fixed shapes, no data-dependent control flow
# --------------------------------------------------------------------------- #
def test_exports_to_opset_17_with_static_shapes(tmp_path):
    onnx = pytest.importorskip("onnx")
    path = tmp_path / "warp.onnx"
    torch.onnx.export(EngineGuidedWarp(H, W).eval(), (_checkerboard(), _shift_field(2, 1)),
                      str(path), input_names=["prev", "motion"], output_names=["warped"],
                      opset_version=17, dynamo=False, do_constant_folding=True)
    g = onnx.load(str(path)).graph
    ops = {n.op_type for n in g.node}
    assert "GridSample" in ops
    assert not ({"If", "Loop", "NonZero", "Where"} & ops), f"data-dependent control flow: {ops}"
    for vi in list(g.input) + list(g.output):
        for d in vi.type.tensor_type.shape.dim:
            assert d.HasField("dim_value") and d.dim_value > 0


# --------------------------------------------------------------------------- #
# the real thing — KB Scenario 4 / §11 "motion -> warp" integration
# --------------------------------------------------------------------------- #
#: A pair qualifies when it has enough motion for the question to mean anything
#: and enough overlap for history to exist. Both bounds are properties of the
#: MEASUREMENT, not of a result: below ~2 px the displacement is comparable to
#: bilinear resampling's own cost, and the ~130 px camera snap-back frames retain
#: barely half the previous frame, so "did the warp align it" is not what is
#: being asked there.
MIN_MEDIAN_PX, MAX_MEDIAN_PX, MIN_KEEP_FRAC = 2.0, 60.0, 0.75

#: Pre-registered against the two anchors this test measures alongside it:
#: doing nothing scores exactly 1.00x by construction, and a sign-flipped field
#: scores BELOW 1.0. A correct warp of a >= 2 px displacement has to land clearly
#: on the far side of the do-nothing anchor, on every qualifying pair.
MIN_RATIO_EVERY_PAIR = 1.0
MIN_RATIO_MEDIAN = 1.5


def _pair_stats(session, pair):
    c = load_geometry(session, pair.curr_id)
    m = c["motion"]
    keep = in_bounds(m) * c["depth_finite"]
    mag = torch.sqrt((m[:, :1] * m.shape[-1]) ** 2 + (m[:, 1:] * m.shape[-2]) ** 2)
    sel = keep > 0.5
    if sel.sum() < 1000:
        return None
    median_px = float(mag[sel].median())
    keep_frac = float(keep.mean())

    def rgb(fid):
        a = session.load_native_rgb(fid).astype(np.float32) / 255.0
        return torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1)[None]

    it, ip = rgb(pair.curr_id), rgb(pair.prev_id)

    def err(x):
        return float(((it - x).abs().mean(1, keepdim=True) * keep).sum() / keep.sum())

    flipped = -m
    return {
        "frame": pair.curr_id, "median_px": median_px, "keep_frac": keep_frac,
        "unwarped": err(ip), "warped": err(backward_warp(ip, m)),
        "sign_flipped": err(backward_warp(ip, flipped)),
        "y_flipped": err(backward_warp(ip, torch.cat([m[:, :1], -m[:, 1:]], 1))),
        "median_v_px": float((m[:, 1:] * m.shape[-2]).abs()[sel].median()),
    }


@pytest.fixture(scope="module")
def measured_pairs(session):
    ids = session.frame_ids()
    rows = []
    for pair in consecutive_pairs(session, ids):
        st = _pair_stats(session, pair)
        if st and MIN_MEDIAN_PX <= st["median_px"] <= MAX_MEDIAN_PX \
                and st["keep_frac"] >= MIN_KEEP_FRAC:
            rows.append(st)
    if len(rows) < 10:
        pytest.skip(f"only {len(rows)} qualifying pairs in this session")
    return rows


def test_warping_native_rgb_beats_not_warping_on_every_qualifying_pair(measured_pairs):
    """KB Scenario 4, on real frames. The do-nothing anchor is the unwarped
    difference over the SAME pixels — not a different pixel set, which is how a
    warp that discards half the frame can look good."""
    bad = [r for r in measured_pairs
           if r["unwarped"] / r["warped"] <= MIN_RATIO_EVERY_PAIR]
    assert not bad, f"{len(bad)} pairs where the warp did not help: {bad[:3]}"


def test_the_median_error_reduction_is_substantial(measured_pairs):
    ratios = sorted(r["unwarped"] / r["warped"] for r in measured_pairs)
    median = ratios[len(ratios) // 2]
    assert median >= MIN_RATIO_MEDIAN, f"median reduction {median:.2f}x over {len(ratios)} pairs"


def test_a_sign_flipped_field_is_worse_than_doing_nothing(measured_pairs):
    """The anchor that makes the two tests above mean something. A metric that
    rewards any warp would also reward this one."""
    worse = [r for r in measured_pairs if r["sign_flipped"] > r["unwarped"]]
    assert len(worse) >= 0.9 * len(measured_pairs), (
        f"a sign-flipped field beat doing nothing on "
        f"{len(measured_pairs) - len(worse)}/{len(measured_pairs)} pairs")


#: A Y-flip test needs vertical motion to flip. This session's camera pans mostly
#: horizontally, and on pairs whose vertical component is a sixth of a pixel the
#: flipped field is the same field to within resampling noise — such a pair
#: cannot discriminate and is excluded from the question, not from the report.
MIN_VERTICAL_PX = 0.5


def test_a_y_flipped_field_is_worse_than_the_correct_one(measured_pairs):
    """§12's Y-axis inversion, on engine data rather than a synthetic shift."""
    rows = [r for r in measured_pairs if r["median_v_px"] >= MIN_VERTICAL_PX]
    if len(rows) < 10:
        pytest.skip(f"only {len(rows)} pairs carry >= {MIN_VERTICAL_PX} px of vertical motion")
    worse = [r for r in rows if r["y_flipped"] > r["warped"]]
    assert len(worse) == len(rows), (
        f"a Y-flipped field matched or beat the correct one on "
        f"{len(rows) - len(worse)}/{len(rows)} pairs with real vertical motion")
