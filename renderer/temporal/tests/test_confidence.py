"""KB §11's required "confidence mask" unit test, one case per hard zero.

§4's ``M_t = exp(-alpha e_D - beta e_N)`` cannot express four of the five cases
below, and gets two of them exactly backwards: S7 and background both carry
``motion = (0, 0)``, which the exponential reads as *perfect* confidence in
history that does not exist. Each hard zero therefore gets a test that would
FAIL if the term were removed, not a test that merely exercises it.

Pure CPU, synthetic tensors — this file runs in full on a clean checkout. The
data-backed distribution work is the calibration artifact, not a test.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from renderer.temporal.confidence import (
    ALPHA,
    BETA,
    E_D_HALF_METRES,
    E_N_HALF,
    ConfidenceMask,
    confidence_mask,
    depth_error,
    hard_valid,
    normal_error,
)
from renderer.temporal.pairs import entities_without_previous_transform, spawned_pixel_mask

H, W = 16, 24


def _flat_scene(depth: float = 5.0):
    """A static, fully-visible, perfectly-tracked frame pair. The control."""
    d = torch.full((1, 1, H, W), depth)
    fin = torch.ones(1, 1, H, W)
    n = torch.zeros(1, 3, H, W)
    n[:, 2] = 1.0
    return dict(depth_t=d, depth_prev=d.clone(), normal_t=n, normal_prev=n.clone(),
                motion=torch.zeros(1, 2, H, W), depth_finite_t=fin,
                depth_finite_prev=fin.clone())


def test_perfectly_tracked_static_content_gets_confidence_one():
    """The anchor every other test is measured against."""
    m = confidence_mask(**_flat_scene())["mask"]
    assert float(m.min()) > 0.999


# --------------------------------------------------------------------------- #
# hard zero 1 — motion_valid == False (frame 0)
# --------------------------------------------------------------------------- #
def test_frame_zero_is_entirely_zero_confidence():
    kw = _flat_scene()
    assert float(confidence_mask(**kw, motion_valid=False)["mask"].max()) == 0.0
    # ... and would otherwise be 1.0: the exponential alone cannot see this.
    assert float(confidence_mask(**kw, motion_valid=True)["mask"].min()) > 0.999


def test_motion_valid_accepts_a_per_batch_tensor():
    kw = _flat_scene()
    kw = {k: (v.repeat(2, 1, 1, 1) if torch.is_tensor(v) else v) for k, v in kw.items()}
    m = confidence_mask(**kw, motion_valid=torch.tensor([True, False]))["mask"]
    assert float(m[0].min()) > 0.999 and float(m[1].max()) == 0.0


# --------------------------------------------------------------------------- #
# hard zero 2 — the sample left the frame
# --------------------------------------------------------------------------- #
def test_samples_that_land_outside_the_frame_are_zeroed():
    kw = _flat_scene()
    kw["motion"] = torch.zeros(1, 2, H, W)
    kw["motion"][:, 0] = -6.0 / W          # every surface was 6 px left at t-1
    m = confidence_mask(**kw)["mask"]
    assert float(m[..., :5].max()) == 0.0, "off-frame samples kept confidence"
    assert float(m[..., 7:].min()) > 0.999, "in-frame samples were wrongly zeroed"


def test_each_term_alone_zeroes_the_off_frame_region():
    """The bounds term and the previous-coverage term OVERLAP, and this says so.

    Under ``padding_mode="zeros"`` an off-frame sample warps the previous
    validity to 0, so the coverage term already catches it — a mutation test that
    deletes ``in_bounds`` from ``hard_valid`` therefore changes nothing, and a
    test asserting only the combined output would be vacuous about it. Each term
    is asserted separately instead. ``in_bounds`` earns its place by being
    geometric rather than a side effect of the padding mode: with ``border``
    padding the coverage term alone would keep off-frame history.
    """
    from renderer.temporal.warp import PADDING_MODE, backward_warp, in_bounds

    assert PADDING_MODE == "zeros", "this redundancy holds only for zeros-padding"
    motion = torch.zeros(1, 2, H, W)
    motion[:, 0] = -6.0 / W
    fin_prev = torch.ones(1, 1, H, W)
    assert float(in_bounds(motion)[..., :5].max()) == 0.0
    assert float(backward_warp(fin_prev, motion)[..., :5].max()) == 0.0


def test_without_the_bounds_term_the_exponential_would_keep_them():
    """Why the term is needed: zero-padding makes e_D a number, not evidence.

    Out of frame, ``W(D_{t-1})`` is 0 by zero-padding, so ``e_D = |5 - 0| = 5``
    metres — large, but finite, and an exponential of a large number is still a
    number. With alpha calibrated to a 0.175 m half point it happens to be tiny
    here; on a scene 40x shallower it would not be. The hard zero does not depend
    on the scene's depth scale.
    """
    kw = _flat_scene()
    kw["motion"] = torch.zeros(1, 2, H, W)
    kw["motion"][:, 0] = -6.0 / W
    e_d = depth_error(kw["depth_t"], kw["depth_prev"], kw["motion"])
    assert float(e_d[..., 0].max()) == pytest.approx(5.0)
    assert float(torch.exp(-ALPHA * e_d[..., 0]).max()) > 0.0  # finite, not a zero


# --------------------------------------------------------------------------- #
# hard zero 3 — S7, an entity with no t-1 transform
# --------------------------------------------------------------------------- #
def test_a_just_registered_entity_is_zeroed_although_its_error_is_zero():
    """THE case the exponential gets backwards.

    The exporter writes ``motion = (0, 0)`` for an entity that has no ``t-1``
    transform — bit-identical to the field it writes for a perfectly static
    surface. Both errors are then 0 and the exponential returns 1.0: maximum
    confidence in history that does not exist.
    """
    kw = _flat_scene()
    fresh = torch.zeros(1, 1, H, W)
    fresh[..., 4:9, 5:11] = 1.0
    out_soft = confidence_mask(**kw)["soft"]
    assert float(out_soft[..., 4:9, 5:11].min()) > 0.999, (
        "precondition: the exponential alone is maximally confident here")
    m = confidence_mask(**kw, no_prev_transform=fresh)["mask"]
    assert float(m[..., 4:9, 5:11].max()) == 0.0
    assert float(m[..., 0, 0]) > 0.999, "the rest of the frame must be untouched"


def test_s7_pixels_are_found_from_the_manifest_not_from_the_motion_field():
    """The set difference of two registration snapshots, per KB Decision 29's
    rule that a derived field is recomputed rather than trusted."""

    class FakeSession:
        def __init__(self):
            self._r = {
                0: {"entities": [{"instance_id": 1, "spawned": True}]},
                1: {"entities": [{"instance_id": 1, "spawned": False},
                                 {"instance_id": 7, "spawned": True}]},
                2: {"entities": [{"instance_id": 1, "spawned": False},
                                 {"instance_id": 7, "spawned": False}]},
            }

        def record(self, f):
            return self._r[f]

    s = FakeSession()
    assert entities_without_previous_transform(s, 0, 1) == [7]
    assert entities_without_previous_transform(s, 1, 2) == []

    eid = torch.zeros(1, 1, 4, 4, dtype=torch.int32)
    eid[..., :2, :] = 7
    assert float(spawned_pixel_mask(eid, [7]).sum()) == 8.0


def test_an_entity_that_merely_became_visible_again_is_not_treated_as_new():
    """Visibility and registration are decoupled (``frame_structure.md`` §5): an
    entity leaving view emits ``hidden``, never ``destroy``, and keeps its
    transform. Zeroing it on re-entry would throw away history it really has."""

    class FakeSession:
        def record(self, f):
            return {"entities": [{"instance_id": 3, "spawned": False,
                                  "visible": f % 2 == 0}]}

    assert entities_without_previous_transform(FakeSession(), 5, 6) == []


# --------------------------------------------------------------------------- #
# hard zeros 4 and 5 — background, on each side of the subtraction
# --------------------------------------------------------------------------- #
def test_background_at_t_is_zeroed():
    kw = _flat_scene()
    kw["depth_finite_t"] = torch.ones(1, 1, H, W)
    kw["depth_finite_t"][..., :3, :] = 0.0
    kw["depth_t"] = kw["depth_t"] * kw["depth_finite_t"]      # sentinel -> 0
    m = confidence_mask(**kw)["mask"]
    assert float(m[..., :3, :].max()) == 0.0
    assert float(m[..., 3:, :].min()) > 0.999


def test_a_sample_whose_source_was_background_is_zeroed():
    """The fifth hard zero, which §4's three lines do not contain.

    'Surface now, sky then' is a textbook disocclusion, and ``e_D`` there is
    ``|D_t - inf|`` — undefined, not large. Zeroing only the current-frame side
    of the subtraction leaves it unhandled.
    """
    kw = _flat_scene()
    kw["depth_finite_prev"] = torch.ones(1, 1, H, W)
    kw["depth_finite_prev"][..., :4, :] = 0.0
    kw["depth_prev"] = kw["depth_prev"] * kw["depth_finite_prev"]
    m = confidence_mask(**kw)["mask"]
    assert float(m[..., :4, :].max()) == 0.0
    assert float(m[..., 5:, :].min()) > 0.999


def test_a_partially_background_bilinear_tap_is_zeroed_too():
    """A half-pixel offset mixes sky into the tap. That sample is not history."""
    kw = _flat_scene()
    kw["depth_finite_prev"] = torch.ones(1, 1, H, W)
    kw["depth_finite_prev"][..., :4, :] = 0.0
    kw["depth_prev"] = kw["depth_prev"] * kw["depth_finite_prev"]
    kw["motion"] = torch.zeros(1, 2, H, W)
    kw["motion"][:, 1] = -0.5 / H                       # half a pixel down
    v = hard_valid(kw["motion"], kw["depth_finite_t"], kw["depth_finite_prev"], True)
    assert float(v[..., 4, :].max()) == 0.0, "the row straddling the sky edge survived"


def test_the_mask_is_finite_everywhere_even_with_an_inf_free_sentinel():
    """No NaN may reach the exponential. ``inf - inf`` is NaN, and NaN times a
    hard zero is still NaN — which is why the errors are zeroed BEFORE exp()."""
    kw = _flat_scene()
    kw["depth_t"] = kw["depth_t"].clone()
    kw["depth_t"][..., :2, :] = 0.0
    kw["depth_finite_t"] = torch.ones(1, 1, H, W)
    kw["depth_finite_t"][..., :2, :] = 0.0
    out = confidence_mask(**kw)
    assert torch.isfinite(out["mask"]).all()


# --------------------------------------------------------------------------- #
# the exponential itself
# --------------------------------------------------------------------------- #
def test_the_depth_term_crosses_one_half_at_the_calibrated_depth_error():
    kw = _flat_scene()
    kw["depth_prev"] = kw["depth_prev"] + E_D_HALF_METRES
    assert float(confidence_mask(**kw)["mask"].mean()) == pytest.approx(0.5, abs=1e-4)


def test_the_normal_term_crosses_one_half_at_a_thirty_degree_turn():
    kw = _flat_scene()
    theta = np.radians(30.0)
    n = torch.zeros(1, 3, H, W)
    n[:, 1], n[:, 2] = float(np.sin(theta)), float(np.cos(theta))
    kw["normal_prev"] = n
    assert float(normal_error(kw["normal_t"], n, kw["motion"]).mean()) == pytest.approx(E_N_HALF, abs=1e-5)
    assert float(confidence_mask(**kw)["mask"].mean()) == pytest.approx(0.5, abs=1e-4)


def test_a_ninety_degree_edge_is_effectively_zero():
    kw = _flat_scene()
    n = torch.zeros(1, 3, H, W)
    n[:, 1] = 1.0
    kw["normal_prev"] = n
    assert float(confidence_mask(**kw)["mask"].max()) < 0.01


def test_both_terms_at_their_half_points_multiply_to_a_quarter():
    kw = _flat_scene()
    kw["depth_prev"] = kw["depth_prev"] + E_D_HALF_METRES
    theta = np.radians(30.0)
    n = torch.zeros(1, 3, H, W)
    n[:, 1], n[:, 2] = float(np.sin(theta)), float(np.cos(theta))
    kw["normal_prev"] = n
    assert float(confidence_mask(**kw)["mask"].mean()) == pytest.approx(0.25, abs=1e-4)


def test_alpha_and_beta_are_derived_from_the_half_points_not_typed_in():
    assert ALPHA == pytest.approx(np.log(2.0) / E_D_HALF_METRES)
    assert BETA == pytest.approx(np.log(2.0) / E_N_HALF)


def test_the_module_matches_the_functional_form():
    kw = _flat_scene()
    kw["depth_prev"] = kw["depth_prev"] + 0.1
    assert torch.equal(ConfidenceMask()(**kw)["mask"], confidence_mask(**kw)["mask"])


# --------------------------------------------------------------------------- #
# §12 — the mask must survive the export boundary too
# --------------------------------------------------------------------------- #
def test_the_mask_exports_to_opset_17_with_static_shapes(tmp_path):
    onnx = pytest.importorskip("onnx")

    class Graph(torch.nn.Module):
        def forward(self, depth_t, depth_prev, normal_t, normal_prev, motion,
                    fin_t, fin_prev, fresh):
            return confidence_mask(depth_t=depth_t, depth_prev=depth_prev,
                                   normal_t=normal_t, normal_prev=normal_prev,
                                   motion=motion, depth_finite_t=fin_t,
                                   depth_finite_prev=fin_prev, motion_valid=True,
                                   no_prev_transform=fresh)["mask"]

    kw = _flat_scene()
    args = (kw["depth_t"], kw["depth_prev"], kw["normal_t"], kw["normal_prev"],
            kw["motion"], kw["depth_finite_t"], kw["depth_finite_prev"],
            torch.zeros(1, 1, H, W))
    path = tmp_path / "mask.onnx"
    torch.onnx.export(Graph().eval(), args, str(path), opset_version=17,
                      dynamo=False, do_constant_folding=True,
                      input_names=["depth_t", "depth_prev", "normal_t", "normal_prev",
                                   "motion", "fin_t", "fin_prev", "fresh"],
                      output_names=["mask"])
    g = onnx.load(str(path)).graph
    ops = {n.op_type for n in g.node}
    assert not ({"If", "Loop", "NonZero"} & ops), f"data-dependent control flow: {ops}"
    for vi in list(g.input) + list(g.output):
        for d in vi.type.tensor_type.shape.dim:
            assert d.HasField("dim_value") and d.dim_value > 0
