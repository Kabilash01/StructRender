"""M6's temporal loss — KB §4 *Training Loss Module*, Decision 41's constraint.

The load-bearing property is negative: the temporal term is an **addition** to the
mandated `1.0*L1 + 0.5*LPIPS(alex)` reconstruction objective and never a
replacement, so at weight 0 the M5 objective must come back unchanged — same
number, same `describe()` string.

CPU only. `lpips` is a hard training dependency (train_baseline and train_temporal
both raise without it), so the perceptual-path tests skip rather than pass on a
checkout that lacks it.
"""

from __future__ import annotations

import pytest
import torch

from renderer.losses.perceptual import PerceptualReconstructionLoss
from renderer.losses.temporal import TemporalObjective, masked_temporal_l1
from renderer.temporal.warp import backward_warp

HW = (32, 48)


def _frames(bs: int = 2):
    torch.manual_seed(0)
    return torch.rand(bs, 3, *HW), torch.rand(bs, 3, *HW), torch.rand(bs, 3, *HW)


# --------------------------------------------------------------------------- #
# the term itself
# --------------------------------------------------------------------------- #
def test_a_perfectly_reprojected_pair_costs_nothing():
    """Zero motion makes the warp the identity (KB Decision 25's UV convention
    composes exactly with `align_corners=False`), so a frame equal to its
    predecessor is perfectly consistent."""
    pred = torch.rand(2, 3, *HW)
    motion = torch.zeros(2, 2, *HW)
    mask = torch.ones(2, 1, *HW)
    assert float(masked_temporal_l1(pred, pred.clone(), motion, mask)) == pytest.approx(0.0, abs=1e-7)


def test_the_term_measures_the_warped_difference_not_the_raw_one():
    """With real motion, the reference is `W(pred_prev)` — not `pred_prev`. A term
    that forgot the warp would score a moving scene as flicker, which is exactly
    the confusion KB Decision 55 records for temporal error measured on a
    near-static split."""
    pred_prev = torch.rand(1, 3, *HW)
    motion = torch.full((1, 2, *HW), 0.05)
    mask = torch.ones(1, 1, *HW)
    pred = backward_warp(pred_prev, motion)
    warped_cost = float(masked_temporal_l1(pred, pred_prev, motion, mask))
    raw_cost = float(masked_temporal_l1(pred, pred_prev, torch.zeros_like(motion), mask))
    assert warped_cost < raw_cost


def test_the_mask_excludes_rather_than_discourages():
    """Pixels the engine says have no usable history must contribute exactly
    nothing — KB Decision 52's lesson, applied to the loss. Corrupting the
    masked-out region must not move the number at all."""
    pred, pred_prev, _ = _frames(1)
    motion = torch.zeros(1, 2, *HW)
    mask = torch.ones(1, 1, *HW)
    mask[..., : HW[0] // 2, :] = 0.0

    before = float(masked_temporal_l1(pred, pred_prev, motion, mask))
    pred2 = pred.clone()
    pred2[..., : HW[0] // 2, :] += 10.0
    after = float(masked_temporal_l1(pred2, pred_prev, motion, mask))
    assert after == pytest.approx(before, abs=1e-7)

    # ...and the same corruption inside the kept region DOES move it, or the test
    # above would pass for a loss that ignores its inputs.
    pred3 = pred.clone()
    pred3[..., HW[0] // 2:, :] += 10.0
    assert float(masked_temporal_l1(pred3, pred_prev, motion, mask)) > before + 1.0


def test_an_all_zero_mask_is_finite_and_zero():
    """Whole-frame invalidity is a real case (frame 0, `motion_valid == false`).
    A naive mean would divide by zero and take the whole run to NaN at step 1."""
    pred, pred_prev, _ = _frames(1)
    v = masked_temporal_l1(pred, pred_prev, torch.zeros(1, 2, *HW), torch.zeros(1, 1, *HW))
    assert torch.isfinite(v) and float(v) == 0.0


def test_the_term_is_differentiable_in_both_frames():
    """§4's formula as written has ground-truth frames on both sides and therefore
    no gradient at all; the substitution this module makes is only meaningful if
    the prediction actually carries one."""
    pred = torch.rand(1, 3, *HW, requires_grad=True)
    pred_prev = torch.rand(1, 3, *HW, requires_grad=True)
    masked_temporal_l1(pred, pred_prev, torch.full((1, 2, *HW), 0.01),
                       torch.ones(1, 1, *HW)).backward()
    assert float(pred.grad.abs().sum()) > 0.0
    assert float(pred_prev.grad.abs().sum()) > 0.0


# --------------------------------------------------------------------------- #
# composition with the mandated objective (KB Decision 41)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("perceptual", [0.0, 0.5])
def test_zero_weight_reduces_to_the_m5_objective(perceptual):
    """The strict-special-case property. At `loss.temporal = 0` the criterion must
    be `PerceptualReconstructionLoss` — same total and same published objective
    string — so the ladder's shared objective is verifiably still inside this one."""
    if perceptual > 0.0:
        pytest.importorskip("lpips")
    torch.manual_seed(0)
    ref = PerceptualReconstructionLoss(w_l1=1.0, w_perceptual=perceptual, net="alex")
    obj = TemporalObjective(w_l1=1.0, w_perceptual=perceptual, net="alex", w_temporal=0.0)
    pred, prev, target = _frames(2)

    ref_total, ref_parts = ref(pred, target)
    obj_total, obj_parts = obj(pred, target)
    assert obj.describe() == ref.describe()
    assert torch.allclose(obj_total, ref_total, atol=1e-6)
    assert obj_parts == ref_parts
    assert "temporal" not in obj_parts
    # ...and it does not even need the temporal inputs at weight 0.
    assert torch.allclose(obj(pred, target, prev, torch.zeros(2, 2, *HW),
                              torch.ones(2, 1, *HW))[0], ref_total, atol=1e-6)


def test_a_non_zero_weight_adds_the_term_and_nothing_else():
    """The addition is exactly `w * L_temp`, so the reconstruction objective is
    arithmetically untouched (Decision 41: it is mandated for every rung)."""
    torch.manual_seed(0)
    ref = PerceptualReconstructionLoss(w_l1=1.0, w_perceptual=0.0)
    obj = TemporalObjective(w_l1=1.0, w_perceptual=0.0, w_temporal=0.25)
    pred, prev, target = _frames(2)
    motion, mask = torch.full((2, 2, *HW), 0.01), torch.rand(2, 1, *HW)

    total, parts = obj(pred, target, prev, motion, mask)
    expected = ref(pred, target)[0] + 0.25 * masked_temporal_l1(pred, prev, motion, mask)
    assert torch.allclose(total, expected, atol=0, rtol=0)
    assert parts["temporal"] == pytest.approx(float(masked_temporal_l1(pred, prev, motion, mask)))


def test_a_non_zero_weight_refuses_to_run_without_its_inputs():
    """Silently dropping the term would train under a different objective than the
    report names (KB Decision 42), which is the failure mode that makes an
    artifact indistinguishable from a stale one."""
    obj = TemporalObjective(w_l1=1.0, w_perceptual=0.0, w_temporal=0.25)
    pred, _prev, target = _frames(1)
    with pytest.raises(ValueError):
        obj(pred, target)


def test_describe_names_the_temporal_term_when_it_is_on():
    obj = TemporalObjective(w_l1=1.0, w_perceptual=0.0, w_temporal=0.5)
    assert obj.describe() == "1*L1 + 0.5*L_temp(M_t-weighted)"
    assert TemporalObjective(w_l1=1.0, w_perceptual=0.0, w_temporal=0.0).describe() == "1*L1"


def test_the_loss_and_the_metric_use_the_same_mask_function():
    """Deliverable, stated as a test: the loss's mask is not "the same formula" as
    the gate's, it is the same function with the same calibrated constants. A
    second implementation is how the two would drift apart on which pixels are
    trustworthy."""
    import inspect

    import eval.metrics.temporal as metric
    import renderer.temporal.pair_dataset as ds
    from renderer.temporal.confidence import ALPHA, BETA, confidence_mask

    assert metric.engine_confidence.__module__ == "eval.metrics.temporal"
    assert ds.engine_confidence is metric.engine_confidence
    assert inspect.signature(confidence_mask).parameters["alpha"].default == ALPHA
    assert inspect.signature(confidence_mask).parameters["beta"].default == BETA
