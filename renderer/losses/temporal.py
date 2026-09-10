"""Temporal consistency loss — KB §4 *Training Loss Module*, Milestone 6.

§4 writes the term as

    L_temp = M_t (*) || I_t - W(I_{t-1}, V_t) ||_1

**and that formula, read literally, has no gradient.** Everywhere else in §4 the
hat distinguishes the two: ``L_rec = ||I_t - \\hat I_t||_1``. So ``I`` is the
teacher target, and a term built from two teacher targets is a constant with
respect to the model — it cannot be a loss. The repair used here substitutes the
**prediction on both sides**::

    L_temp = mean_p [ M_t(p) * | \\hat I_t(p) - W(\\hat I_{t-1}, V_t)(p) | ] / mean_p M_t(p)

which is the only reading that (a) has a gradient and (b) is the quantity M6's
pre-registered gate actually measures — ``eval/metrics/temporal.py`` scores
``|I_t - W(I_{t-1})|`` **on each source's own frames**, so for a model that is
its own predictions. This is a deviation from §4 as written and is flagged for
the Decision Log rather than done quietly.

**The two repairs that were rejected**, recorded because each looks reasonable:

* ``|\\hat I_t - W(I_{t-1}, V_t)|`` — anchor the prediction to the *warped teacher
  target*. Cheap (needs no second forward pass) and blur-resistant, but it drives
  the model toward the teacher's *own* temporal behaviour, and KB Decision 55
  measured the teacher targets as the **least** temporally stable source on this
  scene — roughly 28x the engine's error. It would optimise the gate in the wrong
  direction.
* Penalising the difference of temporal differences against the target. Same
  objection, same reason.

**The degenerate solution is real and is not defended against here.** A frozen or
blurred output has zero temporal error. That is exactly the confound KB Decision
53 found for RCS and it is why Decision 55 pre-registered a *sharpness guard*
alongside the M6 threshold. This module is an **addition** to the mandated
``1.0*L1 + 0.5*LPIPS(alex)`` reconstruction objective (KB Decision 41 fixes that
for every §11 ladder rung), never a replacement, and the reconstruction term is
what makes freezing expensive.

**The mask is the metric's mask.** Not "the same formula" — the same function,
:func:`renderer.temporal.confidence.confidence_mask`, with the same calibrated
``ALPHA``/``BETA`` and the same five hard zeros, produced from engine buffers
only. A loss that trusted different pixels than the gate would optimise a
different quantity than the one M6 is judged on.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from renderer.losses.perceptual import PerceptualReconstructionLoss
from renderer.temporal.warp import backward_warp


def masked_temporal_l1(
    pred_t: torch.Tensor,
    pred_prev: torch.Tensor,
    motion: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """``M_t``-weighted mean of ``|pred_t - W(pred_prev, V_t)|`` over the frame.

    All four tensors are at **image** resolution. ``mask`` is the continuous
    ``M_t`` in ``[0, 1]``, used as a weight rather than thresholded: it is the
    differentiable form, and it is exactly the weighting the metric reports as
    ``temporal_error_mask_weighted``. The metric's headline ``temporal_error``
    thresholds the same mask at its calibrated 0.5 crossing; both are read off
    the same ``M_t``, so loss and gate cannot disagree about which pixels are
    trustworthy.

    The channel mean is taken before the spatial weighting, matching
    ``eval/metrics/temporal.evaluate_source`` term for term.
    """
    err = (pred_t - backward_warp(pred_prev, motion)).abs().mean(dim=1, keepdim=True)
    return (err * mask).sum() / mask.sum().clamp_min(1.0)


class TemporalObjective(nn.Module):
    """``L_rec`` (mandated) ``+ w_temporal * L_temp`` (M6's addition).

    ``w_temporal = 0`` reproduces
    :class:`~renderer.losses.perceptual.PerceptualReconstructionLoss` exactly —
    same total, same ``describe()`` string — so the M5 objective is a strict
    special case and the ladder's shared objective is verifiably still inside
    this one. Asserted by ``test_zero_weight_reduces_to_the_m5_objective``.

    The weight is a config field (``loss.temporal``) and was **not** tuned
    against the M6 gate: tuning a training hyper-parameter against a
    pre-registered exit criterion would convert it into a fitted one.
    """

    def __init__(
        self,
        w_l1: float = 1.0,
        w_perceptual: float = 0.0,
        net: str = "alex",
        w_temporal: float = 0.0,
    ) -> None:
        super().__init__()
        self.reconstruction = PerceptualReconstructionLoss(
            w_l1=w_l1, w_perceptual=w_perceptual, net=net
        )
        self.w_temporal = float(w_temporal)

    @property
    def used_perceptual(self) -> bool:
        return self.reconstruction.used_perceptual

    def describe(self) -> str:
        base = self.reconstruction.describe()
        if self.w_temporal <= 0.0:
            return base
        return f"{base} + {self.w_temporal:g}*L_temp(M_t-weighted)"

    def forward(
        self,
        pred_t: torch.Tensor,
        target_t: torch.Tensor,
        pred_prev: torch.Tensor | None = None,
        motion: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        total, parts = self.reconstruction(pred_t, target_t)
        if self.w_temporal <= 0.0:
            return total, parts
        if pred_prev is None or motion is None or mask is None:
            raise ValueError(
                "a non-zero loss.temporal weight needs pred_prev, motion and mask; "
                "silently skipping the term would train under a different objective "
                "than the report names (KB Decision 42)"
            )
        temp = masked_temporal_l1(pred_t, pred_prev, motion, mask)
        total = total + self.w_temporal * temp
        parts["temporal"] = float(temp.detach())
        parts["total"] = float(total.detach())
        return total, parts
