"""KB §11 Required unit tests — "finite loss" and "gradient flow".

Both are listed as Required in §11's unit-test list and M4 is the first milestone
with a loss and a backward pass, so they land here. §11's *Local Pre-Cloud
Training Gate* item 3 ("verify no NaN/Inf") is enforced in the training loop
itself by :func:`renderer.training.train_baseline.assert_finite_loss`; the tests
below pin that it aborts rather than limping on.

Pure CPU, synthetic tensors: no GPU, no recorded session, no teacher targets, no
checkpoint. These run — and must pass — on a clean checkout.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from renderer.baselines.unet import UNetBaseline
from renderer.training.train_baseline import NonFiniteLossError, assert_finite_loss

# Small enough to be fast on CPU: 4 downsamples means H, W divisible by 16, and
# GroupNorm(8, c) means every channel count must be divisible by 8.
SHAPE = (2, 7, 32, 32)
BASE_CHANNELS = 8


@pytest.fixture
def model_and_batch():
    torch.manual_seed(0)
    model = UNetBaseline(in_channels=SHAPE[1], out_channels=3, base_channels=BASE_CHANNELS)
    x = torch.rand(SHAPE)
    y = torch.rand(SHAPE[0], 3, SHAPE[2], SHAPE[3])
    return model, x, y


# --------------------------------------------------------------------------- #
# finite loss
# --------------------------------------------------------------------------- #
def test_loss_is_finite(model_and_batch):
    model, x, y = model_and_batch
    loss = F.l1_loss(model(x), y)
    assert torch.isfinite(loss), f"L1 loss is not finite: {loss}"
    assert loss.item() > 0.0


def test_loss_is_finite_on_extreme_inputs():
    """The real inputs carry a background sentinel and unit normals; an all-zero
    or all-one frame must not produce a NaN through the norm layers."""
    torch.manual_seed(0)
    model = UNetBaseline(in_channels=SHAPE[1], out_channels=3, base_channels=BASE_CHANNELS)
    y = torch.zeros(SHAPE[0], 3, SHAPE[2], SHAPE[3])
    for name, x in (
        ("zeros", torch.zeros(SHAPE)),
        ("ones", torch.ones(SHAPE)),
        ("large", torch.full(SHAPE, 1e3)),
    ):
        loss = F.l1_loss(model(x), y)
        assert torch.isfinite(loss), f"{name} input gave a non-finite loss: {loss}"


def test_assert_finite_loss_aborts_on_nan_and_inf():
    """§11 gate item 3, enforced. Must raise — never clamp, substitute or skip."""
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(NonFiniteLossError):
            assert_finite_loss(torch.tensor(bad), step=7, epoch=3)


def test_assert_finite_loss_passes_a_finite_value_through_unchanged():
    value = assert_finite_loss(torch.tensor(0.125), step=0)
    assert value == pytest.approx(0.125)
    assert isinstance(value, float) and math.isfinite(value)


def test_assert_finite_loss_error_names_where_it_happened():
    with pytest.raises(NonFiniteLossError) as e:
        assert_finite_loss(torch.tensor(float("nan")), step=42, epoch=5)
    assert "epoch 5, step 42" in str(e.value)


# --------------------------------------------------------------------------- #
# gradient flow
# --------------------------------------------------------------------------- #
def test_every_parameter_receives_a_finite_gradient(model_and_batch):
    model, x, y = model_and_batch
    F.l1_loss(model(x), y).backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} received no gradient — it is off the graph"
        assert torch.isfinite(p.grad).all(), f"{name} has a non-finite gradient"


def test_gradient_reaches_every_parameter(model_and_batch):
    """Not just *a* gradient — a non-zero one. An all-zero gradient on a whole
    tensor is what a detached skip connection or a dead branch looks like, and it
    is invisible to a finite-loss check."""
    model, x, y = model_and_batch
    F.l1_loss(model(x), y).backward()
    dead = [n for n, p in model.named_parameters() if float(p.grad.abs().sum()) == 0.0]
    assert not dead, f"parameters with an all-zero gradient (no flow): {dead}"


def test_gradient_reaches_the_encoder_stem(model_and_batch):
    """The deepest point from the loss: the very first conv. If backprop is
    severed anywhere in the U-Net, this is where it shows."""
    model, x, y = model_and_batch
    F.l1_loss(model(x), y).backward()
    stem = model.enc[0][0].weight
    assert stem.grad is not None and float(stem.grad.abs().sum()) > 0.0


def test_an_optimiser_step_moves_the_parameters_and_lowers_the_loss(model_and_batch):
    """Gradient flow end to end: gradients exist, the optimiser consumes them,
    and the loss on a fixed batch goes down. This is §11 gate item 2 ("verify
    loss decreases") in its smallest CPU form."""
    model, x, y = model_and_batch
    before = [p.detach().clone() for p in model.parameters()]
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    first = None
    for step in range(20):
        loss = F.l1_loss(model(x), y)
        value = assert_finite_loss(loss, step=step)
        first = value if first is None else first
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    with torch.no_grad():
        last = float(F.l1_loss(model(x), y))
    assert last < first, f"loss did not decrease over 20 steps: {first:.5f} -> {last:.5f}"
    assert any(
        not torch.equal(b, p.detach()) for b, p in zip(before, model.parameters())
    ), "no parameter changed"
