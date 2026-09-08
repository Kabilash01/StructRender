"""Accumulated gradients must equal the single-batch gradient they stand in for.

KB Decision 51. The M5 runs were trained with a loop that divided every
microbatch by ``accum_steps``, including a trailing group that held fewer than
``accum_steps`` microbatches — so the last optimiser step of each epoch applied a
fractional-magnitude gradient. Nothing tested it, so nothing caught it.

Pure CPU, synthetic tensors, no session/targets/GPU: runs on a clean checkout.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn


def _grad_vector(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.grad.reshape(-1) for p in model.parameters()])


def _accumulate(model: nn.Module, chunks, accum: int, *, correct: bool) -> torch.Tensor:
    """`correct=True` divides by the group's real size; False reproduces the
    old `/accum` behaviour the M5 runs used."""
    model.zero_grad(set_to_none=True)
    n = len(chunks)
    for i, (x, y) in enumerate(chunks):
        start = (i // accum) * accum
        size = min(accum, n - start) if correct else accum
        (nn.functional.l1_loss(model(x), y) / size).backward()
    return _grad_vector(model)


@pytest.fixture
def fixture():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(6, 12), nn.GELU(), nn.Linear(12, 3))
    x, y = torch.randn(8, 6), torch.randn(8, 3)
    return model, x, y


def test_even_split_matches_the_single_batch_gradient(fixture):
    """8 samples as one batch == 4 groups of 2, when the split is even."""
    model, x, y = fixture
    model.zero_grad(set_to_none=True)
    nn.functional.l1_loss(model(x), y).backward()
    whole = _grad_vector(model)

    chunks = [(x[i:i + 2], y[i:i + 2]) for i in range(0, 8, 2)]
    # 4 groups of one microbatch each accumulate to the mean of the 4 groups
    got = _accumulate(model, chunks, accum=4, correct=True)
    assert torch.allclose(whole, got, atol=1e-6), (whole - got).abs().max()


def test_trailing_partial_group_is_scaled_by_its_real_size(fixture):
    """The M5 shape: an odd number of microbatches at accum=2.

    5 microbatches, accum 2 -> groups of [2, 2, 1]. The corrected form weights
    the trailing group by 1; the old form divides it by 2 and applies half the
    gradient it should.
    """
    model, _, _ = fixture
    torch.manual_seed(1)
    chunks = [(torch.randn(2, 6), torch.randn(2, 3)) for _ in range(5)]

    correct = _accumulate(model, chunks, accum=2, correct=True)
    legacy = _accumulate(model, chunks, accum=2, correct=False)

    assert not torch.allclose(correct, legacy), (
        "corrected and legacy scaling agree — the test is not exercising the trailing group"
    )
    # The whole difference is the trailing group counted at half weight.
    model.zero_grad(set_to_none=True)
    x, y = chunks[-1]
    nn.functional.l1_loss(model(x), y).backward()
    trailing = _grad_vector(model)
    assert torch.allclose(correct - legacy, 0.5 * trailing, atol=1e-6)


def test_accum_1_is_exactly_the_unaccumulated_path(fixture):
    """`accum_steps: 1` must be byte-identical to no accumulation, or Model A's
    historical runs are not reproducible by the shipped code."""
    model, x, y = fixture
    model.zero_grad(set_to_none=True)
    nn.functional.l1_loss(model(x), y).backward()
    whole = _grad_vector(model)
    got = _accumulate(model, [(x, y)], accum=1, correct=True)
    assert torch.equal(whole, got)
