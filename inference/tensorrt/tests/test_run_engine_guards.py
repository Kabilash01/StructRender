"""CPU guards for the TensorRT runner — KB Decision 52.

These call the production helpers in ``run_engine.py`` directly rather than
reimplementing their logic, so a change to the real code path fails here. No
CUDA, no engine, no allocation: importable and runnable on a clean checkout.
"""

from __future__ import annotations

import pytest
import torch

from inference.tensorrt.benchmark.run_engine import execute_or_raise, validate_input_shape


class _Context:
    """Minimal stand-in for an IExecutionContext.

    ``execute_async_v3`` returns False on the nth call, which is exactly how
    TensorRT reports a failed enqueue — it returns a bool, it does not raise.
    """

    def __init__(self, fail_on: int | None = None):
        self.calls = 0
        self.fail_on = fail_on

    def execute_async_v3(self, _stream) -> bool:
        self.calls += 1
        return not (self.fail_on is not None and self.calls == self.fail_on)


# --------------------------------------------------------------------------- #
# finding 3 — execution failures must propagate, in BOTH loops
# --------------------------------------------------------------------------- #
def test_successful_execution_does_not_raise():
    ctx = _Context()
    execute_or_raise(ctx, 0, phase="warmup", index=0)
    assert ctx.calls == 1


@pytest.mark.parametrize("phase", ["warmup", "timed"])
def test_failed_execution_raises_in_either_phase(phase):
    """Without this, a caller publishes a latency for a run that never executed
    and reports run_ok=True for a failed engine."""
    ctx = _Context(fail_on=1)
    with pytest.raises(RuntimeError) as e:
        execute_or_raise(ctx, 0, phase=phase, index=3)
    assert phase in str(e.value) and "3" in str(e.value)


def test_failure_on_a_later_iteration_is_still_caught():
    ctx = _Context(fail_on=5)
    for i in range(4):
        execute_or_raise(ctx, 0, phase="timed", index=i)
    with pytest.raises(RuntimeError):
        execute_or_raise(ctx, 0, phase="timed", index=4)


# --------------------------------------------------------------------------- #
# finding 4 — shape validation, before any device pointer is bound
# --------------------------------------------------------------------------- #
def test_matching_shape_is_accepted():
    validate_input_shape("structure", torch.zeros(1, 7, 288, 512), (1, 7, 288, 512))


@pytest.mark.parametrize(
    "shape, why",
    [
        ((1, 7, 144, 512), "undersized — engine would read past the buffer"),
        ((1, 7, 576, 512), "oversized"),
        ((1, 7, 512, 288), "equal element count, transposed — silently misread"),
        ((7, 1, 288, 512), "equal element count, axes swapped"),
        ((1, 288, 512, 7), "equal element count, channels-last"),
    ],
)
def test_mismatched_shapes_are_rejected(shape, why):
    with pytest.raises(ValueError) as e:
        validate_input_shape("structure", torch.zeros(*shape), (1, 7, 288, 512))
    assert "structure" in str(e.value), why


def test_equal_element_count_is_not_treated_as_equivalent():
    """The dangerous case: numel matches, so nothing downstream would complain."""
    a, b = (1, 7, 288, 512), (1, 7, 512, 288)
    assert torch.zeros(*a).numel() == torch.zeros(*b).numel()
    with pytest.raises(ValueError):
        validate_input_shape("structure", torch.zeros(*b), a)
