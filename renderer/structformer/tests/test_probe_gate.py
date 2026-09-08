"""The export probe's pass/fail gate — KB Decision 52.

Drives the production `_blocked` helper from `scripts/probe_m5_export.py`
directly. The first version counted only ONNX export and FP32 run, so an FP16
failure or an out-of-tolerance FP32 parity produced an empty `blocked` list and
the probe reported success for a variant that had not met its criteria.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from renderer.training.train_baseline import ROOT

spec = importlib.util.spec_from_file_location(
    "probe_m5_export", Path(ROOT) / "scripts" / "probe_m5_export.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)

TOL = probe.TOLERANCE_MAX_ABS


def _variant(*, onnx=True, b32=True, r32=True, d32=1e-6, b16=True, r16=True):
    return {"onnx_export_ok": onnx,
            "trt": {"fp32": {"build_ok": b32, "run_ok": r32, "max_abs_diff": d32},
                    "fp16": {"build_ok": b16, "run_ok": r16}}}


def test_a_fully_passing_variant_is_not_blocked():
    assert probe._blocked({"v": _variant()}, trt=True) == []


@pytest.mark.parametrize("kw, why", [
    ({"onnx": False}, "ONNX export failed"),
    ({"b32": False}, "FP32 build failed"),
    ({"r32": False}, "FP32 run failed"),
    ({"b16": False}, "FP16 build failed — was silently ignored"),
    ({"r16": False}, "FP16 run failed — was silently ignored"),
    ({"d32": TOL * 10}, "FP32 parity out of tolerance — was silently ignored"),
])
def test_required_failures_are_blocked(kw, why):
    assert probe._blocked({"v": _variant(**kw)}, trt=True) == ["v"], why


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), None, "n/a"])
def test_non_finite_parity_is_rejected(bad):
    """`nan <= tol` is False, so a NaN would never be 'within tolerance' — but it
    also must not pass through as an unremarked value."""
    assert probe._blocked({"v": _variant(d32=bad)}, trt=True) == ["v"]


def test_no_trt_does_not_assert_tensorrt_passed():
    """--no-trt is ONNX-only. A variant whose TensorRT results are absent or
    failing must not be reported as blocked, and must not be reported as passing
    TensorRT either — the caller states the scope."""
    v = {"v": _variant(b16=False, r32=False)}
    assert probe._blocked(v, trt=False) == []
    onnx_failed = {"v": _variant(onnx=False)}
    assert probe._blocked(onnx_failed, trt=False) == ["v"], "ONNX is still gated"


def test_finite_helper_matches_its_use():
    assert probe._finite(0.0) and probe._finite(1e-6)
    assert not probe._finite(float("nan")) and not probe._finite(None)
