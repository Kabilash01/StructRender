"""The committed M5 export artifact must itself show what Decision 48 claims.

KB Decision 44's finding, one milestone later: a committed report that nothing
asserts is unfalsifiable evidence — `m5_onnx_export.json` could record a failed
build, a symbolic-shaped graph, or a stale session and the suite would stay
green. Decision 45 closed this for the probe; this closes it for trained width.

Skips (never fails) when the artifact is absent, since it needs CUDA + TensorRT
to regenerate and a clean checkout has neither.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from renderer.training.train_baseline import ROOT

REPORT = Path(ROOT) / "eval" / "reports" / "m5_onnx_export.json"
FP32_TOLERANCE = 1e-3


@pytest.fixture(scope="module")
def report() -> dict:
    if not REPORT.is_file():
        pytest.skip(f"{REPORT.name} not generated — run scripts/export_benchmark_m5.py")
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_report_self_identifies(report):
    """KB Decision 42 — provenance, and non-placeholder."""
    for f in ("session_id", "style_id", "objective", "generated_utc",
              "torch_version", "tensorrt_version", "opset"):
        assert f in report, f"missing {f!r} (Decision 42)"
        assert str(report[f]).strip() not in ("", "none", "unknown", "discovered-newest")


def test_exit_criterion_3_onnx_export_succeeded(report):
    s = report["stages"]["onnx_fp32"]
    assert s["ok"] is True
    # §12: "prefer fixed/bounded shapes". A green build is NOT evidence of a static
    # graph — Decision 45 records TensorRT happily building a symbolic-dim graph.
    assert s["shapes_are_static"] is True
    assert report["opset"] == 17


def test_onnxruntime_parity_is_inside_the_fp32_tolerance(report):
    s = report["stages"]["onnxruntime_parity"]
    assert s["ok"] is True and s["max_abs_diff"] <= FP32_TOLERANCE


def test_exit_criterion_4_tensorrt_engine_built_and_ran(report):
    """§10 M5: 'TensorRT benchmark happens here, not at the end.' The benchmark
    is the deliverable — §12 calls 15-20 ms an engineering target, not a gate —
    so this asserts an engine was built AND executed, and that a latency exists."""
    for prec in ("trt_fp32", "trt_fp16"):
        s = report["stages"][prec]
        assert s["ok"] is True, f"{prec} did not build"
        assert s.get("latency_ms_mean", 0) > 0, f"{prec} has no measured latency"
    assert report["stages"]["trt_fp32"]["max_abs_diff_vs_torch"] <= FP32_TOLERANCE


def test_fp16_tolerance_exceedance_is_visible_in_the_artifact(report):
    """Decision 48 records that FP16 parity exceeds the FP32 1e-3 tolerance and
    does not waive it. A reader of the artifact alone must see that too — the
    prose being right does not help someone reading the JSON."""
    s = report["stages"]["trt_fp16"]
    d = s["max_abs_diff_vs_torch"]
    if d > FP32_TOLERANCE:
        assert "fp32_tolerance_exceeded" in s and s["fp32_tolerance_exceeded"] is True, (
            f"FP16 parity {d:.3e} exceeds the {FP32_TOLERANCE} FP32 tolerance but the "
            "artifact does not say so — Decision 48 requires it flagged, not waived"
        )
        assert "fp16_tolerance_note" in s


def test_operator_allow_list_makes_new_ops_a_visible_diff(report):
    """§12 wants a new operator in M5's graph to be a visible diff. The list is
    the SHIPPED graph's, so `Equal`/`Where` (emitted by the style-token expand)
    are named rather than silently present."""
    census = set(report["stages"]["onnx_fp32"]["op_types"])
    allowed = set(report["operator_allow_list"])
    assert census <= allowed, f"unlisted operators in the shipped graph: {sorted(census - allowed)}"


def test_model_a_tensorrt_failure_is_recorded_as_open(report):
    """Decision 48 narrows the cause but does not identify it. The artifact must
    not overstate that into a diagnosis."""
    m = report["model_a_comparison"]
    assert m["tensorrt_build_ok"] is False
    assert "not identified" in m["diagnosis_status"].lower()
