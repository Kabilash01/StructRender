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


def _load_report():
    return json.loads(REPORT.read_text(encoding="utf-8")) if REPORT.is_file() else None


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


def test_model_a_diagnostic_is_a_separate_dated_artifact(report):
    """Decision 51: the generator cannot re-measure Model A's build failure, so
    embedding its numbers in a freshly dated M5 report would present an old
    measurement as new. It lives in its own file, referenced by name."""
    ref = report["model_a_comparison_ref"]
    art = Path(ROOT) / ref["artifact"]
    assert art.is_file(), f"referenced diagnostic missing: {ref['artifact']}"
    m = json.loads(art.read_text(encoding="utf-8"))
    assert m["tensorrt_build_ok"] is False
    assert "not identified" in m["diagnosis_status"].lower()
    # Scope: unresolved HERE, not "no deployment path anywhere".
    assert "unresolved" in m["scope_correction"].lower()
    assert "measured_utc" in m and "reproduction_status" in m


def test_report_matches_the_generator_schema():
    """The consistency check that was missing.

    The first version of this artifact carried four fields the generator never
    wrote — they had been hand-patched in — so the report validated against
    itself and nothing noticed. This imports the generator's own declared
    schema, so generator/report drift fails here instead of at the next
    regeneration. KB Decision 51.
    """
    gen = Path(ROOT) / "scripts" / "export_benchmark_m5.py"
    head = gen.read_text(encoding="utf-8").split("cfg = load_config")[0]
    ns: dict = {}
    exec(compile(head, str(gen), "exec"), {"__name__": "gen"}, ns)

    report = _load_report()
    if report is None:
        pytest.skip("report not generated yet")
    missing = [k for k in ns["REQUIRED_TOP_LEVEL"] if k not in report]
    assert not missing, f"generator declares keys the committed report lacks: {missing}"

    # The allow-list must be the DECLARED constant, not the observed census —
    # a list derived from the graph it validates can never fail.
    assert set(report["operator_allow_list"]) == set(ns["ALLOWED_OPS"]), (
        "report's allow-list differs from scripts/export_benchmark_m5.py::ALLOWED_OPS"
    )
    census = set(report["stages"]["onnx_fp32"]["op_types"])
    assert census < set(ns["ALLOWED_OPS"]) or census <= set(ns["ALLOWED_OPS"]), (
        f"unlisted operators in the shipped graph: {sorted(census - set(ns['ALLOWED_OPS']))}"
    )


def test_the_allow_list_can_actually_fail():
    """A guard that the guard works: an invented operator must be rejected."""
    gen = Path(ROOT) / "scripts" / "export_benchmark_m5.py"
    head = gen.read_text(encoding="utf-8").split("cfg = load_config")[0]
    ns: dict = {}
    exec(compile(head, str(gen), "exec"), {"__name__": "gen"}, ns)
    assert "NotARealOnnxOp" not in ns["ALLOWED_OPS"]


def test_latency_is_labelled_with_its_measurement_scope(report):
    """Decision 51: an engine-only number read as end-to-end FPS is the easiest
    misreading available in this project."""
    note = report["latency_note"].lower()
    assert "batch 1" in note and "excludes" in note
    for stage in ("trt_fp32", "trt_fp16"):
        s = report["stages"][stage]
        if "latency_ms_p95" not in s:
            assert "not measured" in s.get("latency_ms_p95_status", "")
