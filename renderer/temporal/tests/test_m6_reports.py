"""The committed M6 artifacts must themselves show what the M6 report claims.

KB Decision 44's finding, restated once per milestone: a committed report that
nothing asserts is unfalsifiable evidence. Each artifact below could record a
failed TensorRT build, a symbolic-shaped graph, an uncalibrated alpha or a
temporal-error table with no sharpness column, and the suite would stay green.

Each test SKIPS (never fails) when its artifact is absent — regenerating them
needs CUDA + TensorRT and the gitignored session, and a clean checkout has
neither.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from renderer.training.train_baseline import ROOT

REPORTS = Path(ROOT) / "eval" / "reports"
FP32_TOLERANCE = 1e-3


def _report(name: str, generator: str) -> dict:
    path = REPORTS / name
    if not path.is_file():
        pytest.skip(f"{name} not generated — run {generator}")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def probe() -> dict:
    return _report("m6_gridsample_probe.json", "python -m scripts.probe_m6_gridsample")


@pytest.fixture(scope="module")
def calib() -> dict:
    return _report("m6_mask_calibration.json", "python -m scripts.calibrate_m6_mask")


@pytest.fixture(scope="module")
def baseline() -> dict:
    return _report("m6_temporal_baseline.json",
                   "python -m eval.protocols.run_temporal_baseline")


# --------------------------------------------------------------------------- #
# step 0 — the GridSample probe
# --------------------------------------------------------------------------- #
def test_probe_matches_its_generators_declared_schema(probe):
    from scripts.probe_m6_gridsample import REQUIRED_TOP_LEVEL

    missing = [k for k in REQUIRED_TOP_LEVEL if k not in probe]
    assert not missing, f"committed artifact and generator disagree: {missing}"


def test_probe_verdict_is_not_reached_by_omission(probe):
    """KB Decision 52: an empty ``blocked`` list from an ONNX-only run asserts
    nothing about TensorRT. The scope has to say which it was."""
    assert probe["verdict"] == "PASS" and probe["blocked"] == []
    assert "TENSORRT NOT RUN" not in probe["trt_scope"], (
        "the committed artifact is from an ONNX-only run; it cannot be cited as "
        "TensorRT evidence")


def test_gridsample_reached_a_built_and_executed_tensorrt_engine(probe):
    minimal = {k: v for k, v in probe["variants"].items() if v["gate"] == "minimal"}
    assert len(minimal) >= 5
    for name, v in minimal.items():
        assert "GridSample" in v["op_types"], name
        assert v["shapes_are_static"] is True, name
        f32 = v["trt_fp32"]
        assert f32["build_ok"] and f32["run_ok"], name
        assert f32["within_tolerance"] and f32["output_finite"], name
        assert f32["max_abs_diff_vs_torch"] <= FP32_TOLERANCE, name


def test_the_fp16_positional_finding_is_recorded_not_quietly_passed(probe):
    """The probe's substantive result: FP16 grid arithmetic misses the FP32
    tolerance by orders of magnitude, and the artifact must say so rather than
    carry a bare ``ok``."""
    v = probe["variants"]["image_bilinear_zeros_ac0"]["trt_fp16"]
    assert v["build_ok"] and v["run_ok"] and v["output_finite"]
    assert v["fp32_tolerance_exceeded"] is True
    assert "fp16_tolerance_note" in v


def test_the_fp16_workaround_was_measured_and_reaches_the_tolerance(probe):
    """KB Decision 45's rule: report a blocker with the workaround you MEASURED."""
    v = probe["variants"]["image_bilinear_zeros_ac0_fp32grid"]["trt_fp16"]
    assert v["build_ok"] and v["run_ok"] and v["output_finite"]
    assert v["fp32_tolerance_exceeded"] is False
    assert v["max_abs_diff_vs_torch_fp32"] <= FP32_TOLERANCE


def test_the_shipped_warp_and_mask_graph_ran_on_tensorrt(probe):
    v = probe["variants"]["shipped_warp_and_mask"]
    assert v["shapes_are_static"] is True
    assert v["trt_fp32"]["build_ok"] and v["trt_fp32"]["run_ok"]
    assert v["trt_fp32"]["within_tolerance"] and v["trt_fp32"]["output_finite"]
    assert v["trt_fp16"]["measured"] is False, (
        "an unmeasured FP16 stage must say so, not carry build/run flags")


def test_new_operators_are_a_visible_diff_against_m5(probe):
    """§12 requires a new operator to be a visible diff. GridSample at minimum."""
    assert "GridSample" in probe["new_operators_vs_m5"]
    assert "GridSample" not in probe["m5_allowed_ops"]


# --------------------------------------------------------------------------- #
# step 2 — the mask calibration
# --------------------------------------------------------------------------- #
def test_calibration_matches_its_generators_declared_schema(calib):
    from scripts.calibrate_m6_mask import REQUIRED_TOP_LEVEL

    missing = [k for k in REQUIRED_TOP_LEVEL if k not in calib]
    assert not missing, f"committed artifact and generator disagree: {missing}"


def test_alpha_and_beta_in_the_artifact_are_the_ones_the_code_uses(calib):
    """Prose and code must agree — otherwise the calibration describes a mask
    that never ran."""
    from renderer.temporal.confidence import ALPHA, BETA, E_D_HALF_METRES, E_N_HALF

    c = calib["calibration"]
    assert c["alpha"] == pytest.approx(ALPHA)
    assert c["beta"] == pytest.approx(BETA)
    assert c["e_d_half_metres"] == pytest.approx(E_D_HALF_METRES)
    assert c["e_n_half"] == pytest.approx(E_N_HALF)


def test_the_depth_crossing_is_the_measured_separation_optimum(calib):
    """"Calibrated from measured error distributions" has to be checkable."""
    sep = calib["scene_geometry_scale"]["depth_separation"]
    assert sep["argmax_tau"] == pytest.approx(calib["calibration"]["e_d_half_metres"])
    row = sep["sweep"][f"{calib['calibration']['e_d_half_metres']:.5g}"]
    assert row["across_object_above"] > 0.8, "the crossing must catch real occlusion boundaries"
    assert row["within_surface_above"] < 0.1, "and must not fire on ordinary slanted geometry"


def test_the_error_distributions_are_reported_for_both_tick_deltas(calib):
    for group in ("dt_eq_1", "dt_gt_1"):
        g = calib["error_distributions"][group]
        assert g["n_pairs"] > 0
        assert g["e_d_metres"]["n"] > 0 and g["e_n_literal"]["n"] > 0
        # both forms of e_N, so the "not renormalised" choice stays checkable
        assert g["e_n_renormalised_warped_normal"]["n"] > 0


def test_the_deviations_are_declared_rather_than_absorbed(calib):
    items = {d["item"] for d in calib["deviations_flagged_for_the_decision_log"]}
    assert any("undefined on background" in i for i in items)
    assert any("not renormalised" in i for i in items)
    assert any("S7 never fires" in i for i in items)
    for d in calib["deviations_flagged_for_the_decision_log"]:
        assert "NOT written by this run" in d["action"]


# --------------------------------------------------------------------------- #
# step 3 — the temporal-error baseline
# --------------------------------------------------------------------------- #
def test_baseline_matches_its_generators_declared_schema(baseline):
    from eval.protocols.run_temporal_baseline import REQUIRED_TOP_LEVEL

    missing = [k for k in REQUIRED_TOP_LEVEL if k not in baseline]
    assert not missing, f"committed artifact and generator disagree: {missing}"


def test_no_source_reports_temporal_error_without_a_sharpness_figure(baseline):
    """The single rule this metric may not break (KB Decision 53's confound)."""
    for name, s in baseline["sources"].items():
        if "skipped" in s:
            continue
        for group in ("dt_eq_1", "dt_gt_1"):
            g = s[group]
            if not g["n_pairs"]:
                continue
            assert "temporal_error" in g, name
            assert g["sharpness_ratio_vs_target"]["n"] > 0, f"{name}/{group} has TE and no sharpness"
            assert "do_nothing_anchor_unwarped" in g, name


def test_tick_deltas_are_reported_separately_and_the_gate_is_named(baseline):
    assert baseline["split"]["n_pairs_dt_eq_1"] > 0
    assert baseline["split"]["n_pairs_dt_gt_1"] > 0
    assert "dt_eq_1 is the headline gate" in baseline["gating"]


def test_the_measurability_warning_is_backed_by_the_numbers_in_the_same_file(baseline):
    """Prose and assertion must agree: the warning says the validation pairs
    barely move, so the artifact must show that."""
    g = baseline["sources"]["native_engine"]["dt_eq_1"]
    assert g["pairs_below_half_a_pixel"] >= 0.9 * g["n_pairs"], (
        "the measurability warning claims almost every dt==1 pair is sub-pixel")
    assert g["median_motion_px"]["mean"] < 1.0


def test_the_cnn_across_seed_spread_comes_from_three_seed_only_variants(baseline):
    spread = baseline["cnn_across_seed_spread"]
    assert "unavailable" not in spread, spread
    assert len(set(spread["seeds"].values())) == 3, "three DISTINCT training seeds"
    for key in ("dt_eq_1_temporal_error", "dt_gt_1_temporal_error", "dt_eq_1_sharpness_ratio"):
        assert spread[key]["n"] == 3 and spread[key]["sd"] is not None


def test_the_motion_response_diagnostic_excludes_every_trained_model(baseline):
    """It runs on frames from any split, which is only sound because neither
    source it scores was fit to anything."""
    d = baseline["motion_response_diagnostic"]
    assert set(d["sources"]) == {"native_engine", "teacher_target"}
    assert d["n_pairs"] > 0


def test_the_baseline_records_that_nothing_was_trained(baseline):
    assert "No model was trained" in baseline["no_training_note"]


# --------------------------------------------------------------------------- #
# StructFormer across-seed variance — KB Decisions 57, 58
# --------------------------------------------------------------------------- #
def _variance():
    p = Path(ROOT) / "eval" / "reports" / "m6_structformer_variance.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def test_variance_artifact_self_identifies_and_is_generator_produced():
    """Decision 57's verdict change rests on this file. Decision 51's rule is that
    such an artifact needs a generator and a test; it had neither."""
    v = _variance()
    if v is None:
        pytest.skip("variance artifact not generated yet")
    for f in ("purpose", "pre_commitment", "generated_utc", "torch_version",
              "session_id", "style_id", "spread", "recomputed_gate"):
        assert f in v, f"missing {f!r}"
    assert (Path(ROOT) / "scripts" / "measure_structformer_variance.py").is_file()


def test_the_three_seeds_are_model_c_and_differ_only_by_seed():
    v = _variance()
    if v is None:
        pytest.skip("not generated")
    seeds = {k: r["seed"] for k, r in v["runs"].items() if "seed" in r}
    assert len(set(seeds.values())) == len(seeds) == 3, f"expected 3 distinct seeds, got {seeds}"


def test_the_corrected_margin_is_wider_not_narrower():
    """The pre-commitment was that the recomputed margin binds in either
    direction. It came out HARDER; a narrower bar would be threshold-shopping and
    must be visible if it ever happens."""
    v = _variance()
    if v is None:
        pytest.skip("not generated")
    c = v["comparison_to_cnn"]
    assert c["structformer_relative_sd"] > c["cnn_relative_sd"], (
        "the corrected margin is narrower than the CNN stand-in it replaced — that needs "
        "an explicit Decision entry, not a silent pass"
    )


def test_model_d_figure_is_read_from_the_gate_not_recomputed():
    """Two independent computations of one headline number is how two committed
    artifacts come to disagree (they did, by 0.025%). The variance artifact now
    reads Model D's figure from the gate artifact."""
    v = _variance()
    if v is None:
        pytest.skip("not generated")
    rg = v["recomputed_gate"]
    assert "read from" in rg["model_d_te_source"]
    gate = json.loads((Path(ROOT) / "eval" / "reports" / "m6_temporal_baseline.json")
                      .read_text(encoding="utf-8"))
    assert rg["model_d_te"] == gate["m6_gate"]["sources"]["m6_temporal_model_d"]["temporal_error"]["mean"]


def test_the_frozen_pre_registration_excludes_model_d():
    """Decision 57 claims the threshold was byte-identical before and after Model D
    was scored. That is only checkable if the pre-Model-D artifact is committed —
    it was not, and lived in a temp file (KB Decision 58)."""
    p = Path(ROOT) / "eval" / "reports" / "m6_temporal_gate_prereg.json"
    if not p.is_file():
        pytest.skip("frozen pre-registration not present")
    pre = json.loads(p.read_text(encoding="utf-8"))
    assert "m6_temporal_model_d" not in pre["m6_gate"]["sources"], (
        "the frozen pre-registration contains the model it pre-registers against"
    )
    live = json.loads((Path(ROOT) / "eval" / "reports" / "m6_temporal_baseline.json")
                      .read_text(encoding="utf-8"))
    assert pre["m6_gate"]["pre_registered_threshold"] == live["m6_gate"]["pre_registered_threshold"], (
        "the gate threshold changed after Model D was scored"
    )


# --------------------------------------------------------------------------- #
# M6 export + TensorRT — KB Decision 59
# --------------------------------------------------------------------------- #
def _export():
    p = Path(ROOT) / "eval" / "reports" / "m6_onnx_export.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def test_export_artifact_matches_its_generator_schema():
    e = _export()
    if e is None:
        pytest.skip("m6 export artifact not generated yet")
    gen = Path(ROOT) / "scripts" / "export_benchmark_m6.py"
    head = gen.read_text(encoding="utf-8").split("class _Encoder")[0]
    ns: dict = {}
    exec(compile(head, str(gen), "exec"), {"__name__": "gen"}, ns)
    missing = [k for k in ns["REQUIRED_TOP_LEVEL"] if k not in e]
    assert not missing, f"generator declares keys the artifact lacks: {missing}"
    assert set(e["operator_allow_list"]) == set(ns["ALLOWED_OPS"])
    for name, st in e["stages"].items():
        unlisted = set(st["op_types"]) - set(ns["ALLOWED_OPS"])
        assert not unlisted, f"{name}: unlisted operators {sorted(unlisted)}"


def test_both_graphs_export_build_and_run_with_static_shapes():
    e = _export()
    if e is None:
        pytest.skip("not generated")
    for name in ("encoder", "renderer"):
        st = e["stages"][name]
        assert st["onnx_export_ok"] and st["shapes_are_static"], name
        t = st["trt_fp32"]
        assert t["build_ok"] and t["run_ok"] and t["output_finite"], name
        assert t["latency_ms_mean"] > 0


def test_a_tolerance_exceedance_is_visible_not_waived():
    """KB Decision 59: the encoder stage exceeds the 1e-3 FP32 tolerance. The
    artifact must say so — a reader of the JSON alone must not see a green stage."""
    e = _export()
    if e is None:
        pytest.skip("not generated")
    for name, st in e["stages"].items():
        t = st["trt_fp32"]
        if not t.get("within_tolerance", True):
            assert "tolerance_note" in t, f"{name} exceeds tolerance with no note"
            assert "not waived" in t["tolerance_note"].lower()


def test_the_cross_frame_runtime_contract_is_recorded():
    """Model D needs state between frames; the M5 runtime was stateless. That is
    an M10 consequence and must be in the artifact, not only in prose."""
    e = _export()
    if e is None:
        pytest.skip("not generated")
    rc = e["runtime_contract"]
    assert rc["graphs"] == 2
    assert "prev_features" in rc["cross_frame_state"]
    assert "prev_features" in e["stages"]["renderer"]["inputs"]


def test_the_latency_cost_is_recorded_against_model_c():
    """Model D costs ~47% more per frame than Model C and sits at the top of
    §12's allocation. A rung that failed its criterion should not have that cost
    left unstated."""
    e = _export()
    if e is None:
        pytest.skip("not generated")
    lf = e["latency_finding"]
    assert lf["m6_per_frame_fp32_ms"] > lf["m5_single_graph_fp32_ms"]
    assert "NO FP16 ENGINE" in lf["note"], "the absent FP16 comparison must be stated"
