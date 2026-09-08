"""Every committed report names the data and objective it came from.

KB Decision 42. This test exists because the rule was stated in the Decision Log,
the fields were added to the code, all four reports were manually checked, and
**two of them still carried wrong values** — `m4_metrics.json` said its objective
was `"L_rec = L1 + 0"` (a string-splitting bug that cut at the decimal point of
`0.5*LPIPS`, so the artifact claimed an L1-only run — the configuration Decision
41 records as FAILING the sharpness gate), and `m4_onnx_export.json` recorded
`"session_id": "discovered-newest"`, a sentinel meaning "did not look".

Both survived a 16/16 green suite and a manual four-field check, because
presence was verified and correctness was not. A confidently wrong provenance
field is worse than a missing one: a missing field prompts a question, a wrong
one closes it.

So this asserts the fields are present, non-placeholder, and *mutually
consistent* — every report must agree about which session, style and objective
produced it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from renderer.training.train_baseline import ROOT, objective_string

# M5's reports are here too: Decision 44 recorded that an artifact nothing asserts
# is unfalsifiable evidence, and leaving the M5 set uncovered would have repeated
# that finding one milestone after writing it down. The objective cross-check below
# is per-config, since M5 ships two deliberate attempts (KB Decision 49).
REPORTS = ["m4_train.json", "m4_metrics.json", "m4_overfit.json", "m4_onnx_export.json"]
M5_REPORTS = ["m5_train.json", "m5_metrics.json", "m5_overfit.json", "m5_onnx_export.json",
              "m5_train_warmup.json", "m5_metrics_warmup.json"]
REQUIRED = ("session_id", "style_id", "objective", "generated_utc")

# Values that are technically "present" but carry no information. The sentinel
# below was real: it recorded that the writer never resolved the session.
PLACEHOLDERS = {"", "none", "null", "unknown", "discovered-newest", "n/a", "tbd"}


def _load(name: str) -> dict | None:
    p = Path(ROOT) / "eval" / "reports" / name
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


@pytest.mark.parametrize("name", REPORTS)
def test_report_carries_non_placeholder_provenance(name):
    report = _load(name)
    if report is None:
        pytest.skip(f"{name} not generated yet")
    for field in REQUIRED:
        assert field in report, f"{name} is missing '{field}' (KB Decision 42)"
        value = report[field]
        assert isinstance(value, str) and value.strip(), f"{name}: '{field}' is empty"
        assert value.strip().lower() not in PLACEHOLDERS, (
            f"{name}: '{field}' is the placeholder {value!r} — a report that records "
            "it did not look cannot distinguish a current run from a retired one"
        )


def test_reports_agree_on_the_objective(cfg):
    """A wrong objective is the defect that motivated this file.

    Cross-checked against the config rather than against each other, so three
    reports agreeing on the same wrong string cannot pass.
    """
    expected = objective_string(cfg)
    for name in REPORTS:
        report = _load(name)
        if report is None or "objective" not in report:
            continue
        assert report["objective"] == expected, (
            f"{name} claims objective {report['objective']!r} but the config specifies "
            f"{expected!r} — Decision 41's finding is that the objective drove the "
            "sharpness/SSIM trade, so a report that misnames it is evidence for a run "
            "that did not happen"
        )


def test_reports_agree_on_session_and_style():
    seen: dict[str, set] = {"session_id": set(), "style_id": set()}
    for name in REPORTS:
        report = _load(name)
        if report is None:
            continue
        for field in seen:
            if field in report:
                seen[field].add(report[field])
    for field, values in seen.items():
        assert len(values) <= 1, (
            f"committed reports disagree on '{field}': {sorted(values)} — they cannot "
            "all describe the same run"
        )


def test_committed_overfit_artifact_meets_the_mandated_gate(cfg):
    """The committed overfit report must itself show a passing §11 gate.

    ``test_overfit.py`` asserts against a *freshly computed* result and skips
    entirely without CUDA and >=4 GB VRAM, and the provenance checks above only
    look at who produced a report, not what it says. So a committed
    ``m4_overfit.json`` recording ``n_frames: 8`` — which is precisely the §11
    violation this project already shipped once — passes the whole suite on any
    machine without a GPU.

    §11's 32-128 range is held as a literal below because it is a KB constant and
    not a tunable, so relaxing N is caught outright. The four thresholds are read
    from the config and the artifact must have been scored against exactly those.

    Residual, stated rather than left to be found: **a simultaneous relaxation
    passes.** Lower `min_psnr_db` in the config AND rescore the artifact against
    it, and both `th == want` and the threshold assertions hold. Nothing here can
    catch that — a pre-registered threshold is only pre-registered if someone
    reads the config diff. That is the review step this test does not replace.
    """
    report = _load("m4_overfit.json")
    if report is None:
        pytest.skip("m4_overfit.json not generated yet")

    n = report["n_frames"]
    assert 32 <= n <= 128, (
        f"committed overfit artifact used N={n}; KB §11 item 4 mandates 32-128 frames "
        "('mandatory to protect the compute budget') and N=8 is strictly easier to memorise"
    )
    assert n == cfg["overfit"]["n_frames"], "artifact's N disagrees with the shipped config"

    th, want = report["thresholds"], cfg["overfit"]["thresholds"]
    assert th == want, f"artifact was scored against {th}, config pre-registers {want}"

    assert report["psnr_db"] >= th["min_psnr_db"]
    assert report["final_l1"] <= th["max_final_l1"]
    assert report["loss_drop_factor"] >= th["min_loss_drop_factor"]
    assert report["ssim"] >= th["min_ssim"]


@pytest.mark.parametrize("name", M5_REPORTS)
def test_m5_report_carries_non_placeholder_provenance(name):
    """Same rule as M4's, applied to the milestone that generated six new
    artifacts. KB Decision 42."""
    report = _load(name)
    if report is None:
        pytest.skip(f"{name} not generated yet")
    for field in REQUIRED:
        assert field in report, f"{name} is missing '{field}' (KB Decision 42)"
        value = report[field]
        assert isinstance(value, str) and value.strip(), f"{name}: '{field}' is empty"
        assert value.strip().lower() not in PLACEHOLDERS, f"{name}: '{field}' is a placeholder"


def test_m5_reports_agree_on_session_style_and_objective():
    """Both M5 attempts must describe the same data and the same objective —
    Decision 41 mandates one objective for every ladder rung, so two attempts
    disagreeing here would mean the comparison measured the loss, not warmup."""
    seen = {"session_id": set(), "style_id": set(), "objective": set()}
    for name in M5_REPORTS:
        report = _load(name)
        if report is None:
            continue
        for field in seen:
            if field in report:
                seen[field].add(report[field])
    for field, values in seen.items():
        assert len(values) <= 1, f"M5 reports disagree on '{field}': {sorted(values)}"


def test_the_two_m5_attempts_did_not_overwrite_each_other():
    """Decision 49 promises attempt 1's numbers stand regardless of attempt 2's
    outcome. That promise is only real if both artifacts exist separately."""
    a1, a2 = _load("m5_metrics.json"), _load("m5_metrics_warmup.json")
    if a1 is None or a2 is None:
        pytest.skip("both M5 attempts not present")
    assert a1["checkpoint"] != a2["checkpoint"], "both attempts point at one checkpoint"
    assert a1["selection_set"]["psnr_db"] != a2["selection_set"]["psnr_db"], (
        "the two attempts report identical metrics — one has overwritten the other"
    )
