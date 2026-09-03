"""Exit criterion (2): valid sessions replay correctly."""

from __future__ import annotations

import json

from data_pipeline.validation import validate_session


def _manifest(session) -> dict:
    return json.loads((session / "manifest.json").read_text())


def test_mini_session_passes(session):
    m = _manifest(session)
    report = validate_session(session, write_marker=False)
    assert report.ok, [str(e) for e in report.errors]
    assert report.warnings == []
    assert report.frame_count == m["frame_count"]
    assert report.replayable


def test_pristine_session_passes_and_gets_marker(session):
    m = _manifest(session)
    report = validate_session(session, write_marker=True)
    assert report.ok, [str(e) for e in report.errors]

    marker = session / "VALIDATED"
    assert marker.exists()
    payload = json.loads(marker.read_text())
    assert payload["result"] == "PASS"
    assert payload["validator_version"] == "m2-validator-1"
    assert payload["session_id"] == m["session_id"]
    assert payload["frame_count"] == m["frame_count"]
    assert "layout" in payload["checks_run"] and "frame_buffers" in payload["checks_run"]


def test_revalidation_is_idempotent(session):
    first = validate_session(session, write_marker=True)
    assert first.ok
    marker_before = (session / "VALIDATED").read_text()
    second = validate_session(session, write_marker=True)
    assert second.ok
    # re-validating a VALIDATED session is fine; the marker is simply rewritten
    assert (session / "VALIDATED").exists()
    assert json.loads(marker_before)["result"] == "PASS"


def test_limit_run_does_not_write_marker(session):
    report = validate_session(session, frame_limit=10, write_marker=True)
    assert report.ok
    assert not report.replayable
    assert not (session / "VALIDATED").exists()


def test_full_real_session_replays_and_marks(real_session_copy):
    """The whole seed session must pass clean (exit criterion 2)."""
    m = _manifest(real_session_copy)
    report = validate_session(real_session_copy, write_marker=True)
    assert report.ok, [str(e) for e in report.errors]
    assert report.frame_count == m["frame_count"]
    assert report.stats["frames_buffer_checked"] == m["frame_count"]
    assert report.stats["frames_with_dropout"] == 0
    assert (real_session_copy / "VALIDATED").exists()
