"""Exit criterion (1): corrupted sessions are rejected.

Every test copies the real session (via the ``session`` fixture — a 213-frame
slice), mutates exactly one thing, and asserts the validator flags that specific
fault and returns a FAIL verdict. No mutated fixture is committed.
"""

from __future__ import annotations

import json

import numpy as np
from PIL import Image

from data_pipeline.validation import validate_session

from .conftest import (
    error_codes,
    load_npz,
    patch_frame_line,
    patch_header,
    read_jsonl,
    rewrite_npz,
    write_jsonl,
)


def _run(session):
    report = validate_session(session, write_marker=False)
    assert not report.ok
    assert not (session / "VALIDATED").exists()
    return report


# --------------------------------------------------------------------------- #
# schema versioning
# --------------------------------------------------------------------------- #
def test_schema_version_missing_in_header(session):
    patch_header(session, lambda h: h.pop("schema_version"))
    assert "header.schema_version" in error_codes(_run(session))


def test_schema_version_line_mismatch(session):
    patch_frame_line(session, 5, lambda r: r.__setitem__("schema_version", "m2.1"))
    assert "frame.schema_mismatch" in error_codes(_run(session))


def test_schema_version_too_old(session):
    patch_header(session, lambda h: h.__setitem__("schema_version", "m1.9"))
    assert "header.schema_too_old" in error_codes(_run(session))


# --------------------------------------------------------------------------- #
# frame_id integrity
# --------------------------------------------------------------------------- #
def test_frame_id_not_zero_based(session):
    patch_frame_line(session, 0, lambda r: r.__setitem__("frame_id", 7))
    assert "frame.frame_id_base" in error_codes(_run(session))


def test_frame_id_not_consecutive(session):
    rows = read_jsonl(session / "manifest.jsonl")
    del rows[100]
    write_jsonl(session / "manifest.jsonl", rows)
    assert "frame.frame_id_consecutive" in error_codes(_run(session))


def test_last_frame_id_mismatch(session):
    rows = read_jsonl(session / "manifest.jsonl")
    del rows[-1]
    write_jsonl(session / "manifest.jsonl", rows)
    assert "frame.last_id" in error_codes(_run(session))


def test_truncated_manifest_jsonl(session):
    p = session / "manifest.jsonl"
    p.write_text(p.read_text() + '{"frame_id": 213, "structu')
    assert "frame_lines.parse" in error_codes(_run(session))


# --------------------------------------------------------------------------- #
# npz members: existence / dtype / shape / readability
# --------------------------------------------------------------------------- #
def test_npz_member_missing(session):
    members = load_npz(session, 50)
    members.pop("depth")
    rewrite_npz(session, 50, members)
    assert "buf.member_missing" in error_codes(_run(session))


def test_npz_member_wrong_dtype(session):
    members = load_npz(session, 50)
    members["depth"] = members["depth"].astype(np.float32)
    rewrite_npz(session, 50, members)
    assert "buf.dtype" in error_codes(_run(session))


def test_npz_member_wrong_shape(session):
    members = load_npz(session, 50)
    members["normal"] = members["normal"][:, :, :2]
    rewrite_npz(session, 50, members)
    assert "buf.shape" in error_codes(_run(session))


def test_npz_unreadable(session):
    (session / "frames" / "frame_00050.npz").write_bytes(b"not a zip file at all")
    assert "buf.npz_unreadable" in error_codes(_run(session))


def test_structure_path_unresolved(session):
    npz = session / "frames" / "frame_00050.npz"
    npz.rename(npz.with_suffix(".npz.bak"))
    assert "artifact.npz_missing" in error_codes(_run(session))


# --------------------------------------------------------------------------- #
# native RGB
# --------------------------------------------------------------------------- #
def test_native_rgb_missing(session):
    (session / "frames" / "frame_00050_rgb.png").unlink()
    assert "artifact.rgb_missing" in error_codes(_run(session))


def test_native_rgb_wrong_size(session):
    Image.new("RGB", (100, 100), (120, 120, 120)).save(session / "frames" / "frame_00050_rgb.png")
    assert "artifact.rgb_size" in error_codes(_run(session))


def test_native_rgb_all_black(session):
    Image.new("RGB", (512, 288), (0, 0, 0)).save(session / "frames" / "frame_00050_rgb.png")
    assert "artifact.rgb_black" in error_codes(_run(session))


# --------------------------------------------------------------------------- #
# camera
# --------------------------------------------------------------------------- #
def test_camera_matrix_not_16_floats(session):
    p = session / "frames" / "frame_00050_camera.json"
    cam = json.loads(p.read_text())
    cam["view_matrix"] = cam["view_matrix"][:15]
    p.write_text(json.dumps(cam))
    assert "artifact.camera_matrix" in error_codes(_run(session))


def test_camera_sidecar_missing(session):
    (session / "frames" / "frame_00050_camera.json").unlink()
    assert "artifact.camera_missing" in error_codes(_run(session))


def test_camera_inline_matrix_not_16(session):
    patch_frame_line(session, 50, lambda r: r["camera"].__setitem__(
        "projection_matrix", r["camera"]["projection_matrix"][:12]))
    assert "frame.camera_matrix" in error_codes(_run(session))


# --------------------------------------------------------------------------- #
# partial channel dropout (KB M2 notes / dataset_manifest.md §6.1)
# --------------------------------------------------------------------------- #
def test_partial_dropout_zeroed_normal_patch(session):
    members = load_npz(session, 80)
    members["normal"][120:150, 200:240, :] = 0.0
    rewrite_npz(session, 80, members)
    assert "buf.partial_dropout" in error_codes(_run(session))


def test_partial_dropout_inf_depth_on_entity_pixels(session):
    members = load_npz(session, 80)
    ys, xs = np.where(members["entity_id"] != 0)
    assert ys.size > 20, "expected visible entity geometry in this frame"
    sel = slice(0, 40)
    members["depth"][ys[sel], xs[sel], 0] = np.float16(np.inf)
    rewrite_npz(session, 80, members)
    assert "buf.partial_dropout" in error_codes(_run(session))


# --------------------------------------------------------------------------- #
# motion_valid semantics (frame_structure.md §4)
# --------------------------------------------------------------------------- #
def test_motion_valid_true_on_frame_zero(session):
    patch_frame_line(session, 0, lambda r: r.__setitem__("motion_valid", True))
    codes = error_codes(_run(session))
    assert "frame.motion_valid_frame0" in codes
    # frame 0 is also the first frame of temporal window temporal_0
    assert "seq.temporal_motion_valid" in codes


def test_temporal_window_first_frame_motion_valid(session):
    # carve a second temporal window out of the reappearance span and give its
    # first frame a (wrong) motion_valid=true
    def rewire(h):
        h["sequences"] = [
            {"sequence_id": "temporal_0", "sequence_kind": "temporal", "frame_range": [0, 159]},
            {"sequence_id": "temporal_1", "sequence_kind": "temporal", "frame_range": [160, 212]},
        ]
    patch_header(session, rewire)
    patch_frame_line(session, 160, lambda r: r.__setitem__("motion_valid", True))
    assert "seq.temporal_motion_valid" in error_codes(_run(session))


def test_motion_valid_false_with_nonzero_motion(session):
    patch_frame_line(session, 50, lambda r: r.__setitem__("motion_valid", False))
    assert "buf.motion_valid_false_nonzero" in error_codes(_run(session))


# --------------------------------------------------------------------------- #
# reappearance sequence integrity
# --------------------------------------------------------------------------- #
def _mutate_reap(header, key, value):
    for s in header["sequences"]:
        if s.get("sequence_id") == "reap_camera_away_0.5s":
            s[key] = value


def test_reappearance_last_seen_frame_tampered(session):
    patch_header(session, lambda h: _mutate_reap(h, "last_seen_frame", 170))
    assert "reap.last_seen" in error_codes(_run(session))


def test_reappearance_unseen_seconds_tampered(session):
    patch_header(session, lambda h: _mutate_reap(h, "unseen_seconds", 0.4))
    assert "reap.unseen_seconds" in error_codes(_run(session))


def test_reappearance_unseen_frames_tampered(session):
    patch_header(session, lambda h: _mutate_reap(h, "unseen_frames", 42))
    assert "reap.unseen_frames" in error_codes(_run(session))


def test_reappearance_generation_after_tampered(session):
    patch_header(session, lambda h: _mutate_reap(h, "generation_after", 1))
    assert "reap.gen_after" in error_codes(_run(session))


# --------------------------------------------------------------------------- #
# debug-log vs dataset-manifest separation (dataset_manifest.md §6.2)
# --------------------------------------------------------------------------- #
def test_session_id_debug_namespace(session):
    patch_header(session, lambda h: h.__setitem__("session_id", "session_1788430292"))
    assert "header.session_id_namespace" in error_codes(_run(session))


def test_gamelog_style_leak_in_session_dir(session):
    (session / "logs").mkdir()
    assert "layout.debug_leak" in error_codes(_run(session))


def test_stray_artifact_in_session_dir(session):
    (session / "notes.txt").write_text("scratch")
    assert "layout.stray" in error_codes(_run(session))


# --------------------------------------------------------------------------- #
# events.jsonl consistency
# --------------------------------------------------------------------------- #
def test_events_spawn_event_removed(session):
    p = session / "events.jsonl"
    kept = [l for l in p.read_text().splitlines()
            if l.strip() and json.loads(l)["event"] != "spawn"]
    p.write_text("\n".join(kept) + "\n")
    assert "events.spawn_missing" in error_codes(_run(session))


def test_events_hidden_event_frame_shifted(session):
    p = session / "events.jsonl"
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    for r in rows:
        if r["event"] == "hidden":
            r["frame_id"] += 3   # no longer the real visible->hidden transition
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert "events.vis_no_transition" in error_codes(_run(session))


# --------------------------------------------------------------------------- #
# sequence coverage
# --------------------------------------------------------------------------- #
def test_sequence_coverage_gap(session):
    def shrink(h):
        for s in h["sequences"]:
            if s.get("sequence_id") == "temporal_0":
                s["frame_range"] = [0, 150]
    patch_header(session, shrink)
    assert "seq.coverage" in error_codes(_run(session))


def test_sequence_overlap(session):
    def widen(h):
        for s in h["sequences"]:
            if s.get("sequence_id") == "reap_camera_away_0.5s":
                s["frame_range"] = [140, 212]
    patch_header(session, widen)
    assert "seq.overlap" in error_codes(_run(session))
