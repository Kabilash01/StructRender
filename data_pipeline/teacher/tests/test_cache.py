"""Deliverable 3 — cached styled targets.

KB §3C.5: "Teacher outputs are cached because regeneration is expensive."
The cache key is ``(session_id, frame_id, style_id, teacher_version, seed)``;
a second run must generate zero frames, an interrupted run must resume, and the
finished set must be frame-count- and id-aligned with the structural session.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from data_pipeline.teacher.cache import cache_key, target_rel_path
from data_pipeline.teacher.generate import generate_targets

from .conftest import read_jsonl, write_jsonl


def test_cache_key_depends_on_every_component():
    base = dict(session_id="ds_1_a", frame_id=3, style_id="s", teacher_version="v1", seed=1)
    k = cache_key(**base)
    for field, other in (
        ("session_id", "ds_2_b"), ("frame_id", 4), ("style_id", "t"),
        ("teacher_version", "v2"), ("seed", 2),
    ):
        assert cache_key(**{**base, field: other}) != k, f"key ignores {field}"
    assert cache_key(**base) == k  # stable


def test_second_run_generates_zero_frames(mini_session, ci_style, tmp_path):
    first = generate_targets(mini_session, ci_style, out_root=tmp_path)
    assert first.generated > 0
    assert first.complete

    second = generate_targets(mini_session, ci_style, out_root=tmp_path)
    assert second.generated == 0
    assert second.cache_hits == first.generated
    assert second.complete


def test_target_count_equals_structural_frame_count_and_is_id_aligned(mini_session, ci_style, tmp_path):
    report = generate_targets(mini_session, ci_style, out_root=tmp_path)
    header = json.loads((mini_session / "manifest.json").read_text())
    tdir = Path(report.target_dir)

    rows = read_jsonl(tdir / "manifest.jsonl")
    assert len(rows) == header["frame_count"]
    assert [r["frame_id"] for r in rows] == list(range(header["frame_count"]))
    for r in rows:
        assert r["target_rgb_path"] == target_rel_path(r["frame_id"])
        assert (tdir / r["target_rgb_path"]).is_file()

    pngs = sorted(p.name for p in (tdir / "frames").glob("*.png"))
    assert pngs == [f"frame_{i:05d}_target.png" for i in range(header["frame_count"])]


def test_every_target_is_the_structural_resolution(mini_session, ci_style, tmp_path):
    from PIL import Image

    report = generate_targets(mini_session, ci_style, out_root=tmp_path)
    W, H = json.loads((mini_session / "manifest.json").read_text())["resolution"]
    for png in (Path(report.target_dir) / "frames").glob("*.png"):
        with Image.open(png) as im:
            assert im.size == (W, H), png.name


def test_resume_after_interrupt(mini_session, ci_style, tmp_path):
    """Simulate a kill mid-run: keep the first 5 target PNGs and their manifest
    lines, drop the rest, then re-run. Only the missing frames are generated."""
    full = generate_targets(mini_session, ci_style, out_root=tmp_path)
    tdir = Path(full.target_dir)
    total = full.generated

    rows = read_jsonl(tdir / "manifest.jsonl")
    write_jsonl(tdir / "manifest.jsonl", rows[:5])
    for r in rows[5:]:
        (tdir / r["target_rgb_path"]).unlink()
    (tdir / "manifest.json").unlink()

    resumed = generate_targets(mini_session, ci_style, out_root=tmp_path)
    assert resumed.cache_hits == 5
    assert resumed.generated == total - 5
    assert resumed.complete
    assert [r["frame_id"] for r in read_jsonl(tdir / "manifest.jsonl")] == list(range(total))


def test_resume_tolerates_a_torn_final_manifest_line(mini_session, ci_style, tmp_path):
    full = generate_targets(mini_session, ci_style, out_root=tmp_path)
    tdir = Path(full.target_dir)
    p = tdir / "manifest.jsonl"
    rows = read_jsonl(p)
    p.write_text("\n".join(json.dumps(r, sort_keys=True) for r in rows[:-1])
                 + '\n{"schema_version": "t1.0", "frame_i')

    resumed = generate_targets(mini_session, ci_style, out_root=tmp_path)
    assert resumed.generated == 1
    assert resumed.cache_hits == full.generated - 1
    assert len(read_jsonl(p)) == full.generated


def test_changing_the_style_config_invalidates_the_cache(mini_session, ci_style, tmp_path):
    """An edited prompt/param must not be served from a stale cache entry: the
    style config digest is folded into ``teacher_version``, which is in the key."""
    first = generate_targets(mini_session, ci_style, out_root=tmp_path, limit=4)
    edited = replace(ci_style, prompt=ci_style.prompt + " (v2)", config_digest="a" * 64)

    second = generate_targets(mini_session, edited, out_root=tmp_path, limit=4)
    assert second.teacher_version != first.teacher_version
    assert second.cache_hits == 0
    assert second.generated == 4


def test_generation_refuses_an_unvalidated_structural_session(mini_session_copy, ci_style, tmp_path):
    (mini_session_copy / "VALIDATED").unlink()
    report = generate_targets(mini_session_copy, ci_style, out_root=tmp_path)
    assert not report.ok
    assert any("VALIDATED" in e for e in report.errors)
    assert not (tmp_path / ci_style.style_id).exists()

    allowed = generate_targets(mini_session_copy, ci_style, out_root=tmp_path,
                               allow_unvalidated=True, limit=2)
    assert allowed.ok


def test_generation_never_writes_into_the_structural_session(mini_session_copy, ci_style, tmp_path):
    """KB Decision 31: the m2.0 session is never mutated, re-recorded or duplicated."""
    before = {p.relative_to(mini_session_copy): p.stat().st_mtime_ns
              for p in mini_session_copy.rglob("*") if p.is_file()}
    generate_targets(mini_session_copy, ci_style, out_root=tmp_path)
    after = {p.relative_to(mini_session_copy): p.stat().st_mtime_ns
             for p in mini_session_copy.rglob("*") if p.is_file()}
    assert before == after


def test_force_regenerates_on_a_hit(mini_session, ci_style, tmp_path):
    generate_targets(mini_session, ci_style, out_root=tmp_path, limit=3)
    forced = generate_targets(mini_session, ci_style, out_root=tmp_path, limit=3, force=True)
    assert forced.generated == 3
    assert forced.cache_hits == 0


def test_partial_run_is_marked_incomplete(mini_session, ci_style, tmp_path):
    report = generate_targets(mini_session, ci_style, out_root=tmp_path, stride=4, limit=3)
    assert not report.complete
    header = json.loads((Path(report.target_dir) / "manifest.json").read_text())
    assert header["complete"] is False
    assert [r["frame_id"] for r in read_jsonl(Path(report.target_dir) / "manifest.jsonl")] == [0, 4, 8]
