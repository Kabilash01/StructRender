"""Deliverable 4b — synthetically-misaligned target sets are REJECTED.

Mirrors ``data_pipeline/validation/tests/test_rejects.py``: every test copies a
real generated target set, breaks exactly one thing, and asserts the alignment
validator flags that specific fault and returns FAIL. Nothing corrupted is
committed.

The geometric corruptions (shift / rotate / frame swap / pasted object) are the
ones that matter: a target set can be perfectly well-formed as *files* and still
be useless as supervision, and the M3 exit criterion is precisely about that.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from data_pipeline.teacher.alignment import validate_alignment

from .conftest import error_codes, load_target, patch_header, read_jsonl, save_target, write_jsonl


def _run(targets, mini_session, **kw):
    report = validate_alignment(targets, structural_session_dir=mini_session,
                                write_marker=True, **kw)
    assert not report.ok, "expected the target set to be rejected"
    assert not (targets / "ALIGNED").exists(), "a rejected set must not get the marker"
    return report


# --------------------------------------------------------------------------- #
# geometric misalignment — the point of the milestone
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("k", [3, 5, 8])
def test_target_shifted_by_k_px_is_rejected(targets, mini_session, k):
    """A whole-frame translation keeps every pixel value and destroys alignment.
    Beyond the 2 px matching tolerance it must fail."""
    for fid in range(6):
        img = load_target(targets, fid)
        save_target(targets, fid, np.roll(np.roll(img, k, axis=1), k, axis=0))
    codes = error_codes(_run(targets, mini_session))
    assert codes & {"align.boundary_f", "align.spurious_edges", "align.entity_silhouette"}


@pytest.mark.parametrize("k", [8, 12])
def test_purely_horizontal_shift_is_rejected(targets, mini_session, k):
    """Documented worst case for a boundary metric on this scene: the dominant
    structure (the ground/sky horizon) is a long horizontal line and therefore
    invariant under a purely horizontal translation, so recall barely moves.

    Measured detection floor on the seed scene (recorded in KB Decision 34):
    **3 px** for a diagonal shift, **8 px** for an x-only shift (at 5 px x-only
    the worst frame reaches F=0.717 / spurious=0.294 / silhouette=0.630 — inside
    every gate).
    Smaller x-only shifts are deliberately not asserted here rather than
    weakening the gates until they happen to pass."""
    for fid in range(6):
        img = load_target(targets, fid)
        save_target(targets, fid, np.roll(img, k, axis=1))
    codes = error_codes(_run(targets, mini_session))
    assert codes & {"align.boundary_f", "align.spurious_edges", "align.entity_silhouette"}


def test_target_rotated_is_rejected(targets, mini_session):
    for fid in range(6):
        img = Image.fromarray(load_target(targets, fid))
        rotated = img.rotate(4.0, resample=Image.BILINEAR, fillcolor=(200, 200, 200))
        save_target(targets, fid, np.asarray(rotated, dtype=np.uint8))
    codes = error_codes(_run(targets, mini_session))
    assert "align.boundary_f" in codes or "align.entity_silhouette" in codes


def test_two_frames_targets_swapped_is_rejected(targets, mini_session):
    """Frame-id misalignment: both files are valid targets of *this* session,
    just attached to the wrong frames. Nothing but the structural comparison can
    catch this."""
    a, b = 0, 23
    ia, ib = load_target(targets, a), load_target(targets, b)
    save_target(targets, a, ib)
    save_target(targets, b, ia)
    codes = error_codes(_run(targets, mini_session))
    assert codes & {"align.boundary_f", "align.entity_silhouette", "align.spurious_edges"}


def test_hallucinated_object_is_rejected(targets, mini_session):
    """KB §12 *Teacher Structural Distortion*: the teacher invents geometry that
    the engine never had. Gate 2 is the guard."""
    for fid in range(6):
        img = load_target(targets, fid).copy()
        # a high-contrast object pasted into empty sky, with no structural support
        img[8:60, 300:420] = 20
        img[14:54, 306:414] = 235
        img[20:48, 312:408] = 40
        save_target(targets, fid, img)
    codes = error_codes(_run(targets, mini_session))
    assert "align.spurious_edges" in codes


def test_entity_silhouette_erased_is_rejected(targets, mini_session):
    """Layout mostly survives but one entity's outline is painted over — the
    per-entity gate exists because a frame-average F-measure hides this."""
    from data_pipeline.teacher.structural import StructuralSession

    session = StructuralSession(mini_session)
    hit = 0
    for fid in range(24):
        structure = session.load_structure(fid)
        ids, counts = np.unique(structure.entity_id[structure.entity_id != 0], return_counts=True)
        if not len(ids):
            continue
        big = int(ids[int(np.argmax(counts))])
        mask = structure.entity_id == big
        ys, xs = np.where(mask)
        img = load_target(targets, fid).copy()
        # flood the entity and a 6 px collar around it with a flat fill
        y0, y1 = max(0, ys.min() - 6), min(img.shape[0], ys.max() + 7)
        x0, x1 = max(0, xs.min() - 6), min(img.shape[1], xs.max() + 7)
        img[y0:y1, x0:x1] = 190
        save_target(targets, fid, img)
        hit += 1
    assert hit, "expected visible entities in the mini session"
    assert "align.entity_silhouette" in error_codes(_run(targets, mini_session))


def test_flat_target_is_rejected(targets, mini_session):
    """A teacher that collapses the frame keeps perfect 'precision' (it draws no
    wrong edges) — recall is what rejects it."""
    for fid in range(24):
        save_target(targets, fid, np.full_like(load_target(targets, fid), 180))
    codes = error_codes(_run(targets, mini_session))
    assert "align.boundary_f" in codes


def test_passthrough_native_frame_is_rejected(targets, mini_session):
    """The degenerate teacher: copy the native frame through unchanged.

    It is the exact opposite failure to ``test_flat_target``. Structure survives
    *perfectly* — measured on the seed session an untouched native frame scores
    F=0.976 with a spurious ratio of 0.000, beating every real target — so gates
    1-3 all pass and "turn the conditioning up until it passes" converges on a
    teacher that does nothing. Only the stylisation gate rejects it.
    """
    from data_pipeline.teacher.structural import StructuralSession

    session = StructuralSession(mini_session)
    for fid in range(24):
        save_target(targets, fid, session.load_native_rgb(fid))

    codes = error_codes(_run(targets, mini_session))
    assert "align.not_stylised" in codes
    # and it must be THAT gate doing the work, not a structural one
    assert "align.boundary_f" not in codes
    assert "align.spurious_edges" not in codes


# --------------------------------------------------------------------------- #
# structural / bookkeeping faults
# --------------------------------------------------------------------------- #
def test_missing_target_frame_is_rejected(targets, mini_session):
    (targets / "frames" / "frame_00005_target.png").unlink()
    assert "artifact.target_missing" in error_codes(_run(targets, mini_session))


def test_short_target_set_is_rejected(targets, mini_session):
    rows = read_jsonl(targets / "manifest.jsonl")
    dropped = rows.pop()
    (targets / "frames" / dropped["target_rgb_path"].split("/")[-1]).unlink()
    write_jsonl(targets / "manifest.jsonl", rows)
    assert "frame.count_mismatch" in error_codes(_run(targets, mini_session))


def test_resized_target_is_rejected(targets, mini_session):
    Image.fromarray(load_target(targets, 4)).resize((256, 144)).save(
        targets / "frames" / "frame_00004_target.png")
    assert "artifact.target_size" in error_codes(_run(targets, mini_session))


def test_undecodable_target_is_rejected(targets, mini_session):
    (targets / "frames" / "frame_00004_target.png").write_bytes(b"not a png")
    assert "artifact.target_decode" in error_codes(_run(targets, mini_session))


def test_renamed_target_breaks_the_id_contract(targets, mini_session):
    """``frame_<id>_target.png`` *is* the frame-id alignment contract."""
    p = targets / "frames" / "frame_00004_target.png"
    p.rename(targets / "frames" / "frame_00004_target_v2.png")
    codes = error_codes(_run(targets, mini_session))
    assert "layout.frames_stray" in codes
    assert "artifact.target_missing" in codes


def test_frame_id_not_in_structural_session_is_rejected(targets, mini_session):
    rows = read_jsonl(targets / "manifest.jsonl")
    rows[-1]["frame_id"] = 9999
    rows[-1]["target_rgb_path"] = "frames/frame_09999_target.png"
    write_jsonl(targets / "manifest.jsonl", rows)
    assert "frame.frame_id_unknown" in error_codes(_run(targets, mini_session))


def test_wrong_session_id_is_rejected(targets, mini_session):
    patch_header(targets, lambda h: h.__setitem__("session_id", "ds_1_deadbeef"))
    codes = error_codes(_run(targets, mini_session))
    assert "header.session_dir" in codes or "struct.session_mismatch" in codes


def test_style_id_not_matching_the_directory_is_rejected(targets, mini_session):
    patch_header(targets, lambda h: h.__setitem__("style_id", "some_other_style"))
    assert "header.style_dir" in error_codes(_run(targets, mini_session))


def test_style_ids_out_of_sync_is_rejected(targets, mini_session):
    patch_header(targets, lambda h: h.__setitem__("style_ids", ["native"]))
    assert "header.style_ids" in error_codes(_run(targets, mini_session))


def test_mixed_teacher_version_is_rejected(targets, mini_session):
    """A set half-regenerated with a different teacher is not a valid
    supervision source (KB §6: regenerate only required targets — but a set must
    end up internally consistent)."""
    rows = read_jsonl(targets / "manifest.jsonl")
    rows[3]["teacher_version"] = "some-other-teacher/0000"
    write_jsonl(targets / "manifest.jsonl", rows)
    assert "frame.teacher_version" in error_codes(_run(targets, mini_session))


def test_missing_teacher_metadata_is_rejected(targets, mini_session):
    patch_header(targets, lambda h: h["teacher"].pop("teacher_version"))
    assert "header.teacher_key" in error_codes(_run(targets, mini_session))


def test_stray_artifact_in_target_dir_is_rejected(targets, mini_session):
    (targets / "notes.txt").write_text("scratch")
    assert "layout.stray" in error_codes(_run(targets, mini_session))


def test_unresolvable_structural_session_is_rejected(targets, tmp_path, monkeypatch):
    monkeypatch.setenv("STRUCTRENDER_DATASET_ROOT", str(tmp_path / "empty"))
    report = validate_alignment(targets)
    assert not report.ok
    assert "struct.not_found" in error_codes(report)
    assert not (targets / "ALIGNED").exists()


def test_truncated_manifest_jsonl_is_rejected(targets, mini_session):
    p = targets / "manifest.jsonl"
    p.write_text(p.read_text() + '{"frame_id": 24, "sty')
    assert "frame_lines.parse" in error_codes(_run(targets, mini_session))


# --------------------------------------------------------------------------- #
# the tolerance knob is explicit, and off by default
# --------------------------------------------------------------------------- #
def test_max_fail_frac_default_is_strict(targets, mini_session):
    """One bad frame out of 24 rejects the whole set unless the operator
    explicitly widens the tolerance (mirrors --dropout-tolerance at M2)."""
    img = load_target(targets, 7).copy()
    img[:] = 180
    save_target(targets, 7, img)

    strict = validate_alignment(targets, structural_session_dir=mini_session, write_marker=False)
    assert not strict.ok
    assert strict.stats["frames_failing_a_gate"] == 1

    lenient = validate_alignment(targets, structural_session_dir=mini_session,
                                 max_fail_frac=0.5, write_marker=False)
    assert lenient.ok
    assert any(w.code == "align.within_tolerance" for w in lenient.warnings)


def test_report_json_records_the_per_frame_distribution(targets, mini_session):
    report = validate_alignment(targets, structural_session_dir=mini_session, write_marker=False)
    assert len(report.per_frame) == report.stats["frames_measured"]
    d = report.stats["distributions"]["boundary_f"]
    assert d["n"] == report.stats["frames_measured"]
    assert 0.0 <= d["min"] <= d["median"] <= d["max"] <= 1.0
    assert json.dumps(report.stats)  # serialisable for --report-json
