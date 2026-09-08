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
# Translation-detection floors, re-measured on the enriched scene (KB Decision
# 41): 2 px diagonal, 5 px x-only — improved from 3 px / 8 px on the retired
# scene, whose dominant structure was a long horizontal horizon and therefore
# near-invariant under horizontal translation.
#
# These are ASSERTED here, not merely documented. Decision 34 claimed the floors
# were "pinned by test_purely_horizontal_shift_is_rejected"; they were not — the
# parametrisation sat well above the floor, so a regression back to 3 px / 8 px
# would have gone unnoticed. Each floor is tested at exactly its value.
DIAGONAL_FLOOR_PX = 2
X_ONLY_FLOOR_PX = 5


@pytest.mark.parametrize("k", [DIAGONAL_FLOOR_PX, 3, 5, 8])
def test_target_shifted_by_k_px_is_rejected(targets, mini_session, k):
    """A whole-frame translation keeps every pixel value and destroys alignment.

    Includes `k == DIAGONAL_FLOOR_PX`, so the measured floor is a regression
    test rather than a note: if the scene or the gates ever stop catching a 2 px
    diagonal shift, this fails.
    """
    for fid in range(6):
        img = load_target(targets, fid)
        save_target(targets, fid, np.roll(np.roll(img, k, axis=1), k, axis=0))
    codes = error_codes(_run(targets, mini_session))
    assert codes & {"align.boundary_f", "align.spurious_edges", "align.entity_silhouette"}


@pytest.mark.parametrize("k", [X_ONLY_FLOOR_PX, 8, 12])
def test_purely_horizontal_shift_is_rejected(targets, mini_session, k):
    """The hardest translation for a boundary metric on this scene.

    A long horizontal ground/sky boundary is near-invariant under horizontal
    translation, so recall barely moves and the x-only floor is the weakest
    number in the whole guard. Measured floors (KB Decision 41): **2 px
    diagonal, 5 px x-only**, improved from 3 px / 8 px once the enriched scene
    added vertical structure.

    `k == X_ONLY_FLOOR_PX` pins the floor. Shifts BELOW it are deliberately not
    asserted — a systematic sub-5 px horizontal misalignment still passes every
    gate, and that limitation is recorded rather than papered over by weakening
    a gate until it happens to catch one.
    """
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


# Smallest pasted object gate 2 reliably rejects, measured (KB Decision 43).
#
# The floor is **style-dependent**, because gate 2 is a RATIO: unsupported strong
# edges over TOTAL strong edges. A busier target style raises the denominator, so
# the same hallucination becomes a smaller fraction. Measured on the enriched
# scene, both caught 6/6 at the floor and unreliable below it:
#
#     oil_impasto  (real teacher, ~1800 strong edges/frame)  ->  6.5% of frame
#     ci_posterize (CI teacher,   ~3234 strong edges/frame)  -> 11.0% of frame
#
# These fixtures generate `ci_posterize` targets (no model download), so the
# value here is ITS floor, not the real teacher's. Sizing this from the
# oil_impasto measurement is exactly the mistake that made this test fail: the
# floor must come from the target set the test actually runs against.
HALLUCINATION_FLOOR_HW = (90, 180)


def test_hallucinated_object_is_rejected(targets, mini_session):
    """KB §12 *Teacher Structural Distortion*: the teacher invents geometry that
    the engine never had. Gate 2 is the guard.

    Two things this test had to learn the hard way:

    1. The region is located from the **depth buffer**, not hardcoded. It used
       fixed coordinates assuming empty sky; the enriched scene (KB Decision 38)
       put real geometry there, so the "hallucinated" object landed on a
       supported surface and the test correctly stopped firing. A
       structural-distortion test must define "unsupported" from the structure.
    2. The object is sized at the **measured** detection floor. Scene enrichment
       diluted gate 2's sensitivity to localised hallucination (see
       ``HALLUCINATION_FLOOR_HW``) — a smaller object silently evades it.
    """
    from data_pipeline.teacher.structural import StructuralSession

    session = StructuralSession(mini_session)
    painted = 0
    for fid in range(6):
        structure = session.load_structure(fid)
        bg = ~np.isfinite(structure.depth)          # +inf background: no geometry at all
        ys, xs = np.where(bg)
        if ys.size < 4000:                          # need room for an object with a margin
            continue
        h, w = HALLUCINATION_FLOOR_HW
        y0 = max(0, min(int(np.median(ys)) - h // 2, structure.depth.shape[0] - h))
        x0 = max(0, min(int(np.median(xs)) - w // 2, structure.depth.shape[1] - w))
        img = load_target(targets, fid).copy()
        img[y0:y0 + h, x0:x0 + w] = 20
        img[y0 + 6:y0 + h - 6, x0 + 6:x0 + w - 6] = 235
        img[y0 + 12:y0 + h - 12, x0 + 12:x0 + w - 12] = 40
        save_target(targets, fid, img)
        painted += 1
    assert painted, "no frame had a large enough background region to paste into"
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
