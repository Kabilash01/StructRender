"""Deliverable 4a — alignment validation accepts a genuinely aligned target set,
and stays inside the M2 validator's boundaries while doing it.

The M3 exit criterion ("target frames preserve intended scene layout
sufficiently for supervised training") is exactly the PASS verdict here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from data_pipeline.teacher.alignment import validate_alignment
from data_pipeline.teacher.metrics import Thresholds, frame_metrics, reference_edges, target_edges
from data_pipeline.teacher.structural import StructuralSession

from .conftest import load_target


def test_aligned_target_set_passes_and_gets_the_marker(targets, mini_session):
    report = validate_alignment(targets, structural_session_dir=mini_session)
    assert report.ok, [str(e) for e in report.errors]
    assert report.complete
    assert (targets / "ALIGNED").is_file()

    marker = json.loads((targets / "ALIGNED").read_text())
    assert marker["result"] == "PASS"
    assert marker["style_id"] == targets.parent.name
    assert marker["session_id"] == targets.name
    assert marker["teacher_version"]
    assert marker["thresholds"]["boundary_f"] == Thresholds.boundary_f
    assert marker["distributions"]["boundary_f"]["n"] == report.stats["frames_measured"]


def test_validator_writes_nothing_but_the_marker(targets, mini_session):
    """Report-only (KB §4 / M2 validator boundary): no repair, no regeneration."""
    before = {p.relative_to(targets): p.read_bytes()
              for p in targets.rglob("*") if p.is_file()}
    validate_alignment(targets, structural_session_dir=mini_session)
    after = {p.relative_to(targets): p.read_bytes()
             for p in targets.rglob("*") if p.is_file()}

    new = set(after) - set(before)
    assert new == {Path("ALIGNED")}
    for k in before:
        assert before[k] == after[k], f"validator mutated {k}"


def test_validator_never_touches_the_structural_session(targets, mini_session):
    before = {p.relative_to(mini_session): p.stat().st_mtime_ns
              for p in mini_session.rglob("*") if p.is_file()}
    validate_alignment(targets, structural_session_dir=mini_session)
    after = {p.relative_to(mini_session): p.stat().st_mtime_ns
             for p in mini_session.rglob("*") if p.is_file()}
    assert before == after


def test_limit_run_never_writes_the_marker(targets, mini_session):
    report = validate_alignment(targets, structural_session_dir=mini_session, frame_limit=4)
    assert report.stats["frames_measured"] == 4
    assert not report.complete
    assert not (targets / "ALIGNED").exists()


def test_no_marker_flag_is_honoured(targets, mini_session):
    report = validate_alignment(targets, structural_session_dir=mini_session, write_marker=False)
    assert report.ok
    assert not (targets / "ALIGNED").exists()


def test_revalidation_is_idempotent(targets, mini_session):
    a = validate_alignment(targets, structural_session_dir=mini_session)
    b = validate_alignment(targets, structural_session_dir=mini_session)
    assert a.ok and b.ok
    assert a.stats["frames_measured"] == b.stats["frames_measured"]
    assert (targets / "ALIGNED").is_file()


# --------------------------------------------------------------------------- #
# the metrics themselves
# --------------------------------------------------------------------------- #
def test_native_rgb_is_a_high_scoring_sanity_anchor(mini_session):
    """The engine's own RGB is by definition perfectly aligned with its own
    structural buffers, so it must score near the ceiling. If it does not, the
    reference edge map — not the teacher — is what is wrong."""
    session = StructuralSession(mini_session)
    th = Thresholds()
    for fid in (0, 6, 12):
        structure = session.load_structure(fid)
        m = frame_metrics(structure, session.load_native_rgb(fid), th)
        assert m["boundary_f"] > th.boundary_f, (fid, m["boundary_f"])
        assert m["spurious_edge_ratio"] <= th.spurious_ratio, (fid, m["spurious_edge_ratio"])


def test_reference_edge_map_is_engine_ground_truth_only(mini_session):
    """It must be derivable from the structural buffers alone — swapping the
    RGB cannot change it (that is what keeps the metric honest about a target
    that is *supposed* to look different)."""
    session = StructuralSession(mini_session)
    th = Thresholds()
    structure = session.load_structure(3)
    a = reference_edges(structure, th)
    b = reference_edges(structure, th)
    assert np.array_equal(a, b)
    assert a.dtype == bool
    assert a.sum() > 0
    # entity silhouettes are inside it
    eid = structure.entity_id
    assert (a & (eid != 0)).sum() > 0


def test_target_edge_thresholds_are_nested(targets):
    _thin, e_all, e_strong = target_edges(load_target(targets, 0), Thresholds())
    assert (e_strong & ~e_all).sum() == 0, "strong edges must be a subset of all edges"


def test_pure_noise_target_is_rejected_by_the_hallucination_gate(targets, mini_session):
    """A target with structure-free high-contrast detail everywhere is the
    §12 failure mode, and gate 2 is the guard for it."""
    from PIL import Image

    rng = np.random.default_rng(0)
    session = StructuralSession(mini_session)
    H, W = session.resolution[1], session.resolution[0]
    noise = rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)
    m = frame_metrics(session.load_structure(0), noise, Thresholds())
    assert m["spurious_edge_ratio"] > Thresholds.spurious_ratio
