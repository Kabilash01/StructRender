"""Deliverable 1 — teacher adapter contract.

Shape / dtype / frame-id alignment, bit-exactness of the deterministic backend
across two runs, and a ``--dry-run`` that walks the whole seed session without
generating anything (KB §6 "Replace the Offline Teacher": the adapter is the
swap point, so its contract is what keeps a replacement honest).
"""

from __future__ import annotations

import numpy as np
import pytest

from data_pipeline.teacher.adapter import (
    FrameStructure,
    get_backend_class,
    make_backend,
    resolve_teacher_version,
)
from data_pipeline.teacher.backends.deterministic import DeterministicBackend
from data_pipeline.teacher.generate import generate_targets
from data_pipeline.teacher.structural import StructuralSession


def test_backend_registry_resolves_both_backends():
    assert get_backend_class("deterministic") is DeterministicBackend
    # resolved lazily by name; importing it must not require torch at import time
    assert get_backend_class("sd15_controlnet_depth").NAME == "sd15_controlnet_depth"
    with pytest.raises(KeyError):
        get_backend_class("no_such_backend")


def test_generate_returns_uint8_rgb_of_the_native_shape(mini_session, ci_style):
    session = StructuralSession(mini_session)
    backend = make_backend(ci_style)
    native = session.load_native_rgb(0)
    structure = session.load_structure(0)

    out = backend.generate(native, structure, ci_style)

    assert out.styled_rgb.dtype == np.uint8
    assert out.styled_rgb.shape == native.shape
    assert out.styled_rgb.shape[:2] == (session.resolution[1], session.resolution[0])
    assert out.metadata["backend"] == "deterministic"
    assert out.metadata["research_grade"] is False


def test_structure_is_the_frame_the_teacher_was_asked_for(mini_session, ci_style):
    """Frame-id alignment at the adapter boundary: the structure handed to the
    teacher carries the same session/frame it was loaded for."""
    session = StructuralSession(mini_session)
    for fid in (0, 5, 11):
        structure = session.load_structure(fid)
        assert isinstance(structure, FrameStructure)
        assert structure.frame_id == fid
        assert structure.session_id == session.session_id
        assert structure.entity_id.dtype == np.int32
        assert np.isinf(structure.depth).any()  # +inf background sentinel survives


def test_wrong_shaped_output_is_rejected_by_the_contract(mini_session, ci_style):
    session = StructuralSession(mini_session)
    native = session.load_native_rgb(0)
    with pytest.raises(ValueError):
        DeterministicBackend._check_output(np.zeros((10, 10, 3), np.uint8), native)
    with pytest.raises(TypeError):
        DeterministicBackend._check_output(np.zeros(native.shape, np.float32), native)


def test_deterministic_backend_is_bit_identical_across_runs(mini_session, ci_style, tmp_path):
    """KB Decision 31: the CI teacher is bit-exact, which is what makes the
    cache and alignment tests reproducible. (The diffusion teacher is not, and
    declares ``BIT_EXACT = False``.)"""
    a = generate_targets(mini_session, ci_style, out_root=tmp_path / "a", limit=6)
    b = generate_targets(mini_session, ci_style, out_root=tmp_path / "b", limit=6)
    assert a.generated == b.generated == 6

    for fid in range(6):
        rel = f"frames/frame_{fid:05d}_target.png"
        pa = (tmp_path / "a" / ci_style.style_id / a.session_id / rel).read_bytes()
        pb = (tmp_path / "b" / ci_style.style_id / b.session_id / rel).read_bytes()
        assert pa == pb, f"frame {fid} differs between two runs"

    assert DeterministicBackend.BIT_EXACT is True
    assert get_backend_class("sd15_controlnet_depth").BIT_EXACT is False


def test_teacher_version_changes_with_the_style_config(ci_style):
    v = resolve_teacher_version(ci_style)
    assert v.startswith(DeterministicBackend.BACKEND_VERSION + "/")

    from dataclasses import replace

    edited = replace(ci_style, config_digest="deadbeef" * 8)
    assert resolve_teacher_version(edited) != v


# --------------------------------------------------------------------------- #
# --dry-run
# --------------------------------------------------------------------------- #
def test_dry_run_walks_the_whole_seed_session_without_generating(seed_session, seed_header, ci_style, tmp_path):
    report = generate_targets(seed_session, ci_style, out_root=tmp_path, dry_run=True)

    assert report.ok, report.errors
    assert report.planned == seed_header["frame_count"]
    assert report.generated == 0
    assert report.cache_hits == 0
    assert not (tmp_path / ci_style.style_id).exists(), "a dry run must write nothing"


def test_dry_run_reports_cache_hits_without_regenerating(mini_session, ci_style, tmp_path):
    first = generate_targets(mini_session, ci_style, out_root=tmp_path)
    assert first.generated > 0

    dry = generate_targets(mini_session, ci_style, out_root=tmp_path, dry_run=True)
    assert dry.generated == 0
    assert dry.cache_hits == first.generated


def test_dry_run_flags_a_missing_input_frame(mini_session_copy, ci_style, tmp_path):
    (mini_session_copy / "frames" / "frame_00003_rgb.png").unlink()
    report = generate_targets(mini_session_copy, ci_style, out_root=tmp_path, dry_run=True)
    assert not report.ok
    assert any("frame 3" in e for e in report.errors)
