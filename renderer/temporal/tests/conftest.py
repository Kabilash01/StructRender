"""Shared setup for the M6 temporal suite.

Most of this suite is pure-tensor and runs in full on a clean checkout. The
data-backed checks — the ones that verify KB Decision 25's convention against
frames the engine actually produced — need the gitignored structural session and
SKIP with a reason when it is absent (same convention as M2/M3/M4).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from renderer.training.dataset import discover_session  # noqa: E402
from renderer.training.train_baseline import ROOT  # noqa: E402


@pytest.fixture(scope="session")
def session():
    from data_pipeline.teacher.structural import StructuralSession

    try:
        # PINNED, not discovered (KB Decision 65). `discover_session()` returns the
        # NEWEST session by mtime; recording the 15-session set silently repointed
        # this fixture, and the warp thresholds here were calibrated on the
        # ORIGINAL session's camera path. A test that quietly changes which data
        # it measures is not a regression test.
        pinned = Path(ROOT) / "data_pipeline" / "recorder" / "ds_1788687350_337657ea"
        path = pinned if pinned.is_dir() else discover_session()
    except FileNotFoundError as e:
        pytest.skip(f"no recorded structural session: {e}")
    s = StructuralSession(path)
    if not s.validated:
        pytest.skip(f"{s.session_id} has no VALIDATED marker")
    return s
