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


@pytest.fixture(scope="session")
def session():
    from data_pipeline.teacher.structural import StructuralSession

    try:
        path = discover_session()
    except FileNotFoundError as e:
        pytest.skip(f"no recorded structural session: {e}")
    s = StructuralSession(path)
    if not s.validated:
        pytest.skip(f"{s.session_id} has no VALIDATED marker")
    return s
