"""Shared fixtures for the M4 baseline suite.

The structural session and the teacher targets are both gitignored, so on a
clean checkout every data-backed test SKIPS with a reason — never a false pass
and never a failure (same convention as the M2 validator / M3 teacher suites).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from renderer.training.dataset import discover_session, target_dir_for  # noqa: E402
from renderer.training.train_baseline import load_config  # noqa: E402

REFRESH_ENV = "STRUCTRENDER_REFRESH_REPORTS"


@pytest.fixture(scope="session")
def cfg() -> dict:
    return load_config()


@pytest.fixture(scope="session")
def report_sink(tmp_path_factory):
    """Where a test is allowed to write its report JSON.

    The reports under ``eval/reports/`` are committed **evidence** — KB Decision
    35 quotes their numbers. A test that overwrites one destroys the exact run
    the KB cites, so the default sink is a tmp path and the committed artifact is
    only refreshed on explicit opt-in::

        STRUCTRENDER_REFRESH_REPORTS=1 pytest renderer/training/tests/

    Deliberately opt-*in*: a plain verification run must be read-only with
    respect to committed artifacts.
    """
    refresh = os.environ.get(REFRESH_ENV) == "1"
    tmp = tmp_path_factory.mktemp("reports")

    def sink(name: str) -> Path:
        return (ROOT / "eval" / "reports" / name) if refresh else (tmp / name)

    return sink


@pytest.fixture(scope="session")
def session(cfg):
    from data_pipeline.teacher.structural import StructuralSession

    try:
        path = cfg["data"].get("session_dir") or discover_session()
    except FileNotFoundError as e:
        pytest.skip(f"no recorded structural session: {e}")
    s = StructuralSession(path)
    if not s.validated:
        pytest.skip(f"{s.session_id} has no VALIDATED marker")
    return s


@pytest.fixture(scope="session")
def target_dir(cfg, session) -> Path:
    d = target_dir_for(session.session_id, cfg["data"]["style_id"], cfg["data"].get("target_root"))
    if not (d / "manifest.jsonl").is_file():
        pytest.skip(f"no '{cfg['data']['style_id']}' teacher targets at {d}")
    if not (d / "ALIGNED").is_file():
        pytest.skip(f"{d} has no ALIGNED marker")
    return d


@pytest.fixture(scope="session")
def device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
