"""Fixtures for the M2 dataset-validator tests.

The reference recorded session lives under ``data_pipeline/recorder/`` but that
whole tree is gitignored and each recording's dir name carries a per-run
timestamp (``ds_<unix>_<hex>``, KB Decision 27) — so the tests must *discover*
the seed session, never hardcode its name. Resolution order:

  1. ``$STRUCTRENDER_SEED_SESSION`` (an explicit session dir), else
  2. the newest ``ds_*`` dir with a ``manifest.json`` under
     ``$STRUCTRENDER_DATASET_ROOT`` (default ``data_pipeline/recorder/``).

When none is found every test skips with a clear message (mirrors the
``STRUCTRENDER_SKIP_STRUCTURAL`` convention; here the switch is
``STRUCTRENDER_SKIP_VALIDATOR``). All frame counts / session ids the tests
assert against are read back from the discovered session's own ``manifest.json``.

Corrupted-session fixtures are built *programmatically*: copy the real session
into a tmp dir, then mutate one thing. No corrupted fixture is committed.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
RECORDER_ROOT = Path(
    os.environ.get("STRUCTRENDER_DATASET_ROOT", REPO_ROOT / "data_pipeline" / "recorder")
)


def _discover_seed_session() -> Path | None:
    env = os.environ.get("STRUCTRENDER_SEED_SESSION")
    if env:
        p = Path(env)
        return p if (p / "manifest.json").is_file() else None
    if not RECORDER_ROOT.is_dir():
        return None
    cands = sorted(
        d for d in RECORDER_ROOT.glob("ds_*")
        if d.is_dir() and (d / "manifest.json").is_file()
    )
    return cands[-1] if cands else None


REAL_SESSION = _discover_seed_session()


def _mini_frames(session: Path) -> int:
    """A small but structurally complete prefix of the session: the first
    temporal window + the first reappearance sequence (a self-consistent span
    ending on a sequence boundary)."""
    seqs = sorted(
        json.loads((session / "manifest.json").read_text())["sequences"],
        key=lambda s: s["frame_range"][0],
    )
    return seqs[1]["frame_range"][1] + 1


def _skip_reason() -> str | None:
    if os.environ.get("STRUCTRENDER_SKIP_VALIDATOR") == "1":
        return "STRUCTRENDER_SKIP_VALIDATOR=1"
    if REAL_SESSION is None:
        return (
            f"no seed session found under {RECORDER_ROOT} (data_pipeline/recorder/ is "
            "gitignored — run scripts/record_seed_dataset.sh to generate one, or set "
            "$STRUCTRENDER_SEED_SESSION)"
        )
    return None


def pytest_collection_modifyitems(config, items):
    reason = _skip_reason()
    if reason:
        marker = pytest.mark.skip(reason=reason)
        for item in items:
            item.add_marker(marker)


# --------------------------------------------------------------------------- #
# building sessions on disk
# --------------------------------------------------------------------------- #
def _build_mini_session(dst: Path) -> Path:
    """Copy a self-consistent prefix of the real session (ending on a sequence
    boundary — see ``_mini_frames``) into *dst*."""
    n = _mini_frames(REAL_SESSION)
    frames_src = REAL_SESSION / "frames"
    frames_dst = dst / "frames"
    frames_dst.mkdir(parents=True)

    for fid in range(n):
        for suffix in (".npz", "_rgb.png", "_camera.json"):
            name = f"frame_{fid:05d}{suffix}"
            shutil.copy2(frames_src / name, frames_dst / name)

    header = json.loads((REAL_SESSION / "manifest.json").read_text())
    header["frame_count"] = n
    header["sequences"] = [
        s for s in header["sequences"]
        if s["frame_range"][1] <= n - 1
    ]
    (dst / "manifest.json").write_text(json.dumps(header, indent=2))

    lines = (REAL_SESSION / "manifest.jsonl").read_text().splitlines()[:n]
    (dst / "manifest.jsonl").write_text("\n".join(lines) + "\n")

    events = [
        l for l in (REAL_SESSION / "events.jsonl").read_text().splitlines()
        if l.strip() and json.loads(l)["frame_id"] < n
    ]
    (dst / "events.jsonl").write_text("\n".join(events) + "\n")
    return dst


@pytest.fixture(scope="session")
def mini_session_template(tmp_path_factory) -> Path:
    reason = _skip_reason()
    if reason:
        pytest.skip(reason)
    return _build_mini_session(tmp_path_factory.mktemp("mini_template"))


@pytest.fixture
def session(mini_session_template, tmp_path) -> Path:
    """A fresh writable copy of the mini session for one test to mutate.

    The dir is named after the session id — the validator checks
    ``session_id == directory name``."""
    dst = tmp_path / REAL_SESSION.name
    shutil.copytree(mini_session_template, dst)
    return dst


@pytest.fixture
def real_session_copy(tmp_path) -> Path:
    """A full copy of the whole seed session (for the replay / marker test)."""
    reason = _skip_reason()
    if reason:
        pytest.skip(reason)
    dst = tmp_path / REAL_SESSION.name
    shutil.copytree(REAL_SESSION, dst)
    marker = dst / "VALIDATED"
    if marker.exists():
        marker.unlink()
    return dst


# --------------------------------------------------------------------------- #
# mutation helpers
# --------------------------------------------------------------------------- #
def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def patch_frame_line(session: Path, frame_id: int, mutate) -> None:
    rows = read_jsonl(session / "manifest.jsonl")
    mutate(rows[frame_id])
    write_jsonl(session / "manifest.jsonl", rows)


def patch_header(session: Path, mutate) -> None:
    p = session / "manifest.json"
    h = json.loads(p.read_text())
    mutate(h)
    p.write_text(json.dumps(h, indent=2))


def npz_path(session: Path, frame_id: int) -> Path:
    return session / "frames" / f"frame_{frame_id:05d}.npz"


def load_npz(session: Path, frame_id: int) -> dict[str, np.ndarray]:
    with np.load(npz_path(session, frame_id)) as z:
        return {k: z[k] for k in z.files}


def rewrite_npz(session: Path, frame_id: int, members: dict[str, np.ndarray]) -> None:
    np.savez_compressed(npz_path(session, frame_id), **members)


def error_codes(report) -> set[str]:
    return {f.code for f in report.errors}
