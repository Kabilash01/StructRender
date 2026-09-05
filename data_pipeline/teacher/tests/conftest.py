"""Fixtures for the M3 teacher-target tests.

Same discovery discipline as the M2 validator suite
(``data_pipeline/validation/tests/conftest.py``): the seed structural session
lives under the gitignored ``data_pipeline/recorder/`` and its directory name
carries a per-recording timestamp, so it is **discovered**, never hardcoded, and
every asserted count is read back from its own ``manifest.json``.

Everything here uses the dependency-free ``deterministic`` teacher (KB Decision
31) so the suite runs on a clean checkout with no torch/diffusers and no model
download. Misaligned target sets are built **programmatically** from a real
generated set — no corrupted fixture is committed.

Force-skip with ``STRUCTRENDER_SKIP_TEACHER=1``.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from data_pipeline.teacher.generate import generate_targets
from data_pipeline.teacher.styles import load_style

REPO_ROOT = Path(__file__).resolve().parents[3]
RECORDER_ROOT = Path(
    os.environ.get("STRUCTRENDER_DATASET_ROOT", REPO_ROOT / "data_pipeline" / "recorder")
)
CONFIG_DIR = REPO_ROOT / "configs" / "teacher"

#: How many frames the shared generated target set covers. Small enough that the
#: whole suite stays seconds-fast, large enough to contain visible entities and
#: several sequences.
MINI_FRAMES = 24
CI_STYLE = "ci_posterize"


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


def _skip_reason() -> str | None:
    if os.environ.get("STRUCTRENDER_SKIP_TEACHER") == "1":
        return "STRUCTRENDER_SKIP_TEACHER=1"
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
# the structural session used by every test
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def seed_session() -> Path:
    reason = _skip_reason()
    if reason:
        pytest.skip(reason)
    return REAL_SESSION


@pytest.fixture(scope="session")
def seed_header(seed_session) -> dict:
    return json.loads((seed_session / "manifest.json").read_text())


@pytest.fixture(scope="session")
def mini_session(tmp_path_factory, seed_session) -> Path:
    """A ``MINI_FRAMES``-frame prefix of the seed session, VALIDATED-marked.

    A prefix is a legitimate structural session for *target* purposes (the
    alignment validator only joins on ``session_id`` + ``frame_id``); the M2
    sequence semantics are the M2 validator's business, not M3's.
    """
    dst = tmp_path_factory.mktemp("mini_struct") / seed_session.name
    (dst / "frames").mkdir(parents=True)
    for fid in range(MINI_FRAMES):
        for suffix in (".npz", "_rgb.png", "_camera.json"):
            name = f"frame_{fid:05d}{suffix}"
            shutil.copy2(seed_session / "frames" / name, dst / "frames" / name)

    header = json.loads((seed_session / "manifest.json").read_text())
    header["frame_count"] = MINI_FRAMES
    header["sequences"] = [
        {"sequence_id": "temporal_0", "sequence_kind": "temporal",
         "frame_range": [0, MINI_FRAMES - 1]}
    ]
    (dst / "manifest.json").write_text(json.dumps(header, indent=2))

    lines = (seed_session / "manifest.jsonl").read_text().splitlines()[:MINI_FRAMES]
    (dst / "manifest.jsonl").write_text("\n".join(lines) + "\n")

    events = [
        l for l in (seed_session / "events.jsonl").read_text().splitlines()
        if l.strip() and json.loads(l)["frame_id"] < MINI_FRAMES
    ]
    (dst / "events.jsonl").write_text("\n".join(events) + "\n")

    # generation refuses an unvalidated session; the prefix is derived from a
    # session the M2 validator passed, so re-stamp the marker here.
    (dst / "VALIDATED").write_text(json.dumps({"result": "PASS", "note": "test fixture"}) + "\n")
    return dst


@pytest.fixture
def mini_session_copy(mini_session, tmp_path) -> Path:
    """A fresh writable copy for a test that needs to break its inputs.

    ``mini_session`` itself is session-scoped and must stay pristine.
    """
    dst = tmp_path / mini_session.name
    shutil.copytree(mini_session, dst)
    return dst


@pytest.fixture(scope="session")
def ci_style():
    return load_style(CI_STYLE, CONFIG_DIR)


# --------------------------------------------------------------------------- #
# generated target sets
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def generated_targets(tmp_path_factory, mini_session, ci_style) -> tuple[Path, Path]:
    """``(target_dir, out_root)`` for a complete deterministic target set over
    the mini structural session. Generated once for the whole suite."""
    out_root = tmp_path_factory.mktemp("targets")
    report = generate_targets(mini_session, ci_style, out_root=out_root)
    assert report.ok, report.errors
    assert report.complete
    return Path(report.target_dir), Path(out_root)


@pytest.fixture
def targets(generated_targets, tmp_path) -> Path:
    """A fresh writable copy of the target set for one test to corrupt.

    The copy keeps ``<style_id>/<session_id>/`` so the validator's
    directory-vs-manifest checks stay meaningful.
    """
    src, _root = generated_targets
    dst = tmp_path / src.parent.name / src.name
    dst.parent.mkdir(parents=True)
    shutil.copytree(src, dst)
    marker = dst / "ALIGNED"
    if marker.exists():
        marker.unlink()
    return dst


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def target_png(targets_dir: Path, frame_id: int) -> Path:
    return targets_dir / "frames" / f"frame_{frame_id:05d}_target.png"


def load_target(targets_dir: Path, frame_id: int) -> np.ndarray:
    with Image.open(target_png(targets_dir, frame_id)) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def save_target(targets_dir: Path, frame_id: int, rgb: np.ndarray) -> None:
    Image.fromarray(rgb, mode="RGB").save(target_png(targets_dir, frame_id))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n")


def patch_header(targets_dir: Path, mutate) -> None:
    p = targets_dir / "manifest.json"
    h = json.loads(p.read_text())
    mutate(h)
    p.write_text(json.dumps(h, indent=2, sort_keys=True))


def error_codes(report) -> set[str]:
    return {f.code for f in report.errors}
