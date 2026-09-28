"""The test suites' subject is pinned, never inferred from directory order."""

from pathlib import Path

from data_pipeline.seed_session import RECORDER_ROOT, SEED_SESSION_ID, seed_session_path

ROOT = Path(__file__).resolve().parents[2]


def test_pinned_id_resolves_without_an_override(monkeypatch):
    monkeypatch.delenv("STRUCTRENDER_SEED_SESSION", raising=False)
    p = seed_session_path()
    if p is not None:                       # absent on a fresh clone: suites then SKIP
        assert p == RECORDER_ROOT / SEED_SESSION_ID


def test_newer_sessions_do_not_change_the_subject(monkeypatch, tmp_path):
    """The failure this replaces: recording a newer session moved the fixtures."""
    monkeypatch.delenv("STRUCTRENDER_SEED_SESSION", raising=False)
    fake_newer = RECORDER_ROOT / "ds_9999999999_ffffffff"      # sorts after every real id
    assert not fake_newer.exists()
    fake_newer.mkdir()
    try:
        (fake_newer / "manifest.json").write_text("{}")
        p = seed_session_path()
        assert p is None or p.name == SEED_SESSION_ID
    finally:
        (fake_newer / "manifest.json").unlink()
        fake_newer.rmdir()


def test_override_is_explicit_and_missing_means_none(monkeypatch, tmp_path):
    monkeypatch.setenv("STRUCTRENDER_SEED_SESSION", str(tmp_path / "nope"))
    assert seed_session_path() is None     # no silent fallback to another session


def test_no_suite_picks_the_newest_session():
    for rel in ("data_pipeline/validation/tests/conftest.py", "data_pipeline/teacher/tests/conftest.py"):
        src = (ROOT / rel).read_text(encoding="utf-8")
        assert "seed_session_path" in src, rel
        assert "cands[-1]" not in src and 'glob("ds_*")' not in src, f"{rel} still infers its subject"
